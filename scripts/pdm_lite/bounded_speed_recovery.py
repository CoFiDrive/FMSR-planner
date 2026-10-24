"""Two bounded speed-row experiments reusing the paired FM diagnostic caches."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.pdm_lite.bounded_fm_recovery import FLOW, config, make_head, evaluate, write_json
import jsonpickle
import jsonpickle.ext.numpy as jnp
import numpy as np
import torch


def speed_rows(head):
    """Freeze every parameter except masked continuous-speed output rows."""
    if not head.config.diffusion_speed:
        raise ValueError('A continuous speed token is required')
    head.requires_grad_(False)
    parameters, allowed, effective = [], {}, 0
    for index, layer in enumerate(head.diff_decoder.layers):
        final = layer.task_decoder.plan_reg_branch[-1]
        row = head._num_poses * 3
        if final.out_features != (head._num_poses+1)*3:
            raise ValueError('Unexpected xyz refinement projection layout')
        final.requires_grad_(True)
        weight_mask = torch.zeros_like(final.weight)
        weight_mask[row] = 1
        bias_mask = torch.zeros_like(final.bias)
        bias_mask[row] = 1
        final.weight.register_hook(lambda grad, mask=weight_mask: grad*mask)
        final.bias.register_hook(lambda grad, mask=bias_mask: grad*mask)
        parameters.extend(final.parameters())
        stem = f'diff_decoder.layers.{index}.task_decoder.plan_reg_branch.4.'
        allowed[stem+'weight'] = weight_mask.cpu()
        allowed[stem+'bias'] = bias_mask.cpu()
        effective += final.in_features+1
    return parameters, allowed, effective


def recovery_loss(head, query, bev, truth, scenes=4):
    """Targets enter only the speed loss after a detached deployment prefix."""
    count = min(scenes, len(query))
    selected = torch.randperm(len(query), device=query.device)[:count]
    q, b = query[selected].float(), bev[selected].float()
    anchors = head.plan_anchor.float()[None].expand(count, -1, -1, -1)
    with torch.no_grad(), head._deterministic_cr_decoder():
        mode = head._classify_flow_anchors(anchors, b, b.shape[-2:], q)[-1].argmax(-1)
        chosen = head.gather_mode(anchors, mode)
        # Stratify queries across early, middle and late deployment prefixes.
        indices = torch.randint(head.config.flow_inference_steps, (count,), device=q.device)
        state, time = head._flow_rollout_query(chosen, indices, b, b.shape[-2:], q)
    with head._deterministic_cr_decoder(), torch.autocast(device_type=q.device.type, enabled=False):
        endpoint = head.predict_flow_endpoint(state[:, None], time, chosen[:, None], b, b.shape[-2:], q)
        speed = endpoint[:, 0, -1, 0]
        # One m/s loss scale; no output clamp that would erase low-speed gradients.
        loss = torch.nn.functional.smooth_l1_loss(speed, truth[selected, -1, 0].float())
    return loss


def integrity(head, initial, allowed):
    current = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    changes = []
    for key, value in current.items():
        if key in allowed:
            mask = allowed[key].bool()
            if not torch.equal(value[~mask], initial[key][~mask]):
                raise ValueError(f'Non-speed projection entries changed: {key}')
        elif not torch.equal(value, initial[key]):
            raise ValueError(f'Frozen head parameter changed: {key}')
        if not torch.equal(value, initial[key]):
            changes.append(key)
    return current, changes


def run(args, device):
    output = args.output/args.variant
    if output.exists():
        raise ValueError('Choose a fresh variant output; do not overwrite prior experiments')
    output.mkdir(parents=True)
    train = torch.load(args.cache_root/'flow_training.pt', weights_only=True)
    val = torch.load(args.cache_root/'flow_validation.pt', weights_only=True)
    if {str(Path(p).parent.parent) for p in train['frames']} & {str(Path(p).parent.parent) for p in val['frames']}:
        raise ValueError('Training/development route overlap')
    cfg = config(FLOW.parent/'config.json')
    cfg.flow_contract_delay_steps = 0
    cfg.flow_contract_ramp_steps = 100
    cfg.flow_contract_warmup_steps = 0
    cfg.flow_rollout_recovery_weight = 0.
    source = torch.load(FLOW, map_location='cpu', weights_only=True)
    head = make_head(cfg, source, device)
    parameters, allowed, effective = speed_rows(head)
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
    protocol = {'variant': args.variant, 'steps': args.steps, 'lr': args.lr,
        'batch_size': 8, 'weight_decay': 0., 'effective_trainable_scalars': effective,
        'source': str(FLOW), 'source_sha256': hashlib.sha256(FLOW.read_bytes()).hexdigest(),
        'cache_root': str(args.cache_root.resolve()), 'training_frames': len(train['target']),
        'validation_frames': len(val['target']), 'trainable': 'speed x row and bias in both final projections only',
        'main_loss': 'original FM+spectral CR; common delay=0 and ramp=100',
        'speed_recovery_weight': .1 if args.variant=='speed_rollout' else 0.,
        'speed_recovery_scenes': 4, 'recovery_interval': 1,
        'recovery_loss': 'endpoint speed Huber in m/s at detached deployment rollout states',
        'precision': 'original BF16 main objective; FP32 recovery and deployment validation',
        'development_scope': 'cached fixed context; FM pretraining already used these validation routes'}
    write_json(output/'protocol.json', protocol)
    rng = np.random.default_rng(4137)
    batches = [rng.choice(len(train['target']), 8, replace=False) for _ in range(args.steps)]
    with (output/'training.jsonl').open('w') as log:
        for step, indices in enumerate(batches, 1):
            torch.manual_seed(10000+step)
            head.train(); head.diff_decoder_cls.eval()
            q, b, truth = [train[key][indices].to(device) for key in ('query', 'bev', 'target')]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                result = head.forward_train_flow(q, b, b.shape[-2:], gt_checkpoints=truth)
                original_loss = result['flow_loss']
            recovered = recovery_loss(head, q, b, truth) if args.variant=='speed_rollout' else original_loss.new_zeros(())
            loss = original_loss + protocol['speed_recovery_weight']*recovered
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite speed training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite speed gradients')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            optimizer.step()
            row = {'step': step, 'loss': float(loss.detach()), 'main_loss': float(original_loss.detach()),
                'speed_recovery_loss': float(recovered.detach()), 'grad_norm': norm,
                'flow_tau_mean': float(result['flow_tau_mean']),
                'flow_contract_active': float(result['flow_contract_active']),
                'flow_contract_kappa_mean': float(result.get('flow_contract_kappa_mean', 0.))}
            log.write(json.dumps(row)+'\n'); log.flush()
            if step % 25 == 0:
                print(args.variant, step, '/', args.steps, 'loss', row['loss'], flush=True)
            if step in (250, 500) or step == args.steps:
                report = evaluate(head, val, device, 4)
                write_json(output/f'generation_step{step:06d}.json', report)
                current, changes = integrity(head, initial, allowed)
                full = dict(source)
                full.update({'fmsr_head.'+k: v for k, v in current.items()})
                torch.save(full, output/f'model_step{step:06d}.pth')
                torch.save({'head': current, 'optimizer': optimizer.state_dict(), 'step': step,
                    'protocol': protocol}, output/f'trainer_step{step:06d}.pt')
                write_json(output/f'integrity_step{step:06d}.json', {'changed_tensors': changes,
                    'all_non_speed_head_entries_unchanged': True,
                    'all_non_head_weights_reused_unchanged': True,
                    'effective_trainable_scalars': effective})
                jnp.register_handlers()
                (output/'config.json').write_text(jsonpickle.encode(cfg, indent=2)+'\n')
                print(args.variant, 'validation', step, json.dumps(report['mean']), flush=True)


@torch.no_grad()
def diagnose(args, device):
    """Compare label-conditioned diagnostic queries with untouched real rollouts."""
    cache = torch.load(args.cache_root/'flow_validation.pt', weights_only=True)
    report = {}
    for name, checkpoint in [('initial', FLOW), ('speed_rollout500',
            args.output/'speed_rollout/model_step000500.pth')]:
        cfg = config(checkpoint.parent/'config.json')
        head = make_head(cfg, torch.load(checkpoint, map_location='cpu', weights_only=True), device).eval()
        totals = []
        for start in range(0, len(cache['target']), 4):
            q, b, truth = [cache[key][start:start+4].to(device) for key in ('query', 'bev', 'target')]
            anchors = head.plan_anchor.float()[None].expand(len(q), -1, -1, -1)
            predicted = head._classify_flow_anchors(anchors, b, b.shape[-2:], q)[-1].argmax(-1)
            oracle = head.nearest_anchor_idx(truth, anchors)
            state = anchors.clone()
            grid = head.get_flow_schedule(device=device)
            for i, (time, next_time) in enumerate(zip(grid[:-1], grid[1:])):
                free = head.predict_flow_endpoint(state, time.expand(len(q)), anchors, b, b.shape[-2:], q)
                # This separate, explicitly labelled branch measures training fit.
                # Its result never enters the real rollout state update.
                bridge = (1.-time)*anchors+time*truth[:, None]
                teacher = head.predict_flow_endpoint(bridge, time.expand(len(q)), anchors, b, b.shape[-2:], q)
                if len(totals) <= i:
                    totals.append({'tau': float(time), 'n': 0, 'rollout_pred_mode_speed_MAE': 0.,
                        'rollout_oracle_mode_speed_MAE': 0., 'GT_bridge_pred_mode_speed_MAE': 0.,
                        'GT_bridge_oracle_mode_speed_MAE': 0.})
                row = totals[i]; row['n'] += len(q)
                for key, output, mode in [('rollout_pred_mode_speed_MAE', free, predicted),
                        ('rollout_oracle_mode_speed_MAE', free, oracle),
                        ('GT_bridge_pred_mode_speed_MAE', teacher, predicted),
                        ('GT_bridge_oracle_mode_speed_MAE', teacher, oracle)]:
                    value = head.gather_mode(output, mode)[:, -1, 0].clamp(0, 30)
                    row[key] += float((value-truth[:, -1, 0]).abs().sum())
                velocity = (free.float()-state.float())/(1.-time).clamp_min(1e-6)
                state = head._mask_flow_speed_padding(state.float()+(next_time-time)*velocity)
        for row in totals:
            for key in list(row):
                if key.endswith('_MAE'):
                    row[key] /= row['n']
        report[name] = totals
        del head
    report['scope'] = '128 development frames; GT bridge is a separate label-conditioned diagnostic; real rollout receives no GT'
    write_json(args.output/'speed_state_diagnostic.json', report)
    print(json.dumps({k: v[-1] for k, v in report.items() if isinstance(v, list)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, default=ROOT/'outputs/analysis/bounded_fm_recovery_20261007')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/analysis/bounded_speed_recovery_20261007')
    parser.add_argument('--variant', choices=['speed_bridge', 'speed_rollout'], default='speed_rollout')
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 500 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('This speed experiment is limited to 1--500 updates per variant')
    random.seed(137); np.random.seed(137); torch.manual_seed(137); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if args.diagnose:
        diagnose(args, torch.device('cuda'))
    else:
        run(args, torch.device('cuda'))


if __name__ == '__main__':
    main()
