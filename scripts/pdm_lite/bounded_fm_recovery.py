"""Cached, bounded epoch09 FM/CR diagnostic and paired recovery experiment."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fmsr_pdm_lite.fmsrpdm import bootstrap
import jsonpickle
import jsonpickle.ext.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import GlobalConfig
from data import CARLA_Data
from fmsr_pdm_lite.fmsrpdm.model import FMSRModel
from fmsr_pdm_lite.fmsrpdm.generation_validation import summarize
from fmsr_pdm_lite.fmsrpdm.clean_prior import prior_kwargs
from fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_fd import TrajectoryHead

FLOW = ROOT/'outputs/local_training/tie_spectral_10epochs_seed0_20261006_223036/model_0009.pth'
CLEAN = ROOT/'fmsr_pdm_lite/runs/clean_prior_10epochs_20261006_015234/model_0009.pth'
ASSETS = ROOT/'fmsr_pdm_lite/assets/paper_k60_seed0'
DATA = ROOT/'outputs/local_dataset_mounts/carla_garage_data'


def native_flow_model(cfg):
    """Use the production FM wrapper, including its continuous 30 m/s bound.

    Keep the independent clean-prior snapshot isolated in the same process.
    """
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    folder = ROOT/'3rd_party/carla_garage/team_code'
    interface = load('bounded_fm_interface', folder/'fmsr_interface.py')
    with patch.dict(sys.modules, {'fmsr_interface': interface}):
        module = load('bounded_fm_native_model', folder/'model.py')
    return module.LidarCenterNet(cfg)


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    temp.replace(path)


def config(path):
    raw = json.loads(path.read_text())
    result = GlobalConfig()
    result.__dict__.update({k: v for k, v in raw.items() if not k.startswith('py/')})
    for key in ('fmsr_anchor_path', 'fmsr_stats_path', 'trajectory_stats_path'):
        if getattr(result, key, None):
            value = Path(getattr(result, key))
            if not value.exists():
                value = Path(str(value).replace('/T7/', '/T71/'))
            if not value.exists():
                raise FileNotFoundError(value)
            setattr(result, key, str(value))
    return result


def manifest_sample(lines, n, seed):
    groups = {}
    for line in lines:
        groups.setdefault(line.split('/')[0], []).append(line)
    rng = np.random.default_rng(seed)
    chosen = [str(rng.choice(groups[key])) for key in sorted(groups)]
    pool = sorted(set(lines)-set(chosen))
    chosen += rng.choice(pool, min(n-len(chosen), len(pool)), replace=False).tolist()
    return sorted(chosen)


def loader(cfg, manifest, batch_size):
    cfg = copy.deepcopy(cfg)
    cfg.fmsr_frame_manifest = str(manifest)
    cfg.augment = cfg.use_color_aug = cfg.lidar_aug_prob = 0
    cfg.detect_boxes = cfg.estimate_class_distributions = 0
    # Present only selected routes to the native loader, preserving its quality
    # checks while avoiding a full external-disk scan for a bounded experiment.
    lines = manifest.read_text().splitlines()
    view = manifest.parent/'data_views'/manifest.stem
    for line in lines:
        scenario, route, _ = line.split('/')
        folder = view/scenario
        folder.mkdir(parents=True, exist_ok=True)
        link = folder/route
        if not link.is_symlink():
            link.symlink_to(DATA/scenario/route, target_is_directory=True)
    scenarios = sorted({line.split('/')[0] for line in lines})
    dataset = CARLA_Data([str(view/s) for s in scenarios], cfg, validation=False)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(137))


def inputs(batch, device):
    return dict(rgb=batch['rgb'].to(device, dtype=torch.float32),
        lidar_bev=batch['lidar'].to(device, dtype=torch.float32),
        target_point=batch['target_point'].to(device, dtype=torch.float32),
        ego_vel=batch['speed'].to(device, dtype=torch.float32).reshape(-1, 1),
        command=batch['command'].to(device, dtype=torch.float32),
        target_point_next=batch['target_point_next'].to(device, dtype=torch.float32))


def cache(model, cfg, manifest, batch_size, device, clean=False):
    dataset, batches = loader(cfg, manifest, batch_size)
    q, b, y, priors = [], [], [], []
    verification = None
    for index, batch in enumerate(batches):
        args = inputs(batch, device)
        with torch.no_grad():
            if clean:
                query, bev, _, prior = model.context(**args, return_clean_prior=True)
                priors.append(prior.cpu())
            else:
                captured = {}
                def capture(ego_query, bev_feature, bev_spatial_shape, global_img=None):
                    captured.update(query=ego_query.detach().clone(), bev=bev_feature.detach().clone())
                    return ego_query.new_zeros(len(ego_query), 11, 2)
                with patch.object(model.fmsr_head, 'forward_test_flow', side_effect=capture):
                    model(**args)
                query, bev = captured['query'], captured['bev']
            if index == 0:
                direct = model(**args)
                full = torch.cat([direct[2], torch.cat([direct[1], torch.zeros_like(direct[1])], -1)[:, None]], 1)
                if clean:
                    generator = torch.Generator(device=device).manual_seed(137)
                    # Clean sampler is stochastic, so equality uses a separate seeded call.
                    torch.manual_seed(137)
                    full_result = model(**args, generator=torch.Generator(device=device).manual_seed(137))
                    full = torch.cat([full_result[2], torch.cat([full_result[1], torch.zeros_like(full_result[1])], -1)[:, None]], 1)
                    cached = model.planner.sample(query, bev, generator=generator, **prior_kwargs(prior))[0]
                    cached[:, -1, 0].clamp_(0, 30)
                else:
                    cached = model.fmsr_head.forward_test_flow(query, bev, bev.shape[-2:])
                    cached[:, -1, 0].clamp_(0, 30)
                verification = float((full-cached).abs().max())
                if verification > 1e-5:
                    raise ValueError(f'Cached/full inference disagree: {verification}')
        q.append(query.cpu()); b.append(bev.cpu())
        y.append(torch.cat([batch['route'][:, :10].float(), torch.stack([
            batch['target_speed_value'].float().reshape(-1), torch.zeros(len(query))], -1)[:, None]], 1))
        if index % 20 == 0:
            print('cache', manifest.name, index, '/', len(batches), flush=True)
    frames = [str(Path(paths[0].decode()).resolve()/f'{dataset.sample_start[i]:04d}.json.gz')
              for i, paths in enumerate(dataset.measurements)]
    result = {'query': torch.cat(q).clone(), 'bev': torch.cat(b).clone(), 'target': torch.cat(y).clone(),
        'frames': frames, 'full_vs_cached_max_abs': verification}
    if clean:
        result['prior'] = torch.cat(priors).clone()
    return result


@torch.no_grad()
def evaluate(head, cached, device, batch_size, clean=False):
    head.eval()
    predictions, timeline, modes, oracle_predictions, correct_modes = [], [], [], [], []
    seed_reports = []
    for seed in ((137, 271, 811) if clean else (137,)):
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = []
        for start in range(0, len(cached['target']), batch_size):
            query = cached['query'][start:start+batch_size].to(device)
            bev = cached['bev'][start:start+batch_size].to(device)
            truth = cached['target'][start:start+batch_size]
            if clean:
                result, mode, trace = head.sample(query, bev, generator=generator,
                    clean_prior=cached['prior'][start:start+batch_size].to(device), return_trace=True)
                if seed == 137:
                    for j, row in enumerate(trace):
                        if len(timeline) <= j:
                            timeline.append({'time': row['t'], 'predictions': []})
                        timeline[j]['predictions'].append(row['clean_prediction'].cpu())
            else:
                anchors = head.plan_anchor.float()[None].expand(len(query), -1, -1, -1)
                mode = head._classify_flow_anchors(anchors, bev, bev.shape[-2:], query)[-1].argmax(-1)
                # Integrate every mode exactly as deployment. Algebraically
                # independent single-mode calls can have different TF32 rounding.
                state = anchors.clone()
                schedule = head.get_flow_schedule(device=device)
                for j, (time, next_time) in enumerate(zip(schedule[:-1], schedule[1:])):
                    endpoint = head.predict_flow_endpoint(state, time.expand(len(query)), anchors, bev,
                        bev.shape[-2:], query)
                    if len(timeline) <= j:
                        timeline.append({'time': float(time), 'predictions': []})
                    timeline[j]['predictions'].append(head.gather_mode(endpoint, mode)[:, :10].cpu())
                    velocity = (endpoint.float()-state.float())/(1.-time).clamp_min(1e-6)
                    state = head._mask_flow_speed_padding(state.float()+(next_time-time)*velocity)
                result = head.gather_mode(state, mode)
                oracle_mode = head.nearest_anchor_idx(truth.to(device), anchors)
                oracle = head.gather_mode(state, oracle_mode).clone()
                oracle[:, -1, 0].clamp_(0, 30)
                oracle_predictions.append(oracle.cpu())
                correct_modes.extend((mode == oracle_mode).cpu().tolist())
                if start == 0:
                    all_result = head.forward_test_flow(query, bev, bev.shape[-2:])
                    torch.testing.assert_close(result, all_result, atol=1e-5, rtol=1e-5)
            result[:, -1, 0].clamp_(0, 30)
            outputs.append(result.cpu())
            if seed == 137:
                modes.extend(mode.cpu().tolist())
        generated = torch.cat(outputs)
        predictions.append(generated)
        seed_reports.append(summarize(generated, cached['target']))
    mean = {k: sum(r[k] for r in seed_reports)/len(seed_reports) for k in seed_reports[0]}
    times = [{'time': row['time'], 'endpoint_ADE_m': float(torch.linalg.vector_norm(
        torch.cat(row['predictions'])-cached['target'][:, :10], dim=-1).mean())} for row in timeline]
    path_error = torch.linalg.vector_norm(predictions[0][:, :10]-cached['target'][:, :10], dim=-1).mean(-1)
    return {'mean': mean, 'per_seed': seed_reports, 'timeline': times, 'mode': modes,
        'oracle_mode_metrics': summarize(torch.cat(oracle_predictions), cached['target']) if not clean else None,
        'mode_accuracy': sum(correct_modes)/len(correct_modes) if correct_modes else None,
        'predicted_speed_mps': predictions[0][:, -1, 0].tolist(),
        'target_speed_mps': cached['target'][:, -1, 0].tolist(),
        'per_frame_ADE_m': path_error.tolist(), 'frames': cached['frames'],
        'scope': 'development_diagnostic; FM pretraining used all routes; no labels in generation'}


def make_head(cfg, state, device):
    head = TrajectoryHead(10, cfg.tf_d_ffn, cfg.tf_d_model, cfg.fmsr_anchor_path, cfg).to(device)
    weights = {k.removeprefix('fmsr_head.'): v for k, v in state.items() if k.startswith('fmsr_head.')}
    if not torch.equal(head.plan_anchor.cpu(), weights['plan_anchor'].cpu()):
        raise ValueError('Checkpoint anchor differs from saved assets')
    head.load_state_dict(weights, strict=True)
    return head


def prepare(args, device):
    if (args.output/'baseline.json').exists():
        raise ValueError('Preparation already exists; use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    val_lines = (ASSETS/'validation_frames.txt').read_text().splitlines()
    metadata = json.loads((ASSETS/'assets.json').read_text())
    # The manifests are the actual loader split. Metadata group lists can be stale.
    val_routes = {'/'.join(line.split('/')[:2]) for line in val_lines}
    train_lines = Path('/media/vcdc/T71/pdm_lite_assets/pdm_lite_keyframes.txt').read_text().splitlines()
    train_lines = [x for x in train_lines if x and '/'.join(x.split('/')[:2]) not in val_routes]
    # A moderate route pool keeps I/O bounded and avoids repeating only a few frames.
    rng = np.random.default_rng(4137)
    scene_routes = {}
    for line in train_lines:
        scene_routes.setdefault(line.split('/')[0], set()).add('/'.join(line.split('/')[:2]))
    chosen_routes = {str(route) for scene, routes in sorted(scene_routes.items())
                     for route in rng.choice(sorted(routes), min(4, len(routes)), replace=False)}
    train_lines = [line for line in train_lines if '/'.join(line.split('/')[:2]) in chosen_routes]
    for name, lines, n, seed in [('validation', val_lines, 128, 137), ('training', train_lines, 1024, 4137)]:
        (args.output/f'{name}_frames.txt').write_text('\n'.join(manifest_sample(lines, n, seed))+'\n')
    cfg = config(FLOW.parent/'config.json')
    state = torch.load(FLOW, map_location='cpu', weights_only=True)
    model = native_flow_model(cfg).to(device).eval()
    model.load_state_dict(state, strict=True)
    for name in ('validation', 'training'):
        destination = args.output/f'flow_{name}.pt'
        if not destination.exists():
            cached = cache(model, cfg, args.output/f'{name}_frames.txt', args.batch_size, device)
            torch.save(cached, destination)
    flow_val = torch.load(args.output/'flow_validation.pt', weights_only=True)
    baseline = {'flow': evaluate(model.fmsr_head, flow_val, device, args.batch_size)}
    del model, state
    torch.cuda.empty_cache()
    clean_cfg = config(CLEAN.parent/'config.json')
    model = FMSRModel(clean_cfg).to(device).eval()
    model.load_state_dict(torch.load(CLEAN, map_location='cpu', weights_only=True), strict=True)
    clean_destination = args.output/'clean_validation.pt'
    clean_val = (torch.load(clean_destination, weights_only=True) if clean_destination.exists() else
        cache(model, clean_cfg, args.output/'validation_frames.txt', args.batch_size, device, clean=True))
    if clean_val['frames'] != flow_val['frames']:
        raise ValueError('Two implementations loaded different validation frames')
    torch.save(clean_val, args.output/'clean_validation.pt')
    baseline['clean_prior'] = evaluate(model.planner, clean_val, device, args.batch_size, clean=True)
    train = torch.load(args.output/'flow_training.pt', weights_only=True)
    excluded = {str((DATA/route).resolve()) for route in val_routes}
    keep = [i for i, frame in enumerate(train['frames']) if str(Path(frame).parent.parent) not in excluded]
    removed = len(train['frames'])-len(keep)
    if removed:
        for key in ('query', 'bev', 'target'):
            train[key] = train[key][keep].clone()
        train['frames'] = [train['frames'][i] for i in keep]
        train['excluded_validation_route_frames'] = removed
        torch.save(train, args.output/'flow_training.pt')
    effective_manifest = [Path(frame).parent.parent.parent.name+'/'+Path(frame).parent.parent.name+
                          '/'+Path(frame).name.split('.')[0]
                          for frame in train['frames']]
    (args.output/'effective_training_frames.txt').write_text('\n'.join(effective_manifest)+'\n')
    routes = lambda c: {str(Path(p).parent.parent) for p in c['frames']}
    if routes(train) & routes(flow_val):
        raise ValueError('Bounded training and development routes overlap')
    baseline['setup'] = {'train_frames': len(train['target']), 'validation_frames': len(flow_val['target']),
        'train_routes': len(routes(train)), 'validation_routes': len(routes(flow_val)),
        'cached_frozen_context': True, 'augmentation': False,
        'split_defined_by_actual_frame_manifests': True,
        'excluded_validation_route_frames': train.get('excluded_validation_route_frames', 0),
        'flow_checkpoint': str(FLOW), 'clean_checkpoint': str(CLEAN),
        'flow_sha256': hashlib.sha256(FLOW.read_bytes()).hexdigest(),
        'full_vs_cached_max_abs': {'flow': flow_val['full_vs_cached_max_abs'], 'clean': clean_val['full_vs_cached_max_abs']}}
    write_json(args.output/'baseline.json', baseline)
    print(json.dumps({k: v['mean'] for k, v in baseline.items() if k!='setup'}, indent=2), flush=True)


def train(args, device):
    root = args.output/args.variant
    root.mkdir(exist_ok=True)
    training = torch.load(args.output/'flow_training.pt', weights_only=True)
    validation = torch.load(args.output/'flow_validation.pt', weights_only=True)
    cfg = config(FLOW.parent/'config.json')
    cfg.flow_contract_delay_steps = 0
    cfg.flow_contract_ramp_steps = 100
    cfg.flow_contract_warmup_steps = 0
    cfg.flow_rollout_recovery_weight = .1 if args.variant=='recovery' else 0.
    cfg.flow_rollout_recovery_interval = 4
    cfg.flow_rollout_recovery_scenes = 4
    original = torch.load(FLOW, map_location='cpu', weights_only=True)
    head = make_head(cfg, original, device)
    head.requires_grad_(False)
    parameters = []
    for layer in head.diff_decoder.layers:
        branch = layer.task_decoder.plan_reg_branch
        branch.requires_grad_(True)
        # Freeze final projection rows belonging to the speed token and unused z.
        final = branch[-1]
        mask = torch.zeros_like(final.weight)
        mask[:30] = torch.tensor([1., 1., 0.], device=device).repeat(10)[:, None]
        final.weight.register_hook(lambda grad, mask=mask: grad*mask)
        final.bias.register_hook(lambda grad, mask=mask[:, 0]: grad*mask)
        parameters.extend(branch.parameters())
    # No decay: masked/frozen speed projection rows must remain bitwise unchanged.
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=0.)
    initial = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    start = 0
    resume = root/'trainer.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['variant'] != args.variant:
            raise ValueError('Resume variant mismatch')
        head.load_state_dict(saved['head'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        start = saved['step']
        head._flow_contract_calls = start
    if args.steps <= start:
        raise ValueError('Requested steps must exceed completed steps')
    write_json(root/'experiment.json', {'variant': args.variant, 'max_steps': args.steps,
        'lr': 1e-5, 'weight_decay': 0., 'batch_size': args.train_batch_size,
        'recovery_weight': cfg.flow_rollout_recovery_weight, 'CR_kappa': cfg.flow_contract_tau_kappa,
        'CR_delay': 0, 'CR_ramp': 100, 'trainable': 'two regression MLPs; speed output projection rows frozen',
        'starting_checkpoint': str(FLOW), 'reused_contexts': True})
    random_order = np.random.default_rng(4137)
    indices = [random_order.choice(len(training['target']), args.train_batch_size, replace=False)
               for _ in range(args.steps)]
    with (root/'training.jsonl').open('a') as log:
        for step in range(start+1, args.steps+1):
            # Every shared FM/CR forward sees the same RNG in A and B.
            torch.manual_seed(10000+step)
            head.train(); head.diff_decoder_cls.eval()
            idx = indices[step-1]
            query, bev, target = [training[k][idx].to(device) for k in ('query', 'bev', 'target')]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
                result = head.forward_train_flow(query, bev, bev.shape[-2:], gt_checkpoints=target)
                loss = result['flow_loss']
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError('Nonfinite training gradient')
            grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
            optimizer.step()
            record = {'step': step, 'loss': float(loss.detach()), 'grad_norm': grad_norm,
                **{k: float(v) for k, v in result.items() if k.startswith('flow_') and torch.is_tensor(v) and v.numel()==1}}
            log.write(json.dumps(record)+'\n'); log.flush()
            if step % 25 == 0:
                print(args.variant, step, '/', args.steps, 'loss', record['loss'], flush=True)
            if step in (500, 1500) or step == args.steps:
                report = evaluate(head, validation, device, args.batch_size)
                write_json(root/f'generation_step{step:06d}.json', report)
                current = {k: v.detach().cpu() for k, v in head.state_dict().items()}
                changed = [k for k in current if not torch.equal(initial[k], current[k])]
                if any('task_decoder.plan_reg_branch' not in k for k in changed):
                    raise ValueError('A frozen parameter changed')
                for i in range(2):
                    for suffix in ('weight', 'bias'):
                        key = f'diff_decoder.layers.{i}.task_decoder.plan_reg_branch.4.{suffix}'
                        torch.testing.assert_close(current[key][30:], initial[key][30:], atol=0, rtol=0)
                full = dict(original)
                full.update({'fmsr_head.'+k: v for k, v in current.items()})
                torch.save(full, root/f'model_step{step:06d}.pth')
                state = {'head': current, 'optimizer': optimizer.state_dict(), 'step': step, 'variant': args.variant}
                torch.save(state, root/'trainer.tmp.pt'); (root/'trainer.tmp.pt').replace(resume)
                jnp.register_handlers()
                (root/'config.json').write_text(jsonpickle.encode(cfg, indent=2)+'\n')
                write_json(root/f'integrity_step{step:06d}.json', {'changed_keys': changed,
                    'all_frozen_head_parameters_unchanged': True, 'speed_output_rows_unchanged': True,
                    'non_head_full_checkpoint_reused_unchanged': True, 'step': step})
                print(args.variant, 'validation', step, json.dumps(report['mean']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'train', 'evaluate'], required=True)
    parser.add_argument('--checkpoint', type=Path, default=FLOW)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--variant', choices=['control', 'recovery'], default='control')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--train-batch-size', type=int, default=8)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 1500:
        raise ValueError('This experiment is bounded to 1--1500 updates per variant')
    random.seed(137); np.random.seed(137); torch.manual_seed(137); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; run with host GPU access')
    device = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    if args.phase == 'prepare':
        prepare(args, device)
    elif args.phase == 'train':
        train(args, device)
    else:
        cfg = config(args.checkpoint.parent/'config.json')
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        head = make_head(cfg, state, device)
        cached = torch.load(args.output/'flow_validation.pt', weights_only=True)
        report = evaluate(head, cached, device, args.batch_size)
        if args.report is None:
            raise ValueError('--report required for evaluation phase')
        write_json(args.report, report)
        print(json.dumps({'mean': report['mean'], 'oracle': report['oracle_mode_metrics'],
                          'mode_accuracy': report['mode_accuracy']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
