"""Cache frozen FM outputs and train only a small independent speed corrector."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.pdm_lite.bounded_fm_recovery import FLOW, config, make_head, write_json
from fmsr.tfv6.diffusion_modules.speed_corrector import SpeedCorrector
from fmsr_pdm_lite.fmsrpdm.generation_validation import summarize
import numpy as np
import torch


@torch.no_grad()
def generate(head, cache, device):
    head.eval(); predictions = []
    for start in range(0, len(cache['target']), 4):
        q, b = cache['query'][start:start+4].to(device), cache['bev'][start:start+4].to(device)
        prediction = head.forward_test_flow(q, b, b.shape[-2:])
        prediction[:, -1, 0].clamp_(0, 30)
        predictions.append(prediction.cpu())
        if start % 128 == 0:
            print('cache frozen FM generation', start, '/', len(cache['target']), flush=True)
    return torch.cat(predictions)


@torch.no_grad()
def validate(corrector, cache, generation, device):
    corrector.eval()
    corrected = corrector(cache['query'].to(device), generation.to(device)).cpu()
    if not torch.equal(corrected[:, :10], generation[:, :10]):
        raise ValueError('Corrector changed generated path')
    if not torch.equal(corrected[:, -1, 1], generation[:, -1, 1]):
        raise ValueError('Corrector changed dummy speed coordinate')
    return {'mean': summarize(corrected, cache['target']),
        'target_speed_mps': cache['target'][:, -1, 0].tolist(),
        'predicted_speed_mps': corrected[:, -1, 0].tolist(),
        'frames': cache['frames'], 'generated_path_bitwise_unchanged': True,
        'scope': 'fixed development frames, no GT input in generation/correction'}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'protocol.json').exists():
        raise ValueError('Choose a new output directory; do not overwrite completed experiments')
    device = torch.device('cuda')
    training = torch.load(args.cache_root/'flow_training.pt', weights_only=True)
    validation = torch.load(args.cache_root/'flow_validation.pt', weights_only=True)
    routes = lambda c: {str(Path(x).parent.parent) for x in c['frames']}
    if routes(training) & routes(validation):
        raise ValueError('Training and development route overlap')
    cfg = config(FLOW.parent/'config.json')
    original = torch.load(FLOW, map_location='cpu', weights_only=True)
    head = make_head(cfg, original, device).eval().requires_grad_(False)
    for name, cache in [('training', training), ('validation', validation)]:
        path = args.output/f'generated_{name}.pt'
        if not path.exists():
            torch.save(generate(head, cache, device), path)
    del head, original
    torch.cuda.empty_cache()
    train_gen = torch.load(args.output/'generated_training.pt', weights_only=True)
    val_gen = torch.load(args.output/'generated_validation.pt', weights_only=True)
    previous = json.loads((args.cache_root/'flow_baseline_detailed.json').read_text())
    cache_speed_delta = float(np.abs(val_gen[:, -1, 0].numpy()-
        np.asarray(previous['predicted_speed_mps'])).max())
    # Different full-rollout call sequences can differ at TF32 rounding scale.
    # Compare the new correction to this fresh cache with exact identity below.
    if not np.allclose(val_gen[:, -1, 0].numpy(), previous['predicted_speed_mps'], atol=2e-4, rtol=0):
        raise ValueError('Frozen generation cache disagrees with previous deployment baseline')
    corrector = SpeedCorrector().to(device)
    identity = validate(corrector, validation, val_gen, device)
    torch.testing.assert_close(corrector(validation['query'].to(device), val_gen.to(device)).cpu(),
        val_gen, atol=0, rtol=0)
    write_json(args.output/'generation_step000000.json', identity)
    source_hash = hashlib.sha256(FLOW.read_bytes()).hexdigest()
    old_hash = json.loads((args.cache_root/'baseline.json').read_text())['setup']['flow_sha256']
    if source_hash != old_hash:
        raise ValueError('Source checkpoint changed since baseline diagnosis')
    protocol = {'source': str(FLOW), 'source_sha256': source_hash, 'steps': args.steps,
        'batch_size': 32, 'lr': 3e-4, 'weight_decay': 0., 'width': 64,
        'trainable_scalars': sum(p.numel() for p in corrector.parameters()),
        'features': 'frozen mean route query, navigation query, generated path, generated speed',
        'input_label_use': 'labels only in loss; frozen full20step FM generation receives no GT',
        'correction_applied': 'after full path/speed generation, speed only; not fed back into solver',
        'loss': 'speed Huber beta=0.5 m/s + 0.2 stop/go squared hinge',
        'stop_hinge': 'raw speed <=0.1 when GT<=0.1; raw speed >=0.5 when GT>=2',
        'training_frames': len(train_gen), 'validation_frames': len(val_gen),
        'training_routes': len(routes(training)), 'validation_routes': len(routes(validation)),
        'max_speed_difference_vs_previous_TF32_run_mps': cache_speed_delta,
        'selection': {'step': 500, 'speed_MAE_improvement_min': .1,
            'false_go_increase': 0, 'false_stop_increase': 0, 'path_bitwise_unchanged': True},
        'scope': 'development test, prior FM pretraining used these validation routes; no CARLA launch'}
    write_json(args.output/'protocol.json', protocol)
    optimizer = torch.optim.AdamW(corrector.parameters(), lr=3e-4, weight_decay=0.)
    q, g, targets = training['query'].to(device), train_gen.to(device), training['target'][:, -1, 0].to(device)
    rng = np.random.default_rng(4137)
    with (args.output/'training.jsonl').open('w') as log:
        for step in range(1, args.steps+1):
            corrector.train()
            idx = torch.tensor(rng.choice(len(targets), 32, replace=False), device=device)
            optimizer.zero_grad(set_to_none=True)
            raw = corrector.raw_speed(q[idx], g[idx]); truth = targets[idx]
            huber = torch.nn.functional.smooth_l1_loss(raw, truth, beta=.5)
            stopped, moving = truth <= .1, truth >= 2
            stop = torch.relu(raw[stopped]-.1).square().mean() if stopped.any() else raw.new_zeros(())
            go = torch.relu(.5-raw[moving]).square().mean() if moving.any() else raw.new_zeros(())
            loss = huber+.2*(stop+go)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite corrector loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in corrector.parameters()):
                raise FloatingPointError('Nonfinite corrector gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(corrector.parameters(), 1.))
            optimizer.step()
            row = {'step': step, 'loss': float(loss), 'huber': float(huber), 'stop': float(stop), 'go': float(go), 'grad_norm': norm}
            log.write(json.dumps(row)+'\n'); log.flush()
            if step % 100 == 0:
                print('corrector', step, '/', args.steps, 'loss', row['loss'], flush=True)
            if step in (100, 250, 500) or step == args.steps:
                report = validate(corrector, validation, val_gen, device)
                write_json(args.output/f'generation_step{step:06d}.json', report)
                torch.save({'corrector': corrector.state_dict(), 'optimizer': optimizer.state_dict(),
                    'step': step, 'protocol': protocol}, args.output/f'corrector_step{step:06d}.pt')
                print('corrector validation', step, json.dumps(report['mean']), flush=True)
    assert hashlib.sha256(FLOW.read_bytes()).hexdigest() == source_hash
    write_json(args.output/'integrity.json', {'zero_initialization_reproduces_original_prediction': True,
        'source_checkpoint_hash_unchanged': True, 'only_new_corrector_optimized': True,
        'generated_path_bitwise_unchanged': True, 'updates': args.steps})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/analysis/bounded_speed_corrector_20261007')
    parser.add_argument('--cache-root', type=Path, default=ROOT/'outputs/analysis/bounded_fm_recovery_20261007')
    parser.add_argument('--steps', type=int, default=500)
    args = parser.parse_args()
    if not 1 <= args.steps <= 500:
        raise ValueError('Limit this experiment to 1--500 steps')
    random.seed(137); np.random.seed(137); torch.manual_seed(137); torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.cuda.set_per_process_memory_fraction(.5)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    run(args)


if __name__ == '__main__':
    main()
