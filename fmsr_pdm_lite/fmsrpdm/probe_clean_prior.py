"""Bounded clean-prior architecture experiment; saves no deployable checkpoint."""
import argparse
import copy
import json
from pathlib import Path
import random

import jsonpickle
import jsonpickle.ext.numpy as jnp
import numpy as np
import torch

from .checkpoint import load_initial_checkpoint
from .clean_prior import NOISY_RESIDUAL, TFPP_CLEAN_PRIOR
from .configuration import RunConfig
from .dataset import create_dataset, validate_assets
from .generation_validation import summarize
from .model import FMSRModel
from .probe_rollout_repair import cache, groups
from .regression_health import RegressionHealth


@torch.no_grad()
def evaluate(planner, cached, device, batch_size, use_prior):
    query, feature, target, _, prior = cached
    planner.eval()
    health = RegressionHealth(planner)
    records, paths, timelines = {}, [], {}
    try:
        for seed in (137, 271, 811):
            generator = torch.Generator(device=device).manual_seed(seed)
            outputs, timeline = [], []
            for start in range(0, len(target), batch_size):
                stop = start+batch_size
                output, _, trace = planner.sample(query[start:stop].to(device), feature[start:stop].to(device),
                    generator=generator, return_trace=True,
                    **({'clean_prior': prior[start:stop].to(device)} if use_prior else {}))
                outputs.append(output.cpu())
                for index, row in enumerate(trace):
                    if len(timeline) <= index:
                        timeline.append({'t': row['t'], 'samples': 0, 'clean_prediction_ADE_m': 0.})
                    timeline[index]['samples'] += len(output)
                    timeline[index]['clean_prediction_ADE_m'] += float((row['clean_prediction'].cpu()-
                        target[start:stop, :10]).norm(dim=-1).mean(-1).sum())
            output = torch.cat(outputs); output[:, -1, 0].clamp_(0,30)
            records[str(seed)] = summarize(output, target)
            paths.append(output[:, :10])
            for row in timeline: row['clean_prediction_ADE_m'] /= row['samples']
            timelines[str(seed)] = timeline
        means = {key: sum(row[key] for row in records.values())/len(records) for key in records['137']
                 if isinstance(records['137'][key], float)}
        stacked = torch.stack(paths)
        means['between_seed_path_STD_m'] = float(stacked.std(0, unbiased=False).norm(dim=-1).mean())
        means['correction_from_prior_RMS_m'] = float((stacked-prior).square().mean().sqrt())
        return {'mean_over_seeds': means, 'per_seed': records, 'denoising_timeline': timelines,
                'regression_health': health.report()}
    finally:
        health.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--audit-frames', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--frames-per-scenario', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--geometry-weights', type=float, nargs='+', default=[0., .5])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.steps < 1 or args.frames_per_scenario < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError('Invalid bounded experiment settings')
    if not args.geometry_weights or len(set(args.geometry_weights)) != len(args.geometry_weights) or any(
            not np.isfinite(w) or w < 0 for w in args.geometry_weights):
        raise ValueError('Geometry weights must be finite, nonnegative and distinct')
    if args.output.exists() and any(args.output.iterdir()): raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(137); np.random.seed(137); torch.manual_seed(137); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    root = args.checkpoint.resolve().parent
    run = RunConfig(**json.loads((root/'run_config.json').read_text()))
    validate_assets(run.assets_dir)
    jnp.register_handlers()
    cfg = jsonpickle.decode((root/'config.json').read_text())
    cfg.prediction_base = TFPP_CLEAN_PRIOR
    cfg.fmsr_trajectory_head = 'experimental_tfpp_clean_prior_v1'
    cfg.training_scope = 'clean_path_only'
    cfg.rollout_denoising_weight = cfg.rollout_geometry_weight = cfg.path_geometry_weight = 0.
    model = FMSRModel(cfg)
    planner_cfg = model.planner.config  # released TrajectoryHead deep-copies its configuration
    migration = load_initial_checkpoint(args.checkpoint, model, migrate_noisy_residual=True)
    model.to(device).eval()
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    training_cfg = copy.deepcopy(cfg)
    training_cfg.augment = training_cfg.use_color_aug = training_cfg.lidar_aug_prob = 0
    training_data, validation_data = create_dataset(run, training_cfg), create_dataset(run, cfg, validation=True)
    audit = json.loads(args.audit_frames.read_text())['frames']
    rng = np.random.default_rng(137)
    available = groups(training_data)
    train_indices = [int(i) for name in sorted({r['scenario'] for r in audit}) for i in
                     rng.choice(available[name], min(len(available[name]), args.frames_per_scenario), replace=False)]
    val_indices = [r['dataset_index'] for r in audit]
    train_cache = cache(model, training_data, train_indices, device, args.batch_size, include_prior=True)
    val_cache = cache(model, validation_data, val_indices, device, args.batch_size, include_prior=True)
    assert val_cache[3] == [r['frame'] for r in audit]
    if {str(Path(p).parent.parent) for p in train_cache[3]} & {str(Path(p).parent.parent) for p in val_cache[3]}:
        raise ValueError('Training and validation routes overlap')
    # Only the formula is switched for the original-model control. No weights change.
    planner_cfg.prediction_base, planner_cfg.training_scope = NOISY_RESIDUAL, 'joint'
    original = evaluate(model.planner, val_cache, device, args.batch_size, use_prior=False)
    planner_cfg.prediction_base, planner_cfg.training_scope = TFPP_CLEAN_PRIOR, 'clean_path_only'
    clean_initial = evaluate(model.planner, val_cache, device, args.batch_size, use_prior=True)
    print('Original:', original['mean_over_seeds'], flush=True)
    print('Clean-prior initialization:', clean_initial['mean_over_seeds'], flush=True)
    report = {'scope': 'Architecture experiment on repeatedly used development validation; no DS/SR claim',
              'source_checkpoint': str(args.checkpoint.resolve()), 'migration': migration,
              'settings': {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'training_frames': train_cache[3], 'validation_frames': val_cache[3],
              'train_validation_routes_disjoint': True,
              'original_noisy_residual': original, 'zero_initialized_clean_prior': clean_initial, 'variants': {}}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    order_rng = np.random.default_rng(4137)
    batches = [order_rng.choice(len(train_indices), args.batch_size, replace=False) for _ in range(args.steps)]
    for weight in args.geometry_weights:
        model.load_state_dict(initial, strict=True)
        cfg.path_geometry_weight = weight
        planner_cfg.path_geometry_weight = weight
        optimizer = torch.optim.AdamW(model.planner.clean_path_heads.parameters(), lr=args.lr, weight_decay=.01)
        losses, checkpoints = [], {}
        for step, indices in enumerate(batches, 1):
            model.train()
            query, feature, target = (row[indices].to(device) for row in train_cache[:3])
            prior = train_cache[4][indices].to(device)
            values, _ = model.planner.training_loss(query, feature, target, clean_prior=prior,
                generator=torch.Generator(device=device).manual_seed(7137+step))
            loss = sum(values.values())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite probe loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite probe gradient')
            optimizer.step()
            losses.append({k:float(v.detach()) for k,v in values.items()})
            if step % 10 == 0: print('geometry', weight, 'step', step, '/', args.steps, 'loss', float(loss), flush=True)
            if step in {args.steps//2, args.steps}:
                checkpoints[str(step)] = evaluate(model.planner, val_cache, device, args.batch_size, use_prior=True)
                print('Validation:', weight, step, checkpoints[str(step)]['mean_over_seeds'], flush=True)
        changed = [k for k,v in model.state_dict().items() if not k.startswith('planner.clean_path_heads.')
                   and not torch.equal(v.cpu(), initial[k])]
        if changed: raise AssertionError(f'Frozen parameters/buffers changed: {changed[:5]}')
        folder = args.output/f'geometry_{weight:g}'; folder.mkdir()
        torch.save(model.planner.state_dict(), folder/'probe_planner.pt')
        (folder/'config.json').write_text(jsonpickle.encode(cfg,indent=2)+'\n')
        report['variants'][str(weight)] = {'validation_checkpoints': checkpoints, 'training_losses': losses,
                                          'frozen_parameters_changed': changed}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Clean-prior probe complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
