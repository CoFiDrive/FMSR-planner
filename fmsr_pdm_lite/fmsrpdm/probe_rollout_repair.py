"""Bounded planner-only control versus rollout-supervision experiment, no CARLA."""
import argparse
import copy
import json
from pathlib import Path
import random

import jsonpickle
import jsonpickle.ext.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .configuration import RunConfig
from .dataset import create_dataset
from .generation_validation import summarize
from .model import FMSRModel
from .train import tensor_inputs
from .regression_health import reset_regression_refiners, RegressionHealth
from .clean_prior import planning_context, prediction_base, NOISY_RESIDUAL


def groups(dataset):
    result = {}
    for index, paths in enumerate(dataset.measurements):
        result.setdefault(Path(paths[0].decode()).parent.parent.name, []).append(index)
    return result


def cache(model, dataset, indices, device, batch_size, include_prior=False):
    query_rows, feature_rows, targets, frames, priors = [], [], [], [], []
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(137))
    with torch.inference_mode():
        for batch in loader:
            inputs = tensor_inputs(batch, device)
            query, feature, _, prior = planning_context(model, inputs['rgb'], inputs['lidar_bev'], inputs['target_point'],
                inputs['ego_vel'], inputs['command'], inputs['target_point_next'])
            if include_prior:
                if prior is None: raise ValueError('Caching a clean prior requires clean-prior mode')
                priors.append(prior.cpu())
            query_rows.append(query.cpu()); feature_rows.append(feature.cpu())
            targets.append(torch.cat([inputs['target_route'][:, :10], torch.stack([
                inputs['target_speed'].reshape(-1), torch.zeros(len(query), device=device)], -1)[:, None]], 1).cpu())
    for index in indices:
        directory = Path(dataset.measurements[index][0].decode())
        frames.append(str(directory / f'{dataset.sample_start[index]:04d}.json.gz'))
    # Clone outside inference_mode so the cached constants can be saved for backward.
    result = (torch.cat(query_rows).clone(), torch.cat(feature_rows).clone(), torch.cat(targets).clone(), frames)
    return (*result, torch.cat(priors).clone()) if include_prior else result


@torch.no_grad()
def evaluate(planner, cached, device, batch_size):
    query, feature, target, _ = cached
    planner.eval()
    health = RegressionHealth(planner)
    records = {}
    for seed in (137, 271, 811):
        generator = torch.Generator(device=device).manual_seed(seed)
        output = []
        for start in range(0, len(target), batch_size):
            prediction, _ = planner.sample(query[start:start+batch_size].to(device),
                                          feature[start:start+batch_size].to(device), generator=generator)
            output.append(prediction.cpu())
        prediction = torch.cat(output)
        prediction[:, -1, 0].clamp_(0, 30)
        records[str(seed)] = summarize(prediction, target)
    means = {k: sum(r[k] for r in records.values())/len(records) for k in records['137']
             if isinstance(records['137'][k], float)}
    health.close()
    return {'mean_over_seeds': means, 'per_seed': records, 'regression_health': health.report()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audit-frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--frames-per-scenario', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--rollout-weight', type=float, default=1.)
    parser.add_argument('--rollout-geometry-weight', type=float, default=0.)
    parser.add_argument('--variants', nargs='+', choices=['control', 'rollout', 'reset', 'reset_rollout'],
                        default=['control', 'rollout'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    if args.steps < 1 or args.frames_per_scenario < 1 or args.batch_size < 1 or args.lr <= 0 or args.rollout_weight <= 0:
        raise ValueError('Invalid bounded experiment settings')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Choose a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(137); np.random.seed(137); torch.manual_seed(137); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    root = args.checkpoint.resolve().parent
    run = RunConfig(**json.loads((root/'run_config.json').read_text()))
    jnp.register_handlers()
    cfg = jsonpickle.decode((root/'config.json').read_text())
    if prediction_base(cfg) != NOISY_RESIDUAL:
        raise ValueError('Use probe_clean_prior for clean-prior architecture experiments')
    model = FMSRModel(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    initial = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Fixed pretrained and navigation features isolate the planner objective.
    training_cfg = copy.deepcopy(cfg)
    training_cfg.augment = training_cfg.use_color_aug = 0
    training_cfg.lidar_aug_prob = 0
    training_data = create_dataset(run, training_cfg)
    validation_data = create_dataset(run, cfg, validation=True)
    audit = json.loads(args.audit_frames.read_text())['frames']
    scene_names = {r['scenario'] for r in audit}
    rng = np.random.default_rng(137)
    available = groups(training_data)
    train_indices = [int(i) for name in sorted(scene_names) for i in
                     rng.choice(available[name], min(len(available[name]), args.frames_per_scenario), replace=False)]
    validation_indices = [r['dataset_index'] for r in audit]
    train_cache = cache(model, training_data, train_indices, device, args.batch_size)
    validation_cache = cache(model, validation_data, validation_indices, device, args.batch_size)
    assert validation_cache[3] == [r['frame'] for r in audit]
    train_routes = {str(Path(p).parent.parent) for p in train_cache[3]}
    val_routes = {str(Path(p).parent.parent) for p in validation_cache[3]}
    if train_routes & val_routes:
        raise ValueError('Probe training and validation routes overlap')
    baseline = evaluate(model.planner, validation_cache, device, args.batch_size)
    order_rng = np.random.default_rng(4137)
    batches = [order_rng.choice(len(train_indices), args.batch_size, replace=False) for _ in range(args.steps)]
    report = {'scope': 'Bounded planner-only experimental fine-tuning on fixed unaugmented contexts, not full training',
              'source_checkpoint': str(args.checkpoint.resolve()),
              'settings': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
              'training_frames': train_cache[3], 'validation_frames': validation_cache[3],
              'train_validation_routes_disjoint': True, 'baseline': baseline, 'variants': {}}
    (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    for name in args.variants:
        weight = args.rollout_weight if 'rollout' in name else 0.
        model.load_state_dict(initial, strict=True)
        repaired = reset_regression_refiners(model.planner, [1]) if name.startswith('reset') else []
        model.planner.config.rollout_denoising_weight = weight
        model.planner.config.rollout_denoising_scenes = min(4, args.batch_size)
        model.planner.config.rollout_geometry_weight = args.rollout_geometry_weight if weight else 0.
        optimizer = torch.optim.AdamW([p for p in model.planner.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=.01)
        losses, checkpoints = [], {}
        for step, indices in enumerate(batches, 1):
            model.planner.train()
            torch.manual_seed(6137+step)
            if device.type == 'cuda': torch.cuda.manual_seed_all(6137+step)
            query, feature, target = (data[indices].to(device) for data in train_cache[:3])
            values, metrics = model.planner.training_loss(query, feature, target,
                generator=torch.Generator(device=device).manual_seed(7137+step))
            loss = sum(values.values())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite probe loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.planner.parameters()):
                raise FloatingPointError('Nonfinite probe gradient')
            optimizer.step()
            losses.append({k: float(v.detach()) for k,v in values.items()})
            if step % 10 == 0: print(name, 'step', step, '/', args.steps, 'loss', float(loss.detach()), flush=True)
            if step in {args.steps//2, args.steps}:
                checkpoints[str(step)] = evaluate(model.planner, validation_cache, device, args.batch_size)
                print(name, 'validation', step, checkpoints[str(step)]['mean_over_seeds'], flush=True)
        folder = args.output/name; folder.mkdir()
        # These are deliberately named as experimental artifacts, not deployable model_*.pth.
        torch.save(model.planner.state_dict(), folder/'probe_planner.pt')
        probe_cfg = copy.deepcopy(cfg)
        probe_cfg.rollout_denoising_weight = weight
        probe_cfg.rollout_denoising_scenes = min(4, args.batch_size)
        probe_cfg.rollout_geometry_weight = args.rollout_geometry_weight if weight else 0.
        (folder/'config.json').write_text(jsonpickle.encode(probe_cfg, indent=2)+'\n')
        frozen_changed = [k for k,v in model.state_dict().items()
                          if not k.startswith('planner.') and not torch.equal(v.cpu(), initial[k])]
        assert not frozen_changed
        assert torch.equal(model.planner.plan_anchor.cpu(), initial['planner.plan_anchor'])
        report['variants'][name] = {'rollout_weight': weight, 'validation_checkpoints': checkpoints,
                                    'training_losses': losses, 'frozen_parameters_changed': frozen_changed,
                                    'regression_layers_reset': repaired}
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Probe complete:', args.output, flush=True)


if __name__ == '__main__':
    main()
