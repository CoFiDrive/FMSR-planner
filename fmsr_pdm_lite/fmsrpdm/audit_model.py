"""Bounded offline diagnosis of deployed sampling; never starts CARLA or training."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from unittest.mock import patch

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .configuration import RunConfig
from .dataset import create_dataset, validate_assets
from .model import FMSRModel
from .clean_prior import planning_context, prior_kwargs
from .planner import nearest_anchor
from .train import tensor_inputs
from .speed import continuous_speed_limit


def scenario_indices(dataset):
    groups = defaultdict(list)
    for index, paths in enumerate(dataset.measurements):
        groups[Path(paths[0].decode()).parent.parent.name].append(index)
    return groups


def error_values(prediction, target):
    route = torch.linalg.vector_norm(prediction[:, :10] - target[:, :10], dim=-1)
    speed = prediction[:, -1, 0]
    return {'route_ADE': route.mean(-1), 'route_FDE': route[:, -1],
            'speed_MAE': (speed - target[:, -1, 0]).abs(), 'speed_prediction_mps': speed}


def summarize(rows):
    result = {'samples': len(rows)}
    for mode in ('noisy_gt_anchor', 'sampled_predicted_anchor', 'sampled_gt_anchor', 'sampled_legacy_speed_cap'):
        result[mode] = {key: float(np.mean([row[mode][key] for row in rows]))
                        for key in ('route_ADE', 'route_FDE', 'speed_MAE')}
    result['anchor_accuracy'] = float(np.mean([row['anchor_correct'] for row in rows]))
    braking = [r for r in rows if r['target_speed_mps'] <= .1]
    moving = [r for r in rows if r['target_speed_mps'] >= 2]
    result['braking_frames'] = len(braking)
    result['moving_frames'] = len(moving)
    result['false_go_gt_le_0p1_pred_gt_0p5'] = sum(
        r['sampled_predicted_anchor']['speed_prediction_mps'] > .5 for r in braking)
    result['false_stop_gt_ge_2_pred_lt_0p5'] = sum(
        r['sampled_predicted_anchor']['speed_prediction_mps'] < .5 for r in moving)
    return result


def audit(args):
    if args.samples_per_scenario < 1 or args.batch_size < 1:
        raise ValueError('Sample and batch counts must be positive')
    run_dir = args.checkpoint.resolve().parent
    run = RunConfig(**json.loads((run_dir / 'run_config.json').read_text()))
    validate_assets(run.assets_dir)
    jsonpickle_numpy.register_handlers()
    config = jsonpickle.decode((run_dir / 'config.json').read_text())
    config.reproduction_pretrained_images = False
    torch.set_num_threads(4)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Match SensorAgent inference precision settings. Training BF16 is opt-in.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    model = FMSRModel(config).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    dataset = create_dataset(run, config, validation=True)
    groups = scenario_indices(dataset)
    rng = np.random.default_rng(args.seed)
    indices = [int(i) for name in sorted(groups) for i in
               rng.choice(groups[name], min(args.samples_per_scenario, len(groups[name])), replace=False)]
    prefix_count = min(len(dataset), run.batch_size * run.validation_max_batches) if run.validation_max_batches else len(dataset)
    prefix = Counter(name for name, values in groups.items() for i in values if i < prefix_count)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False, num_workers=0)
    rows = []
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader):
            inputs = tensor_inputs(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.perception_amp and device.type == 'cuda'):
                queries, bev, _, prior = planning_context(model, inputs['rgb'], inputs['lidar_bev'], inputs['target_point'],
                                                inputs['ego_vel'], inputs['command'], inputs['target_point_next'])
            queries, bev = queries.float(), bev.float()
            target = torch.cat([inputs['target_route'][:, :10],
                                torch.stack([inputs['target_speed'].reshape(-1),
                                             torch.zeros(len(queries), device=device)], -1)[:, None]], 1)
            gt_mode = nearest_anchor(target, model.planner.plan_anchor)
            logits = model.planner.classify(queries, bev)[-1]
            predicted_mode = logits.argmax(-1)
            # Reproduce the logged denoising metric, including its true-label input.
            generator = torch.Generator(device=device).manual_seed(args.seed + batch_number)
            anchors = model.planner.plan_anchor[None].expand(len(target), -1, -1, -1)
            clean = target[:, None].expand_as(anchors)
            t = torch.randint(1, 1001, (len(target),), device=device, generator=generator).float() / 1000
            noise = torch.randn(clean.shape, device=device, generator=generator)
            noisy = model.planner.denorm_odo(model.planner.bridge.add_noise(
                model.planner.norm_odo(clean), model.planner.norm_odo(anchors), t, noise))
            noisy[..., -1, 1] = 0
            denoised = model.planner.denoise(noisy, anchors, t, queries, bev, **prior_kwargs(prior))[-1]
            denoised = denoised[torch.arange(len(target), device=device), gt_mode]
            seed = args.seed + 10000 + batch_number
            prediction, _ = model.planner.sample(queries, bev,
                generator=torch.Generator(device=device).manual_seed(seed), **prior_kwargs(prior))
            oracle_logits = torch.nn.functional.one_hot(gt_mode, len(model.planner.plan_anchor)).float() * 100
            with patch.object(model.planner, 'classify', return_value=[oracle_logits]):
                oracle, _ = model.planner.sample(queries, bev,
                    generator=torch.Generator(device=device).manual_seed(seed), **prior_kwargs(prior))
            # Apply the same output bounds as the deployed model.
            prediction[:, -1, 0].clamp_(0, continuous_speed_limit(config))
            oracle[:, -1, 0].clamp_(0, continuous_speed_limit(config))
            legacy_prediction = prediction.clone()
            legacy_prediction[:, -1, 0].clamp_(0, max(config.target_speeds))
            errors = {name: error_values(p, target) for name, p in
                      [('noisy_gt_anchor', denoised), ('sampled_predicted_anchor', prediction),
                       ('sampled_gt_anchor', oracle)]}
            errors['sampled_legacy_speed_cap'] = error_values(legacy_prediction, target)
            for offset in range(len(target)):
                index = indices[len(rows)]
                measurement_dir = Path(dataset.measurements[index][0].decode())
                rows.append({'dataset_index': index, 'scenario': measurement_dir.parent.parent.name,
                             'frame': str(measurement_dir / f'{dataset.sample_start[index]:04d}.json.gz'),
                             'target_speed_mps': float(target[offset, -1, 0]),
                             'predicted_anchor': int(predicted_mode[offset]), 'gt_anchor': int(gt_mode[offset]),
                             'anchor_correct': bool(predicted_mode[offset] == gt_mode[offset]),
                             **{name: {k: float(v[offset]) for k, v in values.items()}
                                for name, values in errors.items()}})
            print(f'Offline diagnosis: {len(rows)}/{len(indices)} frames', flush=True)
    report = {'checkpoint': str(args.checkpoint.resolve()), 'seed': args.seed,
              'sampling': 'uniform frames within each scenario, scenario-balanced; not a closed-loop score',
              'inference_steps': config.step_num, 'logged_validation_prefix_frames': prefix_count,
              'continuous_speed_limit_mps': continuous_speed_limit(config),
              'perception_precision': 'bf16' if args.perception_amp else 'fp32_with_agent_tf32_settings',
              'logged_validation_prefix_scenarios': dict(prefix), 'validation_pool_scenarios': {k:len(v) for k,v in groups.items()},
              'overall': summarize(rows),
              'anchor_correct': summarize([r for r in rows if r['anchor_correct']]) if any(r['anchor_correct'] for r in rows) else None,
              'anchor_wrong': summarize([r for r in rows if not r['anchor_correct']]) if any(not r['anchor_correct'] for r in rows) else None,
              'scenarios': {name: summarize([r for r in rows if r['scenario'] == name]) for name in sorted(groups)},
              'frames': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['overall'], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples-per-scenario', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=137)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--perception-amp', action='store_true', help='Use training BF16 instead of agent inference precision')
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
