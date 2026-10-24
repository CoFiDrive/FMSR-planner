"""Deployment-style validation with private RNG, geometry and per-scene metrics."""
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import validation_subset
from .speed import continuous_speed_limit
from .regression_health import RegressionHealth
from .clean_prior import planning_context, prior_kwargs, prediction_base
from .planner import PaperBridgePlanner
from .precision import agent_precision


class ValidationFrames(Dataset):
    """Attach identity to a validation view without changing training samples."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        source = self.dataset
        while isinstance(source, Subset):
            index = source.indices[index]
            source = source.dataset
        directory = Path(source.measurements[index][0].decode())
        sample['validation_scenario'] = directory.parent.parent.name
        sample['validation_frame'] = str(directory / f'{source.sample_start[index]:04d}.json.gz')
        return sample


def generation_loader(dataset, samples, batch_size, workers=0):
    if samples <= 0:
        raise ValueError('Generation validation requires a positive sample budget')
    view = ValidationFrames(validation_subset(dataset, samples, seed=137))
    return DataLoader(view, batch_size=batch_size, shuffle=False, num_workers=workers,
                      generator=torch.Generator().manual_seed(137))


def per_frame_metrics(prediction, target):
    """Metrics in physical units; target is used only after generation."""
    path, truth = prediction[:, :-1], target[:, :-1]
    delta = path - truth
    route_error = torch.linalg.vector_norm(delta, dim=-1)
    spacing = torch.linalg.vector_norm(path[:, 1:] - path[:, :-1], dim=-1)
    true_spacing = torch.linalg.vector_norm(truth[:, 1:] - truth[:, :-1], dim=-1)
    bend = path[:, 2:] - 2 * path[:, 1:-1] + path[:, :-2]
    true_bend = truth[:, 2:] - 2 * truth[:, 1:-1] + truth[:, :-2]
    return {'route_ADE_m': route_error.mean(-1), 'route_FDE_m': route_error[:, -1],
            'near_lateral_MAE_m': delta[:, :5, 1].abs().mean(-1),
            'speed_MAE_mps': (prediction[:, -1, 0] - target[:, -1, 0]).abs(),
            'point_spacing_MAE_m': (spacing - true_spacing).abs().mean(-1),
            'second_difference_MAE_m': torch.linalg.vector_norm(bend - true_bend, dim=-1).mean(-1)}


def summarize(prediction, target):
    result = {key: float(value.mean()) for key, value in per_frame_metrics(prediction, target).items()}
    stopped = target[:, -1, 0] <= .1
    moving = target[:, -1, 0] >= 2
    result.update(samples=len(target), braking_frames=int(stopped.sum()), moving_frames=int(moving.sum()),
                  false_go_gt_le_0p1_pred_gt_0p5=int((stopped & (prediction[:, -1, 0] > .5)).sum()),
                  false_stop_gt_ge_2_pred_lt_0p5=int((moving & (prediction[:, -1, 0] < .5)).sum()))
    return result


@contextmanager
def validation_mode(model):
    previous = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(previous)


@contextmanager
def preserve_validation_rng():
    # Native loading still draws Python/NumPy RNG even when augmentation is off.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state); np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def evaluate_generation(model, loader, device, seeds=(137, 271, 811)):
    """Generate without labels; cache perception across seeds, preserving train RNG."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Generation validation seeds must be nonempty and distinct')
    mode_before = model.training
    model.eval()
    health = RegressionHealth(model.planner)
    generators = {seed: torch.Generator(device=device).manual_seed(seed) for seed in seeds}
    predictions = {seed: [] for seed in seeds}
    targets, scenarios, frames, priors = [], [], [], []
    timelines = {str(seed): [] for seed in seeds}
    try:
        with preserve_validation_rng(), agent_precision(), torch.autocast(device_type=device.type, enabled=False):
            for batch in loader:
                # Labels never enter context() or sample().
                query, bev, _, prior = planning_context(model,
                    batch['rgb'].to(device, dtype=torch.float32),
                    batch['lidar'].to(device, dtype=torch.float32),
                    batch['target_point'].to(device, dtype=torch.float32),
                    batch['speed'].to(device, dtype=torch.float32).reshape(-1, 1),
                    batch['command'].to(device, dtype=torch.float32),
                    batch['target_point_next'].to(device, dtype=torch.float32))
                truth = torch.cat([batch['route'][:, :10].float(), torch.stack([
                    batch['target_speed_value'].float().reshape(-1),
                    torch.zeros(len(query))], -1)[:, None]], 1)
                targets.append(truth.cpu())
                if prior is not None:
                    priors.append(prior.float().cpu())
                scenarios.extend(batch['validation_scenario'])
                frames.extend(batch['validation_frame'])
                for seed in seeds:
                    with_trace = isinstance(model.planner, PaperBridgePlanner)
                    result = model.planner.sample(query.float(), bev.float(), generator=generators[seed],
                        **prior_kwargs(prior), **({'return_trace': True} if with_trace else {}))
                    generated = result[0]
                    if with_trace:
                        timeline = timelines[str(seed)]
                        for index, step in enumerate(result[2]):
                            if len(timeline) <= index:
                                timeline.append({'t': step['t'], 'samples': 0, 'state_ADE_m': 0.,
                                                 'clean_prediction_ADE_m': 0.})
                            item = timeline[index]
                            item['samples'] += len(query)
                            for name, output in [('state_ADE_m', step['state']),
                                                 ('clean_prediction_ADE_m', step['clean_prediction'])]:
                                item[name] += float((output.cpu()-truth[:, :10]).norm(dim=-1).mean(-1).sum())
                    generated = generated.cpu()
                    generated[:, -1, 0].clamp_(0, continuous_speed_limit(model.config))
                    predictions[seed].append(generated)
        if not targets:
            raise ValueError('Generation validation loader is empty')
        target = torch.cat(targets)
        groups = defaultdict(list)
        for index, name in enumerate(scenarios):
            groups[name].append(index)
        stacked = torch.stack([torch.cat(predictions[seed]) for seed in seeds])
        per_seed = {str(seed): summarize(stacked[i], target) for i, seed in enumerate(seeds)}
        continuous_keys = per_frame_metrics(stacked[0], target).keys()
        average = {key: sum(per_seed[str(seed)][key] for seed in seeds)/len(seeds) for key in continuous_keys}
        spread = stacked[:, :, :10].std(dim=0, unbiased=False)
        average['between_seed_path_STD_m'] = float(spread.norm(dim=-1).mean())
        for timeline in timelines.values():
            for item in timeline:
                for key in ('state_ADE_m', 'clean_prediction_ADE_m'):
                    item[key] /= item['samples']
        report = {'scope': 'predicted_anchor_full_sampling_no_label_input',
                'prediction_base': prediction_base(model.config),
                'precision': 'FP32_with_agent_TF32_settings', 'samples': len(target),
                'inference_steps': model.config.step_num, 'seeds': list(seeds),
                'mean_over_seeds': average, 'per_seed': per_seed,
                'regression_health': health.report(),
                'scenarios': {name: {str(seed): summarize(stacked[i, indices], target[indices])
                                     for i, seed in enumerate(seeds)} for name, indices in sorted(groups.items())},
                'frames': frames, 'denoising_timeline': timelines}
        if getattr(model.config, 'trajectory_generator', 'ddbm') == 'flow_matching':
            report.update(trajectory_generator='flow_matching',
                          sampler='deterministic_Euler_tau_0_to_1',
                          timeline_time_coordinate='t=1-tau',
                          CR_enabled=getattr(model.config, 'flow_contract_enable', False))
        if priors:
            prior_path = torch.cat(priors)
            prior_tokens = torch.cat([prior_path, torch.zeros(len(prior_path), 1, 2)], 1)
            report['frozen_TFPP_prior_path_metrics'] = {k: float(v.mean()) for k,v in
                per_frame_metrics(prior_tokens, target).items() if k != 'speed_MAE_mps'}
            average['correction_from_prior_RMS_m'] = float((stacked[:, :, :10]-prior_path).square().mean().sqrt())
            average['generated_minus_prior_route_ADE_m'] = (average['route_ADE_m'] -
                report['frozen_TFPP_prior_path_metrics']['route_ADE_m'])
        return report
    finally:
        health.close()
        model.train(mode_before)
