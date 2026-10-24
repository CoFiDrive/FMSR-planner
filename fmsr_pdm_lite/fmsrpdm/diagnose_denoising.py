"""Trace real sampling against the training bridge on existing audit frames."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import jsonpickle
import jsonpickle.ext.numpy as jnp
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .configuration import RunConfig
from .dataset import create_dataset
from .model import FMSRModel
from .clean_prior import planning_context, prior_kwargs
from .planner import nearest_anchor
from .train import tensor_inputs


def summarize_prediction(prediction, target):
    delta = prediction[:, :10] - target[:, :10]
    return {'ADE_m': float(delta.norm(dim=-1).mean()),
            'FDE_m': float(delta[:, -1].norm(dim=-1).mean()),
            'per_point_error_m': delta.norm(dim=-1).mean(0).cpu().tolist(),
            'speed_MAE_mps': float((prediction[:, -1, 0] - target[:, -1, 0]).abs().mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--audit-frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(137)
    device = torch.device(args.device)
    root = args.checkpoint.resolve().parent
    run = RunConfig(**json.loads((root/'run_config.json').read_text()))
    jnp.register_handlers()
    config = jsonpickle.decode((root/'config.json').read_text())
    model = FMSRModel(config).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    dataset = create_dataset(run, config, validation=True)
    source_rows = json.loads(args.audit_frames.read_text())['frames']
    selected = []
    seen = set()
    for row in source_rows:
        if row['scenario'] not in seen:
            selected.append(row); seen.add(row['scenario'])
    loader = DataLoader(Subset(dataset, [r['dataset_index'] for r in selected]),
                        batch_size=args.batch_size, shuffle=False, num_workers=0,
                        generator=torch.Generator().manual_seed(137))
    all_rows = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            inputs = tensor_inputs(batch, device)
            query, bev, _, prior = planning_context(model, inputs['rgb'], inputs['lidar_bev'], inputs['target_point'],
                                          inputs['ego_vel'], inputs['command'], inputs['target_point_next'])
            target = torch.cat([inputs['target_route'][:, :10], torch.stack([
                inputs['target_speed'].reshape(-1), torch.zeros(len(query), device=device)], -1)[:, None]], 1)
            mode = model.planner.classify(query, bev)[-1].argmax(-1)
            correct = mode == nearest_anchor(target, model.planner.plan_anchor)
            original_denoise = model.planner.denoise
            traces = {}
            for seed in (137, 271, 811):
                steps = []
                def traced(state, anchors, t, q, b, **kwargs):
                    predictions = original_denoise(state, anchors, t, q, b, **kwargs)
                    coefficient_a, coefficient_b, coefficient_c = model.planner.bridge.coefficients(t)
                    c = float(coefficient_c[0])
                    residual_rms = None
                    if c > 0:
                        # Under the true training bridge, this has Gaussian RMS about 1.
                        residual = (model.planner.norm_odo(state) - coefficient_a[:, None, None, None] *
                                    model.planner.norm_odo(anchors) - coefficient_b[:, None, None, None] *
                                    model.planner.norm_odo(target[:, None])) / coefficient_c[:, None, None, None]
                        residual_rms = float(residual.flatten(2)[..., :-1].square().mean().sqrt())
                    steps.append({'t': float(t[0]), 'state': summarize_prediction(state[:, 0], target),
                                  'layers': [summarize_prediction(p[:, 0], target) for p in predictions],
                                  'training_bridge_residual_RMS': residual_rms})
                    return predictions
                with patch.object(model.planner, 'denoise', side_effect=traced):
                    model.planner.sample(query, bev, generator=torch.Generator(device=device).manual_seed(seed+batch_index),
                                         **prior_kwargs(prior))
                traces[str(seed)] = steps
            anchors = model.planner.plan_anchor[mode][:, None]
            oracle = []
            for step in (1., .95, .8, .5, .2, .05):
                t = torch.full((len(target),), step, device=device)
                noise = torch.randn(anchors.shape, device=device,
                                    generator=torch.Generator(device=device).manual_seed(5137+batch_index))
                state = model.planner.denorm_odo(model.planner.bridge.add_noise(
                    model.planner.norm_odo(target[:, None]), model.planner.norm_odo(anchors), t, noise))
                state[..., -1, 1] = 0
                predictions = original_denoise(state, anchors, t, query, bev, **prior_kwargs(prior))
                oracle.append({'t': step, 'input': summarize_prediction(state[:, 0], target),
                               'layers': [summarize_prediction(p[:, 0], target) for p in predictions]})
            all_rows.append({'batch_index': batch_index, 'samples': len(target),
                             'anchor_correct': int(correct.sum()), 'sampling': traces,
                             'true_training_bridge': oracle})
            print(f'Denoising trace: batch {batch_index+1}/{len(loader)}', flush=True)
    report = {'scope': 'One existing audit frame per scenario, no training or CARLA',
              'checkpoint': str(args.checkpoint.resolve()), 'frames': selected, 'batches': all_rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Saved:', args.output, flush=True)


if __name__ == '__main__':
    main()
