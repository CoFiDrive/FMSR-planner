"""Inspect hidden activations and gradients without updating the checkpoint."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import jsonpickle
import jsonpickle.ext.numpy as jnp
import torch
from torch.utils.data import DataLoader, Subset

from .configuration import RunConfig
from .dataset import create_dataset
from .model import FMSRModel
from .clean_prior import planning_context, prior_kwargs
from .train import tensor_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--audit-frames', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4); torch.manual_seed(137)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    root = args.checkpoint.resolve().parent
    run = RunConfig(**json.loads((root/'run_config.json').read_text()))
    jnp.register_handlers(); cfg = jsonpickle.decode((root/'config.json').read_text())
    model = FMSRModel(cfg).to(device).eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    data = create_dataset(run, cfg, validation=True)
    rows = json.loads(args.audit_frames.read_text())['frames'][::4]
    loader = DataLoader(Subset(data, [r['dataset_index'] for r in rows]), batch_size=8, shuffle=False,
                        generator=torch.Generator().manual_seed(137))
    hidden, handles = {}, []
    for layer, block in enumerate(model.planner.diff_decoder.layers):
        for index in (1, 3):
            key = f'layer{layer}_relu{index}'
            def record(module, inputs, value, key=key):
                hidden[key] = {'nonzero_fraction': float((value != 0).float().mean()),
                               'mean_abs_output': float(value.detach().abs().mean()),
                               'input_mean': float(inputs[0].detach().mean())}
            handles.append(block.task_decoder.plan_reg_branch[index].register_forward_hook(record))
    result = []
    for batch in loader:
        inputs = tensor_inputs(batch, device)
        with torch.no_grad():
            query, bev, _, prior = planning_context(model, inputs['rgb'], inputs['lidar_bev'], inputs['target_point'],
                                          inputs['ego_vel'], inputs['command'], inputs['target_point_next'])
            history = []
            actual = model.planner.denoise
            def traced(state, anchors, t, q, b, **kwargs):
                output = actual(state, anchors, t, q, b, **kwargs)
                history.append({'t': float(t[0]), 'hidden': dict(hidden),
                                'second_layer_delta_mean_abs': float((output[1]-output[0]).abs().mean())})
                return output
            with patch.object(model.planner, 'denoise', side_effect=traced):
                model.planner.sample(query, bev, generator=torch.Generator(device=device).manual_seed(137), **prior_kwargs(prior))
        target = torch.cat([inputs['target_route'][:, :10], torch.stack([
            inputs['target_speed'].reshape(-1), torch.zeros(len(query), device=device)], -1)[:, None]], 1)
        model.planner.train(); model.zero_grad(set_to_none=True)
        losses, _ = model.planner.training_loss(query.detach(), bev.detach(), target,
                                               generator=torch.Generator(device=device).manual_seed(271), **prior_kwargs(prior))
        sum(losses.values()).backward()
        gradients = {}
        for name, parameter in model.planner.named_parameters():
            if 'plan_reg_branch' in name or 'clean_path_heads' in name:
                gradients[name] = None if parameter.grad is None else float(parameter.grad.norm())
        result.append({'samples': len(target), 'sampling': history, 'training_hidden': dict(hidden),
                       'regression_gradient_norms': gradients})
        model.eval()
    for handle in handles: handle.remove()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'scope':'31 audit frames, no optimizer updates', 'batches':result},indent=2)+'\n')
    print('Saved:', args.output, flush=True)


if __name__ == '__main__': main()
