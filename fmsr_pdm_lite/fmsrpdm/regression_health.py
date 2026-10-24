"""Observe inactive regression blocks and explicitly warm-start a damaged refiner."""
import math

import torch


class RegressionHealth:
    def __init__(self, planner):
        self.counts = {}
        self.handles = []
        if not hasattr(planner, 'diff_decoder'):
            return
        branches = [(f'layer{layer}', decoder.task_decoder.plan_reg_branch)
                    for layer, decoder in enumerate(planner.diff_decoder.layers)]
        branches.extend((f'clean_path_layer{layer}', head)
                        for layer, head in enumerate(getattr(planner, 'clean_path_heads', [])))
        for name, branch in branches:
            for index, module in enumerate(branch):
                if isinstance(module, (torch.nn.ReLU, torch.nn.SiLU)):
                    key = f'{name}/{"relu" if isinstance(module, torch.nn.ReLU) else "silu"}{index}'
                    self.counts[key] = {'active': None, 'observed': 0}
                    def record(module, inputs, value, key=key):
                        count = self.counts[key]
                        active = (value.detach() != 0).sum()
                        count['active'] = active if count['active'] is None else count['active'] + active
                        count['observed'] += value.numel()
                    self.handles.append(module.register_forward_hook(record))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self):
        return {key: {'active_fraction': float(value['active']/value['observed']) if value['observed'] else None,
                      'observed_values': value['observed'],
                      'fully_inactive': bool(value['observed'] and int(value['active']) == 0)}
                for key, value in self.counts.items()}


@torch.no_grad()
def reset_regression_refiners(planner, layers, seed=137):
    """Explicit repair experiment; never called during normal loading/training.

    Keep all shapes and keys. Refresh hidden layers and zero the last residual
    projection, so the new refiner initially passes through the preceding path.
    """
    if len(set(layers)) != len(layers) or any(i < 0 or i >= len(planner.diff_decoder.layers) for i in layers):
        raise ValueError('Invalid or duplicate regression layer indices')
    repaired = []
    for layer in layers:
        branch = planner.diff_decoder.layers[layer].task_decoder.plan_reg_branch
        linears = [module for module in branch if isinstance(module, torch.nn.Linear)]
        generator = torch.Generator(device=linears[0].weight.device).manual_seed(seed+layer)
        for linear in linears[:-1]:
            torch.nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5), generator=generator)
            bound = 1/math.sqrt(linear.in_features)
            torch.nn.init.uniform_(linear.bias, -bound, bound, generator=generator)
        linears[-1].weight.zero_(); linears[-1].bias.zero_()
        repaired.append(layer)
    return repaired
