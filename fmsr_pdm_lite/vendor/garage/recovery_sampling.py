"""Distributed fixed-ratio sampling of expert and recovery frames."""

import math
from collections import Counter
import json
from pathlib import Path

import torch
from torch.utils.data import Sampler


class RecoveryMixSampler(Sampler):
  def __init__(self, recovery_flags, recovery_fraction, num_replicas, rank, seed=0,
               max_recovery_repeats=8, recovery_groups=None, recovery_weights=None,
               group_weights=None):
    if not 0.0 < recovery_fraction < 1.0:
      raise ValueError('recovery_fraction must be between zero and one')
    if not 0 <= rank < num_replicas:
      raise ValueError('Invalid distributed rank')
    if max_recovery_repeats < 1:
      raise ValueError('max_recovery_repeats must be positive')
    self.base = [i for i, flag in enumerate(recovery_flags) if not flag]
    self.recovery = [i for i, flag in enumerate(recovery_flags) if flag]
    if not self.base or not self.recovery:
      raise ValueError('Recovery post-training needs both base and recovery frames')
    self.fraction = recovery_fraction
    self.num_replicas = num_replicas
    self.rank = rank
    self.seed = seed
    self.epoch = 0
    self.max_recovery_repeats = max_recovery_repeats
    self.last_epoch_stats = {}
    self.groups = None
    if recovery_groups is not None:
      if recovery_weights is None or group_weights is None:
        raise ValueError('Balanced recovery sampling requires frame and scenario weights')
      self.groups = {}
      self.recovery_weights = recovery_weights
      for index in self.recovery:
        if index not in recovery_groups or index not in recovery_weights:
          raise ValueError(f'Missing recovery sampling metadata for index {index}')
        weight = float(recovery_weights[index])
        if not math.isfinite(weight) or weight <= 0:
          raise ValueError('Every retained frame must have a finite positive weight')
        self.groups.setdefault(recovery_groups[index], []).append(index)
      self.group_weights = {name: float(group_weights[name]) for name in sorted(self.groups)}
      if any(not math.isfinite(w) or w <= 0 for w in self.group_weights.values()):
        raise ValueError('Scenario weights must be finite and positive')
    # With a small recovery collection, traversing all 339k original frames
    # each epoch would repeat each recovery frame hundreds of times.
    capped_total = math.floor(len(self.recovery) * max_recovery_repeats / recovery_fraction)
    if self.groups is not None:
      # Equal/proportional scenario quotas must not repeat the rarest scenario
      # more than the requested per-frame cap. Shorten the epoch accordingly.
      unit = min(len(pool) * max_recovery_repeats / self.group_weights[name]
                 for name, pool in self.groups.items())
      balanced_budget = math.floor(unit * sum(self.group_weights.values()))
      capped_total = min(capped_total, math.floor(balanced_budget / recovery_fraction))
    self.num_samples = max(1, min(math.ceil(len(recovery_flags) / num_replicas),
                                  capped_total // num_replicas))

  def set_epoch(self, epoch):
    self.epoch = epoch

  def __len__(self):
    return self.num_samples

  def scenario_quotas(self, count):
    total_weight = sum(self.group_weights.values())
    exact = {name: count * weight / total_weight for name, weight in self.group_weights.items()}
    quotas = {name: math.floor(value) for name, value in exact.items()}
    left = count - sum(quotas.values())
    for name in sorted(exact, key=lambda name: (-(exact[name] - quotas[name]), name)):
      capacity = len(self.groups[name]) * self.max_recovery_repeats
      if left and quotas[name] < capacity:
        quotas[name] += 1
        left -= 1
    if left or any(quotas[name] > len(self.groups[name]) * self.max_recovery_repeats for name in quotas):
      raise ValueError('Scenario quotas exceed the per-frame repeat cap')
    return quotas

  def describe(self):
    if self.groups is None:
      return {'mode': 'uniform', 'samples_per_rank': self.num_samples}
    n_recovery = round(self.num_samples * self.num_replicas * self.fraction)
    return {'mode': 'scenario_balanced', 'samples_per_rank': self.num_samples,
            'recovery_draws_global': n_recovery, 'per_frame_repeat_cap': self.max_recovery_repeats,
            'scenario_quotas': self.scenario_quotas(n_recovery)}

  def __iter__(self):
    generator = torch.Generator().manual_seed(self.seed + self.epoch)
    count = self.num_samples * self.num_replicas
    n_recovery = round(count * self.fraction)
    n_base = count - n_recovery

    def draw(pool, n):
      indices = []
      while len(indices) < n:
        order = torch.randperm(len(pool), generator=generator).tolist()
        indices.extend(pool[i] for i in order)
      return indices[:n]

    base_indices = draw(self.base, n_base)
    if self.groups is None:
      recovery_indices = draw(self.recovery, n_recovery)
    else:
      recovery_indices = []
      for name, quota in self.scenario_quotas(n_recovery).items():
        pool = self.groups[name]
        weights = torch.tensor([self.recovery_weights[i] for i in pool], dtype=torch.double)
        selected = []
        # Weighted permutations give every frame positive probability, while
        # preventing a short scenario from being repeated dozens of times.
        for _ in range(self.max_recovery_repeats):
          if len(selected) >= quota:
            break
          amount = min(len(pool), quota - len(selected))
          order = torch.multinomial(weights, amount, replacement=False, generator=generator).tolist()
          selected.extend(pool[i] for i in order)
        recovery_indices.extend(selected)
    indices = base_indices + recovery_indices
    order = torch.randperm(count, generator=generator).tolist()
    shuffled = [indices[i] for i in order]
    counts = Counter(recovery_indices)
    self.last_epoch_stats = {'recovery_fraction': n_recovery / count,
                            'recovery_unique_fraction': len(counts) / len(self.recovery),
                            'max_recovery_exposures': max(counts.values(), default=0)}
    if self.groups is not None:
      self.last_epoch_stats['long_stationary_fraction'] = sum(
          self.recovery_weights[i] < 1 for i in recovery_indices) / max(n_recovery, 1)
      for name, quota in self.scenario_quotas(n_recovery).items():
        self.last_epoch_stats[f'recovery_scenario/{name}'] = quota / max(n_recovery, 1)
    return iter(shuffled[self.rank:count:self.num_replicas])


def load_recovery_sampling_plan(dataset, path):
  plan = json.loads(Path(path).read_text())
  if plan['version'] != 1:
    raise ValueError('Unsupported recovery sampling plan version')
  groups, weights, used = {}, {}, set()
  for index, flag in enumerate(dataset.sample_is_recovery):
    if not flag:
      continue
    measurement_root = Path(dataset.measurements[index][0].decode())
    key = f'{measurement_root.parent.parent.name}/{measurement_root.parent.name}/{int(dataset.sample_start[index]):04d}'
    if key not in plan['frames']:
      raise ValueError(f'Sampling plan has no entry for {key}')
    metadata = plan['frames'][key]
    groups[index] = metadata['scenario']
    weights[index] = metadata['weight']
    used.add(key)
  if used != set(plan['frames']):
    raise ValueError('Loaded recovery frames differ from the sampling plan')
  return {'recovery_groups': groups, 'recovery_weights': weights, 'group_weights': plan['scenario_weights']}
