"""Versioned PDM planner interfaces and continuous-speed decoding."""
import math
import torch


def restore_trained_interface(config, saved_config):
  """Preserve learned input semantics when loading or resuming old planners."""
  saved = vars(saved_config)
  config.fmsr_use_target_point = saved.get('fmsr_use_target_point', False)
  config.fmsr_bev_layout = saved.get('fmsr_bev_layout', 'legacy')
  config.fmsr_speed_source = saved.get('fmsr_speed_source', 'planner')
  config.fmsr_curve_max_lateral_accel = saved.get('fmsr_curve_max_lateral_accel', 3.0)
  planner_speed_source(config)


def planner_speed_source(config):
  source = getattr(config, 'fmsr_speed_source', 'planner')
  if source not in ('planner', 'tfpp_uncertainty_v1', 'tfpp_curvature_v1'):
    raise ValueError(f'Unknown FMSR speed source: {source}')
  return source


def decode_frozen_tfpp_speed(logits, config):
  """Original categorical expectation with the saved stop-probability rule."""
  if logits is None or logits.ndim != 2 or logits.shape[-1] != len(config.target_speeds):
    raise ValueError('TF++ speed source requires compatible categorical logits')
  probability = torch.softmax(logits.float(), dim=-1)
  centers = probability.new_tensor(config.target_speeds)
  expected = (probability * centers).sum(-1)
  threshold = float(config.brake_uncertainty_threshold)
  if not math.isfinite(threshold) or not 0 <= threshold <= 1:
    raise ValueError('Invalid saved brake probability threshold')
  speed = torch.where(probability[:, 0] > threshold, centers[0], expected)
  return speed.clamp(0., continuous_speed_limit(config)), probability[:, 0]


def path_curvature_speed_cap(path, config):
  """Experimental geometric cap from the actual predicted spatial path.

  Three-point circumcircle curvature is translation/rotation invariant.
  The configured lateral acceleration is a physical prior, not a fitted label.
  """
  accel = float(getattr(config, 'fmsr_curve_max_lateral_accel', 3.0))
  if not math.isfinite(accel) or accel <= 0:
    raise ValueError('Curvature cap requires positive finite lateral acceleration')
  if path.ndim != 3 or path.shape[1] < 3 or path.shape[2] < 2:
    raise ValueError('Curvature cap requires [batch, >=3 poses, >=2 coordinates]')
  xy = path[..., :2].float()
  first, second = xy[:, 1:-1] - xy[:, :-2], xy[:, 2:] - xy[:, 1:-1]
  chord = xy[:, 2:] - xy[:, :-2]
  denominator = first.norm(dim=-1) * second.norm(dim=-1) * chord.norm(dim=-1)
  cross = (first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]).abs()
  curvature = torch.where(denominator > 1e-6, 2 * cross / denominator.clamp_min(1e-6), 0.)
  peak = curvature.amax(dim=1)
  cap = torch.sqrt(accel / peak.clamp_min(1e-6)).clamp(max=continuous_speed_limit(config))
  return cap, peak


def continuous_speed_limit(config):
  """Continuous regression has its own bound, independent of TF++ class bins."""
  limit = float(getattr(config, 'fmsr_max_speed', 30.0))
  if not math.isfinite(limit) or limit <= 0.0:
    raise ValueError('fmsr_max_speed must be positive and finite')
  return limit


def continuous_speed_for_control(predictions, maximum_speed, speed_factor=1.0):
  """Average scalar ensemble predictions without quantization or pseudo confidence."""
  speed = torch.stack(predictions).mean()
  return float(speed.clamp(0.0, maximum_speed).item()) * speed_factor
