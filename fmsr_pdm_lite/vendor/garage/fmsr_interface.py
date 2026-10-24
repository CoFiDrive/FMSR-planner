"""Versioned PDM planner interfaces and continuous-speed decoding."""
import torch


def restore_trained_interface(config, saved_config):
  """Preserve learned input semantics when loading or resuming old planners."""
  saved = vars(saved_config)
  config.fmsr_use_target_point = saved.get('fmsr_use_target_point', False)
  config.fmsr_bev_layout = saved.get('fmsr_bev_layout', 'legacy')


def continuous_speed_for_control(predictions, maximum_speed, speed_factor=1.0):
  """Average scalar ensemble predictions without quantization or pseudo confidence."""
  speed = torch.stack(predictions).mean()
  return float(speed.clamp(0.0, maximum_speed).item()) * speed_factor
