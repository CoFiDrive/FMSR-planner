"""Bounds for the continuous PDM target, independent of TF++ class centers."""
import math


# Asset preparation accepts expert speed labels in [0, 30] m/s. The TF++
# classification centers end at 20 m/s and do not bound this continuous target.
MAX_TARGET_SPEED_MPS = 30.0


def continuous_speed_limit(config):
    limit = float(getattr(config, 'fmsr_max_target_speed_mps', MAX_TARGET_SPEED_MPS))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Continuous target speed limit must be finite and positive')
    return limit
