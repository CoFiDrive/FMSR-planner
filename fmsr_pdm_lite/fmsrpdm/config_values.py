"""Restore typed controller values from native and earlier plain-JSON configs."""
import numpy as np


def restore_value(value):
    if isinstance(value, dict):
        if set(value) == {'py/tuple'}:
            return tuple(restore_value(v) for v in value['py/tuple'])
        # Earlier runs read jsonpickle as plain JSON. Re-encoding those dicts
        # removed py/object, but retained dtype and values for NumPy arrays.
        if ('dtype' in value and 'values' in value
                and set(value) <= {'dtype', 'values', 'py/object'}):
            return np.asarray(value['values'], dtype=value['dtype'])
        return {k: restore_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [restore_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(restore_value(v) for v in value)
    return value


def normalize_config_values(config):
    """Preserve numeric coefficients and recover a lost tuple from its saved alias."""
    for name, value in list(vars(config).items()):
        setattr(config, name, restore_value(value))
    params = getattr(config, 'longitudinal_params', None)
    if params is None or (isinstance(params, dict) and not params):
        # The native TF++ checkpoint stores the identical seven coefficients
        # under both names; the NumPy alias survives the old JSON round trip.
        params = getattr(config, 'longitudinal_linear_regression_params', None)
    try:
        coefficients = np.asarray(params, dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError('Invalid saved longitudinal controller coefficients') from error
    if coefficients.shape != (7,) or not np.isfinite(coefficients).all():
        raise ValueError('The longitudinal controller requires seven finite saved coefficients')
    config.longitudinal_params = tuple(float(v) for v in coefficients)
    return config
