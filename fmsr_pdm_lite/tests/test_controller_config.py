"""Regression tests for positive-speed control with serialized TF++ coefficients."""
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import pytest

from fmsrpdm.config_values import normalize_config_values
from fmsrpdm.configuration import GlobalConfig
from nav_planner import get_throttle


def test_original_typed_json_restores_tuple_and_numpy_values():
    config = GlobalConfig()
    expected = config.longitudinal_params
    config.longitudinal_params = {'py/tuple': list(expected)}
    config.brake_values = {'py/object': 'numpy.ndarray', 'dtype': 'float64',
                           'values': [1., 2., 3.]}
    normalize_config_values(config)
    assert config.longitudinal_params == expected
    np.testing.assert_array_equal(config.brake_values, np.array([1., 2., 3.]))


def test_saved_empty_tuple_recovers_identical_coefficients_and_cruise_control():
    config = GlobalConfig()
    expected = config.longitudinal_params
    original_throttle = get_throttle(config, False, 5., 5.)
    config.longitudinal_params = {}
    config.longitudinal_linear_regression_params = {'dtype': 'float64', 'values': list(expected)}
    with pytest.raises(KeyError):
        get_throttle(config, False, 5., 5.)
    normalize_config_values(config)
    assert config.longitudinal_params == expected
    assert get_throttle(config, False, 5., 5.) == original_throttle
    assert get_throttle(config, False, 8., 0.) == (1., False)
    assert get_throttle(config, False, 5., 5.3) == (0., True)
    assert get_throttle(config, False, 0., 2.) == (0., True)


def test_missing_saved_coefficients_are_rejected_without_inventing_defaults():
    with pytest.raises(ValueError, match='seven finite'):
        normalize_config_values(SimpleNamespace(longitudinal_params={}))
