"""The expert must find its bundled speed limits from any working directory."""

import importlib.util
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'privileged_speed_limits_test',
    ROOT / '3rd_party/carla_garage/team_code/privileged_route_planner.py')
planner_module = importlib.util.module_from_spec(spec)
with patch.dict('sys.modules', {'carla': types.SimpleNamespace(),
                              'agents.navigation.local_planner': types.SimpleNamespace(RoadOption=None)}):
    spec.loader.exec_module(planner_module)


class PrivilegedSpeedLimitTests(unittest.TestCase):
    def test_module_relative_asset_and_speed_conversion_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            team_code = root / 'garage' / 'team_code'
            (team_code / 'speed_limits').mkdir(parents=True)
            np.save(team_code / 'speed_limits/Town12_speed_limits.npy', {
                'locations': [[0, 0, 0], [100, 0, 0]], 'speed_limits': [36, 72]})
            planner = planner_module.PrivilegedRoutePlanner.__new__(planner_module.PrivilegedRoutePlanner)
            planner.route_points = np.array([[0, 0, 0], [1, 0, 0], [100, 0, 0], [101, 0, 0]])
            planner.speed_limit_waypoints_spacing_check = 2
            original_cwd = Path.cwd()
            try:
                os.chdir(root)
                with patch.object(planner_module, '__file__', str(team_code / 'privileged_route_planner.py')):
                    planner.compute_speed_limits(types.SimpleNamespace(name='Carla/Maps/Town12/Town12'))
            finally:
                os.chdir(original_cwd)
            np.testing.assert_allclose(planner.speed_limits, [10, 10, 20, 20])


if __name__ == '__main__':
    unittest.main()
