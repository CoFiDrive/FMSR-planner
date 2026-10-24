import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('native_speed_test', ROOT/'3rd_party/carla_garage/team_code/fmsr_interface.py')
interface = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interface)


class FrozenSpeedTests(unittest.TestCase):
    def test_old_configs_keep_planner_speed_source(self):
        cfg = SimpleNamespace()
        interface.restore_trained_interface(cfg, SimpleNamespace())
        self.assertEqual(cfg.fmsr_speed_source, 'planner')
        interface.restore_trained_interface(cfg, SimpleNamespace(fmsr_speed_source='tfpp_uncertainty_v1'))
        self.assertEqual(cfg.fmsr_speed_source, 'tfpp_uncertainty_v1')

    def test_decode_preserves_expected_speed_and_stop_probability_rule(self):
        cfg = SimpleNamespace(target_speeds=[0., 4., 8.], brake_uncertainty_threshold=.6, fmsr_max_speed=30.)
        p = torch.tensor([[.1,.7,.2], [.8,.1,.1]])
        speed, stop = interface.decode_frozen_tfpp_speed(p.log(), cfg)
        torch.testing.assert_close(speed, torch.tensor([4.4,0.]))
        torch.testing.assert_close(stop, p[:,0])

    def test_unknown_sources_and_incompatible_logits_fail(self):
        with self.assertRaises(ValueError):
            interface.planner_speed_source(SimpleNamespace(fmsr_speed_source='unknown'))
        cfg = SimpleNamespace(target_speeds=[0.,4.],brake_uncertainty_threshold=.6)
        with self.assertRaises(ValueError):
            interface.decode_frozen_tfpp_speed(torch.zeros(2,3), cfg)

    def test_curvature_cap_straight_and_circle_with_rotation(self):
        cfg = SimpleNamespace(fmsr_max_speed=30., fmsr_curve_max_lateral_accel=3.)
        straight = torch.tensor([[[2.,0.],[3.,0.],[4.,0.],[5.,0.]]])
        cap, curvature = interface.path_curvature_speed_cap(straight, cfg)
        torch.testing.assert_close(cap, torch.tensor([30.]))
        theta=torch.linspace(0., .5, 6)
        circle=torch.stack((10*theta.cos(),10*theta.sin()),-1)[None]
        cap, curvature = interface.path_curvature_speed_cap(circle, cfg)
        torch.testing.assert_close(curvature, torch.tensor([.1]),atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(cap, torch.tensor([30.**.5]),atol=1e-4,rtol=1e-4)
        rotated = torch.stack((-circle[...,1],circle[...,0]),-1) + torch.tensor([4.,-3.])
        rotated_cap,_ = interface.path_curvature_speed_cap(rotated,cfg)
        torch.testing.assert_close(rotated_cap,cap,atol=1e-4,rtol=1e-4)

    def test_curvature_cap_rejects_invalid_physical_limit(self):
        with self.assertRaises(ValueError):
            interface.path_curvature_speed_cap(torch.ones(1,3,2),SimpleNamespace(fmsr_curve_max_lateral_accel=0.))


if __name__=='__main__':
    unittest.main()
