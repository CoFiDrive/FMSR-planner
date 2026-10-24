"""Numerical and training-path checks for finite-difference endpoint CR."""

import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm import TrajectoryHead as BaseHead
from fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_fd import TrajectoryHead


ROOT = Path(__file__).resolve().parents[1]


class TIECRFDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = SimpleNamespace(
            trajectory_generator="residual_flow", diffusion_speed=False, speed_holder=0,
            step_num=20, flow_train_t_max=0.95, flow_min_remain=0.05,
            flow_inference_steps=20, flow_contract_enable=True,
            flow_contract_method="endpoint_fd", flow_contract_fd_epsilon=0.1,
            flow_contract_fd_chunk=4, flow_contract_interval=1,
            flow_contract_warmup_steps=0, flow_contract_fraction=1.0,
            flow_contract_weight=0.001, flow_contract_rho=0.98,
            flow_contract_pred_mode_fraction=0.0,
            flow_contract_on_policy_fraction=0.0,
            flow_contract_lat_min_m=0.02, flow_contract_lat_max_m=0.08,
            flow_contract_lon_max_m=0.03, flow_contract_yaw_max_deg=1.0,
            flow_contract_step_bank=(4, 8, 20),
            tf_d_model=256, tf_d_ffn=128, tf_num_head=8, tf_dropout=0.0,
            lidar_max_x=32, lidar_max_y=32,
        )
        cls.anchor_path = str(ROOT / "anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy")

    def setUp(self):
        torch.manual_seed(17)
        self.head = TrajectoryHead(10, 128, 256, self.anchor_path, self.config)

    def test_checkpoint_parameter_interface_is_unchanged(self):
        base = BaseHead(10, 128, 256, self.anchor_path, self.config)
        self.assertEqual(set(self.head.state_dict()), set(base.state_dict()))
        for key, value in self.head.state_dict().items():
            self.assertEqual(value.shape, base.state_dict()[key].shape)

    def test_recovery_posttraining_loads_fd_weights_without_cr(self):
        config = copy.deepcopy(self.config)
        config.flow_contract_enable = False
        config.flow_contract_method = "euler_ratio"
        head = TrajectoryHead(10, 128, 256, self.anchor_path, config).train()
        head.load_state_dict(self.head.state_dict(), strict=True)
        anchors = head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + 0.2
        logits = torch.zeros(2, anchors.shape[1])
        gain = nn.Parameter(torch.tensor(1.1))
        with patch.object(head, "predict_flow_endpoint", side_effect=lambda x, *_: x * gain), \
             patch.object(head, "_classify_flow_anchors", return_value=[logits]), \
             patch.object(head, "contractive_recovery_loss") as cr:
            result = head.forward_train_flow(
                torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
                (2, 2), gt_checkpoints=gt,
            )
            result["flow_loss"].backward()
            cr.assert_not_called()
        torch.testing.assert_close(result["flow_loss"].detach(), result["flow_fm_loss"])
        self.assertEqual(result["flow_contract_active"].item(), 0.0)
        self.assertTrue(torch.isfinite(gain.grad))
        self.assertGreater(gain.grad.abs().item(), 0.0)

    def test_finite_endpoint_gain_and_gradient_match_linear_map(self):
        gain = nn.Parameter(torch.tensor(1.25))
        states = torch.randn(2, 1, 10, 2)
        anchors = states.clone()
        delta = torch.zeros(2, 10, 2)
        delta[:, :, 1] = 0.05

        with patch.object(self.head, "_physical_state_perturbation", return_value=delta), \
             patch.object(self.head, "predict_flow_endpoint", side_effect=lambda x, *_: x * gain):
            loss, ratios = self.head.contractive_recovery_loss(
                states, torch.full((2,), 0.5), anchors, states,
                torch.zeros(2, dtype=torch.long), torch.zeros(2, 1512, 2, 2),
                (2, 2), torch.zeros(2, 10, 256),
                pred_mode_idx=torch.zeros(2, dtype=torch.long),
            )
            torch.testing.assert_close(ratios, torch.full((2,), gain.detach()),
                                       rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(loss, (gain - 0.98).square(),
                                       rtol=1e-4, atol=1e-4)
            loss.backward()
        torch.testing.assert_close(gain.grad, 2 * (gain.detach() - 0.98),
                                   rtol=1e-4, atol=1e-4)

    def test_below_margin_fd_has_zero_loss_and_gradient(self):
        gain = nn.Parameter(torch.tensor(0.90))
        states = torch.randn(2, 1, 10, 2)
        delta = torch.zeros(2, 10, 2)
        delta[:, :, 1] = 0.05
        with patch.object(self.head, "_physical_state_perturbation", return_value=delta), \
             patch.object(self.head, "predict_flow_endpoint", side_effect=lambda x, *_: x * gain):
            loss, _ = self.head.contractive_recovery_loss(
                states, torch.full((2,), 0.5), states, states,
                torch.zeros(2, dtype=torch.long), torch.zeros(2, 1512, 2, 2),
                (2, 2), torch.zeros(2, 10, 256),
                pred_mode_idx=torch.zeros(2, dtype=torch.long),
            )
            loss.backward()
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(gain.grad.item(), 0.0)

    def test_physical_perturbation_keeps_speed_token_fixed(self):
        config = copy.deepcopy(self.config)
        config.diffusion_speed = True
        config.speed_holder = 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.npy"
            anchors = np.zeros((60, 11, 2), dtype=np.float32)
            anchors[:, :10, 0] = 2.5 + np.arange(10)
            np.save(path, anchors)
            head = TrajectoryHead(10, 128, 256, str(path), config)
            state = torch.from_numpy(anchors[:2, 0:11]).clone()
            state[:, -1, 0] = 3.0
            delta = head._physical_state_perturbation(state)
            torch.testing.assert_close(delta[:, -1], torch.zeros_like(delta[:, -1]))
            self.assertTrue((delta[:, :10].abs().sum(dim=(1, 2)) > 0).all())

    def test_physical_perturbation_caps_every_waypoint(self):
        config = copy.deepcopy(self.config)
        config.flow_contract_max_perturb_m = 0.15
        head = TrajectoryHead(10, 128, 256, self.anchor_path, config)
        state = torch.zeros(4, 10, 2)
        state[:, :, 0] = torch.arange(10).float() * 10.0
        delta = head._physical_state_perturbation(state)
        self.assertLessEqual(torch.linalg.vector_norm(delta, dim=-1).max().item(), 0.150001)
        self.assertTrue((delta.abs().sum(dim=(1, 2)) > 0).all())

    def test_mode_mix_and_actual_rollout_use_matching_conditions(self):
        self.head.config.flow_contract_pred_mode_fraction = 0.5
        self.head.config.flow_contract_on_policy_fraction = 0.5
        states = torch.zeros(2, 2, 10, 2)
        anchors = states.clone()
        anchors[:, 1] = 1.0
        delta = torch.zeros(2, 10, 2)
        delta[:, :, 1] = 0.05
        captured = {}

        def endpoint(x, time, anchor, *_args):
            captured.update(states=x.detach().clone(), time=time.detach().clone(),
                            anchors=anchor.detach().clone())
            return x * 1.1 + 0.02

        with patch.object(self.head, "_physical_state_perturbation", return_value=delta), \
             patch('torch.randint', return_value=torch.tensor([11])), \
             patch.object(self.head, "predict_flow_endpoint", side_effect=endpoint):
            self.head.contractive_recovery_loss(
                states, torch.full((2,), 0.5), anchors, states + 2.0,
                torch.zeros(2, dtype=torch.long), torch.zeros(2, 1512, 2, 2),
                (2, 2), torch.zeros(2, 10, 256),
                pred_mode_idx=torch.ones(2, dtype=torch.long),
            )
        torch.testing.assert_close(captured["time"].sort().values,
                                   torch.tensor([0.5, 0.5, 0.55, 0.55]))
        self.assertEqual(sorted(captured["anchors"][:, 0, 0, 0].tolist()), [0.0, 0.0, 1.0, 1.0])
        self.assertTrue((captured["states"][::2].abs().sum(dim=(1, 2, 3)) > 0).any())
        self.assertTrue((captured["states"][::2].abs().sum(dim=(1, 2, 3)) == 0).any())

    def test_real_training_forward_combines_fm_and_fd_cr(self):
        self.head.train()
        self.head.config.flow_contract_interval = 4
        anchors = self.head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + 0.2
        result = self.head.forward_train_flow(
            torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
            (2, 2), gt_checkpoints=gt,
        )
        self.assertEqual(result["flow_contract_active"].item(), 1.0)
        self.assertTrue(torch.isfinite(result["flow_contract_loss"]))
        self.assertTrue(torch.isfinite(result["flow_contract_ratio"]))
        expected = (result["flow_fm_loss"] +
                    4 * self.config.flow_contract_weight * result["flow_contract_loss"])
        torch.testing.assert_close(result["flow_loss"].detach(), expected)
        result["flow_loss"].backward()
        grad = self.head.diff_decoder.layers[-1].task_decoder.plan_reg_branch[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())

    def test_delayed_linear_ramp_skips_cr_then_reaches_full_weight(self):
        config = copy.deepcopy(self.config)
        config.flow_contract_delay_steps = 2
        config.flow_contract_ramp_steps = 2
        head = TrajectoryHead(10, 128, 256, self.anchor_path, config).train()
        anchors = head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + 0.2
        logits = torch.zeros(2, anchors.shape[1])
        logits[:, 4] = 1.0
        with patch.object(head, "predict_flow_endpoint", side_effect=lambda x, *_: x), \
             patch.object(head, "_classify_flow_anchors", return_value=[logits]), \
             patch.object(head, "contractive_recovery_loss",
                          return_value=(torch.tensor(1.0), torch.ones(2))) as cr:
            for step, ramp in enumerate((0.0, 0.0, 0.5, 1.0), start=1):
                result = head.forward_train_flow(
                    torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
                    (2, 2), gt_checkpoints=gt,
                )
                self.assertEqual(result["flow_contract_active"].item(), float(step > 2))
                self.assertEqual(result["flow_contract_ramp_fraction"].item(), ramp)
                torch.testing.assert_close(
                    result["flow_loss"].detach(),
                    result["flow_fm_loss"] + config.flow_contract_weight * ramp,
                )
            self.assertEqual(cr.call_count, 2)

        # Sparse CR updates restart their interval at the first post-delay step.
        sparse_config = copy.deepcopy(config)
        sparse_config.flow_contract_interval = 4
        sparse_head = TrajectoryHead(10, 128, 256, self.anchor_path, sparse_config).train()
        with patch.object(sparse_head, "predict_flow_endpoint", side_effect=lambda x, *_: x), \
             patch.object(sparse_head, "_classify_flow_anchors", return_value=[logits]), \
             patch.object(sparse_head, "contractive_recovery_loss",
                          return_value=(torch.tensor(1.0), torch.ones(2))) as sparse_cr:
            for _ in range(3):
                result = sparse_head.forward_train_flow(
                    torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
                    (2, 2), gt_checkpoints=gt,
                )
            self.assertEqual(result["flow_contract_active"].item(), 1.0)
            self.assertEqual(sparse_cr.call_count, 1)

    def test_fd_success_uses_kappa_below_one(self):
        self.head.train()
        anchors = self.head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + 0.2
        logits = torch.zeros(2, anchors.shape[1])
        logits[:, 4] = 1.0
        with patch.object(self.head, "predict_flow_endpoint", side_effect=lambda x, *_: 0.99 * x), \
             patch.object(self.head, "_classify_flow_anchors", return_value=[logits]):
            result = self.head.forward_train_flow(
                torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
                (2, 2), gt_checkpoints=gt,
            )
        self.assertGreater(result["flow_contract_ratio"].item(), 0.98)
        self.assertLess(result["flow_contract_ratio"].item(), 1.0)
        self.assertEqual(result["flow_contract_success"].item(), 0.0)
        self.assertGreater(result["flow_contract_loss"].item(), 0.0)

    def test_tau_kappa_boundaries_and_validation(self):
        config = copy.deepcopy(self.config)
        config.flow_contract_tau_kappa = (1.30, 1.20, 1.08, 1.00)
        head = TrajectoryHead(10, 128, 256, self.anchor_path, config)
        tau = torch.tensor([0.0, 0.249, 0.25, 0.499, 0.5, 0.749, 0.75, 0.95])
        torch.testing.assert_close(
            head._contract_threshold(tau),
            torch.tensor([1.30, 1.30, 1.20, 1.20, 1.08, 1.08, 1.00, 1.00]),
        )
        config.flow_contract_tau_kappa = (1.30, 1.20, 1.25, 1.00)
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            TrajectoryHead(10, 128, 256, self.anchor_path, config)

    def test_tau_schedule_only_penalizes_gain_above_its_own_bin(self):
        config = copy.deepcopy(self.config)
        config.flow_contract_tau_kappa = (1.30, 1.20, 1.08, 1.00)
        head = TrajectoryHead(10, 128, 256, self.anchor_path, config).train()
        anchors = head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + 0.2
        tau = torch.tensor([0.1, 0.8])
        logits = torch.zeros(2, anchors.shape[1])
        logits[:, 4] = 1.0
        delta = torch.zeros(2, 10, 2)
        delta[:, :, 1] = 0.05
        with patch.object(head, "_flow_training_state", return_value=(anchors, tau)), \
             patch.object(head, "_classify_flow_anchors", return_value=[logits]), \
             patch.object(head, "_physical_state_perturbation", return_value=delta), \
             patch.object(head, "predict_flow_endpoint", side_effect=lambda x, *_: 1.15 * x):
            result = head.forward_train_flow(
                torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
                (2, 2), gt_checkpoints=gt,
            )
        torch.testing.assert_close(result["flow_contract_loss"], torch.tensor(0.15 ** 2 / 2),
                                   rtol=1e-3, atol=1e-4)
        self.assertEqual(result["flow_contract_success"].item(), 0.5)
        torch.testing.assert_close(result["flow_contract_kappa_mean"], torch.tensor(1.15))
        self.assertEqual(result["flow_contract_count_tau0"].item(), 1.0)
        self.assertEqual(result["flow_contract_count_tau3"].item(), 1.0)
        self.assertNotIn("flow_contract_gain_tau1", result)

    def test_garage_model_selects_fd_head(self):
        sys.path[:0] = [str(ROOT / "3rd_party/carla_garage/team_code"),
                        str(ROOT / "3rd_party/CARLA_0915/PythonAPI/carla")]
        import timm
        from config import GlobalConfig
        from model import LidarCenterNet

        config = GlobalConfig()
        config.use_fmsr = True
        config.use_wp_gru = False
        config.use_semantic = False
        config.use_depth = False
        config.use_bev_semantic = False
        config.detect_boxes = False
        config.use_controller_input_prediction = True
        config.tf_dropout = 0.0
        config.trajectory_generator = "residual_flow"
        config.flow_contract_enable = True
        config.flow_contract_method = "endpoint_fd"
        config.flow_contract_interval = 4
        config.fmsr_trajectory_head = "tie_cr_fd"
        with tempfile.TemporaryDirectory() as directory:
            config.fmsr_anchor_path = directory + "/anchors.npy"
            anchors = np.zeros((60, 11, 2), dtype=np.float32)
            anchors[:, :10, 0] = 2.5 + np.arange(10)
            np.save(config.fmsr_anchor_path, anchors)
            create = timm.create_model
            def offline_create(*args, **kwargs):
                kwargs["pretrained"] = False
                return create(*args, **kwargs)

            with patch("timm.create_model", side_effect=offline_create):
                model = LidarCenterNet(config)
            self.assertIsInstance(model.fmsr_head, TrajectoryHead)
            model.freeze_for_fmsr()
            model.eval()
            model.fmsr_head.train()
            model.fmsr_tp_encoder.train()
            inputs = dict(
                rgb=torch.zeros(1, 3, config.camera_height, config.camera_width),
                lidar_bev=torch.zeros(1, 2 if config.use_ground_plane else 1, 256, 256),
                ego_vel=torch.ones(1, 1),
                command=torch.tensor([[0., 0., 0., 1., 0., 0.]]),
                target_point=torch.tensor([[10., 5.]]),
                fmsr_route=torch.from_numpy(anchors[:1, :10]).clone(),
                fmsr_target_speed=torch.tensor([1.5]),
                return_fmsr_loss=True,
            )
            with torch.no_grad():
                features = model.backbone(inputs["rgb"], inputs["lidar_bev"])
            with patch.object(model.backbone, "forward", return_value=features):
                first = model(**inputs)
                second = model(**inputs)
            self.assertIn("flow_contract_ratio", first["metrics"])
            self.assertIn("flow_contract_kappa_mean", first["metrics"])
            self.assertNotIn("flow_contract_sigma_max", first["metrics"])
            self.assertNotIn("flow_contract_ratio", second["metrics"])


if __name__ == "__main__":
    unittest.main()
