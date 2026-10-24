"""Numerical and training-path checks for the isolated TIE CR v2 head."""

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
from fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_v2 import TrajectoryHead


ROOT = Path(__file__).resolve().parents[1]


class TIECRV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = SimpleNamespace(
            trajectory_generator="residual_flow", diffusion_speed=False, speed_holder=0,
            step_num=20, flow_train_t_max=0.95, flow_min_remain=0.05,
            flow_inference_steps=20, flow_contract_enable=True,
            flow_contract_method="endpoint_jacobian", flow_contract_fd_epsilon=0.1,
            flow_contract_fd_chunk=4, flow_contract_interval=1,
            flow_contract_warmup_steps=0, flow_contract_fraction=1.0,
            flow_contract_weight=0.001, flow_contract_rho=0.98,
            flow_contract_pred_mode_fraction=1.0,
            flow_contract_on_policy_fraction=1.0, flow_recovery_weight=0.05,
            flow_contract_lat_min_m=0.02, flow_contract_lat_max_m=0.08,
            flow_contract_lon_max_m=0.03, flow_contract_step_bank=(4, 8, 20),
            tf_d_model=256, tf_d_ffn=128, tf_num_head=8, tf_dropout=0.0,
            lidar_max_x=32, lidar_max_y=32,
        )
        anchors = str(ROOT / "anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy")
        cls.head = TrajectoryHead(10, 128, 256, anchors, cls.config)
        cls.base = BaseHead(10, 128, 256, anchors, cls.config)

    def setUp(self):
        torch.manual_seed(17)
        self.head.config = copy.deepcopy(self.config)
        self.head._flow_contract_calls = 0
        self.head.zero_grad(set_to_none=True)

    def test_checkpoint_parameter_interface_is_unchanged(self):
        self.assertEqual(set(self.head.state_dict()), set(self.base.state_dict()))
        for key, value in self.head.state_dict().items():
            self.assertEqual(value.shape, self.base.state_dict()[key].shape)

    def test_saved_config_selects_v2_in_full_garage_model(self):
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
        config.flow_contract_method = "endpoint_jacobian"
        config.fmsr_trajectory_head = "tie_cr_v2"
        with tempfile.TemporaryDirectory() as directory:
            config.fmsr_anchor_path = directory + "/anchors.npy"
            anchors = np.zeros((60, 11, 2), dtype=np.float32)
            anchors[:, :10, 0] = 2.5 + np.arange(10)
            np.save(config.fmsr_anchor_path, anchors)
            create = timm.create_model

            def offline_create(*args, **kwargs):
                kwargs["pretrained"] = False
                return create(*args, **kwargs)

            with patch("timm.create_model", offline_create):
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
                result = model(**inputs)
            self.assertIn("flow_recovery_loss", result["metrics"])
            total = sum(result["losses"].values())
            self.assertTrue(torch.isfinite(total))
            total.backward()
            grad = model.fmsr_head.diff_decoder.layers[-1].task_decoder.plan_reg_branch[-1].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())

    def test_spectral_hinge_and_gradient_match_linear_map(self):
        gain = nn.Parameter(torch.tensor(1.25))
        state = torch.randn(1, 10, 2)
        anchors = state.clone()

        def linear_endpoint(x, *_args):
            return x * gain

        with patch.object(self.head, "predict_flow_endpoint", side_effect=linear_endpoint):
            loss, sigma = self.head._spectral_contract_at_state(
                state, torch.tensor([0.5]), anchors,
                torch.zeros(1, 1512, 2, 2), (2, 2), torch.zeros(1, 10, 256),
            )
            torch.testing.assert_close(sigma, gain.detach()[None], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(loss, (gain - 0.98).square(), rtol=1e-4, atol=1e-4)
            loss.backward()
        torch.testing.assert_close(gain.grad, 2 * (gain.detach() - 0.98), rtol=1e-4, atol=1e-4)

    def test_predicted_mode_actual_rollout_and_recovery_gradient(self):
        gain = nn.Parameter(torch.tensor(1.1))
        x_t = torch.zeros(2, 2, 10, 2)
        anchors = torch.zeros_like(x_t)
        anchors[:, 1] = 1.0
        x1_pred = x_t + 2.0
        captured = {}

        def spectral(state, time, anchor, *_args):
            captured.update(state=state.detach().clone(), time=time.detach().clone(),
                            anchor=anchor.detach().clone())
            return gain * 0.0, torch.ones(2)

        def endpoint(state, *_args):
            return state * gain

        with patch.object(self.head, "_spectral_contract_at_state", side_effect=spectral), \
             patch('torch.randint', return_value=torch.tensor([11, 11])), \
             patch.object(self.head, "predict_flow_endpoint", side_effect=endpoint):
            _, _, recovery, pred_fraction, shifted_fraction = self.head._contractive_recovery_v2(
                x_t, torch.full((2,), 0.5), anchors, x1_pred,
                torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long),
                torch.zeros(2, 10, 2), torch.zeros(2, 1512, 2, 2), (2, 2),
                torch.zeros(2, 10, 256),
            )
            recovery.backward()
        self.assertEqual(pred_fraction.item(), 1.0)
        self.assertEqual(shifted_fraction.item(), 1.0)
        torch.testing.assert_close(captured["anchor"], torch.ones_like(captured["anchor"]))
        torch.testing.assert_close(captured["time"], torch.full((2,), 0.55))
        factor = 1.0
        for index in range(11):
            factor *= 1.0 + 0.05 * 0.1 / (1.0 - index * 0.05)
        torch.testing.assert_close(captured['state'], torch.full_like(captured['state'], factor))
        self.assertGreater(recovery.item(), 0)
        self.assertGreater(gain.grad.abs().item(), 0)

    def test_two_sample_cr_batch_covers_both_modes_and_rollout_states(self):
        self.head.config.flow_contract_pred_mode_fraction = 0.5
        self.head.config.flow_contract_on_policy_fraction = 0.5
        x_t = torch.zeros(2, 2, 10, 2)
        with patch.object(self.head, "_spectral_contract_at_state",
                          return_value=(torch.zeros(()), torch.ones(2))), \
             patch.object(self.head, "predict_flow_endpoint", side_effect=lambda x, *_: x):
            *_, pred_fraction, shifted_fraction = self.head._contractive_recovery_v2(
                x_t, torch.full((2,), 0.5), x_t, x_t + 0.1,
                torch.zeros(2, dtype=torch.long), torch.ones(2, dtype=torch.long),
                torch.zeros(2, 10, 2), torch.zeros(2, 1512, 2, 2), (2, 2),
                torch.zeros(2, 10, 256),
            )
        self.assertEqual(pred_fraction.item(), 0.5)
        self.assertEqual(shifted_fraction.item(), 0.5)

    def test_training_loss_combines_fm_recovery_and_cr(self):
        self.head.train()
        gain = nn.Parameter(torch.tensor(1.2))
        anchors = self.head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + 0.2
        logits = torch.zeros(2, anchors.shape[1])
        logits[:, 4] = 1.0

        def endpoint(x, *_args):
            return x * gain

        with patch.object(self.head, "predict_flow_endpoint", side_effect=endpoint), \
             patch.object(self.head, "_classify_flow_anchors", return_value=[logits]):
            output = self.head.forward_train_flow(
                torch.zeros(2, 10, 256), torch.zeros(2, 1512, 2, 2),
                (2, 2), gt_checkpoints=gt,
            )
            self.assertEqual(output["flow_contract_active"].item(), 1.0)
            expected = (output["flow_fm_loss"] +
                        self.config.flow_contract_weight * output["flow_contract_loss"] +
                        self.config.flow_recovery_weight * output["flow_recovery_loss"])
            torch.testing.assert_close(output["flow_loss"].detach(), expected)
            output["flow_loss"].backward()
        self.assertTrue(torch.isfinite(gain.grad))
        self.assertGreater(gain.grad.abs().item(), 0)


if __name__ == "__main__":
    unittest.main()
