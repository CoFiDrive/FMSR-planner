"""Numerical and integration checks for the endpoint-parameterized flow bridge."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm import TrajectoryHead
from fmsr.tfv6.diffusion_modules.multimodal_loss_ddbm import LossComputer
from fmsr.tfv6.planning_decoder_fmsr import PlanningDecoderDDBM
from fmsr.training.config_training import TrainingConfig


ROOT = Path(__file__).resolve().parents[1]


class TrainingConfigCompatibilityTests(unittest.TestCase):
    def make_config(self, overrides):
        with patch.dict("os.environ", {"FMSR_TRAINING_CONFIG": ""}), \
             patch("sys.argv", ["test_residual_flow"]):
            return TrainingConfig(overrides, raise_error_on_missing_key=True)

    def make_head(self, config):
        return TrajectoryHead(
            10, 128, 256,
            str(ROOT / "anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"),
            config,
        )

    def test_legacy_training_config_initializes_head(self):
        # Older checkpoint configs omit the newer recovery settings.
        config = self.make_config({"trajectory_generator": "residual_flow"})
        head = self.make_head(config)
        self.assertEqual(head.config.flow_contract_method, "euler_ratio")
        self.assertEqual(head.config.flow_contract_fd_epsilon, 0.1)
        self.assertEqual(head.config.flow_contract_fd_chunk, 4)
        self.assertEqual(head.config.flow_contract_interval, 1)
        self.assertEqual(head.config.flow_contract_warmup_steps, 0)

    def test_saved_recovery_settings_reach_head(self):
        overrides = {
            "flow_contract_enable": True,
            "flow_contract_method": "endpoint_fd",
            "flow_contract_fd_epsilon": 0.2,
            "flow_contract_fd_chunk": 2,
            "flow_contract_interval": 3,
            "flow_contract_warmup_steps": 100,
        }
        head = self.make_head(self.make_config(overrides.copy()))
        for name, expected in overrides.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(head.config, name), expected)


class ResidualFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = SimpleNamespace(
            trajectory_generator="residual_flow", diffusion_speed=False, speed_holder=0,
            step_num=20, flow_train_t_max=0.95, flow_min_remain=0.05,
            flow_inference_steps=20, flow_contract_enable=False,
            flow_contract_method='euler_ratio', flow_contract_fd_epsilon=0.1,
            flow_contract_fd_chunk=4, flow_contract_interval=1,
            flow_contract_warmup_steps=0,
            flow_contract_fraction=0.125, flow_contract_weight=0.01,
            flow_contract_rho=0.98, flow_contract_lat_min_m=0.02,
            flow_contract_lat_max_m=0.08, flow_contract_lon_max_m=0.03,
            flow_contract_step_bank=(4, 8, 20), tf_d_model=256, tf_d_ffn=128,
            tf_num_head=8, tf_dropout=0.0, lidar_max_x=32, lidar_max_y=32,
        )
        cls.head = TrajectoryHead(
            10, 128, 256, str(ROOT / "anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"),
            cls.config,
        )

    def setUp(self):
        torch.manual_seed(42)
        self.head.config = copy.deepcopy(self.config)
        self.head.trajectory_generator = "residual_flow"
        self.head.eval()
        self.head.zero_grad(set_to_none=True)
        self.ego = torch.randn(2, 10, 256)
        self.bev = torch.randn(2, 1512, 4, 4)
        self.anchors = self.head.plan_anchor.detach()[None].expand(2, -1, -1, -1)
        self.gt = self.anchors[:, 5].clone() + 0.1
        self.scene = (self.bev, (4, 4), self.ego, None)

    def test_straight_path_and_velocity_identity(self):
        gt = self.gt[:, None].expand_as(self.anchors)
        xt, tau = self.head._flow_training_state(self.anchors, gt)
        self.assertEqual(xt.dtype, torch.float32)
        self.assertTrue(((0 <= tau) & (tau < 0.95)).all())
        velocity = (gt - xt) / (1 - tau[:, None, None, None])
        torch.testing.assert_close(velocity, gt - self.anchors)

    def test_gt_mode_supervision_and_predicted_mode_endpoint_interface(self):
        tau = torch.tensor([0.5, 0.75])
        gt = self.gt[:, None].expand_as(self.anchors)
        xt = self.anchors + tau[:, None, None, None] * (gt - self.anchors)
        # Select anchors intentionally different from the nearest-GT anchor.
        modes = torch.tensor([17, 39])
        logits = torch.full((2, 60), -10.0)
        logits[torch.arange(2), modes] = 10
        offsets = torch.arange(60).view(1, 60, 1, 1) * 0.01
        endpoint = (gt + offsets).clone().requires_grad_()
        with patch.object(self.head, "_flow_training_state", return_value=(xt, tau)), \
             patch.object(self.head, "predict_flow_endpoint", return_value=endpoint), \
             patch.object(self.head, "_classify_flow_anchors", return_value=[logits, logits]):
            output = self.head.forward_train(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt)
        gt_modes = self.head.nearest_anchor_idx(self.gt, self.anchors)
        self.assertTrue((gt_modes != modes).all())
        selected = endpoint[torch.arange(2), gt_modes]
        selected_xt = xt[torch.arange(2), gt_modes]
        remain = (1 - tau)[:, None, None]
        expected = F.l1_loss(
            (selected - selected_xt) / remain / self.head.std.squeeze(1),
            (self.gt - selected_xt) / remain / self.head.std.squeeze(1),
        )
        torch.testing.assert_close(output["flow_loss"], expected)
        torch.testing.assert_close(output["best_reg"], endpoint[torch.arange(2), modes])
        torch.testing.assert_close(output["flow_endpoint_l1"], F.l1_loss(endpoint[torch.arange(2), modes], self.gt))
        self.assertEqual(output["best_reg"].shape, (2, 10, 2))
        output["flow_loss"].backward()
        nonzero_modes = endpoint.grad.abs().sum(dim=(-1, -2)) > 0
        self.assertEqual(nonzero_modes.sum().item(), 2)
        self.assertTrue(nonzero_modes[torch.arange(2), gt_modes].all())

    def test_oracle_euler_reaches_endpoint_and_preserves_last_y(self):
        endpoint = self.anchors + torch.tensor([0.125, 1.75])
        seen = []

        def oracle(x, tau, *args):
            seen.append((x.clone(), tau.clone()))
            self.assertEqual(x.dtype, torch.float32)
            return endpoint

        with patch.object(self.head, "predict_flow_endpoint", side_effect=oracle):
            result = self.head.flow_rollout(self.anchors, self.scene)
        self.assertEqual(len(seen), 20)
        torch.testing.assert_close(seen[0][0], self.anchors)
        self.assertTrue(all((tau < 1).all() for _, tau in seen))
        torch.testing.assert_close(seen[-1][1], torch.full((2,), 0.95))
        torch.testing.assert_close(result, endpoint)
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue((result[..., -1, 1] != 0).any())

    def test_fp32_state_with_bfloat16_endpoints(self):
        endpoint = (self.anchors + 0.1).bfloat16()
        with torch.autocast("cpu", dtype=torch.bfloat16), \
             patch.object(self.head, "predict_flow_endpoint", return_value=endpoint):
            result = self.head.flow_rollout(self.anchors, self.scene)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, endpoint.float())

    def test_time_embedding_reversed_and_classifier_parity(self):
        tau = torch.tensor([0.0, 0.95])
        with torch.no_grad(), patch.object(self.head.time_mlp, "forward", wraps=self.head.time_mlp.forward) as time_mlp:
            result = self.head.predict_flow_endpoint(self.anchors, tau, self.anchors, *self.scene[:2], self.ego)
        torch.testing.assert_close(time_mlp.call_args.args[0], torch.tensor([1000., 50.]))
        self.assertEqual(result.dtype, torch.float32)
        from fmsr.tfv6.diffusion_modules.blocks import gen_sineembed_for_position
        with torch.no_grad():
            emb = gen_sineembed_for_position(self.anchors, hidden_dim=64).flatten(-2)
            features = self.head.plan_anchor_encoder(emb).view(2, 60, -1)
            original = self.head.diff_decoder_cls(features, self.anchors, self.bev, (4, 4), self.ego)
            flow = self.head._classify_flow_anchors(self.anchors, self.bev, (4, 4), self.ego)
        for actual, expected in zip(flow, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_real_network_forward_backward_and_flow_dispatch(self):
        self.head.train()
        with patch.object(self.head.diffusion_scheduler, "add_noise", side_effect=AssertionError("DDBM called")):
            output = self.head(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt)
        loss_fn = LossComputer(self.config)
        cls_loss = sum(loss_fn(None, c, {"trajectory": self.gt}, self.anchors, None,
                               compute_regression=False)[0] for c in output["poses_cls_list"])
        total = output["flow_loss"] + cls_loss
        self.assertTrue(torch.isfinite(total))
        total.backward()
        grad = self.head.diff_decoder.layers[-1].task_decoder.plan_reg_branch[-1].weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)
        # Final-endpoint supervision must train the earlier refinement layer.
        first_grad = self.head.diff_decoder.layers[0].task_decoder.plan_reg_branch[-1].weight.grad
        self.assertIsNotNone(first_grad)
        self.assertTrue(torch.isfinite(first_grad).all())
        self.assertGreater(first_grad.abs().sum().item(), 0)
        self.assertIsNotNone(self.head.plan_anchor_encoder[-1].weight.grad)
        self.assertIsNotNone(self.head.diff_decoder_cls.layers[-1].task_decoder_cls.plan_cls_branch[-1].weight.grad)
        for p in self.head.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_training_classifier_parity_under_autocast(self):
        from fmsr.tfv6.diffusion_modules.blocks import gen_sineembed_for_position
        self.head.train()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            emb = gen_sineembed_for_position(self.anchors, hidden_dim=64).flatten(-2)
            features = self.head.plan_anchor_encoder(emb).view(2, 60, -1).float()
            torch.manual_seed(123)
            original = self.head.diff_decoder_cls(features, self.anchors, self.bev, (4, 4), self.ego)
            torch.manual_seed(123)
            flow = self.head._classify_flow_anchors(self.anchors, self.bev, (4, 4), self.ego)
        for actual, expected in zip(flow, original):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_all_modes_rollout_before_top1_selection(self):
        modes = torch.tensor([13, 37])
        logits = torch.zeros(2, 60)
        logits[torch.arange(2), modes] = 10
        with patch.object(self.head, "_classify_flow_anchors", return_value=[logits]), \
             patch.object(self.head, "predict_flow_endpoint", return_value=self.anchors + 0.1) as predict, \
             patch.object(self.head.diffusion_scheduler, "sample_step", side_effect=AssertionError("DDBM called")):
            output = self.head(self.ego, self.bev, (4, 4))
        self.assertEqual(predict.call_count, 20)
        for call in predict.call_args_list:
            self.assertEqual(call.args[0].shape, (2, 60, 10, 2))
        torch.testing.assert_close(output, (self.anchors + 0.1)[torch.arange(2), modes])
        self.assertFalse(output.requires_grad)

    def test_real_bfloat16_network_and_twenty_step_rollout(self):
        self.head.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = self.head(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt)
        self.assertEqual(output["flow_loss"].dtype, torch.float32)
        self.assertEqual(output["best_reg"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(output["flow_loss"]))
        output["flow_loss"].backward()
        for p in self.head.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())
        self.head.eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            result = self.head(self.ego, self.bev, (4, 4))
        self.assertEqual(result.shape, (2, 10, 2))
        self.assertEqual(result.dtype, torch.float32)
        self.assertTrue(torch.isfinite(result).all())

    def test_classifier_loss_unchanged_without_regression(self):
        logits = torch.randn(2, 60, requires_grad=True)
        fn = LossComputer(self.config)
        original_cls, original_reg = fn(None, logits, {"trajectory": self.gt}, self.anchors, self.gt + 0.2)
        flow_cls, unused_reg = fn(None, logits, {"trajectory": self.gt}, self.anchors, None, compute_regression=False)
        torch.testing.assert_close(original_cls, flow_cls, rtol=0, atol=0)
        self.assertGreater(original_reg.item(), 0)
        self.assertEqual(unused_reg.item(), 0)

    def test_clean_state_unchanged_when_contract_enabled(self):
        gt = self.gt[:, None].expand_as(self.anchors)
        torch.manual_seed(7)
        clean = self.head._flow_training_state(self.anchors, gt)
        self.head.config.flow_contract_enable = True
        torch.manual_seed(7)
        contract = self.head._flow_training_state(self.anchors, gt)
        for expected, actual in zip(clean, contract):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_endpoint_jacobian_contract_matches_linear_spectral_norm(self):
        # A diagonal endpoint map has known normalized spectral norm. The
        # extra PDM speed token must stay fixed and outside the trajectory J.
        self.head.config.flow_contract_method = 'endpoint_jacobian'
        self.head.config.flow_contract_fraction = 1.0
        self.head.config.flow_contract_fd_epsilon = 0.1
        self.head.config.flow_contract_fd_chunk = 4
        self.head.config.flow_contract_rho = 0.98
        gain = nn.Parameter(torch.tensor(1.25))
        route = torch.randn(1, 1, 11, 2)
        route[..., -1, 0] = 3.5
        route[..., -1, 1] = 0.0
        anchors = route.clone()
        modes = torch.zeros(1, dtype=torch.long)

        def linear_endpoint(x, *_args):
            out = x.clone()
            out[..., :10, :] = out[..., :10, :] * gain
            return out

        with patch.object(self.head, 'predict_flow_endpoint', side_effect=linear_endpoint):
            loss, sigma = self.head._endpoint_jacobian_contract_loss(
                route, torch.tensor([0.5]), anchors, modes,
                self.bev[:1], (4, 4), self.ego[:1],
            )
            torch.testing.assert_close(sigma, gain.detach()[None], rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(loss, (gain - 0.98).square(), rtol=1e-4, atol=1e-4)
            loss.backward()
        torch.testing.assert_close(gain.grad, 2 * (gain.detach() - 0.98), rtol=1e-4, atol=1e-4)

    def test_local_frame_rotation_bounds_ramp_and_speed_token(self):
        # A vertical route has a horizontal normal, not the ego-y direction.
        route = torch.stack([torch.zeros(10), torch.arange(10.)], dim=-1)[None].expand(4, -1, -1)
        speed = torch.tensor([4., 0.]).expand(4, 1, 2)
        self.head.config.diffusion_speed = True
        traj = torch.cat([route, speed], dim=1)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            torch.manual_seed(13)
            delta = self.head.local_trajectory_perturbation(traj)
        self.assertEqual(delta.dtype, torch.float32)
        torch.testing.assert_close(delta[:, 0], torch.zeros(4, 2))
        torch.testing.assert_close(delta[:, -1], torch.zeros(4, 2))
        lat = delta[:, 9, 0].abs()
        self.assertTrue(((lat >= 0.02) & (lat <= 0.08)).all())
        self.assertTrue((delta[:, 9, 1].abs() <= 0.03).all())
        ramp = torch.linspace(0, 1, 10)[None, :, None]
        torch.testing.assert_close(delta[:, :10], ramp * delta[:, 9:10])
        # Geometry (including a curve) is equivariant to a 90 degree rotation.
        curve = torch.stack([torch.linspace(1, 3, 10), torch.linspace(1, 3, 10).square()], dim=-1)[None]
        rotation = torch.tensor([[0., -1.], [1., 0.]])
        torch.manual_seed(17)
        d1 = self.head.local_trajectory_perturbation(curve)
        torch.manual_seed(17)
        d2 = self.head.local_trajectory_perturbation(curve @ rotation.T)
        torch.testing.assert_close(d2, d1 @ rotation.T)
        stationary = self.head.local_trajectory_perturbation(torch.zeros(2, 10, 2))
        self.assertTrue(torch.isfinite(stationary).all())
        self.assertTrue((stationary[:, -1].norm(dim=-1) >= 0.02).all())

    def test_contract_subset_gt_mode_detach_and_exact_ratio(self):
        # 32 scenes -> exactly 4 additional decoder scenes, with all 60 modes.
        bs = 32
        anchors = self.anchors[:1].expand(bs, -1, -1, -1)
        gt = self.gt[:1].expand(bs, -1, -1)
        modes = self.head.nearest_anchor_idx(gt, anchors)
        tau = torch.full((bs,), 0.5)
        xt = 0.5 * anchors + 0.5 * gt[:, None]
        endpoint = (xt + 0.2).clone().requires_grad_()
        pert_endpoints = []
        selected_indices = []
        ego = torch.arange(bs).float()[:, None, None].expand(bs, 10, 256)
        bev = self.bev[:1].expand(bs, -1, -1, -1)
        images = torch.arange(bs).float()[:, None]
        self.head.config.flow_contract_step_bank = (4,)

        def perturbed(x, t, anchor, b, shape, e, img):
            selected = e[:, 0, 0].long()
            selected_indices.append(selected)
            self.assertEqual(x.shape, (4, 60, 10, 2))
            torch.testing.assert_close(img, images[selected])
            change = x - xt[selected]
            altered = change.abs().sum(dim=(-1, -2)) > 0
            self.assertEqual(altered.sum().item(), 4)
            self.assertTrue(altered[torch.arange(4), modes[selected]].all())
            # Endpoint sensitivity of 3 -> next-state amplification of 2.
            out = (endpoint.detach()[selected] + 3 * change).requires_grad_()
            pert_endpoints.append(out)
            return out

        with patch.object(self.head, "predict_flow_endpoint", side_effect=perturbed) as predict:
            loss, ratio = self.head.contractive_recovery_loss(
                xt, tau, anchors, endpoint, modes, bev, (4, 4), ego, images,
            )
        self.assertEqual(predict.call_count, 1)
        selected = selected_indices[0]
        change = predict.call_args.args[0] - xt[selected]
        delta = self.head.gather_mode(change, modes[selected])
        d0 = delta.norm(dim=-1).mean(dim=-1)
        torch.testing.assert_close(ratio, 2 * d0 / (d0 + 1e-4), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(loss, F.relu(ratio - 0.98).mean())
        loss.backward()
        self.assertIsNone(endpoint.grad)  # No CR gradient through nominal endpoint.
        nonzero = pert_endpoints[0].grad.abs().sum(dim=(-1, -2)) > 0
        self.assertEqual(nonzero.sum().item(), 4)
        self.assertTrue(nonzero[torch.arange(4), modes[selected]].all())

    def test_contract_terminal_step_clipped_and_disabled_cost(self):
        # At tau=.94 a K=4 step must stop at 1, not overshoot to 1.19.
        self.head.config.flow_contract_step_bank = (4,)
        tau = torch.full((2,), 0.94)
        xt = self.anchors.clone()
        endpoint = self.anchors + 0.2
        modes = self.head.nearest_anchor_idx(self.gt, self.anchors)
        with patch.object(self.head, "predict_flow_endpoint", return_value=endpoint[:1]):
            loss, ratio = self.head.contractive_recovery_loss(
                xt, tau, self.anchors, endpoint, modes, *self.scene[:2], self.ego,
            )
        self.assertLess(ratio.max().item(), 1e-4)
        self.assertEqual(loss.item(), 0)
        with patch.object(self.head, "predict_flow_endpoint", return_value=endpoint) as predict:
            out = self.head.forward_train_flow(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt)
        self.assertEqual(predict.call_count, 1)
        torch.testing.assert_close(out["flow_loss"], out["flow_fm_loss"])
        self.assertEqual(out["flow_contract_loss"].item(), 0)
        self.assertEqual(out["flow_contract_success"].item(), 0)

    def test_training_independent_of_both_solver_step_counts(self):
        outputs = []
        with patch.object(self.head, "predict_flow_endpoint", return_value=self.anchors + 0.2):
            for ddbm_steps, flow_steps in ((20, 20), (4, 8), (8, 4)):
                self.head.config.step_num = ddbm_steps
                self.head.config.flow_inference_steps = flow_steps
                torch.manual_seed(99)
                outputs.append(self.head.forward_train_flow(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt))
                self.assertEqual(len(self.head.get_flow_schedule()), flow_steps + 1)
        for output in outputs[1:]:
            torch.testing.assert_close(output["flow_fm_loss"], outputs[0]["flow_fm_loss"], rtol=0, atol=0)

    def test_real_contract_bfloat16_forward_backward(self):
        self.head.train()
        self.head.config.flow_contract_enable = True
        neural_dtypes, state_dtypes = [], []
        handles = []
        for layer in self.head.diff_decoder.layers:
            handles.append(layer.task_decoder.register_forward_hook(lambda m, a, out: neural_dtypes.append(out[0].dtype)))
            handles.append(layer.register_forward_hook(lambda m, a, out: state_dtypes.append(out[0].dtype)))
        try:
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = self.head.forward_train_flow(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt)
            torch.testing.assert_close(out["flow_loss"].detach(), out["flow_fm_loss"] + 0.01 * out["flow_contract_loss"])
            self.assertEqual(len(neural_dtypes), 4)
            self.assertEqual(set(neural_dtypes), {torch.bfloat16})
            self.assertEqual(set(state_dtypes), {torch.float32})
            for name in ("flow_loss", "flow_fm_loss", "flow_contract_loss", "flow_contract_ratio", "best_reg"):
                self.assertEqual(out[name].dtype, torch.float32)
                self.assertTrue(torch.isfinite(out[name]).all())
            out["flow_loss"].backward()
            for param in self.head.parameters():
                if param.grad is not None:
                    self.assertTrue(torch.isfinite(param.grad).all())
        finally:
            for handle in handles:
                handle.remove()

    def test_paired_dropout_matches_nominal_and_subset(self):
        self.head.train()
        tau = torch.tensor([0.3, 0.7])
        args = (self.anchors, tau, self.anchors, self.bev, (4, 4), self.ego)
        with torch.no_grad():
            torch.manual_seed(19)
            original = self.head.predict_flow_endpoint(*args)
            torch.manual_seed(19)
            with self.head._paired_contract_dropout() as pair:
                clean = self.head.predict_flow_endpoint(*args)
                pair["selected"] = torch.tensor([1])
                subset = self.head.predict_flow_endpoint(
                    self.anchors[1:], tau[1:], self.anchors[1:], self.bev[1:], (4, 4), self.ego[1:],
                )
        torch.testing.assert_close(clean, original, rtol=0, atol=0)
        torch.testing.assert_close(subset, clean[1:], rtol=1e-5, atol=1e-5)
        for module in self.head.diff_decoder.modules():
            self.assertFalse(module._forward_hooks)
            self.assertFalse(module._forward_pre_hooks)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "CUDA BF16 unavailable")
    def test_cuda_bf16_contract_and_few_step_inference(self):
        head = copy.deepcopy(self.head).cuda().train()
        head.config.flow_contract_enable = True
        ego, bev, gt = self.ego.cuda(), self.bev.cuda(), self.gt.cuda()
        anchors = self.anchors.cuda()
        tau = torch.tensor([0.3, 0.7], device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            torch.manual_seed(123)
            nominal = head.predict_flow_endpoint(anchors, tau, anchors, bev, (4, 4), ego)
            torch.manual_seed(123)
            with head._paired_contract_dropout():
                paired = head.predict_flow_endpoint(anchors, tau, anchors, bev, (4, 4), ego)
        torch.testing.assert_close(paired, nominal, rtol=0, atol=0)
        neural_dtypes, state_dtypes = [], []
        handles = []
        for layer in head.diff_decoder.layers:
            handles.append(layer.task_decoder.register_forward_hook(lambda m, a, out: neural_dtypes.append(out[0].dtype)))
            handles.append(layer.register_forward_hook(lambda m, a, out: state_dtypes.append(out[0].dtype)))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = head(ego, bev, (4, 4), gt_checkpoints=gt)
            self.assertEqual(set(neural_dtypes), {torch.bfloat16})
            self.assertEqual(set(state_dtypes), {torch.float32})
            self.assertEqual(output["flow_loss"].dtype, torch.float32)
            output["flow_loss"].backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()))
            for p in head.parameters():
                if p.grad is not None:
                    self.assertTrue(torch.isfinite(p.grad).all())
            head.eval()
            for steps in (4, 8, 20):
                head.config.flow_inference_steps = steps
                with torch.autocast("cuda", dtype=torch.bfloat16), \
                     patch.object(head, "predict_flow_endpoint", wraps=head.predict_flow_endpoint) as predict:
                    result = head(ego, bev, (4, 4))
                self.assertEqual(predict.call_count, steps)
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result).all())
        finally:
            for handle in handles:
                handle.remove()

    def test_planning_loss_counts_flow_once_and_preserves_ddbm(self):
        planner = PlanningDecoderDDBM.__new__(PlanningDecoderDDBM)
        nn.Module.__init__(planner)
        planner.config = SimpleNamespace(
            predict_ddbm=True, diffusion_speed=False, torch_float_type=torch.float32,
            predict_temporal_spatial_waypoints=False, predict_target_speed=False,
            log_scalars_frequency=1, trajectory_sampling_num_poses=10,
        )
        planner.device = torch.device("cpu")
        planner.trajectory_head = self.head
        planner.plan_anchor = self.head.plan_anchor
        planner.loss_computer_ddbm = LossComputer(self.config)
        output = {
            "best_reg": self.gt + 0.2, "poses_cls_list": [torch.randn(2, 60), torch.randn(2, 60)],
            "flow_loss": torch.tensor(3.0, requires_grad=True),
            **{name: torch.tensor(0.1) for name in (
                "flow_tau_mean", "flow_velocity_l1", "flow_endpoint_l1", "flow_fm_loss",
                "flow_contract_loss", "flow_contract_ratio", "flow_contract_success",
            )},
        }
        loss, log = {}, {}
        planner.compute_loss(SimpleNamespace(pred_route=output), {"route": self.gt, "iteration": 0}, loss, log)
        self.assertIs(loss["loss_ddbm_reg"], output["flow_loss"])
        self.assertIn("metric/flow_endpoint_l1", log)
        self.assertAlmostEqual(log["metric/route_ade"], 0.2 * 2 ** 0.5, places=5)
        self.assertAlmostEqual(log["metric/route_fde"], 0.2 * 2 ** 0.5, places=5)
        self.assertEqual(set(loss), {"loss_ddbm_reg", "loss_ddbm_cls"})
        self.head.trajectory_generator = "ddbm"
        legacy_loss = {}
        planner.compute_loss(SimpleNamespace(pred_route=output), {"route": self.gt}, legacy_loss, {})
        torch.testing.assert_close(legacy_loss["loss_ddbm_reg"], 2 * F.l1_loss(output["best_reg"], self.gt))
        torch.testing.assert_close(legacy_loss["loss_ddbm_cls"], loss["loss_ddbm_cls"])

    def test_ddbm_switch_and_checkpoint_structure(self):
        config = copy.deepcopy(self.config)
        config.trajectory_generator = "ddbm"
        legacy = TrajectoryHead(10, 128, 256, str(ROOT / "anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"), config)
        legacy.load_state_dict(self.head.state_dict(), strict=True)
        legacy.train()
        with patch.object(legacy.diffusion_scheduler, "add_noise", wraps=legacy.diffusion_scheduler.add_noise) as add_noise:
            output = legacy(self.ego, self.bev, (4, 4), gt_checkpoints=self.gt)
        self.assertEqual(add_noise.call_count, 1)
        self.assertNotIn("flow_loss", output)
        legacy.eval()
        legacy.config.step_num = 2
        legacy.config.flow_inference_steps = 8
        with torch.no_grad(), patch.object(legacy.diffusion_scheduler, "sample_step", wraps=legacy.diffusion_scheduler.sample_step) as sample:
            result = legacy(self.ego, self.bev, (4, 4))
        self.assertEqual(sample.call_count, 2)
        self.assertEqual(result.shape, (2, 10, 2))
        self.assertTrue(torch.isfinite(result).all())

    def test_config_roundtrip_and_solver_overrides(self):
        from fmsr.training.config_training import TrainingConfig
        from fmsr.inference.config_closed_loop import ClosedLoopConfig
        overrides = {
            "FMSR_TRAINING_CONFIG": "step_num=20 flow_inference_steps=8 flow_contract_enable=true flow_contract_step_bank=[4,8,20]",
            "FMSR_CLOSED_LOOP_CONFIG": "step_num=20 flow_inference_steps=4",
            "FMSR_OPEN_LOOP_CONFIG": "",
        }
        with patch.dict("os.environ", overrides):
            training = TrainingConfig({"flow_inference_steps": 20})
            closed = ClosedLoopConfig()
        self.assertEqual(training.step_num, 20)
        self.assertEqual(training.flow_inference_steps, 8)
        self.assertTrue(training.flow_contract_enable)
        self.assertEqual(training.flow_contract_step_bank, [4, 8, 20])
        self.assertEqual(closed.step_num, 20)
        self.assertEqual(closed.flow_inference_steps, 4)

    def test_contract_config_rejects_invalid_settings(self):
        self.head.config.flow_contract_enable = True
        for key, value in (
            ("flow_min_remain", 0), ("flow_inference_steps", 0),
            ("flow_contract_fraction", 0), ("flow_contract_fraction", 1.1),
            ("flow_contract_weight", -1), ("flow_contract_rho", 1),
            ("flow_contract_lat_min_m", 0), ("flow_contract_lat_max_m", 0.01),
            ("flow_contract_lon_max_m", -1), ("flow_contract_step_bank", ()),
            ("flow_contract_step_bank", (4, 0)), ("flow_contract_step_bank", (True,)),
            ("tf_dropout", 0.1),
        ):
            with self.subTest(key=key, value=value), patch.object(self.head.config, key, value):
                with self.assertRaises(ValueError):
                    self.head._validate_flow_config()

    def test_step_counts_and_invalid_config(self):
        for steps in (1, 4, 8, 20):
            schedule = self.head.get_flow_schedule(steps)
            self.assertEqual(len(schedule), steps + 1)
            self.assertEqual(schedule[0], 0)
            self.assertEqual(schedule[-1], 1)
        for steps in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.head.get_flow_schedule(steps)
        self.head.config.flow_train_t_max = 1.0
        with self.assertRaises(ValueError):
            self.head._validate_flow_config()


if __name__ == "__main__":
    unittest.main()
