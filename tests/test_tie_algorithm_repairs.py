"""Regressions for full spectral CR, decoder gradients and deployment rollouts."""
import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_fd import TrajectoryHead


ROOT = Path(__file__).resolve().parents[1]


class TIEAlgorithmRepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = SimpleNamespace(
            trajectory_generator='residual_flow', diffusion_speed=False, speed_holder=0,
            step_num=20, flow_train_t_max=.95, flow_min_remain=.05, flow_inference_steps=20,
            flow_contract_enable=True, flow_contract_method='endpoint_jacobian',
            flow_contract_fd_epsilon=.01, flow_contract_fd_chunk=4,
            flow_contract_interval=1, flow_contract_warmup_steps=0,
            flow_contract_fraction=1., flow_contract_weight=.001, flow_contract_rho=.98,
            flow_contract_tau_kappa=(1.30, 1.20, 1.08, 1.00),
            flow_contract_pred_mode_fraction=0., flow_contract_on_policy_fraction=0.,
            flow_contract_lat_min_m=.02, flow_contract_lat_max_m=.08,
            flow_contract_lon_max_m=.03, flow_contract_yaw_max_deg=1.,
            flow_contract_step_bank=(4, 8, 20),
            tf_d_model=256, tf_d_ffn=128, tf_num_head=8, tf_dropout=0.,
            lidar_max_x=32., lidar_max_y=32.,
        )
        cls.anchor_path = str(ROOT/'anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy')

    def setUp(self):
        torch.manual_seed(42)
        self.head = TrajectoryHead(10, 128, 256, self.anchor_path, self.config)

    def test_full_spectral_cr_detects_expansion_missed_by_structured_directions(self):
        self.head.std = torch.ones_like(self.head.std)
        vector = torch.zeros(20)
        vector[[1, 3, 5]] = torch.tensor([1., -2., 1.])
        vector /= vector.norm()
        matrix = nn.Parameter(.8 * torch.eye(20) + 1.2 * torch.outer(vector, vector))
        state = torch.zeros(2, 10, 2)
        state[:, :, 0] = torch.arange(10) + 2.5

        def endpoint(x, *_args):
            return (x.flatten(-2) @ matrix.T).reshape_as(x)

        with patch.object(self.head, 'predict_flow_endpoint', side_effect=endpoint):
            loss, sigma = self.head._spectral_contract_at_state(
                state, torch.tensor([.1, .8]), state,
                torch.zeros(2, 1512, 2, 2), (2, 2), torch.zeros(2, 10, 256))
            loss.backward()
        torch.testing.assert_close(sigma, torch.full((2,), 2.), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(loss.detach(), torch.tensor((.7 ** 2 + 1. ** 2) / 2),
                                   atol=1e-4, rtol=1e-4)
        self.assertTrue(torch.isfinite(matrix.grad).all())
        self.assertGreater(matrix.grad.norm().item(), 0.)

    def test_all_intended_parameters_in_both_decoder_layers_receive_fm_gradients(self):
        self.head.config.flow_contract_enable = False
        self.head.train()
        anchors = self.head.plan_anchor[None].expand(2, -1, -1, -1)
        gt = anchors[:, 4] + .2
        output = self.head.forward_train_flow(
            torch.randn(2, 10, 256) * .2, torch.randn(2, 1512, 4, 4) * .1,
            (4, 4), gt_checkpoints=gt)
        output['flow_loss'].backward()
        for index, layer in enumerate(self.head.diff_decoder.layers):
            intended = [(name, p) for name, p in layer.named_parameters()
                        if 'plan_cls_branch' not in name]
            self.assertEqual(len(intended), 48)
            for name, parameter in intended:
                with self.subTest(layer=index, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_spectral_cr_alone_reaches_first_regression_layer(self):
        self.head.train()
        state = self.head.plan_anchor[[4]].detach().clone()
        # A low diagnostic margin activates the hinge; production schedule is unchanged.
        with patch.object(self.head, '_contract_threshold', side_effect=lambda t: torch.zeros_like(t)):
            loss, _ = self.head._spectral_contract_at_state(
                state, torch.tensor([.5]), state,
                torch.randn(1, 1512, 4, 4) * .1, (4, 4), torch.randn(1, 10, 256) * .2)
            loss.backward()
        for layer in self.head.diff_decoder.layers:
            grad = layer.task_decoder.plan_reg_branch[-1].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0.)

    def test_on_policy_queries_are_exact_inference_prefixes_and_ignore_gt(self):
        anchors = torch.ones(3, 10, 2)
        nominal = torch.full_like(anchors, 100.)
        times = torch.tensor([.17, .38, .91])
        indices = torch.tensor([0, 7, 19])
        query_times = []
        gain = nn.Parameter(torch.tensor(1.2))

        def endpoint(x, t, *_args):
            query_times.extend(t.tolist())
            return gain * x

        with patch.object(self.head, 'predict_flow_endpoint', side_effect=endpoint), \
             patch('torch.randint', return_value=indices):
            generated, tau = self.head._flow_cr_query_states(
                nominal, times, anchors, torch.ones(3, dtype=torch.bool),
                torch.zeros(3, 1512, 2, 2), (2, 2), torch.zeros(3, 10, 256))
            regenerated, repeated_tau = self.head._flow_cr_query_states(
                nominal * -7, times + .03, anchors, torch.ones(3, dtype=torch.bool),
                torch.zeros(3, 1512, 2, 2), (2, 2), torch.zeros(3, 10, 256))
        expected = []
        for count in indices.tolist():
            value = 1.
            for index in range(count):
                value *= 1. + .05 * .2 / (1. - .05 * index)
            expected.append(value)
        torch.testing.assert_close(generated[:, 0, 0], torch.tensor(expected))
        torch.testing.assert_close(tau, indices.float() / 20.)
        torch.testing.assert_close(generated, regenerated)
        torch.testing.assert_close(tau, repeated_tau)
        self.assertFalse(generated.requires_grad)
        self.assertTrue(all(time < 1. for time in query_times))
        self.assertIsNone(gain.grad)

    def test_single_mode_rollout_matches_batched_decoder_inference(self):
        self.head.train()  # Helper must apply deployment dropout without changing mode.
        anchors = self.head.plan_anchor[[4, 7]].unsqueeze(0)
        bev, ego = torch.randn(1, 1512, 2, 2) * .1, torch.randn(1, 10, 256) * .2
        with torch.no_grad(), self.head._deterministic_cr_decoder():
            expected = self.head.flow_step(anchors, 0., .05, anchors, (bev, (2, 2), ego, None))
        actual, tau = self.head._flow_rollout_query(
            anchors[0], torch.ones(2, dtype=torch.long), bev.expand(2, -1, -1, -1),
            (2, 2), ego.expand(2, -1, -1))
        torch.testing.assert_close(actual, expected[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(tau, torch.full((2,), .05))
        self.assertTrue(self.head.diff_decoder.layers[0].dropout1.training)

    def test_spectral_delayed_schedule_keeps_original_kappa_and_interval_origin(self):
        cfg = copy.deepcopy(self.config)
        cfg.flow_contract_delay_steps = 2
        cfg.flow_contract_ramp_steps = 2
        cfg.flow_contract_interval = 4
        head = TrajectoryHead(10, 128, 256, self.anchor_path, cfg).train()
        anchors = head.plan_anchor[None].expand(1, -1, -1, -1)
        with patch.object(head, 'predict_flow_endpoint', side_effect=lambda x, *_: x), \
             patch.object(head, '_classify_flow_anchors', return_value=[torch.zeros(1, 60)]), \
             patch.object(head, 'contractive_recovery_loss', return_value=(torch.tensor(1.), torch.ones(1))) as cr:
            outputs = [head.forward_train_flow(torch.zeros(1, 10, 256),
                       torch.zeros(1, 1512, 2, 2), (2, 2), gt_checkpoints=anchors[:, 4]) for _ in range(4)]
        self.assertEqual(cr.call_count, 1)
        self.assertEqual([out['flow_contract_active'].item() for out in outputs], [0., 0., 1., 0.])
        torch.testing.assert_close(head._contract_threshold(torch.tensor([.1, .3, .6, .8])),
                                   torch.tensor([1.3, 1.2, 1.08, 1.]))


class TIELauncherRepairTests(unittest.TestCase):
    def test_training_and_recovery_launchers_use_spectral_cr_and_preserve_kappa(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            capture, stub = temp/'capture.jsonl', temp/'python_stub'
            stub.write_text('#!/home/vcdc/anaconda3/envs/fmsr/bin/python\n'
                'import json,os,sys\n'
                "keys=('FLOW_CONTRACT_ENABLE','FLOW_CONTRACT_METHOD','FLOW_CONTRACT_TAU_KAPPA','FMSR_MAX_SPEED')\n"
                "with open(os.environ['TIE_TEST_CAPTURE'],'a') as f: f.write(json.dumps({k:os.environ.get(k) for k in keys})+'\\n')\n"
                "if any('mount_carla_garage_data.py' in a for a in sys.argv): print(os.environ['PDM_DATA_ROOT'])\n")
            stub.chmod(0o755)
            (temp/'recovery_frames.txt').write_text('fixture/route/0010\n')
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith(('FLOW_', 'FMSR_', 'PDM_', 'TFPP_', 'RECOVERY_'))}
            env.update(PYTHON_BIN=str(stub), TIE_TEST_CAPTURE=str(capture), PREFLIGHT_ONLY='1',
                       PDM_DATA_ROOT=str(temp), PDM_ASSET_DIR=str(temp), TFPP_CHECKPOINT=str(temp/'model.pth'),
                       RECOVERY_DATA_ROOT=str(temp), TRAIN_LOG_ROOT=str(temp/'runs'))
            for name in ('train_fmsr_pdm_lite_tie_cr_fd.sh',
                         'train_fmsr_pdm_lite_tie_cr_fd_tau_delayed.sh',
                         'posttrain_recovery_pdm_lite_tie_cr_fd_epoch11.sh',
                         'resume_recovery_pdm_lite_tie_cr_fd_20epochs.sh'):
                with self.subTest(launcher=name):
                    capture.write_text('')
                    result = subprocess.run(['bash', str(ROOT/'scripts'/name)], cwd=ROOT,
                                            env=env, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    record = json.loads(capture.read_text().splitlines()[-1])
                    self.assertIn(record['FLOW_CONTRACT_ENABLE'], ('true', '1'))
                    self.assertEqual(record['FLOW_CONTRACT_METHOD'], 'endpoint_jacobian')
                    self.assertEqual(record['FLOW_CONTRACT_TAU_KAPPA'], '1.30 1.20 1.08 1.00')
                    self.assertEqual(record['FMSR_MAX_SPEED'], '30.0')


class RolloutEndpointRecoveryTests(TIEAlgorithmRepairTests):
    def test_recovery_labels_do_not_enter_rollout_and_loss_has_gradients(self):
        self.head.config.flow_rollout_recovery_scenes = 1
        query = torch.randn(1, 10, 256)
        bev = torch.randn(1, 1512, 2, 2)
        offset = nn.Parameter(torch.tensor(.2))
        states = []

        def prefix(anchors, indices, *_args):
            states.append(anchors.clone())
            return anchors.clone(), torch.full((1,), .5)

        def endpoint(state, *_args):
            return state + offset

        with patch.object(self.head, '_flow_rollout_query', side_effect=prefix), \
                patch.object(self.head, 'predict_flow_endpoint', side_effect=endpoint):
            torch.manual_seed(137)
            loss = self.head.rollout_endpoint_recovery(query, bev, (2, 2),
                gt_checkpoints=torch.zeros(1, 10, 2))
            loss.backward()
            torch.manual_seed(137)
            changed_loss = self.head.rollout_endpoint_recovery(query, bev, (2, 2),
                gt_checkpoints=torch.full((1, 10, 2), 999.))
        torch.testing.assert_close(states[0], states[1], atol=0, rtol=0)
        self.assertNotEqual(loss.item(), changed_loss.item())
        self.assertIsNotNone(offset.grad)
        self.assertTrue(torch.isfinite(offset.grad))

    def test_zero_recovery_weight_does_not_call_auxiliary_objective(self):
        self.head.config.flow_contract_enable = False
        self.head.config.flow_rollout_recovery_weight = 0.
        self.head.train()
        with patch.object(self.head, 'rollout_endpoint_recovery', side_effect=AssertionError('disabled')):
            result = self.head.forward_train_flow(torch.randn(1, 10, 256),
                torch.randn(1, 1512, 2, 2), (2, 2),
                gt_checkpoints=self.head.plan_anchor[[4]].detach())
        self.assertNotIn('flow_rollout_recovery_loss', result)


if __name__ == '__main__':
    unittest.main()
