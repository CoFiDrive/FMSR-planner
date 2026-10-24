"""PDM route-label parity and real-model DDP regression tests (no weight downloads).

Run with the fmsr interpreter and unittest discover. PDM_TEST_WORLD_SIZE=2 tests
multi-process CPU DDP; PDM_TEST_DEVICE=cuda enables the single-GPU BF16 check.
"""
import gzip
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '3rd_party/carla_garage/team_code'),
                str(ROOT / '3rd_party/CARLA_0915/PythonAPI/carla')]
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timm
from config import GlobalConfig
from fmsr_interface import restore_trained_interface, continuous_speed_for_control, continuous_speed_limit
from data import CARLA_Data
from model import LidarCenterNet
from route_geometry import prepare_route
from scripts.pdm_lite.build_fmsr_assets import resample_route
from fmsr.tfv6.diffusion_modules.blocks import GridSampleCrossBEVAttention


class PDMInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_continuous_speed_reaches_control_without_quantization(self):
        for speed in (0., .5, 1., 1.5, 2.5, 5., 7.):
            self.assertAlmostEqual(continuous_speed_for_control([torch.tensor(speed)], 20.), speed)
        self.assertEqual(continuous_speed_for_control([torch.tensor(1.), torch.tensor(2.)], 20.), 1.5)
        self.assertEqual(continuous_speed_for_control([torch.tensor(-1.)], 20.), 0.)
        self.assertEqual(continuous_speed_for_control([torch.tensor(24.)], 20., .8), 16.)

    def test_continuous_speed_bound_is_independent_of_class_bins(self):
        c = config_for_test()
        self.assertEqual(max(c.target_speeds), 20.)
        self.assertEqual(continuous_speed_limit(c), 30.)
        self.assertEqual(continuous_speed_for_control([torch.tensor(24.)], continuous_speed_limit(c)), 24.)
        self.assertEqual(continuous_speed_limit(SimpleNamespace(target_speeds=[0., 20.])), 30.)
        for limit in (0., -1., float('nan'), float('inf')):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                continuous_speed_limit(SimpleNamespace(fmsr_max_speed=limit))

    def test_bev_samples_actual_lidar_point_in_symmetric_and_offset_grids(self):
        for bounds in ((-32, 32, -32, 32), (-16, 48, -8, 24)):
            c = config_for_test()
            c.min_x, c.max_x, c.min_y, c.max_y = bounds
            loader = object.__new__(CARLA_Data)
            loader.config = c
            point = np.array([[10.125, .125, 1.]])
            bev = torch.from_numpy(loader.lidar_to_histogram_features(point, use_ground_plane=False))[None]
            attention = GridSampleCrossBEVAttention(1, 1, in_bev_dims=1, num_points=1, config=c).eval()
            attention.value_proj = torch.nn.Identity()
            attention.output_proj = torch.nn.Identity()
            sampled = attention(torch.zeros(1, 1, 1), torch.tensor(point[:, :2], dtype=torch.float32).reshape(1, 1, 1, 2),
                                bev, bev.shape[-2:])
            self.assertAlmostEqual(sampled.item(), bev.max().item(), places=6)

    def test_legacy_checkpoint_input_semantics_are_explicit(self):
        c = config_for_test()
        restore_trained_interface(c, SimpleNamespace(use_fmsr=True))
        self.assertFalse(c.fmsr_use_target_point)
        self.assertEqual(c.fmsr_bev_layout, 'legacy')
        saved = config_for_test()
        restore_trained_interface(c, saved)
        self.assertTrue(c.fmsr_use_target_point)
        self.assertEqual(c.fmsr_bev_layout, 'carla_garage')

    def test_real_model_navigation_gradients_and_scalar_speed(self):
        torch.manual_seed(7)
        c = config_for_test()
        c.trajectory_generator = 'residual_flow'
        c.flow_inference_steps = 1
        with tempfile.TemporaryDirectory(prefix='pdm_interface_') as directory:
            c.fmsr_anchor_path = directory + '/anchors.npy'
            anchors = np.zeros((60, 11, 2), dtype=np.float32)
            anchors[:, :10, 0] = 2.5 + np.arange(10)
            anchors[:, -1, 0] = 1.5
            np.save(c.fmsr_anchor_path, anchors)
            create = timm.create_model
            def offline_create(*args, **kwargs):
                kwargs['pretrained'] = False
                return create(*args, **kwargs)
            with patch('timm.create_model', offline_create):
                model = LidarCenterNet(c)
            model.freeze_for_fmsr()
            model.eval()
            inputs = dict(rgb=torch.zeros(1, 3, c.camera_height, c.camera_width),
                          lidar_bev=torch.zeros(1, 2 if c.use_ground_plane else 1, 256, 256),
                          ego_vel=torch.ones(1, 1), command=torch.tensor([[0., 0., 0., 1., 0., 0.]]),
                          target_point=torch.tensor([[10., 5.]]))
            # Cache the frozen backbone output so all probes use identical scene features.
            with torch.no_grad():
                features = model.backbone(inputs['rgb'], inputs['lidar_bev'])
            captured = []
            original_classify = model.fmsr_head._classify_flow_anchors
            def capture(*args):
                captured.append(args[-1].detach().clone())
                return original_classify(*args)
            with patch.object(model.backbone, 'forward', return_value=features), \
                 patch.object(model.fmsr_head, '_classify_flow_anchors', side_effect=capture):
                labels = dict(fmsr_route=torch.from_numpy(anchors[:1, :10]),
                              fmsr_target_speed=torch.tensor([1.5]), return_fmsr_loss=True)
                losses = model(**inputs, **labels)['losses']
                parameters = list(model.fmsr_tp_encoder.parameters())
                for loss in losses.values():
                    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
                self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
                with torch.no_grad():
                    first = model(**inputs)
                    inputs['target_point'] = torch.tensor([[10., -5.]])
                    second = model(**inputs)
                self.assertEqual(first[1].shape, (1, 1))
                self.assertEqual(captured[-1].shape[1], 11)
                torch.testing.assert_close(captured[-1][:, :10], captured[-2][:, :10])
                self.assertFalse(torch.equal(captured[-1][:, -1], captured[-2][:, -1]))
                self.assertFalse(torch.equal(first[2], second[2]))
                # Exercise both model decoding and the controller's scalar interface.
                prediction = torch.zeros(1, 11, 2)
                prediction[:, -1, 0] = 24.
                with patch.object(model.fmsr_head, 'forward', return_value=prediction), torch.no_grad():
                    output = model(**inputs)
                self.assertEqual(output[1].item(), 24.)
                self.assertEqual(continuous_speed_for_control([output[1]], continuous_speed_limit(c)), 24.)


def config_for_test():
    c = GlobalConfig()
    for name, value in dict(use_fmsr=True, use_wp_gru=False, use_semantic=False,
                            use_depth=False, use_bev_semantic=False, detect_boxes=False,
                            use_controller_input_prediction=True, augment=False,
                            tf_dropout=0., debug=0).items():
        setattr(c, name, value)
    return c


def ddp_worker(rank, world_size, directory, sample):
    torch.set_num_threads(2)
    torch.manual_seed(123 + rank)
    device = os.environ.get('PDM_TEST_DEVICE', 'cpu')
    dist.init_process_group('gloo', init_method='file://' + directory + '/rdzv',
                            rank=rank, world_size=world_size)
    c = config_for_test()
    c.fmsr_anchor_path = directory + '/anchors.npy'
    create = timm.create_model

    def offline_create(*args, **kwargs):
        kwargs['pretrained'] = False
        return create(*args, **kwargs)

    try:
        for generator, contract in [('ddbm', False), ('residual_flow', False), ('residual_flow', True)]:
            c.trajectory_generator = generator
            c.flow_contract_enable = contract
            with patch('timm.create_model', offline_create):
                model = LidarCenterNet(c).to(device)
            model.freeze_for_fmsr()
            model.eval()
            model.fmsr_head.train()
            ddp = torch.nn.parallel.DistributedDataParallel(
                model, broadcast_buffers=False, find_unused_parameters=True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            anchor_before = model.fmsr_head.plan_anchor.detach().clone()
            frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}

            def tensor(x):
                return torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

            inputs = dict(rgb=tensor(sample['rgb']), lidar_bev=tensor(sample['lidar']),
                          target_point=tensor(sample['target_point']), ego_vel=tensor([sample['speed']]),
                          command=tensor(sample['command']),
                          target_point_next=tensor(sample['target_point_next']) if c.two_tp_input else None,
                          fmsr_route=tensor(sample['route'][:10]),
                          fmsr_target_speed=tensor(sample['target_speed_value']),
                          return_fmsr_loss=True)
            for step in range(3):
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                    result = ddp(**inputs)
                    loss = sum(result['losses'].values())
                assert torch.isfinite(loss)
                assert set(result['metrics']) >= {'route_ade', 'route_fde'}
                if generator == 'residual_flow':
                    assert torch.equal(result['losses']['loss_fmsr_reg'], model._fmsr_output['flow_loss'])
                    metrics = result['metrics']
                    assert torch.allclose(result['losses']['loss_fmsr_reg'].detach(),
                                          metrics['flow_fm_loss'] + c.flow_contract_weight * metrics['flow_contract_loss'])
                    assert not any(v.requires_grad for v in metrics.values())
                loss.backward()
                assert model.fmsr_head.plan_anchor.grad is None
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                assert torch.equal(anchor_before, model.fmsr_head.plan_anchor)
            for name, p in model.named_parameters():
                if name in frozen:
                    assert torch.equal(frozen[name], p), name
            # Validation must compute a fresh label-conditioned objective, not cached train output.
            ddp.eval()
            with torch.inference_mode():
                result = ddp(**inputs)
            assert torch.isfinite(sum(result['losses'].values()))
            # Verify all ranks have the same updated planner weights.
            for p in model.fmsr_head.parameters():
                expected = p.detach().clone()
                dist.broadcast(expected, src=0)
                assert torch.equal(expected, p)
            if rank == 0:
                print(f'PASS {generator}, CR={contract}: 3 DDP updates + validation; anchors frozen', flush=True)
            del ddp, model, optimizer, frozen
    finally:
        dist.destroy_process_group()


class PDMIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for BF16 engine test')
    def test_engine_bf16_logging(self):
        from train import Engine
        from torch.utils.tensorboard import SummaryWriter
        from torch.utils.data import default_collate
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        root = Path(os.environ.get('PDM_TEST_DATA_ROOT', '/media/vcdc/T7/carla_garage_data'))
        scenario = root / 'CrossingBicycleFlow'
        if not scenario.is_dir() or not any(scenario.iterdir()):
            self.skipTest('PDM data not mounted')
        c = config_for_test()
        c.trajectory_generator = 'residual_flow'
        c.flow_contract_enable = True
        c.use_amp = True
        c.use_cosine_schedule = False
        c.detailed_loss_weights = {'loss_fmsr_reg': 1., 'loss_fmsr_cls': 1.}
        dataset = CARLA_Data(root=[str(scenario)], config=c, rank=1)
        sample = dataset[0]
        batch = default_collate([sample])
        with tempfile.TemporaryDirectory(prefix='pdm_engine_') as directory:
            state = np.concatenate([sample['route'][:10], [[float(sample['target_speed_value']), 0]]]).astype('float32')
            c.fmsr_anchor_path = directory + '/anchors.npy'
            np.save(c.fmsr_anchor_path, np.repeat(state[None], 60, axis=0))
            create = timm.create_model
            def offline_create(*args, **kwargs):
                kwargs['pretrained'] = False
                return create(*args, **kwargs)
            dist.init_process_group('nccl', init_method='file://' + directory + '/rdzv', rank=0, world_size=1)
            try:
                with patch('timm.create_model', offline_create):
                    model = LidarCenterNet(c).cuda()
                model.freeze_for_fmsr()
                ddp = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False,
                                                                find_unused_parameters=True)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
                writer = SummaryWriter(directory)
                engine = Engine(ddp, optimizer, [batch] * 3, [batch],
                                SimpleNamespace(logdir=directory, backbone='transFuser', continue_epoch=0),
                                c, writer, 'cuda', None, torch.amp.GradScaler('cuda', enabled=False))
                engine.train()
                engine.validate()
                writer.close()
                events = EventAccumulator(directory).Reload()
                for tag in ('metric/flow_fm_loss', 'metric/flow_contract_loss', 'metric/flow_contract_ratio',
                            'metric/flow_contract_success', 'metric/route_ade', 'metric/route_fde',
                            'train_step/loss_fmsr_reg'):
                    values = events.Scalars(tag)
                    self.assertEqual(len(values), 3, tag)
                    self.assertTrue(all(np.isfinite(v.value) for v in values), tag)
                self.assertIn('val_loss_total', events.Tags()['scalars'])
                self.assertFalse(model.fmsr_head.plan_anchor.requires_grad)
                self.assertIsNone(model.fmsr_head.plan_anchor.grad)
            finally:
                dist.destroy_process_group()

    def test_route_geometry_parity(self):
        loader = object.__new__(CARLA_Data)
        loader.config = config_for_test()
        routes = [np.array([[2., 0], [4., 0], [6., 2], [8., 4], [10., 6], [12., 8]]),
                  np.array([[2., 0], [2., 0], [4., 0], [6., 0]]),
                  np.stack((np.arange(1., 31.), np.sin(np.arange(1., 31.) / 3)), axis=-1)]
        for route in routes:
            expected = loader.smooth_path(prepare_route(route, loader.config.num_route_points))[:10]
            np.testing.assert_allclose(resample_route(route, 10), expected, atol=1e-6)
        # Preserve the upstream straight-route convention: 2.5m first point, 1m spacing.
        np.testing.assert_allclose(resample_route(routes[1], 10)[:, 0], 2.5 + np.arange(10))

    def test_real_data_ddp(self):
        root = Path(os.environ.get('PDM_TEST_DATA_ROOT', '/media/vcdc/T7/carla_garage_data'))
        scenario = root / 'CrossingBicycleFlow'
        if not scenario.is_dir() or not any(scenario.iterdir()):
            self.skipTest('PDM data not mounted; set PDM_TEST_DATA_ROOT')
        c = config_for_test()
        dataset = CARLA_Data(root=[str(scenario)], config=c, rank=1)
        sample = dataset[0]
        # Compare builder against actual unaugmented route labels at multiple frames.
        for index in (0, len(dataset)//2, len(dataset)-1):
            item = dataset[index]
            folder = Path(os.fsdecode(dataset.measurements[index][0]))
            frame = int(dataset.sample_start[index])
            with gzip.open(folder / f'{frame:04}.json.gz', 'rt') as stream:
                raw = json.load(stream)['route']
            np.testing.assert_allclose(resample_route(raw, 10), item['route'][:10], atol=1e-6)
        with tempfile.TemporaryDirectory(prefix='pdm_integration_') as directory:
            state = np.concatenate([sample['route'][:10], [[float(sample['target_speed_value']), 0]]]).astype('float32')
            anchors = np.repeat(state[None], 60, axis=0)
            anchors[:, :, 0] += np.linspace(-.2, .2, 60)[:, None]
            np.save(Path(directory)/'anchors.npy', anchors)
            world_size = int(os.environ.get('PDM_TEST_WORLD_SIZE', '1'))
            if world_size == 1:
                ddp_worker(0, 1, directory, sample)
            else:
                mp.spawn(ddp_worker, args=(world_size, directory, sample), nprocs=world_size, join=True)


if __name__ == '__main__':
    unittest.main()
