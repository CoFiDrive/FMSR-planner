from types import SimpleNamespace as NS
from unittest.mock import patch
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch.utils.data import default_collate

from fmsrpdm.generation_validation import evaluate_generation, per_frame_metrics
from fmsrpdm.planner import PaperBridgePlanner
from fmsrpdm.assets import angles
from fmsrpdm.regression_health import RegressionHealth, reset_regression_refiners
from fmsrpdm.checkpoint import load_initial_checkpoint


class FixedContextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = NS(step_num=20, fmsr_max_target_speed_mps=30)
        self.planner = NS(sample=self.sample)

    def context(self, rgb, lidar, target_point, speed, command, target_point_next):
        return rgb[:, :1].reshape(len(rgb), -1)[:, :1], lidar, None

    def sample(self, query, bev, generator):
        output = torch.zeros(len(query), 11, 2)
        output[:, :10, 0] = torch.arange(10) + 2.5
        output[:, :10, 1] = torch.randn((len(query), 10), generator=generator) * .1
        output[:, -1, 0] = query[:, 0]
        return output, torch.zeros(len(query), dtype=torch.long)


def batches(label_shift=0.):
    samples = []
    for i in range(3):
        route = torch.zeros(20, 2); route[:, 0] = torch.arange(20) + 2.5
        route[:, 1] = label_shift
        samples.append({'rgb': torch.tensor([[[float(i+1)]]]), 'lidar': torch.zeros(1, 2, 2),
                        'target_point': torch.zeros(2), 'target_point_next': torch.zeros(2),
                        'speed': torch.tensor(float(i)), 'command': torch.zeros(6), 'route': route,
                        'target_speed_value': torch.tensor(float(i+1)+label_shift),
                        'validation_scenario': 'A' if i < 2 else 'B', 'validation_frame': f'frame{i}'})
    def collate(items):
        random.random(); np.random.random()
        return default_collate(items)
    return DataLoader(samples, batch_size=2, generator=torch.Generator().manual_seed(4), collate_fn=collate)


def test_generation_uses_private_rng_restores_mode_and_weights_scenes_correctly():
    model = FixedContextModel().train()
    before = torch.get_rng_state().clone()
    python_before, numpy_before = random.getstate(), np.random.get_state()
    first = evaluate_generation(model, batches(), torch.device('cpu'), seeds=(1, 2, 3))
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert python_before == random.getstate()
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0] and np.array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    second = evaluate_generation(model, batches(), torch.device('cpu'), seeds=(1, 2, 3))
    assert first == second
    assert first['samples'] == 3 and first['scenarios']['A']['1']['samples'] == 2
    assert first['scenarios']['B']['1']['samples'] == 1
    assert first['mean_over_seeds']['between_seed_path_STD_m'] > 0
    for seed in ('1', '2', '3'):
        assert first['per_seed'][seed]['speed_MAE_mps'] == 0
    shifted = evaluate_generation(model, batches(5.), torch.device('cpu'), seeds=(1, 2, 3))
    assert shifted['mean_over_seeds']['speed_MAE_mps'] == 5.
    assert shifted['mean_over_seeds']['between_seed_path_STD_m'] == first['mean_over_seeds']['between_seed_path_STD_m']


def test_metrics_measure_geometry_and_exclude_speed_from_path():
    target = torch.zeros(1, 11, 2); target[0, :10, 0] = torch.arange(10)+2.5
    prediction = target.clone(); prediction[0, 4, 1] = 1.; prediction[0, -1, 0] = 24.
    values = per_frame_metrics(prediction, target)
    assert values['route_ADE_m'].item() == pytest.approx(.1)
    assert values['route_FDE_m'].item() == 0
    assert values['near_lateral_MAE_m'].item() == pytest.approx(.2)
    assert values['speed_MAE_mps'].item() == 24.
    assert values['point_spacing_MAE_m'].item() > 0
    assert values['second_difference_MAE_m'].item() > 0


def test_clean_prior_validation_transports_only_sensor_derived_prior():
    class PriorModel(FixedContextModel):
        def __init__(self):
            super().__init__()
            self.config.prediction_base = 'tfpp_clean_prior_v1'

        def context(self, rgb, lidar, target_point, speed, command, target_point_next, return_clean_prior=False):
            assert return_clean_prior
            query, bev, grid = super().context(rgb,lidar,target_point,speed,command,target_point_next)
            prior = torch.zeros(len(query),10,2)
            prior[:, :, 0] = torch.arange(10)+2.5
            return query, bev, grid, prior

        def sample(self, query, bev, generator, clean_prior):
            output = torch.cat([clean_prior, torch.zeros(len(query),1,2)],1)
            output[:, -1, 0] = query[:, 0]
            return output, torch.zeros(len(query),dtype=torch.long)
    model = PriorModel()
    report = evaluate_generation(model,batches(),torch.device('cpu'),seeds=(1,2))
    shifted = evaluate_generation(model,batches(5.),torch.device('cpu'),seeds=(1,2))
    assert report['prediction_base'] == 'tfpp_clean_prior_v1'
    assert report['mean_over_seeds']['correction_from_prior_RMS_m'] == 0
    assert shifted['mean_over_seeds']['correction_from_prior_RMS_m'] == 0
    assert report['frozen_TFPP_prior_path_metrics']['route_ADE_m'] == 0
    assert shifted['frozen_TFPP_prior_path_metrics']['route_ADE_m'] == 5


def head_fixture(root):
    anchors = np.zeros((2, 11, 2), dtype=np.float32)
    anchors[:, :10, 0] = np.arange(10)+2.5
    anchors[1, :10, 1] = 2
    anchors[:, -1, 0] = [0, 6]
    np.save(root/'anchors.npy', anchors)
    np.savez(root/'stats.npz', mean=np.zeros((11, 2)), std=np.ones((11, 2)))
    config = NS(fmsr_anchor_path=str(root/'anchors.npy'),trajectory_stats_path=str(root/'stats.npz'),
                fmsr_num_poses=10,diffusion_speed=True,tf_d_model=256,tf_d_ffn=512,tf_num_head=8,
                tf_dropout=0.,bev_channels=4,min_x=-32,max_x=32,min_y=-32,max_y=32,
                classification_loss='cross_entropy',step_num=4,rollout_denoising_scenes=1)
    return PaperBridgePlanner(config)


def test_rollout_supervision_detaches_prefix_but_trains_selected_context(tmp_path):
    torch.set_num_threads(2)
    head = head_fixture(tmp_path).train()
    query = torch.randn(2,11,256,requires_grad=True)
    bev = torch.randn(2,4,3,3)
    target = head.plan_anchor.detach().clone()
    observations = []
    actual = head.denoise
    def tracked(state, anchors, t, queries, feature):
        observations.append((torch.is_grad_enabled(), state.requires_grad, queries.requires_grad))
        return actual(state, anchors, t, queries, feature)
    with patch.object(head, 'denoise', side_effect=tracked):
        loss, _ = head.rollout_training_loss(query, bev, target, torch.Generator().manual_seed(3))
    assert head.training
    assert all(o == (False, False, False) for o in observations[:-1])
    assert observations[-1] == (True, False, True)
    loss.backward()
    assert query.grad is not None and query.grad.abs().sum() > 0 and torch.isfinite(query.grad).all()
    assert head.plan_anchor.grad is None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.diff_decoder.parameters())


def test_paper_default_has_no_auxiliary_rollout_loss(tmp_path):
    torch.set_num_threads(2)
    head = head_fixture(tmp_path).train()
    head.config.rollout_denoising_weight = 0.
    with patch.object(head, 'rollout_training_loss', side_effect=AssertionError('unexpected auxiliary loss')):
        losses, _ = head.training_loss(torch.zeros(2,11,256), torch.zeros(2,4,3,3), head.plan_anchor.detach().clone())
    assert set(losses) == {'loss_denoising', 'loss_classification'}


def test_waypoint_filter_uses_vehicle_bearing_instead_of_segment_heading():
    state = np.zeros((11, 2), dtype=np.float32)
    state[:10, 0] = np.arange(10)+1
    state[1:10, 1] = 1
    # Second waypoint bearing is atan(1/2), while the first segment heads at 45 degrees.
    assert angles(state)[1] == pytest.approx(np.arctan2(1., 2.))
    assert angles(state, 'segment_heading')[1] == pytest.approx(np.pi/4)
    with pytest.raises(ValueError, match='angle definition'):
        angles(state, 'unknown')


def test_inactive_refiner_is_detected_and_reset_restores_gradient_flow(tmp_path):
    torch.set_num_threads(2)
    head = head_fixture(tmp_path).train()
    branch = head.diff_decoder.layers[1].task_decoder.plan_reg_branch
    with torch.no_grad():
        for index in (0, 2):
            branch[index].weight.zero_(); branch[index].bias.fill_(-1)
    query, bev = torch.randn(2,11,256), torch.randn(2,4,3,3)
    state = head.plan_anchor.detach().clone()[:, None]
    health = RegressionHealth(head)
    head.denoise(state, state, torch.full((2,), .5), query, bev)
    health.close()
    assert health.report()['layer1/relu1']['fully_inactive']
    assert health.report()['layer1/relu3']['fully_inactive']
    original_keys = set(head.state_dict())
    first_layer = {k: v.clone() for k,v in head.diff_decoder.layers[0].state_dict().items()}
    rng_before = torch.get_rng_state().clone()
    reset_regression_refiners(head, [1])
    assert torch.equal(rng_before, torch.get_rng_state())
    assert set(head.state_dict()) == original_keys
    for k,v in first_layer.items(): torch.testing.assert_close(head.diff_decoder.layers[0].state_dict()[k], v)
    output = head.denoise(state, state, torch.full((2,), .5), query, bev)
    torch.testing.assert_close(output[1], output[0])  # zero final projection initially passes through
    optimizer = torch.optim.SGD(branch.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = head.denoise(state, state, torch.full((2,), .5), query, bev)[-1]
        (output-state).abs().mean().backward()
        assert branch[4].weight.grad.abs().sum() > 0
        optimizer.step()
    assert branch[0].weight.grad.abs().sum() > 0
    with pytest.raises(ValueError): reset_regression_refiners(head, [2])


def test_geometry_supervision_preserves_true_curves_and_penalizes_local_jitter():
    target = torch.zeros(1,10,2); target[0, :, 0] = torch.arange(10)+2.5
    target[0, :, 1] = torch.linspace(0, 2, 10).square()
    assert PaperBridgePlanner.path_geometry_loss(target, target) == 0
    predicted = target.clone(); predicted[0, 4, 1] += 1; predicted.requires_grad_()
    loss = PaperBridgePlanner.path_geometry_loss(predicted, target)
    assert loss > 0
    loss.backward()
    assert predicted.grad[0, 4, 1] != 0


def test_warm_start_rejects_changed_assets_before_modifying_weights(tmp_path):
    model = torch.nn.Module()
    model.planner = torch.nn.Module()
    model.planner.plan_anchor = torch.nn.Parameter(torch.zeros(2,11,2), requires_grad=False)
    model.planner.register_buffer('mean', torch.zeros(1,1,11,2))
    model.planner.register_buffer('std', torch.ones(1,1,11,2))
    model.perception = torch.nn.Linear(2,2)
    path = tmp_path/'initial.pth'
    original = {k:v.clone() for k,v in model.state_dict().items()}
    torch.save(original, path)
    load_initial_checkpoint(path, model)
    altered = {k:v.clone() for k,v in original.items()}
    altered['planner.plan_anchor'].add_(1)
    altered['perception.weight'].zero_()
    torch.save(altered, path)
    with pytest.raises(ValueError, match='different training assets'):
        load_initial_checkpoint(path, model)
    for k,v in model.state_dict().items(): torch.testing.assert_close(v, original[k])
