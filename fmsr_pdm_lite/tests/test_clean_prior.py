from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from fmsrpdm.checkpoint import load_initial_checkpoint
from fmsrpdm.clean_prior import TFPP_CLEAN_PRIOR, correction_head
from fmsrpdm.planner import PaperBridgePlanner
from fmsrpdm.model import FMSRModel
from fmsrpdm.configuration import RunConfig
from fmsrpdm.train import best_generation_before
import json


def planner(root):
    torch.set_num_threads(2)
    anchors = np.zeros((2, 11, 2), dtype=np.float32)
    anchors[:, :10, 0] = np.arange(10)+2.5
    anchors[1, :10, 1] = 2
    anchors[:, -1, 0] = [0, 6]
    np.save(root/'anchors.npy', anchors)
    np.savez(root/'stats.npz', mean=np.zeros((11, 2)), std=np.ones((11, 2)))
    config = NS(fmsr_anchor_path=str(root/'anchors.npy'), trajectory_stats_path=str(root/'stats.npz'),
                fmsr_num_poses=10, diffusion_speed=True, tf_d_model=256, tf_d_ffn=512, tf_num_head=8,
                tf_dropout=.1, bev_channels=4, min_x=-32, max_x=32, min_y=-32, max_y=32,
                classification_loss='cross_entropy', step_num=20, rollout_denoising_scenes=1,
                prediction_base=TFPP_CLEAN_PRIOR, training_scope='clean_path_only')
    return PaperBridgePlanner(config)


def test_zero_initialization_outputs_prior_through_full_noisy_sampler(tmp_path):
    head = planner(tmp_path).eval()
    query, bev = torch.randn(2,11,256), torch.randn(2,4,3,3)
    prior = torch.randn(2,10,2)
    outputs = []
    for seed in (137, 271, 811):
        output, mode, trace = head.sample(query, bev, clean_prior=prior,
            generator=torch.Generator().manual_seed(seed), return_trace=True)
        torch.testing.assert_close(output[:, :10], prior, atol=1e-5, rtol=1e-5)
        assert len(trace) == 20
        for step in trace:
            torch.testing.assert_close(step['clean_prediction'], prior)
        assert output[:, -1, 1].count_nonzero() == 0
        outputs.append(trace[1]['state'])
    # Genuine initial Gaussian noise is retained even though the clean prediction is stable.
    assert not torch.equal(outputs[0], outputs[1])
    with pytest.raises(ValueError, match=r'requires the frozen TF\+\+ path'):
        head.sample(query, bev)


def test_clean_path_only_freezes_old_heads_and_restores_new_head_gradients(tmp_path):
    head = planner(tmp_path).train()
    assert head.training and head.clean_path_heads.training
    assert not head.diff_decoder.training and not head.diff_decoder_cls.training
    assert all(p.requires_grad == name.startswith('clean_path_heads.') for name,p in head.named_parameters())
    initial = {k:v.clone() for k,v in head.state_dict().items()}
    query, bev = torch.randn(2,11,256), torch.randn(2,4,3,3)
    target = head.plan_anchor.detach().clone()
    prior = (target[:, :10]+.2).clone().requires_grad_(True)
    optimizer = torch.optim.AdamW(head.clean_path_heads.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        losses, _ = head.training_loss(query, bev, target, clean_prior=prior,
                                       generator=torch.Generator().manual_seed(137))
        assert set(losses) == {'loss_denoising'}
        sum(losses.values()).backward()
        assert prior.grad is None
        assert head.clean_path_heads[0][-1].weight.grad.abs().sum() > 0
        optimizer.step()
    assert head.clean_path_heads[0][1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for name,p in head.named_parameters() if not name.startswith('clean_path_heads.'))
    for name,value in head.state_dict().items():
        if not name.startswith('clean_path_heads.'):
            assert torch.equal(value, initial[name])


def test_explicit_migration_preserves_assets_and_old_parameters(tmp_path):
    model = torch.nn.Module(); model.planner = torch.nn.Module()
    model.planner.register_buffer('plan_anchor', torch.zeros(2,11,2))
    model.planner.register_buffer('mean', torch.zeros(1,1,11,2))
    model.planner.register_buffer('std', torch.ones(1,1,11,2))
    model.planner.register_buffer('clean_prior_contract', torch.tensor(1))
    model.planner.clean_path_heads = torch.nn.ModuleList([correction_head(4,10)])
    model.frozen = torch.nn.Linear(2,2)
    current = {k:v.clone() for k,v in model.state_dict().items()}
    old = {k:v.clone() for k,v in current.items() if not k.startswith('planner.clean_path_heads.')
           and k != 'planner.clean_prior_contract'}
    old['frozen.weight'].fill_(3)
    path = tmp_path/'old.pth'; torch.save(old, path)
    with pytest.raises(ValueError, match='architecture'):
        load_initial_checkpoint(path, model)
    assert torch.equal(model.frozen.weight, current['frozen.weight'])
    result = load_initial_checkpoint(path, model, migrate_noisy_residual=True)
    assert result['migration'] == 'noisy_residual_to_tfpp_clean_prior_v1'
    for name,value in model.state_dict().items():
        assert torch.equal(value, old.get(name, current[name]))
    original = {k:v.clone() for k,v in model.state_dict().items()}
    old['planner.plan_anchor'].add_(1); old['frozen.weight'].zero_(); torch.save(old,path)
    with pytest.raises(ValueError, match='different training assets'):
        load_initial_checkpoint(path, model, migrate_noisy_residual=True)
    for name,value in model.state_dict().items(): assert torch.equal(value, original[name])


def test_clean_prior_rollout_and_geometry_train_without_updating_speed_branch(tmp_path):
    head = planner(tmp_path).train()
    head.config.rollout_denoising_weight = 1.
    head.config.rollout_geometry_weight = .5
    head.config.path_geometry_weight = .5
    target = head.plan_anchor.detach().clone()
    losses, metrics = head.training_loss(torch.randn(2,11,256), torch.randn(2,4,3,3), target,
        clean_prior=target[:, :10]+.1, generator=torch.Generator().manual_seed(137))
    assert set(losses) == {'loss_denoising', 'loss_path_geometry', 'loss_rollout_denoising'}
    sum(losses.values()).backward()
    assert head.training and not head.diff_decoder.training
    assert head.clean_path_heads[0][-1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for name,p in head.named_parameters() if not name.startswith('clean_path_heads.'))


def test_frozen_prior_precision_is_independent_of_outer_amp_and_restores_flags():
    observations = []
    def context(rgb, lidar, tp, velocity, command, next_tp, prior_requested):
        value = rgb @ lidar
        observations.append((prior_requested, value.dtype))
        return value
    fake_model = NS(_context=context)
    rgb, lidar = torch.randn(4,4), torch.randn(4,4)
    inputs = (rgb, lidar, torch.zeros(4,2), torch.zeros(4,1), torch.zeros(4,6), None)
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        prior = FMSRModel.context(fake_model,*inputs,return_clean_prior=True)
        legacy = FMSRModel.context(fake_model,*inputs)
    torch.testing.assert_close(prior,rgb @ lidar)
    assert observations == [(True,torch.float32),(False,torch.bfloat16)]
    assert legacy.dtype == torch.bfloat16
    assert before == (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)


def test_frozen_experiment_rejects_untrained_random_speed_and_classifier():
    run = RunConfig(data_root='',assets_dir='',tfpp_checkpoint='',output_dir='',
                    prediction_base='tfpp_clean_prior_v1',training_scope='clean_path_only')
    with pytest.raises(ValueError,match='trained FMSR initialization'):
        run.validate()


def test_resume_selection_uses_generation_and_excludes_partial_and_future_epochs(tmp_path):
    for epoch, ade, denoising, complete in ((0,.4,.01,True),(1,.3,.1,True),(2,.001,.001,False),(3,.002,.002,True)):
        summary = {'epoch':epoch,'completed_full_epoch':complete,'validation':{'route_ADE':denoising},
                   'validation_generation':{'report':f'generation_epoch_{epoch:04d}.json',
                       'mean_over_seeds':{'route_ADE_m':ade,'speed_MAE_mps':.2}}}
        (tmp_path/f'epoch_{epoch:04d}.json').write_text(json.dumps(summary))
    chosen = best_generation_before(tmp_path,3)
    assert chosen['epoch'] == 1 and chosen['checkpoint'] == 'model_0001.pth'
    assert chosen['route_ADE_m'] == .3
    assert best_generation_before(tmp_path,0) is None
