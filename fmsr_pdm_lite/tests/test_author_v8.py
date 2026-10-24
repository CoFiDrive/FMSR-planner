"""Gradient and spatial contracts for the new PDM v8 adapter."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fmsrpdm.attention import PDMCrossBEVAttention
from fmsrpdm.planner import PaperBridgePlanner
from fmsrpdm.configuration import RunConfig
from fmsrpdm.model import FMSRModel


@pytest.mark.parametrize('author', [False, True])
def test_condition_gradient_boundary_and_backbone_bn_stay_frozen(author):
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm2d(3)

        def forward(self, rgb, lidar):
            fused = self.bn(rgb)
            return fused.new_zeros(len(rgb),64,64,64), fused, None

    class Join(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(256,256)

        def forward(self, query, tokens):
            return self.projection(query + tokens.mean(1,keepdim=True))

    model = FMSRModel.__new__(FMSRModel)
    torch.nn.Module.__init__(model)
    model.config = NS(two_tp_input=False, tp_attention=False,
                      min_x=-32,max_x=32,min_y=-32,max_y=32)
    model.author_v8 = author
    model.extra_sensors = False
    model.backbone = Backbone()
    model.change_channel = torch.nn.Conv2d(3,256,1)
    model.encoder_pos_encoding = lambda value: torch.zeros_like(value)
    model.join = Join()
    model.checkpoint_query = torch.nn.Parameter(torch.randn(1,10,256))
    model.requires_grad_(False)
    if author:
        model.change_channel.requires_grad_(True)
        model.join.requires_grad_(True)
        model.checkpoint_query.requires_grad_(True)
    model.navigation_encoder = torch.nn.Linear(2,256)
    model.planner = torch.nn.Linear(256,1)
    model.train()
    before = model.backbone.bn.running_mean.clone()
    query, bev, _ = model.context(torch.randn(2,3,8,8),torch.zeros(2,2,8,8),
                                  torch.randn(2,2),torch.zeros(2,1),torch.zeros(2,6))
    query.square().sum().backward()
    assert model.navigation_encoder.weight.grad.abs().sum() > 0
    for parameter in (model.change_channel.weight, model.join.projection.weight, model.checkpoint_query):
        if author:
            assert parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
    assert all(p.grad is None for p in model.backbone.parameters())
    assert not model.backbone.training
    torch.testing.assert_close(before, model.backbone.bn.running_mean)
    assert bev.shape == ((2,64,64,64) if author else (2,3,8,8))


def make_head(root, supervision='gt_each_layer'):
    torch.set_num_threads(2)
    anchors = np.zeros((2,11,2), dtype=np.float32)
    anchors[:,:10,0] = np.arange(10)+2.5
    anchors[1,:10,1] = 2
    anchors[:,-1,0] = [0,6]
    np.save(root/'anchors.npy', anchors)
    np.savez(root/'stats.npz', mean=np.zeros((11,2)), std=np.ones((11,2)))
    config = NS(fmsr_anchor_path=str(root/'anchors.npy'), trajectory_stats_path=str(root/'stats.npz'),
                fmsr_num_poses=10, diffusion_speed=True, tf_d_model=256, tf_d_ffn=512,
                tf_num_head=8, tf_dropout=0., bev_channels=64, min_x=-32,max_x=32,min_y=-32,max_y=32,
                classification_loss='focal', step_num=20, regression_supervision=supervision,
                implementation_profile='author_v8_pdm', fmsr_bev_include_speed=True)
    return PaperBridgePlanner(config)


def test_new_head_updates_both_refiners_and_accepts_64_channel_grid(tmp_path):
    torch.manual_seed(7)
    head = make_head(tmp_path).train()
    query = torch.randn(1,11,256,requires_grad=True)
    bev = torch.randn(1,64,64,64)
    original = head.plan_anchor.detach().clone()
    losses,_ = head.training_loss(query, bev, original[:1])
    sum(losses.values()).backward()
    assert query.grad.abs().sum() > 0
    for layer in head.diff_decoder.layers:
        assert layer.task_decoder.plan_reg_branch[-1].weight.grad.abs().sum() > 0
        assert layer.cross_bev_attention.value_proj[0].weight.shape[1] == 64
    assert head.plan_anchor.grad is None
    head.eval()
    result,_,trace = head.sample(query.detach(),bev,generator=torch.Generator().manual_seed(0),return_trace=True)
    assert len(trace)==20 and torch.isfinite(result).all()
    assert torch.equal(result[:,-1,1],torch.zeros(1))
    torch.testing.assert_close(original,head.plan_anchor)


def test_explicit_attachment_loss_repeats_last_selected_mode(tmp_path):
    head = make_head(tmp_path,'author_predicted_last')
    truth = head.plan_anchor[0].detach()[None]
    predictions = [torch.full((1,2,11,2),i+1.,requires_grad=True) for i in range(2)]
    logits = [torch.tensor([[0.,10.]],requires_grad=True) for _ in range(2)]
    with patch.object(head,'classify',return_value=logits),patch.object(head,'denoise',return_value=predictions):
        losses,_ = head.training_loss(torch.zeros(1,11,256),torch.zeros(1,64,64,64),truth)
    expected = 2*torch.nn.functional.l1_loss(predictions[-1][:,1],truth)
    torch.testing.assert_close(losses['loss_denoising'],expected)
    sum(losses.values()).backward()
    assert predictions[0].grad is None
    assert predictions[-1].grad[:,0].count_nonzero()==0
    assert predictions[-1].grad[:,1].count_nonzero()>0


def test_native_grid_samples_correct_axis_and_speed_holder_is_included():
    config = NS(min_x=-32,max_x=32,min_y=-32,max_y=32,tf_dropout=0.,
                fmsr_num_poses=10,fmsr_bev_include_speed=True)
    head = PDMCrossBEVAttention(256,8,in_bev_dims=64,num_points=11,config=config).eval()
    query = torch.zeros(1,1,256)
    points = torch.zeros(1,1,11,2)
    points[...,0] = 8
    points[...,1] = 16
    with patch('fmsrpdm.attention.F.grid_sample',return_value=torch.zeros(1,256,1,11)) as sampled:
        head(query,points,torch.zeros(1,64,64,64),(64,64))
    grid = sampled.call_args.args[1]
    torch.testing.assert_close(grid[0,0,0],torch.tensor([.25,.5]))
    assert grid.shape[-2]==11


def test_author_profile_rejects_clean_prior_and_implicit_legacy_loss():
    run = RunConfig('missing','missing','missing','missing',implementation_profile='author_v8_pdm',
                    prediction_base='tfpp_clean_prior_v1')
    with pytest.raises(ValueError,match='requires noisy_residual'):
        run.validate()
    run = RunConfig('missing','missing','missing','missing',regression_supervision='author_predicted_last')
    with pytest.raises(ValueError,match='requires author_v8_pdm'):
        run.validate()
