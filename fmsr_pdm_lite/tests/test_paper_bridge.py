from pathlib import Path
from types import SimpleNamespace as NS
import json
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from fmsrpdm.schedule import VPBridge
from fmsrpdm.attention import ego_to_grid
from fmsrpdm.assets import split_routes, group_key
from fmsrpdm.planner import PaperBridgePlanner
from fmsrpdm.vendor_ddbm.model_diffusion_head_ddbm import DDBMScheduler
from fmsrpdm.checkpoint import load_resume, save_checkpoint


@pytest.fixture(autouse=True)
def low_cpu_threads():
    torch.set_num_threads(2)


def planner_fixture(root):
    anchors = np.zeros((2,11,2),dtype=np.float32)
    anchors[:, :10, 0]=np.arange(10)+2.5
    anchors[1,:10,1]=4
    anchors[:, -1, 0]=[0,6]
    np.save(root/'anchors.npy',anchors)
    np.savez(root/'stats.npz',mean=np.zeros((11,2)),std=np.ones((11,2)))
    c=NS(fmsr_anchor_path=str(root/'anchors.npy'),trajectory_stats_path=str(root/'stats.npz'),
         fmsr_num_poses=10,diffusion_speed=True,tf_d_model=256,tf_d_ffn=512,tf_num_head=8,
         tf_dropout=0.,bev_channels=4,min_x=-32,max_x=32,min_y=-16,max_y=16,
         classification_loss='cross_entropy',step_num=20)
    return PaperBridgePlanner(c)


def test_coefficients_match_upstream_and_exact_endpoints():
    scheduler=VPBridge()
    t=torch.linspace(.01,.99,20)
    actual=scheduler.coefficients(t)
    expected=DDBMScheduler().get_abc(t)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    a,b,c=scheduler.coefficients(torch.tensor([0.,1.]))
    torch.testing.assert_close(a,torch.tensor([0.,1.]))
    torch.testing.assert_close(b,torch.tensor([1.,0.]))
    torch.testing.assert_close(c,torch.zeros(2))


def test_oracle_last_step_recovers_clean_state_without_nan():
    scheduler=VPBridge();clean=torch.randn(2,1,11,2);anchor=torch.randn_like(clean)
    t=torch.tensor([.4,.8]);noise=torch.randn_like(clean)
    state=scheduler.add_noise(clean,anchor,t,noise)
    result=scheduler.step(state,clean,anchor,t,torch.zeros(2))
    torch.testing.assert_close(result,clean)
    interior=scheduler.step(anchor,clean,anchor,torch.ones(2),t,noise)
    torch.testing.assert_close(interior,state)


def test_pdm_grid_axes_and_asymmetric_bounds():
    config=NS(min_x=-10,max_x=30,min_y=-20,max_y=60)
    points=torch.tensor([[-10.,-20.],[30.,60.],[10.,20.]])
    torch.testing.assert_close(ego_to_grid(points,config),torch.tensor([[-1.,-1.],[1.,1.],[0.,0.]]))


def test_repetition_groups_stay_in_one_split():
    routes=[Path('/data/X/Town12_Rep0_10_0_route0'),Path('/data/X/Town12_Rep1_10_1_route0'),
            Path('/data/Y/Town12_Rep0_20_0_route0'),Path('/data/Z/Town01_Rep0_30_0_route0')]
    train,validation=split_routes(routes,.4,0)
    assert not {group_key(x) for x in train}&{group_key(x) for x in validation}


def test_regression_conditions_on_nearest_anchor_instead_of_predicted_class(tmp_path):
    head=planner_fixture(tmp_path)
    target=head.plan_anchor[0].detach()[None]
    logits=torch.tensor([[0.,10.]],requires_grad=True)  # deliberately choose the wrong class
    predictions=head.plan_anchor[None].clone().requires_grad_(True)
    with patch.object(head,'classify',return_value=[logits]),patch.object(head,'denoise',return_value=[predictions]):
        losses,metrics=head.training_loss(torch.zeros(1,11,256),torch.zeros(1,4,2,2),target)
    assert losses['loss_denoising'].item()==0
    assert metrics['anchor_accuracy'].item()==0
    assert losses['loss_classification'].item()>9


def test_training_backward_fixed_anchors_and_20_step_sampling(tmp_path):
    torch.manual_seed(0)
    head=planner_fixture(tmp_path).train()
    queries=torch.randn(2,11,256,requires_grad=True)
    bev=torch.randn(2,4,3,3)
    original=head.plan_anchor.detach().clone()
    target=original[[0,1]].clone()
    losses,_=head.training_loss(queries,bev,target)
    sum(losses.values()).backward()
    assert queries.grad is not None and torch.isfinite(queries.grad).all()
    assert head.plan_anchor.grad is None
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())
    head.eval()
    with patch.object(head,'classify',wraps=head.classify) as classifier:
        prediction,mode=head.sample(queries.detach(),bev,steps=20,generator=torch.Generator().manual_seed(1))
    assert classifier.call_count==1
    assert prediction.shape==(2,11,2) and torch.isfinite(prediction).all()
    assert torch.equal(prediction[:,-1,1],torch.zeros(2))
    assert torch.equal(original,head.plan_anchor)


def test_checkpoint_resume_preserves_parameters_optimizer_scheduler_rng(tmp_path):
    model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,10,T_mult=2)
    model(torch.ones(1,2)).sum().backward();optimizer.step();scheduler.step(1)
    saved={k:v.clone() for k,v in model.state_dict().items()}
    save_checkpoint(model,optimizer,scheduler,0,7,tmp_path,'assets','configuration')
    expected=torch.rand(3)
    with torch.no_grad():model.weight.zero_()
    epoch,step=load_resume(tmp_path/'resume_0000.pt',model,optimizer,scheduler,'assets','configuration')
    assert (epoch,step)==(1,7)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,saved[k])
    torch.testing.assert_close(torch.rand(3),expected)
    with pytest.raises(ValueError):load_resume(tmp_path/'resume_0000.pt',model,optimizer,scheduler,'changed','configuration')
