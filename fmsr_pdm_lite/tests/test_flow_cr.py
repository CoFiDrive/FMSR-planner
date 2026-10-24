"""Full spectral CR, delayed scheduling, dropout and architecture contracts."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import json
import jsonpickle

import pytest
import torch

from test_flow_matching import make_fm
from fmsrpdm.flow_matching import FlowMatchingPlanner
from fmsrpdm.flow_cr import spectral_contract, contractive_loss, deterministic_decoder
from fmsrpdm.configuration import RunConfig
from fmsrpdm.checkpoint import load_initial_checkpoint
from fmsrpdm import stage_evaluation


def make_cr(root):
    config = make_fm(root).config
    config.flow_contract_enable = True
    config.flow_contract_weight = .01
    config.flow_contract_fraction = 1.
    config.flow_contract_interval = 1
    config.flow_contract_fd_epsilon = .1
    config.flow_contract_fd_chunk = 2
    config.flow_contract_delay_epochs = 0.
    config.flow_contract_ramp_epochs = 0.
    config.flow_contract_tau_kappa = [.1,.1,.1,.1]
    return FlowMatchingPlanner(config)


def test_spectral_jacobian_matches_known_linear_map_and_has_parameter_gradient():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = NS(fmsr_num_poses=10,flow_contract_fd_epsilon=.1,
                             flow_contract_fd_chunk=3,flow_contract_tau_kappa=[.98]*4)
            self.std = torch.ones(1,1,11,2)
            self.weight = torch.nn.Parameter(torch.diag(torch.linspace(.2,1.5,20)))
            self.diff_decoder = torch.nn.Identity()
        def predict_flow_endpoint(self,state,tau,anchors,bev,spatial,query):
            path = (state[...,:10,:].flatten(-2) @ self.weight.T).reshape(*state.shape[:2],10,2)
            return torch.cat([path,state[...,-1:,:]],dim=-2)

    planner = Linear()
    raw,sigma,_ = spectral_contract(planner,torch.randn(1,11,2),torch.zeros(1,11,2),
                                    torch.tensor([.5]),torch.zeros(1,11,256),torch.zeros(1,64,4,4))
    torch.testing.assert_close(sigma,torch.tensor([1.5]),rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(raw,torch.tensor((1.5-.98)**2),rtol=1e-5,atol=1e-5)
    raw.backward()
    assert torch.isfinite(planner.weight.grad).all() and planner.weight.grad.abs().sum()>0


def test_cr_delay_interval_eval_and_progress_are_not_driven_by_validation(tmp_path):
    planner = make_cr(tmp_path)
    planner.config.flow_contract_delay_epochs=2.
    planner.config.flow_contract_ramp_epochs=1.
    planner.config.flow_contract_interval=4
    state = torch.zeros(1,2,11,2);tau=torch.tensor([.5]);mode=torch.tensor([0])
    query,bev=torch.zeros(1,11,256),torch.zeros(1,64,64,64)
    with patch('fmsrpdm.flow_cr.spectral_contract',return_value=(torch.tensor(2.),torch.tensor([1.5]),torch.tensor([1.]))) as call:
        planner.set_training_progress(199,100)
        assert contractive_loss(planner,state,state,tau,mode,query,bev)[1]['flow_contract_active']==0
        planner.set_training_progress(249,100)
        assert contractive_loss(planner,state,state,tau,mode,query,bev)[1]['flow_contract_active']==0
        planner.set_training_progress(248,100)
        loss,metrics=contractive_loss(planner,state,state,tau,mode,query,bev)
        torch.testing.assert_close(loss,torch.tensor(2*.01*.49*4))
        assert metrics['flow_contract_active']==1 and call.call_count==1
        planner.eval()
        assert contractive_loss(planner,state,state,tau,mode,query,bev)[1]['flow_contract_active']==0
        assert planner._training_step==248 and call.call_count==1


def test_cr_pair_disables_attention_dropout_and_restores_modes(tmp_path):
    planner = make_cr(tmp_path).train()
    for module in planner.diff_decoder.modules():
        if isinstance(module,torch.nn.Dropout): module.p=.5
        if isinstance(module,torch.nn.MultiheadAttention): module.dropout=.5
    query,bev=torch.randn(1,11,256),torch.randn(1,64,64,64)
    anchor=planner.plan_anchor[:1,None]
    modes=[module.training for module in planner.diff_decoder.modules()]
    rng=torch.get_rng_state().clone()
    with deterministic_decoder(planner):
        a=planner.predict_flow_endpoint(anchor,torch.tensor([.5]),anchor,bev,(64,64),query)
        b=planner.predict_flow_endpoint(anchor,torch.tensor([.5]),anchor,bev,(64,64),query)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    assert modes==[module.training for module in planner.diff_decoder.modules()]


def test_active_real_cr_backward_and_sampler_are_identical_to_fm(tmp_path):
    torch.manual_seed(9)
    planner=make_cr(tmp_path).train()
    query=torch.randn(1,11,256,requires_grad=True);bev=torch.randn(1,64,64,64)
    losses,metrics=planner.training_loss(query,bev,planner.plan_anchor[:1].detach())
    assert metrics['flow_contract_active']==1 and losses['loss_contractive']>0
    losses['loss_contractive'].backward()
    assert torch.isfinite(query.grad).all() and query.grad.abs().sum()>0
    assert planner.diff_decoder.layers[-1].task_decoder.plan_reg_branch[-1].weight.grad.abs().sum()>0
    planner.eval()
    baseline=make_fm(tmp_path).eval()
    baseline.load_state_dict({k:v for k,v in planner.state_dict().items() if k!='flow_cr_contract'},strict=True)
    actual,_=planner.sample(query.detach(),bev,steps=4)
    expected,_=baseline.sample(query.detach(),bev,steps=4)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_cr_requires_fm_and_valid_hyperparameters(tmp_path):
    values=dict(data_root=str(tmp_path),assets_dir=str(tmp_path),tfpp_checkpoint=str(tmp_path),
                output_dir=str(tmp_path),implementation_profile='author_v8_pdm',
                trajectory_generator='flow_matching',flow_contract_enable=True)
    RunConfig(**values).validate()
    for change in ({'trajectory_generator':'ddbm'},{'flow_contract_interval':0},
                   {'flow_contract_fd_epsilon':0},{'flow_contract_fd_chunk':21},
                   {'flow_contract_fraction':0},{'flow_contract_tau_kappa':[1.]},
                   {'flow_contract_delay_epochs':float('nan')}):
        with pytest.raises(ValueError): RunConfig(**(values|change)).validate()


def test_cr_checkpoint_rejects_fm_only_and_preserves_cr_marker(tmp_path):
    planner = make_cr(tmp_path)
    model=torch.nn.Module();model.planner=planner
    path=tmp_path/'model.pth'
    torch.save(model.state_dict(),path)
    load_initial_checkpoint(path,model)
    weights=model.state_dict();weights.pop('planner.flow_cr_contract')
    torch.save(weights,path)
    with pytest.raises(ValueError,match='architecture'): load_initial_checkpoint(path,model)


def test_cr_evaluation_metadata_uses_same_bev_and_sampler(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.pth';checkpoint.write_bytes(b'test')
    config=NS(fmsr_trajectory_head='author_v8_pdm_fm_cr_v1',
        implementation_profile='author_v8_pdm',trajectory_generator='flow_matching',
        regression_supervision='gt_each_layer',prediction_base='noisy_residual',bev_channels=64,
        step_num=20,flow_contract_enable=True,flow_contract_method='endpoint_jacobian',flow_contract_weight=.01)
    (tmp_path/'config.json').write_text(jsonpickle.encode(config))
    routes=tmp_path/'routes.xml';routes.write_text('<routes><route id="1"/></routes>')
    output=tmp_path/'evaluation'
    monkeypatch.setattr('sys.argv',['stage','--checkpoint',str(checkpoint),'--routes',str(routes),'--output',str(output)])
    stage_evaluation.main()
    source=json.loads((output/'source.json').read_text())
    assert source['CR_enabled'] and source['CR_method']=='endpoint_jacobian'
    assert source['bev_channels']==64 and source['sampler']=='deterministic_Euler_tau_0_to_1'
