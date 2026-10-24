"""FM interpolation, gradient, Euler and checkpoint separation contracts."""
from types import SimpleNamespace as NS
from unittest.mock import patch

import json
import jsonpickle
import pytest
import torch

from test_author_v8 import make_head
from fmsrpdm.flow_matching import FlowMatchingPlanner
from fmsrpdm.checkpoint import load_initial_checkpoint
from fmsrpdm.configuration import RunConfig
from fmsrpdm import stage_evaluation
from scripts.pdm_lite.planner_telemetry import GenerationCapture
from scripts.pdm_lite.probe_eval_flow import restore, probe


def make_fm(root):
    vp = make_head(root)
    vp.config.trajectory_generator = 'flow_matching'
    vp.config.flow_train_t_max = .95
    vp.config.flow_min_remain = .05
    return FlowMatchingPlanner(vp.config)


def test_fm_clean_interpolation_and_both_layers_receive_gradient(tmp_path):
    torch.manual_seed(3)
    head = make_fm(tmp_path).train()
    query = torch.randn(1,11,256,requires_grad=True)
    bev = torch.randn(1,64,64,64)
    target = head.plan_anchor[:1].detach().clone()+.3
    target[:, -1, 1] = 0
    state, anchors, tau = head.flow_training_state(target,torch.Generator().manual_seed(8))
    torch.testing.assert_close(state,(1-tau[:,None,None,None])*anchors+tau[:,None,None,None]*target[:,None])
    assert 0 <= tau.item() < .95
    with patch.object(head.bridge,'add_noise',side_effect=AssertionError('FM must not add VP noise')):
        losses,_ = head.training_loss(query,bev,target)
    assert set(losses) == {'loss_flow_matching','loss_classification'}
    sum(losses.values()).backward()
    assert query.grad.abs().sum()>0
    for layer in head.diff_decoder.layers:
        assert layer.task_decoder.plan_reg_branch[-1].weight.grad.abs().sum()>0
    assert head.plan_anchor.grad is None


def test_velocity_loss_is_normalized_and_dummy_excluded(tmp_path):
    head = make_fm(tmp_path)
    target = head.plan_anchor[:1].detach().clone()
    state = head.plan_anchor[None].detach().clone()
    predictions = [state.clone().requires_grad_() for _ in range(2)]
    # GT mode is zero. A unit endpoint error / remain .5 / std 2 => unit velocity error.
    predictions = [p+1 for p in predictions]
    head.std.fill_(2)
    logits = [torch.zeros(1,2) for _ in range(2)]
    with patch.object(head,'flow_training_state',return_value=(state,state,torch.tensor([.5]))), \
         patch.object(head,'classify',return_value=logits),patch.object(head,'denoise',return_value=predictions):
        losses,_ = head.training_loss(torch.zeros(1,11,256),torch.zeros(1,64,64,64),target)
    torch.testing.assert_close(losses['loss_flow_matching'],torch.tensor(1.))


def test_euler_reaches_constant_endpoint_and_is_deterministic(tmp_path):
    head = make_fm(tmp_path).eval()
    query,bev = torch.zeros(1,11,256),torch.zeros(1,64,64,64)
    endpoint = head.plan_anchor[:1,None].detach().clone()+.5
    endpoint[..., -1,1] = 0
    with patch.object(head,'classify',return_value=[torch.tensor([[10.,0.]])]), \
         patch.object(head,'denoise',return_value=[endpoint]) as predict, \
         patch.object(head.bridge,'step',side_effect=AssertionError('FM must not step VP bridge')):
        result, mode, trace = head.sample(query,bev,steps=20,return_trace=True,
                                         generator=torch.Generator().manual_seed(1))
        other,_ = head.sample(query,bev,steps=20,generator=torch.Generator().manual_seed(7))
    torch.testing.assert_close(result,endpoint[:,0])
    torch.testing.assert_close(result,other,rtol=0,atol=0)
    assert len(trace)==20 and predict.call_count==40 and mode.item()==0
    assert trace[0]['tau']==0 and trace[-1]['tau']<1
    assert result[0,-1,1]==0


def test_vp_weights_cannot_silently_initialize_fm(tmp_path):
    fm = torch.nn.Module(); fm.planner = make_fm(tmp_path)
    weights = fm.state_dict()
    path = tmp_path/'model.pth'
    torch.save(weights,path)
    load_initial_checkpoint(path,fm)
    weights.pop('planner.flow_matching_contract')
    torch.save(weights,path)
    with pytest.raises(ValueError,match='architecture'):
        load_initial_checkpoint(path,fm)


def test_fm_rejects_invalid_profile_time_and_auxiliary_losses(tmp_path):
    values = dict(data_root=str(tmp_path),assets_dir=str(tmp_path),tfpp_checkpoint=str(tmp_path),
                  output_dir=str(tmp_path),trajectory_generator='flow_matching',implementation_profile='author_v8_pdm')
    RunConfig(**values).validate()
    for extra in ({'implementation_profile':'legacy_vp'}, {'flow_train_t_max':1.},
                  {'flow_min_remain':.1}, {'rollout_denoising_weight':1.},
                  {'regression_supervision':'author_predicted_last'}):
        with pytest.raises(ValueError):
            RunConfig(**(values|extra)).validate()


def test_fm_evaluation_versions_generator_and_sampler(tmp_path,monkeypatch):
    checkpoint = tmp_path/'model_0009.pth'; checkpoint.write_bytes(b'test')
    config = NS(fmsr_trajectory_head='author_v8_pdm_fm_only_v1',
                implementation_profile='author_v8_pdm',trajectory_generator='flow_matching',
                regression_supervision='gt_each_layer',prediction_base='noisy_residual',
                bev_channels=64,step_num=20)
    (tmp_path/'config.json').write_text(jsonpickle.encode(config))
    routes = tmp_path/'routes.xml'; routes.write_text('<routes><route id="1"/></routes>')
    output = tmp_path/'evaluation'
    monkeypatch.setattr('sys.argv',['stage','--checkpoint',str(checkpoint),'--routes',str(routes),'--output',str(output)])
    stage_evaluation.main()
    source = json.loads((output/'source.json').read_text())
    assert source['trajectory_generator']=='flow_matching' and not source['CR_enabled']
    assert source['bev_channels']==64 and source['sampler']=='deterministic_Euler_tau_0_to_1'


def test_fm_full_telemetry_preserves_output_and_saved_probe_replays(tmp_path):
    head = make_fm(tmp_path).eval()
    head.config.step_num = 4
    query,bev = torch.randn(1,11,256),torch.randn(1,64,64,64)
    expected,_ = head.sample(query,bev)
    rng = torch.get_rng_state().clone()
    capture = GenerationCapture(1,trace_every=1,snapshot=True)
    head._eval_capture = capture
    capture.scene(head,query,bev,(64,64))
    actual,_ = head.sample(query,bev)
    capture.finish()
    torch.testing.assert_close(expected,actual,rtol=0,atol=0)
    assert torch.equal(rng,torch.get_rng_state())
    report = capture.report()
    assert report['nfe']==4 and len(report['integration_trace'])==4
    path = tmp_path/'model.pth'
    torch.save({'planner.'+k:v for k,v in head.state_dict().items()},path)
    restored = restore(dict(capture.scene_snapshot,checkpoint=str(path)),'cpu')
    result,_ = restored.sample(query,bev)
    torch.testing.assert_close(result,expected,rtol=0,atol=0)
    diagnostic = probe(restored,query,bev,(64,64),scale=0,jacobian_every=0)
    assert diagnostic['fixed_mode_final_difference']==0
