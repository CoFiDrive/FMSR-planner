"""Logging must preserve policy math and support real metric/replay analysis."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch

from scripts.pdm_lite.eval_metrics import EvalMetricLogger, rebuild_metric_info
from scripts.pdm_lite.planner_telemetry import GenerationCapture
from scripts.pdm_lite.probe_eval_flow import probe, restore
from scripts.pdm_lite.analyze_planning_metrics import spatial_deviation, world_path, analyze
from scripts.pdm_lite.summarize_eval_metrics import summarize
from scripts.pdm_lite.probe_flow_state import state_probe
from scripts.pdm_lite.compare_flow_robustness import compare, plot
from scripts.pdm_lite.probe_condition_series import temporal_probe
from fmsr.tfv6.diffusion_modules.model_flow_matching_only import TrajectoryHead
from fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm import TrajectoryHead as DiffusionHead

ROOT = Path(__file__).resolve().parents[1]


def make_head(tmp_path):
    torch.set_num_threads(2)
    anchors = np.zeros((2, 11, 2), dtype=np.float32)
    anchors[:, :10, 0] = np.arange(10)+2.5
    anchors[1, :10, 1] = 1
    anchors[:, -1, 0] = 5
    np.save(tmp_path/'anchors.npy', anchors)
    cfg = NS(trajectory_generator='residual_flow', diffusion_speed=True, step_num=4,
             flow_train_t_max=.95, flow_min_remain=.05, flow_inference_steps=4,
             flow_contract_enable=False, tf_d_model=256, tf_d_ffn=128, tf_num_head=8,
             tf_dropout=0., lidar_max_x=32., lidar_max_y=32., fmsr_num_poses=10,
             fmsr_anchor_path=str(tmp_path/'anchors.npy'))
    return TrajectoryHead(10, 128, 256, cfg.fmsr_anchor_path, cfg).eval()


def test_real_fm_capture_preserves_output_and_snapshot_replays(tmp_path):
    head = make_head(tmp_path)
    ego = torch.randn(1, 10, 256)*.1
    bev = torch.randn(1, 1512, 2, 2)*.1
    with torch.no_grad():
        expected = head(ego, bev, (2,2))
        rng_before = torch.get_rng_state().clone()
        capture = GenerationCapture(20, trace_every=20, snapshot=True)
        head._eval_capture = capture
        capture.scene(head, ego, bev, (2,2))
        actual = head(ego, bev, (2,2))
        capture.finish()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(rng_before, torch.get_rng_state())
    report = capture.report()
    assert report['nfe'] == 4 and len(report['integration_trace']) == 4
    torch.testing.assert_close(report['integration_trace'][-1]['next_state'], actual)
    path = tmp_path/'weights.pth'
    torch.save({'fmsr_head.'+k:v for k,v in head.state_dict().items()}, path)
    (tmp_path/'config.json').write_text('{}')
    class Owner(torch.nn.Module):
        def __init__(self):
            super().__init__();self.fmsr_head=head;self._eval_checkpoint=str(path)
    logger=EvalMetricLogger(tmp_path,'config+metadata_check')
    logger.set_run_metadata(head.config,[Owner()],trace_every=20)
    assert len(logger.metadata['code_sha256'])>=4
    assert logger.metadata['checkpoints'][0]['sha256']
    logger.close()
    snapshot = dict(capture.scene_snapshot, checkpoint=str(path))
    restored = restore(snapshot, 'cpu')
    with torch.no_grad():
        torch.testing.assert_close(restored(ego,bev,(2,2)),expected,rtol=0,atol=0)
    diagnostic = probe(restored,ego,bev,(2,2),scale=.001,jacobian_every=2)
    assert diagnostic['numerical_bound_holds']
    assert len(diagnostic['stage_metrics']) == 4
    assert diagnostic['stage_metrics'][0]['route_sigma_max'] is not None
    zero = probe(restored,ego,bev,(2,2),scale=0,jacobian_every=0)
    assert zero['fixed_mode_final_difference'] == 0


def test_spatial_alignment_removes_ego_motion_and_detects_changed_path():
    metric = {'location':[0,0,0], 'rotation':[0,0,0]}
    before = world_path([[2.5,0],[5,0],[10,0]], metric)
    after = world_path([[2.5,0],[5,0],[10,0]], dict(metric,location=[1,0,0]))
    assert spatial_deviation(before,after)['mean_m'] < 1e-8
    shifted = after.copy();shifted[:,1] += 1
    assert spatial_deviation(before,shifted)['mean_m'] > .9


def test_stochastic_diffusion_capture_does_not_change_random_sampling(tmp_path):
    cfg = make_head(tmp_path).config
    cfg.trajectory_generator = 'ddbm'
    head = DiffusionHead(10,128,256,cfg.fmsr_anchor_path,cfg).eval()
    ego,bev = torch.randn(1,10,256)*.1,torch.randn(1,1512,2,2)*.1
    with torch.no_grad():
        torch.manual_seed(123)
        expected = head(ego,bev,(2,2))
        torch.manual_seed(123)
        capture=GenerationCapture(20,20)
        head._eval_capture=capture
        capture.scene(head,ego,bev,(2,2))
        actual=head(ego,bev,(2,2))
        capture.finish()
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert capture.report()['nfe']==4


class Actor:
    def get_world(self): return self
    def get_snapshot(self): return NS(frame=self.frame, timestamp=NS(elapsed_seconds=self.frame*.05))
    def get_velocity(self): return NS(x=1,y=0,z=0)
    def get_control(self): return NS(steer=0,throttle=0,brake=1)


def test_metrics_recovery_official_coverage_and_bfloat_logging(tmp_path):
    actor = Actor()
    metric = dict(acceleration=[0,0,0], angular_velocity=[0,0,0], forward_vector=[1,0,0],
                  right_vector=[0,1,0], location=[0,0,0], rotation=[0,0,0])
    logger = EvalMetricLogger(tmp_path,'config+old+actual_save_name')
    for i in range(81):
        actor.frame=i+1
        logger.record(i,(i+1)*.05,metric,actor)
        logger.record_plan(i,(i+1)*.05,{'phase':'inference',
                           'predicted_trajectory_ego_m':torch.tensor([[2.5,0],[5,0],[10,0]],dtype=torch.bfloat16)},
                           actor.get_control(),actor.get_control())
    logger.close()
    directory=logger.directory
    assert len(json.loads((directory/'metric_info.json').read_text()))==41
    (directory/'metric_info.json').write_text('{}')
    assert rebuild_metric_info(directory)==41
    result=tmp_path/'merge.json'
    records=[{'save_name':directory.name,'route_id':'r','status':'Completed',
              'infractions':{'min_speed_infractions':['Speed 120%','Speed 180%','Speed 1001%']}}]
    result.write_text(json.dumps({'_checkpoint':{'records':records}}))
    benchmark=ROOT/'3rd_party/Bench2Drive/tools/efficiency_smoothness_benchmark.py'
    report=summarize(result,directory.parent,benchmark)
    assert report['efficiency_percent']==150 and report['comfort_fraction']==1
    analysis=analyze(directory)
    assert analysis['longitudinal_jerk_mps3']['max_abs']==0
    assert analysis['spatial_plan_deviation_m']['mean']==0
    records.append(dict(records[0],save_name='missing'))
    result.write_text(json.dumps({'_checkpoint':{'records':records}}))
    assert summarize(result,directory.parent,benchmark)['comfort_fraction'] is None
    retry=EvalMetricLogger(tmp_path,'config+actual_save_name');retry.close()
    assert list((directory/'attempts').glob('*/planning.jsonl'))


def test_probe_recovers_known_linear_jacobian_and_bound():
    class LinearHead:
        config=NS(fmsr_num_poses=1, flow_contract_tau_kappa=None)
        plan_anchor=torch.tensor([[[2.,0.],[3.,0.]]])
        std=torch.ones(1,1,2,2)
        def _classify_flow_anchors(self,anchors,bev,spatial,ego):return [torch.ones(1,1)]
        def get_flow_schedule(self,device=None):return torch.linspace(0,1,5,device=device)
        def predict_flow_endpoint(self,state,tau,anchor,bev,spatial,ego,img):
            return .5*state + .2*ego.mean()
        def _mask_flow_speed_padding(self,state):
            out=state.clone();out[...,-1,1]=0;return out
    result=probe(LinearHead(),torch.ones(1,1,2),torch.zeros(1,1,1,1),(1,1),
                 scale=.1,jacobian_every=1,epsilon=.01)
    assert result['numerical_bound_holds']
    for row in result['stage_metrics']:
        assert abs(row['route_sigma_max']-.5)<1e-4
        assert row['d_next']<=row['B_obs_next']+1e-6
    fixed=probe(LinearHead(),torch.ones(1,1,2),torch.zeros(1,1,1,1),(1,1),
                scale=.1,jacobian_every=1,epsilon=.01,freeze_speed=True,segment_points=3)
    assert fixed['B_CR_quadrature_estimate']>=fixed['fixed_mode_final_difference']-1e-6
    response=state_probe(LinearHead(),torch.ones(1,1,2),torch.zeros(1,1,1,1),(1,1),
                         magnitude=.01,directions=2,freeze_speed=True)
    for trial in response['trials']:
        j=trial['injection_step'];alpha=1/(4-j)
        assert abs(trial['chi_j']-(1-.5*alpha))<2e-5
        assert trial['terminal_gain']<=trial['chi_j']+1e-5


def test_paired_common_scene_zero_condition_and_heatmap_export(tmp_path):
    import copy
    head=make_head(tmp_path)
    ego,bev=torch.randn(1,10,256)*.1,torch.randn(1,1512,2,2)*.1
    report=compare({'FM':head,'FM_CR':copy.deepcopy(head)},ego,bev,(2,2),
                   condition_scales=[0,.01],state_magnitudes=[.01],seeds=[0],directions=1)
    for trial in report['condition_trials']:
        a,b=trial['models']['FM'],trial['models']['FM_CR']
        assert a['WMD']==b['WMD'] and a['MSD']==b['MSD']
        if trial['scale']==0:assert a['WMD']==0 and a['selected_output_difference']==0
    a,b=report['state_trials'][0]['models'].values()
    assert a['injection_states']==b['injection_states']
    assert a['propagation_gain_mean']==b['propagation_gain_mean']
    plot(report,tmp_path)
    assert (tmp_path/'state_gain_0.01.png').exists()
    assert (tmp_path/'state_gain_0.01.pdf').exists()


def test_temporal_forcing_matches_models_and_labels_virtual_time(tmp_path):
    import copy
    head=make_head(tmp_path)
    ego,bev=torch.randn(1,10,256)*.1,torch.randn(1,1512,2,2)*.1
    response=temporal_probe({'FM':head,'FM_CR':copy.deepcopy(head)},ego,bev,(2,2),
                            length=8,rhos=(0,.8),seed=3)
    assert response['virtual_dt_s']==.1
    for trial in response['trials']:
        a,b=trial['models']['FM'],trial['models']['FM_CR']
        assert a['series']==b['series']
        assert len(a['series'])==8 and a['fixed_vector_acf'] is not None


def test_free_mode_zero_perturbation_has_zero_change_with_forced_reference():
    class Anchored:
        config=NS(fmsr_num_poses=1)
        std=torch.ones(1,1,2,2)
        plan_anchor=torch.tensor([[[0.,0.],[3.,0.]],[[10.,0.],[3.,0.]]])
        def _classify_flow_anchors(self,a,b,s,e):return [torch.tensor([[0.,1.]])]
        def get_flow_schedule(self,device=None):return torch.linspace(0,1,3,device=device)
        def predict_flow_endpoint(self,x,t,a,b,s,e,i):return a.clone()
        def _mask_flow_speed_padding(self,x):return x
    r=probe(Anchored(),torch.ones(1,1,2),torch.ones(1,1,1,1),(1,1),
            scale=0,jacobian_every=0,reference_mode=0)
    assert r['WMD']==0 and r['selected_output_difference']==0
    assert r['MSD']==10 and not r['anchor_switched']
