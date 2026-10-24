import math
import numpy as np
import torch
from scripts.pdm_lite.robustness_protocol import Perturbation,route_identity,attach_head


def make(family,amplitude,**kwargs):
    return Perturbation(dict(family=family,amplitude=amplitude,mode='continuous',start_s=0.,**kwargs),137,'RouteScenario_1773_rep0')


def test_route_identity_ignores_run_timestamp():
    assert route_identity('/a+RouteScenario_1773_rep0_Town12_10_14_1')==route_identity('/b+RouteScenario_1773_rep0_Town12_10_15_2')


def test_state_injection_does_not_touch_anchor_or_speed():
    p=make('state',.1,axis='lateral',tau=.4);p.begin(2.)
    x=torch.zeros(1,60,11,2);before=x.clone();y=p.state(x,.4)
    assert torch.equal(x,before) and torch.equal(y[...,10,:],x[...,10,:])
    assert torch.allclose(y[...,:10,1].abs(),torch.full((1,60,10),.1))
    assert torch.equal(y[...,0],x[...,0])


def test_state_only_selected_sampling_time():
    p=make('state',.1,axis='longitudinal',tau=.4);p.begin(2.)
    x=torch.zeros(1,1,11,2)
    assert p.state(x,.35) is x and not p.events


def test_zero_perturbation_preserves_values_and_global_rng():
    p=make('feature',0.,component='bev');p.begin(5.)
    x=torch.ones(3,4);rng=torch.get_rng_state().clone()
    assert p.feature(x,'bev') is x and torch.equal(rng,torch.get_rng_state())


def test_feature_noise_is_paired_and_has_requested_rms():
    a=make('feature',.03,component='bev');b=make('feature',.03,component='bev')
    a.begin(2.);b.begin(2.);x=torch.ones(4,5);rng=torch.get_rng_state().clone()
    y=a.feature(x,'bev');z=b.feature(x,'bev')
    assert torch.equal(y,z) and torch.equal(rng,torch.get_rng_state())
    assert abs(float((y-x).square().mean().sqrt())-.03)<1e-6


def test_position_bias_is_post_filter_and_leaves_original_unchanged():
    p=make('position',.25,axis='lateral');p.begin(2.)
    x=np.array([0.,0.,0.,3.]);y,c=p.localization(x,0.)
    assert np.array_equal(x,[0,0,0,3]) and abs(y[1])==.25 and abs(y[0])<1e-8
    assert c==0 and y[2]==0 and y[3]==3


def test_heading_updates_pose_and_compass_consistently():
    p=make('heading',1.);p.begin(2.)
    y,c=p.localization(np.array([0.,0.,0.,3.]),0.)
    assert abs(y[2]-c)<1e-12 and abs(abs(c)-math.pi/180)<1e-12


def test_pulse_is_exactly_one_injection():
    p=Perturbation(dict(family='state',amplitude=.1,axis='lateral',tau=.4,mode='pulse',start_s=5.),0,'RouteScenario_1_rep0')
    x=torch.zeros(1,1,11,2);p.begin(4.95);assert p.state(x,.4) is x
    p.begin(5.);assert p.state(x,.4) is not x
    p.begin(5.05);assert p.state(x,.4) is x


def test_wrapper_leaves_clean_inference_identical():
    class Head:
        def flow_step(self,x,tau,next_tau,anchor,scene):return x+1
        def forward_test_flow(self,ego,bev,spatial,global_img=None):return ego+bev
    h=Head();p=make('none',0.);attach_head(h,p)
    x=torch.ones(2,3)
    assert torch.equal(h.forward_test_flow(x,x,(3,)),x+x)
    assert torch.equal(h.flow_step(x,.4,.45,None,None),x+1)
