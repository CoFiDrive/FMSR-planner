"""Deterministic perturbations independent of model RNG, selected modes, and run name."""
import hashlib
import math
import re
from types import MethodType
import numpy as np
import torch


def stable_seed(seed, route, tick, component):
    value=f'{seed}/{route}/{tick}/{component}'.encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'little') % (2**63-1)


def route_identity(argument):
    match=re.search(r'RouteScenario_(\d+)_rep(\d+)', str(argument))
    if not match:raise ValueError('Cannot recover route identity from agent config')
    return match.group(0)


class Perturbation:
    def __init__(self,spec,seed,route):
        self.spec=dict(spec);self.seed=seed;self.route=route
        self.time=0.;self.tick=0;self.events=[];self.pulse_used=False
        if self.spec['family'] not in ('none','state','feature','position','heading'):
            raise ValueError('Unsupported perturbation family')
        if self.spec['amplitude']<0:raise ValueError('Negative amplitude')
        if self.spec.get('mode','continuous') not in ('continuous','pulse'):raise ValueError('Bad mode')

    def begin(self,relative_time):
        self.time=float(relative_time);self.tick=int(round(self.time/.05));self.events=[]

    def active(self):
        if self.spec['family']=='none' or self.spec['amplitude']==0:return False
        if self.spec.get('mode','continuous')=='pulse':
            return not self.pulse_used and self.time>=self.spec.get('start_s',5.)
        return self.time>=self.spec.get('start_s',2.)

    def sign(self):
        # Constant biases for localization; independent signs every simulation tick for state injections.
        tick=self.tick if self.spec['family']=='state' and self.spec.get('mode')!='pulse' else 0
        return 1. if stable_seed(self.seed,self.route,tick,'sign')%2 else -1.

    def record(self,**fields):
        self.events.append(dict(family=self.spec['family'],amplitude=self.spec['amplitude'],
                                relative_time_s=self.time,tick=self.tick,**fields))
        if self.spec.get('mode')=='pulse':self.pulse_used=True

    def state(self,x,tau):
        if self.spec['family']!='state' or not self.active():return x
        if abs(float(tau)-self.spec.get('tau',.4))>1e-5:return x
        result=x.clone();axis=0 if self.spec['axis']=='longitudinal' else 1
        n=self.spec.get('poses',10);offset=self.sign()*self.spec['amplitude']
        result[...,:n,axis]+=offset
        self.record(axis=self.spec['axis'],offset_m=offset,tau=float(tau),route_point_count=n)
        return result

    def feature(self,value,component):
        if self.spec['family']!='feature' or self.spec['component']!=component or not self.active():return value
        g=torch.Generator(device=value.device).manual_seed(stable_seed(self.seed,self.route,self.tick,component))
        noise=torch.randn(value.shape,device=value.device,dtype=torch.float32,generator=g)
        # This normalization makes beta the actual RMS ratio for every tensor.
        noise=noise/noise.square().mean().sqrt().clamp_min(1e-12)
        rms=value.float().square().mean().sqrt();delta=self.spec['amplitude']*rms*noise
        result=value+delta.to(value.dtype)
        self.record(component=component,input_rms=float(rms),delta_rms=float(delta.square().mean().sqrt()))
        return result

    def localization(self,state,compass):
        if self.spec['family'] not in ('position','heading') or not self.active():return state,compass
        result=np.array(state,copy=True);heading=float(state[2]);sign=self.sign()
        if self.spec['family']=='position':
            direction=heading+(math.pi/2 if self.spec['axis']=='lateral' else 0.)
            offset=sign*self.spec['amplitude']*np.array([math.cos(direction),math.sin(direction)])
            result[:2]+=offset
            self.record(axis=self.spec['axis'],offset_world_m=offset.tolist(),injection='post_filter_estimate')
        else:
            offset=sign*math.radians(self.spec['amplitude'])
            result[2]=(heading+offset+math.pi)%(2*math.pi)-math.pi
            compass=(float(compass)+offset+math.pi)%(2*math.pi)-math.pi
            self.record(offset_rad=offset,injection='post_filter_estimate')
        return result,compass


def attach_head(head,perturb):
    original_step=head.flow_step
    def flow_step(self,x,tau,tau_next,anchor,scene):
        return original_step(perturb.state(x,tau),tau,tau_next,anchor,scene)
    head.flow_step=MethodType(flow_step,head)
    original_forward=head.forward_test_flow
    def forward(self,ego_query,bev_feature,bev_spatial_shape,global_img=None):
        ego_query=perturb.feature(ego_query,'query')
        bev_feature=perturb.feature(bev_feature,'bev')
        return original_forward(ego_query,bev_feature,bev_spatial_shape,global_img)
    head.forward_test_flow=MethodType(forward,head)


def attach_bridge_planner(planner, perturb):
    """Inject once at the native diffusion planner's conditioning boundary."""
    from functools import wraps
    if perturb.spec['family'] not in ('none', 'feature', 'state'):
        raise ValueError('Bridge planner wrapper supports feature/state perturbations only')
    if perturb.spec['family'] == 'state':
        original_denoise = planner.denoise
        @wraps(original_denoise)
        def denoise(trajectory, anchors, t, *args, **kwargs):
            # Diffusion runs t:1->0, while FM runs tau:0->1. Update the
            # actual integrator state as well as the denoiser input.
            tau = 1.0 - float(t.flatten()[0])
            shifted = perturb.state(trajectory, tau)
            if shifted is not trajectory:
                trajectory.copy_(shifted)
            return original_denoise(trajectory, anchors, t, *args, **kwargs)
        planner.denoise = denoise
    original = planner.sample
    @wraps(original)
    def sample(ego_query, bev, *args, **kwargs):
        return original(perturb.feature(ego_query, 'query'),
                        perturb.feature(bev, 'bev'), *args, **kwargs)
    planner.sample = sample

