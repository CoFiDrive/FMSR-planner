"""Controlled white/AR(1) feature forcing of a static scene, separate from live time."""
import numpy as np
import torch
from scripts.pdm_lite.probe_eval_flow import probe


def vector_acf(values,limit=20):
    values=np.asarray(values,dtype=float).reshape(len(values),-1)
    centered=values-values.mean(0)
    energy=float(np.sum(centered**2))
    if energy<1e-12:return None
    return [float(np.sum(centered[:len(values)-lag]*centered[lag:])/energy)
            for lag in range(min(limit,len(values)-1)+1)]


@torch.no_grad()
def temporal_probe(heads,ego,bev,spatial,length=32,rhos=(0,.8),scale=.01,seed=0,dt=.1,
                   freeze_speed=False):
    if length<8 or dt<=0 or any(abs(rho)>=1 for rho in rhos):
        raise ValueError('Need >=8 virtual cycles, positive dt and |rho|<1')
    from scipy.signal import welch
    rng=torch.Generator(device=ego.device).manual_seed(seed)
    innovations=torch.randn((length,*ego.shape),device=ego.device,generator=rng)
    amplitude=ego.square().mean().sqrt()*scale
    results=[]
    for rho in rhos:
        noise=innovations[0]
        series={name:[] for name in heads};inputs=[]
        for index in range(length):
            if index:noise=rho*noise+(1-rho**2)**.5*innovations[index]
            delta=noise*amplitude
            inputs.append(delta.cpu().numpy().flatten())
            for name,head in heads.items():
                result=probe(head,ego,bev,spatial,scale,seed,jacobian_every=0,
                             freeze_speed=freeze_speed,condition_delta=delta)
                series[name].append({'cycle':index,'virtual_time_s':index*dt,
                                     'WMD':result['WMD'],'MSD':result['MSD'],
                                     'anchor_switched':result['anchor_switched'],
                                     'fixed_delta':result['fixed_mode_terminal_delta_normalized'].tolist(),
                                     'free_delta':result['free_mode_terminal_delta_normalized'].tolist(),
                                     'stage_d':[r['d_next'] for r in result['stage_metrics']],
                                     'stage_eta':[r['eta_j'] for r in result['stage_metrics']]})
        models={}
        for name,rows in series.items():
            fixed=np.asarray([r['fixed_delta'] for r in rows]);free=np.asarray([r['free_delta'] for r in rows])
            poses=heads[name].config.fmsr_num_poses
            lateral=free.reshape(length,-1,2)[:,:poses,1].mean(1)
            frequency,psd=welch(lateral,fs=1/dt,nperseg=min(64,length))
            models[name]={'fixed_response_rms':float(np.sqrt(np.mean(fixed**2))),
                          'free_response_rms':float(np.sqrt(np.mean(free**2))),
                          'fixed_vector_acf':vector_acf(fixed),'free_vector_acf':vector_acf(free),
                          'ASR':sum(r['anchor_switched'] for r in rows)/length,
                          'mean_lateral_normalized_frequency_hz':frequency.tolist(),
                          'mean_lateral_normalized_psd':psd.tolist(),'series':rows}
        results.append({'rho':rho,'input_vector_acf':vector_acf(inputs),
                        'input_rms':float(np.sqrt(np.mean(np.asarray(inputs)**2))), 'models':models})
    return {'protocol':'one static encoded scene; stationary-start white/AR(1) Gaussian ego_query forcing; '
                       'same innovations across models and rhos; generate anew each virtual cycle',
            'scope':'controlled planner response, not vehicle dynamics or closed-loop recovery',
            'seed':seed,'scale':scale,'virtual_dt_s':dt,'length':length,
            'time_axes':'virtual cycle time t is distinct from within-generator flow tau',
            'acf_protocol':'demean each vector component; biased dot-product ACF, no GP fitting',
            'trials':results}
