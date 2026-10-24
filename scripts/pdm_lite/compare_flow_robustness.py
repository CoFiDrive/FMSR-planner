"""Paired head-level FM/CR probes on one identical encoded scene and direction bank."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from scripts.pdm_lite.probe_eval_flow import restore, probe
from scripts.pdm_lite.probe_flow_state import state_probe
from scripts.pdm_lite.probe_condition_series import temporal_probe


MODULES={'flow_only':'fmsr.tfv6.diffusion_modules.model_flow_matching_only',
         'tie_cr_fd':'fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_fd',
         'tie_cr_v2':'fmsr.tfv6.diffusion_modules.model_diffusion_head_tie_cr_v2',
         'baseline':'fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm'}


def plain(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def load_head(checkpoint,device):
    path=Path(checkpoint).resolve()
    def decode(value):
        if isinstance(value,dict):
            if 'py/tuple' in value:return tuple(decode(v) for v in value['py/tuple'])
            return {k:decode(v) for k,v in value.items() if not k.startswith('py/')}
        if isinstance(value,list):return [decode(v) for v in value]
        return value
    cfg=decode(json.loads((path.parent/'config.json').read_text()))
    variant=cfg.get('fmsr_trajectory_head','baseline')
    if variant not in MODULES:raise ValueError('Paired robustness runner supports FM/TIE heads only')
    snapshot={'checkpoint':str(path),'head_module':MODULES[variant],
              'head_class':'TrajectoryHead','head_config':cfg}
    return restore(snapshot,device)


def fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def compare(heads,ego,bev,spatial,condition_scales=(0,.01,.03),state_magnitudes=(.01,.05),
            seeds=(0,1,2),directions=4,jacobian_every=0,freeze_speed=False,reference_mode=None,
            fixed_anchor_policy='common'):
    names=list(heads)
    reference=heads[names[0]]
    for other in list(heads.values())[1:]:
        for key in ('plan_anchor','std'):
            a,b=getattr(reference,key),getattr(other,key)
            if a.shape!=b.shape or not torch.equal(a.cpu(),b.cpu()):
                raise ValueError(f'Paired comparison requires identical {key}; do not silently change anchors or norm')
        if not torch.equal(reference.get_flow_schedule().cpu(),other.get_flow_schedule().cpu()):
            raise ValueError('Paired comparison requires identical integration schedule')
    if fixed_anchor_policy not in ('common','own'):raise ValueError('Invalid fixed anchor policy')
    common_state_mode=reference_mode
    if common_state_mode is None:
        anchors=reference.plan_anchor.float()[None]
        common_state_mode=int(reference._classify_flow_anchors(anchors,bev,spatial,ego)[-1].argmax(-1).item())
    condition_reference=common_state_mode if fixed_anchor_policy=='common' else reference_mode
    conditions=[];states=[]
    for component in ('ego_query','bev'):
        for scale in condition_scales:
            for seed in seeds:
                paired={}
                for name,head in heads.items():
                    paired[name]=probe(head,ego,bev,spatial,scale,seed,jacobian_every,
                                       freeze_speed=freeze_speed,reference_mode=condition_reference,
                                       condition_component=component)
                conditions.append({'component':component,'scale':scale,'seed':seed,'models':paired})
    anchor=reference.plan_anchor.float()[None,common_state_mode:common_state_mode+1]
    grid=[anchor.clone()]
    times=reference.get_flow_schedule(device=ego.device)
    for tau,next_tau in zip(times[:-1],times[1:]):
        nxt=reference.flow_step(grid[-1],tau,next_tau,anchor,(bev,spatial,ego,None))
        if freeze_speed and nxt.shape[-2]>reference.config.fmsr_num_poses:
            nxt=nxt.clone();nxt[...,reference.config.fmsr_num_poses:,:]=anchor[...,reference.config.fmsr_num_poses:,:]
        grid.append(nxt)
    for magnitude in state_magnitudes:
        for seed in seeds:
            states.append({'magnitude':magnitude,'seed':seed,
                           'models':{name:state_probe(head,ego,bev,spatial,magnitude,seed,directions,
                                                     freeze_speed,common_state_mode,grid)
                                     for name,head in heads.items()}})
    aggregate=[]
    for component in ('ego_query','bev'):
        for scale in condition_scales:
            for name in names:
                trials=[c['models'][name] for c in conditions if c['component']==component and c['scale']==scale]
                aggregate.append({'model':name,'component':component,'scale':scale,'trial_count':len(trials),
                                  'ASR':sum(t['anchor_switched'] for t in trials)/len(trials),
                                  'WMD_mean':sum(t['WMD'] for t in trials)/len(trials),
                                  'MSD_mean':sum(t['MSD'] for t in trials)/len(trials),
                                  'free_output_change_mean':sum(t['selected_output_difference'] for t in trials)/len(trials)})
    return {'protocol':'identical encoded scene, condition noise seeds and normalized state direction bank; FP32',
            'comparison_scope':'head-level controlled mechanism comparison; shared encoded features do not '
                               'evaluate each model own encoder or raw sensor/physical condition perturbations',
            'state_sampling':'same baseline injection-state grid and reference anchor for both heads; '
                             'each head then propagates its matched nominal/perturbed suffix',
            'common_state_mode':common_state_mode,
            'fixed_anchor_policy':fixed_anchor_policy,
            'reference_mode':reference_mode,'freeze_speed_token':freeze_speed,
            'metrics_units':'WMD/MSD/output changes use S-normalized L2; ASR is a fraction',
            'condition_trials':conditions,'state_trials':states,'condition_aggregate':aggregate}


def plot(report,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=list(report['state_trials'][0]['models']) if report['state_trials'] else []
    for magnitude in sorted({t['magnitude'] for t in report['state_trials']}):
        matrices=[]
        for name in names:
            trials=[t['models'][name]['propagation_gain_mean'] for t in report['state_trials'] if t['magnitude']==magnitude]
            arrays=np.asarray(trials,dtype=float)
            valid=np.isfinite(arrays)
            count=valid.sum(axis=0)
            mean=np.divide(np.nansum(arrays,axis=0),count,out=np.full(count.shape,np.nan),where=count>0)
            matrices.append(mean)
        vmax=max(1.,max(float(np.nanmax(m)) for m in matrices))
        fig,axes=plt.subplots(1,len(names),figsize=(6*len(names),5),squeeze=False,layout='constrained')
        for name,matrix,axis in zip(names,matrices,axes[0]):
            im=axis.imshow(matrix,aspect='auto',origin='lower',vmin=0,vmax=vmax,cmap='viridis')
            axis.set(title=name,xlabel='Observed integration state',ylabel='Injection step')
            fig.colorbar(im,ax=axis,label='Normalized perturbation gain')
        fig.suptitle(f'State propagation; injected S-norm={magnitude}')
        for suffix in ('png','pdf'):fig.savefig(directory/f'state_gain_{magnitude}.{suffix}',dpi=180)
        plt.close(fig)
    for component in ('ego_query','bev'):
        selected=[t for t in report['condition_trials'] if t['component']==component and t['scale']>0]
        if not selected:continue
        fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
        scale=max(t['scale'] for t in selected)
        for name in selected[0]['models']:
            trials=[t['models'][name] for t in selected if t['scale']==scale]
            curves=np.asarray([[r['d_next'] for r in t['stage_metrics']] for t in trials])
            times=[r['tau'] for r in trials[0]['stage_metrics']]
            axes[0].plot(range(1,len(times)+1),curves.mean(0),label=name)
            eta=np.asarray([[r['eta_j'] for r in t['stage_metrics']] for t in trials])
            axes[1].plot(times,eta.mean(0),label=name)
        axes[0].set(xlabel='Integration state',ylabel='Fixed-mode separation d',title=f'{component}, scale={scale}')
        axes[1].set(xlabel='Flow tau',ylabel='Direct condition response eta')
        for axis in axes:axis.legend()
        for suffix in ('png','pdf'):fig.savefig(directory/f'condition_response_{component}.{suffix}',dpi=180)
        plt.close(fig)
    for experiment in report.get('temporal_response',[]):
        for trial in experiment['trials']:
            fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
            for name,data in trial['models'].items():
                rows=data['series'];time=[r['virtual_time_s'] for r in rows]
                axes[0].plot(time,[r['WMD'] for r in rows],label=name)
                acf=data['fixed_vector_acf']
                if acf is not None:axes[1].plot(np.arange(len(acf))*experiment['virtual_dt_s'],acf,label=name)
                axes[2].plot(data['mean_lateral_normalized_frequency_hz'],data['mean_lateral_normalized_psd'],label=name)
            axes[0].set(xlabel='Virtual condition-cycle time (s)',ylabel='Fixed-mode WMD')
            axes[1].set(xlabel='Virtual lag (s)',ylabel='Vector response ACF')
            axes[2].set(xlabel='Virtual frequency (Hz)',ylabel='Normalized lateral response PSD')
            for axis in axes:axis.legend()
            fig.suptitle(f'Static-scene forcing: rho={trial["rho"]}, seed={experiment["seed"]}')
            for suffix in ('png','pdf'):
                fig.savefig(directory/f'temporal_rho{trial["rho"]}_seed{experiment["seed"]}.{suffix}',dpi=180)
            plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--baseline-checkpoint',type=Path,required=True)
    p.add_argument('--candidate-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--condition-scales',type=float,nargs='+',default=[0,.01,.03])
    p.add_argument('--state-magnitudes',type=float,nargs='+',default=[.01,.05])
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--directions',type=int,default=4)
    p.add_argument('--jacobian-every',type=int,default=0)
    p.add_argument('--reference-mode',type=int)
    p.add_argument('--fixed-anchor-policy',choices=['common','own'],default='common')
    p.add_argument('--freeze-speed-token',action='store_true')
    p.add_argument('--temporal-length',type=int,default=0)
    p.add_argument('--temporal-rhos',type=float,nargs='+',default=[0,.8])
    p.add_argument('--temporal-dt',type=float,default=.1)
    args=p.parse_args()
    scene=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    heads={'FM':load_head(args.baseline_checkpoint,args.device),
           'FM_CR':load_head(args.candidate_checkpoint,args.device)}
    report=compare(heads,scene['ego_query'].to(args.device).float(),scene['bev_feature'].to(args.device).float(),
                   scene['bev_spatial_shape'],args.condition_scales,args.state_magnitudes,args.seeds,
                   args.directions,args.jacobian_every,args.freeze_speed_token,args.reference_mode,args.fixed_anchor_policy)
    if args.temporal_length:
        report['temporal_response']=[temporal_probe(heads,scene['ego_query'].to(args.device).float(),
                                                   scene['bev_feature'].to(args.device).float(),
                                                   scene['bev_spatial_shape'],args.temporal_length,args.temporal_rhos,
                                                   max(args.condition_scales),seed,args.temporal_dt,args.freeze_speed_token)
                                     for seed in args.seeds]
    report['provenance']={str(path.resolve()):fingerprint(path) for path in
                          (args.snapshot,args.baseline_checkpoint,args.candidate_checkpoint)}
    args.output_dir.mkdir(parents=True,exist_ok=True)
    (args.output_dir/'paired_robustness.json').write_text(json.dumps(plain(report),indent=2,allow_nan=False)+'\n')
    plot(report,args.output_dir)
    print(f'Wrote paired results and figures to {args.output_dir}')


if __name__=='__main__':main()
