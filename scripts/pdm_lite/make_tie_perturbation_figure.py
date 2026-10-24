"""Matched-condition, fixed-anchor offline route perturbation diagnostics."""
import json
import copy
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.pdm_lite.make_tie_analysis_figures import ROOT,OUTPUT,RUNS,COLORS,save,dump
from scripts.pdm_lite.probe_eval_flow import restore


def snapshot_for(model,route_id):
    directory=ROOT/'outputs/local_evaluation'/RUNS[model]
    rows=json.loads((directory/'checkpoint_endpoint.json').read_text())['_checkpoint']['records']
    row=next(r for r in rows if r['route_id']==route_id)
    files=sorted((directory/'eval_metrics'/row['save_name']/'probes').glob('*.pt'))
    assert files,f'No saved conditions for {route_id} {model}'
    # Prefer the second snapshot (~10 s), rather than the initialization snapshot.
    path=files[min(1,len(files)-1)]
    return torch.load(path,map_location='cpu',weights_only=False),path


@torch.no_grad()
def run(head,condition,mode,directions):
    ego=condition['ego_query'].cuda().float();bev=condition['bev_feature'].cuda().float()
    spatial=condition['bev_spatial_shape'];anchor=head.plan_anchor[mode].float()[None,None]
    std=head.std.cuda().float().reshape(11,2)
    n=len(directions);delta=torch.zeros(n+1,1,11,2,device='cuda')
    delta[1:,0,:10]=directions.cuda().reshape(n,10,2)*std[:10]*.1
    x=anchor.expand(n+1,-1,-1,-1).clone()+delta
    anchors=anchor.expand(n+1,-1,-1,-1)
    query=ego.expand(n+1,*ego.shape[1:]);feature=bev.expand(n+1,*bev.shape[1:])
    amplitudes=[torch.linalg.vector_norm(((x[1:]-x[:1])[:,:,:10]/std[:10]).flatten(1),dim=1).cpu().numpy()/.1]
    jacobians=[];sigma=[];steps=[0,5,10,15,19]
    times=torch.linspace(0,1,21,device='cuda')
    for k in range(20):
        tau=times[k]
        if k in steps:
            eps=.05
            offsets=torch.zeros(20,1,11,2,device='cuda')
            offsets[:,:,:10]=torch.eye(20,device='cuda').reshape(20,1,10,2)*std[:10]*eps
            inputs=torch.cat([x[:1]+offsets,x[:1]-offsets])
            out=head.predict_flow_endpoint(inputs,tau.expand(40),anchor.expand(40,-1,-1,-1),
                    bev.expand(40,*bev.shape[1:]),spatial,ego.expand(40,*ego.shape[1:]),None)
            jac=((out[:20,0,:10]-out[20:,0,:10])/(2*eps*std[:10])).flatten(1).T
            jacobians.append(jac.cpu().numpy());sigma.append(float(torch.linalg.svdvals(jac)[0]))
        endpoint=head.predict_flow_endpoint(x,tau.expand(n+1),anchors,feature,spatial,query,None).float()
        alpha=(times[k+1]-tau)/(1-tau)
        x=x+alpha*(endpoint-x)
        # The experiment isolates route sensitivity; holder speed stays identical.
        x[:,:,-1]=anchors[:,:,-1]
        amplitudes.append(torch.linalg.vector_norm(((x[1:]-x[:1])[:,:,:10]/std[:10]).flatten(1),dim=1).cpu().numpy()/.1)
    return dict(gain=np.asarray(amplitudes).T,sigma=np.asarray(sigma),jacobians=np.asarray(jacobians),jacobian_steps=steps)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(137)
    manifest=json.loads((OUTPUT/'figure_manifest.json').read_text())
    first=manifest['cases'][0]['route_id']
    fm_snapshot,_=snapshot_for('FM',first);cr_snapshot,_=snapshot_for('FM+CR',first)
    heads={'FM':restore(fm_snapshot,'cuda'),'FM+CR':restore(cr_snapshot,'cuda')}
    assert torch.equal(heads['FM'].plan_anchor,heads['FM+CR'].plan_anchor)
    assert torch.equal(heads['FM'].std,heads['FM+CR'].std)
    rng=torch.Generator().manual_seed(137)
    directions=torch.randn(32,20,generator=rng)
    directions=directions/directions.norm(dim=1,keepdim=True)
    results=[]
    for case in manifest['cases']:
        snapshot,path=snapshot_for('FM',case['route_id'])
        ego=snapshot['ego_query'].cuda().float();bev=snapshot['bev_feature'].cuda().float()
        with torch.no_grad():
            anchors=heads['FM'].plan_anchor.float()[None]
            mode=int(heads['FM']._classify_flow_anchors(anchors,bev,snapshot['bev_spatial_shape'],ego)[-1].argmax(-1))
        report={m:run(head,snapshot,mode,directions) for m,head in heads.items()}
        for m,r in report.items():
            np.savez_compressed(OUTPUT/f'perturbation_{case["route_id"]}_{m.replace("+","")}.npz',**r)
        results.append(dict(route_id=case['route_id'],scenario=case['scenario'],condition_source=str(path),
                            snapshot_step=snapshot['step'],common_anchor=mode,
                            summary={m:dict(final_gain_mean=float(r['gain'][:,-1].mean()),
                                            final_gain_p90=float(np.quantile(r['gain'][:,-1],.9)),
                                            sigma=r['sigma'].tolist(),jacobian_steps=r['jacobian_steps'])
                                     for m,r in report.items()},data=report))
        print('Completed offline probe:',case['route_id'],flush=True)
    fig,axes=plt.subplots(4,3,figsize=(14,12),layout='constrained')
    for j,case in enumerate(results):
        for i,m in enumerate(['FM','FM+CR']):
            r=case['data'][m]
            image=axes[i,j].imshow(r['gain'],origin='lower',aspect='auto',extent=[0,20,.5,32.5],
                                   vmin=0,vmax=3,cmap='magma')
            axes[i,j].set(ylabel=f'{m}: perturbation ID',xlabel='Euler sampling step')
            fig.colorbar(image,ax=axes[i,j],label='Normalized route gain',extend='max',shrink=.75)
            axes[2,j].plot(range(21),r['gain'].mean(0),color=COLORS[m],label=m)
            axes[2,j].fill_between(range(21),np.quantile(r['gain'],.1,axis=0),np.quantile(r['gain'],.9,axis=0),color=COLORS[m],alpha=.15)
            axes[3,j].plot(r['jacobian_steps'],r['sigma'],'o-',color=COLORS[m],label=m)
        axes[0,j].set_title(case['scenario'].removesuffix('_1')+'\n'+case['route_id'],fontsize=8)
        axes[2,j].axhline(1,color='k',ls='--',lw=.8);axes[2,j].set(xlabel='Euler sampling step',ylabel='Mean route gain (10–90% band)');axes[2,j].legend()
        axes[3,j].axhline(1,color='k',ls='--',lw=.8);axes[3,j].set(xlabel='Euler sampling step',ylabel='Endpoint Jacobian max singular value');axes[3,j].legend()
    fig.suptitle('Offline diagnostic: identical saved FM conditions, common anchor, 32 matched perturbations\nFP32, fixed speed token, initial normalized route norm = 0.1; not a closed-loop stability proof',fontsize=10)
    save(fig,'fig3_offline_perturbation_propagation')
    dump(OUTPUT/'perturbation_manifest.json',dict(protocol='Same FM query/BEV, FM-selected common anchor, 32 seeded Gaussian unit directions; speed token frozen; normalized route perturbation norm .1; central finite differences epsilon .05',
        precision='FP32, TF32 disabled',models={m:heads[m].config.fmsr_trajectory_head for m in heads},
        cases=[{k:v for k,v in c.items() if k!='data'} for c in results]))


if __name__=='__main__':main()
