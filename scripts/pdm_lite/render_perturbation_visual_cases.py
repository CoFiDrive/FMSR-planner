"""Export all clean/perturbed visual comparisons without outcome filtering."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, findfont
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image
from render_visual_cases import records, project_path

ROWS=[('clean','FM'),('state_lateral_0.1','FM'),
      ('clean','FM_CR'),('state_lateral_0.1','FM_CR')]
COLORS={'FM':'#2454E8','FM_CR':'#39CC24'}
LABELS={'FM':'Flow Matching','FM_CR':'+Spectrally Regularized'}


def load(base,condition,model,rid):
    root=base/condition/model
    dirs=sorted((root/'diagnostics').glob('RouteScenario_'+rid+'_rep0*'))
    # Use the exact save_name from the scored record, not a failed setup attempt.
    result=next(r for r in json.loads((root/'checkpoint_endpoint.json').read_text())['_checkpoint']['records']
                if r['route_id']=='RouteScenario_'+rid+'_rep0')
    directory=next(d for d in dirs if d.name==result['save_name'])
    frames=records(directory/'frames.jsonl')
    indexed={r['simulation_frame']:r for r in frames}
    visual=[r for r in records(directory/'visual/frames.jsonl') if r['simulation_frame'] in indexed]
    if not visual:raise ValueError('Missing camera frames: '+str(directory))
    events=[r for r in records(directory/'events.jsonl') if r['type'].startswith('COLLISION')]
    spec=next(s for s in json.loads((directory/'visual/camera_specs.json').read_text()) if s['id']=='visual_third')
    return dict(directory=directory,frames=frames,indexed=indexed,visual=visual,
                events=events,spec=spec,result=result)


def image(ax,case,frame,color=None):
    ax.imshow(Image.open(case['directory']/'visual'/frame['images']['third']))
    if color:
        row=case['indexed'][frame['simulation_frame']]
        pts=project_path(row['policy'],frame['ego_matrix'],frame['camera_world_matrices']['third'],case['spec'])
        if len(pts):ax.plot(pts[:,0],pts[:,1],'-o',color=color,lw=1.6,ms=2.2)
    ax.set(xlim=(0,case['spec']['width']),ylim=(case['spec']['height'],0))


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    args=p.parse_args();base=args.campaign;out=base/'figures';out.mkdir(exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text());rids=manifest['route_ids']
    for f in ['Arial.ttf','Arial_Bold.ttf']:
        fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/'+f)
    findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'font.weight':'bold',
                         'axes.labelweight':'bold','axes.titleweight':'bold','pdf.fonttype':42})
    fig,axes=plt.subplots(6,4,figsize=(12,11.2),gridspec_kw={'height_ratios':[.5625,1,.5625,.5625,.5625,.5625]})
    fig.subplots_adjust(left=.045,right=.998,bottom=.005,top=.975,wspace=.025,hspace=.025)
    summary=[];selections={};dataset={}
    for col,rid in enumerate(rids):
        cases={key:load(base,*key,rid) for key in ROWS};dataset[rid]=cases
        ref=cases['clean','FM_CR']
        pert_fm=cases['state_lateral_0.1','FM']
        if pert_fm['events']:
            anchor=np.asarray(pert_fm['events'][0]['data']['location'][:2])
            anchor_reason='first perturbed FM collision location'
        else:
            path=np.asarray([r['ego']['position_m'][:2] for r in ref['frames']])
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            anchor=path[np.argmin(abs(distance-distance[-1]/2))]
            anchor_reason='clean FM+CR path arclength midpoint (no perturbed FM collision)'
        selections[rid]={'anchor_xy':anchor.tolist(),'anchor_reason':anchor_reason,'runs':{}}
        for (condition,model),c in cases.items():
            if c['events']:
                target=c['events'][0]['event_frame']
                chosen=min(c['visual'],key=lambda r:abs(r['simulation_frame']-target))
                reason='nearest own collision event'
            elif c['result']['status'].startswith('Failed'):
                chosen=c['visual'][-1];reason='last captured frame before termination'
            else:
                chosen=min(c['visual'],key=lambda r:np.linalg.norm(np.asarray(r['ego_matrix'])[:2,3]-anchor))
                reason='spatially matched conflict location'
            c['selected']=chosen
            selections[rid]['runs'][condition+'/'+model]=dict(directory=str(c['directory']),
                selected_frame=chosen['simulation_frame'],reason=reason,status=c['result']['status'],
                scores=c['result']['scores'],collision_events=c['events'])
            result=c['result'];inf=result['infractions']
            # Validate that the requested perturbation actually reached the planning head.
            audit=c['directory'].parent.parent/'eval_metrics'/c['directory'].name/'perturbations.jsonl'
            injections=0
            if audit.exists():
                injections=sum(len(r.get('events',[])) for r in records(audit))
            if condition!='clean' and injections==0:
                raise ValueError('No audited perturbation events: '+str(audit))
            summary.append(dict(route_id=rid,model=model,condition=condition,status=result['status'],
                DS=result['scores']['score_composed'],RC=result['scores']['score_route'],
                vehicle_collisions=len(inf.get('collisions_vehicle',[])),
                static_collisions=len(inf.get('collisions_layout',[])),injection_events=injections,
                source=str(c['directory'])))
        chosen=min(ref['visual'],key=lambda r:np.linalg.norm(np.asarray(r['ego_matrix'])[:2,3]-anchor))
        image(axes[0,col],ref,chosen)
        axes[0,col].set_title('Scenario ({}) | {}'.format(chr(97+col),rid),fontsize=11,pad=4)
        ax=axes[1,col];paths={k:np.asarray([r['ego']['position_m'][:2] for r in c['frames']]) for k,c in cases.items()}
        points=np.concatenate(list(paths.values()));center=(points.min(0)+points.max(0))/2
        span=max(np.ptp(points[:,0]),np.ptp(points[:,1]))+12;low=center-span/2;high=center+span/2
        town=chosen['town'];lanes=json.loads((base/'clean/maps'/(town+'_lanes.json')).read_text())['lanes']
        ax.set_facecolor('#EEEEEE')
        for lane in lanes.values():
            centers=np.asarray([r['center'][:2] for r in lane])
            if len(centers)<2 or np.any(centers.max(0)<low) or np.any(centers.min(0)>high):continue
            left=np.asarray([r['left'] for r in lane]);right=np.asarray([r['right'] for r in lane])
            ax.add_patch(Polygon(np.r_[left,right[::-1]],facecolor='#444444',edgecolor='none'))
            for boundary in [left,right]:ax.plot(boundary[:,0],boundary[:,1],color='white',lw=.4)
        for key,path in paths.items():
            condition,model=key
            ax.plot(path[:,0],path[:,1],color=COLORS[model],ls='-' if condition=='clean' else '--',lw=1.6,
                    label=('FM' if model=='FM' else '+CR')+(' clean' if condition=='clean' else ' perturbed'))
        ax.set(xlim=(low[0],high[0]),ylim=(high[1],low[1]),aspect='equal')
        ax.legend(loc='lower left',fontsize=7,fancybox=False,framealpha=1,handlelength=1.4)
        for row,(condition,model) in enumerate(ROWS,2):
            c=cases[condition,model];ax=axes[row,col];image(ax,c,c['selected'],COLORS[model])
            status='Collision' if c['events'] else ('Tick limit / failed' if c['result']['status'].startswith('Failed') else 'No collision')
            ax.text(.02,.96,'{} | DS {:g}'.format(status,c['result']['scores']['score_composed']),
                    transform=ax.transAxes,va='top',fontsize=7.5,color='white',
                    bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':2})
        for ax in axes[:,col]:
            ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values():s.set_visible(False)
    labels=['','HD Maps / Actual Paths','Flow Matching\nClean','Flow Matching\nPerturbed',
            '+Spectrally Regularized\nClean','+Spectrally Regularized\nPerturbed']
    for row,label in enumerate(labels):axes[row,0].set_ylabel(label,fontsize=10,labelpad=5)
    for ext in ['png','pdf']:
        fig.savefig(out/('four_scenarios_clean_vs_perturbed.'+ext),dpi=600,facecolor='white',bbox_inches='tight',pad_inches=.015)
    plt.close(fig)
    with (out/'all_16_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (out/'keyframe_selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    (out/'caption.txt').write_text('Four selected routes, two models, clean and continuous signed lateral '
        'planning-state perturbation 0.10m. Injection begins at relative agent time 2s at each planning '
        'call tau=0.4; deterministic sign per simulation tick. All 16 prespecified runs retained. '
        'Snapshots are event/spatially aligned from independent closed-loop runs; no exact traffic-state '
        'matching is claimed. Solid map paths: clean; dashed: perturbed. RGB is not perturbed. '
        'Additional camera sensors do not feed the policy. Single-seed qualitative evidence, not statistical proof.\n')
    print('Saved final comparison:',out/'four_scenarios_clean_vs_perturbed.png')


if __name__=='__main__':main()
