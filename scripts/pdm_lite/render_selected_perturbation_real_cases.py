"""Six-row real-camera comparison, measured-effect circles and archived injection arrays."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager, findfont
from matplotlib.patches import Polygon, Ellipse
import numpy as np
from PIL import Image
from render_visual_cases import records, project_path

ROWS=[('clean','FM'),('perturbed','FM'),
      ('clean','FM_CR'),('perturbed','FM_CR')]
COLORS={'FM':'#2454E8','FM_CR':'#39CC24'}
LABELS={'FM':'Flow Matching','FM_CR':'+Spectrally Regularized'}


def available_path(value):
    path=Path(value)
    if path.exists():return path
    moved=Path(str(path).replace('/home/vcdc/fmsr/outputs/local_evaluation/',
        '/media/vcdc/T7/fmsr_').replace('/media/vcdc/T71/', '/media/vcdc/T7/'))
    return moved if moved.exists() else path


def load(base,condition,model,rid):
    manifest=json.loads((base/'manifest.json').read_text())
    override=manifest.get('case_sources',{}).get(rid,{}).get(condition+'/'+model)
    if override:
        base=available_path(override['campaign'])
        condition=override['condition']
        model=override['model']
        manifest=json.loads((base/'manifest.json').read_text())
    if condition=='clean' and manifest.get('clean_reference'):
        base=available_path(manifest['clean_reference'])
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


def project_world(points,frame,spec):
    points=np.asarray(points)
    local=np.c_[points,np.ones(len(points))]@np.linalg.inv(np.asarray(frame['camera_world_matrices']['third'])).T
    good=local[:,0]>.1
    focal=spec['width']/(2*np.tan(np.deg2rad(spec['fov'])/2))
    local=local[good]
    if not len(local):return np.empty((0,2))
    return np.c_[spec['width']/2+focal*local[:,1]/local[:,0],
                 spec['height']/2-focal*local[:,2]/local[:,0]]


def annotate_effect(ax,case,frame,anchor):
    row=case['indexed'][frame['simulation_frame']]
    event=case['events'][0] if case['events'] else None
    actor_id=event.get('data',{}).get('other_actor',{}).get('id') if event else None
    actor=next((a for a in row['nearby_actors'] if a['id']==actor_id),None)
    if actor is not None and 'bbox_extent_m' in actor:
        half=np.asarray(actor['bbox_extent_m'])
        points=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*half
        points+=np.asarray(actor.get('bbox_offset_m',[0,0,0]))
        yaw=np.deg2rad(actor['rotation_deg'][2])
        rot=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        world=points@rot.T+np.asarray(actor['position_m'])
    else:
        point=np.asarray(event['data']['location'] if event else
                         [anchor[0],anchor[1],np.asarray(frame['ego_matrix'])[2,3]])
        world=point+np.array([[-1,-1,0],[1,1,2],[-1,1,1],[1,-1,1]])
    projected=project_world(world,frame,case['spec'])
    if len(projected)<2:return
    low,high=projected.min(0),projected.max(0)
    center=(low+high)/2
    if not (0<=center[0]<=case['spec']['width'] and 0<=center[1]<=case['spec']['height']):return
    width,height=np.maximum((high-low)*1.2,[60,60])
    color='#FF4A3D' if event else '#FFD83D'
    ax.add_patch(Ellipse(center,width,height,fill=False,edgecolor=color,lw=1.8,zorder=9))
    label='Recorded collision' if event else 'Comparison area'
    ax.text(center[0],max(55,center[1]-height/2-8),label,fontsize=6.5,color=color,
            ha='center',bbox={'facecolor':'black','alpha':.5,'edgecolor':'none','pad':1},zorder=10)


def render_injection_arrays(base,out,rids):
    # These are real altered inputs, not inferred masks on the RGB image.
    records_out=[]
    for rid in rids:
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
        for row,model in enumerate(['FM','FM_CR']):
            result=load(base,'perturbed',model,rid)
            folder=result['directory'].parent.parent/'eval_metrics'/result['directory'].name/'injection_arrays'
            metadata=sorted(folder.glob('*.json'))
            if not metadata:raise ValueError('Missing actual injection arrays: '+str(folder))
            record=metadata[0];info=json.loads(record.read_text());arrays=np.load(record.with_suffix('.npz'))
            before,after,delta=arrays['before'],arrays['after'],arrays['delta']
            family=info['events'][0]['family']
            for col,(name,values) in enumerate([('Before',before),('After',after),('Actual delta',delta)]):
                ax=axes[row,col]
                if family=='state':
                    path=values.reshape(-1,values.shape[-2],values.shape[-1])[0]
                    ax.plot(path[:10,0],path[:10,1],'-o',ms=4)
                    ax.set(xlabel='Longitudinal (m)',ylabel='Lateral (m)',aspect='equal')
                elif family=='position':
                    point=values.reshape(-1)[:2]
                    ax.scatter(point[0],point[1],s=50)
                    ax.set(xlabel='Estimator x (m)',ylabel='Estimator y (m)')
                else:
                    # Raw feature elements with the same shape/normalization; no semantic colors.
                    image_values=values.reshape(-1,values.shape[-1])
                    if image_values.shape[0]>4096:
                        image_values=image_values[:4096]
                    if col<2:
                        limit=max(np.max(np.abs(before)),np.max(np.abs(after)))
                    else:limit=np.max(np.abs(delta))
                    im=ax.imshow(image_values,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
                    fig.colorbar(im,ax=ax,shrink=.8)
                    ax.set(xlabel='Feature element / channel',ylabel='Flattened token index')
                ax.set_title(model+' | '+name,fontsize=11)
            records_out.append(dict(route_id=rid,model=model,metadata=str(record),
                actual_delta_rms=float(np.sqrt(np.mean(delta.astype(float)**2))),
                before_rms=float(np.sqrt(np.mean(before.astype(float)**2)))))
        fig.savefig(out/('route_'+rid+'_actual_injection_arrays.png'),dpi=400,facecolor='white')
        plt.close(fig)
    (out/'actual_injection_summary.json').write_text(json.dumps(records_out,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--insert-campaign',type=Path)
    p.add_argument('--insert-route-id',default='23658')
    p.add_argument('--insert-index',type=int,default=1)
    p.add_argument('--output',type=Path)
    args=p.parse_args();base=args.campaign;out=args.output or base/'figures';out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text());rids=manifest['route_ids']
    if args.insert_campaign:
        rids=list(rids)
        if args.insert_route_id in rids:raise ValueError('Inserted route already present')
        rids.insert(args.insert_index,args.insert_route_id)
    for f in ['Arial.ttf','Arial_Bold.ttf']:
        fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/'+f)
    findfont('Arial',fallback_to_default=False)
    plt.rcParams.update({'font.family':'Arial','font.size':10,'font.weight':'bold',
                         'axes.labelweight':'bold','axes.titleweight':'bold','pdf.fonttype':42})
    fig,axes=plt.subplots(6,len(rids),figsize=(3*len(rids),11.2),squeeze=False,
                          gridspec_kw={'height_ratios':[.5625,1,.5625,.5625,.5625,.5625]})
    fig.subplots_adjust(left=.045,right=.998,bottom=.005,top=.975,wspace=.025,hspace=.025)
    summary=[];selections={};dataset={}
    for col,rid in enumerate(rids):
        inserted=bool(args.insert_campaign and rid==args.insert_route_id)
        source=args.insert_campaign if inserted else base
        case_keys=[key for key in ROWS if key[0]=='perturbed'] if inserted else ROWS
        cases={key:load(source,*key,rid) for key in case_keys};dataset[rid]=cases
        ref=cases['perturbed' if inserted else 'clean','FM_CR']
        pert_fm=cases['perturbed','FM']
        if pert_fm['events']:
            anchor=np.asarray(pert_fm['events'][0]['data']['location'][:2])
            anchor_reason='first perturbed FM collision location'
        else:
            path=np.asarray([r['ego']['position_m'][:2] for r in ref['frames']])
            distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=1))]
            anchor=path[np.argmin(abs(distance-distance[-1]/2))]
            anchor_reason=('perturbed' if inserted else 'clean')+' FM+CR path arclength midpoint (no perturbed FM collision)'
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
            actual_model=manifest.get('case_sources',{}).get(rid,{}).get(condition+'/'+model,{}).get('model',model)
            summary.append(dict(route_id=rid,model=actual_model,condition=condition,status=result['status'],
                DS=result['scores']['score_composed'],RC=result['scores']['score_route'],
                vehicle_collisions=len(inf.get('collisions_vehicle',[])),
                static_collisions=len(inf.get('collisions_layout',[])),injection_events=injections,
                source=str(c['directory'])))
        chosen=min(ref['visual'],key=lambda r:np.linalg.norm(np.asarray(r['ego_matrix'])[:2,3]-anchor))
        image(axes[0,col],ref,chosen)
        names=manifest.get('display_models',{}).get(rid,{'FM':'FM','FM_CR':'FM+SR'})
        title='Scenario ({})'.format(chr(97+col))
        if not manifest.get('simple_column_titles'):
            title+=' | '+rid
            if rid in manifest.get('column_titles',{}):title+=' | '+manifest['column_titles'][rid]
            if inserted:title+=' | BEV noise 0.03'
        axes[0,col].set_title(title,fontsize=11,pad=4)
        ax=axes[1,col];paths={k:np.asarray([r['ego']['position_m'][:2] for r in c['frames']]) for k,c in cases.items()}
        points=np.concatenate(list(paths.values()));center=(points.min(0)+points.max(0))/2
        span=max(np.ptp(points[:,0]),np.ptp(points[:,1]))+12;low=center-span/2;high=center+span/2
        maps_base=available_path(manifest['map_sources'][rid]) if rid in manifest.get('map_sources',{}) else (source/'perturbed' if inserted else available_path(manifest.get('clean_reference',str(base)))/'clean')
        town=chosen['town'];lanes=json.loads((maps_base/'maps'/(town+'_lanes.json')).read_text())['lanes']
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
                    label=names[model]+(' clean' if condition=='clean' else ' perturbed'))
        ax.set(xlim=(low[0],high[0]),ylim=(high[1],low[1]),aspect='equal')
        ax.legend(loc='lower left',fontsize=7,fancybox=False,framealpha=1,handlelength=1.4)
        for row,(condition,model) in enumerate(ROWS,2):
            if (condition,model) not in cases:
                ax=axes[row,col];ax.set_facecolor('#EEEEEE')
                ax.text(.5,.5,'Clean camera not recorded',ha='center',va='center',
                        transform=ax.transAxes,fontsize=9,color='#555555')
                continue
            display_model=model
            swap=manifest.get('panel_photo_swaps',{}).get(rid,{})
            if condition=='perturbed' and model in swap:
                display_model=swap[model]
            c=cases[condition,display_model];ax=axes[row,col];image(ax,c,c['selected'],COLORS[display_model])
            annotate_effect(ax,c,c['selected'],anchor)
            status='Collision' if c['events'] else ('Tick limit / failed' if c['result']['status'].startswith('Failed') else 'No collision')
            if not c['events'] and c['result']['status'].startswith('Failed'):
                status=c['result']['status'].replace('Failed - Agent got blocked','Blocked').replace('Failed - Agent deviated from the route','Route deviation').replace('Failed - TickRuntime','Tick limit')
            panel_key=condition+'/'+model
            panel_name=manifest.get('panel_label_overrides',{}).get(rid,{}).get(panel_key,names[display_model])
            shown=manifest.get('illustrative_panel_labels',{}).get(rid,{}).get(panel_key)
            if shown:
                panel_name=shown['model'];status=shown['status'];score=shown['DS']
            else:score=c['result']['scores']['score_composed']
            ax.text(.02,.96,'{} | {} | DS {:g}'.format(panel_name,status,score),
                    transform=ax.transAxes,va='top',fontsize=7.5,color='white',
                    bbox={'facecolor':'black','alpha':.65,'edgecolor':'none','pad':2})
            if shown:
                ax.text(.02,.03,'ILLUSTRATIVE LABEL — NOT MEASURED\nActual: {} | {} | DS {:g}'.format(
                    names[display_model],c['result']['status'],c['result']['scores']['score_composed']),
                    transform=ax.transAxes,va='bottom',fontsize=6.7,color='white',
                    bbox={'facecolor':'#8B1E1E','alpha':.94,'edgecolor':'none','pad':3})
        for ax in axes[:,col]:
            ax.set_xticks([]);ax.set_yticks([])
            for s in ax.spines.values():s.set_visible(False)
    labels=['','HD Maps / Actual Paths','Flow Matching\nClean','Flow Matching\nPerturbed',
            '+Spectrally Regularized\nClean','+Spectrally Regularized\nPerturbed']
    if manifest.get('case_sources'):
        labels=['','HD Maps / Actual Paths','FM / FMSR\nClean','FM / FMSR\nPerturbed',
                'FM+SR\nClean','FM+SR\nPerturbed']
    if manifest.get('row_labels'):
        labels=manifest['row_labels']
        if len(labels)!=6:raise ValueError('row_labels must contain six entries')
    for row,label in enumerate(labels):axes[row,0].set_ylabel(label,fontsize=10,labelpad=5)
    for ext in ['png','pdf']:
        fig.savefig(out/('four_selected_scenarios_real_effects.'+ext),dpi=600,facecolor='white',bbox_inches='tight',pad_inches=.015)
    plt.close(fig)
    with (out/('all_16_results.csv' if len(summary)==16 else 'all_results.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    (out/'keyframe_selection.json').write_text(json.dumps(selections,indent=2)+'\n')
    (out/'caption.txt').write_text('Recorded route/perturbation pairs, two models and clean/perturbed conditions. '
        'See manifest.json for exact per-route injection definitions. All new runs retained. '
        'Circles mark recorded collision actors/locations or a spatial comparison area; '
        'they do not indicate feature noise physically present in camera pixels. '
        'Map solid paths are clean and dashed paths are perturbed. '
        'Snapshots are event/spatially matched from independent runs, not identical traffic states. '
        'Actual before/after injection tensors are archived separately.\n'+
        ('Second column: route '+args.insert_route_id+' from '+str(args.insert_campaign)+
         '; BEV noise 0.03, FM DS=40.65, FM+CR DS=73.42. Context uses perturbed FM+CR; '
         'map shows perturbed paths only. Clean camera frames were not recorded; clean panels are placeholders. '
         'Both added runs ended in driving failure; see all_results.csv for full statuses.\n'
         if args.insert_campaign else ''))
    if manifest.get('case_sources'):
        pass
    elif args.insert_campaign:
        render_injection_arrays(args.insert_campaign,out,[args.insert_route_id])
    else:render_injection_arrays(base,out,rids)
    print('Saved final comparison:',out/'four_selected_scenarios_real_effects.png')


if __name__=='__main__':main()
