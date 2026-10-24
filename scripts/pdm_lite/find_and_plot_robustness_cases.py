"""Screen historical paired runs and draw measured BEV outputs plus injection audits.

Selected cases illustrate favorable outcomes; they are not an unbiased evaluation.
Road geometry is CARLA ground truth, not a predicted semantic/feature heatmap.
"""
import csv
import gzip
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager
from matplotlib.patches import Polygon, Circle
import numpy as np

ROOT=Path('/home/vcdc/fmsr')
DATA=ROOT/'outputs/analysis/robustness_complete_summary_20261019/route_metrics.csv'
OUT=ROOT/'outputs/analysis/robustness_selected_bev_cases_20261021'
REQUESTS=[('2164','pulse_state_lateral_0.1','State pulse: lateral 0.10 m'),
          ('2127','position_longitudinal_0.5','Localization: longitudinal 0.50 m'),
          ('2373','feature_bev_0.03','BEV feature noise: RMS 3%'),
          ('2709','feature_query_0.03','Query feature noise: RMS 3%')]


def read_lines(folder,name):
    p=folder/name
    if not p.exists():p=folder/(name+'.gz')
    with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def load(row):
    job=ROOT/'outputs/local_evaluation'/row['campaign']/'runs'/row['job_id']
    checkpoint=json.loads((job/'checkpoint_endpoint.json').read_text())
    record=next(r for r in checkpoint['_checkpoint']['records'] if r['route_id']==row['route_id'])
    folder=job/'eval_metrics'/record['save_name']
    states=read_lines(folder,'vehicle_state.jsonl')
    plans=[r for r in read_lines(folder,'planning.jsonl') if r['policy'].get('predicted_trajectory_ego_m')]
    return dict(row=row,folder=folder,states=states,state_index={r['step']:r for r in states},plans=plans,
                actors=read_lines(folder,'actors.jsonl'),audit=read_lines(folder,'perturbations.jsonl'),record=record)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=list(csv.DictReader(DATA.open()))
    clean={(r['route_id'],r['model']):r for r in rows if r['condition']=='clean'}
    lookup={(r['campaign'],r['phase'],r['condition'],r['route_id'],r['model']):r for r in rows}
    screened=[]
    for r in rows:
        if r['condition']=='clean' or r['model']!='FM':continue
        partner=lookup[r['campaign'],r['phase'],r['condition'],r['route_id'],'FM_CR']
        f0=float(clean[r['route_id'],'FM']['DS']);c0=float(clean[r['route_id'],'FM_CR']['DS'])
        fd=float(r['DS'])-f0;cd=float(partner['DS'])-c0
        screened.append(dict(campaign=r['campaign'],phase=r['phase'],condition=r['condition'],
            route_id=r['route_id'],scenario=r['scenario'],FM_clean=f0,FM_perturbed=float(r['DS']),FM_change=fd,
            CR_clean=c0,CR_perturbed=float(partner['DS']),CR_change=cd,
            matches_requested_pattern=int(fd < -1e-6 and cd>=-1e-6)))
    write_csv(OUT/'all_screened_pairs.csv',screened)
    write_csv(OUT/'all_matching_pairs.csv',[r for r in screened if r['matches_requested_pattern']])
    for f in ['Arial.ttf','Arial_Bold.ttf']:fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/'+f)
    plt.rcParams.update({'font.family':'Arial','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(5,4,figsize=(16,18),gridspec_kw={'height_ratios':[1,1,1,1,.65]},layout='constrained')
    summaries=[];provenance={}
    lane_path=ROOT/'outputs/local_evaluation/visual_cases_20261021/maps/Town12_lanes.json'
    lanes=json.loads(lane_path.read_text())['lanes']
    for col,(rid,condition,title) in enumerate(REQUESTS):
        route='RouteScenario_'+rid+'_rep0'
        candidates=[r for r in screened if r['route_id']==route and r['condition']==condition]
        matches=[r for r in candidates if r['matches_requested_pattern']]
        chosen=matches[0]  # chronological inventory order; no minimum-score round optimization
        selected={}
        for model in ['FM','FM_CR']:
            selected[model,'clean']=load(clean[route,model])
            selected[model,'perturbed']=load(lookup[chosen['campaign'],chosen['phase'],condition,route,model])
        fm=selected['FM','perturbed']
        collision_messages=[message for k,values in fm['record']['infractions'].items()
                            if k.startswith('collisions_') for message in values]
        match=re.search(r'x=([\d.-]+), y=([\d.-]+)',collision_messages[0]) if collision_messages else None
        if match:
            anchor=np.array([float(match[1]),float(match[2])])
        else:
            # Red-light failure case may stop rather than collide; show its final planned state.
            anchor=np.array(fm['state_index'][fm['plans'][-1]['step']]['location'][:2])
        fmplan=min(fm['plans'],key=lambda r:np.linalg.norm(np.array(fm['state_index'][r['step']]['location'][:2])-anchor))
        heading=np.deg2rad(fm['state_index'][fmplan['step']]['rotation'][2])
        rotation=np.array([[np.cos(heading),-np.sin(heading)],[np.sin(heading),np.cos(heading)]])
        def local(points):
            p=(np.asarray(points)[...,:2]-anchor)@rotation
            return p[...,::-1]  # screen x=lateral; y=forward
        for index,(model,mode) in enumerate([('FM','clean'),('FM','perturbed'),('FM_CR','clean'),('FM_CR','perturbed')]):
            c=selected[model,mode];ax=axes[index,col]
            plan=min(c['plans'],key=lambda r:np.linalg.norm(np.array(c['state_index'][r['step']]['location'][:2])-anchor))
            state=c['state_index'][plan['step']]
            ax.set_facecolor('#F4F6F4')
            for lane in lanes.values():
                center=local([r['center'] for r in lane])
                if len(center)<2 or np.any(center.max(0)<[-18,-14]) or np.any(center.min(0)>[18,26]):continue
                left=local([r['left'] for r in lane]);right=local([r['right'] for r in lane])
                ax.add_patch(Polygon(np.r_[left,right[::-1]],facecolor='#E1E6E1',edgecolor='none'))
                for boundary in [left,right]:ax.plot(boundary[:,0],boundary[:,1],color='#97A297',lw=.6)
            actor_row=min(c['actors'],key=lambda r:abs(r['step']-plan['step']))
            for actor in actor_row['actors']:
                if not actor['type'].startswith(('vehicle.','walker.')):continue
                half=np.asarray(actor.get('bbox_extent_m',[.3,.3,.8]))[:2]
                yaw=np.deg2rad(actor['yaw_deg']);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
                box=np.array([[-1,-1],[-1,1],[1,1],[1,-1]])*half
                world=box@rot.T+np.array(actor['position_m'][:2]);box=local(world)
                ax.add_patch(Polygon(box,facecolor='#F3AF66',edgecolor='#805A2B',lw=.7,zorder=4))
            yaw=np.deg2rad(state['rotation'][2]);rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
            origin=np.array(state['location'][:2]);ego=local(np.array([[-2.2,-.9],[-2.2,.9],[2.2,.9],[2.2,-.9]])@rot.T+origin)
            ax.add_patch(Polygon(ego,facecolor='#444444',edgecolor='white',lw=.7,zorder=5))
            trajectory=np.array(plan['policy']['predicted_trajectory_ego_m'])[:,:2]@rot.T+origin
            pts=local(trajectory);color='#2454E8' if model=='FM' else '#239A45'
            ax.plot(pts[:,0],pts[:,1],'-o',color=color,lw=2,ms=3,zorder=7,label='Predicted path')
            driven=local([r['location'] for r in c['states']])
            ax.plot(driven[:,0],driven[:,1],ls=':',color=color,lw=1.1,alpha=.7,label='Driven path')
            ax.add_patch(Circle((0,0),4.5,fill=False,edgecolor='#D94841',lw=1.3,ls='--',zorder=8))
            score=float(c['row']['DS'])
            status=c['row']['status'].replace('Failed - ','')
            ax.text(.02,.98,'DS {:g} | {}'.format(score,status),transform=ax.transAxes,va='top',fontsize=9,
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
            ax.set(xlim=(-18,18),ylim=(-14,26),aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
            if index==0:ax.set_title('({}) {}\n{}'.format(chr(97+col),title,rid),fontsize=12)
            if col==0:ax.set_ylabel(('Flow Matching' if model=='FM' else '+Spectrally Regularized')+'\n'+mode,fontsize=12)
            key=model+'/'+mode
            provenance.setdefault(rid,{})[key]=dict(directory=str(c['folder']),DS=score,status=c['row']['status'],
                planning_step=plan['step'],distance_to_anchor_m=float(np.linalg.norm(origin-anchor)),
                keyframe_method='nearest own ego position to FM perturbed collision/final-state location')
        # Measured injection, separate from road space: no fabricated feature heatmap.
        ax=axes[4,col]
        for model,color in [('FM','#2454E8'),('FM_CR','#239A45')]:
            c=selected[model,'perturbed'];tt=[];vv=[]
            for row in c['audit']:
                for event in row.get('events',[]):
                    tt.append(event['relative_time_s'])
                    if event['family']=='feature':vv.append(event['delta_rms']/max(event['input_rms'],1e-12)*100)
                    elif event['family']=='position':vv.append(np.linalg.norm(event['offset_world_m']))
                    else:vv.append(event['offset_m'])
            ax.plot(tt,vv,'-o' if len(tt)<5 else '-',color=color,lw=1.5,ms=4,label='FM' if model=='FM' else '+CR')
        ax.set_xlabel('Relative simulation time (s)',fontsize=10)
        ax.set_ylabel('Noise RMS ratio (%)' if condition.startswith('feature') else 'Injected offset (m)',fontsize=10)
        if condition.startswith('feature'):
            ax.set_ylim(0,3.3)
            ax.set_yticks([0,1,2,3])
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        elif condition.startswith('pulse_state'):
            ax.set_ylim(0,.12)
        else:
            ax.set_ylim(0,.6)
        ax.set_title('Measured injection audit',fontsize=10);ax.grid(alpha=.2);ax.legend(fontsize=8)
        a=[r['FM_perturbed'] for r in candidates];b=[r['CR_perturbed'] for r in candidates]
        summaries.append(dict(route_id=rid,scenario=chosen['scenario'],condition=condition,
            baseline_FM=chosen['FM_clean'],baseline_CR=chosen['CR_clean'],
            selected_FM=chosen['FM_perturbed'],selected_CR=chosen['CR_perturbed'],
            selected_campaign=chosen['campaign'],selected_phase=chosen['phase'],
            repeated_rounds=len(candidates),matching_rounds=len(matches),FM_repeat_mean=float(np.mean(a)),
            CR_repeat_mean=float(np.mean(b)),FM_all_rounds=json.dumps(a),CR_all_rounds=json.dumps(b)))
    for ext in ['png','pdf']:
        fig.savefig(OUT/('selected_advantage_bev_and_injection.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    write_csv(OUT/'selected_cases_and_repeats.csv',summaries)
    (OUT/'frame_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (OUT/'README.md').write_text('''# Exploratory favorable-case screening

All historical main route-level pairs were screened using FM delta < 0 and CR delta >= 0
relative to the only available clean pilot run for each model/route. All candidate pairs
are retained in all_screened_pairs.csv; these are outcome-selected examples, not unbiased
robustness estimates. The clean baseline has no independent repeat distribution.
Four cases were selected with both clean models scoring 100 and CR scoring 100 in all
available repeats for that condition. The illustrated round is the first matching round
in the chronological inventory. See selected_cases_and_repeats.csv for every repeat score.

BEV panels use actual CARLA road geometry, measured actor boxes, predicted trajectories
(solid with dots) and actual paths (dotted). Dashed red circles mark a common comparison
region, not a proven causal location. Runs are independent; snapshots are spatially matched,
not identical traffic states. Orange boxes show vehicles/pedestrians, dark boxes show ego.
The bottom row shows measured perturbation audit values. Feature noise is not spatially
localized on the road. Intermediate feature tensors and before/after internal tau=0.4
states were not archived, so no spatial feature/noise heatmap is claimed or fabricated.
These figures reuse archived runs and do not imply reproduction in newly recorded runs.
''')
    print(OUT/'selected_advantage_bev_and_injection.png')


if __name__=='__main__':main()
