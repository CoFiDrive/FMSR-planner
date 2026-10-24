"""Create reproducible figures from the selected recorded driving attempts."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.pdm_lite.analyze_planning_metrics import world_path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'outputs/analysis/tie_paper_figures_20261013'
RUNS={'FM':'bench2drive220_fm_best_full_metrics_20261012',
      'FMSR':'bench2drive220_author_v8_pdm_20261010_165733_epoch10',
      'FM+CR':'bench2drive220_legacy_fm_cr_latest_merged_20261012'}
COLORS={'FM':'#4477AA','FMSR':'#228833','FM+CR':'#CC6677'}


def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def save(fig,name):
    fig.savefig(OUTPUT/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUTPUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def overlap_change(previous,current,grid):
    # Remove repeated points, project current ego onto the preceding world path.
    keep=np.r_[True,np.linalg.norm(np.diff(previous,axis=0),axis=1)>1e-6]
    previous=previous[keep]
    keep=np.r_[True,np.linalg.norm(np.diff(current,axis=0),axis=1)>1e-6]
    current=current[keep]
    if len(previous)<2 or len(current)<2:return np.full(len(grid),np.nan)
    vec=np.diff(previous,axis=0);lens=np.linalg.norm(vec,axis=1)
    fraction=np.clip(np.sum((current[0]-previous[:-1])*vec,axis=1)/lens**2,0,1)
    projections=previous[:-1]+vec*fraction[:,None]
    i=np.argmin(np.linalg.norm(projections-current[0],axis=1))
    old_s=np.r_[0,np.cumsum(lens)];progress=old_s[i]+lens[i]*fraction[i]
    new_s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(current,axis=0),axis=1))]
    valid=grid<=min(old_s[-1]-progress,new_s[-1])
    values=np.full(len(grid),np.nan)
    if valid.any():
        old=np.stack([np.interp(progress+grid[valid],old_s,previous[:,a]) for a in range(2)],axis=-1)
        new=np.stack([np.interp(grid[valid],new_s,current[:,a]) for a in range(2)],axis=-1)
        values[valid]=np.linalg.norm(new-old,axis=1)
    return values


def case_data(directory,record):
    metric=directory/'eval_metrics'/record['save_name']
    states=[json.loads(s) for s in (metric/'vehicle_state.jsonl').open()]
    bystep={s['step']:s for s in states}
    times=np.array([s['timestamp_s'] for s in states]);t0=times[0]
    assert np.all(np.diff(times)>0)
    velocity=np.array([s['velocity_mps'] for s in states]);forward=np.array([s['forward_vector'] for s in states])
    acceleration=np.array([s['acceleration'] for s in states])
    speed=np.sum(velocity*forward,axis=1);acc=np.sum(acceleration*forward,axis=1)
    jerk=np.gradient(acc,times)
    grid=np.linspace(2.5,10,31)
    last=None;heat=[];heat_t=[];planning_t=[];target=[];steer=[];brake=[]
    for line in (metric/'planning.jsonl').open():
        r=json.loads(line);policy=r['policy']
        if policy.get('phase')!='inference':continue
        s=bystep[r['step']];t=r['timestamp_s']-t0
        planning_t.append(t);target.append(policy.get('control_target_speed_mps',np.nan))
        steer.append(r['returned_control']['steer']);brake.append(r['returned_control']['brake'])
        trajectory=policy.get('predicted_trajectory_ego_m')
        if trajectory is not None:
            current=world_path(trajectory,s)
            if last is not None:heat.append(overlap_change(last,current,grid));heat_t.append(t)
            last=current
    assert heat
    return dict(time=times-t0,xy=np.array([s['location'][:2] for s in states]),speed=speed,
                jerk=jerk,planning_t=np.array(planning_t),target=np.array(target,dtype=float),
                steer=np.array(steer),brake=np.array(brake),heat=np.array(heat).T,
                heat_t=np.array(heat_t),distance_grid=grid,first_yaw=states[0]['rotation'][2],
                metric_directory=str(metric))


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    root=ROOT/'outputs/local_evaluation'
    payload={m:json.loads((root/n/'checkpoint_endpoint.json').read_text()) for m,n in RUNS.items()}
    records={m:{r['route_id']:r for r in d['_checkpoint']['records']} for m,d in payload.items()}
    assert all(set(r)==set(records['FM+CR']) for r in records.values())
    groups=defaultdict(list)
    for r in records['FM+CR'].values():groups[r['scenario_name']].append(r['route_id'])
    names=sorted(groups)
    deltas=[];scenario_rows=[]
    for scenario in names:
        ids=groups[scenario];assert len(ids)==5
        means={m:sum(records[m][k]['scores']['score_composed'] for k in ids)/5 for m in RUNS}
        deltas.append([means['FM+CR']-means['FM'],means['FM+CR']-means['FMSR']])
        scenario_rows.append(dict(scenario=scenario,**means))
    matrix=np.array(deltas)
    with (OUTPUT/'scenario_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['scenario',*RUNS]);w.writeheader();w.writerows(scenario_rows)
    fig,axes=plt.subplots(1,3,figsize=(17,10),gridspec_kw={'width_ratios':[1.6,1.5,1.25]},layout='constrained')
    vmax=max(10,float(np.ceil(abs(matrix).max()/10)*10))
    im=axes[0].imshow(matrix,vmin=-vmax,vmax=vmax,cmap='RdBu',aspect='auto')
    axes[0].set_yticks(range(len(names)),[s.removesuffix('_1') for s in names],fontsize=7)
    axes[0].set_xticks([0,1],['vs FM','vs FMSR'])
    for i in range(len(names)):
        for j in range(2):axes[0].text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=6.8,color='white' if abs(matrix[i,j])>.65*vmax else 'black')
    axes[0].set_title('(a) Mean DS difference: FM+CR minus baseline')
    fig.colorbar(im,ax=axes[0],label='DS difference',shrink=.5)
    keys=['collisions_vehicle','collisions_pedestrian','collisions_layout','red_light','vehicle_blocked']
    x=np.arange(len(keys))
    for i,m in enumerate(RUNS):
        g=payload[m]['_checkpoint']['global_record']['infractions']
        axes[1].bar(x+(i-1)*.25,[g[k] for k in keys],width=.25,color=COLORS[m],label=m)
    axes[1].set_xticks(x,['Vehicle\ncollision','Pedestrian\ncollision','Static\ncollision','Red\nlight','Blocked'])
    axes[1].set_ylabel('Events per driven kilometer');axes[1].legend()
    axes[1].set_title('(b) Safety-related event rates');axes[1].grid(axis='y',alpha=.2)
    for m in ['FM','FMSR']:
        diff=np.sort([records['FM+CR'][k]['scores']['score_composed']-records[m][k]['scores']['score_composed'] for k in records[m]])
        axes[2].step(diff,np.arange(1,221)/220,where='post',label=f'vs {m}',color=COLORS[m])
    axes[2].axvline(0,color='k',ls='--',lw=1);axes[2].set_xlabel('Route DS difference (FM+CR minus baseline)')
    axes[2].set_ylabel('Empirical cumulative fraction');axes[2].set_title('(c) All 220 route differences')
    axes[2].legend();axes[2].grid(alpha=.2)
    fig.suptitle('Recorded historical comparison; FM+CR uses latest-attempt failure-rerun composition',fontsize=12)
    save(fig,'fig1_scenario_safety_route_analysis')
    ranked=sorted(zip(names,matrix.mean(axis=1)),key=lambda z:z[1])
    # Two strongest scenario-level gains and one strongest loss; median route gain within each.
    selected=[ranked[-1][0],ranked[-2][0],ranked[0][0]]
    cases=[]
    for case_index,scenario in enumerate(selected,1):
        ids=groups[scenario]
        ids=sorted(ids,key=lambda k:records['FM+CR'][k]['scores']['score_composed']-
                   (records['FM'][k]['scores']['score_composed']+records['FMSR'][k]['scores']['score_composed'])/2)
        route_id=ids[len(ids)//2]
        trace={m:case_data(root/n,records[m][route_id]) for m,n in RUNS.items()}
        origin=trace['FMSR']['xy'][0];yaw=np.deg2rad(trace['FMSR']['first_yaw'])
        rotation=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        xy={m:(t['xy']-origin)@rotation for m,t in trace.items()}
        xmax=max(float(t['time'][-1]) for t in trace.values());window=min(60,xmax)
        speed_values=np.concatenate([np.r_[t['speed'][t['time']<=window],
                                           t['target'][t['planning_t']<=window]] for t in trace.values()])
        speed_values=speed_values[np.isfinite(speed_values)]
        speed_limits=(min(-.5,float(speed_values.min())-1),float(speed_values.max())+1)
        jerk_limit=max(1,max(float(abs(t['jerk'][t['time']<=window]).max()) for t in trace.values()))*1.05
        fig,axes=plt.subplots(5,3,figsize=(13.5,11),layout='constrained',gridspec_kw={'height_ratios':[1.3,1,1,1,1.1]})
        mins=np.min(np.concatenate(list(xy.values())),axis=0);maxs=np.max(np.concatenate(list(xy.values())),axis=0)
        heat_stats={}
        for j,m in enumerate(RUNS):
            t=trace[m];r=records[m][route_id]
            for other in RUNS:
                axes[0,j].plot(xy[other][:,0],xy[other][:,1],color=COLORS[other] if other==m else '#cccccc',lw=1.8 if other==m else .7)
            axes[0,j].plot(xy[m][0,0],xy[m][0,1],'ko',ms=3)
            axes[0,j].set(xlabel='Initial ego-frame x (m)',ylabel='Initial ego-frame y (m)',
                          title=f"{m}: DS={r['scores']['score_composed']:.1f}, RC={r['scores']['score_route']:.1f}",
                          xlim=(mins[0]-2,maxs[0]+2),ylim=(mins[1]-2,maxs[1]+2))
            axes[0,j].set_aspect('equal',adjustable='box')
            axes[1,j].plot(t['time'],t['speed'],color=COLORS[m],label='Actual longitudinal speed')
            axes[1,j].plot(t['planning_t'],t['target'],'k--',lw=1,label='Control target speed')
            axes[1,j].set_ylabel('Speed (m/s)');axes[1,j].legend(fontsize=6)
            axes[1,j].set_ylim(*speed_limits)
            axes[2,j].plot(t['planning_t'],t['steer'],color=COLORS[m],label='Steer')
            axes[2,j].plot(t['planning_t'],t['brake'],'k--',lw=1,label='Brake')
            axes[2,j].set_ylabel('Control');axes[2,j].legend(fontsize=6)
            axes[2,j].set_ylim(-1.05,1.05)
            axes[3,j].plot(t['time'],t['jerk'],color=COLORS[m],lw=.65)
            axes[3,j].set_ylabel('Longitudinal jerk (m/s³)')
            axes[3,j].set_ylim(-jerk_limit,jerk_limit)
            visible=t['heat_t']<=window
            tt=t['heat_t'][visible];h=t['heat'][:,visible]
            cmap=plt.get_cmap('magma').copy();cmap.set_bad('#dddddd')
            image=axes[4,j].pcolormesh(tt,t['distance_grid'],np.ma.masked_invalid(h),shading='nearest',vmin=0,vmax=2,cmap=cmap,rasterized=True)
            axes[4,j].set_ylabel('Ahead distance (m)');axes[4,j].set_xlabel('Elapsed simulation time per run (s)')
            fig.colorbar(image,ax=axes[4,j],label='Replanning change (m)',extend='max',shrink=.85)
            for ax in axes[1:4,j]:ax.set_xlim(0,window);ax.grid(alpha=.2)
            axes[4,j].set_xlim(0,window)
            heat_stats[m]=dict(valid_samples=int(np.isfinite(h).sum()),
                               mean=float(np.nanmean(h)) if np.isfinite(h).any() else None,
                               p95=float(np.nanpercentile(h,95)) if np.isfinite(h).any() else None,
                               fraction_above_display_max=float(np.mean(h[np.isfinite(h)]>2)) if np.isfinite(h).any() else None)
            np.savez_compressed(OUTPUT/f'case{case_index}_{m.replace("+","")}_heatmap.npz',
                                elapsed_seconds=tt,ahead_m=t['distance_grid'],change_m=h)
        fig.suptitle(f'{scenario.removesuffix("_1")} | {route_id}\nIndependent recorded runs; full ego paths above, first {window:.0f}s below; gray = insufficient overlap',fontsize=11)
        save(fig,f'fig2_case{case_index}_{scenario.removesuffix("_1")}')
        cases.append(dict(route_id=route_id,scenario=scenario,
                          selection='Median route gain in one of two highest-gain / one lowest-gain scenarios',
                          scores={m:records[m][route_id]['scores'] for m in RUNS},
                          status={m:records[m][route_id]['status'] for m in RUNS},
                          heatmap_window_seconds=window,heatmap_statistics=heat_stats,
                          metric_sources={m:t['metric_directory'] for m,t in trace.items()}))
    manifest=dict(sources={m:str((root/n).resolve()) for m,n in RUNS.items()},
                  scope='Historical comparison; FM+CR is a failure-rerun composite, not controlled CR-only ablation',
                  heatmap_definition='L2 change between consecutive paths after world-coordinate and arclength alignment',
                  heatmap_grid_m=[2.5,10],heatmap_color_range_m=[0,2],
                  heatmap_not_a_GP_kernel=True,cases=cases)
    dump(OUTPUT/'figure_manifest.json',manifest)
    print(json.dumps(dict(output=str(OUTPUT),cases=[(c['route_id'],c['scenario']) for c in cases]),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
