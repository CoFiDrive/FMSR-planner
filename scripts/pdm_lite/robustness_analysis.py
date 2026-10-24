"""Automatic route-level behaviors, paired baseline degradation, and figures."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from scripts.pdm_lite.analyze_planning_metrics import analyze,statistics


def stats(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    result=statistics(a)
    if result is not None:result.update(median=float(np.median(a)),p95=float(np.percentile(a,95)),median_abs=float(np.median(abs(a))))
    return result


def read(path):
    if path.exists():return path.open()
    return gzip.open(path.with_name(path.name+'.gz'),'rt')


def run_analysis(output):
    benchmark=json.loads((output/'analysis/benchmark_metrics.json').read_text())
    job=json.loads((output/'job.json').read_text());behaviors=[]
    for route in benchmark['routes']:
        directory=output/'eval_metrics'/route['save_name']
        diagnostic=analyze(directory)
        with read(directory/'vehicle_state.jsonl') as f:states=[json.loads(line) for line in f]
        with read(directory/'planning.jsonl') as f:plans=[json.loads(line) for line in f]
        plans=[r for r in plans if r.get('policy',{}).get('phase')=='inference']
        xyz=np.asarray([r['location'] for r in states]);time=np.asarray([r['timestamp_s'] for r in states])
        velocity=np.asarray([r['velocity_mps'] for r in states]);speed=np.linalg.norm(velocity,axis=1)
        distance=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum())
        times=np.asarray([r['timestamp_s'] for r in plans])
        target=np.asarray([r['policy'].get('control_target_speed_mps',np.nan) for r in plans],float)
        steer=np.asarray([r['returned_control']['steer'] for r in plans])
        brake=np.asarray([r['returned_control']['brake'] for r in plans])
        error=[];event_times=[]
        path=directory/'perturbations.jsonl'
        with read(path) as f:
            for line in f:
                row=json.loads(line)
                for event in row['events']:
                    if event['family']!=job['perturbation']['family']:error.append(event)
                    event_times.append(row['timestamp_s'])
        if error:raise ValueError('Unexpected perturbation events')
        # Episodes that end before the injection window are reported, never hidden.
        eligible=bool(plans) and any(r['policy'].get('robustness',{}).get('relative_time_s',0)>=job['perturbation'].get('start_s',0) for r in plans)
        if job['perturbation']['family']!='none' and eligible and not event_times:
            raise ValueError(f'Perturbation was never injected on eligible route {route["route_id"]}')
        summary={k:v for k,v in diagnostic.items() if k!='series'}
        acceleration=np.asarray([r['acceleration'] for r in states])
        forward=np.asarray([r['forward_vector'] for r in states]);right=np.asarray([r['right_vector'] for r in states])
        lon=np.sum(acceleration*forward,axis=1);lat=np.sum(acceleration*right,axis=1)
        summary.update(longitudinal_acceleration_mps2=stats(lon),lateral_acceleration_mps2=stats(lat),
                       longitudinal_jerk_mps3=stats(np.gradient(lon,time)),lateral_jerk_mps3=stats(np.gradient(lat,time)),
                       spatial_plan_deviation_m=stats([r['spatial_plan_deviation']['mean_m'] for r in diagnostic['series'] if r['spatial_plan_deviation']]),
                       agent_policy_wall_ms=stats([r['agent_policy_wall_ms'] for r in diagnostic['series'] if r['agent_policy_wall_ms'] is not None]),
                       planner_gpu_ms=stats([v for r in diagnostic['series'] for v in r['planner_gpu_ms'] if v is not None]))
        summary.update(route_id=route['route_id'],scenario=route['scenario_name'],
                       driven_distance_m=distance,injection_event_count=len(event_times),injection_eligible=eligible,
                       target_speed_jump_mps=stats(np.abs(np.diff(target))),
                       steering_jump=stats(np.abs(np.diff(steer))),brake_jump=stats(np.abs(np.diff(brake))),
                       low_speed_fraction=float(np.mean(speed<.5)),
                       first_moving_time_s=float(time[np.argmax(speed>.5)]-time[0]) if np.any(speed>.5) else None)
        actors_path=directory/'actors.jsonl'
        state_by_step={s['step']:s for s in states}
        plan_by_step={p['step']:p for p in plans}
        distances={'vehicle':[],'pedestrian':[]};ttc=[];risk=[];actor_times=[];brakes=[]
        with read(actors_path) as f:
            for line in f:
                a=json.loads(line);s=state_by_step.get(a['step'])
                if s is None:continue
                origin=np.asarray(s['location'][:2]);own_v=np.asarray(s['velocity_mps'][:2])
                current_ttc=[]
                for actor in a['actors']:
                    kind='vehicle' if actor['type'].startswith('vehicle.') else ('pedestrian' if actor['type'].startswith('walker.') else None)
                    if kind is None:continue
                    position=np.asarray(actor['position_m'][:2])-origin;distance_to_actor=np.linalg.norm(position)
                    distances[kind].append(distance_to_actor)
                    relative_v=np.asarray(actor['velocity_mps'][:2])-own_v
                    closing=-float(np.dot(position,relative_v))/max(distance_to_actor,1e-6)
                    if closing>.5:current_ttc.append(distance_to_actor/closing)
                minimum=min(current_ttc) if current_ttc else np.inf
                if np.isfinite(minimum):ttc.append(minimum)
                actor_times.append(a['timestamp_s']);risk.append(minimum<3.)
                brakes.append(plan_by_step.get(a['step'],{}).get('returned_control',{}).get('brake',np.nan))
        actor_times=np.array(actor_times);risk=np.array(risk,dtype=bool);brakes=np.array(brakes)
        onset=np.flatnonzero(risk & ~np.r_[False,risk[:-1]]) if len(risk) else []
        response=[];missed=0;truncated=0
        for i in onset:
            window=(actor_times>=actor_times[i])&(actor_times<=actor_times[i]+3.)
            hits=np.flatnonzero(window&(brakes>=.1))
            if len(hits):response.append(actor_times[hits[0]]-actor_times[i])
            else:
                missed+=1;truncated+=int(actor_times[-1]<actor_times[i]+3.)
        summary.update(nearby_vehicle_center_distance_m=stats(distances['vehicle']),
                       nearby_pedestrian_center_distance_m=stats(distances['pedestrian']),
                       center_closing_time_s=stats(ttc),heuristic_risk_event_count=len(onset),
                       heuristic_brake_response_s=stats(response),heuristic_no_brake_within_3s=missed,
                       heuristic_truncated_response_windows=truncated,
                       interaction_protocol='Every 0.1s, actors within 50m. Center distance / radial closing speed, not geometric TTC or required braking; risk<3s and brake>=0.1. Descriptive association only.')
        # Store true vehicle path to allow reproducible pulse comparisons after gzip archival.
        relative=np.asarray([r['policy'].get('robustness',{}).get('relative_time_s',np.nan) for r in plans])
        np.savez_compressed(output/'analysis'/f'{route["route_id"]}_behavior.npz',
                            time=time-time[0],xyz=xyz,speed=speed,planning_time=relative,target=target,
                            steer=steer,brake=brake,injection_agent_time=np.asarray(event_times),
                            injection_relative_s=np.asarray(event_times)-states[0]['agent_timestamp_s'])
        behaviors.append(summary)
    (output/'analysis/behavior_metrics.json').write_text(json.dumps(dict(job=job,routes=behaviors),indent=2)+'\n')
    print('Analyzed',len(behaviors),'routes',output,flush=True)


def campaign_analysis(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((output/'campaign.json').read_text());analysis=output/'analysis';analysis.mkdir(exist_ok=True)
    results={};rows=[]
    for job in manifest['jobs']:
        directory=output/'runs'/job['id']
        if not (directory/'complete.json').exists():continue
        report=json.loads((directory/'analysis/benchmark_metrics.json').read_text())
        behavior=json.loads((directory/'analysis/behavior_metrics.json').read_text())
        lookup={r['route_id']:r for r in behavior['routes']}
        results[(job['phase'],job['seed'],job['model'],job['perturbation']['id'])]=(job,report,lookup,directory)
        total_distance=sum(r['driven_distance_m'] for r in behavior['routes'])/1000
        scores=report['global_scores_mean']
        infra={k:sum(len(r['infractions'].get(k,[])) for r in report['routes']) for k in report['routes'][0]['infractions']}
        row=dict(phase=job['phase'],seed=job['seed'],model=job['model'],condition=job['perturbation']['id'],
                 family=job['perturbation']['family'],axis=job['perturbation'].get('axis',job['perturbation'].get('component','heading')),
                 amplitude=job['perturbation']['amplitude'],mode=job['perturbation']['mode'],
                 route_count=report['route_count'],DS=scores['score_composed'],RC=scores['score_route'],IS=scores['score_penalty'],
                 SR=report['route_success_percent'],Efficiency=report['efficiency_percent'],Comfort=report['comfort_percent'],driven_km=total_distance)
        for key,count in infra.items():row[key+'_events']=count;row[key+'_per_km']=count/total_distance if total_distance else None
        rows.append(row)
    def dump_csv(name,records):
        if not records:return
        keys=list(dict.fromkeys(k for r in records for k in r))
        with (analysis/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
    dump_csv('run_metrics.csv',rows)
    paired=[];pulse=[]
    for key,(job,report,behavior,directory) in results.items():
        phase,seed,model,condition=key
        if condition=='clean':continue
        baseline=results.get((phase,seed,model,'clean'))
        if not baseline:continue
        _,clean,clean_behavior,base_directory=baseline
        base={r['route_id']:r for r in clean['routes']}
        for r in report['routes']:
            b=base[r['route_id']];rid=r['route_id'];v=behavior[rid];bv=clean_behavior[rid]
            row=dict(phase=phase,seed=seed,model=model,condition=condition,family=job['perturbation']['family'],
                     mode=job['perturbation']['mode'],amplitude=job['perturbation']['amplitude'],
                     axis=job['perturbation'].get('axis',job['perturbation'].get('component','heading')),
                     route_id=rid,scenario=r['scenario_name'],clean_DS=b['scores']['score_composed'],
                     perturbed_DS=r['scores']['score_composed'],DS_degradation=b['scores']['score_composed']-r['scores']['score_composed'],
                     SR_degradation=float(b['route_success'])-float(r['route_success']),
                     injection_eligible=v['injection_eligible'],injection_count=v['injection_event_count'])
            for scalar in ('mode_switch_fraction','low_speed_fraction','first_moving_time_s'):
                if v.get(scalar) is not None and bv.get(scalar) is not None:row[scalar+'_increment']=v[scalar]-bv[scalar]
            for infra,events in r['infractions'].items():row[infra+'_increment']=len(events)-len(b['infractions'].get(infra,[]))
            for metric in ('spatial_plan_deviation_m','target_speed_jump_mps','steering_jump','brake_jump',
                           'longitudinal_acceleration_mps2','lateral_acceleration_mps2',
                           'longitudinal_jerk_mps3','lateral_jerk_mps3','agent_policy_wall_ms','planner_gpu_ms'):
                for quantile in ('median','p95_abs'):
                    x=v.get(metric);y=bv.get(metric)
                    if x and y and quantile in x and quantile in y:row[metric+'_'+quantile+'_increment']=x[quantile]-y[quantile]
            paired.append(row)
            if job['perturbation']['mode']=='pulse' and v['injection_event_count']:
                # Progress-aligned path discrepancy relative to the clean run, not expert tracking error.
                a=np.load(directory/'analysis'/f'{rid}_behavior.npz');bdata=np.load(base_directory/'analysis'/f'{rid}_behavior.npz')
                xy=a['xyz'][:,:2];bxy=bdata['xyz'][:,:2]
                s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
                bs=np.r_[0,np.cumsum(np.linalg.norm(np.diff(bxy,axis=0),axis=1))]
                keep=np.r_[True,np.diff(bs)>1e-5];bs=bs[keep];bxy=bxy[keep]
                baseline_xy=np.column_stack([np.interp(s,bs,bxy[:,i]) for i in range(2)])
                sep=np.linalg.norm(xy-baseline_xy,axis=1);t=a['time']
                start=float(a['injection_relative_s'][0]) if 'injection_relative_s' in a else job['perturbation']['start_s']
                window=(t>=start)&(t<=start+20)&(s<=bs[-1])
                recover=None
                indices=np.flatnonzero(window)
                # Require below 0.25 m for 1 s within a 20 s observation window.
                peak_index=int(indices[np.argmax(sep[window])]) if len(indices) else len(t)
                recovery_candidates=[i for i in indices if i>=peak_index]
                if len(indices) and np.max(sep[window])<=.25:
                    recovery_candidates=list(indices)
                for index in recovery_candidates:
                    end=np.searchsorted(t,t[index]+1.)
                    if end<len(t) and t[end]<=start+20 and np.all(sep[index:end+1]<=.25):recover=float(t[index]-start);break
                pulse.append(dict(phase=phase,seed=seed,model=model,condition=condition,route_id=rid,
                                  recovered=recover is not None,recovery_time_s=recover,
                                  full_observation_window=bool(len(t) and t[-1]>=start+20),
                                  observation_available=bool(len(indices)),max_progress_aligned_separation_m=float(sep[window].max()) if np.any(window) else None))
    dump_csv('paired_route_degradation.csv',paired);dump_csv('pulse_recovery.csv',pulse)
    clusters=[];rng=np.random.default_rng(137)
    # Seed-averaged per-scenario paired effect; bootstrap scenario units, preserving model pairs.
    for phase in ('smoke','pilot','full'):
        condition_ids=sorted({r['condition'] for r in paired if r['phase']==phase})
        for condition in condition_ids:
            selected=[r for r in paired if r['phase']==phase and r['condition']==condition]
            index={(r['seed'],r['route_id'],r['model']):r for r in selected}
            points=[]
            for k,r in index.items():
                if k[2]!='FM':continue
                cr=index.get((k[0],k[1],'FM_CR'))
                if cr:points.append((r['scenario'],r['DS_degradation']-cr['DS_degradation']))
            if not points:continue
            names=sorted({s for s,_ in points});values=np.array([np.mean([v for s,v in points if s==name]) for name in names])
            boot=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            clusters.append(dict(phase=phase,condition=condition,paired_route_seed_count=len(points),scenario_count=len(names),
                                 CR_degradation_advantage_DS=float(values.mean()),CI_low=float(np.quantile(boot,.025)) if len(names)>=5 else None,
                                 CI_high=float(np.quantile(boot,.975)) if len(names)>=5 else None,seeds=sorted({r['seed'] for r in selected})))
    dump_csv('paired_CR_advantage.csv',clusters)
    # Effect figure with scenario-paired intervals; exclude unmatched incomplete model pairs.
    for phase in ('pilot','full'):
        entries=[r for r in clusters if r['phase']==phase and r['CI_low'] is not None]
        if not entries:continue
        fig,ax=plt.subplots(figsize=(9,max(3,.32*len(entries))),layout='constrained')
        names=[r['condition'] for r in entries];y=np.arange(len(entries))
        means=np.array([r['CR_degradation_advantage_DS'] for r in entries])
        low=np.array([r['CI_low'] for r in entries]);high=np.array([r['CI_high'] for r in entries])
        ax.errorbar(means,y,xerr=np.vstack([np.maximum(0,means-low),np.maximum(0,high-means)]),fmt='o',color='#2878B5',capsize=3)
        ax.axvline(0,color='.4',ls='--');ax.set(yticks=y,yticklabels=names,xlabel='FM DS loss minus FM+CR DS loss (positive favors CR)',
                                                         title=f'{phase}: paired scenario-bootstrap 95% intervals; provisional')
        ax.grid(axis='x',alpha=.2)
        for ext in ('png','pdf'):fig.savefig(analysis/f'{phase}_paired_advantage.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    for phase in ('pilot','full'):
        selected=[r for r in paired if r['phase']==phase and r['mode']=='continuous']
        groups=sorted({(r['family'],r['axis']) for r in selected})
        if not groups:continue
        fig,axes=plt.subplots(len(groups),2,figsize=(10,max(3,2.7*len(groups))),squeeze=False,layout='constrained')
        for row,(family,axis) in enumerate(groups):
            for model,color in [('FM','#2878B5'),('FM_CR','#D95319')]:
                rr=[r for r in selected if r['family']==family and r['axis']==axis and r['model']==model]
                amps=sorted({r['amplitude'] for r in rr})
                for col,metric in enumerate(('DS_degradation','SR_degradation')):
                    means=[0.]+[np.mean([r[metric] for r in rr if r['amplitude']==a])*(100 if col else 1) for a in amps]
                    axes[row,col].plot([0.]+amps,means,'-o',color=color,label=model.replace('_','+'))
                    axes[row,col].set(title=f'{family}/{axis}',xlabel='Perturbation amplitude',ylabel=['DS loss from clean','SR loss (percentage points)'][col])
                    axes[row,col].axhline(0,color='.5',ls='--',lw=.7);axes[row,col].legend();axes[row,col].grid(alpha=.2)
        fig.suptitle(f'{phase}: completed paired runs only; progress is provisional')
        for ext in ('png','pdf'):fig.savefig(analysis/f'{phase}_robustness_degradation.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
        measures=[('spatial_plan_deviation_m_p95_abs_increment','Replanning deviation P95 change (m)'),
                  ('target_speed_jump_mps_p95_abs_increment','Target-speed jump P95 change (m/s)'),
                  ('lateral_jerk_mps3_p95_abs_increment','Lateral jerk P95 change (m/s³)')]
        fig,axes=plt.subplots(len(groups),3,figsize=(13,max(3,2.5*len(groups))),squeeze=False,layout='constrained')
        for row,(family,axis) in enumerate(groups):
            for model,color in [('FM','#2878B5'),('FM_CR','#D95319')]:
                rr=[r for r in selected if r['family']==family and r['axis']==axis and r['model']==model]
                amps=sorted({r['amplitude'] for r in rr})
                for col,(metric,label) in enumerate(measures):
                    means=[0.]+[float(np.mean([r[metric] for r in rr if r['amplitude']==a and metric in r]))
                                 if any(r['amplitude']==a and metric in r for r in rr) else np.nan for a in amps]
                    axes[row,col].plot([0.]+amps,means,'-o',color=color,label=model.replace('_','+'))
                    axes[row,col].set(title=f'{family}/{axis}',xlabel='Perturbation amplitude',ylabel=label)
                    axes[row,col].axhline(0,color='.5',ls='--',lw=.7);axes[row,col].legend();axes[row,col].grid(alpha=.2)
        fig.suptitle(f'{phase}: route-level behavior changes relative to clean runs')
        for ext in ('png','pdf'):fig.savefig(analysis/f'{phase}_behavior_changes.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
        safety=['collisions_vehicle','collisions_pedestrian','collisions_layout','outside_route_lanes','red_light','vehicle_blocked','route_timeout']
        fig,axes=plt.subplots(1,2,figsize=(13,max(4,.28*len(groups)*3)),layout='constrained')
        condition_ids=sorted({r['condition'] for r in selected})
        matrices={}
        for model in ('FM','FM_CR'):
            matrices[model]=np.array([[np.mean([r.get(event+'_increment',0) for r in selected if r['condition']==condition and r['model']==model])
                                      if any(r['condition']==condition and r['model']==model for r in selected) else np.nan
                                      for event in safety] for condition in condition_ids])
        finite=np.concatenate([a[np.isfinite(a)] for a in matrices.values()])
        cap=max(1.,float(np.max(np.abs(finite)))) if len(finite) else 1.
        for ax,model in zip(axes,['FM','FM_CR']):
            matrix=matrices[model]
            im=ax.imshow(matrix,aspect='auto',cmap='RdBu_r',vmin=-cap,vmax=cap)
            ax.set(yticks=range(len(condition_ids)),yticklabels=condition_ids,xticks=range(len(safety)),
                   xticklabels=[s.replace('collisions_','').replace('outside_route_lanes','lane departure') for s in safety],
                   title=model.replace('_','+'))
            ax.tick_params(axis='x',rotation=40,labelsize=8);ax.tick_params(axis='y',labelsize=7)
            fig.colorbar(im,ax=ax,label='Event-count increase per route from clean')
        fig.suptitle(f'{phase}: disturbance-associated safety event increments')
        for ext in ('png','pdf'):fig.savefig(analysis/f'{phase}_safety_increments.{ext}',dpi=220,bbox_inches='tight')
        plt.close(fig)
    summary=dict(completed_jobs=len(rows),planned_jobs=len(manifest['jobs']),paired_route_rows=len(paired),
                 status='complete' if len(rows)==len(manifest['jobs']) else 'partial',
                 limits='One trained checkpoint per method; seeds cover environment and perturbations, not training variability. Pulse recovery is progress-aligned clean-run discrepancy, not GT error. No failure-selected merges.')
    (analysis/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (analysis/'REPORT.md').write_text('# 自动抗扰动分析\n\n'+f'完成 {len(rows)}/{len(manifest["jobs"])} 组；配对路线记录 {len(paired)} 条。\n\n'
        'run_metrics.csv：DS/RC/IS/SR/Efficiency/Comfort、完整事件数和实际每公里频率。\n\n'
        'paired_route_degradation.csv：相对各自无扰动基线的路线退化及行为指标增量。\n\n'
        'paired_CR_advantage.csv：FM退化减去FM+CR退化，正值表示CR退化更少；按场景bootstrap区间。\n\n'
        'pulse_recovery.csv：相对同路线clean运行的进度对齐路径偏差，窗口峰值后回到0.25m内持续1s、观察20s；未恢复样本保留。'
        '不是专家误差，未覆盖完整20s的样本需单独查看。\n\n'
        '种子覆盖交通管理器和干扰，不代表多训练种子。正在进行的结果均为暂定结果。\n')
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--campaign',type=Path)
    args=parser.parse_args()
    if args.run:run_analysis(args.run)
    elif args.campaign:campaign_analysis(args.campaign)
    else:parser.error('Choose --run or --campaign')


if __name__=='__main__':main()
