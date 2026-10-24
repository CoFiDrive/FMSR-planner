"""Same-route score comparison and stationary-period evidence for hybrid pilot."""
import argparse
import collections
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'outputs/local_evaluation/bench2drive32_tie_spectral_epoch09_tfpp_speed_20261007'
ANALYSIS=ROOT/'outputs/analysis/fm_tfpp_speed_closed_loop_20261007'
BASE=ROOT/'outputs/local_evaluation/bench2drive220_tie_spectral_10epochs_seed0_20261006_223036_epoch09/checkpoint_endpoint.json'
CLEAN=ROOT/'outputs/local_evaluation/bench2drive220_clean_prior_20261006_015234_epoch09_seed0/checkpoint_endpoint.json'


def success(row):
    return row['status'] in ('Completed','Perfect') and all(not v for k,v in row['infractions'].items() if k!='min_speed_infractions')


def metrics(rows):
    if not rows:return {'n':0}
    return {'n':len(rows),'DS':statistics.mean(r['scores']['score_composed'] for r in rows),
        'RC':statistics.mean(r['scores']['score_route'] for r in rows),'success':sum(success(r) for r in rows),
        'SR_percent':100*sum(success(r) for r in rows)/len(rows),
        'statuses':dict(collections.Counter(r['status'] for r in rows)),
        'routes_with_infractions':{k:sum(bool(r['infractions'].get(k)) for r in rows) for k in rows[0]['infractions']}}


def scalar(value):
    while isinstance(value,list) and len(value)==1:value=value[0]
    return float(value) if isinstance(value,(int,float)) else None


def diagnostic(path):
    if not path.exists():return {'available':False}
    n=0;gap_sum=0.;commanded_no_motion=0;low_target=0;creep_override=0;stop_override=0;high_tfpp_low_planner=0
    current={};longest={};evidence=[];previous=None
    for line in path.open():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        policy=row['policy']
        if policy.get('phase')!='inference' or policy.get('filter_initial_delay_brake'):continue
        comparison=policy.get('speed_comparison') or []
        if not comparison or not comparison[0]:continue
        pair=comparison[0];planner=scalar(pair.get('planner_speed_mps'));tfpp=scalar(pair.get('tfpp_speed_mps'))
        actual=scalar(policy.get('measured_speed_mps'));target=scalar(policy.get('control_target_speed_mps'))
        if any(v is None for v in (planner,tfpp,actual,target)):continue
        now=row['world_timestamp_s'];dt=min(.1,max(0.,now-previous)) if previous is not None else .05;previous=now
        n+=1;gap_sum+=tfpp-planner;overrides=policy.get('overrides') or []
        # Logged longitudinal velocity is signed: reversing is motion.
        stationary=abs(actual)<.5
        conditions={'low_target_stationary':stationary and target<.5,
            'commanded_motion_stationary':stationary and target>=2.,
            'tfpp_go_planner_stop':tfpp>=2. and planner<.5,
            'override_brake_stationary':stationary and any(k in overrides for k in ['creep_safety_box_brake','stop_sign_brake'])}
        low_target+=int(conditions['low_target_stationary']);commanded_no_motion+=int(conditions['commanded_motion_stationary'])
        high_tfpp_low_planner+=int(conditions['tfpp_go_planner_stop']);creep_override+=int('creep_safety_box_brake' in overrides);stop_override+=int('stop_sign_brake' in overrides)
        for key,active in conditions.items():
            current[key]=current.get(key,0.)+dt if active else 0.
            longest[key]=max(longest.get(key,0.),current[key])
        if (conditions['commanded_motion_stationary'] or conditions['tfpp_go_planner_stop']) and len(evidence)<10:
            evidence.append({'step':row['step'],'time_s':now,'measured_speed_mps':actual,'planner_speed_mps':planner,
                'tfpp_speed_mps':tfpp,'tfpp_stop_probability':scalar(pair.get('tfpp_stop_probability')),
                'returned_control':row['returned_control'],'controller_control':policy.get('controller_control'),
                'overrides':overrides,'route_completion_percent':row.get('route_completion_percent')})
    return {'available':True,'inference_frames':n,'mean_tfpp_minus_planner_speed_mps':gap_sum/n if n else None,
        'low_target_stationary_frames':low_target,'commanded_motion_stationary_frames':commanded_no_motion,
        'tfpp_go_planner_stop_frames':high_tfpp_low_planner,'creep_safety_box_brake_frames':creep_override,
        'stop_sign_brake_frames':stop_override,'longest_contiguous_seconds':longest,'example_frames':evidence,
        'interpretation':'stationary/override categories are evidence, not a proof that each stop was illegal'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--compact',action='store_true');args=parser.parse_args()
    baseline={r['route_id']:r for r in json.loads(BASE.read_text())['_checkpoint']['records']}
    clean={r['route_id']:r for r in json.loads(CLEAN.read_text())['_checkpoint']['records']}
    endpoint=args.output/'checkpoint_endpoint.json'
    if not endpoint.exists():
        print('No route records yet');return
    data=json.loads(endpoint.read_text());cp=data['_checkpoint'];rows=cp['records']
    rows=[r for r in rows if r.get('scores') and not r['status'].startswith('Failed - Agent couldn')]
    pairs=[]
    for row in rows:
        b=baseline[row['route_id']];c=clean[row['route_id']]
        path=args.output/'diagnostics'/row['save_name']/'frames.jsonl'
        pairs.append({'route_id':row['route_id'],'scenario':row.get('scenario_name'),
            'baseline_DS':b['scores']['score_composed'],'hybrid_DS':row['scores']['score_composed'],
            'clean_DS':c['scores']['score_composed'],'baseline_status':b['status'],'hybrid_status':row['status'],
            'baseline_success':success(b),'hybrid_success':success(row),'clean_success':success(c),
            'diagnostic':diagnostic(path)})
    summary={'progress':cp['progress'],'entry_status':data['entry_status'],
        'complete':data['entry_status']=='Finished' and cp['progress']==[32,32],
        'hybrid':metrics(rows),'baseline_same_routes':metrics([baseline[r['route_id']] for r in rows]),
        'clean_same_routes':metrics([clean[r['route_id']] for r in rows]),
        'recovered_successes':sum(not x['baseline_success'] and x['hybrid_success'] for x in pairs),
        'lost_successes':sum(x['baseline_success'] and not x['hybrid_success'] for x in pairs),
        'pairs':pairs,'scope':'development pilot; historical same-route reference, single-run comparisons'}
    temporary=ANALYSIS/'pilot_summary.tmp.json';temporary.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');temporary.replace(ANALYSIS/'pilot_summary.json')
    if args.compact:
        fields=('n','DS','RC','success','SR_percent','statuses')
        visible={'progress':summary['progress'],'complete':summary['complete'],
            'recovered_successes':summary['recovered_successes'],'lost_successes':summary['lost_successes']}
        for name in ('hybrid','baseline_same_routes','clean_same_routes'):
            visible[name]={k:v for k,v in summary[name].items() if k in fields}
    else:
        visible={k:v for k,v in summary.items() if k!='pairs'}
    print(json.dumps(visible,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
