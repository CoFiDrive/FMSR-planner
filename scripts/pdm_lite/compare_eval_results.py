"""Descriptive paired-route comparison; keep failed routes and original scores."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np


def success(record):
    return record['scores']['score_route'] == 100 and not any(
        value for key,value in record['infractions'].items() if key != 'min_speed_infractions')


def read(path):
    data=json.loads(Path(path).read_text())
    records=data['_checkpoint']['records']
    result={r['route_id']:r for r in records}
    if len(result)!=len(records):raise ValueError('Duplicate route_id; compare matching repetitions explicitly')
    return result


def compare(baseline,candidate):
    first,second=read(baseline),read(candidate)
    rows=[]
    for route in sorted(first.keys() & second.keys()):
        a,b=first[route],second[route]
        for key in ('scenario_name','town_name','weather_id'):
            if a.get(key)!=b.get(key):raise ValueError(f'{route}: mismatched {key}')
        sa,sb=success(a),success(b)
        rows.append({'route_id':route,'scenario_name':a.get('scenario_name'),
                     'baseline_DS':a['scores']['score_composed'],'candidate_DS':b['scores']['score_composed'],
                     'delta_DS':b['scores']['score_composed']-a['scores']['score_composed'],
                     'baseline_RS':sa,'candidate_RS':sb,
                     'transition':'retained' if sa and sb else 'rescued' if sb else 'lost' if sa else 'both_failed',
                     'baseline_status':a['status'],'candidate_status':b['status'],
                     'baseline_infraction_counts':{k:len(v) for k,v in a['infractions'].items()},
                     'candidate_infraction_counts':{k:len(v) for k,v in b['infractions'].items()}})
    if not rows:raise ValueError('No matching route IDs')
    delta=np.array([r['delta_DS'] for r in rows])
    groups=defaultdict(list)
    for r in rows:groups[r['scenario_name']].append(r)
    rng=np.random.default_rng(0)
    bootstrap=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(axis=1)
    return {'baseline':str(Path(baseline).resolve()),'candidate':str(Path(candidate).resolve()),
            'matched_route_count':len(rows), 'baseline_only_routes':sorted(first.keys()-second.keys()),
            'candidate_only_routes':sorted(second.keys()-first.keys()),
            'delta_DS_mean':float(delta.mean()),
            'delta_RS_percentage_points':100*sum(int(r['candidate_RS'])-int(r['baseline_RS']) for r in rows)/len(rows),
            'success_transitions':{key:sum(r['transition']==key for r in rows)
                                   for key in ('retained','rescued','lost','both_failed')},
            'route_bootstrap_delta_DS_95CI':np.percentile(bootstrap,[2.5,97.5]).tolist(),
            'inference_scope':'descriptive comparison of historical matched routes; '
                              'CI resamples routes only, not training/traffic randomness or independent seeds',
            'scenario_groups':{key:{'count':len(group),'mean_delta_DS':sum(r['delta_DS'] for r in group)/len(group),
                                    'rescued':sum(r['transition']=='rescued' for r in group),
                                    'lost':sum(r['transition']=='lost' for r in group)} for key,group in groups.items()},
            'routes':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    report=compare(args.baseline,args.candidate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('routes','scenario_groups')},indent=2))


if __name__=='__main__':main()
