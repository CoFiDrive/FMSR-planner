"""Extract pre-incident telemetry without claiming historical causal attribution."""
import json
from pathlib import Path
from analyze_tfpp_speed_pilot import DEFAULT, ANALYSIS, scalar


def main():
    cp=json.loads((DEFAULT/'checkpoint_endpoint.json').read_text())['_checkpoint']
    result=[]
    for route in cp['records']:
        folder=DEFAULT/'diagnostics'/route['save_name']
        if not (folder/'events.jsonl').exists():continue
        events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        events=[e for e in events if any(k in e['type'] for k in ('COLLISION','RED_LIGHT','STOP_INFRACTION'))]
        if not events:continue
        rows=[json.loads(line) for line in (folder/'frames.jsonl').read_text().splitlines()]
        for event in events:
            frame=event['event_frame'];prior=[]
            for row in rows:
                if not frame-100<=row.get('simulation_frame',-1)<=frame:continue
                policy=row['policy'];compare=policy.get('speed_comparison') or []
                if not compare or not compare[0]:continue
                c=compare[0]
                prior.append(dict(frame=row['simulation_frame'],step=row['step'],actual=scalar(policy.get('measured_speed_mps')),
                    fm=scalar(c['planner_speed_mps']),tfpp=scalar(c['tfpp_speed_mps']),stop_probability=scalar(c['tfpp_stop_probability']),
                    control=row['returned_control'],overrides=policy.get('overrides')))
            result.append({'route_id':route['route_id'],'scenario':route['scenario_name'],'event':event,
                'pre_event_window_seconds':5.,'samples':len(prior),
                'tfpp_go_fm_stop_frames':sum(p['tfpp']>=2 and p['fm']<.5 for p in prior),
                'max_actual_mps':max((p['actual'] for p in prior if p['actual'] is not None),default=None),
                'sampled_prior':prior[::10],'event_frame_prediction':prior[-1] if prior else None})
    out={'progress':cp['progress'],'scope':'same visited-state comparison before incidents; historical references do not isolate causal effects','incidents':result}
    (ANALYSIS/'incident_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'progress':cp['progress'],'incidents':[{k:e[k] for k in ('route_id','scenario','samples','tfpp_go_fm_stop_frames','max_actual_mps')} for e in result]}))


if __name__=='__main__':main()
