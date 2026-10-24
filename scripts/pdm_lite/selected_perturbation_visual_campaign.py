"""Prospective visual reassessment of four historically favorable route/condition pairs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from visual_case_campaign import prepare, run, write, ROOT
from fmsrpdm.evaluation_watchdog import inspect_checkpoint

SPECS={
    '2084':dict(id='feature_bev_0.03',family='feature',component='bev',amplitude=.03,mode='continuous',start_s=2.),
    '2204':dict(id='feature_bev_0.03',family='feature',component='bev',amplitude=.03,mode='continuous',start_s=2.),
    '23658':dict(id='feature_bev_0.03',family='feature',component='bev',amplitude=.03,mode='continuous',start_s=2.),
    '2164':dict(id='pulse_state_lateral_0.1',family='state',axis='lateral',amplitude=.1,tau=.4,mode='pulse',start_s=5.),
    '2127':dict(id='position_longitudinal_0.5',family='position',axis='longitudinal',amplitude=.5,mode='continuous',start_s=2.),
    '2373':dict(id='feature_bev_0.03',family='feature',component='bev',amplitude=.03,mode='continuous',start_s=2.),
    '2709':dict(id='feature_query_0.03',family='feature',component='query',amplitude=.03,mode='continuous',start_s=2.),
}


def audit_recordings(out,specs):
    import numpy as np
    rows=[]
    for condition in ['clean','perturbed']:
        for model in ['FM','FM_CR']:
            root=out/condition/model
            expected=['RouteScenario_'+rid+'_rep0' for rid in specs]
            state,payload=inspect_checkpoint(root/'checkpoint_endpoint.json',expected)
            if not state['complete']:raise ValueError('Incomplete evaluation: '+str(root))
            for record in json.loads(payload)['_checkpoint']['records']:
                rid=record['route_id'].split('_')[1]
                diagnostic=root/'diagnostics'/record['save_name']
                metric=root/'eval_metrics'/record['save_name']
                for path in [diagnostic/'frames.jsonl',diagnostic/'visual/frames.jsonl']:
                    if not path.is_file() or not path.stat().st_size:raise ValueError('Missing recording: '+str(path))
                audit=metric/'perturbations.jsonl'
                if not audit.is_file() or (condition=='perturbed' and not audit.stat().st_size):
                    raise ValueError('Missing perturbation audit: '+str(audit))
                if condition=='clean' and audit.stat().st_size:
                    raise ValueError('Unexpected perturbation in clean run: '+str(audit))
                probes=list((metric/'probes').glob('*.pt'))
                if not probes:raise ValueError('Missing planner probes: '+str(metric))
                arrays=list((metric/'injection_arrays').glob('*.npz'))
                if condition=='perturbed' and not arrays:raise ValueError('Missing injection arrays: '+str(metric))
                for path in arrays:
                    with np.load(path) as values:
                        before,after,delta=[values[k] for k in ['before','after','delta']]
                        if not (before.shape==after.shape==delta.shape and np.allclose(after-before,delta)):
                            raise ValueError('Injection array mismatch: '+str(path))
                    metadata=json.loads(path.with_suffix('.json').read_text())
                    if any(event.get('component')!=specs[rid]['component'] for event in metadata['events']):
                        raise ValueError('Incorrect injection component: '+str(path))
                rows.append(dict(condition=condition,model=model,route_id=rid,
                    status=record['status'],scores=record['scores'],probes=len(probes),
                    injection_array_samples=len(arrays),diagnostics=str(diagnostic),metrics=str(metric)))
    write(out/'recording_audit.json',dict(complete=True,route_runs=len(rows),recordings=rows,
        sampling='Camera every 5 ticks; probes every 100 steps (max40); injection arrays first3 and every100 ticks (max40).'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--foreground',action='store_true')
    p.add_argument('--visual-every',type=int,default=5)
    p.add_argument('--route-ids',nargs='+',choices=list(SPECS),default=['2164','2127','2373','2709'])
    args=p.parse_args();out=args.output.resolve()
    if args.visual_every < 1:raise ValueError('visual-every must be positive')
    specs={rid:SPECS[rid] for rid in args.route_ids}
    total_route_runs=len(specs)*4
    if not args.foreground:
        if (out/'manifest.json').exists():raise ValueError('Use a new output directory')
        out.mkdir(parents=True,exist_ok=True)
        for condition in ['clean','perturbed']:
            prepare(out/condition,route_ids=list(specs),model_names=['FM','FM_CR'])
            mapping=specs if condition=='perturbed' else {rid:dict(id='clean',family='none',amplitude=0.,mode='continuous') for rid in specs}
            write(out/condition/'perturbation_job.json',dict(seed=0,route_conditions=mapping))
            for model in ['FM','FM_CR']:
                path=out/condition/model/'source.json';source=json.loads(path.read_text())
                source['analysis_recording'].update(probe_every=100,probe_max=40)
                source.update(visual_sensor_every=args.visual_every,save_actual_injection_arrays=True,route_perturbations=mapping)
                write(path,source)
        write(out/'manifest.json',dict(route_ids=list(specs),route_conditions=specs,
            models=['FM','FM_CR'],conditions=['clean','perturbed'],seed=0,total_route_runs=total_route_runs,
            purpose='Prospective clean/perturbed paired recordings. All new outcomes retained.',
            rendering='Six rows: real scene, four map paths, FM clean, FM perturbed, CR clean, CR perturbed.',
            annotation='Circles identify measured collisions or a matched comparison area, not invisible feature noise in RGB.',
            visual_every=args.visual_every,
            saved_inputs='Actual before/after injection arrays and deltas, plus pre-head input probes every 100 steps (max40).'))
        with (out/'monitor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__,'--output',str(out),'--foreground',
                                    '--route-ids',*specs,'--visual-every',str(args.visual_every)],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'monitor.pid').write_text(str(child.pid)+'\n')
        print('Monitor PID:',child.pid,'Log:',out/'monitor.log');return
    os.environ.update(VISUAL_SAVE_INJECTION_ARRAYS='1',VISUAL_PROBE_EVERY='100',VISUAL_PROBE_MAX='40')
    os.environ['VISUAL_EVERY']=str(args.visual_every)
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'fmsr_pdm_lite'),str(ROOT),os.environ.get('PYTHONPATH','')])
    try:
        for condition in ['clean','perturbed']:
            write(out/'status.json',dict(state='evaluating',condition=condition,total_route_runs=total_route_runs))
            os.environ['ROBUSTNESS_SPEC']=str(out/condition/'perturbation_job.json')
            print('Starting condition:',condition,flush=True)
            run(out/condition)
        audit_recordings(out,specs)
        write(out/'status.json',dict(state='rendering',total_route_runs=total_route_runs))
        subprocess.run([sys.executable,str(ROOT/'scripts/pdm_lite/render_selected_perturbation_real_cases.py'),
                        '--campaign',str(out)],check=True,cwd=ROOT)
        write(out/'status.json',dict(state='complete',total_route_runs=total_route_runs))
    except Exception as error:
        write(out/'status.json',dict(state='error',error=repr(error)));raise


if __name__=='__main__':main()
