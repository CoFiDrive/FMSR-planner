"""Route2709: fresh clean pair and three predeclared Query-noise pairs."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from visual_case_campaign import ROOT, prepare, run, write
from fmsrpdm.evaluation_watchdog import inspect_checkpoint

ROUTE = '2709'
MODELS = ['FM', 'FM_CR']
LEVELS = {'query_0p03': .03, 'query_0p1': .1, 'query_0p5': .5}


def setup(out, amplitudes=None, clean_reference=None):
    if out.exists():
        raise ValueError('Choose a new directory to preserve all earlier runs')
    out.mkdir(parents=True)
    jobs = []
    levels = LEVELS if amplitudes is None else {'query_'+str(float(a)).replace('.','p'):a for a in amplitudes}
    if not levels or any(a<=0 for a in levels.values()):raise ValueError('Noise amplitudes must be positive')
    if clean_reference:
        clean_reference=clean_reference.resolve()
        for model in MODELS:
            state,_=inspect_checkpoint(clean_reference/'clean'/model/'checkpoint_endpoint.json',['RouteScenario_2709_rep0'])
            if not state['complete']:raise ValueError('Clean reference incomplete')
        jobs.append(dict(name='clean',amplitude=0.,target=str(clean_reference/'clean'),reused=True))
    for name, amplitude in ([] if clean_reference else [('clean',0.)])+list(levels.items()):
        target = out / name if name == 'clean' else out / name / 'perturbed'
        spec = dict(id=name, family='none' if amplitude == 0 else 'feature',
                    component='query', amplitude=amplitude, mode='continuous', start_s=2.)
        prepare(target, route_ids=[ROUTE], model_names=MODELS)
        write(target / 'perturbation_job.json', dict(seed=0, route_conditions={ROUTE: spec}))
        for model in MODELS:
            path = target / model / 'source.json'
            source = json.loads(path.read_text())
            source['analysis_recording'].update(probe_every=100, probe_max=40)
            source.update(save_actual_injection_arrays=True, route_perturbations={ROUTE: spec})
            write(path, source)
        if name != 'clean':
            write(out / name / 'manifest.json', dict(route_ids=[ROUTE],
                  route_conditions={ROUTE: spec}, models=MODELS, conditions=['clean', 'perturbed'],
                  clean_reference=str(clean_reference or out), seed=0))
        jobs.append(dict(name=name, amplitude=amplitude, target=str(target)))
    write(out / 'manifest.json', dict(route_ids=[ROUTE], models=MODELS, seed=0,
          query_amplitudes=list(levels.values()), total_route_runs=2*sum(not j.get('reused',False) for j in jobs), jobs=jobs,
          clean_reference=str(clean_reference) if clean_reference else None,
          protocol='FM/FM+CR paired Query-noise strengths; clean provenance is recorded per job; '
          'continuous injection from2s, RMS-normalized noise, fixed models and route; retain all outcomes.',
          recording='Camera every5 ticks; per-tick telemetry; probes first/every100 steps max40; '
          'actual before/after/delta tensors first3 injections and every100 ticks max40.'))
    write(out / 'status.json', dict(state='prepared', total_route_runs=2*sum(not j.get('reused',False) for j in jobs)))


def export(out, plan):
    import numpy as np
    rows = []
    for job in plan['jobs']:
        for model in MODELS:
            root = Path(job['target']) / model
            state, payload = inspect_checkpoint(root / 'checkpoint_endpoint.json', ['RouteScenario_2709_rep0'])
            if not state['complete']:
                raise ValueError('Incomplete result: ' + str(root))
            record = json.loads(payload)['_checkpoint']['records'][0]
            metric = root / 'eval_metrics' / record['save_name']
            diagnostic = root / 'diagnostics' / record['save_name']
            for path in [diagnostic/'frames.jsonl', diagnostic/'visual/frames.jsonl']:
                if not path.is_file() or not path.stat().st_size:
                    raise ValueError('Missing telemetry: ' + str(path))
            probes = list((metric/'probes').glob('*.pt'))
            arrays = list((metric/'injection_arrays').glob('*.npz'))
            audit = metric/'perturbations.jsonl'
            if not probes or not audit.is_file():
                raise ValueError('Missing planner/audit recording: ' + str(metric))
            if job['amplitude'] and (not arrays or not audit.stat().st_size):
                raise ValueError('Missing applied-noise arrays: ' + str(metric))
            if not job['amplitude'] and audit.stat().st_size:
                raise ValueError('Unexpected clean perturbation')
            for path in arrays:
                with np.load(path) as data:
                    if not np.allclose(data['after']-data['before'], data['delta']):
                        raise ValueError('Invalid before/after/delta: ' + str(path))
                metadata=json.loads(path.with_suffix('.json').read_text())
                if not metadata['events'] or any(e.get('component')!='query' or
                    e['amplitude']!=job['amplitude'] for e in metadata['events']):
                    raise ValueError('Incorrect Query perturbation: ' + str(path))
            rows.append(dict(condition=job['name'], amplitude=job['amplitude'], model=model,
                DS=record['scores']['score_composed'], RC=record['scores']['score_route'],
                status=record['status'], probes=len(probes), injection_samples=len(arrays),
                infractions=json.dumps(record['infractions']), source=str(root/'checkpoint_endpoint.json')))
    lookup={(r['amplitude'],r['model']):r for r in rows}
    paired=[]
    for amplitude in plan['query_amplitudes']:
        fm,cr=[lookup[amplitude,m] for m in MODELS]
        fm_change=fm['DS']-lookup[0.,'FM']['DS'];cr_change=cr['DS']-lookup[0.,'FM_CR']['DS']
        paired.append(dict(amplitude=amplitude, FM_DS=fm['DS'], FM_SR_DS=cr['DS'],
            DS_difference=cr['DS']-fm['DS'], FM_change_from_clean=fm_change,
            FM_SR_change_from_clean=cr_change, relative_SR_advantage=cr_change-fm_change,
            FM_status=fm['status'], FM_SR_status=cr['status']))
    for name,data in [('all_results.csv',rows),('paired_comparison.csv',paired)]:
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(out/'recording_audit.json',dict(complete=True,route_runs=len(rows),recordings=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    for model,label,color in [('FM','FM','#2454E8'),('FM_CR','FM+SR','#239A45')]:
        levels=[0.,*plan['query_amplitudes']]
        ax.plot(levels,[lookup[a,model]['DS'] for a in levels],'-o',label=label,color=color)
    ax.set(xlabel='Query noise RMS ratio',ylabel='Driving Score',title='Route2709 | seed0',ylim=(-3,103))
    ax.legend();ax.grid(alpha=.2)
    for ext in ['png','pdf']:fig.savefig(out/('query_strength_DS.'+ext),dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--foreground',action='store_true')
    parser.add_argument('--amplitudes',nargs='+',type=float)
    parser.add_argument('--clean-reference',type=Path)
    args=parser.parse_args();out=args.output.resolve()
    if not args.foreground:
        setup(out,args.amplitudes,args.clean_reference)
        with (out/'monitor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__,'--output',str(out),'--foreground'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'monitor.pid').write_text(str(child.pid)+'\n')
        print('Monitor PID:',child.pid,'Output:',out);return
    os.environ.update(VISUAL_SAVE_INJECTION_ARRAYS='1',VISUAL_PROBE_EVERY='100',VISUAL_PROBE_MAX='40')
    os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'fmsr_pdm_lite'),str(ROOT),os.environ.get('PYTHONPATH','')])
    plan=json.loads((out/'manifest.json').read_text())
    try:
        for job in plan['jobs']:
            if job.get('reused'):continue
            write(out/'status.json',dict(state='evaluating',condition=job['name'],total_route_runs=plan['total_route_runs']))
            target=Path(job['target']);os.environ['ROBUSTNESS_SPEC']=str(target/'perturbation_job.json')
            run(target)
        export(out,plan)
        write(out/'status.json',dict(state='rendering',total_route_runs=plan['total_route_runs']))
        for name in [j['name'] for j in plan['jobs'] if j['amplitude']>0]:
            subprocess.run([sys.executable,str(ROOT/'scripts/pdm_lite/render_selected_perturbation_real_cases.py'),
                            '--campaign',str(out/name)],check=True,cwd=ROOT)
        write(out/'status.json',dict(state='complete',total_route_runs=plan['total_route_runs']))
    except Exception as error:
        write(out/'status.json',dict(state='error',error=repr(error)));raise


if __name__=='__main__':main()
