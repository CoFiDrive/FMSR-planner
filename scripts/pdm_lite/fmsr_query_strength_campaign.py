"""Four prospective FMSR Query perturbations on route2709."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from visual_case_campaign import ROOT, prepare, run, write
from fmsrpdm.evaluation_watchdog import inspect_checkpoint

LEVELS={'query_0p03':.03,'query_0p1':.1,'query_0p5':.5,'query_1p0':1.}


def export(out,plan):
    import numpy as np
    rows=[]
    for job in plan['jobs']:
        root=Path(job['target'])/'FMSR'
        state,payload=inspect_checkpoint(root/'checkpoint_endpoint.json',['RouteScenario_2709_rep0'])
        if not state['complete']:raise ValueError('Incomplete evaluation: '+str(root))
        record=json.loads(payload)['_checkpoint']['records'][0]
        metric=root/'eval_metrics'/record['save_name'];diag=root/'diagnostics'/record['save_name']
        for path in [diag/'frames.jsonl',diag/'visual/frames.jsonl',metric/'perturbations.jsonl']:
            if not path.is_file() or not path.stat().st_size:raise ValueError('Missing recording: '+str(path))
        arrays=list((metric/'injection_arrays').glob('*.npz'))
        if not arrays:raise ValueError('No actual Query arrays: '+str(metric))
        ratios=[]
        for path in arrays:
            meta=json.loads(path.with_suffix('.json').read_text())
            if not meta['events'] or any(e.get('component')!='query' or e['amplitude']!=job['amplitude'] for e in meta['events']):
                raise ValueError('Wrong injection: '+str(path))
            with np.load(path) as data:
                if not np.allclose(data['after']-data['before'],data['delta']):raise ValueError('Array mismatch')
                before=np.sqrt(np.mean(data['before'].astype(float)**2))
                ratio=np.sqrt(np.mean(data['delta'].astype(float)**2))/max(before,1e-12)
                if before>1e-8 and not np.isclose(ratio,job['amplitude'],rtol=1e-4,atol=1e-6):
                    raise ValueError('Incorrect measured noise ratio')
                ratios.append(float(ratio))
        rows.append(dict(route_id='2709',model='FMSR',amplitude=job['amplitude'],
            DS=record['scores']['score_composed'],RC=record['scores']['score_route'],
            status=record['status'],injection_samples=len(arrays),measured_RMS_ratio_mean=float(np.mean(ratios)),
            infractions=json.dumps(record['infractions']),source=str(root/'checkpoint_endpoint.json')))
    with (out/'all_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(out/'recording_audit.json',dict(complete=True,route_runs=4,recordings=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    ax.plot([r['amplitude'] for r in rows],[r['DS'] for r in rows],'-o',label='FMSR',color='#F04438')
    ax.set(xlabel='Query noise RMS ratio',ylabel='Driving Score',title='Route2709 | FMSR | seed0',ylim=(-3,103))
    ax.grid(alpha=.2);ax.legend()
    for ext in ['png','pdf']:fig.savefig(out/('query_strength_DS.'+ext),dpi=300)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--foreground',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not args.foreground:
        if out.exists():raise ValueError('Use a new directory')
        out.mkdir(parents=True);jobs=[]
        for name,amplitude in LEVELS.items():
            target=out/name/'perturbed';prepare(target,route_ids=['2709'],model_names=['FMSR'])
            spec=dict(id=name,family='feature',component='query',amplitude=amplitude,mode='continuous',start_s=2.)
            write(target/'perturbation_job.json',dict(seed=0,route_conditions={'2709':spec}))
            f=target/'FMSR/source.json';source=json.loads(f.read_text())
            source['analysis_recording'].update(probe_every=100,probe_max=40)
            source.update(save_actual_injection_arrays=True,route_perturbations={'2709':spec},
                          injection_interface='planner.sample Query input; shared across classification and denoising')
            write(f,source);jobs.append(dict(name=name,amplitude=amplitude,target=str(target)))
        write(out/'manifest.json',dict(route_ids=['2709'],models=['FMSR'],seed=0,total_route_runs=4,jobs=jobs,
            protocol='Continuous Query noise from2s; RMS ratios0.03/0.1/0.5/1.0. Same native author_v8 PDM model, '
            'route and controller as main baseline. All four outcomes retained; no outcome-selected escalation.',
            baseline_source=str(ROOT/'outputs/local_evaluation/bench2drive220_author_v8_pdm_20261010_165733_epoch10/checkpoint_endpoint.json'),
            baseline_note='Historical clean DS100, not a new clean run. Earlier FM/FM+CR scans are independent runs.',
            recording='Cameras every5 ticks; per-tick trajectories/control; input probes every100 steps max40; '
            'actual injection tensors first3/every100 ticks max40. Input probes precede injection; NPZ records injected inputs.'))
        write(out/'status.json',dict(state='prepared',total_route_runs=4))
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
            write(out/'status.json',dict(state='evaluating',condition=job['name'],total_route_runs=4))
            target=Path(job['target']);os.environ['ROBUSTNESS_SPEC']=str(target/'perturbation_job.json');run(target)
        export(out,plan);write(out/'status.json',dict(state='complete',total_route_runs=4))
    except Exception as error:
        write(out/'status.json',dict(state='error',error=repr(error)));raise


if __name__=='__main__':main()
