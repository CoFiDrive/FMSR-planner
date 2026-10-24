"""Predeclared 1.5x/2x visual perturbation scan; all outcomes retained."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from visual_case_campaign import prepare, run, write, ROOT
from selected_perturbation_visual_campaign import SPECS

REFERENCE=Path('/media/vcdc/T71/fmsr_selected_perturbation_real_cases_20261021')
LEVELS={'strength_1p5':1.5,'strength_2p0':2.0}


def export_results(out):
    plan=json.loads((out/'manifest.json').read_text())
    route_ids=list(next(iter(plan['route_conditions_by_strength'].values())))
    levels=plan['planned_strengths']
    records=[]
    locations=[(0.,REFERENCE/'clean'),(1.,REFERENCE/'perturbed')]
    for previous in plan.get('reference_scans',[]):
        previous=Path(previous)
        old=json.loads((previous/'manifest.json').read_text())
        locations += [(strength,previous/level/'perturbed') for level,strength in old['planned_strengths'].items()]
    locations += [(strength,out/level/'perturbed') for level,strength in levels.items()]
    for strength,base in locations:
        for model in ['FM','FM_CR']:
            payload=json.loads((base/model/'checkpoint_endpoint.json').read_text())
            for r in payload['_checkpoint']['records']:
                rid=r['route_id'].split('_')[1]
                if rid not in route_ids:continue
                records.append(dict(route_id=rid,model=model,strength_multiplier=strength,
                    amplitude=SPECS[rid]['amplitude']*strength,status=r['status'],
                    DS=r['scores']['score_composed'],RC=r['scores']['score_route'],
                    result_source=str(base/model/'checkpoint_endpoint.json')))
    lookup={(r['route_id'],r['model'],r['strength_multiplier']):r for r in records}
    for r in records:
        baseline=lookup[r['route_id'],r['model'],0.]['DS']
        r['clean_DS']=baseline;r['DS_change']=r['DS']-baseline
    comparisons=[]
    strengths=sorted({r['strength_multiplier'] for r in records})
    for rid in route_ids:
        for strength in [s for s in strengths if s>0]:
            f=lookup[rid,'FM',strength];c=lookup[rid,'FM_CR',strength]
            comparisons.append(dict(route_id=rid,strength_multiplier=strength,amplitude=f['amplitude'],
                FM_clean=f['clean_DS'],FM_perturbed=f['DS'],FM_change=f['DS_change'],
                CR_clean=c['clean_DS'],CR_perturbed=c['DS'],CR_change=c['DS_change'],
                matches_requested_pattern=int(f['DS_change']<0 and c['DS_change']>=0),
                relative_CR_advantage=c['DS_change']-f['DS_change']))
    for name,rows in [('all_results_including_reference.csv',records),('all_paired_changes.csv',comparisons)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager
    fontManager.addfont('/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')
    plt.rcParams.update({'font.family':'Arial','font.size':12,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,rid in zip(axes.flat,route_ids):
        for model,color in [('FM','#2454E8'),('FM_CR','#239A45')]:
            x=strengths;y=[lookup[rid,model,s]['DS'] for s in x]
            ax.plot(x,y,'-o',color=color,lw=2,label='Flow Matching' if model=='FM' else '+Spectrally Regularized')
        ax.set(title='Route '+rid,xlabel='Perturbation strength / original amplitude',ylabel='Driving Score',
               ylim=(-3,103),xticks=strengths);ax.grid(alpha=.2);ax.legend(fontsize=9)
    for ax in list(axes.flat)[len(route_ids):]:ax.set_visible(False)
    for ext in ['png','pdf']:fig.savefig(out/('all_strengths_DS.'+ext),dpi=600)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--foreground',action='store_true')
    p.add_argument('--multipliers',nargs='+',type=float,default=[1.5,2.])
    p.add_argument('--route-ids',nargs='+',choices=list(SPECS),default=list(SPECS))
    p.add_argument('--reference-scans',nargs='*',type=Path,default=[])
    args=p.parse_args();out=args.output.resolve()
    if not args.foreground:
        if (out/'manifest.json').exists():raise ValueError('Choose a new output directory')
        out.mkdir(parents=True,exist_ok=True)
        if json.loads((REFERENCE/'status.json').read_text())['state']!='complete':
            raise ValueError('Reference evaluation must complete before starting the scan')
        plans={}
        levels={'strength_'+str(float(value)).replace('.','p'):value for value in args.multipliers}
        new_count=len(args.route_ids)*2*len(levels)
        for level,multiplier in levels.items():
            specs={rid:copy.deepcopy(SPECS[rid]) for rid in args.route_ids}
            for spec in specs.values():
                spec['amplitude']=round(spec['amplitude']*multiplier,6)
                spec['id']=spec['id']+'_strength'+str(multiplier)
            target=out/level
            prepare(target/'perturbed',route_ids=list(specs),model_names=['FM','FM_CR'])
            write(target/'perturbed/perturbation_job.json',dict(seed=0,route_conditions=specs))
            for model in ['FM','FM_CR']:
                path=target/'perturbed'/model/'source.json';s=json.loads(path.read_text())
                s['analysis_recording'].update(probe_every=100,probe_max=40)
                s.update(route_perturbations=specs,save_actual_injection_arrays=True)
                write(path,s)
            write(target/'manifest.json',dict(route_ids=list(specs),route_conditions=specs,
                models=['FM','FM_CR'],conditions=['clean','perturbed'],seed=0,
                clean_reference=str(REFERENCE),new_route_runs=len(specs)*2,comparison_route_records=len(specs)*4,
                annotation='Red circles: measured collisions; yellow: spatial comparison region. '
                           'These are not feature-noise masks in RGB.'))
            plans[level]=specs
        write(out/'manifest.json',dict(reference=str(REFERENCE),planned_strengths=levels,
            route_conditions_by_strength=plans,new_route_runs=new_count,seed=0,
            reference_scans=[str(path.resolve()) for path in args.reference_scans],
            stopping_rule='Complete all predeclared strengths regardless of outcomes; no automatic further escalation.',
            reporting='All outcomes and all original-strength results retained; reused clean baselines disclosed.'))
        with (out/'monitor.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__,'--output',str(out),'--foreground'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (out/'monitor.pid').write_text(str(child.pid)+'\n')
        print('Monitor PID:',child.pid,'Log:',out/'monitor.log');return
    os.environ.update(VISUAL_SAVE_INJECTION_ARRAYS='1',VISUAL_PROBE_EVERY='100',VISUAL_PROBE_MAX='40')
    plan=json.loads((out/'manifest.json').read_text())
    try:
        for level in plan['planned_strengths']:
            target=out/level
            write(out/'status.json',dict(state='evaluating',level=level,new_route_runs=plan['new_route_runs']))
            os.environ['ROBUSTNESS_SPEC']=str(target/'perturbed/perturbation_job.json')
            print('Starting predeclared level:',level,flush=True)
            run(target/'perturbed')
            write(out/'status.json',dict(state='rendering',level=level))
            subprocess.run([sys.executable,str(ROOT/'scripts/pdm_lite/render_selected_perturbation_real_cases.py'),
                            '--campaign',str(target)],check=True,cwd=ROOT)
        export_results(out)
        write(out/'status.json',dict(state='complete',new_route_runs=plan['new_route_runs']))
    except Exception as error:
        write(out/'status.json',dict(state='error',error=repr(error)));raise


if __name__=='__main__':main()
