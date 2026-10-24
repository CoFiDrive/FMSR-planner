"""Sixteen predeclared visual runs, followed by automatic figure export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from visual_case_campaign import prepare, run, write, ROOT

CONDITIONS = {
    'clean': dict(id='clean', family='none', amplitude=0., mode='continuous'),
    'state_lateral_0.1': dict(id='state_lateral_0.1', family='state', axis='lateral',
                              amplitude=.1, tau=.4, mode='continuous', start_s=2.),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--foreground', action='store_true')
    args = p.parse_args()
    out = args.output.resolve()
    if not args.foreground:
        if (out/'manifest.json').exists():
            raise ValueError('Use a new directory; existing recordings are preserved')
        out.mkdir(parents=True, exist_ok=True)
        for name, spec in CONDITIONS.items():
            prepare(out/name, model_names=['FM', 'FM_CR'])
            write(out/name/'perturbation_job.json', dict(seed=0, perturbation=spec))
        write(out/'manifest.json', dict(route_ids=['28093','17563','2204','24785'],
            models=['FM','FM_CR'], conditions=CONDITIONS, total_route_runs=16, seed=0,
            protocol='FP32, TF32 disabled, seed0. From agent time 2s, inject signed 0.10m '
                     'lateral displacement into first 10 planning-state points at generation tau=0.4 '
                     'every planning tick. Signs are route/tick deterministic and shared across models. '
                     'No localization or RGB perturbation. Additional cameras never reach policy.',
            selection='All 16 runs retained, no driving-outcome-selected retries.',
            output_figures='figures/four_scenarios_clean_vs_perturbed.png'))
        with (out/'monitor.log').open('a') as log:
            child = subprocess.Popen([sys.executable,'-u',__file__,'--output',str(out),'--foreground'],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        (out/'monitor.pid').write_text(str(child.pid)+'\n')
        print('Monitor PID:',child.pid,'Log:',out/'monitor.log')
        return
    try:
        for name in CONDITIONS:
            write(out/'status.json',dict(state='evaluating',condition=name,total_route_runs=16))
            os.environ['ROBUSTNESS_SPEC']=str(out/name/'perturbation_job.json')
            print('Starting condition:',name,flush=True)
            run(out/name)
        write(out/'status.json',dict(state='rendering',total_route_runs=16))
        subprocess.run([sys.executable,str(ROOT/'scripts/pdm_lite/render_perturbation_visual_cases.py'),
                        '--campaign',str(out)],check=True,cwd=ROOT)
        write(out/'status.json',dict(state='complete',total_route_runs=16))
    except Exception as error:
        write(out/'status.json',dict(state='error',error=repr(error)))
        raise


if __name__=='__main__': main()
