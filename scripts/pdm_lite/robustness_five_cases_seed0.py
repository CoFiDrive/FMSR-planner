"""Repeat five predeclared FM/FM+CR perturbation pairs, retaining every result."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from visual_case_campaign import ROOT, prepare, run, write
from fmsrpdm.evaluation_watchdog import inspect_checkpoint


SPECS = {
    '2127': dict(id='position_longitudinal_1.25', family='position',
                 axis='longitudinal', amplitude=1.25, mode='continuous', start_s=2.),
    '2373': dict(id='feature_bev_0.075', family='feature', component='bev',
                 amplitude=.075, mode='continuous', start_s=2.),
    '2709': dict(id='feature_query_0.075', family='feature', component='query',
                 amplitude=.075, mode='continuous', start_s=2.),
    '2115': dict(id='position_lateral_0.50', family='position', axis='lateral',
                 amplitude=.50, mode='continuous', start_s=2.),
    '23658': dict(id='feature_bev_0.03', family='feature', component='bev',
                  amplitude=.03, mode='continuous', start_s=2.),
}
MODELS = ['FM', 'FM_CR']


def export_results(out):
    rows = []
    for model in MODELS:
        checkpoint = out / 'perturbed' / model / 'checkpoint_endpoint.json'
        expected = [f'RouteScenario_{rid}_rep0' for rid in SPECS]
        state, payload = inspect_checkpoint(checkpoint, expected)
        if not state['complete']:
            raise RuntimeError(f'Incomplete evaluation for {model}: {state}')
        payload = json.loads(payload)
        for record in payload['_checkpoint']['records']:
            rid = record['route_id'].split('_')[1]
            spec = SPECS[rid]
            rows.append(dict(route_id=rid, model=model, seed=0,
                perturbation=spec['id'], amplitude=spec['amplitude'],
                status=record['status'], DS=record['scores']['score_composed'],
                RC=record['scores']['score_route'],
                infractions=json.dumps(record.get('infractions', {})),
                result_source=str(checkpoint)))
    lookup = {(r['route_id'], r['model']): r for r in rows}
    paired = []
    for rid, spec in SPECS.items():
        fm, cr = lookup[rid, 'FM'], lookup[rid, 'FM_CR']
        paired.append(dict(route_id=rid, perturbation=spec['id'], seed=0,
            FM_DS=fm['DS'], FM_CR_DS=cr['DS'], DS_delta_CR_minus_FM=cr['DS']-fm['DS'],
            FM_RC=fm['RC'], FM_CR_RC=cr['RC'],
            FM_status=fm['status'], FM_CR_status=cr['status']))
    for name, data in [('all_results.csv', rows), ('paired_comparison.csv', paired)]:
        with (out / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--route-ids', nargs='+', choices=list(SPECS))
    args = parser.parse_args()
    if args.route_ids:
        selected = {rid: SPECS[rid] for rid in args.route_ids}
        SPECS.clear()
        SPECS.update(selected)
    route_runs = len(SPECS) * len(MODELS)
    out = args.output.resolve()
    if not args.foreground:
        if (out / 'manifest.json').exists():
            raise ValueError('Use a new output directory to preserve previous results')
        prepare(out / 'perturbed', route_ids=list(SPECS), model_names=MODELS)
        write(out / 'perturbed' / 'perturbation_job.json',
              dict(seed=0, route_conditions=SPECS))
        for model in MODELS:
            path = out / 'perturbed' / model / 'source.json'
            source = json.loads(path.read_text())
            source['analysis_recording'].update(probe_every=100, probe_max=40)
            source.update(route_perturbations=SPECS, save_actual_injection_arrays=True)
            write(path, source)
        write(out / 'manifest.json', dict(route_ids=list(SPECS), route_conditions=SPECS,
            models=MODELS, seed=0, new_route_runs=route_runs,
            purpose='Repeat the declared FM/FM+CR perturbation pairs.',
            previous_three_case_campaign='/media/vcdc/T71/fmsr_perturbation_visual_2p5_20261021',
            stopping_rule='Run every declared evaluation; retain all outcomes without strength escalation.',
            comparison='Paired FM versus FM+CR under the same perturbation; no new clean runs.'))
        write(out / 'status.json', dict(state='prepared', new_route_runs=route_runs))
        with (out / 'monitor.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--output',
                str(out), '--foreground', '--route-ids', *SPECS], cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'monitor.pid').write_text(str(child.pid) + '\n')
        print(f'Monitor PID: {child.pid}; output: {out}')
        return
    os.environ.update(ROBUSTNESS_SPEC=str(out / 'perturbed' / 'perturbation_job.json'),
        VISUAL_SAVE_INJECTION_ARRAYS='1', VISUAL_PROBE_EVERY='100', VISUAL_PROBE_MAX='40')
    try:
        write(out / 'status.json', dict(state='evaluating', new_route_runs=route_runs, pid=os.getpid()))
        run(out / 'perturbed')
        export_results(out)
        write(out / 'status.json', dict(state='complete', new_route_runs=route_runs))
    except Exception as error:
        write(out / 'status.json', dict(state='error', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
