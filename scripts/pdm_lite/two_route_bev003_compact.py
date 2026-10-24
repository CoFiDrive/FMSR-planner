"""Run paired clean/BEV-0.03 evaluations for routes 2084 and 2204."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from visual_case_campaign import ROOT, prepare, write
from fmsrpdm.evaluation_watchdog import inspect_checkpoint, watch


ROUTES = ['2084', '2204']
MODELS = ['FM', 'FM_CR']
CONDITIONS = {
    'clean': dict(id='clean', family='none', amplitude=0., mode='continuous'),
    'bev003': dict(id='feature_bev_0.03', family='feature', component='bev',
                   amplitude=.03, mode='continuous', start_s=2.),
}


def export(out):
    rows = []
    expected = [f'RouteScenario_{rid}_rep0' for rid in ROUTES]
    for condition in CONDITIONS:
        for model in MODELS:
            checkpoint = out / condition / model / 'checkpoint_endpoint.json'
            state, payload = inspect_checkpoint(checkpoint, expected)
            if not state['complete']:
                raise RuntimeError(f'Incomplete {condition}/{model}: {state}')
            for record in json.loads(payload)['_checkpoint']['records']:
                rows.append(dict(route_id=record['route_id'].split('_')[1],
                    condition=condition, model=model, DS=record['scores']['score_composed'],
                    RC=record['scores']['score_route'], status=record['status'],
                    infractions=json.dumps(record.get('infractions', {}))))
    with (out / 'all_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    if not args.foreground:
        if out.exists():
            raise ValueError('Use a new output directory')
        out.mkdir(parents=True)
        for condition, spec in CONDITIONS.items():
            target = out / condition
            prepare(target, route_ids=ROUTES, model_names=MODELS)
            write(target / 'perturbation_job.json',
                  dict(seed=0, route_conditions={rid: spec for rid in ROUTES}))
            for model in MODELS:
                source_path = target / model / 'source.json'
                source = json.loads(source_path.read_text())
                source['analysis_recording'] = dict(trace_every=0, probe_every=0, probe_max=0)
                source.update(route_perturbations={rid: spec for rid in ROUTES})
                write(source_path, source)
        write(out / 'manifest.json', dict(routes=ROUTES, models=MODELS,
            conditions=CONDITIONS, seed=0, total_route_runs=8,
            protocol='PDM-Lite; same checkpoints; continuous BEV RMS noise from 2 s; no extra visual cameras',
            old_incomplete_run='route2084_2204_bev003_fm_cr_seed0_20261022_234100'))
        with (out / 'monitor.log').open('a') as log:
            child = subprocess.Popen([sys.executable, '-u', __file__, '--output', str(out),
                                      '--foreground'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (out / 'monitor.pid').write_text(str(child.pid) + '\n')
        print('Monitor PID:', child.pid, 'Output:', out)
        return
    os.environ['EVALUATION_AGENT'] = str(ROOT / 'scripts/pdm_lite/robustness_agent.py')
    os.environ['FMSR_DIAGNOSTICS'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    try:
        for condition in CONDITIONS:
            os.environ['ROBUSTNESS_SPEC'] = str(out / condition / 'perturbation_job.json')
            for model in MODELS:
                target = out / condition / model
                write(out / 'status.json', dict(state='evaluating', condition=condition,
                    model=model, total_route_runs=8))
                print('Starting', condition, model, flush=True)
                watch(SimpleNamespace(output=target, gpu=0, carla_port=8950, tm_port=8150,
                    poll_seconds=15, retry_seconds=30, stale_seconds=900,
                    heartbeat_seconds=30, rpc_timeout=5, eval_timeout=120,
                    launcher=ROOT / 'scripts/eval_bench2drive_pdm_lite_fmsr.sh'))
        export(out)
        write(out / 'status.json', dict(state='complete', total_route_runs=8))
    except Exception as error:
        write(out / 'status.json', dict(state='error', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
