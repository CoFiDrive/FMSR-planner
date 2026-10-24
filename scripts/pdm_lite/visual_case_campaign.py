"""Run the requested four routes for three pinned models with visual sensors."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

ROOT = Path('/home/vcdc/fmsr')
sys.path[:0] = [str(ROOT / 'fmsr_pdm_lite'), str(ROOT)]
from fmsrpdm.evaluation_watchdog import watch, evaluation_is_locked

MODELS = {
    'FM': ROOT / 'outputs/local_training/flow_only_10epochs_seed0_20261008_143915/model_0009.pth',
    'FM_CR': ROOT / 'outputs/local_training/tie_spectral_10epochs_seed0_20261006_223036/model_0009.pth',
    'FMSR': ROOT / 'fmsr_pdm_lite/runs/author_v8_pdm_20261010_165733/model_0009.pth',
}
ROUTE_IDS = ['28093', '17563', '2204', '24785']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(output, route_ids=None, model_names=None):
    route_ids = route_ids or ROUTE_IDS
    model_names = model_names or list(MODELS)
    output.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ROOT / '3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml')
    selected = ET.Element('routes')
    for rid in route_ids:
        selected.append(copy.deepcopy(next(r for r in tree.getroot().findall('route') if r.get('id') == rid)))
    routes = output / 'routes.xml'
    content = ET.tostring(selected, encoding='utf-8', xml_declaration=True)
    if routes.exists() and routes.read_bytes() != content:
        raise ValueError('Route subset changed')
    routes.write_bytes(content)
    manifest = dict(route_ids=route_ids, seed=0, total_route_runs=len(route_ids)*len(model_names),
                    purpose='Visualization reassessment, independent from historical scores',
                    cameras=['front', 'third_person', 'top_down'], sensor_every_ticks=5,
                    map_source='CARLA OpenDRIVE and 2m lane geometry', models={})
    for model, checkpoint in MODELS.items():
        if model not in model_names:
            continue
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        target = output / model
        agent = target / 'agent'
        agent.mkdir(parents=True, exist_ok=True)
        link = agent / checkpoint.name
        if not link.exists():
            try:
                link.symlink_to(checkpoint)
            except OSError:
                shutil.copy2(checkpoint, link)  # external exFAT storage has no symlinks
        config = checkpoint.parent / 'config.json'
        (agent / 'config.json').write_bytes(config.read_bytes())
        settings = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                        config_sha256=sha(config), routes=str(routes), route_count=len(route_ids),
                        carla_client_timeout_seconds=120,
                        analysis_recording=dict(trace_every=1, probe_every=0, probe_max=0),
                        visual_model=model)
        write(target / 'source.json', settings)
        manifest['models'][model] = settings
    write(output / 'manifest.json', manifest)


def run(output):
    manifest = json.loads((output / 'manifest.json').read_text())
    model_names = list(manifest['models'])
    total = manifest['total_route_runs']
    write(output / 'status.json', dict(state='starting', pid=os.getpid()))
    os.environ['VISUAL_MAP_ROOT'] = str(output / 'maps')
    os.environ.setdefault('VISUAL_EVERY', '5')
    try:
        for model in model_names:
            write(output / 'status.json', dict(state='evaluating', model=model, pid=os.getpid(),
                                              total_route_runs=total))
            os.environ['VISUAL_MODEL'] = model
            print('Starting visual model:', model, flush=True)
            watch(SimpleNamespace(output=output / model, gpu=0, carla_port=8940,
                                  tm_port=8140, poll_seconds=15, retry_seconds=30,
                                  stale_seconds=900, heartbeat_seconds=30, rpc_timeout=5,
                                  eval_timeout=120,
                                  launcher=ROOT / 'scripts/pdm_lite/run_visual_case_evaluation.sh'))
            while evaluation_is_locked(output / model):
                time.sleep(5)
        if set(model_names) == set(MODELS) and len(manifest['route_ids']) == 4:
            write(output / 'status.json', dict(state='rendering', pid=os.getpid()))
            subprocess.run([sys.executable, str(ROOT / 'scripts/pdm_lite/render_visual_cases.py'),
                            '--campaign', str(output)], check=True, cwd=ROOT)
        write(output / 'status.json', dict(state='complete', total_route_runs=total))
    except Exception as error:
        write(output / 'status.json', dict(state='error', error=repr(error)))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--foreground', action='store_true')
    p.add_argument('--route-ids', nargs='+', default=ROUTE_IDS)
    p.add_argument('--models', nargs='+', choices=list(MODELS), default=list(MODELS))
    args = p.parse_args()
    output = args.output.resolve()
    if not args.foreground:
        prepare(output, args.route_ids, args.models)
        with (output / 'monitor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-u', __file__, '--output', str(output),
                                        '--foreground'], cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (output / 'monitor.pid').write_text(str(process.pid) + '\n')
        print('Monitor PID:', process.pid, 'Log:', output / 'monitor.log')
    else:
        run(output)


if __name__ == '__main__':
    main()
