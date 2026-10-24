"""Queue a pinned legacy FM+CR failure subset behind the active evaluation."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from fmsrpdm.evaluation_watchdog import (
    atomic_write, evaluation_is_locked, inspect_checkpoint, processes, updated_at, watch,
)
from fmsrpdm.train_then_evaluate import summarize_analysis

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT/'outputs/local_evaluation/bench2drive220_fm_cr_metrics_20261009_183130'
PREREQUISITE = ROOT/'outputs/local_evaluation/bench2drive220_author_v8_pdm_fm_cr_20261011_201445_epoch10_metrics'
OUTPUT = ROOT/'outputs/local_evaluation/legacy_fm_cr_failed82_reassessment_20261012'
LAUNCHER = ROOT/'scripts/eval_bench2drive_pdm_lite_fmsr.sh'


def write_json(path, value):
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def failed(record):
    return record['scores']['score_route'] != 100 or any(
        events for name, events in record['infractions'].items() if name != 'min_speed_infractions')


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline.resolve()
    data = json.loads((baseline/'checkpoint_endpoint.json').read_text())
    records = data['_checkpoint']['records']
    source = json.loads((baseline/'agent/source.json').read_text())
    count = source['route_count']
    route_ids = [f"RouteScenario_{r.get('id')}_rep0" for r in
                 ET.parse(source['routes']).getroot().findall('route')]
    state, _ = inspect_checkpoint(baseline/'checkpoint_endpoint.json', route_ids)
    if not state['complete'] or len(records) != count:
        raise ValueError('Baseline must be a complete evaluation without infrastructure errors')
    checkpoint = Path(source['checkpoint']).resolve(strict=True)
    config = json.loads((checkpoint.parent/'config.json').read_text())
    if config['fmsr_trajectory_head'] != 'tie_cr_fd' or not config['flow_contract_enable']:
        raise ValueError('Expected the legacy FM+CR checkpoint')
    selected = [row for row in records if failed(row)]
    ids = [row['route_id'] for row in selected]
    full_routes = Path(source['routes']).resolve(strict=True)
    full_root = ET.parse(full_routes).getroot()
    subset = ET.Element(full_root.tag, full_root.attrib)
    for element in full_root.findall('route'):
        if f"RouteScenario_{element.get('id')}_rep0" in ids:
            subset.append(copy.deepcopy(element))
    xml_ids = [f"RouteScenario_{r.get('id')}_rep0" for r in subset.findall('route')]
    if not ids or xml_ids != ids:
        raise ValueError('Failure records do not match the route XML order')
    routes = output/'failed_routes.xml'
    content = ET.tostring(subset, encoding='utf-8', xml_declaration=True)
    if routes.exists() and routes.read_bytes() != content:
        raise ValueError('Existing route subset differs; choose a new output directory')
    atomic_write(routes, content)
    job = dict(baseline=str(baseline), baseline_sha256=sha256(baseline/'checkpoint_endpoint.json'),
               checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
               config_sha256=sha256(checkpoint.parent/'config.json'),
               routes=str(routes), routes_sha256=sha256(routes), route_count=len(ids), route_ids=ids,
               prerequisite=str(args.prerequisite.resolve()), gpu=args.gpu,
               carla_port=args.carla_port, tm_port=args.tm_port,
               selection='RC<100 or any infraction except min_speed_infractions',
               analysis_recording=dict(trace_every=1, probe_every=200, probe_max=4,
                                       vehicle_state='every_agent_tick',
                                       comfort_export_interval_seconds=0.1,
                                       nearby_actors='every_agent_tick', sensor_every=0))
    manifest = output/'reassessment_job.json'
    if manifest.exists() and json.loads(manifest.read_text()) != job:
        raise ValueError('Existing output belongs to different reassessment settings')
    write_json(manifest, job)
    write_json(output/'baseline_selected_records.json', selected)
    return job


def environment(job, output):
    env = os.environ.copy()
    for name in ('SHADOW_TEACHER_CHECKPOINT', 'SHADOW_COLLECTION_ROOT', 'PREFLIGHT_ONLY'):
        env.pop(name, None)
    env.update(CHECKPOINT=job['checkpoint'], ROUTES=job['routes'], EVALUATION_OUTPUT_DIR=str(output),
               EVALUATION_AGENT=str(ROOT/'3rd_party/carla_garage/team_code/sensor_agent.py'),
               GPU_ID=str(job['gpu']), CARLA_PORT=str(job['carla_port']), TM_PORT=str(job['tm_port']),
               EVAL_CAPTURE_ALL_METRICS='1', EVAL_METRIC_ROOT=str(output),
               EVAL_TRACE_EVERY='1', EVAL_PROBE_EVERY='200', EVAL_PROBE_MAX='4',
               FMSR_DIAGNOSTICS='1', FMSR_DIAGNOSTIC_ROOT=str(output/'diagnostics'),
               FMSR_DIAGNOSTIC_SENSOR_EVERY='0', CARLA_RELOAD_EACH_ROUTE='0', EVAL_TIMEOUT='120')
    return env


def ready(job):
    prerequisite = Path(job['prerequisite'])
    try:
        # A repetition can wait behind another failure-subset reassessment.
        status_path = prerequisite/'reassessment_status.json'
        if status_path.exists():
            state = json.loads(status_path.read_text())
            complete = state.get('state') == 'complete'
        else:
            state = json.loads((prerequisite/'auto_eval_status.json').read_text())
            complete = state.get('monitor_status') == 'complete'
        if not complete or evaluation_is_locked(prerequisite):
            return False
    except (OSError, ValueError, KeyError):
        return False
    # This machine has one GPU; wait for all evaluators and CARLA servers to leave.
    return not any(any(a.endswith('/leaderboard_evaluator_v2.py') or
                           a.endswith('/CarlaUE4-Linux-Shipping') for a in proc['arguments'])
                   for proc in processes().values())


def run(args, job):
    output = args.output.resolve()
    with (output/'.reassessment.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('This reassessment is already queued/running.', flush=True)
            return
        def status(state, **extra):
            write_json(output/'reassessment_status.json', dict(state=state, pid=os.getpid(),
                       updated_at=updated_at(), route_count=job['route_count'], **extra))
        env = environment(job, output)
        with (output/'preflight.log').open('a') as handle:
            subprocess.run(['bash', str(LAUNCHER)], cwd=ROOT, env=dict(env, PREFLIGHT_ONLY='1'),
                           stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=180)
        source = json.loads((output/'agent/source.json').read_text())
        source['analysis_recording'] = job['analysis_recording']
        source['carla_client_timeout_seconds'] = 120
        write_json(output/'source.json', source)
        print(f"Prepared {job['route_count']} failure routes; waiting for {job['prerequisite']}", flush=True)
        while not ready(job):
            if (output/'reassessment.stop').exists():
                status('stopped'); return
            status('waiting_for_current_evaluation', prerequisite=job['prerequisite'])
            time.sleep(15)
        for path, expected in ((job['checkpoint'], job['checkpoint_sha256']),
                               (Path(job['checkpoint']).parent/'config.json', job['config_sha256']),
                               (job['routes'], job['routes_sha256'])):
            if sha256(path) != expected:
                status('error', reason=f'Pinned input changed: {path}')
                raise ValueError(f'Pinned input changed while waiting: {path}')
        os.environ.update(env)
        status('evaluating')
        watch(SimpleNamespace(output=output, gpu=job['gpu'], carla_port=job['carla_port'],
                              tm_port=job['tm_port'], poll_seconds=15, retry_seconds=30,
                              stale_seconds=900, heartbeat_seconds=30, rpc_timeout=5,
                              eval_timeout=120, launcher=LAUNCHER))
        # The evaluator may still be flushing telemetry after the final checkpoint.
        while evaluation_is_locked(output):
            status('waiting_for_metrics_flush'); time.sleep(15)
        final, _ = inspect_checkpoint(output/'checkpoint_endpoint.json', job['route_ids'])
        if not final['complete']:
            raise RuntimeError('Watchdog stopped before all selected routes finished')
        status('summarizing')
        reports = summarize_analysis(output)
        records = json.loads((output/'checkpoint_endpoint.json').read_text())['_checkpoint']['records']
        before = {row['route_id']:row for row in json.loads((output/'baseline_selected_records.json').read_text())}
        comparison = []
        for row in records:
            old = before[row['route_id']]
            comparison.append(dict(route_id=row['route_id'], scenario=row['scenario_name'],
                                   previous_status=old['status'], status=row['status'],
                                   previous_DS=old['scores']['score_composed'], DS=row['scores']['score_composed'],
                                   delta_DS=row['scores']['score_composed']-old['scores']['score_composed'],
                                   rescued=not failed(row), infractions=row['infractions']))
        write_json(output/'analysis/paired_failure_reassessment.json',
                   dict(route_count=len(comparison), rescued_count=sum(r['rescued'] for r in comparison),
                        delta_DS_mean=sum(r['delta_DS'] for r in comparison)/len(comparison),
                        scope='selected historical failures; not a 220-route overall score', routes=comparison))
        status('complete', reports=reports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--prerequisite', type=Path, default=PREREQUISITE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--carla-port', type=int, default=8890)
    parser.add_argument('--tm-port', type=int, default=8060)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    parser.add_argument('--wait-for-baseline', action='store_true',
                        help='Select failures only after the prerequisite baseline finishes')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.wait_for_baseline:
        if args.baseline.resolve() != args.prerequisite.resolve() or args.prepare_only:
            parser.error('--wait-for-baseline requires baseline=prerequisite and no --prepare-only')
        args.output.mkdir(parents=True, exist_ok=True)
        if args.foreground:
            with (args.output/'.failure_selection.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print('Failure selection is already queued.', flush=True)
                    return
                while not ready({'prerequisite': str(args.baseline.resolve())}):
                    if (args.output/'reassessment.stop').exists():
                        write_json(args.output/'reassessment_status.json',
                                   dict(state='stopped', pid=os.getpid(), updated_at=updated_at()))
                        return
                    write_json(args.output/'reassessment_status.json',
                               dict(state='waiting_to_select_first_round_failures', pid=os.getpid(),
                                    updated_at=updated_at(), baseline=str(args.baseline.resolve()),
                                    route_count=None,
                                    selection='RC<100 or any infraction except min_speed_infractions'))
                    time.sleep(15)
                records = json.loads((args.baseline/'checkpoint_endpoint.json').read_text())['_checkpoint']['records']
                if not any(failed(row) for row in records):
                    write_json(args.output/'reassessment_status.json',
                               dict(state='complete', pid=os.getpid(), updated_at=updated_at(),
                                    route_count=0, reason='No failures in the first round'))
                    return
                run(args, prepare(args))
                return
        # The selector itself must survive terminal closure while the baseline runs.
        command = [sys.executable, '-u', '-m', 'scripts.pdm_lite.reevaluate_legacy_fm_cr_failures']
        command += sys.argv[1:] + ['--foreground']
        with (args.output/'queue.log').open('a') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Failure-selection queue PID {child.pid}; status: {args.output}/reassessment_status.json')
        return
    job = prepare(args)
    if args.prepare_only:
        env = environment(job, args.output.resolve())
        subprocess.run(['bash', str(LAUNCHER)], cwd=ROOT, env=dict(env, PREFLIGHT_ONLY='1'), check=True)
        print(f"Prepared {job['route_count']} routes at {job['routes']}")
    elif args.foreground:
        run(args, job)
    else:
        command = [sys.executable, '-u', '-m', 'scripts.pdm_lite.reevaluate_legacy_fm_cr_failures']
        command += sys.argv[1:] + ['--foreground']
        with (args.output/'queue.log').open('a') as handle:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Queue PID {child.pid}; status: {args.output}/reassessment_status.json')


if __name__ == '__main__':
    main()
