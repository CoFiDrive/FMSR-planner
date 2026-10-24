"""Attach to an evaluation and resume it after exits until every route is scored."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from .bootstrap import PACKAGE_ROOT, WORKSPACE_ROOT


def log(message):
    print(f'[{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}] {message}', flush=True)


def updated_at():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def atomic_write(path, content):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_bytes(content)
    temporary.replace(path)


def inspect_checkpoint(path, expected_ids):
    if not path.exists():
        return {'progress': [0, len(expected_ids)], 'records': 0,
                'complete': False, 'infrastructure_errors': []}, None
    payload = path.read_bytes()
    result = json.loads(payload)
    checkpoint = result['_checkpoint']
    progress, records = checkpoint['progress'], checkpoint['records']
    if (len(progress) != 2 or progress[1] != len(expected_ids)
            or not isinstance(progress[0], int) or not 0 <= progress[0] <= len(expected_ids)
            or not progress[0] <= len(records) <= len(expected_ids)
            or [r['route_id'] for r in records] != expected_ids[:len(records)]):
        raise ValueError('Checkpoint routes/progress do not match this evaluation')
    def infrastructure_failure(record):
        return record['status'] == 'Failed' or any(term in record['status'].lower()
            for term in ('crash', 'simulation', 'set up', 'setup'))
    # Leaderboard retains later records while rewinding progress for a retry.
    # Those are pending old attempts, not newly failed routes in this pass.
    infrastructure_errors = [r['route_id'] for r in records[:progress[0]]
                             if infrastructure_failure(r)]
    pending_retry_routes = [r['route_id'] for r in records[progress[0]:]
                            if infrastructure_failure(r)]
    # TickRuntime (200 simulated seconds) and Agent got blocked are evaluated
    # driving failures, not infrastructure errors to retry indefinitely.
    complete = (result.get('entry_status') == 'Finished'
                and progress == [len(expected_ids), len(expected_ids)]
                and len(records) == len(expected_ids) and not infrastructure_errors)
    return {'progress': progress, 'records': len(records), 'complete': complete,
            'entry_status': result.get('entry_status'),
            'infrastructure_errors': infrastructure_errors,
            'pending_retry_routes': pending_retry_routes}, payload


def evaluation_is_locked(output):
    with (output/'.evaluation.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def processes():
    """Read only Linux process identity/ancestry; never use a broad pkill."""
    found = {}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        try:
            fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
            arguments = [a.decode(errors='replace') for a in
                         (directory/'cmdline').read_bytes().split(b'\0') if a]
            found[int(directory.name)] = {'ppid': int(fields[1]),
                'start_ticks': int(fields[19]), 'arguments': arguments}
        except (OSError, ValueError, IndexError):
            continue
    return found


def evaluation_processes(table, checkpoint):
    expected = '--checkpoint=' + str(checkpoint)
    return {pid for pid, p in table.items() if expected in p['arguments']
            and any(a.endswith('/leaderboard_evaluator_v2.py') for a in p['arguments'])}


def associated_carla(table, evaluator_pids):
    result = {}
    for pid, proc in table.items():
        if not any(a.endswith('/CarlaUE4-Linux-Shipping') for a in proc['arguments']):
            continue
        parent, visited = proc['ppid'], set()
        while parent in table and parent not in visited:
            if parent in evaluator_pids:
                result[str(pid)] = proc
                break
            visited.add(parent)
            parent = table[parent]['ppid']
    return result


def cleanup_orphan_servers(table, tracked, shutdown_requested):
    remaining = False
    for pid_text, saved in tracked.items():
        pid = int(pid_text)
        live = table.get(pid)
        if (not live or live['start_ticks'] != saved['start_ticks']
                or live['arguments'] != saved['arguments']):
            continue
        remaining = True
        now = time.monotonic()
        previous = shutdown_requested.get(pid)
        if previous is None or now - previous > 30:
            signum = signal.SIGTERM if previous is None else signal.SIGKILL
            try:
                os.kill(pid, signum)
                log(f'Stopping CARLA PID {pid} left by the exited evaluator ({signum.name})')
                shutdown_requested[pid] = now
            except ProcessLookupError:
                pass
    return remaining


def carla_rpc_port(proc):
    for argument in proc['arguments']:
        if argument.startswith('-carla-rpc-port='):
            return int(argument.split('=', 1)[1])
    return None


def probe_carla_frame(servers, timeout):
    """Read-only RPC: never tick, reload, or change the running world."""
    import carla
    frames = []
    for pid, proc in sorted(servers.items()):
        port = carla_rpc_port(proc)
        if port is None:
            continue
        client = carla.Client('127.0.0.1', port)
        client.set_timeout(timeout)
        frame = int(client.get_world().get_snapshot().frame)
        frames.append((pid, proc['start_ticks'], port, frame))
    return tuple(frames) if frames else None


class FrameHeartbeat:
    """Detect a stalled simulator while allowing a long, actively ticking route."""
    def __init__(self, now, interval, timeout):
        self.interval, self.timeout = interval, timeout
        self.reset(now)

    def reset(self, now):
        self.last_activity = now
        self.next_probe = now
        self.frames = None
        self.error = None

    def observe(self, frames, error, now):
        if frames is not None and frames != self.frames:
            self.last_activity = now
            self.frames = frames
        self.error = error

    def poll(self, servers, now):
        if now < self.next_probe:
            return
        self.next_probe = now + self.interval
        try:
            self.observe(isolated_carla_frame(servers, self.timeout) if servers else None, None, now)
        except Exception as error:
            self.observe(None, f'{type(error).__name__}: {str(error)[:400]}', now)

    def report(self, now):
        return {'seconds_without_frame_progress': max(0., now-self.last_activity),
                'frames': self.frames, 'rpc_error': self.error}


def isolated_carla_frame(servers, timeout):
    """Contain CARLA native aborts during world/actor destruction in a short-lived process."""
    command = ('import json,sys,resource; resource.setrlimit(resource.RLIMIT_CORE,(0,0)); '
               'from fmsrpdm.evaluation_watchdog import probe_carla_frame; '
               'print(json.dumps(probe_carla_frame(json.loads(sys.stdin.read()),float(sys.argv[1]))))')
    result = subprocess.run([sys.executable, '-c', command, str(timeout)],
                            input=json.dumps(servers), text=True, capture_output=True,
                            timeout=max(10., timeout*len(servers)+5.))
    if result.returncode:
        raise RuntimeError(f'Isolated CARLA probe exited {result.returncode}: {result.stderr[-400:]}')
    frames = json.loads(result.stdout)
    return tuple(tuple(row) for row in frames) if frames else None


def stop_matching_evaluators(table, pids, stopped, now):
    """Only signal positively identified evaluators for this checkpoint."""
    for pid in pids:
        proc = table[pid]
        key = (pid, proc['start_ticks'])
        previous = stopped.get(key)
        if previous is not None and now-previous < 30:
            continue
        signum = signal.SIGTERM if previous is None else signal.SIGKILL
        try:
            os.kill(pid, signum)
            log(f'Restarting stalled evaluator PID {pid} ({signum.name})')
            if previous is None:
                stopped[key] = now
        except ProcessLookupError:
            pass


def watch(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((output/'source.json').read_text())
    routes = Path(source['routes'])
    route_ids = [f"RouteScenario_{r.get('id')}_rep0" for r in ET.parse(routes).getroot().findall('route')]
    if len(route_ids) != source['route_count'] or len(set(route_ids)) != len(route_ids):
        raise ValueError('Source metadata and route XML differ')
    checkpoint = output/'checkpoint_endpoint.json'
    with (output/'.evaluation_watchdog.lock').open('a') as monitor_lock:
        try:
            fcntl.flock(monitor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('A watchdog already monitors this result directory; no duplicate was started')
            return
        atomic_write(output/'watchdog.pid', f'{os.getpid()}\n'.encode())
        stop = [False]
        def stop_monitor(signum, frame):
            stop[0] = True
        signal.signal(signal.SIGTERM, stop_monitor)
        signal.signal(signal.SIGINT, stop_monitor)
        environment = os.environ.copy()
        environment.update(CHECKPOINT=source['checkpoint'], ROUTES=str(routes),
            EVALUATION_OUTPUT_DIR=str(output), GPU_ID=str(args.gpu),
            CARLA_PORT=str(args.carla_port), TM_PORT=str(args.tm_port), PREFLIGHT_ONLY='0',
            ALLOW_EXPERIMENTAL_CLEAN_PRIOR='1' if source.get('experimental_architecture') else '0',
            EVAL_TIMEOUT=str(source.get('carla_client_timeout_seconds', getattr(args, 'eval_timeout', 120))),
            VERIFY_EVAL_MODEL='0')
        if 'analysis_recording' in source:
            recording = source['analysis_recording']
            environment.update(EVAL_CAPTURE_ALL_METRICS='1', EVAL_METRIC_ROOT=str(output),
                EVAL_TRACE_EVERY=str(recording['trace_every']),
                EVAL_PROBE_EVERY=str(recording['probe_every']), EVAL_PROBE_MAX=str(recording['probe_max']))
        child, attempts, last_exit = None, 0, None
        next_start, last_progress, last_message = 0., -1, None
        tracked_path = output/'watchdog_carla.json'
        tracked = json.loads(tracked_path.read_text()) if tracked_path.exists() else {}
        shutdown_requested = {}
        stalled_evaluators = {}
        stale_seconds = getattr(args, 'stale_seconds', 0.)
        heartbeat = FrameHeartbeat(time.monotonic(), getattr(args, 'heartbeat_seconds', 30.),
                                   getattr(args, 'rpc_timeout', 5.))
        last_good = output/'checkpoint.last_watchdog_valid.json'
        log(f'Monitoring {len(route_ids)} routes in {output}')
        with (output/'watchdog_eval_launcher.log').open('a', buffering=1) as evaluator_log:
            while not stop[0]:
                if child is not None and child.poll() is not None:
                    last_exit = child.returncode
                    log(f'Evaluator exited with code {last_exit}; checking saved results')
                    child = None
                    next_start = time.monotonic() + args.retry_seconds
                table = processes()
                active_pids = evaluation_processes(table, checkpoint)
                new_servers = associated_carla(table, active_pids)
                if any(tracked.get(k) != v for k, v in new_servers.items()):
                    tracked.update(new_servers)
                    atomic_write(tracked_path, (json.dumps(tracked, indent=2)+'\n').encode())
                active = bool(active_pids) or evaluation_is_locked(output) or child is not None
                # A killed evaluator can leave CARLA holding the tee pipe and
                # evaluation lock open. Clean only its previously tracked servers.
                if not active_pids and tracked:
                    cleanup_orphan_servers(table, tracked, shutdown_requested)
                try:
                    state, payload = inspect_checkpoint(checkpoint, route_ids)
                    if payload is not None:
                        atomic_write(last_good, payload)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    state = {'complete': False, 'checkpoint_error': str(error)}
                    # A writer may be between truncate/write. Never restore while
                    # an evaluator is alive. A readable, matching backup is required.
                    if not active and last_good.exists():
                        inspect_checkpoint(last_good, route_ids)
                        if checkpoint.exists():
                            checkpoint.rename(output/f'checkpoint.corrupt_{time.time_ns()}.json')
                        atomic_write(checkpoint, last_good.read_bytes())
                        log('Restored the last readable checkpoint after evaluator exit')
                        state, _ = inspect_checkpoint(checkpoint, route_ids)
                    elif not active and isinstance(error, json.JSONDecodeError):
                        # The first writer may have died before a readable snapshot
                        # existed. Preserve its bytes, then let leaderboard start.
                        if checkpoint.exists():
                            checkpoint.rename(output/f'checkpoint.corrupt_{time.time_ns()}.json')
                        state, _ = inspect_checkpoint(checkpoint, route_ids)
                        log('Archived an unreadable initial checkpoint; restarting without scored records')
                state.update(watchdog_pid=os.getpid(), attempts=attempts,
                             evaluator_pids=sorted(active_pids), last_exit_code=last_exit,
                             updated_at=updated_at())
                if state['complete']:
                    state['monitor_status'] = 'complete'
                    atomic_write(output/'watchdog_status.json', (json.dumps(state, indent=2)+'\n').encode())
                    log(f'All {len(route_ids)} routes have finished. Watchdog exits.')
                    return
                if active:
                    message = ('evaluation_running_with_errors'
                               if state.get('infrastructure_errors') else 'evaluation_running')
                    if stale_seconds:
                        now = time.monotonic()
                        if any((pid,table[pid]['start_ticks']) in stalled_evaluators for pid in active_pids):
                            stop_matching_evaluators(table, active_pids, stalled_evaluators, now)
                        if state.get('progress', [last_progress])[0] != last_progress:
                            heartbeat.reset(now)
                        heartbeat.poll(new_servers, now)
                        state['carla_heartbeat'] = heartbeat.report(now)
                        if now-heartbeat.last_activity >= stale_seconds:
                            stop_matching_evaluators(table, active_pids, stalled_evaluators, now)
                            if child is not None and child.poll() is None:
                                # The launcher is our own new session. Its group
                                # contains tee and the evaluation shell, never training.
                                try:
                                    os.killpg(child.pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                            cleanup_orphan_servers(table, tracked, shutdown_requested)
                            message = 'restarting_stalled_evaluation'
                elif 'checkpoint_error' in state:
                    message = 'waiting_for_readable_checkpoint'
                elif cleanup_orphan_servers(table, tracked, shutdown_requested):
                    message = 'cleaning_orphan_carla'
                elif time.monotonic() < next_start:
                    message = 'waiting_to_restart'
                else:
                    attempts += 1
                    if checkpoint.exists():
                        atomic_write(output/f'checkpoint.before_watchdog_resume_{time.time_ns()}.json',
                                     checkpoint.read_bytes())
                    log(f'Starting resume attempt {attempts} with the same model and routes')
                    launcher = getattr(args, 'launcher', PACKAGE_ROOT/'scripts/evaluate.sh')
                    child = subprocess.Popen(['bash', str(launcher)],
                        cwd=WORKSPACE_ROOT, env=environment, stdin=subprocess.DEVNULL,
                        stdout=evaluator_log, stderr=subprocess.STDOUT, start_new_session=True)
                    message = 'evaluation_starting'
                    state['attempts'] = attempts
                    state['launcher_pid'] = child.pid
                    heartbeat.reset(time.monotonic())
                state['monitor_status'] = message
                atomic_write(output/'watchdog_status.json', (json.dumps(state, indent=2)+'\n').encode())
                progress = state.get('progress', [last_progress])[0]
                if message != last_message or progress != last_progress:
                    log(f'{message}; progress={state.get("progress")}; infrastructure_errors={state.get("infrastructure_errors", [])}')
                    last_message, last_progress = message, progress
                time.sleep(min(args.poll_seconds, 60.))
        atomic_write(output/'watchdog_status.json', (json.dumps({
            'monitor_status': 'stopped', 'watchdog_pid': os.getpid(),
            'updated_at': updated_at()}, indent=2)+'\n').encode())
        log('Watchdog stopped; any running evaluation is left running')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--carla-port', type=int, default=8870)
    parser.add_argument('--tm-port', type=int, default=8040)
    parser.add_argument('--poll-seconds', type=float, default=15)
    parser.add_argument('--retry-seconds', type=float, default=30)
    parser.add_argument('--stale-seconds', type=float, default=900,
                        help='Restart if no simulator frames advance; 0 disables the RPC heartbeat')
    parser.add_argument('--heartbeat-seconds', type=float, default=30)
    parser.add_argument('--rpc-timeout', type=float, default=5)
    parser.add_argument('--eval-timeout', type=float, default=120)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.poll_seconds <= 0 or args.retry_seconds < 0:
        parser.error('Invalid polling/retry interval')
    if args.stale_seconds < 0 or min(args.heartbeat_seconds, args.rpc_timeout, args.eval_timeout) <= 0:
        parser.error('Invalid simulator heartbeat/timeout settings')
    if not (args.output/'source.json').is_file():
        parser.error('Run evaluation preflight once to create source.json in the result directory')
    if args.background:
        output = args.output.resolve()
        command = [sys.executable, '-u', '-m', 'fmsrpdm.evaluation_watchdog']
        command += [a for a in sys.argv[1:] if a != '--background']
        with (output/'watchdog.log').open('a') as handle:
            child = subprocess.Popen(command, cwd=WORKSPACE_ROOT, stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            log(f'Background watchdog PID {child.pid}; log={output/"watchdog.log"}')
        else:
            if code:
                raise SystemExit(f'Watchdog exited with code {code}; inspect {output/"watchdog.log"}')
            log('Watchdog finished or was already running; inspect watchdog.log')
        return
    watch(args)


if __name__ == '__main__':
    main()
