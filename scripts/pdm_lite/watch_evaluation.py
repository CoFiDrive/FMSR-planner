#!/usr/bin/env python3
"""Attach to a Bench2Drive evaluation and resume it after process exit."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / 'scripts/eval_bench2drive_pdm_lite_fmsr.sh'
DEFAULT_OUTPUT = ROOT / 'outputs/local_evaluation/bench2drive220_pdm_lite_seed2_epoch09_continuous_speed'


def atomic_write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def read_checkpoint(path, expected_ids):
    """Reject malformed checkpoints that would make the evaluator restart at zero."""
    data = json.loads(path.read_text())
    checkpoint = data['_checkpoint']
    done, total = checkpoint['progress']
    records = checkpoint['records']
    if (total != len(expected_ids) or not 0 <= done <= len(records) <= total
            or data.get('entry_status') not in ('Started', 'Finished', 'Crashed')):
        raise ValueError('Invalid progress or entry_status')
    for i, record in enumerate(records):
        if record['route_id'] != expected_ids[i]:
            raise ValueError('Route IDs differ from the requested benchmark')
    return data


def completed(data, total):
    if data is None:
        return False
    c = data['_checkpoint']
    return (data['entry_status'] == 'Finished' and c['progress'] == [total, total]
            and len(c['records']) == total and bool(c.get('global_record')))


def process_table():
    result = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            if stat[0] == 'Z':
                continue
            argv = (entry / 'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
            result[int(entry.name)] = dict(argv=argv, ppid=int(stat[1]), birth=stat[19],
                                           cwd=(entry / 'cwd').resolve())
        except (OSError, ValueError):
            continue
    return result


def matching_processes(table, output, launcher):
    found = set()
    endpoint = output / 'checkpoint_endpoint.json'
    for pid, process in table.items():
        argv = process['argv']
        # Inspect actual argv tokens, never substring-match another command's text.
        for index, arg in enumerate(argv):
            if arg.startswith('--checkpoint='):
                value = arg.split('=', 1)[1]
            elif arg == '--checkpoint' and index + 1 < len(argv):
                value = argv[index + 1]
            else:
                continue
            if (process['cwd'] / value).resolve() == endpoint:
                found.add(pid)
        if len(argv) > 1 and Path(argv[0]).name in ('bash', 'sh'):
            if (process['cwd'] / argv[1]).resolve() == launcher:
                try:
                    environ = dict(item.split('=', 1) for item in
                                   (Path('/proc') / str(pid) / 'environ').read_bytes()
                                   .decode(errors='replace').split('\0') if '=' in item)
                    run_output = environ.get('EVALUATION_OUTPUT_DIR', str(DEFAULT_OUTPUT))
                    if (process['cwd'] / run_output).resolve() == output:
                        found.add(pid)
                except OSError:
                    pass
    return found


class Watchdog:
    def __init__(self, output, expected_ids, poll=10, retry_delay=30, max_stalled=10,
                 launcher=LAUNCHER, missing_carla_grace=90, stale_log_seconds=600):
        self.output = output.resolve()
        self.ids = expected_ids
        self.poll = poll
        self.retry_delay = retry_delay
        self.max_stalled = max_stalled
        self.launcher = launcher.resolve()
        self.missing_carla_grace = missing_carla_grace
        self.stale_log_seconds = stale_log_seconds
        self.child = None
        self.carla = {}
        self.restart_count = 0
        self.stalled = 0
        self.best_done = -1
        self.was_active = False
        self.launched_here = False
        self.attempt_progress = False
        self.endpoint = self.output / 'checkpoint_endpoint.json'
        self.backup = self.output / 'checkpoint_endpoint.last_valid.json'
        self.last_log_change = time.monotonic()
        self.last_log_mtime_ns = None
        self.missing_carla_since = None

    def log(self, message):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

    def status(self, state, **extra):
        atomic_write(self.output / 'watchdog_status.json', dict(
            state=state, pid=os.getpid(), restarts=self.restart_count,
            consecutive_no_progress=self.stalled, updated=time.strftime('%Y-%m-%d %H:%M:%S'),
            **extra))

    def remember_carla(self, table, active):
        descendants = set(active)
        while True:
            expanded = descendants | {pid for pid, p in table.items() if p['ppid'] in descendants}
            if expanded == descendants:
                break
            descendants = expanded
        for pid in descendants:
            p = table[pid]
            if any('CarlaUE4' in a for a in p['argv'][:3]):
                self.carla[pid] = p['birth']

    def carla_alive(self, table):
        return any(pid in table and table[pid]['birth'] == birth
                   for pid, birth in self.carla.items())

    def note_log_activity(self):
        try:
            mtime_ns = (self.output / 'evaluation.log').stat().st_mtime_ns
        except OSError:
            return
        if mtime_ns != self.last_log_mtime_ns:
            self.last_log_mtime_ns = mtime_ns
            self.last_log_change = time.monotonic()

    def terminate_stalled_evaluation(self, active, table, reason):
        """Stop only processes whose command line identifies this evaluation."""
        targets = {pid: table[pid]['birth'] for pid in active if pid in table}
        self.log(f'{reason}; terminating evaluation processes {sorted(targets)}')
        self.status('restarting_stalled', reason=reason, evaluation_pids=sorted(targets))
        for sig in (signal.SIGTERM, signal.SIGKILL):
            current = process_table()
            live = [pid for pid, birth in targets.items()
                    if pid in current and current[pid]['birth'] == birth]
            if not live:
                break
            for pid in live:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(5)

    def cleanup_orphans(self):
        # Only the CARLA descendants observed under this evaluation are eligible.
        # PID birth times protect against PID reuse; unrelated CARLA runs are untouched.
        for sig in (signal.SIGTERM, signal.SIGKILL):
            table = process_table()
            live = [pid for pid, birth in self.carla.items()
                    if pid in table and table[pid]['birth'] == birth]
            if not live:
                break
            self.log(f'Cleaning up this evaluation\'s orphan CARLA processes: {live}')
            for pid in live:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
            if sig == signal.SIGTERM:
                time.sleep(5)
        self.carla.clear()

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / '.watchdog.lock').open('w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.log('A watchdog already monitors this output directory.')
                return 0
            (self.output / 'watchdog.pid').write_text(str(os.getpid()) + '\n')
            self.log(f'Monitoring {self.endpoint}; poll={self.poll}s, retry delay={self.retry_delay}s')
            while True:
                if (self.output / 'watchdog.stop').exists():
                    self.status('stopped')
                    self.log('Stop requested; any running evaluation is left running.')
                    return 0
                table = process_table()
                active = matching_processes(table, self.output, self.launcher)
                if self.child is not None and self.child.poll() is None:
                    active.add(self.child.pid)
                active &= table.keys()
                data = None
                try:
                    data = read_checkpoint(self.endpoint, self.ids)
                except (OSError, ValueError, KeyError, TypeError):
                    pass  # It may be halfway through a write while evaluation is alive.
                if data is not None:
                    atomic_write(self.backup, data)
                    done = data['_checkpoint']['progress'][0]
                    if done > self.best_done:
                        if self.best_done >= 0:
                            self.attempt_progress = True
                        self.best_done = done
                        self.stalled = 0
                        self.log(f'Progress: {done}/{len(self.ids)}')
                if completed(data, len(self.ids)):
                    self.status('completed', progress=[len(self.ids), len(self.ids)])
                    self.log('All routes evaluated and global statistics written; stopping watchdog.')
                    return 0
                self.note_log_activity()
                if active:
                    if not self.was_active:
                        self.log(f'Following existing evaluation processes {sorted(active)}; no duplicate launch.')
                    self.remember_carla(table, active)
                    self.was_active = True
                    now = time.monotonic()
                    if self.carla and not self.carla_alive(table):
                        if self.missing_carla_since is None:
                            self.missing_carla_since = now
                        elif now - self.missing_carla_since >= self.missing_carla_grace:
                            self.terminate_stalled_evaluation(active, table, 'CARLA process disappeared')
                            time.sleep(self.poll)
                            continue
                    else:
                        self.missing_carla_since = None
                    if now - self.last_log_change >= self.stale_log_seconds:
                        self.terminate_stalled_evaluation(active, table, 'Evaluation log stopped updating')
                        time.sleep(self.poll)
                        continue
                    self.status('monitoring', evaluation_pids=sorted(active),
                                progress=data['_checkpoint']['progress'] if data else None)
                    time.sleep(self.poll)
                    continue
                if self.was_active or self.launched_here:
                    self.log(f'Evaluation exited before completion; return code: '
                             f'{self.child.poll() if self.child else "unavailable (attached process)"}')
                    self.stalled = 0 if self.attempt_progress else self.stalled + 1
                    self.attempt_progress = False
                    self.was_active = False
                    self.launched_here = False
                    self.missing_carla_since = None
                    self.last_log_change = time.monotonic()
                self.cleanup_orphans()
                if data is None and (self.endpoint.exists() or self.backup.exists()):
                    try:
                        data = read_checkpoint(self.backup, self.ids)
                    except (OSError, ValueError, KeyError, TypeError):
                        self.status('blocked', reason='Unreadable checkpoint and no valid backup')
                        self.log('Cannot safely resume: no valid checkpoint backup. Preserving existing files.')
                        return 2
                    if self.endpoint.exists():
                        archive = self.output / f'checkpoint_unreadable_{time.time_ns()}.json'
                        archive.write_bytes(self.endpoint.read_bytes())
                    atomic_write(self.endpoint, data)
                    self.log('Restored last valid checkpoint after an interrupted checkpoint write.')
                if self.stalled >= self.max_stalled:
                    self.status('blocked', reason='Repeated restarts without route progress')
                    self.log(f'Stopping after {self.stalled} exits without sustained progress; inspect logs.')
                    return 3
                self.status('waiting_to_restart', progress=data['_checkpoint']['progress'] if data else None)
                self.log(f'Resuming in {self.retry_delay}s; completed driving failures remain in results.')
                time.sleep(self.retry_delay)
                if (self.output / 'watchdog.stop').exists():
                    continue
                # A manual launch may have happened during cooldown.
                if matching_processes(process_table(), self.output, self.launcher):
                    continue
                self.restart_count += 1
                env = os.environ.copy()
                env['EVALUATION_OUTPUT_DIR'] = str(self.output)
                env.pop('PREFLIGHT_ONLY', None)
                source = self.output / 'agent/source.json'
                if source.exists():
                    env['CHECKPOINT'] = json.loads(source.read_text())['checkpoint']
                with (self.output / f'restart_{self.restart_count:03d}_{time.time_ns()}.log').open('ab') as log:
                    self.child = subprocess.Popen(['bash', str(self.launcher)], cwd=ROOT,
                                                  env=env, stdin=subprocess.DEVNULL,
                                                  stdout=log, stderr=subprocess.STDOUT,
                                                  start_new_session=True)
                self.launched_here = True
                self.log(f'Started resume attempt {self.restart_count}, launcher PID {self.child.pid}')
                time.sleep(self.poll)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--launcher', type=Path, default=LAUNCHER)
    parser.add_argument('--routes', type=Path,
                        default=Path(os.environ.get('ROUTES', str(
                            ROOT / '3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml'))))
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--retry-delay', type=float, default=30)
    parser.add_argument('--max-no-progress-restarts', type=int, default=10)
    parser.add_argument('--missing-carla-grace-seconds', type=float, default=90)
    parser.add_argument('--stale-log-seconds', type=float, default=600)
    args = parser.parse_args()
    if min(args.poll_seconds, args.retry_delay, args.max_no_progress_restarts,
           args.missing_carla_grace_seconds, args.stale_log_seconds) <= 0:
        parser.error('Intervals and retry limit must be positive')
    routes = args.routes.resolve(strict=True)
    os.environ['ROUTES'] = str(routes)
    expected_ids = [f"RouteScenario_{r.get('id')}_rep0" for r in ET.parse(routes).getroot().findall('route')]
    return Watchdog(args.output_dir, expected_ids, args.poll_seconds,
                    args.retry_delay, args.max_no_progress_restarts,
                    launcher=args.launcher.resolve(strict=True),
                    missing_carla_grace=args.missing_carla_grace_seconds,
                    stale_log_seconds=args.stale_log_seconds).run()


if __name__ == '__main__':
    raise SystemExit(main())
