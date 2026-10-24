"""Run a persisted evaluation queue after a prerequisite evaluation finishes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from scripts.pdm_lite.watch_evaluation import (
    ROOT, atomic_write, completed, matching_processes, process_table, read_checkpoint,
)


def finished_and_idle(directory, ids, launcher):
    directory = Path(directory)
    try:
        if not completed(read_checkpoint(directory/'checkpoint_endpoint.json', ids), len(ids)):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if matching_processes(process_table(), directory.resolve(), Path(launcher).resolve()):
        return False
    for name in ('.evaluation.lock', '.watchdog.lock'):
        with (directory/name).open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.resolve().parent
    ids = [f"RouteScenario_{r.get('id')}_rep0" for r in
           ET.parse(manifest['routes']).getroot().findall('route')]
    with (directory/'queue.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Queue is already running.', flush=True)
            return 0
        (directory/'queue.pid').write_text(str(os.getpid()))
        def status(state, **extra):
            atomic_write(directory/'status.json', dict(state=state, pid=os.getpid(),
                         updated=time.strftime('%Y-%m-%d %H:%M:%S'), **extra))
        prerequisite = manifest['prerequisite']
        print('Waiting for prerequisite:', prerequisite['output'], flush=True)
        while not finished_and_idle(prerequisite['output'], ids, prerequisite['launcher']):
            if (directory/'queue.stop').exists():
                status('stopped')
                return 0
            status('waiting_for_prerequisite', output=prerequisite['output'])
            time.sleep(15)
        for stage in manifest['stages']:
            output = Path(stage['output'])
            output.mkdir(parents=True, exist_ok=True)
            if finished_and_idle(output, ids, stage['launcher']):
                continue
            if (directory/'queue.stop').exists():
                status('stopped')
                return 0
            env = os.environ.copy()
            for name in ('SHADOW_TEACHER_CHECKPOINT', 'SHADOW_COLLECTION_ROOT',
                         'FMSR_DIAGNOSTIC_ROOT', 'PREFLIGHT_ONLY', 'ALLOW_EXPERIMENTAL_CLEAN_PRIOR'):
                env.pop(name, None)
            env.update(manifest['environment'])
            env.update(CHECKPOINT=stage['checkpoint'], EVALUATION_OUTPUT_DIR=str(output),
                       ROUTES=manifest['routes'])
            print('Starting stage:', stage['name'], output, flush=True)
            status('running', stage=stage['name'], output=str(output))
            with (output/'watchdog.log').open('ab') as log:
                child = subprocess.Popen([sys.executable, str(ROOT/'scripts/pdm_lite/watch_evaluation.py'),
                    '--output-dir', str(output), '--routes', manifest['routes'],
                    '--launcher', stage['launcher'], '--max-no-progress-restarts', '10'],
                    cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait()
            if code != 0:
                status('blocked', stage=stage['name'], returncode=code,
                       reason='Watchdog exited before completion; inspect stage watchdog.log')
                return code
            while not finished_and_idle(output, ids, stage['launcher']):
                if (output/'watchdog.stop').exists() or (directory/'queue.stop').exists():
                    status('stopped', stage=stage['name'])
                    return 0
                time.sleep(15)
            print('Completed stage:', stage['name'], flush=True)
        status('completed')
        print('All queued evaluations completed.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
