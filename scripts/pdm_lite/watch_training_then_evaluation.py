#!/usr/bin/env python3
"""Wait for the final recovery epoch, then supervise a resumable evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from watch_evaluation import ROOT, atomic_write, completed, process_table, read_checkpoint


def training_processes(table, run_id):
    found = []
    for pid, info in table.items():
        argv = info['argv']
        if not any(Path(arg).name == 'train.py' for arg in argv):
            continue
        if any(arg == '--id=' + run_id or
               (arg == '--id' and i + 1 < len(argv) and argv[i + 1] == run_id)
               for i, arg in enumerate(argv)):
            found.append(pid)
    return sorted(found)


def checkpoint_files(training_dir, epoch):
    return [training_dir / f'{prefix}_{epoch:04d}.pth'
            for prefix in ('model', 'optimizer', 'scaler', 'scheduler')]


def final_checkpoint_ready(training_dir, epoch, active, stable_seconds=30):
    if active:
        return False
    now = time.time()
    try:
        return all(path.stat().st_size > 0 and now - path.stat().st_mtime >= stable_seconds
                   for path in checkpoint_files(training_dir, epoch))
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--final-epoch', type=int, default=22)
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    training = args.training_dir.resolve(strict=True)
    output = args.output_dir.resolve()
    routes = args.routes.resolve(strict=True)
    ids = [f"RouteScenario_{r.get('id')}_rep0"
           for r in ET.parse(routes).getroot().findall('route')]
    if len(ids) != 96 or len(set(ids)) != 96:
        parser.error('Expected exactly 96 distinct failed benchmark routes')
    training_args = json.loads((training / 'args.txt').read_text())
    if training_args['epochs'] != args.final_epoch + 1:
        parser.error('Final epoch does not match the running training configuration')
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'pipeline_status.json'
    checkpoint = checkpoint_files(training, args.final_epoch)[0]

    def status(state, **extra):
        atomic_write(status_path, dict(state=state, pid=os.getpid(),
                     updated=time.strftime('%Y-%m-%d %H:%M:%S'),
                     training_dir=str(training), checkpoint=str(checkpoint),
                     routes=str(routes), route_count=len(ids), **extra))

    def log(message):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

    with (output / '.pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('This pipeline is already running; leaving it in control.')
            return 0
        (output / 'pipeline.pid').write_text(str(os.getpid()) + '\n')
        last_epoch = None
        while True:
            active = training_processes(process_table(), training.name)
            epochs = sorted(int(p.stem.split('_')[1]) for p in training.glob('model_*.pth'))
            latest = epochs[-1] if epochs else None
            if latest != last_epoch:
                log(f'Latest saved training epoch: {latest}; target: {args.final_epoch}')
                last_epoch = latest
            if final_checkpoint_ready(training, args.final_epoch, active):
                break
            status('waiting_for_training' if active else 'waiting_for_final_checkpoint',
                   training_pids=active, latest_saved_epoch=latest)
            time.sleep(args.poll_seconds)
        # Verify the final model can be fully read before staging or launching CARLA.
        import torch
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        log(f'Training complete; loaded final checkpoint with {len(weights)} tensors: {checkpoint}')
        del weights
        env = os.environ.copy()
        env.update(CHECKPOINT=str(checkpoint), ROUTES=str(routes),
                   EVALUATION_OUTPUT_DIR=str(output), GPU_ID='0', CARLA_PORT='8850',
                   TM_PORT='8002', EVAL_TIMEOUT='120', CARLA_RENDER_OFFSCREEN='1',
                   CARLA_RELOAD_EACH_ROUTE='0', FMSR_DIAGNOSTICS='0',
                   PYTHON_BIN=sys.executable, PYTHONUNBUFFERED='1')
        env.pop('PREFLIGHT_ONLY', None)
        status('evaluation_preflight')
        with (output / 'preflight.log').open('ab') as log_file:
            result = subprocess.run(['bash', str(ROOT / 'scripts/eval_bench2drive_pdm_lite_fmsr.sh')],
                                    cwd=ROOT, env=dict(env, PREFLIGHT_ONLY='1'),
                                    stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode:
            status('evaluation_preflight_failed', returncode=result.returncode)
            return result.returncode
        status('evaluating')
        log('Starting 96-route evaluation with automatic checkpoint resume.')
        with (output / 'watchdog.log').open('ab') as log_file:
            child = subprocess.Popen([sys.executable, str(ROOT / 'scripts/pdm_lite/watch_evaluation.py'),
                                      '--output-dir', str(output), '--routes', str(routes),
                                      '--poll-seconds', '10', '--retry-delay', '30'],
                                     cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log_file, stderr=subprocess.STDOUT)
            while child.poll() is None:
                try:
                    data = read_checkpoint(output / 'checkpoint_endpoint.json', ids)
                    progress = data['_checkpoint']['progress']
                except (OSError, ValueError, KeyError, TypeError):
                    progress = None
                status('evaluating', watchdog_pid=child.pid, progress=progress)
                time.sleep(args.poll_seconds)
        try:
            data = read_checkpoint(output / 'checkpoint_endpoint.json', ids)
        except (OSError, ValueError, KeyError, TypeError):
            data = None
        if completed(data, len(ids)):
            record = data['_checkpoint']['global_record']
            atomic_write(output / 'evaluation_summary.json', dict(
                checkpoint=str(checkpoint), routes=str(routes),
                progress=[96, 96], global_record=record))
            status('completed', progress=[96, 96], scores=record.get('scores'))
            log('All 96 routes evaluated; global statistics and summary saved.')
            return 0
        status('evaluation_needs_attention', returncode=child.returncode)
        log('Evaluation supervisor stopped; inspect watchdog.log before resuming.')
        return child.returncode or 1


if __name__ == '__main__':
    raise SystemExit(main())
