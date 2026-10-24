"""Wait for an exact completed training epoch, then supervise resumable evaluation."""
import argparse
from datetime import datetime
import fcntl
import hashlib
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
from .evaluation_watchdog import atomic_write, inspect_checkpoint, processes


def timestamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def log(message):
    print(f'[{timestamp()}] {message}', flush=True)


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def argument_value(arguments, name):
    for index, argument in enumerate(arguments):
        if argument.startswith(name+'='):
            return argument.split('=',1)[1]
        if argument == name and index+1 < len(arguments):
            return arguments[index+1]
    return None


def training_processes(table, run_dir):
    result = []
    for pid, proc in table.items():
        arguments = proc['arguments']
        output = argument_value(arguments, '--output-dir')
        if 'fmsrpdm.train' in arguments and output and Path(output).resolve() == run_dir:
            result.append(pid)
    return sorted(result)


def inspect_training(run_dir, epoch, table):
    """The optimizer/RNG companion is published after the model, both atomically."""
    run_dir = Path(run_dir).resolve()
    completed = []
    for path in run_dir.glob('epoch_*.json'):
        try:
            row = json.loads(path.read_text())
            if row.get('completed_full_epoch'):
                completed.append(row['epoch'])
        except (OSError, ValueError, KeyError):
            pass  # the training writer may be publishing an epoch report
    model = run_dir/f'model_{epoch:04d}.pth'
    resume = run_dir/f'resume_{epoch:04d}.pt'
    summary = run_dir/f'epoch_{epoch:04d}.json'
    generation = run_dir/f'generation_epoch_{epoch:04d}.json'
    active = training_processes(table, run_dir)
    state = {'target_epoch':epoch, 'completed_epochs':sorted(completed),
             'completed_epoch_count':len(completed), 'training_pids':active,
             'checkpoint':str(model), 'model_published':model.is_file(),
             'resume_published':resume.is_file(), 'ready':False}
    if not all(path.is_file() and path.stat().st_size for path in (model,resume,summary,generation)):
        state['reason'] = 'waiting_for_completed_epoch_and_atomic_checkpoint_files'
        return state
    try:
        row = json.loads(summary.read_text())
        generated = json.loads(generation.read_text())
    except (OSError, ValueError) as error:
        state['reason'] = 'waiting_for_readable_epoch_reports'
        state['error'] = str(error)
        return state
    if row.get('epoch') != epoch or not row.get('completed_full_epoch'):
        state['reason'] = 'target_epoch_is_not_a_completed_full_epoch'
    elif not generated.get('samples') or generated.get('scope') != 'predicted_anchor_full_sampling_no_label_input':
        state['reason'] = 'target_generation_validation_is_not_complete'
    elif active:
        state['reason'] = 'waiting_for_training_processes_to_release_gpu'
    else:
        state.update(ready=True, reason='target_epoch_and_generation_validation_complete',
                     global_step=row['global_step'])
    return state


def verify_training_completion(run_dir, epoch, job):
    """Read only checkpoints produced by this pinned local training run."""
    import torch
    for name,key in (('run_config.json','run_config_sha256'),('config.json','config_sha256')):
        if fingerprint(run_dir/name) != job[key]:
            raise ValueError(f'Training configuration changed after scheduling: {name}')
    provenance = json.loads((run_dir/'provenance.json').read_text())
    summary = json.loads((run_dir/f'epoch_{epoch:04d}.json').read_text())
    companion = torch.load(run_dir/f'resume_{epoch:04d}.pt', map_location='cpu', weights_only=False)
    if (companion['epoch'] != epoch or companion['global_step'] != summary['global_step']
            or companion['config_fingerprint'] != provenance['configuration_fingerprint']
            or companion['asset_fingerprint'] != provenance['asset_fingerprint']):
        raise ValueError('Final checkpoint companion does not match the completed epoch/configuration/assets')
    return {'epoch':epoch, 'global_step':companion['global_step'],
            'config_fingerprint':companion['config_fingerprint'],
            'asset_fingerprint':companion['asset_fingerprint'], 'verified_at':timestamp()}


def lock_is_held(path):
    with Path(path).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def job_settings(args):
    run = json.loads((args.run_dir/'run_config.json').read_text())
    if args.epoch < 0 or run['epochs'] <= args.epoch:
        raise ValueError('The target epoch is outside the configured training run')
    if not run.get('generation_validation_samples'):
        raise ValueError('Generation validation must be enabled for automatic evaluation')
    elements = ET.parse(args.routes).getroot().findall('route')
    ids = [f"RouteScenario_{row.get('id')}_rep0" for row in elements]
    if len(ids) != args.route_count or len(set(ids)) != len(ids):
        raise ValueError(f'Expected exactly {args.route_count} distinct routes')
    settings = {'run_dir':str(args.run_dir), 'target_epoch':args.epoch,
            'checkpoint':str(args.run_dir/f'model_{args.epoch:04d}.pth'),
            'run_config_sha256':fingerprint(args.run_dir/'run_config.json'),
            'config_sha256':fingerprint(args.run_dir/'config.json'),
            'routes':str(args.routes), 'routes_sha256':fingerprint(args.routes),
            'route_count':args.route_count, 'route_ids':ids, 'gpu':args.gpu,
            'carla_port':args.carla_port, 'tm_port':args.tm_port, 'eval_timeout':args.eval_timeout,
            'poll_seconds':args.poll_seconds, 'retry_seconds':args.retry_seconds,
            'stale_seconds':args.stale_seconds, 'heartbeat_seconds':args.heartbeat_seconds,
            'rpc_timeout':args.rpc_timeout, 'prediction_base':run.get('prediction_base','noisy_residual')}
    if getattr(args, 'record_analysis_metrics', False):
        settings['analysis_recording'] = {'trace_every': 1, 'probe_every': 200, 'probe_max': 4,
                                         'vehicle_state': 'every_agent_tick',
                                         'comfort_export_interval_seconds': 0.1}
    return settings


def summarize_analysis(output):
    """Derive reports only after the evaluator releases its result/telemetry lock."""
    analysis = output/'analysis'
    analysis.mkdir(exist_ok=True)
    commands = [
        [sys.executable, '-m', 'scripts.pdm_lite.summarize_eval_metrics',
         '--result', str(output/'checkpoint_endpoint.json'), '--metric-dir', str(output/'eval_metrics'),
         '--output', str(analysis/'benchmark_metrics.json'), '--rebuild'],
        [sys.executable, '-m', 'scripts.pdm_lite.analyze_planning_metrics',
         '--metric-dir', str(output/'eval_metrics'), '--output', str(analysis/'planning_metrics.json')]]
    with (analysis/'analysis.log').open('a') as handle:
        for command in commands:
            subprocess.run(command, cwd=WORKSPACE_ROOT, stdin=subprocess.DEVNULL,
                           stdout=handle, stderr=subprocess.STDOUT, check=True, timeout=1800)
    return {'benchmark_report': str(analysis/'benchmark_metrics.json'),
            'planning_report': str(analysis/'planning_metrics.json'), 'completed_at': timestamp()}


def run_job(args):
    args.run_dir, args.routes, args.output = (Path(p).resolve() for p in (args.run_dir,args.routes,args.output))
    job = job_settings(args)
    output = args.output
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.train_then_evaluate.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            log('The scheduled evaluation is already monitored; no duplicate starts')
            return
        settings = output/'auto_eval_job.json'
        if settings.exists() and json.loads(settings.read_text()) != job:
            raise ValueError('This output directory belongs to a different scheduled evaluation')
        if (output/'checkpoint_endpoint.json').exists() and not settings.exists():
            raise ValueError('Choose a new output directory; existing results are not from this scheduled job')
        atomic_write(settings,(json.dumps(job,indent=2)+'\n').encode())
        atomic_write(output/'auto_eval.pid',f'{os.getpid()}\n'.encode())
        stop = [False]
        def stop_job(signum,frame):stop[0]=True
        signal.signal(signal.SIGTERM,stop_job)
        signal.signal(signal.SIGINT,stop_job)
        prepared = False
        child = None
        preflight_attempts = watchdog_starts = 0
        next_attempt = 0.
        last_status = None
        environment = os.environ.copy()
        environment.update(CHECKPOINT=job['checkpoint'],ROUTES=job['routes'],
            EVALUATION_OUTPUT_DIR=str(output),GPU_ID=str(args.gpu),CARLA_PORT=str(args.carla_port),
            TM_PORT=str(args.tm_port),EVAL_TIMEOUT=str(args.eval_timeout),
            ALLOW_EXPERIMENTAL_CLEAN_PRIOR='1' if job['prediction_base']=='tfpp_clean_prior_v1' else '0',
            VERIFY_EVAL_MODEL='1',PREFLIGHT_ONLY='1',PYTHON_BIN=sys.executable)
        if 'analysis_recording' in job:
            environment.update(EVAL_CAPTURE_ALL_METRICS='1', EVAL_METRIC_ROOT=str(output),
                               EVAL_TRACE_EVERY='1', EVAL_PROBE_EVERY='200', EVAL_PROBE_MAX='4')
        log(f'Waiting for completed epoch {args.epoch:02d} in {args.run_dir}; evaluation={output}')
        with (output/'auto_eval_preflight.log').open('a',buffering=1) as preflight_log, \
                (output/'watchdog.log').open('a',buffering=1) as watchdog_log:
            while not stop[0]:
                training = inspect_training(args.run_dir,args.epoch,processes())
                state = {'monitor_pid':os.getpid(),'updated_at':timestamp(),'training':training,
                         'target_epoch':args.epoch,'evaluation_output':str(output),
                         'preflight_attempts':preflight_attempts,'watchdog_starts':watchdog_starts}
                if not training['ready']:
                    message = 'waiting_for_training'
                elif time.monotonic() < next_attempt:
                    message = 'waiting_to_retry_preflight_or_watchdog'
                else:
                    try:
                        if fingerprint(args.routes) != job['routes_sha256']:
                            raise ValueError('Route XML changed after scheduling')
                        if not prepared:
                            verified = verify_training_completion(args.run_dir,args.epoch,job)
                            atomic_write(output/'training_completion.json',(json.dumps(verified,indent=2)+'\n').encode())
                            preflight_attempts += 1
                            log('Training finished; verifying final weights, agent import and evaluation staging')
                            result = subprocess.run(['bash',str(PACKAGE_ROOT/'scripts/evaluate.sh')],
                                cwd=WORKSPACE_ROOT,env=environment,stdin=subprocess.DEVNULL,
                                stdout=preflight_log,stderr=subprocess.STDOUT,timeout=180)
                            if result.returncode:
                                raise RuntimeError(f'Evaluation preflight failed: exit {result.returncode}')
                            source = json.loads((output/'source.json').read_text())
                            if (source['checkpoint'] != job['checkpoint'] or source['route_count'] != args.route_count
                                    or source['config_sha256'] != job['config_sha256']
                                    or source['routes_sha256'] != job['routes_sha256']
                                    or source.get('analysis_recording') != job.get('analysis_recording')):
                                raise ValueError('Evaluation staging differs from the pinned scheduled job')
                            prepared = True
                            log('Final checkpoint verified; automatic resumable evaluation is enabled')
                        try:
                            evaluation,_ = inspect_checkpoint(output/'checkpoint_endpoint.json',job['route_ids'])
                        except (OSError,ValueError,KeyError,TypeError) as error:
                            # The watchdog owns recovery after a killed writer.
                            # A corrupt endpoint must not prevent restarting it.
                            evaluation={'complete':False,'checkpoint_error':str(error)}
                        state['evaluation'] = evaluation
                        if evaluation['complete']:
                            if 'analysis_recording' in job:
                                if lock_is_held(output/'.evaluation.lock'):
                                    state['monitor_status'] = 'waiting_for_final_telemetry_flush'
                                    atomic_write(output/'auto_eval_status.json',
                                                 (json.dumps(state,indent=2)+'\n').encode())
                                    time.sleep(min(args.poll_seconds,60.))
                                    continue
                                state['monitor_status'] = 'summarizing_analysis_metrics'
                                atomic_write(output/'auto_eval_status.json',
                                             (json.dumps(state,indent=2)+'\n').encode())
                                state['analysis'] = summarize_analysis(output)
                            state.update(monitor_status='complete',preflight_attempts=preflight_attempts,
                                         watchdog_starts=watchdog_starts)
                            atomic_write(output/'auto_eval_status.json',(json.dumps(state,indent=2)+'\n').encode())
                            log(f'All {args.route_count} routes scored; scheduled job complete')
                            return
                        if child is not None and child.poll() is not None:
                            state['watchdog_exit_code'] = child.returncode
                            child = None
                            next_attempt = time.monotonic()+args.retry_seconds
                            log('Evaluation watchdog exited early; it will be restarted')
                        elif child is None and not lock_is_held(output/'.evaluation_watchdog.lock'):
                            command = [sys.executable,'-u','-m','fmsrpdm.evaluation_watchdog',
                                '--output',str(output),'--gpu',str(args.gpu),
                                '--carla-port',str(args.carla_port),'--tm-port',str(args.tm_port),
                                '--poll-seconds',str(args.poll_seconds),'--retry-seconds',str(args.retry_seconds),
                                '--stale-seconds',str(args.stale_seconds),'--heartbeat-seconds',str(args.heartbeat_seconds),
                                '--rpc-timeout',str(args.rpc_timeout),'--eval-timeout',str(args.eval_timeout)]
                            child = subprocess.Popen(command,cwd=WORKSPACE_ROOT,env=environment,
                                stdin=subprocess.DEVNULL,stdout=watchdog_log,stderr=subprocess.STDOUT,start_new_session=True)
                            watchdog_starts += 1
                            log(f'Evaluation watchdog started: PID {child.pid}')
                        if (output/'watchdog_status.json').exists():
                            try:state['watchdog']=json.loads((output/'watchdog_status.json').read_text())
                            except (OSError,ValueError):pass
                        message = 'evaluation_supervised'
                    except Exception as error:
                        state['error'] = f'{type(error).__name__}: {error}'
                        next_attempt = time.monotonic()+args.retry_seconds
                        message = 'retry_after_setup_or_monitor_error'
                        log(state['error'])
                state.update(monitor_status=message,preflight_attempts=preflight_attempts,watchdog_starts=watchdog_starts)
                atomic_write(output/'auto_eval_status.json',(json.dumps(state,indent=2)+'\n').encode())
                signature = (message,training['completed_epoch_count'],bool(training['training_pids']))
                if signature != last_status:
                    log(f'{message}; completed_epochs={training["completed_epoch_count"]}; target_epoch={args.epoch:02d}')
                    last_status = signature
                time.sleep(min(args.poll_seconds,60.))
        atomic_write(output/'auto_eval_status.json',(json.dumps({'monitor_status':'stopped',
            'monitor_pid':os.getpid(),'updated_at':timestamp()},indent=2)+'\n').encode())
        log('Scheduled monitor stopped; any already running evaluation watchdog remains independent')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--routes',type=Path,default=WORKSPACE_ROOT/'3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml')
    parser.add_argument('--epoch',type=int,default=9)
    parser.add_argument('--route-count',type=int,default=220)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--carla-port',type=int,default=8870)
    parser.add_argument('--tm-port',type=int,default=8040)
    parser.add_argument('--poll-seconds',type=float,default=15.)
    parser.add_argument('--retry-seconds',type=float,default=30.)
    parser.add_argument('--stale-seconds',type=float,default=900.)
    parser.add_argument('--heartbeat-seconds',type=float,default=30.)
    parser.add_argument('--rpc-timeout',type=float,default=5.)
    parser.add_argument('--eval-timeout',type=float,default=120.)
    parser.add_argument('--background',action='store_true')
    parser.add_argument('--record-analysis-metrics', action='store_true',
                        help='Record every-tick telemetry and integration traces, bounded probes; summarize on completion')
    args=parser.parse_args()
    if min(args.poll_seconds,args.heartbeat_seconds,args.rpc_timeout,args.eval_timeout) <= 0 or min(args.retry_seconds,args.stale_seconds) < 0:
        parser.error('Invalid polling/retry/timeout settings')
    args.run_dir,args.routes,args.output=(p.resolve() for p in (args.run_dir,args.routes,args.output))
    job_settings(args)  # report obvious configuration errors before detaching
    if args.background:
        args.output.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','fmsrpdm.train_then_evaluate']
        command += [a for a in sys.argv[1:] if a!='--background']
        with (args.output/'auto_eval.log').open('a') as handle:
            child=subprocess.Popen(command,cwd=WORKSPACE_ROOT,stdin=subprocess.DEVNULL,
                stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=child.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            log(f'Scheduled monitor PID {child.pid}; status={args.output/"auto_eval_status.json"}')
        else:
            if code:raise SystemExit(f'Scheduled monitor exited with code {code}; inspect auto_eval.log')
            log('Scheduled job complete or already monitored; inspect auto_eval_status.json')
        return
    run_job(args)


if __name__=='__main__':main()
