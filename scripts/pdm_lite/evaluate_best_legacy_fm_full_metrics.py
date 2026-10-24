"""Full 220-route reevaluation of the historical best FM checkpoint."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from fmsrpdm.evaluation_watchdog import (
    evaluation_is_locked, inspect_checkpoint, processes, updated_at, watch,
)
from fmsrpdm.train_then_evaluate import summarize_analysis
from scripts.pdm_lite.reevaluate_legacy_fm_cr_failures import (
    ROOT, LAUNCHER, environment, sha256, write_json,
)

OUTPUT = ROOT/'outputs/local_evaluation/bench2drive220_fm_best_full_metrics_20261012'
DATA = Path('/media/vcdc/T7/fmsr_evaluation_metrics/bench2drive220_fm_best_full_metrics_20261012')
CHECKPOINT = ROOT/'outputs/local_training/flow_only_10epochs_seed0_20261008_143915/model_0009.pth'


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = CHECKPOINT.resolve(strict=True)
    config_path = checkpoint.parent/'config.json'
    config = json.loads(config_path.read_text())
    if config['fmsr_trajectory_head'] != 'flow_only' or config['flow_contract_enable']:
        raise ValueError('Expected the historical FM-only model_0009 checkpoint')
    routes = ROOT/'3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml'
    ids = [f"RouteScenario_{r.get('id')}_rep0" for r in ET.parse(routes).getroot().findall('route')]
    if len(ids) != 220 or len(set(ids)) != 220:
        raise ValueError('Expected all 220 distinct Bench2Drive routes')
    assets = {name:dict(path=str(Path(config[name]).resolve(strict=True)),
                       sha256=sha256(Path(config[name])))
              for name in ['fmsr_anchor_path','fmsr_stats_path']}
    job = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
               config_sha256=sha256(config_path), routes=str(routes), routes_sha256=sha256(routes),
               route_ids=ids, route_count=220, gpu=args.gpu,
               carla_port=args.carla_port, tm_port=args.tm_port, assets=assets,
               data_root=str(args.data_root.resolve()),
               analysis_recording=dict(trace_every=1,probe_every=200,probe_max=4,
                                       vehicle_state='every_agent_tick',nearby_actors='every_agent_tick',
                                       comfort_export_interval_seconds=0.1,sensor_every=0))
    manifest = output/'full_evaluation_job.json'
    if manifest.exists() and json.loads(manifest.read_text()) != job:
        raise ValueError('Existing result directory uses different evaluation settings')
    write_json(manifest, job)
    args.data_root.mkdir(parents=True, exist_ok=True)
    for name in ('eval_metrics','diagnostics'):
        target = args.data_root.resolve()/name
        target.mkdir(exist_ok=True)
        link = output/name
        if link.is_symlink():
            if link.resolve() != target:
                raise ValueError(f'Data link changed: {link}')
        elif link.exists():
            raise ValueError(f'Existing data directory would mix evaluations: {link}')
        else:
            link.symlink_to(target,target_is_directory=True)
    if shutil.disk_usage(args.data_root).free < 20*1024**3:
        raise ValueError('Metric storage needs at least 20 GiB of free space')
    return job


def other_evaluation_active():
    return any(any(a.endswith('/leaderboard_evaluator_v2.py') or
                       a.endswith('/CarlaUE4-Linux-Shipping') for a in p['arguments'])
               for p in processes().values())


def run(args, job):
    output = args.output.resolve()
    with (output/'.full_metrics_monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('This full FM evaluation is already monitored.',flush=True)
            return
        def status(state, **extra):
            write_json(output/'full_evaluation_status.json',dict(state=state,pid=os.getpid(),
                       updated_at=updated_at(),route_count=220,**extra))
        env = environment(job,output)
        with (output/'preflight.log').open('a') as handle:
            subprocess.run(['bash',str(LAUNCHER)],cwd=ROOT,env=dict(env,PREFLIGHT_ONLY='1'),
                           stdout=handle,stderr=subprocess.STDOUT,check=True,timeout=180)
        source = json.loads((output/'agent/source.json').read_text())
        source.update(analysis_recording=job['analysis_recording'],carla_client_timeout_seconds=120,
                      checkpoint_sha256=job['checkpoint_sha256'],config_sha256=job['config_sha256'],
                      routes_sha256=job['routes_sha256'],metric_storage=job['data_root'])
        write_json(output/'source.json',source)
        while other_evaluation_active():
            # If reattaching to our own running evaluator, the watchdog follows it.
            if evaluation_is_locked(output):
                break
            status('waiting_for_other_evaluation')
            time.sleep(15)
        for path, digest in [(job['checkpoint'],job['checkpoint_sha256']),
                             (CHECKPOINT.parent/'config.json',job['config_sha256']),
                             (job['routes'],job['routes_sha256'])]:
            if sha256(path) != digest:
                raise ValueError(f'Pinned evaluation input changed: {path}')
        for asset in job['assets'].values():
            if sha256(asset['path']) != asset['sha256']:
                raise ValueError(f"Pinned asset changed: {asset['path']}")
        os.environ.update(env)
        status('evaluating')
        print(f"Starting full 220-route FM evaluation: {job['checkpoint']}",flush=True)
        watch(SimpleNamespace(output=output,gpu=job['gpu'],carla_port=job['carla_port'],
                              tm_port=job['tm_port'],poll_seconds=15,retry_seconds=30,
                              stale_seconds=900,heartbeat_seconds=30,rpc_timeout=5,
                              eval_timeout=120,launcher=LAUNCHER))
        while evaluation_is_locked(output):
            status('waiting_for_metrics_flush');time.sleep(15)
        state,_ = inspect_checkpoint(output/'checkpoint_endpoint.json',job['route_ids'])
        if not state['complete']:
            status('error',reason='Watchdog exited before completing all 220 routes')
            raise RuntimeError('Incomplete evaluation')
        # An analysis error must not rerun driving routes. Retry reports alone.
        while True:
            status('summarizing')
            try:
                reports = summarize_analysis(output)
                report = json.loads((output/'analysis/benchmark_metrics.json').read_text())
                if report['route_count']!=220 or not report['comfort_complete']:
                    raise ValueError('Full 220-route Comfort recording is incomplete')
            except Exception as error:
                status('retrying_analysis',error=f'{type(error).__name__}: {error}')
                print(f'Analysis retry: {error}',flush=True);time.sleep(30)
                continue
            status('complete',reports=reports,scores=report['global_scores_mean'])
            return


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--carla-port',type=int,default=8930)
    p.add_argument('--tm-port',type=int,default=8100)
    p.add_argument('--foreground',action='store_true')
    args=p.parse_args()
    job=prepare(args)
    if args.foreground:
        run(args,job)
        return
    command=[sys.executable,'-u','-m','scripts.pdm_lite.evaluate_best_legacy_fm_full_metrics']
    command+=sys.argv[1:]+['--foreground']
    with (args.output/'monitor.log').open('a') as handle:
        child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                               stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
    print(f'Full FM monitor PID {child.pid}; log={args.output}/monitor.log')


if __name__=='__main__':main()
