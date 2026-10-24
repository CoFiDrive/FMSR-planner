"""Pinned, resumable sequential CARLA robustness campaign with automatic analysis."""
import argparse
import copy
from datetime import datetime
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'outputs/local_evaluation/robustness_campaign_20261014'
PYTHON='/home/vcdc/anaconda3/envs/fmsr/bin/python'
LAUNCHER=ROOT/'scripts/eval_bench2drive_pdm_lite_fmsr.sh'
MODELS={'FM':ROOT/'outputs/local_training/flow_only_10epochs_seed0_20261008_143915/model_0009.pth',
        'FM_CR':ROOT/'outputs/local_training/tie_spectral_10epochs_seed0_20261006_223036/model_0009.pth'}


def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def conditions():
    result=[dict(id='clean',family='none',amplitude=0.,mode='continuous')]
    for axis in ('lateral','longitudinal'):
        for a in (.05,.10,.20):
            result.append(dict(id=f'state_{axis}_{a:g}',family='state',axis=axis,amplitude=a,tau=.4,mode='continuous',start_s=2.))
    for component in ('bev','query'):
        for a in (.01,.03,.05):
            result.append(dict(id=f'feature_{component}_{a:g}',family='feature',component=component,amplitude=a,mode='continuous',start_s=2.))
    for axis in ('lateral','longitudinal'):
        for a in (.10,.25,.50):
            result.append(dict(id=f'position_{axis}_{a:g}',family='position',axis=axis,amplitude=a,mode='continuous',start_s=2.))
    for a in (.5,1.,2.):
        result.append(dict(id=f'heading_{a:g}',family='heading',amplitude=a,mode='continuous',start_s=2.))
    result.extend([
        dict(id='pulse_state_lateral_0.1',family='state',axis='lateral',amplitude=.1,tau=.4,mode='pulse',start_s=5.),
        dict(id='pulse_feature_query_0.03',family='feature',component='query',amplitude=.03,mode='pulse',start_s=5.),
        dict(id='pulse_position_lateral_0.25',family='position',axis='lateral',amplitude=.25,mode='pulse',start_s=5.)])
    return result


def prepare(output,data_root):
    output.mkdir(parents=True,exist_ok=True);data_root.mkdir(parents=True,exist_ok=True)
    full=ROOT/'3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml'
    all_routes=ET.parse(full).getroot()
    # Select each scenario's first XML route; no outcomes are used in selecting pilots.
    selected={}
    for element in all_routes.findall('route'):
        scenario=element.find('scenarios/scenario')
        key=scenario.get('type') if scenario is not None else 'unknown'
        selected.setdefault(key,element)
    if len(selected)!=44:raise ValueError(f'Expected 44 scenario types, got {len(selected)}')
    subsets={'full':all_routes,'pilot':ET.Element(all_routes.tag,all_routes.attrib),
             'smoke':ET.Element(all_routes.tag,all_routes.attrib)}
    for element in selected.values():subsets['pilot'].append(copy.deepcopy(element))
    # Short parking-obstacle route, independent of model scores.
    smoke=next(e for e in all_routes.findall('route') if e.get('id')=='1773')
    subsets['smoke'].append(copy.deepcopy(smoke))
    route_files={}
    for name,xml in subsets.items():
        p=output/'routes'/f'{name}.xml';payload=ET.tostring(xml,encoding='utf-8',xml_declaration=True)
        p.parent.mkdir(exist_ok=True)
        if p.exists() and p.read_bytes()!=payload:raise ValueError('Pinned route subset changed')
        p.write_bytes(payload);route_files[name]=str(p)
    specs=conditions()
    smoke_ids={'clean','state_lateral_0.1','feature_bev_0.03','position_lateral_0.25','heading_1'}
    pilot_ids=smoke_ids|{'state_longitudinal_0.1','feature_query_0.03','pulse_state_lateral_0.1',
                         'pulse_position_lateral_0.25','pulse_feature_query_0.03'}
    jobs=[]
    for phase,seeds,accepted in [('smoke',[0],smoke_ids),('pilot',[0],pilot_ids),('full',[0,1,2],None)]:
        for spec in specs:
            if accepted is not None and spec['id'] not in accepted:continue
            for seed in seeds:
                for model in MODELS:
                    jobs.append(dict(id=f'{phase}/{spec["id"]}/seed{seed}/{model}',phase=phase,seed=seed,
                                     model=model,perturbation=spec,routes=route_files[phase]))
    pin_paths=list(MODELS.values())+[p.parent/'config.json' for p in MODELS.values()]
    for model,p in MODELS.items():
        cfg=json.loads((p.parent/'config.json').read_text())
        pin_paths.extend(Path(cfg[k]) for k in ('fmsr_anchor_path','fmsr_stats_path'))
    pin_paths += [full,LAUNCHER,ROOT/'3rd_party/carla_garage/team_code/sensor_agent.py',
                  ROOT/'scripts/pdm_lite/robustness_agent.py',ROOT/'scripts/pdm_lite/robustness_protocol.py']
    manifest=dict(version=1,output=str(output),data_root=str(data_root),models={k:str(v) for k,v in MODELS.items()},
                  pins={str(p):digest(p) for p in pin_paths},jobs=jobs,
                  seed_scope='Traffic Manager, model initialization RNG, and independent perturbation RNG; scenario-provider default seed remains 2000',
                  protocol=dict(fp32=True,tf32=False,full_speed_evolution=True,route_state_speed_not_injected=True,
                                state_sign='independent deterministic sign per 0.05s simulation tick',
                                feature_noise='independent RMS-normalized Gaussian per tensor and simulation tick',
                                localization='post-UKF estimate; heading/position used by navigation and historical lidar alignment',
                                pulse='one planning/tick after relative agent time >=5s, not an event-matched trigger',
                                retries='only infrastructure failures, original attempts archived; driving failures retained',
                                analysis='per-run benchmark and behavior; paired within-route baseline degradation; no failure-selected score merges'))
    target=output/'campaign.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise ValueError('Campaign inputs changed; use a new output')
    write(target,manifest)
    write(output/'job_count.json',dict(jobs=len(jobs),counts={phase:sum(j['phase']==phase for j in jobs) for phase in subsets},
                                     route_attempts=sum(len(ET.parse(j['routes']).getroot().findall('route')) for j in jobs)))
    return manifest


def environment(job,output):
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('SHADOW_','ROBUSTNESS_')) or key=='PREFLIGHT_ONLY':env.pop(key,None)
    env.update(CHECKPOINT=str(MODELS[job['model']]),ROUTES=job['routes'],EVALUATION_OUTPUT_DIR=str(output),
               EVALUATION_AGENT=str(ROOT/'scripts/pdm_lite/robustness_agent.py'),ROBUSTNESS_SPEC=str(output/'job.json'),
               PYTHON_BIN=PYTHON,GPU_ID='0',CARLA_PORT='8950',TM_PORT='8150',TM_SEED=str(job['seed']),SEED=str(job['seed']),
               EVAL_CAPTURE_ALL_METRICS='1',EVAL_METRIC_ROOT=str(output),EVAL_TRACE_EVERY='0',EVAL_PROBE_EVERY='0',EVAL_PROBE_MAX='0',
               FMSR_DIAGNOSTICS='0',CARLA_RELOAD_EACH_ROUTE='0',EVAL_TIMEOUT='120',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
    return env


def analysis(output):
    # Recover losslessly if a prior process stopped partway through archival.
    for path in (output/'eval_metrics').glob('*/*.gz'):
        if path.name.endswith(('.jsonl.gz','metric_info.json.gz')):
            target=path.with_suffix('')
            if not target.exists():
                tmp=target.with_name(target.name+'.tmp')
                with gzip.open(path,'rb') as source,tmp.open('wb') as dest:shutil.copyfileobj(source,dest)
                tmp.replace(target)
    commands=[['-m','scripts.pdm_lite.summarize_eval_metrics','--result',str(output/'checkpoint_endpoint.json'),
               '--metric-dir',str(output/'eval_metrics'),'--output',str(output/'analysis/benchmark_metrics.json'),'--rebuild'],
              ['-m','scripts.pdm_lite.robustness_analysis','--run',str(output)]]
    (output/'analysis').mkdir(exist_ok=True)
    with (output/'analysis/analysis.log').open('a') as log:
        for cmd in commands:subprocess.run([PYTHON,*cmd],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)


def archive(output):
    """Lossless gzip, verified by SHA256 before replacing uncompressed telemetry."""
    inventory=[]
    files=list((output/'eval_metrics').glob('*/*.jsonl'))+list((output/'eval_metrics').glob('*/metric_info.json'))
    for path in files:
        sha=digest(path);dest=path.with_name(path.name+'.gz');tmp=dest.with_name(dest.name+'.tmp')
        with path.open('rb') as source,gzip.open(tmp,'wb',compresslevel=6) as target:shutil.copyfileobj(source,target)
        h=hashlib.sha256()
        with gzip.open(tmp,'rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=sha:raise ValueError(f'Archive verification failed: {path}')
        tmp.replace(dest);inventory.append(dict(path=str(dest),uncompressed_sha256=sha,original_bytes=path.stat().st_size))
        path.unlink()
    write(output/'archive.json',dict(format='gzip; gunzip restores original bytes',files=inventory))


def active_other_evaluation():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=p.joinpath('cmdline').read_bytes().split(b'\0')
        except OSError:continue
        if any(a.endswith((b'/leaderboard_evaluator_v2.py',b'/CarlaUE4-Linux-Shipping')) for a in args):return True
    return False


def run(manifest):
    output=Path(manifest['output']);data=Path(manifest['data_root'])
    lock=(output/'.campaign.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stale_error=output/'error.json'
    if stale_error.exists():stale_error.rename(output/f'error_previous_{time.time_ns()}.json')
    sys.path.insert(0,str(ROOT/'fmsr_pdm_lite'))
    from fmsrpdm.evaluation_watchdog import inspect_checkpoint
    write(output/'pid.json',dict(pid=os.getpid(),started_at=stamp()))
    for path,sha in manifest['pins'].items():
        if digest(path)!=sha:raise ValueError(f'Pinned input changed: {path}')
    for index,job in enumerate(manifest['jobs']):
        if job['phase']=='full' and (output/'STOP_BEFORE_FULL').exists():
            write(output/'status.json',dict(state='pilot_complete',total_jobs=len(manifest['jobs']),
                  completed_jobs=sum((output/'runs'/j['id']/'complete.json').exists() for j in manifest['jobs']),
                  updated_at=stamp(),reason='STOP_BEFORE_FULL: minimal pilot scope completed'))
            return
        directory=output/'runs'/job['id'];directory.mkdir(parents=True,exist_ok=True)
        def status(state,**extra):
            write(output/'status.json',dict(state=state,job_index=index,total_jobs=len(manifest['jobs']),
                       job=job['id'],pid=os.getpid(),updated_at=stamp(),**extra))
        if (directory/'complete.json').exists():continue
        for path,sha in manifest['pins'].items():
            if digest(path)!=sha:raise ValueError(f'Pinned input changed before {job["id"]}: {path}')
        if (output/'STOP_AFTER_CURRENT').exists():status('stopped_between_jobs');return
        while shutil.disk_usage(data).free <20*1024**3 or shutil.disk_usage(output).free<2*1024**3:
            status('waiting_for_disk_space',required_data_free_gib=20,required_workspace_free_gib=2);time.sleep(30)
        while active_other_evaluation():status('waiting_for_other_evaluation');time.sleep(15)
        storage=data/job['id'];storage.mkdir(parents=True,exist_ok=True)
        link=directory/'eval_metrics'
        if not link.exists():link.symlink_to(storage,target_is_directory=True)
        write(directory/'job.json',job)
        env=environment(job,directory)
        status('preflight')
        with (directory/'preflight.log').open('a') as f:
            subprocess.run(['bash',str(LAUNCHER)],cwd=ROOT,env=dict(env,PREFLIGHT_ONLY='1'),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
        source=json.loads((directory/'agent/source.json').read_text())
        source.update(perturbation=job['perturbation'],seed=job['seed'],carla_client_timeout_seconds=120,
                      analysis_recording=dict(trace_every=0,probe_every=0,probe_max=0),
                      checkpoint_sha256=manifest['pins'][str(MODELS[job['model']])])
        write(directory/'source.json',source)
        route_ids=[f'RouteScenario_{r.get("id")}_rep0' for r in ET.parse(job['routes']).getroot().findall('route')]
        status('evaluating')
        watchdog=[PYTHON,'-u','-m','scripts.pdm_lite.robustness_suite','--watch-job',str(directory)]
        watch_env=dict(env,PYTHONPATH=str(ROOT/'fmsr_pdm_lite')+':'+str(ROOT))
        with (directory/'watchdog.log').open('a') as f:
            proc=subprocess.Popen(watchdog,cwd=ROOT,env=watch_env,stdout=f,stderr=subprocess.STDOUT)
            last_progress=-1;stalled_attempts=0;last_attempt=0
            while proc.poll() is None:
                try:
                    w=json.loads((directory/'watchdog_status.json').read_text())
                    progress=w.get('progress',[0,0]);attempt=w.get('attempts',0)
                    status('evaluating',progress=progress,attempts=attempt,watchdog_pid=proc.pid)
                    if attempt!=last_attempt:
                        stalled_attempts=stalled_attempts+1 if progress[0]==last_progress else 0
                        last_progress=progress[0];last_attempt=attempt
                    if stalled_attempts>=5:
                        proc.terminate();proc.wait(timeout=60)
                        raise RuntimeError('Five consecutive restart attempts without route progress; inspect watchdog.log')
                except (FileNotFoundError,json.JSONDecodeError):pass
                time.sleep(10)
        if proc.returncode!=0:raise RuntimeError(f'Watchdog failed: {proc.returncode}')
        state,_=inspect_checkpoint(directory/'checkpoint_endpoint.json',route_ids)
        if not state['complete']:raise RuntimeError(f'Incomplete evaluation: {state}')
        while active_other_evaluation():status('waiting_for_evaluator_flush');time.sleep(10)
        status('analyzing')
        analysis(directory)
        status('archiving')
        archive(directory)
        write(directory/'complete.json',dict(completed_at=stamp(),job=job['id']))
        subprocess.run([PYTHON,'-m','scripts.pdm_lite.robustness_analysis','--campaign',str(output)],cwd=ROOT,check=True)
        phase_jobs=[j for j in manifest['jobs'] if j['phase']==job['phase']]
        if all((output/'runs'/j['id']/'complete.json').exists() for j in phase_jobs):
            write(output/f'{job["phase"]}_validation.json',dict(state='complete',job_count=len(phase_jobs),
                    completed_at=stamp(),next_phase='pilot' if job['phase']=='smoke' else ('full' if job['phase']=='pilot' else None)))
        print(f'Completed {index+1}/{len(manifest["jobs"])}: {job["id"]}',flush=True)
    write(output/'status.json',dict(state='complete',total_jobs=len(manifest['jobs']),updated_at=stamp()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--data-root',type=Path,default=Path('/media/vcdc/T7/fmsr_robustness_campaign_20261014'))
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--foreground',action='store_true')
    parser.add_argument('--watch-job',type=Path)
    args=parser.parse_args()
    if args.watch_job:
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'fmsr_pdm_lite'))
        from fmsrpdm.evaluation_watchdog import watch
        watch(SimpleNamespace(output=args.watch_job,gpu=0,carla_port=8950,tm_port=8150,
                              poll_seconds=10,retry_seconds=15,stale_seconds=600,heartbeat_seconds=30,
                              rpc_timeout=5,eval_timeout=120,launcher=LAUNCHER))
        return
    manifest=prepare(args.output.resolve(),args.data_root.resolve())
    if args.prepare_only:print(json.dumps(json.loads((args.output/'job_count.json').read_text())));return
    if args.foreground:
        try:run(manifest)
        except Exception as e:
            write(args.output/'error.json',dict(error=repr(e),updated_at=stamp()))
            status_path=args.output/'status.json'
            current=json.loads(status_path.read_text()) if status_path.exists() else {}
            current.update(state='error',error=repr(e),updated_at=stamp());write(status_path,current)
            raise
    else:
        with (args.output/'campaign.log').open('a') as f:
            proc=subprocess.Popen([PYTHON,'-u','-m','scripts.pdm_lite.robustness_suite',
                   '--output',str(args.output),'--data-root',str(args.data_root),'--foreground'],cwd=ROOT,
                   stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Campaign PID {proc.pid}; {args.output}/status.json')


if __name__=='__main__':main()
