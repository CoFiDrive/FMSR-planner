"""Two paired seed0 repeats per condition: localization 0.5m, heading 5deg, features 0.03."""
import argparse
import copy
import json
import time
import fcntl
import threading
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.pdm_lite import robustness_suite as suite

SOURCE = suite.DEFAULT
OUTPUT = suite.ROOT / 'outputs/local_evaluation/robustness_inputs5_4groups_seed0_20261018'
DATA = Path('/media/vcdc/T71/fmsr_robustness_inputs5_4groups_seed0_20261018')
CONDITIONS = {'position_lateral_0.5', 'position_longitudinal_0.5', 'heading_5',
              'feature_query_0.03', 'feature_bev_0.03'}
TEMPLATE_IDS = (CONDITIONS - {'heading_5'}) | {'heading_1'}
DEPENDENCY = suite.ROOT / 'outputs/local_evaluation/robustness_positionlong01_seed0_20261017'


def prepare():
    assert OUTPUT.resolve() != SOURCE.resolve()
    original = json.loads((SOURCE / 'campaign.json').read_text())
    manifest = copy.deepcopy(original)
    jobs = [j for j in manifest['jobs'] if j['phase'] == 'full'
            and j['seed'] == 0 and j['perturbation']['id'] in TEMPLATE_IDS]
    assert len(jobs) == 10
    assert {(j['perturbation']['id'], j['model']) for j in jobs} == {
        (c, m) for c in TEMPLATE_IDS for m in suite.MODELS}
    route_path = OUTPUT / 'routes/pilot.xml'
    payload = (SOURCE / 'routes/pilot.xml').read_bytes()
    assert len(ET.fromstring(payload).findall('route')) == 44
    for job in jobs:
        job['routes'] = str(route_path)
    for job in jobs:
        if job['perturbation']['id'] == 'heading_1':
            job['perturbation'].update(id='heading_5', amplitude=5.)
    templates = {(job['perturbation']['id'], job['model']): job for job in jobs}
    jobs = []
    for condition in ['position_lateral_0.5', 'position_longitudinal_0.5', 'heading_5',
                      'feature_query_0.03', 'feature_bev_0.03']:
        for repeat_index in range(1, 3):
            phase = f'repeat{repeat_index:02d}'
            for model in suite.MODELS:
                job = copy.deepcopy(templates[condition, model])
                job.update(id=f'{phase}/{condition}/seed0/{model}', phase=phase,
                           repeat_index=repeat_index)
                jobs.append(job)
    assert len(jobs) == 20 and len({job['id'] for job in jobs}) == 20
    for path, sha in manifest['pins'].items():
        if suite.digest(path) != sha:
            raise ValueError(f'Original pinned input changed: {path}')
    for path in (Path(__file__).resolve(), Path(suite.__file__).resolve(),
                 suite.ROOT / 'scripts/pdm_lite/robustness_analysis.py'):
        manifest['pins'][str(path)] = suite.digest(path)
    manifest.update(output=str(OUTPUT), data_root=str(DATA), jobs=jobs,
                    repeat_of=str(SOURCE), wait_for=str(DEPENDENCY),
                    repeat_scope='5 conditions, 4 jobs each (2 FM and 2 FM_CR repeats), 20 jobs, 880 route attempts; seed0; heading extended to 5 degrees')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    route_path.parent.mkdir(exist_ok=True)
    if route_path.exists() and route_path.read_bytes() != payload:
        raise ValueError('Repeat routes changed')
    route_path.write_bytes(payload)
    target = OUTPUT / 'campaign.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Repeat manifest changed')
    suite.write(target, manifest)
    suite.write(OUTPUT / 'job_count.json', dict(jobs=20, counts=dict(repeat01=10, repeat02=10), route_attempts=880))
    (OUTPUT / 'README.md').write_text(
        '# 五条件各4组：seed0自动实验\n\n'
        '持续定位横向0.5m、纵向0.5m、航向5度、Query噪声0.03、BEV噪声0.03。\n'
        '每条件FM、FM+CR各重复2次，每组44条路线，共20组、880次路线测试，全部seed0。\n'
        '各组独立记录指标与逐帧数据，自动分析、无损压缩，基础设施故障自动恢复。\n'
        '定位和特征扰动沿用原协议；航向沿用heading协议，幅度设为5度；全部从2秒开始持续注入。\n'
        'monitor.log每30秒记录组数、路线进度、当前任务、预计结束时间。\n'
        '原始逐帧数据：'+str(DATA)+'。\n'
        '续跑：`'+suite.PYTHON+' -m scripts.pdm_lite.robustness_inputs5_4groups_seed0`。\n'
        '停止：在此目录创建STOP_AFTER_CURRENT，当前组完成后停止。\n')
    return manifest


def dependency_complete():
    try:
        manifest = json.loads((DEPENDENCY / 'campaign.json').read_text())
        status = json.loads((DEPENDENCY / 'status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (status.get('state') == 'complete' and len(manifest['jobs']) == 2
            and all((DEPENDENCY / 'runs' / job['id'] / 'complete.json').exists()
                    for job in manifest['jobs']))


def wait_for_dependency():
    while not dependency_complete():
        if (OUTPUT / 'STOP_AFTER_CURRENT').exists():
            suite.write(OUTPUT / 'status.json', dict(state='stopped_before_start', updated_at=suite.stamp()))
            return False
        suite.write(OUTPUT / 'status.json', dict(
            state='waiting_for_previous_campaign', wait_for=str(DEPENDENCY),
            total_jobs=20, completed_jobs=0, updated_at=suite.stamp()))
        time.sleep(30)
    return True


def monitor(manifest, stop):
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    with (OUTPUT / 'monitor.log').open('a', buffering=1) as log:
        while True:
            try:
                now = datetime.now(ZoneInfo('Asia/Shanghai'))
                path = OUTPUT / 'status.json'
                status = json.loads(path.read_text()) if path.exists() else {}
                complete = [j for j in manifest['jobs']
                            if (OUTPUT / 'runs' / j['id'] / 'complete.json').exists()]
                routes = len(complete) * 44
                job = status.get('job')
                progress = [0, 44]
                if job and not (OUTPUT / 'runs' / job / 'complete.json').exists():
                    watch = OUTPUT / 'runs' / job / 'watchdog_status.json'
                    if watch.exists():
                        progress = json.loads(watch.read_text()).get('progress', progress)
                        routes += progress[0]
                eta = 'pending'
                pid_path = OUTPUT / 'pid.json'
                if routes and pid_path.exists():
                    start = datetime.fromisoformat(json.loads(pid_path.read_text())['started_at'])
                    elapsed = (now - start).total_seconds()
                    eta = (now + timedelta(seconds=max(0, 880-routes)*elapsed/routes)).isoformat(timespec='seconds')
                log.write(f'{now.isoformat(timespec="seconds")} state={status.get("state", "starting")} '
                          f'jobs={len(complete)}/20 routes={routes}/880 '
                          f'job={job} job_progress={progress} ETA={eta}\n')
            except Exception as error:
                log.write(f'{suite.stamp()} monitor_read_error={error!r}\n')
            if stop.wait(30):
                break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--foreground', action='store_true')
    args = parser.parse_args()
    manifest = prepare()
    if args.prepare_only:
        print((OUTPUT / 'job_count.json').read_text())
        return
    if args.foreground:
        queue_lock = (OUTPUT / '.queue.lock').open('a')
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        suite.write(OUTPUT / 'queue_pid.json', dict(pid=__import__('os').getpid(), started_at=suite.stamp()))
        try:
            if not wait_for_dependency():
                return
            DATA.mkdir(parents=True, exist_ok=True)
            monitor_stop = threading.Event()
            monitor_thread = threading.Thread(target=monitor, args=(manifest, monitor_stop), daemon=True)
            monitor_thread.start()
            try:
                suite.run(manifest)
            finally:
                monitor_stop.set()
                monitor_thread.join(timeout=5)
                with (OUTPUT / 'monitor.log').open('a') as log:
                    log.write(f'{suite.stamp()} campaign_run_returned; final_status=' +
                              (OUTPUT / 'status.json').read_text().replace('\n', ' ') + '\n')
        except Exception as error:
            suite.write(OUTPUT / 'error.json', dict(error=repr(error), updated_at=suite.stamp()))
            path = OUTPUT / 'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            status.update(state='error', error=repr(error), updated_at=suite.stamp())
            suite.write(path, status)
            raise
    else:
        with (OUTPUT / 'campaign.log').open('a') as log:
            process = subprocess.Popen(
                [suite.PYTHON, '-u', '-m', 'scripts.pdm_lite.robustness_inputs5_4groups_seed0', '--foreground'],
                cwd=suite.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(f'Campaign PID {process.pid}; {OUTPUT}/status.json', flush=True)


if __name__ == '__main__':
    main()
