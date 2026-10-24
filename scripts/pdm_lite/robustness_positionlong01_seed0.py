"""Longitudinal localization bias 0.10m, paired seed0 pilot after the 0.25m longitudinal campaign."""
import argparse
import copy
import json
import time
import fcntl
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.pdm_lite import robustness_suite as suite

SOURCE = suite.DEFAULT
OUTPUT = suite.ROOT / 'outputs/local_evaluation/robustness_positionlong01_seed0_20261017'
DATA = Path('/media/vcdc/T71/fmsr_robustness_positionlong01_seed0_20261017')
CONDITIONS = {'position_longitudinal_0.1'}
DEPENDENCY = suite.ROOT / 'outputs/local_evaluation/robustness_positionlong_seed0_20261017'


def prepare():
    assert OUTPUT.resolve() != SOURCE.resolve()
    original = json.loads((SOURCE / 'campaign.json').read_text())
    manifest = copy.deepcopy(original)
    jobs = [j for j in manifest['jobs'] if j['phase'] == 'full'
            and j['seed'] == 0 and j['perturbation']['id'] in CONDITIONS]
    assert len(jobs) == 2
    assert {(j['perturbation']['id'], j['model']) for j in jobs} == {
        (c, m) for c in CONDITIONS for m in suite.MODELS}
    route_path = OUTPUT / 'routes/pilot.xml'
    payload = (SOURCE / 'routes/pilot.xml').read_bytes()
    assert len(ET.fromstring(payload).findall('route')) == 44
    for job in jobs:
        job['routes'] = str(route_path)
    for job in jobs:
        job.update(id=f'repeat01/{job["perturbation"]["id"]}/seed0/{job["model"]}',
                   phase='repeat01', repeat_index=1)
    assert len(jobs) == 2 and len({job['id'] for job in jobs}) == 2
    for path, sha in manifest['pins'].items():
        if suite.digest(path) != sha:
            raise ValueError(f'Original pinned input changed: {path}')
    for path in (Path(__file__).resolve(), Path(suite.__file__).resolve(),
                 suite.ROOT / 'scripts/pdm_lite/robustness_analysis.py'):
        manifest['pins'][str(path)] = suite.digest(path)
    manifest.update(output=str(OUTPUT), data_root=str(DATA), jobs=jobs,
                    repeat_of=str(SOURCE), wait_for=str(DEPENDENCY),
                    repeat_scope='longitudinal localization bias 0.10m, 2 jobs, 88 route attempts, seed0, 44 pilot routes')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    route_path.parent.mkdir(exist_ok=True)
    if route_path.exists() and route_path.read_bytes() != payload:
        raise ValueError('Repeat routes changed')
    route_path.write_bytes(payload)
    target = OUTPUT / 'campaign.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Repeat manifest changed')
    suite.write(target, manifest)
    suite.write(OUTPUT / 'job_count.json', dict(jobs=2, counts=dict(repeat01=2), route_attempts=88))
    (OUTPUT / 'README.md').write_text(
        '# 持续定位纵向偏置0.10m：seed0配对实验\n\n'
        'FM与FM+CR各跑原pilot的44条路线，共2组、88次路线测试，全部seed0。\n'
        '从相对agent时间2秒起，在UKF后估计位置沿纵向注入0.10m持续偏置。\n'
        '使用原始已配置的纵向扰动定义，仅将路线范围设为与横向实验一致的44条。\n'
        '等待纵向0.25m测试全部完成后自动启动；前序异常或暂停则等待。\n'
        '各组独立保存指标和逐帧数据，自动分析、无损压缩、基础设施故障恢复。\n'
        '逐帧数据：'+str(DATA)+'；前序队列：'+str(DEPENDENCY)+'。\n'
        '续跑：`'+suite.PYTHON+' -m scripts.pdm_lite.robustness_positionlong01_seed0`。\n'
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
            total_jobs=2, completed_jobs=0, updated_at=suite.stamp()))
        time.sleep(30)
    return True


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
            suite.run(manifest)
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
                [suite.PYTHON, '-u', '-m', 'scripts.pdm_lite.robustness_positionlong01_seed0', '--foreground'],
                cwd=suite.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(f'Campaign PID {process.pid}; {OUTPUT}/status.json', flush=True)


if __name__ == '__main__':
    main()
