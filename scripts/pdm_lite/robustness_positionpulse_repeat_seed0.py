"""Five fresh seed-0 rounds of the lateral localization pulse, ten jobs total."""
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
OUTPUT = suite.ROOT / 'outputs/local_evaluation/robustness_positionpulse_5rounds_seed0_20261016'
DATA = Path('/media/vcdc/T71/fmsr_robustness_positionpulse_5rounds_seed0_20261016')
CONDITIONS = {'pulse_position_lateral_0.25'}
DEPENDENCY = suite.ROOT / 'outputs/local_evaluation/robustness_statepulse_5rounds_seed0_20261016'


def prepare():
    assert OUTPUT.resolve() != SOURCE.resolve()
    original = json.loads((SOURCE / 'campaign.json').read_text())
    manifest = copy.deepcopy(original)
    jobs = [j for j in manifest['jobs'] if j['phase'] == 'pilot'
            and j['seed'] == 0 and j['perturbation']['id'] in CONDITIONS]
    assert len(jobs) == 2
    assert {(j['perturbation']['id'], j['model']) for j in jobs} == {
        (c, m) for c in CONDITIONS for m in suite.MODELS}
    route_path = OUTPUT / 'routes/pilot.xml'
    payload = Path(jobs[0]['routes']).read_bytes()
    assert len(ET.fromstring(payload).findall('route')) == 44
    for job in jobs:
        assert Path(job['routes']).read_bytes() == payload
        job['routes'] = str(route_path)
    templates = jobs
    jobs = []
    for round_index in range(1, 6):
        phase = f'repeat{round_index:02d}'
        for template in templates:
            job = copy.deepcopy(template)
            job.update(id=f'{phase}/pulse_position_lateral_0.25/seed0/{job["model"]}',
                       phase=phase, repeat_index=round_index)
            jobs.append(job)
    assert len(jobs) == 10 and len({job['id'] for job in jobs}) == 10
    for path, sha in manifest['pins'].items():
        if suite.digest(path) != sha:
            raise ValueError(f'Original pinned input changed: {path}')
    for path in (Path(__file__).resolve(), Path(suite.__file__).resolve(),
                 suite.ROOT / 'scripts/pdm_lite/robustness_analysis.py'):
        manifest['pins'][str(path)] = suite.digest(path)
    manifest.update(output=str(OUTPUT), data_root=str(DATA), jobs=jobs,
                    repeat_of=str(SOURCE), wait_for=str(DEPENDENCY),
                    repeat_scope='5 fresh rounds, 10 jobs, 440 route attempts; all seed0; no clean or full runs')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    route_path.parent.mkdir(exist_ok=True)
    if route_path.exists() and route_path.read_bytes() != payload:
        raise ValueError('Repeat routes changed')
    route_path.write_bytes(payload)
    target = OUTPUT / 'campaign.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Repeat manifest changed')
    suite.write(target, manifest)
    suite.write(OUTPUT / 'job_count.json', dict(jobs=10, counts={f'repeat{i:02d}': 2 for i in range(1, 6)}, route_attempts=440))
    (OUTPUT / 'README.md').write_text(
        '# 单次定位横向脉冲0.25m：seed0五轮复测\n\n'
        '5轮 × 44路线 × 2模型 = 440次路线测试，共10组。每轮重新运行，全部seed0。\n'
        '每轮使用独立repeat01至repeat05目录，依次运行FM、FM+CR。\n'
        '仅在规划状态脉冲五轮测试全部完成后自动启动；前序异常或暂停则继续等待。\n'
        '模型、44路线、5秒单次触发与原扰动协议保持一致；不新增clean或full。\n'
        '这是同种子重复波动测试，不是五个独立seed。原有结果保持不变。\n'
        '逐帧数据：'+str(DATA)+'；每组自动分析、无损压缩，基础设施故障自动恢复。\n'
        'analysis/run_metrics.csv按repeat轮次汇总，不计算新clean配对退化。\n'
        '续跑：`'+suite.PYTHON+' -m scripts.pdm_lite.robustness_positionpulse_repeat_seed0`。\n'
        '停止：在此目录创建STOP_AFTER_CURRENT，当前组完成后停止。\n')
    return manifest


def dependency_complete():
    try:
        manifest = json.loads((DEPENDENCY / 'campaign.json').read_text())
        status = json.loads((DEPENDENCY / 'status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (status.get('state') == 'complete' and len(manifest['jobs']) == 10
            and all((DEPENDENCY / 'runs' / job['id'] / 'complete.json').exists()
                    for job in manifest['jobs']))


def wait_for_dependency():
    while not dependency_complete():
        if (OUTPUT / 'STOP_AFTER_CURRENT').exists():
            suite.write(OUTPUT / 'status.json', dict(state='stopped_before_start', updated_at=suite.stamp()))
            return False
        suite.write(OUTPUT / 'status.json', dict(
            state='waiting_for_previous_campaign', wait_for=str(DEPENDENCY),
            total_jobs=10, completed_jobs=0, updated_at=suite.stamp()))
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
                [suite.PYTHON, '-u', '-m', 'scripts.pdm_lite.robustness_positionpulse_repeat_seed0', '--foreground'],
                cwd=suite.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(f'Campaign PID {process.pid}; {OUTPUT}/status.json', flush=True)


if __name__ == '__main__':
    main()
