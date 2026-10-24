"""Run only the eight explicitly requested seed-0 repeat jobs, independently."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from scripts.pdm_lite import robustness_suite as suite

SOURCE = suite.DEFAULT
OUTPUT = suite.ROOT / 'outputs/local_evaluation/robustness_repeat4_seed0_20261015'
DATA = Path('/media/vcdc/T71/fmsr_robustness_repeat4_seed0_20261015')
CONDITIONS = {
    'feature_bev_0.03', 'feature_query_0.03',
    'pulse_state_lateral_0.1', 'pulse_position_lateral_0.25',
}


def prepare():
    assert OUTPUT.resolve() != SOURCE.resolve()
    original = json.loads((SOURCE / 'campaign.json').read_text())
    manifest = copy.deepcopy(original)
    jobs = [j for j in manifest['jobs'] if j['phase'] == 'pilot'
            and j['seed'] == 0 and j['perturbation']['id'] in CONDITIONS]
    assert len(jobs) == 8
    assert {(j['perturbation']['id'], j['model']) for j in jobs} == {
        (c, m) for c in CONDITIONS for m in suite.MODELS}
    route_path = OUTPUT / 'routes/pilot.xml'
    payload = Path(jobs[0]['routes']).read_bytes()
    assert len(ET.fromstring(payload).findall('route')) == 44
    for job in jobs:
        assert Path(job['routes']).read_bytes() == payload
        job['routes'] = str(route_path)
    for path, sha in manifest['pins'].items():
        if suite.digest(path) != sha:
            raise ValueError(f'Original pinned input changed: {path}')
    for path in (Path(__file__).resolve(), Path(suite.__file__).resolve(),
                 suite.ROOT / 'scripts/pdm_lite/robustness_analysis.py'):
        manifest['pins'][str(path)] = suite.digest(path)
    manifest.update(output=str(OUTPUT), data_root=str(DATA), jobs=jobs,
                    repeat_of=str(SOURCE),
                    repeat_scope='8 fresh jobs, 352 route attempts; seed0 reproduction; no new clean or full runs')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    route_path.parent.mkdir(exist_ok=True)
    if route_path.exists() and route_path.read_bytes() != payload:
        raise ValueError('Repeat routes changed')
    route_path.write_bytes(payload)
    target = OUTPUT / 'campaign.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Repeat manifest changed')
    suite.write(target, manifest)
    suite.write(OUTPUT / 'job_count.json', dict(jobs=8, counts=dict(pilot=8), route_attempts=352))
    (OUTPUT / 'README.md').write_text(
        '# 四项扰动 seed0 独立复测\n\n'
        '4条件 × 44路线 × 2模型 = 352次路线测试，8组。全部重新运行，不复用旧扰动成绩。\n'
        '沿用原始模型、路线、seed0及扰动协议；不新增clean，不进入220路线full矩阵。\n'
        '这是同种子的复现测试，不增加独立随机种子。原始实验保留在 '+str(SOURCE)+'。\n'
        '逐帧数据目录：'+str(DATA)+'；每组自动分析、压缩归档，基础设施故障自动恢复。\n'
        '本轮汇总不计算新的clean配对退化；逐路线、行为与基准指标照常保存。\n'
        '续跑：`'+suite.PYTHON+' -m scripts.pdm_lite.robustness_repeat_seed0`。\n'
        '停止：在此目录创建STOP_AFTER_CURRENT，当前组完成后停止。\n')
    return manifest


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
        DATA.mkdir(parents=True, exist_ok=True)
        try:
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
                [suite.PYTHON, '-u', '-m', 'scripts.pdm_lite.robustness_repeat_seed0', '--foreground'],
                cwd=suite.ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        print(f'Campaign PID {process.pid}; {OUTPUT}/status.json', flush=True)


if __name__ == '__main__':
    main()
