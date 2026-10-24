"""Aggregate all authorized robustness campaigns without selecting favorable repeats."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/analysis/robustness_complete_summary_20261019'
SOURCES = [
    'robustness_campaign_20261014', 'robustness_repeat4_seed0_20261015',
    'robustness_statepulse_5rounds_seed0_20261016',
    'robustness_positionpulse_5rounds_seed0_20261016',
    'robustness_inputs5_seed0_20261017', 'robustness_positionlong_seed0_20261017',
    'robustness_positionlong01_seed0_20261017',
    'robustness_inputs5_4groups_seed0_20261018',
]
LABELS = {
    'clean': '无扰动',
    'state_lateral_0.1': '持续规划状态横向0.10m',
    'state_longitudinal_0.1': '持续规划状态纵向0.10m',
    'pulse_state_lateral_0.1': '单次规划状态横向脉冲0.10m',
    'feature_bev_0.03': '持续BEV噪声0.03',
    'feature_query_0.03': '持续Query噪声0.03',
    'pulse_feature_query_0.03': '单次Query脉冲0.03',
    'position_lateral_0.25': '持续定位横向0.25m',
    'position_lateral_0.5': '持续定位横向0.50m',
    'position_longitudinal_0.1': '持续定位纵向0.10m',
    'position_longitudinal_0.25': '持续定位纵向0.25m',
    'position_longitudinal_0.5': '持续定位纵向0.50m',
    'pulse_position_lateral_0.25': '单次定位横向脉冲0.25m',
    'heading_1': '持续航向偏置1度', 'heading_5': '持续航向偏置5度',
}


def write_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten(obj, prefix=''):
    result = {}
    for key, value in obj.items():
        name = prefix + key
        if isinstance(value, dict):
            result.update(flatten(value, name + '__'))
        elif isinstance(value, (int, float)) and np.isfinite(value):
            result[name] = value
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, smoke, routes, behaviors, inventory, missing = [], [], [], [], [], []
    paired = defaultdict(dict)
    required = ['planning.jsonl', 'vehicle_state.jsonl', 'actors.jsonl',
                'perturbations.jsonl', 'metric_info.json', 'metadata.json', 'robustness_spec.json']
    for source in SOURCES:
        campaign = ROOT / 'outputs/local_evaluation' / source
        manifest = json.loads((campaign / 'campaign.json').read_text())
        metric_rows = {(r['phase'], r['condition'], r['model']): r
                       for r in csv.DictReader((campaign / 'analysis/run_metrics.csv').open())}
        for job in manifest['jobs']:
            directory = campaign / 'runs' / job['id']
            if not (directory / 'complete.json').exists():
                continue
            condition, model = job['perturbation']['id'], job['model']
            row = dict(campaign=source, job_id=job['id'], **metric_rows[job['phase'], condition, model])
            benchmark = json.loads((directory / 'analysis/benchmark_metrics.json').read_text())
            behavior = json.loads((directory / 'analysis/behavior_metrics.json').read_text())
            inventory.append(dict(campaign=source, job_id=job['id'],
                                  route_count=len(benchmark['routes']),
                                  completed_at=json.loads((directory / 'complete.json').read_text())['completed_at']))
            for route in benchmark['routes']:
                folder = directory / 'eval_metrics' / route['save_name']
                for filename in required:
                    if not (folder / filename).exists() and not (folder / (filename + '.gz')).exists():
                        missing.append(str(folder / filename))
            if job['phase'] == 'smoke':
                smoke.append(row)
                continue
            assert len(benchmark['routes']) == 44, job['id']
            runs.append(row)
            common = dict(campaign=source, job_id=job['id'], condition=condition,
                          phase=job['phase'], seed=job['seed'], model=model)
            for route in benchmark['routes']:
                record = dict(common, route_id=route['route_id'], scenario=route['scenario_name'],
                              DS=route['scores']['score_composed'],
                              RC=route['scores']['score_route'], IS=route['scores']['score_penalty'],
                              SR=int(route['route_success']), status=route['status'])
                record.update({k + '_events': len(v) for k, v in route['infractions'].items()})
                routes.append(record)
            for route in behavior['routes']:
                behaviors.append(dict(common, route_id=route['route_id'], **flatten(route)))
            paired[source, job['phase'], condition][model] = (row, benchmark)
    conditions = defaultdict(list)
    pair_rows, scenario_effects = [], defaultdict(lambda: defaultdict(list))
    for (source, phase, condition), data in paired.items():
        assert set(data) == {'FM', 'FM_CR'}, (source, phase, condition)
        conditions[condition].append(data)
        a, b = data['FM'][0], data['FM_CR'][0]
        pair_rows.append(dict(campaign=source, phase=phase, condition=condition,
                              FM_DS=float(a['DS']), FM_CR_DS=float(b['DS']),
                              DS_CR_minus_FM=float(b['DS']) - float(a['DS']),
                              SR_CR_minus_FM_pp=float(b['SR']) - float(a['SR'])))
        baseline = {r['route_id']: r for r in data['FM'][1]['routes']}
        for r in data['FM_CR'][1]['routes']:
            other = baseline[r['route_id']]
            scenario_effects[condition][r['scenario_name']].append(
                r['scores']['score_composed'] - other['scores']['score_composed'])
    summary, safety, metrics = [], [], []
    rng = np.random.default_rng(20261019)
    for condition in LABELS:
        repeats = conditions[condition]
        row = dict(condition=condition, label=LABELS[condition], paired_rounds=len(repeats),
                   routes_per_model=44 * len(repeats))
        for model in ['FM', 'FM_CR']:
            selected = [d[model][0] for d in repeats]
            for metric in ['DS', 'RC', 'IS', 'SR', 'Efficiency', 'Comfort']:
                values = np.array([float(r[metric]) for r in selected])
                for stat, value in [('mean', values.mean()), ('sd', values.std(ddof=1) if len(values) > 1 else None),
                                    ('min', values.min()), ('max', values.max())]:
                    metrics.append(dict(condition=condition, model=model, metric=metric, stat=stat, value=value))
                row[f'{model}_{metric}_mean'] = float(values.mean())
                if metric == 'DS':
                    row[f'{model}_DS_sd'] = float(values.std(ddof=1)) if len(values) > 1 else None
            distance = sum(float(r['driven_km']) for r in selected)
            for key in selected[0]:
                if key.endswith('_events'):
                    count = sum(int(r[key]) for r in selected)
                    safety.append(dict(condition=condition, model=model, event=key,
                                       event_count=count, driven_km=distance,
                                       events_per_km=count / distance if distance else None))
        row['DS_CR_minus_FM'] = row['FM_CR_DS_mean'] - row['FM_DS_mean']
        row['SR_CR_minus_FM_pp'] = row['FM_CR_SR_mean'] - row['FM_SR_mean']
        row['CR_DS_wins'] = sum(float(d['FM_CR'][0]['DS']) > float(d['FM'][0]['DS']) for d in repeats)
        values = np.array([np.mean(v) for v in scenario_effects[condition].values()])
        assert len(values) == 44
        samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        row['absolute_DS_gap_scenario_bootstrap_low'] = float(np.quantile(samples, .025))
        row['absolute_DS_gap_scenario_bootstrap_high'] = float(np.quantile(samples, .975))
        summary.append(row)
    write_csv('condition_summary.csv', summary)
    write_csv('all_run_metrics.csv', runs)
    write_csv('smoke_metrics.csv', smoke)
    write_csv('paired_rounds.csv', pair_rows)
    write_csv('route_metrics.csv', routes)
    write_csv('route_behavior_metrics.csv', behaviors)
    write_csv('safety_summary.csv', safety)
    write_csv('all_metric_statistics.csv', metrics)
    write_csv('completed_job_inventory.csv', inventory)
    audit = dict(completed_jobs=len(inventory), smoke_jobs=len(smoke), main_jobs=len(runs),
                 total_route_records=sum(r['route_count'] for r in inventory),
                 main_route_records=len(routes), paired_rounds=len(pair_rows),
                 missing_record_files=missing, source_campaigns=SOURCES)
    (OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    report = ['# 全部闭环扰动实验汇总', '',
              f'已完成{len(inventory)}组、{audit["total_route_records"]}次路线测试。其中冒烟{len(smoke)}组、'
              f'10次路线测试，单列；主分析{len(runs)}组、{len(routes)}次路线测试，{len(pair_rows)}个配对轮次。', '',
              '每组44条固定路线，覆盖44类场景；全部seed0。各配对轮等权，不选择最佳轮次。', '',
              '| 条件 | 配对轮数 | FM DS | CR DS | DS差(CR-FM) | FM SR% | CR SR% | CR DS胜出轮数 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        report.append(f'| {r["label"]} | {r["paired_rounds"]} | {r["FM_DS_mean"]:.2f} | '
                      f'{r["FM_CR_DS_mean"]:.2f} | {r["DS_CR_minus_FM"]:+.2f} | '
                      f'{r["FM_SR_mean"]:.2f} | {r["FM_CR_SR_mean"]:.2f} | '
                      f'{r["CR_DS_wins"]}/{r["paired_rounds"]} |')
    report += ['', '## 安全事件累计', '',
               '下表顺序均为FM / FM+CR。不同条件重复数不同，跨条件比较应查看safety_summary.csv的每公里频率。', '',
               '| 条件 | 车辆碰撞 | 行人碰撞 |', '|---|---:|---:|']
    for r in summary:
        counts = {(v['model'], v['event']): v['event_count'] for v in safety
                  if v['condition'] == r['condition']}
        vehicle = [counts.get((model, 'collisions_vehicle_events'), 0) for model in ['FM', 'FM_CR']]
        pedestrian = [counts.get((model, 'collisions_pedestrian_events'), 0) for model in ['FM', 'FM_CR']]
        report.append(f'| {r["label"]} | {vehicle[0]} / {vehicle[1]} | {pedestrian[0]} / {pedestrian[1]} |')
    report += ['', '## 配对DS差的探索性区间', '',
               '| 条件 | CR-FM DS | 场景bootstrap 95%区间 |', '|---|---:|---:|']
    for r in summary:
        report.append(f'| {r["label"]} | {r["DS_CR_minus_FM"]:+.2f} | '
                      f'[{r["absolute_DS_gap_scenario_bootstrap_low"]:.2f}, '
                      f'{r["absolute_DS_gap_scenario_bootstrap_high"]:.2f}] |')
    report += ['', '## 分析结论', '',
               '规划状态扰动：持续横纵向单轮均偏向CR；横向脉冲全部7轮平均收益较小，4/7轮DS更高。',
               '条件噪声：BEV与Query各5轮均有4轮DS更高，但BEV平均SR较低；单次Query脉冲仅一轮。',
               '定位和航向：横向偏置、横向脉冲、纵向0.10/0.25m及1/5度航向的平均DS均较低；纵向0.50m略高。',
               '全部条件的场景配对DS差区间都跨零；只能报告观察趋势，不能据此确认普遍或显著的鲁棒性优势。',
               '', '## 统计口径与限制', '',
               '- DS差为扰动下绝对成绩差，不是相对clean的退化优势。clean只有一轮。',
               '- 同seed重复不等于独立环境种子或训练种子；结果存在运行波动。',
               '- SD为各轮平均成绩的样本标准差，单轮不估计SD。',
               '- 区间为先在同场景内平均所有重复的模型配对DS差，再对44个场景bootstrap 10000次；'
               '描述这组固定路线上的场景不确定性，不覆盖训练种子变化，也未作多重比较校正。',
               '- 每条件原始事件数及总实际公里数单列，事件频率按总事件数/总公里数计算。',
               '- 行为表的P95是各条路线已有的时间统计量，不能直接当作全体帧的总体P95。',
               '- 脉冲恢复需使用clean路径对齐；既有恢复指标不是专家GT误差。',
               '- 原始逐帧文件本次只核对存在性，不重新解压校验全部内容。', '',
               '## 数据文件', '',
               '- condition_summary.csv：均值、DS标准差、胜出轮数、场景bootstrap区间。',
               '- paired_rounds.csv / all_run_metrics.csv：每轮差异及全部基准和违规指标。',
               '- route_metrics.csv / route_behavior_metrics.csv：逐路线分数、安全与行为统计。',
               '- safety_summary.csv：全部违规事件数与每公里频率。',
               '- all_metric_statistics.csv：DS/RC/IS/SR/Efficiency/Comfort均值、标准差与范围。',
               '- smoke_metrics.csv / completed_job_inventory.csv / audit.json：冒烟、完成清单与审计。', '',
               '## 来源', ''] + ['- ' + str(ROOT / 'outputs/local_evaluation' / s) for s in SOURCES]
    (OUT / 'REPORT.md').write_text('\n'.join(report) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    displayed = [r for r in summary if r['condition'] != 'clean']
    y = np.arange(len(displayed))
    mean = np.array([r['DS_CR_minus_FM'] for r in displayed])
    low = np.array([r['absolute_DS_gap_scenario_bootstrap_low'] for r in displayed])
    high = np.array([r['absolute_DS_gap_scenario_bootstrap_high'] for r in displayed])
    fig, ax = plt.subplots(figsize=(11, 7), layout='constrained')
    ax.errorbar(mean, y, xerr=np.maximum(0, np.vstack([mean-low, high-mean])), fmt='o', capsize=3)
    ax.axvline(0, color='gray', linestyle='--')
    ax.set(yticks=y, yticklabels=[f'{r["condition"]} (n={r["paired_rounds"]})' for r in displayed],
           xlabel='Absolute DS difference: FM+CR minus FM; scenario bootstrap 95% interval',
           title='All completed 44-route repeats; seed0; exploratory intervals')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'paired_DS_summary.{ext}', dpi=200)
    plt.close(fig)
    print(json.dumps(audit, ensure_ascii=False))
    for r in summary:
        print(r['condition'], r['paired_rounds'], round(r['DS_CR_minus_FM'], 3),
              [round(r[k], 3) for k in ['absolute_DS_gap_scenario_bootstrap_low',
                                        'absolute_DS_gap_scenario_bootstrap_high']])


if __name__ == '__main__':
    main()
