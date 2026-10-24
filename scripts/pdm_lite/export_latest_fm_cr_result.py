"""Create a complete route/scenario report for latest-attempt FM+CR merging."""
import copy
import csv
import json
from pathlib import Path
import shutil
from collections import Counter, defaultdict

from scripts.pdm_lite.merge_legacy_fm_cr_reassessments import (
    ROOT, StatisticsManager, to_route_record, summarize,
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def link(path, target):
    target = target.resolve(strict=True)
    if path.is_symlink():
        assert path.resolve() == target
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def main():
    root = ROOT/'outputs/local_evaluation'
    baseline = root/'bench2drive220_fm_cr_metrics_20261009_183130'
    source = baseline/'merged_latest_reassessment_20261012_223038'
    output = root/'bench2drive220_legacy_fm_cr_latest_merged_20261012'
    output.mkdir(exist_ok=True)
    analysis = output/'analysis'
    analysis.mkdir(exist_ok=True)
    view = output/'eval_metrics'
    view.mkdir(exist_ok=True)
    scenarios = output/'scenarios'
    scenarios.mkdir(exist_ok=True)
    data = json.loads((source/'checkpoint_endpoint.json').read_text())
    provenance = json.loads((source/'merge_provenance.json').read_text())
    by_id = {r['route_id']:r for r in provenance['route_sources']}
    records = data['_checkpoint']['records']
    assert len(records) == len(by_id) == 220
    for record in records:
        selected = by_id[record['route_id']]
        assert record['save_name'] == selected['save_name'] and record['scores'] == selected['scores']
        link(view/record['save_name'], Path(selected['source'])/'eval_metrics'/record['save_name'])
    manager = StatisticsManager(str(output/'checkpoint_endpoint.json'), '')
    manager._total_routes = 220
    manager._results.checkpoint.records = [to_route_record(r) for r in records]
    manager.compute_global_statistics()
    data['_checkpoint']['global_record'] = manager._results.checkpoint.global_record.to_json()
    data['labels'], data['values'] = manager._results.labels, manager._results.values
    data['reassessment_merge']['output_directory'] = str(output)
    write_json(output/'checkpoint_endpoint.json', data)
    report = summarize(output/'checkpoint_endpoint.json', view,
                       ROOT/'3rd_party/Bench2Drive/tools/efficiency_smoothness_benchmark.py')
    report['evaluation_scope'] = provenance['scope']
    report['merge_protocol'] = provenance['protocol']
    assert round(report['global_scores_mean']['score_composed'],4) == 88.8349
    assert report['route_success_percent'] == 75.0
    assert round(report['efficiency_percent'],2) == 246.92
    assert round(report['comfort_percent'],2) == 18.40
    assert report['comfort_route_count'] == 220 and report['comfort_complete']
    write_json(analysis/'benchmark_metrics.json', report)
    write_json(output/'merge_provenance.json', provenance)
    full_routes = ROOT/'3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml'
    shutil.copy2(full_routes, output/'routes.xml')
    groups = defaultdict(list)
    all_rows = []
    for metric, record in zip(report['routes'], records):
        assert metric['route_id'] == record['route_id']
        row = dict(route_id=record['route_id'], scenario=record['scenario_name'],
                   town=record.get('town_name'), weather=record['weather_id'], status=record['status'],
                   DS=record['scores']['score_composed'], RC=record['scores']['score_route'],
                   IS=record['scores']['score_penalty'], success=metric['route_success'],
                   efficiency=metric['efficiency'], comfort=metric['comfort'],
                   source=by_id[record['route_id']]['source'])
        all_rows.append(row)
        groups[row['scenario']].append(row)
        route_dir = scenarios/row['scenario']/row['route_id']
        route_dir.mkdir(parents=True, exist_ok=True)
        write_json(route_dir/'route_record.json', record)
        write_json(route_dir/'metrics.json', dict(metric, selected_source=row['source']))
        link(route_dir/'telemetry', view/record['save_name'])
        diagnostics = Path(row['source'])/'diagnostics/diagnostics'/record['save_name']
        if diagnostics.is_dir():
            link(route_dir/'diagnostics', diagnostics)
    scene_reports = []
    for scenario, rows in sorted(groups.items()):
        valid_eff = [r['efficiency'] for r in rows if r['efficiency'] is not None]
        scene = dict(scenario=scenario, route_count=len(rows),
                     DS=sum(r['DS'] for r in rows)/len(rows),
                     RC=sum(r['RC'] for r in rows)/len(rows), IS=sum(r['IS'] for r in rows)/len(rows),
                     success_count=sum(r['success'] for r in rows),
                     success_rate_percent=100*sum(r['success'] for r in rows)/len(rows),
                     efficiency_percent=sum(valid_eff)/len(valid_eff) if valid_eff else None,
                     efficiency_route_count=len(valid_eff),
                     comfort_percent=100*sum(r['comfort'] for r in rows)/len(rows),
                     statuses=dict(Counter(r['status'] for r in rows)), routes=rows)
        write_json(scenarios/scenario/'summary.json', scene)
        scene_reports.append(scene)
    assert len(scene_reports) == 44 and all(r['route_count']==5 for r in scene_reports)
    assert sum(r['success_count'] for r in scene_reports) == 165
    write_json(analysis/'scenario_metrics.json', dict(scenario_count=44, route_count=220,
                                                   selection_protocol=provenance['protocol'], scenarios=scene_reports))
    with (analysis/'route_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    with (analysis/'scenario_metrics.csv').open('w', newline='') as handle:
        fields = ['scenario','route_count','DS','RC','IS','success_count','success_rate_percent',
                  'efficiency_percent','efficiency_route_count','comfort_percent']
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(scene_reports)
    readme = '''# 旧版 FM+CR 最新复测合并结果

规则：第二轮覆盖第一轮，第一轮覆盖初次；不按 DS 择优。
220 条路线来自初次 138 条、第一轮 21 条、第二轮 61 条。
这是失败路线重复评测的合并成绩，不是独立单次 220 路线评测。

DS=88.8349 / RC=97.6494 / IS=0.9090 / SR=75.00% / Efficiency=246.92% / Comfort=18.40%。

- checkpoint_endpoint.json：重新组合并重新计算全局分数的 220 条路线结果。
- analysis/benchmark_metrics.json：逐路线及整体完整指标。
- analysis/route_metrics.csv：220 条路线指标与来源。
- analysis/scenario_metrics.json、scenario_metrics.csv：44 类场景指标。
- scenarios/<场景>/<路线>/：该路线选中记录、指标、逐帧数据和诊断链接。
- eval_metrics/：仅包含所选 220 次驾驶的原始指标目录链接。
- merge_provenance.json：每条路线来源及输入结果校验信息。

原始逐帧数据通过符号链接引用对应评测目录；复制或导出时需解引用这些链接。
'''
    (output/'README.md').write_text(readme)
    print(json.dumps(dict(output=str(output), scenario_count=len(scene_reports), route_count=220,
                         scores=report['global_scores_mean'], SR=report['route_success_percent'],
                         efficiency=report['efficiency_percent'], comfort=report['comfort_percent'],
                         selected_sources=dict(Counter(r['source'] for r in all_rows))),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
