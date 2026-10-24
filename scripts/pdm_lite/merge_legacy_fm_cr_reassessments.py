"""Replace historical routes by their latest reassessment, preserving provenance."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'3rd_party/Bench2Drive/leaderboard'))
sys.path.insert(0, str(ROOT/'3rd_party/Bench2Drive/scenario_runner'))
from leaderboard.utils.statistics_manager import StatisticsManager, to_route_record
from scripts.pdm_lite.summarize_eval_metrics import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def failed(row):
    return row['scores']['score_route'] != 100 or any(
        v for k, v in row['infractions'].items() if k != 'min_speed_infractions')


def main():
    root = ROOT/'outputs/local_evaluation'
    directories = [root/'bench2drive220_fm_cr_metrics_20261009_183130',
                   root/'legacy_fm_cr_failed82_reassessment_20261012',
                   root/'legacy_fm_cr_round2_first_round_failures_20261012']
    datasets = []
    for directory, count in zip(directories, (220, 82, 61)):
        data = json.loads((directory/'checkpoint_endpoint.json').read_text())
        assert data['entry_status'] == 'Finished' and data['_checkpoint']['progress'] == [count, count]
        rows = data['_checkpoint']['records']
        assert len(rows) == count and len({r['route_id'] for r in rows}) == count
        assert not any(r['status'] == 'Failed' or 'crash' in r['status'].lower() for r in rows)
        datasets.append(data)
    assert {r['route_id'] for r in datasets[1]['_checkpoint']['records']} == {
        r['route_id'] for r in datasets[0]['_checkpoint']['records'] if failed(r)}
    assert {r['route_id'] for r in datasets[2]['_checkpoint']['records']} == {
        r['route_id'] for r in datasets[1]['_checkpoint']['records'] if failed(r)}
    checkpoint_paths = [Path(json.loads((d/'agent/source.json').read_text())['checkpoint']).resolve()
                        for d in directories]
    assert len(set(checkpoint_paths)) == 1, 'All three evaluations must use the same weights'
    baseline = directories[0]
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    backup = baseline/f'backup_before_latest_reassessment_merge_{stamp}'
    backup.mkdir()
    shutil.copy2(baseline/'checkpoint_endpoint.json', backup/'checkpoint_endpoint.json')
    shutil.copytree(baseline/'analysis', backup/'analysis')
    selected = {}
    for directory, data in zip(directories, datasets):
        for row in data['_checkpoint']['records']:
            selected[row['route_id']] = (copy.deepcopy(row), directory)
    final = copy.deepcopy(datasets[0])
    records, provenance = [], []
    merged = baseline/f'merged_latest_reassessment_{stamp}'
    metric_view = merged/'eval_metrics'
    metric_view.mkdir(parents=True)
    for index, original in enumerate(datasets[0]['_checkpoint']['records']):
        row, directory = selected[original['route_id']]
        assert row['scenario_name'] == original['scenario_name'] and str(row['weather_id']) == str(original['weather_id'])
        row['index'] = index
        records.append(row)
        source = directory/'eval_metrics'/row['save_name']
        assert source.is_dir() and (source/'metric_info.json').is_file(), source
        (metric_view/row['save_name']).symlink_to(source.resolve(), target_is_directory=True)
        provenance.append(dict(route_id=row['route_id'], source=str(directory.resolve()),
                               save_name=row['save_name'], status=row['status'], scores=row['scores']))
    manager = StatisticsManager(str(merged/'checkpoint_endpoint.json'), '')
    manager._total_routes = len(records)
    manager._results.checkpoint.records = [to_route_record(r) for r in records]
    manager.compute_global_statistics()
    final['_checkpoint']['records'] = records
    final['_checkpoint']['global_record'] = manager._results.checkpoint.global_record.to_json()
    final['values'] = manager._results.values
    final['labels'] = manager._results.labels
    final['entry_status'] = 'Finished'
    final['eligible'] = True
    manifest = dict(protocol='Latest route result: round2 overrides round1, round1 overrides original',
                    scope='Failure-selected repeated evaluation composite; not an independent single 220-route run',
                    created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                    weights=str(checkpoint_paths[0]), original_backup=str(backup.resolve()),
                    sources=[dict(directory=str(d.resolve()), result_sha256=sha(d/'checkpoint_endpoint.json'))
                             for d in directories], route_sources=provenance)
    final['reassessment_merge'] = {k:v for k,v in manifest.items() if k != 'route_sources'}
    result = merged/'checkpoint_endpoint.json'
    result.write_text(json.dumps(final, ensure_ascii=False, indent=2)+'\n')
    report = summarize(result, metric_view, ROOT/'3rd_party/Bench2Drive/tools/efficiency_smoothness_benchmark.py')
    report['result'] = str((baseline/'checkpoint_endpoint.json').resolve())
    report['merge_protocol'] = manifest['protocol']
    report['evaluation_scope'] = manifest['scope']
    assert report['route_count'] == 220 and report['comfort_complete']
    for key in ('score_composed', 'score_route', 'score_penalty'):
        assert abs(sum(r['scores'][key] for r in records)/220 -
                   final['_checkpoint']['global_record']['scores_mean'][key]) < 1e-6
    (merged/'merge_provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    # Publish only after official global statistics and all route metrics validate.
    temporary = baseline/'checkpoint_endpoint.merged.tmp'
    temporary.write_bytes(result.read_bytes())
    temporary.replace(baseline/'checkpoint_endpoint.json')
    (baseline/'analysis/benchmark_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (baseline/'analysis/latest_reassessment_merge.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'routes'}, ensure_ascii=False, indent=2))
    print('Metric view:', metric_view)
    print('Backup:', backup)


if __name__ == '__main__':
    main()
