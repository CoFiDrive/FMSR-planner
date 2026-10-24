"""Select the highest-DS attempt per route, keeping all metrics from that attempt."""
import copy
from collections import Counter
import json
from pathlib import Path

from scripts.pdm_lite.merge_legacy_fm_cr_reassessments import (
    ROOT, StatisticsManager, to_route_record, summarize, sha,
)


def main():
    root = ROOT/'outputs/local_evaluation'
    original = root/'bench2drive220_fm_cr_metrics_20261009_183130'
    source_paths = [original/'backup_before_latest_reassessment_merge_20261012_223038/checkpoint_endpoint.json',
                    root/'legacy_fm_cr_failed82_reassessment_20261012/checkpoint_endpoint.json',
                    root/'legacy_fm_cr_round2_first_round_failures_20261012/checkpoint_endpoint.json']
    metric_roots = [original, source_paths[1].parent, source_paths[2].parent]
    datasets = [json.loads(p.read_text()) for p in source_paths]
    candidates = {}
    for attempt, (dataset, count) in enumerate(zip(datasets, (220, 82, 61))):
        assert dataset['entry_status'] == 'Finished'
        assert dataset['_checkpoint']['progress'] == [count, count]
        rows = dataset['_checkpoint']['records']
        assert len(rows) == count and len({r['route_id'] for r in rows}) == count
        for row in rows:
            assert row['status'] != 'Failed' and 'crash' not in row['status'].lower()
            candidates.setdefault(row['route_id'], []).append((row, attempt))
    assert len(candidates) == 220
    output = root/'legacy_fm_cr_best_of_original_round1_round2_20261012'
    metric_view = output/'eval_metrics'
    metric_view.mkdir(parents=True, exist_ok=True)
    records, provenance = [], []
    for index, baseline in enumerate(datasets[0]['_checkpoint']['records']):
        options = candidates[baseline['route_id']]
        # Equal DS retains the earlier attempt, without selecting on other metrics.
        row, attempt = max(options, key=lambda pair: pair[0]['scores']['score_composed'])
        row = copy.deepcopy(row)
        assert row['scenario_name'] == baseline['scenario_name']
        assert str(row['weather_id']) == str(baseline['weather_id'])
        assert row['scores']['score_composed'] >= baseline['scores']['score_composed']
        row['index'] = index
        records.append(row)
        source = metric_roots[attempt]/'eval_metrics'/row['save_name']
        assert source.is_dir() and (source/'metric_info.json').is_file()
        link = metric_view/row['save_name']
        if link.is_symlink():
            assert link.resolve() == source.resolve()
        else:
            link.symlink_to(source.resolve(), target_is_directory=True)
        provenance.append(dict(route_id=row['route_id'], attempt=attempt,
                               source_result=str(source_paths[attempt]), metric_directory=str(source),
                               scores=row['scores'], all_attempt_DS=[r['scores']['score_composed'] for r,_ in options]))
    result = copy.deepcopy(datasets[0])
    manager = StatisticsManager(str(output/'checkpoint_endpoint.json'), '')
    manager._total_routes = 220
    manager._results.checkpoint.records = [to_route_record(r) for r in records]
    manager.compute_global_statistics()
    result['_checkpoint']['records'] = records
    result['_checkpoint']['global_record'] = manager._results.checkpoint.global_record.to_json()
    result['values'], result['labels'] = manager._results.values, manager._results.labels
    scope = 'Highest-DS attempt from original and failure-selected reruns; not a single independent benchmark run'
    protocol = 'Select max DS per route; equal DS retains earlier attempt; all other metrics use that same attempt'
    result['best_attempt_selection'] = dict(scope=scope, protocol=protocol)
    endpoint = output/'checkpoint_endpoint.json'
    endpoint.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    report = summarize(endpoint, metric_view, ROOT/'3rd_party/Bench2Drive/tools/efficiency_smoothness_benchmark.py')
    assert report['route_count'] == 220 and report['comfort_complete']
    for key in ('score_composed', 'score_route', 'score_penalty'):
        assert abs(sum(r['scores'][key] for r in records)/220 - report['global_scores_mean'][key]) < 1e-6
    report['evaluation_scope'], report['selection_protocol'] = scope, protocol
    report['selected_attempt_counts'] = dict(Counter(x['attempt'] for x in provenance))
    report['route_status_counts'] = dict(Counter(r['status'] for r in records))
    report['infraction_event_counts'] = {k:sum(len(r['infractions'].get(k, [])) for r in records)
                                       for k in records[0]['infractions'] if k != 'min_speed_infractions'}
    (output/'benchmark_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    (output/'selection_provenance.json').write_text(json.dumps(dict(scope=scope, protocol=protocol,
        sources=[dict(path=str(p), sha256=sha(p)) for p in source_paths], routes=provenance),
        ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'routes'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
