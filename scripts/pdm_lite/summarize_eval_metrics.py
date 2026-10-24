"""Compute Efficiency from route records and audit/export official Comfort inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from scripts.pdm_lite.eval_metrics import rebuild_metric_info


def summarize(result, metric_dir, benchmark, rebuild=False):
    import numpy as np
    spec = importlib.util.spec_from_file_location('b2d_comfort', benchmark)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = json.loads(Path(result).read_text())
    records = data['_checkpoint']['records']
    rows = []
    for record in records:
        name = record['save_name']
        if not name or name in ('.', '..') or Path(name).name != name:
            raise ValueError(f'Invalid save_name: {name!r}')
        row = {'save_name': name, 'route_id': record['route_id'], 'status': record['status'],
               'scenario_name':record.get('scenario_name'), 'town_name':record.get('town_name'),
               'weather_id':record.get('weather_id'), 'scores':record.get('scores'),
               'infractions':record['infractions'], 'meta':record.get('meta')}
        scores = record.get('scores', {})
        row['route_success'] = (scores.get('score_route') == 100 and
                                not any(v for k,v in record['infractions'].items()
                                        if k != 'min_speed_infractions')) if scores else None
        ratios = []
        events = record['infractions'].get('min_speed_infractions')
        row['efficiency_event_count'] = len(events) if events is not None else None
        for event in events or []:
            match = re.search(r'\b\d+\.?\d*%', event)
            if match is None:
                raise ValueError(f'Malformed min_speed event in {name}: {event}')
            ratio = float(match.group().rstrip('%'))
            if ratio <= 1000:
                ratios.append(ratio)
        row['efficiency_valid_event_count'] = len(ratios)
        row['efficiency'] = sum(ratios) / len(ratios) if ratios else None
        directory = Path(metric_dir) / name
        if rebuild and (directory / 'vehicle_state.jsonl').exists():
            rebuild_metric_info(directory)
        metric_path = directory / 'metric_info.json'
        row['comfort'] = None
        if not metric_path.exists():
            row['comfort_error'] = 'missing metric_info.json'
        else:
            try:
                metadata_path = directory / 'metadata.json'
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata.get('irregular_metric_sampling'):
                        raise ValueError('irregular cadence; official script assumes dt=0.1')
                samples = json.loads(metric_path.read_text())
                samples = [samples[k] for k in sorted(samples, key=int)]
                keys = ('acceleration', 'angular_velocity', 'forward_vector',
                        'right_vector', 'location', 'rotation')
                arrays = {key: np.asarray([sample[key] for sample in samples]) for key in keys}
                if any(a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all()
                       for a in arrays.values()):
                    raise ValueError('invalid/nonfinite vector samples')
                row['metric_frame_count'] = len(samples)
                row['comfort'] = float(module.seg_compute_comfort_metric(**arrays))
            except (ValueError, KeyError, ZeroDivisionError) as error:
                row['comfort_error'] = str(error)
        rows.append(row)
    efficiencies = [r['efficiency'] for r in rows if r['efficiency'] is not None]
    comforts = [r['comfort'] for r in rows if r['comfort'] is not None]
    complete = bool(rows) and len(comforts) == len(rows)
    success = [r['route_success'] for r in rows if r['route_success'] is not None]
    return {
        'result': str(Path(result).resolve()), 'progress': data['_checkpoint'].get('progress'),
        'route_count': len(rows), 'efficiency_route_count': len(efficiencies),
        'global_scores_mean': (data['_checkpoint'].get('global_record') or {}).get('scores_mean'),
        'route_success_count':sum(success) if len(success)==len(rows) else None,
        'route_success_percent':100*sum(success)/len(success) if success and len(success)==len(rows) else None,
        'efficiency_percent': sum(efficiencies) / len(efficiencies) if efficiencies else None,
        'comfort_route_count': len(comforts), 'comfort_complete': complete,
        'comfort_fraction': sum(comforts) / len(comforts) if complete else None,
        'comfort_percent': 100 * sum(comforts) / len(comforts) if complete else None,
        'partial_comfort_fraction': sum(comforts) / len(comforts) if comforts else None,
        'benchmark_script': str(Path(benchmark).resolve()),
        'benchmark_sha256': hashlib.sha256(Path(benchmark).read_bytes()).hexdigest(),
        'protocol': 'local Bench2Drive script, raw CARLA angular units, 0.1s metric export; '
                    'physical radian/derivative diagnostics must be reported separately',
        'routes': rows,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True, help='merge.json or checkpoint_endpoint.json')
    parser.add_argument('--metric-dir', type=Path, required=True)
    parser.add_argument('--benchmark-script', type=Path,
                        default=root / '3rd_party/Bench2Drive/tools/efficiency_smoothness_benchmark.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rebuild', action='store_true', help='recover metric_info from vehicle_state.jsonl')
    args = parser.parse_args()
    report = summarize(args.result, args.metric_dir, args.benchmark_script, args.rebuild)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'routes'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
