"""Package verified same-state teacher disagreement and incident windows.

Candidate labels require behavioral review; shape validity does not establish
that the teacher would avoid a collision or recover from the observed state.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
import os
from pathlib import Path

import numpy as np


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def trigger_windows(rows, before=2., after=1., gap=.5):
    """Bound persistent triggers by sampling onset and offset windows."""
    by_reason = defaultdict(list)
    for row in rows:
        if row['eligible']:
            for reason in row['reasons']:
                by_reason[reason].append(row['time_s'])
    windows = []
    for reason, times in sorted(by_reason.items()):
        groups = []
        for timestamp in sorted(set(times)):
            if not groups or timestamp - groups[-1][-1] > gap + 1e-6:
                groups.append([timestamp])
            else:
                groups[-1].append(timestamp)
        for group in groups:
            anchors = sorted(set([group[0], group[-1]]))
            for anchor in anchors:
                windows.append({'reason': reason, 'start_s': anchor - before,
                                'end_s': anchor + after,
                                'trigger_first_s': group[0], 'trigger_last_s': group[-1]})
    return windows


def select_window_rows(rows, windows):
    selected = {}
    for row in rows:
        if not row['eligible']:
            continue
        reasons = sorted({w['reason'] for w in windows
                          if w['start_s'] - 1e-6 <= row['time_s'] <= w['end_s'] + 1e-6})
        if reasons:
            selected[row['frame']] = dict(row, window_reasons=reasons)
    return selected


def package(evaluations, destination):
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy
    from data import CARLA_Data
    jsonpickle_numpy.register_handlers()
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    train_roots, collections = [], []
    selected_keys, incident_keys = set(), set()
    records, route_reports = [], []
    config_text = None
    for evaluation in evaluations:
        evaluation = Path(evaluation)
        settings = json.loads((evaluation / 'collection_config.json').read_text())
        verified = json.loads((evaluation / 'joint_collection_verification.json').read_text())
        if verified['status'] != 'passed':
            raise ValueError(f'Collection has not passed verification: {evaluation}')
        result = json.loads((evaluation / 'checkpoint_endpoint.json').read_text())
        checkpoint = result['_checkpoint']
        if result['entry_status'] != 'Finished' or checkpoint['progress'][0] != checkpoint['progress'][1]:
            raise ValueError(f'Collection is incomplete: {evaluation}')
        text = (Path(settings['student_agent_dir']) / 'config.json').read_text()
        if config_text is not None and config_text != text:
            raise ValueError('Collections use different student configurations')
        config_text = text
        config = jsonpickle.decode(text)
        collection = Path(settings['collection_root']).resolve()
        collections.append(collection)
        train_roots.append(str(collection / 'Recovery_Train'))
        earliest_step = max(float(config.inital_frames_delay),
                            int(config.skip_first) * int(config.data_save_freq))
        for record in checkpoint['records']:
            status = record['status']
            if any(word in status.lower() for word in ('crashed', 'simulation', 'set up', 'setup')):
                raise ValueError(f'Infrastructure or agent failure: {status}')
            name = record['save_name']
            diagnostic = collection / 'diagnostics' / name
            route = collection / 'Recovery_Train' / name.replace('_rep0_', '_Rep0_')
            frames = read_rows(diagnostic / 'frames.jsonl')
            by_frame = {row['simulation_frame']: row for row in frames}
            events = read_rows(route / 'recovery_events.jsonl')
            samples = []
            for event in events:
                frame = event['frame']
                with gzip.open(route / 'measurements' / f'{frame:04d}.json.gz', 'rt') as handle:
                    label = json.load(handle)
                state = by_frame[label['simulation_frame']]
                policy = state['policy']
                eligible = bool(label['label_valid'] and state['step'] >= earliest_step
                                and policy.get('phase') == 'inference'
                                and not policy.get('filter_initial_delay_brake', False)
                                and frame % config.train_sampling_rate == 0
                                and frame >= config.lidar_seq_len - 1
                                and frame + config.seq_len <= len(events))
                if label['label_valid']:
                    trajectory = np.asarray(label['teacher_checkpoints'])
                    if trajectory.shape != (config.fmsr_num_poses, 2) or not np.isfinite(trajectory).all():
                        raise ValueError('Invalid teacher trajectory')
                    speed = float(label['target_speed'])
                    if not math.isfinite(speed) or not 0 <= speed <= 30:
                        raise ValueError('Invalid teacher speed')
                    np.testing.assert_allclose(trajectory, policy['teacher_route_ego_m'], rtol=1e-5, atol=1e-5)
                    np.testing.assert_allclose(speed, policy['teacher_target_speed_mps'], rtol=1e-5, atol=1e-5)
                samples.append({'frame': frame, 'simulation_frame': label['simulation_frame'],
                    'time_s': state['agent_timestamp_s'], 'eligible': eligible,
                    'reasons': event['reasons'], 'measured_speed_mps': label['speed'],
                    'student_target_speed_mps': event.get('student_target_speed'),
                    'teacher_target_speed_mps': label['target_speed'],
                    'student_control': label['student_control'],
                    'teacher_control': label['teacher_control'],
                    'control_overrides': policy.get('overrides', [])})
            windows = trigger_windows(samples)
            selected = select_window_rows(samples, windows)
            incident_windows = []
            for event in read_rows(diagnostic / 'events.jsonl'):
                if event['type'] == 'MIN_SPEED_INFRACTION':
                    continue
                state = by_frame.get(event['event_frame'])
                if state:
                    timestamp = state['agent_timestamp_s']
                    incident_windows.append({'reason': event['type'],
                        'start_s': timestamp - 2., 'end_s': timestamp + 1.})
            incidents = select_window_rows(samples, incident_windows)
            for frame, row in selected.items():
                key = f'Recovery_Train/{route.name}/{frame:04d}'
                if key in selected_keys:
                    raise ValueError(f'Duplicate sample key: {key}')
                selected_keys.add(key)
                records.append(dict(row, key=key, measurement_directory=str(route / 'measurements'),
                                    requires_behavioral_review=True))
            for frame in incidents:
                incident_keys.add(f'Recovery_Train/{route.name}/{frame:04d}')
            reasons = Counter(reason for row in samples if row['eligible'] for reason in row['reasons'])
            route_reports.append({'route_id': record['route_id'], 'save_name': name,
                'rollout_status': status, 'scores': record['scores'],
                'saved_samples': len(events), 'eligible_samples': sum(r['eligible'] for r in samples),
                'trigger_counts': dict(reasons), 'candidate_samples': len(selected),
                'incident_review_samples': len(incidents), 'event_windows': windows,
                'incident_windows': incident_windows,
                'candidate_teacher_stop_samples': sum(r['teacher_target_speed_mps'] < .5 for r in selected.values()),
                'candidate_teacher_go_samples': sum(r['teacher_target_speed_mps'] > 2 for r in selected.values())})
    if not selected_keys:
        raise ValueError('No eligible disagreement windows across completed collections')
    manifest = destination / 'candidate_frames.txt'
    manifest.write_text(''.join(key + '\n' for key in sorted(selected_keys)))
    (destination / 'incident_review_frames.txt').write_text(''.join(key + '\n' for key in sorted(incident_keys)))
    config = jsonpickle.decode(config_text)
    config.fmsr_recovery_root = os.path.commonpath([str(root) for root in collections])
    config.fmsr_recovery_manifest = str(manifest.resolve())
    config.fmsr_frame_manifest = ''
    config.augment = config.use_color_aug = 0
    config.lidar_aug_prob = 0
    dataset = CARLA_Data(train_roots, config)
    if len(dataset) != len(selected_keys):
        raise ValueError(f'Native loader retained {len(dataset)} of {len(selected_keys)} selected samples')
    for index in range(len(dataset)):
        sample = dataset[index]
        frame = int(dataset.sample_start[index])
        directory = Path(dataset.measurements[index][0].decode())
        with gzip.open(directory / f'{frame:04d}.json.gz', 'rt') as handle:
            label = json.load(handle)
        np.testing.assert_allclose(sample['route'][:config.fmsr_num_poses],
                                   label['teacher_checkpoints'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sample['target_speed_value'], label['target_speed'], rtol=1e-5)
        if not np.isfinite(sample['rgb']).all() or not np.isfinite(sample['lidar']).all():
            raise ValueError('Nonfinite native training inputs')
    with (destination / 'candidate_samples.jsonl').open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
    report = {'status': 'passed', 'split': 'development',
        'evaluations': [str(Path(p).resolve()) for p in evaluations],
        'train_roots': train_roots, 'fmsr_recovery_root': config.fmsr_recovery_root,
        'fmsr_recovery_manifest': str(manifest.resolve()),
        'selected_candidate_samples': len(selected_keys),
        'all_selected_samples_loaded_and_teacher_labels_matched': True,
        'incident_review_samples': len(incident_keys),
        'selection': '2s before / 1s after onset and offset of disagreement clusters; initialization excluded',
        'requires_behavioral_review': True,
        'note': 'Valid teacher predictions are candidate supervision, not proof of safe recovery.',
        'routes': route_reports}
    (destination / 'event_samples_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.evaluation, args.output)
