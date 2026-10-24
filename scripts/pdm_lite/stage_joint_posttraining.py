"""Make a local training view of all eligible same-state teacher samples."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FOCUS_SCENARIOS = {'BlockedIntersection', 'Accident',
                   'NonSignalizedJunctionRightTurn', 'OppositeVehicleTakingPriority'}


def stationary_frame_weights(events, sample_interval_s, stationary_speed_mps=.2,
                             stationary_after_s=5., transition_tail_s=1., stationary_weight=.1):
    """Reduce long stationary interiors, retaining entry and exit context."""
    weights = {frame: 1. for frame in events}
    group = []

    def finish():
        if group:
            for frame in group:
                if ((frame - group[0]) * sample_interval_s >= stationary_after_s
                        and (group[-1] - frame) * sample_interval_s >= transition_tail_s):
                    weights[frame] = stationary_weight

    for frame in sorted(events):
        stationary = abs(float(events[frame]['speed'])) < stationary_speed_mps
        if not stationary or (group and frame != group[-1] + 1):
            finish()
            group = []
        if stationary:
            group.append(frame)
    finish()
    return weights


def link(source, destination):
    if destination.exists() or destination.is_symlink():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise ValueError(f'Conflicting training-view entry: {destination}')
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def stage(evaluations, output):
    sys.path.insert(0, str(ROOT / '3rd_party/carla_garage/team_code'))
    from data import CARLA_Data
    from config import GlobalConfig
    output = output.resolve()
    evaluations = [p.resolve() for p in evaluations]
    source_identity = {'evaluations': list(map(str, evaluations)), 'selection': 'all_eligible_frames'}
    output.mkdir(parents=True, exist_ok=True)
    identity_file = output / 'sources.json'
    if identity_file.exists() and json.loads(identity_file.read_text()) != source_identity:
        raise ValueError('Use a different output directory for different data sources')
    jsonpickle_numpy.register_handlers()
    scenario = output / 'Recovery_Train'
    scenario.mkdir(exist_ok=True)
    config_text = None
    keys, records, sampling_frames = [], [], {}
    for evaluation in evaluations:
        settings = json.loads((evaluation / 'collection_config.json').read_text())
        verification = json.loads((evaluation / 'joint_collection_verification.json').read_text())
        endpoint = json.loads((evaluation / 'checkpoint_endpoint.json').read_text())
        if verification['status'] != 'passed' or endpoint['entry_status'] != 'Finished':
            raise ValueError(f'Collection has not finished verification: {evaluation}')
        progress = endpoint['_checkpoint']['progress']
        if progress[0] != progress[1]:
            raise ValueError(f'Incomplete collection: {evaluation}')
        text = (Path(settings['student_agent_dir']) / 'config.json').read_text()
        if config_text is not None and text != config_text:
            raise ValueError('Student configurations differ between collections')
        config_text = text
        config = jsonpickle.decode(text)
        if not isinstance(config, GlobalConfig):
            raise TypeError('Could not restore the saved CARLA Garage configuration')
        scenarios = {}
        for xml_route in ET.parse(evaluation / 'routes.xml').getroot():
            kinds = {s.get('type') for s in xml_route.findall('./scenarios/scenario')}
            if len(kinds) != 1 or None in kinds:
                raise ValueError('Expected exactly one scenario type per collection route')
            scenarios[xml_route.get('id')] = kinds.pop()
        for result in endpoint['_checkpoint']['records']:
            name = result['save_name'].replace('_rep0_', '_Rep0_')
            route = Path(settings['collection_root']) / 'Recovery_Train' / name
            view = scenario / name
            view.mkdir(exist_ok=True)
            # Keep the route itself local: the loader identifies recovery routes
            # by canonical route path. Only sensor/measurement directories link.
            for subdir in ('rgb', 'rgb_augmented', 'lidar', 'measurements'):
                link(route / subdir, view / subdir)
            for filename in ('collection_summary.json', 'recovery_events.jsonl', 'metadata.json'):
                if (route / filename).exists():
                    link(route / filename, view / filename)
            summary = json.loads((route / 'collection_summary.json').read_text())
            events = {r['frame']: r for r in map(json.loads, (route / 'recovery_events.jsonl').read_text().splitlines())}
            frame_weights = stationary_frame_weights(events, config.data_save_freq / config.carla_fps)
            scenario_type = scenarios[name.split('_Rep0_')[0].split('_')[-1]]
            decisions, excluded = Counter(), Counter()
            selected = 0
            for frame in range(summary['saved_frames']):
                if frame < config.skip_first or frame * config.data_save_freq < config.inital_frames_delay:
                    excluded['initialization'] += 1
                    continue
                if frame % config.train_sampling_rate or frame >= summary['saved_frames'] - (config.seq_len - 1):
                    excluded['sequence_or_sampling'] += 1
                    continue
                with gzip.open(route / 'measurements' / f'{frame:04d}.json.gz', 'rt') as handle:
                    label = json.load(handle)
                if not label['label_valid']:
                    excluded['invalid_label'] += 1
                    continue
                trajectory = np.asarray(label['teacher_checkpoints'])
                speed = float(label['target_speed'])
                if trajectory.shape != (config.fmsr_num_poses, 2) or not np.isfinite(trajectory).all() or not 0 <= speed <= 30:
                    raise ValueError(f'Invalid teacher label at {route}/{frame}')
                if not events[frame]['label_valid'] or label['frame'] != frame or label['simulation_frame'] != events[frame]['simulation_frame']:
                    raise ValueError(f'Teacher label/event mismatch: {route}/{frame}')
                for subdir, suffix in (('rgb', '.jpg'), ('rgb_augmented', '.jpg'), ('lidar', '.laz')):
                    if not (view / subdir / f'{frame:04d}{suffix}').is_file():
                        raise FileNotFoundError(f'{view}/{subdir}/{frame:04d}{suffix}')
                key = f'Recovery_Train/{name}/{frame:04d}'
                keys.append(key)
                sampling_frames[key] = {'scenario': scenario_type, 'weight': frame_weights[frame],
                                        'long_stationary': frame_weights[frame] < 1}
                decisions['stop' if speed < .5 else 'go' if speed > 2 else 'slow'] += 1
                selected += 1
            records.append({'route_id': result['route_id'], 'source': str(route),
                            'scenario_type': scenario_type,
                            'long_stationary_frames': sum(sampling_frames[k]['long_stationary']
                                                          for k in keys if k.startswith(f'Recovery_Train/{name}/')),
                            'raw_frames': summary['saved_frames'], 'selected_frames': selected,
                            'excluded': dict(excluded), 'teacher_decisions': dict(decisions)})
    if len(keys) != len(set(keys)) or not keys:
        raise ValueError('Empty selection or duplicate frame keys')
    manifest = output / 'recovery_frames.txt'
    manifest.write_text(''.join(key + '\n' for key in sorted(keys)))
    scenario_counts = Counter(r['scenario'] for r in sampling_frames.values())
    sampling_plan = {'version': 1, 'frames': sampling_frames,
                     'scenario_weights': {name: 2. if name in FOCUS_SCENARIOS else 1.
                                          for name in sorted(scenario_counts)},
                     'policy': {'stationary_speed_mps': .2, 'stationary_after_s': 5.,
                                'transition_tail_s': 1., 'stationary_weight': .1,
                                'grouping': 'scenario_type', 'per_frame_repeats_capped': True}}
    sampling_path = output / 'recovery_sampling_plan.json'
    sampling_path.write_text(json.dumps(sampling_plan, indent=2) + '\n')
    # Check the exact same loader and root expansion used by train.py.
    config = jsonpickle.decode(config_text)
    config.fmsr_recovery_root = str(output)
    config.fmsr_recovery_manifest = str(manifest)
    config.fmsr_frame_manifest = ''
    config.augment = config.use_color_aug = config.lidar_aug_prob = 0
    config.initialize(root_dir=[str(output)], setting='all')
    dataset = CARLA_Data(config.data_roots, config)
    if len(dataset) != len(keys) or not all(dataset.sample_is_recovery):
        raise ValueError(f'Training loader selected {len(dataset)} instead of {len(keys)} recovery samples')
    for index in sorted({0, len(dataset) // 2, len(dataset) - 1}):
        sample = dataset[index]
        frame = int(dataset.sample_start[index])
        measurement_root = Path(dataset.measurements[index][0].decode())
        with gzip.open(measurement_root / f'{frame:04d}.json.gz', 'rt') as handle:
            label = json.load(handle)
        np.testing.assert_allclose(sample['target_speed_value'], label['target_speed'], rtol=1e-5)
        np.testing.assert_allclose(sample['route'][:config.fmsr_num_poses], label['teacher_checkpoints'], rtol=1e-5, atol=1e-5)
        if not np.isfinite(sample['rgb']).all() or not np.isfinite(sample['lidar']).all():
            raise ValueError('Nonfinite sensor tensors')
    report = dict(source_identity, raw_frames=sum(r['raw_frames'] for r in records),
                  selected_frames=len(keys), native_loader_frames=len(dataset),
                  all_samples_identified_as_recovery=True, label_presence_is_not_behavioral_validation=True,
                  scenario_frames=dict(scenario_counts),
                  long_stationary_frames=sum(r['long_stationary'] for r in sampling_frames.values()),
                  sampling_plan=str(sampling_path),
                  split='development', routes=records, recovery_manifest=str(manifest))
    identity_file.write_text(json.dumps(source_identity, indent=2) + '\n')
    (output / 'training_data_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage(args.evaluation, args.output)
