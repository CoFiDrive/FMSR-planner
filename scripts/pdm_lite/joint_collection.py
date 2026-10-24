"""Stage joint collection and verify native PDM training samples."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.collection.mkdir(parents=True, exist_ok=True)
    for checkpoint in (args.student, args.teacher):
        checkpoint.resolve(strict=True)
        directory = args.output / ('student_agent' if checkpoint == args.student else 'teacher_agent')
        directory.mkdir(exist_ok=True)
        config_text = (checkpoint.parent / 'config.json').read_text()
        link = directory / checkpoint.name
        models = list(directory.glob('model*.pth'))
        if models and (models != [link] or link.resolve() != checkpoint.resolve()):
            raise ValueError('Staged directory contains another model')
        if not link.exists():
            link.symlink_to(checkpoint.resolve())
        (directory / 'config.json').write_text(config_text)
    settings = {
        'student_agent_dir': str((args.output / 'student_agent').resolve()),
        'teacher_agent_dir': str((args.output / 'teacher_agent').resolve()),
        'collection_root': str(args.collection.resolve()),
        'student_checkpoint': str(args.student.resolve()), 'teacher_checkpoint': str(args.teacher.resolve()),
        'benchmark_route_used_for_development': True,
    }
    settings_file = args.output / 'collection_config.json'
    if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
        raise ValueError('Collection settings changed; use a new output directory')
    settings_file.write_text(json.dumps(settings, indent=2) + '\n')
    tree = ET.parse(ROOT / '3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml')
    requested = args.route_ids.split(',')
    wanted = set(requested)
    for route in list(tree.getroot()):
        if route.get('id') not in wanted:
            tree.getroot().remove(route)
    if {r.get('id') for r in tree.getroot()} != wanted:
        raise ValueError('Requested route IDs were not all found')
    if len(requested) != len(wanted):
        raise ValueError('Duplicate route IDs')
    selected = {r.get('id'): r for r in tree.getroot()}
    tree.getroot()[:] = [selected[route_id] for route_id in requested]
    route_file = args.output / 'routes.xml'
    if route_file.exists() and ET.tostring(ET.parse(route_file).getroot()) != ET.tostring(tree.getroot()):
        raise ValueError('Collection route set changed; use a new output directory')
    tree.write(route_file, encoding='utf-8', xml_declaration=True)
    print('Joint collection staged:', settings, 'routes:', requested, flush=True)


def verify(args):
    import numpy as np
    import jsonpickle
    import jsonpickle.ext.numpy as jsonpickle_numpy
    from data import CARLA_Data
    from diagnostic_logger import verify_directory
    from recovery_data import build_manifest
    from learned_teacher_labels import verify_training_directory
    jsonpickle_numpy.register_handlers()
    settings = json.loads((args.output / 'collection_config.json').read_text())
    collection = Path(settings['collection_root'])
    result = json.loads((args.output / 'checkpoint_endpoint.json').read_text())
    checkpoint = result['_checkpoint']
    if result['entry_status'] != 'Finished' or checkpoint['progress'][0] != checkpoint['progress'][1]:
        raise ValueError('Student rollout evaluation did not finish')
    diagnostics = []
    native_data_checks = []
    for record in checkpoint['records']:
        if record['meta']['duration_game'] <= 0:
            raise ValueError('Collection failed before driving started')
        diagnostics.append(verify_directory(collection / 'diagnostics' / record['save_name']))
        native_data_checks.append(verify_training_directory(
            collection / 'Recovery_Train' / record['save_name'].replace('_rep0_', '_Rep0_')))
    manifest = collection / 'recovery_frames.txt'
    build_manifest(argparse.Namespace(recovery_root=collection, history_frames=8, skip_first=10, output=manifest))
    config = jsonpickle.decode((Path(settings['student_agent_dir']) / 'config.json').read_text())
    config.fmsr_recovery_root = str(collection)
    config.fmsr_recovery_manifest = str(manifest)
    config.fmsr_frame_manifest = ''
    config.augment = 0
    config.use_color_aug = 0
    config.lidar_aug_prob = 0
    dataset = CARLA_Data([str(collection / 'Recovery_Train')], config)
    if len(dataset) == 0:
        raise ValueError('Native training loader found zero samples')
    loader_checks = []
    for index in sorted(set([0, len(dataset)//2, len(dataset)-1])):
        sample = dataset[index]
        sample_start = int(dataset.sample_start[index])
        measurement_root = Path(dataset.measurements[index][0].decode())
        with gzip.open(measurement_root / f'{sample_start:04d}.json.gz', 'rt') as handle:
            measurement = json.load(handle)
        np.testing.assert_allclose(sample['route'][:config.fmsr_num_poses], measurement['teacher_checkpoints'], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(sample['target_speed_value'], measurement['target_speed'], rtol=1e-5)
        if not np.isfinite(sample['rgb']).all() or not np.isfinite(sample['lidar']).all():
            raise ValueError('Nonfinite native input tensors')
        loader_checks.append({'index': index, 'sample_frame': sample_start,
                              'rgb_shape': list(sample['rgb'].shape), 'lidar_shape': list(sample['lidar'].shape),
                              'route_shape': list(sample['route'].shape), 'target_speed_mps': float(sample['target_speed_value'])})
    raw_count = valid_count = 0
    reason_counts = Counter()
    teacher_stop_student_go = teacher_go_student_stop = 0
    for directory in (collection / 'Recovery_Train').iterdir():
        summary = json.loads((directory / 'collection_summary.json').read_text())
        raw_count += summary['saved_frames']; valid_count += summary['valid_teacher_frames']
        for line in (directory / 'recovery_events.jsonl').read_text().splitlines():
            event = json.loads(line)
            reason_counts.update(event['reasons'])
            teacher_stop_student_go += int('student_go_teacher_stop' in event['reasons'])
            teacher_go_student_stop += int('student_stop_teacher_go' in event['reasons'])
    report = {'status': 'passed', 'saved_frames': raw_count, 'valid_teacher_frames': valid_count,
              'selected_training_samples': len(dataset), 'diagnostics': diagnostics,
              'native_data_checks': native_data_checks,
              'native_loader_checks': loader_checks,
              'event_counts_by_reason': dict(reason_counts),
              'teacher_stop_student_go_samples': teacher_stop_student_go,
              'teacher_go_student_stop_samples': teacher_go_student_stop,
              'note': 'Learned teacher predictions on student states; prediction validity is not proof of safe recovery.'}
    (args.output / 'joint_collection_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Joint collection verified:', json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('prepare')
    stage.add_argument('--output', type=Path, required=True)
    stage.add_argument('--collection', type=Path, required=True)
    stage.add_argument('--student', type=Path, required=True)
    stage.add_argument('--teacher', type=Path, required=True)
    stage.add_argument('--route-ids', default='2204')
    check = commands.add_parser('verify')
    check.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        verify(args)


if __name__ == '__main__':
    main()
