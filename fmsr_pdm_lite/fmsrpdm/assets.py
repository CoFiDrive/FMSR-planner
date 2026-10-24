"""Training-only PDM anchors/statistics and disjoint route-group frame manifests."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from . import bootstrap
from .speed import MAX_TARGET_SPEED_MPS
from route_geometry import prepare_route, smooth_route


def read_measurement(path):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def usable_route(route):
    result = route / 'results.json.gz'
    if route.name.startswith('FAILED_') or not result.is_file():
        return False
    record = read_measurement(result)
    status = record.get('status', '')
    if status == 'Failed' or any(x in status.lower() for x in ('crash', 'set up', 'setup')):
        return False
    infractions = record.get('infractions', {})
    only_minimum_speed = record.get('num_infractions', -1) == len(infractions.get('min_speed_infractions', []))
    return record.get('scores', {}).get('score_composed', 0) >= 100 or only_minimum_speed


def group_key(route):
    # Keep repetitions/weather variants of the same route out of opposite splits.
    match = re.search(r'Town(\d+)_Rep\d+_(\d+)_', route.name)
    return f'Town{match[1]}/route{match[2]}' if match else route.parent.name + '/' + route.name


def split_routes(routes, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be in (0,1)')
    groups = defaultdict(list)
    for route in routes:
        groups[group_key(route)].append(route)
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError('At least two distinct route groups are needed')
    np.random.default_rng(seed).shuffle(keys)
    validation_keys = set(keys[:min(len(keys)-1, max(1, round(len(keys)*fraction)))])
    return [r for r in routes if group_key(r) not in validation_keys], [r for r in routes if group_key(r) in validation_keys]


def label_state(measurement):
    route = smooth_route(prepare_route(np.asarray(measurement['route'], dtype=np.float32), 20), 20)[:10]
    speed = float(measurement['target_speed'])
    if route.shape != (10, 2) or not np.isfinite(route).all() or not math.isfinite(speed) or not 0 <= speed <= MAX_TARGET_SPEED_MPS:
        raise ValueError('Invalid PDM route/speed label')
    return np.vstack([route, [speed, 0]]).astype(np.float32)


def angles(state, definition='waypoint_bearing'):
    if definition == 'waypoint_bearing':
        return np.arctan2(state[:10, 1], state[:10, 0])
    if definition != 'segment_heading':
        raise ValueError('Unknown angle definition')
    delta = np.diff(np.vstack([np.zeros((1, 2)), state[:10]]), axis=0)
    return np.arctan2(delta[:, 1], delta[:, 0])


def select_frames(routes, root, rng, ordinary_fraction=.14, angle_definition='waypoint_bearing'):
    states, manifest, counts = [], [], {'scanned': 0, 'key_frames': 0, 'invalid': 0}
    for index, route in enumerate(routes, 1):
        previous_state = None
        for path in sorted((route / 'measurements').glob('*.json.gz')):
            if int(path.name.split('.')[0]) < 10:
                continue  # match the native TF++ initialization exclusion
            try:
                state = label_state(read_measurement(path))
            except (ValueError, KeyError):
                counts['invalid'] += 1
                continue
            if not all((route / folder / (path.name.split('.')[0] + extension)).is_file()
                       for folder, extension in [('rgb', '.jpg'), ('rgb_augmented', '.jpg'), ('lidar', '.laz')]):
                counts['invalid'] += 1
                continue
            critical = previous_state is None
            if previous_state is not None:
                delta = (angles(state, angle_definition) - angles(previous_state, angle_definition) + np.pi) % (2*np.pi) - np.pi
                critical = abs(float(state[-1, 0] - previous_state[-1, 0])) > .1 or np.abs(delta).max() > math.radians(.5)
            previous_state = state
            counts['scanned'] += 1
            if critical or rng.random() < ordinary_fraction:
                states.append(state)
                manifest.append(str(route.relative_to(root)) + '/' + path.name.split('.')[0])
                counts['key_frames'] += int(critical)
        if index % 100 == 0:
            print(f'Processed {index}/{len(routes)} routes; retained {len(states)} frames', flush=True)
    if not states:
        raise ValueError('No usable filtered samples in this split')
    return np.stack(states), manifest, counts


def build(args):
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new empty asset directory; existing assets are immutable')
    if not 0 <= args.ordinary_fraction <= 1:
        raise ValueError('ordinary_fraction must be in [0,1]')
    routes = [r for r in sorted(root.glob('*/*')) if r.is_dir() and usable_route(r)]
    if args.max_routes:
        routes = routes[:args.max_routes]
    train, validation = split_routes(routes, args.validation_fraction, args.seed)
    definition = getattr(args, 'angle_definition', 'waypoint_bearing')
    train_states, train_manifest, train_counts = select_frames(train, root, np.random.default_rng(args.seed), args.ordinary_fraction, definition)
    _, validation_manifest, validation_counts = select_frames(validation, root, np.random.default_rng(args.seed+1), args.ordinary_fraction, definition)
    if len(train_states) < args.clusters:
        raise ValueError('Not enough training frames for the requested anchor count')
    clusters = MiniBatchKMeans(n_clusters=args.clusters, random_state=args.seed,
                              batch_size=8192, n_init=10, reassignment_ratio=.01)
    anchors = clusters.fit(train_states.reshape(len(train_states), -1)).cluster_centers_.reshape(args.clusters, 11, 2)
    anchors[:, -1, 1] = 0
    mean = train_states.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_states.std(0, dtype=np.float64), 1e-4).astype(np.float32)
    mean[-1, 1], std[-1, 1] = 0, 1
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / 'anchors.npy', anchors.astype(np.float32))
    np.savez(output / 'stats.npz', mean=mean, std=std)
    for name, lines in [('train_frames.txt', train_manifest), ('validation_frames.txt', validation_manifest)]:
        (output / name).write_text('\n'.join(lines) + '\n')
    metadata = {'data_root': str(root), 'seed': args.seed, 'clusters': args.clusters,
                'training_routes': len(train), 'validation_routes': len(validation),
                'train_groups': sorted({group_key(r) for r in train}),
                'validation_groups': sorted({group_key(r) for r in validation}),
                'training_frames': len(train_manifest), 'validation_frames': len(validation_manifest),
                'train_counts': train_counts, 'validation_counts': validation_counts,
                'statistics_source': 'training_split_only', 'debug_subset': bool(args.max_routes),
                'filter': {'speed_change_mps': .1, 'angle_change_degrees': .5, 'ordinary_fraction': args.ordinary_fraction,
                           'angle_definition': definition},
                'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output / 'assets.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k:v for k,v in metadata.items() if k not in ('train_groups', 'validation_groups', 'sha256')}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clusters', type=int, default=60)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-fraction', type=float, default=.05)
    parser.add_argument('--ordinary-fraction', type=float, default=.14)
    parser.add_argument('--angle-definition', choices=['waypoint_bearing', 'segment_heading'], default='waypoint_bearing')
    parser.add_argument('--max-routes', type=int, default=0)
    build(parser.parse_args())


if __name__ == '__main__':
    main()
