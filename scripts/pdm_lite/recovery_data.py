#!/usr/bin/env python3
"""Choose non-test CARLA routes and select validated recovery frames."""

import argparse
import gzip
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET


def route_signature(path):
    routes = ET.parse(path).getroot().findall('route')
    if len(routes) != 1:
        raise ValueError(f'Expected one route per collection XML: {path}')
    route = routes[0]
    points = []
    for trigger in route.findall('./scenarios/scenario/trigger_point'):
        points.append((float(trigger.get('x')), float(trigger.get('y'))))
    return route.get('town'), points


def overlaps_test(path, test_ids, test_points, radius_m=25.0):
    # The CARLA Garage data XML stem is the scenario ID, while its embedded
    # route ID is reused across thousands of files.
    scenario_id = path.stem.split('_')[0]
    if scenario_id in test_ids:
        return True
    town, triggers = route_signature(path)
    limit2 = radius_m * radius_m
    return any((x - tx)**2 + (y - ty)**2 < limit2
               for x, y in triggers for tx, ty in test_points.get(town, ()))


def choose_routes(args):
    test_ids, test_points = test_signatures(args.test_routes)
    candidates = list(args.route_root.rglob('*.xml'))
    if args.scenario:
        candidates = [p for p in candidates if p.parent.name in args.scenario]
    if args.town:
        candidates = [p for p in candidates if args.town in p.parts]
    random.Random(args.seed).shuffle(candidates)
    selected = []
    for path in candidates:
        if not overlaps_test(path, test_ids, test_points, args.exclusion_radius_m):
            selected.append(path.resolve())
            if len(selected) == args.limit:
                break
    if len(selected) < args.limit:
        raise ValueError(f'Only {len(selected)} eligible non-test routes, requested {args.limit}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(str(p) + '\n' for p in selected), encoding='utf-8')
    print(f'Selected {len(selected)} training routes -> {args.output}')


def test_signatures(path):
    tree = ET.parse(path)
    test_ids = {r.get('id') for r in tree.getroot().findall('route')}
    test_points = {}
    for route in tree.getroot().findall('route'):
        town = route.get('town')
        test_points.setdefault(town, []).extend(
            (float(p.get('x')), float(p.get('y')))
            for p in route.findall('./scenarios/scenario/trigger_point'))
    return test_ids, test_points


def valid_label(route, frame):
    paths = [route / 'lidar' / f'{frame:04d}.laz',
             route / 'rgb' / f'{frame:04d}.jpg',
             route / 'measurements' / f'{frame:04d}.json.gz']
    if not all(p.is_file() for p in paths):
        return False
    with gzip.open(paths[2], 'rt', encoding='utf-8') as handle:
        label = json.load(handle)
    speed = label.get('target_speed')
    trajectory = label.get('route')
    if label.get('label_source') == 'learned_teacher' and not label.get('label_valid', False):
        return False
    if label.get('training_eligible') is False:
        return False
    return (isinstance(speed, (int, float)) and math.isfinite(speed) and
            isinstance(trajectory, list) and len(trajectory) >= 10 and
            all(len(point) >= 2 and all(math.isfinite(float(v)) for v in point[:2])
                for point in trajectory[:10]))


def build_manifest(args):
    if not args.recovery_root.is_dir():
        raise FileNotFoundError(args.recovery_root)
    selected = set()
    event_counts = {}
    for event_file in sorted(args.recovery_root.glob('*/*/recovery_events.jsonl')):
        route = event_file.parent
        if route.name.startswith('FAILED_'):
            continue
        rows = [json.loads(line) for line in event_file.read_text(encoding='utf-8').splitlines() if line]
        frames = {int(row['frame']) for row in rows}
        triggers = [row for row in rows if row.get('reasons')]
        if not triggers:
            continue
        event_counts[str(route)] = len(triggers)
        wanted = set()
        for row in triggers:
            event = int(row['frame'])
            wanted.update(range(max(args.skip_first, event - args.history_frames), event + 1))
        for frame in sorted(wanted & frames):
            if valid_label(route, frame):
                selected.add(f'{route.parent.name}/{route.name}/{frame:04d}')
    if not selected:
        raise ValueError('No valid recovery frames; inspect rollout logs and event triggers')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(key + '\n' for key in sorted(selected)), encoding='utf-8')
    report = {'selected_frames': len(selected), 'trigger_counts_by_route': event_counts,
              'history_frames': args.history_frames}
    args.output.with_suffix('.summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Selected {len(selected)} recovery frames -> {args.output}')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    choose = sub.add_parser('choose-routes')
    choose.add_argument('--route-root', type=Path, required=True)
    choose.add_argument('--test-routes', type=Path, required=True)
    choose.add_argument('--town', default='50x38_Town12')
    choose.add_argument('--scenario', nargs='*', default=[])
    choose.add_argument('--limit', type=int, default=8)
    choose.add_argument('--seed', type=int, default=0)
    choose.add_argument('--exclusion-radius-m', type=float, default=25.0)
    choose.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('check-route')
    check.add_argument('--route', type=Path, required=True)
    check.add_argument('--test-routes', type=Path, required=True)
    check.add_argument('--exclusion-radius-m', type=float, default=25.0)
    select = sub.add_parser('build-manifest')
    select.add_argument('--recovery-root', type=Path, required=True)
    select.add_argument('--history-frames', type=int, default=8)
    select.add_argument('--skip-first', type=int, default=10)
    select.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'choose-routes':
        choose_routes(args)
    elif args.command == 'check-route':
        ids, points = test_signatures(args.test_routes)
        if overlaps_test(args.route, ids, points, args.exclusion_radius_m):
            parser.error(f'Route overlaps the 220 test routes: {args.route}')
        print(f'Training route isolation passed: {args.route}')
    else:
        build_manifest(args)


if __name__ == '__main__':
    main()
