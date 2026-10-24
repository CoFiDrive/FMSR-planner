"""Select comparable degraded routes and validate replay diagnostic artifacts."""

import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]


def valid_rollout(record):
    status = record.get('status', '')
    return (record.get('meta', {}).get('duration_game', 0) > 0
            and (status in ('Completed', 'Perfect', 'Failed') or status.startswith('Failed -'))
            and not any(word in status.lower() for word in ('crash', 'simulation', 'set up', 'setup')))


def select_routes(args):
    baseline = json.loads(args.baseline.read_text())['_checkpoint']
    current = json.loads(args.current.read_text())['_checkpoint']
    old = {record['route_id']: record for record in baseline['records']}
    selected, excluded = [], []
    for record in current['records']:
        route_id = record['route_id']
        previous = old.get(route_id)
        if previous is None or not valid_rollout(record) or not valid_rollout(previous):
            excluded.append({'route_id': route_id, 'status': record.get('status')})
            continue
        worse = record['scores']['score_composed'] < previous['scores']['score_composed'] - 1e-6
        failed = record['status'] not in ('Completed', 'Perfect')
        match = {'worse': worse, 'failed': failed, 'union': worse or failed}[args.selection]
        if args.route_id:
            requested = args.route_id if args.route_id.startswith('RouteScenario_') else f'RouteScenario_{args.route_id}_rep0'
            match = match and route_id == requested
        if match:
            selected.append({'route_id': route_id, 'scenario_name': record.get('scenario_name'),
                             'old_scores': previous['scores'], 'new_scores': record['scores'],
                             'old_status': previous['status'], 'new_status': record['status'],
                             'worse': worse, 'failed': failed})
    ids = set()
    for record in selected:
        match = re.fullmatch(r'RouteScenario_(.+)_rep0', record['route_id'])
        if not match:
            raise ValueError(f"Unsupported route/repetition: {record['route_id']}")
        ids.add(match.group(1))
    tree = ET.parse(args.routes)
    root = tree.getroot()
    for route in list(root.findall('route')):
        if route.get('id') not in ids:
            root.remove(route)
    found = {route.get('id') for route in root.findall('route')}
    if not ids or found != ids or len(root.findall('route')) != len(selected):
        raise ValueError(f'Empty, duplicate or missing route selection: wanted={ids}, found={found}')
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Selection snapshot already exists; choose a new output directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(args.output, encoding='utf-8', xml_declaration=True)
    args.report.write_text(json.dumps({
        'selection': args.selection, 'baseline': str(args.baseline.resolve()),
        'current': str(args.current.resolve()), 'current_progress': current['progress'],
        'route_count': len(selected), 'routes': selected,
        'excluded_infrastructure_or_invalid_records': excluded,
    }, indent=2, ensure_ascii=False) + '\n')
    print(f'Selected {len(selected)} {args.selection} routes -> {args.output}')
    print(f'Excluded {len(excluded)} invalid/infrastructure records; snapshot -> {args.report}')


def verify_run(args):
    sys.path.insert(0, str(ROOT / '3rd_party/carla_garage/team_code'))
    from diagnostic_logger import verify_directory
    source = args.output_dir / 'agent/source.json'
    metadata = json.loads(source.read_text()) if source.exists() else {}
    diagnostic_root = Path(metadata.get('diagnostic_root', str(args.output_dir)))
    directories = sorted((diagnostic_root / 'diagnostics').glob('*'))
    if not directories:
        raise ValueError('No diagnostic route directories found')
    results, failures = [], []
    for directory in directories:
        if not directory.is_dir():
            continue
        try:
            results.append(verify_directory(directory))
        except (OSError, ValueError, KeyError) as error:
            failures.append({'directory': str(directory), 'error': str(error)})
    checkpoint = args.output_dir / 'checkpoint_endpoint.json'
    if checkpoint.exists():
        data = json.loads(checkpoint.read_text())
        done, total = data['_checkpoint']['progress']
        if data.get('entry_status') != 'Finished' or done != total:
            failures.append({'error': f'Evaluation is incomplete: {done}/{total}, {data.get("entry_status")}'})
        for record in data['_checkpoint']['records']:
            if not valid_rollout(record):
                continue
            name = record.get('save_name')
            if not name or not any(Path(result['directory']).name == name for result in results):
                failures.append({'route_id': record['route_id'], 'error': 'Completed rollout has no valid diagnostics'})
    report = {'verified_routes': len(results), 'routes': results, 'failures': failures}
    (args.output_dir / 'diagnostic_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    select = commands.add_parser('select')
    for name in ('baseline', 'current', 'routes', 'output', 'report'):
        select.add_argument(f'--{name}', type=Path, required=True)
    select.add_argument('--selection', choices=('worse', 'failed', 'union'), default='worse')
    select.add_argument('--route-id', default='')
    verify = commands.add_parser('verify')
    verify.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'select':
        select_routes(args)
    else:
        verify_run(args)


if __name__ == '__main__':
    main()
