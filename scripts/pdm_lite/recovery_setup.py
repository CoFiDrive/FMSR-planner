"""Stage recovery assets after mount changes and recognize real rollouts."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


ASSET_FIELDS = ('fmsr_anchor_path', 'fmsr_stats_path', 'trajectory_stats_path')


def stage_config(checkpoint, destination, asset_dir=None, media_root=Path('/media')):
    config = json.loads((checkpoint.parent / 'config.json').read_text())
    required = {Path(config[key]).name for key in ASSET_FIELDS if config.get(key)}
    if asset_dir is None and any(not Path(config[key]).is_file()
                                 for key in ASSET_FIELDS if config.get(key)):
        candidates = [path for path in media_root.glob('*/*/pdm_lite_assets')
                      if all((path / name).is_file() for name in required)]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f'Expected one mounted PDM asset directory, found {candidates}. '
                'Set PDM_ASSET_DIR to the original training assets.')
        asset_dir = candidates[0]
    for key in ASSET_FIELDS:
        if not config.get(key):
            continue
        original = Path(config[key])
        resolved = (asset_dir / original.name if asset_dir is not None else original).resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        config[key] = str(resolved)
        print(f'{key}: {resolved}')
    poses = config['fmsr_num_poses'] + int(config['diffusion_speed'])
    anchors = np.load(config['fmsr_anchor_path'], allow_pickle=False)
    if anchors.ndim != 3 or anchors.shape[1:] != (poses, 2) or not np.isfinite(anchors).all():
        raise ValueError(f'Invalid training anchors: {anchors.shape}')
    for key in ('fmsr_stats_path', 'trajectory_stats_path'):
        if config.get(key):
            with np.load(config[key], allow_pickle=False) as stats:
                mean, std = stats['mean'], stats['std']
                if (mean.shape != (poses, 2) or std.shape != (poses, 2)
                        or not np.isfinite(mean).all() or not np.isfinite(std).all()
                        or (std <= 0).any()):
                    raise ValueError(f'Invalid training statistics: {config[key]}')
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / f'.config.{os.getpid()}.json'
    temporary.write_text(json.dumps(config, indent=4) + '\n')
    temporary.replace(destination / 'config.json')


def route_complete(result):
    try:
        data = json.loads(result.read_text())
        checkpoint = data['_checkpoint']
        done, total = checkpoint['progress']
        records = checkpoint['records']
        return (data.get('entry_status') == 'Finished' and total > 0 and done == total
                and len(records) == total
                and all(record.get('meta', {}).get('duration_game', 0) > 0
                        and (record.get('status') in ('Perfect', 'Completed', 'Failed')
                             or record.get('status', '').startswith('Failed -'))
                        and not any(term in record.get('status', '').lower()
                                    for term in ('set up', 'setup', 'crash', 'simulation'))
                        for record in records))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    stage = commands.add_parser('stage')
    stage.add_argument('--checkpoint', type=Path, required=True)
    stage.add_argument('--agent-dir', type=Path, required=True)
    stage.add_argument('--asset-dir', default='')
    status = commands.add_parser('route-complete')
    status.add_argument('result', type=Path)
    args = parser.parse_args()
    if args.command == 'stage':
        stage_config(args.checkpoint, args.agent_dir,
                     Path(args.asset_dir) if args.asset_dir else None)
    else:
        raise SystemExit(0 if route_complete(args.result) else 1)


if __name__ == '__main__':
    main()
