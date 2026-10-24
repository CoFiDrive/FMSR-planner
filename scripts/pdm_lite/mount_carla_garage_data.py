"""Expose compressed CARLA scenarios through a read-only local dataset view."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def dataset_view(data_root, view_root):
    data_root = data_root.resolve(strict=True)
    image_root = data_root.parent / 'carla_garage_squashfs'
    images = {image.stem: image.resolve() for image in sorted(image_root.glob('*.sqfs'))}
    if not images:
        return data_root
    view_root.mkdir(parents=True, exist_ok=True)
    view_root = view_root.resolve()
    source = {'data_root': str(data_root),
              'images': {name: str(image) for name, image in images.items()}}
    with (view_root / '.mount.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        source_file = view_root / 'dataset_sources.json'
        if source_file.exists() and json.loads(source_file.read_text()) != source:
            raise ValueError(f'Dataset view belongs to different sources: {view_root}')
        for name, image in images.items():
            destination = view_root / name
            if destination.is_symlink():
                raise ValueError(f'Expected a mount directory: {destination}')
            destination.mkdir(exist_ok=True)
            if not os.path.ismount(destination):
                if any(destination.iterdir()):
                    raise ValueError(f'Refusing to hide existing files: {destination}')
                subprocess.run(['squashfuse', '-o', 'ro', str(image), str(destination)],
                               check=True, stdout=sys.stderr, stderr=sys.stderr)
                if not os.path.ismount(destination):
                    raise RuntimeError(f'SquashFS mount failed: {image}')
            print(f'Read-only dataset scenario: {destination}', file=sys.stderr)
        for scenario in sorted(data_root.iterdir()):
            if not scenario.is_dir() or scenario.name in images:
                continue
            destination = view_root / scenario.name
            if destination.exists() or destination.is_symlink():
                if not destination.is_symlink() or destination.resolve() != scenario.resolve():
                    raise ValueError(f'Dataset view contains a conflicting entry: {destination}')
            else:
                destination.symlink_to(scenario, target_is_directory=True)
        source_file.write_text(json.dumps(source, indent=2) + '\n')
    return view_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--view-root', type=Path, required=True)
    args = parser.parse_args()
    print(dataset_view(args.data_root, args.view_root))


if __name__ == '__main__':
    main()
