"""Read and validate synchronous teacher labels and exact student network inputs."""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class PairedDataset(Dataset):
    def __init__(self, manifest):
        self.paths = [Path(p) for p in Path(manifest).read_text().splitlines() if p]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as data:
            return {key: torch.from_numpy(data[key].copy()) for key in data.files}


def verify(directory):
    directory = Path(directory)
    samples = []
    last_frame = None
    count = 0
    for line in (directory / 'pairs.jsonl').read_text().splitlines():
        row = json.loads(line)
        if last_frame is not None and row['simulation_frame'] <= last_frame:
            raise ValueError('Non-monotonic synchronous frames')
        last_frame = row['simulation_frame']
        for relative in row['teacher_sensor_files'].values():
            path = directory / relative
            if path.suffix == '.jpg':
                if cv2.imread(str(path)) is None:
                    raise ValueError(f'Invalid teacher image: {path}')
            else:
                with np.load(path, allow_pickle=False) as data:
                    if not np.isfinite(data['value']).all():
                        raise ValueError(f'Invalid teacher sensor: {path}')
        if row['training_sample']:
            path = directory / row['training_sample']
            with np.load(path, allow_pickle=False) as data:
                for key in ('rgb', 'lidar_bev', 'target_point', 'ego_vel', 'command', 'teacher_trajectory', 'teacher_target_speed'):
                    if key not in data or not np.isfinite(data[key]).all():
                        raise ValueError(f'Missing or invalid {key}: {path}')
                if int(data['simulation_frame']) != row['simulation_frame']:
                    raise ValueError('Input/label frame mismatch')
                if not np.allclose(data['teacher_trajectory'], np.asarray(row['teacher']['trajectory_ego_m']).reshape(-1, 2)):
                    raise ValueError('Saved trajectory differs from same-frame teacher output')
                if not np.allclose(data['teacher_target_speed'], np.asarray(row['teacher']['target_speed_mps']).reshape(1)):
                    raise ValueError('Saved speed differs from same-frame teacher output')
            samples.append(str(path.resolve()))
        count += 1
    if not samples:
        raise ValueError('No trainable synchronous samples')
    summary = json.loads((directory / 'paired_summary.json').read_text())
    if len(samples) != summary['training_sample_count']:
        raise ValueError('Training sample count mismatch')
    manifest = directory / 'paired_frames.txt'
    manifest.write_text(''.join(p + '\n' for p in samples))
    dataset = PairedDataset(manifest)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=min(2, len(dataset)))))
    report = {'ok': True, 'frames': count, 'training_samples': len(samples),
              'batch_shapes': {k: list(v.shape) for k, v in batch.items()}}
    (directory / 'paired_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Paired data verified:', report, flush=True)
    return samples


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1])
    paths = []
    for directory in sorted((root / 'diagnostics').iterdir()):
        paths.extend(verify(directory))
    (root / 'paired_frames.txt').write_text(''.join(p + '\n' for p in paths))
