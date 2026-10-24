"""Use the native Garage RGB/LAZ processing without reading old training configs."""
import copy
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset

from . import bootstrap
from data import CARLA_Data


def validate_assets(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'assets.json').read_text())
    if metadata['statistics_source'] != 'training_split_only':
        raise ValueError('Normalization/anchors must be constructed from training frames only')
    if set(metadata['train_groups']) & set(metadata['validation_groups']):
        raise ValueError('Train/validation route groups overlap')
    for name, digest in metadata['sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Asset changed since preparation: {name}')
    train = set((directory / 'train_frames.txt').read_text().splitlines())
    validation = set((directory / 'validation_frames.txt').read_text().splitlines())
    if train & validation or not train or not validation:
        raise ValueError('Train/validation manifests overlap or are empty')
    return metadata


def create_dataset(run, config, validation=False):
    config = copy.deepcopy(config)
    manifest = Path(run.assets_dir) / ('validation_frames.txt' if validation else 'train_frames.txt')
    config.fmsr_frame_manifest = str(manifest)
    config.detect_boxes = 0  # perception detections remain frozen; their labels are unnecessary
    config.estimate_class_distributions = 0
    config.lidar_aug_prob = 0 if validation else config.lidar_aug_prob
    if validation:
        config.augment = config.use_color_aug = 0
    scenarios = sorted({line.split('/')[0] for line in manifest.read_text().splitlines() if line})
    roots = [str(Path(run.data_root) / scenario) for scenario in scenarios]
    return CARLA_Data(roots, config, validation=False)


def validation_subset(dataset, max_samples, seed=137):
    """Cover all available scenarios before sampling the rest of the full pool."""
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    groups = defaultdict(list)
    for index, paths in enumerate(dataset.measurements):
        groups[Path(paths[0].decode()).parent.parent.name].append(index)
    if max_samples < len(groups):
        raise ValueError('Validation sample budget is smaller than the scenario count')
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice(groups[name])) for name in sorted(groups)]
    pool = np.setdiff1d(np.arange(len(dataset)), selected, assume_unique=True)
    selected.extend(rng.choice(pool, max_samples - len(selected), replace=False).tolist())
    return Subset(dataset, sorted(selected))
