"""Write independent inference weights and exact-resume optimizer/RNG companions."""
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
import torch


def asset_fingerprint(directory):
    return hashlib.sha256((Path(directory) / 'assets.json').read_bytes()).hexdigest()


def load_initial_checkpoint(path, model, migrate_noisy_residual=False):
    """Warm-start a fresh run, refusing silently substituted anchor/stat assets."""
    weights = torch.load(path, map_location='cpu', weights_only=True)
    current = model.state_dict()
    new_keys = {k for k in current if k.startswith('planner.clean_path_heads.') or k == 'planner.clean_prior_contract'}
    expected = set(current) - new_keys if migrate_noisy_residual else set(current)
    if migrate_noisy_residual and not new_keys:
        raise ValueError('Migration requires the versioned clean-prior architecture')
    if set(weights) != expected or any(weights[k].shape != current[k].shape for k in expected):
        raise ValueError('Initialization checkpoint does not match the model architecture')
    for name in ('planner.plan_anchor', 'planner.mean', 'planner.std'):
        if not torch.equal(weights[name].cpu(), current[name].cpu()):
            raise ValueError(f'Initialization checkpoint uses different training assets: {name}')
    # Only an explicit migration may retain freshly initialized clean-path heads.
    if migrate_noisy_residual:
        for head in model.planner.clean_path_heads:
            if torch.count_nonzero(head[-1].weight) or torch.count_nonzero(head[-1].bias):
                raise ValueError('Migration requires fresh zero-output correction heads')
    if 'planner.clean_prior_contract' in weights and int(weights['planner.clean_prior_contract']) != 1:
        raise ValueError('Unsupported clean-prior prediction contract')
    if 'author_v8_pdm_contract' in weights and int(weights['author_v8_pdm_contract']) != 1:
        raise ValueError('Unsupported author-v8 PDM contract')
    if 'planner.flow_matching_contract' in weights and int(weights['planner.flow_matching_contract']) != 1:
        raise ValueError('Unsupported flow-matching prediction contract')
    if 'planner.flow_cr_contract' in weights and int(weights['planner.flow_cr_contract']) != 1:
        raise ValueError('Unsupported FM+CR prediction contract')
    model.load_state_dict({**current, **weights}, strict=True)
    return {'migration': 'noisy_residual_to_tfpp_clean_prior_v1' if migrate_noisy_residual else 'none',
            'retained_new_tensors': sorted(new_keys) if migrate_noisy_residual else []}


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, output, fingerprint, config_fingerprint):
    output = Path(output)
    bare = model.module if hasattr(model, 'module') else model
    atomic_save(bare.state_dict(), output / f'model_{epoch:04d}.pth')
    state = {'epoch': epoch, 'global_step': global_step, 'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict(), 'asset_fingerprint': fingerprint,
             'config_fingerprint': config_fingerprint, 'torch_rng': torch.get_rng_state(),
             'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    atomic_save(state, output / f'resume_{epoch:04d}.pt')


def load_resume(path, model, optimizer, scheduler, fingerprint, config_fingerprint):
    path = Path(path)
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['asset_fingerprint'] != fingerprint or state['config_fingerprint'] != config_fingerprint:
        raise ValueError('Exact resume requires unchanged data assets and training configuration')
    model_path = path.parent / f'model_{state["epoch"]:04d}.pth'
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True), strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    torch.set_rng_state(state['torch_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['epoch'] + 1, state['global_step']
