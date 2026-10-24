"""Validate deployment sampling from an existing checkpoint without CARLA."""
import argparse
import json
from pathlib import Path

import jsonpickle
import jsonpickle.ext.numpy as jnp
import torch

from .configuration import RunConfig
from .dataset import create_dataset, validate_assets
from .generation_validation import generation_loader, evaluate_generation
from .model import FMSRModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=128, help='0 validates the full pool')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seeds', type=int, nargs='+', default=[137, 271, 811])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise ValueError('Choose a fresh output path to preserve previous diagnostics')
    root = args.checkpoint.resolve().parent
    run = RunConfig(**json.loads((root/'run_config.json').read_text()))
    validate_assets(run.assets_dir)
    jnp.register_handlers()
    config = jsonpickle.decode((root/'config.json').read_text())
    device = torch.device(args.device)
    model = FMSRModel(config).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True), strict=True)
    dataset = create_dataset(run, config, validation=True)
    report = evaluate_generation(model, generation_loader(dataset, len(dataset) if args.samples == 0 else args.samples,
                                                         args.batch_size), device, args.seeds)
    report['checkpoint'] = str(args.checkpoint.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['mean_over_seeds'], indent=2), flush=True)


if __name__ == '__main__':
    main()
