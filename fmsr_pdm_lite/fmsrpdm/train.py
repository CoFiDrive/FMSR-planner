"""Paper baseline training, with independent validation and resumable checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .configuration import RunConfig, model_config
from .dataset import create_dataset, validate_assets, validation_subset
from .model import FMSRModel
from .checkpoint import asset_fingerprint, load_resume, save_checkpoint, load_initial_checkpoint
from .bootstrap import PACKAGE_ROOT, WORKSPACE_ROOT
from .generation_validation import generation_loader, evaluate_generation, preserve_validation_rng, validation_mode
from .regression_health import reset_regression_refiners
from .clean_prior import prediction_base, TFPP_CLEAN_PRIOR


def worker_seed(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def tensor_inputs(batch, device):
    return dict(rgb=batch['rgb'].to(device, dtype=torch.float32),
                lidar_bev=batch['lidar'].to(device, dtype=torch.float32),
                target_point=batch['target_point'].to(device, dtype=torch.float32),
                target_point_next=batch['target_point_next'].to(device, dtype=torch.float32),
                ego_vel=batch['speed'].to(device, dtype=torch.float32).reshape(-1, 1),
                command=batch['command'].to(device, dtype=torch.float32),
                target_route=batch['route'].to(device, dtype=torch.float32),
                target_speed=batch['target_speed_value'].to(device, dtype=torch.float32), return_loss=True)


def evaluate(model, loader, device, limit, amp):
    totals, samples = {}, 0
    # A private generator makes the noisy validation objective independent of training RNG.
    generator = torch.Generator(device=device).manual_seed(137)
    with validation_mode(model), preserve_validation_rng(), torch.inference_mode():
        for index, batch in enumerate(loader):
            if limit and index >= limit:
                break
            count = len(batch['rgb'])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                result = model(**tensor_inputs(batch, device), generator=generator)
            values = dict(result['metrics'], **result['losses'])
            for key, value in values.items():
                totals[key] = totals.get(key, 0) + float(value)*count
            samples += count
    if not samples:
        raise ValueError('Validation loader is empty')
    return {key: value/samples for key, value in totals.items()}


def generation_choice(summary):
    if not summary.get('completed_full_epoch') or 'validation_generation' not in summary:
        return None
    generated = summary['validation_generation']
    return {'epoch': summary['epoch'], 'route_ADE_m': generated['mean_over_seeds']['route_ADE_m'],
            'criterion': 'mean_route_ADE_over_fixed_generation_seeds',
            'checkpoint': f'model_{summary["epoch"]:04d}.pth', 'report': generated['report'],
            'metrics': generated['mean_over_seeds'],
            'scope': 'development_validation_selection_not_closed_loop_or_final_holdout'}


def best_generation_before(output, start_epoch):
    choices = []
    for path in sorted(Path(output).glob('epoch_*.json')):
        summary = json.loads(path.read_text())
        choice = generation_choice(summary) if summary['epoch'] < start_epoch else None
        if choice is not None: choices.append(choice)
    return min(choices, key=lambda row: row['route_ADE_m']) if choices else None


def train(run, resume='', preflight=False, allow_debug_assets=False):
    run.validate()
    metadata = validate_assets(run.assets_dir)
    if metadata['debug_subset'] and not allow_debug_assets:
        raise ValueError('Debug assets require explicit --allow-debug-assets; rebuild full assets for paper experiments')
    if Path(metadata['data_root']).resolve() != Path(run.data_root).resolve():
        raise ValueError('The data root differs from the one used to build the assets')
    random.seed(run.seed); np.random.seed(run.seed); torch.manual_seed(run.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run.seed)
    device = torch.device(run.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a smoke check')
    config = model_config(run)
    model = FMSRModel(config)
    loading = model.load_tfpp(run.tfpp_checkpoint)
    if not resume:
        if run.init_checkpoint:
            loading['initialization'] = load_initial_checkpoint(run.init_checkpoint, model, run.migrate_noisy_residual)
            loading['initialization_checkpoint'] = str(Path(run.init_checkpoint).resolve())
        if run.reset_regression_layers:
            loading['regression_layers_reset'] = reset_regression_refiners(model.planner,
                                                                           run.reset_regression_layers, run.seed)
    loading['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Frozen TF++:', loading, flush=True)
    model.to(device).train()
    training_data = create_dataset(run, config)
    validation_data = create_dataset(run, config, validation=True)
    if not len(training_data) or not len(validation_data):
        raise ValueError('Empty train/validation dataset')
    for dataset in (training_data, validation_data):
        for index in (0, len(dataset)-1):
            sample = dataset[index]
            if not np.isfinite(sample['lidar']).all() or not np.isfinite(sample['route']).all():
                raise ValueError('Nonfinite input/labels')
    if preflight:
        print(json.dumps({'preflight': 'passed', 'train_samples': len(training_data),
                          'validation_samples': len(validation_data), 'loading': loading}), flush=True)
        return
    output = Path(run.output_dir)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError('Use a fresh output directory or --resume to avoid replacing previous results')
    output.mkdir(parents=True, exist_ok=True)
    config_dict = asdict(run)
    configuration_digest = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()
    if resume and Path(resume).resolve().parent != output.resolve():
        raise ValueError('Exact resume must use the original output directory')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=run.lr,
                                  weight_decay=run.weight_decay,
                                  amsgrad=run.implementation_profile == 'author_v8_pdm')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, run.cosine_t0, T_mult=run.cosine_t_mult)
    fingerprint = asset_fingerprint(run.assets_dir)
    start_epoch = global_step = 0
    if resume:
        start_epoch, global_step = load_resume(resume, model, optimizer, scheduler, fingerprint, configuration_digest)
    jsonpickle_numpy.register_handlers()
    run.save(output / 'run_config.json')
    (output / 'config.json').write_text(jsonpickle.encode(config, indent=2) + '\n')
    provenance = {'upstream': json.loads((PACKAGE_ROOT/'upstream/sources.json').read_text()),
        'asset_fingerprint': fingerprint, 'configuration_fingerprint': configuration_digest,
        'TFPP_checkpoint': str(Path(run.tfpp_checkpoint).resolve()), 'paper_score_reproduced': False,
        'prediction_base': prediction_base(config), 'training_scope': run.training_scope,
        'implementation_profile': run.implementation_profile,
        'trajectory_generator': run.trajectory_generator,
        'regression_supervision': run.regression_supervision, 'initialization': loading}
    if run.implementation_profile == 'author_v8_pdm':
        attachment = WORKSPACE_ROOT/'code_sharing/code_sharing/FMSR_temp.zip'
        provenance['author_v8_adapter'] = {
            'attachment_sha256': hashlib.sha256(attachment.read_bytes()).hexdigest() if attachment.is_file() else None,
            'bev_shape': [64, 64, 64], 'condition_trainable': True,
            'backbone_parameters_and_bn_frozen': True,
            'query_source': 'pretrained_checkpoint_query_plus_navigation_token',
            'coordinate_contract': 'native_garage_x_columns_y_rows',
            'label_and_normalization_contract': 'native_pdm_assets',
            'speed_holder_spatial_sampling': True, 'optimizer_amsgrad': True,
            'exact_author_reproduction': False}
        if run.trajectory_generator == 'flow_matching':
            provenance['flow_matching'] = {
                'contract_version': 1, 'source': 'clean_selected_anchor',
                'training_path': 'linear_anchor_to_GT', 'endpoint_time_embedding': '(1-tau)*1000',
                'objective': 'per_layer_GT_mode_L1_velocity_divided_by_training_std',
                'train_t_max': run.flow_train_t_max, 'min_remain': run.flow_min_remain,
                'sampler': 'deterministic_Euler_tau_0_to_1', 'CR_enabled': run.flow_contract_enable}
            if run.flow_contract_enable:
                provenance['flow_matching']['CR'] = {k: v for k, v in asdict(run).items() if k.startswith('flow_contract_')}
                provenance['flow_matching']['CR'].update(
                    method='endpoint_jacobian', query_states='GT_nearest_anchor_clean_interpolation',
                    normalized_route_jacobian_shape=[20,20], speed_perturbed=False,
                    recovery_supervision=False, loss='mean(relu(sigma_max-kappa(tau))**2)',
                    interval_weight_compensation=True)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    loader_options = dict(batch_size=run.batch_size, num_workers=run.workers, pin_memory=device.type=='cuda',
                          worker_init_fn=worker_seed, persistent_workers=False)
    training_loader = DataLoader(training_data, shuffle=True, **loader_options)
    validation_budget = run.batch_size * run.validation_max_batches
    validation_loader = DataLoader(validation_subset(validation_data, validation_budget),
                                   shuffle=False, **loader_options)
    writer = SummaryWriter(str(output))
    # Rebuild from completed epochs preceding the resume point, never from future artifacts.
    best_generation = best_generation_before(output, start_epoch)
    if resume and best_generation is not None:
        (output/'best_generation.json').write_text(json.dumps(best_generation,indent=2)+'\n')
    amp = run.amp and device.type == 'cuda'
    try:
        if not resume and prediction_base(config) == TFPP_CLEAN_PRIOR and run.generation_validation_samples:
            initial_generation = evaluate_generation(model, generation_loader(validation_data,
                run.generation_validation_samples, run.batch_size), device, run.generation_validation_seeds)
            (output/'generation_initial.json').write_text(json.dumps(initial_generation,indent=2)+'\n')
            print(json.dumps({'initial_generation': initial_generation['mean_over_seeds']}), flush=True)
        for epoch in range(start_epoch, run.epochs):
            model.train()
            total_loss, seen = 0., 0
            for batch_index, batch in enumerate(training_loader):
                if hasattr(model.planner, 'set_training_progress'):
                    model.planner.set_training_progress(global_step, len(training_loader))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    result = model(**tensor_inputs(batch, device))
                    loss = sum(result['losses'].values())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient')
                optimizer.step()
                scheduler.step(epoch + (batch_index+1)/len(training_loader))
                global_step += 1
                total_loss += float(loss.detach()); seen += 1
                for key, value in dict(result['losses'], **result['metrics']).items():
                    writer.add_scalar('train/'+key, float(value.detach()), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)
                if run.max_steps and global_step >= run.max_steps:
                    break
            validation = evaluate(model, validation_loader, device, 0, amp)
            for key, value in validation.items():
                writer.add_scalar('validation/'+key, value, epoch)
            summary = {'epoch': epoch, 'global_step': global_step, 'train_loss': total_loss/seen,
                       'prediction_base': prediction_base(config), 'training_scope': run.training_scope,
                       'validation': validation, 'validation_metric_scope': (
                           'clean_interpolated_GT_anchor_flow_matching' if run.trajectory_generator == 'flow_matching'
                           else 'noisy_gt_anchor_denoising'),
                       'validation_sampling': 'fixed_seed_full_pool_subset_with_scenario_coverage',
                       'validation_samples': len(validation_loader.dataset),
                       'completed_full_epoch': seen == len(training_loader)}
            if run.generation_validation_samples:
                generated = evaluate_generation(model, generation_loader(validation_data,
                    run.generation_validation_samples, run.batch_size), device, run.generation_validation_seeds)
                generation_report = f'generation_epoch_{epoch:04d}.json'
                (output/generation_report).write_text(json.dumps(generated, indent=2)+'\n')
                summary['validation_generation'] = {key: generated[key] for key in
                    ('scope', 'prediction_base', 'precision', 'samples', 'inference_steps', 'seeds',
                     'mean_over_seeds', 'per_seed', 'regression_health')}
                summary['validation_generation'].update(scenario_count=len(generated['scenarios']),
                                                        report=generation_report)
                for key, value in generated['mean_over_seeds'].items():
                    writer.add_scalar('validation_generation/'+key, value, epoch)
                for seed, values in generated['per_seed'].items():
                    for key, value in values.items():
                        writer.add_scalar(f'validation_generation/seed_{seed}/{key}', value, epoch)
                for scenario, seed_values in generated['scenarios'].items():
                    for key in ('route_ADE_m', 'near_lateral_MAE_m', 'speed_MAE_mps'):
                        value = sum(values[key] for values in seed_values.values())/len(seed_values)
                        writer.add_scalar(f'validation_generation/scenario_{scenario}/{key}', value, epoch)
                for name, values in generated['regression_health'].items():
                    writer.add_scalar('regression_health/'+name+'/active_fraction', values['active_fraction'], epoch)
            print(json.dumps(summary), flush=True)
            (output/f'epoch_{epoch:04d}.json').write_text(json.dumps(summary, indent=2)+'\n')
            if summary['completed_full_epoch']:
                save_checkpoint(model, optimizer, scheduler, epoch, global_step, output, fingerprint, configuration_digest)
                choice = generation_choice(summary)
                if choice is not None and (best_generation is None or choice['route_ADE_m'] < best_generation['route_ADE_m']):
                    best_generation = choice
                    (output/'best_generation.json').write_text(json.dumps(best_generation,indent=2)+'\n')
            else:
                # A smoke checkpoint cannot claim exact-resume semantics for a partial epoch.
                from .checkpoint import atomic_save
                atomic_save(model.state_dict(), output/'model_smoke.pth')
            writer.flush()
            if run.max_steps and global_step >= run.max_steps:
                break
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    for key in ('data-root', 'assets-dir', 'tfpp-checkpoint', 'output-dir'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--inference-steps', type=int, default=20)
    parser.add_argument('--classification-loss', choices=['cross_entropy', 'focal'], default='cross_entropy')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--validation-max-batches', type=int, default=100)
    parser.add_argument('--generation-validation-samples', type=int, default=128)
    parser.add_argument('--generation-validation-seeds', type=int, nargs='+', default=[137, 271, 811])
    parser.add_argument('--rollout-denoising-weight', type=float, default=0.0)
    parser.add_argument('--rollout-denoising-scenes', type=int, default=4)
    parser.add_argument('--rollout-geometry-weight', type=float, default=0.)
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--reset-regression-layers', type=int, nargs='*', default=[])
    parser.add_argument('--prediction-base', choices=['noisy_residual', 'tfpp_clean_prior_v1'], default='noisy_residual')
    parser.add_argument('--training-scope', choices=['joint', 'clean_path_only'], default='joint')
    parser.add_argument('--migrate-noisy-residual', action='store_true')
    parser.add_argument('--path-geometry-weight', type=float, default=0.)
    parser.add_argument('--implementation-profile', choices=['legacy_vp', 'author_v8_pdm'], default='legacy_vp')
    parser.add_argument('--regression-supervision', choices=['gt_each_layer', 'author_predicted_last'],
                        default='gt_each_layer')
    parser.add_argument('--trajectory-generator', choices=['ddbm', 'flow_matching'], default='ddbm')
    parser.add_argument('--flow-train-t-max', type=float, default=0.95)
    parser.add_argument('--flow-min-remain', type=float, default=0.05)
    parser.add_argument('--flow-contract-enable', action='store_true')
    parser.add_argument('--flow-contract-weight', type=float, default=0.01)
    parser.add_argument('--flow-contract-fraction', type=float, default=0.25)
    parser.add_argument('--flow-contract-interval', type=int, default=4)
    parser.add_argument('--flow-contract-fd-epsilon', type=float, default=0.1)
    parser.add_argument('--flow-contract-fd-chunk', type=int, default=2)
    parser.add_argument('--flow-contract-delay-epochs', type=float, default=2.)
    parser.add_argument('--flow-contract-ramp-epochs', type=float, default=1.)
    parser.add_argument('--flow-contract-tau-kappa', type=float, nargs=4, default=[1.3,1.2,1.08,1.0])
    parser.add_argument('--resume', default='')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--allow-debug-assets', action='store_true')
    args = vars(parser.parse_args())
    resume = args.pop('resume'); preflight = args.pop('preflight_only'); debug = args.pop('allow_debug_assets')
    args['amp'] = not args.pop('no_amp')
    train(RunConfig(**args), resume, preflight, debug)


if __name__ == '__main__':
    main()
