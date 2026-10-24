"""Stage exactly one reproduction model and enforce resume consistency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import jsonpickle
import jsonpickle.ext.numpy as jnp

from . import bootstrap
from .model import FMSRModel
from .speed import continuous_speed_limit
from .clean_prior import prediction_base, NOISY_RESIDUAL, TFPP_CLEAN_PRIOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-experimental-clean-prior', action='store_true')
    parser.add_argument('--verify-model', action='store_true')
    parser.add_argument('--client-timeout', type=float, default=120.)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    config_path = checkpoint.parent/'config.json'
    jnp.register_handlers()
    config = jsonpickle.decode(config_path.read_text())
    base = prediction_base(config)
    head = getattr(config, 'fmsr_trajectory_head', '')
    generator = getattr(config, 'trajectory_generator', 'ddbm')
    experimental = base == TFPP_CLEAN_PRIOR and head == 'experimental_tfpp_clean_prior_v1'
    author_v8 = (base == NOISY_RESIDUAL and head == 'author_v8_pdm_adapter_v1'
                 and getattr(config, 'implementation_profile', '') == 'author_v8_pdm'
                 and generator == 'ddbm')
    author_fm = (base == NOISY_RESIDUAL and head == 'author_v8_pdm_fm_only_v1'
                 and getattr(config, 'implementation_profile', '') == 'author_v8_pdm'
                 and generator == 'flow_matching' and not getattr(config, 'flow_contract_enable', False))
    author_cr = (base == NOISY_RESIDUAL and head == 'author_v8_pdm_fm_cr_v1'
                 and getattr(config, 'implementation_profile', '') == 'author_v8_pdm'
                 and generator == 'flow_matching' and getattr(config, 'flow_contract_enable', False))
    if experimental and not args.allow_experimental_clean_prior:
        raise ValueError('Explicit --allow-experimental-clean-prior is required for this architecture')
    if not experimental and not author_v8 and not author_fm and not author_cr and not (base == NOISY_RESIDUAL and head == 'paper_vp_ddbm'):
        raise ValueError('Unsupported or inconsistent planner architecture')
    if args.client_timeout <= 0:
        raise ValueError('CARLA client timeout must be positive')
    route_ids = [r.get('id') for r in ET.parse(args.routes).getroot().findall('route')]
    if not route_ids or len(route_ids) != len(set(route_ids)):
        raise ValueError('Empty/duplicate evaluation routes')
    settings = {'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                'routes':str(args.routes.resolve()),'routes_sha256':hashlib.sha256(args.routes.read_bytes()).hexdigest(),
                'route_count':len(route_ids),'agent':str(bootstrap.PACKAGE_ROOT/'fmsrpdm/agent.py'),
                'speed_interface':'continuous_mps_v2','continuous_speed_limit_mps':continuous_speed_limit(config),
                'inference_steps':config.step_num,
                'traffic_manager_seed':0,'perception_source':'isolated_garage_snapshot'}
    if experimental:
        settings.update(prediction_base=base, trajectory_head=head, experimental_architecture=True,
                        carla_client_timeout_seconds=args.client_timeout)
    if author_v8 or author_fm or author_cr:
        settings.update(prediction_base=base, trajectory_head=head,
                        implementation_profile=config.implementation_profile,
                        regression_supervision=config.regression_supervision,
                        bev_channels=config.bev_channels, bev_grid_shape=[64,64],
                        coordinate_contract='native_garage_x_columns_y_rows',
                        carla_client_timeout_seconds=args.client_timeout)
        if author_fm or author_cr:
            settings.update(trajectory_generator=generator, flow_matching_contract=1,
                            sampler='deterministic_Euler_tau_0_to_1', CR_enabled=bool(author_cr))
            if author_cr:
                settings.update(flow_cr_contract=1, CR_method=config.flow_contract_method,
                                CR_weight=config.flow_contract_weight)
    if os.environ.get('EVAL_CAPTURE_ALL_METRICS') == '1':
        settings['analysis_recording'] = {
            'trace_every': int(os.environ.get('EVAL_TRACE_EVERY', '1')),
            'probe_every': int(os.environ.get('EVAL_PROBE_EVERY', '200')),
            'probe_max': int(os.environ.get('EVAL_PROBE_MAX', '4')),
            'vehicle_state': 'every_agent_tick', 'comfort_export_interval_seconds': 0.1}
    source = args.output/'source.json'
    if source.exists() and json.loads(source.read_text()) != settings:
        raise ValueError('Existing evaluation uses different model/config/routes; use a new output directory')
    if (args.output/'checkpoint_endpoint.json').exists() and not source.exists():
        raise ValueError('Refusing to append to unversioned evaluation results')
    if args.verify_model:
        from .checkpoint import load_initial_checkpoint
        model = FMSRModel(config)
        load_initial_checkpoint(checkpoint, model)
        print('Strict model/asset/contract loading passed on CPU', flush=True)
        del model
    agent = args.output/'agent'
    agent.mkdir(parents=True,exist_ok=True)
    models=list(agent.glob('model*.pth'))
    if models and (len(models)!=1 or models[0].resolve()!=checkpoint):
        raise ValueError('Another model is staged in this directory')
    target=agent/checkpoint.name
    if not target.exists():target.symlink_to(checkpoint)
    shutil.copy2(config_path,agent/'config.json')
    source.write_text(json.dumps(settings,indent=2)+'\n')
    print(json.dumps(settings,indent=2),flush=True)


if __name__=='__main__':main()
