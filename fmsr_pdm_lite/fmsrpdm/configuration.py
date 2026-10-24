"""Published settings and explicitly recorded PDM adapter choices."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from . import bootstrap
from config import GlobalConfig
from .config_values import normalize_config_values
from .speed import MAX_TARGET_SPEED_MPS


@dataclass
class RunConfig:
    data_root: str
    assets_dir: str
    tfpp_checkpoint: str
    output_dir: str
    seed: int = 0
    epochs: int = 10
    batch_size: int = 32
    workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.01
    cosine_t0: int = 10
    cosine_t_mult: int = 2
    inference_steps: int = 20
    classification_loss: str = 'cross_entropy'
    device: str = 'cuda'
    amp: bool = True
    max_steps: int = 0
    validation_max_batches: int = 100
    generation_validation_samples: int = 128
    generation_validation_seeds: tuple = (137, 271, 811)
    rollout_denoising_weight: float = 0.0
    rollout_denoising_scenes: int = 4
    rollout_geometry_weight: float = 0.0
    init_checkpoint: str = ''
    reset_regression_layers: tuple = ()
    prediction_base: str = 'noisy_residual'
    training_scope: str = 'joint'
    migrate_noisy_residual: bool = False
    path_geometry_weight: float = 0.0
    implementation_profile: str = 'legacy_vp'
    regression_supervision: str = 'gt_each_layer'
    trajectory_generator: str = 'ddbm'
    flow_train_t_max: float = 0.95
    flow_min_remain: float = 0.05
    flow_contract_enable: bool = False
    flow_contract_weight: float = 0.01
    flow_contract_fraction: float = 0.25
    flow_contract_interval: int = 4
    flow_contract_fd_epsilon: float = 0.1
    flow_contract_fd_chunk: int = 2
    flow_contract_delay_epochs: float = 2.0
    flow_contract_ramp_epochs: float = 1.0
    flow_contract_tau_kappa: tuple = (1.3, 1.2, 1.08, 1.0)

    def validate(self):
        if self.flow_contract_enable:
            if self.trajectory_generator != 'flow_matching':
                raise ValueError('CR requires the v8 flow_matching generator')
            if (not math.isfinite(self.flow_contract_weight) or self.flow_contract_weight <= 0
                    or not math.isfinite(self.flow_contract_fraction) or not 0 < self.flow_contract_fraction <= 1
                    or self.flow_contract_interval < 1 or not 1 <= self.flow_contract_fd_chunk <= 20
                    or not math.isfinite(self.flow_contract_fd_epsilon) or self.flow_contract_fd_epsilon <= 0
                    or any(not math.isfinite(v) or v < 0 for v in
                           (self.flow_contract_delay_epochs, self.flow_contract_ramp_epochs))
                    or len(self.flow_contract_tau_kappa) != 4
                    or any(not math.isfinite(v) or v <= 0 for v in self.flow_contract_tau_kappa)):
                raise ValueError('Invalid spectral CR settings')
        if self.trajectory_generator not in ('ddbm', 'flow_matching'):
            raise ValueError('Unknown trajectory generator')
        if self.trajectory_generator == 'flow_matching':
            if (self.implementation_profile != 'author_v8_pdm' or self.regression_supervision != 'gt_each_layer'
                    or self.prediction_base != 'noisy_residual' or self.training_scope != 'joint'):
                raise ValueError('FM-only requires author_v8_pdm, gt_each_layer and joint noisy_residual architecture')
            if any((self.rollout_denoising_weight, self.rollout_geometry_weight, self.path_geometry_weight)):
                raise ValueError('FM-only baseline excludes auxiliary recovery/geometry losses')
            if (not math.isfinite(self.flow_train_t_max) or not 0 < self.flow_train_t_max < 1
                    or not math.isfinite(self.flow_min_remain) or not 0 < self.flow_min_remain <= 1
                    or self.flow_min_remain > 1-self.flow_train_t_max+1e-8):
                raise ValueError('FM time range must keep 1-tau >= flow_min_remain > 0')
        if self.implementation_profile not in ('legacy_vp', 'author_v8_pdm'):
            raise ValueError('Unknown implementation profile')
        if self.regression_supervision not in ('gt_each_layer', 'author_predicted_last'):
            raise ValueError('Unknown regression supervision')
        if self.implementation_profile == 'author_v8_pdm' and (
                self.prediction_base != 'noisy_residual' or self.training_scope != 'joint'):
            raise ValueError('author_v8_pdm requires noisy_residual and joint training')
        if self.regression_supervision == 'author_predicted_last' and self.implementation_profile != 'author_v8_pdm':
            raise ValueError('Author regression loss requires author_v8_pdm')
        from .clean_prior import prediction_base, correction_only, TFPP_CLEAN_PRIOR
        prediction_base(self); correction_only(self)
        if self.training_scope == 'clean_path_only' and not self.init_checkpoint:
            raise ValueError('clean_path_only requires a trained FMSR initialization checkpoint')
        if self.migrate_noisy_residual and (self.prediction_base != TFPP_CLEAN_PRIOR or not self.init_checkpoint):
            raise ValueError('Migration requires a noisy-residual initialization checkpoint and clean-prior mode')
        if self.prediction_base == TFPP_CLEAN_PRIOR and self.reset_regression_layers:
            raise ValueError('Do not reset the legacy speed regression branch in the clean-prior experiment')
        if not math.isfinite(self.path_geometry_weight) or self.path_geometry_weight < 0:
            raise ValueError('path_geometry_weight must be finite and nonnegative')
        if self.epochs < 1 or self.batch_size < 1 or self.workers < 0 or self.lr <= 0:
            raise ValueError('Invalid training settings')
        if not 1 <= self.inference_steps <= 1000:
            raise ValueError('inference_steps must be in [1, 1000]')
        if self.classification_loss not in ('cross_entropy', 'focal'):
            raise ValueError('Choose paper cross_entropy or released-code focal loss')
        if self.generation_validation_samples < 0:
            raise ValueError('generation_validation_samples must be nonnegative; 0 disables generation validation')
        if not self.generation_validation_seeds or len(set(self.generation_validation_seeds)) != len(self.generation_validation_seeds):
            raise ValueError('Generation validation seeds must be nonempty and distinct')
        if not math.isfinite(self.rollout_denoising_weight) or self.rollout_denoising_weight < 0 or self.rollout_denoising_scenes < 1:
            raise ValueError('Invalid rollout denoising settings')
        if self.rollout_denoising_weight and self.inference_steps < 2:
            raise ValueError('Rollout supervision requires at least two sampling steps')
        if not math.isfinite(self.rollout_geometry_weight) or self.rollout_geometry_weight < 0:
            raise ValueError('Invalid rollout geometry weight')
        if self.rollout_geometry_weight and not self.rollout_denoising_weight:
            raise ValueError('Rollout geometry requires rollout denoising supervision')
        if self.init_checkpoint and not Path(self.init_checkpoint).is_file():
            raise FileNotFoundError(self.init_checkpoint)
        if len(set(self.reset_regression_layers)) != len(self.reset_regression_layers) or any(
                layer not in (0, 1) for layer in self.reset_regression_layers):
            raise ValueError('Invalid regression layer reset selection')
        for path in (self.data_root, self.assets_dir, self.tfpp_checkpoint):
            if not Path(path).exists():
                raise FileNotFoundError(path)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + '\n')


def model_config(run):
    """Retain the exact TF++ sensor and perception architecture from its config."""
    config_path = Path(run.tfpp_checkpoint).parent / 'config.json'
    raw = json.loads(config_path.read_text())
    if raw.get('use_fmsr'):
        raise ValueError('Use an original TF++ perception checkpoint, not a previous FMSR experiment')
    config = GlobalConfig()
    config.__dict__.update({k: v for k, v in raw.items() if not k.startswith('py/')})
    normalize_config_values(config)
    if not config.transformer_decoder_join or config.backbone != 'transFuser':
        raise ValueError('The reproduction requires the TF++ transformer perception architecture')
    if config.use_wp_gru or config.lidar_seq_len != 1:
        raise ValueError('The supplied TF++ checkpoint must use direct path/speed prediction and one LiDAR frame')
    config.use_fmsr = True
    config.trajectory_generator = run.trajectory_generator
    config.flow_train_t_max = run.flow_train_t_max
    config.flow_min_remain = run.flow_min_remain
    config.fmsr_trajectory_head = 'paper_vp_ddbm'
    config.fmsr_num_poses = 10
    config.predict_checkpoint_len = 10
    config.diffusion_speed = True
    config.batch_size = run.batch_size
    config.use_amp = int(run.amp)
    config.fmsr_max_target_speed_mps = MAX_TARGET_SPEED_MPS
    config.speed_holder = 1
    config.tf_d_model = 256
    config.tf_d_ffn = 512
    config.tf_num_head = 8
    config.tf_dropout = 0.1
    config.step_num = run.inference_steps
    config.flow_inference_steps = run.inference_steps
    config.fmsr_use_target_point = True
    config.fmsr_bev_layout = 'carla_garage'
    config.bev_channels = 1512  # validated against the checkpoint backbone at construction
    config.implementation_profile = run.implementation_profile
    config.regression_supervision = run.regression_supervision
    config.backbone_fused_channels = 1512
    if run.implementation_profile == 'author_v8_pdm':
        config.bev_channels = config.bev_features_chanels
        if config.bev_channels != 64:
            raise ValueError('Author PDM head requires the TF++ 64-channel BEV grid')
        config.fmsr_trajectory_head = 'author_v8_pdm_adapter_v1'
        config.fmsr_bev_include_speed = True
        # Version the parameter/gradient contract to reject legacy inference weights.
        config.fmsr_train_condition = True
        if run.trajectory_generator == 'flow_matching':
            config.fmsr_trajectory_head = ('author_v8_pdm_fm_cr_v1' if run.flow_contract_enable
                                                else 'author_v8_pdm_fm_only_v1')
    config.classification_loss = run.classification_loss
    config.rollout_denoising_weight = run.rollout_denoising_weight
    config.rollout_denoising_scenes = run.rollout_denoising_scenes
    config.rollout_geometry_weight = run.rollout_geometry_weight
    config.prediction_base = run.prediction_base
    config.training_scope = run.training_scope
    config.path_geometry_weight = run.path_geometry_weight
    if run.prediction_base == 'tfpp_clean_prior_v1':
        config.fmsr_trajectory_head = 'experimental_tfpp_clean_prior_v1'
    config.reproduction_pretrained_images = False
    config.fmsr_anchor_path = str(Path(run.assets_dir).resolve() / 'anchors.npy')
    config.trajectory_stats_path = str(Path(run.assets_dir).resolve() / 'stats.npz')
    config.fmsr_stats_path = config.trajectory_stats_path
    config.fmsr_frame_manifest = str(Path(run.assets_dir).resolve() / 'train_frames.txt')
    config.fmsr_recovery_root = ''
    config.fmsr_recovery_manifest = ''
    for name in ('flow_contract_enable', 'flow_contract_weight', 'flow_contract_fraction',
                 'flow_contract_interval', 'flow_contract_fd_epsilon', 'flow_contract_fd_chunk',
                 'flow_contract_delay_epochs', 'flow_contract_ramp_epochs', 'flow_contract_tau_kappa'):
        setattr(config, name, getattr(run, name))
    config.flow_contract_method = 'endpoint_jacobian' if run.flow_contract_enable else 'disabled'
    config.flow_recovery_weight = 0.0
    config.use_speed_weights = 0
    config.use_semantic = config.use_depth = config.use_bev_semantic = 0
    config.freeze_backbone = 1
    config.val_towns = []  # route-disjoint train/validation manifests implement the split
    return config
