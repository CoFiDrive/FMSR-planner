"""Frozen TF++ perception plus trainable anchor classifier and VP bridge denoiser."""
import copy
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from . import bootstrap
from model import LidarCenterNet
from .planner import PaperBridgePlanner
from .config_values import normalize_config_values
from .speed import continuous_speed_limit
from .clean_prior import planning_context, prior_kwargs, correction_only
from .precision import agent_precision


class FMSRModel(LidarCenterNet):
    def __init__(self, config):
        normalize_config_values(config)
        frozen_config = copy.deepcopy(config)
        frozen_config.use_fmsr = False
        super().__init__(frozen_config)
        self.config = config
        self.author_v8 = getattr(config, 'implementation_profile', 'legacy_vp') == 'author_v8_pdm'
        expected_channels = getattr(config, 'backbone_fused_channels', config.bev_channels)
        if self.backbone.num_features != expected_channels:
            raise ValueError('TF++ feature channels disagree with the saved reproduction config')
        self.requires_grad_(False)
        tp_dimensions = 4 if config.two_tp_input else 2
        self.navigation_encoder = nn.Sequential(nn.Linear(tp_dimensions, 128), nn.ReLU(), nn.Linear(128, 256))
        if getattr(config, 'trajectory_generator', 'ddbm') == 'flow_matching':
            from .flow_matching import FlowMatchingPlanner
            self.planner = FlowMatchingPlanner(config)
        else:
            self.planner = PaperBridgePlanner(config)
        if self.author_v8:
            self.register_buffer('author_v8_pdm_contract', torch.tensor(1, dtype=torch.int64))
            # The backbone stays fixed; condition projections and pretrained
            # checkpoint queries can adapt to the new PDM spatial planner.
            for name in ('change_channel', 'join', 'extra_sensor_encoder', 'tp_encoder'):
                module = getattr(self, name, None)
                if module is not None:
                    module.requires_grad_(True)
            for name in ('checkpoint_query', 'extra_sensor_pos_embed', 'tp_pos_embed'):
                value = getattr(self, name, None)
                if value is not None:
                    value.requires_grad_(True)
        if correction_only(config):
            self.navigation_encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(False)  # freeze dropout and BN statistics throughout the pretrained policy
        self.training = mode
        self.navigation_encoder.train(mode and not correction_only(self.config))
        self.planner.train(mode)
        if self.author_v8:
            for name in ('change_channel', 'join', 'extra_sensor_encoder', 'tp_encoder'):
                module = getattr(self, name, None)
                if module is not None:
                    module.train(mode)
        return self

    def load_tfpp(self, path):
        weights = torch.load(Path(path), map_location='cpu', weights_only=True)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        weights = {k.removeprefix('module.'): v for k, v in weights.items()}
        own = self.state_dict()
        required = {k for k in own if not k.startswith(('planner.', 'navigation_encoder.'))
                    and k != 'author_v8_pdm_contract'}
        missing = sorted(required - weights.keys())
        mismatch = sorted(k for k in required & weights.keys() if own[k].shape != weights[k].shape)
        if missing or mismatch:
            raise ValueError(f'Incompatible TF++ checkpoint: missing={missing[:8]}, shape_mismatch={mismatch[:8]}')
        self.load_state_dict({k: weights[k] for k in required}, strict=False)
        return {'loaded_perception_tensors': len(required), 'unused_checkpoint_tensors': len(weights.keys()-required)}

    def context(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None,
                return_clean_prior=False):
        if return_clean_prior:
            # Train and deploy the frozen prior under the same precision, even
            # when the caller enables BF16 AMP. All denoiser work is also FP32.
            with agent_precision(), torch.autocast(device_type=rgb.device.type, enabled=False):
                return self._context(rgb.float(), lidar_bev.float(), target_point.float(), ego_vel.float(),
                    command.float(), None if target_point_next is None else target_point_next.float(), True)
        return self._context(rgb, lidar_bev, target_point, ego_vel, command, target_point_next, False)

    def _context(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next, return_clean_prior):
        if self.config.two_tp_input:
            if target_point_next is None:
                raise ValueError('The checkpoint requires the next target point')
            target_point = torch.cat([target_point, target_point_next], dim=-1)
        with torch.no_grad():
            bev_grid, fused, _ = self.backbone(rgb, lidar_bev)
        with (nullcontext() if self.author_v8 else torch.no_grad()):
            tokens = self.change_channel(fused)
            tokens = (tokens + self.encoder_pos_encoding(tokens)).flatten(2)
            if self.extra_sensors:
                extra = []
                if self.config.use_velocity:
                    extra.append(self.velocity_normalization(ego_vel))
                if self.config.use_discrete_command:
                    extra.append(command)
                extra_token = self.extra_sensor_encoder(torch.cat(extra, dim=-1)) + self.extra_sensor_pos_embed
                tokens = torch.cat([tokens, extra_token.unsqueeze(2)], dim=2)
            tokens = tokens.transpose(1, 2)
            if self.config.tp_attention:
                tp = self.tp_encoder(target_point) + self.tp_pos_embed
                tokens = torch.cat([tokens, tp.unsqueeze(1)], dim=1)
                queries, _ = self.join(self.checkpoint_query.expand(len(rgb), -1, -1), tokens)
            else:
                queries = self.join(self.checkpoint_query.expand(len(rgb), -1, -1), tokens)
            # Only sensor/navigation inputs enter this frozen GRU; no labels.
            prior = self.checkpoint_decoder(queries[:, :10], target_point) if return_clean_prior else None
        scale = target_point.new_tensor([max(abs(self.config.min_x), abs(self.config.max_x)),
                                         max(abs(self.config.min_y), abs(self.config.max_y))])
        scale = scale.repeat(target_point.shape[-1] // 2)
        navigation = self.navigation_encoder(target_point / scale).unsqueeze(1)
        if self.author_v8:
            if bev_grid is None or bev_grid.shape[1:] != (64, 64, 64):
                raise ValueError('author_v8_pdm requires BEV [B,64,64,64]')
            # Keep native Garage axes and checkpoint-query initialization; the
            # attachment's missing data/config cannot define a safe replacement.
            result = (torch.cat([queries[:, :10], navigation], dim=1), bev_grid.detach(), bev_grid)
        else:
            result = (torch.cat([queries[:, :10].detach(), navigation], dim=1), fused.detach(), bev_grid)
        return (*result, prior.detach()) if return_clean_prior else result

    def forward(self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next=None,
                target_route=None, target_speed=None, return_loss=False, generator=None):
        queries, bev, bev_grid, prior = planning_context(
            self, rgb, lidar_bev, target_point, ego_vel, command, target_point_next)
        # Bridge coefficients and residual updates remain FP32 under BF16 perception.
        with (agent_precision() if prior is not None else nullcontext()), \
                torch.autocast(device_type=queries.device.type, enabled=False):
            if return_loss:
                if target_route is None or target_speed is None:
                    raise ValueError('Training/validation losses require current labels')
                target = torch.cat([target_route[:, :10].float(),
                                    torch.stack([target_speed.float().reshape(-1), torch.zeros(len(rgb), device=rgb.device)], -1)[:, None]], dim=1)
                losses, metrics = self.planner.training_loss(queries.float(), bev.float(), target, generator,
                                                           **prior_kwargs(prior))
                return {'losses': losses, 'metrics': metrics}
            capture = getattr(self.planner, '_eval_capture', None)
            if capture is not None:
                capture.scene(self.planner, queries.float(), bev.float(), bev.shape[-2:])
                if capture.scene_snapshot is not None:
                    capture.scene_snapshot['clean_prior'] = None if prior is None else prior.detach()
            prediction, _ = self.planner.sample(queries.float(), bev.float(), generator=generator, **prior_kwargs(prior))
            if capture is not None:
                capture.finish()
                self.planner._eval_capture = None
        speed = prediction[:, -1, 0:1].clamp(0, continuous_speed_limit(self.config))
        boxes = self.head(bev_grid) if self.config.detect_boxes else None
        return None, speed, prediction[:, :10], None, None, None, boxes, None, None, None
