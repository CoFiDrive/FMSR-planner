"""Flow Matching only: production endpoint decoder and Euler sampler.

Copied from the FM core used by model_diffusion_head_tie_cr_fd.
CR loss, finite differences, Jacobian/SVD, perturbation and recovery objectives
are removed. Existing neural-network parameter names remain compatible.
"""

import torch


from torch import nn


from fmsr.tfv6.diffusion_modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention


from fmsr.tfv6.diffusion_modules.conditional_unet1d import SinusoidalPosEmb


import copy


from enum import IntEnum


import numpy as np


from typing import Tuple, List, Optional, Dict, Union


T_NORMALIZE = 1000.0


DIFF_LOSS_WEIGHT = 2.0


DEFAULT_EGO_FUT_MODE = 20


DEFAULT_DROPOUT_RATE = 0.1


DEFAULT_BEV_DIMS = 1512


DEFAULT_EMBED_DIMS = 256


DEFAULT_TIME_EMBED_DIMS = 256


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    
    Args:
        x: Input tensor
        target_dims: Target number of dimensions
        
    Returns:
        Tensor with target_dims dimensions
        
    Raises:
        ValueError: If target_dims is less than input tensor's dimensions
    """
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    for _ in range(dims_to_append):
        x = x.unsqueeze(-1)
    return x


class TrajectoryHead(nn.Module):
    """Standalone Flow Matching head copied from the current production core."""

    def __init__(self, 
                 num_poses: int, 
                 d_ffn: int, 
                 d_model: int, 
                 plan_anchor_path: str,
                 config: Dict):
        """
        Initializes trajectory head.
        
        Args:
            num_poses: Number of (x,y,θ) poses to predict
            d_ffn: Dimensionality of feed-forward network
            d_model: Input dimensionality
            plan_anchor_path: Path to plan anchor numpy file
            config: Configuration dictionary
        """
        super().__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = DIFF_LOSS_WEIGHT
        self.ego_fut_mode = DEFAULT_EGO_FUT_MODE
        self.config = copy.deepcopy(config)
        self.trajectory_generator = getattr(config, "trajectory_generator", "residual_flow")
        if self.trajectory_generator != "residual_flow":
            raise ValueError("flow_only requires trajectory_generator=residual_flow")
        if getattr(config, "flow_contract_enable", False):
            raise ValueError("flow_only has no CR objective; use flow_contract_enable=0")
        self._validate_flow_config()

        # Mean and std initialization
        if self.config.diffusion_speed:
            self.config.speed_holder = 1
            self.x_mean = [2.490868595127002, 3.481156543917186, 4.464454848382859, 5.441355354234655, 6.415066909825394, 7.387900003104902, 8.359328053513595, 9.327316075722655, 10.290424566869847, 11.247967869978408, 4.702808110999116]
            self.x_std = [0.08869524498127068, 0.11285663272498085, 0.1441065214135421, 0.1931662097428255, 0.2527258929766773, 0.3145910978556985, 0.38050558536760937, 0.456217660900088, 0.5447450516202552, 0.646182474287767, 4.15365410738134]
            self.y_mean = [-0.012665788951539188, -0.024444082488842975, -0.047354389746646575, -0.07508421073500589, -0.10298006960418869, -0.1273045562122212, -0.14644950739563206, -0.16248700086691703, -0.17780588942014083, -0.19377927505995685, 0]
            self.y_std = [0.19376886519246483, 0.29851686549571804, 0.43366409659231814, 0.6016148827569083, 0.7882027369161408, 0.978601353702765, 1.1677537291437756, 1.3619226368298625, 1.56701254770298, 1.7858588878228894, 1.0]
        else:
            self.config.speed_holder = 0
            self.x_mean = [2.490868595127002, 3.481156543917186, 4.464454848382859, 5.441355354234655, 6.415066909825394, 7.387900003104902, 8.359328053513595, 9.327316075722655, 10.290424566869847, 11.247967869978408]
            self.x_std = [0.08869524498127068, 0.11285663272498085, 0.1441065214135421, 0.1931662097428255, 0.2527258929766773, 0.3145910978556985, 0.38050558536760937, 0.456217660900088, 0.5447450516202552, 0.646182474287767]
            self.y_mean = [-0.012665788951539188, -0.024444082488842975, -0.047354389746646575, -0.07508421073500589, -0.10298006960418869, -0.1273045562122212, -0.14644950739563206, -0.16248700086691703, -0.17780588942014083, -0.19377927505995685]
            self.y_std = [0.19376886519246483, 0.29851686549571804, 0.43366409659231814, 0.6016148827569083, 0.7882027369161408, 0.978601353702765, 1.1677537291437756, 1.3619226368298625, 1.56701254770298, 1.7858588878228894]

        # The released FMSR adaptation used fixed FMSR statistics. PDM-Lite has
        # a different route/speed distribution, so a reproduction must load
        # statistics computed from the same filtered PDM-Lite frames.
        stats_path = getattr(config, "trajectory_stats_path", None)
        if stats_path:
            stats = np.load(stats_path)
            mean = np.asarray(stats["mean"], dtype=np.float32)
            std = np.asarray(stats["std"], dtype=np.float32)
            expected_shape = (num_poses + self.config.speed_holder, 2)
            if mean.shape != expected_shape or std.shape != expected_shape:
                raise ValueError(
                    f"Expected trajectory statistics with shape {expected_shape}, "
                    f"got mean={mean.shape}, std={std.shape}"
                )
            if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError("Trajectory statistics must be finite and have positive std")
            self.x_mean = mean[:, 0].tolist()
            self.y_mean = mean[:, 1].tolist()
            self.x_std = std[:, 0].tolist()
            self.y_std = std[:, 1].tolist()

        # Convert to tensors (will be moved to correct device during forward pass)
        self.mean = torch.tensor([self.x_mean, self.y_mean]).transpose(0, 1).unsqueeze(0).unsqueeze(0)
        self.std = torch.tensor([self.x_std, self.y_std]).transpose(0, 1).unsqueeze(0).unsqueeze(0)

        # Load plan anchor
        plan_anchor = np.load(plan_anchor_path)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) 
        
        # Networks
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 64*(num_poses+self.config.speed_holder)), 
            nn.Linear(d_model, d_model),
        )
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )
        
        # Decoder layers
        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=self.config,
        )

        diff_decoder_layer_cls = CustomTransformerDecoderLayerCls(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=self.config,
        )

        self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.diff_decoder_cls = CustomTransformerDecoderCls(diff_decoder_layer_cls, 2)

    def _validate_flow_config(self):
        if not 0.0 < self.config.flow_train_t_max < 1.0:
            raise ValueError("flow_train_t_max must be strictly between 0 and 1")
        if not 0.0 < self.config.flow_min_remain <= 1.0:
            raise ValueError("flow_min_remain must be in (0, 1]")
        self.get_flow_schedule()

    @staticmethod
    def gather_mode(x: torch.Tensor, mode_idx: torch.Tensor) -> torch.Tensor:
        return x[torch.arange(x.shape[0], device=x.device), mode_idx]

    @staticmethod
    def nearest_anchor_idx(gt, plan_anchor):
        """Same mean pointwise L2 target as the unchanged anchor classifier."""
        with torch.autocast(device_type=gt.device.type, enabled=False):
            distance = torch.linalg.vector_norm(gt.float()[:, None] - plan_anchor.float(), dim=-1)
            return distance.mean(dim=-1).argmin(dim=-1)

    def _mask_flow_speed_padding(self, x: torch.Tensor) -> torch.Tensor:
        # Only an appended [speed, 0] token has a dummy coordinate. Without
        # diffusion_speed the last y is a real route point and must be retained.
        if self.config.diffusion_speed:
            x = x.clone()
            x[..., -1, -1] = 0.0
        return x

    def predict_flow_endpoint(
        self, x_t, tau, plan_anchor, bev_feature, bev_spatial_shape,
        ego_query, global_img=None,
    ):
        """Predict the physical endpoint using the copied production endpoint decoder.

        Neural layers use the caller's autocast policy. Physical coordinates,
        the time coordinate and the returned endpoint are FP32.
        """
        bs = x_t.shape[0]
        x_t = x_t.float()
        traj_pos_embed = gen_sineembed_for_position(x_t, hidden_dim=64).flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed).view(bs, x_t.shape[1], -1)
        # Preserve checkpoint time semantics: anchor=1000, endpoint=0.
        time_embed = self.time_mlp((1.0 - tau.float()) * T_NORMALIZE).view(bs, 1, -1)
        poses_reg_list, _ = self.diff_decoder(
            traj_feature, x_t, plan_anchor.float(), bev_feature,
            bev_spatial_shape, ego_query, time_embed, global_img,
        )
        return self._mask_flow_speed_padding(poses_reg_list[-1].float())

    def _classify_flow_anchors(self, plan_anchor, bev_feature, bev_spatial_shape, ego_query):
        # Same shared anchor encoder, classifier and targets as FMSR.
        pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64).flatten(-2)
        features = self.plan_anchor_encoder(pos_embed).view(plan_anchor.shape[0], plan_anchor.shape[1], -1)
        if self.training:
            # Match the original training branch's FP32 feature boundary.
            features = features.float()
        return self.diff_decoder_cls(features, plan_anchor, bev_feature, bev_spatial_shape, ego_query)

    def _flow_training_state(self, plan_anchor, x_gt):
        """Nominal interpolation is always clean, independently of CR."""
        bs = plan_anchor.shape[0]
        device = plan_anchor.device
        with torch.autocast(device_type=device.type, enabled=False):
            tau = torch.rand(bs, device=device, dtype=torch.float32) * self.config.flow_train_t_max
            tau_view = tau[:, None, None, None]
            x_t = (1.0 - tau_view) * plan_anchor.float() + tau_view * x_gt.float()
            return self._mask_flow_speed_padding(x_t), tau

    def forward_train_flow(self, ego_query, bev_feature, bev_spatial_shape,
                            global_img=None, gt_checkpoints=None):
        if gt_checkpoints is None:
            raise ValueError("Residual flow training requires gt_checkpoints")
        bs = ego_query.shape[0]
        plan_anchor = self.plan_anchor.float().unsqueeze(0).expand(bs, -1, -1, -1)
        gt = gt_checkpoints[..., :2].float()
        if gt.shape[1:] != plan_anchor.shape[2:]:
            raise ValueError(f"GT shape {gt.shape[1:]} must match anchor shape {plan_anchor.shape[2:]}")
        x_gt = gt[:, None].expand_as(plan_anchor)
        gt_mode_idx = self.nearest_anchor_idx(gt, plan_anchor)
        x_t, tau = self._flow_training_state(plan_anchor, x_gt)
        x1_pred = self.predict_flow_endpoint(
            x_t, tau, plan_anchor, bev_feature, bev_spatial_shape, ego_query, global_img,
        )
        poses_cls_list = self._classify_flow_anchors(plan_anchor, bev_feature, bev_spatial_shape, ego_query)
        pred_mode_idx = poses_cls_list[-1].argmax(dim=-1)
        best_reg = self.gather_mode(x1_pred, pred_mode_idx)
        flow_reg = self.gather_mode(x1_pred, gt_mode_idx)
        flow_xt = self.gather_mode(x_t, gt_mode_idx)
        with torch.autocast(device_type=ego_query.device.type, enabled=False):
            remain = (1.0 - tau).clamp_min(self.config.flow_min_remain)[:, None, None]
            std = self.std.to(device=ego_query.device, dtype=torch.float32).squeeze(1)
            v_pred = (flow_reg.float() - flow_xt.float()) / remain
            v_target = (gt - flow_xt.float()) / remain
            loss_fm = nn.functional.l1_loss(v_pred / std, v_target / std)
            endpoint_l1 = nn.functional.l1_loss(best_reg.float(), gt)
        return {
            "poses_reg_list": None,
            "poses_cls_list": poses_cls_list,
            "best_reg": best_reg,
            "flow_loss": loss_fm,
            "flow_tau_mean": tau.mean().detach(),
            "flow_fm_loss": loss_fm.detach(),
            "flow_velocity_l1": loss_fm.detach(),
            "flow_endpoint_l1": endpoint_l1.detach(),
        }

    def get_flow_schedule(self, num_steps=None, device=None):
        num_steps = self.config.flow_inference_steps if num_steps is None else num_steps
        if not isinstance(num_steps, int) or isinstance(num_steps, bool) or num_steps < 1:
            raise ValueError("Flow num_steps must be a positive integer")
        return torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=torch.float32)

    def flow_step(self, x, tau, tau_next, plan_anchor, scene):
        bev, spatial, ego, global_img = scene
        tau = torch.as_tensor(tau, device=x.device, dtype=torch.float32)
        tau_next = torch.as_tensor(tau_next, device=x.device, dtype=torch.float32)
        endpoint = self.predict_flow_endpoint(
            x.float(), tau.expand(x.shape[0]), plan_anchor, bev, spatial, ego, global_img,
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            velocity = (endpoint.float() - x.float()) / (1.0 - tau).clamp_min(1e-6)
            x_next = x.float() + (tau_next - tau) * velocity
        x_next = self._mask_flow_speed_padding(x_next)
        capture = getattr(self, '_eval_capture', None)
        if capture is not None:
            capture.integration(x, endpoint, x_next, tau, tau_next, 'flow_tau')
        return x_next

    def flow_rollout(self, plan_anchor, scene, num_steps=None):
        schedule = self.get_flow_schedule(num_steps, plan_anchor.device)
        x = self._mask_flow_speed_padding(plan_anchor.float())
        for tau, tau_next in zip(schedule[:-1], schedule[1:]):
            x = self.flow_step(x, tau, tau_next, plan_anchor, scene)
        return x

    @torch.no_grad()
    def forward_test_flow(self, ego_query, bev_feature, bev_spatial_shape, global_img=None):
        plan_anchor = self.plan_anchor.float().unsqueeze(0).expand(ego_query.shape[0], -1, -1, -1)
        poses_cls_list = self._classify_flow_anchors(plan_anchor, bev_feature, bev_spatial_shape, ego_query)
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        capture = getattr(self, '_eval_capture', None)
        if capture is not None:
            capture.modes(poses_cls_list[-1], plan_anchor, mode_idx)
        scene = (bev_feature, bev_spatial_shape, ego_query, global_img)
        result = self.flow_rollout(plan_anchor, scene)
        return self.gather_mode(result, mode_idx)

    def norm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """Normalize odometry information using precomputed mean and std."""
        return (odo_info_fut - self.mean.to(odo_info_fut.device)) / self.std.to(odo_info_fut.device)

    def denorm_odo(self, odo_info_fut: torch.Tensor) -> torch.Tensor:
        """Denormalize odometry information using precomputed mean and std."""
        return odo_info_fut * self.std.to(odo_info_fut.device) + self.mean.to(odo_info_fut.device)

    def forward(self, 
                ego_query: torch.Tensor, 
                bev_feature: torch.Tensor,
                bev_spatial_shape: Tuple[int, int], 
                targets: Optional[torch.Tensor] = None,
                global_img: Optional[torch.Tensor] = None, 
                gt_checkpoints: Optional[torch.Tensor] = None) -> Union[Dict, torch.Tensor]:
        """Forward pass of the trajectory head.
        
        Args:
            ego_query: Ego vehicle query tensor
            bev_feature: BEV feature tensor
            bev_spatial_shape: Spatial shape of BEV features
            targets: Target trajectories (unused)
            global_img: Global image features
            gt_checkpoints: Ground truth checkpoints (for training)
            
        Returns:
            Training: Dictionary with prediction outputs
            Evaluation: Predicted trajectory tensor
        """
        if self.training:
            return self.forward_train(ego_query, bev_feature, bev_spatial_shape, global_img=global_img, gt_checkpoints=gt_checkpoints)
        else:
            return self.forward_test(ego_query, bev_feature, bev_spatial_shape, global_img=global_img)

    def forward_train(self, ego_query, bev_feature, bev_spatial_shape,
                      global_img=None, gt_checkpoints=None):
        return self.forward_train_flow(ego_query, bev_feature, bev_spatial_shape,
                                       global_img, gt_checkpoints)

    def forward_test(self, ego_query, bev_feature, bev_spatial_shape, global_img=None):
        return self.forward_test_flow(ego_query, bev_feature, bev_spatial_shape, global_img)

class DiffMotionPlanningRefinementModule(nn.Module):
    """Refinement module for motion planning with regression and classification branches."""
    
    def __init__(
        self,
        embed_dims: int = DEFAULT_EMBED_DIMS,
        ego_fut_ts: int = 8,
        ego_fut_mode: int = DEFAULT_EGO_FUT_MODE,
        if_zeroinit_reg: bool = False, 
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.if_zeroinit_reg = if_zeroinit_reg
        
        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )
        
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )

        self.init_weight()

    def init_weight(self) -> None:
        """Initialize network weights."""
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
        
    def forward(self, traj_feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the refinement module."""
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)
        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs, ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, plan_cls


class DiffMotionPlanningRefinementModuleCls(nn.Module):
    """Classification-only refinement module for motion planning."""
    
    def __init__(
        self,
        embed_dims: int = DEFAULT_EMBED_DIMS,
        ego_fut_ts: int = 8,
        ego_fut_mode: int = DEFAULT_EGO_FUT_MODE,
        if_zeroinit_reg: bool = False,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.if_zeroinit_reg = if_zeroinit_reg

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            nn.Linear(embed_dims, 1),
        )

        self.init_weight()

    def init_weight(self) -> None:
        """Initialize network weights."""
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)

    def forward(self, traj_feature: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classification module."""
        bs, ego_fut_mode, _ = traj_feature.shape

        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)

        return plan_cls


class ModulationLayer(nn.Module):
    """Modulation layer for conditioning features with time and global information."""

    def __init__(self, embed_dims: int, condition_dims: int = DEFAULT_TIME_EMBED_DIMS):
        super().__init__()
        self.if_zeroinit_scale = False
        self.embed_dims = embed_dims
        
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims * 2),
        )
        
        self.init_weight()

    def init_weight(self) -> None:
        """Initialize network weights."""
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature: torch.Tensor,
        time_embed: torch.Tensor,
        global_cond: Optional[torch.Tensor] = None,
        global_img: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the modulation layer."""
        # Combine conditioning features
        if global_cond is not None:
            global_feature = torch.cat([global_cond, time_embed], dim=-1)
        else:
            global_feature = time_embed
            
        if global_img is not None:
            global_img = global_img.flatten(2, 3).permute(0, 2, 1).contiguous()
            global_feature = torch.cat([global_img, global_feature], dim=-1)
        
        # Compute scale and shift
        scale_shift = self.scale_shift_mlp(global_feature)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Apply modulation
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature


class CustomTransformerDecoderLayer(nn.Module):
    """Custom transformer decoder layer for trajectory prediction."""
    
    def __init__(self, 
                 num_poses: int,
                 d_model: int,
                 d_ffn: int,
                 config: Dict,
                 ):
        super().__init__()

        self.config = config
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout1 = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout_T = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout1_T = nn.Dropout(DEFAULT_DROPOUT_RATE)

        # Cross BEV attention layers
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses + self.config.speed_holder,
            config=config,
            in_bev_dims=DEFAULT_BEV_DIMS,
        )
        
        self.cross_bev_attention_T = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses + self.config.speed_holder, 
            config=config,
            in_bev_dims=DEFAULT_BEV_DIMS,
        )

        # Cross ego attention layers
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        self.cross_ego_attention_T = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        
        self.ffn_T = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.norm1_T = nn.LayerNorm(config.tf_d_model)
        self.norm2_T = nn.LayerNorm(config.tf_d_model)
        self.norm3_T = nn.LayerNorm(config.tf_d_model)

        # Modulation and task decoder
        self.time_modulation = ModulationLayer(config.tf_d_model * 2, DEFAULT_TIME_EMBED_DIMS)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model * 2,
            ego_fut_ts=num_poses + self.config.speed_holder, 
            ego_fut_mode=DEFAULT_EGO_FUT_MODE,
        )

    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor,  
                plan_anchor: torch.Tensor,
                bev_feature: torch.Tensor,  
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor, 
                time_embed: Optional[torch.Tensor] = None, 
                global_img: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the transformer decoder layer."""
        # Cross BEV attention
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature_T = self.cross_bev_attention_T(traj_feature, plan_anchor, bev_feature, bev_spatial_shape)
        
        traj_feature = self.norm1(traj_feature)
        traj_feature_T = self.norm1_T(traj_feature_T)

        # Cross attention with ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)

        traj_feature_T = traj_feature_T + self.dropout1_T(self.cross_ego_attention_T(traj_feature_T, ego_query, ego_query)[0])
        traj_feature_T = self.norm2_T(traj_feature_T)

        # Feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        traj_feature_T = self.norm3_T(self.ffn_T(traj_feature_T))
        
        # Concatenate features
        traj_feature = torch.cat([traj_feature, traj_feature_T], dim=-1)
        
        # Time modulation
        traj_feature = self.time_modulation(traj_feature, time_embed, global_cond=None, global_img=global_img)
        
        # Predict waypoints
        poses_reg, poses_cls = self.task_decoder(traj_feature)
        # Keep endpoint arithmetic in FP32, including BF16 training.
        with torch.autocast(device_type=noisy_traj_points.device.type, enabled=False):
            return poses_reg[..., :2].float() + noisy_traj_points.float(), poses_cls


class CustomTransformerDecoderLayerCls(nn.Module):
    """Classification-only transformer decoder layer."""
    
    def __init__(self, 
                 num_poses: int,
                 d_model: int,
                 d_ffn: int,
                 config: Dict,
                 ):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(DEFAULT_DROPOUT_RATE)
        self.dropout1 = nn.Dropout(DEFAULT_DROPOUT_RATE)
        
        # Cross BEV attention
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses + self.config.speed_holder, 
            config=config,
            in_bev_dims=DEFAULT_BEV_DIMS, 
        )
        
        # Cross attention layers
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Modulation and task decoder
        self.time_modulation = ModulationLayer(d_model, DEFAULT_TIME_EMBED_DIMS)
        self.task_decoder_cls = DiffMotionPlanningRefinementModuleCls(
            embed_dims=d_model,
            ego_fut_ts=num_poses + self.config.speed_holder, 
            ego_fut_mode=DEFAULT_EGO_FUT_MODE,
        )

    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor,  
                bev_feature: torch.Tensor,  
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor) -> torch.Tensor:
        """Forward pass of the classification decoder layer."""
        # Cross BEV attention
        traj_feature = self.cross_bev_attention(traj_feature, noisy_traj_points, bev_feature, bev_spatial_shape)
        traj_feature = self.norm1(traj_feature)
        
        # Cross attention with ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query, ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # Feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        
        # Predict classification
        poses_cls = self.task_decoder_cls(traj_feature)

        return poses_cls


class CustomTransformerDecoder(nn.Module):
    """Stack of custom transformer decoder layers for trajectory prediction."""
    
    def __init__(
        self, 
        decoder_layer: nn.Module, 
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor,
                plan_anchor: torch.Tensor,
                bev_feature: torch.Tensor, 
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor, 
                time_embed: torch.Tensor, 
                global_img: Optional[torch.Tensor] = None) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Forward pass of the transformer decoder stack."""
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        
        for mod in self.layers:
            poses_reg, poses_cls = mod(
                traj_feature, traj_points, plan_anchor, bev_feature, 
                bev_spatial_shape, ego_query, time_embed, global_img
            )
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            # FM supervises the last endpoint only. Its earlier refinement
            # layers must stay connected through the final FM endpoint.
            traj_points = poses_reg[..., :2].clone()
            
        return poses_reg_list, poses_cls_list


class CustomTransformerDecoderCls(nn.Module):
    """Stack of classification-only transformer decoder layers."""
    
    def __init__(
        self, 
        decoder_layer: nn.Module, 
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature: torch.Tensor, 
                noisy_traj_points: torch.Tensor, 
                bev_feature: torch.Tensor, 
                bev_spatial_shape: Tuple[int, int], 
                ego_query: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the classification decoder stack."""
        poses_cls_list = []
        traj_points = noisy_traj_points
        
        for mod in self.layers:
            poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, ego_query)
            poses_cls_list.append(poses_cls)
            
        return poses_cls_list


class BaseIndex(IntEnum):
    """Base class for index enums with common functionality."""
    
    @classmethod
    def size(cls) -> int:
        """Get number of valid indices (excluding private/special attributes)."""
        return len([attr for attr in cls.__members__.keys() if not attr.startswith('_')])


class StateSE2Index(BaseIndex):
    """Intenum for SE(2) state arrays (X, Y, Heading)."""
    X = 0
    Y = 1
    HEADING = 2

    @classmethod
    @property
    def POINT(cls) -> slice:
        """Slice for 2D point (X, Y)."""
        return slice(cls.X, cls.Y + 1)

    @classmethod
    @property
    def STATE_SE2(cls) -> slice:
        """Slice for full SE(2) state (X, Y, HEADING)."""
        return slice(cls.X, cls.HEADING + 1)


class BoundingBoxIndex(BaseIndex):
    """Intenum for bounding box arrays."""
    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    HEADING = 6

    @classmethod
    @property
    def POINT2D(cls) -> slice:
        """Slice for 2D point (X, Y)."""
        return slice(cls.X, cls.Y + 1)

    @classmethod
    @property
    def POSITION(cls) -> slice:
        """Slice for 3D position (X, Y, Z)."""
        return slice(cls.X, cls.Z + 1)

    @classmethod
    @property
    def DIMENSION(cls) -> slice:
        """Slice for dimensions (LENGTH, WIDTH, HEIGHT)."""
        return slice(cls.LENGTH, cls.HEIGHT + 1)


class LidarIndex(BaseIndex):
    """Intenum for lidar point cloud arrays."""
    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4
    ID = 5

    @classmethod
    @property
    def POINT2D(cls) -> slice:
        """Slice for 2D point (X, Y)."""
        return slice(cls.X, cls.Y + 1)

    @classmethod
    @property
    def POSITION(cls) -> slice:
        """Slice for 3D position (X, Y, Z)."""
        return slice(cls.X, cls.Z + 1)


