import torch
from torch import nn
from fmsr.tfv6.diffusion_modules.blocks import linear_relu_ln, bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from fmsr.tfv6.diffusion_modules.conditional_unet1d import SinusoidalPosEmb
import copy
from contextlib import contextmanager, nullcontext
from torch.utils.checkpoint import checkpoint
from enum import IntEnum
import numpy as np
from typing import Tuple, List, Optional, Dict, Union

# =========================================
DEFAULT_BETA_D = 2.0
DEFAULT_BETA_MIN = 0.1
DEFAULT_T = 1.0
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

class DDBMScheduler:
    """DDBM (Denoising Diffusion Bridging Model) scheduler for diffusion process."""
    
    def __init__(self, beta_d: float = DEFAULT_BETA_D, beta_min: float = DEFAULT_BETA_MIN, T: float = DEFAULT_T):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.T = T

    def vp_logs(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Calculate VP logs for given timestep."""
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * self.beta_d - 0.5 * t * self.beta_min

    def vp_logsnr(self, t: Union[float, torch.Tensor]) -> torch.Tensor:
        """Calculate VP logSNR for given timestep."""
        t = torch.as_tensor(t)
        return -torch.log((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1)

    def get_abc(self, t: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate a_t, b_t, c_t coefficients for given timestep."""
        logsnr_t = self.vp_logsnr(t)
        logsnr_T = self.vp_logsnr(self.T)
        logs_t = self.vp_logs(t)
        logs_T = self.vp_logs(self.T)

        a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
        b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
        c_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()

        return a_t, b_t, c_t

    def add_noise(self, t: torch.Tensor, x0: torch.Tensor, xT: torch.Tensor, noise: torch.Tensor, 
                  t_normalize: float = T_NORMALIZE) -> torch.Tensor:
        """Add noise to the input tensor according to DDBM schedule.
        
        Args:
            t: Timestep tensor
            x0: Clean input tensor
            xT: Prior tensor
            noise: Noise tensor
            t_normalize: Normalization factor for timestep
            
        Returns:
            Noisy tensor sample
        """
        t = append_dims(t, x0.ndim) / t_normalize
        a_t, b_t, c_t = self.get_abc(t)
        samples = a_t * xT + b_t * x0 + c_t * noise

        return samples

    def sample_step(self, t: torch.Tensor, t_prev: torch.Tensor, xt: torch.Tensor, 
                    x0: torch.Tensor, xT: torch.Tensor, t_normalize: float = T_NORMALIZE) -> torch.Tensor:
        """Perform one sampling step of the DDBM process.
        
        Args:
            t: Current timestep
            t_prev: Previous timestep
            xt: Current tensor sample
            x0: Denoised tensor (from model)
            xT: Prior tensor
            t_normalize: Normalization factor for timestep
            
        Returns:
            Tensor sample at previous timestep
        """
        is_T = ((t / t_normalize) == self.T).all()
        t = append_dims(t, x0.ndim) / t_normalize
        t_prev = append_dims(t_prev, x0.ndim) / t_normalize

        a_t, b_t, c_t = self.get_abc(t)
        a_t_prev, b_t_prev, c_t_prev = self.get_abc(t_prev)

        xt_prev = a_t_prev * xT + b_t_prev * x0 
        if is_T:
            xt_prev = xt_prev + c_t_prev * torch.randn_like(xt_prev)
        else:
            xt_prev = xt_prev + (c_t_prev / c_t) * (xt - a_t * xT - b_t * x0)

        return xt_prev
    
class TrajectoryHead(nn.Module):
    """Trajectory prediction head with diffusion-based refinement."""
    
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
        # Older checkpoints/configs predate the paper-aligned CR option.
        if not hasattr(self.config, "flow_contract_method"):
            self.config.flow_contract_method = "euler_ratio"
        if not hasattr(self.config, "flow_contract_fd_epsilon"):
            self.config.flow_contract_fd_epsilon = 0.1
        if not hasattr(self.config, "flow_contract_fd_chunk"):
            self.config.flow_contract_fd_chunk = 4
        if not hasattr(self.config, "flow_contract_interval"):
            self.config.flow_contract_interval = 1
        if not hasattr(self.config, "flow_contract_warmup_steps"):
            self.config.flow_contract_warmup_steps = 0
        self._flow_contract_calls = 0
        self.trajectory_generator = getattr(config, "trajectory_generator", "ddbm")
        if self.trajectory_generator not in ("ddbm", "residual_flow"):
            raise ValueError(f"Unknown trajectory_generator: {self.trajectory_generator}")
        if self.trajectory_generator == "residual_flow":
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

        # Diffusion scheduler
        self.diffusion_scheduler = DDBMScheduler(
            beta_d=DEFAULT_BETA_D,
            beta_min=DEFAULT_BETA_MIN,
            T=DEFAULT_T,
        )

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
        if self.config.flow_contract_enable:
            if self.config.flow_contract_method not in ("euler_ratio", "endpoint_jacobian", "endpoint_fd"):
                raise ValueError("flow_contract_method must be euler_ratio, endpoint_jacobian, or endpoint_fd")
            if self.config.tf_dropout != 0.0:
                raise ValueError("Contractive Recovery requires tf_dropout=0 for attention; explicit decoder dropout masks are paired")
            if not 0.0 < self.config.flow_contract_fraction <= 1.0:
                raise ValueError("flow_contract_fraction must be in (0, 1]")
            if not 0.0 <= self.config.flow_contract_weight < float("inf"):
                raise ValueError("flow_contract_weight must be finite and nonnegative")
            if not 0.0 < self.config.flow_contract_rho < 1.0:
                raise ValueError("flow_contract_rho must be in (0, 1)")
            if self.config.flow_contract_method in ("endpoint_jacobian", "endpoint_fd"):
                if self.config.flow_contract_method == "endpoint_jacobian":
                    if not 0.0 < self.config.flow_contract_fd_epsilon < float("inf"):
                        raise ValueError("flow_contract_fd_epsilon must be positive and finite")
                    if self.config.flow_contract_fd_chunk < 1:
                        raise ValueError("flow_contract_fd_chunk must be positive")
                if self.config.flow_contract_interval < 1:
                    raise ValueError("flow_contract_interval must be positive")
                if self.config.flow_contract_warmup_steps < 0:
                    raise ValueError("flow_contract_warmup_steps must be nonnegative")
            if not 0.0 < self.config.flow_contract_lat_min_m <= self.config.flow_contract_lat_max_m < float("inf"):
                raise ValueError("Contract lateral offsets must satisfy 0 < min <= max < inf")
            if not 0.0 <= self.config.flow_contract_lon_max_m < float("inf"):
                raise ValueError("flow_contract_lon_max_m must be finite and nonnegative")
            bank = self.config.flow_contract_step_bank
            if not bank or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in bank):
                raise ValueError("flow_contract_step_bank must contain positive integer step counts")

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
        """Predict the physical endpoint using the unchanged DDBM decoder.

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

    def _contract_ramp_fraction(self):
        """Legacy CR warmup; endpoint-FD may replace this with a delayed ramp."""
        warmup = self.config.flow_contract_warmup_steps
        return min(1.0, self._flow_contract_calls / warmup) if warmup else 1.0

    def local_trajectory_perturbation(self, traj):
        """Ramp a signed offset in the local tangent/normal frame, in meters.

        Only route points define the frame; an appended speed token is untouched.
        Degenerate tangents use the scene chord, then ego-forward if stationary.
        """
        with torch.autocast(device_type=traj.device.type, enabled=False):
            points = traj.float()[:, :self._num_poses, :2]
            bs, n, _ = points.shape
            if n < 2:
                raise ValueError("Contractive Recovery requires at least two route points")
            tangent = torch.zeros_like(points)
            tangent[:, 1:-1] = points[:, 2:] - points[:, :-2]
            tangent[:, 0] = points[:, 1] - points[:, 0]
            tangent[:, -1] = points[:, -1] - points[:, -2]
            chord = points[:, -1:] - points[:, :1]
            forward = torch.zeros_like(chord)
            forward[..., 0] = 1.0
            chord = torch.where(torch.linalg.vector_norm(chord, dim=-1, keepdim=True) > 1e-6, chord, forward)
            tangent = torch.where(torch.linalg.vector_norm(tangent, dim=-1, keepdim=True) > 1e-6, tangent, chord)
            tangent = nn.functional.normalize(tangent, dim=-1, eps=1e-6)
            normal = torch.stack([-tangent[..., 1], tangent[..., 0]], dim=-1)
            shape = (bs, 1, 1)
            lat = self.config.flow_contract_lat_min_m + torch.rand(shape, device=traj.device) * (
                self.config.flow_contract_lat_max_m - self.config.flow_contract_lat_min_m
            )
            lat = lat * (torch.randint(0, 2, shape, device=traj.device).float() * 2 - 1)
            lon = (torch.rand(shape, device=traj.device) * 2 - 1) * self.config.flow_contract_lon_max_m
            ramp = torch.linspace(0, 1, n, device=traj.device, dtype=torch.float32)[None, :, None]
            delta = torch.zeros_like(traj, dtype=torch.float32)
            delta[:, :n, :2] = ramp * (lat * normal + lon * tangent)
            return delta

    @contextmanager
    def _paired_contract_dropout(self):
        """Reuse nominal dropout masks for the CR subset, without changing weights.

        Dropout receives ones on the clean pass to expose its exact mask. The
        hook applies that mask to the original features; the perturbed pass
        reuses the selected rows. Classifier dropout is outside this context.
        Attention dropout is configured to zero, as in the supplied baseline.
        """
        pair = {"selected": None}
        masks, inputs, handles = {}, {}, []

        def before(module, args):
            if pair["selected"] is None:
                inputs[module] = args[0]
                return (torch.ones_like(args[0]),)

        def after(module, args, output):
            if pair["selected"] is None:
                masks[module] = output.detach() != 0
                features, mask = inputs.pop(module), masks[module]
            else:
                features, mask = args[0], masks[module][pair["selected"]]
            # Keep the scale in FP32, as in the CUDA dropout kernel. Storing a
            # scaled BF16 mask would round 1/(1-p) before multiplying features.
            return (features.float() * mask * (1.0 / (1.0 - module.p))).to(features.dtype)

        try:
            for module in self.diff_decoder.modules():
                if isinstance(module, nn.Dropout) and module.training and module.p > 0:
                    handles.append(module.register_forward_pre_hook(before))
                    handles.append(module.register_forward_hook(after))
            yield pair
        finally:
            for handle in handles:
                handle.remove()

    def contractive_recovery_loss(self, x_t, tau, plan_anchor, x1_pred, gt_mode_idx,
                                 bev_feature, bev_spatial_shape, ego_query, global_img=None,
                                 dropout_pair=None, pred_mode_idx=None):
        """Apply the selected CR objective on a subset of matched anchors."""
        if self.config.flow_contract_method == "endpoint_jacobian":
            return self._endpoint_jacobian_contract_loss(
                x_t, tau, plan_anchor, gt_mode_idx, bev_feature,
                bev_spatial_shape, ego_query, global_img,
            )
        if self.config.flow_contract_method == "endpoint_fd":
            raise ValueError("endpoint_fd requires the tie_cr_fd trajectory head")
        return self._euler_ratio_contract_loss(
            x_t, tau, plan_anchor, x1_pred, gt_mode_idx,
            bev_feature, bev_spatial_shape, ego_query, global_img, dropout_pair,
        )

    def _endpoint_jacobian_contract_loss(self, x_t, tau, plan_anchor, gt_mode_idx,
                                         bev_feature, bev_spatial_shape, ego_query,
                                         global_img=None):
        """Central-difference S^-1 (df/dx) S, then Eq. (16)'s spectral hinge.

        The state is the 10x2 route. PDM's appended scalar speed token is held
        fixed as a condition, while its unused second coordinate is excluded.
        Each forward uses one matched anchor; scene features are shared across
        finite-difference columns. No second-order grid_sample derivative is
        required: gradients flow through both endpoint evaluations.
        """
        bs = x_t.shape[0]
        count = max(1, int(bs * self.config.flow_contract_fraction))
        selected = torch.randperm(bs, device=x_t.device)[:count]
        modes = gt_mode_idx[selected]
        x_base = x_t[selected, modes].float()
        anchors = plan_anchor[selected, modes].float()
        return self._spectral_contract_at_state(
            x_base, tau[selected], anchors, bev_feature[selected].float(),
            bev_spatial_shape, ego_query[selected].float(),
            global_img[selected].float() if global_img is not None else None,
        )

    @contextmanager
    def _deterministic_cr_decoder(self):
        """Use the deployment dropout policy for CR and detached rollouts."""
        modules = [m for m in self.diff_decoder.modules() if isinstance(m, nn.Dropout)]
        original = [m.training for m in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, training in zip(modules, original):
                module.train(training)

    def _spectral_contract_at_state(self, x_base, tau_base, anchors, bev, spatial,
                                    ego, global_img=None):
        """Full normalized route Jacobian, not sampled directional gains."""
        count = x_base.shape[0]
        std = self.std.to(device=x_base.device, dtype=torch.float32)[0, 0, :self._num_poses]
        dimension = self._num_poses * 2
        epsilon = self.config.flow_contract_fd_epsilon
        columns = []

        for start in range(0, dimension, self.config.flow_contract_fd_chunk):
            indices = torch.arange(start, min(start + self.config.flow_contract_fd_chunk,
                                              dimension), device=x_base.device)
            width = indices.numel()
            delta = torch.zeros(width, dimension, device=x_base.device)
            delta[
                torch.arange(width, device=x_base.device), indices
            ] = epsilon * std.reshape(-1)[indices]
            delta = nn.functional.pad(
                delta.reshape(width, self._num_poses, 2),
                (0, 0, 0, x_base.shape[1] - self._num_poses),
            )
            states = torch.cat((x_base[:, None] + delta[None],
                                x_base[:, None] - delta[None]), dim=1)
            flat = states.reshape(count * 2 * width, 1, x_base.shape[1], 2)
            time_batch = tau_base.repeat_interleave(2 * width)
            anchor_batch = anchors[:, None].repeat_interleave(2 * width, dim=0)
            bev_batch = bev.float().repeat_interleave(2 * width, dim=0)
            ego_batch = ego.float().repeat_interleave(2 * width, dim=0)
            image_batch = (global_img.float().repeat_interleave(2 * width, dim=0)
                           if global_img is not None else None)

            def endpoint(batch_state, batch_time, batch_anchor, batch_bev,
                         batch_ego, batch_image, columns_per_chunk=width):
                endpoint_out = self.predict_flow_endpoint(
                    batch_state, batch_time, batch_anchor, batch_bev,
                    spatial, batch_ego, batch_image,
                )
                return endpoint_out[:, 0, :self._num_poses].reshape(
                    count, 2 * columns_per_chunk, dimension,
                )
            with torch.autocast(device_type=x_base.device.type, enabled=False):
                # Recompute activations during backward to bound memory on
                # the 24 GiB training GPU; retain first-order parameter grads.
                endpoints = checkpoint(
                    endpoint, flat, time_batch, anchor_batch, bev_batch,
                    ego_batch, image_batch, use_reentrant=False,
                    context_fn=lambda: (self._deterministic_cr_decoder(),
                                        self._deterministic_cr_decoder()),
                )
                derivative = (endpoints[:, :width] - endpoints[:, width:])
                derivative = derivative / (2.0 * epsilon * std.reshape(1, 1, -1))
                columns.append(derivative.transpose(1, 2))
        jacobian = torch.cat(columns, dim=-1)
        sigma = torch.linalg.svdvals(jacobian)[:, 0]
        threshold = (self._contract_threshold(tau_base) if hasattr(self, '_contract_threshold')
                     else torch.full_like(tau_base, self.config.flow_contract_rho))
        loss = torch.relu(sigma - threshold).square().mean()
        return loss, sigma.detach()

    def _euler_ratio_contract_loss(self, x_t, tau, plan_anchor, x1_pred, gt_mode_idx,
                                   bev_feature, bev_spatial_shape, ego_query,
                                   global_img=None, dropout_pair=None):
        """Legacy single-Euler-step directional distance penalty."""
        bs = x_t.shape[0]
        count = max(1, int(bs * self.config.flow_contract_fraction))
        selected = torch.randperm(bs, device=x_t.device)[:count]
        modes = gt_mode_idx[selected]
        clean_xt = self.gather_mode(x_t[selected], modes)
        delta = self.local_trajectory_perturbation(clean_xt)
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            mask = nn.functional.one_hot(modes, plan_anchor.shape[1]).float()[:, :, None, None]
            x_pert = x_t[selected].float() + mask * delta[:, None]
        if dropout_pair is not None:
            dropout_pair["selected"] = selected
        endpoint_pert = self.predict_flow_endpoint(
            x_pert, tau[selected], plan_anchor[selected], bev_feature[selected],
            bev_spatial_shape, ego_query[selected],
            global_img[selected] if global_img is not None else None,
        )
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            clean_endpoint = self.gather_mode(x1_pred[selected], modes).float()
            pert_endpoint = self.gather_mode(endpoint_pert, modes).float()
            remain = 1.0 - tau[selected].float()
            bank = torch.tensor(self.config.flow_contract_step_bank, device=x_t.device, dtype=torch.float32)
            dt = bank[torch.randint(len(bank), (count,), device=x_t.device)].reciprocal()
            dt = torch.minimum(dt, remain)[:, None, None]
            pert_xt = clean_xt.float() + delta
            clean_next = (clean_xt.float() + dt * (clean_endpoint - clean_xt.float()) / remain[:, None, None]).detach()
            pert_next = pert_xt + dt * (pert_endpoint - pert_xt) / remain[:, None, None]
            # Distances are physical route distances, never speed-token distances.
            d0 = torch.linalg.vector_norm(delta[:, :self._num_poses], dim=-1).mean(dim=-1)
            d1 = torch.linalg.vector_norm((pert_next - clean_next)[:, :self._num_poses], dim=-1).mean(dim=-1)
            ratio = d1 / (d0 + 1e-4)
            loss = nn.functional.relu(ratio - self.config.flow_contract_rho).mean()
        return loss, ratio

    def forward_train_flow(self, ego_query, bev_feature, bev_spatial_shape,
                           global_img=None, gt_checkpoints=None):
        paired = self.config.flow_contract_enable and self.config.flow_contract_method == "euler_ratio"
        pair_context = self._paired_contract_dropout() if paired else nullcontext(None)
        with pair_context as dropout_pair:
            return self._forward_train_flow(
                ego_query, bev_feature, bev_spatial_shape, global_img,
                gt_checkpoints, dropout_pair,
            )

    def _forward_train_flow(self, ego_query, bev_feature, bev_spatial_shape,
                            global_img, gt_checkpoints, dropout_pair):
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
        loss_ctr = loss_fm.new_zeros(())
        ratio = loss_fm.new_zeros(())
        success = loss_fm.new_zeros(())
        if self.training:
            self._flow_contract_calls += 1
        contract_ramp = self._contract_ramp_fraction()
        # Both finite-difference objectives support the delayed CR schedule.
        interval_origin = getattr(self.config, "flow_contract_delay_steps", 0)
        compute_contract = (self.training and torch.is_grad_enabled() and
                            self.config.flow_contract_enable and
                            contract_ramp > 0.0 and
                            (self._flow_contract_calls - interval_origin - 1) %
                            self.config.flow_contract_interval == 0)
        if compute_contract:
            loss_ctr, ratios = self.contractive_recovery_loss(
                x_t, tau, plan_anchor, x1_pred, gt_mode_idx,
                bev_feature, bev_spatial_shape, ego_query, global_img, dropout_pair,
                pred_mode_idx=pred_mode_idx,
            )
            ratio = ratios.mean().detach()
            threshold = (self.config.flow_contract_rho
                         if self.config.flow_contract_method in ("endpoint_jacobian", "endpoint_fd") else 1.0)
            success = (ratios <= threshold).float().mean().detach()
        with torch.autocast(device_type=ego_query.device.type, enabled=False):
            contract_scale = (self.config.flow_contract_interval
                              if self.config.flow_contract_method in ("endpoint_jacobian", "endpoint_fd") else 1)
            # The interval factor keeps the expected objective weight equal to
            # the configured lambda, while the ramp softens its entry.
            effective_weight = (self.config.flow_contract_weight * contract_scale * contract_ramp
                                if compute_contract else 0.0)
            flow_loss = loss_fm + effective_weight * loss_ctr
        return {
            "poses_reg_list": None,
            "poses_cls_list": poses_cls_list,
            "best_reg": best_reg,
            "flow_loss": flow_loss,
            "flow_tau_mean": tau.mean().detach(),
            "flow_fm_loss": loss_fm.detach(),
            "flow_velocity_l1": loss_fm.detach(),
            "flow_endpoint_l1": endpoint_l1.detach(),
            "flow_contract_loss": loss_ctr.detach(),
            "flow_contract_ratio": ratio,
            "flow_contract_success": success,
            "flow_contract_sigma_max": (ratio if self.config.flow_contract_method == "endpoint_jacobian"
                                        else loss_fm.new_zeros(())),
            "flow_contract_active": loss_fm.new_tensor(float(compute_contract)),
            "flow_contract_weight_effective": loss_fm.new_tensor(effective_weight),
            "flow_contract_ramp_fraction": loss_fm.new_tensor(contract_ramp),
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
    def _flow_rollout_query(self, anchors, step_indices, bev, spatial, ego, global_img=None):
        """States actually reached from anchors at inference query times.

        Each mode is independent in the endpoint decoder. Prefixes use the
        inference Euler step, FP32 coordinates and disabled decoder dropout.
        No expert target or bounded-displacement shortcut enters this rollout.
        """
        schedule = self.get_flow_schedule(device=anchors.device)
        steps = schedule.numel() - 1
        if step_indices.shape != (anchors.shape[0],) or step_indices.dtype != torch.long:
            raise ValueError('Rollout query steps must be one integer per anchor')
        if ((step_indices < 0) | (step_indices >= steps)).any():
            raise ValueError('Rollout queries must precede tau=1')
        state = self._mask_flow_speed_padding(anchors.float()).clone()
        with self._deterministic_cr_decoder(), torch.autocast(device_type=anchors.device.type, enabled=False):
            for index in range(int(step_indices.max().item())):
                active = step_indices > index
                scene = (bev[active].float(), spatial, ego[active].float(),
                         global_img[active].float() if global_img is not None else None)
                state[active] = self.flow_step(
                    state[active, None], schedule[index], schedule[index + 1],
                    anchors[active, None].float(), scene,
                )[:, 0]
        return state.detach(), schedule[step_indices].detach()

    def _flow_cr_query_states(self, nominal, tau, anchors, rollout_mask, bev, spatial,
                              ego, global_img=None):
        """Mix nominal bridge queries with GT-free deployment rollout queries."""
        state, time = nominal.float().detach().clone(), tau.float().detach().clone()
        if rollout_mask.any():
            count = int(rollout_mask.sum().item())
            indices = torch.randint(self.config.flow_inference_steps, (count,), device=state.device)
            generated, generated_time = self._flow_rollout_query(
                anchors[rollout_mask], indices, bev[rollout_mask], spatial, ego[rollout_mask],
                global_img[rollout_mask] if global_img is not None else None,
            )
            state[rollout_mask], time[rollout_mask] = generated, generated_time
        return state.detach(), time.detach()

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

    def forward_train(self, 
                      ego_query: torch.Tensor, 
                      bev_feature: torch.Tensor, 
                      bev_spatial_shape: Tuple[int, int], 
                      global_img: Optional[torch.Tensor] = None, 
                      gt_checkpoints: torch.Tensor = None) -> Dict:
        """Forward pass for training mode."""
        if self.trajectory_generator == "residual_flow":
            return self.forward_train_flow(
                ego_query, bev_feature, bev_spatial_shape, global_img, gt_checkpoints,
            )
        bs = ego_query.shape[0]
        device = ego_query.device

        # Prepare plan anchor and ground truth
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        x_0 = gt_checkpoints.unsqueeze(1)[...,:2].repeat(1, plan_anchor.shape[1], 1, 1)

        # Normalize and add noise
        odo_info_fut = self.norm_odo(x_0)
        odo_plan_anchor = self.norm_odo(plan_anchor)
        timesteps = torch.randint(1, 1001, (bs,), device=device)
        noise = torch.randn(odo_info_fut.shape, device=device)

        noisy_traj_points = self.diffusion_scheduler.add_noise(
            x0=odo_info_fut,
            xT=odo_plan_anchor,
            noise=noise,
            t=timesteps,
        ).float()
        noisy_traj_points = self.denorm_odo(noisy_traj_points)
        
        if self.config.diffusion_speed:
            noisy_traj_points[...,-1,-1] = 0

        # Encode trajectory features
        ego_fut_mode = noisy_traj_points.shape[1]
        ak_ego_fut_mode = plan_anchor.shape[1]

        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs, ego_fut_mode, -1)

        time_embed = self.time_mlp(timesteps)
        time_embed = time_embed.view(bs, 1, -1)

        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64)
        ak_traj_pos_embed = ak_traj_pos_embed.flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed)
        ak_traj_feature = ak_traj_feature.view(bs, ak_ego_fut_mode, -1).to(dtype=torch.float32)

        # Decode predictions
        poses_reg_list, _ = self.diff_decoder(
            traj_feature, noisy_traj_points, plan_anchor, bev_feature, 
            bev_spatial_shape, ego_query, time_embed, global_img
        )
        poses_cls_list = self.diff_decoder_cls(
            ak_traj_feature, plan_anchor, bev_feature, 
            bev_spatial_shape, ego_query
        )

        # Select best prediction
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses+self.config.speed_holder, 2)
        best_reg = torch.gather(poses_reg_list[-1], 1, mode_idx).squeeze(1)
        
        if self.config.diffusion_speed:
            best_reg[...,-1,-1] = 0

        return {"poses_reg_list": None, "poses_cls_list": poses_cls_list, "best_reg": best_reg}

    def forward_test(self, 
                     ego_query: torch.Tensor,
                     bev_feature: torch.Tensor,
                     bev_spatial_shape: Tuple[int, int],
                     global_img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for evaluation mode (sampling)."""
        if self.trajectory_generator == "residual_flow":
            return self.forward_test_flow(ego_query, bev_feature, bev_spatial_shape, global_img)
        device = ego_query.device
        bs = ego_query.shape[0]
        
        step_num = self.config.step_num
        ts = torch.arange(0, 1001, 1000//step_num).to(device)

        # Prepare plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)

        # Encode anchor features
        ak_ego_fut_mode = plan_anchor.shape[1]
        ak_traj_pos_embed = gen_sineembed_for_position(plan_anchor, hidden_dim=64)
        ak_traj_pos_embed = ak_traj_pos_embed.flatten(-2)
        ak_traj_feature = self.plan_anchor_encoder(ak_traj_pos_embed)
        ak_traj_feature = ak_traj_feature.view(bs, ak_ego_fut_mode, -1)

        # Get classification predictions
        poses_cls_list = self.diff_decoder_cls(
            ak_traj_feature, plan_anchor, bev_feature, 
            bev_spatial_shape, ego_query
        )
        mode_idx = poses_cls_list[-1].argmax(dim=-1)
        capture = getattr(self, '_eval_capture', None)
        if capture is not None:
            capture.modes(poses_cls_list[-1], plan_anchor, mode_idx)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, self._num_poses+self.config.speed_holder, 2)

        # Diffusion sampling loop
        xt_prev = plan_anchor
        for i in range(step_num, 0, -1):
            xt = xt_prev
            t = ts[i] * torch.ones([bs], device=device)
            t_prev = ts[i-1] * torch.ones([bs], device=device)

            # Encode current trajectory
            traj_pos_embed = gen_sineembed_for_position(xt, hidden_dim=64)
            traj_pos_embed = traj_pos_embed.flatten(-2)
            traj_feature = self.plan_anchor_encoder(traj_pos_embed)
            traj_feature = traj_feature.view(bs, xt.shape[1], -1)

            time_embed = self.time_mlp(t)
            time_embed = time_embed.view(bs, 1, -1)

            # Get denoised prediction
            poses_reg_list, _ = self.diff_decoder(
                traj_feature, xt, plan_anchor, bev_feature, 
                bev_spatial_shape, ego_query, time_embed, global_img
            )

            x0_mean = poses_reg_list[-1]
            x0_mean = self._mask_flow_speed_padding(x0_mean)

            # Sample previous timestep
            xt_prev_norm = self.diffusion_scheduler.sample_step(
                t=t,
                t_prev=t_prev,
                xt=self.norm_odo(xt.clone()),
                x0=self.norm_odo(x0_mean),
                xT=self.norm_odo(plan_anchor.clone()),
            )

            xt_prev = self.denorm_odo(xt_prev_norm)
            xt_prev = self._mask_flow_speed_padding(xt_prev)
            if capture is not None:
                capture.integration(xt, x0_mean, xt_prev, t, t_prev, 'ddbm_t_0_1000')

        # Select best mode
        x0_sample = torch.gather(xt_prev, 1, mode_idx).squeeze(1)

        return x0_sample


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
        if getattr(self.config, "trajectory_generator", "ddbm") == "residual_flow":
            # Do not assign physical coordinates back into a BF16 network output.
            with torch.autocast(device_type=noisy_traj_points.device.type, enabled=False):
                return poses_reg[..., :2].float() + noisy_traj_points.float(), poses_cls
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points

        return poses_reg[...,:2], poses_cls

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
            # layers must stay connected; DDBM retains its original boundary.
            traj_points = poses_reg[..., :2].clone()
            if getattr(mod.config, 'trajectory_generator', 'ddbm') != 'residual_flow':
                traj_points = traj_points.detach()
            
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

# =========================================
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
