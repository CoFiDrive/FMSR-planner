"""TIE CR with full spectral finite differences and legacy directional FD.

Only the training CR objective differs from the residual-flow head. Network
parameters and the inference path retain the same checkpoint interface.
"""

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm import TrajectoryHead as BaseTrajectoryHead


class TrajectoryHead(BaseTrajectoryHead):
    """Keep the checkpoint interface while selecting the configured CR objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.trajectory_generator != "residual_flow":
            raise ValueError("TIE endpoint FD CR requires residual_flow")
        self._tau_kappa = None
        self._last_fd_stats = None
        # Explicit FM-only ablations can still load existing TIE weights.
        if not self.config.flow_contract_enable:
            return
        if self.config.flow_contract_method not in ("endpoint_fd", "endpoint_jacobian"):
            raise ValueError("TIE CR requires endpoint_fd or endpoint_jacobian")
        for name in ("flow_contract_pred_mode_fraction", "flow_contract_on_policy_fraction"):
            value = getattr(self.config, name, 0.5)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        yaw_max = getattr(self.config, "flow_contract_yaw_max_deg", 1.0)
        if not math.isfinite(yaw_max) or yaw_max < 0.0:
            raise ValueError("flow_contract_yaw_max_deg must be finite and nonnegative")
        for name in ("flow_contract_delay_steps", "flow_contract_ramp_steps"):
            value = getattr(self.config, name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        max_perturb = getattr(self.config, "flow_contract_max_perturb_m", None)
        if max_perturb is not None and (not math.isfinite(max_perturb) or max_perturb <= 0.0):
            raise ValueError("flow_contract_max_perturb_m must be positive and finite")
        schedule = getattr(self.config, "flow_contract_tau_kappa", None)
        if schedule is not None:
            if (len(schedule) != 4 or
                    any(not math.isfinite(value) or value <= 0.0 for value in schedule) or
                    any(left < right for left, right in zip(schedule, schedule[1:]))):
                raise ValueError("flow_contract_tau_kappa needs four finite, positive, nonincreasing limits")
            self._tau_kappa = tuple(float(value) for value in schedule)

    def _contract_ramp_fraction(self):
        delay = getattr(self.config, "flow_contract_delay_steps", 0)
        ramp = getattr(self.config, "flow_contract_ramp_steps", 0)
        if delay == 0 and ramp == 0:
            return super()._contract_ramp_fraction()
        elapsed = self._flow_contract_calls - delay
        if elapsed <= 0:
            return 0.0
        return min(1.0, elapsed / ramp) if ramp else 1.0

    def _contract_threshold(self, tau):
        if self._tau_kappa is None:
            return torch.full_like(tau, self.config.flow_contract_rho)
        bins = torch.bucketize(tau, tau.new_tensor((0.25, 0.5, 0.75)), right=True)
        return tau.new_tensor(self._tau_kappa)[bins]

    def forward_train_flow(self, *args, **kwargs):
        self._last_fd_stats = None
        output = super().forward_train_flow(*args, **kwargs)
        if self._last_fd_stats is not None:
            gain, kappa, tau = self._last_fd_stats
            output["flow_contract_success"] = (gain <= kappa).float().mean()
            output["flow_contract_kappa_mean"] = kappa.mean()
            bins = torch.bucketize(tau, tau.new_tensor((0.25, 0.5, 0.75)), right=True)
            for index in range(4):
                mask = bins == index
                if mask.any():
                    output[f"flow_contract_gain_tau{index}"] = gain[mask].mean()
                    if self.config.flow_contract_method == 'endpoint_jacobian':
                        output[f"flow_contract_sigma_tau{index}"] = gain[mask].mean()
                    output[f"flow_contract_success_tau{index}"] = (gain[mask] <= kappa[mask]).float().mean()
                    output[f"flow_contract_count_tau{index}"] = mask.float().sum()
        weight = getattr(self.config, 'flow_rollout_recovery_weight', 0.0)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError('flow_rollout_recovery_weight must be finite and nonnegative')
        if weight and self.training and torch.is_grad_enabled():
            interval = getattr(self.config, 'flow_rollout_recovery_interval', 4)
            if not isinstance(interval, int) or interval < 1:
                raise ValueError('flow_rollout_recovery_interval must be a positive integer')
            if (self._flow_contract_calls - 1) % interval == 0:
                recovery = self.rollout_endpoint_recovery(*args, **kwargs)
                output['flow_loss'] = output['flow_loss'] + weight * interval * recovery
                output['flow_rollout_recovery_loss'] = recovery.detach()
        return output

    def rollout_endpoint_recovery(self, ego_query, bev_feature, bev_spatial_shape,
                                  global_img=None, gt_checkpoints=None, **_kwargs):
        """Supervise paths at actual deployment states; defaults leave CR unchanged.

        Labels only enter the endpoint loss, never the rollout prefix. This is
        an auxiliary recovery objective, not the original bridge FM target.
        """
        if gt_checkpoints is None:
            raise ValueError('Endpoint recovery requires supervision labels')
        scenes = getattr(self.config, 'flow_rollout_recovery_scenes', 4)
        if not isinstance(scenes, int) or scenes < 1:
            raise ValueError('flow_rollout_recovery_scenes must be a positive integer')
        count = min(scenes, len(ego_query))
        selected = torch.randperm(len(ego_query), device=ego_query.device)[:count]
        anchors = self.plan_anchor.float()[None].expand(count, -1, -1, -1)
        bev, ego = bev_feature[selected].float(), ego_query[selected].float()
        image = global_img[selected].float() if global_img is not None else None
        with torch.no_grad(), self._deterministic_cr_decoder():
            logits = self._classify_flow_anchors(anchors, bev, bev_spatial_shape, ego)[-1]
            modes = logits.argmax(-1)
            chosen = self.gather_mode(anchors, modes)
            indices = torch.randint(self.config.flow_inference_steps, (count,), device=ego.device)
            state, time = self._flow_rollout_query(chosen, indices, bev, bev_spatial_shape, ego, image)
        with self._deterministic_cr_decoder(), torch.autocast(device_type=ego.device.type, enabled=False):
            endpoint = self.predict_flow_endpoint(state[:, None], time, chosen[:, None],
                bev, bev_spatial_shape, ego, image)[:, 0, :self._num_poses]
            std = self.std.to(endpoint)[0, 0, :self._num_poses]
            return nn.functional.smooth_l1_loss(
                (endpoint - gt_checkpoints[selected, :self._num_poses].float()) / std,
                torch.zeros_like(endpoint))

    @contextmanager
    def _deterministic_cr_decoder(self):
        """Remove dropout noise from the clean/perturbed endpoint comparison."""
        modules = [module for module in self.diff_decoder.modules() if isinstance(module, nn.Dropout)]
        original = [module.training for module in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, original):
                module.train(was_training)

    @staticmethod
    def _sample_mask(count, fraction, device):
        n = min(count, max(1, round(count * fraction))) if fraction > 0.0 else 0
        mask = torch.zeros(count, device=device, dtype=torch.bool)
        mask[torch.randperm(count, device=device)[:n]] = True
        return mask

    def _physical_state_perturbation(self, state):
        """Ramp lateral/longitudinal offsets and rotate the route about ego."""
        delta = self.local_trajectory_perturbation(state)
        yaw_max = math.radians(getattr(self.config, "flow_contract_yaw_max_deg", 1.0))
        if yaw_max > 0.0:
            angle = (2.0 * torch.rand(state.shape[0], 1, 1, device=state.device) - 1.0) * yaw_max
            points = state[:, :self._num_poses, :2].float()
            cosine, sine = angle.cos(), angle.sin()
            rotated = torch.cat((cosine * points[..., 0:1] - sine * points[..., 1:2],
                                 sine * points[..., 0:1] + cosine * points[..., 1:2]), dim=-1)
            delta[:, :self._num_poses, :2] += rotated - points
        max_perturb = getattr(self.config, "flow_contract_max_perturb_m", None)
        if max_perturb is not None:
            # Limit the displacement of every waypoint, including those far
            # from ego where a small heading error otherwise moves them most.
            displacement = torch.linalg.vector_norm(delta[:, :self._num_poses, :2], dim=-1)
            largest = displacement.amax(dim=1).clamp_min(1e-6)
            scale = (max_perturb / largest).clamp(max=1.0)
            delta[:, :self._num_poses, :2] *= scale[:, None, None]
        return delta.detach()

    def contractive_recovery_loss(self, x_t, tau, plan_anchor, x1_pred, gt_mode_idx,
                                  bev_feature, bev_spatial_shape, ego_query, global_img=None,
                                  dropout_pair=None, pred_mode_idx=None):
        batch = x_t.shape[0]
        count = max(1, int(batch * self.config.flow_contract_fraction))
        selected = torch.randperm(batch, device=x_t.device)[:count]
        predicted = self._sample_mask(
            count, self.config.flow_contract_pred_mode_fraction, x_t.device)
        shifted = self._sample_mask(
            count, self.config.flow_contract_on_policy_fraction, x_t.device)
        if pred_mode_idx is None:
            raise ValueError("endpoint FD CR requires the anchor modes from the training forward pass")
        modes = torch.where(predicted, pred_mode_idx[selected], gt_mode_idx[selected])
        nominal = x_t[selected, modes].float().detach()
        anchors = plan_anchor[selected, modes].float().detach()
        base_tau = tau[selected].float()
        std_all = self.std.to(device=x_t.device, dtype=torch.float32)[0, 0]
        bev = bev_feature[selected].float()
        ego = ego_query[selected].float()
        image = global_img[selected].float() if global_img is not None else None
        state, time = self._flow_cr_query_states(
            nominal, base_tau, anchors, shifted, bev, bev_spatial_shape, ego, image,
        )
        if self.config.flow_contract_method == 'endpoint_jacobian':
            loss, sigma = self._spectral_contract_at_state(
                state, time, anchors, bev, bev_spatial_shape, ego, image,
            )
            self._last_fd_stats = (sigma, self._contract_threshold(time).detach(), time.detach())
            return loss, sigma

        # Preserve the directional objective for explicitly selected old configs.
        delta = self._physical_state_perturbation(state)
        perturbed = self._mask_flow_speed_padding(state + delta)

        # Both predictions share the same scene, anchor, time, and dropout policy.
        states = torch.stack((state, perturbed), dim=1).reshape(
            count * 2, 1, state.shape[1], 2)
        def paired_endpoints(pair_states, pair_time, pair_anchors, pair_bev,
                             pair_ego, pair_image):
            return self.predict_flow_endpoint(
                pair_states, pair_time, pair_anchors, pair_bev,
                bev_spatial_shape, pair_ego, pair_image,
            ).reshape(count, 2, state.shape[1], 2)

        with torch.autocast(device_type=x_t.device.type, enabled=False):
            endpoints = checkpoint(
                paired_endpoints, states, time.repeat_interleave(2),
                anchors[:, None].repeat_interleave(2, dim=0),
                bev_feature[selected].float().repeat_interleave(2, dim=0),
                ego_query[selected].float().repeat_interleave(2, dim=0),
                image.repeat_interleave(2, dim=0) if image is not None else None,
                use_reentrant=False,
                context_fn=lambda: (self._deterministic_cr_decoder(),
                                    self._deterministic_cr_decoder()),
            )
            std = std_all[:self._num_poses]
            output_delta = ((endpoints[:, 1, :self._num_poses] -
                             endpoints[:, 0, :self._num_poses]) / std).flatten(1)
            input_delta = (delta[:, :self._num_poses] / std).flatten(1)
            gain = torch.linalg.vector_norm(output_delta, dim=1) / (
                torch.linalg.vector_norm(input_delta, dim=1) + 1e-6)
            kappa = self._contract_threshold(time)
            loss = torch.relu(gain - kappa).square().mean()
        self._last_fd_stats = (gain.detach(), kappa.detach(), time.detach())
        return loss, gain.detach()
