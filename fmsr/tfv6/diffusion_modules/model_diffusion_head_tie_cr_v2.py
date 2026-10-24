"""Isolated TIE/CR training variant for the PDM-Lite FMSR head.

The network and inference path are inherited unchanged from the original
TrajectoryHead. Only the training states and losses used by CR differ, so a
checkpoint has the same parameter names and can be compared fairly.
"""

from contextlib import contextmanager

import torch
from torch import nn

from fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm import TrajectoryHead as BaseTrajectoryHead


class TrajectoryHead(BaseTrajectoryHead):
    """Apply the spectral hinge to nominal and GT-free inference rollout states."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.trajectory_generator != "residual_flow" or not self.config.flow_contract_enable:
            raise ValueError("TIE CR v2 requires residual_flow with CR enabled")
        if self.config.flow_contract_method != "endpoint_jacobian":
            raise ValueError("TIE CR v2 requires endpoint_jacobian CR")
        for name in ("flow_contract_pred_mode_fraction", "flow_contract_on_policy_fraction"):
            value = getattr(self.config, name, 0.5)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 <= getattr(self.config, "flow_recovery_weight", 0.1) < float("inf"):
            raise ValueError("flow_recovery_weight must be finite and nonnegative")

    @contextmanager
    def _deterministic_cr_decoder(self):
        """Use identical dropout behavior for every finite-difference column."""
        modules = [m for m in self.diff_decoder.modules() if isinstance(m, nn.Dropout)]
        original = [m.training for m in modules]
        try:
            for module in modules:
                module.eval()
            yield
        finally:
            for module, was_training in zip(modules, original):
                module.train(was_training)

    def _contractive_recovery_v2(self, x_t, tau, plan_anchor, x1_pred,
                                 gt_mode_idx, pred_mode_idx, gt,
                                 bev_feature, bev_spatial_shape, ego_query,
                                 global_img=None):
        """Mix modes and nominal/actual rollout states; supervise recovery FM."""
        batch = x_t.shape[0]
        count = max(1, int(batch * self.config.flow_contract_fraction))
        selected = torch.randperm(batch, device=x_t.device)[:count]

        def sample_mask(fraction):
            n = min(count, max(1, round(count * fraction))) if fraction > 0 else 0
            mask = torch.zeros(count, device=x_t.device, dtype=torch.bool)
            mask[torch.randperm(count, device=x_t.device)[:n]] = True
            return mask

        predicted = sample_mask(getattr(self.config, "flow_contract_pred_mode_fraction", 0.5))
        shifted = sample_mask(getattr(self.config, "flow_contract_on_policy_fraction", 0.5))
        modes = torch.where(predicted, pred_mode_idx[selected], gt_mode_idx[selected])
        nominal = x_t[selected, modes].float().detach()
        anchors = plan_anchor[selected, modes].float().detach()
        base_tau = tau[selected].float()
        std = self.std.to(device=x_t.device, dtype=torch.float32)[0, 0]

        bev = bev_feature[selected].float()
        ego = ego_query[selected].float()
        image = global_img[selected].float() if global_img is not None else None
        x_base, tau_base = self._flow_cr_query_states(
            nominal, base_tau, anchors, shifted, bev, bev_spatial_shape, ego, image,
        )

        contract_loss, sigma = self._spectral_contract_at_state(
            x_base, tau_base, anchors, bev, bev_spatial_shape, ego, image)

        recovery_loss = contract_loss.new_zeros(())
        if shifted.any():
            with self._deterministic_cr_decoder(), torch.autocast(device_type=x_t.device.type,
                                                                  enabled=False):
                recovered = self.predict_flow_endpoint(
                    x_base[shifted].unsqueeze(1), tau_base[shifted],
                    anchors[shifted].unsqueeze(1),
                    bev[shifted], bev_spatial_shape, ego[shifted],
                    image[shifted] if image is not None else None,
                )[:, 0]
                remain = (1.0 - tau_base[shifted]).clamp_min(self.config.flow_min_remain)
                recovery_loss = nn.functional.l1_loss(
                    (recovered - gt[selected[shifted]].float()) /
                    (remain[:, None, None] * std),
                    torch.zeros_like(recovered),
                )

        return contract_loss, sigma, recovery_loss, predicted.float().mean(), shifted.float().mean()

    def _forward_train_flow(self, ego_query, bev_feature, bev_spatial_shape,
                            global_img, gt_checkpoints, dropout_pair):
        if gt_checkpoints is None:
            raise ValueError("Residual flow training requires gt_checkpoints")
        batch = ego_query.shape[0]
        plan_anchor = self.plan_anchor.float().unsqueeze(0).expand(batch, -1, -1, -1)
        gt = gt_checkpoints[..., :2].float()
        if gt.shape[1:] != plan_anchor.shape[2:]:
            raise ValueError(f"GT shape {gt.shape[1:]} must match anchor shape {plan_anchor.shape[2:]}")
        gt_mode_idx = self.nearest_anchor_idx(gt, plan_anchor)
        x_t, tau = self._flow_training_state(plan_anchor, gt[:, None].expand_as(plan_anchor))
        x1_pred = self.predict_flow_endpoint(
            x_t, tau, plan_anchor, bev_feature, bev_spatial_shape, ego_query, global_img)
        poses_cls_list = self._classify_flow_anchors(
            plan_anchor, bev_feature, bev_spatial_shape, ego_query)
        pred_mode_idx = poses_cls_list[-1].argmax(dim=-1)
        best_reg = self.gather_mode(x1_pred, pred_mode_idx)
        flow_reg = self.gather_mode(x1_pred, gt_mode_idx)
        flow_xt = self.gather_mode(x_t, gt_mode_idx)
        with torch.autocast(device_type=ego_query.device.type, enabled=False):
            remain = (1.0 - tau).clamp_min(self.config.flow_min_remain)[:, None, None]
            std = self.std.to(device=ego_query.device, dtype=torch.float32).squeeze(1)
            loss_fm = nn.functional.l1_loss(
                (flow_reg.float() - flow_xt.float()) / remain / std,
                (gt - flow_xt.float()) / remain / std,
            )
            endpoint_l1 = nn.functional.l1_loss(best_reg.float(), gt)

        loss_ctr = loss_fm.new_zeros(())
        recovery_loss = loss_fm.new_zeros(())
        sigma_mean = loss_fm.new_zeros(())
        success = loss_fm.new_zeros(())
        pred_fraction = loss_fm.new_zeros(())
        shifted_fraction = loss_fm.new_zeros(())
        if self.training:
            self._flow_contract_calls += 1
        active = (self.training and torch.is_grad_enabled() and
                  (self._flow_contract_calls - 1) % self.config.flow_contract_interval == 0)
        if active:
            loss_ctr, sigma, recovery_loss, pred_fraction, shifted_fraction = self._contractive_recovery_v2(
                x_t, tau, plan_anchor, x1_pred, gt_mode_idx, pred_mode_idx, gt,
                bev_feature, bev_spatial_shape, ego_query, global_img)
            sigma_mean = sigma.mean().detach()
            success = (sigma <= self.config.flow_contract_rho).float().mean().detach()

        warmup_steps = self.config.flow_contract_warmup_steps
        warmup = min(1.0, self._flow_contract_calls / warmup_steps) if warmup_steps else 1.0
        interval = self.config.flow_contract_interval if active else 0
        contract_weight = self.config.flow_contract_weight * interval * warmup
        recovery_weight = getattr(self.config, "flow_recovery_weight", 0.1) * interval * warmup
        flow_loss = loss_fm + contract_weight * loss_ctr + recovery_weight * recovery_loss
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
            "flow_contract_ratio": sigma_mean,
            "flow_contract_success": success,
            "flow_contract_sigma_max": sigma_mean,
            "flow_contract_active": loss_fm.new_tensor(float(active)),
            "flow_contract_weight_effective": loss_fm.new_tensor(contract_weight),
            "flow_recovery_loss": recovery_loss.detach(),
            "flow_recovery_weight_effective": loss_fm.new_tensor(recovery_weight),
            "flow_contract_pred_mode_fraction": pred_fraction.detach(),
            "flow_contract_on_policy_fraction": shifted_fraction.detach(),
        }
