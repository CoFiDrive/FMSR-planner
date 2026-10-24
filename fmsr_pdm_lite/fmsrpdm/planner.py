"""PDM-specific adapter around the latest published FMSR decoder blocks."""
import numpy as np
import torch
from torch.nn import functional as F

from .attention import PDMCrossBEVAttention
from .schedule import VPBridge
from .vendor_ddbm import model_diffusion_head_ddbm as release
from .clean_prior import prediction_base, correction_only, correction_head, decode_clean_prior, prior_kwargs, TFPP_CLEAN_PRIOR

# This isolated copy resolves its spatial attention to the PDM coordinate adapter.
release.GridSampleCrossBEVAttention = PDMCrossBEVAttention


def nearest_anchor(target, anchors):
    """Released distance: mean point distance in physical ego coordinates, including speed."""
    return torch.linalg.vector_norm(target[:, None] - anchors[None], dim=-1).mean(-1).argmin(-1)


def focal_classification(logits, target):
    labels = F.one_hot(target, logits.shape[-1]).to(logits.dtype)
    probability = logits.sigmoid()
    p_error = torch.where(labels.bool(), 1 - probability, probability)
    alpha = torch.where(labels.bool(), .25, .75)
    return (alpha * p_error.square() * F.binary_cross_entropy_with_logits(logits, labels, reduction='none')).mean()


class PaperBridgePlanner(release.TrajectoryHead):
    def __init__(self, config):
        anchors = np.load(config.fmsr_anchor_path, allow_pickle=False)
        stats = np.load(config.trajectory_stats_path, allow_pickle=False)
        mean, std = stats['mean'], stats['std']
        shape = (config.fmsr_num_poses + 1, 2)
        if anchors.ndim != 3 or anchors.shape[1:] != shape or len(anchors) < 1:
            raise ValueError(f'Expected anchors [K,{shape[0]},2]')
        if mean.shape != shape or std.shape != shape or (std <= 0).any():
            raise ValueError('Incorrect PDM normalization statistics')
        if not all(np.isfinite(x).all() for x in (anchors, mean, std)) or not np.all(anchors[:, -1, 1] == 0):
            raise ValueError('Nonfinite assets or nonzero speed placeholder')
        super().__init__(config.fmsr_num_poses, config.tf_d_ffn, config.tf_d_model,
                         config.fmsr_anchor_path, config)
        # Replace the release's FMSR constants with PDM-only training statistics.
        del self.mean, self.std
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32)[None, None])
        self.register_buffer('std', torch.as_tensor(std, dtype=torch.float32)[None, None])
        self.bridge = VPBridge()
        if prediction_base(config) == TFPP_CLEAN_PRIOR:
            self.register_buffer('clean_prior_contract', torch.tensor(1, dtype=torch.int64))
            self.clean_path_heads = torch.nn.ModuleList([
                correction_head(config.tf_d_model * 2, config.fmsr_num_poses)
                for _ in self.diff_decoder.layers])
            if correction_only(config):
                self.requires_grad_(False)
                self.clean_path_heads.requires_grad_(True)

    def train(self, mode=True):
        if correction_only(self.config):
            super().train(False)  # frozen decoder dropout must not move its feature distribution
            self.training = mode
            self.clean_path_heads.train(mode)
            return self
        return super().train(mode)

    def encode(self, trajectory):
        embedding = release.gen_sineembed_for_position(trajectory, hidden_dim=64).flatten(-2)
        return self.plan_anchor_encoder(embedding)

    def classify(self, ego_query, bev):
        anchors = self.plan_anchor[None].expand(ego_query.shape[0], -1, -1, -1)
        return self.diff_decoder_cls(self.encode(anchors), anchors, bev, bev.shape[-2:], ego_query)

    def denoise(self, trajectory, anchors, t, ego_query, bev, clean_prior=None):
        time = self.time_mlp(t * 1000).unsqueeze(1)
        if prediction_base(self.config) == TFPP_CLEAN_PRIOR:
            return decode_clean_prior(self, self.encode(trajectory), trajectory, anchors, bev, ego_query, time, clean_prior)
        if clean_prior is not None:
            raise ValueError('A clean prior cannot be supplied to noisy_residual mode')
        predictions, _ = self.diff_decoder(self.encode(trajectory), trajectory, anchors,
                                           bev, bev.shape[-2:], ego_query, time, None)
        # Preserve the dummy dimension without in-place writes to autograd outputs.
        mask = trajectory.new_ones(trajectory.shape[-2:])
        mask[-1, 1] = 0
        return [prediction * mask for prediction in predictions]

    def training_loss(self, ego_query, bev, target, generator=None, clean_prior=None):
        batch = len(target)
        anchor_idx = nearest_anchor(target, self.plan_anchor)
        anchors = self.plan_anchor[None].expand(batch, -1, -1, -1)
        clean = target[:, None].expand_as(anchors)
        t = torch.randint(1, 1001, (batch,), device=target.device, generator=generator).float() / 1000
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        noisy = self.denorm_odo(self.bridge.add_noise(self.norm_odo(clean), self.norm_odo(anchors), t, noise))
        mask = noisy.new_ones(noisy.shape[-2:]); mask[-1, 1] = 0
        noisy = noisy * mask
        logits = self.classify(ego_query, bev)
        predictions = self.denoise(noisy, anchors, t, ego_query, bev, **prior_kwargs(clean_prior))
        classifier = focal_classification if self.config.classification_loss == 'focal' else F.cross_entropy
        classification = torch.stack([classifier(x.float(), anchor_idx) for x in logits]).mean()
        author_loss = getattr(self.config, 'regression_supervision', 'gt_each_layer') == 'author_predicted_last'
        if author_loss:
            predicted_idx = logits[-1].argmax(-1)
            last = predictions[-1][torch.arange(batch, device=target.device), predicted_idx]
            # Explicit attachment-reproduction option. The attachment repeats
            # this last prediction for each decoder layer and includes dummy y.
            chosen = [last for _ in predictions]
            classification = torch.stack([classifier(x.float(), anchor_idx) for x in logits]).sum()
        else:
            chosen = [p[torch.arange(batch, device=p.device), anchor_idx] for p in predictions]
        # Default regression follows the GT-nearest anchor at every layer.
        targets_flat = target.flatten(1)[:, :-1]
        regression = torch.stack([F.l1_loss(p[:, :10].float(), target[:, :10]) if correction_only(self.config)
                                 else F.l1_loss(p.float().flatten(1)[:, :-1], targets_flat) for p in chosen]).mean()
        if author_loss:
            regression = torch.stack([F.l1_loss(p.float(), target) for p in chosen]).sum()
        route_error = torch.linalg.vector_norm(chosen[-1][:, :10].float().detach() - target[:, :10], dim=-1)
        metrics = {'route_ADE': route_error.mean(), 'route_FDE': route_error[:, -1].mean(),
                   'speed_MAE': (chosen[-1][:, -1, 0].float().detach() - target[:, -1, 0]).abs().mean(),
                   'anchor_accuracy': (logits[-1].argmax(-1) == anchor_idx).float().mean()}
        losses = {'loss_denoising': regression}
        if not correction_only(self.config):
            losses['loss_classification'] = classification
        geometry_weight = getattr(self.config, 'path_geometry_weight', 0.0)
        if geometry_weight:
            losses['loss_path_geometry'] = geometry_weight * torch.stack([
                self.path_geometry_loss(p[:, :10], target[:, :10]) for p in chosen]).mean()
        rollout_weight = getattr(self.config, 'rollout_denoising_weight', 0.0)
        if rollout_weight > 0:
            rollout, rollout_metrics = self.rollout_training_loss(ego_query, bev, target, generator, **prior_kwargs(clean_prior))
            losses['loss_rollout_denoising'] = rollout * rollout_weight
            metrics.update(rollout_metrics)
        return losses, metrics

    def rollout_training_loss(self, ego_query, bev, target, generator=None, clean_prior=None):
        """Optional experiment: supervise states reached by the current sampler.

        The sampled state and its prefix are detached. GT-nearest anchors isolate
        denoiser robustness from classifier errors. This auxiliary loss is not
        claimed to be part of the published paper protocol.
        """
        count = min(len(target), getattr(self.config, 'rollout_denoising_scenes', 4))
        if count < 1:
            raise ValueError('rollout_denoising_scenes must be positive')
        selected = torch.randperm(len(target), device=target.device, generator=generator)[:count]
        query, feature, truth = ego_query[selected], bev[selected], target[selected]
        selected_prior = None if clean_prior is None else clean_prior[selected].detach()
        anchors = self.plan_anchor[nearest_anchor(truth, self.plan_anchor)][:, None]
        steps = self.config.step_num
        if steps < 2:
            raise ValueError('Rollout supervision requires at least two sampling steps')
        # Interior states cover the late denoising region where drift was measured.
        candidates = sorted({max(1, round(steps * fraction)) for fraction in (.5, .8, .95)})
        candidates = [min(steps-1, i) for i in candidates]
        candidate = int(torch.randint(len(candidates), (), device=target.device, generator=generator))
        stop_index = candidates[candidate]
        times = torch.linspace(1, 0, steps+1, device=target.device)
        mask = target.new_ones(target.shape[-2:]); mask[-1, 1] = 0
        training_before = self.training
        try:
            self.eval()
            with torch.no_grad():
                state = anchors.clone()
                for index in range(stop_index):
                    t = times[index].expand(count)
                    previous = times[index+1].expand(count)
                    prediction = self.denoise(state, anchors, t, query.detach(), feature.detach(),
                                              **prior_kwargs(selected_prior))[-1]
                    noise = (torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
                             if index == 0 else torch.zeros_like(state))
                    state = self.denorm_odo(self.bridge.step(self.norm_odo(state), self.norm_odo(prediction),
                        self.norm_odo(anchors), t, previous, noise)) * mask
        finally:
            self.train(training_before)
        t = times[stop_index].expand(count)
        predictions = self.denoise(state.detach(), anchors, t, query, feature, **prior_kwargs(selected_prior))
        loss = torch.stack([F.l1_loss(p[:, 0, :10], truth[:, :10]) if correction_only(self.config)
                            else F.l1_loss(p[:, 0].flatten(1)[:, :-1], truth.flatten(1)[:, :-1])
                            for p in predictions]).mean()
        geometry_weight = getattr(self.config, 'rollout_geometry_weight', 0.0)
        geometry = torch.stack([self.path_geometry_loss(p[:, 0, :10], truth[:, :10])
                                for p in predictions]).mean() if geometry_weight else loss.new_zeros(())
        route_error = torch.linalg.vector_norm(predictions[-1][:, 0, :10].detach()-truth[:, :10], dim=-1)
        return loss + geometry_weight * geometry, {'rollout_denoising_L1': loss.detach(),
                      'rollout_geometry_L1': geometry.detach(), 'rollout_training_t': t.mean(),
                      'rollout_route_ADE': route_error.mean()}

    @staticmethod
    def path_geometry_loss(prediction, truth):
        # Match the target's local bends, including real turns; do not force straight paths.
        bend = prediction[:, 2:] - 2 * prediction[:, 1:-1] + prediction[:, :-2]
        target_bend = truth[:, 2:] - 2 * truth[:, 1:-1] + truth[:, :-2]
        return F.l1_loss(bend, target_bend)

    @torch.no_grad()
    def sample(self, ego_query, bev, steps=None, generator=None, clean_prior=None, return_trace=False):
        steps = self.config.step_num if steps is None else steps
        if not isinstance(steps, int) or not 1 <= steps <= 1000:
            raise ValueError('steps must be an integer in [1,1000]')
        logits = self.classify(ego_query, bev)[-1]
        mode = logits.argmax(-1)
        capture = getattr(self, '_eval_capture', None)
        if capture is not None:
            capture.modes(logits, self.plan_anchor[None].expand(len(ego_query), -1, -1, -1), mode)
        anchors = self.plan_anchor[mode].unsqueeze(1)
        state = anchors.clone()
        times = torch.linspace(1, 0, steps + 1, device=state.device)
        mask = state.new_ones(state.shape[-2:]); mask[-1, 1] = 0
        trace = []
        for index in range(steps):
            t, previous_t = times[index].expand(len(state)), times[index + 1].expand(len(state))
            prediction = self.denoise(state, anchors, t, ego_query, bev, **prior_kwargs(clean_prior))[-1]
            if return_trace:
                trace.append({'t': float(times[index]), 'state': state[:, 0, :10].detach(),
                              'clean_prediction': prediction[:, 0, :10].detach()})
            noise = (torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
                     if index == 0 else torch.zeros_like(state))
            previous_state = state
            state = self.denorm_odo(self.bridge.step(self.norm_odo(state), self.norm_odo(prediction),
                                                    self.norm_odo(anchors), t, previous_t, noise)) * mask
            if capture is not None:
                capture.integration(previous_state, prediction, state, t, previous_t, 'vp_bridge_t_0_1')
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite bridge prediction')
        result = (state[:, 0], mode)
        return (*result, trace) if return_trace else result
