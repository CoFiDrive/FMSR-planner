"""Clean anchor-to-GT FM using the v8 PDM BEV and condition architecture."""
import torch
from torch.nn import functional as F

from .planner import PaperBridgePlanner, nearest_anchor, focal_classification


class FlowMatchingPlanner(PaperBridgePlanner):
    def __init__(self, config):
        if (getattr(config, 'implementation_profile', '') != 'author_v8_pdm'
                or getattr(config, 'regression_supervision', 'gt_each_layer') != 'gt_each_layer'
                or getattr(config, 'prediction_base', 'noisy_residual') != 'noisy_residual'):
            raise ValueError('FM planner requires the v8 PDM architecture with per-layer GT supervision')
        super().__init__(config)
        self.register_buffer('flow_matching_contract', torch.tensor(1, dtype=torch.int64))
        if getattr(config, 'flow_contract_enable', False):
            self.register_buffer('flow_cr_contract', torch.tensor(1, dtype=torch.int64))
        self._training_step = 0
        self._steps_per_epoch = 1

    def set_training_progress(self, global_step, steps_per_epoch):
        self._training_step = global_step
        self._steps_per_epoch = steps_per_epoch

    @staticmethod
    def mask_holder(state):
        mask = state.new_ones(state.shape[-2:])
        mask[-1, 1] = 0
        return state * mask

    def flow_training_state(self, target, generator=None):
        anchors = self.plan_anchor[None].expand(len(target), -1, -1, -1)
        tau = torch.rand(len(target), device=target.device, generator=generator) * self.config.flow_train_t_max
        fraction = tau[:, None, None, None]
        state = (1-fraction)*anchors + fraction*target[:, None]
        return self.mask_holder(state), anchors, tau

    def training_loss(self, ego_query, bev, target, generator=None, clean_prior=None):
        if clean_prior is not None:
            raise ValueError('FM-only does not use a TF++ clean path prior')
        target = self.mask_holder(target.float())
        state, anchors, tau = self.flow_training_state(target, generator)
        logits = self.classify(ego_query, bev)
        # Reuse the endpoint decoder and residual architecture. FM integrates
        # velocity (endpoint-state)/(1-tau), with the same time direction as v8.
        predictions = self.denoise(state, anchors, 1-tau, ego_query, bev)
        mode = nearest_anchor(target, self.plan_anchor)
        batch = torch.arange(len(target), device=target.device)
        chosen = [prediction[batch, mode] for prediction in predictions]
        selected_state = state[batch, mode]
        remain = (1-tau).clamp_min(self.config.flow_min_remain)[:, None, None]
        std = self.std[0, 0]
        target_velocity = ((target-selected_state)/remain/std).flatten(1)[:, :-1]
        loss_fm = torch.stack([F.l1_loss(((endpoint-selected_state)/remain/std).flatten(1)[:, :-1],
                                      target_velocity) for endpoint in chosen]).mean()
        classifier = focal_classification if self.config.classification_loss == 'focal' else F.cross_entropy
        loss_cls = torch.stack([classifier(logit.float(), mode) for logit in logits]).mean()
        final = chosen[-1].detach()
        error = (final[:, :10]-target[:, :10]).norm(dim=-1)
        losses = {'loss_flow_matching': loss_fm, 'loss_classification': loss_cls}
        metrics = {
            'route_ADE': error.mean(), 'route_FDE': error[:, -1].mean(),
            'speed_MAE': (final[:, -1, 0]-target[:, -1, 0]).abs().mean(),
            'anchor_accuracy': (logits[-1].argmax(-1) == mode).float().mean(),
            'flow_tau_mean': tau.mean().detach()}
        if getattr(self.config, 'flow_contract_enable', False):
            from .flow_cr import contractive_loss
            contract, diagnostics = contractive_loss(self, state, anchors, tau, mode, ego_query, bev, generator)
            losses['loss_contractive'] = contract
            metrics.update(diagnostics)
        return losses, metrics

    def integrate_endpoint(self, state, endpoint, tau, tau_next):
        # Physical-space integration; normalization belongs to the FM loss.
        remain = (1-tau).clamp_min(torch.finfo(state.dtype).eps)
        return self.mask_holder(state+(tau_next-tau)*(endpoint-state)/remain)

    # Interfaces used by the existing read-only condition/Jacobian probe tools.
    def get_flow_schedule(self, device=None):
        return torch.linspace(0, 1, self.config.step_num+1, device=device)

    def _mask_flow_speed_padding(self, state):
        return self.mask_holder(state)

    def _classify_flow_anchors(self, anchors, bev, spatial, ego):
        return self.classify(ego, bev)

    def predict_flow_endpoint(self, state, tau, anchors, bev, spatial, ego, image=None):
        return self.denoise(state, anchors, 1-tau, ego, bev)[-1]

    def flow_step(self, state, tau, tau_next, anchors, scene):
        bev, spatial, ego, image = scene
        tau = torch.as_tensor(tau, device=state.device, dtype=state.dtype)
        tau_next = torch.as_tensor(tau_next, device=state.device, dtype=state.dtype)
        endpoint = self.predict_flow_endpoint(state, tau.expand(len(state)), anchors, bev, spatial, ego, image)
        return self.integrate_endpoint(state, endpoint, tau, tau_next)

    @torch.no_grad()
    def sample(self, ego_query, bev, steps=None, generator=None, clean_prior=None, return_trace=False):
        if clean_prior is not None:
            raise ValueError('FM-only does not use a TF++ clean path prior')
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
        times = torch.linspace(0, 1, steps+1, device=state.device)
        trace = []
        for index in range(steps):
            tau, next_tau = times[index], times[index+1]
            endpoint = self.denoise(state, anchors, (1-tau).expand(len(state)), ego_query, bev)[-1]
            if return_trace:
                trace.append({'t': float(1-tau), 'tau': float(tau),
                              'state': state[:, 0, :10].detach(),
                              'clean_prediction': endpoint[:, 0, :10].detach()})
            previous = state
            state = self.integrate_endpoint(state, endpoint, tau, next_tau)
            if capture is not None:
                capture.integration(previous, endpoint, state, tau.expand(len(state)),
                                    next_tau.expand(len(state)), 'flow_matching_tau_0_1')
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite FM prediction')
        result = (state[:, 0], mode)
        return (*result, trace) if return_trace else result
