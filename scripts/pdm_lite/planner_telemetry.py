"""Read-only planner instrumentation; no additional decoder calls or random draws."""
import time
import torch


class GenerationCapture:
    def __init__(self, step, trace_every=20, snapshot=False):
        self.step = step
        self.trace_enabled = trace_every > 0 and step % trace_every == 0
        self.snapshot_enabled = snapshot
        self.trace = []
        self.nfe = 0
        self.mode = None
        self.scene_snapshot = None
        self.started = time.perf_counter()
        self.cuda_events = None

    def scene(self, head, ego, bev, spatial):
        self.head = head
        self.started = time.perf_counter()
        if ego.is_cuda:
            self.cuda_events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            self.cuda_events[0].record()
        if self.snapshot_enabled:
            # Copy to CPU after the production forward, outside measured planner GPU time.
            self.scene_snapshot = {'ego_query': ego.detach(), 'bev_feature': bev.detach(),
                                   'bev_spatial_shape': spatial,
                                   'head_module': type(head).__module__,
                                   'head_class': type(head).__name__,
                                   'head_config': dict(vars(head.config)),
                                   'torch_rng_state': torch.get_rng_state(),
                                   'cuda_rng_state': torch.cuda.get_rng_state(ego.device) if ego.is_cuda else None,
                                   'autocast_enabled': torch.is_autocast_enabled(),
                                   'autocast_dtype': str(torch.get_autocast_gpu_dtype()) if ego.is_cuda else None}

    def modes(self, logits, anchors, mode):
        self.logits = logits.detach()
        self.mode = mode.detach()
        self.anchor = self.select(anchors).detach().clone()

    def select(self, value):
        if value.shape[1] == 1:
            return value[:, 0]
        return value[torch.arange(len(value), device=value.device), self.mode]

    def integration(self, state, endpoint, next_state, start, end, kind):
        self.nfe += 1
        self.generated_mode_count = int(state.shape[1])
        self.final_state = self.select(next_state).detach()
        if self.trace_enabled:
            self.trace.append({'time_start': torch.as_tensor(start).detach().clone(),
                               'time_end': torch.as_tensor(end).detach().clone(),
                               'time_kind': kind,
                               'state': self.select(state).detach().clone(),
                               'endpoint': self.select(endpoint).detach().clone(),
                               'next_state': self.select(next_state).detach().clone()})

    def finish(self):
        if self.cuda_events:
            self.cuda_events[1].record()
        self.wall_submit_ms = 1000 * (time.perf_counter() - self.started)
        if self.scene_snapshot is not None and self.mode is not None:
            self.scene_snapshot.update(reference_final_state=self.final_state,
                                       reference_mode=self.mode, reference_logits=self.logits)

    def report(self):
        if self.mode is None:
            return {'available': False, 'reason': 'planner does not support instrumentation'}
        gpu_ms = None
        if self.cuda_events:
            self.cuda_events[1].synchronize()
            gpu_ms = self.cuda_events[0].elapsed_time(self.cuda_events[1])
        std = getattr(self.head, 'std', None)
        return {'available': True, 'head': type(self.head).__module__,
                'selected_mode': self.mode, 'mode_logits': self.logits,
                'mode_probabilities': self.logits.float().softmax(-1),
                'probability_interpretation': 'softmax-normalized ranking scores; not calibrated uncertainty',
                'selected_anchor': self.anchor, 'normalization_std': std,
                'nfe': self.nfe, 'candidate_count': int(self.logits.shape[-1]),
                'generated_mode_count': self.generated_mode_count,
                'decoder_layer_count':len(self.head.diff_decoder.layers),
                'trace_sampled':self.trace_enabled, 'snapshot_sampled':self.snapshot_enabled,
                'nfe_definition': 'batched iterative endpoint/denoiser calls; classification excluded',
                'planner_gpu_ms': gpu_ms, 'planner_wall_submit_ms': self.wall_submit_ms,
                'integration_trace': self.trace,
                'trajectory_kind': 'spatial checkpoints; no supervised future timestamps',
                'trajectory_coordinates': 'ego x forward, y right, meters; final token [speed_mps,0]'}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_tree(v) for v in value)
    return value
