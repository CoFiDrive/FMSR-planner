"""Stable analytical VP bridge, paper equations (6)-(7), beta_d=2, beta_min=.1."""
import torch


class VPBridge:
    def __init__(self, beta_d=2.0, beta_min=0.1):
        self.beta_d, self.beta_min = beta_d, beta_min

    def coefficients(self, t):
        t = torch.as_tensor(t)
        if not t.is_floating_point():
            t = t.float()
        if not torch.isfinite(t).all() or (t < 0).any() or (t > 1).any():
            raise ValueError('Normalized bridge time must be finite and in [0, 1]')
        q = .5 * self.beta_d * t.square() + self.beta_min * t
        q_T = t.new_tensor(.5 * self.beta_d + self.beta_min)
        alpha = torch.exp(-.5 * q)
        gamma2 = torch.expm1(q) / torch.expm1(q_T)
        remainder = (1 - gamma2).clamp_min(0)
        a = alpha * gamma2 / torch.exp(-.5 * q_T)
        b = alpha * remainder
        c = torch.sqrt((-torch.expm1(-q) * remainder).clamp_min(0))
        return a, b, c

    @staticmethod
    def expand(value, tensor):
        return value.reshape(value.shape + (1,) * (tensor.ndim - value.ndim))

    def add_noise(self, clean, anchor, t, noise):
        a, b, c = [self.expand(x, clean) for x in self.coefficients(t)]
        return a * anchor + b * clean + c * noise

    def step(self, current, clean_prediction, anchor, t, previous_t, initial_noise=None):
        a, b, c = [self.expand(x, current) for x in self.coefficients(t)]
        ap, bp, cp = [self.expand(x, current) for x in self.coefficients(previous_t)]
        initial = self.expand(torch.as_tensor(t, device=current.device) == 1, current)
        # The pinned release initializes the first interior state stochastically.
        residual = (current - a * anchor - b * clean_prediction) / c.clamp_min(torch.finfo(current.dtype).tiny)
        if initial_noise is None:
            initial_noise = torch.randn_like(current) if initial.any() else torch.zeros_like(current)
        residual = torch.where(initial, initial_noise, residual)
        return ap * anchor + bp * clean_prediction + cp * residual
