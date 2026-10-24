"""Zero-initialized output speed correction; route generation stays frozen."""
import torch
from torch import nn


class SpeedCorrector(nn.Module):
    def __init__(self, embed_dims=256, poses=10, width=64, max_speed=30.):
        super().__init__()
        self.embed_dims, self.poses, self.max_speed = embed_dims, poses, max_speed
        inputs = 2*embed_dims+2*poses+1
        self.net = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def features(self, query, prediction):
        if query.shape[1:] != (self.poses+1, self.embed_dims):
            raise ValueError('Expected frozen route queries followed by navigation token')
        if prediction.shape[1:] != (self.poses+1, 2):
            raise ValueError('Expected generated route and continuous speed token')
        query, prediction = query.detach().float(), prediction.detach().float()
        return torch.cat([query[:, :self.poses].mean(1), query[:, -1],
            prediction[:, :self.poses].flatten(1)/16., prediction[:, -1, :1]/self.max_speed], -1)

    def raw_speed(self, query, prediction):
        correction = self.net(self.features(query, prediction))[:, 0]
        return prediction[:, -1, 0].detach().float()+correction

    def forward(self, query, prediction):
        result = prediction.detach().clone()
        result[:, -1, 0] = self.raw_speed(query, prediction).clamp(0., self.max_speed)
        return result
