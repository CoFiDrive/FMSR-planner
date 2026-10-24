"""PDM spatial attention: histogram columns are ego x, rows are ego y."""
import torch
from torch import nn
from torch.nn import functional as F


def ego_to_grid(points, config):
    result = points.clone()
    result[..., 0] = 2 * (points[..., 0] - config.min_x) / (config.max_x - config.min_x) - 1
    result[..., 1] = 2 * (points[..., 1] - config.min_y) / (config.max_y - config.min_y) - 1
    return result


class PDMCrossBEVAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_levels=1, in_bev_dims=64, num_points=11, config=None):
        super().__init__()
        self.config, self.num_points = config, num_points
        self.attention_weights = nn.Linear(embed_dims, num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(config.tf_dropout)
        self.value_proj = nn.Sequential(nn.Conv2d(in_bev_dims, embed_dims, 3, padding=1), nn.ReLU())
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, queries, traj_points, bev_feature, spatial_shape):
        # v8 samples all 11 holders, including [speed, 0]; legacy VP samples
        # only the 10 physical route points. Both use native Garage axes.
        count = (self.num_points if getattr(self.config, 'fmsr_bev_include_speed', False)
                 else self.config.fmsr_num_poses)
        spatial_points = traj_points[..., :count, :]
        grid = ego_to_grid(spatial_points, self.config).to(bev_feature.dtype)
        weights = self.attention_weights(queries)[..., :count].softmax(-1)
        value = self.value_proj(bev_feature)
        sampled = F.grid_sample(value, grid.to(value.dtype), mode='bilinear', padding_mode='zeros', align_corners=False)
        output = (sampled * weights.to(sampled.dtype).unsqueeze(1)).sum(-1).transpose(1, 2).contiguous()
        return self.dropout(self.output_proj(output)) + queries
