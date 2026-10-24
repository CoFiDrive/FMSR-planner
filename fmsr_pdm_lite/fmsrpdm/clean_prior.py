"""Versioned experiment: predict a correction to a frozen TF++ clean path."""
import torch
from torch import nn


NOISY_RESIDUAL = 'noisy_residual'
TFPP_CLEAN_PRIOR = 'tfpp_clean_prior_v1'


def prediction_base(config):
    value = getattr(config, 'prediction_base', NOISY_RESIDUAL)
    if value not in (NOISY_RESIDUAL, TFPP_CLEAN_PRIOR):
        raise ValueError(f'Unknown prediction base: {value}')
    return value


def correction_only(config):
    scope = getattr(config, 'training_scope', 'joint')
    if scope not in ('joint', 'clean_path_only'):
        raise ValueError(f'Unknown training scope: {scope}')
    if scope == 'clean_path_only' and prediction_base(config) != TFPP_CLEAN_PRIOR:
        raise ValueError('clean_path_only requires tfpp_clean_prior_v1')
    return scope == 'clean_path_only'


def correction_head(width, poses):
    # LayerNorm controls the modulated feature scale; SiLU avoids a dead ReLU block.
    head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(),
                         nn.Linear(width, width), nn.SiLU(), nn.Linear(width, poses * 2))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def planning_context(model, *args, **kwargs):
    """Explicit prior transport; legacy callers retain their three-value context."""
    if prediction_base(model.config) == TFPP_CLEAN_PRIOR:
        return model.context(*args, **kwargs, return_clean_prior=True)
    query, bev, grid = model.context(*args, **kwargs)
    return query, bev, grid, None


def prior_kwargs(prior):
    return {'clean_prior': prior} if prior is not None else {}


def decode_clean_prior(planner, feature, state, anchors, bev, query, time, prior):
    """Use the released attention blocks, with a separate clean path projection.

    The old regression branch still predicts the speed residual. Each layer's
    path is prior + correction, and the next layer sees its detached prediction,
    as in the released decoder. No noisy path is added to the clean output.
    """
    if prior is None or prior.shape != (len(state), planner.config.fmsr_num_poses, 2):
        raise ValueError('tfpp_clean_prior_v1 requires the frozen TF++ path [B,10,2]')
    if not torch.isfinite(prior).all():
        raise ValueError('Nonfinite TF++ clean path')
    prior = prior.detach().to(state)
    predictions = []
    points = state
    for layer, head in zip(planner.diff_decoder.layers, planner.clean_path_heads):
        noisy_feature = layer.norm1(layer.cross_bev_attention(feature, points, bev, bev.shape[-2:]))
        anchor_feature = layer.norm1_T(layer.cross_bev_attention_T(feature, anchors, bev, bev.shape[-2:]))
        noisy_feature = layer.norm2(noisy_feature + layer.dropout1(
            layer.cross_ego_attention(noisy_feature, query, query)[0]))
        anchor_feature = layer.norm2_T(anchor_feature + layer.dropout1_T(
            layer.cross_ego_attention_T(anchor_feature, query, query)[0]))
        combined = torch.cat([layer.norm3(layer.ffn(noisy_feature)),
                              layer.norm3_T(layer.ffn_T(anchor_feature))], -1)
        combined = layer.time_modulation(combined, time, global_cond=None, global_img=None)
        correction = head(combined).reshape(*state.shape[:2], prior.shape[1], 2)
        path = prior[:, None] + correction
        residual, _ = layer.task_decoder(combined)
        speed = points[..., -1:, 0:1] + residual[..., -1:, 0:1]
        speed_token = torch.cat([speed, torch.zeros_like(speed)], -1)
        prediction = torch.cat([path, speed_token], -2)
        predictions.append(prediction)
        points = prediction.detach()
    return predictions
