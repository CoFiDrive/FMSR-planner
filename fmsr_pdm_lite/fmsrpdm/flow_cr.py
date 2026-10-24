"""Spectral endpoint CR with central finite differences and bounded activations."""
from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@contextmanager
def deterministic_decoder(planner):
    # The condition query is shared, not independently recomputed for +/- states.
    # MultiheadAttention also applies functional dropout internally; changing
    # only nn.Dropout would contaminate finite differences with attention noise.
    modules = [module for module in planner.diff_decoder.modules()
               if isinstance(module, (nn.Dropout,nn.MultiheadAttention))]
    original = [module.training for module in modules]
    try:
        for module in modules:
            module.train(False)
        yield
    finally:
        for module, mode in zip(modules, original):
            module.train(mode)


def spectral_contract(planner, state, anchors, tau, query, bev):
    """Full 20x20 normalized route Jacobian; speed remains a fixed holder."""
    count = len(state)
    poses = planner.config.fmsr_num_poses
    dimension = poses*2
    std = planner.std[0,0,:poses].float()
    epsilon = planner.config.flow_contract_fd_epsilon
    columns = []
    for start in range(0,dimension,planner.config.flow_contract_fd_chunk):
        indices = torch.arange(start,min(start+planner.config.flow_contract_fd_chunk,dimension),device=state.device)
        width = len(indices)
        delta = state.new_zeros(width, state.shape[1]*2)
        delta[torch.arange(width,device=state.device),indices] = epsilon*std.flatten()[indices]
        delta = delta.reshape(width,*state.shape[1:])
        states = torch.cat([state[:,None]+delta[None],state[:,None]-delta[None]],dim=1)
        flat = states.reshape(count*2*width,1,*state.shape[1:])

        def endpoint(x,t,a,q,b,columns_per_chunk=width):
            result = planner.predict_flow_endpoint(x,t,a,b,b.shape[-2:],q)
            return result[:,0,:poses].reshape(count,2*columns_per_chunk,dimension)

        result = checkpoint(endpoint,flat,tau.repeat_interleave(2*width),
            anchors[:,None].repeat_interleave(2*width,dim=0),
            query.repeat_interleave(2*width,dim=0), bev.repeat_interleave(2*width,dim=0),
            use_reentrant=False,
            context_fn=lambda: (deterministic_decoder(planner),deterministic_decoder(planner)))
        derivative = (result[:,:width]-result[:,width:])/(2*epsilon*std.reshape(1,1,-1))
        columns.append(derivative.transpose(1,2))
    jacobian = torch.cat(columns,dim=-1)
    sigma = torch.linalg.svdvals(jacobian.float())[:,0]
    bins = torch.bucketize(tau,tau.new_tensor([.25,.5,.75]),right=True)
    threshold = tau.new_tensor(planner.config.flow_contract_tau_kappa)[bins]
    return torch.relu(sigma-threshold).square().mean(), sigma, threshold


def contractive_loss(planner, state, anchors, tau, mode, query, bev, generator=None):
    config = planner.config
    progress = (planner._training_step+1)/planner._steps_per_epoch
    elapsed = progress-config.flow_contract_delay_epochs
    ramp = (min(1.,max(0.,elapsed/config.flow_contract_ramp_epochs))
            if config.flow_contract_ramp_epochs else float(elapsed>0))
    active = (planner.training and ramp>0 and planner._training_step % config.flow_contract_interval==0)
    weight = config.flow_contract_weight*ramp*config.flow_contract_interval if active else 0.
    zero = state.new_zeros(())
    metrics = {'flow_contract_active':zero.new_tensor(float(active)),
               'flow_contract_ramp_fraction':zero.new_tensor(ramp),
               'flow_contract_weight_effective':zero.new_tensor(weight)}
    if not active:
        return zero,metrics
    count = max(1,math.ceil(len(state)*config.flow_contract_fraction))
    selected = torch.randperm(len(state),device=state.device,generator=generator)[:count]
    selected_mode = mode[selected]
    base = state[selected,selected_mode].detach().float()
    anchor = anchors[selected,selected_mode].detach().float()
    with torch.autocast(device_type=state.device.type,enabled=False):
        raw, sigma, threshold = spectral_contract(planner,base,anchor,tau[selected].detach(),
                                                 query[selected].float(),bev[selected].float())
    metrics.update(flow_contract_loss=raw.detach(),flow_contract_sigma_max=sigma.detach().mean(),
                   flow_contract_success=(sigma.detach()<=threshold).float().mean(),
                   flow_contract_kappa_mean=threshold.mean(),flow_contract_scene_count=zero.new_tensor(count))
    return raw*weight,metrics
