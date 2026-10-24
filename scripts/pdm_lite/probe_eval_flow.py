"""Replay saved FM scenes and measure finite-step condition response and route Jacobians."""
import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch


def restore(snapshot, device):
    config = SimpleNamespace(**snapshot['head_config'])
    cls = getattr(importlib.import_module(snapshot['head_module']), snapshot['head_class'])
    if (snapshot['head_module'] == 'fmsrpdm.flow_matching'
            and snapshot['head_class'] == 'FlowMatchingPlanner'
            and config.trajectory_generator == 'flow_matching'):
        head = cls(config)
        prefix = 'planner.'
    elif hasattr(cls, 'predict_flow_endpoint') and config.trajectory_generator == 'residual_flow':
        head = cls(config.fmsr_num_poses, config.tf_d_ffn, config.tf_d_model,
                   config.fmsr_anchor_path, config)
        prefix = 'fmsr_head.'
    else:
        raise ValueError('This probe requires an FM/residual-flow snapshot')
    weights = torch.load(snapshot['checkpoint'], map_location='cpu', weights_only=True)
    weights = weights.get('state_dict', weights)
    weights = {k.removeprefix('module.'): v for k, v in weights.items()}
    head.load_state_dict({k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}, strict=True)
    return head.to(device).eval()


@torch.no_grad()
def probe(head, ego, bev, spatial, scale=.01, seed=0, jacobian_every=5, epsilon=.1,
          freeze_speed=False, segment_points=0, reference_mode=None, condition_component='ego_query',
          condition_delta=None):
    if scale < 0 or epsilon <= 0 or jacobian_every < 0:
        raise ValueError('Invalid perturbation scale, FD epsilon or Jacobian interval')
    if segment_points == 1 or segment_points < 0:
        raise ValueError('Segment quadrature needs >=2 points, or 0 to disable')
    anchors = head.plan_anchor.float()[None]
    if ego.shape[0] != 1:
        raise ValueError('Probe expects one scene snapshot')
    rng = torch.Generator(device=ego.device).manual_seed(seed)
    if condition_component not in ('ego_query','bev'):
        raise ValueError('Condition component must be ego_query or bev')
    target = ego if condition_component == 'ego_query' else bev
    noise = torch.randn(target.shape, generator=rng, device=target.device, dtype=target.dtype)
    delta = noise * (target.float().square().mean().sqrt() * scale) if condition_delta is None else condition_delta.to(target)
    if delta.shape != target.shape or not torch.isfinite(delta).all():
        raise ValueError('Explicit condition delta must match the target and be finite')
    changed_ego = ego + delta if condition_component == 'ego_query' else ego
    changed_bev = bev + delta if condition_component == 'bev' else bev
    logits = head._classify_flow_anchors(anchors, bev, spatial, ego)[-1]
    changed_logits = head._classify_flow_anchors(anchors, changed_bev, spatial, changed_ego)[-1]
    nominal_mode = logits.argmax(-1)
    mode = nominal_mode if reference_mode is None else nominal_mode.new_tensor([reference_mode])
    if not 0 <= int(mode.item()) < anchors.shape[1]:
        raise ValueError('Invalid reference mode')
    changed_mode = changed_logits.argmax(-1)
    anchor = anchors[:, mode.item():mode.item()+1]
    std = head.std.to(ego.device).float().reshape(-1, 2)
    poses = head.config.fmsr_num_poses
    rows, states, changed_states = [], [anchor.detach().cpu()], [anchor.detach().cpu()]
    x, y = anchor.clone(), anchor.clone()
    times = head.get_flow_schedule(device=ego.device)

    def norm(value):
        if freeze_speed:
            value = value[..., :poses, :]
            normalizer = std[:poses]
        else:
            normalizer = std
        return float(torch.linalg.vector_norm(value.float() / normalizer))

    def endpoint(state, tau, condition, given_anchor=anchor, given_bev=bev):
        return head.predict_flow_endpoint(state, tau.expand(len(state)), given_anchor,
                                          given_bev, spatial, condition, None).float()

    def apply(state, prediction, alpha, given_anchor=anchor):
        result = head._mask_flow_speed_padding(state + alpha * (prediction - state))
        if freeze_speed and result.shape[-2] > poses:
            result = result.clone()
            result[..., poses:, :] = given_anchor[..., poses:, :]
        return result

    def sigma_at(state, tau):
        columns = []
        for index in range(poses*2):
            perturbation = torch.zeros_like(state)
            perturbation[..., index//2, index%2] = epsilon*std[index//2,index%2]
            difference = endpoint(state+perturbation,tau,ego)-endpoint(state-perturbation,tau,ego)
            columns.append((difference[..., :poses, :]/(2*epsilon*std[:poses])).flatten())
        return float(torch.linalg.svdvals(torch.stack(columns,dim=-1))[0])

    bound = 0.
    estimated_cr_bound = 0.
    for j, (tau, next_tau) in enumerate(zip(times[:-1], times[1:])):
        fx = endpoint(x, tau, ego)
        fy_nominal = endpoint(y, tau, ego)
        fy_changed = endpoint(y, tau, changed_ego, given_bev=changed_bev)
        d = norm(x-y)
        s = norm(fx-fy_nominal)
        eta = norm(fy_nominal-fy_changed)
        alpha = float((next_tau-tau)/(1-tau))
        gain = s/d if d > 0 else None
        q = 1-alpha+alpha*(gain or 0.)
        bound = q * bound + alpha * eta
        sigma = None
        segment_sigmas = None
        nu_estimate = None
        kappa_schedule = getattr(head.config,'flow_contract_tau_kappa',None)
        kappa = (kappa_schedule[min(3,int(float(tau)*4))] if kappa_schedule is not None
                 else getattr(head.config,'flow_contract_rho',.98))
        q_bar_estimate = None
        if jacobian_every and j % jacobian_every == 0:
            sigma = sigma_at(y,tau)
            if segment_points:
                segment_sigmas = [sigma_at(y+(x-y)*fraction,tau)
                                  for fraction in torch.linspace(0,1,segment_points,device=y.device)]
                violations = [max(0,s-kappa)**2 for s in segment_sigmas]
                integral = sum((a+b)*.5 for a,b in zip(violations[:-1],violations[1:]))/(segment_points-1)
                nu_estimate = integral**.5
                q_bar_estimate = 1-alpha*(1-kappa-nu_estimate)
        if freeze_speed and jacobian_every==1 and segment_points:
            estimated_cr_bound = q_bar_estimate * estimated_cr_bound + alpha*eta
        x = apply(x,fx,alpha)
        y = apply(y,fy_changed,alpha)
        rows.append({'tau':float(tau), 'alpha':alpha, 'd_j':d, 's_j':s, 'eta_j':eta,
                     'kappa_obs_j':gain, 'near_zero_d_j':d <= 1e-8, 'q_j':q,
                     'B_obs_next':bound, 'd_next':norm(x-y), 'route_sigma_max':sigma,
                     'kappa_threshold':float(kappa), 'segment_route_sigmas':segment_sigmas,
                     'nu_segment_quadrature_estimate':nu_estimate, 'q_bar_estimate':q_bar_estimate,
                     'condition_fixed_speed_fixed_route_jacobian':True})
        states.append(x.detach().cpu());changed_states.append(y.detach().cpu())
    alternate = anchors[:,changed_mode.item():changed_mode.item()+1]
    z = alternate.clone()
    for tau, next_tau in zip(times[:-1], times[1:]):
        z = apply(z,endpoint(z,tau,changed_ego,alternate,changed_bev),float((next_tau-tau)/(1-tau)),alternate)
    mode_gap = norm(y-z)
    nominal_free=x
    if int(mode.item()) != int(nominal_mode.item()):
        nominal_anchor=anchors[:,nominal_mode.item():nominal_mode.item()+1]
        nominal_free=nominal_anchor.clone()
        for tau,next_tau in zip(times[:-1],times[1:]):
            nominal_free=apply(nominal_free,endpoint(nominal_free,tau,ego,nominal_anchor),
                               float((next_tau-tau)/(1-tau)),nominal_anchor)
    nominal_mode_gap=norm(nominal_free-x)
    final_gap = norm(nominal_free-z)
    return {
        'perturbation': 'relative RMS Gaussian perturbation of encoded condition; not raw ego pose noise',
        'condition_component':condition_component,
        'scale': scale, 'seed':seed, 'condition_delta_norm':float(delta.norm()),
        'ego_query_delta_norm':float(delta.norm()) if condition_component=='ego_query' else 0.,
        'norm_scope':'route_only_speed_fixed' if freeze_speed else 'joint_route_and_speed_excluding_zero_padding',
        'production_rollout':not freeze_speed,
        'mode':int(mode.item()), 'changed_mode':int(changed_mode.item()),
        'nominal_classifier_mode':int(nominal_mode.item()), 'reference_mode_forced':reference_mode is not None,
        'anchor_switched':int(nominal_mode.item()) != int(changed_mode.item()),
        'fixed_reference_anchor_changed':int(mode.item()) != int(changed_mode.item()),
        'WMD':norm(x-y), 'MSD':mode_gap,
        'fixed_mode_terminal_delta_normalized':((y-x)/std).detach().cpu(),
        'free_mode_terminal_delta_normalized':((z-nominal_free)/std).detach().cpu(),
        'metric_definitions':{'WMD':'normalized within-fixed-mode terminal deviation',
                              'ASR':'fraction of condition trials changing classifier argmax',
                              'MSD':'normalized fixed-vs-free mode deviation under perturbed condition'},
        'D_mode':mode_gap, 'fixed_mode_final_difference':norm(x-y),
        'selected_output_difference':final_gap, 'B_obs_M':bound,
        'reference_to_free_perturbed_difference':norm(x-z),
        'nominal_reference_mode_gap':nominal_mode_gap,
        'bound_includes_mode':bound+mode_gap+nominal_mode_gap,
        'B_CR_quadrature_estimate':estimated_cr_bound if freeze_speed and jacobian_every==1 and segment_points else None,
        'B_CR_scope':'requires speed-fixed path probe, every-stage Jacobians and segment quadrature; '
                     'finite sampling is not a certified integral bound',
        'numerical_bound_holds': final_gap <= bound+mode_gap+nominal_mode_gap+1e-5,
        'normalization_std':std.detach().cpu(), 'stage_metrics':rows,
        'states':states, 'changed_states':changed_states,
        'kappa_schedule':getattr(head.config,'flow_contract_tau_kappa',None),
        'spectral_scope':'route Jacobian conditional on speed; does not certify joint route/speed contraction',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--condition-scale',type=float,default=.01)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--jacobian-every',type=int,default=5)
    parser.add_argument('--fd-epsilon',type=float,default=.1)
    parser.add_argument('--freeze-speed-token',action='store_true')
    parser.add_argument('--segment-points',type=int,default=0)
    parser.add_argument('--reference-mode',type=int)
    parser.add_argument('--condition-component',choices=['ego_query','bev'],default='ego_query')
    args=parser.parse_args()
    snapshot=torch.load(args.snapshot,map_location='cpu',weights_only=False)
    head=restore(snapshot,args.device)
    ego=snapshot['ego_query'].to(args.device).float()
    bev=snapshot['bev_feature'].to(args.device).float()
    spatial=snapshot['bev_spatial_shape']
    # FP32 diagnostics are explicit, separate from the precision used by live evaluation.
    with torch.no_grad():
        live_replay=head.forward_test_flow(ego,bev,spatial)
    report=probe(head,ego,bev,spatial,args.condition_scale,args.seed,
                 args.jacobian_every,args.fd_epsilon,args.freeze_speed_token,args.segment_points,
                 args.reference_mode,args.condition_component)
    report.update(snapshot=str(args.snapshot.resolve()), checkpoint=snapshot['checkpoint'],
                  precision='FP32 offline probe; compare replay error before interpreting live equivalence',
                  replay_max_abs_error=float((live_replay.cpu()-snapshot['reference_final_state']).abs().max()))
    from scripts.pdm_lite.planner_telemetry import cpu_tree
    def plain(value):
        if isinstance(value,torch.Tensor):return value.tolist()
        if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
        if isinstance(value,list):return [plain(v) for v in value]
        return value
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(plain(cpu_tree(report)),indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ['mode','changed_mode','B_obs_M','D_mode',
                                           'numerical_bound_holds','replay_max_abs_error']}))


if __name__=='__main__':main()
