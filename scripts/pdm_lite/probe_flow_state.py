"""Fixed-condition, fixed-anchor state injections and subsequent flow propagation."""
import torch


@torch.no_grad()
def state_probe(head, ego, bev, spatial, magnitude=.01, seed=0, directions=4,
                freeze_speed=False, reference_mode=None, reference_states=None):
    if magnitude <= 0 or directions < 1:
        raise ValueError('State perturbation magnitude and direction count must be positive')
    anchors=head.plan_anchor.float()[None]
    mode=int(head._classify_flow_anchors(anchors,bev,spatial,ego)[-1].argmax(-1).item())
    mode=mode if reference_mode is None else reference_mode
    if not 0 <= mode < anchors.shape[1]:raise ValueError('Invalid reference mode')
    anchor=anchors[:,mode:mode+1]
    poses=head.config.fmsr_num_poses
    std=head.std.to(ego.device).float().reshape(-1,2)
    times=head.get_flow_schedule(device=ego.device)
    count=len(times)-1

    def norm(value):
        if freeze_speed:return float(torch.linalg.vector_norm(value[...,:poses,:]/std[:poses]))
        return float(torch.linalg.vector_norm(value/std))

    def endpoint(state,tau):
        return head.predict_flow_endpoint(state,tau.expand(len(state)),anchor,bev,spatial,ego,None).float()

    def update(state,prediction,index):
        alpha=(times[index+1]-times[index])/(1-times[index])
        result=head._mask_flow_speed_padding(state+alpha*(prediction-state))
        if freeze_speed and result.shape[-2]>poses:
            result=result.clone();result[...,poses:,:]=anchor[...,poses:,:]
        return result

    nominal=[anchor.clone()]
    if reference_states is None:
        for j in range(count):
            nominal.append(update(nominal[-1],endpoint(nominal[-1],times[j]),j))
    else:
        nominal=[torch.as_tensor(x,device=ego.device,dtype=anchor.dtype) for x in reference_states]
        if len(nominal)!=count+1 or any(x.shape!=anchor.shape for x in nominal):
            raise ValueError('Reference states must match the schedule and selected anchor shape')
    # Use the same normalized direction bank at every injection time and model.
    rng=torch.Generator(device=ego.device).manual_seed(seed)
    bank=torch.randn((directions,poses,2),device=ego.device,generator=rng)
    bank=bank/bank.flatten(1).norm(dim=-1)[:,None,None]
    trials=[];gains=[[[] for _ in range(count+1)] for _ in range(count)]
    for injection in range(count):
        clean=[nominal[injection]]
        first_endpoint=endpoint(clean[0],times[injection])
        clean.append(update(clean[0],first_endpoint,injection))
        for index in range(injection+1,count):
            clean.append(update(clean[-1],endpoint(clean[-1],times[index]),index))
        for direction in range(directions):
            delta=torch.zeros_like(anchor)
            delta[...,:poses,:]=magnitude*bank[direction]*std[:poses]
            y=nominal[injection]+delta
            denominator=norm(delta)
            curve=[None]*(count+1);curve[injection]=1.
            gains[injection][injection].append(1.)
            prediction=endpoint(y,times[injection])
            endpoint_gain=norm(prediction-first_endpoint)/denominator
            y=update(y,prediction,injection)
            chi=norm(y-clean[1])/denominator
            for k in range(injection+1,count+1):
                if k>injection+1:y=update(y,endpoint(y,times[k-1]),k-1)
                gain=norm(y-clean[k-injection])/denominator
                gains[injection][k].append(gain);curve[k]=gain
            alpha=float((times[injection+1]-times[injection])/(1-times[injection]))
            trials.append({'injection_step':injection,'direction':direction,
                           'delta_normalized_norm':denominator,'chi_j':chi,
                           'endpoint_directional_gain':endpoint_gain,
                           'triangle_gain_bound':1-alpha+alpha*endpoint_gain,
                           'propagation_gain':curve,'terminal_gain':curve[-1]})
    def reduce(quantile=None):
        return [[None if not cell else float(torch.tensor(cell).mean() if quantile is None
                                             else torch.quantile(torch.tensor(cell),quantile))
                 for cell in row] for row in gains]
    return {'experiment':'fixed condition and anchor; one-time path-state injection',
            'magnitude_S_norm':magnitude,'seed':seed,'direction_count':directions,'mode':mode,
            'condition_unchanged':True,'anchor_unchanged':True,
            'injection_state_source':'own nominal rollout' if reference_states is None else 'shared reference state grid',
            'injection_states':[x.cpu().tolist() for x in nominal],
            'norm_scope':'route_only_speed_fixed' if freeze_speed else 'joint_route_and_speed',
            'production_rollout':not freeze_speed,
            'tau':times.cpu().tolist(),'normalized_direction_bank':bank.cpu().tolist(),
            'propagation_gain_mean':reduce(),'propagation_gain_p95':reduce(.95),
            'trials':trials,
            'heatmap_axes':'rows=injection integration step; columns=observed integration state',
            'interpretation':'sampled directional update gain; chi<1 is local empirical attenuation, '
                             'not proof of contraction in every direction'}
