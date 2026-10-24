> FMSR source-release note: module and model-attribute names have changed. Compatibility with historical checkpoint keys has not been validated.

# Anchor-relative Residual Flow + Contractive Recovery

FMSR's backbone, anchors, shared anchor encoder, time embedding,
classifier, decoder parameters and checkpoint keys are preserved. Select
`trajectory_generator=residual_flow` (default) or `trajectory_generator=ddbm`.
For old DDBM checkpoints lacking this selector, explicitly set `ddbm` at
inference. A DDBM checkpoint can initialize Flow, but must be trained with the
new objective before evaluating Flow performance.

## Clean Flow supervision and predicted-mode metrics

The CARLA anchor bank is `[60, 10, 2]`: spatial route points, not time-indexed
FlowDrive poses. For every scene, sample `tau ~ U(0, flow_train_t_max)` and form
all modes on an **unperturbed** path:

```text
x_tau = (1 - tau) * anchor + tau * GT
endpoint = decoder(x_tau, anchor, scene, 1000 * (1 - tau))
gt_mode = argmin_k mean_i ||GT_i - anchor_ki||_2
pred_mode = argmax(classifier(anchor, scene))
remain = max(1 - tau, flow_min_remain)
v_pred = (endpoint[gt_mode] - x_tau[gt_mode]) / remain
v_target = (GT - x_tau[gt_mode]) / remain
L_FM = mean(abs((v_pred - v_target) / std))
best_reg = endpoint[pred_mode]
```

The GT-nearest mode uses the same mean pointwise L2 criterion as the original
classification target. The classification focal loss is unchanged.
`best_reg`, `flow_endpoint_l1`, route ADE/FDE and visualization use the predicted
mode, never the oracle mode. Training metrics still use GT-interpolated inputs;
they are **not** free-rollout or closed-loop metrics.

`flow_min_remain=0.05` is independent of both solvers. With
`flow_train_t_max=0.95` the clamp is inactive (up to rounding), so the clean target
is exactly the anchor-relative residual. Changing `step_num` or
`flow_inference_steps` does not change the training objective.

## Original Euler-ratio Contractive Recovery

CR is off by default. When enabled, select
`max(1, int(batch_size * flow_contract_fraction))` scenes: 4 out of 32 by default.
Only their GT-nearest modes receive perturbations; all 60 modes remain present
in the additional decoder call.

The perturbation is expressed in each trajectory's local tangent/normal frame,
using centered differences internally and one-sided differences at the ends.
Its magnitude ramps from exactly zero at the first point to the sampled signed
lateral and longitudinal offsets at the last point. Degenerate tangents use the
scene chord, or ego-forward for a stationary trajectory. An optional appended
`[speed, 0]` token is never perturbed or included in contraction distances.

```text
pert_xt = clean_xt + delta
K ~ Uniform({4, 8, 20})       # independently for each selected scene
dt = min(1 / K, 1 - tau)
clean_next = stop_gradient(clean_xt + dt * (clean_endpoint - clean_xt) / (1 - tau))
pert_next = pert_xt + dt * (pert_endpoint - pert_xt) / (1 - tau)
d0 = mean_i ||delta_i||_2
d1 = mean_i ||pert_next_i - clean_next_i||_2
A = d1 / (d0 + 1e-4)
L_CR = mean(relu(A - rho))
L_flow = L_FM + flow_contract_weight * L_CR
```

The clean endpoint is reused. CR adds one small-batch decoder forward, no
additional backbone forward or rollout. Explicit decoder dropout masks are
shared between clean samples and their perturbed counterparts, so stochastic
noise is not mistaken for perturbation amplification. This pairing requires
`tf_dropout=0` for internal attention dropout, which is the supplied default;
nonzero attention dropout with CR is rejected. BF16 neural kernels may still
introduce small rounding differences between full-batch and subset calls; FP32
dynamics do not eliminate neural-network quantization. The classifier is unchanged.
CR gradients pass through the perturbed branch; the clean next-state target is
detached. Shared network weights are still updated by CR.

The trainer continues to receive `loss_ddbm_reg = L_flow` exactly once and the
original `loss_ddbm_cls`. There is no extra trainer loss key, LAR/PAC/RPR branch,
or second rollout in this original Euler-ratio variant. Flow keeps final-layer
regression supervision, with gradients propagated through both refinement
layers. DDBM keeps the original inter-layer trajectory detach.

## PDM TIE spectral CR repair (2026-10-06)

The PDM TIE launchers now default to `flow_contract_method=endpoint_jacobian`,
using the existing `tie_cr_fd` head name and unchanged checkpoint parameter keys.
For the ten route points, central differences form the complete 20x20 matrix
`S^-1 (df/dx) S`; SVD supplies its largest singular value. Input perturbations
are `epsilon * std`, and endpoint differences are divided by `2 * epsilon * std`.
The appended speed token is fixed while this route Jacobian is computed.
`endpoint_fd` remains available only when explicitly selecting the older
directional objective; loading an old config does not rewrite its loss selector.

CR queries mix nominal bridge states with genuine inference rollout states.
The latter start at an anchor, execute the same Euler solver and inference time
grid with FP32 neural kernels and disabled decoder dropout, then sample a query
index from 0 through 19 for the default 20-step solver. Rollout prefixes are
detached, require no expert trajectory, and use no displacement cap. Their
state and time correspond to the same integration prefix. Independent anchor
queries produce the same states as the corresponding modes in batched inference.
Both the directional and spectral TIE heads, including `tie_cr_v2`, use these
actual rollout states.

The user's four-bin kappa schedule is retained as `[1.30, 1.20, 1.08, 1.00]`.
This is a relaxed spectral gain penalty; it does not impose the strict
`kappa < 1` condition used by the contraction theorem. Main TIE training keeps
the existing delay, ramp and interval weighting. Recovery launchers enable
spectral CR by default and start its ramp immediately; explicit
`FLOW_CONTRACT_ENABLE=false` remains an FM-only ablation.

Continuous PDM speed uses `fmsr_max_speed=30.0` m/s independently of
TF++ categorical speed centers. Model decoding and the sensor controller use
the same bound, which is stored in new configs and defaults to 30 for old
configs lacking the field. Anchor speeds above the configured continuous bound
are rejected. This preserves the existing 24 m/s anchors without capping them
to the old 20 m/s classification center.

Spectral diagnostics include `flow_contract_sigma_max`, scheduled-kappa success,
and per-bin `flow_contract_sigma_tau0` through `tau3`. The Jacobian is computed
in chunks with activation checkpointing; full spectral CR and rollout prefixes
require more work than a single directional pair. Unit and DDP smoke checks do
not establish closed-loop scores.

The main delayed TIE entry point is
`scripts/train_fmsr_pdm_lite_tie_cr_fd_tau_delayed.sh`; new runs use
timestamped `tie_spectral` output names. This entry point trains the legacy
residual-flow planner, not `fmsr_pdm_lite/`'s VP clean-prior planner.

## Precision and inference

Neural layers retain the surrounding BF16 autocast policy on the RTX 4090.
Anchors, interpolated states, decoder coordinate additions, local-frame geometry,
velocities, losses, contraction distances and Euler updates are FP32. Coordinate
updates are not written back into BF16 decoder output buffers. Trainer loss
accumulation is also FP32.

Flow inference integrates all 60 modes from their anchors, then selects the
classifier's top mode:

```text
x = anchor
for tau, tau_next in adjacent_pairs(linspace(0, 1, flow_inference_steps + 1)):
    endpoint = decoder(x, anchor, scene, 1000 * (1 - tau))
    x += (tau_next - tau) * (endpoint - x) / (1 - tau)
```

There is no decoder call at `tau=1`. DDBM still uses `step_num=20` and its original
scheduler. Flow exclusively uses `flow_inference_steps` (20, 8 or 4), so one
trained checkpoint can be evaluated at all three resolutions. Closed-loop
`sensor_agent_fmsr.py` reads these settings independently from
`FMSR_CLOSED_LOOP_CONFIG`; open-loop inference uses the training config.

## Defaults and logging

```python
trajectory_generator = "residual_flow"
step_num = 20
flow_train_t_max = 0.95
flow_min_remain = 0.05
flow_inference_steps = 20
flow_contract_enable = False
flow_contract_fraction = 0.125
flow_contract_weight = 0.01
flow_contract_rho = 0.98
flow_contract_lat_min_m = 0.02
flow_contract_lat_max_m = 0.08
flow_contract_lon_max_m = 0.03
flow_contract_step_bank = (4, 8, 20)
```

TensorBoard diagnostics (all prefixed `metric/`): `flow_fm_loss`,
`flow_endpoint_l1`, `flow_tau_mean`, `flow_contract_loss`, `flow_contract_ratio`,
`flow_contract_success`, `route_ade`, `route_fde`. `flow_velocity_l1` remains an
alias of the clean FM loss for existing dashboards. Success is `P(A < 1)`.
CR diagnostics are zero when disabled; zero success then means **not measured**.
The old recovery configuration fields and mixed clean/perturbed objective have
been removed; old saved fields do not enable CR.

## Launch and validation

Run from `/home/vcdc/fmsr` in the `fmsr` environment. The `env -u PYTHONPATH`
prefix avoids the incompatible Python 3.7 CARLA egg inherited from `.bashrc`;
`scripts/main.sh` restores project paths. No global shell configuration changes
are needed.

```bash
# Preflight only: module locations and effective checkpoint-merged config.
env -u PYTHONPATH PREFLIGHT_ONLY=1 bash scripts/posttrain_fmsr.sh

# Stage A: original DDBM control.
env -u PYTHONPATH TRAJECTORY_GENERATOR=ddbm bash scripts/posttrain_fmsr.sh

# Stage B: clean anchor-relative Flow; CR remains off.
env -u PYTHONPATH bash scripts/posttrain_fmsr.sh

# Stage C: fresh fine-tuning run initialized from a good Stage B checkpoint.
# Replace the checkpoint path; the neighboring config.json must exist.
env -u PYTHONPATH FLOW_CONTRACT_ENABLE=true LEARNING_RATE=5e-5 \
  LOAD_FILE=outputs/local_training/STAGE_B_RUN/model_0030.pth \
  bash scripts/posttrain_fmsr.sh

# Tests, including a CUDA BF16 test when a compatible GPU is available.
env -u PYTHONPATH FMSR_PROJECT_ROOT="$PWD" bash -c \
  'source scripts/main.sh; python -m unittest discover -s tests -p test_residual_flow.py -v'
```

Each launch creates a separate timestamped output directory; `TRAIN_LOGDIR` can
override it. `PYTHON_BIN` selects the interpreter. `LOAD_FILE`, `LEARNING_RATE`,
`FLOW_CONTRACT_ENABLE`, `FLOW_CONTRACT_WEIGHT` and `FLOW_INFERENCE_STEPS` are
launcher overrides. The launcher starts fresh optimizer/scheduler state, including
Stage C; it does not resume a DDBM optimizer as Flow training.

For evaluation set `flow_inference_steps=20`, `8` or `4` in
`FMSR_CLOSED_LOOP_CONFIG`, keeping `step_num=20`. The supplied FMSR
benchmark scripts also accept `FLOW_INFERENCE_STEPS=8`; configure their
checkpoint and distinct output paths before launching. Closed-loop performance
and latency require actual training and benchmark runs; unit tests do not
establish them.
