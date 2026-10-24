#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_tie_spectral_tau_delayed_seed${SEED:-0}_$(date +%Y%m%d_%H%M%S)}"
export FLOW_CONTRACT_TAU_KAPPA="${FLOW_CONTRACT_TAU_KAPPA:-1.30 1.20 1.08 1.00}"
# Batch size 32 gives 9227 updates per epoch on the current PDM-Lite split.
# Let FM train for two epochs, then increase lambda over the third epoch.
export FLOW_CONTRACT_DELAY_STEPS="${FLOW_CONTRACT_DELAY_STEPS:-18454}"
export FLOW_CONTRACT_RAMP_STEPS="${FLOW_CONTRACT_RAMP_STEPS:-9227}"
export FLOW_CONTRACT_WARMUP_STEPS=0
# A heading perturbation can move a distant waypoint far more than its angle
# suggests. Cap each route waypoint's total physical displacement.
export FLOW_CONTRACT_MAX_PERTURB_M="${FLOW_CONTRACT_MAX_PERTURB_M:-0.15}"
exec bash "${ROOT}/scripts/train_fmsr_pdm_lite_tie_cr_fd.sh"
