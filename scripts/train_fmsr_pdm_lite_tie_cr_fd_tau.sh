#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_tie_cr_fd_tau_seed0}"
export FLOW_CONTRACT_TAU_KAPPA="${FLOW_CONTRACT_TAU_KAPPA:-1.30 1.20 1.08 1.00}"
exec bash "${ROOT}/scripts/train_fmsr_pdm_lite_tie_cr_fd.sh"
