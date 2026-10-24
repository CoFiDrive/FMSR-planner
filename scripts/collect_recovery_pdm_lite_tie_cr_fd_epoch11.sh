#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Use the retained snapshot: the training loop removes older top-level models.
export STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_tau_delayed_seed0/ckpt/model_0011.pth}"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_recovery_data_tie_cr_fd_epoch11}"

exec bash "${ROOT}/scripts/collect_recovery_pdm_lite.sh"
