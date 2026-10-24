#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# Fine-tune the retained network with FM and full spectral CR on recovery data.
export TFPP_CHECKPOINT="${TFPP_CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_tau_delayed_seed0/ckpt/model_0011.pth}"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_recovery_data_tie_cr_fd_epoch11}"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_tie_spectral_epoch11_recovery_seed0_$(date +%Y%m%d_%H%M%S)}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export FMSR_TRAJECTORY_HEAD=tie_cr_fd
export FLOW_CONTRACT_ENABLE="${FLOW_CONTRACT_ENABLE:-true}"
export FLOW_CONTRACT_METHOD="${FLOW_CONTRACT_METHOD:-endpoint_jacobian}"
export FLOW_CONTRACT_TAU_KAPPA="${FLOW_CONTRACT_TAU_KAPPA:-1.30 1.20 1.08 1.00}"
export FLOW_CONTRACT_FRACTION="${FLOW_CONTRACT_FRACTION:-0.25}"
export FLOW_CONTRACT_WEIGHT="${FLOW_CONTRACT_WEIGHT:-0.01}"
export FLOW_CONTRACT_INTERVAL="${FLOW_CONTRACT_INTERVAL:-4}"
export FLOW_CONTRACT_WARMUP_STEPS="${FLOW_CONTRACT_WARMUP_STEPS:-0}"
export FLOW_CONTRACT_DELAY_STEPS="${FLOW_CONTRACT_DELAY_STEPS:-0}"
export FLOW_CONTRACT_RAMP_STEPS="${FLOW_CONTRACT_RAMP_STEPS:-0}"
export FMSR_MAX_SPEED="${FMSR_MAX_SPEED:-30.0}"
export FLOW_RECOVERY_WEIGHT=0

# Compressed scenarios may be unmounted after a reboot. Mount their read-only
# view locally because unprivileged FUSE mounts cannot be placed on exFAT.
PDM_DATA_ROOT="$("${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}" \
  "${ROOT}/scripts/pdm_lite/mount_carla_garage_data.py" \
  --data-root "${PDM_DATA_ROOT:-/media/vcdc/T7/carla_garage_data}" \
  --view-root "${ROOT}/outputs/local_dataset_mounts/carla_garage_data")"
export PDM_DATA_ROOT

exec bash "${ROOT}/scripts/posttrain_recovery_pdm_lite.sh"
