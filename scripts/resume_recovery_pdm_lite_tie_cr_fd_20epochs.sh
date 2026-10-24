#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export TFPP_CHECKPOINT="${TFPP_CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_20261003_181046/model_0002.pth}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T71/pdm_lite_assets}"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_recovery_data_tie_cr_fd_epoch11}"

PDM_DATA_ROOT="$("${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/mount_carla_garage_data.py" \
  --data-root "${PDM_DATA_ROOT:-/media/vcdc/T71/carla_garage_data}" \
  --view-root "${ROOT}/outputs/local_dataset_mounts/carla_garage_data")"
export PDM_DATA_ROOT

export RUN_ID="${RUN_ID:-fmsr_pdm_lite_tie_spectral_recovery_resume3to22_$(date +%Y%m%d_%H%M%S)}"
export CONTINUE_EPOCH=1
export RESET_SCHEDULER=1
# --epochs is exclusive: resume model_0002 and train epochs 3 through 22.
export EPOCHS=23
export COSINE_T0=20
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export SEED="${SEED:-0}"
export GPU_ID="${GPU_ID:-0}"
export TRAJECTORY_GENERATOR=residual_flow
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
export RECOVERY_FRACTION="${RECOVERY_FRACTION:-0.25}"
export RECOVERY_MAX_REPEATS="${RECOVERY_MAX_REPEATS:-8}"

mkdir -p "${ROOT}/outputs/local_recovery_runs"
bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh" 2>&1 \
  | tee "${ROOT}/outputs/local_recovery_runs/${RUN_ID}.log"
