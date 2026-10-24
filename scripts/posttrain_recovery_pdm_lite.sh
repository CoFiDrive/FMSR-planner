#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PDM_DATA_ROOT="${PDM_DATA_ROOT:-/media/vcdc/T7/carla_garage_data}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T7/pdm_lite_assets}"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_recovery_data}"
export TFPP_CHECKPOINT="${TFPP_CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_flow_fixed_bs64_20260929_092208/model_0009.pth}"
export TRAJECTORY_GENERATOR=residual_flow
export FMSR_TRAJECTORY_HEAD="${FMSR_TRAJECTORY_HEAD:-baseline}"
export FLOW_CONTRACT_ENABLE="${FLOW_CONTRACT_ENABLE:-true}"
export FLOW_CONTRACT_METHOD="${FLOW_CONTRACT_METHOD:-endpoint_jacobian}"
export CONTINUE_EPOCH=0
export EPOCHS="${EPOCHS:-3}"
export COSINE_T0="${COSINE_T0:-${EPOCHS}}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export RECOVERY_FRACTION="${RECOVERY_FRACTION:-0.25}"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_flow_recovery_seed0_$(date +%Y%m%d_%H%M%S)}"

exec bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh"
