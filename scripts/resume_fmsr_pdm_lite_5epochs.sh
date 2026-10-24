#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PDM_DATA_ROOT=/media/vcdc/T7/carla_garage_data
export PDM_ASSET_DIR=/media/vcdc/T7/pdm_lite_assets
export TFPP_CHECKPOINT="${ROOT}/outputs/local_training/fmsr_pdm_lite_residual_flow_k60_20260926_142355/model_0009.pth"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_residual_flow_k60_resume10to14_$(date +%Y%m%d_%H%M%S)}"
export CONTINUE_EPOCH=1
export RESET_SCHEDULER=1
# --epochs is the exclusive final epoch: run epochs 10, 11, 12, 13, 14.
export EPOCHS=15
export COSINE_T0=5
export LEARNING_RATE=3e-5
export TRAJECTORY_GENERATOR=residual_flow
export FLOW_CONTRACT_ENABLE=false
exec bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh"
