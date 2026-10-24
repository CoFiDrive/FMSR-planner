#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${CR_SOURCE_DIR:-${ROOT}/outputs/local_training/fmsr_pdm_lite_tie_cr_epoch05_snapshot_20261001}"

for name in model_0005.pth optimizer_0005.pth scheduler_0005.pth scaler_0005.pth config.json; do
  if [[ ! -s "${SOURCE_DIR}/${name}" ]]; then
    echo "Missing resume asset: ${SOURCE_DIR}/${name}" >&2
    exit 2
  fi
done

# Run epoch 6 from an isolated epoch-5 snapshot. The optimizer and scheduler
# are restored, and the live training run remains untouched. A fresh process
# does not restore its earlier RNG state, so compare trends across runs.
export PDM_DATA_ROOT="${PDM_DATA_ROOT:-/media/vcdc/T7/carla_garage_data}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T7/pdm_lite_assets}"
export TFPP_CHECKPOINT="${SOURCE_DIR}/model_0005.pth"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_tie_cr_weight_probe_$(date +%Y%m%d_%H%M%S)}"
export CONTINUE_EPOCH=1
export RESET_SCHEDULER=0
export EPOCHS=7
export BATCH_SIZE=32
export LEARNING_RATE=1e-4
export COSINE_T0=20
export TRAJECTORY_GENERATOR=residual_flow
export FLOW_CONTRACT_ENABLE=1
export FLOW_CONTRACT_METHOD=endpoint_jacobian
export FLOW_CONTRACT_FRACTION=0.0625
export FLOW_CONTRACT_WEIGHT="${FLOW_CONTRACT_WEIGHT:-0.001}"
export FLOW_CONTRACT_KAPPA=0.98
export FLOW_CONTRACT_FD_EPSILON=0.1
export FLOW_CONTRACT_FD_CHUNK=4
export FLOW_CONTRACT_INTERVAL=8
# The source run already finished its first 1,000-step CR warmup.
export FLOW_CONTRACT_WARMUP_STEPS=0

exec bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh"
