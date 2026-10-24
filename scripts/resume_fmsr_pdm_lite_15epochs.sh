#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_RUN="${ROOT}/outputs/local_training/fmsr_pdm_lite_residual_flow_k60_resume10to14_20260927_095359"
export PDM_DATA_ROOT="${PDM_DATA_ROOT:-/media/vcdc/T7/carla_garage_data}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T7/pdm_lite_assets}"
export TFPP_CHECKPOINT="${SOURCE_RUN}/model_0014.pth"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_residual_flow_k60_resume15to29_bs${BATCH_SIZE}_$(date +%Y%m%d_%H%M%S)}"
export CONTINUE_EPOCH=1
export RESET_SCHEDULER=1
# Fifteen additional epochs: indices 15 through 29 (completed epochs 16-30).
export EPOCHS=30
export COSINE_T0=15
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
export TRAJECTORY_GENERATOR=residual_flow
export FLOW_CONTRACT_ENABLE=false
export USE_AMP="${USE_AMP:-1}"
export TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${ROOT}/outputs/local_training}"

for file in model_0014.pth optimizer_0014.pth scaler_0014.pth config.json; do
  [[ -f "${SOURCE_RUN}/${file}" ]] || { echo "Missing continuation file: ${SOURCE_RUN}/${file}" >&2; exit 2; }
done
[[ "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "BATCH_SIZE must be a positive integer" >&2; exit 2; }

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exec env -u PYTHONPATH bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh"
fi

OUTPUT_DIR="${TRAIN_LOG_ROOT}/${RUN_ID}"
mkdir -p "$TRAIN_LOG_ROOT"
# Do not overwrite a previous run or its checkpoints.
mkdir "$OUTPUT_DIR" || { echo "Use a new RUN_ID; output directory already exists: $OUTPUT_DIR" >&2; exit 2; }
echo "Source checkpoint: $TFPP_CHECKPOINT"
echo "Epochs: 15-29; batch size: $BATCH_SIZE; learning rate: $LEARNING_RATE"
echo "Output directory: $OUTPUT_DIR"
set +e
env -u PYTHONPATH PYTHONUNBUFFERED=1 bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh" \
  2>&1 | tee "$OUTPUT_DIR/train.log"
result=${PIPESTATUS[0]}
printf '%s\n' "$result" > "$OUTPUT_DIR/train.exitcode"
exit "$result"
