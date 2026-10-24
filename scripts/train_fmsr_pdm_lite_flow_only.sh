#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PDM_DATA_ROOT="${PDM_DATA_ROOT:-${ROOT}/outputs/local_dataset_mounts/carla_garage_data}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T71/pdm_lite_assets}"
# This is a fresh ablation, not a continuation of the caller's recovery run.
# Use a dedicated override so an exported checkpoint from an old experiment
# cannot silently initialize a different policy or trigger recovery loading.
export TFPP_CHECKPOINT="${FLOW_ONLY_INIT_CHECKPOINT:-/media/vcdc/T71/tfpp_pretrained/pretrained_models/all_towns/model_0030_2.pth}"
export TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${ROOT}/outputs/local_training}"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_flow_only_$(date +%Y%m%d_%H%M%S)}"
export CONTINUE_EPOCH=0 RESET_SCHEDULER=0
export EPOCHS="${EPOCHS:-10}" BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}" COSINE_T0="${FLOW_ONLY_COSINE_T0:-${EPOCHS}}" SEED="${SEED:-0}"
export TRAJECTORY_GENERATOR=residual_flow FMSR_TRAJECTORY_HEAD=flow_only
export FMSR_MAX_SPEED="${FMSR_MAX_SPEED:-30.0}"
# Fixed FM-only ablation; inherited CR settings cannot reactivate a loss.
export FLOW_CONTRACT_ENABLE=0 FLOW_CONTRACT_WEIGHT=0 FLOW_RECOVERY_WEIGHT=0
export FLOW_CONTRACT_METHOD=euler_ratio
unset FLOW_CONTRACT_TAU_KAPPA FLOW_CONTRACT_MAX_PERTURB_M
unset RECOVERY_DATA_ROOT RECOVERY_SAMPLING_PLAN RECOVERY_FRACTION RECOVERY_MAX_REPEATS

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exec env -u PYTHONPATH bash "${ROOT}/scripts/train_fmsr_pdm_lite_flow_only_base.sh"
fi

mkdir -p "${TRAIN_LOG_ROOT}"
OUTPUT_DIR="${TRAIN_LOG_ROOT}/${RUN_ID}"
mkdir "${OUTPUT_DIR}" || { echo "Run directory exists: ${OUTPUT_DIR}" >&2; exit 2; }
"${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}" - "${OUTPUT_DIR}" <<'PY'
import json
import os
from pathlib import Path
import sys

keys = (
    'TFPP_CHECKPOINT', 'PDM_DATA_ROOT', 'PDM_ASSET_DIR', 'EPOCHS',
    'BATCH_SIZE', 'LEARNING_RATE', 'COSINE_T0', 'SEED',
    'TRAJECTORY_GENERATOR', 'FMSR_TRAJECTORY_HEAD', 'FMSR_MAX_SPEED',
    'FLOW_CONTRACT_ENABLE', 'FLOW_CONTRACT_WEIGHT', 'FLOW_RECOVERY_WEIGHT',
)
variant = {key: os.environ[key] for key in keys}
Path(sys.argv[1], 'variant.json').write_text(json.dumps(variant, indent=2) + '\n')
PY
echo "Flow Matching only run: ${OUTPUT_DIR}"
set +e
env -u PYTHONPATH PYTHONUNBUFFERED=1 bash "${ROOT}/scripts/train_fmsr_pdm_lite_flow_only_base.sh" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
result=${PIPESTATUS[0]}
printf '%s\n' "$result" > "${OUTPUT_DIR}/train.exitcode"
exit "$result"
