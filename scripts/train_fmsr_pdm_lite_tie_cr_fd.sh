#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PDM_DATA_ROOT="${PDM_DATA_ROOT:-/media/vcdc/T7/carla_garage_data}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T7/pdm_lite_assets}"
export TFPP_CHECKPOINT="${TFPP_CHECKPOINT:-/media/vcdc/T7/tfpp_pretrained/pretrained_models/all_towns/model_0030_2.pth}"
export TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-${ROOT}/outputs/local_training}"
export RUN_ID="${RUN_ID:-fmsr_pdm_lite_tie_spectral_$(date +%Y%m%d_%H%M%S)}"
export CONTINUE_EPOCH=0 RESET_SCHEDULER=0
export EPOCHS="${EPOCHS:-20}" BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}" COSINE_T0="${COSINE_T0:-20}" SEED="${SEED:-0}"
export TRAJECTORY_GENERATOR=residual_flow FMSR_TRAJECTORY_HEAD=tie_cr_fd
export FLOW_CONTRACT_ENABLE=1
# Same network/checkpoint keys; full Jacobian + SVD is the repaired default.
export FLOW_CONTRACT_METHOD="${FLOW_CONTRACT_METHOD:-endpoint_jacobian}"
export FLOW_CONTRACT_FD_EPSILON="${FLOW_CONTRACT_FD_EPSILON:-0.1}"
export FLOW_CONTRACT_FD_CHUNK="${FLOW_CONTRACT_FD_CHUNK:-4}"
export FMSR_MAX_SPEED="${FMSR_MAX_SPEED:-30.0}"
export FLOW_CONTRACT_FRACTION="${FLOW_CONTRACT_FRACTION:-0.25}"
export FLOW_CONTRACT_WEIGHT="${FLOW_CONTRACT_WEIGHT:-0.01}"
export FLOW_CONTRACT_KAPPA="${FLOW_CONTRACT_KAPPA:-0.98}"
export FLOW_CONTRACT_TAU_KAPPA="${FLOW_CONTRACT_TAU_KAPPA:-1.30 1.20 1.08 1.00}"
export FLOW_CONTRACT_INTERVAL="${FLOW_CONTRACT_INTERVAL:-4}"
export FLOW_CONTRACT_WARMUP_STEPS="${FLOW_CONTRACT_WARMUP_STEPS:-1000}"
export FLOW_CONTRACT_DELAY_STEPS="${FLOW_CONTRACT_DELAY_STEPS:-0}"
export FLOW_CONTRACT_RAMP_STEPS="${FLOW_CONTRACT_RAMP_STEPS:-0}"
export FLOW_CONTRACT_PRED_MODE_FRACTION="${FLOW_CONTRACT_PRED_MODE_FRACTION:-0.5}"
export FLOW_CONTRACT_ON_POLICY_FRACTION="${FLOW_CONTRACT_ON_POLICY_FRACTION:-0.5}"
export FLOW_RECOVERY_WEIGHT=0
export FLOW_CONTRACT_LAT_MIN_M="${FLOW_CONTRACT_LAT_MIN_M:-0.02}"
export FLOW_CONTRACT_LAT_MAX_M="${FLOW_CONTRACT_LAT_MAX_M:-0.08}"
export FLOW_CONTRACT_LON_MAX_M="${FLOW_CONTRACT_LON_MAX_M:-0.03}"
export FLOW_CONTRACT_YAW_MAX_DEG="${FLOW_CONTRACT_YAW_MAX_DEG:-1.0}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exec env -u PYTHONPATH bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh"
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
    'FMSR_TRAJECTORY_HEAD', 'FLOW_CONTRACT_METHOD',
    'FLOW_CONTRACT_FD_EPSILON', 'FLOW_CONTRACT_FD_CHUNK', 'FMSR_MAX_SPEED',
    'FLOW_CONTRACT_FRACTION', 'FLOW_CONTRACT_WEIGHT', 'FLOW_CONTRACT_KAPPA',
    'FLOW_CONTRACT_TAU_KAPPA',
    'FLOW_CONTRACT_INTERVAL', 'FLOW_CONTRACT_WARMUP_STEPS',
    'FLOW_CONTRACT_DELAY_STEPS', 'FLOW_CONTRACT_RAMP_STEPS',
    'FLOW_CONTRACT_PRED_MODE_FRACTION', 'FLOW_CONTRACT_ON_POLICY_FRACTION',
    'FLOW_RECOVERY_WEIGHT',
    'FLOW_CONTRACT_LAT_MIN_M', 'FLOW_CONTRACT_LAT_MAX_M',
    'FLOW_CONTRACT_LON_MAX_M', 'FLOW_CONTRACT_YAW_MAX_DEG',
)
variant = {key: os.environ[key] for key in keys}
if os.environ.get('FLOW_CONTRACT_MAX_PERTURB_M'):
    variant['FLOW_CONTRACT_MAX_PERTURB_M'] = os.environ['FLOW_CONTRACT_MAX_PERTURB_M']
Path(sys.argv[1], 'variant.json').write_text(json.dumps(variant, indent=2) + '\n')
PY
echo "TIE ${FLOW_CONTRACT_METHOD} CR run: ${OUTPUT_DIR}"
set +e
env -u PYTHONPATH PYTHONUNBUFFERED=1 bash "${ROOT}/scripts/train_fmsr_pdm_lite.sh" \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
result=${PIPESTATUS[0]}
printf '%s\n' "$result" > "${OUTPUT_DIR}/train.exitcode"
exit "$result"
