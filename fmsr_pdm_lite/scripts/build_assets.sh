#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/vendor/garage${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
PDM_DATA_ROOT="${PDM_DATA_ROOT:-$ROOT/../outputs/local_dataset_mounts/carla_garage_data}"
PDM_ASSET_DIR="${PDM_ASSET_DIR:-$ROOT/assets/paper_k60_seed0}"
exec "$PYTHON_BIN" -m fmsrpdm.assets --data-root "$PDM_DATA_ROOT" --output-dir "$PDM_ASSET_DIR" \
  --clusters "${CLUSTERS:-60}" --seed "${SEED:-0}" --validation-fraction "${VALIDATION_FRACTION:-0.05}" "$@"
