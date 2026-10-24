#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/fmsr_pdm_lite:$ROOT/fmsr_pdm_lite/vendor/garage:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
exec "${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}" \
  -m scripts.pdm_lite.evaluate_best_legacy_fm_full_metrics "$@"
