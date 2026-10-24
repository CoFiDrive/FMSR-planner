#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
OUTPUT="${EVALUATION_OUTPUT_DIR:-$WORKSPACE/outputs/local_evaluation/bench2drive220_fmsr_pdm_lite_paper_seed0_epoch10}"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export PYTHONPATH="$ROOT:$ROOT/vendor/garage${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$PYTHON_BIN" -m fmsrpdm.evaluation_watchdog --output "$OUTPUT" \
  --gpu "${GPU_ID:-0}" --carla-port "${CARLA_PORT:-8870}" --tm-port "${TM_PORT:-8040}" \
  --poll-seconds "${WATCH_POLL_SECONDS:-15}" --retry-seconds "${WATCH_RETRY_SECONDS:-30}" \
  --background "$@"
