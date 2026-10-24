#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
: "${TRAIN_RUN_DIR:?Set TRAIN_RUN_DIR to the ongoing training directory}"
: "${EVALUATION_OUTPUT_DIR:?Set a separate evaluation result directory}"
export PYTHONPATH="$ROOT:$ROOT/vendor/garage:$WORKSPACE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
exec "$PYTHON_BIN" -m fmsrpdm.train_then_evaluate --run-dir "$TRAIN_RUN_DIR" \
  --output "$EVALUATION_OUTPUT_DIR" --epoch "${TARGET_EPOCH:-9}" --route-count 220 \
  --gpu "${GPU_ID:-0}" --carla-port "${CARLA_PORT:-8870}" --tm-port "${TM_PORT:-8040}" \
  --poll-seconds "${WATCH_POLL_SECONDS:-15}" --retry-seconds "${WATCH_RETRY_SECONDS:-30}" \
  --stale-seconds "${WATCH_STALE_SECONDS:-900}" --eval-timeout "${EVAL_TIMEOUT:-120}" \
  --background "$@"
