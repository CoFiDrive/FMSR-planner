#!/usr/bin/env bash
# PDM adapter of the author's v8 head; keep native PDM labels/assets explicit.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-author_v8_pdm_$(date +%Y%m%d_%H%M%S)}"
export CLASSIFICATION_LOSS="${CLASSIFICATION_LOSS:-focal}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-$TASK_ROOT/assets/paper_k60_seed0}"
if [[ -n "${INIT_CHECKPOINT:-}" ]]; then
  echo 'This architecture needs fresh TF++ initialization; do not warm-start legacy planner weights.' >&2
  exit 1
fi
exec bash "$TASK_ROOT/scripts/train.sh" \
  --implementation-profile author_v8_pdm \
  --regression-supervision "${REGRESSION_SUPERVISION:-gt_each_layer}" "$@"
