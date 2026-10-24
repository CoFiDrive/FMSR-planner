#!/usr/bin/env bash
# Matched v8 PDM architecture, clean-anchor FM-only objective and Euler sampler.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-author_v8_pdm_fm_$(date +%Y%m%d_%H%M%S)}"
export ROLLOUT_DENOISING_WEIGHT=0 ROLLOUT_GEOMETRY_WEIGHT=0 PATH_GEOMETRY_WEIGHT=0
export REGRESSION_SUPERVISION=gt_each_layer
exec bash "$TASK_ROOT/scripts/train_author_v8.sh" \
  --trajectory-generator flow_matching \
  --flow-train-t-max "${FLOW_TRAIN_T_MAX:-0.95}" \
  --flow-min-remain "${FLOW_MIN_REMAIN:-0.05}" "$@"
