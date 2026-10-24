#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export CHECKPOINT="${CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_20261003_181046/model_0002.pth}"
export EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${ROOT}/outputs/local_evaluation/bench2drive_model0002_degraded_diagnostics_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$EVALUATION_OUTPUT_DIR"
export EVALUATION_OUTPUT_DIR="$(cd "$EVALUATION_OUTPUT_DIR" && pwd)"

# Freeze the route selection once, so resume never changes the evaluated set.
if [[ -z "${ROUTES:-}" ]]; then
  export ROUTES="$EVALUATION_OUTPUT_DIR/selected_routes.xml"
  if [[ ! -f "$ROUTES" ]]; then
    SELECT_ARGS=()
    if [[ -n "${DIAGNOSTIC_ROUTE_ID:-}" ]]; then
      SELECT_ARGS+=(--route-id "$DIAGNOSTIC_ROUTE_ID")
    fi
    "$PYTHON_BIN" scripts/pdm_lite/diagnostic_evaluation.py select \
      --baseline "${BASELINE_RESULT:-${ROOT}/outputs/local_evaluation/bench2drive220_pdm_lite_tie_cr_fd_tau_delayed_seed0_epoch11/checkpoint_endpoint.json}" \
      --current "${CURRENT_RESULT:-${ROOT}/outputs/local_evaluation/bench2drive_failed96_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_model0002/checkpoint_endpoint.json}" \
      --routes "$ROOT/3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml" \
      --selection "${DIAGNOSTIC_SELECTION:-worse}" \
      --output "$ROUTES" --report "$EVALUATION_OUTPUT_DIR/route_selection.json" \
      "${SELECT_ARGS[@]}"
  fi
fi
export FMSR_DIAGNOSTICS=1
export FMSR_DIAGNOSTIC_SENSOR_EVERY="${FMSR_DIAGNOSTIC_SENSOR_EVERY:-1}"
export FMSR_DIAGNOSTIC_ACTOR_RADIUS="${FMSR_DIAGNOSTIC_ACTOR_RADIUS:-50}"
export CARLA_PORT="${CARLA_PORT:-8860}"
export TM_PORT="${TM_PORT:-8010}"
export CARLA_RELOAD_EACH_ROUTE="${CARLA_RELOAD_EACH_ROUTE:-1}"
bash scripts/eval_bench2drive_pdm_lite_fmsr.sh

if [[ "${PREFLIGHT_ONLY:-0}" != 1 ]]; then
  "$PYTHON_BIN" scripts/pdm_lite/diagnostic_evaluation.py verify --output-dir "$EVALUATION_OUTPUT_DIR"
fi
