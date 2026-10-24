#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0  # keep simulator crash dumps from consuming checkpoint/result storage
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="$(cd "$ROOT/.." && pwd)"
: "${CHECKPOINT:?Set CHECKPOINT to a model produced by the new reproduction}"
OUTPUT="${EVALUATION_OUTPUT_DIR:-$ROOT/evaluations/$(basename "$CHECKPOINT" .pth)_$(date +%Y%m%d_%H%M%S)}"
ROUTES="${ROUTES:-$WORKSPACE/3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml}"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export CARLA_ROOT="$WORKSPACE/3rd_party/CARLA_0915"
export LEADERBOARD_ROOT="$WORKSPACE/3rd_party/Bench2Drive/leaderboard"
export SCENARIO_RUNNER_ROOT="$WORKSPACE/3rd_party/Bench2Drive/scenario_runner"
export PYTHONPATH="$ROOT:$ROOT/vendor/garage:$WORKSPACE:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla"
export FMSR_PROJECT_ROOT="$WORKSPACE" CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" PYTHONUNBUFFERED=1
export IS_BENCH2DRIVE=1 CHALLENGE_TRACK_CODENAME=SENSORS DIRECT=1 COMPILE=0 DEBUG_CHALLENGE=0
export CARLA_RENDER_OFFSCREEN=1 CARLA_RELOAD_EACH_ROUTE=0 FMSR_DIAGNOSTICS=0
export LD_LIBRARY_PATH=""
mkdir -p "$OUTPUT/debug_checkpoint"
OUTPUT="$(cd "$OUTPUT" && pwd)"
exec 9>"$OUTPUT/.evaluation.lock"
flock -n 9 || { echo "Evaluation already running for $OUTPUT" >&2; exit 75; }
export SAVE_PATH="$OUTPUT"
stage_args=()
if [[ "${ALLOW_EXPERIMENTAL_CLEAN_PRIOR:-0}" == 1 ]]; then stage_args+=(--allow-experimental-clean-prior); fi
if [[ "${VERIFY_EVAL_MODEL:-0}" == 1 ]]; then stage_args+=(--verify-model); fi
"$PYTHON_BIN" -m fmsrpdm.stage_evaluation --checkpoint "$CHECKPOINT" --routes "$ROUTES" --output "$OUTPUT" \
  --client-timeout "${EVAL_TIMEOUT:-120}" "${stage_args[@]}"
"$PYTHON_BIN" - "$ROOT/fmsrpdm/agent.py" <<'PY'
import importlib
from pathlib import Path
import sys
path = Path(sys.argv[1])
# Match leaderboard's top-level agent import before starting CARLA.
sys.path.insert(0, str(path.parent))
agent = importlib.import_module(path.stem)
print('Agent import preflight passed:', agent.get_entry_point(), flush=True)
PY
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
"$PYTHON_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_v2.py" \
  --routes="$ROUTES" --track=SENSORS --checkpoint="$OUTPUT/checkpoint_endpoint.json" \
  --agent="$ROOT/fmsrpdm/agent.py" --agent-config="$OUTPUT/agent" \
  --debug=0 --record= --resume=True --port="${CARLA_PORT:-8870}" \
  --traffic-manager-port="${TM_PORT:-8040}" --traffic-manager-seed=0 --timeout="${EVAL_TIMEOUT:-120}" \
  --repetitions=1 --debug-checkpoint="$OUTPUT/debug_checkpoint/live.txt" 2>&1 | tee -a "$OUTPUT/evaluation.log"
