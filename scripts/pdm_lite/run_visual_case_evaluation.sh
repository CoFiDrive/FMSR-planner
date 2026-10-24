#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ROOT=/home/vcdc/fmsr
cd "$ROOT"
PYTHON_BIN=/home/vcdc/anaconda3/envs/fmsr/bin/python
export CARLA_ROOT="$ROOT/3rd_party/CARLA_0915"
export LEADERBOARD_ROOT="$ROOT/3rd_party/Bench2Drive/leaderboard"
export SCENARIO_RUNNER_ROOT="$ROOT/3rd_party/Bench2Drive/scenario_runner"
GARAGE="$ROOT/3rd_party/carla_garage/team_code"
if [[ "$VISUAL_MODEL" == FMSR ]]; then GARAGE="$ROOT/fmsr_pdm_lite/vendor/garage"; fi
export PYTHONPATH="$GARAGE:$ROOT/fmsr_pdm_lite:$ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla"
export LD_LIBRARY_PATH=""
export CUDA_VISIBLE_DEVICES=0 IS_BENCH2DRIVE=1 CHALLENGE_TRACK_CODENAME=SENSORS
export DIRECT=1 DEBUG_CHALLENGE=0 COMPILE=0 CARLA_RENDER_OFFSCREEN=1
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export SAVE_PATH="$EVALUATION_OUTPUT_DIR" EVAL_METRIC_ROOT="$EVALUATION_OUTPUT_DIR"
export FMSR_PROJECT_ROOT="$ROOT" FMSR_DIAGNOSTICS=1
export FMSR_DIAGNOSTIC_ROOT="$EVALUATION_OUTPUT_DIR" FMSR_DIAGNOSTIC_SENSOR_EVERY=0
export EVAL_CAPTURE_ALL_METRICS=1 EVAL_TRACE_EVERY=1
export EVAL_PROBE_EVERY="${VISUAL_PROBE_EVERY:-0}" EVAL_PROBE_MAX="${VISUAL_PROBE_MAX:-0}"
export CARLA_RELOAD_EACH_ROUTE=0 SEED=0 TM_SEED=0
mkdir -p "$EVALUATION_OUTPUT_DIR/debug_checkpoint"
exec 9>"$EVALUATION_OUTPUT_DIR/.evaluation.lock"
flock -n 9 || exit 75
"$PYTHON_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_v2.py" \
 --routes="$ROUTES" --track=SENSORS --checkpoint="$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json" \
 --agent="$ROOT/scripts/pdm_lite/visual_case_agent.py" --agent-config="$EVALUATION_OUTPUT_DIR/agent" \
 --debug=0 --record= --resume=True --port="$CARLA_PORT" --traffic-manager-port="$TM_PORT" \
 --traffic-manager-seed=0 --timeout=120 --repetitions=1 \
 --debug-checkpoint="$EVALUATION_OUTPUT_DIR/debug_checkpoint/live.txt"
