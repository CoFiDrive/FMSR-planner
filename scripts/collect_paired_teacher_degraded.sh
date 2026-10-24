#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export FMSR_PROJECT_ROOT="$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-$ROOT/outputs/local_evaluation/model0002_teacher0030_paired_degraded}"
export FMSR_DIAGNOSTIC_ROOT="$EVALUATION_OUTPUT_DIR"
export FMSR_DIAGNOSTICS=1 FMSR_DIAGNOSTIC_SENSOR_EVERY=1
export CHECKPOINT="$ROOT/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_20261003_181046/model_0002.pth"
export PAIRED_TEACHER_DIR="$ROOT/outputs/local_training/posttrain_fmsr_k60_4090_32_residual_flow_20260917_160256_45841/epoch30"
export ROUTES="$EVALUATION_OUTPUT_DIR/selected_routes.xml"
mkdir -p "$EVALUATION_OUTPUT_DIR"
exec 8>"$EVALUATION_OUTPUT_DIR/.paired_collection.lock"
flock -n 8 || { echo 'Paired collection already running' >&2; exit 75; }
trap 'code=$?; echo "$code" > "$EVALUATION_OUTPUT_DIR/exit_code.txt"' EXIT
"$PYTHON_BIN" - "$ROOT" "$ROUTES" <<'PY'
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
root, output = map(Path, sys.argv[1:])
tree = ET.parse(root / 'outputs/local_evaluation/bench2drive_model0002_degraded_diagnostics/selected_routes.xml')
routes = tree.getroot()
ordered = sorted(list(routes), key=lambda r: (r.get('id') != '2204', list(routes).index(r)))
routes[:] = ordered
if not output.exists():
    tree.write(output, encoding='utf-8', xml_declaration=True)
PY
PREFLIGHT_ONLY=1 bash scripts/eval_bench2drive_pdm_lite_fmsr.sh
export CARLA_ROOT="$ROOT/3rd_party/CARLA_0915"
export LEADERBOARD_ROOT="$ROOT/3rd_party/Bench2Drive/leaderboard"
export SCENARIO_RUNNER_ROOT="$ROOT/3rd_party/Bench2Drive/scenario_runner"
export PYTHONPATH="$ROOT/3rd_party/carla_garage/team_code:$ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export LD_LIBRARY_PATH="" PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
export IS_BENCH2DRIVE=1 CHALLENGE_TRACK_CODENAME=SENSORS DIRECT=1 DEBUG_CHALLENGE=0 COMPILE=0
export CARLA_RENDER_OFFSCREEN=1 CARLA_RELOAD_EACH_ROUTE=1 PLANNER_TYPE=only_traj
export SAVE_PATH="$EVALUATION_OUTPUT_DIR"
export FMSR_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 flow_inference_steps=20 diffusion_speed=False produce_debug_video=False produce_debug_image=False produce_demo_video=False produce_demo_image=False"
export FMSR_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"
"$PYTHON_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_v2.py" \
  --routes="$ROUTES" --track=SENSORS --checkpoint="$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json" \
  --agent="$ROOT/3rd_party/carla_garage/team_code/paired_teacher_agent.py" \
  --agent-config="$EVALUATION_OUTPUT_DIR/agent" --debug=0 --record=None --resume=True \
  --port="${CARLA_PORT:-8870}" --traffic-manager-port="${TM_PORT:-8020}" --timeout=180 \
  --debug-checkpoint="$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt" \
  --traffic-manager-seed=0 --repetitions=1 2>&1 | tee -a "$EVALUATION_OUTPUT_DIR/collection.log"
"$PYTHON_BIN" scripts/pdm_lite/paired_data.py "$EVALUATION_OUTPUT_DIR"
