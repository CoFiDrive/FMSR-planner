#!/usr/bin/env bash
# Complete the existing pilot audit, then collect the four requested routes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export FMSR_PROJECT_ROOT="$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
PILOT="$ROOT/outputs/local_evaluation/joint_teacher0030_student0002_route2204"
export EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-$ROOT/outputs/local_evaluation/joint_teacher0030_student0002_priority4}"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_learned_teacher/student0002_teacher0030_priority4}"
export COLLECTION_ROUTE_IDS=2215,24816,2115,2127
export CARLA_PORT="${CARLA_PORT:-8910}" TM_PORT="${TM_PORT:-8110}"
export PYTHONPATH="$ROOT/3rd_party/carla_garage/team_code:$ROOT:$ROOT/scripts/pdm_lite:$ROOT/3rd_party/Bench2Drive/leaderboard:$ROOT/3rd_party/Bench2Drive/scenario_runner:$ROOT/3rd_party/CARLA_0915/PythonAPI/carla"
mkdir -p "$EVALUATION_OUTPUT_DIR"
exec 8>"$EVALUATION_OUTPUT_DIR/.pipeline.lock"
flock -n 8 || { echo 'Priority collection pipeline already running' >&2; exit 75; }
exec > >(tee -a "$EVALUATION_OUTPUT_DIR/pipeline.log") 2>&1
trap 'code=$?; echo "$code" > "$EVALUATION_OUTPUT_DIR/pipeline_exit_code.txt"' EXIT
echo 'Verifying completed route 2204 before collecting priority routes'
"$PYTHON_BIN" scripts/pdm_lite/joint_collection.py verify --output "$PILOT"
"$PYTHON_BIN" scripts/pdm_lite/joint_event_samples.py --output "$EVALUATION_OUTPUT_DIR/event_samples" --evaluation "$PILOT"
echo 'Collecting routes in order: 2215, 24816, 2115, 2127'
bash scripts/collect_learned_teacher_pdm_lite.sh
"$PYTHON_BIN" scripts/pdm_lite/joint_event_samples.py --output "$EVALUATION_OUTPUT_DIR/event_samples" --evaluation "$PILOT" "$EVALUATION_OUTPUT_DIR"
echo 'Completed: all five routes verified and candidate event samples packaged'
