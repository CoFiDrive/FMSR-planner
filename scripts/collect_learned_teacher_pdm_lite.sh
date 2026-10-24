#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export FMSR_PROJECT_ROOT="$ROOT"
export CARLA_ROOT="$ROOT/3rd_party/CARLA_0915"
export LEADERBOARD_ROOT="$ROOT/3rd_party/Bench2Drive/leaderboard"
export SCENARIO_RUNNER_ROOT="$ROOT/3rd_party/Bench2Drive/scenario_runner"
export PYTHONPATH="$ROOT/3rd_party/carla_garage/team_code:$ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$CARLA_ROOT/PythonAPI/carla:$ROOT/scripts/pdm_lite"
export LD_LIBRARY_PATH=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export IS_BENCH2DRIVE=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export DIRECT=1 COMPILE=0 DEBUG_CHALLENGE=0
export CARLA_RENDER_OFFSCREEN=1 CARLA_RELOAD_EACH_ROUTE=1
export PLANNER_TYPE=only_traj
export FMSR_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 flow_inference_steps=20 diffusion_speed=False produce_debug_video=False produce_debug_image=False produce_demo_video=False produce_demo_image=False"
export FMSR_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"
unset FMSR_DIAGNOSTICS
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${EVALUATION_OUTPUT_DIR:-$ROOT/outputs/local_evaluation/joint_teacher0030_student0002_route2204_$RUN_TAG}"
COLLECTION="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_learned_teacher/student0002_teacher0030_route2204_$RUN_TAG}"
mkdir -p "$OUTPUT/debug_checkpoint"
OUTPUT="$(cd "$OUTPUT" && pwd)"
exec 9>"$OUTPUT/.evaluation.lock"
flock -n 9 || { echo "Collection already running for $OUTPUT" >&2; exit 75; }
export SAVE_PATH="$OUTPUT"
"$PYTHON_BIN" scripts/pdm_lite/joint_collection.py prepare \
  --output "$OUTPUT" --collection "$COLLECTION" --route-ids "${COLLECTION_ROUTE_IDS:-2204}" \
  --student "${STUDENT_CHECKPOINT:-$ROOT/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_20261003_181046/model_0002.pth}" \
  --teacher "${TEACHER_CHECKPOINT:-$ROOT/outputs/local_training/posttrain_fmsr_k60_4090_32_residual_flow_20260917_160256_45841/model_0030.pth}"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
"$PYTHON_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_v2.py" \
  --routes="$OUTPUT/routes.xml" --track=SENSORS \
  --checkpoint="$OUTPUT/checkpoint_endpoint.json" \
  --agent="$ROOT/3rd_party/carla_garage/team_code/learned_teacher_recovery_agent.py" \
  --agent-config="$OUTPUT/collection_config.json" \
  --debug=0 --record= --resume=True \
  --port="${CARLA_PORT:-8880}" --traffic-manager-port="${TM_PORT:-8030}" \
  --timeout="${EVAL_TIMEOUT:-180}" --traffic-manager-seed=0 --repetitions=1 \
  --debug-checkpoint="$OUTPUT/debug_checkpoint/live.txt" \
  2>&1 | tee -a "$OUTPUT/collection.log"
"$PYTHON_BIN" scripts/pdm_lite/joint_collection.py verify --output "$OUTPUT" \
  2>&1 | tee "$OUTPUT/verification.log"
