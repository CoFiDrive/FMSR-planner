#!/bin/bash

set -u

export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/vcdc/fmsr/outputs/local_training/posttrain_fmsr_k60_4090_32_residual_flow_20260917_160256_45841}"
export ROUTES="${ROUTES:-data/benchmark_routes/bench2drive_split/bench2drive_failed28_posttrain0030.xml}"

export FMSR_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$FMSR_PROJECT_ROOT"

export LD_LIBRARY_PATH=""
source "$FMSR_PROJECT_ROOT/scripts/main.sh"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
export BENCHMARK_ROUTE_ID="$(basename "$ROUTES" .xml)"
export EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-outputs/local_evaluation/bench2drive_failed28_posttrain_0030_visual_${RUN_TAG}}"
export PYTHONPATH="3rd_party/Bench2Drive/leaderboard:${PYTHONPATH}"
export PYTHONPATH="3rd_party/Bench2Drive/scenario_runner:${PYTHONPATH}"
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export SAVE_PATH="$EVALUATION_OUTPUT_DIR/"
export PYTHONUNBUFFERED=1

# Render in a window by default. Set CARLA_RENDER_OFFSCREEN=1 to run headlessly.
export CARLA_RENDER_OFFSCREEN="${CARLA_RENDER_OFFSCREEN:-0}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"

export FMSR_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 flow_inference_steps=${FLOW_INFERENCE_STEPS:-20} diffusion_speed=False produce_debug_video=False produce_debug_image=False produce_demo_video=False produce_demo_image=False"
export FMSR_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python3}"

mkdir -p "$EVALUATION_OUTPUT_DIR/debug_checkpoint"
printf '%s\n' "$EVALUATION_OUTPUT_DIR" > outputs/local_evaluation/latest_failed28_visual_run.txt

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" 3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_v2.py \
    --routes="$ROUTES" \
    --track=SENSORS \
    --checkpoint="$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json" \
    --agent=fmsr/inference/sensor_agent_fmsr.py \
    --agent-config="$CHECKPOINT_DIR" \
    --debug=0 \
    --record=None \
    --port="${CARLA_PORT:-8849}" \
    --traffic-manager-port="${TM_PORT:-8001}" \
    --timeout=60 \
    --debug-checkpoint="$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt" \
    --traffic-manager-seed=0 \
    --repetitions=1
