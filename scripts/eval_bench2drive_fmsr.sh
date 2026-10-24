#!/bin/bash

# Checkpoints
# export CHECKPOINT_DIR=outputs/local_training/fmsr_k60
# export CHECKPOINT_DIR=/home/vcdc/disk2t/fmsr/outputs/eval_ckpt/posttrain_fmsr_0030
export CHECKPOINT_DIR=/home/vcdc/fmsr/outputs/local_training/fmsr_pdm_lite_residual_flow_k60_resume10to14_20260927_095359
# export ROUTES=data/benchmark_routes/bench2drive_split/bench2drive_111.xml
export ROUTES=3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml
# Set environment variables
export FMSR_PROJECT_ROOT=$(pwd)
export LD_LIBRARY_PATH=""
source $(pwd)/scripts/main.sh
export BENCHMARK_ROUTE_ID=$(basename $ROUTES .xml) # Last part of the route file name, e.g., 0 for 0.xml
# export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/$BENCHMARK_ROUTE_ID/
export EVALUATION_OUTPUT_DIR=outputs/local_evaluation/${BENCHMARK_ROUTE_ID}_posttrain_pdm/
export PYTHONPATH=3rd_party/Bench2Drive/leaderboard:$PYTHONPATH
export PYTHONPATH=3rd_party/Bench2Drive/scenario_runner:$PYTHONPATH
export SCENARIO_RUNNER_ROOT=3rd_party/Bench2Drive/scenario_runner
export LEADERBOARD_ROOT=3rd_party/Bench2Drive/leaderboard
export IS_BENCH2DRIVE=1
export PLANNER_TYPE=only_traj
export SAVE_PATH=$EVALUATION_OUTPUT_DIR/
export PYTHONUNBUFFERED=1
export FMSR_CLOSED_LOOP_CONFIG="steer_modality=route throttle_modality=target_speed brake_modality=target_speed step_num=20 flow_inference_steps=${FLOW_INFERENCE_STEPS:-20} diffusion_speed=False produce_debug_video=False produce_debug_image=False produce_demo_video=False produce_demo_image=False"
# export FMSR_TRAINING_CONFIG="diffusion_speed=False"
export FMSR_TRAINING_CONFIG="diffusion_speed=False plan_anchor_path=anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy"

set -x
set +e

# Recreate output folders
# rm -rf $EVALUATION_OUTPUT_DIR/
mkdir -p $EVALUATION_OUTPUT_DIR

CUDA_VISIBLE_DEVICES=0 python3 3rd_party/Bench2Drive/leaderboard/leaderboard/leaderboard_evaluator_v2.py \
    --routes=$ROUTES \
    --track=SENSORS \
    --checkpoint=$EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json \
    --agent=fmsr/inference/sensor_agent_fmsr.py \
    --agent-config=$CHECKPOINT_DIR \
    --debug=0 \
    --record=None \
    --resume=True \
    --port=8848 \
    --traffic-manager-port=8000 \
    --timeout=60 \
    --debug-checkpoint=$EVALUATION_OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt \
    --traffic-manager-seed=0 \
    --repetitions=1