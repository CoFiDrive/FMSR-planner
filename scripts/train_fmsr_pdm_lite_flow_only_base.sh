#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GARAGE_ROOT="${ROOT}/3rd_party/carla_garage"
PYTHON_BIN=${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}

: "${PDM_DATA_ROOT:?Set PDM_DATA_ROOT to the extracted CARLA Garage PDM-Lite data directory}"
: "${TFPP_CHECKPOINT:?Set TFPP_CHECKPOINT to an official TF++ or trained PDM-Lite FMSR checkpoint}"

PDM_ASSET_DIR=${PDM_ASSET_DIR:-${ROOT}/anchor_utils/anchor_data/pdm_lite}
PDM_ANCHORS=${PDM_ANCHORS:-${PDM_ASSET_DIR}/pdm_cp_kmeans_spd_60_10.npy}
PDM_STATS=${PDM_STATS:-${PDM_ASSET_DIR}/pdm_cp_stats_spd_10.npz}
PDM_MANIFEST=${PDM_MANIFEST:-${PDM_ASSET_DIR}/pdm_lite_keyframes.txt}
TRAIN_LOG_ROOT=${TRAIN_LOG_ROOT:-${ROOT}/outputs/local_training}
TRAJECTORY_GENERATOR=${TRAJECTORY_GENERATOR:-residual_flow}
RUN_ID=${RUN_ID:-fmsr_pdm_lite_${TRAJECTORY_GENERATOR}_k60_$(date +%Y%m%d_%H%M%S)}
GPU_ID=${GPU_ID:-0}
CPU_CORES=${CPU_CORES:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
SEED=${SEED:-0}

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${ROOT}:${GARAGE_ROOT}/team_code:${ROOT}/3rd_party/CARLA_0915/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"

data_roots=("${PDM_DATA_ROOT}")
"${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/preflight.py" \
  --data-root "${PDM_DATA_ROOT}" \
  --checkpoint "${TFPP_CHECKPOINT}" \
  --anchors "${PDM_ANCHORS}" \
  --stats "${PDM_STATS}" \
  --manifest "${PDM_MANIFEST}"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

cd "${GARAGE_ROOT}/team_code"
exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=1 --max_restarts=0 \
  train.py \
  --id "${RUN_ID}" \
  --logdir "${TRAIN_LOG_ROOT}" \
  --root_dir "${data_roots[@]}" \
  --load_file "${TFPP_CHECKPOINT}" \
  --continue_epoch "${CONTINUE_EPOCH:-0}" \
  --reset_scheduler "${RESET_SCHEDULER:-0}" \
  --setting all \
  --epochs "${EPOCHS:-10}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE:-3e-4}" \
  --seed "${SEED}" \
  --cpu_cores "${CPU_CORES}" \
  --use_fmsr 1 \
  --fmsr_anchor_path "${PDM_ANCHORS}" \
  --fmsr_stats_path "${PDM_STATS}" \
  --fmsr_frame_manifest "${PDM_MANIFEST}" \
  --fmsr_steps 20 \
  --fmsr_max_speed "${FMSR_MAX_SPEED:-30.0}" \
  --trajectory_generator residual_flow \
  --fmsr_trajectory_head flow_only \
  --flow_train_t_max "${FLOW_TRAIN_T_MAX:-0.95}" \
  --flow_min_remain "${FLOW_MIN_REMAIN:-0.05}" \
  --flow_inference_steps "${FLOW_INFERENCE_STEPS:-20}" \
  --flow_contract_enable 0 \
  --flow_contract_weight 0 \
  --flow_recovery_weight 0 \
  --fmsr_tf_dropout "${FMSR_TF_DROPOUT:-0}" \
  --freeze_backbone 1 \
  --use_controller_input_prediction 1 \
  --use_wp_gru 0 \
  --use_semantic 0 \
  --use_depth 0 \
  --use_bev_semantic 0 \
  --detect_boxes 0 \
  --use_speed_weights 0 \
  --use_cosine_schedule 1 \
  --cosine_t0 "${COSINE_T0:-${EPOCHS:-10}}" \
  --zero_redundancy_optimizer 0 \
  --use_amp "${USE_AMP:-1}" \
  --use_disk_cache 0
