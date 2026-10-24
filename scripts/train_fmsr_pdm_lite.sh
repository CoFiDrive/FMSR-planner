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

# Accept the same true/false overrides as the FMSR launcher.
case "${FLOW_CONTRACT_ENABLE:-false}" in
  1|true|True) contract_enable=1 ;;
  0|false|False) contract_enable=0 ;;
  *) echo "FLOW_CONTRACT_ENABLE must be true/false or 1/0" >&2; exit 2 ;;
esac
read -r -a contract_steps <<< "${FLOW_CONTRACT_STEP_BANK:-4 8 20}"
tau_kappa_args=()
perturb_args=()
if [[ -n "${FLOW_CONTRACT_MAX_PERTURB_M:-}" ]]; then
  perturb_args=(--flow_contract_max_perturb_m "${FLOW_CONTRACT_MAX_PERTURB_M}")
fi
if [[ -n "${FLOW_CONTRACT_TAU_KAPPA:-}" ]]; then
  read -r -a tau_kappa_values <<< "${FLOW_CONTRACT_TAU_KAPPA}"
  if [[ "${#tau_kappa_values[@]}" -ne 4 ]]; then
    echo "FLOW_CONTRACT_TAU_KAPPA must contain exactly four thresholds" >&2
    exit 2
  fi
  tau_kappa_args=(--flow_contract_tau_kappa "${tau_kappa_values[@]}")
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${ROOT}:${GARAGE_ROOT}/team_code:${ROOT}/3rd_party/CARLA_0915/PythonAPI/carla${PYTHONPATH:+:${PYTHONPATH}}"

data_roots=("${PDM_DATA_ROOT}")
recovery_args=()
if [[ -n "${RECOVERY_DATA_ROOT:-}" ]]; then
  [[ -f "${RECOVERY_DATA_ROOT}/recovery_frames.txt" ]] || {
    echo "Missing recovery manifest: ${RECOVERY_DATA_ROOT}/recovery_frames.txt" >&2; exit 2;
  }
  data_roots+=("${RECOVERY_DATA_ROOT}")
  recovery_args=(--fmsr_recovery_root "${RECOVERY_DATA_ROOT}"
                 --fmsr_recovery_manifest "${RECOVERY_DATA_ROOT}/recovery_frames.txt"
                 --fmsr_recovery_fraction "${RECOVERY_FRACTION:-0.25}"
                 --fmsr_recovery_max_repeats "${RECOVERY_MAX_REPEATS:-8}")
  if [[ -n "${RECOVERY_SAMPLING_PLAN:-}" ]]; then
    [[ -f "$RECOVERY_SAMPLING_PLAN" ]] || { echo "Missing recovery sampling plan: $RECOVERY_SAMPLING_PLAN" >&2; exit 2; }
    recovery_args+=(--fmsr_recovery_sampling_plan "$RECOVERY_SAMPLING_PLAN")
  fi
fi

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
  "${recovery_args[@]}" \
  --fmsr_steps 20 \
  --fmsr_max_speed "${FMSR_MAX_SPEED:-30.0}" \
  --trajectory_generator "${TRAJECTORY_GENERATOR}" \
  --fmsr_trajectory_head "${FMSR_TRAJECTORY_HEAD:-baseline}" \
  --flow_train_t_max "${FLOW_TRAIN_T_MAX:-0.95}" \
  --flow_min_remain "${FLOW_MIN_REMAIN:-0.05}" \
  --flow_inference_steps "${FLOW_INFERENCE_STEPS:-20}" \
  --flow_contract_enable "${contract_enable}" \
  --flow_contract_method "${FLOW_CONTRACT_METHOD:-euler_ratio}" \
  --flow_contract_fraction "${FLOW_CONTRACT_FRACTION:-0.125}" \
  --flow_contract_weight "${FLOW_CONTRACT_WEIGHT:-0.01}" \
  --flow_contract_pred_mode_fraction "${FLOW_CONTRACT_PRED_MODE_FRACTION:-0.5}" \
  --flow_contract_on_policy_fraction "${FLOW_CONTRACT_ON_POLICY_FRACTION:-0.5}" \
  --flow_recovery_weight "${FLOW_RECOVERY_WEIGHT:-0.1}" \
  --flow_contract_rho "${FLOW_CONTRACT_KAPPA:-${FLOW_CONTRACT_RHO:-0.98}}" \
  --flow_contract_fd_epsilon "${FLOW_CONTRACT_FD_EPSILON:-0.1}" \
  --flow_contract_fd_chunk "${FLOW_CONTRACT_FD_CHUNK:-4}" \
  --flow_contract_interval "${FLOW_CONTRACT_INTERVAL:-1}" \
  --flow_contract_warmup_steps "${FLOW_CONTRACT_WARMUP_STEPS:-0}" \
  --flow_contract_delay_steps "${FLOW_CONTRACT_DELAY_STEPS:-0}" \
  --flow_contract_ramp_steps "${FLOW_CONTRACT_RAMP_STEPS:-0}" \
  "${perturb_args[@]}" \
  --flow_contract_lat_min_m "${FLOW_CONTRACT_LAT_MIN_M:-0.02}" \
  --flow_contract_lat_max_m "${FLOW_CONTRACT_LAT_MAX_M:-0.08}" \
  --flow_contract_lon_max_m "${FLOW_CONTRACT_LON_MAX_M:-0.03}" \
  --flow_contract_yaw_max_deg "${FLOW_CONTRACT_YAW_MAX_DEG:-1.0}" \
  --flow_contract_step_bank "${contract_steps[@]}" \
  "${tau_kappa_args[@]}" \
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
