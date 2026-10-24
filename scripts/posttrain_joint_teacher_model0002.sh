#!/usr/bin/env bash
# Full-data joint learned-teacher post-training, with a fresh optimizer.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export PYTHON_BIN
export TFPP_CHECKPOINT="${TFPP_CHECKPOINT:-$ROOT/outputs/local_training/fmsr_pdm_lite_tie_cr_fd_epoch11_recovery_seed0_20261003_181046/model_0002.pth}"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-$ROOT/outputs/local_training_inputs/joint_teacher0030_student0002_all}"
export RUN_ID="${RUN_ID:-fmsr_model0002_joint_all_lr1e5_seed0_$(date +%Y%m%d_%H%M%S)}"
export TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-$ROOT/outputs/local_training}"
export FMSR_TRAJECTORY_HEAD=tie_cr_fd
export EPOCHS="${EPOCHS:-3}" BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}" RECOVERY_FRACTION="${RECOVERY_FRACTION:-0.10}"
export RECOVERY_MAX_REPEATS="${RECOVERY_MAX_REPEATS:-2}" FLOW_RECOVERY_WEIGHT=0
export GPU_ID="${GPU_ID:-0}" SEED="${SEED:-0}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T71/pdm_lite_assets}"
if [[ -d "$TRAIN_LOG_ROOT/$RUN_ID" ]]; then
    echo "Choose a new RUN_ID; output already exists: $TRAIN_LOG_ROOT/$RUN_ID" >&2
    exit 2
fi
evaluations=("$ROOT/outputs/local_evaluation/joint_teacher0030_student0002_route2204"
             "$ROOT/outputs/local_evaluation/joint_teacher0030_student0002_priority4")
if [[ "${INCLUDE_DEGRADED11:-0}" == "1" ]]; then
    evaluations+=("$ROOT/outputs/local_evaluation/joint_teacher0030_student0002_degraded11")
fi
if [[ -n "${EXTRA_JOINT_EVALUATIONS:-}" ]]; then
    while IFS= read -r evaluation; do
        [[ -n "$evaluation" ]] && evaluations+=("$evaluation")
    done <<< "$EXTRA_JOINT_EVALUATIONS"
fi
"$PYTHON_BIN" scripts/pdm_lite/stage_joint_posttraining.py \
    --evaluation "${evaluations[@]}" \
    --output "$RECOVERY_DATA_ROOT"
PDM_DATA_ROOT="$("$PYTHON_BIN" scripts/pdm_lite/mount_carla_garage_data.py \
    --data-root "${PDM_DATA_ROOT:-/media/vcdc/T71/carla_garage_data}" \
    --view-root "$ROOT/outputs/local_dataset_mounts/carla_garage_data")"
export PDM_DATA_ROOT
printf 'Training output: %s/%s\n' "$TRAIN_LOG_ROOT" "$RUN_ID"
exec bash scripts/posttrain_recovery_pdm_lite.sh
