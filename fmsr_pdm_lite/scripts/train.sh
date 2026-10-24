#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/vendor/garage${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
PDM_DATA_ROOT="${PDM_DATA_ROOT:-$ROOT/../outputs/local_dataset_mounts/carla_garage_data}"
PDM_ASSET_DIR="${PDM_ASSET_DIR:-$ROOT/assets/paper_k60_seed0}"
TFPP_CHECKPOINT="${TFPP_CHECKPOINT:-/media/vcdc/T71/tfpp_pretrained/pretrained_models/all_towns/model_0030_2.pth}"
RUN_ID="${RUN_ID:-paper_vp_ddbm_$(date +%Y%m%d_%H%M%S)}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-$ROOT/runs/$RUN_ID}"
extra=()
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then extra+=(--preflight-only); fi
if [[ -n "${RESUME_STATE:-}" ]]; then extra+=(--resume "$RESUME_STATE"); fi
exec "$PYTHON_BIN" -m fmsrpdm.train --data-root "$PDM_DATA_ROOT" --assets-dir "$PDM_ASSET_DIR" \
  --tfpp-checkpoint "$TFPP_CHECKPOINT" --output-dir "$TRAIN_OUTPUT_DIR" \
  --epochs "${EPOCHS:-10}" --lr "${LEARNING_RATE:-3e-4}" --batch-size "${BATCH_SIZE:-32}" \
  --workers "${WORKERS:-4}" --seed "${SEED:-0}" --inference-steps "${INFERENCE_STEPS:-20}" \
  --generation-validation-samples "${GENERATION_VALIDATION_SAMPLES:-128}" \
  --rollout-denoising-weight "${ROLLOUT_DENOISING_WEIGHT:-0}" \
  --rollout-denoising-scenes "${ROLLOUT_DENOISING_SCENES:-4}" \
  --rollout-geometry-weight "${ROLLOUT_GEOMETRY_WEIGHT:-0}" \
  --init-checkpoint "${INIT_CHECKPOINT:-}" \
  --prediction-base "${PREDICTION_BASE:-noisy_residual}" --training-scope "${TRAINING_SCOPE:-joint}" \
  --path-geometry-weight "${PATH_GEOMETRY_WEIGHT:-0}" \
  --classification-loss "${CLASSIFICATION_LOSS:-cross_entropy}" --device "${DEVICE:-cuda}" "${extra[@]}" "$@"
