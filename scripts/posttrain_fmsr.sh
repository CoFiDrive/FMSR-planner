#!/bin/bash

set -eo pipefail

# Resolve the checkout from this script, even when invoked from another directory.
FMSR_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FMSR_PROJECT_ROOT
cd "${FMSR_PROJECT_ROOT}"
# module unload cuda
# module load cuda/12.1.1
source "${FMSR_PROJECT_ROOT}/scripts/main.sh"
# export TMPDIR=path_to_your_tmp_folder/fmsr_tmp # change tmp dir if needed
export CUDA_VISIBLE_DEVICES=0

# export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=""

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 # Shuts off numpy multithreading, to avoid threads spawning other threads.
export NCCL_P2P_DISABLE=1 # https://github.com/huggingface/accelerate/issues/314
export NCCL_P2P_LEVEL=NVL # https://github.com/huggingface/accelerate/issues/314
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((10000 + RANDOM % 50000))

# Override with TRAJECTORY_GENERATOR=ddbm for the original FMSR baseline.
export TRAJECTORY_GENERATOR=${TRAJECTORY_GENERATOR:-residual_flow}
PYTHON_BIN=${PYTHON_BIN:-python}
# Keep each run separate from previous experiments, including mislabeled DDBM runs.
TRAIN_LOGDIR=${TRAIN_LOGDIR:-outputs/local_training/posttrain_fmsr_k60_4090_32_${TRAJECTORY_GENERATOR}_$(date +%Y%m%d_%H%M%S)_$$}
export FMSR_TRAINING_CONFIG="logdir=${TRAIN_LOGDIR} \
load_file=${LOAD_FILE:-data/fmsr_ckpt/tfv6/model_0030_2.pth} \
carla_root=data/carla_leaderboard2 \
use_planning_decoder=true \
freeze_backbone=true \
continue_failed_training=${CONTINUE_FAILED_TRAINING:-false} \
diffusion_speed=false \
trajectory_generator=${TRAJECTORY_GENERATOR} \
step_num=20 \
flow_train_t_max=0.95 \
flow_min_remain=0.05 \
flow_inference_steps=${FLOW_INFERENCE_STEPS:-20} \
flow_contract_enable=${FLOW_CONTRACT_ENABLE:-false} \
flow_contract_fraction=0.125 \
flow_contract_weight=${FLOW_CONTRACT_WEIGHT:-0.01} \
flow_contract_rho=0.98 \
flow_contract_lat_min_m=0.02 \
flow_contract_lat_max_m=0.08 \
flow_contract_lon_max_m=0.03 \
flow_contract_step_bank=[4,8,20] \
visualize_training=false \
force_rebuild_data_cache=false \
plan_anchor_path=anchor_utils/anchor_data/fmsr_cp_kmeans_60_10.npy \
batch_size=${BATCH_SIZE:-32} \
lr=${LEARNING_RATE:-3e-4}"

# Override the total epochs for a fresh fine-tuning stage when requested.
if [[ -n "${EPOCHS:-}" ]]; then
    export FMSR_TRAINING_CONFIG="${FMSR_TRAINING_CONFIG} epochs=${EPOCHS}"
fi

# Check actual imports and checkpoint-merged configuration before allocating GPUs.
"${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

import fmsr.training.config_training as config_module
import fmsr.training.training_utils_fmsr as training_module
import fmsr.tfv6.diffusion_modules.model_diffusion_head_ddbm as head_module
import fmsr.tfv6.planning_decoder_fmsr as planning_module

root = Path(os.environ["FMSR_PROJECT_ROOT"]).resolve()
print(f"Training Python: {sys.executable}")
for module in (config_module, training_module, head_module, planning_module):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(root / "fmsr"):
        raise RuntimeError(f"Imported {module.__name__} from the wrong checkout: {path}")
    print(f"{module.__name__}: {path}")

config = training_module.initialize_config()
requested = os.environ["TRAJECTORY_GENERATOR"]
if requested not in ("residual_flow", "ddbm"):
    raise RuntimeError(f"Unknown trajectory generator: {requested}")
if getattr(config, "trajectory_generator", None) != requested:
    raise RuntimeError("The effective trajectory_generator differs from the requested value")
if not (config.use_planning_decoder and config.predict_spatial_path and config.predict_ddbm) or config.use_tfv5_planning_decoder:
    raise RuntimeError("FMSR trajectory training is not enabled in the effective config")
if requested == "residual_flow" and not hasattr(head_module.TrajectoryHead, "forward_train_flow"):
    raise RuntimeError("The imported trajectory head has no Residual Flow training branch")
print(f"Verified trajectory_generator={config.trajectory_generator}")
print(f"flow_contract_enable={config.flow_contract_enable}, weight={config.flow_contract_weight}")
print(f"flow_inference_steps={config.flow_inference_steps}, step_num={config.step_num}")
print(f"flow_min_remain={config.flow_min_remain}, flow_contract_step_bank={config.flow_contract_step_bank}")
print(f"epochs={config.epochs}, batch_size={config.batch_size}, lr={config.lr}")
print(f"continue_failed_training={config.continue_failed_training}, load_file={config.load_file}")
print(f"Training output: {config.logdir}")
if config.continue_failed_training:
    checkpoint = Path(config.load_file)
    for prefix in ("model", "optimizer", "scheduler", "scaler"):
        path = checkpoint.with_name(checkpoint.name.replace("model_", f"{prefix}_", 1))
        if not path.is_file():
            raise FileNotFoundError(f"Missing resume checkpoint: {path}")
    start_epoch = int("".join(filter(str.isdigit, checkpoint.stem))) + 1
    print(f"Resuming from {checkpoint}, starting at epoch {start_epoch} (total epochs: {config.epochs})")
PY

# Useful for checking the launch environment without starting a training job.
if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    --max_restarts=0 \
    --module fmsr.training.train_fmsr
