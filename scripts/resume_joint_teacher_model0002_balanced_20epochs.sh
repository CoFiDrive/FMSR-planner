#!/usr/bin/env bash
# Continue the completed three-epoch balanced run for twenty more epochs.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-$ROOT/outputs/local_training/fmsr_model0002_joint_balanced_lr1e5_seed0_20261004_225835}"
export TFPP_CHECKPOINT="$SOURCE_RUN/model_0002.pth"
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-$ROOT/outputs/local_training_inputs/joint_teacher0030_student0002_all16_balanced}"
export RECOVERY_SAMPLING_PLAN="$RECOVERY_DATA_ROOT/recovery_sampling_plan.json"
export RECOVERY_FRACTION="${RECOVERY_FRACTION:-0.10}"
export RECOVERY_MAX_REPEATS="${RECOVERY_MAX_REPEATS:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export BATCH_SIZE="${BATCH_SIZE:-32}" SEED="${SEED:-0}" GPU_ID="${GPU_ID:-0}"
export PDM_ASSET_DIR="${PDM_ASSET_DIR:-/media/vcdc/T71/pdm_lite_assets}"
export TRAIN_LOG_ROOT="${TRAIN_LOG_ROOT:-$ROOT/outputs/local_training}"
export RUN_ID="${RUN_ID:-fmsr_model0002_joint_balanced_resume3to22_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"

[[ ! -e "$TRAIN_LOG_ROOT/$RUN_ID" ]] || {
    echo "Choose a new RUN_ID; output already exists: $TRAIN_LOG_ROOT/$RUN_ID" >&2
    exit 2
}
for filename in model_0002.pth optimizer_0002.pth scheduler_0002.pth scaler_0002.pth config.json args.txt; do
    [[ -s "$SOURCE_RUN/$filename" ]] || { echo "Missing continuation file: $SOURCE_RUN/$filename" >&2; exit 2; }
done
for filename in recovery_frames.txt recovery_sampling_plan.json training_data_report.json; do
    [[ -s "$RECOVERY_DATA_ROOT/$filename" ]] || { echo "Missing balanced data file: $RECOVERY_DATA_ROOT/$filename" >&2; exit 2; }
done
"$PYTHON_BIN" - "$SOURCE_RUN" "$RECOVERY_DATA_ROOT" <<'PY'
import json
import os
from pathlib import Path
import sys

source, data = map(lambda p: Path(p).resolve(), sys.argv[1:])
args = json.loads((source / 'args.txt').read_text())
expected = {
    'batch_size': int(os.environ['BATCH_SIZE']),
    'seed': int(os.environ['SEED']),
    'fmsr_recovery_fraction': float(os.environ['RECOVERY_FRACTION']),
    'fmsr_recovery_max_repeats': int(os.environ['RECOVERY_MAX_REPEATS']),
}
for name, value in expected.items():
    if args.get(name) != value:
        raise ValueError(f'Continuation must retain {name}={args.get(name)!r}; requested {value!r}')
if args['epochs'] != 3 or args['fmsr_trajectory_head'] != 'tie_cr_fd':
    raise ValueError('Expected the completed three-epoch tie_cr_fd run')
if Path(args['fmsr_recovery_root']).resolve() != data:
    raise ValueError('Continuation must use the source run recovery dataset')
if Path(args['fmsr_recovery_sampling_plan']).resolve() != data / 'recovery_sampling_plan.json':
    raise ValueError('Continuation must use the source run balanced sampling plan')
report = json.loads((data / 'training_data_report.json').read_text())
print(f"Source checkpoint: {source / 'model_0002.pth'}")
print(f"Balanced recovery data: {report['selected_frames']} frames")
print('Restore model and optimizer; restart cosine schedule for 20 epochs.')
print('Resume epochs 3 through 22: 20 additional epochs, 23 total.')
PY

# This entry point sets CONTINUE_EPOCH=1, RESET_SCHEDULER=1, EPOCHS=23,
# COSINE_T0=20. Avoid the fresh post-training wrapper, which sets continuation=0.
exec bash "$ROOT/scripts/resume_recovery_pdm_lite_tie_cr_fd_20epochs.sh"
