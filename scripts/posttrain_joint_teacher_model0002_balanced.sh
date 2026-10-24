#!/usr/bin/env bash
# Merge all 16 existing rollouts and balance scenario/long-stop sampling.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export INCLUDE_DEGRADED11=1
export RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-$ROOT/outputs/local_training_inputs/joint_teacher0030_student0002_all16_balanced}"
export RECOVERY_SAMPLING_PLAN="$RECOVERY_DATA_ROOT/recovery_sampling_plan.json"
export RUN_ID="${RUN_ID:-fmsr_model0002_joint_balanced_lr1e5_seed0_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
exec bash "$ROOT/scripts/posttrain_joint_teacher_model0002.sh"
