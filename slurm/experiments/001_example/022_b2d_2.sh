#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/012_postrain32_2/251025_182334
export FMSR_TRAINING_CONFIG="$FMSR_TRAINING_CONFIG"
export FMSR_CLOSED_LOOP_CONFIG="$FMSR_CLOSED_LOOP_CONFIG"

evaluate_bench2drive
