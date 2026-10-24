#!/usr/bin/bash

source slurm/init.sh

export CHECKPOINT_DIR=outputs/training/001_example/010_postrain32_0/251025_182327
export FMSR_TRAINING_CONFIG="$FMSR_TRAINING_CONFIG"
export FMSR_CLOSED_LOOP_CONFIG="$FMSR_CLOSED_LOOP_CONFIG custom_weather=ClearSunrise"

evaluate_longest6
