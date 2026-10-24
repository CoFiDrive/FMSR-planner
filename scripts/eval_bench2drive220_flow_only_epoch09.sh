#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
TRAIN_DIR="${1:-$ROOT/outputs/local_training/flow_only_10epochs_seed0_20261008_143915}"
TRAIN_DIR="$(cd -- "$TRAIN_DIR" && pwd)"
export CHECKPOINT="$TRAIN_DIR/model_0009.pth"
export ROUTES="$ROOT/3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
export EVALUATION_OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-$ROOT/outputs/local_evaluation/bench2drive220_$(basename -- "$TRAIN_DIR")_epoch09}"
mkdir -p "$EVALUATION_OUTPUT_DIR"
EVALUATION_OUTPUT_DIR="$(cd -- "$EVALUATION_OUTPUT_DIR" && pwd)"
export EVALUATION_OUTPUT_DIR

# Isolate this evaluation from previous teacher/data-collection experiments.
unset SHADOW_TEACHER_CHECKPOINT SHADOW_COLLECTION_ROOT FMSR_DIAGNOSTIC_ROOT
export EVALUATION_AGENT="$ROOT/3rd_party/carla_garage/team_code/sensor_agent.py"
export FMSR_DIAGNOSTICS=0
export CARLA_RELOAD_EACH_ROUTE=0
export GPU_ID="${GPU_ID:-0}"
export CARLA_PORT="${CARLA_PORT:-8860}"
export TM_PORT="${TM_PORT:-8020}"
export PYTHON_BIN

"$PYTHON_BIN" - "$TRAIN_DIR" "$ROUTES" <<'PY'
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
training, routes = map(Path, sys.argv[1:])
assert (training / 'model_0009.pth').is_file(), 'epoch09 checkpoint is not available yet'
exitcode = training / 'train.exitcode'
if exitcode.exists():
    assert exitcode.read_text().strip() == '0', 'Training did not exit successfully'
config = json.loads((training / 'config.json').read_text())
assert config['fmsr_trajectory_head'] == 'flow_only', 'Expected a pure FM checkpoint'
assert not config.get('flow_contract_enable', False), 'CR must be disabled'
assert len(ET.parse(routes).getroot().findall('route')) == 220
print('Pure FM epoch09 and 220 routes verified')
PY

exec 8>"$EVALUATION_OUTPUT_DIR/.watchdog.lock"
if ! flock -n 8; then
  echo "Automatic evaluation already running: $EVALUATION_OUTPUT_DIR"
  exit 0
fi
# Check model staging/assets/weather before putting the job in the background.
PREFLIGHT_ONLY=1 bash "$ROOT/scripts/eval_bench2drive_pdm_lite_fmsr.sh"
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ -e "$EVALUATION_OUTPUT_DIR/watchdog.stop" ]]; then
  echo "Remove $EVALUATION_OUTPUT_DIR/watchdog.stop to re-enable automatic continuation." >&2
  exit 1
fi
flock -u 8
nohup "$PYTHON_BIN" "$ROOT/scripts/pdm_lite/watch_evaluation.py" \
  --output-dir "$EVALUATION_OUTPUT_DIR" --routes "$ROUTES" \
  --retry-delay 30 --max-no-progress-restarts "${MAX_NO_PROGRESS_RESTARTS:-10}" \
  --missing-carla-grace-seconds 90 --stale-log-seconds 600 \
  >> "$EVALUATION_OUTPUT_DIR/watchdog.log" 2>&1 < /dev/null 8>&- &
echo "Watchdog started (PID $!). It resumes the same checkpoint until all 220 routes finish."
echo "Results: $EVALUATION_OUTPUT_DIR/checkpoint_endpoint.json"
echo "Log: $EVALUATION_OUTPUT_DIR/watchdog.log"
