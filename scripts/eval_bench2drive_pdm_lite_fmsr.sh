#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export FMSR_PROJECT_ROOT="$ROOT"
cd "$ROOT"
GARAGE_ROOT="${ROOT}/3rd_party/carla_garage"
export CARLA_ROOT="${ROOT}/3rd_party/CARLA_0915"
export LEADERBOARD_ROOT="${ROOT}/3rd_party/Bench2Drive/leaderboard"
export SCENARIO_RUNNER_ROOT="${ROOT}/3rd_party/Bench2Drive/scenario_runner"
# Keep CARLA Garage's config/model/loader ahead of other installations.
export PYTHONPATH="${GARAGE_ROOT}/team_code:${ROOT}:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${CARLA_ROOT}/PythonAPI/carla"
export LD_LIBRARY_PATH=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export IS_BENCH2DRIVE=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export DIRECT=1
export DEBUG_CHALLENGE=0
export COMPILE=0
export CARLA_RENDER_OFFSCREEN="${CARLA_RENDER_OFFSCREEN:-1}"
export PLANNER_TYPE=only_traj
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
export EVALUATION_AGENT="${EVALUATION_AGENT:-$GARAGE_ROOT/team_code/sensor_agent.py}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_residual_flow_k60_20260926_142355/model_0009.pth}"
ROUTES="${ROUTES:-${ROOT}/3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml}"
OUTPUT_DIR="${EVALUATION_OUTPUT_DIR:-${ROOT}/outputs/local_evaluation/bench2drive220_pdm_lite_seed2_epoch09_continuous_speed}"
mkdir -p "$OUTPUT_DIR/debug_checkpoint"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
exec 9>"$OUTPUT_DIR/.evaluation.lock"
flock -n 9 || { echo "Evaluation already running for $OUTPUT_DIR" >&2; exit 75; }
AGENT_DIR="$OUTPUT_DIR/agent"
export SAVE_PATH="$OUTPUT_DIR"
export FMSR_DIAGNOSTIC_ROOT="${FMSR_DIAGNOSTIC_ROOT:-$OUTPUT_DIR}"

# The Garage agent loads every model*.pth in its config directory. Stage exactly
# the requested checkpoint, retaining the matching training config and PDM assets.
"$PYTHON_BIN" - "$CHECKPOINT" "$AGENT_DIR" "$ROUTES" <<'PY'
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
from config import GlobalConfig
from fmsr_interface import restore_trained_interface
from sensor_agent import SensorAgent
from leaderboard.leaderboard_evaluator_v2 import get_weather_id
from leaderboard.utils.route_parser import RouteParser

jsonpickle_numpy.register_handlers()
checkpoint, destination, routes = map(Path, sys.argv[1:])
checkpoint = checkpoint.resolve(strict=True)
config_path = checkpoint.parent / 'config.json'
config_text = config_path.read_text()
config = jsonpickle.decode(config_text)
assert isinstance(config, GlobalConfig), 'Unexpected checkpoint config class'
assert config.use_fmsr and config.trajectory_generator == 'residual_flow'
assert config.flow_inference_steps == 20, 'Expected the trained 20-step flow configuration'
restore_trained_interface(config, config)
# A resumed evaluation must not mix old speed decoding with the fixed policy.
source_path = destination / 'source.json'
result_path = destination.parent / 'checkpoint_endpoint.json'
if result_path.exists():
    previous = json.loads(source_path.read_text()) if source_path.exists() else {}
    if previous.get('agent') != str(Path(os.environ['EVALUATION_AGENT']).resolve()):
        raise RuntimeError('Existing results use a different agent; choose a new output directory')
    if previous.get('shadow_teacher_checkpoint') != os.environ.get('SHADOW_TEACHER_CHECKPOINT'):
        raise RuntimeError('Reference teacher changed; choose a new output directory')
    if previous.get('shadow_collection_root') != os.environ.get('SHADOW_COLLECTION_ROOT'):
        raise RuntimeError('Collection root changed; choose a new output directory')
    if previous.get('speed_interface') != 'continuous_v1':
        raise RuntimeError('Existing results use the old speed interface; choose a new EVALUATION_OUTPUT_DIR')
    if previous.get('fmsr_speed_source', 'planner') != config.fmsr_speed_source:
        raise RuntimeError('Existing results use a different speed source; choose a new EVALUATION_OUTPUT_DIR')
    if (config.fmsr_speed_source == 'tfpp_curvature_v1' and
            previous.get('fmsr_curve_max_lateral_accel') != config.fmsr_curve_max_lateral_accel):
        raise RuntimeError('Existing results use a different curvature cap; choose a new EVALUATION_OUTPUT_DIR')
    if previous.get('routes') != str(routes.resolve()):
        raise RuntimeError('Existing results use a different route file; choose a new EVALUATION_OUTPUT_DIR')
    if bool(previous.get('diagnostics', False)) != (os.environ.get('FMSR_DIAGNOSTICS', '0') == '1'):
        raise RuntimeError('Existing results use different diagnostic settings; choose a new EVALUATION_OUTPUT_DIR')
    if previous.get('diagnostics'):
        if (previous.get('diagnostic_root') != str(Path(os.environ['FMSR_DIAGNOSTIC_ROOT']).resolve())
                or previous.get('diagnostic_sensor_every') != int(os.environ.get('FMSR_DIAGNOSTIC_SENSOR_EVERY', '5'))):
            raise RuntimeError('Resumed diagnostics require the same root and sensor frequency')
for attr in ('fmsr_anchor_path', 'fmsr_stats_path'):
    if not Path(getattr(config, attr)).is_file():
        raise FileNotFoundError(getattr(config, attr))
route_elements = ET.parse(routes).getroot().findall('route')
route_count = len(route_elements)
assert route_count > 0, 'Route file is empty'
for route in route_elements:
    for _, weather in RouteParser.parse_weather(route):
        if get_weather_id(weather) is None:
            raise ValueError(f"Unrecognized weather for route {route.get('id')}")
destination.mkdir(exist_ok=True)
link = destination / checkpoint.name
existing = list(destination.glob('model*.pth'))
if existing and (existing != [link] or link.resolve() != checkpoint):
    raise RuntimeError('Output directory contains a different model; choose a new EVALUATION_OUTPUT_DIR')
staged_config = destination / 'config.json'
if staged_config.exists() and staged_config.read_text() != config_text:
    raise RuntimeError('Output config differs; choose a new EVALUATION_OUTPUT_DIR')
if not link.exists():
    link.symlink_to(checkpoint)
staged_config.write_text(config_text)
(destination / 'source.json').write_text(json.dumps({
    'checkpoint': str(checkpoint), 'config': str(config_path),
    'agent': str(Path(os.environ['EVALUATION_AGENT']).resolve()),
    'shadow_teacher_checkpoint': os.environ.get('SHADOW_TEACHER_CHECKPOINT'),
    'shadow_collection_root': os.environ.get('SHADOW_COLLECTION_ROOT'),
    'routes': str(routes.resolve()), 'route_count': route_count, 'flow_inference_steps': 20,
    'speed_interface': 'continuous_v1',
    'fmsr_speed_source': config.fmsr_speed_source,
    'fmsr_curve_max_lateral_accel': config.fmsr_curve_max_lateral_accel,
    'diagnostics': os.environ.get('FMSR_DIAGNOSTICS', '0') == '1',
    'diagnostic_sensor_every': int(os.environ.get('FMSR_DIAGNOSTIC_SENSOR_EVERY', '5')),
    'diagnostic_root': str(Path(os.environ['FMSR_DIAGNOSTIC_ROOT']).resolve()),
    'reload_each_route': os.environ.get('CARLA_RELOAD_EACH_ROUTE', '0') == '1',
    'fmsr_use_target_point': config.fmsr_use_target_point,
    'fmsr_bev_layout': config.fmsr_bev_layout,
}, indent=2) + '\n')
print('Checkpoint:', checkpoint)
print('Agent:', str(Path(os.environ['EVALUATION_AGENT']).resolve()))
print(f'Routes: {route_count}; residual flow inference: 20 steps')
print('Continuous speed; trained target point:', config.fmsr_use_target_point,
      '; trained BEV layout:', config.fmsr_bev_layout)
print(f'Weather lookup passed for all {route_count} routes')
if os.environ.get('FMSR_DIAGNOSTICS', '0') == '1':
    import tempfile
    import numpy as np
    from types import SimpleNamespace
    from diagnostic_logger import DiagnosticLogger, verify_directory, diagnostic_directory
    # Round-trip the real file writer before launching CARLA; no GPU or world required.
    diagnostic_root = Path(os.environ['FMSR_DIAGNOSTIC_ROOT'])
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.diagnostic_preflight_', dir=diagnostic_root) as tmp:
        directory = diagnostic_directory(tmp, str(destination) + '+preflight_route')
        logger = DiagnosticLogger(directory, {}, sensor_every=1)
        vector = SimpleNamespace(x=0., y=0., z=0.)
        rotation = SimpleNamespace(roll=0., pitch=0., yaw=0.)
        ego = SimpleNamespace(id=1, type_id='vehicle.test',
            get_transform=lambda: SimpleNamespace(location=vector, rotation=rotation),
            get_velocity=lambda: vector,
            bounding_box=SimpleNamespace(extent=vector, location=vector))
        world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(frame=1,
            timestamp=SimpleNamespace(elapsed_seconds=0.05)), get_actors=lambda: [ego])
        logger.bind(world, ego)
        control = SimpleNamespace(steer=0., throttle=0., brake=1.)
        logger.record(0, 0.05, {'rgb_front': (1, np.zeros((4, 4, 4), dtype=np.uint8)),
            'lidar': (1, np.zeros((2, 4), dtype=np.float32))},
            {'predicted_trajectory_ego_m': [[0., 0.], [1., 0.]]}, control, control)
        logger.close()
        verify_directory(directory)
    print('Diagnostic preflight passed: JSONL, RGB and LiDAR saved and decoded')
PY

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Evaluation preparation passed: $AGENT_DIR"
  exit 0
fi

"$PYTHON_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator_v2.py" \
  --routes="$ROUTES" \
  --track=SENSORS \
  --checkpoint="$OUTPUT_DIR/checkpoint_endpoint.json" \
  --agent="$EVALUATION_AGENT" \
  --agent-config="$AGENT_DIR" \
  --debug=0 \
  --record=None \
  --resume=True \
  --port="${CARLA_PORT:-8850}" \
  --traffic-manager-port="${TM_PORT:-8002}" \
  --timeout="${EVAL_TIMEOUT:-120}" \
  --debug-checkpoint="$OUTPUT_DIR/debug_checkpoint/debug_checkpoint_endpoint.txt" \
  --traffic-manager-seed="${TM_SEED:-0}" \
  --repetitions=1 \
  2>&1 | tee -a "$OUTPUT_DIR/evaluation.log"
