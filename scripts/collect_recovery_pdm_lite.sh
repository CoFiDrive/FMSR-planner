#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GARAGE="${ROOT}/3rd_party/carla_garage"
CARLA_ROOT="${ROOT}/3rd_party/CARLA_0915"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
STUDENT_CHECKPOINT="${STUDENT_CHECKPOINT:-${ROOT}/outputs/local_training/fmsr_pdm_lite_flow_fixed_bs64_20260929_092208/model_0009.pth}"
RECOVERY_DATA_ROOT="${RECOVERY_DATA_ROOT:-/media/vcdc/T7/pdm_lite_recovery_data}"
TEST_ROUTES="${ROOT}/3rd_party/Bench2Drive/leaderboard/data/bench2drive220.xml"
TRAINING_XML_ROOT="${ROOT}/3rd_party/carla_garage/data"
mkdir -p "${RECOVERY_DATA_ROOT}/Recovery_Train" "${RECOVERY_DATA_ROOT}/run"
RECOVERY_DATA_ROOT="$(cd "${RECOVERY_DATA_ROOT}" && pwd)"
STUDENT_CHECKPOINT="$(realpath "${STUDENT_CHECKPOINT}")"
# External recovery drives may use exFAT, which cannot store symlinks. Keep
# only the agent's config/model link on the local filesystem; rollout data and
# route results remain under RECOVERY_DATA_ROOT.
AGENT_DIR="${RECOVERY_STUDENT_AGENT_DIR:-${ROOT}/outputs/local_recovery_agents/$(basename "${RECOVERY_DATA_ROOT}")}"
mkdir -p "${AGENT_DIR}"
AGENT_MODEL="${AGENT_DIR}/$(basename "${STUDENT_CHECKPOINT}")"
existing_models=("${AGENT_DIR}"/model*.pth)
for existing_model in "${existing_models[@]}"; do
  [[ -e "${existing_model}" || -L "${existing_model}" ]] || continue
  if [[ "${existing_model}" != "${AGENT_MODEL}" || "$(realpath "${existing_model}")" != "${STUDENT_CHECKPOINT}" ]]; then
    echo "Agent directory contains a different checkpoint: ${existing_model}" >&2
    exit 2
  fi
done
ln -sfn "${STUDENT_CHECKPOINT}" "${AGENT_MODEL}"
"${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/recovery_setup.py" stage \
  --checkpoint "${STUDENT_CHECKPOINT}" --agent-dir "${AGENT_DIR}" \
  --asset-dir "${PDM_ASSET_DIR:-}"
export RECOVERY_STUDENT_AGENT_DIR="${AGENT_DIR}"

ROUTE_LIST="${ROUTE_LIST:-${RECOVERY_DATA_ROOT}/run/training_routes.txt}"
if [[ ! -f "${ROUTE_LIST}" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/recovery_data.py" choose-routes \
    --route-root "${TRAINING_XML_ROOT}" --test-routes "${TEST_ROUTES}" \
    --town "${RECOVERY_TOWN:-50x38_Town12}" \
    --scenario NonSignalizedJunctionLeftTurn BlockedIntersection Accident \
    --limit "${RECOVERY_ROUTE_COUNT:-8}" --seed "${RECOVERY_ROUTE_SEED:-0}" \
    --output "${ROUTE_LIST}"
fi

export CARLA_ROOT
export LEADERBOARD_ROOT="${GARAGE}/leaderboard_autopilot"
export SCENARIO_RUNNER_ROOT="${GARAGE}/scenario_runner_autopilot"
export PYTHONPATH="${GARAGE}/team_code:${ROOT}:${LEADERBOARD_ROOT}:${SCENARIO_RUNNER_ROOT}:${CARLA_ROOT}/PythonAPI/carla"
export LD_LIBRARY_PATH=""
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export CHALLENGE_TRACK_CODENAME=MAP
export DATAGEN=1
export REPETITION=0
export SAVE_PATH="${RECOVERY_DATA_ROOT}/Recovery_Train"
export DEBUG_CHALLENGE=0
export COMPILE=0
export DIRECT=1
export PLANNER_TYPE=only_traj
export IS_BENCH2DRIVE=0

while IFS= read -r ROUTE_XML; do
  [[ -z "${ROUTE_XML}" || "${ROUTE_XML}" == \#* ]] && continue
  "${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/recovery_data.py" check-route \
    --route "${ROUTE_XML}" --test-routes "${TEST_ROUTES}"
  TOWN="$(basename "$(dirname "$(dirname "${ROUTE_XML}")")" | sed -E 's/^.*_(Town[0-9]+)$/\1/')"
  export TOWN
  SPEED_LIMITS="${GARAGE}/team_code/speed_limits/${TOWN}_speed_limits.npy"
  if [[ ! -r "${SPEED_LIMITS}" ]]; then
    echo "Missing expert speed limits: ${SPEED_LIMITS}" >&2
    exit 2
  fi
  ROUTE_KEY="$(basename "$(dirname "${ROUTE_XML}")")_$(basename "${ROUTE_XML}" .xml)"
  RESULT="${RECOVERY_DATA_ROOT}/run/${TOWN}_${ROUTE_KEY}.json"
  if [[ -f "${RESULT}" ]] && "${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/recovery_setup.py" route-complete "${RESULT}"; then
    echo "Skipping completed route: ${ROUTE_XML}"
    continue
  fi
  if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
    echo "Ready: ${ROUTE_XML} (${TOWN})"
    continue
  fi
  if [[ -f "${RESULT}" ]]; then
    echo "Retrying incomplete/failed setup route: ${ROUTE_XML}"
    RETRY_TAG="$(date +%Y%m%d_%H%M%S)_$$"
    cp -p "${RESULT}" "${RESULT}.failed.${RETRY_TAG}"
    ROUTE_LOG="${RECOVERY_DATA_ROOT}/run/${TOWN}_${ROUTE_KEY}.log"
    if [[ -f "${ROUTE_LOG}" ]]; then
      cp -p "${ROUTE_LOG}" "${ROUTE_LOG}.failed.${RETRY_TAG}"
    fi
  fi
  "${PYTHON_BIN}" "${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_local.py" \
    --routes="${ROUTE_XML}" --track=MAP \
    --checkpoint="${RESULT}" \
    --agent="${GARAGE}/team_code/recovery_agent.py" \
    --agent-config="${ROUTE_XML}" \
    --debug=0 --record= --resume=0 \
    --port="${CARLA_PORT:-8850}" --traffic-manager-port="${TM_PORT:-8002}" \
    --timeout="${EVAL_TIMEOUT:-900}" --traffic-manager-seed="${TM_SEED:-0}" \
    --repetitions=1 \
    2>&1 | tee "${RECOVERY_DATA_ROOT}/run/${TOWN}_${ROUTE_KEY}.log"
  if ! "${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/recovery_setup.py" route-complete "${RESULT}"; then
    echo "Collection stopped after an agent/simulator failure. Inspect ${RESULT} and its route log; rerun after fixing it." >&2
    exit 1
  fi
done < "${ROUTE_LIST}"

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "Collection preflight passed; CARLA was not started."
  exit 0
fi

"${PYTHON_BIN}" "${ROOT}/scripts/pdm_lite/recovery_data.py" build-manifest \
  --recovery-root "${RECOVERY_DATA_ROOT}" \
  --history-frames "${RECOVERY_HISTORY_FRAMES:-8}" \
  --output "${RECOVERY_DATA_ROOT}/recovery_frames.txt"
