#!/usr/bin/env bash
set -uo pipefail

DOWNLOAD_DIR=${1:?Usage: download_data.sh DOWNLOAD_DIR EXTRACT_DIR [--force-low-space]}
EXTRACT_DIR=${2:?Usage: download_data.sh DOWNLOAD_DIR EXTRACT_DIR [--force-low-space]}
FORCE=${3:-}

SCENARIOS=(
  Accident AccidentTwoWays BlockedIntersection ConstructionObstacle
  ConstructionObstacleTwoWays ControlLoss CrossingBicycleFlow
  DynamicObjectCrossing EnterActorFlow EnterActorFlowV2 HardBreakRoute
  HazardAtSideLane HazardAtSideLaneTwoWays HighwayCutIn HighwayExit
  InterurbanActorFlow InterurbanAdvancedActorFlow InvadingTurn
  MergerIntoSlowTraffic MergerIntoSlowTrafficV2
  NonSignalizedJunctionLeftTurn NonSignalizedJunctionRightTurn noScenarios
  OppositeVehicleRunningRedLight OppositeVehicleTakingPriority
  ParkedObstacle ParkedObstacleTwoWays ParkingCrossingPedestrian
  ParkingCutIn ParkingExit PedestrianCrossing PriorityAtJunction
  SignalizedJunctionLeftTurn SignalizedJunctionRightTurn StaticCutIn
  VehicleOpensDoorTwoWays VehicleTurningRoute
  VehicleTurningRoutePedestrian YieldToEmergencyVehicle
)
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/dataset"

mkdir -p "${DOWNLOAD_DIR}" "${EXTRACT_DIR}"

# Prevent two jobs from writing the same archives at the same time.
exec 9>"${DOWNLOAD_DIR}/.download.lock"
if ! flock -n 9; then
  echo "Another CARLA Garage download is already using ${DOWNLOAD_DIR}." >&2
  exit 3
fi

# If interrupted archives exist, recover only those archives. The other
# scenarios were already downloaded, extracted, and removed by the old job.
shopt -s nullglob
partial_archives=("${DOWNLOAD_DIR}"/*.tar)
if (( ${#partial_archives[@]} > 0 )); then
  SCENARIOS=()
  for archive in "${partial_archives[@]}"; do
    SCENARIOS+=("$(basename "${archive}" .tar)")
  done
  # Finish archives with the smallest conservative working-space requirement
  # first. Completing and deleting those archives may free enough room for
  # larger scenarios later in the recovery.
  mapfile -t SCENARIOS < <(
    for scenario in "${SCENARIOS[@]}"; do
      archive="${DOWNLOAD_DIR}/${scenario}.tar"
      url="${BASE_URL}/${scenario}.tar"
      current_size=$(stat -c '%s' "${archive}" 2>/dev/null || printf '0')
      remote_size=$(curl --noproxy '*' -fsSI --max-time 30 "${url}" 2>/dev/null |
        awk 'tolower($1) == "content-length:" {gsub("\\r", "", $2); print $2; exit}')
      if [[ "${remote_size}" =~ ^[0-9]+$ ]]; then
        remaining_size=$((remote_size > current_size ? remote_size - current_size : 0))
        printf '%020d\t%s\n' "$((remaining_size + remote_size))" "${scenario}"
      else
        printf '%020d\t%s\n' 9999999999999999999 "${scenario}"
      fi
    done | sort -n -k1,1 -u | cut -f2-
  )
  echo "Recovery mode: resuming ${#SCENARIOS[@]} existing archives sequentially."
else
  available_kib=$(df -Pk "${EXTRACT_DIR}" | awk 'NR==2 {print $4}')
  required_kib=$((380 * 1024 * 1024))
  if (( available_kib < required_kib )) && [[ "${FORCE}" != "--force-low-space" ]]; then
    available_gib=$((available_kib / 1024 / 1024))
    echo "Insufficient free space for a fresh download: ${available_gib} GiB available; require at least 380 GiB." >&2
    echo "Use a larger data disk, or pass --force-low-space only if extracted data already exists." >&2
    exit 2
  fi
  echo "Fresh mode: downloading all ${#SCENARIOS[@]} archives sequentially."
fi

failed=()
for scenario in "${SCENARIOS[@]}"; do
  archive="${DOWNLOAD_DIR}/${scenario}.tar"
  url="${BASE_URL}/${scenario}.tar"

  # Conservatively reserve enough room for the rest of the download and a
  # full extraction before the archive is removed. Existing extracted files
  # will usually lower the real peak, but the conservative check prevents an
  # interrupted recovery from filling the filesystem.
  remote_size=$(curl --noproxy '*' -fsSI --max-time 30 "${url}" 2>/dev/null |
    awk 'tolower($1) == "content-length:" {gsub("\\r", "", $2); print $2; exit}')
  if [[ "${remote_size}" =~ ^[0-9]+$ ]]; then
    current_size=$(stat -c '%s' "${archive}" 2>/dev/null || printf '0')
    available_bytes=$(df -PB1 "${EXTRACT_DIR}" | awk 'NR == 2 {print $4}')
    safety_bytes=$((10 * 1024 * 1024 * 1024))
    remaining_bytes=$((remote_size > current_size ? remote_size - current_size : 0))
    required_bytes=$((remaining_bytes + remote_size + safety_bytes))
    if (( available_bytes < required_bytes )); then
      printf 'Insufficient safe working space for %s: %.2f GiB available, %.2f GiB required.\n' \
        "${scenario}" \
        "$(awk -v n="${available_bytes}" 'BEGIN {print n/1073741824}')" \
        "$(awk -v n="${required_bytes}" 'BEGIN {print n/1073741824}')" >&2
      echo "All partial archives are retained; free space and run this command again." >&2
      exit 4
    fi
  else
    echo "Warning: could not read the remote size for ${scenario}; continuing without the preflight space check." >&2
  fi

  echo
  echo "[$(date '+%F %T')] Downloading/resuming ${scenario}"
  # The local proxy has intermittently accepted TCP connections and then
  # failed the TLS handshake. S3 is reachable directly, so bypass the proxy
  # by default. An outer loop also retries TLS errors, which wget otherwise
  # treats as fatal even when --tries=0 is set.
  wget_args=(
    --continue
    --retry-connrefused
    --waitretry=10
    --timeout=60
    --tries=0
  )
  if [[ "${CARLA_GARAGE_USE_PROXY:-0}" != "1" ]]; then
    wget_args+=(--no-proxy)
  fi

  while ! (
    cd "${DOWNLOAD_DIR}" &&
    wget "${wget_args[@]}" "${url}"
  ); do
    echo "[$(date '+%F %T')] Connection failed; retaining the partial archive and retrying ${scenario} in 30 seconds." >&2
    sleep 30
  done

  echo "[$(date '+%F %T')] Verifying ${archive}"
  if ! tar -tf "${archive}" >/dev/null; then
    echo "Archive verification failed; keeping ${archive}." >&2
    failed+=("${scenario}:verify")
    continue
  fi

  echo "[$(date '+%F %T')] Extracting ${scenario}"
  if ! tar -xf "${archive}" -C "${EXTRACT_DIR}"; then
    echo "Extraction failed; keeping ${archive}." >&2
    failed+=("${scenario}:extract")
    continue
  fi

  rm -f "${archive}"
  echo "[$(date '+%F %T')] Completed ${scenario}"
done

if (( ${#failed[@]} > 0 )); then
  printf 'Unfinished items:\n' >&2
  printf '  %s\n' "${failed[@]}" >&2
  exit 1
fi

echo
echo "All selected CARLA Garage archives completed successfully."
