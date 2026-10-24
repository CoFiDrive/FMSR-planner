#!/usr/bin/env bash
set -Eeuo pipefail

# Compact completed CARLA Garage scenario directories on an exFAT disk.
# Run as root (for the kernel SquashFS mounts), normally through pkexec.

DATA_ROOT=${1:-/media/vcdc/T7/carla_garage_data}
IMAGE_ROOT=${2:-/media/vcdc/T7/carla_garage_squashfs}
DOWNLOAD_ROOT=${3:-/media/vcdc/T7/carla_garage_download}
BACKUP_ROOT=${4:-/media/vcdc/T7/carla_garage_raw_backup}
PROCESSORS=${SQUASHFS_PROCESSORS:-8}
RESERVE_BYTES=$((1024 * 1024 * 1024))

if (( EUID != 0 )); then
  echo "This script needs root only to mount SquashFS images." >&2
  echo "Run it with: pkexec $0" >&2
  exit 2
fi

for command in mksquashfs unsquashfs findmnt mount umount python3 flock; do
  command -v "${command}" >/dev/null || {
    echo "Missing required command: ${command}" >&2
    exit 2
  }
done

mkdir -p "${DATA_ROOT}" "${IMAGE_ROOT}" "${BACKUP_ROOT}"

# Do not compact a scenario while the downloader may be extracting into it.
exec 9>"${DOWNLOAD_ROOT}/.download.lock"
if ! flock -n 9; then
  echo "The CARLA Garage downloader is active; stop it before compacting." >&2
  exit 3
fi

LOG_FILE="${IMAGE_ROOT}/compact_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[$(date '+%F %T')] Starting CARLA Garage compaction"
echo "Data root: ${DATA_ROOT}"
echo "Image root: ${IMAGE_ROOT}"
echo "Log: ${LOG_FILE}"

declare -A incomplete=()
shopt -s nullglob
for archive in "${DOWNLOAD_ROOT}"/*.tar; do
  incomplete["$(basename "${archive}" .tar)"]=1
done

if (( ${#incomplete[@]} > 0 )); then
  echo "Skipping scenarios with unfinished archives:"
  printf '  %s\n' "${!incomplete[@]}" | sort
fi

is_sqfs_mounted() {
  local path=$1
  findmnt -rn -M "${path}" -t squashfs >/dev/null 2>&1
}

make_raw_manifest() {
  local source=$1 output=$2
  find "${source}" -type f -printf '%P\t%s\n' | LC_ALL=C sort > "${output}"
}

make_image_manifest() {
  local image=$1 output=$2
  unsquashfs -lln "${image}" |
    awk '$1 ~ /^-/ {
      size=$3
      $1=$2=$3=$4=$5=""
      sub(/^ +/, "", $0)
      sub(/^squashfs-root\//, "", $0)
      print $0 "\t" size
    }' | LC_ALL=C sort > "${output}"
}

remove_verified_backup() {
  local backup=$1 expected_parent=$2
  python3 - "${backup}" "${expected_parent}" <<'PY'
from pathlib import Path
import shutil
import sys

target = Path(sys.argv[1])
parent = Path(sys.argv[2]).resolve()
if target.parent.resolve() != parent or not target.name or target.name in {'.', '..'}:
    raise SystemExit(f"Refusing unsafe removal: {target}")
if not target.is_dir() or target.is_symlink():
    raise SystemExit(f"Expected a real backup directory: {target}")
shutil.rmtree(target)
print(f"Removed verified raw backup: {target}")
PY
}

mapfile -t candidates < <(
  for source in "${DATA_ROOT}"/*; do
    [[ -d "${source}" ]] || continue
    name=$(basename "${source}")
    [[ -n "${incomplete[${name}]:-}" ]] && continue
    is_sqfs_mounted "${source}" && continue
    routes=$(find "${source}" -mindepth 1 -maxdepth 1 -type d | wc -l)
    printf '%08d\t%s\n' "${routes}" "${name}"
  done | sort -n -k1,1 | cut -f2-
)

echo "Completed raw scenarios queued: ${#candidates[@]}"

for name in "${candidates[@]}"; do
  source="${DATA_ROOT}/${name}"
  image="${IMAGE_ROOT}/${name}.sqfs"
  partial="${image}.partial"
  backup="${BACKUP_ROOT}/${name}"
  raw_manifest=$(mktemp "/tmp/${name}.raw.XXXXXX")
  image_manifest=$(mktemp "/tmp/${name}.sqfs.XXXXXX")

  cleanup_manifests() {
    rm -f "${raw_manifest}" "${image_manifest}"
  }
  trap 'cleanup_manifests' EXIT

  echo
  echo "[$(date '+%F %T')] Processing ${name}"

  if [[ -e "${backup}" ]]; then
    echo "Backup path already exists; refusing to overwrite it: ${backup}" >&2
    exit 3
  fi

  if [[ ! -f "${image}" ]]; then
    apparent_bytes=$(du -sb "${source}" | awk '{print $1}')
    available_bytes=$(df -PB1 "${IMAGE_ROOT}" | awk 'NR == 2 {print $4}')
    required_bytes=$((apparent_bytes + RESERVE_BYTES))
    if (( available_bytes < required_bytes )); then
      printf 'Insufficient temporary space for %s: %.2f GiB available, %.2f GiB conservatively required.\n' \
        "${name}" \
        "$(awk -v n="${available_bytes}" 'BEGIN {print n/1073741824}')" \
        "$(awk -v n="${required_bytes}" 'BEGIN {print n/1073741824}')" >&2
      exit 4
    fi

    [[ ! -e "${partial}" ]] || rm -f "${partial}"
    mksquashfs "${source}" "${partial}" \
      -comp zstd \
      -Xcompression-level 3 \
      -b 1M \
      -noappend \
      -processors "${PROCESSORS}"
    unsquashfs -s "${partial}" >/dev/null
    mv "${partial}" "${image}"
  else
    echo "Using existing image: ${image}"
    unsquashfs -s "${image}" >/dev/null
  fi

  echo "[$(date '+%F %T')] Comparing complete file manifests"
  make_raw_manifest "${source}" "${raw_manifest}"
  make_image_manifest "${image}" "${image_manifest}"
  if ! cmp -s "${raw_manifest}" "${image_manifest}"; then
    echo "Manifest mismatch for ${name}; original directory retained." >&2
    diff -u "${raw_manifest}" "${image_manifest}" | head -100 || true
    exit 5
  fi

  sample=$(find "${source}" -type f -name results.json.gz -printf '%P\n' -quit)
  if [[ -z "${sample}" ]]; then
    sample=$(find "${source}" -type f -printf '%P\n' -quit)
  fi
  if [[ -n "${sample}" ]]; then
    raw_hash=$(sha256sum "${source}/${sample}" | awk '{print $1}')
  else
    raw_hash=""
  fi

  echo "[$(date '+%F %T')] Switching ${name} to the verified read-only image"
  mv "${source}" "${backup}"
  mkdir "${source}"
  if ! mount -t squashfs -o loop,ro "${image}" "${source}"; then
    rmdir "${source}"
    mv "${backup}" "${source}"
    echo "Mount failed; restored original directory for ${name}." >&2
    exit 7
  fi

  mounted_files=$(find "${source}" -type f | wc -l)
  original_files=$(wc -l < "${raw_manifest}")
  if [[ "${mounted_files}" -ne "${original_files}" ]]; then
    umount "${source}"
    rmdir "${source}"
    mv "${backup}" "${source}"
    echo "Mounted file-count validation failed; restored ${name}." >&2
    exit 8
  fi

  if [[ -n "${sample}" ]]; then
    mounted_hash=$(sha256sum "${source}/${sample}" | awk '{print $1}')
    if [[ "${raw_hash}" != "${mounted_hash}" ]]; then
      umount "${source}"
      rmdir "${source}"
      mv "${backup}" "${source}"
      echo "Mounted sample content validation failed; restored ${name}: ${sample}" >&2
      exit 9
    fi
  fi

  remove_verified_backup "${backup}" "${BACKUP_ROOT}"
  image_bytes=$(stat -c '%s' "${image}")
  available_bytes=$(df -PB1 "${IMAGE_ROOT}" | awk 'NR == 2 {print $4}')
  printf '[%s] Completed %s: %d files, image %.2f GiB, free %.2f GiB\n' \
    "$(date '+%F %T')" "${name}" "${mounted_files}" \
    "$(awk -v n="${image_bytes}" 'BEGIN {print n/1073741824}')" \
    "$(awk -v n="${available_bytes}" 'BEGIN {print n/1073741824}')"

  cleanup_manifests
  trap - EXIT
done

echo
echo "[$(date '+%F %T')] Compaction pass completed."
echo "Mounted SquashFS scenarios: $(findmnt -rn -R "${DATA_ROOT}" -t squashfs | wc -l)"
echo "Remaining raw or incomplete scenarios:"
for source in "${DATA_ROOT}"/*; do
  [[ -d "${source}" ]] || continue
  is_sqfs_mounted "${source}" || basename "${source}"
done | sort
df -h "${DATA_ROOT}"
