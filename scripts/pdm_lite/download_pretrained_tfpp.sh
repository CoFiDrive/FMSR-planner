#!/usr/bin/env bash
set -euo pipefail

DEST=${1:?Usage: download_pretrained_tfpp.sh DESTINATION_DIRECTORY}
mkdir -p "${DEST}"
exec 9>"${DEST}/.download.lock"
flock -n 9 || { echo "Checkpoint download already running" >&2; exit 3; }
archive="${DEST}/pretrained_models.zip"
url="https://s3.eu-central-1.amazonaws.com/avg-projects-2/garage_2/models/pretrained_models.zip"

# A nonempty file may still be an interrupted download. Only a ZIP that passes
# CRC verification may bypass resume; never mistake a partial file for success.
if ! unzip -tq "${archive}" >/dev/null 2>&1; then
  wget_args=(--continue --retry-connrefused --waitretry=10 --timeout=60 --tries=0 --progress=dot:giga)
  if [[ "${CARLA_GARAGE_USE_PROXY:-0}" != "1" ]]; then
    wget_args+=(--no-proxy)
  fi
  until wget "${wget_args[@]}" "${url}" -O "${archive}"; do
    echo "Download interrupted; retaining checkpoint archive and retrying in 30 seconds."
    sleep 30
  done
fi
unzip -tq "${archive}"
unzip -n "${archive}" -d "${DEST}"
find "${DEST}" -type f \( -name 'model_0030_0.pth' -o -name 'model_0030_1.pth' -o -name 'model_0030_2.pth' \) -print
