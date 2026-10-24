#!/usr/bin/env bash
# Continue the degraded-route collection only after the live pilot passes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PILOT="${PILOT_EVALUATION_OUTPUT_DIR:?Set the pilot evaluation output directory}"
PYTHON_BIN="${PYTHON_BIN:-/home/vcdc/anaconda3/envs/fmsr/bin/python}"
[[ -f "$PILOT/.evaluation.lock" ]] || { echo "Missing pilot lock: $PILOT" >&2; exit 1; }
echo "Waiting for pilot completion: $PILOT"
flock -x "$PILOT/.evaluation.lock" true
"$PYTHON_BIN" - "$PILOT/joint_collection_verification.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file() or json.loads(path.read_text()).get('status') != 'passed':
    raise SystemExit('Pilot verification did not pass; follow-up collection will not start')
print('Pilot passed; collecting the remaining 11 degraded routes', flush=True)
PY
export COLLECTION_ROUTE_IDS="${COLLECTION_ROUTE_IDS:-2115,2127,2215,17563,23670,23918,24206,24252,24816,25300,25424}"
exec bash scripts/collect_learned_teacher_pdm_lite.sh
