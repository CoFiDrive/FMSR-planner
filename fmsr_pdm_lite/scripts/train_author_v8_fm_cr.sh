#!/usr/bin/env bash
# Same v8 FM architecture and sampler, plus delayed spectral endpoint CR.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-author_v8_pdm_fm_cr_$(date +%Y%m%d_%H%M%S)}"
exec bash "$TASK_ROOT/scripts/train_author_v8_fm.sh" \
  --flow-contract-enable \
  --flow-contract-weight "${FLOW_CONTRACT_WEIGHT:-0.01}" \
  --flow-contract-fraction "${FLOW_CONTRACT_FRACTION:-0.25}" \
  --flow-contract-interval "${FLOW_CONTRACT_INTERVAL:-4}" \
  --flow-contract-fd-epsilon "${FLOW_CONTRACT_FD_EPSILON:-0.1}" \
  --flow-contract-fd-chunk "${FLOW_CONTRACT_FD_CHUNK:-2}" \
  --flow-contract-delay-epochs "${FLOW_CONTRACT_DELAY_EPOCHS:-2}" \
  --flow-contract-ramp-epochs "${FLOW_CONTRACT_RAMP_EPOCHS:-1}" "$@"
