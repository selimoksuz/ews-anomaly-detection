#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
  if [[ "${PYTHON_BIN}" == "python3" ]]; then
    PYTHON_BIN="python"
  fi
fi

SOURCE="${1:-oracle}"
SCORING_MONTH="${2:--}"
MAX_TRAIN_ROWS="${3:-all}"
MAX_SCORE_ROWS="${4:--}"

"${PYTHON_BIN}" cli.py run-multivar-anomaly "${SOURCE}" "${SCORING_MONTH}" "${MAX_TRAIN_ROWS}" "${MAX_SCORE_ROWS}"
