#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_venv.sh .venv
#   TORCH_CHANNEL=cu128 bash scripts/setup_venv.sh .venv
#
# TORCH_CHANNEL options:
#   - cpu
#   - cu124
#   - cu126
#   - cu128
#
# Notes:
# - This script installs Python deps only. Install ffmpeg separately.
# - For GPU builds, you still need a compatible NVIDIA driver.

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CHANNEL="${TORCH_CHANNEL:-}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN or install python3." >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

EXTRA_INDEX_URL=""
if [[ -n "${TORCH_CHANNEL}" ]]; then
  case "${TORCH_CHANNEL}" in
    cpu|cu124|cu126|cu128)
      EXTRA_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CHANNEL}"
      ;;
    *)
      echo "ERROR: Unknown TORCH_CHANNEL='${TORCH_CHANNEL}'" >&2
      echo "Allowed: cpu | cu124 | cu126 | cu128 (or unset)" >&2
      exit 2
      ;;
  esac

  python -m pip install -r requirements.txt --extra-index-url "${EXTRA_INDEX_URL}"
else
  python -m pip install -r requirements.txt
fi

echo ""
echo "OK: environment ready"
echo "Activate: source ${VENV_DIR}/bin/activate"
