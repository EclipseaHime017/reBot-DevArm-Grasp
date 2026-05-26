#!/usr/bin/env bash
set -euo pipefail

# Install pyorbbecsdk into the currently active Python environment.
#
# Use the wheel path by default:
#   conda activate graspnet
#   bash scripts/install_pyorbbecsdk.sh
#
# Build the cloned source tree only when you really need a local SDK build:
#   bash scripts/install_pyorbbecsdk.sh --from-source

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

echo "[INFO] python: $("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"
echo "[INFO] version: $("$PYTHON_BIN" -c 'import sys; print(sys.version.split()[0])')"

if [[ "${1:-}" == "--from-source" ]]; then
  SDK_DIR="$PROJECT_ROOT/sdk/pyorbbecsdk"
  if [[ ! -d "$SDK_DIR" ]]; then
    echo "[FAIL] source tree not found: $SDK_DIR" >&2
    exit 1
  fi

  echo "[INFO] installing source-build requirements"
  "$PYTHON_BIN" -m pip install -U pip wheel pybind11 pybind11-global cmake

  echo "[INFO] building pyorbbecsdk source tree"
  (
    cd "$SDK_DIR"
    PATH="$(dirname "$("$PYTHON_BIN" -c 'import sys; print(sys.executable)')"):$PATH" \
      bash scripts/build_whl/build_linux_whl_local.sh
    "$PYTHON_BIN" -m pip install --force-reinstall dist/pyorbbecsdk2-*.whl
  )
else
  echo "[INFO] installing prebuilt pyorbbecsdk2 wheel from PyPI"
  "$PYTHON_BIN" -m pip install -U pyorbbecsdk2
fi

"$PYTHON_BIN" - <<'PY'
import pyorbbecsdk
print("[OK] import pyorbbecsdk:", pyorbbecsdk.__file__)
PY

echo "[OK] pyorbbecsdk install check passed"

