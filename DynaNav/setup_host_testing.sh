#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${TICVLA_TEST_ENV_FILE:-${REPO_ROOT}/.env.testing}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Error: a conda environment is active: ${CONDA_PREFIX}" >&2
  echo "Run 'conda deactivate' until CONDA_PREFIX is unset, then retry." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: testing environment file not found: ${ENV_FILE}" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

required_paths=(
  "${ISAAC_SIM_PYTHON}"
  "${TICVLA_BASE_MODEL_PATH}/config.json"
  "${TICVLA_CHECKPOINT_PATH}"
  "${TICVLA_DYNANAV_ROOT}/benchmark.py"
  "${TICVLA_DYNANAV_ROOT}/assets/office.usd"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "Error: required path does not exist: ${path}" >&2
    exit 1
  fi
done

glibc_version="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [[ "$(printf '%s\n%s\n' "2.34" "${glibc_version}" | sort -V | head -n 1)" != "2.34" ]]; then
  echo "Error: host GLIBC must be >= 2.34; found ${glibc_version:-unknown}." >&2
  exit 1
fi

mkdir -p "${TICVLA_OUTPUT_DIR}"

# DynaNav ships its runtime model modules inside DynaNav/ and adds that
# directory to sys.path itself. Avoid an editable top-level install here:
# shared host/container workspaces may contain root-owned ticvla.egg-info
# metadata, which a non-root host user cannot update.
"${ISAAC_SIM_PYTHON}" -m pip install -r "${REPO_ROOT}/requirements-test.txt"

"${ISAAC_SIM_PYTHON}" - <<'PY'
import torch
import transformers
from isaacsim import SimulationApp

print(f"Isaac Sim Python imports OK; torch={torch.__version__}, transformers={transformers.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available from Isaac Sim Python")
PY

echo "DynaNav host setup complete. Run:"
echo "  source ${ENV_FILE}"
echo "  ${REPO_ROOT}/DynaNav/run_benchmark.sh ${REPO_ROOT}/DynaNav/configs/benchmark_example.yaml"
