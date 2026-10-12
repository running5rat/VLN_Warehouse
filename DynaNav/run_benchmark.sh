#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TICVLA_DYNANAV_ROOT="${TICVLA_DYNANAV_ROOT:-${SCRIPT_DIR}}"
export ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
export ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-${ISAAC_SIM_ROOT}/python.sh}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

version_ge() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" == "$2" ]]
}

if [[ ! -x "${ISAAC_SIM_PYTHON}" ]]; then
  echo "Error: Isaac Sim Python launcher is not executable: ${ISAAC_SIM_PYTHON}" >&2
  echo "Set ISAAC_SIM_ROOT/ISAAC_SIM_PYTHON in .env.testing." >&2
  exit 1
fi

# Isaac Sim 5.0 supports Ubuntu 22.04/24.04. Its native libcarb.so needs
# GLIBC >= 2.34 and GLIBCXX >= 3.4.30; fail here with an actionable message
# instead of letting the dynamic loader emit a long list of missing symbols.
GLIBC_VERSION="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
SYSTEM_LIBSTDCXX="$(ldconfig -p 2>/dev/null | awk '/libstdc\+\+\.so\.6 \(/ && !path {path=$NF} END {print path}')"
GLIBCXX_VERSION=""
if [[ -n "${SYSTEM_LIBSTDCXX}" && -r "${SYSTEM_LIBSTDCXX}" ]]; then
  GLIBCXX_VERSION="$(strings "${SYSTEM_LIBSTDCXX}" | grep -E '^GLIBCXX_[0-9]+\.[0-9]+\.[0-9]+$' | sort -V | tail -n 1 | cut -d_ -f2)"
fi

if [[ -z "${GLIBC_VERSION}" ]] || ! version_ge "${GLIBC_VERSION}" "2.34"; then
  echo "Error: Isaac Sim 5.0 requires GLIBC >= 2.34; found ${GLIBC_VERSION:-unknown}." >&2
  echo "This usually means the container is based on Ubuntu 20.04." >&2
  echo "Run DynaNav in Ubuntu 22.04/24.04 or nvcr.io/nvidia/isaac-sim:5.0.0." >&2
  echo "Do not replace or symlink the system libc inside this container." >&2
  exit 1
fi

if [[ -z "${GLIBCXX_VERSION}" ]] || ! version_ge "${GLIBCXX_VERSION}" "3.4.30"; then
  echo "Error: Isaac Sim 5.0 requires GLIBCXX >= 3.4.30; found ${GLIBCXX_VERSION:-unknown}." >&2
  echo "Install a current libstdc++6 package in the Ubuntu 22.04/24.04 runtime." >&2
  exit 1
fi

CONFIG="${1:-${TICVLA_DYNANAV_ROOT}/configs/benchmark_example.yaml}"
shift || true

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="$(cd "$(dirname "${CONFIG}")" && pwd)/$(basename "${CONFIG}")"
fi

cd "${TICVLA_DYNANAV_ROOT}"
echo "Launching Isaac Sim with CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
exec "${ISAAC_SIM_PYTHON}" benchmark.py -c "${CONFIG}" --navigation_method ticvla "$@"
