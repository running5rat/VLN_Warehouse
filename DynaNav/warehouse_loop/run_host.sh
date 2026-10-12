#!/usr/bin/env bash
# Launch this workspace's Isaac 5.0 circuit from either the host or its container.
set -euo pipefail

LOOP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOOP_REPO_ROOT="$(cd -- "${LOOP_SCRIPT_DIR}/../.." && pwd)"

if [[ -f /.dockerenv ]]; then
  LOOP_HOST_USER="${TICVLA_HOST_USER:-xinglinlu}"
  LOOP_HOST_REPO="${TICVLA_HOST_REPO:-/home/xinglinlu/autodriving_ws/docker/VLN/TIC-VLA}"
  if [[ "$(id -u)" != 0 ]]; then
    echo 'Container launch requires root and access to the host PID/mount namespaces. Alternatively run this script from a host terminal.' >&2
    exit 1
  fi
  echo "Entering host runtime as ${LOOP_HOST_USER}: ${LOOP_HOST_REPO}"
  exec nsenter -t 1 -m -- /usr/sbin/runuser -u "${LOOP_HOST_USER}" -- \
    /usr/bin/env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME \
    -u PYTHONEXE -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD \
    -u DISPLAY -u XAUTHORITY -u XDG_RUNTIME_DIR \
    /bin/bash "${LOOP_HOST_REPO}/DynaNav/warehouse_loop/run_host.sh" "$@"
fi

cd -- "${LOOP_REPO_ROOT}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV PYTHONHOME PYTHONEXE LD_LIBRARY_PATH LD_PRELOAD
export PYTHONPATH="${LOOP_REPO_ROOT}"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
LOOP_ISAAC_PYTHON="${TICVLA_ISAAC_PYTHON:-/data/isaacsim/python.sh}"
if [[ ! -x "${LOOP_ISAAC_PYTHON}" ]]; then
  echo "Isaac Python launcher is missing: ${LOOP_ISAAC_PYTHON}" >&2
  exit 1
fi
exec /usr/bin/python3 "${LOOP_SCRIPT_DIR}/host_runtime.py" \
  "${LOOP_ISAAC_PYTHON}" "${LOOP_SCRIPT_DIR}/run.py" "$@"
