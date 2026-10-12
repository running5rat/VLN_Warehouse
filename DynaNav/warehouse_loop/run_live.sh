#!/usr/bin/env bash
# One container command starts Isaac on the host and perception in its own env.
set -euo pipefail
LOOP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOOP_REPO_ROOT="$(cd -- "${LOOP_SCRIPT_DIR}/../.." && pwd)"
cd -- "${LOOP_REPO_ROOT}"
export PYTHONPATH="${LOOP_REPO_ROOT}"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for LOOP_ARG in "$@"; do
  if [[ "$LOOP_ARG" == "yoloe-realtime" || "$LOOP_ARG" == "--perception-backend=yoloe-realtime" ]]; then
    export TICVLA_IO_PYTHON="${TICVLA_IO_PYTHON:-/data/ticvla/envs/warehouse-realtime/bin/python}"
    export YOLO_AUTOINSTALL=false PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
  fi
done
exec "${TICVLA_IO_PYTHON:-/data/ticvla/envs/warehouse-io/bin/python}" \
  -m DynaNav.warehouse_loop.live "$@"
