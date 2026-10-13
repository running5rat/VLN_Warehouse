#!/usr/bin/env bash
set -euo pipefail
NAV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$NAV_SCRIPT_DIR/../.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export YOLO_AUTOINSTALL=false PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
exec /data/ticvla/envs/warehouse-realtime/bin/python -m DynaNav.warehouse_loop.navigation "$@"
