#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export PYTHONPATH="$PWD"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export YOLO_AUTOINSTALL=false PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
# Isaac asset fetches must not inherit conflicting 7890/7892 proxy settings.
# Scope this to the demo subprocess; leave other jobs and API transport untouched.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
exec /data/ticvla/envs/warehouse-realtime/bin/python -u -m DynaNav.warehouse_loop.codex_demo "$@"
