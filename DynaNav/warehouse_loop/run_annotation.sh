#!/usr/bin/env bash
set -euo pipefail
ANNOTATION_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${ANNOTATION_SCRIPT_DIR}/../.."
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
exec "${TICVLA_ANNOTATION_PYTHON:-/data/ticvla/envs/warehouse-io/bin/python}" \
  -m DynaNav.warehouse_loop.annotation "$@"
