#!/usr/bin/env bash
set -euo pipefail
ALTRO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ALTRO_ENV="${TICVLA_ALTRO_ENV:-/data/ticvla/envs/warehouse-oracle}"
ALTRO_JULIA="${TICVLA_JULIA:-/data/ticvla/tools/julia-1.10.10/bin/julia}"
export JULIA_DEPOT_PATH="${ALTRO_ENV}/depot"
export JULIA_NUM_THREADS=2
exec "${ALTRO_JULIA}" --project="${ALTRO_ENV}" \
  "${ALTRO_SCRIPT_DIR}/../../ticvla/warehouse/dataset_tools/altro_worker.jl" "$@"
