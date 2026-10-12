#!/usr/bin/env bash
set -euo pipefail
RT_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TICVLA_IO_PYTHON=/data/ticvla/envs/warehouse-realtime/bin/python
export PATH="/data/ticvla/envs/warehouse-realtime/bin:$PATH"
export YOLO_AUTOINSTALL=false PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
exec bash "$RT_SCRIPT_DIR/run_live.sh" --semantic-backend llm --llm-mode objects-only --base-model-only --perception-backend yoloe-realtime --render-hz 5 "$@"
