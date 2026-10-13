#!/usr/bin/env bash
# Run a new original-scene circuit, record live publications and package results.
set -euo pipefail
DEMO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEMO_REPO_ROOT="$(cd -- "$DEMO_SCRIPT_DIR/../.." && pwd)"
cd -- "$DEMO_REPO_ROOT"
DEMO_STAMP="$(date +%Y%m%d-%H%M%S)"
DEMO_RUN="${1:-/data/ticvla/outputs/warehouse-loop/demo-original-$DEMO_STAMP}"
DEMO_DELIVERY="${2:-$DEMO_SCRIPT_DIR/original-$DEMO_STAMP}"
DEMO_PYTHON=/data/ticvla/envs/warehouse-realtime/bin/python
export PYTHONPATH="$DEMO_REPO_ROOT"
"$DEMO_PYTHON" "$DEMO_SCRIPT_DIR/record_demo.py" --run "$DEMO_RUN" --delivery "$DEMO_DELIVERY"
"$DEMO_PYTHON" "$DEMO_SCRIPT_DIR/package_demo.py" --run "$DEMO_RUN" --delivery "$DEMO_DELIVERY"
"$DEMO_PYTHON" "$DEMO_SCRIPT_DIR/verify_demo.py" --delivery "$DEMO_DELIVERY"
echo "Demo ready: $DEMO_DELIVERY/index.html"
