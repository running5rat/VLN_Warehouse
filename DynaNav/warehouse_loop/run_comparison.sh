#!/usr/bin/env bash
# Explicit phases keep development, frozen evaluation and independent review separate.
set -euo pipefail
COMPARISON_ROOT="${1:?Usage: run_comparison.sh OUTPUT_ROOT capture|development|evaluation|truth|report}"
COMPARISON_PHASE="${2:?Missing phase}"
COMPARISON_PYTHON="${COMPARISON_PYTHON:-/data/ticvla/envs/warehouse-io/bin/python}"
COMPARISON_WORKERS="${COMPARISON_WORKERS:-3}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p -- "$COMPARISON_ROOT"
case "$COMPARISON_PHASE" in
  capture)
    for variant in development original color layout surface lighting occlusion; do
      bash DynaNav/warehouse_loop/run_host.sh --landmark-variant "$variant" --capture-hz 1 \
        --output "$COMPARISON_ROOT/capture-$variant" > "$COMPARISON_ROOT/capture-$variant.log" 2>&1
    done
    ;;
  development|evaluation)
    "$COMPARISON_PYTHON" -m ticvla.warehouse.run_landmark_benchmark --root "$COMPARISON_ROOT" --stage prepare --split "$COMPARISON_PHASE"
    "$COMPARISON_PYTHON" -m ticvla.warehouse.run_landmark_benchmark --root "$COMPARISON_ROOT" --stage qa --split "$COMPARISON_PHASE"
    workers="$COMPARISON_WORKERS"
    if [[ "$COMPARISON_PHASE" == development ]]; then workers=1; fi
    for backend in yoloe-ppocr florence2; do
      "$COMPARISON_PYTHON" -m ticvla.warehouse.run_landmark_benchmark --root "$COMPARISON_ROOT" --stage infer \
        --split "$COMPARISON_PHASE" --backend "$backend" --workers "$workers"
    done
    ;;
  truth)
    if [[ -e "$COMPARISON_ROOT/evaluation/annotations.json" ]]; then
      echo 'Annotations already exist; refusing to overwrite readability review.' >&2
      exit 2
    fi
    "$COMPARISON_PYTHON" -m ticvla.warehouse.run_landmark_benchmark --root "$COMPARISON_ROOT" --stage truth
    ;;
  report)
    "$COMPARISON_PYTHON" -m ticvla.warehouse.run_landmark_benchmark --root "$COMPARISON_ROOT" --stage evaluate
    "$COMPARISON_PYTHON" -m ticvla.warehouse.perception.report "$COMPARISON_ROOT"
    ;;
  *) echo "Unknown phase: $COMPARISON_PHASE" >&2; exit 2 ;;
esac
