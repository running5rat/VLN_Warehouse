# Warehouse MVP implementation validation

This records implementation checks, not a trained navigation benchmark. See
[the usage guide](warehouse_mvp.md) for the data contract and full experiment
commands. Existing user changes and pretrained checkpoints were preserved.

## Completed checks

| Check | Result | Scope |
| --- | --- | --- |
| `python -m unittest discover -s tests/warehouse -v` | 21 passed | Causal input isolation, grouping/splits, RGB-D geometry, snapshot alignment/expiry, missing labels, zero-length/padded K/V, processed query state, LoRA gradients, spline gradients, default architecture, Flow/Query interfaces, training and checkpoint reload |
| Ruff on `ticvla/warehouse` and `tests/warehouse` | Passed | New Python implementation and tests |
| `git diff --check` | Passed | Patch whitespace |
| Real InternVL and TIC-VLA checkpoint integration | Passed | Reference forward/backward, actual generated cache, Flow forward/backward, save/reload and offline evaluation |
| Original TIC-VLA evaluation adapter | Passed | Original checkpoint loaded and one synthetic sample evaluated |
| Isaac Sim capture adapter | Passed | 8 synchronized rendered frames, RGB/depth/calibration/odometry capture, causal map preparation and nonempty reference labels |

The real-weight integration used the local `InternVL3-1B` and
`TIC-VLA-model.ckpt` on an RTX 4090, through the existing `tic-vla` Python
environment. It deliberately used small new heads, synthetic labels, one
optimizer step per stage and a 24-token generation limit. Unit tests additionally
cover the default 512-dimensional architecture. The real-weight smoke report is
at `/data/ticvla/outputs/warehouse-integration-bupqbflw/smoke_report.json`; it
includes exact configuration, checkpoints, manifest hash and evaluation output.
The legacy smoke report is at
`/data/ticvla/outputs/warehouse-legacy-smoke-n20v89_c/eval/report.json`.

The headless Isaac check created a small local USD scene and required no asset
download. Its prepared manifest is
`/tmp/warehouse-isaac-smoke-7xg3hzik/prepared/manifest.json`. Preparation yielded
4 mapped reference samples. Action supervision was deliberately unavailable
because this capture was not a task-aligned demonstration. This single-layout
capture has no validation/test split and establishes no generalization result.
Temporary capture files may be removed by normal system cleanup; the smoke
script in `tests/warehouse/run_isaac_capture_smoke.py` reproduces the check.

## Experiment status

The real-weight smoke had **0% valid structured parses** and invalid generated
snapshots were masked. Its deliberately truncated, minimally trained outputs do
not satisfy the ≥99% parse target. The randomly initialized action head's
trajectory errors and constraint violations are also not acceptance results.

Formal training, fixed-test-set ablations and a model-quality report still need
audited, task-aligned demonstrations from independent layout families. The
implementation provides the collection/conversion, reference and Flow stages,
matched Query baseline, original baseline adapter, map/condition/Oracle/delay
ablations, prediction artifacts and metric reports for those experiments.
No assertion is made about learned obstacle avoidance, unseen-layout accuracy,
closed-loop success, safety, or Orin latency. The next experiment should first
establish teacher-forced reference learning and actual-generation protocol
quality before interpreting Flow results conditioned on predicted snapshots.
