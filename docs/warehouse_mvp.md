# Warehouse MVP: local map → reference line → Flow AE

This opt-in pipeline implements the first, offline milestone of
`dynamic-warehouse-vln-design-v0.6.pdf` (§4–6, §9). It supports local forward
instructions, turns, obstacle detours and waiting. Legacy TIC-VLA, its configs,
checkpoints, DynaNav evaluation and RL remain separate.

The MVP has a TEXT-only InternVL3-1B branch, a spline reference decoder, a
trainable K/V compressor and a single-trajectory Flow Matching action head.
Nine modes, person following, reverse motion, learned scoring, closed-loop policy
deployment, persistent semantic mapping and multiple robots are subsequent work.

## Environment and entrypoints

Use the existing training environment (`requirements-train.txt`). The new
pipeline uses PyTorch directly; its reference adapter implements attention LoRA
without requiring PEFT. Data preparation requires NumPy, SciPy and Pillow. No
Isaac imports occur in offline training. The existing `ticvla.TICVLA` public
import is preserved, now loaded lazily so data tools do not import the VLM.

```bash
source .env.training
export TICVLA_WAREHOUSE_DATA=/data/ticvla/warehouse_v1

python -m ticvla.training.train --pipeline warehouse_mvp \
  --stage reference --config configs/train_warehouse_reference.yaml
```

`python -m ticvla.warehouse.train` is an equivalent standalone entrypoint.
Without the new pipeline/stage, the original `vlm` and `action` commands retain
their existing behavior. All output directories should be outside source files.

## 1. Collect and prepare data

### Raw episode contract

Each episode contains `episode.json` and `frames.jsonl`. The metadata declares:

```json
{
  "schema_version": 1,
  "layout_family": "layout-a",
  "episode_id": "layout-a-001",
  "frames_file": "frames.jsonl",
  "task": {
    "task_id": "task-001",
    "instruction": "Move straight through the visible free passage and stop.",
    "hard_constraints": ["Stay in observed free space."]
  },
  "supervision": {
    "local_goal_odom": [4.0, 0.0, 0.0],
    "actions_task_aligned": true
  }
}
```

Every frame records `t_obs` in seconds, paths `rgb` and optional `depth`,
`T_odom_body` (4×4, body→odom), and `odom_epoch`. RGB-D frames additionally require
`K` and `T_body_camera` (optical camera→FLU body). Depth is optical-axis Z in
metres, **not ray distance**. Optional `robot_state` is `[v, omega]` in m/s and
rad/s. Missing state is derived from previous/current poses only.

The annotated goal is supervisory information, never an Actor input. Set
`actions_task_aligned=true` only for demonstrations confirmed to execute the
declared task. Missing depth, map, task alignment or reference labels do not
become fabricated targets. Raw geometry must share a calibrated clock and frame.

### Isaac Sim adapter

Copy and edit `configs/warehouse_capture.example.json`: set the actual camera
and chassis prim paths, layout family, episode ID and task. The capture metadata
adds `camera_prim`, `body_prim`, `resolution`, and `capture_hz` to the contract.
The example leaves action alignment **false** because the existing collector
uses randomly generated robot commands.

```bash
source .env.testing
cd DynaNav
"${ISAAC_SIM_PYTHON}" collect_data.py -c configs/data_collection/warehouse.yaml \
  --warehouse-metadata /absolute/path/capture.json \
  --warehouse-output /data/ticvla/warehouse_raw/layout-a-001
```

The optional writer captures RGB, depth and camera calibration in the same
Replicator callback. It uses the rendered frame's `reference_time`, not the
callback's current simulation/wall time. Body pose is recovered from that frame's
camera pose and a fixed camera/chassis extrinsic, expressed relative to the first
recorded body pose. This is **simulated odometry**, not a validated VIO system.
Only metre-based, fixed-extrinsic pinhole cameras are supported. Resets require
a new episode. The reusable `EpisodeRecorder` can also receive synchronized
real-world packets. Privileged annotations, when supplied, are stored separately.

### Split before windowing

Write an explicit `splits.json`, grouping every variant of the same physical
layout together:

```json
{"layout-a": "train", "layout-b": "val", "layout-c": "test"}
```

```bash
python -m ticvla.warehouse.prepare \
  --episodes /data/ticvla/warehouse_raw/layout-*/episode.json \
  --splits /absolute/path/splits.json --output "${TICVLA_WAREHOUSE_DATA}"

python -m ticvla.warehouse.audit \
  --manifest "${TICVLA_WAREHOUSE_DATA}/manifest.json"
```

The output is immutable: use a new directory for a changed dataset. Preparation
produces `manifest.json`, `audit.json`, per-sample JSON, occupancy/age arrays and
map images. Each sample has separate `observation`, `supervision`, and
`provenance` namespaces. Validate coverage before training; both training and
validation must have usable labels.

Local occupancy uses only past/current RGB-D, projects through 3-D transforms,
and keeps a 5-second evidence window. Same-frame occupied hits win over free rays;
new observations supersede old ones. The robot's current footprint is additional
proprioceptive free-space evidence, without clearing observed obstacles. Space
outside observed geometry and that footprint remains unknown. This is a local
geometry baseline, not permanent static/dynamic semantic fusion.

Reference labels use A* on footprint-inflated known free cells, visibility
simplification and collision-checked cubic smoothing. References are sampled by
arc length. Action labels are interpolated at fixed physical times; gaps and
truncated futures are masked. The two targets are never substituted for one
another. `--derive-local-tasks` creates geometry-grounded reference-only template
tasks and deliberately disables action supervision.

For existing DynaNav data, `python -m ticvla.warehouse.import_dynanav --help`
provides an importer requiring an audited catalog. Each catalog entry is keyed
by the original episode name and specifies `layout_family`, `split`,
`instruction`, `actions_task_aligned`, `pose_frame` (`body_flu`, `usd_camera`, or
`optical_camera`), and `time_unit` (`seconds` or `frame_index`). Frame-index clocks
also require `source_hz`; camera poses require `T_body_camera`. The importer
uses raw `trajectory.csv`, deduplicates overlapping JSON windows, excludes Spot,
and marks its split as **episode-only generalization**. It supplies no map or
reference ground truth. Do not mix its validation episodes into a warehouse
layout-holdout experiment and call that unseen-layout evaluation.

## 2. Train the reference branch

Use `configs/train_warehouse_reference.yaml` with the prepared manifest. The
default imports only matching VLM parameters from `TICVLA_CHECKPOINT_PATH`.
Set `vlm_checkpoint: null` to initialize from base InternVL instead.

The vision encoder and existing visual projection are frozen. Rank-16 attention
Q/K/V/O LoRA and the reference decoder train jointly. Map images use the existing
visual interface. RGB history is sampled at 0/1/2/3 seconds; each present image
retains 256 tokens. The PDF's unequal history-token compression is deferred.

The response protocol is case-sensitive:

```text
<landmark>[]</landmark><local>unknown</local>
<Route>remaining local instruction</Route>
<waypoint>(x,y);(x,y);(x,y);(x,y);(x,y)</waypoint><REFLINE>
```

`[]` is permitted for an absent route. Empty landmark and unknown Local outputs
are explicit MVP placeholders, not semantic map detections. The immutable
instruction and constraints remain in every prompt separately from the working
Route. IDs/versions are metadata, not identity embeddings.

The decoder reads the final hidden state of the **processed** query marker.
Generation performs one visual/prompt prefill followed by cached token steps;
it does not rerun the full generated dialogue. Invalid/truncated responses remain
invalid even if a query is appended for diagnostic decoding.

Losses are masked text CE, valid-point SmoothL1, length CE, validity BCE and
anchor geometry consistency. The latter compares differentiable line samples
with audited anchors, not parsed generated numbers. Validation uses actual
generation; GT answer text is never supplied there. Confidence is a supervised
validity output, not a certified collision probability.

Checkpoints include adapter/decoder weights, optimizer/RNG state, architecture,
configuration, vocabulary hash and split manifest. The tokenizer is saved beside
them. Keep the configured base model and legacy initialization checkpoint
available: adapter checkpoints do not duplicate those frozen weights.

## 3. Cache semantics and train the action branch

```bash
python -m ticvla.warehouse.cache --config configs/train_warehouse_flow.yaml \
  --checkpoint "${TICVLA_OUTPUT_DIR}/warehouse/reference/best.pt"

python -m ticvla.training.train --pipeline warehouse_mvp \
  --stage flow --config configs/train_warehouse_flow.yaml

# Same visual/map/reference/semantic conditions; learned query regression baseline.
python -m ticvla.training.train --pipeline warehouse_mvp \
  --stage flow --config configs/train_warehouse_flow.yaml --action-head query
```

Cache files contain actual generated text/reference, **raw final-layer K+V** and
frozen current-image features. The trainable 16-query context compressor executes
inside the action model. Cache indices bind the reference checkpoint, manifest,
observation content, geometry settings and map ablation. Changed versions require
a new cache directory; partial caches cannot silently supply missing records.

Replay selects a past snapshot from the same episode, task and odom epoch. It
checks route compatibility and age, then transforms/crops the reference line.
Invalid, unavailable or expired semantics are masked; current perception remains.
Configured delays are **synthetic replay ages**, not measured service completion
times. Cache wall-clock timings are diagnostic only. Missing previous proposals
remain absent; no expert future is inserted into the condition stream.

Flow uses a 512-dimensional, 3-layer, 8-head Transformer, normalized Gaussian
trajectory interpolation and a velocity-field MSE over valid physical timesteps.
Context is encoded once per sampling call. Four Euler steps produce
`[B,1,1,30,2]` candidates at 0.1-second spacing, in metres. The Query baseline
uses the same condition encoder. Its output directory is `warehouse/query/flow`.
These defaults are starting configurations, not validated deployment budgets.

## 4. Evaluate and run ablations

```bash
python -m ticvla.warehouse.evaluate --stage reference \
  --config configs/train_warehouse_reference.yaml \
  --checkpoint "${TICVLA_OUTPUT_DIR}/warehouse/reference/best.pt" \
  --suite --output "${TICVLA_OUTPUT_DIR}/warehouse/eval/reference"

python -m ticvla.warehouse.evaluate --stage flow \
  --config configs/train_warehouse_flow.yaml \
  --checkpoint "${TICVLA_OUTPUT_DIR}/warehouse/flow/best.pt" \
  --suite --output "${TICVLA_OUTPUT_DIR}/warehouse/eval/flow"

python -m ticvla.warehouse.evaluate --stage legacy \
  --config configs/train_warehouse_flow.yaml --checkpoint "${TICVLA_CHECKPOINT_PATH}" \
  --output "${TICVLA_OUTPUT_DIR}/warehouse/eval/legacy"
```

Reference suites compare correct/removed/shuffled map inputs and recursive Route
replay. Action suites scan configured delays and none/time/full pose compensation,
reference-only/semantic-only/both conditions, and a clearly labeled privileged
Oracle-reference diagnostic. For action-side map ablations, build separate cache
directories with `cache --map-mode removed|shuffled`, then point evaluation
configs to those caches. For Query evaluation, set `action_head: query` in the
evaluation config and use its checkpoint. Match splits and training budgets.

Reports contain per-sample results, aggregate errors, validity/length quality,
parse rate, geometric feasibility, ADE/FDE, speed/acceleration violations,
reference deviation, current-map collision/unknown checks, P50/P95/P99 latency,
peak action-model CUDA memory, and plots. Saved predictions are metric `.npz`
arrays. Flow timing excludes VLM/cache loading; legacy/reference timing includes
their respective service paths. Current-map checks do not measure collisions
against unobserved future dynamic actors. Empty metrics are absent, not zero.

The ≥99% parse-rate target remains an experiment acceptance target. A successful
smoke test or low teacher-forced loss does not establish navigation quality,
unseen-layout generalization, closed-loop success, safe execution, or Orin speed.
The original RGB-only baseline is reported separately because its information
budget differs; use the matched Query baseline for architecture attribution.

## Verification

The completed checks and remaining experiment requirements are recorded in
[warehouse_mvp_validation.md](warehouse_mvp_validation.md).

```bash
python -m unittest discover -s tests/warehouse -v

# Explicit integration check with local real weights and synthetic labels.
python tests/warehouse/run_local_smoke.py \
  --model "${TICVLA_BASE_MODEL_PATH}" --vlm-checkpoint "${TICVLA_CHECKPOINT_PATH}" \
  --output-parent "${TICVLA_OUTPUT_DIR}"

# Headless RGB-D capture on a tiny locally generated USD scene; no downloaded asset.
"${ISAAC_SIM_PYTHON}" tests/warehouse/run_isaac_capture_smoke.py
```

Tests cover frame projection, odom rotation/prefix trimming, unknown-space and
footprint checks, split leakage, future-label isolation, cache validity, missing
labels, spline gradients, K/V padding, LoRA freezing, processed query markers,
sampling determinism, checkpoint reload and action training/evaluation.
The explicit smoke scripts write fresh temporary/output directories and never
modify the pretrained checkpoint or supply navigation benchmark evidence.
