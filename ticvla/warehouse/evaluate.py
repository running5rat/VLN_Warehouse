"""Offline, causal evaluation. No closed-loop or real-time claims are made here."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import CachedActionDataset, RecordDataset, action_collate, to_device
from .geometry import collision_free
from .schema import TaskContract, TaskState, fingerprint
from .train import load_action, load_reference, seed_all


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def summarize(rows):
    report = {"samples": len(rows)}
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    for key in sorted(keys):
        values = [
            float(r[key])
            for r in rows
            if isinstance(r.get(key), (int, float, bool)) and np.isfinite(r[key])
        ]
        if values:
            report[key] = float(np.mean(values))
            if key.endswith("_s") or key == "age":
                report[key + "_percentiles"] = dict(
                    zip(
                        ("p50", "p95", "p99"),
                        np.percentile(values, [50, 95, 99]).tolist(),
                    )
                )
    return report


def plot_prediction(record, predicted, path, reference=None):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    obs, sup = record["observation"], record["supervision"]
    fig, ax = plt.subplots(figsize=(6, 6))
    if obs.get("map_valid"):
        with np.load(obs["map_grid"], allow_pickle=False) as data:
            grid = data["grid"]
            resolution = (
                float(data["resolution"])
                if "resolution" in data
                else obs.get("map_resolution", 0.1)
            )
            high = (len(grid) // 2 + 0.5) * resolution
            low = (len(grid) // 2 - len(grid) + 0.5) * resolution
            ax.imshow(
                grid,
                origin="upper",
                extent=[high, low, low, high],
                cmap="gray_r",
                vmin=-1,
                vmax=1,
                alpha=0.3,
            )
    gt = np.array(sup["actions"])[np.array(sup["action_mask"], bool)]
    if len(gt):
        ax.plot(gt[:, 1], gt[:, 0], label="timed demonstration")
    if len(predicted):
        ax.plot(predicted[:, 1], predicted[:, 0], label="prediction")
    if reference is not None and len(reference):
        ax.plot(reference[:, 1], reference[:, 0], "--", label="reference")
    ax.set(xlabel="body y (left), m", ylabel="body x (forward), m", aspect="equal")
    if ax.get_xlim()[0] < ax.get_xlim()[1]:
        ax.invert_xaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def trajectory_metrics(prediction, record, cfg):
    sup = record["supervision"]
    truth, mask = np.array(sup["actions"]), np.array(sup["action_mask"], bool)
    metrics = {}
    if mask.any():
        error = np.linalg.norm(prediction - truth, axis=-1)
        metrics.update(
            ade=float(error[mask].mean()), fde=float(error[np.flatnonzero(mask)[-1]])
        )
    velocity = (
        np.diff(np.vstack((np.zeros((1, 2)), prediction)), axis=0) / cfg.action_dt
    )
    current = np.array([record["observation"]["robot_state"][0], 0.0])
    acceleration = np.diff(np.vstack((current[None], velocity)), axis=0) / cfg.action_dt
    metrics["speed_violation_rate"] = float(
        (np.linalg.norm(velocity, axis=-1) > cfg.max_speed).mean()
    )
    metrics["acceleration_violation_rate"] = float(
        (np.linalg.norm(acceleration, axis=-1) > cfg.max_acceleration).mean()
    )
    if record["observation"].get("map_valid"):
        with np.load(record["observation"]["map_grid"], allow_pickle=False) as data:
            metrics["observed_map_collision_or_unknown"] = not collision_free(
                np.vstack((np.zeros((1, 2)), prediction)),
                data["grid"],
                cfg.map_resolution,
                cfg.footprint_radius,
            )
    return metrics


@torch.no_grad()
def evaluate_reference(
    cfg,
    checkpoint,
    output,
    split="test",
    map_mode="correct",
    recursive=False,
    limit=None,
):
    model, _ckpt = load_reference(checkpoint, cfg.device)
    model.cfg.max_new_tokens = cfg.max_new_tokens
    model.eval()
    dataset = RecordDataset(cfg.manifest, split)
    order = sorted(
        range(len(dataset)),
        key=lambda i: (
            dataset[i]["provenance"]["episode_id"],
            dataset[i]["observation"]["t_obs"],
        ),
    )
    rows, states = [], {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for position, i in enumerate(order):
        if limit is not None and position >= limit:
            break
        record = dataset[i]
        obs = copy.deepcopy(record["observation"])
        key = (
            record["provenance"]["layout_family"],
            record["provenance"]["episode_id"],
            obs["task"]["task_id"],
        )
        if recursive:
            if key not in states:
                task = obs["task"]
                states[key] = TaskState(
                    TaskContract(
                        task["task_id"],
                        task["instruction"],
                        tuple(task.get("hard_constraints", [])),
                    ),
                    obs["route"],
                )
            obs["route"], obs["route_revision"] = (
                states[key].route,
                states[key].route_revision,
            )
        donor = dataset[(i + max(1, len(dataset) // 2)) % len(dataset)]["observation"]
        if map_mode == "shuffled" and len(dataset) < 2:
            raise ValueError("Map shuffle requires two or more samples")
        sync(cfg.device)
        start = time.perf_counter()
        prediction = model.generate_record(obs, map_mode, donor.get("map_image"))
        sync(cfg.device)
        elapsed = time.perf_counter() - start
        sup = record["supervision"]
        gt, mask = np.array(sup["reference_xy"]), np.array(sup["reference_mask"], bool)
        pred = prediction["reference_xy"][0].cpu().numpy()
        pred_mask = prediction["reference_mask"][0].cpu().numpy()
        parsed = prediction["parsed"]
        row = {
            "sample_id": record["provenance"]["sample_id"],
            "parse_valid": parsed is not None,
            "service_s": elapsed,
            "predicted_length": int(pred_mask.sum()),
            "text": prediction["text"],
        }
        if sup["reference_known"]:
            row["length_absolute_error"] = abs(int(pred_mask.sum()) - int(mask.sum()))
            confidence = float(prediction["confidence_logits"].sigmoid())
            row["validity_brier"] = (confidence - float(sup["reference_valid"])) ** 2
            if mask.any():
                errors = np.linalg.norm(pred - gt, axis=-1)
                row["reference_point_error"] = float(errors[mask].mean())
                row["reference_endpoint_error"] = float(
                    np.linalg.norm(
                        (pred[pred_mask][-1] if pred_mask.any() else np.zeros(2))
                        - gt[mask][-1]
                    )
                )
                row["nonempty_reference"] = bool(pred_mask.any())
        if pred_mask.any() and obs.get("map_valid"):
            with np.load(obs["map_grid"], allow_pickle=False) as data:
                row["reference_geometric_feasible"] = collision_free(
                    pred[pred_mask],
                    data["grid"],
                    cfg.map_resolution,
                    cfg.footprint_radius,
                )
        if parsed and len(parsed["anchors"]) == 5 and pred_mask.any():
            anchors = np.array(parsed["anchors"])
            indices = np.rint(np.linspace(0, pred_mask.sum() - 1, 5)).astype(int)
            row["generated_anchor_consistency"] = float(
                np.linalg.norm(anchors - pred[indices], axis=-1).mean()
            )
        if recursive and parsed:
            states[key].commit_route(
                parsed["route"], obs["task"]["task_id"], obs["route_revision"]
            )
        rows.append(row)
        if position < 8:
            plot_prediction(
                record,
                np.empty((0, 2)),
                output / f"{position:03d}.png",
                pred[pred_mask],
            )
    report = summarize(rows)
    report.update(
        {
            "stage": "reference",
            "map_mode": map_mode,
            "recursive_route": recursive,
            "teacher_forced": False,
            "manifest_hash": fingerprint(dataset.manifest),
            "config": cfg.to_dict(),
        }
    )
    (output / "samples.json").write_text(json.dumps(rows, indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def evaluate_action(
    cfg,
    checkpoint,
    output,
    split="test",
    delay=0.0,
    condition="both",
    compensation="full",
    reference_mode="predicted",
    limit=None,
):
    model, ckpt = load_action(checkpoint, cfg.device)
    model.eval()
    data = CachedActionDataset(
        cfg,
        split,
        delay=delay,
        condition=condition,
        compensation=compensation,
        reference_mode=reference_mode,
    )
    if ckpt["reference_checkpoint_hash"] != data.cache_index["checkpoint_hash"]:
        raise ValueError("Action checkpoint trained with a different semantic model")
    for name in (
        "hidden_dim",
        "horizon",
        "action_dt",
        "num_reference_points",
        "coordinate_scale",
        "action_head",
    ):
        if ckpt["config"][name] != getattr(cfg, name):
            raise ValueError(f"Action evaluation config mismatch: {name}")
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    rows = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for i in range(len(data)):
        if limit is not None and i >= limit:
            break
        sample = data[i]
        batch = to_device(action_collate([sample]), cfg.device)
        generator = torch.Generator(device=cfg.device).manual_seed(cfg.seed + i)
        sync(cfg.device)
        start = time.perf_counter()
        prediction = (
            model.sample(batch, generator=generator)["trajectories"][0, 0, 0]
            .cpu()
            .numpy()
        )
        sync(cfg.device)
        elapsed = time.perf_counter() - start
        row = {
            "sample_id": sample["sample_id"],
            "age": sample["age"],
            "snapshot_valid": sample["snapshot_valid"],
            "action_s": elapsed,
            **trajectory_metrics(prediction, sample["record"], cfg),
        }
        ref = sample["reference_xy"][sample["reference_mask"]].numpy()
        if len(ref):
            row["reference_deviation"] = float(
                np.linalg.norm(prediction[:, None] - ref[None], axis=-1).min(-1).mean()
            )
        rows.append(row)
        np.savez_compressed(
            output / (sample["sample_id"] + ".npz"),
            prediction=prediction,
            reference=ref,
        )
        if i < 8:
            plot_prediction(sample["record"], prediction, output / f"{i:03d}.png", ref)
    report = summarize(rows)
    report.update(
        {
            "stage": "flow",
            "head": cfg.action_head,
            "requested_delay": delay,
            "condition": condition,
            "compensation": compensation,
            "reference_mode": reference_mode,
            "map_mode": data.cache_index["map_mode"],
            "oracle_is_privileged_ablation": reference_mode == "oracle",
            "config": cfg.to_dict(),
            "manifest_hash": fingerprint(data.records.manifest),
            "reference_hash": data.cache_index["checkpoint_hash"],
            "peak_cuda_bytes": torch.cuda.max_memory_allocated()
            if cfg.device.startswith("cuda")
            else None,
            "timing_scope": "action model only; cached VLM and data loading excluded",
            "collision_scope": "current observed occupancy, unknown treated as blocked; no future dynamic ground truth",
        }
    )
    (output / "samples.json").write_text(json.dumps(rows, indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


@torch.no_grad()
def evaluate_legacy(cfg, checkpoint, output, split="test", limit=None):
    """Original RGB-only interface, causal state inputs, separately labeled results."""
    from ticvla.models.ticvla import TICVLA

    model = TICVLA(
        model_path=cfg.model_path, action_horizon_steps=cfg.horizon, train_vlm=False
    ).to(cfg.device)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source = {
        k.removeprefix("model."): v for k, v in raw.get("state_dict", raw).items()
    }
    mapped = {}
    for key, value in model.state_dict().items():
        candidates = [key]
        if key.startswith("vlm_model.vlm."):
            candidates.append("vlm." + key[len("vlm_model.vlm.") :])
        if key.startswith("vlm."):
            candidates.append("vlm_model.vlm." + key[len("vlm.") :])
        for candidate in candidates:
            if candidate in source and source[candidate].shape == value.shape:
                mapped[key] = source[candidate]
                break
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Legacy checkpoint incompatible: {len(missing)} missing keys; use original horizon/dimensions"
        )
    model.eval()
    data = RecordDataset(cfg.manifest, split)
    rows = []
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for i in range(len(data)):
        if limit is not None and i >= limit:
            break
        record = data[i]
        obs = record["observation"]
        if not any(record["supervision"]["action_mask"]):
            continue
        # Legacy [vx,vy,yaw_rate,dx,dy] at zero delay, using observed state only.
        state = torch.tensor(
            [obs["robot_state"][0], 0.0, obs["robot_state"][1], 0.0, 0.0]
        )
        sync(cfg.device)
        start = time.perf_counter()
        _, waypoints, _ = model.predict(
            delayed_image_paths=[x["path"] for x in obs["rgb_history"]],
            current_image_path=obs["rgb_current"],
            instruction=obs["task"]["instruction"],
            robot_state=state,
            time_delay=0.0,
        )
        sync(cfg.device)
        pred = waypoints[0].float().cpu().numpy()
        rows.append(
            {
                "sample_id": record["provenance"]["sample_id"],
                "service_s": time.perf_counter() - start,
                **trajectory_metrics(pred, record, cfg),
            }
        )
        if i < 8:
            plot_prediction(record, pred, output / f"{i:03d}.png")
    report = {
        **summarize(rows),
        "stage": "legacy",
        "interface": "original RGB-only, causal robot state, zero delay",
        "comparison_note": "Map information differs; use the matched-condition Query control for architectural attribution",
        "manifest_hash": fingerprint(data.manifest),
        "config": cfg.to_dict(),
    }
    (output / "samples.json").write_text(json.dumps(rows, indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--stage", choices=["reference", "flow", "legacy"], required=True
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--map-mode", choices=["correct", "removed", "shuffled"], default="correct"
    )
    parser.add_argument("--recursive-route", action="store_true")
    parser.add_argument(
        "--condition", choices=["both", "reference", "semantic"], default="both"
    )
    parser.add_argument(
        "--compensation", choices=["full", "time", "none"], default="full"
    )
    parser.add_argument(
        "--reference-mode", choices=["predicted", "oracle"], default="predicted"
    )
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Run reference or action ablations; never retrain on test",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg.seed)
    if args.delay < 0 or (args.limit is not None and args.limit < 1):
        parser.error("delay must be nonnegative and limit positive")
    if args.stage == "legacy":
        reports = [
            evaluate_legacy(cfg, args.checkpoint, args.output, args.split, args.limit)
        ]
    elif args.stage == "reference":
        modes = (
            [
                ("correct", False),
                ("removed", False),
                ("shuffled", False),
                ("correct", True),
            ]
            if args.suite
            else [(args.map_mode, args.recursive_route)]
        )
        reports = [
            evaluate_reference(
                cfg,
                args.checkpoint,
                Path(args.output) / f"{m}-recursive-{r}",
                args.split,
                m,
                r,
                args.limit,
            )
            for m, r in modes
        ]
    else:
        settings = [
            (args.delay, args.condition, args.compensation, args.reference_mode)
        ]
        if args.suite:
            settings = [
                (d, "both", c, "predicted")
                for d in cfg.delay_seconds
                for c in ("none", "time", "full")
            ]
            settings += [
                (0.0, c, "full", "predicted") for c in ("reference", "semantic")
            ] + [(0.0, "both", "full", "oracle")]
        reports = [
            evaluate_action(
                cfg,
                args.checkpoint,
                Path(args.output) / f"delay-{d}-{c}-{p}-{r}",
                args.split,
                d,
                c,
                p,
                r,
                args.limit,
            )
            for d, c, p, r in settings
        ]
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "summary.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
