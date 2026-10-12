"""Replay RGB-D through OCR -> real YOLO-E -> persistent map, then evaluate.

Evaluation-only catalog/instance buffers never enter perception or map updates.
"""

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ticvla.warehouse.geometry import transform_points
from ticvla.warehouse.mapping import MapStore, MappingConfig, SemanticEvent
from ticvla.warehouse.mapping.cli import import_episode, export_view
from ticvla.warehouse.mapping.markers import ShelfMarkerReader, planar_marker_mask
from ticvla.warehouse.mapping.segmentation import YOLOESegmenter
from ticvla.warehouse.mapping.store import semantic_points, voxelize, overlap
from ticvla.warehouse.schema import file_sha256


def map_observations(capture, root, weights, host_tesseract):
    """Deployment path: reads raw RGB-D/odom only, never supervision/catalog."""
    store = MapStore(root / "map", MappingConfig(fixed_shelf_markers=True))
    ids = import_episode(store, capture / "raw/episode.json")
    reader = ShelfMarkerReader(host_tesseract=host_tesseract)
    backend = YOLOESegmenter(weights)
    (root / "masks").mkdir()
    (root / "overlays").mkdir()
    rows = []
    for index, image_id in enumerate(ids):
        obs, depth = store.source_arrays(image_id)
        rgb = np.array(Image.open(obs["rgb"]).convert("RGB"))
        overlay = rgb.copy()
        detections = reader.detect(rgb)
        for di, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["roi"]
            if (x2 - x1) * (y2 - y1) < 1600:
                rows.append(
                    {
                        "frame": index,
                        "label": detection["label"],
                        "status": "rejected",
                        "reason": "Small OCR proposal",
                        "roi": detection["roi"],
                    }
                )
                continue
            event = SemanticEvent(
                f"ocr-{index}-{di}",
                image_id,
                "shelf_marker",
                source="perception",
                roi=detection["roi"],
                name=detection["label"],
                static_confirmed=False,
            )
            mask, confidence, provenance = backend.segment(obs, event)
            raw_yolo_mask = None if mask is None else mask.copy()
            provenance.update(
                ocr=reader.version,
                ocr_confidence=detection["ocr_confidence"],
                static_policy="installed fixed shelf placards",
                roi_source="RGB color proposal + perspective-rectified OCR",
            )
            shape_iou = 0.0
            if mask is not None:
                quad_mask = np.zeros(mask.shape, np.uint8)
                cv2.fillConvexPoly(quad_mask, np.int32(detection["quad"]), 1)
                shape_iou = float(
                    (mask & (quad_mask > 0)).sum()
                    / max((mask | (quad_mask > 0)).sum(), 1)
                )
                if shape_iou < 0.65:
                    provenance["rejection"] = (
                        "YOLO mask disagrees with RGB sign outline"
                    )
                    mask = None
                else:
                    # A placard's OCR outline also excludes mounting hardware
                    # and nearby surfaces, even when they share its depth plane.
                    mask &= quad_mask > 0
                    provenance["geometry_filter"] = (
                        "RGB sign outline intersection + dominant depth plane"
                    )
            provenance["raw_ocr"] = detection["raw_ocr"]
            if mask is not None:
                mask = planar_marker_mask(mask, depth, obs["K"])
                if mask is not None:
                    try:
                        points = semantic_points(mask, depth, obs, store.config)
                        # Only repeated independent image evidence with matching
                        # OCR and overlapping 3D geometry can establish staticness.
                        event.static_confirmed = any(
                            lm["name"] == event.name
                            and event.image_id not in lm["observation_ids"]
                            and overlap(
                                points,
                                store.points(lm),
                                store.config.association_distance,
                            )
                            >= store.config.association_overlap
                            for lm in store.records("landmarks")
                        )
                        provenance["static_confirmed_by_repeat"] = (
                            event.static_confirmed
                        )
                    except ValueError:
                        mask = None
            result = store.consume(event, mask, confidence, segmentation=provenance)
            row = {
                "frame": index,
                "label": event.name,
                "event_id": event.message_id,
                "status": result["status"],
                "reason": result["reason"],
                "landmark_uid": result["landmark_uid"],
                "roi": event.roi,
                "mask_outline_iou": shape_iou,
            }
            if mask is not None:
                mask_file = f"masks/{index:08d}-{di}.npz"
                np.savez_compressed(
                    root / mask_file, mask=mask, raw_yolo_mask=raw_yolo_mask
                )
                row["mask_file"] = mask_file
                overlay[mask] = (
                    overlay[mask] * 0.45 + np.array([30, 240, 100]) * 0.55
                ).astype(np.uint8)
            rows.append(row)
        canvas = Image.fromarray(overlay)
        draw = ImageDraw.Draw(canvas)
        for d in detections:
            draw.rectangle(d["roi"], outline="yellow", width=2)
            draw.text(
                (d["roi"][0], d["roi"][1] - 12),
                d["label"],
                fill="yellow",
                stroke_width=1,
                stroke_fill="black",
            )
        canvas.save(root / "overlays" / f"{index:08d}.png")
        if index % 10 == 0:
            print(
                "MAP_FRAME",
                index,
                "landmarks",
                len(store.records("landmarks")),
                flush=True,
            )
    (root / "events.json").write_text(json.dumps(rows, indent=2))
    summary = store.summary()
    version = store.version
    accepted = next((row for row in rows if row["status"] == "accepted"), None)
    duplicate_pass = False
    if accepted:
        old = store.get("events", accepted["event_id"])
        duplicate = store.consume(SemanticEvent(**old["event"]))
        duplicate_pass = duplicate["duplicate"] and version == store.version
    negative = store.consume(
        SemanticEvent(
            "missing-source-negative", "absent", "shelf_marker", source="perception"
        )
    )
    store.close()
    store = MapStore(root / "map")
    # Negative events change event counts but cannot modify map version/geometry.
    persisted = (
        store.version == version
        and store.summary()["landmarks"] == summary["landmarks"]
    )
    return (
        store,
        ids,
        rows,
        {
            "segmenter": backend.provenance,
            "ocr": reader.version,
            "duplicate_pass": duplicate_pass,
            "reload_pass": persisted,
            "invalid_source_rejected": negative["status"] == "rejected",
        },
    )


def evaluate(capture, root, store, ids, rows):
    """Privileged data first enters here, AFTER all map decisions have completed."""
    catalog = json.loads((capture / "shelf-catalog.json").read_text())
    origin = np.array(
        json.loads((capture / "evaluation-origin.json").read_text())["T_world_odom"]
    )
    by_label = {r["label"]: r for r in catalog}
    assessments, candidate_assessments = [], []
    landmarks = [lm for lm in store.records("landmarks") if lm["status"] == "verified"]
    verified_ids = {lm["landmark_uid"] for lm in landmarks}
    visible = set()
    for index, image_id in enumerate(ids):
        with np.load(
            capture / "raw/supervision" / f"{index:08d}.npz", allow_pickle=False
        ) as data:
            truth_ids = data["instance_ids"].copy()
            paths = json.loads(bytes(data["instance_labels_json"]).decode())
        truth = {}
        for spec in catalog:
            instance_ids = [
                int(k)
                for k, v in paths.items()
                if str(v).startswith(spec["prim"] + "/")
            ]
            if not instance_ids:
                continue
            mask = np.isin(truth_ids, instance_ids)
            if mask.sum() >= 1600:
                visible.add(spec["label"])
            if mask.any():
                truth[spec["label"]] = mask
        obs, depth = store.source_arrays(image_id)
        for row in (
            r for r in rows if r["frame"] == index and r["status"] == "accepted"
        ):
            arrays = np.load(root / row["mask_file"], allow_pickle=False)
            mask = arrays["mask"]
            raw = arrays["raw_yolo_mask"]
            ious = {
                label: float((mask & m).sum() / max((mask | m).sum(), 1))
                for label, m in truth.items()
            }
            actual = max(ious, key=ious.get) if ious else None
            points = semantic_points(mask, depth, obs, store.config)
            world_points = transform_points(points, origin)
            spec = by_label.get(row["label"])
            error = surface_p95 = None
            if spec:
                center, half = (
                    np.array(spec["center_world"]),
                    np.array(spec["size_m"]) / 2,
                )
                lo, hi = center - half, center + half
                outside = np.linalg.norm(
                    np.maximum(np.maximum(lo - world_points, world_points - hi), 0),
                    axis=1,
                )
                inside = np.min(
                    np.minimum(world_points - lo, hi - world_points), axis=1
                )
                surface_p95 = float(
                    np.percentile(np.where(outside > 0, outside, inside), 95)
                )
                error = float(np.linalg.norm(world_points.mean(0) - center))
            target_assessments = (
                assessments
                if row["landmark_uid"] in verified_ids
                else candidate_assessments
            )
            target_assessments.append(
                {
                    "frame": index,
                    "predicted": row["label"],
                    "actual": actual,
                    "identity_correct": actual == row["label"],
                    "mask_iou": ious.get(row["label"], 0),
                    "raw_yolo_mask_iou": float(
                        (raw & truth[row["label"]]).sum()
                        / max((raw | truth[row["label"]]).sum(), 1)
                    )
                    if row["label"] in truth
                    else 0.0,
                    "surface_p95_error_m": surface_p95,
                    "centroid_error_m": error,
                }
            )
    landmark_errors = []
    for lm in landmarks:
        spec = by_label.get(lm["name"])
        position = transform_points(np.array(lm["position"])[None], origin)[0]
        landmark_errors.append(
            {
                "label": lm["name"],
                "observation_count": len(lm["observation_ids"]),
                "position_world": position.tolist(),
                "centroid_error_m": float(
                    np.linalg.norm(position - np.array(spec["center_world"]))
                )
                if spec
                else None,
            }
        )
    required = {f"{i:02d}/{r}R" for r in (4, 5) for i in range(1, 5)}
    mapped = {lm["name"] for lm in landmarks}
    counts = Counter(lm["name"] for lm in landmarks)
    # Declared gates: 8 route-adjacent plaques, no false identity or duplicates,
    # <= one 5 cm voxel surface error, >= .75 mask IoU, <=15cm centroid error.
    gates = {
        "required_eight_markers": required <= mapped,
        "identity_precision": bool(assessments)
        and all(r["identity_correct"] for r in assessments),
        "mask_quality": bool(assessments)
        and all(r["mask_iou"] >= 0.75 for r in assessments),
        "projection_one_voxel": bool(assessments)
        and all(
            r["surface_p95_error_m"] is not None and r["surface_p95_error_m"] <= 0.05
            for r in assessments
        ),
        "landmark_position": bool(landmark_errors)
        and all(
            r["centroid_error_m"] is not None and r["centroid_error_m"] <= 0.15
            for r in landmark_errors
        ),
        "no_duplicate_identity": all(count == 1 for count in counts.values()),
        "repeat_evidence": bool(landmarks)
        and all(len(lm["observation_ids"]) >= 2 for lm in landmarks),
        "loop_revisit_association": any(
            lm["first_seen"] < store.get("observations", ids[0])["t_obs"] + 5
            and lm["last_seen"] > store.get("observations", ids[-1])["t_obs"] - 5
            for lm in landmarks
        ),
    }
    frames = [store.get("observations", image_id) for image_id in ids]
    trace = [
        json.loads(line)
        for line in (capture / "trajectory.jsonl").read_text().splitlines()
    ]
    times = np.array([frame["t_obs"] for frame in frames])
    trace_times = np.array([row["t"] for row in trace])
    trace_positions = np.array([row["position"] for row in trace])
    camera_positions = np.array(
        [(origin @ np.array(frame["T_odom_body"]))[:2, 3] for frame in frames]
    )
    interpolated = np.column_stack(
        [np.interp(times, trace_times, trace_positions[:, i]) for i in (0, 1)]
    )
    sync_error = float(np.linalg.norm(camera_positions - interpolated, axis=1).max())
    gates["source_time_and_pose"] = bool(
        np.all(np.diff(times) > 0) and sync_error <= 0.05
    )
    return {
        "gates": gates,
        "required_labels": sorted(required),
        "mapped_labels": sorted(mapped),
        "visible_large_labels": sorted(visible),
        "unmapped_labels": sorted(set(by_label) - mapped),
        "instances": assessments,
        "unconfirmed_instances": candidate_assessments,
        "unconfirmed_labels": [
            lm["name"]
            for lm in store.records("landmarks")
            if lm["status"] != "verified"
        ],
        "landmarks": landmark_errors,
        "max_source_pose_vs_physics_trace_error_m": sync_error,
        "source_duration_s": float(times[-1] - times[0]),
    }


def artifacts(capture, root, store, ids):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clouds = [store.points(obs) for obs in store.records("observations")]
    points = voxelize(np.concatenate(clouds), 0.08)
    body_path = np.array(
        [np.array(store.get("observations", i)["T_odom_body"])[:3, 3] for i in ids]
    )
    # World origin has minor simulated chassis tilt; use map Z only for display.
    obstacles = points[(points[:, 2] > 0.15) & (points[:, 2] < 3.5)]
    np.savez_compressed(root / "global-map.npz", points=points, trajectory=body_path)
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.scatter(
        obstacles[:, 0],
        obstacles[:, 1],
        s=0.2,
        c=obstacles[:, 2],
        cmap="Greys",
        alpha=0.35,
        rasterized=True,
    )
    ax.plot(
        body_path[:, 0],
        body_path[:, 1],
        color="royalblue",
        lw=2,
        label="RGB-D source trajectory",
    )
    ax.scatter(*body_path[0, :2], s=80, c="green", marker="o", label="Start")
    ax.scatter(*body_path[-1, :2], s=80, c="red", marker="x", label="Finish")
    for lm in store.records("landmarks"):
        if lm["status"] != "verified":
            continue
        x, y, _ = lm["position"]
        ax.scatter(x, y, s=32, c="darkorange")
        ax.annotate(
            lm["name"], (x, y), xytext=(3, 5), textcoords="offset points", fontsize=9
        )
    ax.set(
        xlabel="Odom X (m)",
        ylabel="Odom Y (m)",
        title="Observed RGB-D geometry + OCR / YOLO-E shelf markers",
    )
    ax.axis("equal")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "semantic-map.png", dpi=160)
    plt.close(fig)
    export_view(store, ids[-1], root / "final-mapview")
    for name, folder in (
        ("mapping", root / "overlays"),
        ("overview", capture / "overview"),
    ):
        files = sorted(folder.glob("*.png"))
        if not files:
            continue
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                "6",
                "-i",
                str(folder / "%08d.png"),
                "-c:v",
                "libx264",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(root / f"{name}.mp4"),
            ],
            check=True,
        )
    return len(points), len(obstacles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", default="/data/ticvla/models/yoloe-v8l-seg.pt")
    parser.add_argument("--host-tesseract", action="store_true")
    args = parser.parse_args()
    capture, root = Path(args.capture).resolve(), Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    store, ids, rows, engineering = map_observations(
        capture, root, args.weights, args.host_tesseract
    )
    try:
        evaluation = evaluate(capture, root, store, ids, rows)
        point_count, obstacle_count = artifacts(capture, root, store, ids)
        report = {
            "capture": json.loads((capture / "capture-report.json").read_text()),
            "engineering": engineering,
            "evaluation": evaluation,
            "geometry_points": point_count,
            "obstacle_points": obstacle_count,
            "map_summary": store.summary(),
            "script_sha256": file_sha256(__file__),
        }
        report["pass"] = (
            report["capture"]["loop_complete"]
            and all(evaluation["gates"].values())
            and all(
                engineering[k]
                for k in ("duplicate_pass", "reload_pass", "invalid_source_rejected")
            )
        )
        (root / "report.json").write_text(json.dumps(report, indent=2))
        print(
            "VALIDATION",
            json.dumps(
                {
                    "pass": report["pass"],
                    "gates": evaluation["gates"],
                    "mapped": evaluation["mapped_labels"],
                }
            ),
            flush=True,
        )
        if not report["pass"]:
            raise SystemExit(2)
    finally:
        store.close()


if __name__ == "__main__":
    main()
