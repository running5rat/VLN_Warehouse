"""Replayable category-to-instance consumer for navigation semantics."""

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..geometry import transform_points
from ..schema import file_sha256
from .store import semantic_points
from .store import SemanticEvent


def reprojection_check(points, mask, depth, observation):
    """Source consistency only; no claim of independent absolute 3D accuracy."""
    if not len(points):
        return dict(passed=False, points=0)
    camera_to_odom = np.asarray(observation["T_odom_body"]) @ np.asarray(
        observation["T_body_camera"]
    )
    camera = transform_points(points, np.linalg.inv(camera_to_odom))
    K = np.asarray(observation["K"])
    pixels = camera @ K.T
    uv = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(int)
    height, width = depth.shape
    inside = (
        (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    indices = np.flatnonzero(inside)
    hits = mask[uv[indices, 1], uv[indices, 0]]
    residual = np.abs(camera[indices, 2] - depth[uv[indices, 1], uv[indices, 0]])
    fraction = float(np.count_nonzero(hits) / len(points))
    error = float(residual.max()) if len(residual) else None
    return dict(
        passed=bool(fraction == 1 and error is not None and error < 1e-4),
        points=len(points),
        source_mask_hit_fraction=fraction,
        max_source_depth_residual_m=error,
        scope="source reprojection self-consistency, not world-ground-truth accuracy",
    )


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    )
    temporary.replace(path)


def consume_categories(store, events, backend, output, *, timings=None):
    """Persist the inference result before map writes; replay never re-runs YOLO."""
    started = time.perf_counter()
    segmentation_s = 0.0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    canonical = {}
    for event in events:
        if event.kind != "object":
            continue
        category = " ".join(event.category.split()).casefold()
        canonical.setdefault(
            (event.image_id, category), replace(event, category=category)
        )
    events = list(canonical.values())
    if len({e.image_id for e in events}) > 1:
        raise ValueError("Single-frame consumer requires one source image")
    requests = [asdict(e) for e in events]
    snapshot_path = output / "detections.json"
    obs, depth, source_error = None, None, None
    if events:
        try:
            obs, depth = store.source_arrays(events[0].image_id)
        except (ValueError, OSError, KeyError) as exc:
            source_error = str(exc)
    source = (
        None
        if obs is None
        else {
            k: obs[k]
            for k in (
                "image_id",
                "rgb_hash",
                "depth_hash",
                "K",
                "T_body_camera",
                "T_odom_body",
                "t_obs",
                "odom_epoch",
                "submap_id",
            )
        }
    )
    if snapshot_path.exists():
        snapshot = json.loads(snapshot_path.read_text())
        saved_requests = [asdict(SemanticEvent(**row)) for row in snapshot["requests"]]
        if saved_requests != requests or snapshot["source"] != source:
            raise ValueError("Saved inference source/request mismatch")
    else:
        batch = dict(instances=[], provenance={}, possibly_truncated=False)
        if events and source_error is None:
            try:
                inference_start = time.perf_counter()
                try:
                    batch = backend.segment_instances(obs, [e.category for e in events])
                finally:
                    segmentation_s = time.perf_counter() - inference_start
            except (ValueError, OSError, RuntimeError) as exc:
                source_error = str(exc)
        rows = []
        for index, instance in enumerate(batch["instances"]):
            instance = dict(instance)
            mask_path = output / f"mask-{index:03d}.npz"
            np.savez_compressed(mask_path, mask=instance.pop("mask"))
            instance.update(
                instance_index=index,
                mask_file=mask_path.name,
                mask_sha256=file_sha256(mask_path),
            )
            rows.append(instance)
        snapshot = dict(
            requests=requests,
            source=source,
            instances=rows,
            provenance=batch["provenance"],
            possibly_truncated=batch["possibly_truncated"],
            error=source_error,
            texts=batch.get("texts", []),
            failures=batch.get("failures", []),
            calls=batch.get("calls", {}),
            timings=batch.get("timings", {}),
        )
        write_json(snapshot_path, snapshot)
    by_category = {e.category: e for e in events}
    applied = []
    for instance in snapshot["instances"]:
        index = instance["instance_index"]
        parent = by_category[instance["category"]]
        event = replace(
            parent,
            message_id=f"{parent.message_id}:instance:{index:03d}",
            roi=None,
            static_confirmed=False,
            text_evidence=instance.get("texts", []),
        )
        path = output / instance["mask_file"]
        if file_sha256(path) != instance["mask_sha256"]:
            raise ValueError("Saved instance mask changed")
        with np.load(path, allow_pickle=False) as arrays:
            mask = arrays["mask"]
        diagnostics, points = {}, np.empty((0, 3))
        projection_error = None
        reprojection = None
        try:
            if source_error is not None:
                raise ValueError(source_error)
            points = semantic_points(
                mask, depth, obs, store.config, diagnostics=diagnostics
            )
            reprojection = reprojection_check(points, mask, depth, obs)
        except (ValueError, OSError) as exc:
            projection_error = str(exc)
        provenance = dict(
            snapshot["provenance"],
            parent_message_id=parent.message_id,
            mask_sha256=instance["mask_sha256"],
            bbox=instance["bbox"],
            diagnostics=diagnostics,
            projection_error=projection_error,
        )
        result = store.consume(
            event, mask, instance["confidence"], segmentation=provenance
        )
        points_path = output / f"points-{index:03d}.npz"
        np.savez_compressed(points_path, points=points)
        landmark = (
            store.get("landmarks", result["landmark_uid"])
            if result["landmark_uid"]
            else None
        )
        applied.append(
            dict(
                instance,
                category_index=next(
                    i
                    for i, e in enumerate(events)
                    if e.category == instance["category"]
                ),
                event=asdict(event),
                result=result,
                diagnostics=diagnostics,
                reprojection=reprojection,
                points_file=points_path.name,
                landmark_status=landmark["status"] if landmark else None,
                identity_status=landmark.get("identity_status") if landmark else None,
                identity_text=landmark.get("identity_text") if landmark else None,
                needs_reobservation=landmark.get("needs_reobservation")
                if landmark
                else None,
                reobservation_reasons=landmark.get("reobservation_reasons", [])
                if landmark
                else [],
                potentially_dynamic=landmark.get("potentially_dynamic", False)
                if landmark
                else None,
            )
        )
    categories = []
    for event in events:
        children = [r for r in applied if r["category"] == event.category]
        failures = [
            r
            for r in snapshot.get("failures", [])
            if r.get("category") == event.category
        ]
        categories.append(
            dict(
                request=asdict(event),
                status="source_or_inference_error"
                if snapshot["error"]
                else (
                    ("partial_failure" if children else "perception_failure")
                    if failures
                    else ("processed" if children else "no_detection")
                ),
                reason=snapshot["error"]
                or (
                    "; ".join(dict.fromkeys(r["reason"] for r in failures))
                    if failures
                    else None
                ),
                instances=len(children),
                accepted=sum(r["result"]["status"] == "accepted" for r in children),
            )
        )
    report = dict(
        categories=categories,
        instances=applied,
        provenance=snapshot["provenance"],
        possibly_truncated=snapshot["possibly_truncated"],
        error=snapshot["error"],
        texts=snapshot.get("texts", []),
        failures=snapshot.get("failures", []),
        calls=snapshot.get("calls", {}),
        perception_timings=snapshot.get("timings", {}),
    )
    write_json(output / "processed.json", report)
    if timings is not None:
        timings.update(
            segmentation_s=segmentation_s,
            projection_and_persistence_s=time.perf_counter() - started - segmentation_s,
        )
    return report
