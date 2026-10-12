"""RGB-only proposal/YOLO inference, separated from the single map writer."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ticvla.warehouse.mapping import SemanticEvent
from ticvla.warehouse.mapping.markers import ShelfMarkerReader, planar_marker_mask
from ticvla.warehouse.mapping.segmentation import YOLOESegmenter
from ticvla.warehouse.mapping.store import semantic_points, overlap
from ticvla.warehouse.mapping.semantic_frame import write_json
from ticvla.warehouse.schema import file_sha256


def save_marker_predictions(result, output):
    """Durable inference snapshot; the main writer can replay without GPU/OCR."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    serial = dict(
        result,
        detections=[
            {k: v for k, v in d.items() if k != "rectified"}
            for d in result["detections"]
        ],
        predictions=[],
    )
    for i, prediction in enumerate(result["predictions"]):
        row = {k: v for k, v in prediction.items() if k not in ("mask", "raw_mask")}
        arrays = {
            k: prediction[k] for k in ("mask", "raw_mask") if prediction[k] is not None
        }
        path = output / f"mask-{i:03d}.npz"
        np.savez_compressed(path, **arrays)
        row.update(mask_file=path.name, mask_sha256=file_sha256(path))
        serial["predictions"].append(row)
    write_json(output / "predictions.json", serial)


def load_marker_predictions(output):
    output = Path(output)
    result = json.loads((output / "predictions.json").read_text())
    for row in result["predictions"]:
        path = output / row["mask_file"]
        if file_sha256(path) != row["mask_sha256"]:
            raise ValueError("Saved shelf sign mask changed")
        with np.load(path, allow_pickle=False) as arrays:
            row.update(
                {k: arrays[k] if k in arrays else None for k in ("mask", "raw_mask")}
            )
    return result


def infer_frame(obs, reader, backend):
    started = time.time()
    rgb = np.array(Image.open(obs["rgb"]).convert("RGB"))
    depth = np.load(obs["depth"], allow_pickle=False)
    detections = reader.detect(rgb)
    ocr_done = time.time()
    predictions = []
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection["roi"]
        if (x2 - x1) * (y2 - y1) < 1600:
            continue
        event = SemanticEvent(
            f"live-ocr:{obs['image_id']}:{i}",
            obs["image_id"],
            "shelf_marker",
            source="perception",
            roi=detection["roi"],
            name=detection["label"],
        )
        mask, confidence, provenance = backend.segment(obs, event)
        raw_mask = None if mask is None else mask.copy()
        provenance.update(
            ocr=reader.version,
            ocr_confidence=detection["ocr_confidence"],
            raw_ocr=detection["raw_ocr"],
            static_policy="installed fixed shelf placards",
            roi_source="RGB color proposal + perspective-rectified OCR",
        )
        shape_iou = 0.0
        if mask is not None:
            quad_mask = np.zeros(mask.shape, np.uint8)
            cv2.fillConvexPoly(quad_mask, np.int32(detection["quad"]), 1)
            shape_iou = float(
                (mask & (quad_mask > 0)).sum() / max((mask | (quad_mask > 0)).sum(), 1)
            )
            if shape_iou < 0.65:
                provenance["rejection"] = "YOLO mask disagrees with RGB sign outline"
                mask = None
            else:
                mask &= quad_mask > 0
                mask = planar_marker_mask(mask, depth, obs["K"])
                provenance["geometry_filter"] = (
                    "RGB sign outline intersection + dominant depth plane"
                )
        predictions.append(
            dict(
                event=asdict(event),
                mask=mask,
                raw_mask=raw_mask,
                confidence=confidence,
                provenance=provenance,
                mask_outline_iou=shape_iou,
            )
        )
    return dict(
        image_id=obs["image_id"],
        source_hashes={k: obs[k] for k in ("rgb_hash", "depth_hash")},
        t_obs=obs["t_obs"],
        started=started,
        finished=time.time(),
        ocr_seconds=ocr_done - started,
        predictions=predictions,
        detections=detections,
    )


def apply_predictions(store, result, root, *, publish=True):
    """Only the geometry/map process calls this; workers never write SQLite."""
    obs, depth = store.source_arrays(result["image_id"])
    if result.get("source_hashes") and any(
        obs[k] != v for k, v in result["source_hashes"].items()
    ):
        raise ValueError("Shelf sign source hashes changed")
    if any(p["event"]["image_id"] != result["image_id"] for p in result["predictions"]):
        raise ValueError("Shelf sign event/source mismatch")
    (root / "masks").mkdir(exist_ok=True)
    rgb = np.array(Image.open(obs["rgb"]).convert("RGB")) if publish else None
    rows = []
    for i, prediction in enumerate(result["predictions"]):
        event = SemanticEvent(**prediction["event"])
        mask, provenance = prediction["mask"], prediction["provenance"]
        if mask is not None:
            try:
                points = semantic_points(mask, depth, obs, store.config)
                event.static_confirmed = any(
                    lm["name"] == event.name
                    and event.image_id not in lm["observation_ids"]
                    and overlap(
                        points, store.points(lm), store.config.association_distance
                    )
                    >= store.config.association_overlap
                    for lm in store.records("landmarks")
                )
                provenance["static_confirmed_by_repeat"] = event.static_confirmed
            except ValueError:
                mask = None
        applied = store.consume(
            event, mask, prediction["confidence"], segmentation=provenance
        )
        row = dict(
            image_id=event.image_id,
            label=event.name,
            event_id=event.message_id,
            status=applied["status"],
            reason=applied["reason"],
            landmark_uid=applied["landmark_uid"],
            roi=event.roi,
            mask_outline_iou=prediction["mask_outline_iou"],
            confidence=prediction["confidence"],
            category=event.category,
            landmark_status=store.get("landmarks", applied["landmark_uid"])["status"]
            if applied["landmark_uid"]
            else None,
        )
        if mask is not None:
            stem = event.image_id.rsplit(":", 1)[-1]
            name = f"masks/{stem}-{i}.npz"
            np.savez_compressed(
                root / name, mask=mask, raw_yolo_mask=prediction["raw_mask"]
            )
            row["mask_file"] = name
            if publish:
                rgb[mask] = (rgb[mask] * 0.45 + np.array([30, 240, 100]) * 0.55).astype(
                    np.uint8
                )
        rows.append(row)
    if not publish:
        return rows
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    for d in result["detections"]:
        draw.rectangle(d["roi"], outline="yellow", width=2)
        draw.text(
            (d["roi"][0], d["roi"][1] - 12),
            d["label"],
            fill="yellow",
            stroke_width=1,
            stroke_fill="black",
        )
    temp = root / "semantic.tmp.png"
    if publish:
        canvas.save(temp)
        temp.replace(root / "semantic.png")
    return rows


def semantic_worker(requests, results, weights, host_tesseract):
    """One outstanding request: no unbounded GPU queue or stale event dropping."""
    import traceback

    try:
        import torch

        torch.set_num_threads(2)
        cv2.setNumThreads(1)
        reader = ShelfMarkerReader(host_tesseract=host_tesseract)
        backend = YOLOESegmenter(weights)
        results.put(
            {"ready": True, "backend": backend.provenance, "ocr": reader.version}
        )
        while True:
            obs = requests.get()
            if obs is None:
                return
            results.put(infer_frame(obs, reader, backend))
    except BaseException:
        results.put({"error": traceback.format_exc()})
