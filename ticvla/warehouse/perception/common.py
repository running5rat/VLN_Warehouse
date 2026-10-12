"""Coordinate checks and conservative text-to-physical-instance association."""

import unicodedata
import numpy as np
from PIL import Image, ImageDraw


def normalize_text(value):
    return " ".join(unicodedata.normalize("NFKC", value).split())


def polygon_mask(polygon, shape):
    points = np.asarray(polygon, dtype=float).reshape(-1, 2)
    h, w = shape
    if len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Invalid polygon")
    if (points < 0).any() or (points[:, 0] > w).any() or (points[:, 1] > h).any():
        raise ValueError("Polygon outside source RGB")
    canvas = Image.new("1", (w, h))
    ImageDraw.Draw(canvas).polygon([tuple(x) for x in points], fill=1)
    return np.array(canvas, dtype=bool)


def crop_box(box, shape, padding=0.08):
    h, w = shape
    x1, y1, x2, y2 = map(float, box)
    if not np.isfinite(box).all() or not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Invalid source bbox")
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [
        max(0, int(x1 - dx)),
        max(0, int(y1 - dy)),
        min(w, int(np.ceil(x2 + dx))),
        min(h, int(np.ceil(y2 + dy))),
    ]


def associate_texts(instances, texts, shape):
    """Smallest unambiguous physical carrier; never copy a read to all parents."""
    unique = []
    for text in texts:
        text = dict(text)
        text["normalized"] = normalize_text(text["text"])
        if not text["normalized"]:
            continue
        text_mask = polygon_mask(text["polygon"], shape)
        duplicate = next(
            (
                old
                for old in unique
                if old["normalized"] == text["normalized"]
                and (old["_mask"] & text_mask).sum()
                / max(1, (old["_mask"] | text_mask).sum())
                > 0.5
            ),
            None,
        )
        if duplicate:
            continue
        text["_mask"] = text_mask
        unique.append(text)
    for instance in instances:
        instance["texts"] = []
    results = []
    for text in unique:
        mask = text.pop("_mask")
        candidates = []
        for i, instance in enumerate(instances):
            coverage = float((mask & instance["mask"]).sum() / max(1, mask.sum()))
            if coverage >= 0.8:
                candidates.append((int(instance["mask"].sum()), i, coverage))
        candidates.sort()
        text["candidate_instances"] = [i for _, i, _ in candidates]
        text["owner_instance"] = None
        confidence = text.get("confidence")
        if confidence is not None and (not np.isfinite(confidence) or confidence < 0.8):
            text["status"] = "low_confidence"
        elif not candidates:
            text["status"] = "unassigned"
        else:
            area, owner, _ = candidates[0]
            ambiguous = False
            for other_area, other, _ in candidates[1:]:
                nesting = (
                    instances[owner]["mask"] & instances[other]["mask"]
                ).sum() / max(1, area)
                if other_area < area * 1.5 or nesting < 0.95:
                    ambiguous = True
            text["status"] = "ambiguous" if ambiguous else "bound"
            if not ambiguous:
                text["owner_instance"] = owner
                instances[owner]["texts"].append(dict(text))
        results.append(text)
    return results
