"""Source-aligned detection/commit views with explicit provenance and outcomes."""

from pathlib import Path
from io import BytesIO
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ticvla.warehouse.mapping.semantic_frame import write_json


def publish_inspection(root, source, result, processed, marker_rows=(), markers=None):
    root, out = Path(root), Path(result["directory"])
    rgb = np.array(Image.open(source["rgb"]).convert("RGB"))
    rows, masks = [], []
    for row in processed["instances"]:
        with np.load(
            out / "segmentation" / row["mask_file"], allow_pickle=False
        ) as arrays:
            mask = arrays["mask"]
        rows.append(
            dict(
                instance_id=f"L{row['instance_index']}",
                category=row["category"],
                origin=("LLM -> " + processed["provenance"]["backend"])
                if processed["provenance"].get("backend")
                in ("yoloe-ppocr", "florence2")
                else "LLM text -> YOLO",
                confidence=row["confidence"],
                bbox=row["bbox"],
                status=row["result"]["status"],
                reason=row["result"]["reason"],
                landmark_uid=row["result"]["landmark_uid"],
                landmark_status=row["landmark_status"],
                name=row.get("identity_text"),
                identity_status=row.get("identity_status"),
                texts=row.get("texts", []),
                needs_reobservation=row.get("needs_reobservation"),
                reobservation_reasons=row.get("reobservation_reasons", []),
            )
        )
        masks.append(mask)
    for i, row in enumerate(marker_rows):
        prediction = markers["predictions"][i]
        rows.append(
            dict(
                instance_id=f"S{i}",
                category="shelf_marker",
                origin="RGB OCR -> YOLO ROI",
                name=row["label"],
                bbox=row["roi"],
                confidence=row["confidence"],
                status=row["status"],
                reason=row["reason"],
                landmark_uid=row["landmark_uid"],
                landmark_status=row["landmark_status"],
            )
        )
        masks.append(
            prediction["mask"]
            if prediction["mask"] is not None
            else prediction["raw_mask"]
        )
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    palette = [
        (10, 170, 240),
        (230, 110, 10),
        (130, 80, 230),
        (20, 180, 100),
        (220, 60, 160),
    ]

    def render(accepted_only):
        blended = rgb.copy()
        for i, (row, mask) in enumerate(zip(rows, masks)):
            if accepted_only and row["status"] != "accepted":
                continue
            if mask is not None and mask.shape == rgb.shape[:2]:
                mask = mask.astype(bool)
                blended[mask] = (
                    blended[mask] * 0.60 + np.array(palette[i % len(palette)]) * 0.40
                ).astype(np.uint8)
        canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + 60), "#13212e")
        canvas.paste(Image.fromarray(blended), (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 5),
            "IN MAP: accepted masks + boxes"
            if accepted_only
            else "Perception: all detected masks + boxes",
            font=font,
            fill="white",
        )
        draw.text(
            (10, 29),
            source["image_id"] + " | source RGB (may lag robot)",
            font=font,
            fill="white",
        )
        for i, row in enumerate(rows):
            if accepted_only and row["status"] != "accepted":
                continue
            x1, y1, x2, y2 = row["bbox"]
            color = (
                palette[i % len(palette)]
                if row["status"] == "accepted"
                else (240, 40, 50)
            )
            draw.rectangle((x1, y1 + 60, x2, y2 + 60), outline=color, width=3)
            score = (
                f"{row['confidence']:.2f}" if row["confidence"] is not None else "N/A"
            )
            label = f"{row['instance_id']} {row['name'] or row['category']} {score}"
            status = (
                row["landmark_status"] if row["status"] == "accepted" else row["status"]
            )
            label += f" | {status} | {row.get('identity_status') or ''}"
            for txt in row.get("texts", []):
                poly = [(p[0], p[1] + 60) for p in txt["polygon"]]
                draw.line(poly + poly[:1], fill="yellow", width=2)
                draw.text(poly[0], txt["text"], font=font, fill="yellow")
            box = draw.textbbox((0, 0), label, font=font)
            x = max(0, min(int(x1), canvas.width - (box[2] - box[0]) - 8))
            y = max(60, min(int(y1) + 60 - 23, canvas.height - 23))
            draw.rectangle((x, y, x + box[2] - box[0] + 8, y + 23), fill="black")
            draw.text((x + 3, y), label, font=font, fill=color)
        if not accepted_only:
            for txt in processed.get("texts", []):
                if txt.get("status") == "bound":
                    continue
                poly = [(p[0], p[1] + 60) for p in txt["polygon"]]
                draw.line(poly + poly[:1], fill="yellow", width=2)
                draw.text(
                    poly[0],
                    txt["text"] + " | " + txt["status"],
                    font=font,
                    fill="yellow",
                )
        return canvas

    detection, committed = render(False), render(True)
    for name, canvas in (("detections.png", detection), ("semantic.png", committed)):
        encoded = BytesIO()
        canvas.save(encoded, format="PNG", compress_level=1)
        data = encoded.getvalue()
        (out / name).write_bytes(data)
        if name == "semantic.png":
            (out / "segmentation.png").write_bytes(data)
        temp = root / (name + ".tmp.png")
        temp.write_bytes(data)
        temp.replace(root / name)
    report = dict(
        updated_wall=time.time(),
        image_id=source["image_id"],
        t_obs=source["t_obs"],
        raw_answer=result.get("raw_answer", ""),
        llm_valid=result["llm_valid"],
        llm_error=result.get("llm_error"),
        categories=processed["categories"],
        filtered_categories=result.get("filtered_categories", []),
        instances=rows,
        texts=processed.get("texts", []),
        perception_failures=processed.get("failures", []),
        note="Perception categories are prompted by the VLM, not independent class agreement. Accepted means written, not verified static.",
    )
    write_json(out / "inspection.json", report)
    # Commit marker written after both images; UI reloads by this version.
    write_json(root / "inspection.json", report)
    return report
