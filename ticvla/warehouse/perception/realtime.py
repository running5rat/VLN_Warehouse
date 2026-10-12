"""Fixed vocabulary perception. Selection, masks and late text have separate provenance."""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..schema import fingerprint, file_sha256
from ..mapping import SemanticEvent
from ..mapping.semantic_frame import write_json
from .common import associate_texts

CORE = (
    "sign",
    "label",
    "storage rack",
    "door",
    "pillar",
    "fire extinguisher",
    "electrical cabinet",
)
VOCAB_VERSION = "warehouse-static-candidates-v1"
ALIASES = {c: c for c in CORE}
ALIASES.update(
    dict(
        zip(
            [
                "signs",
                "labels",
                "storage racks",
                "doors",
                "pillars",
                "fire extinguishers",
                "electrical cabinets",
            ],
            CORE,
        )
    )
)
ALIASES.update(
    {
        "shelf": "storage rack",
        "shelves": "storage rack",
        "rack": "storage rack",
        "racks": "storage rack",
        "column": "pillar",
        "columns": "pillar",
        "shelf sign": "sign",
        "shelf signs": "sign",
    }
)


def selection_question(task):
    return (
        "<image>\nTask: "
        + task
        + "\nSelect visible object categories useful as long-term location references. "
        "Look carefully for small signs and labels as well as large structures. Core vocabulary: "
        + ", ".join(CORE)
        + ". Do not read any text or number. "
        "Reply only with a JSON object containing selected_categories (core names only) and "
        "proposed_categories (visible useful categories absent from the core vocabulary). "
        "Both fields are lists of short English category names. Empty lists are allowed. "
        "Exclude people, vehicles, carts, loose boxes and plain background surfaces. "
        "No explanation. Category selection is not evidence of permanent staticness."
    )


def parse_selection(text):
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(value)
    if not isinstance(data, dict) or set(data) != {
        "selected_categories",
        "proposed_categories",
    }:
        raise ValueError(
            "Expected selected_categories and proposed_categories JSON lists"
        )
    out = {}
    for field in data:
        if not isinstance(data[field], list) or any(
            not isinstance(x, str) or not x.strip() or len(x) > 80 for x in data[field]
        ):
            raise ValueError("Invalid category list")
        values = [" ".join(x.casefold().split()) for x in data[field]]
        out[field] = list(dict.fromkeys(ALIASES.get(x, x) for x in values))
    if any(x not in CORE for x in out["selected_categories"]):
        raise ValueError("Selected category is outside frozen vocabulary")
    out["proposed_categories"] = [
        x for x in out["proposed_categories"] if x not in CORE
    ]
    return out


class SelectionState:
    def __init__(self):
        self.latest = None

    def update(self, result):
        if result["valid"] and (
            self.latest is None or result["source_time"] >= self.latest["source_time"]
        ):
            self.latest = result

    def snapshot(self, source_time):
        if (
            self.latest is None
            or not 0 <= source_time - self.latest["source_time"] <= 5
        ):
            return dict(
                selected_categories=[],
                proposed_categories=[],
                status="expired",
                request_id=None,
            )
        return {
            **self.latest["parsed"],
            "status": "valid",
            "request_id": self.latest["request_id"],
            "source_image_id": self.latest["source_image_id"],
            "source_time": self.latest["source_time"],
        }


class Selector:
    def __init__(self, model="/data/ticvla/models/InternVL3-1B", device="cuda:0"):
        from ..structured_io import StructuredVLM
        from ..config import WarehouseConfig

        self.model = StructuredVLM(
            WarehouseConfig(model_path=model, vlm_checkpoint=None, device=device)
        )

    def predict(self, source, task):
        from ..object_io import object_observation

        obs = object_observation(source, "navigation", task)
        obs["core_vocabulary"] = list(CORE)
        start = time.time()
        result = dict(
            request_id=fingerprint(obs)[:24],
            source_image_id=source["image_id"],
            source_time=source["t_obs"],
            rgb_hash=source["rgb_hash"],
            vocabulary_version=VOCAB_VERSION,
            valid=False,
            parsed=None,
            error=None,
            llm_calls=1,
            structured_inference_executed=False,
        )
        try:
            raw = self.model.describe_objects(obs)
            result.update(raw=raw, text=raw["answer"], prompt=raw["question"])
            if raw["token_count"] >= raw["max_new_tokens"]:
                raise ValueError("Selection answer truncated")
            result.update(parsed=parse_selection(raw["answer"]), valid=True)
        except (ValueError, RuntimeError) as exc:
            result["error"] = str(exc)
        result.update(
            started=start, finished=time.time(), elapsed_s=time.time() - start
        )
        return result


class FixedSegmenter:
    def __init__(self, weights, device="cuda:0"):
        self.weights = str(weights)
        self.engine = Path(weights).suffix == ".engine"
        if self.engine:
            from ultralytics import YOLO

            config = json.loads(Path(str(weights) + ".json").read_text())
            if config["categories"] != list(CORE) or config["sha256"] != file_sha256(
                weights
            ):
                raise ValueError("Engine vocabulary/hash mismatch")
            self.model = YOLO(weights, task="segment")
            self.provenance = {
                **config,
                "runtime": "tensorrt-fp16",
                "backend": "yoloe-realtime",
            }
        else:
            from ..mapping.segmentation import YOLOESegmenter

            self.model = YOLOESegmenter(weights, device, cache_classes=True, half=True)
            self.provenance = {
                **self.model.provenance,
                "backend": "yoloe-realtime",
                "runtime": "pytorch-fp16",
                "categories": list(CORE),
            }
        self.device = device

    def predict(self, source):
        start = time.time()
        if file_sha256(source["rgb"]) != source["rgb_hash"]:
            raise ValueError("Segmentation source hash mismatch")
        if not self.engine:
            data = self.model.segment_instances(source, list(CORE))
            self.provenance.update(
                self.model.provenance, backend="yoloe-realtime", runtime="pytorch-fp16"
            )
        else:
            r = self.model.predict(
                source["rgb"],
                device=self.device,
                imgsz=640,
                conf=0.25,
                max_det=100,
                retina_masks=True,
                verbose=False,
                save=False,
            )[0]
            if [r.names[i] for i in range(len(CORE))] != list(CORE):
                raise ValueError("Runtime engine class mapping mismatch")
            instances = []
            with Image.open(source["rgb"]) as im:
                shape = (im.height, im.width)
            if r.masks is not None:
                for mask, box, cls, conf in zip(
                    r.masks.data.cpu().numpy(),
                    r.boxes.xyxy.cpu().numpy(),
                    r.boxes.cls.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                ):
                    if mask.shape != shape:
                        raise ValueError("Mask/source shape mismatch")
                    instances.append(
                        dict(
                            category=CORE[int(cls)],
                            confidence=float(conf),
                            bbox=box.tolist(),
                            mask=mask > 0.5,
                        )
                    )
            instances.sort(key=lambda x: (x["category"], *x["bbox"], -x["confidence"]))
            data = dict(instances=instances, possibly_truncated=len(instances) >= 100)
        data.update(
            provenance=dict(self.provenance),
            elapsed_s=time.time() - start,
            source_image_id=source["image_id"],
            rgb_hash=source["rgb_hash"],
        )
        return data


class MobileReader:
    def __init__(self):
        from paddleocr import PaddleOCR
        from .worker import hashes

        self.model = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            cpu_threads=4,
            enable_mkldnn=True,
            text_det_limit_side_len=960,
            text_det_limit_type="max",
            text_recognition_batch_size=16,
            text_rec_score_thresh=0.0,
        )
        self.provenance = dict(
            backend="PP-OCRv5_mobile",
            device="cpu",
            threads=4,
            det_max_side=960,
            mkldnn=True,
            files=hashes(Path.home() / ".paddlex/official_models/PP-OCRv5_mobile_det")
            | {
                "rec/" + k: v
                for k, v in hashes(
                    Path.home() / ".paddlex/official_models/PP-OCRv5_mobile_rec"
                ).items()
            },
        )

    def predict(self, source):
        if file_sha256(source["rgb"]) != source["rgb_hash"]:
            raise ValueError("OCR source hash mismatch")
        start = time.time()
        # Paddle's pipeline detects once, crops original-resolution RGB and batches recognition.
        result = list(self.model.predict(source["rgb"]))[0].json["res"]
        texts = [
            dict(text=t, confidence=float(c), polygon=np.asarray(p).tolist())
            for t, c, p in zip(
                result["rec_texts"], result["rec_scores"], result["rec_polys"]
            )
        ]
        return dict(
            texts=texts,
            raw=result,
            elapsed_s=time.time() - start,
            provenance=self.provenance,
            source_image_id=source["image_id"],
            rgb_hash=source["rgb_hash"],
        )


def consume_masks(store, source, batch, selection):
    if (
        batch["source_image_id"] != source["image_id"]
        or batch["rgb_hash"] != source["rgb_hash"]
    ):
        raise ValueError("Mask result source mismatch")
    instance_mode = selection.get("selection_mode") == "instances"
    if instance_mode:
        from .instance_selection import validate_selection
        validate_selection(source, batch, selection)
    ids = {v["instance_id"] for v in selection.get("selected_instances", [])}
    rows = []
    for i, inst in enumerate(batch["instances"]):
        selected = (f"d{i}" in ids if instance_mode else inst["category"] in selection["selected_categories"])
        event = SemanticEvent(
            f"rt:{source['image_id']}:{i}",
            source["image_id"],
            inst["category"],
            source="llm",
            request_id=selection.get("request_id"),
        )
        result = (
            store.consume(
                event,
                inst["mask"],
                inst["confidence"],
                segmentation={**batch["provenance"], "category_selection": selection},
                prepared_projection=batch.get("prepared_projections", {}).get(i),
            )
            if selected
            else dict(
                status="not_selected",
                reason="VLM did not select source instance" if instance_mode else "LLM did not select category",
                landmark_uid=None,
            )
        )
        rows.append(
            dict(
                instance_index=i,
                category=inst["category"],
                bbox=inst["bbox"],
                confidence=inst["confidence"],
                selected=selected,
                parent_event_id=event.message_id,
                result=result,
                geometry_file=str(
                    store.root
                    / store.get("landmarks", result["landmark_uid"])["geometry"]
                )
                if result.get("landmark_uid")
                else None,
            )
        )
    return rows


def consume_text(store, source, batch, rows, ocr):
    if (
        ocr["source_image_id"] != source["image_id"]
        or ocr["rgb_hash"] != source["rgb_hash"]
    ):
        raise ValueError("OCR result source mismatch")
    with Image.open(source["rgb"]) as im:
        shape = (im.height, im.width)
    texts = associate_texts(batch["instances"], ocr["texts"], shape)
    updates = []
    for i, inst in enumerate(batch["instances"]):
        if not inst["texts"] or rows[i]["result"]["status"] != "accepted":
            continue
        event = dict(
            message_id=rows[i]["parent_event_id"]
            + ":text:"
            + fingerprint(ocr["provenance"])[:12],
            parent_event_id=rows[i]["parent_event_id"],
            image_id=source["image_id"],
            rgb_hash=source["rgb_hash"],
            depth_hash=source["depth_hash"],
            texts=inst["texts"],
            provenance=ocr["provenance"],
        )
        updates.append(store.attach_text_evidence(event))
    return dict(texts=texts, updates=updates)


def save_frame(root, source, batch, rows, selection, ocr=None, binding=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    instances = []
    for i, inst in enumerate(batch["instances"]):
        np.savez_compressed(root / f"mask-{i:03}.npz", mask=inst["mask"])
        if rows[i].get("geometry_file"):
            import shutil

            shutil.copy2(rows[i]["geometry_file"], root / f"points-{i:03}.npz")
        if i in batch.get("prepared_projections", {}):
            prepared = batch["prepared_projections"][i]
            np.savez_compressed(root / f"candidate-points-{i:03}.npz", points=prepared["points"])
        instances.append({k: v for k, v in inst.items() if k != "mask"})
    if ocr:
        with Image.open(source["rgb"]) as rgb:
            for i, text in enumerate(ocr["texts"]):
                xy = np.asarray(text["polygon"])
                box = [
                    max(0, int(xy[:, 0].min())),
                    max(0, int(xy[:, 1].min())),
                    min(rgb.width, int(np.ceil(xy[:, 0].max())) + 1),
                    min(rgb.height, int(np.ceil(xy[:, 1].max())) + 1),
                ]
                if box[2] > box[0] and box[3] > box[1]:
                    rgb.crop(box).save(root / f"text-{i:03}.png")
    write_json(
        root / "frame.json",
        dict(
            source=source,
            selection=selection,
            instances=instances,
            rows=rows,
            provenance=batch["provenance"],
            segmentation_s=batch["elapsed_s"],
            candidate_request=batch.get("candidate_request"),
            ocr=ocr,
            binding=binding,
        ),
    )
