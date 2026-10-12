"""Real YOLO-E adapter. Imports are optional and confined to this worker."""

import importlib.metadata
from pathlib import Path

import numpy as np
from PIL import Image

from ..schema import file_sha256


class YOLOESegmenter:
    def __init__(
        self,
        weights,
        device="cuda:0",
        confidence=0.25,
        *,
        cache_classes=False,
        half=False,
    ):
        from ultralytics import YOLOE

        self.weights = str(Path(weights).resolve())
        if not Path(self.weights).is_file():
            raise FileNotFoundError(
                "Download the pinned YOLO-E checkpoint before running"
            )
        self.model = YOLOE(self.weights).to(device)
        self.model.model.eval().requires_grad_(False)
        self._text_encoder = None
        self.device, self.confidence = device, confidence
        self.cache_classes, self.half = cache_classes, half
        self._cached_categories = None
        self.provenance = {
            "backend": "yoloe",
            "weights": self.weights,
            "sha256": file_sha256(self.weights),
            "ultralytics": importlib.metadata.version("ultralytics"),
        }

    def _set_text_classes(self, categories):
        import clip
        import torch

        encoder_path = Path(self.weights).parent / "mobileclip_blt.ts"
        if not encoder_path.is_file():
            raise FileNotFoundError(
                "Text prompts require mobileclip_blt.ts beside the YOLO-E weights"
            )
        if self._text_encoder is None:
            self._text_encoder = torch.jit.load(
                str(encoder_path), map_location=self.device
            ).eval()
            self.provenance["text_encoder_sha256"] = file_sha256(encoder_path)
        with torch.inference_mode():
            features = self._text_encoder(
                clip.tokenize(categories).to(self.device)
            ).float()
            # Same MobileCLIP + RepRTA path as pinned YOLOEModel.get_text_pe,
            # with explicit local assets rather than cwd-dependent downloads.
            embedding = self.model.model.model[-1].get_tpe(features[None])
            self.model.set_classes(categories, embedding)

    def segment_instances(self, observation, categories):
        """Text-prompt all categories together; preserve every detected instance."""
        if file_sha256(observation["rgb"]) != observation["rgb_hash"]:
            raise ValueError("Source RGB changed")
        categories = list(
            dict.fromkeys(" ".join(c.split()).casefold() for c in categories)
        )
        if any(not c for c in categories):
            raise ValueError("Empty segmentation category")
        if not categories:
            return {
                "instances": [],
                "provenance": dict(self.provenance),
                "possibly_truncated": False,
            }
        if not self.cache_classes or self._cached_categories != tuple(categories):
            self.model.predictor = None
            self._set_text_classes(categories)
            self._cached_categories = tuple(categories)
        result = self.model.predict(
            observation["rgb"],
            device=self.device,
            conf=self.confidence,
            imgsz=640,
            max_det=100,
            retina_masks=True,
            verbose=False,
            save=False,
            half=self.half,
            rect=not self.cache_classes,
        )[0]
        with Image.open(observation["rgb"]) as source:
            width, height = source.size
        instances = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy() > 0.5
            if masks.shape != (len(result.boxes), height, width):
                raise ValueError("Instance masks/source dimensions differ")
            for mask, box, cls, conf in zip(
                masks,
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                if not 0 <= int(cls) < len(categories) or not np.isfinite(conf):
                    raise ValueError("Invalid instance class/confidence")
                instances.append(
                    dict(
                        category=categories[int(cls)],
                        confidence=float(conf),
                        bbox=box.tolist(),
                        mask=mask,
                    )
                )
        instances.sort(key=lambda x: (x["category"], *x["bbox"], -x["confidence"]))
        return {
            "instances": instances,
            "provenance": {
                **self.provenance,
                "prompt": "text",
                "categories": categories,
                "imgsz": 640,
                "confidence_threshold": self.confidence,
                "max_det": 100,
                "retina_masks": True,
            },
            "possibly_truncated": len(instances) >= 100,
        }

    def segment(self, observation, event):
        if event.image_id != observation["image_id"]:
            raise ValueError("Segmentation source image mismatch")
        if file_sha256(observation["rgb"]) != observation["rgb_hash"]:
            raise ValueError("Source RGB changed")
        # Text and visual prompt predictors have different state; do not reuse a
        # previous visual predictor when switching back to text.
        self.model.predictor = None
        kwargs = {}
        if event.roi is not None:
            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

            kwargs["predictor"] = YOLOEVPSegPredictor
            roi = np.asarray(event.roi, dtype=np.float32)
            with Image.open(observation["rgb"]) as image:
                w, h = image.size
            if (
                roi.shape != (4,)
                or not np.isfinite(roi).all()
                or not (0 <= roi[0] < roi[2] <= w and 0 <= roi[1] < roi[3] <= h)
            ):
                raise ValueError("Invalid visual prompt ROI")
            kwargs["visual_prompts"] = {"bboxes": roi[None], "cls": np.array([0])}
        else:
            self._set_text_classes([event.category])
        result = self.model.predict(
            observation["rgb"],
            device=self.device,
            conf=self.confidence,
            retina_masks=True,
            verbose=False,
            save=False,
            **kwargs,
        )[0]
        if result.masks is None:
            return (
                None,
                0.0,
                {
                    **self.provenance,
                    "prompt": "roi" if event.roi else "text",
                    "detections": 0,
                },
            )
        masks = result.masks.data.cpu().numpy() > 0.5
        scores = result.boxes.conf.cpu().numpy()
        if event.roi is None and len(masks) > 1:
            # A category name is not a unique instance identity. Require an ROI
            # instead of arbitrarily choosing the most confident repeated shelf.
            return (
                None,
                0.0,
                {
                    **self.provenance,
                    "detections": len(masks),
                    "reason": "Ambiguous text prompt; provide ROI",
                },
            )
        if event.roi is not None:
            x1, y1, x2, y2 = map(int, event.roi)
            roi_mask = np.zeros_like(masks[0])
            roi_mask[y1:y2, x1:x2] = True
            relevance = [
                (m & roi_mask).sum() / max((m | roi_mask).sum(), 1) for m in masks
            ]
            idx = int(np.argmax(relevance))
        else:
            idx = int(np.argmax(scores))
        return (
            masks[idx],
            float(scores[idx]),
            {
                **self.provenance,
                "detections": len(masks),
                "prompt": "roi" if event.roi else "text",
            },
        )


def process_event(store, event, segmenter):
    """Exactly-once consumer: reject missing sources before GPU work."""
    existing = store.get("events", event.message_id)
    if existing:
        return store.consume(event)
    if event.kind != "object":
        return store.consume(event)
    try:
        obs, _ = store.source_arrays(event.image_id)
        mask, confidence, provenance = segmenter.segment(obs, event)
    except (ValueError, OSError) as exc:
        return store.consume(event, segmentation={"error": str(exc)})
    return store.consume(event, mask, confidence, segmentation=provenance)
