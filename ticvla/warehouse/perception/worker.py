"""Isolated real-model worker: no GT, no colour/layout rules, no model fallback."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from ..mapping.semantic_frame import write_json
from ..schema import file_sha256
from .common import associate_texts, crop_box, polygon_mask


def hashes(root):
    return {
        str(p.relative_to(root)): file_sha256(p)
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
        and ".cache" not in p.parts
        and p.suffix in (".json", ".safetensors", ".pdiparams", ".yml", ".py")
    }


class YoloPPOCR:
    def __init__(self, args):
        from ..mapping.segmentation import YOLOESegmenter
        from paddleocr import PaddleOCR

        self.segmenter = YOLOESegmenter(args.weights, args.device)
        # CPU inference deliberately isolates Paddle from the CUDA allocator used
        # by InternVL/YOLO/Isaac. Actual timing, including CPU OCR, is reported.
        self.reader = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
            cpu_threads=2,
        )
        self.provenance = dict(
            self.segmenter.provenance,
            backend="yoloe-ppocr",
            text_detector="PP-OCRv5_server_det",
            text_recognizer="PP-OCRv5_server_rec",
            ocr_device="cpu",
            paddleocr=importlib.metadata.version("paddleocr"),
            ocr_files=hashes(Path.home() / ".paddlex/official_models"),
        )

    def perceive(self, obs, categories, output):
        t = time.perf_counter()
        batch = self.segmenter.segment_instances(obs, categories)
        timings = {"segmentation_s": time.perf_counter() - t}
        rgb = Image.open(obs["rgb"]).convert("RGB")
        texts, calls = [], 0
        t = time.perf_counter()
        for i, instance in enumerate(batch["instances"]):
            box = crop_box(instance["bbox"], (rgb.height, rgb.width))
            crop = rgb.crop(box)
            crop.save(output / f"crop-{i:03d}.png")
            preds = list(self.reader.predict(np.array(crop)[:, :, ::-1].copy()))
            calls += 1
            for k, pred in enumerate(preds):
                pred.save_to_json(str(output / f"ocr-{i:03d}-{k}.json"))
                data = pred.json["res"]
                for txt, conf, poly in zip(
                    data.get("rec_texts", []),
                    data.get("rec_scores", []),
                    data.get("rec_polys", []),
                ):
                    points = np.asarray(poly, float) + np.array(box[:2])
                    texts.append(
                        dict(
                            text=txt,
                            confidence=float(conf),
                            polygon=points.tolist(),
                            crop_instance=i,
                        )
                    )
        timings["text_recognition_s"] = time.perf_counter() - t
        batch.update(
            texts=texts,
            timings=timings,
            calls=dict(segmentation=int(bool(categories)), text_recognition=calls),
        )
        return batch


class Florence:
    def __init__(self, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.torch = torch
        self.device = args.device
        self.dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                args.florence_model,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(
            args.florence_model, trust_remote_code=True
        )
        self.provenance = dict(
            backend="florence2",
            model=str(args.florence_model),
            files=hashes(Path(args.florence_model)),
            confidence_kind="unavailable",
            transformers=importlib.metadata.version("transformers"),
        )

    def generate(self, image, task, text, output, name):
        started = time.perf_counter()
        inputs = self.processor(text=task + text, images=image, return_tensors="pt").to(
            self.device, self.dtype
        )
        with self.torch.inference_mode():
            ids = self.model.generate(
                **inputs, max_new_tokens=1024, num_beams=3, do_sample=False
            )
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        truncated = ids.shape[-1] >= 1024
        parsed = None
        error = None
        try:
            if truncated:
                raise ValueError("Florence output may be truncated")
            clean = raw
            for special in ("<s>", "</s>", "<pad>"):
                clean = clean.replace(special, "")
            parsed = self.processor.post_process_generation(
                clean, task=task, image_size=image.size
            )[task]
        except Exception as exc:
            error = str(exc)
        write_json(
            output / (name + ".json"),
            dict(
                task=task,
                prompt=text,
                raw=raw,
                parsed=parsed,
                error=error,
                tokens=int(ids.shape[-1]),
                elapsed_s=time.perf_counter() - started,
            ),
        )
        self.task_elapsed[task] = (
            self.task_elapsed.get(task, 0.0) + time.perf_counter() - started
        )
        if error:
            raise ValueError(error)
        return parsed

    def perceive(self, obs, categories, output):
        self.task_elapsed = {}
        rgb = Image.open(obs["rgb"]).convert("RGB")
        instances, texts, failures = [], [], []
        calls = dict(localization=0, segmentation=0, text_recognition=0)
        timings = dict(localization_s=0.0, segmentation_s=0.0, text_recognition_s=0.0)
        for category_index, category in enumerate(categories):
            t = time.perf_counter()
            calls["localization"] += 1
            det = self.generate(
                rgb,
                "<OPEN_VOCABULARY_DETECTION>",
                category,
                output,
                f"detect-{category_index}",
            )
            timings["localization_s"] += time.perf_counter() - t
            for bi, box in enumerate(det.get("bboxes", [])):
                i = len(instances)
                try:
                    crop_bounds = crop_box(box, (rgb.height, rgb.width))
                    # Explicit per-instance region task: do not union all same-class polygons.
                    loc = "".join(
                        f"<loc_{max(0, min(999, int(v / s * 1000)))}>"
                        for v, s in zip(
                            box, (rgb.width, rgb.height, rgb.width, rgb.height)
                        )
                    )
                    t = time.perf_counter()
                    calls["segmentation"] += 1
                    seg = self.generate(
                        rgb,
                        "<REGION_TO_SEGMENTATION>",
                        loc,
                        output,
                        f"seg-{category_index}-{bi}",
                    )
                    timings["segmentation_s"] += time.perf_counter() - t
                    mask = np.zeros((rgb.height, rgb.width), dtype=bool)
                    for group in seg.get("polygons", []):
                        for polygon in group:
                            mask |= polygon_mask(polygon, mask.shape)
                    if not mask.any():
                        raise ValueError("Florence returned no polygon mask")
                    x1, y1, x2, y2 = map(int, box)
                    if mask[y1:y2, x1:x2].sum() / mask.sum() < 0.5:
                        raise ValueError("Florence mask does not match instance box")
                    instance = dict(
                        category=category,
                        bbox=list(map(float, box)),
                        confidence=None,
                        mask=mask,
                    )
                    instances.append(instance)
                    crop = rgb.crop(crop_bounds)
                    crop.save(output / f"crop-{i:03d}.png")
                    t = time.perf_counter()
                    calls["text_recognition"] += 1
                    ocr = self.generate(
                        crop, "<OCR_WITH_REGION>", "", output, f"ocr-{i:03d}"
                    )
                    timings["text_recognition_s"] += time.perf_counter() - t
                    for txt, quad in zip(
                        ocr.get("labels", []), ocr.get("quad_boxes", [])
                    ):
                        points = np.array(quad, float).reshape(-1, 2) + np.array(
                            crop_bounds[:2]
                        )
                        texts.append(
                            dict(
                                text=txt,
                                confidence=None,
                                polygon=points.tolist(),
                                crop_instance=i,
                            )
                        )
                except (ValueError, RuntimeError) as exc:
                    failures.append(dict(category=category, bbox=box, reason=str(exc)))
        timings.update(
            localization_s=self.task_elapsed.get("<OPEN_VOCABULARY_DETECTION>", 0.0),
            segmentation_s=self.task_elapsed.get("<REGION_TO_SEGMENTATION>", 0.0),
            text_recognition_s=self.task_elapsed.get("<OCR_WITH_REGION>", 0.0),
        )
        return dict(
            instances=instances,
            texts=texts,
            failures=failures,
            timings=timings,
            calls=calls,
            possibly_truncated=False,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--florence-model", required=True)
    args = p.parse_args()
    try:
        import torch

        torch.set_num_threads(2)
        backend = YoloPPOCR(args) if args.backend == "yoloe-ppocr" else Florence(args)
        write_json(args.root / "ready.json", dict(provenance=backend.provenance))
        while not (args.root / "stop").exists():
            for request in sorted(args.root.glob("*/request.json")):
                out = request.parent
                if (out / "result.json").exists():
                    continue
                try:
                    job = json.loads(request.read_text())
                    obs = job["observation"]
                    if file_sha256(obs["rgb"]) != obs["rgb_hash"]:
                        raise ValueError("Source RGB hash changed")
                    start = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                    batch = backend.perceive(obs, job["categories"], out)
                    with Image.open(obs["rgb"]) as im:
                        shape = (im.height, im.width)
                    for inst in batch["instances"]:
                        if (
                            inst["category"] not in job["categories"]
                            or inst["mask"].shape != shape
                            or inst["mask"].dtype != bool
                        ):
                            raise ValueError("Invalid instance category/mask")
                    batch["texts"] = associate_texts(
                        batch["instances"], batch.pop("texts"), shape
                    )
                    for i, instance in enumerate(batch["instances"]):
                        path = out / f"mask-{i:03d}.npz"
                        np.savez_compressed(path, mask=instance.pop("mask"))
                        instance.update(
                            mask_file=path.name, mask_sha256=file_sha256(path)
                        )
                    batch.update(
                        source=obs,
                        categories=job["categories"],
                        provenance={
                            **batch.get("provenance", {}),
                            **backend.provenance,
                        },
                        error=None,
                    )
                    batch["timings"]["total_s"] = time.perf_counter() - start
                    batch["peak_cuda_allocated_bytes"] = (
                        torch.cuda.max_memory_allocated()
                        if torch.cuda.is_available()
                        else 0
                    )
                    write_json(out / "result.json", batch)
                except Exception:
                    write_json(out / "result.json", dict(error=traceback.format_exc()))
            time.sleep(0.05)
    except Exception:
        write_json(args.root / "ready.json", dict(error=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
