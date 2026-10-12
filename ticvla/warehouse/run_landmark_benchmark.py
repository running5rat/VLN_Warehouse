"""Frozen RGB-D/VLM comparison. Evaluation truth is never sent to perception.

prepare -> qa -> infer (one backend at a time) -> evaluate.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .mapping import MapStore, SourceObservation, SemanticEvent
from .mapping.semantic_frame import consume_categories, write_json
from .schema import fingerprint, file_sha256
from .object_io import object_observation, run_object_qa
from .run_semantic_frame import TASK

VARIANTS = ("original", "color", "layout", "surface", "lighting", "occlusion")


def source_from_frame(episode, index):
    episode = Path(episode)
    meta = json.loads(episode.read_text())
    frames = [
        json.loads(x)
        for x in (episode.parent / meta["frames_file"]).read_text().splitlines()
    ]
    f = frames[index]
    if f["depth_convention"] != "optical_z_metres":
        raise ValueError("Depth convention mismatch")
    return SourceObservation(
        f"{meta['episode_id']}:{index:08d}",
        str(episode.parent / f["rgb"]),
        str(episode.parent / f["depth"]),
        f["K"],
        f["T_body_camera"],
        f["T_odom_body"],
        f["t_obs"],
        f["odom_epoch"],
        f["odom_epoch"],
        "landmark-comparison",
    )


def prepare(root, development_only=False):
    rows = []
    # Uniform path-progress sampling: based solely on recorded odometry, not
    # perception outputs or whether text happens to be readable.
    for variant in (
        ("development",) if development_only else ("development",) + VARIANTS
    ):
        episode = root / f"capture-{variant}/raw/episode.json"
        capture_report = json.loads(
            (episode.parent.parent / "capture-report.json").read_text()
        )
        if not capture_report["loop_complete"]:
            raise ValueError("Capture did not complete the fixed circuit: " + variant)
        frames = [
            json.loads(x)
            for x in (episode.parent / "frames.jsonl").read_text().splitlines()
        ]
        points = np.array([np.array(f["T_odom_body"])[:2, 3] for f in frames])
        distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        targets = np.linspace(
            0, distance[-1], 12 if variant == "development" else 24, endpoint=False
        )
        indices = [int(np.argmin(abs(distance - t))) for t in targets]
        if len(set(indices)) != len(indices):
            raise ValueError("Insufficient distinct trajectory samples")
        for index in indices:
            source = source_from_frame(episode, index)
            rows.append(
                dict(
                    key=f"{variant}-{index:08d}",
                    variant=variant,
                    frame=index,
                    episode=str(episode),
                    source=asdict(source),
                    rgb_hash=file_sha256(source.rgb),
                    depth_hash=file_sha256(source.depth),
                )
            )
    manifest = dict(
        selection="uniform odometric path progress, fixed before inference",
        task=TASK,
        rows=rows,
    )
    path = root / "manifest.json"
    if (
        path.exists()
        and json.loads(path.read_text()) != manifest
        and any(
            r["variant"] != "development" for r in json.loads(path.read_text())["rows"]
        )
    ):
        raise ValueError("Frozen manifest differs")
    frozen_files = {
        p: file_sha256(p)
        for p in [
            "ticvla/warehouse/object_io.py",
            "ticvla/warehouse/perception/worker.py",
            "ticvla/warehouse/perception/common.py",
            "ticvla/warehouse/mapping/identity.py",
        ]
    }
    freeze_path = root / "freeze.json"
    if (
        path.exists()
        and freeze_path.exists()
        and any(
            r["variant"] != "development" for r in json.loads(path.read_text())["rows"]
        )
    ):
        old = json.loads(freeze_path.read_text())
        if old["code_files"] != frozen_files:
            raise ValueError(
                "Evaluation code freeze differs; use a new experiment directory"
            )
        print("FROZEN_UNCHANGED", len(rows), flush=True)
        return
    write_json(path, manifest)
    write_json(
        freeze_path,
        dict(
            manifest_sha256=file_sha256(path),
            task=TASK,
            code_files=frozen_files,
        ),
    )
    print("FROZEN", len(rows), flush=True)


def qa(root, split):
    import torch
    from .config import WarehouseConfig
    from .structured_io import StructuredVLM

    torch.set_num_threads(2)
    torch.manual_seed(42)
    manifest = json.loads((root / "manifest.json").read_text())
    model = StructuredVLM(
        WarehouseConfig(
            model_path="/data/ticvla/models/InternVL3-1B", vlm_checkpoint=None
        )
    )
    model_dir = Path("/data/ticvla/models/InternVL3-1B")
    write_json(
        root / "qa-model-provenance.json",
        dict(
            model=str(model_dir),
            base_model_only=True,
            files={
                p.name: file_sha256(p)
                for p in model_dir.iterdir()
                if p.is_file() and p.suffix in (".json", ".py", ".safetensors")
            },
            decoding={"do_sample": False, "max_new_tokens": 160},
            input="one source RGB plus navigation task; no map, OCR, Oracle or catalog",
        ),
    )
    for row in manifest["rows"]:
        if (row["variant"] == "development") != (split == "development"):
            continue
        out = root / "qa" / row["key"]
        if (out / "result.json").exists():
            continue
        observation = object_observation(row["source"], "navigation", manifest["task"])
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "observation.json", observation)
        request = dict(
            request_id=fingerprint(observation)[:24],
            source_hashes={k: row[k] for k in ("rgb_hash", "depth_hash")},
            manifest_sha256=file_sha256(root / "manifest.json"),
        )
        write_json(out / "request.json", request)
        result, events, timings = run_object_qa(
            model, observation, request["request_id"], out
        )
        write_json(
            out / "result.json",
            dict(result=result, events=[asdict(e) for e in events], timings=timings),
        )
        print("QA", row["key"], result["valid"], result.get("parsed"), flush=True)


def infer(root, name, split, diagnostic=False, shard=0, shards=1, only_keys=None):
    from .perception import PerceptionClient
    from .mapping.semantic_visualization import render_semantic_frame

    manifest = json.loads((root / "manifest.json").read_text())
    if only_keys and not set(only_keys) <= {r["key"] for r in manifest["rows"]}:
        raise ValueError("Diagnostic key is absent from frozen manifest")
    output = root / (("diagnostic-" if diagnostic else "") + name)
    output.mkdir(exist_ok=True)
    backend = PerceptionClient(
        name, output / "perception", "/data/ticvla/models/yoloe-v8l-seg.pt"
    )
    stores = {}
    try:
        for row in manifest["rows"]:
            if only_keys and row["key"] not in only_keys:
                continue
            if (row["variant"] == "development") != (split == "development"):
                continue
            variants = ("development",) if split == "development" else VARIANTS
            if variants.index(row["variant"]) % shards != shard:
                continue
            if row["variant"] not in stores:
                stores[row["variant"]] = MapStore(output / "maps" / row["variant"])
            store = stores[row["variant"]]
            out = output / row["key"]
            out.mkdir(exist_ok=True)
            record = store.add_observation(SourceObservation(**row["source"]))
            if any(record[k] != row[k] for k in ("rgb_hash", "depth_hash")):
                raise ValueError("Manifest source changed")
            qa_result = json.loads(
                (root / "qa" / row["key"] / "result.json").read_text()
            )
            events = [SemanticEvent(**e) for e in qa_result["events"]]
            if diagnostic:
                events = [
                    SemanticEvent(
                        "diagnostic:" + row["key"] + ":" + c,
                        record["image_id"],
                        c,
                        source="manual",
                    )
                    for c in ("sign", "storage rack", "box", "container")
                ]
            processed = consume_categories(store, events, backend, out / "segmentation")
            before_replay = store.summary()
            consume_categories(store, events, None, out / "segmentation")
            if before_replay != store.summary():
                raise AssertionError("Replaying saved evidence changed the map")
            write_json(
                out / "report.json",
                dict(
                    source=record,
                    qa=qa_result,
                    processed=processed,
                    map_version=store.version,
                    diagnostic=diagnostic,
                    manual_category_prompts=[e.category for e in events]
                    if diagnostic
                    else None,
                    replay_unchanged=True,
                    offline_workers=shards,
                    qa_sha256=file_sha256(root / "qa" / row["key"] / "result.json"),
                ),
            )
            # Render each selected source with its actual depth and masks.
            render_semantic_frame(
                store,
                record["image_id"],
                processed,
                qa_result["result"],
                out,
                weight_mode="internvl_base_control",
            )
            print(
                "PERCEPTION",
                name,
                row["key"],
                len(processed["instances"]),
                processed["error"],
                flush=True,
            )
    finally:
        backend.close()
        for store in stores.values():
            store.close()


def truth(root):
    """Offline-only evaluator builds masks from simulator identities, not predictions."""
    manifest = json.loads((root / "manifest.json").read_text())
    out = root / "evaluation"
    out.mkdir(exist_ok=True)
    if (out / "annotations.json").exists():
        raise ValueError("Refusing to overwrite existing independent annotations")
    all_rows = []
    for row in manifest["rows"]:
        episode = Path(row["episode"])
        supervision = episode.parent / "supervision" / f"{row['frame']:08d}.npz"
        with np.load(supervision, allow_pickle=False) as data:
            ids = data["instance_ids"]
            labels = json.loads(data["instance_labels_json"].tobytes())
        catalog = json.loads((episode.parent.parent / "shelf-catalog.json").read_text())
        targets = []
        for ci, spec in enumerate(catalog):
            values = [
                int(i) for i, label in labels.items() if spec["prim"] in str(label)
            ]
            mask = np.isin(ids, values)
            if mask.sum() < 16:
                continue
            ys, xs = np.where(mask)
            path = out / f"{row['key']}-{ci}.npz"
            np.savez_compressed(path, mask=mask)
            targets.append(
                dict(
                    uid=spec["prim"],
                    text=spec["label"],
                    category=spec["category"],
                    identity_expected_unique=spec["identity_expected_unique"],
                    bbox=[
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() + 1),
                        int(ys.max() + 1),
                    ],
                    mask_file=path.name,
                    pixels=int(mask.sum()),
                    readable=None,
                    readability_review="pending",
                    text_fully_visible=spec["text_fully_visible"],
                )
            )
        all_rows.append(dict(key=row["key"], targets=targets))
    write_json(
        out / "annotations.json",
        dict(
            source="Isaac instance identities plus independent visual readability review",
            rows=all_rows,
        ),
    )
    # Native crops at up to 3x scale, no OCR output in annotation galleries.
    for variant in ("development",) + VARIANTS:
        selected = [r for r in all_rows if r["key"].startswith(variant + "-")]
        cells = []
        for r in selected:
            src = next(x for x in manifest["rows"] if x["key"] == r["key"])
            image = Image.open(src["source"]["rgb"])
            for i, t in enumerate(r["targets"]):
                crop = image.crop(t["bbox"])
                scale = min(3, 360 / crop.width, 130 / crop.height)
                crop = crop.resize(
                    (max(1, int(crop.width * scale)), max(1, int(crop.height * scale)))
                )
                cell = Image.new("RGB", (380, 190), "#26323e")
                cell.paste(crop, ((380 - crop.width) // 2, 40))
                ImageDraw.Draw(cell).text(
                    (6, 5),
                    f"{r['key']} t{i}: {t['text']}\n{t['bbox']} pixels={t['pixels']}",
                    fill="white",
                )
                cells.append(cell)
        for start in range(0, len(cells), 30):
            page = cells[start : start + 30]
            canvas = Image.new(
                "RGB", (380 * 5, 190 * int(np.ceil(len(page) / 5))), "white"
            )
            for i, cell in enumerate(page):
                canvas.paste(cell, ((i % 5) * 380, (i // 5) * 190))
            canvas.save(out / f"{variant}-review-{start // 30}.png")
    print("TRUTH_GALLERIES", str(out), flush=True)


def evaluate(root):
    from .perception.evaluate import evaluate_comparison

    evaluate_comparison(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=("prepare", "qa", "infer", "truth", "evaluate"),
        required=True,
    )
    p.add_argument(
        "--split", choices=("development", "evaluation"), default="evaluation"
    )
    p.add_argument("--backend", choices=("yoloe-ppocr", "florence2"))
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument(
        "--keys",
        nargs="+",
        help="Restrict manual diagnostics only; never filter held-out evaluation",
    )
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", type=int)
    a = p.parse_args()
    if a.keys and (a.stage != "infer" or not a.diagnostic):
        p.error("--keys requires --stage infer --diagnostic")
    if a.stage == "prepare":
        prepare(a.root, a.split == "development")
    elif a.stage == "qa":
        qa(a.root, a.split)
    elif a.stage == "infer":
        if not a.backend:
            p.error("infer requires backend")
        if not 1 <= a.workers <= 3 or (a.split == "development" and a.workers != 1):
            p.error("workers must be 1..3; development uses one worker")
        if a.workers > 1 and a.shard is None:
            import subprocess
            import sys

            children = []
            logs = []
            try:
                for shard in range(a.workers):
                    log = (a.root / f"{a.backend}-shard-{shard}.log").open("w")
                    logs.append(log)
                    cmd = [
                        sys.executable,
                        "-m",
                        "ticvla.warehouse.run_landmark_benchmark",
                        "--root",
                        str(a.root),
                        "--stage",
                        "infer",
                        "--backend",
                        a.backend,
                        "--split",
                        a.split,
                        "--workers",
                        str(a.workers),
                        "--shard",
                        str(shard),
                    ]
                    if a.diagnostic:
                        cmd.append("--diagnostic")
                    if a.keys:
                        cmd.extend(["--keys", *a.keys])
                    children.append(
                        subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    )
                codes = [child.wait() for child in children]
                if any(codes):
                    raise RuntimeError("Inference shard failed: " + str(codes))
            finally:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
                for log in logs:
                    log.close()
        else:
            infer(
                a.root,
                a.backend,
                a.split,
                a.diagnostic,
                a.shard or 0,
                a.workers,
                a.keys,
            )
    elif a.stage == "truth":
        truth(a.root)
    else:
        evaluate(a.root)


if __name__ == "__main__":
    main()
