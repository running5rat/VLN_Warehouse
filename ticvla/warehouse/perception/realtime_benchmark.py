"""Shared selectors, independently persisted candidates, frozen realtime model selection."""

import argparse
import json
import time
from pathlib import Path
import numpy as np
from .realtime import (
    Selector,
    FixedSegmenter,
    MobileReader,
    consume_masks,
    consume_text,
    save_frame,
    CORE,
)
from ..mapping import MapStore, SourceObservation
from ..mapping.semantic_frame import write_json
from ..schema import file_sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("/data/ticvla/outputs/landmark-comparison-20261010"),
    )
    p.add_argument(
        "--stage", choices=["qa", "ocr", "infer", "score", "freeze"], required=True
    )
    p.add_argument(
        "--split", choices=["development", "evaluation"], default="development"
    )
    p.add_argument("--model", default="11s")
    p.add_argument("--runtime", choices=["engine", "pt"], default="engine")
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.baseline / "manifest.json").read_text())
    write_json(a.root / "manifest.json", manifest)
    rows = [
        r
        for r in manifest["rows"]
        if (r["variant"] == "development") == (a.split == "development")
    ]
    if a.stage == "qa":
        model = Selector()
        for r in rows:
            dst = a.root / "qa" / r["key"] / "result.json"
            if dst.exists():
                continue
            source = {
                **r["source"],
                "rgb_hash": r["rgb_hash"],
                "depth_hash": r["depth_hash"],
            }
            result = model.predict(source, manifest["task"])
            write_json(dst, result)
            print("QA", r["key"], result["valid"], result["parsed"], flush=True)
    elif a.stage == "ocr":
        model = MobileReader()
        for r in rows:
            dst = a.root / "ocr" / r["key"] / "result.json"
            if dst.exists():
                continue
            source = {
                **r["source"],
                "rgb_hash": r["rgb_hash"],
                "depth_hash": r["depth_hash"],
            }
            result = model.predict(source)
            write_json(dst, result)
            print(
                "OCR",
                r["key"],
                round(result["elapsed_s"], 3),
                len(result["texts"]),
                flush=True,
            )
    elif a.stage == "infer":
        if a.split == "evaluation":
            freeze = json.loads((a.root / "freeze.json").read_text())
            if a.model not in ("11s", "v8s", "v8l") or freeze["runtime"] != a.runtime:
                raise ValueError(
                    "Evaluation backend differs from frozen comparison candidates"
                )
            frozen_code = dict(freeze["code"])
            execution_path = a.root / "execution-freeze.json"
            if execution_path.exists():
                execution = json.loads(execution_path.read_text())
                if execution.get("all_156_frames_points_and_statuses_identical"):
                    for f, h in execution["code"].items():
                        if f in frozen_code:
                            frozen_code[f] = h
            if any(file_sha256(f) != h for f, h in frozen_code.items()):
                raise ValueError("Frozen inference code changed")
        weights = (
            a.root / "engines" / a.model / "model.engine"
            if a.runtime == "engine"
            else Path("/data/ticvla/models") / f"yoloe-{a.model}-seg.pt"
        )
        model = FixedSegmenter(weights)
        model.predict({**rows[0]["source"], "rgb_hash": rows[0]["rgb_hash"]})
        base = a.root / (a.model + "-" + a.runtime)
        stores = {}
        try:
            for r in rows:
                dst = base / r["key"]
                if (dst / "frame.json").exists():
                    continue
                s = stores.setdefault(r["variant"], None)
                if s is None:
                    s = stores[r["variant"]] = MapStore(base / "maps" / r["variant"])
                source = s.add_observation(SourceObservation(**r["source"]))
                qa = json.loads((a.root / "qa" / r["key"] / "result.json").read_text())
                selection = {
                    **(
                        qa["parsed"]
                        if qa["valid"]
                        else dict(selected_categories=[], proposed_categories=[])
                    ),
                    "request_id": qa["request_id"],
                    "source_image_id": qa["source_image_id"],
                }
                batch = model.predict(source)
                start = time.time()
                instances = consume_masks(s, source, batch, selection)
                projection = time.time() - start
                ocr = json.loads(
                    (a.root / "ocr" / r["key"] / "result.json").read_text()
                )
                bound = consume_text(s, source, batch, instances, ocr)
                save_frame(dst, source, batch, instances, selection, ocr, bound)
                write_json(
                    dst / "timing.json",
                    dict(segmentation_s=batch["elapsed_s"], projection_s=projection),
                )
                print(
                    "INFER",
                    a.model,
                    a.runtime,
                    r["key"],
                    len(instances),
                    len(bound["updates"]),
                    flush=True,
                )
        finally:
            for s in stores.values():
                s.close()
    elif a.stage == "score":
        score(a.root, a.baseline, a.model + "-" + a.runtime, rows, a.split)
    else:
        stats = {
            n: json.loads((a.root / f"{n}-engine/development-score.json").read_text())
            for n in ["11s", "v8s"]
        }
        eligible = {
            n: s
            for n, s in stats.items()
            if s["mask_publish_p95_s"] <= 0.5
            and s["segmentation_p95_s"] <= 0.2
            and s["matched"] > 0
        }
        chosen = None
        if eligible:
            best = max(eligible, key=lambda n: eligible[n]["recall"])
            near = [
                n
                for n, s in eligible.items()
                if eligible[best]["recall"] - s["recall"] <= 0.02
            ]
            chosen = min(near, key=lambda n: eligible[n]["mask_publish_p95_s"])
        write_json(
            a.root / "freeze.json",
            dict(
                selected_model=chosen,
                runtime="engine",
                categories=list(CORE),
                development=stats,
                manifest_sha256=file_sha256(a.root / "manifest.json"),
                code={
                    str(f): file_sha256(f)
                    for f in [
                        Path(__file__).with_name("realtime.py"),
                        Path("ticvla/warehouse/mapping/identity.py"),
                        Path("ticvla/warehouse/mapping/store.py"),
                        Path("ticvla/warehouse/structured_io.py"),
                        Path("ticvla/warehouse/object_io.py"),
                        Path("ticvla/warehouse/mapping/segmentation.py"),
                        Path(__file__).with_name("common.py"),
                    ]
                },
            ),
        )
        print("FREEZE", chosen, flush=True)
        if chosen:
            engine = a.root / "engines" / chosen / "model.engine"
            write_json(
                Path("/data/ticvla/models/warehouse-realtime/selected.json"),
                dict(
                    weights=str(engine.resolve()),
                    sha256=file_sha256(engine),
                    categories=list(CORE),
                    freeze=str((a.root / "freeze.json").resolve()),
                    development_selected=True,
                    quality_and_live_verified=False,
                ),
            )


def score(root, baseline, name, rows, split):
    from .evaluate import edit_distance, reading_key

    annotations = {
        r["key"]: r["targets"]
        for r in json.loads((baseline / "evaluation/annotations.json").read_text())[
            "rows"
        ]
    }
    total = matched = readable = correct = wrong_bind = 0
    selected_targets = selected_matched = exact = missing_read = bound_scored = 0
    selection_valid = 0
    unscored_bindings = 0
    binding_errors, correct_cases = [], []
    timings = []
    per = []
    uid_targets = {}
    ious = []
    errors = chars = 0
    for r in rows:
        folder = root / name / r["key"]
        d = json.loads((folder / "frame.json").read_text())
        targets = annotations[r["key"]]
        selected = set(d["selection"]["selected_categories"])
        qa = json.loads((root / "qa" / r["key"] / "result.json").read_text())
        selection_valid += bool(qa["valid"])
        masks = [
            np.load(folder / f"mask-{i:03}.npz")["mask"]
            for i in range(len(d["instances"]))
        ]
        pairs = []
        for ti, target in enumerate(targets):
            truth = np.load(baseline / "evaluation" / target["mask_file"])["mask"]
            for pi, mask in enumerate(masks):
                iou = float((mask & truth).sum() / max(1, (mask | truth).sum()))
                if iou >= 0.5:
                    pairs.append((iou, ti, pi))
        used_t = set()
        used_p = set()
        hits = {}
        for iou, ti, pi in sorted(pairs, reverse=True):
            if ti in used_t or pi in used_p:
                continue
            used_t.add(ti)
            used_p.add(pi)
            hits[ti] = (pi, iou)
        rc = rm = rr = rs = rsm = re = 0
        for ti, target in enumerate(targets):
            total += 1
            chosen = (
                bool(selected & {"sign", "label"})
                if target["category"] == "shelf_marker"
                else False
            )
            selected_targets += chosen
            rs += chosen
            isread = bool(
                target.get("readable") and target.get("identity_expected_unique")
            )
            readable += isread
            rr += isread
            if ti not in hits:
                if isread:
                    errors += len(target.get("text", ""))
                    chars += len(target.get("text", ""))
                    missing_read += 1
                continue
            pi, iou = hits[ti]
            matched += 1
            selected_matched += chosen
            rsm += chosen
            rm += 1
            ious.append(iou)
            pred = d["instances"][pi]
            row = d["rows"][pi]
            texts = pred.get("texts", [])
            reading = " | ".join(
                t["normalized"]
                for t in sorted(
                    texts,
                    key=lambda t: (
                        round(min(p[1] for p in t["polygon"]) / 12),
                        min(p[0] for p in t["polygon"]),
                    ),
                )
            )
            expected = target.get("text", "")
            admitted = row["result"]["status"] == "accepted"
            if isread:
                errors += edit_distance(reading, expected)
                chars += len(expected)
                ok = reading_key(reading) == reading_key(expected) and admitted
                exact += reading == expected
                re += reading == expected
                missing_read += not bool(reading)
                correct += ok
                rc += ok
                if ok:
                    correct_cases.append(
                        dict(
                            key=r["key"],
                            instance=pi,
                            target_uid=target["uid"],
                            expected=expected,
                            reading=reading,
                            landmark_uid=row["result"].get("landmark_uid"),
                        )
                    )
            uid = row["result"].get("landmark_uid")
            if uid:
                uid_targets.setdefault((r["variant"], uid), set()).add(target["uid"])
        # Audit every bound text, including a sign read incorrectly attached to a rack
        # whose large mask does not match any annotated sign instance.
        from .common import polygon_mask

        for text in d.get("binding", {}).get("texts", []):
            owner = text.get("owner_instance")
            if text.get("status") != "bound" or owner is None:
                continue
            tm = polygon_mask(text["polygon"], masks[owner].shape)
            carriers = []
            for ti, target in enumerate(targets):
                truth = np.load(baseline / "evaluation" / target["mask_file"])["mask"]
                coverage = (tm & truth).sum() / max(1, tm.sum())
                if coverage >= 0.5:
                    carriers.append((coverage, ti, truth))
            if not carriers:
                unscored_bindings += 1
                continue
            _, ti, truth = max(carriers, key=lambda x: x[0])
            scored_iou = (masks[owner] & truth).sum() / max(
                1, (masks[owner] | truth).sum()
            )
            bound_scored += 1
            if scored_iou < 0.5:
                wrong_bind += 1
                binding_errors.append(
                    dict(
                        key=r["key"],
                        text=text["text"],
                        polygon=text["polygon"],
                        owner=owner,
                        owner_category=d["instances"][owner]["category"],
                        target_uid=targets[ti]["uid"],
                        owner_mask_iou=float(scored_iou),
                        map_status=d["rows"][owner]["result"]["status"],
                    )
                )
        timing = json.loads((folder / "timing.json").read_text())
        timings.append(timing)
        per.append(
            dict(
                key=r["key"],
                variant=r["variant"],
                targets=len(targets),
                matched=rm,
                readable=rr,
                correct_3d=rc,
                exact_text=re,
                vlm_selected_targets=rs,
                selected_matched=rsm,
            )
        )
    s = dict(
        frames=len(rows),
        targets=total,
        matched=matched,
        recall=matched / max(1, total),
        readable_unique=readable,
        selection_valid_frames=selection_valid,
        vlm_selected_targets=selected_targets,
        vlm_target_selection_coverage=selected_targets / max(1, total),
        selected_category_instance_recall=selected_matched / max(1, selected_targets),
        exact_text_matches=exact,
        exact_text_rate=exact / max(1, readable),
        no_reading_fraction=missing_read / max(1, readable),
        scored_bindings=bound_scored,
        bindings_outside_annotated_targets=unscored_bindings,
        correct_binding_rate=(bound_scored - wrong_bind) / bound_scored
        if bound_scored
        else None,
        correct_3d=correct,
        wrong_binding_observations=wrong_bind,
        binding_errors=binding_errors,
        correct_cases=correct_cases,
        false_merges=sum(len(v) > 1 for v in uid_targets.values()),
        matched_mask_iou=float(np.mean(ious)) if ious else None,
        character_error_rate=errors / chars if chars else None,
        segmentation_p50_s=float(
            np.percentile([t["segmentation_s"] for t in timings], 50)
        ),
        segmentation_p95_s=float(
            np.percentile([t["segmentation_s"] for t in timings], 95)
        ),
        mask_publish_p95_s=float(
            np.percentile(
                [t["segmentation_s"] + t["projection_s"] for t in timings], 95
            )
        ),
        rows=per,
        scope="Installed targets only; shelf_marker matches sign/label for selection scoring; other facilities lack exhaustive annotations; strict text separate from normalized 3D identity; offline timings exclude live contention and queueing; unreadable strings not scored",
    )
    import sqlite3

    confirmed = []
    variants = sorted({r["variant"] for r in rows})
    for variant in variants:
        db = sqlite3.connect(
            f"file:{root / name / 'maps' / variant / 'map.sqlite'}?mode=ro", uri=True
        )
        for (payload,) in db.execute("SELECT payload FROM landmarks"):
            lm = json.loads(payload)
            if lm.get("identity_status") == "identity_confirmed":
                confirmed.append(
                    dict(
                        variant=variant,
                        landmark_uid=lm["landmark_uid"],
                        text=lm.get("identity_text"),
                        matched_targets=sorted(
                            uid_targets.get((variant, lm["landmark_uid"]), [])
                        ),
                    )
                )
        db.close()
    s["confirmed_identities"] = confirmed
    s["wrong_confirmed_identities"] = 0 if not confirmed else None
    s["identity_confirmation_scope"] = (
        "No confirmations observed"
        if not confirmed
        else "Confirmation content requires source-ground-truth audit"
    )
    write_json(root / name / f"{split}-score.json", s)
    print("SCORE", name, {k: v for k, v in s.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    main()
