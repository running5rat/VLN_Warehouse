"""Offline evaluation with a strict privilege boundary and explicit denominators."""

import json
from collections import Counter
from pathlib import Path
import numpy as np
from ..mapping import MapStore
from ..mapping.semantic_frame import write_json
from .common import normalize_text
from ..geometry import project_depth, transform_points
from scipy.spatial import cKDTree


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


def reading_key(value):
    # Report literal accuracy separately; spacing may be split into OCR regions.
    return "".join(normalize_text(value).split()).replace("|", "")


def evaluate_comparison(root):
    root = Path(root)
    annotations = json.loads((root / "evaluation/annotations.json").read_text())
    annotation_by = {r["key"]: r["targets"] for r in annotations["rows"]}
    manifest = json.loads((root / "manifest.json").read_text())
    evaluation_rows = [r for r in manifest["rows"] if r["variant"] != "development"]
    reports = {}
    for backend in ("yoloe-ppocr", "florence2"):
        rows = []
        times = []
        qa_times = []
        text_times = []
        seg_times = []
        localization_times = []
        peak_memory = []
        request_times, completion_times = [], []
        task_calls = {}
        identity_states, reobservation_reasons = Counter(), Counter()
        perception_failures, admission_states, binding_states = (
            Counter(),
            Counter(),
            Counter(),
        )
        instance_total = 0
        qa_valid = 0
        worker_counts = set()
        selection_hits = selection_total = background_categories = (
            raw_category_count
        ) = 0
        confirmed = 0
        wrong_confirmed = 0
        false_merge = 0
        unscored_confirmed = 0
        uid_targets = {}
        uid_predictions = {}
        confirmed_correct_unique_targets = set()
        confirmed_correct_unique_landmarks = 0
        for row in evaluation_rows:
            path = root / backend / row["key"] / "report.json"
            if not path.exists():
                rows.append(dict(key=row["key"], variant=row["variant"], missing=True))
                continue
            report = json.loads(path.read_text())
            worker_counts.add(report.get("offline_workers", 1))
            processed = report["processed"]
            instance_total += len(processed["instances"])
            perception_failures.update(
                f["reason"] for f in processed.get("failures", [])
            )
            if processed.get("error"):
                perception_failures.update([processed["error"]])
            admission_states.update(
                i["result"]["status"] for i in processed["instances"]
            )
            binding_states.update(t["status"] for t in processed.get("texts", []))
            qa = report["qa"]
            qa_valid += int(qa["result"]["valid"])
            qa_times.append(qa["timings"]["object_qa_s"])
            timing = processed["perception_timings"]
            if timing:
                times.append(timing["total_s"])
                text_times.append(timing.get("text_recognition_s", 0))
                seg_times.append(timing.get("segmentation_s", 0))
                localization_times.append(timing.get("localization_s", 0))
            for name, count in processed.get("calls", {}).items():
                task_calls[name] = task_calls.get(name, 0) + count
            artifact = processed.get("provenance", {}).get("artifact_directory")
            if artifact:
                artifact = Path(artifact)
                worker_result = artifact / "result.json"
                if worker_result.exists():
                    raw_result = json.loads(worker_result.read_text())
                    if raw_result.get("peak_cuda_allocated_bytes") is not None:
                        peak_memory.append(raw_result["peak_cuda_allocated_bytes"])
                    request_times.append((artifact / "request.json").stat().st_mtime)
                    completion_times.append(worker_result.stat().st_mtime)
            targets = annotation_by[row["key"]]
            parsed_categories = qa["result"].get("parsed") or {}
            chosen = parsed_categories.get("categories", [])
            raw_category_count += len(parsed_categories.get("raw_categories", []))
            background_categories += len(
                parsed_categories.get("filtered_categories", [])
            )
            for target in targets:
                family = (
                    ("sign", "label", "marker")
                    if target["category"] == "shelf_marker"
                    else ("box", "boxes", "container", "bin", "crate")
                )
                selection_total += 1
                selection_hits += int(
                    any(any(word in c for word in family) for c in chosen)
                )
            pred = []
            for instance in processed["instances"]:
                with np.load(
                    path.parent / "segmentation" / instance["mask_file"],
                    allow_pickle=False,
                ) as data:
                    mask = data["mask"]
                readings = instance.get("texts", [])
                text = " | ".join(
                    t["normalized"]
                    for t in sorted(
                        readings,
                        key=lambda t: (
                            round(min(p[1] for p in t["polygon"]) / 12),
                            min(p[0] for p in t["polygon"]),
                        ),
                    )
                    if t.get("status") == "bound"
                )
                pred.append((instance, mask, text))
            pair = []
            masks = []
            for ti, target in enumerate(targets):
                with np.load(
                    root / "evaluation" / target["mask_file"], allow_pickle=False
                ) as data:
                    tm = data["mask"]
                masks.append(tm)
                for pi, (_, mask, _) in enumerate(pred):
                    iou = float((mask & tm).sum() / max(1, (mask | tm).sum()))
                    pair.append((iou, ti, pi))
            assignments = {}
            used = set()
            for iou, ti, pi in sorted(pair, reverse=True):
                if iou >= 0.5 and ti not in assignments and pi not in used:
                    assignments[ti] = (pi, iou)
                    used.add(pi)
            target_rows = []
            for ti, target in enumerate(targets):
                found = ti in assignments
                item = dict(
                    uid=target["uid"],
                    expected=target["text"],
                    readable=target["readable"],
                    identity_expected_unique=target["identity_expected_unique"],
                    detected=found,
                    iou=0.0,
                    recognized=None,
                    strict_text_correct=False,
                    text_correct=False,
                    accepted=False,
                    landmark_uid=None,
                    oracle_visible_surface_p95_m=None,
                )
                if found:
                    pi, iou = assignments[ti]
                    instance, mask, text = pred[pi]
                    item.update(
                        iou=iou,
                        recognized=text,
                        strict_text_correct=normalize_text(text)
                        == normalize_text(target["text"]),
                        text_correct=reading_key(text) == reading_key(target["text"]),
                        accepted=instance["result"]["status"] == "accepted",
                        landmark_uid=instance["result"]["landmark_uid"],
                    )
                    depth = np.load(row["source"]["depth"], allow_pickle=False)
                    reference = project_depth(
                        np.where(masks[ti], depth, 0),
                        row["source"]["K"],
                        row["source"]["T_body_camera"],
                    )
                    reference = transform_points(
                        reference, row["source"]["T_odom_body"]
                    )
                    with np.load(
                        path.parent / "segmentation" / instance["points_file"],
                        allow_pickle=False,
                    ) as points_file:
                        measured = points_file["points"]
                    if len(reference) and len(measured):
                        item["oracle_visible_surface_p95_m"] = float(
                            np.percentile(cKDTree(reference).query(measured)[0], 95)
                        )
                    if item["landmark_uid"]:
                        key = (row["variant"], item["landmark_uid"])
                        uid_targets.setdefault(key, set()).add(target["uid"])
                        uid_predictions.setdefault(key, []).append(item)
                target_rows.append(item)
            rows.append(
                dict(
                    key=row["key"],
                    variant=row["variant"],
                    missing=False,
                    targets=target_rows,
                    instances=len(pred),
                    unmatched_instances=len(pred) - len(used),
                    error=processed["error"],
                    failures=processed.get("failures", []),
                    module_timings=timing,
                    raw_categories=(qa["result"].get("parsed") or {}).get(
                        "raw_categories", []
                    ),
                )
            )
        for variant in {r["variant"] for r in evaluation_rows}:
            map_path = root / backend / "maps" / variant
            if not (map_path / "map.sqlite").exists():
                continue
            store = MapStore(map_path)
            for landmark in store.records("landmarks"):
                identity_states.update([landmark.get("identity_status", "unknown")])
                reobservation_reasons.update(landmark.get("reobservation_reasons", []))
                if landmark.get("identity_status") not in (
                    "identity_confirmed",
                    "duplicate_text",
                ):
                    continue
                confirmed += 1
                observations = uid_predictions.get(
                    (variant, landmark["landmark_uid"]), []
                )
                if not observations:
                    unscored_confirmed += 1
                elif any(
                    reading_key(landmark["identity_text"]) != reading_key(o["expected"])
                    for o in observations
                ):
                    wrong_confirmed += 1
                elif all(o["identity_expected_unique"] for o in observations):
                    confirmed_correct_unique_landmarks += 1
                    confirmed_correct_unique_targets.update(
                        (variant, o["uid"]) for o in observations
                    )
            store.close()
        false_merge = sum(len(uids) > 1 for uids in uid_targets.values())
        targets = [t for r in rows for t in r.get("targets", [])]
        readable = [
            t
            for t in targets
            if t["readable"] is True and t["identity_expected_unique"]
        ]
        readable_unique_physical_targets = {
            (r["variant"], t["uid"])
            for r in rows
            for t in r.get("targets", [])
            if t["readable"] is True and t["identity_expected_unique"]
        }
        correct = [
            t for t in readable if t["detected"] and t["text_correct"] and t["accepted"]
        ]
        correct_landmarks, correct_physical = set(), set()
        for row in rows:
            for target in row.get("targets", []):
                if (
                    target["readable"] is True
                    and target["identity_expected_unique"]
                    and target["detected"]
                    and target["text_correct"]
                    and target["accepted"]
                ):
                    correct_landmarks.add((row["variant"], target["landmark_uid"]))
                    correct_physical.add((row["variant"], target["uid"]))

        def latency(values):
            return {
                "p50_s": float(np.median(values)) if values else None,
                "p95_s": float(np.percentile(values, 95)) if values else None,
                "samples": len(values),
            }

        per_variant = {}
        for variant in sorted({r["variant"] for r in evaluation_rows}):
            variant_targets = [
                t for r in rows if r["variant"] == variant for t in r.get("targets", [])
            ]
            group = [
                t
                for t in variant_targets
                if t["readable"] is True and t["identity_expected_unique"]
            ]
            readings = [t for t in group if t["recognized"]]
            matched = [t for t in variant_targets if t["detected"]]
            per_variant[variant] = dict(
                target_observations=len(variant_targets),
                matched_instances=len(matched),
                instance_recall=len(matched) / len(variant_targets)
                if variant_targets
                else None,
                matched_mask_iou_mean=float(np.mean([t["iou"] for t in matched]))
                if matched
                else None,
                readable_targets=len(group),
                strict_text_accuracy=sum(t["strict_text_correct"] for t in group)
                / len(group)
                if group
                else None,
                readable_rejection_rate=sum(not t["recognized"] for t in group)
                / len(group)
                if group
                else None,
                correct_text_and_binding_rate_among_readings=sum(
                    t["text_correct"] for t in readings
                )
                / len(readings)
                if readings
                else None,
                perception_latency=latency(
                    [
                        r["module_timings"]["total_s"]
                        for r in rows
                        if r["variant"] == variant and r.get("module_timings")
                    ]
                ),
                correct_3d=sum(
                    t["detected"] and t["text_correct"] and t["accepted"] for t in group
                ),
            )
        reviewed = sum(t["readable"] is not None for t in targets)
        complete = len(rows) == 144 and not any(r["missing"] for r in rows)
        gate = bool(
            complete
            and reviewed == len(targets)
            and wrong_confirmed == 0
            and false_merge == 0
            and unscored_confirmed == 0
            and per_variant["color"]["correct_3d"] > 0
            and per_variant["surface"]["correct_3d"] > 0
        )
        reports[backend] = dict(
            gate_pass=gate,
            complete=complete,
            frames=sum(not r["missing"] for r in rows),
            planned_frames=len(rows),
            qa_valid=qa_valid,
            vlm_selection=dict(
                annotated_target_category_family_coverage=selection_hits
                / selection_total
                if selection_total
                else None,
                background_category_fraction=background_categories / raw_category_count
                if raw_category_count
                else None,
                scope="category-family proxy on annotated targets; not instance grounding or usefulness precision",
                retained_category_usefulness_precision=None,
            ),
            targets=len(targets),
            reviewed_targets=reviewed,
            readable_unique_targets=len(readable),
            correct_3d=len(correct),
            correct_3d_landmark_uids=len(correct_landmarks),
            correct_3d_unique_physical_targets=len(correct_physical),
            correct_3d_coverage=len(correct) / len(readable) if readable else 0.0,
            detected_target_recall=sum(t["detected"] for t in targets) / len(targets)
            if targets
            else None,
            matched_mask_iou_mean=float(
                np.mean([t["iou"] for t in targets if t["detected"]])
            )
            if any(t["detected"] for t in targets)
            else None,
            strict_readable_text_accuracy=sum(
                t["strict_text_correct"] for t in readable
            )
            / len(readable)
            if readable
            else None,
            readable_text_accuracy=sum(t["text_correct"] for t in readable)
            / len(readable)
            if readable
            else None,
            correct_text_and_binding_rate_among_readings=(
                sum(t["text_correct"] for t in readable if t["recognized"])
                / sum(bool(t["recognized"]) for t in readable)
                if any(t["recognized"] for t in readable)
                else None
            ),
            readable_rejection_rate=sum(not t["recognized"] for t in readable)
            / len(readable)
            if readable
            else None,
            character_error_rate=sum(
                edit_distance(
                    reading_key(t["recognized"] or ""), reading_key(t["expected"])
                )
                for t in readable
            )
            / max(1, sum(len(reading_key(t["expected"])) for t in readable)),
            confirmed=confirmed,
            confirmed_correct_unique_landmarks=confirmed_correct_unique_landmarks,
            readable_unique_physical_targets=len(readable_unique_physical_targets),
            confirmed_unique_identity_coverage=(
                len(confirmed_correct_unique_targets & readable_unique_physical_targets)
                / len(readable_unique_physical_targets)
                if readable_unique_physical_targets
                else None
            ),
            wrong_confirmed=wrong_confirmed,
            unscored_confirmed=unscored_confirmed,
            false_merge=false_merge,
            latency=dict(
                perception=latency(times),
                qa=latency(qa_times),
                segmentation=latency(seg_times),
                text=latency(text_times),
                localization=latency(localization_times),
            ),
            actual_task_calls=task_calls,
            predicted_instance_observations=instance_total,
            admission_states=dict(admission_states),
            text_binding_states=dict(binding_states),
            persistent_identity_states=dict(identity_states),
            reobservation_reasons=dict(reobservation_reasons),
            perception_failure_reasons=dict(perception_failures),
            cuda_peak_allocated_bytes=max(peak_memory) if peak_memory else None,
            cuda_memory_scope="Peak PyTorch allocated bytes in one perception worker; not total GPU reservation across processes.",
            offline_completed_perceptions=len(completion_times),
            offline_aggregate_completion_hz=(
                len(completion_times) / (max(completion_times) - min(request_times))
                if completion_times and max(completion_times) > min(request_times)
                else None
            ),
            offline_workers=sorted(worker_counts),
            timing_scope="scene-shard workers share one GPU; not live single-worker throughput",
            variants=per_variant,
            rows=rows,
        )
    eligible = [name for name, r in reports.items() if r["gate_pass"]]
    winner = None
    if len(eligible) == 1:
        winner = eligible[0]
    elif len(eligible) == 2:
        a, b = eligible
        if (
            abs(reports[a]["correct_3d_coverage"] - reports[b]["correct_3d_coverage"])
            > 0.02
        ):
            winner = max(eligible, key=lambda n: reports[n]["correct_3d_coverage"])
        else:
            winner = min(
                eligible, key=lambda n: reports[n]["latency"]["perception"]["p95_s"]
            )
    result = dict(
        winner=winner,
        backends=reports,
        live_rollout_eligible=winner is not None,
        limitations=[
            "Mask/identity metrics cover independently annotated installed targets, not all warehouse objects.",
            "Unmatched object instances are unscored, not assumed false positives.",
            "Identity consistency does not establish static anchors.",
            "correct_3d counts correctly read, admitted source observations; it is not the count of cross-frame confirmed identities.",
            "False merge and identity-error audits cover matched annotated targets; unknown ordinary objects remain unscored.",
            "Oracle visible-surface distances use the same recorded depth/pose and independent instance masks; they measure mask-surface agreement, not independent calibration accuracy.",
            "No absolute geometric accuracy reported without separate visible-surface truth comparison.",
        ],
    )
    write_json(root / "comparison.json", result)
    print(
        "COMPARISON",
        winner,
        {
            n: {
                k: r[k]
                for k in (
                    "gate_pass",
                    "correct_3d",
                    "confirmed",
                    "wrong_confirmed",
                    "false_merge",
                )
            }
            for n, r in reports.items()
        },
        flush=True,
    )
    return result
