"""V1-V3 actual segmentation/map validation; privileged data is evaluator-only.

Run from repository root with the warehouse-io Python environment.
"""

import argparse
import json
import shutil
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ticvla.warehouse.geometry import planar_pose
from ticvla.warehouse.mapping import MapStore, SemanticEvent, SourceObservation
from ticvla.warehouse.mapping.cli import export_view, import_episode
from ticvla.warehouse.mapping.segmentation import YOLOESegmenter
from ticvla.warehouse.mapping.store import semantic_points
from ticvla.warehouse.schema import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--prompts", default="configs/warehouse_mapping_prompts.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, capture = Path(args.output).resolve(), Path(args.capture).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.prompts).read_text())
    thresholds = config["acceptance"]
    report = {
        "purpose": "fixed-fixture engineering verification, not semantic generalization",
        "capture": json.loads((capture / "capture-report.json").read_text()),
        "thresholds": thresholds,
        "prompt_sha256": file_sha256(args.prompts),
        "V0_capture_pass": True,
    }
    store = MapStore(root / "map")
    oracle = MapStore(root / "oracle-map", oracle=True)
    try:
        ids = import_episode(store, capture / "raw/episode.json")
        import_episode(oracle, capture / "raw/episode.json")
        backend = YOLOESegmenter(args.weights)
        report["segmenter"] = backend.provenance
        rows, masks, sample_events = [], {}, []
        for index in (0, len(ids) - 1):
            obs, depth = store.source_arrays(ids[index])
            # Only evaluator reads supervision. The segmenter sees obs + manual ROI.
            with np.load(
                capture / f"raw/supervision/{index:08d}.npz", allow_pickle=False
            ) as gt:
                labels = json.loads(bytes(gt["instance_labels_json"]).decode())
                gt_ids = gt["instance_ids"].copy()
            rgb = np.array(Image.open(obs["rgb"]).convert("RGB"))
            overlay = rgb.copy()
            for i, spec in enumerate(config["objects"]):
                event = SemanticEvent(
                    f"manual-{index}-{i}",
                    ids[index],
                    spec["category"],
                    roi=spec["roi"],
                    name=spec["name"],
                    static_confirmed=True,
                )
                sample_events.append(event)
                mask, confidence, provenance = backend.segment(obs, event)
                result = store.consume(event, mask, confidence, segmentation=provenance)
                truth = np.isin(
                    gt_ids,
                    [int(k) for k, v in labels.items() if v == spec["evaluation_prim"]],
                )
                gt_event = replace(
                    event, message_id=f"oracle-{index}-{i}", source="oracle"
                )
                gt_result = oracle.consume(
                    gt_event,
                    truth,
                    1.0,
                    segmentation={"backend": "Isaac instance segmentation Oracle"},
                )
                iou, error = 0.0, None
                if mask is not None:
                    np.save(root / f"mask-{index}-{i}.npy", mask, allow_pickle=False)
                    iou = float((mask & truth).sum() / max((mask | truth).sum(), 1))
                    xyz = semantic_points(mask, depth, obs, store.config)
                    gt_xyz = semantic_points(truth, depth, obs, store.config)
                    if len(xyz) and len(gt_xyz):
                        error = float(np.linalg.norm(xyz.mean(0) - gt_xyz.mean(0)))
                    color = np.array(((255, 120, 0), (0, 255, 120), (255, 0, 255))[i])
                    overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
                    if index == 0:
                        masks[i] = (mask.copy(), confidence, provenance)
                rows.append(
                    {
                        "image_id": ids[index],
                        "object": spec["name"],
                        "status": result["status"],
                        "reason": result["reason"],
                        "mask_iou": iou,
                        "surface_centroid_error_m": error,
                        "oracle_status": gt_result["status"],
                        "confidence": confidence,
                    }
                )
            Image.fromarray(overlay).save(root / f"overlay-{index}.png")
        report["instances"] = rows
        report["V2_pass"] = all(
            r["status"] == "accepted"
            and r["mask_iou"] >= thresholds["minimum_mask_iou"]
            and r["surface_centroid_error_m"] is not None
            and r["surface_centroid_error_m"]
            <= thresholds["maximum_surface_centroid_error_m"]
            for r in rows
        )
        # Independent analytic USD box surfaces catch calibration/coordinate errors
        # that comparing two masks through the same projection function would miss.
        surface_checks = []
        for label, source_store in (("oracle", oracle), ("yoloe", store)):
            for lm in source_store.records("landmarks"):
                spec = next(x for x in config["objects"] if x["name"] == lm["name"])
                lower, upper = np.asarray(spec["evaluation_bounds"])
                xyz = source_store.points(lm)
                outside = np.linalg.norm(
                    np.maximum(np.maximum(lower - xyz, xyz - upper), 0), axis=1
                )
                inside = np.min(np.minimum(xyz - lower, upper - xyz), axis=1)
                errors = np.where(outside > 0, outside, inside)
                surface_checks.append(
                    {
                        "backend": label,
                        "object": lm["name"],
                        "surface_p95_error_m": float(np.percentile(errors, 95)),
                    }
                )
        report["analytic_geometry"] = surface_checks
        report["V1_projection_pass"] = (
            len(surface_checks) == 2 * len(config["objects"])
            and all(r["oracle_status"] == "accepted" for r in rows)
            and all(
                r["surface_p95_error_m"]
                <= thresholds["maximum_geometry_surface_p95_error_m"]
                for r in surface_checks
            )
        )
        version = store.version
        duplicate = store.consume(sample_events[0])
        assert duplicate["duplicate"] and store.version == version
        assert len(store.records("landmarks")) == len(config["objects"]), (
            "Cross-frame association duplicated landmarks"
        )
        missing = store.consume(SemanticEvent("invalid-source", "missing", "box"))
        no_detection = store.consume(
            SemanticEvent("no-detection", ids[0], "unseen-object")
        )
        topology = store.consume(
            SemanticEvent("relation", ids[0], "end of aisle", kind="relation")
        )
        report["negative_events"] = [missing, no_detection, topology]
        assert all(x["status"] == "rejected" for x in report["negative_events"])
        assert store.version == version
        mapview = export_view(store, ids[-1], root / "view")
        export_view(oracle, ids[-1], root / "oracle-view")
        view = store.view(ids[-1])
        report["occupied_cells"] = int((view["grid"] == 1).sum())
        assert report["occupied_cells"] > 0
        saved = store.summary()
        store.close()
        store = MapStore(root / "map")
        assert store.summary() == saved
        # Copy for destructive fault injection; keep the deployment input map intact.
        store.close()
        shutil.copytree(root / "map", root / "registration-map")
        store = MapStore(root / "map")
        replay = MapStore(root / "registration-map")
        try:
            source = store.get("observations", ids[0])
            fields_set = {f.name for f in fields(SourceObservation)}
            original = SourceObservation(
                **{k: v for k, v in source.items() if k in fields_set}
            )
            revisit = replace(
                original, image_id="revisit:0", t_obs=100.0, task_id="second-task"
            )
            replay.add_observation(revisit)
            for i, spec in enumerate(config["objects"]):
                mask, confidence, provenance = masks[i]
                replay.consume(
                    SemanticEvent(
                        f"revisit-{i}",
                        revisit.image_id,
                        spec["category"],
                        roi=spec["roi"],
                        name=spec["name"],
                        static_confirmed=True,
                    ),
                    mask,
                    confidence,
                    segmentation=provenance,
                )
            assert len(replay.records("landmarks")) == 3
            tf = planar_pose(1.1, -0.7, 0.35)
            shifted = replace(
                original,
                image_id="epoch-b:0",
                odom_epoch="epoch-b",
                submap_id="submap-b",
                task_id="third-task",
                T_odom_body=(
                    np.linalg.inv(tf) @ np.array(original.T_odom_body)
                ).tolist(),
            )
            replay.add_observation(shifted)
            assert (
                replay.register("submap-b", original.submap_id)["status"] == "rejected"
            )
            for i, spec in enumerate(config["objects"]):
                mask, confidence, provenance = masks[i]
                replay.consume(
                    SemanticEvent(
                        f"new-epoch-{i}",
                        shifted.image_id,
                        spec["category"],
                        roi=spec["roi"],
                        name=spec["name"],
                        static_confirmed=True,
                    ),
                    mask,
                    confidence,
                    segmentation=provenance,
                )
            registration = replay.register("submap-b", original.submap_id)
            report["registration"] = registration
            if registration["status"] == "accepted":
                estimated = np.asarray(registration["T_target_source"])
                position_error = float(np.linalg.norm(estimated[:2, 3] - tf[:2, 3]))
                yaw_error = float(
                    abs(np.arctan2(estimated[1, 0], estimated[0, 0]) - 0.35)
                )
                report["registration_errors"] = {
                    "translation_m": position_error,
                    "yaw_rad": yaw_error,
                    "scope": "actual RGB-D/masks replayed with a synthetic odom-origin change; no physical relocation claim",
                }
                replay.revoke(registration["constraint_id"])
                report["V3_pass"] = (
                    position_error
                    <= thresholds["maximum_registration_translation_error_m"]
                    and yaw_error <= thresholds["maximum_registration_yaw_error_rad"]
                    and replay.get("submaps", "submap-b")["T_map_submap"] is None
                )
            else:
                report["V3_pass"] = False
        finally:
            replay.close()
        report["mapview"] = mapview
        report["llm_image_id"] = ids[-1]
        task = json.loads((capture / "raw/episode.json").read_text())["task"]
        (root / "task.json").write_text(json.dumps(task, indent=2))
        assert all(report[k] for k in ("V1_projection_pass", "V2_pass", "V3_pass")), (
            "Mapping acceptance failed"
        )
        print(
            "MAPPING_VALIDATION",
            json.dumps({k: v for k, v in report.items() if k.endswith("pass")}),
            flush=True,
        )
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2))
        store.close()
        oracle.close()


if __name__ == "__main__":
    main()
