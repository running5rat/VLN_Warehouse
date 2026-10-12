"""Live adapter for the single-frame navigation semantics experiment.

Each request has a private input-map snapshot. Only live_mapping writes the
accumulated map; it replays saved masks and repeats association against that map.
"""

from dataclasses import asdict, fields
from pathlib import Path
import time
import traceback


from ticvla.warehouse.mapping import MapStore, SemanticEvent, SourceObservation
from ticvla.warehouse.mapping.semantic_frame import consume_categories, write_json
from ticvla.warehouse.schema import file_sha256, fingerprint
from ticvla.warehouse.object_io import (
    add_mode_argument,
    object_observation,
    run_object_qa,
)
from ticvla.warehouse.perception import (
    PerceptionClient,
    add_arguments as add_perception_arguments,
)


def add_arguments(parser):
    add_mode_argument(parser)
    add_perception_arguments(parser)
    parser.add_argument(
        "--shelf-signs",
        action="store_true",
        help="Add RGB OCR + YOLO ROI masks for installed shelf signs; separate from LLM text prompts",
    )
    parser.add_argument("--semantic-backend", choices=("ocr", "llm"), default="ocr")
    parser.add_argument("--model", default="/data/ticvla/models/InternVL3-1B")
    parser.add_argument(
        "--vlm-checkpoint", default="/data/ticvla/dataset/TIC-VLA-model.ckpt"
    )
    parser.add_argument("--base-model-only", action="store_true")
    parser.add_argument(
        "--object-evidence",
        action="store_true",
        help="Optional first visual QA before structured generation",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--task", default="沿仓库通道前进，利用可见设施辅助定位，并识别需要避让的物体。"
    )


def forwarded_arguments(args):
    result = []
    for name in (
        "semantic_backend",
        "llm_mode",
        "object_policy",
        "model",
        "vlm_checkpoint",
        "max_new_tokens",
        "device",
        "task",
    ):
        result.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    for name in ("base_model_only", "object_evidence", "shelf_signs"):
        if getattr(args, name, False):
            result.append("--" + name.replace("_", "-"))
    for name in ("perception_backend", "perception_python", "florence_model", "realtime_profile", "selection_mode"):
        if getattr(args, name, None):
            result.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return result


def llm_worker(requests, results, weights, root, options):
    # Heavy ML imports and all GPU calls remain in this spawned process.
    try:
        import torch
        from ticvla.warehouse.config import WarehouseConfig
        from ticvla.warehouse.mapping.cli import export_view
        from ticvla.warehouse.mapping.segmentation import YOLOESegmenter
        from ticvla.warehouse.run_semantic_frame import QUESTION
        from ticvla.warehouse.structured_io import (
            StructuredVLM,
            make_observation,
            packet_and_events,
            save_result,
        )

        torch.set_num_threads(2)
        torch.manual_seed(42)
        cfg = WarehouseConfig(
            model_path=options["model"],
            vlm_checkpoint=None
            if options["base_model_only"]
            else options["vlm_checkpoint"],
            device=options["device"],
            max_new_tokens=options["max_new_tokens"],
        )
        started = time.time()
        model = StructuredVLM(cfg)
        backend = (
            PerceptionClient(
                options["perception_backend"],
                Path(root) / "perception",
                weights,
                cfg.device,
                options.get("perception_python"),
                options.get("florence_model"),
            )
            if options.get("perception_backend")
            else YOLOESegmenter(weights, cfg.device)
        )
        reader = None
        if options.get("shelf_signs"):
            from ticvla.warehouse.mapping.markers import ShelfMarkerReader

            reader = ShelfMarkerReader(
                host_tesseract=options.get("host_tesseract", False)
            )
        weight_mode = (
            "internvl_base_control"
            if options["base_model_only"]
            else "ticvla_initialized"
        )
        root = Path(root)
        write_json(
            root / "model-provenance.json",
            dict(
                weight_mode=weight_mode,
                config=cfg.to_dict(),
                checkpoint_sha256=file_sha256(cfg.vlm_checkpoint)
                if cfg.vlm_checkpoint
                else None,
                base_files_sha256={
                    p.name: file_sha256(p)
                    for p in sorted(Path(cfg.model_path).iterdir())
                    if p.is_file() and p.suffix in (".json", ".py", ".safetensors")
                },
                segmentation=backend.provenance,
            ),
        )
        if options["llm_mode"] == "structured":
            torch.save(
                dict(
                    reference=model.decoder.state_dict(),
                    compressor=model.compressor.state_dict(),
                    config=cfg.to_dict(),
                    quality_validated=False,
                ),
                root / "interface-heads.pt",
            )
        results.put(
            dict(
                ready=True,
                backend="llm",
                weight_mode=weight_mode,
                load_s=time.time() - started,
            )
        )
        while True:
            job = requests.get()
            if job is None:
                break
            source = job["source"]
            started = time.time()
            out = root / "requests" / fingerprint(source["image_id"])[:20]
            out.mkdir(parents=True, exist_ok=False)
            timings = {}
            task = dict(
                task_id=source["task_id"],
                instruction=options["task"],
                hard_constraints=["Use observed geometry only."],
            )
            # Deliberately retain the tested one-RGB / one-depth MapView input.
            # This isolated snapshot never contains OCR, prior semantic objects,
            # or Oracle labels, and does not lock the live SQLite writer.
            snapshot = MapStore(out / "input-map-store")
            try:
                record = snapshot.add_observation(
                    SourceObservation(
                        **{
                            field.name: source[field.name]
                            for field in fields(SourceObservation)
                        }
                    )
                )
                for key in ("rgb_hash", "depth_hash"):
                    if record[key] != source[key]:
                        raise ValueError("Live source changed: " + key)
                view = export_view(snapshot, source["image_id"], out / "input-map")
                obs = make_observation(snapshot, source["image_id"], view, task)
                if options["llm_mode"] == "objects-only":
                    obs = object_observation(
                        record,
                        options.get("object_policy", "navigation"),
                        options["task"] if options.get("perception_backend") else None,
                    )
                else:
                    obs["navigation_object_question"] = QUESTION
                timings["input_s"] = time.time() - started
                if options["object_evidence"]:
                    t = time.time()
                    evidence = model.describe_objects(obs)
                    timings["object_evidence_s"] = time.time() - t
                    write_json(out / "object-evidence.json", evidence)
                    if evidence["token_count"] >= evidence["max_new_tokens"]:
                        raise ValueError("Visual evidence may be truncated")
                    obs["visual_object_evidence"] = evidence
                request = dict(
                    request_id=fingerprint(obs)[:24],
                    observation_hash=fingerprint(obs),
                    source_image_id=source["image_id"],
                    source_hashes={k: source[k] for k in ("rgb_hash", "depth_hash")},
                    live_map_version_at_dispatch=job["map_version"],
                    map_version=view["map_version"],
                    map_scope="isolated current-frame geometric MapView",
                    llm_mode=options["llm_mode"],
                    map_used_by_llm=options["llm_mode"] == "structured",
                    weight_mode=weight_mode,
                    config=cfg.to_dict(),
                    object_evidence_used=options["object_evidence"],
                    dispatched_wall=job["dispatched_wall"],
                )
                write_json(out / "observation.json", obs)
                write_json(out / "request.json", request)
                packet, events, state = None, [], None
                if options["llm_mode"] == "objects-only":
                    result, events, qa_timings = run_object_qa(
                        model, obs, request["request_id"], out
                    )
                    timings.update(qa_timings)
                else:
                    prompt, images = model.build_prompt(obs)
                    (out / "prompt.txt").write_text(prompt)
                    write_json(out / "input-images.json", images)
                    t = time.time()
                    try:
                        result = model.infer(
                            obs,
                            mode="TEXT",
                            progress_callback=lambda value: write_json(
                                out / "generation-progress.json", value
                            ),
                        )
                    except (ValueError, RuntimeError) as exc:
                        result = dict(
                            valid=False,
                            error=str(exc),
                            text="",
                            parsed=None,
                            prompt=prompt,
                            inference_mode="TEXT",
                            generated_tokens=0,
                        )
                    timings["llm_inference_s"] = time.time() - t
                    packet, events, state = None, [], None
                    if result["valid"]:
                        packet, events, state = packet_and_events(
                            result,
                            obs,
                            request["request_id"],
                            0,
                            timings["llm_inference_s"],
                        )
                    result.update(
                        llm_calls=1 + int(options["object_evidence"]),
                        structured_inference_executed=True,
                    )
                # Saved snapshot timing follows the offline interface. It is an
                # audit artifact, not a command to the moving robot/controller.
                save_result(out / "llm", request, result, packet, events, state)
                t = time.time()
                processed = consume_categories(
                    snapshot, events, backend, out / "segmentation", timings=timings
                )
                timings["segmentation_and_preview_projection_s"] = time.time() - t
                markers = None
                if reader is not None:
                    from .live_perception import infer_frame, save_marker_predictions

                    t = time.time()
                    markers = infer_frame(record, reader, backend)
                    save_marker_predictions(markers, out / "shelf-signs")
                    timings["shelf_signs_s"] = time.time() - t
                write_json(out / "preview-processed.json", processed)
                timings["total_s"] = time.time() - started
                write_json(out / "timings.json", timings)
                results.put(
                    dict(
                        image_id=source["image_id"],
                        t_obs=source["t_obs"],
                        started=started,
                        finished=time.time(),
                        directory=str(out),
                        events=[asdict(e) for e in events],
                        timings=timings,
                        llm_mode=options["llm_mode"],
                        llm_calls=result["llm_calls"],
                        structured_inference_executed=result[
                            "structured_inference_executed"
                        ],
                        llm_valid=result["valid"],
                        llm_error=result["error"],
                        raw_answer=result["text"],
                        filtered_categories=(result.get("parsed") or {}).get(
                            "filtered_categories", []
                        ),
                        marker_directory=str(out / "shelf-signs")
                        if markers is not None
                        else None,
                        categories=[
                            r["request"]["category"] for r in processed["categories"]
                        ],
                        yolo_calls=processed.get("calls", {}).get("segmentation", 0)
                        if options.get("perception_backend") == "yoloe-ppocr"
                        else (
                            0
                            if options.get("perception_backend")
                            else int(bool(processed["categories"]))
                        ),
                        perception_backend=options.get("perception_backend"),
                        perception_calls=processed.get("calls", {}),
                        detections=len(processed["instances"]),
                        weight_mode=weight_mode,
                    )
                )
            finally:
                snapshot.close()
    except BaseException:
        results.put(dict(error=traceback.format_exc()))
        raise
    finally:
        if "backend" in locals() and hasattr(backend, "close"):
            backend.close()


def apply_llm_predictions(store, result, root):
    """Replay immutable source masks into the accumulated map, without inference."""
    out = Path(result["directory"])
    before = store.version
    processed = consume_categories(
        store,
        [SemanticEvent(**e) for e in result["events"]],
        None,
        out / "segmentation",
    )
    accepted = [
        r for r in processed["instances"] if r["result"]["status"] == "accepted"
    ]
    consistent = all(
        r["reprojection"] and r["reprojection"]["passed"] for r in accepted
    )
    marker_rows, markers = [], None
    if result.get("marker_directory"):
        from .live_perception import load_marker_predictions, apply_predictions

        markers = load_marker_predictions(result["marker_directory"])
        if markers["image_id"] != result["image_id"]:
            raise ValueError("Shelf sign/source image mismatch")
        marker_rows = apply_predictions(store, markers, Path(root), publish=False)
        write_json(out / "shelf-signs" / "processed.json", marker_rows)
    result["marker_accepted_instances"] = sum(
        r["status"] == "accepted" for r in marker_rows
    )
    result["marker_yolo_calls"] = len(markers["predictions"]) if markers else 0
    write_json(
        out / "live-commit.json",
        dict(
            map_version_before=before,
            map_version_after=store.version,
            accepted_instances=len(accepted),
            source_projection_consistent=consistent,
            llm_valid=result["llm_valid"],
            timings=result["timings"],
            categories=processed["categories"],
            shelf_signs=marker_rows,
            live_wall=time.time(),
            quality_validated=False,
        ),
    )
    from .live_inspection import publish_inspection

    source, _ = store.source_arrays(result["image_id"])
    publish_inspection(root, source, result, processed, marker_rows, markers)
    return processed, len(accepted), consistent
