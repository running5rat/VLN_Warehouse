"""Real navigation semantics -> multi-instance masks -> source-aligned 3D evidence."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import WarehouseConfig
from .object_io import (
    add_mode_argument,
    validate_mode_args,
    object_observation,
    run_object_qa,
)
from .mapping import MapStore, SourceObservation
from .mapping.cli import export_view
from .mapping.segmentation import YOLOESegmenter
from .perception import PerceptionClient, add_arguments as add_perception_arguments
from .mapping.semantic_frame import consume_categories, write_json
from .mapping.semantic_visualization import render_semantic_frame
from .schema import file_sha256, fingerprint
from .structured_io import (
    StructuredVLM,
    make_observation,
    packet_and_events,
    save_result,
)

TASK = "沿仓库通道前进，利用可见设施辅助定位，并识别需要避让的物体。"
QUESTION = (
    "Look at the current source RGB. What individual physical objects do you see? "
    "Name up to three different object categories useful for localization or obstacle avoidance. "
    "Do not name the overall scene, building, room, or aisle. List each object category "
    "only once with its navigation role and a short reason about that object. Use short English object "
    "class names suitable for text-prompted segmentation. Only report objects visible "
    "in the source RGB; the geometry map is context, not a source image. "
    "Return up to three relevant categories in landmark, or [] if none are visible."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", type=int, default=200)
    parser.add_argument("--task", default=TASK, help="Navigation instruction text")
    parser.add_argument("--model", default="/data/ticvla/models/InternVL3-1B")
    parser.add_argument(
        "--vlm-checkpoint", default="/data/ticvla/dataset/TIC-VLA-model.ckpt"
    )
    parser.add_argument(
        "--segmenter-weights", default="/data/ticvla/models/yoloe-v8l-seg.pt"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--base-model-only",
        action="store_true",
        help="Explicit diagnostic control: omit TIC-VLA override; never an implicit fallback",
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--object-evidence",
        action="store_true",
        help="Use a saved same-model visual object QA pass before structured decoding",
    )
    parser.add_argument("--output", required=True)
    add_mode_argument(parser)
    add_perception_arguments(parser)
    args = parser.parse_args()
    validate_mode_args(parser, args)
    if args.perception_backend == "yoloe-realtime":
        from .perception.realtime_entry import run_frame
        return run_frame(args)
    if args.base_model_only:
        args.vlm_checkpoint = None
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    episode_path = Path(args.episode).resolve()
    metadata = json.loads(episode_path.read_text())
    frames_path = episode_path.parent / metadata["frames_file"]
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
    if not 0 <= args.frame < len(frames):
        raise ValueError("Frame outside captured episode")
    frame = frames[args.frame]
    if frame.get("depth_convention") != "optical_z_metres":
        raise ValueError("Expected optical Z depth in metres")
    image_id = f"{metadata['episode_id']}:{args.frame:08d}"
    task = dict(
        task_id="single-frame-navigation",
        instruction=args.task,
        hard_constraints=["Use observed geometry only."],
    )
    source = SourceObservation(
        image_id,
        str(episode_path.parent / frame["rgb"]),
        str(episode_path.parent / frame["depth"]),
        frame["K"],
        frame["T_body_camera"],
        frame["T_odom_body"],
        frame["t_obs"],
        frame["odom_epoch"],
        frame["odom_epoch"],
        task["task_id"],
    )
    timings = {}
    start = time.perf_counter()
    store = MapStore(root / "map")
    try:
        record = store.add_observation(source)
        view = export_view(store, image_id, root / "input-map")
        observation = make_observation(store, image_id, view, task)
        if args.llm_mode == "objects-only":
            observation = object_observation(
                record,
                args.object_policy,
                args.task if args.perception_backend else None,
            )
        else:
            observation["navigation_object_question"] = QUESTION
        assert len(observation["rgb_history"]) == 1
        assert not observation.get("map_landmarks")
        write_json(root / "observation.json", observation)
        before_grid = store.view(image_id)["grid"].copy()
        timings["geometry_and_input_s"] = time.perf_counter() - start
        cfg = WarehouseConfig(
            model_path=args.model,
            vlm_checkpoint=args.vlm_checkpoint,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        torch.manual_seed(42)
        request_id = fingerprint(observation)[:24]
        request = dict(
            request_id=request_id,
            observation_hash=fingerprint(observation),
            source_image_id=image_id,
            map_version=view["map_version"],
            config=cfg.to_dict(),
            navigation_object_question=QUESTION
            if args.llm_mode == "structured"
            else None,
            llm_mode=args.llm_mode,
            map_used_by_llm=args.llm_mode == "structured",
            source_hashes={k: record[k] for k in ("rgb_hash", "depth_hash")},
            episode_sha256=file_sha256(episode_path),
            frames_sha256=file_sha256(frames_path),
            checkpoint_sha256=file_sha256(args.vlm_checkpoint)
            if args.vlm_checkpoint
            else None,
            weight_mode="internvl_base_control"
            if args.base_model_only
            else "ticvla_initialized",
            base_model_files_sha256={
                p.name: file_sha256(p)
                for p in sorted(Path(args.model).iterdir())
                if p.is_file() and p.suffix in (".json", ".py", ".safetensors")
            },
        )
        write_json(root / "request.json", request)
        start = time.perf_counter()
        model = StructuredVLM(cfg)
        timings["llm_load_s"] = time.perf_counter() - start
        packet, events, state = None, [], None
        if args.llm_mode == "objects-only":
            result, events, qa_timings = run_object_qa(
                model, observation, request_id, root
            )
            timings.update(qa_timings)
        else:
            if args.object_evidence:
                start = time.perf_counter()
                evidence = model.describe_objects(observation)
                timings["object_evidence_s"] = time.perf_counter() - start
                write_json(root / "object-evidence.json", evidence)
                if evidence["token_count"] >= evidence["max_new_tokens"]:
                    raise ValueError("Visual object evidence may be truncated")
                observation["visual_object_evidence"] = evidence
                write_json(root / "observation.json", observation)
                request_id = fingerprint(observation)[:24]
                request.update(
                    request_id=request_id,
                    observation_hash=fingerprint(observation),
                    object_evidence_used=True,
                )
                write_json(root / "request.json", request)
            torch.save(
                dict(
                    reference=model.decoder.state_dict(),
                    compressor=model.compressor.state_dict(),
                    config=cfg.to_dict(),
                    quality_validated=False,
                ),
                root / "interface-heads.pt",
            )
            prompt, images = model.build_prompt(observation)
            (root / "prompt.txt").write_text(prompt)
            write_json(root / "input-images.json", images)
            start = time.perf_counter()
            try:
                result = model.infer(
                    observation,
                    mode="TEXT",
                    progress_callback=lambda value: write_json(
                        root / "generation-progress.json", value
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
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            timings["llm_inference_s"] = time.perf_counter() - start
            packet, events, state = None, [], None
            if result["valid"]:
                packet, events, state = packet_and_events(
                    result, observation, request_id, 0, timings["llm_inference_s"]
                )
                packet.validate(
                    t_use=packet.t_ready,
                    task_id=task["task_id"],
                    odom_epoch=source.odom_epoch,
                    route_revision=state.route_revision,
                    max_age=timings["llm_inference_s"] + 1,
                )
            result.update(
                llm_calls=1 + int(args.object_evidence),
                structured_inference_executed=True,
            )
        save_result(root / "llm", request, result, packet, events, state)
        print("LLM_RESPONSE", result["text"], flush=True)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        start = time.perf_counter()
        backend = (
            PerceptionClient(
                args.perception_backend,
                root / "perception",
                args.segmenter_weights,
                args.device,
                args.perception_python,
                args.florence_model,
            )
            if args.perception_backend
            else YOLOESegmenter(args.segmenter_weights, args.device)
        )
        timings["segmenter_load_s"] = time.perf_counter() - start
        start = time.perf_counter()
        processed = consume_categories(
            store, events, backend, root / "segmentation", timings=timings
        )
        timings["segmentation_and_projection_s"] = time.perf_counter() - start
        before_replay = store.summary()
        replayed = consume_categories(store, events, backend, root / "segmentation")
        replay_unchanged = before_replay == store.summary()
        assert replay_unchanged
        if hasattr(backend, "close"):
            backend.close()
        del backend
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        start = time.perf_counter()
        render_semantic_frame(
            store, image_id, processed, result, root, weight_mode=request["weight_mode"]
        )
        timings["visualization_s"] = time.perf_counter() - start
        unchanged = np.array_equal(before_grid, store.view(image_id)["grid"])
        landmarks = store.records("landmarks")
        accepted = sum(
            r["result"]["status"] == "accepted" for r in processed["instances"]
        )
        projection_consistent = bool(accepted) and all(
            r["reprojection"] and r["reprojection"]["passed"]
            for r in processed["instances"]
            if r["result"]["status"] == "accepted"
        )
        report = dict(
            source_image_id=image_id,
            request=request,
            timings=timings,
            llm_mode=args.llm_mode,
            llm_calls=result["llm_calls"],
            structured_inference_executed=result["structured_inference_executed"],
            llm_valid=result["valid"],
            llm_error=result["error"],
            categories=processed["categories"],
            instances=processed["instances"],
            segmentation_provenance=processed["provenance"],
            segmentation_error=processed["error"],
            possibly_truncated=processed["possibly_truncated"],
            accepted_instances=accepted,
            source_reprojection_consistent=projection_consistent,
            landmarks=len(landmarks),
            map_version=str(store.version),
            replay_unchanged=replay_unchanged,
            all_instance_replays_duplicate=all(
                r["result"].get("duplicate") for r in replayed["instances"]
            ),
            geometry_occupancy_unchanged=unchanged,
            all_landmarks_tentative=all(
                lm["status"] == "tentative" for lm in landmarks
            ),
            end_to_end_pass=bool(
                result["valid"]
                and accepted
                and unchanged
                and replay_unchanged
                and projection_consistent
            ),
            specified_weights_acceptance_pass=bool(
                not args.base_model_only
                and result["valid"]
                and accepted
                and unchanged
                and replay_unchanged
                and projection_consistent
            ),
            model_accuracy_evaluated=False,
            absolute_3d_error_measured=False,
            mask_iou_measured=False,
            static_map_validated=False,
            visual_review="pending",
            source_clock="offline source observation",
            limitations=[
                "Single frame; observed surface centroids, not full object centers.",
                "No Oracle masks or world truth. No people/cart benchmark.",
                "Navigation outputs skipped in objects-only; structured heads have unverified quality.",
            ],
        )
        write_json(root / "report.json", report)
        print(
            "SEMANTIC_FRAME",
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "end_to_end_pass",
                        "accepted_instances",
                        "landmarks",
                        "timings",
                    )
                }
            ),
            flush=True,
        )
        if not report["end_to_end_pass"]:
            raise SystemExit("No successful real-model 3D instance; see saved outputs")
    finally:
        store.close()


if __name__ == "__main__":
    main()
