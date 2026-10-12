"""Replay a saved objects-only answer against another independent perception map."""

import argparse
import json
from pathlib import Path
import shutil
import time

from .mapping import MapStore, SemanticEvent
from .mapping.semantic_frame import consume_categories, write_json
from .mapping.semantic_visualization import render_semantic_frame
from .perception import PerceptionClient, add_arguments
from .run_landmark_benchmark import source_from_frame
from .schema import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--frame", required=True, type=int)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    add_arguments(parser)
    args = parser.parse_args()
    if not args.perception_backend:
        parser.error("--perception-backend is required")
    if args.perception_backend == "yoloe-realtime":
        from .perception.realtime_entry import run_frame
        return run_frame(args, saved_response=args.response)
    response = json.loads(args.response.read_text())
    request = response["request"]
    episode_path = Path(args.episode)
    metadata = json.loads(episode_path.read_text())
    for field, path in (
        ("episode_sha256", episode_path),
        ("frames_sha256", episode_path.parent / metadata["frames_file"]),
    ):
        if request.get(field) != file_sha256(path):
            raise ValueError("Saved QA calibration/pose metadata changed: " + field)
    source = source_from_frame(args.episode, args.frame)
    if response.get("inference_mode") != "OBJECTS_ONLY":
        raise ValueError("Only saved objects-only QA is accepted")
    if request["source_image_id"] != source.image_id:
        raise ValueError("Saved QA belongs to another image")
    for field in ("rgb", "depth"):
        if (
            file_sha256(getattr(source, field))
            != request["source_hashes"][field + "_hash"]
        ):
            raise ValueError("Saved QA source content changed")
    events = [SemanticEvent(**e) for e in response.get("semantic_events", [])]
    if any(e.image_id != source.image_id for e in events):
        raise ValueError("Saved QA event source mismatch")
    if not response["valid"] and events:
        raise ValueError("Invalid QA cannot supply events")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.response, args.output / "shared-qa-response.json")
    store = MapStore(args.output / "map")
    backend = None
    try:
        record = store.add_observation(source)
        start = time.perf_counter()
        backend = PerceptionClient(
            args.perception_backend,
            args.output / "perception",
            "/data/ticvla/models/yoloe-v8l-seg.pt",
            python=args.perception_python,
            florence_model=args.florence_model,
        )
        load_s = time.perf_counter() - start
        processed = consume_categories(
            store, events, backend, args.output / "segmentation"
        )
        before = store.summary()
        consume_categories(store, events, None, args.output / "segmentation")
        if before != store.summary():
            raise AssertionError("Replay added duplicate evidence")
        report = dict(
            source=record,
            processed=processed,
            llm_calls_this_run=0,
            shared_original_llm_calls=response["llm_calls"],
            qa_sha256=file_sha256(args.response),
            model_load_s=load_s,
            map_version=store.version,
            replay_unchanged=True,
            end_to_end_pass=bool(
                response["valid"]
                and any(
                    i["result"]["status"] == "accepted" for i in processed["instances"]
                )
            ),
        )
        write_json(args.output / "report.json", report)
        render_semantic_frame(
            store,
            source.image_id,
            processed,
            response,
            args.output,
            weight_mode="shared_internvl_base_answer",
        )
        print("REPLAY", args.perception_backend, report["end_to_end_pass"], flush=True)
    finally:
        if backend is not None:
            backend.close()
        store.close()


if __name__ == "__main__":
    main()
