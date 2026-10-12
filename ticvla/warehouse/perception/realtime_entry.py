"""Single-frame/replay facade over the exact live mask/text consumers."""

import json
from pathlib import Path
from .realtime import (
    Selector,
    FixedSegmenter,
    MobileReader,
    consume_masks,
    consume_text,
    save_frame,
)
from ..mapping import MapStore
from ..mapping.semantic_frame import write_json
from ..run_landmark_benchmark import source_from_frame
from ..schema import file_sha256


def run_frame(args, saved_response=None):
    if getattr(args, "selection_mode", "instances") == "instances":
        return run_instance_frame(args, saved_response)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    profile = json.loads(Path(args.realtime_profile).read_text())
    store = MapStore(root / "map")
    try:
        source = store.add_observation(source_from_frame(args.episode, args.frame))
        if saved_response:
            qa = json.loads(Path(saved_response).read_text())
            if (
                qa["source_image_id"] != source["image_id"]
                or qa["rgb_hash"] != source["rgb_hash"]
                or qa.get("depth_hash") != source["depth_hash"]
            ):
                raise ValueError("Saved selector source/hash mismatch")
            meta = json.loads(Path(args.episode).read_text())
            for key, p in [
                ("episode_sha256", Path(args.episode)),
                ("frames_sha256", Path(args.episode).parent / meta["frames_file"]),
            ]:
                if qa.get(key) != file_sha256(p):
                    raise ValueError("Saved calibration/pose metadata mismatch")
        else:
            model = Selector(args.model, args.device)
            qa = model.predict(source, args.task)
            del model
            import torch

            torch.cuda.empty_cache()
            meta = json.loads(Path(args.episode).read_text())
            qa.update(
                depth_hash=source["depth_hash"],
                episode_sha256=file_sha256(args.episode),
                frames_sha256=file_sha256(
                    Path(args.episode).parent / meta["frames_file"]
                ),
            )
        write_json(root / "response.json", qa)
        selection = {
            **(
                qa["parsed"]
                if qa["valid"]
                else dict(selected_categories=[], proposed_categories=[])
            ),
            "request_id": qa["request_id"],
            "source_image_id": qa["source_image_id"],
            "source_time": qa["source_time"],
            "status": "valid" if qa["valid"] else "failed",
        }
        seg = FixedSegmenter(profile["weights"])
        batch = seg.predict(source)
        rows = consume_masks(store, source, batch, selection)
        ocr = MobileReader().predict(source)
        binding = consume_text(store, source, batch, rows, ocr)
        save_frame(root, source, batch, rows, selection, ocr, binding)
        from DynaNav.warehouse_loop.realtime_mapping import render
        from DynaNav.warehouse_loop.live_mapping import LiveBEV

        render(
            root,
            source,
            batch,
            rows,
            store.landmark_view()["landmarks"],
            selection,
            qa,
            {**binding, "source_image_id": source["image_id"]},
        )
        bev = LiveBEV()
        bev.add(store.points(source), source["T_odom_body"])
        bev.render(store.landmark_view()["landmarks"], root / "map.png")
        write_json(root / "landmark-view.json", store.landmark_view())
        write_json(
            root / "report.json",
            dict(
                source=source,
                llm_calls=0 if saved_response else 1,
                structured_inference_executed=False,
                accepted_instances=sum(
                    r["result"]["status"] == "accepted" for r in rows
                ),
                text_updates=len(binding["updates"]),
                all_landmarks_tentative=all(
                    lm["status"] == "tentative" for lm in store.records("landmarks")
                ),
                segmentation_s=batch["elapsed_s"],
                ocr_s=ocr["elapsed_s"],
                map_version=store.version,
            ),
        )
    finally:
        store.close()


def run_instance_frame(args, saved_response=None):
    import gc
    import time
    import torch
    from .instance_selection import candidate_request, predict, selection_result
    from DynaNav.warehouse_loop.realtime_mapping import render
    from DynaNav.warehouse_loop.live_mapping import LiveBEV
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    profile = json.loads(Path(args.realtime_profile).read_text())
    store = MapStore(root / "map")
    try:
        source = store.add_observation(source_from_frame(args.episode, args.frame))
        seg = FixedSegmenter(profile["weights"], getattr(args, "device", "cuda:0"))
        batch = seg.predict(source)
        del seg
        gc.collect()
        torch.cuda.empty_cache()
        request = candidate_request(source, batch, root / "qa")
        batch["candidate_request"] = request
        batch["prepared_projections"] = {i:store.prepare_projection(source["image_id"],x["mask"]) for i,x in enumerate(batch["instances"])}
        if saved_response:
            qa = json.loads(Path(saved_response).read_text())
            if qa.get("batch_id") != request["batch_id"]:
                raise ValueError("Saved response source/detection batch mismatch")
        else:
            model = Selector(args.model, args.device)
            qa = predict(model, source, request)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        write_json(root / "response.json", qa)
        selection = selection_result(qa)
        start = time.time()
        rows = consume_masks(store, source, batch, selection)
        map_s = time.time() - start
        ocr = MobileReader().predict(source)
        binding = consume_text(store, source, batch, rows, ocr)
        save_frame(root, source, batch, rows, selection, ocr, binding)
        render(root, source, batch, rows, store.landmark_view()["landmarks"], selection, qa,
               {**binding, "source_image_id":source["image_id"]})
        bev = LiveBEV()
        bev.add(store.points(source),source["T_odom_body"])
        bev.render(store.landmark_view()["landmarks"], root / "map.png")
        write_json(root / "landmark-view.json", store.landmark_view())
        write_json(root / "report.json", dict(selection_mode="instances",navigation_task_included=False,
             source=source,llm_calls=0 if saved_response else 1,structured_inference_executed=False,
             segmentation_s=batch["elapsed_s"],qa_s=qa["elapsed_s"],ocr_s=ocr["elapsed_s"],map_s=map_s,
             accepted_instances=sum(x["result"]["status"]=="accepted" for x in rows),
             text_updates=len(binding["updates"]),store=store.summary()))
    finally:
        store.close()
