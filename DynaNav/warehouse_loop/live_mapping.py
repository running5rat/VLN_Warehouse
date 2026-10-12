"""Incremental single-writer map + isolated semantic process, fed by committed frames."""

import argparse
from collections import deque
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ticvla.warehouse.mapping import MapStore, MappingConfig, SourceObservation
from DynaNav.warehouse_loop.live_metrics import atomic_json, complete_lines, rate_report
from DynaNav.warehouse_loop.live_perception import semantic_worker, apply_predictions


def enqueue_source(pending, record, latest_only=False):
    """Keep the active request untouched; supersede only unsent observations."""
    skipped = list(pending) if latest_only else []
    if latest_only:
        pending.clear()
    pending.append(record)
    return skipped


class LiveBEV:
    """Accumulated observed obstacle surfaces, not a free-space/planning map."""

    def __init__(self):
        self.cells = set()
        self.trail = []
        self.font = ImageFont.truetype("DejaVuSans.ttf", 15)

    def add(self, points, pose):
        p = points[(points[:, 2] > 0.15) & (points[:, 2] < 3.5), :2]
        self.cells.update(map(tuple, np.floor(p / 0.1).astype(int)))
        self.trail.append(np.asarray(pose)[:2, 3].tolist())

    def render(self, landmarks, path, *, compress_level=6):
        points = np.array(list(self.cells), dtype=float).reshape(-1, 2) * 0.1
        trail = np.array(self.trail).reshape(-1, 2)
        all_xy = np.concatenate([points, trail])
        lo = all_xy.min(0) - 2 if len(all_xy) else np.array([-5.0, -5.0])
        hi = all_xy.max(0) + 2 if len(all_xy) else np.array([5.0, 5.0])
        span = max(float((hi - lo).max()), 10.0)
        center = (hi + lo) / 2
        lo = center - span / 2
        size = 800

        def px(xy):
            q = (np.asarray(xy) - lo) / span * (size - 40) + 20
            return np.column_stack([q[:, 0], size - q[:, 1]])

        canvas = Image.new("RGB", (size, size + 45), (246, 248, 250))
        data = np.array(canvas)
        if len(points):
            q = px(points).astype(int)
            data[q[:, 1], q[:, 0]] = [75, 95, 112]
        canvas = Image.fromarray(data)
        draw = ImageDraw.Draw(canvas)
        if len(trail) > 1:
            draw.line(list(map(tuple, px(trail))), fill=(35, 130, 245), width=3)
        if len(trail):
            x, y = px(trail[-1:])[0]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")
        for lm in landmarks:
            if not lm.get("landmark_view", {}).get("visible", True):
                continue
            x, y = px(np.array(lm["position"][:2])[None])[0]
            color = "green" if lm["status"] == "verified" else "orange"
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            draw.text(
                (x + 5, y),
                f"{lm['category']} {lm.get('identity_text') or lm.get('name') or ''} [{lm.get('identity_status', lm['status'])}]"
                + (" [review]" if lm.get("landmark_view", {}).get("decision") == "review" else ""),
                fill=color,
                font=self.font,
                stroke_width=1,
                stroke_fill="white",
            )
        draw.text(
            (12, size + 5),
            "Observed geometry | blue: trajectory | green: verified | orange: tentative",
            fill="black",
        )
        draw.text(
            (12, size + 22),
            f"Odom coordinates; view width {span:.1f} m; 0.10 m display cells",
            fill="black",
        )
        tmp = path.with_suffix(".tmp.png")
        canvas.save(tmp, compress_level=compress_level)
        tmp.replace(path)


def run(capture, root, weights, semantic_hz, host_tesseract, options=None):
    options = options or {}
    if options.get("perception_backend") == "yoloe-realtime":
        from .realtime_mapping import run as realtime_run

        return realtime_run(capture, root, weights, options)
    llm_mode = options.get("semantic_backend") == "llm"
    if semantic_hz <= 0:
        raise ValueError("semantic-hz must be positive")
    root.mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    context = mp.get_context("spawn")
    requests, results = context.Queue(maxsize=1), context.Queue()
    if llm_mode:
        from .live_llm import llm_worker

        worker = context.Process(
            target=llm_worker, args=(requests, results, weights, root, options)
        )
    else:
        worker = context.Process(
            target=semantic_worker, args=(requests, results, weights, host_tesseract)
        )
    worker.start()
    store = MapStore(
        root / "map", MappingConfig(fixed_shelf_markers=True), array_compression_level=1
    )
    bev = LiveBEV()
    geometry_rows, semantic_rows, display_rows, event_rows = [], [], [], []
    pending = deque()
    busy = False
    last_sem = -float("inf")
    last_obs = None
    frame_stream = None
    metadata = None
    backlog_max = 0
    dropped = 0
    start = time.time()
    end = start
    metrics = (root / "metrics.jsonl").open("w", buffering=1)

    def log(module, row):
        metrics.write(json.dumps(dict(module=module, **row)) + "\n")

    try:
        while True:
            dirty = False
            while True:
                try:
                    result = results.get_nowait()
                except queue.Empty:
                    break
                if "error" in result:
                    raise RuntimeError(result["error"])
                if result.get("ready"):
                    atomic_json(root / "ready.json", result)
                    continue
                t = time.time()
                extras = {}
                if llm_mode:
                    from .live_llm import apply_llm_predictions

                    processed, accepted, consistent = apply_llm_predictions(
                        store, result, root
                    )
                    event_rows.extend(r["result"] for r in processed["instances"])
                    extras = dict(
                        llm_mode=result["llm_mode"],
                        perception_backend=result.get("perception_backend"),
                        perception_calls=result.get("perception_calls", {}),
                        perception_timings=processed.get("perception_timings", {}),
                        llm_calls=result["llm_calls"],
                        structured_inference_executed=result[
                            "structured_inference_executed"
                        ],
                        live_projection_and_persistence_s=time.time() - t,
                        llm_valid=result["llm_valid"],
                        marker_accepted_instances=result.get(
                            "marker_accepted_instances", 0
                        ),
                        marker_yolo_calls=result.get("marker_yolo_calls", 0),
                        llm_error=result["llm_error"],
                        accepted_instances=accepted,
                        source_projection_consistent=consistent,
                        timings=result["timings"],
                        categories=result["categories"],
                        directory=result["directory"],
                        weight_mode=result["weight_mode"],
                    )
                else:
                    event_rows.extend(apply_predictions(store, result, root))
                row = dict(
                    wall=time.time(),
                    t_obs=result["t_obs"],
                    image_id=result["image_id"],
                    service_s=time.time() - t + result["finished"] - result["started"],
                    ocr_s=result.get("ocr_seconds", 0),
                    detections=result["detections"]
                    if llm_mode
                    else len(result["detections"]),
                    yolo_calls=result["yolo_calls"]
                    if llm_mode
                    else len(result["predictions"]),
                    **extras,
                    source_lag_s=max(0.0, last_obs["t_obs"] - result["t_obs"])
                    if last_obs
                    else 0.0,
                )
                semantic_rows.append(row)
                log("semantic", row)
                busy = False
                dirty = True
            if not worker.is_alive():
                raise RuntimeError(
                    f"Semantic process stopped unexpectedly: {worker.exitcode}"
                )
            episode = capture / "raw/episode.json"
            if frame_stream is None and episode.exists():
                # Metadata is created before frames.jsonl; wait for both.
                try:
                    metadata = json.loads(episode.read_text())
                    frame_stream = (episode.parent / metadata["frames_file"]).open()
                except (json.JSONDecodeError, FileNotFoundError):
                    pass
            if frame_stream is not None:
                # One frame per iteration keeps semantic results/UI responsive.
                frame = next(complete_lines(frame_stream), None)
                if frame is not None:
                    t = time.time()
                    index = len(geometry_rows)
                    obs = SourceObservation(
                        f"{metadata['episode_id']}:{index:08d}",
                        str(episode.parent / frame["rgb"]),
                        str(episode.parent / frame["depth"]),
                        frame["K"],
                        frame["T_body_camera"],
                        frame["T_odom_body"],
                        frame["t_obs"],
                        frame["odom_epoch"],
                        frame["odom_epoch"],
                        metadata["task"]["task_id"],
                    )
                    record = store.add_observation(obs)
                    bev.add(store.points(record), frame["T_odom_body"])
                    last_obs = record
                    row = dict(
                        wall=time.time(),
                        service_s=time.time() - t,
                        t_obs=obs.t_obs,
                        image_id=obs.image_id,
                        capture_to_geometry_s=time.time()
                        - frame.get("wall_time", time.time()),
                    )
                    geometry_rows.append(row)
                    log("geometry", row)
                    if obs.t_obs - last_sem >= 1 / semantic_hz - 1e-6:
                        for old in enqueue_source(pending, record, llm_mode):
                            dropped += 1
                            log(
                                "semantic_skipped",
                                dict(
                                    wall=time.time(),
                                    image_id=old["image_id"],
                                    replaced_by=record["image_id"],
                                    reason="latest_pending_source",
                                ),
                            )
                        last_sem = obs.t_obs
                    dirty = True
            if pending and not busy:
                source = pending.popleft()
                requests.put(
                    dict(
                        source=source,
                        map_version=str(store.version),
                        dispatched_wall=time.time(),
                    )
                    if llm_mode
                    else source
                )
                busy = True
            backlog_max = max(backlog_max, len(pending) + int(busy))
            if dirty:
                t = time.time()
                landmarks = store.records("landmarks")
                bev.render(landmarks, root / "map.png")
                status = dict(
                    updated_wall=time.time(),
                    semantic_backend="llm" if llm_mode else "ocr",
                    semantic_sample_hz=semantic_hz,
                    semantic_frames_skipped=dropped,
                    latest_semantic=semantic_rows[-1] if semantic_rows else None,
                    source_time=last_obs["t_obs"] if last_obs else None,
                    geometry_frames=len(geometry_rows),
                    semantic_frames=len(semantic_rows),
                    semantic_pending=len(pending) + int(busy),
                    last_semantic_source=semantic_rows[-1]["t_obs"]
                    if semantic_rows
                    else None,
                    verified=sum(lm["status"] == "verified" for lm in landmarks),
                    tentative=sum(lm["status"] != "verified" for lm in landmarks),
                    geometry_hz=rate_report(geometry_rows[-30:], 5, start, time.time())[
                        "completion_wall_hz"
                    ],
                    semantic_hz=rate_report(
                        semantic_rows[-10:], semantic_hz, start, time.time()
                    )["completion_wall_hz"],
                )
                atomic_json(root / "status.json", status)
                row = dict(wall=time.time(), service_s=time.time() - t)
                display_rows.append(row)
                log("map_publish", row)
            done_path = capture.parent / "capture-process.json"
            if done_path.exists():
                done = json.loads(done_path.read_text())
                if done["exit_code"] != 0:
                    raise RuntimeError("Isaac capture failed; see isaac.log")
                count = json.loads((capture / "capture-report.json").read_text())[
                    "frames"
                ]
                if len(geometry_rows) == count and not busy and not pending:
                    break
            if not dirty:
                time.sleep(0.01)
        end = time.time()
        report = dict(
            geometry=rate_report(geometry_rows, 5, start, end),
            semantic=rate_report(semantic_rows, semantic_hz, start, end),
            map_publish=rate_report(display_rows, 5, start, end),
            semantic_backlog_max=backlog_max,
            semantic_source_times=[r["t_obs"] for r in semantic_rows],
            yolo_calls=sum(r["yolo_calls"] for r in semantic_rows),
            perception_calls={
                kind: sum(
                    r.get("perception_calls", {}).get(kind, 0) for r in semantic_rows
                )
                for kind in ("localization", "segmentation", "text_recognition")
            },
            max_semantic_source_lag_s=max(
                (r["source_lag_s"] for r in semantic_rows), default=0
            ),
            capture_to_geometry_p95_s=float(
                np.percentile([r["capture_to_geometry_s"] for r in geometry_rows], 95)
            ),
            semantic_backend="llm" if llm_mode else "ocr",
            semantic_sample_hz=semantic_hz,
            semantic_frames_dropped=dropped,
            llm_valid_frames=sum(r.get("llm_valid", False) for r in semantic_rows),
            llm_mode=options.get("llm_mode") if llm_mode else None,
            perception_backend=options.get("perception_backend"),
            identity_confirmed=sum(
                lm.get("identity_status") == "identity_confirmed"
                for lm in store.records("landmarks")
            ),
            needs_reobservation=sum(
                bool(lm.get("needs_reobservation")) for lm in store.records("landmarks")
            ),
            llm_calls=sum(r.get("llm_calls", 0) for r in semantic_rows),
            structured_inference_executed=any(
                r.get("structured_inference_executed", False) for r in semantic_rows
            ),
            accepted_instances=sum(
                r.get("accepted_instances", 0) for r in semantic_rows
            ),
            marker_accepted_instances=sum(
                r.get("marker_accepted_instances", 0) for r in semantic_rows
            ),
            marker_yolo_calls=sum(r.get("marker_yolo_calls", 0) for r in semantic_rows),
            source_projection_consistent=all(
                r.get("source_projection_consistent", True) for r in semantic_rows
            ),
            weight_mode=semantic_rows[0].get("weight_mode") if semantic_rows else None,
            geometry_frames_dropped=0,
            store=store.summary(),
            events=event_rows,
            policy=(
                "Geometry consumes every frame; LLM uses one active + latest pending sampled source. "
                "Per-request single-frame input-map snapshots; one accumulated-map writer. "
                "No Oracle/OCR inputs or LLM control of motion. Sample rate is not inference throughput."
                if llm_mode
                else "Geometry consumes all committed frames; semantic queues without dropping. One SQLite writer. No Oracle input."
            ),
        )
        atomic_json(root / "report.json", report)
        atomic_json(root / "events.json", event_rows)
        print(
            "LIVE_MAPPING_REPORT",
            json.dumps(
                {k: v for k, v in report.items() if k not in ("events", "store")}
            ),
            flush=True,
        )
    finally:
        if frame_stream:
            frame_stream.close()
        metrics.close()
        store.close()
        if worker.is_alive():
            try:
                requests.put_nowait(None)
            except queue.Full:
                pass
            worker.join(5)
            if worker.is_alive():
                worker.terminate()
                worker.join(5)
        requests.close()
        results.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--semantic-hz", type=float, default=1.0)
    p.add_argument("--host-tesseract", action="store_true")
    from .live_llm import add_arguments

    add_arguments(p)
    args = p.parse_args()
    from ticvla.warehouse.object_io import validate_mode_args

    validate_mode_args(p, args)
    run(
        args.capture,
        args.output,
        args.weights,
        args.semantic_hz,
        args.host_tesseract,
        vars(args),
    )


if __name__ == "__main__":
    main()
