"""Independent geometry, fixed masks, low-rate selector, and asynchronous text."""

import json
import multiprocessing as mp
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw

from ticvla.warehouse.mapping import MapStore, SourceObservation
from ticvla.warehouse.mapping.semantic_frame import write_json
from ticvla.warehouse.perception.realtime import (
    CORE,
    FixedSegmenter,
    Selector,
    MobileReader,
    SelectionState,
    consume_masks,
    consume_text,
    save_frame,
)
from ticvla.warehouse.schema import file_sha256
from .live_metrics import atomic_json, complete_lines, rate_report


def worker(kind, requests, results, weights, options, warm):
    try:
        import torch

        torch.set_num_threads(2)
        started = time.time()
        model = (
            FixedSegmenter(weights)
            if kind == "seg"
            else Selector(options["model"])
            if kind == "qa"
            else MobileReader()
        )
        instance_mode = options.get("selection_mode", "instances") == "instances"
        if kind == "qa" and instance_mode:
            from ticvla.warehouse.perception.instance_selection import candidate_request, predict
            from pathlib import Path
            empty = dict(source_image_id=warm["image_id"],rgb_hash=warm["rgb_hash"],instances=[],provenance={})
            request = candidate_request(warm, empty, Path(warm["rgb"]).parent / "warmup-qa")
            predict(model, warm, request)
        elif kind == "qa":
            model.predict(warm, options["task"])
        else:
            model.predict(warm)
        results.put(
            dict(
                kind=kind,
                ready=True,
                load_and_warm_s=time.time() - started,
                provenance=getattr(model, "provenance", {}),
            )
        )
        while True:
            source = requests.get()
            if source is None:
                break
            started = time.time()
            try:
                data = (
                    predict(model, source, source["candidate_request"])
                    if kind == "qa" and instance_mode else model.predict(source, options["task"])
                    if kind == "qa"
                    else model.predict(source)
                )
                results.put(
                    dict(
                        kind=kind,
                        source=source,
                        data=data,
                        started=started,
                        finished=time.time(),
                        gpu_peak_bytes=torch.cuda.max_memory_allocated()
                        if kind != "ocr"
                        else None,
                    )
                )
            except Exception:
                results.put(
                    dict(
                        kind=kind,
                        source=source,
                        error=traceback.format_exc(),
                        started=started,
                        finished=time.time(),
                    )
                )
    except Exception:
        results.put(dict(kind=kind, fatal=traceback.format_exc()))


def render(root, source, batch, rows, landmarks, selection, qa, text_result):
    rgb = np.array(Image.open(source["rgb"]).convert("RGB"))
    by = {x["landmark_uid"]: x for x in landmarks}
    palette = [
        (40, 180, 255),
        (255, 155, 30),
        (170, 100, 245),
        (60, 200, 100),
        (240, 90, 150),
    ]
    for only, name in [(False, "detections"), (True, "semantic")]:
        out = rgb.copy()
        for i, inst in enumerate(batch["instances"]):
            accepted = rows[i]["result"]["status"] == "accepted"
            if only and not accepted:
                continue
            m = inst["mask"]
            color = palette[i % len(palette)] if accepted else (150, 150, 150)
            out[m] = (out[m] * 0.65 + np.array(color) * 0.35).astype("uint8")
        canvas = Image.fromarray(out)
        draw = ImageDraw.Draw(canvas)
        for i, inst in enumerate(batch["instances"]):
            r = rows[i]
            lm = by.get(r["result"].get("landmark_uid"), {})
            if only and r["result"]["status"] != "accepted":
                continue
            color = palette[i % len(palette)] if r["selected"] else (150, 150, 150)
            draw.rectangle(inst["bbox"], outline=color, width=2)
            text = f"{'d' if selection.get('selection_mode') == 'instances' else ''}{i} {inst['category']} {lm.get('identity_text') or ''} [{lm.get('identity_status', r['result']['status'])}]"
            if lm.get("landmark_view"):
                text += " [view:" + lm["landmark_view"]["decision"] + "]"
            draw.text(
                inst["bbox"][:2], text, fill=color, stroke_width=1, stroke_fill="black"
            )
        temp = root / (name + ".tmp.png")
        canvas.save(temp, compress_level=0)
        temp.replace(root / (name + ".png"))
    instances = []
    for i, r in enumerate(rows):
        lm = by.get(r["result"].get("landmark_uid"), {})
        instances.append(
            dict(
                instance_id=f"d{i}" if selection.get("selection_mode") == "instances" else f"L{i}",
                origin="core vocabulary",
                category=r["category"],
                name=lm.get("identity_text"),
                confidence=r["confidence"],
                bbox=r["bbox"],
                status=r["result"]["status"],
                reason=r["result"].get("reason"),
                landmark_uid=r["result"].get("landmark_uid"),
                landmark_status=lm.get("status"),
                identity_status=lm.get("identity_status"),
                landmark_view=lm.get("landmark_view"),
                resolved_identity=lm.get("resolved_identity"),
                reobservation_reasons=lm.get("reobservation_reasons", []),
            )
        )
    atomic_json(
        root / "inspection.json",
        dict(
            updated_wall=time.time(),
            image_id=source["image_id"],
            t_obs=source["t_obs"],
            source_publish_latency_s=time.time() - source["submitted_wall"]
            if "submitted_wall" in source
            else None,
            raw_answer=(qa or {}).get("text", ""),
            llm_valid=selection.get("status") == "valid",
            llm_error=(qa or {}).get("error"),
            category_selection=selection,
            instances=instances,
            filtered_categories=[],
            perception_failures=[],
            texts=(text_result or {}).get("texts", []),
            text_source_image_id=(text_result or {}).get("source_image_id"),
            categories=[
                dict(
                    request=dict(category=c),
                    status="selected"
                    if any(r["category"] == c and r["selected"] for r in rows)
                    else "not_selected",
                    instances=sum(r["category"] == c for r in rows),
                    accepted=sum(
                        r["category"] == c and r["result"]["status"] == "accepted"
                        for r in rows
                    ),
                    reason=None,
                )
                for c in CORE
            ],
        ),
    )


def run(capture, root, weights, options):
    if options.get("selection_mode", "instances") == "instances":
        from .instance_mapping import run as instance_run
        return instance_run(capture, root, weights, options)
    from .live_mapping import LiveBEV

    root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    results = context.Queue(maxsize=8)
    warm_path = root / "warmup.png"
    Image.new("RGB", (960, 720)).save(warm_path)
    warm = dict(
        image_id="warmup-only",
        rgb=str(warm_path),
        rgb_hash=file_sha256(warm_path),
        depth_hash="warmup",
        t_obs=0,
        T_odom_body=np.eye(4).tolist(),
        odom_epoch="warmup",
    )
    queues = {k: context.Queue(maxsize=1) for k in ("seg", "qa", "ocr")}
    workers = {
        k: context.Process(target=worker, args=(k, q, results, weights, options, warm))
        for k, q in queues.items()
    }
    for w in workers.values():
        w.start()
    store = MapStore(root / "map", array_compression_level=0)
    bev = LiveBEV()
    policy = SelectionState()
    ready = set()
    pending = {k: None for k in queues}
    busy = {k: False for k in queues}
    contexts = {}
    metrics = (root / "metrics.jsonl").open("w", buffering=1)
    rows_by = {k: [] for k in ("geometry", "semantic", "ocr", "qa", "map_publish")}
    drop = {k: 0 for k in queues}
    due = {"qa": -1e10, "ocr": -1e10}
    sampled_ocr = []
    pending_text_publication = []
    frame_stream = None
    metadata = None
    last_source = None
    latest_qa = None
    latest_display = None
    latest_text = None
    start = time.time()
    load_start = start
    count = 0
    archive = ThreadPoolExecutor(max_workers=1)
    futures = []

    def log(kind, row):
        metrics.write(json.dumps(dict(module=kind, **row), allow_nan=False) + "\n")
        if kind in rows_by:
            rows_by[kind].append(row)

    def offer(kind, source):
        if pending[kind] is not None:
            old = pending[kind]
            drop[kind] += 1
            log(
                kind + "_skipped",
                dict(
                    image_id=old["image_id"],
                    replaced_by=source["image_id"],
                    wall=time.time(),
                ),
            )
            if kind == "seg":
                contexts[old["image_id"]]["seg_skipped"] = True
        pending[kind] = source

    def schedule_archive(ctx):
        if not ctx.get("rows") and "rows" not in ctx:
            return
        if ctx.get("archived"):
            return
        if (
            ctx["ocr_due"]
            and not ctx.get("binding_done")
            and not ctx.get("ocr_skipped")
        ):
            return
        ctx["archived"] = True
        futures.append(
            archive.submit(
                save_frame,
                root / "frames" / ctx["source"]["image_id"].replace(":", "_"),
                ctx["source"],
                ctx["batch"],
                ctx["rows"],
                ctx["selection"],
                ctx.get("ocr"),
                ctx.get("binding"),
            )
        )

    def bind(ctx):
        nonlocal latest_text
        if "batch" not in ctx or "ocr" not in ctx or ctx.get("binding_done"):
            return
        t = time.time()
        ctx["binding"] = consume_text(
            store, ctx["source"], ctx["batch"], ctx["rows"], ctx["ocr"]
        )
        ctx["binding_done"] = True
        latest_text = {**ctx["binding"], "source_image_id": ctx["source"]["image_id"]}
        canvas = Image.open(ctx["source"]["rgb"]).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for text in ctx["binding"]["texts"]:
            points = [tuple(p) for p in text["polygon"]]
            color = "lime" if text["status"] == "bound" else "orange"
            draw.line(points + points[:1], fill=color, width=2)
            draw.text(
                points[0],
                f"{text['text']} [{text['status']}]",
                fill=color,
                stroke_width=1,
                stroke_fill="black",
            )
        canvas.save(root / "ocr.tmp.png", compress_level=1)
        (root / "ocr.tmp.png").replace(root / "ocr.png")
        accepted = sum(x["status"] == "accepted" for x in ctx["binding"]["updates"])
        pending_text_publication.append(
            dict(
                wall=time.time(),
                image_id=ctx["source"]["image_id"],
                t_obs=ctx["source"]["t_obs"],
                service_s=ctx["ocr"]["elapsed_s"] + time.time() - t,
                latency_s=time.time() - ctx["source"]["submitted_wall"],
                bound_updates=accepted,
                readings=len(ctx["ocr"]["texts"]),
                submitted_wall=ctx["source"]["submitted_wall"],
            ),
        )
        schedule_archive(ctx)

    try:
        while True:
            dirty = False
            for future in list(futures):
                if future.done():
                    future.result()
                    futures.remove(future)
            if len(futures) > 16:
                raise RuntimeError(
                    "Artifact writer backlog exceeded 16 frames; run not realtime"
                )
            while True:
                try:
                    r = results.get_nowait()
                except queue.Empty:
                    break
                kind = r["kind"]
                if r.get("fatal"):
                    raise RuntimeError(r["fatal"])
                if r.get("ready"):
                    ready.add(kind)
                    write_json(root / (kind + "-provenance.json"), r)
                    if len(ready) == 3:
                        atomic_json(
                            root / "ready.json",
                            dict(
                                ready=True,
                                backend="yoloe-realtime",
                                load_s=time.time() - load_start,
                            ),
                        )
                    continue
                busy[kind] = False
                log(
                    "worker_service",
                    dict(
                        kind=kind,
                        wall=time.time(),
                        service_s=r["finished"] - r["started"],
                        gpu_peak_bytes=r.get("gpu_peak_bytes"),
                        queued_results=results.qsize(),
                        pending_frames=sum(v is not None for v in pending.values()),
                    ),
                )
                source = r["source"]
                ctx = contexts[source["image_id"]]
                if r.get("error"):
                    log(
                        kind + "_failure",
                        dict(
                            wall=time.time(),
                            image_id=source["image_id"],
                            error=r["error"],
                        ),
                    )
                    if kind == "ocr":
                        ctx["ocr_skipped"] = True
                    if kind == "seg":
                        ctx["seg_skipped"] = True
                    continue
                if kind == "qa":
                    latest_qa = r["data"]
                    policy.update(latest_qa)
                    futures.append(
                        archive.submit(
                            write_json,
                            root
                            / "qa"
                            / source["image_id"].replace(":", "_")
                            / "result.json",
                            latest_qa,
                        )
                    )
                    log(
                        "qa",
                        dict(
                            wall=time.time(),
                            image_id=source["image_id"],
                            service_s=r["data"]["elapsed_s"],
                            valid=r["data"]["valid"],
                        ),
                    )
                elif kind == "seg":
                    t = time.time()
                    ctx["batch"] = r["data"]
                    ctx["selection"] = policy.snapshot(source["t_obs"])
                    ctx["rows"] = consume_masks(
                        store, source, r["data"], ctx["selection"]
                    )
                    ctx["projection_s"] = time.time() - t
                    latest_display = ctx
                    bind(ctx)
                    # Timestamp is finalized after map and mask publication below.
                    ctx["pending_publish"] = True
                    dirty = True
                    schedule_archive(ctx)
                else:
                    ctx["ocr"] = r["data"]
                    bind(ctx)
                    dirty = True
            if any(not w.is_alive() for w in workers.values()):
                raise RuntimeError("Realtime inference worker exited")
            episode = capture / "raw/episode.json"
            if frame_stream is None and episode.exists():
                try:
                    metadata = json.loads(episode.read_text())
                    frame_stream = (episode.parent / metadata["frames_file"]).open()
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            if frame_stream:
                f = next(complete_lines(frame_stream), None)
                if f is not None:
                    t = time.time()
                    obs = SourceObservation(
                        f"{metadata['episode_id']}:{count:08d}",
                        str(episode.parent / f["rgb"]),
                        str(episode.parent / f["depth"]),
                        f["K"],
                        f["T_body_camera"],
                        f["T_odom_body"],
                        f["t_obs"],
                        f["odom_epoch"],
                        f["odom_epoch"],
                        metadata["task"]["task_id"],
                    )
                    count += 1
                    source = store.add_observation(obs)
                    source["submitted_wall"] = f["wall_time"]
                    last_source = source
                    bev.add(store.points(source), f["T_odom_body"])
                    log(
                        "geometry",
                        dict(
                            wall=time.time(),
                            image_id=obs.image_id,
                            t_obs=obs.t_obs,
                            service_s=time.time() - t,
                            capture_to_geometry_s=time.time()
                            - source["submitted_wall"],
                        ),
                    )
                    ocr_due = source["t_obs"] - due["ocr"] >= 1 - 1e-6
                    contexts[source["image_id"]] = dict(source=source, ocr_due=ocr_due)
                    offer("seg", source)
                    if source["t_obs"] - due["qa"] >= 2 - 1e-6:
                        offer("qa", source)
                        due["qa"] = source["t_obs"]
                    if ocr_due:
                        old = pending["ocr"]
                        if old:
                            contexts[old["image_id"]]["ocr_skipped"] = True
                            schedule_archive(contexts[old["image_id"]])
                        offer("ocr", source)
                        sampled_ocr.append(source["image_id"])
                        due["ocr"] = source["t_obs"]
                    dirty = True
            for kind in queues:
                if kind in ready and pending[kind] is not None and not busy[kind]:
                    queues[kind].put(pending[kind])
                    busy[kind] = True
                    pending[kind] = None
            if dirty and last_source:
                t = time.time()
                view = store.landmark_view()
                landmarks = view["landmarks"]
                atomic_json(root / "landmark-view.json", view)
                bev.render(landmarks, root / "map.png", compress_level=0)
                if latest_display:
                    c = latest_display
                    render(
                        root,
                        c["source"],
                        c["batch"],
                        c["rows"],
                        landmarks,
                        c["selection"],
                        latest_qa,
                        latest_text,
                    )
                    if c.pop("pending_publish", False):
                        log(
                            "semantic",
                            dict(
                                wall=time.time(),
                                image_id=c["source"]["image_id"],
                                t_obs=c["source"]["t_obs"],
                                service_s=c["batch"]["elapsed_s"]
                                + c["projection_s"]
                                + time.time()
                                - t,
                                latency_s=time.time() - c["source"]["submitted_wall"],
                                projection_s=c["projection_s"],
                                accepted_instances=sum(
                                    x["result"]["status"] == "accepted"
                                    for x in c["rows"]
                                ),
                                detections=len(c["rows"]),
                                selection_status=c["selection"].get("status"),
                            ),
                        )
                atomic_json(
                    root / "status.json",
                    dict(
                        updated_wall=time.time(),
                        source_time=last_source["t_obs"],
                        geometry_frames=count,
                        semantic_frames=len(rows_by["semantic"]),
                        semantic_pending=sum(busy.values())
                        + sum(v is not None for v in pending.values()),
                        geometry_hz=rate_report(
                            rows_by["geometry"][-30:], 5, start, time.time()
                        )["completion_wall_hz"],
                        semantic_hz=rate_report(
                            rows_by["semantic"][-30:], 5, start, time.time()
                        )["completion_wall_hz"],
                        verified=0,
                        tentative=len(landmarks),
                        selection=policy.snapshot(last_source["t_obs"]),
                        text_source_image_id=(latest_text or {}).get("source_image_id"),
                        ocr_completed=len(rows_by["ocr"]),
                        drops=drop,
                    ),
                )
                log("map_publish", dict(wall=time.time(), service_s=time.time() - t))
                for item in pending_text_publication:
                    item["wall"] = time.time()
                    item["latency_s"] = item["wall"] - item.pop("submitted_wall")
                    log("ocr", item)
                pending_text_publication.clear()
            done = capture.parent / "capture-process.json"
            if done.exists():
                if json.loads(done.read_text())["exit_code"] != 0:
                    raise RuntimeError("Capture failed")
                target = json.loads((capture / "capture-report.json").read_text())[
                    "frames"
                ]
                if (
                    count == target
                    and not any(busy.values())
                    and not any(v is not None for v in pending.values())
                ):
                    break
            if not dirty:
                time.sleep(0.005)
        for c in contexts.values():
            schedule_archive(c)
        for f in futures:
            f.result()
        end = time.time()
        # Measured over motion only; queue drain cannot rescue deadline failures.
        sim = json.loads((capture / "performance-report.json").read_text())
        a, b = sim["run_wall_start"], sim["run_wall_end"]

        def active(kind):
            return [r for r in rows_by[kind] if a <= r["wall"] <= b]

        def p95(values):
            return float(np.percentile(values, 95)) if values else None

        seg = active("semantic")
        ocr = active("ocr")
        geo = active("geometry")
        result = dict(
            geometry_hz=len(geo) / (b - a),
            mask_hz=len(seg) / (b - a),
            mask_p95_s=p95([r["latency_s"] for r in rows_by["semantic"]]),
            ocr_p95_s=p95([r["latency_s"] for r in rows_by["ocr"]]),
            mask_deadline_misses=sum(r["latency_s"] > 0.5 for r in rows_by["semantic"]),
            ocr_deadline_misses=sum(r["latency_s"] > 2 for r in rows_by["ocr"]),
            qa_completed_hz=len(active("qa")) / (b - a),
            mask_completed=len(rows_by["semantic"]),
            ocr_completed=len(rows_by["ocr"]),
            browser_paint_latency_measured=False,
            publication_boundary="Atomic map/inspection/status files ready to read; excludes browser drawing",
            service_times={
                k: dict(
                    p50_s=float(np.percentile([r["service_s"] for r in values], 50))
                    if values
                    else None,
                    p95_s=p95([r["service_s"] for r in values]),
                    completed=len(values),
                )
                for k, values in rows_by.items()
            },
            ocr_samples=len(sampled_ocr),
            ocr_completion_fraction=len(rows_by["ocr"]) / max(1, len(sampled_ocr)),
            ocr_completed_during_motion=len(ocr),
            text_updates_during_motion=sum(r["bound_updates"] for r in ocr),
            drops=drop,
            geometry_all_consumed=count == target,
            quality_correct_text_updates_verified=False,
        )
        result["timing_pass"] = bool(
            result["geometry_all_consumed"]
            and result["geometry_hz"] >= 4.75
            and result["mask_hz"] >= 4.75
            and result["mask_p95_s"] is not None
            and result["mask_p95_s"] <= 0.5
            and result["ocr_p95_s"] is not None
            and result["ocr_p95_s"] <= 2
            and result["ocr_completion_fraction"] >= 0.95
            and result["text_updates_during_motion"] > 0
        )
        write_json(root / "realtime-report.json", result)
        write_json(
            root / "report.json",
            dict(
                geometry=rate_report(rows_by["geometry"], 5, start, end),
                semantic=rate_report(rows_by["semantic"], 5, start, end),
                semantic_sample_hz=5,
                semantic_backlog_max=2,
                max_semantic_source_lag_s=max(
                    (r["latency_s"] for r in rows_by["semantic"]), default=None
                ),
                capture_to_geometry_p95_s=p95(
                    [r["capture_to_geometry_s"] for r in rows_by["geometry"]]
                ),
                yolo_calls=len(rows_by["semantic"]),
                perception_backend="yoloe-realtime",
                perception_calls=dict(segmentation=len(rows_by["semantic"])),
                store=store.summary(),
                semantic_backend="llm",
                realtime=result,
                semantic_frames_dropped=drop["seg"],
                llm_valid_frames=sum(r["valid"] for r in rows_by["qa"]),
                llm_calls=len(rows_by["qa"]),
                structured_inference_executed=False,
                weight_mode="internvl_base_control",
                llm_mode="objects-only",
                accepted_instances=sum(
                    r["accepted_instances"] for r in rows_by["semantic"]
                ),
                source_projection_consistent=None,
                projection_validation="source checks enforced; absolute live accuracy not measured",
                identity_confirmed=sum(
                    lm.get("identity_status") == "identity_confirmed"
                    for lm in store.records("landmarks")
                ),
            ),
        )
        print("REALTIME", json.dumps(result), flush=True)
    finally:
        for q in queues.values():
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for w in workers.values():
            w.join(timeout=10)
            if w.is_alive():
                w.terminate()
                w.join(timeout=10)
        archive.shutdown(wait=True)
        metrics.close()
        store.close()
        if frame_stream:
            frame_stream.close()
