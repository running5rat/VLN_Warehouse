"""Single map writer; independent latest-only VLM/Flow/perception workers."""

import argparse
import copy
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import time
import traceback
import numpy as np
from PIL import Image, ImageDraw
from ..config import WarehouseConfig
from ..geometry import (
    LocalMapBuilder,
    align_reference,
    delay_features,
    render_map_image,
    collision_free,
    transform_points,
)
from ..mapping import MapStore, SourceObservation, SemanticEvent
from .core import (
    NavigationState,
    PersonBinding,
    select_landmark,
    neighborhood_path,
    frontier_path,
    check_candidates,
    align_timed_candidates,
    pursuit,
)
from .workers import worker
from .geometry import render_local
from .route_geometry import corridor_observation, instruction_path
from DynaNav.warehouse_loop.live_metrics import atomic_json, complete_lines


def arr(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def flow_batch(cfg, grid, age, source, state, packet, previous):
    xy, mask = align_reference(
        arr(packet.refline_xy),
        arr(packet.refline_mask),
        packet.T_odom_body_obs,
        source["T_odom_body"],
        packet.odom_epoch,
        source["odom_epoch"],
    )
    batch = dict(
        bev=np.stack(
            (grid == 0, grid == 1, grid == -1, np.clip(age / cfg.map_max_age, 0, 1))
        )[None].astype("float32"),
        state=np.array([state], np.float32),
        state_mask=np.full((1, 2), source.get("state_valid", True), bool),
        motion_direction=np.array(
            [
                (
                    -1
                    if getattr(packet, "motion_frame_or_direction", "forward")
                    == "backward"
                    else 1
                )
            ],
            np.float32,
        ),
        map_valid=np.ones(1, bool),
        reference_xy=xy[None].astype("float32"),
        reference_mask=mask[None],
        reference_confidence=np.array([packet.refline_conf], np.float32),
        z_sem=arr(packet.z_sem)[None].astype("float32"),
        semantic_valid=np.ones(1, bool),
        delay=delay_features(
            packet.T_odom_body_obs,
            source["T_odom_body"],
            time.monotonic() - packet.t_obs,
        )[None].astype("float32"),
        inference_mode=np.array([packet.inference_mode == "PREFILL_FOLLOW"], np.int64),
        previous_plan=np.zeros((1, cfg.horizon, 3), np.float32),
        previous_plan_mask=np.zeros((1, cfg.horizon), bool),
    )
    if previous is not None and previous["epoch"] == source["odom_epoch"]:
        elapsed = max(0, source["t_obs"] - previous["t_obs"])
        times = np.arange(1, len(previous["path"]) + 1) * cfg.action_dt
        keep = times > elapsed
        points = previous["path"][keep]
        tf = np.linalg.inv(np.asarray(source["T_odom_body"])) @ np.asarray(
            previous["pose"]
        )
        points = transform_points(np.c_[points, np.zeros(len(points))], tf)[:, :2]
        n = min(len(points), cfg.horizon)
        batch["previous_plan"][0, :n, :2] = points[:n]
        batch["previous_plan"][0, :n, 2] = (times[keep] - elapsed)[:n]
        batch["previous_plan_mask"][0, :n] = True
    return batch


def draw_view(
    root, grid, source, landmarks, ref, candidates, selected, status, trace=None
):
    canvas = render_map_image(grid, 16.0 / len(grid), 720)
    draw = ImageDraw.Draw(canvas)

    def pixel(p):
        return (360 - p[1] / 16 * 720, 340 - p[0] / 16 * 680)

    for path in (
        [] if candidates is None else candidates.reshape(-1, candidates.shape[-2], 2)
    ):
        pts = [pixel(p) for p in path if np.isfinite(p).all()]
        if len(pts) > 1:
            draw.line(pts, fill=(160, 180, 230), width=1)
    for path, color in ((ref, "orange"), (selected, "green")):
        if path is not None and len(path) > 1:
            draw.line([pixel(p) for p in path], fill=color, width=3)
    waypoint = status.get("waypoint_current") or []
    for index, point in enumerate(waypoint, 1):
        x, y = pixel(point)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="cyan", outline="black")
        draw.text((x + 5, y + 5), "W" + str(index), fill="cyan")
    inv = np.linalg.inv(np.array(source["T_odom_body"]))
    if trace is not None and len(trace) > 1:
        local_trace = transform_points(np.asarray(trace), inv)
        draw.line([pixel(p) for p in local_trace], fill="purple", width=2)
    for lm in landmarks:
        if lm.get("submap_id") != source["submap_id"] or not lm.get(
            "landmark_view", {}
        ).get("visible", True):
            continue
        xy = transform_points(np.array([lm["position"]]), inv)[0, :2]
        x, y = pixel(xy)
        if 0 <= x < 720 and 0 <= y < 680:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red")
            draw.text(
                (x + 4, y), lm.get("identity_text") or lm["category"], fill="blue"
            )
    canvas.save(root / "map.tmp.png", compress_level=0)
    (root / "map.tmp.png").replace(root / "map.png")
    Image.open(source["rgb"]).save(root / "rgb.tmp.png", compress_level=0)
    (root / "rgb.tmp.png").replace(root / "rgb.png")
    status.update(
        source_image_id=source["image_id"],
        t_obs=source["t_obs"],
        updated_wall=time.time(),
    )
    atomic_json(root / "status.json", status)


def run(options):
    root = Path(options["output"]) / "navigation"
    root.mkdir(parents=True, exist_ok=True)
    capture = root.parent / "capture"
    ctx = mp.get_context("spawn")
    replies = ctx.Queue(maxsize=8)
    kinds = ["vlm", "flow"] + (
        ["person", "semantic"]
        if options.get("instruction")
        else (["person"] if options["task_mode"] == "follow" else ["semantic"])
    )
    queues = {k: ctx.Queue(maxsize=1) for k in kinds}
    processes = {
        k: ctx.Process(target=worker, args=(k, queues[k], replies, root / k, options))
        for k in kinds
    }
    # Serialized initialization avoids competing model loads and huge transient peaks.
    ready = set()
    pending = {k: None for k in kinds}
    busy = {k: False for k in kinds}
    jobs = {}
    counter = 0
    rows = []
    metrics = (root / "metrics.jsonl").open("w", buffering=1)

    def log(module, **values):
        row = dict(
            module=module, monotonic=time.monotonic(), wall=time.time(), **values
        )
        rows.append(row)
        metrics.write(json.dumps(row, allow_nan=False) + "\n")

    for kind, p in processes.items():
        p.start()
        deadline = time.monotonic() + 300
        while kind not in ready:
            try:
                r = replies.get(timeout=1)
            except queue.Empty:
                if not p.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError(f"{kind} initialization failed/timed out")
                continue
            if r.get("fatal"):
                raise RuntimeError(r["fatal"])
            ready.add(r["kind"])
            atomic_json(root / (r["kind"] + "-provenance.json"), r)
    cfg = WarehouseConfig(
        model_path=options["model"],
        max_new_tokens=2048,
        depth_stride=32,
        map_max_age=2.0,
        map_resolution=0.05,
    )
    if options.get("map"):
        shutil.copytree(options["map"], root / "map")
    store = MapStore(root / "map", array_compression_level=0)
    builder = LocalMapBuilder(cfg)
    nav = NavigationState(
        root.parent.name,
        options["task_mode"],
        options.get("target_landmark"),
        instruction=options.get("instruction"),
    )
    checkpoint_file = root / "state.json"
    if checkpoint_file.exists():
        nav = NavigationState.restore(json.loads(checkpoint_file.read_text()))
    binding = PersonBinding()
    follow = None
    requires_text = True
    binding_text = None
    stream = None
    metadata = None
    count = 0
    source = None
    grid = None
    age = None
    history = []
    state = [0.0, 0.0]
    previous_source = None
    previous_plan = None
    actual_trace = []
    latest_flow = None
    last_request = -1e9
    last_flow = -1e9
    last_view = -1e9
    last_control = -1e9
    closed_capture_at = None
    roi_used = None
    person_history = {}
    blacklist = []
    frontier_attempt = None
    scan_started = None
    follow_scan_start = None
    registration = None
    registration_attempted = False
    last_plan_time = -1e9
    planned_pose = None
    planned_path = None
    planned_reason = None
    landmark_goal_range = float("inf")
    last_candidates = None
    last_scores = []
    ref = np.empty((0, 2))
    selected = None
    command_count = 0
    prior_ids = [
        int(p.name)
        for k in kinds
        for p in (root / k).glob("*")
        if p.is_dir() and p.name.isdigit()
    ]
    counter = max([counter, nav.last_generation, *prior_ids])
    direct_command = None
    route_revision_planned = -1
    facts_event_count = 0
    last_topology = -1e9
    topology = {}

    def persist_facts():
        nonlocal facts_event_count
        nav.state.execution["facts"] = nav.facts.record()
        new = nav.facts.data["events"][facts_event_count:]
        if new:
            with (root / "execution-events.jsonl").open("a") as stream:
                for event in new:
                    stream.write(json.dumps(event) + "\n")
            facts_event_count += len(new)
        atomic_json(checkpoint_file, nav.checkpoint())

    def offer(kind, job):
        nonlocal counter
        counter += 1
        job = {**job, "id": counter}
        if pending[kind]:
            log("queue_replaced", kind=kind, request_id=pending[kind]["id"])
        pending[kind] = job

    def stop(reason):
        atomic_json(
            root / "command.json",
            dict(
                v=0.0,
                w=0.0,
                created_monotonic=time.monotonic(),
                valid_until=time.monotonic() + 0.3,
                task_id=nav.state.contract.task_id,
                source_image_id=source["image_id"] if source else None,
                reason=reason,
                backend=options["execution_backend"],
            ),
        )

    def navigation_landmarks():
        values = copy.deepcopy(store.landmark_view()["landmarks"])
        if registration and source:
            tf = np.asarray(registration["T_current_old"])
            for lm in values:
                if lm["submap_id"] == registration["old_submap"]:
                    lm["position"] = transform_points(np.asarray([lm["position"]]), tf)[
                        0
                    ].tolist()
                    lm["original_submap_id"] = lm["submap_id"]
                    lm["submap_id"] = source["submap_id"]
        return values

    def observation(trigger=None):
        landmarks = navigation_landmarks()
        usable = [
            x
            for x in landmarks
            if x["submap_id"] == source["submap_id"]
            and x.get("landmark_view", {}).get("visible", True)
        ]
        usable.sort(
            key=lambda x: (
                options.get("target_landmark")
                not in (x["landmark_uid"], x.get("identity_text")),
                x["landmark_uid"],
            )
        )
        inv = np.linalg.inv(np.asarray(source["T_odom_body"]))
        mapped = [
            dict(
                category=x["category"],
                name=x.get("identity_text"),
                position=transform_points(np.asarray([x["position"]]), inv)[0].tolist(),
                orientation_kind=x["orientation_kind"],
                status=x["status"],
                identity_status=x.get("identity_status"),
            )
            for x in usable[:16]
        ]
        folder = root / "inputs" / f"{counter + 1:06d}"
        folder.mkdir(parents=True, exist_ok=True)
        render_map_image(grid, cfg.map_resolution).save(folder / "map.png")
        hist = []
        for offset in (3, 2, 1, 0):
            earlier = [
                x for x in history if x["t_obs"] <= source["t_obs"] - offset + 1e-6
            ]
            if earlier and earlier[-1]["image_id"] not in [x["image_id"] for x in hist]:
                x = earlier[-1]
                hist.append(
                    dict(
                        image_id=x["image_id"],
                        path=x["rgb"],
                        t_obs=x["t_obs"],
                        T_odom_body=x["T_odom_body"],
                    )
                )
        target_observation = None
        if follow:
            target_observation = dict(
                visible=follow["visible"],
                bbox=follow.get("bbox"),
                observation_age_sim_s=source["t_obs"]
                - (follow.get("last_t_obs") or source["t_obs"]),
                position_body=(
                    transform_points(np.asarray([follow["position_odom"]]), inv)[
                        0
                    ].tolist()
                    if follow.get("position_odom")
                    else None
                ),
            )
        execution = copy.deepcopy(nav.state.execution)
        if "facts" in execution:
            execution["facts"]["events"] = execution["facts"]["events"][-12:]
        execution["target_binding"] = target_observation
        obs = dict(
            request_trigger=trigger,
            task=asdict(nav.state.contract),
            t_obs=source["t_obs"],
            source_monotonic=source["source_monotonic"],
            T_odom_body=source["T_odom_body"],
            odom_epoch=source["odom_epoch"],
            rgb_history=hist,
            rgb_current=source["rgb"],
            route=nav.state.route,
            route_revision=nav.state.route_revision,
            execution_state=execution,
            robot_state=state,
            robot_state_mask=[len(history) > 1] * 2,
            map_valid=True,
            map_image=str(folder / "map.png"),
            map_landmarks=mapped,
            map_version=str(store.version),
            task_mode=options["task_mode"],
            landmark_schema="legacy",
            route_schema="structured-v1" if nav.natural else "legacy",
            previous_reference=ref.tolist(),
            target_binding_id=(follow or {}).get("binding_id"),
            target_visible=(follow or {}).get("visible", False),
            follow_target=target_observation,
        )
        obs["previous_local"] = nav.local_input(obs)
        atomic_json(folder / "observation.json", obs)
        return obs

    atomic_json(root / "task.json", asdict(nav.state.contract))
    atomic_json(
        root / "ready.json", dict(ready=True, load_completed_monotonic=time.monotonic())
    )
    try:
        while True:
            now = time.monotonic()
            if (root.parent / "cancel.json").exists():
                stop("cancelled")
                nav.status = nav.terminal = "cancelled"
                nav.facts.data["terminal"] = "cancelled"
                nav.facts.event("cancelled")
                persist_facts()
                break
            for kind, p in processes.items():
                if not p.is_alive():
                    raise RuntimeError(f"{kind} worker exited")
            while True:
                try:
                    r = replies.get_nowait()
                except queue.Empty:
                    break
                kind = r["kind"]
                busy[kind] = False
                if r.get("fatal"):
                    raise RuntimeError(r["fatal"])
                job = jobs.pop(r["id"])
                log(
                    "worker",
                    kind=kind,
                    request_id=r["id"],
                    service_s=r["ready_at"] - r["started"],
                    error=r.get("error"),
                )
                if r.get("error"):
                    continue
                result = r["data"]
                if kind == "vlm":
                    obs = job["observation"]
                    max_age = 3 if job["mode"] == "PREFILL_FOLLOW" else 10
                    # Recheck live binding: an in-flight visible result must not revive a lost target.
                    binding_ok = (
                        (nav.natural and job["mode"] == "TEXT")
                        or nav.mode != "follow"
                        or (
                            follow
                            and follow["visible"]
                            and follow["binding_id"] == obs.get("target_binding_id")
                        )
                    )
                    committed = False
                    try:
                        if not binding_ok:
                            raise ValueError(
                                "Follow binding lost or changed during inference"
                            )
                        if source["odom_epoch"] != obs["odom_epoch"]:
                            raise ValueError("Current odom epoch changed")
                        packet = nav.commit(
                            result,
                            obs,
                            r["id"],
                            r["started"],
                            r["ready_at"],
                            time.monotonic(),
                            max_age,
                        )
                        committed = True
                        if job["mode"] == "TEXT":
                            with (root / "route-history.jsonl").open("a") as route_log:
                                route_log.write(
                                    json.dumps(nav.route_history[-1]) + "\n"
                                )
                        persist_facts()
                        nav.route_was_clear = (
                            collision_free(
                                arr(packet.refline_xy)[arr(packet.refline_mask)],
                                job["reference_check_grid"],
                                cfg.map_resolution,
                                cfg.footprint_radius,
                            )
                            if packet.reference_usable
                            and arr(packet.refline_mask).any()
                            else False
                        )
                        snapshot = {
                            k: v
                            for k, v in asdict(packet).items()
                            if k not in ("z_sem", "refline_xy", "refline_mask")
                        }
                        atomic_json(
                            root / "committed.json",
                            dict(
                                task_state=nav.record(),
                                snapshot=snapshot,
                                artifact=r["artifact"],
                            ),
                        )
                        log(
                            "snapshot_committed",
                            mode=job["mode"],
                            request_id=r["id"],
                            age_s=time.monotonic() - packet.t_obs,
                        )
                        if (
                            nav.mode == "follow"
                            and job["mode"] == "TEXT"
                            and obs.get("target_binding_id")
                            and follow
                            and follow["visible"]
                        ):
                            requires_text = False
                            binding_text = obs.get("target_binding_id")
                            atomic_json(
                                root / "follow-ready.json",
                                dict(binding_id=binding_text),
                            )
                    except ValueError as exc:
                        log("snapshot_rejected", request_id=r["id"], reason=str(exc))
                    if (
                        committed
                        and result.get("valid")
                        and nav.mode != "follow"
                        and "semantic" in kinds
                        and job["mode"] == "TEXT"
                    ):
                        grouped = {}
                        for item in result["parsed"]["landmark"]:
                            grouped.setdefault(item["image_id"], []).append(
                                item["category"]
                            )
                        for image_id, categories in grouped.items():
                            saved = store.get("observations", image_id)
                            if saved:
                                offer(
                                    "semantic",
                                    dict(
                                        source=saved,
                                        categories=list(dict.fromkeys(categories)),
                                        parent_request=r["id"],
                                    ),
                                )
                        local = result["parsed"]["local"]
                        if local.casefold() != "unknown":
                            ev = SemanticEvent(
                                f"nav:{r['id']}:local",
                                obs["rgb_history"][-1]["image_id"],
                                local,
                                source="llm",
                                kind="relation",
                            )
                            log("local_event", result=store.consume(ev))
                    atomic_json(
                        root / "last-vlm.json",
                        {
                            k: v
                            for k, v in result.items()
                            if not isinstance(v, np.ndarray)
                        },
                    )
                elif kind == "flow":
                    latest_flow = dict(result=result, job=job, ready=r["ready_at"])
                    last_candidates = result["trajectories"][0]
                    condition_grid = np.full(
                        job["batch"]["bev"].shape[-2:], -1, np.int8
                    )
                    condition_grid[job["batch"]["bev"][0, 0] > 0.5] = 0
                    condition_grid[job["batch"]["bev"][0, 1] > 0.5] = 1
                    condition_ref = job["batch"]["reference_xy"][0][
                        job["batch"]["reference_mask"][0]
                    ]
                    _, source_scores = check_candidates(
                        last_candidates,
                        condition_grid,
                        condition_ref,
                        resolution=cfg.map_resolution,
                        footprint=cfg.footprint_radius,
                    )
                    last_scores = source_scores
                    atomic_json(
                        Path(r["artifact"]) / "source-scores.json",
                        dict(
                            scores=source_scores,
                            source_image_id=job["source"]["image_id"],
                            generation=job["generation"],
                            scope="source-frame diagnostic; execution independently rechecks current observations",
                        ),
                    )
                    log(
                        "flow_candidates",
                        request_id=r["id"],
                        shape=list(result["trajectories"].shape),
                        source_image_id=job["source"]["image_id"],
                        source_age_s=time.monotonic()
                        - job["source"]["source_monotonic"],
                    )
                elif kind == "semantic":
                    from ..perception.realtime import consume_text, save_frame

                    batch = result["segmentation"]
                    saved = job["source"]
                    accepted = []
                    for i, inst in enumerate(batch["instances"]):
                        event = SemanticEvent(
                            f"nav:{job['parent_request']}:{saved['image_id']}:{i}",
                            saved["image_id"],
                            inst["category"],
                            source="llm",
                            request_id=str(job["parent_request"]),
                        )
                        outcome = store.consume(
                            event,
                            inst["mask"],
                            inst["confidence"],
                            segmentation=batch["provenance"],
                        )
                        accepted.append(
                            dict(
                                instance_index=i,
                                category=inst["category"],
                                bbox=inst["bbox"],
                                confidence=inst["confidence"],
                                selected=True,
                                parent_event_id=event.message_id,
                                result=outcome,
                                geometry_file=(
                                    str(
                                        store.root
                                        / store.get(
                                            "landmarks", outcome["landmark_uid"]
                                        )["geometry"]
                                    )
                                    if outcome.get("landmark_uid")
                                    else None
                                ),
                            )
                        )
                    texts = consume_text(store, saved, batch, accepted, result["ocr"])
                    save_frame(
                        root / "semantic-frames" / str(r["id"]),
                        saved,
                        batch,
                        accepted,
                        dict(source="structured-text"),
                        result["ocr"],
                        texts,
                    )
                    log(
                        "semantic_commit",
                        source_image_id=saved["image_id"],
                        instances=len(accepted),
                        accepted=sum(
                            x["result"]["status"] == "accepted" for x in accepted
                        ),
                        text_updates=len(texts["updates"]),
                    )
                    atomic_json(root / "landmarks.json", store.landmark_view())
                elif kind == "person":
                    saved = job["source"]
                    for inst in result["instances"]:
                        p = store.prepare_projection(saved["image_id"], inst["mask"])
                        inst["position_odom"] = (
                            np.median(p["points"], axis=0).tolist()
                            if len(p["points"]) >= 10
                            else None
                        )
                    roi = None
                    roi_file = root / "target-roi.json"
                    if roi_file.exists():
                        supplied = json.loads(roi_file.read_text())
                        if (
                            supplied.get("image_id") == saved["image_id"]
                            and supplied != roi_used
                        ):
                            if supplied.get("rgb_hash") not in (
                                None,
                                saved["rgb_hash"],
                            ):
                                raise ValueError("ROI RGB hash mismatch")
                            roi = supplied["bbox"]
                            roi_used = supplied
                    # Retain source-bound geometry, not full masks, so delayed manual
                    # annotation can replay measured continuity up to the current frame.
                    person_history[saved["image_id"]] = (
                        saved,
                        [
                            {k: x[k] for k in ("bbox", "position_odom")}
                            for x in result["instances"]
                        ],
                    )
                    if len(person_history) > 128:
                        oldest = min(
                            person_history, key=lambda k: person_history[k][0]["t_obs"]
                        )
                        person_history.pop(oldest)
                    follow = binding.update(
                        result["instances"], saved, roi, auto_unique=nav.natural
                    )
                    if roi is not None:
                        requires_text = True
                        replayed = 0
                        for old_source, old_instances in sorted(
                            person_history.values(), key=lambda x: x[0]["t_obs"]
                        ):
                            if (
                                old_source["odom_epoch"] == saved["odom_epoch"]
                                and old_source["t_obs"] > saved["t_obs"]
                            ):
                                follow = binding.update(old_instances, old_source)
                                replayed += 1
                        log(
                            "manual_binding_replay",
                            annotated_image_id=saved["image_id"],
                            replayed=replayed,
                            latest_source_image_id=follow["source_image_id"],
                            visible=follow["visible"],
                            binding_id=follow["binding_id"],
                        )
                    if not follow["visible"]:
                        requires_text = True
                    log("person_track", **follow)
                    atomic_json(root / "person.json", follow)
                    im = Image.open(saved["rgb"])
                    draw = ImageDraw.Draw(im)
                    for inst in result["instances"]:
                        draw.rectangle(inst["bbox"], outline="yellow", width=3)
                    im.save(root / "person.tmp.png")
                    (root / "person.tmp.png").replace(root / "person.png")
            episode = capture / "raw/episode.json"
            if stream is None and episode.exists():
                try:
                    metadata = json.loads(episode.read_text())
                    stream = (episode.parent / metadata["frames_file"]).open()
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
            if stream:
                f = next(complete_lines(stream), None)
                if f is not None:
                    started = time.monotonic()
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
                        nav.state.contract.task_id,
                    )
                    source = store.add_observation(obs)
                    source["source_monotonic"] = f.get(
                        "source_monotonic",
                        time.monotonic() - (time.time() - f["wall_time"]),
                    )
                    source["state_valid"] = False
                    if (
                        previous_source is not None
                        and previous_source["odom_epoch"] == source["odom_epoch"]
                    ):
                        tf = np.linalg.inv(
                            np.array(previous_source["T_odom_body"])
                        ) @ np.array(source["T_odom_body"])
                        dt = source["t_obs"] - previous_source["t_obs"]
                        if dt <= 0:
                            raise ValueError("Nonmonotonic source simulation time")
                        source["state_valid"] = True
                        state = [
                            float(tf[0, 3] / dt),
                            float(np.arctan2(tf[1, 0], tf[0, 0]) / dt),
                        ]
                    if not registration_attempted and options.get("map_anchor_image"):
                        registration_attempted = True
                        from .registration import verify_anchor

                        old = store.get("observations", options["map_anchor_image"])
                        try:
                            if old is None:
                                raise ValueError("Unknown overlap anchor image")
                            registration = verify_anchor(
                                store.points(old), store.points(source)
                            )
                            registration.update(
                                old_submap=old["submap_id"],
                                new_submap=source["submap_id"],
                                old_image_id=old["image_id"],
                                new_image_id=source["image_id"],
                            )
                            atomic_json(root / "map-registration.json", registration)
                            log("map_registration", status="accepted", **registration)
                        except ValueError as exc:
                            log("map_registration", status="rejected", reason=str(exc))
                    nav.facts.observe(
                        source["T_odom_body"],
                        source["t_obs"],
                        source["odom_epoch"],
                        state,
                    )
                    if nav.facts.data.get("terminal"):
                        nav.terminal = nav.facts.data["terminal"]
                    nav.state.execution.update(
                        budget_seconds=options.get("budget_seconds", 120),
                        budget_distance=options.get("budget_distance", 30),
                    )
                    previous_source = source
                    history.append(source)
                    history = history[-32:]
                    _, depth = store.source_arrays(source["image_id"])
                    builder.add(
                        depth=depth,
                        K=f["K"],
                        T_body_camera=f["T_body_camera"],
                        T_odom_body=f["T_odom_body"],
                        t_obs=f["t_obs"],
                        odom_epoch=f["odom_epoch"],
                    )
                    grid, age = render_local(
                        builder, f["T_odom_body"], f["t_obs"], f["odom_epoch"]
                    )
                    if source["t_obs"] - last_topology >= 1:
                        topology = corridor_observation(
                            grid, cfg.map_resolution, cfg.footprint_radius
                        )
                        last_topology = source["t_obs"]
                        nav.state.execution["observed_corridor"] = topology
                        nav.facts.intersection(
                            np.asarray(source["T_odom_body"])[:3, 3],
                            topology["inside_intersection"],
                            source["image_id"],
                        )
                    persist_facts()
                    # Isaac safety reads latest local occupancy and its exact pose/time.
                    with (root / "safety.tmp").open("wb") as out:
                        np.savez(
                            out,
                            grid=grid,
                            pose=np.array(f["T_odom_body"]),
                            source_monotonic=source["source_monotonic"],
                            odom_epoch=f["odom_epoch"],
                            resolution=cfg.map_resolution,
                        )
                    (root / "safety.tmp").replace(root / "safety.npz")
                    count += 1
                    actual_trace.append(
                        np.asarray(source["T_odom_body"])[:3, 3].tolist()
                    )
                    log(
                        "geometry",
                        image_id=source["image_id"],
                        service_s=time.monotonic() - started,
                        age_s=time.monotonic() - source["source_monotonic"],
                    )
                    atomic_json(
                        root / "source.json",
                        dict(
                            image_id=source["image_id"],
                            rgb=source["rgb"],
                            rgb_hash=source["rgb_hash"],
                        ),
                    )
                    if nav.mode == "follow" and closed_capture_at is None:
                        offer("person", dict(source=source, categories=["person"]))
            roi_file = root / "target-roi.json"
            if (
                nav.mode == "follow"
                and source is not None
                and roi_file.exists()
                and not busy["person"]
            ):
                supplied = json.loads(roi_file.read_text())
                if supplied != roi_used:
                    saved = store.get("observations", supplied.get("image_id", ""))
                    if saved:
                        offer("person", dict(source=saved, categories=["person"]))
            if source is not None and closed_capture_at is None:
                now = time.monotonic()
                if (
                    nav.mode == "follow"
                    and follow
                    and source["t_obs"] - (follow.get("last_t_obs") or 0) > 0.6
                ):
                    follow = {**follow, "visible": False, "status": "stale"}
                    requires_text = True
                age_limit = (
                    3
                    if nav.snapshot and nav.snapshot.inference_mode == "PREFILL_FOLLOW"
                    else 10
                )
                packet, invalid = nav.valid_snapshot(
                    now, source["odom_epoch"], age_limit
                )
                ref = np.empty((0, 2))
                if packet:
                    xy, mask = align_reference(
                        arr(packet.refline_xy),
                        arr(packet.refline_mask),
                        packet.T_odom_body_obs,
                        source["T_odom_body"],
                        packet.odom_epoch,
                        source["odom_epoch"],
                    )
                    ref = xy[mask]
                    if getattr(nav, "route_was_clear", False) and not collision_free(
                        ref, grid, cfg.map_resolution, cfg.footprint_radius
                    ):
                        log(
                            "snapshot_invalidated",
                            reason="observed_map_changed_on_reference",
                            generation=packet.generation_id,
                        )
                        nav.snapshot = packet = None
                        invalid = "observed_map_changed_on_reference"
                        requires_text = True
                        ref = np.empty((0, 2))
                nav.state.execution.update(
                    status=nav.status,
                    current_pose=source["T_odom_body"],
                    target_binding=follow,
                )
                if "start_pose" not in nav.state.execution:
                    nav.state.execution["start_pose"] = source["T_odom_body"]
                if closed_capture_at is None and (
                    nav.natural
                    or nav.mode != "follow"
                    or (follow and follow["visible"])
                ):
                    mode = (
                        "PREFILL_FOLLOW"
                        if nav.mode == "follow"
                        and follow
                        and follow["visible"]
                        and not requires_text
                        and binding_text == follow["binding_id"]
                        and follow["confirmations"] >= 2
                        else "TEXT"
                    )
                    due = now - last_request >= (1 if mode == "PREFILL_FOLLOW" else 2)
                    refresh = (
                        nav.natural
                        or mode == "PREFILL_FOLLOW"
                        or packet is None
                        or now - packet.t_obs > 5
                        or not len(ref)
                    )
                    if due and refresh and not nav.terminal:
                        trigger = (
                            "task_initialization"
                            if nav.state.route_revision == 0
                            else (
                                "target_lost"
                                if nav.mode == "follow"
                                and not (follow and follow["visible"])
                                else (
                                    "reference_unusable"
                                    if packet is None or not packet.reference_usable
                                    else "execution_observation_refresh"
                                )
                            )
                        )
                        obs = observation(trigger)
                        offer(
                            "vlm",
                            dict(
                                observation=obs,
                                mode=mode,
                                reference_check_grid=grid.copy(),
                            ),
                        )
                        last_request = now
                if (
                    closed_capture_at is None
                    and packet
                    and now - last_flow >= 0.2
                    and not busy["flow"]
                ):
                    batch = flow_batch(
                        cfg, grid, age, source, state, packet, previous_plan
                    )
                    offer(
                        "flow",
                        dict(
                            source=source,
                            rgb=source["rgb"],
                            batch=batch,
                            generation=packet.generation_id,
                            route_revision=packet.route_revision,
                        ),
                    )
                    last_flow = now
                if now - last_control >= 0.1:
                    last_control = now
                    selected = None
                    reason = invalid or "waiting_flow"
                    if options["execution_backend"] == "geometry-baseline":
                        if nav.state.route_revision != route_revision_planned:
                            planned_path = None
                            direct_command = None
                            route_revision_planned = nav.state.route_revision
                        if now - last_plan_time >= 1 or planned_path is None:
                            if nav.mode == "explore":
                                blocked = [
                                    transform_points(
                                        np.array([[*p, 0.0]]),
                                        np.linalg.inv(
                                            np.asarray(source["T_odom_body"])
                                        ),
                                    )[0, :2]
                                    for p, until in blacklist
                                    if now < until
                                ]
                                selected = frontier_path(
                                    grid,
                                    cfg.map_resolution,
                                    cfg.footprint_radius,
                                    blocked,
                                )
                                reason = (
                                    "frontier"
                                    if len(selected)
                                    else "no_reachable_frontier"
                                )
                            elif nav.mode == "landmark":
                                lm, reason = select_landmark(
                                    navigation_landmarks(),
                                    nav.target,
                                    source["submap_id"],
                                )
                                if lm:
                                    goal = transform_points(
                                        np.array([lm["position"]]),
                                        np.linalg.inv(
                                            np.asarray(source["T_odom_body"])
                                        ),
                                    )[0, :2]
                                    selected = neighborhood_path(
                                        grid,
                                        goal,
                                        cfg.map_resolution,
                                        cfg.footprint_radius,
                                        1.0,
                                    )
                                    landmark_goal_range = float(np.linalg.norm(goal))
                                    reason = "landmark_neighborhood"
                            elif nav.mode == "instruction":
                                goal = (
                                    nav.state.route.get("current_subgoal", {})
                                    if isinstance(nav.state.route, dict)
                                    else {}
                                )
                                selected, reason, direct_command = instruction_path(
                                    goal,
                                    nav.facts.record(),
                                    grid,
                                    source["T_odom_body"],
                                    cfg.map_resolution,
                                    cfg.footprint_radius,
                                )
                            elif follow and follow["visible"]:
                                goal = transform_points(
                                    np.array([follow["position_odom"]]),
                                    np.linalg.inv(np.asarray(source["T_odom_body"])),
                                )[0, :2]
                                if np.linalg.norm(goal) > 1.5:
                                    selected = neighborhood_path(
                                        grid,
                                        goal,
                                        cfg.map_resolution,
                                        cfg.footprint_radius,
                                        1.5,
                                    )
                                reason = "follow_neighborhood"
                            else:
                                reason = "follow_target_unavailable"
                            planned_path = selected
                            planned_pose = source["T_odom_body"]
                            planned_reason = reason
                            last_plan_time = now
                        selected = planned_path
                        reason = planned_reason
                        if selected is not None and len(selected):
                            aligned, mask = align_reference(
                                selected,
                                np.ones(len(selected), bool),
                                planned_pose,
                                source["T_odom_body"],
                                source["odom_epoch"],
                                source["odom_epoch"],
                            )
                            selected = aligned[mask]
                    elif (
                        options["execution_backend"] == "model"
                        and packet
                        and packet.reference_usable
                        and latest_flow
                        and latest_flow["job"]["generation"] == packet.generation_id
                    ):
                        fjob = latest_flow["job"]
                        elapsed = source["t_obs"] - fjob["source"]["t_obs"]
                        if (
                            now - fjob["source"]["source_monotonic"] < 0.5
                            and elapsed < 0.5
                        ):
                            paths = latest_flow["result"]["trajectories"][0]
                            aligned = align_timed_candidates(
                                paths,
                                np.asarray(fjob["source"]["T_odom_body"]),
                                np.asarray(source["T_odom_body"]),
                                max(0, elapsed),
                                cfg.action_dt,
                            )
                            if aligned.shape[-2] > 1:
                                selected, scores = check_candidates(
                                    aligned,
                                    grid,
                                    ref,
                                    resolution=cfg.map_resolution,
                                    footprint=cfg.footprint_radius,
                                )
                                last_scores = scores
                                log(
                                    "candidate_selection",
                                    request_id=fjob["id"],
                                    scores=scores,
                                    selected=selected is not None,
                                )
                            reason = (
                                "model_candidate"
                                if selected is not None
                                else "no_safe_flow_candidate"
                            )
                        else:
                            reason = "flow_result_expired"
                    if nav.mode == "follow" and (not follow or not follow["visible"]):
                        selected = None
                        reason = "follow_target_lost"
                    if now - source["source_monotonic"] > 0.6:
                        selected = None
                        reason = "geometry_expired"
                    command = (0.0, 0.0)
                    if selected is not None and len(selected) > 1:
                        command, reason = pursuit(
                            selected,
                            grid,
                            resolution=cfg.map_resolution,
                            footprint=cfg.footprint_radius,
                            speed=(
                                min(
                                    0.3,
                                    float(np.linalg.norm(selected[0]) / cfg.action_dt),
                                )
                                if options["execution_backend"] == "model"
                                else 0.3
                            ),
                        )
                        previous_plan = dict(
                            path=selected,
                            pose=source["T_odom_body"],
                            epoch=source["odom_epoch"],
                            t_obs=source["t_obs"],
                        )
                    if (
                        options["execution_backend"] == "geometry-baseline"
                        and nav.mode in ("explore", "landmark")
                        and reason in ("no_reachable_frontier", "landmark_not_observed")
                        and now - source["source_monotonic"] < 0.6
                    ):
                        if scan_started is None:
                            scan_started = source["t_obs"]
                        if source["t_obs"] - scan_started < 22:
                            phase = (source["t_obs"] - scan_started) % 6
                            command = (
                                0.0,
                                0.3 if phase < 1.5 or phase >= 4.5 else -0.3,
                            )
                            reason = "scan_observed_clearance"
                    if (
                        options["execution_backend"] == "geometry-baseline"
                        and nav.mode == "follow"
                        and follow
                        and follow["visible"]
                        and (selected is None or len(selected) < 2)
                        and now - source["source_monotonic"] < 0.6
                    ):
                        if follow_scan_start is None:
                            follow_scan_start = source["t_obs"]
                        elapsed = source["t_obs"] - follow_scan_start
                        if elapsed < 6:
                            command = (
                                0.0,
                                0.3 if elapsed < 1.5 or elapsed >= 4.5 else -0.3,
                            )
                            reason = "scan_follow_clearance"
                    if (
                        reason == "neighborhood_reached"
                        and nav.mode == "explore"
                        and selected is not None
                        and len(selected)
                    ):
                        endpoint = transform_points(
                            np.array([[*selected[-1], 0.0]]),
                            np.asarray(source["T_odom_body"]),
                        )[0, :2]
                        blacklist.append(
                            (endpoint, now + options.get("budget_seconds", 120) + 1)
                        )
                        planned_path = None
                        nav.facts.event(
                            "frontier_reached",
                            position=endpoint.tolist(),
                            image_id=source["image_id"],
                        )
                    if (
                        reason == "neighborhood_reached"
                        and nav.mode == "landmark"
                        and landmark_goal_range <= 1.8
                    ):
                        command = (0.0, 0.0)
                        nav.facts.event(
                            "landmark_neighborhood_reached",
                            target=nav.target,
                            image_id=source["image_id"],
                        )
                        atomic_json(
                            root
                            / (
                                "landmark-arrival.json"
                                if nav.natural
                                else "task-finished.json"
                            ),
                            dict(
                                reason="observed_target_neighborhood",
                                target=nav.target,
                                source_image_id=source["image_id"],
                                execution_backend=options["execution_backend"],
                            ),
                        )
                    if (
                        options["execution_backend"] == "geometry-baseline"
                        and nav.mode == "instruction"
                        and direct_command is not None
                    ):
                        command = direct_command
                    if reason in (
                        "corridor_end_reached",
                        "relative_distance_reached",
                        "turn_observed_complete",
                    ):
                        nav.facts.event(reason, image_id=source["image_id"])
                    if np.linalg.norm(state) < 0.02 and source["state_valid"]:
                        nav.facts.event("observed_stopped", image_id=source["image_id"])
                    if (
                        options["execution_backend"] == "geometry-baseline"
                        and nav.mode == "explore"
                        and selected is not None
                        and len(selected)
                    ):
                        endpoint = transform_points(
                            np.asarray([[*selected[-1], 0.0]]),
                            np.asarray(source["T_odom_body"]),
                        )[0, :2]
                        travelled = nav.facts.data["path_m"]
                        if (
                            frontier_attempt is None
                            or np.linalg.norm(endpoint - frontier_attempt["goal"]) > 0.8
                            or travelled - frontier_attempt["path_m"] > 0.2
                        ):
                            frontier_attempt = dict(
                                goal=endpoint.copy(),
                                path_m=travelled,
                                started=source["t_obs"],
                            )
                        elif source["t_obs"] - frontier_attempt["started"] > 5:
                            blacklist.append((endpoint.copy(), now + 20))
                            nav.facts.event(
                                "frontier_temporarily_rejected",
                                position=endpoint.tolist(),
                                reason="no_measured_progress_for_5_sim_seconds",
                            )
                            planned_path = None
                            selected, command = None, (0.0, 0.0)
                            reason = "frontier_temporarily_rejected"
                            frontier_attempt = None
                    current_goal = (
                        nav.state.route.get("current_subgoal", {})
                        if isinstance(nav.state.route, dict)
                        else {}
                    )
                    if current_goal.get("action") in ("wait", "stop"):
                        command, selected = (0.0, 0.0), None
                        reason = "model_requested_" + current_goal["action"]
                    backward = (
                        current_goal.get("action") == "relative_motion"
                        and current_goal.get("direction") == "backward"
                    )
                    if options["execution_backend"] == "reference-only" or backward:
                        command, selected = (0.0, 0.0), None
                        reason = (
                            "execution_not_evaluated" if backward else "reference_only"
                        )
                    if nav.natural and (
                        not isinstance(nav.state.route, dict)
                        or nav.state.route["status"] != "active"
                    ):
                        command, selected = (0.0, 0.0), None
                        reason = "waiting_active_model_subgoal"
                    if now - source["source_monotonic"] > 0.6 or (
                        nav.mode == "follow" and not (follow and follow["visible"])
                    ):
                        command = (0.0, 0.0)
                        reason = "stale_observation_or_lost_target"
                    facts = nav.facts.data
                    if source["t_obs"] - facts.get(
                        "task_start_time", source["t_obs"]
                    ) >= options.get("budget_seconds", 120) or facts[
                        "path_m"
                    ] >= options.get(
                        "budget_distance", 30
                    ):
                        nav.terminal = facts["terminal"] = "budget_exhausted"
                    if source["t_obs"] - facts.get(
                        "task_start_time", source["t_obs"]
                    ) >= options.get("timeout_seconds", 300):
                        nav.terminal = facts["terminal"] = "task_timeout"
                    if nav.terminal:
                        command = (0.0, 0.0)
                        reason = nav.terminal
                        nav.facts.event(reason)
                        atomic_json(
                            root / "task-finished.json",
                            dict(reason=reason, task_success=reason == "completed"),
                        )
                    nav.status = reason
                    persist_facts()
                    command_count += 1
                    command_row = dict(
                        v=command[0],
                        w=command[1],
                        created_monotonic=time.monotonic(),
                        valid_until=time.monotonic() + 0.35,
                        task_id=nav.state.contract.task_id,
                        source_image_id=source["image_id"],
                        odom_epoch=source["odom_epoch"],
                        reason=reason,
                        backend=options["execution_backend"],
                    )
                    atomic_json(root / "command.json", command_row)
                    log("command", **command_row)
                if now - last_view > 0.5:
                    last_view = now
                    display_candidates = last_candidates
                    if last_candidates is not None and latest_flow:
                        tf = np.linalg.inv(
                            np.asarray(source["T_odom_body"])
                        ) @ np.asarray(latest_flow["job"]["source"]["T_odom_body"])
                        flat = last_candidates.reshape(-1, 2)
                        display_candidates = transform_points(
                            np.c_[flat, np.zeros(len(flat))], tf
                        )[:, :2].reshape(last_candidates.shape)
                    wp = nav.route_history[-1]["waypoint"] if nav.route_history else []
                    wp_current = []
                    if wp and nav.snapshot:
                        tf = np.linalg.inv(
                            np.asarray(source["T_odom_body"])
                        ) @ np.asarray(nav.snapshot.T_odom_body_obs)
                        wp_current = transform_points(np.c_[wp, np.zeros(len(wp))], tf)[
                            :, :2
                        ].tolist()
                    draw_view(
                        root,
                        grid,
                        source,
                        navigation_landmarks(),
                        ref,
                        display_candidates,
                        selected,
                        dict(
                            waypoint_current=wp_current,
                            task=nav.record(),
                            status=nav.status,
                            backend=options["execution_backend"],
                            frames=count,
                            follow=follow,
                            snapshot_error=invalid,
                            previous_route=(
                                nav.route_history[-1]["previous_route"]
                                if nav.route_history
                                else None
                            ),
                            reference_validation=(
                                nav.snapshot.reference_validation
                                if nav.snapshot
                                else None
                            ),
                            waypoint=(
                                nav.route_history[-1].get("waypoint")
                                if nav.route_history
                                else None
                            ),
                            raw_reference=(
                                arr(nav.snapshot.refline_xy).tolist()
                                if nav.snapshot
                                else None
                            ),
                            model_quality_validated=False,
                            queue_busy=busy,
                            candidate_scores=last_scores,
                            candidate_source_image_id=(
                                latest_flow["job"]["source"]["image_id"]
                                if latest_flow
                                else None
                            ),
                            candidate_source_age_s=(
                                now - latest_flow["job"]["source"]["source_monotonic"]
                                if latest_flow
                                else None
                            ),
                        ),
                        trace=actual_trace,
                    )
            for kind in kinds:
                if pending[kind] and not busy[kind]:
                    job = pending[kind]
                    pending[kind] = None
                    if (
                        kind == "vlm"
                        and job["observation"]["route_revision"]
                        != nav.state.route_revision
                    ):
                        log(
                            "queue_discarded",
                            kind=kind,
                            request_id=job["id"],
                            reason="route_version_changed",
                        )
                        last_request = 0
                        continue
                    busy[kind] = True
                    jobs[job["id"]] = job
                    queues[kind].put(job)
            done = root.parent / "capture-process.json"
            if done.exists():
                if closed_capture_at is None:
                    closed_capture_at = time.monotonic()
                    stop("capture_complete")
                for k in kinds:
                    pending[k] = None
                expected = (
                    json.loads((capture / "capture-report.json").read_text())["frames"]
                    if (capture / "capture-report.json").exists()
                    else count
                )
                if count >= expected and (
                    not any(busy.values()) or time.monotonic() - closed_capture_at > 20
                ):
                    break
            time.sleep(0.005)
        stop("finished")
        if nav.terminal is None:
            nav.terminal = "capture_limit_reached"
            nav.facts.data["terminal"] = nav.terminal
            nav.facts.event(nav.terminal)
            persist_facts()
        report = dict(
            geometry_frames=count,
            execution_backend=options["execution_backend"],
            task_mode=nav.mode,
            model_quality_validated=False,
            task_completion=nav.terminal or "not_completed",
            route_commits=len(nav.route_history),
            route_progress_correctness="requires_review_of_actual_outputs_and_execution_facts",
            reference_consistent=sum(
                x["validation"]["consistency_valid"] for x in nav.route_history
            ),
            backward_execution_evaluated=False,
            semantic_TEXT_branch_pass=any(
                r["module"] == "semantic_commit" and r.get("accepted", 0) > 0
                for r in rows
            ),
            task_state=nav.record(),
            store=store.summary(),
            valid_text=sum(
                r["module"] == "snapshot_committed" and r["mode"] == "TEXT"
                for r in rows
            ),
            valid_prefill=sum(
                r["module"] == "snapshot_committed" and r["mode"] == "PREFILL_FOLLOW"
                for r in rows
            ),
            flow_batches=sum(r["module"] == "flow_candidates" for r in rows),
            nonzero_commands=sum(
                r["module"] == "command" and (abs(r["v"]) + abs(r["w"]) > 0)
                for r in rows
            ),
            rejected_snapshots=[r for r in rows if r["module"] == "snapshot_rejected"],
            failures=[r for r in rows if r.get("error")],
            quality_metrics_evaluated=False,
        )
        report["model_interface_pass"] = report["flow_batches"] > 0 and (
            report["valid_text"] > 0 or report["valid_prefill"] > 0
        )
        report["model_motion_pass"] = (
            options["execution_backend"] == "model" and report["nonzero_commands"] > 0
        )
        for kind in kinds:
            times = [
                r["service_s"]
                for r in rows
                if r["module"] == "worker" and r["kind"] == kind and not r.get("error")
            ]
            report[kind + "_service_s"] = dict(
                count=len(times),
                p50=float(np.percentile(times, 50)) if times else None,
                p95=float(np.percentile(times, 95)) if times else None,
            )
        perf_path = capture / "performance-report.json"
        if perf_path.exists():
            perf = json.loads(perf_path.read_text())
            t0, t1 = perf["run_wall_start"], perf["run_wall_end"]
            active = [r for r in rows if t0 <= r["wall"] <= t1]
            elapsed = max(1e-9, t1 - t0)
            report["motion_window_s"] = elapsed
            report["completed_hz"] = {
                k: sum(r["module"] == k for r in active) / elapsed
                for k in (
                    "geometry",
                    "flow_candidates",
                    "snapshot_committed",
                    "semantic_commit",
                    "person_track",
                )
            }
            report["queue_replacements"] = {
                k: sum(
                    r["module"] == "queue_replaced" and r["kind"] == k for r in active
                )
                for k in kinds
            }
            geometry_ages = [r["age_s"] for r in active if r["module"] == "geometry"]
            report["geometry_age_p95_s"] = (
                float(np.percentile(geometry_ages, 95)) if geometry_ages else None
            )
            report["timing_targets_pass"] = (
                report["completed_hz"]["geometry"] >= 4.75
                and report["completed_hz"]["flow_candidates"] >= 4.75
            )
            report["timing_scope"] = (
                "Publication/consumption during capture, not browser draw latency. No real-time acceptance claim."
            )
        report["navigation_grid_config"] = {
            "depth_stride": cfg.depth_stride,
            "history_age_s": cfg.map_max_age,
            "resolution_m": cfg.map_resolution,
        }
        atomic_json(root / "report.json", report)
        atomic_json(root / "landmarks.json", store.landmark_view())
    except BaseException:
        stop("runtime_error")
        atomic_json(root / "error.json", dict(traceback=traceback.format_exc()))
        raise
    finally:
        for q in queues.values():
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        for p in processes.values():
            p.join(2)
            if p.is_alive():
                p.terminate()
                p.join(5)
        store.close()
        metrics.close()
        if stream:
            stream.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    run(json.loads(Path(a.config).read_text()))


if __name__ == "__main__":
    main()
