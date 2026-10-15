"""Manual episode extraction, source-bound revisions and schema v2 export.

Annotations are deliberately separate from raw recordings. Future demonstration
geometry is annotation assistance, never a deployment observation.
"""

import copy
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from ..config import WarehouseConfig
from ..geometry import LocalMapBuilder, render_map_image, transform_points, xy_to_grid
from ..prepare import causal_state, select_history
from ..schema import file_sha256, fingerprint
from ..navigation.geometry import render_local

VERSION = 2


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(tmp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_lines(path):
    if not Path(path).exists():
        return []
    # Incomplete live writer line must never be treated as a committed frame.
    text = Path(path).read_text()
    return [json.loads(x) for x in text.splitlines(keepends=True) if x.endswith("\n")]


def load_episode(path):
    path = Path(path).resolve()
    meta = read_json(path)
    frames = meta.get("frames")
    if frames is None:
        frames = read_lines(path.parent / meta["frames_file"])
    if not frames:
        raise ValueError("Episode has no committed RGB-D frames")
    frames = copy.deepcopy(frames)
    previous = -float("inf")
    for i, f in enumerate(frames):
        if not np.isfinite(f["t_obs"]) or f["t_obs"] <= previous:
            raise ValueError("Nonmonotonic frame clock")
        previous = f["t_obs"]
        pose = np.asarray(f["T_odom_body"], float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid pose")
        if not np.allclose(pose[3], [0, 0, 0, 1]) or not np.allclose(
            pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4):
            raise ValueError("Non-rigid pose")
        for field in ("rgb", "depth"):
            if field in f:
                f[field] = str((path.parent / f[field]).resolve())
        f["image_id"] = f"{meta['episode_id']}:{i:08d}"
    return meta, frames


def resample_line(points, count=32, spacing=0.25):
    p = np.asarray(points, float).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("Nonfinite reference")
    out, mask = np.zeros((count, 2)), np.zeros(count, bool)
    if len(p) < 2:
        return out, mask
    p = p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-5]]
    if len(p) < 2:
        return out, mask
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    # Preserve short routes and their endpoint, never silently drop the last turn.
    end = min(s[-1], (count - 1) * spacing)
    q = np.arange(0., end, spacing)
    q = np.r_[q, end][:count]
    out[:len(q)] = np.column_stack([np.interp(q, s, p[:, j]) for j in (0, 1)])
    mask[:len(q)] = True
    return out, mask


def anchors(points):
    p = np.asarray(points, float).reshape(-1, 2)
    if len(p) < 2:
        return []
    s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if s[-1] < 1e-5:
        return []
    q = np.linspace(0., s[-1], 6)[1:]
    return np.column_stack([np.interp(q, s, p[:, j]) for j in (0, 1)]).round(3).tolist()


def reference_geometry(points, grid, resolution, radius=0.35):
    """Dense swept-disc diagnostics, not an RGB visibility or label validity test."""
    from scipy.ndimage import binary_dilation

    p, valid = resample_line(points, count=4096, spacing=min(0.025, resolution / 2))
    p = p[valid]
    if not len(p):
        return dict(collision=False, unknown=False, checked_points=0,
                    center_unknown_points=0, footprint_unknown_points=0,
                    outside_map_points=0, footprint_radius_m=radius)
    r = int(np.ceil(radius / resolution))
    yy, xx = np.mgrid[-r:r+1, -r:r+1]
    footprint = (xx**2 + yy**2) * resolution**2 <= radius**2
    occupied = binary_dilation(grid == 1, structure=footprint)
    unknown = binary_dilation(grid == -1, structure=footprint, border_value=1)
    ij = xy_to_grid(p, len(grid), resolution)
    inside = ((ij >= 0) & (ij < len(grid))).all(-1)
    rc = ij[inside]
    center_unknown = grid[rc[:, 0], rc[:, 1]] == -1
    footprint_unknown = unknown[rc[:, 0], rc[:, 1]]
    return dict(collision=bool(occupied[rc[:, 0], rc[:, 1]].any()),
                unknown=bool(not inside.all() or footprint_unknown.any()),
                checked_points=len(p), center_unknown_points=int(center_unknown.sum()),
                footprint_unknown_points=int(footprint_unknown.sum()),
                outside_map_points=int((~inside).sum()), footprint_radius_m=radius)


def frame_binding(meta, frame):
    return fingerprint(dict(frame=frame, task=meta["task"],
                            rgb_sha256=file_sha256(frame["rgb"]),
                            depth_sha256=file_sha256(frame["depth"]) if frame.get("depth") else None))


def verify_observation_files(sample):
    for h in sample["observation"]["rgb_history"]:
        if h.get("sha256") and file_sha256(h["path"]) != h["sha256"]:
            raise ValueError("Historical RGB changed after extraction")
    view = sample["observation"]["mapview"]
    for field in ("image", "arrays"):
        if view.get(field + "_sha256") and file_sha256(view[field]) != view[field + "_sha256"]:
            raise ValueError("Causal map changed after extraction")


def write_projection_camera(folder, frame, binding):
    """Preview-only calibration sidecar; never change source/approved records."""
    from PIL import Image
    K, T = np.asarray(frame["K"],float), np.asarray(frame["T_body_camera"],float)
    if K.shape != (3,3) or T.shape != (4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
        raise ValueError("Invalid source camera calibration")
    if min(K[0,0],K[1,1])<=0 or not np.allclose(K[2],[0,0,1]) or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-4):
        raise ValueError("Expected pinhole intrinsics and rigid optical camera transform")
    with Image.open(frame["rgb"]) as im: width,height=im.size
    value=dict(image_id=frame["image_id"],source_binding=binding,
               K=K.tolist(),T_body_camera=T.tolist(),width=width,height=height,
               plane="source_body_z_0",occlusion_checked=False,display_only=True)
    write_json(Path(folder)/"projection-camera.json",value)
    return value


def prepare_projection_cameras(root):
    root=Path(root)
    project=read_json(root/"project.json")
    meta,frames=load_episode(project["source_episode"])
    for name in project["samples"]:
        sample=read_json(root/name/"source.json")
        f=frames[sample["provenance"]["frame_index"]]
        binding=frame_binding(meta,f)
        if binding!=sample["provenance"]["source_binding"]:
            raise ValueError("Projection source differs from annotation source")
        write_projection_camera(root/name,f,binding)


def extract(episode, output, *, count=10, indices=None):
    meta, frames = load_episode(episode)
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "project.json").exists():
        raise FileExistsError("Annotation project exists; use it without re-extraction")
    capture_root = Path(episode).resolve().parents[2]
    events = read_lines(capture_root / "annotation-events.jsonl")
    odometry = read_lines(capture_root / "navigation/executed-commands.jsonl")
    intervals = read_lines(capture_root / "recording-events.jsonl")
    def recorded(f):
        if not intervals:
            return True
        before = [e for e in intervals if e["monotonic"] <= f.get("source_monotonic", -1)]
        return bool(before and before[-1]["state"] == "recording")
    eligible = [i for i, f in enumerate(frames) if recorded(f)]
    if not eligible:
        raise ValueError("No frames in a user-started recording interval")
    chosen = indices if indices is not None else [eligible[j] for j in np.linspace(
        0, len(eligible)-1, min(count, len(eligible)), dtype=int)]
    for e in events:
        if e.get("image_id"):
            j = next((i for i, f in enumerate(frames) if f["image_id"] == e["image_id"]), -1)
            if j in eligible:
                chosen.append(j)
    chosen = sorted(set(int(i) for i in chosen))
    if not chosen or any(i not in eligible for i in chosen):
        raise ValueError("Keyframe is outside the recording or episode")
    cfg = WarehouseConfig(depth_stride=32)
    builder = LocalMapBuilder(cfg)
    # Persistent observed history is reconstructed causally, never from final map.
    builder.config.map_max_age = 1e9
    all_xyz = np.array([np.asarray(f["T_odom_body"])[:3, 3] for f in frames])
    samples = []
    for i, f in enumerate(frames):
        if f.get("depth"):
            builder.add(depth=np.load(f["depth"], allow_pickle=False), K=f["K"],
                        T_body_camera=f["T_body_camera"], T_odom_body=f["T_odom_body"],
                        t_obs=f["t_obs"], odom_epoch=f["odom_epoch"])
        if i not in chosen:
            continue
        folder = root / "samples" / f"{i:08d}"
        folder.mkdir(parents=True)
        grid, age = render_local(builder, f["T_odom_body"], f["t_obs"], f["odom_epoch"])
        np.savez_compressed(folder / "map.npz", grid=grid, age=age)
        render_map_image(grid, cfg.map_resolution, 640).save(folder / "map.png")
        history = select_history(frames, i, cfg.history_offsets)
        for h in history:
            j = next(j for j in range(i+1) if frames[j]["t_obs"] == h["t_obs"])
            h["image_id"] = frames[j]["image_id"]
            image_name = f"history-{j:08d}.png"
            shutil.copyfile(h["path"], folder / image_name)
            h["preview"] = f"samples/{i:08d}/{image_name}"
            h["sha256"] = file_sha256(h["path"])
        shutil.copyfile(f["rgb"], folder / "rgb.png")
        state, state_mask = causal_state(frames, i)
        # Stop at an epoch boundary or recording pause. A resumed trace is not continuous GT.
        end = i
        while end < len(frames) and frames[end]["odom_epoch"] == f["odom_epoch"] and recorded(frames[end]):
            end += 1
        future = transform_points(all_xyz[i:end], np.linalg.inv(f["T_odom_body"]))[:, :2]
        trace_source = "rgbd_source_poses_5hz"
        trace = [o for o in odometry if o.get("t_obs") is not None
                 and f["t_obs"] <= o["t_obs"] <= frames[end-1]["t_obs"]
                 and o.get("odom_epoch") == f["odom_epoch"]]
        if len(trace) >= 2 and all(b["t_obs"] > a["t_obs"] for a,b in zip(trace, trace[1:])):
            xyz = np.array([np.asarray(o["pose"])[:3, 3] for o in trace])
            future = transform_points(np.vstack([all_xyz[i], xyz]), np.linalg.inv(f["T_odom_body"]))[:, :2]
            trace_source = "executed_command_source_odometry_30hz"
        line, mask = resample_line(future)
        past = [q for j,q in enumerate(frames[:i+1]) if j >= eligible[0] and q["odom_epoch"] == f["odom_epoch"]]
        start_pose = np.asarray(past[0]["T_odom_body"])
        measured_poses = np.asarray([q["T_odom_body"] for q in past])
        displacement = np.linalg.inv(start_pose) @ np.asarray(f["T_odom_body"])
        yaw = np.unwrap(np.arctan2(measured_poses[:,1,0], measured_poses[:,0,0]))
        facts = dict(source="measured_source_odometry", source_until=f["t_obs"],
                     start_pose=start_pose.tolist(), displacement_xy=displacement[:2,3].tolist(),
                     accumulated_distance_m=float(np.linalg.norm(np.diff(measured_poses[:,:2,3],axis=0),axis=1).sum()),
                     accumulated_yaw_rad=float(yaw[-1]-yaw[0]))
        obs = dict(task=meta["task"], t_obs=f["t_obs"], odom_epoch=f["odom_epoch"],
                   image_id=f["image_id"], T_odom_body=f["T_odom_body"],
                   rgb_history=history, route=meta["task"]["instruction"], route_revision=0,
                   robot_state=state, robot_state_mask=state_mask,
                   mapview=dict(image=str(folder / "map.png"), arrays=str(folder / "map.npz"),
                                resolution=cfg.map_resolution, map_size_m=cfg.map_size_m,
                                image_sha256=file_sha256(folder / "map.png"),
                                arrays_sha256=file_sha256(folder / "map.npz"),
                                source_until=f["t_obs"], landmarks=[], interface="MapView-rendered"),
                   execution={"events": [e for e in events if e.get("t_obs", float("inf")) <= f["t_obs"]],
                              "measured_facts": facts})
        sample = dict(schema_version=VERSION, observation=obs,
                      supervision=dict(text_state="unlabeled", reference_state="unlabeled",
                                       action_state="unlabeled", candidates=[]),
                      provenance=dict(sample_id=f["image_id"], episode_id=meta["episode_id"],
                                      layout_family=meta["layout_family"], pool="manual-golden",
                                      role="regression", source_binding=frame_binding(meta, f),
                                      source_episode=str(Path(episode).resolve()), frame_index=i),
                      assistance=dict(future_trajectory=future.tolist(), suggested_reference=line[mask].tolist(),
                                      trajectory_source=trace_source,
                                      suggested_waypoint=anchors(line[mask]),
                                      geometry=reference_geometry(line[mask], grid, cfg.map_resolution),
                                      warning="Future demonstration for annotation only; not model input"))
        write_json(folder / "source.json", sample)
        write_projection_camera(folder, f, sample["provenance"]["source_binding"])
        samples.append(f"samples/{i:08d}")
    project = dict(schema_version=VERSION, status="awaiting_human_annotation", task=meta["task"],
                   source_episode=str(Path(episode).resolve()), samples=samples,
                   scene_task_validated=False, required_events=["target_visible", "stopped", "arrival_confirmed"],
                   sample_count=len(samples), source_frames=len(frames), config=cfg.to_dict())
    write_json(root / "project.json", project)
    return project


def sample_context(root, index):
    root = Path(root)
    project = read_json(root / "project.json")
    if not 0 <= index < len(project["samples"]):
        raise ValueError("Keyframe index outside project")
    folder = root / project["samples"][index]
    sample = read_json(folder / "source.json")
    # A reviewed language variant changes model text, never raw capture identity
    # or its hash. Require explicit provenance instead of silently replacing tasks.
    translation = project.get("task_translation")
    if translation is not None:
        task = sample["observation"]["task"]
        if (translation.get("source_instruction") != task["instruction"]
            or translation.get("instruction") != project["task"]["instruction"]
            or task.get("task_id") != project["task"].get("task_id")
            or not translation.get("language")
            or not str(translation.get("instruction", "")).strip()):
            raise ValueError("Task translation does not match source/project")
        task["instruction"] = translation["instruction"]
        sample["provenance"]["task_translation"] = copy.deepcopy(translation)
    if "acceptance_criteria" in project:
        sample["observation"]["task"]["acceptance_criteria"] = copy.deepcopy(project["acceptance_criteria"])
    from ..local_context import local_memory, local_context
    memory = None
    replay_events = []
    previous_route, revision, parent = project["task"]["instruction"], 0, None
    for name in project["samples"][:index]:
        path = root / name / "approved.json"
        if not path.exists():
            raise ValueError("Approve earlier keyframes before labeling this Route")
        approved = read_json(path)
        if approved["parent_revision"] != parent:
            raise ValueError("Earlier annotation is stale after a Route revision")
        replay_events.extend(approved.get("replay_events", []))
        previous_source = read_json(root / name / "source.json")["observation"]
        memory = local_memory(approved["parsed"]["local"], previous_source,
                              origin="human_approved", revision=approved["revision_id"])
        previous_route = approved["parsed"]["route"]
        parent = approved["revision_id"]
        revision += 1
    sample["observation"].update(route=previous_route, route_revision=revision)
    sample["observation"]["previous_local"] = local_context(memory, sample["observation"])
    sample["parent_revision"] = parent
    current = folder / "approved.json"
    sample["annotation"] = read_json(current) if current.exists() else None
    sample["folder"] = project["samples"][index]
    if sample["annotation"]:
        replay_events.extend(sample["annotation"].get("replay_events", []))
    obs = sample["observation"]
    # Only facts assigned to current/past source frames enter the causal input.
    obs["execution"]["events"].extend(e for e in replay_events
        if e["t_obs"] <= obs["t_obs"] and e["odom_epoch"] == obs["odom_epoch"])
    return sample


def tagged_text(landmark, local, route, waypoint):
    wp = ";".join(f"({x:.3f},{y:.3f})" for x, y in waypoint) if waypoint else "[]"
    return (f"<landmark>{json.dumps(landmark, ensure_ascii=False)}</landmark>"
            f"<local>{local}</local><Route>{json.dumps(route, ensure_ascii=False)}</Route>"
            f"<waypoint>{wp}</waypoint><REFLINE>")


def approve(root, index, data):
    from ..structured_io import parse_structured

    root = Path(root)
    sample = sample_context(root, index)
    if data.get("source_binding") != sample["provenance"]["source_binding"]:
        raise ValueError("Wrong source binding")
    if data.get("parent_revision") != sample["parent_revision"]:
        raise ValueError("Stale parent Route; reload sample")
    if not str(data.get("annotator", "")).strip() or data.get("human_reviewed") is not True:
        raise ValueError("Named human approval is required")
    meta, frames = load_episode(sample["provenance"]["source_episode"])
    f = frames[sample["provenance"]["frame_index"]]
    if frame_binding(meta, f) != sample["provenance"]["source_binding"]:
        raise ValueError("Raw source changed after extraction")
    verify_observation_files(sample)
    state = data.get("reference_state")
    if state not in ("valid", "valid_empty"):
        raise ValueError("Choose an explicitly valid or valid_empty reference")
    points = np.asarray(data.get("control_points", []), float).reshape(-1, 2)
    if len(points) > 256 or not np.isfinite(points).all():
        raise ValueError("Invalid control points")
    line, mask = resample_line(points)
    if state == "valid" and mask.sum() < 2:
        raise ValueError("A valid line needs at least two distinct points")
    if state == "valid_empty" and len(points):
        raise ValueError("Empty reference cannot contain points")
    with np.load(root / sample["folder"] / "map.npz", allow_pickle=False) as arrays:
        check = reference_geometry(line[mask], arrays["grid"], sample["observation"]["mapview"]["resolution"])
    if check["collision"]:
        raise ValueError("Reference intersects observed inflated occupancy")
    # A causal sparse depth map is allowed to lack coverage of a human-reviewed
    # future demonstration. Unknown is neither an obstacle nor proof of an
    # invalid reference label. Keep coverage diagnostics; do not promote cells
    # to free or relax the separate online controller's checks.
    wp = anchors(line[mask])
    text = tagged_text(data["landmark"], data["local"], data["route"], wp)
    ids = [h["image_id"] for h in sample["observation"]["rgb_history"]]
    parsed = parse_structured(text, ids, structured_route=True)
    replay_kinds = data.get("replay_event_kinds", [])
    allowed = {"start", "intersection", "turn_started", "turn_completed",
               "target_visible", "stopped", "arrival_confirmed"}
    if (not isinstance(replay_kinds, list) or any(not isinstance(k, str) or k not in allowed for k in replay_kinds)
        or len(set(replay_kinds)) != len(replay_kinds)):
        raise ValueError("Invalid or duplicate replay event kinds")
    note = str(data.get("replay_event_note", "")).strip()
    if replay_kinds and not note:
        raise ValueError("Replay facts need a human evidence note; commands are not execution evidence")
    obs = sample["observation"]
    replay_events = [dict(kind=kind, image_id=obs["image_id"], t_obs=obs["t_obs"],
        odom_epoch=obs["odom_epoch"], source="human_replay_review",
        annotator=str(data["annotator"]).strip(), note=note[:2000],
        source_binding=sample["provenance"]["source_binding"], reviewed_wall=time.time())
        for kind in replay_kinds]
    # Revisions replace this frame's earlier manual facts; never silently retain unchecked facts.
    obs["execution"]["events"] = [e for e in obs["execution"]["events"]
        if not (e.get("source") in ("human_replay_review", "user_authorized_replay_correction") and e.get("image_id") == obs["image_id"])] + replay_events
    if parsed["route"]["status"] == "completed":
        kinds = {e["kind"] for e in sample["observation"]["execution"]["events"]}
        if not {"target_visible", "stopped", "arrival_confirmed"}.issubset(kinds):
            raise ValueError("Completion needs recorded target_visible, stopped and arrival_confirmed events")
    annotation = dict(revision_id=uuid.uuid4().hex, parent_revision=sample["parent_revision"],
                      source_binding=data["source_binding"], approved_wall=time.time(),
                      annotator=data["annotator"], human_reviewed=True, parsed=parsed, text=text,
                      replay_events=replay_events,
                      reference_state=state, reference_xy=line.tolist(), reference_mask=mask.tolist(),
                      waypoint=wp, control_points=points.tolist(), geometry=check,
                      unknown_review_note=data.get("unknown_review_note", ""),
                      reference_review_policy="human_review_with_map_diagnostics_v2",
                      training_eligible=True, action_state="unlabeled")
    folder = root / sample["folder"]
    write_json(folder / "revisions" / (annotation["revision_id"] + ".json"), annotation)
    write_json(folder / "approved.json", annotation)
    project = read_json(root / "project.json")
    project.update(human_episode_approved=False, status="awaiting_human_finalization")
    write_json(root / "project.json", project)
    return annotation


def export_project(root, output):
    root, output = Path(root), Path(output)
    project = read_json(root / "project.json")
    records = []
    for i in range(len(project["samples"])):
        sample = sample_context(root, i)
        a = sample["annotation"]
        if a is None or a["parent_revision"] != sample["parent_revision"]:
            raise ValueError("Missing or stale approval")
        meta, frames = load_episode(sample["provenance"]["source_episode"])
        if frame_binding(meta, frames[sample["provenance"]["frame_index"]]) != a["source_binding"]:
            raise ValueError("Raw source changed after approval")
        verify_observation_files(sample)
        record = {k: sample[k] for k in ("schema_version", "observation", "provenance")}
        record["supervision"] = {k: a[k] for k in ("text", "parsed", "reference_state", "reference_xy",
                                                   "reference_mask", "waypoint", "action_state")}
        record["supervision"].update(text_state="approved", candidates=[],
                                     reference_loss_valid=a["training_eligible"],
                                     geometry_diagnostics=a.get("geometry", {}),
                                     reference_review_policy=a.get("reference_review_policy", "legacy_unknown_gate"))
        record["provenance"].update(annotation_revision=a["revision_id"], annotator=a["annotator"])
        if "correction" in a:
            record["provenance"]["annotation_correction"] = copy.deepcopy(a["correction"])
        validate_v2(record)
        records.append(record)
    output.mkdir(parents=True, exist_ok=False)
    for i, r in enumerate(records):
        write_json(output / f"{i:08d}.json", r)
    manifest = dict(schema_version=VERSION, role="regression", optimizer_updates_permitted=False,
                    records=[dict(path=f"{i:08d}.json", sample_id=r["provenance"]["sample_id"],
                                  sha256=file_sha256(output / f"{i:08d}.json")) for i,r in enumerate(records)])
    write_json(output / "manifest.json", manifest)
    return manifest


def validate_v2(r):
    if r.get("schema_version") != VERSION:
        raise ValueError("Expected dataset schema v2")
    obs, sup, prov = r["observation"], r["supervision"], r["provenance"]
    forbidden = {"future", "oracle", "goal_pose_gt", "reference_xy", "actions", "assistance", "future_trajectory"}
    def walk(value):
        if isinstance(value, dict):
            if forbidden.intersection(value):
                raise ValueError("Privileged field in observation")
            for v in value.values(): walk(v)
        elif isinstance(value, list):
            for v in value: walk(v)
    walk(obs)
    if not prov.get("source_binding") or not obs["task"]["instruction"].strip():
        raise ValueError("Missing source/task")
    if any(h["t_obs"] > obs["t_obs"] for h in obs["rgb_history"]):
        raise ValueError("Future RGB")
    if obs.get("mapview", {}).get("source_until", obs["t_obs"]) > obs["t_obs"]:
        raise ValueError("Future map")
    if sup.get("reference_state") not in ("unlabeled", "valid", "valid_empty"):
        raise ValueError("Invalid reference label state")
    if sup["reference_state"] != "unlabeled":
        xy, mask = np.asarray(sup["reference_xy"]), np.asarray(sup["reference_mask"])
        if xy.shape != (32, 2) or mask.shape != (32,) or mask.dtype != bool or not np.isfinite(xy).all():
            raise ValueError("Invalid reference shape/value/mask")
        if not np.array_equal(mask, np.arange(32) < mask.sum()):
            raise ValueError("Non-prefix reference")
        if (sup["reference_state"] == "valid") != bool(mask.sum() >= 2):
            raise ValueError("Reference state disagrees with mask")
        if sup.get("waypoint") != anchors(xy[mask]):
            raise ValueError("Anchor/reference mismatch")
    if sup.get("text_state") == "approved":
        from ..structured_io import parse_structured
        parsed = parse_structured(sup["text"], [h["image_id"] for h in obs["rgb_history"]], structured_route=True)
        if parsed["anchors"] != sup["waypoint"]:
            raise ValueError("Text/anchor mismatch")
    if sup.get("action_state") == "unlabeled" and sup.get("candidates"):
        raise ValueError("Unlabeled actions cannot contain positive candidates")
    return r


def training_record(record):
    """Explicit reference-training adapter; preserve full text and causal context.

    Missing action arrays are compatibility padding with false masks, never GT.
    Golden/regression records may be forwarded for diagnostics, not optimizer use.
    """
    validate_v2(record)
    result = copy.deepcopy(record)
    obs, sup = result["observation"], result["supervision"]
    view = obs.get("mapview", {})
    obs.update(route_protocol="structured-v1", map_valid=bool(view.get("arrays")),
               map_image=view.get("image"), map_grid=view.get("arrays"),
               map_landmarks=view.get("landmarks", []), execution_state=obs.get("execution", {}))
    state = sup["reference_state"]
    known = state != "unlabeled" and sup.get("reference_loss_valid", True)
    sup.update(reference_known=known, reference_valid=known and state == "valid",
               text_valid=sup.get("text_state") == "approved",
               actions=np.zeros((30,2)).tolist(), action_mask=[False]*30,
               anchors=sup.get("waypoint", []))
    if state == "unlabeled":
        sup.update(reference_xy=np.zeros((32,2)).tolist(), reference_mask=[False]*32)
    result["adapter"] = "schema-v2-to-reference-input-v1"
    return result


def finalize_project(root, annotator):
    """Explicit human milestone; never called by tests/teachers as a substitute."""
    root = Path(root)
    project = read_json(root / "project.json")
    if not str(annotator).strip():
        raise ValueError("Named human finalization required")
    if len(project["samples"]) < 8:
        raise ValueError("A multi-step golden episode needs at least eight reviewed keyframes")
    for i in range(len(project["samples"])):
        s = sample_context(root, i)
        a = s["annotation"]
        if not a or a["parent_revision"] != s["parent_revision"]:
            raise ValueError("Missing or stale annotation")
    events = s["observation"]["execution"]["events"]
    # Turn count depends on the human-reviewed instruction, not the original
    # two-turn example. Optional start/turn markers remain useful audit facts.
    missing = {"target_visible", "stopped", "arrival_confirmed"} - {e["kind"] for e in events}
    if missing:
        raise ValueError("Missing completion evidence: " + ", ".join(sorted(missing)))
    project.update(human_episode_approved=True, scene_task_validated=True,
                   status="human_baseline_approved", approver=annotator,
                   final_revision=s["annotation"]["revision_id"], finalized_wall=time.time(),
                   finalization_policy="human_task_review_with_completion_evidence_v2",
                   recorded_completed_turns=len({e["image_id"] for e in events if e["kind"] == "turn_completed"}))
    write_json(root / "project.json", project)
    return project
