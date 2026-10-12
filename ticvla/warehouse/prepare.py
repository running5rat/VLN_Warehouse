"""Convert timestamped episodes to causal records, splitting BEFORE windows.

Raw episode.json: schema_version, layout_family, episode_id, task (id/instruction),
frames (or frames_file JSONL). Frames contain t_obs, rgb, depth?, K?,
T_body_camera?, T_odom_body, odom_epoch, robot_state? ([v, omega]).
Supervision.local_goal_odom is an explicitly annotated local task goal, never
copied into deployment inputs. It must be in observed free space to label a line.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .config import WarehouseConfig, load_config
from .geometry import (
    LocalMapBuilder,
    astar_reference,
    grid_to_xy,
    render_map_image,
    sample_reference,
    transform_points,
    traversable,
    xy_to_grid,
)
from .schema import SCHEMA_VERSION, fingerprint, load_manifest, validate_record


def path_at(root, value):
    return str((root / str(value)).resolve())


def causal_state(frames, i):
    now = frames[i]
    if "robot_state" in now:
        state = np.asarray(now["robot_state"], np.float32)
        if state.shape != (2,) or not np.isfinite(state).all():
            raise ValueError("robot_state must be finite [v, omega]")
        return state.tolist(), [True, True]
    if i == 0 or frames[i - 1]["odom_epoch"] != now["odom_epoch"]:
        return [0.0, 0.0], [False, False]
    dt = now["t_obs"] - frames[i - 1]["t_obs"]
    if dt <= 0:
        raise ValueError("Non-monotonic sensor time")
    delta = np.linalg.inv(frames[i - 1]["T_odom_body"]) @ now["T_odom_body"]
    return [
        float(delta[0, 3] / dt),
        float(math.atan2(delta[1, 0], delta[0, 0]) / dt),
    ], [True, True]


def time_actions(frames, i, cfg):
    """Interpolate only SUPERVISION at physical times, with a gap/epoch mask."""
    result = np.zeros((cfg.horizon, 2), np.float32)
    mask = np.zeros(cfg.horizon, bool)
    now = frames[i]
    times = np.array([f["t_obs"] for f in frames])
    inverse = np.linalg.inv(now["T_odom_body"])
    for k in range(cfg.horizon):
        t = now["t_obs"] + (k + 1) * cfg.action_dt
        right = int(np.searchsorted(times, t, side="left"))
        if right >= len(frames):
            break
        left = max(i, right - 1)
        if any(frames[j]["odom_epoch"] != now["odom_epoch"] for j in (left, right)):
            continue
        interval = times[right] - times[left]
        if interval > max(0.5, 3 * cfg.action_dt):
            continue
        fraction = (t - times[left]) / interval if interval else 0.0
        p = (1 - fraction) * np.asarray(frames[left]["T_odom_body"])[
            :3, 3
        ] + fraction * np.asarray(frames[right]["T_odom_body"])[:3, 3]
        result[k] = transform_points(p[None], inverse)[0, :2]
        mask[k] = True
    return result, mask


def select_history(frames, i, offsets):
    t = frames[i]["t_obs"]
    history = []
    for offset in sorted(offsets, reverse=True):
        candidates = [
            f
            for f in frames[: i + 1]
            if f["t_obs"] <= t - offset + 1.0e-6
            and f["odom_epoch"] == frames[i]["odom_epoch"]
        ]
        if candidates:
            frame = candidates[-1]
            if not history or history[-1]["t_obs"] != frame["t_obs"]:
                history.append(
                    {
                        "path": frame["rgb"],
                        "t_obs": frame["t_obs"],
                        "T_odom_body": frame["T_odom_body"],
                    }
                )
    return history


def derived_task(grid, cfg):
    """Reference-only templates from CURRENT known geometry, never future motion."""
    allowed = traversable(grid, cfg.map_resolution, cfg.footprint_radius)
    cells = np.argwhere(allowed)
    xy = grid_to_xy(cells, len(grid), cfg.map_resolution)
    if len(xy) == 0:
        return "Wait in place.", None
    eligible = (xy[:, 0] > 1.0) & (np.linalg.norm(xy, axis=-1) < 6.0)
    candidates = xy[eligible]
    candidates = sorted(candidates, key=lambda p: float(-p[0] + 0.2 * abs(p[1])))
    for goal in candidates[:100]:
        if len(astar_reference(grid, goal, cfg.map_resolution, cfg.footprint_radius)):
            direction = (
                "left"
                if goal[1] > 0.75
                else "right"
                if goal[1] < -0.75
                else "straight ahead"
            )
            return f"Move {direction} through the visible free passage.", np.asarray(
                goal
            )
    return "Wait in place.", None


def prepare_episodes(
    episode_paths,
    output,
    split_map,
    cfg,
    split_unit="layout_family",
    stride=5,
    derive_tasks=False,
):
    output = Path(output).resolve()
    if (output / "manifest.json").exists():
        raise FileExistsError(
            "Use a new output directory; existing manifests are immutable"
        )
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for episode_path in episode_paths:
        episode_path = Path(episode_path)
        episode = json.loads(episode_path.read_text())
        root = episode_path.parent
        layout, episode_id = episode["layout_family"], episode["episode_id"]
        key = episode[split_unit]
        if key not in split_map:
            raise ValueError(f"No explicit split for {split_unit}={key}")
        split = split_map[key]
        if "frames" in episode:
            frames = episode["frames"]
        else:
            frames = [
                json.loads(x)
                for x in (root / episode["frames_file"]).read_text().splitlines()
                if x.strip()
            ]
        if any(b["t_obs"] <= a["t_obs"] for a, b in itertools.pairwise(frames)):
            raise ValueError("Raw frame times must be strictly increasing")
        for f in frames:
            f["rgb"] = path_at(root, f["rgb"])
        builder = LocalMapBuilder(cfg)
        for i, frame in enumerate(frames):
            if frame.get("depth"):
                builder.add(
                    depth=np.load(root / frame["depth"], allow_pickle=False),
                    K=frame["K"],
                    T_body_camera=frame["T_body_camera"],
                    T_odom_body=frame["T_odom_body"],
                    t_obs=frame["t_obs"],
                    odom_epoch=frame["odom_epoch"],
                )
            if i % stride:
                continue
            sample_id = fingerprint([layout, episode_id, frame["t_obs"]])[:24]
            folder = output / "samples" / sample_id
            folder.mkdir(parents=True)
            grid, age = builder.render(
                frame["T_odom_body"], frame["t_obs"], frame["odom_epoch"]
            )
            map_valid = bool(np.any(grid != -1))
            np.savez_compressed(
                folder / "map.npz", grid=grid, age=age, resolution=cfg.map_resolution
            )
            render_map_image(grid, cfg.map_resolution, cfg.image_size).save(
                folder / "map.png"
            )
            task = dict(episode.get("task", {}))
            task.setdefault("task_id", episode_id)
            goal = episode.get("supervision", {}).get("local_goal_odom")
            goal_xy = None
            if goal is not None:
                goal = np.array(goal, dtype=float)
                if goal.shape == (2,):
                    goal = np.r_[goal, 0.0]
                goal_xy = transform_points(
                    goal[None], np.linalg.inv(frame["T_odom_body"])
                )[0, :2]
            if derive_tasks:
                task["instruction"], goal_xy = derived_task(grid, cfg)
                task["task_id"] = sample_id  # independent local reference-only task
            if not task.get("instruction"):
                raise ValueError(
                    "Annotated task instruction is required, or use --derive-local-tasks"
                )
            task = {
                "task_id": task["task_id"],
                "instruction": task["instruction"],
                "hard_constraints": task.get("hard_constraints", []),
                "task_type": "INSTRUCTION",
            }
            path, reference_known = np.empty((0, 2)), False
            if map_valid and goal_xy is not None:
                cell = xy_to_grid(goal_xy, len(grid), cfg.map_resolution)
                if ((cell >= 0) & (cell < len(grid))).all() and grid[tuple(cell)] == 0:
                    reference_known = True
                    path = astar_reference(
                        grid, goal_xy, cfg.map_resolution, cfg.footprint_radius
                    )
            wait = bool(episode.get("supervision", {}).get("wait", False)) or (
                derive_tasks and goal_xy is None
            )
            if wait:
                reference_known = True
            ref, ref_mask = sample_reference(
                path, cfg.num_reference_points, cfg.reference_spacing
            )
            actions, action_mask = time_actions(frames, i, cfg)
            # Automatically derived tasks are not necessarily the demonstrated task.
            if derive_tasks or not episode.get("supervision", {}).get(
                "actions_task_aligned", False
            ):
                action_mask[:] = False
            # Exclude reverse/sideways demonstrations from this forward-only MVP.
            if np.any(actions[action_mask, 0] < -0.1):
                action_mask[:] = False
            state, state_mask = causal_state(frames, i)
            history = select_history(frames, i, cfg.history_offsets)
            n = int(ref_mask.sum())
            anchors = (
                ref[np.rint(np.linspace(0, n - 1, 5)).astype(int)].tolist() if n else []
            )
            supervision = {
                "reference_xy": ref.tolist(),
                "reference_mask": ref_mask.tolist(),
                "reference_known": reference_known,
                "reference_valid": bool(n),
                "anchors": anchors,
                "route": (
                    task["instruction"]
                    if n or wait
                    else "Wait until the visible passage is clear; then "
                    + task["instruction"]
                ),
                "text_valid": reference_known,
                "actions": actions.tolist(),
                "action_mask": action_mask.tolist(),
                "label_source": "observed_map_astar+recorded_motion",
            }
            obs = {
                "frame_id": "body_flu",
                "t_obs": frame["t_obs"],
                "T_odom_body": frame["T_odom_body"],
                "odom_epoch": frame["odom_epoch"],
                "rgb_history": history,
                "rgb_current": frame["rgb"],
                "map_grid": str(folder / "map.npz"),
                "map_image": str(folder / "map.png"),
                "map_valid": map_valid,
                "map_version": sample_id,
                "map_resolution": cfg.map_resolution,
                "task": task,
                "route": task["instruction"],
                "route_revision": 0,
                "execution_state": {},
                "robot_state": state,
                "robot_state_mask": state_mask,
                "motion_direction": "forward",
                "previous_plan": frame.get("previous_plan"),
            }
            provenance = {
                "layout_family": layout,
                "episode_id": episode_id,
                "sample_id": sample_id,
                "split": split,
                "raw_episode": str(episode_path.resolve()),
                "pose_source": episode.get("pose_source", "unspecified"),
            }
            record = {
                "schema_version": SCHEMA_VERSION,
                "observation": obs,
                "supervision": supervision,
                "provenance": provenance,
            }
            validate_record(record)
            (folder / "record.json").write_text(json.dumps(record, indent=2))
            records.append(
                {
                    **provenance,
                    "path": str((folder / "record.json").relative_to(output)),
                }
            )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "split_unit": split_unit,
        "config": cfg.to_dict(),
        "records": records,
        "generalization": "unseen_layout"
        if split_unit == "layout_family"
        else "episode_only",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    load_manifest(path)
    from .audit import audit_manifest

    (output / "audit.json").write_text(json.dumps(audit_manifest(path), indent=2))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes", nargs="+", required=True, help="episode.json files"
    )
    parser.add_argument(
        "--splits",
        required=True,
        help="JSON mapping layout family (or episode) to train/val/test",
    )
    parser.add_argument(
        "--split-unit", choices=["layout_family", "episode_id"], default="layout_family"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--derive-local-tasks", action="store_true")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("stride must be positive")
    cfg = load_config(args.config) if args.config else WarehouseConfig()
    print(
        prepare_episodes(
            args.episodes,
            args.output,
            json.loads(Path(args.splits).read_text()),
            cfg,
            args.split_unit,
            args.stride,
            args.derive_local_tasks,
        )
    )


if __name__ == "__main__":
    main()
