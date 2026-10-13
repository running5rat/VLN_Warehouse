"""Task lifecycle, source-bound snapshots, observed-map planning and control."""

import copy
from dataclasses import asdict
import math
import numpy as np
from scipy.ndimage import binary_dilation, label
from ..schema import SemanticSnapshot, TaskContract, TaskState
from ..geometry import (
    astar_reference,
    collision_free,
    grid_to_xy,
    traversable,
    transform_points,
)


class NavigationState:
    def __init__(self, task_id, mode, target=None):
        instructions = {
            "explore": "Explore reachable frontiers of the observed warehouse map.",
            "landmark": f"Navigate to a reachable free neighborhood of landmark {target}.",
            "follow": "Follow the explicitly bound person at low speed, keep 1.5 metres separation; stop if lost.",
        }
        kind = {
            "explore": "INSTRUCTION",
            "landmark": "GOAL",
            "follow": "PERSON_FOLLOW",
        }[mode]
        self.state = TaskState(
            TaskContract(
                task_id,
                instructions[mode],
                ("Use observed free space only.", "Preserve target identity."),
                kind,
            ),
            instructions[mode],
        )
        self.mode, self.target, self.snapshot = mode, target, None
        self.last_generation = -1
        self.status = "waiting_observation"
        self.state.execution = {"status": self.status, "target": target}

    def commit(self, result, obs, generation, start, ready, now, max_age):
        if not result["valid"]:
            raise ValueError(result.get("error") or "Invalid VLM output")
        if (
            obs["task"]["task_id"] != self.state.contract.task_id
            or obs["route_revision"] != self.state.route_revision
        ):
            raise ValueError("Stale task/Route response")
        if generation <= self.last_generation:
            raise ValueError("Repeated/old generation")
        z, xy, mask = (
            np.asarray(result["z_sem"]),
            np.asarray(result["reference_xy"]),
            np.asarray(result["reference_mask"]),
        )
        if (
            z.shape != (16, 512)
            or xy.shape != (32, 2)
            or mask.shape != (32,)
            or mask.dtype != bool
        ):
            raise ValueError("Snapshot tensor interface mismatch")
        if (
            not np.isfinite(z).all()
            or not np.isfinite(xy).all()
            or not np.isfinite(result["confidence"])
        ):
            raise ValueError("Nonfinite snapshot tensor")
        if not np.array_equal(mask, np.arange(32) < mask.sum()):
            raise ValueError("Invalid reference prefix mask")
        mode = result["inference_mode"]
        if mode == "PREFILL_FOLLOW" and (
            self.mode != "follow"
            or not obs.get("target_visible")
            or not obs.get("target_binding_id")
        ):
            raise ValueError("PREFILL requires a bound visible person")
        route = result["parsed"]["route"] if mode == "TEXT" else self.state.route
        next_state = copy.deepcopy(self.state)
        if mode == "TEXT":
            next_state.commit_route(
                route, obs["task"]["task_id"], obs["route_revision"]
            )
        packet = SemanticSnapshot(
            result["z_sem"],
            result["reference_xy"],
            result["reference_mask"],
            result["confidence"],
            obs["source_monotonic"],
            start,
            ready,
            obs["T_odom_body"],
            obs["odom_epoch"],
            generation,
            obs["task"]["task_id"],
            obs["route_revision"],
            next_state.route_revision,
            obs["map_version"],
            mode,
            target_binding_id=obs.get("target_binding_id"),
            target_visible=obs.get("target_visible"),
        )
        packet.validate(
            t_use=now,
            task_id=self.state.contract.task_id,
            odom_epoch=obs["odom_epoch"],
            route_revision=next_state.route_revision,
            max_age=max_age,
            last_generation=self.last_generation,
        )
        # Single writer commits state and packet together, after every check passes.
        self.state, self.snapshot, self.last_generation = next_state, packet, generation
        return packet

    def valid_snapshot(self, now, epoch, max_age):
        if self.snapshot is None:
            return None, "no_snapshot"
        try:
            self.snapshot.validate(
                t_use=now,
                task_id=self.state.contract.task_id,
                odom_epoch=epoch,
                route_revision=self.state.route_revision,
                max_age=max_age,
                last_generation=self.last_generation,
            )
        except ValueError as exc:
            return None, str(exc)
        return self.snapshot, None

    def record(self):
        return asdict(self.state)


def select_landmark(landmarks, query, epoch):
    matches = [
        x
        for x in landmarks
        if x.get("submap_id") == epoch
        and (
            x["landmark_uid"] == query
            or x.get("identity_text") == query
            or x.get("name") == query
        )
    ]
    if len(matches) != 1:
        return (
            None,
            "landmark_not_observed" if not matches else "ambiguous_landmark_use_uid",
        )
    return matches[0], None


def reachable_cells(grid, resolution, footprint):
    allowed = traversable(grid, resolution, footprint)
    components, _ = label(allowed)
    origin = len(grid) // 2
    component = components[origin, origin]
    return (
        np.argwhere(components == component)
        if component
        else np.empty((0, 2), dtype=int)
    )


def neighborhood_path(
    grid, goal, resolution, footprint, standoff=1.0, *, reachable=None
):
    rc = (
        reachable_cells(grid, resolution, footprint) if reachable is None else reachable
    )
    if not len(rc):
        return np.empty((0, 2), np.float32)
    xy = grid_to_xy(rc, len(grid), resolution)
    distances = np.linalg.norm(xy - np.asarray(goal), axis=1)
    valid = (distances >= standoff) & (distances < standoff + 0.8)
    choices = xy[valid]
    order = np.argsort(np.linalg.norm(choices, axis=1), kind="stable")
    for candidate in choices[order[:30]]:
        path = astar_reference(grid, candidate, resolution, footprint)
        if len(path):
            return path
    # A partial path is allowed only to an already reachable cell that reduces
    # target distance. It does not declare arrival at the target neighborhood.
    progress = np.linalg.norm(np.asarray(goal)) - np.linalg.norm(
        xy - np.asarray(goal), axis=1
    )
    if len(progress) and progress.max() > resolution * 0.5:
        return astar_reference(grid, xy[int(progress.argmax())], resolution, footprint)
    return np.empty((0, 2), np.float32)


def frontier_path(grid, resolution, footprint, blocked=()):
    # Frontier itself abuts unknown and is not footprint-safe; approach from known-free side.
    frontier = (grid == 0) & binary_dilation(grid == -1)
    components, count = label(frontier)
    candidates = []
    reachable = reachable_cells(grid, resolution, footprint)
    if len(reachable) < 2:
        return np.empty((0, 2), np.float32)
    for i in range(1, count + 1):
        rc = np.argwhere(components == i)
        if len(rc) < 3:
            continue
        representative = rc[np.argmin(np.linalg.norm(rc - rc.mean(0), axis=1))]
        goal = grid_to_xy(representative, len(grid), resolution)
        if any(np.linalg.norm(goal - x) < 0.8 for x in blocked):
            continue
        path = neighborhood_path(
            grid, goal, resolution, footprint, footprint + 0.15, reachable=reachable
        )
        if len(path) < 2 or np.linalg.norm(path[-1]) < 0.5:
            continue
        length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        score = min(len(rc) * resolution, 3.0) / (1.0 + length)
        candidates.append((score, i, path))
    if candidates:
        return max(candidates, key=lambda x: (x[0], -x[1]))[2]
    # Narrow observed clearance can support one cell of motion before a longer
    # frontier approach is visible. Never extend this prefix into unknown cells.
    xy = grid_to_xy(reachable, len(grid), resolution)
    if len(xy) > 1:
        target = xy[int(np.linalg.norm(xy, axis=1).argmax())]
        if np.linalg.norm(target) >= resolution * 0.9:
            return astar_reference(grid, target, resolution, footprint)
    return np.empty((0, 2), np.float32)


def check_candidates(
    candidates, grid, reference, resolution=0.1, footprint=0.35, dt=0.1, max_speed=0.3
):
    candidates = np.asarray(candidates)
    reports, choices = [], []
    for index, path in enumerate(candidates.reshape(-1, candidates.shape[-2], 2)):
        reason = None
        points = np.concatenate((np.zeros((1, 2)), path))
        velocity = np.diff(points, axis=0) / dt
        speed = np.linalg.norm(velocity, axis=1)
        heading = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
        if not np.isfinite(path).all():
            reason = "nonfinite"
        elif speed.max(initial=0) > max_speed + 0.02:
            reason = "speed_limit"
        elif (
            np.linalg.norm(np.diff(velocity, axis=0), axis=1).max(initial=0) / dt > 0.8
        ):
            reason = "acceleration_limit"
        elif (
            np.abs(np.diff(heading))[np.minimum(speed[:-1], speed[1:]) > 0.03] / dt
        ).max(initial=0) > 0.8:
            reason = "turn_rate_limit"
        elif (path[:, 0] < -0.05).any():
            reason = "reverse_sensor_coverage_unavailable"
        elif not collision_free(points, grid, resolution, footprint):
            reason = "occupied_or_unknown"
        score = None
        if reason is None:
            deviation = (
                float(
                    np.linalg.norm(path[:, None] - reference[None], axis=-1)
                    .min(1)
                    .mean()
                )
                if len(reference)
                else 0.0
            )
            score = float(
                np.linalg.norm(path[-1])
                - 2 * deviation
                - 0.1 * np.square(np.diff(velocity, axis=0)).sum()
            )
            choices.append((score, -index, path))
        reports.append(
            dict(
                index=index,
                mode=index // candidates.shape[-3],
                sample=index % candidates.shape[-3],
                reason=reason,
                score=score,
            )
        )
    chosen = max(choices, key=lambda x: (x[0], x[1])) if choices else None
    return (chosen[2] if chosen else None), reports


def pursuit(path, grid, resolution=0.1, footprint=0.35, speed=0.3):
    path = np.asarray(path)
    if len(path) < 2 or not np.isfinite(path).all():
        return (0.0, 0.0), "no_valid_path"
    distance = np.linalg.norm(path, axis=1)
    if distance[-1] < 0.02:
        return (0.0, 0.0), "neighborhood_reached"
    target = path[
        min(int(np.searchsorted(np.maximum.accumulate(distance), 0.6)), len(path) - 1)
    ]
    angle = math.atan2(target[1], target[0])
    v = (
        float(min(speed, distance[-1]) * max(0, math.cos(angle)))
        if abs(angle) < 0.65
        else 0.0
    )
    curvature = 2 * target[1] / max(float(np.square(target).sum()), 1e-6)
    w = float(np.clip(v * curvature if v > 0 else 2 * angle, -0.6, 0.6))
    # Check the controller's actual circular arc, not only the planner polyline.
    t = np.linspace(0, 0.7, 15)
    arc = (
        np.stack((v * t, np.zeros_like(t)), -1)
        if abs(w) < 1e-5
        else np.stack((v * np.sin(w * t) / w, v * (1 - np.cos(w * t)) / w), -1)
    )
    if not collision_free(arc, grid, resolution, footprint):
        return (0.0, 0.0), "shield_occupied_or_unknown"
    return (v, w), "tracking"


class PersonBinding:
    def __init__(self):
        self.binding_id = None
        self.position = None
        self.bbox = None
        self.last_time = None
        self.visible = False
        self.status = "waiting_manual_roi"
        self.confirmations = 0

    def update(self, instances, source, roi=None):
        stamp = source["t_obs"]
        if self.last_time is not None and stamp <= self.last_time:
            return self.record(source)
        candidates = []
        for inst in instances:
            if inst.get("position_odom") is None:
                continue
            pos = np.asarray(inst["position_odom"])
            if roi is not None:
                a, b = np.asarray(roi), np.asarray(inst["bbox"])
                lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
                inter = np.maximum(hi - lo, 0).prod()
                overlap = inter / max(
                    1, (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - inter
                )
                if overlap < 0.25:
                    continue
            elif self.position is not None and self.last_time is not None:
                dt = stamp - self.last_time
                if dt > 2 and self.confirmations != 1:
                    continue
                if np.linalg.norm(pos[:2] - self.position[:2]) > (
                    0.3 if dt > 2 else 0.4 + 0.4 * dt
                ):
                    continue
                # Independent image-space continuity gate; ambiguity still stops.
                a, b = np.asarray(self.bbox), np.asarray(inst["bbox"])
                centre_shift = np.linalg.norm((a[:2] + a[2:] - b[:2] - b[2:]) / 2)
                if centre_shift > 1.5 * max(
                    np.linalg.norm(a[2:] - a[:2]), np.linalg.norm(b[2:] - b[:2])
                ):
                    continue
            else:
                continue
            candidates.append((inst, pos))
        if len(candidates) != 1:
            self.visible = False
            self.confirmations = 0
            self.status = (
                "ambiguous"
                if candidates
                else "lost"
                if self.binding_id
                else "waiting_manual_roi"
            )
        else:
            inst, pos = candidates[0]
            if self.binding_id is None:
                self.binding_id = source["image_id"] + ":manual-person"
            self.position, self.bbox, self.last_time = pos, inst["bbox"], stamp
            self.visible = True
            self.confirmations += 1
            self.status = "visible"
        return self.record(source)

    def record(self, source):
        return dict(
            binding_id=self.binding_id,
            visible=self.visible,
            status=self.status,
            position_odom=self.position.tolist() if self.position is not None else None,
            bbox=self.bbox,
            confirmations=self.confirmations,
            last_t_obs=self.last_time,
            source_image_id=source["image_id"],
        )


def align_timed_candidates(paths, source_pose, current_pose, elapsed, dt):
    """Resample to current physical time before source->current coordinate alignment."""
    paths = np.asarray(paths)
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError("Invalid plan elapsed time")
    horizon = paths.shape[-2]
    times = np.arange(horizon + 1) * dt
    query = elapsed + np.arange(1, horizon + 1) * dt
    query = query[query <= times[-1] + 1e-8]
    flat = paths.reshape(-1, horizon, 2)
    results = []
    tf = np.linalg.inv(current_pose) @ source_pose
    for path in flat:
        path = np.concatenate((np.zeros((1, 2)), path))
        xy = np.stack([np.interp(query, times, path[:, i]) for i in range(2)], -1)
        results.append(transform_points(np.c_[xy, np.zeros(len(xy))], tf)[:, :2])
    return np.asarray(results).reshape(*paths.shape[:-2], len(query), 2)
