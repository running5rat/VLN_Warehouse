"""Deployment observations, labels and provenance have separate namespaces."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TaskContract:
    task_id: str
    instruction: str
    hard_constraints: tuple[str, ...] = ()
    task_type: str = "INSTRUCTION"


@dataclass
class TaskState:
    contract: TaskContract
    route: str | dict
    route_revision: int = 0
    clause_id: str = "local-0"
    execution: dict = field(default_factory=dict)

    def commit_route(self, route, task_id, base_revision):
        if task_id != self.contract.task_id or base_revision != self.route_revision:
            raise ValueError("Route belongs to a different task or stale revision")
        if not route or (isinstance(route, str) and not route.strip()):
            raise ValueError("Empty Route")
        self.route = copy.deepcopy(route)
        self.route_revision += 1


@dataclass
class SemanticSnapshot:
    z_sem: object
    refline_xy: object
    refline_mask: object
    refline_conf: float
    t_obs: float
    t_start: float
    t_ready: float
    T_odom_body_obs: object
    odom_epoch: str
    generation_id: int
    task_id: str
    base_route_revision: int
    route_revision: int
    map_version: str
    inference_mode: str = "TEXT"
    motion_frame_or_direction: str = "forward"
    target_binding_id: str | None = None
    target_visible: bool | None = None
    quality_validated: bool = False
    reference_validation: dict = field(default_factory=dict)
    reference_usable: bool = True

    def validate(
        self,
        *,
        t_use,
        task_id,
        odom_epoch,
        route_revision,
        max_age,
        last_generation=-1,
        map_changed_on_route=False,
    ):
        if not all(
            math.isfinite(t) for t in (self.t_obs, self.t_start, self.t_ready, t_use)
        ):
            raise ValueError("Nonfinite snapshot clock")
        if not self.t_obs <= self.t_start <= self.t_ready <= t_use:
            raise ValueError("Snapshot timestamps are not causal")
        if self.task_id != task_id or self.odom_epoch != odom_epoch:
            raise ValueError("Snapshot task/odom epoch mismatch")
        if (
            self.route_revision != route_revision
            or self.generation_id < last_generation
        ):
            raise ValueError("Stale Route/generation")
        if t_use - self.t_obs > max_age or map_changed_on_route:
            raise ValueError("Expired or invalidated snapshot")
        if self.inference_mode not in ("TEXT", "PREFILL_FOLLOW"):
            raise ValueError("Unsupported inference mode")


def validate_record(record):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported warehouse record schema")
    obs = record["observation"]
    for key in (
        "task",
        "t_obs",
        "odom_epoch",
        "T_odom_body",
        "rgb_history",
        "route",
        "route_revision",
        "robot_state",
        "robot_state_mask",
    ):
        if key not in obs:
            raise ValueError(f"Missing observation.{key}")
    pose = np.asarray(obs["T_odom_body"])
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("T_odom_body must be finite 4x4")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1.0e-5) or not np.allclose(
        pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1.0e-3
    ):
        raise ValueError("T_odom_body is not a rigid transform")
    if not math.isfinite(obs["t_obs"]):
        raise ValueError("Observation time must be finite")
    if (
        len(obs["robot_state"]) != 2
        or len(obs["robot_state_mask"]) != 2
        or not np.isfinite(obs["robot_state"]).all()
    ):
        raise ValueError("Expected finite [v, omega] and availability mask")
    for frame in obs["rgb_history"]:
        if frame["t_obs"] > obs["t_obs"]:
            raise ValueError("Future image leaked into observation")
    for forbidden in ("future", "oracle", "goal_pose_gt", "reference_xy", "actions"):
        if forbidden in obs:
            raise ValueError(f"Privileged field in observation: {forbidden}")
    for key in ("layout_family", "episode_id", "sample_id", "split"):
        if key not in record["provenance"]:
            raise ValueError(f"Missing provenance.{key}")
    sup = record["supervision"]
    for points_key, mask_key in (
        ("reference_xy", "reference_mask"),
        ("actions", "action_mask"),
    ):
        points, mask = np.asarray(sup[points_key]), np.asarray(sup[mask_key], bool)
        if points.shape != (len(mask), 2) or not np.isfinite(points).all():
            raise ValueError(f"Invalid supervised {points_key}")
    mask = np.asarray(sup["reference_mask"], bool)
    if not np.array_equal(mask, np.arange(len(mask)) < mask.sum()):
        raise ValueError("Reference validity mask must be a prefix")
    return record


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest schema")
    groups, episodes, ids = {}, {}, set()
    grouping = manifest.get("split_unit", "layout_family")
    for entry in manifest["records"]:
        if entry["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        if entry["sample_id"] in ids:
            raise ValueError("Duplicate sample ID")
        ids.add(entry["sample_id"])
        for table, key in (
            (groups, entry[grouping]),
            (episodes, (entry["layout_family"], entry["episode_id"])),
        ):
            if key in table and table[key] != entry["split"]:
                raise ValueError("Layout/episode leakage between splits")
            table[key] = entry["split"]
    return manifest
