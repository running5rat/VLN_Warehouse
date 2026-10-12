"""Manifest-backed records and replay of versioned pre-compression semantic caches."""

import bisect
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import align_reference, delay_features, transform_points
from .schema import SemanticSnapshot, fingerprint, load_manifest, validate_record


class RecordDataset(Dataset):
    def __init__(self, manifest, split):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = load_manifest(manifest)
        self.entries = [e for e in self.manifest["records"] if e["split"] == split]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        record = validate_record(
            json.loads((self.manifest_path.parent / entry["path"]).read_text())
        )
        for field in ("sample_id", "split", "layout_family", "episode_id"):
            if record["provenance"][field] != entry[field]:
                raise ValueError("Record/manifest provenance mismatch")
        return record


def reference_targets(record, device):
    source = record["supervision"]
    types = {
        "reference_xy": torch.float32,
        "reference_mask": torch.bool,
        "reference_known": torch.bool,
        "reference_valid": torch.bool,
        "text_valid": torch.bool,
    }
    return {
        key: torch.as_tensor(source[key], dtype=dtype, device=device).unsqueeze(0)
        for key, dtype in types.items()
    }


def bev_tensor(observation, cfg):
    if not observation.get("map_valid"):
        n = round(cfg.map_size_m / cfg.map_resolution)
        return torch.zeros(4, n, n), False
    with np.load(observation["map_grid"], allow_pickle=False) as data:
        grid, age = data["grid"], data["age"]
        channels = np.stack(
            [grid == 0, grid == 1, grid == -1, np.clip(age / cfg.map_max_age, 0, 1)]
        )
    return torch.from_numpy(channels.astype(np.float32)), True


def validate_cache(index, cfg, manifest):
    if index.get("architecture") != "warehouse-mvp-v1":
        raise ValueError("Cache architecture mismatch")
    if index["manifest_hash"] != fingerprint(manifest):
        raise ValueError("Cache is stale: manifest changed")
    for key in (
        "num_reference_points",
        "coordinate_scale",
        "image_size",
        "horizon",
        "action_dt",
        "map_size_m",
        "map_resolution",
        "footprint_radius",
    ):
        if index["config"][key] != getattr(cfg, key):
            raise ValueError(f"Cache/config mismatch: {key}")


def previous_plan_tokens(obs, cfg):
    points = np.zeros((cfg.horizon, 3), np.float32)
    valid = np.zeros(cfg.horizon, bool)
    proposal = obs.get("previous_plan")
    if not proposal:
        return torch.from_numpy(points), torch.from_numpy(valid)
    if proposal.get("source") != "policy_proposal":
        raise ValueError(
            "Previous plan must be a recorded policy proposal, not demonstration labels"
        )
    if proposal["t_obs"] > obs["t_obs"] or proposal["odom_epoch"] != obs["odom_epoch"]:
        return torch.from_numpy(points), torch.from_numpy(valid)
    xy = np.asarray(proposal["xy"], dtype=np.float32)
    remaining = np.asarray(proposal["time_offsets"], dtype=np.float32) - (
        obs["t_obs"] - proposal["t_obs"]
    )
    if (
        xy.shape != (len(remaining), 2)
        or not np.isfinite(xy).all()
        or not np.isfinite(remaining).all()
    ):
        raise ValueError("Invalid recorded proposal")
    xyz = transform_points(
        np.c_[xy, np.zeros(len(xy))],
        np.linalg.inv(obs["T_odom_body"]) @ proposal["T_odom_body"],
    )
    keep = np.flatnonzero(remaining > 0)[: cfg.horizon]
    points[: len(keep), :2] = xyz[keep, :2]
    points[: len(keep), 2] = remaining[keep]
    valid[: len(keep)] = True
    return torch.from_numpy(points), torch.from_numpy(valid)


class CachedActionDataset(Dataset):
    def __init__(
        self,
        cfg,
        split,
        training=False,
        delay=None,
        condition="both",
        compensation="full",
        reference_mode="predicted",
    ):
        self.cfg, self.training, self.fixed_delay = cfg, training, delay
        self.condition, self.compensation, self.reference_mode = (
            condition,
            compensation,
            reference_mode,
        )
        self.records = RecordDataset(cfg.manifest, split)
        self.cache_dir = Path(cfg.cache_dir)
        self.cache_index = json.loads((self.cache_dir / "index.json").read_text())
        validate_cache(self.cache_index, cfg, self.records.manifest)
        self.groups = {}
        self.items = []
        for i in range(len(self.records)):
            record = self.records[i]
            obs, prov = record["observation"], record["provenance"]
            sid = prov["sample_id"]
            cache_entry = self.cache_index["records"].get(sid)
            if cache_entry is None:
                raise ValueError(f"Missing semantic cache: {sid}")
            if cache_entry["observation_hash"] != fingerprint(obs):
                raise ValueError(f"Stale observation cache: {sid}")
            key = (
                prov["layout_family"],
                prov["episode_id"],
                obs["task"]["task_id"],
                obs["odom_epoch"],
            )
            self.groups.setdefault(key, []).append((obs["t_obs"], i))
            if any(record["supervision"]["action_mask"]):
                self.items.append((i, key))
        for group in self.groups.values():
            group.sort()

    def __len__(self):
        return len(self.items)

    def cached(self, record):
        sid = record["provenance"]["sample_id"]
        entry = self.cache_index["records"][sid]
        cache = torch.load(
            self.cache_dir / entry["path"], map_location="cpu", weights_only=True
        )
        if cache["checkpoint_hash"] != self.cache_index["checkpoint_hash"]:
            raise ValueError("Mixed reference versions inside cache")
        return cache

    def __getitem__(self, index):
        i, key = self.items[index]
        current = self.records[i]
        obs = current["observation"]
        delay = self.fixed_delay
        if delay is None:
            delay = (
                float(np.random.choice(self.cfg.delay_seconds))
                if self.training
                else 0.0
            )
        group = self.groups[key]
        position = bisect.bisect_right([t for t, _ in group], obs["t_obs"] - delay) - 1
        available = position >= 0
        source = self.records[group[max(position, 0)][1]]
        src = source["observation"]
        source_cache, current_cache = self.cached(source), self.cached(current)
        age = obs["t_obs"] - src["t_obs"]
        snapshot = SemanticSnapshot(
            z_sem=None,
            refline_xy=source_cache["reference_xy"],
            refline_mask=source_cache["reference_mask"],
            refline_conf=source_cache["confidence"],
            t_obs=src["t_obs"],
            t_start=src["t_obs"],
            t_ready=src["t_obs"],
            T_odom_body_obs=src["T_odom_body"],
            odom_epoch=src["odom_epoch"],
            generation_id=source_cache["generation_id"],
            task_id=src["task"]["task_id"],
            base_route_revision=src["route_revision"],
            route_revision=src["route_revision"],
            map_version=src["map_version"],
        )
        try:
            snapshot.validate(
                t_use=obs["t_obs"],
                task_id=obs["task"]["task_id"],
                odom_epoch=obs["odom_epoch"],
                route_revision=obs["route_revision"],
                max_age=self.cfg.max_snapshot_age,
            )
        except ValueError:
            available = False
        available = available and source_cache["parse_valid"]
        ref = source_cache["reference_xy"].numpy().copy()
        mask = source_cache["reference_mask"].numpy().copy()
        confidence = source_cache["confidence"]
        if self.reference_mode == "oracle":
            sup = source["supervision"]
            ref, mask = (
                np.array(sup["reference_xy"], np.float32),
                np.array(sup["reference_mask"], bool),
            )
            # Explicit diagnostic upper bound, NEVER default deployment input.
            available = (
                position >= 0
                and age <= self.cfg.max_snapshot_age
                and sup["reference_known"]
            )
            confidence = float(sup["reference_valid"])
        if self.compensation == "full":
            ref, mask = align_reference(
                ref,
                mask,
                src["T_odom_body"],
                obs["T_odom_body"],
                src["odom_epoch"],
                obs["odom_epoch"],
            )
        delay_vector = delay_features(
            src["T_odom_body"], obs["T_odom_body"], max(age, 0.0)
        )
        if self.compensation == "none":
            delay_vector[:] = 0
        elif self.compensation == "time":
            delay_vector[1:] = [0, 0, 0, 1]
        sem_valid = available and self.condition != "reference"
        mask &= available and self.condition != "semantic"
        if self.training:
            ref += (
                np.random.normal(0, self.cfg.reference_noise_std, ref.shape).astype(
                    np.float32
                )
                * mask[:, None]
            )
            if np.random.rand() < self.cfg.condition_dropout:
                mask[:] = False
            if np.random.rand() < self.cfg.condition_dropout:
                sem_valid = False
        map_obs = dict(obs)
        map_mode = self.cache_index["map_mode"]
        if map_mode == "removed":
            map_obs["map_valid"] = False
        elif map_mode == "shuffled":
            map_obs["map_grid"] = self.cache_index["records"][
                current["provenance"]["sample_id"]
            ]["map_grid"]
        bev, map_valid = bev_tensor(map_obs, self.cfg)
        sup = current["supervision"]
        previous_plan, previous_plan_mask = previous_plan_tokens(obs, self.cfg)
        return {
            "visual": current_cache["visual"].float(),
            "visual_mask": current_cache["visual_mask"].bool(),
            "kv": source_cache["raw_kv"].float(),
            "kv_mask": source_cache["kv_mask"].bool(),
            "bev": bev,
            "map_valid": torch.tensor(map_valid),
            "semantic_valid": torch.tensor(sem_valid),
            "state": torch.tensor(obs["robot_state"], dtype=torch.float32),
            "state_mask": torch.tensor(obs["robot_state_mask"], dtype=torch.bool),
            "reference_xy": torch.from_numpy(ref.astype(np.float32)),
            "reference_mask": torch.from_numpy(mask),
            "reference_confidence": torch.tensor(confidence, dtype=torch.float32),
            "delay": torch.from_numpy(delay_vector),
            "previous_plan": previous_plan,
            "previous_plan_mask": previous_plan_mask,
            "actions": torch.tensor(sup["actions"], dtype=torch.float32),
            "action_mask": torch.tensor(sup["action_mask"], dtype=torch.bool),
            "sample_id": current["provenance"]["sample_id"],
            "age": age,
            "snapshot_valid": available,
            "map_grid_path": obs["map_grid"],
            "record": current,
        }


def action_collate(samples):
    result = {}
    for key in samples[0]:
        values = [s[key] for s in samples]
        if key in ("visual", "kv", "visual_mask", "kv_mask"):
            result[key] = torch.nn.utils.rnn.pad_sequence(
                values, batch_first=True, padding_value=0
            )
        elif isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def to_device(batch, device):
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
    }
