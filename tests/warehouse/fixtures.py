"""Small synthetic integration fixtures; not navigation training evidence."""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.geometry import planar_pose
from ticvla.warehouse.schema import fingerprint


def make_fixture(root):
    root = Path(root)
    cfg = WarehouseConfig(
        device="cpu",
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        context_queries=4,
        num_control_points=4,
        num_reference_points=8,
        reference_hidden_dim=16,
        horizon=6,
        map_size_m=4.0,
        map_resolution=0.1,
        manifest=str(root / "manifest.json"),
        cache_dir=str(root / "cache"),
        output_dir=str(root / "outputs"),
        model_path="/data/ticvla/models/InternVL3-1B",
        epochs=1,
        max_steps=2,
        batch_size=2,
    )
    (root / "cache").mkdir(parents=True)
    Image.new("RGB", (64, 64), (160, 160, 160)).save(root / "rgb.png")
    Image.new("RGB", (448, 448), "white").save(root / "map.png")
    np.savez_compressed(
        root / "map.npz",
        grid=np.zeros((40, 40), np.int8),
        age=np.zeros((40, 40), np.float32),
    )
    entries = []
    caches = {}
    for split in ("train", "val", "test"):
        for i in range(3):
            sid = f"{split}-{i}"
            t = float(i)
            pose = planar_pose(i * 0.1, 0, 0.1 * i).tolist()
            obs = {
                "t_obs": t,
                "T_odom_body": pose,
                "odom_epoch": split,
                "task": {
                    "task_id": split,
                    "instruction": "Move straight forward.",
                    "hard_constraints": [],
                },
                "route": "Move straight forward.",
                "route_revision": 0,
                "execution_state": {},
                "rgb_history": [
                    {"path": str(root / "rgb.png"), "t_obs": t, "T_odom_body": pose}
                ],
                "rgb_current": str(root / "rgb.png"),
                "map_image": str(root / "map.png"),
                "map_grid": str(root / "map.npz"),
                "map_valid": True,
                "map_version": sid,
                "robot_state": [0.1, 0.0],
                "robot_state_mask": [True, True],
                "motion_direction": "forward",
                "previous_plan": None,
            }
            ref = np.stack((np.linspace(0, 1.4, 8), np.zeros(8)), -1).astype(np.float32)
            actions = np.stack((np.linspace(0.01, 0.06, 6), np.zeros(6)), -1).astype(
                np.float32
            )
            sup = {
                "reference_xy": ref.tolist(),
                "reference_mask": [True] * 8,
                "reference_known": True,
                "reference_valid": True,
                "text_valid": True,
                "anchors": ref[[0, 2, 4, 5, 7]].tolist(),
                "route": "Move straight forward.",
                "actions": actions.tolist(),
                "action_mask": [True] * 6,
            }
            prov = {
                "sample_id": sid,
                "layout_family": split,
                "episode_id": split,
                "split": split,
            }
            record = {
                "schema_version": 1,
                "observation": obs,
                "supervision": sup,
                "provenance": prov,
            }
            (root / (sid + ".json")).write_text(json.dumps(record))
            entries.append({**prov, "path": sid + ".json"})
            cache = {
                "checkpoint_hash": "fixture",
                "generation_id": i,
                "raw_kv": torch.randn(5 + i, 8),
                "kv_mask": torch.ones(5 + i, dtype=torch.bool),
                "visual": torch.randn(4, 12),
                "visual_mask": torch.ones(4, dtype=torch.bool),
                "reference_xy": torch.from_numpy(ref),
                "reference_mask": torch.ones(8, dtype=torch.bool),
                "confidence": 0.8,
                "parse_valid": True,
                "text": "fixture",
                "elapsed_s": 0.0,
            }
            torch.save(cache, root / "cache" / (sid + ".pt"))
            caches[sid] = {
                "path": sid + ".pt",
                "observation_hash": fingerprint(obs),
                "map_grid": obs["map_grid"],
            }
    manifest = {
        "schema_version": 1,
        "split_unit": "layout_family",
        "records": entries,
        "config": cfg.to_dict(),
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    index = {
        "architecture": "warehouse-mvp-v1",
        "checkpoint_hash": "fixture",
        "manifest_hash": fingerprint(manifest),
        "visual_dim": 12,
        "kv_dim": 8,
        "map_mode": "correct",
        "config": cfg.to_dict(),
        "records": caches,
    }
    (root / "cache" / "index.json").write_text(json.dumps(index))
    return cfg
