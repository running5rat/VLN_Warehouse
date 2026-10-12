"""Import audited wheeled DynaNav episodes without inventing RGB-D/map labels.

The catalog supplies episode layout/split, task instruction, CSV pose convention,
and camera extrinsics when the CSV stores a camera pose. Sampling time is explicit.
Overlapping JSON windows are deduplicated by original episode timestamp.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import WarehouseConfig, load_config
from .prepare import prepare_episodes


def import_episodes(json_root, raw_root, catalog, output, cfg, stride=5):
    json_root, raw_root, output = Path(json_root), Path(raw_root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episode_paths = []
    splits = {}
    for episode_id, metadata in catalog.items():
        if "spot" in episode_id.lower():
            raise ValueError(
                "warehouse_mvp is wheeled/forward-only; do not import Spot episodes"
            )
        if metadata.get("pose_frame") not in (
            "body_flu",
            "usd_camera",
            "optical_camera",
        ):
            raise ValueError("Catalog must declare pose_frame explicitly")
        if metadata.get("time_unit") not in ("seconds", "frame_index"):
            raise ValueError("Catalog must declare time_unit: seconds or frame_index")
        rows = np.genfromtxt(
            raw_root / episode_id / "trajectory.csv", delimiter=",", names=True
        )
        factor = (
            1.0 if metadata["time_unit"] == "seconds" else 1.0 / metadata["source_hz"]
        )
        times = np.asarray(rows["time"]) * factor
        times -= times[0]
        poses = []
        for row in rows:
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(
                [row[k] for k in ("qx", "qy", "qz", "qw")]
            ).as_matrix()
            matrix[:3, 3] = [row[k] for k in ("x", "y", "z")]
            if metadata["pose_frame"] != "body_flu":
                if "T_body_camera" not in metadata:
                    raise ValueError(
                        "Camera CSV requires audited optical-camera-to-body extrinsics"
                    )
                if metadata["pose_frame"] == "usd_camera":
                    matrix = matrix @ np.diag([1.0, -1.0, -1.0, 1.0])
                matrix = matrix @ np.linalg.inv(metadata["T_body_camera"])
            poses.append(matrix)
        origin = np.linalg.inv(poses[0])
        poses = [origin @ p for p in poses]
        frames = {}
        for directory in sorted(json_root.glob(episode_id + "_*s_*s")):
            if not re.fullmatch(re.escape(episode_id) + r"_\d+s_\d+s", directory.name):
                continue
            for sample in sorted(directory.glob("*.json")):
                record = json.loads(sample.read_text())
                t = float(record["original_timestamp"])
                if t in frames:
                    continue
                nearest = int(np.argmin(np.abs(times - t)))
                if abs(times[nearest] - t) > max(0.002, factor * 0.1):
                    raise ValueError(
                        "JSON timestamp and audited trajectory clock do not align"
                    )
                rgb = (directory / record["current"]["img"]).resolve()
                if not rgb.is_file():
                    # Published paths sometimes retain an absolute source-machine root.
                    raw = str(record["current"]["img"])
                    marker = "DynaNav_data/"
                    if marker in raw:
                        rgb = raw_root / raw.split(marker, 1)[1]
                if not rgb.is_file():
                    raise FileNotFoundError(rgb)
                frames[t] = {
                    "t_obs": t,
                    "rgb": str(rgb),
                    "T_odom_body": poses[nearest].tolist(),
                    "odom_epoch": episode_id,
                }
        if not frames:
            raise ValueError(f"No JSON windows found for {episode_id}")
        episode = {
            "schema_version": 1,
            "layout_family": metadata["layout_family"],
            "episode_id": episode_id,
            "task": {
                "task_id": episode_id,
                "instruction": metadata["instruction"],
                "hard_constraints": metadata.get("hard_constraints", []),
            },
            "pose_source": "audited_dynanav_csv",
            "frames": [frames[t] for t in sorted(frames)],
            "supervision": {
                "actions_task_aligned": metadata.get("actions_task_aligned", False)
            },
        }
        path = output / (episode_id + ".json")
        path.write_text(json.dumps(episode, indent=2))
        episode_paths.append(path)
        splits[episode_id] = metadata["split"]
    # Published DynaNav repeats layouts; this is explicitly an episode-only split.
    return prepare_episodes(
        episode_paths,
        output / "prepared",
        splits,
        cfg,
        split_unit="episode_id",
        stride=stride,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-root", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else WarehouseConfig()
    print(
        import_episodes(
            args.json_root,
            args.raw_root,
            json.loads(Path(args.catalog).read_text()),
            args.output,
            cfg,
            args.stride,
        )
    )


if __name__ == "__main__":
    main()
