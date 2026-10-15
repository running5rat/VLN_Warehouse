"""Inventory all local sources without assuming that a directory is a dataset.

CSV poses without audited frame/clock metadata are not robot action labels.
"""

import csv
import io
from pathlib import Path
from collections import Counter

import numpy as np

from ..schema import file_sha256
from .core import read_json, write_json


def csv_rows(path):
    raw = Path(path).read_text()
    # Some distributed files literally encode CR/LF characters as backslash text.
    escaped = "\\r\\n" in raw and "\n" not in raw
    text = raw.replace("\\r\\n", "\n") if escaped else raw
    rows = list(csv.DictReader(io.StringIO(text)))
    return rows, "literal_crlf" if escaped else "standard"


def audit(root, output):
    root = Path(root).resolve()
    report = dict(sources={}, actual_api_calls=0, training_started=False,
                  selected_conversion_limit_per_source=10)
    readme = root / "README.md"
    report["dataset_card"] = dict(path=str(readme), sha256=file_sha256(readme),
        declared_license="cc-by-4.0" if "license: cc-by-4.0" in readme.read_text() else None) if readme.exists() else None
    for name in ("DynaNav", "SCAND", "GND"):
        folder = root / name
        trajectories = sorted(folder.rglob("trajectory.csv")) if folder.exists() else []
        episodes = []
        for path in trajectories:
            entry = dict(episode_id=path.parent.name, trajectory=str(path),
                         sha256=file_sha256(path), eligible=False, unknown_fields=[
                             "pose_frame", "time_unit", "camera_extrinsics", "task_action_alignment",
                             "license_review", "nominal_reference"])
            try:
                rows, encoding = csv_rows(path)
                times = np.array([float(r["time"]) for r in rows])
                xyz = np.array([[float(r[k]) for k in ("x", "y", "z")] for r in rows])
                entry.update(rows=len(rows), newline_encoding=encoding,
                             raw_time_range=[float(times[0]), float(times[-1])] if len(times) else None,
                             duration_seconds=None, monotonic=bool(len(times)>1 and np.all(np.diff(times)>0)),
                             finite=bool(np.isfinite(xyz).all()),
                             rgb_count=sum(1 for folder in path.parent.rglob("rgb") for p in folder.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")),
                             depth_count=sum(1 for folder in path.parent.rglob("depth") for p in folder.iterdir() if p.is_file()),
                             calibration_directories=[str(p) for p in path.parent.rglob("camera_params")],
                             embodiment_hint="quadruped" if "spot" in path.parent.name.lower() else "requires_audit",
                             status="quarantined_missing_frame_clock_and_task_audit")
            except (ValueError, KeyError, OSError) as exc:
                entry.update(status="invalid_csv", error=str(exc))
            episodes.append(entry)
        files = Counter(p.suffix.lower() for p in folder.rglob("*") if p.is_file()) if folder.exists() else {}
        candidates = [e for e in episodes if e.get("embodiment_hint") != "quadruped"][:10]
        report["sources"][name] = dict(root=str(folder), present=folder.exists(),
                                      status="present" if files else "empty_or_missing",
                                      file_extensions=dict(files), episodes=episodes,
                                      episode_count=len(episodes), eligible_episodes=0,
                                      fixed_candidate_selection=[e["episode_id"] for e in candidates],
                                      license_status="not_audited")
    report["ready_for_training"] = False
    report["reason"] = "Raw availability is not verified navigation supervision; apply a reviewed catalog before conversion"
    write_json(output, report)
    return report


def convert_audited(catalog_path, output):
    """Convert explicitly audited RGB/pose episodes, never fabricate missing depth.

    Catalog entries refer to CSV + ordered image list and audited physical units.
    This is observation/trajectory import; semantic/mode labels remain unlabeled.
    """
    from scipy.spatial.transform import Rotation
    from ..collect import EpisodeRecorder

    catalog = read_json(catalog_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report, counts, groups, raw_splits = [], Counter(), {}, {}
    for e in catalog["episodes"]:
        source = e["source"]
        if counts[source] >= 10:
            report.append(dict(episode_id=e["episode_id"], status="limit_not_selected"))
            continue
        counts[source] += 1  # Failed selected episodes also count; no success-based reselection.
        try:
            if not e.get("audit_approved") or e.get("embodiment") != "wheeled":
                raise ValueError("Reviewed wheeled embodiment required; human/quadruped conversion deferred")
            if e.get("pose_frame") not in ("body_flu", "usd_camera", "optical_camera"):
                raise ValueError("Missing audited pose frame")
            if e.get("time_unit") not in ("seconds", "frame_index") or not e.get("license_review"):
                raise ValueError("Missing clock or license review")
            split = e["split"]
            if split not in ("train", "val", "test", "regression"):
                raise ValueError("Invalid split")
            if e["layout_family"] in groups and groups[e["layout_family"]] != split:
                raise ValueError("Layout leakage")
            groups[e["layout_family"]] = split
            digest = file_sha256(e["trajectory"])
            if digest in raw_splits:
                raise ValueError("Duplicate source trajectory, possibly across splits")
            raw_splits[digest] = split
            rows, encoding = csv_rows(e["trajectory"])
            images = e["images"]
            if len(rows) != len(images):
                raise ValueError("Catalog must explicitly synchronize images and poses")
            hz = float(e.get("source_hz", 0))
            if e["time_unit"] == "frame_index" and hz <= 0:
                raise ValueError("Frame-index clock needs source_hz")
            tf = np.asarray(e.get("T_body_camera", np.eye(4)), float)
            if e["pose_frame"] != "body_flu" and "T_body_camera" not in e:
                raise ValueError("Camera-to-body extrinsics required")
            if tf.shape != (4,4) or not np.isfinite(tf).all() or not np.allclose(tf[3],[0,0,0,1]) or not np.allclose(tf[:3,:3].T@tf[:3,:3],np.eye(3),atol=1e-4):
                raise ValueError("Invalid rigid camera extrinsics")
            frames, prev = [], -float("inf")
            for r, image in zip(rows, images):
                t = float(r["time"]) / (hz if e["time_unit"] == "frame_index" else 1)
                if not np.isfinite(t) or t <= prev or not Path(image).is_file():
                    raise ValueError("Unsynchronized/missing source")
                prev = t
                pose = np.eye(4)
                pose[:3, :3] = Rotation.from_quat([float(r[k]) for k in ("qx", "qy", "qz", "qw")]).as_matrix()
                pose[:3, 3] = [float(r[k]) for k in ("x", "y", "z")]
                if e["pose_frame"] == "usd_camera": pose = pose @ np.diag([1,-1,-1,1])
                if e["pose_frame"] != "body_flu": pose = pose @ np.linalg.inv(tf)
                if not np.isfinite(pose).all(): raise ValueError("Nonfinite pose")
                frames.append(dict(t_obs=t, rgb=str(Path(image).resolve()), T_odom_body=pose.tolist(),
                                   odom_epoch=e["episode_id"]))
            item = dict(schema_version=1, episode_id=e["episode_id"], layout_family=e["layout_family"],
                        task=dict(task_id=e["episode_id"], instruction=e["instruction"]), frames=frames,
                        supervision=dict(actions_task_aligned=e.get("actions_task_aligned", False)),
                        provenance=dict(source=source, split=split, catalog_sha256=file_sha256(catalog_path),
                                        csv_sha256=file_sha256(e["trajectory"]), newline_encoding=encoding,
                                        missing=["depth", "map", "semantic_labels", "mode_labels"]))
            # Never use untrusted external IDs as paths.
            dest = output / f"episode-{len(report):06d}.json"
            write_json(dest, item)
            report.append(dict(episode_id=e["episode_id"], status="observations_imported", path=str(dest),
                               action_gt_ready=False))
        except (ValueError, KeyError, OSError) as exc:
            report.append(dict(episode_id=e["episode_id"], status="rejected", reason=str(exc)))
    write_json(output / "conversion-report.json", report)
    return report
