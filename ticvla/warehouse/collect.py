"""Synchronized RGB-D episode recording, with an optional Isaac Replicator adapter.

This module imports no Isaac packages until attach_isaac_recorder is called.
Writer callback time comes from render reference_time, not the latest wall clock.
"""

import json
import time
from pathlib import Path

import numpy as np


class EpisodeRecorder:
    def __init__(self, output, metadata):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "episode.json").exists():
            raise FileExistsError("Refusing to overwrite an episode")
        for key in ("layout_family", "episode_id", "task"):
            if key not in metadata:
                raise ValueError(f"Episode metadata needs {key}")
        self.metadata = {**metadata, "schema_version": 1, "frames_file": "frames.jsonl"}
        self.metadata.pop("frames", None)
        (self.output / "episode.json").write_text(json.dumps(self.metadata, indent=2))
        for folder in ("rgb", "depth", "supervision"):
            (self.output / folder).mkdir()
        self.stream = (self.output / "frames.jsonl").open("x")
        self.count = 0
        self.last_t = -np.inf
        self.previous = None

    def append(
        self,
        *,
        rgb,
        depth,
        K,
        T_body_camera,
        T_odom_body,
        t_obs,
        odom_epoch,
        privileged=None,
        source_monotonic=None,
    ):
        from PIL import Image

        if not np.isfinite(t_obs) or t_obs <= self.last_t:
            raise ValueError(
                "Duplicate/backwards capture timestamp; start a new episode after reset"
            )
        if np.asarray(rgb).shape[:2] != np.asarray(depth).shape:
            raise ValueError("RGB and depth must be aligned to the same camera")
        pose = np.asarray(T_odom_body, dtype=float)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("Invalid body pose")
        key = f"{self.count:08d}"
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)[..., :3]).save(
            self.output / "rgb" / (key + ".png"),
            compress_level=int(self.metadata.get("png_compress_level", 6)),
        )
        np.save(
            self.output / "depth" / (key + ".npy"),
            np.asarray(depth, dtype=np.float32),
            allow_pickle=False,
        )
        record = {
            "t_obs": float(t_obs),
            "rgb": f"rgb/{key}.png",
            "depth": f"depth/{key}.npy",
            "depth_convention": "optical_z_metres",
            "K": np.asarray(K).tolist(),
            "T_body_camera": np.asarray(T_body_camera).tolist(),
            "T_odom_body": pose.tolist(),
            "odom_epoch": odom_epoch,
        }
        if self.previous is not None:
            delta = np.linalg.inv(self.previous) @ pose
            dt = t_obs - self.last_t
            record["robot_state"] = [
                float(delta[0, 3] / dt),
                float(np.arctan2(delta[1, 0], delta[0, 0]) / dt),
            ]
        if privileged:
            # Deliberately inaccessible to the deployment observation builder.
            np.savez_compressed(
                self.output / "supervision" / (key + ".npz"), **privileged
            )
        # Commit time is separate from sensor/physics time; used by live latency metrics.
        record["wall_time"] = time.time()
        if source_monotonic is not None:
            record["source_monotonic"] = float(source_monotonic)
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()
        self.count += 1
        self.last_t = t_obs
        self.previous = pose

    def close(self):
        self.stream.close()


def camera_calibration(params):
    """Replicator pinhole camera -> optical +Z / image-down metric coordinates."""
    width, height = map(int, params["renderProductResolution"])
    aperture = np.asarray(params["cameraAperture"], float)
    offset = np.asarray(params.get("cameraApertureOffset", [0.0, 0.0]), float)
    focal = float(params["cameraFocalLength"])
    K = np.array(
        [
            [focal / aperture[0] * width, 0.0, width * (0.5 - offset[0] / aperture[0])],
            [
                0.0,
                focal / aperture[1] * height,
                height * (0.5 + offset[1] / aperture[1]),
            ],
            [0.0, 0.0, 1.0],
        ]
    )
    # USD matrices multiply row vectors; convert before inversion.
    view = np.asarray(params["cameraViewTransform"], float).reshape(4, 4).T
    T_world_optical = np.linalg.inv(view) @ np.diag([1.0, -1.0, -1.0, 1.0])
    return K, T_world_optical


def attach_isaac_recorder(metadata_path, output):
    import omni.replicator.core as rep
    import omni.usd
    from pxr import Usd, UsdGeom

    metadata = json.loads(Path(metadata_path).read_text())
    stage = omni.usd.get_context().get_stage()
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1.0e-6:
        raise ValueError("Recorder requires a metre-based USD stage")
    camera_path, body_path = metadata["camera_prim"], metadata["body_prim"]
    camera = stage.GetPrimAtPath(camera_path)
    body = stage.GetPrimAtPath(body_path)
    if not camera.IsValid() or not body.IsValid():
        raise ValueError("Recorder camera/body prim does not exist")
    usd_camera = UsdGeom.Camera(camera)
    if usd_camera.GetProjectionAttr().Get() != "perspective":
        raise ValueError("MVP recorder supports pinhole perspective cameras")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    T_world_body = np.array(transforms.GetLocalToWorldTransform(body)).T
    T_world_camera = np.array(transforms.GetLocalToWorldTransform(camera)).T @ np.diag(
        [1.0, -1.0, -1.0, 1.0]
    )
    T_body_camera = np.linalg.inv(T_world_body) @ T_world_camera
    metadata["pose_source"] = "simulated_odometry_from_synchronized_camera_pose"
    recorder = EpisodeRecorder(output, metadata)
    resolution = tuple(metadata.get("resolution", [640, 480]))
    render_product = rep.create.render_product(camera_path, resolution)

    class WarehouseWriter(rep.Writer):
        def __init__(self):
            self.annotators = [
                rep.AnnotatorRegistry.get_annotator(name)
                for name in ("rgb", "distance_to_image_plane", "camera_params")
            ]
            if metadata.get("capture_oracle", False):
                self.annotators.append(
                    rep.AnnotatorRegistry.get_annotator(
                        "instance_segmentation", init_params={"colorize": False}
                    )
                )
            self.origin = None
            self.min_dt = 1.0 / float(metadata.get("capture_hz", 10.0))

        def write_metadata(self):
            # EpisodeRecorder already persists task, calibration conventions and source.
            pass

        def write(self, data):
            # reference_time is provided to Writer.write for the rendered frame.
            stamp = data["reference_time"]
            if isinstance(stamp, dict):
                numerator, denominator = (
                    stamp["referenceTimeNumerator"],
                    stamp["referenceTimeDenominator"],
                )
            else:
                numerator, denominator = stamp
            t = float(numerator) / float(denominator)
            if t < recorder.last_t:
                raise ValueError("Simulation reset during capture")
            if t - recorder.last_t < self.min_dt - 1.0e-6:
                return

            def get(name):
                keys = [k for k in data if k == name or k.startswith(name + "-")]
                if len(keys) != 1:
                    raise ValueError(f"Expected exactly one synchronized {name}")
                return data[keys[0]]

            K, T_camera = camera_calibration(get("camera_params"))
            T_body = T_camera @ np.linalg.inv(T_body_camera)
            if self.origin is None:
                self.origin = np.linalg.inv(T_body)
            privileged = None
            if metadata.get("capture_oracle", False):
                instances = get("instance_segmentation")
                privileged = {
                    "instance_ids": instances["data"],
                    "instance_labels_json": np.frombuffer(
                        json.dumps(instances["info"]["idToLabels"]).encode(),
                        dtype=np.uint8,
                    ),
                }
            recorder.append(
                rgb=get("rgb"),
                depth=get("distance_to_image_plane"),
                K=K,
                T_body_camera=T_body_camera,
                T_odom_body=self.origin @ T_body,
                t_obs=t,
                odom_epoch=metadata["episode_id"],
                privileged=privileged,
            )

    writer = WarehouseWriter()
    writer.attach([render_product])
    return writer, recorder, render_product
