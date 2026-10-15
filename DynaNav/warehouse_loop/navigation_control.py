"""Isaac-side command watchdog and current-pose swept-footprint check."""

import json
from pathlib import Path
import time
import numpy as np
from ticvla.warehouse.geometry import collision_free, transform_points


class NavigationControl:
    def __init__(self, root):
        self.root = Path(root)
        self.last = (0.0, 0.0)
        self.last_t = time.monotonic()
        self.cached_safety = None
        self.stamp = None
        self.last_observed_pose = None
        self.last_observed_time = None
        self.last_observed_epoch = None
        self.stream = (self.root / "executed-commands.jsonl").open("w", buffering=1)

    def command(self, pose, epoch, t_obs=None):
        now = time.monotonic()
        v = w = 0.0
        reason = "no_command"
        source = None
        backend = None
        try:
            cmd = json.loads((self.root / "command.json").read_text())
            source = cmd.get("source_image_id")
            backend = cmd.get("backend")
            if cmd.get("odom_epoch", epoch) != epoch:
                reason = "epoch_mismatch"
            elif now > cmd["valid_until"] or now < cmd["created_monotonic"]:
                reason = "command_expired"
            elif (self.root.parent / "cancel.json").exists():
                reason = "cancelled"
            elif (
                backend == "reference-only"
                or cmd.get("reason") == "execution_not_evaluated"
            ):
                reason = "reference_only"
            else:
                v = float(np.clip(cmd["v"], 0, 0.3))
                w = float(np.clip(cmd["w"], -0.6, 0.6))
                reason = cmd["reason"]
                if not np.isfinite([v, w]).all():
                    raise ValueError("nonfinite command")
                stamp = (self.root / "safety.npz").stat().st_mtime_ns
                if stamp != self.stamp:
                    with np.load(self.root / "safety.npz") as data:
                        self.cached_safety = {k: data[k].copy() for k in data.files}
                    self.stamp = stamp
                safe = self.cached_safety
                if (
                    str(safe["odom_epoch"]) != epoch
                    or now - float(safe["source_monotonic"]) > 0.6
                ):
                    v = w = 0.0
                    reason = "sensor_expired"
                else:
                    # Acceleration limiting then test the actual command arc in the source grid.
                    dt = min(0.1, max(0, now - self.last_t))
                    v = float(
                        np.clip(v, self.last[0] - 0.8 * dt, self.last[0] + 0.8 * dt)
                    )
                    w = float(
                        np.clip(w, self.last[1] - 1.5 * dt, self.last[1] + 1.5 * dt)
                    )
                    times = np.linspace(0, 0.35 + abs(v) / 0.8, 20)
                    xy = (
                        np.stack((v * times, np.zeros_like(times)), -1)
                        if abs(w) < 1e-5
                        else np.stack(
                            (
                                v * np.sin(w * times) / w,
                                v * (1 - np.cos(w * times)) / w,
                            ),
                            -1,
                        )
                    )
                    tf = np.linalg.inv(safe["pose"]) @ pose
                    xy = transform_points(np.c_[xy, np.zeros(len(xy))], tf)[:, :2]
                    # A human in the simulator must be able to explore beyond
                    # the current camera's free-space rays. Only the explicit
                    # annotation backend permits unknown cells; occupied cells,
                    # map bounds, stale sensors and watchdogs still stop it.
                    manual_exploration = backend == "manual" and cmd.get("safety_policy") == "human_exploration"
                    check_grid = np.where(safe["grid"] == -1, 0, safe["grid"]) if manual_exploration else safe["grid"]
                    if not collision_free(
                        xy, check_grid, float(safe.get("resolution", 0.1)), 0.35
                    ):
                        v = w = 0.0
                        reason = "shield_collision" if manual_exploration else "shield_collision_or_unknown"
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
            OSError,
        ) as exc:
            v = w = 0.0
            reason = "unavailable:" + str(exc)
        self.last = (v, w)
        self.last_t = now
        measured = None
        if (t_obs is not None and self.last_observed_time is not None
                and t_obs > self.last_observed_time and self.last_observed_epoch == epoch):
            delta = np.linalg.inv(self.last_observed_pose) @ pose
            elapsed = t_obs - self.last_observed_time
            measured = [float(delta[0, 3] / elapsed),
                        float(np.arctan2(delta[1, 0], delta[0, 0]) / elapsed)]
        self.last_observed_pose = pose.copy()
        self.last_observed_time = t_obs
        self.last_observed_epoch = epoch
        self.stream.write(
            json.dumps(
                dict(
                    monotonic=now,
                    t_obs=t_obs,
                    odom_epoch=epoch,
                    measured_v_w=measured,
                    v=v,
                    w=w,
                    reason=reason,
                    source_image_id=source,
                    backend=backend,
                    pose=pose.tolist(),
                )
            )
            + "\n"
        )
        return v, w

    def close(self):
        self.stream.close()


class Pedestrian:
    """A human USD translated at 0.2 m/s; no pose/ID is sent to the actor."""

    def __init__(self, stage, root, asset):
        from pxr import UsdGeom, Gf

        self.UsdGeom, self.Gf = UsdGeom, Gf
        self.root = Path(root)
        self.prim = stage.DefinePrim("/World/NavigationPerson", "Xform")
        self.prim.GetReferences().AddReference(asset)
        x = UsdGeom.Xformable(self.prim)
        x.ClearXformOpOrder()
        self.move = x.AddTranslateOp()
        self.rot = x.AddRotateZOp()
        self.rot.Set(90.0)
        self.move.Set(Gf.Vec3d(-13, 9, 0))
        self.started = None
        self.imageable = UsdGeom.Imageable(self.prim)

    def update(self, sim_time):
        enabled = (self.root / "follow-ready.json").exists()
        if enabled and self.started is None:
            self.started = sim_time
        elapsed = 0 if self.started is None else sim_time - self.started
        # Deliberate loss case after initial motion; scene intervention only, never a target observation.
        self.imageable.GetVisibilityAttr().Set(
            "invisible" if 15 <= elapsed < 18 else "inherited"
        )
        self.move.Set(self.Gf.Vec3d(-13, 9 + 0.2 * min(elapsed, 40), 0))
