"""Deterministic Isaac 5.0 mapping fixture; GT is written only under supervision.

Run using Isaac Python, --output an unused directory on the shared /data mount.
This is a camera replay fixture, not a robot-control/navigation benchmark.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "width": 640, "height": 480}
    )
    capture = None
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
        from isaacsim.core.utils.semantics import add_update_semantics
        from ticvla.warehouse.collect import attach_isaac_recorder

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

        def box(name, position, size, color, label):
            name = name.replace("-", "neg")
            obj = UsdGeom.Cube.Define(stage, f"/World/{name}")
            obj.AddTranslateOp().Set(Gf.Vec3d(*position))
            obj.AddScaleOp().Set(Gf.Vec3f(*(v / 2 for v in size)))
            obj.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            add_update_semantics(obj.GetPrim(), label)
            return obj

        box("Floor", (0, 0, -0.1), (20, 20, 0.2), (0.45, 0.45, 0.45), "floor")
        box("BoxA", (3.0, 1.0, 0.55), (0.9, 0.85, 1.1), (0.65, 0.36, 0.12), "box_a")
        box("BoxB", (4.5, -0.7, 0.8), (0.9, 0.9, 1.6), (0.12, 0.3, 0.65), "box_b")
        box("BoxC", (6.0, 0.15, 0.4), (1.1, 0.8, 0.8), (0.65, 0.12, 0.12), "box_c")
        # Unselected obstacles still have geometry; shelf frame is not a semantic prompt.
        for y in (-2.5, 2.5):
            for x in (3.0, 6.5):
                box(
                    f"Post_{str(x).replace('.', '_')}_{str(y).replace('.', '_')}",
                    (x, y, 1.3),
                    (0.15, 0.15, 2.6),
                    (0.12, 0.2, 0.55),
                    "shelf",
                )
            for z in (0.2, 1.4, 2.6):
                box(
                    f"Shelf_{str(y).replace('.', '_')}_{str(z).replace('.', '_')}",
                    (4.75, y, z),
                    (3.7, 0.7, 0.12),
                    (0.65, 0.35, 0.1),
                    "shelf",
                )
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1300.0)
        body = UsdGeom.Xform.Define(stage, "/World/Body")
        move = body.AddTranslateOp()
        camera = UsdGeom.Camera.Define(stage, "/World/Body/Camera")
        camera.CreateFocalLengthAttr(16.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(18.0)
        camera.AddTransformOp().Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(0, 0, 1.4), Gf.Vec3d(4, 0, 0.6), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        metadata = {
            "layout_family": "mapping-fixture-a",
            "episode_id": "mapping-a",
            "task": {
                "task_id": "map-task-a",
                "instruction": "Observe the boxes and shelves in the aisle.",
                "hard_constraints": ["Use observed geometry only."],
            },
            "camera_prim": "/World/Body/Camera",
            "body_prim": "/World/Body",
            "resolution": [640, 480],
            "capture_hz": 10,
            "capture_oracle": True,
            "supervision": {"actions_task_aligned": False},
        }
        config = root / "capture.json"
        config.write_text(json.dumps(metadata, indent=2))
        for _ in range(5):
            app.update()
        capture = attach_isaac_recorder(config, root / "raw")
        for i in range(13):
            move.Set(Gf.Vec3d(i * 0.025, 0, 0))
            rep.orchestrator.step(rt_subframes=4, delta_time=0.5, pause_timeline=False)
        rep.orchestrator.wait_until_complete()
        capture[0].detach()
        capture[1].close()
        capture = None
        frames = [
            json.loads(x) for x in (root / "raw/frames.jsonl").read_text().splitlines()
        ]
        assert len(frames) == 13
        assert all(b["t_obs"] > a["t_obs"] for a, b in zip(frames, frames[1:]))
        report = {
            "isaac_version": Path("/data/isaacsim/VERSION").read_text().strip(),
            "frames": len(frames),
            "fixture": "camera replay; no control or physics claim",
            "episode": str(root / "raw/episode.json"),
            "capture_pass": True,
        }
        import hashlib

        report["fixture_script_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        report["seed"] = 0
        report["source_duration_s"] = frames[-1]["t_obs"] - frames[0]["t_obs"]
        (root / "capture-report.json").write_text(json.dumps(report, indent=2))
        print("MAPPING_CAPTURE", json.dumps(report), flush=True)
    finally:
        if capture:
            capture[0].detach()
            capture[1].close()
        app.close()


if __name__ == "__main__":
    main()
