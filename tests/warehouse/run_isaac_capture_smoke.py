"""Explicit headless recorder check on a small local USD scene (no asset download)."""

import itertools
import json
import tempfile
from pathlib import Path


def main():
    from isaacsim import SimulationApp

    app = SimulationApp(
        {"headless": True, "renderer": "RayTracedLighting", "width": 320, "height": 240}
    )
    capture = None
    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        from ticvla.warehouse.collect import attach_isaac_recorder
        from ticvla.warehouse.config import WarehouseConfig
        from ticvla.warehouse.prepare import prepare_episodes

        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        floor = UsdGeom.Cube.Define(stage, "/World/Floor")
        floor.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.1))
        floor.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.1))
        box = UsdGeom.Cube.Define(stage, "/World/Box")
        box.AddTranslateOp().Set(Gf.Vec3d(3, 1, 0.5))
        box.AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.5))
        light = UsdLux.DomeLight.Define(stage, "/World/Light")
        light.CreateIntensityAttr(1000.0)
        body = UsdGeom.Xform.Define(stage, "/World/Body")
        translate = body.AddTranslateOp()
        camera = UsdGeom.Camera.Define(stage, "/World/Body/Camera")
        camera.CreateFocalLengthAttr(12.0)
        camera.CreateHorizontalApertureAttr(24.0)
        camera.CreateVerticalApertureAttr(18.0)
        transform = (
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(0, 0, 1), Gf.Vec3d(3, 0, 0), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        camera.AddTransformOp().Set(transform)
        root = Path(tempfile.mkdtemp(prefix="warehouse-isaac-smoke-"))
        metadata = {
            "layout_family": "smoke-layout",
            "episode_id": "smoke-episode",
            "task": {"task_id": "smoke-task", "instruction": "Move straight forward."},
            "camera_prim": "/World/Body/Camera",
            "body_prim": "/World/Body",
            "resolution": [320, 240],
            "capture_hz": 10,
            "supervision": {
                "local_goal_odom": [2.0, 0.0, 0.0],
                "actions_task_aligned": False,
            },
        }
        metadata_path = root / "capture.json"
        metadata_path.write_text(json.dumps(metadata))
        for _ in range(5):
            app.update()
        capture = attach_isaac_recorder(metadata_path, root / "raw")
        for i in range(8):
            translate.Set(Gf.Vec3d(i * 0.02, 0, 0))
            rep.orchestrator.step(rt_subframes=2, delta_time=0.1, pause_timeline=False)
        rep.orchestrator.wait_until_complete()
        capture[0].detach()
        capture[1].close()
        capture = None
        frames = [
            json.loads(x)
            for x in (root / "raw" / "frames.jsonl").read_text().splitlines()
        ]
        assert len(frames) >= 2, f"Only {len(frames)} frames captured"
        assert all(b["t_obs"] > a["t_obs"] for a, b in itertools.pairwise(frames))
        cfg = WarehouseConfig(depth_stride=8)
        manifest = prepare_episodes(
            [root / "raw" / "episode.json"],
            root / "prepared",
            {"smoke-layout": "train"},
            cfg,
            stride=2,
        )
        entries = json.loads(manifest.read_text())["records"]
        labeled = [
            json.loads((manifest.parent / e["path"]).read_text())["supervision"]
            for e in entries
        ]
        assert any(any(s["reference_mask"]) for s in labeled), (
            "Observed passage produced no valid reference"
        )
        print(
            "ISAAC_SMOKE",
            json.dumps(
                {"root": str(root), "frames": len(frames), "manifest": str(manifest)}
            ),
            flush=True,
        )
    finally:
        if capture:
            capture[0].detach()
            capture[1].close()
        app.close()


if __name__ == "__main__":
    main()
