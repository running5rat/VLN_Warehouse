import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ticvla.warehouse.collect import EpisodeRecorder, camera_calibration
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.geometry import (
    LocalMapBuilder,
    align_reference,
    astar_reference,
    collision_free,
    planar_pose,
    project_depth,
)
from ticvla.warehouse.schema import (
    SemanticSnapshot,
    TaskContract,
    TaskState,
    load_manifest,
)


class GeometryTests(unittest.TestCase):
    def test_optical_depth_projection_uses_extrinsics(self):
        transform = np.array(
            [[0, 0, 1, 1], [-1, 0, 0, 2], [0, -1, 0, 0.5], [0, 0, 0, 1.0]]
        )
        points = project_depth(
            np.array([[2.0, np.nan], [0.0, 4.0]]), np.eye(3), transform
        )
        np.testing.assert_allclose(points, [[3, 2, 0.5], [5, -2, -3.5]])

    def test_alignment_rotation_and_prefix(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        aligned, mask = align_reference(
            points,
            np.ones(4, bool),
            planar_pose(),
            planar_pose(1.5, 0, np.pi / 2),
            "a",
            "a",
        )
        np.testing.assert_allclose(
            aligned[mask], [[0, 0], [0, -0.5], [0, -1.5]], atol=1.0e-6
        )
        with self.assertRaises(ValueError):
            align_reference(
                points, np.ones(4, bool), planar_pose(), planar_pose(), "a", "b"
            )
        _empty, valid = align_reference(
            points, np.zeros(4, bool), planar_pose(), planar_pose(), "a", "a"
        )
        self.assertFalse(valid.any())

    def test_astar_unknown_and_footprint(self):
        grid = np.zeros((40, 40), np.int8)
        grid[12:17, 17:23] = 1
        path = astar_reference(grid, [2, 0], 0.1, 0.1)
        # Goal outside grid is rejected.
        self.assertEqual(len(path), 0)
        path = astar_reference(grid, [1.7, 0], 0.1, 0.1)
        self.assertGreater(len(path), 1)
        self.assertTrue(collision_free(path, grid, 0.1, 0.1))
        self.assertFalse(collision_free([[0, 0], [1.7, 0]], grid, 0.1, 0.1))
        grid[0:20] = -1
        self.assertEqual(len(astar_reference(grid, [1.7, 0], 0.1, 0.1)), 0)

    def test_map_render_never_reads_future(self):
        cfg = WarehouseConfig(depth_stride=1, map_resolution=0.5, map_size_m=8)
        builder = LocalMapBuilder(cfg)
        extrinsic = np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1.0]]
        )
        builder.add(
            depth=np.array([[2.0]]),
            K=np.eye(3),
            T_body_camera=extrinsic,
            T_odom_body=planar_pose(),
            t_obs=2.0,
            odom_epoch="a",
        )
        old, _ = builder.render(planar_pose(), 1.0, "a")
        self.assertTrue((old == -1).all())
        current, _ = builder.render(planar_pose(), 2.0, "a")
        self.assertTrue((current == 1).any())

    def test_snapshot_and_contract(self):
        state = TaskState(
            TaskContract("task", "turn left", ("keep goal",)),
            "turn left",
            execution={"origin": [0, 0]},
        )
        state.commit_route("continue", "task", 0)
        self.assertEqual(state.contract.instruction, "turn left")
        self.assertEqual(state.execution["origin"], [0, 0])
        with self.assertRaises(ValueError):
            state.commit_route("again", "task", 0)
        snapshot = SemanticSnapshot(
            None, [], [], 0.8, 0.0, 0.1, 0.2, planar_pose(), "a", 1, "task", 0, 1, "map"
        )
        snapshot.validate(
            t_use=1, task_id="task", odom_epoch="a", route_revision=1, max_age=2
        )
        with self.assertRaises(ValueError):
            snapshot.validate(
                t_use=3, task_id="task", odom_epoch="a", route_revision=1, max_age=2
            )

    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            records = [
                {
                    "sample_id": str(i),
                    "layout_family": "one",
                    "episode_id": str(i),
                    "split": s,
                }
                for i, s in enumerate(["train", "test"])
            ]
            path.write_text(json.dumps({"schema_version": 1, "records": records}))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_recorder_and_camera_calibration(self):
        params = {
            "renderProductResolution": [640, 480],
            "cameraAperture": [20.0, 15.0],
            "cameraFocalLength": 10.0,
            "cameraViewTransform": np.eye(4),
        }
        K, T = camera_calibration(params)
        np.testing.assert_allclose(K, [[320, 0, 320], [0, 320, 240], [0, 0, 1]])
        np.testing.assert_allclose(T, np.diag([1, -1, -1, 1]))
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(
                tmp,
                {
                    "layout_family": "a",
                    "episode_id": "e",
                    "task": {"instruction": "forward"},
                },
            )
            kwargs = {
                "rgb": np.zeros((2, 2, 3), np.uint8),
                "depth": np.ones((2, 2)),
                "K": np.eye(3),
                "T_body_camera": np.eye(4),
                "T_odom_body": np.eye(4),
                "t_obs": 0.1,
                "odom_epoch": "e",
            }
            recorder.append(**kwargs, privileged={"instances": np.ones((2, 2))})
            with self.assertRaises(ValueError):
                recorder.append(**kwargs)
            recorder.close()
            frame = json.loads((Path(tmp) / "frames.jsonl").read_text())
            self.assertNotIn("instances", frame)


if __name__ == "__main__":
    unittest.main()
