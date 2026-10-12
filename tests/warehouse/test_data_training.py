import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from fixtures import make_fixture

from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.data import (
    CachedActionDataset,
    RecordDataset,
    action_collate,
    previous_plan_tokens,
)
from ticvla.warehouse.evaluate import evaluate_action
from ticvla.warehouse.geometry import planar_pose
from ticvla.warehouse.import_dynanav import import_episodes
from ticvla.warehouse.prepare import (
    causal_state,
    prepare_episodes,
    select_history,
    time_actions,
)
from ticvla.warehouse.train import load_action, train


class DataTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_causal_state_and_history_ignore_future(self):
        frames = [
            {
                "t_obs": i * 0.1,
                "rgb": "image.png",
                "T_odom_body": planar_pose(i * 0.01, 0, i * 0.001).tolist(),
                "odom_epoch": "a",
            }
            for i in range(5)
        ]
        state = causal_state(frames, 2)
        history = select_history(frames, 2, [0.2, 0.1, 0])
        changed = copy.deepcopy(frames)
        changed[3]["T_odom_body"] = planar_pose(999, 999).tolist()
        self.assertEqual(state, causal_state(changed, 2))
        self.assertEqual(history, select_history(changed, 2, [0.2, 0.1, 0]))
        cfg = WarehouseConfig(horizon=8)
        _, mask = time_actions(frames, 2, cfg)
        self.assertEqual(int(mask.sum()), 2)

    def test_raw_conversion_masks_missing_map_and_task_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_fixture(root)
            episode = {
                "layout_family": "raw",
                "episode_id": "raw",
                "task": {"instruction": "forward"},
                "frames": [
                    {
                        "t_obs": i * 0.1,
                        "rgb": "rgb.png",
                        "T_odom_body": planar_pose(i * 0.01).tolist(),
                        "odom_epoch": "raw",
                    }
                    for i in range(4)
                ],
                "supervision": {
                    "local_goal_odom": [1, 0, 0],
                    "actions_task_aligned": False,
                },
            }
            path = root / "episode.json"
            path.write_text(json.dumps(episode))
            manifest = prepare_episodes(
                [path], root / "prepared", {"raw": "train"}, cfg, stride=1
            )
            data = RecordDataset(manifest, "train")
            self.assertFalse(data[0]["observation"]["map_valid"])
            self.assertFalse(data[0]["supervision"]["reference_known"])
            self.assertFalse(any(data[0]["supervision"]["action_mask"]))

    def test_cached_delay_padding_and_expiry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_fixture(tmp)
            dataset = CachedActionDataset(cfg, "train", delay=1.0)
            first, last = dataset[0], dataset[2]
            self.assertFalse(first["snapshot_valid"])
            self.assertTrue(last["snapshot_valid"])
            self.assertEqual(last["age"], 1.0)
            self.assertAlmostEqual(float(last["delay"][0]), 1.0)
            batch = action_collate([first, last])
            self.assertEqual(batch["kv"].shape, (2, 6, 8))
            self.assertFalse(bool(batch["kv_mask"][0, -1]))
            cfg.max_snapshot_age = 0.5
            expired = CachedActionDataset(cfg, "train", delay=1.0)[2]
            self.assertFalse(expired["semantic_valid"])
            self.assertFalse(expired["reference_mask"].any())

    def test_labels_do_not_change_deployment_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_fixture(tmp)
            data = CachedActionDataset(cfg, "train", delay=0.0)
            before = data[1]
            path = Path(tmp) / "train-1.json"
            record = json.loads(path.read_text())
            record["supervision"]["actions"] = [[999.0, 999.0]] * cfg.horizon
            record["supervision"]["reference_xy"] = [
                [999.0, 999.0]
            ] * cfg.num_reference_points
            path.write_text(json.dumps(record))
            after = data[1]
            for key in (
                "visual",
                "kv",
                "bev",
                "state",
                "delay",
                "reference_xy",
                "reference_mask",
            ):
                torch.testing.assert_close(before[key], after[key])

    def test_stale_cache_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_fixture(tmp)
            path = Path(tmp) / "train-0.json"
            record = json.loads(path.read_text())
            record["observation"]["route"] = "changed"
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Stale observation"):
                CachedActionDataset(cfg, "train")

    def test_previous_plan_requires_causal_policy_provenance(self):
        cfg = WarehouseConfig(horizon=3)
        obs = {
            "t_obs": 1.0,
            "odom_epoch": "e",
            "T_odom_body": planar_pose(1.0).tolist(),
            "previous_plan": {
                "source": "policy_proposal",
                "t_obs": 0.0,
                "odom_epoch": "e",
                "T_odom_body": planar_pose().tolist(),
                "xy": [[0.2, 0.0], [2.0, 0.0]],
                "time_offsets": [0.2, 1.2],
            },
        }
        points, valid = previous_plan_tokens(obs, cfg)
        self.assertEqual(int(valid.sum()), 1)
        np.testing.assert_allclose(
            points[valid].numpy(), [[1.0, 0.0, 0.2]], atol=1.0e-6
        )
        obs["previous_plan"]["source"] = "expert_future"
        with self.assertRaises(ValueError):
            previous_plan_tokens(obs, cfg)

    def test_flow_training_checkpoint_and_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_fixture(tmp)
            checkpoint = train(cfg, "flow")
            model, metadata = load_action(checkpoint, "cpu")
            self.assertEqual(metadata["step"], 2)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            report = evaluate_action(cfg, checkpoint, Path(tmp) / "eval", limit=1)
            self.assertEqual(report["samples"], 1)
            self.assertTrue(np.isfinite(report["ade"]))
            self.assertTrue((Path(tmp) / "eval" / "report.json").exists())

    def test_dynanav_import_deduplicates_and_does_not_mix_robot_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = make_fixture(root)
            raw = root / "raw" / "run"
            raw.mkdir(parents=True)
            (raw / "trajectory.csv").write_text(
                "time,x,y,z,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n1,.1,0,0,0,0,0,1\n2,.2,0,0,0,0,0,1\n"
            )
            windows = root / "json"
            for window in ("run_0s_20s", "run_10s_30s", "run_spot_0s_20s"):
                folder = windows / window
                folder.mkdir(parents=True)
                for i in range(3):
                    sample = {
                        "original_timestamp": 999.0 if "spot" in window else i * 0.1,
                        "current": {"img": str(root / "rgb.png")},
                    }
                    (folder / f"rgb_{i}.json").write_text(json.dumps(sample))
            catalog = {
                "run": {
                    "layout_family": "layout",
                    "split": "train",
                    "instruction": "forward",
                    "pose_frame": "body_flu",
                    "time_unit": "frame_index",
                    "source_hz": 10,
                    "actions_task_aligned": True,
                }
            }
            manifest = import_episodes(
                windows, root / "raw", catalog, root / "imported", cfg, stride=1
            )
            records = RecordDataset(manifest, "train")
            self.assertEqual(len(records), 3)
            self.assertFalse(records[0]["observation"]["map_valid"])
            self.assertEqual(sum(records[0]["supervision"]["action_mask"]), 2)


if __name__ == "__main__":
    unittest.main()
