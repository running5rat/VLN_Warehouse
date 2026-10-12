import json
from collections import deque
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image

from DynaNav.warehouse_loop.live_metrics import complete_lines, rate_report
from DynaNav.warehouse_loop.live import aggregate
from DynaNav.warehouse_loop.live_recording import AsyncEpisodeRecorder
from DynaNav.warehouse_loop.live_mapping import enqueue_source


class LiveStreamingTests(unittest.TestCase):
    def test_latest_source_keeps_backlog_bounded_and_ocr_preserves_all(self):
        pending, skipped = deque(), []
        for i in range(100):
            skipped.extend(enqueue_source(pending, {"image_id": i}, latest_only=True))
        self.assertEqual(list(pending), [{"image_id": 99}])
        self.assertEqual(len(skipped), 99)
        self.assertEqual([r["image_id"] for r in skipped], list(range(99)))
        pending.clear()
        for i in range(10):
            self.assertEqual(enqueue_source(pending, {"image_id": i}), [])
        self.assertEqual(len(pending), 10)

    def test_async_writer_copies_buffers_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = AsyncEpisodeRecorder(
                root / "raw", dict(layout_family="test", episode_id="test", task={})
            )
            rgb = np.zeros((8, 8, 3), np.uint8)
            for i in range(3):
                rgb.fill(i)
                writer.append(
                    rgb=rgb,
                    depth=np.ones((8, 8)),
                    K=np.eye(3),
                    T_body_camera=np.eye(4),
                    T_odom_body=np.eye(4),
                    t_obs=float(i),
                    odom_epoch="epoch",
                )
            rgb.fill(99)
            writer.close()
            rows = [
                json.loads(line)
                for line in (root / "raw/frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["t_obs"] for r in rows], [0.0, 1.0, 2.0])
            for i, r in enumerate(rows):
                self.assertTrue(
                    (np.array(Image.open(root / "raw" / r["rgb"])) == i).all()
                )
            self.assertEqual(len(writer.metrics), 3)

    def test_async_writer_failure_reaches_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = AsyncEpisodeRecorder(
                Path(directory) / "raw",
                dict(layout_family="test", episode_id="test", task={}),
            )
            writer.append(
                rgb=np.zeros((8, 8, 3), np.uint8),
                depth=np.ones((4, 4)),
                K=np.eye(3),
                T_body_camera=np.eye(4),
                T_odom_body=np.eye(4),
                t_obs=0.0,
                odom_epoch="epoch",
            )
            with self.assertRaisesRegex(RuntimeError, "disk writer failed"):
                writer.close()

    def test_partial_commit_is_retried_without_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            path.write_text('{"id":1}\n{"id":')
            with path.open() as reader:
                self.assertEqual(list(complete_lines(reader)), [{"id": 1}])
                self.assertEqual(list(complete_lines(reader)), [])
                with path.open("a") as writer:
                    writer.write("2}\n")
                self.assertEqual(list(complete_lines(reader)), [{"id": 2}])
                self.assertEqual(list(complete_lines(reader)), [])

    def test_simulation_rate_cannot_pass_wall_clock_target(self):
        rows = [dict(wall=i * 0.4, t_obs=i * 0.2, service_s=0.01) for i in range(30)]
        result = rate_report(rows, 5, 0, 12)
        self.assertAlmostEqual(result["completion_wall_hz"], 2.5)
        self.assertFalse(result["meets_target_95pct"])

    def test_drain_burst_does_not_pass_live_throughput(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "capture").mkdir()
            (root / "mapping").mkdir()
            capture = dict(loop_complete=True, distance_m=50, frames=50)
            sim = dict(
                run_wall_start=0,
                run_wall_end=10,
                real_time_factor=1,
                wall_seconds=10,
                simulation_elapsed_seconds=10,
                capture_source_hz=5,
            )
            for k in ("physics", "viewport_render", "capture", "map_panel"):
                sim[k] = {"meets_target_95pct": True}
            mapping = dict(
                geometry={"count": 50},
                semantic={"count": 10},
                semantic_backlog_max=10,
                max_semantic_source_lag_s=9,
                capture_to_geometry_p95_s=0.1,
                yolo_calls=1,
                store={},
            )
            for path, data in [
                ("capture/capture-report.json", capture),
                ("capture/performance-report.json", sim),
                ("mapping/report.json", mapping),
            ]:
                (root / path).write_text(json.dumps(data))
            rows = [
                dict(module="semantic", wall=11 + i * 0.01, service_s=0.01)
                for i in range(10)
            ]
            (root / "mapping/metrics.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            report = aggregate(root)
            self.assertFalse(report["all_requested_rates_met"])
            self.assertEqual(report["modules"]["semantic"]["wall_hz_during_motion"], 0)
            mapping.update(
                semantic_backend="llm",
                semantic_frames_dropped=40,
                semantic_sample_hz=1,
                llm_valid_frames=10,
                accepted_instances=0,
                source_projection_consistent=True,
                weight_mode="internvl_base_control",
            )
            (root / "mapping/report.json").write_text(json.dumps(mapping))
            self.assertFalse(aggregate(root)["pipeline_pass"])
            mapping["accepted_instances"] = 1
            (root / "mapping/report.json").write_text(json.dumps(mapping))
            self.assertTrue(aggregate(root)["pipeline_pass"])
            mapping["source_projection_consistent"] = False
            (root / "mapping/report.json").write_text(json.dumps(mapping))
            self.assertFalse(aggregate(root)["pipeline_pass"])


if __name__ == "__main__":
    unittest.main()
