import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from ticvla.warehouse.mapping import MapStore, SemanticEvent, SourceObservation
from ticvla.warehouse.mapping.semantic_frame import (
    consume_categories,
    reprojection_check,
)
from ticvla.warehouse.mapping.store import semantic_points
from ticvla.warehouse.structured_io import parse_structured, response_grammar


class NavigationProtocolTests(unittest.TestCase):
    def response(self, items):
        return (
            "<landmark>"
            + json.dumps(items)
            + "</landmark><local>unknown</local><Route>Continue.</Route><waypoint>[]</waypoint><REFLINE>"
        )

    def test_extended_and_legacy_and_empty(self):
        old = dict(image_id="a:0", category="box")
        new = dict(old, navigation_role="avoidance", reason="May obstruct the aisle.")
        for items in ([], [old], [new]):
            parsed = parse_structured(self.response(items), ["a:0"])
            self.assertEqual(parsed["landmark"], items)
        parser = response_grammar(["a:0"], extended=True)
        compact = self.response([new]).replace(
            json.dumps([new]), json.dumps([new], separators=(",", ":"))
        )
        for char in compact:
            self.assertIn(char, parser.get_allowed_characters())
            parser = parser.add_character(char)
        self.assertTrue(parser.can_end())

    def test_invalid_source_role_and_reason(self):
        base = dict(
            image_id="a:0",
            category="box",
            navigation_role="both",
            reason="Visible object.",
        )
        for update in (
            dict(image_id="map"),
            dict(navigation_role="static"),
            dict(reason="x" * 121),
            dict(reason=""),
        ):
            with self.assertRaises(ValueError):
                parse_structured(self.response([dict(base, **update)]), ["a:0"])
        with self.assertRaises(ValueError):
            parse_structured(self.response([base])[:-4], ["a:0"])

    def test_generation_cannot_cross_reason_length_bound(self):
        item = dict(
            image_id="a:0", category="box", navigation_role="both", reason="a" * 121
        )
        text = self.response([item]).replace(
            json.dumps([item]), json.dumps([item], separators=(",", ":"))
        )
        parser = response_grammar(["a:0"], extended=True)
        rejected = False
        for char in text:
            if char not in parser.get_allowed_characters():
                rejected = True
                break
            parser = parser.add_character(char)
        self.assertTrue(rejected)


class CategoryConsumerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Image.new("RGB", (80, 60), "white").save(self.root / "rgb.png")
        depth = np.full((60, 80), 2.0, dtype=np.float32)
        np.save(self.root / "depth.npy", depth)
        self.store = MapStore(self.root / "map")
        pose = np.eye(4)
        pose[0, 3] = 3
        self.source = SourceObservation(
            "a:0",
            str(self.root / "rgb.png"),
            str(self.root / "depth.npy"),
            [[80, 0, 40], [0, 80, 30], [0, 0, 1]],
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]],
            pose.tolist(),
            1.0,
            "epoch",
            "epoch",
            "task",
        )
        self.store.add_observation(self.source)
        self.masks = []
        for x in (4, 52):
            mask = np.zeros((60, 80), bool)
            mask[20:40, x : x + 20] = True
            self.masks.append(mask)
        self.event = SemanticEvent(
            "request:0",
            "a:0",
            "box",
            source="llm",
            request_id="request",
            navigation_role="avoidance",
            reason="Obstacle.",
        )
        self.backend = Mock()
        self.backend.segment_instances.return_value = dict(
            instances=[
                dict(
                    category="box",
                    confidence=0.9,
                    bbox=[4, 20, 24, 40],
                    mask=self.masks[0],
                ),
                dict(
                    category="box",
                    confidence=0.8,
                    bbox=[52, 20, 72, 40],
                    mask=self.masks[1],
                ),
            ],
            provenance={"backend": "test-fixture"},
            possibly_truncated=False,
        )

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_multiple_instances_projection_and_replay(self):
        result = consume_categories(
            self.store, [self.event], self.backend, self.root / "result"
        )
        self.assertEqual(result["categories"][0]["accepted"], 2)
        self.assertTrue(all(r["reprojection"]["passed"] for r in result["instances"]))
        self.assertEqual(len(self.store.records("landmarks")), 2)
        for lm, expected in zip(
            sorted(self.store.records("landmarks"), key=lambda lm: -lm["position"][1]),
            ([5.0, 0.6625, 1.0125], [5.0, -0.5375, 1.0125]),
        ):
            np.testing.assert_allclose(
                lm["position"], expected, atol=self.store.config.voxel_size
            )
        for lm in self.store.records("landmarks"):
            self.assertAlmostEqual(
                lm["position"][0], 5.0, delta=self.store.config.voxel_size
            )
            self.assertEqual(lm["status"], "tentative")
            self.assertEqual(lm["observation_ids"], ["a:0"])
        self.assertEqual(self.store.view("a:0")["landmarks"], [])
        before = self.store.summary()
        replay = consume_categories(
            self.store, [self.event], self.backend, self.root / "result"
        )
        self.assertEqual(before, self.store.summary())
        self.assertTrue(all(r["result"]["duplicate"] for r in replay["instances"]))
        self.backend.segment_instances.assert_called_once()

    def test_live_replay_uses_source_pose_after_vehicle_has_moved(self):
        from DynaNav.warehouse_loop.live_llm import apply_llm_predictions

        out = self.root / "request"
        snapshot = MapStore(self.root / "input-snapshot")
        try:
            snapshot.add_observation(self.source)
            consume_categories(
                snapshot, [self.event], self.backend, out / "segmentation"
            )
        finally:
            snapshot.close()
        pose = np.array(self.source.T_odom_body)
        pose[0, 3] += 10
        self.store.add_observation(
            replace(self.source, image_id="a:1", t_obs=2.0, T_odom_body=pose.tolist())
        )
        result = dict(
            directory=str(out),
            events=[asdict(self.event)],
            image_id="a:0",
            llm_valid=True,
            timings={},
            categories=["box"],
            weight_mode="test-fixture",
        )
        _, accepted, consistent = apply_llm_predictions(self.store, result, self.root)
        self.assertEqual(accepted, 2)
        self.assertTrue(consistent)
        for lm in self.store.records("landmarks"):
            self.assertAlmostEqual(
                lm["position"][0], 5, delta=self.store.config.voxel_size
            )
            self.assertEqual(lm["status"], "tentative")
            self.assertEqual(lm["observation_ids"], ["a:0"])
        before = self.store.summary()
        apply_llm_predictions(self.store, result, self.root)
        self.assertEqual(before, self.store.summary())
        self.backend.segment_instances.assert_called_once()
        self.assertTrue((self.root / "semantic.png").exists())

    def test_single_frame_entry_skips_structured_inference(self):
        from ticvla.warehouse.run_semantic_frame import main

        episode = self.root / "episode.json"
        episode.write_text(
            json.dumps(dict(episode_id="test", frames_file="frames.jsonl"))
        )
        frame = asdict(self.source)
        frame["depth_convention"] = "optical_z_metres"
        (self.root / "frames.jsonl").write_text(json.dumps(frame) + "\n")
        model = Mock()
        model.describe_objects.return_value = dict(
            image_id="test:00000000",
            answer="- boxes",
            token_count=2,
            max_new_tokens=160,
        )
        # Keep the fixture's class consistent with the real model answer.
        model.describe_objects.return_value["answer"] = "- box\n- floor"
        output = self.root / "single-frame"
        argv = [
            "run_semantic_frame",
            "--episode",
            str(episode),
            "--frame",
            "0",
            "--model",
            str(self.root),
            "--base-model-only",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
        with (
            patch("sys.argv", argv),
            patch(
                "ticvla.warehouse.run_semantic_frame.StructuredVLM", return_value=model
            ),
            patch(
                "ticvla.warehouse.run_semantic_frame.YOLOESegmenter",
                return_value=self.backend,
            ),
            patch("ticvla.warehouse.run_semantic_frame.render_semantic_frame"),
        ):
            main()
        report = json.loads((output / "report.json").read_text())
        self.assertTrue(report["end_to_end_pass"])
        self.assertEqual(report["llm_calls"], 1)
        self.assertFalse(report["structured_inference_executed"])
        model.describe_objects.assert_called_once()
        model.infer.assert_not_called()
        model.build_prompt.assert_not_called()
        model.decoder.state_dict.assert_not_called()
        self.assertFalse((output / "llm/snapshot.npz").exists())
        response = json.loads((output / "llm/response.json").read_text())
        self.assertEqual(
            response["parsed"]["filtered_categories"][0]["category"], "floor"
        )
        self.assertTrue(report["geometry_occupancy_unchanged"])

    def test_inspection_distinguishes_detection_from_map_acceptance(self):
        from DynaNav.warehouse_loop.live_llm import apply_llm_predictions

        output = self.root / "request"
        self.backend.segment_instances.return_value["instances"][1]["confidence"] = 0.1
        consume_categories(
            self.store, [self.event], self.backend, output / "segmentation"
        )
        result = dict(
            directory=str(output),
            events=[asdict(self.event)],
            image_id="a:0",
            llm_valid=True,
            timings={},
            categories=["box"],
            weight_mode="test",
            raw_answer="- box",
        )
        apply_llm_predictions(self.store, result, self.root)
        inspection = json.loads((self.root / "inspection.json").read_text())
        self.assertEqual(
            [r["status"] for r in inspection["instances"]], ["accepted", "rejected"]
        )
        self.assertEqual(inspection["instances"][0]["origin"], "LLM text -> YOLO")
        detection = np.array(Image.open(self.root / "detections.png"))
        accepted = np.array(Image.open(self.root / "semantic.png"))
        # Pixel inside rejected second mask, away from its border/text.
        self.assertTrue((accepted[95, 65] == [255, 255, 255]).all())
        self.assertFalse((detection[95, 65] == [255, 255, 255]).all())

    def test_numbered_shelf_signs_are_separate_replayable_evidence(self):
        from DynaNav.warehouse_loop.live_perception import (
            save_marker_predictions,
            load_marker_predictions,
            apply_predictions,
        )

        event = replace(
            self.event,
            message_id="ocr:a:0:0",
            source="perception",
            category="shelf_marker",
            name="03/2R",
            roi=[4, 20, 24, 40],
        )
        source, _ = self.store.source_arrays("a:0")
        result = dict(
            image_id="a:0",
            source_hashes={k: source[k] for k in ("rgb_hash", "depth_hash")},
            detections=[],
            predictions=[
                dict(
                    event=asdict(event),
                    mask=self.masks[0],
                    raw_mask=self.masks[0],
                    provenance={},
                    confidence=0.9,
                    mask_outline_iou=1,
                )
            ],
        )
        save_marker_predictions(result, self.root / "markers")
        restored = load_marker_predictions(self.root / "markers")
        rows = apply_predictions(self.store, restored, self.root, publish=False)
        self.assertEqual(rows[0]["label"], "03/2R")
        self.assertEqual(rows[0]["landmark_status"], "tentative")
        before = self.store.summary()
        apply_predictions(self.store, restored, self.root, publish=False)
        self.assertEqual(before, self.store.summary())
        restored["predictions"][0]["event"]["image_id"] = "wrong"
        with self.assertRaisesRegex(ValueError, "event/source mismatch"):
            apply_predictions(self.store, restored, self.root, publish=False)
        (self.root / "markers/mask-000.npz").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "mask changed"):
            load_marker_predictions(self.root / "markers")

    def test_live_worker_skips_structured_inference(self):
        import queue
        from DynaNav.warehouse_loop.live_llm import llm_worker, apply_llm_predictions

        jobs, results = queue.Queue(), queue.Queue()
        source, _ = self.store.source_arrays("a:0")
        jobs.put(
            dict(source=source, map_version=str(self.store.version), dispatched_wall=0)
        )
        jobs.put(None)
        model = Mock()
        model.describe_objects.return_value = dict(
            image_id="a:0", answer="- box", token_count=2, max_new_tokens=160
        )
        self.backend.provenance = {"backend": "test-fixture"}
        output = self.root / "live"
        output.mkdir()
        options = dict(
            model=str(self.root),
            vlm_checkpoint=None,
            base_model_only=True,
            device="cpu",
            max_new_tokens=768,
            llm_mode="objects-only",
            object_evidence=False,
            task="Move along aisle.",
        )
        with (
            patch("ticvla.warehouse.structured_io.StructuredVLM", return_value=model),
            patch(
                "ticvla.warehouse.mapping.segmentation.YOLOESegmenter",
                return_value=self.backend,
            ),
        ):
            llm_worker(jobs, results, "unused-test-weights", output, options)
        self.assertTrue(results.get_nowait()["ready"])
        result = results.get_nowait()
        self.assertEqual(result["llm_calls"], 1)
        self.assertFalse(result["structured_inference_executed"])
        self.assertTrue(results.empty())
        model.describe_objects.assert_called_once()
        model.infer.assert_not_called()
        model.build_prompt.assert_not_called()
        model.decoder.state_dict.assert_not_called()
        self.assertFalse((output / "interface-heads.pt").exists())
        processed, accepted, consistent = apply_llm_predictions(
            self.store, result, output
        )
        self.assertEqual(accepted, 2)
        self.assertTrue(consistent)
        before = self.store.summary()
        apply_llm_predictions(self.store, result, output)
        self.assertEqual(before, self.store.summary())

    def test_category_normalization_and_empty(self):
        duplicate = replace(self.event, message_id="other", category="  BOX ")
        result = consume_categories(
            self.store, [self.event, duplicate], self.backend, self.root / "result"
        )
        self.assertEqual(len(result["categories"]), 1)
        self.backend.reset_mock()
        empty = consume_categories(self.store, [], self.backend, self.root / "empty")
        self.assertEqual(empty["instances"], [])
        self.backend.segment_instances.assert_not_called()

    def test_invalid_source_does_not_run_gpu(self):
        result = consume_categories(
            self.store,
            [replace(self.event, image_id="missing")],
            self.backend,
            self.root / "bad-source",
        )
        self.assertEqual(result["categories"][0]["status"], "source_or_inference_error")
        self.backend.segment_instances.assert_not_called()

    def test_mask_shape_and_missing_depth_rejected(self):
        batch = self.backend.segment_instances.return_value
        batch["instances"][0]["mask"] = np.ones((3, 4), bool)
        result = consume_categories(
            self.store, [self.event], self.backend, self.root / "shape"
        )
        self.assertEqual(result["instances"][0]["result"]["status"], "rejected")
        obs, depth = self.store.source_arrays("a:0")
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            semantic_points(self.masks[0], np.zeros_like(depth), obs, self.store.config)

    def test_no_detection_and_mask_tampering(self):
        saved = self.backend.segment_instances.return_value
        self.backend.segment_instances.return_value = dict(saved, instances=[])
        report = consume_categories(
            self.store, [self.event], self.backend, self.root / "none"
        )
        self.assertEqual(report["categories"][0]["status"], "no_detection")
        self.backend.segment_instances.return_value = saved
        consume_categories(self.store, [self.event], self.backend, self.root / "result")
        (self.root / "result/mask-000.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "mask changed"):
            consume_categories(
                self.store, [self.event], self.backend, self.root / "result"
            )

    def test_cart_never_static_even_manual_claim(self):
        event = replace(
            self.event, source="manual", category="hand truck", static_confirmed=True
        )
        result = self.store.consume(event, self.masks[0], 0.9)
        lm = self.store.get("landmarks", result["landmark_uid"])
        self.assertTrue(lm["potentially_dynamic"])
        self.assertEqual(lm["status"], "tentative")

    def test_legacy_event_replay(self):
        result = self.store.consume(
            replace(self.event, navigation_role=None, reason=None), self.masks[0], 0.9
        )
        del result["event"]["navigation_role"]
        del result["event"]["reason"]
        with self.store.db:
            self.store._put("events", self.event.message_id, result)
        duplicate = self.store.consume(
            replace(self.event, navigation_role=None, reason=None)
        )
        self.assertTrue(duplicate["duplicate"])

    def test_wrong_pose_reprojection_is_detected(self):
        obs, depth = self.store.source_arrays("a:0")
        points = semantic_points(self.masks[0], depth, obs, self.store.config)
        points[:, 0] += 0.5
        self.assertFalse(
            reprojection_check(points, self.masks[0], depth, obs)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
