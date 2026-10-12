from dataclasses import asdict, replace
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
from ticvla.warehouse.perception.common import associate_texts, crop_box, polygon_mask
from ticvla.warehouse.mapping import MapStore, SemanticEvent, SourceObservation
from ticvla.warehouse.mapping.semantic_frame import consume_categories
from ticvla.warehouse.object_io import (
    object_observation,
    object_question,
)


class BindingTests(unittest.TestCase):
    def text(self, text="AB-17"):
        return dict(
            text=text, confidence=0.95, polygon=[[12, 12], [22, 12], [22, 18], [12, 18]]
        )

    def test_smallest_carrier_no_copy(self):
        rack = np.ones((50, 50), bool)
        sign = np.zeros_like(rack)
        sign[8:25, 8:30] = True
        instances = [dict(mask=rack), dict(mask=sign)]
        texts = associate_texts(instances, [self.text()], rack.shape)
        self.assertEqual(texts[0]["owner_instance"], 1)
        self.assertFalse(instances[0]["texts"])
        self.assertEqual(len(instances[1]["texts"]), 1)

    def test_ambiguity_low_confidence_and_empty(self):
        masks = [dict(mask=np.ones((50, 50), bool)) for _ in range(2)]
        texts = associate_texts(masks, [self.text()], (50, 50))
        self.assertEqual(texts[0]["status"], "ambiguous")
        t = self.text()
        t["confidence"] = 0.1
        self.assertEqual(
            associate_texts(masks, [t], (50, 50))[0]["status"], "low_confidence"
        )
        self.assertEqual(associate_texts([], [], (50, 50)), [])

    def test_crop_restoration_and_polygon_bounds(self):
        box = crop_box([10, 10, 20, 30], (50, 50))
        local = np.array([[1, 1], [5, 1], [5, 6], [1, 6]])
        full = local + box[:2]
        mask = polygon_mask(full, (50, 50))
        self.assertTrue(mask[box[1] + 2, box[0] + 2])
        with self.assertRaises(ValueError):
            polygon_mask([[-1, 0], [1, 0], [1, 2]], (50, 50))


class IdentityMapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        Image.new("RGB", (100, 80)).save(self.root / "rgb.png")
        np.save(self.root / "depth.npy", np.full((80, 100), 2.0, np.float32))
        self.store = MapStore(self.root / "map")
        self.obs = SourceObservation(
            "a:0",
            str(self.root / "rgb.png"),
            str(self.root / "depth.npy"),
            [[100, 0, 50], [0, 100, 40], [0, 0, 1]],
            np.eye(4).tolist(),
            np.eye(4).tolist(),
            0,
            "a",
            "a",
            "task",
        )
        self.mask = np.zeros((80, 100), bool)
        self.mask[20:55, 15:55] = True

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add(self, i, text="EQ-17", mask=None, shift=None, confidence=0.9):
        pose = np.eye(4)
        pose[0, 3] = i * 0.08 if shift is None else shift
        obs = replace(
            self.obs, image_id=f"a:{i}", t_obs=float(i), T_odom_body=pose.tolist()
        )
        self.store.add_observation(obs)
        texts = (
            []
            if text is None
            else [
                dict(
                    text=text,
                    normalized=text,
                    confidence=0.99,
                    status="bound",
                    polygon=[[20, 25], [40, 25], [40, 30], [20, 30]],
                )
            ]
        )
        event = SemanticEvent(
            f"e{i}", obs.image_id, "box", source="llm", text_evidence=texts
        )
        result = self.store.consume(
            event,
            self.mask if mask is None else mask,
            confidence,
            segmentation={
                "backend": "florence2" if confidence is None else "yoloe-ppocr"
            },
        )
        self.assertEqual(result["status"], "accepted", result)
        return result, event

    def test_three_views_identity_not_static_restart_and_duplicate(self):
        for i in range(3):
            result, event = self.add(i)
        record = self.store.get("landmarks", result["landmark_uid"])
        self.assertEqual(record["identity_status"], "identity_confirmed")
        self.assertEqual(record["status"], "tentative")
        before = self.store.summary()
        self.assertTrue(self.store.consume(event, self.mask, 0.9)["duplicate"])
        self.assertEqual(before, self.store.summary())
        self.store.close()
        self.store = MapStore(self.root / "map")
        self.assertEqual(
            self.store.get("landmarks", result["landmark_uid"])["identity_text"],
            "EQ-17",
        )
        self.assertEqual(self.store.view("a:2")["landmarks"], [])

    def test_same_view_not_confirmation_and_conflict(self):
        for i in range(3):
            result, _ = self.add(i, shift=0)
        self.assertEqual(
            self.store.get("landmarks", result["landmark_uid"])["identity_status"],
            "candidate",
        )
        result, _ = self.add(3, text="EQ-18", shift=0)
        record = self.store.get("landmarks", result["landmark_uid"])
        self.assertEqual(record["identity_status"], "conflict")
        self.assertEqual(record["identity_candidates"], ["EQ-17", "EQ-18"])
        self.assertIsNone(record["name"])

    def test_spatially_separated_equal_text_not_merged(self):
        result, _ = self.add(0, shift=0)
        mask = np.zeros_like(self.mask)
        mask[20:55, 70:98] = True
        second, _ = self.add(1, mask=mask, shift=0)
        self.assertNotEqual(result["landmark_uid"], second["landmark_uid"])
        self.assertTrue(
            all(r["duplicate_text"] for r in self.store.records("landmarks"))
        )

    def test_unknown_confidence_is_not_fabricated(self):
        result, _ = self.add(0, confidence=None)
        self.assertIsNone(
            self.store.get("landmarks", result["landmark_uid"])["confidence"]
        )

    def test_source_and_depth_rejections(self):
        event = SemanticEvent("bad", "missing", "sign", source="llm")
        self.assertEqual(
            self.store.consume(event, self.mask, 0.9)["status"], "rejected"
        )
        obs = replace(self.obs, image_id="invalid-depth")
        np.save(self.root / "depth.npy", np.zeros((80, 100), np.float32))
        self.store.add_observation(obs)
        event = replace(event, message_id="depth", image_id=obs.image_id)
        self.assertEqual(
            self.store.consume(event, self.mask, 0.9)["status"], "rejected"
        )

    def test_task_is_in_actual_question(self):
        obs = object_observation(asdict(self.obs), task="Find the charging station.")
        self.assertIn("Find the charging station.", object_question(obs))
        self.assertNotIn("map_landmarks", obs)
        self.assertNotIn("navigation_task", object_observation(asdict(self.obs)))

    def test_legacy_snapshot_without_text_field_replays(self):
        self.store.add_observation(self.obs)
        event = SemanticEvent("legacy", self.obs.image_id, "box", source="llm")
        mask = self.mask

        class Backend:
            def segment_instances(self, observation, categories):
                return dict(
                    instances=[
                        dict(
                            category="box",
                            bbox=[15, 20, 55, 55],
                            mask=mask,
                            confidence=0.9,
                        )
                    ],
                    provenance={},
                    possibly_truncated=False,
                )

        output = self.root / "segmentation"
        consume_categories(self.store, [event], Backend(), output)
        snapshot = json.loads((output / "detections.json").read_text())
        del snapshot["requests"][0]["text_evidence"]
        (output / "detections.json").write_text(json.dumps(snapshot))
        before = self.store.summary()
        consume_categories(self.store, [event], None, output)
        self.assertEqual(before, self.store.summary())


class LiveGateTests(unittest.TestCase):
    def test_no_eligible_backend_does_not_launch(self):
        from unittest.mock import patch
        from ticvla.warehouse.perception.validate_live import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.json").write_text(
                json.dumps(
                    dict(
                        winner=None,
                        backends={
                            n: dict(complete=True, gate_pass=False)
                            for n in ("yoloe-ppocr", "florence2")
                        },
                    )
                )
            )
            with patch("subprocess.run") as launch:
                run(root)
                launch.assert_not_called()
            result = json.loads((root / "live-validation.json").read_text())
            self.assertEqual(result["status"], "not_eligible")
            self.assertEqual(result["runs"], [])

    def test_incomplete_evaluation_cannot_launch(self):
        from unittest.mock import patch
        from ticvla.warehouse.perception.validate_live import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "comparison.json").write_text(
                json.dumps(
                    dict(
                        winner="florence2",
                        backends={"florence2": dict(complete=False, gate_pass=True)},
                    )
                )
            )
            with patch("subprocess.run") as launch, self.assertRaises(ValueError):
                run(root)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
