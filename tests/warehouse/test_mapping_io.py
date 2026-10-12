import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from ticvla.warehouse.geometry import planar_pose
from ticvla.warehouse.mapping import (
    MapStore,
    MappingConfig,
    SemanticEvent,
    SourceObservation,
)
from ticvla.warehouse.mapping.store import semantic_points
from ticvla.warehouse.structured_io import parse_structured, response_grammar


class MappingTests(unittest.TestCase):
    def test_fast_npz_is_lossless_and_reopens_with_default_reader(self):
        fast = MapStore(self.root / "fast-map", array_compression_level=1)
        fast_record = fast.add_observation(self.obs)
        np.testing.assert_array_equal(
            fast.points(fast_record),
            self.store.points(self.store.get("observations", "a:0")),
        )
        fast.close()
        reopened = MapStore(self.root / "fast-map")
        try:
            _, depth = reopened.source_arrays("a:0")
            np.testing.assert_array_equal(depth, np.load(self.root / "depth.npy"))
            self.assertEqual(
                reopened.add_observation(self.obs)["geometry"], fast_record["geometry"]
            )
        finally:
            reopened.close()

    def test_fixed_marker_policy_is_explicit_and_class_scoped(self):
        event = SemanticEvent(
            "marker", "a:0", "shelf_marker", source="perception", static_confirmed=True
        )
        result = self.store.consume(event, self.masks[0], 0.9)
        self.assertEqual(
            self.store.get("landmarks", result["landmark_uid"])["status"], "tentative"
        )
        enabled = MapStore(
            self.root / "fixed-markers", MappingConfig(fixed_shelf_markers=True)
        )
        try:
            enabled.add_observation(self.obs)
            result = enabled.consume(event, self.masks[0], 0.9)
            self.assertEqual(
                enabled.get("landmarks", result["landmark_uid"])["status"], "verified"
            )
            person = replace(event, message_id="person", category="person")
            result = enabled.consume(person, self.masks[1], 0.9)
            self.assertEqual(
                enabled.get("landmarks", result["landmark_uid"])["status"], "tentative"
            )
        finally:
            enabled.close()

    def test_marker_depth_plane_rejects_nearby_background(self):
        from ticvla.warehouse.mapping.markers import planar_marker_mask

        depth = np.full((80, 100), 2.0, dtype=np.float32)
        depth[:, :15] = 2.15  # Within generic MAD floor, but not on the sign.
        mask = np.ones_like(depth, bool)
        K = [[100, 0, 50], [0, 100, 40], [0, 0, 1]]
        refined = planar_marker_mask(mask, depth, K)
        self.assertIsNotNone(refined)
        self.assertFalse(refined[:, :15].any())
        self.assertTrue(refined[:, 15:].all())
        self.assertIsNone(planar_marker_mask(mask, np.zeros_like(depth), K))

    def test_conflicting_text_cannot_duplicate_same_geometry(self):
        self.add_landmarks("a:0")
        conflict = self.store.consume(
            SemanticEvent(
                "conflict", "a:0", "box", name="wrong-name", static_confirmed=True
            ),
            self.masks[0],
            0.9,
        )
        self.assertEqual(conflict["status"], "rejected")
        self.assertIn("Conflicting landmark identity", conflict["reason"])
        self.assertEqual(len(self.store.records("landmarks")), 3)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.store = MapStore(self.root / "map")
        Image.new("RGB", (60, 40), "white").save(self.root / "rgb.png")
        depth = np.zeros((40, 60), dtype=np.float32)
        self.masks = []
        for i, (x, y, d) in enumerate(((3, 8, 2.0), (26, 13, 4.0), (45, 22, 3.0))):
            mask = np.zeros_like(depth, bool)
            mask[y : y + 10, x : x + 10] = True
            depth[mask] = d
            self.masks.append(mask)
        np.save(self.root / "depth.npy", depth)
        self.obs = SourceObservation(
            "a:0",
            str(self.root / "rgb.png"),
            str(self.root / "depth.npy"),
            [[50, 0, 30], [0, 50, 20], [0, 0, 1]],
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1]],
            np.eye(4).tolist(),
            1.0,
            "epoch-a",
            "a",
            "task-a",
        )
        self.store.add_observation(self.obs)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def add_landmarks(self, image_id):
        results = []
        for i, mask in enumerate(self.masks):
            results.append(
                self.store.consume(
                    SemanticEvent(
                        f"{image_id}:{i}",
                        image_id,
                        "box",
                        name=f"box-{i}",
                        static_confirmed=True,
                    ),
                    mask,
                    0.9,
                )
            )
        return results

    def test_projection_persistence_and_deduplication(self):
        results = self.add_landmarks("a:0")
        self.assertTrue(all(r["status"] == "accepted" for r in results))
        first = self.store.get("landmarks", results[0]["landmark_uid"])
        # Optical Z=2 maps to body X=2; metric projection has no ray-distance conversion.
        self.assertAlmostEqual(first["position"][0], 2.0, places=6)
        before = self.store.summary()
        result = self.store.consume(
            SemanticEvent("a:0:0", "a:0", "box", name="box-0", static_confirmed=True)
        )
        self.assertTrue(result["duplicate"])
        self.assertEqual(self.store.summary(), before)
        with self.assertRaises(ValueError):
            self.store.consume(SemanticEvent("a:0:0", "a:0", "shelf"))
        self.store.close()
        self.store = MapStore(self.root / "map")
        self.assertEqual(before, self.store.summary())
        self.store.add_observation(
            replace(self.obs, image_id="a:1", task_id="task-b", t_obs=2.0)
        )
        self.add_landmarks("a:1")
        self.assertEqual(len(self.store.records("landmarks")), 3)
        self.assertTrue(
            all(len(x["observation_ids"]) == 2 for x in self.store.records("landmarks"))
        )

    def test_rejections_do_not_change_geometry_or_version(self):
        version = self.store.version
        for event, mask in [
            (SemanticEvent("missing", "missing", "box"), self.masks[0]),
            (SemanticEvent("none", "a:0", "box"), None),
            (SemanticEvent("oracle", "a:0", "box", source="oracle"), self.masks[0]),
            (SemanticEvent("relation", "a:0", "left", kind="relation"), self.masks[0]),
            (SemanticEvent("shape", "a:0", "box"), np.ones((2, 2))),
        ]:
            result = self.store.consume(event, mask, 0.9)
            self.assertEqual(result["status"], "rejected")
        self.assertEqual(self.store.version, version)
        self.assertEqual(self.store.records("landmarks"), [])
        with self.assertRaises(ValueError):
            MapStore(self.root / "map", oracle=True)

    def test_dynamic_and_llm_candidates_are_not_static_anchors(self):
        dynamic = self.store.consume(
            SemanticEvent("d", "a:0", "person", static_confirmed=True),
            self.masks[0],
            0.9,
        )
        predicted = self.store.consume(
            SemanticEvent("l", "a:0", "box", source="llm", static_confirmed=True),
            self.masks[1],
            0.9,
        )
        for item in (dynamic, predicted):
            lm = self.store.get("landmarks", item["landmark_uid"])
            self.assertFalse(lm["static_confirmed"])
        view = self.store.view("a:0")
        self.assertEqual(view["landmarks"], [])
        self.assertGreater(
            int((view["grid"] == 1).sum()), 0
        )  # Unselected geometry remains.

    def test_new_epoch_registration_rejection_and_revoke(self):
        self.add_landmarks("a:0")
        tf = planar_pose(1.1, -0.7, 0.35)
        b = replace(
            self.obs,
            image_id="b:0",
            submap_id="b",
            odom_epoch="epoch-b",
            task_id="task-b",
            T_odom_body=np.linalg.inv(tf).tolist(),
        )
        self.store.add_observation(b)
        self.assertEqual(self.store.register("b", "a")["status"], "rejected")
        self.add_landmarks("b:0")
        result = self.store.register("b", "a")
        self.assertEqual(result["status"], "accepted")
        np.testing.assert_allclose(result["T_target_source"], tf, atol=0.05)
        self.assertEqual(len(self.store.view("b:0")["landmarks"]), 3)
        raw = {
            x["landmark_uid"]: self.store.points(x)
            for x in self.store.records("landmarks")
        }
        self.store.revoke(result["constraint_id"])
        self.assertIsNone(self.store.get("submaps", "b")["T_map_submap"])
        for lm in self.store.records("landmarks"):
            np.testing.assert_array_equal(
                raw[lm["landmark_uid"]], self.store.points(lm)
            )

    def test_future_geometry_and_mutated_sources_are_rejected(self):
        self.store.add_observation(replace(self.obs, image_id="a:future", t_obs=100.0))
        self.add_landmarks("a:future")
        self.assertEqual(self.store.view("a:0")["landmarks"], [])
        with self.assertRaises(ValueError):
            self.store.add_observation(replace(self.obs, image_id="a:back", t_obs=2.0))
        Image.new("RGB", (60, 40), "black").save(self.root / "rgb.png")
        result = self.store.consume(
            SemanticEvent("tampered", "a:0", "box"), self.masks[0], 0.9
        )
        self.assertEqual(result["status"], "rejected")

    def test_mask_boundary_background_does_not_move_landmark(self):
        depth = np.load(self.root / "depth.npy")
        mask = self.masks[0].copy()
        mask[7, 3:13] = True
        depth[7, 3:13] = 20.0
        points = semantic_points(
            mask, depth, self.store.get("observations", "a:0"), self.store.config
        )
        np.testing.assert_allclose(points[:, 0], 2.0)

    def test_delayed_event_uses_source_pose_and_revocation_preserves_evidence(self):
        self.store.add_observation(
            replace(
                self.obs,
                image_id="a:later",
                t_obs=2.0,
                T_odom_body=planar_pose(5, 0).tolist(),
            )
        )
        item = self.store.consume(
            SemanticEvent("delayed", "a:0", "box", static_confirmed=True),
            self.masks[0],
            0.9,
        )
        lm = self.store.get("landmarks", item["landmark_uid"])
        self.assertAlmostEqual(lm["position"][0], 2.0)
        points = self.store.points(lm)
        self.store.revoke_landmark(lm["landmark_uid"], "manual wrong association")
        archived = self.store.get("landmarks", lm["landmark_uid"])
        np.testing.assert_array_equal(points, self.store.points(archived))
        self.assertEqual(self.store.view("a:later")["landmarks"], [])


class ProtocolTests(unittest.TestCase):
    def test_source_aware_response(self):
        text = '<landmark>[{"image_id":"a:0","category":"box"}]</landmark><local>unknown</local><Route>Go forward.</Route><waypoint>[]</waypoint><REFLINE>'
        self.assertEqual(len(parse_structured(text, ["a:0"])["landmark"]), 1)
        with self.assertRaises(ValueError):
            parse_structured(text, ["b:0"])
        with self.assertRaises(ValueError):
            parse_structured(text[:-4], ["a:0"])

    @unittest.skipUnless(
        importlib.util.find_spec("lmformatenforcer"), "Optional I/O requirements"
    )
    def test_grammar_accepts_real_protocol_and_rejects_invalid_source(self):
        for landmarks in ("[]", '[{"image_id":"a:0","category":"box"}]'):
            text = f"<landmark>{landmarks}</landmark><local>unknown</local><Route>Go forward.</Route><waypoint>(0,0);(1,0);(2,0);(3,0);(4,0)</waypoint><REFLINE>"
            parser = response_grammar(["a:0"])
            for char in text:
                self.assertIn(char, parser.get_allowed_characters())
                parser = parser.add_character(char)
            self.assertTrue(parser.can_end())


if __name__ == "__main__":
    unittest.main()
