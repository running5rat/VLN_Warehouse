import json
import unittest
from test_landmark_perception import IdentityMapTests
from ticvla.warehouse.perception.realtime import parse_selection, SelectionState


class SelectionTests(unittest.TestCase):
    def test_alias_empty_and_unapproved(self):
        self.assertEqual(
            parse_selection(
                '{"selected_categories":["Signs","sign"],"proposed_categories":["clock"]}'
            )["selected_categories"],
            ["sign"],
        )
        self.assertEqual(
            parse_selection('{"selected_categories":[],"proposed_categories":[]}')[
                "selected_categories"
            ],
            [],
        )
        for text in [
            "sign",
            '{"selected_categories":["car"],"proposed_categories":[]}',
            '{"selected_categories":[],"proposed_categories":[]} extra',
        ]:
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                parse_selection(text)

    def test_expiry_failure_empty_and_out_of_order(self):
        s = SelectionState()
        self.assertEqual(s.snapshot(0)["status"], "expired")

        def r(t, selected, valid=True):
            return dict(
                valid=valid,
                source_time=t,
                source_image_id=str(t),
                request_id=str(t),
                parsed=dict(selected_categories=selected, proposed_categories=[]),
            )

        s.update(r(1, ["sign"]))
        s.update(r(2, [], False))
        self.assertEqual(s.snapshot(5)["selected_categories"], ["sign"])
        self.assertEqual(s.snapshot(6.1)["status"], "expired")
        s.update(r(7, []))
        s.update(r(3, ["sign"]))
        self.assertEqual(s.snapshot(7)["selected_categories"], [])


class LateTextTests(IdentityMapTests):
    def late(self, event, word="A-17", suffix="a"):
        obs = self.store.get("observations", event.image_id)
        return dict(
            message_id=event.message_id + ":text:" + suffix,
            parent_event_id=event.message_id,
            image_id=event.image_id,
            rgb_hash=obs["rgb_hash"],
            depth_hash=obs["depth_hash"],
            provenance={"backend": "test"},
            texts=[
                dict(
                    text=word,
                    confidence=0.99,
                    status="bound",
                    polygon=[[20, 25], [40, 25], [40, 30], [20, 30]],
                )
            ],
        )

    def test_fill_empty_dedup_geometry_unchanged_restart(self):
        result, e = self.add(0, None)
        before = self.store.get("landmarks", result["landmark_uid"])
        event = self.late(e)
        r = self.store.attach_text_evidence(event)
        self.assertEqual(r["status"], "accepted", r)
        after = self.store.get("landmarks", result["landmark_uid"])
        for key in ["geometry", "observation_ids", "position", "status"]:
            self.assertEqual(before[key], after[key])
        self.assertEqual(after["identity_text"], "A-17")
        self.assertEqual(len(after["text_evidence"]), 1)
        v = self.store.version
        self.assertTrue(self.store.attach_text_evidence(event)["duplicate"])
        self.assertEqual(v, self.store.version)
        self.store.close()
        from ticvla.warehouse.mapping import MapStore

        self.store = MapStore(self.root / "map")
        self.assertTrue(self.store.attach_text_evidence(event)["duplicate"])
        conflict = self.late(e, "B-18", "b")
        self.store.attach_text_evidence(conflict)
        self.assertEqual(
            self.store.get("landmarks", result["landmark_uid"])["identity_status"],
            "conflict",
        )

    def test_source_parent_and_three_distinct_frames(self):
        result, e = self.add(0, None)
        bad = self.late(e)
        bad["rgb_hash"] = "wrong"
        self.assertEqual(self.store.attach_text_evidence(bad)["status"], "rejected")
        for suffix in ["b", "c", "d"]:
            self.store.attach_text_evidence(self.late(e, suffix=suffix))
        self.assertEqual(
            self.store.get("landmarks", result["landmark_uid"])["identity_status"],
            "candidate",
        )
        bad = self.late(e, suffix="badparent")
        bad["parent_event_id"] = "absent"
        self.assertEqual(self.store.attach_text_evidence(bad)["status"], "rejected")

    def test_late_out_of_order_confirmation_remains_tentative(self):
        events = []
        for i in range(3):
            result, event = self.add(i, None)
            events.append(event)
        uid = result["landmark_uid"]
        geometry = self.store.get("landmarks", uid)["geometry"]
        for event in reversed(events):
            self.assertEqual(
                self.store.attach_text_evidence(self.late(event))["status"], "accepted"
            )
        record = self.store.get("landmarks", uid)
        self.assertEqual(record["identity_status"], "identity_confirmed")
        self.assertEqual(record["status"], "tentative")
        self.assertEqual(record["geometry"], geometry)
        self.assertEqual(self.store.view("a:2")["landmarks"], [])

    def test_rejected_parent_cannot_receive_text(self):
        import numpy as np
        from ticvla.warehouse.mapping import SemanticEvent

        obs = self.store.add_observation(self.obs)
        event = SemanticEvent("rejected", obs["image_id"], "sign", source="llm")
        self.assertEqual(
            self.store.consume(event, np.zeros_like(self.mask), 0.9)["status"],
            "rejected",
        )
        self.assertEqual(
            self.store.attach_text_evidence(self.late(event))["status"], "rejected"
        )
        self.assertEqual(self.store.records("landmarks"), [])


class ExactVoxelTests(unittest.TestCase):
    def test_scalar_keys_preserve_first_points_exactly(self):
        import numpy as np
        from ticvla.warehouse.mapping.store import voxelize

        rng = np.random.default_rng(7)
        for points in [
            rng.normal(size=(10000, 3)),
            np.ones((40, 3)),
            np.empty((0, 3)),
            np.array([[1e10, 1e10, 1e10], [-1e10, -1e10, -1e10]]),
        ]:
            _, ids = np.unique(
                np.floor(points / 0.05).astype(np.int64), axis=0, return_index=True
            )
            np.testing.assert_array_equal(voxelize(points, 0.05), points[np.sort(ids)])
