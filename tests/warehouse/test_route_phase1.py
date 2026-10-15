import copy
import json
import unittest
import numpy as np
from ticvla.warehouse.navigation.route_protocol import (
    parse_route,
    reference_diagnostics,
)
from ticvla.warehouse.navigation.execution import ExecutionFacts
from ticvla.warehouse.navigation.core import NavigationState
from ticvla.warehouse.structured_io import parse_structured, response_grammar
from ticvla.warehouse.geometry import planar_pose


def route(action="relative_motion", **kwargs):
    goal = dict(action=action)
    if action == "relative_motion":
        goal.update(direction="backward", distance_m=5.0, origin="subgoal_start")
    goal.update(kwargs)
    return dict(
        remaining_instruction="Move backwards five metres.",
        current_subgoal=goal,
        status="active",
    )


class RouteTests(unittest.TestCase):
    def setup_response(self):
        nav = NavigationState("t", instruction="倒退5m")
        obs = dict(
            task={"task_id": "t"},
            route_revision=0,
            source_monotonic=1.0,
            T_odom_body=np.eye(4).tolist(),
            odom_epoch="e",
            map_version="1",
            rgb_history=[dict(image_id="f")],
        )
        result = dict(
            valid=True,
            inference_mode="TEXT",
            parsed=dict(route=route(), anchors=[], landmark=[]),
            z_sem=np.zeros((16, 512)),
            reference_xy=np.c_[np.linspace(0, 1, 32), np.zeros(32)],
            reference_mask=np.ones(32, bool),
            confidence=0.5,
        )
        return nav, obs, result

    def test_protocol_validation_and_legacy(self):
        self.assertEqual(parse_route("Keep going"), "Keep going")
        self.assertEqual(parse_route(json.dumps(route()), True), route())
        for bad in (
            route(direction="side"),
            route(distance_m=float("nan")),
            route(action="invent"),
            route(action="advance_to_intersection", ordinal=True),
            route(action="turn", direction="forward"),
        ):
            with self.assertRaises(ValueError):
                parse_route(bad, True)
        with self.assertRaises(ValueError):
            parse_route("legacy", True)

    def test_real_grammar_accepts_structured_and_empty_waypoint(self):
        text = (
            "<landmark>[]</landmark><local>unknown</local><Route>"
            + json.dumps(route(), separators=(",", ":"))
            + "</Route><waypoint>[]</waypoint><REFLINE>"
        )
        parser = response_grammar(["f"], structured_route=True)
        for c in text:
            self.assertIn(c, parser.get_allowed_characters(), text)
            parser = parser.add_character(c)
        self.assertTrue(parser.can_end())
        self.assertEqual(
            parse_structured(text, ["f"], structured_route=True)["route"], route()
        )
        with self.assertRaises(ValueError):
            parse_structured(text[:-4], ["f"], structured_route=True)
        bad = text.replace(
            "[]</landmark>", '[{"image_id":"wrong","category":"sign"}]</landmark>'
        )
        with self.assertRaises(ValueError):
            parse_structured(bad, ["f"], structured_route=True)

    def test_recursive_atomic_invalid_reference_and_reverse_no_execution(self):
        n, o, r = self.setup_response()
        original = n.state.contract
        p = n.commit(r, o, 1, 1.1, 1.2, 1.3, 10)
        self.assertEqual(n.state.route, route())
        self.assertIs(n.state.contract, original)
        self.assertFalse(p.reference_usable)
        self.assertEqual(p.reference_validation["execution"], "execution_not_evaluated")
        nextobs = copy.deepcopy(o)
        nextobs.update(route=n.state.route, route_revision=1)
        n.commit(r, nextobs, 2, 1.1, 1.2, 1.3, 10)
        self.assertEqual(n.route_history[-1]["previous_route"], route())
        with self.assertRaises(ValueError):
            n.commit(r, o, 3, 1.1, 1.2, 1.3, 10)
        with self.assertRaises(ValueError):
            n.commit(r, dict(nextobs, route_revision=2), 3, 1.1, 1.2, 100, 10)
        self.assertEqual(n.state.route_revision, 2)

    def test_committed_route_does_not_alias_response_dictionary(self):
        n, o, r = self.setup_response()
        n.commit(r, o, 1, 1.1, 1.2, 1.3, 10)
        r["parsed"]["route"]["remaining_instruction"] = "changed externally"
        self.assertNotEqual(
            n.state.route["remaining_instruction"], "changed externally"
        )

    def test_empty_reference_is_unavailable_not_text_inconsistency(self):
        d = reference_diagnostics(
            dict(anchors=[]), np.zeros((32, 2)), np.zeros(32, bool)
        )
        self.assertTrue(d["consistency_valid"])
        self.assertFalse(d["reference_available"])

    def test_waypoint_distance_order_direction(self):
        line = np.c_[np.linspace(-0.2, -5, 32), np.zeros(32)]
        parsed = dict(route=route(), anchors=[[-i, 0] for i in range(1, 6)])
        d = reference_diagnostics(parsed, line, np.ones(32, bool))
        self.assertTrue(d["consistency_valid"])
        parsed["anchors"].reverse()
        self.assertIn(
            "anchor_order",
            reference_diagnostics(parsed, line, np.ones(32, bool))["reasons"],
        )
        self.assertFalse(
            reference_diagnostics(dict(anchors=[]), line, np.ones(32, bool))[
                "consistency_valid"
            ]
        )
        self.assertTrue(
            reference_diagnostics({}, line, np.ones(32, bool), "PREFILL_FOLLOW")[
                "consistency_valid"
            ]
        )

    def test_odometry_anchors_and_intersections(self):
        f = ExecutionFacts()
        f.observe(np.eye(4), 0, "e")
        f.switch(route()["current_subgoal"], np.eye(4).tolist(), 1)
        f.observe(planar_pose(-2, 0, 0.2), 1, "e")
        f.switch(
            route(distance_m=3)["current_subgoal"], planar_pose(-2, 0, 0.2).tolist(), 2
        )
        self.assertAlmostEqual(f.data["subgoal_path_m"], 2)
        self.assertEqual(f.data["subgoal"]["distance_m"], 5)
        self.assertAlmostEqual(f.data["subgoal_turn_rad"], 0.2)
        for i in range(3):
            f.intersection([0, 0], True, str(i))
        self.assertEqual(len(f.data["intersections"]), 0)
        f.intersection([1, 0], False, "exit")
        f.intersection([0, 0], True, "again")
        f.intersection([1, 0], False, "exit2")
        self.assertEqual(len(f.data["intersections"]), 1)

    def test_completion_cancel_restore(self):
        n, o, r = self.setup_response()
        r["parsed"]["route"]["status"] = "completed"
        n.commit(r, o, 1, 1.1, 1.2, 1.3, 10)
        self.assertIsNone(n.terminal)
        self.assertTrue(n.state.execution["completion_claim_rejected"])
        restored = NavigationState.restore(json.loads(json.dumps(n.checkpoint())))
        self.assertEqual(restored.state.route, n.state.route)
        self.assertIsNone(restored.snapshot)
        restored.terminal = "cancelled"
        with self.assertRaises(ValueError):
            restored.commit(r, dict(o, route_revision=1), 2, 1.1, 1.2, 1.3, 10)

    def test_subgoal_locks_latest_execution_pose_not_stale_request_pose(self):
        n, o, r = self.setup_response()
        n.facts.observe(planar_pose(1, 0, 0), 2, "e")
        n.commit(r, o, 1, 2.1, 2.2, 2.3, 10)
        self.assertAlmostEqual(n.facts.data["subgoal_start_pose"][0][3], 1.0)
        np.testing.assert_equal(n.snapshot.T_odom_body_obs, o["T_odom_body"])

    def test_prefill_cannot_promote_unsupported_completion(self):
        n, o, r = self.setup_response()
        r["parsed"]["route"] = route(
            "follow_person", description="the only person ahead"
        )
        r["parsed"]["route"]["status"] = "completed"
        n.commit(r, o, 1, 1.1, 1.2, 1.3, 10)
        self.assertIsNone(n.terminal)
        o.update(route_revision=1, target_visible=True, target_binding_id="person")
        r.update(inference_mode="PREFILL_FOLLOW", parsed=None)
        n.commit(r, o, 2, 1.1, 1.2, 1.3, 10)
        self.assertIsNone(n.terminal)
        self.assertEqual(n.state.route_revision, 1)


class GeometryAndBindingTests(unittest.TestCase):
    def test_forward_then_turn_with_measured_feedback(self):
        """Kinematic fixture, not Isaac or a model answer: exercise real geometry functions."""
        from ticvla.warehouse.navigation.route_geometry import instruction_path
        from ticvla.warehouse.navigation.core import pursuit

        grid = np.zeros((160, 160), np.int8)
        f = ExecutionFacts()
        x = y = yaw = stamp = 0.0
        f.observe(planar_pose(x, y, yaw), stamp, "fixture")
        goals = [
            dict(action="relative_motion", direction="forward", distance_m=0.6, origin="subgoal_start"),
            dict(action="turn", direction="left"),
        ]
        for revision, goal in enumerate(goals):
            f.switch(goal, planar_pose(x, y, yaw).tolist(), revision)
            for _ in range(400):
                path, reason, cmd = instruction_path(goal, f.record(), grid,
                                                     planar_pose(x, y, yaw), 0.1, 0.35)
                if cmd == (0.0, 0.0):
                    break
                if cmd is None:
                    self.assertIsNotNone(path, reason)
                    cmd, _ = pursuit(path, grid, speed=0.15)
                v, w = cmd
                x += v * np.cos(yaw) * 0.1
                y += v * np.sin(yaw) * 0.1
                yaw += w * 0.1
                stamp += 0.1
                f.observe(planar_pose(x, y, yaw), stamp, "fixture", cmd)
            else:
                self.fail("Correct fixture subgoal did not finish: " + reason)
            self.assertTrue(f.completion_evidence())
        self.assertAlmostEqual(x, 0.6, delta=0.06)
        self.assertAlmostEqual(yaw, np.pi / 2, delta=0.11)

    def test_watchdog_valid_then_cancel_expire_and_obstacle(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from DynaNav.warehouse_loop.navigation_control import NavigationControl

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "navigation"
            root.mkdir()
            with patch("DynaNav.warehouse_loop.navigation_control.time.monotonic", return_value=10.0):
                control = NavigationControl(root)
            command = dict(backend="geometry-baseline", v=0.2, w=0,
                           created_monotonic=10.0, valid_until=11.0, odom_epoch="e", reason="fixture")
            (root / "command.json").write_text(json.dumps(command))
            np.savez(root / "safety.npz", grid=np.zeros((160, 160), np.int8),
                     pose=np.eye(4), odom_epoch="e", source_monotonic=10.0, resolution=0.1)
            with patch("DynaNav.warehouse_loop.navigation_control.time.monotonic", return_value=10.1):
                self.assertGreater(control.command(np.eye(4), "e")[0], 0)
                (root.parent / "cancel.json").write_text("{}")
                self.assertEqual(control.command(np.eye(4), "e"), (0, 0))
                (root.parent / "cancel.json").unlink()
                self.assertEqual(control.command(np.eye(4), "wrong-epoch"), (0, 0))
                control.cached_safety["grid"][:] = 1
                self.assertEqual(control.command(np.eye(4), "e"), (0, 0))
            with patch("DynaNav.warehouse_loop.navigation_control.time.monotonic", return_value=12.0):
                self.assertEqual(control.command(np.eye(4), "e"), (0, 0))
            control.close()

    def test_audit_preserves_actual_answer_and_masked_points(self):
        import tempfile
        from pathlib import Path
        from ticvla.warehouse.navigation.audit import write_audit

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            obs = dict(task=dict(instruction="倒退5 m。"), route="倒退5 m。")
            result = dict(prompt="actual prompt", text="invalid actual output", valid=False,
                          error="parse failed", reference_xy=np.array([[0, 0], [-5, 0]]),
                          reference_mask=np.array([True, False]))
            write_audit(root, obs, result)
            self.assertEqual((root / "answer.txt").read_text(), result["text"])
            audit = json.loads((root / "audit.json").read_text())
            self.assertFalse(audit["interface_valid"])
            self.assertFalse(audit["quality_validated"])
            points = np.loadtxt(root / "reference.csv", delimiter=",", skiprows=1)
            self.assertEqual(points.shape, (2, 4))
            self.assertEqual(points[1, 1], -5)
            self.assertEqual(points[1, 3], 0)

    def test_corridor_unknown_is_not_end_and_crossing_is_observed(self):
        from ticvla.warehouse.navigation.route_geometry import corridor_observation

        grid = np.full((200, 200), -1, np.int8)
        grid[15:180, 90:111] = 0
        obs = corridor_observation(grid, 0.05, 0.2)
        self.assertFalse(obs["corridor_end_observed"])
        self.assertFalse(obs["inside_intersection"])
        grid[90:111, 15:185] = 0
        obs = corridor_observation(grid, 0.05, 0.2)
        self.assertTrue(obs["inside_intersection"])

    def test_backward_never_calls_geometry_planner(self):
        from unittest.mock import patch
        from ticvla.warehouse.navigation.route_geometry import instruction_path

        with patch(
            "ticvla.warehouse.navigation.route_geometry.astar_reference",
            side_effect=AssertionError("must not plan reverse"),
        ):
            path, reason, cmd = instruction_path(
                route()["current_subgoal"],
                {},
                np.full((20, 20), -1),
                np.eye(4),
                0.05,
                0.35,
            )
        self.assertIsNone(path)
        self.assertEqual(reason, "execution_not_evaluated")
        self.assertIsNone(cmd)

    def test_isaac_watchdog_enforces_diagnostic_backend(self):
        import tempfile
        import time
        from pathlib import Path
        from DynaNav.warehouse_loop.navigation_control import NavigationControl

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            controller = NavigationControl(root)
            now = time.monotonic()
            (root / "command.json").write_text(
                json.dumps(
                    dict(
                        backend="reference-only",
                        v=0.3,
                        w=0.2,
                        created_monotonic=now,
                        valid_until=now + 10,
                        reason="test",
                    )
                )
            )
            self.assertEqual(controller.command(np.eye(4), "e"), (0.0, 0.0))
            controller.close()

    def test_unique_person_binding_ambiguity_and_no_auto_switch(self):
        from ticvla.warehouse.navigation.core import PersonBinding

        b = PersonBinding()
        inst = dict(position_odom=[2, 0, 0], bbox=[0, 0, 100, 200])
        self.assertFalse(
            b.update([inst, inst], dict(t_obs=0, image_id="a"), auto_unique=True)[
                "visible"
            ]
        )
        self.assertTrue(
            b.update([inst], dict(t_obs=1, image_id="b"), auto_unique=True)["visible"]
        )
        uid = b.binding_id
        b.update([inst], dict(t_obs=1.2, image_id="c"), auto_unique=True)
        self.assertFalse(
            b.update([], dict(t_obs=2, image_id="d"), auto_unique=True)["visible"]
        )
        self.assertFalse(
            b.update([inst], dict(t_obs=8, image_id="e"), auto_unique=True)["visible"]
        )
        self.assertEqual(b.binding_id, uid)


if __name__ == "__main__":
    unittest.main()
