import unittest
import numpy as np
import torch
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.navigation.flow import FlowMatchingActionExpert, MODES
from ticvla.warehouse.navigation.core import (
    NavigationState,
    PersonBinding,
    check_candidates,
    pursuit,
    select_landmark,
    frontier_path,
)


class NavigationTests(unittest.TestCase):
    def test_flow_nine_modes_reproducible_and_shared_context(self):
        torch.set_num_threads(1)
        cfg = WarehouseConfig(
            hidden_dim=32, num_heads=4, num_layers=1, horizon=6, num_reference_points=8
        )
        m = FlowMatchingActionExpert(12, cfg).eval()
        b = dict(
            visual=torch.randn(1, 4, 12),
            bev=torch.zeros(1, 4, 16, 16),
            state=torch.zeros(1, 2),
            state_mask=torch.ones(1, 2, dtype=torch.bool),
            motion_direction=torch.ones(1),
            reference_xy=torch.zeros(1, 8, 2),
            reference_mask=torch.ones(1, 8, dtype=torch.bool),
            reference_confidence=torch.ones(1),
            z_sem=torch.zeros(1, 16, 32),
            inference_mode=torch.zeros(1, dtype=torch.long),
            delay=torch.zeros(1, 5),
            previous_plan=torch.zeros(1, 6, 3),
            previous_plan_mask=torch.zeros(1, 6, dtype=torch.bool),
            visual_mask=torch.ones(1, 4, dtype=torch.bool),
            map_valid=torch.ones(1, dtype=torch.bool),
            semantic_valid=torch.ones(1, dtype=torch.bool),
        )
        a = m.sample(b, torch.Generator().manual_seed(2))["trajectories"]
        c = m.sample(b, torch.Generator().manual_seed(2))["trajectories"]
        self.assertEqual(a.shape, (1, 9, 1, 6, 2))
        torch.testing.assert_close(a, c)
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(len(MODES), 9)
        # z_sem is consumed directly: no raw KV or compressor in deployment model.
        self.assertFalse(any("compressor" in x for x in m.state_dict()))
        context, valid = m.encode(b)
        x = torch.zeros(2, 6, 2)
        tau = torch.zeros(2)
        out = m.velocity(
            x,
            tau,
            context.repeat(2, 1, 1),
            valid.repeat(2, 1),
            torch.tensor([[0, 0], [2, 2]]),
        )
        self.assertFalse(torch.equal(out[0], out[1]))

    def test_atomic_commit_stale_and_prefill_route_unchanged(self):
        nav = NavigationState("task", "follow")
        obs = dict(
            task={"task_id": "task"},
            route_revision=0,
            source_monotonic=1.0,
            T_odom_body=np.eye(4).tolist(),
            odom_epoch="e",
            map_version="1",
            target_visible=True,
            target_binding_id="p",
        )
        r = dict(
            valid=True,
            inference_mode="TEXT",
            parsed={"route": "Keep following."},
            z_sem=np.zeros((16, 512)),
            reference_xy=np.zeros((32, 2)),
            reference_mask=np.ones(32, bool),
            confidence=0.5,
        )
        original = nav.state.contract
        for bad in (
            dict(r, z_sem=np.zeros((15, 512))),
            dict(r, reference_xy=np.full((32, 2), np.nan)),
            dict(r, reference_mask=np.arange(32) % 2 == 0),
        ):
            with self.assertRaises(ValueError):
                nav.commit(bad, obs, 1, 1.1, 1.2, 1.3, 10)
            self.assertIsNone(nav.snapshot)
            self.assertEqual(nav.state.route_revision, 0)
        nav.commit(r, obs, 1, 1.1, 1.2, 1.3, 10)
        self.assertEqual(nav.state.contract, original)
        self.assertEqual(nav.state.route_revision, 1)
        old = nav.snapshot
        with self.assertRaises(ValueError):
            nav.commit(r, obs, 2, 1.1, 1.2, 1.3, 10)
        self.assertIs(nav.snapshot, old)
        obs["route_revision"] = 1
        r["inference_mode"] = "PREFILL_FOLLOW"
        r["parsed"] = None
        nav.commit(r, obs, 2, 1.1, 1.2, 1.3, 10)
        self.assertEqual(nav.state.route_revision, 1)
        obs["target_visible"] = False
        with self.assertRaises(ValueError):
            nav.commit(r, obs, 3, 1.1, 1.2, 1.3, 10)
        self.assertEqual(nav.last_generation, 2)
        self.assertIsNone(nav.valid_snapshot(20, "e", 3)[0])
        self.assertIsNone(nav.valid_snapshot(2, "other", 3)[0])

    def test_filter_and_shield_do_not_execute_bad_candidate(self):
        grid = np.zeros((160, 160), np.int8)
        p = np.stack((np.arange(1, 31) * 0.02, np.zeros(30)), -1)
        selected, rows = check_candidates(np.tile(p, (9, 1, 1, 1)), grid, p)
        self.assertIsNotNone(selected)
        bad = np.tile(p * 100, (9, 1, 1, 1))
        self.assertIsNone(check_candidates(bad, grid, p)[0])
        occupied = np.ones_like(grid)
        self.assertEqual(pursuit(p, occupied)[0], (0.0, 0.0))
        self.assertIsNone(check_candidates(np.tile(p, (9, 1, 1, 1)), occupied, p)[0])
        self.assertEqual(pursuit([], grid)[0], (0.0, 0.0))

    def test_duplicate_landmarks_and_person_binding(self):
        records = [
            dict(landmark_uid="a", identity_text="01", submap_id="e"),
            dict(landmark_uid="b", identity_text="01", submap_id="e"),
        ]
        self.assertIsNone(select_landmark(records, "01", "e")[0])
        self.assertEqual(select_landmark(records, "a", "e")[0]["landmark_uid"], "a")
        self.assertIsNone(select_landmark(records, "a", "other")[0])
        b = PersonBinding()
        s = dict(t_obs=1, image_id="f1")
        inst = dict(bbox=[0, 0, 50, 100], position_odom=[2, 0, 0])
        self.assertFalse(b.update([inst], s)["visible"])
        self.assertTrue(b.update([inst], s, [0, 0, 50, 100])["visible"])
        uid = b.binding_id
        self.assertFalse(
            b.update([inst, inst], dict(t_obs=1.2, image_id="f2"))["visible"]
        )
        self.assertEqual(b.binding_id, uid)
        self.assertFalse(b.update([], dict(t_obs=2, image_id="f3"))["visible"])

    def test_frontier_requires_observed_free_space(self):
        grid = np.full((160, 160), -1, np.int8)
        grid[40:120, 40:120] = 0
        p = frontier_path(grid, 0.1, 0.35)
        self.assertGreater(len(p), 1)
        self.assertEqual(len(frontier_path(np.full_like(grid, -1), 0.1, 0.35)), 0)


class NavigationGeometryTests(unittest.TestCase):
    def test_vectorized_rays_match_original_and_current_hits_win(self):
        from ticvla.warehouse.geometry import LocalMapBuilder, planar_pose
        from ticvla.warehouse.navigation.geometry import render_local

        cfg = WarehouseConfig(depth_stride=2, map_size_m=8, map_resolution=0.1)
        builder = LocalMapBuilder(cfg)
        rng = np.random.default_rng(42)
        for t in range(3):
            # Camera optical z -> body x, optical x -> -body y.
            extrinsic = np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1], [0, 0, 0, 1.0]]
            )
            builder.add(
                depth=rng.uniform(0.5, 5, (24, 32)),
                K=[[20, 0, 16], [0, 20, 12], [0, 0, 1]],
                T_body_camera=extrinsic,
                T_odom_body=planar_pose(0.1 * t, 0, 0.05 * t),
                t_obs=t,
                odom_epoch="e",
            )
        pose = planar_pose(0.2, 0, 0.1)
        fast = render_local(builder, pose, 2, "e")
        old = builder.render(pose, 2, "e")
        np.testing.assert_array_equal(fast[0], old[0])
        np.testing.assert_allclose(fast[1], old[1])


class NavigationSafetyTests(unittest.TestCase):
    def test_pursuit_uses_curvature_with_speed_feedforward(self):
        grid = np.zeros((160, 160), np.int8)
        path = np.array([[0.2, 0.05], [0.6, 0.15], [1.0, 0.25]])
        (v, w), status = pursuit(path, grid, speed=0.15)
        self.assertEqual(status, "tracking")
        self.assertLessEqual(v, 0.15)
        self.assertAlmostEqual(w, v * 2 * 0.15 / (0.6**2 + 0.15**2))

    def test_timed_plan_alignment_interpolates_and_expires(self):
        from ticvla.warehouse.navigation.core import align_timed_candidates
        from ticvla.warehouse.geometry import planar_pose

        path = np.stack((np.arange(1, 31) * 0.02, np.zeros(30)), -1)[None, None]
        current = planar_pose(0.03, 0, 0)
        aligned = align_timed_candidates(path, np.eye(4), current, 0.15, 0.1)
        np.testing.assert_allclose(aligned[0, 0, :3, 0], [0.02, 0.04, 0.06], atol=1e-7)
        self.assertEqual(
            align_timed_candidates(path, np.eye(4), current, 4, 0.1).shape, (1, 1, 0, 2)
        )
        with self.assertRaises(ValueError):
            align_timed_candidates(path, np.eye(4), current, -1, 0.1)

    def test_explicit_anchor_recovers_transform_and_rejects_missing_geometry(self):
        from ticvla.warehouse.navigation.registration import verify_anchor
        from ticvla.warehouse.geometry import planar_pose, transform_points

        rng = np.random.default_rng(17)
        old = rng.uniform([-3, -2, 0.2], [4, 3, 2.5], (800, 3))
        tf = planar_pose(0.2, -0.1, 0.07)
        result = verify_anchor(old, transform_points(old, tf))
        np.testing.assert_allclose(result["T_current_old"], tf, atol=0.01)
        with self.assertRaises(ValueError):
            verify_anchor(old[:10], old[:10])

    def test_watchdog_expiry_epoch_cancel_and_fresh_observed_free(self):
        import json
        import tempfile
        import time
        from pathlib import Path
        from DynaNav.warehouse_loop.navigation_control import NavigationControl

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "navigation"
            root.mkdir()
            control = NavigationControl(root)

            def issue(**changes):
                now = time.monotonic()
                cmd = dict(
                    v=0.2,
                    w=0.0,
                    odom_epoch="e",
                    created_monotonic=now,
                    valid_until=now + 1,
                    reason="test",
                    backend="model",
                )
                cmd.update(changes)
                (root / "command.json").write_text(json.dumps(cmd))
                return control.command(np.eye(4), "e")

            np.savez(
                root / "safety.npz",
                grid=np.zeros((160, 160), np.int8),
                pose=np.eye(4),
                odom_epoch="e",
                source_monotonic=time.monotonic(),
                resolution=0.1,
            )
            control.last_t -= 0.1
            self.assertGreater(issue()[0], 0)
            self.assertEqual(issue(valid_until=0), (0.0, 0.0))
            self.assertEqual(issue(odom_epoch="old"), (0.0, 0.0))
            (root.parent / "cancel.json").write_text("{}")
            self.assertEqual(issue(), (0.0, 0.0))
            control.close()

    def test_follow_loss_never_changes_identity_without_explicit_reconfirmation(self):
        binding = PersonBinding()
        inst = dict(bbox=[0, 0, 50, 100], position_odom=[2, 0, 0])
        binding.update([inst], dict(t_obs=1, image_id="a"), inst["bbox"])
        binding.update([inst], dict(t_obs=1.2, image_id="b"))
        uid = binding.binding_id
        self.assertFalse(binding.update([], dict(t_obs=2, image_id="c"))["visible"])
        self.assertFalse(binding.update([inst], dict(t_obs=5, image_id="d"))["visible"])
        result = binding.update([inst], dict(t_obs=5.2, image_id="e"), inst["bbox"])
        self.assertTrue(result["visible"])
        self.assertEqual(result["binding_id"], uid)


class GrammarDecodeTests(unittest.TestCase):
    def test_greedy_matches_exhaustive_grammar_argmax(self):
        from lmformatenforcer import StringParser
        from ticvla.warehouse.navigation.grammar_decode import GreedyGrammarProcessor

        class Tokenizer:
            vocab = ["x", "<", "a", ">", "<a", "a>"]

            def decode(self, ids, **kwargs):
                return "".join(self.vocab[i] for i in ids)

        tok = Tokenizer()
        grammar = GreedyGrammarProcessor(tok, StringParser("<a>"), 1)
        ids = torch.tensor([[99]])
        scores = torch.tensor([[10.0, 8.0, 7.0, 6.0, 9.0, 5.0]])
        result = grammar(ids, scores)
        self.assertEqual(int(result.argmax()), 4)
        ids = torch.tensor([[99, 4]])
        result = grammar(ids, scores)
        self.assertEqual(int(result.argmax()), 3)
        self.assertEqual(grammar.prefix, "<a>")
        with self.assertRaises(ValueError):
            grammar(torch.tensor([[99, 0]]), scores)

    def test_greedy_exhaustive_fallback_does_not_drop_constraints(self):
        from lmformatenforcer import StringParser
        from ticvla.warehouse.navigation.grammar_decode import GreedyGrammarProcessor

        class Tokenizer:
            def decode(self, ids, **kwargs):
                return "".join("a" if i == 599 else "x" for i in ids)

        grammar = GreedyGrammarProcessor(Tokenizer(), StringParser("a"), 1)
        result = grammar(torch.tensor([[9]]), -torch.arange(600)[None].float())
        self.assertEqual(int(result.argmax()), 599)
        self.assertEqual(grammar.exhaustive_calls, 1)


if __name__ == "__main__":
    unittest.main()
