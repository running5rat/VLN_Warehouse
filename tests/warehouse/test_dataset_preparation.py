import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ticvla.warehouse.collect import EpisodeRecorder
from ticvla.warehouse.geometry import planar_pose
from ticvla.warehouse.dataset_tools.core import (
    anchors, approve, export_project, extract, read_json, resample_line,
    sample_context, validate_v2, write_json, reference_geometry, finalize_project,
)
from DynaNav.warehouse_loop.annotation import AnnotationSession
from ticvla.warehouse.dataset_tools.teacher import run_teacher, teacher_input
from ticvla.warehouse.dataset_tools.oracle import predict_tracks, score_candidate, classify_mode


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        raw = self.root / "capture/raw"
        r = EpisodeRecorder(raw, dict(layout_family="fixture", episode_id="episode", task={
            "task_id": "t", "instruction": "Go forward, turn right and stop by the extinguisher."}))
        for i in range(5):
            r.append(rgb=np.zeros((12, 16, 3), np.uint8), depth=np.ones((12, 16)),
                     K=[[12, 0, 8], [0, 12, 6], [0, 0, 1]], T_body_camera=np.eye(4),
                     T_odom_body=planar_pose(i * .2), t_obs=i*.2,
                     odom_epoch="e", source_monotonic=100+i*.2)
        r.close()
        self.project = self.root / "annotation"
        extract(raw / "episode.json", self.project, indices=[0, 2, 4])
        # Test geometry is explicit, not dependent on an artificial optical-camera fixture.
        for s in read_json(self.project / "project.json")["samples"]:
            np.savez(self.project / s / "map.npz", grid=np.zeros((160, 160), np.int8), age=np.zeros((160, 160)))
            from ticvla.warehouse.schema import file_sha256
            fixture = read_json(self.project / s / "source.json")
            fixture["observation"]["mapview"]["arrays_sha256"] = file_sha256(self.project / s / "map.npz")
            write_json(self.project / s / "source.json", fixture)

    def tearDown(self):
        self.tmp.cleanup()

    def payload(self, i=0):
        s = sample_context(self.project, i)
        return dict(source_binding=s["provenance"]["source_binding"], parent_revision=s["parent_revision"],
                    annotator="test-fixture-not-human-data", human_reviewed=True,
                    landmark=[], local="unknown", route=dict(remaining_instruction="Continue to the extinguisher.",
                    current_subgoal={"action": "advance_to_corridor_end"}, status="active"),
                    control_points=[[0, 0], [1, 0]], reference_state="valid")

    def test_sequential_route_restart_and_stale_descendants(self):
        with self.assertRaises(ValueError): sample_context(self.project, 1)
        a = approve(self.project, 0, self.payload())
        self.assertEqual(sample_context(self.project, 1)["observation"]["route"], a["parsed"]["route"])
        approve(self.project, 1, self.payload(1))
        approve(self.project, 2, self.payload(2))
        m = export_project(self.project, self.root / "export")
        self.assertEqual(len(m["records"]), 3)
        r = read_json(self.root / "export/00000000.json")
        self.assertNotIn("assistance", r)
        self.assertEqual(r["supervision"]["action_state"], "unlabeled")
        self.assertEqual(r["supervision"]["candidates"], [])
        approve(self.project, 0, self.payload())
        with self.assertRaisesRegex(ValueError, "stale"):
            sample_context(self.project, 2)
        with self.assertRaises(ValueError): export_project(self.project, self.root / "export2")

    def test_reviewed_task_translation_reaches_context_teacher_and_export(self):
        project = read_json(self.project / "project.json")
        original = project["task"]["instruction"]
        translated = "Proceed along the aisle, turn right, and stop by the extinguisher."
        project["task_translation"] = dict(source_instruction=original,
            instruction=translated, language="en", actor="test-fixture")
        project["task"]["instruction"] = translated
        write_json(self.project / "project.json", project)
        source_path = self.project / project["samples"][0] / "source.json"
        raw = source_path.read_bytes()
        sample = sample_context(self.project, 0)
        self.assertEqual(sample["observation"]["task"]["instruction"], translated)
        self.assertEqual(sample["observation"]["route"], translated)
        self.assertEqual(teacher_input(sample)[0]["task"]["instruction"], translated)
        for i in range(3):
            approve(self.project, i, self.payload(i))
        export_project(self.project, self.root / "translated-export")
        record = read_json(self.root / "translated-export/00000001.json")
        self.assertEqual(record["observation"]["task"]["instruction"], translated)
        self.assertEqual(record["provenance"]["task_translation"], project["task_translation"])
        self.assertEqual(record["observation"]["route"], sample_context(self.project, 0)["annotation"]["parsed"]["route"])
        self.assertEqual(source_path.read_bytes(), raw)
        for key, bad in (("source_instruction", "another source"), ("instruction", "another target")):
            broken = copy.deepcopy(project)
            broken["task_translation"][key] = bad
            write_json(self.project / "project.json", broken)
            with self.assertRaisesRegex(ValueError, "Task translation"):
                sample_context(self.project, 0)

    def test_previous_local_is_causal_exported_and_teacher_visible(self):
        self.assertFalse(sample_context(self.project, 0)["observation"]["previous_local"]["valid"])
        data = self.payload()
        data["local"] = "Near a sign"
        approve(self.project, 0, data)
        sample = sample_context(self.project, 1)
        previous = sample["observation"]["previous_local"]
        self.assertTrue(previous["valid"])
        self.assertEqual(previous["text"], "Near a sign")
        self.assertAlmostEqual(previous["current_in_previous_body"]["x_m"], .4)
        self.assertEqual(teacher_input(sample)[0]["previous_local"], previous)
        approve(self.project, 1, self.payload(1))
        self.assertFalse(sample_context(self.project, 2)["observation"]["previous_local"]["valid"])
        approve(self.project, 2, self.payload(2))
        export_project(self.project, self.root / "export")
        exported = read_json(self.root / "export/00000001.json")
        self.assertEqual(exported["observation"]["previous_local"], previous)

    def test_replay_completion_facts_persist_export_and_are_causal(self):
        approve(self.project, 0, self.payload())
        data = self.payload(1)
        data.update(replay_event_kinds=["target_visible", "stopped", "arrival_confirmed"],
                    replay_event_note="Fixture-only human review of target and measured stop.",
                    reference_state="valid_empty", control_points=[])
        data["route"] = dict(remaining_instruction="Arrived and stopped.",
                             current_subgoal={"action":"stop"}, status="completed")
        source = self.project / "samples/00000002/source.json"
        original = source.read_bytes()
        approved = approve(self.project, 1, data)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(len(approved["replay_events"]), 3)
        self.assertEqual(sample_context(self.project, 0)["observation"]["execution"]["events"], [])
        for i in (1, 2):
            events = sample_context(self.project, i)["observation"]["execution"]["events"]
            self.assertEqual(len(events), 3)
            self.assertTrue(all(e["source"] == "human_replay_review" and e["t_obs"] == .4 for e in events))
        # Resubmission replaces this source's facts rather than duplicating them.
        approve(self.project, 1, data)
        self.assertEqual(len(sample_context(self.project, 1)["observation"]["execution"]["events"]), 3)
        approve(self.project, 2, self.payload(2))
        export_project(self.project, self.root / "export")
        self.assertEqual(len(read_json(self.root / "export/00000002.json")["observation"]["execution"]["events"]), 3)
        # User-authorized corrected facts must also be removable in the UI.
        current_path = self.project / "samples/00000002/approved.json"
        current = read_json(current_path)
        for event in current["replay_events"]:
            event["source"] = "user_authorized_replay_correction"
        write_json(current_path, current)
        removed = self.payload(1)
        approve(self.project, 1, removed)
        self.assertEqual(sample_context(self.project, 1)["observation"]["execution"]["events"], [])
        with self.assertRaisesRegex(ValueError, "stale"):
            export_project(self.project, self.root / "export2")

    def test_stop_instruction_needs_no_completion_claim_and_fact_checks(self):
        data = self.payload()
        data["route"] = dict(remaining_instruction="Stop here.", current_subgoal={"action":"stop"}, status="active")
        approve(self.project, 0, data)
        for kinds, note in [(["stopped"], ""), (["invented"], "fixture"), (["stopped", "stopped"], "fixture")]:
            invalid = dict(data, replay_event_kinds=kinds, replay_event_note=note)
            with self.assertRaises(ValueError): approve(self.project, 0, invalid)
        invalid = dict(data, replay_event_kinds=["target_visible"], replay_event_note="Target only")
        invalid["route"] = dict(data["route"], status="completed")
        with self.assertRaisesRegex(ValueError, "Completion"):
            approve(self.project, 0, invalid)
        self.assertEqual(sample_context(self.project, 0)["annotation"]["replay_events"], [])

    def test_finalize_optional_turn_and_start_markers(self):
        from unittest.mock import patch
        # Isolate finalization policy; sample_context's real revision/causality
        # checks have separate integration coverage in this suite.
        root = self.root / "finalization-policy-fixture"
        events = [dict(kind=k, image_id="last") for k in
                  ("target_visible", "stopped", "arrival_confirmed")]
        for turn_count in (0, 1, 2):
            with self.subTest(turn_count=turn_count):
                write_json(root / "project.json", dict(samples=[str(i) for i in range(8)],
                    task={"instruction":"At the corridor end turn left and stop by the extinguisher."}))
                sample = dict(parent_revision="parent", annotation=dict(parent_revision="parent", revision_id="last"),
                    observation=dict(execution=dict(events=events + [dict(kind="turn_completed", image_id=str(i)) for i in range(turn_count)])))
                with patch("ticvla.warehouse.dataset_tools.core.sample_context", return_value=sample) as context:
                    result = finalize_project(root, "test-fixture")
                self.assertEqual(context.call_count, 8)
                self.assertTrue(result["human_episode_approved"])
                self.assertEqual(result["recorded_completed_turns"], turn_count)
                self.assertEqual(read_json(root / "project.json")["finalization_policy"],
                                 "human_task_review_with_completion_evidence_v2")

    def test_finalize_still_rejects_missing_completion_and_stale_approval(self):
        from unittest.mock import patch
        root = self.root / "finalization-rejection-fixture"
        for missing in ("target_visible", "stopped", "arrival_confirmed"):
            write_json(root / "project.json", dict(samples=[str(i) for i in range(8)], human_episode_approved=False))
            sample = dict(parent_revision="p", annotation=dict(parent_revision="p", revision_id="last"),
                observation=dict(execution=dict(events=[dict(kind=k, image_id="last") for k in
                    ("target_visible", "stopped", "arrival_confirmed") if k != missing])))
            with patch("ticvla.warehouse.dataset_tools.core.sample_context", return_value=sample):
                with self.assertRaisesRegex(ValueError, missing): finalize_project(root, "fixture")
            self.assertFalse(read_json(root / "project.json")["human_episode_approved"])
        sample["annotation"]["parent_revision"] = "stale"
        with patch("ticvla.warehouse.dataset_tools.core.sample_context", return_value=sample):
            with self.assertRaisesRegex(ValueError, "stale"): finalize_project(root, "fixture")

    def test_task_acceptance_context_and_correction_provenance_export(self):
        project = read_json(self.project / "project.json")
        criteria = {"arrival": {"max_horizontal_distance_m": 4.0, "reference": "body_origin_to_visible_target_surface"}}
        project["acceptance_criteria"] = criteria
        write_json(self.project / "project.json", project)
        for i in range(3):
            approve(self.project, i, self.payload(i))
        sample = sample_context(self.project, 0)
        self.assertEqual(sample["observation"]["task"]["acceptance_criteria"], criteria)
        self.assertEqual(teacher_input(sample)[0]["task"]["acceptance_criteria"], criteria)
        source = read_json(self.project / sample["folder"] / "source.json")
        self.assertNotIn("acceptance_criteria", source["observation"]["task"])
        path = self.project / sample["folder"] / "approved.json"
        approved = read_json(path)
        approved["correction"] = {"kind":"user_authorized_correction", "actor":"fixture"}
        write_json(path, approved)
        export_project(self.project, self.root / "export")
        exported = read_json(self.root / "export/00000000.json")
        self.assertEqual(exported["provenance"]["annotation_correction"], approved["correction"])
        self.assertEqual(exported["observation"]["task"]["acceptance_criteria"], criteria)

    def test_wrong_source_raw_mutation_and_completion_without_evidence(self):
        d = self.payload()
        d["source_binding"] = "other"
        with self.assertRaises(ValueError): approve(self.project, 0, d)
        d = self.payload()
        d["route"]["status"] = "completed"
        with self.assertRaisesRegex(ValueError, "Completion"):
            approve(self.project, 0, d)
        d = self.payload()
        image = self.root / "capture/raw/rgb/00000000.png"
        image.write_bytes(image.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            approve(self.project, 0, d)

    def test_reference_empty_is_not_unlabeled_or_stop_action(self):
        d = self.payload()
        d.update(control_points=[], reference_state="valid_empty")
        a = approve(self.project, 0, d)
        self.assertEqual(a["waypoint"], [])
        self.assertFalse(any(a["reference_mask"]))
        self.assertEqual(a["action_state"], "unlabeled")
        d["reference_state"] = "valid"
        with self.assertRaises(ValueError): approve(self.project, 0, d)

    def test_endpoint_anchor_and_coordinate_error(self):
        line, mask = resample_line([[0, 0], [0.12, 0]])
        np.testing.assert_allclose(line[mask][-1], [.12, 0], atol=.0001)
        self.assertEqual(len(anchors(line[mask])), 5)
        s = read_json(self.project / "samples/00000002/source.json")
        self.assertAlmostEqual(s["assistance"]["future_trajectory"][-1][0], .4)
        self.assertLess(abs(s["assistance"]["future_trajectory"][-1][1]), .05)

    def test_geometry_checks_segments_not_only_vertices(self):
        grid = np.zeros((160, 160), np.int8)
        grid[70, 80] = 1
        self.assertTrue(reference_geometry([[0, 0], [2, 0]], grid, .1)["collision"])
        grid[70, 80] = -1
        check = reference_geometry([[0, 0], [2, 0]], grid, .1)
        self.assertTrue(check["unknown"])
        self.assertFalse(check["collision"])

    def test_human_unknown_reference_keeps_supervision_and_diagnostics(self):
        from ticvla.warehouse.schema import file_sha256
        folder=self.project/"samples/00000000"
        grid=np.full((160,160),-1,np.int8)
        def replace_fixture_map():
            np.savez(folder/"map.npz",grid=grid,age=np.zeros_like(grid,dtype=float))
            source=read_json(folder/"source.json")
            source["observation"]["mapview"]["arrays_sha256"]=file_sha256(folder/"map.npz")
            write_json(folder/"source.json",source)
        replace_fixture_map()
        result=approve(self.project,0,self.payload())
        self.assertTrue(result["geometry"]["unknown"])
        self.assertGreater(result["geometry"]["footprint_unknown_points"],0)
        self.assertTrue(result["training_eligible"])
        self.assertEqual(result["unknown_review_note"],"")
        np.testing.assert_array_equal(np.load(folder/"map.npz")["grid"],grid)
        approve(self.project,1,self.payload(1));approve(self.project,2,self.payload(2))
        export_project(self.project,self.root/"export-unknown")
        r=read_json(self.root/"export-unknown/00000000.json")
        self.assertTrue(r["supervision"]["reference_loss_valid"])
        self.assertTrue(r["supervision"]["geometry_diagnostics"]["unknown"])
        self.assertEqual(r["supervision"]["action_state"],"unlabeled")
        # Unknown must not hide a known obstacle or disable its rejection.
        grid[75,80]=1;replace_fixture_map()
        with self.assertRaisesRegex(ValueError,"intersects observed"):
            approve(self.project,0,self.payload())

    def test_projection_camera_is_source_bound_preview_only(self):
        from ticvla.warehouse.dataset_tools.core import prepare_projection_cameras
        path=self.project/"samples/00000000/source.json"
        before=path.read_bytes()
        prepare_projection_cameras(self.project)
        s=read_json(path);camera=read_json(path.parent/"projection-camera.json")
        self.assertEqual(camera["image_id"],s["observation"]["image_id"])
        self.assertEqual(camera["source_binding"],s["provenance"]["source_binding"])
        self.assertEqual((camera["width"],camera["height"]),(16,12))
        self.assertTrue(camera["display_only"])
        self.assertFalse(camera["occlusion_checked"])
        self.assertEqual(path.read_bytes(),before)
        image=self.root/"capture/raw/rgb/00000000.png"
        image.write_bytes(image.read_bytes()+b"changed")
        with self.assertRaisesRegex(ValueError,"Projection source differs"):
            prepare_projection_cameras(self.project)

    def test_no_future_nested_map_or_images(self):
        approve(self.project, 0, self.payload())
        approve(self.project, 1, self.payload(1))
        approve(self.project, 2, self.payload(2))
        export_project(self.project, self.root / "export")
        r = read_json(self.root / "export/00000000.json")
        for change in (
            lambda o: o["mapview"].update(source_until=99),
            lambda o: o["execution"].update(oracle={"x": 2}),
            lambda o: o["rgb_history"][0].update(t_obs=99),
        ):
            bad = copy.deepcopy(r)
            change(bad["observation"])
            with self.assertRaises(ValueError): validate_v2(bad)

    def test_manual_deadman_pause_and_replay_cannot_drive(self):
        (self.root / "navigation").mkdir()
        s = AnnotationSession(self.root, "capture")
        s.source = dict(image_id="e:0", t_obs=0, odom_epoch="e")
        self.assertEqual(s.command(dict(v=.3))["v"], 0)
        s.post("/record", {"state": "recording"})
        c = s.command(dict(v=.3))
        self.assertEqual(c["v"], .3)
        self.assertAlmostEqual(c["valid_until"]-c["created_monotonic"], .3)
        s.post("/record", {"state": "paused"})
        self.assertEqual(read_json(self.root / "navigation/command.json")["v"], 0)
        with self.assertRaises(ValueError): s.command(dict(v=-.2))
        with self.assertRaises(ValueError): AnnotationSession(self.root, "replay").command(dict(v=.3))

    def test_teacher_gate_and_causal_payload(self):
        sample = sample_context(self.project, 0)
        context, images = teacher_input(sample)
        self.assertNotIn("assistance", context)
        self.assertNotIn("suggested_reference", json.dumps(context))
        self.assertEqual(len(images), len(sample["observation"]["rgb_history"])+1)
        config = self.root / "teacher.json"
        write_json(config, {"model": "configured-by-user"})
        r = run_teacher(self.project, config, self.root / "teacher-result")
        self.assertEqual(r["actual_api_calls"], 0)
        self.assertEqual(r["status"], "blocked")

    def test_dynamic_prediction_no_future_and_independent_collision(self):
        tr=dict(source="rgbd_tracking", times=[0.,1.], positions=[[1.,1.],[1.,.9]], radius=.2)
        prediction=predict_tracks([tr],1.)
        np.testing.assert_allclose(prediction[0]["xy"][-1],[1.,.6])
        with self.assertRaises(ValueError): predict_tracks([tr],.5)
        bad=copy.deepcopy(tr); bad["source"]="simulation_gt"
        with self.assertRaises(ValueError): predict_tracks([bad],1.)
        x=np.zeros((31,5));x[:,0]=np.arange(31)*.03;x[:,3]=.3
        c=dict(states=x.tolist(), controls=np.zeros((30,2)).tolist(), mode_index=5)
        score=score_candidate(c,[[0,0],[3,0]],[],[],initial_state=x[0])
        self.assertEqual(score["status"],"verified")
        score=score_candidate(c,[[0,0],[3,0]],[[.5,0,.2]],[],initial_state=x[0])
        self.assertIn("static_collision",score["reasons"])
        self.assertEqual(classify_mode(x,[[0,0],[3,0]]),5)

    def test_actual_text_retained_by_training_adapter(self):
        from ticvla.warehouse.dataset_tools.core import training_record
        from ticvla.warehouse.vlm import target_text
        approve(self.project,0,self.payload());approve(self.project,1,self.payload(1));approve(self.project,2,self.payload(2))
        export_project(self.project,self.root/"export")
        r=read_json(self.root/"export/00000000.json")
        adapted=training_record(r)
        self.assertEqual(target_text(adapted["supervision"]),r["supervision"]["text"])
        self.assertFalse(any(adapted["supervision"]["action_mask"]))

    def test_high_rate_source_trace_and_map_mutation(self):
        nav=self.root/"navigation";nav.mkdir()
        rows=[dict(t_obs=t,odom_epoch="e",pose=planar_pose(t).tolist()) for t in np.linspace(0,.8,25)]
        (nav/"executed-commands.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
        project=self.root/"high-rate"
        extract(self.root/"capture/raw/episode.json",project,indices=[0])
        s=sample_context(project,0)
        self.assertEqual(s["assistance"]["trajectory_source"],"executed_command_source_odometry_30hz")
        self.assertGreater(len(s["assistance"]["future_trajectory"]),20)
        d=self.payload()
        (self.project/"samples/00000000/map.npz").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"map changed"):
            approve(self.project,0,d)

    def test_manual_unknown_policy_does_not_bypass_known_obstacles(self):
        import time
        from DynaNav.warehouse_loop.navigation_control import NavigationControl
        nav=self.root/"navigation";nav.mkdir()
        grid=np.full((160,160),-1,np.int8)
        def safety():
            np.savez(nav/"safety.npz",grid=grid,pose=np.eye(4),resolution=.1,odom_epoch="e",source_monotonic=time.monotonic())
        safety()
        now=time.monotonic()
        cmd=dict(backend="manual",safety_policy="human_exploration",v=.2,w=0,reason="human_teleoperation",
                 created_monotonic=now,valid_until=now+10,odom_epoch="e")
        write_json(nav/"command.json",cmd)
        control=NavigationControl(nav);control.last_t-=.1
        self.assertGreater(control.command(np.eye(4),"e",0)[0],0)
        grid[80,80]=1;safety();control.stamp=None
        self.assertEqual(control.command(np.eye(4),"e",.1),(0,0))
        grid[:]=-1;safety();control.stamp=None
        cmd["backend"]="model";write_json(nav/"command.json",cmd)
        self.assertEqual(control.command(np.eye(4),"e",.2),(0,0))
        control.close()

    def test_pure_rotation_is_not_xy_stop_ground_truth(self):
        x=np.zeros((31,5));x[:,2]=np.arange(31)*.05;x[:,4]=.5
        c=dict(states=x.tolist(),controls=np.zeros((30,2)).tolist(),mode_index=3)
        score=score_candidate(c,[[0,0],[3,0]],[],[],initial_state=x[0])
        self.assertIn("pure_rotation_not_expressible_by_xy_action",score["reasons"])


class NineModeTrainingTests(unittest.TestCase):
    def test_gradients_sparse_modes_checkpoint_and_no_update(self):
        import torch
        from ticvla.warehouse.config import WarehouseConfig
        from ticvla.warehouse.dataset_tools.flow_training import FlowTrainingAdapter
        torch.set_num_threads(1)
        cfg=WarehouseConfig(hidden_dim=32,num_heads=4,num_layers=1,horizon=6,num_reference_points=8)
        m=FlowTrainingAdapter(12,16,cfg)
        b=dict(visual=torch.randn(1,4,12),bev=torch.zeros(1,4,16,16),state=torch.zeros(1,2),
               state_mask=torch.ones(1,2,dtype=torch.bool),motion_direction=torch.ones(1),
               reference_xy=torch.zeros(1,8,2),reference_mask=torch.ones(1,8,dtype=torch.bool),
               reference_confidence=torch.ones(1),inference_mode=torch.zeros(1,dtype=torch.long),
               delay=torch.zeros(1,5),previous_plan=torch.zeros(1,6,3),
               previous_plan_mask=torch.zeros(1,6,dtype=torch.bool),visual_mask=torch.ones(1,4,dtype=torch.bool),
               map_valid=torch.ones(1,dtype=torch.bool),semantic_valid=torch.ones(1,dtype=torch.bool))
        kv=torch.randn(1,8,16,requires_grad=True)
        targets=torch.zeros(1,9,1,6,2)
        point=torch.ones(1,9,1,6,dtype=torch.bool)
        valid=torch.zeros(1,9,1,dtype=torch.bool);valid[:,5]=True
        before={k:v.clone() for k,v in m.state_dict().items()}
        loss=m.loss(b,kv,torch.ones(1,8,dtype=torch.bool),targets,point,valid,torch.Generator().manual_seed(9))
        loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertIsNone(kv.grad)
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in m.compressor.parameters()))
        for k,v in m.state_dict().items(): torch.testing.assert_close(v,before[k])
        # Unknown modes do not affect the loss.
        changed=targets.clone();changed[:,0]=999
        other=m.loss(b,kv,torch.ones(1,8,dtype=torch.bool),changed,point,valid,torch.Generator().manual_seed(9))
        # Flow blocks have dropout; compare in evaluation mode to isolate target selection.
        m.eval()
        a=m.loss(b,kv,torch.ones(1,8,dtype=torch.bool),targets,point,valid,torch.Generator().manual_seed(9))
        c=m.loss(b,kv,torch.ones(1,8,dtype=torch.bool),changed,point,valid,torch.Generator().manual_seed(9))
        torch.testing.assert_close(a,c)
        with self.assertRaises(ValueError):
            m.loss(b,kv,torch.ones(1,8,dtype=torch.bool),targets,point,torch.zeros_like(valid))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"interface-fixture.pt"
            m.save(p,{"test_only":True,"optimizer_updates":0})
            loaded=FlowTrainingAdapter.load(p)
            for k,v in m.flow.state_dict().items(): torch.testing.assert_close(v,loaded.flow.state_dict()[k])


if __name__ == "__main__":
    unittest.main()
