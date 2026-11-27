import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image
from DynaNav.warehouse_loop.codex_demo import fit_waypoints, remaining_curve, outside_camera, stopped_evidence
from DynaNav.warehouse_loop.codex_demo_control import response_curve
from DynaNav.warehouse_loop.codex_demo_agent import body_anchors, submit
from ticvla.warehouse.geometry import planar_pose, transform_points


class CodexDemoTests(unittest.TestCase):
    def test_anchor_fit_and_source_alignment(self):
        anchors=np.array([[1,0],[2,0],[3,.2],[4,.8],[4.5,1.5]])
        line=fit_waypoints(anchors)
        np.testing.assert_allclose(line[0],[0,0]);np.testing.assert_allclose(line[-1],anchors[-1])
        self.assertTrue(np.isfinite(line).all())
        for point in anchors:self.assertLess(np.linalg.norm(line-point,axis=1).min(),.06)
        pose=planar_pose(4,2,.7)
        world=transform_points(np.c_[line,np.zeros(len(line))],pose)[:,:2]
        np.testing.assert_allclose(remaining_curve(world,pose),line,atol=1e-8)

    def test_invalid_output_is_not_repaired(self):
        for points in ([],[[0,0]]*5,[[float('nan'),0]]*5,[[50,0]]*5):
            with self.assertRaises(ValueError):fit_waypoints(points)

    def test_rejected_response_removes_previous_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);line=np.array([[0.,0.],[1.,0.]])
            np.savez(p/'curve-test.npz',odom=line)
            np.testing.assert_array_equal(response_curve(p,dict(request_id='test',accepted=True)),line)
            self.assertIsNone(response_curve(p,dict(request_id='test',accepted=False)))
            self.assertIsNone(response_curve(p,dict(request_id='test',accepted=True,action='finish')))

    def test_frustum_uses_source_pose_and_actual_camera(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb=Path(tmp)/'rgb.png';Image.new('RGB',(960,720)).save(rgb)
            source=dict(T_odom_body=np.eye(4).tolist(),T_body_camera=np.eye(4).tolist(),
                        K=[[640,0,480],[0,640,360],[0,0,1]],rgb=str(rgb))
            self.assertFalse(outside_camera([0,0,5],source))
            self.assertTrue(outside_camera([0,0,-5],source))
            self.assertTrue(outside_camera([10,0,5],source))

    def test_completion_requires_observed_stop(self):
        history={str(i):dict(t_obs=i*.2,T_odom_body=planar_pose(0,0,0).tolist()) for i in range(8)}
        control=dict(reason='waypoint_endpoint_reached')
        self.assertEqual(stopped_evidence(history,control)['speed_mps'],0)
        history['7']['T_odom_body']=planar_pose(.2,0,0).tolist()
        with self.assertRaises(ValueError):stopped_evidence(history,control)
        with self.assertRaises(ValueError):stopped_evidence(history,dict(reason='tracking'))

    def test_long_horizon_anchors_remain_source_bound(self):
        p=planar_pose(21,-3.2,-np.pi/2)
        world=np.array([[21,-5],[20.3,-5],[10,-5],[.7,-5],[0,-5]])
        local=body_anchors(world,dict(T_odom_body=p.tolist()))
        rebuilt=transform_points(np.c_[local,np.zeros(5)],p)
        np.testing.assert_allclose(rebuilt[:,:2],world,atol=1e-8)
        self.assertGreater(np.linalg.norm(np.diff(fit_waypoints(local),axis=0),axis=1).sum(),20)

    def test_recursive_route_and_new_task_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            nav=Path(tmp)/'navigation';nav.mkdir()
            source=dict(image_id='frame')
            prior=dict(request_id='old',task_id='mapping',accepted=True,route='Turn right, return and STOP.')
            (nav/'last-response.json').write_text(json.dumps(prior))
            submit(tmp,dict(request_id='next',task_id='mapping',instruction='Whole task',route='Return and STOP.'),source=source)
            inp=json.loads((nav/'requests/next/input.json').read_text())
            self.assertEqual(inp['previous_route'],prior['route'])
            submit(tmp,dict(request_id='target',task_id='target',instruction='New task',initial_route='Turn left, approach sign and STOP.',route='Turn left, approach sign and STOP.'),source=source)
            inp=json.loads((nav/'requests/target/input.json').read_text())
            self.assertEqual(inp['previous_route'],'Turn left, approach sign and STOP.')


if __name__=='__main__':unittest.main()
