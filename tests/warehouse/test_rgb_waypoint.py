import numpy as np
import unittest
import tempfile
from pathlib import Path
from PIL import Image
from ticvla.warehouse.navigation.rgb_waypoint import project_waypoints, PeriodicLatestRequest
from ticvla.warehouse.schema import file_sha256


def fixture(tmp_path):
    K=np.array([[100,0,100],[0,100,60],[0,0,1.]])
    T=np.array([[0,0,1,0],[-1,0,0,0],[0,-1,0,1],[0,0,0,1.]])
    d=np.full((120,200),np.nan)
    for v in range(61,120):d[v,:]=100/(v-60)
    Image.new('RGB',(200,120)).save(tmp_path/'rgb.png');np.save(tmp_path/'depth.npy',d)
    src=dict(rgb=str(tmp_path/'rgb.png'),depth=str(tmp_path/'depth.npy'),K=K.tolist(),
             T_body_camera=T.tolist(),T_odom_body=np.eye(4).tolist())
    src.update(rgb_hash=file_sha256(src['rgb']),depth_hash=file_sha256(src['depth']))
    return src


class RGBWaypointTests(unittest.TestCase):
    def test_depth_projection_and_no_metric_reinterpretation(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=fixture(Path(tmp));p=project_waypoints([[100,110],[100,100],[100,90],[100,80],[100,70]],s)
            np.testing.assert_allclose(np.asarray(p['points_body'])[:,0],[2,2.5,10/3,5,10],atol=.05)
            np.testing.assert_allclose(np.asarray(p['points_body'])[:,2],0,atol=.001)
            with self.assertRaisesRegex(ValueError,'OUTSIDE'):project_waypoints([[200,100]]*5,s)
            with self.assertRaisesRegex(ValueError,'DEPTH'):project_waypoints([[100,30]]*5,s)
            self.assertEqual(project_waypoints([],s)['status'],'empty')


    def test_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=fixture(Path(tmp));s['depth_hash']='wrong'
            with self.assertRaisesRegex(ValueError,'SOURCE_HASH'):project_waypoints([],s)

    def test_visible_anchors_do_not_authorize_occluded_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=fixture(Path(tmp));d=np.load(s['depth']);d[104:107,99:102]=1.0
            np.save(s['depth'],d);s['depth_hash']=file_sha256(s['depth'])
            with self.assertRaisesRegex(ValueError,'OCCLUDED'):
                project_waypoints([[100,110],[100,100],[100,90],[100,80],[100,70]],s)


    def test_periodic_replaces_pending_not_wait_for_endpoint(self):
        q=PeriodicLatestRequest(2)
        q.offer(0,dict(image_id='0'));self.assertEqual(q.take()['image_id'],'0')
        q.offer(2,dict(image_id='2'));q.offer(4,dict(image_id='4'))
        self.assertEqual(q.replaced,1);self.assertIsNone(q.take())
        with self.assertRaises(ValueError):q.finish('4')
        q.finish('0');self.assertEqual(q.take()['image_id'],'4')

    def test_empty_path_and_curve_replacement(self):
        from DynaNav.warehouse_loop.codex_demo import remaining_curve
        from DynaNav.warehouse_loop.codex_demo_control import response_curve
        self.assertEqual(remaining_curve(np.empty((0,2)),np.eye(4)).shape,(0,2))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);s=fixture(root)
            p=project_waypoints([[100,110],[100,100],[100,90],[100,80],[100,70]],s)
            np.savez(root/'curve-new.npz',odom=p['curve_odom'])
            msg=dict(request_id='new',accepted=True,waypoint_protocol='rgb-pixel-waypoint-v1')
            np.testing.assert_allclose(response_curve(root,msg),p['curve_odom'])
            self.assertIsNone(response_curve(root,{**msg,'accepted':False}))
