import copy
import unittest
import numpy as np
from ticvla.warehouse.dataset_tools.stage1a_data import habitat_body_pose, route_group
from ticvla.warehouse.geometry import project_depth
from ticvla.warehouse.dataset_tools.reference_schedule import select_updates, audit_coverage, reference_in_source


class Stage1aDataTests(unittest.TestCase):
    def test_habitat_axes_and_source_projection(self):
        first=habitat_body_pose([2,0,3],np.eye(3))
        forward=habitat_body_pose([2,0,2],np.eye(3))
        np.testing.assert_allclose((np.linalg.inv(first)@forward)[:3,3],[1,0,0],atol=1e-8)
        left=habitat_body_pose([1,0,3],np.eye(3))
        np.testing.assert_allclose((np.linalg.inv(first)@left)[:3,3],[0,1,0],atol=1e-8)
        cam=np.eye(4);cam[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];cam[2,3]=1.25
        pts=project_depth(np.array([[2.]]),np.eye(3),cam)
        np.testing.assert_allclose(pts,[[2,0,1.25]],atol=.05)
        np.testing.assert_allclose(first[:3,:3].T@first[:3,:3],np.eye(3))

    def test_alternative_instructions_group_together(self):
        e=dict(scene_id='mp3d/s/s.glb',reference_path=[[0,0,0],[1,0,0]],instruction='left')
        f=copy.deepcopy(e);f['instruction']='turn around';f['episode_id']=999
        self.assertEqual(route_group(e),route_group(f))
        f['scene_id']='mp3d/t/t.glb'
        self.assertNotEqual(route_group(e),route_group(f))

    def test_turn_then_forward_schedule_preserves_coverage(self):
        poses=[]
        from ticvla.warehouse.geometry import planar_pose
        for i in range(13):poses.append(planar_pose(yaw=i*np.pi/12))
        for i in range(1,65):poses.append(planar_pose(x=-i*.25,yaw=np.pi))
        poses=np.array(poses);times=np.arange(len(poses))*.2
        updates=select_updates(times,poses)
        lines=[]
        for u in updates:
            xy,mask=reference_in_source(poses,u['index']);lines.append(xy[mask])
        report=audit_coverage(times,poses,updates,lines)
        self.assertEqual(report['exhausted_intervals'],0)
        self.assertEqual(report['empty_reference_motion_intervals'],0)
        self.assertTrue(any(u['reason']=='heading_change' for u in updates))


if __name__=='__main__':unittest.main()
