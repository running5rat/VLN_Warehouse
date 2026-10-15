import unittest
import numpy as np
from ticvla.warehouse.dataset_tools.replay_video import relative_poses, replay_indices, point_distances
from ticvla.warehouse.dataset_tools.episode_video import active_annotation, align_points


class ReplayVideoTests(unittest.TestCase):
    def test_annotations_switch_at_source_time_and_hold_until_next(self):
        times=[10,12,20]
        self.assertEqual(active_annotation(9,times),-1)
        self.assertEqual(active_annotation(10,times),0)
        self.assertEqual(active_annotation(11.99,times),0)
        self.assertEqual(active_annotation(12,times),1)
        self.assertEqual(active_annotation(30,times),2)

    def test_reference_alignment_preserves_episode_coordinates(self):
        episode=np.eye(4);episode[:2,3]=[10,20]
        source=episode.copy();source[:2,:2]=[[0,-1],[1,0]];source[:2,3]+=[2,3]
        np.testing.assert_allclose(align_points([[0,0],[1,0],[0,1]],source,episode),[[2,3],[2,4],[1,3]])
        self.assertEqual(align_points([],source,episode).shape,(0,2))

    def test_relative_pose_uses_source_frame_and_heading(self):
        source=np.array([[0,-1,0,10],[1,0,0,20],[0,0,1,0],[0,0,0,1]],float)
        moved=source.copy();moved[1,3]+=2
        xy,yaw=relative_poses([source,moved],source)
        np.testing.assert_allclose(xy,[[0,0],[2,0]],atol=1e-10)
        np.testing.assert_allclose(yaw,[0,0],atol=1e-10)
        # A genuine left turn must not become right after changing coordinates.
        moved[:3,:3]=source[:3,:3]@np.array([[0,-1,0],[1,0,0],[0,0,1]])
        _,yaw=relative_poses([source,moved],source)
        self.assertAlmostEqual(yaw[-1],np.pi/2)

    def test_clock_original_speed_and_no_future_pose_extrapolation(self):
        times=np.arange(31)/30
        indices=replay_indices(times,5)
        np.testing.assert_array_equal(indices,[0,6,12,18,24,30])
        np.testing.assert_array_equal(replay_indices([56.5],5),[0])
        self.assertLessEqual(max(replay_indices([0,.11,.39],5)),2)
        for times in ([],[1,0],[0,0],[float('nan')]):
            with self.assertRaises(ValueError):replay_indices(times,5)

    def test_distance_is_to_segments_not_only_discrete_vertices(self):
        line=np.array([[0,0],[2,0],[2,2]],float)
        np.testing.assert_allclose(point_distances(np.array([[1,.2],[2,1],[3,2]]),line),[.2,0,1])
        self.assertEqual(point_distances(np.array([[1,1]]),np.empty((0,2))),[None])

    def test_stationary_poses_do_not_become_synthetic_motion(self):
        pose=np.eye(4);pose[:2,3]=[7,-5]
        xy,yaw=relative_poses([pose]*15,pose)
        np.testing.assert_array_equal(xy,np.zeros((15,2)))
        np.testing.assert_array_equal(yaw,np.zeros(15))
