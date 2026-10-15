import copy
import unittest
import numpy as np
from ticvla.warehouse.geometry import planar_pose
from ticvla.warehouse.local_context import local_memory, local_context
from ticvla.warehouse.navigation.core import NavigationState


class LocalContextTests(unittest.TestCase):
    def observation(self, t=1, x=0, y=0, yaw=0):
        return dict(image_id=f'f{t}', t_obs=t, odom_epoch='e',
                    T_odom_body=planar_pose(x, y, yaw).tolist())

    def test_transform_direction_rotated_origin_and_inverse(self):
        old = self.observation(x=10, y=4, yaw=np.pi/2)
        mem = local_memory('Near sign', old, origin='model', revision=1)
        current = self.observation(t=2, x=10, y=6, yaw=np.pi)
        result = local_context(mem, current)
        self.assertTrue(result['valid'])
        motion = result['current_in_previous_body']
        np.testing.assert_allclose([motion['x_m'], motion['y_m'], motion['yaw_rad']],
                                   [2, 0, np.pi/2], atol=1e-10)
        relative = planar_pose(motion['x_m'], motion['y_m'], motion['yaw_rad'])
        np.testing.assert_allclose(relative @ np.linalg.inv(relative), np.eye(4), atol=1e-10)
        self.assertEqual(result['image_id'], 'f1')

    def test_prompt_really_contains_local_and_motion(self):
        from types import SimpleNamespace
        from ticvla.warehouse.vlm import InternVLReference
        class Conversation:
            roles = ["user", "assistant"]
            def __init__(self):
                self.messages = []
            def append_message(self, role, message):
                self.messages.append(message or "")
            def get_prompt(self):
                return self.system_message + "\n" + "\n".join(self.messages)
        model = SimpleNamespace(backbone=SimpleNamespace(conv_template=Conversation(), num_image_token=1))
        old = self.observation()
        obs = dict(self.observation(2, x=2), rgb_history=[], map_valid=False,
                   task={"instruction":"Go ahead", "acceptance_criteria":{"arrival_radius_m":4.0}}, route="Go ahead",
                   robot_state=[0,0], robot_state_mask=[True,True])
        obs["previous_local"] = local_context(local_memory("Near sign", old, origin="model", revision=1), obs)
        prompt, images = InternVLReference.build_prompt(model, obs)
        self.assertIn('"previous_local"', prompt)
        self.assertIn('"x_m": 2.0', prompt)
        self.assertIn('Near sign', prompt)
        self.assertIn('"task_acceptance_criteria": {"arrival_radius_m": 4.0}', prompt)
        self.assertIn('CURRENT body pose in the PREVIOUS', prompt)
        self.assertEqual(images, [])

    def test_unknown_epoch_expiry_future_invalid_pose(self):
        old = self.observation()
        self.assertIsNone(local_memory('unknown', old, origin='model', revision=1))
        mem = local_memory('Near sign', old, origin='model', revision=1)
        for current, reason in [
            (dict(self.observation(2), odom_epoch='new'), 'odom_epoch_changed'),
            (self.observation(32), 'expired'),
            (self.observation(1), 'non_past_source'),
            (self.observation(0), 'non_past_source'),
            (dict(self.observation(2), T_odom_body=np.zeros((4,4))), 'invalid_pose')]:
            result = local_context(mem, current)
            self.assertFalse(result['valid'])
            self.assertEqual(result['reason'], reason)
            self.assertNotIn('current_in_previous_body', result)

    def test_commit_is_atomic_and_checkpoint_retains_source(self):
        nav = NavigationState('task', 'explore')
        obs = dict(self.observation(), task={'task_id':'task'}, route_revision=0,
                   source_monotonic=10., map_version='v1')
        result = dict(valid=True, inference_mode='TEXT',
                      parsed=dict(local='Near sign', route='Explore', anchors=[]),
                      z_sem=np.zeros((16,512)), reference_xy=np.zeros((32,2)),
                      reference_mask=np.zeros(32,bool), confidence=.5)
        nav.commit(result, obs, 1, 10.1, 10.2, 10.3, 10.)
        self.assertEqual(nav.previous_local['t_obs'], 1)
        self.assertEqual(nav.local_input(self.observation(2,x=1))['current_in_previous_body']['x_m'], 1)
        before = copy.deepcopy(nav.previous_local)
        result['parsed']['local'] = 'Bad update'
        with self.assertRaises(ValueError):
            nav.commit(result, obs, 2, 10.1, 10.2, 10.3, 10.)
        self.assertEqual(nav.previous_local, before)
        restored = NavigationState.restore(nav.checkpoint())
        self.assertEqual(restored.previous_local, before)
        result['parsed']['local'] = 'unknown'
        restored.commit(result, dict(obs, route_revision=1), 2, 10.1, 10.2, 10.3, 10.)
        self.assertIsNone(restored.previous_local)
