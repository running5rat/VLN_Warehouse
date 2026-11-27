import json
from pathlib import Path
import tempfile
import unittest

from ticvla.warehouse.dataset_tools.core import write_json, read_json
from ticvla.warehouse.dataset_tools.r2r_full_prepare import replay_key, clone_capture
from ticvla.warehouse.dataset_tools.stage1a_data import source_config


class FullR2RTests(unittest.TestCase):
    def episode(self, uid=1):
        return dict(episode_id=uid,scene_id='mp3d/a/a.glb',start_position=[0,0,0],
                    start_rotation=[0,0,0,1],instruction=dict(instruction_text=f'Task {uid}'))

    def test_equivalence_requires_pose_actions_and_locations(self):
        e=self.episode();t=dict(actions=[1,0],locations=[[0,0,0],[0,0,-.25]])
        self.assertEqual(replay_key(e,t),replay_key(self.episode(2),t))
        self.assertNotEqual(replay_key(e,t),replay_key({**e,'start_rotation':[0,1,0,0]},t))
        self.assertNotEqual(replay_key(e,t),replay_key(e,{**t,'actions':[2,0]}))
        self.assertNotEqual(replay_key(e,t),replay_key(e,{**t,'locations':[[0,0,0]]}))

    def test_pixels_shared_metadata_isolated_no_supervision_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'old';dst=Path(tmp)/'new';raw=src/'capture/raw'
            for name in ['rgb','depth','supervision']:
                (raw/name).mkdir(parents=True)
                (raw/name/'0.bin').write_bytes(b'fixture')
            e=self.episode();other=self.episode(2);t=dict(actions=[0],locations=[[0,0,0]])
            write_json(src/'replay-report.json',dict(passed=True,uid='r2r-1'))
            write_json(src/'verification-only.json',dict(original_episode=e,original_gt=t))
            write_json(raw/'episode.json',dict(episode_id='r2r-1',task={'instruction':'old'},sensor_config=source_config('r2r')))
            (raw/'frames.jsonl').write_text(json.dumps({'odom_epoch':'r2r-1','t_obs':0})+'\n')
            row=dict(uid='r2r-2',source_dataset='source',replay_group=replay_key(other,t),source_frames=1,role='train')
            clone_capture(src,dst,row,other,t)
            self.assertEqual((raw/'rgb/0.bin').stat().st_ino,(dst/'capture/raw/rgb/0.bin').stat().st_ino)
            self.assertEqual(read_json(dst/'capture/raw/episode.json')['task']['instruction'],'Task 2')
            self.assertEqual(json.loads((dst/'capture/raw/frames.jsonl').read_text())['odom_epoch'],'r2r-2')
            self.assertEqual(json.loads((raw/'frames.jsonl').read_text())['odom_epoch'],'r2r-1')
            self.assertFalse(list((dst/'capture/raw/supervision').iterdir()))
            self.assertEqual(read_json(dst/'verification-only.json')['original_episode']['episode_id'],2)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                clone_capture(src,Path(tmp)/'bad',row,{**other,'start_position':[9,0,0]},t)


if __name__=='__main__':unittest.main()
