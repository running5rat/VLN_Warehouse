import unittest
import numpy as np
from ticvla.warehouse.dataset_tools.reference_schedule import select_updates,reference_in_source,audit_coverage

class ScheduleTests(unittest.TestCase):
    def test_long_drive_refreshes_before_reference_end(self):
        times=np.arange(101)*.2;poses=np.repeat(np.eye(4)[None],len(times),axis=0);poses[:,0,3]=times
        updates=select_updates(times,poses,max_gap_s=50)
        self.assertLess(updates[1]['t_obs'],7.75)
        lines=[]
        for u in updates:
            line,mask=reference_in_source(poses,u['index']);lines.append(line[mask])
        report=audit_coverage(times,poses,updates,lines)
        self.assertEqual(report['exhausted_intervals'],0)
    def test_stationary_tail_does_not_refresh_every_frame(self):
        times=np.arange(101)*.2;poses=np.repeat(np.eye(4)[None],len(times),axis=0)
        updates=select_updates(times,poses)
        self.assertEqual([u['index'] for u in updates],[0,25,50,75,100])
    def test_turn_and_mandatory_frames_preserved(self):
        times=np.arange(21)*.2;poses=np.repeat(np.eye(4)[None],len(times),axis=0)
        poses[5:,:2,:2]=[[0,-1],[1,0]]
        updates=select_updates(times,poses,mandatory=[8])
        self.assertIn(5,[u['index'] for u in updates]);self.assertIn(8,[u['index'] for u in updates])
    def test_expired_reference_detected(self):
        poses=np.repeat(np.eye(4)[None],3,axis=0);poses[:,0,3]=[0,1,3]
        report=audit_coverage([0,1,2],poses,[{'index':0}],[np.array([[0,0],[2,0]])])
        self.assertEqual(report['exhausted_intervals'],1)
