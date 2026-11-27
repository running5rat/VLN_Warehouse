import unittest
import numpy as np
from ticvla.warehouse.dataset_tools.map_memory_review import project_visibility,choose_target,validate_answer,build_request
from ticvla.warehouse.schema import fingerprint

class MapMemoryReviewTests(unittest.TestCase):
    def setUp(self):
        self.frame=dict(T_odom_body=np.eye(4).tolist(),T_body_camera=np.eye(4).tolist(),K=[[10,0,5],[0,10,5],[0,0,1]])
        self.points=np.tile([0.,0.,5.],(20,1))
    def test_visibility_is_not_detector_absence(self):
        self.assertTrue(project_visibility(self.points,self.frame,np.ones((10,10))*2)['nonvisible_evidence'])
        self.assertFalse(project_visibility(self.points,self.frame,np.ones((10,10))*5)['nonvisible_evidence'])
        self.assertFalse(project_visibility(self.points,self.frame,np.zeros((10,10)))['nonvisible_evidence'])
        outside=self.points+np.array([20,0,0]);self.assertTrue(project_visibility(outside,self.frame,np.zeros((10,10)))['nonvisible_evidence'])
    def test_target_requires_unique_named_nonvisible_marker(self):
        row=dict(landmark_uid='a',category='sign',identity_text='K4',identity_status='identity_confirmed',
                 landmark_view=dict(decision='keep'),resolved_identity=dict(full_support_frames=['1','2','3']))
        vis={'a':dict(nonvisible_evidence=True)}
        self.assertEqual(choose_target([row],vis)['landmark_uid'],'a')
        with self.assertRaises(ValueError):choose_target([{**row,'duplicate_text':True}],vis)
        with self.assertRaises(ValueError):choose_target([row],{'a':dict(nonvisible_evidence=False)})
    def test_new_task_completion_and_format(self):
        self.assertEqual(validate_answer(dict(local='unknown',route='DONE'))['rule_flags'],['completion_without_execution'])
        with self.assertRaises(ValueError):validate_answer(dict(local='a'*81,route='Approach sign K4.'))
        with self.assertRaises(ValueError):validate_answer(dict(local='unknown',route='Continue.',waypoint=[]))
        self.assertTrue(validate_answer(dict(local='Inferred: At the mapped aisle entrance.',route='Use the observed passage to reach sign K4.'))['format_valid'])

    def test_frozen_input_cannot_change_between_providers(self):
        value=dict(instructions='Review',context={'instruction':'Go to K4'},images=[])
        value['input_sha256']=fingerprint(value)
        cfg=dict(model='test',max_output_tokens=100)
        self.assertEqual(build_request(value,cfg)['messages'][0]['content'],'Review')
        value['context']['instruction']='Different task'
        with self.assertRaises(ValueError):build_request(value,cfg)

if __name__=='__main__':unittest.main()
