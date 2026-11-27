import json
from pathlib import Path
import tempfile
import unittest
from ticvla.warehouse.dataset_tools.annotation_batch_summary import summarize


class AnnotationBatchSummaryTests(unittest.TestCase):
    def test_keeps_unreturned_frames_and_partial_tasks_in_denominator(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def save(p,x):
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x))
            e=root/'episodes/e';other=root/'episodes/other'
            save(root/'pilot.json',{'episodes':[dict(uid='e',scene_id='building',folder=str(e),instruction='Go and stop.',frames=3,source_indices=[0,1,2]),dict(uid='other',scene_id='building',folder=str(other),instruction='Go.',frames=1,source_indices=[0])]})
            save(root/'teacher-config.json',dict(teacher_instruction='route rule'))
            save(e/'plus/report.json',dict(status='running',samples=[dict(source_image_id='e:00000000',folder='0',request_sent=True,status='format_valid_unreviewed'),dict(source_image_id='e:00000001',folder='1',request_sent=True,status='failed',reason='Local too long')]))
            for folder,route in [('0','Stop.'),('1','Turn right.')]:
                save(e/f'plus/{folder}/response.json',{})
                save(e/f'plus/{folder}/raw.txt',dict(landmark=[],local='unknown',route=route))
            save(e/'plus/0/input.json',dict(context=dict(route='Go and stop.'),instructions='route rule',user_text='Previous Route: Go and stop.'))
            save(e/'plus/1/input.json',dict(context=dict(route='Stop.'),instructions='route rule',user_text='Previous Route: Stop.'))
            result=summarize(root,'running',deep=True);s=result['summary']
            self.assertEqual((s['planned_tasks'],s['planned_frames'],s['returned'],s['rule_valid']),(2,4,2,1))
            self.assertEqual(s['tasks_all_frames_rule_valid'],0)
            self.assertEqual(s['tasks_partial_rule_valid'],1)
            self.assertEqual(s['tasks_not_started'],1)
            self.assertEqual(s['frame_rule_valid_rate'],.25)
            self.assertEqual(s['input_contract_errors'],0)
            self.assertEqual(s['heuristic_flags']['STOP_WAIT_TO_NEW_COMMAND_REVIEW'],1)
            self.assertIsNone(s['semantic_success_tasks'])
            self.assertEqual(len(json.loads((root/'frame-statistics.json').read_text())),4)

if __name__=='__main__':unittest.main()
