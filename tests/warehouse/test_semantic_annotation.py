import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from ticvla.warehouse.dataset_tools.core import write_json, read_json
from ticvla.warehouse.dataset_tools.semantic_annotation import (
    write_bundle, validate_sample, build_request, assemble_answer, agreement, run_pilot)
from ticvla.warehouse.dataset_tools.dynanav_semantic import prepare_dynanav, R_USD_TO_FLU
from ticvla.warehouse.schema import file_sha256
from ticvla.warehouse.dataset_tools.session_teacher import start_session, prepare_next, submit_answer


class SemanticTeacherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.image=self.root/'rgb.jpg';Image.new('RGB',(16,12)).save(self.image)
        self.sample=dict(observation=dict(task=dict(task_id='t',instruction='Turn left and approach the extinguisher.'),
            image_id='e:0',t_obs=0.,odom_epoch='e',T_odom_body=np.eye(4).tolist(),
            rgb_history=[dict(image_id='e:0',t_obs=0.,path=str(self.image),sha256=file_sha256(self.image),T_odom_body=np.eye(4).tolist())],
            route='GOLD PREVIOUS ROUTE MUST NOT LEAK',route_revision=0,robot_state=[0.,0.],robot_state_mask=[True,True],
            execution=dict(events=[]),mapview=dict(image=None),previous_local=dict(valid=False,reason='none')),
            provenance=dict(source_binding='fixture-only'),
            geometric_supervision=dict(reference_xy=[[0,0]]*32,reference_mask=[False]*32,
                waypoint=[],reference_state='valid_empty',approved=True,source='fixture_approved_reference'),
            evaluation_only=dict(human_text=dict(landmark=[],local='HUMAN ANSWER MUST NOT LEAK',
                route=dict(remaining_instruction='Human wording.',current_subgoal=dict(action='turn',direction='left'),status='active'))))
        self.config=self.root/'config.json';write_json(self.config,dict(model='fixture-not-a-real-model',api_key_env='TEST_SEMANTIC_KEY'))

    def tearDown(self):self.tmp.cleanup()

    def answer(self):return dict(landmark=[dict(image_id='e:0',category='sign')],local='Near a sign.',
        route=dict(remaining_instruction='Turn left, then approach the extinguisher.',current_subgoal=dict(action='turn',direction='left'),status='active'))

    def raw(self,answer):return dict(status='completed',model='fixture-only',output=[dict(type='message',content=[dict(type='output_text',text=json.dumps(answer))])])

    def sequence(self):
        second=copy.deepcopy(self.sample);second['observation'].update(image_id='e:1',t_obs=1.)
        second['observation']['rgb_history'][0].update(image_id='e:1',t_obs=1.)
        return [self.sample,second]

    def test_dense_review_requires_explicit_scope_and_remains_unapproved(self):
        samples=[]
        for i in range(7):
            s=copy.deepcopy(self.sample)
            s['observation'].update(image_id=f'e:{i}',t_obs=float(i))
            s['observation']['rgb_history'][0].update(image_id=f'e:{i}',t_obs=float(i))
            samples.append(s)
        with self.assertRaises(ValueError):
            write_bundle(self.root/'too-large',samples,source='fixture')
        write_bundle(self.root/'dense',samples,source='fixture',episode_review=True)
        start_session(self.root/'dense',self.root/'session')
        report=read_json(self.root/'session/report.json')
        self.assertEqual(report['expected_samples'],7)
        self.assertFalse(report['human_approved'])

    def test_no_geometry_or_human_labels_and_correct_jpeg_mime(self):
        request,summary=build_request(self.sample,read_json(self.config))
        text=json.dumps(request)
        self.assertNotIn('HUMAN ANSWER MUST NOT LEAK',text)
        self.assertNotIn('reference_xy',text);self.assertNotIn('reference_mask',text)
        self.assertIn('data:image/jpeg;base64,',text)
        self.assertIn('unavailable',summary['context']['map_interface'])

    def test_source_future_and_file_rejections(self):
        for mutate in (
            lambda s:s['observation']['rgb_history'][0].update(t_obs=1),
            lambda s:s['observation']['execution'].update(events=[dict(kind='stopped',t_obs=2)]),
            lambda s:s['observation']['mapview'].update(source_until=1),
            lambda s:s['observation']['execution'].update(future=[[1,2]]),
            lambda s:s['observation']['rgb_history'][0].update(sha256='bad')):
            s=copy.deepcopy(self.sample);mutate(s)
            with self.assertRaises(ValueError):validate_sample(s)

    def test_field_source_and_grammar_validation(self):
        a=self.answer();proposal=assemble_answer(a,self.sample)
        self.assertFalse(proposal['approved']);self.assertEqual(proposal['geometry'],self.sample['geometric_supervision'])
        self.assertEqual(proposal['field_sources']['waypoint'],'fixture_approved_reference')
        for mutate in (
            lambda a:a.update(waypoint=[]),lambda a:a.update(local='中文'),
            lambda a:a['landmark'][0].update(image_id='other'),lambda a:a.update(local='')):
            bad=copy.deepcopy(a);mutate(bad)
            with self.assertRaises(ValueError):assemble_answer(bad,self.sample)

    def test_agreement_is_not_semantic_accuracy_and_completion_needs_evidence(self):
        a=self.answer();r=agreement(a,self.sample)
        self.assertIsNone(r['semantic_accuracy']);self.assertTrue(r['route_action_exact'])
        self.assertFalse(r['remaining_instruction_exact'])
        a['route']['status']='completed'
        self.assertTrue(agreement(a,self.sample)['rule_flags'])
        s=copy.deepcopy(self.sample);s['evaluation_only']={}
        self.assertFalse(agreement(a,s)['human_reference_available'])

    def test_recursive_uses_teacher_route_not_gold_and_no_approval(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        calls=[]
        def send(request,cfg):
            ctx=json.loads(request['input'][0]['content'][0]['text']);calls.append(ctx)
            answer=self.answer();answer['landmark'][0]['image_id']=ctx['source_rgb'][-1]['image_id']
            return self.raw(answer)
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture'}),patch('ticvla.warehouse.dataset_tools.semantic_annotation.send_request',side_effect=send):
            r=run_pilot(bundle,self.config,self.root/'result')
        self.assertEqual(r['actual_api_calls'],2)
        self.assertEqual(calls[0]['route'],self.sample['observation']['task']['instruction'])
        self.assertEqual(calls[1]['route'],self.answer()['route'])
        self.assertEqual(calls[1]['previous_local']['text'],'Near a sign.')
        self.assertNotIn('GOLD PREVIOUS',json.dumps(calls))
        self.assertFalse(r['human_approved']);self.assertIsNone(r['semantic_accuracy'])

    def test_failed_response_stops_recursive_chain_no_repair(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        bad=self.raw({'landmark':[]})
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture'}),patch('ticvla.warehouse.dataset_tools.semantic_annotation.send_request',return_value=bad) as send:
            r=run_pilot(bundle,self.config,self.root/'result')
        self.assertEqual(send.call_count,1);self.assertEqual(r['actual_api_calls'],1)
        self.assertEqual(r['samples'][1]['status'],'not_run_previous_proposal_failed')
        self.assertTrue((self.root/'result/00000000/raw.txt').exists())
        self.assertFalse(r['real_teacher_interface_pass'])

    def test_missing_key_prepares_without_call_or_fake_proposal(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        with patch.dict(os.environ,{},clear=True),patch('ticvla.warehouse.dataset_tools.semantic_annotation.send_request') as send:
            r=run_pilot(bundle,self.config,self.root/'result')
        send.assert_not_called();self.assertEqual(r['actual_api_calls'],0)
        self.assertFalse((self.root/'result/00000000/proposal.json').exists())
        self.assertFalse(read_json(self.root/'result/00000001/input.json')['request_ready'])
        self.assertFalse(r['real_teacher_interface_pass'])

    def session_reply(self, output, index, answer):
        inp = read_json(output/f'{index:08d}'/'input.json')
        path = self.root/f'reply-{index}.json'
        write_json(path, dict(input_sha256=inp['input_sha256'], source_binding=inp['source_binding'], answer=answer))
        return path

    def test_session_recursive_no_api_no_gold_and_immutable_geometry(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        output=self.root/'session'
        with patch('ticvla.warehouse.dataset_tools.semantic_annotation.send_request') as network:
            start_session(bundle,output)
            first=read_json(output/'00000000/input.json')
            self.assertNotIn('HUMAN ANSWER',json.dumps(first))
            self.assertNotIn('GOLD PREVIOUS',json.dumps(first))
            self.assertNotIn('reference_xy',json.dumps(first))
            self.assertFalse((output/'00000001').exists())
            submit_answer(output,self.session_reply(output,0,self.answer()))
            self.assertFalse((output/'00000000/comparison.json').exists())
            prepare_next(output)
            second=read_json(output/'00000001/input.json')
            self.assertEqual(second['context']['route'],self.answer()['route'])
            self.assertEqual(second['context']['previous_local']['text'],'Near a sign.')
            answer=self.answer();answer['landmark'][0]['image_id']='e:1'
            submit_answer(output,self.session_reply(output,1,answer))
            network.assert_not_called()
        report=read_json(output/'report.json')
        self.assertEqual(report['rule_pass_count'],2);self.assertEqual(report['actual_api_calls'],0)
        self.assertFalse(report['independent_blind_evaluation']);self.assertFalse(report['human_approved'])
        self.assertIsNone(report['usage']);self.assertFalse(report['api_interface_verified'])
        proposal=read_json(output/'00000001/proposal.json')
        self.assertEqual(proposal['geometry'],self.sample['geometric_supervision'])
        self.assertEqual(proposal['field_sources']['route'],'codex_conversation_assisted')
        self.assertTrue((output/'index.html').exists())
        with self.assertRaises(ValueError):submit_answer(output,self.root/'reply-1.json')

    def test_session_source_mismatch_and_truncation_preserved_not_repaired(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        for suffix,payload in [('wrong',None),('truncated','{"answer":')]:
            output=self.root/suffix;start_session(bundle,output)
            path=self.session_reply(output,0,self.answer())
            if payload is None:
                reply=read_json(path);reply['source_binding']='wrong';write_json(path,reply)
            else:path.write_text(payload)
            result=submit_answer(output,path)
            self.assertEqual(result['status'],'stopped_failed_response')
            self.assertEqual((output/'00000000/raw.txt').read_text(),path.read_text())
            self.assertFalse((output/'00000000/proposal.json').exists())
            self.assertEqual(prepare_next(output)['status'],'stopped_failed_response')
            self.assertFalse((output/'00000001').exists())

    def test_session_resume_and_reject_prepared_input_tampering(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        output=self.root/'session';first=start_session(bundle,output)
        self.assertEqual(first,prepare_next(output))
        reply=self.session_reply(output,0,self.answer())
        path=output/'00000000/input.json';inp=read_json(path);inp['context']['route']='changed';write_json(path,inp)
        with self.assertRaisesRegex(ValueError,'changed'):submit_answer(output,reply)
        self.assertFalse((output/'00000000/raw.txt').exists())

    def test_session_unsupported_completion_does_not_propagate(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        output=self.root/'session';start_session(bundle,output)
        answer=self.answer();answer['route']['status']='completed'
        result=submit_answer(output,self.session_reply(output,0,answer))
        self.assertEqual(result['last_result'],'rule_rejected_unreviewed')
        self.assertFalse((output/'00000001').exists())
        self.assertFalse(read_json(output/'00000000/proposal.json')['approved'])

    def test_session_rejects_injected_geometry_and_cross_frame_ids(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        for suffix in ('geometry','cross-frame'):
            output=self.root/suffix;start_session(bundle,output);answer=self.answer()
            if suffix=='geometry':answer['reference_xy']=[[1,0]]
            else:answer['landmark'][0]['image_id']='e:1'
            result=submit_answer(output,self.session_reply(output,0,answer))
            self.assertEqual(result['status'],'stopped_failed_response')
            self.assertFalse((output/'00000000/proposal.json').exists())

    def test_bundle_tampering_and_duplicate_time(self):
        samples=self.sequence();samples[1]['observation']['t_obs']=0
        with self.assertRaises(ValueError):write_bundle(self.root/'bad',samples,source='fixture')
        bundle=self.root/'bundle';write_bundle(bundle,[self.sample],source='fixture')
        (bundle/'00000000.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'source mismatch'):run_pilot(bundle,self.config,self.root/'result')

    def test_dynanav_adapter_causal_inputs_camera_convention_and_missing_map(self):
        data=self.root/'DynaNav';raw=data/'DynaNav_data/warehouse_1';cam=raw/'front_camera'
        window=data/'DynaNav_json/warehouse_1_0s_20s';window.mkdir(parents=True)
        (cam/'rgb').mkdir(parents=True);(cam/'camera_params').mkdir()
        (raw/'instruction.txt').write_text('Go straight to the forklift.')
        rows=['time,x,y,z,qx,qy,qz,qw']
        for tick in range(19):
            pose=np.eye(4);pose[:3,:3]=R_USD_TO_FLU;pose[0,3]=tick*.02
            q=Rotation.from_matrix(pose[:3,:3]).as_quat()
            rows.append(','.join(map(str,[tick,*pose[:3,3],*q])))
            Image.new('RGB',(16,12)).save(cam/'rgb'/f'rgb_{tick:05d}.jpg')
            write_json(cam/'camera_params'/f'camera_params_{tick:05d}.json',dict(cameraViewTransform=np.linalg.inv(pose).T.ravel().tolist(),metersPerSceneUnit=1,cameraModel='pinhole'))
            if tick%3==0:
                future=[dict(original_timestamp=j/30,offset=[(j-tick)*.02,0,0]) for j in range(tick+3,19,3)]
                write_json(window/f'rgb_{tick:05d}.json',dict(original_timestamp=tick/30,
                    current=dict(img=f'../../DynaNav_data/warehouse_1/front_camera/rgb/rgb_{tick:05d}.jpg',orientation=[0,0,0,1]),
                    future=future,cot='SECRET FUTURE COT',instruction_file='DO NOT USE THIS'))
        (raw/'trajectory.csv').write_text('\n'.join(rows)+'\n')
        out=self.root/'external';prepare_dynanav(data,'warehouse_1',out)
        audit=read_json(out/'adapter-audit.json');self.assertFalse(audit['static_scene_verified'])
        self.assertLess(max(x['max_future_offset_reconstruction_error_m'] for x in audit['rows']),1e-8)
        sample=read_json(out/'00000002.json');validate_sample(sample)
        request,summary=build_request(sample,read_json(self.config))
        self.assertNotIn('SECRET FUTURE COT',json.dumps(request))
        self.assertNotIn('offset',json.dumps(summary['context']))
        self.assertNotIn('T_odom_body',sample['observation'])
        self.assertFalse(sample['geometric_supervision']['usable_for_robot_reference_training'])
        self.assertIsNone(sample['observation']['mapview']['image'])
        self.assertEqual(sample['observation']['execution']['events'],[])
