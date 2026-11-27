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

    def test_route_only_uses_initialized_route_then_excludes_original_task_from_payload(self):
        sample=copy.deepcopy(self.sample)
        original=sample['observation']['task']['instruction']
        sample['observation']['route']=original
        cfg=dict(model='fixture',api='chat_completions',task_input_mode='route_only')
        request,summary=build_request(sample,cfg)
        self.assertNotIn('Instruction:',summary['user_text'])
        self.assertEqual(summary['user_text'].count(original),1)
        self.assertIn('Previous Route: '+original,summary['user_text'])
        self.assertNotIn('instruction',summary['context'])
        sample['observation']['route']='Stop.'
        request,summary=build_request(sample,cfg)
        self.assertIn('Previous Route: Stop.',summary['user_text'])
        self.assertNotIn(original,json.dumps(request))
        self.assertEqual(sample['observation']['task']['instruction'],original)
        self.assertEqual(summary['audit_context']['task']['instruction'],original)
        _,legacy=build_request(sample,dict(model='fixture'))
        self.assertIn('Instruction: '+original,legacy['user_text'])
        with self.assertRaises(ValueError):build_request(sample,dict(model='fixture',task_input_mode='typo'))

    def answer(self):return dict(landmark=[dict(image_id='e:0',category='sign')],local='Near a sign.',
        route=dict(remaining_instruction='Turn left, then approach the extinguisher.',current_subgoal=dict(action='turn',direction='left'),status='active'))

    def test_compact_teacher_output_binds_current_and_keeps_text_compact(self):
        from ticvla.warehouse.dataset_tools.semantic_annotation import assemble_answer, agreement
        answer=self.answer(); answer['landmark']=['sign','fire extinguisher']
        before=copy.deepcopy(answer)
        proposal=assemble_answer(answer,self.sample)
        self.assertEqual(answer,before)
        self.assertIn('<landmark>["sign", "fire extinguisher"]</landmark>',proposal['text'])
        self.assertEqual(proposal['parsed']['landmark'],[
            dict(image_id='e:0',category='sign'), dict(image_id='e:0',category='fire extinguisher')])
        self.assertFalse(proposal['approved'])
        self.assertEqual(proposal['geometry'],self.sample['geometric_supervision'])
        self.assertEqual(agreement(proposal['parsed'],self.sample)['rule_flags'],[])
        _,summary=build_request(self.sample,read_json(self.config))
        self.assertEqual(summary['audit_context']['current_image_id'],'e:0')
        self.assertIn('CURRENT RGB only',summary['instructions'])

    def test_compact_route_teacher_and_completion_guard(self):
        from ticvla.warehouse.dataset_tools.semantic_annotation import assemble_answer, agreement
        answer=self.answer();answer.update(landmark=['sign'],route='Turn left, then approach the extinguisher.')
        proposal=assemble_answer(answer,self.sample)
        self.assertEqual(proposal['route_protocol'],'compact-v2')
        self.assertIn('<Route>'+answer['route']+'</Route>',proposal['text'])
        self.assertNotIn('current_subgoal',proposal['text'])
        self.assertIsNone(agreement(proposal['parsed'],self.sample)['route_action_exact'])
        answer['route']='DONE'
        proposal=assemble_answer(answer,self.sample)
        self.assertIn('completed_without_recorded_target_arrival_and_stop_evidence',
                      agreement(proposal['parsed'],self.sample)['rule_flags'])
        self.sample['observation']['execution']['events']=[dict(kind=k,t_obs=0.)
            for k in ('target_visible','stopped','arrival_confirmed')]
        self.assertEqual(agreement(proposal['parsed'],self.sample)['rule_flags'],[])

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
        self.assertIn('JSON',request['input'][0]['content'][0]['text'])
        self.assertIn('unavailable',summary['context']['map'])

    def test_chat_provider_uses_same_causal_content_and_rejects_truncation(self):
        from ticvla.warehouse.dataset_tools.semantic_annotation import parse_api_response
        config=read_json(self.config)
        first,summary=build_request(self.sample,config)
        config.update(api='chat_completions',thinking={'type':'enabled'})
        chat,other=build_request(self.sample,config)
        self.assertEqual(summary['context'],other['context'])
        self.assertEqual(summary['images'],other['images'])
        self.assertEqual(first['instructions'],chat['messages'][0]['content'])
        self.assertNotIn('reference_xy',json.dumps(chat))
        self.assertNotIn('HUMAN ANSWER MUST NOT LEAK',json.dumps(chat))
        raw={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.answer())}}]}
        self.assertEqual(json.loads(parse_api_response(raw)),self.answer())
        raw['choices'][0]['finish_reason']='length'
        with self.assertRaises(ValueError):parse_api_response(raw)

    def test_explicit_rgb_encoding_preserves_resolution_and_source(self):
        import base64,io
        cfg=read_json(self.config);cfg['rgb_jpeg_quality']=95
        original=file_sha256(self.image)
        request,summary=build_request(self.sample,cfg)
        url=request['input'][0]['content'][2]['image_url']
        decoded=Image.open(io.BytesIO(base64.b64decode(url.split(',',1)[1])))
        self.assertEqual(decoded.size,(16,12));self.assertEqual(file_sha256(self.image),original)
        cfg['api']='chat_completions'
        chat,other=build_request(self.sample,cfg)
        self.assertEqual(summary['images'],other['images'])
        self.assertEqual(url,chat['messages'][1]['content'][2]['image_url']['url'])

    def test_budget_reserves_failures_and_deduplicates_paid_requests(self):
        from ticvla.warehouse.dataset_tools.teacher_budget import budgeted_call
        cfg=dict(budget_file=str(self.root/'budget.json'),model='fixture',base_url='https://example.invalid',
            input_token_reservation=1000,output_token_reservation=1000,
            usd_per_million_tokens={'input':1.,'output':2.})
        count=[]
        def send():
            count.append(1)
            return {'usage':{'prompt_tokens':100,'completion_tokens':50}}
        budgeted_call({'job':1},cfg,send)
        with self.assertRaisesRegex(ValueError,'already recorded'):budgeted_call({'job':1},cfg,send)
        self.assertEqual(len(count),1)

        def fail():raise OSError('unknown network outcome')
        with self.assertRaises(OSError):budgeted_call({'job':2},cfg,fail)
        ledger=read_json(self.root/'budget.json')
        self.assertAlmostEqual(sum(c['accounted_usd'] for c in ledger['calls'].values()),.0032)
        cfg['input_token_reservation']=200000000
        with self.assertRaisesRegex(ValueError,'exhausted'):budgeted_call({'job':3},cfg,send)
        self.assertEqual(len(count),1)

    def test_budget_rejection_is_not_counted_as_real_api_call(self):
        from ticvla.warehouse.dataset_tools.teacher_budget import RequestNotSent
        write_bundle(self.root/'bundle',[self.sample],source='fixture',context_mode='recursive')
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture-only'}), patch(
            'ticvla.warehouse.dataset_tools.semantic_annotation.send_request',
            side_effect=RequestNotSent('budget exhausted')):
            report=run_pilot(self.root/'bundle',self.config,self.root/'rejected')
        self.assertEqual(report['actual_api_calls'],0)
        self.assertEqual(report['samples'][0]['status'],'not_sent_preflight_rejected')

    def test_transport_failure_stops_remaining_provider_requests(self):
        import io
        import urllib.error
        cfg=read_json(self.config);cfg['stop_on_transport_error']=True;write_json(self.config,cfg)
        write_bundle(self.root/'bundle',self.sequence(),source='fixture',context_mode='gold_context')
        error=urllib.error.HTTPError('https://example.invalid',401,'Unauthorized',{},io.BytesIO(b'{"message":"fixture-only"}'))
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture-only'}), patch(
            'ticvla.warehouse.dataset_tools.semantic_annotation.send_request',side_effect=error) as send:
            report=run_pilot(self.root/'bundle',self.config,self.root/'failed')
        self.assertEqual(send.call_count,1)
        self.assertEqual(report['actual_api_calls'],1)
        self.assertEqual(report['samples'][1]['status'],'not_run_provider_error')
        self.assertNotIn('fixture-only',(self.root/'failed/00000000/http-error.txt').read_text())

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

    def test_local_90_limit_is_shared_by_annotation_and_decoder_grammar(self):
        from ticvla.warehouse.structured_io import response_grammar
        for length in (80, 81, 90):
            answer = dict(landmark=[], local='A' * length, route='Wait here.')
            proposal = assemble_answer(answer, self.sample)
            parser = response_grammar(['e:0'], categories_only=True, compact_route=True)
            for char in proposal['text']:
                self.assertIn(char, parser.get_allowed_characters())
                parser = parser.add_character(char)
            self.assertTrue(parser.can_end())
            self.assertEqual(answer['local'], 'A' * length)
        with self.assertRaisesRegex(ValueError, '1 to 90'):
            assemble_answer(dict(landmark=[], local='A' * 91, route='Wait here.'), self.sample)
        with self.assertRaises(ValueError):
            assemble_answer(dict(landmark=[], local='Turn ~30 degrees.', route='Wait here.'), self.sample)

    def test_v2_completion_input_matches_guard_without_geometry_leakage(self):
        cfg = read_json(Path(__file__).resolve().parents[2] /
                        'configs/dataset_preparation/qwen_plus_teacher.v2.json')
        sample = copy.deepcopy(self.sample)
        sample['evaluation_only'] = {}
        answer = dict(landmark=[], local='unknown', route='DONE')
        for kinds, expected in (([], 'unverified'),
                                (['target_visible', 'stopped'], 'unverified'),
                                (['target_visible', 'stopped', 'arrival_confirmed'], 'confirmed')):
            sample['observation']['execution']['events'] = [dict(kind=k) for k in kinds]
            request, summary = build_request(sample, cfg)
            actual_text = request['messages'][1]['content'][0]['text']
            self.assertEqual(actual_text, summary['user_text'])
            self.assertIn('Completion evidence: ' + expected, actual_text)
            self.assertNotIn('reference_xy', json.dumps(request))
            self.assertEqual(bool(agreement(answer, sample)['rule_flags']), expected != 'confirmed')
        answer['route'] = 'Wait here. DONE'
        self.assertIn('noncanonical_completion_marker', agreement(answer, sample)['rule_flags'])

    def test_v4_route_grounding_reaches_actual_system_without_extra_evidence(self):
        import hashlib
        root = Path(__file__).resolve().parents[2]
        cfg = read_json(root/'configs/dataset_preparation/qwen_plus_teacher.v4.json')
        previous = read_json(root/'configs/dataset_preparation/qwen_plus_teacher.v2.json')
        lock = read_json(root/'configs/dataset_preparation/qwen_plus_teacher.v4.lock.json')
        self.assertEqual(file_sha256(root/lock['config_file']), lock['config_sha256'])
        self.assertEqual(hashlib.sha256(cfg['teacher_instruction'].encode()).hexdigest(),
                         lock['teacher_instruction_sha256'])
        request, summary = build_request(self.sample, cfg)
        old, _ = build_request(self.sample, previous)
        self.assertEqual(request['messages'][1], old['messages'][1])
        for key in ('model', 'max_tokens', 'enable_thinking', 'thinking_budget'):
            self.assertEqual(request[key], old[key])
        system = request['messages'][0]['content']
        self.assertEqual(system, summary['instructions'])
        self.assertIn('the supplied images OR the observed map', system)
        self.assertIn('Previous Route may remain unchanged', system)
        self.assertIn('remain as a search goal even if not yet observed', system)
        self.assertNotIn('Do not copy', system)
        self.assertIn('Remove substeps only when evidence supports their completion', system)
        self.assertIn('This restriction does NOT prevent', system)
        self.assertIn('must not exceed 90 characters', system)
        self.assertNotIn('reference_xy', json.dumps(request))

    def test_agreement_is_not_semantic_accuracy_and_completion_needs_evidence(self):
        a=self.answer();r=agreement(a,self.sample)
        self.assertIsNone(r['semantic_accuracy']);self.assertTrue(r['route_action_exact'])
        self.assertFalse(r['remaining_instruction_exact'])
        a['route']['status']='completed'
        self.assertTrue(agreement(a,self.sample)['rule_flags'])
        s=copy.deepcopy(self.sample);s['evaluation_only']={}
        self.assertFalse(agreement(a,s)['human_reference_available'])

    def test_embedded_DONE_is_rejected_without_rewriting_teacher_text(self):
        a=self.answer()
        for route in ('Wait here. DONE','DONE: Go to the sink.'):
            a['route']=route
            flags=agreement(a,self.sample)['rule_flags']
            self.assertIn('noncanonical_completion_marker',flags)
            self.assertIn('completed_without_recorded_target_arrival_and_stop_evidence',flags)
            self.assertEqual(a['route'],route)

    def test_recursive_uses_teacher_route_not_gold_and_no_approval(self):
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        calls=[]
        def send(request,cfg):
            calls.append(request['input'][0]['content'][0]['text'])
            answer=self.answer();answer['landmark']=['sign']
            return self.raw(answer)
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture'}),patch('ticvla.warehouse.dataset_tools.semantic_annotation.send_request',side_effect=send):
            r=run_pilot(bundle,self.config,self.root/'result')
        self.assertEqual(r['actual_api_calls'],2)
        self.assertIn('Previous Route: '+self.sample['observation']['task']['instruction'],calls[0])
        self.assertIn('Previous Route: '+self.answer()['route']['remaining_instruction'],calls[1])
        self.assertIn('Previous Local: Near a sign.',calls[1])
        self.assertNotIn('GOLD PREVIOUS',json.dumps(calls))
        self.assertFalse(r['human_approved']);self.assertIsNone(r['semantic_accuracy'])

    def test_failed_response_stops_recursive_chain_no_repair(self):
        cfg=read_json(self.config);cfg['annotation_failure_policy']='stop_on_invalid_v1';write_json(self.config,cfg)
        bundle=self.root/'bundle';write_bundle(bundle,self.sequence(),source='fixture')
        bad=self.raw({'landmark':[]})
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture'}),patch('ticvla.warehouse.dataset_tools.semantic_annotation.send_request',return_value=bad) as send:
            r=run_pilot(bundle,self.config,self.root/'result')
        self.assertEqual(send.call_count,1);self.assertEqual(r['actual_api_calls'],1)
        self.assertEqual(r['samples'][1]['status'],'not_run_previous_proposal_failed')
        self.assertTrue((self.root/'result/00000000/raw.txt').exists())
        self.assertFalse(r['real_teacher_interface_pass'])

    def test_overlength_frame_skipped_without_corrupting_recursive_context(self):
        samples=self.sequence();third=copy.deepcopy(samples[1]);third['observation']['image_id']='e:2'
        third['observation']['t_obs']=2.
        third['observation']['rgb_history'][-1].update(image_id='e:2',t_obs=2.)
        samples.append(third)
        write_bundle(self.root/'bundle',samples,source='fixture')
        calls=[]
        good=dict(landmark=['sign'],local='Near a sign.',route='Keep moving toward the aisle end.')
        bad=dict(good,local='A'*91,route='THIS REJECTED ROUTE MUST NOT PROPAGATE')
        def send(request,cfg):
            calls.append(request['input'][0]['content'][0]['text'])
            return self.raw(bad if len(calls)==2 else good)
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture'}),patch(
                'ticvla.warehouse.dataset_tools.semantic_annotation.send_request',side_effect=send):
            r=run_pilot(self.root/'bundle',self.config,self.root/'result')
        self.assertEqual(r['actual_api_calls'],3)
        self.assertEqual([x['status'] for x in r['samples']],['format_valid_unreviewed','failed','format_valid_unreviewed'])
        self.assertIn(good['route'],calls[2]);self.assertIn(good['local'],calls[2])
        self.assertNotIn(bad['route'],calls[2])
        self.assertNotIn('GOLD PREVIOUS',calls[2])
        self.assertEqual(r['samples'][2]['context_recovery']['skipped_source_image_ids'],['e:1'])
        self.assertEqual(r['samples'][2]['context_recovery']['context_source_image_id'],'e:0')
        self.assertEqual(json.loads((self.root/'result/00000001/raw.txt').read_text()),bad)
        self.assertFalse(r['all_samples_completed'])

    def test_map_loop_skips_invalid_answer_without_committing_it(self):
        from contextlib import ExitStack
        from unittest.mock import MagicMock
        from ticvla.warehouse.dataset_tools import semantic_loop as loop
        samples=[]
        for i in range(3):
            s=copy.deepcopy(self.sample)
            s['observation'].update(image_id=f'e:{i}',t_obs=float(i))
            s['provenance'].update(source_episode='fixture',frame_index=i)
            samples.append(s)
        prepared=self.root/'prepared';prepared.mkdir()
        write_json(prepared/'manifest.json',dict(complete=True,records=[
            dict(folder=f'{i:08d}',geometry_sha256='fixture') for i in range(3)]))
        cfg=read_json(self.config);cfg['native_per_million_tokens']={'input':1,'output':1}
        write_json(self.config,cfg)
        store=MagicMock();store.version=0
        store.landmark_view.return_value={'landmarks':[]};store.summary.return_value={}
        def add(source):
            store.version+=1
            return source
        store.add_observation.side_effect=add
        def load(_,entry):
            i=int(entry['folder']);s=copy.deepcopy(samples[i]);obs=s['observation']
            source=dict(image_id=obs['image_id'],t_obs=obs['t_obs'],T_odom_body=np.eye(4).tolist(),submap_id='e')
            p=dict(source=source,resolution=.1,geometry_source_until=i-.1,
                geometry_image_id=f'e:prior{i}',candidate_request={},elapsed_s=0,ocr={'elapsed_s':0})
            return s,p,{'instances':[]},np.zeros((2,2),dtype=np.int8)
        inputs=[]
        def request(sample,cfg):
            inputs.append(copy.deepcopy(sample['observation']))
            return {},{}
        good=dict(landmark=[],local='Near the sign.',route='Continue down the aisle.')
        responses=[self.raw(good),self.raw(dict(good,local='X'*91,route='Rejected route.')),self.raw(good)]
        with ExitStack() as stack:
            for name,value in dict(load_credentials=lambda _:None,load_frame=load,
                validate_sample=lambda s:s,build_request=request,render_report=lambda *a:None,
                MapStore=lambda _:store,source_from_frame=lambda _,i:dict(image_id=f'e:{i}',t_obs=float(i),
                    T_odom_body=np.eye(4).tolist(),submap_id='e')).items():
                stack.enter_context(patch.object(loop,name,value))
            stack.enter_context(patch.object(loop,'send_request',side_effect=responses))
            for name,result in [('consume_masks',[]),('consume_text',{'updates':[]}),('save_frame',None)]:
                stack.enter_context(patch('ticvla.warehouse.perception.realtime.'+name,return_value=result))
            stack.enter_context(patch('DynaNav.warehouse_loop.realtime_mapping.render'))
            report=loop.run(prepared,self.config,self.root/'map-run','unused')
        self.assertEqual(report['actual_api_calls'],3)
        self.assertTrue(report['all_frames_attempted']);self.assertFalse(report['fatal_stop'])
        self.assertFalse(report['all_frames_completed'])
        self.assertEqual([c.args[0]['image_id'] for c in store.add_observation.call_args_list],['e:0','e:2'])
        self.assertEqual(inputs[2]['route'],good['route'])
        self.assertEqual(inputs[2]['semantic_input']['map_version'],1)
        self.assertEqual(inputs[2]['semantic_input']['semantic_source_until'],0.)
        self.assertEqual(report['samples'][2]['context_recovery']['skipped_source_image_ids'],['e:1'])

    def test_rejected_done_not_used_by_next_frame(self):
        write_bundle(self.root/'bundle',self.sequence(),source='fixture')
        answers=[dict(landmark=[],local='unknown',route='DONE'),dict(landmark=[],local='unknown',route='Continue straight.')]
        calls=[]
        def send(request,cfg):
            calls.append(request['input'][0]['content'][0]['text'])
            return self.raw(answers[len(calls)-1])
        with patch.dict(os.environ,{'TEST_SEMANTIC_KEY':'fixture'}),patch(
                'ticvla.warehouse.dataset_tools.semantic_annotation.send_request',side_effect=send):
            r=run_pilot(self.root/'bundle',self.config,self.root/'result')
        self.assertEqual(r['actual_api_calls'],2)
        self.assertEqual(r['samples'][0]['status'],'rule_rejected_unreviewed')
        self.assertNotIn('Previous Route: DONE',calls[1])
        self.assertIn(self.sample['observation']['task']['instruction'],calls[1])

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
            self.assertEqual(second['context']['route'],self.answer()['route']['remaining_instruction'])
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
