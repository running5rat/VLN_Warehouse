import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

from ticvla.warehouse.schema import file_sha256
from ticvla.warehouse.semantic_input import map_records, render_semantic_map, visual_context
from ticvla.warehouse.perception.instance_selection import candidate_request
from ticvla.warehouse.dataset_tools.teacher import teacher_input
from ticvla.warehouse.dataset_tools.semantic_annotation import build_request
from ticvla.warehouse.vlm import InternVLReference


class Template:
    roles=('user','assistant')
    system_message=''
    def __init__(self):self.messages=[]
    def append_message(self, role, value):self.messages.append(value or '')
    def get_prompt(self):return self.system_message+'\n'+'\n'.join(self.messages)


class SemanticInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        rgb=self.root/'rgb.png';Image.new('RGB',(32,24),'white').save(rgb)
        self.source=dict(image_id='e:2',rgb=str(rgb),rgb_hash=file_sha256(rgb),t_obs=2.,
            T_odom_body=np.eye(4).tolist(),submap_id='e')
        batch=dict(source_image_id='e:2',rgb_hash=file_sha256(rgb),provenance={'model':'fixture'},
            instances=[dict(category='sign',mask=np.ones((24,32),bool),bbox=[0,0,32,24],confidence=.8)])
        self.candidate=candidate_request(self.source,batch,self.root/'candidates')
        self.lm=dict(landmark_uid='l',submap_id='e',category='sign',status='tentative',
            position=[2.,1.,0.],last_seen=1.,identity_text='A1',observation_ids=['e:1'])
        records=map_records([self.lm],self.source)
        path=self.root/'map.png';render_semantic_map(np.full((160,160),-1),.1,records).save(path)
        self.obs=dict(image_id='e:2',t_obs=2.,T_odom_body=np.eye(4).tolist(),task={'instruction':'Approach the sign.'},
            route='Approach the sign.',route_revision=0,robot_state=[0,0],robot_state_mask=[True,True],execution={},
            rgb_history=[dict(path=str(rgb),image_id='e:2',t_obs=2.,T_odom_body=np.eye(4).tolist())],
            map_valid=True,map_image=str(path),mapview=dict(image=str(path),landmarks=records),
            semantic_input=dict(source_until=1.,version='causal-semantic-map-v1'),perception_candidates=self.candidate)

    def test_explicit_teacher_template_is_archived_without_changing_evidence(self):
        sample=dict(observation=self.obs)
        _, before=build_request(sample,{'model':'fixture'})
        for api in ('responses','chat_completions'):
            request, after=build_request(sample,{'model':'fixture','api':api,
                'teacher_instruction':'Return annotation JSON only.',
                'teacher_template_version':'fixture-v1'})
            self.assertEqual(after['instructions'],'Return annotation JSON only.')
            self.assertEqual(after['context'],before['context'])
            self.assertEqual(after['images'],before['images'])
            self.assertEqual(after['user_text'],before['user_text'])
            self.assertEqual(after['teacher_template_version'],'fixture-v1')
            actual=request['messages'][0]['content'] if api=='chat_completions' else request['instructions']
            self.assertEqual(actual,after['instructions'])
        with self.assertRaises(ValueError):
            build_request(sample,{'teacher_instruction':''})
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            build_request(sample,{'teacher_instruction':'Changed prompt.',
                'teacher_instruction_sha256':'incorrect'})

    def test_teacher_deployment_same_images_and_semantics(self):
        context,images=teacher_input(dict(observation=self.obs))
        fake=SimpleNamespace(backbone=SimpleNamespace(conv_template=Template(),num_image_token=1))
        prompt,model_images=InternVLReference.build_prompt(fake,self.obs)
        self.assertEqual(images,model_images)
        self.assertEqual(len(images),3)
        self.assertNotIn('A1',prompt)
        self.assertIn('Current detectable categories: sign',prompt)
        self.assertEqual(context['current_candidates']['allowed_categories'],['sign'])
        self.assertIn('same-frame masks AND current RGB',prompt)
        self.assertIn('position and heading relative to identifiable landmarks',prompt)
        self.assertEqual(context['observed_map_landmarks'][0]['source_image_ids'],['e:1'])
        _,summary=build_request(dict(observation=self.obs),{'model':'fixture'})
        self.assertIn('detector boxes',summary['images'][-1]['label'])
        self.assertIn('Previous committed semantic',summary['images'][-2]['label'])
        self.assertNotIn('supervision',context)

    def test_current_future_semantics_rejected(self):
        for stamp in (2.,3.):
            with self.assertRaisesRegex(ValueError,'current/future'):
                map_records([dict(self.lm,last_seen=stamp)],self.source)
        self.obs['semantic_input']['source_until']=2.
        with self.assertRaisesRegex(ValueError,'cutoff'):
            visual_context(self.obs)

    def test_forged_source_overlay_rejected(self):
        self.obs['perception_candidates']['source_image_id']='another:2'
        with self.assertRaisesRegex(ValueError,'binding'):
            teacher_input(dict(observation=self.obs))
        self.obs['perception_candidates']['source_image_id']='e:2'
        Image.new('RGB',(32,24),'red').save(self.candidate['overlay'])
        with self.assertRaisesRegex(ValueError,'binding'):
            visual_context(self.obs)

    def test_relative_pose_and_unknown_unchanged(self):
        pose=np.eye(4);pose[0,3]=1
        records=map_records([self.lm],dict(self.source,T_odom_body=pose.tolist()))
        np.testing.assert_allclose(records[0]['position'],[1,1,0])
        grid=np.full((160,160),-1);before=grid.copy()
        render_semantic_map(grid,.1,records)
        np.testing.assert_array_equal(grid,before)
        self.assertEqual(records[0]['status'],'tentative')
        self.assertEqual(map_records([dict(self.lm,submap_id='another')],self.source),[])

    def test_current_source_cannot_be_hidden_in_history(self):
        self.obs['mapview']['landmarks'][0]['source_image_ids'].append('e:2')
        with self.assertRaisesRegex(ValueError,'Noncausal'):
            visual_context(self.obs)

    def test_empty_candidates_still_keep_raw_rgb(self):
        self.obs['perception_candidates']['candidates']=[]
        context,images=teacher_input(dict(observation=self.obs))
        self.assertEqual(context['current_candidates']['instances'],[])
        self.assertEqual(context['current_candidates']['allowed_categories'],[])
        self.assertIn('Empty candidates require []',context['semantic_input_instruction'])
        self.assertEqual(images[0],self.source['rgb'])

    def test_compact_request_excludes_audit_but_keeps_navigation_evidence(self):
        from ticvla.warehouse.compact_input import model_context, audit_context, render_context, model_user_text
        self.obs['execution']={'events':[dict(kind='turn_started',t_obs=1.,
            image_id='private-frame',annotator='private-reviewer',source_binding='secret-hash')],
            'measured_facts':{'accumulated_distance_m':2.3,'accumulated_yaw_rad':.4}}
        self.obs['task']['acceptance_criteria']={'arrival':{'max_horizontal_distance_m':4.,
            'require_target_visible':True,'require_stopped':True},'authority':'private-reviewer'}
        before=copy.deepcopy(self.obs)
        request,summary=build_request(dict(observation=self.obs),{'model':'fixture'})
        serialized=json.dumps(request)
        for forbidden in ('batch_id','image_id','source_binding','private-reviewer','secret-hash',
                          'T_odom_body','sha256','route_revision','map_version','source_time'):
            self.assertNotIn(forbidden,serialized)
        self.assertEqual(summary['context'],model_context(self.obs))
        for removed in ('landmarks','candidates','arrival','execution','rgb_history','velocity_mps_radps'):
            self.assertNotIn(removed,summary['context'])
        self.assertNotIn('turn_started',summary['user_text'])
        self.assertNotIn('A1',summary['user_text'])
        self.assertEqual(model_user_text(summary),request['input'][0]['content'][0]['text'])
        self.assertEqual(model_user_text({'context':{'old':1},'user_text_prefix':'legacy\n'}), 'legacy\n'+json.dumps({'old':1}))
        self.assertEqual(audit_context(summary)['current_image_id'],'e:2')
        self.assertEqual(audit_context(summary)['execution']['events'][0]['source_binding'],'secret-hash')
        self.assertEqual(self.obs,before)
        fake=SimpleNamespace(backbone=SimpleNamespace(conv_template=Template(),num_image_token=1))
        prompt,_=InternVLReference.build_prompt(fake,self.obs)
        self.assertIn(render_context(summary['context']),prompt)

    def test_expired_local_and_removed_map_are_not_used(self):
        from ticvla.warehouse.compact_input import model_context
        self.obs['previous_local']={'valid':False,'text':'Stale near A1','reason':'expired',
                                   'current_in_previous_body':{'x_m':9.}}
        c=model_context(self.obs,include_map=False)
        self.assertIsNone(c['previous_local'])
        self.assertNotIn('landmarks',c)
        self.assertEqual(c['map'],'unavailable')
        self.assertNotIn('Stale',json.dumps(c))

    def test_previous_local_and_motion_reach_both_prompts(self):
        from ticvla.warehouse.local_context import local_memory, local_context
        previous=copy.deepcopy(self.obs)
        previous.update(image_id='e:1',t_obs=1.,odom_epoch='e')
        self.obs['odom_epoch']='e'
        previous['T_odom_body'][0][3]=-.5
        memory=local_memory('By sign A1, facing toward sign B2.',previous,
                            origin='unreviewed_teacher_proposal',revision='response-1')
        self.obs['previous_local']=local_context(memory,self.obs)
        context,_=teacher_input(dict(observation=self.obs))
        self.assertTrue(context['previous_local']['valid'])
        self.assertEqual(context['previous_local']['text'],memory['text'])
        self.assertEqual(context['previous_local']['current_in_previous_body']['x_m'],.5)
        fake=SimpleNamespace(backbone=SimpleNamespace(conv_template=Template(),num_image_token=1))
        prompt,_=InternVLReference.build_prompt(fake,self.obs)
        self.assertIn(memory['text'],prompt)
        self.assertIn('Motion since previous Local',prompt)
        self.assertIn("Mark memory-only inference 'Inferred:'",prompt)

    def test_map_commit_feedback_and_restart_preserve_evidence(self):
        from ticvla.warehouse.mapping import MapStore, SourceObservation
        from ticvla.warehouse.perception.realtime import consume_masks
        depth=self.root/'depth.npy';np.save(depth,np.full((24,32),2.,np.float32))
        source=SourceObservation('e:1',self.source['rgb'],str(depth),
            [[32,0,16],[0,32,12],[0,0,1]],np.eye(4).tolist(),np.eye(4).tolist(),1.,'e','e','task')
        store=MapStore(self.root/'store')
        obs=store.add_observation(source)
        batch=dict(source_image_id='e:1',rgb_hash=obs['rgb_hash'],provenance={'model':'fixture'},
            instances=[dict(category='sign',mask=np.ones((24,32),bool),bbox=[0,0,32,24],confidence=.9)])
        selection=dict(selected_categories=['sign'],request_id='test')
        rows=consume_masks(store,obs,batch,selection)
        self.assertEqual(rows[0]['result']['status'],'accepted',rows)
        version=store.version
        records=map_records(store.landmark_view()['landmarks'],self.source)
        self.assertEqual(records[0]['source_image_ids'],['e:1'])
        self.assertEqual(records[0]['status'],'tentative')
        consume_masks(store,obs,batch,selection)
        self.assertEqual(version,store.version)
        store.close()
        store=MapStore(self.root/'store')
        try:
            self.assertEqual(records,map_records(store.landmark_view()['landmarks'],self.source))
            with self.assertRaisesRegex(ValueError,'current/future'):
                map_records(store.landmark_view()['landmarks'],dict(self.source,t_obs=1.))
        finally:store.close()

if __name__=='__main__':unittest.main()
