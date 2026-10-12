import copy
import json
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
import test_landmark_perception as fixtures
from ticvla.warehouse.perception.instance_selection import candidate_request, parse_answer, question, selection_result
from ticvla.warehouse.perception.realtime import consume_masks, consume_text
from ticvla.warehouse.mapping import MapStore
from DynaNav.warehouse_loop.instance_mapping import expired

class InstanceSelectionTests(unittest.TestCase):
    setUp=fixtures.IdentityMapTests.setUp
    tearDown=fixtures.IdentityMapTests.tearDown

    def make(self):
        source=self.store.add_observation(self.obs)
        batch=dict(source_image_id=source['image_id'],rgb_hash=source['rgb_hash'],instances=[dict(category='sign',mask=self.mask.copy(),bbox=[15,20,55,55],confidence=.9)],provenance={'test':'fixture'},elapsed_s=0)
        request=candidate_request(source,batch,self.root/'qa')
        batch['candidate_request']=request
        qa=dict(valid=True,parsed=dict(selected_instances=[dict(instance_id='d0',purpose='identity_candidate')],missed_categories=[]),request_id='r',batch_id=request['batch_id'],source_image_id=source['image_id'],rgb_hash=source['rgb_hash'],source_time=0)
        return source,batch,selection_result(qa)

    def test_source_ids_and_prompt(self):
        source,batch,s=self.make()
        prompt=question(batch['candidate_request'])
        self.assertNotIn('Task:',prompt)
        self.assertNotIn('navigation',prompt.lower())
        self.assertEqual(prompt.count('<image>'),2)
        for field,value in [('batch_id','bad'),('source_image_id','another-frame'),('rgb_hash','wrong')]:
            bad=copy.deepcopy(s);bad[field]=value
            with self.assertRaises(ValueError):consume_masks(self.store,source,batch,bad)
        self.assertEqual(len(self.store.records('landmarks')),0)
        self.assertEqual(consume_masks(self.store,source,batch,s)[0]['result']['status'],'accepted')

    def test_empty_bad_duplicate_and_truncation(self):
        source,batch,s=self.make();request=batch['candidate_request']
        for answer in ['{"selected_instances":',json.dumps(dict(selected_instances=[dict(instance_id='d99',purpose='identity_candidate')],missed_categories=[])),json.dumps(dict(selected_instances=s['selected_instances']*2,missed_categories=[]))]:
            with self.assertRaises(ValueError):parse_answer(answer,request)
        s['selected_instances']=[]
        self.assertEqual(consume_masks(self.store,source,batch,s)[0]['result']['status'],'not_selected')
        self.assertFalse(self.store.records('landmarks'))

    def test_projection_reused_and_depth_failure(self):
        source,batch,s=self.make()
        prepared=self.store.prepare_projection(source['image_id'],self.mask)
        expected=prepared['points'].copy();batch['prepared_projections']={0:prepared}
        with patch('ticvla.warehouse.mapping.store.semantic_points',side_effect=AssertionError('reprojected')):
            r=consume_masks(self.store,source,batch,s)[0]['result']
        self.assertEqual(r['status'],'accepted')
        lm=self.store.get('landmarks',r['landmark_uid'])
        np.testing.assert_allclose(lm['position'],expected.mean(0),atol=.05)
        self.assertFalse(lm['static_confirmed'])
        bad=copy.copy(prepared);bad['mask_hash']='wrong'
        from ticvla.warehouse.mapping import SemanticEvent
        result=self.store.consume(SemanticEvent('new',source['image_id'],'sign',source='llm'),self.mask,.9,prepared_projection=bad)
        self.assertEqual(result['status'],'rejected')
        np.save(self.root/'depth.npy',np.zeros((80,100),np.float32))
        new=self.store.add_observation(replace(self.obs,image_id='bad-depth',t_obs=1))
        self.assertIn('Insufficient',self.store.prepare_projection(new['image_id'],self.mask)['error'])

    def test_early_late_text_duplicate_restart(self):
        source,batch,s=self.make()
        ocr=dict(source_image_id=source['image_id'],rgb_hash=source['rgb_hash'],provenance={'test':'ocr'},texts=[dict(text='Q-17',confidence=.99,polygon=[[20,25],[40,25],[40,30],[20,30]])])
        # OCR existed before QA admission; same consumer also handles late arrival.
        rows=consume_masks(self.store,source,batch,s)
        uid=rows[0]['result']['landmark_uid'];before=self.store.get('landmarks',uid)
        result=consume_text(self.store,source,batch,rows,ocr)
        self.assertEqual(result['updates'][0]['status'],'accepted')
        after=self.store.get('landmarks',uid)
        self.assertEqual(before['geometry'],after['geometry'])
        self.assertEqual(before['observation_ids'],after['observation_ids'])
        version=self.store.version
        self.store.close();self.store=MapStore(self.root/'map')
        rows=consume_masks(self.store,source,batch,s)
        self.assertTrue(rows[0]['result']['duplicate'])
        self.assertTrue(consume_text(self.store,source,batch,rows,ocr)['updates'][0]['duplicate'])
        self.assertEqual(version,self.store.version)
        self.assertEqual(self.store.view(source['image_id'])['landmarks'],[])

    def test_changed_mask_and_geometry_source(self):
        source,batch,s=self.make();batch['instances'][0]['mask'][25,25]=False
        with self.assertRaises(ValueError):consume_masks(self.store,source,batch,s)
        _,batch2,s2=self.make()
        new=self.store.add_observation(replace(self.obs,image_id='new-frame',t_obs=1))
        with self.assertRaises(ValueError):consume_masks(self.store,new,batch2,s2)

    def test_expiry_is_original_submission_time(self):
        self.assertFalse(expired({'submitted_wall':10},15))
        self.assertTrue(expired({'submitted_wall':10},15.01))


    def test_one_visual_generation_without_navigation_heads(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import Mock
        from ticvla.warehouse.perception.instance_selection import predict
        source,batch,selection=self.make()
        answer=json.dumps(dict(selected_instances=selection['selected_instances'],missed_categories=[]))
        chat=Mock(return_value=answer)
        model=SimpleNamespace(backbone=SimpleNamespace(system_message='previous',chat=chat),
            tokenizer=SimpleNamespace(encode=lambda value,add_special_tokens=False:list(range(30))),
            _tokenizer_data=object(),cfg=SimpleNamespace(image_size=448),device='cpu',compute_dtype=torch.float32)
        with patch('lmformatenforcer.integrations.transformers.build_transformers_prefix_allowed_tokens_fn',return_value=lambda *a:[]):
            result=predict(SimpleNamespace(model=model),source,batch['candidate_request'])
        self.assertTrue(result['valid'],result)
        self.assertEqual(chat.call_count,1)
        self.assertEqual(chat.call_args.args[1].shape,(2,3,448,448))
        self.assertEqual(chat.call_args.kwargs['num_patches_list'],[1,1])
        self.assertFalse(result['navigation_task_included'])
        self.assertFalse(result['structured_inference_executed'])
        self.assertEqual(model.backbone.system_message,'previous')

    def test_empty_decision_is_not_admission_in_metrics(self):
        from types import SimpleNamespace
        from DynaNav.warehouse_loop.instance_mapping import summarize
        capture=self.root/'capture';capture.mkdir()
        (capture/'performance-report.json').write_text(json.dumps(dict(run_wall_start=0,run_wall_end=10)))
        records=[dict(module='geometry',wall=.1),dict(module='candidate',wall=.2,latency_s=.1),
                 dict(module='admission',wall=1,image_id='empty',accepted_instances=0,latency_s=.5),
                 dict(module='admission',wall=3,image_id='picked',accepted_instances=1,latency_s=1),
                 dict(module='text',wall=3.1,image_id='picked',accepted_instances=1,latency_s=1.1),
                 dict(module='ocr_binding',wall=3.1,image_id='picked',status='processed',latency_s=1.1,bound_updates=1),
                 dict(module='ocr_model_completed',wall=2,image_id='picked')]
        summarize(self.root/'metrics',capture,records,1,1,['empty','picked'],['empty','picked'],dict(seg=0,qa=0,ocr=0),SimpleNamespace(summary=lambda:{}))
        report=json.loads((self.root/'metrics/realtime-report.json').read_text())
        self.assertEqual(report['selection_completed_hz'],.2)
        self.assertEqual(report['admission_hz'],.1)
        self.assertEqual(report['ocr_binding_eligible_samples'],1)
        self.assertEqual(report['ocr_samples'],2)
        self.assertEqual(report['ocr_bindings_fraction_all_samples'],.5)
