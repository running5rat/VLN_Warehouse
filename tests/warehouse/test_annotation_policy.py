import hashlib
import json
from pathlib import Path
import unittest

from ticvla.warehouse.dataset_tools.annotation_policy import CONTINUE, STOP, failure_policy, must_stop


class AnnotationPolicyTests(unittest.TestCase):
    def test_only_response_rejections_may_continue(self):
        self.assertEqual(failure_policy({}), CONTINUE)
        self.assertFalse(must_stop({}, ValueError('Local exceeds 80 characters'), 'response_validation'))
        for stage in ('api_request', 'source_validation', 'map_commit'):
            self.assertTrue(must_stop({}, ValueError('failure'), stage))
        self.assertTrue(must_stop({}, OSError('write failed'), 'response_validation'))
        exc=ValueError('budget exceeded');exc.request_sent=False
        self.assertTrue(must_stop({}, exc, 'response_validation'))
        self.assertTrue(must_stop({'annotation_failure_policy':STOP}, ValueError(), 'response_validation'))
        with self.assertRaises(ValueError):failure_policy({'annotation_failure_policy':'typo'})

    def test_selected_profile_matches_lock(self):
        root=Path(__file__).resolve().parents[2]
        lock=json.loads((root/'configs/dataset_preparation/qwen_plus_teacher.v1.lock.json').read_text())
        content=(root/lock['config_file']).read_bytes();cfg=json.loads(content)
        self.assertEqual(hashlib.sha256(content).hexdigest(),lock['config_sha256'])
        self.assertEqual(hashlib.sha256(cfg['teacher_instruction'].encode()).hexdigest(),lock['teacher_instruction_sha256'])
        self.assertEqual(cfg['model'],'qwen3.7-plus')
        self.assertTrue(cfg['enable_thinking'])
        self.assertEqual(cfg['thinking_budget'],2048)
        self.assertEqual(cfg['max_output_tokens'],8192)
        self.assertEqual(cfg['annotation_failure_policy'],CONTINUE)

    def test_v2_profile_lock_and_explicit_contract(self):
        root=Path(__file__).resolve().parents[2]
        lock=json.loads((root/'configs/dataset_preparation/qwen_plus_teacher.v2.lock.json').read_text())
        content=(root/lock['config_file']).read_bytes();cfg=json.loads(content)
        self.assertEqual(hashlib.sha256(content).hexdigest(),lock['config_sha256'])
        self.assertEqual(hashlib.sha256(cfg['teacher_instruction'].encode()).hexdigest(),lock['teacher_instruction_sha256'])
        self.assertEqual(cfg['local_max_characters'],90)
        self.assertTrue(cfg['completion_evidence_input'])
        for phrase in ('must be concise and must not exceed 90 characters', 'exactly "DONE"', 'Completion evidence: confirmed',
                       'target_visible', 'arrival_confirmed', 'instead of ~'):
            self.assertIn(phrase,cfg['teacher_instruction'])
        self.assertEqual(cfg['model'],'qwen3.7-plus')
        self.assertEqual(cfg['thinking_budget'],2048)
        self.assertEqual(cfg['max_output_tokens'],8192)
