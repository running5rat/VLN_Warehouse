import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ticvla.warehouse.dataset_tools.credentials import load_credentials
from ticvla.warehouse.dataset_tools.core import read_json, write_json
from ticvla.warehouse.dataset_tools.oracle_comparison import run
from ticvla.warehouse.schema import file_sha256


class OracleExecutionTests(unittest.TestCase):
    def test_named_evaluation_repeats_payload_once_without_erasing_prior_charge(self):
        from ticvla.warehouse.dataset_tools.teacher_budget import budgeted_call, RequestNotSent
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(budget_file=str(Path(folder)/'budget.json'),model='fixture',base_url='https://example.invalid',
                input_token_reservation=1000,output_token_reservation=1000,
                usd_per_million_tokens=dict(input=1.,output=2.))
            request={'id':1}
            send=lambda:dict(usage=dict(prompt_tokens=100,completion_tokens=50))
            budgeted_call(request,cfg,send)
            old=read_json(cfg['budget_file'])['calls']
            cfg['authorized_evaluation_run']=dict(id='user-requested-experiment',reason='Explicit repeated evaluation')
            budgeted_call(request,cfg,send)
            new=read_json(cfg['budget_file'])['calls']
            self.assertEqual(len(new),2)
            for key,row in old.items():self.assertEqual(new[key],row)
            with self.assertRaises(RequestNotSent):budgeted_call(request,cfg,lambda:self.fail('Duplicate experiment request'))
            cfg['authorized_evaluation_run']={'id':''}
            with self.assertRaises(RequestNotSent):budgeted_call({'id':2},cfg,lambda:self.fail('Missing authorization'))

    def test_explicit_unknown_recovery_keeps_old_reservation_and_allows_only_one_attempt(self):
        from ticvla.warehouse.dataset_tools.teacher_budget import budgeted_call, RequestNotSent
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(budget_file=str(Path(folder)/'budget.json'),model='fixture',base_url='https://example.invalid',
                input_token_reservation=1000,output_token_reservation=1000,
                usd_per_million_tokens=dict(input=1.,output=2.))
            request={'id':1}
            with self.assertRaises(TimeoutError):
                budgeted_call(request,cfg,lambda: (_ for _ in ()).throw(TimeoutError()))
            original=read_json(cfg['budget_file']);key=next(iter(original['calls']))
            with self.assertRaises(RequestNotSent):budgeted_call(request,cfg,lambda:self.fail('Unexpected retry'))
            cfg['authorized_request_recovery']={'id':'user-authorized-fixture','request_keys':[key]}
            budgeted_call(request,cfg,lambda:dict(usage=dict(prompt_tokens=100,completion_tokens=50)))
            after=read_json(cfg['budget_file'])
            self.assertEqual(after['calls'][key],original['calls'][key])
            self.assertEqual(len(after['calls']),2)
            self.assertEqual(sum(v.get('retry_of')==key for v in after['calls'].values()),1)
            with self.assertRaises(RequestNotSent):budgeted_call(request,cfg,lambda:self.fail('Second retry'))

    def test_recovery_cannot_reissue_a_successful_response(self):
        from ticvla.warehouse.dataset_tools.teacher_budget import budgeted_call, RequestNotSent
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(budget_file=str(Path(folder)/'budget.json'),model='fixture',base_url='https://example.invalid',
                input_token_reservation=1000,output_token_reservation=1000,
                usd_per_million_tokens=dict(input=1.,output=2.))
            budgeted_call({'id':1},cfg,lambda:dict(usage=dict(prompt_tokens=10,completion_tokens=5)))
            key=next(iter(read_json(cfg['budget_file'])['calls']))
            cfg['authorized_request_recovery']={'id':'fixture','request_keys':[key]}
            with self.assertRaises(RequestNotSent):budgeted_call({'id':1},cfg,lambda:self.fail('Duplicate'))

    def test_budget_reservations_allow_concurrent_http_without_lost_charges(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        from ticvla.warehouse.dataset_tools.teacher_budget import budgeted_call
        with tempfile.TemporaryDirectory() as folder:
            cfg = dict(budget_file=str(Path(folder)/'budget.json'), model='fixture',
                base_url='https://example.invalid', input_token_reservation=1000,
                output_token_reservation=1000, usd_per_million_tokens=dict(input=1., output=2.))
            barrier = Barrier(2)
            def send():
                barrier.wait(timeout=5)
                return dict(usage=dict(prompt_tokens=100, completion_tokens=50))
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(budgeted_call, {'id': i}, cfg, send) for i in range(2)]
                for job in jobs: job.result(timeout=10)
            ledger = read_json(cfg['budget_file'])
            self.assertEqual(len(ledger['calls']), 2)
            self.assertAlmostEqual(sum(r['accounted_usd'] for r in ledger['calls'].values()), .0004)
            self.assertTrue(all(r['status']=='usage_received' for r in ledger['calls'].values()))

    def test_inflight_reservation_counts_against_budget(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        from ticvla.warehouse.dataset_tools.teacher_budget import budgeted_call
        with tempfile.TemporaryDirectory() as folder:
            cfg=dict(budget_file=str(Path(folder)/'budget.json'),model='fixture',
                base_url='https://example.invalid',input_token_reservation=1000,
                output_token_reservation=1000,budget_limit_usd=.004,
                usd_per_million_tokens=dict(input=1.,output=2.))
            entered,release=Event(),Event()
            def send():
                entered.set()
                if not release.wait(timeout=5):raise RuntimeError('Test synchronization failed')
                return dict(usage=dict(prompt_tokens=100,completion_tokens=50))
            with ThreadPoolExecutor(max_workers=1) as pool:
                first=pool.submit(budgeted_call,{'id':1},cfg,send)
                self.assertTrue(entered.wait(timeout=5))
                try:
                    with self.assertRaisesRegex(ValueError,'exhausted'):
                        budgeted_call({'id':2},cfg,lambda:self.fail('Over-budget HTTP sent'))
                finally:release.set()
                first.result(timeout=5)
            self.assertEqual(len(read_json(cfg['budget_file'])['calls']),1)

    def test_credentials_are_data_and_only_provider_variables_are_loaded(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {}, clear=True):
            path = Path(folder) / 'credentials.env'
            path.write_text("export OPENAI_API_KEY='literal-$(echo secret)'\nPATH=/bad\nGLM_API_KEY=abc # comment\n")
            self.assertEqual(load_credentials(path), ['GLM_API_KEY', 'OPENAI_API_KEY'])
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'literal-$(echo secret)')
            self.assertNotIn('PATH', os.environ)
            path.write_text('OPENAI_API_KEY=one\nOPENAI_API_KEY=two\n')
            with self.assertRaises(ValueError): load_credentials(path)
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'literal-$(echo secret)')

    def test_resume_reuses_completed_calls_and_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); pack = root/'pack'; out = root/'out'; cfg = root/'config.json'
            write_json(cfg, dict(model='fixture', budget_file='unused', base_url='https://example.invalid'))
            write_json(root/'stage-b.json', dict(passed=True))
            bundles = []
            for name in ('first', 'second', 'pending'):
                path = pack/name; write_json(path/'manifest.json', dict(name=name))
                bundles.append(dict(name=name, path=str(path), samples=1,
                    manifest_sha256=file_sha256(path/'manifest.json'), context_ready=name!='pending'))
            write_json(pack/'comparison.json', dict(bundles=bundles, selected_samples_per_model=3,
                stage_B_report=str(root/'stage-b.json'), stage_B_report_sha256=file_sha256(root/'stage-b.json')))
            calls = []
            def fake(bundle, config, output, prepare_only=False):
                calls.append(str(bundle))
                report = dict(status='completed_with_unreviewed_proposals', actual_api_calls=1,
                    config_sha256=file_sha256(config), bundle_sha256=file_sha256(Path(bundle)/'manifest.json'),
                    samples=[dict(status='format_valid_unreviewed')])
                write_json(Path(output)/'report.json', report)
                return report
            with patch('ticvla.warehouse.dataset_tools.oracle_comparison.run_pilot', side_effect=fake):
                self.assertEqual(run(pack,cfg,out,max_bundles=1)['actual_api_calls'],1)
                report=run(pack,cfg,out,resume=True)
                self.assertEqual(report['actual_api_calls'],2)
                self.assertEqual(len(calls),2)
                self.assertEqual(report['results'][-1]['status'],'not_run_gold_context_review_required')
                write_json(cfg,dict(model='changed',budget_file='unused'))
                with self.assertRaisesRegex(ValueError,'configuration or source changed'):
                    run(pack,cfg,out,resume=True)


if __name__ == '__main__': unittest.main()
