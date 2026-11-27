import io
import os
import unittest
import urllib.error
import urllib.request
from unittest.mock import MagicMock, patch

from ticvla.warehouse.dataset_tools.teacher_transport import open_teacher_request, transport_policy
from ticvla.warehouse.dataset_tools.semantic_annotation import send_request
from ticvla.warehouse.dataset_tools.teacher_budget import RequestNotSent


class TeacherTransportTests(unittest.TestCase):
    def test_provider_routing_is_exact_and_other_providers_keep_environment(self):
        for host, expected in [('dashscope.aliyuncs.com','direct'),
                               ('api.openai.com','environment'),
                               ('dashscope.aliyuncs.com.example.org','environment')]:
            self.assertEqual(transport_policy('https://'+host, {})['resolved_proxy_mode'], expected)
        self.assertEqual(transport_policy('https://dashscope.aliyuncs.com',
            {'http_proxy_mode':'environment'})['resolved_proxy_mode'],'environment')

    def test_direct_ignores_conflicting_environment_without_mutating_it(self):
        request=urllib.request.Request('https://dashscope.aliyuncs.com/compatible-mode/v1/models')
        opener=MagicMock()
        with patch.dict(os.environ, {'HTTPS_PROXY':'http://127.0.0.1:7890',
                                     'https_proxy':'http://127.0.0.1:7892'}), \
             patch('urllib.request.build_opener',return_value=opener) as build, \
             patch('urllib.request.urlopen') as ambient:
            before=dict(os.environ)
            open_teacher_request(request, {'timeout_seconds':13})
            self.assertEqual(build.call_args.args[0].proxies,{})
            opener.open.assert_called_once_with(request,timeout=13.)
            ambient.assert_not_called()
            self.assertEqual(dict(os.environ),before)

    def test_generation_uses_direct_opener_and_preserves_payload(self):
        cfg=dict(base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
                 api='chat_completions',api_key_env='FIXTURE_KEY',timeout_seconds=7)
        opener=MagicMock();opener.open.return_value=io.BytesIO(b'{"choices":[]}')
        with patch.dict(os.environ,{'FIXTURE_KEY':'fixture'}), \
             patch('urllib.request.build_opener',return_value=opener):
            self.assertEqual(send_request({'model':'fixture'},cfg),{'choices':[]})
        req=opener.open.call_args.args[0]
        self.assertEqual(req.full_url,cfg['base_url']+'/chat/completions')
        self.assertEqual(req.get_method(),'POST')
        self.assertEqual(req.data,b'{"model": "fixture"}')

    def test_timeout_does_not_retry_or_fall_back(self):
        req=urllib.request.Request('https://dashscope.aliyuncs.com/compatible-mode/v1/models')
        opener=MagicMock();opener.open.side_effect=urllib.error.URLError('timed out')
        with patch('urllib.request.build_opener',return_value=opener), \
             patch('urllib.request.urlopen') as fallback:
            with self.assertRaises(urllib.error.URLError):open_teacher_request(req,{})
            opener.open.assert_called_once();fallback.assert_not_called()

    def test_invalid_policy_fails_before_budget_or_network(self):
        with patch('ticvla.warehouse.dataset_tools.teacher_budget.budgeted_call') as paid:
            with self.assertRaises(RequestNotSent):
                send_request({},dict(base_url='https://dashscope.aliyuncs.com',
                    http_proxy_mode='invalid',budget_file='unused'))
            paid.assert_not_called()


if __name__=='__main__':unittest.main()
