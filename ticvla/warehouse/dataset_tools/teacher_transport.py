"""Explicit teacher HTTP routing; no retries or process-wide proxy mutation."""
import urllib.request
from urllib.parse import urlsplit

from .teacher_budget import RequestNotSent


def transport_policy(base_url, cfg):
    host = urlsplit(base_url).hostname
    configured = cfg.get('http_proxy_mode', 'provider-default')
    if configured not in ('provider-default', 'direct', 'environment'):
        raise RequestNotSent('http_proxy_mode must be provider-default, direct or environment')
    resolved = configured
    if configured == 'provider-default':
        resolved = 'direct' if host == 'dashscope.aliyuncs.com' else 'environment'
    return dict(version='teacher-http-v1', endpoint_host=host,
                configured_proxy_mode=configured, resolved_proxy_mode=resolved)


def open_teacher_request(request, cfg):
    policy = transport_policy(request.full_url, cfg)
    timeout = float(cfg.get('timeout_seconds', 180))
    if policy['resolved_proxy_mode'] == 'direct':
        # An empty ProxyHandler disables environment proxies for this opener
        # only. Do not install a global opener: other provider threads may need
        # their existing proxy configuration.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)
