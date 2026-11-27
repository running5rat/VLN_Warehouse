"""Serialized API budget accounting; failed/unknown calls keep their reservation."""
import fcntl
import math
from pathlib import Path
from ..schema import fingerprint
from .core import read_json,write_json


class RequestNotSent(ValueError):
    request_sent = False


def usage_charge(raw, cfg):
    usage=raw.get('usage') or {}
    inp=usage.get('input_tokens',usage.get('prompt_tokens'))
    out=usage.get('output_tokens',usage.get('completion_tokens'))
    if inp is None or out is None:return None
    rates=cfg['usd_per_million_tokens']
    if any(not math.isfinite(float(rates[k])) or rates[k]<0 for k in ('input','output')):
        raise ValueError('Finite nonnegative rates required')
    return (float(inp)*rates['input']+float(out)*rates['output'])/1e6


def budgeted_call(request,cfg,send):
    """Reserve before HTTP; crash/retry cannot silently issue the same paid job twice.

    Charges ignore cache/promotional discounts and are estimates, not invoices.
    Reservation bounds are explicitly configured; no fabricated zero-cost failures.
    """
    path=Path(cfg['budget_file']);path.parent.mkdir(parents=True,exist_ok=True)
    rates=cfg['usd_per_million_tokens']
    if any(not math.isfinite(float(rates[k])) or rates[k]<0 for k in ('input','output')):
        raise RequestNotSent('Finite nonnegative rates required')
    reserve=(cfg['input_token_reservation']*rates['input']+
             cfg['output_token_reservation']*rates['output'])/1e6
    if not math.isfinite(reserve) or reserve<=0:raise RequestNotSent('Positive finite per-request reservation required')
    payload_key=fingerprint(dict(endpoint=cfg['base_url'],api=cfg.get('api','responses'),request=request))
    # Explicit user-requested experiments may repeat a previously evaluated
    # payload. A fixed run ID isolates accounting, never authorizes retries
    # within that run, and is not sent to the model.
    evaluation=cfg.get('authorized_evaluation_run')
    if evaluation is not None and (not isinstance(evaluation,dict) or
            not isinstance(evaluation.get('id'),str) or not evaluation['id'].strip() or
            not evaluation.get('reason')):
        raise RequestNotSent('Explicit evaluation run ID and authorization reason required')
    key=fingerprint(dict(payload_key=payload_key,evaluation_run_id=evaluation['id'])) if evaluation else payload_key
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        limit=min(100.,float(cfg.get('budget_limit_usd',100.)))
        if not math.isfinite(limit) or limit<=0:raise RequestNotSent('Invalid budget limit')
        ledger=read_json(path) if path.exists() else dict(limit_usd=limit,calls={})
        if ledger['limit_usd']!=limit:raise RequestNotSent('Budget limit changed')
        if any(c.get('reservation_exceeded') for c in ledger['calls'].values()):
            raise RequestNotSent('Unexpected provider usage; reconcile cost before more calls')
        retry_of = None
        if key in ledger['calls']:
            # Explicit recovery of named unknown outcomes only. Keep the original
            # reservation; this is a separately accounted, single new attempt.
            recovery=cfg.get('authorized_request_recovery',{})
            if (key not in recovery.get('request_keys',[]) or not recovery.get('id')
                    or ledger['calls'][key]['status']!='reserved_or_outcome_unknown'):
                raise RequestNotSent('Paid request already recorded; inspect saved result, do not automatically retry')
            retry_of=key
            key=fingerprint(dict(retry_of=retry_of,recovery_id=recovery['id']))
            if key in ledger['calls']:
                raise RequestNotSent('Authorized recovery attempt already recorded; no automatic retry')
        spent=sum(c['accounted_usd'] for c in ledger['calls'].values())
        if spent+reserve>limit:raise RequestNotSent('API budget exhausted before request')
        row=dict(status='reserved_or_outcome_unknown',model=cfg['model'],accounted_usd=reserve,reservation_usd=reserve)
        if evaluation:row.update(payload_key=payload_key,evaluation_run_id=evaluation['id'],evaluation_reason=evaluation['reason'])
        if retry_of:row.update(retry_of=retry_of,recovery_id=recovery['id'])
        ledger['calls'][key]=row;write_json(path,ledger)
    # The durable reservation already counts toward the shared limit. Do not hold
    # the ledger lock across network I/O: independent episodes can run concurrently.
    # A failed/interrupted request keeps its reservation and cannot be reissued.
    raw=send()
    charge=usage_charge(raw,cfg)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ledger=read_json(path)
        row=ledger['calls'][key]
        if charge is not None:
            row.update(status='usage_received',accounted_usd=charge,usage=raw.get('usage'),
                       cost_kind=cfg.get('cost_kind','undiscounted_rate_estimate_not_invoice'),reservation_exceeded=charge>reserve)
        else:row.update(status='response_without_usage_reservation_retained')
        write_json(path,ledger)
        return raw
