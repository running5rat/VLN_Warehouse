"""Bounded Qwen teacher comparison; preserves failures and keeps approval manual.

Use the existing causal comparison pack. Reuse paid Max responses, compare
independent recursive sequences, and report matched first-frame/gold-context
results separately from recursive results. No GPT calls or training updates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import html
import json
from pathlib import Path
import shutil
import time
import urllib.error

import numpy as np

from .core import read_json, write_json
from .credentials import load_credentials
from .oracle_review import token_counts, format_diagnostics
from .semantic_annotation import run_pilot, send_request, parse_api_response
from ..schema import file_sha256, fingerprint

VARIANTS = {
    'max-thinking-reused': ('qwen3.8-max', True, 8192, 12., 36.),
    'max-no-thinking': ('qwen3.8-max', False, None, 12., 36.),
    'plus-no-thinking': ('qwen3.7-plus', False, None, 2., 8.),
    'plus-thinking-2048': ('qwen3.7-plus', True, 2048, 2., 8.),
    'omni-flash-no-thinking': ('qwen3.8-omni-flash', False, None, .8, 2.7),
}


def prepare(source, output):
    source, output = Path(source), Path(output)
    if (output/'experiment.json').exists():
        raise ValueError('Experiment already frozen; use run to resume')
    manifest = read_json(source/'source-pack.json')
    if file_sha256(manifest['stage_B_report']) != manifest['stage_B_report_sha256']:
        raise ValueError('Stage B changed')
    bundles = [b for b in manifest['bundles'] if b['context_ready']]
    for b in bundles:
        if file_sha256(Path(b['path'])/'manifest.json') != b['manifest_sha256']:
            raise ValueError('Source bundle changed')
    output.mkdir(parents=True, exist_ok=True)
    base = read_json(source/'configs/qwen3.8-max.json')
    configs = {}
    for variant, (model, thinking, budget, inp, out) in VARIANTS.items():
        cfg = copy.deepcopy(base)
        cfg.update(model=model, enable_thinking=thinking,
                   price_checked_date='2026-10-27', account_access_verified=False,
                   timeout_seconds=180, native_per_million_tokens=dict(input=inp, output=out),
                   usd_per_million_tokens=dict(input=inp, output=out),
                   price_note='Undiscounted Beijing rates; not invoice; ignores promotional/cache/batch discounts')
        cfg.pop('thinking_budget', None)
        if budget is not None:
            cfg['thinking_budget'] = budget
        path = output/'configs'/f'{variant}.json'
        write_json(path, cfg)
        configs[variant] = dict(path=str(path.resolve()), sha256=file_sha256(path))
    shutil.copytree(source/'qwen3.8-max', output/'max-thinking-reused')
    write_json(output/'experiment.json', dict(bundles=bundles, configs=configs,
        source=str(source), source_manifest_sha256=file_sha256(source/'source-pack.json'),
        reused_model='qwen3.8-max', selected_per_variant=sum(b['samples'] for b in bundles),
        gpt_status='pending_user_requested_2026-10-27', gpt_new_calls=0,
        semantic_labels_approved=0, training=False,
        protocol='current-categories-compact-route-v3',
        fairness='Manual gold-context and first recursive frames share input. Later recursive contexts intentionally diverge.',
        budget='Existing shared $100 ceiling retained; CNY estimates accounted conservatively as USD, not an FX conversion.'))


def run(output, credentials, probe_only=False):
    root = Path(output)
    manifest = read_json(root/'experiment.json')
    load_credentials(credentials)

    def lane(variant):
        binding = manifest['configs'][variant]
        if file_sha256(binding['path']) != binding['sha256']:
            raise ValueError('Frozen configuration changed')
        stopped = False
        for b in manifest['bundles'][:1] if probe_only else manifest['bundles']:
            if file_sha256(Path(b['path'])/'manifest.json') != b['manifest_sha256']:
                raise ValueError('Frozen source bundle changed')
            dest = root/variant/b['name']
            if dest.exists():
                r = read_json(dest/'report.json')
                if r.get('status') == 'running':
                    raise ValueError('Interrupted paid request; inspect before resuming')
                if r['config_sha256'] != binding['sha256'] or r['bundle_sha256'] != b['manifest_sha256']:
                    raise ValueError('Existing run binding changed')
            elif stopped:
                continue
            else:
                r = run_pilot(b['path'], binding['path'], dest)
            if any(x['status'] in ('not_run_provider_error', 'not_sent_preflight_rejected') or
                   str(x.get('reason', '')).startswith('HTTP ') for x in r['samples']):
                stopped = True
            print(variant, b['name'], r['actual_api_calls'], r['rule_pass_count'], flush=True)
        return variant

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(lane, v) for v in VARIANTS if v != 'max-thinking-reused']
        for f in futures:
            f.result()
    summarize(root)


def summarize_rows(rows, selected):
    responses = [r for r in rows if r.get('response_sha256')]
    times = [r['elapsed_seconds'] for r in responses if r.get('elapsed_seconds') is not None]
    priced = [r for r in responses if r.get('cost_native_estimate') is not None]
    passed = [r for r in rows if r['status'] == 'format_valid_unreviewed']
    return dict(selected=selected, responses=len(responses), rule_pass=len(passed),
        rule_pass_over_selected=len(passed)/selected if selected else None,
        format_failed=sum(r['status']=='failed' for r in rows),
        without_response=selected-len(responses),
        not_run=sum(r['status'].startswith('not_') for r in rows)+max(0,selected-len(rows)),
        rule_rejected=sum(r['status']=='rule_rejected_unreviewed' for r in rows),
        historical_and_new_usage_cost_cny=sum(r['cost_native_estimate'] for r in priced),
        priced_responses=len(priced),
        mean_cny_per_response=float(np.mean([r['cost_native_estimate'] for r in priced])) if priced else None,
        latency_seconds={k:float(fn(times)) if times else None for k,fn in
                         [('mean',np.mean),('p50',lambda x:np.percentile(x,50)),('p95',lambda x:np.percentile(x,95))]},
        semantic_accuracy=None, human_approved=0)


def map_probe(output, source, credentials):
    """Separate stress case; no synthetic trajectory and no robot commands."""
    from .map_memory_review import build_request, validate_answer
    root, source = Path(output), Path(source)
    manifest = read_json(root/'experiment.json')
    inp = read_json(source/'input.json')
    load_credentials(credentials)
    baseline=root/'map-probe/max-thinking-reused'
    if not baseline.exists():shutil.copytree(source/'qwen3.8-max',baseline)
    write_json(root/'map-probe/source.json',dict(source=str(source),input_sha256=inp['input_sha256'],
        scope='Separate offscreen-target stress test; never pooled into R2R/RxR cost or accuracy'))

    def one(variant):
        folder=root/'map-probe'/variant
        if folder.exists():
            r=read_json(folder/'report.json')
            if r['status']=='request_started':raise ValueError('Inspect interrupted paid map request')
            return r
        binding=manifest['configs'][variant]
        if file_sha256(binding['path'])!=binding['sha256']:raise ValueError('Config changed')
        cfg=read_json(binding['path']);folder.mkdir(parents=True)
        req=build_request(inp,cfg)
        write_json(folder/'input.json',inp);write_json(folder/'config.json',cfg)
        row=dict(model=cfg['model'],variant=variant,input_sha256=inp['input_sha256'],
                 request_sha256=fingerprint(req),status='request_started',human_approved=False,actual_api_calls=1)
        write_json(folder/'report.json',row);start=time.monotonic()
        try:
            raw=send_request(req,cfg);write_json(folder/'response.json',raw)
            row['usage']=raw.get('usage')
            (folder/'raw.txt').write_text((raw.get('choices') or [{}])[0].get('message',{}).get('content') or '')
            from .teacher_budget import usage_charge
            row['cost_native_estimate']=usage_charge(raw,dict(cfg,usd_per_million_tokens=cfg['native_per_million_tokens']))
            answer=json.loads(parse_api_response(raw));row['validation']=validate_answer(answer)
            write_json(folder/'parsed.json',answer);row['status']='format_valid_awaiting_review'
        except (ValueError,KeyError,TypeError,OSError) as exc:
            row.update(status='failed',reason=str(exc))
            if getattr(exc,'request_sent',True) is False:row['actual_api_calls']=0
            if isinstance(exc,urllib.error.HTTPError):row['http_status']=exc.code
        row['elapsed_seconds']=time.monotonic()-start;write_json(folder/'report.json',row)
        print('map',variant,row['status'],flush=True)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one,[v for v in VARIANTS if v!='max-thinking-reused']))


def summarize(output):
    root = Path(output)
    manifest = read_json(root/'experiment.json')
    sources = {}
    for b in manifest['bundles']:
        p = Path(b['path'])
        for i, entry in enumerate(read_json(p/'manifest.json')['records']):
            sample = read_json(p/entry['path'])
            sources[entry['image_id']] = dict(sample=sample, bundle=b['name'], index=i,
                                            common_input=b['name'].startswith('manual') or i==0)
    results, models = {}, {}
    review = read_json(root/'assistant-review.json') if (root/'assistant-review.json').exists() else {}
    for variant in VARIANTS:
        rows = []
        for report in sorted((root/variant).glob('*/report.json')):
            r = read_json(report)
            for original in r['samples']:
                row = copy.deepcopy(original)
                folder = report.parent/row['folder']
                row.update(variant=variant, artifact=str(folder.relative_to(root)),
                    common_input=sources[row['source_image_id']]['common_input'],
                    tokens=token_counts(row.get('usage')),
                    response_sha256=file_sha256(folder/'response.json') if (folder/'response.json').exists() else None)
                if (folder/'raw.txt').exists():
                    row['raw'] = (folder/'raw.txt').read_text()
                    row['format_diagnostics'] = format_diagnostics(row['raw'])
                if (folder/'input.json').exists():
                    inp=read_json(folder/'input.json')
                    row['causal_input_hash']=fingerprint({k:inp[k] for k in ('instructions','context','images')})
                rows.append(row)
                results.setdefault(row['source_image_id'], {})[variant] = row
        models[variant] = summarize_rows(rows, manifest['selected_per_variant'])
        models[variant]['new_calls'] = 0 if variant=='max-thinking-reused' else sum(
            read_json(p)['actual_api_calls'] for p in (root/variant).glob('*/report.json'))
        models[variant]['common_input_subset'] = summarize_rows([r for r in rows if r['common_input']],
                                                               sum(s['common_input'] for s in sources.values()))
    mismatch = []
    for source, variants in results.items():
        hashes={r['causal_input_hash'] for r in variants.values() if r['common_input'] and r.get('causal_input_hash')}
        if len(hashes)>1: mismatch.append(source)
    report=dict(models=models, common_input_mismatches=mismatch, semantic_review=review,
                gpt_status=manifest['gpt_status'], gpt_new_calls=0, labels_approved=0,
                new_usage_cost_cny=sum(m['historical_and_new_usage_cost_cny'] for v,m in models.items() if v!='max-thinking-reused'))
    write_json(root/'results.json',results)
    write_json(root/'report.json',report)
    parts=['<!doctype html><meta charset="utf-8"><title>Qwen 标注分级对照</title>',
        '<style>body{font:16px sans-serif;margin:24px;background:#15212b;color:#edf2f7}a{color:#71cafa}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #657181}pre{white-space:pre-wrap;overflow-wrap:anywhere}.answers{display:flex;gap:15px;overflow-x:auto}.answer{min-width:310px;max-width:460px;flex:1}img{max-width:100%}details{margin:12px 0}summary{cursor:pointer} .images{display:flex;flex-wrap:wrap;gap:8px}.images img{width:360px;object-fit:contain}</style>',
        '<h1>Qwen：标注正确性与成本对照</h1><p>GPT 暂缓，无新 GPT 调用。旧 Max 思考结果复用。文本提案未自动批准；格式通过不等于标注正确。</p>',
        '<p>10 个人工关键帧＋8 个 episode ×4 个递归时刻。人工 10 帧和各序列首帧共 18 帧输入相同；后续 Route 由各模型独立递归，不能视为完全相同输入。</p>',
        '<p><a href="report.json">完整统计</a> · <a href="results.json">逐帧输出与费用</a> · <a href="experiment.json">冻结配置</a> · <a href="REPORT.md">结论报告</a></p>',
        '<table><tr><th>配置</th><th>响应/计划</th><th>规则通过</th><th>一轮费用 CNY</th><th>p50/p95 秒</th></tr>']
    for v,m in models.items():
        p=m['latency_seconds'];parts.append(f'<tr><td>{html.escape(v)}</td><td>{m["responses"]}/{m["selected"]}</td><td>{m["rule_pass"]}</td><td>{m["historical_and_new_usage_cost_cny"]:.4f}</td><td>{p["p50"]} / {p["p95"]}</td></tr>')
    review_by_key={(r['source_image_id'],r['variant']):r for r in review.get('reviews',[])}
    parts.append('</table><h2>Codex 辅助审核（不是独立人工批准）</h2><p>'+html.escape(review.get('scope','尚未完成'))+'</p>')
    parts.append('<table><tr><th>配置</th><th>语义未发现问题</th><th>局部修订</th><th>严重问题</th><th>同时符合协议和语义检查</th></tr>')
    for v,s in review.get('summary',{}).items():
        j=s['judgments'];parts.append(f'<tr><td>{v}</td><td>{j.get("acceptable",0)}/18</td><td>{j.get("minor_edit",0)}</td><td>{j.get("major_error",0)}</td><td>{s["protocol_and_semantic_acceptable"]}/18</td></tr>')
    parts.append('</table><p>这是已知样本的非盲辅助复核，不是总体模型准确率。未自动批准任何训练标签。</p><details><summary>审核定义、逐条理由和限制</summary><pre>'+html.escape(json.dumps(review,ensure_ascii=False,indent=2))+'</pre></details>')
    if (root/'cost-projections.json').exists():parts.append('<p><a href="cost-projections.json">实测费用与首批／全量外推</a> · <a href="human-review-template.json">人工审核空白模板</a></p>')
    if (root/'map-probe').exists():
        parts.append('<h2>超视距路标：独立压力案例</h2><p>不混入上述费用/正确性分母。当前画面、地图与全部上下文见 input.json。</p><div class="answers">')
        for v in VARIANTS:
            f=root/'map-probe'/v
            if (f/'report.json').exists():
                raw=(f/'raw.txt').read_text() if (f/'raw.txt').exists() else '无输出'
                parts.append('<div class="answer"><h3>'+v+'</h3><pre>'+html.escape(raw)+'</pre><a href="map-probe/'+v+'/report.json">耗时/费用</a> · <a href="map-probe/'+v+'/input.json">输入</a></div>')
        parts.append('</div>')
        if (root/'map-probe/assistant-review.json').exists():parts.append('<details><summary>超视距案例逐模型审核意见</summary><pre>'+html.escape(json.dumps(read_json(root/'map-probe/assistant-review.json'),ensure_ascii=False,indent=2))+'</pre></details>')
    for image_id, variants in results.items():
        src=sources[image_id];obs=src['sample']['observation']
        parts.append('<details><summary>'+html.escape(image_id)+' — '+('共同输入' if src['common_input'] else '各自递归上下文')+'</summary>')
        parts.append('<p>Instruction: '+html.escape(obs['task']['instruction'])+'</p><div class="images">')
        images=[(h['path'],h['image_id']) for h in obs['rgb_history']]
        if obs.get('mapview',{}).get('image'):images.append((obs['mapview']['image'],'MapView'))
        for path,label in images:
            path=Path(path); dest=root/'images'/(file_sha256(path)+path.suffix)
            if not dest.exists():dest.parent.mkdir(exist_ok=True);shutil.copyfile(path,dest)
            parts.append('<figure><img loading="lazy" src="'+str(dest.relative_to(root))+'"><figcaption>'+html.escape(label)+'</figcaption></figure>')
        parts.append('</div><details><summary>人工参照，仅审核可见，不进入当前答案输入</summary><pre>'+html.escape(json.dumps(src['sample'].get('evaluation_only'),ensure_ascii=False,indent=2))+'</pre></details><div class="answers">')
        for v,r in variants.items():
            parts.append('<div class="answer"><h3>'+html.escape(v)+'</h3><p>'+html.escape(r['status']+' '+r.get('reason',''))+'</p><pre>'+html.escape(r.get('raw','未调用'))+'</pre>')
            judged=review_by_key.get((image_id,v))
            if judged:parts.append('<p><b>辅助审核：'+html.escape(judged['semantic_verdict'])+'</b> '+html.escape(judged['notes'])+'</p>')
            for filename in ('input.json','response.json','proposal.json','comparison.json','raw.txt','http-error.txt'):
                if (root/r['artifact']/filename).exists():parts.append('<a href="'+r['artifact']+'/'+filename+'">'+filename+'</a> ')
            parts.append('</div>')
        parts.append('</div></details>')
    (root/'index.html').write_text(''.join(parts))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','summarize','map-probe'])
    p.add_argument('--output',required=True)
    p.add_argument('--source')
    p.add_argument('--credentials',default='/home/xinglinlu/.config/llm/credentials.env')
    p.add_argument('--probe-only',action='store_true')
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.output)
    elif a.mode=='run':run(a.output,a.credentials,a.probe_only)
    elif a.mode=='map-probe':map_probe(a.output,a.source,a.credentials)
    else:summarize(a.output)
