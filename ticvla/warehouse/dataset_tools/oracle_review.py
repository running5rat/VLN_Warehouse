"""Build an auditable, offline human-review page and usage-based cost estimates."""
from ..compact_input import audit_context

import argparse
from collections import Counter
import html
import json
import re
from pathlib import Path
import shutil

import numpy as np

from .core import read_json, write_json
from ..schema import file_sha256, fingerprint

FIELDS = ['landmark_visibility', 'landmark_usefulness', 'local_grounding',
          'route_goal_preservation', 'route_progress', 'completion_evidence', 'future_leakage']
VERDICTS = ['pending', 'pass', 'minor_edit', 'major_error', 'unscorable']
PRICES = {
    'gpt-6-luna': ('USD', .1, .5, 'https://developers.openai.com/api/docs/pricing'),
    'gpt-6.1-sol': ('USD', 2., 10., 'https://developers.openai.com/api/docs/pricing'),
    'gpt-6-astra': ('USD', 10., 50., 'https://developers.openai.com/api/docs/pricing'),
    'glm-5.3-flashx': ('CNY', 2., 7., 'https://docs.bigmodel.cn/cn/guide/start/pricing'),
    'qwen3.8-max': ('CNY', 12., 36., 'https://help.aliyun.com/zh/model-studio/model-pricing'),
}


def format_diagnostics(raw):
    """Explain unchanged deployment limits without silently rewriting an answer."""
    try: answer=json.loads(raw)
    except (ValueError,TypeError): return dict(json_valid=False, violations=['invalid_json'])
    if not isinstance(answer,dict): return dict(json_valid=True,violations=['not_an_object'])
    issues=[]
    local=answer.get('local'); route=answer.get('route')
    if isinstance(local,str):
        if len(local)>90:issues.append('local_exceeds_90_characters')
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 ,.';:()_/-]*",local):issues.append('local_character_outside_deployment_alphabet')
    if isinstance(route,str) and len(route)>600:issues.append('route_exceeds_600_characters')
    return dict(json_valid=True,local_characters=len(local) if isinstance(local,str) else None,
                route_characters=len(route) if isinstance(route,str) else None,violations=issues)


def token_counts(usage):
    usage = usage or {}
    inp = usage.get('input_tokens', usage.get('prompt_tokens'))
    out = usage.get('output_tokens', usage.get('completion_tokens'))
    details = usage.get('output_tokens_details', usage.get('completion_tokens_details')) or {}
    reasoning = details.get('reasoning_tokens')
    visible = details.get('text_tokens')
    # The residual can contain protocol overhead: never label it as exact visible NTP.
    residual = out-reasoning if out is not None and reasoning is not None else None
    return dict(input=inp, output=out, reasoning=reasoning, visible_reported=visible,
                nonreasoning_residual=residual)


def estimate_cost(rows, counts):
    """No zero-filling missing cohorts, no 'approved-label cost' before review."""
    cohort_means = {}
    for cohort in counts:
        values = [r['cost_native_estimate'] for r in rows
                  if r['source_image_id'].startswith(cohort+'-') and r.get('cost_native_estimate') is not None]
        cohort_means[cohort] = dict(priced_responses=len(values),
            mean_per_call=float(np.mean(values)) if values else None,
            p10=float(np.percentile(values,10)) if values else None,
            p90=float(np.percentile(values,90)) if values else None)
    total = (sum(counts[k]*cohort_means[k]['mean_per_call'] for k in counts)
             if all(cohort_means[k]['mean_per_call'] is not None for k in counts) else None)
    return dict(cohorts=cohort_means, counts=counts, one_attempt_per_frame=total,
                review_rework_20_percent_scenario=total*1.2 if total is not None else None,
                cost_per_human_approved_label=None,
                caveat='Small recursive pilot; failures can truncate sequences. Not a confidence interval or invoice. Rework is a scenario, not automatic retry authorization.')


def validate_reviews(data, template):
    if data.get('report_id') != template['report_id']:
        raise ValueError('Review belongs to a different report')
    expected = {r['key']: r for r in template['reviews']}
    seen = set()
    for row in data.get('reviews', []):
        key = row.get('key')
        if key in seen or key not in expected: raise ValueError('Duplicate/unknown review source')
        seen.add(key)
        if row.get('response_sha256') != expected[key]['response_sha256']:
            raise ValueError('Response hash mismatch')
        if set(row.get('fields', {})) != set(FIELDS): raise ValueError('Review fields missing')
        if any(v not in VERDICTS for v in row['fields'].values()): raise ValueError('Invalid review verdict')
        if row.get('overall') not in VERDICTS: raise ValueError('Invalid overall verdict')
        if row['overall'] != 'pending' and not str(data.get('reviewer','')).strip():
            raise ValueError('Reviewer required')
        if row['overall'] == 'pass' and any(v not in ('pass','unscorable') for v in row['fields'].values()):
            raise ValueError('Overall pass requires all fields reviewed')
    return data


def reference_svg(geometry):
    """Fixed source-body XY preview. Demonstration geometry is review-only."""
    points=np.asarray(geometry.get('reference_xy',[]),dtype=float).reshape(-1,2)
    mask=np.asarray(geometry.get('reference_mask',[]),dtype=bool)
    if len(mask)!=len(points):return '<p>参考线维度错误</p>'
    points=points[mask]
    anchors=np.asarray(geometry.get('waypoint',[]),dtype=float).reshape(-1,2)
    if not np.isfinite(points).all() or not np.isfinite(anchors).all():return '<p>非有限几何</p>'
    if not len(points):return '<p>该源帧没有有效参考线；不是未标注动作的停车 GT。</p>'
    span=max(2.,float(np.abs(points).max())+.5)
    def xy(p):return (220-p[1]*180/span,220-p[0]*180/span)
    line=' '.join(f'{x:.1f},{y:.1f}' for x,y in map(xy,points))
    content=f'<svg width="440" height="440" viewBox="0 0 440 440" role="img" aria-label="示范参考线，仅供审核"><rect width="440" height="440" fill="#0b1720"/><path d="M220 20V420 M20 220H420" stroke="#788c9a"/><text x="230" y="30" fill="white">+x forward</text><text x="20" y="210" fill="white">+y left</text><text x="15" y="430" fill="white">源车体系；半幅 {span:.1f} m</text><polyline points="{line}" fill="none" stroke="#ffb24c" stroke-width="3"/><circle cx="220" cy="220" r="5" fill="#ff5f68"/>'
    for i,p in enumerate(anchors):
        x,y=xy(p);content+=f'<circle cx="{x}" cy="{y}" r="4" fill="#59dae0"/><text x="{x+6}" y="{y}" fill="#59dae0">W{i+1}</text>'
    return content+'</svg>'


def build(root, stage_b):
    root = Path(root)
    manifest = read_json(root/'source-pack.json')
    dataset = read_json(stage_b)
    if file_sha256(stage_b) != manifest['stage_B_report_sha256']:
        raise ValueError('Stage B data changed')
    counts = Counter()
    for r in dataset['results']:
        if r['role'] == 'train': counts[r['uid'].split('-')[0]] += len(r['records'])
    blocked = read_json(root/'provider-status.json')
    human_references, prepared_images = {}, {}
    for b in manifest['bundles']:
        if b['context_ready']:
            bundle=Path(b['path'])
            for entry in read_json(bundle/'manifest.json')['records']:
                sample=read_json(bundle/entry['path'])
                human_references[sample['observation']['image_id']]=sample.get('evaluation_only')
                obs=sample['observation']
                images=[dict(path=h['path'],label='RGB '+h['image_id']) for h in obs['rgb_history']]
                if obs.get('mapview',{}).get('image'):images.append(dict(path=obs['mapview']['image'],label='Causal observed MapView'))
                prepared_images[obs['image_id']]=images
    models, source_rows, reviews = [], {}, []
    for config in sorted((root/'configs').glob('*.json')):
        cfg = read_json(config); model = cfg['model']; rows = []; calls = 0
        if cfg.get('native_per_million_tokens') and cfg['native_per_million_tokens'] != dict(input=PRICES[model][1],output=PRICES[model][2]):
            raise ValueError('Cost rates differ from checked price table')
        for report_path in sorted((root/model).glob('*/report.json')):
            report = read_json(report_path)
            if report.get('status') == 'running': raise ValueError('Wait for all running bundles')
            calls += report['actual_api_calls']
            for r in report['samples']:
                folder = report_path.parent/r['folder']
                row = dict(r, artifact=str(folder.relative_to(root)), context_mode=report.get('context_mode'))
                row['tokens'] = token_counts(r.get('usage'))
                for name in ('raw.txt','input.json','proposal.json','comparison.json','response.json','geometric-supervision.json'):
                    p = folder/name
                    if p.exists(): row[name] = p.read_text() if name=='raw.txt' else read_json(p)
                row['response_sha256'] = file_sha256(folder/'response.json') if (folder/'response.json').exists() else None
                row['format_diagnostics'] = format_diagnostics(row.get('raw.txt','')) if row['response_sha256'] else None
                # Strictly distinguish the requested compact JSON from backward-compatible parser acceptance.
                try:
                    raw = json.loads(row.get('raw.txt',''))
                    row['compact_protocol'] = (set(raw)=={'landmark','local','route'} and
                        isinstance(raw['landmark'],list) and all(isinstance(c,str) for c in raw['landmark']) and
                        isinstance(raw['local'],str) and isinstance(raw['route'],str))
                except (ValueError,TypeError): row['compact_protocol'] = False
                if row['compact_protocol'] and 'input.json' in row:
                    events={e['kind'] for e in audit_context(row['input.json'])['execution'].get('events',[])}
                    row['completion_evidence_diagnostic']=dict(claims_done=raw['route']=='DONE',
                        required_event_types_present=sorted(events & {'target_visible','stopped','arrival_confirmed'}),
                        unsupported_done=raw['route']=='DONE' and not {'target_visible','stopped','arrival_confirmed'} <= events,
                        note='Independent audit even if output failed length grammar; not propagated into recursive state.')
                row['human_reference_for_review_only']=human_references.get(r['source_image_id'])
                if 'input.json' not in row:row['prepared_source_images_not_submitted']=prepared_images[r['source_image_id']]
                row['review_key'] = model+'|'+r['source_image_id']
                if row['response_sha256']:
                    reviews.append(dict(key=row['review_key'], source_image_id=r['source_image_id'], model=model,
                        response_sha256=row['response_sha256'], fields={f:'pending' for f in FIELDS},
                        overall='pending', notes='', corrected_json=None))
                rows.append(row)
                by_model = source_rows.setdefault(r['source_image_id'],{})
                if model in by_model: raise ValueError('Duplicate source/model in report')
                by_model[model] = row
        priced = [r for r in rows if r.get('cost_native_estimate') is not None]
        received = [r for r in rows if r['response_sha256']]
        latencies = [r['elapsed_seconds'] for r in rows if r.get('elapsed_seconds') is not None]
        totals = {key:sum(r['tokens'][key] or 0 for r in received) for key in ('input','output','reasoning')}
        means = {key:float(np.mean([r['tokens'][key] for r in received if r['tokens'][key] is not None]))
                 if any(r['tokens'][key] is not None for r in received) else None for key in ('input','output','reasoning','nonreasoning_residual')}
        model_summary = dict(model=model, status=blocked.get(model,dict(status='completed_unreviewed')),
            config_sha256=file_sha256(config), api_base_url=cfg['base_url'],
            selected=manifest['selected_samples_per_model'], ready=manifest['approved_context_samples_per_model'],
            pending_gold_context=32, new_api_calls=calls, reused_responses=sum(bool(r.get('reused')) for r in rows),
            received=len(received), format_valid=sum(r['status'] in ('format_valid_unreviewed','rule_rejected_unreviewed') for r in rows),
            compact_protocol_responses=sum(r['compact_protocol'] for r in rows),
            format_violation_counts=dict(Counter(v for r in rows for v in (r['format_diagnostics'] or {}).get('violations',[]))),
            done_claims=sum(r.get('completion_evidence_diagnostic',{}).get('claims_done',False) for r in rows),
            unsupported_done_claims=sum(r.get('completion_evidence_diagnostic',{}).get('unsupported_done',False) for r in rows),
            rule_pass=sum(r['status']=='format_valid_unreviewed' for r in rows),
            status_counts=dict(Counter(r['status'] for r in rows)),
            native_currency=PRICES[model][0], rates_per_million=dict(input=PRICES[model][1],output=PRICES[model][2]),
            price_source=PRICES[model][3], price_checked_date='2026-10-26',
            priced_responses=len(priced), known_usage_cost=sum(r['cost_native_estimate'] for r in priced),
            new_usage_cost=sum(r['cost_native_estimate'] for r in priced if not r.get('reused')),
            unknown_cost_calls=calls-sum('response.json' in r and not r.get('reused') and r.get('cost_native_estimate') is not None for r in rows),
            tokens_total=totals, tokens_mean=means,
            latency_seconds={f'p{p}':float(np.percentile(latencies,p)) if latencies else None for p in (50,95)},
            cost_per_human_approved_label=None, human_approved=0, semantic_accuracy=None,
            stage1a_200=estimate_cost(rows,dict(counts)),
            stage1a_2000=estimate_cost(rows,{k:v*10 for k,v in counts.items()}))
        if model in blocked:
            model_summary['known_usage_cost'] = None
            model_summary['new_usage_cost'] = 0
            # Provider tokenization and thinking costs are unknown: this is only a budget scenario.
            model_summary['assumed_token_cost_scenarios'] = [dict(input_tokens=6000,output_tokens=out,
                per_frame=(6000*PRICES[model][1]+out*PRICES[model][2])/1e6,
                first_3458=(6000*PRICES[model][1]+out*PRICES[model][2])/1e6*sum(counts.values()),
                extended_34580=(6000*PRICES[model][1]+out*PRICES[model][2])/1e6*sum(counts.values())*10)
                for out in (1000,3000)]
        models.append(model_summary)
    # Lightweight JSON report excludes full base64 API payloads (which are not stored in input.json).
    report = dict(protocol='compact-v2', status='human_review_pending', models=models,
        actual_training_keyframes=dict(counts), actual_training_total=sum(counts.values()),
        expansion_keyframes_estimated=sum(counts.values())*10, human_approved=0,
        future_geometry_teacher_input=False, created_date='2026-10-26',
        cost_caveats=['Undiscounted token-rate estimate, not a provider invoice; includes reasoning output.',
          'No currency conversion. Budget reserves and conservative 1 USD per CNY accounting are not actual USD spend.',
          '2000-episode keyframe count is estimated, not a completed dataset inventory.',
          'Excludes human review, simulation, storage, training and later API repair calls.',
          '32 additional common-context sources/model remain unreviewed and were not submitted.'])
    report_id = fingerprint(dict(sources=[dict(key=r['key'],sha256=r['response_sha256']) for r in reviews],
                                 pack_sha256=file_sha256(root/'source-pack.json')))
    report['report_id'] = report_id
    # Historical calls are a separate experiment/protocol, not new-format measurements.
    prior_path=root.parent/'oracle-live-20261022-v2/comparison-report.json'
    if prior_path.exists():
        prior=read_json(prior_path)
        report['historical_old_protocol_pilot']=dict(path=str(prior_path),sha256=file_sha256(prior_path),
            not_account_lifetime_total=True,
            models=[{k:m.get(k) for k in ('model','actual_api_calls','received_responses',
                'known_usage_cost_estimate','native_currency','priced_responses')}
                for m in prior['models']],
            note='Separate old-protocol pilot; OpenAI failed/interrupted calls may have unknown billed cost. Reservations are not invoices.')
    write_json(root/'review-report.json',report)
    write_json(root/'review-template.json',dict(report_id=report_id,reviewer='',reviews=reviews,approved_for_training=False))
    write_json(root/'sample-results.json',source_rows)
    render(root,report,source_rows,reviews)
    write_markdown(root, report)
    return report


def write_markdown(root, report):
    lines=['# 简化协议 Oracle 人工审核与成本报告','',
        '日期：2026-10-26。所有回答尚未人工批准；格式合法不等于语义正确。',
        '本轮 OpenAI 余额不足，未调用。GLM/Qwen 按相同源帧比较；递归组各自使用自己的上一轮输出。',
        '教师仅生成 Landmark 类别、Local 和自然语言 Route；waypoint/ref_line 从既有轨迹派生，不进入教师输入。','',
        '|模型|新请求/复用|收到响应|格式合法/42|规则通过/42|用量估算费用|200 episode 估算|2,000 episode 估算|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m in report['models']:
        money=lambda x: '未实测' if x is None else f'{x:.4f} {m["native_currency"]}'
        fmt=f'{m["format_valid"]}/42' if m['received'] else '未调用'
        rules=f'{m["rule_pass"]}/42' if m['received'] else '未调用'
        lines.append(f'|{m["model"]}|{m["new_api_calls"]}/{m["reused_responses"]}|{m["received"]}|{fmt}|{rules}|{money(m["known_usage_cost"])}|{money(m["stage1a_200"]["one_attempt_per_frame"])}|{money(m["stage1a_2000"]["one_attempt_per_frame"])}|')
    lines += ['', '## 审核方法','',
        '打开 index.html，先看当前/历史 RGB、截至源时刻的地图和实际输入，再逐项审核地标可见性与用途、Local 依据、Route 最终目标保持与进度、完成依据及未来信息泄漏。',
        '每个模型保留 raw.txt、input.json、response.json、proposal.json、示范几何与规则检查。组装文本中的 waypoint 是程序派生，不是教师输出。',
        '填写姓名、逐字段结论、修订意见，可下载审核 JSON。浏览器草稿不能替代导出文件。导出不自动批准训练标签。',
        '共同上下文组尚有 32 帧/模型等待独立审核；不能把当前 42 帧描述成 74 帧已完成。', '', '## 费用解释','',
        '首批实际为 R2R 1,298 + RxR 2,160 = 3,458 帧；扩展按相同比例预计 34,580 帧。按两个数据来源分别测算单次调用均价后加权。',
        '所有失败响应中可获得的 token 用量均计费；递归失败后未调用的后续样本不算成功或免费标签。计费包括 reasoning，不能只数 raw.txt 字数。',
        '人工审核前无法计算每条审核合格标签成本。预测为每帧一次尝试，另附 20% 返工预算情景，不代表自动重试授权。',
        '价格按标准未缓存费率估算，不是供应商账单；未扣免费额度、缓存或批处理折扣。人民币与美元分别报告。旧实验预算预留不当作已花费用。',
        '少量递归样本的请求长度与思考量不代表全数据集；Route 误差传播及长指令可能增加返工。人工审核、回放渲染、存储、训练费用不含在 API 估算中。','',
        '## OpenAI：仅预算情景，未实测','',
        '以下假设每帧 6,000 输入 tokens、1,000–3,000 输出 tokens（含思考）。不是从 GLM/Qwen 推导的 OpenAI 实际用量。','']
    for m in report['models']:
        if 'assumed_token_cost_scenarios' in m:
            s=m['assumed_token_cost_scenarios'];lines.append(f'- {m["model"]}：200 episode ${s[0]["first_3458"]:.2f}–${s[1]["first_3458"]:.2f}；2,000 episode ${s[0]["extended_34580"]:.2f}–${s[1]["extended_34580"]:.2f}。')
    lines += ['', '## 官方价格','']
    for m in report['models']:
        r=m['rates_per_million'];lines.append(f'- [{m["model"]}]({m["price_source"]})：每百万输入 {r["input"]}、输出 {r["output"]} {m["native_currency"]}。')
    lines += ['', '## 失败与用量诊断','',
        '下面是程序可核验的格式/证据诊断，不替代图像语义审核。Local 上限为90字符；没有截断或修写模型原文来提高通过率。','']
    for m in report['models']:
        if not m['received']:continue
        t=m['tokens_mean'];cost=m['stage1a_200']['review_rework_20_percent_scenario']
        lines.append(f'- {m["model"]}：compact JSON {m["compact_protocol_responses"]}/{m["received"]}；部署格式通过 {m["format_valid"]}/{m["ready"]}（准备就绪全部源帧为分母），其中未调用/失败见状态计数。')
        lines.append(f'  - 格式违规：`{json.dumps(m["format_violation_counts"],ensure_ascii=False)}`；状态：`{json.dumps(m["status_counts"])}`。')
        lines.append(f'  - 平均输入 {t["input"]:.1f}，全部输出 {t["output"]:.1f}，reasoning {t["reasoning"] or 0:.1f} tokens；延迟 p50/p95 {m["latency_seconds"]["p50"]:.1f}/{m["latency_seconds"]["p95"]:.1f} s。')
        lines.append(f'  - DONE 声明 {m["done_claims"]}，缺少指定完成事件的声明 {m["unsupported_done_claims"]}。文字可以在格式失败后继续审核，但不能回写为原始成功结果。')
        if cost is not None:lines.append(f'  - 首批加入 20% 返工预算情景：{cost:.2f} {m["native_currency"]}；不是已经消费。')
    if report.get('historical_old_protocol_pilot'):
        lines += ['', '## 历史试验与本轮分开','',
            '2026-10-22 旧协议试验不并入本轮格式通过率或新格式单价；历史报告不代表整个账号账单。']
        for m in report['historical_old_protocol_pilot']['models']:
            if m.get('priced_responses'):
                lines.append(f'- 旧协议 {m["model"]}：已知用量估算 {m["known_usage_cost_estimate"]:.6f} {m["native_currency"]}。')
        lines += ['- 旧 OpenAI 失败/中断请求的实际账单未核对，不能把内部预留算成已消费；本轮新增 OpenAI 请求为 0。',
                  '- 本轮 Qwen 总样本费用包含一条已付费复用结果；本轮新增费用请看 JSON 的 new_usage_cost。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def render(root, report, sources, reviews):
    def esc(v): return html.escape(str(v),quote=True)
    def pretty(v): return esc(json.dumps(v,ensure_ascii=False,indent=2))
    def asset(path):
        path=Path(path); dest=root/'assets'/(file_sha256(path)+path.suffix.lower())
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists(): shutil.copyfile(path,dest)
        return str(dest.relative_to(root))
    parts=['<!doctype html><meta charset="utf-8"><title>Oracle 人工审核</title>',
      '<style>body{font:16px system-ui;background:#13212c;color:#e5edf3;margin:24px}a{color:#8dd4ff}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0b1720;padding:12px}table{border-collapse:collapse}td,th{border:1px solid #52606d;padding:8px}.columns,.images{display:flex;gap:16px;overflow:auto}.answer{min-width:380px;flex:1;background:#21313f;padding:16px}.images figure{min-width:280px;margin:8px}.images img{max-height:260px;max-width:480px}.source{border-top:2px solid #647787;padding:20px 0}textarea{width:95%;height:75px}select,input,button{padding:7px;margin:4px}label{display:block}summary{cursor:pointer;padding:8px}header{position:sticky;top:0;background:#13212c;z-index:1;padding:8px;border-bottom:1px solid #888}.error{color:#ffb79a}</style>',
      '<h1>Oracle 简化格式：逐帧人工审核</h1><p>所有模型答案均待审核。OpenAI 余额不足，本轮未调用；不存在可供审核的新答案。</p>',
      '<p>10 帧人工 demo + R2R/RxR 各 16 帧递归对照。另 32 帧共同历史上下文未审核，保持暂停。waypoint/ref_line 来自示范轨迹，不交给教师生成。</p>',
      '<p><a href="REPORT.md">说明与成本预测</a> · <a href="review-report.json">统计 JSON</a> · <a href="sample-results.json">全部逐帧记录</a> · <a href="review-template.json">空白审核模板</a></p>',
      '<table><tr><th>模型</th><th>新请求/复用</th><th>收到响应</th><th>格式/42</th><th>规则/42</th><th>已知用量费用</th><th>首批 3,458 帧</th><th>扩展预计 34,580 帧</th></tr>']
    for m in report['models']:
        fmt=lambda n: '未实测' if n is None else f'{n:.4f} {m["native_currency"]}'
        parts.append('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [m['model'],f'{m["new_api_calls"]}/{m["reused_responses"]}',m['received'],m['format_valid'] if m['received'] else '未调用',m['rule_pass'] if m['received'] else '未调用',fmt(m['known_usage_cost']),fmt(m['stage1a_200']['one_attempt_per_frame']),fmt(m['stage1a_2000']['one_attempt_per_frame'])])+'</tr>')
    parts.extend(['</table><p>费用按全部输出（包括 reasoning）估算；未扣缓存优惠，不是账单。格式/规则通过不是语义准确率；人工合格标签成本尚不可计算。</p>',
      '<header>审核人 <input id="reviewer" placeholder="姓名或编号"><button id="download">导出审核 JSON</button><label style="display:inline">导入 <input type="file" id="import" accept=".json"></label><span id="status"></span><br>筛选 <select id="filter"><option value="all">全部源帧</option><option value="manual">人工 demo</option><option value="r2r">R2R</option><option value="rxr">RxR</option><option value="failed">含失败或未调用</option></select></header>',
      '<p>逐项判定：pending 待审，pass 可接受，minor_edit 小修，major_error 严重错误，unscorable 无法判定。文字不同不一定错误；不得仅靠人工原文的字面相似度判错。导出仅保存审核意见，不批准训练标签。</p>'])
    labels=dict(zip(FIELDS,['地标是否可见','地标是否值得登记','Local 是否有依据','Route 是否保留最终目标','Route 进度是否合理','完成声明是否有证据','是否避免未来信息泄漏']))
    for source, answers in sorted(sources.items()):
        available=next((r for r in answers.values() if 'input.json' in r),None)
        failed=any(r['status']!='format_valid_unreviewed' for r in answers.values())
        parts.append(f'<section class="source" data-cohort="{esc(source.split("-")[0])}" data-failed="{str(failed).lower()}"><h2>{esc(source)}</h2>')
        human=next((r['human_reference_for_review_only'] for r in answers.values() if r.get('human_reference_for_review_only')),None)
        if human:
            parts.append('<details><summary>已批准人工参考（仅供审核，未作为当前答案输入教师；不是穷尽物体清单）</summary><pre>'+pretty(human)+'</pre></details>')
        if available:
            parts.append('<details><summary>查看源图像、历史帧与地图（点击展开）</summary><div class="images">')
            for im in available['input.json']['images']:
                url=asset(im['path'])
                parts.append(f'<figure><a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a><figcaption>{esc(im["label"])}</figcaption></figure>')
            parts.append('</div></details>')
            ctx=available['input.json']['context']
            parts.append('<details><summary>任务及输入上下文；递归组请另看每个模型自己的 input.json</summary><pre>'+pretty(ctx)+'</pre></details>')
            if available.get('geometric-supervision.json'):
                parts.append('<details><summary>示范参考线与五个 waypoint（未来轨迹监督，仅供审核，未输入教师）</summary>'+reference_svg(available['geometric-supervision.json'])+'</details>')
        else:
            parts.append('<p>递归链已中断，该源帧未向模型提交；没有实际请求或回答。</p><details><summary>查看待标注源图（不是已发送请求）</summary><div class="images">')
            for im in next(iter(answers.values())).get('prepared_source_images_not_submitted',[]):
                url=asset(im['path']);parts.append(f'<figure><a href="{url}" target="_blank"><img loading="lazy" src="{url}"></a><figcaption>{esc(im["label"])}</figcaption></figure>')
            parts.append('</div></details>')
        parts.append('<div class="columns">')
        for model in ('glm-5.3-flashx','qwen3.8-max'):
            r=answers.get(model)
            if not r:parts.append('<article class="answer"><h3>'+esc(model)+'</h3>无结果</article>');continue
            parts.append('<article class="answer"><h3>'+esc(model)+'</h3><p>'+esc(r['status'])+' '+esc(r.get('reason',''))+'</p>')
            if r.get('reused'):parts.append('<p>复用完全相同请求的已付费结果，本轮未重复调用。</p>')
            parts.append('<pre>'+esc(r.get('raw.txt','无模型原文，未补造回答。'))+'</pre>')
            parts.append('<p>'+esc(dict(tokens=r['tokens'],elapsed_seconds=r.get('elapsed_seconds'),cost_CNY=r.get('cost_native_estimate')) )+'</p>')
            if r['format_diagnostics']:
                parts.append('<p class="error">'+esc(r['format_diagnostics'])+'</p>')
            if r.get('completion_evidence_diagnostic',{}).get('claims_done'):
                parts.append('<p class="error">'+esc(r['completion_evidence_diagnostic'])+'</p>')
            for name in ('input.json','raw.txt','response.json','proposal.json','comparison.json','geometric-supervision.json'):
                if name in r:parts.append(f'<a href="{esc(r["artifact"]+"/"+name)}" target="_blank">{name}</a> ')
            if 'input.json' in r:
                parts.append('<details><summary>该模型实际输入（含自己的上一轮 Route / Local）</summary><pre>'+pretty(r['input.json'])+'</pre></details>')
            if 'proposal.json' in r:
                parts.append('<details><summary>完整组装文本（waypoint 非模型生成）与程序检查</summary><pre>'+esc(r['proposal.json']['text'])+'</pre><pre>'+pretty(r.get('comparison.json'))+'</pre></details>')
            if r['response_sha256']:
                parts.append('<div class="review" data-key="'+esc(r['review_key'])+'">')
                for f in FIELDS+['overall']:
                    parts.append('<label>'+labels.get(f,'整体结论')+'<select data-field="'+f+'">'+''.join('<option>'+v+'</option>' for v in VERDICTS)+'</select></label>')
                parts.append('<label>理由 / 修改建议<textarea data-field="notes"></textarea></label><label>修订后的三字段 JSON（可选，保留原文）<textarea data-field="corrected_json"></textarea></label></div>')
            parts.append('</article>')
        parts.append('</div></section>')
    template=dict(report_id=report['report_id'],reviewer='',reviews=reviews,approved_for_training=False)
    parts.append('<script id="template" type="application/json">'+json.dumps(template,ensure_ascii=False).replace('<','\\u003c')+'</script>')
    parts.append(SCRIPT)
    (root/'index.html').write_text(''.join(parts))


SCRIPT = r'''<script>
const template=JSON.parse(document.getElementById('template').textContent), storage='oracle-review:'+template.report_id;
let state=structuredClone(template);const fields=Object.keys(template.reviews[0]?.fields||{}), allowed=['pending','pass','minor_edit','major_error','unscorable'];
const status=document.getElementById('status');
function validate(s){if(s.report_id!==template.report_id)throw Error('报告 ID 不一致');const seen=new Set();for(const r of s.reviews){const t=template.reviews.find(t=>t.key===r.key);if(!t||seen.has(r.key)||t.response_sha256!==r.response_sha256)throw Error('源帧/响应哈希不匹配');seen.add(r.key);if(Object.keys(r.fields).sort().join()!=[...fields].sort().join()||!allowed.includes(r.overall)||Object.values(r.fields).some(v=>!allowed.includes(v)))throw Error('非法审核值');if(r.overall!=='pending'&&!s.reviewer.trim())throw Error('请填写审核人');if(r.overall==='pass'&&Object.values(r.fields).some(v=>!['pass','unscorable'].includes(v)))throw Error('整体通过前请完成逐字段审核');}}
function populate(){document.getElementById('reviewer').value=state.reviewer;document.querySelectorAll('.review').forEach(el=>{let r=state.reviews.find(r=>r.key===el.dataset.key);if(!r)return;el.querySelectorAll('[data-field]').forEach(i=>{let f=i.dataset.field;i.value=fields.includes(f)?r.fields[f]:(f==='corrected_json'?(r[f]?JSON.stringify(r[f],null,2):''):(r[f]||''));});});}
function save(){state.reviewer=document.getElementById('reviewer').value;state.updated_at=new Date().toISOString();state.approved_for_training=false;try{localStorage.setItem(storage,JSON.stringify(state));status.textContent='本地草稿已保存；请导出留档';}catch(e){status.textContent='本地存储不可用，请导出';}}
try{const saved=JSON.parse(localStorage.getItem(storage)||'null');if(saved){validate(saved);state=saved;}}catch(e){status.textContent=e.message;}populate();
document.querySelectorAll('.review').forEach(el=>el.addEventListener('change',e=>{const f=e.target.dataset.field;if(!f)return;let r=state.reviews.find(r=>r.key===el.dataset.key);if(fields.includes(f))r.fields[f]=e.target.value;else if(f==='corrected_json'){try{r[f]=e.target.value.trim()?JSON.parse(e.target.value):null;}catch(err){status.textContent='修订 JSON 非法，尚未保存';return;}}else r[f]=e.target.value;save();}));
document.getElementById('reviewer').addEventListener('input',save);
document.getElementById('download').onclick=()=>{try{document.querySelectorAll('.review').forEach(el=>{const r=state.reviews.find(r=>r.key===el.dataset.key);const value=el.querySelector('[data-field="corrected_json"]').value.trim();r.corrected_json=value?JSON.parse(value):null;});save();validate(state);const u=URL.createObjectURL(new Blob([JSON.stringify(state,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=u;a.download='oracle-human-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}catch(e){status.textContent=e.message;}};
document.getElementById('import').onchange=async e=>{try{const s=JSON.parse(await e.target.files[0].text());validate(s);state={...template,...s,reviews:template.reviews.map(t=>s.reviews.find(r=>r.key===t.key)||t),approved_for_training:false};populate();save();}catch(err){status.textContent=err.message;}};
document.getElementById('filter').onchange=e=>document.querySelectorAll('.source').forEach(s=>s.hidden=!(e.target.value==='all'||s.dataset.cohort===e.target.value||(e.target.value==='failed'&&s.dataset.failed==='true')));
</script>'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--stage-b');p.add_argument('--review')
    a=p.parse_args()
    if a.review:
        result=validate_reviews(read_json(a.review),read_json(Path(a.root)/'review-template.json'))
        print(json.dumps(dict(valid=True,reviewer=result['reviewer'],counts=dict(Counter(r['overall'] for r in result['reviews'])),approved_for_training=False)))
    else:
        if not a.stage_b:p.error('--stage-b is required for report generation')
        build(a.root,a.stage_b)
