"""Fixed five-provider comparison packs; no semantic labels are auto-approved."""
import argparse
import copy
import html
import json
from pathlib import Path
import numpy as np
from .core import read_json,write_json
from .semantic_annotation import write_bundle,validate_sample,run_pilot
from ..schema import file_sha256


def teacher_sample(record,human=False):
    r=copy.deepcopy(record);sup=r['supervision']
    sample=dict(observation=r['observation'],provenance=r['provenance'],
        geometric_supervision={k:sup[k] for k in ('reference_xy','reference_mask','reference_state','waypoint')})
    sample['geometric_supervision'].update(source='approved_human_demonstration' if human else 'verified_dataset_action_replay',
        approved=human,supervision_for_teacher=False)
    if human:sample['evaluation_only']=dict(human_text=sup['parsed'],reference_is_not_exhaustive_object_inventory=True)
    return validate_sample(sample)


def prepare(manual_export,dataset,output):
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    b=read_json(dataset/'report.json')
    if not b['passed']:raise ValueError('Stage B must pass first')
    bundles=[]
    def add(name,samples,mode,ready):
        folder=output/'bundles'/name
        write_bundle(folder,samples,source=name,context_mode=mode)
        bundles.append(dict(name=name,path=str(folder.resolve()),manifest_sha256=file_sha256(folder/'manifest.json'),
            samples=len(samples),context_ready=ready,reason=None if ready else 'common prior Route/Local needs independent human review'))
        if not ready:
            write_json(folder/'context-review-required.json',dict(approved=False,reviewer=None,
                instruction='Review common causal prior Route/Local for these sources before any gold-context calls; current answers are not input.',
                samples=[dict(image_id=s['observation']['image_id'],source_time=s['observation']['t_obs'],
                              current_route=s['observation']['route']) for s in samples]))
    manual_export=Path(manual_export);m=read_json(manual_export/'manifest.json')
    manual=[teacher_sample(read_json(manual_export/e['path']),True) for e in m['records']]
    add('manual-0',manual[:5],'gold_context',True);add('manual-1',manual[5:],'gold_context',True)
    for r in b['results']:
        if r['role']!='oracle_comparison':continue
        samples=[teacher_sample(read_json(e['path'])) for e in r['records']]
        n=len(samples)
        if n<8:raise ValueError('Fixed comparison episode has fewer than eight keyframes')
        start=(n-4)//2;recursive=list(range(start,start+4))
        remaining=[i for i in range(n) if i not in recursive]
        gold=[remaining[i] for i in np.linspace(0,len(remaining)-1,4,dtype=int)]
        add(r['uid']+'-recursive',[samples[i] for i in recursive],'recursive',True)
        add(r['uid']+'-gold',[samples[i] for i in gold],'gold_context',False)
    configs=[]
    for model,provider,base,key,inp,out in [
        ('gpt-6-luna','responses','https://api.openai.com/v1','OPENAI_API_KEY',.1,.5),
        ('gpt-6.1-sol','responses','https://api.openai.com/v1','OPENAI_API_KEY',2.,10.),
        ('gpt-6-astra','responses','https://api.openai.com/v1','OPENAI_API_KEY',10.,50.),
        ('glm-5.3-flashx','chat_completions','https://api.z.ai/api/paas/v4','ZAI_API_KEY',.37,1.25),
        ('qwen3.8-max','chat_completions',None,'DASHSCOPE_API_KEY',2.,6.)]:
        cfg=dict(model=model,api=provider,base_url=base,api_key_env=key,max_output_tokens=8192,timeout_seconds=180,
            budget_file=str(output/'budget.json'),budget_limit_usd=100.,input_token_reservation=131072,
            output_token_reservation=16384,usd_per_million_tokens=dict(input=inp,output=out),
            price_checked_date='2026-10-21',account_access_verified=False,
            price_source=('https://developers.openai.com/api/docs/pricing' if provider=='responses' else
                'https://docs.z.ai/guides/overview/pricing' if model.startswith('glm') else
                'https://www.alibabacloud.com/help/en/model-studio/model-pricing'))
        if provider=='responses':cfg['reasoning_effort']='medium'
        if model.startswith('glm'):cfg.update(thinking=dict(type='enabled'),reasoning_effort='max')
        if model.startswith('qwen'):cfg.update(enable_thinking=True,thinking_budget=8192,region='Singapore price assumption; confirm endpoint and region before call')
        path=output/'configs'/f'{model}.json';write_json(path,cfg);configs.append(str(path))
    report=dict(status='prepared_no_api_calls',actual_api_calls=0,bundles=bundles,configs=configs,
        selected_samples_per_model=sum(b['samples'] for b in bundles),models=5,
        approved_context_samples_per_model=sum(b['samples'] for b in bundles if b['context_ready']),
        pending_gold_context_review=32,semantic_labels_approved=0,
        stage_B_report=str(dataset/'report.json'),stage_B_report_sha256=file_sha256(dataset/'report.json'))
    write_json(output/'comparison.json',report)
    (output/'index.html').write_text('<meta charset="utf-8"><h1>Five-model Oracle comparison: prepared, not executed</h1><p>74 sources/model: 10 manual, 32 recursive, 32 common-context review pending. No semantic accuracy measured.</p><a href="comparison.json">Manifest</a><ul>'+''.join(f'<li>{b["name"]}: {b["samples"]} sources; context_ready={b["context_ready"]}</li>' for b in bundles)+'</ul>')
    return report


def run(pack,config,output,prepare_only=False,resume=False,max_bundles=None):
    pack=Path(pack);m=read_json(pack/'comparison.json');cfg=read_json(config)
    if not cfg.get('budget_file'):raise ValueError('Shared budget ledger required')
    if file_sha256(m['stage_B_report'])!=m['stage_B_report_sha256']:raise ValueError('Stage B report changed')
    if max_bundles is not None and max_bundles<1:raise ValueError('Positive max_bundles required')
    output=Path(output);output.mkdir(parents=True,exist_ok=resume);results=[];executed=0;provider_failed=False
    binding=dict(config_sha256=file_sha256(config),pack_sha256=file_sha256(pack/'comparison.json'),prepare_only=prepare_only)
    if (output/'run-binding.json').exists():
        if read_json(output/'run-binding.json')!=binding:raise ValueError('Resume configuration or source changed')
    else:write_json(output/'run-binding.json',binding)
    for b in m['bundles']:
        if file_sha256(Path(b['path'])/'manifest.json')!=b['manifest_sha256']:raise ValueError('Comparison source changed')
        if not b['context_ready']:
            results.append(dict(bundle=b['name'],status='not_run_gold_context_review_required',selected_samples=b['samples']));continue
        if not cfg.get('base_url'):
            results.append(dict(bundle=b['name'],status='not_run_endpoint_required',selected_samples=b['samples']));continue
        folder=output/b['name']
        if folder.exists():
            if not resume or not (folder/'report.json').is_file():raise ValueError('Incomplete existing bundle; inspect before retry')
            r=read_json(folder/'report.json')
            if (r.get('status')=='running' or r.get('config_sha256')!=binding['config_sha256']
                or r.get('bundle_sha256')!=b['manifest_sha256']):raise ValueError('Cannot safely resume this bundle')
        elif provider_failed or (max_bundles is not None and executed>=max_bundles):
            results.append(dict(bundle=b['name'],status='not_run_provider_error' if provider_failed else 'not_run_bundle_limit',selected_samples=b['samples']));continue
        else:
            r=run_pilot(b['path'],config,folder,prepare_only=prepare_only);executed+=1
        if any(x['status'] in ('not_run_provider_error','not_sent_preflight_rejected') or
               str(x.get('reason','')).startswith('HTTP ') for x in r['samples']):provider_failed=True
        results.append(dict(bundle=b['name'],status=r['status'],actual_api_calls=r['actual_api_calls'],selected_samples=b['samples']))
    write_json(output/'report.json',dict(model=cfg['model'],results=results,
        actual_api_calls=sum(r.get('actual_api_calls',0) for r in results),human_review_complete=False,
        selected_denominator=m['selected_samples_per_model'],semantic_accuracy=None))
    return read_json(output/'report.json')


def summarize(pack,run_root):
    """Summarize completed and failed requests without granting semantic approval."""
    pack=Path(pack);root=Path(run_root);manifest=read_json(pack/'comparison.json');models=[]
    for config in sorted((root/'configs').glob('*.json')):
        cfg=read_json(config);folder=root/cfg['model'];rows=[];bundles=[]
        # Interface probes have a root report; full comparisons have bundle reports.
        reports=list(folder.glob('*/report.json'))
        if not reports and (folder/'report.json').exists():reports=[folder/'report.json']
        for path in reports:
            r=read_json(path)
            if 'samples' not in r:continue
            for row in r['samples']:
                rows.append(dict(row,artifact=str((path.parent/row['folder']).relative_to(root))))
            bundles.append(dict(path=str(path.relative_to(root)),actual_api_calls=r['actual_api_calls']))
        finished=[r for r in rows if r.get('elapsed_seconds') is not None]
        latencies=[r['elapsed_seconds'] for r in finished]
        valid=[r for r in rows if r['status'] in ('format_valid_unreviewed','rule_rejected_unreviewed')]
        cohorts={}
        for cohort in ('manual','r2r','rxr'):
            selected=[b for b in manifest['bundles'] if b['name'].startswith(cohort+'-')]
            subset=[r for r in rows if r['source_image_id'].startswith(cohort+'-')]
            cohorts[cohort]=dict(selected_total=sum(b['samples'] for b in selected),
                context_ready=sum(b['samples'] for b in selected if b['context_ready']),
                pending_gold_context=sum(b['samples'] for b in selected if not b['context_ready']),
                format_valid=sum(r['status'] in ('format_valid_unreviewed','rule_rejected_unreviewed') for r in subset),
                rule_pass=sum(r['status']=='format_valid_unreviewed' for r in subset),
                failed=sum(r['status']=='failed' for r in subset),
                stopped_recursive=sum(r['status']=='not_run_previous_proposal_failed' for r in subset))
        models.append(dict(model=cfg['model'],bundles=bundles,samples=rows,
            cohorts=cohorts,
            actual_api_calls=sum(b['actual_api_calls'] for b in bundles),
            format_valid=len(valid),rule_pass=sum(r['status']=='format_valid_unreviewed' for r in rows),
            selected_denominator=manifest['selected_samples_per_model'],
            ready_context_denominator=manifest['approved_context_samples_per_model'],
            received_responses=sum((root/r['artifact']/'response.json').exists() for r in rows),
            latency_including_failures_seconds={f'p{p}':float(np.percentile(latencies,p)) if latencies else None for p in (50,95)},
            native_currency=cfg.get('native_currency','USD'),
            priced_responses=sum(r.get('cost_native_estimate',r.get('cost_estimate_usd')) is not None for r in rows),
            known_usage_cost_estimate=sum(r.get('cost_native_estimate',r.get('cost_estimate_usd',0)) or 0 for r in rows),
            cost_is_not_invoice=True,semantic_accuracy=None,human_review_complete=False))
    report=dict(status='partial_comparison_unreviewed',models=models,
        pending_gold_context_review=manifest['pending_gold_context_review'],
        teacher_labels_approved=0,stage_D_started=False,
        note='No winner selected. Format success, exact wording agreement and semantic correctness differ. Missing API results remain missing.')
    budget=pack/'budget.json'
    if budget.exists():
        ledger=read_json(budget)
        report['all_attempts_budget']=dict(limit_usd=ledger['limit_usd'],http_submissions=len(ledger['calls']),
            accounted_or_reserved_usd=sum(r['accounted_usd'] for r in ledger['calls'].values()),
            unresolved_outcomes=sum(r['status']=='reserved_or_outcome_unknown' for r in ledger['calls'].values()),
            unreconciled_cost_reservations=sum(r['status']!='usage_received' for r in ledger['calls'].values()),
            is_invoice=False)
    if (root/'provider-status.json').exists():report['provider_status']=read_json(root/'provider-status.json')
    write_json(root/'comparison-report.json',report)
    parts=['<meta charset="utf-8"><title>Oracle comparison</title>',
        '<style>body{font:16px sans-serif;max-width:1200px;margin:2em auto}td,th{padding:.6em;border:1px solid #ccc}pre{white-space:pre-wrap}img{max-width:95%}</style>',
        '<h1>Oracle 标注对照：部分执行，尚未人工批准</h1>',
        '<p>共同历史上下文未经人工审核的组保持暂停。waypoint 和参考线来自示范轨迹，未交给教师生成。</p>',
        '<p>传输编码、端点和费率见各模型 configs；原图哈希和传输哈希见 input.json。格式通过不等于语义正确。费用为用量估算，不是账单；预算还包含失败请求的预留。</p>',
        '<pre>'+html.escape(json.dumps(report.get('provider_status',{}),ensure_ascii=False,indent=2))+'</pre>',
        '<p><a href="comparison-report.json">完整报告</a></p><table><tr><th>模型</th><th>请求</th><th>收到响应</th><th>格式通过</th><th>已知用量费用</th></tr>']
    for m in models:
        cost=f'{m["known_usage_cost_estimate"]:.4f} {m["native_currency"]}' if m['priced_responses'] else '无用量信息；不是零费用结论'
        parts.append(f'<tr><td>{html.escape(m["model"])}</td><td>{m["actual_api_calls"]}</td><td>{m["received_responses"]}</td><td>{m["format_valid"]}</td><td>{cost}</td></tr>')
    parts.append('</table>')
    for m in models:
        parts.append('<h2>'+html.escape(m['model'])+'</h2>')
        for b in m['bundles']:
            page=str(Path(b['path']).with_name('index.html'))
            if (root/page).exists():parts.append('<p><a href="'+html.escape(page)+'">'+html.escape(Path(b['path']).parent.name)+'：图像与完整回答</a></p>')
        for r in m['samples']:
            link=r['artifact'];parts.append('<details><summary>'+html.escape(r['source_image_id']+' — '+r['status']+' '+str(r.get('reason','')))+'</summary>')
            for filename in ('input.json','response.json','raw.txt','proposal.json','comparison.json','http-error.txt'):
                if (root/link/filename).exists():parts.append(f'<a href="{html.escape(link+"/"+filename)}">{filename}</a> ')
            if (root/link/'raw.txt').exists():parts.append('<pre>'+html.escape((root/link/'raw.txt').read_text())+'</pre>')
            parts.append('</details>')
    by_source={}
    for m in models:
        for row in m['samples']:by_source.setdefault(row['source_image_id'],{})[m['model']]=row
    parts.append('<h2>按源帧并排审核</h2><p>仅展示保存结果；空缺与失败不会补造回答。</p>')
    for source,answers in sorted(by_source.items()):
        parts.append('<details><summary>'+html.escape(source)+'</summary>')
        previews=[p for row in answers.values() for p in (root/row['artifact']).glob('source.*')]
        if previews:parts.append('<img loading="lazy" src="'+html.escape(str(previews[0].relative_to(root)))+'">')
        parts.append('<div style="display:flex;gap:1em;overflow:auto">')
        for model,row in answers.items():
            folder=root/row['artifact'];raw=folder/'raw.txt'
            answer=raw.read_text() if raw.exists() else row.get('reason',row['status'])
            parts.append('<div style="min-width:320px;flex:1"><h3>'+html.escape(model)+'</h3><p>'+html.escape(row['status'])+'</p><pre>'+html.escape(answer)+'</pre></div>')
        parts.append('</div></details>')
    for name in ('manual-field-agreement.json','protocol-and-review-notes.json','assistant-visual-checks.json',
                 'geometry-consistency-review.json','input-fairness.json','teacher-input-leakage-check.json'):
        if (root/name).exists():parts.append('<p><a href="'+name+'">'+name+'</a></p>')
    (root/'index.html').write_text(''.join(parts))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','run','summarize'])
    p.add_argument('--output',required=True);p.add_argument('--manual-export');p.add_argument('--dataset')
    p.add_argument('--pack');p.add_argument('--config');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--credentials-file');p.add_argument('--resume',action='store_true');p.add_argument('--max-bundles',type=int)
    a=p.parse_args()
    if a.credentials_file:
        from .credentials import load_credentials
        load_credentials(a.credentials_file)
    if a.mode=='prepare':prepare(a.manual_export,a.dataset,a.output)
    elif a.mode=='summarize':summarize(a.pack,a.output)
    else:run(a.pack,a.config,a.output,a.prepare_only,a.resume,a.max_bundles)


if __name__=='__main__':main()
