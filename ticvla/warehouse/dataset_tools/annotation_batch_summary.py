"""Whole-task annotation counts; mechanical checks never imply semantic approval."""
from collections import Counter
import csv
import html
import json
from pathlib import Path
import re
import time


def read(path, fallback=None):
    return json.loads(path.read_text()) if path.exists() else fallback


def summarize(root, phase='running', deep=False):
    root=Path(root)
    manifest=read(root/'pilot.json')
    cfg=read(root/'teacher-config.json')
    tasks=[];frames=[];failures=Counter();risks=Counter()
    for entry in manifest['episodes']:
        folder=Path(entry['folder'])/'plus'
        report=read(folder/'report.json',{})
        samples={s['source_image_id']:s for s in report.get('samples',[])}
        counts=Counter();previous=entry['instruction'];issues=[]
        for index in entry['source_indices']:
            image_id=f"{entry['uid']}:{index:08d}"
            s=samples.get(image_id,{})
            current=folder/s.get('folder','NOT_STARTED')
            returned=(current/'response.json').exists()
            valid=s.get('status')=='format_valid_unreviewed'
            flags=[];contract_errors=[]
            row=dict(episode=entry['uid'],frame=index,image_id=image_id,
                requested=bool(s.get('request_sent',False)),returned=returned,rule_valid=valid,
                status=s.get('status','not_started'),reason=s.get('reason'),
                cost_cny=s.get('cost_cny_estimate',0) or 0,
                context_source=s.get('context_recovery',{}).get('context_source_image_id'),
                initial_route_fallback=s.get('context_recovery',{}).get('original_instruction_if_no_valid_response',False))
            if deep:
                inp=read(current/'input.json')
                if inp:
                    if 'instruction' in inp.get('context',{}) or any(l.startswith('Instruction:') for l in inp['user_text'].splitlines()):
                        contract_errors.append('UNEXPECTED_INSTRUCTION_INPUT')
                    if inp.get('instructions')!=cfg['teacher_instruction']:contract_errors.append('PROMPT_MISMATCH')
                    if inp.get('context',{}).get('route')!=previous:contract_errors.append('ROUTE_RECURSION_MISMATCH')
                p=current/'raw.txt'
                answer=None
                if p.exists():
                    try:answer=json.loads(p.read_text())
                    except ValueError:pass
                if isinstance(answer,dict):
                    row['answer']=answer
                    route=answer.get('route','');local=answer.get('local','')
                    row['local_length']=len(local) if isinstance(local,str) else None
                    row['route_length']=len(route) if isinstance(route,str) else None
                    if isinstance(route,str):
                        if re.fullmatch(r'\s*(?:stop|wait)[.!]?\s*',previous,re.I) and not re.fullmatch(r'\s*(?:stop|wait|DONE)[.!]?\s*',route,re.I):
                            flags.append('STOP_WAIT_TO_NEW_COMMAND_REVIEW')
                        if re.search(r'\bstop\b',previous,re.I) and not re.search(r'\bstop\b',route,re.I) and route!='DONE':
                            flags.append('STOP_REQUIREMENT_DISAPPEARED_REVIEW')
                        if valid:previous=route
                    for key in ['local','route']:
                        if isinstance(answer.get(key),str) and re.search(r"[^A-Za-z0-9 ,.'\s]",answer[key]):
                            flags.append(key.upper()+'_PUNCTUATION_STYLE_REVIEW')
                row['heuristic_review_flags']=flags
                row['input_contract_errors']=contract_errors
                risks.update(flags);issues.extend(contract_errors)
            frames.append(row)
            for key in ['requested','returned','rule_valid']:counts[key]+=row[key]
            if row['status']=='failed':counts['failed']+=1;failures.update([row['reason'] or 'unspecified'])
            counts['initial_route_fallback']+=bool(row['initial_route_fallback'])
        n=entry['frames'];valid=counts['rule_valid'];returned=counts['returned']
        tasks.append(dict(episode=entry['uid'],scene_id=entry['scene_id'],planned=n,
            requested=counts['requested'],returned=returned,rule_valid=valid,failed=counts['failed'],
            unreturned=n-returned,valid_rate=valid/n,all_returned=returned==n,
            all_frames_rule_valid=valid==n,initial_route_fallback_frames=counts['initial_route_fallback'],
            status=report.get('status','not_started'),input_contract_errors=issues,
            cost_cny=sum(f['cost_cny'] for f in frames if f['episode']==entry['uid']),
            human_approved=0,semantic_success=None))
    total=dict(phase=phase,updated_unix=time.time(),planned_tasks=len(tasks),
        buildings=len({e['scene_id'] for e in tasks}),planned_frames=sum(t['planned'] for t in tasks),
        requested=sum(t['requested'] for t in tasks),returned=sum(t['returned'] for t in tasks),
        rule_valid=sum(t['rule_valid'] for t in tasks),failed=sum(t['failed'] for t in tasks),
        tasks_all_responses=sum(t['all_returned'] for t in tasks),
        tasks_all_frames_rule_valid=sum(t['all_frames_rule_valid'] for t in tasks),
        tasks_partial_rule_valid=sum(0<t['rule_valid']<t['planned'] for t in tasks),
        tasks_zero_rule_valid=sum(t['rule_valid']==0 for t in tasks),
        tasks_not_started=sum(t['requested']==0 for t in tasks),
        tasks_at_least_90_percent_rule_valid=sum(t['valid_rate']>=.9 for t in tasks),
        cost_cny_estimate=sum(t['cost_cny'] for t in tasks),failure_reasons=dict(failures),
        human_approved=0,semantic_success_tasks=None,
        semantic_note='Rule-valid is not semantic correctness. Route preservation, landmark visibility and Local geometry require review.',
        heuristic_flags=dict(risks),heuristic_note='Review candidates only; not automatically counted as semantic failures.',
        input_contract_errors=sum(len(t['input_contract_errors']) for t in tasks) if deep else None)
    total['frame_rule_valid_rate']=total['rule_valid']/total['planned_frames']
    total['task_all_frames_rule_valid_rate']=total['tasks_all_frames_rule_valid']/total['planned_tasks']
    result=dict(summary=total,tasks=tasks)
    (root/'task-statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (root/'task-statistics.csv').open('w') as f:
        columns=['episode','scene_id','planned','requested','returned','rule_valid','failed','unreturned','valid_rate','all_frames_rule_valid','initial_route_fallback_frames','status','cost_cny','human_approved','semantic_success']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(tasks)
    if deep:
        (root/'frame-statistics.json').write_text(json.dumps(frames,ensure_ascii=False,indent=2)+'\n')
    esc=html.escape
    parts=['<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>120任务标注统计</title>',
        '<style>body{font:16px system-ui;margin:28px}td,th{padding:7px;border:1px solid #bbb}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
        f'<h1>{esc(cfg.get("model","unspecified"))} · {len(tasks)}任务标注统计</h1>',
        '<p>成功口径：整任务全部预定帧通过程序规则。此指标不等于语义正确或任务完成；尚无人工批准。新旧原文均保留，不补问或修复答案。</p>',
        '<p><a href="task-statistics.csv">逐任务CSV</a> · <a href="task-statistics.json">完整统计JSON</a> · <a href="index.html">任务视频与完整审核（生成后）</a> · <a href="teacher-config.json">冻结配置</a></p>',
        '<pre>'+esc(json.dumps(total,ensure_ascii=False,indent=2))+'</pre>',
        '<table><tr><th>任务</th><th>返回/计划</th><th>规则通过/计划</th><th>整任务规则通过</th><th>失败</th><th>费用</th><th>状态</th></tr>']
    for t in tasks:
        folder=root/'episodes'/t['episode'];page='review/index.html' if (folder/'review/index.html').exists() else 'plus/index.html'
        name=f'<a href="episodes/{t["episode"]}/{page}">{t["episode"]}</a>' if (folder/page).exists() else t['episode']
        parts.append(f'<tr><td>{name}</td><td>{t["returned"]}/{t["planned"]}</td><td>{t["rule_valid"]}/{t["planned"]}</td><td>{t["all_frames_rule_valid"]}</td><td>{t["failed"]}</td><td>{t["cost_cny"]:.4f}</td><td>{esc(t["status"])}</td></tr>')
    parts.append('</table>');tmp=root/'statistics.html.tmp';tmp.write_text('\n'.join(parts));tmp.replace(root/'statistics.html')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--phase',default='completed_unreviewed');p.add_argument('--deep',action='store_true');a=p.parse_args()
    print(json.dumps(summarize(a.root,a.phase,a.deep)['summary'],ensure_ascii=False))
