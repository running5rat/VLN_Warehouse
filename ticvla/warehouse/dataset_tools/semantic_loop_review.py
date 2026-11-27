"""Read-only repackaging of semantic-loop runs to the repository Agent.md standard.

No model/API calls; source artifacts are hashed and never rewritten. Video is
recorded RGB replay aligned to source time, not a simulation of API response time.
"""
from ..compact_input import audit_context, model_user_text

import argparse
import base64
from collections import Counter
import copy
import hashlib
import html
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
import textwrap
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def stats(values):
    return dict(n=len(values),p50=float(np.percentile(values,50)) if values else None,
                p95=float(np.percentile(values,95)) if values else None)


def rel(path, root):
    return os.path.relpath(path,root)


def detail(title,value):
    return '<details><summary>'+html.escape(title)+'</summary><pre>'+html.escape(
        value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'


def frame_at(indices, current):
    return int(np.searchsorted(indices,current,side='right')-1)


def build(source, output, reviews):
    source=Path(source).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();decisions=read(reviews)
    manifest=read(source/'perception/manifest.json')
    sample0=read(source/'perception'/manifest['records'][0]['folder']/'sample.json')
    raw=Path(manifest['source_episode']).parent;episode=read(raw/'episode.json')
    capture=[json.loads(s) for s in (raw/episode['frames_file']).read_text().splitlines()]
    keyframes=[];archive={};models={};rows=[];humans=[];decisions_by={(r['variant'],r['image_id']):r for r in decisions}
    def archive_file(path):
        path=Path(path);archive[str(path)]=sha(path)
    for f in [source/'perception/manifest.json',raw/'episode.json',raw/episode['frames_file'],Path(reviews)]:archive_file(f)
    sources=[]
    for i,e in enumerate(manifest['records']):
        folder=source/'perception'/e['folder'];sample=read(folder/'sample.json');p=read(folder/'perception.json')
        for name,key in [('sample.json','sample_sha256'),('perception.json','perception_sha256'),('prior-geometry.npz','geometry_sha256')]:
            if sha(folder/name)!=e[key]:raise ValueError('Prepared source changed')
            archive_file(folder/name)
        s=p['source'];index=sample['provenance']['frame_index']
        for name in ['rgb','depth']:
            if sha(s[name])!=s[name+'_hash']:raise ValueError('Changed source '+name)
        for inst in p['instances']:
            if sha(inst['mask_path'])!=inst['mask_sha256']:raise ValueError('Changed frozen mask')
            archive_file(inst['mask_path'])
        if capture[index]['t_obs']!=s['t_obs']:raise ValueError('Video/source clock mismatch')
        t0=sample0['observation']['t_obs'];prev=sources[-1][1]['source'] if sources else None
        motion=np.linalg.inv(np.asarray(prev['T_odom_body']))@np.asarray(s['T_odom_body']) if prev else np.eye(4)
        k=dict(index=i,frame_index=index,image_id=s['image_id'],source_time=s['t_obs'],episode_elapsed_s=s['t_obs']-t0,
            delta_time_s=s['t_obs']-prev['t_obs'] if prev else None,
            relative_previous_body=dict(x_m=float(motion[0,3]),y_m=float(motion[1,3]),yaw_rad=float(math.atan2(motion[1,0],motion[0,0]))),
            events=sample['observation']['execution']['events'])
        keyframes.append(k);sources.append((sample,p))
        for file in [s['rgb'],s['depth'],p['candidate_request']['overlay']]:archive_file(file)
    write(out/'keyframes.json',keyframes)
    for variant in (read(source/'experiment.json').get('variants', ['plus','max']) if (source/'experiment.json').exists() else ['plus','max']):
        r=read(source/variant/'report.json');cfg=read(source/variant/'config.json')
        archive_file(source/variant/'report.json');archive_file(source/variant/'config.json')
        report=dict(model=cfg['model'],config=cfg,planned=len(keyframes),requested=r['actual_api_calls'],returned=0,
            format_valid=0,not_run=0,human_reviewed=0,human_approved=0,assistant_reviewed=0,
            historic_cost_cny=r['cost_cny_estimate'],new_api_calls=0,fields={},timings={},
            server_model_revision='not_recorded',request_wall_times='not_recorded; response.created is provider metadata, not measured send/receive times')
        report['final_map']=read(source/variant/'final-map.json');report['causality_audit']=read(source/variant/'causality-audit.json')
        for f in ['final-map.json','causality-audit.json']:archive_file(source/variant/f)
        timings={k:[] for k in ['api_wall_s','vlm_request_and_validation_s','segmentation_s','ocr_s','map_update_s','failed_total_s']}
        token_totals=Counter();field_counts={f:Counter() for f in ['landmark','local','route']}
        for i,(orig,k,(sample,p)) in enumerate(zip(r['samples'],keyframes,sources)):
            folder=source/variant/orig['folder'];dest=out/'frames'/f'{variant}-{k["frame_index"]:08d}';dest.mkdir(parents=True)
            if orig['source_image_id']!=k['image_id']:raise ValueError('Model keyframe order mismatch')
            row=dict(copy.deepcopy(orig),variant=variant,model=cfg['model'],keyframe=k,
                directory=rel(dest,out),input=None,answer=None,raw=None,response_sha256=None,
                downstream=None,source_artifact=rel(folder,out),waypoint='not_applicable',reference='not_applicable')
            for name in ['input.json','sample.json','response.json','raw.txt','proposal.json','checks.json','map-after.json','map-after.png']:
                if (folder/name).exists():archive_file(folder/name);shutil.copy2(folder/name,dest/name)
            inp=folder/'input.json';response=folder/'response.json';text=folder/'raw.txt'
            if inp.exists():
                row['input']=read(inp)
                content=row['input'];obs=content['context']
                prompt='SYSTEM\n'+content['instructions']+'\n\nUSER\n'+model_user_text(content)
                for j,img in enumerate(content['images']):
                    if sha(img['path'])!=img['sha256']:raise ValueError('Model input image changed')
                    destimg=dest/'input-images'/f'{j:02d}{Path(img["path"]).suffix}';destimg.parent.mkdir(exist_ok=True)
                    shutil.copy2(img['path'],destimg);img['review_path']=rel(destimg,out);archive_file(img['path'])
                    prompt+='\n\n'+img['label']+'\n[image: '+img['path']+'; transmitted_sha256='+img.get('transmitted_sha256','not_recorded')+']'
                (dest/'prompt.txt').write_text(prompt)
            if response.exists():
                report['returned']+=1;row['response_sha256']=sha(response);response_data=read(response)
                row['provider_response_id']=response_data.get('id');row['provider_created']=response_data.get('created')
                usage=response_data.get('usage',{})
                for key in ['prompt_tokens','completion_tokens','total_tokens']:token_totals[key]+=usage.get(key,0)
                token_totals['reasoning_tokens']+=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
            if text.exists():
                row['raw']=text.read_text()
                try:row['answer']=json.loads(row['raw'])
                except json.JSONDecodeError:pass
                # Syntactic JSON decoding is distinct from deployment grammar acceptance.
                write(dest/'parsed.json',dict(json_decoded=row['answer'],deployment_valid=orig['status']=='format_valid_unreviewed',
                    rejection_reason=orig.get('reason'),program_assembled_geometry_is_not_teacher_output=True))
            if orig['status']=='format_valid_unreviewed':report['format_valid']+=1
            if orig['status'].startswith('not_run'):report['not_run']+=1
            if (folder/'mapping/frame.json').exists():
                f=read(folder/'mapping/frame.json');archive_file(folder/'mapping/frame.json')
                row['downstream']=dict(instances=f['rows'],selection=f['selection'],texts=f['binding']['texts'],
                    text_updates=f['binding']['updates'],map_after=read(folder/'map-after.json'))
                write(dest/'mapping-review.json',row['downstream'])
                for name in ['semantic.png','detections.png']:
                    if (folder/'mapping'/name).exists():
                        shutil.copy2(folder/'mapping'/name,dest/name);archive_file(folder/'mapping'/name)
            decision=copy.deepcopy(decisions_by[(variant,k['image_id'])])
            if decision['response_sha256']!=row['response_sha256']:raise ValueError('Review response hash mismatch')
            row['assistant_review']=decision
            if row['raw'] is not None:report['assistant_reviewed']+=1
            for f in field_counts:field_counts[f][decision['fields'][f]['verdict']]+=1
            human=dict(run_id=source.name,episode_id=episode['episode_id'],image_id=k['image_id'],model=cfg['model'],
                variant=variant,mode='teacher_annotation',response_sha256=row['response_sha256'],
                status='review_pending' if response.exists() else 'not_run',
                fields={f:dict(verdict='review_pending' if response.exists() else 'not_run',reason='',evidence=[])
                        for f in ['landmark','local','route']},error_codes=[],suggested_revision=None,
                approved=False,reviewer=None,reviewed_at=None)
            human['fields']['waypoint']=dict(verdict='not_applicable',reason='Teacher does not generate geometry',evidence=[])
            humans.append(human)
            for key,outkey in [('api_wall_s','api_wall_s'),('vlm_s','vlm_request_and_validation_s'),('segmentation_s','segmentation_s'),('ocr_s','ocr_s'),('mapping_s','map_update_s')]:
                if orig.get(key) is not None:timings[outkey].append(orig[key])
            if orig['status']=='failed' and orig.get('elapsed_s') is not None:timings['failed_total_s'].append(orig['elapsed_s'])
            write(dest/'review.json',decision)
            write(dest/'checks.json',dict(original_status=orig['status'],rejection_reason=orig.get('reason'),
                response_sha256=row['response_sha256'],human_approved=False,actual_input_available=inp.exists(),
                request_send_wall=orig.get('request_send_wall_unix'),response_receive_wall=orig.get('response_receive_wall_unix'),waypoint='not_applicable',ref_line='not_applicable',
                deployment_format_valid=orig['status']=='format_valid_unreviewed'))
            # This displayed capture was NOT sent to a model which did not run.
            common=dest/'source-rgb.png';shutil.copy2(p['source']['rgb'],common)
            row['capture_rgb']=rel(common,out);rows.append(row)
        report['tokens']=dict(token_totals)
        report['fields']={f:dict(counts=dict(c),scorable_denominator=sum(c[v] for v in ['pass','partial','fail']),
            semantic_pass_fraction_scored=c['pass']/sum(c[v] for v in ['pass','partial','fail']) if sum(c[v] for v in ['pass','partial','fail']) else None,
            semantic_pass_over_planned=c['pass']/len(keyframes),
            usable_pass_over_planned=sum(x['variant']==variant and x['status']=='format_valid_unreviewed' and x['assistant_review']['fields'][f]['verdict']=='pass' for x in rows)/len(keyframes))
            for f,c in field_counts.items()}
        report['timings']={k:stats(v) for k,v in timings.items()}
        report['accepted_events']=sum(x.get('accepted_instances',0) for x in r['samples'])
        report['text_updates']=sum(x.get('text_updates',0) for x in r['samples'])
        report['unique_landmarks']=len(report['final_map']['landmarks'])
        report['identity_confirmed']=sum(x.get('identity_status')=='identity_confirmed' for x in report['final_map']['landmarks'])
        report['api_cost_per_human_approved_label']=None
        models[variant]=report
    write(out/'assistant-review.json',decisions);write(out/'human-review.json',humans)
    files=['Agent.md','ticvla/warehouse/dataset_tools/semantic_loop_review.py','ticvla/warehouse/dataset_tools/semantic_loop.py']
    experiment=dict(mode='teacher_annotation',run_id=source.name,standard='vlm-human-review-v1',source=str(source),
        episode_id=episode['episode_id'],instruction=sample0['observation']['task']['instruction'],
        original_capture_instruction=episode['task']['instruction'],instruction_note='English approved version used by teachers; capture metadata retains original Chinese.',
        control_source='human teleoperation recording; no new simulator commands',context='self-recursive teacher outputs and independent per-model maps',
        start_source_time=keyframes[0]['source_time'],end_source_time=keyframes[-1]['source_time'],duration_s=keyframes[-1]['source_time']-keyframes[0]['source_time'],
        frame_start=keyframes[0]['frame_index'],frame_end=keyframes[-1]['frame_index'],capture_frames_total=len(capture),
        window_reason='Recorded task start through annotated stopped/arrival_confirmed frame; preceding capture is history, outside this task interval.',
        keyframe_selection='Existing 10 approved manual-demo keyframes reused; source perception manifest frozen before teacher inference; not picked by model success.',
        keyframe_gaps_s=[k['delta_time_s'] for k in keyframes[1:]],
        input='4 raw history/current RGB + past semantic map + current detector box/ID overlay; pixel masks review-only',
        camera=dict(resolution=episode['resolution'],K=sources[0][1]['source']['K'],T_body_camera=sources[0][1]['source']['T_body_camera'],depth='optical_z_metres'),
        odometry='Ideal synchronized simulator odometry; not VLM-estimated localization',
        detector=sources[0][1]['provenance'],ocr=sources[0][1]['ocr']['provenance'],
        map=dict(frame='current request source body: x forward, y left, metres',voxel_m=.05,occupancy_m=sources[0][1]['resolution'],
            max_records=16,sort='sign/label first, then distance, then UID',static_status='tentative',
            geometry='All preceding capture frames; semantic updates only at selected keyframes',
            registration='single shared odom epoch; no hidden map'),
        inference_code_version=(read(source/'experiment.json').get('code_head','unknown') if (source/'experiment.json').exists() else 'Original inference commit/dirty diff not fully recorded'),
        review_code_git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        review_code_hashes={p:sha(p) for p in files},new_model_calls=0,new_model_cost=0,training=False,
        video_alignment='source frame time; offline retrospective replay, not inference completion time',
        reference_evaluation='not_applicable: teacher mode; no decoder was run',
        measurement_limitations=['Request send/receive clocks are available only where archived in per-frame checks; never inferred from file times',
            'vlm_s includes request, archive and validation; no prefill/decode breakdown',
            'failed elapsed_s is overall failed attempt, not separately measured API inference',
            '10 frames do not provide dense online inference or independent benchmark accuracy',
            'Assistant visual review is not human approval; no exhaustive instance GT or mask IoU'])
    write(out/'experiment.json',experiment)
    report=dict(models=models,new_api_calls=0,new_cost_cny=0,historic_cost_cny=sum(m['historic_cost_cny'] for m in models.values()),
        human_approved=0,assistant_review_rows=len(decisions),build_elapsed_s=time.monotonic()-started,
        semantic_accuracy_claim='None; field judgments are evidence-based assistant review, not benchmark accuracy',
        verdict='See per-field reviews and completion counts; no automatic approval.')
    write(out/'report.json',report);write(out/'rows.json',rows)
    write(out/'source-hashes.json',archive)
    materialize_inputs(out)
    report_markdown(out)
    render(out)
    return out


CSS='''body{font:16px system-ui;margin:24px;background:#14202b;color:#e9f1fa}a{color:#75d6ff}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526679;padding:9px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px monospace}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.card{padding:15px;background:#1e3041;min-width:0}.images{display:flex;flex-wrap:wrap;gap:10px}.images figure{width:30%;margin:0}.images img{width:100%}figcaption{font-size:13px}details{margin:12px 0}h2{border-top:2px solid #5b7792;padding-top:18px}video{max-width:100%;width:1200px}.bad{color:#ffb28a}textarea{width:95%;min-height:50px}select,input,textarea,button{font:inherit}button{padding:8px;margin:8px}.thumb{width:150px}.sticky{position:sticky;top:0;background:#14202b;padding:8px;z-index:2}.verdict{font-weight:bold}@media(max-width:1100px){.pair{grid-template-columns:1fr}}'''


def materialize_inputs(output):
    """Reconstruct transmitted image bytes and verify the exact original request hash."""
    out=Path(output);rows=read(out/'rows.json');report=read(out/'report.json');checks=[]
    for row in rows:
        inp=row['input']
        if inp is None:continue
        cfg=report['models'][row['variant']]['config']
        content=[dict(type='text',text=model_user_text(inp))]
        for j,img in enumerate(inp['images']):
            path=Path(img['path']);data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=img['sha256']:raise ValueError('Changed original input')
            encoding=img['encoding'];match=re.fullmatch(r'jpeg_quality_(\d+)_444_original_resolution',encoding)
            if match:
                with Image.open(io.BytesIO(data)) as rgb:
                    buf=io.BytesIO();rgb.convert('RGB').save(buf,format='JPEG',quality=int(match[1]),subsampling=0)
                data=buf.getvalue()
            elif encoding!='original_file':raise ValueError('Unsupported archived encoding')
            if hashlib.sha256(data).hexdigest()!=img['transmitted_sha256']:
                raise ValueError('Transmitted bytes cannot be reproduced with this image encoder')
            target=out/row['directory']/'transmitted-images'/f'{j:02d}{".jpg" if img["mime_type"]=="image/jpeg" else path.suffix}'
            target.parent.mkdir(exist_ok=True);target.write_bytes(data)
            img.setdefault('original_review_path',img['review_path']);img['review_path']=rel(target,out)
            img['transmitted_bytes_verified']=True
            content += [dict(type='text',text=img['label']),dict(type='image_url',image_url=dict(
                url='data:'+img['mime_type']+';base64,'+base64.b64encode(data).decode()))]
        if inp['api']!='chat_completions':raise ValueError('This comparison archives Chat Completions requests')
        request=dict(model=inp['model'],messages=[dict(role='system',content=inp['instructions']),
            dict(role='user',content=content)],max_tokens=inp['max_output_tokens'],response_format=dict(type='json_object'))
        for key in ['thinking','enable_thinking','thinking_budget','reasoning_effort']:
            if key in cfg:request[key]=cfg[key]
        digest=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if digest!=inp['request_sha256']:raise ValueError('Reconstructed request differs from original')
        checks.append(dict(model=row['model'],image_id=row['keyframe']['image_id'],images=len(inp['images']),
                           request_sha256=digest,exact_request_reconstruction=True))
    write(out/'rows.json',rows);write(out/'request-reconstruction.json',checks)


def report_markdown(out):
    out=Path(out);exp=read(out/'experiment.json');r=read(out/'report.json');rows=read(out/'rows.json')
    lines=['# 完整移动任务 VLM 标注对照', '',
        '按 Agent.md 展示全部预定帧。Agent 视觉/日志复核不是人工批准；标签未进入训练。', '',
        '**Instruction:** '+exp['instruction'], '',
        f"场景 {exp['episode_id']}，任务 {exp['duration_s']:.1f} 秒。",
        '[审核页面](index.html) · [视频](episode.mp4) · [人工审核模板](human-review.json)', '',
        '## 覆盖、费用与耗时', '',
        '| 模型 | 返回/预定 | 协议有效 | 未运行 | 入图事件/地标 | 估算费用 CNY | API p50/p95 秒 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for m in r['models'].values():
        timing=m['timings'].get('api_wall_s',{})
        lat=f"{timing['p50']:.2f}/{timing['p95']:.2f}" if timing.get('n') else '未单独记录'
        lines.append(f"| {m['model']} | {m['returned']}/{m['planned']} | {m['format_valid']} | {m['not_run']} | {m['accepted_events']}/{m['unique_landmarks']} | {m['historic_cost_cny']:.6f} | {lat} |")
    lines+=['',f"归档调用共 {sum(m['requested'] for m in r['models'].values())} 次，费用估算 ¥{r['historic_cost_cny']:.6f}（含失败响应）。本报告整理过程不额外调用模型。",
        '价格按调用配置的北京端点未折扣牌价与实际 usage 计算，含思考输出，不扣缓存/促销，不是账单。',
        '[官方价格](https://help.aliyun.com/zh/model-studio/model-pricing)', '',
        '## 逐字段辅助复核', '',
        '| 模型 | 字段 | pass | partial | fail | not_run | 格式与语义均通过/预定 |',
        '| --- | --- | --- | --- | --- | --- | --- |']
    for m in r['models'].values():
        for f,stats_ in m['fields'].items():
            c=stats_['counts'];usable=round(stats_['usable_pass_over_planned']*m['planned'])
            lines.append(f"| {m['model']} | {f} | {c.get('pass',0)} | {c.get('partial',0)} | {c.get('fail',0)} | {c.get('not_run',0)} | {usable}/{m['planned']} |")
    lines+=['','## 全部关键帧','', '| 帧 | 模型 | Landmark/Local/Route | 运行状态 | 依据 |','| --- | --- | --- | --- | --- |']
    for row in rows:
        fs=row['assistant_review']['fields'];notes='；'.join(fs[f]['reason'] for f in ['landmark','local','route'])
        verdict='/'.join(fs[f]['verdict'] for f in ['landmark','local','route'])
        lines.append(f"| [{row['keyframe']['frame_index']}](index.html#f{row['keyframe']['frame_index']}) | {row['variant']} | {verdict} | {row['status']} | {notes.replace('|','/')} |")
    lines+=['','## 解释边界','',
        '- 每个模型使用自己的 Route、Local 和独立语义地图。只有首次请求完全共享证据，后续不是固定输入单轮对照。',
        '- 格式失败后停止递归，未运行帧保留在分母；不补问、不用人工历史修复。格式有效不等于语义正确。',
        '- 10 个人工关键帧是同一任务的回归审核，不能据此报告独立泛化准确率。没有穷尽实例真值，不报告 mask IoU。',
        '- API wall 时间含网络和服务端处理，不是纯 decode；检测和 OCR 时间来自共享的历史真实缓存，不是本轮新推理性能。',
        '- 输入地图在调用前冻结，当前帧新增地标仅供下一次请求。显示的像素 mask、事件与未来轨迹是审计材料，未输入教师。',
        '- Waypoint/ref_line 不适用于教师模型输出，保留独立轨迹监督；没有运行 Decoder 或车辆控制。',
        '- 视频使用实际录制 RGB 按源时间回放，关键帧时更新原文。不能证明该回答已在在线源时刻到达。',
        '- 人工批准为 0，审核页面可导出审核记录，不自动更新训练集。']
    if (out/'findings.json').exists():
        findings=read(out/'findings.json')
        lines[1:1]=['','## 本轮结论','',findings['summary'],'']+['- '+x for x in findings['points']]+['']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def render(out):
    out=Path(out);exp=read(out/'experiment.json');report=read(out/'report.json');rows=read(out/'rows.json');keys=read(out/'keyframes.json')
    representative(out, rows, exp)
    esc=html.escape
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>完整任务 VLM 审核</title><style>'+CSS+'</style>',
        '<h1>完整移动任务：多模型标注人工审核</h1>',
        '<p><b>teacher_annotation · 10 个预定关键帧 · assistant_review，不是人工批准</b></p>',
        '<p>Instruction: '+esc(exp['instruction'])+'</p>',
        '<p>任务区间 '+str(exp['frame_start'])+'–'+str(exp['frame_end'])+f"，{exp['duration_s']:.1f} 秒。原始遥操作回放；本页归档真实 API 标注，页面整理不再调用模型，也不控制车辆。</p>",
        '<p><a href="REPORT.md">报告</a> · <a href="experiment.json">实验配置与限制</a> · <a href="report.json">统计</a> · <a href="assistant-review.json">Agent 逐字段复核</a> · <a href="human-review.json">未批准人工模板</a> · <a href="keyframes.json">完整关键帧</a></p>',
        '<p class="bad">输入是原始 RGB、框/ID 叠图、历史语义地图；彩色 mask 是审核辅助，未送入 VLM。教师不生成 Waypoint 或 ref_line，本页不把轨迹 GT 当作教师输出。</p>']
    if (out/'findings.json').exists():
        findings=read(out/'findings.json')
        parts+=['<h2>本轮结论</h2><p>'+esc(findings['summary'])+'</p><ul>'+''.join('<li>'+esc(x)+'</li>' for x in findings['points'])+'</ul>']
    if (out/'episode.mp4').exists():parts+=['<video controls preload="metadata" src="episode.mp4" poster="representative.png"></video>',
        '<p>视频按源帧时间离线对齐；地图/输出在对应关键帧更新，冻结期间显示年龄。不能用于声称在线低延迟。</p>']
    else:parts.append('<p>视频生成中。</p>')
    parts.append('<table><tr><th>模型</th><th>返回 / 计划</th><th>格式有效</th><th>后续未运行</th><th>入图事件 / 独立地标</th><th>原调用估算费用</th></tr>')
    for v,m in report['models'].items():
        parts.append(f"<tr><td>{m['model']}</td><td>{m['returned']}/{m['planned']}</td><td>{m['format_valid']}</td><td>{m['not_run']}</td><td>{m['accepted_events']} / {m['unique_landmarks']}</td><td>¥{m['historic_cost_cny']:.6f}</td></tr>")
    parts.append('</table><p>后续模型各自使用自身地图/Route；不宣称各轮输入完全相同。p95 仅来自极小样本。</p>')
    parts.append(detail('逐字段计数、端到端覆盖与分模块耗时',report))
    parts.append('<div class="sticky">人工审核者：<input id="reviewer" placeholder="填写姓名"><button onclick="exportReview()">导出人工审核 JSON</button><span>只下载审核文件，不批准训练或改写原文；未填写保持待审。</span></div>')
    parts.append('<table><tr><th>帧 / 任务时间</th><th>当前 RGB</th>'+''.join('<th>'+esc(v)+' Landmark / Local / Route</th>' for v in report['models'])+'</tr>')
    for k in keys:
        subset=[r for r in rows if r['keyframe']['image_id']==k['image_id']]
        parts.append(f'<tr><td><a href="#f{k["frame_index"]}">{k["frame_index"]}</a><br>{k["episode_elapsed_s"]:.1f}s</td><td><img class="thumb" src="{subset[0]["capture_rgb"]}"></td>')
        for row in subset:
            parts.append('<td>'+esc(row['status'])+'<pre>'+esc(json.dumps(row['answer'],ensure_ascii=False) if row['answer'] else 'not_run / 无模型输出')+'</pre></td>')
        parts.append('</tr>')
    parts.append('</table>')
    for k in keys:
        parts.append(f'<h2 id="f{k["frame_index"]}">帧 {k["frame_index"]} · 源时间 {k["source_time"]:.3f}s · 任务 +{k["episode_elapsed_s"]:.1f}s</h2>')
        parts.append(detail('时序、相对运动与已有事件',k));parts.append('<div class="pair">')
        for row in [r for r in rows if r['keyframe']['image_id']==k['image_id']]:
            parts.append('<article class="card"><h3>'+esc(row['model'])+' · '+esc(row['status'])+'</h3>')
            inp=row['input'];directory=row['directory']
            if inp:
                ctx=audit_context(inp);imgs=inp['images'];n=len(ctx['source_rgb'])
                # Primary inputs shown first; all history retained below in exact order.
                parts.append('<h4>实际输入：当前 RGB / 历史语义地图 / 检测框</h4><div class="images">')
                for j in [n-1,*range(n,len(imgs))]:
                    im=imgs[j];parts.append(f'<figure><a href="{im["review_path"]}"><img src="{im["review_path"]}"></a><figcaption>{esc(im["label"])}</figcaption></figure>')
                parts.append('</div><details><summary>实际历史 RGB（按请求顺序，全部展示）</summary><div class="images">')
                for j in range(n-1):
                    im=imgs[j];h=ctx['source_rgb'][j]
                    parts.append(f'<figure><a href="{im["review_path"]}"><img src="{im["review_path"]}"></a><figcaption>{esc(im["label"])} t={h["t_obs"]:.3f}s</figcaption></figure>')
                parts.append('</div></details>')
                parts.append('<p><b>原始 Instruction：</b>'+esc(ctx['task']['instruction'])+'</p><p><b>上一轮 Route：</b>'+esc(str(ctx['route']))+'</p>')
                parts.append(detail('审计信息：Local/运动来源及未发送的执行事实',{a:ctx.get(a) for a in ['previous_local','execution','robot_state','source_pose']}))
                parts.append(detail('审计信息：候选表与地标记录，不作为文字输入',{a:ctx.get(a) for a in ['current_candidates','observed_map_landmarks','semantic_input']}))
                parts.append(detail('实际模型上下文',inp['context']))
                parts.append(detail('完整请求归档（audit_context 不发送给模型）',inp))
                parts.append(f'<p><a href="{directory}/prompt.txt">完整 prompt 文本</a> · <a href="{directory}/input.json">原 input.json</a></p>')
            else:
                parts.append('<p class="bad">未发起请求：没有本轮实际模型输入。以下仅为录制源帧，未送入此模型；不能推测该模型会怎样回答。</p>')
                parts.append(f'<img width="100%" src="{row["capture_rgb"]}">')
            parts.append('<h4>实际输出</h4><pre>'+esc(row['raw'] or 'not_run: no response')+'</pre>')
            if row['raw']:
                parts.append(f'<a href="{directory}/raw.txt">原始文本</a> · <a href="{directory}/response.json">原始 API 响应</a>')
                parts.append(detail('JSON 解析展示（不等于通过部署协议）',row['answer']))
            if row.get('reason'):parts.append('<p class="bad">拒绝原因：'+esc(row['reason'])+'</p>')
            parts.append('<p>Waypoint / ref_line：not_applicable。没有教师几何预测；不展示程序组装的 GT 为教师答案。</p>')
            parts.append('<h4>Agent 逐字段复核（尚非人工批准）</h4><table><tr><th>字段</th><th>评价</th><th>依据</th></tr>')
            for f,d in row['assistant_review']['fields'].items():
                parts.append('<tr><td>'+esc(f)+'</td><td>'+esc(d['verdict'])+'</td><td>'+esc(d['reason'])+'<br>'+esc('; '.join(d.get('evidence',[])))+'</td></tr>')
            parts.append('</table>'+detail('错误码、修订建议及审核来源',row['assistant_review']))
            if row['downstream']:
                parts.append('<h4>下游审核辅助（不是模型输入）</h4><div class="images">')
                for name,label in [('semantic.png','被接受实例 mask'),('detections.png','全部实例 mask'),('map-after.png','本轮入图后地图（未反馈到本轮输入）')]:
                    if (out/directory/name).exists():parts.append(f'<figure><a href="{directory}/{name}"><img src="{directory}/{name}"></a><figcaption>{label}</figcaption></figure>')
                parts.append('</div>')
                parts.append(detail('完整投影点数、类别准入、OCR 归属和更新后的地标',row['downstream']))
                parts.append(f'<p>地图版本 {row.get("map_version_input")} → {row.get("map_version_output")}；有效实例事件 {row.get("accepted_instances")}；文字追加 {row.get("text_updates")}。</p>')
            else:parts.append('<p>本轮入图/OCR 绑定：not_run（格式失败或请求未执行）。</p>')
            ident=rows.index(row);parts.append(f'<fieldset data-review="{ident}"><legend>人工审核（可导出）</legend>')
            for f in ['landmark','local','route']:
                initial='review_pending' if row['raw'] else 'not_run'
                parts.append(f'<label>{f} <select data-field="{f}">'+''.join(f'<option {"selected" if x==initial else ""}>{x}</option>' for x in ['review_pending','pass','partial','fail','unscorable','not_run'])+'</select></label> ')
            parts.append('<textarea placeholder="具体依据、错误码或修订建议" data-note></textarea>')
            parts.append('<label><input type="checkbox" data-approve>本人批准该帧原始文本标签（仅全部字段 pass 且格式有效时可选）</label></fieldset>')
            parts.append('</article>')
        parts.append('</div>')
    human=read(out/'human-review.json')
    payload=json.dumps(human,ensure_ascii=False).replace('<','\\u003c')
    valid=json.dumps([r['status']=='format_valid_unreviewed' for r in rows])
    parts.append('<script>const template='+payload+';const valid='+valid+';'+r'''
function exportReview(){const name=document.getElementById('reviewer').value.trim();if(!name){alert('请填写审核者姓名');return;}
const data=structuredClone(template);for(const box of document.querySelectorAll('[data-review]')){const i=Number(box.dataset.review),r=data[i];
for(const s of box.querySelectorAll('[data-field]'))r.fields[s.dataset.field].verdict=s.value;
const note=box.querySelector('[data-note]').value.trim(),approve=box.querySelector('[data-approve]').checked;
const modified=note||approve||[...box.querySelectorAll('[data-field]')].some(s=>s.value!==template[i].fields[s.dataset.field].verdict);
if(!modified)continue;
if(!r.response_sha256){alert('未运行帧没有可审核的模型输出');return;}
if(approve&&(!valid[i]||Object.values(r.fields).some(f=>!['pass','not_applicable'].includes(f.verdict)))){alert('批准要求格式有效、全部适用字段 pass');return;}
if(!note){alert('填写修改条目的审核依据');return;}
for(const f of ['landmark','local','route'])r.fields[f].reason=note;
r.status='human_reviewed';r.reviewer=name;r.reviewed_at=new Date().toISOString();r.approved=approve;r.notes=note;
}const blob=new Blob([JSON.stringify(data,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='human-review.json';a.click();URL.revokeObjectURL(a.href);}
</script></html>''')
    (out/'index.html').write_text(''.join(parts))


def representative(out, rows, exp):
    candidates=[r for r in rows if r['input'] and r['answer']]
    row=next((r for r in candidates if r['assistant_review']['fields']['route']['verdict']=='fail'),candidates[0])
    canvas=Image.new('RGB',(1920,1280),'#14202b');d=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',25)
    d.text((20,12),f'Actual VLM input/output | {row["model"]} | source frame {row["keyframe"]["frame_index"]}',font=title,fill='white')
    d.multiline_text((20,48),'\n'.join(textwrap.wrap('Instruction: '+exp['instruction'],130)),font=font,fill='#bfe8ff',spacing=5)
    n=len(audit_context(row['input'])['source_rgb']);imgs=row['input']['images']
    for x,index,label in [(0,n-1,'Current raw RGB'),(640,len(imgs)-1,'Actual detector boxes / IDs'),(1280,n,'Prior semantic map (not current update)')]:
        d.text((x+8,125),label,font=font,fill='#ffce80')
        with Image.open(out/imgs[index]['review_path']) as image:
            pic=ImageOps.contain(image.convert('RGB'),(630,630));canvas.paste(pic,(x+5,165))
    for y,key in [(810,'landmark'),(855,'local'),(930,'route')]:
        text=json.dumps(row['answer'][key]) if key=='landmark' else row['answer'][key]
        d.multiline_text((20,y),'\n'.join(textwrap.wrap(key+': '+text,132)),font=font,fill='white',spacing=5)
    events=', '.join(e['kind'] for e in row['keyframe']['events'])
    d.text((20,1190),'Audit-only events (not model input): '+events,font=font,fill='#ffac96')
    d.text((20,1235),'Assistant review only; no human approval. Waypoint/ref_line: not applicable for teacher.',font=font,fill='#ffce80')
    canvas.save(out/'representative.png')


def video(output, fps=5):
    """One recorded episode, arbitrary model lanes; never substitutes failed outputs."""
    out=Path(output);exp=read(out/'experiment.json');rows=read(out/'rows.json');keys=read(out/'keyframes.json')
    variants=list(read(out/'report.json')['models'])
    if not 1<=len(variants)<=3:raise ValueError('Video supports one to three model lanes')
    source=Path(exp['source']);manifest=read(source/'perception/manifest.json');raw=Path(manifest['source_episode']).parent
    meta=read(raw/'episode.json');capture=[json.loads(x) for x in (raw/meta['frames_file']).read_text().splitlines()]
    first,last=exp['frame_start'],exp['frame_end'];segment=capture[first:last+1]
    from .replay_video import replay_indices
    selection=replay_indices([f['t_obs'] for f in segment],fps);indices=[k['frame_index'] for k in keys]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',22)
    W,H=1920,1440;cache={};lastkey=-1;started=time.monotonic()
    def paste(canvas,path,box):
        if str(path) not in cache:
            with Image.open(path) as img:cache[str(path)]=img.convert('RGB').copy()
        canvas.paste(ImageOps.contain(cache[str(path)],box[2:]),box[:2])
    def lines(d,xy,text,width=60,font_=small,color='white'):
        d.multiline_text(xy,'\n'.join(textwrap.wrap(text,width=width)),font=font_,fill=color,spacing=5)
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-',
         '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'episode.mp4')]
    with (out/'ffmpeg.log').open('w') as err,(out/'video-frames.jsonl').open('w') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        try:
            for n,offset in enumerate(selection):
                index=first+int(offset);f=capture[index];ki=frame_at(indices,index);k=keys[ki];t=f['t_obs']
                canvas=Image.new('RGB',(W,H),'#14202b');d=ImageDraw.Draw(canvas)
                lines(d,(20,12),'Instruction: '+exp['instruction'],width=135,font_=title)
                d.text((20,75),f'Recorded RGB | frame {index} | task +{t-exp["start_source_time"]:.1f}s | NO model control',font=font,fill='#6de4ff')
                paste(canvas,raw/f['rgb'],(20,110,900,675))
                record=dict(video_frame=n,video_time_s=n/fps,capture_index=index,source_time=t,pose=f['T_odom_body'],models={})
                for vi,v in enumerate(variants):
                    attempts=[r for r in rows if r['variant']==v and r['keyframe']['frame_index']<=index and r['input']]
                    lastrow=attempts[-1] if attempts else None
                    x=950+vi*315
                    d.text((x,95),v+' last INPUT map',font=small,fill='white')
                    if lastrow:
                        ni=len(audit_context(lastrow['input'])['source_rgb']);inputmap=out/lastrow['input']['images'][ni]['review_path']
                        paste(canvas,inputmap,(x,130,300,300));age=t-lastrow['keyframe']['source_time']
                        lines(d,(x,440),f'Source {lastrow["keyframe"]["frame_index"]}; age {age:.1f}s',width=28)
                        record['models'][v]=dict(last_request_frame=lastrow['keyframe']['frame_index'],age_s=age,status=lastrow['status'])
                    current=next(r for r in rows if r['variant']==v and r['keyframe']['index']==ki)
                    px=20+vi*630
                    d.rectangle((px,850,px+610,1420),outline='#637d97',width=2)
                    d.text((px+10,865),v.upper()+' | '+current['status'],font=small,fill='#72e5bc' if current['status']=='format_valid_unreviewed' else '#ff967d')
                    shown=current if current['answer'] else lastrow
                    if shown and shown['answer']:
                        ans=shown['answer'];age=t-shown['keyframe']['source_time']
                        lines(d,(px+10,900),f'Output source {shown["keyframe"]["frame_index"]}; age {age:.1f}s'+(' HISTORICAL / REJECTED' if current['status']!='format_valid_unreviewed' else ''),width=56,color='#ffcd7a')
                        for yy,key in [(960,'landmark'),(1020,'local'),(1120,'route')]:
                            value=json.dumps(ans[key]) if key=='landmark' else ans[key]
                            lines(d,(px+10,yy),key+': '+value,width=56)
                    else:d.text((px+10,960),'No generated response',font=font,fill='white')
                active=next((r for r in rows if r['keyframe']['index']==ki and r['input']),None)
                if active:
                    paste(canvas,out/active['input']['images'][-1]['review_path'],(950,510,500,300))
                    d.text((950,485),f'Actual detector overlay, source {k["frame_index"]}',font=small,fill='#6de4ff')
                lines(d,(1470,520),'Offline source-time replay. Responses shown at source frames, NOT API arrival.',width=38,color='#ffcd7a')
                lines(d,(1470,650),'Waypoint / ref_line: N/A. Teacher only; no Decoder run. No human approval.',width=38,color='#ffcd7a')
                d.text((20,810),'Independent recursive maps and text. NOT RUN slots retain their last response with explicit age.',font=font,fill='#ffcd7a')
                if ki!=lastkey:canvas.save(out/f'video-keyframe-{ki:02d}.jpg',quality=90);lastkey=ki
                proc.stdin.write(canvas.tobytes());log.write(json.dumps(record)+'\n')
                if n%150==0:print('video',n,'/',len(selection),flush=True)
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('ffmpeg failed')
        except BaseException:proc.kill();proc.wait();raise
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_format','-show_streams','-of','json',str(out/'episode.mp4')],text=True))
    write(out/'video.json',dict(fps=fps,frames=len(selection),source_frames=len(segment),frame_start=first,frame_end=last,
        source_duration_s=exp['duration_s'],build_s=time.monotonic()-started,alignment='source_time_offline',
        trajectory='actual recorded frames and odometry; no synthetic camera motion',sha256=sha(out/'episode.mp4'),probe=probe))
    render(out)


def audit(output):
    from html.parser import HTMLParser
    out=Path(output).resolve();rows=read(out/'rows.json');keys=read(out/'keyframes.json')
    exp=read(out/'experiment.json');sources=read(out/'source-hashes.json')
    changed=[p for p,h in sources.items() if sha(p)!=h]
    if changed:raise ValueError('Source artifacts changed: '+str(changed))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self, tag, attrs):
            self.paths += [v for k,v in attrs if k in ('src','href','poster') and v]
    parser=Links();parser.feed((out/'index.html').read_text());missing=[]
    for p in parser.paths:
        if p.startswith(('http:','https:','data:','#')):continue
        if not (out/p.split('#',1)[0]).exists():missing.append(p)
    if missing:raise ValueError('Broken links: '+str(missing))
    variants=list(read(out/'report.json')['models'])
    assert len(rows)==len(variants)*len(keys)
    for v in variants:
        model_rows=[r for r in rows if r['variant']==v]
        assert [r['keyframe']['image_id'] for r in model_rows]==[k['image_id'] for k in keys]
        for r in model_rows:
            if r['status'].startswith('not_run'):assert r['input'] is None and r['answer'] is None
    assert all(not h['approved'] and h['reviewer'] is None for h in read(out/'human-review.json'))
    ledger=[json.loads(s) for s in (out/'video-frames.jsonl').read_text().splitlines()]
    video_info=read(out/'video.json')
    assert len(ledger)==video_info['frames']
    assert ledger[0]['capture_index']==exp['frame_start'] and ledger[-1]['capture_index']==exp['frame_end']
    assert all(a['source_time']<=b['source_time'] for a,b in zip(ledger,ledger[1:]))
    for f in ledger:
        assert all(m['last_request_frame']<=f['capture_index'] and m['age_s']>=-1e-6 for m in f['models'].values())
        for v,m in f['models'].items():
            expected=max(r['keyframe']['frame_index'] for r in rows if r['variant']==v and r['input'] and r['keyframe']['frame_index']<=f['capture_index'])
            assert m['last_request_frame']==expected
    # Decode the complete file, not just its MP4 metadata.
    subprocess.run(['ffmpeg','-v','error','-i',str(out/'episode.mp4'),'-f','null','-'],check=True)
    audit_info=dict(standard='vlm-human-review-v1',passed=True,source_files_unchanged=len(sources),
        html_links_checked=len(parser.paths),broken_links=[],all_planned_slots=len(rows),
        human_approved=0,video_decoded=True,video_frames=len(ledger),
        video_duration_s=float(video_info['probe']['format']['duration']),
        source_alignment_verified=True,not_an_online_latency_test=True,
        missing_original_metadata=exp['measurement_limitations'])
    write(out/'artifact-audit.json',audit_info)
    exp['final_packager_sha256']=sha(__file__);write(out/'experiment.json',exp)
    print(json.dumps(audit_info,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('build');a.add_argument('--source',required=True);a.add_argument('--output',required=True);a.add_argument('--reviews',required=True)
    a=s.add_parser('video');a.add_argument('--output',required=True);a.add_argument('--fps',type=int,default=5)
    a=s.add_parser('render');a.add_argument('--output',required=True)
    a=s.add_parser('audit');a.add_argument('--output',required=True)
    a=s.add_parser('materialize');a.add_argument('--output',required=True)
    args=vars(p.parse_args());command=args.pop('command')
    {'build':build,'video':video,'render':lambda output:render(output),'audit':audit,'materialize':materialize_inputs}[command](**args)

if __name__=='__main__':main()
