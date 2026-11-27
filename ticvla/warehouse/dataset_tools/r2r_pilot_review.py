"""Full-episode R2R teacher review. Geometry is demonstration GT, never model output."""
import argparse
from collections import Counter
import hashlib
import html
import io
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from .core import read_json, write_json
from .r2r_pilot import verify
from ..schema import file_sha256
from ..compact_input import audit_context

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
CSS='body{font:16px system-ui;background:#14202b;color:#eef5fa;margin:24px}a{color:#70d9ff}pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%}td,th{border:1px solid #526679;padding:8px;vertical-align:top}img{max-width:100%}.imgs{display:flex;flex-wrap:wrap;gap:10px}.imgs figure{width:30%;margin:0}section{border-top:2px solid #7799aa;margin-top:24px}textarea{width:95%;height:65px}video{width:100%;max-width:1500px}button,select{font:inherit;padding:6px}.warn{color:#ffb77c}'


def review_flags(answer, sample):
    """Post-run checks never rewrite accepted history or pretend to be human review."""
    flags=[]
    if isinstance(answer,dict):
        route=answer.get('route','')
        events={e['kind'] for e in sample['observation']['execution'].get('events',[])}
        if isinstance(route,str) and re.search(r'\bDONE\b',route) and not {'target_visible','stopped','arrival_confirmed'}<=events:
            flags.append('DONE_TOKEN_WITHOUT_RECORDED_COMPLETION_EVIDENCE')
        if isinstance(answer.get('local'),str) and len(answer['local'])>90:flags.append('LOCAL_EXCEEDS_90_CHARACTERS')
        for key in ('local','route'):
            if isinstance(answer.get(key),str) and re.search(r"[^A-Za-z0-9 ,.';:()_/-]",answer[key]):
                flags.append(key.upper()+'_UNSUPPORTED_GRAMMAR_CHARACTER')
    return flags


def rel(path, root):return os.path.relpath(path,root)
def esc(value):return html.escape(str(value))
def detail(title,value):
    return '<details><summary>'+esc(title)+'</summary><pre>'+esc(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre></details>'


def show_image(path, root, label):
    r=rel(path,root)
    return f'<figure><a href="{esc(r)}"><img loading="lazy" src="{esc(r)}"></a><figcaption>{esc(label)}</figcaption></figure>'


def reference_images(sample, source, map_path, destination):
    g=sample['geometric_supervision'];xy=np.asarray(g['reference_xy'],float)[np.asarray(g['reference_mask'],bool)]
    wp=np.asarray(g['waypoint'],float).reshape(-1,2)
    bev=Image.open(map_path).convert('RGB');d=ImageDraw.Draw(bev)
    span=16.;w,h=bev.size
    def pixels(points):return [(w*(.5-y/span),(h-40)*(.5-x/span)) for x,y in points]
    if len(xy)>1:d.line(pixels(xy),fill='orange',width=4)
    for i,(u,v) in enumerate(pixels(wp)):
        d.ellipse((u-4,v-4,u+4,v+4),fill='cyan');d.text((u+4,v),f'W{i+1}',fill='blue')
    d.text((5,5),'REVIEW ONLY: demonstration GT',fill='red')
    bev.save(destination/'reference-bev.png')
    rgb=Image.open(source['rgb']).convert('RGB');d=ImageDraw.Draw(rgb)
    pts=np.c_[xy,np.zeros(len(xy)),np.ones(len(xy))]
    cam=(np.linalg.inv(np.asarray(source['T_body_camera']))@pts.T).T[:,:3]
    k=np.asarray(source['K']);valid=[]
    for p in cam:
        uv=k@p
        if p[2]>0:
            u,v=uv[:2]/uv[2]
            if 0<=u<rgb.width and 0<=v<rgb.height:
                d.ellipse((u-3,v-3,u+3,v+3),fill='orange');valid.append([float(u),float(v)])
    d.text((5,5),'REVIEW ONLY: GT projected at body z=0',fill='red')
    rgb.save(destination/'reference-rgb.png')
    write_json(destination/'geometry-gt.json',dict(g,teacher_generated=False,
        projection_assumption='Body z=0 XY path; stairs/elevation are not represented by XY labels',
        visible_projection_points=valid))


def mask_preview(perception, destination):
    rgb=np.asarray(Image.open(perception['source']['rgb']).convert('RGB')).copy()
    colors=[(255,90,70),(70,220,110),(80,140,255),(255,200,50),(210,80,220)]
    for i,inst in enumerate(perception['instances']):
        if file_sha256(inst['mask_path'])!=inst['mask_sha256']:raise ValueError('Mask changed')
        mask=np.load(inst['mask_path']);color=np.asarray(colors[i%len(colors)])
        rgb[mask]=(rgb[mask]*.55+color*.45).astype(np.uint8)
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    for i,inst in enumerate(perception['instances']):
        box=inst['bbox'];draw.text((box[0],box[1]),f'd{i} {inst["category"]} {inst["confidence"]:.2f}',fill='yellow',stroke_width=1,stroke_fill='black')
    im.save(destination/'masks-review-only.png')


def export_actual_image(item,dest):
    if file_sha256(item['path'])!=item['sha256']:raise ValueError('Actual input source changed')
    if item.get('encoding','').startswith('jpeg_quality_'):
        quality=int(item['encoding'].split('_')[2]);buffer=io.BytesIO()
        with Image.open(item['path']) as im:im.convert('RGB').save(buffer,format='JPEG',quality=quality,subsampling=0)
        content=buffer.getvalue();dest=dest.with_suffix('.jpg')
    else:content=Path(item['path']).read_bytes();dest=dest.with_suffix(Path(item['path']).suffix)
    if hashlib.sha256(content).hexdigest()!=item['transmitted_sha256']:raise ValueError('Transmitted image hash mismatch')
    dest.write_bytes(content);return dest


def episode_review(root, entry):
    folder=Path(entry['folder']);out=folder/'review';out.mkdir(exist_ok=True)
    manifest=read_json(folder/'perception/manifest.json');run=read_json(folder/'plus/report.json')
    revalidation=read_json(root/'postpilot-completion-guard.json') if (root/'postpilot-completion-guard.json').exists() else None
    rechecked={r['image_id']:r for r in revalidation['records']} if revalidation else {}
    if len(run['samples'])!=entry['frames']:raise ValueError('Missing planned result rows')
    parts=['<!doctype html><meta charset="utf-8"><style>'+CSS+'</style>',f'<h1>{esc(entry["uid"])} · Qwen Plus 标注审核</h1>',
        '<p>'+esc(entry['instruction'])+'</p>',
        '<p class="warn">Habitat 示范动作离线回放；理想局部 odom。模型没有控制运动。视频按源时间更新，不代表实时推理。</p>',
        '<p>教师只生成 Landmark / Local / Route。橙色参考线与青色 waypoint 是未来示范轨迹 GT，仅供审核，未输入教师。</p>',
        '<p><a href="../../../index.html">试点总览</a> · <a href="report.json">统计</a> · <a href="human-review.json">人工审核模板</a> · <a href="../plus/causality-audit.json">因果审计</a> · <a href="../plus/final-map.json">最终地图</a> · <a href="../plus/map/map.sqlite">地图数据库</a> · <a href="../plus/config.json">冻结教师配置</a></p>',
        '<video controls preload="metadata" src="episode.mp4" poster="poster.jpg"></video>',
        '<p>审核者：<input id="reviewer"><button onclick="exportReview()">导出人工审核 JSON</button>只下载文件，不自动批准训练。</p>']
    rows=[];humans=[];tokens=Counter();archive={}
    for i,(e,result) in enumerate(zip(manifest['records'],run['samples'])):
        source_folder=folder/'perception'/e['folder'];f=folder/'plus'/result['folder'];dest=out/'frames'/e['folder'];dest.mkdir(parents=True,exist_ok=True)
        sample=read_json(source_folder/'sample.json');p=read_json(source_folder/'perception.json');s=p['source']
        if result['source_image_id']!=s['image_id']:raise ValueError('Review source order mismatch')
        for name,key in [('sample.json','sample_sha256'),('perception.json','perception_sha256'),('prior-geometry.npz','geometry_sha256')]:
            if file_sha256(source_folder/name)!=e[key]:raise ValueError('Prepared evidence changed')
            archive[str(source_folder/name)]=e[key]
        for key in ['rgb','depth']:
            if file_sha256(s[key])!=s[key+'_hash']:raise ValueError('Capture source changed')
            archive[s[key]]=s[key+'_hash']
        shutil.copy2(s['rgb'],dest/'source.png')
        for name in ['input.json','sample.json','response.json','raw.txt','proposal.json','checks.json','map-after.json','map-after.png']:
            if (f/name).exists():shutil.copy2(f/name,dest/name);archive[str(f/name)]=file_sha256(f/name)
        inp=read_json(f/'input.json') if (f/'input.json').exists() else None
        raw=(f/'raw.txt').read_text() if (f/'raw.txt').exists() else None
        response=read_json(f/'response.json') if (f/'response.json').exists() else None
        answer=None
        if raw:
            try:answer=json.loads(raw)
            except json.JSONDecodeError:pass
        if response:
            usage=response.get('usage',{})
            tokens.update({k:usage.get(k,0) for k in ['prompt_tokens','completion_tokens','total_tokens']})
            tokens['reasoning_tokens']+=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
        row=dict(result,index=i,frame_index=sample['provenance']['frame_index'],source_time=s['t_obs'],
            directory=rel(dest,out),raw=raw,answer=answer,input_images=[],instruction=entry['instruction'],
            geometry_gt=sample['geometric_supervision'],response_sha256=file_sha256(f/'response.json') if response else None,
            format_valid=(f/'proposal.json').exists(),
            posthoc_review_flags=review_flags(answer,sample),
            output_lengths={k:len(answer[k]) for k in ('local','route') if isinstance(answer,dict) and isinstance(answer.get(k),str)})
        depth=np.load(s['depth'])
        row['depth_valid_fraction']=float((np.isfinite(depth)&(depth>0)).mean())
        row['postpilot_revalidation']=rechecked.get(s['image_id'])
        parts.append(f'<section id="f{i}"><h2>{esc(s["image_id"])} · {s["t_obs"]:.1f}s · {esc(result["status"])}</h2>')
        parts.append(detail('状态、耗时、实际上下文来源与失败原因',dict(result,output_lengths=row['output_lengths'])))
        if row['posthoc_review_flags']:
            parts.append('<p class="warn">事后审核标记（不修改当时递归结果）：'+esc(', '.join(row['posthoc_review_flags']))+'</p>')
        if row['postpilot_revalidation']:
            parts.append(detail('完成标记修复后的独立复核；未重发请求或重算原地图',row['postpilot_revalidation']))
        if inp:
            ctx=audit_context(inp);images=inp['images'];parts.append('<h3>实际输入（请求顺序，包含所有历史 RGB）</h3><div class="imgs">')
            for j,item in enumerate(images):
                imagefile=export_actual_image(item,dest/f'input-{j:02d}')
                row['input_images'].append(dict(path=rel(imagefile,out),label=item['label']))
                parts.append(show_image(imagefile,out,item['label']))
            parts.append('</div>')
            prompt='SYSTEM\n'+inp['instructions']+'\n\nUSER\n'+inp['user_text']
            (dest/'prompt.txt').write_text(prompt)
            row['input_context']=inp['context']
            row['map_input_path']=row['input_images'][len(ctx['source_rgb'])]['path']
            parts.append(detail('实际 SYSTEM / USER 文本',prompt))
            parts.append(detail('审计信息（不等于模型输入）',ctx))
            map_path=out/row['map_input_path']
        else:
            from ..semantic_input import render_semantic_map
            render_semantic_map(np.load(source_folder/'prior-geometry.npz')['grid'],p['resolution'],[]).save(dest/'not-sent-map.png')
            map_path=dest/'not-sent-map.png';parts.append('<p>未调用：以下仅为源数据，未发送给模型。</p>')
        mask_preview(p,dest);reference_images(sample,s,map_path,dest)
        parts.append('<h3>模型原文</h3><pre>'+esc(raw if raw is not None else 'not_run / no response')+'</pre>')
        if answer and isinstance(answer,dict):
            tags='\n'.join(f'<{tag}>{json.dumps(answer[k],ensure_ascii=False) if k=="landmark" else answer[k]}</{tag}>'
                for k,tag in [('landmark','landmark'),('local','local'),('route','Route')] if k in answer)
            parts.append(detail('程序标签式展示；不是第二次生成',tags))
        parts.append('<h3>审核辅助（未输入模型）</h3><div class="imgs">')
        for name,label in [('masks-review-only.png','真实像素 mask / 检测类别'),('reference-bev.png','同源车体系示范 GT：参考线和 waypoint'),('reference-rgb.png','GT 投影；XY 路线按车体 z=0 展示')]:
            parts.append(show_image(dest/name,out,label))
        parts.append('</div>'+detail('waypoint / ref_line GT 与来源',sample['geometric_supervision']))
        if (f/'mapping/frame.json').exists():
            mapping=read_json(f/'mapping/frame.json');write_json(dest/'mapping.json',mapping)
            row['mapping']=mapping
            parts.append(detail('逐实例入图、拒绝与 OCR 绑定',mapping))
            if (f/'mapping/semantic.png').exists():
                shutil.copy2(f/'mapping/semantic.png',dest/'semantic.png')
                parts.append('<div class="imgs">'+show_image(dest/'semantic.png',out,'入图与文字结果')+'</div>')
        human=dict(run_id=root.name,episode_id=entry['uid'],image_id=s['image_id'],model=run['model'],mode='teacher_annotation',
            response_sha256=row['response_sha256'],status='review_pending' if response else 'not_run',approved=False,
            reviewer=None,reviewed_at=None,suggested_revision=None,error_codes=[],fields={})
        for field in ['landmark','local','route','map_feedback','geometry_gt']:
            human['fields'][field]=dict(verdict='review_pending' if response else 'not_run',reason='',evidence=[])
            parts.append(f'<p>{field}: <select id="v-{i}-{field}">'+''.join(f'<option>{v}</option>' for v in ['review_pending','pass','partial','fail','unscorable','not_applicable','not_run'])+'</select></p>'+f'<textarea id="r-{i}-{field}" placeholder="依据与原因"></textarea>')
        human['fields']['waypoint']=dict(verdict='not_applicable',reason='Teacher does not generate geometry',evidence=[])
        humans.append(human);rows.append(row)
        parts.append(f'<textarea id="revision-{i}" placeholder="建议修订；不覆盖原始模型输出"></textarea>')
        parts.append(f'<p><a href="{rel(dest,out)}/raw.txt">原文</a> · <a href="{rel(dest,out)}/geometry-gt.json">几何 GT</a> · <a href="{rel(dest,out)}/input.json">实际输入记录</a></p></section>' if inp and raw else '</section>')
    write_json(out/'human-review.json',humans);write_json(out/'rows.json',rows);write_json(out/'source-hashes.json',archive)
    def stats(values):return dict(n=len(values),p50=float(np.percentile(values,50)) if values else None,p95=float(np.percentile(values,95)) if values else None)
    final=read_json(folder/'plus/final-map.json')
    report=dict(episode_id=entry['uid'],model=run['model'],planned=entry['frames'],requested=run['actual_api_calls'],
        returned=sum(r['response_sha256'] is not None for r in rows),format_valid=sum(r['format_valid'] for r in rows),
        rule_valid=sum(r['status']=='format_valid_unreviewed' for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)),human_reviewed=0,human_approved=0,semantic_accuracy=None,
        accepted_events=sum(r.get('accepted_instances',0) for r in rows),unique_landmarks=len(final['landmarks']),
        map_categories=dict(Counter(lm['category'] for lm in final['landmarks'])),
        map_statuses=dict(Counter(lm['status'] for lm in final['landmarks'])),
        per_instance_outcomes=dict(Counter(x['result']['status'] for r in rows for x in r.get('mapping',{}).get('rows',[]))),
        per_instance_rejections=dict(Counter(x['result'].get('reason','unspecified') for r in rows for x in r.get('mapping',{}).get('rows',[]) if x['result']['status'] not in ('accepted','not_selected'))),
        text_updates=sum(r.get('text_updates',0) for r in rows),cost_cny_estimate=run['cost_cny_estimate'],tokens=dict(tokens),
        timings={k:stats([r[k] for r in rows if k in r]) for k in ['api_wall_s','segmentation_s','ocr_s','mapping_s']},
        all_frames_attempted=run.get('all_frames_attempted'),all_frames_valid=run['all_frames_completed'],
        fatal_stop=run.get('fatal_stop'),geometry_source='verified_dataset_action_replay',
        posthoc_review_flags=dict(Counter(flag for r in rows for flag in r['posthoc_review_flags'])),
        depth_valid_fraction=stats([r['depth_valid_fraction'] for r in rows]),
        pure_turn_is_not_stop=True,live_navigation_evaluated=False,pose_source='ideal_simulator_odometry')
    if revalidation:
        report['strict_rule_pass']=sum(bool(r['postpilot_revalidation']['strict_rule_pass']) for r in rows)
        report['context_affected_frames']=sum(bool(r['postpilot_revalidation']['context_requires_review_due_to_prior_rejected_completion']) for r in rows)
    write_json(out/'report.json',report)
    table='<h2>全部计划关键帧</h2><table><tr><th>帧 / 时间</th><th>状态</th><th>Landmark / Local / Route</th></tr>'
    for r in rows:table+=f'<tr><td><a href="#f{r["index"]}">{r["frame_index"]} / {r["source_time"]:.1f}s</a></td><td>{esc(r["status"])}</td><td><pre>{esc(r["raw"] or "no response")}</pre></td></tr>'
    table+='</table>';parts.insert(8,table+detail('计数、费用与分模块耗时',report))
    parts.append('<script>const initial='+json.dumps(humans,ensure_ascii=False).replace('<','\\u003c')+''';
function exportReview(){const data=structuredClone(initial);for(let i=0;i<data.length;i++){
 for(const f of ['landmark','local','route','map_feedback','geometry_gt']){
  data[i].fields[f].verdict=document.getElementById(`v-${i}-${f}`).value;
  data[i].fields[f].reason=document.getElementById(`r-${i}-${f}`).value;
 }
 data[i].suggested_revision=document.getElementById(`revision-${i}`).value||null;
 data[i].reviewer=document.getElementById('reviewer').value||null;
 data[i].reviewed_at=new Date().toISOString();data[i].approved=false;
 data[i].status=Object.values(data[i].fields).some(x=>x.verdict==='review_pending')?'review_pending':'reviewed_not_approved';
}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='human-review.json';a.click();}
</script>''')
    (out/'index.html').write_text('\n'.join(parts))
    return out,rows,report


def video(out, entry, rows):
    raw=Path(entry['source_episode']).parent;meta=read_json(entry['source_episode'])
    frames=[json.loads(x) for x in (raw/meta['frames_file']).read_text().splitlines()]
    if frames[0]['t_obs']!=0 or not np.allclose(np.diff([f['t_obs'] for f in frames]),.2):
        raise ValueError('Video expects verified 0.2 s action-step clock')
    font=ImageFont.truetype(FONT,19);small=ImageFont.truetype(FONT,16);title=ImageFont.truetype(FONT,23)
    w,h=1600,1120;indices=[r['frame_index'] for r in rows]
    cmd=['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','5','-i','-',
         '-an','-c:v','libx264','-threads','4','-preset','veryfast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'episode.mp4')]
    ledger=[]
    with (out/'ffmpeg.log').open('w') as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        try:
            for i,f in enumerate(frames):
                ri=int(np.searchsorted(indices,i,side='right')-1);r=rows[max(ri,0)];dest=out/r['directory']
                canvas=Image.new('RGB',(w,h),'#14202b');d=ImageDraw.Draw(canvas)
                d.multiline_text((16,10),'\n'.join(textwrap.wrap(entry['instruction'],110)),font=title,fill='white')
                labels=['Recorded RGB (not model control)','Detector input at keyframe','Frozen prior semantic map','Demonstration GT (not input)']
                paths=[raw/f['rgb'], out/r['input_images'][-1]['path'] if r['input_images'] else dest/'source.png',
                       out/r['map_input_path'] if r.get('map_input_path') else dest/'not-sent-map.png',dest/'reference-bev.png']
                for j,(label,path) in enumerate(zip(labels,paths)):
                    x=12+j*397;d.text((x,95),label,font=small,fill='#90ddff')
                    with Image.open(path) as im:canvas.paste(ImageOps.contain(im.convert('RGB'),(385,385)),(x,125))
                age=f['t_obs']-r['source_time']
                text=f"{entry['uid']} | source {i} t={f['t_obs']:.1f}s | output frame {r['frame_index']} age={age:.1f}s | {r['status']}"
                d.text((16,535),text,font=font,fill='#ffd180')
                y=580
                if isinstance(r['answer'],dict):
                    for key in ['landmark','local','route']:
                        value=r['answer'].get(key,'missing');value=json.dumps(value) if not isinstance(value,str) else value
                        lines=textwrap.wrap(key+': '+value,125);d.multiline_text((16,y),'\n'.join(lines),font=font,fill='white',spacing=5);y+=len(lines)*27+18
                else:d.multiline_text((16,y),'\n'.join(textwrap.wrap(r['raw'] or 'No generated response',125)),font=font,fill='white')
                display_wp=[[round(float(v),2) for v in point] for point in r['geometry_gt']['waypoint']]
                d.text((16,1000),'waypoint GT (rounded display): '+json.dumps(display_wp),font=small,fill='#66e4e4')
                d.text((16,1045),'Offline source-time replay. RGB-D + ideal odom. Rejected output is displayed, never silently repaired.',font=small,fill='#ffd180')
                if i==0:canvas.save(out/'poster.jpg')
                proc.stdin.write(canvas.tobytes());ledger.append(dict(video_frame=i,source_frame=i,source_time=f['t_obs'],output_frame=r['frame_index'],output_age_s=age,status=r['status']))
        finally:
            proc.stdin.close()
            code=proc.wait()
        if code:raise RuntimeError('Video encoding failed')
    write_json(out/'video-frames.json',ledger)
    subprocess.run(['ffmpeg','-v','error','-i',str(out/'episode.mp4'),'-f','null','-'],check=True)


def build(root):
    root=Path(root).resolve();info=verify(root);reports=[]
    for entry in info['episodes']:
        if not (Path(entry['folder'])/'plus/final-map.json').exists():continue
        out,rows,report=episode_review(root,entry);video(out,entry,rows);reports.append(report)
    total=dict(episodes=reports,planned=sum(e['frames'] for e in info['episodes']),
        requested=sum(r['requested'] for r in reports),returned=sum(r['returned'] for r in reports),
        format_valid=sum(r['format_valid'] for r in reports),human_approved=0,human_reviewed=0,
        rule_valid=sum(r['rule_valid'] for r in reports),
        cost_cny_estimate=sum(r['cost_cny_estimate'] for r in reports),training=False,
        all_episodes_completed=len(reports)==len(info['episodes']),semantic_accuracy=None)
    if (root/'postpilot-completion-guard.json').exists():
        revalidation=read_json(root/'postpilot-completion-guard.json')
        total['postpilot_revalidation']={k:revalidation[k] for k in ['strict_rule_pass','context_affected_frames','strict_pass_without_flagged_context','new_api_calls','original_history_not_replayed']}
    write_json(root/'report.json',total)
    write_json(root/'experiment.json',dict(mode='teacher_annotation',run_id=root.name,
        source_dataset='R2R-VLN-CE train-derived teacher comparison episodes, not blind test',
        control_source='verified demonstration action replay in Habitat; no teacher control',
        context='self-recursive Route/Local and independent per-episode maps',
        camera='per-episode sensor_config in source episode; K/T_body_camera archived for every perception frame',
        odometry='ideal_simulator_odometry',clock='synthetic_action_step_clock, 0.2 s per action',
        text_fields_generated_by_teacher=['landmark','local','route'],
        geometric_fields_generated_by_teacher=False,future_geometry_sent_to_teacher=False,
        vocabulary=read_json(root/'vocabulary.json'),teacher_config=read_json(root/'teacher-config.json'),
        keyframe_selection='pre-existing reference-coverage schedule; every scheduled frame included',
        video_alignment='source time; not API arrival time',human_approved=False,
        limitations=['No exhaustive instance or semantic text GT; no mask IoU or absolute object-center accuracy',
                     'No odometry noise, real-world SLAM, realtime control, or navigation success evaluation',
                     'Development showed category confusions; format-valid proposals can still be wrong',
                     'GT XY does not encode stair height or pure-rotation yaw']))
    parts=['<!doctype html><meta charset="utf-8"><style>'+CSS+'</style><h1>R2R · Qwen Plus 完整任务标注试点</h1>',
        '<p>4 个预定任务 / 40 个关键帧；独立室内词表；RGB-D + 理想 odom。模型未控制车辆，标签未获人工批准。</p>',
        '<p><a href="REPORT.md">报告</a> · <a href="report.json">统计</a> · <a href="pilot.json">固定清单</a> · <a href="teacher-config.json">教师配置</a> · <a href="vocabulary.json">室内词表</a> · <a href="development-review.json">12 帧预检</a></p>',detail('覆盖与费用',total)]
    if 'postpilot_revalidation' in total:
        v=total['postpilot_revalidation']
        parts.append(f'<p class="warn">原运行格式有效{total["format_valid"]}帧；完成标记修复后独立复核通过{v["strict_rule_pass"]}帧，{v["context_affected_frames"]}个后续帧的上下文需要额外审核。以上都不等于语义正确或训练批准。原响应和地图未重写。</p><p><a href="postpilot-completion-guard.json">独立复核与受影响来源</a></p>')
    for entry in info['episodes']:
        r=next((r for r in reports if r['episode_id']==entry['uid']),None)
        parts.append('<section><h2>'+esc(entry['uid'])+'</h2><p>'+esc(entry['instruction'])+'</p>')
        if r:
            base=f'episodes/{entry["uid"]}/review'
            parts.append(f'<p><a href="{base}/index.html">完整输入、输出与逐帧人工审核</a> · <a href="{base}/episode.mp4">下载视频</a></p><p>返回 {r["returned"]}/{r["planned"]}；格式有效 {r["format_valid"]}；入图事件 {r["accepted_events"]}；独立地标 {r["unique_landmarks"]}；估算 ¥{r["cost_cny_estimate"]:.6f}</p><video controls preload="metadata" src="{base}/episode.mp4" poster="{base}/poster.jpg"></video>')
        else:parts.append('<p>未完成；不隐藏计划任务。</p>')
        parts.append('</section>')
    (root/'index.html').write_text('\n'.join(parts))
    md=['# R2R Plus 语义标注试点报告','',
        '四个固定教师比较任务；完整关键帧；独立室内26类词表。RGB-D + 理想局部 odom，Habitat示范动作回放，不是模型控制。',
        '',f"请求 {total['requested']}/{total['planned']}，返回 {total['returned']}，格式通过 {total['format_valid']}，原运行规则通过 {total['rule_valid']}。人工批准 0；不能将格式通过视为正确标签。",
        '', '| Episode | 返回/计划 | 格式/原规则有效 | 入图事件/独立地标 | 估算人民币 |', '|---|---:|---:|---:|---:|']
    for r in reports:
        uid=r['episode_id'];md.append(f"| [{uid}](episodes/{uid}/review/index.html) | {r['returned']}/{r['planned']} | {r['format_valid']}/{r['rule_valid']} | {r['accepted_events']}/{r['unique_landmarks']} | ¥{r['cost_cny_estimate']:.6f} |")
    md+=['',f"总估算费用 ¥{total['cost_cny_estimate']:.6f}，包含失败请求。按冻结的输入¥2/输出¥8每百万tokens基础价估算，非账单，未扣优惠/缓存折扣；思考用量包含在API返回的输出用量中。",
         '', '## 审核与限制', '',
         '- 单帧失败保留原文并跳过，不自动修复；最近有效Route/Local继续递归。各任务地图独立。',
         '- 12帧感知预检通过尺寸/来源检查，但发现台球桌→bed、吊扇→lamp等混淆；词表保持计划版本，未借试点答案调参。',
         '- 完成事件为空。本轮原运行规则只识别精确DONE；发现后已补上正文DONE标记校验，供后续运行使用。已保存响应另做独立复核，不改写原历史、地图或请求；原地图仍包含当时接受的证据。',
         '- Local可能合理但超长，格式失败与语义错误分开。无人工独立真值，不报语义准确率或mask IoU。',
         '- waypoint/ref_line来自已核验未来示范，仅作GT。XY地面投影不表达楼梯高度及纯转向yaw。',
         '- 当前/历史RGB、实际检测叠图和历史地图全部展示；彩色mask与GT是审核辅助。',
         '- 视频按源时间更新，合成时钟每动作0.2秒；API延迟另报，不是实时在线演示。',
         '- 四个任务仍属回归/比较集合，未开展训练、扩批或120个任务标注。',
         '', '逐帧审核页可以下载人工意见JSON；原文与建议修订分开，默认不批准。']
    if 'postpilot_revalidation' in total:
        v=total['postpilot_revalidation'];md+=['','## 运行后完成标记修复与复核','',
            f"修复后独立规则复核通过 {v['strict_rule_pass']}/40；{v['context_affected_frames']} 帧的历史上下文含此前被新规则拒绝的完成声明，单独标记待核验。仅格式/规则通过且没有该历史标记的为 {v['strict_pass_without_flagged_context']}/40，仍不是语义准确率或批准标签数。",
            '', '没有重发API，没有将重新生成的地图冒充当时输入。详见 [逐帧复核](postpilot-completion-guard.json)。']
    findings=root/'findings.json'
    if findings.exists():md+=['','## 观察到的具体问题','']+['- '+v for v in read_json(findings)['points']]
    (root/'REPORT.md').write_text('\n'.join(md)+'\n')
    return total


def audit(root):
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.paths=[]
        def handle_starttag(self,tag,attrs):
            self.paths.extend(v for k,v in attrs if k in ('href','src','poster') and v)
    root=Path(root).resolve();info=verify(root);checked=0;videos=[]
    for page in root.rglob('review/index.html'):
        parser=Links();parser.feed(page.read_text())
        for link in parser.paths:
            if link.startswith(('#','http:','https:','data:')):continue
            if not (page.parent/link.split('#')[0]).exists():raise ValueError('Broken review link: '+link)
            checked+=1
        for path,sha in read_json(page.parent/'source-hashes.json').items():
            if file_sha256(path)!=sha:raise ValueError('Review source modified: '+path)
        rows=read_json(page.parent/'rows.json');ledger=read_json(page.parent/'video-frames.json')
        indices=[r['frame_index'] for r in rows]
        for f in ledger:
            expected=max(i for i in indices if i<=f['source_frame'])
            if f['output_frame']!=expected or f['output_age_s']<0:raise ValueError('Video output alignment failed')
        if not set(indices)<=set(f['source_frame'] for f in ledger):raise ValueError('Video omitted a keyframe')
        if any(h['approved'] for h in read_json(page.parent/'human-review.json')):raise ValueError('Unexpected auto-approval')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=nb_frames,width,height',
            '-of','json',str(page.parent/'episode.mp4')],text=True))
        if int(probe['streams'][0]['nb_frames'])!=len(ledger):raise ValueError('Video frame count mismatch')
        videos.append(dict(episode_id=page.parent.parent.name,frames=len(ledger),probe=probe))
    if len(videos)!=len(info['episodes']):raise ValueError('Not all episodes have review videos')
    parser=Links();parser.feed((root/'index.html').read_text())
    for link in parser.paths:
        if link.startswith(('#','http:','https:','data:')):continue
        if not (root/link.split('#')[0]).exists():raise ValueError('Broken index link: '+link)
        checked+=1
    result=dict(passed=True,html_links_checked=checked,videos=videos,expected_keyframes=info['planned_requests'],
        human_approved=0,source_hashes_unchanged=True)
    write_json(root/'review-audit.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--audit',action='store_true')
    args=p.parse_args();(audit if args.audit else build)(args.root)
