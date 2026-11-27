"""Frozen causal replay, RGB-pixel Oracle comparison. Does not move Isaac."""
import argparse
import base64
import copy
import html
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image, ImageDraw
from ticvla.warehouse.dataset_tools.core import read_json, write_json
from ticvla.warehouse.schema import file_sha256
from ticvla.warehouse.geometry import LocalMapBuilder
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.mapping import MapStore, SourceObservation
from ticvla.warehouse.perception.realtime import consume_masks, consume_text
from DynaNav.warehouse_loop.live_mapping import LiveBEV
from ticvla.warehouse.navigation.rgb_waypoint import project_waypoints, PROTOCOL

SYSTEM = '''Return only an English JSON object with exactly landmark, local, route and waypoint.
landmark: Up to three useful visible object categories from Current detectable categories. No IDs. Use [] if none.
local: Current position and heading relative to supported visible or mapped landmarks, within 90 characters. Mark memory-only localization Inferred. Use unknown if unsupported.
route: Remaining navigation commands only, within 600 characters. Preserve ALL unfinished Previous Route steps in order, including the final STOP. Delete a step only with observation or motion evidence. Never describe scene layout as a route. An unseen target may remain a search goal; do not invent its location. STOP alone requires verified completion, otherwise retain remaining commands.
waypoint: Exactly five ordered [u,v] PIXEL pairs on visible navigable FLOOR in CURRENT original RGB, or [] when stopped or no safe visible continuation can be specified. Original RGB is 960 by 720; origin top left; u right, v down. These are NOT metres or map coordinates. Order near to far along the intended local route. All points and the connecting route must remain inside the visible image, on unobstructed floor, not on signs or shelves. Show only the currently visible route prefix, not the whole offscreen task. The system uses the same-frame depth to project these pixels to 3D and fits a controller curve. Never invent depth or an offscreen point. If the required turn is not visible, keep that turn in route and return []. No extra rotation action is inferred from empty waypoints.
The map is accumulated PAST observed geometry and tentative semantic objects, in odom coordinates, not hidden scene truth. Red marker is current pose; the heading is printed on the map. Blue is observed past travel; background is unobserved, not proven free. OCR may be wrong. Original Instruction is only used to initialize Previous Route for the first frame. Relative motion is current body in previous Local frame, metres x forward y left and counterclockwise yaw radians.'''


def batch_from(path):
    f=read_json(path);s=f['source'];inst=copy.deepcopy(f['instances'])
    for i,v in enumerate(inst):
        v['mask']=np.load(path.parent/f'mask-{i:03}.npz')['mask']
        v.pop('texts',None)
    return f,dict(instances=inst,source_image_id=s['image_id'],rgb_hash=s['rgb_hash'],
                  provenance=f['provenance'],elapsed_s=f['segmentation_s'])


def prepare(source, output):
    source,root=Path(source),Path(output)
    if root.exists():raise ValueError('New output directory required')
    root.mkdir(parents=True)
    paths=sorted((source/'navigation/perception').glob('*/frame.json'))
    rows=[(p,read_json(p)) for p in paths]
    # Predeclare frame selection by time + measured rotation, never model success.
    selected=[];last_t=-1e9;last_yaw=None;last_task=None
    usable=[(p,f) for p,f in rows if 77<=int(p.parent.name)<=1302]
    for p,f in usable:
        s=f['source'];n=int(p.parent.name);task='mapping' if n<=946 else 'target'
        T=np.array(s['T_odom_body']);yaw=math.atan2(T[1,0],T[0,0])
        dyaw=0 if last_yaw is None else abs(math.atan2(math.sin(yaw-last_yaw),math.cos(yaw-last_yaw)))
        if task!=last_task or s['t_obs']-last_t>=12 or dyaw>=math.radians(30) or p==usable[-1][0] or n==946:
            selected.append(dict(frame=n,task=task,t_obs=s['t_obs'],perception=str(p)))
            last_t=s['t_obs'];last_yaw=yaw;last_task=task
    # Include the final recorded stop even if the detector did not process it.
    # A missing detection frame remains explicit (no borrowed masks).
    plan=read_json(source/'review/decision-plan.json')
    instructions=dict(mapping=plan['loop_instruction'],target='Navigate to sign 04/5R in the farther second aisle and STOP beside it.')
    write_json(root/'experiment.json',dict(protocol=PROTOCOL,source=str(source),selected=selected,instructions=instructions,
        selection_rule='12 s maximum nominal source interval plus >=30 degree yaw changes; task boundaries; cached same-frame perception only',
        replay_control_source='prior Codex metric decision demo, NOT these Oracle predictions',
        mode='oracle_replay',decoder='not_run',flow='not_run',closed_loop_navigation=False,
        input_map='shared independent causal reconstruction from original actual perception policy, not each Oracle output',
        oracle_field_sources='new visual pixel annotations; prior metric waypoint scripts excluded',
        task_stop_coverage='Last processed perception before final stop; final STOP not invented',
        inference_interval_online_s=2,source_max_age_online_s=5))
    (root/'system.txt').write_text(SYSTEM)
    cfg=read_json('configs/dataset_preparation/qwen_plus_teacher.v6.json')
    cfg.update(authorized_evaluation_run=dict(id=root.name,reason='User requested RGB pixel waypoint comparison for both warehouse instructions'))
    write_json(root/'qwen-config.json',cfg)
    store=MapStore(root/'causal-map',array_compression_level=1);bev=LiveBEV()
    builder=LocalMapBuilder(WarehouseConfig(depth_stride=24,map_resolution=.1,map_size_m=16,map_max_age=1e9))
    by_frame={r['frame']:r for r in selected}
    for p,f in rows:
        s=f['source'];n=int(p.parent.name)
        if n>selected[-1]['frame']:break
        if n in by_frame:
            dest=root/'inputs'/f'{n:08d}';dest.mkdir(parents=True)
            # The map contains strictly prior source evidence. Add current pose
            # without adding current RGB-D geometry or semantic evidence.
            bev.trail.append(np.array(s['T_odom_body'])[:2,3].tolist())
            bev.render(store.landmark_view()['landmarks'],dest/'map.png')
            pose=np.array(s['T_odom_body']);im=Image.open(dest/'map.png');d=ImageDraw.Draw(im)
            d.text((10,8),'CURRENT HEADING: odom yaw %.1f deg'%math.degrees(math.atan2(pose[1,0],pose[0,0])),fill='red')
            im.save(dest/'map.png');bev.trail.pop()
            shutil.copy2(s['rgb'],dest/'rgb.png')
            im=Image.open(s['rgb']).convert('RGB');arr=np.array(im,dtype=float);draws=[]
            _,batch=batch_from(p)
            colors=['orange','cyan','magenta','lime','yellow']
            for i,v in enumerate(batch['instances']):
                from PIL import ImageColor
                mask=v['mask'].astype(bool);color=ImageColor.getrgb(colors[i%5]);arr[mask]=arr[mask]*.65+np.array(color)*.35
                draws.append((v['bbox'],f'd{i} {v["category"]}',colors[i%5]))
            im=Image.fromarray(arr.astype('uint8'));d=ImageDraw.Draw(im)
            for box,label,color in draws:d.rectangle(box,outline=color,width=2);d.text(tuple(box[:2]),label,fill='white',stroke_width=1,stroke_fill='black')
            im.save(dest/'detections.png')
            write_json(dest/'source.json',s)
            write_json(dest/'audit.json',dict(source=s,perception=str(p),map_version=store.version,
                map_source_cutoff=max([x['t_obs'] for x in store.records('observations')],default=None),
                map_landmarks=store.landmark_view()['landmarks'],categories=list(dict.fromkeys(v['category'] for v in batch['instances'])),
                images={k:file_sha256(dest/k) for k in ('rgb.png','detections.png','map.png')}))
        obs=SourceObservation(**{k:s[k] for k in SourceObservation.__dataclass_fields__})
        bound=store.add_observation(obs)
        _,batch=batch_from(p);selection=f['selection'];results=consume_masks(store,bound,batch,selection)
        ocr={**f['ocr'], 'source_image_id':bound['image_id'],'rgb_hash':bound['rgb_hash']}
        consume_text(store,bound,batch,results,ocr)
        builder.add(depth=np.load(s['depth']),**{k:s[k] for k in ('K','T_body_camera','T_odom_body','t_obs','odom_epoch')})
        bev.add(builder.frames[-1]['points'],np.array(s['T_odom_body']));builder.frames=[]
    store.close()
    print('Prepared',len(selected),'frames',flush=True)


def user_input(root,row,previous,prev_source):
    n=row['frame'];inp=root/'inputs'/f'{n:08d}';s=read_json(inp/'source.json');audit=read_json(inp/'audit.json')
    instruction=read_json(root/'experiment.json')['instructions'][row['task']]
    motion='none';local='unknown'
    if prev_source is not None:
        delta=np.linalg.inv(prev_source['T_odom_body'])@np.array(s['T_odom_body'])
        age=s['t_obs']-prev_source['t_obs']
        if age<=30:
            local=previous['local'];motion=f"{age:.1f}s, x={delta[0,3]:.3f}, y={delta[1,3]:.3f}, yaw={math.atan2(delta[1,0],delta[0,0]):.3f}"
    user=f"Previous Route: {previous['route'] if previous else instruction}\nPrevious Local: {local}\nMotion: {motion}\nCurrent detectable categories: {', '.join(audit['categories']) or 'none'}\nCompletion evidence: {'confirmed' if row.get('completion_confirmed') else 'unverified'}\nImages in order: current original RGB, current segmentation overlay, prior semantic map."
    return dict(user_text=user,source=s,original_instruction_audit_only=instruction,
                images=[str(inp/k) for k in ('rgb.png','detections.png','map.png')],
                previous=previous,previous_source=prev_source,map_cutoff=audit['map_source_cutoff'])


def finish_frame(root,provider,row,inp,raw,elapsed=None,api=None):
    dest=root/provider/f"{row['frame']:08d}";dest.mkdir(parents=True,exist_ok=True)
    (dest/'raw.txt').write_text(raw);write_json(dest/'input.json',inp)
    result=dict(frame=row['frame'],task=row['task'],t_obs=row['t_obs'],request_elapsed_s=elapsed,
                status='failed',semantic_review='review_pending',api_usage=(api or {}).get('usage'),decoder='not_run',
                actual_control='not_executed',format_valid=False,projection_status='not_run')
    parsed=None
    try:
        parsed=json.loads(raw)
        if set(parsed)!={'landmark','local','route','waypoint'}:raise ValueError('Wrong output fields')
        if not isinstance(parsed['landmark'],list) or len(parsed['landmark'])>3:raise ValueError('Landmark list invalid')
        cats=read_json(root/'inputs'/f"{row['frame']:08d}"/'audit.json')['categories']
        if any(not isinstance(x,str) or x not in cats for x in parsed['landmark']):raise ValueError('Landmark not in current candidates')
        if not isinstance(parsed['local'],str) or not isinstance(parsed['route'],str):raise ValueError('Text fields invalid')
        if len(parsed['local'])>90 or len(parsed['route'])>600:raise ValueError('Text length exceeded')
        if parsed['route'].strip().upper() in ('DONE','STOP') and not row.get('completion_confirmed'):raise ValueError('No verified completion evidence at sampled source')
        uv=np.asarray(parsed['waypoint'],float)
        if uv.size and (uv.shape!=(5,2) or not np.isfinite(uv).all()):raise ValueError('Waypoint shape invalid')
        result.update(format_valid=True,status='format_valid_unreviewed');write_json(dest/'parsed.json',parsed)
        try:
            projection=project_waypoints(parsed['waypoint'],inp['source'])
            result['projection_status']=projection['status'];write_json(dest/'projection.json',projection)
        except ValueError as exc:result.update(projection_status='rejected',projection_error=str(exc))
    except (ValueError,TypeError) as exc:result['error']=str(exc)
    if api is not None:
        u=api.get('usage',{});result['estimated_CNY']=(u.get('prompt_tokens',0)*2+u.get('completion_tokens',0)*8)/1e6
    im=Image.open(inp['images'][0]).convert('RGB');d=ImageDraw.Draw(im)
    if parsed is not None:
        try:
            for i,p in enumerate(parsed['waypoint']):
                u,v=map(float,p)
                if 0<=u<im.width and 0<=v<im.height:
                    d.ellipse((u-5,v-5,u+5,v+5),fill='orange');d.text((u+8,v),f'W{i+1}',fill='yellow',stroke_width=2,stroke_fill='black')
        except (ValueError,TypeError):pass
    if (dest/'projection.json').exists():
        pr=read_json(dest/'projection.json')
        if pr.get('curve_pixels'):d.line([tuple(p) for p in pr['curve_pixels']],fill='cyan',width=2)
    im.save(dest/'waypoints.png');write_json(dest/'result.json',result)
    return parsed if result['format_valid'] else None


def run_qwen(root,credentials):
    from ticvla.warehouse.dataset_tools.credentials import load_credentials
    from ticvla.warehouse.dataset_tools.semantic_annotation import send_request,parse_api_response
    root=Path(root);load_credentials(credentials);cfg=read_json(root/'qwen-config.json');exp=read_json(root/'experiment.json')
    previous=None;prev_source=None;task=None
    for row in exp['selected']:
        if row['task']!=task:previous=None;prev_source=None;task=row['task']
        dest=root/'qwen3.7-plus'/f"{row['frame']:08d}"
        if (dest/'result.json').exists():
            r=read_json(dest/'result.json')
            if r['format_valid']:previous=read_json(dest/'parsed.json');prev_source=read_json(dest/'input.json')['source']
            continue
        if dest.exists():raise ValueError('Interrupted paid request: inspect before resuming')
        inp=user_input(root,row,previous,prev_source);dest.mkdir(parents=True);write_json(dest/'input.json',inp)
        content=[dict(type='text',text=inp['user_text'])]
        for path in inp['images']:
            data=base64.b64encode(Path(path).read_bytes()).decode()
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+data)))
        req=dict(model=cfg['model'],messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],
                 max_tokens=cfg['max_output_tokens'],enable_thinking=True,thinking_budget=2048,
                 response_format=dict(type='json_object'))
        write_json(dest/'request-audit.json',dict(model=cfg['model'],system=SYSTEM,user=inp['user_text'],
                   image_hashes=[file_sha256(p) for p in inp['images']],thinking_budget=2048,max_tokens=cfg['max_output_tokens']))
        start=time.monotonic()
        try:
            api=send_request(req,cfg);write_json(dest/'api-response.json',api)
        except Exception as exc:
            write_json(dest/'failure.json',dict(error=type(exc).__name__+': '+str(exc),elapsed_s=time.monotonic()-start))
            print('Provider stopped:',type(exc).__name__,flush=True);break
        raw=api.get('choices',[{}])[0].get('message',{}).get('content','')
        try:parse_api_response(api)
        except ValueError as exc:
            (dest/'raw.txt').write_text(raw);write_json(dest/'failure.json',dict(error=str(exc)));continue
        parsed=finish_frame(root,'qwen3.7-plus',row,inp,raw,time.monotonic()-start,api)
        if parsed is not None:previous=parsed;prev_source=inp['source']
        print(row['task'],row['frame'],read_json(dest/'result.json'),flush=True)
        report(root)
    report(root)


def report(root):
    root=Path(root);exp=read_json(root/'experiment.json');summaries={}
    for provider in exp.get('providers', ('qwen3.7-plus','codex-session')):
        parts=['<!doctype html><meta charset="utf-8"><style>body{background:#18232e;color:#eee;font:16px sans-serif;padding:20px}img{width:32%}pre{white-space:pre-wrap}a{color:#8cf}section{border-top:1px solid #888;padding:15px}</style>',
            f'<h1>{provider}: RGB pixel waypoint review</h1>',
            '<p>完整两任务固定回放；不是本次预测驱动的运动。cyan 是深度投影后 PCHIP 曲线，不是 Decoder 输出。地图为共享的因果重建基线。所有语义等待人工审核。</p>']
        parts.append('<p><a href="../REPORT.md">结论与限制</a> · <a href="../system.txt">完整 SYSTEM</a> · <a href="report.json">统计 JSON</a></p>')
        for task in ('mapping','target'):
            if (root/provider/f'{task}.mp4').exists():parts.append(f'<h2>{task} 连续回放</h2><video controls preload="metadata" width="900" src="{task}.mp4"></video>')
        results=[]
        for row in exp['selected']:
            n=f"{row['frame']:08d}";folder=root/provider/n;inp=root/'inputs'/n
            result=read_json(folder/'result.json') if (folder/'result.json').exists() else dict(frame=row['frame'],status='not_run_or_provider_failure')
            results.append(result)
            parts.append(f'<section id="frame-{n}"><h2>{row["task"]} / {n} / {row["t_obs"]:.1f}s</h2><p>{html.escape(exp["instructions"][row["task"]])}</p>')
            parts.append(f'<p><a href="../inputs/{n}/audit.json">来源与地图证据审计（未全部发送给模型）</a></p>')
            if (folder/'input.json').exists():
                context=read_json(folder/'input.json')
                prior=context.get('previous')
                previous_route=prior['route'] if prior else context['original_instruction_audit_only']
                generated=read_json(folder/'parsed.json').get('route') if (folder/'parsed.json').exists() else 'INVALID: see raw output'
                parts.append('<p><b>Previous Route</b>: '+html.escape(previous_route)+'</p><p><b>Generated Route</b>: '+html.escape(generated)+'</p>')
            for name in ('rgb.png','detections.png','map.png'):parts.append(f'<a href="../inputs/{n}/{name}"><img src="../inputs/{n}/{name}"></a>')
            if (folder/'waypoints.png').exists():parts.append(f'<img src="{n}/waypoints.png">')
            if (folder/'projection-3d.png').exists():parts.append(f'<a href="{n}/projection-3d.png"><img src="{n}/projection-3d.png"></a>')
            for name in ('raw.txt','input.json','projection.json','controller-diagnostic.json','result.json','failure.json'):
                p=folder/name
                if p.exists():parts.append(f'<details {"open" if name=="raw.txt" else ""}><summary>{name}</summary><pre>{html.escape(p.read_text())}</pre></details>')
            parts.append(f'<p>{html.escape(str(result))}</p></section>')
        times=[r['request_elapsed_s'] for r in results if r.get('request_elapsed_s') is not None]
        stats=dict(planned=len(results),returned=sum((root/provider/f"{r['frame']:08d}"/'raw.txt').exists() for r in results),
            format_valid=sum(r.get('format_valid',False) for r in results),
            projected=sum(r.get('projection_status')=='projected' for r in results),empty=sum(r.get('projection_status')=='empty' for r in results),
            rejected=sum(r.get('projection_status')=='rejected' for r in results),
            request_p50_s=float(np.median(times)) if times else None,request_p95_s=float(np.percentile(times,95)) if times else None,
            within_2s=sum(t<=2 for t in times),within_5s_source_age=sum(t<=5 for t in times),estimated_CNY=sum(r.get('estimated_CNY',0) for r in results) if provider=='qwen3.7-plus' else None,
            human_approved=0,actual_navigation='not_run',reference_decoder='not_run',results=results)
        stats['by_task']={task:dict(planned=sum(x['task']==task for x in exp['selected']),
            returned=sum((root/provider/f"{x['frame']:08d}"/'raw.txt').exists() for x in exp['selected'] if x['task']==task),
            format_valid=sum(r.get('format_valid',False) for r in results if r.get('task')==task),
            projected=sum(r.get('projection_status')=='projected' for r in results if r.get('task')==task),
            empty=sum(r.get('projection_status')=='empty' for r in results if r.get('task')==task),
            rejected=sum(r.get('projection_status')=='rejected' for r in results if r.get('task')==task)) for task in ('mapping','target')}
        (root/provider).mkdir(exist_ok=True);write_json(root/provider/'report.json',stats)
        (root/provider/'index.html').write_text(''.join(parts).replace('<img ','<img loading="lazy" '));summaries[provider]={k:v for k,v in stats.items() if k!='results'}
    write_json(root/'summary.json',summaries)
    (root/'index.html').write_text('<meta charset="utf-8"><h1>RGB pixel waypoint Oracle comparison</h1><p>Offline causal replay, not autonomous navigation acceptance.</p>'+''.join(f'<p><a href="{p}/index.html">{p} 完整报告</a></p>' for p in summaries)+'<pre>'+html.escape(json.dumps(summaries,indent=2))+'</pre>')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','qwen','report']);p.add_argument('--source');p.add_argument('--output',required=True);p.add_argument('--credentials',default='/home/xinglinlu/.config/llm/credentials.env');a=p.parse_args()
    if a.mode=='prepare':prepare(a.source,a.output)
    elif a.mode=='qwen':run_qwen(a.output,a.credentials)
    else:report(a.output)
