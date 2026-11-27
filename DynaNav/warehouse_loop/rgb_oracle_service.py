"""Periodic source-bound Oracle service. No endpoint-completion trigger or fallback.

codex-session exchanges immutable request folders with a human/session; it is
not a simulated model. Qwen is an actual paid API call with shared budget guard.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
import math
import numpy as np
from .live_metrics import atomic_json
from ticvla.warehouse.navigation.rgb_waypoint import PeriodicLatestRequest, PROTOCOL, project_waypoints
from ticvla.warehouse.dataset_tools.credentials import load_credentials
from ticvla.warehouse.dataset_tools.semantic_annotation import send_request,parse_api_response
from demo.warehouse_navigation.rgb_oracle_review import SYSTEM


def run(root,provider):
    root=Path(root);nav=root/'navigation';out=nav/'rgb-oracle';out.mkdir(exist_ok=True)
    cfg=json.loads(Path('configs/dataset_preparation/qwen_plus_teacher.v6.json').read_text())
    cfg['authorized_evaluation_run']=dict(id=root.name+'-rgb-oracle',reason='Explicit RGB Oracle live launch')
    if provider=='qwen3.7-plus':load_credentials('/home/xinglinlu/.config/llm/credentials.env')
    previous=None;previous_source=None;task_id=None;seen=None;future=None;job=None;seq=0
    schedule=PeriodicLatestRequest(2)
    log=(out/'events.jsonl').open('a',buffering=1)
    def call(inp,folder):
        if provider=='codex-session':
            # Session writes only model fields; it cannot supply a source pose.
            answer=folder/'oracle-answer.json'
            end=time.monotonic()+180
            while not answer.exists():
                if time.monotonic()>end or (root/'cancel.json').exists():raise TimeoutError('No session answer')
                time.sleep(.1)
            return answer.read_text()
        content=[dict(type='text',text=inp['user'])]
        for p in inp['images']:
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(Path(p).read_bytes()).decode())))
        req=dict(model=cfg['model'],messages=[dict(role='system',content=SYSTEM),dict(role='user',content=content)],
                 response_format=dict(type='json_object'),enable_thinking=True,thinking_budget=2048,max_tokens=8192)
        api=send_request(req,cfg);atomic_json(folder/'api-response.json',api)
        return parse_api_response(api)
    with ThreadPoolExecutor(max_workers=1) as pool:
        while not (root/'cancel.json').exists() and not (nav/'task-finished.json').exists():
            now=time.monotonic()
            packet_path=nav/'rgb-oracle-packet.json'
            if packet_path.exists():
                packet=json.loads(packet_path.read_text());source=packet['source']
                if source['image_id']!=seen and now-source['source_monotonic']<=2:
                    if schedule.offer(now,dict(image_id=source['image_id'],packet=packet)):
                        seen=source['image_id'];log.write(json.dumps(dict(event='scheduled',image_id=seen,replaced=schedule.replaced))+'\n')
            if future is not None and future.done():
                source=job['packet']['source'];folder=job['folder']
                try:
                    raw=future.result();(folder/'raw.txt').write_text(raw);answer=json.loads(raw)
                    if set(answer)!={'landmark','local','route','waypoint'}:raise ValueError('Wrong fields')
                    if not isinstance(answer['landmark'],list) or len(answer['landmark'])>3 or any(x not in job['packet']['categories'] for x in answer['landmark']):raise ValueError('Invalid landmark')
                    if not isinstance(answer['local'],str) or len(answer['local'])>90 or not isinstance(answer['route'],str) or len(answer['route'])>600:raise ValueError('Invalid text')
                    if answer['route'].strip().upper() in ('DONE','STOP'):raise ValueError('Completion unverified')
                    current=json.loads((nav/'task.json').read_text())
                    if current['task_id']!=job['task']['task_id']:raise ValueError('Task changed')
                    age=time.monotonic()-source['source_monotonic']
                    if age>5:raise ValueError(f'Source expired {age:.3f}s; no motion publication')
                    project_waypoints([],source)
                    # Valid recursive text survives a geometric rejection; the
                    # runtime clears guidance and handles semantics separately.
                    try:projection=project_waypoints(answer['waypoint'],source)
                    except ValueError as exc:projection=dict(status='rejected',error=str(exc))
                    atomic_json(folder/'projection.json',projection)
                    proposal=dict(**answer,request_id=job['id'],task_id=current['task_id'],instruction=current['instruction'],
                                  source_image_id=source['image_id'],provider=provider,waypoint_protocol=PROTOCOL)
                    atomic_json(nav/'proposal.json',proposal)
                    previous=answer;previous_source=source;atomic_json(folder/'result.json',dict(status='published_for_runtime_checks',source_age_s=age))
                except Exception as exc:
                    atomic_json(folder/'result.json',dict(status='failed',error=str(exc),source_age_s=time.monotonic()-source['source_monotonic']))
                    if not isinstance(exc,(ValueError,TypeError)):
                        # Transport/budget failures stop new paid calls, never retry.
                        log.write(json.dumps(dict(event='provider_stopped',error=type(exc).__name__))+'\n');break
                schedule.finish(source['image_id']);future=None
            pending=schedule.take() if future is None else None
            if pending is not None:
                packet=pending['packet'];source=packet['source'];task=json.loads((nav/'task.json').read_text())
                if task_id!=task['task_id']:previous=None;previous_source=None;task_id=task['task_id']
                if now-source['source_monotonic']>2:
                    schedule.finish(source['image_id']);continue
                seq+=1;rid=f'rgb-{seq:06d}';folder=out/rid;folder.mkdir()
                for name,p in [('rgb.png',source['rgb']),('detections.png',packet['overlay']),('map.png',packet['map'])]:shutil.copy2(p,folder/name)
                local='unknown';motion='none'
                if previous_source is not None and 0<=source['t_obs']-previous_source['t_obs']<=30:
                    delta=np.linalg.inv(previous_source['T_odom_body'])@np.array(source['T_odom_body'])
                    local=previous['local'];motion=f"{source['t_obs']-previous_source['t_obs']:.1f}s, x={delta[0,3]:.3f}, y={delta[1,3]:.3f}, yaw={math.atan2(delta[1,0],delta[0,0]):.3f}"
                user=f"Previous Route: {previous['route'] if previous else task['instruction']}\nPrevious Local: {local}\nMotion: {motion}\nCurrent detectable categories: {', '.join(packet['categories']) or 'none'}\nCompletion evidence: unverified\nImages: current RGB, current segmentation overlay, prior observed semantic map."
                inp=dict(system=SYSTEM,user=user,source=source,task=task,images=[str(folder/n) for n in ('rgb.png','detections.png','map.png')],
                         previous_source=previous_source,protocol=PROTOCOL)
                atomic_json(folder/'input.json',inp);job=dict(packet=packet,task=task,id=rid,folder=folder)
                future=pool.submit(call,inp,folder)
            time.sleep(.03)
    log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--provider',choices=['qwen3.7-plus','codex-session'],required=True);a=p.parse_args();run(a.root,a.provider)
