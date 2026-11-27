"""Source-bound Codex waypoint demonstration; real RGB-D, masks, OCR and control.

No InternVL, Flow, API teacher, scene catalogue, hidden goal or route oracle.
An external Codex response is committed through navigation/proposal.json.
"""
import argparse
from dataclasses import asdict
import functools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import shutil
import subprocess
import hashlib
import sys
import threading
import time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator

from .live_metrics import atomic_json, complete_lines
from .live_mapping import LiveBEV
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.geometry import LocalMapBuilder, transform_points, render_map_image
from ticvla.warehouse.navigation.geometry import render_local
from ticvla.warehouse.navigation.core import pursuit
from ticvla.warehouse.mapping import MapStore, SourceObservation
from ticvla.warehouse.schema import file_sha256


def fit_waypoints(waypoint):
    p=np.asarray(waypoint,float)
    if p.shape!=(5,2) or not np.isfinite(p).all():raise ValueError('Exactly five finite XY anchors required')
    p=np.vstack([np.zeros(2),p]);p=p[np.r_[True,np.linalg.norm(np.diff(p,axis=0),axis=1)>1e-5]]
    if len(p)<2:raise ValueError('Empty motion; use explicit wait/finish')
    s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    if s[-1]>35:raise ValueError('Waypoints exceed 35m demonstration horizon')
    return PchipInterpolator(s,p,axis=0)(np.linspace(0,s[-1],max(2,int(s[-1]/.04)+1)))


def remaining_curve(curve, pose):
    if len(curve)==0:return np.empty((0,2))
    xy=transform_points(np.c_[curve,np.zeros(len(curve))],np.linalg.inv(pose))[:,:2]
    k=int(np.linalg.norm(xy,axis=1).argmin())
    return xy[k:]


def outside_camera(position, source):
    tf=np.asarray(source['T_odom_body'])@np.asarray(source['T_body_camera'])
    q=transform_points(np.asarray(position)[None],np.linalg.inv(tf))[0]
    if q[2]<=0:return True
    uv=np.asarray(source['K'])@q;uv=uv[:2]/uv[2]
    with Image.open(source['rgb']) as im:w,h=im.size
    return not (0<=uv[0]<w and 0<=uv[1]<h)


def stopped_evidence(history, control):
    rows=list(history.values())[-10:]
    if len(rows)<2 or control.get('reason')!='waypoint_endpoint_reached':
        raise ValueError('Completion requires the controller to reach and stop at the waypoint endpoint')
    end=rows[-1];start=next((x for x in reversed(rows[:-1]) if end['t_obs']-x['t_obs']>=1.),None)
    if start is None:raise ValueError('Need one second of stop observation')
    dt=end['t_obs']-start['t_obs']
    delta=np.linalg.inv(np.asarray(start['T_odom_body']))@np.asarray(end['T_odom_body'])
    speed=float(np.linalg.norm(delta[:2,3])/dt);yaw_rate=abs(math.atan2(delta[1,0],delta[0,0]))/dt
    if speed>.03 or yaw_rate>.03:raise ValueError('Vehicle is still moving')
    return dict(speed_mps=speed,yaw_rate_radps=yaw_rate,interval_s=dt)


def perception_worker(requests,results,weights):
    from ticvla.warehouse.perception.realtime import FixedSegmenter, MobileReader
    try:
        segmenter=FixedSegmenter(weights);reader=MobileReader()
        results.put(dict(ready=True,segmenter=segmenter.provenance,ocr=reader.provenance))
        while True:
            source=requests.get()
            if source is None:break
            batch=segmenter.predict(source);ocr=reader.predict(source)
            results.put(dict(source=source,batch=batch,ocr=ocr))
    except Exception as exc:results.put(dict(error=repr(exc)))


HTML='''<!doctype html><meta charset="utf-8"><title>Codex warehouse waypoint demo</title>
<style>body{background:#16222c;color:#eef;font:16px sans-serif}img{width:32%;vertical-align:top}pre{white-space:pre-wrap}a{color:#7df}</style>
<h2>Warehouse: loop mapping → unseen numbered landmark</h2>
<p>Codex delegate, not trained VLA. RGB-D/YOLO-E/PP-OCR are real. Orange: fitted waypoint curve; blue: observed trajectory; landmarks remain tentative.</p>
<img id="rgb"><img id="map"><img id="plan"><pre id="status"></pre><pre id="response"></pre>
<p><a href="navigation/decisions.jsonl">Codex responses</a> · <a href="navigation/landmarks.json">Observed landmarks</a> · <a href="report.json">Final report</a></p>
<script>async function tick(){try{let s=await(await fetch('navigation/status.json?t='+Date.now())).json();document.querySelector('#status').textContent=JSON.stringify(s,null,2);for(let k of ['rgb','map','plan'])document.querySelector('#'+k).src='navigation/'+k+'.png?t='+s.wall;let r=await(await fetch('navigation/last-response.json?t='+Date.now())).json();document.querySelector('#response').textContent=JSON.stringify(r,null,2)}catch(e){}setTimeout(tick,1000)}tick()</script>'''


def runtime(root,weights):
    from ticvla.warehouse.perception.realtime import consume_masks,consume_text,save_frame
    nav=root/'navigation';raw=root/'capture/raw'
    run_config=json.loads((root/'config.json').read_text()) if (root/'config.json').exists() else {}
    rgb_oracle=bool(run_config.get('rgb_oracle'))
    ctx=mp.get_context('spawn');requests=ctx.Queue(1);results=ctx.Queue(2)
    proc=ctx.Process(target=perception_worker,args=(requests,results,weights));proc.start()
    ready=results.get(timeout=300)
    if 'error' in ready:raise RuntimeError(ready['error'])
    atomic_json(nav/'perception-provenance.json',ready)
    from .codex_demo_control import run as control_run
    controller=ctx.Process(target=control_run,args=(root,));controller.start()
    store=MapStore(nav/'map',array_compression_level=1);bev=LiveBEV()
    cfg=WarehouseConfig(depth_stride=24,map_resolution=.1,map_size_m=16,map_max_age=1e9,footprint_radius=.35)
    builder=LocalMapBuilder(cfg);source=None;grid=None;stream=None;meta=None;index=0
    curve=None;active=None;seen_ids=set();history={};trail=[];distance=0.;phase='mapping'
    last_sem=-1e9;last_view=0;last_control=0;busy=False;landmarks=[];sem_count=0
    key_geometry=[];key_pose=None;last_yaw=None;scan_yaw=0.;scan_complete=False
    (nav/'rgb-oracle-maps').mkdir(exist_ok=True)
    if (nav/'decisions.jsonl').exists():
        for row in (nav/'decisions.jsonl').read_text().splitlines():
            msg=json.loads(row);seen_ids.add(msg['request_id'])
            if msg.get('accepted') and msg.get('waypoint'):
                active=msg;curve=np.load(nav/f"curve-{msg['request_id']}.npz")['odom']
    response_log=(nav/'decisions.jsonl').open('a',buffering=1)
    metrics=(nav/'metrics.jsonl').open('a',buffering=1)
    atomic_json(nav/'ready.json',dict(ready=True,provider='codex-session',trained_VLA=False))
    try:
        while not (nav/'task-finished.json').exists() and not (root/'cancel.json').exists():
            now=time.monotonic()
            if stream is None and (raw/'episode.json').exists():
                meta=json.loads((raw/'episode.json').read_text());stream=(raw/meta['frames_file']).open()
            if stream:
                # Freeze the committed batch. Re-render only its newest source;
                # otherwise an expensive historical-view render can starve control.
                fresh=list(complete_lines(stream))
                for f in fresh:
                    src=SourceObservation(f"{meta['episode_id']}:{index:08d}",str(raw/f['rgb']),str(raw/f['depth']),
                        f['K'],f['T_body_camera'],f['T_odom_body'],f['t_obs'],f['odom_epoch'],f['odom_epoch'],phase)
                    source=store.add_observation(src);source['source_monotonic']=f['source_monotonic']
                    history[source['image_id']]=dict(source);index+=1
                    pose=np.asarray(f['T_odom_body']);xy=pose[:2,3]
                    if rgb_oracle and f is fresh[-1] and not busy and source['t_obs']-last_sem>=1.0:
                        # Freeze prior map BEFORE adding this source's geometry.
                        map_path=nav/'rgb-oracle-maps'/f"{source['image_id'].split(':')[-1]}.png"
                        bev.trail.append(xy.tolist());bev.render(landmarks,map_path);bev.trail.pop()
                        map_image=Image.open(map_path);ImageDraw.Draw(map_image).text((10,8),
                            f"CURRENT HEADING: odom yaw {math.degrees(math.atan2(pose[1,0],pose[0,0])):.1f} deg",fill='red')
                        map_image.save(map_path)
                    if trail:distance+=float(np.linalg.norm(xy-np.asarray(trail[-1])))
                    trail.append(xy.tolist())
                    builder.add(depth=np.load(source['depth']),**{k:f[k] for k in ('K','T_body_camera','T_odom_body','t_obs','odom_epoch')})
                    yaw=float(np.arctan2(pose[1,0],pose[0,0]))
                    if last_yaw is not None and not scan_complete:
                        scan_yaw+=math.atan2(math.sin(yaw-last_yaw),math.cos(yaw-last_yaw))
                    last_yaw=yaw
                    if scan_yaw>=2*math.pi-.035:scan_complete=True
                    current_geometry=builder.frames[-1]
                    delta=None if key_pose is None else np.linalg.inv(key_pose)@pose
                    if delta is None or np.linalg.norm(delta[:2,3])>=.4 or abs(math.atan2(delta[1,0],delta[0,0]))>=.18:
                        key_geometry.append(current_geometry);key_geometry=key_geometry[-48:];key_pose=pose.copy()
                    builder.frames=key_geometry+([] if key_geometry and key_geometry[-1] is current_geometry else [current_geometry])
                    bev.add(builder.frames[-1]['points'],pose)
                if fresh:
                    atomic_json(nav/'source.json',source)
                    if not busy and source['t_obs']-last_sem>=1.0:
                        requests.put(dict(source));busy=True;last_sem=source['t_obs']
            try:
                result=results.get_nowait()
                if 'error' in result:raise RuntimeError(result['error'])
                s,batch,ocr=result['source'],result['batch'],result['ocr']
                selection=dict(selected_categories=[] if rgb_oracle else ['sign','label','fire extinguisher','door'],
                    request_id='codex-demo-mapping-policy',provider='codex-authored-category-policy',event_source='perception',
                    note='Fixed delegate category policy, not an InternVL inference')
                rows=consume_masks(store,s,batch,selection);binding=consume_text(store,s,batch,rows,ocr)
                save_frame(nav/'perception'/s['image_id'].split(':')[-1],s,batch,rows,selection,ocr,binding)
                landmarks=store.landmark_view()['landmarks'];atomic_json(nav/'landmarks.json',landmarks)
                atomic_json(nav/'latest-perception.json',dict(source=s,categories=list(dict.fromkeys(r['category'] for r in rows if r.get('selected') and r['result']['status']=='accepted'))))
                if rgb_oracle:
                    overlay=Image.open(s['rgb']).convert('RGB');pixels=np.asarray(overlay,dtype=float).copy()
                    for inst in batch['instances']:
                        m=inst['mask'].astype(bool);pixels[m]=pixels[m]*.65+np.array([255,165,0])*.35
                    overlay=Image.fromarray(pixels.astype('uint8'));draw=ImageDraw.Draw(overlay)
                    for i,inst in enumerate(batch['instances']):
                        draw.rectangle(inst['bbox'],outline='orange',width=2)
                        draw.text(tuple(inst['bbox'][:2]),f"d{i} {inst['category']}",fill='yellow',stroke_width=1,stroke_fill='black')
                    overlay_path=nav/'perception'/s['image_id'].split(':')[-1]/'candidate-overlay.png';overlay.save(overlay_path)
                    atomic_json(nav/'rgb-oracle-packet.json',dict(source=s,overlay=str(overlay_path),
                        map=str(nav/'rgb-oracle-maps'/f"{s['image_id'].split(':')[-1]}.png"),
                        categories=list(dict.fromkeys(i['category'] for i in batch['instances'])),
                        overlay_kind='same-frame segmentation masks and detector boxes'))
                sem_count+=1;busy=False
                metrics.write(json.dumps(dict(module='perception',image_id=s['image_id'],detections=len(rows),
                    accepted=sum(r['result']['status']=='accepted' for r in rows),source_lag_s=source['t_obs']-s['t_obs'],
                    segmentation_s=batch['elapsed_s'],ocr_s=ocr['elapsed_s']))+'\n')
            except queue.Empty:pass
            proposal=nav/'proposal.json'
            if proposal.exists() and source:
                msg=json.loads(proposal.read_text())
                if msg['request_id'] not in seen_ids:
                    seen_ids.add(msg['request_id'])
                    try:
                        src=history[msg['source_image_id']]
                        if source['t_obs']-src['t_obs']>30:raise ValueError('Proposal source older than 30 simulated seconds')
                        if msg.get('action')=='complete_mapping':
                            if distance<40 or np.linalg.norm(np.asarray(trail[-1])-trail[0])>.8:raise ValueError('Loop closure or minimum travelled distance not established')
                            phase='target_selection';curve=None;active=None
                            candidates=[lm for lm in landmarks if lm.get('identity_text') and outside_camera(lm['position'],source)
                                and np.linalg.norm(np.asarray(lm['position'][:2])-np.asarray(trail[-1]))>4]
                            atomic_json(nav/'target-candidates.json',dict(source=source,candidates=candidates))
                            atomic_json(nav/'loop-complete.json',dict(distance_m=distance,closure_error_m=float(np.linalg.norm(np.asarray(trail[-1])-trail[0])),source=source))
                        elif msg.get('action')=='finish':
                            if not active or 'target_uid' not in active:raise ValueError('No committed target')
                            stop=stopped_evidence(history,json.loads((nav/'control-status.json').read_text()))
                            target=next(x for x in landmarks if x['landmark_uid']==active['target_uid'])
                            gap=float(np.linalg.norm(np.asarray(target['position'][:2])-np.asarray(trail[-1])))
                            if gap>2:raise ValueError('Not within 2m of mapped visible surface')
                            atomic_json(nav/'task-finished.json',dict(completed=True,distance_to_target_m=gap,target=target,source=source,stopped=stop))
                        else:
                            if msg.get('waypoint_protocol') == 'rgb-pixel-waypoint-v1':
                                from ticvla.warehouse.navigation.rgb_waypoint import project_waypoints
                                if source['t_obs']-src['t_obs']>5:raise ValueError('RGB guidance source expired')
                                if rgb_oracle and msg.get('task_id')!=json.loads((nav/'task.json').read_text())['task_id']:
                                    raise ValueError('RGB Oracle task changed')
                                project_waypoints([],src)  # Source integrity before any semantic mutation.
                                msg['source_t_obs']=src['t_obs']
                                msg['guidance_max_age_s']=5.0
                                if rgb_oracle:
                                    fp=nav/'perception'/src['image_id'].split(':')[-1]
                                    cached=json.loads((fp/'frame.json').read_text())
                                    instances=cached['instances']
                                    allowed={i['category'] for i in instances}
                                    if not isinstance(msg.get('landmark'),list) or any(c not in allowed for c in msg['landmark']):
                                        raise ValueError('Landmark lacks same-source detection')
                                    for i,inst in enumerate(instances):
                                        inst['mask']=np.load(fp/f'mask-{i:03}.npz')['mask'];inst.pop('texts',None)
                                    batch=dict(instances=instances,source_image_id=src['image_id'],rgb_hash=src['rgb_hash'],provenance=cached['provenance'])
                                    selected=dict(selected_categories=msg['landmark'],request_id=msg['request_id'],provider=msg['provider'],event_source='llm')
                                    mapped=consume_masks(store,src,batch,selected)
                                    ocr={**cached['ocr'],'source_image_id':src['image_id'],'rgb_hash':src['rgb_hash']}
                                    text=consume_text(store,src,batch,mapped,ocr)
                                    atomic_json(nav/f"oracle-map-update-{msg['request_id']}.json",dict(rows=mapped,text=text))
                                    landmarks=store.landmark_view()['landmarks'];atomic_json(nav/'landmarks.json',landmarks)
                                try:projection=project_waypoints(msg['waypoint'],src)
                                except ValueError as exc:
                                    atomic_json(nav/f"projection-{msg['request_id']}.json",dict(status='rejected',error=str(exc)))
                                    raise
                                atomic_json(nav/f"projection-{msg['request_id']}.json",projection)
                                local=np.asarray(projection['curve_body']).reshape(-1,2)
                                curve=np.asarray(projection['curve_odom']).reshape(-1,2)
                            elif msg.get('waypoint_protocol','legacy-body-metric') == 'legacy-body-metric':
                                local=fit_waypoints(msg['waypoint']);curve=transform_points(np.c_[local,np.zeros(len(local))],src['T_odom_body'])[:,:2]
                            else:
                                raise ValueError('Unknown waypoint coordinate protocol')
                            if msg.get('target_uid'):
                                target=next(x for x in landmarks if x['landmark_uid']==msg['target_uid'])
                                if phase=='target_selection':
                                    if not outside_camera(target['position'],src):raise ValueError('Target must be outside source RGB frustum')
                                    atomic_json(nav/'target-task.json',dict(instruction=msg['instruction'],target=target,source=src,outside_rgb_frustum=True))
                                phase='target_navigation'
                            active=msg
                            np.savez_compressed(nav/f"curve-{msg['request_id']}.npz",local=local,odom=curve,source_pose=src['T_odom_body'])
                        msg.update(accepted=True,wall=time.time(),accepted_monotonic=time.monotonic())
                        msg.setdefault('provider','codex-session')
                    except (ValueError,KeyError,StopIteration) as exc:
                        msg.update(accepted=False,error=str(exc));curve=None
                    response_log.write(json.dumps(msg)+'\n');atomic_json(nav/'last-response.json',msg)
            if not controller.is_alive():raise RuntimeError('Independent controller failed')
            if source and now-last_view>=1:
                if not (nav/'safety.npz').exists():continue
                with np.load(nav/'safety.npz') as safe:grid=safe['grid'].copy();display_pose=safe['pose'].copy()
                control=json.loads((nav/'control-status.json').read_text())
                reason=control['reason'];scan_complete=control['initial_scan_complete'];scan_yaw=control['scan_yaw_rad']
                shutil.copyfile(source['rgb'],nav/'rgb.png');bev.render(landmarks,nav/'map.png')
                im=render_map_image(grid,.1,640);draw=ImageDraw.Draw(im)
                if curve is not None:
                    path=remaining_curve(curve,display_pose)
                    pixels=[(320-y*640/12,320-x*640/12) for x,y in path]
                    if len(pixels)>1:draw.line(pixels,fill='orange',width=3)
                    if active and active.get('waypoint'):
                        src_pose=history[active['source_image_id']]['T_odom_body']
                        anchors=transform_points(np.c_[active['waypoint'],np.zeros(5)],np.linalg.inv(display_pose)@np.asarray(src_pose))[:,:2]
                        for k,(x,y) in enumerate(anchors):
                            u,v=320-y*640/12,320-x*640/12
                            draw.ellipse((u-4,v-4,u+4,v+4),fill='cyan');draw.text((u+5,v),f'W{k+1}',fill='cyan')
                im.save(nav/'plan.png')
                atomic_json(nav/'status.json',dict(wall=time.time(),phase=phase,frames=index,perception_frames=sem_count,
                    source_image_id=source['image_id'],t_obs=source['t_obs'],position=source['T_odom_body'],distance_m=distance,
                    command=reason,active_request=active['request_id'] if active else None,landmarks=len(landmarks),
                    initial_scan_complete=scan_complete,scan_yaw_rad=scan_yaw,
                    numbered_landmarks=[dict(uid=x['landmark_uid'],text=x.get('identity_text'),position=x['position']) for x in landmarks if x.get('identity_text')],
                    trained_VLA=False,control_source='Codex source-frame waypoints -> PCHIP -> Pure Pursuit'))
                last_view=now
            if (root/'capture/capture-report.json').exists():break
            time.sleep(.005)
    finally:
        controller.terminate();controller.join(10);proc.terminate();proc.join(10);store.close();response_log.close();metrics.close()
        atomic_json(root/'report.json',dict(geometry_frames=index,semantic_frames=sem_count,distance_m=distance,
            loop_complete=(nav/'loop-complete.json').exists(),target_reached=(nav/'task-finished.json').exists(),
            provider=run_config.get('provider','codex-session'),trained_VLA=False,
            api_calls_source='navigation/rgb-oracle/*/api-response.json' if rgb_oracle else 'none',landmarks=landmarks))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=8800);p.add_argument('--gui',action='store_true')
    p.add_argument('--weights',default='/data/ticvla/outputs/landmark-realtime-20261012/engines/v8s/model.engine')
    p.add_argument('--decision-script',type=Path,help='Explicit replay of reviewed Codex decisions, not live VLA inference')
    p.add_argument('--rgb-oracle',choices=['qwen3.7-plus','codex-session'])
    p.add_argument('--instruction',help='Complete task, including final STOP, for periodic RGB Oracle mode')
    p.add_argument('--runtime-only',action='store_true');a=p.parse_args();root=a.output.resolve()
    if a.rgb_oracle and (a.decision_script or not a.instruction):p.error('RGB Oracle requires --instruction and excludes --decision-script')
    if a.runtime_only:return runtime(root,a.weights)
    root.mkdir(parents=True,exist_ok=False);nav=root/'navigation';nav.mkdir();os.chmod(root,0o777);os.chmod(nav,0o777)
    (root/'index.html').write_text(HTML)
    atomic_json(nav/'task.json',dict(task_id=root.name,instruction=a.instruction or 'Complete one circuit of the warehouse aisles and build a semantic map.',hard_constraints=[]))
    atomic_json(root/'config.json',dict(scene_source='manual-static-004',landmark_variant=None,scene_person=False,
        provider=a.rgb_oracle or ('codex-reviewed-decision-replay' if a.decision_script else 'codex-session'),
        rgb_oracle=a.rgb_oracle,waypoint_protocol='rgb-pixel-waypoint-v1' if a.rgb_oracle else 'legacy-body-metric',
        trained_VLA=False,curve='PCHIP on projected pixel waypoints' if a.rgb_oracle else 'PCHIP on five source-frame waypoints',controller='Pure Pursuit',weights=a.weights))
    code=root/'source-snapshot';code.mkdir()
    files=[Path(__file__),Path(__file__).with_name('codex_demo_agent.py'),Path(__file__).with_name('codex_demo_control.py'),
           Path(__file__).with_name('rgb_oracle_service.py'),Path('ticvla/warehouse/navigation/rgb_waypoint.py'),
           Path('demo/warehouse_navigation/rgb_oracle_review.py')]
    for file in files:shutil.copy2(file,code/file.name)
    atomic_json(code/'hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    server=ThreadingHTTPServer(('0.0.0.0',a.port),functools.partial(SimpleHTTPRequestHandler,directory=str(root)))
    threading.Thread(target=server.serve_forever,daemon=True).start();children=[];logs=[]
    def launch(cmd,name):
        f=(root/name).open('w');logs.append(f);proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);children.append(proc);return proc
    try:
        worker=launch([sys.executable,'-u','-m',__name__.replace('__main__','DynaNav.warehouse_loop.codex_demo'),'--runtime-only','--output',str(root),'--weights',a.weights],'runtime.log')
        while not (nav/'ready.json').exists():
            if worker.poll() is not None:raise RuntimeError('Runtime failed; see runtime.log')
            time.sleep(.3)
        cmd=['bash','DynaNav/warehouse_loop/run_host.sh','--output',str(root/'capture'),'--capture-hz','5','--render-hz','5','--realtime',
            '--live-map',str(nav),'--navigation-root',str(nav),'--navigation-task','instruction','--max-steps','108000']
        if a.gui:cmd.append('--gui')
        sim=launch(cmd,'isaac.log')
        recording=launch([sys.executable,'demo/warehouse_navigation/record_live.py','--root',str(root),
                          '--output',str(root/'review')],'recording.log')
        if a.decision_script:
            launch([sys.executable,'-u','-m','DynaNav.warehouse_loop.codex_demo_agent','--root',str(root),
                    '--replay',str(a.decision_script.resolve())],'decision-replay.log')
        if a.rgb_oracle:
            launch([sys.executable,'-u','-m','DynaNav.warehouse_loop.rgb_oracle_service','--root',str(root),
                    '--provider',a.rgb_oracle],'rgb-oracle.log')
        while sim.poll() is None and worker.poll() is None:time.sleep(.5)
        if sim.poll() is None:
            if (nav/'task-finished.json').exists():sim.wait(timeout=60)
            else:sim.terminate()
        if worker.poll() is None:worker.wait(timeout=60)
        recording.wait(timeout=60)
        subprocess.run([sys.executable,'demo/warehouse_navigation/package_run.py','--root',str(root),
                        '--output',str(root/'review')],check=True)
        if a.decision_script and json.loads(a.decision_script.read_text()).get('version')=='codex-continuous-v2':
            subprocess.run([sys.executable,'demo/warehouse_navigation/summarize_continuity.py','--root',str(root)],check=True)
        (root/'index.html').write_text(HTML+'<p><a href="review/index.html">Completed demo: video and report</a></p>')
        print('DEMO_FINISHED',root,flush=True)
        while True:time.sleep(1)
    finally:
        for proc in children:
            if proc.poll() is None:proc.terminate()
        for f in logs:f.close()
        server.shutdown()


if __name__=='__main__':main()
