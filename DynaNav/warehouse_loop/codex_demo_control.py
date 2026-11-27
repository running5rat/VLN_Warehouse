"""Independent latest RGB-D/controller process; no OCR, SQLite or image rendering."""
import json
import math
from pathlib import Path
import time
import numpy as np
from .live_metrics import atomic_json,complete_lines
from .codex_demo import remaining_curve
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.geometry import LocalMapBuilder
from ticvla.warehouse.navigation.geometry import render_local
from ticvla.warehouse.navigation.core import pursuit


def response_curve(nav, msg):
    """Every new response supersedes motion, including rejected responses."""
    p=Path(nav)/f"curve-{msg['request_id']}.npz"
    if not msg.get('accepted') or msg.get('action') or not p.exists():
        return None
    with np.load(p) as data:
        return data['odom'].copy()


def run(root):
    root=Path(root);nav=root/'navigation';raw=root/'capture/raw'
    cfg=WarehouseConfig(depth_stride=32,map_resolution=.1,map_size_m=12,map_max_age=1e9,footprint_radius=.35)
    builder=LocalMapBuilder(cfg);keys=[];key_pose=None;source=None;stream=None;count=0;grid=None
    curve=None;request=None;scan=0.;last_yaw=None;done=False;last_status=0;initial=True
    old=nav/'status.json'
    if old.exists():
        value=json.loads(old.read_text());done=value.get('initial_scan_complete',False);scan=value.get('scan_yaw_rad',0.)
    metrics=(nav/'control-metrics.jsonl').open('a',buffering=1)
    try:
        while not (root/'cancel.json').exists() and not (nav/'task-finished.json').exists():
            start=time.monotonic()
            if stream is None:
                if not (raw/'frames.jsonl').exists():time.sleep(.05);continue
                meta=json.loads((raw/'episode.json').read_text());stream=(raw/'frames.jsonl').open()
            fresh=list(complete_lines(stream))
            if fresh:
                # On resume seed historical views; later only latest source is used for control.
                selected=[]
                for f in fresh if initial else fresh[-1:]:
                    pose=np.asarray(f['T_odom_body']);delta=None if key_pose is None else np.linalg.inv(key_pose)@pose
                    if delta is None or np.linalg.norm(delta[:2,3])>.6 or abs(math.atan2(delta[1,0],delta[0,0]))>.3 or f is fresh[-1]:
                        selected.append(f);key_pose=pose
                for f in selected:
                    pose=np.asarray(f['T_odom_body'])
                    builder.add(depth=np.load(raw/f['depth']),**{k:f[k] for k in ('K','T_body_camera','T_odom_body','t_obs','odom_epoch')})
                    current=builder.frames[-1]
                    delta=None if not keys else np.linalg.inv(keys[-1][0])@pose
                    if delta is None or np.linalg.norm(delta[:2,3])>.6 or abs(math.atan2(delta[1,0],delta[0,0]))>.3:
                        keys.append((pose.copy(),current));keys=keys[-24:]
                    builder.frames=[v[1] for v in keys]+([] if keys and keys[-1][1] is current else [current])
                initial=False;count+=len(fresh);f=fresh[-1];pose=np.asarray(f['T_odom_body']);source=f
                yaw=math.atan2(pose[1,0],pose[0,0])
                if last_yaw is not None and not done:scan+=math.atan2(math.sin(yaw-last_yaw),math.cos(yaw-last_yaw))
                last_yaw=yaw
                if scan>=2*math.pi-.035:done=True
                grid,_=render_local(builder,pose,f['t_obs'],f['odom_epoch'])
                tmp=nav/'safety.control.tmp.npz';np.savez_compressed(tmp,grid=grid,pose=pose,resolution=.1,
                    odom_epoch=f['odom_epoch'],source_monotonic=f['source_monotonic']);tmp.replace(nav/'safety.npz')
            response=nav/'last-response.json'
            if response.exists():
                msg=json.loads(response.read_text())
                if msg['request_id']!=request:
                    request=msg['request_id'];curve=response_curve(nav,msg)
            if source:
                if (curve is not None and msg.get('waypoint_protocol')=='rgb-pixel-waypoint-v1'
                        and source['t_obs']-msg['source_t_obs']>msg['guidance_max_age_s']):
                    curve=None
                path=remaining_curve(curve,pose) if curve is not None else np.empty((0,2))
                if not done:v,w,reason=0.,.45,'initial_observation_scan'
                elif len(path)>1 and np.linalg.norm(path[-1])>.16:
                    (v,w),reason=pursuit(path,grid,.1,.35,.3)
                    # A safe on-the-spot look toward the route can expose a turn's unknown space.
                    if reason=='shield_occupied_or_unknown':
                        k=min(int(np.searchsorted(np.maximum.accumulate(np.linalg.norm(path,axis=1)),.6)),len(path)-1)
                        angle=math.atan2(path[k,1],path[k,0])
                        if abs(angle)>.08:v,w,reason=0.,float(np.clip(2*angle,-.6,.6)),'observe_turn_before_translation'
                else:v,w,reason=0.,0.,'waypoint_endpoint_reached' if curve is not None else 'waiting_codex_waypoints'
                now=time.monotonic()
                atomic_json(nav/'command.json',dict(v=v,w=w,reason=reason,backend='codex-waypoint-demo',
                    source_image_id=f"{meta['episode_id']}:{count-1:08d}",odom_epoch=source['odom_epoch'],created_monotonic=now,valid_until=now+.3))
                if now-last_status>.2:
                    status=dict(initial_scan_complete=done,scan_yaw_rad=scan,reason=reason,source_index=count-1,
                        source_age_s=now-source['source_monotonic'],position=pose.tolist(),request_id=request,wall=time.time())
                    atomic_json(nav/'control-status.json',status);metrics.write(json.dumps(status)+'\n');last_status=now
            time.sleep(max(0,.033-(time.monotonic()-start)))
    finally:
        metrics.close()
