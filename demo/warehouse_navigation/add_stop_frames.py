"""Explicit completion coverage addendum, not selected using Oracle success."""
from pathlib import Path
import json
import math
import shutil
import numpy as np
from PIL import Image,ImageDraw
from ticvla.warehouse.mapping import MapStore
from ticvla.warehouse.dataset_tools.core import read_json,write_json
from ticvla.warehouse.schema import file_sha256
from DynaNav.warehouse_loop.live_mapping import LiveBEV


def run(root):
    root=Path(root);exp=read_json(root/'experiment.json');source=Path(exp['source'])
    if exp.get('completion_frame_addendum'):return
    shutil.copy2(root/'experiment.json',root/'experiment.initial-35.json')
    frames=[json.loads(x) for x in (source/'capture/raw/frames.jsonl').read_text().splitlines()]
    store=MapStore(root/'causal-map')
    for n,task,event,base in [(950,'mapping','loop-complete',951),(1314,'target','task-finished',None)]:
        s=read_json(source/'navigation'/f'{event}.json')['source'];assert s['image_id'].endswith(f'{n:08d}')
        dest=root/'inputs'/f'{n:08d}';dest.mkdir()
        for field in ('rgb','depth'):assert file_sha256(s[field])==s[field+'_hash']
        landmarks=read_json(root/'inputs'/f'{base:08d}'/'audit.json')['map_landmarks'] if base else store.landmark_view()['landmarks']
        assert all(l['last_seen']<s['t_obs'] for l in landmarks)
        bev=LiveBEV();cutoff=None
        for obs in store.records('observations'):
            if obs['t_obs']>=s['t_obs']:continue
            with np.load(store.root/obs['geometry']) as z:points=z['points']
            bev.add(points,np.array(obs['T_odom_body']));cutoff=max(cutoff or 0,obs['t_obs'])
        bev.trail=[np.array(f['T_odom_body'])[:2,3].tolist() for f in frames[:n+1]]
        bev.render(landmarks,dest/'map.png');im=Image.open(dest/'map.png');pose=np.array(s['T_odom_body'])
        ImageDraw.Draw(im).text((10,8),f'CURRENT HEADING: odom yaw {math.degrees(math.atan2(pose[1,0],pose[0,0])):.1f} deg',fill='red');im.save(dest/'map.png')
        for name in ('rgb.png','detections.png'):shutil.copy2(s['rgb'],dest/name)
        write_json(dest/'source.json',s);write_json(dest/'audit.json',dict(source=s,perception=None,map_version=store.version,
            map_source_cutoff=cutoff,map_landmarks=landmarks,categories=[],detection_status='not_processed_at_terminal_source',
            completion_evidence=str(source/'navigation'/f'{event}.json'),
            images={k:file_sha256(dest/k) for k in ('rgb.png','detections.png','map.png')}))
        exp['selected'].append(dict(frame=n,task=task,t_obs=s['t_obs'],completion_confirmed=True,perception=None,
            selection_reason='Recorded task completion, included regardless of model quality'))
    store.close();exp['selected'].sort(key=lambda r:r['frame']);exp['completion_frame_addendum']=True
    exp['task_stop_coverage']='Recorded task completion sources 950 and 1314 added before requesting their annotations; no same-frame detections, [] categories.'
    write_json(root/'experiment.json',exp)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
