"""Rebuild a causal occupancy preview using the existing local map implementation."""
import argparse,json,hashlib
from pathlib import Path
from dataclasses import asdict
import numpy as np
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.geometry import LocalMapBuilder,render_map_image
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--capture',type=Path,required=True)
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();source=json.loads(a.input.read_text())['source'];cfg=WarehouseConfig()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
frames=[json.loads(s) for s in (a.capture/'frames.jsonl').read_text().splitlines()]
frames=[f for f in frames if 0<=source['t_obs']-f['t_obs']<=cfg.map_max_age and f['odom_epoch']==source['odom_epoch']]
assert frames and frames[-1]['t_obs']==source['t_obs']
assert sha(a.capture/frames[-1]['depth'])==source['depth_hash']
builder=LocalMapBuilder(cfg)
for f in frames:builder.add(depth=np.load(a.capture/f['depth']),**{k:f[k] for k in ('K','T_body_camera','T_odom_body','t_obs','odom_epoch')})
grid,age=builder.render(source['T_odom_body'],source['t_obs'],source['odom_epoch'])
a.output.mkdir(parents=True,exist_ok=True)
np.savez_compressed(a.output/'occupancy.npz',grid=grid,age=age)
render_map_image(grid,cfg.map_resolution,640).save(a.output/'occupancy.png')
report=dict(source_image_id=source['image_id'],frames=len(frames),first_t=frames[0]['t_obs'],last_t=frames[-1]['t_obs'],
 source_manifest=str(a.capture/'frames.jsonl'),source_manifest_sha256=sha(a.capture/'frames.jsonl'),
 geometry_implementation_sha256=sha(Path('ticvla/warehouse/geometry.py')),config={k:asdict(cfg)[k] for k in ('map_size_m','map_resolution','map_max_age','depth_stride','obstacle_min_height','obstacle_max_height','footprint_radius')},
 source_depth_hashes={f['depth']:sha(a.capture/f['depth']) for f in frames},image_sha256=sha(a.output/'occupancy.png'),
 scope='Causal depth/ideal-odom local occupancy replay; not a new SLAM localization evaluation.')
(a.output/'occupancy-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print('Occupancy rendered from',len(frames),'causal frames')
