"""Verify persisted instance/text event replay adds no map evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
from ..mapping import MapStore
from ..mapping.semantic_frame import write_json
from .realtime import consume_masks, consume_text


def check(map_root, folder):
    d=json.loads((folder/'frame.json').read_text())
    batch=dict(source_image_id=d['source']['image_id'],rgb_hash=d['source']['rgb_hash'],
               provenance=d['provenance'],candidate_request=d['candidate_request'],
               instances=[{**v,'mask':np.load(folder/f'mask-{i:03}.npz')['mask']} for i,v in enumerate(d['instances'])])
    store=MapStore(map_root)
    try:
        before=store.summary()
        rows=consume_masks(store,d['source'],batch,d['selection'])
        text=consume_text(store,d['source'],batch,rows,d['ocr']) if d.get('ocr') else None
        after=store.summary()
        if before!=after:raise AssertionError('Replay changed persisted evidence')
        return dict(map=str(map_root),image_id=d['source']['image_id'],unchanged=True,
                    accepted_event_replays=sum(x['result'].get('duplicate',False) for x in rows),
                    text_event_replays=sum(x.get('duplicate',False) for x in (text or {}).get('updates',[])),map_version=store.version)
    finally:store.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();r=a.root
    results=[check(r/'C/maps/color',r/'C/color-00000013')]
    for name,frame in [('original',300),('combined',0)]:
        results.append(check(r/f'live-{name}/mapping/map',r/f'live-{name}/mapping/frames/live-{name}_{frame:08d}'))
    write_json(r/'recovery-report.json',dict(passed=True,checks=results))
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
