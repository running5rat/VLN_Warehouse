"""Verify videos, recorded timeline, packaged map recovery and artifact hashes."""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--delivery',type=Path,required=True)
    args=p.parse_args(); root=args.delivery.resolve()
    summary=json.loads((root/'summary.json').read_text())
    recording=json.loads((root/'recording.json').read_text())
    videos={}
    for name,expected in [('mapping_live.mp4',recording['frames']),('isaac_camera.mp4',summary['capture_frames'])]:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration',
            '-of','json',str(root/name)],text=True))['streams'][0]
        assert int(info['nb_read_frames'])==expected,(name,info)
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(root/name),'-f','null','-'],check=True)
        videos[name]=info
    timeline=[json.loads(x) for x in (root/'video-timeline.jsonl').read_text().splitlines()]
    assert len(timeline)==recording['frames']
    assert all(a['wall_sample']<=b['wall_sample'] for a,b in zip(timeline,timeline[1:]))
    manifest=json.loads((root/'manifest.json').read_text())
    for name,entry in manifest.items():
        h=hashlib.sha256()
        with (root/name).open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        assert h.hexdigest()==entry['sha256'],name
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ('src','href','poster') and value and not value.startswith(('http:','https:','#')):
                    assert (self.base/value).exists(),str(self.base/value)
    for path in root.rglob('*.html'):
        checker=Links();checker.base=path.parent;checker.feed(path.read_text())
    from ticvla.warehouse.mapping import MapStore
    with tempfile.TemporaryDirectory(prefix='semantic-demo-verify-') as tmp:
        with tarfile.open(root/'map.tar.gz','r:gz') as tar:
            tar.extractall(tmp, filter='data')
        store=MapStore(Path(tmp)/'map')
        try:
            recovered=store.summary()
            observations=store.records('observations')
            assert len(observations)==summary['capture_frames']
            assert len(store.records('landmarks'))==summary['raw_landmarks']
            points=0
            for obs in observations:
                p=store.points(obs); assert np.isfinite(p).all(); points+=len(p)
            version=store.version
        finally: store.close()
        store=MapStore(Path(tmp)/'map')
        try: assert store.version==version and store.summary()==recovered
        finally:store.close()
    result=dict(passed=True,videos=videos,manifest_files_verified=len(manifest),
        html_links_exist=True,timeline_frames=len(timeline),map_restart_identical=True,
        map_observations=len(observations),map_geometry_points=points,map_version=version,
        scope='Artifact integrity and recovery; not model accuracy or real-time acceptance.')
    (root/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
