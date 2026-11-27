"""Render a measured depth frame for the architecture figure; no synthetic depths."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--depth',type=Path,required=True)
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads(a.input.read_text())['source']
assert sha(a.depth)==source['depth_hash'], 'Depth does not match source frame'
depth=np.load(a.depth)
valid=np.isfinite(depth)&(depth>0)
a.output.mkdir(parents=True,exist_ok=True)
image=a.output/'depth.png'
cmap=plt.get_cmap('magma').copy();cmap.set_bad('#20252b')
plt.imsave(image,np.ma.masked_where(~valid,depth),cmap=cmap,vmin=0,vmax=12)
report=dict(source_image_id=source['image_id'],source_depth=str(a.depth),source_depth_sha256=sha(a.depth),
            image_sha256=sha(image),shape=list(depth.shape),valid_pixels=int(valid.sum()),
            display=dict(units='metres',colormap='magma',range=[0,12],invalid='dark gray',clipping='display only'))
(a.output/'depth-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
