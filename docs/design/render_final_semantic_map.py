"""Render an existing final map read-only, without changing any predictions."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--map',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    database=args.map/'map.sqlite';before=sha(database)
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        obs=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]
        records=[json.loads(r[0]) for r in db.execute('SELECT payload FROM landmarks')]
    epochs={r['submap_id'] for r in obs}
    if len(epochs)!=1:raise ValueError('Do not overlay independent odom epochs')
    obs.sort(key=lambda r:r['t_obs']);cells=set();trail=[]
    for r in obs:
        with np.load(args.map/r['geometry']) as a:pts=a['points']
        pts=pts[(pts[:,2]>.15)&(pts[:,2]<3.5)]
        cells.update(map(tuple,np.floor(pts[:,:2]/.1).astype(int)))
        trail.append(np.asarray(r['T_odom_body'])[:2,3])
    xy=np.asarray(list(cells))*.1;trail=np.asarray(trail)
    fig,ax=plt.subplots(figsize=(9,6),dpi=220)
    ax.set_facecolor('#f4f7fa');ax.scatter(*xy.T,s=.9,color='#8596a6',alpha=.75,rasterized=True)
    ax.plot(*trail.T,color='#277cc4',lw=1.8,label='Travelled path')
    colors={'sign':'#e69a2c','fire extinguisher':'#bc5861','storage rack':'#7d72aa'}
    markers={'sign':'o','fire extinguisher':'P','storage rack':'s'}
    labels=set();active=[r for r in records if r['status']!='revoked']
    for r in active:
        category=r['category'];x,y=r['position'][:2]
        label=category if category not in labels else None;labels.add(category)
        ax.scatter(x,y,s=40,c=colors.get(category,'#7d72aa'),marker=markers.get(category,'s'),edgecolors='white',linewidths=.5,label=label,zorder=4)
        if r.get('identity_status')=='identity_confirmed' and r.get('identity_text'):
            ax.annotate(r['identity_text'],(x,y),xytext=(3,8),textcoords='offset points',fontsize=9,color='#16674d',fontweight='bold',zorder=5)
    ax.scatter(*trail[-1],s=55,color='#d34848',zorder=6)
    ax.set(xlim=(trail[:,0].min()-5,trail[:,0].max()+6),ylim=(trail[:,1].min()-7,trail[:,1].max()+7),xlabel='Odom x (m)',ylabel='Odom y (m)')
    ax.set_aspect('equal');ax.grid(alpha=.14);ax.legend(loc='upper left',fontsize=8,framealpha=.95)
    fig.tight_layout();dest=args.output/'final-semantic-map.png';fig.savefig(dest,bbox_inches='tight');plt.close(fig)
    report=dict(source_database=str(database),source_database_sha256=before,source_database_unchanged=before==sha(database),
        observations=len(obs),landmarks=len(active),image_sha256=sha(dest),all_objects_tentative=all(r['status']=='tentative' for r in active),
        scope='Actual final saved observations, cropped viewport around the travelled aisles. Not a causal input for frame 197. Blank background is unobserved; geometry is observed surfaces, not certified free space.')
    (args.output/'final-map-provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
