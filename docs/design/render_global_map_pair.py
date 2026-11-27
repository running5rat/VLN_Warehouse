"""Render matching global occupancy / semantic result examples from saved observations."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
import numpy as np
from numba import njit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

@njit
def carve(free,start,ends):
    h,w=free.shape
    for e in ends:
        dx=e[0]-start[0];dy=e[1]-start[1];n=max(abs(dx),abs(dy))
        for i in range(n):
            x=int(round(start[0]+dx*i/max(n,1)));y=int(round(start[1]+dy*i/max(n,1)))
            if 0<=x<w and 0<=y<h:free[y,x]=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--map',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);database=a.map/'map.sqlite';before=sha(database)
    with sqlite3.connect(database.as_uri()+'?mode=ro',uri=True) as db:
        obs=[json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')]
        landmarks=[json.loads(r[0]) for r in db.execute('SELECT payload FROM landmarks')]
    assert len({r['submap_id'] for r in obs})==1
    obs.sort(key=lambda o:o['t_obs']);trail=np.asarray([np.asarray(o['T_odom_body'])[:2,3] for o in obs]);res=.1
    lo=np.floor((trail.min(0)-[5,7])/res)*res;hi=np.ceil((trail.max(0)+[6,7])/res)*res
    w,h=np.rint((hi-lo)/res).astype(int);free=np.zeros((h,w),bool);occupied=free.copy()
    for i,o in enumerate(obs):
        with np.load(a.map/o['geometry']) as ar:p=ar['points']
        p=p[np.isfinite(p).all(1)&(p[:,2]>=-.15)&(p[:,2]<3.5)]
        ends=np.floor((p[:,:2]-lo)/res).astype(np.int64)
        body=np.asarray(o['T_odom_body']);origin=(body @ np.asarray(o['T_body_camera']))[:2,3]
        start=np.floor((origin-lo)/res).astype(np.int64)
        carve(free,start,np.unique(ends,axis=0))
        hits=ends[p[:,2]>.15];inside=((hits>=0)&(hits<[w,h])).all(1);hits=hits[inside];occupied[hits[:,1],hits[:,0]]=True
        if i%250==0:print('OBSERVATIONS',i,flush=True)
    grid=np.full((h,w),-1,np.int8);grid[free]=0;grid[occupied]=1
    np.savez_compressed(a.output/'global-occupancy.npz',grid=grid,origin_xy=lo,resolution=res,extent=np.r_[lo,hi])
    active=[r for r in landmarks if r['status']!='revoked'];colors={'sign':'#e79a27','fire extinguisher':'#c34e5b'}
    outputs={}
    for semantic,name in [(False,'global-occupancy.png'),(True,'final-semantic-map.png')]:
        fig,ax=plt.subplots(figsize=(9,6),dpi=220)
        ax.imshow(grid,origin='lower',extent=[lo[0],hi[0],lo[1],hi[1]],vmin=-1,vmax=1,cmap=ListedColormap(['#cbd2da','#fafbfd','#33495e']),interpolation='nearest')
        ax.plot(*trail.T,color='#2386b7',lw=1.5,label='Travelled path')
        ax.scatter(*trail[-1],s=35,c='#cf4855',zorder=6)
        handles=[Patch(color='#cbd2da',label='Unknown'),Patch(color='#fafbfd',ec='#9aa6b1',label='Free'),Patch(color='#33495e',label='Occupied')]
        labels=set()
        if semantic:
            for r in active:
                cat=r['category'];x,y=r['position'][:2]
                ax.scatter(x,y,s=48,c=colors.get(cat,'#8871af'),marker='P' if cat=='fire extinguisher' else 'o',edgecolors='white',linewidths=.65,zorder=5)
                if cat not in labels:handles.append(Patch(color=colors.get(cat,'#8871af'),label=cat));labels.add(cat)
        ax.legend(handles=handles,loc='upper left',fontsize=8,framealpha=.96,ncol=1)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='Odom x (m)',ylabel='Odom y (m)');ax.set_aspect('equal');ax.tick_params(labelsize=9)
        fig.subplots_adjust(left=.10,right=.99,bottom=.11,top=.98);dest=a.output/name;fig.savefig(dest);plt.close(fig);outputs[name]=sha(dest)
    assert before==sha(database)
    report=dict(source_database=str(database),source_database_sha256=before,source_database_unchanged=True,observations=len(obs),landmarks=len(active),resolution_m=res,extent_xyxy=[*lo,*hi],image_hashes=outputs,grid_sha256=sha(a.output/'global-occupancy.npz'),all_objects_tentative=all(r['status']=='tentative' for r in active),
        method='All stored depth-derived geometry, odom camera origins, ray-carved free cells and occupied endpoint union; persistent hits win. No hidden scene map, no invented free space.',
        scope='Matched full-episode result examples, identical grid, view extent and axes. Not causal inputs for frame 197; not a new SLAM localization evaluation.')
    for name in ['global-map-provenance.json','final-map-provenance.json']:(a.output/name).write_text(json.dumps(report,indent=2)+'\n')
    print('GLOBAL_MAP_PAIR_COMPLETE',flush=True)

if __name__=='__main__':main()
