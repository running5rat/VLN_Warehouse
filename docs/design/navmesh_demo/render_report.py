"""Render actual Recast polygons and validate saved Detour path queries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.path import Path as PlotPath

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
p.add_argument('--publish', required=True)
a = p.parse_args()
root, pub = Path(a.output), Path(a.publish)
pub.mkdir(parents=True, exist_ok=True)
load = lambda f: json.loads(Path(f).read_text())
sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
mesh = load(root/'radius-035/navmesh.json')
stats = load(root/'radius-035/build-report.json')
queries = load(root/'radius-035/queries.json')
source = load(root/'scene-source.json')
verts = np.array(mesh['vertices'])
polys = [verts[x['vertices']] for x in mesh['polygons']]
before = np.genfromtxt(root/'radius-035/walkable-before.csv', delimiter=',', names=True)
after = np.genfromtxt(root/'radius-035/walkable-after.csv', delimiter=',', names=True)
with np.load(root/'scene_world.npz') as data:
    rawv, rawt, bounds = data['vertices'], data['triangles'], data['bounds']
tri = rawv[rawt]
tri = tri[tri[:,:,2].max(1)>.15]
# Plot using the demo's initial heading; exports stay in Isaac world coordinates.
def demo(x):
    x = np.asarray(x)
    return np.stack((x[...,1]-6, -(x[...,0]+13)), axis=-1)

obstacles = demo(tri)
poly_xy = [demo(x) for x in polys]
def axes(title, size=(10,5)):
    fig, ax = plt.subplots(figsize=size, layout='constrained')
    ax.set(title=title, xlabel='Demo forward (m)', ylabel='Demo left (m)',
           xlim=(-4,25), ylim=(-10,5), aspect='equal')
    ax.set_facecolor('#f1f5f9'); ax.grid(alpha=.15,zorder=0)
    return fig, ax
def geometry(ax):
    ax.add_collection(PolyCollection(obstacles,facecolors='#65758a',edgecolors='none',alpha=.45,zorder=1,rasterized=True))
def polygons(ax, alpha=.8, colored=True):
    colors=plt.get_cmap('tab20')(np.arange(len(polys))%20) if colored else '#42c9b1'
    ax.add_collection(PolyCollection(poly_xy,facecolors=colors,edgecolors='#125867',linewidths=.8,alpha=alpha,zorder=2))
def save(fig, name):
    fig.savefig(pub/name,dpi=180,facecolor='white');plt.close(fig)

fig,ax=axes('Warehouse static scene geometry | real USD triangle projection')
geometry(ax)
save(fig,'01-scene-geometry.png')
fig,ax=axes(f'Recast NavMesh | {len(polys)} walkable polygons, radius 0.35 m')
geometry(ax);polygons(ax)
for i,poly in enumerate(poly_xy):
    center=poly.mean(0)
    ax.text(*center,str(i),fontsize=7,ha='center',va='center',color='#17354b',zorder=3)
save(fig,'02-navmesh-polygons.png')

fig,axs=plt.subplots(1,2,figsize=(14,4.8),layout='constrained')
for ax,name,title in zip(axs,['radius-000','radius-035'],['Point agent: radius 0 m','Robot: radius 0.35 m (4 raster cells)']):
    ax.set(title=title,xlabel='Demo forward (m)',ylabel='Demo left (m)',xlim=(-4,25),ylim=(-10,5),aspect='equal')
    ax.set_facecolor('#f1f5f9');geometry(ax)
    m=load(root/name/'navmesh.json');v=np.array(m['vertices'])
    ax.add_collection(PolyCollection([demo(v[x['vertices']]) for x in m['polygons']],facecolors='#42c9b1',edgecolors='#147b74',linewidths=.6,alpha=.8,zorder=2))
save(fig,'03-radius-comparison.png')

q=next(x for x in queries if x['name']=='other_aisle_goal')
fig,ax=axes('Detour query | polygon corridor and funnel path to the next aisle')
geometry(ax);polygons(ax,alpha=.18,colored=False)
ax.add_collection(PolyCollection([poly_xy[i] for i in q['corridor_polygon_ids']],facecolors='#ffc964',edgecolors='#ad7621',linewidths=1,alpha=.7,zorder=3))
if q['straight_path']:
    path=demo(q['straight_path']);ax.plot(*path.T,'o-',color='#0869bc',lw=2.4,ms=4,zorder=5,label='Detour funnel path')
start,goal=demo(q['requested_start']),demo(q['requested_end'])
ax.plot([start[0],goal[0]],[start[1],goal[1]],'--',color='#d44c57',lw=1.4,zorder=4,label='Direct line (not the planned path)')
for xy,label,col in [(start,'Start','#076dab'),(goal,'Goal','#d06517')]:
    ax.scatter(*xy,c=col,s=55,zorder=6);ax.annotate(label,xy,xytext=(7,8),textcoords='offset points',weight='bold',color=col,zorder=7)
ax.legend(loc='upper left');save(fig,'04-path-query.png')

fig,ax=axes('Ordered circuit queries | four legs, not a vehicle execution')
geometry(ax);polygons(ax,alpha=.3,colored=False)
for j,qloop in enumerate(queries[:4]):
    if qloop['straight_path']:
        xy=demo(qloop['straight_path']);ax.plot(*xy.T,'o-',lw=2,color=plt.get_cmap('tab10')(j),label=qloop['name'],zorder=5)
ax.legend(loc='upper left');save(fig,'05-loop-queries.png')

# Independent saved-output checks. The footprint check uses pre-erosion raster
# cells, not the polygon corridor; it is not a PhysX execution test.
cs=stats['cell_size_m'];lo=bounds[0,:2]
gridshape=np.ceil((bounds[1,:2]-lo)/cs).astype(int)
free=np.zeros(gridshape,dtype=bool)
ij=np.floor((np.c_[before['x'],before['y']]-lo)/cs).astype(int)
free[ij[:,0],ij[:,1]]=True
offsets=np.arange(-.35,.35001,.05)
dx,dy=np.meshgrid(offsets,offsets)
disk=np.c_[dx.ravel(),dy.ravel()];disk=disk[(disk**2).sum(1)<=.35**2+1e-9]
path_checks=[]
for q in queries:
    if not q['complete']:
        path_checks.append({'name':q['name'],'complete':False,'expected_rejection':q['name']=='blocked_rack'})
        continue
    v=np.array(q['straight_path']);lengths=np.linalg.norm(np.diff(v[:,:2],axis=0),axis=1)
    sample=np.concatenate([s+(t-s)*np.linspace(0,1,max(2,int(np.ceil(d/.05))+1))[:,None] for s,t,d in zip(v[:-1,:2],v[1:,:2],lengths)])
    inside=np.zeros(len(sample),bool)
    for poly in polys:
        shape=PlotPath(np.r_[poly[:,:2],poly[0:1,:2]])
        inside |= shape.contains_points(sample,radius=1e-5)|shape.contains_points(sample,radius=-1e-5)
    swept=(sample[:,None,:]+disk).reshape(-1,2)
    ij=np.floor((swept-lo)/cs).astype(int)
    valid=(ij>=0).all(1)&(ij<gridshape).all(1)
    allowed=np.zeros(len(ij),bool);allowed[valid]=free[ij[valid,0],ij[valid,1]]
    path_checks.append({'name':q['name'],'complete':True,'path_length_m':float(lengths.sum()),
                        'samples':len(sample),'inside_navmesh':bool(inside.all()),
                        'circular_footprint_raster_pass':bool(allowed.all()),
                        'unverified_or_blocked_footprint_samples':int((~allowed).sum()),
                        'max_endpoint_snap_xy_m':max(float(np.linalg.norm(np.array(q['snapped_start'])[:2]-np.array(q['requested_start'])[:2])),float(np.linalg.norm(np.array(q['snapped_end'])[:2]-np.array(q['requested_end'])[:2])))})
fixture=load(root/'fixture/queries.json') if (root/'fixture/queries.json').exists() else []
checks={'disk_reload':stats['disk_reload_passed'],
        'five_requested_routes_complete':all(q['complete'] for q in queries[:5]),
        'rack_target_rejected':not queries[5]['complete'],
        'paths_inside_polygon_mesh':all(x.get('inside_navmesh',True) for x in path_checks),
        'circular_footprint_raster':all(x.get('circular_footprint_raster_pass',True) for x in path_checks),
        'fixture_detour_and_rejection':bool(fixture and fixture[0]['complete'] and not fixture[1]['complete']),
        'radius_reduces_area':len(after)<len(before)}
report={'source':source['source'],'source_geometry':'Isaac 5.0 static USD triangles in the demo aisle crop',
        'build':stats,'walkable_before_m2':len(before)*cs*cs,'walkable_after_m2':len(after)*cs*cs,
        'checks':checks,'paths':path_checks,
        'limits':['Offline scene mesh, not the robot-observed SLAM map.',
                  'Circular footprint raster check only; rectangular footprint, yaw and dynamics not evaluated.',
                  'No controller or new simulated vehicle motion; path-query success is not navigation success.',
                  'GT scene geometry is an offline Oracle-side asset and is not supplied as a VLM input.'],
        'hashes':{str(f.relative_to(root)):sha(f) for f in [root/'scene.bin',root/'scene_world.npz',root/'scene-source.json',root/'radius-035/navmesh.bin',root/'radius-035/navmesh.json',root/'radius-035/queries.json']}}
(root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
for file in ['report.json','scene-source.json']:
    shutil.copyfile(root/file,pub/file)
for file in ['navmesh.json','navmesh.bin','navmesh.obj','queries.json','build-report.json']:
    shutil.copyfile(root/'radius-035'/file,pub/file)
panels=[('01-scene-geometry.png','静态场景几何'),('02-navmesh-polygons.png','NavMesh 可通行多边形'),('03-radius-comparison.png','机器人半径收缩对照'),('04-path-query.png','跨通道路径查询'),('05-loop-queries.png','四段有序绕环查询')]
html='''<!doctype html><meta charset="utf-8"><title>Warehouse NavMesh</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px system-ui;color:#17354b;background:#f5f8fc}img{width:100%;background:white}pre{white-space:pre-wrap;background:white;padding:16px}a{color:#0869bc}</style><h1>Warehouse · Recast / Detour 实测</h1><p>Isaac Sim 5.0 静态场景几何 → 高度场 → 半径收缩 → 多边形 NavMesh → 路径查询。未执行车辆控制。</p>'''
html+=f'<p>{stats["input_triangles"]:,} 个输入三角形，{stats["navmesh_polygons"]} 个导航多边形；建网、保存重载与查询合计 {stats["elapsed_s"]:.3f} 秒（不含 Isaac 导出、编译和绘图）。</p>'
html+='<p><a href="navmesh.obj">OBJ 网格</a> · <a href="navmesh.json">顶点/多边形/邻接 JSON</a> · <a href="navmesh.bin">Detour 二进制</a> · <a href="report.json">完整报告</a></p>'
for name,title in panels:html+=f'<h2>{title}</h2><img src="{name}">'
html+='<h2>验证结果</h2><pre>'+json.dumps(checks,indent=2)+'</pre><p>静态三角形几何与 PhysX 的最终碰撞形状可能不同；实际控制还需要 footprint、转向和执行核验。</p>'
(pub/'index.html').write_text(html)
print(json.dumps(report,indent=2))
