"""Chapter 3.3 figures. Read-only demo extraction; no model or Oracle execution.

Run with the analysis Python to prepare assets, then use build_v07.py normally.
The document build needs only the archived PNG/JSON assets, not the /data source.
"""
from pathlib import Path
import json
import hashlib

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'docs/design/figures/dataengine-assets'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare():
    import sqlite3
    import shutil
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Circle, Polygon
    from PIL import Image
    ASSETS.mkdir(exist_ok=True)
    dbpath=Path('/data/ticvla/outputs/navigation/codex-continuous-20261112-v2/navigation/map/map.sqlite')
    before=sha(dbpath)
    with sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True) as db:
        rows=sorted([json.loads(r[0]) for r in db.execute('SELECT payload FROM observations')],key=lambda r:r['t_obs'])
    obs=next(r for r in rows if r['image_id'].endswith(':00000440'))
    before_turn=obs
    trajectory=np.array([np.array(r['T_odom_body'])[:3,3] for r in rows])
    future=np.array([np.array(r['T_odom_body'])[:3,3] for r in rows if r['t_obs']>=obs['t_obs'] and r['task_id']=='mapping'])
    distance=np.r_[0,np.cumsum(np.linalg.norm(np.diff(future[:,:2],axis=0),axis=1))]
    keep=np.r_[True,np.diff(distance)>1e-5]; future=future[keep]; distance=distance[keep]
    sample=np.arange(0,distance[-1],.05)
    xyz=np.column_stack([np.interp(sample,distance,future[:,j]) for j in range(3)])
    T=np.array(obs['T_odom_body'])@np.array(obs['T_body_camera']);K=np.array(obs['K'])
    cam=(np.linalg.inv(T)@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
    projected=cam@K.T
    uv=projected[:,:2]/projected[:,2:]
    depth=np.load(obs['depth']);h,w=depth.shape
    inframe=(cam[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    measured=np.full(len(sample),np.nan)
    pix=np.rint(uv[inframe]).astype(int); pix[:,0]=pix[:,0].clip(0,w-1);pix[:,1]=pix[:,1].clip(0,h-1)
    measured[inframe]=depth[pix[:,1],pix[:,0]]
    supported=inframe & np.isfinite(measured)&(measured>0)&(np.abs(cam[:,2]-measured)<.15)
    start=int(np.flatnonzero(supported)[0]);end=start
    # Existing 32-point / .25-m horizon, no training setting changed.
    limit=31*.25
    while end+1<len(sample) and supported[end+1] and sample[end+1]<=limit:end+=1
    valid=np.arange(start,end+1)
    assert len(valid)>2
    anchors=np.array([valid[len(valid)//2],valid[-1]])
    base=ROOT/'docs/design/figures/warehouse-assets'
    with np.load(base/'global-occupancy.npz') as ar:
        print('map keys',ar.files)
        grid=ar['grid'];extent=ar['extent'][[0,2,1,3]]
    cmap=ListedColormap(['#cbd2da','#f8fafc','#344c60'])
    def axes():
        fig,ax=plt.subplots(figsize=(6.4,4.1),layout='constrained')
        ax.imshow(grid,origin='lower',extent=extent,cmap=cmap,vmin=-1,vmax=1)
        ax.set(xlabel='Odom x (m)',ylabel='Odom y (m)',xlim=(-2,23),ylim=(-8,4))
        ax.tick_params(labelsize=9)
        return fig,ax
    def save(fig,name):
        fig.savefig(ASSETS/name,dpi=200,facecolor='white');plt.close(fig)
    fig,ax=axes()
    turn_future=np.array([np.array(r['T_odom_body'])[:3,3] for r in rows if r['t_obs']>=before_turn['t_obs'] and r['task_id']=='mapping'])
    ax.plot(turn_future[:,0],turn_future[:,1],color='#8c9db6',lw=2,label='Future demonstration')
    bp=np.array(before_turn['T_odom_body'])[:2,3]
    ax.scatter(*bp,color='#7854a7',s=45,zorder=9,label='Before turn')
    ax.set(xlim=(11,23),ylim=(-8,0))
    ax.plot(xyz[valid,0],xyz[valid,1],color='#00a9ac',lw=4,label='Visible prefix')
    a=xyz[0,:2];ray=7.75
    heading=np.arctan2(T[1,2],T[0,2]);fov=np.arctan(w/(2*K[0,0]))
    wedge=np.array([a,a+ray*np.array([np.cos(heading-fov),np.sin(heading-fov)]),a+ray*np.array([np.cos(heading+fov),np.sin(heading+fov)])])
    ax.add_patch(Polygon(wedge,facecolor='#2da7d0',alpha=.12,edgecolor='#2da7d0',ls='--'))
    ax.scatter(*a,color='#da4e55',s=40,zorder=8,label='Current pose')
    ax.scatter(xyz[anchors,0],xyz[anchors,1],c='#ffb336',s=42,zorder=9)
    ax.annotate('After turn: visible prefix',xyz[end,:2],xytext=(11.5,-1),fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2},arrowprops={'arrowstyle':'->','color':'#dc8b18'})
    ax.legend(loc='lower right',fontsize=8)
    save(fig,'demonstration-bev.png')
    # Overlay the actual future projection and its retained reference on source RGB.
    from matplotlib.font_manager import FontProperties
    cn=FontProperties(fname='/root/.local/share/fonts/NotoSansCJK-Regular.ttc',size=13)
    fig=plt.figure(figsize=(w/150,h/150),dpi=150)
    ax=fig.add_axes([0,0,1,1]);ax.imshow(Image.open(obs['rgb']))
    future_uv=uv.copy();future_uv[~inframe]=np.nan
    ax.plot(future_uv[:,0],future_uv[:,1],color='#ee82ee',lw=7,ls=(0,(5,3)),
            label='当前帧的未来轨迹投影',zorder=3)
    ax.plot(uv[valid,0],uv[valid,1],color='#11d5dd',lw=3,
            label='截取后的参考线段',zorder=4)
    ax.scatter(uv[anchors,0],uv[anchors,1],c='#ffbd42',s=60,edgecolors='white',zorder=5)
    for j,i in enumerate(anchors):
        ax.annotate(f'W{j+1}',uv[i],xytext=(-35 if uv[i,0]>.85*w else 9,-3),textcoords='offset points',
                    color='#ffbd42',weight='bold',fontsize=11,zorder=6)
    box={'facecolor':'#101c2c','edgecolor':'none','alpha':.8,'pad':4}
    ax.annotate('可见边界截断',uv[end],xytext=(w*.65,h*.79),
                fontproperties=cn,color='#11d5dd',bbox=box,
                arrowprops={'arrowstyle':'->','color':'#11d5dd'},zorder=7)
    legend=ax.legend(loc='upper left',prop=cn,facecolor='#101c2c',edgecolor='none',
                     framealpha=.85,labelcolor='white',borderpad=.6)
    legend.set_zorder(8)
    ax.set(xlim=(0,w),ylim=(h,0));ax.axis('off')
    fig.savefig(ASSETS/'demonstration-rgb.png',dpi=150);plt.close(fig)
    # The right panel uses the same source body frame and exact retained indices.
    Tb=np.asarray(obs['T_odom_body'])
    body=(np.linalg.inv(Tb)@np.c_[xyz,np.ones(len(xyz))].T).T[:,:3]
    show=sample<=10.0
    gy,gx=np.mgrid[0:h:12,0:w:12]
    gz=depth[gy,gx]
    cp=np.stack(((gx-K[0,2])*gz/K[0,0],(gy-K[1,2])*gz/K[1,1],gz),axis=-1).reshape(-1,3)
    bp=(np.asarray(obs['T_body_camera'])@np.c_[cp,np.ones(len(cp))].T).T[:,:3]
    good=np.isfinite(bp).all(axis=1)&(cp[:,2]>0)&(bp[:,0]>=0)&(bp[:,0]<=7)&(bp[:,1]>=-7)&(bp[:,1]<=2)&(bp[:,2]>.2)&(bp[:,2]<2.2)
    fig,ax=plt.subplots(figsize=(6.4,4.8),dpi=180,layout='constrained')
    points=bp[good][::3]
    ax.scatter(points[:,1],points[:,0],c='#8b9cad',s=3,alpha=.22,zorder=1)
    ax.plot(body[show,1],body[show,0],color='#ee82ee',ls='--',lw=3,label='未来位置 GT（前 10 m）',zorder=3)
    ax.plot(body[valid,1],body[valid,0],color='#00a9ac',lw=5,label='截断后参考线',zorder=4)
    ax.scatter(body[anchors,1],body[anchors,0],c='#ffbd42',s=55,edgecolors='white',zorder=6)
    for j,i in enumerate(anchors):
        ax.annotate(f'W{j+1}',(body[i,1],body[i,0]),xytext=(0,-16),textcoords='offset points',color='#b77800',fontsize=11,ha='center',zorder=7)
    ax.scatter(0,0,c='#da4e55',s=38,zorder=6)
    ax.annotate('',xy=(0,.9),xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#da4e55','lw':1.8},zorder=7)
    ax.text(-.35,.35,'当前车体',fontproperties=cn,color='#bb3943',zorder=7)
    ax.annotate('截断点',(body[end,1],body[end,0]),xytext=(-4.5,5.5),
                fontproperties=cn,color='#00878b',arrowprops={'arrowstyle':'->','color':'#00878b'},zorder=7)
    ax.set(xlim=(1.5,-7),ylim=(-.5,7),aspect='equal')
    ax.set_xlabel('y left (m)');ax.set_ylabel('x forward (m)')
    ax.tick_params(labelsize=9);ax.grid(alpha=.16)
    ax.legend(loc='upper left',prop=cn,framealpha=.95,borderpad=.3)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    fig.savefig(ASSETS/'demonstration-topdown.png',dpi=180);plt.close(fig)
    # Oracle illustration uses existing authored route and actual recorded mapping motion.
    planfile=ROOT/'demo/warehouse_navigation/decision-plan-v2.json'
    plan=json.loads(planfile.read_text())
    planned=np.array([[0,0]]+[p for s in plan['loop_stages'] for p in s['waypoints_odom']])
    actual=np.array([np.array(r['T_odom_body'])[:2,3] for r in rows if r['task_id']=='mapping'])
    fig,ax=axes()
    ax.plot(planned[:,0],planned[:,1],'--',lw=2,color='#da8b24',label='Planned route illustration')
    ax.plot(actual[:,0],actual[:,1],color='#168aab',lw=1.6,label='Recorded execution')
    for x,y,title in [(21,0,'1'),(21,-5,'2'),(0,-5,'3')]:
        ax.scatter(x,y,color='#da8b24',s=35,zorder=8);ax.annotate(title,(x,y),xytext=(4,8),textcoords='offset points',weight='bold')
    ax.add_patch(Circle((0,0),.65,fill=False,lw=2,edgecolor='#168aab'))
    ax.annotate('Start / goal region',(0,0),xytext=(1.5,3.25),fontsize=9,bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':2},arrowprops={'arrowstyle':'->'})
    ax.legend(loc='lower right',fontsize=8);save(fig,'oracle-bev.png')
    taskdir=ROOT/'demo/semantic_mapping/warehouse-navigation/frames/00000951'
    shutil.copyfile(taskdir/'rgb.png',ASSETS/'task-b-rgb.png')
    with Image.open(taskdir/'map.png') as im:
        im.crop((0,70,775,760)).save(ASSETS/'task-a-map.png')
    inp=json.loads((taskdir/'input.json').read_text())
    evidence={'source_database':str(dbpath),'database_sha256':before,'database_unchanged':sha(dbpath)==before,
      'map_background':{'file':str((base/'global-occupancy.npz').relative_to(ROOT)),'sha256':sha(base/'global-occupancy.npz'),'scope':'Full-episode observed geometry for illustration only, not a causal model input.'},
      'demonstration':{'image_id':obs['image_id'],'rgb_sha256':obs['rgb_hash'],'depth_sha256':obs['depth_hash'],'max_arc_length_m':limit,'truncation_reason':'outside_image','next_projected_uv':uv[end+1].tolist(),'depth_tolerance_m':.15,'visible_start_m':float(sample[start]),'visible_end_m':float(sample[end]),'waypoint_uv':uv[anchors].tolist(),'reference_uv':uv[valid].tolist(),'reference_odom':xyz[valid].tolist(),'future_odom':future.tolist(),'reference_body_xyz':body[valid].tolist(),'future_display_body_xyz':body[show].tolist(),'topdown_display_arc_length_m':10.0,'future_inframe_uv':uv[inframe].tolist(),'rgb_overlay':{'future':'magenta dashed, positive-depth in-frame projection only','reference':'cyan solid, depth-supported prefix within 7.75 m','anchors':'two gold dots'}},
      'memory':{'input':str(taskdir/'input.json'),'input_sha256':sha(taskdir/'input.json'),'map_sha256':sha(taskdir/'map.png'),'source_time':inp['source']['t_obs'],'map_cutoff':inp['map_cutoff'],'crop_xyxy':[0,70,775,760],'target_index':16,'target_text':'04/5R','target_marker_source_pixel':[547,499]},
      'oracle':{'plan_source':str(planfile.relative_to(ROOT)),'plan_sha256':sha(planfile),'planned_xy':planned.tolist(),'execution_xy':actual.tolist(),'status':'Existing authored plan and recorded motion illustrate the proposed Oracle; no new NavMesh/SE2 execution or collision validation.'},
      'selected_turn_frame':{'image_id':obs['image_id'],'t_obs':obs['t_obs'],'scope':'After the first right turn, actual future-position trajectory projected into current RGB. Visible prefix is clipped at the camera boundary.'},
      'scope':'Documentation illustration only; source data and training labels unchanged.'}
    (ASSETS/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print('projected prefix',sample[start],sample[end],uv[anchors].tolist());print('next',sample[end+1],uv[end+1].tolist(),'inside',bool(inframe[end+1]),'depth_error',float(cam[end+1,2]-measured[end+1]))


def draw_dataengine(d,root,width):
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    c=d.canv;assets=root/'docs/design/figures/dataengine-assets'
    def photo(name,x,y,w,h):
        with Image.open(assets/name) as im:
            iw,ih=im.size;scale=min(w/iw,h/ih);pw=iw*scale;ph=ih*scale
            ox=x+(w-pw)/2;oy=y+(h-ph)/2
            c.drawImage(ImageReader(im),ox,oy,pw,ph,mask='auto')
            return ox,oy,pw,ph,iw,ih
    def card(title,name,x,y,w,h):
        d.panel(x,y,w,h)
        d.label(title,x+7,y+h-7,w-14,'diagram_title',TA_CENTER)
        return photo(name,x+4,y+4,w-8,h-30)
    if d.kind=='demonstration':
        panel_w=(width-12)/2
        card('当前 RGB · 未来轨迹与截取段','demonstration-rgb.png',0,0,panel_w,222)
        card('二维俯视 · GT 与截断后参考线','demonstration-topdown.png',panel_w+12,0,panel_w,222)
        return
    if d.kind=='memorytask':
        w=(width-20)/2
        ox,oy,pw,ph,iw,ih=card('任务 A → 历史语义地图','task-a-map.png',0,92,w,238)
        # Stored map legend identifies instance 16 as 04/5R. Display-only target callout.
        px=ox+547/iw*pw;py=oy+(ih-(499-70))/ih*ph
        c.setStrokeColor(colors.HexColor('#d89020'));c.setLineWidth(1.6);c.circle(px,py,7,fill=0,stroke=1)
        d.label('16 · 04/5R',px-40,py-9,90,'diagram_title',TA_CENTER)
        card('任务 B → 当前 RGB','task-b-rgb.png',w+20,92,w,238)
        d.arrow(w+2,206,w+18,206)
        d.label('Go to sign 04/5R and STOP beside it.',0,83,width,'diagram_title',TA_CENTER)
        for j,(title,detail) in enumerate((('已登记目标','地图检索 → Local / Route → 接近'),('尚未登记目标','搜索 → 发现 → 登记 → 接近'))):
            x=j*(w+20);d.panel(x,0,w,53)
            d.label(title,x+5,46,w-10,'diagram_title',TA_CENTER)
            d.label(detail,x+5,24,w-10,'diagram',TA_CENTER)
        return
    if d.kind in ('navmesh','navmeshquery'):
        photo('navmesh-polygons.png' if d.kind=='navmesh' else 'navmesh-query.png',0,0,width,260)
        return
    if d.kind=='staticoracle':
        left=width*.66
        card('规划示意与已有执行轨迹','oracle-bev.png',0,44,left,256)
        x=left+15;w=width-x
        for i,title in enumerate(('几何 → NavMesh','A* 搜索／中线偏好','footprint／SE(2)','控制器执行','实际轨迹与几何 GT')):
            y=254-i*48;d.panel(x,y,w,36)
            d.label(title,x+4,y+25,w-8,'diagram',TA_CENTER)
            if i<4:d.arrow(x+w/2,y-1,x+w/2,y-11)
        d.label('轨迹 Oracle：几何路线　　语义教师：Instruction / Landmark / Local / Route',0,29,width,'diagram',TA_CENTER)

if __name__=='__main__':prepare()
