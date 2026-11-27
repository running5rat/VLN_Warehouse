"""Review-only 3D/command diagnostics and continuous source-time replay videos."""
import json
from pathlib import Path
import subprocess
import textwrap
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ticvla.warehouse.dataset_tools.core import read_json,write_json
from ticvla.warehouse.geometry import LocalMapBuilder,project_depth,transform_points
from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.navigation.geometry import render_local
from ticvla.warehouse.navigation.core import pursuit


def diagnostics(root):
    root=Path(root);exp=read_json(root/'experiment.json')
    frames=[json.loads(x) for x in (Path(exp['source'])/'capture/raw/frames.jsonl').read_text().splitlines()]
    for provider in exp.get('providers', ('qwen3.7-plus','codex-session')):
        for row in exp['selected']:
            folder=root/provider/f"{row['frame']:08d}"
            if not (folder/'parsed.json').exists():continue
            s=read_json(folder/'input.json')['source'];parsed=read_json(folder/'parsed.json');pixels=np.asarray(parsed['waypoint'],float)
            if not pixels.size:continue
            depth=np.load(s['depth']);K=np.asarray(s['K']);h,w=depth.shape
            inside=(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
            uv=np.floor(pixels[inside]).astype(int);z=depth[uv[:,1],uv[:,0]];valid=np.isfinite(z)&(z>.05)&(z<30)
            body=transform_points((np.c_[uv[valid],np.ones(valid.sum())]@np.linalg.inv(K).T)*z[valid,None],s['T_body_camera'])
            write_json(folder/'unvalidated-point-diagnostic.json',dict(points_body=body.tolist(),
                       valid_depth_pixel_indices=np.where(inside)[0][valid].tolist(),warning='Diagnostic points only; not accepted control targets'))
            cloud=project_depth(depth,s['K'],s['T_body_camera'],stride=16)
            cloud=cloud[(np.linalg.norm(cloud[:,:2],axis=1)<15)&(cloud[:,2]<4)]
            fig=plt.figure(figsize=(11,5));ax=fig.add_subplot(121);ax.scatter(cloud[:,0],cloud[:,1],s=1,c='lightgray')
            if len(body):ax.scatter(body[:,0],body[:,1],c='orange');ax.plot(body[:,0],body[:,1],c='orange')
            ax.scatter([0],[0],c='red');ax.set(xlabel='source body x forward / m',ylabel='source body y left / m',title='BEV of raw projected pixels');ax.axis('equal')
            ax=fig.add_subplot(122,projection='3d');ax.scatter(cloud[:,0],cloud[:,1],cloud[:,2],s=1,c='lightgray')
            if len(body):ax.scatter(body[:,0],body[:,1],body[:,2],c='orange',s=30)
            ax.set(xlabel='x / m',ylabel='y / m',zlabel='z / m',title='Depth hits: rejection is retained')
            fig.tight_layout();fig.savefig(folder/'projection-3d.png',dpi=100);plt.close(fig)
            if (folder/'projection.json').exists():
                pr=read_json(folder/'projection.json')
                if pr['status']!='projected':continue
                cfg=WarehouseConfig(depth_stride=16,map_resolution=.1,map_size_m=16,map_max_age=5,footprint_radius=.35)
                builder=LocalMapBuilder(cfg);builder.add(depth=depth,**{k:s[k] for k in ('K','T_body_camera','T_odom_body','t_obs','odom_epoch')})
                grid,_=render_local(builder,np.array(s['T_odom_body']),s['t_obs'],s['odom_epoch'])
                cmd,reason=pursuit(np.array(pr['curve_body']),grid,.1,.35,.3)
                write_json(folder/'controller-diagnostic.json',dict(v_mps=float(cmd[0]),w_radps=float(cmd[1]),reason=reason,
                    actual_execution=False,map_context='same-frame depth only; stricter than live accumulated map',
                    warning='Finite command diagnostic, not proof of collision-free execution'))
        coverage=[]
        for task in ('mapping','target'):
            keys=[r for r in exp['selected'] if r['task']==task]
            for a,b in zip(keys,keys[1:]):
                path=root/provider/f"{a['frame']:08d}"/'projection.json'
                item=dict(source_frame=a['frame'],next_frame=b['frame'],gap_s=b['t_obs']-a['t_obs'],
                          kind='offline recorded motion audit, not movement caused by this prediction')
                if path.exists() and read_json(path)['status']=='projected':
                    pr=read_json(path);curve=np.asarray(pr['curve_body'])
                    length=float(np.linalg.norm(curve[0])+np.linalg.norm(np.diff(curve,axis=0),axis=1).sum())
                    trail=np.array([np.array(f['T_odom_body'])[:2,3] for f in frames[a['frame']:b['frame']+1]])
                    travelled=float(np.linalg.norm(np.diff(trail,axis=0),axis=1).sum())
                    item.update(approximate_visible_prefix_plus_connector_m=length,recorded_travel_m=travelled,
                                distance_exhausted_before_next_sample=travelled>length,
                                note='Length check only; does not establish route agreement or safe connector')
                else:item['status']='no_valid_projected_guidance'
                coverage.append(item)
        write_json(root/provider/'coverage.json',coverage)


def videos(root):
    root=Path(root);exp=read_json(root/'experiment.json');source=Path(exp['source'])/'capture/raw'
    frames=[json.loads(x) for x in (source/'frames.jsonl').read_text().splitlines()]
    font=ImageFont.truetype('DejaVuSans.ttf',22);small=ImageFont.truetype('DejaVuSans.ttf',18)
    for provider in exp.get('providers', ('qwen3.7-plus','codex-session')):
        for task in ('mapping','target'):
            keys=[r for r in exp['selected'] if r['task']==task];indices=[r['frame'] for r in keys]
            output=root/provider/f'{task}.mp4'
            proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1440',
                 '-r','5','-i','-','-an','-c:v','libx264','-preset','fast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
            key=None;parsed={};pr={};status={};mapim=None;raw='';previous_route='';controller={}
            try:
                for i in range(indices[0],indices[-1]+1):
                    active=max(n for n in indices if n<=i)
                    if key!=active:
                        key=active;folder=root/provider/f'{active:08d}'
                        parsed=read_json(folder/'parsed.json') if (folder/'parsed.json').exists() else {}
                        raw=(folder/'raw.txt').read_text() if (folder/'raw.txt').exists() else 'No generated response'
                        pr=read_json(folder/'projection.json') if (folder/'projection.json').exists() else {}
                        status=read_json(folder/'result.json') if (folder/'result.json').exists() else dict(status='not_run')
                        mapim=ImageOps.contain(Image.open(root/'inputs'/f'{active:08d}'/'map.png').convert('RGB'),(620,480))
                        inp=read_json(folder/'input.json')
                        previous_route=(inp.get('previous') or {}).get('route',inp['original_instruction_audit_only'])
                        controller=read_json(folder/'controller-diagnostic.json') if (folder/'controller-diagnostic.json').exists() else {}
                    f=frames[i];rgb=Image.open(source/f['rgb']).convert('RGB');d=ImageDraw.Draw(rgb)
                    if pr.get('curve_odom'):
                        cam=transform_points(np.c_[pr['curve_odom'],np.zeros(len(pr['curve_odom']))],np.linalg.inv(np.array(f['T_odom_body'])@np.array(f['T_body_camera'])))
                        uv=cam@np.array(f['K']).T;uv=uv[:,:2]/np.maximum(uv[:,2:3],1e-5)
                        depth=np.load(source/f['depth'])
                        for p,z in zip(uv,cam[:,2]):
                            u,v=p
                            if 0<=u<960 and 0<=v<720 and z>0:
                                measured=depth[int(v),int(u)]
                                if np.isfinite(measured) and abs(measured-z)<.25:d.ellipse((u-1,v-1,u+1,v+1),fill='cyan')
                    canvas=Image.new('RGB',(1920,1440),(20,32,43));canvas.paste(rgb.resize((1280,960)),(0,0));canvas.paste(mapim,(1290,0));draw=ImageDraw.Draw(canvas)
                    def text(x,y,s,width,font=font,color='white'):
                        for line in textwrap.wrap(s,width):draw.text((x,y),line,font=font,fill=color);y+=27
                        return y
                    y=text(1290,490,provider+' / '+task,45)
                    y=text(1290,y,f'Output source frame: {key}',45)
                    y=text(1290,y,'Landmark: '+json.dumps(parsed.get('landmark',[])),46)
                    y=text(1290,y,'Local: '+parsed.get('local','invalid or missing'),46)
                    y=text(1290,y,'Waypoint pixels: '+json.dumps(parsed.get('waypoint',[])),48,small)
                    y=text(1290,y,'Projection: '+status.get('projection_status','not_run'),45,small,'orange')
                    y=text(1290,y,'Projection is NOT route approval.',45,small,'orange')
                    y=text(1290,y,'Controller diagnostic: '+controller.get('reason','not_run'),45,small)
                    route=parsed.get('route') or ('INVALID OUTPUT: '+raw)
                    text(12,980,'Previous Route: '+previous_route,130,font)
                    text(12,1180,'Generated Route: '+route,130,font,color='cyan')
                    text(12,1400,'Text context: '+('committed to next inference' if status.get('format_valid') else 'rejected; keep last valid context')+' | New prediction did NOT control this recorded vehicle.',145,small,'orange')
                    draw.rectangle((0,0,1250,32),fill=(20,32,43));draw.text((8,3),f'Replay {f["t_obs"]:.1f}s | Oracle frame {key} | age {f["t_obs"]-frames[key]["t_obs"]:.1f}s | source-time alignment, NOT online latency',font=small,fill='white')
                    proc.stdin.write(np.array(canvas).tobytes())
            finally:
                proc.stdin.close();rc=proc.wait()
            if rc:raise RuntimeError('Video encoding failed')
            print('video',provider,task,flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='diagnostics':diagnostics(sys.argv[2])
    else:videos(sys.argv[2])
