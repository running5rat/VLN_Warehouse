"""Fresh session-assisted labels, goal-only target and causal readable maps. No API."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ticvla.warehouse.geometry import project_depth, transform_points
from ticvla.warehouse.schema import file_sha256
from ticvla.warehouse.dataset_tools.core import read_json, write_json
from demo.warehouse_navigation.rgb_oracle_review import SYSTEM, user_input, finish_frame, report


def prepare(old, root):
    old, root=Path(old),Path(root)
    root.mkdir(parents=True,exist_ok=False)
    exp=read_json(old/'experiment.json')
    exp.update(providers=['codex-session'],parent_review=str(old),
        annotation_mode='current Codex session assisted reannotation; not blind; no API',
        input_map='same causal semantic evidence, legible indexed rendering; original independent map policy retained',
        comparison_limitation='instruction and map rendering changed, with manually revised pixel labels; not a single-factor model experiment')
    exp['instructions']['target']='Go to sign 04/5R and STOP beside it.'
    write_json(root/'experiment.json',exp)
    (root/'system.txt').write_text(SYSTEM+'\nMap labels use indices linked to an on-image legend. The red arrow shows current heading. No route to the target is supplied. Infer remaining commands from the observed map and localization; if uncertain, preserve the goal and state a reobservation command.\n')
    shutil.copytree(old/'inputs',root/'inputs')
    source=Path(exp['source'])
    perception=[read_json(p)['source'] for p in sorted((source/'navigation/perception').glob('*/frame.json'))]
    cells=set();trail=[];cursor=0
    font=ImageFont.truetype('DejaVuSans.ttf',17);small=ImageFont.truetype('DejaVuSans.ttf',14)
    for row in exp['selected']:
        dest=root/'inputs'/f"{row['frame']:08d}";s=read_json(dest/'source.json');a=read_json(dest/'audit.json')
        while cursor<len(perception) and perception[cursor]['t_obs']<s['t_obs']:
            q=perception[cursor];cursor+=1
            points=project_depth(np.load(q['depth']),q['K'],q['T_body_camera'],stride=24)
            points=transform_points(points,q['T_odom_body']);points=points[(points[:,2]>.15)&(points[:,2]<3.5),:2]
            cells.update(map(tuple,np.floor(points/.1).astype(int)));trail.append(np.array(q['T_odom_body'])[:2,3])
        assert a['map_source_cutoff'] is None or a['map_source_cutoff']<s['t_obs']
        landmarks=[x for x in a['map_landmarks'] if x.get('landmark_view',{}).get('visible',True)]
        assert all(x['last_seen']<s['t_obs'] for x in landmarks)
        pose=np.array(s['T_odom_body']);xy=pose[:2,3]
        focus=np.array([xy,*trail,*[x['position'][:2] for x in landmarks]])
        lo=focus.min(0)-3;hi=focus.max(0)+3;center=(lo+hi)/2
        span=max(float((hi-lo).max()),12);lo=center-span/2
        def px(p):
            q=(np.asarray(p)-lo)/span*740+30
            return np.stack([q[...,0],800-q[...,1]],axis=-1)
        im=Image.new('RGB',(1400,900),(246,248,250));arr=np.array(im)
        cloud=np.array(list(cells),dtype=float).reshape(-1,2)*.1
        if len(cloud):
            uv=px(cloud).astype(int);uv=uv[(uv[:,0]>=20)&(uv[:,0]<780)&(uv[:,1]>=20)&(uv[:,1]<780)]
            arr[uv[:,1],uv[:,0]]=[100,115,125]
        im=Image.fromarray(arr);d=ImageDraw.Draw(im)
        if len(trail)>1:d.line([tuple(p) for p in px(trail)],fill='royalblue',width=3)
        for i,lm in enumerate(landmarks):
            point=tuple(px(lm['position'][:2]));color='darkgreen' if lm.get('identity_status')=='identity_confirmed' else 'darkorange'
            d.ellipse((point[0]-4,point[1]-4,point[0]+4,point[1]+4),fill=color)
            d.text((point[0]+5,point[1]-8),str(i+1),font=small,fill=color,stroke_width=1,stroke_fill='white')
            label=f"{i+1}. {lm['category']} | {lm.get('identity_text') or 'no text'}"
            state=lm.get('identity_status','unknown')
            d.text((810,30+i*21),label[:53]+' ['+state.replace('identity_confirmed','C').replace('candidate','?')+']',font=small,fill=color)
        origin=px(xy);ahead=px(xy+pose[:2,0]*1.7)
        d.line([tuple(origin),tuple(ahead)],fill='red',width=5);d.ellipse((* (origin-5),*(origin+5)),fill='red')
        direction=ahead-origin;direction/=max(np.linalg.norm(direction),1e-6);normal=np.array([-direction[1],direction[0]])
        d.polygon([tuple(ahead),tuple(ahead-14*direction+7*normal),tuple(ahead-14*direction-7*normal)],fill='red')
        d.text((30,805),f'Prior observed surfaces and semantic objects. Odom x right, y up. Width {span:.1f} m.',font=font,fill='black')
        d.text((30,832),'Red: current robot / heading. Blue: past travel, NOT planned route. Background: unobserved.',font=font,fill='black')
        d.text((30,859),'C: repeated text confirmed; all objects still tentative for static mapping. ?: uncertain OCR.',font=font,fill='black')
        shutil.copy2(dest/'map.png',dest/'map-original.png');im.save(dest/'map.png')
        a['original_map_sha256']=a['images']['map.png'];a['images']['map.png']=file_sha256(dest/'map.png')
        a['map_rendering']='indexed-legible-v2; no goal-highlight or planned path; geometry strictly before current time'
        a['map_legend']=[dict(index=i+1,uid=l['landmark_uid']) for i,l in enumerate(landmarks)]
        write_json(dest/'audit.json',a)
    print('Prepared',len(exp['selected']),'causal inputs',flush=True)


if __name__=='__main__':
    import sys
    prepare(sys.argv[1],sys.argv[2])
