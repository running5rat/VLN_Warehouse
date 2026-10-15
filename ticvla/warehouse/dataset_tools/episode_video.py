"""One recorded scene/task per video; source-bound annotations switch in time."""
import argparse
import json
import html
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..schema import file_sha256
from .core import read_json, write_json
from .semantic_annotation import validate_sample
from .replay_video import (Panel, load_recorded, relative_poses, add_webm, W, H, BG, CYAN)


def active_annotation(source_time, update_times):
    return int(np.searchsorted(update_times,source_time+1e-9,side='right')-1)


def align_points(points, label_pose, episode_pose):
    points=np.asarray(points,float).reshape(-1,2)
    transform=np.linalg.inv(episode_pose)@label_pose
    xyz=np.column_stack([points,np.zeros(len(points)),np.ones(len(points))])
    return (xyz@transform.T)[:,:2]


def load_labels(root,group):
    session=read_json(root/group/'report.json');bundle=Path(session['bundle']);manifest=read_json(bundle/'manifest.json')
    if file_sha256(bundle/'manifest.json')!=session['bundle_sha256']:raise ValueError('Bundle changed')
    labels=[]
    for row in session['samples']:
        source=bundle/(row['folder']+'.json');proposal=root/group/row['folder']/'proposal.json'
        if file_sha256(source)!=manifest['records'][row['index']]['sha256'] or file_sha256(proposal)!=row['proposal_sha256']:
            raise ValueError('Changed source/proposal')
        sample=validate_sample(read_json(source));answer=read_json(proposal)
        if answer['source_binding']!=sample['provenance']['source_binding']:raise ValueError('Source mismatch')
        labels.append((sample,answer))
    times=[s['observation']['t_obs'] for s,_ in labels]
    if np.any(np.diff(times)<=0):raise ValueError('Unordered annotations')
    if len({s['observation']['task']['task_id'] for s,_ in labels})!=1:raise ValueError('Mixed tasks')
    return labels


def render_episode(labels,dest,fps=10):
    dest.mkdir(parents=True,exist_ok=False)
    sample=labels[0][0];frames,origin,kind,sources=load_recorded(sample)
    times=np.array([f['t'] for f in frames]);xy,yaw=relative_poses([f['pose'] for f in frames],origin)
    update_times=np.array([s['observation']['t_obs'] for s,_ in labels])
    if update_times[-1]>times[-1]+1e-6:raise ValueError('Label is beyond replay')
    # Source clocks have microsecond drift. Snap only within 10 us; never switch
    # based on a future label before its actual source RGB has been displayed.
    clock=times[0]+np.arange(round((times[-1]-times[0])*fps)+1)/fps
    chosen=np.clip(np.searchsorted(times,clock+1e-5,side='right')-1,0,len(times)-1).tolist()
    chosen[-1]=len(times)-1
    real_count=len(chosen);chosen += [chosen[-1]]*(2*fps)
    plots=[]
    for s,a in labels:
        o=s['observation'];pose=np.array(o.get('T_odom_body',o.get('T_odom_camera_flu')))
        g=a['geometry'];line=np.array(g['reference_xy'])[np.array(g['reference_mask'],bool)]
        plots.append((align_points(line,pose,origin),align_points(g['waypoint'],pose,origin)))
    bounds=np.concatenate([xy,np.zeros((1,2)),*[p for pair in plots for p in pair]])
    panels=[]
    for n,((s,a),(line,anchors)) in enumerate(zip(labels,plots)):
        panel=Panel(s,a,xy,line,anchors,bounds_points=bounds);d=ImageDraw.Draw(panel.base)
        d.rectangle((0,0,W,75),fill=BG)
        panel.text(d,(20,12),s['observation']['task']['task_id']+f' | Scene/task replay | Annotation {n+1}/{len(labels)}',panel.title)
        panel.text(d,(20,49),'Active label: '+s['observation']['image_id']+' | Saved answers switch at their source frame; no new inference.',panel.small,CYAN)
        d.rectangle((990,75,W,110),fill=BG)
        panel.text(d,(1000,79),'Episode-start XY (m); active reference transformed into this frame',panel.small)
        d.rectangle((10,1000,990,1037),fill=BG)
        panel.text(d,(20,1010),'Waypoint numbers remain in ACTIVE LABEL frame. Plot uses EPISODE frame.',panel.small)
        panels.append(panel)
    updates=[];last_active=-1;last_index=None;rgb=None
    command=['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}',
        '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'replay.mp4')]
    with (dest/'ffmpeg.log').open('w') as error,(dest/'replay-frames.jsonl').open('w') as ledger:
        proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=error)
        try:
            for n,i in enumerate(chosen):
                f=frames[i];active=active_annotation(f['t'],update_times)
                if active<0:raise ValueError('Missing initial annotation')
                panel=panels[active];obs=labels[active][0]['observation'];age=f['t']-obs['t_obs']
                if i!=last_index:
                    with Image.open(f['path']) as image:rgb=image.convert('RGB')
                    sha=file_sha256(f['path']);last_index=i
                canvas=panel.frame(rgb,xy[i],yaw[i],xy[:i+1],
                    f'RGB {f["index"]:08d} | task +{f["t"]-times[0]:.2f}s | label age {age:.2f}s',kind,False,n>=real_count)
                if active!=last_active:
                    updates.append(dict(annotation=active+1,source_image_id=obs['image_id'],source_time_s=obs['t_obs'],
                        video_frame=n,video_time_s=n/fps,displayed_source_index=f['index'],displayed_source_time_s=f['t'],
                        source_delay_s=age,parsed=labels[active][1]['parsed'],geometry=labels[active][1]['geometry']))
                    canvas.save(dest/f'update-{active+1:02d}.jpg',quality=92)
                    last_active=active
                if n==0:canvas.save(dest/'poster.jpg',quality=92)
                proc.stdin.write(canvas.tobytes())
                ledger.write(json.dumps(dict(video_frame=n,video_time_s=n/fps,source_index=f['index'],source_time_s=f['t'],
                    rgb=f['path'],rgb_sha256=sha,active_label=obs['image_id'],label_age_s=age,
                    display_hold=n>=real_count,x_m=float(xy[i,0]),y_m=float(xy[i,1]),yaw_rad=float(yaw[i])))+'\n')
        finally:proc.stdin.close();code=proc.wait()
    if code:raise RuntimeError('Video encode failed')
    if len(updates)!=len(labels):raise ValueError('Some annotation updates were not displayed')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest/'replay.mp4')]))
    if int(probe['streams'][0]['nb_frames'])!=len(chosen):raise ValueError('Frame count mismatch')
    report=dict(folder=dest.name,episode_id=sample['observation']['task']['task_id'],instruction=sample['observation']['task']['instruction'],
        mode='one_scene_task_per_video',model_calls=0,answers_changed=False,labels_switch_at_source_frame=True,
        plot_frame='episode_start_'+kind,waypoint_numeric_frame='active_annotation_source',
        sources=sources,source_start_s=float(times[0]),source_end_s=float(times[-1]),
        recorded_duration_s=float(times[-1]-times[0]),output_fps=fps,output_frames=len(chosen),display_hold_frames=2*fps,
        video_duration_s=float(probe['format']['duration']),video_sha256=file_sha256(dest/'replay.mp4'),updates=updates,
        display_is_inference_latency_measurement=False,action_gt_generated=False)
    write_json(dest/'case.json',report);write_json(dest/'updates.json',updates)
    print(json.dumps(dict(episode=report['episode_id'],duration=report['video_duration_s'],updates=len(updates))),flush=True)
    return report


def run(root,output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    (output/'renderer-source.py').write_bytes(Path(__file__).read_bytes())
    groups=sorted(p.name for p in root.iterdir() if p.is_dir() and (p/'report.json').exists() and 'bundle' in read_json(p/'report.json'))
    if not groups:raise ValueError('No episode sessions found')
    cases=[render_episode(load_labels(root,group),output/group) for group in groups]
    write_json(output/'report.json',dict(status='complete',cases=cases,episode_count=len(cases),total_updates=sum(len(c['updates']) for c in cases),
        source_session=str(root),model_calls=0))
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>按任务回放：到源帧更新推理结果</title>',
        '<style>body{font:17px system-ui;max-width:1500px;margin:24px auto;padding:16px;background:#15212c;color:#edf3fa}video{width:100%}a{color:#7ddcff}button{margin:5px;padding:9px;cursor:pointer}section{margin:32px 0;border-top:1px solid #678}p{line-height:1.6}</style>',
        '<h1>一个场景任务，一个连续视频</h1><p>到达标注源帧时，五项结果一起切换；两次更新之间保持上一份结果，并显示 label age。右上图固定为任务起点坐标系，参考线按各自源位姿转换，文字坐标值仍属于该次标注帧。原始时间1倍回放，输出10 fps；人工RGB本身为5 Hz，重复显示不算新增观测。末尾增加2秒静止展示，便于阅读最后一次更新。</p>',
        '<p>使用各源帧已保存的会话辅助回答；视频播放时不重新调用模型，切换不包含真实推理延迟。每段从首个标注时刻播放。相机路径重渲染与实际车辆驾驶在来源中分别标注，未认定为动作GT。新增几何候选尚需审核。</p>']
    for c in cases:
        folder=c['folder'];parts.extend([f'<section><h2>{html.escape(c["episode_id"])}</h2><p>{html.escape(c["instruction"])}</p>',
            f'<video id="{folder}" controls preload="metadata" poster="{folder}/poster.jpg"><source src="{folder}/replay.mp4" type="video/mp4"></video>',
            '<p>跳转到结果更新：</p>'])
        for u in c['updates']:
            parts.append(f'<button onclick="document.getElementById(\'{folder}\').currentTime={u["video_time_s"]}">{u["source_image_id"].split(":")[-1]} · {u["video_time_s"]:.1f}s</button>')
        parts.append(f'<p><a href="{folder}/replay.mp4">下载MP4</a> · <a href="{folder}/updates.json">{len(c["updates"])}次完整结果及时间</a> · <a href="{folder}/replay-frames.jsonl">逐视频帧来源</a></p></section>')
    (output/'index.html').write_text(''.join(parts)+'</html>')
    return add_webm(output)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--session-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.session_root,args.output)
