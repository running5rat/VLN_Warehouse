"""Recorded RGB/pose review videos with frozen source semantic and route labels.

No simulation, model inference, action supervision or annotation approval occurs.
Future poses are evaluation material only. DynaNav remains camera-centred.
"""
import argparse
import csv
import html
import json
import math
from pathlib import Path
import subprocess
import textwrap
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from ..schema import file_sha256
from .catalog import csv_rows
from .core import load_episode, read_json, read_lines, write_json
from .dynanav_semantic import R_USD_TO_FLU
from .semantic_annotation import validate_sample

W, H = 1920, 1080
BG = '#15212c'
WHITE = '#edf3fa'
ORANGE = '#ffb02e'
CYAN = '#55deef'
GREEN = '#48e89f'
RED = '#ff6868'
FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def relative_poses(poses, origin):
    relative = np.linalg.inv(np.asarray(origin)) @ np.asarray(poses)
    xy = relative[:, :2, 3]
    yaw = np.unwrap(np.arctan2(relative[:, 1, 0], relative[:, 0, 0]))
    return xy, yaw


def replay_indices(times, fps):
    """Zero-order hold recorded images at original speed; no invented end motion."""
    times = np.asarray(times, float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Expected nonempty monotonic source times')
    if fps <= 0:
        raise ValueError('Positive fps required')
    ticks = times[0] + np.arange(int(math.ceil((times[-1]-times[0])*fps-1e-9))+1)/fps
    idx = np.maximum(0, np.searchsorted(times, ticks+1e-8, side='right')-1)
    return np.minimum(idx, len(times)-1)


def load_recorded(sample):
    provenance = sample['provenance']; obs = sample['observation']
    if provenance.get('source_episode'):
        episode = Path(provenance['source_episode'])
        _, frames = load_episode(episode)
        intervals = read_lines(episode.parents[2]/'recording-events.jsonl')
        def recorded(frame):
            preceding = [e for e in intervals if e['monotonic'] <= frame.get('source_monotonic', -1)]
            return not intervals or bool(preceding and preceding[-1]['state']=='recording')
        start = provenance['frame_index']; result = []
        for i in range(start, len(frames)):
            f = frames[i]
            if f['odom_epoch'] != obs['odom_epoch'] or not recorded(f):
                break
            result.append(dict(index=i, t=f['t_obs'], path=f['rgb'], pose=f['T_odom_body']))
        origin = obs['T_odom_body']
        kind = obs.get('pose_reference','robot_body_flu')
        sources = [episode, episode.parent/'frames.jsonl']
        if intervals:sources.append(episode.parents[2]/'recording-events.jsonl')
    else:
        raw = Path(provenance['original_instruction_file']).parent
        rows, _ = csv_rows(raw/'trajectory.csv')
        image = Path(obs['rgb_history'][-1]['path'])
        start = int(obs['image_id'].split(':')[-1]); result=[]
        for i in range(start, len(rows)):
            r=rows[i]
            if float(r['time']) != i:
                raise ValueError('Unexpected DynaNav frame clock')
            pose=np.eye(4)
            pose[:3,:3]=Rotation.from_quat([float(r[k]) for k in ('qx','qy','qz','qw')]).as_matrix() @ R_USD_TO_FLU.T
            pose[:3,3]=[float(r[k]) for k in ('x','y','z')]
            result.append(dict(index=i,t=i/30.,path=str(image.parent/f'rgb_{i:05d}.jpg'),pose=pose.tolist()))
        origin=obs['T_odom_camera_flu'];kind='camera_center_flu_not_robot_base'
        sources=[raw/'trajectory.csv',Path(provenance['camera_file'])]
    if not result or abs(result[0]['t']-obs['t_obs'])>1e-5 or not np.allclose(result[0]['pose'],origin,atol=1e-5):
        raise ValueError('Source pose/time does not match recorded replay')
    return result,np.asarray(origin),kind,[dict(path=str(p),sha256=file_sha256(p)) for p in sources]


def point_distances(points, line):
    if len(line)<2:return [None]*len(points)
    a=line[:-1];delta=np.diff(line,axis=0);length2=np.sum(delta**2,axis=1)
    t=np.clip(((points[:,None]-a)*delta).sum(-1)/np.maximum(length2,1e-12),0,1)
    distances=np.linalg.norm(points[:,None]-(a+t[:,:,None]*delta),axis=-1)
    return distances.min(axis=1).tolist()


class Panel:
    def __init__(self, sample, proposal, xy, line, anchors, bounds_points=None):
        self.base=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(self.base)
        self.font=ImageFont.truetype(FONT_PATH,21);self.small=ImageFont.truetype(FONT_PATH,18)
        self.title=ImageFont.truetype(FONT_PATH,26)
        self.text(d,(20,12),sample['observation']['image_id']+'  |  Recorded future replay  |  1x source time',self.title)
        self.text(d,(20,49),'Labels stay FROZEN at the source frame. No per-frame inference, AE prediction or new Isaac rollout.',self.small,'#ffdc94')
        self.text(d,(20,79),'RGB: recorded camera view',self.font)
        self.text(d,(1000,79),'Source-frame XY: x forward/up, y left/left (metres)',self.font)
        self.line=line;self.anchors=anchors
        pts=np.concatenate([xy,line,anchors,np.zeros((1,2))]) if bounds_points is None else np.asarray(bounds_points)
        mn=pts.min(axis=0)-.8;mx=pts.max(axis=0)+.8
        span=np.maximum(mx-mn,2); self.scale=min(830/span[1],460/span[0]); self.center=(mn+mx)/2
        self.cx=1440;self.cy=360
        d.rectangle((1000,110,1900,620),fill='#20313c')
        for k in (0,1):
            step=max(1,int(math.ceil(span[k]/10)))
            for v in range(int(math.floor(mn[k])),int(math.ceil(mx[k]))+1,step):
                if k==0:a,b=self.px([v,mn[1]]),self.px([v,mx[1]])
                else:a,b=self.px([mn[0],v]),self.px([mx[0],v])
                d.line([a,b],fill='#344854');self.text(d,a,str(v),self.small,'#8a9aa5')
        self.draw_line(d,xy,'#53697a',2)
        self.draw_line(d,line,ORANGE,5)
        for i,p in enumerate(anchors):
            x,y=self.px(p);d.ellipse((x-5,y-5,x+5,y+5),fill=CYAN)
            self.text(d,(x+7,y-18),'W'+str(i+1),self.small,CYAN)
        x,y=self.px([0,0]);d.ellipse((x-5,y-5,x+5,y+5),fill='white')
        self.text(d,(1010,630),'Gray: recorded future (review only)   Green: replayed path',self.small)
        self.text(d,(1010,657),'Orange: source ref_line   Cyan: waypoints   Red: current pose',self.small)
        parsed=proposal['parsed'];route=parsed['route'];geo=proposal['geometry']
        labels='; '.join(x['category']+' @'+x['image_id'].split(':')[-1] for x in parsed['landmark']) or '[]'
        y=700
        for heading,value in [('landmark',labels),('local',parsed['local']),('Route',route['remaining_instruction']),
                              ('subgoal / status',json.dumps(route['current_subgoal'])+' / '+route['status'])]:
            self.text(d,(1005,y),heading+':',self.font,CYAN);y+=26
            for text in textwrap.wrap(value,width=78):
                self.text(d,(1005,y),text,self.small);y+=23
            y+=9
        if y>1070:raise ValueError('Semantic panel text overflow')
        y=852
        for label,value in [('waypoint',json.dumps(geo['waypoint'])),
                            ('ref_line',str(len(line))+' valid points / 32; '+geo['reference_state']),
                            ('Geometry source',geo['source'])]:
            for text in textwrap.wrap(label+': '+value,width=82):
                self.text(d,(20,y),text,self.small,ORANGE if label=='ref_line' else WHITE);y+=24
        self.text(d,(20,1010),'Full arrays, frame timestamps and poses: case.json + replay-frames.jsonl',self.small)
        self.text(d,(20,1038),'Review proposals only. Format validity is not semantic/trajectory correctness.',self.small,'#ffdc94')

    def text(self,d,p,s,font=None,fill=WHITE):d.text(p,s,font=font or self.font,fill=fill)
    def px(self,p):return (self.cx-(p[1]-self.center[1])*self.scale,self.cy-(p[0]-self.center[0])*self.scale)
    def draw_line(self,d,points,color,width):
        if len(points)>1:d.line([self.px(p) for p in points],fill=color,width=width)

    def frame(self,rgb,xy,yaw,path,clock,kind,no_future,hold):
        out=self.base.copy();im=rgb.copy();im.thumbnail((960,636))
        out.paste(im,(20+(960-im.width)//2,110+(636-im.height)//2))
        d=ImageDraw.Draw(out)
        self.draw_line(d,path,GREEN,4)
        self.draw_line(d,self.line,ORANGE,3)
        p=self.px(xy);q=self.px(xy+np.array([math.cos(yaw),math.sin(yaw)])*.6)
        d.line([p,q],fill=RED,width=5);d.ellipse((p[0]-5,p[1]-5,p[0]+5,p[1]+5),fill=RED)
        direction=np.array(q)-p;direction/=max(np.linalg.norm(direction),1e-9)
        perpendicular=np.array([-direction[1],direction[0]])
        d.polygon([q,tuple(np.array(q)-direction*10+perpendicular*5),
                   tuple(np.array(q)-direction*10-perpendicular*5)],fill=RED)
        d.rectangle((20,746,980,830),fill='#15212c')
        self.text(d,(30,751),clock,self.font)
        self.text(d,(30,778),f'x={xy[0]:.2f} m  y={xy[1]:.2f} m  yaw={math.degrees(yaw):+.1f} deg | {kind}',self.small)
        self.text(d,(30,804),'NO FUTURE RECORD: display hold only' if no_future else ('DISPLAY HOLD: no additional motion evidence' if hold else 'Recorded RGB and pose share the displayed source timestamp.'),self.small,'#ffdc94')
        return out


def render_case(sample, proposal, dest, fps=5):
    dest.mkdir(parents=True,exist_ok=False)
    frames,origin,kind,sources=load_recorded(sample)
    times=np.array([f['t'] for f in frames]);poses=np.array([f['pose'] for f in frames]);xy,yaw=relative_poses(poses,origin)
    geo=proposal['geometry'];line=np.array(geo['reference_xy'])[np.array(geo['reference_mask'],bool)]
    anchors=np.array(geo['waypoint'],float).reshape(-1,2)
    chosen=replay_indices(times,fps).tolist();actual_count=len(chosen)
    duration=float(times[-1]-times[0])
    # A terminal source still gets a viewable clip, explicitly marked as a hold.
    if len(chosen)<3*fps:chosen += [chosen[-1]]*(3*fps-len(chosen))
    panel=Panel(sample,proposal,xy,line,anchors)
    report=dict(source_image_id=sample['observation']['image_id'],source_binding=sample['provenance']['source_binding'],
        mode='recorded_rgb_pose_replay',labels_frozen_at_source=True,model_calls=0,new_simulator_rollout=False,
        geometry_frame=kind,action_gt_generated=False,source_start_s=float(times[0]),source_end_s=float(times[-1]),
        recorded_future_duration_s=duration,has_future_motion_record=len(frames)>1,
        output_fps=fps,output_frames=len(chosen),display_hold_frames=len(chosen)-actual_count,
        playback_speed=1,source_records=len(frames),sampled_unique_rgb_frames=len(set(chosen)),
        sources=sources,source_pose=origin.tolist(),parsed=proposal['parsed'],geometry=geo,
        human_approved=False,independent_semantic_accuracy=None,
        limitations=['Source Local describes source time, not the evolving playback frame.',
                     'Reference is a finite local geometric line, not a timed AE prediction.',
                     'Future trace is review material only; full-tail deviation is not a route accuracy score.'])
    write_json(dest/'case.json',report)
    dist=point_distances(xy,line)
    with (dest/'pose-track.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['source_index','t_obs','elapsed_s','x_m','y_m','yaw_rad','distance_to_source_reference_m'])
        for i,f in enumerate(frames):writer.writerow([f['index'],f['t'],f['t']-times[0],*xy[i],yaw[i],dist[i]])
    ffmpeg=['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}',
        '-framerate',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','23',
        '-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'replay.mp4')]
    cache_idx=None;image=None
    with (dest/'ffmpeg.log').open('w') as errors, (dest/'replay-frames.jsonl').open('w') as ledger:
        proc=subprocess.Popen(ffmpeg,stdin=subprocess.PIPE,stderr=errors)
        try:
            for n,i in enumerate(chosen):
                f=frames[i]
                if i!=cache_idx:
                    with Image.open(f['path']) as opened:image=opened.convert('RGB')
                    cache_idx=i;sha=file_sha256(f['path'])
                hold=n>=actual_count
                clock=f'RGB frame {f["index"]:08d} | t={f["t"]:.3f}s | source age +{f["t"]-times[0]:.2f}s'
                canvas=panel.frame(image,xy[i],yaw[i],xy[:i+1],clock,kind,len(frames)==1,hold)
                if n==0:canvas.save(dest/'poster.jpg',quality=92)
                if n==actual_count-1:canvas.save(dest/'last-recorded.jpg',quality=92)
                proc.stdin.write(canvas.tobytes())
                ledger.write(json.dumps(dict(video_frame=n,video_time_s=n/fps,source_index=f['index'],source_time_s=f['t'],
                    rgb=f['path'],rgb_sha256=sha,display_hold=hold,label_source_image_id=report['source_image_id'],
                    x_m=float(xy[i,0]),y_m=float(xy[i,1]),yaw_rad=float(yaw[i])))+'\n')
        finally:
            proc.stdin.close();code=proc.wait()
        if code:raise RuntimeError(f'ffmpeg failed: {dest}/ffmpeg.log')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest/'replay.mp4')]))
    stream=probe['streams'][0]
    if int(stream['nb_frames'])!=len(chosen) or stream['codec_name']!='h264':raise ValueError('Encoded video mismatch')
    report.update(video_sha256=file_sha256(dest/'replay.mp4'),video_duration_s=float(probe['format']['duration']),
                  video_bytes=(dest/'replay.mp4').stat().st_size,codec=stream['codec_name'],encoded_frames=int(stream['nb_frames']))
    write_json(dest/'case.json',report)
    return report


def add_webm(output):
    """Browser fallback for builds without proprietary H.264 decoding."""
    output=Path(output);report=read_json(output/'report.json')
    def encode(case):
        folder=output/case['folder'];dest=folder/'replay.webm'
        if not dest.exists():
            subprocess.run(['ffmpeg','-v','error','-i',str(folder/'replay.mp4'),'-an','-c:v','libvpx-vp9',
                '-deadline','realtime','-cpu-used','8','-row-mt','1','-threads','2','-crf','35','-b:v','0',str(dest)],check=True)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)]))
        if probe['streams'][0]['codec_name']!='vp9':raise ValueError('Expected VP9 fallback')
        if abs(float(probe['format']['duration'])-case['video_duration_s'])>.05:raise ValueError('Fallback clock mismatch')
        case['webm']=dict(sha256=file_sha256(dest),bytes=dest.stat().st_size,codec='vp9')
        saved=read_json(folder/'case.json');saved['webm']=case['webm'];write_json(folder/'case.json',saved)
        print('WebM '+case['folder'],flush=True)
        return case
    with ThreadPoolExecutor(max_workers=2) as pool:report['cases']=list(pool.map(encode,report['cases']))
    page=(output/'index.html').read_text()
    for case in report['cases']:
        folder=case['folder'];tag=f'<source src="{folder}/replay.webm" type="video/webm">'
        if tag not in page:page=page.replace(f'<source src="{folder}/replay.mp4" type="video/mp4">',tag+f'<source src="{folder}/replay.mp4" type="video/mp4">')
    (output/'index.html').write_text(page)
    report['webm_renderer_sha256']=file_sha256(__file__)
    write_json(output/'report.json',report)
    return report


def run(session_root,output,fps=5):
    session_root=Path(session_root).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'renderer-source.py').write_bytes(Path(__file__).read_bytes())
    rows=[]
    for group in ('manual','dynanav'):
        session=read_json(session_root/group/'report.json');bundle=Path(session['bundle']);manifest=read_json(bundle/'manifest.json')
        if file_sha256(bundle/'manifest.json')!=session['bundle_sha256']:raise ValueError('Bundle changed')
        for row in session['samples']:
            path=bundle/(row['folder']+'.json')
            if file_sha256(path)!=manifest['records'][row['index']]['sha256']:raise ValueError('Sample changed')
            sample=validate_sample(read_json(path));p=session_root/group/row['folder']/'proposal.json'
            if file_sha256(p)!=row['proposal_sha256']:raise ValueError('Proposal changed')
            proposal=read_json(p)
            if proposal['source_binding']!=sample['provenance']['source_binding']:raise ValueError('Proposal source mismatch')
            name=group+'-'+row['source_image_id'].split(':')[-1]
            result=render_case(sample,proposal,output/name,fps)
            result.update(folder=name,proposal_file=str(p),proposal_sha256=file_sha256(p));rows.append(result)
            print(json.dumps(dict(case=name,frames=result['output_frames'],duration=result['video_duration_s'])),flush=True)
            write_json(output/'report.json',dict(status='running',cases=rows))
    report=dict(status='complete',cases=rows,case_count=len(rows),model_calls=0,action_gt_generated=False,
                labels_modified=False,source_session=str(session_root),renderer_sha256=file_sha256(output/'renderer-source.py'))
    write_json(output/'report.json',report)
    parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>12 个标注时刻：文本与未来运动回放</title>',
      '<style>body{font:17px system-ui;background:#15212c;color:#edf3fa;max-width:1500px;margin:24px auto;padding:16px}a{color:#77d9ff}video{width:100%;background:#000}section{margin:36px 0;border-top:1px solid #789;padding-top:20px}.note{background:#493b1e;padding:16px}</style>',
      '<h1>12 个标注时刻：文本、参考线与实录未来运动</h1>',
      f'<p class="note">每段从该标注时刻回放至连续记录结束，按原始时间 1 倍播放，输出 {fps} fps。右侧文本及橙色参考线固定在源帧；后续图像不触发模型推理。绿色为已经回放的实测轨迹，灰色为完整未来轨迹（仅审核辅助），青色为五个锚点，红色为当前位置与朝向。图中坐标单位米。</p>',
      '<p>这是同步 RGB/位姿实录回放，不是重新驱动车辆的 Isaac 仿真。DynaNav 为相机中心轨迹，缺少已核验车体外参。终帧无未来数据时只静止展示并注明，不制造动作 GT。局部参考线覆盖有限，后续离开其范围不能直接算作导航失败。</p>',
      '<p><a href="report.json">完整报告</a></p>']
    for row in rows:
        n=row['folder'];parts.extend([f'<section><h2>{html.escape(row["source_image_id"])}</h2>',
          f'<p>真实未来记录 {row["recorded_future_duration_s"]:.2f} 秒；视频 {row["video_duration_s"]:.2f} 秒；坐标：{html.escape(row["geometry_frame"])}</p>',
          f'<video controls preload="metadata" poster="{n}/poster.jpg"><source src="{n}/replay.mp4" type="video/mp4"></video>',
          f'<p><a href="{n}/replay.mp4">下载 MP4</a> · <a href="{n}/case.json">标注及完整参考线</a> · <a href="{n}/pose-track.csv">实际轨迹 CSV</a> · <a href="{n}/replay-frames.jsonl">逐视频帧来源</a></p></section>'])
    (output/'index.html').write_text(''.join(parts)+'</html>')
    return add_webm(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--session-root');p.add_argument('--output',required=True);p.add_argument('--fps',type=int,default=5)
    p.add_argument('--add-webm',action='store_true',help='Add browser fallback to an existing render')
    a=p.parse_args()
    if a.add_webm:add_webm(a.output)
    elif a.session_root:run(a.session_root,a.output,a.fps)
    else:p.error('--session-root is required when rendering a new session')


if __name__=='__main__':main()
