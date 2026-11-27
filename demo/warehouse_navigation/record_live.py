"""Record actual published RGB/map/curve panels; never substitute future maps."""
import argparse
import json
from pathlib import Path
import subprocess
import textwrap
import time
from PIL import Image,ImageDraw,ImageFont


def run(root, output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('DejaVuSans.ttf',22);nav=root/'navigation';fps=5
    process=None;last={};finished=None;index=0
    timeline=(output/'video-timeline.jsonl').open('w',buffering=1)
    try:
        while True:
            started=time.monotonic()
            if not (nav/'status.json').exists():time.sleep(.5);continue
            status=json.loads((nav/'status.json').read_text())
            response=json.loads((nav/'last-response.json').read_text()) if (nav/'last-response.json').exists() else {}
            source=json.loads((nav/'source.json').read_text())
            canvas=Image.new('RGB',(1920,1080),'#17232d');draw=ImageDraw.Draw(canvas)
            for i,key in enumerate(['rgb','map','plan']):
                try:
                    with Image.open(source['rgb'] if key=='rgb' else nav/f'{key}.png') as im:last[key]=im.convert('RGB').copy()
                except (OSError,ValueError):pass
                if key in last:
                    im=last[key].copy();im.thumbnail((640,640));canvas.paste(im,(i*640,40))
                draw.text((i*640+10,8),{'rgb':'RGB','map':'Semantic map','plan':'Waypoint + fitted curve'}[key],font=font,fill='white')
            waypoint='; '.join(f'({x:.2f}, {y:.2f})' for x,y in response.get('waypoint',[])) or '[]'
            lines=['Instruction: '+response.get('instruction',''),
                'Landmark: '+json.dumps(response.get('landmark',[])),
                'Local: '+response.get('local','unknown'),'Route: '+response.get('route',''),
                ('Waypoint RGB pixels: ' if response.get('waypoint_protocol')=='rgb-pixel-waypoint-v1' else 'Waypoint body metres: ')+waypoint,
                'Output source: '+response.get('source_image_id','pending')+' | '+status['command']]
            y=688
            for line in lines:
                for segment in textwrap.wrap(line,140):draw.text((10,y),segment,font=font,fill='white');y+=28
            if process is None:
                process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1920x1080','-framerate',str(fps),'-i','-',
                    '-c:v','libx264','-preset','veryfast','-crf','23','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'live_demo.mp4')],stdin=subprocess.PIPE)
            process.stdin.write(canvas.tobytes());canvas.save(output/'latest.jpg',quality=85)
            timeline.write(json.dumps(dict(video_index=index,wall=time.time(),source=source['image_id'],phase=status['phase'],request=response.get('request_id'),fps=fps))+'\n');index+=1
            if (nav/'task-finished.json').exists() or (root/'cancel.json').exists() or (root/'report.json').exists():
                finished=finished or time.monotonic()
                if time.monotonic()-finished>2:break
            time.sleep(max(0,1/fps-(time.monotonic()-started)))
    finally:
        timeline.close()
        if process:process.stdin.close();process.wait(timeout=60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(a.root,a.output)
