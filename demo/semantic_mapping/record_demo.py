"""Record the actual asynchronously published mapping panels during an Isaac run.

This observer never changes perception, map state or vehicle control. Video time
is wall time (including held frames), not model completion frequency.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont, ImageOps


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--delivery', type=Path, required=True)
    p.add_argument('--port', type=int, default=8791)
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    root, delivery = args.run.resolve(), args.delivery.resolve()
    delivery.mkdir(parents=True, exist_ok=True)
    video = delivery / 'mapping_live.mp4'
    if root.exists() or video.exists():
        raise FileExistsError('Use a new run and delivery directory')
    command = ['bash', 'DynaNav/warehouse_loop/run_realtime.sh',
               '--selection-mode', 'instances', '--landmark-variant', 'original',
               '--headless', '--port', str(args.port), '--output', str(root)]
    (delivery / 'command.json').write_text(json.dumps(command, indent=2))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    fps, count, start = 5, 0, None
    launched = time.monotonic()
    encoder = None
    latest, buffer, stream = None, '', None
    cached = {}
    log = (delivery / 'launcher.log').open('w')
    process = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
    timeline = (delivery / 'video-timeline.jsonl').open('w')
    error = None
    try:
        while True:
            now = time.time()
            framefile = root / 'capture/raw/frames.jsonl'
            if stream is None and framefile.exists():
                stream = framefile.open()
            if stream:
                buffer += stream.read()
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line:
                        latest = json.loads(line)
            if latest and start is None:
                start = now
                encoder = subprocess.Popen([
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
                    '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '1920x1200',
                    '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
                    '-threads', '2', '-preset', 'ultrafast', '-crf', '23',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
            if latest:
                status = read_json(root / 'mapping/status.json')
                inspection = read_json(root / 'mapping/inspection.json')
                canvas = Image.new('RGB', (1920, 1200), '#172330')
                d = ImageDraw.Draw(canvas)
                d.text((20, 12), 'ISAAC SIM 5.0 | ORIGINAL WAREHOUSE | LIVE SEMANTIC MAPPING', font=font, fill='white')
                d.text((20, 48), f"Wall elapsed {now-start:.1f}s | RGB sim time {latest['t_obs']:.2f}s | "
                       f"Geometry frames {status.get('geometry_frames', 0)} | All landmarks tentative", font=small, fill='#b8d8ef')
                paths = [root / 'capture/raw' / latest['rgb'], root / 'mapping/map.png',
                         root / 'mapping/detections.png', root / 'mapping/semantic.png']
                labels = ['Isaac on-board RGB (latest committed capture)', 'Accumulating map (observed geometry + landmark view)',
                          'YOLO-E candidates (asynchronously published)', 'Map-admitted instances (asynchronously published)']
                versions = []
                for i, (path, label) in enumerate(zip(paths, labels)):
                    x, y = 20 + (i % 2)*960, 90 + (i // 2)*530
                    d.text((x, y), label, font=small, fill='white')
                    try:
                        stamp = path.stat().st_mtime_ns
                        if i not in cached or cached[i][0] != stamp or cached[i][2] != path:
                            with Image.open(path) as im:
                                panel = ImageOps.contain(im.convert('RGB'), (920, 472))
                            cached[i] = (stamp, panel, path)
                        panel = cached[i][1]
                        canvas.paste(panel, (x+(920-panel.width)//2, y+32+(472-panel.height)//2))
                        versions.append(stamp)
                    except (OSError, ValueError):
                        d.text((x+20, y+150), 'Waiting for publication', font=font, fill='#aab8c8')
                        versions.append(None)
                d.text((20, 1155), f"RGB: {Path(latest['rgb']).stem} | Detection source: {inspection.get('image_id', 'pending')} "
                       f"| OCR source: {inspection.get('text_source_image_id', 'pending')}", font=small, fill='#b8d8ef')
                # Duplicate held display frames to retain the measured wall timeline.
                target = int((now-start)*fps)+1
                raw = canvas.tobytes()
                while count < target:
                    encoder.stdin.write(raw)
                    timeline.write(json.dumps(dict(video_frame=count, wall_sample=now,
                        rgb=latest['rgb'], rgb_t_obs=latest['t_obs'],
                        detection_source=inspection.get('image_id'), ocr_source=inspection.get('text_source_image_id'),
                        panel_mtime_ns=versions, geometry_frames=status.get('geometry_frames'))) + '\n')
                    count += 1
                canvas.save(delivery / 'last_live_frame.jpg', quality=90)
            if process.poll() is not None:
                break
            if start is None and time.monotonic() - launched > 1200:
                raise TimeoutError('No camera frame within startup timeout')
            time.sleep(.15 if latest else .5)
        if process.returncode:
            raise RuntimeError(f'Pipeline exited {process.returncode}; see launcher.log')
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=60)
        if encoder:
            encoder.stdin.close()
            encoder.wait(timeout=60)
        timeline.close()
        log.close()
        if stream:
            stream.close()
        (delivery / 'recording.json').write_text(json.dumps(dict(
            run=str(root), fps=fps, frames=count, duration_s=count/fps, first_wall=start,
            pipeline_exit=process.returncode, encoder_exit=encoder.returncode if encoder else None,
            error=error, scope='Actual live panel sampling; held frames included; not a desktop screen recording'), indent=2))


if __name__ == '__main__':
    main()
