"""Package measured demo artifacts without changing the live map or predictions."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tarfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--delivery', type=Path, required=True)
    args = p.parse_args()
    root, dest = args.run.resolve(), args.delivery.resolve()
    repo = Path(__file__).resolve().parents[2]
    report = load(root/'report.json')
    if not report['loop_complete']:
        raise ValueError('Incomplete circuit; retain artifacts but do not package as completed')
    from ticvla.warehouse.run_map_postprocess import run as postprocess
    post = postprocess(root/'mapping/map', dest/'postprocess')
    for name in ('map.png', 'semantic.png', 'detections.png', 'ocr.png', 'landmark-view.json'):
        shutil.copy2(root/'mapping'/name, dest/name)
    evidence = dest/'evidence'
    evidence.mkdir()
    for relative in ('report.json', 'capture/capture-report.json', 'capture/performance-report.json',
                     'capture/raw/episode.json', 'capture/raw/frames.jsonl',
                     'mapping/realtime-report.json', 'mapping/metrics.jsonl',
                     'mapping/qa-provenance.json', 'mapping/seg-provenance.json', 'mapping/ocr-provenance.json',
                     'mapping/inspection.json', 'mapping/status.json', 'mapping/report.json',
                     'mapping.log', 'isaac.log', 'gpu-metrics.jsonl'):
        target = evidence/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/relative, target)
    shutil.copytree(root/'mapping/qa', evidence/'qa')
    scene=dest/'scene';scene.mkdir()
    for name in ('capture.json','dynanav-warehouse.yaml','evaluation-origin.json',
                 'warehouse-labeled.usda','warehouse-labels-only.usda','shelf-catalog.json'):
        shutil.copy2(root/'capture'/name,scene/name)
    shutil.copytree(root/'capture/textures',scene/'textures')
    shutil.copytree(root/'capture/overview',scene/'overview')
    # Full frame records include source binding and all unsuccessful outcomes.
    (evidence/'frame-records').mkdir()
    for f in sorted((root/'mapping/frames').glob('*/frame.json')):
        shutil.copy2(f, evidence/'frame-records'/f'{f.parent.name}.json')
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-framerate', '5',
        '-i', str(root/'capture/raw/rgb/%08d.png'), '-c:v', 'libx264', '-threads', '2',
        '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        str(dest/'isaac_camera.mp4')], check=True)
    print('Camera video complete', flush=True)
    # Preserve every byte of the database/immutable arrays in a compressed archive.
    with tarfile.open(dest/'map.tar.gz', 'w:gz', compresslevel=3) as tar:
        tar.add(root/'mapping/map', arcname='map')
    with tarfile.open(dest/'map.tar.gz', 'r:gz') as tar:
        with tar.extractfile('map/map.sqlite') as f:
            archived_db_sha = hashlib.sha256(f.read()).hexdigest()
    if archived_db_sha != sha(root/'mapping/map/map.sqlite'):
        raise ValueError('Archived database checksum mismatch')
    print('Map archive complete', flush=True)
    view = load(dest/'postprocess/landmark-view.json')
    landmarks = view['landmarks']
    visible = [r for r in landmarks if r['landmark_view']['visible']]
    with (dest/'landmarks.csv').open('w', newline='') as f:
        fields = ['landmark_uid','category','identity_text','identity_status','status',
                  'decision','x_m','y_m','z_m','source_frames']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in landmarks:
            w.writerow({**{k:r.get(k) for k in fields[:5]}, 'decision':r['landmark_view']['decision'],
                'x_m':r['position'][0], 'y_m':r['position'][1], 'z_m':r['position'][2],
                'source_frames':len(r['observation_ids'])})
    db = sqlite3.connect((root/'mapping/map/map.sqlite').as_uri()+'?mode=ro', uri=True)
    observations = [json.loads(r[0]) for r in db.execute('SELECT payload FROM observations ORDER BY id')]
    db.close()
    all_geometry = []
    for obs in observations:
        with np.load(root/'mapping/map'/obs['geometry']) as a:
            pts = a['points']
            pts = pts[(pts[:,2]>.15)&(pts[:,2]<3.5)]
            all_geometry.append(pts[::max(1,len(pts)//400)])
    geometry = np.concatenate(all_geometry)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13,9)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*geometry.T, s=.15, c='#909ba8', alpha=.15)
    arrays = {'observed_geometry_sample':geometry}
    for i,r in enumerate(visible):
        with np.load(root/'mapping/map'/r['geometry']) as a: pts=a['points']
        arrays['landmark_'+r['landmark_uid']] = pts
        color = plt.get_cmap('tab20')(i%20)
        ax.scatter(*pts[::max(1,len(pts)//1000)].T, s=2, color=color)
        ax.text(*r['position'], r['identity_text'] or r['category'], fontsize=6)
    trajectory = np.array([np.array(o['T_odom_body'])[:3,3] for o in observations])
    ax.plot(*trajectory.T, c='#1967c9', linewidth=2)
    ax.set(xlabel='Odom X (m)', ylabel='Odom Y (m)', zlabel='Z (m)',
           title='Original scene | observed geometry and tentative semantic surfaces')
    ax.set_box_aspect((2,2,.6)); ax.view_init(elev=50, azim=-110)
    fig.tight_layout(); fig.savefig(dest/'semantic_3d.png', dpi=160); plt.close(fig)
    np.savez_compressed(dest/'semantic_points.npz', trajectory=trajectory, **arrays)
    # Pick one actually admitted text-bearing frame per sign landmark; preserve raw evidence.
    samples = dest/'samples'; samples.mkdir()
    sample_files, covered = [], set()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    for f in sorted((root/'mapping/frames').glob('*/frame.json')):
        fr = load(f)
        good = [r for r in fr['rows'] if r['result']['status']=='accepted' and r['category'] in ('sign','label')]
        fresh = {r['result'].get('landmark_uid') for r in good} - covered
        if not fresh or not any(x.get('texts') for x in fr['instances']): continue
        covered |= fresh
        example = samples/f.parent.name
        shutil.copytree(f.parent, example)
        src = Path(fr['source']['rgb'])
        shutil.copy2(src, example/'rgb.png')
        rgb = np.asarray(Image.open(src).convert('RGB')).copy()
        for r in fr['rows']:
            if r['result']['status'] != 'accepted': continue
            maskfile = f.parent/f"mask-{r['instance_index']:03d}.npz"
            if maskfile.exists():
                with np.load(maskfile) as a: mask=a['mask'].astype(bool)
                rgb[mask] = .6*rgb[mask] + .4*np.array([30,210,100])
        im = Image.fromarray(rgb); d=ImageDraw.Draw(im)
        for i, inst in enumerate(fr['instances']):
            box=inst['bbox']; d.rectangle(box, outline='yellow', width=2)
            text=' | '.join(t['text'] for t in inst.get('texts', []))
            d.text((box[0], max(0, box[1]-20)), f"d{i} {inst['category']} {text}", font=font, fill='yellow', stroke_width=1, stroke_fill='black')
            for t in inst.get('texts', []):
                poly=[tuple(x) for x in t['polygon']]; d.line(poly+poly[:1], fill='cyan', width=2)
        im.save(example/'overlay.png')
        sample_files.append(str((example/'overlay.png').relative_to(dest)))
    tracked = ['DynaNav/warehouse_loop/run_realtime.sh','DynaNav/warehouse_loop/live.py',
               'DynaNav/warehouse_loop/live_mapping.py','DynaNav/warehouse_loop/realtime_mapping.py',
               'DynaNav/warehouse_loop/instance_mapping.py','ticvla/warehouse/mapping',
               'ticvla/warehouse/perception','ticvla/warehouse/instance_io.py']
    diff=subprocess.check_output(['git','diff','--name-only','7b97f85622de91674327166f6f84d5390059a306','--',*tracked], cwd=repo, text=True)
    summary = dict(source_commit='7b97f85622de91674327166f6f84d5390059a306',
        execution_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        mapping_core_diff=diff.splitlines(), source_run=str(root), scene='original',
        loop_complete=report['loop_complete'], distance_m=report['distance_m'], capture_frames=report['capture_frames'],
        map_archive_database_sha256=archived_db_sha,
        raw_landmarks=len(landmarks), raw_categories=dict(Counter(r['category'] for r in landmarks)),
        visible_landmarks=len(visible), visible_categories=dict(Counter(r['category'] for r in visible)),
        sign_text_candidates=[dict(uid=r['landmark_uid'], text=r['identity_text'], status=r['identity_status'],
            position=r['position']) for r in visible if r['category'] in ('sign','label') and r['identity_text']],
        postprocess=post, realtime={k:v for k,v in report['realtime'].items() if k!='store'}, qualitative_samples=sample_files,
        limitations=['No independent mask/number ground-truth scoring in this demo.',
            'All semantic objects are tentative; OCR text is not verified unique identity.',
            'Map archive is complete for loading geometry; reprocessing RGB-D needs the original source run.',
            'Camera video uses 5 Hz simulation samples. Live composite uses wall time and asynchronous panels.',
            'Recording overhead is included in measured performance.'])
    save(dest/'summary.json', summary)
    shutil.copy2(dest/'postprocess/after.png',dest/'semantic_bev.png')
    page=['<!doctype html><meta charset="utf-8"><title>原始仓库语义建图 Demo</title>',
          '<style>body{max-width:1400px;margin:30px auto;background:#15202c;color:#e8f1fa;font:17px sans-serif;padding:20px}a{color:#8dcfff}video,img{max-width:100%}pre{white-space:pre-wrap}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}</style>',
          '<h1>Isaac Sim 5.0 · 原始仓库语义建图</h1>',
          '<p>真实原场景重新绕环：InternVL 实例筛选 → YOLO-E mask → RGB-D 投影 → 异步 PP-OCRv5 文字绑定 → 持久地图。没有导航问答或 Flow 控制。</p>',
          f'<p>完成一圈 {summary["distance_m"]:.2f} m；采集 {summary["capture_frames"]} 帧；原始地标 {len(landmarks)} 个，后处理显示 {len(visible)} 个。全部为待确认候选，未测量识别准确率。</p>',
          '<p><a href="summary.json">本次统计</a> · <a href="landmarks.csv">全部地标与位置</a> · <a href="map.tar.gz">完整地图归档</a> · <a href="postprocess/index.html">后处理证据</a> · <a href="evidence/report.json">运行报告</a></p>',
          '<h2>运动期间实时发布录像</h2><p>左上相机，右上地图，左下全部检测，右下入图 mask。各面板异步；底部标明源帧。保留等待与结果滞后，不把视频帧率算作模型完成频率。</p>',
          '<video controls preload="metadata" src="mapping_live.mp4" poster="last_live_frame.jpg"></video>',
          '<h2>车载相机完整一圈</h2><p>Isaac 原始 RGB 按仿真采样时间 5 Hz 编码；非桌面录屏。</p><video controls preload="metadata" src="isaac_camera.mp4"></video>',
          '<h2>最终地图</h2><div class="grid"><img src="semantic_bev.png"><img src="semantic_3d.png"></div>',
          '<h2>源帧、mask 和 OCR 文字证据</h2><p>黄色为检测框，绿色为被接纳的 mask，青色为文字区域。图片上的文字是原始读数，最终采用或拒绝情况见后处理报告。</p><div class="grid">']
    page += [f'<a href="{html.escape(s)}"><img src="{html.escape(s)}"></a>' for s in sample_files]
    page += ['</div><h2>实测时延与频率</h2><pre>'+html.escape(json.dumps(summary['realtime'],ensure_ascii=False,indent=2))+'</pre>']
    (dest/'index.html').write_text('\n'.join(page))
    (repo/'demo/semantic_mapping/index.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Semantic mapping demo</title>'
        f'<h1>语义建图 Demo</h1><p><a href="{html.escape(dest.name)}/index.html">打开原始场景视频、地图与报告</a></p>'
        '<p>运行说明见 README.md。</p>')
    manifest={str(f.relative_to(dest)):dict(bytes=f.stat().st_size, sha256=sha(f))
              for f in sorted(dest.rglob('*')) if f.is_file() and f.name!='manifest.json'}
    save(dest/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('postprocess','realtime')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
