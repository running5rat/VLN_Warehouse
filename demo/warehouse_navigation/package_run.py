"""Package completed, real Codex/Isaac demo evidence without claiming VLA quality."""
import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import numpy as np


def run(root,out,archive_map=False):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);nav=root/'navigation'
    report=json.loads((root/'report.json').read_text());target=json.loads((nav/'target-task.json').read_text()) if (nav/'target-task.json').exists() else None
    decisions=[json.loads(s) for s in (nav/'decisions.jsonl').read_text().splitlines()]
    report['runtime_report_provider']=report.get('provider')
    report['providers']=sorted(set(d.get('provider','unknown') for d in decisions))
    report['provider']=report['providers'][0] if len(report['providers'])==1 else 'mixed'
    frames=[json.loads(s) for s in (root/'capture/raw/frames.jsonl').read_text().splitlines()]
    trail=np.asarray([np.asarray(f['T_odom_body'])[:2,3] for f in frames])
    metrics=[json.loads(s) for s in (nav/'metrics.jsonl').read_text().splitlines()]
    report.update(actual_perception_results=len(metrics),source_capture_frames=len(frames),
        accepted_responses=sum(bool(d['accepted']) for d in decisions),rejected_responses=sum(not d['accepted'] for d in decisions),
        full_raw_output=str(root.resolve()),target_task=target,
        note='Codex-session supplied navigation decisions; fixed Codex category policy, real YOLO-E/OCR. Not trained VLA inference or navigation benchmark.')
    report['post_task_unconsumed_capture_frames']=len(frames)-report['geometry_frames']
    report['perception_timing_seconds']={k:dict(n=len(metrics),p50=float(np.percentile([m[k] for m in metrics],50)),p95=float(np.percentile([m[k] for m in metrics],95))) for k in ['segmentation_s','ocr_s','source_lag_s']}
    report['static_anchor_count']=sum(bool(x.get('static_confirmed')) for x in report['landmarks'])
    report['human_review']='pending; assistant visual inspection only'
    report['map_directory']=str((nav/'map').resolve())
    final=json.loads((nav/'task-finished.json').read_text()) if (nav/'task-finished.json').exists() else {}
    report['target_identity_at_finish']=final.get('target',{}).get('identity_status')
    report['target_binding_policy']='Task-start UID retained; later OCR conflicts remain visible, not overwritten by requested text'
    if not (out/'REPORT.md').exists():
        (out/'REPORT.md').write_text('# Demo report\n\nCodex-reviewed waypoint decisions, real Isaac physics and RGB-D/YOLO-E/OCR. '
            'No trained VLA inference. See report.json and every accepted/rejected response below. '
            'Map landmarks remain tentative; identity correctness needs human review. '
            'The video samples published panels; video-timeline.jsonl records actual wall-clock times.\n')
    for name in ['loop-complete.json','target-task.json','task-finished.json','landmarks.json','decisions.jsonl','control-metrics.jsonl','stopped-evidence.json','perception-provenance.json','target-start-rgb.png','target-start-map.png','target-final-rgb.png']:
        if (nav/name).exists():shutil.copy2(nav/name,out/name)
    shutil.copy2(nav/'map.png',out/'live-final-map.png')
    shutil.copy2(nav/'rgb.png',out/'final-rgb.png')
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Focused plot of actual odom and observed landmarks; no hidden scene data.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,7));ax.plot(trail[:,0],trail[:,1],c='#2873be',label='Measured robot trajectory')
    for i,lm in enumerate(report['landmarks']):
        if lm.get('identity_text') and '/' in lm['identity_text']:
            x,y=lm['position'][:2];ax.scatter(x,y,marker='s',c='#df8e13');ax.annotate(lm['identity_text'],(x,y),xytext=(5,8),textcoords='offset points')
    if target:
        x,y=target['target']['position'][:2];ax.scatter(x,y,s=180,facecolors='none',edgecolors='red',label='Task 2 target, from map')
        ax.annotate(str(target['target'].get('identity_text'))+' (task-start binding; final '+str(report['target_identity_at_finish'])+')',(x,y),xytext=(5,-22),textcoords='offset points',color='red',fontsize=8)
    ax.scatter(*trail[0],c='green',label='Start');ax.scatter(*trail[-1],c='red',label='Final robot position')
    ax.set(xlabel='Odom x (m)',ylabel='Odom y (m)',title='Observed numbered landmarks and actual motion (tentative landmarks)')
    ax.axis('equal');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'trajectory-and-landmarks.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(11,7));ax=fig.add_subplot(projection='3d')
    ax.plot(trail[:,0],trail[:,1],np.zeros(len(trail)),label='Measured robot motion')
    for lm in report['landmarks']:
        with np.load(nav/'map'/lm['geometry']) as data:points=data['points'][::max(1,len(data['points'])//300)]
        ax.scatter(*points.T,s=2,alpha=.5)
        if lm.get('identity_text') and '/' in lm['identity_text']:ax.text(*lm['position'],lm['identity_text'],fontsize=7)
    ax.set(xlabel='Odom x (m)',ylabel='Odom y (m)',zlabel='z (m)',title='Real projected instance surfaces; candidate semantics, no static anchors')
    fig.tight_layout();fig.savefig(out/'semantic-3d.png',dpi=140);plt.close(fig)
    cards=[]
    for d in decisions:
        folder=out/'decisions'/d['request_id'];folder.mkdir(parents=True,exist_ok=True)
        old=nav/'requests'/d['request_id']
        for n in ['input.json','response.json','raw.txt','map-at-submit.png']:
            if (old/n).exists():shutil.copy2(old/n,folder/n)
        inp=json.loads((old/'input.json').read_text());shutil.copy2(inp['source']['rgb'],folder/'rgb.png')
        curve=nav/f"curve-{d['request_id']}.npz"
        if curve.exists():shutil.copy2(curve,folder/'curve.npz')
        frozen=f'<img width="480" src="decisions/{d["request_id"]}/map-at-submit.png">' if (folder/'map-at-submit.png').exists() else '<p>此请求未冻结地图图片；输入JSON保留当时地标表。不能用最终地图冒充当时输入。</p>'
        cards.append(f'<section><h3>{html.escape(d["request_id"])} — accepted={d["accepted"]}</h3><a href="decisions/{d["request_id"]}/input.json">输入与来源</a> · <a href="decisions/{d["request_id"]}/raw.txt">原文</a><br><img width="480" src="decisions/{d["request_id"]}/rgb.png">'+frozen+f'<pre>{html.escape(json.dumps(d,ensure_ascii=False,indent=2))}</pre></section>')
    if not (out/'map-local').exists():(out/'map-local').symlink_to((nav/'map').resolve(),target_is_directory=True)
    (out/'MAP_LOCATION.txt').write_text(str((nav/'map').resolve())+'\nComplete SQLite + immutable geometry arrays; map-local is a local link, not portable.\n')
    if archive_map:
        with tarfile.open(out/'map.tar.gz','w:gz',compresslevel=1) as tar:tar.add(nav/'map',arcname='map')
    if (out/'live_demo.mp4').exists() and (not (out/'demo_4x.mp4').exists() or (out/'demo_4x.mp4').stat().st_mtime<(out/'live_demo.mp4').stat().st_mtime):
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(out/'live_demo.mp4'),'-vf','setpts=PTS/4','-r','15','-c:v','libx264','-preset','fast','-crf','23','-threads','2','-an',str(out/'demo_4x.mp4')],check=True)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Warehouse Codex navigation demo</title><style>body{font:17px sans-serif;max-width:1200px;margin:auto;background:#18222c;color:#eef}a{color:#8df}pre{white-space:pre-wrap}video{width:100%}</style><h1>环绕建图 → 视野外编号导航</h1><p>人工标注时的原始仓库。Codex代答＋waypoint曲线＋控制器；不是已训练VLA的能力验证。</p><video controls src="live_demo.mp4"></video><p>上方为原速采样录像；<a href="demo_4x.mp4">4倍速概览</a> · <a href="REPORT.md">说明与限制</a> · <a href="report.json">报告</a> · <a href="map-local/">完整本地地图（本机链接）</a> · <a href="landmarks.json">全部地标</a></p><h2>任务2开始与结束</h2><img width="49%" src="target-start-rgb.png"><img width="49%" src="target-final-rgb.png"><img width="100%" src="trajectory-and-landmarks.png"><pre>'+html.escape(json.dumps({k:v for k,v in report.items() if k not in ('landmarks','target_task')},ensure_ascii=False,indent=2))+'</pre>'+''.join(cards))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--archive-map',action='store_true');a=p.parse_args();run(a.root,a.output,a.archive_map)
