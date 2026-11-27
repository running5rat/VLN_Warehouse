"""Export a bounded Codex review for Pages. No credentials, databases or raw captures."""
import argparse
import html
import json
from pathlib import Path
import shutil


def export(source, destination):
    source=Path(source).resolve();dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=False)
    exp=json.loads((source/'experiment.json').read_text());summary=json.loads((source/'summary.json').read_text())['codex-session']
    def dump(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    def esc(s):return html.escape(str(s))
    (dest/'system.txt').write_text((source/'system.txt').read_text())
    dump(dest/'summary.json',summary)
    dump(dest/'provenance.json',json.loads((source/'codex-session/provenance.json').read_text()))
    shutil.copy2(source/'route-history.json',dest/'route-history.json')
    css='body{max-width:1400px;margin:24px auto;padding:0 16px;background:#18232e;color:#eee;font:16px system-ui}a{color:#9df}pre{white-space:pre-wrap;overflow-wrap:anywhere}video{width:100%;max-width:1100px}.images{display:flex;gap:8px;flex-wrap:wrap}.images figure{width:31%;margin:0}.images img{width:100%}section{border-top:1px solid #777;margin-top:28px;padding-top:20px}td,th{padding:7px;text-align:left}table{border-collapse:collapse}td{border-bottom:1px solid #567}'
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>仓库两任务 · Codex会话标注</title><style>'+css+'</style><h1>仓库两任务：Codex会话标注审核</h1>',
        '<p>2026-11-13 发布快照。37个关键帧；两个任务的全部录制过程。<strong>离线回放，新预测没有驱动车辆。</strong>当前会话已知旧演示，属于非盲Oracle辅助标注，不是独立GPT API评测。</p>',
        '<p>输入：当前RGB、同帧检测叠图、此前语义地图、Previous Route/Local与相对运动。第二任务只给目标，不预置路径。原地转向时可能返回空像素点；空点不编码旋转命令。蓝色地图轨迹为历史运动。</p>',
        '<p>37帧格式有效；24帧非空投影通过、13帧空点。24帧单帧控制诊断均拒绝运动（占据或未知区域）。投影通过不等于路线或控制通过，人工批准数0。</p>',
        '<p><a href="system.txt">标注SYSTEM</a> · <a href="summary.json">完整统计</a> · <a href="route-history.json">Route递归记录</a> · <a href="provenance.json">来源与局限</a> · <a href="../index.html">原语义建图演示</a></p>']
    for task,label in [('mapping','任务一：环绕与建图'),('target','任务二：前往04/5R')]:
        shutil.copy2(source/'codex-session'/f'{task}.mp4',dest/f'{task}.mp4')
        parts += [f'<h2>{label}</h2><p>初始指令：{esc(exp["instructions"][task])}</p>',f'<video controls preload="metadata" playsinline src="{task}.mp4"></video><p><a href="{task}.mp4">打开或下载原视频</a></p>']
    parts.append('<h2>逐帧审核</h2><p>点图片查看原尺寸。视频以源时刻同步结果，不表示实际推理延迟。</p><p>'+ ' · '.join(f'<a href="#f{r["frame"]}">{r["frame"]}</a>' for r in exp['selected'])+'</p>')
    for row in exp['selected']:
        n=row['frame'];key=f'{n:08d}';inp=source/'inputs'/key;out=source/'codex-session'/key;folder=dest/'frames'/key;folder.mkdir(parents=True)
        original=json.loads((out/'input.json').read_text());parsed=json.loads((out/'parsed.json').read_text());audit=json.loads((inp/'audit.json').read_text());result=json.loads((out/'result.json').read_text())
        public_input={k:original[k] for k in ['user_text','system_text','original_instruction_audit_only','map_cutoff']}
        public_input['images']=['rgb.png','detections.png','map.png'];public_input['source']={k:original['source'][k] for k in ['image_id','t_obs','rgb_hash','depth_hash','K','T_body_camera','T_odom_body','odom_epoch']}
        public_input['images_sha256']=audit['images'];public_input['previous']=original.get('previous')
        dump(folder/'input.json',public_input)
        for name in ['rgb.png','detections.png','map.png']:shutil.copy2(inp/name,folder/name)
        for name in ['raw.txt','parsed.json','result.json','projection.json','waypoints.png','projection-3d.png','controller-diagnostic.json']:
            if (out/name).exists():shutil.copy2(out/name,folder/name)
        prefix=f'frames/{key}/';parts += [f'<section id="f{n}"><h2>{row["task"]} · 帧{n} · {row["t_obs"]:.2f}s</h2><div class="images">']
        for name,title in [('rgb.png','当前原始RGB'),('detections.png','同帧检测输入'),('map.png','推理前历史语义地图')]:parts.append(f'<figure><a href="{prefix+name}"><img loading="lazy" src="{prefix+name}" alt="{title}"></a><figcaption>{title}</figcaption></figure>')
        parts += ['</div><details><summary>完整标注输入</summary><pre>'+esc(json.dumps(public_input,ensure_ascii=False,indent=2))+'</pre></details>',
            '<p><b>Previous Route:</b> '+esc((original.get('previous') or {}).get('route',original['original_instruction_audit_only']))+'</p>',
            '<p><b>Generated Route:</b> '+esc(parsed['route'])+'</p>',
            '<pre>'+esc((out/'raw.txt').read_text())+'</pre>',
            f'<p>状态：{esc(result["status"])}；投影：{esc(result["projection_status"])}。实际执行：未执行；人工审核：待审核。</p><div class="images">']
        for name,title in [('waypoints.png','源RGB像素点与工程曲线'),('projection-3d.png','深度投影诊断')]:
            if (folder/name).exists():parts.append(f'<figure><a href="{prefix+name}"><img loading="lazy" src="{prefix+name}" alt="{title}"></a><figcaption>{title}</figcaption></figure>')
        parts.append('</div><p>'+ ' · '.join(f'<a href="{prefix+name}">{name}</a>' for name in ['input.json','raw.txt','parsed.json','projection.json','controller-diagnostic.json','result.json'] if (folder/name).exists())+'</p></section>')
    (dest/'index.html').write_text('\n'.join(parts)+'</html>')
    dump(dest/'publication.json',dict(run='codex-goal-only-20261112-v2',published_snapshot='2026-11-13',keyframes=37,external_api_calls=0,human_approved=0,
        local_absolute_paths_removed=True,scope='bounded review images, model fields, source transforms and projected predictions; no raw depth/database/config/credential export'))
    print('Exported',len(exp['selected']),'frames;',sum(p.stat().st_size for p in dest.rglob('*') if p.is_file())//1024**2,'MiB')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args();export(a.source,a.output)
