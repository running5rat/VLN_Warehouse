"""Run a frozen extension batch once: perception, paid annotation, then review.

No automatic retry, budget increase, alternate model, or replacement episode.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .core import read_json,write_json
from .r2r_pilot import verify
from .annotation_batch_summary import summarize
from ..schema import file_sha256


def progress(root,phase,child=None):
    info=read_json(root/'pilot.json');ready=0;perception_frames=0
    for row in info['episodes']:
        p=Path(row['folder'])/'perception/manifest.json'
        if p.exists():
            try:
                d=read_json(p);ready+=bool(d['complete']);perception_frames+=len(d['records'])
            except json.JSONDecodeError:pass
    state=summarize(root,phase)
    s=state['summary']
    value=dict(phase=phase,updated_unix=time.time(),new_tasks=len(info['episodes']),planned_frames=info['planned_requests'],
        previously_annotated_unique=info['previous_unique_count'],target_unique=info['total_target'],
        perception_ready_tasks=ready,perception_ready_frames=perception_frames,
        returned=s['returned'],requested=s['requested'],rule_valid=s['rule_valid'],
        newly_fully_returned_tasks=s['tasks_all_responses'],
        cumulative_fully_returned_tasks=info['previous_unique_count']+s['tasks_all_responses'],
        cost_cny_estimate=s['cost_cny_estimate'],human_approved=0,pid=os.getpid(),child_pid=child.pid if child else None)
    write_json(root/'progress.json',value)
    html='<meta charset="utf-8"><meta http-equiv="refresh" content="30"><h1>R2R 标注扩至总计300任务</h1><p>先准备分割/OCR，再调用教师。准备期间没有响应是正常状态，不计作完成。</p><p><a href="statistics.html">逐任务统计</a> · <a href="index.html">审核报告</a></p><pre>'+json.dumps(value,ensure_ascii=False,indent=2)+'</pre>'
    tmp=root/'progress.html.tmp';tmp.write_text(html);tmp.replace(root/'progress.html')
    return state


def run(root,credentials,perception_python):
    root=Path(root).resolve();repo=Path.cwd();lock=(root/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    info=verify(root)
    # Neither an interrupted preparation nor paid run is silently repeated.
    if any((Path(r['folder'])/'perception').exists() or (Path(r['folder'])/'plus').exists() for r in info['episodes']):
        raise ValueError('Existing execution artifacts: inspect and authorize explicit recovery instead of rerunning')
    write_json(root/'execution-code-hashes.json',{str(p.relative_to(repo)):file_sha256(p) for p in (repo/'ticvla/warehouse/dataset_tools').glob('*.py')})
    (root/'index.html').write_text('<meta charset="utf-8"><h1>R2R 300任务扩展批次正在执行</h1><p><a href="progress.html">准备与标注进度</a> · <a href="statistics.html">任务统计</a></p>')
    def execute(phase,module,args,logname):
        if shutil.disk_usage(root).free<40*1024**3:raise RuntimeError('Disk reserve below 40 GiB')
        with (root/logname).open('x') as log:
            child=subprocess.Popen([sys.executable,'-u','-m',module,*args],cwd=repo,stdout=log,stderr=subprocess.STDOUT)
            write_json(root/(phase+'-process.json'),dict(pid=child.pid,started_unix=time.time()))
            while child.poll() is None:
                progress(root,phase,child);time.sleep(20)
            if child.returncode:raise RuntimeError(f'{phase} failed ({child.returncode}); no automatic retry')
    try:
        execute('perception_preparation','ticvla.warehouse.dataset_tools.r2r_batch',
            ['prepare','--root',str(root),'--python',perception_python],'perception.log')
        execute('annotation','ticvla.warehouse.dataset_tools.r2r_batch',
            ['run','--root',str(root),'--credentials',credentials,'--workers','3'],'annotation.log')
        state=progress(root,'annotation_finished')
        if state['summary']['returned']!=info['planned_requests']:
            progress(root,'stopped_partial');return
        execute('review_generation','ticvla.warehouse.dataset_tools.r2r_batch',['report','--root',str(root)],'review.log')
        execute('review_audit','ticvla.warehouse.dataset_tools.r2r_pilot_review',['--root',str(root),'--audit'],'review-audit.log')
        state=summarize(root,'completed_unreviewed',deep=True)
        progress(root,'completed_unreviewed')
        with (root/'REPORT.md').open('a') as f:
            f.write('\n\n## 去重后的累计覆盖\n\n已有124任务，本轮新增176任务，共300任务。全部为机器候选，人工批准数为0。\n\n```json\n'+json.dumps(state['summary'],ensure_ascii=False,indent=2)+'\n```\n')
        write_json(root/'completion.json',dict(previous_unique_tasks=124,new_tasks=176,total_unique_tasks=300,
            new_frames=info['planned_requests'],all_responses_received=True,human_approved=0,summary=state['summary']))
    except Exception as exc:
        write_json(root/'fatal-error.json',dict(error=str(exc),retry_issued=False))
        progress(root,'infrastructure_failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--credentials',default='/home/xinglinlu/.config/llm/credentials.env')
    p.add_argument('--perception-python',default='/data/ticvla/envs/warehouse-realtime/bin/python')
    a=p.parse_args();run(a.root,a.credentials,a.perception_python)
