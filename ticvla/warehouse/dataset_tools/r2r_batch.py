"""Frozen multi-building R2R annotation batches; no training or auto-approval."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import numpy as np
from .core import read_json, write_json, validate_v2
from .oracle_comparison import teacher_sample
from .semantic_annotation import write_bundle
from .reference_schedule import audit_coverage
from .r2r_pilot import verify
from ..schema import file_sha256


def select_tasks(selection, count=12, seed=42):
    groups={}
    for row in selection:
        if row['source']=='r2r' and row['role']=='train':
            groups.setdefault(row['scene_id'],[]).append(row)
    if not 1<=count<=len(groups):raise ValueError('Count must fit distinct available buildings')
    rng=random.Random(seed);scenes=sorted(groups);rng.shuffle(scenes)
    return [rng.choice(sorted(groups[s],key=lambda x:x['uid'])) for s in scenes[:count]]


def remaining_tasks(selection, exclude_runs=()):
    """Freeze all remaining train UIDs; prior attempted tasks are not auto-retried."""
    excluded=set()
    for run in exclude_runs:
        excluded.update(e['uid'] for e in read_json(Path(run)/'pilot.json')['episodes'])
    rows=[r for r in selection if r['source']=='r2r' and r['role']=='train' and r['uid'] not in excluded]
    if len({r['uid'] for r in rows})!=len(rows):raise ValueError('Duplicate task UID in selection')
    if not rows:raise ValueError('No remaining training tasks')
    return sorted(rows,key=lambda r:r['uid'])


def initialize(dataset, selection, output, config, previous_pilot, count=12, all_remaining=False, exclude_run=()):
    dataset=Path(dataset).resolve();root=Path(output).resolve();old=Path(previous_pilot)
    source_report=dataset/'report.json'
    if source_report.exists():
        source=read_json(source_report)
        if not source['passed']:raise ValueError('Source preparation has not passed')
    elif dataset.name=='episodes' and (dataset.parent/'status.json').exists():
        source_report=dataset.parent/'status.json'
        source=read_json(source_report)
        if not source.get('all_geometry_prepared'):raise ValueError('Full source geometry preparation has not passed')
    else:
        raise ValueError('Missing verified source preparation report')
    selection_rows=read_json(selection)['selected']
    if exclude_run and not all_remaining:raise ValueError('Exclusions require --all-remaining')
    chosen=remaining_tasks(selection_rows,exclude_run) if all_remaining else select_tasks(selection_rows,count)
    # Freeze selection before perception or teacher results; never replace a failed task.
    prior=verify(old)
    root.mkdir(parents=True,exist_ok=False)
    for src,dest in [(Path(config),'teacher-config.json'),(old/'vocabulary.json','vocabulary.json'),
                     (old/'development-review.json','development-review.json')]:shutil.copy2(src,root/dest)
    config_lock=Path(config).with_suffix('.lock.json')
    lock=read_json(config_lock)
    if file_sha256(config)!=lock['config_sha256']:raise ValueError('Teacher config lock mismatch')
    shutil.copy2(config_lock,root/'teacher-config.lock.json')
    info=dict(protocol='r2r-training-pool-annotation-v1',dataset=str(dataset),episodes=[],development=[],
        selection_seed=None if all_remaining else 42,
        selection_rule='all remaining R2R train tasks sorted by UID' if all_remaining else 'shuffle sorted buildings; choose one sorted-UID task per building with seed42',
        excluded_runs=[dict(path=str(Path(p).resolve()),manifest_sha256=file_sha256(Path(p)/'pilot.json')) for p in exclude_run],
        selection_path=str(Path(selection).resolve()),selection_sha256=file_sha256(selection),
        frozen_selection=chosen,teacher_config_sha256=file_sha256(root/'teacher-config.json'),
        vocabulary_sha256=file_sha256(root/'vocabulary.json'),weights=prior['weights'],weights_sha256=prior['weights_sha256'],
        source_report_path=str(source_report),source_report_sha256=file_sha256(source_report),development_reference=str(old),
        optimizer_updates_permitted=False,human_approved=False,training=False,pose_source='ideal_simulator_odometry')
    write_json(root/'selection.json',info)
    for row in chosen:
        uid=row['uid'];samples=[];hashes={}
        for e in read_json(dataset/uid/'manifest.json')['records']:
            p=Path(e['path'])
            if file_sha256(p)!=e['sha256']:raise ValueError('Source record changed')
            r=read_json(p);validate_v2(r)
            if r['supervision']['text_state']!='unlabeled':raise ValueError('Expected unlabeled source')
            s=teacher_sample(r);s['provenance'].update(role='training_pool_annotation_candidate',optimizer_updates_permitted=False)
            samples.append(s);hashes[str(p)]=e['sha256']
        raw=dataset/uid/'capture/raw';meta=read_json(raw/'episode.json')
        if meta['pose_source']!='ideal_simulator_odometry':raise ValueError('Unexpected odometry')
        frames=[json.loads(x) for x in (raw/meta['frames_file']).read_text().splitlines()]
        indices=[s['provenance']['frame_index'] for s in samples]
        lines=[np.asarray(s['geometric_supervision']['reference_xy'])[s['geometric_supervision']['reference_mask']] for s in samples]
        coverage=audit_coverage([f['t_obs'] for f in frames],np.asarray([f['T_odom_body'] for f in frames]),
            [dict(index=i) for i in indices],lines)
        if coverage['exhausted_intervals'] or coverage['empty_reference_motion_intervals']:raise ValueError('Coverage gap; do not replace task')
        folder=root/'episodes'/uid;write_bundle(folder/'bundle',samples,source=uid,episode_review=True)
        write_json(folder/'coverage.json',coverage)
        info['episodes'].append(dict(uid=uid,scene_id=row['scene_id'],folder=str(folder),frames=len(samples),
            source_hashes=hashes,source_episode=str(raw/'episode.json'),source_episode_sha256=file_sha256(raw/'episode.json'),
            source_frames_sha256=file_sha256(raw/meta['frames_file']),source_indices=indices,instruction=meta['task']['instruction'],
            coverage_pass=True,bundle_sha256=file_sha256(folder/'bundle/manifest.json')))
    info['planned_requests']=sum(r['frames'] for r in info['episodes'])
    info['code_hashes']={str(p):file_sha256(p) for p in [Path(__file__),Path('ticvla/warehouse/dataset_tools/semantic_loop.py'),Path('ticvla/warehouse/dataset_tools/semantic_annotation.py'),Path('ticvla/warehouse/compact_input.py')]}
    write_json(root/'pilot.json',info)
    print(json.dumps(dict(tasks=len(chosen),keyframes=info['planned_requests'],uids=[r['uid'] for r in info['episodes']])),flush=True)


def prepare(root, python):
    root=Path(root);info=verify(root)
    for row in info['episodes']:
        folder=Path(row['folder'])
        with (folder/'prepare.log').open('w') as log:
            subprocess.run([python,'-u','-m','ticvla.warehouse.dataset_tools.semantic_loop','prepare',
                '--bundles',str(folder/'bundle'),'--output',str(folder/'perception'),
                '--weights',info['weights'],'--vocabulary',str(root/'vocabulary.json')],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('PREPARED',row['uid'],row['frames'],flush=True)


def run(root, credentials, workers=3):
    from .semantic_loop import run as run_episode, audit
    if not 1<=workers<=3:raise ValueError('At most three independent episodes in flight')
    root=Path(root);info=verify(root);stop=threading.Event();start=time.monotonic();outcomes=[]
    for row in info['episodes']:
        if not read_json(Path(row['folder'])/'perception/manifest.json')['complete']:raise ValueError('Perception incomplete')
        if (Path(row['folder'])/'plus').exists():raise ValueError('No automatic retry or overwrite of paid work')
    def worker(row):
        if stop.is_set():return dict(uid=row['uid'],status='not_run_batch_stop')
        folder=Path(row['folder'])
        try:
            result=run_episode(folder/'perception',root/'teacher-config.json',folder/'plus',credentials,stop_requested=stop.is_set)
            checks=audit(folder/'plus')
            if result.get('fatal_stop') or not checks['interface_pass']:stop.set()
            return dict(uid=row['uid'],status=result['status'],fatal_stop=result.get('fatal_stop'),audit=checks['interface_pass'])
        except Exception as exc:
            stop.set();return dict(uid=row['uid'],status='infrastructure_failed',error=str(exc))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(worker,r) for r in info['episodes']]
        for f in as_completed(futures):
            outcomes.append(f.result());write_json(root/'batch-run.json',dict(workers=workers,outcomes=outcomes,wall_s=time.monotonic()-start,stopped=stop.is_set()))
            print('TASK_DONE',json.dumps(outcomes[-1]),flush=True)
    return outcomes


def report(root):
    from .r2r_pilot_review import episode_review, video, detail, esc, CSS
    root=Path(root);info=verify(root);reports=[]
    cfg=read_json(root/'teacher-config.json')
    profile=cfg.get('profile_id', 'unspecified')
    title=f"R2R {profile} · {len(info['episodes'])}个任务完整标注"
    parts=['<!doctype html><meta charset="utf-8"><title>R2R 扩展标注</title><style>'+CSS+'</style>',
        '<h1>'+esc(title)+'</h1>',
        f"<p>固定{len(info['episodes'])}个任务、全部{info['planned_requests']}个关键帧；训练池任务，不是独立盲测。理想odom与Habitat动作回放；未训练、未自动批准。waypoint/ref_line为示范GT，未输入教师。</p>",
        '<p><a href="REPORT.md">报告</a> · <a href="report.json">统计</a> · <a href="pilot.json">固定清单</a> · <a href="teacher-config.json">冻结配置</a> · <a href="batch-run.json">执行记录</a></p>']
    for entry in info['episodes']:
        folder=Path(entry['folder']);parts.append('<section><h2>'+esc(entry['uid'])+'</h2><p>'+esc(entry['instruction'])+'</p>')
        if (folder/'plus/final-map.json').exists():
            out,rows,r=episode_review(root,entry);video(out,entry,rows);reports.append(r)
            base=f'episodes/{entry["uid"]}/review'
            parts.append(f'<p>返回{r["returned"]}/{r["planned"]}；规则通过{r["rule_valid"]}；¥{r["cost_cny_estimate"]:.4f}。<a href="{base}/index.html">全部帧输入、输出、审核表</a> · <a href="{base}/episode.mp4">视频</a></p><video controls preload="metadata" src="{base}/episode.mp4" poster="{base}/poster.jpg"></video>')
        else:parts.append('<p>未完成；仍计入计划分母。</p>')
        parts.append('</section>')
    total=dict(episodes=reports,planned=info['planned_requests'],planned_tasks=len(info['episodes']),
        returned=sum(r['returned'] for r in reports),requested=sum(r['requested'] for r in reports),
        rule_valid=sum(r['rule_valid'] for r in reports),format_valid=sum(r['format_valid'] for r in reports),
        cost_cny_estimate=sum(r['cost_cny_estimate'] for r in reports),human_approved=0,semantic_accuracy=None,
        all_tasks_attempted=len(reports)==len(info['episodes']),source='R2R-VLNCE train pool',model=cfg['model'],teacher_profile=profile)
    # Aggregate actual responses, including rejected outputs; never infer cost or
    # completion from the number of valid labels.
    api_times=[];finish_reasons=Counter();local_lengths=[]
    for entry in info['episodes']:
        folder=Path(entry['folder'])/'plus'
        if not (folder/'report.json').exists():continue
        for row in read_json(folder/'report.json')['samples']:
            if 'api_wall_s' in row:api_times.append(row['api_wall_s'])
            response=folder/row['folder']/'response.json'
            if response.exists():
                raw=read_json(response)
                finish_reasons.update(c.get('finish_reason','unknown') for c in raw.get('choices',[]))
            text=folder/row['folder']/'raw.txt'
            if text.exists():
                try:
                    answer=json.loads(text.read_text())
                    if isinstance(answer,dict) and isinstance(answer.get('local'),str):local_lengths.append(len(answer['local']))
                except ValueError:pass
    batch=read_json(root/'batch-run.json') if (root/'batch-run.json').exists() else {}
    total.update(tokens=dict(sum((Counter(r['tokens']) for r in reports),Counter())),
        api_total_s=sum(api_times),api_p50_s=float(np.percentile(api_times,50)) if api_times else None,
        api_p95_s=float(np.percentile(api_times,95)) if api_times else None,
        annotation_parallel_wall_s=batch.get('wall_s'),workers=batch.get('workers'),
        api_finish_reasons=dict(finish_reasons),
        local_lengths=dict(n=len(local_lengths),over90=sum(n>90 for n in local_lengths),maximum=max(local_lengths,default=None)),
        accepted_map_events=sum(r['accepted_events'] for r in reports),
        tentative_landmark_records=sum(r['unique_landmarks'] for r in reports),
        all_frames_attempted=total['requested']==total['planned'],
        all_responses_received=total['returned']==total['planned'],
        unrequested=total['planned']-total['requested'],
        requested_without_response=total['requested']-total['returned'])
    failures=[];answers=[]
    for entry in info['episodes']:
        folder=Path(entry['folder'])/'plus'
        if not (folder/'report.json').exists():continue
        for row in read_json(folder/'report.json')['samples']:
            if row['status']!='format_valid_unreviewed':failures.append(dict(episode=entry['uid'],**row))
            f=folder/row['folder']/'raw.txt'
            if f.exists():
                try:a=json.loads(f.read_text())
                except ValueError:continue
                answers.append(dict(episode=entry['uid'],image_id=row['source_image_id'],status=row['status'],answer=a))
    write_json(root/'failures.json',failures);write_json(root/'all-answers.json',answers)
    total['failure_reasons']=dict(Counter(r.get('reason',r['status']) for r in failures));write_json(root/'report.json',total)
    parts.insert(4,detail('覆盖、费用、规则结果与失败原因',total));(root/'index.html').write_text('\n'.join(parts))
    md=['# '+title,'',f"固定{len(info['episodes'])}个任务，{total['planned']}个关键帧；返回{total['returned']}，规则通过{total['rule_valid']}，人工批准0。",'',
        '|任务|返回/计划|规则通过|估算人民币|','|---|---:|---:|---:|']
    for r in reports:md.append(f"|[{r['episode_id']}](episodes/{r['episode_id']}/review/index.html)|{r['returned']}/{r['planned']}|{r['rule_valid']}|{r['cost_cny_estimate']:.6f}|")
    md+=['',f"费用估算合计¥{total['cost_cny_estimate']:.6f}，包含失败响应和思考tokens，不是账单。冻结基础价输入{cfg['native_per_million_tokens']['input']}元/百万、输出{cfg['native_per_million_tokens']['output']}元/百万；未扣活动优惠。价格依据：[官方模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)。",'',
        '任务按冻结清单选取，不按模型结果换样本。所有预定关键帧保留。每个任务独立递归Route/Local及地图，最多3个任务并行；任务内逐帧顺序执行，不重试。', '',
        'Local必须简洁且最多90字符；Route仅描述剩余导航命令，合理的Previous Route可以原样保留。指令指定的未见目标可保留为搜索目标，不编造其位置或方向。未证实完成时不允许DONE。', '',
        '规则通过不是语义正确率；没有独立文本/实例真值，不报告语义准确率或mask IoU。所有标签待人工审核，未启用训练。地图对象保持tentative。', '',
        '完整实际输入、教师原文、审计信息、mask及未来轨迹GT分栏展示。视频按合成源时间回放，不是模型控制或API实时延迟。', '',
        '[失败清单](failures.json) · [全部原始回答索引](all-answers.json) · [任务选择与来源](pilot.json)']
    md+=['','## 实际用量与地图结果','',
        f"实际API用量：输入{total['tokens'].get('prompt_tokens',0):,} tokens，输出{total['tokens'].get('completion_tokens',0):,} tokens；其中思考{total['tokens'].get('reasoning_tokens',0):,} tokens已包含在输出内，不重复计费。",
        f"API请求往返耗时累计{total['api_total_s']/60:.2f}分钟，p50={total['api_p50_s']}秒，p95={total['api_p95_s']}秒；不是服务端prefill/decode细分。并行标注墙钟{total['annotation_parallel_wall_s']}秒。",
        f"未发起{total['unrequested']}帧；已发起但无响应{total['requested_without_response']}帧。Local超过90字符{total['local_lengths']['over90']}帧，全部保留原文。",
        f"接受地图事件{total['accepted_map_events']}次，独立地图地标记录合计{total['tentative_landmark_records']}条；各任务间不做物理实例去重，均为待核验候选。",
        '以上是工程与规则统计，不是Landmark/Local/Route语义准确率。逐字段语义审核仍未批准。']
    (root/'REPORT.md').write_text('\n'.join(md)+'\n')
    print(json.dumps({k:v for k,v in total.items() if k!='episodes'},ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('initialize')
    for name in ('dataset','selection','output','config','previous-pilot'):a.add_argument('--'+name,required=True)
    a.add_argument('--count',type=int,default=12)
    a.add_argument('--all-remaining',action='store_true')
    a.add_argument('--exclude-run',action='append',default=[])
    a=sub.add_parser('prepare');a.add_argument('--root',required=True);a.add_argument('--python',default='/data/ticvla/envs/warehouse-realtime/bin/python')
    a=sub.add_parser('run');a.add_argument('--root',required=True);a.add_argument('--credentials',required=True);a.add_argument('--workers',type=int,default=3)
    a=sub.add_parser('report');a.add_argument('--root',required=True)
    a=p.parse_args();kw=vars(a);cmd=kw.pop('cmd');globals()[cmd](**kw)

if __name__=='__main__':main()
