"""Complete original R2R train preparation. No credentials, API calls or training.

Separate task metadata; hard-link RGB-D only after exact replay equivalence.
An interrupted job resumes committed stages, preserving partial attempts.
"""
import argparse
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .core import read_json, write_json
from .stage1a_data import load_sources, route_group, source_config, prepare_one, finite_metadata
from ..schema import fingerprint, file_sha256


def replay_key(episode, trace):
    # Rounded reference paths alone do NOT establish sensor/replay equivalence.
    return fingerprint(dict(scene=episode['scene_id'], position=episode['start_position'],
        rotation=episode['start_rotation'], actions=trace['actions'],
        locations=trace['locations'], sensors=source_config('r2r')))


def catalog(source, output, reuse=None, previous_selection=None):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    episodes,traces,path,gt=load_sources(source, {'r2r'})['r2r']
    groups=defaultdict(list);rows=[];missing=[]
    comparison_groups=set()
    if previous_selection:
        comparison_groups={r['route_group'] for r in read_json(previous_selection)['selected']
                           if r['source']=='r2r' and r['role']=='oracle_comparison'}
    scenes=[]
    for scene in sorted({e['scene_id'] for e in episodes}):
        p=Path(source)/'scene_datasets'/scene
        assets=[]
        for asset in (p,p.with_suffix('.navmesh')):
            assets.append(dict(path=str(asset),exists=asset.is_file(),
                sha256=file_sha256(asset) if asset.is_file() else None,
                bytes=asset.stat().st_size if asset.is_file() else None))
        scenes.append(dict(scene_id=scene,assets=assets))
    for e in sorted(episodes,key=lambda e:int(e['episode_id'])):
        uid=f"r2r-{e['episode_id']}";trace=traces.get(str(e['episode_id']))
        if trace is None:
            missing.append(dict(uid=uid,reason='missing_trace'));continue
        key=replay_key(e,trace);group=route_group(e)
        row=dict(source='r2r',uid=uid,episode_id=e['episode_id'],scene_id=e['scene_id'],
            trajectory_id=e['trajectory_id'],route_group=group,replay_group=key,
            role='oracle_comparison' if group in comparison_groups else 'train',
            source_dataset=str(path),source_gt=str(gt),source_frames=len(trace['actions'])+1)
        rows.append(row);groups[key].append(row)
    replay_groups=[]
    for key,members in groups.items():
        reusable=[]
        if reuse:
            for r in members:
                old=Path(reuse)/r['uid'];p=old/'replay-report.json'
                if p.exists() and read_json(p).get('passed'):
                    v=read_json(old/'verification-only.json')
                    if replay_key(v['original_episode'],v['original_gt'])==key:
                        meta=read_json(old/'capture/raw/episode.json')
                        if meta.get('sensor_config')==source_config('r2r'):
                            reusable.append(r)
        canonical=(reusable or members)[0]
        for r in members:r['canonical_uid']=canonical['uid']
        replay_groups.append(dict(key=key,canonical=canonical,members=members,
            reuse=str(Path(reuse)/canonical['uid']) if reusable else None))
    replay_groups.sort(key=lambda g:(g['canonical']['scene_id'],g['canonical']['episode_id']))
    value=dict(schema='stage1a-selection-v1',root=str(Path(source).resolve()),selected=rows,
        source_dataset_sha256=file_sha256(path),source_gt_sha256=file_sha256(gt),
        original_train_tasks=len(episodes),unique_replays=len(groups),scene_count=len(scenes),
        source_frames=sum(r['source_frames'] for r in rows),
        unique_source_frames=sum(g['canonical']['source_frames'] for g in replay_groups),
        scenes=scenes,missing=missing,replay_groups=replay_groups,
        official_evaluation_splits_used=False,envdrop_included=False,
        pose_source='ideal_simulator_odometry',time_source='synthetic_action_step_clock',
        synthetic_dt_s=.2,annotation_language='en',actual_api_calls=0,
        teacher_calls_authorized=False,semantic_labels_generated=False,
        sensor_contract=source_config('r2r'),reuse_root=str(reuse) if reuse else None)
    value['assets_complete']=not missing and all(a['exists'] for s in scenes for a in s['assets'])
    write_json(root/'selection.json',value)
    write_json(root/'source-inventory.json',{k:v for k,v in value.items() if k not in ('selected','replay_groups')})
    summarize(root, 'catalogued')
    return value


def archive_partial(dest, names):
    present=[dest/n for n in names if (dest/n).exists()]
    if not present:return
    archive=dest/'interrupted-attempts'/str(time.time_ns());archive.mkdir(parents=True)
    for p in present:shutil.move(str(p),str(archive/p.name))


def clone_capture(original, dest, row, episode, trace):
    """Reuse immutable pixels, never another task's identity/instruction/epoch."""
    original=Path(original);dest=Path(dest)
    if (dest/'replay-report.json').exists():return
    report=read_json(original/'replay-report.json')
    verification=read_json(original/'verification-only.json')
    if not report['passed'] or replay_key(verification['original_episode'], verification['original_gt'])!=replay_key(episode,trace):
        raise ValueError('Replay reuse mismatch')
    archive_partial(dest,['capture','verification-only.json'])
    src=original/'capture/raw';raw=dest/'capture/raw';raw.mkdir(parents=True)
    for name in ('rgb','depth'):
        (raw/name).mkdir()
        for p in (src/name).iterdir():os.link(p,raw/name/p.name)
    (raw/'supervision').mkdir()
    meta=read_json(src/'episode.json')
    meta.update(episode_id=row['uid'],source_episode_id=episode['episode_id'],
        source_dataset=row['source_dataset'],
        task=dict(task_id=row['uid'],instruction=episode['instruction']['instruction_text'],task_type='INSTRUCTION'),
        pixel_reuse=dict(source=str(original),replay_key=row['replay_group']))
    write_json(raw/'episode.json',meta)
    count=0
    with (raw/'frames.jsonl').open('x') as out:
        for line in (src/'frames.jsonl').read_text().splitlines():
            f=json.loads(line);f['odom_epoch']=row['uid'];out.write(json.dumps(f)+'\n');count+=1
    if count!=row['source_frames']:raise ValueError('Incomplete reusable capture')
    missing=[];verification.update(original_episode=finite_metadata(episode,missing=missing),
                                  original_gt=trace,missing_original_timing_fields=missing)
    write_json(dest/'verification-only.json',verification)
    report.update(uid=row['uid'],role=row['role'],pixel_reuse_source=str(original),replay_key=row['replay_group'])
    write_json(dest/'replay-report.json',report)


def prepare_task(dest, row):
    from .oracle_comparison import teacher_sample
    from .semantic_annotation import write_bundle
    dest=Path(dest)
    try:
        if (dest/'preparation-report.json').exists():
            report=read_json(dest/'preparation-report.json')
        else:
            archive_partial(dest,['preannotation','records','manifest.json','schedule.json','index.html'])
            report=prepare_one(dest,row)
        if not report['passed']:return dict(uid=row['uid'],stage='geometry_failed')
        bundle=dest/'teacher-bundle'
        if not (bundle/'manifest.json').exists():
            archive_partial(dest,['teacher-bundle'])
            samples=[teacher_sample(read_json(r['path'])) for r in report['records']]
            write_bundle(bundle,samples,source=row['uid'],episode_review=True)
        write_json(dest/'ready.json',dict(uid=row['uid'],samples=report['samples'],
            manifest_sha256=file_sha256(dest/'manifest.json'),
            bundle_sha256=file_sha256(bundle/'manifest.json'),
            semantic_labels_generated=False,perception_cache='not_prepared',actual_api_calls=0))
        return dict(uid=row['uid'],stage='prepared',samples=report['samples'])
    except Exception as exc:
        failure=dict(uid=row['uid'],stage='preparation_failed',error_type=type(exc).__name__,error=str(exc))
        write_json(dest/'preparation-failure.json',failure);return failure


def summarize(root, phase=None):
    root=Path(root);sel=read_json(root/'selection.json');counts=Counter();rows=[];samples=0
    for r in sel['selected']:
        d=root/'episodes'/r['uid'];status='pending';n=0;reason=None
        if (d/'ready.json').exists():
            status='prepared';n=read_json(d/'ready.json')['samples'];samples+=n
        elif (d/'preparation-failure.json').exists():
            status='preparation_failed';reason=read_json(d/'preparation-failure.json').get('error')
        elif (d/'preparation-report.json').exists():
            q=read_json(d/'preparation-report.json');status='geometry_passed' if q['passed'] else 'geometry_failed'
        elif (d/'replay-report.json').exists():
            q=read_json(d/'replay-report.json');status='replayed' if q['passed'] else 'replay_failed';reason=q.get('error')
        counts[status]+=1;rows.append(dict(uid=r['uid'],status=status,samples=n,reason=reason,role=r['role']))
    old=read_json(root/'status.json') if (root/'status.json').exists() else {}
    report=dict(phase=phase or old.get('phase'),updated_unix=time.time(),tasks=len(rows),
        counts=dict(counts),prepared_keyframes=samples,actual_api_calls=0,semantic_labels_generated=0,
        train_ready=False,perception_cache='not_prepared',all_geometry_prepared=counts['prepared']==len(rows),
        free_disk_gib=shutil.disk_usage(root).free/2**30,episodes=rows)
    write_json(root/'status.json',report)
    body=''.join(f'<tr><td><a href="episodes/{r["uid"]}/index.html">{r["uid"]}</a></td><td>{r["status"]}</td><td>{r["samples"]}</td><td>{html.escape(r["reason"] or "")}</td></tr>' for r in rows)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>R2R 全量数据准备</title><h1>R2R 全量训练数据准备</h1><p>仅准备数据，API调用 0；语义标签未生成，训练未启用。RGB-D＋理想odom；动作回放时间不是真实采集时间。</p><p>phase: '+html.escape(str(report['phase']))+'</p><pre>'+html.escape(json.dumps({k:v for k,v in report.items() if k!='episodes'},ensure_ascii=False,indent=2))+'</pre><a href="source-inventory.json">资源核验</a> · <a href="status.json">完整状态</a><table><tr><th>任务</th><th>状态</th><th>关键帧</th><th>原因</th></tr>'+body+'</table>')
    return report


def run(root, habitat_python, workers=4, group_limit=None, reserve_gib=40):
    root=Path(root).resolve();lock=(root/'worker.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sel=read_json(root/'selection.json')
    if not sel['assets_complete']:raise ValueError('Missing source assets; inspect inventory')
    episodes,traces,path,gt=load_sources(sel['root'],{'r2r'})['r2r']
    if file_sha256(path)!=sel['source_dataset_sha256'] or file_sha256(gt)!=sel['source_gt_sha256']:
        raise ValueError('Source changed after catalog')
    byid={e['episode_id']:e for e in episodes}
    groups=sel['replay_groups'][:group_limit] if group_limit else sel['replay_groups']
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
         'PYTHONPATH':str(Path(__file__).resolve().parents[3]),'MAGNUM_LOG':'quiet','HABITAT_SIM_LOG':'quiet'}
    (root/'logs').mkdir(exist_ok=True);(root/'shards').mkdir(exist_ok=True)
    summarize(root,'running_geometry')
    try:
        # Bounded shards retain scene reuse without launching thousands of simulators.
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for offset in range(0,len(groups),24):
                if shutil.disk_usage(root).free<reserve_gib*2**30:raise RuntimeError('Disk reserve reached; resumable stop')
                shard=groups[offset:offset+24];todo=[]
                for g in shard:
                    row=g['canonical'];dest=root/'episodes'/row['uid']
                    if not (dest/'replay-report.json').exists():
                        if g['reuse']:
                            clone_capture(g['reuse'],dest,row,byid[row['episode_id']],traces[str(row['episode_id'])])
                        else:
                            archive_partial(dest,['capture','verification-only.json']);todo.append(row)
                if todo:
                    selection=root/'shards'/f'{offset:06d}.json'
                    write_json(selection,dict(root=sel['root'],selected=todo))
                    with (root/'logs'/f'replay-{offset:06d}.log').open('a') as log:
                        subprocess.run([habitat_python,'-u','-m','ticvla.warehouse.dataset_tools.stage1a_data','replay',
                            '--selection',str(selection),'--output',str(root/'episodes')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                futures=[]
                for g in shard:
                    src=root/'episodes'/g['canonical']['uid'];rr=read_json(src/'replay-report.json')
                    for row in g['members']:
                        dest=root/'episodes'/row['uid']
                        if not rr['passed']:
                            write_json(dest/'replay-report.json',{**rr,'uid':row['uid'],'canonical_failure':str(src)});continue
                        if dest!=src:clone_capture(src,dest,row,byid[row['episode_id']],traces[str(row['episode_id'])])
                        if not (dest/'ready.json').exists() and not (dest/'preparation-failure.json').exists():
                            futures.append(pool.submit(prepare_task,dest,row))
                summarize(root)
                last_status=time.monotonic()
                for f in as_completed(futures):
                    print(json.dumps(f.result()),flush=True)
                    if time.monotonic()-last_status>=15:
                        summarize(root);last_status=time.monotonic()
                summarize(root)
        report=summarize(root,'geometry_complete' if not group_limit else 'geometry_subset_complete')
        # Final manifest contains only verified geometry and still disables text/action supervision.
        records=[]
        for row in sel['selected']:
            d=root/'episodes'/row['uid']
            if (d/'ready.json').exists() and row['role']=='train':records.extend(read_json(d/'manifest.json')['records'])
        write_json(root/'manifest.json',dict(schema_version=2,role='preannotation',optimizer_updates_permitted=False,
            split_unit='layout_family',records=records,semantic_labels_generated=False,actual_api_calls=0))
        return report
    except BaseException as exc:
        write_json(root/'worker-error.json',dict(error_type=type(exc).__name__,error=str(exc),time=time.time()))
        summarize(root,'stopped');raise
    finally:lock.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['catalog','run','status'])
    p.add_argument('--output',required=True);p.add_argument('--source',default='/project/docker/VLN/NaVILA/data')
    p.add_argument('--reuse');p.add_argument('--previous-selection');p.add_argument('--workers',type=int,default=4)
    p.add_argument('--habitat-python',default='/root/miniconda3/envs/navila-eval/bin/python')
    p.add_argument('--group-limit',type=int);p.add_argument('--reserve-gib',type=float,default=40)
    a=p.parse_args()
    if a.mode=='catalog':catalog(a.source,a.output,a.reuse,a.previous_selection)
    elif a.mode=='status':summarize(a.output)
    else:run(a.output,a.habitat_python,a.workers,a.group_limit,a.reserve_gib)


if __name__=='__main__':main()
