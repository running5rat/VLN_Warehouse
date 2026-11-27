"""Causal teacher -> real masks/OCR -> persistent map -> next teacher replay.

No controller, training, ground-truth map or human text enters the model request.
Perception is cached once; each teacher owns a new independent MapStore.
"""
import argparse
import copy
from dataclasses import asdict
import html
import json
from pathlib import Path
import time
import numpy as np
from .core import read_json, write_json
from .semantic_annotation import validate_sample, build_request, send_request, parse_api_response, assemble_answer, agreement
from .credentials import load_credentials
from .teacher_budget import usage_charge
from ..schema import file_sha256, fingerprint
from ..semantic_input import map_records, render_semantic_map
from ..mapping import MapStore
from ..run_landmark_benchmark import source_from_frame
from ..compact_input import audit_context
from ..local_context import local_memory, local_context
from .annotation_policy import failure_policy, must_stop, gap_metadata


def samples_from(bundles):
    samples = []
    for bundle in bundles:
        bundle = Path(bundle)
        for item in read_json(bundle/'manifest.json')['records']:
            path = bundle/item['path']
            if file_sha256(path) != item['sha256']:
                raise ValueError('Changed source bundle')
            samples.append(validate_sample(read_json(path)))
    samples.sort(key=lambda s:s['observation']['t_obs'])
    if len({s['observation']['task']['task_id'] for s in samples}) != 1:
        raise ValueError('One episode per closed-loop run')
    if any(b['observation']['t_obs'] <= a['observation']['t_obs'] for a,b in zip(samples,samples[1:])):
        raise ValueError('Duplicate or nonmonotonic source')
    return samples


def prepare(bundles, output, weights, vocabulary=None):
    from ..perception.realtime import FixedSegmenter, MobileReader
    from ..perception.instance_selection import candidate_request
    from ..config import WarehouseConfig
    from ..geometry import LocalMapBuilder
    from ..navigation.geometry import render_local
    samples = samples_from(bundles)
    root = Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    segmenter = FixedSegmenter(weights, vocabulary=vocabulary); reader = MobileReader()
    cfg = WarehouseConfig(depth_stride=32, map_max_age=1e9)
    builder = LocalMapBuilder(cfg)
    episode = Path(samples[0]['provenance']['source_episode'])
    meta = read_json(episode)
    frames = [json.loads(x) for x in (episode.parent/meta['frames_file']).read_text().splitlines()]
    cursor=0; records=[]
    for i,sample in enumerate(samples):
        folder=root/f'{i:08d}';folder.mkdir()
        index=sample['provenance']['frame_index']
        if Path(sample['provenance']['source_episode']) != episode:
            raise ValueError('Episode changed')
        # Every earlier geometry frame, but neither current geometry nor current semantics.
        while cursor < index:
            f=frames[cursor]
            builder.add(depth=np.load(episode.parent/f['depth']),**{k:f[k] for k in
                ('K','T_body_camera','T_odom_body','t_obs','odom_epoch')})
            cursor+=1
        source=asdict(source_from_frame(episode,index))
        source.update(rgb_hash=file_sha256(source['rgb']),depth_hash=file_sha256(source['depth']))
        o=sample['observation']
        if (source['image_id']!=o['image_id'] or source['rgb_hash']!=o['rgb_history'][-1]['sha256']
                or source['t_obs']!=o['t_obs'] or not np.allclose(source['T_odom_body'],o['T_odom_body'])):
            raise ValueError('Teacher/capture source mismatch')
        grid,age=render_local(builder,source['T_odom_body'],source['t_obs'],source['odom_epoch'])
        np.savez_compressed(folder/'prior-geometry.npz',grid=grid,age=age)
        batch=segmenter.predict(source)
        request=candidate_request(source,batch,folder)
        instances=[]
        for j,inst in enumerate(batch['instances']):
            path=folder/f'mask-{j:03d}.npy';np.save(path,inst['mask'])
            instances.append({**{k:v for k,v in inst.items() if k!='mask'},'mask_path':str(path),'mask_sha256':file_sha256(path)})
        ocr=reader.predict(source)
        payload=dict(source=source,candidate_request=request,instances=instances,
            provenance=batch['provenance'],elapsed_s=batch['elapsed_s'],ocr=ocr,
            possibly_truncated=batch.get('possibly_truncated',False),
            geometry_source_until=frames[index-1]['t_obs'] if index else None,
            geometry_image_id=f"{meta['episode_id']}:{index-1:08d}" if index else None,
            resolution=cfg.map_resolution)
        write_json(folder/'sample.json',sample);write_json(folder/'perception.json',payload)
        records.append(dict(folder=folder.name,sample_sha256=file_sha256(folder/'sample.json'),
            perception_sha256=file_sha256(folder/'perception.json'),geometry_sha256=file_sha256(folder/'prior-geometry.npz')))
        print('prepared',source['image_id'],len(instances),'detections',len(ocr['texts']),'OCR regions',flush=True)
        write_json(root/'manifest.json',dict(protocol='semantic-loop-v1',complete=False,records=records,
            weights=str(weights),weights_sha256=file_sha256(weights),source_episode=str(episode),
            vocabulary=segmenter.vocabulary))
    manifest=read_json(root/'manifest.json');manifest['complete']=True;write_json(root/'manifest.json',manifest)


def load_frame(root, entry):
    folder=root/entry['folder']
    for name,key in [('sample.json','sample_sha256'),('perception.json','perception_sha256'),('prior-geometry.npz','geometry_sha256')]:
        if file_sha256(folder/name)!=entry[key]:raise ValueError('Frozen replay input changed')
    sample=read_json(folder/'sample.json');p=read_json(folder/'perception.json')
    instances=[]
    for inst in p['instances']:
        if file_sha256(inst['mask_path'])!=inst['mask_sha256']:raise ValueError('Mask changed')
        instances.append({**inst,'mask':np.load(inst['mask_path'])})
    batch=dict(instances=instances,source_image_id=p['source']['image_id'],rgb_hash=p['source']['rgb_hash'],
        provenance=p['provenance'],elapsed_s=p['elapsed_s'],candidate_request=p['candidate_request'])
    return sample,p,batch,np.load(folder/'prior-geometry.npz')['grid']


def recovery_request_key(request,cfg):
    payload=fingerprint(dict(endpoint=cfg['base_url'],api=cfg.get('api','responses'),request=request))
    evaluation=cfg.get('authorized_evaluation_run')
    return fingerprint(dict(payload_key=payload,evaluation_run_id=evaluation['id'])) if evaluation else payload


def recovery_prefix(root,report,cfg,manifest):
    """Only resume a durable completed prefix followed by one interrupted HTTP call."""
    rows=report['samples'];prefix=[];pending=None
    for i,row in enumerate(rows):
        if i>=len(manifest['records']) or row['folder']!=manifest['records'][i]['folder']:
            raise ValueError('Recovery source ordering mismatch')
        folder=root/row['folder']
        if row['status'] in ('format_valid_unreviewed','failed'):
            if pending or not (folder/'response.json').is_file():
                raise ValueError('Recovery requires saved response for completed frame')
            if row.get('failure_stage')=='map_commit':
                raise ValueError('Partial map commit requires separate reconciliation')
            prefix.append(row)
        elif row['status']=='request_started' and i==len(rows)-1:
            if (folder/'response.json').exists():
                raise ValueError('Saved but unprocessed response requires separate reconciliation')
            request,_=build_request(read_json(folder/'sample.json'),cfg)
            pending=recovery_request_key(request,cfg)
            if pending not in cfg.get('authorized_request_recovery',{}).get('request_keys',[]):
                raise ValueError('Interrupted request not explicitly authorized for recovery')
        else:raise ValueError('Unsupported recovery state')
    return prefix,pending


def run(prepared, config, output, credentials, stop_requested=None, resume=False):
    from ..perception.realtime import consume_masks, consume_text, save_frame, ALIASES, load_vocabulary
    prepared=Path(prepared).resolve();manifest=read_json(prepared/'manifest.json')
    if not manifest['complete']:raise ValueError('Incomplete perception preparation')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=resume)
    cfg=read_json(config);policy=failure_policy(cfg);load_credentials(credentials)
    vocabulary=load_vocabulary(manifest['vocabulary']) if 'vocabulary' in manifest else None
    aliases=vocabulary['aliases'] if vocabulary else ALIASES
    if resume:
        old_cfg=read_json(root/'config.json')
        if {k:v for k,v in cfg.items() if k!='authorized_request_recovery'} != old_cfg:
            raise ValueError('Recovery cannot change frozen model or prompt settings')
    else:write_json(root/'config.json',cfg)
    report=dict(status='running',model=cfg['model'],protocol='semantic-loop-v1',samples=[],actual_api_calls=0,
        human_approved=False,training=False,vehicle_commands=0,semantic_accuracy=None,annotation_failure_policy=policy,
        prepared_manifest_sha256=file_sha256(prepared/'manifest.json'),config_sha256=file_sha256(config),
        vocabulary=vocabulary,
        scope=f"{len(manifest['records'])} keyframe semantic updates; geometry includes all prior capture frames. Offline feedback, not realtime control.")
    completed_prefix=[]
    pending_key=None
    if resume:
        report=read_json(root/'report.json')
        if report['prepared_manifest_sha256']!=file_sha256(prepared/'manifest.json'):
            raise ValueError('Changed prepared manifest during recovery')
        completed_prefix,pending_key = recovery_prefix(root,report,cfg,manifest)
        report['samples']=copy.deepcopy(completed_prefix)
        report['recovery_config_sha256']=file_sha256(config)
        report['status']='running'
    store=MapStore(root/'map');previous=None;blocked=False;rejected=[]
    try:
        for i,entry in enumerate(manifest['records']):
            if i<len(completed_prefix):
                prior=completed_prefix[i];saved_folder=root/prior['folder']
                if prior['status']=='format_valid_unreviewed':
                    previous=dict(parsed=read_json(saved_folder/'proposal.json')['parsed'],
                        observation=read_json(saved_folder/'sample.json')['observation'],
                        hash=file_sha256(saved_folder/'response.json'))
                    rejected=[]
                else:rejected.append(prior['source_image_id'])
                continue
            sample,p,batch,grid=load_frame(prepared,entry);obs=sample['observation'];source=p['source']
            folder=root/entry['folder'];folder.mkdir(exist_ok=resume and i==len(completed_prefix) and pending_key is not None)
            row=dict(source_image_id=source['image_id'],folder=folder.name,status='pending')
            row['context_recovery']=gap_metadata(previous,rejected)
            row['context_recovery']['policy']=policy
            report['samples'].append(row)
            if stop_requested is not None and stop_requested():
                blocked=True
                row['status']='not_run_batch_stop'
                continue
            if blocked:
                row['status']='not_run_previous_failure';continue
            records=map_records(store.landmark_view()['landmarks'],source)
            version=store.version
            render_semantic_map(grid,p['resolution'],records).save(folder/'input-map.png')
            mapfile=str(folder/'input-map.png')
            obs['mapview']=dict(image=mapfile,image_sha256=file_sha256(mapfile),
                arrays=str(prepared/entry['folder']/'prior-geometry.npz'),
                arrays_sha256=entry['geometry_sha256'],landmarks=records,
                source_until=p['geometry_source_until'],resolution=p['resolution'],interface='causal-semantic-map-v1')
            obs['perception_candidates']=p['candidate_request']
            obs['semantic_input']=dict(version='causal-semantic-map-v1',map_version=version,
                source_until=p['geometry_source_until'],semantic_source_until=previous['observation']['t_obs'] if previous else None,
                parent_response_sha256=previous['hash'] if previous else None,
                geometry_source_image_id=p['geometry_image_id'],coordinate_frame='current_source_body_flu_metres',
                update_order='freeze past map -> infer current RGB/candidates -> commit masks/OCR -> next request')
            obs['route']=previous['parsed']['route'] if previous else obs['task']['instruction']
            obs['route_revision']=i
            obs['previous_local']=local_context(local_memory(previous['parsed']['local'],previous['observation'],
                origin='unreviewed_teacher_proposal',revision=previous['hash']),obs) if previous else dict(valid=False,reason='first_request')
            validate_sample(sample)
            request,summary=build_request(sample,cfg)
            if pending_key is not None and i==len(completed_prefix):
                if recovery_request_key(request,cfg)!=pending_key:
                    raise ValueError('Recovered request differs from interrupted source/context')
            write_json(folder/'input.json',summary);write_json(folder/'sample.json',sample)
            row.update(status='request_started',map_version_input=version,input_landmarks=len(records),
                geometry_source_until=p['geometry_source_until'],semantic_source_until=obs['semantic_input']['semantic_source_until'],
                detector_instances=len(batch['instances']),segmentation_s=p['elapsed_s'],ocr_s=p['ocr']['elapsed_s'])
            row['request_send_wall_unix']=time.time()
            row['request_sent']=True
            report['actual_api_calls']+=1;write_json(root/'report.json',report);start=time.monotonic()
            stage='api_request'
            try:
                raw=send_request(request,cfg)
                row['response_receive_wall_unix']=time.time()
                row['api_wall_s']=time.monotonic()-start
                write_json(folder/'response.json',raw)
                row['usage']=raw.get('usage')
                row['cost_cny_estimate']=usage_charge(raw,dict(cfg,usd_per_million_tokens=cfg['native_per_million_tokens']))
                partial=(raw.get('choices') or [{}])[0].get('message',{}).get('content')
                if isinstance(partial,str):(folder/'raw.txt').write_text(partial)
                stage='response_validation'
                answer=json.loads(parse_api_response(raw));proposal=assemble_answer(answer,sample)
                checks=agreement(proposal['parsed'],sample)
                write_json(folder/'proposal.json',proposal);write_json(folder/'checks.json',checks)
                row['vlm_s']=time.monotonic()-start
                if checks['rule_flags']:raise ValueError('Route rejected: '+','.join(checks['rule_flags']))
                names=[x['category'] for x in proposal['parsed']['landmark']]
                selected=list(dict.fromkeys(aliases.get(' '.join(x.lower().split()),' '.join(x.lower().split())) for x in names))
                if vocabulary and any(x not in {v['category'] for v in batch['instances']} for x in selected):
                    raise ValueError('Landmark category lacks same-frame detection')
                # Only now admit the current observation; no model reads its own update.
                stage='map_commit'
                saved=store.add_observation(source_from_frame(sample['provenance']['source_episode'],sample['provenance']['frame_index']))
                selection=dict(selected_categories=selected,request_id=file_sha256(folder/'response.json'),
                    source_image_id=saved['image_id'],source_time=saved['t_obs'],status='valid')
                start_map=time.monotonic()
                rows=consume_masks(store,saved,batch,selection)
                binding=consume_text(store,saved,batch,rows,p['ocr'])
                save_frame(folder/'mapping',saved,batch,rows,selection,p['ocr'],binding)
                view=store.landmark_view();write_json(folder/'map-after.json',view)
                # Render post-update for review only, using a slightly later diagnostic cutoff.
                after=map_records(view['landmarks'],dict(saved,t_obs=saved['t_obs']+1e-6))
                render_semantic_map(grid,p['resolution'],after).save(folder/'map-after.png')
                from DynaNav.warehouse_loop.realtime_mapping import render
                render(folder/'mapping',saved,batch,rows,view['landmarks'],selection,
                    dict(text=json.dumps(answer),elapsed_s=row['vlm_s']),{**binding,'source_image_id':saved['image_id']})
                row.update(status='format_valid_unreviewed',selected_categories=selected,
                    no_detection_categories=[x for x in selected if x not in {v['category'] for v in batch['instances']}],
                    accepted_instances=sum(x['result']['status']=='accepted' for x in rows),
                    text_updates=len(binding['updates']),map_version_output=store.version,
                    output_landmarks=len(view['landmarks']),mapping_s=time.monotonic()-start_map)
                previous=dict(parsed=proposal['parsed'],observation=copy.deepcopy(obs),hash=file_sha256(folder/'response.json'))
                rejected=[]
            except (ValueError,KeyError,TypeError,OSError) as exc:
                blocked=must_stop(cfg,exc,stage)
                row.update(status='failed',reason=str(exc),failure_stage=stage,
                    failure_action='stop_run' if blocked else 'skip_frame_keep_last_valid',
                    label_eligible=False)
                rejected.append(source['image_id'])
                if getattr(exc,'request_sent',True) is False:
                    report['actual_api_calls']-=1
                    row['request_sent']=False
            row['elapsed_s']=time.monotonic()-start
            print(cfg['model'],row['source_image_id'],row['status'],row.get('accepted_instances'),flush=True)
            write_json(root/'report.json',report);render_report(root,report)
        write_json(root/'final-map.json',store.landmark_view())
        report.update(status='completed_with_failures' if any(r['status']!='format_valid_unreviewed' for r in report['samples']) else 'completed_unreviewed',
            all_frames_attempted=all(r.get('request_sent',False) for r in report['samples']),
            fatal_stop=blocked,
            cost_cny_estimate=sum(r.get('cost_cny_estimate',0) or 0 for r in report['samples']),
            closed_loop_interface_pass=any(r.get('input_landmarks',0)>0 and r['status']=='format_valid_unreviewed' for r in report['samples']),
            all_frames_completed=all(r['status']=='format_valid_unreviewed' for r in report['samples']),
            store=store.summary())
        write_json(root/'report.json',report);render_report(root,report)
    finally:store.close()
    return report


def render_report(root,report):
    parts=['<!doctype html><meta charset="utf-8"><title>Semantic feedback review</title>',
        '<style>body{font:16px sans-serif;background:#17232d;color:white;padding:20px}a{color:cyan}img{width:32%;vertical-align:top}pre{white-space:pre-wrap}section{border-top:1px solid gray;margin-top:20px}</style>',
        '<h1>语义建图与导航标注闭环</h1><p>原始 RGB + 当前检测候选 + 上一轮语义地图；本轮输出提交后才进入下一轮。无车辆控制，标签待审核。</p>',
        '<pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>']
    for r in report['samples']:
        f=root/r['folder'];parts.append('<section><h2>'+html.escape(r['source_image_id'])+'</h2>')
        if (f/'input.json').exists():
            inp=read_json(f/'input.json')
            for j,image in enumerate(inp['images']):
                name=f'input-{j}.png'
                from PIL import Image
                Image.open(image['path']).save(f/name)
                parts.append(f'<a href="{r["folder"]}/{name}"><img title="{html.escape(image["label"])}" src="{r["folder"]}/{name}"></a>')
        if (f/'raw.txt').exists():parts.append('<pre>'+html.escape((f/'raw.txt').read_text())+'</pre>')
        for name in ['input.json','proposal.json','map-after.json','mapping/frame.json','mapping/semantic.png','map-after.png']:
            if (f/name).exists():parts.append(f'<a href="{r["folder"]}/{name}">{name}</a> · ')
        parts.append('</section>')
    (root/'index.html').write_text(''.join(parts))


def audit(output):
    """Check stored causal edges and all image hashes; no API requests."""
    root=Path(output);r=read_json(root/'report.json');previous=None;checks=[]
    for row in r['samples']:
        folder=root/row['folder']
        if not (folder/'input.json').exists():continue
        inp=read_json(folder/'input.json');c=audit_context(inp);binding=c['semantic_input']
        assert binding['map_version']==(previous['map_version_output'] if previous else 0)
        expected=file_sha256(root/previous['folder']/'response.json') if previous else None
        assert binding['parent_response_sha256']==expected
        for img in inp['images']:assert file_sha256(img['path'])==img['sha256']
        for lm in c.get('observed_map_landmarks',[]):
            assert lm['last_seen'] < c['source_time']
            assert c['current_image_id'] not in lm['source_image_ids']
        for key in ('source_until','semantic_source_until'):
            assert binding.get(key) is None or binding[key]<c['source_time']
        checks.append(dict(image_id=c['current_image_id'],past_only=True,hashes_valid=True,
            previous_output_map_version_reused=True,landmarks=len(c.get('observed_map_landmarks',[]))))
        if row['status']=='format_valid_unreviewed':previous=row
    store=MapStore(root/'map')
    try:
        assert json.loads(json.dumps(store.landmark_view()))==read_json(root/'final-map.json')
    finally:store.close()
    result=dict(checks=checks,restart_restored=True,interface_pass=True,
        semantic_accuracy_verified=False,all_planned_frames_completed=r['all_frames_completed'])
    write_json(root/'causality-audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--bundles',nargs='+',required=True);a.add_argument('--output',required=True)
    a.add_argument('--weights',required=True);a.add_argument('--vocabulary')
    a=sub.add_parser('run');a.add_argument('--prepared',required=True);a.add_argument('--config',required=True)
    a.add_argument('--output',required=True);a.add_argument('--credentials',required=True)
    a=sub.add_parser('audit');a.add_argument('--output',required=True)
    args=vars(p.parse_args());command=args.pop('command')
    {'prepare':prepare,'run':run,'audit':audit}[command](**args)

if __name__=='__main__':main()
