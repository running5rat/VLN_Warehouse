"""Source-frame-only mapping selections; geometry/detections never await QA."""
import copy
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from ticvla.warehouse.mapping import MapStore, SourceObservation
from ticvla.warehouse.mapping.semantic_frame import write_json
from ticvla.warehouse.perception.realtime import consume_masks, consume_text, save_frame
from ticvla.warehouse.perception.instance_selection import candidate_request, selection_result
from ticvla.warehouse.schema import file_sha256
from .live_metrics import atomic_json, complete_lines
from .realtime_mapping import worker, render

def p95(values):
    return float(np.percentile(values,95)) if values else None


def pending_rows(source,batch,status):
    return [dict(instance_index=i,category=x['category'],bbox=x['bbox'],confidence=x['confidence'],
                 selected=False,parent_event_id=f"rt:{source['image_id']}:{i}",geometry_file=None,
                 result=dict(status=status,reason=None,landmark_uid=None)) for i,x in enumerate(batch['instances'])]


def expired(source,now):
    return now-source['submitted_wall'] > 5


def run(capture,root,weights,options):
    from .live_mapping import LiveBEV
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ctxmp=mp.get_context('spawn');results=ctxmp.Queue(maxsize=8)
    warm_path=root/'warmup.png';Image.new('RGB',(960,720)).save(warm_path)
    warm=dict(image_id='warmup-only',rgb=str(warm_path),rgb_hash=file_sha256(warm_path),depth_hash='warmup',t_obs=0,T_odom_body=np.eye(4).tolist(),odom_epoch='warmup')
    queues={k:ctxmp.Queue(maxsize=1) for k in ('seg','qa','ocr')}
    workers={k:ctxmp.Process(target=worker,args=(k,q,results,weights,options,warm)) for k,q in queues.items()}
    for w in workers.values():w.start()
    store=MapStore(root/'map',array_compression_level=0);bev=LiveBEV()
    contexts={};ready=set();pending={k:None for k in queues};busy={k:False for k in queues}
    drops={k:0 for k in queues};due={'qa':-1e10,'ocr':-1e10};records=[];count=0;target=None
    stream=None;metadata=None;start=time.time();last_source=None
    archive=ThreadPoolExecutor(max_workers=1);futures=[];ocr_samples=[];qa_samples=[]
    metrics=(root/'metrics.jsonl').open('w',buffering=1)
    def log(module,**values):
        row=dict(module=module,wall=time.time(),**values);records.append(row)
        metrics.write(json.dumps(row,allow_nan=False)+'\n')
    def folder(source):return root/'frames'/source['image_id'].replace(':','_')
    def save(c):
        if c.get('saved') or 'batch' not in c:return
        c['saved']=True
        # Immutable handoff: projection owner stays in main process, archive reads points only.
        futures.append(archive.submit(save_frame,folder(c['source']),c['source'],c['batch'],copy.deepcopy(c['rows']),copy.deepcopy(c['selection']),copy.deepcopy(c.get('ocr')),copy.deepcopy(c.get('binding'))))
    def state(c,status):
        c['selection']=dict(selection_mode='instances',status=status,selected_instances=[],selected_categories=[],source_image_id=c['source']['image_id'])
        if 'batch' in c:c['rows']=pending_rows(c['source'],c['batch'],status)
    def finish(c,status):
        if c.get('decision_done'):return
        c['decision_done']=True;state(c,status)
        log('selection_disposition',image_id=c['source']['image_id'],status=status)
    def offer(kind,source):
        old=pending[kind]
        if old is not None:
            drops[kind]+=1;log(kind+'_skipped',image_id=old['image_id'],replaced_by=source['image_id'])
            oldctx=contexts.get(old['image_id'])
            if oldctx:
                oldctx[kind+'_skipped']=True
                if kind=='qa':finish(oldctx,'replaced')
        pending[kind]=source
    def publish(c,kind=None):
        t=time.time();source=c['source'];view=store.landmark_view();landmarks=view['landmarks']
        if kind in ('admission','text'):
            atomic_json(root/'landmark-view.json',view)
        render(root,source,c['batch'],c['rows'],landmarks,c['selection'],c.get('qa'),
               {**c['binding'],'source_image_id':source['image_id']} if c.get('binding') else None)
        if kind=='candidate':
            path=root/'candidate-points.npz'
            with (root/'candidate-points.tmp').open('wb') as f:
                np.savez(f,**{f'd{i}':p['points'] for i,p in c['batch']['prepared_projections'].items()})
            (root/'candidate-points.tmp').replace(path)
        if kind in ('admission','text'):bev.render(landmarks,root/'map.png',compress_level=0)
        now=time.time()
        if kind:
            log(kind,image_id=source['image_id'],t_obs=source['t_obs'],latency_s=now-source['submitted_wall'],
                service_s=now-t,accepted_instances=sum(r['result']['status']=='accepted' for r in c['rows']))
        def recent_rate(module):
            values=[r['wall'] for r in records if r['module']==module][-30:]
            return (len(values)-1)/max(1e-9,values[-1]-values[0]) if len(values)>1 else 0.0
        atomic_json(root/'status.json',dict(updated_wall=now,frames=count,selection_mode='instances',
            geometry_frames=count,geometry_hz=recent_rate('geometry'),
            semantic_frames=sum(r['module']=='candidate' for r in records),semantic_hz=recent_rate('candidate'),
            admitted_frames=sum(r['module']=='admission' and r['accepted_instances']>0 for r in records),
            selection_completed_frames=sum(r['module']=='admission' for r in records),
            verified=0,tentative=len(landmarks),landmark_view_counts=view['counts'],source_time=(last_source or source)['t_obs'],
            semantic_pending=sum(busy.values())+sum(x is not None for x in pending.values()),store=store.summary()))
    def bind(c):
        if 'ocr' not in c or 'batch' not in c or not c.get('decision_done') or c.get('binding_done'):return
        c['binding_done']=True
        if expired(c['source'],time.time()):
            log('ocr_disposition',image_id=c['source']['image_id'],status='expired');return
        c['binding']=consume_text(store,c['source'],c['batch'],c['rows'],c['ocr'])
        image=Image.open(c['source']['rgb']).convert('RGB');draw=ImageDraw.Draw(image)
        for text in c['binding']['texts']:
            xy=[tuple(p) for p in text['polygon']]
            draw.line(xy+xy[:1],fill='yellow',width=2)
            draw.text(xy[0],text['text']+' ['+text['status']+']',fill='yellow',stroke_width=1,stroke_fill='black')
        image.save(root/'ocr.tmp.png',compress_level=0);(root/'ocr.tmp.png').replace(root/'ocr.png')
        accepted=any(r['result']['status']=='accepted' for r in c['rows'])
        publish(c,'text' if accepted else None)
        log('ocr_binding',image_id=c['source']['image_id'],t_obs=c['source']['t_obs'],
            status='processed' if accepted else 'not_applicable_no_admitted_parent',
            latency_s=time.time()-c['source']['submitted_wall'],
            bound_updates=sum(x['status']=='accepted' for x in c['binding']['updates']))
    try:
        while True:
            dirty=False
            for f in list(futures):
                if f.done():f.result();futures.remove(f)
            if len(futures) >= 32:
                log('artifact_backpressure', pending=len(futures))
                futures[0].result()
            while True:
                try:r=results.get_nowait()
                except queue.Empty:break
                kind=r['kind']
                if r.get('fatal'):raise RuntimeError(r['fatal'])
                if r.get('ready'):
                    ready.add(kind);write_json(root/(kind+'-provenance.json'),r)
                    if len(ready)==3:atomic_json(root/'ready.json',dict(ready=True,backend='yoloe-realtime',selection_mode='instances',load_s=time.time()-start))
                    continue
                busy[kind]=False;source=r['source'];c=contexts.get(source['image_id'])
                log('worker_service',kind=kind,image_id=source['image_id'],service_s=r['finished']-r['started'],pending_frames=sum(x is not None for x in pending.values()),queued_results=results.qsize())
                if r.get('error'):
                    log(kind+'_failure',image_id=source['image_id'],error=r['error'])
                    if c:
                        c[kind+'_skipped']=True
                        if kind in ('qa','seg'):finish(c,'failed')
                    continue
                data=r['data']
                if kind in ('qa','ocr'):
                    futures.append(archive.submit(write_json,root/kind/source['image_id'].replace(':','_')/'result.json',data))
                if kind=='ocr':log('ocr_model_completed',image_id=source['image_id'],latency_s=time.time()-source['submitted_wall'])
                if c is None:
                    log(kind+'_late',image_id=source['image_id'],status='expired_or_evicted');continue
                if kind=='seg':
                    t=time.time();c['batch']=data
                    data['prepared_projections']={i:store.prepare_projection(source['image_id'],x['mask']) for i,x in enumerate(data['instances'])}
                    state(c,'pending_selection' if c['qa_due'] else 'not_sampled')
                    publish(c,'candidate')
                    log('segmentation',image_id=source['image_id'],t_obs=source['t_obs'],service_s=data['elapsed_s'],projection_s=time.time()-t)
                    if c['qa_due'] and not expired(source,time.time()):
                        request=candidate_request(source,data,root/'qa'/source['image_id'].replace(':','_'))
                        data['candidate_request']=request
                        offer('qa',{**source,'candidate_request':request})
                    else:finish(c,'expired' if c['qa_due'] else 'not_sampled')
                    bind(c)
                elif kind=='qa':
                    c['qa']=data
                    if expired(source,time.time()):finish(c,'expired')
                    else:
                        c['selection']=selection_result(data);c['decision_done']=True
                        c['rows']=consume_masks(store,c['source'],c['batch'],c['selection'])
                        publish(c,'admission')
                    log('qa',image_id=source['image_id'],t_obs=source['t_obs'],valid=data['valid'],service_s=data['elapsed_s'],status=c['selection']['status'])
                    bind(c)
                else:c['ocr']=data;bind(c)
                dirty=True
            now=time.time()
            for key,c in list(contexts.items()):
                complete=c.get('decision_done') and ('batch' in c) and (not c['ocr_due'] or c.get('binding_done') or c.get('ocr_skipped'))
                evict=expired(c['source'],now) or len(contexts)>32
                if complete or evict:
                    if evict:
                        if not c.get('decision_done'):finish(c,'expired')
                        log('context_evicted',image_id=key,status='expired' if expired(c['source'],now) else 'capacity')
                    save(c);contexts.pop(key)
            if any(not w.is_alive() for w in workers.values()):raise RuntimeError('Inference worker exited')
            episode=capture/'raw/episode.json'
            if stream is None and episode.exists():
                try:
                    metadata=json.loads(episode.read_text());stream=(episode.parent/metadata['frames_file']).open()
                except (FileNotFoundError,json.JSONDecodeError):pass
            if stream:
                f=next(complete_lines(stream),None)
                if f is not None:
                    t=time.time();obs=SourceObservation(f"{metadata['episode_id']}:{count:08d}",str(episode.parent/f['rgb']),str(episode.parent/f['depth']),f['K'],f['T_body_camera'],f['T_odom_body'],f['t_obs'],f['odom_epoch'],f['odom_epoch'],metadata['task']['task_id'])
                    count+=1;source=store.add_observation(obs);source['submitted_wall']=f['wall_time'];last_source=source
                    bev.add(store.points(source),f['T_odom_body'])
                    log('geometry',image_id=source['image_id'],t_obs=source['t_obs'],service_s=time.time()-t,capture_to_geometry_s=time.time()-source['submitted_wall'])
                    qa_due=source['t_obs']-due['qa']>=2-1e-6
                    ocr_due=qa_due or source['t_obs']-due['ocr']>=1-1e-6
                    c=dict(source=source,qa_due=qa_due,ocr_due=ocr_due);contexts[source['image_id']]=c
                    offer('seg',source)
                    if qa_due:due['qa']=source['t_obs'];qa_samples.append(source['image_id'])
                    if ocr_due:due['ocr']=source['t_obs'];ocr_samples.append(source['image_id']);offer('ocr',source)
                    if count%5==1:bev.render(store.landmark_view()['landmarks'],root/'map.png',compress_level=0)
                    dirty=True
            for kind in queues:
                if kind in ready and pending[kind] is not None and not busy[kind]:
                    if kind=='qa' and expired(pending[kind],time.time()):
                        log('qa_expired_before_start',image_id=pending[kind]['image_id']);pending[kind]=None
                    else:queues[kind].put(pending[kind]);pending[kind]=None;busy[kind]=True
            done=capture.parent/'capture-process.json'
            if done.exists():
                if json.loads(done.read_text())['exit_code']!=0:raise RuntimeError('Capture failed')
                target=json.loads((capture/'capture-report.json').read_text())['frames']
                if count==target and not any(busy.values()) and not any(x is not None for x in pending.values()):break
            if not dirty:time.sleep(.005)
        for c in contexts.values():
            if not c.get('decision_done'):finish(c,'unfinished')
            save(c)
        for f in futures:f.result()
        summarize(root,capture,records,count,target,ocr_samples,qa_samples,drops,store)
    finally:
        for q in queues.values():
            try:q.put_nowait(None)
            except queue.Full:pass
        for w in workers.values():
            w.join(5)
            if w.is_alive():w.terminate();w.join(5)
        archive.shutdown(wait=True);store.close();metrics.close()
        if stream:stream.close()


def summarize(root,capture,records,count,target,ocr_samples,qa_samples,drops,store):
    sim=json.loads((capture/'performance-report.json').read_text())
    start,end=sim['run_wall_start'],sim['run_wall_end']
    geometry=[x for x in records if x['module']=='geometry' and start<=x['wall']<=end]
    candidates=[x for x in records if x['module']=='candidate' and start<=x['wall']<=end]
    ocrmotion=[x for x in records if x['module']=='ocr_binding' and start<=x['wall']<=end]
    duration=max(end-start,1e-9)
    def motion(module):return [r for r in records if r['module']==module and start<=r['wall']<=end]
    decisions=motion('admission');admits=[r for r in decisions if r['accepted_instances']>0]
    texts=motion('text');bindings=[r for r in ocrmotion if r['status']=='processed']
    eligible={r['image_id'] for r in records if r['module']=='admission' and r['accepted_instances']>0 and r['image_id'] in ocr_samples}
    processed={r['image_id'] for r in records if r['module']=='ocr_binding' and r['status']=='processed'}
    timely={r['image_id'] for r in ocrmotion if r['status']=='processed' and r['latency_s']<=2}
    all_candidates=[r for r in records if r['module']=='candidate'];all_texts=[r for r in records if r['module']=='text']
    result=dict(selection_mode='instances',navigation_task_included=False,geometry_all_consumed=count==target,
        motion_seconds=duration,selection_completed_hz=len(decisions)/duration,geometry_hz=len(geometry)/duration,mask_hz=len(candidates)/duration,candidate_hz=len(candidates)/duration,
        mask_p95_s=p95([r['latency_s'] for r in candidates]),admission_hz=len(admits)/duration,
        admission_p95_s=p95([r['latency_s'] for r in admits]),ocr_p95_s=p95([r['latency_s'] for r in texts]),
        ocr_samples=len(ocr_samples),qa_samples=len(qa_samples),
        candidate_all_completed_p95_s=p95([r['latency_s'] for r in all_candidates]),
        text_all_completed_p95_s=p95([r['latency_s'] for r in all_texts]),
        candidate_completed_total=len(all_candidates),
        candidate_deadline_misses=sum(r['latency_s']>.5 for r in all_candidates),
        text_deadline_misses=sum(r['latency_s']>2 for r in all_texts),
        ocr_binding_eligible_samples=len(eligible),ocr_binding_completed_eligible=len(eligible&processed),
        ocr_binding_completed_eligible_fraction=len(eligible&processed)/max(1,len(eligible)),
        ocr_binding_timely_during_motion_fraction=len(eligible&timely)/max(1,len(eligible)),
        ocr_model_completed=sum(r['module']=='ocr_model_completed' for r in records),
        ocr_model_completed_during_motion=len(motion('ocr_model_completed')),
        ocr_binding_applicable_during_motion=len(bindings),ocr_binding_not_applicable_during_motion=sum(r['status']!='processed' for r in ocrmotion),
        ocr_bindings_fraction_all_samples=len(bindings)/max(1,len(ocr_samples)),
        ocr_model_completion_fraction_during_motion=len(motion('ocr_model_completed'))/max(1,len(ocr_samples)),
        text_updates_during_motion=sum(r['bound_updates'] for r in bindings),
        source_frame_only=True,mask_completed=len(candidates),drops=drops,
        formal_admitted_instances_during_motion=sum(r['accepted_instances'] for r in admits),
        browser_paint_latency_measured=False,publication_boundary='files ready to read; candidate and admitted separate',
        store=store.summary())
    result['candidate_timing_pass']=bool(result['geometry_all_consumed'] and result['geometry_hz']>=4.75 and result['mask_hz']>=4.75 and result['mask_p95_s'] is not None and result['mask_p95_s']<=.5)
    result['ocr_timing_pass']=bool(result['ocr_p95_s'] is not None and result['ocr_p95_s']<=2 and result['ocr_model_completion_fraction_during_motion']>=.95)
    result['timing_pass']=result['candidate_timing_pass'] and result['ocr_timing_pass']
    write_json(root/'realtime-report.json',result)
    write_json(root/'report.json',dict(perception_backend='yoloe-realtime',selection_mode='instances',geometry={'count':count},realtime=result,store=store.summary(),llm_calls=sum(r['module']=='qa' for r in records),structured_inference_executed=False))
    print('REALTIME',json.dumps({k:v for k,v in result.items() if k!='store'}),flush=True)
