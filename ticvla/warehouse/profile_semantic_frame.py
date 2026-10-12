"""Real-model single-frame latency audit; production prompts are not modified."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import torch

from .mapping import MapStore
from .mapping.semantic_frame import write_json
from .perception import instance_selection as selection
from .perception.realtime import Selector, FixedSegmenter, MobileReader, consume_masks, consume_text, save_frame
from .run_landmark_benchmark import source_from_frame


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync(); start = time.perf_counter()
    result = fn()
    sync()
    return result, (time.perf_counter()-start)*1000


def compact_question(request):
    table = [{k:c[k] for k in ('instance_id','category','bbox_normalized')} for c in request['candidates']]
    return ('Image-1: <image>\nImage-2: <image>\n'
            'Select visible candidate IDs useful for a semantic map. Image-1 is RGB; '
            'Image-2 numbers detector boxes. Prefer text/symbol carriers or distinctive '
            'spatial references; skip ordinary repeated structures. Repeated signs may '
            'have different IDs. Detector classes may be wrong. Do not read text or infer '
            'uniqueness/staticness. Return JSON: selected_instances [{instance_id,purpose}], '
            'purpose identity_candidate or spatial_reference; missed_categories (diagnostic only). '
            'Empty lists allowed. Candidates: '+json.dumps(table,separators=(',',':')))


class VLMTrace:
    """CUDA event windows within real vision/prefill/decode calls, without per-token sync."""
    def __init__(self, selector):
        self.model = selector.model
        self.events = []; self.forwards = []; self.image_ms = []; self.constraint_ms = []
        self.callback_ms = []; self.generation_ms = None
        self.start = None; self.first_token_ready_ms = None

    def event_start(self):
        event = torch.cuda.Event(enable_timing=True); event.record()
        return event

    def timed_gpu(self, name, fn):
        def call(*a, **kw):
            start = self.event_start(); result = fn(*a, **kw); end = self.event_start()
            self.events.append((name,start,end)); return result
        return call

    def __enter__(self):
        from . import vlm
        import lmformatenforcer.integrations.transformers as fmt
        self.stack = ExitStack(); backbone = self.model.backbone
        self.stack.enter_context(patch.object(backbone,'extract_feature',self.timed_gpu('vision',backbone.extract_feature)))
        original_image = vlm.image_tensor
        def image(*a,**kw):
            t=time.perf_counter();out=original_image(*a,**kw)
            self.image_ms.append((time.perf_counter()-t)*1000);return out
        self.stack.enter_context(patch.object(vlm,'image_tensor',image))
        original_builder = fmt.build_transformers_prefix_allowed_tokens_fn
        def builder(*a,**kw):
            t=time.perf_counter();fn=original_builder(*a,**kw)
            self.constraint_ms.append((time.perf_counter()-t)*1000)
            def allowed(*a,**kw):
                t=time.perf_counter();out=fn(*a,**kw)
                self.callback_ms.append((time.perf_counter()-t)*1000);return out
            return allowed
        self.stack.enter_context(patch.object(fmt,'build_transformers_prefix_allowed_tokens_fn',builder))
        original_generate = backbone.language_model.generate
        def generate(*a,**kw):
            self.start=time.perf_counter();out=original_generate(*a,**kw)
            self.generation_ms=(time.perf_counter()-self.start)*1000
            self.generated_tokens=int(out.shape[-1]);return out
        self.stack.enter_context(patch.object(backbone.language_model,'generate',generate))
        def pre(module,args,kwargs):
            x=kwargs.get('inputs_embeds')
            if x is None:x=kwargs.get('input_ids')
            if len(self.forwards)==1 and self.first_token_ready_ms is None:
                # The first sampled token is ready when its next forward begins.
                self.first_token_ready_ms=(time.perf_counter()-self.start)*1000
            self.pending=(self.event_start(),int(x.shape[1]) if x is not None else None)
        def post(module,args,kwargs,output):
            start,n=self.pending;end=self.event_start()
            self.forwards.append((start,end,n))
        self.pre=backbone.language_model.register_forward_pre_hook(pre,with_kwargs=True)
        self.post=backbone.language_model.register_forward_hook(post,with_kwargs=True)
        return self

    def __exit__(self,*args):
        self.pre.remove();self.post.remove();self.stack.close();sync()

    def result(self):
        durations=[a.elapsed_time(b) for a,b,_ in self.forwards]
        visual_tokens=2*self.model.backbone.num_image_token
        return dict(image_preprocess_cpu_ms=sum(self.image_ms),
            vision_cuda_ms=sum(a.elapsed_time(b) for name,a,b in self.events if name=='vision'),
            prefill_cuda_ms=durations[0] if durations else None,
            decode_forward_cuda_ms=sum(durations[1:]),decode_forward_calls=max(0,len(durations)-1),
            decode_forward_mean_cuda_ms=float(np.mean(durations[1:])) if len(durations)>1 else None,
            language_generate_wall_ms=self.generation_ms,
            first_token_ready_from_lm_generate_ms=self.first_token_ready_ms,
            grammar_setup_cpu_ms=sum(self.constraint_ms),grammar_callbacks_cpu_ms=sum(self.callback_ms),
            grammar_callback_calls=len(self.callback_ms),generated_tokens_including_special=self.generated_tokens,
            input_sequence_tokens=self.forwards[0][2],visual_context_tokens=visual_tokens,
            text_and_special_tokens=self.forwards[0][2]-visual_tokens)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode',required=True);p.add_argument('--frame',type=int,default=30)
    p.add_argument('--output',required=True);p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--model',default='/data/ticvla/models/InternVL3-1B')
    p.add_argument('--profile',default='/data/ticvla/models/warehouse-realtime/selected.json')
    p.add_argument('--array-compression-level',type=int,default=0,choices=range(10),
                   help='0 matches the live mapper; 6 matches the older single-frame entry')
    args=p.parse_args()
    if args.repeats<1: p.error('repeats must be positive')
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    profile=json.loads(Path(args.profile).read_text())
    def progress(text):print(text,flush=True)
    progress('Loading fixed segmenter, base InternVL and mobile OCR; timings separate from warm inference.')
    seg,seg_load=timed(lambda:FixedSegmenter(profile['weights']))
    model,vlm_load=timed(lambda:Selector(args.model))
    reader,ocr_load=timed(MobileReader)
    obs=source_from_frame(args.episode,args.frame)
    warm_store=MapStore(root/'warmup/map',array_compression_level=args.array_compression_level);source=warm_store.add_observation(obs)
    warm_batch,seg_warm=timed(lambda:seg.predict(source))
    req=selection.candidate_request(source,warm_batch,root/'warmup/qa')
    warm_qa,qa_warm=timed(lambda:selection.predict(model,source,req))
    warm_ocr,ocr_warm=timed(lambda:reader.predict(source))
    write_json(root/'warmup/response.json',warm_qa);warm_store.close()
    progress('Warmup complete. Measuring production pipeline on one source frame in fresh maps.')
    rows_out=[]
    for index in range(args.repeats):
        folder=root/f'current-{index:02d}';store=MapStore(folder/'map',array_compression_level=args.array_compression_level)
        t=time.perf_counter();stages={}
        source,stages['geometry_ms']=timed(lambda:store.add_observation(obs))
        batch,stages['segmentation_ms']=timed(lambda:seg.predict(source))
        def projections():
            batch['prepared_projections']={i:store.prepare_projection(source['image_id'],x['mask']) for i,x in enumerate(batch['instances'])}
        _,stages['candidate_projection_ms']=timed(projections)
        request,stages['candidate_request_and_overlay_ms']=timed(lambda:selection.candidate_request(source,batch,folder/'qa'))
        batch['candidate_request']=request
        with VLMTrace(model) as trace:
            qa,stages['vlm_total_ms']=timed(lambda:selection.predict(model,source,request))
        admitted,stages['admission_ms']=timed(lambda:consume_masks(store,source,batch,selection.selection_result(qa)))
        ocr,stages['ocr_ms']=timed(lambda:reader.predict(source))
        binding,stages['text_binding_ms']=timed(lambda:consume_text(store,source,batch,admitted,ocr))
        view,stages['postprocess_ms']=timed(store.landmark_view)
        from DynaNav.warehouse_loop.realtime_mapping import render
        from DynaNav.warehouse_loop.live_mapping import LiveBEV
        def publish():
            render(folder,source,batch,admitted,view['landmarks'],selection.selection_result(qa),qa,
                {**binding,'source_image_id':source['image_id']})
            bev=LiveBEV();bev.add(store.points(source),source['T_odom_body']);bev.render(view['landmarks'],folder/'map.png')
            write_json(folder/'landmark-view.json',view)
        _,stages['render_publish_ms']=timed(publish)
        pipeline_ms=(time.perf_counter()-t)*1000
        def archive():
            (folder/'frame').mkdir(parents=True,exist_ok=True)
            save_frame(folder/'frame',source,batch,admitted,selection.selection_result(qa),ocr,binding)
            write_json(folder/'response.json',qa)
        _,stages['archive_ms']=timed(archive)
        data=dict(iteration=index,stages_ms=stages,vlm=trace.result(),pipeline_to_view_ms=pipeline_ms,
                  candidates=len(batch['instances']),valid=qa['valid'],output_tokens=qa.get('token_count'),
                  selected=qa.get('parsed'),accepted_instances=sum(x['result']['status']=='accepted' for x in admitted),
                  text_updates=len(binding['updates']),map_version=store.version)
        rows_out.append(data);write_json(folder/'timings.json',data);store.close()
        progress(f'current {index}: {pipeline_ms:.1f} ms, VLM {stages["vlm_total_ms"]:.1f} ms')
    progress('Diagnostic shorter prompt: same two images, same candidate table/schema/model; not deployed.')
    short=[]
    with patch.object(selection,'question',compact_question):
        _,short_warm=timed(lambda:selection.predict(model,source,request))
        for index in range(args.repeats):
            with VLMTrace(model) as trace:
                qa,total=timed(lambda:selection.predict(model,source,request))
            data=dict(iteration=index,total_ms=total,vlm=trace.result(),valid=qa['valid'],output_tokens=qa.get('token_count'),selected=qa.get('parsed'))
            short.append(data);write_json(root/f'compact-{index:02d}/response.json',qa);write_json(root/f'compact-{index:02d}/timings.json',data)
    def stats(vals):return dict(p50=float(np.percentile(vals,50)),p95=float(np.percentile(vals,95)),min=float(min(vals)),max=float(max(vals)))
    report=dict(episode=args.episode,frame=args.frame,repeats=args.repeats,gpu=torch.cuda.get_device_name(0),
        python=platform.python_version(),torch=torch.__version__,profile=profile,model=args.model,
        array_compression_level=args.array_compression_level,
        startup_ms=dict(segmenter_load=seg_load,vlm_load=vlm_load,ocr_load=ocr_load),
        warmup_ms=dict(segmentation=seg_warm,vlm=qa_warm,ocr=ocr_warm,compact_vlm=short_warm),
        stages_ms={k:stats([r['stages_ms'][k] for r in rows_out]) for k in rows_out[0]['stages_ms']},
        pipeline_to_view_ms=stats([r['pipeline_to_view_ms'] for r in rows_out]),
        vlm_current={k:stats([r['vlm'][k] for r in rows_out]) for k in rows_out[0]['vlm'] if all(r['vlm'][k] is not None for r in rows_out)},
        vlm_compact={k:stats([r['vlm'][k] for r in short]) for k in short[0]['vlm'] if all(r['vlm'][k] is not None for r in short)},
        compact_total_ms=stats([r['total_ms'] for r in short]),current=rows_out,compact=short,
        scope=['Real warmed models; same source image repeated, not a scene-wide p95.',
               'Serialized isolated replay; excludes Isaac rendering, capture, IPC, queues and browser paint.',
               'Fresh map each repeat; model initialization and warmup excluded from stage times.',
               'CUDA event windows and CPU callback timings are nested, not additive wall stages.',
               'First-token timing starts at language_model.generate, after vision features.',
               'Compact prompt is a diagnostic only, no production prompt change or quality equivalence claim.'])
    write_json(root/'report.json',report)
    progress('Report: '+str(root/'report.json'))


if __name__=='__main__':main()
