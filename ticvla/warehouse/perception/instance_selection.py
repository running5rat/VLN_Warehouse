"""One mapping-only visual question over immutable, source-bound detections."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ..schema import fingerprint, file_sha256

PURPOSES = ('identity_candidate', 'spatial_reference')


def candidate_request(source, batch, directory):
    if batch['source_image_id'] != source['image_id'] or batch['rgb_hash'] != source['rgb_hash']:
        raise ValueError('Candidate source mismatch')
    if file_sha256(source['rgb']) != source['rgb_hash']:
        raise ValueError('Source RGB changed')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    canvas = Image.open(source['rgb']).convert('RGB')
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 22)
    except OSError:
        font = ImageFont.load_default()
    candidates = []
    for i, inst in enumerate(batch['instances']):
        mask = np.asarray(inst['mask'], dtype=bool)
        box = np.asarray(inst['bbox'], dtype=float)
        if mask.shape != (h, w) or box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('Invalid candidate dimensions')
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError('Candidate bbox outside RGB')
        item = dict(instance_id=f'd{i}', category=inst['category'],
                    bbox=box.tolist(), bbox_normalized=(box / [w,h,w,h]).tolist(),
                    confidence=inst['confidence'],
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
        candidates.append(item)
        draw.rectangle(box.tolist(), outline='cyan', width=2)
        text = item['instance_id']
        xy = (min(max(0,box[0]), w-65), max(0,box[1]-27))
        draw.rectangle(draw.textbbox(xy, text, font=font), fill='black')
        draw.text(xy, text, fill='yellow', font=font)
    if len(candidates) > 100:
        raise ValueError('More than 100 candidates')
    overlay = directory / 'candidates.png'
    canvas.save(overlay)
    identity = dict(source_image_id=source['image_id'], rgb_hash=source['rgb_hash'],
                    depth_hash=source.get('depth_hash'),
                    source_geometry={k:source.get(k) for k in ('K','T_body_camera','T_odom_body','odom_epoch','t_obs')},
                    provenance=batch['provenance'], candidates=candidates)
    request = dict(**identity, batch_id=fingerprint(identity), overlay=str(overlay),
                   overlay_hash=file_sha256(overlay), possibly_truncated=batch.get('possibly_truncated',False))
    (directory / 'request.json').write_text(json.dumps(request,ensure_ascii=False,indent=2))
    return request


def question(request):
    table = [{k:c[k] for k in ('instance_id','category','bbox_normalized')} for c in request['candidates']]
    return (
        'Image-1: <image>\nImage-2: <image>\n'
        'Build a semantic map from this observation. Image-1 is the original RGB; '
        'Image-2 shows detector boxes with instance IDs. The candidate table below '
        'uses those IDs and normalized coordinates. Detector labels are suggestions, not truth. '
        'Which candidate instances are worth recording as map observations or inspecting '
        'for distinguishing identity information? Prioritize markers or surfaces that may '
        'carry text or symbols, and facilities with distinctive appearance or spatial context. '
        'Do not select ordinary repeated structures merely because they are large or stationary. '
        'Do not reject repeated markers: their identifiers may differ. '
        'Do not read or output any identifier text. Do not claim global uniqueness or permanent staticness. '
        'Select only visible candidate IDs; identity_candidate means inspect for identity evidence, '
        'spatial_reference means useful distinctive spatial context. '
        'Report useful visible object categories missing from the detections in missed_categories '
        'for diagnosis only. Empty lists are allowed. Return JSON with selected_instances '
        '(each has instance_id and purpose) and missed_categories. No explanation.\n'
        'Candidates: '+json.dumps(table,separators=(',',':'))
    )


def parse_answer(text, request):
    d = json.loads(text)
    if not isinstance(d,dict) or set(d) != {'selected_instances','missed_categories'}:
        raise ValueError('Invalid instance selection fields')
    if not isinstance(d['selected_instances'],list) or not isinstance(d['missed_categories'],list):
        raise ValueError('Selection fields must be lists')
    ids={c['instance_id'] for c in request['candidates']}; seen=set()
    for v in d['selected_instances']:
        if not isinstance(v,dict) or set(v) != {'instance_id','purpose'}:
            raise ValueError('Invalid selected instance')
        if v['instance_id'] not in ids or v['instance_id'] in seen or v['purpose'] not in PURPOSES:
            raise ValueError('Invalid or duplicate candidate ID/purpose')
        seen.add(v['instance_id'])
    if len(d['missed_categories']) > 5 or any(not isinstance(x,str) or not x.strip() or len(x)>80 for x in d['missed_categories']):
        raise ValueError('Invalid missed categories')
    return d


def predict(selector, source, request):
    import torch
    from lmformatenforcer import JsonSchemaParser, RegexParser
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn
    from ..vlm import image_tensor
    if request['source_image_id'] != source['image_id'] or request['rgb_hash'] != file_sha256(source['rgb']):
        raise ValueError('VLM request source mismatch')
    if file_sha256(request['overlay']) != request['overlay_hash']:
        raise ValueError('Candidate overlay changed')
    ids=[c['instance_id'] for c in request['candidates']]
    item=dict(type='object',properties={'instance_id':dict(type='string',enum=ids or ['__no_candidate__']),
              'purpose':dict(type='string',enum=list(PURPOSES))},required=['instance_id','purpose'],additionalProperties=False)
    schema=dict(type='object',properties={
        'selected_instances':dict(type='array',items=item,maxItems=len(ids)),
        'missed_categories':dict(type='array',maxItems=5,items=dict(type='string',minLength=1,maxLength=80))},
        required=['selected_instances','missed_categories'],additionalProperties=False)
    prompt=question(request); start=time.time(); model=selector.model
    result=dict(selection_mode='instances',request_id=fingerprint({'batch':request['batch_id'],'prompt':prompt})[:24],
                source_image_id=source['image_id'],rgb_hash=source['rgb_hash'],batch_id=request['batch_id'],
                source_time=source['t_obs'],valid=False,parsed=None,error=None,llm_calls=1,
                structured_inference_executed=False,navigation_task_included=False,prompt=prompt,
                max_new_tokens=2048,format_constraint='JSON schema; one generation; two 448px images',
                input_images=[source['rgb'],request['overlay']])
    previous=model.backbone.system_message
    try:
        model.backbone.system_message='You are a visual assistant selecting observations for a semantic map.'
        if model._tokenizer_data is None:
            model._tokenizer_data=build_token_enforcer_tokenizer_data(model.tokenizer)
        # lm-format-enforcer's installed array parser treats maxItems=0 as unset.
        # An empty detection batch must permit only [], while missed names remain model-generated.
        parser = JsonSchemaParser(schema) if ids else RegexParser(
            r'\{\s*"selected_instances"\s*:\s*\[\s*\]\s*,\s*"missed_categories"\s*:\s*\[\s*("[A-Za-z][A-Za-z0-9 -]{0,79}"(\s*,\s*"[A-Za-z][A-Za-z0-9 -]{0,79}"){0,4})?\s*\]\s*\}')
        result['format_constraint'] += '; empty candidates use empty-array regex' if not ids else ''
        decoding=dict(max_new_tokens=2048,do_sample=False,prefix_allowed_tokens_fn=build_transformers_prefix_allowed_tokens_fn(model._tokenizer_data,parser))
        with torch.inference_mode():
            pixels=torch.stack([image_tensor(p,model.cfg.image_size) for p in result['input_images']]).to(model.device,model.compute_dtype)
            answer=model.backbone.chat(model.tokenizer,pixels,prompt,decoding,num_patches_list=[1,1])
        result.update(text=answer,token_count=len(model.tokenizer.encode(answer,add_special_tokens=False)))
        if result['token_count'] >= 2048:
            raise ValueError('Instance selection truncated')
        result.update(parsed=parse_answer(answer,request),valid=True)
    except (ValueError,RuntimeError) as exc:
        result['error']=str(exc)
    finally:
        model.backbone.system_message=previous
    result.update(started=start,finished=time.time(),elapsed_s=time.time()-start)
    return result


def selection_result(qa):
    return dict(selection_mode='instances',status='valid' if qa['valid'] else 'failed',
                selected_instances=qa['parsed']['selected_instances'] if qa['valid'] else [],
                missed_categories=qa['parsed']['missed_categories'] if qa['valid'] else [],
                selected_categories=[],request_id=qa['request_id'],batch_id=qa['batch_id'],
                source_image_id=qa['source_image_id'],rgb_hash=qa['rgb_hash'],source_time=qa['source_time'])


def validate_selection(source,batch,selection):
    request=batch['candidate_request']
    if any(selection.get(k) != request.get(k) for k in ('source_image_id','rgb_hash','batch_id')):
        raise ValueError('Instance selection source/batch mismatch')
    if selection['source_image_id'] != source['image_id'] or selection['rgb_hash'] != source['rgb_hash']:
        raise ValueError('Instance selection observation mismatch')
    for i,inst in enumerate(batch['instances']):
        c=request['candidates'][i]
        if c['category']!=inst['category'] or c['mask_sha256']!=hashlib.sha256(np.asarray(inst['mask'],dtype=bool).tobytes()).hexdigest() or c['bbox']!=list(inst['bbox']):
            raise ValueError('Candidate changed after selection')
    parse_answer(json.dumps({'selected_instances':selection['selected_instances'],'missed_categories':selection.get('missed_categories',[])}),request)
