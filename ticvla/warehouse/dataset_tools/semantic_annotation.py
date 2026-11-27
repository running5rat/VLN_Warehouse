"""Small causal semantic-teacher jobs; geometric labels never enter teacher input.

Proposals are unreviewed, including when fed recursively to the next request.
Agreement with an incomplete human annotation is not semantic accuracy.
"""
import base64
import copy
import html
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from ..text_protocol import LOCAL_MAX_CHARACTERS, COMPLETION_EVENTS, completion_evidence
from ..schema import file_sha256, fingerprint
from ..structured_io import parse_structured, response_grammar
from ..local_context import local_memory, local_context
from .core import (read_json, write_json, sample_context, tagged_text, verify_observation_files,
                   load_episode, frame_binding)
from .teacher import teacher_input, TEACHER_INSTRUCTION
from ..compact_input import VERSION, model_context, image_label, render_context, audit_context
from .teacher_transport import open_teacher_request, transport_policy


def write_bundle(output, samples, *, source, context_mode='recursive', episode_review=False):
    if context_mode not in ('recursive', 'gold_context'):
        raise ValueError('Unknown context mode')
    if not 1 <= len(samples) <= (128 if episode_review else 6):
        raise ValueError('One to six samples per pilot; explicit episode review permits up to 128')
    tasks = {s['observation']['task']['task_id'] for s in samples}
    instructions = {s['observation']['task']['instruction'] for s in samples}
    times = [s['observation']['t_obs'] for s in samples]
    if len(tasks) != 1 or len(instructions) != 1 or any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Pilot must be one task with chronological distinct sources')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    records = []
    for i, sample in enumerate(samples):
        path = output / f'{i:08d}.json'; write_json(path, sample)
        records.append(dict(path=path.name, sha256=file_sha256(path), image_id=sample['observation']['image_id']))
    manifest = dict(schema_version=1, protocol='semantic-episode-review-v1' if episode_review else 'semantic-teacher-pilot-v1', source=source,
        context_mode=context_mode, records=records, optimizer_updates_permitted=False,
        teacher_fields=['landmark', 'local', 'route'], geometric_fields_generated_by_teacher=False)
    write_json(output/'manifest.json',manifest)
    return manifest


def prepare_manual(project, output, indices=(0,3,4,6,8,9), context_mode='recursive'):
    project = Path(project); info = read_json(project/'project.json')
    if not info.get('human_episode_approved'):
        raise ValueError('Manual baseline must already be human approved')
    meta, frames = load_episode(info['source_episode'])
    indices = list(indices)
    if indices != sorted(set(indices)) or not 1 <= len(indices) <= 6:
        raise ValueError('Select at most six distinct chronological keyframes')
    samples = []
    for i in indices:
        s = sample_context(project, i); a = s['annotation']
        if not a or a['parent_revision'] != s['parent_revision']:
            raise ValueError('Missing/stale annotation')
        if frame_binding(meta, frames[s['provenance']['frame_index']]) != a['source_binding']:
            raise ValueError('Original capture changed after human approval')
        verify_observation_files(s)
        geometry = {k:copy.deepcopy(a[k]) for k in ('reference_xy','reference_mask','waypoint','reference_state')}
        geometry.update(source='approved_human_demonstration', approved=True,
                        annotation_revision=a['revision_id'], supervision_for_teacher=False)
        record = dict(observation=copy.deepcopy(s['observation']), provenance=copy.deepcopy(s['provenance']),
            geometric_supervision=geometry, evaluation_only=dict(human_text=a['parsed'],
            human_revision=a['revision_id'], reference_is_not_exhaustive_object_inventory=True))
        # Current text belongs only to evaluation. Raw task/source hashes stay intact.
        record['provenance']['project_sha256'] = file_sha256(project/'project.json')
        samples.append(record)
    return write_bundle(output,samples,source='manual-static-approved',context_mode=context_mode)


def validate_sample(sample):
    o = sample['observation']
    if not o['task']['instruction'].strip() or re.search(r'[\u3400-\u9fff]',o['task']['instruction']):
        raise ValueError('Pilot requires an English instruction')
    if not o['rgb_history'] or o['rgb_history'][-1]['image_id'] != o['image_id']:
        raise ValueError('Missing current source RGB')
    if any(h['t_obs'] > o['t_obs'] for h in o['rgb_history']):
        raise ValueError('Future image in teacher input')
    if any(e.get('t_obs', float('inf')) > o['t_obs'] for e in o['execution']['events']):
        raise ValueError('Future execution fact')
    cutoff = o.get('mapview',{}).get('source_until')
    if cutoff is not None and cutoff > o['t_obs']:
        raise ValueError('Future map')
    for h in o['rgb_history']:
        if file_sha256(h['path']) != h['sha256']:
            raise ValueError('Source image changed')
    view=o.get('mapview',{})
    for field in ('image','arrays'):
        if view.get(field) and view.get(field+'_sha256') != file_sha256(view[field]):
            raise ValueError('Map changed')
    # Reject leaks even if a future editor adds a field to an allowed namespace.
    def walk(v):
        if isinstance(v,dict):
            if {'future','future_trajectory','reference_xy','waypoint','goal_pose_gt','oracle','cot','supervision','evaluation_only'} & v.keys():
                raise ValueError('Privileged field in teacher observation')
            for x in v.values(): walk(x)
        elif isinstance(v,list):
            for x in v: walk(x)
    walk(o)
    return sample


def assemble_answer(answer, sample):
    if not isinstance(answer,dict) or set(answer) != {'landmark','local','route'}:
        raise ValueError('Teacher must return exactly landmark, local, route; no geometric fields')
    if not isinstance(answer['local'], str) or not 1 <= len(answer['local']) <= LOCAL_MAX_CHARACTERS:
        raise ValueError(f'Local must contain 1 to {LOCAL_MAX_CHARACTERS} characters')
    ids=[h['image_id'] for h in sample['observation']['rgb_history']]
    geometry=sample['geometric_supervision']
    text=tagged_text(answer['landmark'],answer['local'],answer['route'],geometry['waypoint'])
    compact=isinstance(answer['route'],str)
    parsed=parse_structured(text,ids,structured_route=not compact,compact_route=compact,
        current_image_id=sample['observation']['image_id'])
    grammar=response_grammar(ids,structured_route=not compact,compact_route=compact,
        categories_only=all(isinstance(x,str) for x in answer['landmark']))
    for ch in text:
        if ch not in grammar.get_allowed_characters():
            raise ValueError('Answer violates deployment English generation grammar')
        grammar=grammar.add_character(ch)
    if not grammar.can_end(): raise ValueError('Truncated protocol')
    return dict(text=text,parsed=parsed,approved=False,semantic_review='required',
        route_protocol='compact-v2' if compact else 'structured-v1',
        source_image_id=sample['observation']['image_id'],source_binding=sample['provenance']['source_binding'],
        landmark_protocol=('current-categories-v1' if all(isinstance(x,str) for x in answer['landmark']) else 'legacy-source-records'),
        geometry=copy.deepcopy(geometry),field_sources=dict(landmark='teacher',
        landmark_image_id=('request_current_source' if all(isinstance(x,str) for x in answer['landmark']) else 'teacher_source_reference'),local='teacher',route='teacher',
        waypoint=geometry['source'],reference_xy=geometry['source']))


def agreement(answer, sample):
    """Deterministic diagnostics only. No model grading itself or invented accuracy."""
    obs=sample['observation']; route=answer['route']
    from ..navigation.route_protocol import route_completed, remaining_instruction
    events={e['kind'] for e in obs['execution']['events']}
    flags=[]
    completion_token=isinstance(route,str) and bool(re.search(r'\bDONE\b',route))
    if completion_token and route != 'DONE':
        flags.append('noncanonical_completion_marker')
    if (route_completed(route) or completion_token) and not COMPLETION_EVENTS <= events:
        flags.append('completed_without_recorded_target_arrival_and_stop_evidence')
    result=dict(rule_flags=flags,semantic_accuracy=None,semantic_review='required')
    human=sample.get('evaluation_only',{}).get('human_text')
    if human is None:
        result.update(human_reference_available=False,reason='No independent Landmark/Local/Route truth')
        return result
    def labels(v): return {(x['image_id'], ' '.join(x['category'].lower().split())) for x in v}
    expected,actual=labels(human['landmark']),labels(answer['landmark'])
    result.update(human_reference_available=True,
        landmark_agreement=dict(matched=len(expected&actual),human_only=len(expected-actual),teacher_only=len(actual-expected),
            exact=expected==actual,interpretation='Agreement with selected human objects, not exhaustive detection precision/recall'),
        local_exact=answer['local']==human['local'],
        route_action_exact=(route['current_subgoal']['action']==human['route']['current_subgoal']['action']
                            if isinstance(route,dict) and isinstance(human['route'],dict) else None),
        route_subgoal_exact=(route['current_subgoal']==human['route']['current_subgoal']
                            if isinstance(route,dict) and isinstance(human['route'],dict) else None),
        route_status_exact=(route['status']==human['route']['status']
                            if isinstance(route,dict) and isinstance(human['route'],dict) else None),
        completion_claim_exact=route_completed(route)==route_completed(human['route']),
        remaining_instruction_exact=remaining_instruction(route)==remaining_instruction(human['route']),
        text_exact_is_not_semantic_correctness=True)
    return result


def build_request(sample, cfg):
    instruction = cfg.get('teacher_instruction', TEACHER_INSTRUCTION)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('Teacher instruction must be a nonempty string')
    instruction_hash = hashlib.sha256(instruction.encode()).hexdigest()
    if cfg.get('teacher_instruction_sha256', instruction_hash) != instruction_hash:
        raise ValueError('Frozen teacher instruction hash mismatch')
    audit, images=teacher_input(sample)
    context=model_context(sample["observation"])
    task_input_mode=cfg.get('task_input_mode','persistent_instruction')
    if task_input_mode not in ('persistent_instruction','route_only'):
        raise ValueError('Unknown teacher task_input_mode')
    if task_input_mode=='route_only':
        # The episode loop initializes Route once from the original task, then
        # carries the latest valid model Route. Keep the contract in audit only.
        context.pop('instruction')
    user_text = 'Return the requested annotation as JSON.\n' + render_context(context)
    if cfg.get('completion_evidence_input'):
        user_text += '\nCompletion evidence: ' + completion_evidence(sample['observation'])
    content=[dict(type='input_text',text=user_text)]
    descriptions=[]
    for i,path in enumerate(images):
        p=Path(path);mime=mimetypes.guess_type(str(p))[0]
        if mime not in ('image/png','image/jpeg','image/webp'): raise ValueError('Unsupported image MIME')
        label = image_label(sample['observation'],i) if i < len(audit['source_rgb']) else 'Causal observed MapView'
        if path == sample['observation'].get('perception_candidates', {}).get('overlay'):
            label = 'Current RGB detector boxes and IDs; fallible proposals, not map evidence'
        elif audit.get('semantic_input') and i >= len(audit['source_rgb']):
            label = 'Previous committed semantic map reprojected into current source body frame'
        image_bytes=p.read_bytes();encoding='original_file'
        if cfg.get('rgb_jpeg_quality') is not None and i<len(audit['source_rgb']):
            from PIL import Image
            quality=int(cfg['rgb_jpeg_quality'])
            if not 1<=quality<=100:raise ValueError('JPEG quality must be in 1..100')
            with Image.open(p) as source:
                buf=io.BytesIO();source.convert('RGB').save(buf,format='JPEG',quality=quality,subsampling=0)
            image_bytes=buf.getvalue();mime='image/jpeg';encoding=f'jpeg_quality_{quality}_444_original_resolution'
        content.extend([dict(type='input_text',text=label),dict(type='input_image',detail='high',
            image_url=f'data:{mime};base64,'+base64.b64encode(image_bytes).decode())])
        descriptions.append(dict(path=str(p),sha256=file_sha256(p),label=label,mime_type=mime,
            transmitted_sha256=hashlib.sha256(image_bytes).hexdigest(),transmitted_bytes=len(image_bytes),encoding=encoding))
    request=dict(model=cfg.get('model'),instructions=instruction,store=False,
        max_output_tokens=int(cfg.get('max_output_tokens',2048)),text=dict(format=dict(type='json_object')),
        input=[dict(role='user',content=content)])
    if cfg.get('reasoning_effort'): request['reasoning']=dict(effort=cfg['reasoning_effort'])
    api=cfg.get('api','responses')
    if api=='chat_completions':
        chat_content=[]
        for item in content:
            if item['type']=='input_text':chat_content.append(dict(type='text',text=item['text']))
            else:chat_content.append(dict(type='image_url',image_url=dict(url=item['image_url'])))
        request=dict(model=cfg.get('model'),messages=[dict(role='system',content=instruction),
            dict(role='user',content=chat_content)],max_tokens=int(cfg.get('max_output_tokens',2048)),
            response_format=dict(type='json_object'))
        # Provider-specific parameters must be explicit in the frozen config.
        for key in ('thinking','enable_thinking','thinking_budget','reasoning_effort'):
            if key in cfg:request[key]=cfg[key]
    elif api!='responses':raise ValueError('Unsupported teacher API protocol')
    summary=dict(model=cfg.get('model'),instructions=instruction,context=context,images=descriptions,
        task_input_mode=task_input_mode,
        transport=transport_policy(cfg.get('base_url','https://api.openai.com/v1'),cfg),
        teacher_instruction_sha256=instruction_hash,
        teacher_template_version=cfg.get('teacher_template_version','legacy-shared-rules'),
        audit_context=audit,input_format=VERSION, user_text=user_text,
        api=api,max_output_tokens=int(cfg.get('max_output_tokens',2048)),response_format='json_object_then_deployment_grammar_validation',
        reasoning=request.get('reasoning',request.get('thinking')),store=request.get('store'),request_sha256=fingerprint(request),
        user_text_prefix='Return the requested annotation as JSON.\n')
    return request,summary


def send_request(request, cfg):
    base=cfg.get('base_url','https://api.openai.com/v1').rstrip('/')
    url=urlparse(base)
    if url.scheme != 'https' or url.username or url.password or url.query or url.fragment:
        from .teacher_budget import RequestNotSent
        raise RequestNotSent('Use an HTTPS API base without credentials/query/fragment')
    transport_policy(base, cfg)  # Validate before reserving a paid request.
    key=os.environ[cfg.get('api_key_env','OPENAI_API_KEY')]
    endpoint='/chat/completions' if cfg.get('api')=='chat_completions' else '/responses'
    req=urllib.request.Request(base+endpoint,data=json.dumps(request).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    def send():
        with open_teacher_request(req,cfg) as response:
            return json.load(response)
    if cfg.get('budget_file'):
        from .teacher_budget import budgeted_call
        return budgeted_call(request,cfg,send)
    return send()


def parse_api_response(raw):
    if 'choices' in raw:
        if len(raw['choices'])!=1 or raw['choices'][0].get('finish_reason')!='stop':
            raise ValueError('Incomplete or failed chat response')
        message=raw['choices'][0]['message']
        if message.get('refusal') or not isinstance(message.get('content'),str):
            raise ValueError('Teacher refused or did not return text')
        return message['content']
    if raw.get('status') != 'completed': raise ValueError('Incomplete or failed API response')
    contents=[c for x in raw.get('output',[]) if x.get('type')=='message' for c in x.get('content',[])]
    if any(c.get('type')=='refusal' for c in contents): raise ValueError('Teacher refused')
    text=''.join(c['text'] for c in contents if c.get('type')=='output_text')
    # Retain text even when JSON is invalid; no repair or second generation.
    return text


def render_report(output, report):
    output=Path(output)
    parts=['<!doctype html><meta charset="utf-8"><title>Semantic teacher pilot</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:24px auto;background:#17232f;color:#eee}a{color:#8cd6ff}pre{white-space:pre-wrap;background:#243749;padding:12px}img{max-width:48%}section{border-top:1px solid #999;padding:15px}</style>',
        '<h1>Landmark / Local / Route teacher pilot</h1><p>'+html.escape(report['status'])+'</p>',
        '<p>Proposals are unreviewed. Automatic rules and text agreement do not measure full semantic accuracy. No teacher-generated waypoint or reference line.</p>',
        '<a href="report.json">Report JSON</a><pre>'+html.escape(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))+'</pre>']
    for row in report['samples']:
        folder=output/row['folder'];parts.append('<section><h2>'+html.escape(row['source_image_id'])+'</h2>')
        if (folder/'input.json').exists():
            inp=read_json(folder/'input.json');current=inp['images'][len(audit_context(inp)['source_rgb'])-1]['path']
            suffix=Path(current).suffix;dest=folder/('source'+suffix)
            if not dest.exists(): dest.write_bytes(Path(current).read_bytes())
            parts.append('<img src="'+row['folder']+'/'+dest.name+'"><p><a href="'+row['folder']+'/input.json">Teacher input</a></p>')
        for name in ('raw.txt','proposal.json','comparison.json'):
            if (folder/name).exists():parts.append('<details open><summary>'+name+'</summary><pre>'+html.escape((folder/name).read_text())+'</pre></details>')
        parts.append('<pre>'+html.escape(json.dumps(row,indent=2))+'</pre></section>')
    (output/'index.html').write_text(''.join(parts))


def run_pilot(bundle, config, output, prepare_only=False):
    from .annotation_policy import failure_policy, must_stop, gap_metadata
    bundle=Path(bundle);manifest=read_json(bundle/'manifest.json');cfg=read_json(config)
    policy=failure_policy(cfg)
    if manifest.get('protocol')!='semantic-teacher-pilot-v1': raise ValueError('Invalid bundle protocol')
    if not 1 <= len(manifest['records']) <= 6: raise ValueError('At most six pilot requests')
    mode=manifest['context_mode']
    if mode not in ('recursive','gold_context'): raise ValueError('Invalid context mode')
    samples=[]
    for item in manifest['records']:
        p=bundle/item['path']
        if not p.resolve().is_relative_to(bundle.resolve()) or file_sha256(p)!=item['sha256']:
            raise ValueError('Bundle source mismatch')
        samples.append(validate_sample(read_json(p)))
    times=[s['observation']['t_obs'] for s in samples]
    if any(b<=a for a,b in zip(times,times[1:])) or len({s['observation']['task']['instruction'] for s in samples})!=1:
        raise ValueError('Invalid task sequence')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    ready=bool(cfg.get('model') and os.environ.get(cfg.get('api_key_env','OPENAI_API_KEY')))
    report=dict(status='running',actual_api_calls=0,model=cfg.get('model'),context_mode=mode,
        bundle=str(bundle),bundle_sha256=file_sha256(bundle/'manifest.json'),config_sha256=file_sha256(config),
        samples=[],semantic_accuracy=None,human_approved=False,training=False,annotation_failure_policy=policy,
        waypoint_and_reference_teacher_generated=False,expected_samples=len(samples))
    previous=None;failed=False;transport_failed=False;rejected=[]
    for i,original in enumerate(samples):
        sample=copy.deepcopy(original);obs=sample['observation'];folder=output/f'{i:08d}';folder.mkdir()
        row=dict(index=i,source_image_id=obs['image_id'],folder=folder.name)
        row['context_recovery']=dict(gap_metadata(previous,rejected),policy=policy) if mode=='recursive' else None
        if transport_failed:
            row.update(status='not_run_provider_error');report['samples'].append(row);continue
        if mode=='recursive':
            if failed:
                row.update(status='not_run_previous_proposal_failed');report['samples'].append(row);continue
            obs['route']=previous['parsed']['route'] if previous else obs['task']['instruction']
            obs['route_revision']=i
            obs['previous_local']=dict(valid=False,reason='no_previous_teacher_response')
            if previous and 'T_odom_body' in obs:
                memory=local_memory(previous['parsed']['local'],previous['observation'],origin='unreviewed_teacher_proposal',revision=previous['response_hash'])
                obs['previous_local']=local_context(memory,obs)
            elif previous:
                obs['previous_local']=dict(valid=False,reason='external_robot_base_pose_not_audited')
            obs['execution']['route_context_source']='previous_unreviewed_teacher_proposal' if previous else 'original_instruction'
        request,summary=build_request(sample,cfg)
        summary['request_ready']=bool(mode=='gold_context' or i==0 or previous or (ready and not prepare_only))
        summary['source_binding']=sample['provenance']['source_binding']
        write_json(folder/'input.json',summary)
        # Geometry and human labels are saved separately, never in request/summary.
        write_json(folder/'geometric-supervision.json',sample['geometric_supervision'])
        if not ready or prepare_only:
            row.update(status='prepared_no_api_call',recursive_context_pending=(mode=='recursive' and i>0))
            report['samples'].append(row);continue
        stage='api_request'
        started=time.monotonic();report['actual_api_calls']+=1
        row['status']='request_started';report['samples'].append(row);write_json(output/'report.json',report)
        try:
            raw=send_request(request,cfg);write_json(folder/'response.json',raw)
            row['usage']=raw.get('usage')
            if cfg.get('budget_file'):
                from .teacher_budget import usage_charge
                row['cost_estimate_usd']=usage_charge(raw,cfg)
                row['cost_kind']=cfg.get('cost_kind','undiscounted_rate_estimate_not_invoice')
                if cfg.get('native_currency'):
                    native_cfg=dict(cfg,usd_per_million_tokens=cfg['native_per_million_tokens'])
                    row['cost_native_estimate']=usage_charge(raw,native_cfg)
                    row['native_currency']=cfg['native_currency']
                    row['budget_accounted_usd']=row.pop('cost_estimate_usd')
            # Archive partial NTP output even when finish/status validation fails.
            partial=(raw.get('choices') or [{}])[0].get('message',{}).get('content')
            if partial is None:
                partial=''.join(c.get('text','') for item in raw.get('output',[]) for c in item.get('content',[]) if c.get('type')=='output_text')
            if isinstance(partial,str):(folder/'raw.txt').write_text(partial)
            stage='response_validation'
            text=parse_api_response(raw);(folder/'raw.txt').write_text(text)
            answer=json.loads(text);proposal=assemble_answer(answer,sample)
            comparison=agreement(proposal['parsed'],sample)
            proposal.update(response_sha256=file_sha256(folder/'response.json'),model_returned=raw.get('model'),
                rule_checks_passed=not comparison['rule_flags'],context_mode=mode)
            write_json(folder/'proposal.json',proposal);write_json(folder/'comparison.json',comparison)
            row.update(status='format_valid_unreviewed',usage=raw.get('usage'),model_returned=raw.get('model'),
                cost=None,comparison=comparison)
            # Unsupported completion is archived, never propagated as accepted progress.
            if comparison['rule_flags']:
                failed=must_stop(cfg,ValueError('Unsupported completion'),stage)
                row.update(status='rule_rejected_unreviewed',label_eligible=False,
                    failure_action='stop_run' if failed else 'skip_frame_keep_last_valid')
                rejected.append(obs['image_id'])
            else:
                previous=dict(parsed=proposal['parsed'],observation=obs,response_hash=proposal['response_sha256'])
                rejected=[]
        except urllib.error.HTTPError as exc:
            body=exc.read();key=os.environ.get(cfg.get('api_key_env','OPENAI_API_KEY'),'')
            if key:body=body.replace(key.encode(),b'[REDACTED]')
            (folder/'http-error.txt').write_bytes(body);row.update(status='failed',reason=f'HTTP {exc.code}');failed=True
            transport_failed=True
            row.update(failure_stage=stage,failure_action='stop_run',label_eligible=False)
        except (ValueError,KeyError,TypeError,OSError) as exc:
            failed=must_stop(cfg,exc,stage)
            row.update(status='failed',reason=str(exc),failure_stage=stage,label_eligible=False,
                failure_action='stop_run' if failed else 'skip_frame_keep_last_valid')
            rejected.append(obs['image_id'])
            if getattr(exc,'request_sent',True) is False:
                report['actual_api_calls']-=1
                row['status']='not_sent_preflight_rejected'
            if stage!='response_validation' or isinstance(exc,OSError) or getattr(exc,'request_sent',True) is False:
                transport_failed=True
        row['elapsed_seconds']=time.monotonic()-started
        write_json(output/'report.json',report)
    report['status']=('prepared_only' if prepare_only else 'blocked_missing_model_or_credentials') if not ready or prepare_only else 'completed_with_unreviewed_proposals'
    report['format_valid_count']=sum(r['status'] in ('format_valid_unreviewed','rule_rejected_unreviewed') for r in report['samples'])
    report['rule_pass_count']=sum(r['status']=='format_valid_unreviewed' for r in report['samples'])
    report['real_teacher_interface_pass']=report['actual_api_calls']>0 and report['rule_pass_count']>0
    report['all_samples_completed']=report['rule_pass_count']==len(samples)
    report['format_success_fraction_all_selected']=report['format_valid_count']/len(samples) if report['actual_api_calls'] else None
    comparisons=[r['comparison'] for r in report['samples'] if r.get('comparison',{}).get('human_reference_available')]
    report['human_comparison_count']=len(comparisons)
    report['agreement_counts']={key:sum(bool(c.get(key)) for c in comparisons) for key in
        ('local_exact','route_action_exact','route_subgoal_exact','route_status_exact','remaining_instruction_exact')}
    report['agreement_interpretation']='Counts over human_comparison_count; lexical/action agreement, not full semantic accuracy. Failed and unrun samples remain in samples and all-selected format denominator.'
    write_json(output/'report.json',report);render_report(output,report)
    return report
