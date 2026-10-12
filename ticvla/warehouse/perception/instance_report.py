"""Paired A/B/C target audit and reviewable failure images, truth stays here."""
import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .common import polygon_mask
from .evaluate import reading_key
from .realtime_report import overlay
from .report import render_persistent_maps
from ..mapping.semantic_frame import write_json


def match(targets,masks,truths,instances,compatible=False):
    pairs=[]
    for ti,t in enumerate(targets):
        for pi,m in enumerate(masks):
            if compatible and t['category']=='shelf_marker' and instances[pi]['category'] not in ('sign','label'):continue
            iou=float((truths[ti]&m).sum()/max(1,(truths[ti]|m).sum()))
            if iou>=.5:pairs.append((iou,ti,pi))
    hits={};used=set()
    for iou,ti,pi in sorted(pairs,reverse=True):
        if ti not in hits and pi not in used:hits[ti]=(pi,iou);used.add(pi)
    return hits


def reading(instance):
    texts=instance.get('texts',[])
    return ' | '.join(x['normalized'] for x in sorted(texts,key=lambda t:(round(min(p[1] for p in t['polygon'])/12),min(p[0] for p in t['polygon']))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--annotations',type=Path,default=Path('/data/ticvla/outputs/landmark-comparison-20261010/evaluation'))
    p.add_argument('--skip-maps',action='store_true');a=p.parse_args();root=a.root
    annotations={r['key']:r['targets'] for r in json.loads((a.annotations/'annotations.json').read_text())['rows']}
    manifest=json.loads((root/'manifest.json').read_text());rows=[r for r in manifest['rows'] if r['variant']!='development']
    landmarks={arm:{} for arm in 'ABC'}
    for arm in 'ABC':
        for path in (root/arm/'maps').glob('*/observed-landmarks.json'):
            landmarks[arm].update({v['landmark_uid']:v for v in json.loads(path.read_text())})
    counts={arm:defaultdict(Counter) for arm in 'ABC'};cases=[];errors=[];uid_targets={arm:defaultdict(set) for arm in 'ABC'}
    qa_stats=Counter();qa_times=[];selected_total=Counter();selected_scored=Counter()
    for r in rows:
        key=r['key'];targets=annotations[key];truths=[np.load(a.annotations/t['mask_file'])['mask'].astype(bool) for t in targets]
        data={arm:json.loads((root/arm/key/'frame.json').read_text()) for arm in 'ABC'}
        masks=[np.load(root/'A'/key/f'mask-{i:03}.npz')['mask'].astype(bool) for i in range(len(data['A']['instances']))]
        for arm in 'BC':
            for i,m in enumerate(masks):
                if not np.array_equal(m,np.load(root/arm/key/f'mask-{i:03}.npz')['mask']):raise ValueError('Paired masks differ')
        hits=match(targets,masks,truths,data['A']['instances']);compatible=match(targets,masks,truths,data['A']['instances'],True)
        qa=json.loads((root/'qa'/key/'result.json').read_text());qa_stats['valid' if qa['valid'] else 'invalid']+=1;qa_times.append(qa['elapsed_s'])
        if not qa['valid']:qa_stats['error:'+qa['error']]+=1
        else:
            selected=qa['parsed']['selected_instances']
            qa_stats['selected_all_nonempty_batches']+=bool(masks) and len(selected)==len(masks)
            qa_stats['nonempty_batches']+=bool(masks)
            for item in selected:qa_stats['purpose:'+item['purpose']]+=1
        for arm in 'ABC':
            d=data[arm];selected_total[arm]+=sum(x['selected'] for x in d['rows'])
            matched_indices={pi for pi,_ in hits.values()}
            selected_scored[arm]+=sum(x['selected'] and i in matched_indices for i,x in enumerate(d['rows']))
            for text in (d.get('binding') or {}).get('texts',[]):
                owner=text.get('owner_instance')
                if owner is None:continue
                region=polygon_mask(text['polygon'],masks[owner].shape)
                candidates=[(float((region&g).sum()/max(1,region.sum())),ti) for ti,g in enumerate(truths)]
                if not candidates or max(candidates)[0]<.5:continue
                _,ti=max(candidates);iou=float((masks[owner]&truths[ti]).sum()/max(1,(masks[owner]|truths[ti]).sum()))
                if iou<.5:
                    admitted=d['rows'][owner]['result']['status']=='accepted'
                    errors.append(dict(arm=arm,key=key,text=text['text'],owner=owner,owner_category=d['instances'][owner]['category'],target=targets[ti]['uid'],owner_mask_iou=iou,admitted=admitted,polygon=text['polygon']))
                    counts[arm][r['variant']]['wrong_binding_observations']+=1
                    counts[arm][r['variant']]['wrong_binding_admitted']+=admitted
        for ti,t in enumerate(targets):
            case=dict(key=key,variant=r['variant'],target_uid=t['uid'],gt_category=t['category'],expected=t['text'],bbox=t['bbox'],pixels=t['pixels'],readable=bool(t.get('readable') and t.get('identity_expected_unique')),arms={})
            case['ocr_regions_on_target']=[]
            for text in (data['C'].get('ocr') or {}).get('texts',[]):
                region=polygon_mask(text['polygon'],truths[ti].shape)
                if (region&truths[ti]).sum()/max(1,region.sum()) >= .5:
                    bound=next((v for v in (data['C'].get('binding') or {}).get('texts',[]) if v['polygon']==text['polygon'] and v['text']==text['text']),{})
                    case['ocr_regions_on_target'].append(dict(text=text['text'],confidence=text.get('confidence'),binding_status=bound.get('status','empty_or_deduplicated'),owner=bound.get('owner_instance')))
            for arm in 'ABC':
                c=counts[arm][r['variant']];d=data[arm];s=Counter(targets=1)
                s['sign_targets' if t['category']=='shelf_marker' else 'container_targets']+=1
                read=case['readable'];s['readable']+=read
                s['readable_sign' if t['category']=='shelf_marker' else 'readable_container']+=read
                v=dict(mask_matched=ti in hits,compatible_mask_matched=ti in compatible,selected=False,admitted=False,reading='',text_correct=False,correct_3d=False)
                if ti not in hits:v['failure']='no_matching_mask'
                else:
                    pi,iou=hits[ti];inst=d['instances'][pi];row=d['rows'][pi];txt=reading(inst)
                    v.update(instance=pi,detector_label=inst['category'],iou=iou,selected=row['selected'],admitted=row['result']['status']=='accepted',map_status=row['result']['status'],map_reason=row['result'].get('reason'),reading=txt,landmark_uid=row['result'].get('landmark_uid'))
                    v['text_correct']=read and reading_key(txt)==reading_key(t['text']);v['strict_text_correct']=read and txt==t['text'];v['correct_3d']=v['text_correct'] and v['admitted']
                    if not row['selected']:v['failure']='invalid_vlm_output' if arm=='C' and not qa['valid'] else 'candidate_not_selected'
                    elif not v['admitted']:v['failure']='projection_or_association_rejected'
                    elif read and not txt:
                        regions=case['ocr_regions_on_target']
                        if any(x['binding_status']=='ambiguous' for x in regions):v['failure']='text_binding_ambiguous'
                        elif any(x['binding_status']=='low_confidence' for x in regions):v['failure']='low_confidence_text'
                        elif any(x['owner'] is not None and x['owner']!=pi for x in regions):v['failure']='text_bound_to_other_instance'
                        elif not regions:v['failure']='no_text_region_detected'
                        else:v['failure']='no_bound_ocr_text'
                    elif read and not v['text_correct']:v['failure']='incorrect_or_partial_text'
                    elif not read:v['failure']='text_unscorable'
                    else:
                        v['identity_status_final']=landmarks[arm].get(v['landmark_uid'],{}).get('identity_status','unknown')
                        v['failure']='none' if v['identity_status_final']=='identity_confirmed' else 'identity_not_confirmed'
                    if v['admitted']:uid_targets[arm][(r['variant'],v['landmark_uid'])].add(t['uid'])
                    s['mask_matched']+=1;s['selected_matched']+=v['selected'];s['admitted_matched']+=v['admitted']
                    s['text_correct']+=v['text_correct'];s['strict_text_correct']+=v['strict_text_correct'];s['correct_3d']+=v['correct_3d']
                    if t['category']=='shelf_marker':
                        s['sign_mask_matched']+=1;s['sign_selected_matched']+=v['selected'];s['sign_admitted_matched']+=v['admitted'];s['sign_correct_3d']+=v['correct_3d']
                    else:s['container_correct_3d']+=v['correct_3d']
                if ti in compatible and t['category']=='shelf_marker':s['sign_compatible_mask_matched']+=1
                s['failure:'+v['failure']]+=1;c.update(s);case['arms'][arm]=v
            old,new=case['arms']['A'],case['arms']['C']
            case['comparison']='improved' if (new['correct_3d'] and not old['correct_3d']) else 'regressed' if (old['correct_3d'] and not new['correct_3d']) else 'retained_success' if new['correct_3d'] else 'still_unresolved'
            case['wrong_binding_events']=[x for x in errors if x['key']==key and x['target']==t['uid'] and x['arm']=='C']
            case['evidence']='Saved mask, selected ID, bound text and map event; no model output used as truth.'
            case['visual_hypothesis']='Small visible target may lose detail after resizing; not experimentally isolated.' if t['pixels']<500 else 'Requires visual review; the recorded failure stage alone does not prove its perceptual cause.'
            cases.append(case)
        for arm in 'ABC':overlay(root/arm/key)
    report=dict(scope='Paired regression on previously inspected data; counts are target observations, not unique objects. A/B original navigation QA, C mapping-only two-image instance QA: whole-pipeline comparison.',qa=dict(qa_stats),qa_p50_s=float(np.percentile(qa_times,50)),qa_p95_s=float(np.percentile(qa_times,95)),arms={},binding_errors=errors)
    for arm in 'ABC':
        total=Counter()
        for c in counts[arm].values():total.update(c)
        total['false_merges']=sum(len(v)>1 for v in uid_targets[arm].values())
        total['selected_predictions']=selected_total[arm];total['selected_predictions_matching_annotated_targets']=selected_scored[arm]
        total['selected_predictions_outside_annotations_unscored']=selected_total[arm]-selected_scored[arm]
        confirmed=[dict(landmark_uid=lm['landmark_uid'],category=lm['category'],identity_text=lm.get('identity_text'),submap_id=lm['submap_id'],position=lm['position']) for lm in landmarks[arm].values() if lm.get('identity_status')=='identity_confirmed']
        report['arms'][arm]=dict(total=dict(total),variants={k:dict(v) for k,v in counts[arm].items()},confirmed_identities=confirmed)
        if not a.skip_maps:render_persistent_maps(root,arm)
    report['interpretation']={'unrelated_selection_rate':'Not estimable: facility annotations are incomplete. Unmatched predictions are unscored, not automatically false positives.', 'container_scope':'Fixed vocabulary lacks container; class-agnostic mask/text positives do not establish correct container classification.', 'ocr_scope':'Unreadable targets excluded from text accuracy; strict and normalized reading reported separately.', 'all_mask_arrays_equal':True}
    write_json(root/'comparison.json',report);write_json(root/'target-cases.json',cases)
    render_gallery(root,rows,cases,report)
    print(json.dumps(report['arms'],ensure_ascii=False),flush=True)


def render_gallery(root,rows,cases,report):
    (root/'review').mkdir(exist_ok=True)
    font=ImageFont.truetype('DejaVuSans.ttf',18);small=ImageFont.truetype('DejaVuSans.ttf',13)
    by=defaultdict(list)
    for case in cases:by[case['key']].append(case)
    sections=['<h1>建图实例筛选 A/B/C 回归对照</h1>', '<p>A：旧类别匹配；B：sign/label 映射；C：无导航任务的真实 VLM 实例筛选。灰色未选中，彩色被选择，黄色 OCR。所有地标仍待确认。</p>', '<p><a href="comparison.json">统计</a> · <a href="target-cases.json">逐目标原因</a> · <a href="freeze.json">冻结配置</a></p>', '<p>计数为观测次数；候选来自同一批真实检测。已分析回归集，非盲测；未标注设施不可直接判为误检。</p>']
    sections.append('<table><tr><th>组别</th><th>sign mask 检出</th><th>sign 被选择/检出</th><th>正确 sign 编号入图/可读</th><th>容器文字三维观测</th><th>入图错绑文字</th></tr>')
    for arm,v in report['arms'].items():
        t=v['total'];sections.append(f"<tr><td>{arm}</td><td>{t.get('sign_mask_matched',0)}/{t['sign_targets']}</td><td>{t.get('sign_selected_matched',0)}/{t.get('sign_mask_matched',0)}</td><td>{t.get('sign_correct_3d',0)}/{t.get('readable_sign',0)}</td><td>{t.get('container_correct_3d',0)}/{t.get('readable_container',0)}</td><td>{t.get('wrong_binding_admitted',0)}</td></tr>")
    sections.append('</table><h2>分场景正确编号三维观测</h2><table><tr><th>场景</th><th>A</th><th>B</th><th>C</th><th>可读目标</th></tr>')
    for variant in ['original','color','layout','surface','lighting','occlusion']:
        vals=[report['arms'][arm]['variants'][variant] for arm in 'ABC']
        sections.append('<tr><td>'+variant+'</td>'+''.join('<td>'+str(v.get('correct_3d',0))+'</td>' for v in vals)+'<td>'+str(vals[0].get('readable',0))+'</td></tr>')
    sections.append('</table><p>surface 为 container 承载物，固定词表缺少该类；这里的几何和文字正例不代表 container 分类正确。occlusion 无可读编号样本，不报告 OCR 准确率。</p>')
    q=report['qa'];sections.append(f"<p>新 VLM 有候选帧中全选：{q.get('selected_all_nonempty_batches',0)}/{q.get('nonempty_batches',0)}。准入召回提高并不代表已经学会去冗余。</p>")
    for variant in ['original','color','layout','surface','lighting','occlusion']:
        sections.append(f'<h2>{variant}</h2><p><a href="C/maps/{variant}/semantic_bev.png">新地图 BEV</a> · <a href="C/maps/{variant}/semantic_3d.png">三维图</a></p>')
        for row in [r for r in rows if r['variant']==variant]:
            key=row['key'];images=[Image.open(row['source']['rgb']).convert('RGB')]+[Image.open(root/arm/key/'segmentation.png').convert('RGB') for arm in 'ABC']
            canvas=Image.new('RGB',(1920,410),'#16212b');draw=ImageDraw.Draw(canvas)
            for i,(label,im) in enumerate(zip(['Source RGB','A: exact category','B: sign/label mapping','C: instance selection'],images)):
                canvas.paste(im.resize((480,360)),(i*480,40));draw.text((i*480+5,10),label,font=font,fill='white')
            canvas.save(root/'review'/f'{key}.jpg',quality=88)
            sections.append(f'<details><summary>{key} — {len(by[key])} 个已标注目标</summary><a href="qa/{key}/result.json">VLM 原文</a><img loading="lazy" src="review/{key}.jpg">')
            for ti,case in enumerate(by[key]):
                box=case['bbox'];x1,y1,x2,y2=box;pad=max(24,int(max(x2-x1,y2-y1)*.3));crop=(max(0,x1-pad),max(0,y1-pad),min(images[0].width,x2+pad),min(images[0].height,y2+pad))
                card=Image.new('RGB',(1440,410),'#16212b');d=ImageDraw.Draw(card)
                for i,im in enumerate(images):
                    shown=im.copy()
                    if i==0:ImageDraw.Draw(shown).rectangle(box,outline='lime',width=3)
                    view=shown.crop(crop);scale=min(350/view.width,280/view.height);view=view.resize((max(1,round(view.width*scale)),max(1,round(view.height*scale))),Image.Resampling.NEAREST);card.paste(view,(i*360+5,40))
                    label='GT '+case['expected'] if i==0 else str(case['arms']['ABC'[i-1]].get('reading',''))
                    d.text((i*360+5,5),label,font=font,fill='white')
                    if i:
                        v=case['arms']['ABC'[i-1]];d.text((i*360+5,325),v['failure'],font=small,fill='white');d.text((i*360+5,345),f"selected={v['selected']} 3D={v['correct_3d']}",font=small,fill='white')
                d.text((5,380),key+' '+case['target_uid']+' '+case['comparison'],font=small,fill='white')
                name=f'{key}-target-{ti:02}.jpg';card.save(root/'review'/name,quality=90)
                case['review_image']='review/'+name
                sections.append(f'<p>{html.escape(case["target_uid"])} / {html.escape(case["expected"])} — {case["comparison"]}: {case["arms"]["C"]["failure"]}</p><img loading="lazy" src="review/{name}">')
            sections.append('</details>')
    write_json(root/'target-cases.json',cases)
    page='<!doctype html><meta charset="utf-8"><title>Mapping-only instance selection</title><style>body{background:#101a24;color:#e8eef5;font:16px sans-serif;margin:24px}a{color:#75caff}img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #53606d}details{margin:14px 0}summary{cursor:pointer}</style>'+''.join(sections)
    (root/'index.html').write_text(page)

if __name__=='__main__':main()
