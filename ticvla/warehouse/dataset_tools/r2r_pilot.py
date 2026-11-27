"""Four fixed R2R episodes: frozen preparation, sequential teacher calls, review.

Raw datasets are read-only. No training or automatic label approval is performed.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .core import read_json, write_json, validate_v2
from .oracle_comparison import teacher_sample
from .semantic_annotation import write_bundle
from .reference_schedule import audit_coverage
from ..schema import file_sha256
from ..perception.realtime import load_vocabulary

EPISODES = {'r2r-5351':10, 'r2r-6244':12, 'r2r-6508':10, 'r2r-7275':8}


def initialize(dataset, output, config, vocabulary, weights):
    dataset=Path(dataset).resolve();root=Path(output).resolve()
    if root.exists():raise ValueError('Use a new pilot directory')
    source=read_json(dataset/'report.json')
    if not source['passed']:raise ValueError('Source replay preparation failed')
    rows={r['uid']:r for r in source['results']}
    development=sorted(r['uid'] for r in rows.values() if r['uid'].startswith('r2r-') and r['role']=='train')[:3]
    root.mkdir(parents=True)
    cfg=read_json(config);write_json(root/'teacher-config.json',cfg)
    vocab=load_vocabulary(vocabulary);write_json(root/'vocabulary.json',vocab)
    report=dict(protocol='r2r-semantic-pilot-v1',dataset=str(dataset),episodes=[],development=[],
        teacher_config_sha256=file_sha256(root/'teacher-config.json'),vocabulary_sha256=file_sha256(root/'vocabulary.json'),
        source_report_sha256=file_sha256(dataset/'report.json'),weights=str(Path(weights).resolve()),
        weights_sha256=file_sha256(weights),training=False,optimizer_updates_permitted=False,
        human_approved=False,code_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        pose_source='ideal_simulator_odometry',time_source='synthetic_action_step_clock',
        teacher_input_excludes_future_geometry=True,official_validation_used=False)
    code=['ticvla/warehouse/perception/realtime.py','ticvla/warehouse/dataset_tools/semantic_loop.py',
          'ticvla/warehouse/dataset_tools/r2r_pilot.py','ticvla/warehouse/dataset_tools/semantic_annotation.py']
    report['code_hashes']={p:file_sha256(p) for p in code}
    for uid in [*development,*EPISODES]:
        dev=uid in development;row=rows[uid]
        if not dev and (row['role']!='oracle_comparison' or row['samples']!=EPISODES[uid]):
            raise ValueError('Fixed comparison selection changed')
        entries=read_json(dataset/uid/'manifest.json')['records']
        indices=np.linspace(0,len(entries)-1,4,dtype=int).tolist() if dev else list(range(len(entries)))
        samples=[];hashes={}
        for i in indices:
            entry=entries[i];p=Path(entry['path'])
            if file_sha256(p)!=entry['sha256']:raise ValueError('Source record changed')
            record=read_json(p);validate_v2(record)
            if record['supervision']['text_state']!='unlabeled':raise ValueError('Expected unlabeled source text')
            sample=teacher_sample(record)
            sample['provenance'].update(role='perception_development' if dev else 'oracle_comparison',
                optimizer_updates_permitted=False)
            samples.append(sample);hashes[str(p)]=entry['sha256']
        raw=dataset/uid/'capture/raw';meta=read_json(raw/'episode.json')
        frames=[json.loads(x) for x in (raw/meta['frames_file']).read_text().splitlines()]
        if meta['pose_source']!='ideal_simulator_odometry':raise ValueError('Unexpected odometry')
        updates=[dict(index=s['provenance']['frame_index']) for s in samples]
        lines=[np.asarray(s['geometric_supervision']['reference_xy'])[s['geometric_supervision']['reference_mask']] for s in samples]
        coverage=audit_coverage([f['t_obs'] for f in frames],np.asarray([f['T_odom_body'] for f in frames]),updates,lines)
        if not dev and (coverage['exhausted_intervals'] or coverage['empty_reference_motion_intervals']):
            raise ValueError('Reference coverage gap; revise frozen frame list before any API call')
        folder=root/('development' if dev else 'episodes')/uid
        write_bundle(folder/'bundle',samples,source=uid,episode_review=True)
        write_json(folder/'coverage.json',coverage)
        row=dict(uid=uid,folder=str(folder),frames=len(samples),source_hashes=hashes,
            source_episode=str(raw/'episode.json'),source_episode_sha256=file_sha256(raw/'episode.json'),
            source_frames_sha256=file_sha256(raw/meta['frames_file']),
            source_indices=[s['provenance']['frame_index'] for s in samples],instruction=meta['task']['instruction'],
            coverage_pass=not(coverage['exhausted_intervals'] or coverage['empty_reference_motion_intervals']),
            bundle_sha256=file_sha256(folder/'bundle/manifest.json'))
        report['development' if dev else 'episodes'].append(row)
    report['planned_requests']=sum(x['frames'] for x in report['episodes'])
    report['development_frames']=sum(x['frames'] for x in report['development'])
    write_json(root/'pilot.json',report)
    print(json.dumps(dict(planned_requests=report['planned_requests'],development_frames=report['development_frames'])))
    return report


def verify(root):
    root=Path(root);info=read_json(root/'pilot.json')
    for name,key in [('teacher-config.json','teacher_config_sha256'),('vocabulary.json','vocabulary_sha256')]:
        if file_sha256(root/name)!=info[key]:raise ValueError('Frozen pilot configuration changed')
    if file_sha256(info['weights'])!=info['weights_sha256']:raise ValueError('Detector weights changed')
    for row in info['episodes']+info['development']:
        if file_sha256(Path(row['folder'])/'bundle/manifest.json')!=row['bundle_sha256']:
            raise ValueError('Pilot bundle changed')
        if file_sha256(row['source_episode'])!=row['source_episode_sha256']:raise ValueError('Episode changed')
        meta=read_json(row['source_episode'])
        if file_sha256(Path(row['source_episode']).parent/meta['frames_file'])!=row['source_frames_sha256']:
            raise ValueError('Capture frames changed')
        for path,sha in row['source_hashes'].items():
            if file_sha256(path)!=sha:raise ValueError('Source record changed')
    return info


def prepare(root, development=False):
    """One subprocess per episode releases model/device memory between episodes."""
    root=Path(root).resolve();info=verify(root)
    for row in info['development' if development else 'episodes']:
        folder=Path(row['folder']);dest=folder/'perception'
        if dest.exists():raise ValueError('Prepared output exists; do not overwrite a frozen run')
        subprocess.run([sys.executable,'-m','ticvla.warehouse.dataset_tools.semantic_loop','prepare',
            '--bundles',str(folder/'bundle'),'--output',str(dest),'--weights',info['weights'],
            '--vocabulary',str(root/'vocabulary.json')],check=True)


def precheck(root):
    """Mechanical perception precheck; semantic quality is explicitly unverified."""
    from PIL import Image, ImageDraw, ImageOps
    root=Path(root).resolve();info=verify(root);rows=[];tiles=[]
    if (root/'development-review.json').exists():raise ValueError('Preserve existing development review')
    for entry in info['development']:
        folder=Path(entry['folder'])/'perception';manifest=read_json(folder/'manifest.json')
        if not manifest['complete']:raise ValueError('Incomplete development perception')
        for item in manifest['records']:
            f=folder/item['folder'];p=read_json(f/'perception.json')
            if file_sha256(f/'perception.json')!=item['perception_sha256']:raise ValueError('Perception changed')
            with Image.open(p['source']['rgb']) as im:shape=(im.height,im.width)
            for inst in p['instances']:
                mask=np.load(inst['mask_path'])
                if (file_sha256(inst['mask_path'])!=inst['mask_sha256'] or mask.shape!=shape
                        or mask.dtype!=np.bool_ or not mask.any() or inst['category'] not in manifest['vocabulary']['categories']):
                    raise ValueError('Invalid development mask/class')
            rows.append(dict(image_id=p['source']['image_id'],instances=len(p['instances']),
                categories=[x['category'] for x in p['instances']],segmentation_s=p['elapsed_s'],ocr_s=p['ocr']['elapsed_s']))
            with Image.open(p['candidate_request']['overlay']) as im:
                tile=Image.new('RGB',(520,560),'white');tile.paste(ImageOps.contain(im.convert('RGB'),(512,512)),(4,32))
            ImageDraw.Draw(tile).text((6,8),p['source']['image_id'],fill='black');tiles.append(tile)
    if len(rows)!=12:raise ValueError('Expected exactly twelve fixed development frames')
    for start in range(0,len(tiles),6):
        sheet=Image.new('RGB',(1560,1120),'#dddddd')
        for j,tile in enumerate(tiles[start:start+6]):sheet.paste(tile,((j%3)*520,(j//3)*560))
        sheet.save(root/f'development-{start//6}.jpg')
    report=dict(interface_pass=True,vocabulary_sha256=info['vocabulary_sha256'],frames=rows,
        actual_teacher_calls=0,human_approved=False,quality_verified=False,visual_review_status='review_pending',
        decision='Interface only. Inspect development images before freezing a new vocabulary; no accuracy claim.')
    write_json(root/'development-review.json',report);return report


def run_episode(root, episode, credentials):
    from .semantic_loop import run, audit
    root=Path(root).resolve();info=verify(root)
    index=list(EPISODES).index(episode)
    for previous in info['episodes'][:index]:
        folder=Path(previous['folder'])
        r=read_json(folder/'plus/report.json');a=read_json(folder/'plus/causality-audit.json')
        if r.get('fatal_stop') or not r.get('all_frames_attempted') or not a['interface_pass']:
            raise ValueError('Earlier episode engineering check failed; stop paid requests')
    precheck=read_json(root/'development-review.json')
    if not precheck['interface_pass'] or precheck['vocabulary_sha256']!=info['vocabulary_sha256']:
        raise ValueError('Development precheck must pass before paid requests')
    row=next(r for r in info['episodes'] if r['uid']==episode);folder=Path(row['folder'])
    result=run(folder/'perception',root/'teacher-config.json',folder/'plus',credentials)
    checks=audit(folder/'plus')
    if result.get('fatal_stop') or not result['all_frames_attempted'] or not checks['interface_pass']:
        raise RuntimeError('Episode incomplete; inspect artifacts before any later requests')
    # A wholly rejected first task cannot validate the feedback interface.
    if index==0 and not any(r['status']=='format_valid_unreviewed' for r in result['samples']):
        raise RuntimeError('No valid teacher output in first episode')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('initialize');a.add_argument('--dataset',required=True);a.add_argument('--output',required=True)
    a.add_argument('--config',default='configs/dataset_preparation/qwen_plus_teacher.v6.json')
    a.add_argument('--vocabulary',default='configs/dataset_preparation/r2r_indoor.v1.json')
    a.add_argument('--weights',default='/data/ticvla/models/yoloe-v8s-seg.pt')
    a=sub.add_parser('prepare');a.add_argument('--root',required=True);a.add_argument('--development',action='store_true')
    a=sub.add_parser('precheck');a.add_argument('--root',required=True)
    a=sub.add_parser('run');a.add_argument('--root',required=True);a.add_argument('--episode',choices=EPISODES,required=True)
    a.add_argument('--credentials',required=True)
    args=vars(p.parse_args());cmd=args.pop('command')
    {'initialize':initialize,'prepare':prepare,'precheck':precheck,'run':run_episode}[cmd](**args)


if __name__=='__main__':main()
