"""Stage1a source inventory, deterministic selection, Habitat replay and v2 geometry.

Run replay with the existing navila-eval Python; prepare with warehouse-io.
No semantic teacher is called and no human approval is manufactured.
"""
import argparse
import copy
import gzip
import html
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..schema import file_sha256, fingerprint
from .core import read_json, write_json

DEFAULT_ROOT = '/project/docker/VLN/NaVILA/data'


def source_config(source):
    if source == 'rxr':
        return dict(width=640,height=480,hfov=79,camera_height=.88,agent_height=.88,
                    radius=.18,turn_deg=30,allow_sliding=False,min_depth=.5,max_depth=5.)
    return dict(width=512,height=512,hfov=90,camera_height=1.25,agent_height=1.5,
                radius=.1,turn_deg=15,allow_sliding=True,min_depth=0.,max_depth=10.)


def finite_metadata(value, path='', missing=None):
    """NaN timing annotations remain explicit missing values in verifier metadata."""
    if missing is None: missing=[]
    if isinstance(value,dict):return {k:finite_metadata(v,path+'.'+k,missing) for k,v in value.items()}
    if isinstance(value,list):return [finite_metadata(v,path+f'[{i}]',missing) for i,v in enumerate(value)]
    if isinstance(value,float) and not np.isfinite(value):missing.append(path);return None
    return value


def habitat_body_pose(position, rotation):
    """Habitat right/up/back axes -> body forward/left/up, in Habitat world."""
    basis=np.array([[0,-1,0],[0,0,1],[-1,0,0.]])
    result=np.eye(4)
    result[:3,:3]=np.asarray(rotation)@basis
    result[:3,3]=position
    return result


def load_sources(root, names=None):
    root=Path(root)
    sources={}
    for name, directory, stem in [('r2r','R2R_VLNCE_v1-3_preprocessed','train'),
                                   ('rxr','RxR_VLNCE_v0','train_guide')]:
        if names is not None and name not in names:
            continue
        path=root/'datasets'/directory/'train'/f'{stem}.json.gz'
        gt=path.with_name(f'{stem}_gt.json.gz')
        with gzip.open(path) as f: episodes=json.load(f)['episodes']
        with gzip.open(gt) as f: traces=json.load(f)
        sources[name]=(episodes,traces,path,gt)
    return sources


def route_group(e):
    # Cross-source physical paths and alternative instructions share a split.
    return fingerprint(dict(scene=Path(e['scene_id']).stem,
        path=np.round(np.asarray(e['reference_path']),2).tolist()))


def catalog(root, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    sources=load_sources(root); rng=np.random.default_rng(42)
    selected=[]; inventory={};used=set(); used_trajectory=set()
    for name,(episodes,traces,path,gt) in sources.items():
        english=[e for e in episodes if name=='r2r' or e['instruction'].get('language','').startswith('en')]
        buckets=defaultdict(list); rejected=[]
        for e in english:
            trace=traces.get(str(e['episode_id']))
            scene=Path(root)/'scene_datasets'/e['scene_id']
            if not trace or not scene.exists():
                rejected.append(dict(episode_id=e['episode_id'],reason='missing_trace_or_scene'));continue
            length=float(np.linalg.norm(np.diff(np.asarray(trace['locations']),axis=0),axis=1).sum())
            turns=sum(a in (2,3) for a in trace['actions'])
            key=(Path(e['scene_id']).stem,int(np.digitize(length,[5,10,20])),int(np.digitize(turns,[8,24])))
            buckets[key].append(e)
        keys=list(buckets);rng.shuffle(keys)
        for v in buckets.values(): rng.shuffle(v)
        chosen=[]; needed=(120 if name=='r2r' else 80)+4
        while len(chosen)<needed and any(buckets.values()):
            for key in keys:
                while buckets[key]:
                    e=buckets[key].pop();group=route_group(e)
                    trajectory=(name,Path(e['scene_id']).stem,e['trajectory_id'])
                    if group not in used and trajectory not in used_trajectory:
                        used.add(group);used_trajectory.add(trajectory)
                        chosen.append((e,key,group));break
                if len(chosen)==needed:break
        if len(chosen)!=needed:raise ValueError('Insufficient independent routes')
        for i,(e,key,group) in enumerate(chosen):
            selected.append(dict(source=name,episode_id=e['episode_id'],
                uid=f'{name}-{e["episode_id"]}',role='oracle_comparison' if i<4 else 'train',
                stratum=list(key),route_group=group,trajectory_id=e['trajectory_id'],
                scene_id=e['scene_id'],source_dataset=str(path),source_gt=str(gt)))
        inventory[name]=dict(all_train_episodes=len(episodes),english_train_episodes=len(english),
            scenes=len({e['scene_id'] for e in english}),rejected=rejected,
            dataset_sha256=file_sha256(path),gt_sha256=file_sha256(gt))
    value=dict(schema='stage1a-selection-v1',seed=42,root=str(Path(root).resolve()),
        selected=selected,inventory=inventory,official_evaluation_splits_used=False,
        annotation_language='en',sensor_contract=dict(by_source={s:source_config(s) for s in sources},step_m=.25,replay_dt_s=.2,
        pose_source='ideal_simulator_odometry',time_source='synthetic_action_step_clock'))
    write_json(output/'selection.json',value)
    return value


def replay(selection, output, limit=None):
    import habitat_sim
    import quaternion
    from habitat_sim.utils.common import quat_from_coeffs
    from ..collect import EpisodeRecorder
    sel=read_json(selection);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=sel['selected'][:limit] if limit else sel['selected']
    sources=load_sources(sel['root'], {r['source'] for r in rows})
    # Sort for scene reuse without changing selection membership or IDs.
    rows=sorted(rows,key=lambda r:(r['scene_id'],r['uid']))
    report=[]; sim=None;current_scene=None
    optical=np.eye(4);optical[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
    try:
        for row in rows:
            dest=output/row['uid'];report_path=dest/'replay-report.json'
            if report_path.exists():report.append(read_json(report_path));continue
            dest.mkdir(parents=True,exist_ok=True)
            episodes,traces,_,_=sources[row['source']]
            e=next(e for e in episodes if e['episode_id']==row['episode_id']);gt=traces[str(e['episode_id'])]
            recorder=None
            try:
                scene=str(Path(sel['root'])/'scene_datasets'/e['scene_id'])
                settings=source_config(row['source']);turn_deg=settings['turn_deg']
                optical[2,3]=settings['camera_height']
                w,h=settings['width'],settings['height'];f=w/(2*np.tan(np.deg2rad(settings['hfov'])/2))
                K=[[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1]]
                if (scene,turn_deg)!=current_scene:
                    if sim:sim.close()
                    sc=habitat_sim.SimulatorConfiguration();sc.scene_id=scene;sc.enable_physics=False
                    sc.gpu_device_id=0;sc.allow_sliding=settings['allow_sliding']
                    ac=habitat_sim.agent.AgentConfiguration();ac.radius=settings['radius'];ac.height=settings['agent_height']
                    ac.sensor_specifications=[]
                    for uuid,typ in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH)]:
                        spec=habitat_sim.SensorSpec();spec.uuid=uuid;spec.sensor_type=typ
                        spec.sensor_subtype=habitat_sim.SensorSubType.PINHOLE
                        spec.resolution=[h,w];spec.position=[0,settings['camera_height'],0];spec.parameters['hfov']=str(settings['hfov'])
                        ac.sensor_specifications.append(spec)
                    ac.action_space={
                        'forward':habitat_sim.agent.ActionSpec('move_forward',habitat_sim.agent.ActuationSpec(amount=.25)),
                        'left':habitat_sim.agent.ActionSpec('turn_left',habitat_sim.agent.ActuationSpec(amount=turn_deg)),
                        'right':habitat_sim.agent.ActionSpec('turn_right',habitat_sim.agent.ActuationSpec(amount=turn_deg))}
                    sim=habitat_sim.Simulator(habitat_sim.Configuration(sc,[ac]));current_scene=(scene,turn_deg)
                state=habitat_sim.AgentState();state.position=e['start_position'];state.rotation=quat_from_coeffs(e['start_rotation'])
                sim.initialize_agent(0,state)
                def world_body():
                    st=sim.get_agent(0).get_state()
                    return habitat_body_pose(st.position,quaternion.as_rotation_matrix(st.rotation))
                origin=world_body();poses=[];forward_locations=[np.asarray(state.position).tolist()]
                meta=dict(layout_family=Path(scene).stem,episode_id=row['uid'],
                    task=dict(task_id=row['uid'],instruction=e['instruction']['instruction_text'],task_type='INSTRUCTION'),
                    pose_source='ideal_simulator_odometry',time_source='synthetic_action_step_clock',
                    synthetic_dt_s=.2,scene=scene,agent_radius_m=settings['radius'],sensor_config=settings,source_dataset=row['source_dataset'],
                    source_episode_id=e['episode_id'],png_compress_level=1)
                recorder=EpisodeRecorder(dest/'capture/raw',meta)
                actions=gt['actions']; executed=[]
                for i in range(len(actions)+1):
                    if i:
                        a=int(actions[i-1]);executed.append(a)
                        if a:
                            sim.step({1:'forward',2:'left',3:'right'}[a])
                            if a==1:forward_locations.append(sim.get_agent(0).get_state().position.tolist())
                    sensors=sim.get_sensor_observations();pose=np.linalg.inv(origin)@world_body();poses.append(pose.tolist())
                    depth=np.asarray(sensors['depth']).squeeze();depth=np.where(np.isfinite(depth)&(depth>settings['min_depth'])&(depth<=settings['max_depth']),depth,0)
                    recorder.append(rgb=sensors['rgb'][...,:3],depth=depth,K=K,T_body_camera=optical,
                        T_odom_body=pose,t_obs=i*.2,odom_epoch=row['uid'])
                recorder.close();recorder=None
                loc=np.asarray(forward_locations);truth=np.asarray(gt['locations'])
                error=float(np.max(np.linalg.norm(loc-truth,axis=1))) if loc.shape==truth.shape else None
                value=dict(uid=row['uid'],passed=error is not None and error<=.05 and actions[-1]==0,
                    source=row['source'],role=row['role'],frames=len(poses),forward_locations=len(loc),
                    max_replay_position_error_m=error,original_locations=len(truth),
                    max_vertical_excursion_m=float(np.ptp(np.asarray(poses)[:,2,3])),
                    actual_actions=executed,sensor_config=settings,coordinate_convention='local_body_FLU_m',
                    time_source='synthetic_action_step_clock',action_labels_exported=False)
                # Privileged original route/goals remain in the independent verifier only.
                missing=[];clean_episode=finite_metadata(e,missing=missing)
                write_json(dest/'verification-only.json',dict(original_episode=clean_episode,original_gt=gt,missing_original_timing_fields=missing,
                    replayed_forward_locations=forward_locations,world_origin=origin.tolist()))
            except Exception as err:
                if recorder:recorder.close()
                value=dict(uid=row['uid'],passed=False,error_type=type(err).__name__,error=str(err))
            write_json(report_path,value);report.append(value);print(json.dumps(value),flush=True)
            write_json(output/'replay-summary.json',dict(selected_count=len(rows),results=report,
                completed=len(report),passed=sum(r['passed'] for r in report)))
    finally:
        if sim:sim.close()


def video(raw, frames, project, updates, dest):
    from PIL import Image,ImageDraw
    times=[u['index'] for u in updates]
    cmd=['ffmpeg','-v','error','-y','-framerate','5','-i',str(raw/'rgb/%08d.png'),
         '-c:v','libx264','-preset','fast','-crf','25','-threads','1','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'rgb-replay.mp4')]
    subprocess.run(cmd,check=True)
    # Keyframe contact sheets show causal maps and label-only future reference.
    for u in updates:
        index=u['index'];folder=project/'samples'/f'{index:08d}'
        s=read_json(folder/'source.json');line=np.array(s['assistance']['suggested_reference'])
        with Image.open(folder/'map.png') as im:bev=im.convert('RGB')
        d=ImageDraw.Draw(bev)
        if len(line)>1:
            pix=[(320-y/16*640,300-x/16*600) for x,y in line]
            d.line(pix,fill='orange',width=3)
        with Image.open(raw/'rgb'/f'{index:08d}.png') as im:rgb=im.convert('RGB').resize((640,640))
        canvas=Image.new('RGB',(1280,680),'#17232d');canvas.paste(rgb,(0,0));canvas.paste(bev,(640,0))
        ImageDraw.Draw(canvas).text((8,648),f'frame {index} | semantic labels UNLABELED | orange: future-demonstration label only',fill='white')
        canvas.save(dest/f'keyframe-{index:08d}.jpg')


def prepare_one(dest, selection_row):
    from .core import load_episode, extract, anchors, validate_v2, resample_line
    from .reference_schedule import select_updates,audit_coverage
    from ..data import RecordDataset
    replay_report=read_json(dest/'replay-report.json')
    if not replay_report['passed']:return dict(uid=dest.name,passed=False,reason='replay_failed')
    raw=dest/'capture/raw';meta,frames=load_episode(raw/'episode.json')
    poses=np.asarray([f['T_odom_body'] for f in frames]);times=np.array([f['t_obs'] for f in frames])
    updates=select_updates(times,poses)
    from ..config import WarehouseConfig
    project=dest/'preannotation';extract(raw/'episode.json',project,indices=[u['index'] for u in updates],
        config=WarehouseConfig(depth_stride=32,footprint_radius=meta.get('agent_radius_m',.1)))
    records=[];lines=[]
    for u in updates:
        i=u['index'];folder=project/'samples'/f'{i:08d}';s=read_json(folder/'source.json')
        xy,mask=resample_line(s['assistance']['suggested_reference'])
        state='valid' if mask.sum()>=2 else 'valid_empty'
        remaining_yaw=np.unwrap(np.arctan2(poses[i:,1,0],poses[i:,0,0]))
        pure_turn=not mask.any() and float(np.ptp(remaining_yaw))>.05
        record={k:copy.deepcopy(s[k]) for k in ('schema_version','observation','provenance')}
        record['supervision']=dict(text_state='unlabeled',reference_state=state,
            reference_xy=xy.tolist(),reference_mask=mask.tolist(),waypoint=anchors(xy[mask]),
            reference_loss_valid=not pure_turn,action_state='unlabeled',candidates=[])
        record['provenance'].update(pool='static_geometry',role='pending_semantic_annotation',
            source=selection_row['source'],route_group=selection_row['route_group'],split='train',
            geometric_label_source='verified_dataset_action_replay',semantic_review_state='unlabeled',
            route_context_status='pending_recursive_teacher',pure_rotation_not_xy_stop=pure_turn)
        validate_v2(record);path=dest/'records'/f'{i:08d}.json';write_json(path,record)
        records.append(dict(path=str(path.resolve()),sha256=file_sha256(path),
            **{k:record['provenance'][k] for k in ('sample_id','layout_family','episode_id','split')}))
        lines.append(xy[mask])
    coverage=audit_coverage(times,poses,updates,lines)
    write_json(dest/'schedule.json',dict(updates=updates,coverage=coverage))
    manifest=dict(schema_version=2,role='preannotation',optimizer_updates_permitted=False,
        split_unit='layout_family',records=records)
    write_json(dest/'manifest.json',manifest)
    data=RecordDataset(dest/'manifest.json','train')
    for i in range(len(data)):
        r=data[i]
        if r['supervision']['text_valid'] or any(r['supervision']['action_mask']):raise AssertionError('Invented supervision')
    video(raw,frames,project,updates,dest)
    passed=not coverage['exhausted_intervals'] and not coverage['empty_reference_motion_intervals']
    value=dict(uid=dest.name,passed=passed,samples=len(records),source_frames=len(frames),
        coverage=coverage,semantic_labels_generated=False,action_labels_generated=False,
        max_replay_position_error_m=replay_report['max_replay_position_error_m'],
        records=records,role=selection_row['role'])
    write_json(dest/'preparation-report.json',value)
    links=''.join(f'<a href="keyframe-{u["index"]:08d}.jpg"><img width="480" src="keyframe-{u["index"]:08d}.jpg"></a>' for u in updates)
    (dest/'index.html').write_text('<meta charset="utf-8"><h1>'+html.escape(meta['task']['instruction'])+'</h1><p>Semantic labels pending; no Oracle call.</p><video width="640" controls src="rgb-replay.mp4"></video>'+links)
    return value


def prepare(selection, output, limit=None):
    from concurrent.futures import ProcessPoolExecutor,as_completed
    sel=read_json(selection);root=Path(output)
    rows=sel['selected'][:limit] if limit else sel['selected']
    reports=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={}
        for row in rows:
            path=root/row['uid']/'preparation-report.json'
            if path.exists():reports.append(read_json(path));continue
            futures[pool.submit(prepare_one,root/row['uid'],row)]=row
        for future in as_completed(futures):
            row=futures[future]
            try:r=future.result()
            except Exception as err:
                r=dict(uid=row['uid'],passed=False,error_type=type(err).__name__,error=str(err))
                write_json(root/row['uid']/'preparation-failure.json',r)
            reports.append(r);print(json.dumps({k:v for k,v in r.items() if k not in ('records','coverage')}),flush=True)
            write_json(root/'preparation-progress.json',dict(completed=len(reports),selected=len(rows),passed=sum(r['passed'] for r in reports)))
    reports.sort(key=lambda r:r['uid'])
    train_records=[e for r in reports if r['passed'] and r['role']=='train' for e in r['records']]
    write_json(root/'manifest.json',dict(schema_version=2,role='preannotation',optimizer_updates_permitted=False,
        split_unit='layout_family',records=train_records))
    summary=dict(passed=len(reports)==len(rows) and all(r['passed'] for r in reports),
        selected_episodes=len(rows),passed_episodes=sum(r['passed'] for r in reports),
        training_samples=len(train_records),teacher_calls=0,results=reports)
    write_json(root/'report.json',summary)
    (root/'index.html').write_text('<meta charset="utf-8"><h1>Stage1a preannotation</h1><p>RGB-D + ideal odometry; semantic labels pending.</p><a href="report.json">Report</a><ul>'+''.join(
        f'<li><a href="{r["uid"]}/index.html">{r["uid"]}</a> passed={r["passed"]}</li>' for r in sorted(reports,key=lambda r:r['uid']))+'</ul>')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['catalog','replay','prepare']);p.add_argument('--root',default=DEFAULT_ROOT)
    p.add_argument('--output',required=True);p.add_argument('--selection');p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.mode=='catalog':catalog(a.root,a.output)
    elif a.mode=='replay':replay(a.selection,a.output,a.limit)
    else:prepare(a.selection,a.output,a.limit)


if __name__=='__main__':main()
