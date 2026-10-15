"""DynaNav RGB/semantic pilot adapter, without claiming robot-base GT.

Published future offsets are kept as unreviewed geometric candidates. They are
never sent to the semantic teacher; depth and a body lever arm are not invented.
"""
from pathlib import Path
import re

import numpy as np
from scipy.spatial.transform import Rotation

from ..schema import file_sha256, fingerprint
from .catalog import csv_rows
from .core import read_json, write_json, resample_line, anchors
from .semantic_annotation import write_bundle

# Exactly the USD-camera to camera-centred FLU convention used by the publisher's
# data/s01_batch_json_generation.py; translation to the robot base is NOT known.
R_USD_TO_FLU = np.array([[0,0,-1],[-1,0,0],[0,1,0]],float)


def inventory(data_root, benchmark, output):
    import yaml
    root=Path(data_root);raw=root/'DynaNav_data';windows=root/'DynaNav_json'
    records=[]
    for path in sorted(raw.glob('warehouse*')):
        records.append(dict(episode_id=path.name,embodiment='quadruped' if 'spot' in path.name else 'wheeled_name_hint',
            instruction=(path/'instruction.txt').read_text().strip() if (path/'instruction.txt').exists() else None,
            instruction_path=str(path/'instruction.txt'),
            windows=len(list(windows.glob(path.name+'_*s_*s'))),
            depth_files=sum(1 for p in path.rglob('*') if p.is_file() and 'depth' in p.parts),
            static_scene_verified=False,reason='Raw episode does not provide audited dynamic-agent metadata'))
    config=yaml.safe_load(Path(benchmark).read_text());static=[]
    for key,episode in config.items():
        if isinstance(episode,dict) and 'warehouse' in str(episode.get('scene','')).lower() and episode.get('num_people')==0:
            static.append(dict(episode=key,instruction=episode['instruction'],robot_type=episode.get('robot_type'),num_people=0))
    report=dict(raw_warehouse_episodes=len(records),warehouse_windows=sum(x['windows'] for x in records),
        records=records,static_benchmark_tasks=static,benchmark_path=str(benchmark),benchmark_sha256=file_sha256(benchmark),
        warning='Benchmark tasks and downloaded trajectories are separate; matching scene names do not establish matching tasks or staticness.')
    write_json(output,report);return report


def prepare_dynanav(data_root, episode_id, output, count=6):
    if not re.fullmatch(r'warehouse_\d+',episode_id):
        raise ValueError('This pilot supports a named wheeled warehouse episode only')
    if not 1<=count<=6:raise ValueError('At most six frames')
    root=Path(data_root).resolve();raw=root/'DynaNav_data'/episode_id
    instruction_path=raw/'instruction.txt';instruction=instruction_path.read_text().strip()
    rows,encoding=csv_rows(raw/'trajectory.csv')
    if not rows: raise ValueError('No poses')
    poses=[]
    for i,row in enumerate(rows):
        if float(row['time'])!=i:raise ValueError('Publisher frame-index convention does not hold')
        pose=np.eye(4);pose[:3,:3]=Rotation.from_quat([float(row[k]) for k in ('qx','qy','qz','qw')]).as_matrix()
        pose[:3,3]=[float(row[k]) for k in ('x','y','z')]
        if not np.isfinite(pose).all():raise ValueError('Nonfinite pose')
        poses.append(pose)
    poses=np.asarray(poses);flu=poses.copy();flu[:,:3,:3]=poses[:,:3,:3]@R_USD_TO_FLU.T
    # Deduplicate overlapping windows before selecting frames. Selection is based
    # only on source order, never on a teacher success or a future target location.
    unique={};duplicates=0
    for directory in sorted((root/'DynaNav_json').glob(episode_id+'_*s_*s')):
        for path in sorted(directory.glob('rgb_*.json')):
            tick=int(path.stem.split('_')[-1]);record=read_json(path)
            if abs(float(record['original_timestamp'])-tick/30.)>1e-5:
                raise ValueError('Published timestamp/image-index disagreement')
            if tick in unique:
                old=unique[tick][1]
                if old['current']!=record['current']:
                    raise ValueError('Conflicting source across overlapping windows')
                duplicates+=1;continue
            unique[tick]=(path,record)
    ticks=sorted(unique)
    if len(ticks)<count: raise ValueError('Insufficient frames')
    selected=[ticks[i] for i in np.linspace(0,len(ticks)-1,count,dtype=int)]
    samples=[];audit=[]
    for tick in selected:
        path,record=unique[tick]
        image=(path.parent/record['current']['img']).resolve()
        if not image.is_relative_to(raw) or not image.is_file():raise ValueError('Invalid source image')
        camera=image.parent.parent/'camera_params'/f'camera_params_{tick:05d}.json';params=read_json(camera)
        measured=np.linalg.inv(np.array(params['cameraViewTransform']).reshape(4,4).T)
        if params['metersPerSceneUnit']!=1 or not np.allclose(measured,poses[tick],atol=1e-5):
            raise ValueError('CSV pose/calibration or scale mismatch')
        if not np.allclose(Rotation.from_quat(record['current']['orientation']).as_matrix(),flu[tick,:3,:3],atol=1e-5):
            raise ValueError('Published current orientation convention mismatch')
        history=[]
        for offset in (90,60,30,0):
            h=tick-offset
            if h<0: continue
            hp=image.parent/f'rgb_{h:05d}.jpg'
            if not hp.is_file():raise ValueError('Missing causal history RGB')
            history.append(dict(image_id=f'{episode_id}:{h:08d}',t_obs=h/30.,path=str(hp),sha256=file_sha256(hp),
                T_odom_camera_flu=flu[h].tolist()))
        # Independently compare published future offsets against camera poses.
        # This validates representation only, not static task correctness.
        offsets=[[0.,0.]];max_error=0.
        last_time=tick/30.
        for f in record['future']:
            ft=float(f['original_timestamp']);future_tick=round(ft*30)
            if ft<=last_time or future_tick>=len(poses):raise ValueError('Invalid future sequence')
            expected=flu[tick,:3,:3].T@(flu[future_tick,:3,3]-flu[tick,:3,3])
            err=float(np.linalg.norm(expected-np.asarray(f['offset'])));max_error=max(max_error,err)
            if err>.05:raise ValueError('Future offset convention error exceeds5 cm')
            offsets.append(f['offset'][:2]);last_time=ft
        line,mask=resample_line(offsets)
        delta=np.linalg.inv(flu[0])@flu[tick]
        yaw=np.unwrap(np.arctan2(flu[:tick+1,1,0],flu[:tick+1,0,0]))
        events=[]  # A final recording frame does not prove arrival or STOP.
        observation=dict(task=dict(task_id=episode_id,instruction=instruction,hard_constraints=[]),
            image_id=f'{episode_id}:{tick:08d}',t_obs=tick/30.,odom_epoch=episode_id,
            T_odom_camera_flu=flu[tick].tolist(),pose_reference='camera_center_flu_not_robot_base',
            rgb_history=history,route=instruction,route_revision=0,robot_state=[0.,0.],robot_state_mask=[False,False],
            previous_local=dict(valid=False,reason='no_previous_teacher_response'),
            mapview=dict(image=None,arrays=None,source_until=tick/30.,reason='No recorded depth; map unavailable',landmarks=[]),
            execution=dict(events=events,measured_facts=dict(source='published_camera_trajectory',source_until=tick/30.,
                camera_displacement_xy=delta[:2,3].tolist(),
                camera_path_length_m=float(np.linalg.norm(np.diff(flu[:tick+1,:2,3],axis=0),axis=1).sum()),
                camera_accumulated_yaw_rad=float(yaw[-1]-yaw[0]),robot_base_pose_audited=False)))
        binding=fingerprint(dict(source_json_sha256=file_sha256(path),camera_sha256=file_sha256(camera),
            csv_sha256=file_sha256(raw/'trajectory.csv'),instruction_sha256=file_sha256(instruction_path),
            images=[h['sha256'] for h in history]))
        samples.append(dict(observation=observation,
            provenance=dict(source='DynaNav',episode_id=episode_id,layout_family='dynanav-simple-warehouse',
                role='regression_transfer_pilot',source_binding=binding,source_record=str(path),camera_file=str(camera),
                original_instruction_file=str(instruction_path),source_json_used_for_input=['current','original_timestamp'],
                excluded_from_teacher=['future','cot','per-frame rewritten instruction','simulator labels','third-person images']),
            geometric_supervision=dict(reference_xy=line.tolist(),reference_mask=mask.tolist(),waypoint=anchors(line[mask]),
                reference_state='candidate_unreviewed' if mask.any() else 'unlabeled',approved=False,
                source='published_future_offsets_camera_center_flu',usable_for_robot_reference_training=False,
                limitations=['Body-camera lever arm unverified','Staticness and task-route correctness unverified',
                    'No depth/collision check','Empty future does not imply a valid stop reference']),
            evaluation_only=dict(human_text=None,semantic_accuracy_available=False)))
        audit.append(dict(tick=tick,source_time=tick/30.,max_future_offset_reconstruction_error_m=max_error,
            future_points=len(record['future']),candidate_points=int(mask.sum()),camera_model=params['cameraModel'],
            actual_camera_pose_verified=True,robot_base_extrinsics_verified=False))
    manifest=write_bundle(output,samples,source='DynaNav-warehouse-semantic-transfer',context_mode='recursive')
    write_json(Path(output)/'adapter-audit.json',dict(episode_id=episode_id,instruction=instruction,selected_ticks=selected,
        selection='Six uniform original timestamps after overlap deduplication; fixed before model inference',
        unique_source_frames=len(ticks),duplicates_removed=duplicates,rows=audit,newline_encoding=encoding,
        static_scene_verified=False,depth_available=False,body_reference_gt_verified=False,
        language_labels_are_independent_truth=False,license_card=str(root.parent/'README.md'),
        license_note='Local dataset card declares CC-BY-4.0; simulator asset redistribution not assessed.',
        source_hz=30,source_hz_evidence='data/s01_batch_json_generation.py and published original_timestamp cross-check',
        future_offset_scope='Geometric candidate only; never model input or an approved nominal reference'))
    return manifest
