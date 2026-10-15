"""Offline annotation scheduling using recorded motion and finite reference coverage.

Future motion is used only to propose reference labels and audit coverage, never
as semantic teacher input. This is not an online inference latency guarantee.
"""
import numpy as np
from .core import resample_line
from .replay_video import point_distances


def reference_in_source(poses, start):
    relative=np.linalg.inv(poses[start])@poses[start:]
    return resample_line(relative[:,:2,3])


def select_updates(times, poses, *, mandatory=(), references=None, max_gap_s=5.,
                   reserve_m=1., max_deviation_m=.25, turn_rad=np.deg2rad(20)):
    times=np.asarray(times,float);poses=np.asarray(poses,float)
    if not len(times) or np.any(np.diff(times)<=0) or not np.isfinite(times).all():raise ValueError('Invalid times')
    if poses.shape!=(len(times),4,4) or not np.isfinite(poses).all():raise ValueError('Invalid poses')
    if min(max_gap_s,reserve_m,max_deviation_m,turn_rad)<=0:raise ValueError('Positive thresholds required')
    forced=set(mandatory)|{0,len(times)-1};refs=references or {}
    if any(i<0 or i>=len(times) for i in forced):raise ValueError('Mandatory index outside trace')
    updates=[];active=0;travel=0.
    def reset(i,reason):
        if i in refs:
            line=np.asarray(refs[i],float)
        else:
            candidate,mask=reference_in_source(poses,i)
            line=candidate[mask]
        length=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum())
        updates.append(dict(index=i,t_obs=float(times[i]),reason=reason,reference_length_m=length))
        return line,length
    line,length=reset(0,'initial')
    for i in range(1,len(times)):
        travel+=float(np.linalg.norm(poses[i,:2,3]-poses[i-1,:2,3]))
        relative=np.linalg.inv(poses[active])@poses[i]
        deviation=point_distances(relative[None,:2,3],line)[0]
        yaw=abs(float(np.arctan2(relative[1,0],relative[0,0])))
        reason=None
        if i in forced:reason='mandatory'
        elif times[i]-times[active]>=max_gap_s-1e-5:reason='max_source_age'
        elif travel>max(.05,length-min(reserve_m,length*.25)):reason='reference_reserve'
        elif deviation is not None and deviation>max_deviation_m:reason='reference_deviation'
        elif yaw>=turn_rad:reason='heading_change'
        if reason:
            active=i;travel=0.;line,length=reset(i,reason)
    return updates


def audit_coverage(times,poses,updates,lines,tolerance_m=.05):
    """Measure all source frames, including frames not encoded in the video."""
    times=np.asarray(times);poses=np.asarray(poses);rows=[]
    for n,u in enumerate(updates):
        a=u['index'];b=updates[n+1]['index'] if n+1<len(updates) else len(times)
        line=np.asarray(lines[n]);relative=np.linalg.inv(poses[a])@poses[a:b];xy=relative[:,:2,3]
        distances=point_distances(xy,line)
        length=float(np.linalg.norm(np.diff(line,axis=0),axis=1).sum())
        traveled=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum())
        valid=[d for d in distances if d is not None]
        rows.append(dict(index=a,end_exclusive=b,source_age_max_s=float(times[b-1]-times[a]),
            reference_length_m=length,travel_before_next_update_m=traveled,
            exhausted=traveled>length+tolerance_m,
            max_distance_m=max(valid,default=None),empty_reference_with_motion=not len(line) and traveled>tolerance_m))
    return dict(intervals=rows,exhausted_intervals=sum(r['exhausted'] for r in rows),
        empty_reference_motion_intervals=sum(r['empty_reference_with_motion'] for r in rows),
        max_source_age_s=max(r['source_age_max_s'] for r in rows),tolerance_m=tolerance_m)


def prepare_review(episode, output, *, seed_session=None, seed_group='manual'):
    """Prepare a dense chronological review without auto-generating semantic text."""
    import copy
    from pathlib import Path
    from .core import load_episode,extract,read_json,write_json,anchors
    from .episode_video import load_labels
    from .replay_video import load_recorded
    from .semantic_annotation import write_bundle
    from .session_teacher import start_session
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    meta,all_frames=load_episode(episode)
    seeds=load_labels(Path(seed_session),seed_group) if seed_session else []
    if seeds:
        if Path(seeds[0][0]['provenance']['source_episode']).resolve()!=Path(episode).resolve():
            raise ValueError('Seed and recording episodes differ')
        frames,_,_,_=load_recorded(seeds[0][0])
    else:
        frames=[dict(index=i,t=f['t_obs'],pose=f['T_odom_body']) for i,f in enumerate(all_frames)]
    poses=np.array([f['pose'] for f in frames]);times=np.array([f['t'] for f in frames]);start=frames[0]['index']
    old={s['provenance']['frame_index']:s for s,_ in seeds}
    refs={i-start:np.array(s['geometric_supervision']['reference_xy'])[s['geometric_supervision']['reference_mask']] for i,s in old.items()}
    updates=select_updates(times,poses,mandatory=list(refs),references=refs)
    indices=[frames[u['index']]['index'] for u in updates]
    write_json(output/'schedule.json',dict(updates=updates,source_indices=indices,max_gap_s=5,reserve_m=1,
        heading_change_deg=20,max_deviation_m=.25,online_latency_claim=False))
    project=output/'project';extract(episode,project,indices=indices)
    samples=[];lines=[]
    for i in indices:
        src=read_json(project/'samples'/f'{i:08d}'/'source.json')
        if i in old:
            sample=copy.deepcopy(old[i]);sample['observation']=src['observation']
        else:
            line,mask=resample_line(src['assistance']['suggested_reference'])
            sample=dict(observation=src['observation'],provenance=src['provenance'],evaluation_only=dict(human_text=None),
                geometric_supervision=dict(reference_xy=line.tolist(),reference_mask=mask.tolist(),waypoint=anchors(line[mask]),
                    reference_state='candidate_unreviewed' if mask.any() else 'unlabeled',
                    source='recorded_path_unreviewed',approved=False))
        if seeds:
            sample['observation']['task']=copy.deepcopy(seeds[0][0]['observation']['task'])
            sample['observation']['execution']['events']=[copy.deepcopy(e) for e in seeds[-1][0]['observation']['execution']['events']
                if e.get('t_obs',float('inf'))<=sample['observation']['t_obs']]
        if meta.get('pose_reference')=='camera_center_flu_not_robot_base':
            sample['observation']['pose_reference']=meta['pose_reference']
            sample['observation']['robot_state_mask']=[False,False]
            sample['observation']['execution']['measured_facts']['source']='camera_path_not_robot_execution'
            sample['geometric_supervision']['usable_for_robot_reference_training']=False
            sample['provenance']['capture_kind']='static_camera_replay' if meta.get('static_scene_verified') else 'camera_replay'
        samples.append(sample);g=sample['geometric_supervision'];lines.append(np.array(g['reference_xy'])[g['reference_mask']])
    coverage=audit_coverage(times,poses,updates,lines);write_json(output/'coverage.json',coverage)
    if coverage['exhausted_intervals'] or coverage['empty_reference_motion_intervals']:
        raise ValueError('Recorded motion exceeds reference coverage; inspect coverage.json')
    write_bundle(output/'bundle',samples,source='dense-recorded-episode-review',episode_review=True)
    return start_session(output/'bundle',output/'session')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed-session')
    parser.add_argument('--seed-group',default='manual')
    args=parser.parse_args()
    print(prepare_review(args.episode,args.output,seed_session=args.seed_session,seed_group=args.seed_group))
