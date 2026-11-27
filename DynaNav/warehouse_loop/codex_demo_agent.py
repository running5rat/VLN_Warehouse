"""Codex proposal submission and explicitly labelled replay of reviewed decisions."""
import argparse
import json
import shutil
from pathlib import Path
import time
import numpy as np
from .live_metrics import atomic_json
from .codex_demo import outside_camera
from ticvla.warehouse.geometry import transform_points


def submit(root, request, *, source=None, provider='codex-session'):
    nav=Path(root)/'navigation'
    source=source or json.loads((nav/'source.json').read_text())
    request={**request,'source_image_id':source['image_id'],'provider':provider}
    folder=nav/'requests'/request['request_id'];folder.mkdir(parents=True,exist_ok=False)
    prior=json.loads((nav/'last-response.json').read_text()) if (nav/'last-response.json').exists() else {}
    previous_route=prior.get('route') if prior.get('accepted') and prior.get('task_id')==request.get('task_id') else request.get('initial_route',request.get('instruction'))
    if (nav/'map.png').exists():shutil.copy2(nav/'map.png',folder/'map-at-submit.png')
    atomic_json(folder/'input.json',dict(source=source,map=str(folder/'map-at-submit.png'),
        landmarks=json.loads((nav/'landmarks.json').read_text()) if (nav/'landmarks.json').exists() else [],
        instruction=request.get('instruction'),previous_route=previous_route,provider=provider))
    atomic_json(folder/'response.json',request)
    (folder/'raw.txt').write_text('<landmark>'+json.dumps(request.get('landmark',[]))+'</landmark>\n<local>'+request.get('local','unknown')+'</local>\n<Route>'+request.get('route','')+'</Route>\n<waypoint>'+(';' .join(f'({x:.3f},{y:.3f})' for x,y in request['waypoint']) if request.get('waypoint') else '[]')+'</waypoint>\n<REFLINE>\n')
    atomic_json(nav/'proposal.json',request)
    return request


def goal_anchors(goal, source):
    target=transform_points(np.r_[goal,0.][None],np.linalg.inv(source['T_odom_body']))[0,:2]
    return (np.linspace(.2,1,5)[:,None]*target).tolist()


def replay(root, plan_path):
    """Replays only Codex-reviewed loop goals. Target is looked up in THIS run's map."""
    root=Path(root);nav=root/'navigation';plan=json.loads(Path(plan_path).read_text())
    if plan.get('version')=='codex-continuous-v2':return replay_continuous(root,plan)
    def state():return json.loads((nav/'source.json').read_text())
    def wait_ready():
        while not (nav/'source.json').exists():time.sleep(.5)
    def wait_goal(goal,timeout=180):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if (root/'cancel.json').exists():raise RuntimeError('Cancelled')
            if np.linalg.norm(np.asarray(state()['T_odom_body'])[:2,3]-goal)<.2:return
            time.sleep(.2)
        raise TimeoutError('Reviewed waypoint not reached; no automatic safety bypass')
    provider='codex-reviewed-decision-replay';wait_ready()
    for i,goal in enumerate(plan['loop_odom_goals']):
        src=state();submit(root,dict(request_id=f'replay-loop-{i}',instruction=plan['loop_instruction'],
            local='Following the reviewed warehouse aisle circuit.',route='Continue the circuit and map visible signs.',
            landmark=['sign','label'],waypoint=goal_anchors(goal,src)),source=src,provider=provider)
        wait_goal(np.asarray(goal))
    submit(root,dict(request_id='replay-loop-complete',action='complete_mapping',route='Stop after completing the circuit.'),provider=provider)
    deadline=time.monotonic()+10
    while not (nav/'target-candidates.json').exists():
        if time.monotonic()>deadline:raise RuntimeError('Loop completion was rejected')
        time.sleep(.2)
    src=state();landmarks=json.loads((nav/'landmarks.json').read_text())
    targets=[x for x in landmarks if x.get('identity_text')==plan['target_text']]
    if len(targets)!=1:raise RuntimeError('Missing or ambiguous current-map target; cannot replay hidden coordinates')
    target=targets[0]
    if not outside_camera(target['position'],src):raise RuntimeError('Target currently in RGB frustum')
    frames=[json.loads(x) for x in (root/'capture/raw/frames.jsonl').read_text().splitlines()]
    trail=np.asarray([np.asarray(f['T_odom_body'])[:2,3] for f in frames])
    k=int(np.linalg.norm(trail-np.asarray(target['position'][:2]),axis=1).argmin())
    forward=trail[:k+1];backward=trail[k:][::-1]
    length=lambda path:np.linalg.norm(np.diff(path,axis=0),axis=1).sum()
    path=forward if length(forward)<=length(backward) else backward
    if np.linalg.norm(path[-1]-np.asarray(target['position'][:2]))>2:raise RuntimeError('No observed trail within target neighborhood')
    # Decimate only near-straight observed trail sections; every leg still exposes five anchors.
    chosen=[];start=0
    for i in range(1,len(path)):
        if np.linalg.norm(path[i]-path[start])>=2:
            chosen.append(path[i]);start=i
    chosen.append(path[-1])
    for i,goal in enumerate(chosen):
        src=state();submit(root,dict(request_id=f'replay-target-{i}',target_uid=target['landmark_uid'],
            instruction=f"Navigate to shelf sign {plan['target_text']} and stop beside it.",
            local='Returning along the observed aisle map.',route=f"Go to sign {plan['target_text']} and stop.",
            landmark=[],waypoint=goal_anchors(goal,src)),source=src,provider=provider)
        wait_goal(goal)
    time.sleep(1.5)  # Observe a stationary interval; reaching a coordinate alone is insufficient.
    submit(root,dict(request_id='replay-finish',action='finish',route='Stop beside the mapped shelf sign.'),provider=provider)


def body_anchors(world,source):
    world=np.asarray(world,float)
    if world.shape!=(5,2):raise ValueError('Five world anchors required')
    return transform_points(np.c_[world,np.zeros(5)],np.linalg.inv(source['T_odom_body']))[:,:2].tolist()


def replay_continuous(root,plan):
    """Concurrent Codex-authored decisions, never reported as model inference."""
    nav=root/'navigation';provider='codex-reviewed-decision-replay'
    def state():
        s=json.loads((nav/'source.json').read_text());p=nav/'latest-perception.json'
        if p.exists():
            seen=json.loads(p.read_text())
            if 0<=s['t_obs']-seen['source']['t_obs']<2:
                s=seen['source'];s={**s,'detected_categories':seen['categories']}
        return s
    def latest_position():return np.asarray(json.loads((nav/'control-status.json').read_text())['position'])[:2,3]
    def wait_for(test,timeout=180):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            if (root/'cancel.json').exists():raise RuntimeError('Cancelled')
            if test():return
            time.sleep(.1)
        raise TimeoutError('No progress; no hidden fallback or watchdog bypass')
    wait_for(lambda:(nav/'source.json').exists() and (nav/'control-status.json').exists(),300)
    wait_for(lambda:json.loads((nav/'control-status.json').read_text())['initial_scan_complete'],90)
    def send(task,request_id,route,world,local,**extra):
        started=time.monotonic();s=state()
        result=submit(root,dict(request_id=request_id,task_id=task,instruction=plan['loop_instruction'] if task=='mapping' else target_instruction,
            initial_route=plan['loop_stages'][0]['route'] if task=='mapping' else target_initial,
            previous_route_policy='Retain every unfinished step in order, including final STOP.',
            landmark=s.get('detected_categories',[])[:3],local=local,route=route,waypoint=body_anchors(world,s),
            planning_started_monotonic=started,**extra),source=s,provider=provider)
        wait_for(lambda:(nav/'last-response.json').exists() and json.loads((nav/'last-response.json').read_text())['request_id']==request_id,15)
        response=json.loads((nav/'last-response.json').read_text())
        if not response['accepted']:raise RuntimeError(response.get('error'))
        return result
    for i,stage in enumerate(plan['loop_stages']):
        send('mapping',f'continuous-loop-{i}',stage['route'],stage['waypoints_odom'],stage['local'])
        end=np.asarray(stage['waypoints_odom'][-1])
        # Keep tracking the old curve while the independent decision process prepares its successor.
        radius=plan['handoff_distance_m'] if i<len(plan['loop_stages'])-1 else .18
        wait_for(lambda:np.linalg.norm(latest_position()-end)<radius)
    wait_for(lambda:json.loads((nav/'control-status.json').read_text())['reason']=='waypoint_endpoint_reached')
    time.sleep(1.5)
    submit(root,dict(request_id='continuous-loop-complete',task_id='mapping',action='complete_mapping',
        instruction=plan['loop_instruction'],local='Back at the starting cross-aisle.',route='STOP'),provider=provider)
    wait_for(lambda:(nav/'target-candidates.json').exists(),15)
    # New task is selected only after mapping and an actual STOP.
    target_text=plan['target_text'];landmarks=json.loads((nav/'landmarks.json').read_text())
    matches=[x for x in landmarks if x.get('identity_text')==target_text]
    if len(matches)!=1:raise RuntimeError('Far-aisle target missing or ambiguous in this map')
    target=matches[0];s=state()
    if not outside_camera(target['position'],s):raise RuntimeError('Target is currently inside the RGB frustum')
    x,y=target['position'][:2]
    if y>-5 or x<10:raise RuntimeError('Mapped target does not belong to the farther second aisle')
    # Route uses the corridor traversed in this episode. Validate all proposed points against its recorded trail.
    frames=[json.loads(v) for v in (root/'capture/raw/frames.jsonl').read_text().splitlines()]
    trail=np.asarray([np.asarray(f['T_odom_body'])[:2,3] for f in frames])
    lane=float(np.median(trail[(trail[:,0]>5)&(trail[:,1]<-3),1]))
    x_entry=float(np.percentile(trail[:,0],1));standoff=y+1.35
    first=np.array([[x_entry,-.7],[x_entry,lane+.7],[x_entry,lane],[x_entry+.7,lane],[x-5,lane]])
    second=np.array([[x-4,lane],[x-2.5,lane],[x-1.8,lane+.5],[x-1.4,lane+.2],[x-1,standoff]])
    for point in np.vstack([first,second]):
        if np.linalg.norm(trail-point,axis=1).min()>.85:raise RuntimeError('Proposed route leaves observed traversed corridor')
    target_instruction=f'Navigate to sign {target_text} in the farther second aisle and STOP beside it.'
    target_initial=f'Turn around, cross the near-end cross-aisle, turn left into the second aisle, follow it to sign {target_text}, approach the sign and STOP beside it.'
    shutil.copy2(s['rgb'],nav/'target-start-rgb.png');shutil.copy2(nav/'map.png',nav/'target-start-map.png')
    send('target','continuous-target-0',target_initial,first,
         f'At the starting cross-aisle. Mapped sign {target_text} is in the farther parallel aisle.',target_uid=target['landmark_uid'])
    wait_for(lambda:np.linalg.norm(latest_position()-first[-1])<plan['handoff_distance_m'])
    send('target','continuous-target-1',f'Continue along the second aisle to sign {target_text}, approach it and STOP beside it.',second,
         f'In the second aisle, approaching the mapped position of sign {target_text}.',target_uid=target['landmark_uid'])
    wait_for(lambda:np.linalg.norm(latest_position()-second[-1])<.18)
    wait_for(lambda:json.loads((nav/'control-status.json').read_text())['reason']=='waypoint_endpoint_reached')
    time.sleep(1.5);s=state();shutil.copy2(s['rgb'],nav/'target-final-rgb.png')
    submit(root,dict(request_id='continuous-finish',task_id='target',action='finish',instruction=target_instruction,
        landmark=s.get('detected_categories',[])[:3],local=f'Stopped beside mapped sign {target_text}.',route='STOP'),source=s,provider=provider)
    wait_for(lambda:(nav/'task-finished.json').exists(),15)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--replay',required=True)
    a=p.parse_args()
    try:replay(a.root,a.replay)
    except Exception as exc:
        atomic_json(Path(a.root)/'cancel.json',dict(reason=repr(exc),source='codex-decision-replay'))
        raise
