"""Audit actual Route recursion and measured motion, without scoring a trained VLA."""
import argparse
import json
import time
import re
from pathlib import Path
import numpy as np


def run(root):
    root=Path(root);nav=root/'navigation'
    decisions=[json.loads(s) for s in (nav/'decisions.jsonl').read_text().splitlines()]
    commands=[json.loads(s) for s in (nav/'executed-commands.jsonl').read_text().splitlines()]
    previous={};audit=[];handoffs=[];clock_offset=time.time()-time.monotonic()
    for d in decisions:
        inp=json.loads((nav/'requests'/d['request_id']/'input.json').read_text())
        task=d.get('task_id');old=previous.get(task)
        row=dict(request_id=d['request_id'],task_id=task,accepted=d['accepted'],route=d.get('route'),
                 previous_route=inp.get('previous_route'),recursion_matches=old is None or old['route']==inp.get('previous_route'),
                 ends_with_stop=d.get('route','').rstrip('.').endswith('STOP'),
                 terminal_stop_command=bool(re.search(r'\bSTOP(?: beside (?:it|the sign))?[.!]?$',d.get('route',''))))
        if old is None:row['initial_full_route']=d.get('initial_route',d.get('route'))
        audit.append(row)
        if d['accepted']:
            if old and d.get('waypoint'):
                stamp=d.get('accepted_monotonic',d['wall']-clock_offset)
                near=min(commands,key=lambda c:abs(c['monotonic']-stamp))
                handoffs.append(dict(request_id=d['request_id'],measured_v_w=near['measured_v_w'],
                    command_reason=near['reason'],delta_to_command_s=abs(near['monotonic']-stamp),
                    timestamp_basis='accepted_monotonic' if 'accepted_monotonic' in d else 'accepted_wall_converted_with_local_clock_offset',
                    moving=abs(near['measured_v_w'][0])>.03 or abs(near['measured_v_w'][1])>.03))
            previous[task]=d
    motion=[c for c in commands if c.get('measured_v_w') is not None and c['reason'] not in ['initial_observation_scan','waiting_codex_waypoints']]
    duration=commands[-1]['t_obs']-commands[0]['t_obs'] if commands else 0
    out=dict(routes=audit,handoffs=handoffs,all_handoffs_while_moving=bool(handoffs) and all(x['moving'] for x in handoffs),
        total_recorded_simulated_seconds=duration,command_samples=len(commands),wall_minus_monotonic_s=clock_offset,
        unmeasured_samples=sum(c.get('measured_v_w') is None for c in commands),
        expired_command_samples=sum(c['reason'] in ['command_expired','sensor_expired'] for c in commands),
        stationary_fraction_including_task_stops=float(np.mean([abs(c['measured_v_w'][0])<.02 and abs(c['measured_v_w'][1])<.02 for c in motion])) if motion else None,
        interpretation='Codex-authored asynchronous replay; motion samples include required STOPs. Not an LLM latency benchmark.')
    (root/'review'/'continuity.json').write_text(json.dumps(out,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
