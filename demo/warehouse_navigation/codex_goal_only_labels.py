"""Current-session visual judgments v2; deterministic serialization, not an API call.

The original capture and earlier conversation are known to the annotator. These
are non-blind Oracle revisions, not evidence of a standalone GPT model's score.
"""
import json
from pathlib import Path
from demo.warehouse_navigation.rgb_oracle_review import user_input,finish_frame,report
from ticvla.warehouse.dataset_tools.core import read_json,write_json

FWD=[[480,710],[480,670],[480,630],[480,590],[480,560]]
NEAR=[[480,710],[480,700],[480,685],[480,670],[480,650]]
RIGHT=[[480,710],[550,700],[620,690],[700,680],[780,670]]
LEFT=[[480,710],[410,700],[340,690],[260,680],[180,670]]
LOOP='Continue to the far end, turn right across the cross-aisle, turn right into the next parallel aisle, follow it to the near end, turn right back to the starting point and STOP.'
CROSS='Turn right across the far cross-aisle, turn right into the next parallel aisle, follow it to the near end, turn right back to the starting point and STOP.'
ACROSS='Cross to the next aisle entrance, turn right into it, follow it to the near end, turn right back to the starting point and STOP.'
RETURN='Follow this aisle to the near end, turn right across the cross-aisle, return to the starting point and STOP.'
HOME='Turn right across the near cross-aisle, return to the starting point and STOP.'
TARGET='Turn around, follow the cross-aisle to the aisle marked 01/5R, turn left into it, continue past 02/5R and 03/5R to 04/5R and STOP beside it.'
TARGET_CROSS='Follow the cross-aisle to the aisle marked 01/5R, turn left into it, continue past 02/5R and 03/5R to 04/5R and STOP beside it.'
LABELS={
77:(['sign'],'In the starting aisle; sign 01/4R right, facing toward 02/4R.',LOOP,FWD),
137:(['sign'],'In the starting aisle; sign 02/4R right, facing toward 03/4R.',LOOP,FWD),
197:(['sign'],'In the starting aisle; sign 03/4R right, facing toward 04/4R.',LOOP,FWD),
257:(['sign'],'Near sign 04/4R on right, facing the end wall and aisle exit.',LOOP,NEAR),
317:([],'At the aisle exit facing the wall, with racks on both sides.',CROSS,NEAR),
377:([],'At the far cross-aisle facing the wall, with open floor on the right.',CROSS,RIGHT),
427:([],'Turning right beside the far wall; cross-aisle floor opens on the right.',CROSS,RIGHT),
437:(['fire extinguisher'],'Along the far cross-aisle; wall left, rack ends and extinguisher right.',ACROSS,NEAR),
497:(['fire extinguisher'],'At the next aisle entrance; extinguisher on the right rack end.','Turn right into the aisle, follow it to the near end, turn right back to the starting point and STOP.',RIGHT),
512:(['sign','fire extinguisher'],'Turning right by the rack corner; sign 04/5R ahead right.','Finish turning right to face the clear aisle, follow it to the near end, turn right back to the starting point and STOP.',[]),
524:(['sign'],'In the next aisle; sign 04/5R left, facing toward 03/5R.',RETURN,FWD),
587:(['sign'],'In the aisle; sign 03/5R left, facing toward 02/5R.',RETURN,FWD),
648:([],'In the aisle; sign 02/5R left, facing toward 01/5R.',RETURN,FWD),
709:(['sign'],'Near sign 01/5R on left, facing the open aisle exit.',RETURN,NEAR),
771:(['sign'],'At the aisle exit; sign 01/5R high left and open cross-aisle ahead.',HOME,NEAR),
831:([],'Inferred: beyond the mapped 01/5R aisle exit, facing the near wall.',HOME,RIGHT),
861:(['fire extinguisher'],'Turning across the near cross-aisle; wall extinguisher ahead right.',HOME,RIGHT),
871:(['fire extinguisher'],'Facing across the cross-aisle; rack ends right and extinguisher ahead.','Cross to the starting point beside the first aisle and STOP.',NEAR),
931:(['fire extinguisher'],'Inferred: near the starting aisle entrance, facing the wall extinguisher.','Approach the starting point beside the first aisle and STOP.',NEAR),
946:(['fire extinguisher'],'Inferred: back at the starting aisle entrance, facing the wall.','Stop at the starting point after the circuit.',[]),
950:([],'Inferred: stopped at the mapped starting aisle entrance.','STOP',[]),
951:(['fire extinguisher'],'Inferred: at aisle 4 entrance; mapped 04/5R is behind and to the right.',TARGET,[]),
961:(['fire extinguisher'],'Inferred: turning away from aisle 4 entrance; mapped 04/5R remains out of view.',TARGET,[]),
966:([],'Inferred: turning at the near cross-aisle; target aisle not yet visible.',TARGET,[]),
971:([],'Inferred: still turning toward the cross-aisle; mapped 04/5R is out of view.',TARGET,[]),
976:([],'Inferred: nearly facing the cross-aisle toward the mapped 01/5R entrance.','Finish turning toward the cross-aisle, approach the aisle marked 01/5R, turn left into it, continue past 02/5R and 03/5R to 04/5R and STOP beside it.',[]),
986:([],'Inferred: facing along the near cross-aisle, with the mapped target aisle ahead.',TARGET_CROSS,NEAR),
1046:(['fire extinguisher'],'Inferred: at the mapped 01/5R entrance; rack corner and extinguisher left.','Turn left into the aisle marked 01/5R, continue past 02/5R and 03/5R to 04/5R and STOP beside it.',LEFT),
1061:(['sign','fire extinguisher'],'Turning left at the rack corner marked 01/5R, facing along the sign sequence.','Finish turning left to face the aisle, continue past 02/5R and 03/5R to 04/5R and STOP beside it.',[]),
1074:(['sign'],'In the aisle beside 01/5R on the right, facing toward 02/5R.','Continue past 02/5R and 03/5R to 04/5R and STOP beside it.',FWD),
1136:(['sign'],'Alongside 02/5R on the right, facing toward 03/5R and 04/5R.','Continue past 02/5R and 03/5R to 04/5R and STOP beside it.',FWD),
1196:(['sign'],'Alongside 03/5R on the right, with 04/5R visible farther ahead.','Continue past 03/5R to a clear position beside 04/5R and STOP.',FWD),
1256:(['sign'],'Approaching 04/5R on the right near the aisle end.','Approach a clear floor position beside 04/5R and STOP.',NEAR),
1277:([],'Inferred: near mapped 04/5R, now offscreen; facing the aisle exit.','Reobserve 04/5R before confirming the stopping position, then STOP beside it.',[]),
1292:(['sign'],'Beside the rack with visible 04/5R on the right, facing the end wall.','Stop beside 04/5R and confirm arrival.',[]),
1302:(['sign'],'Beside 04/5R, looking toward the rack carrying the sign.','Stop beside 04/5R and confirm arrival.',[]),
1314:([],'Inferred: stopped beside 04/5R, now partly cropped in the image.','STOP',[]),
}


def run(root):
    root=Path(root);exp=read_json(root/'experiment.json');previous=None;previous_source=None;task=None;history=[]
    assert exp['instructions']['target']=='Go to sign 04/5R and STOP beside it.'
    for row in exp['selected']:
        if row['task']!=task:task=row['task'];previous=None;previous_source=None
        dest=root/'codex-session'/f"{row['frame']:08d}"
        if (dest/'raw.txt').exists():raise ValueError('Keep immutable prior annotation; use a new output directory')
        inp=user_input(root,row,previous,previous_source)
        inp['system_text']=(root/'system.txt').read_text()
        lm,local,route,points=LABELS[row['frame']]
        parsed=finish_frame(root,'codex-session',row,inp,json.dumps(dict(landmark=lm,local=local,route=route,waypoint=points),indent=2))
        history.append(dict(frame=row['frame'],task=task,previous_route=previous['route'] if previous else exp['instructions'][task],generated_route=route,committed=parsed is not None,previous_frame=int(previous_source['image_id'].split(':')[-1]) if previous_source else None))
        if parsed is not None:previous=parsed;previous_source=inp['source']
    write_json(root/'route-history.json',history)
    write_json(root/'codex-session/provenance.json',dict(provider='current Codex session assisted visual reannotation',
        actual_api_model_id=None,requested_provider_label='Codex GPT6',api_calls=0,qwen_calls=0,
        latency=None,api_tokens=None,api_cost=None,human_approval=False,
        limitation='Known prior conversation and recorded episode; not a blind independently callable GPT6 API evaluation.',
        method='Visual review of RGB, causal maps and prior context; manual pixel choices. No future poses used to generate pixels. Original recorded completion events only for stop validation.',
        revision='goal-only-v2; source outputs frozen before depth validation; failures retained without repair'))
    report(root)

if __name__=='__main__':
    import sys
    run(sys.argv[1])
