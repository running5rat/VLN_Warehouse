"""Codex session visual annotations, 2026-11-12. NOT an inference service.

Pixel coordinates were chosen from viewed RGB contact sheets. No prior metric
curve, future pose, hidden scene coordinate or automatic floor search is used.
This session has prior demo context; these labels are not a blind API benchmark.
"""
import json
from pathlib import Path
from demo.warehouse_navigation.rgb_oracle_review import user_input,finish_frame,report
from ticvla.warehouse.dataset_tools.core import read_json,write_json

STRAIGHT=[[480,700],[480,620],[480,560],[480,510],[480,470]]
SHORT=[[480,710],[480,680],[480,650],[480,620],[480,590]]
RIGHT=[[480,710],[560,690],[640,670],[730,650],[820,630]]
LEFT=[[480,710],[400,690],[320,670],[230,650],[140,630]]
LOOP='Continue to the far end, turn right across the cross-aisle, turn right into the second aisle, follow it to the near end, turn right to the starting point and STOP.'
CROSS='Turn right across the far cross-aisle, turn right into the second aisle, follow it to the near end, turn right to the starting point and STOP.'
ACROSS='Cross to the second aisle, turn right into it, follow it to the near end, turn right to the starting point and STOP.'
RETURN='Follow the second aisle to the near end, turn right to the starting point and STOP.'
HOME='Turn right across the near cross-aisle, return to the starting point and STOP.'
TARGET='Turn around, cross the near cross-aisle, turn left into the second aisle, follow it to sign 04/5R and STOP beside it.'
TARGET_CROSS='Cross to the second aisle, turn left into it, follow it to sign 04/5R and STOP beside it.'
TARGET_AISLE='Continue along the second aisle to sign 04/5R and STOP beside it.'

# Explicit visual judgments, including legitimate empty paths when the next
# required heading is outside RGB. Do not change these based on projection scores.
LABELS={
77: (['sign'],'In the starting aisle; sign 01/4R right, facing toward 02/4R.',LOOP,STRAIGHT),
137: (['sign'],'In the starting aisle; sign 02/4R right, facing toward 03/4R.',LOOP,STRAIGHT),
197: (['sign'],'In the starting aisle; sign 03/4R right, facing toward 04/4R.',LOOP,STRAIGHT),
257: (['sign'],'Near sign 04/4R on right, facing the end wall and aisle exit.',LOOP,STRAIGHT),
317: ([], 'At the aisle exit facing the wall; racks border both sides.',LOOP,SHORT),
377: ([], 'At the far cross-aisle, facing the wall with the aisle behind.',CROSS,RIGHT),
427: ([], 'Turning right along the far wall; the cross-aisle opens to the right.',CROSS,[[550,710],[620,700],[690,685],[760,665],[835,640]]),
437: (['fire extinguisher'],'Along the far cross-aisle; wall left, rack ends and extinguisher right.',ACROSS,STRAIGHT),
497: (['fire extinguisher'],'At the second aisle entrance; extinguisher on the right rack end.', 'Turn right into the second aisle, follow it to the near end, turn right to the starting point and STOP.',RIGHT),
512: (['sign','fire extinguisher'],'Turning into the second aisle; sign 04/5R is visible on the right.', 'Finish turning right into the second aisle, follow it to the near end, turn right to the starting point and STOP.',[[480,710],[550,650],[620,590],[690,540],[760,510]]),
524: (['sign'],'In the second aisle; sign 04/5R left, facing toward 03/5R.',RETURN,STRAIGHT),
587: (['sign'],'In the second aisle; sign 03/5R left, facing toward 02/5R.',RETURN,STRAIGHT),
648: ([], 'In the second aisle; sign 02/5R left, facing toward 01/5R.',RETURN,STRAIGHT),
709: (['sign'],'Near sign 01/5R on left, facing the open end of the second aisle.',RETURN,STRAIGHT),
771: (['sign'],'Leaving the second aisle; sign 01/5R high left, open cross-aisle ahead.',RETURN,SHORT),
831: ([], 'Inferred: at the near cross-aisle beyond the second aisle exit.',HOME,RIGHT),
861: (['fire extinguisher'],'Turning toward the starting aisle; wall-mounted extinguisher ahead right.',HOME,RIGHT),
871: (['fire extinguisher'],'Facing across the near cross-aisle; rack ends right, extinguisher ahead.', 'Cross to the starting point beside the first aisle and STOP.',SHORT),
931: (['fire extinguisher'],'Inferred: near the starting aisle entrance, extinguisher on the wall ahead.', 'Approach the starting point beside the first aisle and STOP.',SHORT),
946: (['fire extinguisher'],'Inferred: back beside the starting aisle entrance, facing the wall.', 'Stop at the starting point after completing the circuit.',[]),
950: ([], 'Inferred: stopped back at the starting aisle after the circuit.', 'STOP',[]),
951: (['fire extinguisher'],'Inferred: at the starting aisle entrance; target 04/5R is in the mapped second aisle.',TARGET,[]),
961: (['fire extinguisher'],'Turning away from the starting aisle; extinguisher moves toward the right.',TARGET,[]),
966: ([], 'Inferred: turning around at the starting cross-aisle; blank wall ahead.',TARGET,[]),
971: ([], 'Inferred: continuing the turn toward the second aisle; no sign visible.',TARGET,[]),
976: ([], 'Inferred: facing toward the near cross-aisle after turning around.', 'Finish turning toward the second aisle, cross to it, turn left, follow it to sign 04/5R and STOP beside it.',SHORT),
986: ([], 'At the near cross-aisle; rack ends left, facing toward the second aisle.',TARGET_CROSS,STRAIGHT),
1046: (['fire extinguisher'],'Near the second aisle entrance; rack end and extinguisher on the left.', 'Turn left into the second aisle, follow it to sign 04/5R and STOP beside it.',LEFT),
1061: (['sign','fire extinguisher'],'Turning left into the second aisle, with sign 01/5R on the rack.', 'Finish turning left into the second aisle, follow it to sign 04/5R and STOP beside it.',[[480,710],[410,650],[340,590],[270,540],[200,510]]),
1074: (['sign'],'In the second aisle; sign 01/5R right, facing toward 02/5R.',TARGET_AISLE,STRAIGHT),
1136: (['sign'],'In the second aisle; sign 02/5R right, facing toward 03/5R.',TARGET_AISLE,STRAIGHT),
1196: (['sign'],'In the second aisle; sign 03/5R right, target sign 04/5R ahead right.',TARGET_AISLE,STRAIGHT),
1256: (['sign'],'Near the far end of the second aisle, with target 04/5R on the right.', 'Approach a clear floor position beside sign 04/5R and STOP.',SHORT),
1277: ([], 'Inferred: near sign 04/5R, now outside the image; facing a rack end and wall.', 'Turn to reacquire sign 04/5R, approach a clear position beside it and STOP.',[]),
1292: (['sign'],'Near sign 04/5R on the right; facing the cross-aisle and end wall.', 'Approach a clear position beside sign 04/5R and STOP.',[[480,710],[510,690],[540,670],[570,650],[600,630]]),
1302: (['sign'],'Beside the rack carrying sign 04/5R, looking toward its shelving.', 'Stop beside sign 04/5R.',[]),
1314: ([], 'Inferred: stopped beside sign 04/5R; its number is now partly cropped.', 'STOP',[]),
}


def run(root):
    root=Path(root);exp=read_json(root/'experiment.json');previous=None;prev_source=None;task=None
    for row in exp['selected']:
        if row['task']!=task:previous=None;prev_source=None;task=row['task']
        inp=user_input(root,row,previous,prev_source)
        landmark,local,route,waypoint=LABELS[row['frame']]
        raw=json.dumps(dict(landmark=landmark,local=local,route=route,waypoint=waypoint),indent=2)
        parsed=finish_frame(root,'codex-session',row,inp,raw)
        if parsed is not None:previous=parsed;prev_source=inp['source']
    write_json(root/'codex-session/provenance.json',dict(provider='current Codex session assisted visual annotation',
        actual_api_model_id=None,api_tokens=None,api_cost=None,latency='not an online callable provider; no inference latency claim',
        input_context_limitation='This conversation contains prior demo context; not blind or API-equivalent comparison',
        pixel_source='manually chosen from current RGB, no prior metric path projection',
        human_approval=False))
    report(root)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
