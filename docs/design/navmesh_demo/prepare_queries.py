"""Fixed query set plus a deterministic wall fixture for the headless builder."""
import argparse
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
(root/'queries.txt').write_text(
    'aisle_out -13 6 0 -13 27 0\n'
    'cross_far -13 27 0 -8 27 0\n'
    'aisle_return -8 27 0 -8 6 0\n'
    'cross_near -8 6 0 -13 6 0\n'
    'other_aisle_goal -13 6 0 -8 22 0\n'
    'blocked_rack -13 6 0 -10.46 22 0\n')
for name in ['radius-000','radius-035','fixture']:(root/name).mkdir(exist_ok=True)
v=[(0,0,0),(20,0,0),(20,12,0),(0,12,0)];f=[(0,1,2),(0,2,3)]
offset=len(v);v += [(x,y,z) for z in (0,2) for y in (2,10) for x in (8,12)]
for q in [(0,2,3,1),(4,5,7,6),(0,1,5,4),(2,6,7,3),(0,4,6,2),(1,3,7,5)]:
    for t in [(q[0],q[1],q[2]),(q[0],q[2],q[3])]:f.append(tuple(i+offset for i in t))
v=np.array(v,dtype='<f4');v=v[:,[0,2,1]];v[:,2]*=-1;f=np.array(f,dtype='<i4')
with (root/'fixture/scene.bin').open('wb') as stream:
    np.array([len(v),len(f)],dtype='<u4').tofile(stream)
    np.array([[0,-.1,-12],[20,3,0]],dtype='<f4').tofile(stream)
    v.tofile(stream);f.tofile(stream)
(root/'fixture/queries.txt').write_text('around_wall 2 6 0 18 6 0\nblocked_goal 2 6 0 10 6 0\n')
