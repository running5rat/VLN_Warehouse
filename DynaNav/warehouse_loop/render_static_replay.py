"""Re-render a recorded camera path in the static base warehouse (no agents).

This is a camera replay, not a physical robot execution or an action oracle.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','width':960,'height':540})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom,Gf
        from scipy.spatial.transform import Rotation
        from DynaNav.warehouse_loop.scene import WAREHOUSE
        from ticvla.warehouse.collect import EpisodeRecorder,camera_calibration
        from ticvla.warehouse.schema import file_sha256
        stage_context=omni.usd.get_context();stage_context.new_stage();stage=stage_context.get_stage()
        stage.GetRootLayer().subLayerPaths.append(WAREHOUSE)
        for _ in range(100):app.update()
        camera=UsdGeom.Camera.Define(stage,'/ReviewCamera')
        camera.GetFocalLengthAttr().Set(24.)
        camera.GetHorizontalApertureAttr().Set(36.)
        camera.GetVerticalApertureAttr().Set(20.25)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(.05,1000))
        op=UsdGeom.Xformable(camera).AddTransformOp()
        product=rep.create.render_product('/ReviewCamera',(960,540))
        annotators={n:rep.AnnotatorRegistry.get_annotator(n) for n in ['rgb','distance_to_image_plane','camera_params']}
        for a in annotators.values():a.attach(product)
        rows=list(csv.DictReader((args.source/'trajectory.csv').open(encoding='utf-8-sig')))
        ticks=list(range(0,len(rows),6))
        if ticks[-1]!=len(rows)-1:ticks.append(len(rows)-1)
        episode='warehouse-static-forklift-replay'
        metadata=dict(episode_id=episode,layout_family='dynanav-simple-warehouse',
            task=dict(task_id=episode,instruction='Move along the aisle toward the yellow forklift and stop nearby.',hard_constraints=[]),
            capture_hz=5,static_scene_verified=True,staticness_basis='Base warehouse only; no characters/robots spawned; physics not stepped; USD time held fixed.',
            pose_reference='camera_center_flu_not_robot_base',physical_robot_execution=False,
            nominal_route_approved=False,source_trajectory=str(args.source/'trajectory.csv'),
            source_trajectory_sha256=file_sha256(args.source/'trajectory.csv'),
            route_caveat='Path originally recorded with pedestrians; re-rendering without pedestrians does not establish optimal static route GT.',
            original_dynamic_rgb_reused=False,scene=WAREHOUSE,isaac_version='5.0',png_compress_level=1)
        recorder=EpisodeRecorder(args.output/'capture/raw',metadata)
        mapping=[]
        for j,tick in enumerate(ticks):
            r=rows[tick];T=np.eye(4);T[:3,:3]=Rotation.from_quat([float(r[k]) for k in ('qx','qy','qz','qw')]).as_matrix();T[:3,3]=[float(r[k]) for k in ('x','y','z')]
            op.Set(Gf.Matrix4d(*T.T.flatten()))
            rep.orchestrator.step(rt_subframes=4,delta_time=0.0,pause_timeline=True)
            data={n:a.get_data() for n,a in annotators.items()}
            K,optical=camera_calibration(data['camera_params'])
            body=T.copy();body[:3,:3]=T[:3,:3]@np.array([[0,-1,0],[0,0,1],[-1,0,0]])
            if not np.allclose(optical,T@np.diag([1,-1,-1,1]),atol=1e-4):raise ValueError('Rendered/source camera pose mismatch')
            recorder.append(rgb=data['rgb'],depth=data['distance_to_image_plane'],K=K,T_body_camera=np.linalg.inv(body)@optical,
                T_odom_body=body,t_obs=tick/30.,odom_epoch=episode)
            mapping.append(dict(frame=j,source_index=tick,t_obs=tick/30.))
            if j%25==0:print('STATIC_FRAME',j,len(ticks),flush=True)
        recorder.close()
        (args.output/'source-frame-map.json').write_text(json.dumps(mapping,indent=2))
        (args.output/'static-scene-audit.json').write_text(json.dumps(dict(**metadata,frames=len(ticks),
            dynamic_prim_name_candidates=[str(q.GetPath()) for q in stage.Traverse() if any(w in str(q.GetPath()).lower() for w in ('character','pedestrian'))]),indent=2))
        print('STATIC_REPLAY_COMPLETE',flush=True)
    finally:app.close()

if __name__=='__main__':main()
