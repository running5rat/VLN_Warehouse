"""Render actual left/right Isaac 5.0 views at a recorded warehouse body pose.

Static scene camera replay only: no vehicle control, inference, or map updates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--capture',type=Path,required=True)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.input.read_text())['source']
    world_odom=np.asarray(json.loads((args.capture/'evaluation-origin.json').read_text())['T_world_odom'])
    world_body=world_odom @ np.asarray(source['T_odom_body'])
    source_extrinsic=np.asarray(source['T_body_camera'])
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','width':960,'height':720,'multi_gpu':False})
    try:
        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom,Gf
        from PIL import Image
        from ticvla.warehouse.collect import camera_calibration
        scene=args.capture/'warehouse-labels-only.usda'
        print('OPEN_STAGE',str(scene),flush=True)
        ctx=omni.usd.get_context();ctx.open_stage(str(scene))
        print('STAGE_OPENED',flush=True)
        for _ in range(120):app.update()
        stage=ctx.get_stage()
        print('STAGE_READY',flush=True)
        cameras={};rows=[]
        for name,degrees in [('front',0),('left',90),('right',-90)]:
            angle=np.deg2rad(degrees);R=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
            bc=source_extrinsic.copy();bc[:3,:3]=R @ bc[:3,:3]
            optical=world_body @ bc
            usd=optical @ np.diag([1,-1,-1,1])
            path='/DocumentViews/'+name
            cam=UsdGeom.Camera.Define(stage,path)
            cam.CreateFocalLengthAttr(16.);cam.CreateHorizontalApertureAttr(24.);cam.CreateVerticalApertureAttr(18.)
            cam.CreateClippingRangeAttr(Gf.Vec2f(.05,60))
            cam.AddTransformOp().Set(Gf.Matrix4d(*usd.T.flatten()))
            product=rep.create.render_product(path,(960,720))
            ann={key:rep.AnnotatorRegistry.get_annotator(key) for key in ['rgb','camera_params']}
            for a in ann.values():a.attach(product)
            cameras[name]=(ann,optical,bc,degrees)
        for _ in range(60):app.update()
        for _ in range(8):rep.orchestrator.step(rt_subframes=8,delta_time=0.,pause_timeline=True)
        for name,(ann,optical,bc,degrees) in cameras.items():
            rgb=np.asarray(ann['rgb'].get_data())[...,:3]
            K,actual=camera_calibration(ann['camera_params'].get_data())
            if not np.allclose(actual,optical,atol=1e-4):raise ValueError(name+' camera pose mismatch')
            if rgb.shape!=(720,960,3) or rgb.std()<5:raise ValueError(name+' empty rendering')
            dest=args.output/(name+'.png');Image.fromarray(rgb).save(dest)
            rows.append(dict(view=name,yaw_degrees=degrees,sha256=digest(dest),K=K.tolist(),T_body_camera=bc.tolist(),T_world_camera=actual.tolist()))
            print('RENDERED',name,flush=True)
        report=dict(isaac_version='5.0',renderer='RayTracedLighting',scene=str(scene),scene_sha256=digest(scene),
            source_image_id=source['image_id'],source_input_sha256=digest(args.input),source_rgb_sha256=source['rgb_hash'],
            T_world_body=world_body.tolist(),views=rows,physical_robot_execution=False,
            scope='New static camera render at the recorded body pose; shared camera centre, orientations rotated +/-90 deg. Not a live multi-camera navigation validation.')
        (args.output/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
        print('MULTIVIEW_COMPLETE',flush=True)
    finally:app.close()


if __name__=='__main__':main()
