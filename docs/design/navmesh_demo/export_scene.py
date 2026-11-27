"""Export static warehouse triangles using the installed Isaac 5.0 runtime."""
import argparse
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
args, _ = p.parse_known_args()
import faulthandler
faulthandler.dump_traceback_later(90, repeat=True)
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 640, 'height': 480, 'renderer': 'RayTracedLighting', 'sync_loads': False})
try:
    import numpy as np
    import omni.usd
    from pxr import Usd, UsdGeom
    source = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/full_warehouse.usd'
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print('SCENE_LOAD_BEGIN', flush=True)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    stage.GetRootLayer().subLayerPaths.append(source)
    print('SCENE_COMPOSED', flush=True)
    for _ in range(5):
        app.update()
    print('SCENE_EXPORT_BEGIN', flush=True)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    vertices, triangles, records = [], [], []
    offset = 0
    # World coordinates, Z up. Keep the demo's three aisles and cross aisles.
    lo = np.array([-18., 2., -.1])
    hi = np.array([-3., 31., 3.])
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        if not points.size:
            continue
        mat = np.asarray(cache.GetLocalToWorldTransform(prim))
        points = (np.c_[points, np.ones(len(points))] @ mat)[:, :3]
        if np.any(points.max(0) < lo) or np.any(points.min(0) > hi):
            continue
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        faces, at = [], 0
        for count in counts:
            face = indices[at:at+count]
            at += count
            for j in range(1, count-1):
                faces.append([face[0], face[j], face[j+1]])
        if not faces:
            continue
        faces = np.asarray(faces, dtype=np.int32)
        if mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.leftHanded:
            faces = faces[:, [0, 2, 1]]
        # Reflection of an instance reverses its winding.
        if np.linalg.det(mat[:3, :3]) < 0:
            faces = faces[:, [0, 2, 1]]
        xyz = points[faces]
        keep = np.all(xyz.max(1) >= lo, axis=1) & np.all(xyz.min(1) <= hi, axis=1)
        faces = faces[keep]
        if not len(faces):
            continue
        used, inverse = np.unique(faces, return_inverse=True)
        vertices.append(points[used])
        triangles.append(inverse.reshape(-1, 3) + offset)
        offset += len(used)
        records.append({'path': str(prim.GetPath()), 'triangles': len(faces)})
    if not triangles:
        raise RuntimeError('No scene triangles resolved; no synthetic floor fallback')
    v = np.concatenate(vertices).astype('<f4')
    f = np.concatenate(triangles).astype('<i4')
    np.savez_compressed(output/'scene_world.npz', vertices=v, triangles=f, bounds=np.array([lo, hi]))
    # Proper-handed Z-up -> Recast Y-up: (x,y,z) -> (x,z,-y).
    rv = v[:, [0, 2, 1]].copy(); rv[:, 2] *= -1
    rlo, rhi = [lo[0], lo[2], -hi[1]], [hi[0], hi[2], -lo[1]]
    with (output/'scene.bin').open('wb') as stream:
        np.array([len(v), len(f)], dtype='<u4').tofile(stream)
        np.array([rlo, rhi], dtype='<f4').tofile(stream)
        rv.tofile(stream); f.tofile(stream)
    report = {'source': source, 'isaac_version': '5.0', 'up_axis': str(UsdGeom.GetStageUpAxis(stage)),
              'metres_per_unit': UsdGeom.GetStageMetersPerUnit(stage), 'bounds_world': [lo.tolist(), hi.tolist()],
              'vertices': len(v), 'triangles': len(f), 'meshes': records,
              'elapsed_s': time.monotonic()-started,
              'scope': 'Static scene triangle geometry; not PhysX cooked collision geometry. No robot, labels or hidden targets added.'}
    (output/'scene-source.json').write_text(json.dumps(report, indent=2)+'\n')
    print('EXPORTED', len(v), len(f), flush=True)
finally:
    app.close()
