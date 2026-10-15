"""Independent observed-topology fixture. This module runs only inside Isaac.

The fixture description is never included in navigation/VLM observations.
"""


def build_junction_fixture(stage, output):
    from pxr import Gf, UsdGeom, UsdPhysics, UsdLux
    from isaacsim.core.utils.semantics import add_update_semantics

    UsdGeom.Xform.Define(stage, "/World")

    def box(name, center, size, color):
        cube = UsdGeom.Cube.Define(stage, "/World/Fixture/" + name)
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        add_update_semantics(cube.GetPrim(), "wall" if name != "Floor" else "floor")

    box("Floor", [-13, 14, -0.1], [22, 24, 0.2], [0.35, 0.35, 0.35])
    # Robot starts at (-13,6), looking towards +Y. Cross corridor at Y=10.
    for side, x in [("Left", -15), ("Right", -11)]:
        box(side + "Near", [x, 6, 1.5], [0.2, 5.0, 3.0], [0.65, 0.68, 0.72])
        box(side + "Far", [x, 15, 1.5], [0.2, 7.0, 3.0], [0.65, 0.68, 0.72])
    box("End", [-13, 18.5, 1.5], [4.2, 0.2, 3.0], [0.65, 0.68, 0.72])
    for name, y in [("Lower", 8.5), ("Upper", 11.5)]:
        box(name + "Left", [-18, y, 1.5], [6.0, 0.2, 3.0], [0.65, 0.68, 0.72])
        box(name + "Right", [-8, y, 1.5], [6.0, 0.2, 3.0], [0.65, 0.68, 0.72])
    dome = UsdLux.DomeLight.Define(stage, "/World/FixtureLight")
    dome.CreateIntensityAttr(1200.0)
    return []
