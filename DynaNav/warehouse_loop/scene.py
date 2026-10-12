"""Add uniquely numbered aisle flags to the unchanged DynaNav warehouse asset."""

import json
from pathlib import Path

import numpy as np

WAREHOUSE = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"
ROBOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
# One closed circuit around the four central rack bays, in world metres.
WAYPOINTS = [[-13.0, 6.0], [-13.0, 27.0], [-8.0, 27.0], [-8.0, 6.0], [-13.0, 6.0]]


def label_texture(path, label):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("RGB", (1024, 512), "white")
    draw = ImageDraw.Draw(im)
    red = (185, 20, 38)
    draw.rectangle((580, 0, 1023, 511), fill=red)
    draw.rectangle((3, 3, 1020, 508), outline=red, width=10)
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    number, row = label.split("/")
    draw.text(
        (280, 236), number, anchor="mm", font=ImageFont.truetype(font, 300), fill=red
    )
    draw.text(
        (802, 250),
        "/" + row,
        anchor="mm",
        font=ImageFont.truetype(font, 176),
        fill="white",
    )
    im.save(path)


def add_labels(stage, output, variant=None):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
    from isaacsim.core.utils.semantics import add_update_semantics

    output = Path(output)
    textures = output / "textures"
    textures.mkdir(parents=True, exist_ok=True)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), ["default", "render", "proxy"], useExtentsHint=True
    )
    bays = {}
    for prim in stage.GetPrimAtPath("/Root").GetChildren():
        if not prim.GetName().startswith("SM_RackShelf_"):
            continue
        bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        key = tuple(np.round(np.r_[lo[:2], hi[:2]], 1))
        bays.setdefault(key, []).append(
            {"prim": str(prim.GetPath()), "min": lo.tolist(), "max": hi.tolist()}
        )
    xs = sorted(set(key[0] for key in bays))
    catalog = []
    for row, x in enumerate(xs, 1):
        keys = sorted((key for key in bays if key[0] == x), key=lambda key: key[1])
        for index, key in enumerate(keys, 1):
            shelves = bays[key]
            lo = np.min([s["min"] for s in shelves], axis=0)
            hi = np.max([s["max"] for s in shelves], axis=0)
            label = f"{index:02d}/{row}R"
            if variant:
                from .landmark_variants import label_for

                label = label_for(index, row, variant)
            # A double-sided flag next to the upper front upright; no second ID
            # on another side of the rack. Its two faces are only 12 mm apart.
            center = np.array(
                [hi[0] + 0.34 if row == 1 else lo[0] - 0.34, lo[1] + 0.50, 2.55]
            )
            is_surface = variant == "surface" or (
                variant == "combined" and index % 2 == 0
            )
            if is_surface:
                center[2] = 1.15
            name = f"/ShelfLabels/R{row:02d}_B{index:02d}"
            UsdGeom.Xform.Define(stage, name)
            texture = textures / f"R{row:02d}_B{index:02d}.png"
            if variant:
                from .landmark_variants import texture as variant_texture

                variant_texture(texture, label, "surface" if is_surface else variant)
            else:
                label_texture(texture, label)
            material = UsdShade.Material.Define(stage, name + "/Material")
            shader = UsdShade.Shader.Define(stage, name + "/Material/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
            tex = UsdShade.Shader.Define(stage, name + "/Material/Texture")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
                str(texture.resolve())
            )
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            uv = UsdShade.Shader.Define(stage, name + "/Material/UV")
            uv.CreateIdAttr("UsdPrimvarReader_float2")
            uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
                uv.ConnectableAPI(), "result"
            )
            shader.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(tex.ConnectableAPI(), "rgb")
            # Modest emissive component keeps printed text legible under warehouse lights.
            shader.CreateInput(
                "emissiveColor", Sdf.ValueTypeNames.Color3f
            ).ConnectToSource(tex.ConnectableAPI(), "rgb")
            if variant == "lighting":
                shader.GetInput("emissiveColor").DisconnectSource()
                shader.GetInput("emissiveColor").Set(Gf.Vec3f(0, 0, 0))
            shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            if is_surface:
                box = UsdGeom.Cube.Define(stage, name + "/Container")
                box.AddTranslateOp().Set(Gf.Vec3d(*center))
                box.AddScaleOp().Set(Gf.Vec3f(0.42, 0.30, 0.23))
                box.CreateDisplayColorAttr([Gf.Vec3f(0.57, 0.59, 0.58)])
                add_update_semantics(box.GetPrim(), "container")
            for side in (-1, 1):
                mesh = UsdGeom.Mesh.Define(
                    stage, name + ("/South" if side < 0 else "/North")
                )
                # Face local image-right reverses on the north side (no mirrored text).
                right = np.array([-side, 0.0, 0.0])
                c = center + [0, side * (0.301 if is_surface else 0.006), 0]
                vertices = [
                    c + right * dx + [0, 0, dz]
                    for dx, dz in [
                        (-0.40, -0.20),
                        (0.40, -0.20),
                        (0.40, 0.20),
                        (-0.40, 0.20),
                    ]
                ]
                if variant == "occlusion":
                    angle = np.deg2rad(20)
                    rot = np.array(
                        [
                            [np.cos(angle), 0, np.sin(angle)],
                            [0, 1, 0],
                            [-np.sin(angle), 0, np.cos(angle)],
                        ]
                    )
                    vertices = [(np.asarray(v) - c) @ rot.T + c for v in vertices]
                mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
                mesh.CreateFaceVertexCountsAttr([4])
                mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                mesh.CreateSubdivisionSchemeAttr("none")
                mesh.CreateDoubleSidedAttr(False)
                st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                    "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
                )
                st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
                add_update_semantics(
                    mesh.GetPrim(), "container" if is_surface else "shelf_marker"
                )
            # Thin horizontal mounting bracket joins the flag to the rack upright.
            arm = UsdGeom.Cube.Define(stage, name + "/Bracket")
            arm.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), 2.79))
            arm.AddScaleOp().Set(Gf.Vec3f(0.40, 0.022, 0.022))
            arm.CreateDisplayColorAttr([Gf.Vec3f(0.2, 0.2, 0.2)])
            if is_surface:
                arm.GetVisibilityAttr().Set("invisible")
            catalog.append(
                {
                    "label": label,
                    "variant": variant,
                    "category": "container" if is_surface else "shelf_marker",
                    "identity_expected_unique": bool(label) and label != "CAUTION",
                    "text_fully_visible": variant != "occlusion",
                    "prim": name,
                    "center_world": center.tolist(),
                    "size_m": [0.8, 0.012, 0.4],
                    "rack_min": lo.tolist(),
                    "rack_max": hi.tolist(),
                    "shelf_prims": [s["prim"] for s in shelves],
                }
            )
    if len(catalog) != 28 or (
        not variant and len({r["label"] for r in catalog}) != len(catalog)
    ):
        raise ValueError("Warehouse asset layout changed; review shelf discovery")
    (output / "shelf-catalog.json").write_text(json.dumps(catalog, indent=2))
    return catalog
