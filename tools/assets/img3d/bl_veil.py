"""Blender 4.5 (background): a small bonnet and a net veil for a combo character, built from the MetaHuman head mesh,
as two garments on the combo's skeleton (the character tools import each like an outfit; the veil's material is
masked by a net opacity map, see char_render.import_outfit "T_outfit_Opacity.png").

    blender -b --factory-startup --python bl_veil.py -- JOB.json

JOB {"combo": bl_combo.py's out folder (combo.blend: the skeleton and its bind pose),
     "head_fbx": the character's head FBX (as for bl_combo), "out": folder,
     "state": "down" (the veil hangs over the face to below the chin) | "raised" (turned back over the bonnet: the face
              open, the net framing it and lying doubled over the crown),
     "hair_up_m": 0.07 (how far an up-do stands above and behind the scalp: the bonnet sits on it),
     "clear_face_m": 0.035, "clear_back_m": 0.05, "front_hem_below_chin_m": 0.06, "back_hem": "spine_03",
     "net_cell_m": 0.006, "colour": [0.02, 0.02, 0.022]}
Out: out/bonnet/outfit.fbx + textures, out/veil/outfit.fbx + textures (with T_outfit_Opacity.png), report.json.
Weights: everything above the chin rides the head; below it the veil blends to neck_01 then spine_05 (and the back
panel to spine_03/04), so a turned head swings the face panel and the shoulders keep the hem.
"""
import json
import math
import os
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Vector

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
OUT = JOB["out"]
STATE = JOB.get("state", "down")
REP = {"state": STATE}
bpy.ops.wm.open_mainfile(filepath=os.path.join(JOB["combo"], "combo.blend"))
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
for o in list(bpy.data.objects):
    if o is not arm:
        bpy.data.objects.remove(o)
CM = 100.0
B = arm.data.bones


def J(n):
    return np.array((arm.matrix_world @ B[n].head_local)[:]) / CM


before = set(bpy.data.objects)
bpy.ops.import_scene.fbx(filepath=JOB["head_fbx"], use_anim=False, ignore_leaf_bones=False,
                         automatic_bone_orientation=False)
new = [o for o in bpy.data.objects if o not in before]
hm = max((o for o in new if o.type == "MESH"), key=lambda o: len(o.data.vertices))
hv = np.array([(hm.matrix_world @ v.co)[:] for v in hm.data.vertices])
if hv[:, 2].max() > 10:
    hv /= 100.0
for o in new:
    bpy.data.objects.remove(o)

hj, nj = J("head"), J("neck_01")
cx, cy = float(hj[0]), float(hj[1])
top = float(hv[:, 2].max())
prof = hv[np.abs(hv[:, 0] - cx) < 0.012]
zs = np.arange(J("neck_02")[2] - 0.03, hj[2] + 0.02, 0.003)
fy = np.array([prof[np.abs(prof[:, 2] - z) < 0.0015, 1].min() if (np.abs(prof[:, 2] - z) < 0.0015).any() else np.nan
               for z in zs])
ok = ~np.isnan(fy)
chin = float(zs[ok][int(np.argmin(np.diff(fy[ok])))])
REP.update(chin_z=round(chin, 3), top_z=round(top, 3))
HAIR = float(JOB.get("hair_up_m", 0.07))
CF, CB = float(JOB.get("clear_face_m", 0.035)), float(JOB.get("clear_back_m", 0.05))
FA = -math.pi / 2                     # the front: -Y


def hull_r(z, ang, pts_xy_z):
    """Largest radius of the head points within 1 cm of height z along direction ang from (cx, cy)."""
    band = pts_xy_z[np.abs(pts_xy_z[:, 2] - z) < 0.01]
    if len(band) < 8:
        return None
    d = np.array([math.cos(ang), math.sin(ang)])
    rel = band[:, :2] - [cx, cy]
    along = rel @ d
    perp = np.abs(rel @ np.array([-d[1], d[0]]))
    sel = perp < 0.02
    return float(along[sel].max()) if sel.any() else None


# ------------------------------------------------------------------ bonnet: a shallow cap over the crown and up-do
NA = 48
z_brim = top - 0.045
bon_rings = []
for k in range(7):
    t = k / 6
    ring = []
    for a in range(NA):
        ang = FA + 2 * math.pi * a / NA
        back = 0.5 - 0.5 * math.cos(ang - FA)
        r0 = hull_r(z_brim, ang, hv) or 0.08
        r = (r0 + 0.012 + HAIR * 0.35 * back) * math.cos(t * math.pi / 2) ** 0.7
        z = z_brim + (0.012 + HAIR * (0.25 + 0.5 * back)) * math.sin(t * math.pi / 2) + 0.01 * back
        ring.append([cx + r * math.cos(ang), cy - 0.01 * back + r * math.sin(ang), z])
    bon_rings.append(ring)
bv = [p for r in bon_rings for p in r] + [[cx, cy - 0.01, bon_rings[-1][0][2] + 0.003]]
bf = []
for k in range(6):
    for a in range(NA):
        b = (a + 1) % NA
        bf.append((k * NA + a, k * NA + b, (k + 1) * NA + b, (k + 1) * NA + a))
cap = len(bv) - 1
for a in range(NA):
    bf.append((6 * NA + a, 6 * NA + (a + 1) % NA, cap))

# ------------------------------------------------------------------ veil: rings from the brim down
z_front_hem = chin - float(JOB.get("front_hem_below_chin_m", 0.06))
z_back_hem = float(J(JOB.get("back_hem", "spine_03"))[2])
NR = 22
veil_rings, uvs = [], []
open_front = math.radians(JOB.get("raised_open_deg", 62)) if STATE == "raised" else 0.0
for k in range(NR + 1):
    t = k / NR
    ring = []
    for a in range(NA):
        ang = FA + 2 * math.pi * a / NA
        back = 0.5 - 0.5 * math.cos(ang - FA)
        hem = z_front_hem + (z_back_hem - z_front_hem) * back ** 1.5
        z = z_brim + 0.005 - (z_brim + 0.005 - hem) * t
        zq = max(z, chin + 0.01)                      # under the chin the veil keeps the jaw's clearance
        r0 = max(hull_r(zq, ang, hv) or 0.08, hull_r(z_brim, ang, hv) * 0.85 if hull_r(z_brim, ang, hv) else 0.07)
        r = r0 + CF + (CB - CF) * back + HAIR * 0.3 * back * (1 - t)
        r += 0.05 * max(0.0, (chin - z)) / 0.1        # flares a little below the chin, over the collar
        r += 0.012 * t                               # and as it falls
        ring.append([cx + r * math.cos(ang), cy + r * math.sin(ang), z])
    veil_rings.append(ring)
vv = [p for r in veil_rings for p in r]
cell = float(JOB.get("net_cell_m", 0.006))
tile = 8 * cell                                     # the net texture holds 8 x 8 cells
vuv = []
for k, r in enumerate(veil_rings):
    s = 0.0
    for a in range(NA):
        if a:
            s += math.dist(r[a][:2], r[a - 1][:2])
        vuv.append((s / tile, (z_brim - r[a][2]) / tile))
vf = []
for k in range(NR):
    for a in range(NA):
        b = (a + 1) % NA
        if open_front:
            ma = abs(((2 * math.pi * a / NA) + math.pi) % (2 * math.pi) - math.pi)
            mb = abs(((2 * math.pi * b / NA) + math.pi) % (2 * math.pi) - math.pi)
            if min(ma, mb) < open_front and veil_rings[k + 1][a][2] < z_brim - 0.015:
                continue
        vf.append((k * NA + a, k * NA + b, (k + 1) * NA + b, (k + 1) * NA + a))
fold = []
if STATE == "raised":
    # the front panel turned back: a second net layer lying over the bonnet from its brim to the nape
    base = len(vv)
    for k in range(9):
        t = k / 8
        for a in range(NA):
            p = bon_rings[min(6, int(t * 6))][a]
            ang = FA + 2 * math.pi * a / NA
            back = 0.5 - 0.5 * math.cos(ang - FA)
            q = [p[0] + 0.008 * math.cos(ang), p[1] + 0.008 * math.sin(ang), p[2] + 0.006]
            if back > 0.6:
                q[2] -= 0.12 * t * (back - 0.6) / 0.4
            vv.append(q)
            vuv.append((a * 0.03 / tile, k * 0.02 / tile))
    for k in range(8):
        for a in range(NA):
            b = (a + 1) % NA
            vf.append((base + k * NA + a, base + k * NA + b, base + (k + 1) * NA + b, base + (k + 1) * NA + a))
REP.update(z_brim=round(z_brim, 3), z_front_hem=round(z_front_hem, 3), z_back_hem=round(z_back_hem, 3))


def make(name, verts, faces, uv=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(float(c) * CM for c in v) for v in verts], [], faces)
    me.update()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    lay = me.uv_layers.new(name="UVMap")
    li = np.empty(len(me.loops), np.int64)
    me.loops.foreach_get("vertex_index", li)
    U = np.array(uv if uv is not None else [(0.5, 0.5)] * len(verts), np.float32)
    lay.data.foreach_set("uv", U[li].ravel())
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    # weights: head above the chin, then neck_01 -> spine_05 -> spine_03 by height
    P = np.array(verts)
    g = {n: ob.vertex_groups.new(name=n) for n in ("head", "neck_01", "spine_05", "spine_04", "spine_03")}
    zn, z5 = float(J("neck_01")[2]), float(J("spine_05")[2])
    for i, p in enumerate(P):
        z = p[2]
        if z >= chin:
            g["head"].add([i], 1.0, "REPLACE")
        elif z >= zn:
            u = (chin - z) / max(chin - zn, 1e-3)
            g["head"].add([i], 1 - u, "REPLACE")
            g["neck_01"].add([i], u, "REPLACE")
        elif z >= z5:
            u = (zn - z) / max(zn - z5, 1e-3)
            g["neck_01"].add([i], 1 - u, "REPLACE")
            g["spine_05"].add([i], u, "REPLACE")
        else:
            u = min(1.0, (z5 - z) / 0.15)
            g["spine_05"].add([i], 1 - u, "REPLACE")
            g["spine_04"].add([i], u * 0.6, "REPLACE")
            g["spine_03"].add([i], u * 0.4, "REPLACE")
    ob.parent = arm
    m = ob.modifiers.new("Armature", "ARMATURE")
    m.object = arm
    mat = bpy.data.materials.new(f"M_{name}")
    me.materials.append(mat)
    return ob


def export(ob, sub, rgb, opacity=None):
    d = os.path.join(OUT, sub)
    os.makedirs(os.path.join(d, "textures"), exist_ok=True)
    for o in bpy.data.objects:
        o.select_set(False)
    ob.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.export_scene.fbx(filepath=os.path.join(d, "outfit.fbx"), use_selection=True,
                             object_types={"ARMATURE", "MESH"}, apply_unit_scale=False, global_scale=1.0,
                             apply_scale_options="FBX_SCALE_NONE", axis_forward="-Z", axis_up="Y",
                             add_leaf_bones=False, use_armature_deform_only=False, primary_bone_axis="Y",
                             secondary_bone_axis="X", armature_nodetype="NULL", bake_anim=False,
                             mesh_smooth_type="FACE", use_tspace=True, path_mode="STRIP")
    maps = [("BaseColor", list(rgb), False), ("Normal", [0.5, 0.5, 1.0], True), ("RMA", [0.85, 0.0, 1.0], True)]
    for name, c, lin in maps:
        im = bpy.data.images.new(f"T_{sub}_{name}", 16, 16, alpha=False)
        if lin:
            im.colorspace_settings.name = "Non-Color"
        im.pixels.foreach_set(np.tile(np.array(c + [1.0], np.float32), 16 * 16))
        im.filepath_raw = os.path.join(d, "textures", f"T_outfit_{name}.png")
        im.file_format = "PNG"
        im.save()
    if opacity is not None:
        n = opacity.shape[0]
        im = bpy.data.images.new(f"T_{sub}_Opacity", n, n, alpha=False)
        im.colorspace_settings.name = "Non-Color"
        im.pixels.foreach_set(np.dstack([opacity, opacity, opacity, np.ones_like(opacity)]).astype(np.float32).ravel())
        im.filepath_raw = os.path.join(d, "textures", "T_outfit_Opacity.png")
        im.file_format = "PNG"
        im.save()


col = list(JOB.get("colour", [0.02, 0.02, 0.022]))
bon = make("bonnet", bv, bf)
export(bon, "bonnet", col)
bpy.data.objects.remove(bon)
veil = make("veil", vv, vf, vuv)
# the net: a diamond mesh of 8 x 8 cells per tile, threads ~1/7 of a cell wide, nearly opaque; the holes let
# NET_HOLE of the light through (a mourning veil hides the face's detail, not its shape)
N = 256
y, x = np.mgrid[0:N, 0:N] / N * 8
u, v = (x + y) % 1.0, (x - y) % 1.0
thread = (np.minimum(u, 1 - u) < 0.07) | (np.minimum(v, 1 - v) < 0.07)
op = np.where(thread, 1.0, float(JOB.get("net_hole_opacity", 0.18)))
export(veil, "veil", col, op)
REP["verts"] = {"bonnet": len(bv), "veil": len(vv)}
json.dump(REP, open(os.path.join(OUT, "report.json"), "w"), indent=1)
print("VEIL", json.dumps(REP))
