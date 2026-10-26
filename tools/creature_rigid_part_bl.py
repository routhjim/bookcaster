"""HYBRID creature: a DETAILED rigid part (e.g. the head, image-to-3D from a head-only reference) on a simple
deforming procedural body (any book; creature_spec.json "rigid_parts").

    blender -b --factory-startup -t 16 --python tools/creature_rigid_part_bl.py -- RIGGED.blend OUT.blend \
        --part head.glb --paint head_ref_rgba.png --bone head [--jaw jaw] [--neck neck5] [--scale 1.15]

Detail goes where the part is RIGID, simplicity where the body deforms:
1. the part mesh (largest island, holes filled), painted by a front projection of its own reference drawing (the
   image-to-3D import frame: screen right = +X, up = +Z);
2. seated on the body: forward = from its neck stub to its snout (the lower part's farthest point), rotated so it
   points +X like the body, scaled to the body's own skull length x --scale, the neck stub sunk into the neck;
3. with --jaw: the lower jaw cut along the mouth plane (in front of a hinge 65 % back from the snout) as a separate
   rigid piece on the jaw bone; both cuts capped with a dark mouth-interior material;
4. weights: the skull 100 % on --bone, the jaw piece 100 % on --jaw, a short COLLAR band at the neck stub blended
   into --neck (no hard seam when the neck bends); the body's own head shrunk inside the new one;
5. joined into the body mesh "model" (one object, so every renderer / QA tool reads it unchanged).
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

a = sys.argv[sys.argv.index("--") + 1:]
src, dst = a[0], a[1]


def arg(k, d=None):
    return a[a.index(k) + 1] if k in a else d


bpy.ops.wm.open_mainfile(filepath=src)
body = bpy.data.objects["model"]
arm = bpy.data.objects["rig"]
RJ = json.loads((Path(src).parent / "rig.json").read_text())
J = {k: np.asarray(v, float) for k, v in RJ["joints"].items()}
S = float(RJ.get("height", 6.0))
BONE, JAW, NECK = arg("--bone", "head"), arg("--jaw"), arg("--neck", "neck5")

# 1. the part
bpy.ops.import_scene.gltf(filepath=arg("--part"))
hobs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
bpy.ops.object.select_all(action="DESELECT")
for o in hobs:
    o.select_set(True)
bpy.context.view_layer.objects.active = hobs[0]
if len(hobs) > 1:
    bpy.ops.object.join()
hd = bpy.context.view_layer.objects.active
hd.parent = None
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bm = bmesh.new()
bm.from_mesh(hd.data)
bm.verts.ensure_lookup_table()
lab, isl = {}, []
for v in bm.verts:
    if v.index in lab:
        continue
    st, mem = [v], []
    lab[v.index] = 1
    while st:
        x = st.pop()
        mem.append(x)
        for e in x.link_edges:
            y = e.other_vert(x)
            if y.index not in lab:
                lab[y.index] = 1
                st.append(y)
    isl.append(mem)
big = max(isl, key=len)
bmesh.ops.delete(bm, geom=[v for m in isl if m is not big for v in m], context="VERTS")
bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0)
bm.to_mesh(hd.data)
bm.free()
HV = np.array([v.co[:] for v in hd.data.vertices])
# paint: front projection of its reference (import frame)
if arg("--paint"):
    im = bpy.data.images.load(arg("--paint"))
    w_, h_ = im.size
    px = np.array(im.pixels[:], np.float32).reshape(h_, w_, 4)[::-1]
    ys, xs = np.where(px[:, :, 3] > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    lo, hi = HV.min(0), HV.max(0)
    u = (HV[:, 0] - lo[0]) / max(hi[0] - lo[0], 1e-9)
    vv = (hi[2] - HV[:, 2]) / max(hi[2] - lo[2], 1e-9)
    xi = np.clip((x0 + u * (x1 - x0)).astype(int), 0, w_ - 1)
    yi = np.clip((y0 + vv * (y1 - y0)).astype(int), 0, h_ - 1)
    C = px[yi, xi, :3] * float(arg("--dark", "0.65"))  # the drawing's paint reads pale under the still lighting
    ca = hd.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.concatenate([C, np.ones((len(C), 1), np.float32)], 1).ravel())

# 2. seat it
z0, z1 = HV[:, 2].min(), HV[:, 2].max()
neckc = HV[HV[:, 2] < z0 + 0.08 * (z1 - z0)].mean(0)
low = HV[(HV[:, 2] > z0 + 0.12 * (z1 - z0)) & (HV[:, 2] < z0 + 0.55 * (z1 - z0))]
hz = low - neckc
hz[:, 2] = 0
tip = low[np.argmax(np.linalg.norm(hz, axis=1))]
fwd = tip - neckc
fwd[2] = 0
fwd /= max(np.linalg.norm(fwd), 1e-9)
gi = {g.index: g.name for g in body.vertex_groups}
BV = np.array([v.co[:] for v in body.data.vertices])
dom = np.array([gi.get(max(v.groups, key=lambda g: g.weight).group, "") if len(v.groups) else "" for v in body.data.vertices])
skull = np.isin(dom, [BONE, "head_end", JAW or "jaw"])
len_body = float(np.ptp(BV[skull][:, 0]))
len_new = float(np.ptp(HV @ fwd))
s = len_body / max(len_new, 1e-6) * float(arg("--scale", "1.15"))
ang = -math.atan2(fwd[1], fwd[0])
Rz = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
seat = J.get(NECK, J[BONE]) + (J[BONE] - J.get(NECK, J[BONE])) * 0.35
A = Rz * s
# SNOUT-TIP alignment: the new snout tip on the body's snout tip (x, z), centred in y; the neck stub then
# reaches back/down into the neck by itself
bs_tip = BV[skull][np.argmax(BV[skull][:, 0])]
off = np.array([bs_tip[0], 0.0, bs_tip[2]]) - A @ tip
off[1] = -(A @ HV.mean(0))[1]
T = Matrix.Identity(4)
for i in range(3):
    for j in range(3):
        T[i][j] = float(A[i, j])
    T[i][3] = float(off[i])
hd.data.transform(T)
hd.data.update()
W = np.array([v.co[:] for v in hd.data.vertices])

# 3. the jaw cut (the mouth plane at ~40 % of the snout's height, in front of a hinge 65 % back from the tip)
jaw_ob = None
if JAW:
    snout = W[np.argmax(W[:, 0])]
    front = W[W[:, 0] > snout[0] - 0.3 * np.ptp(W[:, 0])]
    mz = front[:, 2].min() + 0.42 * np.ptp(front[:, 2])
    hinge_x = snout[0] - 0.65 * (snout[0] - (A @ neckc + off)[0])
    bm = bmesh.new()
    bm.from_mesh(hd.data)
    res = bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                                 plane_co=Vector((0, 0, mz)), plane_no=Vector((0, 0, 1)))
    cut_e = [e for e in res["geom_cut"] if isinstance(e, bmesh.types.BMEdge)
             and min(v.co.x for v in e.verts) > hinge_x]
    bmesh.ops.split_edges(bm, edges=cut_e)
    jdepth = 0.5 * (mz - front[:, 2].min())               # the jaw's own depth below the mouth line (not the throat)
    jf = [f for f in bm.faces if mz - jdepth * 2.2 < f.calc_center_median().z < mz and f.calc_center_median().x > hinge_x
          and f.calc_center_median().x > hinge_x + 0.25 * (snout[0] - hinge_x) * (1 if f.calc_center_median().z < mz - jdepth else 0)]
    # the jaw faces -> their own mesh
    jm = bpy.data.meshes.new("jaw_part")
    bm2 = bmesh.new()
    vmap = {}
    cl2 = bm.verts.layers.float_color.get("Col")
    for f in jf:
        vs = []
        for v in f.verts:
            if v not in vmap:
                nv = bm2.verts.new(v.co)
                vmap[v] = nv
            vs.append(vmap[v])
        try:
            bm2.faces.new(vs)
        except ValueError:
            pass
    bmesh.ops.delete(bm, geom=jf, context="FACES")
    capA = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=0).get("faces", [])
    capB = bmesh.ops.holes_fill(bm2, edges=[e for e in bm2.edges if e.is_boundary], sides=0).get("faces", [])
    for f in capA:
        f.material_index = 1
    for f in capB:
        f.material_index = 1
    bm.to_mesh(hd.data)
    bm2.to_mesh(jm)
    bm.free()
    bm2.free()
    jaw_ob = bpy.data.objects.new("jaw_part", jm)
    bpy.context.scene.collection.objects.link(jaw_ob)
    # its colours: the nearest skull vertex's
    if hd.data.color_attributes.get("Col"):
        from mathutils.kdtree import KDTree
        Wc = np.array([v.co[:] for v in hd.data.vertices])
        cc = np.zeros(len(Wc) * 4, np.float32)
        hd.data.color_attributes["Col"].data.foreach_get("color", cc)
        cc = cc.reshape(-1, 4)
        kd = KDTree(len(Wc))
        for i, p in enumerate(Wc):
            kd.insert(p, i)
        kd.balance()
        ca2 = jm.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        for i, v in enumerate(jm.vertices):
            _, k, _ = kd.find(v.co)
            ca2.data[i].color = tuple(cc[k])
    print("RIGID jaw cut at z", round(mz, 3), "hinge x", round(hinge_x, 3), "jaw faces", len(jf))

# 4. weights; the body's own head inside the new one
mi = bpy.data.materials.new("mouth")
mi.use_nodes = True
mi.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.12, 0.02, 0.02, 1)
for ob_, bone in ((hd, BONE), (jaw_ob, JAW)):
    if ob_ is None:
        continue
    g = ob_.vertex_groups.new(name=bone)
    g.add(list(range(len(ob_.data.vertices))), 1.0, "REPLACE")
    while len(ob_.data.materials) < 1:
        ob_.data.materials.append(body.data.materials[0] if body.data.materials else None)
    ob_.data.materials.append(mi)
# the COLLAR: the new head's lowest band blends into the neck bone
Wv = np.array([v.co[:] for v in hd.data.vertices])
zc0 = Wv[:, 2].min()
band = 0.12 * float(np.ptp(Wv[:, 2]))
gn = hd.vertex_groups.new(name=NECK)
gh = hd.vertex_groups[BONE]
for i, p in enumerate(Wv):
    t = (p[2] - zc0) / max(band, 1e-6)
    if t < 1:
        gn.add([i], float(1 - t), "REPLACE")
        gh.add([i], float(t), "REPLACE")
# shrink the body's own skull inside
for i in np.where(skull)[0]:
    v = body.data.vertices[int(i)]
    v.co = Vector((J[BONE] + (np.asarray(v.co) - J[BONE]) * 0.55).tolist())

# 5. join into the body
bpy.ops.object.select_all(action="DESELECT")
for o in (hd, jaw_ob):
    if o is not None:
        o.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.join()
body = bpy.context.view_layer.objects.active
body.name = "model"
for p in body.data.polygons:
    p.use_smooth = True
bpy.ops.wm.save_as_mainfile(filepath=dst)
print("RIGID part joined: scale", round(s, 3), "verts", len(body.data.vertices))
