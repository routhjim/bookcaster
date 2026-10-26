"""Import a USER-SUPPLIED rigged creature model onto our creature template (any book).

    blender -b --factory-startup --disable-autoexec -t 16 --python tools/creature_import_bl.py -- MAP.json OUT_DIR

MAP (book data, e.g. stage/characters/<slug>/import_<name>.json): "source" .blend (opened with scripts disabled),
"mesh" / "armature" object names, "delete" extra objects, "facing" of the model ("-Y" ...), "decimate_verts",
"scale_from" {template_rig rig.json, measure [joint a, joint b]} (or "height_m"), "joints" {template joint: source
bone (its TAIL; "^name" = its HEAD; a list = the mean)}, "targets" {template bone: [source bones whose vertex
groups MERGE into it]}, "parents" (S = l/r), "anchors", "qa_poses". Writes OUT_DIR/rigged.blend ("model" + "rig",
the same layout as our other creature rigs) and rig.json with checks (islands, boundary edges, zero-weight verts,
far verts) + the model's credit.
"""

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MAP = json.loads(Path(argv[0]).read_text())
out = Path(argv[1])
out.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=MAP["source"], load_ui=False, use_scripts=False)
for n in MAP.get("delete", []):
    o = bpy.data.objects.get(n)
    if o:
        bpy.data.objects.remove(o)
src_arm = bpy.data.objects[MAP["armature"]]
src = bpy.data.objects[MAP["mesh"]]

# the model frame -> ours (faces +X, left +Y, Z up)
rot = {"-Y": 90, "+Y": -90, "+X": 0, "-X": 180}[MAP.get("facing", "+X")]
R = np.array(Matrix.Rotation(math.radians(rot), 3, "Z"))
AW = src_arm.matrix_world
B = {b.name: (np.array(AW @ b.head_local), np.array(AW @ b.tail_local)) for b in src_arm.data.bones}


def jpos(spec):
    if isinstance(spec, list):
        return np.mean([jpos(s) for s in spec], axis=0)
    return B[spec[1:]][0] if spec.startswith("^") else B[spec][1]


J0 = {k: R @ jpos(v) for k, v in MAP["joints"].items()}
# scale: a template measure (chest-pelvis of the reference rig) or a height
if MAP.get("scale_from"):
    ref = json.loads(Path(MAP["scale_from"]["template_rig"]).read_text())["joints"]
    a, b = MAP["scale_from"]["measure"]
    s = float(np.linalg.norm(np.asarray(ref[a]) - np.asarray(ref[b]))) / float(np.linalg.norm(J0[a] - J0[b]))
else:
    s = 1.0
# the mesh (rest shape, world space), transformed
me0 = src.data
W0 = np.array(src.matrix_world)
V = np.array([v.co[:] for v in me0.vertices])
V = (W0[:3, :3] @ V.T).T + W0[:3, 3]
V = (R @ V.T).T * s
J = {k: v * s for k, v in J0.items()}
if MAP.get("height_m"):
    k_ = float(MAP["height_m"]) / float(V[:, 2].max() - V[:, 2].min())
    V, J = V * k_, {k: v * k_ for k, v in J.items()}
    s *= k_
off = np.array([-J["pelvis"][0], -J["pelvis"][1], -V[:, 2].min()])
V = V + off
J = {k: v + off for k, v in J.items()}

# the new mesh: same topology, vertex groups MERGED onto the template bones
gname = {g.index: g.name for g in src.vertex_groups}
src_to_t = {}
for t, lst in MAP["targets"].items():
    for n in lst:
        src_to_t[n] = t
nv = len(V)
Wt = {t: np.zeros(nv, np.float32) for t in MAP["targets"]}
unmapped = {}
for v in me0.vertices:
    for g in v.groups:
        n = gname[g.group]
        t = src_to_t.get(n)
        if t:
            Wt[t][v.index] += g.weight
        elif g.weight > 0.01:
            unmapped[n] = unmapped.get(n, 0) + 1
me = bpy.data.meshes.new("model")
me.from_pydata(V.tolist(), [], [list(p.vertices) for p in me0.polygons])
me.update()
ob = bpy.data.objects.new("model", me)
bpy.context.scene.collection.objects.link(ob)
for t, w in Wt.items():
    g = ob.vertex_groups.new(name=t)
    idx = np.where(w > 1e-4)[0]
    for i in idx:
        g.add([int(i)], float(min(w[i], 1.0)), "REPLACE")
# drop the source objects
for o in list(bpy.data.objects):
    if o is not ob:
        bpy.data.objects.remove(o)
bpy.context.view_layer.objects.active = ob
ob.select_set(True)
if MAP.get("decimate_verts") and nv > MAP["decimate_verts"]:
    dm = ob.modifiers.new("dec", "DECIMATE")
    dm.ratio = float(MAP["decimate_verts"]) / nv
    bpy.ops.object.modifier_apply(modifier=dm.name)
for p in ob.data.polygons:
    p.use_smooth = True

# the template armature
ad = bpy.data.armatures.new("rig")
arm = bpy.data.objects.new("rig", ad)
bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="EDIT")
par = {}
for k, p in MAP["parents"].items():
    if "_S_" in k or k.startswith("S_"):
        for sd in ("l", "r"):
            par[k.replace("_S_", f"_{sd}_")] = p.replace("_S_", f"_{sd}_")
    else:
        par[k] = p
eb = {}
pe = ad.edit_bones.new("pelvis")
pe.head = Vector(J["pelvis"].tolist())
pe.tail = Vector((J["pelvis"] + (J["spine1"] - J["pelvis"]) * 0.3).tolist())
eb["pelvis"] = pe
for b in MAP["targets"]:
    e = ad.edit_bones.new(b)
    e.head = Vector(J[par[b]].tolist()) if par.get(b) in J else Vector(J["pelvis"].tolist())
    e.tail = Vector(J[b].tolist())
    if (e.tail - e.head).length < 1e-4:
        e.tail = e.head + Vector((0, 0, 0.05))
    eb[b] = e
for b, e in eb.items():
    if par.get(b) in eb:
        e.parent = eb[par[b]]
bpy.ops.object.mode_set(mode="OBJECT")
ob.parent = arm
m = ob.modifiers.new("rig", "ARMATURE")
m.object = arm
bpy.context.view_layer.objects.active = ob
bpy.ops.object.vertex_group_limit_total(group_select_mode="ALL", limit=4)
bpy.ops.object.vertex_group_normalize_all(lock_active=False)

# checks
bm = bmesh.new()
bm.from_mesh(ob.data)
bm.verts.ensure_lookup_table()
seen, isl = set(), []
for v in bm.verts:
    if v.index in seen:
        continue
    st, n = [v], 0
    seen.add(v.index)
    while st:
        x = st.pop()
        n += 1
        for e in x.link_edges:
            y = e.other_vert(x)
            if y.index not in seen:
                seen.add(y.index)
                st.append(y)
    isl.append(n)
nb = sum(1 for e in bm.edges if e.is_boundary)
bm.free()
P = np.array([v.co[:] for v in ob.data.vertices])
lo, hi = np.min(list(J.values()), axis=0), np.max(list(J.values()), axis=0)
ext = hi - lo
far = int(np.sum(np.any((P < lo - 0.5 * ext) | (P > hi + 0.5 * ext), axis=1)))
zero = sum(1 for v in ob.data.vertices if sum(g.weight for g in v.groups) < 1e-4)
checks = {"islands": len(isl), "largest_island": max(isl), "boundary_edges": nb, "zero_weight": zero, "far_verts": far,
          "verts": len(P), "scale": s, "unmapped_groups": unmapped}
print("IMPORT_CHECK", json.dumps({k: v for k, v in checks.items() if k != "unmapped_groups"}), "unmapped", len(unmapped))
bpy.ops.wm.save_as_mainfile(filepath=str(out / "rigged.blend"))
H = float(P[:, 2].max())
rj = {"template": "winged_quadruped", "clean": True, "imported": MAP["source"], "credit": MAP.get("credit"),
      "height": H, "joints": {k: v.round(4).tolist() for k, v in J.items()}, "anchors": MAP.get("anchors", {}),
      "qa_poses": MAP.get("qa_poses", {}), "skin": "source model weights (merged)", "check": checks,
      "bones": ["pelvis"] + list(MAP["targets"])}
(out / "rig.json").write_text(json.dumps(rj, indent=1))
