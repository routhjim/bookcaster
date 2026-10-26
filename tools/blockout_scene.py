"""Blender side of the blockout stage: build a scene from a primitives JSON and render clay + depth.

Run by tools/blockout.py, never by hand:
    blender --background --factory-startup --python tools/blockout_scene.py -- scene.json

scene.json: {"size": [W, H], "samples": 16, "frames": [{"name": "start", "prims": [...],
             "camera": {"pos", "R", "lens"}, "clay": "/abs/x_clay.png", "depth": "/abs/x_depth.npy",
             "blend": "/abs/x.blend" (optional)}]}
Prims (as tools/blockout_rig.py makes them): capsule {a, b, r}, ellipsoid {c, M}, box {c, M}
(M columns = half-extent axes), mesh {verts, tris}; each with a "tag": body | prop | hot | wing | set
| ground. Optional "model": {"path": "/abs/x.glb", "M": 4x4} imports a mesh as is (the 3D-model stage).

Rendering: Cycles on the CPU (no GPU, no display), a few samples + OpenImageDenoise for the clay
picture; the depth is Cycles' Z pass read back through the compositor's Viewer node (float, metres
along the view axis; the background is 1e10).
"""

import json
import math
import os
import sys
import time

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
spec = json.load(open(argv[0]))
W, H = spec.get("size", [1536, 864])

COLORS = {"body": (0.75, 0.75, 0.75), "prop": (0.45, 0.45, 0.48), "hot": (1.0, 0.45, 0.1),
          "wing": (0.6, 0.6, 0.6), "set": (0.55, 0.55, 0.55), "ground": (0.42, 0.42, 0.42),
          "model": (0.75, 0.75, 0.75), "spark": (1.0, 0.85, 0.4), "stump": (0.17, 0.1, 0.055)}
# untextured stand-ins (a ledge, a facade, a proxy) read as unpainted WHITE under the bright still lighting: earth tones
COLORS.update({"set": (0.3, 0.27, 0.23), "ground": (0.26, 0.23, 0.2)})


def mat4(M, c):
    m = Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = M[i][j]
        m[i][3] = c[i]
    return m


def frame_from_dir(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    z = b - a
    L = float(np.linalg.norm(z))
    z = z / L if L > 1e-9 else np.array([0, 0, 1.0])
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1), L


def add_prim(bm, p):
    t = p["t"]
    if t == "capsule":
        a, b, r = np.asarray(p["a"]), np.asarray(p["b"]), float(p["r"])
        M, L = frame_from_dir(a, b)
        if L > 1e-4:
            bmesh.ops.create_cone(bm, cap_ends=False, cap_tris=False, segments=16, radius1=r, radius2=r,
                                  depth=L, matrix=mat4(M, (a + b) / 2))
        for c in (a, b):
            bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=r,
                                      matrix=mat4(np.eye(3), c))
    elif t == "ellipsoid":
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0,
                                  matrix=mat4(np.asarray(p["M"]), p["c"]))
    elif t == "box":
        bmesh.ops.create_cube(bm, size=2.0, matrix=mat4(np.asarray(p["M"]), p["c"]))
    elif t == "mesh":
        vs = [bm.verts.new(v) for v in p["verts"]]
        for tri in p["tris"]:
            try:
                bm.faces.new([vs[i] for i in tri])
            except ValueError:
                pass


def material(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    if name != "spark":
        m.diffuse_color = (*rgb, 1.0)                   # the Workbench MATERIAL colour (the contact view)
    if name == "spark" and m.get("fx"):
        return m
    if name != "spark":
        bsdf = m.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.8
    if name == "spark" and not m.get("fx"):
        # glow only: emission ADDED to a fully transparent surface, so a spark lights the clay picture
        # (img2img init, canny) but writes no depth (below the pass alpha threshold)
        nt = m.node_tree
        for n_ in list(nt.nodes):
            nt.nodes.remove(n_)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (1.0, 0.82, 0.38, 1.0)
        em.inputs["Strength"].default_value = 30.0
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        add = nt.nodes.new("ShaderNodeAddShader")
        nt.links.new(em.outputs[0], add.inputs[0])
        nt.links.new(tr.outputs[0], add.inputs[1])
        nt.links.new(add.outputs[0], out.inputs["Surface"])
        m.diffuse_color = (1.0, 0.85, 0.4, 1.0)
        m["fx"] = True
        return m
    if name == "hot":
        bsdf.inputs["Emission Color"].default_value = (1.0, 0.4, 0.05, 1.0)
        bsdf.inputs["Emission Strength"].default_value = 2.0
    return m


def clear():
    keep = {o for o in bpy.data.objects if o.get("keep")}
    for o in list(bpy.data.objects):
        if o not in keep:
            bpy.data.objects.remove(o, do_unlink=True)
    used = {o.data for o in keep if o.type == "MESH"}
    for me in list(bpy.data.meshes):
        if me not in used:
            bpy.data.meshes.remove(me)


_WORLD = {}
_MODELS = {}


def ensure_world(wspec):
    """A HY-World set (tools/world.py output) as the scene's set: imported ONCE and kept across frames
    (world_blender.import_world), uniformly scaled by "scale" about the origin (a metric fix: e.g. match a
    known prop's height), unlit painted colours ("emission") or shaded ("principled")."""
    key = (wspec["dir"], float(wspec.get("scale", 1.0)), wspec.get("material", "emission"), wspec.get("mesh"),
           tuple(wspec.get("offset", (0, 0, 0))))
    ob = _WORLD.get(key)
    try:
        if ob is not None and ob.name in bpy.data.objects:
            return ob
    except ReferenceError:                             # deleted by a scene rebuild: import again
        pass
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import world_blender as wb
    kw = dict(material=wspec.get("material", "emission"), backdrop=bool(wspec.get("backdrop", True)))
    if wspec.get("mesh"):
        kw["mesh"] = wspec["mesh"]
    if wspec.get("holdout"):
        kw["holdout"] = True
    w = wb.import_world(wspec["dir"], **kw)
    sc_ = float(wspec.get("scale", 1.0))
    off = wspec.get("offset", (0, 0, 0))            # e.g. lift the terrain under the action to z = 0
    ob = w["objects"]["set"]
    for o in w["objects"].values():
        if o is not None:
            o.scale = (sc_, sc_, sc_)
            o.location = tuple(float(x) for x in off)
            o["keep"] = True

    _WORLD[key] = ob
    return ob


_PROPMESH = {}


def import_propmesh(p):
    """A modelled prop (tools/props_bl.py blend: object "prop" + "joint_<name>" children): linked once per
    blend, copied per use, placed by p["M"] (the prop frame: axes x scale + origin); joint angles (deg) turn
    each part about its pivot; a state's tint/emission recolours it (a heated blade glows)."""
    path = p["blend"]
    # cache the object NAME (a frame rebuild deletes objects: a cached object reference raises ReferenceError)
    nm = _PROPMESH.get(path)
    src_root = bpy.data.objects.get(nm) if nm else None
    if src_root is None:
        with bpy.data.libraries.load(path) as (src, dst):
            dst.objects = [n for n in src.objects if n == "prop" or n.startswith("joint_")]
        objs = [o for o in dst.objects if o is not None]
        root = next(o for o in objs if o.name.startswith("prop"))
        for o in objs:
            o.hide_render = True
            bpy.context.scene.collection.objects.link(o)
        _PROPMESH[path] = root.name
        src_root = root
    root = src_root.copy()
    root.hide_render = False
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = Matrix(p["M"])
    jdefs = {j["name"]: j for j in p.get("joint_defs") or []}
    for ch in src_root.children:
        c = ch.copy()
        c.hide_render = False
        c.parent = root
        bpy.context.scene.collection.objects.link(c)
        nm = ch.name.split(".")[0].replace("joint_", "")
        ang = float((p.get("joints") or {}).get(nm, jdefs.get(nm, {}).get("rest", 0.0)))
        ax = {"x": 0, "y": 1, "z": 2}.get(jdefs.get(nm, {}).get("axis", "y"), 1)
        e = [0.0, 0.0, 0.0]
        e[ax] = math.radians(ang)
        c.rotation_euler = e
    st = p.get("state") or {}
    if st.get("tint") or st.get("emission"):
        tint = st.get("tint", [1, 1, 1])
        for o in [root] + list(root.children):
            m = bpy.data.materials.new("state")
            m.use_nodes = True
            b = m.node_tree.nodes["Principled BSDF"]
            vc = m.node_tree.nodes.new("ShaderNodeVertexColor")
            vc.layer_name = "Col"
            mix = m.node_tree.nodes.new("ShaderNodeMix")
            mix.data_type = "RGBA"
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = 0.85
            m.node_tree.links.new(vc.outputs["Color"], mix.inputs[6])
            mix.inputs[7].default_value = (tint[0], tint[1], tint[2], 1)
            m.node_tree.links.new(mix.outputs[2], b.inputs["Base Color"])
            if st.get("emission"):
                b.inputs["Emission Color"].default_value = (tint[0], tint[1], tint[2], 1)
                b.inputs["Emission Strength"].default_value = float(st["emission"])
            m.diffuse_color = (tint[0], tint[1], tint[2], 1)
            o.data = o.data.copy()
            o.data.materials.clear()
            o.data.materials.append(m)
    lt = p.get("light")
    if lt:
        ld = bpy.data.lights.new("prop_light", "POINT")
        ld.energy = float(lt.get("energy", 25))
        ld.color = tuple(lt.get("color", [1.0, 0.75, 0.4]))
        lo = bpy.data.objects.new("prop_light", ld)
        lo.matrix_world = Matrix(p["M"])
        bpy.context.scene.collection.objects.link(lo)
    # a HELD prop is its own depth layer "prop" (10-22 mannequin flow: the set + props depth without the people keeps
    # the hammer and the anvil); "figure" before, so a people-only layer list still takes ["figure", "prop"]
    root["layer"] = "prop" if p.get("tag") in ("prop", "grip") and p.get("layer") == "figure" else \
        ("figure" if p.get("tag") in ("prop", "grip") else "set")
    return root


def build(frame):
    clear()
    if frame.get("world"):
        ensure_world(frame["world"])
    groups = {}
    for p in frame["prims"]:
        if p.get("proxy") and not frame.get("show_proxies"):
            continue                          # a modelled prop's proxy: contacts only, never rendered (the
                                              # contact view, tools/contacts_view.py, shows them)
        if p.get("t") == "propmesh":
            import_propmesh(p)
            continue
        lay_ = "prop" if p.get("tag") in ("prop", "hot", "grip") and p.get("layer") == "figure" else p.get("layer", "")
        groups.setdefault((p.get("tag", "body"), lay_), []).append(p)
    for (tag, layer), ps in groups.items():
        bm = bmesh.new()
        tmp = bpy.data.meshes.new("_prim")
        for p in ps:                # each primitive in its own bmesh, appended (bmesh ops on one big
            bt = bmesh.new()        # growing bmesh cost O(total) each: 6 s a frame for a creature)
            add_prim(bt, p)
            bt.to_mesh(tmp)
            bt.free()
            bm.from_mesh(tmp)
        bpy.data.meshes.remove(tmp)
        me = bpy.data.meshes.new(tag)
        bm.to_mesh(me)
        bm.free()
        for poly in me.polygons:
            poly.use_smooth = tag not in ("set", "ground", "prop")
        ob = bpy.data.objects.new(tag, me)
        ob["layer"] = layer or ("set" if tag in ("set", "ground", "hot") else "")
        ob.data.materials.append(material(tag, {**COLORS, **(frame.get("colors") or {})}.get(tag, (0.7, 0.7, 0.7))))
        bpy.context.scene.collection.objects.link(ob)
    for m in frame.get("models", []):
        mo = import_model(m)
        if m.get("fist") and mo is not None and "fist" in bpy.data.objects:
            # fist proxies take the character's own hand colour (vertex colours weighted to its hand bones)
            col = _hand_colour(mo)
            if col is not None:
                fo = bpy.data.objects["fist"]
                mat = bpy.data.materials.get("fist_" + mo.name) or material("fist_" + mo.name, col)
                fo.data.materials.clear()
                fo.data.materials.append(mat)
        lay = m.get("layer", "creature" if m.get("static") else "figure")
        for o in [mo] + ([mo.parent] if mo is not None and mo.parent else []) + \
                [x for x in bpy.data.objects if x.type == "ARMATURE" and mo is not None and mo.parent is x]:
            if o is not None:
                o["layer"] = lay
        if lay == "creature" and mo is not None and mo.type == "MESH":
            _creature_light(mo, float(frame.get("exposure", 0.85)))
        if m.get("who") and not m.get("static") and not m.get("template"):   # the posed hands (humans)
            arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][-1]
            out = {}
            for s in ("r", "l"):
                pb = arm.pose.bones[f"{s}_hand"]
                h, t = arm.matrix_world @ pb.head, arm.matrix_world @ pb.tail
                out[s] = [list(h), list(t)]
            print("BLOCKOUT_HANDS " + json.dumps({"frame": frame["name"], "who": m["who"], "hands": out}),
                  flush=True)
    # camera
    cam_d = bpy.data.cameras.new("cam")
    cam_d.lens = frame["camera"]["lens"]
    cam_d.sensor_fit = "HORIZONTAL"
    cam_d.sensor_width = 36.0
    cam_d.clip_start = 0.05
    cam_d.clip_end = 2000.0
    cam = bpy.data.objects.new("cam", cam_d)
    cam.matrix_world = mat4(frame["camera"]["R"], frame["camera"]["pos"])
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    # light: a key sun from the camera's upper left + a grey sky
    sun_d = bpy.data.lights.new("sun", "SUN")
    sun_d.energy = float(frame.get("key_light", 7.2))          # the key (faces must read: mean luma >= 0.35)
    sun_d.angle = 0.2
    sun = bpy.data.objects.new("sun", sun_d)
    R = np.asarray(frame["camera"]["R"])
    d = -R[:, 2] + R[:, 0] * 0.6 - R[:, 1] * 0.9          # light travels away from upper left
    Md, _ = frame_from_dir([0, 0, 0], -d)
    sun.matrix_world = mat4(Md, [0, 0, 10])
    bpy.context.scene.collection.objects.link(sun)
    # a soft FILL from the camera side (a dark set otherwise leaves faces black)
    fl_d = bpy.data.lights.new("fill", "AREA")
    fl_d.energy = float(frame.get("fill_light", 440.0))
    fl_d.size = 3.0
    fl_ = bpy.data.objects.new("fill", fl_d)
    cp_ = np.asarray(frame["camera"]["pos"], float)
    Mf, _ = frame_from_dir(cp_.tolist(), (cp_ + R[:, 2]).tolist())   # local -Z (the emission) toward the scene
    fl_.matrix_world = mat4(Mf, (cp_ + R[:, 1] * 1.0 + R[:, 2] * 1.5).tolist())   # behind + above the camera
    bpy.context.scene.collection.objects.link(fl_)
    # EXPOSURE: keys paint at low strength over the still and keep its brightness: + half a stop by default
    # (the set's own pano light and colour kept, just brighter; mean luma ~0.35 on a dark interior)
    bpy.context.scene.view_settings.exposure = float(frame.get("exposure", 0.85))
    # a RIM from behind-above the subject (separates hair and shoulders from a dark set)
    rim_e = float(frame.get("rim_light", 280.0))
    if rim_e > 0:
        rm_d = bpy.data.lights.new("rim", "AREA")
        rm_d.energy = rim_e
        rm_d.size = 2.0
        rm_ = bpy.data.objects.new("rim", rm_d)
        tgt_ = cp_ - R[:, 2] * float(frame.get("subject_dist", 3.0))
        pos_ = tgt_ - R[:, 2] * 2.5 + R[:, 1] * 2.0 + R[:, 0] * 1.0
        Mr, _ = frame_from_dir(tgt_.tolist(), pos_.tolist())    # local -Z (the emission) toward the subject
        rm_.matrix_world = mat4(Mr, pos_.tolist())
        bpy.context.scene.collection.objects.link(rm_)


def _basis(d, l):
    d = d.normalized()
    l = (l - d * l.dot(d)).normalized()
    return Matrix((d, l, d.cross(l))).transposed()          # columns d, l, n


def pose_template(obs, m):
    """Pose a template-rigged creature: the armature object carries the placement M (scale + yaw +
    position); every bone (named by its joint; head = parent joint) turns by the smallest rotation that
    points it at its target joint (targets in WORLD metres, converted to armature space), after its
    parent's turn; the root follows the pelvis target."""
    ob, arm = obs["model"], obs["rig"]
    M = Matrix(m["M"])
    arm.matrix_world = M
    if ob.parent is not arm:
        ob.matrix_world = M
    Minv = M.inverted()
    T = {k: Minv @ Vector(v) for k, v in m["targets"].items()}
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    ad = arm.data
    order = [b.name for b in ad.bones if b.parent is None]
    i = 0
    while i < len(order):
        order += [c.name for c in ad.bones[order[i]].children]
        i += 1
    D, head_p = {}, {}
    for name in order:
        b = ad.bones[name]
        pb = arm.pose.bones[name]
        rest = b.matrix_local.to_3x3()
        rh, rt = b.head_local.copy(), b.tail_local.copy()
        if b.parent is None:
            Dp = Matrix.Identity(3)
            hp = T.get(name, rh)                    # the root joint itself
            Db = Dp
        else:
            pn = b.parent.name
            Dp = D[pn]
            hp = head_p[pn] + Dp @ (rh - b.parent.head_local) if pn in head_p else rh
            pj = b.parent.name                         # bone = parent joint -> this joint
            if name in T and pj in T:
                p_dir = Dp @ (rt - rh)
                t_dir = T[name] - T[pj]
                Db = p_dir.rotation_difference(t_dir).to_matrix() @ Dp
            else:
                Db = Dp
        D[name], head_p[name] = Db, hp
        Mb = (Db @ rest).to_4x4()
        Mb.translation = hp
        pb.matrix = Mb
        bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")
    return ob


_HANDCOL = {}


def _hand_colour(ob):
    """Mean vertex colour of the vertices weighted (> 0.5) to the hand bones, or None."""
    if ob.name in _HANDCOL:
        return _HANDCOL[ob.name]
    col = None
    attr = ob.data.color_attributes.get("Col")
    gi = {g.index for g in ob.vertex_groups if g.name in ("r_hand", "l_hand")}
    if attr is not None and gi:
        cs = []
        for v in ob.data.vertices:
            if any(g.group in gi and g.weight > 0.5 for g in v.groups):
                c = attr.data[v.index].color
                cs.append((c[0], c[1], c[2]))
        if len(cs) > 20:
            col = tuple(float(x) for x in np.mean(np.array(cs), 0))
    _HANDCOL[ob.name] = col
    return col


def _matte(ob):
    """Stills read hair and skin as MATTE (no specular: the silver hair read as metal under the key light);
    keys paint at low strength over the still, so its shading is what they keep."""
    for ms in ob.material_slots:
        mt = ms.material
        if mt is None or not mt.use_nodes or mt.get("_matte"):
            continue
        for nd in list(mt.node_tree.nodes):
            if nd.type == "BSDF_PRINCIPLED" and nd.inputs["Base Color"].is_linked:
                # ALBEDO HIGHLIGHT CLAMP: a silver/white painted texture (hair, a white beard, a linen shirt)
                # must not blow out to pure white under the key: the top of the albedo range compresses to ~0.72
                lk = nd.inputs["Base Color"].links[0]
                cv = mt.node_tree.nodes.new("ShaderNodeRGBCurve")
                c0 = cv.mapping.curves[3]
                c0.points[0].location = (0.0, 0.0)
                c0.points[1].location = (1.0, 0.6)
                c0.points.new(0.5, 0.45)
                cv.mapping.update()
                mt.node_tree.links.new(lk.from_socket, cv.inputs["Color"])
                mt.node_tree.links.new(cv.outputs["Color"], nd.inputs["Base Color"])
        for nd in mt.node_tree.nodes:
            if nd.type == "BSDF_PRINCIPLED":
                for nm, val in (("Specular IOR Level", 0.0), ("Specular", 0.0), ("Roughness", 1.0),
                                ("Coat Weight", 0.0), ("Sheen Weight", 0.0), ("Metallic", 0.0)):
                    if nm in nd.inputs:
                        nd.inputs[nm].default_value = val
        mt["_matte"] = True


def _creature_light(ob, exposure):
    """CREATURE LIGHT PRESET: the still's exposure boost is for people; a creature's dark albedo (burnt iron,
    charcoal scales) must stay dark, so its base colour is scaled down by the same exposure (2^-exposure)."""
    k = 2.0 ** -exposure
    for ms in ob.material_slots:
        mt = ms.material
        if mt is None or mt.get("_creature_lit"):
            continue
        if mt.use_nodes:
            for nd in list(mt.node_tree.nodes):
                if nd.type != "BSDF_PRINCIPLED":
                    continue
                bc = nd.inputs["Base Color"]
                if bc.is_linked:
                    mx = mt.node_tree.nodes.new("ShaderNodeMix")
                    mx.data_type = "RGBA"
                    mx.blend_type = "MULTIPLY"
                    mx.inputs[0].default_value = 1.0
                    mx.inputs[7].default_value = (k, k, k, 1.0)
                    lk = bc.links[0]
                    mt.node_tree.links.new(lk.from_socket, mx.inputs[6])
                    mt.node_tree.links.new(mx.outputs[2], bc)
                else:
                    c = bc.default_value
                    bc.default_value = (c[0] * k, c[1] * k, c[2] * k, 1.0)
        mt["_creature_lit"] = True


def creature_mask(path, size):
    """A mask of the CREATURE layer in the camera (white = creature): the key painter draws it at a higher
    strength than the rest (differential strength), so a plain procedural body is embellished, the set kept.
    (Blender's python: no PIL -> numpy triangle fill, saved as a Blender image.)"""
    from bpy_extras.object_utils import world_to_camera_view
    sc_ = bpy.context.scene
    cam_ = sc_.camera
    Wd, Hd = int(size[0]), int(size[1])
    M = np.zeros((Hd, Wd), np.float32)
    dg_ = bpy.context.evaluated_depsgraph_get()
    n_ = 0
    for o in bpy.data.objects:
        if o.type != "MESH" or o.hide_render or o.get("layer") != "creature":
            continue
        ev = o.evaluated_get(dg_)
        me_ = ev.to_mesh()
        me_.calc_loop_triangles()
        P2 = np.array([tuple(world_to_camera_view(sc_, cam_, ev.matrix_world @ v.co)) for v in me_.vertices])
        P2[:, 0] *= Wd
        P2[:, 1] = (1 - P2[:, 1]) * Hd
        T = np.array([t.vertices[:] for t in me_.loop_triangles])
        ev.to_mesh_clear()
        n_ += 1
        for tri in T:
            q = P2[tri]
            if np.any(q[:, 2] <= 0):
                continue
            x0, x1 = int(max(0, np.floor(q[:, 0].min()))), int(min(Wd - 1, np.ceil(q[:, 0].max())))
            y0, y1 = int(max(0, np.floor(q[:, 1].min()))), int(min(Hd - 1, np.ceil(q[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            (ax, ay), (bx, by), (cx_, cy_) = q[0, :2], q[1, :2], q[2, :2]
            d = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
            if abs(d) < 1e-9:
                continue
            l1 = ((by - cy_) * (xs - cx_) + (cx_ - bx) * (ys - cy_)) / d
            l2 = ((cy_ - ay) * (xs - cx_) + (ax - cx_) * (ys - cy_)) / d
            inside = (l1 >= 0) & (l2 >= 0) & (l1 + l2 <= 1)
            M[y0:y1 + 1, x0:x1 + 1][inside] = 1.0
    if n_:
        img = bpy.data.images.new("creature_mask", Wd, Hd, alpha=False)
        px = np.repeat(M[::-1].reshape(-1, 1), 4, axis=1)
        px[:, 3] = 1.0
        img.pixels.foreach_set(px.ravel())
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
        bpy.data.images.remove(img)
    return n_


def import_model(m):
    """A rigged model from the 3D-model stage (tools/blockout_model.py): append its "model" mesh and
    "rig" armature and pose the rig onto the target joints (world space, the joint names of
    blockout_rig.py). Each bone turns by the smallest rotation that points it at its target, after its
    parent's turn; the spine, neck and head also take the body's left-right axis from the hips or the
    shoulders, which carries the facing. Bottom-anchored figures are then put back on the ground."""
    key = (m["blend"], m.get("who"))
    obs = _MODELS.get(key)
    try:
        stale = obs is None or any(o.name not in bpy.data.objects for o in obs.values())
    except ReferenceError:                             # removed by clear() (not kept): import again
        stale = True
    if stale:
        with bpy.data.libraries.load(m["blend"]) as (src, dst):
            dst.objects = [n for n in src.objects if n in ("model", "rig")]
        obs = {o.name.split(".")[0]: o for o in dst.objects if o is not None}
        for o in obs.values():
            bpy.context.scene.collection.objects.link(o)
            if spec.get("engine") == "workbench":      # videos: load once, re-pose every frame
                o["keep"] = True
        _MODELS[key] = obs
        if "model" in obs and not m.get("clay"):
            _matte(obs["model"])
    if "rig" in obs:
        for pb in obs["rig"].pose.bones:                # (re-posed below; fists re-applied)
            pb.scale = (1.0, 1.0, 1.0)
    if m.get("template") and "rig" in obs:       # a rigged CREATURE (template skeleton): pose by targets
        return pose_template(obs, m)
    if m.get("static") or "rig" not in obs:      # an unrigged mesh (a creature): just place it
        ob = obs["model"]
        Mw = Matrix(m.get("M", np.eye(4).tolist()))
        if m.get("feet_anchor"):     # put the centroid of the feet (lowest 12% of the mesh), not the bbox
            #                          centre (pulled back by a long tail), on the placement point
            vs = np.array([v.co[:] for v in ob.data.vertices])
            z0, z1 = vs[:, 2].min(), vs[:, 2].max()
            low = vs[vs[:, 2] < z0 + 0.12 * (z1 - z0)]
            Mw = Mw @ Matrix.Translation(Vector((-float(low[:, 0].mean()), -float(low[:, 1].mean()), 0.0)))
        ob.matrix_world = Mw
        if m.get("clay"):
            ob.data.materials.clear()
            ob.data.materials.append(material("model", COLORS["model"]))
        return ob
    ob, arm = obs["model"], obs["rig"]
    if m.get("clay"):
        mat = material("model", COLORS["model"])
        ob.data.materials.clear()
        ob.data.materials.append(mat)
    T = {k: Vector(v) for k, v in m["targets"].items()}
    T.setdefault("spine_mid", (T["pelvis"] + T["neck"]) / 2)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    ad = arm.data
    lr_hip = T["l_hip"] - T["r_hip"]
    lr_sh = T["l_shoulder"] - T["r_shoulder"]
    D, head_p = {}, {}
    twist = {"pelvis": ("hip", lr_hip), "spine": ("sh", lr_sh), "neck": ("sh", lr_sh), "head": ("sh", lr_sh)}
    order = [b.name for b in ad.bones if b.parent is None]
    i = 0
    while i < len(order):
        order += [c.name for c in ad.bones[order[i]].children]
        i += 1
    tails = {}
    for name in order:
        b = ad.bones[name]
        pb = arm.pose.bones[name]
        rest = b.matrix_local.to_3x3()
        rh, rt = b.head_local.copy(), b.tail_local.copy()
        tails[name] = None
        if b.parent is None:
            hp = T.get("pelvis", rh)
            Dp = Matrix.Identity(3)
        else:
            pn = b.parent.name
            Dp = D[pn]
            hp = head_p[pn] + Dp @ (rh - b.parent.head_local)
        tgt_name = TAIL_OF.get(name)
        if name in twist:
            lr_rest = REST_LR[twist[name][0]]
            d_rest = (rt - rh)
            t_dir = (T[tgt_name] - T[HEAD_OF[name]]) if tgt_name in T and HEAD_OF[name] in T else Dp @ d_rest
            Db = _basis(t_dir, twist[name][1]) @ _basis(d_rest, lr_rest).inverted()
        elif tgt_name in T and HEAD_OF.get(name) in T:
            p_dir = Dp @ (rt - rh)
            t_dir = T[tgt_name] - T[HEAD_OF[name]]
            Q = p_dir.rotation_difference(t_dir).to_matrix()
            Db = Q @ Dp
        else:
            Db = Dp
        D[name] = Db
        head_p[name] = hp
        M = (Db @ rest).to_4x4()
        M.translation = hp
        pb.matrix = M
        bpy.context.view_layer.update()
    for s in m.get("fist", []):                      # gripping hands: fold the (open) mesh hand into the
        pb = arm.pose.bones.get(f"{s}_hand")          # wrist; a fist proxy primitive takes its place
        if pb is not None:
            pb.scale = (0.02, 0.02, 0.02)
    bpy.context.view_layer.update()
    bpy.ops.object.mode_set(mode="OBJECT")
    if m.get("ground", True):
        deps = bpy.context.evaluated_depsgraph_get()
        ev = ob.evaluated_get(deps)
        zmin = min((ev.matrix_world @ v.co).z for v in ev.data.vertices)
        arm.location.z -= zmin
    return ob


HEAD_OF = {"pelvis": "pelvis", "spine": "spine_mid", "neck": "neck", "head": "head", "r_clav": "neck",
           "l_clav": "neck", "r_uarm": "r_shoulder", "l_uarm": "l_shoulder", "r_farm": "r_elbow",
           "l_farm": "l_elbow", "r_hand": "r_wrist", "l_hand": "l_wrist", "r_hipb": "pelvis", "l_hipb": "pelvis",
           "r_thigh": "r_hip", "l_thigh": "l_hip", "r_shin": "r_knee", "l_shin": "l_knee", "r_foot": "r_ankle",
           "l_foot": "l_ankle"}
TAIL_OF = {"pelvis": "spine_mid", "spine": "neck", "neck": "head", "head": "head_top", "r_clav": "r_shoulder",
           "l_clav": "l_shoulder", "r_uarm": "r_elbow", "l_uarm": "l_elbow", "r_farm": "r_wrist",
           "l_farm": "l_wrist", "r_hand": "r_hand", "l_hand": "l_hand", "r_hipb": "r_hip", "l_hipb": "l_hip",
           "r_thigh": "r_knee", "l_thigh": "l_knee", "r_shin": "r_ankle", "l_shin": "l_ankle", "r_foot": "r_toe",
           "l_foot": "l_toe"}
REST_LR = {"hip": Vector((0, 1, 0)), "sh": Vector((0, 1, 0))}      # rest pose faces +X: left = +Y


def setup_render(samples):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.denoiser = "OPENIMAGEDENOISE"
    sc.cycles.max_bounces = 2
    sc.render.resolution_x, sc.render.resolution_y = W, H
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = False
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.view_settings.view_transform = "Standard"
    # HIGHLIGHT ROLL-OFF: silver hair, white linen, polished steel stop clipping to pure white under the key
    sc.view_settings.use_curve_mapping = True
    cm_ = sc.view_settings.curve_mapping
    cc_ = cm_.curves[3]
    cc_.points[0].location = (0.0, 0.0)
    cc_.points[1].location = (1.0, 0.86)
    cc_.points.new(0.62, 0.62)
    cm_.update()
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.25, 0.27, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
    sc.world = world
    sc.view_layers[0].use_pass_z = True
    sc.use_nodes = True
    nt = sc.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    rl = nt.nodes.new("CompositorNodeRLayers")
    comp = nt.nodes.new("CompositorNodeComposite")
    fo = nt.nodes.new("CompositorNodeOutputFile")
    fo.name = "depth_out"
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.format.exr_codec = "ZIP"
    fo.file_slots[0].path = "depth_"
    nt.links.new(rl.outputs["Image"], comp.inputs["Image"])
    nt.links.new(rl.outputs["Depth"], fo.inputs[0])
    sc.render.use_compositing = True
    if spec.get("engine") == "workbench":           # the control-video path: ~0.1 s a frame (OpenGL)
        sc.render.engine = "BLENDER_WORKBENCH"
        sh = sc.display.shading
        sh.light = "STUDIO"
        sh.color_type = "SINGLE"
        sh.single_color = (0.8, 0.8, 0.8)
        sh.show_shadows = False
        sh.show_cavity = False
        sc.display.render_aa = "FXAA"                # 10-23: "8" = 8 draws a frame; the clay is a reference only
        if spec.get("xray"):                        # the contact view: see-through, each part its own colour
            sh.color_type = "MATERIAL"
            sh.show_xray = True
            sh.xray_alpha = float(spec["xray"])


def _exr_depth(exr):
    """The Z of the compositor's EXR as a float32 array (foreach_get: the pixels[:] list took ~0.1 s a read)."""
    img = bpy.data.images.load(exr)
    img.colorspace_settings.name = "Non-Color"
    buf = np.empty(img.size[0] * img.size[1] * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    px = buf.reshape(img.size[1], img.size[0], -1)[::-1, :, 0].copy()
    bpy.data.images.remove(img)
    os.remove(exr)
    return px


setup_render(int(spec.get("samples", 16)))
for fr in spec["frames"]:
    t0 = time.time()
    build(fr)
    t_build = time.time() - t0
    sc = bpy.context.scene
    sc.render.resolution_x, sc.render.resolution_y = fr.get("size", [W, H])
    sc.render.filepath = fr["clay"]
    tmp = fr["depth"] + ".d"
    os.makedirs(tmp, exist_ok=True)
    sc.node_tree.nodes["depth_out"].base_path = tmp
    sc.frame_set(1)
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    t_main = time.time() - t1
    try:                                               # never let the mask break a render (STILLS only: the
        if spec.get("engine") != "workbench":           # per-frame video passes don't need it)
            creature_mask(os.path.splitext(fr["clay"])[0] + "_creature_mask.png", fr.get("size", [W, H]))
    except Exception as ex_:                           # noqa: BLE001
        print("creature_mask failed:", ex_, flush=True)
    exr = os.path.join(tmp, "depth_0001.exr")
    np.save(fr["depth"], _exr_depth(exr))
    t_read = time.time() - t1 - t_main
    t1 = time.time()
    # per-layer depth (fr["layers"] = {name: [layers to keep]}): the other objects hidden, Z re-rendered
    for lname, keep in (fr.get("layers") or {}).items():
        hid = []
        for o in bpy.data.objects:
            if o.type in ("MESH",) and not o.hide_render:
                lay = o.get("layer") or ("set" if o.get("keep") and o.get("world_dir") else "")
                if lay not in keep:
                    o.hide_render = True
                    hid.append(o)
        # depth only (10-23 profile: 3 layer passes = 0.65 s of a 1.1 s frame): no clay image written, no AA
        aa_ = sc.display.render_aa
        sc.display.render_aa = "OFF"
        bpy.ops.render.render(write_still=False)
        sc.display.render_aa = aa_
        np.save(fr["depth"].replace(".npy", f"_{lname}.npy"), _exr_depth(exr))
        for o in hid:
            o.hide_render = False
        if os.path.exists(os.path.join(tmp, "layer_clay.png")):
            os.remove(os.path.join(tmp, "layer_clay.png"))
    os.rmdir(tmp)
    if fr.get("blend"):
        bpy.ops.wm.save_as_mainfile(filepath=fr["blend"], compress=True)
    print(f"BLOCKOUT {fr['name']} {time.time() - t0:.2f}s build {t_build:.2f}s render {t_main:.2f}s "
          f"depth-read {t_read:.2f}s layers {time.time() - t1:.2f}s", flush=True)
