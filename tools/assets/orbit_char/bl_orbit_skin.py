"""Blender 4.5, GPU (Cycles HIP): skin an image-to-3D mesh from orbit frames.

    blender -b -t 4 --factory-startup --python bl_orbit_skin.py -- job.json

job: {"mesh": "shape.glb", "out": dir, "name": "orbit_fig", "height_m": 1.85,
      "views": [{"image": "f.png", "az": 0.0}, ...],      az in degrees: the camera's angle around the figure,
                                                          0 = in front (at -Y), +90 = at the figure's right (-X)
      "tex_px": 2048, "power": 4.0, "card": "05", "lock_wait_s": 1800}

1. The mesh (glTF, Y-up as the image-to-3D model writes it) is brought to the asset contract: metres, front -Y,
   feet on Z = 0, centred in X/Y, scaled to height_m; one object, Smart-UV unwrapped.
2. Each frame is mapped onto the mesh by an orthographic camera at its azimuth: the frame's figure (keyed from its
   corner colour) and the mesh's projected bounds are matched in height and centre.
3. Visibility per view: a sun from that view lights a white copy of the mesh and the direct diffuse light is baked
   (shadowed texels get 0), so a frame never paints a surface it could not see.
4. BaseColor = sum(vis_i^power * frame_i) / sum(vis_i^power), baked as emission; plain dielectric ORM.
Everything bakes on one XTX under its lock (flock -n polled); nothing renders on the CPU.
Outputs: <out>/<name>.blend, <name>.glb, T_<name>_BaseColor.png, T_<name>_ORM.png, skin_report.json.
"""

import fcntl
import json
import math
import os
import subprocess
import sys
import time

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
JOB = json.load(open(argv[0]))
OUT = JOB["out"]
NAME = JOB.get("name", "orbit_fig")
os.makedirs(OUT, exist_ok=True)
REP = {"times_s": {}}
T0 = time.time()


def tick(k, t):
    REP["times_s"][k] = round(time.time() - t, 2)


def gpu_lock():
    card = JOB.get("card", "05")
    if os.path.exists(os.path.expanduser("~/ue/GPU_HOLD")):
        raise SystemExit("GPU_HOLD present: no new card job")
    fd = os.open(f"/tmp/gpu-{card}.lock", os.O_RDWR | os.O_CREAT, 0o666)
    t = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t > JOB.get("lock_wait_s", 1800):
                raise SystemExit("card lock not free in time")
            time.sleep(2)
    REP["lock_wait_s"] = round(time.time() - t, 1)
    idx = subprocess.run([os.path.expanduser("~/bin/hip-index"), card], capture_output=True, text=True).stdout.strip()
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "HIP"
    prefs.get_devices()
    hips = [d for d in prefs.devices if d.type == "HIP"]
    if not hips:
        raise SystemExit("no HIP device: refusing to bake on the CPU")
    for d in prefs.devices:
        d.use = False
    # HIP_VISIBLE_DEVICES is set by the caller to this card only; take the first HIP device
    hips[0].use = True
    REP["device"] = [hips[0].name, card, idx]
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "GPU"
    return fd


def load_mesh():
    t = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=JOB["mesh"])
    obs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]
    if len(obs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    for o in list(bpy.context.scene.objects):
        if o is not ob:
            mw = ob.matrix_world.copy()
            if ob.parent is o:
                ob.parent = None
                ob.matrix_world = mw
            bpy.data.objects.remove(o)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    me = ob.data
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    lo, hi = co.min(0), co.max(0)
    s = JOB.get("height_m", 1.85) / (hi[2] - lo[2])
    rot = JOB.get("rot_z_deg", 0.0)
    c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    x, y = co[:, 0] - (lo[0] + hi[0]) / 2, co[:, 1] - (lo[1] + hi[1]) / 2
    co = np.stack([c * x - sn * y, sn * x + c * y, co[:, 2] - lo[2]], 1) * s
    co[:, 0] -= (co[:, 0].min() + co[:, 0].max()) / 2
    co[:, 1] -= (co[:, 1].min() + co[:, 1].max()) / 2
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    ob.name = me.name = NAME
    for p in me.polygons:
        p.use_smooth = True
    REP["dims_m"] = [round(float(v), 3) for v in co.max(0) - co.min(0)]
    REP["verts"], REP["faces"] = len(me.vertices), len(me.polygons)
    tick("load", t)
    t = time.time()
    if not JOB.get("keep_uvs"):
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.003)
        bpy.ops.object.mode_set(mode="OBJECT")
    tick("uv_unwrap_cpu", t)
    return ob, co


def key_bbox(img):
    """(x0, y0, x1, y1) of the figure in pixels (top-left origin): pixels far from the corner colour."""
    w, h = img.size
    a = np.asarray(img.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, :3]   # top row first
    bg = np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3)]).mean(0)
    d = np.abs(a - bg).sum(-1)
    m = d > JOB.get("key_thresh", 0.12)
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    # ignore thin noise: rows/cols with few pixels
    rc, cc = m.sum(1), m.sum(0)
    rows = np.where(rc > max(2, 0.002 * w))[0]
    cols = np.where(cc > max(2, 0.002 * h))[0]
    return [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]


def view_basis(az):
    t = math.radians(az)
    cam = np.array([-math.sin(t), -math.cos(t), 0.0])        # unit vector from the figure to the camera
    right = np.array([math.cos(t), -math.sin(t), 0.0])        # camera's screen right
    return cam, right


def load_mask(path, f=4):
    m = bpy.data.images.load(path, check_existing=True)
    w, h = m.size
    a = np.asarray(m.pixels[:], dtype=np.float32).reshape(h, w, 4)[::-1, :, 3]        # top row first
    bpy.data.images.remove(m)
    return a[::f, ::f] > 0.5


def splat(co, az, bb, f, shape):
    """Silhouette of the mesh's vertices at azimuth az, placed as mappings() would place it, on the f-times
    downsampled frame grid."""
    _, right = view_basis(az)
    u, z = co @ right, co[:, 2]
    s = (bb[3] - bb[1]) / (z.max() - z.min())
    px = ((u - (u.min() + u.max()) / 2) * s + (bb[0] + bb[2]) / 2) / f
    py = (bb[1] + (z.max() - z) * s) / f
    g = np.zeros(shape, dtype=bool)
    ix = np.clip(px.astype(int), 0, shape[1] - 1)
    iy = np.clip(py.astype(int), 0, shape[0] - 1)
    g[iy, ix] = True
    g2 = g.copy()
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        g2 |= np.roll(np.roll(g, dy, 0), dx, 1)
    return g2


def refine_az(co, v):
    """The azimuth (within +-span of the nominal one) whose mesh silhouette best overlaps the keyed frame's."""
    f = 4
    mask = load_mask(v["mask"], f)
    span = JOB.get("az_span", 25)
    best = (-1.0, v["az"])
    for d in range(-span, span + 1):
        az = v["az"] + d
        g = splat(co, az, v["bbox"], f, mask.shape)
        iou = (g & mask).sum() / max(1, (g | mask).sum())
        if iou > best[0]:
            best = (float(iou), az)
    return best


def mappings(co):
    out = []
    t = time.time()
    for v in JOB["views"]:
        im = bpy.data.images.load(v["image"], check_existing=True)
        im.colorspace_settings.name = "sRGB"
        w, h = im.size
        bb = v.get("bbox") or key_bbox(im)
        if v.get("mask") and not v.get("fixed"):
            iou, az = refine_az(co, {**v, "bbox": bb})
            v["az_nominal"], v["az"], v["iou"] = v["az"], az, round(iou, 3)
        cam, right = view_basis(v["az"])
        u = co @ right
        z = co[:, 2]
        fh = bb[3] - bb[1]
        s = fh / (z.max() - z.min())                          # pixels per metre
        if v.get("feet_cut"):                                  # frame cut at the bottom: match the top and width
            s = (bb[2] - bb[0]) / (u.max() - u.min())
        cx_img = (bb[0] + bb[2]) / 2
        cx_m = (u.min() + u.max()) / 2
        # pixel x = (u - cx_m) * s + cx_img ; pixel y (top-left) = bb[1] + (z.max() - z) * s
        out.append({"image": im, "w": w, "h": h, "cam": cam, "right": right, "s": s, "cx_img": cx_img,
                    "cx_m": cx_m, "top_px": bb[1], "zmax": float(z.max()), "bbox": bb, "az": v["az"]})
    REP["views"] = [{"image": m["image"].filepath, "az": m["az"], "bbox": m["bbox"],
                     "px_per_m": round(m["s"], 1), "az_nominal": v.get("az_nominal"), "iou": v.get("iou")}
                    for m, v in zip(out, JOB["views"])]
    REP["times_s"]["az_refine_cpu"] = round(time.time() - t, 2)
    return out


def node(nt, typ, **kw):
    n = nt.nodes.new(typ)
    for k, v in kw.items():
        setattr(n, k, v)
    return n


def mathn(nt, op, a, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            nt.links.new(x, n.inputs[i])
    return n.outputs[0]


def proj_uv(nt, pos, m):
    """Shader: UV of this surface point in view m's frame (Blender image space: origin bottom-left)."""
    dot = node(nt, "ShaderNodeVectorMath", operation="DOT_PRODUCT")
    nt.links.new(pos, dot.inputs[0])
    dot.inputs[1].default_value = tuple(m["right"])
    sep = node(nt, "ShaderNodeSeparateXYZ")
    nt.links.new(pos, sep.inputs[0])
    px = mathn(nt, "ADD", mathn(nt, "MULTIPLY", mathn(nt, "SUBTRACT", dot.outputs["Value"], m["cx_m"]), m["s"]),
               m["cx_img"])
    py = mathn(nt, "ADD", mathn(nt, "MULTIPLY", mathn(nt, "SUBTRACT", m["zmax"], sep.outputs["Z"]), m["s"]),
               m["top_px"])
    u = mathn(nt, "DIVIDE", px, m["w"])
    v = mathn(nt, "SUBTRACT", 1.0, mathn(nt, "DIVIDE", py, m["h"]))
    comb = node(nt, "ShaderNodeCombineXYZ")
    nt.links.new(u, comb.inputs[0])
    nt.links.new(v, comb.inputs[1])
    return comb.outputs[0]


def bake_target(ob, name, px, color):
    im = bpy.data.images.new(name, px, px, alpha=False, float_buffer=not color)
    im.colorspace_settings.name = "sRGB" if color else "Non-Color"
    mat = ob.data.materials[0]
    nt = mat.node_tree
    tn = nt.nodes.get("BAKE_TARGET") or nt.nodes.new("ShaderNodeTexImage")
    tn.name = "BAKE_TARGET"
    tn.image = im
    for n in nt.nodes:
        n.select = False
    tn.select = True
    nt.nodes.active = tn
    return im


def visibility(ob, views, px):
    """Per view: a sun from the camera's side, white diffuse material, direct diffuse bake (shadowed = 0)."""
    sc = bpy.context.scene
    mat = bpy.data.materials.new("M_vis")
    mat.use_nodes = True
    nt = mat.node_tree
    pr = nt.nodes["Principled BSDF"]
    pr.inputs["Base Color"].default_value = (1, 1, 1, 1)
    pr.inputs["Roughness"].default_value = 1.0
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    sc.world = bpy.data.worlds.new("black")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    sc.render.bake.use_pass_direct = True
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.use_pass_color = False
    sc.render.bake.margin = 4
    sc.cycles.samples = JOB.get("vis_samples", 4)
    sun_d = bpy.data.lights.new("sun", "SUN")
    sun_d.energy = 1.0
    sun_d.angle = math.radians(JOB.get("sun_angle_deg", 4))
    sun = bpy.data.objects.new("sun", sun_d)
    sc.collection.objects.link(sun)
    ims = []
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob
    for i, m in enumerate(views):
        c = m["cam"]
        # the sun shines along -cam (from the camera towards the figure); Blender suns point down their -Z
        sun.rotation_euler = (0, 0, 0)
        z = -(-c)
        rot = np.array(_look_rot(z))
        sun.matrix_world = _mat(rot)
        im = bake_target(ob, f"vis_{i}", px, False)
        t = time.time()
        bpy.ops.object.bake(type="DIFFUSE", use_clear=True)
        REP["times_s"][f"bake_vis_{i}"] = round(time.time() - t, 2)
        ims.append(im)
    bpy.data.objects.remove(sun)
    return ims


def _look_rot(z):
    """3x3 rotation whose +Z column is z (unit)."""
    z = z / np.linalg.norm(z)
    up = np.array([0, 0, 1.0]) if abs(z[2]) < 0.99 else np.array([0, 1.0, 0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def _mat(r):
    import mathutils
    m = mathutils.Matrix.Identity(4)
    for i in range(3):
        for j in range(3):
            m[i][j] = float(r[i, j])
    return m


def colour(ob, views, vis, px):
    sc = bpy.context.scene
    mat = bpy.data.materials.new(f"M_{NAME}_proj")
    mat.use_nodes = True
    nt = mat.node_tree
    out = nt.nodes["Material Output"]
    nt.nodes.remove(nt.nodes["Principled BSDF"])
    geo = node(nt, "ShaderNodeNewGeometry")
    uvn = node(nt, "ShaderNodeUVMap")
    pos = geo.outputs["Position"]
    p = JOB.get("power", 4.0)
    acc_c = None
    acc_w = None
    for m, vi in zip(views, vis):
        tv = node(nt, "ShaderNodeTexImage", image=vi, interpolation="Linear")
        nt.links.new(uvn.outputs["UV"], tv.inputs["Vector"])
        sepv = node(nt, "ShaderNodeSeparateColor")
        nt.links.new(tv.outputs["Color"], sepv.inputs["Color"])
        w = mathn(nt, "POWER", mathn(nt, "MAXIMUM", sepv.outputs["Red"], 0.0), p)
        ti = node(nt, "ShaderNodeTexImage", image=m["image"], interpolation="Cubic", extension="CLIP")
        nt.links.new(proj_uv(nt, pos, m), ti.inputs["Vector"])
        sc_ = node(nt, "ShaderNodeVectorMath", operation="SCALE")
        nt.links.new(ti.outputs["Color"], sc_.inputs[0])
        nt.links.new(w, sc_.inputs["Scale"])
        if acc_c is None:
            acc_c, acc_w = sc_.outputs[0], w
        else:
            a = node(nt, "ShaderNodeVectorMath", operation="ADD")
            nt.links.new(acc_c, a.inputs[0])
            nt.links.new(sc_.outputs[0], a.inputs[1])
            acc_c = a.outputs[0]
            acc_w = mathn(nt, "ADD", acc_w, w)
    den = mathn(nt, "MAXIMUM", acc_w, 1e-6)
    inv = mathn(nt, "DIVIDE", 1.0, den)
    fin = node(nt, "ShaderNodeVectorMath", operation="SCALE")
    nt.links.new(acc_c, fin.inputs[0])
    nt.links.new(inv, fin.inputs["Scale"])
    em = node(nt, "ShaderNodeEmission")
    nt.links.new(fin.outputs[0], em.inputs["Color"])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    im = bake_target(ob, f"T_{NAME}_BaseColor", px, True)
    sc.cycles.samples = JOB.get("color_samples", 4)
    sc.render.bake.margin = 8
    t = time.time()
    bpy.ops.object.bake(type="EMIT", use_clear=True)
    tick("bake_color", t)
    return im


def final_material(ob, bc):
    mat = bpy.data.materials.new(f"M_{NAME}")
    mat.use_nodes = True
    nt = mat.node_tree
    pr = nt.nodes["Principled BSDF"]
    tn = node(nt, "ShaderNodeTexImage", image=bc)
    nt.links.new(tn.outputs["Color"], pr.inputs["Base Color"])
    pr.inputs["Roughness"].default_value = JOB.get("roughness", 0.75)
    pr.inputs["Metallic"].default_value = 0.0
    ob.data.materials.clear()
    ob.data.materials.append(mat)
    return mat


def check_renders(ob, co):
    """Orthographic check renders of the skinned mesh (Cycles on the locked card): az 0, -45, -90, 180, 90, 45."""
    sc = bpy.context.scene
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.35, 0.35, 1)
    sc.render.resolution_x = sc.render.resolution_y = JOB.get("check_px", 512)
    sc.cycles.samples = 32
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "Standard"
    cam_d = bpy.data.cameras.new("cam")
    cam_d.type = "ORTHO"
    lo, hi = co.min(0), co.max(0)
    cam_d.ortho_scale = float(max(hi - lo)) * 1.1
    cam = bpy.data.objects.new("cam", cam_d)
    sc.collection.objects.link(cam)
    sc.camera = cam
    ctr = (lo + hi) / 2
    paths = []
    for az in JOB.get("check_az", [0, -45, -90, 180, 90, 45]):
        c, _ = view_basis(az)
        p = ctr + c * 5.0
        cam.matrix_world = _mat(_look_rot(c))
        cam.location = tuple(p)
        out = os.path.join(OUT, f"check_az{int(az):+04d}.png")
        sc.render.filepath = out
        t = time.time()
        bpy.ops.render.render(write_still=True)
        REP["times_s"][f"check_az{int(az)}"] = round(time.time() - t, 2)
        paths.append(out)
    REP["check_renders"] = paths


def co_now(ob):
    a = np.zeros(len(ob.data.vertices) * 3)
    ob.data.vertices.foreach_get("co", a)
    return a.reshape(-1, 3)


def main():
    ob, co = load_mesh()
    views = mappings(co)
    fd = gpu_lock()
    px = JOB.get("tex_px", 2048)
    try:
        vis = visibility(ob, views, JOB.get("vis_px", 1024))
        bc = colour(ob, views, vis, px)
        final_material(ob, bc)
        if JOB.get("check", True):
            check_renders(ob, co_now(ob))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    bc.filepath_raw = os.path.join(OUT, f"T_{NAME}_BaseColor.png")
    bc.file_format = "PNG"
    bc.save()
    orm = bpy.data.images.new(f"T_{NAME}_ORM", 4, 4, alpha=False)
    orm.pixels[:] = [1.0, JOB.get("roughness", 0.75), 0.0, 1.0] * 16
    orm.filepath_raw = os.path.join(OUT, f"T_{NAME}_ORM.png")
    orm.file_format = "PNG"
    orm.save()
    for i, v in enumerate(bpy.data.images):
        if v.name.startswith("vis_") and JOB.get("save_vis"):
            v.filepath_raw = os.path.join(OUT, f"{v.name}.png")
            v.file_format = "PNG"
            v.save()
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, f"{NAME}.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(OUT, f"{NAME}.glb"), use_selection=True, export_format="GLB")
    REP["total_s"] = round(time.time() - T0, 1)
    json.dump(REP, open(os.path.join(OUT, "skin_report.json"), "w"), indent=1)
    print("SKIN_REPORT " + json.dumps(REP))


main()
