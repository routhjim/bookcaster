"""Blender side of the texture-painting stage (tools/assets/paint.py): guide views out, painted views back in.

    blender -b --factory-startup -t 16 --python tools/assets/bl_paint.py -- JOB.json

JOB {"mode": "guides" | "project" | "stills", "blend": the asset's baked source .blend, "id": asset id,
     "device": "CPU" | "HIP", "hip_bus": "05", "hiprt": 0|1, ... mode keys below}

guides  {"out": folder, "views": {"ring": 8, "ring_el": 12, "upper": 4, "upper_el": 50, "top": true,
         "area_px": 1572864, "close": {"box": [x0, y0, z0, x1, y1, z1] (fractions of the bounding box),
         "ring": 4, "el": 40, "fill": 1.0}}, "samples": 48}
        One render per view of the asset with a DIFFUSE guide material (its baked BaseColor, roughness 1, no
        specular, no normal map) under even light (a white world + a soft key from the camera's upper left), so
        the render is albedo x a smooth shading term. Per view: <view>_guide.png (the picture the image model
        repaints: display sRGB at the view's exposure on a flat grey), <view>_mask.png, <view>_depth.png (near =
        bright, the ControlNet depth convention), <view>_Image.exr (linear, denoised), <view>_DiffCol.exr,
        <view>_Position.exr; views.json (camera, size, exposure, files).
project {"views_json": path, "painted": {view: png}, "res": 2048, "out_dir": textures folder, "glb": path,
         "params": {...}} (see project() for the parameters)
        For every texel (its world position and normal baked from the UVs): each view that sees it (its rendered
        position matches) gives the painted colour divided by the view's shading (painted / (exposure x
        Image / DiffCol)), i.e. the painted albedo with the guide's lighting divided out. Views are weighted by how
        squarely the texel faces the camera, the view's pixels per metre there, its distance from silhouettes and
        depth steps, and the guide's light level; low and high frequencies blend separately (low: soft weights,
        all views; high: sharp weights, the best view wins), per-view gains are matched first. Unseen texels fall
        back to the procedural colour x the global gain; the gutter is dilated. Optional detail maps from the
        painted colour: height = luminance high-pass -> tangent normal added to the procedural normal; roughness
        moved by the same high-pass. Writes T_<id>_BaseColor/Normal/ORM.png into out_dir (never the procedural
        ones) and exports the asset with them to glb.
stills  {"maps": "procedural" | "painted" (folder), "out": folder, "shots": [{"name", "az", "el", "fill",
         "target": [fx, fy, fz] fractions, "px": [w, h]}], "samples": 64}
        Cycles stills of the asset with its real PBR maps under the harness's key/rim/world light, for before/after.
"""

import json
import math
import os
import sys
import time
import traceback

import bpy
import numpy as np
from mathutils import Matrix, Vector

JOB = json.load(open(sys.argv[sys.argv.index("--") + 1]))
AID = JOB["id"]
REPORT = {"mode": JOB["mode"], "times_s": {}, "warnings": []}


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()

    def __exit__(self, *a):
        REPORT["times_s"][self.name] = round(time.time() - self.t, 2)


# ---------------------------------------------------------------- helpers
def set_device(sc):
    sc.render.engine = "CYCLES"
    if JOB.get("device") == "HIP":
        p = bpy.context.preferences.addons["cycles"].preferences
        p.compute_device_type = "HIP"
        p.get_devices()
        want = f"0000:{JOB.get('hip_bus', '')}:00"
        hit = False
        for d in p.devices:
            d.use = d.type == "HIP" and d.id.endswith(want)
            hit = hit or d.use
        if hit:
            p.use_hiprt = bool(JOB.get("hiprt", 1))
            sc.cycles.device = "GPU"
            REPORT["device"] = f"HIP {want}"
            return
        REPORT["warnings"].append(f"no HIP device {want}; CPU")
    sc.cycles.device = "CPU"
    REPORT["device"] = "CPU"


def open_asset():
    bpy.ops.wm.open_mainfile(filepath=JOB["blend"])
    ob = bpy.data.objects[f"SM_{AID}"]
    for o in list(bpy.data.objects):
        if o is not ob:
            bpy.data.objects.remove(o, do_unlink=True)
    sc = bpy.context.scene
    lo = np.array([min(v.co[i] for v in ob.data.vertices) for i in range(3)]) if len(ob.data.vertices) < 2000 else None
    vs = np.empty(len(ob.data.vertices) * 3, dtype=np.float32)
    ob.data.vertices.foreach_get("co", vs)
    vs = vs.reshape(-1, 3)
    lo, hi = vs.min(0), vs.max(0)
    return sc, ob, vs, lo, hi


def mat_images(ob):
    """The baked material's image nodes: {"BaseColor": node, "Normal": node, "ORM": node}."""
    m = ob.data.materials[0]
    out = {}
    for n in m.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            for k in ("BaseColor", "Normal", "ORM"):
                if n.image.name.startswith(f"T_{AID}_{k}"):
                    out[k] = n
    return m, out


def px(im):
    w, h = im.size
    a = np.empty(w * h * im.channels, dtype=np.float32)
    im.pixels.foreach_get(a)
    return a.reshape(h, w, im.channels)          # rows bottom-up (Blender's order)


def load(path, raw=True):
    im = bpy.data.images.load(path, check_existing=False)
    if raw:
        im.colorspace_settings.name = "Non-Color"
    a = px(im)
    bpy.data.images.remove(im)
    return a


def save_png(a, path, mode="RGB"):
    """a: (h, w[, c]) 0-1 floats, rows bottom-up, already display-encoded (written as is)."""
    a = np.clip(a, 0, 1).astype(np.float32)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    h, w = a.shape[:2]
    rgba = np.ones((h, w, 4), np.float32)
    rgba[..., :a.shape[2]] = a[..., :4] if a.shape[2] <= 4 else a[..., :4]
    im = bpy.data.images.new(os.path.basename(path), w, h, alpha=(mode == "RGBA"))
    im.colorspace_settings.name = "Non-Color"
    im.pixels.foreach_set(rgba.ravel())
    im.filepath_raw = path
    im.file_format = "PNG"
    im.save()
    bpy.data.images.remove(im)


def srgb_enc(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def srgb_dec(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def box(a, r):
    """Box blur of radius r (pixels) over the first two axes, edges clamped; separable, via cumulative sums."""
    if r < 1:
        return a.copy()
    r = int(r)
    out = a.astype(np.float32)
    for ax in (0, 1):
        n = out.shape[ax]
        pad = [(0, 0)] * out.ndim
        pad[ax] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=ax, dtype=np.float64)
        hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax)
        lo = np.take(c, np.arange(0, n), axis=ax)
        out = ((hi - lo) / (2 * r + 1)).astype(np.float32)
    return out


def blur(a, r, passes=2):
    for _ in range(passes):
        a = box(a, r)
    return a


def masked_blur(a, m, r):
    """Blur of a over the mask m only (normalised convolution)."""
    mm = m[..., None] if a.ndim == 3 else m
    num = blur(a * mm, r)
    den = blur(mm.astype(np.float32), r)
    return num / np.maximum(den, 1e-6)


def dilate_fill(a, valid, iters):
    """Fill invalid texels from valid neighbours, one ring per iteration (mean of the valid 8-neighbours)."""
    a = a.copy()
    v = valid.copy()
    for _ in range(iters):
        if v.all():
            break
        acc = np.zeros_like(a)
        cnt = np.zeros(v.shape, np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                sv = np.roll(np.roll(v, dy, 0), dx, 1)
                sa = np.roll(np.roll(a, dy, 0), dx, 1)
                acc += sa * sv[..., None] if a.ndim == 3 else sa * sv
                cnt += sv
        new = (~v) & (cnt > 0)
        if not new.any():
            break
        a[new] = (acc[new] / (cnt[new][..., None] if a.ndim == 3 else cnt[new]))
        v = v | new
    return a, v


def bilinear(img, x, y):
    """Sample img (h, w, c) at float pixel coords (x right, y up from the bottom row; pixel centres at +0.5)."""
    h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0, w - 1.001)
    y = np.clip(y - 0.5, 0, h - 1.001)
    x0 = x.astype(np.int32)
    y0 = y.astype(np.int32)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    if img.ndim == 2:
        fx, fy = fx[:, 0], fy[:, 0]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


# ---------------------------------------------------------------- views
def look_at(cam, loc, target):
    cam.location = loc
    cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    bpy.context.view_layer.update()


def fit_view(cam, pts, direction, centre, area_px, fill=0.94, clip_pts=None):
    """Camera along `direction` (unit, from the centre outward) framing pts tightly: aim at the angular centre of
    the points, then focal length and an aspect (0.5-2) that fit them; size ~area_px, sides multiples of 32.
    clip_pts: a close view frames these (a region) and lets the rest of the asset run off the picture."""
    fit = clip_pts if clip_pts is not None else pts
    rad = float(np.linalg.norm(fit - fit.mean(0), axis=1).max())
    dist = rad * (3.2 if clip_pts is None else 2.2)
    target = Vector(centre)
    for _ in range(3):
        look_at(cam, target + Vector(direction) * dist, target)
        M = np.array(cam.matrix_world.inverted())
        pc = fit @ M[:3, :3].T + M[:3, 3]
        tx, ty = pc[:, 0] / -pc[:, 2], pc[:, 1] / -pc[:, 2]
        cx, cy = (tx.min() + tx.max()) / 2, (ty.min() + ty.max()) / 2
        # re-aim at the angular centre: rotate the forward axis toward (cx, cy, -1)
        R = np.array(cam.matrix_world)[:3, :3]
        fwd = R @ np.array([cx, cy, -1.0])
        target = Vector(np.array(cam.location) + fwd / np.linalg.norm(fwd) * dist)
        cam.rotation_euler = Vector(fwd).to_track_quat("-Z", "Y").to_euler()
        bpy.context.view_layer.update()
        if abs(cx) < 1e-3 and abs(cy) < 1e-3:
            break
    M = np.array(cam.matrix_world.inverted())
    pc = fit @ M[:3, :3].T + M[:3, 3]
    tx, ty = pc[:, 0] / -pc[:, 2], pc[:, 1] / -pc[:, 2]
    ex, ey = max(abs(tx).max(), 1e-4), max(abs(ty).max(), 1e-4)
    asp = float(np.clip(ex / ey, 0.5, 2.0))
    W = int(round(math.sqrt(area_px * asp) / 32)) * 32
    H = int(round(math.sqrt(area_px / asp) / 32)) * 32
    half_w = max(ex, ey * W / H) / fill          # tan of half the horizontal field
    cd = cam.data
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36.0
    cd.lens = 18.0 / half_w
    cd.clip_start = max(0.001, dist * 0.02)
    cd.clip_end = dist * 10
    return W, H


def view_list(lo, hi, vs, cfg):
    """[(name, kind, direction, centre, clip_pts or None)]."""
    c = (lo + hi) / 2
    out = []

    def d(az, el):
        a, e = math.radians(az), math.radians(el)
        return (math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))

    n = int(cfg.get("ring", 8))
    for i in range(n):
        az = 360.0 * i / n
        out.append((f"r{int(az):03d}", "ring", d(az, cfg.get("ring_el", 12)), c, None))
    m = int(cfg.get("upper", 4))
    for i in range(m):
        az = 360.0 * i / m + 180.0 / max(m, 1)
        out.append((f"u{int(az):03d}", "upper", d(az, cfg.get("upper_el", 50)), c, None))
    if cfg.get("top", True):
        out.append(("top", "top", (0.0, -0.08, 1.0), c, None))
    if cfg.get("low", False):
        for i in range(4):
            out.append((f"l{90 * i:03d}", "low", d(90 * i + 45, -25), c, None))
    cl = cfg.get("close")
    if cl:
        f = np.array(cl["box"], dtype=np.float32)
        b0 = lo + (hi - lo) * f[:3]
        b1 = lo + (hi - lo) * f[3:]
        inside = vs[np.all((vs >= b0) & (vs <= b1), axis=1)]
        corners = np.array([[x, y, z] for x in (b0[0], b1[0]) for y in (b0[1], b1[1]) for z in (b0[2], b1[2])])
        clip = inside if len(inside) > 8 else corners
        cc = (b0 + b1) / 2
        k = int(cl.get("ring", 4))
        for i in range(k):
            az = 360.0 * i / k + float(cl.get("az0", 0))
            out.append((f"c{int(az):03d}", "close", d(az, cl.get("el", 40)), cc, clip))
        if cl.get("top", True):
            out.append(("ctop", "close", (0.0, -0.08, 1.0), cc, clip))
    return out


def guides():
    sc, ob, vs, lo, hi = open_asset()
    cfg = JOB.get("views", {})
    out = JOB["out"]
    os.makedirs(out, exist_ok=True)
    set_device(sc)
    m, imgs = mat_images(ob)
    base = imgs["BaseColor"].image
    # albedo level -> exposure: the guide should read mid-grey whatever the asset's albedo
    a = px(base)[..., :3]
    lum = (a * [0.2126, 0.7152, 0.0722]).sum(-1)
    lumc = lum[lum > 1e-4]
    mean_alb = float(np.median(srgb_dec(lumc))) if base.colorspace_settings.name == "sRGB" else float(np.median(lumc))
    # bpy pixels of an sRGB 8-bit image are already linear floats in 4.x? Check: Blender returns the stored
    # values (display-encoded) for byte images; decode was applied above when the colour space is sRGB.
    target = float(JOB.get("guide_level", 0.22))           # linear mid level of the guide render
    expo0 = float(np.clip(math.log2(target / max(mean_alb * 1.0, 1e-3)), -2, 5))
    REPORT["albedo_median"] = round(mean_alb, 4)
    REPORT["exposure"] = round(expo0, 3)
    # guide material: diffuse only
    g = bpy.data.materials.new("paint_guide")
    g.use_nodes = True
    nt = g.node_tree
    pr = nt.nodes["Principled BSDF"]
    tb = nt.nodes.new("ShaderNodeTexImage")
    tb.image = base
    tc = nt.nodes.new("ShaderNodeTexCoord")
    nt.links.new(tc.outputs["UV"], tb.inputs["Vector"])
    nt.links.new(tb.outputs["Color"], pr.inputs["Base Color"])
    pr.inputs["Roughness"].default_value = 1.0
    pr.inputs["Metallic"].default_value = 0.0
    pr.inputs["Specular IOR Level"].default_value = 0.0
    ob.data.materials.clear()
    ob.data.materials.append(g)
    w = bpy.data.worlds.new("paint_world")
    sc.world = w
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)
    bg.inputs["Strength"].default_value = float(JOB.get("world_strength", 0.75))
    ld = bpy.data.lights.new("paint_key", "SUN")
    ld.energy = float(JOB.get("key_strength", 1.2))
    ld.angle = math.radians(25)
    key = bpy.data.objects.new("paint_key", ld)
    sc.collection.objects.link(key)
    cd = bpy.data.cameras.new("paint_cam")
    cam = bpy.data.objects.new("paint_cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    vl = sc.view_layers[0]
    vl.use_pass_diffuse_color = True
    vl.use_pass_position = True
    sc.render.film_transparent = True
    sc.cycles.samples = int(JOB.get("samples", 48))
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 4
    sc.render.resolution_percentage = 100
    sc.view_settings.view_transform = "Standard"
    sc.view_settings.look = "None"
    sc.view_settings.exposure = 0.0
    sc.use_nodes = True
    tree = sc.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    comp = tree.nodes.new("CompositorNodeComposite")
    tree.links.new(rl.outputs["Image"], comp.inputs["Image"])
    fo = tree.nodes.new("CompositorNodeOutputFile")
    fo.format.file_format = "OPEN_EXR"
    fo.format.color_depth = "32"
    fo.format.exr_codec = "ZIP"
    fo.file_slots.clear()
    for s in ("Image", "DiffCol", "Position"):
        fo.file_slots.new(s)
        tree.links.new(rl.outputs[s], fo.inputs[s])
    sc.frame_set(1)
    sc.render.filepath = os.path.join(out, "_composite.png")
    area = float(cfg.get("area_px", 1536 * 1024))
    views = []
    bg_disp = float(JOB.get("bg_grey", 0.5))
    for name, kind, direction, centre, clip in view_list(lo, hi, vs, cfg):
        t = time.time()
        W, H = fit_view(cam, vs[:: max(1, len(vs) // 20000)], direction, centre,
                        area * (float(cfg.get("close", {}).get("area_scale", 1.0)) if kind == "close" else 1.0),
                        fill=float(cfg.get("fill", 0.94)), clip_pts=clip)
        sc.render.resolution_x, sc.render.resolution_y = W, H
        # key light from the camera's upper left, 35 degrees off its axis
        R = np.array(cam.matrix_world)[:3, :3]
        kd = R @ np.array([-0.5, 0.6, 1.0])
        key.rotation_euler = Vector(kd / np.linalg.norm(kd)).to_track_quat("Z", "Y").to_euler()
        tmpd = os.path.join(out, f"_tmp_{name}")
        fo.base_path = tmpd + os.sep
        bpy.ops.render.render(write_still=False)
        files = {}
        for s in ("Image", "DiffCol", "Position"):
            src = os.path.join(tmpd, f"{s}0001.exr")
            dst = os.path.join(out, f"{name}_{s}.exr")
            os.replace(src, dst)
            files[s] = dst
        os.rmdir(tmpd)
        img = load(files["Image"])                       # linear, premultiplied, alpha in [3]
        al = img[..., 3]
        # per-view exposure: the brightest 0.5 % of the object just below white (a polished face must not clip:
        # clipped pixels carry no ratio), but never more than the albedo-based level
        lumv = (img[..., :3] * [0.2126, 0.7152, 0.0722]).sum(-1)[al > 0.99]
        if lumv.size:
            expo = float(np.clip(min(math.log2(float(JOB.get("white", 0.85)) / max(np.percentile(lumv, 99.5), 1e-4)),
                                     expo0 + 1.5), -3, 6))
        else:
            expo = expo0
        lin = img[..., :3] * 2 ** expo + srgb_dec(np.full(3, bg_disp)) * (1 - al[..., None])
        save_png(srgb_enc(lin), os.path.join(out, f"{name}_guide.png"))
        save_png((al > 0.5).astype(np.float32), os.path.join(out, f"{name}_mask.png"))
        pos = load(files["Position"])[..., :3]
        camp = np.array(cam.location)
        dist = np.linalg.norm(pos - camp, axis=-1)
        msk = al > 0.5
        if msk.any():
            inv = 1.0 / np.maximum(dist, 1e-6)
            a0, a1 = inv[msk].min(), inv[msk].max()
            dep = np.where(msk, 0.15 + 0.85 * (inv - a0) / max(a1 - a0, 1e-9), 0.0)
        else:
            dep = np.zeros_like(dist)
        save_png(dep, os.path.join(out, f"{name}_depth.png"))
        Mi = np.array(cam.matrix_world)
        views.append({"name": name, "kind": kind, "size": [W, H], "matrix_world": Mi.tolist(),
                      "lens": cd.lens, "sensor_width": cd.sensor_width, "exposure": expo, "bg_grey": bg_disp,
                      "coverage": round(float(msk.mean()), 3), "seconds": round(time.time() - t, 2),
                      "files": dict(files, guide=os.path.join(out, f"{name}_guide.png"),
                                    mask=os.path.join(out, f"{name}_mask.png"),
                                    depth=os.path.join(out, f"{name}_depth.png"))})
    if os.path.exists(os.path.join(out, "_composite.png")):
        os.remove(os.path.join(out, "_composite.png"))
    with open(os.path.join(out, "views.json"), "w") as f:
        json.dump({"id": AID, "exposure": expo0, "bbox": [lo.tolist(), hi.tolist()], "views": views}, f, indent=1)
    REPORT["views"] = len(views)
    REPORT["pixels"] = int(sum(v["size"][0] * v["size"][1] * v["coverage"] for v in views))


# ---------------------------------------------------------------- project
def bake_geometry(sc, ob, res, lo, hi):
    """World position (normalised to the bounding box) and normal of every texel, -1 where no face covers it."""
    m = bpy.data.materials.new("paint_geo")
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sub = nt.nodes.new("ShaderNodeVectorMath")
    sub.operation = "SUBTRACT"
    sub.inputs[1].default_value = tuple(lo)
    div = nt.nodes.new("ShaderNodeVectorMath")
    div.operation = "DIVIDE"
    div.inputs[1].default_value = tuple(np.maximum(hi - lo, 1e-6))
    nt.links.new(geo.outputs["Position"], sub.inputs[0])
    nt.links.new(sub.outputs[0], div.inputs[0])
    nt.links.new(em.outputs[0], out.inputs["Surface"])
    tn = nt.nodes.new("ShaderNodeTexImage")
    nt.nodes.active = tn
    saved = [s.material for s in ob.material_slots]
    ob.data.materials.clear()
    ob.data.materials.append(m)
    sc.render.bake.margin = 0
    sc.cycles.samples = 1
    res_ = {}
    for key, sock in (("pos", div.outputs[0]), ("nrm", None)):
        im = bpy.data.images.new(f"paint_{key}", res, res, alpha=True, float_buffer=True)
        im.colorspace_settings.name = "Non-Color"
        fill = np.full(res * res * 4, -1.0, dtype=np.float32)
        im.pixels.foreach_set(fill)
        tn.image = im
        if key == "pos":
            nt.links.new(sock, em.inputs["Color"])
        else:
            mul = nt.nodes.new("ShaderNodeVectorMath")
            mul.operation = "MULTIPLY_ADD"
            mul.inputs[1].default_value = (0.5, 0.5, 0.5)
            mul.inputs[2].default_value = (0.5, 0.5, 0.5)
            nt.links.new(geo.outputs["Normal"], mul.inputs[0])
            nt.links.new(mul.outputs[0], em.inputs["Color"])
        bpy.ops.object.bake(type="EMIT", use_clear=False)
        res_[key] = px(im)
        bpy.data.images.remove(im)
    ob.data.materials.clear()
    for s in saved:
        ob.data.materials.append(s)
    bpy.data.materials.remove(m)
    cov = res_["pos"][..., 0] > -0.5
    pos = lo + res_["pos"][..., :3] * (hi - lo)
    nrm = res_["nrm"][..., :3] * 2 - 1
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    return pos, nrm, cov


def view_proj(v):
    Mw = np.array(v["matrix_world"])
    Mi = np.linalg.inv(Mw)
    W, H = v["size"]
    f = v["lens"] / v["sensor_width"] * W
    return Mw, Mi, W, H, f


def project():
    t0 = time.time()
    sc, ob, vs, lo, hi = open_asset()
    P = dict(JOB.get("params", {}))
    res = int(JOB["res"])
    set_device(sc)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    m, imgs = mat_images(ob)
    with Timer("geometry_bake"):
        pos, nrm, cov = bake_geometry(sc, ob, res, lo, hi)
    old = px(imgs["BaseColor"].image)[..., :3]
    if imgs["BaseColor"].image.colorspace_settings.name == "sRGB":
        old_lin = srgb_dec(old)
    else:
        old_lin = old
    idx = np.nonzero(cov.ravel())[0]
    T = len(idx)
    tp = pos.reshape(-1, 3)[idx]
    tn = nrm.reshape(-1, 3)[idx]
    ta = old_lin.reshape(-1, 3)[idx]
    V = json.load(open(JOB["views_json"]))
    size = float(np.linalg.norm(hi - lo))
    tol = float(P.get("vis_tol", 0.006)) * size
    edge_r = float(P.get("edge_px", 6))
    low_r = float(P.get("low_px", 10))
    p_soft, p_sharp = float(P.get("p_soft", 2)), float(P.get("p_sharp", 8))
    stats = []
    per_view = []
    with Timer("views_prepare"):
        for v in V["views"]:
            pp = JOB["painted"].get(v["name"])
            if not pp or not os.path.exists(pp):
                continue
            Mw, Mi, W, H, f = view_proj(v)
            img = load(v["files"]["Image"])
            al = img[..., 3]
            dc = load(v["files"]["DiffCol"])[..., :3]
            posv = load(v["files"]["Position"])[..., :3]
            pain = load(pp)[..., :3]
            if pain.shape[:2] != (H, W):
                REPORT["warnings"].append(f"{v['name']}: painted {pain.shape[:2]} v guide {(H, W)}; skipped")
                continue
            E = 2 ** v["exposure"]
            # shading S = Image / DiffCol (diffuse-only guide: Image = DiffCol x S), denoised through Image
            S = img[..., :3] / np.maximum(al[..., None], 1e-4) / np.maximum(dc, 1e-4)
            S = np.where(dc > 1e-4, S, 0)
            Sl = S.mean(-1)
            inside = (al > 0.995) & (dc.mean(-1) > 1e-4)
            pl = srgb_dec(pain)
            alb = pl / np.maximum(E * S, 1e-4)
            mode = P.get("delight", "ratio")
            if mode == "none":                    # the painted colour as is, only the exposure removed
                alb = pl / E / max(float(np.median(Sl[inside])) if inside.any() else 1.0, 1e-4)
            alb = np.clip(alb, 0, 1.5)
            med = float(np.median(Sl[inside])) if inside.any() else 1.0
            conf = np.clip(Sl / (0.45 * med), 0, 1) ** 2
            # distance from the silhouette and from depth steps (self-occlusion edges)
            camp = Mw[:3, 3]
            dist = np.linalg.norm(posv - camp, axis=-1)
            dz = np.abs(dist - blur(dist * inside, 1) / np.maximum(blur(inside.astype(np.float32), 1), 1e-6))
            step = (dz > 0.01 * np.median(dist[inside]) if inside.any() else dz > 0) & inside
            e1 = blur(inside.astype(np.float32), edge_r, 1)
            e2 = 1 - np.clip(blur(step.astype(np.float32), edge_r, 1) * 4, 0, 1)
            wimg = np.clip((e1 - 0.5) * 2, 0, 1) * e2 * conf * inside
            low = masked_blur(alb, inside, low_r)
            high = alb - low
            per_view.append(dict(v=v, Mi=Mi, Mw=Mw, W=W, H=H, f=f, posv=posv, wimg=wimg, low=low, high=high,
                                 inside=inside))
    if not per_view:
        raise RuntimeError("no painted views")

    def sample(pv, sel=None):
        """For texels (all or sel) seen by this view: (ids, weights soft/sharp, low, high, density)."""
        Mi, W, H, f = pv["Mi"], pv["W"], pv["H"], pv["f"]
        p = tp if sel is None else tp[sel]
        n = tn if sel is None else tn[sel]
        pc = p @ Mi[:3, :3].T + Mi[:3, 3]
        z = -pc[:, 2]
        ok = z > 1e-6
        x = W / 2 + pc[:, 0] / np.maximum(z, 1e-6) * f
        y = H / 2 + pc[:, 1] / np.maximum(z, 1e-6) * f
        ok &= (x >= 1) & (x < W - 1) & (y >= 1) & (y < H - 1)
        ii = np.nonzero(ok)[0]
        xi, yi = x[ii], y[ii]
        ix, iy = xi.astype(np.int32), yi.astype(np.int32)
        rp = pv["posv"][iy, ix]
        vis = np.linalg.norm(rp - p[ii], axis=1) < tol + z[ii] / f * 1.5      # within ~1.5 pixels' footprint
        vis &= pv["inside"][iy, ix]
        ii = ii[vis]
        xi, yi = xi[vis], yi[vis]
        camp = pv["Mw"][:3, 3]
        vd = camp - p[ii]
        dd = np.linalg.norm(vd, axis=1)
        cosv = np.clip((n[ii] * vd).sum(1) / np.maximum(dd, 1e-9), 0, 1)
        dens = f / dd                                             # pixels per metre at the texel
        wb = bilinear(pv["wimg"], xi, yi)
        lowv = bilinear(pv["low"], xi, yi)
        highv = bilinear(pv["high"], xi, yi)
        return ii, cosv, dens, wb, lowv, highv

    with Timer("gain_match"):
        rng = np.random.default_rng(0)
        sub = rng.choice(T, size=min(T, 200000), replace=False)
        gains = []
        for pv in per_view:
            ii, cosv, dens, wb, lowv, highv = sample(pv, sub)
            good = (wb > 0.5) & (cosv > 0.5)
            if good.sum() < 200:
                gains.append(None)
                continue
            r = np.median(lowv[good], 0) / np.maximum(np.median(ta[sub][ii][good], 0), 1e-4)
            gains.append(r)
        ok = [g for g in gains if g is not None]
        G = np.median(np.array(ok), 0) if ok else np.ones(3)
        for pv, g in zip(per_view, gains):
            pv["gain"] = (G / np.maximum(g, 1e-4)) if (g is not None and P.get("gain_match", True)) else np.ones(3)
            pv["gain"] = np.clip(pv["gain"], 0.5, 2.0)
        REPORT["global_gain"] = [round(float(x), 3) for x in G]
        REPORT["view_gains"] = {pv["v"]["name"]: [round(float(x), 3) for x in pv["gain"]] for pv in per_view}
    dens_ref = None
    with Timer("project_blend"):
        acc_l = np.zeros((T, 3), np.float64)
        acc_lw = np.zeros(T, np.float64)
        acc_h = np.zeros((T, 3), np.float64)
        acc_hw = np.zeros(T, np.float64)
        best = np.zeros(T, np.float32)
        owner = np.full(T, -1, np.int16)
        nviews = np.zeros(T, np.int16)
        samples = []
        for pv in per_view:
            ii, cosv, dens, wb, lowv, highv = sample(pv)
            if dens_ref is None:
                dens_ref = float(np.median(dens)) if len(dens) else 1.0
            g = pv["gain"]
            dn = (dens / dens_ref) ** float(P.get("density_pow", 1.0))
            ws = wb * cosv ** p_soft * dn
            wh = wb * cosv ** p_sharp * dn ** 2
            acc_l[ii] += ws[:, None] * lowv * g
            acc_lw[ii] += ws
            acc_h[ii] += wh[:, None] * highv * g
            acc_hw[ii] += wh
            win = wh > best[ii]
            owner[ii[win]] = len(stats)
            best[ii] = np.maximum(best[ii], wh)
            nviews[ii] += (wb > 0.2).astype(np.int16)
            stats.append({"view": pv["v"]["name"], "texels": int(len(ii)), "weight_sum": round(float(ws.sum()), 1)})
        # sharpen the high band: the strongest view dominates (reduces ghosting where views disagree)
        w0 = float(P.get("fallback_w", 0.02))
        fb = ta * G
        low = (acc_l + w0 * fb) / (acc_lw + w0)[:, None]
        high = acc_h / np.maximum(acc_hw, 1e-9)[:, None] * np.clip(acc_hw / (acc_hw + 0.05), 0, 1)[:, None]
        anchor = float(P.get("anchor", 0.0))
        if anchor > 0:
            # keep the procedural layout's low frequencies (x the global gain): fewer seams, less painted character
            low = (1 - anchor) * low + anchor * fb
        alb = np.clip(low + high * float(P.get("detail_gain", 1.0)), 0, 1)
        seen = acc_lw > float(P.get("seen_w", 0.05))
        REPORT["texels"] = T
        REPORT["seen_fraction"] = round(float(seen.mean()), 4)
        REPORT["views_per_texel"] = {str(k): int((nviews == k).sum()) for k in range(0, int(nviews.max()) + 1)}
        REPORT["view_stats"] = stats
    with Timer("maps"):
        full = np.zeros((res * res, 3), np.float32)
        full[idx] = alb
        full = full.reshape(res, res, 3)
        fill_px = int(P.get("margin_px", max(4, res // 128)))
        full, valid = dilate_fill(full, cov, fill_px)
        full[~valid] = full[valid].mean(0) if valid.any() else 0
        out_dir = JOB["out_dir"]
        os.makedirs(out_dir, exist_ok=True)
        files = {}
        bc_path = os.path.join(out_dir, f"T_{AID}_BaseColor.png")
        save_png(srgb_enc(full), bc_path)
        files["BaseColor"] = bc_path
        # coverage / view-count map for checking (not used by UE)
        cm = np.zeros(res * res, np.float32)
        cm[idx] = np.clip(nviews / 4.0, 0, 1)
        cmr = np.zeros(res * res, np.float32)
        cmr[idx] = (~seen).astype(np.float32)
        save_png(np.stack([cmr.reshape(res, res), cm.reshape(res, res), cov.astype(np.float32)], -1),
                 os.path.join(out_dir, f"_coverage_{AID}.png"))
        # seams: where the dominant (sharpest-weighted) view changes between neighbouring texels, the step in the
        # smoothed colour there against the typical step elsewhere; seam texels = boundary texels whose step is
        # over 3x the typical one and over 2 % (absolute, linear)
        own = np.full(res * res, -1, np.int16)
        own[idx] = owner
        own = own.reshape(res, res)
        sm = blur(full, 1, 1)
        lum_s = (sm * [0.2126, 0.7152, 0.0722]).sum(-1)
        bnd = np.zeros((res, res), bool)
        stp = np.zeros((res, res), np.float32)
        for ax in (0, 1):
            o2 = np.roll(own, -1, ax)
            v2 = np.roll(valid & cov, -1, ax)
            b = (own != o2) & (own >= 0) & (o2 >= 0) & v2 & cov
            st = np.abs(np.roll(lum_s, -2, ax) - np.roll(lum_s, 1, ax))
            bnd |= b
            stp = np.maximum(stp, np.where(b, st, 0))
        inner = cov & ~bnd
        gx = np.abs(np.roll(lum_s, -2, 1) - np.roll(lum_s, 1, 1))[inner]
        typ = float(np.median(gx)) if gx.size else 0.0
        bad = bnd & (stp > max(3 * typ, 0.02))
        REPORT["seams"] = {"boundary_texels": int(bnd.sum()), "seam_texels": int(bad.sum()),
                           "seam_fraction_of_boundary": round(float(bad.sum()) / max(int(bnd.sum()), 1), 4),
                           "typical_step": round(typ, 4),
                           "median_boundary_step": round(float(np.median(stp[bnd])) if bnd.any() else 0, 4)}
        save_png(np.stack([bad.astype(np.float32), bnd.astype(np.float32) * 0.4, cov * 0.15], -1),
                 os.path.join(out_dir, f"_seams_{AID}.png"))
        # detail from the painted colour: luminance high-pass, relative to the local level
        L = (full * [0.2126, 0.7152, 0.0722]).sum(-1)
        hr = float(P.get("height_px", 3))
        Lb = masked_blur(L, valid, hr * 3)
        hp = np.where(valid, (L - Lb) / np.maximum(Lb + 0.02, 1e-3), 0)
        hp = np.clip(hp, -1, 1)
        ns = float(P.get("normal_strength", 0.0))
        nor = px(imgs["Normal"].image)[..., :3] * 2 - 1
        orm = px(imgs["ORM"].image)[..., :3].copy()
        if ns > 0:
            h = blur(hp, 1, 1)
            gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
            gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
            dx, dy = -gx * ns, -gy * ns
            # whiteout blend of the detail onto the procedural tangent normal
            nn = np.stack([nor[..., 0] + dx, nor[..., 1] + dy, nor[..., 2]], -1)
            nn /= np.maximum(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-6)
            nor = np.where(valid[..., None], nn, nor)
        rs = float(P.get("rough_strength", 0.0))
        if rs > 0:
            # brighter than the surroundings (worn, polished) -> smoother; darker (pits, scale, rust) -> rougher
            orm[..., 1] = np.clip(orm[..., 1] * (1 - rs * np.clip(hp, -1, 1)), 0.04, 1)
        npath = os.path.join(out_dir, f"T_{AID}_Normal.png")
        save_png(nor * 0.5 + 0.5, npath)
        files["Normal"] = npath
        opath = os.path.join(out_dir, f"T_{AID}_ORM.png")
        save_png(orm, opath)
        files["ORM"] = opath
        REPORT["files"] = files
    with Timer("export"):
        for k, n in imgs.items():
            im = bpy.data.images.load(files[k], check_existing=False)
            im.colorspace_settings.name = "sRGB" if k == "BaseColor" else "Non-Color"
            old_name = n.image.name
            n.image.name = old_name + "_procedural"
            im.name = old_name.split(".")[0]                       # UE texture names stay T_<id>_<map>
            n.image = im
        if JOB.get("glb"):
            bpy.ops.object.select_all(action="DESELECT")
            ob.select_set(True)
            bpy.context.view_layer.objects.active = ob
            bpy.ops.export_scene.gltf(filepath=JOB["glb"], export_format="GLB", use_selection=True,
                                      export_apply=True, export_yup=True, export_image_format="AUTO",
                                      export_tangents=True)
            REPORT["files"]["glb"] = JOB["glb"]
        if JOB.get("save_blend"):
            bpy.ops.file.pack_all()
            bpy.ops.wm.save_as_mainfile(filepath=JOB["save_blend"], compress=True)
    REPORT["times_s"]["total"] = round(time.time() - t0, 2)


# ---------------------------------------------------------------- stills
def stills():
    sc, ob, vs, lo, hi = open_asset()
    set_device(sc)
    m, imgs = mat_images(ob)
    maps = JOB.get("maps", "procedural")
    if maps != "procedural":
        for k, n in imgs.items():
            p = os.path.join(maps, f"T_{AID}_{k}.png")
            im = bpy.data.images.load(p, check_existing=False)
            im.colorspace_settings.name = "sRGB" if k == "BaseColor" else "Non-Color"
            n.image = im
    dims = hi - lo
    r = float(np.linalg.norm(dims) / 2)
    w = bpy.data.worlds.new("st")
    sc.world = w
    w.use_nodes = True
    bgn = w.node_tree.nodes["Background"]
    bgn.inputs["Color"].default_value = (0.35, 0.36, 0.38, 1)
    bgn.inputs["Strength"].default_value = 0.6
    for name, loc, energy, size in (("key", (-3, -4, 5), 900, 3), ("rim", (3, 4, 3), 400, 2)):
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy * (r / 0.4) ** 2
        ld.size = size * r / 0.4 + 0.5
        lo_ = bpy.data.objects.new(name, ld)
        lo_.location = Vector(loc) * (r / 0.4 + 0.3)
        sc.collection.objects.link(lo_)
        d = -lo_.location + Vector((0, 0, dims[2] / 2))
        lo_.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    cd = bpy.data.cameras.new("st_cam")
    cam = bpy.data.objects.new("st_cam", cd)
    sc.collection.objects.link(cam)
    sc.camera = cam
    sc.cycles.samples = int(JOB.get("samples", 64))
    sc.cycles.use_denoising = True
    sc.render.film_transparent = False
    sc.view_settings.view_transform = "AgX"
    sc.render.image_settings.file_format = "PNG"
    os.makedirs(JOB["out"], exist_ok=True)
    outs = []
    for s in JOB["shots"]:
        W, H = s.get("px", [1024, 768])
        sc.render.resolution_x, sc.render.resolution_y = W, H
        tgt = lo + dims * np.array(s.get("target", [0.5, 0.5, 0.5]))
        a, e = math.radians(s["az"]), math.radians(s["el"])
        d = np.array([math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)])
        cd.lens = float(s.get("lens", 50))
        fov = 2 * math.atan(18 / cd.lens)
        span = float(s.get("span", 1.0)) * r                       # half-width of what the frame shows, metres
        dist = span / math.tan(fov / 2)
        cd.clip_start = dist * 0.01
        look_at(cam, Vector(tgt + d * dist), Vector(tgt))
        p = os.path.join(JOB["out"], f"{s['name']}.png")
        sc.render.filepath = p
        bpy.ops.render.render(write_still=True)
        outs.append(p)
    REPORT["files"] = outs


def main():
    try:
        {"guides": guides, "project": project, "stills": stills}[JOB["mode"]]()
        REPORT["status"] = "ok"
    except Exception:
        REPORT["status"] = "error"
        REPORT["error"] = traceback.format_exc()
        sys.stderr.write(REPORT["error"])
    with open(JOB["report"], "w") as f:
        json.dump(REPORT, f, indent=1)
    sys.exit(0 if REPORT["status"] == "ok" else 2)


main()
