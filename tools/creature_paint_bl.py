"""Paint a procedural/imported creature from its reference drawings (any book): vertex colours projected from
the full-body FRONT drawing (the creature sheet's body) and, on the head, the SIDE head drawing.

    blender -b --factory-startup -t 16 --python tools/creature_paint_bl.py -- RIGGED.blend OUT.blend \
        --body body_front_rgba.png --head head_side_rgba.png [--head-faces left|right] [--dark 0.8]

The model faces +X, left = +Y, Z up (rig.json joints beside the blend: "head", "neck5"/"neck4"). Front projection:
image u = model y (its left on the image's right), v = z, fitted bbox to the drawing's opaque bbox; head:
u = model x (the snout toward the drawing's snout side), v = z, over the head region, blended over ~0.1 S.
The painter (low strength keys) then follows real scale colour, eyes and horn colour instead of plain clay.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

a = sys.argv[sys.argv.index("--") + 1:]
src, dst = a[0], a[1]


def arg(k, d=None):
    return a[a.index(k) + 1] if k in a else d


bpy.ops.wm.open_mainfile(filepath=src)
ob = bpy.data.objects["model"]
RJ = json.loads((Path(src).parent / "rig.json").read_text())
J = {k: np.asarray(v, float) for k, v in RJ["joints"].items()}
S = float(RJ.get("height", 6.0))
P = np.array([v.co[:] for v in ob.data.vertices])


def load(path):
    im = bpy.data.images.load(path)
    w, h = im.size
    px = np.array(im.pixels[:], np.float32).reshape(h, w, 4)[::-1]
    al = px[:, :, 3] > 0.5
    ys, xs = np.where(al)
    return px, (xs.min(), xs.max(), ys.min(), ys.max())


def sample(px, bb, u, v):
    x0, x1, y0, y1 = bb
    xi = np.clip((x0 + u * (x1 - x0)).astype(int), 0, px.shape[1] - 1)
    yi = np.clip((y0 + v * (y1 - y0)).astype(int), 0, px.shape[0] - 1)
    c = px[yi, xi, :3]
    al = px[yi, xi, 3:4]
    return c, al


body, bbb = load(arg("--body"))
lo, hi = P.min(0), P.max(0)
u = (P[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-9)
v = (hi[2] - P[:, 2]) / max(hi[2] - lo[2], 1e-9)
C, A = sample(body, bbb, u, v)
base = np.array([0.08, 0.06, 0.05], np.float32)            # off the drawing's figure: the look's dark scale colour
C = C * A + base * (1 - A)
if arg("--head"):
    head, hbb = load(arg("--head"))
    hc = J.get("head")
    nk = J.get("neck5", J.get("neck4", hc))
    R0 = 0.22 * S
    d = np.linalg.norm(P - hc, axis=1)
    # the SKULL's own skin (dominant bone head / head_end / jaw): its bbox maps to the drawing's head box
    gi = {g.index: g.name for g in ob.vertex_groups}
    dom = np.array([gi.get(max(vv_.groups, key=lambda g: g.weight).group, "") if len(vv_.groups) else ""
                    for vv_ in ob.data.vertices])
    hv = np.isin(dom, ["head", "head_end", "jaw"])
    if arg("--head-box"):                                  # the head's part of the drawing (fractions x0 y0 x1 y1)
        fx0, fy0, fx1, fy1 = [float(x) for x in arg("--head-box").split(",")]
        H_, W_ = head.shape[:2]
        hbb = (int(fx0 * W_), int(fx1 * W_), int(fy0 * H_), int(fy1 * H_))
    if arg("--head-fit") and hv.any():
        # LANDMARK FIT: model (x, z) as fractions of S -> drawing pixels, least-squares affine over >= 3 pairs
        # (eye, snout tip, horn root, mouth corner, chin): the drawing lands on the skull the way it is drawn
        pr = [q.split(":") for q in arg("--head-fit").split(";") if q]
        Mx = np.array([[float(v_) for v_ in a_.split(",")] for a_, _ in pr]) * S
        Ix = np.array([[float(v_) for v_ in b_.split(",")] for _, b_ in pr])
        Aa = np.c_[Mx, np.ones(len(Mx))]
        Ac, *_ = np.linalg.lstsq(Aa, Ix, rcond=None)
        hv2 = hv | (np.isin(dom, ["neck5"]))
        Q = P[hv2]
        UV = np.c_[Q[:, [0, 2]], np.ones(len(Q))] @ Ac
        xi = np.clip(UV[:, 0].astype(int), 0, head.shape[1] - 1)
        yi = np.clip(UV[:, 1].astype(int), 0, head.shape[0] - 1)
        Ch, Ah = head[yi, xi, :3], head[yi, xi, 3:4]
        inb = ((UV[:, 0] >= 0) & (UV[:, 0] < head.shape[1]) & (UV[:, 1] >= 0) & (UV[:, 1] < head.shape[0]))[:, None]
        w = Ah * inb
        C[hv2] = C[hv2] * (1 - w) + Ch * w
        hv = hv2
        resid = float(np.abs(np.c_[Mx, np.ones(len(Mx))] @ Ac - Ix).mean())
        print("HEAD FIT mean residual px", round(resid, 1))
    elif hv.any():
        Q = P[hv]
        hl, hh = Q.min(0), Q.max(0)
        uu = (Q[:, 0] - hl[0]) / max(hh[0] - hl[0], 1e-9)
        if arg("--head-faces", "left") == "left":
            uu = 1.0 - uu                                  # the drawing's snout points left
        vv = (hh[2] - Q[:, 2]) / max(hh[2] - hl[2], 1e-9)
        Ch, Ah = sample(head, hbb, uu, vv)
        w = Ah                                             # (transparent drawing pixels keep the body colour)
        C[hv] = C[hv] * (1 - w) + Ch * w
C = np.clip(C * float(arg("--dark", "1.0")), 0, 1)
ca = ob.data.color_attributes.get("Col") or ob.data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
ca.data.foreach_set("color", np.concatenate([C, np.ones((len(C), 1), np.float32)], 1).ravel())
m = bpy.data.materials.new("creature_painted")
m.use_nodes = True
nt = m.node_tree
bs = nt.nodes["Principled BSDF"]
at = nt.nodes.new("ShaderNodeAttribute")
at.attribute_name = "Col"
nt.links.new(at.outputs["Color"], bs.inputs["Base Color"])
bs.inputs["Roughness"].default_value = 0.75
ob.data.materials.clear()
ob.data.materials.append(m)
bpy.ops.wm.save_as_mainfile(filepath=dst)
print("PAINTED", len(P), "verts; head verts", int(hv.sum()) if arg("--head") else 0)
