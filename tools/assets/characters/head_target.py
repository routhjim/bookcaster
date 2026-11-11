"""Target head for the MetaHuman conform: a front picture of a face -> image-to-3D mesh -> vertices in the
picture's own pixel frame, so the conform's 2D face tracking (done on the same picture) and the 3D target agree.

    python head_target.py prep sheet.png head.png --box X0,Y0,X1,Y1 [--cut-y Y] [--neck X0,X1,Y]
        crop the head out of a reference picture and key its plain background to alpha (by the corner colour);
        rows below --cut-y are cleared, and below the neck line Y only the columns X0..X1 are kept (drops the
        collar and shoulders so the target is a head and neck). Pasting and keying only (numpy, milliseconds).
    <image-to-3D>  head.png -> head.glb   (tools/blockout_shape.py, single view, on a locked card)
    python head_target.py target head.glb head.png target.json [--forward +z|-z|+x|-x] [--faces N]
        mesh (Y up, as the image-to-3D model writes it) -> turned so the face looks at the viewer, scaled and
        placed so that its front projection covers the picture's keyed foreground (bounding boxes matched, the
        scale from the height). Writes {"image", "size", "verts": [x_px, y_px, z_px, ...] (x right, y down, z
        toward the viewer, all in picture pixels), "tris", "forward", "x_scale_mismatch"}. The UE side
        (char_fit.py) maps pixels to centimetres and gives the conform a camera that sees this picture.

Needs numpy + Pillow (prep) and trimesh (target): run `target` with the image-to-3D tool's interpreter.
"""

import argparse
import json
import sys

import numpy as np
from PIL import Image


def prep(a):
    x0, y0, x1, y1 = (int(v) for v in a.box.split(","))
    im = Image.open(a.src).convert("RGB").crop((x0, y0, x1, y1))
    px = np.asarray(im).astype(np.float32)
    bg = np.concatenate([px[:6, :6].reshape(-1, 3), px[:6, -6:].reshape(-1, 3)]).mean(0)
    al = np.clip((np.abs(px - bg).sum(-1) - 18) * 12, 0, 255)
    yy, xx = np.mgrid[0:al.shape[0], 0:al.shape[1]]
    if a.cut_y is not None:
        al[yy >= a.cut_y - y0] = 0
    if a.neck:
        nx0, nx1, ny = (int(v) for v in a.neck.split(","))
        al[(yy >= ny - y0) & ((xx < nx0 - x0) | (xx > nx1 - x0))] = 0
    for y in range(al.shape[0]):                  # close holes inside the outline row by row (pale skin keys out)
        idx = np.where(al[y] > 128)[0]
        if len(idx) > 1:
            al[y, idx[0]:idx[-1] + 1] = 255
    out = np.concatenate([px.astype(np.uint8), al.astype(np.uint8)[..., None]], -1)
    Image.fromarray(out, "RGBA").save(a.out)
    print(json.dumps({"out": a.out, "box": [x0, y0, x1, y1], "bg": [round(float(v), 1) for v in bg]}))


AXES = {"+z": (2, 1), "-z": (2, -1), "+x": (0, 1), "-x": (0, -1)}


def nose_score(v, axis, sign):
    """How pointed the mesh is at its extreme along +-axis in the face band (mid height, central columns): the
    face side ends in the nose (few vertices near the extreme), the back of the head is broad."""
    up = v[:, 1]
    lat = v[:, 2 if axis == 0 else 0]
    h0, h1 = up.min(), up.max()
    band = (up > h0 + 0.35 * (h1 - h0)) & (up < h0 + 0.7 * (h1 - h0))
    d = sign * v[:, axis]
    ext = d[band].max()
    depth = d.max() - d.min()
    near = band & (d > ext - 0.04 * depth)
    return float(np.ptp(lat[near])) / float(np.ptp(lat[band]) + 1e-9) if near.any() else 1.0


def target(a):
    import trimesh
    m = trimesh.load(a.mesh, force="mesh")
    if a.faces and len(m.faces) > a.faces:
        m = m.simplify_quadric_decimation(face_count=a.faces)
    v = np.asarray(m.vertices, dtype=np.float64)
    scores = {k: nose_score(v, *AXES[k]) for k in AXES}
    fwd = a.forward or min(scores, key=scores.get)
    axis, sign = AXES[fwd]
    # rotate about Y so the chosen forward becomes +Z (x right as seen from the front, y up)
    if axis == 2:
        x, z = (v[:, 0], v[:, 2]) if sign > 0 else (-v[:, 0], -v[:, 2])
    else:
        x, z = (-v[:, 2], v[:, 0]) if sign > 0 else (v[:, 2], -v[:, 0])
    y = v[:, 1]
    al = np.asarray(Image.open(a.image).convert("RGBA"))[..., 3]
    rows, cols = np.where(al > 128)
    u0, u1, v0, v1 = cols.min(), cols.max(), rows.min(), rows.max()
    k = (v1 - v0) / (y.max() - y.min())                       # pixels per mesh unit, from the height
    xs = (u0 + u1) / 2 + (x - (x.min() + x.max()) / 2) * k
    ys = v1 - (y - y.min()) * k
    zs = z * k
    out = {"image": a.image, "size": [int(al.shape[1]), int(al.shape[0])], "forward": fwd, "nose_scores": scores,
           "fg_box": [int(u0), int(v0), int(u1), int(v1)],
           "x_scale_mismatch": round(float((x.max() - x.min()) * k / (u1 - u0)), 3),
           "verts": np.round(np.stack([xs, ys, zs], 1).reshape(-1), 3).tolist(),
           "tris": np.asarray(m.faces, dtype=np.int64).reshape(-1).tolist()}
    json.dump(out, open(a.out, "w"))
    print(json.dumps({k_: out[k_] for k_ in ("forward", "nose_scores", "fg_box", "x_scale_mismatch", "size")}
                     | {"verts": len(v), "faces": len(m.faces)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--box", required=True)
    p.add_argument("--cut-y", type=int)
    p.add_argument("--neck")
    t = sub.add_parser("target")
    t.add_argument("mesh")
    t.add_argument("image")
    t.add_argument("out")
    t.add_argument("--forward", choices=list(AXES))
    t.add_argument("--faces", type=int, default=0)
    a = ap.parse_args()
    return prep(a) if a.cmd == "prep" else target(a)


if __name__ == "__main__":
    sys.exit(main())
