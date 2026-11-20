"""A place's concept picture as a room: its depth (MoGe-2) read into the room's size and the place of each thing it
shows, for the set builder (sets/place_build.py). Run at the image station with the depth model's interpreter:

    python tools/assets/img3d/concept_depth.py CONCEPT.png CONCEPT.json OUT.json

CONCEPT.json is scenes/concept.py's reading of the picture: the spec's pieces it shows ("spec": {id: {"box"}}) and the
things it shows besides ("seen": [{"kind", "box"}]), each box [x0, y0, x1, y1] in 0..1 of the picture.

The depth model gives every pixel a point in metres (camera frame). The floor is the plane the bottom of the picture
lies on (fitted to it); the room's frame stands on it: x to the picture's right, y away from the eye, z up. In it:
  the back wall    where the points at standing height, across the middle of the picture, end;
  the side walls   where they end to the left and the right;
  the ceiling      the top of the points overhead;
  each thing       where the ray through the middle of its box's bottom edge meets the floor (its front foot), its
                   width from its bottom corners, its height from its box's top, its depth from its kind's shape;
                   the wall it stands against when its back comes within AGAINST_M of one.
Scaled so a door seen in the picture is DOOR_M high (the model's metres are near, not exact). Then the room is
centred on its floor, the eye standing inside its front part.

OUT.json: {"room_m": [W, D, H], "eye": [x, y, z], "things": {"spec:<id>" | "seen:<i>": {"xy", "size_m", "against",
"facing", "raised_m" (it stands on a raised part of the floor, this high)}}, "off_floor": {key: {"kind"}} (things the
picture shows hanging: not placed),
"scale", "floor_fit"} (set frame: the room's middle at 0, y toward the back wall).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOOR_M = 2.0                 # a door's height, for the scale
SCALE_CLAMP = (0.6, 1.6)     # the depth model's metres are trusted within this much
AGAINST_M = 0.5              # a thing whose back is this near a wall stands against it
OFF_FLOOR_M = 0.4            # a thing whose own points are this much nearer than its box's foot on the floor is
                             # not standing there: it hangs on a wall or from the roof, or stands on another thing
RAISED_MOST_M = 1.0          # ... unless its bottom is no higher than this: it stands on a raised part of the floor
FRONT_M = 1.2                # the room reaches at least this far behind the eye (the picture's front wall is unseen)
STAND = (0.3, 2.0)           # the height band (m) whose points mark where the walls are


def infer(image: np.ndarray, device: str):
    import torch
    from moge.model.v2 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    t = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)
    out = model.infer(t)
    pts, mask = out["points"].cpu().numpy(), out["mask"].cpu().numpy()
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return pts, mask > 0


def fit_floor(pts: np.ndarray, mask: np.ndarray, rng=None) -> tuple:
    """The floor plane (n, d: n.x + d = 0, n toward the eye) from the picture's bottom fifth: a RANSAC plane."""
    rng = rng or np.random.default_rng(7)
    h = pts.shape[0]
    sel = pts[int(h * 0.8):][mask[int(h * 0.8):]]
    sel = sel[np.isfinite(sel).all(1)]
    if len(sel) < 50:
        raise SystemExit("too few floor points")
    best, bn = None, 0
    for _ in range(300):
        a, b, c = sel[rng.choice(len(sel), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ a
        inl = np.abs(sel @ n + d) < 0.03
        if inl.sum() > bn:
            best, bn = (n, d), inl.sum()
    n, d = best
    inl = sel[np.abs(sel @ n + d) < 0.03]
    c0 = inl.mean(0)
    # the plane's normal: the direction the inliers vary least in (their 3x3 covariance; a full SVD of the N points
    # builds an N x N matrix: 116 GB for a real picture's floor, which took the box down on 11-17)
    n = np.linalg.eigh(np.cov((inl - c0).T))[1][:, 0]
    d = -n @ c0
    if d < 0:                                  # the eye (the origin) on the plane's positive side
        n, d = -n, -d
    return n, float(d), float(bn) / len(sel)


def frame(n: np.ndarray, d: float):
    """The room frame's axes in camera coordinates: right (the picture's x along the floor), forward, up."""
    up = n
    r = np.array([1.0, 0.0, 0.0])
    r = r - up * (r @ up)
    r /= np.linalg.norm(r)
    fwd = np.cross(up, r)
    return r, fwd, up


def to_room(p: np.ndarray, axes, d: float) -> np.ndarray:
    r, f, u = axes
    return np.stack([p @ r, p @ f, p @ u + d], -1)


def ray(intr: np.ndarray, u: float, v: float) -> np.ndarray:
    """The camera ray through a picture point (u, v in 0..1), from normalised intrinsics."""
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    return np.array([(u - cx) / fx, (v - cy) / fy, 1.0])


def on_floor(rv: np.ndarray, n: np.ndarray, d: float) -> np.ndarray | None:
    den = n @ rv
    if abs(den) < 1e-6:
        return None
    t = -d / den
    return rv * t if t > 0 else None


def stand_before(things: dict) -> None:
    """Two things against one wall on the same stretch of it: the deeper one stands before the flatter one, clear of
    it (a seat and the hanging behind it were both put flat on the wall, one inside the other, and no camera could
    be found for the one who sits there, 11-19)."""
    ks = [k for k, t in things.items() if t.get("against")]
    for a in ks:
        for b in ks:
            ta, tb = things[a], things[b]
            if a == b or ta["against"] != tb["against"] or ta["size_m"][1] <= tb["size_m"][1]:
                continue
            i = 0 if ta["against"] == "north" else 1          # the axis along the wall
            if abs(ta["xy"][i] - tb["xy"][i]) >= (ta["size_m"][0] + tb["size_m"][0]) / 2:
                continue
            j, sg = 1 - i, {"north": -1.0, "east": -1.0, "west": 1.0}[ta["against"]]
            ta["xy"][j] = round(tb["xy"][j] + sg * (tb["size_m"][1] / 2 + ta["size_m"][1] / 2 + 0.05), 3)


RELIEF_RISE_M = 0.4         # what stands this far above the ground is the land's form, not the ground
RELIEF_FRONT_M = 1.0        # ... and no nearer the eye than the form's own front, less this
RELIEF_FAR_M = 26.0         # ... and no farther from the eye than this: the far land is the panorama's
RELIEF_GRID = 200           # points across the picture
RELIEF_ON_M = (8.0, 10.0)   # a form the frame cuts is carried on past it: this far up, this far sideways
RELIEF_ON_STEP_M = 0.5
RELIEF_GAP_M = 5.0          # a face this long joins two things: left out
RELIEF_BACK_M = 1.0         # the form's back stands this far behind its farthest point
RELIEF_LEAST = 40           # a form of fewer faces across than this is no form


def _grid(pts, mask, axes, d: float) -> tuple:
    """The picture's points thinned to RELIEF_GRID across, in the room's frame -> (P, ok, U, V)."""
    h0, w0 = pts.shape[:2]
    st = max(1, w0 // RELIEF_GRID)
    rr, cc = np.arange(st // 2, h0, st), np.arange(st // 2, w0, st)
    P = to_room(pts[rr][:, cc], axes, d)
    ok = mask[rr][:, cc] & np.isfinite(P).all(-1)
    P = np.where(ok[..., None], P, 0.0)
    U, V = np.meshgrid((cc + 0.5) / w0, (rr + 0.5) / h0)
    return P, ok, U, V


def _carried(P, reg, U, V) -> tuple:
    """What the frame's edges cut is carried on past them: up from the top row, sideways from the end columns (the
    picture mirrored there)."""
    h, w = reg.shape
    ku, ks = int(RELIEF_ON_M[0] / RELIEF_ON_STEP_M), int(RELIEF_ON_M[1] / RELIEF_ON_STEP_M)
    if reg[0].any():
        k = np.arange(ku, 0, -1)[:, None]
        up = np.repeat(P[:1], ku, 0) + np.stack([0 * k, 0 * k, k * RELIEF_ON_STEP_M], -1) * np.ones((1, w, 1))
        P, reg = np.concatenate([up, P]), np.concatenate([np.repeat(reg[:1], ku, 0), reg])
        U = np.concatenate([np.repeat(U[:1], ku, 0), U])
        V = np.concatenate([np.repeat(V[1:ku + 1][::-1], 1, 0) if ku < h else np.repeat(V[:1], ku, 0), V])
    for side in (0, -1):
        col = reg[:, side]
        if not col.any():
            continue
        k = (np.arange(ks, 0, -1) if side == 0 else np.arange(1, ks + 1))[None, :]
        sg = -1.0 if side == 0 else 1.0
        edge = P[:, side][:, None, :]
        on = edge + np.stack([sg * k * RELIEF_ON_STEP_M, 0 * k, 0 * k], -1)
        ue = (U[:, 1:ks + 1][:, ::-1] if side == 0 else U[:, -ks - 1:-1][:, ::-1]) if ks < w else np.repeat(U[:, side][:, None], ks, 1)
        ve = np.repeat(V[:, side][:, None], ks, 1)
        re = np.repeat(col[:, None], ks, 1)
        P = np.concatenate([on, P], 1) if side == 0 else np.concatenate([P, on], 1)
        U = np.concatenate([ue, U], 1) if side == 0 else np.concatenate([U, ue], 1)
        V = np.concatenate([ve, V], 1) if side == 0 else np.concatenate([V, ve], 1)
        reg = np.concatenate([re, reg], 1) if side == 0 else np.concatenate([reg, re], 1)
    return P, reg, U, V


def _front(q) -> tuple:
    """The faces of the squares q of a grid, wound toward the eye -> (faces, the points they use)."""
    h, w = q.shape[0] + 1, q.shape[1] + 1
    qi, qj = np.nonzero(q)
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, e = idx[qi, qj], idx[qi + 1, qj], idx[qi, qj + 1], idx[qi + 1, qj + 1]
    front_f = np.concatenate([np.stack([a, b, c], -1), np.stack([c, b, e], -1)])
    return front_f, np.unique(front_f)


def _closed(F, B, front_f, U, V) -> tuple:
    """One solid of a front (points F, faces front_f) and its back (the same points at B): the back's faces turned
    away, every edge of the front that only one face has joined to the back. -> (points, faces, uv: the front on
    the left half of a picture twice as wide, the back on its right)."""
    n0 = len(F)
    back_f = front_f[:, ::-1] + n0
    ed = np.concatenate([front_f[:, [0, 1]], front_f[:, [1, 2]], front_f[:, [2, 0]]])
    key = np.sort(ed, 1)
    _, inv, cnt = np.unique(key[:, 0] * (2 * n0) + key[:, 1], return_inverse=True, return_counts=True)
    rim = ed[cnt[inv] == 1]
    u_, v_ = rim[:, 0], rim[:, 1]
    side_f = np.concatenate([np.stack([u_, u_ + n0, v_ + n0], -1), np.stack([u_, v_ + n0, v_], -1)])
    vtx = np.concatenate([F, B])
    faces = np.concatenate([front_f, back_f, side_f])
    uv = np.stack([U.reshape(-1) / 2, 1.0 - V.reshape(-1)], -1)
    uv = np.concatenate([uv, uv + [0.5, 0.0]])
    keep = np.unique(faces)
    re_ = np.full(len(vtx), -1)
    re_[keep] = np.arange(len(keep))
    return vtx[keep], re_[faces], uv[keep]


def relief(pts, mask, axes, d: float, box: list, to_set, region: dict | None = None) -> tuple | None:
    """The land's own form (a cliff, a rock face, a crag) as the picture shows it: every point of the picture that
    rises from the ground and is joined to the form's box, at the form's distance or beyond it, as one solid whose
    front is the picture's own surface and whose back is flat; where the frame cuts the form it is carried on, up
    and to the sides. -> (points [n, 3] in the set's frame, faces, uv in a picture twice as wide as the place's:
    the front on its left half, the back on its right), or None when no form is found. region, when given, is
    filled with the form's own points of the grid ("reg").

    Why not a shape made from the form's part of the picture: a picture-to-3D model makes a whole separate thing of
    what it is given, and a rock face with a cave mouth, cut from its picture, came back as a stone hut three metres
    across standing alone on a plain (11-19). The form runs out of its picture; only the picture's depth has it
    where it is and as it is."""
    import scipy.ndimage as ndi
    P, ok, U, V = _grid(pts, mask, axes, d)
    h, w = ok.shape
    near = ok & (np.hypot(P[..., 0], P[..., 1]) < RELIEF_FAR_M)
    rise = near & (P[..., 2] > RELIEF_RISE_M)
    r0, r1 = int(box[1] * h), max(int(box[3] * h), int(box[1] * h) + 1)
    c0, c1 = int(box[0] * w), max(int(box[2] * w), int(box[0] * w) + 1)
    seed = np.zeros_like(rise)
    seed[r0:r1, c0:c1] = rise[r0:r1, c0:c1]
    if seed.sum() < RELIEF_LEAST:
        return None
    front = float(np.percentile(P[..., 1][seed], 5)) - RELIEF_FRONT_M
    lab, _ = ndi.label(rise & (P[..., 1] > front))
    ids = np.unique(lab[seed & (lab > 0)])
    if not len(ids):
        return None
    reg = np.isin(lab, ids)
    # down to the ground it stands on (the points just under its foot)
    reg = ndi.binary_dilation(reg, iterations=2) & near & (P[..., 1] > front)
    reg0 = reg
    P, reg, U, V = _carried(P, reg, U, V)
    P[..., 2] = np.maximum(P[..., 2], 0.0)
    # the faces: each square of four of its points, none of them a stretch between two things
    q = reg[:-1, :-1] & reg[1:, :-1] & reg[:-1, 1:] & reg[1:, 1:]
    q &= (np.linalg.norm(P[:-1, :-1] - P[1:, 1:], axis=-1) < RELIEF_GAP_M) & \
         (np.linalg.norm(P[1:, :-1] - P[:-1, 1:], axis=-1) < RELIEF_GAP_M)
    q = ndi.binary_opening(q, structure=np.ones((2, 2), bool))
    lab, n_ = ndi.label(q)
    if not n_:
        return None
    q = lab == (1 + int(np.argmax(ndi.sum(q, lab, range(1, n_ + 1)))))
    if q.sum() < RELIEF_LEAST:
        return None
    front_f, used = _front(q)
    F = to_set(P.reshape(-1, 3))
    # its back: the same points on an even grid, flat, behind its farthest point
    h, w = reg.shape
    lo, hi = F[used].min(0), F[used].max(0)
    gx, gz = np.meshgrid(np.linspace(lo[0], hi[0], w), np.linspace(hi[2], 0.0, h))
    B = np.stack([gx, np.full_like(gx, hi[1] + RELIEF_BACK_M), gz], -1).reshape(-1, 3)
    if region is not None:
        region["reg"] = reg0
    return _closed(F, B, front_f, U, V)


LAND_FAR_M = 90.0           # the picture's land is built this far out from its eye; beyond it the panorama's
LAND_UNDER_M = 0.03         # its ground lies this far under the set's real ground, out of sight
LAND_JUMP = 1.35            # neighbours whose distances from the eye differ by more than this are two things, one
#                             behind the other: no face between them (the far one is seen past the near one's edge)
LAND_BACK_K = 1.15          # its back: every point this much farther out along its own line from the eye, so the
#                             solid's sides are seen end on from where the picture was drawn
LAND_CUT_FAR_M = 24.0       # a thing of the picture nearer its eye than this is built as a piece of the set ...
LAND_CUT_BIG_M = 30.0       # ... unless it is larger than this: those are cut out of the land, the rest are the land's
LAND_CUT_BEHIND_M = 0.5


def land(pts, mask, axes, d: float, to_set, things: dict, boxes: dict, image: np.ndarray,
         keep_out: np.ndarray | None = None) -> tuple | None:
    """All the land a place in the open's picture shows, as one solid standing where the picture has it: its ground
    (just under the set's real ground), and every bank, slope, rock, tree and far hill rising from it, the front the
    picture's own surface. What the frame cuts is carried on. The things that are built as pieces of the set
    (things: their place and size in the set's frame; boxes: their part of the picture) are taken out of it, the
    land behind each closed over; keep_out is a part of the grid left out whole (the land's own form, a solid of
    its own). -> (points in the set's frame, faces, uv, the picture with the pieces painted over, the keys of the
    things left in it, not to be built again), or None.

    Why: the set of a place in the open was a flat disc of ground forty metres across with the pieces cut from the
    picture stood on it, and nothing of the picture's land: every shot of every such place had bare flat ground to
    the horizon behind its people (11-19)."""
    import scipy.ndimage as ndi
    P, ok, U, V = _grid(pts, mask, axes, d)
    h, w = ok.shape
    eye = to_room(np.zeros(3), axes, d)
    reg = ok & (np.hypot(P[..., 0] - eye[0], P[..., 1] - eye[1]) < LAND_FAR_M)
    if keep_out is not None:
        reg &= ~keep_out
    if reg.sum() < RELIEF_LEAST:
        return None
    F0 = to_set(P.reshape(-1, 3)).reshape(h, w, 3)
    e_set = to_set(eye[None])[0]
    img = image.copy()
    sy, sx = image.shape[0] / h, image.shape[1] / w
    has = []
    for key, t in things.items():
        if key not in boxes:
            continue
        dist = math.hypot(t["xy"][0] - e_set[0], t["xy"][1] - e_set[1])
        if key.startswith("seen:") and (dist > LAND_CUT_FAR_M or max(t["size_m"]) > LAND_CUT_BIG_M):
            has.append(key)                    # (far off or vast: it is the land's, and not built again)
            continue
        b = boxes[key]
        r0, r1 = max(0, int(b[1] * h)), min(h, int(math.ceil(b[3] * h)))
        c0, c1 = max(0, int(b[0] * w)), min(w, int(math.ceil(b[2] * w)))
        m = np.zeros((h, w), bool)
        m[r0:r1, c0:c1] = reg[r0:r1, c0:c1] & (P[r0:r1, c0:c1, 2] > RELIEF_RISE_M / 2) & \
            (F0[r0:r1, c0:c1, 1] <= t["xy"][1] + t["size_m"][1] / 2 + LAND_CUT_BEHIND_M)
        if not m.any():
            continue
        m = ndi.binary_dilation(m, iterations=1) & reg
        # the land behind it, closed over: along each row from what lies on either side of it
        for r in np.nonzero(m.any(1))[0]:
            cols = np.nonzero(m[r])[0]
            for run in np.split(cols, np.nonzero(np.diff(cols) > 1)[0] + 1):
                a, z = run[0] - 1, run[-1] + 1
                la, lz = a >= 0 and reg[r, a], z < w and reg[r, z]
                if not (la and lz):           # (nothing on one side of it to close the land over from)
                    reg[r, run] = False
                    continue
                f = ((run - a) / max(z - a, 1))[:, None]
                P[r, run] = P[r, a] * (1 - f) + P[r, z] * f
                y0, y1 = int(r * sy), max(int((r + 1) * sy), int(r * sy) + 1)
                xa, xz = min(int((a + 0.5) * sx), image.shape[1] - 1), min(int((z + 0.5) * sx), image.shape[1] - 1)
                x0, x1 = int(run[0] * sx), min(int((run[-1] + 1) * sx), image.shape[1])
                if x1 > x0:
                    g = (np.arange(x0, x1) - xa)[None, :, None] / max(xz - xa, 1) if xz != xa else np.zeros((1, x1 - x0, 1))
                    g = np.clip(g, 0.0, 1.0)
                    img[y0:y1, x0:x1] = (image[y0:y1, xa:xa + 1].astype(np.float32) * (1 - g)
                                         + image[y0:y1, xz:xz + 1].astype(np.float32) * g).astype(np.uint8)
    P, reg, U, V = _carried(P, reg, U, V)
    # its ground under the real ground; what rises, risen from there
    # (eased down to the ground over the first RELIEF_RISE_M, not cut off there: the foot of an upright face would
    # be so many points at one spot)
    z = np.maximum(P[..., 2], 0.0)
    P[..., 2] = np.where(z < RELIEF_RISE_M, z * z / RELIEF_RISE_M, z) - LAND_UNDER_M
    R = np.linalg.norm(P - eye, axis=-1)
    q = reg[:-1, :-1] & reg[1:, :-1] & reg[:-1, 1:] & reg[1:, 1:]
    four = np.stack([R[:-1, :-1], R[1:, :-1], R[:-1, 1:], R[1:, 1:]])
    q &= four.max(0) < LAND_JUMP * np.maximum(four.min(0), 1e-3)
    q = ndi.binary_opening(q, structure=np.ones((2, 2), bool))
    lab, n_ = ndi.label(q)
    if not n_:
        return None
    sizes = np.asarray(ndi.sum(q, lab, range(1, n_ + 1)))
    q = np.isin(lab, 1 + np.nonzero(sizes >= RELIEF_LEAST)[0])
    if not q.any():
        return None
    front_f, _ = _front(q)
    Pf = P.reshape(-1, 3)
    F = to_set(Pf)
    B = to_set(eye + (Pf - eye) * LAND_BACK_K)
    return (*_closed(F, B, front_f, U, V), img, has)


def write_relief(out: Path, vtx, faces, uv, image: np.ndarray) -> dict:
    """The form's .glb (asset frame: z up, its front toward -y, standing on z 0, the middle of its footprint at the
    origin; written y-up) with the place's picture as its texture. -> where it stands in the set's frame, its size."""
    import trimesh
    from PIL import Image
    lo, hi = vtx.min(0), vtx.max(0)
    mid = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    v = vtx - mid
    im = Image.fromarray(image)
    im = im.resize((min(2048, im.width), int(im.height * min(2048, im.width) / im.width)), Image.LANCZOS)
    tex = Image.fromarray(np.concatenate([np.asarray(im), np.asarray(im)], 1))
    yup = np.stack([v[:, 0], v[:, 2], -v[:, 1]], -1)
    trimesh.Trimesh(yup, faces, visual=trimesh.visual.TextureVisuals(uv=uv, image=tex), process=False).export(out)
    return {"glb": str(out), "xy": [round(float(mid[0]), 3), round(float(mid[1]), 3)],
            "size_m": [round(float(x), 3) for x in hi - lo], "faces": int(len(faces)), "z0": round(float(lo[2]), 4)}


def room_from(pts, mask, intr, boxes: dict, kinds_size: dict, keep: dict | None = None) -> dict:
    """The room and its things from the picture's points (see the module's words). boxes: {key: (box, kind)}.
    keep: filled with the frame the room was measured in (for relief())."""
    n, d, fit = fit_floor(pts, mask)
    axes = frame(n, d)
    q = to_room(pts[mask], axes, d)
    q = q[np.isfinite(q).all(1)]
    h_, w_ = pts.shape[:2]
    stand = q[(q[:, 2] > STAND[0]) & (q[:, 2] < STAND[1])]
    cols = to_room(pts[:, int(w_ * 0.25):int(w_ * 0.75)][mask[:, int(w_ * 0.25):int(w_ * 0.75)]], axes, d)
    cols = cols[(cols[:, 2] > STAND[0]) & (cols[:, 2] < STAND[1])]
    back = float(np.percentile(cols[:, 1], 92))
    left, right = float(np.percentile(stand[:, 0], 3)), float(np.percentile(stand[:, 0], 97))
    ceil = float(np.clip(np.percentile(q[:, 2], 97), 2.2, 9.0))
    things, hung, raised = {}, {}, {}
    for key, (b, kind) in boxes.items():
        x0, y0, x1, y1 = b
        yb = y1 - 0.01
        foot = on_floor(ray(intr, (x0 + x1) / 2, yb), n, d)
        fl, fr = on_floor(ray(intr, x0, yb), n, d), on_floor(ray(intr, x1, yb), n, d)
        if foot is None or fl is None or fr is None:
            continue
        fq, lq, rq = (to_room(v[None], axes, d)[0] for v in (foot, fl, fr))
        # the thing's own points (the middle of its box): where it really is. Far nearer than its foot on the floor,
        # it does not stand there (a rack on a wall measured as a 1.9 m block outside the room, 11-17): not a thing
        # to stand on the floor of the set
        r0, r1 = int(h_ * (y0 + 0.25 * (y1 - y0))), max(int(h_ * (y0 + 0.75 * (y1 - y0))), int(h_ * y0) + 1)
        c0, c1 = int(w_ * (x0 + 0.25 * (x1 - x0))), max(int(w_ * (x0 + 0.75 * (x1 - x0))), int(w_ * x0) + 1)
        own = to_room(pts[r0:r1, c0:c1][mask[r0:r1, c0:c1]], axes, d)
        own = own[np.isfinite(own).all(1)]
        if len(own) and fq[1] - float(np.median(own[:, 1])) > OFF_FLOOR_M:
            # where its box's bottom really is, at its own distance: a little above the floor, it stands on a raised
            # part of it (a seat on a dais was taken for a thing hanging and left out of the room, 11-19): placed
            # there, on the floor. Higher, or with no bottom to find, it hangs
            bv = ray(intr, (x0 + x1) / 2, yb)
            yo = float(np.median(own[:, 1]))
            den = float(bv @ axes[1])
            bq = to_room((bv * (yo / den))[None], axes, d)[0] if abs(den) > 1e-6 else None
            if bq is None or not (0.0 < float(bq[2]) <= RAISED_MOST_M):
                hung[key] = {"kind": kind, "at_m": round(yo, 3)}
                if bq is not None and float(bq[2]) > RAISED_MOST_M:
                    # where it hangs: how high its lowest part is, how wide and tall it is at its own distance
                    tv_ = ray(intr, (x0 + x1) / 2, y0)
                    dn_ = float(tv_ @ axes[1])
                    tq_ = to_room((tv_ * (yo / dn_))[None], axes, d)[0] if abs(dn_) > 1e-6 else None
                    k_ = yo / max(1e-6, float(fq[1]))
                    hung[key].update(hangs_m=round(float(bq[2]), 3), x=round(float(bq[0]), 3),
                                     width_m=round(float(abs(rq[0] - lq[0])) * k_, 3),
                                     height_m=round(max(0.1, float(tq_[2] - bq[2])) if tq_ is not None else 0.3, 3))
                continue
            k = yo / max(1e-6, float(fq[1]))               # (its width, seen at its own distance and not the floor's)
            fq, lq, rq = bq, np.array([bq[0] - abs(rq[0] - lq[0]) * k / 2, yo, 0.0]), np.array([bq[0] + abs(rq[0] - lq[0]) * k / 2, yo, 0.0])
            foot = bv * (yo / den)
            raised[key] = round(float(bq[2]), 3)
        width = max(0.2, float(abs(rq[0] - lq[0])))
        # its top: the ray through the box's top middle, at the depth of its front foot
        tv = ray(intr, (x0 + x1) / 2, y0)
        tq = to_room((tv * ((foot @ axes[1]) / max(1e-6, tv @ axes[1])))[None], axes, d)[0]
        height = max(0.2, float(tq[2]) - raised.get(key, 0.0))
        ks = kinds_size.get(kind) or [width, width * 0.6, height]
        depth = max(0.2, width * float(ks[1]) / max(1e-6, float(ks[0])))
        things[key] = {"front": [float(fq[0]), float(fq[1])], "width": width, "depth": depth, "height": height,
                       "kind": kind}
    # the scale: a door seen in the picture stands DOOR_M high
    doors = [t["height"] for t in things.values() if t["kind"] == "door"]
    scale = float(np.clip(DOOR_M / doors[0], *SCALE_CLAMP)) if doors else 1.0
    front = -max(FRONT_M, 0.2 * back)
    W, D, H = (right - left) * scale, (back - front) * scale, ceil * scale
    cx, cy = (left + right) / 2, (front + back) / 2
    out = {}
    for key, t in things.items():
        w, dp, hh = t["width"] * scale, t["depth"] * scale, t["height"] * scale
        x, y = (t["front"][0] - cx) * scale, (t["front"][1] - cy) * scale + dp / 2
        against, facing = None, [0.0, -1.0]            # (a free thing faces the eye's side of the room)
        y = min(y, D / 2 - dp / 2 - 0.02)
        if D / 2 - (y + dp / 2) < AGAINST_M:
            against, facing, y = "north", [0.0, -1.0], D / 2 - dp / 2 - 0.02
        elif x - w / 2 < -W / 2 + AGAINST_M:
            against, facing, x = "west", [1.0, 0.0], -W / 2 + dp / 2 + 0.02
        elif x + w / 2 > W / 2 - AGAINST_M:
            against, facing, x = "east", [-1.0, 0.0], W / 2 - dp / 2 - 0.02
        out[key] = {"xy": [round(x, 3), round(y, 3)], "size_m": [round(w, 3), round(dp, 3), round(hh, 3)],
                    "against": against, "facing": facing, "kind": t["kind"],
                    **({"raised_m": round(raised[key] * scale, 3)} if key in raised else {})}
    for key, h in hung.items():                  # (a hung thing's measures in the room's own metres and middle)
        if "hangs_m" in h:
            x_ = (h.pop("x") - cx) * scale
            y_ = (h["at_m"] - cy) * scale
            near = min((("north", D / 2 - y_), ("west", x_ + W / 2), ("east", W / 2 - x_)), key=lambda t: t[1])[0]
            h.update(hangs_m=round(h["hangs_m"] * scale, 3), wall=near, xy=[round(x_, 3), round(y_, 3)],
                     size_m=[round(h.pop("width_m") * scale, 3), 0.15, round(h.pop("height_m") * scale, 3)])
    stand_before(out)
    if keep is not None:
        keep.update(axes=axes, d=d, cx=cx, cy=cy, scale=scale)
    return {"room_m": [round(W, 2), round(D, 2), round(H, 2)],
            "eye": [round(-cx * scale, 3), round(-cy * scale, 3), round(d * scale, 3)],
            "things": out, "off_floor": hung, "scale": round(scale, 3), "floor_fit": round(fit, 3)}


def land_form_key(concept: dict) -> str | None:
    """The thing of the picture that is the land's own main form: the spec's piece the concept reader found it in,
    or the form it saw itself; else a landmark of the land itself (the largest)."""
    sp = concept.get("spec") or {}
    for pid, v in sp.items():
        if v.get("land_form") and v.get("box"):
            return f"spec:{pid}"
    for i, sn in enumerate(concept.get("seen") or []):
        if sn.get("group") and sn.get("natural") and sn.get("box"):
            return f"seen:{i}"
    kinds = concept.get("_spec_kinds") or {}
    nat = [(pid, v) for pid, v in sp.items() if v.get("natural") and v.get("box") and kinds.get(pid) == "landmark"]
    if nat:
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])  # noqa: E731
        return "spec:" + max(nat, key=lambda t: area(t[1]["box"]))[0]
    return None


def boxes_of(concept: dict) -> dict:
    out = {}
    for pid, v in (concept.get("spec") or {}).items():
        if v.get("box"):
            out[f"spec:{pid}"] = (v["box"], v.get("kind"))
    for i, sn in enumerate(concept.get("seen") or []):
        if sn.get("box") and sn.get("kind"):
            out[f"seen:{i}"] = (sn["box"], sn["kind"])
    return out


def main():
    from PIL import Image
    img_p, con_p, out_p = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    concept = json.loads(con_p.read_text())
    kinds = json.loads((HERE.parent / "places" / "piece_kinds.json").read_text())["kinds"]
    spec_kinds = concept.get("_spec_kinds") or {}
    bx = {k: (b, kd or spec_kinds.get(k.split(":", 1)[1])) for k, (b, kd) in boxes_of(concept).items()}
    image = np.asarray(Image.open(img_p).convert("RGB"))
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pts, mask = infer(image, dev)
    import utils3d  # noqa: F401  (the model's own package: the intrinsics it estimates)
    from moge.model.v2 import MoGeModel  # noqa: F401
    h, w = pts.shape[:2]
    # normalised intrinsics from the points themselves (pixels whose point is known: x/z, y/z against u, v)
    v_, u_ = np.mgrid[0:h, 0:w]
    ok = mask & (pts[..., 2] > 0.1)
    xz, yz = pts[..., 0][ok] / pts[..., 2][ok], pts[..., 1][ok] / pts[..., 2][ok]
    A = np.stack([xz, np.ones_like(xz)], 1)
    fxi, cxi = np.linalg.lstsq(A, (u_[ok] + 0.5) / w, rcond=None)[0]
    fyi, cyi = np.linalg.lstsq(np.stack([yz, np.ones_like(yz)], 1), (v_[ok] + 0.5) / h, rcond=None)[0]
    intr = np.array([[fxi, 0, cxi], [0, fyi, cyi], [0, 0, 1]])
    fr = {}
    res = room_from(pts, mask, intr, bx, {k: v["size_m"] for k, v in kinds.items()}, fr)
    # the land's own form of a place in the open (scenes/concept.py: a piece of the land itself), made from the
    # picture's depth where it stands
    lf = land_form_key(concept) if concept.get("space") == "outdoors" else None
    lf_reg, lf_done = {}, None
    if lf and lf in res["things"] and fr:
        k_, off = fr["scale"], np.array([fr["cx"], fr["cy"], 0.0])
        try:
            got = relief(pts, mask, fr["axes"], fr["d"], bx[lf][0], lambda q: (q - off) * k_, lf_reg)
        except ImportError as e:
            got = None
            print(f"no relief: {e}", file=sys.stderr)
        if got:
            rl = write_relief(out_p.with_name(out_p.stem.replace("_depth", "") + "_relief.glb"), *got, image)
            res["relief"] = dict(rl, of=lf)
            res["things"][lf].update(xy=rl["xy"], size_m=rl["size_m"], against=None)
            lf_done = lf
        else:
            lf_reg = {}
    if concept.get("space") == "outdoors" and fr:
        # all the land the picture shows, for every place in the open: its scenery, standing where the picture has it
        k_, off = fr["scale"], np.array([fr["cx"], fr["cy"], 0.0])
        try:
            got = land(pts, mask, fr["axes"], fr["d"], lambda q: (q - off) * k_,
                       {k: v for k, v in res["things"].items() if k != lf_done}, {k: v[0] for k, v in bx.items()},
                       image, keep_out=lf_reg.get("reg"))
        except ImportError as e:
            got = None
            print(f"no land: {e}", file=sys.stderr)
        if got:
            ld = write_relief(out_p.with_name(out_p.stem.replace("_depth", "") + "_land.glb"), *got[:3], got[3])
            # (its lowest point, the back of its ground, is under the set's ground by under_m)
            res["land"] = dict(ld, has=got[4], under_m=round(-ld["z0"], 4))
    # how wide the picture sees (degrees): a place in the open's panorama is made from the picture, and the land
    # built from the picture itself stands for that much of it
    res["hfov_deg"] = round(float(np.degrees(2 * np.arctan(0.5 / max(abs(fxi), 1e-6)))), 1)
    out_p.write_text(json.dumps(res, indent=1))
    print(json.dumps({"room_m": res["room_m"], "things": len(res["things"]), "scale": res["scale"],
                      "land": (res.get("land") or {}).get("faces")}))


if __name__ == "__main__":
    main()
