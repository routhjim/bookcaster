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
A made floor in the open (no walls: a deck, a pier, a rooftop) is measured by its own floor instead: its sides where
the points lying on the floor plane end (floor_span), kept within reason (floor_bounds).
Its result has "land" too (stage_land, "stage": true): what the picture shows standing on and round the floor, one
solid in the room's own frame with the picture as its texture, the loose things on the floor taken out of it.

OUT.json: {"room_m": [W, D, H], "eye": [x, y, z], "things": {"spec:<id>" | "seen:<i>": {"xy", "size_m", "against",
"facing", "raised_m" (it stands on a raised part of the floor, this high)}}, "off_floor": {key: {"kind"}} (things the
picture shows hanging: not placed),
"scale", "floor_fit"} (set frame: the room's middle at 0, y toward the back wall).
A room's result has "faces" too: the picture on the room's own walls, floor and roof where it sees them (room_faces),
each a PNG beside OUT.json.
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
# A made floor in the open (a deck, a pier, a rooftop: scenes/concept.py "ground": "built") has no walls to end it:
# it ends where the floor itself ends. Measured as a room is, by everything at standing height across the whole
# picture, one came out 38 m wide and 15 m deep (what stood beyond its edge was taken for its walls) and the people
# stood on a vast empty floor, 11-27
FLOOR_ON_M = 0.15            # a point of the picture this near the fitted floor plane lies on the floor
FLOOR_CELL_M = 0.5           # the floor's extent is read in cells of the plane this wide (each cell counts once: near
FLOOR_CELL_LEAST = 4         # the eye a cell holds far more of the picture's points), each with this many points
FLOOR_CELLS_LEAST = 12       # fewer cells than this: no floor to read; where the points at standing height end
FLOOR_CELL_GAP = 2           # the floor is the cells joined to those before the eye, across gaps under this many cells
#                              (another floor at the same height beyond a gap, the foot of a far wall: not this floor)
OPEN_FLOOR_ROUND_M = 6.0     # a made floor reaches no farther than this beyond the things on it and the eye (as the
#                              stage of open land does, sets/place_build.py OPEN_ROUND_M): what lies beyond is no
#                              place anyone is staged, and the place's world is drawn round the floor's edge
OPEN_FLOOR_M = (3.0, 24.0)   # a made floor's side, least and most, each side on its own (a pier is long and narrow:
#                              no square is forced). Least: two people abreast and a camera beside them. Most: the camera's
#                              own light reaches 25 m (scenes/ue/scene_job.py FILL_REACH_M) and a person farther off
#                              than that is under a twentieth of a wide frame's height
SCALE_BY_M = 1.5             # a made floor's scale is corrected by the judged real heights of its things (those this
SCALE_VOICES = 3             # tall or taller, a door at DOOR_M among them) only when at least this many speak, by
#                              the middle one: a single misjudged thing once shrank a whole place to 0.6 of itself


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


def floor_span(q: np.ndarray) -> tuple | None:
    """How far the floor itself reaches (room frame, the picture's metres): the points lying on the fitted floor
    plane before the eye, counted by cells of the plane; of those, the cells joined to the one that holds most of
    the picture (the floor at the eye's feet); left and right from the 2nd to the 98th hundredth of their x, the
    back at the 98th of their y (a ragged edge is not the floor's width).
    -> (left, right, back), or None when too little of the floor is seen."""
    on = q[(np.abs(q[:, 2]) <= FLOOR_ON_M) & (q[:, 1] > 0.0)]
    if not len(on):
        return None
    ij, n = np.unique(np.floor(on[:, :2] / FLOOR_CELL_M).astype(int), axis=0, return_counts=True)
    ij, n = ij[n >= FLOOR_CELL_LEAST], n[n >= FLOOR_CELL_LEAST]
    if len(ij) < FLOOR_CELLS_LEAST:
        return None
    left_, todo, got = {(int(a), int(b)) for a, b in ij}, [tuple(int(v) for v in ij[int(np.argmax(n))])], []
    left_.discard(todo[0])
    near = [(a, b) for a in range(-FLOOR_CELL_GAP, FLOOR_CELL_GAP + 1) for b in range(-FLOOR_CELL_GAP, FLOOR_CELL_GAP + 1)]
    while todo:
        c = todo.pop()
        got.append(c)
        for a, b in near:
            m = (c[0] + a, c[1] + b)
            if m in left_:
                left_.discard(m)
                todo.append(m)
    ij = np.array(got)
    if len(ij) < FLOOR_CELLS_LEAST:
        return None
    return (float(np.percentile(ij[:, 0], 2)) * FLOOR_CELL_M, float(np.percentile(ij[:, 0], 98) + 1) * FLOOR_CELL_M,
            float(np.percentile(ij[:, 1], 98) + 1) * FLOOR_CELL_M)


def floor_bounds(left: float, right: float, back: float, feet: list, scale: float) -> tuple:
    """A made floor's measured sides kept believable (room frame, the picture's metres; the eye at 0, 0): no farther
    than OPEN_FLOOR_ROUND_M beyond the things standing on it (feet: each one's front foot) and the eye, each side
    within OPEN_FLOOR_M, the eye on it. -> (left, right, front, back)."""
    rnd, least, most = OPEN_FLOOR_ROUND_M / scale, OPEN_FLOOR_M[0] / scale, OPEN_FLOOR_M[1] / scale
    xs, ys = [0.0] + [f[0] for f in feet], [0.0] + [f[1] for f in feet]
    if feet:
        left, right, back = max(left, min(xs) - rnd), min(right, max(xs) + rnd), min(back, max(ys) + rnd)
    left, right = min(left, -least / 2), max(right, least / 2)
    if right - left > most:                  # (the stretch about its things and the eye is the one kept)
        mid = min(max((min(xs) + max(xs)) / 2, left + most / 2), right - most / 2)
        left, right = mid - most / 2, mid + most / 2
    front = -max(FRONT_M, 0.2 * back)
    back = min(max(back, front + least), front + most)
    return left, right, front, back


def things_scale(things: dict, real_m: dict, door: float) -> float:
    """A made floor's scale from how tall its things really are (real_m: {key: the judge's height in metres}; a
    door stands DOOR_M): the middle one of their ratios, within SCALE_CLAMP, when at least SCALE_VOICES of them
    speak; with fewer, the door's own (as a room's)."""
    rs = [DOOR_M / t["height"] if t["kind"] == "door" else float(real_m[k]) / t["height"] for k, t in things.items()
          if t["height"] > 0.2 and (t["kind"] == "door" or float(real_m.get(k) or 0.0) >= SCALE_BY_M)]
    if len(rs) < SCALE_VOICES:
        return door
    return float(np.clip(np.median(rs), *SCALE_CLAMP))


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


def _carried(P, reg, U, V, steep_only: bool = False) -> tuple:
    """What the frame's edges cut is carried on past them: up from the top row, sideways from the end columns (the
    picture mirrored there). steep_only: carried up only where what the top edge cuts stands steeper than it lies
    (a rock face, a trunk): a grass slope that ran out of the top of its picture was carried up as a wall of
    grass 8 m tall (11-19)."""
    h, w = reg.shape
    ku, ks = int(RELIEF_ON_M[0] / RELIEF_ON_STEP_M), int(RELIEF_ON_M[1] / RELIEF_ON_STEP_M)
    top = reg[0].copy()
    if steep_only and h > 3:
        t = P[0] - P[3]
        top &= reg[3] & (t[:, 2] > np.hypot(t[:, 0], t[:, 1]))
    if top.any():
        k = np.arange(ku, 0, -1)[:, None]
        up = np.repeat(P[:1], ku, 0) + np.stack([0 * k, 0 * k, k * RELIEF_ON_STEP_M], -1) * np.ones((1, w, 1))
        P, reg = np.concatenate([up, P]), np.concatenate([np.repeat(top[None], ku, 0), reg])
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
    the left half of a picture twice as wide, the back on its right). A side, from its front edge to the back, is
    all of the picture's one point at that edge: its front edge's points are its own (the same places again, with
    the back's uv), not the front's.

    Why its own points: a side that ran from the front's point (the picture's left half) to the back's (its right
    half) had half the width of the doubled picture drawn across it, and every edge of every thing in a picture's
    land, seen from a side, was a band of the whole picture smeared lengthways (11-27)."""
    n0 = len(F)
    back_f = front_f[:, ::-1] + n0
    ed = np.concatenate([front_f[:, [0, 1]], front_f[:, [1, 2]], front_f[:, [2, 0]]])
    key = np.sort(ed, 1)
    _, inv, cnt = np.unique(key[:, 0] * (2 * n0) + key[:, 1], return_inverse=True, return_counts=True)
    rim = ed[cnt[inv] == 1]
    u_, v_ = rim[:, 0], rim[:, 1]
    side_f = np.concatenate([np.stack([u_ + 2 * n0, u_ + n0, v_ + n0], -1),
                             np.stack([u_ + 2 * n0, v_ + n0, v_ + 2 * n0], -1)])
    vtx = np.concatenate([F, B, F])
    faces = np.concatenate([front_f, back_f, side_f])
    uv = np.stack([U.reshape(-1) / 2, 1.0 - V.reshape(-1)], -1)
    uv = np.concatenate([uv, uv + [0.5, 0.0], uv + [0.5, 0.0]])
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


LAND_END_ON_DEG = 78.0      # a standing face turned further than this from facing the picture's eye is seen end on:
LAND_STANDS = 0.5           # ... standing: its upward share under this (ground and slopes lie, and are seen so)


def _end_on(P, eye) -> np.ndarray:
    """The squares of the grid P that stand and are seen end on from the picture's eye (LAND_END_ON_DEG): the gap
    between a thing and what is behind it, or a wall running straight away. One row of the picture is all such a
    face has, drawn out along its whole depth.

    Why: neighbours up to LAND_JUMP apart were joined, so at 10 m a face 3 m deep had one point of the picture
    across it, and from any camera off the picture's own line every edge of every near thing was a band of
    streaks metres long (11-27)."""
    a, b, c, e = P[:-1, :-1], P[1:, :-1], P[:-1, 1:], P[1:, 1:]
    n = np.cross(e - a, c - b)
    ln = np.maximum(np.linalg.norm(n, axis=-1), 1e-12)
    ray = (a + b + c + e) / 4 - eye
    facing = np.abs((n * ray).sum(-1)) / (ln * np.maximum(np.linalg.norm(ray, axis=-1), 1e-9))
    return (np.abs(n[..., 2]) / ln < LAND_STANDS) & (facing < math.cos(math.radians(LAND_END_ON_DEG)))


LAND_LEVEL_M = (25.0, 50.0)  # the picture's ground is levelled this near its eye; left as it lies beyond the second
LAND_LIES = 0.8             # a surface whose upward share is at least this lies (ground); less, it stands
LAND_LEVEL_CELLS = (5.0, 20.0)
# A made floor in the open that keeps its picture's own structure (stage_land): the picture's land is built round
# its real floor too. Its floor is level as fitted (nothing is levelled: a raised part of it, a roof, a step would
# be pressed flat), and a thing of the picture that takes up more of it than this is no loose thing standing on the
# floor but the place's own structure (rigging over a whole deck, a canopy, a frame): left in the land, not built
STAGE_CUT_BOX = 0.2


def _ground_rise(P, reg) -> np.ndarray:
    """How high the picture's own ground lies over the fitted plane at each point of the grid: the height of the
    surfaces that lie (not those that stand: a rock face, a trunk), smoothed, carried under what stands on it."""
    import scipy.ndimage as ndi
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    up = np.abs(n[..., 2]) / np.maximum(np.linalg.norm(n, axis=-1), 1e-9)
    lies = ndi.binary_erosion(reg, iterations=1) & (up >= LAND_LIES)
    z = np.maximum(P[..., 2], 0.0)
    g = np.zeros(z.shape)
    done = np.zeros(z.shape, bool)
    for sg in LAND_LEVEL_CELLS:
        wt = ndi.gaussian_filter(lies.astype(np.float64), sg)
        val = ndi.gaussian_filter(np.where(lies, z, 0.0), sg) / np.maximum(wt, 1e-9)
        ok = ~done & (wt > 0.02)
        g[ok], done[ok] = val[ok], True
    return np.where(reg, np.minimum(g, z), 0.0)


def land(pts, mask, axes, d: float, to_set, things: dict, boxes: dict, image: np.ndarray,
         keep_out: np.ndarray | None = None, level: bool = True, cut_most: float | None = None) -> tuple | None:
    """All the land a place in the open's picture shows, as one solid standing where the picture has it: its ground
    (just under the set's real ground), and every bank, slope, rock, tree and far hill rising from it, the front the
    picture's own surface. What the frame cuts is carried on. The things that are built as pieces of the set
    (things: their place and size in the set's frame; boxes: their part of the picture) are taken out of it, the
    land behind each closed over; keep_out is a part of the grid left out whole (the land's own form, a solid of
    its own). level False: the picture's ground is left as it lies (a made floor's, stage_land); cut_most: a thing
    the picture shows that takes up more of it than this share is left in the land.
    -> (points in the set's frame, faces, uv, the picture with the pieces painted over, the keys of the
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
        if key.startswith("seen:") and cut_most is not None and (b[2] - b[0]) * (b[3] - b[1]) > cut_most:
            has.append(key)                    # (it takes up much of the picture: the place's own structure)
            continue
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
    # the picture's own ground, where it does not lie on the fitted plane (a hillside, a bank's slope), is taken off
    # everything near the eye, fading out farther off: people stand on the set's level ground, and land that rose
    # with the hill stood over their heads (a man fighting on a hillside was under the hill, his sword's point out
    # of it, 11-19)
    if level:
        g = _ground_rise(P, reg)
        rr = np.hypot(P[..., 0] - eye[0], P[..., 1] - eye[1])
        P[..., 2] -= g * (1.0 - np.clip((rr - LAND_LEVEL_M[0]) / (LAND_LEVEL_M[1] - LAND_LEVEL_M[0]), 0.0, 1.0))
    P, reg, U, V = _carried(P, reg, U, V, steep_only=True)
    # its ground under the real ground; what rises, risen from there
    # (eased down to the ground over the first RELIEF_RISE_M, not cut off there: the foot of an upright face would
    # be so many points at one spot)
    z = np.maximum(P[..., 2], 0.0)
    P[..., 2] = np.where(z < RELIEF_RISE_M, z * z / RELIEF_RISE_M, z) - LAND_UNDER_M
    R = np.linalg.norm(P - eye, axis=-1)
    q = reg[:-1, :-1] & reg[1:, :-1] & reg[:-1, 1:] & reg[1:, 1:]
    four = np.stack([R[:-1, :-1], R[1:, :-1], R[:-1, 1:], R[1:, 1:]])
    q &= four.max(0) < LAND_JUMP * np.maximum(four.min(0), 1e-3)
    q &= ~_end_on(P, eye)
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


def stage_land(pts, mask, fr: dict, things: dict, boxes: dict, image: np.ndarray) -> tuple | None:
    """The picture's own land round a made floor in the open (fr: the frame its floor was measured in, room_from's
    keep; the set's frame is that room's): everything the picture shows standing on and round the floor, as land()
    builds it for open land, with the floor left level as it was fitted and what takes up much of the picture left
    in it (STAGE_CUT_BOX). The loose things on the floor are taken out, as from open land: each is a piece of the
    set. -> as land().

    Why: a made floor was built as a bare floor inside four plain low walls with a world drawn from words round it,
    and nothing of what its picture shows standing on and round it: its posts, its house, its railings (11-27)."""
    k_, off = fr["scale"], np.array([fr["cx"], fr["cy"], 0.0])
    return land(pts, mask, fr["axes"], fr["d"], lambda q: (q - off) * k_, things, boxes, image,
                level=False, cut_most=STAGE_CUT_BOX)


TOPS_CELL_M = 1.0           # how high the land stands is kept in cells this wide
TOPS_CLEAR_M = 2.5          # what the land has over a cell of seen ground no lower than this is overhead, not on it


def tops_of(vtx, uv) -> dict:
    """How high the land's front stands over each cell of the ground (the set's frame): {"cell_m", "x0", "y0",
    "z": rows from y0 up, each from x0 right}; the staging keeps people off what rises and cameras out of it.
    A cell whose ground the picture shows, with nothing of the land over it lower than TOPS_CLEAR_M, stands at 0:
    what is there hangs or spans overhead (a line of lamps, rigging, a bough, an arch's top), and people walk and
    cameras stand under it.

    Why: the highest point over a cell was how high the land stood there, so under every string of lamps and every
    rope the picture showed over a floor nobody was stood and no camera placed: most of the floor was shut (11-27)."""
    f = vtx[uv[:, 0] <= 0.5]
    x0, y0 = float(np.floor(f[:, 0].min())), float(np.floor(f[:, 1].min()))
    j = ((f[:, 0] - x0) / TOPS_CELL_M).astype(int)
    i = ((f[:, 1] - y0) / TOPS_CELL_M).astype(int)
    z = np.zeros((i.max() + 1, j.max() + 1))
    zf = np.maximum(f[:, 2], 0.0)
    np.maximum.at(z, (i, j), zf)
    low = np.full(z.shape, np.inf)                     # the lowest of what rises over the cell
    up = zf > RELIEF_RISE_M
    np.minimum.at(low, (i[up], j[up]), zf[up])
    ground = np.zeros(z.shape, bool)                   # the picture shows the cell's own ground
    ground[i[~up], j[~up]] = True
    z[ground & (low >= TOPS_CLEAR_M)] = 0.0
    return {"cell_m": TOPS_CELL_M, "x0": x0, "y0": y0, "z": np.round(z, 1).tolist()}


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


def room_from(pts, mask, intr, boxes: dict, kinds_size: dict, keep: dict | None = None, open_top: bool = False,
              real_m: dict | None = None) -> dict:
    """The room and its things from the picture's points (see the module's words). boxes: {key: (box, kind)}.
    keep: filled with the frame the room was measured in (for relief()). open_top: a made floor in the open: its
    sides are where the floor itself ends (floor_span), kept within reason (floor_bounds), its scale its things'
    (things_scale; real_m: {key: how tall the judge says the thing really is})."""
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
    if open_top:
        scale = things_scale(things, real_m or {}, scale)
        left, right, back = floor_span(q) or (left, right, back)
        left, right, front, back = floor_bounds(left, right, back, [t["front"] for t in things.values()], scale)
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
        keep.update(axes=axes, d=d, cx=cx, cy=cy, scale=scale, W=W, D=D, H=H)
    return {"room_m": [round(W, 2), round(D, 2), round(H, 2)],
            "eye": [round(-cx * scale, 3), round(-cy * scale, 3), round(d * scale, 3)],
            "things": out, "off_floor": hung, "scale": round(scale, 3), "floor_fit": round(fit, 3)}


FACE_PX_PER_M = 128         # a face's picture: this many texels a metre ...
FACE_PX_MOST = 2048         # ... and no more than this along its longer side
FACE_BEFORE_M = {"wall": (0.3, 0.03), "floor": (0.2, 0.0)}   # a picture point this far out from a face into the
#                             room (metres, plus this share of the face's distance from the eye) stands before it: the
#                             face is hidden there, not painted with that thing (a wall, or the floor and the roof).
#                             Measured square to the face, not along the eye's line: a side wall seen end on lies a
#                             little inside the measured box, and along the line that little was a metre and more
FACE_EDGE_PX = 2            # ... and the face is left unknown this many texels round what hides it (its edge's halo)
FACE_LEAST = 0.03           # a face the picture sees less of than this share of is left out


def _faces(W: float, D: float, H: float) -> dict:
    """The room box's faces in the set's frame: {name: (corner at texel (0, 0), along the columns, along the rows,
    columns' length, rows' length, the axes the columns and rows run along, into the room)}. A wall's picture reads as one sees it
    standing in the room facing it (its top row the top of the wall); the floor's and the roof's as seen from above
    with the back wall at the top."""
    return {
        "north": (np.array([-W / 2, D / 2, H]), np.array([1.0, 0, 0]), np.array([0, 0, -1.0]), W, H, ("x", "z"), (0, -1, 0)),
        "west": (np.array([-W / 2, -D / 2, H]), np.array([0, 1.0, 0]), np.array([0, 0, -1.0]), D, H, ("y", "z"), (1, 0, 0)),
        "east": (np.array([W / 2, D / 2, H]), np.array([0, -1.0, 0]), np.array([0, 0, -1.0]), D, H, ("y", "z"), (-1, 0, 0)),
        "south": (np.array([W / 2, -D / 2, H]), np.array([-1.0, 0, 0]), np.array([0, 0, -1.0]), W, H, ("x", "z"), (0, 1, 0)),
        "floor": (np.array([-W / 2, D / 2, 0.0]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), W, D, ("x", "y"), (0, 0, 1)),
        "ceiling": (np.array([-W / 2, D / 2, H]), np.array([1.0, 0, 0]), np.array([0, -1.0, 0]), W, D, ("x", "y"), (0, 0, -1)),
    }


def _bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The picture's colour at pixel coordinates (x, y: pixel centres at whole numbers)."""
    h, w = image.shape[:2]
    x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    im = image.astype(np.float32)
    top = im[y0, x0] * (1 - fx) + im[y0, x1] * fx
    bot = im[y1, x0] * (1 - fx) + im[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _grow(m: np.ndarray, n: int) -> np.ndarray:
    """m grown by n texels every way (no scipy at the test station)."""
    for _ in range(n):
        g = m.copy()
        g[1:] |= m[:-1]
        g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        m = g
    return m


def room_faces(pts, mask, intr, image: np.ndarray, fr: dict) -> list:
    """The picture on the room's own faces: for each face of the measured room box (fr: the frame room_from measured
    it in), every texel of the face's plane is taken through the picture's camera to its pixel and given the
    picture's colour there, where the picture sees that face: the texel in front of the eye and inside the frame, and
    nothing standing before it (the picture's own point at that pixel no farther out into the room from the face
    than FACE_BEFORE_M). A point
    beyond the face, or none (the model's mask: the sky), is seen through it (a window, an arch): painted. Hidden,
    outside the frame or behind the eye, the texel is unknown (alpha 0). -> [(record, RGBA texture cropped to what
    is known)], the record: {"face", "u": [axis, at the left edge, at the right edge], "v": [axis, at the top, at the
    bottom] (set metres), the rectangle as "x_m" / "y_m" / "z_m" ranges, "at_m" (where its plane stands on the third
    axis), "px_per_m", "share" (of the whole face seen), "share_of_rect"}; faces seen less than FACE_LEAST left out.

    Why: a room's walls were one tile of bare wall repeated, and everything the picture showed on them that was not a
    thing to stand on the floor (its windows, arches, hangings, a drape, the roof's beams) was lost (11-20)."""
    axes, d, cx, cy, sc = fr["axes"], fr["d"], fr["cx"], fr["cy"], fr["scale"]
    r_, f_, u_ = (np.asarray(a, float) for a in axes)
    W, D, H = fr["W"], fr["D"], fr["H"]
    hp, wp = pts.shape[:2]
    hi, wi = image.shape[:2]
    fx, fy, ci, cj = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    out = []
    for name, (o, eu, ev, lu, lv, (au, av), n_in) in _faces(W, D, H).items():
        ppm = min(FACE_PX_PER_M, FACE_PX_MOST / max(lu, lv, 1e-6))
        nu, nv = max(1, int(round(lu * ppm))), max(1, int(round(lv * ppm)))
        du, dv = lu / nu, lv / nv
        jj, ii = np.mgrid[0:nv, 0:nu]
        S = o + ((ii + 0.5) * du)[..., None] * eu + ((jj + 0.5) * dv)[..., None] * ev
        q = S / sc + np.array([cx, cy, 0.0])                   # the room's own metres, then the camera's
        p = q[..., :1] * r_ + q[..., 1:2] * f_ + (q[..., 2:3] - d) * u_
        z = p[..., 2]
        ahead = z > 1e-3
        zs = np.where(ahead, z, 1.0)
        un, vn = fx * p[..., 0] / zs + ci, fy * p[..., 1] / zs + cj   # (0..1 of the picture)
        inside = ahead & (un >= 0) & (un < 1) & (vn >= 0) & (vn < 1)
        rp = np.clip((vn * hp).astype(int), 0, hp - 1)
        cp = np.clip((un * wp).astype(int), 0, wp - 1)
        P = pts[rp, cp]
        has = mask[rp, cp] & np.isfinite(P).all(-1)
        Ps = (to_room(np.where(has[..., None], P, 0.0), axes, d) - np.array([cx, cy, 0.0])) * sc
        out_m = (Ps - o) @ np.asarray(n_in, float)          # (how far out into the room the picture's point is)
        tol = FACE_BEFORE_M["wall" if name not in ("floor", "ceiling") else "floor"]
        before = inside & has & (out_m > tol[0] + tol[1] * np.linalg.norm(p, axis=-1) * sc)
        known = inside & ~_grow(before, FACE_EDGE_PX)
        share = float(known.mean())
        if share < FACE_LEAST:
            continue
        rows, cols = np.nonzero(known.any(1))[0], np.nonzero(known.any(0))[0]
        j0, j1, i0, i1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        rgb = _bilinear(image, un[j0:j1, i0:i1] * wi - 0.5, vn[j0:j1, i0:i1] * hi - 0.5)
        k = known[j0:j1, i0:i1]
        tex = np.concatenate([np.where(k[..., None], rgb, 0.0), (k * 255.0)[..., None]], -1)
        tex = np.clip(np.round(tex), 0, 255).astype(np.uint8)
        a, b = o + i0 * du * eu + j0 * dv * ev, o + i1 * du * eu + j1 * dv * ev
        ax = {"x": 0, "y": 1, "z": 2}
        rec = {"face": name, "u": [au, round(float(a[ax[au]]), 3), round(float(b[ax[au]]), 3)],
               "v": [av, round(float(a[ax[av]]), 3), round(float(b[ax[av]]), 3)]}
        for n_ in (au, av):
            rec[f"{n_}_m"] = sorted([round(float(a[ax[n_]]), 3), round(float(b[ax[n_]]), 3)])
        third = ({"x", "y", "z"} - {au, av}).pop()
        rec.update(at_m=round(float(o[ax[third]]), 3), px_per_m=round(1.0 / du, 2), share=round(share, 3),
                   share_of_rect=round(float(k.mean()), 3))
        out.append((rec, tex))
    return out


def write_faces(out_p: Path, got: list) -> list:
    """Each face's picture as a PNG beside OUT.json (<stem>_face_<face>.png) -> the records, each with its "file"."""
    from PIL import Image
    recs = []
    for rec, tex in got:
        f = out_p.with_name(out_p.stem.replace("_depth", "") + f"_face_{rec['face']}.png")
        Image.fromarray(tex).save(f)
        recs.append(dict(rec, file=str(f.resolve())))
    return recs


FACE_RING = ("north", "east", "south", "west")   # facing each wall from inside the room, its right end meets the
#                                                  next one's left end (and the last's the first's)
FACE_BAND_SHARE = 0.5       # a face's seen run: the lines of it seen at least this share as well as its best seen line
FACE_CLEAN_SHARE = 0.85     # ... what is mirrored from it, or carried whole over a floor: those seen this well (a run
#                             mirrored from its half-seen lines carried their holes on; a floor's, the pieces' feet)
FACE_EVEN_M = 1.2           # a floor's or a roof's picture evened out over this (the picture's own light pools and the
#                             shadows of its pieces: the set lights it again, and carried across they were stripes)
FACE_PERIOD_M = 0.8         # a face's rhythm (bays, windows, piers, flags) repeats no oftener than this ...
FACE_PERIOD_R = 0.5         # ... and is taken where the seen run matches itself shifted by it this well (correlation)
FACE_PERIOD_BINS = 8        # (the run is compared in this many bands across it, each band's own brightness)
FACE_TRIM_M = 0.5           # a floor's or a roof's seen run, carried whole: this much off each of its ends first
FACE_FEATHER_M = 0.3        # a wall's seen part fades into what is carried over the rest across this
FACE_FILL_EDGE_M = 1.0      # what a wall's picture lacks above or below what it sees: mirrored from it this far
FACE_WHOLE_PX_MOST = 4096   # a whole face's picture: no more texels than this along its longer side


def face_frame(name: str, W: float, D: float, H: float) -> tuple:
    """A face's whole rectangle as _faces lays its picture out: (u at the left edge, u's sign, its length along u,
    v at the top, v's sign, its length along v), u and v the axes _faces gives it."""
    return {"north": (-W / 2, 1, W, H, -1, H), "south": (W / 2, -1, W, H, -1, H),
            "west": (-D / 2, 1, D, H, -1, H), "east": (D / 2, -1, D, H, -1, H),
            "floor": (-W / 2, 1, W, D / 2, -1, D), "ceiling": (-W / 2, 1, W, D / 2, -1, D)}[name]


def _resize(tex: np.ndarray, w: int, h: int) -> np.ndarray:
    """An RGBA picture (float) at another size, its colour carried by what is known (alpha-weighted)."""
    from PIL import Image
    if tex.shape[1] == w and tex.shape[0] == h:
        return tex
    a = tex[..., 3] / 255.0
    ch = [np.asarray(Image.fromarray(np.ascontiguousarray(x, np.float32), "F").resize((w, h), Image.BILINEAR))
          for x in [tex[..., i] * a for i in range(3)] + [a]]
    al = ch[3]
    rgb = np.stack(ch[:3], -1) / np.maximum(al, 1e-6)[..., None]
    return np.concatenate([np.clip(rgb, 0, 255), (np.where(al > 0.5, 255.0, 0.0))[..., None]], -1).astype(np.float32)


def _paste(rec: dict, tex: np.ndarray, room_m, ppm: float) -> np.ndarray:
    """A face's picture (its record's rectangle) laid into its whole face, ppm texels a metre: RGBA float, alpha 0
    where nothing is known."""
    u0, su, lu, v0, sv, lv = face_frame(rec["face"], *room_m)
    nu, nv = max(1, int(round(lu * ppm))), max(1, int(round(lv * ppm)))
    out = np.zeros((nv, nu, 4), np.float32)
    c0, c1 = sorted([(rec["u"][1] - u0) * su * ppm, (rec["u"][2] - u0) * su * ppm])
    r0, r1 = sorted([(rec["v"][1] - v0) * sv * ppm, (rec["v"][2] - v0) * sv * ppm])
    i0, j0 = int(round(c0)), int(round(r0))
    t = _resize(np.asarray(tex, np.float32), max(1, int(round(c1)) - i0), max(1, int(round(r1)) - j0))
    a0, b0 = max(0, -i0), max(0, -j0)
    a1, b1 = min(t.shape[1], nu - i0), min(t.shape[0], nv - j0)
    if a1 > a0 and b1 > b0:
        out[j0 + b0:j0 + b1, i0 + a0:i0 + a1] = t[b0:b1, a0:a1]
    return out


def _band(known: np.ndarray, axis: int, share: float = FACE_BAND_SHARE):
    """The seen run of a face along axis (1: its columns, 0: its rows): the longest unbroken run of lines each seen at
    least share as well as the best seen one -> (first, past the last) or None."""
    cov = known.mean(axis=1 - axis)
    if cov.max() <= 0:
        return None
    ok = cov >= share * cov.max()
    best, start = None, None
    for i, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if best is None or i - start > best[1] - best[0]:
                best = (start, i)
            start = None
    return best


def _period(tex: np.ndarray, known: np.ndarray, b0: int, b1: int, axis: int, ppm: float):
    """The face's own rhythm along axis within its seen run (texels), or None: the shortest shift (at least
    FACE_PERIOD_M, at most half the run, so that two of it are seen) at which the run matches itself at least
    FACE_PERIOD_R and nearly as well as at any shift."""
    lum = tex[..., :3].mean(-1)
    if axis == 0:
        lum, known = lum.T, known.T
    lum, known = lum[:, b0:b1], known[:, b0:b1]
    n = b1 - b0
    lo, hi = max(2, int(FACE_PERIOD_M * ppm)), n // 2
    if hi <= lo:
        return None
    e = np.linspace(0, lum.shape[0], FACE_PERIOD_BINS + 1).astype(int)
    prof = []
    x = np.arange(n)
    for a, b in zip(e[:-1], e[1:]):
        if b <= a:
            continue
        k = known[a:b]
        cnt = k.sum(0)
        m = np.where(cnt >= 0.3 * (b - a), (lum[a:b] * k).sum(0) / np.maximum(cnt, 1), np.nan)
        ok = np.isfinite(m)
        if ok.sum() > 3:                     # (the light falling off along the run is not its rhythm)
            m = m - np.polyval(np.polyfit(x[ok], m[ok], 1), x)
        prof.append(m)
    prof = np.array(prof)
    corr = np.full(hi + 1, np.nan)
    for lag in range(lo, hi + 1):
        A, B = prof[:, :-lag], prof[:, lag:]
        ok = np.isfinite(A) & np.isfinite(B)
        if ok.sum() < 0.3 * n:
            continue
        a_, b_ = A[ok], B[ok]
        corr[lag] = float((a_ * b_).sum() / np.sqrt((a_ * a_).sum() * (b_ * b_).sum() + 1e-9))
    if not np.isfinite(corr).any() or np.nanmax(corr) < FACE_PERIOD_R:
        return None
    top = np.nanmax(corr)
    for lag in range(lo + 1, hi):            # (a true peak: a match falling away from the least shift is no rhythm)
        c = corr[lag]
        if np.isfinite(c) and c >= max(FACE_PERIOD_R, 0.85 * top) and \
                np.isfinite(corr[lag - 1]) and corr[lag - 1] <= c and not (np.isfinite(corr[lag + 1]) and corr[lag + 1] > c):
            return lag
    return None


def _source(idx: np.ndarray, b0: int, b1: int, period) -> np.ndarray:
    """Where each line along a face (any index, before or past the face too) takes its picture from in the seen run
    [b0, b1): the run's own lines in it; outside it the same place of its rhythm (shifted by whole periods: the
    windows and the stonework keep their spacing and their scale), else the run mirrored at its ends (no hard cut)."""
    idx = np.asarray(idx)
    if period:
        return np.where(idx < b0, b0 + np.mod(idx - b0, period),
                        np.where(idx >= b1, b1 - period + np.mod(idx - b1, period), idx))
    n = b1 - b0
    t = np.mod(idx - b0, 2 * n)
    return b0 + np.where(t < n, t, 2 * n - 1 - t)


def _extend(tex: np.ndarray, axis: int, ppm: float, whole: bool = False) -> tuple:
    """A face's picture carried along axis over what its picture does not see (_source); within the seen run, what
    stood before the face filled from the same place a period or two along. whole: lines outside the run are taken
    from it whole, not only where unknown (a floor's glimpses between the pieces carry their feet and shadows)
    -> (picture, (b0, b1, period) or None)."""
    known = tex[..., 3] > 127
    b = _band(known, axis)
    if not b:
        return tex, None
    b0, b1 = b
    P = _period(tex, known, b0, b1, axis, ppm)
    if not P or whole:                       # (mirrored, or carried whole: only from what is seen well)
        b0, b1 = _band(known, axis, FACE_CLEAN_SHARE) or b
        t = int(FACE_TRIM_M * ppm)
        if whole and b1 - b0 > 4 * t:        # (a floor's seen run is edged by what stands on it)
            b0, b1 = b0 + t, b1 - t
        if P and P > b1 - b0:
            P = None
    L = tex.shape[axis]
    moved = np.take(tex, _source(np.arange(L), b0, b1, P), axis=axis)
    has = moved[..., 3] > 127
    if whole:
        line = np.zeros(L, bool)
        line[:b0], line[b1:] = True, True
        put = (line[None, :] if axis == 1 else line[:, None]) & has
        put |= ~known & has
        out = np.where(put[..., None], moved, tex)
    else:
        put = ~known & has
        out = np.where(put[..., None], moved, tex)
        # (the seen part fades into what is carried over it across FACE_FEATHER_M: no hard cut at the join)
        n = max(1, int(FACE_FEATHER_M * ppm))
        w = np.ones(known.shape, np.float32)
        reached, front = put.copy(), put
        for s in range(1, n + 1):
            g = _grow(front, 1) & known & ~reached
            if not g.any():
                break
            w[g] = s / (n + 1.0)
            reached |= g
            front = g
        soft = known & has & (w < 1)
        out = np.where(soft[..., None], tex * w[..., None] + moved * (1 - w[..., None]), out)
        out[..., 3] = np.where(known | put, 255.0, out[..., 3])
    if P:
        for s in (P, -P, 2 * P, -2 * P):
            if abs(s) >= L:
                continue
            k = out[..., 3] > 127
            sh = np.roll(out, s, axis=axis)
            valid = np.zeros(L, bool)
            if s > 0:
                valid[s:] = True
            else:
                valid[:L + s] = True
            valid = valid[None, :] if axis == 1 else valid[:, None]
            put = ~k & valid & (sh[..., 3] > 127)
            out = np.where(put[..., None], sh, out)
    return out, (b0, b1, P)


def _box_mean(x: np.ndarray, w: np.ndarray, r: int) -> np.ndarray:
    """The w-weighted mean of x over a (2r+1)-square round each texel (summed tables: no scipy at the test station)."""
    def box(a):
        c = np.pad(a, ((r + 1, r), (r + 1, r)), mode="constant").cumsum(0).cumsum(1)
        return c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1]
    return box(x * w) / np.maximum(box(w), 1e-6)


def _even(tex: np.ndarray, ppm: float) -> np.ndarray:
    """A floor's or a roof's picture with its light evened out: each known texel scaled by its seen surroundings'
    brightness (FACE_EVEN_M round it) to the whole seen face's middle brightness; the stones' own grain kept."""
    k = (tex[..., 3] > 127).astype(np.float64)
    if k.sum() < 16:
        return tex
    lum = tex[..., :3].astype(np.float64).mean(-1)
    near = _box_mean(lum, k, max(1, int(FACE_EVEN_M * ppm / 2)))
    f = np.clip(float(np.median(lum[k > 0])) / np.maximum(near, 1.0), 0.33, 3.0)
    out = tex.copy()
    out[..., :3] = np.where(k[..., None] > 0, np.clip(tex[..., :3] * f[..., None], 0, 255), tex[..., :3])
    return out


def _fill_edges(tex: np.ndarray, n: int) -> np.ndarray:
    """What a wall's picture lacks above its topmost and below its lowest seen texel, column by column, mirrored from
    what is seen there, up to n texels."""
    known = tex[..., 3] > 127
    h = known.shape[0]
    has = known.any(0)
    first = np.argmax(known, 0)
    last = h - 1 - np.argmax(known[::-1], 0)
    cols = np.arange(known.shape[1])
    out = tex.copy()
    for d in range(1, n + 1):
        for r, s in ((first - d, first + d - 1), (last + d, last - d + 1)):
            ok = has & (r >= 0) & (r < h) & (s >= 0) & (s < h)
            rr, ss, cc = r[ok], s[ok], cols[ok]
            put = (out[rr, cc, 3] <= 127) & (out[ss, cc, 3] > 127)
            out[rr[put], cc[put]] = out[ss[put], cc[put]]
    return out


def whole_faces(recs: list, room_m) -> dict:
    """The room's whole walls, floor and roof from what its picture sees of them (room_faces' records, each with its
    "file"): {face: (RGBA uint8 picture of the whole face as _faces lays it out, texels a metre, how it was made)}.

    A wall the picture sees: its seen run carried along the rest of it (_extend: by the wall's own rhythm, else
    mirrored at the join), then up and down over what it lacks there (_fill_edges). A wall it never sees (the one
    behind its eye): one neighbour's run continued round the corner over all of it (the one with a rhythm, else the
    one seen more); failing both, the opposite wall mirrored. (Not the opposite wall first: the wall the picture
    faces is the room's own one, what it is drawn for, a hearth, a raised end, an altar; the side walls are its running
    fabric, and continued round a corner they meet the unseen wall without a cut.) The floor and the roof: their light evened out (_even), carried
    across, then along, their lines outside the seen run taken whole. What is still unknown stays alpha 0 (places/generate.py face_fill).
    Every face shares one scale: the coarsest any seen face has (texels are never made up)."""
    from PIL import Image
    seen = {r["face"]: r for r in recs if r.get("face") and r.get("file") and Path(r["file"]).exists()}
    if not seen:
        return {}
    W, D, H = (float(v) for v in room_m)
    ppm = min(float(r.get("px_per_m") or FACE_PX_PER_M) for r in seen.values())
    ppm = min(ppm, FACE_WHOLE_PX_MOST / max(W, D, H))
    out, rule, how, seen_share = {}, {}, {}, {}
    for name, r in seen.items():
        cv = _paste(r, np.asarray(Image.open(r["file"]).convert("RGBA"), np.float32), (W, D, H), ppm)
        seen_share[name] = float((cv[..., 3] > 127).mean())
        if name in FACE_RING:
            cv, rule[name] = _extend(cv, 1, ppm)
            cv = _fill_edges(cv, int(FACE_FILL_EDGE_M * ppm))
            p = (rule[name] or (0, 0, None))[2]
        else:
            cv = _even(cv, ppm)
            cv, ru = _extend(cv, 1, ppm, whole=True)
            cv, rv = _extend(cv, 0, ppm, whole=True)
            p = (ru or (0, 0, None))[2] or (rv or (0, 0, None))[2]
        out[name] = cv
        how[name] = "its picture, " + (f"its rhythm of {p / ppm:.2f} m repeated" if p else "mirrored") + \
            " over what it does not see"
    for i, name in enumerate(FACE_RING):
        if name in out:
            continue
        left, right, opp = FACE_RING[i - 1], FACE_RING[(i + 1) % 4], FACE_RING[(i + 2) % 4]
        nu = max(1, int(round(face_frame(name, W, D, H)[2] * ppm)))
        # one neighbour carried round its corner over the whole wall: the one with a rhythm, else the one seen more
        # (a cut at the other corner reads as the corner; two runs met in the middle of the wall read as a seam)
        cand = [(bool(rule[nb][2]), float(seen_share.get(nb, 0.0)), k, nb) for k, nb in enumerate((left, right))
                if out.get(nb) is not None and rule.get(nb)]
        if cand:
            _, _, k, nb = max(cand)
            Ln = out[nb].shape[1]
            idx = Ln + np.arange(nu) if k == 0 else np.arange(nu) - nu    # (past its right end / before its left)
            cv = out[nb][:, _source(idx, *rule[nb])]
            how[name] = f"the {nb} wall continued round the corner" + \
                (f", its rhythm of {rule[nb][2] / ppm:.2f} m" if rule[nb][2] else ", mirrored")
        elif out.get(opp) is not None:
            cv = _resize(out[opp], nu, out[opp].shape[0])[:, ::-1]
            how[name] = f"the {opp} wall mirrored"
        else:
            continue
        out[name] = _fill_edges(cv, int(FACE_FILL_EDGE_M * ppm))
    return {k: (np.clip(np.round(v), 0, 255).astype(np.uint8), round(ppm, 2), how[k]) for k, v in out.items()}


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
    # (a made floor in the open: measured by its own floor; its things' real heights, where the judge gave them)
    real = {f"spec:{pid}": v["real_m"][1] for pid, v in (concept.get("spec") or {}).items()
            if v.get("real_m") and not v.get("natural")}
    real.update({f"seen:{i}": sn["real_m"][1] for i, sn in enumerate(concept.get("seen") or [])
                 if sn.get("real_m") and not sn.get("natural")})
    res = room_from(pts, mask, intr, bx, {k: v["size_m"] for k, v in kinds.items()}, fr,
                    open_top=concept.get("space") == "outdoors" and concept.get("ground") == "built", real_m=real)
    # the land's own form of a place in the open (scenes/concept.py: a piece of the land itself), made from the
    # picture's depth where it stands
    # (11-26: a place in the open whose ground is a made floor, scenes/concept.py "ground": "built", is measured as
    # a room is: its floor's size and each thing's place on it; 11-27: and the land its picture shows round that
    # floor is built as open land's is, stage_land)
    open_land = concept.get("space") == "outdoors" and concept.get("ground") != "built"
    made = concept.get("space") == "outdoors" and concept.get("ground") == "built"
    lf = land_form_key(concept) if open_land else None
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
    if open_land and fr:
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
            res["land"] = dict(ld, has=got[4], under_m=round(-ld["z0"], 4), tops=tops_of(got[0], got[2]))
    if made and fr:
        # a made floor: what its picture shows standing on and round the floor, in the room's own frame ("stage":
        # the set builder keeps the floor and sets this round it, sets/place_build.py picture_stage)
        try:
            got = stage_land(pts, mask, fr, res["things"], {k: v[0] for k, v in bx.items()}, image)
        except ImportError as e:
            got = None
            print(f"no land: {e}", file=sys.stderr)
        if got:
            ld = write_relief(out_p.with_name(out_p.stem.replace("_depth", "") + "_land.glb"), *got[:3], got[3])
            res["land"] = dict(ld, has=got[4], under_m=round(-ld["z0"], 4), tops=tops_of(got[0], got[2]), stage=True)
    if not open_land and fr:
        # the picture on the room's own walls, floor and roof, where it sees them (room_faces)
        res["faces"] = write_faces(out_p, room_faces(pts, mask, intr, image, fr))
    # how wide the picture sees (degrees): a place in the open's panorama is made from the picture, and the land
    # built from the picture itself stands for that much of it
    res["hfov_deg"] = round(float(np.degrees(2 * np.arctan(0.5 / max(abs(fxi), 1e-6)))), 1)
    out_p.write_text(json.dumps(res, indent=1))
    print(json.dumps({"room_m": res["room_m"], "things": len(res["things"]), "scale": res["scale"],
                      "land": (res.get("land") or {}).get("faces"),
                      "room_faces": [f["face"] for f in res.get("faces") or []]}))


if __name__ == "__main__":
    main()
