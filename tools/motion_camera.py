"""MOTION-ANCHORED cameras (any story, any creature): solve a shot's camera from the figures' per-frame poses.

    ~/tts-ab/.venv-pose/bin/python -I tools/motion_camera.py MOTION_DIR [--talk] [--size 1152x640] [--write] [--json OUT]

A shot's camera was solved by the blockout from the STATIC staging pose; a creature that moves (wings that open, a
neck that swings, a fall) can then put a wing across the lens, hide the subject or leave the frame. This solver reads
the batch output of one moment (MOTION_DIR = motion_batch's <out>/<id>: shot_render.json + joints_<who>.npz) and
builds each figure's VOLUME per frame:
  - a composite creature: the same primitives its mesh was built from (creature_sdf.prims on the frozen template +
    creature_spec: tapered capsules, trunk ellipsoids, the wing MEMBRANES as triangles), each sphere sample riding
    its nearest bone (bone frame = bone axis + the body's lateral axis), membranes on their joints;
  - any other creature: capsules along its rig parents; a person: capsules on the body joints (scaled by height).
Candidates are camera positions inside the set's view_region (world.json; a pano set is only valid near its capture
point) looking at the framing's subject; each is REJECTED when, on any checked frame,
  - the lens is inside / brushing a figure (clearance),
  - a creature part sits in the frustum close to the lens (the "wing edge near the lens"),
  - the subject's key points are occluded (by another figure or by the creature's own wing / arm),
  - the subject cannot be framed at a usable lens,
  - it lies outside the view region;
and the survivors are ranked by background parallax (the pano depth from points.npy: rays whose background is seen at a
large angle from the capture point show the set's empty back side), the shot's 180-degree line side, height for the
framing, and closeness to the shot's own camera intent. Authored moves (camera.path, e.g. pull_back) keep their move:
the start is solved and the end is the same move from it, every frame checked.
--write puts the explicit camera into MOTION_DIR/shot_render.json and shot.json (the old one kept as camera_pre_motion).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SENSOR_W = 36.0
PARALLAX_MAX = 32.0      # deg: a ray whose background the capture point saw at more than this shows the set's back
UP = np.array([0.0, 0.0, 1.0])
# framing -> (subject fill of the frame, both directions), the subject selection
FILL = {"close-up": 0.62, "medium close-up": 0.78, "medium": 0.85, "two-shot": 0.85, "wide": 0.9,
        "low angle": 0.9, "high angle": 0.9}
HUMAN_CAPS = [("pelvis", "spine_mid", 0.15), ("spine_mid", "neck", 0.16), ("neck", "head", 0.07),
              ("l_shoulder", "l_elbow", 0.06), ("l_elbow", "l_wrist", 0.05), ("r_shoulder", "r_elbow", 0.06),
              ("r_elbow", "r_wrist", 0.05), ("l_hip", "l_knee", 0.09), ("l_knee", "l_ankle", 0.06),
              ("r_hip", "r_knee", 0.09), ("r_knee", "r_ankle", 0.06), ("l_shoulder", "r_shoulder", 0.08),
              ("l_hip", "r_hip", 0.1)]
HUMAN_HEAD_R = 0.12


def _n(v):
    v = np.asarray(v, float)
    d = float(np.linalg.norm(v))
    return v / d if d > 1e-9 else v


# ---------------------------------------------------------------------------------------------- body volumes
class Body:
    """One figure's volume, posed per frame: spheres (centres riding bones) + triangles (membranes on joints)."""

    def __init__(self, who, kind, height, names):
        self.who, self.kind, self.height, self.names = who, kind, float(height or 1.8), list(names)
        self.sph = []        # (bone or None, local offset (3,) / joint name, radius)
        self.caps = []       # (joint a, joint b, ra, rb): sampled into spheres at pose time
        self.tris = []       # (joint, joint, joint)
        self.bones = []      # (name, head joint, tail joint, ref choice)
        self.key = {}        # named key points: joint names
        self.face = None     # (from joint, to joint): the way the face looks

    def pose(self, J):
        """-> (centres (n,3), radii (n,), triangles (m,3,3)) for one frame of joints {name: (3,)}."""
        C, Rr = [], []
        for a, b, ra, rb in self.caps:
            if a not in J or b not in J:
                continue
            pa, pb = np.asarray(J[a], float), np.asarray(J[b], float)
            L = float(np.linalg.norm(pb - pa))
            k = max(1, int(math.ceil(L / max(0.6 * min(ra, rb), 0.03))))
            for i in range(k + 1):
                t = i / k
                C.append(pa + (pb - pa) * t)
                Rr.append(ra + (rb - ra) * t)
        if self.sph:
            frames = {}
            for bn, hj, tj, ref in self.bones:
                if hj in J and tj in J:
                    frames[bn] = (np.asarray(J[hj], float), self._frame(J, hj, tj, ref))
            for bn, off, r in self.sph:
                if bn in frames:
                    o, F = frames[bn]
                    C.append(o + F @ off)
                    Rr.append(r)
        T = np.array([[J[a], J[b], J[c]] for a, b, c in self.tris if a in J and b in J and c in J], float).reshape(-1, 3, 3)
        return (np.array(C, float).reshape(-1, 3), np.array(Rr, float), T)

    @staticmethod
    def _frame(J, hj, tj, ref):
        z = _n(np.asarray(J[tj], float) - np.asarray(J[hj], float))
        r = _lateral(J) if ref == "lat" else _n(np.asarray(J["neck"], float) - np.asarray(J["pelvis"], float))
        x = _n(r - np.dot(r, z) * z)
        y = np.cross(z, x)
        return np.stack([x, y, z], 1)


def _lateral(J):
    for a, b in (("l_clav", "r_clav"), ("l_shoulder", "r_shoulder"), ("l_hipb", "r_hipb"), ("l_hip", "r_hip")):
        if a in J and b in J:
            return _n(np.asarray(J[a], float) - np.asarray(J[b], float))
    return np.array([0, 1.0, 0])


def composite_body(who, model, names, height):
    """The frozen composite creature's own primitives (creature_sdf.prims) as a posable Body, or None."""
    d = os.path.dirname(model)
    tp, sp, sk = (os.path.join(d, f) for f in ("composite_creatures.json", "creature_spec.json", "skeleton.json"))
    if not (os.path.exists(tp) and os.path.exists(sk)):
        return None
    import creature_sdf as CS
    sk_ = json.load(open(sk))
    spec = json.load(open(sp)) if os.path.exists(sp) else {}
    tname = spec.get("template") or str(sk_.get("template", "")).split(":")[-1]
    tpl = json.load(open(tp)).get(tname)
    if tpl is None:
        return None
    T = copy.deepcopy(tpl)
    for k, v in spec.items():
        if k == "joints":
            T["joints"].update(v)
        elif k != "template":
            T[k] = v
    T["height_m"] = float(spec.get("height_m") or sk_.get("height_m") or height)
    Jr = {k: np.asarray(v, float) for k, v in sk_["joints"].items()}
    B = Body(who, "creature", T["height_m"], names)
    for bn, par, hj, tj in sk_["bones"]:
        if hj in Jr and tj in Jr:
            z = _n(Jr[tj] - Jr[hj])
            ref = "lat" if abs(np.dot(z, _lateral(Jr))) < 0.8 else "up"
            B.bones.append((bn, hj, tj, ref))
    # the rest-pose primitives; joint-named capsule ends stay ON their joints, the rest rides the nearest bone
    pr = CS.prims(T, {k: v.copy() for k, v in CS.joints_m(T).items()} | Jr)
    S = T["height_m"]
    jn_of = lambda p: next((k for k, v in Jr.items() if np.allclose(v, p, atol=1e-6)), None)  # noqa: E731
    for it in pr:
        kind, par_ = it[0], it[1]
        if kind == "cap":
            a, b, ra, rb = par_
            ja, jb = jn_of(a), jn_of(b)
            if ja and jb:
                B.caps.append((ja, jb, float(ra), float(rb)))
                continue
            L = float(np.linalg.norm(b - a))
            k = max(1, int(math.ceil(L / max(0.6 * min(ra, rb), 0.01 * S))))
            for i in range(k + 1):
                t = i / k
                _attach(B, Jr, a + (b - a) * t, ra + (rb - ra) * t)
        elif kind == "ell":
            c, r = par_
            rm = float(min(r))
            _attach(B, Jr, c, rm)
            for ax in range(3):                       # the long axes as a row of spheres of the short radius
                if r[ax] > rm * 1.2:
                    e = np.zeros(3)
                    e[ax] = 1.0
                    for s in (-1, 1):
                        for f in (0.45, 0.8):
                            _attach(B, Jr, c + s * e * r[ax] * f, rm * (1.0 - 0.5 * f))
        elif kind == "tri":
            ids = [jn_of(p) for p in par_[:3]]
            if all(ids):
                B.tris.append(tuple(ids))
    B.face = ("head", "snout") if "snout" in Jr else None
    B.key = {"head": [k for k in ("head", "snout", "head_top", "jaw") if k in Jr],
             "body": [k for k in ("chest", "spine", "pelvis", "neck") if k in Jr]}
    return B


def _attach(B, Jr, p, r):
    best = None
    for bn, hj, tj, ref in B.bones:
        a, b = Jr[hj], Jr[tj]
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(ab @ ab, 1e-12), 0, 1))
        d = float(np.linalg.norm(p - (a + t * ab)))
        if best is None or d < best[0]:
            best = (d, bn, hj, tj, ref)
    if best is None:
        return
    _, bn, hj, tj, ref = best
    F = Body._frame(Jr, hj, tj, ref)
    B.sph.append((bn, F.T @ (p - Jr[hj]), float(r)))


def generic_body(who, kind, height, names, parents=None):
    B = Body(who, kind, height, names)
    s = float(height or 1.8) / 1.8
    if kind == "human":
        for a, b, r in HUMAN_CAPS:
            if a in names and b in names:
                B.caps.append((a, b, r * s, r * s))
        if "head" in names:
            B.caps.append(("head", "head", HUMAN_HEAD_R * s, HUMAN_HEAD_R * s))
        B.face = ("head", "nose") if "nose" in names and "head" in names else None
        B.key = {"head": [k for k in ("head", "nose", "head_top") if k in names],
                 "body": [k for k in ("spine_mid", "pelvis", "neck") if k in names]}
    else:
        for c, p in (parents or {}).items():
            if c in names and p in names:
                B.caps.append((p, c, 0.05 * float(height), 0.05 * float(height)))
        B.key = {"head": [k for k in ("head", "snout") if k in names], "body": [k for k in ("chest", "pelvis") if k in names]}
    return B


# ---------------------------------------------------------------------------------------------- the set
class SetView:
    """The set's view_region + the pano's depth seen from its capture point (for the back-side / parallax test)."""

    def __init__(self, shot):
        self.center, self.R, self.dep = None, None, None
        st = shot.get("set") or {}
        wdir = (st.get("world") or {}).get("dir")
        if not wdir:
            ap = os.environ.get("BLOCKOUT_ASSETS")
            bk = shot.get("book")
            cands = [ap] + ([os.path.join(bk, "work", "video", "stage", "assets.json")] if bk and os.path.isdir(bk) else [])
            if bk and not os.path.isdir(bk):
                try:
                    import characters as CH
                    cands.append(str(CH.vdir_of(bk) / "stage" / "assets.json"))
                except Exception:  # noqa: BLE001
                    pass
            where = st.get("where") or shot.get("where")
            for c in [c for c in cands if c and os.path.exists(c)]:
                ent = next((e for e in json.load(open(c)).get("sets", []) if e.get("where") == where), None)
                if ent:
                    wdir = ent["world_dir"]
                    break
        if not wdir or not os.path.exists(os.path.join(wdir, "world.json")):
            return
        W = json.load(open(os.path.join(wdir, "world.json")))
        vr = W.get("view_region") or {}
        self.center = np.asarray(vr.get("center") or W["capture_camera"]["pos"], float)
        self.R = float(vr.get("radius_m", 6.0))
        self.O = np.asarray(W["capture_camera"]["pos"], float)
        self.eye = float((W.get("ground") or {}).get("eye_height_m", self.O[2]))
        pp = os.path.join(wdir, "points.npy")
        if os.path.exists(pp):
            P = np.load(pp, mmap_mode="r")[::2, ::2].reshape(-1, 3).astype(float)
            q = P - self.O
            d = np.linalg.norm(q, axis=1)
            ok = np.isfinite(d) & (d > 0.2)
            q, d = q[ok], d[ok]
            az = np.arctan2(q[:, 1], q[:, 0])
            el = np.arcsin(np.clip(q[:, 2] / d, -1, 1))
            self.na, self.ne = 720, 360
            ia = ((az + math.pi) / (2 * math.pi) * self.na).astype(int) % self.na
            ie = np.clip(((el + math.pi / 2) / math.pi * self.ne).astype(int), 0, self.ne - 1)
            dep = np.full((self.ne, self.na), np.inf)
            np.minimum.at(dep, (ie, ia), d)
            for _ in range(4):                     # fill empty bins from their neighbours
                bad = ~np.isfinite(dep)
                if not bad.any():
                    break
                nb = np.stack([np.roll(dep, s, ax) for ax in (0, 1) for s in (-1, 1)])
                nb[~np.isfinite(nb)] = np.nan
                with np.errstate(all="ignore"):
                    fill = np.nanmin(nb, 0)
                dep[bad] = fill[bad]
            dep[~np.isfinite(dep)] = 1e4
            self.dep = dep

    def inside(self, p, margin=0.3):
        return self.center is None or float(np.linalg.norm(np.asarray(p) - self.center)) <= self.R - margin

    def blocks(self, p, k, stop=0.6):
        """Does the SET (the pano surface seen from the capture point) stand between p and k?  The ground plane is
        not an occluder (figures stand on it)."""
        if self.dep is None:
            return False
        L = float(np.linalg.norm(k - p))
        if L <= stop:
            return False
        ts = np.linspace(0.3, L - stop, max(8, int((L - stop) / 0.25)))
        X = p[None, :] + ts[:, None] * _n(k - p)[None, :]
        q = X - self.O
        r = np.linalg.norm(q, axis=1)
        az = np.arctan2(q[:, 1], q[:, 0])
        el = np.arcsin(np.clip(q[:, 2] / np.maximum(r, 1e-9), -1, 1))
        ia = ((az + math.pi) / (2 * math.pi) * self.na).astype(int) % self.na
        ie = np.clip(((el + math.pi / 2) / math.pi * self.ne).astype(int), 0, self.ne - 1)
        return bool(np.any((r >= self.dep[ie, ia]) & (X[:, 2] > self.O[2] - self.eye + 0.3)))

    def parallax(self, p, dirs):
        """Per ray: the angle (deg) at which the capture point sees the background this ray hits (0 = sky / none)."""
        if self.dep is None:
            return np.zeros(len(dirs))
        out = np.zeros(len(dirs))
        ts = np.geomspace(0.5, 150.0, 90)
        for i, d in enumerate(dirs):
            X = p[None, :] + ts[:, None] * d[None, :]
            q = X - self.O
            r = np.linalg.norm(q, axis=1)
            az = np.arctan2(q[:, 1], q[:, 0])
            el = np.arcsin(np.clip(q[:, 2] / np.maximum(r, 1e-9), -1, 1))
            ia = ((az + math.pi) / (2 * math.pi) * self.na).astype(int) % self.na
            ie = np.clip(((el + math.pi / 2) / math.pi * self.ne).astype(int), 0, self.ne - 1)
            hit = np.nonzero(r >= self.dep[ie, ia])[0]
            if len(hit) and r[hit[0]] < 1e3 and X[hit[0], 2] > self.O[2] - self.eye + 0.25:   # the ground plane reads
                qh = q[hit[0]]                                                                    # from anywhere
                out[i] = math.degrees(math.acos(np.clip(np.dot(_n(qh), d), -1, 1)))
        return out


# ---------------------------------------------------------------------------------------------- geometry
def cam_basis(p, look):
    f = _n(np.asarray(look, float) - p)
    r = np.cross(f, UP)
    r = _n(r if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0, 0]))
    u = np.cross(r, f)
    return f, r, u


def project(P, p, look, lens, aspect):
    """-> (x, y in [-1, 1] at the frame edges, depth)."""
    f, r, u = cam_basis(p, look)
    Q = np.asarray(P, float) - p
    z = Q @ f
    zz = np.maximum(z, 1e-6)
    x = (Q @ r) / zz * lens / (SENSOR_W / 2)
    y = (Q @ u) / zz * lens / (SENSOR_W / aspect / 2)
    return x, y, z


def seg_hits(p, K, C, Rr, T, own=False, head_r=0.0):
    """For each key point K[i]: is the segment p->K[i] blocked by a sphere (C, Rr) or a triangle T?  own: the key lies
    INSIDE this body (a joint): the segment stops where it leaves the spheres that contain the key (the body's own
    surface there), and those spheres are ignored - the rest of the same body (a wing, an arm in front) still blocks."""
    out = np.zeros(len(K), bool)
    for i, k in enumerate(K):
        d = k - p
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        u = d / L
        Lc = L
        keep = np.ones(len(C), bool)
        if own and len(C):
            q = k - C
            inn = np.linalg.norm(q, axis=1) < Rr
            if inn.any():
                b = -(q[inn] @ u)                            # exit distance back toward the camera
                cc = np.einsum("ij,ij->i", q[inn], q[inn]) - Rr[inn] ** 2
                ex = b + np.sqrt(np.maximum(b * b - cc, 0))
                Lc = max(L - float(ex.max()) - 0.05, 0.0)
                keep = ~inn
            keep &= np.linalg.norm(C - k, axis=1) > head_r + Rr     # the head's own parts (snout, jaw, skull)
        if len(C):
            w = C - p
            t = np.clip(w @ u, 0.0, Lc)
            dist = np.linalg.norm(w - t[:, None] * u, axis=1)
            m = keep & (dist < Rr) & (t > 0.05) & (t < Lc - 0.02)
            if m.any():
                out[i] = True
                continue
        if len(T):
            e1, e2 = T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]
            h = np.cross(u, e2)
            a = np.einsum("ij,ij->i", e1, h)
            ok = np.abs(a) > 1e-9
            f_ = np.where(ok, 1.0 / np.where(ok, a, 1), 0)
            s = p - T[:, 0]
            uu = f_ * np.einsum("ij,ij->i", s, h)
            q = np.cross(s, e1)
            vv = f_ * (q @ u)
            tt = f_ * np.einsum("ij,ij->i", e2, q)
            hit = ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 0.05) & (tt < Lc - 0.05)
            if hit.any():
                out[i] = True
    return out


def tri_samples(T, n=5):
    if not len(T):
        return np.zeros((0, 3))
    ws = [(i / n, j / n) for i in range(n + 1) for j in range(n + 1 - i)]
    return np.concatenate([T[:, 0] + a * (T[:, 1] - T[:, 0]) + b * (T[:, 2] - T[:, 0]) for a, b in ws])


# ---------------------------------------------------------------------------------------------- the solver
def load_moment(mdir):
    shot = json.load(open(os.path.join(mdir, "shot_render.json")))
    R = json.load(open(os.path.join(mdir, "render.json")))
    figs = {f["who"]: f for f in shot["figures"]}
    seqs, bodies = {}, {}
    for w in R["figures"]:
        z = np.load(os.path.join(mdir, f"joints_{w}.npz"))
        names = [str(n) for n in z["names"]]
        seqs[w] = [{n: z["pos"][f, k] for k, n in enumerate(names) if not n.startswith("_")} for f in range(len(z["pos"]))]
        f = figs.get(w, {})
        kind = f.get("kind", R["figures"][w])
        mdl = f.get("model")
        mdl = mdl if isinstance(mdl, str) else (mdl or {}).get("blend")
        B = composite_body(w, mdl, names, f.get("height")) if kind == "creature" and mdl else None
        if B is None:
            par = {}
            if mdl and os.path.exists(os.path.join(os.path.dirname(mdl), "rig.json")):
                par = json.load(open(os.path.join(os.path.dirname(mdl), "rig.json"))).get("parents", {})
            B = generic_body(w, kind, f.get("height", 1.8), names, par)
        bodies[w] = B
    return shot, R, seqs, bodies


def _whole(B, J0):
    """A whole figure for framing: a person entirely; a creature's head, trunk and limbs (its long extremities - tail
    beyond the first third, wing tips and fingers - may leave a wide frame)."""
    if B.kind != "creature":
        return list(J0)
    tails = sorted([k for k in J0 if k.startswith("tail") and k[4:].isdigit()], key=lambda k: int(k[4:]))
    drop = set(tails[max(1, len(tails) // 3):]) | {k for k in J0 if k.startswith("wing_") and
                                                   k.split("_")[-1] in ("tip", "finger1", "finger2", "finger3", "wrist")}
    return [k for k in J0 if k not in drop]


def subject_of(cam, bodies, seqs):
    """The framing's subject: (who list, key-point names per who, framing)."""
    fr = cam.get("framing", "medium")
    tg = cam.get("target")
    tl = [tg] if isinstance(tg, str) else (tg if isinstance(tg, list) and tg and isinstance(tg[0], str) else [])
    subj = {}
    for t in tl:
        w, _, part = t.partition(":")
        if w not in bodies:
            continue
        B = bodies[w]
        if part:
            subj[w] = B.key.get(part, [part]) + (B.key["body"][:1] if fr != "close-up" else [])
        elif B.kind == "creature" and fr in ("close-up", "medium close-up"):
            subj[w] = B.key["head"] + ([] if fr == "close-up" else ["neck"])
        elif B.kind == "human" and fr in ("close-up", "medium close-up", "medium"):
            keep = {"close-up": ("head", "head_top", "nose", "neck"), "medium close-up": ("head", "head_top", "neck", "l_shoulder", "r_shoulder", "spine_mid"),
                    "medium": ("head", "head_top", "neck", "l_shoulder", "r_shoulder", "spine_mid", "pelvis", "l_hand", "r_hand")}[fr]
            subj[w] = [k for k in keep if k in seqs[w][0]]
        else:
            subj[w] = _whole(bodies[w], seqs[w][0])
    if not subj:                                       # a wide / no target: everyone, whole bodies
        subj = {w: _whole(bodies[w], seqs[w][0]) for w in bodies}
    return subj, fr


def solve(mdir, talk=False, size=(1152, 640), log=print):
    shot, R, seqs, bodies = load_moment(mdir)
    W, H = size
    aspect = W / H
    cam0 = dict(shot.get("camera") or {})
    sv = SetView(shot)
    N = min(len(s) for s in seqs.values())
    frames = [0] if talk else sorted(set(list(range(0, N, max(1, N // 16))) + [N - 1]))
    posed = {f: {w: bodies[w].pose(seqs[w][f]) for w in bodies} for f in frames}
    subj, fr = subject_of(cam0, bodies, seqs)
    fill = FILL.get(fr, 0.85)
    S_pts = np.array([seqs[w][f][k] for f in frames for w, ks in subj.items() for k in ks], float)
    # the visibility keys are HEADS (the subject's, and in wider framings everyone's): a body joint sits deep inside
    # its own volume, and an arm or wing in front of a chest is fine; a hidden head is not
    keys = [(w, k) for w, B in bodies.items() if w in subj for k in B.key["head"][:2]]
    if fr in ("wide", "low angle", "high angle", "two-shot", "medium"):
        keys += [(w, B.key["head"][0]) for w, B in bodies.items() if w not in subj and B.key["head"]]
    look = S_pts.mean(0) if len(S_pts) else np.zeros(3)
    if fr in ("close-up", "medium close-up") and len(S_pts):
        look = (S_pts.min(0) + S_pts.max(0)) / 2
    # the shot's intent: its own camera direction (path start or a solved/explicit pos), else the line's right side
    intent = None
    if cam0.get("path"):
        intent = _n(np.asarray(cam0["path"][0]["pos"], float) - np.asarray(cam0["path"][0]["look"], float))
    elif cam0.get("pos") is not None and cam0.get("look") is not None:
        intent = _n(np.asarray(cam0["pos"], float) - np.asarray(cam0["look"], float))
    line = cam0.get("line") or []
    side = None
    if len(line) >= 2 and all(w in seqs for w in line[:2]):
        a = np.mean([seqs[line[0]][0][k] for k in bodies[line[0]].key["body"][:1] or list(seqs[line[0]][0])[:1]], 0)
        b = np.mean([seqs[line[1]][0][k] for k in bodies[line[1]].key["body"][:1] or list(seqs[line[1]][0])[:1]], 0)
        v = _n(np.array([b[0] - a[0], b[1] - a[1], 0.0]))
        side = np.array([v[1], -v[0], 0.0]) * (-1 if cam0.get("side") == "left" else 1)
    # candidate positions inside the view region (or around the subject when the set has none)
    c0 = sv.center if sv.center is not None else np.array([look[0], look[1], 1.6])
    Rv = (sv.R - 0.3) if sv.R else 8.0
    cands = []
    for rr in (0.0, 0.25, 0.5, 0.72, 0.92):
        for th in (np.arange(0, 360, 15) if rr else [0.0]):
            for z in (1.2, 1.7, 2.4, 3.2, 4.2, 5.2, 6.2):
                p = np.array([c0[0] + rr * Rv * math.cos(math.radians(th)), c0[1] + rr * Rv * math.sin(math.radians(th)), z])
                if sv.inside(p):
                    cands.append(p)
    path0 = cam0.get("path")
    move_d, lens_ratio = 0.0, 1.0
    if path0 and len(path0) >= 2:
        P0, P1 = sorted(path0, key=lambda k: k["frame"])[0], sorted(path0, key=lambda k: k["frame"])[-1]
        d0 = float(np.linalg.norm(np.asarray(P0["pos"]) - np.asarray(P0["look"])))
        d1 = float(np.linalg.norm(np.asarray(P1["pos"]) - np.asarray(P1["look"])))
        move_d, lens_ratio = d1 - d0, float(P1["lens"]) / float(P0["lens"])
    res = []
    S_all = np.array([seqs[w][f][k] for f in frames for w, ks in subj.items() for k in ks], float)
    look_c = look
    for p in cands:
        look = _center_look(p, look_c, S_all, aspect)
        why = _check(p, look, frames, posed, seqs, subj, keys, bodies, fill, aspect, sv, fr,
                     move_d, lens_ratio, N)
        if isinstance(why, str):
            res.append((None, p, why))
            continue
        lens, endp, occ, near_m = why
        f, r_, u_ = cam_basis(p, look)
        grid = [_n(f + r_ * x * (SENSOR_W / 2) / lens + u_ * y * (SENSOR_W / aspect / 2) / lens)
                for x in (-0.9, -0.45, 0, 0.45, 0.9) for y in (-0.8, 0, 0.8)]
        par = sv.parallax(p, grid)
        if float(np.percentile(par, 90)) > PARALLAX_MAX:
            res.append((None, p, f"pano back side (p90 parallax {np.percentile(par, 90):.0f} deg)"))
            continue
        d_ = _n(p - look)
        hz = 0.0 if sv.center is None else max(0.0, p[2] - (sv.center[2] + 1.5)) / 2.0
        # parallax inside the view region is what the region allows; only a ray that sees its background at a large
        # angle from the capture point (the set's empty back side, stretched silhouettes) costs
        score = hz + max(0.0, float(np.mean(par)) - 10.0) / 5.0 + max(0.0, float(np.percentile(par, 90)) - 25.0) / 8.0 + 3.0 * occ
        if intent is not None:
            score += 1.5 * (1 - float(np.dot(_n(d_ * [1, 1, 0]), _n(intent * [1, 1, 0])))) / 2
        if side is not None and np.dot(d_, side) < 0:
            score += 1.0                                       # crosses the 180-degree line
        elev = math.degrees(math.asin(np.clip(-d_[2], -1, 1)))   # >0: looking up at the subject
        if fr in ("close-up", "medium close-up"):
            score += max(0.0, elev - 15) / 15.0                 # from above / beside the head, not from below it
        else:
            score += abs(elev) / 40.0
        score += 1.5 * max(0.0, (24 - lens) / 10) + 0.3 * max(0.0, (35 - lens) / 20) + 0.3 * max(0.0, (lens - 85) / 40)
        res.append((score, p, {"lens": lens, "end": endp, "look": look.tolist(), "occluded_frac": occ, "near_lens_m": near_m,
                               "parallax_mean_deg": round(float(np.mean(par)), 1),
                               "parallax_p90_deg": round(float(np.percentile(par, 90)), 1), "elev_deg": round(elev, 1)}))
    ok = sorted([x for x in res if x[0] is not None], key=lambda x: x[0])
    rej = {}
    for x in res:
        if x[0] is None:
            rej[x[2].split(":")[0]] = rej.get(x[2].split(":")[0], 0) + 1
    out = {"framing": fr, "subject": subj, "frames_checked": len(frames), "candidates": len(res), "rejected": rej}
    if not ok:
        out["camera"] = None
        log(f"{os.path.basename(mdir)}: NO camera passes ({rej})")
        return out
    sc, p, m = ok[0]
    look = np.asarray(m["look"], float)
    cam = {"framing": fr, "line": cam0.get("line"), "target": cam0.get("target"), "pos": p.tolist(),
           "look": look.tolist(), "lens": round(m["lens"], 1), "solved_by": "motion_camera"}
    if m["end"] is not None:
        cam["path"] = [{"frame": 0, "pos": p.tolist(), "look": look.tolist(), "lens": round(m["lens"], 1)},
                       {"frame": N - 1, "pos": m["end"].tolist(), "look": look.tolist(),
                        "lens": round(m["lens"] * lens_ratio, 1)}]
        cam["move"] = cam0.get("move")
    out["alternatives"] = [{"score": round(x[0], 3), "pos": np.round(x[1], 2).tolist(), "lens": round(x[2]["lens"], 1),
                            "occ": x[2]["occluded_frac"], "parallax": x[2]["parallax_mean_deg"], "elev": x[2]["elev_deg"]}
                           for x in ok[1:int(os.environ.get("MOTION_CAM_ALTS", 8))]]
    out.update(camera=cam, score=round(sc, 3), metrics={k: v for k, v in m.items() if k not in ("end", "look")},
               region={"center": None if sv.center is None else sv.center.tolist(), "radius_m": sv.R,
                       "cam_dist_m": None if sv.center is None else round(float(np.linalg.norm(p - sv.center)), 2)})
    log(f"{os.path.basename(mdir)}: {fr} -> pos {np.round(p, 2).tolist()} lens {m['lens']:.0f} "
        f"occ {m['occluded_frac']:.2f} near {m['near_lens_m']} parallax {m['parallax_mean_deg']}/{m['parallax_p90_deg']} "
        f"({len(ok)}/{len(res)} pass; rejected {rej})")
    return out


def _center_look(p, look, P, aspect, it=3):
    """Aim at the centre of the subject's ANGULAR extent seen from p (a near person and a far creature both fit)."""
    for _ in range(it):
        x, y, z = project(P, p, look, 1.0, aspect)
        if (z <= 0.3).any():
            return look
        f, r, u = cam_basis(p, look)
        cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
        d = _n(f + r * cx * (SENSOR_W / 2) + u * cy * (SENSOR_W / aspect / 2))
        look = p + d * float(np.linalg.norm(look - p))
    return look


def _check(p, look, frames, posed, seqs, subj, keys, bodies, fill, aspect, sv, fr, move_d, lens_ratio, N):
    """-> reason string (rejected) or (lens, end pos or None, occluded fraction, nearest creature part in frame (m))."""
    d0 = float(np.linalg.norm(look - p))
    if d0 < 1.0:
        return "too close"
    endp = None
    if move_d:
        endp = p + _n(p - look) * move_d
        if not sv.inside(endp):
            return "move leaves region"
    # lens: fit the subject (all checked frames, the whole move) at the framing's fill
    lens = 200.0
    for f in frames:
        pc = p if endp is None else p + (endp - p) * (f / max(N - 1, 1))
        lf = 1.0 if endp is None else 1.0 + (lens_ratio - 1.0) * (f / max(N - 1, 1))
        P = np.array([seqs[w][f][k] for w, ks in subj.items() for k in ks], float)
        x, y, z = project(P, pc, look, 1.0, aspect)
        if (z <= 0.3).any():
            return "subject behind camera"
        ext = max(float(np.abs(x).max()), float(np.abs(y).max()), 1e-6)
        need = fill / ext                                    # the lens (mm) at which the subject fills `fill`
        lens = min(lens, need / lf)
    if lens < 14.0:
        return "cannot frame (lens < 14 mm)"
    lens = min(lens, 135.0)
    occ_n = occ_k = 0
    near = 99.0
    for f in frames:
        pc = p if endp is None else p + (endp - p) * (f / max(N - 1, 1))
        lf = 1.0 if endp is None else 1.0 + (lens_ratio - 1.0) * (f / max(N - 1, 1))
        for w, (C, Rr, T) in posed[f].items():
            clear = 0.8 if bodies[w].kind == "creature" else 0.45
            if len(C) and float(np.min(np.linalg.norm(C - pc, axis=1) - Rr)) < clear:
                return f"inside {w}"
            if bodies[w].kind == "creature":
                pts = np.concatenate([C, tri_samples(T)]) if len(T) else C
                if len(pts):
                    x, y, z = project(pts, pc, look, lens * lf, aspect)
                    inf = (np.abs(x) < 1.05) & (np.abs(y) < 1.05) & (z > 0)
                    if inf.any():
                        dmin = float(z[inf].min())
                        near = min(near, dmin)
                        near_lim = max(1.5, 0.35 * d0) if w not in subj else max(1.2, 0.25 * d0)
                        if dmin < near_lim:
                            return f"{w} part near the lens"
        for w, k in keys:
            K = np.asarray(seqs[w][f][k], float)[None, :]
            occ_k += 1
            if sv.blocks(pc, K[0]):                           # the set's own rock / wall in the way
                occ_n += 1
                continue
            occ_k -= 1
            for w2, (C, Rr, T) in posed[f].items():
                hr = (0.14 * bodies[w].height if bodies[w].kind == "creature" else 0.14) if w2 == w else 0.0
                if seg_hits(pc, K, C, Rr, T, own=(w2 == w), head_r=hr)[0]:
                    occ_n += 1
                    break
            occ_k += 1
    if fr in ("close-up", "medium close-up", "medium"):    # the subject shows its FACE (not the back of its head)
        lim_deg = 70.0 if fr != "medium" else 100.0
        bad = tot = 0
        for w in subj:
            fwd = bodies[w].face
            if not fwd:
                continue
            for f in frames:
                pc = p if endp is None else p + (endp - p) * (f / max(N - 1, 1))
                J = seqs[w][f]
                if fwd[0] in J and fwd[1] in J:
                    v = _n(np.asarray(J[fwd[1]], float) - np.asarray(J[fwd[0]], float))
                    c = _n(pc - np.asarray(J[fwd[0]], float))
                    tot += 1
                    bad += math.degrees(math.acos(np.clip(np.dot(v, c), -1, 1))) > lim_deg
        if tot and bad / tot > 0.2:
            return "faces away"
    occ = occ_n / max(occ_k, 1)
    lim = 0.05 if fr in ("close-up", "medium close-up", "medium") else 0.15
    if occ > lim:
        return f"occluded {occ:.2f}"
    return lens, endp, round(occ, 3), round(near, 2)


def write(mdir, res):
    for fn in ("shot_render.json", "shot.json"):
        fp = os.path.join(mdir, fn)
        if not os.path.exists(fp):
            continue
        s = json.load(open(fp))
        if "camera_pre_motion" not in s:
            s["camera_pre_motion"] = s.get("camera")
        s["camera"] = res["camera"]
        json.dump(s, open(fp, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mdir")
    ap.add_argument("--talk", action="store_true", help="a talking still: only frame 0 matters")
    ap.add_argument("--size", default="1152x640")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = solve(a.mdir, a.talk, tuple(int(x) for x in a.size.split("x")))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    if a.write and res.get("camera"):
        write(a.mdir, res)
    sys.exit(0 if res.get("camera") else 2)
