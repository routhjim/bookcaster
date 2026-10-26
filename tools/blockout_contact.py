"""Held props that obey contact rules, for the blockout (pure numpy, any python; per-frame capable).

Three jobs, all on a figure's world joints W (blockout_rig.place_human's output) and ex (its extras):

1. fk_model(W, rig_json): a rigged 3D model (tools/blockout_model.py) has its OWN bone lengths. Blender's
   import_model() turns each bone toward the target joints but keeps its length, so the model's hands end
   up somewhere else than the mannequin's and a prop built at the mannequin hand floats. This runs the
   SAME forward kinematics in numpy (bone table, twist bases and rotation order copied from
   blockout_scene.import_model) and returns the joints the posed model will really have, grounded by its
   feet. Props and contacts are then solved on those joints and Blender gets them as targets (they
   reproduce themselves), so the prop in Blender sits in the model's real hand.

2. solve_contacts(...): the shot's "contacts" list, e.g.
       {"who": "Edric", "prop": "hammer", "point": "face", "to": "glowing blade", "surface": "top",
        "at": [0.8, 0.0], "frame": "end"}
       {"who": "Edric", "prop": "sword", "point": "blade", "to": "Vorgrath:claw_r", "frame": "end"}
       {"who": "Edric", "prop": "sword", "point": "tip", "to": "Vorgrath:chest", "frame": "end"}
   "to" is a set prop by name (its "top" face; "at" = [x, y] on it, default the face centre) or a creature
   anchor "Who:anchor". The prop is posed so its contact point lies ON the target (hammer: the striking
   face flush on the surface, face normal = -surface normal, handle level; tongs: the jaws on the
   surface; sword: the blade's upper surface under an anchor, or the tip at it), then the holding arm is
   IK'd so the closed hand's centre is the prop's grip point. Out of reach: the figure steps toward the
   target (up to 0.6 m, logged) before the IK. Two-handed grips re-IK the other hand onto the handle.

3. check(...): the numbers: grip distance (hand centre to the handle axis, each holding hand), contact
   gap (+) / penetration (-) and face angle, the IK residual. LIMITS: grip < 2 cm, |gap| < 1 cm, no
   penetration over 1 cm, face angle < 5 deg. blockout.py writes them to control.json and fails loudly.

Props (blockout_rig.prop_prims) are built from (grip point `at`, axis `d` from the hand toward the
working end, `flat`): their parts and affordances live in the prop library (blockout_props).
"""

from __future__ import annotations

import math

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
LIMITS = {"grip": 0.02, "gap": 0.01, "pen": 0.01, "angle": 12.0}   # angle: a strike lands on the face's edge


def _n(v):
    v = np.asarray(v, float)
    d = float(np.linalg.norm(v))
    return v / d if d > 1e-9 else v


# ---------------------------------------------------------------------------------------------------
# Props come from the library (blockout_props: tools/stage_props/props.json + per-book additions);
# contacts name AFFORDANCES (grip, strike_face, tip, jaws, edge, top...), solved by affordance TYPE.
# ---------------------------------------------------------------------------------------------------
import blockout_props as props  # noqa: E402

LEGACY_POINT = {"face": "strike_face", "blade": "edge"}      # old shot files


def pk(ex):
    """Prop scale: the library is in metres for a 1.8 m figure (ex["scale"] is the BODY-unit scale)."""
    from blockout_rig import _standing_height
    return float(ex["scale"]) * _standing_height() / 1.8


prop_axes = props.prop_axes


def prop_point(item, name, at, d, flat, k):
    return props.point(item, LEGACY_POINT.get(name, name), at, d, flat, k)


def handle_seg(item, at, d, k):
    return props.handle_seg(item, at, d, k)


def _aff(c):
    name = LEGACY_POINT.get(c.get("point", ""), c.get("point", ""))
    return name, props.affordance(c["prop"], name)


def seg_dist(p, a, b):
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + ab * t)))


# ---------------------------------------------------------------------------------------------------
# Rigged-model forward kinematics (a copy of blockout_scene.import_model in numpy)
# ---------------------------------------------------------------------------------------------------
BONES = [  # name, head joint, tail joint, parent  (= blockout_model_bl.BONES)
    ("pelvis", "pelvis", "spine_mid", None), ("spine", "spine_mid", "neck", "pelvis"),
    ("neck", "neck", "head", "spine"), ("head", "head", "head_top", "neck"),
    ("r_clav", "neck", "r_shoulder", "spine"), ("l_clav", "neck", "l_shoulder", "spine"),
    ("r_uarm", "r_shoulder", "r_elbow", "r_clav"), ("l_uarm", "l_shoulder", "l_elbow", "l_clav"),
    ("r_farm", "r_elbow", "r_wrist", "r_uarm"), ("l_farm", "l_elbow", "l_wrist", "l_uarm"),
    ("r_hand", "r_wrist", "r_hand", "r_farm"), ("l_hand", "l_wrist", "l_hand", "l_farm"),
    ("r_hipb", "pelvis", "r_hip", "pelvis"), ("l_hipb", "pelvis", "l_hip", "pelvis"),
    ("r_thigh", "r_hip", "r_knee", "r_hipb"), ("l_thigh", "l_hip", "l_knee", "l_hipb"),
    ("r_shin", "r_knee", "r_ankle", "r_thigh"), ("l_shin", "l_knee", "l_ankle", "l_thigh"),
    ("r_foot", "r_ankle", "r_toe", "r_shin"), ("l_foot", "l_ankle", "l_toe", "l_shin"),
]
TWIST = {"pelvis": "hip", "spine": "sh", "neck": "sh", "head": "sh"}
REST_LR = np.array([0.0, 1.0, 0.0])          # rest pose faces +X: left = +Y


def _basis(d, l):
    d = _n(d)
    l = _n(l - d * np.dot(l, d))
    return np.stack([d, l, np.cross(d, l)], 1)


def _rot_between(a, b):
    a, b = _n(a), _n(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        ax = _n(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]))
        return 2 * np.outer(ax, ax) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def grip_centre(wrist, hand_tail, height=1.8):
    """The closed fist's centre on a model hand bone (wrist -> fingertips-ish)."""
    v = hand_tail - wrist
    return wrist + _n(v) * min(0.07 * height / 1.8, 0.45 * float(np.linalg.norm(v)))


_CANON_FOOT_PITCH = None


def _canon_foot_pitch():
    """The pose library's foot pitch when STANDING (ankle->toe vs the ground): a flat sole in the template."""
    global _CANON_FOOT_PITCH
    if _CANON_FOOT_PITCH is None:
        import blockout_rig as rig_
        J, _, _ = rig_.human_joints("standing", 1.8)
        v = np.asarray(J["r_toe"], float) - np.asarray(J["r_ankle"], float)
        _CANON_FOOT_PITCH = math.atan2(float(v[2]), float(np.hypot(v[0], v[1])))
    return _CANON_FOOT_PITCH


def foot_dir(t_dir, d_rest):
    """The model's foot target direction: the template's heading and pitch, re-pitched by the model's own rest
    foot pitch minus the template's flat pitch (a mesh whose ankle->toe bone points 24 deg down at rest keeps a
    FLAT sole when the template's foot is flat; aiming it horizontal lifted every toe ~28 deg)."""
    td = np.asarray(t_dir, float)
    hz = np.array([td[0], td[1], 0.0])
    hn = float(np.linalg.norm(hz))
    if hn < 1e-6:
        return td
    p_t = math.atan2(float(td[2]), hn)
    p_r = math.atan2(float(d_rest[2]), float(np.hypot(d_rest[0], d_rest[1])))
    p = p_t + (p_r - _canon_foot_pitch())
    L = float(np.linalg.norm(td))
    return (hz / hn * math.cos(p) + np.array([0.0, 0.0, 1.0]) * math.sin(p)) * L


def fk_model(T: dict, rest: dict, height: float = 1.8) -> dict:
    """Joints the rigged model reaches when import_model poses it onto targets T (mannequin joints).
    rest: the model's rest joints (rig.json "joints"). Returns new W (same keys as T), grounded so its
    lowest foot joint sits where the mannequin's did; W["r_hand"/"l_hand"] = the fist centres."""
    T = {k: np.asarray(v, float) for k, v in T.items()}
    rest = {k: np.asarray(v, float) for k, v in rest.items()}
    T.setdefault("spine_mid", (T["pelvis"] + T["neck"]) / 2)
    lr = {"hip": T["l_hip"] - T["r_hip"], "sh": T["l_shoulder"] - T["r_shoulder"]}
    D, head_p, tail_p = {}, {}, {}
    par = {b[0]: b[3] for b in BONES}
    hd = {b[0]: b[1] for b in BONES}
    tl = {b[0]: b[2] for b in BONES}
    for name, h, t, p in BONES:                    # BONES is parent-first already
        rh, rt = rest[h], rest[t]
        if p is None:
            hp, Dp = T["pelvis"], np.eye(3)
        else:
            Dp = D[p]
            hp = head_p[p] + Dp @ (rh - rest[hd[p]])
        d_rest = rt - rh
        if t in T and h in T:
            t_dir = T[t] - T[h]
            if name.endswith("_foot"):
                t_dir = foot_dir(t_dir, d_rest)
        else:
            t_dir = Dp @ d_rest
        if name in TWIST:
            Db = _basis(t_dir, lr[TWIST[name]]) @ np.linalg.inv(_basis(d_rest, REST_LR))
        elif t in T and h in T:
            Db = _rot_between(Dp @ d_rest, t_dir) @ Dp
        else:
            Db = Dp
        D[name], head_p[name] = Db, hp
        tail_p[name] = hp + Db @ d_rest
    W = {}
    joint_of = {"pelvis": head_p["pelvis"], "spine_mid": head_p["spine"], "neck": head_p["neck"],
                "head": head_p["head"], "head_top": tail_p["head"]}
    for s in ("r", "l"):
        joint_of.update({f"{s}_shoulder": head_p[f"{s}_uarm"], f"{s}_elbow": head_p[f"{s}_farm"],
                         f"{s}_wrist": head_p[f"{s}_hand"], f"{s}_hip": head_p[f"{s}_thigh"],
                         f"{s}_knee": head_p[f"{s}_shin"], f"{s}_ankle": head_p[f"{s}_foot"],
                         f"{s}_toe": tail_p[f"{s}_foot"]})
        joint_of[f"{s}_hand"] = grip_centre(head_p[f"{s}_hand"], tail_p[f"{s}_hand"], height)
    dh = joint_of["head"] - T["head"]
    for k, v in T.items():
        if k.startswith("_"):
            W[k] = v
        elif k in joint_of:
            W[k] = joint_of[k]
        elif k in ("nose", "r_eye", "l_eye", "r_ear", "l_ear", "helm"):
            W[k] = v + dh                          # rides with the head
        elif k.endswith("_heel"):
            s = k[0]
            W[k] = v + (joint_of[f"{s}_ankle"] - T[f"{s}_ankle"])
        else:
            W[k] = v
    feet = [k for k in ("r_ankle", "l_ankle", "r_toe", "l_toe", "r_heel", "l_heel") if k in T]
    dz = min(float(T[k][2]) for k in feet) - min(float(W[k][2]) for k in feet)
    for k in W:
        if not k.startswith("_"):
            W[k] = W[k] + np.array([0.0, 0.0, dz])
    return W


# ---------------------------------------------------------------------------------------------------
# IK and targets
# ---------------------------------------------------------------------------------------------------

def ik2(root, l1, l2, target, pole):
    d = target - root
    dist = float(np.linalg.norm(d))
    dist = min(max(dist, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    ax = _n(d)
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    p = pole - ax * np.dot(pole, ax)
    p = _n(p) if np.linalg.norm(p) > 1e-6 else _n(np.cross(ax, UP + 1e-3))
    return root + ax * a + p * h, root + ax * dist


def arm_lengths(W, s):
    return (float(np.linalg.norm(W[f"{s}_elbow"] - W[f"{s}_shoulder"])),
            float(np.linalg.norm(W[f"{s}_wrist"] - W[f"{s}_elbow"])),
            float(np.linalg.norm(W[f"{s}_hand"] - W[f"{s}_wrist"])))


def ik_hand(W, s, target, pole=None, iters=6):
    """Move side s's arm so the hand CENTRE lands on target (lengths taken from W). Returns residual (m)."""
    l1, l2, h = arm_lengths(W, s)
    sh = W[f"{s}_shoulder"]
    if pole is None:
        pole = W[f"{s}_elbow"] - 0.5 * (sh + W[f"{s}_wrist"])
    u = _n(target - sh)
    for _ in range(iters):
        el, wr = ik2(sh, l1, l2, target - u * h, pole)
        u = _n(wr - el)
    W[f"{s}_elbow"], W[f"{s}_wrist"] = el, wr
    W[f"{s}_hand"] = wr + u * h
    return float(np.linalg.norm(W[f"{s}_hand"] - target))


# ---------------------------------------------------------------------------------------------------
# Grip frames (the cleanup layer: takes ANY pose and minimally adjusts the arm chain)
# ---------------------------------------------------------------------------------------------------
# A power grip: the handle runs along the fist's THROUGH axis t (index knuckle -> little-finger knuckle;
# the working end on the thumb side, so t = -d), through the fist centre, which sits FIST_A along the
# fingers and FIST_B toward the palm from the wrist joint. The forearm meets the handle at 55-110 deg
# (ulnar .. radial deviation of a neutral fist at 90). Metres for a 1.8 m figure.
FIST_A, FIST_B = 0.075, 0.012
FIST_HALF = (0.045, 0.052, 0.040)               # fist proxy half sizes: along t, along the fingers, palm normal
DEV = (55.0, 110.0)
DEV2 = (35.0, 115.0)        # the SECOND hand on a two-handed handle (the pommel hand angles in)


def hand_frame(s, u_fa, d):
    """(u_f fingers, t through-axis, n_p palm normal) of side s's fist holding a handle along d."""
    t = -_n(d)
    u_f = _n(u_fa - t * np.dot(u_fa, t))
    n_p = np.cross(u_f, t) if s == "r" else np.cross(t, u_f)
    return u_f, t, n_p


def _clamp_angle(v, d, lo, hi):
    v, d = _n(v), _n(d)
    ang = math.degrees(math.acos(float(np.clip(np.dot(v, d), -1, 1))))
    if lo <= ang <= hi:
        return v
    tgt = math.radians(lo if ang < lo else hi)
    perp = v - d * np.dot(v, d)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(d, UP if abs(d[2]) < 0.9 else np.array([1.0, 0, 0]))
    perp = _n(perp)
    return _n(d * math.cos(tgt) + perp * math.sin(tgt))


def arm_pole(ex, s):
    """Natural elbow: down and out (never through the torso), a little back."""
    out = -ex["left"] if s == "r" else ex["left"]
    return -UP * 0.8 + out * 0.55 - ex["fwd"] * 0.15


def _fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


_SPH = _fib_sphere(6000)


def _grip_candidates(U, s, g, d, sh, l1, l2, ks, pole, W, dev=DEV, plane=None):
    """Vectorised: forearm directions U (n,3) -> (cost, elbow, wrist, reach error, angle, twist)."""
    t = -d
    uf = U - t[None] * (U @ t)[:, None]
    uf /= np.maximum(np.linalg.norm(uf, axis=1, keepdims=True), 1e-9)
    n_p = np.cross(uf, t[None]) if s == "r" else np.cross(t[None], uf)
    w = g[None] - (uf * FIST_A + n_p * FIST_B) * ks
    el = w - U * l2
    du = el - sh[None]
    lu = np.linalg.norm(du, axis=1)
    err = np.abs(lu - l1)
    upd = du / np.maximum(lu, 1e-9)[:, None]
    ang = np.degrees(np.arccos(np.clip(U @ d, -1, 1)))
    flex = np.degrees(np.arccos(np.clip(np.sum(upd * U, 1), -1, 1)))
    t0 = upd - U * np.sum(upd * U, 1)[:, None]
    tn = np.linalg.norm(t0, axis=1)
    t0 = t0 / np.maximum(tn, 1e-9)[:, None]
    tw = np.degrees(np.arctan2(np.sum(np.cross(t0, t[None]) * U, 1), t0 @ t))
    tw = np.where(tn > math.sin(math.radians(25)), tw, 0.0)
    mid = 0.5 * (sh[None] + w)
    eo = el - mid
    eo /= np.maximum(np.linalg.norm(eo, axis=1, keepdims=True), 1e-9)
    ax0, ax1 = W["pelvis"], W["neck"]
    axv = ax1 - ax0
    tt = np.clip(((el - ax0) @ axv) / max(float(axv @ axv), 1e-9), 0, 1)
    tors = np.linalg.norm(el - (ax0[None] + tt[:, None] * axv[None]), axis=1)
    cost = (err * 400.0                                     # reach (cm-ish, dominant)
            + np.maximum(dev[0] - ang, 0) ** 2 * 0.05 + np.maximum(ang - dev[1], 0) ** 2 * 0.05
            + ((ang - min(max(85.0, dev[0]), dev[1])) / 25.0) ** 2 * 0.6   # a neutral-ish wrist
            + np.maximum(np.abs(tw) - 85.0, 0) ** 2 * 0.02
            + np.maximum(flex - 145.0, 0) ** 2 * 0.05
            + np.maximum(0.13 * ks - tors, 0) * 300.0           # elbow outside the torso
            - (eo @ pole) * 0.8)                               # a natural elbow: down/out
    if plane is not None:                                  # a strike: forearm in the swing plane
        cost = cost + (np.degrees(np.arcsin(np.clip(np.abs(U @ plane), 0, 1))) / 10.0) ** 2
    return cost, el, w, err, ang, tw


def solve_grip(W, s, g, d, ex, key=None, second=False, dev=None, pose_pole=False, plane=None):
    """CLEANUP: put side s's FIST on a handle, minimally changing the arm chain of ANY pose: fist centre at
    g, through-axis along -d, forearm 55-110 deg to the handle (wrist deviation), forearm twist and elbow
    flexion within range, elbow down/out and off the torso. The shoulder stays; the forearm direction is
    searched (6000 directions + a local refinement) so the elbow satisfies the upper-arm length exactly.
    Stores the frame in ex["grip"][key or hand]; returns it (+ "residual", m: fist vs g)."""
    ks = pk(ex)
    l1, l2, _ = arm_lengths(W, s)
    sh = np.asarray(W[f"{s}_shoulder"], float)
    g = np.asarray(g, float)
    d = _n(d)
    pole = _n(arm_pole(ex, s))
    if pose_pole:                                   # keep the motion's own elbow (blended with natural)
        mp = W[f"{s}_elbow"] - 0.5 * (sh + W[f"{s}_wrist"])
        if np.linalg.norm(mp) > 0.02:
            pole = _n(0.7 * _n(mp) + 0.3 * pole)
    dev = tuple(dev) if dev is not None else (DEV2 if second else DEV)
    cost, el, w, err, ang, tw = _grip_candidates(_SPH, s, g, d, sh, l1, l2, ks, pole, W, dev, plane)
    i = int(np.argmin(cost))
    best = _SPH[i]
    rng = np.random.default_rng(0)
    for sc in (0.05, 0.015):                                # local refinement
        U = best[None] + rng.normal(0, sc, (1500, 3))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        U = np.vstack([best[None], U])
        c2, el2, w2, e2, a2, t2 = _grip_candidates(U, s, g, d, sh, l1, l2, ks, pole, W, dev, plane)
        j = int(np.argmin(c2))
        best = U[j]
    _, el1, w1, e1, a1, t1 = _grip_candidates(best[None], s, g, d, sh, l1, l2, ks, pole, W, dev, plane)
    u_fa = best
    el_, wr = el1[0], w1[0]
    el_ = sh + _n(el_ - sh) * l1                            # exact upper-arm length
    wr = el_ + _n(wr - el_) * l2                            # exact forearm length
    u_f, t, n_p = hand_frame(s, _n(wr - el_), d)
    fist = wr + (u_f * FIST_A + n_p * FIST_B) * ks
    W[f"{s}_elbow"], W[f"{s}_wrist"], W[f"{s}_hand"] = el_, wr, fist
    info = {"side": s, "g": g, "d": d, "u_f": u_f, "t": t, "n_p": n_p, "fist": fist, "second": second, "dev": dev,
            "residual": float(np.linalg.norm(fist - g))}
    hand = key or ("right" if s == "r" else "left")
    ex.setdefault("grip", {})[hand] = info
    del u_fa
    return info


def adapt_prop_axis(W, s, d, twist_max=85.0):
    """A FREE held prop (no contact) follows the hand: the smallest turn of its axis that a real fist
    allows from this forearm (handle 55-110 deg to the forearm, forearm twist within +-twist_max)."""
    fa = _n(W[f"{s}_wrist"] - W[f"{s}_elbow"])
    up_d = _n(W[f"{s}_elbow"] - W[f"{s}_shoulder"])
    d2 = _clamp_angle(_n(d), fa, *DEV)
    t0 = up_d - fa * np.dot(up_d, fa)
    if np.linalg.norm(t0) > math.sin(math.radians(25)):    # near-straight arm: the humerus takes the twist
        t0 = _n(t0)
        t = -d2
        tw = math.degrees(math.atan2(float(np.dot(np.cross(t0, t), fa)), float(np.dot(t0, t))))
        tc = max(-twist_max, min(twist_max, tw))
        if tw != tc:
            from blockout_rig import rodrigues
            d2 = _n(rodrigues(d2, fa, tc - tw))
    return d2


def _cone(d, deg, n=48, seed=0):
    """d plus n directions within a cone of half-angle deg around it."""
    from blockout_rig import rodrigues
    rng = np.random.default_rng(seed)
    out = [_n(d)]
    a = np.cross(d, UP if abs(d[2]) < 0.9 else np.array([1.0, 0, 0]))
    for _ in range(n):
        ax = rodrigues(_n(a), d, rng.uniform(0, 360))
        out.append(_n(rodrigues(d, ax, deg * math.sqrt(rng.uniform(0, 1)))))
    return out


def choose_tip_axis(W, s, tgt, d0, along, ex, n=None, max_dip=20.0, dev=None):
    """The prop axis (within 40 deg of d0) whose grip is most natural; on a surface (normal n) the axis
    may dip into it by at most max_dip degrees (it rests along the surface)."""
    best = None
    for dc in _cone(_n(tgt - W[f"{s}_shoulder"]) if d0 is None else d0, 40.0 if not ex.get("_fast") else 15.0,
                    n=60 if not ex.get("_fast") else 8):
        if n is not None and float(np.dot(dc, n)) < -math.sin(math.radians(max_dip)):
            continue
        Wt = _copyW(W)
        gi = solve_grip(Wt, s, tgt - dc * along, dc, dict(ex, grip={}), dev=dev)
        bad = grip_badness(Wt, gi) + max(0.0, 0.03 - float(np.dot(gi["fist"] - Wt["pelvis"], ex["fwd"]))) * 300
        if best is None or bad < best[0]:
            best = (bad, dc)
    return best[1]


def grip_badness(W, gi, second=False):
    """How far a solved grip is from natural (0 = inside every range); used to choose a free prop axis."""
    s = gi["side"]
    fa = _n(W[f"{s}_wrist"] - W[f"{s}_elbow"])
    up_d = _n(W[f"{s}_elbow"] - W[f"{s}_shoulder"])
    dev = tuple(gi.get("dev") or (DEV2 if second else DEV))
    ang = math.degrees(math.acos(float(np.clip(np.dot(fa, gi["d"]), -1, 1))))
    b = gi["residual"] * 1000 + max(dev[0] - ang, 0) + max(ang - dev[1], 0)
    t0 = up_d - fa * np.dot(up_d, fa)
    if np.linalg.norm(t0) > math.sin(math.radians(25)):
        t0 = _n(t0)
        tw = math.degrees(math.atan2(float(np.dot(np.cross(t0, gi["t"]), fa)), float(np.dot(t0, gi["t"]))))
        b += max(abs(tw) - (120 if second else 95), 0)
    return b + abs(ang - 85) * 0.01


def _copyW(W):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in W.items()}


def roll_flat(W, s, item, d):
    """The prop's ROLL about its handle from the prop data ("roll": {"axis": "flat"|"side", "rule":
    "forearm"}): the named axis = the forearm direction perpendicular to the handle (it lies in the swing
    plane of forearm + handle and leads the stroke). Returns the prop's 'flat' vector, or None (no rule)."""
    r = props.get(item).get("roll")
    if not r:
        return None
    fa = _n(W[f"{s}_wrist"] - W[f"{s}_elbow"])
    d = _n(d)
    e = fa - d * np.dot(fa, d)
    if np.linalg.norm(e) < 1e-3:
        return None
    e = _n(e)
    if r.get("axis", "flat") == "flat":
        return e
    return _n(np.cross(e, d))                 # side = cross(d, flat) = e  ->  flat = e x d


def roll_limit(prev, target, d, max_deg=30.0):
    """Turn prev toward target about d by at most max_deg (no flips, no spinning about the handle)."""
    d = _n(d)
    p = _n(prev - d * np.dot(prev, d))
    t = _n(target - d * np.dot(target, d))
    ang = math.degrees(math.atan2(float(np.dot(np.cross(p, t), d)), float(np.dot(p, t))))
    if abs(ang) <= max_deg:
        return t, ang
    from blockout_rig import rodrigues
    step = max_deg if ang > 0 else -max_deg
    return _n(rodrigues(p, d, step)), step


def grip_all(W, ex, fig, two_handed=False, fast=False, skip=()):
    """Every held handle item: a power grip at the current hand; the prop axis adapts to the hand."""
    for hand, item in (fig.get("holds") or {}).items():
        if not props.has_grip(item) or item in skip:     # (a contact poses this prop instead)
            continue
        s = "r" if hand == "right" else "l"
        g = W[f"{s}_hand"].copy()
        d0 = ex["hold"][hand]
        best = None
        two = two_handed and hand == "right" and props.has_grip(item)
        if fast:
            ex["_fast"] = True
        view = ex.get("_view")                    # READABILITY (staging knows the camera): the working
        wk = props.get(item).get("work")          # part must sit >= 1.5 fist sizes from the fist on screen
        cone_deg, cone_n = (35.0, 40) if two and not fast else (0.0, 0)
        if view is not None and wk and not fast and readability(g, d0, item, wk, ex, view) < 1.5:
            cone_deg, cone_n = max(cone_deg, 35.0), max(cone_n, 40)
        for dc in _cone(d0, cone_deg, n=cone_n):
            Wt, ext = _copyW(W), dict(ex, grip=dict(ex.get("grip", {})), grip_at=dict(ex.get("grip_at", {})))
            d = dc
            for _ in range(2 if fast else 3):    # the arm settles, the prop re-adapts
                d = adapt_prop_axis(Wt, s, d)
                gi = solve_grip(Wt, s, g, d, ext)
            bad = grip_badness(Wt, gi) + 0.02 * math.degrees(math.acos(float(np.clip(np.dot(d, d0), -1, 1))))
            if view is not None and wk:
                bad += max(0.0, 1.5 - readability(g, d, item, wk, ext, view)) * 20.0
            if two:                              # the second fist on the same handle must be natural too
                ext["hold"] = dict(ex["hold"], right=d)
                ext["grip_at"]["right"] = g
                regrip_two_handed(Wt, ext, fig, True)
                bad += grip_badness(Wt, ext["grip"]["left2"], second=True) if "left2" in ext["grip"] else \
                    (0.0 if "support" in ext else 5.0)
            if best is None or bad < best[0]:
                best = (bad, Wt, ext, d)
        _, Wt, ext, d = best
        W.update(Wt)
        ex["grip"] = ext["grip"]
        ex["hold"][hand] = d
        rf = roll_flat(W, s, item, d)                    # the head/edge leads the stroke
        if rf is not None:
            ex.setdefault("flat", {})[hand] = rf
        ex.setdefault("grip_at", {})[hand] = g


def readability(fist, d, item, work, ex, view):
    """Screen gap between the fist and the prop's working part, in fist sizes, seen from camera pos view."""
    k = pk(ex)
    wp = props.point(item, work, fist, d, ex.get("flat", {}).get("right"), k)
    v = _n(np.asarray(fist, float) - np.asarray(view, float))
    dist = float(np.linalg.norm(np.asarray(fist, float) - np.asarray(view, float)))
    gap = wp - fist
    perp = float(np.linalg.norm(gap - v * np.dot(gap, v)))
    return perp / max(2 * FIST_HALF[0] * k, 1e-6) * 1.0 * (1.0 if dist > 0 else 0)


def fist_prims(ex, ks):
    """Clean fist proxies (control renders): one rounded block per gripping hand, oriented by its frame."""
    P = []
    for hand, gi in (ex.get("grip") or {}).items():
        a = np.array(FIST_HALF) * ks
        c = gi["fist"] - gi["u_f"] * 0.01 * ks
        P.append({"t": "ellipsoid", "c": c, "M": np.stack([gi["t"] * a[0], gi["u_f"] * a[1], gi["n_p"] * a[2]], 1),
                  "tag": "fist"})
        # knuckles: a row of four across the through-axis at the front of the fist
        for j in range(4):
            q = gi["fist"] + gi["t"] * a[0] * (-0.75 + 0.5 * j) + gi["u_f"] * a[1] * 0.55 - gi["n_p"] * a[2] * 0.45
            P.append({"t": "ellipsoid", "c": q, "M": np.eye(3) * 0.013 * ks, "tag": "fist"})
        # back of the hand to the wrist
        w = gi["fist"] - (gi["u_f"] * FIST_A + gi["n_p"] * FIST_B) * ks
        P.append({"t": "capsule", "a": w, "b": gi["fist"] - gi["u_f"] * a[1] * 0.3 - gi["n_p"] * a[2] * 0.3,
                  "r": 0.024 * ks, "tag": "fist"})
        # the thumb wrapped over the handle's top (thumb side = the working end)
        th0 = gi["fist"] - gi["t"] * a[0] * 0.9 + gi["n_p"] * a[2] * 0.6
        P.append({"t": "capsule", "a": th0 - gi["u_f"] * a[1] * 0.6, "b": th0 + gi["u_f"] * a[1] * 0.5,
                  "r": 0.011 * ks, "tag": "fist"})
    return P


def grip_checks(W, ex, fig, engaged_items=()):
    """Orientation + limit checks per gripping hand. engaged_items: items whose contact is engaged now
    (their working point must be in FRONT of the body plane)."""
    rows, bad = [], []
    ks = pk(ex)
    who = fig.get("who")
    for hand, gi in (ex.get("grip") or {}).items():
        s = gi["side"]
        sh, el, wr = W[f"{s}_shoulder"], W[f"{s}_elbow"], W[f"{s}_wrist"]
        up_d, fa = _n(el - sh), _n(wr - el)
        ang_hf = math.degrees(math.acos(float(np.clip(np.dot(fa, gi["d"]), -1, 1))))
        flex = math.degrees(math.acos(float(np.clip(np.dot(up_d, fa), -1, 1))))
        t0 = up_d - fa * np.dot(up_d, fa)
        if np.linalg.norm(t0) > math.sin(math.radians(25)):   # near-straight arm: the humerus rotates
            t0 = _n(t0)
            tw = math.degrees(math.atan2(float(np.dot(np.cross(t0, gi["t"]), fa)), float(np.dot(t0, gi["t"]))))
        else:
            tw = 0.0
        # through-axis: the actual fist frame from the final forearm vs the handle
        u_f, t, _ = hand_frame(s, fa, gi["d"])
        thr = math.degrees(math.acos(float(np.clip(np.dot(t, -gi["d"]), -1, 1))))
        tors = seg_dist(el, W["pelvis"], W["neck"])
        dev = tuple(gi.get("dev") or (DEV2 if gi.get("second") else DEV))
        twl = 125.0 if gi.get("second") else 100.0
        rr = [("fist on handle (residual)", gi["residual"] * 100, "cm", 2.0, "max"),
              ("through-axis vs handle", thr, "deg", 20.0, "max"),
              ("handle vs forearm", ang_hf, "deg", (dev[0] - 2, dev[1] + 2), "range"),
              ("elbow flexion", flex, "deg", 150.0, "max"),
              ("forearm twist", tw, "deg", (-twl, twl), "range"),
              ("elbow off the torso axis", tors * 100, "cm", 11.0 * ks, "min")]
        for name, v, unit, lim, kind in rr:
            ok = (v <= lim) if kind == "max" else (v >= lim) if kind == "min" else (lim[0] <= v <= lim[1])
            r = {"who": who, "check": f"{hand} {name}", f"value_{unit}": round(float(v), 2), "limit": lim}
            rows.append(r)
            if not ok:
                bad.append(r)
    holds = fig.get("holds") or {}
    for item in engaged_items:
        if isinstance(item, dict):               # a contact: only working-point blows (a face or a point)
            nm, a = _aff(item)
            if a.get("type") not in ("face", "point"):
                continue
            item, wname = item["prop"], nm
        else:
            wname = props.get(item).get("work")
            if not wname:
                continue
        hand = held_hand(holds, item)
        if hand is None or hand not in (ex.get("grip") or {}):
            continue
        gi = ex["grip"][hand]
        name = wname
        wp = prop_point(item, name, ex["grip_at"][hand], gi["d"], ex.get("flat", {}).get(hand), ks)
        fr = float(np.dot(wp - W["pelvis"], ex["fwd"]))
        fg = float(np.dot(gi["fist"] - W["pelvis"], ex["fwd"]))
        for nm, v in ((f"{item} {name} in front of the body", fr), (f"{hand} fist in front of the hip", fg)):
            r = {"who": who, "check": nm, "value_cm": round(v * 100, 2), "limit": 0.0}
            rows.append(r)
            if v < 0:
                bad.append(r)
    return rows, bad


def face_strike(W, item, name, aff, P, n, k, tilt_deg, side="r"):
    """A FACE affordance at impact (any striking prop): the handle comes from a fist in front of the waist,
    angled down by tilt toward the working end; the face meets the surface at P with its lowest edge
    touching (the face's colliding part never below the surface). Returns (grip point, d, flat)."""
    # the swing plane is vertical through the striking SHOULDER and the target, so at impact the face
    # (pointing -n) lies in the plane of forearm + handle (no roll jump at contact)
    dh = P - W[f"{side}_shoulder"]
    dh = _n(dh - n * np.dot(dh, n))
    a = math.radians(tilt_deg)
    d = _n(dh * math.cos(a) - n * math.sin(a))
    fn = np.asarray(aff.get("normal", [0, 1, 0]), float)        # in the prop frame [d, flat, side]
    if abs(fn[1]) >= abs(fn[2]):
        flat = _n(-n - d * np.dot(-n, d)) * (1 if fn[1] > 0 else -1)
    else:
        side = _n(-n - d * np.dot(-n, d)) * (1 if fn[2] > 0 else -1)
        flat = _n(np.cross(side, d))
    at = P - props.point(item, name, np.zeros(3), d, flat, k)
    if aff.get("collide") is not None:
        low = min(float(np.dot(q - P, n)) for q in props.part_corners(item, aff["collide"], at, d, flat, k))
        at = at - n * low
    return at, d, flat

def flatten_sole(fr, s, rest_pitch):
    """Re-orient side s's foot (toe and heel about the ankle, the lateral axis) so the ankle->toe pitch equals
    the model's rest pitch (= a flat sole for that mesh); the heading stays the motion's."""
    a = np.asarray(fr[f"{s}_ankle"], float)
    v = np.asarray(fr[f"{s}_toe"], float) - a
    hz = np.array([v[0], v[1], 0.0])
    hn = float(np.linalg.norm(hz))
    if hn < 1e-6:
        return 0.0
    p_now = math.atan2(float(v[2]), hn)
    dp = rest_pitch - p_now
    lat = _n(np.cross(UP, hz / hn))                  # rotating about +lat by +dp raises the toe
    for nm in (f"{s}_toe", f"{s}_heel"):
        if nm in fr:
            fr[nm] = a + _rot_axis(np.asarray(fr[nm], float) - a, -lat, dp)
    # splay: a planted foot's toes within SPLAY_MAX of the pelvis facing (a stance may turn the feet; a foot
    # pointing sideways reads as broken)
    if "l_hip" in fr and "r_hip" in fr:
        face = np.cross(np.asarray(fr["l_hip"], float) - np.asarray(fr["r_hip"], float), UP)
        face[2] = 0
        if np.linalg.norm(face) > 1e-6:
            face = _n(face)
            v2 = np.asarray(fr[f"{s}_toe"], float) - a
            h2 = _n(np.array([v2[0], v2[1], 0.0]))
            ang = math.degrees(math.atan2(float(np.cross(face, h2)[2]), float(np.dot(face, h2))))
            if abs(ang) > SPLAY_MAX:
                turn = math.radians(math.copysign(SPLAY_MAX, ang) - ang)
                for nm in (f"{s}_toe", f"{s}_heel"):
                    if nm in fr:
                        fr[nm] = a + _rot_axis(np.asarray(fr[nm], float) - a, UP, turn)
    return math.degrees(dp)


SPLAY_MAX = 35.0        # clamp when a foot plants
SPLAY_CHECK = 50.0      # violation: the hips turned so far over a planted foot that it reads broken


def _rot_axis(v, axis, ang):
    axis = _n(axis)
    return v * math.cos(ang) + np.cross(axis, v) * math.sin(ang) + axis * np.dot(axis, v) * (1 - math.cos(ang))


def sole_rows(fr, s, rest_pitch):
    """(sole pitch vs flat, toe heading vs the pelvis facing) in degrees for side s."""
    a = np.asarray(fr[f"{s}_ankle"], float)
    v = np.asarray(fr[f"{s}_toe"], float) - a
    hz = np.array([v[0], v[1], 0.0])
    pitch = math.degrees(math.atan2(float(v[2]), float(np.linalg.norm(hz)) + 1e-9) - rest_pitch)
    hip = np.asarray(fr["l_hip"], float) - np.asarray(fr["r_hip"], float)
    face = _n(np.cross(hip, UP))
    face[2] = 0
    ang = math.degrees(math.acos(float(np.clip(np.dot(_n(hz), _n(face)), -1, 1)))) if np.linalg.norm(face) > 1e-6 else 0.0
    return pitch, ang


def lock_feet(seq, ground=0.0, vel=0.012, height=0.06, rest_pitch=None):
    """seq: list of joint dicts (one per frame, changed in place). A foot is PLANTED while its ankle is
    within `height` of its lowest and moves < `vel` m/frame; during a planted run its ankle/heel/toe are
    pinned to where the run began and the leg re-solved (hip fixed, 2-bone IK, knee pole forward).
    Feet never go below the ground. Returns per-foot planted-frame counts."""
    out = {}
    for s in ("r", "l"):
        names = [f"{s}_ankle", f"{s}_heel", f"{s}_toe"]
        if any(nm not in seq[0] for nm in names):
            continue
        z = np.array([min(fr[nm][2] for nm in names) for fr in seq])
        planted = []
        for i, fr in enumerate(seq):
            v = np.linalg.norm(fr[f"{s}_ankle"] - seq[i - 1][f"{s}_ankle"]) if i else 0.0
            planted.append(z[i] - ground < height and v < vel)
        pin, cnt = None, 0
        for i, fr in enumerate(seq):
            if planted[i]:
                if pin is None:
                    if rest_pitch is not None:            # a planted foot keeps a FLAT sole (the mesh's rest pitch)
                        flatten_sole(fr, s, rest_pitch.get(s, rest_pitch.get("_", 0.08)) if isinstance(rest_pitch, dict)
                                     else float(rest_pitch))
                    pin = {nm: fr[nm].copy() for nm in names}
                    m = min(p[2] for p in pin.values()) - ground
                    for nm in names:
                        pin[nm][2] -= m              # flat on the ground
                # PIVOT: the hips turned too far over the planted foot -> the foot turns on the ball of the foot
                # (the toe stays, the heel swings) back to SPLAY_MAX, instead of the knee twisting
                if "l_hip" in fr and "r_hip" in fr:
                    face = np.cross(np.asarray(fr["l_hip"], float) - np.asarray(fr["r_hip"], float), UP)
                    face[2] = 0
                    hv = pin[f"{s}_toe"] - pin[f"{s}_ankle"]
                    hv[2] = 0
                    if np.linalg.norm(face) > 1e-6 and np.linalg.norm(hv) > 1e-6:
                        face, hvn = _n(face), _n(hv)
                        ang = math.degrees(math.atan2(float(np.cross(face, hvn)[2]), float(np.dot(face, hvn))))
                        if abs(ang) > SPLAY_CHECK:
                            turn = math.radians(math.copysign(SPLAY_MAX, ang) - ang)
                            ball = pin[f"{s}_toe"].copy()
                            for nm in names:
                                pin[nm] = ball + _rot_axis(pin[nm] - ball, UP, turn)
                hip = fr[f"{s}_hip"]
                l1 = float(np.linalg.norm(fr[f"{s}_knee"] - hip))
                l2 = float(np.linalg.norm(fr[f"{s}_ankle"] - fr[f"{s}_knee"]))
                pole = fr[f"{s}_knee"] - 0.5 * (hip + fr[f"{s}_ankle"])
                kn, an = ik2(hip, l1, l2, pin[f"{s}_ankle"], pole if np.linalg.norm(pole) > 1e-4 else UP)
                fr[f"{s}_knee"] = kn
                fr[f"_planted_{s}"] = True
                off = an - pin[f"{s}_ankle"]
                for nm in names:
                    fr[nm] = pin[nm] + off
                cnt += 1
            else:
                pin = None
                m = min(fr[nm][2] for nm in names) - ground
                if m < 0:
                    for nm in names:
                        fr[nm] = fr[nm] - np.array([0, 0, m])
        out[s] = cnt
    return out


def find_set_prop(set_spec, name):
    name = name.lower()
    for pr in (set_spec or {}).get("props", []):
        if pr.get("name", "").lower() == name:
            return pr
    for pr in (set_spec or {}).get("props", []):
        if name in pr.get("name", "").lower():
            return pr
    return None


def surface_point(set_spec, c):
    """(point, normal, set prop) of a set prop's top affordance; "at" = [x, y] on it."""
    pr = find_set_prop(set_spec, c["to"])
    if pr is None:
        raise KeyError(f"contact target {c['to']!r} is not a set prop")
    if c.get("surface", "top") != "top":
        raise NotImplementedError("only 'top' surfaces so far")
    P, n, _ = props.set_top(pr, c.get("at"))
    return P, n, pr

def all_set_tops(set_spec):
    out = []
    for pr in (set_spec or {}).get("props", []):
        x, y, z, w, d, h = pr["box"]
        yaw = math.radians(pr.get("yaw", 0))
        hw, hd = w / 2, d / 2
        if "anvil" in pr.get("name", "").lower():
            hw, hd = w * 0.38, d * 0.4               # the anvil's face block
        out.append((pr.get("name", ""), z + h, (x, y, hw, hd, yaw)))
    return out


def _move_figure(W, v):
    for k in W:
        if not k.startswith("_"):
            W[k] = W[k] + v


# ---------------------------------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------------------------------

def held_hand(holds: dict, item: str):
    """The hand holding `item`; a state of the held prop is the same prop (10-23: the contact's 'sword__finished'
    against the grip's 'sword' failed the killing thrust's build every run, so it was staged from stale joints)."""
    return next((h for h, it in holds.items() if it == item), None) or next(
        (h for h, it in holds.items() if str(it).split("__")[0] == str(item).split("__")[0]), None)


def solve_contacts(W, ex, fig, contacts, set_spec, anchors, two_handed=False, creature_prims=None):
    """CLEANUP: pose fig's held props onto their contact targets by AFFORDANCE TYPE (changes W and ex in
    place): face -> flush on a surface (strike), point -> onto a surface or an anchor (tip, jaws, rim),
    edge -> under an anchor (a parry; the nearest creature part touches). The fist then grips the handle
    (solve_grip); if the grip can't be natural, the figure steps toward the target. Returns a log."""
    k = pk(ex)
    holds = fig.get("holds") or {}
    ex.setdefault("flat", {})
    log = []
    for c in contacts:
        item = c["prop"]
        hand = held_hand(holds, item)
        if hand is None:
            raise ValueError(f"{fig.get('who')} does not hold a {item} (holds {holds})")
        s = "r" if hand == "right" else "l"
        name, aff = _aff(c)
        aff = dict(aff)
        typ = aff.get("type")
        to = c["to"]
        if ":" in to:
            who, an = to.split(":", 1)
            P = np.asarray(anchors[who][an], float)
            n = None
        else:
            P, n, _ = surface_point(set_spec, c)
        tilt = float(c.get("tilt", aff.get("default_tilt_deg", 0.0)))

        def place():
            if typ == "face":
                if n is None:
                    raise ValueError(f"a face contact needs a surface, got {to!r}")
                return face_strike(W, item, name, aff, P, n, k, tilt, s)
            if typ == "point":
                half = aff.get("half", 0.0) * k
                tgt = P + (n * half if n is not None else 0)
                if c.get("_push") and n is None and hand in ex.get("hold", {}):   # follow-through
                    tgt = tgt + _n(ex["hold"][hand]) * float(c["_push"])
                along = aff["at"][0] * k
                # per-frame cleanup (fast): start from the motion's own (interpolated) prop axis
                d = _n(ex["hold"][hand]) if ex.get("_fast") and hand in ex.get("hold", {}) else _n(tgt - W[f"{s}_shoulder"])
                if n is not None:                  # rest along the surface, not into it: dip <= 20 deg
                    dn = float(np.dot(d, n))
                    if dn < -math.sin(math.radians(20)):
                        dh = _n(d - n * dn)
                        d = _n(dh * math.cos(math.radians(20)) - n * math.sin(math.radians(20)))
                # the axis is otherwise free: the most natural grip
                d = choose_tip_axis(W, s, tgt, d, along, ex, n=n, dev=aff.get("forearm_angle"))
                return tgt - d * along, d, ex["flat"].get(hand)
            if typ == "lie":                       # held, but resting flat on a support surface (every frame)
                if n is None:
                    raise ValueError(f"a lie contact needs a surface, got {to!r}")
                if c.get("axis") is not None:      # data: grip -> far end direction on the surface
                    d = np.asarray(c["axis"], float)
                elif c.get("_d") is not None:      # keep the axis found at the first solve (no drift)
                    d = np.asarray(c["_d"], float)
                else:
                    d = P - W[f"{s}_shoulder"]
                d = _n(d - n * np.dot(d, n))
                c["_d"] = d.tolist()
                at = P - d * aff.get("along", 0.4) * k + n * aff.get("half", 0.0) * k
                return at, d, n.copy()
            if typ == "edge":
                d, side, flat = prop_axes(ex["hold"][hand], ex["flat"].get(hand))
                hs, hf = aff.get("half_side", 0.0) * k, aff.get("half_flat", 0.0) * k
                ref = UP if n is None else n
                if abs(np.dot(side, ref)) > abs(np.dot(flat, ref)):
                    u, half = side * np.sign(np.dot(side, ref)), hs
                else:
                    u, half = flat * np.sign(np.dot(flat, ref)), hf
                b0, b1 = aff["from"], aff["to"]
                if c.get("along") is not None:
                    along = float(c["along"]) * k
                else:                              # where the grip ends up nearest the shoulder
                    t = float(np.dot(P - u * half - W[f"{s}_shoulder"], d))
                    along = float(np.clip(t, (b0 + 0.08) * k, (b1 - 0.05) * k))
                at = P - u * half - d * along
                who_ = to.split(":", 1)[0]
                if creature_prims and who_ in creature_prims:      # the NEAREST creature part touches
                    for _ in range(3):
                        pen = body_penetration(item, at, d, flat, k, creature_prims[who_])
                        at = at + u * pen
                return at, d, flat
            raise NotImplementedError(f"affordance type {typ!r} ({item}.{name})")

        at, d, flat = place()
        moved = 0.0
        # the grip must be natural too: if the fist can't reach with the wrist in range, the FEET move
        # (small steps toward the target), never the wrist
        rng_ = tuple(aff.get("forearm_angle") or DEV)    # e.g. a thrust holds the blade near the arm line
        for _ in range(25):
            Wt = _copyW(W)
            # (optional) forearm-in-swing-plane preference; off: it traded the wrist range for the plane
            pl = _n(np.cross(d, n)) if (typ == "face" and n is not None and c.get("plane_pref")) else None
            gi = solve_grip(Wt, s, at, d, ex, dev=rng_, pose_pole=bool(ex.get("_fast")), plane=pl)
            fa = _n(Wt[f"{s}_wrist"] - Wt[f"{s}_elbow"])
            ang = math.degrees(math.acos(float(np.clip(np.dot(fa, d), -1, 1))))
            if (gi["residual"] < 0.008 and rng_[0] - 1 <= ang <= rng_[1] + 1) or fig.get("anchor_fixed") is True:
                break
            v = at - W[f"{s}_shoulder"]
            v[2] = 0.0
            if np.linalg.norm(v) < 0.05 or moved > 0.8:
                break
            _move_figure(W, _n(v) * 0.04)
            ex["moved"] = ex.get("moved", np.zeros(3)) + _n(v) * 0.04
            moved += 0.04
            at, d, flat = place()
        W.update(Wt)
        ex["hold"][hand] = d
        ex["flat"][hand] = flat
        ex.setdefault("grip_at", {})[hand] = at
        log.append(f"{fig.get('who')} {item}.{name} -> {to}: moved {moved:.2f} m, grip residual {gi['residual'] * 100:.1f} cm")
    regrip_two_handed(W, ex, fig, two_handed)
    return log

def regrip_two_handed(W, ex, fig, two_handed):
    """Two-handed grip: the second (left) fist on the same handle, below the first, same through-axis."""
    ex.pop("support", None)
    if not two_handed:
        return
    holds = fig.get("holds") or {}
    item = holds.get("right")
    if not item or not props.has_grip(item):
        return
    k = pk(ex)
    at = ex.get("grip_at", {}).get("right", W["r_hand"])
    d = _n(ex["hold"]["right"])
    # pose data "second": "flat": the second hand SUPPORTS the blade's flat (an overhead block: one hand on
    # the hilt, the open palm under the blade) instead of a second fist on the grip
    if (props.pose3d().get(fig.get("pose")) or {}).get("second") == "flat" and props.affordance(item, "flat"):
        fl = ex.get("flat", {}).get("right")
        _, side_, flat_ = prop_axes(d, fl)
        along = float(props.affordance(item, "flat").get("along", 0.45)) * 0.8
        P = at + d * along * k
        n_ = flat_ if np.dot(W["l_shoulder"] - P, flat_) > 0 else -flat_
        tgt = P + n_ * 0.04 * k
        Wt = _copyW(W)
        res = ik_hand(Wt, "l", tgt)
        if res < 0.03:
            W.update(Wt)
            ex["support"] = {"hand": "left", "item": item, "on": "flat", "point": P, "residual": res}
            ex.setdefault("log", []).append(f"second hand supports the {item} flat ({res * 100:.1f} cm)")
        else:
            ex.setdefault("log", []).append(f"blade-flat support unreachable ({res * 100:.1f} cm): one-handed")
        return
    lo = props.affordance(item, "grip")["from"]
    tgt = at + d * max(lo * 0.9, -0.09) * k
    Wt = _copyW(W)
    ext = dict(ex, grip=dict(ex.get("grip", {})))
    sd = props.affordance(item, "grip").get("second_dev")   # per prop: the pommel hand's natural range
    if (props.pose3d().get(fig.get("pose")) or {}).get("second") == "stack":
        sd = (0.0, 180.0)            # both hands STACKED on the grip (a point-down quench): forearms near the blade
    gi = solve_grip(Wt, "l", tgt, d, ext, key="left2", second=True, dev=tuple(sd) if sd else None)
    if gi["residual"] > 0.015:            # the second hand can't reach the grip in this pose: one-handed
        ex.setdefault("log", []).append(f"two-handed grip released (second fist {gi['residual'] * 100:.1f} cm short)")
        return
    W.update(Wt)
    ex["grip"] = ext["grip"]
    ex.setdefault("grip_at", {})["left2"] = tgt

def _seg_seg(a0, a1, b0, b1, n=24):
    """Approximate distance between two segments (sampled)."""
    ts = np.linspace(0, 1, n)[:, None]
    A = a0 + (a1 - a0) * ts
    return min(seg_dist(p, b0, b1) for p in A)


def body_penetration(item, at, d, flat, k, prims):
    """Deepest penetration (negative = into) of a creature's capsules into a prop's working edge (or its
    handle axis when it has no edge)."""
    d, side, flat = prop_axes(d, flat)
    affs = props.get(item).get("affordances", {})
    e = next((a for a in affs.values() if a.get("type") == "edge"), None)
    if e is not None:
        a0, a1 = at + d * e["from"] * k, at + d * e["to"] * k
        half = e.get("half_flat", 0.0) * k
    else:
        a0, a1 = handle_seg(item, at, d, k)
        half = affs.get("grip", {}).get("radius", 0.016) * k
    best = 1e9
    for p in as_capsules(prims):
        if True:
            g = _seg_seg(np.asarray(p["a"], float), np.asarray(p["b"], float), a0, a1) - float(p["r"]) - half
            best = min(best, g)
    return best

def check(W, ex, fig, contacts, set_spec, anchors, two_handed=False, creature_prims=None):
    """Position numbers for one posed figure: grips (fist to handle axis) and contacts by affordance type
    (gap +, penetration -, face angle). Returns (rows, violations)."""
    k = pk(ex)
    holds = fig.get("holds") or {}
    who_ = fig.get("who")
    rows, bad = [], []

    def row(name, v, lim, unit="cm", ok=None):
        r = {"who": who_, "check": name, f"value_{unit}": round(float(v), 2), "limit": lim}
        rows.append(r)
        if ok is not None and not ok:
            bad.append(r)
    for hand, item in holds.items():
        if not props.has_grip(item):
            continue
        s = "r" if hand == "right" else "l"
        at = ex.get("grip_at", {}).get(hand, W[f"{s}_hand"])
        a, b = handle_seg(item, at, ex["hold"][hand], k)
        g = seg_dist(W[f"{s}_hand"], a, b)
        row(f"grip {hand} {item}", g * 100, LIMITS["grip"] * 100, ok=g <= LIMITS["grip"])
        if two_handed and hand == "right" and (ex.get("support") or {}).get("item") == item:
            sp = ex["support"]
            g2 = float(np.linalg.norm(W["l_hand"] - (np.asarray(sp["point"]) + 0 * at)))
            row(f"left palm on the {item} flat", g2 * 100, 6.0, ok=g2 <= 0.06 * k)
        if two_handed and hand == "right" and "left2" in (ex.get("grip") or {}):
            g2 = seg_dist(W["l_hand"], a - _n(ex["hold"][hand]) * 0.02 * k, b)
            row(f"grip left (two-handed) {item}", g2 * 100, LIMITS["grip"] * 100, ok=g2 <= LIMITS["grip"])
    for c in contacts:
        item = c["prop"]
        name, aff = _aff(c)
        typ = aff.get("type")
        hand = held_hand(holds, item)
        s = "r" if hand == "right" else "l"
        at = ex.get("grip_at", {}).get(hand, W[f"{s}_hand"])
        d, side, flat = prop_axes(ex["hold"][hand], ex.get("flat", {}).get(hand))
        to = c["to"]
        if ":" in to:
            who, an = to.split(":", 1)
            P = np.asarray(anchors[who][an], float)
            if typ == "edge" and creature_prims and who in creature_prims:
                gap = body_penetration(item, at, d, flat, k, creature_prims[who])   # the nearest body part
            elif typ == "edge":
                a0, a1 = at + d * aff["from"] * k, at + d * aff["to"] * k
                gap = seg_dist(P, a0, a1) - aff.get("half_flat", 0.0) * k
            else:
                P = P + d * float(c.get("_push", 0.0))           # follow-through target
                gap = float(np.linalg.norm(prop_point(item, name, at, d, flat, k) - P))
            row(f"contact {item}.{name} -> {to}", gap * 100, LIMITS["gap"] * 100, ok=abs(gap) <= LIMITS["gap"])
            if creature_prims and who in creature_prims and not c.get("penetrate") and typ != "edge":
                pen = body_penetration(item, at, d, flat, k, creature_prims[who])
                row(f"{who} body vs {item} (gap + / penetration -)", pen * 100, -LIMITS["pen"] * 100,
                    ok=pen >= -LIMITS["pen"])
            continue
        P, n, _ = surface_point(set_spec, c)
        if typ == "face":
            gap = float(np.dot(prop_point(item, name, at, d, flat, k) - P, n))
            row(f"contact {item}.{name} -> {to} gap", gap * 100, LIMITS["gap"] * 100, ok=abs(gap) <= LIMITS["gap"])
            if aff.get("collide") is not None:
                pen = min(float(np.dot(q - P, n)) for q in props.part_corners(item, aff["collide"], at, d, flat, k))
                row(f"{item} {name} part deepest point vs surface", pen * 100, -LIMITS["pen"] * 100,
                    ok=pen >= -LIMITS["pen"])
            fn = np.asarray(aff.get("normal", [0, 1, 0]), float)
            fw = _n(d * fn[0] + flat * fn[1] + side * fn[2])
            ang = math.degrees(math.acos(float(np.clip(np.dot(fw, -n), -1, 1))))
            lim = float(aff.get("max_tilt_deg", LIMITS["angle"]))
            row(f"{item} {name} angle to surface", ang, lim, unit="deg", ok=ang <= lim)
        elif typ == "lie":
            centre = at + d * aff.get("along", 0.4) * k
            gap = float(np.dot(centre - P, n)) - aff.get("half", 0.0) * k
            row(f"{item} lies on {to}: gap", gap * 100, LIMITS["gap"] * 100, ok=abs(gap) <= LIMITS["gap"])
            ang = math.degrees(math.acos(float(np.clip(abs(np.dot(flat, n)), -1, 1))))
            row(f"{item} lies flat (tilt)", ang, 5.0, unit="deg", ok=ang <= 5.0)
            off = float(np.linalg.norm((centre - P) - n * np.dot(centre - P, n)))
            row(f"{item} lies where planned (offset)", off * 100, 2.0, ok=off <= 0.02)
        else:
            q = prop_point(item, name, at, d, flat, k)
            gap = float(np.dot(q - P, n)) - aff.get("half", 0.0) * k
            row(f"contact {item}.{name} -> {to} gap", gap * 100, LIMITS["gap"] * 100, ok=abs(gap) <= LIMITS["gap"])
    return rows, bad


# ---------------------------------------------------------------------------------------------------
# Creature volumes (rigged creatures: per-bone capsules from rig.json radii)
# ---------------------------------------------------------------------------------------------------
HUMAN_BODY_R = {"pelvis": 0.14, "spine_mid": 0.14, "neck": 0.09, "head": 0.11, "r_shoulder": 0.07,
                "l_shoulder": 0.07, "r_elbow": 0.05, "l_elbow": 0.05, "r_hip": 0.08, "l_hip": 0.08,
                "r_knee": 0.06, "l_knee": 0.06, "r_ankle": 0.05, "l_ankle": 0.05}


def creature_capsules(J, parents, radii, scale=1.0, skip=("wing_",)):
    """Capsule prims (t, a, b, r) along each bone (parent joint -> joint) with the rig's measured radius."""
    out = []
    for j, p in parents.items():
        if p is None or j not in J or p not in J or j not in radii or j.startswith(skip):
            continue
        out.append({"t": "capsule", "a": np.asarray(J[p], float), "b": np.asarray(J[j], float),
                    "r": float(radii[j]) * scale, "tag": "body"})
    return out


def as_capsules(prims):
    """Any creature prim list -> capsules: rigged models give capsules already; proxy creatures are built from
    ellipsoids ({"c", "M"}, M columns = semi-axes) -> a capsule along the longest semi-axis with radius =
    the middle one (inscribed-ish, so a check never flags air). Other shapes are ignored."""
    out = []
    for c in prims or ():
        if "a" in c and "b" in c and "r" in c:
            out.append(c)
        elif "c" in c and "M" in c:
            M = np.asarray(c["M"], float).reshape(3, 3)
            ln = np.linalg.norm(M, axis=0)
            o = np.argsort(ln)[::-1]
            d = M[:, o[0]] / max(ln[o[0]], 1e-9)
            r = float(ln[o[1]])
            h = max(float(ln[o[0]]) - r, 0.0)
            ctr = np.asarray(c["c"], float)
            out.append({"t": "capsule", "a": ctr - d * h, "b": ctr + d * h, "r": r, "tag": c.get("tag", "body")})
    return out


def human_in_creature(W, caps, k=1.0):
    """Deepest penetration (m, positive = inside) of a human body joint sphere into creature capsules."""
    caps = as_capsules(caps)
    worst, wj = 0.0, None
    for n, r in HUMAN_BODY_R.items():
        if n not in W:
            continue
        p = np.asarray(W[n], float)
        for c in caps:
            d = seg_dist(p, c["a"], c["b"]) - c["r"] - r * k
            if -d > worst:
                worst, wj = -d, n
    return worst, wj


def push_out(W, caps, k=1.0, max_m=1.5):
    """Move a human (whole figure, horizontally) out of creature volumes: away from the creature's
    nearest capsule axis, in 2 cm steps. Returns metres moved."""
    caps = as_capsules(caps)
    moved = 0.0
    for _ in range(int(max_m / 0.02)):
        dep, j = human_in_creature(W, caps, k)
        if dep <= 0.005:
            break
        p = np.asarray(W[j], float)
        c = min(caps, key=lambda c: seg_dist(p, c["a"], c["b"]) - c["r"])
        ab = c["b"] - c["a"]
        t = float(np.clip(np.dot(p - c["a"], ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
        v = p - (c["a"] + ab * t)
        v[2] = 0.0
        if np.linalg.norm(v) < 1e-6:
            v = np.array([1.0, 0, 0])
        _move_figure(W, _n(v) * 0.02)
        moved += 0.02
    return moved
