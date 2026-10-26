"""Blockout rigs: the 3D mannequin, the dragon proxy, held props and the shot camera, as pure numpy.

Shared by tools/blockout.py (the driver, any python with numpy) and tools/blockout_scene.py (inside
Blender), so the joints the skeleton pass is drawn from are the joints the meshes are built on.

World: metres, Z up, ground at z = 0. A figure's "facing" is a compass angle in degrees, CCW from +X
(0 = +X, 90 = +Y). A body frame is (fwd, left, up).

HUMANS: the 52 named poses of tools/poses.py are LIFTED to 3D from their 2D authoring specs (SPECS): a
spec's screen angles (0 = forward, 90 = down, clockwise on screen) become directions in the body's
sagittal plane (forward/up), and the shoulders and hips get their true width along the body's left axis.
Poses authored facing the camera (yaw < 35: standing_front, arms_crossed, cheering, raise_cup, dancing)
are lifted into the plane the 2D picture was drawn in instead (a blend of forward and left). A few poses
get 3D fixes on top (POSE3D): two-handed grips (the left hand IK'd onto the hilt), a helm held between
the hands, lying on the back. Output: named 3D joints, including the COCO-18 face points, so any camera
gives an OpenPose skeleton.

CREATURES: a dragon proxy (body, neck, head, 4 legs, 2 wings, tail) in a few poses (DRAGON_POSES),
built from ellipsoids, capsules and membrane polygons. It is drawn upright like Vorgrath's sheet (a
bipedal stance, forelegs used as arms). Named anchor points: "claws" (between the fore claws), "claw_r",
"claw_l", "head", "jaw", "chest".
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from poses import BODY, SPECS, KP_NAMES  # noqa: E402  (pure numpy at import)
import blockout_props as _props  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])


def _n(v):
    v = np.asarray(v, float)
    d = float(np.linalg.norm(v))
    return v / d if d > 1e-9 else v


def _v2(a):
    r = math.radians(a)
    return np.array([math.cos(r), math.sin(r)])


def _rot2(v, a):
    r = math.radians(a)
    c, s = math.cos(r), math.sin(r)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def frame(facing_deg: float):
    r = math.radians(facing_deg)
    fwd = np.array([math.cos(r), math.sin(r), 0.0])
    left = np.array([-math.sin(r), math.cos(r), 0.0])
    return fwd, left, UP.copy()


def rodrigues(v, axis, deg):
    axis = _n(axis)
    t = math.radians(deg)
    return v * math.cos(t) + np.cross(axis, v) * math.sin(t) + axis * np.dot(axis, v) * (1 - math.cos(t))


def ik2(root, l1, l2, target, pole):
    """Two-bone IK: the middle joint for a chain root -(l1)- mid -(l2)- end reaching target, bending
    toward the pole direction. Returns (mid, end); end is clamped to reach."""
    d = target - root
    dist = float(np.linalg.norm(d))
    dist = min(max(dist, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
    ax = _n(d)
    end = root + ax * dist
    a = (l1 * l1 - l2 * l2 + dist * dist) / (2 * dist)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    p = pole - ax * np.dot(pole, ax)
    p = _n(p) if np.linalg.norm(p) > 1e-6 else _n(np.cross(ax, UP + 1e-3))
    return root + ax * a + p * h, end


# --------------------------------------------------------------------------------------------------
# Humans
# --------------------------------------------------------------------------------------------------

# 3D fixes on top of the lifted 2D specs.
#   grip: "both" = left hand IK'd onto the right hand's grip (two-handed sword);
#         "between" = both hands either side of an object held in front.
#   hold: per hand, the held item's axis: a screen angle in the sagittal plane (like the specs) or a
#         3D body-frame vector [fwd, left, up]; default = the forearm's direction.
#   mirror: lift the mirror image (lying on the back reads face-up), facing turned 180.
# 3D fixes per pose live in DATA: tools/stage_props/pose3d.json (grip: both|helm|head, hold axes, mirror, hug)
from blockout_props import pose3d as _pose3d  # noqa: E402
POSE3D = _pose3d()

# body-unit radii (standing height ~1.05 body units)
R = dict(head=0.062, neck=0.028, uarm=0.026, farm=0.022, thigh=0.040, shin=0.030, hand=0.024,
         foot=0.022)


def _spec(name):
    if name not in SPECS:
        raise KeyError(f"unknown pose {name!r}; see `poses.py list`")
    desc, anchor, spec = SPECS[name]
    return desc, anchor, dict(spec)


def human_joints(pose: str, height: float = 1.8):
    """Joints of `pose` in the body frame (fwd = +X, left = +Y, up = +Z), metres, feet NOT yet
    grounded. Returns (joints dict name -> np.array(3), anchor, extras dict)."""
    desc, anchor, s = _spec(pose)
    fix = POSE3D.get(pose, {})
    if fix.get("mirror"):
        for k in ("ra", "la", "rl", "ll"):
            if k in s:
                s[k] = [180 - s[k][0], 180 - s[k][1]] + list(s[k][2:])
        s["t"] = 180 - s.get("t", 90)
        s["tilt"] = -s.get("tilt", 0)
    yaw = s.get("yaw", 45)
    fwd, left, up = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), UP
    # screen x axis of the 2D drawing, in the body frame
    if yaw >= 35:
        X = fwd
    else:
        ry = math.radians(yaw)
        X = _n(math.sin(ry) * fwd + math.cos(ry) * left)

    def sag(v2):          # torso/head: always the sagittal plane
        return v2[0] * fwd - v2[1] * up

    def lim(v2):          # limbs: the plane the pose was drawn in
        return v2[0] * X - v2[1] * up

    B = BODY
    t = s.get("t", 90)
    J = {}
    N = np.zeros(3)
    P = N + sag(_v2(t)) * B["torso"] * s.get("tk", 1.0)
    J["neck"], J["pelvis"] = N, P
    J["r_shoulder"] = N - left * B["shoulder"] * 0.95
    J["l_shoulder"] = N + left * B["shoulder"] * 0.95
    J["r_hip"] = P - left * B["hip"] * 1.35
    J["l_hip"] = P + left * B["hip"] * 1.35

    def limb(root, seg, l1, l2):
        f1 = seg[2] if len(seg) > 2 else 1.0
        f2 = seg[3] if len(seg) > 3 else 1.0
        j1 = root + lim(_v2(seg[0])) * l1 * f1
        j2 = j1 + lim(_v2(seg[1])) * l2 * f2
        return j1, j2

    J["r_elbow"], J["r_wrist"] = limb(J["r_shoulder"], s.get("ra", [95, 92]), B["uarm"], B["farm"])
    J["l_elbow"], J["l_wrist"] = limb(J["l_shoulder"], s.get("la", [85, 88]), B["uarm"], B["farm"])
    J["r_knee"], J["r_ankle"] = limb(J["r_hip"], s.get("rl", [92, 90]), B["thigh"], B["shin"])
    J["l_knee"], J["l_ankle"] = limb(J["l_hip"], s.get("ll", [88, 90]), B["thigh"], B["shin"])
    # a sagittal lift puts both legs in the same plane: keep the hip width at the feet
    # head
    u2 = -_v2(t)
    hu2 = _rot2(u2, s.get("tilt", 0))
    hs2 = np.array([-hu2[1], hu2[0]])
    head_up = _n(sag(hu2))
    head_fwd = _n(sag(hs2))
    hyaw = s.get("hyaw", yaw)
    turn = hyaw - yaw
    if abs(turn) > 90:
        head_fwd = -head_fwd
    elif abs(turn) > 5 and yaw >= 35:
        head_fwd = rodrigues(head_fwd, head_up, -turn * 0.5)
    if yaw < 35:                                   # drawn facing the camera: face along X's normal
        head_fwd = _n(head_fwd)
    C = N + head_up * B["head"]
    J["head"] = C
    J["head_top"] = C + head_up * 0.075
    hl = _n(np.cross(head_up, head_fwd))           # head's left
    r = R["head"]
    J["nose"] = C + head_fwd * r * 1.05 - head_up * r * 0.15
    J["r_eye"] = C + head_fwd * r * 0.85 + head_up * r * 0.22 - hl * r * 0.38
    J["l_eye"] = C + head_fwd * r * 0.85 + head_up * r * 0.22 + hl * r * 0.38
    J["r_ear"] = C - hl * r * 1.0 + head_up * r * 0.05
    J["l_ear"] = C + hl * r * 1.0 + head_up * r * 0.05
    J["_head_fwd"], J["_head_up"] = head_fwd, head_up

    # 3D fixes: two-handed grips
    grip = fix.get("grip")
    l_pole = -fwd * 0.3 - up + left * 0.6
    if grip == "both":
        hd = _hold_dir(J, "right", fix, sag, lim)
        tgt = J["r_wrist"] - hd * 0.06
        J["l_elbow"], J["l_wrist"] = ik2(J["l_shoulder"], B["uarm"], B["farm"], tgt, l_pole)
    elif grip == "between":                        # an object held in front between both hands
        c = 0.5 * (J["r_wrist"] + J["l_wrist"])
        c = c + left * (0.0 - np.dot(c, left))       # centred on the body's midline
        J["between"] = c
        r_pole = -fwd * 0.3 - up - left * 0.6
        J["r_elbow"], J["r_wrist"] = ik2(J["r_shoulder"], B["uarm"], B["farm"], c - left * 0.07, r_pole)
        J["l_elbow"], J["l_wrist"] = ik2(J["l_shoulder"], B["uarm"], B["farm"], c + left * 0.07, l_pole)
    elif grip == "head":
        # both hands on the sides of something worn on the head
        r_pole = -fwd * 0.2 - up * 0.6 - left * 0.8
        J["r_elbow"], J["r_wrist"] = ik2(J["r_shoulder"], B["uarm"], B["farm"], C - left * 0.085 - up * 0.01, r_pole)
        J["l_elbow"], J["l_wrist"] = ik2(J["l_shoulder"], B["uarm"], B["farm"], C + left * 0.085 - up * 0.01, r_pole * np.array([1, -1, 1]))
    elif fix.get("hug"):
        # arms wrap toward the midline in front of the chest
        for side, sg in (("r", -1), ("l", 1)):
            tgt = J["neck"] + fwd * 0.16 + (J["pelvis"] - J["neck"]) * 0.35 - left * sg * 0.05
            pole = left * sg - up * 0.3
            J[f"{side}_elbow"], J[f"{side}_wrist"] = ik2(J[f"{side}_shoulder"], B["uarm"], B["farm"],
                                                         tgt, pole)
    # hands, feet
    for side in ("r", "l"):
        fa = _n(J[f"{side}_wrist"] - J[f"{side}_elbow"])
        J[f"{side}_hand"] = J[f"{side}_wrist"] + fa * 0.035
        shin = _n(J[f"{side}_ankle"] - J[f"{side}_knee"])
        # the foot points forward, at right angles to the shin, in the shin's sagittal plane
        foot = fwd - shin * np.dot(fwd, shin)
        if np.linalg.norm(foot) < 0.3:
            foot = -up - shin * np.dot(-up, shin)
        J[f"{side}_toe"] = J[f"{side}_ankle"] + _n(foot) * 0.075 - shin * 0.012
        J[f"{side}_heel"] = J[f"{side}_ankle"] - _n(foot) * 0.018 + shin * 0.03
    # metres
    std_h = _standing_height()
    k = height / std_h
    out = {n: (v * k if not n.startswith("_") else v) for n, v in J.items()}
    extras = {"hold": {h: _hold_dir(J, h, fix, sag, lim) for h in ("right", "left")},
              "scale": k, "mirror": bool(fix.get("mirror")), "desc": desc}
    return out, anchor, extras


def _hold_dir(J, hand, fix, sag, lim):
    side = "r" if hand == "right" else "l"
    h = fix.get("hold", {}).get(hand)
    if h is None:
        return _n(J[f"{side}_wrist"] - J[f"{side}_elbow"])
    if isinstance(h, (int, float)):
        return _n(sag(_v2(h)))
    v = np.asarray(h, float)
    return _n(v[0] * np.array([1.0, 0, 0]) + v[1] * np.array([0, 1.0, 0]) + v[2] * UP)


_STD = None


def _standing_height():
    global _STD
    if _STD is None:
        _, _, s = _spec("standing")
        B = BODY
        # neck -> head top + neck -> sole, standing straight
        _STD = (B["head"] + 0.075) + B["torso"] + B["thigh"] * 0.999 + B["shin"] * 0.999 + 0.012 + R["foot"]
    return _STD


def place_human(fig: dict):
    """World joints for a figure dict {pose, at:[x,y], facing, height, z?}. Bottom-anchored poses
    stand on the ground (lowest body point at z=0); centre-anchored ones (airborne, carried) are
    centred at z (default 1.2 m) unless placed by an attach."""
    H = float(fig.get("height", 1.8))
    J, anchor, ex = human_joints(fig.get("pose", "standing"), H)
    facing = float(fig.get("_facing", 0.0)) + (180.0 if ex["mirror"] else 0.0)
    fwd, left, up = frame(facing)
    M = np.stack([fwd, left, up], 1)               # body -> world rotation
    roll = float(fig.get("rotate", 0) or 0)        # extra pitch about the body's left axis
    W = {}
    for n, v in J.items():
        if n.startswith("_"):
            W[n] = M @ v
        else:
            W[n] = M @ v
    if roll:
        piv = W["pelvis"].copy()
        for n in list(W):
            if n.startswith("_"):
                W[n] = rodrigues(W[n], left, roll)
            else:
                W[n] = piv + rodrigues(W[n] - piv, left, roll)
    pts = np.array([v for n, v in W.items() if not n.startswith("_")])
    x, y = fig.get("at", [0.0, 0.0])[:2]
    if fig.get("_attach") is not None:
        # the attach point is where the hands (hanging) or the waist (carried) go
        a = np.asarray(fig["_attach"], float)
        key = fig.get("_attach_by", "pelvis")
        if key == "hands":
            ref = 0.5 * (W["r_shoulder"] + W["l_shoulder"])
        else:
            ref = W[key]
        off = a - ref
    elif anchor == "bottom" and "z" not in fig:
        zmin = min(float(v[2]) for n, v in W.items() if not n.startswith("_")) - R["foot"] * ex["scale"]
        off = np.array([x, y, -zmin + float(fig.get("ground_z", 0.0))])   # ground_z: standing on a ledge/step
    else:
        c = 0.5 * (pts.min(0) + pts.max(0))
        off = np.array([x - c[0], y - c[1], float(fig.get("z", 1.2)) - c[2]])
        off[0], off[1] = x - W["pelvis"][0], y - W["pelvis"][1]
    for n in W:
        if not n.startswith("_"):
            W[n] = W[n] + off
    hold = {h: M @ d for h, d in ex["hold"].items()}
    if roll:
        hold = {h: rodrigues(d, left, roll) for h, d in hold.items()}
    return W, {"hold": hold, "scale": ex["scale"], "fwd": fwd, "left": left, "anchor": anchor}


def aim(W: dict, ex: dict, hand: str, target, height: float = 1.8):
    """Straighten `hand`'s arm toward a world point (a thrust or a point at something); the held item's
    axis follows the arm. Changes W and ex in place."""
    side = "r" if hand == "right" else "l"
    k = ex["scale"]
    sh = W[f"{side}_shoulder"]
    d = _n(np.asarray(target, float) - sh)
    l1, l2 = BODY["uarm"] * k, BODY["farm"] * k
    reach = min(float(np.linalg.norm(np.asarray(target, float) - sh)), (l1 + l2) * 0.97)
    pole = -UP * 0.6 + (-ex["left"] if side == "r" else ex["left"])
    W[f"{side}_elbow"], W[f"{side}_wrist"] = ik2(sh, l1, l2, sh + d * reach, pole)
    W[f"{side}_hand"] = W[f"{side}_wrist"] + d * 0.035 * k
    ex["hold"][hand] = d


def coco18(W: dict, cam_pos=None) -> list:
    """COCO-18 world points (None where a face point turns away from the camera)."""
    out = []
    for n in KP_NAMES:
        p = W.get(n)
        if p is None:
            out.append(None)
            continue
        if cam_pos is not None and n in ("nose", "r_eye", "l_eye", "r_ear", "l_ear"):
            nrm = _n(p - W["head"])
            view = _n(np.asarray(cam_pos) - p)
            lim = -0.05 if n.endswith("ear") else 0.05
            if float(np.dot(nrm, view)) < lim:
                out.append(None)
                continue
        out.append(p)
    return out


def human_prims(W: dict, scale: float, wear=(), hold_items=None, hold_dirs=None, fwd=None, left=None,
                hold_flats=None, hold_at=None, fists=()):
    """Mannequin primitives: [{"t": "capsule", "a", "b", "r"}, {"t": "ellipsoid", "c", "M"}...]."""
    k = scale
    P = []

    def cap(a, b, r, tag="body"):
        P.append({"t": "capsule", "a": W[a] if isinstance(a, str) else a,
                  "b": W[b] if isinstance(b, str) else b, "r": r * k, "tag": tag})

    head_up, head_fwd = _n(W["_head_up"]), _n(W["_head_fwd"])
    hl = _n(np.cross(head_up, head_fwd))
    # head: slightly egg-shaped, chin forward
    r = R["head"] * k
    P.append({"t": "ellipsoid", "c": W["head"] + head_up * r * 0.1,
              "M": np.stack([head_fwd * r * 1.05, hl * r * 0.88, head_up * r * 1.2], 1), "tag": "body"})
    # nose bump so the face direction reads in clay/canny
    P.append({"t": "ellipsoid", "c": W["head"] + head_fwd * r * 0.95 - head_up * r * 0.1,
              "M": np.stack([head_fwd * r * 0.22, hl * r * 0.16, head_up * r * 0.3], 1), "tag": "body"})
    neck_top = W["head"] - head_up * r * 0.6
    cap("neck", neck_top, R["neck"])
    # torso: chest + belly + pelvis ellipsoids along the spine
    spine = W["pelvis"] - W["neck"]
    L = float(np.linalg.norm(spine))
    ax = _n(spine)
    lat = _n(W["l_shoulder"] - W["r_shoulder"])
    chest_n = _n(np.cross(lat, -ax))
    P.append({"t": "ellipsoid", "c": W["neck"] + spine * 0.3,
              "M": np.stack([lat * 0.115 * k, chest_n * 0.075 * k, ax * L * 0.36], 1), "tag": "body"})
    P.append({"t": "ellipsoid", "c": W["neck"] + spine * 0.68,
              "M": np.stack([lat * 0.095 * k, chest_n * 0.065 * k, ax * L * 0.3], 1), "tag": "body"})
    hl2 = _n(W["l_hip"] - W["r_hip"])
    P.append({"t": "ellipsoid", "c": W["pelvis"] + ax * 0.01 * k,
              "M": np.stack([hl2 * 0.105 * k, chest_n * 0.07 * k, ax * 0.07 * k], 1), "tag": "body"})
    for s in ("r", "l"):
        P.append({"t": "ellipsoid", "c": W[f"{s}_shoulder"], "M": np.eye(3) * 0.035 * k, "tag": "body"})
        cap(f"{s}_shoulder", f"{s}_elbow", R["uarm"])
        cap(f"{s}_elbow", f"{s}_wrist", R["farm"])
        if s not in fists:                              # a gripping hand is drawn as a fist proxy instead
            P.append({"t": "ellipsoid", "c": W[f"{s}_hand"], "M": np.eye(3) * R["hand"] * k, "tag": "body"})
        cap(f"{s}_hip", f"{s}_knee", R["thigh"])
        cap(f"{s}_knee", f"{s}_ankle", R["shin"])
        cap(f"{s}_heel", f"{s}_toe", R["foot"])
    for item in wear or ():                       # worn props from the library (head slot)
        P += _props.wear_prims(item, W["head"], head_fwd, hl, head_up, r)
    for hand, item in (hold_items or {}).items():
        side = "r" if hand == "right" else "l"
        if "between" in W and _props.get(item).get("kind") == "wear":
            continue                                 # held between both hands: below
        at = (hold_at or {}).get(hand, W[f"{side}_hand"])
        # props are authored in metres for a 1.8 m figure (k is the body-unit scale, ~1.73x that)
        P += prop_prims(item, at, hold_dirs[hand], fwd, left, k * _standing_height() / 1.8,
                        flat=(hold_flats or {}).get(hand))
    for item in {v for v in (hold_items or {}).values() if "between" in W and _props.get(v).get("kind") == "wear"}:
        # a worn item carried between the hands: its own parts, in head radii, in front of the body
        P += _props.wear_prims(item, W["between"] + fwd * 0.04 * k - UP * 0.2 * R["head"] * k, fwd, left, UP,
                               R["head"] * k * 0.95)
    return P


def prop_prims(item: str, at, d, fwd, left, k=1.0, flat=None):
    """A held item from the prop library (blockout_props): grip at `at`, axis d, turned about d by flat;
    k = height / 1.8 (the library is in metres for a 1.8 m figure)."""
    return _props.held_prims(item, at, d, fwd, left, k, flat=flat)


# --------------------------------------------------------------------------------------------------
# The dragon proxy
# --------------------------------------------------------------------------------------------------
# Units: the dragon's "height" (standing head height, default 6 m). Parameters per pose, in the
# dragon's frame (fwd, left, up), angles in degrees:
#   lift: pelvis height (fraction of height; flying poses add "alt", metres, given by the shot)
#   pitch: spine angle above horizontal (90 = upright like the sheet)
#   neck: [angle above the spine's forward direction, curl], head: head pitch (+ = nose down)
#   fore: [shoulder angle, elbow angle] for the forelegs in the sagittal plane (screen-style: 0 fwd,
#         90 down, 270 up), r/l can differ; hind: same for the hind legs; wings: [raise, spread]
DRAGON_POSES = {
    "standing": dict(lift=0.42, pitch=62, neck=[25, 20], head=30, fore_r=[100, 60], fore_l=[95, 55],
                     hind=[[85, 110], [95, 100]], wings=[35, 0.55], tail=[200, 30], jaw=0),
    "rearing": dict(lift=0.5, pitch=78, neck=[15, 40], head=55, fore_r=[300, 330], fore_l=[330, 20],
                    hind=[[75, 115], [105, 100]], wings=[60, 0.9], tail=[200, 20], jaw=20),
    "striking": dict(lift=0.42, pitch=45, neck=[20, 30], head=45, fore_r=[320, 40], fore_l=[60, 50],
                     hind=[[70, 110], [105, 100]], wings=[50, 1.0], tail=[195, 25], jaw=25),
    "lunging": dict(lift=0.34, pitch=25, neck=[5, -10], head=10, fore_r=[30, 60], fore_l=[60, 90],
                    hind=[[110, 60], [80, 100]], wings=[25, 0.7], tail=[185, 10], jaw=30),
    "flying": dict(lift=0.0, pitch=8, neck=[20, 10], head=15, fore_r=[140, 60], fore_l=[140, 60],
                   hind=[[160, 120], [165, 120]], wings=[20, 1.0], tail=[180, 5], jaw=0, air=True),
    "flying_carry": dict(lift=0.0, pitch=12, neck=[25, 15], head=35, fore_r=[60, 120], fore_l=[60, 120],
                         hind=[[150, 110], [155, 110]], wings=[35, 1.0], tail=[180, 8], jaw=0, air=True),
}
CREATURE_POSES = list(DRAGON_POSES)


def dragon(fig: dict):
    """World primitives and anchors for a dragon figure {pose, at, facing, height, z}."""
    pose = fig.get("pose", "standing")
    if pose not in DRAGON_POSES:
        raise KeyError(f"unknown dragon pose {pose!r}; one of {CREATURE_POSES}")
    p = DRAGON_POSES[pose]
    S = float(fig.get("height", 4.5))
    fwd, left, up = frame(float(fig.get("_facing", 0.0)))
    x, y = fig.get("at", [0.0, 0.0])[:2]

    def sg(a):                                     # sagittal screen angle -> world direction
        r = math.radians(a)
        return math.cos(r) * fwd - math.sin(r) * up

    pitch = math.radians(p["pitch"])
    spine = math.cos(pitch) * fwd + math.sin(pitch) * up        # pelvis -> chest
    belly = _n(np.cross(spine, left))                          # toward the belly / front
    if np.dot(belly, -up) < 0 and np.dot(belly, fwd) < 0:
        belly = -belly
    # sizes (fractions of S)
    body_len, body_r = 0.36 * S, 0.11 * S
    pelvis = np.array([x, y, 0.0]) + up * (p["lift"] * S)
    if p.get("air"):
        pelvis = np.array([x, y, float(fig.get("z", 1.6 * S))])
    chest = pelvis + spine * body_len
    P = []
    A = {}
    P.append({"t": "ellipsoid", "c": 0.5 * (pelvis + chest),
              "M": np.stack([spine * body_len * 0.68, left * body_r * 0.95, belly * body_r * 1.05], 1), "tag": "body"})
    P.append({"t": "ellipsoid", "c": chest - spine * 0.02 * S,
              "M": np.stack([spine * body_r * 1.1, left * body_r * 0.9, belly * body_r * 1.1], 1), "tag": "body"})
    P.append({"t": "ellipsoid", "c": pelvis,
              "M": np.stack([spine * body_r * 0.95, left * body_r * 0.95, belly * body_r * 0.95], 1), "tag": "body"})
    A["chest"] = chest + belly * body_r * 0.9
    # neck: a curved chain from the chest forward/up
    na, curl = p["neck"]
    n0 = chest + spine * 0.04 * S
    ndir = rodrigues(spine, left, -na)            # positive tilts forward (toward belly side)
    pts = [n0]
    d = ndir
    seg = 0.075 * S
    for i in range(5):
        d = rodrigues(d, left, -curl / 5.0)
        pts.append(pts[-1] + d * seg)
    for i in range(5):
        P.append({"t": "capsule", "a": pts[i], "b": pts[i + 1], "r": (0.06 - 0.007 * i) * S, "tag": "body"})
    # head: pitched down by head degrees from the neck direction projected horizontal
    hfwd = _n(fwd * 1.0)
    hdir = rodrigues(hfwd, left, -p["head"])
    hdir = _n(hdir)
    hb = pts[-1]
    hup = _n(np.cross(left, hdir)) * -1
    if np.dot(hup, up) < 0:
        hup = -hup
    P.append({"t": "ellipsoid", "c": hb + hdir * 0.07 * S,
              "M": np.stack([hdir * 0.1 * S, left * 0.045 * S, hup * 0.05 * S], 1), "tag": "body"})
    # snout + jaw
    jaw = math.radians(p.get("jaw", 0))
    P.append({"t": "ellipsoid", "c": hb + hdir * 0.17 * S + hup * 0.005 * S,
              "M": np.stack([hdir * 0.08 * S, left * 0.03 * S, hup * 0.028 * S], 1), "tag": "body"})
    jd = _n(math.cos(jaw) * hdir - math.sin(jaw) * hup)
    P.append({"t": "ellipsoid", "c": hb + hdir * 0.05 * S + jd * 0.1 * S - hup * 0.03 * S,
              "M": np.stack([jd * 0.09 * S, left * 0.025 * S, np.cross(jd, left) * -0.018 * S], 1), "tag": "body"})
    # horns
    for s in (-1, 1):
        h0 = hb + hup * 0.04 * S + left * s * 0.03 * S
        P.append({"t": "capsule", "a": h0, "b": h0 - hdir * 0.09 * S + hup * 0.06 * S, "r": 0.012 * S, "tag": "body"})
    A["head"] = hb + hdir * 0.08 * S
    A["jaw"] = hb + hdir * 0.22 * S
    # tail: from the pelvis backward/down, curling
    ta, tcurl = p["tail"]
    td = _n(-spine * 0.4 + sg(ta) * 0.6 + (-up * 0.3 if not p.get("air") else 0))
    tp = [pelvis - spine * 0.08 * S]
    for i in range(8):
        td = rodrigues(td, left, tcurl / 8.0)
        nxt = tp[-1] + td * 0.075 * S
        if not p.get("air") and nxt[2] < 0.03 * S:      # the tail lies on the ground
            nxt[2] = 0.03 * S
            td = _n(nxt - tp[-1])
        tp.append(nxt)
    for i in range(8):
        P.append({"t": "capsule", "a": tp[i], "b": tp[i + 1], "r": (0.075 - 0.0085 * i) * S, "tag": "body"})
    # legs
    hip_w, sh_w = 0.08 * S, 0.085 * S
    for i, (s, side) in enumerate(((-1, "r"), (1, "l"))):
        a1, a2 = p["hind"][i]
        h0 = pelvis + left * s * hip_w
        k1 = h0 + sg(a1) * 0.2 * S
        k2 = k1 + sg(a2) * 0.2 * S
        P.append({"t": "capsule", "a": h0, "b": k1, "r": 0.06 * S, "tag": "body"})
        P.append({"t": "capsule", "a": k1, "b": k2, "r": 0.04 * S, "tag": "body"})
        foot_d = fwd if not p.get("air") else _n(sg(a2 + 30))
        P.append({"t": "capsule", "a": k2, "b": k2 + foot_d * 0.09 * S, "r": 0.025 * S, "tag": "body"})
        A[f"hind_{side}"] = k2
        f1, f2 = p[f"fore_{side}"]
        s0 = chest + left * s * sh_w - spine * 0.02 * S
        e1 = s0 + sg(f1) * 0.17 * S
        e2 = e1 + sg(f2) * 0.16 * S
        rc = (fig.get("reach") or {})
        if rc.get("limb", "fore_r") == f"fore_{side}" and rc.get("_point") is not None:
            pole = -spine * 0.3 - up * 0.5 + fwd * 0.4
            tgt = np.asarray(rc["_point"], float)
            # by "tip": the middle claw's TIP lands on the point (a contact), else the wrist does
            for _ in range(4 if rc.get("by") == "tip" else 1):
                e1, e2 = ik2(s0, 0.17 * S, 0.16 * S, tgt, pole)
                f2 = math.degrees(math.atan2(-float(np.dot(e2 - e1, up)), float(np.dot(e2 - e1, fwd))))
                if rc.get("by") == "tip":
                    tgt = np.asarray(rc["_point"], float) - _n(sg(f2 + 25)) * 0.068 * S
        P.append({"t": "capsule", "a": s0, "b": e1, "r": 0.035 * S, "tag": "body"})
        P.append({"t": "capsule", "a": e1, "b": e2, "r": 0.027 * S, "tag": "body"})
        cd = _n(sg(f2 + 25))
        for c in (-1, 0, 1):
            P.append({"t": "capsule", "a": e2, "b": e2 + cd * 0.06 * S + left * c * 0.02 * S, "r": 0.008 * S, "tag": "body"})
        A[f"claw_{side}"] = e2 + cd * 0.04 * S
        A[f"claw_tip_{side}"] = e2 + cd * 0.068 * S          # the middle claw's tip (its surface)
        # wings: from behind the shoulders, a finger fan with a membrane back to the hip
        raise_, spread = p["wings"]
        w0 = chest + left * s * 0.06 * S - belly * 0.08 * S
        out_dir = _n(left * s)
        arm = _n(rodrigues(out_dir, spine, s * 0.0) + up * math.tan(math.radians(raise_)) - spine * 0.2)
        span = 0.75 * S * spread + 0.15 * S
        elbow = w0 + arm * span * 0.35
        wrist = elbow + _n(arm + up * 0.35 * (1 if raise_ > 30 else 0) - fwd * 0.1) * span * 0.35
        P.append({"t": "capsule", "a": w0, "b": elbow, "r": 0.02 * S, "tag": "body"})
        P.append({"t": "capsule", "a": elbow, "b": wrist, "r": 0.015 * S, "tag": "body"})
        tips = []
        for j, ang in enumerate((10, 40, 75, 110)):
            fd = _n(rodrigues(arm, -_n(np.cross(arm, -fwd)) if s > 0 else _n(np.cross(arm, -fwd)), 0) )
            fd = _n(math.cos(math.radians(ang)) * arm + math.sin(math.radians(ang)) * (-fwd * 0.8 - up * 0.6))
            tip = wrist + fd * span * (0.55 - 0.08 * j)
            tips.append(tip)
            P.append({"t": "capsule", "a": wrist, "b": tip, "r": 0.008 * S, "tag": "body"})
        root_back = pelvis + left * s * 0.04 * S - belly * 0.06 * S
        poly = [w0, elbow, wrist] + tips + [root_back]
        tris = []
        # fan from the wrist over the tips, plus wrist-elbow-root and elbow-w0-root panels
        for j in range(len(tips) - 1):
            tris.append([2, 3 + j, 4 + j])
        tris.append([2, 3 + len(tips) - 1, len(poly) - 1])
        tris.append([1, 2, len(poly) - 1])
        tris.append([0, 1, len(poly) - 1])
        tris.append([2, 1, 3])
        P.append({"t": "mesh", "verts": poly, "tris": tris, "tag": "wing"})
    A["claws"] = 0.5 * (A["claw_r"] + A["claw_l"])
    A["pelvis"] = pelvis
    return P, A


# --------------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------------
SENSOR_W = 36.0
FRAMING = {
    # frame height covered at the target (multiples of the target's height), look-at height
    # (fraction of height), camera height (fraction; None = level with the look-at), lens mm
    "wide": dict(cover=3.2, look=0.5, cam=0.75, lens=28),
    "medium": dict(cover=1.35, look=0.55, cam=None, lens=40),
    "medium close-up": dict(cover=0.75, look=0.72, cam=None, lens=50),
    "close-up": dict(cover=0.45, look=0.8, cam=None, lens=60),
    "insert": dict(cover=0.45, look=0.5, cam=None, lens=60),      # a close-up of the hands + what they hold
    "low angle": dict(cover=1.6, look=0.6, cam=0.2, lens=24),
    "high angle": dict(cover=1.8, look=0.4, cam=1.9, lens=35),
}


def look_at(cam, target):
    """Camera rotation matrix (columns: right, up, -forward) for a camera at cam looking at target."""
    f = _n(np.asarray(target, float) - np.asarray(cam, float))
    r = np.cross(f, UP)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r = _n(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f], 1)


def project(cam: dict, pts, W=1536, H=864):
    """World points -> pixel coords (None behind the camera)."""
    Rm = np.asarray(cam["R"], float)
    C = np.asarray(cam["pos"], float)
    fpx = cam["lens"] / SENSOR_W * W
    out = []
    for p in pts:
        if p is None:
            out.append(None)
            continue
        q = Rm.T @ (np.asarray(p, float) - C)
        if q[2] > -1e-3:
            out.append(None)
            continue
        out.append([W / 2 + fpx * q[0] / -q[2], H / 2 - fpx * q[1] / -q[2]])
    return out
