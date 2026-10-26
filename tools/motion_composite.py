"""Composite creatures: a HUMANOID body (human rig joint names) + extra chains (long neck, jaw, tail, wings), any story.

    from motion_composite import composite_motion, COMPOSITE_ACTIONS
    seq, val = composite_motion(rig_json, body_cm, layers, fps, N, at=(x, y), facing_deg=f)

- body: a humanoid CanonicalMotion (HY-Motion) retargeted onto the rig's own bones (tools/motion_retarget.py: rig.json
  carries the human aliases pelvis, spine_mid, neck, head, head_top, {r,l}_{shoulder,elbow,wrist,hand,hip,knee,ankle,toe});
- neck chain (neck2..neckK between neck and head): rides the neck bone's rotation from its rest curve, and bends a
  little more toward the head the way the head turns (the long neck follows, the head leads);
- jaw / snout / head_end / head_top: ride the head bone; the jaw opens about the head's lateral axis by the jaw layer;
- tail1..tailN: a damped spring chain from the pelvis (rest offsets in the pelvis frame), ground collision, sweeps;
- wings: tools/motion_creature.Creature.wing (fold/spread/flap within the template limits) in the CHEST frame, the
  segment directions rebuilt on the rig's own wing lengths from the rig's wing shoulder (which rides the chest);
- flight: the whole body follows a root path (template "fly"/"carry" layer) at the staged altitude.
layers: {"wings": [(t0, t1, pose: folded|half|spread, flap_amp_deg, flap_hz)], "jaw": [(t0, t1, deg)] or
         {"wav": path, "t": s}, "tail": [(t0, t1, sweep_deg)], "root": {"from": [x,y,z], "to": [x,y,z]} (m, world)}
val (validation numbers): bone-length error, lowest foot / tail z, jaw range, wing angle ranges, max joint speed.
"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion_retarget import BONE_NAMES, retarget  # noqa: E402

# verb (from tools/motion_batch.py CREATURE_MAP / the shot's creature verbs) -> humanoid body prompt + layers
COMPOSITE_ACTIONS = {
    # on the ground a winged creature stands on ALL FOURS (its arms are its forelegs) and rears up to strike or roar
    # (10-23: every ground move was a man's - it fought standing upright, a 6.5 m man with a tail)
    "idle": {"prompt": "stands on all fours on hands and feet, head raised, breathing slowly, shifting weight slightly",
             "wings": "folded"},
    "brace": {"prompt": "crouches low on all fours, head forward and low, ready to spring", "wings": "half"},
    "strike": {"prompt": "on all fours, rears up onto the legs, raises the right arm high and swipes it down hard in "
                         "front of the body, then drops back onto the hands", "wings": "half", "mode": "stop"},
    "roar": {"prompt": "rears up from all fours onto the legs, throws the head back and roars with both arms spread "
                       "wide", "wings": "spread", "jaw": 35},
    "roar_long": {"prompt": "on all fours, stretches the head forward low and roars for a long time", "wings": "spread",
                  "jaw": 38},
    "recoil": {"prompt": "on all fours, jerks back and rears up as if struck hard in the chest", "wings": "spread",
               "jaw": 25},
    "fall": {"prompt": "on all fours, the arms give way and the body collapses forward onto the chest and lies still",
             "wings": "spread", "flap": 50},
    "dead": {"prompt": "lies flat on the chest and belly on the ground, arms and legs splayed, completely still",
             "wings": "spread", "flap": 0},
    "carry": {"prompt": "holds something in both arms in front of the chest, legs dangling", "wings": "spread",
              "flap": 30, "fly": True},
    "fly": {"prompt": "spreads both arms out to the sides, legs dangling", "wings": "spread", "flap": 35, "fly": True},
    "flap": {"prompt": "stands on all fours, head raised, breathing hard", "wings": "spread", "flap": 45},
    "tail": {"prompt": "on all fours, swings the hips and the body hard to the right and back", "wings": "folded",
             "tail": 70},
    "look": {"prompt": "on all fours, turns the head to look to the side", "wings": "folded"},
}
WING_POSES = {"folded": (10, 25, 15), "half": (20, 95, 90), "spread": (25, 165, 170)}


def _n(v):
    v = np.asarray(v, float)
    d = np.linalg.norm(v)
    return v / d if d > 1e-9 else v


def rot_axis(axis, ang):
    a = _n(axis)
    c, s = math.cos(ang), math.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def load_rig(path):
    r = json.load(open(path))
    J = {k: np.asarray(v, float) for k, v in r["joints"].items()}
    return r, J


def _env(t, t0, t1, ramp=0.25):
    if t < t0 or t > t1:
        return 0.0
    return min(1.0, (t - t0) / ramp, (t1 - t) / ramp) if t1 > t0 else 0.0


def composite_motion(rig_json, body_cm, layers, fps, N, at=(0.0, 0.0), facing_deg=0.0):
    from motion_creature import Creature, Spring, speech_features
    rig, RJ = load_rig(rig_json)
    par = rig.get("parents", {})
    rm = retarget(body_cm, {"joints": {k: v.tolist() for k, v in RJ.items()}, "height": rig.get("height", 6.5)},
                  at=at, facing_deg=facing_deg)
    F = min(N, rm.frames)
    names = rm.names
    ix = {n: i for i, n in enumerate(names)}
    bix = {b: i for i, b in enumerate(BONE_NAMES)}
    dt = 1.0 / fps
    # root path (flight): translate the whole body
    root = layers.get("root")
    shift = np.zeros((N, 3))
    if root:
        a, b = np.asarray(root["from"], float), np.asarray(root["to"], float)
        p0 = rm.pos[0, ix["pelvis"]]
        for f in range(N):
            u = f / max(N - 1, 1)
            shift[f] = a + (b - a) * u - p0 + np.array([0, 0, 0.15 * math.sin(2 * math.pi * 0.8 * f * dt)])
    jaw_feat = None
    if isinstance(layers.get("jaw"), dict) and layers["jaw"].get("wav"):
        jaw_feat = speech_features(layers["jaw"]["wav"], fps)
    C = Creature("winged_quadruped")
    tail = sorted([k for k in RJ if k.startswith("tail") and k[4:].isdigit()], key=lambda k: int(k[4:]))
    necks = sorted([k for k in RJ if k.startswith("neck") and k[4:].isdigit()], key=lambda k: int(k[4:]))
    tail_pos = tail_vel = None
    tip_lag = {}
    wing_sp = None
    Rw_prev = None
    out, val = [], {"bone_len_err_m": 0.0, "min_foot_z": 9.0, "min_tail_z": 9.0, "jaw_deg": [99, -99],
                    "wing_elbow_deg": [999, -999], "max_speed_mps": 0.0, "min_joint_z": 9.0}
    prev = None
    for f in range(N):
        g = min(f, F - 1)
        t = f * dt
        J = {n: rm.pos[g, ix[n]] + shift[f] for n in names}
        Rb = {b: rm.bone_rot[g, bix[b]] for b in BONE_NAMES}
        Rz = Rb["pelvis"]
        Rc = Rb["spine"]                           # the chest frame (spine_mid -> neck bone)
        if "chest" in RJ:
            J["chest"] = J["spine_mid"] + Rc @ (RJ["chest"] - RJ["spine_mid"])
            for s_ in ("l", "r"):                   # the arm hangs from the shoulder the chest frame implies
                sh_ = J["chest"] + Rc @ (RJ[f"{s_}_shoulder"] - RJ["chest"])
                dsh = sh_ - J[f"{s_}_shoulder"]
                for k_ in (f"{s_}_shoulder", f"{s_}_elbow", f"{s_}_wrist", f"{s_}_hand"):
                    J[k_] = J[k_] + dsh
            J["neck"] = J["chest"] + Rc @ (RJ["neck"] - RJ["chest"])
        # ---- neck chain: ride the neck bone + extra bend toward the head turn
        if necks:
            Rn = Rb["neck"]
            Rh = Rb["head"]
            base = J["neck"]
            k_ = len(necks)
            for i, nk in enumerate(necks):
                if nk == "neck":
                    continue
                w = (i / k_)
                R_i = rot_axis(np.cross([0, 0, 1], [1, 0, 0]), 0) @ Rn
                off = RJ[nk] - RJ["neck"]
                p_rig = base + R_i @ off
                p_head = J["head"] + Rh @ (RJ[nk] - RJ["head"])
                J[nk] = p_rig * (1 - 0.5 * w) + p_head * (0.5 * w)
            chain_ = ["neck"] + [nk for nk in necks if nk != "neck"] + ["head"]
            Rh0 = Rb["head"]
            head_old = J["head"].copy()
            for a_, b_ in zip(chain_[:-1], chain_[1:]):     # re-walk on the rest lengths (the head ends the chain)
                L_ = float(np.linalg.norm(RJ[b_] - RJ[a_]))
                tgt_ = J[b_] if b_ != "head" else head_old
                J[b_] = J[a_] + _n(tgt_ - J[a_]) * L_
        # ---- head extras + jaw
        Rh = Rb["head"]
        for k in ("snout", "head_end", "head_top", "jaw"):
            if k in RJ:
                J[k] = J["head"] + Rh @ (RJ[k] - RJ["head"])
        jaw = 0.0
        if jaw_feat is not None:
            i_ = int(round((t - layers["jaw"].get("t", 0.0)) * fps))
            if 0 <= i_ < len(jaw_feat["env"]):
                jaw = 28.0 * jaw_feat["env"][i_] + 14.0 * jaw_feat["onset"][i_]
        else:
            for t0, t1, deg in layers.get("jaw") or []:
                jaw = max(jaw, deg * _env(t, t0, t1, 0.2))
        jaw = float(np.clip(jaw, 0, 40))
        if "jaw" in RJ:
            lat = Rh @ np.array([0, 1.0, 0])
            J["jaw"] = J["head"] + rot_axis(lat, math.radians(jaw)) @ (J["jaw"] - J["head"])
        # ---- tail spring chain from the pelvis
        if tail:
            tgt = [J["pelvis"]]
            prevk = "pelvis"
            sweep = 0.0
            for t0, t1, deg in layers.get("tail") or []:
                sweep += deg * math.sin(math.pi * min(max((t - t0) / max(t1 - t0, 1e-3), 0), 1)) if t0 <= t <= t1 else 0.0
            for i, tk in enumerate(tail):
                seg = Rz @ (RJ[tk] - RJ[prevk])
                seg = rot_axis([0, 0, 1], math.radians(sweep / len(tail) + 3 * math.sin(2 * math.pi * 0.25 * t + i * 0.5))) @ seg
                tgt.append(tgt[-1] + seg)
                prevk = tk
            tgt = np.array(tgt[1:])
            if tail_pos is None:
                tail_pos, tail_vel = tgt.copy(), np.zeros_like(tgt)
            for i in range(len(tail)):
                kk = (2 * math.pi * (2.2 - 0.12 * i)) ** 2
                a = kk * (tgt[i] - tail_pos[i]) - 2 * 0.55 * math.sqrt(kk) * tail_vel[i]
                tail_vel[i] += a * dt
                tail_pos[i] += tail_vel[i] * dt
            prevp, prevk = J["pelvis"], "pelvis"
            for i, tk in enumerate(tail):
                L = float(np.linalg.norm(RJ[tk] - RJ[prevk]))
                p_ = prevp + _n(tail_pos[i] - prevp) * L
                p_ = _on_ground(prevp, p_, L)              # lies ON the ground, keeping its length
                tail_pos[i] = p_
                J[tk] = p_.copy()
                prevp, prevk = p_, tk
        # ---- wings: the template solver in the chest frame, rebuilt on the rig's wing lengths
        pose, amp, hz = "folded", 0.0, 1.0
        for t0, t1, ps, a_, h_ in layers.get("wings") or []:
            if t0 <= t <= t1:
                pose, amp, hz = ps, a_, h_
        target_pose = np.array(WING_POSES[pose], float)
        if wing_sp is None:
            wing_sp = Spring(target_pose, 1.6, 0.9)          # ~0.4 s to change pose (no snapping)
        el0, eb0, wr0 = wing_sp.step(target_pose, dt)
        # heavy appendages lag the torso: the wings take a smoothed chest frame (the roots stay attached)
        Rw = Rc.copy() if Rw_prev is None else Rw_prev * 0.7 + Rc * 0.3
        u_, _, vt_ = np.linalg.svd(Rw)
        Rw = u_ @ vt_
        Rw_prev = Rw
        for s in ("l", "r"):
            sh = f"wing_{s}_shoulder"
            if sh not in RJ:
                continue
            for k in (f"wing_{s}_root", f"wing_{s}_back", sh):
                if k in RJ:
                    J[k] = J["spine_mid"] + Rc @ (RJ[k] - RJ["spine_mid"])
            flap = amp * math.sin(2 * math.pi * hz * t)
            W = C.wing(s, np.zeros(3), Rw, el0, eb0, wr0, flap)
            sh0 = W[sh]
            chain = [(sh, f"wing_{s}_elbow"), (f"wing_{s}_elbow", f"wing_{s}_wrist")] + \
                    [(f"wing_{s}_wrist", f"wing_{s}_{q}") for q in ("tip", "finger1", "finger2", "finger3")]
            for a_, b_ in chain:
                if a_ in RJ and b_ in RJ:
                    L = float(np.linalg.norm(RJ[b_] - RJ[a_]))
                    d_ = _n(W[b_] - W[a_])
                    p_ = J[a_] + d_ * L
                    if b_.endswith(("tip", "finger1", "finger2", "finger3")):
                        sp = tip_lag.setdefault(b_, Spring(p_, 3.5, 0.55))
                        p_ = J[a_] + _n(sp.step(p_, dt) - J[a_]) * L
                    J[b_] = _on_ground(J[a_], p_, L)        # a lying/crouching body spreads its wings ON the ground
            ang = math.degrees(math.acos(np.clip(np.dot(_n(J[sh] - J[f"wing_{s}_elbow"]),
                                                        _n(J[f"wing_{s}_wrist"] - J[f"wing_{s}_elbow"])), -1, 1)))
            val["wing_elbow_deg"] = [min(val["wing_elbow_deg"][0], ang), max(val["wing_elbow_deg"][1], ang)]
        # bone-tail aliases the rig also carries (the renderer poses bones by these names)
        alias = {"spine": "spine_mid", "r_clav": "r_shoulder", "l_clav": "l_shoulder", "r_uarm": "r_elbow",
                 "l_uarm": "l_elbow", "r_farm": "r_wrist", "l_farm": "l_wrist", "r_hipb": "r_hip", "l_hipb": "l_hip",
                 "r_thigh": "r_knee", "l_thigh": "l_knee", "r_shin": "r_ankle", "l_shin": "l_ankle", "r_foot": "r_toe",
                 "l_foot": "l_toe"}
        for tl, hn in alias.items():
            if tl in RJ and hn in J:
                J[tl] = J[hn]
        for k in RJ:
            if k not in J:
                J[k] = J["pelvis"] + Rz @ (RJ[k] - RJ["pelvis"])
        # validation
        val["min_joint_z"] = min(val["min_joint_z"], float(min(v[2] for v in J.values())))
        val["min_foot_z"] = min(val["min_foot_z"], float(min(J["l_toe"][2], J["r_toe"][2])))
        if tail:
            val["min_tail_z"] = min(val["min_tail_z"], float(min(J[k][2] for k in tail)))
        val["jaw_deg"] = [min(val["jaw_deg"][0], jaw), max(val["jaw_deg"][1], jaw)]
        if prev is not None:
            sp_ = max(float(np.linalg.norm(J[k] - prev[k])) for k in J) * fps
            val["max_speed_mps"] = max(val["max_speed_mps"], sp_)
        prev = J
        out.append(J)
    for c_, p_ in par.items():                      # bone-length error over the clip (rig's own lengths)
        if p_ and c_ in RJ and p_ in RJ:
            L0 = float(np.linalg.norm(RJ[c_] - RJ[p_]))
            err = max(abs(float(np.linalg.norm(J[c_] - J[p_])) - L0) for J in out)
            val["bone_len_err_m"] = max(val["bone_len_err_m"], err)
    val = {k: (np.round(v, 3).tolist() if isinstance(v, list) else round(float(v), 3)) for k, v in val.items()}
    return out, val


def layers_for(verbs, dur, kind_from_batch):
    """Composite layers from the creature verbs a moment got (tools/motion_batch.py): the body prompt of the leading
    verb + wing/jaw/tail/root layers over the verbs' times."""
    lead = next((v["verb"] for v in verbs if v["verb"] not in ("idle", "look", "lower_for")), verbs[0]["verb"] if verbs else "idle")
    A = COMPOSITE_ACTIONS.get(lead, COMPOSITE_ACTIONS["idle"])
    L = {"wings": [], "jaw": [], "tail": []}
    for v in verbs:
        a = COMPOSITE_ACTIONS.get(v["verb"], COMPOSITE_ACTIONS["idle"])
        t0 = float(v.get("t", 0.0)) if not isinstance(v.get("t"), str) else 0.0
        t1 = t0 + float(v.get("dur", dur - t0))
        L["wings"].append((t0, t1, a.get("wings", "folded"), float(a.get("flap", 0.0)), 1.1 if a.get("fly") else 1.4))
        if a.get("jaw"):
            L["jaw"].append((t0, t1, float(a["jaw"])))
        if a.get("tail"):
            L["tail"].append((t0, t1, float(a["tail"])))
        if v["verb"] == "speak" and v.get("wav"):
            L["jaw"] = {"wav": v["wav"], "t": t0}
    if not L["wings"] or L["wings"][0][0] > 0:
        L["wings"].insert(0, (0.0, dur, "folded", 0.0, 1.0))
    return A["prompt"], A.get("mode", "none"), L, bool(A.get("fly"))


GROUND_Z = 0.05


def _on_ground(parent, p, L, floor=GROUND_Z):
    """Child joint `p` at bone length L from `parent`, pushed up to lie ON the floor (length kept) if it went under.
    Any chain (tail, wings) on any creature: a body that lies or crouches drapes its appendages on the ground."""
    p = np.asarray(p, float)
    if p[2] >= floor:
        return p
    dz = floor - parent[2]
    if abs(dz) >= L:                                # parent itself is under/at the floor: go straight up
        return parent + np.array([0, 0, L if dz > 0 else -L])
    hor = np.array([p[0] - parent[0], p[1] - parent[1], 0.0])
    if np.linalg.norm(hor) < 1e-6:
        hor = np.array([1.0, 0, 0])
    return parent + _n(hor) * math.sqrt(L * L - dz * dz) + np.array([0, 0, dz])


def _crouch(J, drop):
    """Lower everything above the legs by `drop` m; knees re-solved by 2-bone IK (feet stay planted, knees forward)."""
    from blockout_contact import ik2
    if drop <= 0:
        return dict(J)
    legs = ("hip", "knee", "ankle", "toe", "thigh", "shin", "foot", "hipb")
    J2 = {}
    for q, v in J.items():
        v = np.asarray(v, float)
        J2[q] = v if q.startswith(("l_", "r_")) and q[2:] in legs else v - np.array([0, 0, drop])
    for s_ in ("l", "r"):
        hip0, kn0, an0 = (np.asarray(J[f"{s_}_{q}"], float) for q in ("hip", "knee", "ankle"))
        hip = hip0 - np.array([0, 0, drop])
        l1, l2 = float(np.linalg.norm(kn0 - hip0)), float(np.linalg.norm(an0 - kn0))
        fwd = np.asarray(J["chest"] if "chest" in J else J["neck"], float) - np.asarray(J["pelvis"], float)
        pole = np.array([fwd[0], fwd[1], 0.0]) + (kn0 - (hip0 + an0) / 2)
        kn, an = ik2(hip, l1, l2, an0, pole)
        J2[f"{s_}_hip"], J2[f"{s_}_knee"] = hip, kn
        for al, tl in (("hipb", "hip"), ("thigh", "knee")):
            if f"{s_}_{al}" in J2:
                J2[f"{s_}_{al}"] = J2[f"{s_}_{tl}"]
    return J2


def lean_to(seq, joint, target, imp, fps, offset=None, lead=0.6, hold=1.0, max_deg=85.0, xy=True, crouch_max=0.4):
    """A tall humanoid brings a body point down to a smaller opponent: around frame `imp` the upper body (all but
    the pelvis, legs and tail) pivots forward about the pelvis by the angle that puts `joint` (+ offset) at the
    target's height (solved at the impact), eased in over `lead` s and out after `hold` s; then (xy=True) the
    whole figure shifts horizontally so the point is over the target. Rigid rotation: bone lengths unchanged.
    Returns (seq, angle_deg, residual_m)."""
    lower = ("pelvis", "hip", "knee", "ankle", "toe", "thigh", "shin", "foot", "hipb", "tail")
    J0 = seq[imp]
    piv = np.asarray(J0["pelvis"], float)
    fwd = np.asarray(J0["chest"] if "chest" in J0 else J0["neck"], float) - piv
    fwd = np.array([fwd[0], fwd[1], 0.0])
    if np.linalg.norm(fwd) < 1e-6:                       # facing from the hips
        lr = np.asarray(J0["l_hip"], float) - np.asarray(J0["r_hip"], float)
        fwd = np.cross(lr, [0, 0, 1.0])
    fwd = _n(fwd)
    axis = _n(np.cross([0, 0, 1.0], fwd))               # rotating up toward forward = pitching forward-down
    tgt = np.asarray(target, float)

    def point(J, ang):
        R = rot_axis(axis, math.radians(ang))
        p = np.asarray(J[joint], float) + (np.asarray(offset, float) if offset is not None else 0.0)
        return np.asarray(J["pelvis"], float) + R @ (p - np.asarray(J["pelvis"], float))
    leg = float(np.linalg.norm(np.asarray(J0["r_hip"], float) - np.asarray(J0["r_knee"], float)) +
                np.linalg.norm(np.asarray(J0["r_knee"], float) - np.asarray(J0["r_ankle"], float)))
    best, bdrop, bz = 0.0, 0.0, 1e9
    for drop in np.linspace(0.0, crouch_max * leg, 9):   # crouch depth x lean angle: least change that reaches
        Jc = _crouch(J0, drop)
        for ang in np.linspace(-20.0, max_deg, 106):
            e = abs(point(Jc, ang)[2] - tgt[2])
            cost = e + 0.002 * abs(ang) + 0.05 * drop
            if cost < bz:
                best, bdrop, bz = float(ang), float(drop), cost
    n = len(seq)
    a0, a1 = imp - int(lead * fps), imp + int(hold * fps)
    out = []
    for f, J in enumerate(seq):
        if f < a0 or f > a1 + int(0.5 * fps):
            k = 0.0
        elif f < imp:
            u = (f - a0) / max(imp - a0, 1)
            k = u * u * (3 - 2 * u)
        elif f <= a1:
            k = 1.0
        else:
            u = (f - a1) / max(int(0.5 * fps), 1)
            k = 1 - u * u * (3 - 2 * u)
        R = rot_axis(axis, math.radians(best * k))
        Jc = _crouch(J, bdrop * k)
        pv = np.asarray(Jc["pelvis"], float)
        J2 = {}
        for q, v in Jc.items():
            v = np.asarray(v, float)
            J2[q] = v if q.startswith(lower) or q in ("l_hip", "r_hip") else pv + R @ (v - pv)
        out.append(J2)
    if xy:
        d = tgt - point(out[imp], 0.0)
        d[2] = 0.0
        for J in out:
            for q in J:
                J[q] = J[q] + d
    res = float(np.linalg.norm(point(out[imp], 0.0) - tgt)) if xy else float(abs(point(out[imp], 0.0)[2] - tgt[2]))
    return out, best, res, bdrop


def ground_pass(seq, par, floor=GROUND_Z, terrain=None):
    """Final floor constraint over a whole clip, parent-first on the rig's parent map (any creature).
    A joint keeps its bone length to its (possibly moved) parent and is lifted ON the floor if under it;
    joints already above the floor with an unmoved parent are untouched. terrain(x, y): the set's surface height
    (motion_terrain: a pano world's slopes and rocks), the floor is then terrain + floor. Returns (seq, n_fixed,
    min clearance above the floor)."""
    flo = (lambda v: floor) if terrain is None else (lambda v: float(terrain(v[0], v[1])) + floor)  # noqa: E731
    if not par:
        return seq, 0, min(min(v[2] - flo(v) + floor for v in J.values()) for J in seq)
    order, seen = [], set()
    def visit(k):
        if k in seen:
            return
        seen.add(k)
        if par.get(k):
            visit(par[k])
        order.append(k)
    for J in seq[:1]:
        for k in J:
            visit(k)
    out, fixed, mz = [], 0, 9.0
    for J in seq:
        J2, moved = dict(J), set()
        for k in order:
            if k not in J:
                continue
            pk = par.get(k)
            v = np.asarray(J[k], float)
            fl = flo(v)
            if pk in J2 and (pk in moved or v[2] < fl):
                L = float(np.linalg.norm(np.asarray(J[k], float) - np.asarray(J[pk], float)))
                q = J2[pk] + _n(v - J2[pk]) * L if pk in moved else v
                q = _on_ground(J2[pk], q, L, flo(q))
                if np.linalg.norm(q - v) > 1e-6:
                    moved.add(k); fixed += 1
                J2[k] = q
            elif v[2] < fl:
                J2[k] = np.array([v[0], v[1], fl]); moved.add(k); fixed += 1
        mz = min(mz, min(float(x[2]) - flo(x) + floor for x in J2.values()))
        out.append(J2)
    return out, fixed, round(mz, 3)
