"""Animated control VIDEOS for a shot ("Mode 2"): start -> end interpolated in 3D, rendered fast.

    ~/tts-ab/.venv-pose/bin/python -I tools/blockout_video.py shot.json OUT_DIR [--frames 121] [--fps 24]
        [--size 768x448] [--move dolly_in|dolly_out|pan_left|pan_right|orbit_left|orbit_right|none]
        [--move-amount 0.15] [--hold 6]

Writes OUT_DIR/depth.mp4 + depth_figures.mp4 at --size (--all-passes: also {pose,canny,clay,depth_static}.mp4
and half-size clips, which nothing reads), figures0.png (frame 0's figure mask), OUT_DIR/strip.jpg (8 frames x 4
passes; --key-frames: Blender renders only frames 0 and N-1, no clips, the checks still cover every frame),
OUT_DIR/checks.json (every frame's grip/contact numbers) and OUT_DIR/video.json. LTX wants frames = 1+8k
and W, H multiples of 64 (both asserted).

How a frame is made (tools/blockout.py stages start and end, contacts solved there; this interpolates):
- Humans: joints are interpolated BONE BY BONE: every joint's offset from its parent is slerped in
  direction and lerped in length (no linear joint drift that shrinks limbs), the pelvis lerped. Rigged
  models are interpolated on their own (fk_model) proportions, so Blender reproduces them exactly.
- Timing: a figure with a contact that engages at the END accelerates into it (ease-in, power 2.2: a
  strike); others ease in-out (smoothstep). --hold frames stay on the start/end pose.
- Held props ride the interpolated hand along the slerped grip axis; frame:"both" contacts (tongs on the
  blade) are re-solved at EVERY frame; two-handed grips re-IK the left fist every frame.
- Checks at every frame (blockout_contact.check): grips always; engaged contacts full; contacts that
  engage later are checked for PENETRATION on the way in (a hammer head must not pass through the
  anvil before it lands). Violations -> exit 2 (BLOCKOUT_ALLOW_VIOLATIONS=1 to keep the files).
- Creature proxies: their primitives are lerped between the start and end builds (same structure).
- Camera: the shot's camera solved over start+end (as blockout.py), plus an optional eased move.
- Render: Blender Workbench (OpenGL, ~0.1 s a frame) in ONE process: depth from the Z pass, a studio-lit
  clay picture for canny. Depth is normalised with ONE near/far for the whole clip (no flicker).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockout as B  # noqa: E402
import blockout_contact as C  # noqa: E402
import blockout_rig as rig  # noqa: E402
import blockout_props as P_  # noqa: E402
from poses import draw_people  # noqa: E402

PYAV = os.environ.get("BLOCKOUT_PYAV", os.path.expanduser("~/tts-ab/.venv-ltx/bin/python"))
ENCODE = """
import sys, av, numpy as np
from PIL import Image
pat, n, fps, out, size = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
c = av.open(out, "w")
st = None
for i in range(n):
    im = Image.open(pat % i).convert("RGB")
    if size:
        w, h = (int(x) for x in size.split("x"))
        im = im.resize((w, h), Image.BOX)
    if st is None:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = im.width, im.height, "yuv420p"
        st.options = {"crf": "12", "preset": "medium"}
    for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
        c.mux(p)
for p in st.encode():
    c.mux(p)
c.close()
"""

CREATURE_MOTION = {}  # motion={who: [joints]} for rigged creatures (template joint names)
IMPACTS = {}         # main(impacts={contact index | who: frame}) from a motion source
LAYERS = {}          # --layers: {"static": ["set", "creature"], "figures": ["figure"]}
MANNEQUIN = [False]  # --mannequin (10-22 mannequin flow): people as capsule mannequins posed by the model's joints, no
                     # character mesh - the render feeds only the control maps (pose, depth), never a keyframe

PARENT = {"spine_mid": "pelvis", "neck": "spine_mid", "head": "neck", "head_top": "head",
          "nose": "head", "r_eye": "head", "l_eye": "head", "r_ear": "head", "l_ear": "head", "helm": "neck",
          "r_shoulder": "neck", "l_shoulder": "neck", "r_elbow": "r_shoulder", "l_elbow": "l_shoulder",
          "r_wrist": "r_elbow", "l_wrist": "l_elbow", "r_hand": "r_wrist", "l_hand": "l_wrist",
          "r_hip": "pelvis", "l_hip": "pelvis", "r_knee": "r_hip", "l_knee": "l_hip",
          "r_ankle": "r_knee", "l_ankle": "l_knee", "r_toe": "r_ankle", "l_toe": "l_ankle",
          "r_heel": "r_ankle", "l_heel": "l_ankle"}


def slerp(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    la, lb = np.linalg.norm(a), np.linalg.norm(b)
    if la < 1e-9 or lb < 1e-9:
        return a + (b - a) * t
    ua, ub = a / la, b / lb
    c = float(np.clip(np.dot(ua, ub), -1, 1))
    th = math.acos(c)
    if th < 1e-5:
        u = ua
    elif th > math.pi - 1e-3:                  # opposite: go round via any perpendicular
        perp = np.cross(ua, [0, 0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(ua, [1.0, 0, 0])
        perp /= np.linalg.norm(perp)
        u = ua * math.cos(math.pi * t) + perp * math.sin(math.pi * t)
    else:
        u = (math.sin((1 - t) * th) * ua + math.sin(t * th) * ub) / math.sin(th)
    return u * (la + (lb - la) * t)


def interp_joints(J0, J1, t):
    out = {}
    order = ["pelvis"]
    seen = {"pelvis"}
    while len(order) < len(PARENT) + 1:          # parents before children
        grew = False
        for c, p in PARENT.items():
            if c not in seen and p in seen:
                order.append(c)
                seen.add(c)
                grew = True
        if not grew:
            break
    out["pelvis"] = J0["pelvis"] + (J1["pelvis"] - J0["pelvis"]) * t
    for c in order[1:]:
        if c not in J0 or c not in J1:
            continue
        p = PARENT[c]
        out[c] = out[p] + slerp(J0[c] - J0[p], J1[c] - J1[p], t)
    for k in J0:                                 # anything else (and the _vectors): slerp / lerp
        if k in out:
            continue
        if k.startswith("_"):
            out[k] = slerp(J0[k], J1[k], t)
        elif k in J1:
            out[k] = J0[k] + (J1[k] - J0[k]) * t
    return out


def ease(t, kind):
    t = min(max(t, 0.0), 1.0)
    if kind == "in":
        return t ** 2.2
    if kind == "linear":
        return t
    return t * t * (3 - 2 * t)


def lerp_prims(P0, P1, t):
    if len(P0) != len(P1):
        return P0 if t < 0.5 else P1
    out = []
    for a, b in zip(P0, P1):
        q = dict(a)
        for k in ("a", "b", "c", "M", "verts"):
            if k in a and k in b:
                va, vb = np.asarray(a[k], float), np.asarray(b[k], float)
                if va.shape == vb.shape:
                    q[k] = va + (vb - va) * t
        if "r" in a and "r" in b:
            q["r"] = a["r"] + (b["r"] - a["r"]) * t
        out.append(q)
    return out


def path_camera(cam, t):
    """An AUTHORED camera path (shot["camera"]["path"] = [{frame, pos, look, lens}, ...], frames 0..N-1):
    eased (smoothstep) between consecutive keys; t in [0, 1] over the clip."""
    P = sorted(cam["path"], key=lambda k: k["frame"])
    f = t * max(P[-1]["frame"], 1)
    k1 = next((i for i, k in enumerate(P) if k["frame"] >= f), len(P) - 1)
    k0 = max(k1 - 1, 0)
    a, b = P[k0], P[k1]
    u = 0.0 if b["frame"] == a["frame"] else (f - a["frame"]) / (b["frame"] - a["frame"])
    u = u * u * (3 - 2 * u)
    lerp = lambda x, y: (np.asarray(x, float) + (np.asarray(y, float) - np.asarray(x, float)) * u)
    pos, look = lerp(a["pos"], b["pos"]), lerp(a.get("look", a.get("look_at")), b.get("look", b.get("look_at")))
    out = dict(cam)
    out["pos"], out["look"] = pos.tolist(), look.tolist()
    out["lens"] = float(a.get("lens", cam["lens"]) + (b.get("lens", cam["lens"]) - a.get("lens", cam["lens"])) * u)
    out["R"] = rig.look_at(pos, look).tolist()
    return out


def camera_at(cam, move, amount, t):
    if cam.get("path") and len(cam["path"]) >= 2:
        return path_camera(cam, t)
    if not move or move == "none":
        return cam
    pos, look = np.asarray(cam["pos"], float), np.asarray(cam["look"], float)
    e = ease(t, "inout")
    if move in ("dolly_in", "dolly_out"):
        sg = 1 if move == "dolly_in" else -1
        pos = pos + (look - pos) * amount * sg * e
    elif move in ("pan_left", "pan_right"):
        sg = 1 if move == "pan_left" else -1
        look = pos + rig.rodrigues(look - pos, rig.UP, sg * amount * 100 * e)     # amount 0.15 = 15 deg
    elif move in ("orbit_left", "orbit_right"):
        sg = 1 if move == "orbit_left" else -1
        pos = look + rig.rodrigues(pos - look, rig.UP, sg * amount * 100 * e)
    out = dict(cam)
    out["pos"], out["look"] = pos.tolist(), look.tolist()
    out["R"] = rig.look_at(pos, look).tolist()
    return out


def motion_interp(info0, info1, cts, N, hold, impact=1.0):
    """The legacy motion source ("interp"): per human, per frame {J, ex, fig, two} by bone slerp between
    the two staged keyframes; a figure with an END contact accelerates into it (a strike)."""
    seqs = {}
    for w, h0 in info0.items():
        if w.startswith("_") or h0.get("kind") != "human":
            continue
        h1 = info1[w]
        kind = "in" if any(c.get("who") == w and c.get("frame", "end") == "end" for c in cts) else "inout"
        seq = []
        for i in range(N):
            span = max((N - 1 - 2 * hold) * impact, 1)       # the action reaches its end pose at the impact
            tt = min(max((i - hold) / span, 0.0), 1.0)
            s_ = ease(tt, kind)
            J = interp_joints(h0["joints"], h1["joints"], s_)
            ex0, ex1 = h0["ex"], h1["ex"]
            ex = {"scale": ex0["scale"], "fwd": ex0["fwd"], "left": ex0["left"], "hold": {}, "flat": {}, "grip_at": {}}
            for hand in ("right", "left"):
                if hand in ex0["hold"] and hand in ex1["hold"]:
                    ex["hold"][hand] = C._n(slerp(C._n(ex0["hold"][hand]), C._n(ex1["hold"][hand]), s_))
                f0, f1 = ex0.get("flat", {}).get(hand), ex1.get("flat", {}).get(hand)
                if f0 is not None or f1 is not None:
                    _, _, ff0 = C.prop_axes(ex0["hold"][hand], f0)
                    _, _, ff1 = C.prop_axes(ex1["hold"][hand], f1)
                    ex["flat"][hand] = C._n(slerp(ff0, ff1, s_))
            seq.append({"J": J, "ex": ex, "fig": dict(h1["fig"] if s_ >= 0.5 else h0["fig"], anchor_fixed=True),
                        "two": h1["two_handed"] if s_ >= 0.5 else h0["two_handed"], "s": s_,
                        "model": h1["model"] or h0["model"]})
        seqs[w] = seq
    return seqs


TEMPLATES = {}
try:
    with open(os.path.join(HERE, "stage_templates", "creatures.json")) as _fh:
        TEMPLATES = json.load(_fh).get("templates", {})
except OSError:
    pass


def wing_cover(cam, J, tpl, W, H, scale=4, people=None):
    """% of a covered TARGET hull hidden by a projected wing-membrane polygon (template "membrane" order) that is
    nearer the camera than that target; the worst target counts. Targets: the creature's own torso front (chest,
    spine, neck1, fore-leg hips/knees, belly) and, if given, each PERSON (dict of body joints) -- a wing hanging
    in front of the knight hides the hero just as badly. None if nothing can be measured."""
    from PIL import Image, ImageDraw
    from scipy.spatial import ConvexHull
    w_, h_ = W // scale, H // scale
    cpos = np.asarray(cam["pos"], float)

    def pr(P3):
        pts = [rig.project(cam, [p], W, H)[0] for p in P3]
        return [(p[0] / scale, p[1] / scale) for p in pts if p is not None]
    torso_names = [n for n in ("chest", "spine1", "spine2", "spine3", "neck1", "pelvis") if n in J] + \
        [n for n in J if n.startswith("fleg_") and (n.endswith("_hip") or n.endswith("_knee"))]
    targets = [[np.asarray(J[n], float) for n in torso_names]]
    for Jp in people or ():
        targets.append([np.asarray(v, float) for k, v in Jp.items() if not k.startswith("_")
                        and np.asarray(v).shape == (3,)])
    mem = [[np.asarray(J[n], float) for n in names if n in J] for names in (tpl.get("membrane") or {}).values()]
    worst = None
    for T in targets:
        tp = pr(T)
        if len(tp) < 3:
            continue
        try:
            hull = [tp[k] for k in ConvexHull(np.array(tp)).vertices]
        except Exception:            # noqa: BLE001 - degenerate (all in a line)
            continue
        tm = Image.new("L", (w_, h_), 0)
        ImageDraw.Draw(tm).polygon(hull, fill=255)
        t_ = np.asarray(tm) > 0
        if t_.sum() < 20:
            continue
        dt = np.mean([np.linalg.norm(p - cpos) for p in T])
        cover = 0.0
        for M3 in mem:               # only a membrane IN FRONT of the target (nearer the camera) covers it
            if len(M3) < 3 or np.mean([np.linalg.norm(p - cpos) for p in M3]) >= dt:
                continue
            poly = pr(M3)
            if len(poly) >= 3:
                m1 = Image.new("L", (w_, h_), 0)
                ImageDraw.Draw(m1).polygon(poly, fill=255)
                cover = max(cover, float(((np.asarray(m1) > 0) & t_).sum()) / float(t_.sum()) * 100)
        worst = cover if worst is None else max(worst, cover)
    return worst


def detect_impact(track, end_point, tol=0.02):
    """The impact frame from the MOTION: after the speed peak of a tracked point (the striking hand), the
    first frame within tol of where it ends (or where the speed falls below 15 % of the peak)."""
    P = np.asarray(track, float)
    v = np.r_[0.0, np.linalg.norm(np.diff(P, axis=0), axis=1)]
    if v.max() < 1e-6:
        return len(P) - 1
    pk = int(np.argmax(v))
    for i in range(pk, len(P)):
        if np.linalg.norm(P[i] - end_point) < tol or v[i] < 0.15 * v[pk]:
            return i
    return len(P) - 1


def render_frames(shot, info0, info1, seqs, cam, move, move_amount, N, fdir):
    """CLEANUP + VALIDATION + scene frames for any per-frame poses (seqs: who -> [{J, ex, fig, two, model}]):
    foot lock, power grips (fast), contacts from their phase (impact detected from the motion), fist
    proxies, fx at the impact, checks on every frame."""
    cts = shot.get("contacts", [])
    W_, H_ = cam["size"]
    rest_pitch = {}
    for w, seq in seqs.items():                           # feet: planted feet stay put, FLAT, never below ground
        rp = {"_": C._canon_foot_pitch()}
        mdl = seq[0].get("model") or {}
        rj = os.path.join(os.path.dirname(mdl["blend"]), "rig.json") if mdl.get("blend") else None
        if rj and os.path.exists(rj):
            RJ = json.load(open(rj))["joints"]
            for sd in ("r", "l"):
                if f"{sd}_ankle" in RJ and f"{sd}_toe" in RJ:
                    v = np.asarray(RJ[f"{sd}_toe"], float) - np.asarray(RJ[f"{sd}_ankle"], float)
                    rp[sd] = math.atan2(float(v[2]), float(np.hypot(v[0], v[1])))
        rest_pitch[w] = rp
        C.lock_feet([fr["J"] for fr in seq], rest_pitch=rp)
    impacts = {}
    for c in cts:                                         # impact frame per END contact, from the motion
        if c.get("frame", "end") != "end" or c["who"] not in seqs:
            continue
        hand = next((h for h, it in (seqs[c["who"]][-1]["fig"].get("holds") or {}).items() if it == c["prop"]), "right")
        s = "r" if hand == "right" else "l"
        track = [fr["J"][f"{s}_hand"] for fr in seqs[c["who"]]]
        ci_ = cts.index(c)
        fixed = (IMPACTS.get(ci_) if ci_ in IMPACTS else IMPACTS.get(c["who"])) if IMPACTS else None
        if fixed is None:
            fixed = (shot.get("timing") or {}).get("impact_frame")
        if fixed is None and shot.get("impact") == "none":
            impacts[id(c)] = 10 ** 9                       # a HOLD: the end contact never engages
            continue
        # a motion source knows its impact (a lunge's hand slows WHILE extending: detection is a fallback)
        impacts[id(c)] = int(fixed) if fixed is not None else detect_impact(track, track[-1])
    creatures = [w for w, v in info0.items() if not w.startswith("_") and v.get("kind") == "creature"]
    frames, metas, all_checks, nbad, worst = [], [], [], 0, {}
    roll_prev = {}
    axis_prev = {}
    push_off = {}                       # per human: horizontal offset (m) out of creature bodies, carried on
    ops = []                            # OPERATED set props (10-22: the bellows pump): (op, prop entry, mesh)
    for op in shot.get("operate") or []:
        pr_ = next((q for q in (shot.get("set") or {}).get("props", []) if (q.get("prop") or q.get("name")) == op["prop"]), None)
        if pr_ is not None and P_.joint_handle(pr_, op["joint"], None) is not None:
            ops.append((op, pr_, P_.set_entry(pr_)[1].get("mesh")))
    for i in range(N):
        prims = list(info0["_set_prims"])
        op_ang = {}
        for op, pr_, mesh_ in ops:      # the part's angle this frame; its mesh turned to it in the render
            ang_ = P_.joint_cycle(pr_, op["joint"], i / float(shot.get("fps", 24)))
            op_ang[id(op)] = ang_
            prims = [dict(q, joints={**(q.get("joints") or {}), op["joint"]: ang_})
                     if q.get("t") == "propmesh" and q.get("blend") == mesh_ else q for q in prims]
        models, pts2d, anchors, cprims = [], [], {}, {}
        figmeta = []
        tt = seqs[next(iter(seqs))][i]["s"] if seqs else i / max(N - 1, 1)
        for p in prims:
            p["layer"] = "set"
        for w in creatures:
            s_ = ease(tt, "inout")
            P = [dict(p, layer="creature") for p in lerp_prims(info0[w]["prims"], info1[w]["prims"], s_)]
            prims += P
            cprims[w] = P
            A0, A1 = info0["_anchors"][w], info1["_anchors"][w]
            anchors[w] = {k: np.asarray(A0[k]) + (np.asarray(A1.get(k, A0[k])) - np.asarray(A0[k])) * s_ for k in A0}
        for m0 in info0["_models"]:
            if m0.get("static"):
                m1 = next((m for m in info1["_models"] if m.get("static") and m.get("who") == m0.get("who")), m0)
                M = np.asarray(m0["M"]) + (np.asarray(m1["M"]) - np.asarray(m0["M"])) * ease(tt, "inout")
                w0 = m0.get("who")
                cm = CREATURE_MOTION.get(w0)
                if cm is not None and m0.get("joints"):
                    # a rigged creature with per-frame poses (template joint names, world metres): the
                    # renderer poses its bones; contacts/anchors follow the posed joints
                    Jc = {k_: np.asarray(v_, float) for k_, v_ in cm[i].items()}
                    models.append(dict(m0, M=M.tolist(), static=False, template=True,
                                       targets={k_: v_.tolist() for k_, v_ in Jc.items()}))
                    rjp = os.path.join(os.path.dirname(m0["blend"]), "rig.json")
                    rjd = json.load(open(rjp))
                    yaw = math.atan2(M[1, 0], M[0, 0])
                    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
                    blen = float(np.linalg.norm(Jc["chest"] - Jc["pelvis"])) if "chest" in Jc else 1.0
                    A = {n_: q for n_, q in Jc.items()}
                    for an, ad_ in rjd.get("anchors", {}).items():
                        if ad_["joint"] in Jc:
                            A[an] = Jc[ad_["joint"]] + Rz @ (np.asarray(ad_["offset"], float) * blen)
                    anchors[w0] = A
                else:
                    models.append(dict(m0, M=M.tolist()))
                # creature VOLUME this frame (rigged creatures): capsules along the posed bones
                rjp_ = os.path.join(os.path.dirname(m0["blend"]), "rig.json")
                if m0.get("joints") and os.path.exists(rjp_):
                    rjd_ = json.load(open(rjp_))
                    Jv = models[-1].get("targets") or m0["joints"]
                    tp_ = TEMPLATES.get(rjd_.get("template"), {})
                    if tp_:
                        cprims[w0] = C.creature_capsules({k_: np.asarray(v_, float) for k_, v_ in Jv.items()},
                                                         tp_["parents"], rjd_.get("radii", {}),
                                                         float(np.linalg.norm(np.asarray(M)[:3, 0])))
        # creature readability: a wing membrane must not cover the torso front (chest/belly/fore legs) from
        # the shot camera (it reads as an apron/cape); projected overlap of membrane polygons vs torso hull
        people_i = [seq[i]["J"] for seq in seqs.values()]      # this frame's people (pre-cleanup is close enough)
        for mm in models:
            if not mm.get("joints") and not mm.get("targets"):
                continue
            Jc = {k_: np.asarray(v_, float) for k_, v_ in (mm.get("targets") or mm["joints"]).items()}
            tpl_ = TEMPLATES.get("winged_quadruped") if any(k_.startswith("wing_") for k_ in Jc) else None
            if tpl_ is None:
                continue
            ci_ = camera_at(cam, move, move_amount, i / max(N - 1, 1))
            cov = wing_cover(ci_, Jc, tpl_, W_, H_, people=people_i)
            if cov is not None and cov > 25.0:
                # REFOLD: swing each wing back about its shoulder (vertical axis), 10 deg steps up to 90,
                # in the direction that uncovers the torso; the renderer then poses the bones
                best = (cov, Jc)
                for sgn in (1, -1):
                    for ang in range(10, 100, 10):
                        Jt = {k_: v_.copy() for k_, v_ in Jc.items()}
                        for sd in ("l", "r"):
                            sh_ = Jt.get(f"wing_{sd}_shoulder")
                            if sh_ is None:
                                continue
                            a_ = ang * sgn * (1 if sd == "l" else -1)
                            for k_ in [k_ for k_ in Jt if k_.startswith(f"wing_{sd}_") and not k_.endswith("shoulder")]:
                                Jt[k_] = sh_ + rig.rodrigues(Jt[k_] - sh_, rig.UP, a_)
                        c2 = wing_cover(ci_, Jt, tpl_, W_, H_, people=people_i)
                        if c2 is not None and c2 < best[0]:
                            best = (c2, Jt)
                        if c2 is not None and c2 <= 25.0:
                            break
                    if best[0] <= 25.0:
                        break
                if best[0] < cov:
                    cov, Jc = best
                    mm.update(static=False, template=True, targets={k_: v_.tolist() for k_, v_ in Jc.items()})
            if cov is not None:
                r_ = {"who": mm.get("who"), "check": "wing membrane over the torso front or a person", "value_pct": round(cov, 1),
                      "limit": 25.0}
                all_checks.append({"frame": i, "who": mm.get("who"), "rows": [r_],
                                   "violations": [r_] if cov > 25.0 else []})
                nbad += cov > 25.0
        for w, seq in seqs.items():
            fr_ = seq[i]
            J, ex, f, two = fr_["J"], fr_["ex"], fr_["fig"], fr_["two"]
            ex["grip"] = {}
            # SAFETY NET: never inside a creature's body. Carry the previous frame's push (easing back 1 cm a
            # frame, so the figure glides rather than pops), then push out of every creature volume this
            # frame (whole figure, horizontal). Logged as a cleanup step, never silent.
            off = push_off.get(w, np.zeros(3))
            if np.linalg.norm(off) > 0:
                off = off * max(0.0, 1 - 0.01 / max(np.linalg.norm(off), 1e-9))
            if np.linalg.norm(off) > 0:
                C._move_figure(J, off)
            mv_ = 0.0
            for cw, caps in cprims.items():
                if caps and cw != w:
                    p0 = np.asarray(J["pelvis"], float).copy()
                    mv_ += C.push_out(J, caps)
                    off = off + (np.asarray(J["pelvis"], float) - p0)
            push_off[w] = off
            if np.linalg.norm(off) > 0.005:
                r_ = {"who": w, "check": "cleanup: pushed out of a creature body", "value_cm": round(float(np.linalg.norm(off)) * 100, 1),
                      "new_this_frame_cm": round(mv_ * 100, 1)}
                all_checks.append({"frame": i, "who": w, "cleanup": "push_out", "rows": [r_], "violations": []})
                if mv_ > 0:
                    print(f"[f{i}] cleanup: {w} pushed {mv_ * 100:.0f} cm out of a creature body "
                          f"(offset now {np.linalg.norm(off) * 100:.0f} cm)", file=sys.stderr)
            engaged = [c for c in cts if c.get("who") == w and (c.get("frame") == "both"
                       or (c.get("frame", "end") == "end" and i >= impacts.get(id(c), N)
                           and (c.get("hold_frames") is None or i <= impacts.get(id(c), N) + int(c["hold_frames"])))
                       or (c.get("frame") == "start" and i == 0))]
            # a released SURFACE contact keeps its surface as a FLOOR: on any later frame where the
            # motion would push the prop's colliding part through it, the contact re-engages (the prop
            # rests on the surface and the arm follows) until the motion lifts it clear again
            for c in [c for c in cts if c.get("who") == w and c.get("hold_frames") is not None
                      and c not in engaged and ":" not in c["to"] and i > impacts.get(id(c), N)]:
                Jt = C._copyW(J)
                ext = dict(ex, grip={}, hold=dict(ex["hold"]), grip_at=dict(ex.get("grip_at", {})),
                           flat=dict(ex.get("flat", {})))
                C.grip_all(Jt, ext, f, two_handed=two, fast=True)
                r_, _ = C.check(Jt, ext, f, [c], shot.get("set"), anchors, creature_prims=cprims)
                if any("deepest" in r["check"] and r["value_cm"] < 0 for r in r_):
                    engaged.append(c)
            # a hand whose two-handed grip is ending lets go when it can no longer hold (rather than
            # dragging the arm): try the regrip on a copy, keep it only if the fist really reaches
            if two and not (seq[-1]["two"] and seq[0]["two"]):
                Jt, ext = C._copyW(J), dict(ex, grip={}, hold=dict(ex["hold"]), grip_at=dict(ex["grip_at"]))
                C.grip_all(Jt, ext, f, two_handed=True, fast=True, skip=[c["prop"] for c in engaged])
                if ext["grip"].get("left2", {}).get("residual", 1.0) > 0.01:
                    two = False
            C.grip_all(J, ex, f, two_handed=two, fast=True, skip=[c["prop"] for c in engaged])
            # a FREE prop's handle axis may not jump between frames (the per-frame axis choice can hop
            # between the limits of the wrist range): at most 12 deg a frame, then the fist re-grips it
            for hand, item in (f.get("holds") or {}).items():
                if not P_.has_grip(item) or item in [c["prop"] for c in engaged] or hand not in ex.get("hold", {}):
                    continue
                key_ = (w, hand)
                d_ = C._n(ex["hold"][hand])
                pd = axis_prev.get(key_)
                if pd is not None:
                    ang = math.degrees(math.acos(float(np.clip(np.dot(pd, d_), -1, 1))))
                    if ang > 12.0:
                        ax_ = np.cross(pd, d_)
                        if np.linalg.norm(ax_) > 1e-6:
                            d_ = C._n(rig.rodrigues(pd, ax_, 12.0))
                            ex["hold"][hand] = d_
                            s_ = "r" if hand == "right" else "l"
                            C.solve_grip(J, s_, ex["grip_at"][hand], d_, ex, pose_pole=True)
                            rf = C.roll_flat(J, s_, item, d_)
                            if rf is not None:
                                ex["flat"][hand] = rf
                axis_prev[key_] = d_
            for c in engaged:                                # follow-through: the point drives on past
                if c.get("follow") and id(c) in impacts:     # the contact after the impact (ease-out)
                    u = (i - impacts[id(c)]) / max(N - 1 - impacts[id(c)], 1)
                    c["_push"] = float(c["follow"]) * (1 - (1 - min(max(u, 0), 1)) ** 2)
            C.solve_contacts(J, ex, f, engaged, shot.get("set"), anchors, two_handed=two, creature_prims=cprims)
            for op, pr_, _ in ops:                       # the operator's hands ride the moving handle
                if op.get("who") != w:
                    continue
                hw_, pw_ = P_.joint_handle(pr_, op["joint"], op_ang[id(op)])
                sd_ = C._n(np.cross(hw_ - pw_, rig.UP))
                ta, tb = hw_ + sd_ * 0.08, hw_ - sd_ * 0.08
                if np.linalg.norm(ta - J["r_shoulder"]) > np.linalg.norm(tb - J["r_shoulder"]):
                    ta, tb = tb, ta
                rows_ = []
                for s_, tg in (("r", ta), ("l", tb)):
                    res_ = C.ik_hand(J, s_, tg)
                    rows_.append({"who": w, "check": f"{s_} hand on the {op['prop']} handle", "value_cm": round(res_ * 100, 1),
                                  "limit": 3.0})
                bad_ = [r_ for r_ in rows_ if r_["value_cm"] > 3.0]
                all_checks.append({"frame": i, "who": w, "rows": rows_, "violations": bad_})
                nbad += bool(bad_)
            # ROLL: the head/edge leads the stroke (prop data "roll"), converges on the contact's flush
            # orientation over the 8 frames before an impact, never turns > 30 deg a frame about the handle
            roll_rows = []
            for hand, item in (f.get("holds") or {}).items():
                rr = P_.get(item).get("roll") if P_.has_grip(item) else None
                if not rr or hand not in ex.get("hold", {}):
                    continue
                s_ = "r" if hand == "right" else "l"
                d_ = C._n(ex["hold"][hand])
                eng = next((c for c in engaged if c["prop"] == item), None)
                if eng is not None and C._aff(eng)[1].get("type") not in ("face", "lie"):
                    eng = None                        # a tip/edge contact does not fix the roll: keep the rule
                    rf_ = C.roll_flat(J, "r" if hand == "right" else "l", item, ex["hold"][hand])
                    if rf_ is not None:
                        ex["flat"][hand] = rf_
                target = ex.get("flat", {}).get(hand)
                if target is None:
                    continue
                for c in cts:                         # approach: blend toward the impact orientation
                    if c.get("who") == w and c["prop"] == item and c.get("frame", "end") == "end" and \
                            ":" not in c["to"] and id(c) in impacts and impacts[id(c)] - 8 <= i < impacts[id(c)]:
                        _, n_, _ = C.surface_point(shot.get("set"), c)
                        aff_ = P_.affordance(item, C.LEGACY_POINT.get(c.get("point"), c.get("point")))
                        fn = np.asarray(aff_.get("normal", [0, 1, 0]), float)
                        flush = -n_ if abs(fn[1]) >= abs(fn[2]) else np.cross(-n_, d_)
                        u = (i - (impacts[id(c)] - 8) + 1) / 9.0
                        target = C._n((1 - u) * C._n(target) + u * C._n(flush))
                key_ = (w, hand)
                prev = roll_prev.get(key_)
                # the forearm rule's sign is undefined when the forearm passes near the handle axis: keep
                # the side nearer the previous frame (no 180-degree flip of the head mid-swing)
                if prev is not None and eng is None and float(np.dot(C._n(target), prev)) < -0.2:
                    target = -C._n(target)
                if prev is None or eng is not None:
                    new_, rate = C._n(target - d_ * np.dot(target, d_)), 0.0
                    if prev is not None:
                        rate = C.roll_limit(prev, target, d_, 1e9)[1]
                else:
                    new_, rate = C.roll_limit(prev, target, d_, 30.0)
                ex["flat"][hand] = new_
                roll_prev[key_] = new_
                fa_ = C._n(J[f"{s_}_wrist"] - J[f"{s_}_elbow"])
                pn = np.cross(fa_, d_)
                ax = new_ if rr.get("axis", "flat") == "flat" else C._n(np.cross(d_, new_))
                approach = any(c.get("who") == w and c["prop"] == item and id(c) in impacts
                               and impacts[id(c)] - 8 <= i < impacts[id(c)] for c in cts)
                # (the 8 approach frames turn the face toward the flush impact pose by design: rate-checked only)
                if np.linalg.norm(pn) > 0.2 and eng is None and not approach:
                    outp = math.degrees(math.asin(min(1.0, abs(float(np.dot(ax, C._n(pn)))))))
                    tol = float(rr.get("plane_tol_deg", 20.0))     # per prop: a sword's edge may cant more
                    roll_rows.append({"who": w, "check": f"{item} {rr.get('axis')} out of the swing plane",
                                      "value_deg": round(outp, 2), "limit": tol, "_ok": outp <= tol})
                roll_rows.append({"who": w, "check": f"{item} roll rate", "value_deg": round(abs(rate), 2),
                                  "limit": 30.0, "_ok": abs(rate) <= 30.0 + 1e-6})
            rows, bad = C.check(J, ex, f, engaged, shot.get("set"), anchors, two_handed=two, creature_prims=cprims)
            r2, b2 = C.grip_checks(J, ex, f, engaged_items=engaged)
            rows, bad = rows + r2, bad + b2
            for sd in ("r", "l"):                          # soles: flat while planted, toes forward (no splay)
                if f"{sd}_toe" not in J:
                    continue
                low = bool(J.get(f"_planted_{sd}"))           # pinned by lock_feet (a planted run)
                pr = rest_pitch.get(w, {})
                pdev, splay = C.sole_rows(J, sd, pr.get(sd, pr.get("_", 0.08)))
                if low:
                    r_ = {"who": w, "check": f"{sd} sole pitch vs flat (planted)", "value_deg": round(pdev, 1), "limit": 10.0}
                    rows.append(r_)
                    if abs(pdev) > 10.0:
                        bad.append(r_)
                r_ = {"who": w, "check": f"{sd} toes vs facing", "value_deg": round(splay, 1), "limit": C.SPLAY_CHECK}
                rows.append(r_)
                if splay > C.SPLAY_CHECK and low:
                    bad.append(r_)
            for cw, caps in cprims.items():                  # people never inside a creature's body
                if caps and cw != w:
                    dep, jn = C.human_in_creature(J, caps)
                    r_ = {"who": w, "check": f"inside {cw}'s body", "value_cm": round(dep * 100, 2), "limit": 1.0}
                    rows.append(r_)
                    if dep > 0.01:
                        bad.append(dict(r_, check=f"inside {cw}'s body ({jn})"))
            for r in roll_rows:
                ok = r.pop("_ok")
                rows.append(r)
                if not ok:
                    bad.append(r)
            for c in [c for c in cts if c.get("who") == w and c not in engaged and not c.get("penetrate")]:
                r3, _ = C.check(J, ex, f, [c], shot.get("set"), anchors, creature_prims=cprims)
                for r in r3:                              # on the way in: no penetration
                    if ("deepest" in r["check"] or "penetration" in r["check"]) and r["value_cm"] < -C.LIMITS["pen"] * 100:
                        bad.append(dict(r, check=r["check"] + " (on the way in)"))
                        rows.append(bad[-1])
            for r in rows:
                key = (w, r["check"])
                v = r.get("value_cm", r.get("value_deg"))
                worst[key] = (min(worst.get(key, 1e9), v) if ("deepest" in key[1] or "penetration" in key[1]
                              or "off the torso" in key[1] or "in front" in key[1]) else max(worst.get(key, -1e9), abs(v)))
            all_checks.append({"frame": i, "who": w, "rows": rows, "violations": bad})
            nbad += len(bad)
            fists = sorted({gi["side"] for gi in ex["grip"].values()})
            hp = rig.human_prims(J, ex["scale"], wear=() if f.get("model") else f.get("wear", ()),
                                 hold_items=f.get("holds"), hold_dirs=ex["hold"], fwd=ex["fwd"], left=ex["left"],
                                 hold_flats=ex.get("flat"), hold_at=ex.get("grip_at"), fists=fists)
            hp += C.fist_prims(ex, C.pk(ex))
            if fr_["model"] and not MANNEQUIN[0]:     # (--mannequin: the capsule body from the same joints)
                m = dict(fr_["model"])
                m["targets"] = {k: np.asarray(v).tolist() for k, v in J.items() if not k.startswith("_")}
                m["ground"] = False
                m["fist"] = fists
                models.append(m)
                hp = [p for p in hp if p.get("tag") != "body"]
            prims += [dict(p, layer="figure") for p in hp]
            # the face points only where the face turns to the camera (10-23: drawn always, every skeleton read
            # as facing the viewer and the keys turned Edric round from the dragon he faced in the staging)
            pts2d.append(rig.coco18(J, camera_at(cam, move, move_amount, i / max(N - 1, 1))["pos"]))
            # what a drawn keyframe must show (keyframe_check.py): each figure, each gripping fist on its
            # handle, the handle's working part away from the fist
            gm = []
            for hk, gi in (ex.get("grip") or {}).items():
                hand = "right" if hk in ("right", "left2") else "left"
                item = (f.get("holds") or {}).get(hand)
                if not item or not P_.has_grip(item):
                    continue
                at = ex.get("grip_at", {}).get(hand, J[f"{gi['side']}_hand"])
                a_, b_ = P_.handle_seg(item, at, ex["hold"][hand], C.pk(ex))
                wk = P_.get(item).get("work")
                wp = P_.point(item, wk, at, ex["hold"][hand], ex.get("flat", {}).get(hand), C.pk(ex)) if wk else None
                gm.append({"who": w, "hand": "r" if gi["side"] == "r" else "l", "item": item, "fist": gi["fist"],
                           "handle": [a_, b_], "work": wp, "work_name": wk})
            cm = []
            for c_ in engaged:
                try:
                    Pc = (C.surface_point(shot.get("set"), c_)[0] if ":" not in c_["to"]
                          else anchors[c_["to"].split(":")[0]][c_["to"].split(":")[1]])
                    cm.append({"prop": c_["prop"], "point": c_.get("point"), "P": Pc})
                except Exception:              # noqa: BLE001 - anchors only
                    pass
            figmeta.append({"who": w, "joints": [v for k, v in J.items() if not k.startswith("_")], "grips": gm,
                            "hands": {"r": J["r_hand"], "l": J["l_hand"]}, "contacts": cm})
        # fx (BLOCKOUT_VIDEO_FX=1): at the impact frame of their contact, growing over ~0.6 s. Off by default:
        # in Mode 2 the video model makes sparks/dust itself; geometry would only add false depth
        for k_, fx in enumerate(shot.get("fx", []) if os.environ.get("BLOCKOUT_VIDEO_FX") else []):
            ci_ = fx.get("at", {}).get("contact") if isinstance(fx.get("at"), dict) else None
            c = cts[ci_] if ci_ is not None else None
            i0 = impacts.get(id(c), N - 1) if c is not None else N - 1
            if fx.get("frame", "end") in ("end", "impact") and i0 <= i < i0 + 15:
                prims += B.fx_prims([dict(fx, frame="end", seed=fx.get("seed", 1) + (i - i0) // 3)], "end", cts,
                                    shot.get("set"), anchors, t=(i - i0 + 1) / 15.0)
        ci = camera_at(cam, move, move_amount, i / max(N - 1, 1))
        prims += B.room_walls(shot.get("set") or {}, ci["pos"])
        fr = {"name": f"f{i:04d}", "prims": [B._jsonable(p) for p in prims], "models": models,
              "camera": {"pos": ci["pos"], "R": ci["R"], "lens": ci["lens"]},
              "clay": os.path.join(fdir, f"clay_{i:04d}.png"), "depth": os.path.join(fdir, f"depth_{i:04d}.npy")}
        if (shot.get("set") or {}).get("world"):
            fr["world"] = shot["set"]["world"]
        if LAYERS:
            fr["layers"] = LAYERS
        frames.append(fr)
        metas.append((ci, pts2d, figmeta))
    return frames, metas, all_checks, nbad, worst, {k: v for k, v in impacts.items()}


def main(argv=None, motion=None, impacts=None) -> int:
    """motion: {who: [joint dict] * N} (per-frame poses, e.g. tools/motion_shot.py); impacts: {contact
    index | who: frame} (the motion's own impact frames; else shot timing.impact_frame; else detected)."""
    IMPACTS.clear()
    IMPACTS.update(impacts or {})
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("out")
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--size", default="768x448")
    ap.add_argument("--move", default=None)
    ap.add_argument("--move-amount", type=float, default=0.15)
    ap.add_argument("--hold", type=int, default=6, help="frames held on the start pose and on the end pose")
    ap.add_argument("--keep-frames", action="store_true")
    ap.add_argument("--target", choices=["ltx", "none"], default="none",
                    help="ltx: 1152x640 (W, H multiples of 128 for the ref-0.5 IC-LoRA), 121 frames at 24 fps")
    ap.add_argument("--stills", action="store_true",
                    help="also render frames 0 and N-1 in COLOUR (Cycles) at the same size/camera: "
                         "start_color.png / end_color.png + start_depth.png / end_depth.png (Qwen keyframes)")
    ap.add_argument("--key-frames", action="store_true",
                    help="Blender renders only frames 0 and N-1 (the stills, their depth / figure mask / expectations); "
                         "the motion is still staged and CHECKED on every frame. No clip passes: a later run without "
                         "it renders the clip (10-21: the staging step spent ~90 s a moment on 121 frames nobody "
                         "looked at before the video step)")
    ap.add_argument("--all-passes", action="store_true",
                    help="also encode pose / canny / clay / depth_static and the half-size clips (nothing reads them)")
    ap.add_argument("--mannequin", action="store_true", help="people as capsule mannequins (no character mesh)")
    ap.add_argument("--layers", default="static=set+creature,figures=figure",
                    help="extra depth passes by object layer (set, creature, figure, fx): "
                         "name=layer+layer,... -> depth_<name>.mp4; '' for none")
    a = ap.parse_args(argv)
    MANNEQUIN[0] = bool(a.mannequin)
    LAYERS.clear()
    for part in filter(None, a.layers.split(",")):
        nm, keep = part.split("=")
        LAYERS[nm] = keep.split("+")
    if a.target == "ltx":
        if a.size == "768x448":
            a.size = "1152x640"
    W, H = (int(x) for x in a.size.lower().split("x"))
    assert (a.frames - 1) % 8 == 0, "LTX wants frames = 1 + 8k"
    assert W % 64 == 0 and H % 64 == 0, "LTX wants W, H multiples of 64"
    if a.target == "ltx":
        assert W % 128 == 0 and H % 128 == 0, "LTX + ref-0.5 IC-LoRA wants W, H multiples of 128"
    shot = B.resolve_set(json.load(open(a.shot)))
    if shot.get("book"):
        P_.load_book(shot["book"])                    # the book's modelled props render as meshes
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    fdir = os.path.join(out, "frames")
    os.makedirs(fdir, exist_ok=True)
    t0 = time.time()

    # stage the keyframes exactly as blockout.run does (contacts solved, contact-reach steps propagated)
    frames_kf = [("start", copy.deepcopy(shot["figures"]))]
    endf = B.merge_end(shot)
    frames_kf.append(("end", endf or copy.deepcopy(shot["figures"])))
    staged = B.stage_frames(shot, frames_kf)
    info0, info1 = staged[0][4], staged[1][4]
    figs0 = staged[0][1]
    cts = shot.get("contacts", [])
    cinfo = dict(info0)                                   # camera over both frames (as blockout.run)
    for who, v in info1.items():
        if who.startswith("_") or who not in cinfo:
            continue
        u = dict(cinfo[who])
        u["points"] = list(u["points"]) + list(v["points"])
        if "anchors" in u:
            u["anchors"] = {k: np.vstack([np.atleast_2d(x), np.atleast_2d(v["anchors"].get(k, x))])
                            for k, x in u["anchors"].items()}
        cinfo[who] = u
    cam_spec = dict(shot.get("camera", {}))
    cam = B.solve_camera(cam_spec, figs0, cinfo, (W, H))
    cam = B.after_solve(cam, cam_spec, cinfo, shot)  # the same camera steps as blockout.run (10-22: three were missing)
    path = (shot.get("camera") or {}).get("path")
    if path and len(path) >= 2:                  # an authored 3D camera path: frame 0 is the solved start
        cam = dict(cam, path=path)
        cam = dict(path_camera(cam, 0.0), path=path)
    # readability: re-stage with the camera known (free held props turn so the working part reads)
    B.STAGE_VIEW["pos"] = cam["pos"]
    try:
        staged = B.stage_frames(shot, frames_kf)
    finally:
        B.STAGE_VIEW.clear()
    info0, info1 = staged[0][4], staged[1][4]
    move = a.move or (shot.get("camera") or {}).get("move")
    N = a.frames
    seqs = motion_interp(info0, info1, cts, N, a.hold, float((shot.get("timing") or {}).get("impact", 1.0)))
    CREATURE_MOTION.clear()
    if motion:                                            # PER-FRAME POSE INTERFACE: {who: [joints] * N}
        for w in [w for w in motion if w not in seqs]:   # creatures (not humans): posed in the renderer
            CREATURE_MOTION[w] = motion.pop(w)
        for w, js in motion.items():
            seqs[w] = [dict(seqs[w][i], J={k: np.asarray(v, float) for k, v in js[i].items()}) for i in range(N)]
    t_clean = time.time()
    frames, metas, all_checks, nbad, worst, impacts = render_frames(shot, info0, info1, seqs, cam, move,
                                                                    a.move_amount, N, fdir)
    nbad += 0
    t1 = time.time()
    sys.stderr.write(f"staged {N} frames in {t1 - t0:.1f} s; violations {nbad}\n")
    for (w, k), v in sorted(worst.items()):
        sys.stderr.write(f"  worst over the clip: {w}: {k} = {v:.2f}\n")
    with open(os.path.join(out, "checks.json"), "w") as fh:
        json.dump({"violations": nbad, "worst": {f"{w}: {k}": v for (w, k), v in worst.items()},
                   "frames": all_checks}, fh, default=lambda o: np.asarray(o).tolist())

    sel = sorted({0, N - 1}) if a.key_frames else list(range(N))
    spath = os.path.join(out, "scene.json")
    with open(spath, "w") as fh:
        json.dump({"size": [W, H], "samples": 1, "engine": "workbench", "frames": [frames[i] for i in sel]}, fh,
                  default=lambda o: np.asarray(o).tolist())
    r = subprocess.run([B.BLENDER, "--background", "--factory-startup", "-t", os.environ.get("BLOCKOUT_THREADS", "16"),
                        "--python", os.path.join(HERE, "blockout_scene.py"), "--", spath],
                       capture_output=True, text=True, timeout=float(os.environ.get("BLOCKOUT_TIMEOUT", 1800)))
    done = [ln for ln in r.stdout.splitlines() if ln.startswith("BLOCKOUT ")]
    with open(os.path.join(out, "blender_times.txt"), "w") as fh:     # per-frame build / render / layers (10-23)
        fh.write("\n".join(done) + "\n")
    if r.returncode or len(done) != len(sel):
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise RuntimeError("blender failed")
    t2 = time.time()
    # model hands in Blender vs solved, every frame
    hand_dev = 0.0
    for ln in r.stdout.splitlines():
        if ln.startswith("BLOCKOUT_HANDS "):
            hj = json.loads(ln.split(" ", 1)[1])
            i = int(hj["frame"][1:])
            for s_, (h_, t_) in hj["hands"].items():
                mdl = next(m for m in frames[i]["models"] if m.get("who") == hj["who"])
                hand_dev = max(hand_dev, float(np.linalg.norm(np.array(h_) - np.asarray(mdl["targets"][f"{s_}_wrist"]))))
    if hand_dev > C.LIMITS["grip"]:
        nbad += 1
        sys.stderr.write(f"CONTACT VIOLATION: model hand Blender vs solved {hand_dev * 100:.2f} cm\n")
    sys.stderr.write(f"model hands Blender vs solved, worst over the clip: {hand_dev * 100:.2f} cm\n")

    # passes: one depth normalisation for the whole clip
    from PIL import Image
    deps = {i: np.load(frames[i]["depth"]) for i in sel}
    inv = np.concatenate([1.0 / d[(d > 0) & (d < 1e6)][::37] for d in deps.values()])
    near, far = np.percentile(inv, 99.5), np.percentile(inv, 1.0)
    for i, d in deps.items():
        fr = frames[i]
        img = np.zeros(d.shape)
        v = (d > 0) & (d < 1e6)
        img[v] = 0.12 + 0.88 * np.clip((1.0 / d[v] - far) / max(near - far, 1e-9), 0, 1)
        Image.fromarray((img * 255).round().astype(np.uint8)).convert("RGB").save(os.path.join(fdir, f"d_{i:04d}.png"))
        for ln in LAYERS:                                  # the same near/far as depth_all
            dl = np.load(fr["depth"].replace(".npy", f"_{ln}.npy"))
            il = np.zeros(dl.shape)
            vl = (dl > 0) & (dl < 1e6)
            il[vl] = 0.12 + 0.88 * np.clip((1.0 / dl[vl] - far) / max(near - far, 1e-9), 0, 1)
            Image.fromarray((il * 255).round().astype(np.uint8)).convert("RGB").save(
                os.path.join(fdir, f"d{ln}_{i:04d}.png"))
        ci, pts = metas[i][:2]
        pp = [rig.project(ci, [p if p is None else np.asarray(p) for p in person], W, H) for person in pts]
        Image.fromarray(draw_people(pp, W, H)).save(os.path.join(fdir, f"p_{i:04d}.png"))
        B.canny_png(fr["clay"], os.path.join(fdir, f"c_{i:04d}.png"))
    t3 = time.time()

    def mp4(pattern, name, scale=None):
        # Fedora's ffmpeg has no libx264: encode with PyAV's libx264 from the LTX venv (the consumer's)
        size = scale.replace(":", "x") if scale else ""
        subprocess.run([PYAV, "-I", "-c", ENCODE, os.path.join(fdir, pattern), str(N), str(a.fps),
                        os.path.join(out, name), size], check=True)

    # the clip passes: depth (LTX Mode 2) + depth_figures (the keyframe's figure mask); the rest only on request
    made = []
    if not a.key_frames:
        for pat, nm in [("d_%04d.png", "depth"), ("p_%04d.png", "pose"), ("c_%04d.png", "canny"), ("clay_%04d.png", "clay")] + \
                [(f"d{ln}_%04d.png", f"depth_{ln}") for ln in LAYERS]:
            if a.all_passes or nm in ("depth", "depth_figures"):
                mp4(pat, f"{nm}.mp4")
                made.append(nm)
            if a.all_passes and nm != "clay":
                mp4(pat, f"{nm}_half.mp4", f"{W // 2}:{H // 2}")
                made.append(f"{nm}_half")
    if "figures" in LAYERS:                        # frame 0's figures-only depth: the keyframe's figure mask
        shutil.copy(os.path.join(fdir, "dfigures_0000.png"), os.path.join(out, "figures0.png"))
    # frame strip: 8 frames x (clay, depth, pose, canny)
    idx = np.linspace(0, N - 1, 8).round().astype(int) if not a.key_frames else np.array(sel)
    tw, th = 320, int(320 * H / W)
    S = Image.new("RGB", (tw * 8, th * 4), (0, 0, 0))
    for j, i in enumerate(idx):
        for r_, pat in enumerate((f"clay_{i:04d}.png", f"d_{i:04d}.png", f"p_{i:04d}.png", f"c_{i:04d}.png")):
            S.paste(Image.open(os.path.join(fdir, pat)).convert("RGB").resize((tw, th)), (j * tw, r_ * th))
    S.save(os.path.join(out, "strip.jpg"), quality=86)
    # per-frame 2D anchors (output pixels at --size): hands, grips, working parts, contact targets; for
    # trackers that measure a generated clip against the true per-frame expectations
    anc = {"size": [W, H], "fps": a.fps, "frames": []}
    for i in range(N):
        ci, _, fm = metas[i]

        def pr(p_):
            return None if p_ is None else rig.project(ci, [np.asarray(p_, float)], W, H)[0]
        fr_a = {"frame": i, "figures": []}
        for fg in fm:
            fr_a["figures"].append({"who": fg["who"], "hands": {k: pr(v) for k, v in fg.get("hands", {}).items()},
                                    "grips": [{"hand": g["hand"], "item": g["item"], "fist": pr(g["fist"]),
                                               "grip_segment": [pr(g["handle"][0]), pr(g["handle"][1])],
                                               "work": pr(g["work"]), "work_name": g["work_name"]}
                                              for g in fg["grips"]],
                                    "contacts": [{"prop": c_["prop"], "point": c_["point"], "target": pr(c_["P"])}
                                                 for c_ in fg.get("contacts", [])]})
        anc["frames"].append(fr_a)
    with open(os.path.join(out, "anchors.json"), "w") as fh:
        json.dump(anc, fh, default=lambda o: np.asarray(o).tolist())
    if a.stills:                                   # colour keyframes, the exact frame 0 / N-1 scenes
        st = {"size": [W, H], "samples": 16, "frames": []}
        for nm, i in (("start", 0), ("end", N - 1)):
            st["frames"].append(dict(frames[i], name=nm, clay=os.path.join(out, f"{nm}_color.png"),
                                     depth=os.path.join(fdir, f".{nm}_cyc.npy")))
            shutil.copy(os.path.join(fdir, f"d_{i:04d}.png"), os.path.join(out, f"{nm}_depth.png"))
            shutil.copy(os.path.join(fdir, f"p_{i:04d}.png"), os.path.join(out, f"{nm}_pose.png"))
            ci, _, fm = metas[i]                       # 2D expectations for the keyframe check

            def pr(p):
                return None if p is None else rig.project(ci, [np.asarray(p, float)], W, H)[0]
            meta = {"size": [W, H], "figures": [], "offscreen": []}
            for fg in fm:
                q = [pr(p) for p in fg["joints"]]
                q = np.array([x for x in q if x is not None], float).reshape(-1, 2)
                # a figure entirely off-screen / behind the camera this frame: no 2D expectation (it is not
                # drawn), listed so the checks know it was expected to be absent
                if len(q) == 0 or q[:, 0].max() < 0 or q[:, 0].min() > W or q[:, 1].max() < 0 or q[:, 1].min() > H:
                    meta["offscreen"].append(fg["who"])
                    continue
                fg = dict(fg, grips=[g for g in fg["grips"] if pr(g["fist"]) is not None and pr(g["work"]) is not None
                                     and None not in (pr(g["handle"][0]), pr(g["handle"][1]))])
                meta["figures"].append({"who": fg["who"], "bbox": [float(max(q[:, 0].min(), 0)), float(max(q[:, 1].min(), 0)),
                                                                   float(min(q[:, 0].max(), W)), float(min(q[:, 1].max(), H))],
                                        "grips": [{"hand": g["hand"], "item": g["item"], "fist": pr(g["fist"]),
                                                   "handle": [pr(g["handle"][0]), pr(g["handle"][1])],
                                                   "work": pr(g["work"]), "work_name": g["work_name"],
                                                   "fist_px": float(np.linalg.norm(np.subtract(
                                                       pr(np.asarray(g["fist"]) + rig._n(np.subtract(ci["pos"], g["fist"])) * 0
                                                          + np.array([0, 0, 0.045])), pr(g["fist"]))) * 2)}
                                                  for g in fg["grips"]]})
            with open(os.path.join(out, f"{nm}_expect.json"), "w") as fh:
                json.dump(meta, fh, indent=1, default=lambda o: np.asarray(o).tolist())
        sp = os.path.join(out, "stills.json")
        with open(sp, "w") as fh:
            json.dump(st, fh, default=lambda o: np.asarray(o).tolist())
        r2 = subprocess.run([B.BLENDER, "--background", "--factory-startup", "-t", os.environ.get("BLOCKOUT_THREADS", "16"),
                             "--python", os.path.join(HERE, "blockout_scene.py"), "--", sp],
                            capture_output=True, text=True, timeout=1200)
        if r2.returncode:
            sys.stderr.write(r2.stdout[-2000:] + r2.stderr[-2000:])
            raise RuntimeError("stills failed")
    if not a.keep_frames:
        shutil.rmtree(fdir)
    res = {"out": out, "frames": N, "fps": a.fps, "size": [W, H], "move": move, "violations": nbad,
           "rendered": "keys" if a.key_frames else "clip",
           "seconds": {"stage": round(t1 - t0, 1), "blender": round(t2 - t1, 1),
                       "blender_per_frame": round((t2 - t1) / len(sel), 3), "passes": round(t3 - t2, 1),
                       "total": round(time.time() - t0, 1)},
           "files": {k: os.path.join(out, f"{k}.mp4") for k in made}}
    with open(os.path.join(out, "video.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res))
    if nbad and not os.environ.get("BLOCKOUT_ALLOW_VIOLATIONS"):
        sys.stderr.write(f"FAILED: {nbad} grip/contact violation(s) over the clip; see checks.json\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
