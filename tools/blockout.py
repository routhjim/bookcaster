"""3D blockout ("previs") for a shot: stage the figures in 3D, then render control passes for the drawer.

    ~/tts-ab/.venv-pose/bin/python -I tools/blockout.py shot.json OUT_DIR [--size 1536x864] [--samples 16]
                                                                         [--blend] [--camera cam.json]

Writes, for the start frame and (if the shot has "end") the end frame, all from ONE camera:
    OUT_DIR/start_pose.png    OpenPose skeleton projected from the 3D joints (tools/poses.py's look)
    OUT_DIR/start_depth.png   depth, near = bright (relative inverse depth, like MiDaS/Depth-Anything)
    OUT_DIR/start_canny.png   edges of the clay render
    OUT_DIR/start_clay.png    the grey clay render itself
    OUT_DIR/start_joints.json COCO-18 pixel points per person (+ 3D joints) for other tools
    OUT_DIR/camera.json       the camera (reuse it with --camera or "camera": {"from": ...})
    OUT_DIR/sheet.png         contact sheet of every pass
    OUT_DIR/control.json      ready-made qwen_image.py "control" fields: {"start": {"pose": {...}, ...}}
and prints one JSON line {"out", "frames", "seconds", "render_s"}.

THE SHOT JSON
-------------
{"set": {"kind": "interior"|"exterior", "ground": true, "room": [w, d, h],
         "props": [{"name": "...", "prop": "<library key>", "box": [x, y, z, w, d, h], "yaw": 0}]},
 "figures": [{"who": "Edric", "kind": "human", "pose": "hammer_raised", "at": [x, y], "facing": 0,
              "height": 1.8, "holds": {"right": "hammer"}, "wear": ["helm"]},
             {"who": "Vorgrath", "kind": "creature", "pose": "striking", "at": [4, 0], "facing": "Edric",
              "height": 6, "z": 8},
             {"who": "Maren", "pose": "carried_limp", "attach": {"to": "Vorgrath", "point": "claws",
              "by": "pelvis"}}],
 "camera": {"framing": "wide"|"medium"|"medium close-up"|"close-up"|"low angle"|"high angle",
            "target": "Edric" | ["Edric", "Vorgrath"] | [x, y, z], "line": ["Edric", "Vorgrath"],
            "side": "right"|"left", "angle": 0, "distance": null, "height": null, "lens": null,
            "from": "/abs/camera.json"},
 "end": {"figures": [{"who": "Edric", "pose": "hammer_strike"}]}}

- Units are metres; z is up; "facing" is a compass angle (0 = +x, 90 = +y) or the name of another figure
  to face. "at" is the ground position (a human's feet, a creature's pelvis).
- "pose": a human pose from tools/poses.py (all 52, lifted to 3D in blockout_rig.py), or a creature pose
  (standing, rearing, striking, lunging, flying, flying_carry). "z": height for airborne poses.
- "holds": {"right"|"left": <a prop library key: tools/stage_props/props.json>}; two-handed poses
  (grip "both" in tools/stage_props/pose3d.json) put the left hand on the grip;
  helm_remove / helm_put_on hold a helm between the hands. "wear": ["helm"] puts a helm on the head.
- "attach": hang a person from a creature's anchor ("claws", "claw_r", "claw_l", "jaw", "chest") by the
  pelvis (carried) or the shoulders ("hands": hanging_from_claws).
- "end": figures that change, by "who" (missing fields are kept from the start); the camera is the same.
- Screen direction: with two or more figures the camera stays on one side of the "line" (default: the
  first two figures, in order), so the first is screen-left and the second screen-right, in every shot
  that uses the same line. "side": "left" flips to the other side for the whole scene (crossing the line
  on purpose). "angle" swings the camera around the target (degrees, clamped to +-70, never crossing).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockout_rig as rig  # noqa: E402
import blockout_contact as contact  # noqa: E402
import blockout_props as props  # noqa: E402
from poses import draw_people  # noqa: E402

BLENDER = os.environ.get("BLOCKOUT_BLENDER", os.path.expanduser("~/opt/blender/current/blender"))


# --------------------------------------------------------------------------------------------------
# Staging
# --------------------------------------------------------------------------------------------------

def _is_creature(f):
    """kind "creature" decides; a bare pose name only when it is NOT also a human pose (e.g. "standing")."""
    if f.get("kind"):
        return f["kind"] == "creature"
    return f.get("pose") in rig.DRAGON_POSES and f.get("pose") not in rig.SPECS


def merge_end(shot):
    """The end frame's figures: the start figures with the end's changes applied by "who"."""
    end = shot.get("end")
    if not end:
        return None
    figs = copy.deepcopy(shot["figures"])
    by = {f.get("who"): f for f in figs}
    for e in end.get("figures", []):
        if e.get("who") in by:
            by[e["who"]].update(copy.deepcopy(e))
        else:
            figs.append(copy.deepcopy(e))
    return figs


STAGE_VIEW = {}       # {"pos": camera position} once the camera is known (readability-aware grips)


def stage(figures, set_spec, contacts=None):
    """Resolve facings and attachments; build every primitive and the people's joints. contacts: this
    frame's held-prop contacts (blockout_contact); every held prop's grip and contact is checked."""
    pos = {f.get("who", f"fig{i}"): np.array(list(f.get("at", [0, 0]))[:2] + [0.0], float)
           for i, f in enumerate(figures)}
    for i, f in enumerate(figures):
        fc = f.get("facing", 0)
        if isinstance(fc, str):
            if fc in pos:
                d = pos[fc] - pos[f.get("who", f"fig{i}")]
                fc = math.degrees(math.atan2(d[1], d[0]))
            else:
                fc = 0.0
        f["_facing"] = float(fc)
    prims, people, anchors, info, models, checks, cprims = [], [], {}, {}, [], [], {}
    # a creature's "reach": {"limb": "fore_r", "to": who, "point": "head"|"chest"|..., "above": m}
    # needs the person's place first: stage the people once without attachments to find it
    pre = {}
    for f in figures:
        if not _is_creature(f) and not f.get("attach"):
            Wj, _ = rig.place_human(dict(f))
            pre[f.get("who")] = Wj
    for f in figures:                                  # creatures first: people attach to them
        if _is_creature(f):
            rc = f.get("reach")
            if rc and rc.get("to") in pre:
                J = pre[rc["to"]]
                pt = {"chest": 0.6 * J["neck"] + 0.4 * J["pelvis"]}.get(rc.get("point", "head"),
                                                                      J.get(rc.get("point", "head"), J["head"]))
                f["reach"] = dict(rc, _point=(np.asarray(pt) + rig.UP * float(rc.get("above", 0.0))).tolist())
            P, A = rig.dragon(f)
            if not f.get("model"):
                cprims[f.get("who")] = P
            mdl = f.get("model")
            rj = None
            if mdl:
                mb = mdl if isinstance(mdl, str) else mdl.get("blend")
                rjp = os.path.join(os.path.dirname(mb), "rig.json")
                if os.path.exists(rjp):
                    rj = json.load(open(rjp))
                    if not rj.get("template"):
                        rj = None
            if rj:    # a RIGGED creature (template skeleton, tools/blockout_creature_bl.py): the mesh in its
                #       rest pose placed by its hind feet at "at", scaled to "height"; anchors from its bones
                mdl = {"blend": mdl} if isinstance(mdl, str) else dict(mdl)
                RJ = {k_: np.asarray(v_, float) for k_, v_ in rj["joints"].items()}
                k = float(f.get("height", rj["height"])) / float(rj["height"])
                yaw = math.radians(f["_facing"])
                Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
                feet = [RJ[n_] for n_ in RJ if n_.endswith("_toe") or n_.endswith("_ankle")]
                fc = np.mean([q for q in feet if q[2] < 0.1 * rj["height"]] or feet, 0)
                x_, y_ = f.get("at", [0, 0])[:2]
                M = np.eye(4)
                M[:3, :3] = k * Rz
                M[:3, 3] = np.array([x_, y_, float(f.get("z", 0.0))]) - k * Rz @ np.array([fc[0], fc[1], 0.0])
                W_ = {n_: M[:3, :3] @ q + M[:3, 3] for n_, q in RJ.items()}
                # FOLDED WINGS at rest: a winged creature's rig rest pose has its wings spread (an 11 m span fills
                # the shot and swallows the camera); unless the moment flies / flaps / spreads, they fold against
                # the back (the QA "wings_folded" directions, in the creature's own frame)
                if not f.get("pose_joints") and not re.search(r"fly|flap|spread|soar|glide|wing", str(f.get("pose", "")) +
                                                               " " + str(f.get("motion", ""))):
                    for sd_, sg_ in (("l", 1.0), ("r", -1.0)):
                        sh_ = f"wing_{sd_}_shoulder"
                        if sh_ not in W_:
                            continue
                        dirs = {f"wing_{sd_}_elbow": (sh_, (-0.75, 0.35, 0.3)), f"wing_{sd_}_wrist": (f"wing_{sd_}_elbow", (-0.3, 0.12, -0.95)),
                                f"wing_{sd_}_tip": (f"wing_{sd_}_wrist", (-0.2, 0.15, -0.97)),
                                f"wing_{sd_}_finger1": (f"wing_{sd_}_wrist", (-0.28, 0.12, -0.95)),
                                f"wing_{sd_}_finger2": (f"wing_{sd_}_wrist", (-0.34, 0.1, -0.93)),
                                f"wing_{sd_}_finger3": (f"wing_{sd_}_wrist", (-0.4, 0.08, -0.91))}
                        for jn, (pn, dv) in dirs.items():
                            if jn in W_ and pn in W_ and jn in RJ and pn in RJ:
                                L_ = float(np.linalg.norm(RJ[jn] - RJ[pn])) * k
                                d_ = Rz @ (np.array([dv[0], dv[1] * sg_, dv[2]]) / np.linalg.norm(dv))
                                W_[jn] = W_[pn] + d_ * L_
                if f.get("pose_joints"):      # a motion source's pose for this keyframe (world metres, template
                    W_.update({n_: np.asarray(v_, float) for n_, v_ in f["pose_joints"].items()})  # names)
                blen = float(np.linalg.norm(W_["chest"] - W_["pelvis"])) if "chest" in W_ else k
                A = {}
                for an, ad_ in rj.get("anchors", {}).items():
                    A[an] = W_[ad_["joint"]] + Rz @ (np.asarray(ad_["offset"], float) * blen)
                A.update({n_: q for n_, q in W_.items() if n_ not in A})
                A.setdefault("claws", 0.5 * (A.get("claw_l", W_["pelvis"]) + A.get("claw_r", W_["pelvis"])))
                P = []
                TP = json.load(open(os.path.join(HERE, "stage_templates", "creatures.json")))["templates"].get(
                    rj["template"]) or {"parents": rj.get("parents", {})}   # a composite / imported rig: its own
                cprims[f.get("who")] = contact.creature_capsules(W_, TP["parents"], rj.get("radii", {}), k)
                models.append({"blend": mdl["blend"], "static": True, "M": M.tolist(), "who": f.get("who"),
                               "layer": "creature", "joints": {n_: q.tolist() for n_, q in W_.items()},
                               # posed by its joints (folded wings / a motion pose): the bones aim at them
                               "template": rj["template"], "targets": {n_: q.tolist() for n_, q in W_.items()}})
                mdl = None
                cpts = list(W_.values())
            elif mdl:   # a static 3D mesh (blockout_model_bl.py prep: Z-up, facing +X, feet at 0, centred) replaces
                #       the proxy in the render; the proxy still gives the anchors and the camera framing
                mdl = {"blend": mdl} if isinstance(mdl, str) else dict(mdl)
                pts = np.array([p for q in P for p in _prim_pts(q)])
                # same height on screen as the proxy in this pose (its z extent), unless "scale" is given
                k = float(mdl.get("scale", float(np.ptp(pts[:, 2])) / float(mdl.get("height", 6.0))))
                yaw = math.radians(f["_facing"])
                M = np.eye(4)
                M[:3, :3] = k * np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0],
                                          [0, 0, 1]])
                M[:3, 3] = [A["pelvis"][0], A["pelvis"][1], float(pts[:, 2].min()) if "z" not in f
                            else float(A["pelvis"][2]) - k * float(mdl.get("pelvis_z", 0.4 * float(mdl.get("height", 6.0))))]
                back = float(mdl.get("back", 0.0))       # step the static mesh back along its facing (m)
                M[0, 3] -= back * math.cos(yaw)
                M[1, 3] -= back * math.sin(yaw)
                models.append({"blend": mdl["blend"], "static": True, "M": M.tolist(), "feet_anchor": True,
                               "who": f.get("who")})
            elif not rj:
                prims += P
            anchors[f.get("who")] = A
            if rj:                             # framing points: the rigged mesh's joints (+ a margin)
                P_pts = cpts
            else:
                P_pts = [p for q in P for p in _prim_pts(q)]
            info[f.get("who")] = {"kind": "creature", "points": P_pts,
                                  "prims": P if not f.get("model") else [], "fig": f,
                                  "height": float(f.get("height", 6.0)),
                                  "anchors": {k: np.asarray(v, float) for k, v in A.items()}}
    for f in figures:
        if _is_creature(f):
            continue
        g = dict(f)
        att = f.get("attach")
        if att and att.get("to") in anchors:
            g["_attach"] = anchors[att["to"]][att.get("point", "claws")]
            g["_attach_by"] = {"hands": "hands", "shoulders": "hands"}.get(att.get("by", "pelvis"), "pelvis")
        Wj, ex = rig.place_human(g)
        am = f.get("aim")
        if am and am.get("to") in anchors:
            tp = anchors[am["to"]].get(am.get("point", "chest"))
            if tp is not None:
                rig.aim(Wj, ex, am.get("hand", "right"), tp)
        elif am and am.get("to") in info:
            rig.aim(Wj, ex, am.get("hand", "right"), info[am["to"]]["joints"].get(am.get("point", "head")))
        mdl = f.get("model")
        if mdl:                                        # a rigged 3D model replaces the mannequin's body:
            mdl = {"blend": mdl} if isinstance(mdl, str) else dict(mdl)
            rj = os.path.join(os.path.dirname(mdl["blend"]), "rig.json")
            with open(rj) as fh:                       # pose on ITS bone lengths (the hands move)
                Wj = contact.fk_model(Wj, json.load(fh)["joints"], float(f.get("height", 1.8)))
        two = rig.POSE3D.get(f.get("pose"), {}).get("grip") == "both"
        mine = [c for c in (contacts or []) if c.get("who") == f.get("who")]
        if STAGE_VIEW.get("pos") is not None:
            ex["_view"] = np.asarray(STAGE_VIEW["pos"], float)
        contact.grip_all(Wj, ex, f, two_handed=two)    # every handle in a power-grip fist (cleanup layer)
        clog = contact.solve_contacts(Wj, ex, f, mine, set_spec, anchors, two_handed=two,
                                      creature_prims=cprims)
        # people never enter creature volumes: push out (horizontally), then re-solve the contacts in place
        for cw, caps in cprims.items():
            if caps and caps[0].get("tag") == "body" and any(m_.get("who") == cw for m_ in models):
                mv = contact.push_out(Wj, caps)
                if mv > 0:
                    clog.append(f"{f.get('who')} pushed {mv:.2f} m out of {cw}'s body")
                    ex2 = dict(f, anchor_fixed=True)
                    contact.grip_all(Wj, ex, ex2, two_handed=two)
                    contact.solve_contacts(Wj, ex, ex2, mine, set_spec, anchors, two_handed=two, creature_prims=cprims)
        rows, bad = contact.check(Wj, ex, f, mine, set_spec, anchors, two_handed=two, creature_prims=cprims)
        for cw, caps in cprims.items():
            if caps and any(m_.get("who") == cw for m_ in models):
                dep, jn = contact.human_in_creature(Wj, caps)
                r_ = {"who": f.get("who"), "check": f"inside {cw}'s body ({jn})", "value_cm": round(dep * 100, 2),
                      "limit": 1.0}
                rows.append(r_)
                if dep > 0.01:
                    bad.append(r_)
        r2, b2 = contact.grip_checks(Wj, ex, f, engaged_items=mine)
        rows, bad = rows + r2, bad + b2
        fists = sorted({gi["side"] for gi in (ex.get("grip") or {}).values()})
        checks.append({"who": f.get("who"), "log": clog, "rows": rows, "violations": bad})
        hp = rig.human_prims(Wj, ex["scale"], wear=() if f.get("model") else f.get("wear", ()),
                             hold_items=f.get("holds"), hold_dirs=ex["hold"], fwd=ex["fwd"], left=ex["left"],
                             hold_flats=ex.get("flat"), hold_at=ex.get("grip_at"), fists=fists)
        hp += contact.fist_prims(ex, contact.pk(ex))
        if mdl:
            mdl["targets"] = {k: np.asarray(v).tolist() for k, v in Wj.items() if not k.startswith("_")}
            mdl["ground"] = False                      # fk_model grounded it by the feet joints
            mdl["who"] = f.get("who")
            mdl["fist"] = fists                          # Blender folds these model hands away (fist proxies)
            mdl["height"] = float(f.get("height", 1.8))
            models.append(mdl)
            hp = [p for p in hp if p.get("tag") != "body"]
        prims += hp
        people.append({"who": f.get("who"), "joints": Wj})
        info[f.get("who")] = {"kind": "human", "joints": Wj, "height": float(f.get("height", 1.8)),
                              "ex": ex, "fig": f, "two_handed": two, "model": mdl or None,
                              "points": [v for n, v in Wj.items() if not n.startswith("_")] +
                                        [q for pr in hp if pr.get("tag") in ("prop", "hot") for q in _prim_pts(pr)]}
    info["_set_prims"] = set_prims(set_spec or {})
    info["_anchors"] = anchors
    info["_cprims"] = cprims
    prims += info["_set_prims"]
    info["_models"] = models
    info["_checks"] = checks
    return prims, people, info


def _prim_pts(p):
    if p["t"] == "propmesh":
        return []                                      # its proxy prims carry the geometry
    if p["t"] == "capsule":
        return [np.asarray(p["a"]), np.asarray(p["b"])]
    if p["t"] in ("ellipsoid", "box"):
        c, M = np.asarray(p["c"]), np.asarray(p["M"])
        return [c + s * M[:, i] for i in range(3) for s in (-1, 1)]
    if p["t"] == "mesh":
        return [np.asarray(v) for v in p["verts"]]
    return []


def set_prims(s):
    P = []
    if s.get("ground", True):
        size = 60.0 if s.get("kind") == "interior" else 400.0
        P.append({"t": "box", "c": [0, 0, -0.05], "M": np.diag([size, size, 0.05]), "tag": "ground"})
    for pr in s.get("props", []):                  # every set prop from the prop library (by "prop" key or alias)
        if not pr.get("hidden"):                       # hidden: a contact-only proxy of world geometry
            P += props.set_prims(pr)
    return P


def fx_prims(fx_list, frame, contacts, set_spec, anchors, t=1.0):
    """Effects as emissive primitives (tag "spark"): {"type": "sparks", "at": [x,y,z] | "Who:anchor" |
    {"contact": i}, "frame": "end"|"start"|"both", "count": 60, "size": 0.25, "seed": 1}. Sparks: thin
    bright streaks radiating outward and a little up from the point (what the drawer should paint at an
    impact). t (0..1) grows them in for a video."""
    P = []
    for fx in fx_list or []:
        if fx.get("frame", "end") not in (frame, "both"):
            continue
        at = fx.get("at")
        if isinstance(at, dict):
            c = contacts[int(at.get("contact", 0))]
            at = contact.surface_point(set_spec, c)[0] if ":" not in c["to"] else \
                anchors[c["to"].split(":")[0]][c["to"].split(":")[1]]
        elif isinstance(at, str):
            who, an = at.split(":", 1)
            at = anchors[who][an]
        at = np.asarray(at, float)
        if fx.get("type", "sparks") == "sparks":
            rng = np.random.default_rng(int(fx.get("seed", 1)))
            size = float(fx.get("size", 0.25))
            for _ in range(int(fx.get("count", 60))):
                th = rng.uniform(0, 2 * math.pi)
                u = rng.uniform(0.05, 0.9)
                d = rig._n(np.array([math.cos(th), math.sin(th), u]))
                r0 = rng.uniform(0.02, 0.5) * size * t
                ln = rng.uniform(0.15, 0.9) * size * min(1.0, 0.3 + t)
                a = at + d * r0
                P.append({"t": "capsule", "a": a, "b": a + d * ln, "r": rng.uniform(0.0015, 0.0035), "tag": "spark"})
    return P


def room_walls(s, cam_pos):
    """Interior: walls on the sides of the room the camera looks at (not the one behind it)."""
    if s.get("kind") != "interior":
        return []
    w, d, h = s.get("room", [10.0, 8.0, 4.5])
    cx, cy = s.get("room_at", [0.0, 0.0])
    P = []
    for nrm, c, M in (((1, 0), (cx - w / 2, cy, h / 2), np.diag([0.1, d / 2, h / 2])),
                      ((-1, 0), (cx + w / 2, cy, h / 2), np.diag([0.1, d / 2, h / 2])),
                      ((0, 1), (cx, cy - d / 2, h / 2), np.diag([w / 2, 0.1, h / 2])),
                      ((0, -1), (cx, cy + d / 2, h / 2), np.diag([w / 2, 0.1, h / 2]))):
        # inward normal nrm; keep the wall if the camera is inside on its side (it faces the camera)
        # a wall the camera stands behind (outside the room) would block the view: skip it
        if np.dot(np.array(cam_pos[:2]) - np.array(c[:2]), nrm) > 0.3:
            P.append({"t": "box", "c": list(c), "M": M, "tag": "set"})
    return P


# --------------------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------------------

FRAME_PARTS = {
    "close-up": ("head_top", "head", "nose", "l_ear", "r_ear", "neck"),
    "medium close-up": ("head_top", "head", "neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow"),
    "medium": ("head_top", "head", "neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_hand", "r_hand",
               "pelvis", "l_hip", "r_hip"),
    "two-shot": ("head_top", "head", "neck", "l_shoulder", "r_shoulder", "pelvis", "l_hip", "r_hip"),
    "insert": ("l_elbow", "r_elbow", "l_hand", "r_hand"),     # + the held props (below)
}
FRAME_HEAD_ROOM = {"close-up": 0.06, "medium close-up": 0.1, "medium": 0.14, "two-shot": 0.16, "insert": 0.08}
FRAME_FOOT_ROOM = {"close-up": 0.12, "medium close-up": 0.04, "medium": 0.08, "two-shot": 0.1, "insert": 0.08}


def _fac(f, info):
    """A figure's resolved facing (deg): its own "_facing", else the staged copy's in info, else a number."""
    if "_facing" in f:
        return float(f["_facing"])
    g = (info.get(f.get("who"), {}) or {}).get("fig") or {}
    if "_facing" in g:
        return float(g["_facing"])
    fc = f.get("facing", 0.0)
    return float(fc) if not isinstance(fc, str) else 0.0


def ots_camera(cs, figures, info, size):
    """Over-the-shoulder: cs["ots"] = [A, B]: the camera behind A's shoulder (on the side of the A->B line
    given by cs["side"], default right = the master's side), looking at B's face; A's shoulder and head edge
    frame the shot. Lens 50 unless given."""
    W, H = size
    a, b = cs["ots"][:2]
    Ja, Jb = info[a]["joints"], info[b]["joints"]
    v = Jb["head"] - Ja["head"]
    v[2] = 0
    v = rig._n(v) if np.linalg.norm(v) > 1e-3 else np.array([1.0, 0, 0])
    # the camera side comes from the SCENE AXIS (line [L, R]: L screen-left in the master), never from the
    # order of this shot's pair, so a reverse over-the-shoulder stays on the same side of the 180-degree line
    ln = cs.get("line") or [a, b]
    if not all(w in info for w in ln[:2]):
        ln = [a, b]
    def _c(w):
        v = info[w]
        return np.asarray(v["joints"]["head"], float) if "joints" in v else np.mean(np.array(v["points"]), 0)
    ax = _c(ln[1]) - _c(ln[0])
    ax[2] = 0
    ax = rig._n(ax) if np.linalg.norm(ax) > 1e-3 else v
    nrm = np.array([ax[1], -ax[0], 0.0]) * (-1 if cs.get("side") == "left" else 1)
    sh = Ja["r_shoulder"] if np.dot(Ja["r_shoulder"] - Ja["neck"], nrm) > 0 else Ja["l_shoulder"]
    lens = float(cs.get("lens") or 50)
    pos = sh - v * float(cs.get("back", 0.9)) + nrm * float(cs.get("out", 0.35)) + rig.UP * 0.18
    look = Jb["head"] * 0.7 + Jb["neck"] * 0.3
    # the far face must read for lip-sync: >= ~18 % of the frame height (a longer lens when they stand apart)
    face_h = float(np.linalg.norm(Jb["head_top"] - Jb["neck"])) * 0.85
    sens_h = rig.SENSOR_W * H / W
    need = 0.18 * sens_h * float(np.linalg.norm(look - pos)) / max(face_h, 1e-3)
    lens = min(max(lens, need), 100.0)
    return {"pos": pos.tolist(), "R": rig.look_at(pos, look).tolist(), "lens": lens, "look": look.tolist(),
            "size": [W, H], "framing": "over-the-shoulder", "line": list(ln)}


def after_solve(cam: dict, cam_spec: dict, cinfo: dict, shot: dict) -> dict:
    """Every step after solve_camera, for BOTH renderers (blockout.run and the clip renderer blockout_video: 10-22,
    three of these lived in run() only - walled-in forge cameras, a talk close-up from inside Edric's head): the face
    orbit, everyone seen in a wide, the set clearance, the pano set's radius, the creature volume."""
    # OCCLUSION: a talk / single-person shot's subject must show >= 80 % of the face; else the camera orbits the
    # look point (+-12, +-24, +-36 deg) and rises a little until it does (the first that passes, else the best)
    subj = cam_spec.get("speaker") or (cam_spec.get("ots") or [None, None])[-1] or \
        (cam_spec.get("target") if isinstance(cam_spec.get("target"), str) else None)
    fv = face_visibility(cam["pos"], subj, cinfo) if subj else None
    if fv is not None and fv < 0.8 and not cam_spec.get("pos") and not cam_spec.get("path"):
        look = np.asarray(cam["look"], float)
        best_ = (fv, cam)
        for ang in (12, -12, 24, -24, 36, -36):
            for dz in (0.0, 0.2):
                p_ = np.asarray(cam["pos"], float) - look
                ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                p_ = np.array([p_[0] * ca - p_[1] * sa, p_[0] * sa + p_[1] * ca, p_[2] + dz]) + look
                f2 = face_visibility(p_, subj, cinfo)
                if f2 > best_[0]:
                    c2 = dict(cam, pos=p_.tolist(), R=rig.look_at(p_, look).tolist())
                    best_ = (f2, c2)
            if best_[0] >= 0.8:
                break
        sys.stderr.write(f"camera: {subj} face {fv:.0%} visible -> {best_[0]:.0%} after the occlusion orbit\n")
        fv, cam = best_
    if fv is not None:
        cam["face_visible"] = {"who": subj, "frac": round(float(fv), 2), "ok": fv >= 0.8}
    cam = see_everyone(cam, cam_spec, cinfo)
    cam = clear_of_set(cam, shot, cam_spec, cinfo)
    # A PANO SET holds up only near its capture point: a camera farther than SET_CAM_R moves along its view line
    # toward the look point until inside, and widens its lens to keep the same coverage at the look point
    capxy = shot.get("_capture")
    _tg = cam_spec.get("target")
    _tgl = [_tg] if isinstance(_tg, str) else (_tg if isinstance(_tg, list) else [])
    _cre_wide = cam_spec.get("framing") in ("wide", "low angle", "high angle") and (
        not _tgl or any(isinstance(t_, str) and cinfo.get(t_.split(":")[0], {}).get("kind") == "creature" for t_ in _tgl)) \
        and any(x_.get("kind") == "creature" for n_, x_ in cinfo.items() if not n_.startswith("_"))
    # (a WIDE on a huge creature needs its distance more than the set's view region: streaks beat a frame of wing)
    if capxy is not None and (shot.get("set") or {}).get("world") and not cam_spec.get("path") and not _cre_wide:
        Rmax = float((shot["set"]["world"] or {}).get("view_r") or os.environ.get("SET_CAM_R", 6.0))   # the set's view_region
        p_ = np.asarray(cam["pos"], float)
        lk = np.asarray(cam["look"], float)
        c2 = np.array([float(capxy[0]), float(capxy[1])])
        if np.linalg.norm(p_[:2] - c2) > Rmax:
            d0 = float(np.linalg.norm(lk - p_))
            u_ = (lk - p_) / max(d0, 1e-9)
            t_ = 0.0
            while t_ < d0 * 0.9 and np.linalg.norm((p_ + u_ * t_)[:2] - c2) > Rmax:
                t_ += 0.1
            p2 = p_ + u_ * t_
            d1 = max(float(np.linalg.norm(lk - p2)), 0.5)
            lens2 = max(12.0, float(cam["lens"]) * d1 / d0)
            sys.stderr.write(f"camera: {np.linalg.norm(p_[:2] - c2):.1f} m from the set's capture point -> moved "
                             f"{t_:.1f} m in, lens {cam['lens']:.0f} -> {lens2:.0f}\n")
            cam = dict(cam, pos=p2.tolist(), lens=lens2, R=rig.look_at(p2, lk).tolist(), clamped_to_capture=Rmax)
    # CREATURE VOLUME: the lens must not sit inside a creature, nor shoot a person through one (a wing between the
    # camera and its subject); else orbit the look point (+-15..60 deg, a little higher) to the clearest spot
    cpts_ = [np.asarray(q, float) for n_, x_ in cinfo.items() if not n_.startswith("_") and x_.get("kind") == "creature"
             for q in list(x_.get("points", []))[::2]]
    tgt_is_creature = any(isinstance(t_, str) and t_.split(":")[0] in cinfo and cinfo[t_.split(":")[0]].get("kind") == "creature"
                          for t_ in ([cam_spec.get("target")] if isinstance(cam_spec.get("target"), str) else (cam_spec.get("target") or [])))

    def _vol_bad(p_):
        if not cpts_:
            return 0.0
        P_ = np.array(cpts_)
        dmin = float(np.min(np.linalg.norm(P_ - p_, axis=1)))
        bad = max(0.0, 1.2 - dmin)                              # inside / brushing the creature
        if tgt_is_creature:                                     # a creature subject: no PERSON (a carried figure) between
            Hp = np.array([np.asarray(q, float) for n_, x_ in cinfo.items() if not n_.startswith("_")
                           and x_.get("kind") == "human" for q in x_.get("joints", {}).values()] or np.zeros((0, 3)))
            if len(Hp):
                lk_ = np.asarray(cam["look"], float)
                ab = lk_ - p_
                t_ = np.clip(((Hp - p_) @ ab) / max(ab @ ab, 1e-9), 0, 0.95)
                dd = np.linalg.norm(Hp - (p_ + t_[:, None] * ab), axis=1)
                bad += float(np.sum(dd < 0.3)) * 0.5
        if not tgt_is_creature:
            lk_ = np.asarray(cam["look"], float)
            ab = lk_ - p_
            t_ = np.clip(((P_ - p_) @ ab) / max(ab @ ab, 1e-9), 0, 0.92)
            dd = np.linalg.norm(P_ - (p_ + t_[:, None] * ab), axis=1)
            bad += float(np.sum(dd < 0.35)) / max(len(P_), 1) * 50.0     # the creature between the lens and its subject
        return bad
    if cpts_ and not cam_spec.get("pos") and not cam_spec.get("path"):
        b0 = _vol_bad(np.asarray(cam["pos"], float))
        if b0 > 0.05:
            lk_ = np.asarray(cam["look"], float)
            best_ = (b0, cam)
            for ang in (15, -15, 30, -30, 45, -45, 60, -60):
                for dz in (0.0, 0.8):
                    p_ = np.asarray(cam["pos"], float) - lk_
                    ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                    p_ = np.array([p_[0] * ca - p_[1] * sa, p_[0] * sa + p_[1] * ca, p_[2] + dz]) + lk_
                    b_ = _vol_bad(p_)
                    if b_ < best_[0]:
                        best_ = (b_, dict(cam, pos=p_.tolist(), R=rig.look_at(p_, lk_).tolist()))
                if best_[0] <= 0.05:
                    break
            sys.stderr.write(f"camera: creature-volume clash {b0:.2f} -> {best_[0]:.2f} after the orbit\n")
            cam = best_[1]
            cam["creature_clear"] = round(float(best_[0]), 3)
    return cam

def see_everyone(cam: dict, cam_spec: dict, info: dict) -> dict:
    """A WIDE shot sees every person in it (10-22: Maren on her ledge behind the coiled dragon, Edric ducking under its
    wings - hidden by the creature from a ground-level camera): when someone's head is mostly blocked, the camera
    CRANES up (2 / 4 / 6 m) and orbits the look point (0, +-25, +-50 deg), looking down; the best position wins."""
    if cam_spec.get("path") or cam_spec.get("from") or cam_spec.get("framing", "medium") not in (
            "wide", "establishing", "extreme wide", "two-shot", "low angle", "high angle"):
        return cam
    hum = [n for n, x in info.items() if not n.startswith("_") and x.get("kind") == "human"]
    if not hum:
        return cam

    def score(cp):
        v = [face_visibility(cp, n, info) for n in hum]
        return min(x for x in v if x is not None) if any(x is not None for x in v) else 1.0
    s0 = score(cam["pos"])
    if s0 >= 0.5:
        return cam
    look = np.asarray(cam["look"], float)
    p0 = np.asarray(cam["pos"], float)
    best = (s0, cam)
    for dz in (2.0, 4.0, 6.0):
        for ang in (0, 25, -25, 50, -50):
            v = p0 - look
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            p_ = look + np.array([v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca, v[2] + dz])
            sc_ = score(p_)
            if sc_ > best[0]:
                best = (sc_, dict(cam, pos=p_.tolist(), R=rig.look_at(p_, look).tolist(),
                                  crane={"up_m": dz, "orbit_deg": ang}))
        if best[0] >= 0.5:
            break
    if best[1] is not cam:
        sys.stderr.write(f"camera: a person hidden in the wide shot ({s0:.0%} of a face seen): craned "
                         f"{best[1]['crane']['up_m']:.0f} m, orbit {best[1]['crane']['orbit_deg']:+d} -> {best[0]:.0%}\n")
    return best[1]


def clear_of_set(cam: dict, shot: dict, cam_spec: dict, info: dict | None = None) -> dict:
    """The camera moved clear of the pano set (blockout.run AND the clip renderer, blockout_video: 10-22, the clip path
    skipped it and six forge-strike cameras stayed inside the Old Forge's wall)."""
    # SET CLEARANCE (10-22: a wide shot's camera stood inside the cave rock; the frame showed rock only): the pano
    # set must not stand between the camera and its look point; else the camera comes in along its view line (the lens
    # widened to keep the coverage) and/or orbits the look point, the first clear position wins
    # (explicitly placed cameras too: 10-22, the contact camera of six forge strikes stood inside the Old Forge's wall)
    if not cam_spec.get("path") and not cam_spec.get("from"):
        try:
            import motion_camera as MC
            sv = MC.SetView(shot)
        except Exception as e_:                      # noqa: BLE001 - no set depth: nothing to test
            sv = None
            sys.stderr.write(f"camera: set clearance not tested ({e_})\n")
        if sv is not None and sv.dep is not None:
            look = np.asarray(cam["look"], float)
            p0 = np.asarray(cam["pos"], float)
            # the line to the look point AND to every person's head (10-22: an over-the-shoulder looking up at the
            # dragon cleared its look line while the rock hid Edric rolling on the ash)
            heads = [np.asarray(x["joints"].get("head", x["joints"].get("neck")), float) for n_, x in (info or {}).items()
                     if not n_.startswith("_") and x.get("kind") == "human" and x.get("joints")]

            def buried(p_, margin=0.4):
                """The camera point itself in (or within margin of) the set's rock (10-22: a talk close-up's camera
                sat 2 cm inside the Burnt Mountain's overhang; the line test starts 0.3 m out and never saw it)."""
                q = np.asarray(p_, float) - sv.O
                r = float(np.linalg.norm(q))
                if r < 0.5:
                    return False
                az = math.atan2(q[1], q[0])
                el = math.asin(max(-1.0, min(1.0, q[2] / r)))
                ia = int((az + math.pi) / (2 * math.pi) * sv.na) % sv.na
                ie = min(sv.ne - 1, max(0, int((el + math.pi / 2) / math.pi * sv.ne)))
                return r >= float(sv.dep[ie, ia]) - margin

            def blocked(p_):
                return buried(p_) or sv.blocks(p_, look) or any(sv.blocks(p_, h_) for h_ in heads)
            if blocked(p0):
                d0 = float(np.linalg.norm(p0 - look))
                found = None
                for fr_ in (1.0, 0.8, 0.65, 0.5, 0.38):
                    for ang in (0, 20, -20, 40, -40):
                        if fr_ == 1.0 and ang == 0:
                            continue
                        v_ = (p0 - look) * fr_
                        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                        p_ = look + np.array([v_[0] * ca - v_[1] * sa, v_[0] * sa + v_[1] * ca, v_[2]])
                        if not blocked(p_):
                            found = (p_, fr_, ang)
                            break
                    if found:
                        break
                if found:
                    p_, fr_, ang = found
                    lens0 = float(cam["lens"])
                    cam = dict(cam, pos=p_.tolist(), R=rig.look_at(p_, look).tolist(),
                               lens=max(16.0, lens0 * fr_), set_clearance={"moved_in": round(1 - fr_, 2), "orbit_deg": ang})
                    sys.stderr.write(f"camera: the set stood between camera and subject; moved in {1 - fr_:.0%} "
                                     f"(orbit {ang:+d} deg, lens {lens0:.0f}->{max(16.0, lens0 * fr_):.0f})\n")
                else:
                    cam["set_clearance"] = {"blocked": True}
                    sys.stderr.write("camera: !!! the set blocks every tried camera position (left as solved)\n")
    return cam


def solve_camera(cam_spec, figures, info, size):
    cs = dict(cam_spec or {})
    if cs.get("from"):
        with open(cs["from"]) as f:
            return json.load(f)
    W, H = size
    if cs.get("pos") is not None and cs.get("look") is not None:      # an explicit camera
        pos, look = np.asarray(cs["pos"], float), np.asarray(cs["look"], float)
        return {"pos": pos.tolist(), "R": rig.look_at(pos, look).tolist(), "lens": float(cs.get("lens", 35)),
                "look": look.tolist(), "size": [W, H], "framing": cs.get("framing", "explicit"),
                "line": cs.get("line")}
    if cs.get("ots"):                                       # over-the-shoulder: behind A, looking at B
        return ots_camera(cs, figures, info, size)
    # a PERSON facing a CREATURE in a medium shot: over the person's shoulder, up at the creature's head (10-22: "Edric
    # waist-up + the dragon's head" put the camera so far back that Edric was tiny and the dragon a cut-off leg)
    tg_ = cs.get("target")
    tgl = [tg_] if isinstance(tg_, str) else (tg_ if isinstance(tg_, list) else [])
    hum = [n for n in tgl if isinstance(n, str) and info.get(n, {}).get("kind") == "human"]
    cre = [f.get("who") for f in figures if info.get(f.get("who"), {}).get("kind") == "creature" and not f.get("attach")
           and "head" in (info[f.get("who")].get("anchors") or {})]
    if len(hum) == 1 and cre and cs.get("framing", "medium") in ("medium", "two-shot", "low angle") and \
            all(k in info[hum[0]].get("joints", {}) for k in ("r_shoulder", "l_shoulder", "neck", "head")):
        Ja = info[hum[0]]["joints"]
        ch = np.atleast_2d(info[cre[0]]["anchors"]["head"]).mean(0)
        v = ch - Ja["head"]
        v[2] = 0
        v = rig._n(v) if np.linalg.norm(v) > 1e-3 else np.array([1.0, 0, 0])
        nrm = np.array([v[1], -v[0], 0.0]) * (-1 if cs.get("side") == "left" else 1)
        sh = Ja["r_shoulder"] if np.dot(Ja["r_shoulder"] - Ja["neck"], nrm) > 0 else Ja["l_shoulder"]
        # far behind the shoulder with a LONGER lens: perspective compression keeps the creature looming behind the
        # person (10-22: a 14 mm lens close behind Edric made the dragon look no taller than him)
        W, H = size
        spread = float(np.linalg.norm(ch - np.asarray(Ja["head"], float))) + 1.5
        sens_h = rig.SENSOR_W * H / W
        look = ch * 0.55 + np.asarray(Ja["head"], float) * 0.45
        need = spread * 35.0 / sens_h                      # the distance that fits both heads at 35 mm
        back = float(np.clip(need - float(np.linalg.norm(look - np.asarray(Ja["head"], float))), 2.5, 9.0))
        pos = sh - v * back + nrm * 0.55 - rig.UP * 0.35          # behind the shoulder, a little low: looking UP
        dist = float(np.linalg.norm(look - pos))
        lens = float(np.clip(rig.SENSOR_W * H / W * dist / max(spread, 1e-3), 24.0, 50.0))
        return {"pos": pos.tolist(), "R": rig.look_at(pos, look).tolist(), "lens": lens, "look": look.tolist(),
                "size": [W, H], "framing": "over-the-shoulder (creature)", "line": cs.get("line")}
    fr = rig.FRAMING.get(cs.get("framing", "medium"), rig.FRAMING["medium"])
    lens = float(cs.get("lens") or fr["lens"])
    names = [f.get("who") for f in figures]
    humans_ = [n for n in names if info.get(n, {}).get("kind") == "human"]
    # no target: a medium/medium close-up of SEVERAL people is a two-shot of them (not the first one alone)
    if cs.get("target") is None and len(humans_) >= 2 and cs.get("framing", "medium") in ("medium", "medium close-up"):
        cs["framing"], cs["target"] = "two-shot", humans_
    if cs.get("target") is None and cs.get("framing", "medium") in ("wide", "establishing", "extreme wide") and len(names) > 1:
        # a wide shot without a target frames EVERYONE in it (10-22: "Wide shot. Vorgrath lies coiled ... Princess Maren
        # sits on a rock ledge behind" framed the dragon alone - names[0] - and Maren stood 54 deg off the axis)
        cs["target"] = [n for n in names if not next((f for f in figures if f.get("who") == n), {}).get("attach")]
    tgt = cs.get("target", names[0] if names else None)
    tnames = [tgt] if isinstance(tgt, str) else (tgt if isinstance(tgt, list) and tgt and isinstance(tgt[0], str) else [])
    # a creature PART someone aims or strikes at is in frame at any framing (10-22: the sword's insert at the dragon's
    # chest showed no dragon)
    for f_ in figures:
        am_ = f_.get("aim") or {}
        w_ = am_.get("to")
        if w_ and info.get(w_, {}).get("kind") == "creature" and am_.get("point") in info[w_].get("anchors", {}) \
                and tnames and not any(str(n).startswith(f"{w_}:") for n in tnames):
            tnames = list(tnames) + [f"{w_}:{am_['point']}"]
    # a CREATURE in the shot stays in it (10-22: the dragon fight framed Edric waist-up, the dragon 60-98 deg off
    # the axis in 8 moments): a person-targeted shot wider than a close-up adds the creature's head (its anchor box)
    if tnames and cs.get("framing", "medium") not in ("close-up", "extreme close-up", "insert", "medium close-up"):
        for f_ in figures:
            w_ = f_.get("who")
            if info.get(w_, {}).get("kind") == "creature" and w_ not in tnames and not f_.get("attach") \
                    and not any(str(n).startswith(f"{w_}:") for n in tnames) and "head" in info[w_].get("anchors", {}):
                tnames = list(tnames) + [f"{w_}:head"]
                cs["creature_kept"] = w_
    pts = []
    for n in tnames or names:
        if ":" in n:          # "Vorgrath:head": one anchor of a creature (plus a 0.6 m margin), not its whole body
            w, a = n.split(":", 1)
            c = info.get(w, {}).get("anchors", {}).get(a)
            if c is not None:
                pts += [q + s * 0.6 * e for q in np.atleast_2d(c) for e in np.eye(3) for s in (-1, 1)]
            continue
        pts += list(info.get(n, {}).get("points", []))
    # HUMAN framings frame BODY PARTS, not whole bodies: close-up = head and neck, medium close-up = chest up,
    # medium = waist up (+ hands and held props), two-shot = both people waist up; wide = everything
    fk = cs.get("framing", "medium")
    part_pts = []
    if fk in FRAME_PARTS:
        for n in (tnames or names):
            if ":" in n or info.get(n, {}).get("kind") != "human":
                continue
            Jn = info[n]["joints"]
            part_pts += [Jn[j] for j in FRAME_PARTS[fk] if j in Jn]
            if fk == "medium":
                part_pts += [q for q in info[n]["points"][len([k for k in Jn if not k.startswith("_")]):]
                             if q[2] > Jn["pelvis"][2] - 0.1]                      # held props above the waist
            elif fk == "insert":                                                   # the held props, whole
                part_pts += list(info[n]["points"][len([k for k in Jn if not k.startswith("_")]):])
    if cs.get("points"):                                     # explicit world points that must be in frame
        extra = [np.asarray(q, float) for q in cs["points"]]
        if part_pts:
            part_pts += extra
        else:
            pts = list(pts) + extra
    if part_pts:
        anchor_pts = []                                      # "Creature:head" targets keep their anchor box
        for n in tnames:
            if ":" in n:
                w, a_ = n.split(":", 1)
                c = info.get(w, {}).get("anchors", {}).get(a_)
                if c is not None:
                    anchor_pts += [q + s_ * 0.6 * e for q in np.atleast_2d(c) for e in np.eye(3) for s_ in (-1, 1)]
        pts = np.array(part_pts + anchor_pts)
        hp_ = np.array(part_pts)
        top = float(hp_[:, 2].max()) + FRAME_HEAD_ROOM.get(fk, 0.08)
        bot = float(hp_[:, 2].min()) - FRAME_FOOT_ROOM.get(fk, 0.05)
        pts = np.vstack([pts, [[pts[:, 0].mean(), pts[:, 1].mean(), top]], [[pts[:, 0].mean(), pts[:, 1].mean(), bot]]])
    pts = np.array(pts) if len(pts) else np.zeros((1, 3))
    lo, hi = pts.min(0), pts.max(0)
    hn = [n for n in (tnames or names) if ":" not in n]
    hgt = max(info.get(n, {}).get("height", 1.8) for n in hn) if hn else 1.8
    ext = max(float(hi[2] - lo[2]), 0.5 * hgt)
    if isinstance(tgt, list) and tgt and not isinstance(tgt[0], str):
        look = np.array(tgt, float)
    else:
        look = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] + fr["look"] * (hi[2] - lo[2])])
        if part_pts:                                         # the centre of the framed parts (incl. headroom)
            look = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
            ext = float(hi[2] - lo[2])
    # direction from the look point to the camera (horizontal)
    line = cs.get("line")
    free = [f.get("who") for f in figures if not f.get("attach")]
    if not line and len(free) >= 2:
        line = free[:2]
    centre = {n: np.mean(np.array(info[n]["points"]), 0) for n in names if n in info}
    if line and all(n in centre for n in line[:2]):
        v = centre[line[1]] - centre[line[0]]
        v[2] = 0
        if np.linalg.norm(v) < 1e-3:
            v = np.array([1.0, 0, 0])
        v = rig._n(v)
        nrm = np.array([v[1], -v[0], 0.0])        # camera on the right of A->B: A left, B right
        if cs.get("side") == "left":
            nrm = -nrm
        ang = cs.get("angle")
        if ang is None:
            ang = 0.0
            if len(tnames) == 1 and tnames[0] in info and info[tnames[0]]["kind"] == "human":
                f = next(x for x in figures if x.get("who") == tnames[0])
                fwd = rig.frame(_fac(f, info))[0]
                s = math.copysign(1.0, float(np.cross(nrm, fwd)[2])) if abs(np.cross(nrm, fwd)[2]) > 1e-3 else 0
                ang = 40.0 * s if np.dot(fwd, nrm) > -0.2 else 0.0
                if cs.get("face_on") is not None and np.dot(fwd, nrm) > -0.2:
                    # NEAR FRONT-ON (talk singles, user 10-21): the camera this many degrees off the face's direction,
                    # still on this side of the line (the 40-degree default left the speaker at ~50 deg = a 3/4 head)
                    ang = (90.0 - float(cs["face_on"])) * s
        lim = 85.0 if cs.get("face_on") is not None else 70.0
        ang = max(-lim, min(lim, float(ang)))
        d = rig.rodrigues(nrm, rig.UP, ang)
    else:
        f = next((x for x in figures if x.get("who") in tnames), figures[0] if figures else {})
        fwd = rig.frame(_fac(f, info))[0]
        side = -1 if cs.get("side") == "left" else 1
        a_ = (90.0 - float(cs["face_on"])) if cs.get("face_on") is not None else float(cs.get("angle", 45))
        d = rig.rodrigues(fwd, rig.UP, -side * (90 - a_))
    # distance: cover the framing's height; wide/medium also fit the whole group
    vfov = 2 * math.atan((rig.SENSOR_W * H / W) / 2 / lens)
    hfov = 2 * math.atan(rig.SENSOR_W / 2 / lens)
    cover = fr["cover"] * ext
    right = rig._n(np.cross(-d, rig.UP))
    if part_pts:                                             # exactly the parts (+ room), both directions
        span_h = float(np.ptp(pts @ right)) if len(pts) > 1 else 0.3
        cover = max(ext, span_h * 1.15 * H / W)
    elif cs.get("framing", "medium") in ("wide", "medium", "low angle", "high angle"):
        allp = np.array([p for n in names for p in info.get(n, {}).get("points", [])] or [look])
        grp = pts if tnames else allp
        span_h = float(np.ptp(grp @ right)) if len(grp) > 1 else 0.0
        span_v = float(np.ptp(grp[:, 2])) if len(grp) > 1 else 0.0
        cover = max(cover, span_v * 1.15, span_h * 1.15 * H / W)
    dist = float(cs.get("distance") or (cover / 2) / math.tan(vfov / 2))
    if cs.get("framing") == "low angle":
        dist *= 0.8
    cz = cs.get("height")
    if cz is None and part_pts:
        cz = look[2] + (0.1 if fk in ("close-up", "medium close-up") else 0.0)  # eye level
    if cz is None:
        cz = look[2] if fr["cam"] is None else (lo[2] + fr["cam"] * hgt if cs.get("framing") != "wide" else
                                                 max(look[2], 1.6))
    pos = look + d * dist
    pos[2] = float(cz)
    Rm = rig.look_at(pos, look)
    return {"pos": pos.tolist(), "R": Rm.tolist(), "lens": lens, "look": look.tolist(), "size": [W, H],
            "framing": cs.get("framing", "medium"), "line": line}


# --------------------------------------------------------------------------------------------------
# Passes
# --------------------------------------------------------------------------------------------------

def depth_png(z, out):
    from PIL import Image
    z = np.asarray(z, np.float64)
    valid = np.isfinite(z) & (z > 0) & (z < 1e6)
    img = np.zeros(z.shape, np.float64)
    if valid.any():
        inv = 1.0 / z[valid]
        # relative inverse depth like MiDaS / Depth-Anything: the nearest ~0.5% white, the far end of the
        # scene (the back wall, the distant ground) a dark grey, not black: only empty sky is black (a dark
        # band at the frame's sides read as a picture border to Qwen's ControlNet, 10-19)
        near, far = np.percentile(inv, 99.5), np.percentile(inv, 1.0)
        img[valid] = 0.12 + 0.88 * np.clip((inv - far) / max(near - far, 1e-9), 0, 1)
    Image.fromarray((img * 255).round().astype(np.uint8)).convert("RGB").save(out)


def edges_png(depth_png_path, out, lo=6, hi=18):
    import cv2
    g = cv2.imread(depth_png_path, cv2.IMREAD_GRAYSCALE)
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), lo, hi)
    cv2.imwrite(out, cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))


def canny_png(clay, out):
    import cv2
    g = cv2.cvtColor(cv2.imread(clay), cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    e = cv2.Canny(g, 30, 90)
    cv2.imwrite(out, cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))


def sheet(rows, out, labels):
    from PIL import Image, ImageDraw
    tw, th = 480, 270
    cols = max(len(r) for r in rows)
    img = Image.new("RGB", (cols * tw, len(rows) * (th + 20)), (20, 20, 20))
    d = ImageDraw.Draw(img)
    for i, r in enumerate(rows):
        for j, p in enumerate(r):
            if not p or not os.path.exists(p):
                continue
            t = Image.open(p).convert("RGB").resize((tw, th))
            img.paste(t, (j * tw, i * (th + 20) + 20))
            d.text((j * tw + 4, i * (th + 20) + 4), labels[i][j], fill=(255, 255, 255))
    img.save(out)


def stage_frames(shot, frames):
    """Stage each (name, figures) keyframe. A figure the contact solver had to MOVE (stepping into reach)
    at one keyframe, and that is meant to stand in the same place at the others (same "at"), is moved
    by the same amount at all of them, so the shot has no foot sliding between keyframes."""
    def go(frames):
        out = []
        for name, figs in frames:
            cts = [c for c in shot.get("contacts", []) if c.get("frame", "end") in (name, "both")]
            prims, people, info = stage(copy.deepcopy(figs), shot.get("set"), cts)
            prims += fx_prims(shot.get("fx"), name, shot.get("contacts", []), shot.get("set"), info["_anchors"])
            out.append((name, figs, prims, people, info))
        return out
    staged = go(frames)
    at0 = {f.get("who"): list(f.get("at", [0, 0])) for f in shot["figures"]}
    total = {}
    for _ in range(4):                          # until no keyframe needs a step
        moved = {}
        for name, figs, _, _, info in staged:
            for who, v in info.items():
                mv = v.get("ex", {}).get("moved") if isinstance(v, dict) and v.get("kind") == "human" else None
                if mv is not None and np.linalg.norm(mv) > 1e-3 and who not in moved:
                    moved[who] = mv
        if not moved:
            break
        new = []
        for name, figs in frames:
            figs = copy.deepcopy(figs)
            for f in figs:
                w = f.get("who")
                if w in moved and list(f.get("at", [0, 0])) == at0.get(w):
                    f["at"] = [float(f["at"][0] + moved[w][0]), float(f["at"][1] + moved[w][1])]
            new.append((name, figs))
        for w, mv in moved.items():
            total[w] = total.get(w, np.zeros(3)) + mv
            at0[w] = [at0[w][0] + float(mv[0]), at0[w][1] + float(mv[1])]
        frames = new
        staged = go(frames)
    if total:
        sys.stderr.write("stage: " + ", ".join(f"{w} stands {np.linalg.norm(m[:2]):.2f} m closer in every keyframe "
                                              f"(contact reach)" for w, m in total.items()) + "\n")
    return staged


def resolve_set(shot):
    """The DEFAULT set source: when the shot names a location ("where", in shot["set"] or the shot) and the
    book's stage/assets.json "sets" has a HY-World for it, the set becomes that world (world_blender
    import_world: exteriors use the hole-free "cut" mesh + backdrop). assets.json is found from
    shot["assets"], $BLOCKOUT_ASSETS or shot["book"] (<book>/work/video/stage/assets.json). Changes shot."""
    st = shot.setdefault("set", {})
    where = st.get("where") or shot.get("where")
    if not where or st.get("world"):
        return shot
    bk = shot.get("book")
    if bk and not os.path.isdir(bk):                  # a book NAME (the scene shots): its folder
        import characters as CH
        bk = str(CH.vdir_of(bk).parent.parent)
    cands = [shot.get("assets"), os.environ.get("BLOCKOUT_ASSETS"),
             os.path.join(bk, "work", "video", "stage", "assets.json") if bk else None]
    for ap in [c for c in cands if c and os.path.exists(c)]:
        ent = next((e for e in json.load(open(ap)).get("sets", []) if e.get("where") == where), None)
        if ent:
            st.update(kind="world", ground=False)
            st["world"] = {"dir": ent["world_dir"], "material": "emission",
                           "mesh": "cut" if ent.get("kind") == "exterior" else "full",
                           "scale": float(st.get("world_scale", 1.0)), "offset": st.get("world_offset", [0, 0, 0]),
                           "view_r": (ent.get("view_region") or {}).get("radius_m")}
            sys.stderr.write(f"set: {where} -> HY-World {ent['world_dir']}\n")
            for pr_ in st.get("props", []):              # support stand-ins (a ledge box): the world has its own
                if pr_.get("support") and not pr_.get("prop"):   # ground; the box stays a support, unseen
                    pr_["hidden"] = True
            break
    return shot


def _seg_seg(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = float(d1 @ d1), float(d2 @ d2), float(d2 @ r)
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s_, t_ = 0.0, np.clip(f / e, 0, 1)
    else:
        c = float(d1 @ r)
        if e < 1e-12:
            t_, s_ = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = float(d1 @ d2)
            den = a * e - b * b
            s_ = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t_ = (b * s_ + f) / e
            if t_ < 0:
                t_, s_ = 0.0, np.clip(-c / a, 0, 1)
            elif t_ > 1:
                t_, s_ = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s_) - (p2 + d2 * t_)))


BODY_SEGS = [("pelvis", "neck"), ("neck", "head"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_wrist"),
             ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_hip", "r_knee"), ("r_knee", "r_ankle"),
             ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_shoulder", "l_shoulder")]


def face_visibility(cam_pos, who, info):
    """The fraction of a person's FACE (a 5x5 patch of points around the head, facing the camera) that no
    other figure's body blocks from the camera: other people as joint capsules, creatures as their points."""
    if who not in info or info[who].get("kind") != "human":
        return None
    J = info[who]["joints"]
    hc = np.asarray(J.get("head", J.get("neck")), float)
    cp = np.asarray(cam_pos, float)
    v = cp - hc
    v /= max(np.linalg.norm(v), 1e-9)
    u = np.cross(v, [0, 0, 1.0])
    u = u / max(np.linalg.norm(u), 1e-9)
    w = np.cross(u, v)
    S = [hc + v * 0.08 + u * a * 0.08 + w * b * 0.09 for a in np.linspace(-1, 1, 5) for b in np.linspace(-1, 1, 5)]
    occ = []
    for n, x in info.items():
        if n.startswith("_") or n == who:
            continue
        if x.get("kind") == "human":
            Jx = x["joints"]
            occ += [(np.asarray(Jx[a], float), np.asarray(Jx[b], float), 0.13 if "pelvis" in a else 0.07)
                    for a, b in BODY_SEGS if a in Jx and b in Jx]
            if "head" in Jx:
                occ.append((np.asarray(Jx["head"], float), np.asarray(Jx["head"], float), 0.12))
        else:
            for q in list(x.get("points", []))[::3]:
                q = np.asarray(q, float)
                occ.append((q, q, 0.22))
    vis = 0
    for p_ in S:
        if not any(_seg_seg(cp, p_, a, b) < r for a, b, r in occ):
            vis += 1
    return vis / len(S)


def run(shot, out_dir, size=(1536, 864), samples=16, blend=False, camera_from=None):
    t0 = time.time()
    if shot.get("book"):
        props.load_book(shot["book"])                  # the book's modelled props render as meshes
    os.makedirs(out_dir, exist_ok=True)
    W, H = size
    resolve_set(shot)
    frames = [("start", copy.deepcopy(shot["figures"]))]
    endf = merge_end(shot)
    if endf:
        frames.append(("end", endf))
    staged = stage_frames(shot, frames)
    cam_spec = dict(shot.get("camera", {}))
    if camera_from:
        cam_spec["from"] = camera_from
    # ONE camera for start and end, solved over BOTH frames' points so a figure that moves stays in frame
    cinfo = dict(staged[0][4])
    for _, _, _, _, inf in staged[1:]:
        for who, v in inf.items():
            if who.startswith("_") or who not in cinfo:
                continue
            u = dict(cinfo[who])
            u["points"] = list(u["points"]) + list(v["points"])
            if "anchors" in u:
                u["anchors"] = {k: np.vstack([np.atleast_2d(a), np.atleast_2d(v["anchors"].get(k, a))])
                                for k, a in u["anchors"].items()}
            cinfo[who] = u
    cam = solve_camera(cam_spec, staged[0][1], cinfo, size)
    cam = after_solve(cam, cam_spec, cinfo, shot)
    # READABILITY: now that the camera is known, re-stage so free held props turn (within the grip limits)
    # until their working part reads >= 1.5 fist sizes away from the fist on screen
    if any((f_.get("holds") or {}) for _, figs_, *_ in staged for f_ in figs_) and not STAGE_VIEW.get("pos"):
        STAGE_VIEW["pos"] = cam["pos"]
        try:
            staged = stage_frames(shot, frames)
        finally:
            STAGE_VIEW.clear()
    # held-prop grips and contacts, per frame (blockout_contact): printed, written to control.json
    checks = {name: info["_checks"] for name, _, _, _, info in staged}
    nbad = 0
    for name, cl in checks.items():
        for c in cl:
            for ln in c["log"]:
                sys.stderr.write(f"[{name}] {ln}\n")
            for r in c["rows"]:
                v = r.get("value_cm", r.get("value_deg"))
                sys.stderr.write(f"[{name}] {r['who']}: {r['check']} = {v}\n")
            for r in c["violations"]:
                nbad += 1
                sys.stderr.write(f"[{name}] CONTACT VIOLATION {r['who']}: {r['check']} = "
                                 f"{r.get('value_cm', r.get('value_deg'))} (limit {r.get('limit_cm', r.get('limit_deg'))})\n")
    if os.environ.get("BLOCKOUT_DRY"):
        return {"out": out_dir, "checks": checks, "violations": nbad}
    with open(os.path.join(out_dir, "camera.json"), "w") as f:
        json.dump(cam, f, indent=1)
    scene = {"size": [W, H], "samples": samples, "frames": []}
    meta = {}
    for name, figs, prims, people, info in staged:
        prims = prims + room_walls(shot.get("set") or {}, cam["pos"])
        fr = {"name": name, "prims": [_jsonable(p) for p in prims],
              "camera": {"pos": cam["pos"], "R": cam["R"], "lens": cam["lens"]},
              "clay": os.path.join(out_dir, f"{name}_clay.png"),
              "depth": os.path.join(out_dir, f"{name}_depth.npy")}
        if blend:
            fr["blend"] = os.path.join(out_dir, f"{name}.blend")
        models = info.get("_models")
        if models:
            fr["models"] = models
        if (shot.get("set") or {}).get("world"):        # a HY-World set (mesh) instead of a primitive room
            fr["world"] = shot["set"]["world"]
        scene["frames"].append(fr)
        pts2d = []
        for p in people:
            world = rig.coco18(p["joints"], cam["pos"])
            pts2d.append({"who": p["who"], "coco18": rig.project(cam, world, W, H),
                          "joints3d": {k: v.tolist() for k, v in p["joints"].items() if not k.startswith("_")}})
        meta[name] = pts2d
    # close-ups of every gripping fist (BLOCKOUT_CLOSEUPS=1): a camera from the back-of-hand side,
    # across the handle, so "is the handle inside a closed hand" can be LOOKED at
    closeups = []
    if os.environ.get("BLOCKOUT_CLOSEUPS"):
        for (name, figs, prims, people, info), fr0 in zip(staged, list(scene["frames"])):
            for who, v in info.items():
                if who.startswith("_") or v.get("kind") != "human":
                    continue
                for hand, gi in (v["ex"].get("grip") or {}).items():
                    tocam = rig._n(np.asarray(cam["pos"]) - gi["fist"])      # mostly from the shot camera's
                    view = rig._n(tocam * 0.6 - gi["n_p"] * 0.4 + rig.UP * 0.1)  # side: nothing in between
                    pos = gi["fist"] + view * 0.55
                    cn = f"{name}_grip_{who}_{hand}"
                    closeups.append(os.path.join(out_dir, f"{cn}.png"))
                    scene["frames"].append(dict(fr0, name=cn, size=[512, 512],
                                                camera={"pos": pos.tolist(), "R": rig.look_at(pos, gi["fist"]).tolist(),
                                                        "lens": 50.0},
                                                clay=closeups[-1], depth=os.path.join(out_dir, f".{cn}.npy")))
    spath = os.path.join(out_dir, "scene.json")
    with open(spath, "w") as f:
        json.dump(scene, f, default=lambda o: np.asarray(o).tolist())
    t1 = time.time()
    r = subprocess.run([BLENDER, "--background", "--factory-startup",
                        "-t", os.environ.get("BLOCKOUT_THREADS", "16"), "--python",
                        os.path.join(HERE, "blockout_scene.py"), "--", spath],
                       capture_output=True, text=True, timeout=float(os.environ.get("BLOCKOUT_TIMEOUT", 600)))
    render_s = {ln.split()[1]: float(ln.split()[2][:-1]) for ln in r.stdout.splitlines() if ln.startswith("BLOCKOUT ")}
    if r.returncode or len(render_s) != len(frames) + len(closeups):
        sys.stderr.write(r.stdout[-3000:] + r.stderr[-3000:])
        raise RuntimeError("blender failed")
    t2 = time.time()
    # the posed model's real hands (Blender) vs the numpy FK the props were solved on
    for ln in r.stdout.splitlines():
        if not ln.startswith("BLOCKOUT_HANDS "):
            continue
        hj = json.loads(ln.split(" ", 1)[1])
        if hj["frame"] not in [nm for nm, *_ in staged]:
            continue                                    # a close-up frame
        info = next(inf for nm, _, _, _, inf in staged if nm == hj["frame"])
        J = info[hj["who"]]["joints"]
        mdl = next(m for m in info["_models"] if m.get("who") == hj["who"])
        for s_, (h_, t_) in hj["hands"].items():
            dv = float(np.linalg.norm(np.array(h_) - J[f"{s_}_wrist"]))
            row = {"who": hj["who"], "check": f"model {s_} wrist: Blender vs solved", "value_cm": round(dv * 100, 2),
                   "limit_cm": contact.LIMITS["grip"] * 100}
            c = next(c for c in info["_checks"] if c["who"] == hj["who"])
            c["rows"].append(row)
            sys.stderr.write(f"[{hj['frame']}] {hj['who']}: {row['check']} = {row['value_cm']}\n")
            if dv > contact.LIMITS["grip"]:
                c["violations"].append(row)
                nbad += 1
                sys.stderr.write(f"[{hj['frame']}] CONTACT VIOLATION {hj['who']}: {row['check']} = {row['value_cm']}\n")
    control = {}
    rows, labels = [], []
    for name, *_ in staged:
        pose = os.path.join(out_dir, f"{name}_pose.png")
        from PIL import Image
        people = [p["coco18"] for p in meta[name]]
        Image.fromarray(draw_people(people, W, H)).save(pose)
        depth = os.path.join(out_dir, f"{name}_depth.png")
        depth_png(np.load(os.path.join(out_dir, f"{name}_depth.npy")), depth)
        canny = os.path.join(out_dir, f"{name}_canny.png")
        canny_png(os.path.join(out_dir, f"{name}_clay.png"), canny)
        edges = os.path.join(out_dir, f"{name}_edges.png")    # depth-discontinuity edges: silhouettes of every
        edges_png(depth, edges)                                # object incl. thin props, no painted texture
        with open(os.path.join(out_dir, f"{name}_joints.json"), "w") as f:
            json.dump(meta[name], f)
        control[name] = {"pose": {"image": pose, "type": "pose", "scale": 0.8},
                         "depth": {"image": depth, "type": "depth", "scale": 0.8},
                         "canny": {"image": canny, "type": "canny", "scale": 0.7, "prepared": True},
                         "edges": {"image": edges, "type": "canny", "scale": 0.6, "prepared": True}}
        rows.append([os.path.join(out_dir, f"{name}_clay.png"), depth, pose, canny])
        labels.append([f"{name} clay", f"{name} depth", f"{name} pose", f"{name} canny"])
    control["checks"] = checks
    with open(os.path.join(out_dir, "control.json"), "w") as f:
        json.dump(control, f, indent=1)
    sheet(rows, os.path.join(out_dir, "sheet.png"), labels)
    res = {"out": out_dir, "frames": [n for n, *_ in staged], "seconds": round(time.time() - t0, 2),
           "blender_s": round(t2 - t1, 2), "render_s": render_s, "violations": nbad}
    return res


def _jsonable(p):
    return {k: (np.asarray(v).tolist() if isinstance(v, (np.ndarray, list)) and k != "tris" else v)
            for k, v in p.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("shot")
    ap.add_argument("out")
    ap.add_argument("--size", default="1536x864")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--blend", action="store_true", help="also save start.blend / end.blend")
    ap.add_argument("--camera", help="reuse this camera.json (same camera across shots)")
    a = ap.parse_args(argv)
    with open(a.shot) as f:
        shot = json.load(f)
    w, h = (int(x) for x in a.size.lower().split("x"))
    res = run(shot, a.out, (w, h), a.samples, a.blend, a.camera)
    print(json.dumps(res, default=lambda o: np.asarray(o).tolist()))
    if res.get("violations") and not os.environ.get("BLOCKOUT_ALLOW_VIOLATIONS"):
        sys.stderr.write(f"FAILED: {res['violations']} grip/contact violation(s); see control.json \"checks\" "
                         "(BLOCKOUT_ALLOW_VIOLATIONS=1 to render anyway)\n")
        sys.exit(2)


if __name__ == "__main__":
    main()
