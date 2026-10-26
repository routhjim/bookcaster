"""Creature template skeletons: ONE definition shared by rigging (blockout agent) and motion (layer 3).

    python tools/motion_templates.py          # (re)writes tools/stage_templates/creatures.json from TEMPLATES
    from motion_templates import load_template
    T = load_template("winged_quadruped")     # dict: joints (parent-first), parents, rest, chains, ...

The JSON is the contract; this file only generates and reads it. Units: BODY LENGTH = 1.0, where body
length = |chest - pelvis| (the trunk). Frame: +X = forward (head end), Z up, +Y = the creature's left,
feet on z = 0. Assets are scaled by their own body length (assets.json: "body_length_m").

Per template:
  joints   names, parent-first          parents  {joint: parent or null}
  rest     {joint: [x, y, z]}           chains   named joint chains (spine, neck, tail, legs, wings, jaw)
  feet     joints that plant            anchors  {name: {"joint": j, "offset": [dx, dy, dz]}} (offset in the
                                                  rest frame, rides the joint's rotation)
  membrane {side: [joints the wing membrane spans, in polygon order]} (winged only)
  limits   {joint: {"angle": [lo, hi]}}: the bend angle (deg) at the joint between its incoming and outgoing
           segment, 180 = straight. Wings: elbow/wrist small = folded, large = spread.
           {"elev": [lo, hi]}: a wing shoulder's humerus elevation above horizontal.
           {"open": [lo, hi]}: jaw opening (deg).
Naming: lowercase snake case, side suffix _l / _r.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "stage_templates", "creatures.json")


def _mirror(rest, parents, names):
    """Copy every *_l joint to *_r with y negated."""
    for n in list(names):
        if n.endswith("_l") or "_l_" in n:
            r = n.replace("_l_", "_r_") if "_l_" in n else n[:-2] + "_r"
            p = parents[n]
            if p and ("_l_" in p or p.endswith("_l")):
                p = p.replace("_l_", "_r_") if "_l_" in p else p[:-2] + "_r"
            x, y, z = rest[n]
            rest[r] = [x, -y, z]
            parents[r] = p
            names.append(r)


def _chain(prefix, n, start, step, parent, rest, parents, names, bend=None):
    prev = parent
    for i in range(n):
        nm = f"{prefix}{i + 1}"
        p = np.asarray(start, float) + np.asarray(step, float) * i
        if bend is not None:
            p = p + np.asarray(bend, float) * (i ** 2)
        rest[nm] = [round(float(v), 4) for v in p]
        parents[nm] = prev
        names.append(nm)
        prev = nm
    return prev


def leg(prefix, pts, parent, rest, parents, names):
    prev = parent
    for part, p in zip(("hip", "knee", "ankle", "toe"), pts):
        nm = f"{prefix}_{part}"
        rest[nm] = list(p)
        parents[nm] = prev
        names.append(nm)
        prev = nm


def build_winged_quadruped(wings=True):
    names, parents, rest = [], {}, {}

    def add(n, p, xyz):
        names.append(n)
        parents[n] = p
        rest[n] = list(xyz)
    add("pelvis", None, [0.0, 0.0, 0.80])
    add("spine1", "pelvis", [0.25, 0.0, 0.83])
    add("spine2", "spine1", [0.50, 0.0, 0.86])
    add("spine3", "spine2", [0.75, 0.0, 0.88])
    add("chest", "spine3", [1.00, 0.0, 0.88])
    add("neck1", "chest", [1.14, 0.0, 1.00])
    add("neck2", "neck1", [1.27, 0.0, 1.17])
    add("neck3", "neck2", [1.38, 0.0, 1.36])
    add("neck4", "neck3", [1.47, 0.0, 1.55])
    add("head", "neck4", [1.57, 0.0, 1.70])
    add("jaw", "head", [1.60, 0.0, 1.64])
    _chain("tail", 8, [-0.22, 0.0, 0.78], [-0.24, 0.0, -0.065], "pelvis", rest, parents, names)
    leg("fleg_l", [[0.95, 0.22, 0.78], [1.00, 0.26, 0.45], [0.95, 0.26, 0.12], [1.08, 0.27, 0.0]], "chest",
        rest, parents, names)
    leg("hleg_l", [[0.05, 0.25, 0.78], [0.24, 0.29, 0.46], [0.04, 0.29, 0.15], [0.18, 0.29, 0.0]], "pelvis",
        rest, parents, names)
    if wings:   # rest = half-spread (good for skinning); the motion folds/flaps within "limits"
        add("wing_l_shoulder", "spine3", [0.85, 0.18, 1.02])
        add("wing_l_elbow", "wing_l_shoulder", [0.68, 0.80, 1.18])
        add("wing_l_wrist", "wing_l_elbow", [0.78, 1.45, 1.28])
        add("wing_l_tip", "wing_l_wrist", [0.55, 2.45, 1.32])
        add("wing_l_finger1", "wing_l_wrist", [0.18, 2.05, 1.14])
        add("wing_l_finger2", "wing_l_wrist", [-0.12, 1.55, 1.02])
        add("wing_l_finger3", "wing_l_wrist", [-0.28, 0.95, 0.96])
    _mirror(rest, parents, names)
    order = _parent_first(names, parents)
    T = {"joints": order, "parents": parents, "rest": {k: [round(v, 4) for v in rest[k]] for k in order},
         "body_length_def": "|chest - pelvis| = 1.0",
         "chains": {"spine": ["pelvis", "spine1", "spine2", "spine3", "chest"],
                    "neck": ["chest", "neck1", "neck2", "neck3", "neck4", "head"],
                    "tail": ["pelvis"] + [f"tail{i}" for i in range(1, 9)], "jaw": ["head", "jaw"]},
         "feet": ["fleg_l_toe", "fleg_r_toe", "hleg_l_toe", "hleg_r_toe"],
         "anchors": {"head": {"joint": "head", "offset": [0.10, 0.0, 0.05]},
                     "snout": {"joint": "head", "offset": [0.42, 0.0, -0.02]},
                     "jaw": {"joint": "jaw", "offset": [0.36, 0.0, -0.06]},
                     "chest": {"joint": "chest", "offset": [0.12, 0.0, -0.12]},
                     "claw_l": {"joint": "fleg_l_toe", "offset": [0.06, 0.0, 0.0]},
                     "claw_r": {"joint": "fleg_r_toe", "offset": [0.06, 0.0, 0.0]},
                     "claw_tip_l": {"joint": "fleg_l_toe", "offset": [0.12, 0.0, 0.01]},
                     "claw_tip_r": {"joint": "fleg_r_toe", "offset": [0.12, 0.0, 0.01]},
                     "belly": {"joint": "spine2", "offset": [0.0, 0.0, -0.22]}},
         "limits": {}}
    for s in ("l", "r"):
        T["chains"][f"leg_f{s}"] = [f"fleg_{s}_{p}" for p in ("hip", "knee", "ankle", "toe")]
        T["chains"][f"leg_h{s}"] = [f"hleg_{s}_{p}" for p in ("hip", "knee", "ankle", "toe")]
        T["limits"].update({f"fleg_{s}_knee": {"angle": [60, 178]}, f"fleg_{s}_ankle": {"angle": [70, 180]},
                            f"hleg_{s}_knee": {"angle": [50, 175]}, f"hleg_{s}_ankle": {"angle": [40, 175]}})
        if wings:
            T["chains"][f"wing_{s}"] = [f"wing_{s}_{p}" for p in ("shoulder", "elbow", "wrist", "tip")]
            T["chains"][f"wing_{s}_fingers"] = [f"wing_{s}_finger{i}" for i in (1, 2, 3)]
            T["membrane"] = T.get("membrane", {})
            T["membrane"][s] = [f"wing_{s}_shoulder", f"wing_{s}_elbow", f"wing_{s}_wrist", f"wing_{s}_tip",
                                f"wing_{s}_finger1", f"wing_{s}_finger2", f"wing_{s}_finger3", "pelvis", "spine2"]
            T["limits"].update({f"wing_{s}_shoulder": {"elev": [-35, 85]},
                                f"wing_{s}_elbow": {"angle": [20, 170]},
                                f"wing_{s}_wrist": {"angle": [12, 175]}})
    for j in ("neck1", "neck2", "neck3", "neck4"):
        T["limits"][j] = {"angle": [120, 180]}
    for i in range(1, 9):
        T["limits"][f"tail{i}"] = {"angle": [140, 180]}
    T["limits"]["jaw"] = {"open": [0, 45]}
    if wings:
        T["wing_poses"] = {"folded": {"elbow": 25, "wrist": 15, "elev": 10},
                           "half": {"elbow": 95, "wrist": 90, "elev": 20},
                           "spread": {"elbow": 165, "wrist": 170, "elev": 25}}
    return T


def build_biped_creature():
    names, parents, rest = [], {}, {}

    def add(n, p, xyz):
        names.append(n)
        parents[n] = p
        rest[n] = list(xyz)
    # upright: trunk vertical, body length = pelvis->chest (1.0)
    add("pelvis", None, [0.0, 0.0, 1.10])
    add("spine1", "pelvis", [0.0, 0.0, 1.35])
    add("spine2", "spine1", [0.0, 0.0, 1.60])
    add("spine3", "spine2", [0.0, 0.0, 1.85])
    add("chest", "spine3", [0.02, 0.0, 2.10])
    add("neck1", "chest", [0.05, 0.0, 2.25])
    add("neck2", "neck1", [0.08, 0.0, 2.38])
    add("head", "neck2", [0.12, 0.0, 2.50])
    add("jaw", "head", [0.16, 0.0, 2.44])
    add("arm_l_shoulder", "chest", [0.0, 0.32, 2.05])
    add("arm_l_elbow", "arm_l_shoulder", [0.0, 0.62, 1.70])
    add("arm_l_wrist", "arm_l_elbow", [0.05, 0.78, 1.35])
    add("arm_l_hand", "arm_l_wrist", [0.10, 0.84, 1.18])
    leg("leg_l", [[0.0, 0.20, 1.05], [0.05, 0.24, 0.58], [0.0, 0.24, 0.10], [0.20, 0.24, 0.0]], "pelvis",
        rest, parents, names)
    _chain("tail", 5, [-0.25, 0.0, 1.00], [-0.25, 0.0, -0.14], "pelvis", rest, parents, names)
    _mirror(rest, parents, names)
    order = _parent_first(names, parents)
    T = {"joints": order, "parents": parents, "rest": {k: [round(v, 4) for v in rest[k]] for k in order},
         "body_length_def": "|chest - pelvis| = 1.0",
         "chains": {"spine": ["pelvis", "spine1", "spine2", "spine3", "chest"], "neck": ["chest", "neck1", "neck2", "head"],
                    "tail": ["pelvis"] + [f"tail{i}" for i in range(1, 6)], "jaw": ["head", "jaw"]},
         "feet": ["leg_l_toe", "leg_r_toe"],
         "anchors": {"head": {"joint": "head", "offset": [0.08, 0.0, 0.05]},
                     "jaw": {"joint": "jaw", "offset": [0.22, 0.0, -0.04]},
                     "chest": {"joint": "chest", "offset": [0.15, 0.0, -0.10]},
                     "claw_l": {"joint": "arm_l_hand", "offset": [0.05, 0.0, -0.08]},
                     "claw_r": {"joint": "arm_r_hand", "offset": [0.05, 0.0, -0.08]}},
         "limits": {"jaw": {"open": [0, 40]}}}
    for s in ("l", "r"):
        T["chains"][f"arm_{s}"] = [f"arm_{s}_{p}" for p in ("shoulder", "elbow", "wrist", "hand")]
        T["chains"][f"leg_{s}"] = [f"leg_{s}_{p}" for p in ("hip", "knee", "ankle", "toe")]
        T["limits"].update({f"arm_{s}_elbow": {"angle": [30, 180]}, f"leg_{s}_knee": {"angle": [40, 180]}})
    return T


def build_serpent(n=16):
    names, parents, rest = [], {}, {}
    names.append("head")
    parents["head"] = None
    rest["head"] = [0.0, 0.0, 0.12]
    prev = "head"
    for i in range(1, n + 1):     # body1..n trail behind the head (-X); body length = 16 segments of 1/16...
        nm = f"body{i}"
        names.append(nm)
        parents[nm] = prev
        rest[nm] = [round(-0.25 * i, 4), 0.0, 0.10]
        prev = nm
    names.append("jaw")
    parents["jaw"] = "head"
    rest["jaw"] = [0.05, 0.0, 0.08]
    T = {"joints": names, "parents": parents, "rest": rest,
         "body_length_def": "|head - body4| = 1.0 (serpents: the first 4 segments)",
         "chains": {"body": names[:n + 1], "jaw": ["head", "jaw"]},
         "feet": [], "anchors": {"head": {"joint": "head", "offset": [0.12, 0.0, 0.03]},
                                 "jaw": {"joint": "jaw", "offset": [0.20, 0.0, -0.02]}},
         "limits": {f"body{i}": {"angle": [135, 180]} for i in range(1, n + 1)}}
    T["limits"]["jaw"] = {"open": [0, 60]}
    return T


def _parent_first(names, parents):
    order, seen = [], set()

    def visit(n):
        if n in seen:
            return
        p = parents[n]
        if p:
            visit(p)
        seen.add(n)
        order.append(n)
    for n in names:
        visit(n)
    return order


TEMPLATES = {"winged_quadruped": lambda: build_winged_quadruped(True),
             "quadruped": lambda: build_winged_quadruped(False),
             "biped_creature": build_biped_creature,
             "serpent": build_serpent}


def write(path=PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"format": "creature_templates/1",
            "frame": "metres at body length 1.0 (|chest - pelvis|); +X forward, Z up, +Y = left, feet on z=0",
            "templates": {k: f() for k, f in TEMPLATES.items()}}
    with open(path, "w") as f:
        json.dump(data, f, indent=1)
    return path


_CACHE = {}


def load_template(name, path=PATH) -> dict:
    if (name, path) not in _CACHE:
        _CACHE[(name, path)] = json.load(open(path))["templates"][name]
    return copy.deepcopy(_CACHE[(name, path)])


if __name__ == "__main__":
    p = write()
    d = json.load(open(p))
    for k, t in d["templates"].items():
        print(k, len(t["joints"]), "joints", sorted(t["chains"]))
