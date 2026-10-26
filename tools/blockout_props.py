"""The prop library + affordances (layer 1), pure numpy: tools/stage_props/props.json (shared) plus an
optional per-book library (BLOCKOUT_PROPS_EXTRA=/abs/props.json, or a shot's "props_library").

- held props: parts in the prop frame [d, flat, side] (d = grip -> working end), metres at 1.8 m.
- set props: parts in a normalised box frame (x in [-0.5, 0.5] of w along yaw, y of d, z in [0, 1] of h).
- wear props: parts in a head frame in head radii.
- affordances: grip / face / point / edge / top (see props.json "_doc").
No prop is special-cased in code: everything a solver needs comes from these records.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "stage_props", "props.json")
POSE3D_PATH = os.path.join(HERE, "stage_props", "pose3d.json")
UP = np.array([0.0, 0.0, 1.0])
_LIB = None


def _n(v):
    v = np.asarray(v, float)
    d = float(np.linalg.norm(v))
    return v / d if d > 1e-9 else v


def library(extra=None) -> dict:
    global _LIB
    if _LIB is None:
        with open(LIB_PATH) as f:
            _LIB = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        for p in filter(None, [os.environ.get("BLOCKOUT_PROPS_EXTRA")]):
            with open(p) as f:
                _LIB.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    if extra:
        _LIB.update(extra)
    return _LIB


def pose3d() -> dict:
    with open(POSE3D_PATH) as f:
        return json.load(f)


def get(item: str) -> dict:
    lib = library()
    if item not in lib:
        raise KeyError(f"prop {item!r} is not in the library ({LIB_PATH})")
    return lib[item]


def affordance(item: str, name: str) -> dict:
    a = get(item).get("affordances", {})
    if name not in a:
        raise KeyError(f"prop {item!r} has no affordance {name!r} (has {sorted(a)})")
    return a[name]


def has_grip(item: str) -> bool:
    return item in library() and "grip" in get(item).get("affordances", {})


def set_entry(pr: dict):
    """The library record of a set prop: its "prop" key, else the first library alias found in its name."""
    lib = library()
    if pr.get("prop"):
        return pr["prop"], lib[pr["prop"]]
    name = pr.get("name", "").lower()
    for k, v in lib.items():
        if v.get("kind") == "set" and any(a in name for a in v.get("aliases", [])):
            return k, v
    return "box", lib["box"]


# --------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------

def prop_axes(d, flat=None, left=None):
    """(d, side, flat): flat defaults to the 'up-ish' normal of d."""
    d = _n(d)
    if flat is None:
        side = _n(np.cross(d, UP)) if abs(np.dot(d, UP)) < 0.95 else _n(left if left is not None else [0, 1, 0])
        flat = _n(np.cross(side, d))
    else:
        flat = _n(np.asarray(flat, float) - d * np.dot(flat, d))
        side = _n(np.cross(d, flat))
    return d, side, flat


def _part_prims(parts, origin, axes, scale, tag, ref_r=None):
    """parts in a local frame (columns of axes = local x, y, z) -> primitives."""
    A = np.stack(axes, 1)
    sc = np.asarray(scale, float) * np.ones(3)
    P = []
    for p in parts:
        t = p["t"]
        if t == "capsule":
            r = p.get("r", 0.01) * (ref_r if ref_r is not None else sc[0])
            if "r_min_dh" in p:
                r = p["r_min_dh"] * min(sc[1], sc[2])
            P.append({"t": "capsule", "a": origin + A @ (np.asarray(p["a"]) * sc), "b": origin + A @ (np.asarray(p["b"]) * sc),
                      "r": float(r), "tag": p.get("tag", tag)})
        elif t in ("box", "ellipsoid"):
            h = np.asarray(p["half"]) * sc
            P.append({"t": t, "c": origin + A @ (np.asarray(p["c"]) * sc), "M": A * h[None, :], "tag": p.get("tag", tag)})
        elif t == "prism":
            z0, z1 = p.get("z", [0, 1])
            poly = p["poly"]
            vs = [origin + A @ (np.array([x, y, z]) * sc) for z in (z0, z1) for x, y in poly]
            n = len(poly)
            tris = [[0, 1, 2]] if n == 3 else [[0, i, i + 1] for i in range(1, n - 1)]
            tris = [list(reversed(t_)) for t_ in tris] + [[i + n for i in t_] for t_ in tris]
            for i in range(n):
                j = (i + 1) % n
                tris += [[i, j, j + n], [i, j + n, i + n]]
            P.append({"t": "mesh", "verts": [v.tolist() for v in vs], "tris": tris, "tag": p.get("tag", tag)})
    return P


def load_book(book):
    """Add the book's modelled props (<book>/work/video/stage/props/props.json, tools/props_build.py) to the library."""
    import characters as CH
    pth = CH.vdir_of(book) / "stage" / "props" / "props.json"
    if pth.exists():
        lib = library()
        shared = {k: dict(v) for k, v in lib.items() if isinstance(v, dict)}
        for k, v in json.loads(pth.read_text()).items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            base = shared.get(k.split("__")[0]) or {}
            # the shared library's contact points reach a book built before them (10-22: the shield was built with
            # none, so its grip was never checked): an affordance the book's entry lacks comes from the library
            for an, av in (base.get("affordances") or {}).items():
                v.setdefault("affordances", {}).setdefault(an, av)
            lib[k] = v


def _with_mesh(e, P, origin, axes, scale, tag, state=None, joint_state=None):
    """A prop with a 3D model renders its MESH; its library parts stay as the (unrendered) proxy for contacts,
    collisions and framing."""
    if not e.get("mesh"):
        return P
    for p in P:
        p["proxy"] = True
    A = np.stack(axes, 1) * (np.asarray(scale, float) * np.ones(3))[None, :]
    M = np.eye(4)
    M[:3, :3] = A
    M[:3, 3] = np.asarray(origin, float)
    st = (e.get("states") or {}).get(state) if state else None
    blend = (e.get("state_meshes") or {}).get(state) or e["mesh"]       # a state with its own shape: its mesh
    if (e.get("state_meshes") or {}).get(state):
        st = None                                                       # the mesh IS the state (no tint)
    P.append({"t": "propmesh", "blend": blend, "M": M.tolist(), "tag": tag, "state": st,
              "joints": joint_state or {}, "joint_defs": e.get("joints") or [], "light": e.get("light")})
    return P


def held_prims(item, at, d, fwd, left, ks, flat=None):
    """A held prop with its grip at `at`, its axis along d, turned about d by flat (ks = height/1.8)."""
    e = get(item)
    tag = e.get("tag", "prop")
    if e.get("frame") == "body":
        axes = (fwd, left, UP)
    elif e.get("frame") == "up":
        dd = _n(np.asarray(d) - UP * np.dot(d, UP)) if abs(np.dot(_n(d), UP)) < 0.99 else _n(fwd)
        axes = (dd, _n(np.cross(UP, dd)), UP)
    else:
        d_, side, fl = prop_axes(d, flat, left)
        axes = (d_, fl, side)
    P = _part_prims(e["parts"], np.asarray(at, float), axes, ks, tag)
    return _with_mesh(e, P, at, axes, ks, tag, state=e.get("state"))


def set_prims(pr: dict):
    """A set prop {name|prop, box: [x, y, z, w, d, h], yaw, hot} from its library parts."""
    _, e = set_entry(pr)
    x, y, z, w, d, h = pr["box"]
    yaw = math.radians(pr.get("yaw", 0))
    fx = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    fy = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    fz = UP.copy()
    if pr.get("roll"):                         # a held prop laid down: turned about its own length (a blade flat)
        r = math.radians(pr["roll"])
        fy, fz = fy * math.cos(r) + fz * math.sin(r), -fy * math.sin(r) + fz * math.cos(r)
    tag = "hot" if pr.get("hot") else ("prop" if e.get("kind") == "held" else "set")
    P = _part_prims(e["parts"], np.array([x, y, z], float), (fx, fy, fz), (w, d, h), tag)
    if e.get("mesh") and e.get("extent_m"):
        # a MODELLED prop renders at its own (class) size in metres, never stretched by the proxy box; a class
        # "stand_m" (an anvil's working height) puts it on a plain stump so its top is at that height
        lift = max(0.0, float(e.get("stand_m") or 0.0) - float(e["extent_m"][2]))
        if lift > 0.02 and not pr.get("roll"):
            r_ = 0.3 * min(float(e["extent_m"][0]), float(e["extent_m"][1])) + 0.08
            P.append({"t": "box", "c": np.array([x, y, z]) + UP * lift / 2,
                      "M": np.stack([fx, fy, UP], 1) * np.array([r_, r_, lift / 2])[None, :], "tag": "set",
                      "stump": True})
        else:
            lift = 0.0
        out_ = _with_mesh(e, P, [x, y, z + lift], (fx, fy, fz), (1.0, 1.0, 1.0), tag, state=pr.get("state"),
                          joint_state=pr.get("joints"))
        for q in out_:
            if q.get("stump"):
                q["proxy"] = False                      # the stump RENDERS (it is not a proxy of the mesh)
                q["tag"] = "stump"
        return out_
    return _with_mesh(e, P, [x, y, z], (fx, fy, fz), (w, d, h), tag, state=pr.get("state"),
                      joint_state=pr.get("joints"))


def wear_prims(item, head_c, head_fwd, head_left, head_up, r):
    e = get(item)
    return _part_prims(e["parts"], np.asarray(head_c, float), (head_fwd, head_left, head_up), r, e.get("tag", "prop"))


def set_top(pr: dict, at_xy=None):
    """(point, normal, half extents in metres) of a set prop's 'top' affordance."""
    _, e = set_entry(pr)
    x, y, z, w, d, h = pr["box"]
    a = e.get("affordances", {}).get("top", {"at": [0, 0, 1], "normal": [0, 0, 1], "half": [0.5, 0.5]})
    yaw = math.radians(pr.get("yaw", 0))
    fx = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    fy = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    c = np.array([x, y, z]) + fx * a["at"][0] * w + fy * a["at"][1] * d + UP * a["at"][2] * h
    if e.get("mesh") and e.get("extent_m") and not pr.get("roll"):
        # the MODELLED prop's real top (its own height, on its stand)
        mh = float(e["extent_m"][2])
        c[2] = z + max(mh, float(e.get("stand_m") or 0.0))
    if at_xy is not None:
        c = np.array([float(at_xy[0]), float(at_xy[1]), c[2]])
    return c, _n(a["normal"]), (a["half"][0] * w, a["half"][1] * d)


# --------------------------------------------------------------------------------------------------
# Affordance points (prop frame -> world)
# --------------------------------------------------------------------------------------------------

def point(item, name, at, d, flat, ks):
    a = affordance(item, name)
    d_, side, fl = prop_axes(d, flat)
    g = a["at"]
    return np.asarray(at, float) + (d_ * g[0] + fl * g[1] + side * g[2]) * ks


def handle_seg(item, at, d, ks):
    g = affordance(item, "grip")
    return np.asarray(at, float) + _n(d) * g["from"] * ks, np.asarray(at, float) + _n(d) * g["to"] * ks


def part_corners(item, idx, at, d, flat, ks):
    """The 8 corners of a box part (for penetration checks)."""
    p = get(item)["parts"][idx]
    d_, side, fl = prop_axes(d, flat)
    A = np.stack([d_, fl, side], 1)
    c = np.asarray(at, float) + A @ (np.asarray(p["c"]) * ks)
    h = np.asarray(p["half"]) * ks
    return [c + A @ (np.array([sx, sy, sz]) * h) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]


def joint_handle(pr: dict, jname: str, angle_deg: float | None = None):
    """A modelled set prop's jointed part: (handle, pivot) in WORLD metres with the part turned angle_deg about its
    hinge (None = rest), from the model's handle_m / pivot_m (props_build) and the prop's box, yaw and stand - the
    same placement as set_prims. None when the prop has no such joint or handle (10-22: the bellows pump)."""
    _, e = set_entry(pr)
    jd = next((j for j in e.get("joints") or [] if j.get("name") == jname), None)
    if not jd or not jd.get("handle_m") or not jd.get("pivot_m") or not e.get("mesh") or not e.get("extent_m"):
        return None
    x, y, z, w, d, h = pr["box"]
    lift = max(0.0, float(e.get("stand_m") or 0.0) - float(e["extent_m"][2]))
    lift = lift if lift > 0.02 and not pr.get("roll") else 0.0
    yaw = math.radians(pr.get("yaw", 0))
    A = np.stack([[math.cos(yaw), math.sin(yaw), 0.0], [-math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]], 1)
    o = np.array([x, y, z + lift], float)
    pv, hd = np.asarray(jd["pivot_m"], float), np.asarray(jd["handle_m"], float)
    a = math.radians(jd.get("rest", 0.0) if angle_deg is None else angle_deg)
    ax = {"x": 0, "y": 1, "z": 2}[jd.get("axis", "y")]
    c, s = math.cos(a), math.sin(a)
    R = {0: [[1, 0, 0], [0, c, -s], [0, s, c]], 1: [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         2: [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[ax]                   # Blender's euler about that axis
    hd_ = pv + np.asarray(R) @ (hd - pv)
    return o + A @ hd_, o + A @ pv


def joint_cycle(pr: dict, jname: str, t_s: float) -> float | None:
    """The part's angle at t_s seconds of its operating cycle (joints[].cycle: period_s, range): eased in and out
    (cosine), starting at rest. None without a cycle."""
    _, e = set_entry(pr)
    jd = next((j for j in e.get("joints") or [] if j.get("name") == jname), None)
    cy = (jd or {}).get("cycle")
    if not cy:
        return None
    lo, hi = cy.get("range", jd.get("limits", [0, 20]))
    return float(lo + (hi - lo) * 0.5 * (1 - math.cos(2 * math.pi * t_s / float(cy.get("period_s", 1.4)))))
