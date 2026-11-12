"""Synthetic scenes for the staging and framing rule tests (stdlib): people as measured boxes, cameras, a tiny
uncompressed EXR writer for depth passes, a library in a temp folder, and a stand-in for the engine.

People are "a", "b", "c". Nothing here is a real scene: the numbers are picked so each rule has a case on each side
of its bar, and the bars themselves are read from the code under test.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
for p in ("assets/scenes", "assets/redraw", "assets"):
    sys.path.insert(0, str(TOOLS / p))

import framing  # noqa: E402

RES = (1152, 640)
FAR_CM = 1.0e5


def unit(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return [v[0] / n, v[1] / n]


def person(xy, facing=(0.0, -1.0), head=None, seated=False, joints=True) -> dict:
    """A measured person standing (or sitting) at xy: {"measured": {...}} as the scene tool reports one."""
    f = unit(facing)
    hf = unit(head if head is not None else facing)
    right = [f[1], -f[0]]
    eye_z = 1.15 if seated else 1.62
    m = {"feet_z": 0.0, "eye_z": eye_z, "head_top_z": eye_z + 0.12, "chin_z": eye_z - 0.12,
         "waist_z": 0.55 if seated else 1.0, "eye_xy": list(xy), "chest_xy": list(xy), "pelvis_xy": list(xy),
         "facing": f, "head_fwd_xy": hf}
    if joints:
        j = {}
        for s, sg in (("l", -1.0), ("r", 1.0)):
            ox, oy = right[0] * sg, right[1] * sg
            j[f"upperarm_{s}"] = [xy[0] + ox * 0.2, xy[1] + oy * 0.2, eye_z - 0.2]
            j[f"lowerarm_{s}"] = [xy[0] + ox * 0.24, xy[1] + oy * 0.24, eye_z - 0.5]
            j[f"hand_{s}"] = [xy[0] + ox * 0.26, xy[1] + oy * 0.26, eye_z - 0.75]
            j[f"thigh_{s}"] = [xy[0] + ox * 0.09, xy[1] + oy * 0.09, 0.5 if seated else 0.92]
            j[f"calf_{s}"] = [xy[0] + ox * 0.09, xy[1] + oy * 0.09, 0.45]
            j[f"foot_{s}"] = [xy[0] + ox * 0.09, xy[1] + oy * 0.09, 0.05]
        m["joints"] = j
    return {"measured": m}


def cam(position, target, focal=50) -> dict:
    return {"position": [float(x) for x in position], "target": [float(x) for x in target], "focal_mm": focal}


def front_cam(subj: dict, dist: float, focal=50, side=0.0, up=0.0) -> dict:
    """A camera straight in front of a person's head (side: metres to the person's own right; up: above eye line)."""
    m = subj["measured"]
    f = unit(m["head_fwd_xy"])
    r = [f[1], -f[0]]
    e = m["eye_xy"]
    return cam([e[0] + f[0] * dist + r[0] * side, e[1] + f[1] * dist + r[1] * side, m["eye_z"] - 0.1 + up],
               [e[0], e[1], m["eye_z"] - 0.1], focal)


# ------------------------------------------------------------------ depth passes
def write_exr(path: Path, rows: list) -> None:
    """An uncompressed scanline EXR with one float channel R (depth, cm): rows[y][x]."""
    h, w = len(rows), len(rows[0])

    def attr(name, typ, val):
        return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(val)) + val
    chl = b"R\0" + struct.pack("<iB3xii", 2, 0, 1, 1) + b"\0"
    hdr = (attr("channels", "chlist", chl) + attr("compression", "compression", b"\0")
           + attr("dataWindow", "box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
           + attr("displayWindow", "box2i", struct.pack("<4i", 0, 0, w - 1, h - 1))
           + attr("lineOrder", "lineOrder", b"\0") + attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
           + attr("screenWindowCenter", "v2f", struct.pack("<2f", 0.0, 0.0))
           + attr("screenWindowWidth", "float", struct.pack("<f", 1.0)) + b"\0")
    head = struct.pack("<ii", 20000630, 2) + hdr
    line = 8 + 4 * w
    base = len(head) + 8 * h
    offs = b"".join(struct.pack("<Q", base + y * line) for y in range(h))
    body = b"".join(struct.pack("<ii", y, 4 * w) + struct.pack(f"<{w}f", *rows[y]) for y in range(h))
    Path(path).write_bytes(head + offs + body)


def body_rect(camera: dict, m: dict, pad=0.02) -> tuple | None:
    """The frame rectangle (u0, v0, u1, v1) a person's box covers, or None when all of it is behind the camera."""
    e, p = m["eye_xy"], m["pelvis_xy"]
    f = unit(m.get("facing", [0, -1]))
    r = [f[1], -f[0]]
    pts = [[e[0], e[1], m["head_top_z"]], [p[0], p[1], m["feet_z"]]]
    for s in (-0.25, 0.25):
        pts += [[e[0] + r[0] * s, e[1] + r[1] * s, m["chin_z"] - 0.1], [p[0] + r[0] * s, p[1] + r[1] * s, m["feet_z"]],
                [e[0] + r[0] * s * 0.45, e[1] + r[1] * s * 0.45, m["head_top_z"]]]
    pr = [framing.project(camera, q, RES) for q in pts]
    pr = [q for q in pr if q]
    if len(pr) < 2:
        return None
    return (min(q[0] for q in pr) - pad, min(q[1] for q in pr) - pad, max(q[0] for q in pr) + pad,
            max(q[1] for q in pr) + pad)


def depth_guides(folder: Path, name: str, camera: dict, subjects: dict, w=144, h=80, hide=()) -> dict:
    """Depth passes for one still, people drawn as their box at their own distance: {"depth": scene, "depth_<key>":
    that person alone}. hide: keys drawn in their own pass but covered in the scene's (something nearer in front)."""
    folder.mkdir(parents=True, exist_ok=True)
    scene = [[FAR_CM] * w for _ in range(h)]
    out = {}
    for k, v in subjects.items():
        m = v["measured"]
        own = [[FAR_CM] * w for _ in range(h)]
        rect = body_rect(camera, m)
        pz = framing.project(camera, [m["pelvis_xy"][0], m["pelvis_xy"][1], 1.0], RES)
        if rect and pz:
            d = pz[2] * 100.0
            for y in range(max(0, int(rect[1] * h)), min(h, int(math.ceil(rect[3] * h)))):
                for x in range(max(0, int(rect[0] * w)), min(w, int(math.ceil(rect[2] * w)))):
                    own[y][x] = d
                    scene[y][x] = min(scene[y][x], d * (0.5 if k in hide else 1.0))
        p = folder / f"{name}_depth_{k}.exr"
        write_exr(p, own)
        out[f"depth_{k}"] = str(p)
    p = folder / f"{name}_depth.exr"
    write_exr(p, scene)
    out["depth"] = str(p)
    return out


# ------------------------------------------------------------------ a library in a temp folder
def seat_block(half_in=0.28, seat_z=0.45, back_y=0.3, front_y=-0.3) -> dict:
    return {"seat_z_m": seat_z, "inner_x_m": [-half_in, half_in], "arm_z_m": 0.65, "back_y_m": back_y,
            "front_y_m": front_y, "front_asset": [0.0, -1.0]}


def make_library(tmp: Path, fixtures: list, size=(6.0, 5.0, 3.0), lights=(), openings=(), set_id="room_x",
                 description="A sitting room with a fireplace, a desk and armchairs", props=()) -> Path:
    """<tmp>/library with one set. fixtures: [{"name", "position", "size": [x, y, z], "seat": bool, "facing"}]."""
    lib = tmp / "library"
    spec = {"size_m": list(size), "structure": [], "lights": list(lights), "fixtures": [], "props": []}
    for kind, items in (("fixtures", fixtures), ("props", props)):
        for f in items:
            aid = f.get("asset") or f"{f['name']}_asset"
            d = lib / "prop" / aid
            d.mkdir(parents=True, exist_ok=True)
            rec = {"description": f.get("description", f["name"].replace("_", " ")), "size_m": f.get("size", [0.8, 0.8, 1.0])}
            if f.get("seat"):
                rec["seat"] = seat_block()
            if f.get("variant_of"):
                rec["variant_of"] = f["variant_of"]
            (d / "asset.json").write_text(json.dumps(rec))
            item = {"asset": aid, "name": f["name"], "position": list(f["position"])}
            for kk in ("facing", "look"):
                if kk in f:
                    item[kk] = f[kk]
            spec[kind].append(item)
    by_side = {}
    for o in openings:                       # {"side", "kind", "from_m", "to_m"}
        by_side.setdefault(o["side"], []).append({k: v for k, v in o.items() if k != "side"})
    for side, ops in by_side.items():
        spec["structure"].append({"kind": "wall", "side": side, "openings": ops})
    s = lib / "set" / set_id
    s.mkdir(parents=True, exist_ok=True)
    (s / "spec.json").write_text(json.dumps(spec))
    (s / "asset.json").write_text(json.dumps({"id": set_id, "description": description, "tags": ["set", "room"]}))
    return lib


def add_character(lib: Path, cid: str, record: dict | None = None) -> None:
    d = lib / "character" / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / "character.json").write_text(json.dumps(record or {}))


# ------------------------------------------------------------------ a stand-in for the engine
class FakeEngine:
    """Stands in for scenes/scene.py: reads the round's spec and writes result.json with every person measured where
    the spec puts them and each camera's still described (camera, optional depth passes). Records every call."""

    def __init__(self, out_dir: Path, depth=False, hide=None):
        self.out_dir, self.depth, self.hide = out_dir, depth, hide or (lambda spec, cam_name: ())
        self.calls = []

    def measured(self, c: dict) -> dict:
        pose = ((c.get("body_pose") or {}).get("pose") or "")
        seated = "sit" in pose or "seat" in pose
        head = c.get("facing", [0, -1])
        lk = c.get("_look")
        if lk:                               # the head turned by the look steps (about z, toward the right +)
            a = -math.radians(lk["yaw"])
            f = unit(c.get("facing", [0, -1]))
            head = [f[0] * math.cos(a) - f[1] * math.sin(a), f[0] * math.sin(a) + f[1] * math.cos(a)]
        v = person(c["position"][:2], c.get("facing", [0, -1]), head, seated)
        v["measured"]["feet_above_floor_m"] = 0.0
        return v

    def camera(self, c: dict, subjects: dict) -> dict:
        if c.get("position"):
            return {"position": c["position"], "target": c["target"], "focal_mm": c["focal_mm"]}
        s = framing.shot(subjects[c["subject"]]["measured"], c["kind"], c.get("focal_mm", 50), RES, c.get("az", 0.0),
                         c.get("el", 0.0), c.get("screen_x", 0.5))
        return {"position": s["position"], "target": s["target"], "focal_mm": s["focal_mm"]}

    def __call__(self, spec_path, cams=None):
        spec = json.loads(Path(spec_path).read_text())
        self.calls.append({"spec": str(spec_path), "cams": list(cams) if cams else None,
                           "positions": {k: list(v["position"]) for k, v in spec["characters"].items()}})
        odir = self.out_dir / spec["id"]
        odir.mkdir(parents=True, exist_ok=True)
        subjects = {k: self.measured(c) for k, c in spec["characters"].items()}
        shots = []
        for c in spec["cameras"]:
            if cams and c["name"] not in cams:
                continue
            cm = self.camera(c, subjects)
            sh = {"name": c["name"], "kind": c["kind"], "camera": cm, "path": str(odir / f"{c['name']}.png")}
            if self.depth:
                sh["guides"] = depth_guides(odir / "guides", f"{c['name']}_{len(self.calls)}", cm, subjects,
                                            hide=self.hide(spec, c["name"]))
            shots.append(sh)
        (odir / "result.json").write_text(json.dumps({"subjects": subjects, "shots": shots}))
