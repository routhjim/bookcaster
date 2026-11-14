"""Seating: a seated person is placed from the chair's own geometry and checked from the engine's facts (stdlib).

A character in a scene spec with a sitting pose ("sit..." / "seated") and "seat": "<set item name>" (or "seat":
true: the set's nearest seat to its "position") gets, before staging (with "seated_from": "<base id>" in place of
"id", the library's "<base>_seated_*" stance settled for that seat is chosen: seated_variant; params "hands": "lap"
adds the character tools' hands-in-lap steps for the seat's height),
  position  the hip point on the seat: centred between the arms (or on the nearest of a sofa's places), at the
            seat's depth with the hips a hip's depth in front of the back,
  facing    the seat's real front (its library "seat" block, scenes/seatbox.py: the front found from the mesh, so a
            mesh that breaks "front -Y" still seats the sitter facing out of it),
  params    seat_z_m (the cushion), arm_z_m / arm_x_m (the arm rests, forearms along them).
After the engine has posed and measured the person, check() reads their joints: both hip joints between the
arms' inner faces with room for hips and dress, above the cushion, in front of the back, behind the knees' front
edge, the feet on the floor. A failure is re-placed (place() again with the measured error taken off), never
redrawn; a seat narrower than the person's hips and dress is refused and the next free seat that fits is used.

Per-person allowances (character.json "seat_fit", else the defaults): hip_half_m (half the hips and dress across
the seat, from the hip joint line's centre), hip_back_m (hip joint to the seat back when sitting back).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HIP_HALF_M = 0.20        # half the width a seated adult's hips and clothes take (dress: settle allowance included)
HIP_BACK_M = 0.16        # hip joint to the chair back when sitting back against it
TOL_M = 0.03             # how far clothes may press into an arm or the back
FEET_TOL_M = 0.03
SEAT_WORDS = ("chair", "sofa", "settee", "stool", "bench", "seat")


def _rot(v, deg):
    a = math.radians(deg)
    return [v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)]


def _ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def seats(library: Path, set_dir: Path) -> dict:
    """{item name: {"pos": [x, y], "front": [dx, dy] (world), "box": seat block, "to_world": (deg, [x, y])}} for the
    set's items whose library record has a "seat" block."""
    spec = json.loads((set_dir / "spec.json").read_text())
    built = {}                                       # the yaw each item was really placed with (the set's scene.json)
    if (set_dir / "scene.json").exists():
        built = {a["name"]: a["yaw"] for a in json.loads((set_dir / "scene.json").read_text()).get("assets", [])}
    out = {}
    for kind in ("fixtures", "props"):
        for f in spec.get(kind, []):
            if f.get("outside") or "row" in f:
                continue
            for cls in ("prop", "set"):
                p = library / cls / f["asset"] / "asset.json"
                if p.exists():
                    aj = json.loads(p.read_text())
                    sb = aj.get("seat")
                    if not sb:
                        break
                    fc = f.get("facing", [0, -1])
                    # set frame rotation of the placed asset: the engine yaw it was built with (mirrored), else
                    # the spec's facing (the asset's -Y turned onto it)
                    yaw = -built[f["name"]] if f["name"] in built else _ang(fc) + 90.0
                    fa = sb.get("front_asset", [0.0, -1.0])
                    phi = _ang(fa) + 90.0                    # the seat frame (front -y) inside the asset frame
                    if aj.get("seat_other"):                 # a bed's edge: the side that faces the room
                        so = aj["seat_other"]
                        fr = lambda q: _rot([0.0, -1.0], yaw + _ang(q["front_asset"]) + 90.0)   # noqa: E731
                        to_mid = [-f["position"][0], -f["position"][1]]
                        if sum(a_ * b_ for a_, b_ in zip(fr(so), to_mid)) > sum(a_ * b_ for a_, b_ in zip(fr(sb), to_mid)):
                            sb, fa = so, so["front_asset"]
                            phi = _ang(fa) + 90.0
                    out[f["name"]] = {"pos": f["position"][:2], "asset": f["asset"], "box": sb,
                                      "rot": yaw + phi, "front": _rot([0.0, -1.0], yaw + phi)}
                    break
    return out


def to_world(seat: dict, p):
    q = _rot(p, seat["rot"])
    return [seat["pos"][0] + q[0], seat["pos"][1] + q[1]]


def to_seat(seat: dict, w):
    d = [w[0] - seat["pos"][0], w[1] - seat["pos"][1]]
    return _rot(d, -seat["rot"])


def fit(char_json: dict | None) -> dict:
    f = (char_json or {}).get("seat_fit") or {}
    return {"hip_half_m": float(f.get("hip_half_m", HIP_HALF_M)), "hip_back_m": float(f.get("hip_back_m", HIP_BACK_M))}


def is_seated(c: dict) -> bool:
    p = ((c.get("body_pose") or {}).get("pose") or "")
    return "sit" in p or "seat" in p


SEAT_TURN_DEG = 30.0         # how far a sitter on an open seat turns out of the seat's front toward whom they face


def turn_for(c: dict, seat: dict, chars: dict) -> float:
    """Degrees a sitter turns out of the seat's front toward where the plan has them face (the person they face,
    "facing_to", else the facing the plan gave them when others are in the room), at most SEAT_TURN_DEG either way.
    Only on a seat open at the sides (a bed's edge, a bench): between arm rests, or inside a cabin, the sitter
    stays square. Two people seated square to the room read as a line-up (11-13)."""
    b, t = seat["box"], chars.get(c.get("facing_to") or "")
    if b.get("arm_z_m") or b.get("inside"):
        return 0.0
    if t and t is not c:
        d = [t["position"][0] - c["position"][0], t["position"][1] - c["position"][1]]
    elif len(chars) > 1 and c.get("facing"):
        d = list(c["facing"][:2])
    else:
        return 0.0
    if math.hypot(*d) < 0.2:
        return 0.0
    a = (_ang(d) - _ang(seat["front"]) + 180.0) % 360.0 - 180.0
    return round(max(-SEAT_TURN_DEG, min(SEAT_TURN_DEG, a)), 1)


def place(c: dict, seat: dict, fitd: dict, place_x: float | None = None, nudge=(0.0, 0.0), turn: float = 0.0) -> dict:
    """The character spec with position, facing and seat params from the seat block (nudge: seat-frame metres;
    turn: degrees out of the seat's front, see turn_for)."""
    b = seat["box"]
    xs = b.get("places") or [0.0]
    x = place_x if place_x is not None else min(xs, key=abs)
    y = b["back_y_m"] - fitd["hip_back_m"]
    p = to_world(seat, [x + nudge[0], y + nudge[1]])
    out = dict(c)
    out["position"] = [round(p[0], 3), round(p[1], 3)]
    fr = _rot(seat["front"], turn) if turn else seat["front"]
    out["facing"] = [round(fr[0], 4), round(fr[1], 4)]
    bp = dict(out.get("body_pose") or {})
    pr = dict(bp.get("params") or {})
    pr["seat_z_m"] = b["seat_z_m"]
    if b.get("arm_z_m"):
        half_in = (b["inner_x_m"][1] - b["inner_x_m"][0]) / 2
        pr["arm_z_m"] = b["arm_z_m"]
        pr["arm_x_m"] = round(half_in + 0.05, 3)        # forearms on the arm rests, just outside their inner faces
    bp["params"] = pr
    out["body_pose"] = bp
    out["_seat"] = {"name": seat.get("name"), "x": x, "nudge": list(nudge), "turn": turn}
    if b.get("inside"):                       # seated inside an enclosure (a cabin): the cameras work inside it
        out["_inside"] = True
    return out


def fits(seat: dict, fitd: dict) -> bool:
    b = seat["box"]
    return (b["inner_x_m"][1] - b["inner_x_m"][0]) / 2 + TOL_M >= fitd["hip_half_m"]


def check(measured: dict, seat: dict, fitd: dict, place_x: float) -> dict:
    """From the engine's measured joints: is the person sitting IN the seat."""
    j = measured.get("joints") or {}
    b = seat["box"]
    why = []
    if not all(k in j for k in ("thigh_l", "thigh_r")):
        return {"ok": False, "why": ["no hip joints measured"]}
    hs = [to_seat(seat, j[k][:2]) + [j[k][2]] for k in ("thigh_l", "thigh_r")]
    hx = (hs[0][0] + hs[1][0]) / 2
    hy = (hs[0][1] + hs[1][1]) / 2
    hz = (hs[0][2] + hs[1][2]) / 2
    half = max(abs(hs[0][0] - hs[1][0]) / 2, 0.0)
    lo, hi = b["inner_x_m"]
    spread = max(fitd["hip_half_m"], half + 0.08)
    if hx - spread < lo - TOL_M or hx + spread > hi + TOL_M:
        why.append(f"hips and clothes span {hx - spread:+.2f}..{hx + spread:+.2f} m, the arms' inner faces "
                   f"{lo:+.2f}..{hi:+.2f}")
    if hz < b["seat_z_m"] - 0.02:
        why.append(f"hips {hz:.2f} m, under the cushion ({b['seat_z_m']:.2f})")
    if hz > b["seat_z_m"] + 0.25:
        why.append(f"hips {hz:.2f} m, floating above the cushion ({b['seat_z_m']:.2f})")
    if hy + 0.10 > b["back_y_m"] + TOL_M:
        why.append(f"hips {hy:+.2f} m, into the back ({b['back_y_m']:+.2f})")
    if hy < b["front_y_m"] + 0.12:
        why.append(f"hips {hy:+.2f} m, off the front edge ({b['front_y_m']:+.2f})")
    feet = measured.get("feet_above_floor_m")
    if feet is not None and abs(feet) > FEET_TOL_M:
        why.append(f"feet {feet:+.2f} m off the floor")
    err = [place_x - hx, (b["back_y_m"] - fitd["hip_back_m"]) - hy]
    return {"ok": not why, "why": why, "hips_seat_frame": [round(hx, 3), round(hy, 3), round(hz, 3)],
            "error_m": [round(e, 3) for e in err]}


def same_mesh(library: Path, a: str | None, b: str | None) -> bool:
    """Two prop ids for the same mesh: equal, or one a recolour of the other ("variant_of" in its record)."""
    if not a or not b:
        return False
    base = lambda i: json.loads(p.read_text()).get("variant_of") or i if (p := library / "prop" / i / "asset.json").exists() \
        else i
    return a == b or base(a) == base(b)


def seated_variant(library: Path, base: str, seat: dict) -> str | None:
    """The library character settled for this seat: among "<base>_seated_*", the one whose record names the seat's
    asset ("seat_asset"), else whose settled seat height ("seat_z_m", else "seat N m" in its note) is nearest the
    seat's own (within 3 cm). A dress settled for one seat height and width passes through another chair."""
    import re
    best = (0.031, None)
    for d in sorted((library / "character").glob(f"{base}_seated_*")):
        if "_veil" in d.name[len(base) + 8:]:     # a wardrobe variant of a stance, not a stance (seated_variant
            continue                               # picks the stance; "variant" in the spec adds its suffix)
        p = d / "character.json"
        if not p.exists():
            continue
        cj = json.loads(p.read_text())
        if same_mesh(library, cj.get("seat_asset") or (cj.get("stance") or {}).get("seat_asset"), seat.get("asset")):
            return d.name
        z = cj.get("seat_z_m")
        if z is None:
            m = re.search(r"seat (\d\.\d+) m", cj.get("note") or "")
            z = float(m.group(1)) if m else None
        if z is not None and abs(z - seat["box"]["seat_z_m"]) < best[0]:
            best = (abs(z - seat["box"]["seat_z_m"]), d.name)
    return best[1]


def lap_steps(seat_z_m: float) -> list:
    """Hands resting in the lap, from the character tools' poses_extra.hands_in_lap (SCENES_CHAR_TOOLS)."""
    import importlib.util
    import os
    d = os.environ.get("SCENES_CHAR_TOOLS")
    f = Path(d) / "poses_extra.py" if d else None
    if not f or not f.exists():
        return []
    sp = importlib.util.spec_from_file_location("poses_extra", f)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod.hands_in_lap(seat_z_m)


def plan(spec: dict, library: Path) -> dict:
    """Seat every seated character that names a seat (or seat: true). Returns {key: seat name}; edits spec."""
    set_dir = library / "set" / spec["set"]
    S = seats(library, set_dir)
    for n, s in S.items():
        s["name"] = n
    taken = {}
    for k, c in spec["characters"].items():
        if not c.get("seat") or not is_seated(c) or c.get("asset"):
            continue
        base = c.get("seated_from")             # a stance per chair: pick the one settled for the chosen seat
        cj_p = library / "character" / (c.get("id") or "") / "character.json"
        fitd = fit(json.loads(cj_p.read_text()) if cj_p.exists() else None)
        if c["seat"] is True:
            order = sorted(S, key=lambda n: math.dist(S[n]["pos"], c["position"][:2]))
        else:
            order = [c["seat"]] + sorted((n for n in S if n != c["seat"]),
                                         key=lambda n: math.dist(S[n]["pos"], S[c["seat"]]["pos"]))
        for n in order:
            # a seat takes as many sitters as it has places (a sofa, a bed's edge), each on a place of their own
            places = sorted(S[n]["box"].get("places") or [0.0], key=abs)
            sat = [kk for kk, nn in taken.items() if nn == n]
            if len(sat) >= len(places) or not fits(S[n], fitd):
                continue
            used = [spec["characters"][kk].get("_seat", {}).get("x") for kk in sat]
            px = next(x for x in places if x not in used)
            if base:
                vid = seated_variant(library, base, S[n])
                if not vid:
                    continue                     # no stance of this person settled for that seat
                if c.get("variant"):              # the same stance's wardrobe variant (e.g. "veiled")
                    vv = f"{vid}_{c['variant']}"
                    if (library / "character" / vv / "character.json").exists():
                        vid = vv
                c = dict(c, id=vid)
            spec["characters"][k] = place(c, S[n], fitd, place_x=px)
            spec["characters"][k]["_seat"]["asked_facing"] = list(c.get("facing") or [])[:2]
            pr = spec["characters"][k]["body_pose"]["params"]
            if pr.get("hands") == "lap":
                if not pr.get("action"):     # a start action has the hands (the lap steps ran after it, 11-12)
                    pr["steps"] = lap_steps(pr["seat_z_m"]) + list(pr.get("steps") or [])
                pr.pop("hands")
            spec["characters"][k]["_seat"]["fit"] = fitd
            if n != (c["seat"] if c["seat"] is not True else n):
                spec["characters"][k]["_seat"]["note"] = f"{c['seat']} is narrower than the sitter; {n} fits"
            taken[k] = n
            break
        else:
            raise SystemExit(f"{k}: no seat in the set fits (hips and dress {fitd['hip_half_m'] * 2:.2f} m across)")
    for k, n in taken.items():                   # once everyone has a place: sitters turned to whom they face
        c = spec["characters"][k]
        st = c["_seat"]
        tn = turn_for(dict(c, facing=st.get("asked_facing")), S[n], spec["characters"])
        if tn:
            c0 = {kk: vv for kk, vv in c.items() if kk != "_seat"}
            spec["characters"][k] = place(c0, S[n], st["fit"], st["x"], st["nudge"], tn)
            spec["characters"][k]["_seat"].update({kk: vv for kk, vv in st.items() if kk in ("fit", "note", "asked_facing")})
    return {"taken": taken, "seats": S}
