"""Soft rules for a room layout: small pure cost functions, the camera pocket, and the front check.

Each cost function returns 0 (as wished) to 1 (as bad as it gets) for one instance; ``total`` averages a term's
instances and weighs the terms by ``layout_rules.json``. The forms follow published interior-design cost terms
(face the focal point, conversation distance and angle, reach, clearance, alignment, wall prior, symmetry,
balance); the weights and bands are starting values, not measured on built rooms. Nothing here knows a story.

A *thing* is a placed piece as place_layout makes it: ``{"id", "kind", "c", "facing", "rect", "size_m", "flat",
"seat", ...}`` with ``rect = (x0, y0, x1, y1)`` in plan and ``facing`` a unit vector (where its front points).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

RULES = Path(__file__).resolve().parent / "layout_rules.json"


def load_rules(path=None) -> dict:
    return json.loads(Path(path or RULES).read_text())


# ------------------------------------------------------------------ small geometry
def unit(v) -> tuple:
    n = math.hypot(v[0], v[1])
    return (0.0, -1.0) if n < 1e-9 else (v[0] / n, v[1] / n)


def rgap(a, b) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def area(r) -> float:
    return max(0.0, r[2] - r[0]) * max(0.0, r[3] - r[1])


def inter_area(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def seg_hits_rect(p, q, r) -> bool:
    """Does the segment p-q cross the rectangle r (Liang-Barsky)?"""
    dx, dy = q[0] - p[0], q[1] - p[1]
    t0, t1 = 0.0, 1.0
    for pp, qq in ((-dx, p[0] - r[0]), (dx, r[2] - p[0]), (-dy, p[1] - r[1]), (dy, r[3] - p[1])):
        if abs(pp) < 1e-12:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return t0 <= t1


def inside(p, r, pad=0.0) -> bool:
    return r[0] - pad < p[0] < r[2] + pad and r[1] - pad < p[1] < r[3] + pad


def wall_dist(r, W, D) -> float:
    return max(0.0, min(r[0] + W / 2, W / 2 - r[2], r[1] + D / 2, D / 2 - r[3]))


# ------------------------------------------------------------------ cost terms (each 0..1 for one instance)
def plateau(d: float, m: float, M: float) -> float:
    """1 inside the band [m, M], falling off smoothly outside it (so a search is pulled toward the band)."""
    if d < m:
        return (d / m) ** 2 if m > 1e-9 else 1.0
    if d > M:
        return (M / d) ** 2 if d > 1e-9 else 1.0
    return 1.0


def c_face(facing, c, target) -> float:
    """A front that points at the target costs 0, away from it 1."""
    d = unit((target[0] - c[0], target[1] - c[1]))
    return (1.0 - (facing[0] * d[0] + facing[1] * d[1])) / 2.0


def c_conversation_distance(d: float, band) -> float:
    return 1.0 - plateau(d, band[0], band[1])


def c_conversation_angle(fa, ca, fb, cb) -> float:
    """Two seats turned toward each other cost 0; back to back 1."""
    ab = unit((cb[0] - ca[0], cb[1] - ca[1]))
    cos_a = fa[0] * ab[0] + fa[1] * ab[1]
    cos_b = -(fb[0] * ab[0] + fb[1] * ab[1])
    return 1.0 - (cos_a + 1.0) * (cos_b + 1.0) / 4.0


def c_reach(gap_m: float, band) -> float:
    return 1.0 - plateau(gap_m, band[0], band[1])


def c_where(piece_c, target_c, target_facing, where: str) -> float:
    """Is a companion on the wanted side of its target: at its side, before its front, or at its head end
    (the end its front points away from)? 0 yes, 1 no."""
    v = (piece_c[0] - target_c[0], piece_c[1] - target_c[1])
    along = v[0] * target_facing[0] + v[1] * target_facing[1]
    side = abs(v[0] * target_facing[1] - v[1] * target_facing[0])
    if where == "side":
        return 0.0 if side >= abs(along) else 1.0
    if where == "front":
        return 0.0 if along > 0 and along >= side else 1.0
    if where == "head":
        return 0.0 if along <= 0 else 1.0
    return 0.0


def c_clearance(zone, others) -> float:
    """Share of a piece's comfortable use-space taken by other pieces."""
    a = area(zone)
    return 0.0 if a < 1e-9 else min(1.0, sum(inter_area(zone, o) for o in others) / a)


def c_align(angle_rad: float) -> float:
    """Parallel and perpendicular both count as aligned (0); 45 degrees off costs 1."""
    return (1.0 - math.cos(4.0 * angle_rad)) / 2.0


def c_wall_prior(dist_m: float, band) -> float:
    return 1.0 - plateau(dist_m, band[0], band[1])


def mirror(p, axis_pt, axis_dir) -> tuple:
    v = (p[0] - axis_pt[0], p[1] - axis_pt[1])
    along = v[0] * axis_dir[0] + v[1] * axis_dir[1]
    foot = (axis_pt[0] + axis_dir[0] * along, axis_pt[1] + axis_dir[1] * along)
    return (2 * foot[0] - p[0], 2 * foot[1] - p[1])


def c_symmetry(c, like_centres, axis_pt, axis_dir, scale_m: float) -> float:
    """A piece with a like piece mirrored across the focal axis (or standing on the axis itself) costs 0."""
    m = mirror(c, axis_pt, axis_dir)
    ds = [math.hypot(m[0] - c[0], m[1] - c[1]) / 2.0]          # its own distance from the axis
    ds += [math.hypot(m[0] - o[0], m[1] - o[1]) for o in like_centres]
    return 1.0 - math.exp(-min(ds) / scale_m)


def c_balance(rects, W: float, D: float) -> float:
    """The furniture's area-weighted middle near the room's middle costs 0."""
    a = sum(area(r) for r in rects)
    if a < 1e-9:
        return 0.0
    cx = sum(area(r) * (r[0] + r[2]) / 2 for r in rects) / a
    cy = sum(area(r) * (r[1] + r[3]) / 2 for r in rects) / a
    return min(1.0, math.hypot(cx, cy) / (0.25 * math.hypot(W, D)))


def c_empty_middle(rects, W: float, D: float, em: dict) -> float:
    """Everything on the walls and nothing in the middle of the floor costs 1."""
    if len(rects) < 3:
        return 0.0
    hug = sum(1 for r in rects if wall_dist(r, W, D) < em["wall_touch_m"]) / len(rects)
    s = em["middle_share"]
    mid = (-W * s / 2, -D * s / 2, W * s / 2, D * s / 2)
    empty = 0.0 if any(inter_area(r, mid) > 1e-9 for r in rects) else 1.0
    over = max(0.0, hug - em["hug_allowed"]) / max(1e-9, 1.0 - em["hug_allowed"])
    return 0.5 * over + 0.5 * empty


def c_bed_door(d: float, W: float, D: float) -> float:
    return 1.0 - min(1.0, d / math.hypot(W, D))


# ------------------------------------------------------------------ groups and the focal point
def solid(placed: dict) -> list:
    return [q for q in placed.values() if not q.get("flat") and not q.get("on")]


def seating(placed: dict, R: dict) -> list:
    ex = set(R["groups"]["seating"]["exclude_kinds"])
    return [q for q in solid(placed) if q.get("seat") and q["kind"] not in ex and q["kind"] != "fireplace"]


def focal_point(ctx, placed: dict, R: dict):
    """(point, axis direction into the room) of what the room looks at: the focal opening, else the main window,
    else the middle of the seats; None when there is nothing to face."""
    if ctx.focal:
        return tuple(ctx.focal["point"]), tuple(ctx.focal["normal"])
    wins = [o for o in ctx.openings if o["kind"] == "window"]
    if wins:
        return tuple(wins[0]["point"]), tuple(wins[0]["normal"])
    seats = seating(placed, R)
    if len(seats) >= 2:
        c = (sum(q["c"][0] for q in seats) / len(seats), sum(q["c"][1] for q in seats) / len(seats))
        return c, None
    return None


def is_kind(q: dict, want: str, R: dict) -> bool:
    if want == "seat":
        return bool(q.get("seat")) and q["kind"] not in R["groups"]["seating"]["exclude_kinds"]
    if want == "desk":
        return q["kind"] in R["desk_kinds"]
    return q["kind"] == want


def front_zone(q: dict, depth: float) -> tuple:
    f, r = q["facing"], q["rect"]
    hx, hy = (r[2] - r[0]) / 2, (r[3] - r[1]) / 2
    ext = abs(f[0]) * hx + abs(f[1]) * hy
    zc = (q["c"][0] + f[0] * (ext + depth / 2), q["c"][1] + f[1] * (ext + depth / 2))
    zh = (depth / 2, hy) if abs(f[0]) > abs(f[1]) else (hx, depth / 2)
    return (zc[0] - zh[0], zc[1] - zh[1], zc[0] + zh[0], zc[1] + zh[1])


def comfort_depth(q: dict, rule: dict, R: dict) -> float:
    cf = R["comfort_front_m"]
    if rule.get("front_clear_m", 0.0) <= 0:
        return 0.0
    if q.get("seat"):
        return cf["seat"]
    if q["kind"] in R["storage_kinds"]:
        return cf["storage"]
    return cf["default"]


# ------------------------------------------------------------------ the camera pocket
def persons(ctx, placed: dict, R: dict, stand_point) -> list:
    """One person per named place: where their face is, which way it looks, how high, and the piece they are on."""
    cam, out, seen = R["camera"], [], set()
    for phrase, i in (ctx.spec.get("named_places") or {}).items():
        if i in seen:
            continue
        host, pos, facing, z, own = None, None, None, cam["face_z_m"]["standing"], None
        at_desk = False
        if i in placed and not placed[i].get("flat"):
            host = placed[i]
            if host.get("on"):
                host = next((q for q in placed.values() if q.get("name") == host["on"]), host)
            sup = ctx.data["piece_kinds"].get(host["kind"], {}).get("support")
            if host["kind"] in R["desk_kinds"]:   # a person at a desk with a chair sits in the chair
                chairs = [q for q in placed.values() if q.get("seat") and not q.get("on")
                          and q["kind"] in R["groups"]["seating"]["exclude_kinds"]
                          and rgap(q["rect"], host["rect"]) <= 0.35]
                if chairs:
                    # (a desk the text puts at a wall or a window keeps the old reading: the sitter turns from it)
                    at_desk = ctx.mode(host["id"])[0] != "wall"
                    host = min(chairs, key=lambda q: (rgap(q["rect"], host["rect"]), q["id"]))
            if at_desk and host.get("seat"):
                # the plan names the desk as the person's place: they sit at it, facing it, as the staging seats
                # them; the camera's pocket is across the desk (a desk against a wall leaves none: 11-12)
                pos, facing, z, own = host["c"], host["facing"], cam["face_z_m"]["seated"], host["id"]
            elif host.get("seat") and host["kind"] in R["groups"]["seating"]["exclude_kinds"]:
                # on a desk chair or a stool a person turns to whoever they talk to, away from the desk
                pos, facing, z, own = host["c"], None, cam["face_z_m"]["seated"], host["id"]
            elif host.get("seat"):
                pos, facing, z, own = host["c"], host["facing"], cam["face_z_m"]["seated"], host["id"]
            elif sup == "lie":
                f = host["facing"]
                L = abs(f[0]) * (host["rect"][2] - host["rect"][0]) + abs(f[1]) * (host["rect"][3] - host["rect"][1])
                pos = (host["c"][0] - f[0] * 0.3 * L, host["c"][1] - f[1] * 0.3 * L)
                facing, z, own = f, cam["face_z_m"]["lying"], host["id"]
            else:
                pos = stand_point(host, ctx.data)
        elif i in ctx.by_id:
            o = ctx.by_id[i]
            if o["kind"] == "fireplace" and i in placed:
                pos = stand_point(placed[i], ctx.data)
            else:
                pos = stand_point({"opening_kind": o["kind"], "c": o["point"], "facing": o["normal"]}, ctx.data)
        if pos is None:
            continue
        seen.add(i)
        out.append({"place": phrase, "id": i, "name": ctx.names.get(i, i), "pos": (pos[0], pos[1]),
                    "facing": facing, "z": z, "own": own})
    fp = focal_point(ctx, placed, R)
    for p in out:                                   # a standing person may turn to anyone in the room: the nearest
        if p["facing"] is None:                     # first, then the others, the focal point, the room's middle
            others = sorted((math.hypot(q["pos"][0] - p["pos"][0], q["pos"][1] - p["pos"][1]), k)
                            for k, q in enumerate(out) if q is not p)
            tg = [out[k]["pos"] for d, k in others if d > 0.3]
            tg += ([fp[0]] if fp else []) + [(0.0, 0.0)]
            p["turns"] = [unit((t[0] - p["pos"][0], t[1] - p["pos"][1])) for t in tg
                          if math.hypot(t[0] - p["pos"][0], t[1] - p["pos"][1]) > 0.3] or [(0.0, -1.0)]
            p["facing"] = p["turns"][0]
    return out


def pocket(person: dict, blockers: list, W: float, D: float, cam: dict) -> dict:
    """Camera positions in front of a face. ``blockers`` are ``(rect, height_m, id)``. A sample is valid when the
    lens is inside the room off the walls, not inside a piece as tall as the lens, and its line to the face clears
    every piece that stands higher than that line (the person's own seat aside)."""
    f, p = person["facing"], person["pos"]
    base = math.atan2(f[1], f[0])
    n, half = cam["angle_steps"], math.radians(cam["head_deg"])
    sight = min(person["z"], cam["lens_z_m"]) - cam["sight_under_m"]
    m = cam["wall_margin_m"]
    res = {}
    for name, dists in cam["framings"].items():
        valid, total, pts = 0, 0, []
        for d in dists:
            for k in range(n):
                a = base + (-half + 2 * half * k / (n - 1) if n > 1 else 0.0)
                s = (p[0] + d * math.cos(a), p[1] + d * math.sin(a))
                total += 1
                if not (-W / 2 + m <= s[0] <= W / 2 - m and -D / 2 + m <= s[1] <= D / 2 - m):
                    continue
                bad = False
                for r, h, bid in blockers:
                    if h > cam["lens_z_m"] - 0.2 and inside(s, r, 0.1):
                        bad = True
                        break
                    if bid != person["own"] and h > sight and seg_hits_rect(s, p, r):
                        bad = True
                        break
                if not bad:
                    valid += 1
                    pts.append((round(s[0], 3), round(s[1], 3)))
        res[name] = {"valid": valid, "total": total, "points": pts}
    return res


def blockers_of(ctx, placed: dict) -> list:
    out = [(q["rect"], float(q["size_m"][2]), q["id"]) for q in solid(placed)]
    out += [(o["breast"], 9.0, "breast") for o in ctx.openings if o["kind"] == "fireplace"]
    return out


def pockets(ctx, placed: dict, R: dict, stand_point) -> list:
    bl = blockers_of(ctx, placed)
    out = []
    for p in persons(ctx, placed, R, stand_point):
        best = None
        for f in p.pop("turns", None) or [p["facing"]]:
            pk = pocket(dict(p, facing=f), bl, ctx.W, ctx.D, R["camera"])
            key = (min(r["valid"] > 0 for r in pk.values()), sum(r["valid"] for r in pk.values()))
            if best is None or key > best[0]:
                best = (key, f, pk)
        out.append(dict(p, facing=best[1], pocket=best[2]))
    return out


def pocket_problems(pks: list) -> list:
    out = []
    for p in pks:
        for name, r in p["pocket"].items():
            if r["valid"] == 0:
                out.append(f"no camera position for a {name} shot of a person at {p['name']} ({p['place']!r})")
    return out


# ------------------------------------------------------------------ the whole cost
def total(ctx, placed: dict, R: dict, rel_ok=None, stand_point=None, with_pockets=True) -> tuple:
    """-> (weighted cost, {term: mean cost}, {piece id: {term: its weighted share}})."""
    W, D, wts = ctx.W, ctx.D, R["weights"]
    inst = []                                        # (term, value, ids)
    sol = solid(placed)
    seats = seating(placed, R)
    fp = focal_point(ctx, placed, R)
    if fp:
        for q in seats:
            inst.append(("face_focal", c_face(q["facing"], q["c"], fp[0]), [q["id"]]))
    band = R["conversation"]["distance_m"]
    for i, a in enumerate(seats):
        for b in seats[i + 1:]:
            d = math.hypot(a["c"][0] - b["c"][0], a["c"][1] - b["c"][1])
            inst.append(("conversation_distance", c_conversation_distance(d, band), [a["id"], b["id"]]))
            inst.append(("conversation_angle", c_conversation_angle(a["facing"], a["c"], b["facing"], b["c"]),
                         [a["id"], b["id"]]))
            da = abs(math.atan2(a["facing"][1], a["facing"][0]) - math.atan2(b["facing"][1], b["facing"][0]))
            inst.append(("align_group", c_align(da), [a["id"], b["id"]]))
    anchored = {}
    for rr in R["reach"]:
        tg = [t for t in sol if is_kind(t, rr["to"], R)]
        if not tg:
            continue
        for q in placed.values():
            if q["kind"] != rr["kind"] or q.get("flat"):
                continue
            host = q
            if q.get("on"):
                continue
            t = min(tg, key=lambda t: (rgap(host["rect"], t["rect"]), t["id"]))
            v = 0.5 * c_reach(rgap(host["rect"], t["rect"]), rr["gap_m"]) \
                + 0.5 * c_where(host["c"], t["c"], t["facing"], rr["where"])
            inst.append(("reach", v, [q["id"], t["id"]]))
            anchored.setdefault((t["id"], q["kind"]), []).append((q, t))
    for (tid, kind), lst in anchored.items():        # two like companions of one anchor stand either side of it
        if len(lst) >= 2:
            t = lst[0][1]
            sides = [(q["c"][0] - t["c"][0]) * t["facing"][1] - (q["c"][1] - t["c"][1]) * t["facing"][0]
                     for q, _ in lst]
            same = all(s > 0 for s in sides) or all(s < 0 for s in sides)
            inst.append(("pair_sides", 1.0 if same else 0.0, [q["id"] for q, _ in lst]))
    for q in sol:
        if q["kind"] == "fireplace" or q["id"] not in ctx.pieces:
            continue
        rule = ctx.rule(q["id"])
        dep = comfort_depth(q, rule, R)
        if dep > 0:
            z = front_zone(q, dep)
            mates = {b["id"] for (tid, _k), lst in anchored.items() for b, t in lst if tid == q["id"]}
            mates |= {t["id"] for (_tid, _k), lst in anchored.items() for b, t in lst if b["id"] == q["id"]}
            others = [o["rect"] for o in sol if o["id"] != q["id"] and o["id"] not in mates]
            out_room = area(z) - inter_area(z, (-W / 2, -D / 2, W / 2, D / 2))
            v = min(1.0, c_clearance(z, others) + (out_room / area(z) if area(z) > 0 else 0.0))
            inst.append(("clearance", v, [q["id"]]))
        if not q.get("wall"):
            ang = math.atan2(q["facing"][1], q["facing"][0])
            inst.append(("align_wall", c_align(ang), [q["id"]]))
        wp = R["wall_prior"].get(q["kind"])
        if wp and not q.get("wall"):
            inst.append(("wall_prior", c_wall_prior(wall_dist(q["rect"], W, D), wp["dist_m"]), [q["id"]]))
    if fp and fp[1] and len(seats) >= 2:
        for q in seats:
            like = [o["c"] for o in seats if o is not q]
            inst.append(("symmetry", c_symmetry(q["c"], like, fp[0], fp[1], R["symmetry"]["scale_m"]), [q["id"]]))
    body = [q["rect"] for q in sol if q["kind"] != "fireplace"]
    if len(body) >= 2:
        inst.append(("balance", c_balance(body, W, D), []))
    em = R["empty_middle"]
    counted = [q["rect"] for q in sol if q["kind"] != "fireplace"
               and (q["kind"] in em["kinds_counted"] or (q.get("seat") and "seat" in em["kinds_counted"]))]
    if counted:
        inst.append(("empty_middle", c_empty_middle(counted, W, D, em), []))
    room = (-W / 2, -D / 2, W / 2, D / 2)
    for q in sol:                                    # a bed is got into from both long sides when the room allows
        if q["kind"] == "bed":
            f, r, dep = q["facing"], q["rect"], R["comfort_front_m"]["bed_side"]
            strips = ([(r[0], r[1] - dep, r[2], r[1]), (r[0], r[3], r[2], r[3] + dep)] if abs(f[0]) > abs(f[1])
                      else [(r[0] - dep, r[1], r[0], r[3]), (r[2], r[1], r[2] + dep, r[3])])
            mates = [o["rect"] for o in sol if o["kind"] in ("bedside_table",)]
            others = [o["rect"] for o in sol if o["id"] != q["id"] and o["kind"] != "bedside_table"]
            for z in strips:
                lost = (area(z) - inter_area(z, room)) + sum(inter_area(z, o) for o in others)
                inst.append(("bed_sides", min(1.0, lost / area(z)) if area(z) > 0 else 0.0, [q["id"]]))
            del mates
    doors = [o for o in ctx.openings if o["kind"] == "door"]
    for q in sol:
        if q["kind"] == "bed" and doors:
            d = min(math.hypot(q["c"][0] - o["point"][0], q["c"][1] - o["point"][1]) for o in doors)
            inst.append(("bed_door", c_bed_door(d, W, D), [q["id"]]))
    if rel_ok is not None:
        for r in getattr(ctx, "inferred", []):
            ok = rel_ok(ctx, r, placed)
            if ok is not None:
                inst.append(("inferred", 0.0 if ok else 1.0, [x for x in (r.get("a"), r.get("b")) if x in placed]))
    if with_pockets and stand_point is not None:
        for p in pockets(ctx, placed, R, stand_point):
            vals = [1.0 - r["valid"] / r["total"] if r["total"] else 0.0 for r in p["pocket"].values()]
            none = any(r["valid"] == 0 for r in p["pocket"].values())
            inst.append(("camera_pocket", 1.0 if none else sum(vals) / len(vals), [p["id"]] if p["id"] in placed else []))
    by_term, counts = {}, {}
    for t, v, _ in inst:
        by_term[t] = by_term.get(t, 0.0) + v
        counts[t] = counts.get(t, 0) + 1
    by_piece = {}
    for t, v, ids in inst:
        for i in ids:
            d = by_piece.setdefault(i, {})
            d[t] = d.get(t, 0.0) + wts.get(t, 1.0) * v / counts[t]
    means = {t: by_term[t] / counts[t] for t in by_term}
    cost = sum(wts.get(t, 1.0) * m for t, m in means.items())
    return cost, {t: round(m, 4) for t, m in means.items()}, \
        {i: {t: round(v, 4) for t, v in d.items()} for i, d in by_piece.items()}


# ------------------------------------------------------------------ fronts of library records
def front_problems(record: dict, rule: dict, R: dict) -> list:
    """Why a library record's front cannot be trusted for layout: every relation is in the piece's own frame, so
    a wrong front turns a whole group. The library's contract is a front along -Y as modelled; a seat's measured
    seat box says where the open side really is."""
    out = []
    seat = record.get("seat") or {}
    fa = seat.get("front_asset")
    if fa:
        if -float(fa[1]) < R["front"]["seat_front_cos"]:
            out.append(f"the seat's open side is {list(fa)} in the mesh's own frame, not -Y")
    elif rule.get("seat"):
        out.append("a seat with no measured seat box: its front is not verified")
    elif rule.get("front_clear_m", 0.0) > 0 or rule.get("face"):
        if record.get("front") is None:
            out.append("front not recorded: assumed -Y as modelled, not verified")
    size = record.get("size_m")
    if not size or len(size) < 3 or min(size) <= 0:
        out.append("no usable size_m")
    return out
