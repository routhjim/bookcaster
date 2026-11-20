"""Lay out a room in plan view from a place spec (stdlib only; no engine, no model).

    python tools/assets/sets/place_layout.py PLACE_SPEC.json [RECORDS.json] [--seed N] [--out LAYOUT.json]

Input: a validated place spec (schema ``place_spec/1``: kind, size class, openings with a wall or "not stated",
pieces with kind / look / phrases / against, relations between ids, named places) and, per piece or fireplace
id, the matched library record (``{"asset": id, "size_m": [w, d, h], "seat": {...}}``). A piece without a record
takes its kind's default size from ``room_kinds.json`` (it will be generated).

Output (``solve``): the room's size, every opening's wall and span, and a position and facing for every piece, in
the set builder's frame (SETS.md: metres, X east, Y north, origin at the middle of the floor, a library asset's
front is -Y as modelled, ``facing`` is where that front points). ``to_set_spec`` turns it into the walls, the
chimney breast and the fixtures / props of a set spec; ``named_places`` maps each plan phrase to a set item name.

Order of work: openings on walls first (a fireplace in the middle of its wall, windows on an outer wall, a door
clear of both), then pieces: against a wall or free by their kind's rule in the data file, stated relations as hard
constraints, clearances (a seat's front, a door's swing and approach, walkways) kept free. The search is a beam
over enumerated candidates in a fixed order, scored by the soft rules of ``place_cost.py`` and then polished
(a depth-first placement is the fallback), with the seed only breaking ties; it either meets every hard
constraint (``ok: true``) or returns the best layout it has with ``unmet`` naming each constraint that could not
be met. ``solve_recover`` adds the recovery ladder. Nothing here knows a story: every rule about a kind of room
or piece is data (``room_kinds.json``, ``layout_rules.json``).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import place_cost as pc  # noqa: E402

DATA = Path(__file__).resolve().parent / "room_kinds.json"
NORMAL = {"north": (0.0, -1.0), "south": (0.0, 1.0), "east": (-1.0, 0.0), "west": (1.0, 0.0)}
WALLS = ("north", "west", "east", "south")
NOT_STATED = (None, "", "not stated")
DEFAULT_SIZE = [0.6, 0.6, 0.8]


def load_data(path=None) -> dict:
    d = json.loads(Path(path or DATA).read_text())
    # a kind of room with no rules of its own is laid out as the kind it is most like ("room_kinds_like": a dining
    # room as a sitting room): the director may name any kind, and a set is built for every one
    for k, base in (d.get("room_kinds_like") or {}).items():
        if k not in d["room_kinds"] and base in d["room_kinds"]:
            d["room_kinds"][k] = d["room_kinds"][base]
    return d


# a kind the layout has no rule for stands where its piece record says (places/piece_kinds.json "placement"):
# a small thing that belongs on a surface is placed on one, not on the floor against a wall
try:
    PK = json.loads((Path(__file__).resolve().parents[1] / "places" / "piece_kinds.json").read_text())["kinds"]
    SURFACE_KINDS = {k for k, v in PK.items() if v.get("placement") == "surface"}
    # a thing a person sits or lies on (the kind's "needs") is not a surface to stand small things on
    PERSON_KINDS = {k for k, v in PK.items() if v.get("needs")}
except (OSError, KeyError, ValueError):
    PK = {}
    SURFACE_KINDS = PERSON_KINDS = set()


# ------------------------------------------------------------------ geometry
def wall_pt(side: str, u: float, W: float, D: float) -> tuple:
    return {"north": (u, D / 2), "south": (u, -D / 2), "east": (W / 2, u), "west": (-W / 2, u)}[side]


def wall_len(side: str, W: float, D: float) -> float:
    return W if side in ("north", "south") else D


def unit(v) -> tuple:
    n = math.hypot(v[0], v[1])
    return (0.0, -1.0) if n < 1e-9 else (v[0] / n, v[1] / n)


def half_extents(size, facing) -> tuple:
    """Half extents in plan of a w x d footprint whose front (-Y as modelled) points along ``facing``: the
    axis-aligned box of the turned rectangle (what the set builder's own plan check uses)."""
    a = math.atan2(facing[1], facing[0]) + math.pi / 2
    c, s = abs(math.cos(a)), abs(math.sin(a))
    return (size[0] * c + size[1] * s) / 2, (size[0] * s + size[1] * c) / 2


def rect_at(c, h) -> tuple:
    return (c[0] - h[0], c[1] - h[1], c[0] + h[0], c[1] + h[1])


def overlap(a, b, gap=0.0) -> bool:
    return min(a[2], b[2]) - max(a[0], b[0]) > -gap + 1e-9 and min(a[3], b[3]) - max(a[1], b[1]) > -gap + 1e-9


def gap(a, b) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    return math.hypot(dx, dy)


def centre(r) -> tuple:
    return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)


def r3(v):
    return [round(float(x), 3) for x in v]


# ------------------------------------------------------------------ openings
def breast_rect(o: dict, od: dict, W: float, D: float) -> tuple:
    """The chimney breast of a fireplace: a block on its wall, wider than the fireplace by the side margin."""
    hw, d = o["width_m"] / 2 + od["breast_side_m"], od["breast_depth_m"]
    return span_rect(o["wall"], o["at_m"], hw, d, W, D)


def span_rect(side: str, u: float, hw: float, depth: float, W: float, D: float) -> tuple:
    """A rectangle standing on a wall: half width ``hw`` along it around ``u``, ``depth`` into the room."""
    n = NORMAL[side]
    p = wall_pt(side, u, W, D)
    c = (p[0] + n[0] * depth / 2, p[1] + n[1] * depth / 2)
    h = (hw, depth / 2) if side in ("north", "south") else (depth / 2, hw)
    return rect_at(c, h)


def place_openings(spec: dict, records: dict, data: dict, W: float, D: float, rules: dict | None = None) -> tuple:
    rk = data["room_kinds"][spec["kind"]]
    margin = data["opening_margin_m"]
    outer = list(spec.get("outer_walls") or rk.get("outer_walls") or [])
    defaults = [dict(o) for o in rk.get("openings", [])]
    done, unmet, counts = [], [], {}
    # a room the text gives no door or window still has its kind's own (a sealed, unlit box otherwise): the kind's
    # default openings of a kind the spec states none of are added, on the kind's wall
    have = {o["kind"] for o in spec.get("openings", [])}
    kind_own = [{"id": f"kind_{d['kind']}", "kind": d["kind"], "wall": "not stated", "words": "", "from_kind": True}
                for d in defaults if d["kind"] in ("door", "window") and d["kind"] not in have]
    order = sorted(list(spec.get("openings", [])) + kind_own,
                   key=lambda o: ("fireplace", "window", "door").index(o["kind"]))

    rules = rules or pc.load_rules()
    dress = float((rules["hard"].get("window_dressing") or {}).get("width_m", 0.0))

    def clear(kind, side, u, w, od):
        L = wall_len(side, W, D)
        reach_w = max(w, dress) if kind == "window" else w
        if abs(u) + reach_w / 2 > L / 2 - 0.1:
            return False
        for q in done:
            qw = q["width_m"] / 2 + (data["openings"]["fireplace"]["breast_side_m"] if q["kind"] == "fireplace" else 0)
            if q["wall"] == side and abs(q["at_m"] - u) < qw + w / 2 + margin:
                return False
        mine = span_rect(side, u, w / 2, w if kind == "door" else 0.05, W, D)
        for q in done:                              # a door's swing stays off a chimney breast and its hearth
            if q["kind"] == "fireplace" and (kind == "door" or q["wall"] != side):
                fo = data["openings"]["fireplace"]
                if overlap(mine, span_rect(q["wall"], q["at_m"], q["width_m"] / 2 + fo["breast_side_m"],
                                           fo["breast_depth_m"] + fo["hearth_depth_m"], W, D)):
                    return False
            if q["kind"] == "door" and q["wall"] != side and overlap(mine, q["swing"]):
                return False
        return True

    for o in order:
        kind = o["kind"]
        od = data["openings"][kind]
        rec = records.get(o["id"]) or {}
        w = float((rec.get("size_m") or od.get("size_m") or [od["width_m"]])[0]) if kind == "fireplace" \
            else float(od["width_m"])
        stated = o.get("wall") not in NOT_STATED
        if stated:
            walls = [o["wall"]]
        else:
            walls = [d["wall"] for d in defaults if d["kind"] == kind and not d.get("used")]
            if kind == "window":
                walls += outer
            fire_walls = {q["wall"] for q in done if q["kind"] == "fireplace"}
            walls += [s for s in WALLS if not (kind == "door" and s in fire_walls)]
            walls += list(WALLS)
            walls = list(dict.fromkeys(walls))
        found = None
        for side in walls:
            L = wall_len(side, W, D)
            if kind == "fireplace":
                us = [0.0]
            elif kind == "window":
                us = sorted({round(k * 0.25, 2) for k in range(-40, 41)}, key=lambda u: (abs(u), u))
            else:
                end = L / 2 - od["corner_m"] - w / 2
                us = [-end, end] + sorted({round(k * 0.25, 2) for k in range(-40, 41)}, key=lambda u: (-abs(u), u))
            for u in us:
                if clear(kind, side, u, w, od):
                    found = (side, u)
                    break
            if found:
                break
        if not found:
            where = f"the {o['wall']} wall" if stated else "any wall"
            unmet.append(f"opening {o['id']} ({kind}): no clear position on {where}")
            continue
        side, u = found
        for d in defaults:
            if d["kind"] == kind and d["wall"] == side and not d.get("used"):
                d["used"] = True
                break
        counts[kind] = counts.get(kind, 0) + 1
        n = NORMAL[side]
        p = wall_pt(side, u, W, D)
        rec_o = {"id": o["id"], "kind": kind, "wall": side, "at_m": round(u, 3), "width_m": round(w, 3),
                 "from_m": round(u - w / 2, 3), "to_m": round(u + w / 2, 3), "normal": list(n), "point": r3(p),
                 "name": kind if counts[kind] == 1 else f"{kind}_{counts[kind]}", "stated": stated,
                 "rect": span_rect(side, u, w / 2, 0.05, W, D)}
        if kind == "door":
            rec_o["swing"] = span_rect(side, u, w / 2, w, W, D)
        if kind in ("window", "door"):
            rec_o["sill_m"], rec_o["head_m"] = od["sill_m"], od["head_m"]
        if kind == "fireplace":
            rec_o["breast"] = breast_rect(rec_o, od, W, D)
        done.append(rec_o)
    return done, unmet


# ------------------------------------------------------------------ the picture's pieces, fire pieces
PICTURE_ID = re.compile(r"^c\d+$")       # a piece place_build.with_concept took from the place's concept picture
PICTURE_RULES = {"seat_keep_m": 2.5, "named_keep_m": 1.0, "keep_side_m": 0.5, "repeat_default": 4, "repeat_most": 8,
                 "row_gap_m": 1.0, "double_m": 1.0, "double_k": 0.6, "raised_most_m": 0.5, "wall_depth_k": 2.0}
FIRE_RULES = {"front_clear_m": 1.0, "companion_rels": ["beside", "by"], "work_top_m": [0.35, 1.3],
              "work_front_m": [1.8, 2.5], "work_side_m": 0.6, "work_clear_m": 1.5, "work_clear_sides": 3,
              "relaxed": {"work_front_m": [1.2, 3.0], "work_side_m": 1.2, "work_clear_m": 0.9, "work_clear_sides": 3}}
NUMBER_WORDS = {"two": 2, "pair": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "twelve": 12}
MANY_WORDS = {"rows", "row", "several", "many", "lines", "ranks"}
_NAME_STOP = set("a an the of in on at by to with and or for from its their one some very old new small large long "
                 "short tall low high big little wide narrow heavy light dark pale rough smooth worn plain simple "
                 "wood wooden wood iron metal stone glass fabric cloth leather brown grey gray black white red dark "
                 "green blue gold golden silver yellow".split())


def picture_rules(R: dict) -> dict:
    return dict(PICTURE_RULES, **(R.get("picture") or {}))


def fire_rules(R: dict) -> dict:
    return dict(FIRE_RULES, **(R.get("fire") or {}))


def _root(kind: str) -> str:
    seen = set()
    while (PK.get(kind) or {}).get("parent") and kind not in seen:
        seen.add(kind)
        kind = PK[kind]["parent"]
    return kind


def _words(s) -> list:
    return [t for t in re.sub(r"[^a-z0-9 ]+", " ", str(s or "").lower().replace("-", " ")).split() if t]


def _name_of(p: dict) -> list:
    """The words a piece is called by: what the story or the picture's judge calls it, else the first clause of its
    look (with_concept writes a picture piece's look as "<its name>; <material, colour>")."""
    head = p.get("called") or re.split(r"[;,]", str(p.get("look") or ""), maxsplit=1)[0]
    return [t for t in _words(head) if t not in _NAME_STOP]


def _stem(t: str) -> str:
    return t[:-2] if t.endswith("es") and len(t) > 4 else t[:-1] if t.endswith("s") and len(t) > 3 else t


def _kind_words(kind: str) -> set:
    d = PK.get(kind) or {}
    return {kind.replace("_", " ")} | set(d.get("synonyms") or [])


def plural_count(p: dict, R: dict) -> int:
    """How many of a picture piece the picture shows, from its name: a number word ("three tables"), a word of many
    ("rows of benches"), or its kind's own word in the plural ("tables", "benches") -> that many (the default when
    no number is said), else 1. Only a thing of the floor repeats."""
    d = PK.get(p.get("kind")) or {}
    if d.get("placement") != "floor" or d.get("made_as") or d.get("sized_by") or p.get("kind") == "rug":
        return 1
    toks = _words(re.split(r";", str(p.get("look") or ""), maxsplit=1)[0])
    pr = picture_rules(R)
    for t in toks:
        if t in NUMBER_WORDS:
            return min(int(pr["repeat_most"]), NUMBER_WORDS[t])
        if t.isdigit() and 1 < int(t) <= 99:
            return min(int(pr["repeat_most"]), int(t))
    names = [_words(w) for w in _kind_words(p.get("kind"))]
    for i, t in enumerate(toks):
        for nw in names:
            n = len(nw)
            if n and i + n <= len(toks) and toks[i:i + n - 1] == nw[:-1] and toks[i + n - 1] in (nw[-1] + "s", nw[-1] + "es"):
                return int(pr["repeat_default"])
    if MANY_WORDS & set(toks):
        return int(pr["repeat_default"])
    return 1


def stated_ids(spec: dict) -> set:
    tied = set()
    for r in spec.get("relations") or []:
        if not (r.get("inferred") or r.get("stated") is False):
            tied.update(x for x in (r.get("a"), r.get("b"), r.get("c")) if x)
    return tied


def is_picture_piece(p: dict, spec: dict) -> bool:
    """A piece the place's picture adds that the text never names: no phrase, no named place, no stated relation."""
    return bool(p.get("_picture") or (PICTURE_ID.match(str(p.get("id") or "")) and not p.get("phrases")
                                       and p["id"] not in set((spec.get("named_places") or {}).values())
                                       and p["id"] not in stated_ids(spec)))


def same_thing(c: dict, p: dict, size_c=None, size_p=None, R: dict | None = None) -> bool:
    """A picture piece that is the spec's piece seen again: the same kind (or the spec kind's own word ends the
    picture's name for it, a "workbench" for a bench), or, for a kind that holds any thing (a work piece, a
    landmark), a name they share; and where the spec piece stands, or the spec piece is not placed by the picture."""
    kc, kp = c.get("kind"), p.get("kind")
    any_thing = (PK.get(kp) or {}).get("sized_by") == "longest" or (PK.get(kc) or {}).get("sized_by") == "longest"
    nc, np_ = _name_of(c), _name_of(p)
    if kc == kp or (_root(kc) == _root(kp) and not any_thing):
        if any_thing and not ({_stem(t) for t in nc} & {_stem(t) for t in np_}):
            return False
    else:
        head = _stem(nc[-1]) if nc else ""
        if not head or not any(head.endswith(_stem(w.split()[-1])) for w in _kind_words(kp)):
            return False
    if not p.get("at_xy"):
        return bool(c.get("at_xy"))
    if not c.get("at_xy"):
        return False
    pr = picture_rules(R or {})
    big = max([max((size_c or [0, 0])[:2]), max((size_p or [0, 0])[:2])])
    return math.dist(c["at_xy"], p["at_xy"]) <= max(float(pr["double_m"]), float(pr["double_k"]) * big)


def prepare(spec: dict, records: dict, R: dict) -> tuple:
    """The spec's pieces with the picture's own ones settled before the layout: a picture piece that is a spec piece
    seen again becomes that piece (the spec piece takes the picture's place and keeps its own look and mesh); one the
    picture's depth did not stand on the floor, or stood well off it, is refused; one the picture shows several of
    stands several times, the copies in a row along the room's long side sharing its one record (one mesh).
    -> (spec, records, notes). A spec with no picture pieces comes back as it was."""
    pcs = [dict(p) for p in spec.get("pieces") or []]
    if not any(is_picture_piece(p, spec) for p in pcs):
        return spec, records, []
    pr = picture_rules(R)
    recs, notes, out = dict(records or {}), [], []
    size = lambda p: (recs.get(p["id"]) or {}).get("size_m") or p.get("_seen_size_m") or \
        (PK.get(p.get("kind")) or {}).get("size_m") or DEFAULT_SIZE      # noqa: E731
    pic = [p for p in pcs if is_picture_piece(p, spec)]
    own = [p for p in pcs if not is_picture_piece(p, spec)]
    measured = bool(spec.get("_room_m"))
    taken = set()
    for c in pic:
        twin = next((p for p in own if p["id"] not in taken and same_thing(c, p, size(c), size(p), R)), None)
        if twin is not None and plural_count(c, R) == 1:
            taken.add(twin["id"])
            if not twin.get("at_xy") and c.get("at_xy"):
                twin.update(at_xy=list(c["at_xy"]), facing_xy=list(c.get("facing_xy") or [0.0, -1.0]))
                if c.get("_seen_size_m") and not twin.get("_seen_size_m"):
                    twin["_seen_size_m"] = c["_seen_size_m"]
            notes.append({"id": c["id"], "kind": c["kind"], "what": "merged",
                          "why": f"the same thing as {twin['id']}: built once, as it"})
            continue
        place = (PK.get(c.get("kind")) or {}).get("placement")
        if measured and not c.get("at_xy") and not isinstance(c.get("hangs_m"), (int, float)) and place != "surface":
            notes.append({"id": c["id"], "kind": c["kind"], "what": "refused",
                          "why": "the picture's depth does not stand it on the floor (overhead, on a wall, or no floor under it)"})
            continue
        if isinstance(c.get("_raised_m"), (int, float)) and place == "floor" and \
                float(c["_raised_m"]) > float(pr["raised_most_m"]):
            notes.append({"id": c["id"], "kind": c["kind"], "what": "refused", "why": f"the picture has it {c['_raised_m']} m off the floor"})
            continue
        c["_picture"] = True
        out.append(c)
        n = plural_count(c, R) if c.get("at_xy") else 1
        if n > 1:
            W, D = [float(x) for x in (spec.get("_room_m") or [0.0, 0.0])[:2]]
            f = tuple(c.get("facing_xy") or (0.0, -1.0))
            h = half_extents(size(c), f)
            ax = (0.0, 1.0) if D >= W else (1.0, 0.0)
            step = 2 * (h[0] * ax[0] + h[1] * ax[1]) + float(pr["row_gap_m"])
            # its row along the room's long side, then rows beside it (first on its own side of the room's middle),
            # each spot tried in turn: what does not fit or stands on the plan's floor is passed over
            lat = (ax[1], ax[0])
            lstep = 2 * (h[0] * lat[0] + h[1] * lat[1]) + float(pr["row_gap_m"])
            own_side = 1 if (c["at_xy"][0] * lat[0] + c["at_xy"][1] * lat[1]) >= 0 else -1
            spots = []
            for m in (0, own_side, 2 * own_side, -own_side, -2 * own_side):
                for k in range(0, 2 * int(pr["repeat_most"]) + 1):
                    for sgn in ((1,) if k == 0 else (1, -1)):
                        xy = [round(c["at_xy"][0] + sgn * k * step * ax[0] + m * lstep * lat[0], 3),
                              round(c["at_xy"][1] + sgn * k * step * ax[1] + m * lstep * lat[1], 3)]
                        if xy not in spots:
                            spots.append(xy)
            c["_row"] = spots                        # (its own spot first)
            for k in range(2, n + 1):
                cid = f"{c['id']}x{k}"
                out.append(dict(c, id=cid, at_xy=None, _row=spots, _copy_of=c["id"], _picture=True))
                if c["id"] in recs:
                    recs[cid] = recs[c["id"]]
            notes.append({"id": c["id"], "kind": c["kind"], "what": "repeated",
                          "why": f"the picture shows several: {n} asked, in rows along the room"})
    keep_ids = {p["id"] for p in own + out}
    sp = dict(spec, pieces=own + out,
              relations=[r for r in spec.get("relations") or [] if {r.get("a"), r.get("b")} <= keep_ids | set(
                  o.get("id") for o in spec.get("openings") or [])])
    return sp, recs, notes


def fire_of(ctx, pid: str) -> bool:
    """A fire piece: a piece of a floor kind that burns (the director's gives_light), not a lamp."""
    p = ctx.pieces[pid]
    k = p.get("kind")
    d = PK.get(k) or {}
    return p.get("gives_light") is True and d.get("placement", "floor") == "floor" and _root(k) != "lamp" \
        and k not in SURFACE_KINDS and not d.get("needs")


# ------------------------------------------------------------------ pieces
class Ctx:
    def __init__(self, spec, records, data, seed, rules=None):
        self.spec, self.records, self.data = spec, records, data
        self.R = rules or pc.load_rules()
        self.rk = data["room_kinds"][spec["kind"]]
        size = self.rk["sizes_m"][spec.get("size") if spec.get("size") in self.rk["sizes_m"] else "medium"]
        # a room measured from the place's own concept picture (img3d/concept_depth.py): its size, not the kind's
        size = spec.get("_room_m") or size
        self.W, self.D, self.H = [float(x) for x in size]
        self.rng = random.Random(seed)
        self.rel = data["relations"]
        self.gap = data["piece_gap_m"]
        self.openings, self.unmet = place_openings(spec, records, data, self.W, self.D, self.R)
        self.by_id = {o["id"]: o for o in self.openings}
        self.pieces = {p["id"]: p for p in spec.get("pieces", [])}
        allrel = [r for r in spec.get("relations", [])]
        # a relation the text states is hard; one the language model only infers (marked so) is a wish
        self.inferred = [r for r in allrel if r.get("inferred") or r.get("stated") is False]
        self.relations = [r for r in allrel if r not in self.inferred]
        self.partners, self.close = {}, {}          # related at all; related closely enough to share clear floor
        for r in self.relations + self.inferred:
            ids = [r.get("a"), r.get("b"), r.get("c")]
            for i in ids:
                if i:
                    self.partners.setdefault(i, set()).update(x for x in ids if x and x != i)
                    if r["rel"] in ("in_front_of", "behind", "beside"):
                        self.close.setdefault(i, set()).update(x for x in ids if x and x != i)
        self.names = self._names()
        focal = [o for o in self.openings if o["kind"] == self.rk.get("focal")]
        self.focal = focal[0] if focal else None
        # the picture's own pieces (prepare): dressing that stands only where it keeps off the plan's floor
        self.optional = {pid for pid, p in self.pieces.items() if p.get("_picture")}
        self.refused = []
        # fire pieces, what serves them beside them on their wall, and the piece worked at before each
        fr = fire_rules(self.R)
        self.fire = [pid for pid in self.pieces if fire_of(self, pid)]
        self.companion, self.work = {}, {}
        for r in self.relations + self.inferred:
            for x, y in ((r.get("a"), r.get("b")), (r.get("b"), r.get("a"))):
                if y in self.fire and x in self.pieces and x not in self.fire:
                    if r["rel"] in fr["companion_rels"]:
                        self.companion.setdefault(x, y)
                    elif self.works_at(x):
                        self.work.setdefault(x, y)
        if len(self.fire) == 1:
            # (the plan's own work piece in a room with one fire is worked at before it, related or not)
            for pid, p in self.pieces.items():
                if pid not in self.fire and pid not in self.companion and pid not in self.work \
                        and p.get("kind") == "work_piece" and p.get("phrases") and self.works_at(pid):
                    self.work[pid] = self.fire[0]

    def works_at(self, pid) -> bool:
        """A piece a standing person works at with their hands: a floor thing, no seat or bed, its top in the band."""
        k = self.pieces[pid].get("kind")
        d = PK.get(k) or {}
        lo, hi = fire_rules(self.R)["work_top_m"]
        return d.get("placement", "floor") == "floor" and not d.get("needs") and not self.rule(pid).get("seat") \
            and k not in SURFACE_KINDS and lo <= self.size(pid)[2] <= hi

    def rule(self, pid) -> dict:
        return self.data["piece_kinds"].get(self.pieces[pid]["kind"], {})

    def base(self, pid) -> str:
        return self.rule(pid).get("set_name") or self.pieces[pid]["kind"]

    def size(self, pid) -> list:
        rec = self.records.get(pid) or {}
        s = [float(x) for x in (rec.get("size_m") or self.rule(pid).get("size_m") or DEFAULT_SIZE)]
        k = self.pieces[pid].get("kind")
        d = PK.get(k) or {}
        if d.get("placement") == "wall" and k != "fireplace" and d.get("size_m") and len(s) == 3:
            # a thing hung flat on a wall takes no more floor than its kind does (by wall_depth_k): a made curtain
            # measured 1.46 m deep, was clamped off its wall onto the seat before it, and the seated person was
            # "inside" it, so no camera could see them and the curtain was dropped (11-20)
            s[1] = min(s[1], max(float(d["size_m"][1]), 0.1) * float(picture_rules(self.R)["wall_depth_k"]))
        return s

    def _names(self) -> dict:
        names, seen = {o["id"]: o["name"] for o in self.openings}, {}
        for o in self.openings:
            seen[o["kind"]] = seen.get(o["kind"], 0) + 1
        for pid in self.pieces:
            b = self.base(pid)
            seen[b] = seen.get(b, 0) + 1
            names[pid] = b if seen[b] == 1 else f"{b}_{seen[b]}"
        return names

    def mode(self, pid) -> tuple:
        """(how the piece stands: wall / free / floor / on, the opening it stands at or None)."""
        ag = self.pieces[pid].get("against")
        if pid in getattr(self, "fire", ()) or pid in getattr(self, "companion", {}):
            # a fire piece and what serves it stand back to a wall (a hearth out on the floor was a block no camera
            # could see past, 11-20)
            if isinstance(ag, str) and ag.startswith("opening:") and ag.split(":", 1)[1] in self.by_id:
                return "wall", self.by_id[ag.split(":", 1)[1]]
            return "wall", None
        place = self.rule(pid).get("place") or ("on" if self.pieces[pid]["kind"] in SURFACE_KINDS else "free")
        if place == "wall" and self.pieces[pid]["kind"] in self.R.get("desk_kinds", ()) and \
                pid in set((self.spec.get("named_places") or {}).values()):
            place = "free"       # a desk the plan seats someone at: off the wall, the camera's floor across it
        if isinstance(ag, str) and ag.startswith("opening:") and ag.split(":", 1)[1] in self.by_id:
            return "wall", self.by_id[ag.split(":", 1)[1]]
        if ag in ("wall", "free") and place in ("wall", "free"):
            return ag, None
        return place, None

    def thing(self, i, placed) -> dict | None:
        """A relation's end as a thing with a rectangle, a middle and a facing: a placed piece or an opening."""
        if i in placed:
            return placed[i]
        o = self.by_id.get(i)
        if o:
            return {"rect": o.get("breast") or o["rect"], "c": tuple(o["point"]), "facing": tuple(o["normal"]),
                    "opening": True, "id": i}
        return None


def rel_ok(ctx: Ctx, r: dict, placed: dict) -> bool | None:
    """True / False when every end of the relation stands; None while one is still to place."""
    A, B = ctx.thing(r.get("a"), placed), ctx.thing(r.get("b"), placed)
    if A is None or B is None:
        return None
    k, lim = r["rel"], ctx.rel.get(r["rel"], {})
    if k in ("by", "near"):
        return gap(A["rect"], B["rect"]) <= (lim["opening_gap_m"] if B.get("opening") else lim["gap_m"]) + 1e-6
    if k == "beside":
        return gap(A["rect"], B["rect"]) <= lim["gap_m"] + 1e-6
    v = (A["c"][0] - B["c"][0], A["c"][1] - B["c"][1])
    if k == "facing":
        d = unit((-v[0], -v[1]))
        return A["facing"][0] * d[0] + A["facing"][1] * d[1] >= math.cos(math.radians(lim["angle_deg"])) - 1e-6
    if k in ("in_front_of", "behind"):
        f = B["facing"] if k == "in_front_of" else (-B["facing"][0], -B["facing"][1])
        along = v[0] * f[0] + v[1] * f[1]
        side = abs(v[0] * f[1] - v[1] * f[0])
        hb = ((B["rect"][3] - B["rect"][1]) if abs(f[0]) > abs(f[1]) else (B["rect"][2] - B["rect"][0])) / 2
        return along > 0 and side <= max(hb, 0.3) + lim["side_m"] and gap(A["rect"], B["rect"]) <= lim["gap_m"]
    if k == "between":
        C = ctx.thing(r.get("c"), placed)
        if C is None:
            return None
        bx, by = B["c"]
        dx, dy = C["c"][0] - bx, C["c"][1] - by
        L2 = dx * dx + dy * dy
        if L2 < 1e-9:
            return False
        t = ((A["c"][0] - bx) * dx + (A["c"][1] - by) * dy) / L2
        off = math.hypot(A["c"][0] - (bx + t * dx), A["c"][1] - (by + t * dy))
        return lim["t"][0] <= t <= lim["t"][1] and off <= lim["off_m"]
    return True


def rel_words(ctx: Ctx, r: dict) -> str:
    n = ctx.names
    s = f"{n.get(r.get('a'), r.get('a'))} {r['rel'].replace('_', ' ')} {n.get(r.get('b'), r.get('b'))}"
    return s + (f" and {n.get(r.get('c'), r.get('c'))}" if r.get("c") else "")


def zones_of(ctx: Ctx, pid: str, c, facing, size) -> list:
    """The floor a piece keeps clear: in front of a seat or a cupboard, all round a table."""
    rule, out = ctx.rule(pid), []
    h = half_extents(size, facing)
    f = unit(facing)
    fc = rule.get("front_clear_m", 0.0)
    ar = rule.get("all_round_clear_m", 0.0)
    if pid in ctx.fire:                             # (back to its wall: the floor before it only)
        fc, ar = float(fire_rules(ctx.R)["front_clear_m"]), 0.0
    elif pid in ctx.companion:
        fc, ar = min(fc or 0.5, 0.5), 0.0
    if fc > 0:
        ext = abs(f[0]) * h[0] + abs(f[1]) * h[1]
        zc = (c[0] + f[0] * (ext + fc / 2), c[1] + f[1] * (ext + fc / 2))
        zh = (fc / 2, h[1]) if abs(f[0]) > abs(f[1]) else (h[0], fc / 2)
        out.append(rect_at(zc, zh))
    if ar > 0:
        out.append(rect_at(c, (h[0] + ar, h[1] + ar)))
    return out


def make(ctx: Ctx, pid: str, c, facing, extra=None) -> dict:
    size = ctx.size(pid)
    facing = unit(facing)
    h = half_extents(size, facing)
    d = {"id": pid, "name": ctx.names[pid], "kind": ctx.pieces[pid]["kind"], "c": (c[0], c[1]), "facing": facing,
         "rect": rect_at(c, h), "size_m": size, "flat": bool(ctx.rule(pid).get("flat")),
         "seat": bool(ctx.rule(pid).get("seat")), "zones": zones_of(ctx, pid, c, facing, size)}
    d.update(extra or {})
    return d


def exempt(ctx: Ctx, owner: str, other: str) -> bool:
    """A piece may stand in another's clear floor when the two belong together: a stated relation, or the kind's
    own rule (a desk chair at a desk)."""
    if other in ctx.close.get(owner, ()):
        return True
    if owner in ctx.optional and other in ctx.optional:
        return True                                 # (the picture's own things stand as close as it shows them)
    for a, b in ((owner, other), (other, owner)):
        want = ctx.rule(a).get("prefer_beside") if a in ctx.pieces else None
        if want and b in ctx.pieces and (want == ctx.base(b) or want == ctx.pieces[b]["kind"]
                                         or (want == "seat" and ctx.rule(b).get("seat"))):
            return True
    return False


def drape_zones(ctx: Ctx, o: dict) -> list:
    """The floor a window's drapes take (``hard.window_dressing``): ``(rect, is_a_side_column)`` for the part
    before the window and the columns beyond each side of it."""
    dr = ctx.R["hard"].get("window_dressing") or {}
    dw, dd = float(dr.get("width_m", 0.0)), float(dr.get("depth_m", 0.0))
    if dw <= 0 or dd <= 0:
        return []
    out = [(span_rect(o["wall"], o["at_m"], o["width_m"] / 2, dd, ctx.W, ctx.D), False)]
    side = (dw - o["width_m"]) / 2
    if side > 1e-6:
        for sgn in (1.0, -1.0):
            u = o["at_m"] + sgn * (o["width_m"] / 2 + side / 2)
            out.append((span_rect(o["wall"], u, side / 2, dd, ctx.W, ctx.D), True))
    return out


def fits(ctx: Ctx, cand: dict, placed: dict, fixed_zones: list) -> bool:
    r = cand["rect"]
    if r[0] < -ctx.W / 2 - 1e-6 or r[2] > ctx.W / 2 + 1e-6 or r[1] < -ctx.D / 2 - 1e-6 or r[3] > ctx.D / 2 + 1e-6:
        return False
    if cand["flat"] or cand.get("on"):
        return True
    for o in ctx.openings:
        if o["kind"] == "fireplace" and cand.get("fire") != o["id"] and overlap(r, o["breast"]):
            return False
    for z in fixed_zones:
        if overlap(r, z):
            return False
    hd = ctx.R["hard"]
    for o in ctx.openings:
        if o["kind"] != "window":
            continue
        tall = cand["size_m"][2] > o.get("sill_m", 0.0) + hd["window_tall_over_sill_m"] \
            and cand.get("at_window") != o["id"]     # what the place puts at its window may stand before it
        if tall and not cand.get("wall") and overlap(r, span_rect(o["wall"], o["at_m"], o["width_m"] / 2,
                                                                  hd["window_keepout_m"], ctx.W, ctx.D)):
            return False                            # a tall piece does not stand in front of a window
        for z, side in drape_zones(ctx, o):         # nor in the drapes; nothing stands in their side columns
            if overlap(r, z) and (side or tall):
                return False
    for q in placed.values():
        if q["flat"] or q.get("on"):
            continue
        ok = exempt(ctx, q["id"], cand["id"])
        air = ctx.gap if (ok or cand.get("wall") or q.get("wall") or q["kind"] == "fireplace") \
            else max(ctx.gap, hd["around_free_m"])
        if overlap(r, q["rect"], air):
            return False
        if not ok and any(overlap(r, z) for z in q["zones"]):
            return False
        if not ok and any(overlap(z, q["rect"]) for z in cand["zones"]):
            return False
    for z in cand["zones"]:                         # its own clear floor lies inside the room and off a breast
        if z[0] < -ctx.W / 2 - 1e-6 or z[2] > ctx.W / 2 + 1e-6 or z[1] < -ctx.D / 2 - 1e-6 or z[3] > ctx.D / 2 + 1e-6:
            return False
        if any(o["kind"] == "fireplace" and overlap(z, o["breast"]) for o in ctx.openings):
            return False
    return True


def near_score(ctx: Ctx, pid: str, cand: dict, placed: dict) -> float:
    rule, s = ctx.rule(pid), 0.0
    for key, sign in (("prefer", 1.0), ("avoid", -1.0)):
        for p in rule.get(key, []):
            ds = [gap(cand["rect"], o["rect"]) for o in ctx.openings if o["kind"] == p["near"]]
            if ds:
                s += sign * p.get("weight", 1.0) * min(ds)
    want = rule.get("prefer_beside")
    if want and not ctx.close.get(pid):             # a stated companion outranks the kind's usual one
        ds = [gap(cand["rect"], q["rect"]) for q in placed.values() if not q["flat"] and q["id"] in ctx.pieces
              and (want == ctx.base(q["id"]) or want == q["kind"]
                   or (want == "seat" and pc.is_kind(q, "seat", ctx.R)))]
        if ds:
            s += 2.0 * min(ds)
    for r in ctx.relations:                         # pull toward what it is related to, before the hard test
        if pid not in (r.get("a"), r.get("b"), r.get("c")):
            continue
        for other in (r.get("a"), r.get("b"), r.get("c")):
            t = ctx.thing(other, placed) if other and other != pid else None
            if t is not None and r["rel"] in ("by", "near", "beside", "in_front_of", "behind"):
                s += 4.0 * gap(cand["rect"], t["rect"])
                if r["rel"] in ("in_front_of", "behind") and other == r.get("b"):
                    f, v = t["facing"], (cand["c"][0] - t["c"][0], cand["c"][1] - t["c"][1])
                    s += 4.0 * abs(v[0] * f[1] - v[1] * f[0])       # square before it, not off to a side
    fd = rule.get("focal_distance_m")
    if fd and ctx.focal:
        d = math.hypot(cand["c"][0] - ctx.focal["point"][0], cand["c"][1] - ctx.focal["point"][1])
        s += 3.0 * max(fd[0] - d, d - fd[1], 0.0) + 0.2 * abs(d - (fd[0] + fd[1]) / 2)
    fs = rule.get("focal_side_m")
    if fs and ctx.focal:                            # to one side of the focal opening's axis, not straight before it
        n = ctx.focal["normal"]
        v = (cand["c"][0] - ctx.focal["point"][0], cand["c"][1] - ctx.focal["point"][1])
        side = abs(v[0] * n[1] - v[1] * n[0])
        s += 3.0 * max(fs[0] - side, side - fs[1], 0.0)
    return s


def face_for(ctx: Ctx, pid: str, c, placed: dict) -> tuple:
    """Where a free piece's front points: at what it is stated to face, else by its kind's rule (its partner,
    the room's focal opening), else as modelled."""
    for r in ctx.relations:
        if r.get("a") == pid and r["rel"] == "facing":
            t = ctx.thing(r.get("b"), placed)
            if t is not None:
                return unit((t["c"][0] - c[0], t["c"][1] - c[1]))
    rule = ctx.rule(pid)
    if rule.get("face") == "partner":
        ts = [ctx.thing(o, placed) for o in sorted(ctx.partners.get(pid, ()))]
        ts = [t for t in ts if t is not None]
        want = rule.get("prefer_beside")
        if not ts and want:
            ts = [q for q in placed.values() if q["id"] in ctx.pieces
                  and (want == ctx.base(q["id"]) or want == q["kind"])]
            ts.sort(key=lambda q: math.hypot(q["c"][0] - c[0], q["c"][1] - c[1]))
        if ts:
            return unit((ts[0]["c"][0] - c[0], ts[0]["c"][1] - c[1]))
        return unit((-c[0], -c[1]))
    if rule.get("face") == "focal":
        tgt = ctx.focal["point"] if ctx.focal else (0.0, 0.0)
        return unit((tgt[0] - c[0], tgt[1] - c[1]))
    return (0.0, -1.0)


def free_facings(ctx: Ctx, pid: str, c, placed: dict) -> list:
    """The fronts tried for a free piece at a point: where its rule turns it (at its partner, at the focal
    opening), for a seat also that direction snapped to a round angle and to the nearest wall direction, and for
    a piece with no rule both ways round."""
    rule = ctx.rule(pid)
    f0 = face_for(ctx, pid, c, placed)
    out = [f0]
    stated = any(r.get("a") == pid and r["rel"] == "facing" for r in ctx.relations)
    if rule.get("seat") and rule.get("face") == "focal" and not stated:
        step = math.radians(ctx.R["search"]["seat_snap_deg"])
        a = math.atan2(f0[1], f0[0])
        lim = math.radians(ctx.R["search"]["polish_turn_max_deg"])
        for b in (round(a / step) * step, round(a / (math.pi / 2)) * (math.pi / 2)):
            if abs((b - a + math.pi) % (2 * math.pi) - math.pi) <= lim + 1e-9:   # still turned to what it faces
                out.append((math.cos(b), math.sin(b)))
    elif not rule.get("face") and not stated:
        size = ctx.size(pid)
        if abs(size[0] - size[1]) > 0.05:
            out.append((1.0, 0.0))
    seen, uniq = set(), []
    for f in out:
        f = unit(f)
        k = (round(f[0], 3), round(f[1], 3))
        if k not in seen:
            seen.add(k)
            uniq.append(f)
    return uniq


def slot_points(ctx: Ctx, pid: str, placed: dict) -> list:
    """Where a companion stands by the group template (``reach`` in layout_rules.json): square before its
    anchor's front, at either side of it, or at either side of its head end, at the wished gap. The grid alone
    misses these by up to a cell."""
    kind, size, out = ctx.pieces[pid]["kind"], ctx.size(pid), []
    for rr in ctx.R["reach"]:
        if rr["kind"] != kind:
            continue
        g = max(rr["gap_m"][0], ctx.gap + 0.01)
        for t in placed.values():
            if t.get("flat") or t.get("on") or t["id"] == pid or not pc.is_kind(t, rr["to"], ctx.R):
                continue
            f = t["facing"]
            lat = (f[1], -f[0])
            hx, hy = (t["rect"][2] - t["rect"][0]) / 2, (t["rect"][3] - t["rect"][1]) / 2
            ef, el = abs(f[0]) * hx + abs(f[1]) * hy, abs(lat[0]) * hx + abs(lat[1]) * hy
            mine = max(size[0], size[1]) / 2
            if rr["where"] == "front":
                out.append((t["c"][0] + f[0] * (ef + g + mine), t["c"][1] + f[1] * (ef + g + mine)))
            else:
                back = (ef - mine) if rr["where"] == "head" else 0.0
                for sgn in (1.0, -1.0):
                    out.append((t["c"][0] - f[0] * back + sgn * lat[0] * (el + g + mine),
                                t["c"][1] - f[1] * back + sgn * lat[1] * (el + g + mine)))
    return [(round(c[0], 3), round(c[1], 3)) for c in out]


def rug_place(ctx: Ctx, placed: dict) -> tuple:
    """Where a rug lies: under the seats gathered at the focal point, else beside a bed, else the middle of
    the floor. -> (centre, the direction its long side runs or None)."""
    seats = pc.seating(placed, ctx.R)
    if len(seats) >= 2:
        pts = [q["c"] for q in seats]
        if ctx.focal and ctx.focal["id"] in placed:
            pts = pts + [stand_point(placed[ctx.focal["id"]], ctx.data)]
        c = (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
        n = ctx.focal["normal"] if ctx.focal else None
        return c, ((0.0, -1.0) if n is None or abs(n[1]) > abs(n[0]) else (1.0, 0.0))
    beds = [q for q in placed.values() if q["kind"] == "bed"]
    if beds:
        b = beds[0]
        f = b["facing"]
        side = (f[1], -f[0])
        if (b["c"][0] + side[0]) ** 2 + (b["c"][1] + side[1]) ** 2 > b["c"][0] ** 2 + b["c"][1] ** 2:
            side = (-side[0], -side[1])             # the side toward the room's middle
        hw = (abs(side[0]) * (b["rect"][2] - b["rect"][0]) + abs(side[1]) * (b["rect"][3] - b["rect"][1])) / 2
        c = (b["c"][0] + side[0] * (hw + 0.6), b["c"][1] + side[1] * (hw + 0.6))
        return c, ((1.0, 0.0) if abs(side[0]) > abs(side[1]) else (0.0, -1.0))
    return (0.0, 0.0), None


def keeps_off(ctx: Ctx, cand: dict, placed: dict) -> bool:
    """A picture piece keeps off the floor the plan's people need: the floor before a seat (seat_keep_m) or before
    another piece the plan names (named_keep_m), and the floor between two pieces the spec relates or two named
    seats (keep_side_m either side of the line between them). The roof's beams read as a table stood on the floor
    straight in front of the chair the plan seats someone on, between them and the one they spoke to (11-20)."""
    pr = picture_rules(ctx.R)
    r, side = cand["rect"], float(pr["keep_side_m"])
    named = set((ctx.spec.get("named_places") or {}).values())
    own = [q for q in placed.values() if q["id"] not in ctx.optional and not q.get("flat") and not q.get("on")]
    for q in own:
        if not (q.get("seat") or q["id"] in named):
            continue
        depth = float(pr["seat_keep_m"] if q.get("seat") else pr["named_keep_m"])
        f = unit(q["facing"])
        hx, hy = (q["rect"][2] - q["rect"][0]) / 2, (q["rect"][3] - q["rect"][1]) / 2
        ext = abs(f[0]) * hx + abs(f[1]) * hy
        zc = (q["c"][0] + f[0] * (ext + depth / 2), q["c"][1] + f[1] * (ext + depth / 2))
        zh = (depth / 2, hy + side) if abs(f[0]) > abs(f[1]) else (hx + side, depth / 2)
        if overlap(r, rect_at(zc, zh)):
            return False
    def line(a, b):
        # (from the near side of each end, not its middle: what stands behind a seat is not between it and anyone)
        L_ = math.dist(a["c"], b["c"])
        if L_ < 1e-6:
            return None
        u = ((b["c"][0] - a["c"][0]) / L_, (b["c"][1] - a["c"][1]) / L_)
        ka = min(L_ / 2, max(a["rect"][2] - a["rect"][0], a["rect"][3] - a["rect"][1]) / 2)
        kb = min(L_ / 2, max(b["rect"][2] - b["rect"][0], b["rect"][3] - b["rect"][1]) / 2)
        return ((a["c"][0] + u[0] * ka, a["c"][1] + u[1] * ka), (b["c"][0] - u[0] * kb, b["c"][1] - u[1] * kb))
    lines = []
    for rel in ctx.relations:
        ends = [ctx.thing(x, placed) for x in (rel.get("a"), rel.get("b"), rel.get("c")) if x]
        ends = [t for t in ends if t is not None and t.get("id") not in ctx.optional]
        lines += [line(ends[0], t) for t in ends[1:]]
    seats = [q for q in own if q.get("seat") and q["id"] in named]
    lines += [line(a, b) for i, a in enumerate(seats) for b in seats[i + 1:]]
    lines = [x for x in lines if x]
    big = (r[0] - side, r[1] - side, r[2] + side, r[3] + side)
    return not any(pc.seg_hits_rect(a, b, big) for a, b in lines)


def work_spots(ctx: Ctx, pid: str, placed: dict, relaxed: bool = False) -> list:
    """Where a piece worked at before a fire stands: out on the floor in front of the fire, work_front_m from the
    fire's front to its own, its front to the fire, at most work_side_m off the fire's middle; kept only with clear
    floor work_clear_m wide on work_clear_sides of its sides (inside the room, nothing standing there), so a person
    stands at it and a camera sees them both from the room."""
    fr = fire_rules(ctx.R)
    if relaxed:
        fr = dict(fr, **(fr.get("relaxed") or {}))
    F = placed.get(ctx.work[pid])
    if F is None:
        return []
    f = unit(F["facing"])
    lat = (f[1], -f[0])
    hx, hy = (F["rect"][2] - F["rect"][0]) / 2, (F["rect"][3] - F["rect"][1]) / 2
    ef = abs(f[0]) * hx + abs(f[1]) * hy
    face = (-f[0], -f[1])
    size = ctx.size(pid)
    h = half_extents(size, face)
    eo = abs(f[0]) * h[0] + abs(f[1]) * h[1]
    lo, hi = [float(x) for x in fr["work_front_m"]]
    sw, cm, need = float(fr["work_side_m"]), float(fr["work_clear_m"]), int(fr["work_clear_sides"])
    solid_ = [q["rect"] for q in placed.values() if not q.get("flat") and not q.get("on")]
    solid_ += [o["breast"] for o in ctx.openings if o["kind"] == "fireplace"]
    W, D, out = ctx.W, ctx.D, []
    n_d = int(round((hi - lo) / 0.1))
    lats = [0.0] + [s_ * k * 0.3 for k in range(1, int(sw / 0.3) + 1) for s_ in (1, -1)]
    for i in range(n_d + 1):
        d = lo + i * 0.1
        for s_ in lats:
            c = (F["c"][0] + f[0] * (ef + d + eo) + lat[0] * s_, F["c"][1] + f[1] * (ef + d + eo) + lat[1] * s_)
            rr = rect_at(c, h)
            strips = [(rr[0] - cm, rr[1], rr[0], rr[3]), (rr[2], rr[1], rr[2] + cm, rr[3]),
                      (rr[0], rr[1] - cm, rr[2], rr[1]), (rr[0], rr[3], rr[2], rr[3] + cm)]
            free = sum(1 for z in strips if z[0] >= -W / 2 - 1e-6 and z[2] <= W / 2 + 1e-6 and z[1] >= -D / 2 - 1e-6
                       and z[3] <= D / 2 + 1e-6 and not any(overlap(z, q) for q in solid_))
            if free >= need:
                out.append(make(ctx, pid, (round(c[0], 3), round(c[1], 3)), face,
                                {"work_d": round(d, 2), "work_side": round(s_, 2), "work_of": ctx.work[pid]}))
    return out


def candidates(ctx: Ctx, pid: str, placed: dict, fixed_zones: list, hard: bool = True, relaxed: bool = False) -> list:
    pin = ctx.pieces[pid].get("at_xy")
    if pid in ctx.fire or pid in ctx.companion or pid in ctx.work:
        pin = None                                  # (the fire's rules place these, not the picture's estimate)
    row = ctx.pieces[pid].get("_row")
    if pid in ctx.fire or pid in ctx.companion or pid in ctx.work:
        row = None
    if row:
        # a copy of a picture piece shown several times: the next spot of its row that it fits
        f = tuple(ctx.pieces[pid].get("facing_xy") or (0.0, -1.0))
        h = half_extents(ctx.size(pid), f)
        got = []
        for k_, xy in enumerate(row):
            if abs(xy[0]) > ctx.W / 2 - h[0] + 1e-6 or abs(xy[1]) > ctx.D / 2 - h[1] + 1e-6:
                continue
            cand = make(ctx, pid, tuple(xy), f)
            cand["pinned"], cand["near"] = True, round(0.5 * k_, 4)      # (the nearer spots of its row first)
            if fits(ctx, dict(cand, zones=[]), placed, fixed_zones) and keeps_off(ctx, cand, placed):
                got.append(cand)
        return got
    if pin:
        # where the place's concept picture has it (concept_depth.py): that spot and facing only, kept inside the
        # room; the picture is the place's look, so it stands there even where a rule would want it elsewhere
        h = half_extents(ctx.size(pid), tuple(ctx.pieces[pid].get("facing_xy") or (0.0, -1.0)))
        c = (max(-ctx.W / 2 + h[0], min(ctx.W / 2 - h[0], pin[0])), max(-ctx.D / 2 + h[1], min(ctx.D / 2 - h[1], pin[1])))
        cand = make(ctx, pid, c, tuple(ctx.pieces[pid].get("facing_xy") or (0.0, -1.0)))
        cand["pinned"] = True
        # (its own clear floor is not asked where the picture stands it: only that it is clear of the others')
        if pid in ctx.optional and not (fits(ctx, dict(cand, zones=[]), placed, fixed_zones)
                                        and keeps_off(ctx, cand, placed)):
            return []                               # (a picture piece stands where the picture has it, or not at all)
        return [cand]
    mode, at = ctx.mode(pid)
    size, rule = ctx.size(pid), ctx.rule(pid)
    W, D = ctx.W, ctx.D
    wg, out = ctx.data["wall_gap_m"], []
    if mode == "fireplace":
        return []
    if mode == "floor":                             # a rug: the middle of the floor, long side along the room's
        c, along = rug_place(ctx, placed)
        if along is None:
            facing = (0.0, -1.0) if (size[0] >= size[1]) == (W >= D) else (1.0, 0.0)
        else:                                       # its long side across the group it lies under
            facing = along if size[0] >= size[1] else (along[1], -along[0])
        h = half_extents(size, facing)
        sc = min(1.0, (W - 0.3) / (2 * h[0]), (D - 0.3) / (2 * h[1]))
        fx = placed.get(ctx.focal["id"]) if ctx.focal and along is not None else None
        if fx is not None:                          # before the hearth, not under the grate
            n = ctx.focal["normal"]
            fr = fx["rect"]
            edge = (fr[0] if n[0] < 0 else fr[2]) if abs(n[0]) > abs(n[1]) else (fr[1] if n[1] < 0 else fr[3])
            if abs(n[0]) > abs(n[1]):
                c = (edge + n[0] * (h[0] + 0.05) if (c[0] - edge) * n[0] < h[0] + 0.05 else c[0], c[1])
            else:
                c = (c[0], edge + n[1] * (h[1] + 0.05) if (c[1] - edge) * n[1] < h[1] + 0.05 else c[1])
        c = (max(-W / 2 + h[0], min(W / 2 - h[0], c[0])) if 2 * h[0] <= W else 0.0,
             max(-D / 2 + h[1], min(D / 2 - h[1], c[1])) if 2 * h[1] <= D else 0.0)
        cand = make(ctx, pid, c, facing)
        if sc < 1.0:
            cand["note"] = "larger than the floor"
        return [cand] if fits(ctx, cand, placed, fixed_zones) else []
    if mode == "on":
        # a related piece hosts it when it is a surface; a relation to a bed or a seat is nearness, not a host: the
        # surface nearest that piece takes it (a clock "near the bed" stood on the bed, 11-12)
        rel = [placed[o] for o in sorted(ctx.partners.get(pid, ())) if o in placed]
        hosts = [q for q in rel if not q["seat"] and q["kind"] not in PERSON_KINDS]
        if not hosts:
            near = lambda q: min((math.dist(q["c"], r["c"]) for r in rel), default=0.0)
            hosts = sorted((q for q in placed.values() if not q["flat"] and not q["seat"] and not q.get("on")
                            and q["size_m"][2] <= 1.2 and q["kind"] != "fireplace"
                            and q["kind"] not in PERSON_KINDS),
                           key=lambda q: (round(near(q), 2), q["size_m"][0] * q["size_m"][1], q["name"]))
        for hst in hosts:
            # things on one host stand side by side along its width, not through each other
            n_on = sum(1 for q in placed.values() if q.get("on") == hst["name"])
            step_ = ((n_on + 1) // 2) * (1 if n_on % 2 else -1) * 0.45
            lim = max(0.0, hst["size_m"][0] / 2 - 0.15)
            off = max(-lim, min(lim, step_))
            f_ = hst["facing"]
            c_ = (hst["c"][0] - f_[1] * off, hst["c"][1] + f_[0] * off)
            # the top it stands on: a record's height includes what is modelled on the piece (a desk's own books and
            # pen: 0.90 m against a 0.76 m top, things floated above it), so never above the kind's own height
            kz = (ctx.rule(hst["id"]).get("size_m") or [0, 0, hst["size_m"][2]])[2]
            return [make(ctx, pid, c_, hst["facing"], {"on": hst["name"], "on_z_m": min(hst["size_m"][2], kz)})]
        return []
    step = ctx.data["grid_m"] if mode == "wall" else ctx.data["free_grid_m"]
    if pid in ctx.work:
        out = work_spots(ctx, pid, placed, relaxed)
    elif mode == "wall":
        walls = [at["wall"]] if at else list(WALLS)
        for side in walls:
            n, L = NORMAL[side], wall_len(side, W, D)
            hw = size[0] / 2
            lo, hi = -(L / 2 - hw - 0.02), (L / 2 - hw - 0.02)
            if at:                                  # at an opening: under a window, in the alcove by a fireplace
                span_w = at["width_m"]
                if at["kind"] == "window" and drape_zones(ctx, at):
                    span_w = max(span_w, ctx.R["hard"]["window_dressing"]["width_m"])
                reach_m = span_w / 2 + hw + (2 * hw + 0.4 if at["kind"] == "fireplace" else 0.0)
                lo, hi = max(lo, at["at_m"] - reach_m), min(hi, at["at_m"] + reach_m)
            k0, k1 = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
            for k in range(k0, k1 + 1):
                u = round(k * step, 3)
                p = wall_pt(side, u, W, D)
                off = wg
                if size[2] <= ctx.R["hard"].get("low_under_window_m", 1.0) or (at and at["kind"] == "window"):
                    # a low piece at a window, or any piece the place puts at that window, stands out from the
                    # wall by the drapes' depth, so they hang behind it
                    for o in ctx.openings:
                        if o["kind"] == "window" and o["wall"] == side and drape_zones(ctx, o):
                            dz = ctx.R["hard"]["window_dressing"]
                            if abs(o["at_m"] - u) < max(o["width_m"], dz["width_m"]) / 2 + hw - 1e-6:
                                off = max(off, dz["depth_m"] + wg)
                c = (p[0] + n[0] * (size[1] / 2 + off), p[1] + n[1] * (size[1] / 2 + off))
                cand = make(ctx, pid, c, n, {"wall": side, "u": u})
                if at and at["kind"] == "window":
                    cand["at_window"] = at["id"]
                blocked = False
                for o in ctx.openings:
                    if o["wall"] != side or o["kind"] == "fireplace":
                        continue
                    if abs(o["at_m"] - u) < o["width_m"] / 2 + hw - 1e-6:
                        low = size[2] <= o.get("sill_m", 0.0) + 0.15
                        if o["kind"] == "door" or not ((rule.get("under_window") and low) or (at and at is o)):
                            blocked = True
                if not blocked:
                    out.append(cand)
    else:
        nx, ny = int((W / 2) / step), int((D / 2) / step)
        for i in range(-nx, nx + 1):
            for j in range(-ny, ny + 1):
                c = (round(i * step, 3), round(j * step, 3))
                fs = free_facings(ctx, pid, c, placed)
                for f in fs:
                    out.append(make(ctx, pid, c, f, {"base_facing": fs[0]}))
        for c in slot_points(ctx, pid, placed):     # its place in a group: before a desk, beside a seat
            fs = free_facings(ctx, pid, c, placed)
            for f in fs:
                out.append(make(ctx, pid, c, f, {"base_facing": fs[0], "slot": True}))
    keep = []
    sc_m = rule.get("side_clear_m", 0.0)
    for cand in out:
        if sc_m > 0:                                # one long side stays free to walk to (a bed is got into)
            f, r = cand["facing"], cand["rect"]
            strips = ([(r[0], r[1] - sc_m, r[2], r[1]), (r[0], r[3], r[2], r[3] + sc_m)] if abs(f[0]) > abs(f[1])
                      else [(r[0] - sc_m, r[1], r[0], r[3]), (r[2], r[1], r[2] + sc_m, r[3])])
            free = [z for z in strips if z[0] >= -W / 2 - 1e-6 and z[2] <= W / 2 + 1e-6 and z[1] >= -D / 2 - 1e-6
                    and z[3] <= D / 2 + 1e-6 and not any(overlap(z, q["rect"]) for q in placed.values()
                                                         if not q["flat"] and not q.get("on")
                                                         and not exempt(ctx, pid, q["id"]))]
            if not free:
                continue
            cand["zones"] = cand["zones"] + [free[0]]
        if not fits(ctx, cand, placed, fixed_zones):
            continue
        if pid in ctx.optional and not keeps_off(ctx, cand, placed):
            continue
        trial = dict(placed)
        trial[pid] = cand
        if hard and any(rel_ok(ctx, r, trial) is False for r in ctx.relations
                        if pid in (r.get("a"), r.get("b"), r.get("c"))):
            continue
        sc = near_score(ctx, pid, cand, placed)
        if pid in ctx.work:                         # (square before the fire, at the middle of its band)
            lo, hi = fire_rules(ctx.R)["work_front_m"]
            sc += abs(cand["work_d"] - (lo + hi) / 2) + 0.5 * abs(cand["work_side"])
        elif mode == "wall":
            sc += 0.02 * abs(cand["u"]) + 0.01 * WALLS.index(cand["wall"])
            at_ = ctx.pieces[pid].get("at_xy")
            if at_ and (pid in ctx.fire or pid in ctx.companion):
                # as near where the picture has it as its wall allows, on the wall its front turns from
                fx = ctx.pieces[pid].get("facing_xy") or cand["facing"]
                sc += 0.5 * math.dist(cand["c"], at_) + 2.0 * (1.0 - (fx[0] * cand["facing"][0] + fx[1] * cand["facing"][1]))
        else:
            sc += 0.03 * min(W / 2 - abs(cand["c"][0]), D / 2 - abs(cand["c"][1]))
        if cand.get("slot"):                        # a template slot is tried before the grid cells round it
            sc -= ctx.R["search"]["slot_bonus"]
        cand["near"] = round(sc, 4)
        keep.append((round(sc, 4), cand))
    ties = {}
    for sc, cand in keep:
        ties.setdefault(sc, []).append(cand)
    ordered = []
    for sc in sorted(ties):
        group = ties[sc]
        if len(group) > 1:
            ctx.rng.shuffle(group)
        ordered += group
    if pid in ctx.work and not ordered and not relaxed:
        return candidates(ctx, pid, placed, fixed_zones, hard, relaxed=True)
    return ordered


def fire_piece(ctx: Ctx, o: dict) -> dict:
    """The fireplace as a placed thing in front of its chimney breast (a fixture of the set)."""
    od = ctx.data["openings"]["fireplace"]
    rec = ctx.records.get(o["id"]) or {}
    size = [float(x) for x in (rec.get("size_m") or od["size_m"])]
    n = o["normal"]
    p = wall_pt(o["wall"], o["at_m"], ctx.W, ctx.D)
    c = (p[0] + n[0] * (od["breast_depth_m"] + size[1] / 2), p[1] + n[1] * (od["breast_depth_m"] + size[1] / 2))
    h = half_extents(size, n)
    fc = od.get("front_clear_m", 0.0)
    ext = abs(n[0]) * h[0] + abs(n[1]) * h[1]
    zc = (c[0] + n[0] * (ext + fc / 2), c[1] + n[1] * (ext + fc / 2))
    zone = rect_at(zc, (fc / 2, h[1]) if abs(n[0]) > abs(n[1]) else (h[0], fc / 2))
    return {"id": o["id"], "name": o["name"], "kind": "fireplace", "c": c, "facing": tuple(n),
            "rect": rect_at(c, h), "size_m": size, "flat": False, "seat": False, "zones": [zone] if fc > 0 else [],
            "fire": o["id"], "asset": rec.get("asset") or od.get("default_asset")}


# ------------------------------------------------------------------ paths
def stand_point(t: dict, data: dict) -> tuple:
    """Where a person stands to be at a thing: in front of a piece, just inside an opening."""
    if t.get("opening_kind"):
        s = data["openings"][t["opening_kind"]]["stand_m"]
        return (t["c"][0] + t["facing"][0] * s, t["c"][1] + t["facing"][1] * s)
    h = ((t["rect"][2] - t["rect"][0]) / 2, (t["rect"][3] - t["rect"][1]) / 2)
    ext = abs(t["facing"][0]) * h[0] + abs(t["facing"][1]) * h[1]
    return (t["c"][0] + t["facing"][0] * (ext + 0.45), t["c"][1] + t["facing"][1] * (ext + 0.45))


def free_grid(W, D, rects, clear, cell=0.1):
    nx, ny = int(round(W / cell)), int(round(D / cell))
    grid = [[True] * ny for _ in range(nx)]
    for i in range(nx):
        x = -W / 2 + (i + 0.5) * cell
        for j in range(ny):
            y = -D / 2 + (j + 0.5) * cell
            if min(x + W / 2, W / 2 - x, y + D / 2, D / 2 - y) < clear - 1e-9:
                grid[i][j] = False
                continue
            for r in rects:
                if r[0] - clear < x < r[2] + clear and r[1] - clear < y < r[3] + clear:
                    grid[i][j] = False
                    break
    return grid, nx, ny


def reach(W, D, rects, points, clear, cell=0.1, snap=0.6) -> list:
    """For each named point: can a person walk to it from the first one (a flood fill over the free floor)?
    A point inside a piece's keep-off margin snaps to the nearest free cell within ``snap`` metres."""
    grid, nx, ny = free_grid(W, D, rects, clear, cell)

    def cell_of(p):
        i0, j0 = int((p[0] + W / 2) / cell), int((p[1] + D / 2) / cell)
        best, R = None, int(snap / cell)
        for i in range(max(0, i0 - R), min(nx, i0 + R + 1)):
            for j in range(max(0, j0 - R), min(ny, j0 + R + 1)):
                if grid[i][j]:
                    d = (i - i0) ** 2 + (j - j0) ** 2
                    if d <= R * R and (best is None or d < best[0]):
                        best = (d, i, j)
        return None if best is None else (best[1], best[2])

    cells = [cell_of(p) for p in points]
    if not cells or cells[0] is None:
        return [False] * len(points)
    seen = {cells[0]}
    q = deque([cells[0]])
    while q:
        i, j = q.popleft()
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if 0 <= a < nx and 0 <= b < ny and grid[a][b] and (a, b) not in seen:
                seen.add((a, b))
                q.append((a, b))
    return [c is not None and c in seen for c in cells]


def path_problems(ctx: Ctx, placed: dict) -> list:
    """Named places (and every door) joined by clear floor at least a walkway wide."""
    rects = [q["rect"] for q in placed.values() if not q["flat"] and not q.get("on")]
    rects += [o["breast"] for o in ctx.openings if o["kind"] == "fireplace"]
    pts, labels = [], []
    for o in ctx.openings:
        if o["kind"] == "door":
            pts.append(stand_point({"opening_kind": "door", "c": o["point"], "facing": o["normal"]}, ctx.data))
            labels.append(o["name"])
    for phrase, i in (ctx.spec.get("named_places") or {}).items():
        if i in placed and not placed[i]["flat"]:
            host = placed[i]
            if host.get("on"):
                host = next((q for q in placed.values() if q["name"] == host["on"]), host)
            pts.append(stand_point(host, ctx.data))
            labels.append(ctx.names[i])
        elif i in ctx.by_id:
            o = ctx.by_id[i]
            if o["kind"] == "fireplace" and i in placed:
                pts.append(stand_point(placed[i], ctx.data))
            else:
                pts.append(stand_point({"opening_kind": o["kind"], "c": o["point"], "facing": o["normal"]}, ctx.data))
            labels.append(o["name"])
    if len(pts) < 2:
        return []
    ok = reach(ctx.W, ctx.D, rects, pts, ctx.data["walkway_m"] / 2, snap=ctx.R["hard"]["stand_snap_m"])
    if not ok[0]:
        return [f"no clear floor at {labels[0]}"]
    return [f"no clear path from {labels[0]} to {lab}" for lab, good in zip(labels[1:], ok[1:]) if not good]


# ------------------------------------------------------------------ solve
def order_of(ctx: Ctx) -> list:
    rank = {"wall": 0, "free": 1, "floor": 3, "on": 4}

    def key(pid):                                   # walls first, big before small, companions after their hosts
        mode, at = ctx.mode(pid)
        size = ctx.size(pid)
        r = rank.get(mode, 1)
        if r < 2 and ctx.rule(pid).get("prefer_beside") and not at:
            r = 2
        # (a piece the concept picture places stands first: the others find their room round it. Placed after its
        # companion, the forge's anvil came down inside the bellows, 11-17)
        # (a fire piece first, what serves it next, then what the picture places; the picture's own pieces last,
        # each where it keeps off the floor the others need)
        grp = 0 if pid in ctx.fire else 1 if pid in ctx.companion else 4 if pid in ctx.optional else \
            2 if ctx.pieces[pid].get("at_xy") and pid not in ctx.work else 3
        return (grp, r, 0 if at else 1, -round(size[0] * size[1], 3), -len(ctx.partners.get(pid, ())), pid)

    return sorted(ctx.pieces, key=key)


def fixed_zones(ctx: Ctx) -> list:
    """Floor nothing stands on: each door's swing and the strip a person needs to come through it."""
    ap = ctx.R["hard"]["door_approach_m"]
    return [span_rect(o["wall"], o["at_m"], o["width_m"] / 2, o["width_m"] + ap, ctx.W, ctx.D)
            for o in ctx.openings if o["kind"] == "door"]


def objective(ctx: Ctx, placed: dict, with_pockets: bool = True) -> tuple:
    """The whole cost of a (partial) layout: the soft terms plus each piece's pull toward what its kind likes."""
    cost, terms, pieces = pc.total(ctx, placed, ctx.R, rel_ok, stand_point, with_pockets)
    near = sum(q.get("near", 0.0) for q in placed.values() if q["id"] in ctx.pieces)
    return cost + near, terms, pieces


def hard_ok(ctx: Ctx, placed: dict, pid: str, fixed: list) -> bool:
    cand = placed[pid]
    others = {k: v for k, v in placed.items() if k != pid}
    if not fits(ctx, cand, others, fixed):
        return False
    if any(rel_ok(ctx, r, placed) is False for r in ctx.relations):
        return False
    return bool(cand["flat"] or cand.get("on") or not path_problems(ctx, placed))


def beam_search(ctx: Ctx, order: list, start: dict, fixed: list) -> dict | None:
    """Place the pieces in order, keeping the few cheapest partial layouts at each step. Fixed order, stable
    sort, the seed only breaking exact ties inside ``candidates``: the same inputs give the same room."""
    sr = ctx.R["search"]
    states = [dict(start)]
    for pid in order:
        nxt, seq = [], 0
        for placed in states:
            for cand in candidates(ctx, pid, placed, fixed)[:sr["expand"]]:
                st = dict(placed)
                st[pid] = cand
                cost = objective(ctx, st, with_pockets=False)[0]
                nxt.append((round(cost, 6), seq, st))
                seq += 1
        if not nxt and pid in ctx.optional:
            continue                                # (a picture piece with no room is left out, not a failure)
        nxt.sort(key=lambda t: (t[0], t[1]))
        keep = []
        for _c, _s, st in nxt:
            cand = st[pid]
            if cand["flat"] or cand.get("on") or not path_problems(ctx, st):
                keep.append(st)
                if len(keep) == sr["beam"]:
                    break
        if not keep and pid in ctx.optional:
            continue
        if not keep:
            return None
        states = keep
    best = min(enumerate(states), key=lambda t: (round(objective(ctx, t[1])[0], 6), t[0]))
    return best[1]


def polish(ctx: Ctx, placed: dict, order: list, fixed: list) -> dict:
    """Coordinate descent: one free piece at a time, a small shift or turn, kept only when every hard rule
    still holds and the cost falls."""
    sr = ctx.R["search"]
    step, turn = sr["polish_step_m"], math.radians(sr["polish_turn_deg"])
    tmax = math.radians(sr["polish_turn_max_deg"])
    best = objective(ctx, placed)[0]
    for _ in range(int(sr["polish_passes"])):
        moved = False
        for pid in order:
            q = placed.get(pid)
            if q is None:
                continue
            if q["flat"] or q.get("on") or q.get("wall") or ctx.rule(pid).get("side_clear_m") \
                    or pid in ctx.work or pid in ctx.optional:
                continue
            base = q.get("base_facing") or q["facing"]
            a0 = math.atan2(q["facing"][1], q["facing"][0])
            ab = math.atan2(base[1], base[0])
            tries = [((q["c"][0] + dx, q["c"][1] + dy), q["facing"])
                     for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step))]
            stated = any(r.get("a") == pid and r["rel"] == "facing" for r in ctx.relations)
            if q["seat"] and not stated:
                for da in (turn, -turn):
                    a = a0 + da
                    off = (a - ab + math.pi) % (2 * math.pi) - math.pi
                    if abs(off) <= tmax + 1e-9:
                        tries.append((q["c"], (math.cos(a), math.sin(a))))
            for c, f in tries:
                cand = make(ctx, pid, (round(c[0], 3), round(c[1], 3)), f, {"base_facing": base})
                cand["near"] = round(near_score(ctx, pid, cand, {k: v for k, v in placed.items() if k != pid}), 4)
                trial = dict(placed)
                trial[pid] = cand
                for k, v in placed.items():         # what stands on it moves with it
                    if v.get("on") == q["name"]:
                        trial[k] = dict(v, c=cand["c"], rect=rect_at(cand["c"], half_extents(v["size_m"],
                                                                                              v["facing"])))
                cost = objective(ctx, trial)[0]
                if cost < best - 1e-6 and hard_ok(ctx, trial, pid, fixed):
                    placed, best, q, moved = trial, cost, cand, True
                    a0 = math.atan2(q["facing"][1], q["facing"][0])
        if not moved:
            break
    for pid in order:                               # a rug follows the group it lies under
        q = placed.get(pid)
        if q is not None and q["flat"] and ctx.mode(pid)[0] == "floor":
            cs = candidates(ctx, pid, {k: v for k, v in placed.items() if k != pid}, fixed)
            if cs:
                placed[pid] = cs[0]
    return placed


def solve(spec: dict, records: dict | None = None, data: dict | None = None, seed: int = 0,
          budget: int = 6000, top: int = 10, rules: dict | None = None) -> dict:
    """The layout of a place: ``ok`` with every hard constraint met, or the best found with ``unmet`` naming each
    constraint that could not be met. Same inputs and seed give the same answer. The search is a beam over
    enumerated candidates and then a polish (``layout_rules.json``); a depth-first search with backtracking is the
    fallback when the beam finds no room that meets every hard rule. ``solve_recover`` adds the recovery ladder."""
    data = data or load_data()
    records = records or {}
    if spec.get("kind") not in data["room_kinds"]:
        raise ValueError(f"unknown room kind {spec.get('kind')!r}")
    ctx = Ctx(spec, records, data, seed, rules)
    placed = {o["id"]: fire_piece(ctx, o) for o in ctx.openings if o["kind"] == "fireplace"}
    fixed = fixed_zones(ctx)
    order = order_of(ctx)
    left = [budget]
    how = "beam"

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        pid = order[k]
        for cand in candidates(ctx, pid, placed, fixed)[:top]:
            left[0] -= 1
            if left[0] < 0:
                return False
            placed[pid] = cand
            if (cand["flat"] or cand.get("on") or not path_problems(ctx, placed)) and dfs(k + 1):
                return True
            del placed[pid]
        if pid in ctx.optional:                     # (a picture piece with no room is left out)
            return dfs(k + 1)
        return False

    unmet = list(ctx.unmet)
    solved = False
    if not unmet:
        found = beam_search(ctx, order, placed, fixed)
        if found is not None:
            placed, solved = found, True
        else:
            how = "depth-first"
            solved = dfs(0)
        if solved:
            placed = polish(ctx, placed, order, fixed)
    if not solved:
        how = "best effort"
        for pid in list(placed):
            if pid in ctx.pieces:
                del placed[pid]
        if left[0] < 0:
            unmet.append("search budget reached before every constraint could be met together")
        for pid in order:                           # best effort, so the report names what fails
            cs = candidates(ctx, pid, placed, fixed)
            if not cs:
                cs = candidates(ctx, pid, placed, fixed, hard=False)
            if cs:
                placed[pid] = cs[0]
                for cand in cs[:40]:                # the first that leaves the paths open, when one does
                    placed[pid] = cand
                    if cand["flat"] or cand.get("on") or not path_problems(ctx, placed):
                        break
                else:
                    placed[pid] = cs[0]
            elif pid not in ctx.optional:
                unmet.append(f"piece {ctx.names[pid]} ({ctx.pieces[pid]['kind']}): no free position "
                             f"{'on a wall' if ctx.mode(pid)[0] == 'wall' else 'on the floor'}")
        for r in ctx.relations:
            ok = rel_ok(ctx, r, placed)
            if ok is False:
                unmet.append(f"relation not met: {rel_words(ctx, r)}")
            elif ok is None:
                unmet.append(f"relation not checked (an end is not placed): {rel_words(ctx, r)}")
        unmet += path_problems(ctx, placed)
    for phrase, i in (spec.get("named_places") or {}).items():
        if i not in placed and i not in ctx.by_id and i != "floor":      # ("floor": the open floor, no piece)
            unmet.append(f"named place {phrase!r}: {i} is not in the room")
    pks = pc.pockets(ctx, placed, ctx.R, stand_point)
    if solved:
        unmet += pc.pocket_problems(pks)
    for pid in sorted(ctx.optional - set(placed)):
        ctx.refused.append({"id": pid, "kind": ctx.pieces[pid]["kind"], "what": "refused",
                            "why": "no further room in its row" if ctx.pieces[pid].get("_row") else
                            "no place that keeps off the floor the plan's people need"})
    return export(ctx, placed, unmet, how, pks)


def _unnamed(spec: dict) -> list:
    """Pieces the text does not name: no plan phrase, no named place, no stated relation."""
    named = set((spec.get("named_places") or {}).values())
    tied = set()
    for r in spec.get("relations", []):
        if not (r.get("inferred") or r.get("stated") is False):
            tied.update(x for x in (r.get("a"), r.get("b"), r.get("c")) if x)
    return [p for p in spec.get("pieces", []) if not p.get("phrases") and p["id"] not in named
            and p["id"] not in tied]


def solve_recover(spec: dict, records: dict | None = None, data: dict | None = None, seed: int = 0,
                  rules: dict | None = None) -> dict:
    """``solve`` with the recovery ladder. Each rung gives up a little more and is logged with what it gave up;
    the first rung that yields a room meeting every hard rule wins. A piece the text names is never dropped, and
    a room that still fails comes back ``status: unresolved`` with the failures to put to the language model
    again, never as a quiet partial layout."""
    data0 = data or load_data()
    R0 = rules or pc.load_rules()
    lad = R0["ladder"]
    # the picture's own pieces settled first (doubles, not-on-the-floor, several of a thing): see prepare
    spec, records, notes = prepare(spec, records or {}, R0)
    spec_k, data_k, R_k = copy.deepcopy(spec), copy.deepcopy(data0), copy.deepcopy(R0)
    log, first = [], None

    def attempt(step: str, gave_up: str):
        lay = solve(spec_k, records, data_k, seed, rules=R_k)
        log.append({"step": step, "gave_up": gave_up, "ok": lay["ok"], "unmet": list(lay["unmet"])})
        return lay

    lay = attempt("as asked", "nothing")
    first = lay
    if not lay["ok"]:
        f, floor = lad["clearance_factor"], lad["clearance_floor_m"]
        for rule in data_k["piece_kinds"].values():
            if rule.get("front_clear_m", 0.0) > 0:
                rule["front_clear_m"] = max(min(floor, rule["front_clear_m"]), round(rule["front_clear_m"] * f, 3))
        R_k["hard"]["around_free_m"] = 0.0
        R_k["hard"]["door_approach_m"] = round(R_k["hard"]["door_approach_m"] * f, 3)
        lay = attempt("clearances shrunk to their floor", "comfortable space in front of and around pieces")
    if not lay["ok"]:
        inf = [r for r in spec_k.get("relations", []) if r.get("inferred") or r.get("stated") is False]
        if inf:
            R_k["weights"]["inferred"] = 0.0
            spec_k["relations"] = [r for r in spec_k["relations"] if r not in inf]
            lay = attempt("inferred relations dropped", f"{len(inf)} relation(s) the text does not state")
        else:
            log.append({"step": "inferred relations dropped", "gave_up": "nothing (the spec has none)",
                        "ok": False, "unmet": list(lay["unmet"]), "skipped": True})
    if not lay["ok"]:
        w = lad["band_widen"]
        for k, lim in data_k["relations"].items():
            for key in ("gap_m", "opening_gap_m", "side_m", "off_m"):
                if key in lim:
                    lim[key] = round(lim[key] * w, 3)
            if "angle_deg" in lim:
                lim["angle_deg"] = lim["angle_deg"] + lad["angle_widen_deg"]
        lay = attempt("stated relations' bands widened", f"bands x{w}, facing +{lad['angle_widen_deg']} degrees")
    if not lay["ok"]:
        sizes = [s for s in lad["size_order"] if s in data_k["room_kinds"][spec_k["kind"]]["sizes_m"]]
        cur = spec_k.get("size") if spec_k.get("size") in sizes else "medium"
        if cur in sizes and sizes.index(cur) + 1 < len(sizes):
            spec_k["size"] = sizes[sizes.index(cur) + 1]
            lay = attempt("room stepped up one size class", f"size {cur} -> {spec_k['size']}")
        else:
            log.append({"step": "room stepped up one size class", "gave_up": "nothing (already the largest)",
                        "ok": False, "unmet": list(lay["unmet"]), "skipped": True})
    dropped = []
    kept_k, kept_lay = copy.deepcopy(spec_k), lay
    gone_all = []
    if not lay["ok"]:
        rank = {k: i for i, k in enumerate(lad["drop_order"])}
        for _ in range(int(lad["max_drops"])):
            cand = sorted(_unnamed(spec_k), key=lambda p: (rank.get(p["kind"], len(rank)), p["id"]))
            if not cand:
                break
            gone = cand[0]
            dropped.append({"id": gone["id"], "kind": gone["kind"]})
            gone_all.append((gone, [r for r in spec_k.get("relations", [])
                                    if gone["id"] in (r.get("a"), r.get("b"), r.get("c"))]))
            spec_k["pieces"] = [p for p in spec_k["pieces"] if p["id"] != gone["id"]]
            spec_k["relations"] = [r for r in spec_k.get("relations", [])
                                   if gone["id"] not in (r.get("a"), r.get("b"), r.get("c"))]
            lay = attempt("an unnamed piece dropped", f"{gone['kind']} {gone['id']} (the text does not name it)")
            if lay["ok"]:
                break
        if lay["ok"] and len(gone_all) > 1:
            # the drop that made the room work was the last one: each piece dropped before it stands again if the
            # room still works with it (a hall lost both its tables on the way to dropping the one drape that was
            # in the way, and stood bare, 11-20)
            for back, rels in gone_all[:-1]:
                try_k = copy.deepcopy(spec_k)
                spec_k["pieces"] = spec_k["pieces"] + [back]
                spec_k["relations"] = list(spec_k.get("relations", [])) + rels
                lay2 = attempt("a dropped piece put back", f"{back['kind']} {back['id']}")
                if lay2["ok"]:
                    lay, dropped = lay2, [d_ for d_ in dropped if d_["id"] != back["id"]]
                else:
                    spec_k = try_k
        if not lay["ok"]:
            # a drop that did not make the room work gives nothing: the pieces stand again (the forge lost its
            # bench for a relation the bench had no part in)
            spec_k, lay, dropped = kept_k, kept_lay, []
            log.append({"step": "dropped pieces put back", "gave_up": "nothing (dropping them did not help)",
                        "ok": False, "unmet": list(lay["unmet"])})
    lay = dict(lay)
    lay["ladder"] = log
    lay["dropped"] = dropped
    lay["refused"] = notes + list(lay.get("refused") or [])
    if lay["ok"]:
        lay["status"] = "ok" if len(log) == 1 else "ok after recovery"
        lay["size_class"] = spec_k.get("size")
    else:
        lay["status"] = "unresolved"
        lay["reask"] = list(first["unmet"])          # the failures of the room as it was asked for
    return lay


def export(ctx: Ctx, placed: dict, unmet: list, how: str = "", pks: list | None = None) -> dict:
    pieces = []
    for pid in list(ctx.pieces) + [o["id"] for o in ctx.openings if o["kind"] == "fireplace"]:
        q = placed.get(pid)
        if q is None:
            continue
        rec = ctx.records.get(pid) or {}
        d = {"id": pid, "name": q["name"], "kind": q["kind"], "asset": q.get("asset") or rec.get("asset"),
             "position": r3(q["c"]), "facing": r3(q["facing"]),
             "yaw_deg": round(math.degrees(math.atan2(q["facing"][1], q["facing"][0])) + 90.0, 2),
             "size_m": q["size_m"], "rect": r3(q["rect"]), "flat": q["flat"], "seat": q["seat"],
             "look": (ctx.pieces.get(pid) or {}).get("look"), "generate": not (q.get("asset") or rec.get("asset"))}
        if (ctx.pieces.get(pid) or {}).get("called"):
            d["called"] = ctx.pieces[pid]["called"]
        if (ctx.pieces.get(pid) or {}).get("_copy_of"):
            d["copy_of"] = ctx.pieces[pid]["_copy_of"]         # (another of a picture piece: its one mesh)
        if pid in ctx.work:
            d["works_before"] = ctx.work[pid]
        for k in ("wall", "on", "on_z_m", "note"):
            if q.get(k) is not None:
                d[k] = q[k]
        sup = ctx.data["piece_kinds"].get(q["kind"], {}).get("support")
        if sup:
            d["support"] = sup
        pieces.append(d)
    ops = []
    for o in ctx.openings:
        d = {k: o[k] for k in ("id", "kind", "name", "wall", "at_m", "from_m", "to_m", "width_m", "point", "normal",
                               "stated") if k in o}
        for k in ("sill_m", "head_m"):
            if k in o:
                d[k] = o[k]
        if "swing" in o:
            d["swing"] = r3(o["swing"])
        if "breast" in o:
            d["breast"] = r3(o["breast"])
        ops.append(d)
    cost, terms, by_piece = objective(ctx, placed)
    for d in pieces:
        d["cost"] = by_piece.get(d["id"], {})
    pockets = [{"place": p["place"], "item": p["name"], "position": r3(p["pos"]), "facing": r3(p["facing"]),
                "shots": {k: {"valid": v["valid"], "of": v["total"], "points": v["points"][:12]}
                          for k, v in p["pocket"].items()}} for p in (pks or [])]
    return {"schema": "place_layout/1", "place": ctx.spec.get("place"), "kind": ctx.spec["kind"],
            "ok": not unmet, "unmet": unmet, "size_m": [ctx.W, ctx.D, ctx.H], "openings": ops, "pieces": pieces,
            "names": dict(ctx.names), "search": how, "cost": {"total": round(cost, 4), "terms": terms},
            "camera_pockets": pockets, "refused": list(ctx.refused)}


# ------------------------------------------------------------------ check, named places, set spec
def check_layout(layout: dict, spec: dict | None = None, data: dict | None = None) -> list:
    """Problems of a layout (solved here or written elsewhere): pieces outside the room or through each other, a
    piece in a door's swing or on a chimney breast, named places without a clear path to the door."""
    data = data or load_data()
    W, D, _ = layout["size_m"]
    gap_m, probs = data["piece_gap_m"], []
    solid = [p for p in layout["pieces"] if not p.get("flat") and not p.get("on")]
    for p in layout["pieces"]:
        r = p["rect"]
        if r[0] < -W / 2 - 1e-3 or r[2] > W / 2 + 1e-3 or r[1] < -D / 2 - 1e-3 or r[3] > D / 2 + 1e-3:
            probs.append(f"{p['name']}: outside the room")
    for i, p in enumerate(solid):
        for q in solid[i + 1:]:
            if overlap(p["rect"], q["rect"], 0.0) and gap(p["rect"], q["rect"]) < gap_m:
                probs.append(f"{p['name']} overlaps {q['name']}")
        for o in layout["openings"]:
            if o["kind"] == "door" and overlap(p["rect"], o["swing"]):
                probs.append(f"{p['name']}: in the swing of {o['name']}")
            if o["kind"] == "fireplace" and p["kind"] != "fireplace" and overlap(p["rect"], o["breast"]):
                probs.append(f"{p['name']}: on the chimney breast")
    named = (spec or {}).get("named_places") or {}
    by_id = {p["id"]: p for p in layout["pieces"]}
    ops = {o["id"]: o for o in layout["openings"]}
    pts, labels = [], []
    for o in layout["openings"]:
        if o["kind"] == "door":
            pts.append(stand_point({"opening_kind": "door", "c": o["point"], "facing": o["normal"]}, data))
            labels.append(o["name"])
    for phrase, i in named.items():
        if i == "floor":                     # (the open floor, or the open land: a place with no piece, as in solve)
            continue
        if i in by_id and not by_id[i].get("flat"):
            p = by_id[i]
            if p.get("on"):
                p = next((q for q in layout["pieces"] if q["name"] == p["on"]), p)
            pts.append(stand_point({"c": p["position"], "facing": p["facing"], "rect": p["rect"]}, data))
            labels.append(p["name"])
        elif i in ops:
            o = ops[i]
            pts.append(stand_point({"opening_kind": o["kind"], "c": o["point"], "facing": o["normal"]}, data)
                       if i not in by_id else stand_point({"c": by_id[i]["position"], "facing": by_id[i]["facing"],
                                                           "rect": by_id[i]["rect"]}, data))
            labels.append(o["name"])
        else:
            probs.append(f"named place {phrase!r}: {i} is not in the room")
    if len(pts) >= 2:
        rects = [p["rect"] for p in solid] + [o["breast"] for o in layout["openings"] if o["kind"] == "fireplace"]
        ok = reach(W, D, rects, pts, data["walkway_m"] / 2, snap=pc.load_rules()["hard"]["stand_snap_m"])
        if not ok[0]:
            probs.append(f"no clear floor at {labels[0]}")
        else:
            probs += [f"no clear path from {labels[0]} to {lab}" for lab, good in zip(labels[1:], ok[1:]) if not good]
    return probs


def named_places(spec: dict, layout: dict, data: dict | None = None) -> dict:
    """Each plan phrase -> the set item it names, with the item's place and where a person stands to be at it."""
    data = data or load_data()
    by_id = {p["id"]: p for p in layout["pieces"]}
    ops = {o["id"]: o for o in layout["openings"]}
    out = {}
    for phrase, i in (spec.get("named_places") or {}).items():
        if i in by_id:
            p = by_id[i]
            host = next((q for q in layout["pieces"] if q["name"] == p.get("on")), p)
            st = stand_point({"c": host["position"], "facing": host["facing"], "rect": host["rect"]}, data)
            out[phrase] = {"item": p["name"], "id": i, "kind": p["kind"], "position": p["position"],
                           "stand": r3(st), **({"support": p["support"]} if p.get("support") else {})}
        elif i in ops:
            o = ops[i]
            st = stand_point({"opening_kind": o["kind"], "c": o["point"], "facing": o["normal"]}, data)
            out[phrase] = {"item": o["name"], "id": i, "kind": o["kind"], "position": o["point"], "stand": r3(st)}
        else:
            out[phrase] = {"item": None, "id": i, "unresolved": True}
    return out


def write_named_places(path, spec: dict, layout: dict, data: dict | None = None) -> dict:
    m = named_places(spec, layout, data)
    Path(path).write_text(json.dumps(m, indent=1) + "\n")
    return m


def to_set_spec(spec: dict, layout: dict, data: dict | None = None) -> dict:
    """The part of a set spec (SETS.md) the layout decides: size, the four walls with their windows and doors, a
    chimney breast for each fireplace, and the fixtures and props with position and facing. Materials, lights
    and cameras are not laid out here; ``to_generate`` lists pieces that have no library asset yet."""
    data = data or load_data()
    W, D, H = layout["size_m"]
    walls = {s: {"kind": "wall", "side": s, "openings": []} for s in ("north", "south", "west", "east")}
    structure = []
    for o in layout["openings"]:
        if o["kind"] == "window":
            walls[o["wall"]]["openings"].append({"kind": "window_band", "name": o["name"], "from_m": o["from_m"],
                                                 "to_m": o["to_m"], "sill_m": o["sill_m"], "head_m": o["head_m"]})
        elif o["kind"] == "door":
            walls[o["wall"]]["openings"].append({"kind": "door", "name": o["name"], "from_m": o["from_m"],
                                                 "to_m": o["to_m"], "sill_m": o["sill_m"], "head_m": o["head_m"]})
        elif o["kind"] == "fireplace":
            b = o["breast"]
            structure.append({"kind": "box", "name": "chimney_breast" if o["name"] == "fireplace"
                              else f"chimney_breast_{o['name'].split('_')[-1]}",
                              "base": [round((b[0] + b[2]) / 2, 3), round((b[1] + b[3]) / 2, 3), 0],
                              "size_m": [round(b[2] - b[0], 3), round(b[3] - b[1], 3), H]})
    fixtures, props, gen = [], [], []
    for p in layout["pieces"]:
        item = {"asset": p["asset"], "name": p["name"], "position": list(p["position"]), "facing": list(p["facing"])}
        if p.get("look"):
            item["look"] = p["look"]
        if p.get("called"):
            item["called"] = p["called"]
        if p.get("on"):
            item["on"], item["on_z_m"] = p["on"], p["on_z_m"]
        if p.get("hang"):                        # (place_build.hang: fixed up on a wall, its lowest point at z_m)
            item["hang"], item["position"] = True, list(p["position"][:2]) + [p["z_m"]]
        (props if p.get("flat") or p.get("on") else fixtures).append(item)
        if p.get("generate"):
            gen.append({"name": p["name"], "id": p["id"], "kind": p["kind"], "look": p.get("look"),
                        "size_m": p["size_m"]})
    return {"size_m": [W, D, H], "structure": [walls[s] for s in ("north", "south", "west", "east")] + structure,
            "fixtures": fixtures, "props": props, "to_generate": gen,
            "named_places": {k: v.get("item") for k, v in named_places(spec, layout, data).items()}}


def missing_dressing(spec: dict, data: dict | None = None) -> list:
    """The usual pieces of this kind of room that the spec does not have (by kind or by set item name)."""
    data = data or load_data()
    pk = data["piece_kinds"]
    have = {p["kind"] for p in spec.get("pieces", [])} | {pk.get(p["kind"], {}).get("set_name") for p in
                                                           spec.get("pieces", [])}
    return [k for k in data["room_kinds"][spec["kind"]].get("dressing", []) if k not in have]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("records", nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    records = json.loads(Path(a.records).read_text()) if a.records else {}
    lay = solve_recover(spec, records, seed=a.seed)
    text = json.dumps({"layout": lay, "set": to_set_spec(spec, lay)}, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    else:
        print(text)
    sys.exit(0 if lay["ok"] else 2)


if __name__ == "__main__":
    main()
