"""A place spec (place_spec/1, written by the chapter's directing model) to a buildable set spec (stdlib).

    python tools/assets/sets/place_build.py PLACE_SPEC.json SET_ID [--library DIR] [--write]

1. the spec's pieces to library assets (places/pieces.py: lookup; a piece with no match is reported and left out:
   generation is a later step);
2. the layout in plan (sets/place_layout.py: openings, kind rules, stated relations, clearances) and its checks;
3. the structure, materials, window and door detail, window dressing and lights from the period's kit
   (sets/period_kits.json: a period's interior defaults, no layout);
4. the set spec in SETS.md's shape, its named places (every plan phrase -> a set item), and a report.
With --write, <library>/set/<SET_ID>/spec.json and named_places.json are written (the level is built by
build_set.py through the line, as for any set).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "places"))
sys.path.insert(0, str(HERE.parent))
import place_layout as L  # noqa: E402
import pieces as P  # noqa: E402

NORMAL = {"north": (0.0, -1.0), "south": (0.0, 1.0), "east": (-1.0, 0.0), "west": (1.0, 0.0)}   # into the room


def records_for(spec: dict, library) -> tuple:
    recs = P.load_records(library)
    # a piece made for another period does not furnish this one (a ship's cabin was offered mahogany): a record
    # whose tags or description name a period other than the place's is left out; one that names none serves any
    periods = set(json.loads((HERE.parent / "scenes" / "place_kinds.json").read_text())["PERIODS"])
    own = spec.get("period")
    if own in periods:
        def named(r):
            words = {str(t).lower() for t in r.get("tags") or []} | set(
                re.findall(r"[a-z_]+", (r.get("description") or "").lower().replace("far future", "far_future")))
            return periods & words
        recs = [r for r in recs if not named(r) or own in named(r)]
    res = P.resolve(spec.get("pieces") or [], recs)
    by = {r["asset_id"]: r for r in recs}
    # a piece's room on the floor is its mesh's own measure, where the record has it (a made piece's "size_m" is the
    # box it was asked for: a bunk made 1.8 m long was laid out as 3 m and found no place in its room)
    size = lambda r: [round(float(v), 3) for v in (r.get("dims_m") or r.get("size_m"))]   # noqa: E731
    rec = {p["id"]: {"asset": p["asset_id"], "size_m": size(by[p["asset_id"]])}
           for p in res["pieces"] if p["status"] == "library"}
    # a piece still to be generated stands in the room as the library's nearest of its kind until then (reported:
    # a bedroom was built without its bed because the bed's look matched no record closely enough)
    for p in res["pieces"]:
        if p["status"] == "generate" and p.get("nearest"):
            rec[p["id"]] = {"asset": p["nearest"]["asset_id"], "size_m": size(by[p["nearest"]["asset_id"]])}
    return rec, res


def dress(ss: dict, layout: dict, spec: dict, kit: dict) -> dict:
    """The set spec's structure, materials, window dressing and lights from the period's kit."""
    W, D, H = layout["size_m"]
    out = {"size_m": [W, D, H], "materials": kit["materials"], "exposure_ev100": kit.get("exposure_ev100", 5.0),
           "post": kit.get("post") or {}}
    st = [dict(kit["floor"])]
    for w in ss["structure"]:
        if w.get("kind") == "wall":
            w2 = dict(kit["wall"], **{k: v for k, v in w.items() if k != "openings"})
            w2["material"] = kit["wall_material_ns"] if w["side"] in ("north", "south") else kit["wall_material_ew"]
            ops = []
            for o in w.get("openings") or []:
                extra = kit["window"] if o["kind"] == "window_band" else kit["door"] if o["kind"] == "door" else {}
                ops.append(dict(extra, **{k: v for k, v in o.items() if k != "name"}))
            w2["openings"] = ops
            st.append(w2)
        elif w.get("kind") == "box" and w.get("name") == "chimney_breast":
            st.append(dict(w, material=kit.get("chimney_breast_material", kit["wall_material_ns"])))
        else:
            st.append(w)
    if spec.get("_roof") == "rafters" and "panel_raised" in kit["materials"]:
        # an open roof of timber (the concept shows beams overhead): rafters across the room's short side under a
        # ceiling of dark planks, instead of the period's plaster
        # a pitched timber roof on the walls' tops: a tie beam across the short side at each truss, two rafters up
        # to the ridge, a post under it; dark boards over them, above the ridge. No columns, no lattice (the first
        # try was a factory's trusses 0.4 m under a flat ceiling: a low red frame over everyone's head)
        span = min(W, D)
        rise = round(max(0.9, 0.32 * span), 2)
        st.append({"kind": "trusses", "span_axis": "x" if W <= D else "y", "spacing_m": 1.4, "bottom_m": round(H - 0.1, 2),
                   "top_m": round(H - 0.02, 2), "rise_m": rise, "chord_m": [0.18, 0.14], "web_m": 0.1,
                   "panel_m": round(span / 2, 2), "columns": False, "material": "panel_wood"})
        st.append(dict(kit["ceiling"], height_m=round(H + rise + 0.2, 2), material="panel_wood"))
    else:
        st.append(dict(kit["ceiling"], height_m=H))
    out["structure"] = st
    fixtures, props = list(ss["fixtures"]), list(ss["props"])
    lights = [dict(kit["lights"]["sky"])]
    wins = [o for o in layout["openings"] if o["kind"] == "window"]
    dress_by = {d["name"]: d["asset"] for d in kit.get("window_dressing") or []}
    for i, o in enumerate(wins):
        n = NORMAL[o["wall"]]
        p = o["point"]
        sfx = "" if i == 0 else f"_{i + 1}"
        # (a window the place's own picture shows is as the picture has it: the period kit's dressing is for the
        # windows a room is given unseen. A shed's one small window was hung with crimson damask drapes, 11-19)
        if "drapes" in dress_by and not str(o.get("id") or "").startswith(SEEN_OPENING):
            fixtures.append({"asset": dress_by["drapes"], "name": "drapes" + sfx,
                             "position": [round(p[0] + n[0] * 0.125, 3), round(p[1] + n[1] * 0.125, 3), 0.0],
                             "facing": list(n), "hang": True, "look": kit.get("drapes_look")})
        if "street_backdrop" in dress_by:
            fixtures.append({"asset": dress_by["street_backdrop"], "name": "street_backdrop" + sfx,
                             "position": [round(p[0] - n[0] * 9.5, 3), round(p[1] - n[1] * 9.5, 3), -6.0],
                             "facing": list(n), "outside": True})
        if not kit["lights"].get("window_rect"):
            continue                              # (a period whose windows give no daylight: a ship's viewport)
        lights.append(dict(kit["lights"]["window_rect"], position=[round(p[0] + n[0] * 0.1, 3), round(p[1] + n[1] * 0.1, 3), 1.75],
                           toward=[round(p[0] + n[0] * 3.5, 3), round(p[1] + n[1] * 3.5, 3), 1.45]))
    if kit["lights"].get("ceiling"):
        # a room lit from its ceiling (the period's own fitting): a grid over the floor, wider apart and stronger
        # the higher the ceiling (one fitting's light at the floor falls with the square of the height)
        cl = kit["lights"]["ceiling"]
        h0 = float(kit["ceiling"].get("height_m") or 2.4)
        gap = max(3.0, 1.2 * H)
        nx, ny = max(1, round(W / gap)), max(1, round(D / gap))
        for i in range(nx):
            for j in range(ny):
                lights.append(dict(cl, lumens=round(cl["lumens"] * (H / h0) ** 2),
                                   radius_m=round(max(cl.get("radius_m", 12), 2.5 * H), 1),
                                   position=[round((i + 0.5) * W / nx - W / 2, 2), round((j + 0.5) * D / ny - D / 2, 2),
                                             round(H - 0.12, 3)]))
    if wins and kit["lights"].get("sun"):       # daylight from the first window's side
        n = NORMAL[wins[0]["wall"]]
        lights.insert(0, dict(kit["lights"]["sun"], toward_sun=[round(-n[0] * 0.8 + n[1] * 0.3, 3),
                                                               round(-n[1] * 0.8 - n[0] * 0.3, 3), 0.35]))
    for f in fixtures:
        if (f.get("name") or "").startswith("fireplace"):
            fx, fy = f["facing"]
            dx, dy, dz = kit["lights"]["fire_offset_m"]
            lights.append(dict(kit["lights"]["fire"], position=[round(f["position"][0] - fx * dy, 3),
                                                              round(f["position"][1] - fy * dy, 3), dz]))
    # a piece that burns or glows (the director's "gives_light": a hearth standing in the room, a brazier) lights the
    # room round it from its top, as a fireplace does from its grate: a forge's hearth stood dark, 11-16
    lit = {p.get("id") for p in spec.get("pieces") or [] if p.get("gives_light") is True}
    for lp in layout.get("pieces") or []:
        if lp.get("id") in lit and lp.get("position"):
            top = float((lp.get("size_m") or [0, 0, 0.9])[2])
            lights.append(dict(kit["lights"]["fire"], position=[round(lp["position"][0], 3), round(lp["position"][1], 3),
                                                              round(top + 0.15, 3)]))
    # a piece people sit inside (a cabin) is lit inside: the room's light does not reach under its roof (the pilot
    # in his cockpit was staged in the dark)
    for it in fixtures + props:
        if (it.get("name") or "").startswith("vehicle_cabin"):
            lights.append(dict(kit["lights"]["lamp"], lumens=kit["lights"]["lamp"]["lumens"] * 3, shadows=False,
                               position=[it["position"][0], it["position"][1], 1.6]))
    for pr in props:
        if "lamp" in (pr.get("asset") or "") or (pr.get("name") or "").startswith("lamp"):
            lights.append(dict(kit["lights"]["lamp"], position=[pr["position"][0], pr["position"][1],
                                                               round((pr.get("on_z_m") or 0.0) + 0.42, 3)]))
    props = [pr for pr in props if pr.get("asset")]   # a piece with no library asset waits for generation (report)
    fixtures = [f for f in fixtures if f.get("asset")]
    out.update(fixtures=fixtures, props=props, lights=lights)
    cams = []
    for i, (sx, sy) in enumerate(((1, -1), (-1, -1), (1, 1), (-1, 1))):
        cams.append({"name": f"wide_{i}", "position": [round(sx * (W / 2 - 0.5), 2), round(sy * (D / 2 - 0.5), 2), 1.6],
                     "target": [round(-sx * 0.6, 2), round(-sy * 0.6, 2), 1.1], "focal_mm": 20})
    out["cameras"] = cams
    out["validation"] = {"figures": [{"position": [0.0, 0.0], "facing": [0, -1]}]}
    return out


def room_words(out: dict, lay: dict, kit: dict, recs: dict) -> tuple:
    """What the built room holds, and its surfaces, in words for a picture of it (the redraw's room sentence): each
    placed thing by its own library record's opening words, each opening and the surfaces by the period kit's."""
    kw = kit.get("words") or {}
    kind = (out.get("tags") or ["room"])[0] or "room"
    held = []
    for it in (out.get("fixtures") or []) + (out.get("props") or []):
        if it.get("outside") or it.get("hang") or any((it.get("name") or "").startswith(k) for k in kw):
            continue                              # (a window's dressing and an opening are in the kit's own words)
        rc = recs.get(it["asset"]) or {}
        d = re.split(r"[,:;.]", rc.get("description") or "", maxsplit=1)[0].strip()
        if rc.get("piece_kind") and rc.get("look"):          # a piece made on request: its kind and its own look
            d = f"{rc['piece_kind'].replace('_', ' ')} ({rc['look']})"
        if it.get("called"):                      # what the story calls it, not its kind's name ("a landmark (...)")
            rest = str(rc.get("look") or "").strip()
            rest = rest[len(it["called"]):].lstrip(" ,;") if rest.lower().startswith(it["called"].lower()) else rest
            d = it["called"] + (f" ({rest})" if rest else "")
        d = d or (it.get("name") or "").replace("_", " ")
        if out.get("style") and d.lower().startswith(out["style"].lower() + " "):
            d = d[len(out["style"]) + 1:]         # (the period is said once, for the room)
        d = d[0].lower() + d[1:] if d[:2] != "A " else "a" + d[1:]
        held.append(d if re.match(r"(a|an|the) ", d) else ("an " if d[:1] in "aeiou" else "a ") + d)
    for o in lay.get("openings") or []:
        w = kw.get(o.get("kind"))
        if str(o.get("id") or "").startswith(SEEN_OPENING):
            w = ("an " if str(o.get("kind"))[:1] in "aeiou" else "a ") + str(o.get("kind"))
        if w and w not in held:
            held.append(w)
    held = list(dict.fromkeys(held))
    return (f"A {(out.get('style') or '').replace('_', ' ')} {kind.replace('_', ' ')}: ".replace("  ", " ").replace("A v", "A V")
            + "; ".join(held) + ".", kw.get("surfaces") or "")




WALL_MIN_M = 2.7                 # a measured room's walls are at least this high
SEEN_OPENING = "oc"              # the id of an opening taken from the place's concept picture begins with this
OPENING_KINDS = ("door", "window", "fireplace")      # (a place's openings are its walls', never a piece to stand in it)


def with_concept(spec: dict, concept: dict) -> dict:
    """The place spec checked against its concept picture's facts (scenes/concept.py). A piece the plan never names
    (no phrase of the plan lands on it) and the picture does not show is dropped: it was a guess. A piece in the
    picture takes its look there. A place the picture shows as a cockpit (a space people only sit in) is one built
    thing: a single cabin piece drawn from the concept, on a stage with room for the cameras round it, every phrase
    of the plan landing on it."""
    sp = json.loads(json.dumps(spec))
    chk = concept.get("spec") or {}
    kept = []
    for p in sp.get("pieces") or []:
        c = chk.get(p["id"]) or {}
        if not p.get("phrases") and not c.get("in_picture"):
            continue
        if p.get("kind") in ("work_piece", "landmark") and str(p.get("look") or "").strip():
            # what the story calls it (its look begins with its own name): its name in every picture's words. Named
            # "the landmark" with the judge's words for its shape, a cave mouth was drawn as a built stone doorway
            p["called"] = str(p["look"]).split(",")[0].strip().rstrip(".").lower()
        if c.get("words") and p.get("kind") == "landmark":
            p["look"] = c["words"]           # (a thing of the open: what it is, whole, as the picture shows it)
        elif c.get("look"):
            # (a kind that holds any thing, a work piece or a landmark, is what its own words say: the picture adds
            # its material and colour. A hearth made from "stone, grey" was a block with a pyramid on it, 11-18)
            own = str(p.get("look") or "").strip().rstrip(".")
            p["look"] = f"{own}; {c['look']}" if p.get("kind") in ("work_piece", "landmark") and own \
                and c["look"] not in own else c["look"]
        if c.get("crop"):                    # its own part of the concept picture: the drawing it is shaped from
            p["source"] = c["crop"]
        kept.append(p)
    for p in kept:                           # a bed added for a place people lie in is the picture's bed again
        if not p.get("source"):
            twin = next((q for q in kept if q["kind"] == p["kind"] and q.get("source")), None)
            if twin and p["id"] not in chk:
                p.update(source=twin["source"], look=twin.get("look") or p.get("look"))
    # what the picture shows that the spec does not have is built too (one of each kind the spec lacks)
    have = {p["kind"] for p in kept}
    for i, sn in enumerate(concept.get("seen") or []):
        if f"seen:{i}" in ((concept.get("depth") or {}).get("off_floor") or {}):
            continue                         # (it hangs on a wall or from the roof: no piece to stand on the floor)
        # (in the open every large thing standing on the ground is built, unless a piece of the spec is that thing:
        # mostly the same part of the picture)
        # (the same thing, not a small thing standing before a large one: of like size too)
        twin = sn.get("kind") == "landmark" and sn.get("box") and not any(
            _same_part(sn["box"], (chk.get(q["id"]) or {}).get("box"), like=OPEN_LIKE) for q in kept) and not any(
            _same_part(sn["box"], o.get("box"), like=OPEN_LIKE) for o in (concept.get("seen") or [])[:i]
            if o.get("kind") == "landmark") and sum(1 for q in kept if q["id"].startswith("c")) < OPEN_SEEN_MOST
        # (a seen thing that is the same part of the picture as a piece of the spec is that piece, whatever kind
        # the judge gave it: a block the spec's stone was found in was built a second time on the same spot, 11-19)
        if sn.get("box") and any(_same_part(sn["box"], (chk.get(q["id"]) or {}).get("box"), like=0.5) for q in kept
                                 if (chk.get(q["id"]) or {}).get("in_picture")):
            continue
        if sn.get("kind") and sn.get("crop") and (sn["kind"] not in have or twin or sn.get("group")) and sn["kind"] not in OPENING_KINDS:
            have.add(sn["kind"])
            kept.append({"id": f"c{i + 1}", "kind": sn["kind"], "phrases": [],
                         "look": sn.get("words") or "; ".join(x for x in (sn["name"], sn.get("look")) if x),
                         "against": "not stated", "source": sn["crop"], **({"natural": True} if sn.get("natural") else {}),
                         # (a kind that holds any thing is called what the picture's judge named it)
                         **({"called": seen_called(sn)} if sn["kind"] in ("work_piece", "landmark") and seen_called(sn) else {})})
    sp["pieces"] = kept
    ids = {p["id"] for p in kept}
    sp["relations"] = [r for r in sp.get("relations") or [] if r.get("a") in ids and r.get("b") in ids]
    dp = concept.get("depth") or {}
    if dp.get("room_m") and concept.get("space") == "room":
        # the room as the concept picture shows it (img3d/concept_depth.py): its size, and each piece the picture
        # shows where it stands there, against the wall it stands against (11-17: the forge was laid out as a hall
        # 2.8 m wide, the hearth in the middle of the floor, from the kind's table and free spots)
        sp["_room_m"] = list(dp["room_m"])
        # (the picture's top edge cuts the room off: what it measures overhead is the least the walls can be)
        sp["_room_m"][2] = max(float(sp["_room_m"][2]), WALL_MIN_M)
        seen_key = {f"c{i + 1}": f"seen:{i}" for i in range(len(concept.get("seen") or []))}
        for p in kept:
            t = (dp.get("things") or {}).get(f"spec:{p['id']}") or (dp.get("things") or {}).get(seen_key.get(p["id"], ""))
            if t:
                p.update(at_xy=t["xy"], facing_xy=t["facing"], against="wall" if t.get("against") else "free")
                if size_agrees(p.get("longest_m"), t["size_m"]):
                    p["_seen_size_m"] = t["size_m"]
                else:
                    # the picture's thing is far from the size the story gives it: the story's thing is made from
                    # its own words at its own size, where the picture has it (a stone "the size of a sleeping dog",
                    # 0.8 m, was shaped from the picture's squared block 1.4 m long and a metre high, 11-19)
                    p.pop("source", None)
    if concept.get("space") == "room" and sp.get("kind") != "outdoors":
        # an opening the picture shows and the spec states none of is the room's (a hall drawn with tall windows down
        # both walls was built a sealed box with its kind's one door, 11-19): one of each kind, on the wall the
        # layout finds for it
        stated = {o.get("kind") for o in sp.get("openings") or []}
        for i, sn in enumerate(concept.get("seen") or []):
            if sn.get("kind") in OPENING_KINDS and sn.get("box") and sn["kind"] not in stated:
                stated.add(sn["kind"])
                sp.setdefault("openings", []).append({"id": f"{SEEN_OPENING}{i + 1}", "kind": sn["kind"], "wall": "not stated",
                                                      "words": "; ".join(x for x in (sn.get("name"), sn.get("look")) if x)})
    # (the roof's timbers, in the judge's words for the overhead or among the things it saw)
    if re.search(r"\b(beam|beams|rafter|rafters|timber|timbers|roof|thatch)\b", str((concept.get("surfaces") or {}).get("ceiling") or ""), re.I) \
            or any(re.search(r"\b(beams?|rafters?)\b", str(x.get("name") or ""), re.I) for x in concept.get("seen") or []):
        sp["_roof"] = "rafters"              # the picture's overhead: an open roof on timber, not a plastered ceiling
    if concept.get("space") == "cockpit":
        seen = [x["name"] + (f" ({x['look']})" if x.get("look") else "") for x in concept.get("seen") or []
                if x.get("name") and "view" not in x["name"]]
        # (what the place is, in the director's own words and its period: the kind's name alone drew any cabin)
        inside = re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or spec.get("room") or ""),
                        flags=re.I).strip().rstrip(".")
        inside = f"{str(spec.get('period') or '').replace('_', ' ')}: {inside}".strip(": ")
        sf = concept.get("surfaces") or {}                   # (its surfaces as the picture shows them)
        walls = "; ".join(f"the {k}: {str(sf[k]).strip().rstrip('.')}" for k in ("walls", "floor") if sf.get(k))
        sp.update(kind="cockpit", openings=[], relations=[], pieces=[{
            "id": "p1", "kind": "vehicle_cabin", "look": "; ".join(x for x in (inside, ", ".join(dict.fromkeys(seen)), walls, str(spec.get("_style") or "")) if x),
            "phrases": [ph for p in spec.get("pieces") or [] for ph in p.get("phrases") or []], "against": "free",
            "source": concept.get("concept"), "seats_abreast": int(concept.get("seats_abreast") or 1)}])
        sp["named_places"] = {ph: "p1" for ph in spec.get("named_places") or {}}
        # what lies outside the cabin: the director's own line for the place when it is an outside place
        m = re.match(r"\s*outside\s*:\s*(.+)", str(spec.get("_director_room") or ""), re.I | re.S)
        if m:
            sp["outside"] = m.group(1).strip().rstrip(".")
        elif str(spec.get("outside") or "").strip():         # (else what the place spec says is seen outside it)
            sp["outside"] = str(spec["outside"]).strip().rstrip(".")
    if concept.get("space") == "outdoors" or spec.get("kind") == "outdoors":
        sp = outdoors(sp, spec, concept)
    sp["_concept"] = {k: concept.get(k) for k in ("concept", "space", "surfaces_words", "bare", "surfaces")}
    return sp


def seen_called(sn: dict) -> str:
    """What a thing the picture shows is called in a picture's words: the head of the judge's words for its own
    part of the picture (looked at alone: the surer of its two readings), else the name it was listed by; things
    read together as one are the buildings."""
    if sn.get("group"):
        return "buildings"
    w = re.split(r"[,.;:]|\b(?:with|topped|that|which|standing|made|forming)\b", str(sn.get("words") or ""), maxsplit=1)[0]
    w = re.sub(r"^(a|an|the)\s+", "", w.strip(), flags=re.I).strip().lower()
    # (a head of one word is as often its first adjective as its noun: "dark, gnarled ...")
    return w if 2 <= len(w.split()) <= 5 else str(sn.get("name") or "").strip().lower()


SIZE_OFF = 1.6                   # the picture's thing this many times longer or shorter than the story's is not it


def size_agrees(stated, seen) -> bool:
    """Whether a thing as the concept picture measures it (seen: its size in metres) can be the piece whose longest
    side the story states; a piece with no stated size agrees with any."""
    try:
        st, sn = float(stated), max(float(v) for v in seen)
    except (TypeError, ValueError):
        return True
    return st <= 0 or sn <= 0 or (st / SIZE_OFF <= sn <= st * SIZE_OFF)


def _same_part(a, b, k: float = 0.6, like: float = 0.0) -> bool:
    """Two boxes of a picture that are mostly the same part of it (their overlap over the smaller one's area), and,
    with `like`, of like size (the smaller at least that much of the larger)."""
    if not a or not b:
        return False
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return w > 0 and h > 0 and w * h >= k * min(aa, ab) and min(aa, ab) >= like * max(aa, ab)


OPEN_FAR_M = 32.0                # a thing farther than this from the picture's eye, or larger than OPEN_BIG_M, is
OPEN_BIG_M = 30.0                # part of what lies beyond the set (its panorama), not a piece on it
OPEN_FRONT_M = 4.0               # the set's middle: open ground this far in front of the nearest piece
OPEN_ROUND_M = 6.0               # ... and the stage reaches this far beyond its pieces on every side
OPEN_STAGE_M = (16.0, 70.0)      # the stage's side, least and most
OPEN_LIKE = 0.4                  # two boxes are one thing seen twice only when the smaller is this much of the larger
OPEN_SEEN_MOST = 8               # the things of the picture built besides the plan's own, at most
OPEN_SCALE = (0.6, 3.0)          # how far the picture's measured metres may be corrected by its things' real sizes


def open_scale(pcs: list, concept: dict) -> float:
    """The measured picture's metres corrected by how tall its things really are (the judge's "real_m" for each,
    scenes/concept.py): the middle one of their ratios, within OPEN_SCALE. A whole smithy measured 4 m wide and 3 m
    tall, and was built so (11-18)."""
    things = (concept.get("depth") or {}).get("things") or {}
    chk, seen = concept.get("spec") or {}, concept.get("seen") or []
    rs = []
    for p in pcs:
        i = int(p["id"][1:]) - 1 if p["id"].startswith("c") and p["id"][1:].isdigit() else -1
        src = chk.get(p["id"]) or (seen[i] if 0 <= i < len(seen) else {})
        t = things.get(f"spec:{p['id']}") or things.get(f"seen:{i}")
        if t and src.get("real_m") and t["size_m"][2] > 0.2:
            rs.append(float(src["real_m"][1]) / float(t["size_m"][2]))
    if not rs:
        return 1.0
    rs.sort()
    return round(min(OPEN_SCALE[1], max(OPEN_SCALE[0], rs[len(rs) // 2])), 3)


def open_places(pcs: list, concept: dict) -> tuple:
    """Where the pieces of a place in the open stand, from its concept picture measured (img3d/concept_depth.py):
    each one's place and size as the picture has them, about a middle on the open ground in front of them, its front
    toward that ground. -> (the pieces that are built, the stage's size or None). A piece the picture does not
    place stands where the layout finds room, as before."""
    dp = concept.get("depth") or {}
    things, eye = dp.get("things") or {}, (dp.get("eye") or [0.0, 0.0])[:2]
    seen_key = {f"c{i + 1}": f"seen:{i}" for i in range(len(concept.get("seen") or []))}
    at, out = {}, []
    sc = open_scale(pcs, concept)
    for p in pcs:
        t = things.get(f"spec:{p['id']}") or things.get(seen_key.get(p["id"], ""))
        if not t:
            out.append(p)
            continue
        t = dict(t, size_m=[float(v) * sc for v in t["size_m"]])
        x, y = (t["xy"][0] - eye[0]) * sc, (t["xy"][1] - eye[1]) * sc
        far = math.hypot(x, y) > OPEN_FAR_M or max(t["size_m"]) > OPEN_BIG_M
        if far and not p.get("phrases"):
            continue                         # (it lies beyond the set: the panorama's)
        k = min(1.0, OPEN_FAR_M / max(math.hypot(x, y), 1e-6))
        size = [min(float(v), OPEN_BIG_M) for v in t["size_m"]]
        at[p["id"]] = (x * k, y * k, size)
        out.append(p)
    if not at:
        return out, None
    ox = sum(v[0] for v in at.values()) / len(at)
    oy = min(v[1] - v[2][1] / 2 for v in at.values()) - OPEN_FRONT_M
    half = 0.0
    for p in out:
        if p["id"] in at:
            x, y, size = at[p["id"]]
            p.update(at_xy=[round(x - ox, 3), round(y - oy, 3)], facing_xy=[0.0, -1.0], _seen_size_m=[round(v, 3) for v in size])
            half = max(half, abs(x - ox) + size[0] / 2, abs(y - oy) + size[1] / 2)
    side = round(min(OPEN_STAGE_M[1], max(OPEN_STAGE_M[0], 2 * (half + OPEN_ROUND_M))), 1)
    tall = max([v[2][2] for v in at.values()] + [6.0])
    return out, [side, side, round(tall, 1)]


def technique(style: str) -> str:
    """A book's art style with only its way of painting left (as world.set_picture_prompt): the clause before its
    colon, without the names of building styles. Its other words are things to draw: given whole to a panorama of
    bare far land, "cathedral-like stone and iron, candlelit" stood a cathedral with candles on the open ground,
    11-18."""
    st = (style.split(":", 1)[0] if ":" in (style or "") else (style or "")).strip().rstrip(".")
    return re.sub(r"(?i)\b(gothic|baroque|romanesque|victorian|byzantine|art deco|art nouveau|rococo|"
                  r"neo-?classical|renaissance|medieval)\b\s*", "", st).strip(" ,")


ABSENT = re.compile(r"^\s*(there\s+(is|are)\s+(no|not|nothing)\b|no\s|nothing\s|it\s+(has|holds)\s+no\b)", re.I)


def present_only(text: str) -> str:
    """A place's words with the sentences that say what is absent left out: a picture model draws what it is told
    of, absent or not ("There are no buildings, tools, or standing objects on the ground" stood in every key's
    words of a bare summit, and its keys had a carved stone doorway, 11-19)."""
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return " ".join(x for x in parts if x and not ABSENT.match(x)).strip()


def outdoors(sp: dict, spec: dict, concept: dict) -> dict:
    """A place in the open (its concept picture shows it outdoors, or its spec says so) is built as a world: the
    land round where its people stand, the sky, and a flat patch of ground under them, all one piece made from a
    panorama of the place drawn from the concept picture (pieces "terrain", made_as "world"); the pieces the plan
    names stand free on that ground. No walls, openings or views outside; a phrase that lands on no piece is the open
    ground."""
    sp = dict(sp, kind="outdoors", openings=[], outside="")
    sp.pop("_views", None)
    land = re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or spec.get("room") or ""),
                  flags=re.I).strip().rstrip(".")
    concept = dict(concept, land=present_only(concept.get("land")))
    if str(concept.get("land") or "").strip():     # (its picture was drawn anew as the whole place: those words)
        land = f"{str(concept['land']).strip().rstrip('.')}. With {land[0].lower() + land[1:]}" if land else str(concept["land"]).strip().rstrip(".")
    land = f"{str(spec.get('period') or '').replace('_', ' ')}: {land}".strip(": ") if spec.get("period") != "not stated" \
        else land
    sf = concept.get("surfaces") or {}
    said = "; ".join(f"{k}: {str(v).strip().rstrip('.')}" for k, v in (("the ground", sf.get("floor")),
                                                                         ("overhead", sf.get("ceiling")),
                                                                         ("the light", concept.get("light"))) if v)
    # what the plan sends people to and what the place's picture shows standing on its ground is built to stand on
    # the set's own ground, each where the picture has it (open_places); what lies far off is the panorama's (a far
    # mountain made a piece stood 80 m tall on the stage)
    pcs = [dict(p, against="free") for p in sp.get("pieces") or [] if p.get("kind") not in ("terrain",)
           and (p.get("phrases") or p.get("source"))]
    # (a landmark the plan names that the place's picture does not show is not made from its words alone: the
    # picture is the place, and people sent there stand on the open ground. Two such were a boulder with a hole
    # and a cube of rock, 11-18)
    # everything built there as one thing (the concept reader's "group"): it stands for every landmark of the
    # plan and of the picture, and what the plan sends people to at any of them is at it
    gi = next((i for i, sn in enumerate(concept.get("seen") or []) if sn.get("group")), None)
    grp = next((p for p in pcs if gi is not None and p["id"] == f"c{gi + 1}"), None)
    gone = {}
    # (the land's own things, a boulder or a bush, are not built: they stand apart, each where the picture has it)
    chk = concept.get("spec") or {}
    nat = {p["id"] for p in pcs if p.get("natural") or (chk.get(p["id"]) or {}).get("natural")}
    if grp:
        gone = {p["id"]: grp["id"] for p in pcs if p is not grp and p["id"] not in nat}
        grp["phrases"] = list(dict.fromkeys(ph for p in pcs if p["id"] not in nat for ph in p.get("phrases") or []))
        pcs = [grp] + [p for p in pcs if p["id"] in nat]
    pcs = [p for p in pcs if not (p.get("kind") == "landmark" and not p.get("source")
                                  and p["id"] in chk and not chk[p["id"]].get("in_picture"))]
    pcs, stage = open_places(pcs, concept)
    if stage:
        sp["_room_m"] = stage
    ids = {p["id"] for p in pcs}
    # the part of the land the plan sends people to (the director's land_places, the first the shots name): the
    # panorama is drawn with it close in front of the eye, and the stage is set there (world_stage)
    # (the world is what lies round the set and beyond it: its real ground with the panorama far off. No part of
    # the land is drawn close in the panorama and no thing is looked for in it: a landmark is a piece on the set,
    # shaped from its own part of the concept picture)
    # the part of the land the plan sends people to (the director's land_places, the first the shots name: a
    # water's edge, a path): the panorama is drawn with it close by, on the ground, and the stage is set there
    lp = spec.get("land_places") if isinstance(spec.get("land_places"), dict) else {}
    order = [ph for ph in (spec.get("_phrases") or []) if ph in lp] + [ph for ph in lp if ph not in (spec.get("_phrases") or [])]
    spot = str(lp[order[0]]).strip().rstrip(".") if order else ""
    # the panorama is drawn from what the picture shows far off, its ground and its sky, in words alone: with the
    # picture as its reference, or the place's own words, it had the place's buildings in it again, close by
    far = str(concept.get("far") or "").strip().rstrip(".")
    per = str(spec.get("period") or "").replace("_", " ")
    far_look = "; ".join(x for x in (
        f"{per}: {far}".strip(": ") if per != "not stated" else far,
        "; ".join(f"{k}: {str(v).strip().rstrip('.')}" for k, v in (("the ground", sf.get("floor")), ("overhead", sf.get("ceiling"))) if v),
        (f"painted as a {technique(str(spec.get('_style') or ''))}" if technique(str(spec.get("_style") or "")) else "")) if x) \
        if far else ""
    # the place's own ground seen from straight above (places/ground_map.py), drawn from the whole land's words and
    # its ground's, the parts of the land the plan sends people to on it, the concept picture as the land's look
    ground = "; ".join(x for x in (land, f"the ground: {str(sf['floor']).strip().rstrip('.')}" if sf.get("floor") else "") if x)
    pcs.append({"id": "t1", "kind": "terrain", "against": "free", "phrases": [],
                "look": far_look or "; ".join(x for x in (land, said, str(spec.get("_style") or "")) if x),
                **({"spot": spot} if spot else {}), "land_words": land, "surface_words": said,
                **({"source": concept["concept"]} if concept.get("concept") and not far_look else {}),
                **({"map_words": ground, "land_places": {ph: str(lp[ph]).strip().rstrip(".") for ph in order},
                    "map_style": technique(str(spec.get("_style") or ""))} if ground else {}),
                **({"map_source": concept["concept"]} if ground and concept.get("concept") else {})})
    sp["pieces"] = pcs
    sp["relations"] = [r for r in sp.get("relations") or [] if r.get("a") in ids and r.get("b") in ids]
    sp["named_places"] = {ph: (gone.get(v, v) if gone.get(v, v) in ids else "floor")
                          for ph, v in (spec.get("named_places") or {}).items()}
    return sp


VIEW_SIZE_M = (48.0, 24.0)       # a view's picture: wide enough to fill a canopy from the seats
VIEW_AHEAD_M = 12.0              # how far beyond the cabin it stands
VIEW_EMIT = 30.0                 # the picture's own light


def view_id(words: str) -> str:
    import hashlib
    return "backdrop_view_" + hashlib.sha1(" ".join(re.findall(r"[a-z0-9]+", words.lower())).encode()).hexdigest()[:6]


def view_spec(words: str) -> dict:
    """The source spec of a view's backdrop (models/backdrop_view): one picture of what lies outside, its own light."""
    w, h = VIEW_SIZE_M
    return {"id": view_id(words), "class": "prop", "tier": "background", "texture_px": 2048, "placement": "floor",
            "description": f"Backdrop for the view outside a place: {words}. One flat picture, {w:g} x {h:g} m, giving "
                           "its own light. Front -Y; origin at the middle of the bottom edge.",
            "tags": ["backdrop", "view", "generated-on-request"], "size_m": [w, 0.02, h], "size_tolerance": 0.05,
            "source": "scripted", "backdrop": {"w": w, "h": h, "emit": VIEW_EMIT}, "view_words": words,
            "licence": "CC0 (original script; picture drawn by the local image model)",
            "cameras": [{"name": "front", "position": [0.0, -w * 1.2, h / 2], "target": [0.0, 0.0, h / 2], "focal": 28}]}


# a floor as a concept picture's words name it -> the surface it is built with (its colour is the picture's own)
FLOOR_OF = {
    "stone|flag|slab|cobble|paving|paved|rock": {"preset": "worn_concrete", "panel_m": [0.7, 0.55, 0], "joint_m": 0.025,
                                                  "joint_darken": 0.45, "variation": 0.12, "grime": 0.4,
                                                  "roughness": [0.7, 0.95]},
    "earth|dirt|soil|mud|clay|sand": {"preset": "worn_concrete", "panel_m": [0, 0, 0], "variation": 0.15, "grime": 0.5},
}
FACE_M = 3.0                     # a wall face's panel: this wide and this high


def face_spec(words: str) -> dict:
    """The source spec of a wall's face (models/backdrop_view, no light of its own): the place's own wall surface,
    drawn flat from the words its concept picture's walls were read into."""
    import hashlib
    fid = "wall_face_" + hashlib.sha1(" ".join(re.findall(r"[a-z0-9]+", words.lower())).encode()).hexdigest()[:6]
    return {"id": fid, "class": "prop", "tier": "background", "texture_px": 2048, "placement": "floor",
            "description": f"The face of a wall, one flat picture {FACE_M:g} x {FACE_M:g} m: {words} Front -Y; origin "
                           "at the middle of the bottom edge.",
            "tags": ["wall", "face", "generated-on-request"], "size_m": [FACE_M, 0.02, FACE_M], "size_tolerance": 0.05,
            "source": "scripted", "backdrop": {"w": FACE_M, "h": FACE_M, "emit": 0.0}, "wall_words": words,
            "licence": "CC0 (original script; picture drawn by the local image model)",
            "cameras": [{"name": "front", "position": [0.0, -FACE_M * 1.6, FACE_M / 2], "target": [0.0, 0.0, FACE_M / 2],
                         "focal": 28}]}


def wall_faces(out: dict, lay: dict, spec: dict, library) -> None:
    """The room's walls wear the concept's own wall: its face panels side by side along every wall, a hand's
    breadth inside it, left off where a door, window or fireplace stands (there the period's wall shows)."""
    import config as C
    words = (((spec.get("_concept") or {}).get("bare") or {}).get("walls") or {}).get("words")
    if not words or not any(w.get("kind") == "wall" for w in out.get("structure") or []):
        return
    fs = face_spec(words)
    out["face"] = fs
    if not (Path(library or C.LIBRARY) / "prop" / fs["id"] / "asset.json").exists():
        return
    W, D, H = out["size_m"]
    gaps = {}
    for o in lay.get("openings") or []:
        if o.get("wall") and o.get("point"):
            gaps.setdefault(o["wall"], []).append((o["point"], float(o.get("width_m") or 1.2)))
    for side, n in NORMAL.items():
        L_ = W if side in ("north", "south") else D
        cols = max(1, math.ceil(L_ / FACE_M))
        for i in range(cols):
            t = (i + 0.5) * L_ / cols - L_ / 2                       # along the wall
            if side in ("north", "south"):
                pos = [t, -n[1] * (D / 2 - 0.015)]
            else:
                pos = [-n[0] * (W / 2 - 0.015), t]
            if any(math.dist(pos, g[0][:2]) < FACE_M / 2 + g[1] / 2 for g in gaps.get(side, [])):
                continue
            for r in range(max(1, math.ceil(H / FACE_M))):
                out["fixtures"].append({"asset": fs["id"], "name": f"wall_face_{side}_{i}_{r}", "hang": True,
                                        "surface": True, "position": [round(pos[0], 3), round(pos[1], 3), r * FACE_M],
                                        "facing": [n[0], n[1]]})


def cabin_stage(out: dict, spec: dict, library) -> None:
    """A cabin with something outside it is filmed in the open, not in a room: no stage walls, ceiling or floor (they
    were all the canopy showed), the view's backdrop standing beyond its nose, and its light coming from there:
    a soft key through the canopy in the view's own colour, and the glow of its instruments on the people in it."""
    import config as C
    lib = Path(library or C.LIBRARY)
    cab = next((f for f in out["fixtures"] + out["props"] if (f.get("name") or "").startswith("vehicle_cabin")), None)
    words = spec.get("outside")
    if not cab or not words:
        return
    rec = json.loads((lib / "prop" / cab["asset"] / "asset.json").read_text())
    f = [float(v) for v in cab.get("facing", [0, -1])]     # the builder turns the piece's own front onto its facing
    x, y = cab["position"][:2]
    ln, ht = float(rec["size_m"][1]), float(rec["size_m"][2])
    vid = view_id(words)
    out["structure"] = []
    # a backdrop for each thing the story sees outside (place_spec's views: the director's account, shot by shot),
    # one behind the other beyond the nose; a still shows its own shot's (scenes/ue/scene_job.py hides the rest).
    # With no account, the one general view.
    per_shot = {k: v for k, v in (spec.get("_views") or {}).items() if str(v).strip()}
    each = list(dict.fromkeys(per_shot.values())) or [words]
    names = {}
    for n, vw in enumerate(each):
        if not (lib / "prop" / view_id(vw) / "asset.json").exists():
            continue
        d = ln / 2 + VIEW_AHEAD_M + 0.5 * n
        names[vw] = "outside_view" if not names else f"outside_view_{n}"
        out["fixtures"].append({"asset": view_id(vw), "name": names[vw], "outside": True, "view_words": vw,
                                "position": [round(x + f[0] * d, 3), round(y + f[1] * d, 3),
                                             round(ht / 2 - VIEW_SIZE_M[1] / 2, 3)],
                                "facing": [round(-f[0], 4), round(-f[1], 4)]})
    out["views"] = {"by_shot": {k: names[v] for k, v in per_shot.items() if v in names},
                    "words": {nm: vw for vw, nm in names.items()}, "ahead": [round(f[0], 4), round(f[1], 4)],
                    "beyond_m": VIEW_AHEAD_M + 2.0,   # (a still hangs its view at least this far beyond what it frames)
                    "width_m": VIEW_SIZE_M[0]}
    # (a made cabin is a closed shell: a light outside its nose lit nobody, so the view's light is set inside it,
    # ahead of and above the seats, without shadows)
    # (the lights were set for a cabin 4.75 m high: nearer the people in a lower one, they are that much weaker)
    dim = min(1.0, ht / 4.75)
    key = {"type": "rect", "size_m": [1.2, 0.6], "lumens": round(4200 * dim), "kelvin": 5200, "radius_m": 8, "shadows": False,
           "position": [round(x, 3), round(y, 3), round(ht * 0.72, 3)],
           "toward": [round(x - f[0] * ln * 0.25, 3), round(y - f[1] * ln * 0.25, 3), 1.1]}
    pic = lib / "_refs" / vid / "picture.png"
    if pic.exists():                              # the key takes the view's own colour (its brightest tenth)
        from PIL import Image
        px = sorted(Image.open(pic).convert("RGB").resize((64, 32)).getdata(), key=sum)[-205:]
        m = [sum(c[i] for c in px) / len(px) for i in range(3)]
        k = 255.0 / max(max(m), 1.0)
        key.pop("kelvin")
        key["color"] = "#%02x%02x%02x" % tuple(int(min(255, c * k)) for c in m)
    glow = {"type": "point", "lumens": round(420 * dim), "kelvin": 6500, "radius_m": 3, "source_radius_m": 0.15, "shadows": False,
            "position": [round(x - f[0] * ln * 0.08, 3), round(y - f[1] * ln * 0.08, 3), 0.95]}
    out["lights"] = [{"type": "sky", "intensity": 0.05, "atmosphere": False}, key, glow]
    # (what is outside is said shot by shot, by which way the camera looks: redraw.setting_words)
    out["surfaces_words"] = "The light comes in from outside through the canopy, and the instruments glow."
    out["view"] = view_spec(words)


def with_beds(spec: dict) -> dict:
    """Every place of the plan where someone lies has a bed. A lying phrase that lands on a piece nobody can lie on
    (an alcove written as its curtain) gets a bed of its own beside that piece, the phrase moved onto it, and the
    room the next size up for it (grandfather was laid in the family's one bunk, three and four to a bed)."""
    if spec.get("kind") == "outdoors":   # (in the open, whoever lies down lies on the ground)
        return spec
    K = json.loads((HERE.parent / "places" / "piece_kinds.json").read_text())["kinds"]
    kind = {p["id"]: p["kind"] for p in spec.get("pieces") or []}
    named = spec.get("named_places") or {}
    lie = lambda pid: (K.get(kind.get(pid)) or {}).get("needs") == "lie"       # noqa: E731
    need, move = {}, {}
    for ph in spec.get("_lying_phrases") or []:
        at = named.get(ph)
        if at in kind and not lie(at):
            # the spec's own bed for that place, when it wrote one (the phrase among a bed's own), else a new bed
            own = next((p["id"] for p in spec["pieces"] if lie(p["id"]) and ph in (p.get("phrases") or [])), None)
            if not own:
                own = next((named[q] for q in named if named[q] != at and lie(named[q])
                            and any(q in (p.get("phrases") or []) and ph in (p.get("phrases") or [])
                                    for p in spec["pieces"] if p["id"] == at)), None)
            if own:
                move[ph] = own
            else:
                need.setdefault(at, []).append(ph)
    if not need and not move:
        return spec
    sp = json.loads(json.dumps(spec))
    sp["named_places"].update(move)
    if not need:
        return sp
    look = next((p.get("look") for p in sp["pieces"] if (K.get(p["kind"]) or {}).get("needs") == "lie"), "")
    for n, (at, phs) in enumerate(need.items()):
        bid = f"b{n + 1}"
        sp["pieces"].append({"id": bid, "kind": "bed", "look": look, "against": "wall", "phrases": phs})
        sp.setdefault("relations", []).append({"a": bid, "rel": "beside", "b": at})
        for ph in phs:
            sp["named_places"][ph] = bid
    sizes = json.loads((HERE.parent / "scenes" / "place_kinds.json").read_text())["SIZES"]
    if sp.get("size") in sizes[:-1]:
        sp["size"] = sizes[sizes.index(sp["size"]) + 1]
    return sp


CROWD_M2 = 20.0                  # floor per bystander: a crowd is a few figures far off, never a full room
CROWD_MAX = 6


def with_crowd(spec: dict) -> dict:
    """The director's crowd of the place as a few standing figures: made like any piece (drawn from the crowd's own
    words, shaped, stood on the free floor), with no rig and no likeness to keep. Nobody of the cast is among them;
    the keys' pictures fill the rest of the crowd in behind (redraw.py)."""
    # (the director's own line for the place's crowd says who they are and what they wear; each of its groups one term)
    terms = [t.strip().rstrip(".") for t in re.split(r",| and ", str(spec.get("_director_crowd") or spec.get("crowd") or ""))
             if len(t.strip()) > 3]
    rk = L.load_data()["room_kinds"]
    size = (rk.get(spec.get("kind")) or {}).get("sizes_m", {}).get(spec.get("size"))
    n = min(CROWD_MAX, int((size[0] * size[1]) // CROWD_M2)) if terms and size else 0
    if not n:
        return spec
    sp = json.loads(json.dumps(spec))
    period = str(sp.get("period") or "").replace("_", " ")
    for i in range(n):
        t = terms[i % len(terms)]
        who = "one of them" if i < len(terms) else "another of them, unlike the first"
        sp.setdefault("pieces", []).append({
            "id": f"x{i + 1}", "kind": "bystander", "against": "free", "phrases": [],
            "look": f"{t[0].lower() + t[1:]} ({period}), {who}: a single person standing at ease, the whole figure "
                    "from head to feet"})
    return sp


WORLD_SUN_LUX = 120.0            # the key light of a place in the open at a bright sky (scaled by the sky's own
#                                  brightness): measured against the land's own light (img3d/shell_one.py WORLD_EMIT)
WORLD_SKY = 1.0                  # the sky light: it captures the land and the sky dome, so the people take their light


def kelvin_of(hex_colour: str) -> int:
    """A light's colour temperature from its colour (bluer is higher), 2500 to 12000 K."""
    r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return int(min(12000, max(2500, round(6500 * (max(b, 1) / max(r, 1)) ** 1.2, -2))))


def world_stage(out: dict, spec: dict, library) -> dict:
    """A place in the open (with_concept: outdoors): no structure; the place's world (its terrain piece, made from
    its panorama) set so the point under the panorama's eye is the set's origin, the ground people stand on at
    z 0; lit by the world's own land and sky (a sky light that captures them) and a key from where the picture's
    light comes (the brightest of its sky), in that light's colour. -> what was done, for the report."""
    import config as C
    lib = Path(library or C.LIBRARY)
    t = next((p for p in spec.get("pieces") or [] if p.get("kind") == "terrain"), None)
    out["structure"] = []
    out["interior"] = False
    out["tags"] = ["outdoors"]
    if not t:
        return {"world": None, "why": "no terrain piece"}
    aid = P.generation_request(t)["asset_id"]
    rp = lib / "prop" / aid / "asset.json"
    if not rp.exists():
        return {"world": None, "why": f"the place's world {aid} is not made"}
    w = json.loads(rp.read_text()).get("world") or {}
    # (the set's middle is the point under the panorama's eye, where its real ground shows the picture's own ground
    # best; a part of the land the plan names is drawn a few steps from there)
    st = w.get("stand_m") or [0.0, 0.0, 0.0]
    out["fixtures"].insert(0, {"asset": aid, "name": "world", "position": [round(-st[0], 3), round(-st[1], 3),
                                                                           round(-st[2], 3)], "facing": [0.0, -1.0],
                               "outside": True, "surface": True, "world": True})
    sun = w.get("sun")
    lights = [{"type": "sky", "intensity": WORLD_SKY, "atmosphere": False}]
    if sun:
        lights.insert(0, {"type": "sun", "toward_sun": [sun["dir"][0], sun["dir"][1], max(0.15, sun["dir"][2])],
                          "lux": round(WORLD_SUN_LUX * max(0.08, min(1.0, 2.0 * float(sun.get("level") or 0.3))), 1),
                          "kelvin": kelvin_of(sun.get("color") or "#ffffff"), "volumetric": 0.0})
    out["lights"] = lights
    # what the redraw is told of the place: the open land and what stands on it, its ground, sky and light (a
    # room's "walls" and "ceiling" are not said of it)
    held = str(out.get("words") or "").partition(": ")[2].strip().rstrip(".")
    if t.get("land_words"):
        out["words"] = f"In the open air, {t['land_words']}" + (f"; standing there: {held}" if held else "") + "."
    if t.get("surface_words"):
        sw = str(t["surface_words"])
        out["surfaces_words"] = sw[0].upper() + sw[1:] + "."
    out["world"] = {"asset": aid, "flat_r_m": w.get("flat_r_m"), "sun": sun, "spot": t.get("spot"),
                    "at_spot": bool(w.get("spot_m")), "ground_map": bool(w.get("ground_map")),
                    "water_region": w.get("water_region")}
    if (w.get("water") or {}).get("runs"):
        # the map's water (set frame: the world is set with the point under its eye at the origin): nobody is stood
        # in it unless the plan's place is in the water (scenes/from_plan.py)
        out["water"] = w["water"]
    out["validation"] = {"figures": [{"position": [0.0, 0.0], "facing": [0, -1]}]}
    return {"world": aid, "why": "placed"}


def land_marks(spec: dict, world: dict, half: float = 0.0) -> tuple:
    """The parts of the land the plan sends people to (the director's land_places, named_places "floor") that the
    vision judge found on the place's ground map (img3d/pano_shell.py "places_m", set frame): each a place on the
    ground, "land_at<n>", with its words and whether it is in the water; kept on the stage (half: its half side).
    -> (marks, {phrase: mark name}). A land place the map did not show stays the open ground. (Before the map every
    one was "floor", and people sent to a water's edge were stood anywhere, 11-18.)"""
    got = (world or {}).get("places_m") or {}
    lp = spec.get("land_places") if isinstance(spec.get("land_places"), dict) else {}
    marks, named = [], {}
    for ph, v in (spec.get("named_places") or {}).items():
        g = got.get(ph)
        if v != "floor" or ph not in lp or not g or not g.get("at"):
            continue
        x, y = float(g["at"][0]), float(g["at"][1])
        r = math.hypot(x, y)
        if half > 0 and r > half:
            x, y = x * half / r, y * half / r
        nm = f"land_at{len(marks) + 1}"
        marks.append({"name": nm, "position": [round(x, 3), round(y, 3)], "toward": None,
                      "look": g.get("words") or str(lp[ph]), "wet": bool(g.get("wet"))})
        named[ph] = nm
    return marks, named


def world_marks(spec: dict, library) -> dict:
    """The landmarks of a place in the open that its made world shows (img3d/pano_shell.py mark_of), by piece id:
    where people sent to each stand, in the set's frame (the world is set with its spot at the origin, world_stage),
    which way the thing lies from there, and its look. {} for a place with no world or none found in it."""
    import config as C
    t = next((p for p in spec.get("pieces") or [] if p.get("kind") == "terrain"), None)
    if not t or not t.get("marks"):
        return {}
    rp = Path(library or C.LIBRARY) / "prop" / P.generation_request(t)["asset_id"] / "asset.json"
    if not rp.exists():
        return {}
    w = json.loads(rp.read_text()).get("world") or {}
    st = w.get("spot_m") or w.get("stand_m") or [0.0, 0.0, 0.0]
    looks = {p["id"]: p.get("look") or "" for p in spec.get("pieces") or []}
    return {pid: {"at": [mk["stand"][0] - st[0], mk["stand"][1] - st[1]], "toward": mk["toward"], "look": looks.get(pid, "")}
            for pid, mk in (w.get("marks_m") or {}).items() if pid in t["marks"] and pid in looks}


def kit_for(period) -> dict:
    """The period's kit; a period without one takes the kit of the period it is most like (period_kits.json
    "_periods_like"; the knight story's medieval rooms fell to no kit lights at all, 11-16), else victorian."""
    kits = json.loads((HERE / "period_kits.json").read_text())
    p = period if period in kits else (kits.get("_periods_like") or {}).get(period)
    return kits.get(p) or kits["victorian"]


ROOM_LM_M2 = 300.0               # a room whose own lights give less than this (lumens a square metre of floor; a
ROOM_LIGHT_LM = 600              # fire inside a forge's hood lit nothing) gets soft lights over its floor, each at
ROOM_LIGHT_GAP_M = 3.0           # least this, this far apart. (At 150 the forge's stills averaged 12 to 20 of 255
#                                  and its faces were lost in the redraw, 11-17)
FIRE_ROOM_SHARE = 0.25           # a fire's light counts this much toward the room's own: it burns low, in a grate or
#                                  under a hood, and lights the hearth more than the faces round the room


def light_of(picture) -> str:
    """The colour of a picture's light: its brightest hundredth, as #rrggbb at full brightness ("" without one)."""
    try:
        from PIL import Image
        px = sorted(Image.open(picture).convert("RGB").resize((96, 54)).getdata(), key=sum)[-52:]
    except (OSError, ValueError, TypeError):
        return ""
    m = [sum(c[i] for c in px) / len(px) for i in range(3)]
    k = 255.0 / max(max(m), 1.0)
    return "#%02x%02x%02x" % tuple(int(min(255, c * k)) for c in m)


def is_fire(light: dict) -> bool:
    """A fire's light (the kit's fire: a low warm point), not a lamp or the room's own fittings."""
    return light.get("type") == "point" and float(light.get("kelvin") or 6500) <= 2000 and \
        float((light.get("position") or [0, 0, 9])[2]) < 1.5


def room_light(out: dict, spec: dict) -> bool:
    """Every room is lit inside: a closed room without daylight whose kit and pieces gave it too little light of its
    own (a sky light does not reach in) gets a grid of soft lights under its ceiling, in the colour of its concept
    picture's light. Every still of three such sets was black, 11-16. -> whether added."""
    if not any(w.get("kind") == "ceiling" for w in out.get("structure") or []):
        return False
    ls = out.get("lights") or []
    if any(l.get("type") in ("sun", "strip_lights") for l in ls):        # (daylight through its windows; a strip)
        return False
    W, D, H = out["size_m"]
    own = sum(float(l.get("lumens") or 0) * (FIRE_ROOM_SHARE if is_fire(l) else 1.0)
              for l in ls if l.get("type") in ("point", "spot", "rect"))
    want = ROOM_LM_M2 * W * D
    if own >= want:
        return False
    nx, ny = max(1, round(W / ROOM_LIGHT_GAP_M)), max(1, round(D / ROOM_LIGHT_GAP_M))
    each = max(ROOM_LIGHT_LM, (want - own) / (nx * ny))
    col = light_of((spec.get("_concept") or {}).get("concept") or "")
    for i in range(nx):
        for j in range(ny):
            l = {"type": "point", "lumens": round(each), "radius_m": round(max(6.0, 2.5 * H), 1),
                 "source_radius_m": 0.3, "shadows": True,
                 "position": [round((i + 0.5) * W / nx - W / 2, 2), round((j + 0.5) * D / ny - D / 2, 2), round(H - 0.4, 2)]}
            l.update({"kelvin": kelvin_of(col)} if col else {"kelvin": 3200})
            out["lights"].append(l)
    return True


HANG_CLEAR_M = 0.9               # a hung thing with no opening named for it keeps this far along its wall from any
HANG_OVER_M = 0.1                # over an opening: this far above its head


def hang(lay: dict, hung: list, rec: dict) -> None:
    """The hung pieces put on their walls, into the layout's pieces: over the opening they are against (above its
    head), else on the first stretch of wall clear of the openings and of the pieces standing against it; the back
    on the wall, the front into the room, the lowest point "hangs_m" above the floor."""
    W, D, H = lay["size_m"]
    ops = {o["id"]: o for o in lay.get("openings") or []}
    for p in hung:
        r = rec.get(p["id"])
        if not r:
            continue
        sx, sy, sz = r["size_m"]
        z = float(p["hangs_m"])
        o = ops.get(str(p.get("against") or "").replace("opening:", ""))
        if o:
            n, pt = o["normal"], o["point"]
            z = max(z, float(o.get("head_m") or 0.0) + HANG_OVER_M)
        else:
            n = pt = None
            for side, nn in NORMAL.items():
                L_ = W if side in ("north", "south") else D
                for t in (0.0, L_ / 4, -L_ / 4):
                    c = [t, -nn[1] * D / 2] if side in ("north", "south") else [-nn[0] * W / 2, t]
                    busy = [q["point"] for q in ops.values() if q.get("wall") == side] + \
                           [q["position"] for q in lay["pieces"] if q.get("wall") == side or q.get("hang")]
                    if all(math.dist(c, b[:2]) >= HANG_CLEAR_M + sx / 2 for b in busy):
                        n, pt = nn, c
                        break
                if n:
                    break
            if not n:
                continue
        z = min(z, max(0.0, H - sz))
        c = [round(pt[0] + n[0] * (sy / 2 + 0.01), 3), round(pt[1] + n[1] * (sy / 2 + 0.01), 3)]
        hx, hy = (sx / 2, sy / 2) if abs(n[1]) > abs(n[0]) else (sy / 2, sx / 2)
        lay["pieces"].append({"id": p["id"], "name": f"{p['kind']}_{p['id']}", "kind": p["kind"], "asset": r["asset"],
                              "position": c, "facing": [n[0], n[1]], "size_m": [sx, sy, sz], "flat": False,
                              "seat": False, "rect": [round(c[0] - hx, 3), round(c[1] - hy, 3), round(c[0] + hx, 3),
                                                      round(c[1] + hy, 3)],
                              "look": p.get("look"), "generate": False, "hang": True, "z_m": round(z, 3)})


def build(spec: dict, set_id: str, library=None) -> tuple:
    rec, res = records_for(spec, library)
    spec0, world = spec, spec.get("kind") == "outdoors"
    marks = world_marks(spec, library) if world else {}
    if world:                                 # (the place's world is not laid out: it holds the whole stage)
        # ... nor is a landmark the world itself shows (world_marks): it is a place on the ground there
        spec = dict(spec, pieces=[p for p in spec.get("pieces") or [] if p.get("kind") != "terrain" and p["id"] not in marks],
                    relations=[r for r in spec.get("relations") or [] if not {r.get("a"), r.get("b"), r.get("c")} & set(marks)],
                    named_places={ph: v for ph, v in (spec.get("named_places") or {}).items() if v not in marks})
    data = None
    if world:
        # the stage in the open holds what stands on it: each size made wide enough for the largest piece with
        # room all round it and for people beside it (a 20 m landmark found no place on a 12 m stage)
        data = L.load_data()
        big = max([max(rec[p["id"]]["size_m"][:2]) for p in spec.get("pieces") or [] if p["id"] in rec] + [0.0])
        rk = json.loads(json.dumps(data["room_kinds"]["outdoors"]))
        for k, v in rk["sizes_m"].items():
            side = max(v[0], round(big + v[0] / 2 + 2.0, 1))
            rk["sizes_m"][k] = [side, side, max(v[2], round(big, 1))]
        data["room_kinds"]["outdoors"] = rk
    # a thing fixed up on a wall or over an opening ("hangs_m") takes no floor: the room is laid out without it
    # and it is hung afterwards (a sword over the door was a piece with "no free position on a wall", 11-17)
    hung = [p for p in spec.get("pieces") or [] if isinstance(p.get("hangs_m"), (int, float))]
    floor_spec = spec if not hung else dict(
        spec, pieces=[p for p in spec["pieces"] if p not in hung],
        relations=[r for r in spec.get("relations") or []
                   if not {r.get("a"), r.get("b"), r.get("c")} & {p["id"] for p in hung}],
        named_places={ph: v for ph, v in (spec.get("named_places") or {}).items() if v not in {p["id"] for p in hung}})
    lay = L.solve_recover(floor_spec, rec, data)    # solve, then the ladder (never drops a piece the plan names)
    hang(lay, hung, rec)
    ss = L.to_set_spec(spec, lay, data)
    kit = kit_for(spec.get("period"))
    out = dict(id=set_id, place=spec.get("place"),        # (the plan's place it was built for: from_plan picks it)
               description=spec.get("room") or "", style=spec.get("period"), tags=[spec.get("kind")],
               **dress(ss, lay, spec, kit))
    if str(spec.get("_director_crowd") or spec.get("crowd") or "").strip():      # the place's crowd, in the director's words
        out["crowd"] = str(spec.get("_director_crowd") or spec["crowd"]).strip()
    bare = (spec.get("_concept") or {}).get("bare") or {}       # the place's own walls and floor: the concept's
    if bare and out.get("structure"):
        mats = json.loads(json.dumps(out["materials"]))
        if bare.get("walls"):
            # the walls are the picture's: the period's own wall dressing (dado, rails, skirting, panels) comes off
            # them (a white rail ran round the forge's stone walls and through every key, 11-17)
            for w in out["structure"]:
                if w.get("kind") == "wall":
                    for k in ("dado", "trim", "plates"):
                        w.pop(k, None)
        said = str(((spec.get("_concept") or {}).get("surfaces") or {}).get("floor") or "").lower()
        how = next((v for k, v in FLOOR_OF.items() if re.search(rf"\b({k})", said)), None)
        if how:                              # the floor the picture shows, where the period's is another
            for w in out["structure"]:
                if w.get("kind") == "floor" and w.get("material") in mats:
                    mats[w["material"]] = dict(how)
        used = {"walls": {w.get("material") for w in out["structure"] if w.get("kind") == "wall"}
                | {(w.get("dado") or {}).get("material") for w in out["structure"] if w.get("kind") == "wall"},
                "floor": {w.get("material") for w in out["structure"] if w.get("kind") == "floor"}}
        for k, names in used.items():
            for n in names - {None}:
                if bare.get(k, {}).get("colour") and n in mats:
                    mats[n] = dict(mats[n], color=bare[k]["colour"])
                    mats[n].pop("color2", None)
        out["materials"] = mats
    out["words"], out["surfaces_words"] = room_words(out, lay, kit, {r["asset_id"]: r for r in P.load_records(library)})
    if (spec.get("_concept") or {}).get("surfaces_words"):       # the place's own surfaces, as its concept shows them
        out["surfaces_words"] = spec["_concept"]["surfaces_words"]
        out["concept"] = spec["_concept"]["concept"]
    if not world:                             # (a place in the open has no walls to face: a ring of wall panels
        wall_faces(out, lay, spec, library)   # stood round the forge yard's stage, 11-18)
    cabin_stage(out, spec, library)
    staged = world_stage(out, spec0, library) if world else None
    lit = room_light(out, spec)
    # a window or door is a set thing named by its wall (setfacts: "<window|door>_<wall>"), not the layout's name
    alias = {o["name"]: f"{'window' if 'window' in o['kind'] else 'door'}_{o['wall']}" for o in lay["openings"]
             if o.get("name") and o.get("wall") and o["kind"] in ("window", "door")}
    ss["named_places"] = {ph: alias.get(v, v) for ph, v in ss["named_places"].items()}
    for ph, v in (spec.get("named_places") or {}).items():       # (the open floor is a place with no piece)
        if v == "floor":
            ss["named_places"][ph] = "floor"
    if marks:                                 # the land's own things: places on the ground, named as their pieces were
        half = min(out["size_m"][:2]) / 2 - 1.0
        out["marks"] = []
        for pid, mk in marks.items():
            x, y = mk["at"]
            r = math.hypot(x, y)
            if r > half > 0:                  # (kept on the stage)
                x, y = x * half / r, y * half / r
            out["marks"].append({"name": f"land_{pid}", "position": [round(x, 3), round(y, 3)], "toward": mk["toward"],
                                 "look": mk["look"]})
        for ph, v in (spec0.get("named_places") or {}).items():
            if v in marks:
                ss["named_places"][ph] = f"land_{v}"
    if world:                                 # the land's places the ground map shows: places on the ground
        import config as C
        t_ = next((p for p in spec0.get("pieces") or [] if p.get("kind") == "terrain"), None)
        rp_ = Path(library or C.LIBRARY) / "prop" / P.generation_request(t_)["asset_id"] / "asset.json" if t_ else None
        lm, ln = land_marks(spec0, (json.loads(rp_.read_text()).get("world") or {}) if rp_ and rp_.exists() else {},
                            min(out["size_m"][:2]) / 2 - 1.0)
        if lm:
            out["marks"] = list(out.get("marks") or []) + lm
            ss["named_places"].update(ln)
    report = {"pieces": res["summary"], "left_out": [p for p in res["pieces"] if p["status"] != "library"],
              "layout_ok": lay["ok"], "unmet": lay["unmet"], "status": lay.get("status"),
              "ladder": lay.get("ladder"), "dropped": lay.get("dropped"), "reask": lay.get("reask"), "checks": L.check_layout(lay, spec),
              "named_places": ss["named_places"], **({"world": staged} if staged else {}),
              **({"room_light": "a grid of soft lights: the room had no light of its own"} if lit else {}),
              "unresolved_phrases": [ph for ph in spec.get("_phrases") or [] if ph not in ss["named_places"]]}
    return out, ss["named_places"], report


def main():
    a = sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    spec = spec.get("sheet", spec)
    spec = with_beds(spec)
    if "--style" in a:                                 # the book's art style: what its pieces are drawn in
        spec["_style"] = Path(a[a.index("--style") + 1]).read_text().strip().rstrip(".")
    if "--concept" in a:
        spec = with_concept(spec, json.loads(Path(a[a.index("--concept") + 1]).read_text()))
    spec = with_crowd(spec)
    if "--emit" in a:                                  # the spec as it is built (after the concept check)
        Path(a[a.index("--emit") + 1]).write_text(json.dumps(spec, indent=1))
    lib = Path(a[a.index("--library") + 1]) if "--library" in a else None
    out, named, report = build(spec, a[1], lib)
    print(json.dumps(report, indent=1)[:4000])
    if "--write" in a:
        import config as C
        d = Path(lib or C.LIBRARY) / "set" / a[1]
        d.mkdir(parents=True, exist_ok=True)
        (d / "spec.json").write_text(json.dumps(out, indent=1))
        (d / "named_places.json").write_text(json.dumps(named, indent=1))
        (d / "place_build_report.json").write_text(json.dumps(report, indent=1))
        print("written", d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
