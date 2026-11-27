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
    recs = P.period_records(spec, recs)
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
    # a made floor in the open (open_floor): the floor ends at its edge, the walls are that edge, as high as the spec
    # says and bare, and nothing is overhead
    edge = float(spec.get("_edge_m") or OPEN_EDGE_M) if spec.get("_open_top") else None
    # (a made floor that keeps its picture's own structure, picture_stage: its edge is the picture's, standing round
    # the floor as the picture's land; no wall of the set's own is built there, and the floor reaches past its own
    # size only as far as a hung thing's post stands.
    # The set's size is still the floor's: people and cameras are kept on it as in any room)
    stage = bool(spec.get("_picture_stage"))
    st = [dict(kit["floor"])] if edge is None else \
        [dict(kit["floor"], extend_m=OPEN_POST_M + 0.02 if stage else float(kit["wall"].get("thickness_m", 0.3)))]
    for w in ss["structure"]:
        if w.get("kind") == "wall" and stage:
            continue
        if w.get("kind") == "wall":
            w2 = dict(kit["wall"], **{k: v for k, v in w.items() if k != "openings"})
            w2["material"] = kit["wall_material_ns"] if w["side"] in ("north", "south") else kit["wall_material_ew"]
            if edge is not None:
                w2 = dict({k: v for k, v in w2.items() if k not in SHELL_DRESSING}, height_m=edge)
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
    if edge is not None:
        pass                                  # (open to the sky: no ceiling, no roof)
    elif spec.get("_roof") == "rafters" and "panel_raised" in kit["materials"]:
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
    dark = spec.get("_time") in DARK_TIMES
    sky = dict(kit["lights"]["sky"])
    if dark:                                       # (the night sky gives a room next to nothing)
        sky["intensity"] = round(float(sky.get("intensity", 1.0)) * NIGHT_SKY, 3)
    lights = [sky]
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
        if "street_backdrop" in dress_by and not dark:     # (at night nothing is seen outside a window)
            fixtures.append({"asset": dress_by["street_backdrop"], "name": "street_backdrop" + sfx,
                             "position": [round(p[0] - n[0] * 9.5, 3), round(p[1] - n[1] * 9.5, 3), -6.0],
                             "facing": list(n), "outside": True})
        if not kit["lights"].get("window_rect"):
            continue                              # (a period whose windows give no daylight: a ship's viewport)
        wl = dict(kit["lights"]["window_rect"], position=[round(p[0] + n[0] * 0.1, 3), round(p[1] + n[1] * 0.1, 3), 1.75],
                  toward=[round(p[0] + n[0] * 3.5, 3), round(p[1] + n[1] * 3.5, 3), 1.45])
        if dark:
            # no daylight comes in: what the window gives at night is a little cold light, the room's own lamps
            # and fires the rest (lit(): every room is lit inside)
            wl = {**wl, **{k: round(wl[k] * NIGHT_WINDOW, 1) for k in ("lumens", "lux", "intensity") if isinstance(wl.get(k), (int, float))},
                  "kelvin": NIGHT_KELVIN}
            wl.pop("color", None)
        lights.append(wl)
    if kit["lights"].get("ceiling") and edge is None:
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
    rise_ = next((w.get("rise_m") for w in st if w.get("kind") == "trusses"), None)
    if wins and kit["lights"].get("sun") and rise_ and not dark:
        # under a timber roof the wall the sun stands behind is carried up to the roof's boards: over that wall's
        # top the sun fell into the room in sheets (a hall and a shed, 11-19). The other walls' tops stay open to
        # the sky's light: closed all round, a room with one small window was black
        side, th, up = wins[0]["wall"], 0.25, round(rise_ + 0.2, 2)
        base, size = {"north": ([0, D / 2 + th / 2, H], [W + 2 * th, th, up]),
                      "south": ([0, -D / 2 - th / 2, H], [W + 2 * th, th, up]),
                      "east": ([W / 2 + th / 2, 0, H], [th, D + 2 * th, up]),
                      "west": ([-W / 2 - th / 2, 0, H], [th, D + 2 * th, up])}[side]
        st.append({"kind": "box", "name": f"wall_top_{side}", "base": [round(v, 3) for v in base],
                   "size_m": [round(v, 3) for v in size],
                   "material": kit["wall_material_ns" if side in ("north", "south") else "wall_material_ew"]})
    if wins and kit["lights"].get("sun") and not dark:       # daylight from the first window's side
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
            if lp.get("hang"):
                # (one that hangs burns where it hangs, hang(): at its own middle, HUNG_LIGHT_OUT_M before its face.
                # Lit from "its top" as if it stood on the floor, a lantern hung at 2 m gave its light at knee height)
                fx, fy = lp.get("facing") or [0.0, 0.0]
                out_ = float((lp.get("size_m") or [0, 0.3, 0])[1]) / 2 + HUNG_LIGHT_OUT_M
                lights.append(dict(kit["lights"]["fire"], position=[
                    round(lp["position"][0] + fx * out_, 3), round(lp["position"][1] + fy * out_, 3),
                    round(float(lp.get("z_m") or 0.0) + top / 2, 3)]))
                continue
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
        if stage and sy > 0:
            # (a picture stage is seen from its picture's side only, PICTURE_DIR: the two wide pictures that would
            # look back out of it are taken from the eye's edge of the floor, straight along the picture's view)
            x = round(sx * (W / 2 - 0.5) / 3, 2)
            cams[-1].update(position=[x, round(-(D / 2 - 0.5), 2), 1.6], target=[x, round(D / 2, 2), 1.1])
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
OPEN_EDGE_M = 1.1                # the wall round a made floor in the open (open_floor): its edge, about waist high
OPEN_FLOOR_OVER_M = 0.3          # ... and the floor stands this far over the ground of the world round it
OPEN_TOP_STAGE_M = 6.0           # ... and its stage is at least this high (the open stage's own: no ceiling stops a
#                                  camera or a tall thing; the walls are OPEN_EDGE_M whatever this is)
OPEN_HEAD_M = 0.3                # ... and this much higher than the tallest thing standing on it
# a thing the picture shows on a wall, built and hung there: OFF. The first one made (a wall of hanging tools, as wide
# as the picture) was set in the middle of the floor as a solid four metres across: nobody could be stood in the
# room and no camera placed, and six of its eight stills were black (11-19). On again when a hung thing is put
# flat on its own wall and takes no floor.
ROOM_HUNG = False
ROOM_SEEN_MOST = 14              # the things of a room's picture built besides the plan's own, at most
ANY_KINDS = ("work_piece", "landmark")   # the kinds that hold any thing (its look says what it is)
SURFACE_STANDS_M = 1.2           # a thing of a table-top kind the picture stands on the floor this tall is a floor thing
OPEN_POST_M = 0.14               # ... and what hangs higher than its edge hangs on a plain post this thick (hang())
STAGE_NO_EDGE_M = 0.01           # a picture stage's own edge wall: none (what hangs there hangs on a post, hang())
STAGE_FRONT_DEG = 180.0          # a picture stage's world (picture_stage): nothing of its panorama stands near, on any
#                                  side: all of it lies beyond the picture's own land (img3d/pano_shell.py surround)
HUNG_LIGHT_OUT_M = 0.1           # a hung piece that gives light gives it this far before its own face
DARK_TIMES = ("night",)          # the times of day with no sun: no daylight in a room, the moon over open land
NIGHT_WINDOW = 0.04              # a window's light at night, of its daylight
NIGHT_KELVIN = 9000              # ... and its colour; the moon's over open land
NIGHT_SKY = 0.08                 # the sky's own light at night, of the day's
NIGHT_SUN = 0.06                 # the moon's light over open land, of the sun's the place's picture shows
DUSK_SUN = 0.35                  # ... and the low sun's at dusk
# ... and never so little that the set cannot be seen. A mid grey is fully lit at pi x 0.125 x 2^EV100 / 0.18 lux
# (full_lux; 70 lux at the kit's EV100 5.0). The moon over a dark sky gave 0.06 x 120 x 0.08 = 0.6 lux: under a
# hundredth of that, 5 of 255 on the screen, and with no light found in the sky at all, nothing (11-27). At a quarter
# (two stops under: a mid grey about 60 of 255) the night still reads as night and the floor and the people on it
# are seen; at dusk a half (one stop under, about 85 of 255)
OPEN_KEY_LEAST = {"night": 0.25, "dusk": 0.5}
# where the light of a place in the open comes from when its sky shows none (img3d/pano_shell.py sun_of finds no sky
# in a dark panorama): from over the left shoulder of the place's picture's eye (the eye stands on the set's -y side
# and looks along +y), 38 degrees up: what faces the picture's eye is lit, with shape. Every place in the open has
# this light at any hour, so every shot of it has its key (scenes/scene.py key_light turns it to the camera's side)
OPEN_SUN_TOWARD = (-0.5, -0.6, 0.62)
OPEN_SUN_LEVEL = 0.3             # ... and how bright that sky is taken to be (world_stage: as a sky that gives none)
SEEN_OPENINGS = ("window", "door")   # the kinds of opening a room takes from its picture
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
    # a piece another stands on is built as a plain base under it (place_layout.take_bases): what the picture shows
    # where the reader "found" it is no part of it (asked for a bench, the reader found the picture's benches for the
    # platform a seat stands on; folded into it, they were never built, 11-20)
    bases = {r.get("b") for r in sp.get("relations") or [] if L.is_base_relation(sp, r)}
    kept = []
    for p in sp.get("pieces") or []:
        c = chk.get(p["id"]) or {}
        if not p.get("phrases") and not c.get("in_picture") and p["id"] not in bases:
            continue                         # (a base is not looked for in the picture: scenes/concept.py)
        if p.get("look"):
            p["look"] = no_likeness(p["look"])
        if p.get("kind") in ("work_piece", "landmark") and str(p.get("look") or "").strip():
            # what the story calls it (its look begins with its own name): its name in every picture's words. Named
            # "the landmark" with the judge's words for its shape, a cave mouth was drawn as a built stone doorway
            p["called"] = str(p["look"]).split(",")[0].strip().rstrip(".").lower()
        if c.get("words") and p.get("kind") == "landmark":
            p["look"] = c["words"]           # (a thing of the open: what it is, whole, as the picture shows it)
        elif c.get("look") and p["id"] not in bases:
            # (what the director's words say it is, and the picture's material and colour: given the picture's
            # words alone, a seat said to be ornate and upholstered was "wooden, brown", 11-20. A hearth made from
            # "stone, grey" was a block with a pyramid on it, 11-18)
            own = str(p.get("look") or "").strip().rstrip(".")
            p["look"] = f"{own}; {c['look']}" if own and c["look"].lower() not in own.lower() else (own or c["look"])
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
    # (11-26: a place in the open whose ground is a made floor is a room in all of this: its picture's things each
    # built, its floor measured, each thing where the picture has it. open_floor() then takes its roof off)
    open_top = made_floor(concept)
    in_room_ = open_top or (concept.get("space") == "room" and sp.get("kind") != "outdoors")
    # (a room's things by the reader's own name for them, where it names a kind: see seen_kind)
    seen_all = [dict(o, kind=seen_kind(o)) if in_room_ else o for o in concept.get("seen") or []]
    for i, sn in enumerate(seen_all):
        hung_ = ((concept.get("depth") or {}).get("off_floor") or {}).get(f"seen:{i}")
        if hung_ is not None and not (ROOM_HUNG and in_room_ and hung_.get("hangs_m") and sn.get("kind") == "work_piece"):
            continue                         # (it hangs on a wall or from the roof: no piece to stand on the floor)
        # (in the open every large thing standing on the ground is built, unless a piece of the spec is that thing:
        # mostly the same part of the picture)
        # (the same thing, not a small thing standing before a large one: of like size too)
        twin = sn.get("kind") == "landmark" and sn.get("box") and not any(
            _same_part(sn["box"], (chk.get(q["id"]) or {}).get("box"), like=OPEN_LIKE) for q in kept) and not any(
            _same_part(sn["box"], o.get("box"), like=OPEN_LIKE) for o in seen_all[:i]
            if o.get("kind") == "landmark") and sum(1 for q in kept if q["id"].startswith("c")) < OPEN_SEEN_MOST
        # (a seen thing that is the same part of the picture as a piece of the spec is that piece, whatever kind
        # the judge gave it: a block the spec's stone was found in was built a second time on the same spot, 11-19;
        # not a base, which is built as a plain box under what stands on it)
        if sn.get("box") and any(_same_part(sn["box"], (chk.get(q["id"]) or {}).get("box"), like=0.5) for q in kept
                                 if (chk.get(q["id"]) or {}).get("in_picture") and q["id"] not in bases):
            continue
        # (a room takes every thing of its picture, as a place in the open does: each once, the first ROOM_SEEN_MOST;
        # two things in the same part of the picture are one seen twice only when of like kind: benches drawn at
        # their tables overlap them, and a seat tucked at a table is no second sight of it, 11-20)
        more_ = in_room_ and sn.get("box") and sum(1 for q in kept if q["id"].startswith("c")) < ROOM_SEEN_MOST and \
            not any(_same_part(sn["box"], o.get("box"), like=0.5) for o in seen_all[:i]
                    if o.get("kind") and like_kinds(o["kind"], sn.get("kind")))
        if sn.get("kind") and sn.get("crop") and (sn["kind"] not in have or twin or sn.get("group") or more_) and sn["kind"] not in OPENING_KINDS:
            have.add(sn["kind"])
            if hung_ is not None:            # on its wall, as high as the picture has it
                kept.append({"id": f"c{i + 1}", "kind": sn["kind"], "phrases": [], "against": "wall",
                             "look": sn.get("words") or "; ".join(x for x in (sn["name"], sn.get("look")) if x),
                             "source": sn["crop"], "hangs_m": hung_["hangs_m"], "longest_m": round(max(hung_["size_m"]), 2),
                             **({"called": seen_called(sn)} if seen_called(sn) else {})})
                continue
            kept.append({"id": f"c{i + 1}", "kind": sn["kind"], "phrases": [],
                         "look": sn.get("words") or "; ".join(x for x in (sn["name"], sn.get("look")) if x),
                         "against": "not stated", "source": sn["crop"], **({"natural": True} if sn.get("natural") else {}),
                         # (a kind that holds any thing is called what the picture's judge named it)
                         **({"called": seen_called(sn)} if sn["kind"] in ("work_piece", "landmark") and seen_called(sn) else {}),
                         # (how many the reader counted, and whether on both sides of the room: place_layout.prepare)
                         **({"_count": int(sn["count"])} if isinstance(sn.get("count"), int) and sn["count"] >= 1 else {}),
                         **({"_sides": sn["sides"]} if sn.get("sides") in ("both", "one") else {})})
    sp["pieces"] = kept
    ids = {p["id"] for p in kept}
    sp["relations"] = [r for r in sp.get("relations") or [] if r.get("a") in ids and r.get("b") in ids]
    dp = concept.get("depth") or {}
    if dp.get("room_m") and (concept.get("space") == "room" or open_top):
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
                if isinstance(t.get("raised_m"), (int, float)):
                    p["_raised_m"] = t["raised_m"]   # (how far off the floor its foot is: place_layout.prepare)
                if p.get("kind") == "work_piece" and not p.get("longest_m"):
                    p["longest_m"] = round(max(t["size_m"]), 2)      # (a thing of the picture: as big as it shows it)
                if size_agrees(p.get("longest_m"), t["size_m"]):
                    p["_seen_size_m"] = t["size_m"]
                else:
                    # the picture's thing is far from the size the story gives it: the story's thing is made from
                    # its own words at its own size, where the picture has it (a stone "the size of a sleeping dog",
                    # 0.8 m, was shaped from the picture's squared block 1.4 m long and a metre high, 11-19)
                    p.pop("source", None)
        if sp.get("kind") != "outdoors" or open_top:
            sp["pieces"], sp["_picture_left"], sp["_picture_merged"] = picture_pieces_settled(sp["pieces"], bases)
    if concept.get("space") == "room" and sp.get("kind") != "outdoors":
        # an opening the picture shows and the spec states none of is the room's (a hall drawn with tall windows down
        # both walls was built a sealed box with its kind's one door, 11-19): one of each kind, on the wall the
        # layout finds for it
        stated = {o.get("kind") for o in sp.get("openings") or []}
        for i, sn in enumerate(concept.get("seen") or []):
            # (a window or a door only: a fire the picture shows is the room's own thing, built from the picture.
            # Taken as the kind's fireplace, a forge got the period kit's white mantelpiece on a panelled breast, 11-19)
            if sn.get("kind") in SEEN_OPENINGS and sn.get("box") and sn["kind"] not in stated:
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
    if open_top:
        sp = open_floor(sp, spec, concept)
    elif concept.get("space") == "outdoors" or spec.get("kind") == "outdoors":
        sp = outdoors(sp, spec, concept)
    sp["_concept"] = {k: concept.get(k) for k in ("concept", "space", "surfaces_words", "bare", "surfaces")}
    if dp.get("faces") and open_top:
        # (a made floor in the open: its floor as the picture shows it; its low edge walls and the sky take none)
        # (laid in the floor as it is built: the stage over it is higher than the picture measured, open_floor)
        sp["_concept"]["faces"] = [dict(f, room_m=sp.get("_room_m") or dp.get("room_m")) for f in dp["faces"]
                                   if f.get("face") == "floor"]
    if dp.get("faces") and concept.get("space") == "room" and sp.get("kind") != "outdoors":
        # the room's own walls as its picture shows them (img3d/concept_depth.py room_faces), with the room's
        # size they were measured in
        sp["_concept"]["faces"] = [dict(f, room_m=dp.get("room_m")) for f in dp["faces"]]
    return sp


LIKENESS = re.compile(r"\s*[,;]?\s*\b(?:(?:about |roughly |nearly )?(?:the )?(?:size|shape|height|length|width) of|"
                      r"(?:as|about as) (?:big|large|small|long|tall|wide|heavy) as|"
                      r"no (?:bigger|larger|taller|longer) than|(?:shaped|sized) like)\s+[^,;.]+", re.I)


def no_likeness(look: str) -> str:
    """A thing's look without what it is likened to for its size or shape: the words go to a picture model, which
    draws every thing they name (a stone "the size of a sleeping dog" was made a stone statue of a sleeping dog,
    11-19). Its size is the spec's own number."""
    out = LIKENESS.sub("", str(look or "")).strip(" ,;")
    return out or str(look or "")


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


def seen_kind(sn: dict) -> str | None:
    """A thing of a room's picture by the reader's own name for it: when the name ends with a kind's word ("long
    dining tables", "wooden sideboard") and holds no word of the kind the judge filed it under, it is that kind (the
    judge filed tables as a bench and a sideboard as a table, 11-20). A thing the judge filed under no kind stays
    so (a wall, a window: no box was taken for it)."""
    judged = sn.get("kind")
    if not judged or judged in ANY_KINDS and sn.get("group"):
        return judged
    toks = L._words(sn.get("name"))
    if not toks:
        return judged
    stems = {L.noun_stem(t) for t in toks}
    if any(L.noun_stem(w) in stems for syn in L._kind_words(judged) for w in L._words(syn)):
        return judged
    best = None
    for k in L.PK:
        if k in OPENING_KINDS or k in ANY_KINDS or (L.PK[k] or {}).get("placement") not in ("floor", "surface", "wall"):
            continue
        for syn in L._kind_words(k):
            sw = L._words(syn)
            n = len(sw)
            if n and n <= len(toks) and toks[len(toks) - n:-1] == sw[:-1] and L.noun_stem(toks[-1]) == L.noun_stem(sw[-1]) \
                    and (best is None or n > best[0]):
                best = (n, k)
    return best[1] if best else judged


def like_kinds(a, b) -> bool:
    """Two kinds one thing could be read as twice: the same, or of one family, or one that holds any thing; never a
    seat and a thing that is no seat (a bench at its table)."""
    if a == b or a in ANY_KINDS or b in ANY_KINDS:
        return True
    seat = lambda k: (L.PK.get(k) or {}).get("needs") == "seat"     # noqa: E731
    return seat(a) == seat(b) and L._root(a) == L._root(b)


def picture_pieces_settled(pcs: list, bases=()) -> tuple:
    """A measured room's pieces from its picture settled before they are made (places/generate.py makes every piece
    of the emitted spec): -> (pieces, left, merged).
    - a thing of a table-top kind the picture stands on its floor SURFACE_STANDS_M tall or more is a thing of the
      floor, made as what the reader called it at the size it shows (a candlestick kind is 0.3 m: a 2.2 m standing
      candle-branch was made 0.54 m, 11-20);
    - a thing the picture's depth did not stand on the floor (on a wall, overhead; not a table-top thing) is left in
      the picture the room's walls carry: no piece to make (a wall hanging was made a work piece, then refused by the
      layout, 11-20);
    - a thing the picture shows once that is a piece of the spec seen again (place_layout.same_thing) is that piece:
      it takes the picture's drawing, place, size and how high it stands, and adds the picture's words to its own;
      its phrases stay (a seat the plan names and the picture's seat were made twice, 11-20)."""
    R = L.pc.load_rules()
    out, left, merged = [], [], []
    for p in pcs:
        d = L.PK.get(p.get("kind")) or {}
        if L.PICTURE_ID.match(str(p.get("id"))) and d.get("placement") == "surface" and p.get("at_xy") and \
                len(p.get("_seen_size_m") or []) == 3 and float(p["_seen_size_m"][2]) >= SURFACE_STANDS_M:
            name = re.split(r"[;,]", str(p.get("look") or ""), maxsplit=1)[0].strip().lower()
            many = L.plural_noun((L._words(name) or [""])[-1]) and not p.get("_count")
            p = dict(p, kind="work_piece", called=p.get("called") or name, longest_m=round(max(p["_seen_size_m"]), 2),
                     **({"_count": int(L.picture_rules(R)["repeat_default"])} if many else {}))
            d = L.PK.get("work_piece") or {}
        if L.PICTURE_ID.match(str(p.get("id"))) and not p.get("at_xy") and d.get("placement") != "surface" \
                and not isinstance(p.get("hangs_m"), (int, float)):
            left.append({"id": p["id"], "kind": p.get("kind"), "called": p.get("called") or str(p.get("look") or "")[:60],
                         "why": "not on the floor in the picture: left in the picture the room's walls carry"})
            continue
        out.append(p)
    size = lambda p: p.get("_seen_size_m") or (L.PK.get(p.get("kind")) or {}).get("size_m")   # noqa: E731
    own = [p for p in out if not L.PICTURE_ID.match(str(p.get("id"))) and p["id"] not in set(bases)]
    taken, keep = set(), []
    for c in out:
        if not L.PICTURE_ID.match(str(c.get("id"))) or L.plural_count(c, R) != 1:
            keep.append(c)
            continue
        twin = next((p for p in own if p["id"] not in taken and L.same_thing(c, p, size(c), size(p), R)), None)
        if twin is None:
            keep.append(c)
            continue
        taken.add(twin["id"])
        if c.get("source") and not twin.get("source"):
            twin["source"] = c["source"]
        for k in ("at_xy", "facing_xy", "_seen_size_m", "_raised_m"):
            if c.get(k) is not None and twin.get(k) is None:
                twin[k] = c[k]
        if c.get("against") and twin.get("against") in (None, "", "not stated"):
            twin["against"] = c["against"]
        seen_words = str(c.get("look") or "").split(";", 1)[-1].strip()
        if seen_words and seen_words.lower() not in str(twin.get("look") or "").lower():
            twin["look"] = "; ".join(x for x in (str(twin.get("look") or "").strip().rstrip("."), seen_words) if x)
        merged.append({"id": c["id"], "into": twin["id"]})
    return keep, left, merged


DRESS_HEAD = re.compile(r"\s+(?:of|with|made|covered|in|on|upon|atop|from|holding|hanging|hung|displayed|wrapped|"
                        r"strewn|depicting|across|along|set|standing|for|at|mimicking|bearing|draped|lining|lined|"
                        r"filled|showing|under|over|against|beside|by|near|around|that|which)\b", re.I)
DRESS_ON_FACE = re.compile(r"\b(?:on|across|along|over|from|against|covering|lining|up)\b(?:\s+[\w-]+){0,3}?\s+"
                           r"(?:walls?|floors?|ceilings?|roofs?|rafters|beams)\b|\b(?:strewn|scattered)\b", re.I)
DRESS_HANGS = re.compile(r"\b(?:hanging|hung|hangs|suspended)\b", re.I)
DRESS_ON = re.compile(r"\b(?:on|upon|atop)\s+(?:the\s+|a\s+|an\s+|its\s+)?([a-z][a-z\- ]*?)(?=$|[,;.]|\s+(?:with|of|and|"
                      r"in|along|against|by|beside|near)\b)", re.I)


def dress_nouns(item: str) -> tuple:
    """(the item's head words, the nouns it names: the last word of each part of its head, "goblets and plates")."""
    head = DRESS_HEAD.split(str(item or ""), maxsplit=1)[0]
    parts = [L._words(x) for x in re.split(r"\s+(?:and|&)\s+|,", head)]
    return head.strip(), [x[-1] for x in parts if x]


def kind_of_noun(noun: str, head: str = "") -> str | None:
    """The floor or table-top kind whose word ends the head ("tables" -> table, "sideboards" -> cabinet)."""
    toks = L._words(head) or [noun]
    best = None
    for k, d in L.PK.items():
        if k in OPENING_KINDS or k in ANY_KINDS or (d or {}).get("placement") not in ("floor", "surface", "wall"):
            continue
        for syn in L._kind_words(k):
            sw = L._words(syn)
            n = len(sw)
            if n and n <= len(toks) and toks[len(toks) - n:-1] == sw[:-1] and \
                    L.noun_stem(toks[-1]) == L.noun_stem(sw[-1]) and (best is None or n > best[0]):
                best = (n, k)
    return best[1] if best else None


def with_dressing(spec: dict, dress: dict, rules: dict | None = None) -> dict:
    """The place's dressing (scenes/place_dress.py: what its picture was drawn furnished with) checked against the
    pieces the set will hold, each item put somewhere and said where ("_dressing": [{"item", "where", "piece"}]):
    - a piece of the set already (a noun of the item's head among the piece's name words, layout_rules "dressing"):
      that piece, its drawing the picture's own;
    - said to hang, or to be on or across the walls, floor or roof, or a part of the room itself, or a thing that
      hangs: in the pictures the room's walls, floor and roof carry (no piece);
    - said to stand on a piece of the set: on it, in its drawing; on a thing the set lacks: that thing is made;
    - else made from its words: of the floor or table-top kind its head names, else a work piece called by it, as
      many as its plural asks, placed by the layout's own rules ("_dressing": kept off the plan's floor as the
      picture's own things are). The picture drew a rich room's tables, benches and plate; its set had one of each
      of four and nothing else, and nobody checked (11-20)."""
    sp = json.loads(json.dumps(spec))
    items = [str(x).strip() for x in (dress or {}).get("dressing") or [] if str(x).strip()]
    if not items:
        return sp
    R = rules or L.pc.load_rules()
    dr = R.get("dressing") or {}
    hangs = {L.noun_stem(w) for w in dr.get("hangs") or []}
    room_own = {L.noun_stem(w) for w in dr.get("room_own") or []}
    many = int(L.picture_rules(R)["repeat_default"])
    bases = {r.get("b"): r.get("a") for r in sp.get("relations") or [] if L.is_base_relation(sp, r)}
    pcs = sp.get("pieces") or []

    def words_of(p):
        ws = set(L._name_of(p)) | set(L._words(p.get("called")))
        ws |= set(L._words(re.split(r"[;,]", str(p.get("look") or ""), maxsplit=1)[0]))
        return {L.noun_stem(t) for t in ws}

    def found(nouns, head=""):
        for p in pcs:
            if {L.noun_stem(n) for n in nouns} & words_of(p):
                return p
        # (a head of two nouns names the thing by the first as well: the word before its last, when that is a
        # piece's own kind - seats named by what they are were made a second time as chairs, 11-20)
        ws = L._words(head)
        if len(ws) > 1:
            for p in pcs:
                if L.noun_stem(ws[-2]) in {L.noun_stem(t) for t in L._words(str(p.get("kind") or "").replace("_", " "))}:
                    return p
        return None

    left = {L.noun_stem(t) for x in sp.get("_picture_left") or [] for t in L._words(x.get("called"))}
    report, n_new = [], 0
    for it in items:
        head, nouns = dress_nouns(it)
        stems = {L.noun_stem(n) for n in nouns}
        p = found(nouns, head)
        if p is not None:
            where = f"the piece {p['id']} ({p.get('kind')})" + (f", built as the base under {bases[p['id']]}"
                                                                 if p["id"] in bases else "")
            report.append({"item": it, "where": where, "piece": p["id"]})
            continue
        if stems & left:
            report.append({"item": it, "where": "on a wall in the picture: in the picture the room's walls carry",
                           "piece": None})
            continue
        kd = kind_of_noun(nouns[-1] if nouns else "", head) if nouns else None
        if stems & hangs or (kd and (L.PK.get(kd) or {}).get("placement") == "wall") or DRESS_HANGS.search(it):
            report.append({"item": it, "where": "hangs: in the pictures the room's walls and roof carry", "piece": None})
            continue
        # (a part of the room named first and told about after, "walls painted pale": any word of a head that names
        # no floor or table-top kind)
        own = stems & room_own or (not kd and {L.noun_stem(t) for t in L._words(head)} & room_own)
        if DRESS_ON_FACE.search(it) or own:
            report.append({"item": it, "where": "the room's own walls, floor or roof: in the pictures they carry",
                           "piece": None})
            continue
        m = DRESS_ON.search(it[len(head):])
        on = L._words(m.group(1)) if m else []
        if on:
            host = found([on[-1]])
            if host is not None:
                report.append({"item": it, "where": f"on the piece {host['id']} ({host.get('kind')}), in its drawing",
                               "piece": host["id"]})
                continue
            # (on a thing the set lacks: that thing is made, with the item on it in its words)
            head, nouns, kd = m.group(1).strip(), [on[-1]], kind_of_noun(on[-1], m.group(1))
        if not nouns:
            continue
        n_new += 1
        pid = f"d{n_new}"
        while any(q["id"] == pid for q in pcs):
            n_new += 1
            pid = f"d{n_new}"
        kd = kd if kd and (L.PK.get(kd) or {}).get("placement") in ("floor", "surface") else None
        # (made from words alone, its age is said in them: no picture of the place tells its maker)
        age = str(sp.get("period") or "").strip()
        piece = {"id": pid, "kind": kd or "work_piece", "phrases": [],
                 "look": f"{it}; {age}" if age and age.lower() not in it.lower() else it, "against": "not stated",
                 "_dressing": True, "_picture": True,
                 "_count": many if L.plural_noun(L._words(head)[-1] if L._words(head) else "") else 1}
        if not kd:
            piece["called"] = re.sub(r"^(a|an|the)\s+", "", head.strip().lower())
        pcs.append(piece)
        report.append({"item": it, "where": f"not in the picture: made from its words as {pid} ({piece['kind']}), "
                                            f"{piece['_count']} placed by the layout", "piece": pid})
    sp["pieces"] = pcs
    sp["_dressing"] = report
    return sp


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
OPEN_SCALE_BY_M = 1.5            # ... those of them that are really this tall or taller
OPEN_BARE_M = 6.0                # a place with nothing standing on it: its middle this far before the picture's eye
LAND_ID = "land"                 # the piece that is the picture's own land (scenery beside the world)
VIEW_EYE_M = (1.2, 6.0)           # the picture's eye over the ground, within reason
VIEW_SIDE_DEG = 12.0             # the picture's own land stands for its width and this much more on either side
VIEW_FRONT_DEG = (30.0, 75.0)


def open_scale(pcs: list, concept: dict) -> float:
    """The measured picture's metres corrected by how tall its things really are (the judge's "real_m" for each,
    scenes/concept.py): the middle one of their ratios, within OPEN_SCALE. A whole smithy measured 4 m wide and 3 m
    tall, and was built so (11-18)."""
    dp_ = concept.get("depth") or {}
    things = dp_.get("things") or {}
    chk, seen = concept.get("spec") or {}, concept.get("seen") or []
    rs = []
    for p in pcs:
        i = int(p["id"][1:]) - 1 if p["id"].startswith("c") and p["id"][1:].isdigit() else -1
        src = chk.get(p["id"]) or (seen[i] if 0 <= i < len(seen) else {})
        key = f"spec:{p['id']}" if f"spec:{p['id']}" in things else f"seen:{i}"
        t = things.get(key)
        # (not by the land's own things: how big a rock or a cliff really is is anyone's guess, and the form made
        # from the picture's depth is measured with what the frame cut carried on; nor by a small thing, whose box
        # is loose: a bench judged 0.4 m tall made a cave mouth 3 m wide for a dragon to live in, 11-19)
        if src.get("natural") or p.get("natural") or (dp_.get("relief") or {}).get("of") == key:
            continue
        if t and src.get("real_m") and t["size_m"][2] > 0.2 and float(src["real_m"][1]) >= OPEN_SCALE_BY_M:
            rs.append(float(src["real_m"][1]) / float(t["size_m"][2]))
    if not rs:
        return 1.0
    rs.sort()
    return round(min(OPEN_SCALE[1], max(OPEN_SCALE[0], rs[len(rs) // 2])), 3)


HIGH_M = (0.45, 1.2)             # land that rises this much: nobody is stood on it; this much: no camera stands in it
HIGH_CELL_M, HIGH_HALF_M = 0.5, 40.0
HIGH_NEAR_M, HIGH_MOST = 10.0, 0.5


def land_high(tops: dict, eye: list, sc: float, ox: float, oy: float) -> dict | None:
    """Where the picture's land rises (img3d/concept_depth.py tops_of), as cells of the set's ground in the form
    of its water (places/ground_map.py: {"cell_m", "half_m", "runs"}): "runs" where nobody is stood, "tall_runs"
    where no camera stands. The staged people of a river bank stood inside the bank's own rise, and cameras looked
    at them from within it (11-19)."""
    import ground_map as GM
    z = (tops or {}).get("z")
    if not z:
        return None
    c, x0, y0 = float(tops["cell_m"]), float(tops["x0"]), float(tops["y0"])
    n = GM.cells(HIGH_HALF_M, HIGH_CELL_M)
    out = {"cell_m": HIGH_CELL_M, "half_m": HIGH_HALF_M}
    for name, least in (("runs", HIGH_M[0]), ("tall_runs", HIGH_M[1])):
        m = []
        for i in range(n):
            row = []
            for j in range(n):
                x, y = GM.cell_mid(i, j, HIGH_HALF_M, HIGH_CELL_M)
                # (the set's frame -> the measured picture's)
                a, b = int(((y + oy) / sc + eye[1] - y0) // c), int(((x + ox) / sc + eye[0] - x0) // c)
                row.append(0 <= a < len(z) and 0 <= b < len(z[a]) and z[a][b] * sc >= least)
            m.append(row)
        # (land that rises over most of the ground round the set's middle is the ground itself, tilted in the
        # picture's measure: it bars nobody)
        k = int(HIGH_NEAR_M / HIGH_CELL_M)
        mid = [row[n // 2 - k:n // 2 + k] for row in m[n // 2 - k:n // 2 + k]]
        if sum(map(sum, mid)) > HIGH_MOST * max(1, sum(map(len, mid))):
            m = [[False] * n for _ in range(n)]
        out[name] = GM.runs_of(m)
    return out if out["runs"] or out["tall_runs"] else None


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
    rl = dp.get("relief") or {}
    ld = dp.get("land") if isinstance(dp.get("land"), dict) and (dp.get("land") or {}).get("glb") else {}
    for p in pcs:
        key = f"spec:{p['id']}" if f"spec:{p['id']}" in things else seen_key.get(p["id"], "")
        t = things.get(key)
        if not t:
            out.append(p)
            continue
        if key in (ld.get("has") or []) and not p.get("phrases"):
            continue                         # (the picture's land has it, img3d/concept_depth.py land: not built twice)
        t = dict(t, size_m=[float(v) * sc for v in t["size_m"]])
        x, y = (t["xy"][0] - eye[0]) * sc, (t["xy"][1] - eye[1]) * sc
        if rl.get("of") == key and rl.get("glb"):
            # the land's own form, shaped from the picture's depth (img3d/concept_depth.py relief): where and as
            # big as that shape is, whole
            p["relief"] = {"glb": rl["glb"], "size_m": rl["size_m"]}
            at[p["id"]] = (x, y, list(t["size_m"]))
            out.append(p)
            continue
        far = math.hypot(x, y) > OPEN_FAR_M or max(t["size_m"]) > OPEN_BIG_M
        if far and not p.get("phrases"):
            continue                         # (it lies beyond the set: the panorama's)
        k = min(1.0, OPEN_FAR_M / max(math.hypot(x, y), 1e-6))
        size = [min(float(v), OPEN_BIG_M) for v in t["size_m"]]
        at[p["id"]] = (x * k, y * k, size)
        out.append(p)
    if not at and not ld:
        return out, None
    if at:
        ox = sum(v[0] for v in at.values()) / len(at)
        oy = min(v[1] - v[2][1] / 2 for v in at.values()) - OPEN_FRONT_M
    else:                                    # (nothing stands there: the set's middle on the ground before the eye)
        ox, oy = 0.0, OPEN_BARE_M * sc
    if ld:
        # all the land the picture shows (img3d/concept_depth.py land), where the picture has it: the place's
        # scenery, beside its world; nothing is laid out against it and nobody is kept off it
        g = [float(v) * sc for v in ld["size_m"]]
        out.append({"id": LAND_ID, "kind": "terrain", "against": "free", "phrases": [], "scenery": True,
                    "look": "the land of the place's picture", "relief": {"glb": ld["glb"], "size_m": ld["size_m"]},
                    "at_xy": [round((ld["xy"][0] - eye[0]) * sc - ox, 3), round((ld["xy"][1] - eye[1]) * sc - oy, 3)],
                    "under_m": round(float(ld.get("under_m") or 0.0) * sc, 4), "_seen_size_m": [round(v, 3) for v in g],
                    "high": land_high(ld.get("tops"), eye, sc, ox, oy),
                    # (the picture's eye in the set's frame, how high and how wide it sees: the place's panorama is
                    # made from the picture, from there, places/pieces.py "view")
                    "view": {"eye_at": [round(-ox, 2), round(-oy, 2)],
                             "eye_h": round(min(VIEW_EYE_M[1], max(VIEW_EYE_M[0], float((dp.get("eye") or [0, 0, 1.6])[2]) * sc)), 2),
                             "front_deg": round(min(VIEW_FRONT_DEG[1], max(VIEW_FRONT_DEG[0], float(dp.get("hfov_deg") or 60.0) / 2
                                                                         + VIEW_SIDE_DEG)), 0)}})
    if not at:
        return out, None
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


# what lies round a place in the open (the concept reader's "around", scenes/concept.py AROUND_ASK), as its world's
# look begins. A place with water or a town round it has its world drawn as that (places/pieces.py
# WORLD_ROUND_WORDS), from these words and what the picture shows far off: never from the place's own words or its
# picture (drawn from those, a panorama had the place's own building in it again, close by). "land", and a concept
# read before the question was asked: the words and the world it had before, untouched
ROUND_LOOK = {"water": "open water on every side, out to a far level horizon",
              "buildings": "the walls and roofs of a town far off on every side, along the horizon"}
PICTURE_DIR = [0.0, 1.0]         # the way the place's picture looks, in the set's frame: its eye stands on the -y side
#                                  and looks along +y (open_places; a made floor's room frame, img3d/concept_depth.py)


def round_of(concept: dict) -> str:
    """What lies round a place in the open: "water", "buildings", else "land" (as before the question was asked)."""
    a = concept.get("around") if concept.get("space") == "outdoors" else None
    return a if a in ROUND_LOOK else "land"


def round_look(rnd: str, per: str, far: str, overhead: str, tech: str) -> str:
    """The look of the world of a place with water or buildings round it: what lies round it, what the picture shows
    far off, its sky, the way it is painted. Nothing of the place itself and no ground."""
    lead = ROUND_LOOK[rnd] + (f"; far off: {far[0].lower() + far[1:]}" if far else "")
    return "; ".join(x for x in (f"{per}: {lead}".strip(": ") if per != "not stated" else lead,
                                 f"overhead: {overhead}" if overhead else "", f"painted as a {tech}" if tech else "") if x)


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
    scenery = [p for p in pcs if p.get("scenery")]
    if stage or scenery:
        # what stands there and the land round it are placed from the one picture, seen from its eye: the cameras
        # keep to that side (world_stage "picture_dir", scenes/speakers.py picture_side)
        sp["_picture_dir"] = list(PICTURE_DIR)
    pcs = [p for p in pcs if not p.get("scenery")]
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
    rnd = round_of(concept)
    if rnd != "land":                        # (water or a town round it: drawn as that, with no ground of the land's)
        far_look = round_look(rnd, per, far, str(sf.get("ceiling") or "").strip().rstrip("."),
                              technique(str(spec.get("_style") or "")))
    # the place's own ground seen from straight above (places/ground_map.py), drawn from the whole land's words and
    # its ground's, the parts of the land the plan sends people to on it, the concept picture as the land's look
    ground = "; ".join(x for x in (land, f"the ground: {str(sf['floor']).strip().rstrip('.')}" if sf.get("floor") else "") if x)
    # (with the picture's own land built: the panorama is that picture carried on all the way round, in the whole
    # land's words, from the picture's eye)
    view = next((p["view"] for p in scenery if p.get("view")), None) if concept.get("concept") else None
    pcs.append({"id": "t1", "kind": "terrain", "against": "free", "phrases": [],
                "look": ((not view and far_look) or "; ".join(x for x in (land, said, technique(str(spec.get("_style") or "")) if view
                                                                      else str(spec.get("_style") or "")) if x))
                # (the picture carried on all the way round: behind its eye, what lies round the place)
                + (f"; round it, behind and to both sides: {ROUND_LOOK[rnd]}" if view and rnd != "land" else ""),
                **({"around": rnd} if rnd != "land" and not view else {}),
                **({"spot": spot} if spot and not view else {}), "land_words": land, "surface_words": said,
                **({"view": view} if view else {}),
                **({"source": concept["concept"]} if concept.get("concept") and (view or not far_look) else {}),
                **({"map_words": ground, "land_places": {ph: str(lp[ph]).strip().rstrip(".") for ph in order},
                    "map_style": technique(str(spec.get("_style") or ""))} if ground else {}),
                **({"map_source": concept["concept"]} if ground and concept.get("concept") else {})})
    sp["pieces"] = pcs + scenery             # (after the world: the first terrain piece is the world)
    sp["relations"] = [r for r in sp.get("relations") or [] if r.get("a") in ids and r.get("b") in ids]
    sp["named_places"] = {ph: (gone.get(v, v) if gone.get(v, v) in ids else "floor")
                          for ph, v in (spec.get("named_places") or {}).items()}
    return sp


def made_floor(concept: dict) -> bool:
    """A place in the open whose ground is itself built: a made floor with an edge round it, under the sky (the
    concept reader's "ground": "built", scenes/concept.py GROUND_ASK)."""
    return concept.get("space") == "outdoors" and concept.get("ground") == "built"


def picture_stage(concept: dict) -> dict:
    """The picture's own land of a made floor in the open (img3d/concept_depth.py stage_land: the depth's "land",
    "stage": true), when its picture was measured with it; {} for every other place, and for a made floor measured
    before there was one: that one is built as it was (open_floor's low walls and a world from words) until its
    picture is measured again."""
    dp = concept.get("depth") or {}
    ld = dp.get("land") if isinstance(dp.get("land"), dict) else {}
    return ld if made_floor(concept) and dp.get("room_m") and ld.get("stage") and ld.get("glb") else {}


def stage_pieces(pcs: list, concept: dict, land: str, said: str, tech: str) -> tuple:
    """A picture stage's own pieces: -> (the floor's pieces without those the picture's land has, the place's world,
    the picture's land).

    The land (LAND_ID, scenery) is the picture itself at its own depth, in the room's frame: everything the picture
    shows standing on and round the floor that is no loose piece of the set. It stands where it was measured, its
    own floor just under the real one ("under_m": the real floor, with the floor's own picture on it, is the one
    people stand on and the one seen); where it rises nobody is stood and no camera stands ("high", land_high).
    The world is the picture carried on all the way round from the picture's own eye ("view": places/pieces.py),
    every part of it beyond the land (STAGE_FRONT_DEG: the place's structure is in that panorama again, and stood
    near it stood twice); its flat ground is water for a floor that stands in water, else there is none ("ground":
    "none"): under the floor's edge is what the land and the panorama show there."""
    dp, ld = concept.get("depth") or {}, picture_stage(concept)
    eye = [float(v) for v in (dp.get("eye") or [0.0, 0.0, 1.6])]
    seen_key = {f"c{i + 1}": f"seen:{i}" for i in range(len(concept.get("seen") or []))}
    has = set(ld.get("has") or [])
    # (what the picture's land has is not built again: open_places)
    pcs = [p for p in pcs if p.get("phrases") or seen_key.get(p["id"], f"spec:{p['id']}") not in has]
    rnd = round_of(concept)
    view = {"eye_at": [round(eye[0], 2), round(eye[1], 2)],
            "eye_h": round(min(VIEW_EYE_M[1], max(VIEW_EYE_M[0], eye[2])), 2), "front_deg": STAGE_FRONT_DEG}
    scenery = {"id": LAND_ID, "kind": "terrain", "against": "free", "phrases": [], "scenery": True,
               # (the picture is drawn with its own light, its lamps and lit windows: its land gives that light as
               # the world round it does, places/pieces.py "own_light". Lit only by the set's lights at night, a
               # stage's wide pictures showed under a quarter of the frame over near black, 11-27)
               "look": "the land of the place's picture", "relief": {"glb": ld["glb"], "size_m": ld["size_m"],
                                                                     "own_light": True},
               "at_xy": [round(float(ld["xy"][0]), 3), round(float(ld["xy"][1]), 3)],
               "under_m": round(float(ld.get("under_m") or 0.0), 4),
               "_seen_size_m": [round(float(v), 3) for v in ld["size_m"]],
               # (the land's frame is the set's own: no eye to move it by, no scale)
               "high": land_high(ld.get("tops"), [0.0, 0.0], 1.0, 0.0, 0.0), "view": view}
    world = {"id": "t1", "kind": "terrain", "against": "free", "phrases": [],
             "look": "; ".join(x for x in (land, said, tech) if x)
             + (f"; round it, behind and to both sides: {ROUND_LOOK[rnd]}" if rnd != "land" else ""),
             "ground": "water" if rnd == "water" else "none", "land_words": land, "surface_words": said,
             "view": view, "source": concept["concept"]}
    return pcs, world, scenery


def open_floor(sp: dict, spec: dict, concept: dict) -> dict:
    """A place in the open whose ground is itself built (made_floor) is built as a room with no roof, not as open
    land: people stand on a flat floor of the picture's floor, a low wall of the picture's sides stands round its
    edge (OPEN_EDGE_M: seen over, never a room's wall), nothing is overhead, and the picture's things stand on it
    where the picture has them, each made as a room's things are (with_concept). Round it and over it lies the
    place's world, as round a place on open land (pieces "terrain", made_as "world"): the far land and the sky,
    drawn from words alone; the set is lit by it (open_floor_stage).

    (11-26: built as open land, such a place was sandy ground with every built thing of its picture, its floor and
    its edge too, shaped as one piece ten metres wide: a smear of spikes, and no floor to stand on.)

    The spec keeps the kind "outdoors" (no door or window of a room kind's own, none of its dressing) with the
    marks "_open_top" and "_edge_m"; "_room_m" is the floor as the picture measures it, its height the stage's (how
    high a camera may stand), not the walls'.

    (11-27: a made floor whose picture was measured with its own land, picture_stage, keeps that picture's
    structure: the floor, its things and its stage as here; in place of the low walls the picture's land standing
    round the floor, and in place of the world from words the picture carried on all the way round, stage_pieces.
    Marked "_picture_stage". Built with the plain low walls, a floor's posts, house and railings were all lost.)"""
    sp = dict(sp, kind="outdoors", openings=[], outside="", _open_top=True, _edge_m=OPEN_EDGE_M)
    for k in ("_views", "_roof"):            # (no view through a window, no roof on timbers)
        sp.pop(k, None)
    if sp.get("_room_m"):                    # (measured as a room: WALL_MIN_M; open to the sky, the stage's height)
        sp["_room_m"] = [sp["_room_m"][0], sp["_room_m"][1], max(float(sp["_room_m"][2]), OPEN_TOP_STAGE_M)]
    pcs = [p for p in sp.get("pieces") or [] if p.get("kind") != "terrain"]
    ids = {p["id"] for p in pcs}
    land = re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or spec.get("room") or ""),
                  flags=re.I).strip().rstrip(".")
    per = str(spec.get("period") or "").replace("_", " ")
    land = f"{per}: {land}".strip(": ") if per != "not stated" else land
    sf = concept.get("surfaces") or {}
    stage = picture_stage(concept) if concept.get("concept") else {}
    said = "; ".join(f"{k}: {str(v).strip().rstrip('.')}" for k, v in (("the floor", sf.get("floor")),
                                                                         ("its edge" if stage else "the low wall round its edge", sf.get("walls")),
                                                                         ("overhead", sf.get("ceiling")),
                                                                         ("the light", concept.get("light"))) if v)
    # the world round the floor is what the picture shows far off and overhead, in words alone (drawn from the
    # picture, a panorama had the place's own built things in it again, close by: outdoors()); the floor is no part
    # of it, and no ground map is drawn: nobody is sent out onto that ground
    far = str(concept.get("far") or "").strip().rstrip(".")
    tech = technique(str(spec.get("_style") or ""))
    far_look = "; ".join(x for x in (
        f"{per}: {far}".strip(": ") if per != "not stated" else far,
        f"overhead: {str(sf['ceiling']).strip().rstrip('.')}" if sf.get("ceiling") else "",
        f"painted as a {tech}" if tech else "") if x) if far else ""
    rnd = round_of(concept)
    if rnd != "land":
        # (water or a town round the floor: its world drawn as that, and never from the place's own words and
        # picture, which the look falls back to when the picture shows nothing far off. In water, the world's own
        # flat ground is water too: the floor stands in it, places/pieces.py "ground")
        far_look = round_look(rnd, per, far, str(sf.get("ceiling") or "").strip().rstrip("."), tech)
    if (concept.get("depth") or {}).get("room_m"):
        sp["_picture_dir"] = list(PICTURE_DIR)   # (measured from the one picture: the cameras keep to its side)
    scenery = []
    if stage:
        sp["_picture_stage"] = True
        pcs, world, land_piece = stage_pieces(pcs, concept, land, said, tech)
        ids = {p["id"] for p in pcs}
        pcs.append(world)
        scenery = [land_piece]
    else:
        pcs.append({"id": "t1", "kind": "terrain", "against": "free", "phrases": [],
                    "look": far_look or "; ".join(x for x in (land, said, str(spec.get("_style") or "")) if x),
                    **({"around": rnd} if rnd != "land" else {}), **({"ground": "water"} if rnd == "water" else {}),
                    "land_words": land, "surface_words": said,
                    **({"source": concept["concept"]} if concept.get("concept") and not far_look else {})})
    sp["pieces"] = pcs + scenery             # (after the world: the first terrain piece is the world)
    sp["relations"] = [r for r in sp.get("relations") or [] if r.get("a") in ids and r.get("b") in ids]
    # (a phrase that lands on no piece of the floor, a door of the spec's among them, is the open floor)
    sp["named_places"] = {ph: (v if v in ids else "floor") for ph, v in (sp.get("named_places") or {}).items()}
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


FACE_PIC_LEAST_M2 = 1.0         # a face's own picture (img3d/concept_depth.py room_faces) smaller than this is not used
FACE_PIC_OFF_M = 0.04           # a wall's picture hangs this far inside its wall: before the tiles (0.015 m)
FACE_PIC_ROOM_TOL = 0.02        # ... and only in a room of the size it was measured in (this share either way)
FACE_PIC_END_M = 0.06           # a wall's picture stops this short of each corner (the next wall's picture stands there)
FACE_PIC_FLOOR_Z = -0.015       # the floor's picture lies with its underside this low: its face 5 mm over the slab
FACE_PIC_CEIL_IN_M = 0.045      # the ceiling's picture: this short of every wall (clear of the walls' pictures) and
FACE_PIC_CEIL_UNDER_M = 0.005   # ... its face this far under the ceiling
FACE_PIC_RULE = "whole-2"       # (how the pictures are made: concept_depth.whole_faces; a new rule is a new panel.
#                                  2: the panel's picture over its whole texture, "keep_uvs")
FLATS = ("floor", "ceiling")
SHELL_DRESSING = ("dado", "trim", "plates", "columns")   # what sets/layout.py build_wall puts on a wall's inner face


def face_pictures(spec: dict) -> list:
    """The room's walls, floor and ceiling as its concept picture shows them (img3d/concept_depth.py room_faces,
    carried in the spec's "_concept" "faces"), each made whole (concept_depth.whole_faces: a seen wall carried along
    its unseen run by its own rhythm, else mirrored; the wall the picture never sees continued from its neighbours
    round the corners; the floor and the ceiling carried across and along) -> [(panel spec, {"face", "room_m"})].

    Once the picture sees any wall at least FACE_PIC_LEAST_M2, all four walls get one; the floor and the ceiling
    one each when the picture sees them that much, and the ceiling never under an open roof (the spec's "_roof":
    its trusses and boards are built, and the picture's roof plane cut across them would hide them). A panel is
    the whole face less FACE_PIC_END_M at each corner; the floor's and the ceiling's lie flat
    (places/generate.py make_face_picture). Its id is its pictures', its tile's and the rule's: anything new is a
    new panel."""
    import hashlib
    con = spec.get("_concept") or {}
    words = ((con.get("bare") or {}).get("walls") or {}).get("words")
    tile = face_spec(words)["id"] if words else ""
    floor_col = ((con.get("bare") or {}).get("floor") or {}).get("colour")

    def area(f):
        return abs(f["u"][2] - f["u"][1]) * abs(f["v"][2] - f["v"][1])
    recs = [f for f in con.get("faces") or [] if (f.get("face") in NORMAL or f.get("face") in FLATS)
            and f.get("file") and Path(f["file"]).exists() and f.get("room_m") and area(f) >= FACE_PIC_LEAST_M2]
    if not recs:
        return []
    rm = [float(v) for v in recs[0]["room_m"]]
    W, D, H = rm
    seen = {f["face"] for f in recs}
    names = [n for n in ("north", "east", "south", "west")] if seen & set(NORMAL) else []
    names += [n for n in FLATS if n in seen and not (n == "ceiling" and spec.get("_roof") == "rafters")]
    h_ = hashlib.sha1((FACE_PIC_RULE + tile + json.dumps(rm)).encode())
    for f in sorted(recs, key=lambda f: f["face"]):
        h_.update(f["face"].encode() + Path(f["file"]).read_bytes())
    dg = h_.hexdigest()
    srcs = [{k: f[k] for k in ("face", "u", "v", "px_per_m", "file") if k in f} for f in recs]
    got = []
    for name in names:
        flat = name in FLATS
        if flat:
            ins = 0.005 if name == "floor" else FACE_PIC_CEIL_IN_M
            w, h, iu, iv = W - 2 * ins, D - 2 * ins, ins, ins
        else:
            w, h, iu, iv = (W if name in ("north", "south") else D) - 2 * FACE_PIC_END_M, H, FACE_PIC_END_M, 0.0
        pid = f"{name if flat else 'wall'}_face_" + hashlib.sha1((dg + name).encode()).hexdigest()[:6]
        what = f"the {name} wall" if not flat else f"the {name}"
        sp = {"id": pid, "class": "prop", "tier": "background", "texture_px": 2048, "placement": "floor",
              "description": f"The face of {'a floor' if name == 'floor' else 'a ceiling' if flat else 'a wall'}, one "
                             f"flat picture {w:.2f} x {h:.2f} m: {what} of a room as its own picture shows it. " +
                             ("Lying flat, the picture's top toward +Y; origin at the middle of its underside." if flat
                              else "Front -Y; origin at the middle of the bottom edge."),
              "tags": [name if flat else "wall", "face", "generated-on-request"],
              "size_m": [round(w, 3), round(h, 3), 0.02] if flat else [round(w, 3), 0.02, round(h, 3)],
              "size_tolerance": 0.05, "source": "scripted", "backdrop": {"w": round(w, 3), "h": round(h, 3), "emit": 0.0},
              "face_whole": {"face": name, "room_m": rm, "faces": srcs, "inset_m": [round(iu, 3), round(iv, 3)]},
              "face_fill": "" if flat else tile, "face_colour": floor_col if name == "floor" else None,
              "face_flat": flat,
              # (the picture laid over the whole texture by the panel's own UVs, period_kit.picture_uvs: the
              # harness's unwrap gave a 21.6 x 8 m wall 38 texels of height and it showed as a smear, 11-20)
              "keep_uvs": True,
              "licence": "CC0 (original script; picture from the place's own concept picture)",
              "cameras": [{"name": "front", "position": [0.0, -max(w, h) * 0.4, max(w, h) * 1.3], "target": [0.0, 0.0, 0.0],
                           "focal": 28}] if flat else
                         [{"name": "front", "position": [0.0, -max(w, h) * 1.6, h / 2], "target": [0.0, 0.0, h / 2],
                           "focal": 28}]}
        got.append((sp, {"face": name, "room_m": rm}))
    return got


def wanted_openings(lay: dict, spec: dict) -> list:
    """The layout's openings the story itself asks for: on a wall it states, or named by the plan's places (by its id,
    or a phrase naming its kind). The rest (the room kind's own door, a window the picture shows elsewhere, given a
    wall by the layout) are the set's guesses."""
    np_ = spec.get("named_places") or {}
    said = " ".join(str(k).lower() for k in np_)
    vals = {str(v) for v in np_.values()}
    return [o for o in lay.get("openings") or []
            if o.get("stated") or str(o.get("id")) in vals or str(o.get("name")) in vals
            or re.search(rf"\b{re.escape(str(o.get('kind') or '-'))}s?\b", said)]


def wall_pictures(out: dict, spec: dict, library, lay: dict | None = None) -> dict:
    """The room's own pictures (face_pictures) hung: each wall's over the whole wall, FACE_PIC_OFF_M inside it; the
    floor's on the floor, the ceiling's under a flat ceiling -> {wall: [(from, to along the wall, z from, z to)]}
    that they cover, for the tiles to leave. out["room_faces"]: what each face got, and why.

    A wall with an opening the story asks for (wanted_openings) takes no picture: a panel cannot be cut round it, and
    the door must be seen; its tiles leave the opening as before. Over every other opening the picture hangs (the
    picture's own walls show its windows and doors: a bright square of the period's window cut into its pattern, on
    a wall the picture never sees, was the set's guess against its picture), its daylight kept, its drapes taken
    down. A room built to another size than the picture was measured at takes none (its pictures would not meet its
    corners). A wall that carries its picture is built bare behind it (SHELL_DRESSING off, no openings of its own):
    nothing the shell builds stands proud of the panel."""
    import config as C
    W, D, H = out["size_m"]
    hung, why = {}, {}
    lay = lay or {}
    wanted = {o.get("wall") for o in wanted_openings(lay, spec)}
    roof = any(s.get("kind") == "trusses" for s in out.get("structure") or [])
    ceil = next((s for s in out.get("structure") or [] if s.get("kind") == "ceiling"), None)
    for fs, f in face_pictures(spec):
        side = f["face"]
        rm = f.get("room_m")
        if rm and (abs(rm[0] - W) > FACE_PIC_ROOM_TOL * W or abs(rm[1] - D) > FACE_PIC_ROOM_TOL * D
                   or abs(rm[2] - H) > FACE_PIC_ROOM_TOL * H):
            why[side] = "the room is built to another size than its picture was measured at"
            continue
        if side == "ceiling" and (roof or not ceil or abs(float(ceil.get("height_m") or 0) - H) > 0.05):
            why[side] = "an open roof: its own timbers and boards stand for it"
            continue
        if side in wanted:
            why[side] = "an opening the story asks for is on it: the tiles, left off round it"
            continue
        out.setdefault("face_pictures", []).append(fs)
        if not (Path(library or C.LIBRARY) / "prop" / fs["id"] / "asset.json").exists():
            why[side] = "its panel is not made yet"
            continue
        why[side] = "its picture, over the whole wall" if side in NORMAL else "its picture, laid over it whole"
        if side == "floor":
            out["fixtures"].append({"asset": fs["id"], "name": "floor_face_picture", "surface": True,
                                    "position": [0.0, 0.0, FACE_PIC_FLOOR_Z], "facing": [0.0, -1.0]})
            continue
        if side == "ceiling":
            out["fixtures"].append({"asset": fs["id"], "name": "ceiling_face_picture", "hang": True, "surface": True,
                                    "position": [0.0, 0.0, round(H - FACE_PIC_CEIL_UNDER_M - 0.02, 3)],
                                    "facing": [0.0, -1.0]})
            continue
        n = NORMAL[side]
        pos = [0.0, -n[1] * (D / 2 - FACE_PIC_OFF_M)] if side in ("north", "south") else [-n[0] * (W / 2 - FACE_PIC_OFF_M), 0.0]
        out["fixtures"].append({"asset": fs["id"], "name": f"wall_face_{side}_picture", "hang": True, "surface": True,
                                "position": [round(pos[0], 3), round(pos[1], 3), 0.0], "facing": [n[0], n[1]]})
        L_ = W if side in ("north", "south") else D
        hung.setdefault(side, []).append((-L_ / 2, L_ / 2, 0.0, H))
        # (the shell's dressing and what it builds in its openings stood up to 0.14 m out of the wall, the panel
        # 0.04: a white rail ran through the picture round a whole room and an oak door showed through it, 11-20)
        for w in out.get("structure") or []:
            if w.get("kind") == "wall" and w.get("side") == side:
                for k in SHELL_DRESSING:
                    w.pop(k, None)
                if w.get("openings"):
                    w["openings"] = []
        for o in lay.get("openings") or []:            # (a window under the picture: no drapes before a bare wall)
            if o.get("wall") == side and o.get("kind") == "window" and o.get("point"):
                out["fixtures"] = [x for x in out["fixtures"] if not (str(x.get("name") or "").startswith("drapes")
                                   and math.dist(x["position"][:2], o["point"][:2]) < 0.4)]
    if spec.get("_roof") == "rafters" and "ceiling" not in why and any(
            f.get("face") == "ceiling" for f in (spec.get("_concept") or {}).get("faces") or []):
        why["ceiling"] = "an open roof: its own timbers and boards stand for it"
    if why:
        out["room_faces"] = why
    return hung


def wall_faces(out: dict, lay: dict, spec: dict, library) -> None:
    """The room's walls wear the concept's own wall: its face panels side by side along every wall, a hand's
    breadth inside it, left off where a door, window or fireplace stands (there the period's wall shows). Where
    the room has its concept picture's walls (wall_pictures: every wall, made whole from what the picture sees), that
    picture covers the wall and it takes no tiles; the floor and a flat ceiling take theirs too."""
    import config as C
    if not any(w.get("kind") == "wall" for w in out.get("structure") or []):
        return
    pics = wall_pictures(out, spec, library, lay)
    words = (((spec.get("_concept") or {}).get("bare") or {}).get("walls") or {}).get("words")
    if not words:
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
                if any(p[0] - 0.05 <= t - L_ / cols / 2 and t + L_ / cols / 2 <= p[1] + 0.05 and p[2] - 0.05 <= r * FACE_M
                       and min(H, (r + 1) * FACE_M) <= p[3] + 0.05 for p in pics.get(side, [])):
                    continue                                         # (the wall's own picture covers it whole)
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
CROWD_MANY_M2 = 4.0              # ... but a place the story has full of people gets a figure to this much floor
CROWD_MANY_MAX = 24              # (the keys' pictures keep to the engine's frame behind the people: a crowd that is
#                                  not standing in the set is not in the picture, whatever the words say, 11-26)


CROWD_LOOKS = ("one of them", "another of them, unlike the first", "a woman among them", "an older one among them")


def crowd_line(spec: dict) -> str:
    """The place's unnamed people in words: those the story's passage has there (the spec's "crowd") and those the
    director's set sheet adds, each group once. (A ball's sheet named its musicians and servants and left out the
    guests: the room was drawn all but empty, 11-26.)"""
    out = []
    for src in (spec.get("_crowd_who"), spec.get("crowd"), spec.get("_director_crowd")):
        for t in re.split(r";", str(src or "")):
            t = t.strip().rstrip(".")
            if t and t.lower() not in [o.lower() for o in out]:
                out.append(t)
    return "; ".join(out)


def with_crowd(spec: dict) -> dict:
    """The director's crowd of the place as a few standing figures: made like any piece (drawn from the crowd's own
    words, shaped, stood on the free floor), with no rig and no likeness to keep. Nobody of the cast is among them;
    the keys' pictures fill the rest of the crowd in behind (redraw.py)."""
    # (the director's own line for the place's crowd says who they are and what they wear; each of its groups one term)
    terms = [t.strip().rstrip(".") for t in re.split(r",| and |;", crowd_line(spec)) if len(t.strip()) > 3]
    rk = L.load_data()["room_kinds"]
    size = (rk.get(spec.get("kind")) or {}).get("sizes_m", {}).get(spec.get("size"))
    # (the room as its picture was measured, where it was: a hall "large" by its kind's table was 20 m2, by its
    # picture 92)
    seen = spec.get("_room_m")
    if isinstance(seen, (list, tuple)) and len(seen) >= 2 and seen[0] * seen[1] > 0:
        size = seen
    full = spec.get("_crowd_size") == "many"            # a place the story has full of people (place_spec.crowd_size)
    n = min(CROWD_MANY_MAX if full else CROWD_MAX, int((size[0] * size[1]) // (CROWD_MANY_M2 if full else CROWD_M2))) \
        if terms and size else 0
    if not n:
        return spec
    sp = json.loads(json.dumps(spec))
    period = str(sp.get("period") or "").replace("_", " ")
    lead = bool(str(spec.get("_crowd_who") or "").strip()) and len(terms) > 1
    seen_of = {}
    for i in range(n):
        # (the story's own people of the place, its first term, are two in three of a crowd; those the set sheet
        # adds share the rest: of a ball's figures a third were guests and two thirds its musicians and servants)
        t = terms[0] if lead and i % 3 != 2 else terms[1 + (i // 3) % (len(terms) - 1)] if lead else terms[i % len(terms)]
        k = seen_of[t] = seen_of.get(t, -1) + 1
        # (a few looks of each group; a fuller crowd stands the same figures again elsewhere on the floor)
        who = CROWD_LOOKS[k % (len(CROWD_LOOKS) if lead and t == terms[0] else 2)]
        sp.setdefault("pieces", []).append({
            # (half of a full place's figures stand round its walls, half about the floor)
            "id": f"x{i + 1}", "kind": "bystander", "against": "wall" if full and i % 2 == 0 else "free", "phrases": [],
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


def full_lux(ev100: float) -> float:
    """The light (lux) that shows a mid grey (0.18) as a mid grey at an exposure: pi x 0.125 x 2^EV100 / 0.18 (the
    camera's own light is sized by the same measure, scenes/ue/scene_job.py camera_fill)."""
    return math.pi * 0.125 * 2.0 ** float(ev100) / 0.18


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
    if spec.get("_picture_dir"):
        # the place is built from one picture of it (with_concept): which way that picture looks, and whether the
        # world round the set is a true panorama, good from every side ("all_round": no route makes one yet). The
        # cameras look into the picture's side unless it is (scenes/speakers.py picture_side)
        out["picture_dir"] = list(spec["_picture_dir"])
        out["all_round"] = False
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
    land_said = []
    for ln in [p for p in spec.get("pieces") or [] if p.get("scenery") and p.get("at_xy")]:
        # the picture's own land, where the picture has it, its ground just under the real ground
        la = P.generation_request(ln)["asset_id"]
        land_said.append(la if (lib / "prop" / la / "asset.json").exists() else f"{la} is not made")
        if (lib / "prop" / la / "asset.json").exists():
            out["fixtures"].insert(1, {"asset": la, "name": ln["id"], "facing": [0.0, -1.0], "outside": True,
                                       "position": [ln["at_xy"][0], ln["at_xy"][1], -float(ln.get("under_m") or 0.0)],
                                       "world": True})
            if ln.get("high"):
                out["high_land"] = ln["high"]
    # (a world whose sky shows no light still has one, from OPEN_SUN_TOWARD: said in the set's "world" "sun_from")
    sun = w.get("sun") or {"dir": list(OPEN_SUN_TOWARD), "level": OPEN_SUN_LEVEL}
    lights = [{"type": "sky", "intensity": round(WORLD_SKY * (NIGHT_SKY * 2 if spec.get("_time") in DARK_TIMES else 1.0), 4),
               "atmosphere": False}]
    tod = spec.get("_time")
    # the place's own time of day (the picture it was made from is drawn by day whatever the story says): at
    # night its one light from the sky is the moon's, low and cold; at dusk a low sun; neither under its share
    # (OPEN_KEY_LEAST) of the light the set's own exposure needs
    k_ = NIGHT_SUN if tod in DARK_TIMES else DUSK_SUN if tod == "dusk" else 1.0
    least = OPEN_KEY_LEAST.get(tod, 0.0) * full_lux(out.get("exposure_ev100") or 5.0)
    lights.insert(0, {"type": "sun", "toward_sun": [sun["dir"][0], sun["dir"][1], max(0.15, sun["dir"][2])],
                      "lux": round(max(least, k_ * WORLD_SUN_LUX * max(0.08, min(1.0, 2.0 * float(sun.get("level") or OPEN_SUN_LEVEL)))), 1),
                      "kelvin": NIGHT_KELVIN if tod in DARK_TIMES else kelvin_of(sun.get("color") or "#ffffff"),
                      "volumetric": 0.0})
    out["lights"] = lights
    # what the redraw is told of the place: the open land and what stands on it, its ground, sky and light (a
    # room's "walls" and "ceiling" are not said of it)
    held = str(out.get("words") or "").partition(": ")[2].strip().rstrip(".")
    if t.get("land_words"):
        when = {"night": " at night under a dark sky", "dusk": " at dusk, the light low"}.get(tod, "")
        out["words"] = f"In the open air{when}, {t['land_words']}" + (f"; standing there: {held}" if held else "") + "."
    if t.get("surface_words"):
        sw = str(t["surface_words"])
        out["surfaces_words"] = sw[0].upper() + sw[1:] + "."
    out["world"] = {"asset": aid, "flat_r_m": w.get("flat_r_m"), "sun": sun, "spot": t.get("spot"),
                    "sun_from": "its sky" if w.get("sun") else "no light in its sky: the set's own",
                    "at_spot": bool(w.get("spot_m")), "ground_map": bool(w.get("ground_map")),
                    "water_region": w.get("water_region"),
                    **({"ground": w["ground"]} if w.get("ground") else {})}     # ("water": all its flat ground is)
    if (w.get("water") or {}).get("runs"):
        # the map's water (set frame: the world is set with the point under its eye at the origin): nobody is stood
        # in it unless the plan's place is in the water (scenes/from_plan.py)
        out["water"] = w["water"]
    out["validation"] = {"figures": [{"position": [0.0, 0.0], "facing": [0, -1]}]}
    return {"world": aid, "why": "placed", "land": land_said}


def open_floor_stage(out: dict, spec: dict, library) -> dict:
    """A made floor in the open (open_floor) on its stage: the room's own structure kept (the floor, the low walls
    round its edge), the place's world set round it as round a place on open land (world_stage: the sky and the far
    land, their sky light and the key from where the sky is brightest, the time of day, the redraw's words), its
    ground OPEN_FLOOR_OVER_M under the floor; the lamps and fires of the floor's own pieces burn on. The set's
    "open_top" says what it is: its cameras stay on the floor (scenes/speakers.py open_ground) and its floor is
    still called a floor (redraw/redraw.py). A picture stage (open_floor) has no walls of its own: the picture's
    land stands round the floor where it was measured (world_stage places it, with where it rises: "high_land"),
    and the set says "picture_stage". -> what was done, for the report."""
    structure = out.get("structure") or []
    own = [l for l in out.get("lights") or [] if l.get("type") not in ("sky", "sun")]
    done = world_stage(out, spec, library)
    out["structure"] = structure
    out["open_top"] = True
    if spec.get("_picture_stage"):
        out["picture_stage"] = True
    if done.get("world"):
        fx = next(f for f in out["fixtures"] if f.get("world"))
        fx["position"] = [fx["position"][0], fx["position"][1], round(fx["position"][2] - OPEN_FLOOR_OVER_M, 3)]
        out["lights"] = list(out["lights"]) + own
    return done


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
    kit = kits.get(p) or kits["victorian"]
    if period in ((kits.get("_plain_periods") or {}).get("periods") or []):
        kit = plain(kit)
    return kit


PLAIN_WORDS = {"window": "a small unglazed window opening in a timber frame", "door": "a door of plain planks",
               "fireplace": "an open hearth"}


def plain(kit: dict) -> dict:
    """Another period's kit for a period of plainer buildings that has none of its own (period_kits.json
    "_plain_periods"): its lights and its build, with nothing of that period's style. Its openings are plain ones (an unglazed window in a timber frame, a door of planks), no window
    dressing and no view of that period's street; its words name them so. (A medieval forge got a white
    mantelpiece, a shed crimson drapes and a glazed sash window, each from the kit made for a Victorian sitting
    room, 11-19.) What the room looks like comes from its own picture."""
    k = json.loads(json.dumps(kit))
    dark = next((m for m in ("door_oak", "panel_wood") if m in k.get("materials", {})), None)
    if dark:
        k["window"] = dict(k.get("window") or {}, panes=[1, 1], frame_material=dark, glass_material=None)
        k["door"] = dict(k.get("door") or {}, panel_m=[0.22, 3.0], material=dark, frame_material=dark, braced=False)
    k["window_dressing"] = []
    k.pop("drapes_look", None)
    k["words"] = dict(k.get("words") or {}, **PLAIN_WORDS, surfaces="")
    k["_plain"] = True
    return k


ROOM_LM_M2 = 450.0               # a room whose own lights give less than this (lumens a square metre of floor; a
ROOM_LIGHT_LM = 600              # fire inside a forge's hood lit nothing) gets soft lights over its floor, each at
ROOM_LIGHT_GAP_M = 3.0           # least this, this far apart. (At 150 the forge's stills averaged 12 to 20 of 255
#                                  and its faces were lost in the redraw, 11-17)
ROOM_LIGHT_UNDER_M = 0.4        # the soft lights hang this far under the ceiling; under a ceiling that carries its
ROOM_LIGHT_SPREAD = 0.6          # picture this share of their spacing under it (0.4 m under it each lit a bright
ROOM_LIGHT_HEAD_M = 2.6          # disc on the picture, 11-20), never lower than this over the floor
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
    # (daylight through a window does not excuse a room: with the sun at its one window a hall and a shed were dark
    # with hot patches on their walls and nobody in them could be seen. The user's rule, 11-19: inside a building
    # everyone is visible, whatever the time of day; the light from outside is kept on top of it)
    if any(l.get("type") == "strip_lights" for l in ls):
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
    under = ROOM_LIGHT_UNDER_M
    if any(f.get("name") == "ceiling_face_picture" for f in out.get("fixtures") or []):
        under = max(under, min(ROOM_LIGHT_SPREAD * max(W / nx, D / ny), H - ROOM_LIGHT_HEAD_M))
    for i in range(nx):
        for j in range(ny):
            l = {"type": "point", "lumens": round(each), "radius_m": round(max(6.0, 2.5 * H), 1),
                 "source_radius_m": 0.3, "shadows": True,
                 "position": [round((i + 0.5) * W / nx - W / 2, 2), round((j + 0.5) * D / ny - D / 2, 2), round(H - under, 2)]}
            l.update({"kelvin": kelvin_of(col)} if col else {"kelvin": 3200})
            out["lights"].append(l)
    return True


HANG_CLEAR_M = 0.9               # a hung thing with no opening named for it keeps this far along its wall from any
HANG_OVER_M = 0.1                # over an opening: this far above its head


def hang(lay: dict, hung: list, rec: dict, top: float | None = None) -> None:
    """The hung pieces put on their walls, into the layout's pieces: over the opening they are against (above its
    head), else on the first stretch of wall clear of the openings and of the pieces standing against it; the back
    on the wall, the front into the room, the lowest point "hangs_m" above the floor. top: the walls' own height
    where they are lower than the room (a made floor's edge, open_floor): a thing that hangs higher than they stand
    keeps its own height (a lantern forced down onto a waist-high edge lit nobody's face, 11-27) and hangs there
    on a post rising from the edge ("post_m": the post's height, to the thing's top; built by hung_posts): nothing
    hangs in the air."""
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
        post = {"post_m": round(z + sz, 3), "post_at": [round(pt[0] - n[0] * (OPEN_POST_M / 2 + 0.01), 3),
                                                       round(pt[1] - n[1] * (OPEN_POST_M / 2 + 0.01), 3)]} \
            if top and z + sz > top + 1e-6 else {}
        c = [round(pt[0] + n[0] * (sy / 2 + 0.01), 3), round(pt[1] + n[1] * (sy / 2 + 0.01), 3)]
        hx, hy = (sx / 2, sy / 2) if abs(n[1]) > abs(n[0]) else (sy / 2, sx / 2)
        lay["pieces"].append({"id": p["id"], "name": f"{p['kind']}_{p['id']}", "kind": p["kind"], "asset": r["asset"],
                              "position": c, "facing": [n[0], n[1]], "size_m": [sx, sy, sz], "flat": False,
                              "seat": False, "rect": [round(c[0] - hx, 3), round(c[1] - hy, 3), round(c[0] + hx, 3),
                                                      round(c[1] + hy, 3)],
                              "look": p.get("look"), "generate": False, "hang": True, "z_m": round(z, 3), **post})


def hung_posts(out: dict, lay: dict) -> list:
    """The posts of a made floor's hung things (hang "post_m") built: each a plain square post in the edge wall's
    own material (structure "box", built by sets/layout.py), standing in the wall's thickness behind the thing,
    from the floor to the thing's top. -> their names."""
    # (a picture stage has no wall of its own, open_floor: its posts are of the floor's material)
    mat = next((w.get("material") for k in ("wall", "floor") for w in out.get("structure") or [] if w.get("kind") == k), None)
    done = []
    for q in lay.get("pieces") or []:
        if q.get("post_m") and mat:
            out["structure"].append({"kind": "box", "name": f"post_{q['id']}", "base": [*q["post_at"], 0.0],
                                     "size_m": [OPEN_POST_M, OPEN_POST_M, q["post_m"]], "material": mat})
            done.append(f"post_{q['id']}")
    return done


def base_boxes(out: dict, lay: dict) -> list:
    """The layout's bases (place_layout.lay_bases) built: each a plain box in the room's floor material on the
    floor (structure "box", built by sets/layout.py), and the piece it carries told where its base's front edge
    is (its set item's "base": scenes/from_plan.py stands people sent to it on the floor before that edge). A room
    with no structure of its own (a place in the open) builds none: said in the result."""
    done = []
    floor = next((w for w in out.get("structure") or [] if w.get("kind") == "floor"), None)
    items = {f.get("name"): f for f in out.get("fixtures") or []}
    for b in lay.get("bases") or []:
        if not b.get("of"):
            done.append({"id": b["id"], "built": False, "why": b.get("why")})
            continue
        if floor is None:
            done.append({"id": b["id"], "built": False, "of": b["of"], "why": "no structure to build a box in"})
        else:
            out["structure"].append({"kind": "box", "name": f"base_{b['id']}", "base": [b["center"][0], b["center"][1], 0.0],
                                     "size_m": list(b["size_m"]), "yaw_deg": b["yaw_deg"], "material": floor["material"]})
            done.append({"id": b["id"], "built": True, "of": b["of"], "box": f"base_{b['id']}", "size_m": b["size_m"],
                         "center": b["center"], "said_kind": b.get("kind_said")})
        for n in b["of"]:
            if n in items:
                items[n]["base"] = {"name": f"base_{b['id']}", "edge": b["edge"], "front": b["front"],
                                    "rect": b["rect"], "height_m": b["size_m"][2]}
    return done


def dressing_placed(report: list, lay: dict) -> list:
    """Each dressing item (with_dressing) with how many of its piece, and its copies, the layout placed."""
    got = {}
    for q in lay.get("pieces") or []:
        got[q.get("copy_of") or q["id"]] = got.get(q.get("copy_of") or q["id"], 0) + 1
    return [dict(r, placed=got.get(r["piece"], 0)) if r.get("piece") else dict(r) for r in report or []]


def build(spec: dict, set_id: str, library=None) -> tuple:
    # a piece another floor piece stands on is that piece's base (place_layout.take_bases): no mesh of the kind
    # the spec gave it, no place of its own; laid under what it carries once the room is solved
    spec, bases = L.take_bases(spec)
    rec, res = records_for(spec, library)
    # (a made floor in the open, open_floor: laid out and built as a room, its world round it)
    open_top = bool(spec.get("_open_top"))
    spec0, world = spec, spec.get("kind") == "outdoors" and not open_top
    marks = world_marks(spec, library) if world else {}
    if open_top:                              # (its world is not laid out: it lies round the floor)
        spec = dict(spec, pieces=[p for p in spec.get("pieces") or [] if p.get("kind") != "terrain"])
        # (nothing is overhead: a thing taller than the stage, a post or a tower on the floor, is no fault of the
        # set's, sets/layout.py check "taller than the room")
        tall = max([float(rec[p["id"]]["size_m"][2]) for p in spec["pieces"] if p["id"] in rec] + [0.0])
        rm = spec.get("_room_m")
        if rm and tall + OPEN_HEAD_M > float(rm[2]):
            spec = dict(spec, _room_m=[rm[0], rm[1], round(tall + OPEN_HEAD_M, 2)])
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
    # (a picture stage has no wall to hang anything on: whatever hangs there hangs on a post, however low)
    hang(lay, hung, rec, (STAGE_NO_EDGE_M if spec.get("_picture_stage") else float(spec.get("_edge_m") or OPEN_EDGE_M))
         if open_top else None)
    lay["bases"] = L.lay_bases(lay, bases)
    ss = L.to_set_spec(spec, lay, data)
    kit = kit_for(spec.get("period"))
    out = dict(id=set_id, place=spec.get("place"),        # (the plan's place it was built for: from_plan picks it)
               description=spec.get("room") or "", style=spec.get("period"), tags=[spec.get("kind")],
               **dress(ss, lay, spec, kit))
    if crowd_line(spec):                                 # the place's crowd, in the story's and the director's words
        out["crowd"] = crowd_line(spec)
        if spec.get("_crowd_size") == "many":
            out["crowd_size"] = "many"
    bare = (spec.get("_concept") or {}).get("bare") or {}       # the place's own walls and floor: the concept's
    if open_top and not (bare.get("walls") or {}).get("colour"):
        # (the edge of a made floor whose picture shows no stretch of it: of the floor's own material, never the
        # period's wall of a room)
        fm = next((w.get("material") for w in out["structure"] if w.get("kind") == "floor"), None)
        for w in out["structure"]:
            if w.get("kind") == "wall" and fm:
                w["material"] = fm
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
    if open_top:                              # (a made floor in the open: its floor's own picture, no wall's)
        wall_pictures(out, spec, library, lay)
    elif not world:                           # (a place in the open has no walls to face: a ring of wall panels
        wall_faces(out, lay, spec, library)   # stood round the forge yard's stage, 11-18)
    cabin_stage(out, spec, library)
    hung_posts(out, lay)
    based = base_boxes(out, lay)
    for b in lay["bases"]:                   # (what else stands where a base is: reported, it is laid out first)
        on_ = [q["name"] for q in lay["pieces"] if b.get("rect") and q["name"] not in b["of"] and not q.get("hang")
               and not q.get("flat") and L.overlap(q["rect"], b["rect"])]
        if on_:
            next(d for d in based if d["id"] == b["id"])["stands_on_it"] = on_
    staged = world_stage(out, spec0, library) if world else open_floor_stage(out, spec0, library) if open_top else None
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
              **({"bases": based} if based else {}),
              **({"base_dropped_relations": spec["_base_dropped_relations"]} if spec.get("_base_dropped_relations") else {}),
              **({"room_light": "a grid of soft lights: the room had no light of its own"} if lit else {}),
              "unresolved_phrases": [ph for ph in spec.get("_phrases") or [] if ph not in ss["named_places"]],
              **({"dressing": dressing_placed(spec.get("_dressing"), lay)} if spec.get("_dressing") else {}),
              **({"picture_left": spec["_picture_left"]} if spec.get("_picture_left") else {}),
              **({"picture_refused": [n for n in lay.get("refused") or [] if n.get("what") != "repeated"]}
                 if lay.get("refused") else {})}
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
    if "--dress" in a:                                 # the place's dressing (scenes/place_dress.py), each item placed
        spec = with_dressing(spec, json.loads(Path(a[a.index("--dress") + 1]).read_text()))
    if "--time" in a:                                  # the place's time of day (scenes/place_spec.py time_of_day)
        spec["_time"] = a[a.index("--time") + 1]
    spec = with_crowd(spec)
    spec, _ = L.take_bases(spec)                       # (a base is no piece to make: places/generate.py reads this)
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
