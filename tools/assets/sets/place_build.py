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
    rec = {p["id"]: {"asset": p["asset_id"], "size_m": by[p["asset_id"]].get("size_m")}
           for p in res["pieces"] if p["status"] == "library"}
    # a piece still to be generated stands in the room as the library's nearest of its kind until then (reported:
    # a bedroom was built without its bed because the bed's look matched no record closely enough)
    for p in res["pieces"]:
        if p["status"] == "generate" and p.get("nearest"):
            rec[p["id"]] = {"asset": p["nearest"]["asset_id"], "size_m": by[p["nearest"]["asset_id"]].get("size_m")}
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
        if "drapes" in dress_by:
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
            d = f"{rc['piece_kind'].replace('_', ' ')} ({rc['look'].replace(CABIN_LOOK, '')})"   # (not how it was drawn)
        d = d or (it.get("name") or "").replace("_", " ")
        if out.get("style") and d.lower().startswith(out["style"].lower() + " "):
            d = d[len(out["style"]) + 1:]         # (the period is said once, for the room)
        d = d[0].lower() + d[1:] if d[:2] != "A " else "a" + d[1:]
        held.append(d if re.match(r"(a|an|the) ", d) else ("an " if d[:1] in "aeiou" else "a ") + d)
    for o in lay.get("openings") or []:
        w = kw.get(o.get("kind"))
        if w and w not in held:
            held.append(w)
    held = list(dict.fromkeys(held))
    return (f"A {(out.get('style') or '').replace('_', ' ')} {kind.replace('_', ' ')}: ".replace("  ", " ").replace("A v", "A V")
            + "; ".join(held) + ".", kw.get("surfaces") or "")


CABIN_LOOK = ("the whole cabin as one open model for filming inside it: no glass in its canopy frame, and the nose "
              "cut away just ahead of the instrument panel, so that from in front and above one looks over the panel "
              "straight at the seats: ")


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
        if c.get("look"):
            p["look"] = c["look"]
        kept.append(p)
    sp["pieces"] = kept
    ids = {p["id"] for p in kept}
    sp["relations"] = [r for r in sp.get("relations") or [] if r.get("a") in ids and r.get("b") in ids]
    if concept.get("space") == "cockpit":
        seen = [x["name"] for x in concept.get("seen") or [] if x.get("name") and "view" not in x["name"]]
        sp.update(kind="cockpit", openings=[], relations=[], pieces=[{
            "id": "p1", "kind": "vehicle_cabin", "look": CABIN_LOOK + ", ".join(dict.fromkeys(seen)),
            "phrases": [ph for p in spec.get("pieces") or [] for ph in p.get("phrases") or []], "against": "free",
            "source": concept.get("concept"), "seats_abreast": int(concept.get("seats_abreast") or 1)}])
        sp["named_places"] = {ph: "p1" for ph in spec.get("named_places") or {}}
        # what lies outside the cabin: the director's own line for the place when it is an outside place
        m = re.match(r"\s*outside\s*:\s*(.+)", str(spec.get("_director_room") or ""), re.I | re.S)
        if m:
            sp["outside"] = m.group(1).strip().rstrip(".")
    sp["_concept"] = {k: concept.get(k) for k in ("concept", "space", "surfaces_words")}
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
    if (lib / "prop" / vid / "asset.json").exists():
        d = ln / 2 + VIEW_AHEAD_M
        out["fixtures"].append({"asset": vid, "name": "outside_view", "outside": True, "view_words": words,
                                "position": [round(x + f[0] * d, 3), round(y + f[1] * d, 3),
                                             round(ht / 2 - VIEW_SIZE_M[1] / 2, 3)],
                                "facing": [round(-f[0], 4), round(-f[1], 4)]})
    # (a made cabin is a closed shell: a light outside its nose lit nobody, so the view's light is set inside it,
    # ahead of and above the seats, without shadows)
    key = {"type": "rect", "size_m": [1.2, 0.6], "lumens": 4200, "kelvin": 5200, "radius_m": 8, "shadows": False,
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
    glow = {"type": "point", "lumens": 420, "kelvin": 6500, "radius_m": 3, "source_radius_m": 0.15, "shadows": False,
            "position": [round(x - f[0] * ln * 0.08, 3), round(y - f[1] * ln * 0.08, 3), 0.95]}
    out["lights"] = [{"type": "sky", "intensity": 0.05, "atmosphere": False}, key, glow]
    out["surfaces_words"] = (f"Outside, seen through the canopy and its windows: {words}. The light comes in from "
                             "outside, and the instruments glow.")
    out["view"] = view_spec(words)


def build(spec: dict, set_id: str, library=None) -> tuple:
    rec, res = records_for(spec, library)
    lay = L.solve_recover(spec, rec)          # solve, then the ladder (never drops a piece the plan names)
    ss = L.to_set_spec(spec, lay)
    kit = json.loads((HERE / "period_kits.json").read_text()).get(spec.get("period")) or \
        json.loads((HERE / "period_kits.json").read_text())["victorian"]
    out = dict(id=set_id, place=spec.get("place"),        # (the plan's place it was built for: from_plan picks it)
               description=spec.get("room") or "", style=spec.get("period"), tags=[spec.get("kind")],
               **dress(ss, lay, spec, kit))
    out["words"], out["surfaces_words"] = room_words(out, lay, kit, {r["asset_id"]: r for r in P.load_records(library)})
    if (spec.get("_concept") or {}).get("surfaces_words"):       # the place's own surfaces, as its concept shows them
        out["surfaces_words"] = spec["_concept"]["surfaces_words"]
        out["concept"] = spec["_concept"]["concept"]
    for it in out["fixtures"] + out["props"]:                    # (how a cabin was drawn is not how it looks)
        if it.get("look"):
            it["look"] = it["look"].replace(CABIN_LOOK, "")
    cabin_stage(out, spec, library)
    # a window or door is a set thing named by its wall (setfacts: "<window|door>_<wall>"), not the layout's name
    alias = {o["name"]: f"{'window' if 'window' in o['kind'] else 'door'}_{o['wall']}" for o in lay["openings"]
             if o.get("name") and o.get("wall") and o["kind"] in ("window", "door")}
    ss["named_places"] = {ph: alias.get(v, v) for ph, v in ss["named_places"].items()}
    report = {"pieces": res["summary"], "left_out": [p for p in res["pieces"] if p["status"] != "library"],
              "layout_ok": lay["ok"], "unmet": lay["unmet"], "status": lay.get("status"),
              "ladder": lay.get("ladder"), "dropped": lay.get("dropped"), "reask": lay.get("reask"), "checks": L.check_layout(lay, spec),
              "named_places": ss["named_places"],
              "unresolved_phrases": [ph for ph in spec.get("_phrases") or [] if ph not in ss["named_places"]]}
    return out, ss["named_places"], report


def main():
    a = sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    spec = spec.get("sheet", spec)
    if "--concept" in a:
        spec = with_concept(spec, json.loads(Path(a[a.index("--concept") + 1]).read_text()))
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
