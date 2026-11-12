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
    res = P.resolve(spec.get("pieces") or [], recs)
    by = {r["asset_id"]: r for r in recs}
    rec = {p["id"]: {"asset": p["asset_id"], "size_m": by[p["asset_id"]].get("size_m")}
           for p in res["pieces"] if p["status"] == "library"}
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
        lights.append(dict(kit["lights"]["window_rect"], position=[round(p[0] + n[0] * 0.1, 3), round(p[1] + n[1] * 0.1, 3), 1.75],
                           toward=[round(p[0] + n[0] * 3.5, 3), round(p[1] + n[1] * 3.5, 3), 1.45]))
    if wins:                                    # daylight from the first window's side
        n = NORMAL[wins[0]["wall"]]
        lights.insert(0, dict(kit["lights"]["sun"], toward_sun=[round(-n[0] * 0.8 + n[1] * 0.3, 3),
                                                               round(-n[1] * 0.8 - n[0] * 0.3, 3), 0.35]))
    for f in fixtures:
        if f.get("name", "").startswith("fireplace"):
            fx, fy = f["facing"]
            dx, dy, dz = kit["lights"]["fire_offset_m"]
            lights.append(dict(kit["lights"]["fire"], position=[round(f["position"][0] - fx * dy, 3),
                                                              round(f["position"][1] - fy * dy, 3), dz]))
    for pr in props:
        if "lamp" in pr.get("asset", "") or pr.get("name", "").startswith("lamp"):
            lights.append(dict(kit["lights"]["lamp"], position=[pr["position"][0], pr["position"][1],
                                                               round((pr.get("on_z_m") or 0.0) + 0.42, 3)]))
    out.update(fixtures=fixtures, props=props, lights=lights)
    cams = []
    for i, (sx, sy) in enumerate(((1, -1), (-1, -1), (1, 1), (-1, 1))):
        cams.append({"name": f"wide_{i}", "position": [round(sx * (W / 2 - 0.5), 2), round(sy * (D / 2 - 0.5), 2), 1.6],
                     "target": [round(-sx * 0.6, 2), round(-sy * 0.6, 2), 1.1], "focal_mm": 20})
    out["cameras"] = cams
    out["validation"] = {"figures": [{"position": [0.0, 0.0], "facing": [0, -1]}]}
    return out


def build(spec: dict, set_id: str, library=None) -> tuple:
    rec, res = records_for(spec, library)
    lay = L.solve(spec, rec)
    ss = L.to_set_spec(spec, lay)
    kit = json.loads((HERE / "period_kits.json").read_text()).get(spec.get("period")) or \
        json.loads((HERE / "period_kits.json").read_text())["victorian"]
    out = dict(id=set_id, description=spec.get("room") or "", style=spec.get("period"), tags=[spec.get("kind")],
               **dress(ss, lay, spec, kit))
    report = {"pieces": res["summary"], "left_out": [p for p in res["pieces"] if p["status"] != "library"],
              "layout_ok": lay["ok"], "unmet": lay["unmet"], "checks": L.check_layout(lay, spec),
              "named_places": ss["named_places"],
              "unresolved_phrases": [ph for ph in spec.get("_phrases") or [] if ph not in ss["named_places"]]}
    return out, ss["named_places"], report


def main():
    a = sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    spec = spec.get("sheet", spec)
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
