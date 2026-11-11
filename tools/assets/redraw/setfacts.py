"""What a library set holds, as facts for an image model's words (stdlib): things, their places, which are lit.

    things(set_dir) -> [{"name", "words", "center": [x, y, z], "half": [hx, hy, hz], "lit": bool, "fire": bool,
                         "kind": "fixture" | "prop" | "window" | "door"}]
    daylight(set_dir) -> "daylight" | "low sun" | None

From the set builder's spec (spec.json: fixtures and props with library asset ids and floor positions, wall openings,
lights) and each asset's library size. A thing is lit when one of the set's point/spot lights sits inside its
footprint (a lamp's flame, a fire in a grate); a lit thing whose name or asset names a fire place is a fire.
Words come from the set builder's own names for things ("side_table" -> "side table"), never from story text; a
window's view is the "view_words" of the set's outside backdrop, when it has one. A thing's look (its colours,
pattern, fabric, what a picture shows) is the placement's "look" in the set spec, else the asset's own "look" in
its library entry: said in every shot that sees the thing, it keeps a redraw from repainting the rug, a chair's
cover or a picture's subject differently from one shot to the next.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

FIRE = ("fireplace", "hearth", "stove", "grate", "brazier", "campfire")


def _lib(set_dir: Path) -> Path:
    return set_dir.parent.parent            # <library>/set/<id> -> <library>


def _size(lib: Path, asset: str) -> list:
    for cls in ("prop", "set", "character"):
        p = lib / cls / asset / "asset.json"
        if p.exists():
            return json.loads(p.read_text()).get("size_m") or [0.5, 0.5, 0.5]
    return [0.5, 0.5, 0.5]


def _short(lib: Path, asset: str) -> str | None:
    """The asset's own short words: its library description up to the first colon or comma ("Victorian paraffin
    (oil) table lamp, lit, 0.60 m: ..." -> "Victorian paraffin (oil) table lamp"). Naming the very thing (which
    lamp, what the picture shows) keeps a redraw from swapping it for another of its kind between shots."""
    for cls in ("prop", "set", "character"):
        p = lib / cls / asset / "asset.json"
        if p.exists():
            d = json.loads(p.read_text()).get("description") or ""
            s = d.split(":")[0].split(". ")[0]
            depth, cut = 0, len(s)
            for i, ch in enumerate(s):               # the first clause, commas inside brackets kept
                depth += (ch == "(") - (ch == ")")
                if ch == "," and depth == 0:
                    cut = i
                    break
            s = s[:cut].split(" 0.")[0].split(" front -Y")[0].strip()
            return s if s and len(s) <= 160 else None
    return None


def _look(lib: Path, asset: str) -> str | None:
    for cls in ("prop", "set", "character"):
        p = lib / cls / asset / "asset.json"
        if p.exists():
            return json.loads(p.read_text()).get("look") or None
    return None


def things(set_dir) -> list:
    set_dir = Path(set_dir)
    spec = json.loads((set_dir / "spec.json").read_text())
    lib = _lib(set_dir)
    sx, sy = spec["size_m"][0] / 2, spec["size_m"][1] / 2
    out = []
    by_name = {}
    for kind in ("fixtures", "props"):
        for f in spec.get(kind, []):
            if f.get("outside") or "row" in f:      # a backdrop beyond the walls is seen only as the window's view
                continue
            sz = _size(lib, f["asset"])
            fc = f.get("facing", [0, -1])
            yaw = math.atan2(fc[1], fc[0]) + math.pi / 2          # front -Y at yaw 0
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            hx, hy = (sz[0] * c + sz[1] * s) / 2, (sz[0] * s + sz[1] * c) / 2
            z0 = float(f.get("on_z_m", f["position"][2] if len(f["position"]) > 2 else 0.0))
            t = {"name": f["name"], "asset": f["asset"], "kind": kind[:-1], "short": _short(lib, f["asset"]),
                 "look": f.get("look") or _look(lib, f["asset"]),
                 "center": [f["position"][0], f["position"][1], z0 + sz[2] / 2], "half": [hx, hy, sz[2] / 2],
                 "on": f.get("on"), "lit": False, "fire": False}
            out.append(t)
            by_name[f["name"]] = t
    for w in spec.get("structure", []):
        if w.get("kind") != "wall":
            continue
        for o in w.get("openings", []):
            a, b = o["from_m"], o["to_m"]
            z0, z1 = o.get("sill_m", 0.0), o.get("head_m", 2.1)
            side = w["side"]
            if side in ("west", "east"):
                x = -sx if side == "west" else sx
                center, half = [x, (a + b) / 2, (z0 + z1) / 2], [0.05, abs(b - a) / 2, (z1 - z0) / 2]
            else:
                y = sy if side == "north" else -sy
                center, half = [(a + b) / 2, y, (z0 + z1) / 2], [abs(b - a) / 2, 0.05, (z1 - z0) / 2]
            k = "window" if "window" in o["kind"] else "door" if "door" in o["kind"] else o["kind"]
            out.append({"name": f"{k}_{side}", "asset": None, "kind": k, "center": center, "half": half,
                        "wall": side, "lit": False, "fire": False})
    for L in spec.get("lights", []):
        if L.get("type") not in ("point", "spot") or "position" not in L:
            continue
        p = L["position"]
        best = None
        for t in out:
            if t["kind"] not in ("fixture", "prop"):
                continue
            c, h = t["center"], t["half"]
            if all(abs(p[i] - c[i]) <= h[i] + (0.12 if i < 2 else 0.35) for i in range(3)):
                vol = h[0] * h[1] * h[2]
                if best is None or vol < best[0]:
                    best = (vol, t)
        if best:
            t = best[1]
            t["lit"] = True
            t["light_pos"] = list(p)
            t["fire"] = any(w in (t["name"] + " " + (t["asset"] or "")).lower() for w in FIRE)
    # where each piece of furniture stands in the room, by the room's landmarks (the fire, the windows), the same
    # words in every shot: two armchairs named only by their looks swapped looks between shots (11-11)
    marks = [t for t in out if t.get("fire") or t["kind"] == "window"]
    for t in out:
        if t["kind"] != "fixture" or t.get("fire") or not marks:
            continue
        m = min(marks, key=lambda q: math.dist(q["center"][:2], t["center"][:2]))
        mw = "the fireplace" if m.get("fire") else "the window"
        to = [m["center"][0], m["center"][1]]                 # from the room's middle, facing the landmark
        right = [to[1], -to[0]]
        d = [t["center"][0] - m["center"][0], t["center"][1] - m["center"][1]]
        along = d[0] * right[0] + d[1] * right[1]
        n = math.hypot(*right) or 1.0
        along /= n
        reach = max(m["half"][0], m["half"][1])
        t["place"] = (f"in front of {mw}" if abs(along) <= reach * 0.6 else
                      f"to the {'right' if along > 0 else 'left'} of {mw}")
        t["large"] = max(t["half"][0], t["half"][1]) >= 0.4
    for t in out:
        if t["kind"] == "prop" and t["half"][0] * t["half"][1] >= 1.0:
            t["large"] = True              # a carpet
    view = next((f.get("view_words") for k in ("fixtures", "props") for f in spec.get(k, [])
                 if f.get("outside") and f.get("view_words")), None)
    for t in out:
        n = " ".join(w for i, w in enumerate(t["name"].split("_")) if i == 0 or len(w) > 2)   # "candle_w" -> "candle"
        if t["kind"] == "window":
            t["words"] = "the window"
            if view:                      # what the set puts outside it (a backdrop's own words in the spec)
                t["view"] = view
        elif t["kind"] == "door":
            t["words"] = "the door"
        else:
            own = "; ".join(x for x in (t.get("short"), t.get("look")) if x) if t["kind"] in ("fixture", "prop") else ""
            what = f"the {n}" + (f" ({own})" if own else "")
            t["words"] = what + (f" on the {by_name[t['on']]['name'].replace('_', ' ')}" if t.get("on") in by_name else "")
        if t["fire"]:
            t["words"] += " with a fire burning in the grate, flames and glowing coals"
        elif t["lit"]:
            t["words"] += ", lit, its flame burning"
    return out


def daylight(set_dir) -> str | None:
    spec = json.loads((Path(set_dir) / "spec.json").read_text())
    for L in spec.get("lights", []):
        if L.get("type") == "sun":
            v = L.get("toward_sun", [0, 0, 1])
            el = math.degrees(math.asin(v[2] / max(1e-6, math.sqrt(sum(x * x for x in v)))))
            return "daylight" if el > 15 else "low sun" if el > 0 else None
    return None
