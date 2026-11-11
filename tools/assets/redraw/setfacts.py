"""What a library set holds, as facts for an image model's words (stdlib): things, their places, which are lit.

    things(set_dir) -> [{"name", "words", "center": [x, y, z], "half": [hx, hy, hz], "lit": bool, "fire": bool,
                         "kind": "fixture" | "prop" | "window" | "door"}]
    daylight(set_dir) -> "daylight" | "low sun" | None

From the set builder's spec (spec.json: fixtures and props with library asset ids and floor positions, wall openings,
lights) and each asset's library size. A thing is lit when one of the set's point/spot lights sits inside its
footprint (a lamp's flame, a fire in a grate); a lit thing whose name or asset names a fire place is a fire.
Words come from the set builder's own names for things ("side_table" -> "side table"), never from story text.
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


def things(set_dir) -> list:
    set_dir = Path(set_dir)
    spec = json.loads((set_dir / "spec.json").read_text())
    lib = _lib(set_dir)
    sx, sy = spec["size_m"][0] / 2, spec["size_m"][1] / 2
    out = []
    by_name = {}
    for kind in ("fixtures", "props"):
        for f in spec.get(kind, []):
            sz = _size(lib, f["asset"])
            fc = f.get("facing", [0, -1])
            yaw = math.atan2(fc[1], fc[0]) + math.pi / 2          # front -Y at yaw 0
            c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
            hx, hy = (sz[0] * c + sz[1] * s) / 2, (sz[0] * s + sz[1] * c) / 2
            z0 = float(f.get("on_z_m", 0.0))
            t = {"name": f["name"], "asset": f["asset"], "kind": kind[:-1],
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
    for t in out:
        n = t["name"].replace("_", " ")
        if t["kind"] == "window":
            t["words"] = "the window"
        elif t["kind"] == "door":
            t["words"] = "the door"
        else:
            t["words"] = f"the {n}" + (f" on the {by_name[t['on']]['name'].replace('_', ' ')}" if t.get("on") in by_name else "")
        if t["fire"]:
            t["words"] += " with a fire burning in the grate, flames and glowing coals"
        elif t["lit"]:
            t["words"] += ", lit, glowing warmly"
    return out


def daylight(set_dir) -> str | None:
    spec = json.loads((Path(set_dir) / "spec.json").read_text())
    for L in spec.get("lights", []):
        if L.get("type") == "sun":
            v = L.get("toward_sun", [0, 0, 1])
            el = math.degrees(math.asin(v[2] / max(1e-6, math.sqrt(sum(x * x for x in v)))))
            return "daylight" if el > 15 else "low sun" if el > 0 else None
    return None
