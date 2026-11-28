"""A place's world plan written by the directing model (stdlib): the big solid things of the place and of what lies
round it out to the horizon, each by kind, real size and position (sets/world_parts.py, world/1). Validated by code
and re-asked with the faults stated (never patched by hand); cached by the prompt.

    python tools/assets/scenes/world_plan.py PLACE_SPEC.json OUT.json [--concept CONCEPT.json] [--time WORDS]
                                             [--cache DIR] [--url URL]

The model decides what stands where and how large; code builds every part as plain solids at that size. Structure
and scale only: what a surface looks like is the drawing step's.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "sets"))
import world_parts as WP  # noqa: E402

SEED = 7
MAX_ASKS = 3
CLEAR_R_M = 1.5            # the middle of the stage is kept clear of solids: the people stand there
HORIZON_M = 2000           # a ground or water at least this wide: the world reaches the horizon
MAX_PARTS = 220
# real sizes a part's numbers must lie in (metres): a wrong scale is the fault this plan exists to prevent
RANGES = {"building": {"w": (2.5, 60), "d": (2.5, 80), "h": (2.2, 45)}, "mast": {"height_m": (3, 60)},
          "hull": {"length_m": (3, 120), "beam_m": (1, 25), "depth_m": (0.5, 15)}, "lamp": {"height_m": (1.5, 8)}}

ASK = """You are laying out a film set as a SOLID 3D WORLD for the place "{place}" of a story.
What the story says of the place: {words}
{seen}{time}
Write the big solid things of the place and of everything round it, out to the horizon, each by its kind, its REAL
size in metres and where it stands. Small things (furniture, tools, things people hold) are not yours: leave them out.
Surfaces need no detail, only the right shape, size, place and material.

Frame: metres; x = east, y = north, z = up. The people of the story stand on a flat floor round the origin [0, 0],
its top at z = 0. Scale: a person is 1.75 m tall, a door 2 m, one storey of a house 2.8 m.

Reply with JSON only:
{{"schema": "world/1",
 "stage_m": [W, D, H],        the flat floor the people stand and walk on, centred on the origin (H: the clear height)
 "ground": [{{"name", "material", "z_m"}}],     what lies under and round everything, out to the horizon: water, or land
 "parts": [ ... ],
 "spots": {{{spots}}},        where each named thing of the place stands: {{"at": [x, y], "facing": [dx, dy]}}
 "night": true | false}}

Part kinds. Every part has "kind", "name" (short, lowercase, what it is), "at": [x, y, z] (the middle of its BASE)
and may have "yaw_deg" (turned counter-clockwise seen from above):
- "deck": a flat floor people can stand on, its TOP at at's z. "size_m": [x, y], "material", "rail_m" (height of a
  rail or low wall round it, 0 for none), "rail_open": the sides without it, of "north", "south", "east", "west".
- "block": any plain solid (a wall, a cabin, a quay, a rock, steps). "size_m": [x, y, z], "material".
- "building": a house, a shed, a hall, a tower. "size_m": [width x, depth y, height to the eaves], "roof": "gable" or
  "flat", "pitch_deg", "ridge": "x" or "y", "wall": material, "roof_material", "windows": {{"rows", "cols", "lit":
  the share of windows with light in them, 0 to 1, "sides": [...]}}, "door_side".
- "hull": the body of a ship or boat under its deck, pointed toward +x before "yaw_deg". "at" is the middle of its
  DECK LINE. "length_m", "beam_m", "depth_m" (deck down to the keel; the water stands about two thirds up it).
- "mast": "height_m", "diameter_m", "yards": [{{"z_m", "length_m", "sail": "furled" | "set" | null}}],
  "stays": [[x, y, z], ...] where its ropes come down to.
- "pole": a round bar from "at" to "to": [x, y, z], "diameter_m", "material" (a spar, a post, a flagstaff).
- "lamp": a post with a lit head. "height_m", "lumens" (a lantern 600, a street lamp 1500).
- "lights": a string of small lamps hanging from "from": [x, y, z] to "to": [x, y, z], "count", "sag_m".
- "hill": far high land. "size_m": [x, y, z], "material".
Materials: {materials}.

Rules:
- The stage's floor is real: a "deck" (or the ground itself at z_m 0) lies under the whole stage, its top at z = 0.
- Nothing solid stands within {clear} m of the origin, and every spot is on the stage, clear of every solid.
- Build the place itself whole (all of it, not only the side a picture shows), then what is near it (what one
  would see standing there and turning right round), then the far things, and the ground or water to the horizon.
- Three distances: the place itself in full; within about 100 m the nearby things as plain buildings and blocks;
  beyond that only large masses (hills, a far shore, a far town as a few big blocks).
- Where the light of the place comes from at this hour is yours too: lamps and strings of lights where the story
  has them, lit windows at night.
- Sizes are real. A thing the people stand beside is as large beside them as it would be in life.
"""


def words_of(spec: dict) -> str:
    s = spec.get("sheet", spec)
    out = [str(s.get(k) or "").strip() for k in ("_director_room", "whole", "room", "outside")]
    fx = s.get("fixtures") or []
    return " ".join(dict.fromkeys(x for x in out if x)) + (f" Fixed things named: {', '.join(fx)}." if fx else "")


def spot_ids(spec: dict) -> list:
    s = spec.get("sheet", spec)
    return [p["id"] for p in s.get("pieces") or [] if p.get("id")]


def prompt(spec: dict, concept: dict | None, time: str) -> str:
    s = spec.get("sheet", spec)
    seen = ""
    if concept:
        things = [f"{x['name']}" + (f" (about {max(x['real_m']):g} m)" if x.get("real_m") else "")
                  for x in concept.get("seen") or [] if x.get("name")]
        bits = [f"round it: {concept['around']}" if concept.get("around") else "",
                f"far off: {concept['far']}" if concept.get("far") else "",
                f"the light: {concept['light']}" if concept.get("light") else "",
                f"things seen there: {', '.join(dict.fromkeys(things))}" if things else ""]
        seen = "A picture of the place shows, " + "; ".join(b for b in bits if b) + ".\n" if any(bits) else ""
    pieces = {p["id"]: p for p in s.get("pieces") or [] if p.get("id")}
    spots = ", ".join(f'"{i}" ({(p.get("look") or p.get("kind") or "")[:80]})' for i, p in pieces.items())
    return ASK.format(place=s.get("place") or "", words=words_of(spec), seen=seen,
                      time=f"The hour: {time}.\n" if time else "", spots=spots or "(none)",
                      materials=", ".join(m for m in WP.MATERIALS if not m.startswith(("glow_", "window_"))),
                      clear=CLEAR_R_M)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _in_solid(xy, sol, pad=0.0) -> bool:
    cx, cy, hx, hy, z0, z1, yaw = sol
    if z1 < 0.3 or z0 > 2.0:                       # (under the floor, or overhead: nothing a person walks into)
        return False
    c, s = math.cos(math.radians(-yaw)), math.sin(math.radians(-yaw))
    dx, dy = xy[0] - cx, xy[1] - cy
    lx, ly = dx * c - dy * s, dx * s + dy * c
    return abs(lx) <= hx + pad and abs(ly) <= hy + pad


def faults(plan, spec: dict) -> list:
    """What is wrong with a plan, in words the model can act on; [] when it can be built."""
    if not isinstance(plan, dict) or plan.get("schema") != WP.SCHEMA:
        return [f'the answer is not a JSON object with "schema": "{WP.SCHEMA}"']
    f = []
    st = plan.get("stage_m")
    if not (isinstance(st, list) and len(st) == 3 and all(_num(v) and v > 0 for v in st)):
        return ['"stage_m" must be three positive numbers [W, D, H]']
    if not (3 <= st[0] <= 80 and 3 <= st[1] <= 80):
        f.append(f"the stage is {st[0]:g} by {st[1]:g} m: between 3 and 80 m each way")
    parts = plan.get("parts")
    if not isinstance(parts, list) or not parts:
        return ['"parts" must list the solid things of the place']
    if len(parts) > MAX_PARTS:
        f.append(f"{len(parts)} parts: at most {MAX_PARTS}; far things as fewer, larger masses")
    names = set()
    for i, p in enumerate(parts):
        k, n = p.get("kind"), p.get("name") or f"part {i}"
        if k not in WP.KINDS:
            f.append(f'{n}: unknown kind "{k}"; the kinds are {", ".join(WP.KINDS)}')
            continue
        if k != "lights" and not (isinstance(p.get("at"), list) and len(p["at"]) in (2, 3) and all(_num(v) for v in p["at"])):
            f.append(f'{n}: "at" must be [x, y, z]')
        names.add(n)
        for m in (p.get(x) for x in ("material", "wall", "roof_material")):
            if m and m not in WP.MATERIALS:
                f.append(f'{n}: unknown material "{m}"')
        need = {"deck": ("size_m",), "block": ("size_m",), "building": ("size_m",), "hill": ("size_m",),
                "hull": ("length_m", "beam_m"), "mast": ("height_m",), "pole": ("to",), "lights": ("from", "to")}.get(k, ())
        for key in need:
            v = p.get(key)
            if key == "size_m":
                want = 2 if k == "deck" else 3
                if not (isinstance(v, list) and len(v) >= want and all(_num(x) and x > 0 for x in v[:want])):
                    f.append(f'{n}: "size_m" must be {want} positive numbers')
            elif key in ("to", "from"):
                if not (isinstance(v, list) and len(v) == 3 and all(_num(x) for x in v)):
                    f.append(f'{n}: "{key}" must be [x, y, z]')
            elif not (_num(v) and v > 0):
                f.append(f'{n}: "{key}" must be a positive number')
        rg = RANGES.get(k) or {}
        vals = dict(zip(("w", "d", "h"), p.get("size_m") or [])) if k == "building" else p
        for key, (lo, hi) in rg.items():
            v = vals.get(key)
            if _num(v) and not lo <= v <= hi:
                f.append(f"{n}: {key} {v:g} m is not a real size for a {k} ({lo:g} to {hi:g} m)")
    if f:
        return f
    try:
        built = WP.build(plan)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return [f"the plan cannot be built: {type(exc).__name__} {exc}"]
    # the stage's floor: a deck whose top is at z = 0 under the whole stage, or a ground at z = 0
    hw, hd = st[0] / 2, st[1] / 2
    floor = any(abs((g.get("z_m") or 0)) < 0.06 for g in plan.get("ground") or [])
    for p in parts:
        if p["kind"] == "deck" and abs((list(p["at"]) + [0])[2]) < 0.06 and not (p.get("yaw_deg") or 0) % 180:
            a, sz = p["at"], p["size_m"]
            floor = floor or (a[0] - sz[0] / 2 <= -hw + 0.05 and a[0] + sz[0] / 2 >= hw - 0.05
                              and a[1] - sz[1] / 2 <= -hd + 0.05 and a[1] + sz[1] / 2 >= hd - 0.05)
    if not floor:
        f.append("no floor under the whole stage: one deck with its top at z = 0 that covers the stage "
                 f"({st[0]:g} by {st[1]:g} m round the origin), or a ground at z_m 0")
    if not any((g.get("extent_m") or 6000) >= HORIZON_M for g in plan.get("ground") or []):
        f.append('"ground" must give what lies round everything out to the horizon (water or land)')
    for sol in built["solids"]:
        if _in_solid([0.0, 0.0], sol, CLEAR_R_M):
            f.append(f"a solid at [{sol[0]:g}, {sol[1]:g}] stands within {CLEAR_R_M} m of the origin, where the "
                     "people stand")
    spots = plan.get("spots") or {}
    for i in spot_ids(spec):
        sp = spots.get(i)
        if not (isinstance(sp, dict) and isinstance(sp.get("at"), list) and len(sp["at"]) >= 2
                and all(_num(v) for v in sp["at"][:2])):
            f.append(f'"spots" must say where "{i}" stands: {{"at": [x, y], "facing": [dx, dy]}}')
            continue
        x, y = sp["at"][:2]
        if abs(x) > hw or abs(y) > hd:
            f.append(f'spot "{i}" at [{x:g}, {y:g}] is off the stage ({st[0]:g} by {st[1]:g} m round the origin)')
        elif any(_in_solid([x, y], sol, 0.2) for sol in built["solids"]):
            f.append(f'spot "{i}" at [{x:g}, {y:g}] is inside a solid')
    return f


def parse(txt: str):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def ask(url: str, text: str, cache: Path | None) -> str:
    key = hashlib.sha256(f"world/1\n{SEED}\n{text}".encode()).hexdigest()[:24]
    fp = cache / f"{key}.txt" if cache else None
    if fp and fp.exists():
        return fp.read_text()
    body = {"model": "default", "messages": [{"role": "user", "content": text}], "temperature": 0.0, "seed": SEED,
            "max_tokens": 9000, "chat_template_kwargs": {"enable_thinking": False}}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=1800) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    if fp:
        cache.mkdir(parents=True, exist_ok=True)
        fp.write_text(txt)
    return txt


def write(spec: dict, concept: dict | None, time: str, url: str, cache: Path | None, log=print):
    """(plan or None, faults, asks): the first answer that can be built, else the last one with its faults."""
    text = prompt(spec, concept, time)
    plan, fl, said = None, ["no answer"], text
    for n in range(MAX_ASKS):
        plan = parse(ask(url, said, cache))
        fl = faults(plan, spec)
        if not fl:
            return plan, [], n + 1
        log(f"world plan: answer {n + 1} has {len(fl)} fault(s): {fl[:4]}")
        said = (text + "\nYour last answer had these faults; write the whole plan again without them:\n- "
                + "\n- ".join(fl[:20]) + ("\nYour last answer:\n" + json.dumps(plan)[:6000] if plan else ""))
    return plan, fl, MAX_ASKS


def main():
    a = sys.argv[1:]
    spec = json.loads(Path(a[0]).read_text())
    out = Path(a[1])
    concept = json.loads(Path(a[a.index("--concept") + 1]).read_text()) if "--concept" in a else None
    time = a[a.index("--time") + 1] if "--time" in a else ""
    url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:18100/v1/chat/completions"
    cache = Path(a[a.index("--cache") + 1]) if "--cache" in a else None
    out.with_suffix(".prompt.txt").write_text(prompt(spec, concept, time))
    plan, fl, n = write(spec, concept, time, url, cache)
    if plan is None:
        print("world plan: no answer that is JSON")
        return 1
    if fl:
        plan["_faults"] = fl
    plan["_asks"] = n
    out.write_text(json.dumps(plan, indent=1))
    print(f"world plan: {len(plan.get('parts') or [])} parts after {n} ask(s)" + (f"; {len(fl)} fault(s) left" if fl else ""))
    return 1 if fl else 0


if __name__ == "__main__":
    sys.exit(main())
