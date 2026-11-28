"""A place built as a solid world: a world plan's parts -> the set spec's structure, lights and solids.

A world plan (`world/1`) names the big things of a place by kind, real size and position; every kind here turns
into plain solids (boxes, slabs, poles, balls) at that size. Structure and scale only: the drawing step paints the
surfaces, so a part carries a colour and a material name and no detail. Nothing here is a picture, so the place
looks right from every side and a camera may stand anywhere.

    {"schema": "world/1", "stage_m": [W, D, H], "ground": [...], "parts": [...], "lights": [...]}

Frame as in SETS.md: metres, X east, Y north, Z up, the stage's floor at z = 0 round the origin.

Part kinds (each: "kind", "name", "at": [x, y] or [x, y, z] = the middle of its base, "yaw_deg", and its own keys):
  ship      a whole sailing vessel from "length_m" (beam_m, masts, cabin, sails, mast_m): "at" is the middle of
            its open main deck, the bow toward +x
  row       buildings side by side from [x, y, z] to [x, y, z], fronts toward "front" [x, y]; width_m [lo, hi],
            height_m [lo, hi], depth_m, gap_m, lit, walls [...], roofs [...]
  block     size_m [x, y, z], material
  deck      size_m [x, y], thickness_m, material, rail_m (a rail of that height round it), rail_open [sides]
  building  size_m [w, d, eaves_h], roof "gable" | "flat", pitch_deg, ridge "x" | "y", wall, roof_material,
            windows {"rows", "cols", "lit" 0..1, "sides"} (lit windows glow), door_side
  hull      length_m, beam_m, depth_m (keel to deck), material: a ship's body under a deck, pointed at +x
  mast      height_m, diameter_m, yards [{"z_m", "length_m", "sail": "furled" | "set" | None}], stays [[x, y, z], ...]
  pole      to [x, y, z] (from "at"), diameter_m, material
  lamp      height_m, lumens, kelvin: a post with a glowing head and its light
  lights    from [x, y, z], to [x, y, z], count, sag_m, nits: a string of small lamps
  hill      size_m [x, y, z]: far land, a low stepped mound

`build(plan)` -> {"structure": [...], "lights": [...], "materials": {...}, "solids": [...]}: `solids` are the
boxes people and cameras cannot pass through, [cx, cy, hx, hy, z0, z1, yaw_deg], for the staging facts.
"""
import math

SCHEMA = "world/1"
FAR_TILE_M = 40.0

# a material name a part may give, as a preset of presets.py with the colour the drawing step should read
MATERIALS = {
    "planks": {"preset": "wood_floor", "color": "#8a6238", "grime": 0.3},
    "timber": {"preset": "dark_wood", "color": "#4a3120", "color2": "#5c3d27"},
    "painted_timber": {"preset": "painted_wood", "color": "#7b2f22", "grime": 0.2},
    "tarred_timber": {"preset": "dark_wood", "color": "#241a14", "color2": "#2f231b"},
    "stone": {"preset": "worn_concrete", "color": "#6f6a62", "color2": "#857f75", "panel_m": [0.6, 0.6, 0.35]},
    "plaster": {"preset": "painted_plaster", "color": "#cfc6ae", "grime": 0.15},
    "brick": {"preset": "painted_plaster", "color": "#7c4a38", "color2": "#8a5642", "panel_m": [0.22, 0.22, 0.075],
              "joint_m": 0.012, "joint_darken": 0.25},
    "slate": {"preset": "worn_concrete", "color": "#3a3d42", "color2": "#474a50", "panel_m": [0.3, 0.3, 0.3]},
    "tile": {"preset": "painted_plaster", "color": "#8a4630", "color2": "#9a5238"},
    "thatch": {"preset": "fabric", "color": "#8f7a45", "color2": "#a08a52"},
    "canvas": {"preset": "fabric", "color": "#d8cfb4", "color2": "#e2dac2"},
    "rope": {"preset": "fabric", "color": "#6e5c3e"},
    "iron": {"preset": "bare_steel", "color": "#1d1c1b", "metallic": 0.6, "roughness": [0.45, 0.7]},
    "water": {"preset": "epoxy_floor", "color": "#0c1a26", "color2": "#10222f", "roughness": [0.05, 0.15],
              "panel_m": [0, 0, 0], "grime": 0.0},
    "grass": {"preset": "fabric", "color": "#3e5a2c", "color2": "#4a6a34", "feature_m": 3.0},
    "earth": {"preset": "worn_concrete", "color": "#5a4a38", "color2": "#6a5846", "panel_m": [0, 0, 0]},
    "sand": {"preset": "worn_concrete", "color": "#bfae84", "color2": "#cbbb92", "panel_m": [0, 0, 0]},
    "snow": {"preset": "painted_plaster", "color": "#e8ecef", "grime": 0.0},
    "glow_warm": {"preset": "emissive", "color": "#ffc27a", "nits": 600.0},
    "glow_lamp": {"preset": "emissive", "color": "#ffd59a", "nits": 4000.0},
    "window_dark": {"preset": "painted_wood", "color": "#14161a", "roughness": [0.1, 0.2]},
}


def _rot(x, y, yaw):
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return x * c - y * s, x * s + y * c


class _Out:
    def __init__(self):
        self.structure, self.lights, self.solids, self.used = [], [], [], set()

    def mat(self, m):
        self.used.add(m)
        return m

    def box(self, name, c, size, mat, yaw=0.0, solid=True):
        """A box by the middle of its base, its size in its own frame, turned by yaw."""
        self.structure.append({"kind": "box", "name": name, "base": [round(v, 3) for v in c],
                               "size_m": [round(v, 3) for v in size], "material": self.mat(mat),
                               **({"yaw_deg": round(yaw, 2)} if yaw % 360 else {}),
                               # (far and very large solids in one piece: tiles are for surfaces seen close)
                               **({"tile": False} if max(size[:2]) > FAR_TILE_M else {})})
        if solid and size[2] > 0.3:
            self.solids.append([round(c[0], 3), round(c[1], 3), round(size[0] / 2, 3), round(size[1] / 2, 3),
                                round(c[2], 3), round(c[2] + size[2], 3), round(yaw % 360, 2)])

    def beam(self, name, p0, p1, wh, mat):
        self.structure.append({"kind": "beam", "name": name, "from": [round(v, 3) for v in p0],
                               "to": [round(v, 3) for v in p1], "size_m": list(wh), "material": self.mat(mat)})

    def pole(self, name, p0, p1, d, mat):
        self.structure.append({"kind": "pole", "name": name, "from": [round(v, 3) for v in p0],
                               "to": [round(v, 3) for v in p1], "diameter_m": d, "material": self.mat(mat)})

    def ball(self, name, c, d, mat):
        self.structure.append({"kind": "ball", "name": name, "center": [round(v, 3) for v in c], "diameter_m": d,
                               "material": self.mat(mat)})


def _at(p):
    a = list(p.get("at") or [0, 0])
    return a + [0.0] if len(a) == 2 else a


def _local(p):
    """A function local (x, y, z) -> set frame, for a part at "at" turned by "yaw_deg"."""
    a, yaw = _at(p), float(p.get("yaw_deg") or 0.0)

    def f(x, y, z=0.0):
        rx, ry = _rot(x, y, yaw)
        return [a[0] + rx, a[1] + ry, a[2] + z]
    return f, yaw


def block(o, p):
    f, yaw = _local(p)
    o.box(p["name"], f(0, 0), p["size_m"], p.get("material", "stone"), yaw)


def deck(o, p):
    """A flat floor people stand on, its top at "at"'s height, with a rail round it."""
    f, yaw = _local(p)
    sx, sy = p["size_m"][:2]
    th = p.get("thickness_m", 0.3)
    o.box(p["name"], f(0, 0, -th), [sx, sy, th], p.get("material", "planks"), yaw, solid=False)
    h = p.get("rail_m") or 0.0
    if h:
        t, gone = 0.15, set(p.get("rail_open") or [])
        for side, (x, y, lx, ly) in {"north": (0, sy / 2, sx, t), "south": (0, -sy / 2, sx, t),
                                     "east": (sx / 2, 0, t, sy), "west": (-sx / 2, 0, t, sy)}.items():
            if side not in gone:
                o.box(f"{p['name']}_rail_{side}", f(x, y), [lx, ly, h], p.get("rail_material", "timber"), yaw)


def building(o, p):
    """Four walls as one solid to the eaves, a pitched or flat roof, windows that glow or are dark."""
    f, yaw = _local(p)
    w, d, h = p["size_m"]
    n = p["name"]
    o.box(n, f(0, 0), [w, d, h], p.get("wall", "plaster"), yaw)
    roof = p.get("roof", "gable")
    if roof == "gable":
        pitch = math.radians(p.get("pitch_deg", 42))
        along_x = p.get("ridge", "x" if w >= d else "y") == "x"
        span, run = (d, w) if along_x else (w, d)          # (span: across the ridge; run: along it)
        rise = math.tan(pitch) * span / 2
        over, th = 0.35, 0.18
        slope = math.hypot(span / 2 + over, rise * (span / 2 + over) / (span / 2))
        for sgn in (-1, 1):                                # the two slopes: slabs whose long side is the ridge
            mid_s = sgn * (span / 2 + over) / 2
            mid_z = h + rise - rise * (abs(mid_s) / (span / 2))
            nrm = (0.0, sgn * math.sin(pitch), math.cos(pitch))       # (in the ridge's frame: along, across, up)
            a = (-run / 2 - over, mid_s, mid_z)
            b = (run / 2 + over, mid_s, mid_z)
            if along_x:
                p0, p1 = f(*a), f(*b)
                up = _rot(nrm[0], nrm[1], yaw) + (nrm[2],)
            else:
                p0, p1 = f(a[1], a[0], a[2]), f(b[1], b[0], b[2])
                up = _rot(nrm[1], nrm[0], yaw) + (nrm[2],)
            o.structure.append({"kind": "beam", "name": f"{n}_roof", "from": [round(v, 3) for v in p0],
                                "to": [round(v, 3) for v in p1], "size_m": [round(slope, 3), th],
                                "up": [round(v, 4) for v in up], "material": o.mat(p.get("roof_material", "slate"))})
        steps = max(3, int(rise / 0.5))                    # the gable ends: the wall carried up in steps
        for i in range(steps):
            z0 = h + rise * i / steps
            half = (span / 2) * (1 - (i + 0.5) / steps)
            sz = [run, 2 * half, rise / steps] if along_x else [2 * half, run, rise / steps]
            o.box(f"{n}_gable", f(0, 0, z0), sz, p.get("wall", "plaster"), yaw, solid=False)
        o.solids[-1][5] = round(_at(p)[2] + h + rise, 3)    # (the building's solid: as tall as its ridge)
    else:
        o.box(f"{n}_roof", f(0, 0, h), [w + 0.4, d + 0.4, 0.25], p.get("roof_material", "slate"), yaw, solid=False)
    win = p.get("windows") or {}
    rows, cols = int(win.get("rows", 0)), int(win.get("cols", 0))
    if rows and cols:
        lit = float(win.get("lit", 0.5))
        k = 0
        for side in win.get("sides") or ["south", "north", "east", "west"]:
            length = w if side in ("south", "north") else d
            for r in range(rows):
                for c in range(cols):
                    k += 1
                    u = -length / 2 + length * (c + 0.5) / cols
                    z = h * (r + 0.55) / rows - 0.6
                    glow = ((k * 7919 + len(n) * 104729) % 100) / 100.0 < lit     # (the same windows every build)
                    x, y, sz = {"south": (u, -d / 2, [0.8, 0.08, 1.1]), "north": (u, d / 2, [0.8, 0.08, 1.1]),
                                "east": (w / 2, u, [0.08, 0.8, 1.1]), "west": (-w / 2, u, [0.08, 0.8, 1.1])}[side]
                    o.box(f"{n}_window", f(x, y, max(z, 0.9)), sz, "glow_warm" if glow else "window_dark", yaw,
                          solid=False)
    ds = p.get("door_side")
    if ds:
        x, y, sz = {"south": (0, -d / 2, [1.0, 0.1, 2.0]), "north": (0, d / 2, [1.0, 0.1, 2.0]),
                    "east": (w / 2, 0, [0.1, 1.0, 2.0]), "west": (-w / 2, 0, [0.1, 1.0, 2.0])}[ds]
        o.box(f"{n}_door", f(x, y, 0), sz, "timber", yaw, solid=False)


def hull(o, p):
    """A ship's body, its deck line at "at"'s height: a straight middle and a bow that narrows, pointed at +x."""
    f, yaw = _local(p)
    L, B, Dp = p["length_m"], p["beam_m"], p.get("depth_m", 4.0)
    m = p.get("material", "tarred_timber")
    mid = L * 0.7
    o.box(p["name"], f(-(L - mid) / 2, 0, -Dp), [mid, B, Dp - 0.05], m, yaw, solid=False)
    bow, n_ = L - mid, 5
    for i in range(n_):                                     # (the bow in slices, each narrower)
        x0 = -L / 2 + mid + bow * i / n_
        wd = B * (1 - ((i + 0.5) / n_) ** 1.6)
        o.box(f"{p['name']}_bow", f(x0 + bow / n_ / 2, 0, -Dp), [bow / n_, max(wd, 0.4), Dp - 0.05], m, yaw,
              solid=False)


def mast(o, p):
    f, yaw = _local(p)
    H, d = p["height_m"], p.get("diameter_m", 0.5)
    n = p["name"]
    o.pole(n, f(0, 0, 0), f(0, 0, H), d, p.get("material", "timber"))
    a = _at(p)
    o.solids.append([round(a[0], 3), round(a[1], 3), d / 2, d / 2, round(a[2], 3), round(a[2] + H, 3), 0.0])
    for y_ in p.get("yards") or []:
        z, ln = y_["z_m"], y_["length_m"]
        o.pole(f"{n}_yard", f(0, -ln / 2, z), f(0, ln / 2, z), 0.22, "timber")
        if y_.get("sail") == "furled":
            o.pole(f"{n}_sail", f(0, -ln * 0.46, z - 0.3), f(0, ln * 0.46, z - 0.3), 0.5, "canvas")
        elif y_.get("sail") == "set":
            dz = y_.get("drop_m", 5.0)
            o.beam(f"{n}_sail", f(0.25, 0, z - 0.3), f(0.25, 0, z - 0.3 - dz), [ln * 0.92, 0.06], "canvas")
    for s in p.get("stays") or []:
        o.pole(f"{n}_stay", f(0, 0, H * 0.92), list(s), 0.05, "rope")


def pole(o, p):
    o.pole(p["name"], _at(p), p["to"], p.get("diameter_m", 0.15), p.get("material", "timber"))


def lamp(o, p):
    f, _ = _local(p)
    H = p.get("height_m", 3.0)
    o.pole(p["name"], f(0, 0, 0), f(0, 0, H), 0.1, "iron")
    o.ball(f"{p['name']}_head", f(0, 0, H + 0.15), 0.3, "glow_lamp")
    o.lights.append({"type": "point", "position": [round(v, 3) for v in f(0, 0, H + 0.15)],
                     "lumens": p.get("lumens", 1200), "kelvin": p.get("kelvin", 2400), "radius_m": p.get("radius_m", 14),
                     "source_radius_m": 0.12, "shadows": bool(p.get("shadows", True))})


def lights(o, p):
    """A string of small lamps hanging between two points; every `light_every`-th one also lights the place."""
    a, b = p["from"], p["to"]
    n, sag = int(p.get("count", 12)), p.get("sag_m", 0.6)
    o.used.add("glow_lamp")
    for i in range(n):
        t = (i + 0.5) / n
        c = [a[j] + (b[j] - a[j]) * t for j in range(3)]
        c[2] -= sag * 4 * t * (1 - t)
        o.ball(p["name"], c, p.get("bulb_m", 0.12), "glow_lamp")
        if p.get("light_every") and i % int(p["light_every"]) == int(p["light_every"]) // 2:
            o.lights.append({"type": "point", "position": [round(v, 3) for v in c], "lumens": p.get("lumens", 300),
                             "kelvin": 2500, "radius_m": 9, "source_radius_m": 0.06, "shadows": False})


def hill(o, p):
    f, yaw = _local(p)
    sx, sy, sz = p["size_m"]
    for i in range(4):                                      # (far land: a stepped mound reads as a slope)
        k = 1 - i * 0.22
        o.box(p["name"], f(0, 0, sz * i / 4), [sx * k, sy * k, sz / 4], p.get("material", "grass"), yaw, solid=False)


def _sub(p, f, yaw, **kw):
    """A part of a whole thing, given in the whole's own frame."""
    q = dict(kw)
    q["at"] = f(*q["at"])
    q["yaw_deg"] = yaw + q.get("yaw_deg", 0.0)
    for k in ("to", "from"):
        if k in q:
            q[k] = f(*q[k])
    if "stays" in q:
        q["stays"] = [f(*s) for s in q["stays"]]
    q["name"] = f"{p['name']}_{q['name']}"
    return q


def ship(o, p):
    """A whole sailing vessel from its length: hull, an open main deck with rails, a raised stern over a cabin, a
    foredeck, masts with yards and stays, a bowsprit. "at" is the middle of the OPEN main deck (its top), the bow
    toward +x before "yaw_deg"; the masts stand clear of that middle."""
    f, yaw = _local(p)
    L = float(p["length_m"])
    B = float(p.get("beam_m") or max(2.0, L / 4.2))
    n = int(p.get("masts") or (1 if L < 12 else 2 if L < 24 else 3))
    sail = p.get("sails", "furled")
    x_stern, x_bow = -0.42 * L, 0.58 * L                    # (the open deck's middle sits aft of the hull's)
    cab = 0.16 * L if p.get("cabin", L >= 14) else 0.0
    fore = 0.14 * L
    hull(o, _sub(p, f, yaw, name="hull", at=[(x_stern + x_bow) / 2, 0, 0], length_m=L, beam_m=B,
                 depth_m=p.get("depth_m") or max(1.2, L * 0.13), material=p.get("material", "tarred_timber")))
    x0, x1 = x_stern + cab, x_bow - fore - 0.12 * L
    deck(o, _sub(p, f, yaw, name="deck", at=[(x0 + x1) / 2, 0, 0], size_m=[x1 - x0, B], rail_m=1.0 if L >= 8 else 0.5,
                 rail_open=["east", "west"]))
    if cab:
        ch = 2.4
        block(o, _sub(p, f, yaw, name="cabin", at=[x_stern + cab / 2, 0, 0], size_m=[cab, B, ch], material="timber"))
        deck(o, _sub(p, f, yaw, name="stern_deck", at=[x_stern + cab / 2, 0, ch + 0.3], size_m=[cab, B], rail_m=1.0))
        block(o, _sub(p, f, yaw, name="cabin_door", at=[x0 + 0.05, 0, 0], size_m=[0.1, 0.9, 1.9],
                      material="painted_timber"))
        for sy in (-1, 1):
            block(o, _sub(p, f, yaw, name="cabin_window", at=[x0 + 0.04, sy * B * 0.28, 1.0], size_m=[0.08, B * 0.16, 0.8],
                          material="glow_warm" if p.get("lit", True) else "window_dark"))
        block(o, _sub(p, f, yaw, name="stern_steps", at=[x0 + 0.8, B / 2 - 0.9, 0], size_m=[1.6, 1.0, ch / 2],
                      material="timber"))
    deck(o, _sub(p, f, yaw, name="foredeck", at=[x1 + fore / 2, 0, 0.0], size_m=[fore, B * 0.75], rail_m=0.9,
                 rail_open=["west"]))
    H = float(p.get("mast_m") or L * 0.75)
    spots = {1: [0.22], 2: [0.30, -0.02 if not cab else None], 3: [0.20, 0.42, None]}[min(n, 3)]
    for i, fx in enumerate(spots):
        on_stern = fx is None
        mx = x_stern + cab * 0.6 if on_stern else fx * L
        mz = (2.7 if cab else 0.0) if on_stern else 0.0
        mh = H * (1.0 if i == 0 else 0.85 if not on_stern else 0.62)
        yards = [{"z_m": round(mh * k, 2), "length_m": round(B * w, 2), "sail": sail}
                 for k, w in ((0.35, 2.0), (0.62, 1.5), (0.85, 1.0))][:3 if mh > 12 else 2 if mh > 6 else 1]
        mast(o, _sub(p, f, yaw, name=f"mast_{i + 1}", at=[mx, 0, mz], height_m=mh,
                     diameter_m=round(max(0.15, mh / 45), 2), yards=yards,
                     stays=[[mx - 0.12 * L, B / 2, mz + 1.0], [mx - 0.12 * L, -B / 2, mz + 1.0],
                            [mx + 0.06 * L, B / 2, mz + 1.0], [mx + 0.06 * L, -B / 2, mz + 1.0]]))
    if L >= 10:
        pole(o, _sub(p, f, yaw, name="bowsprit", at=[x_bow - 0.06 * L, 0, 0.5], to=[x_bow + 0.2 * L, 0, 0.5 + 0.09 * L],
                     diameter_m=round(max(0.15, L / 90), 2)))


def row(o, p):
    """Buildings side by side along a line, their fronts toward "front" (a point they look at): a street, a
    waterfront, a far town's edge. Sizes vary inside the ranges, the same at every build."""
    a, b = p["from"], p["to"]
    a, b = (list(a) + [0.0])[:3], (list(b) + [0.0])[:3]
    ln = math.hypot(b[0] - a[0], b[1] - a[1])
    if ln < 1.0:
        return
    ux, uy = (b[0] - a[0]) / ln, (b[1] - a[1]) / ln
    yaw = math.degrees(math.atan2(uy, ux))
    fr = p.get("front") or [0.0, 0.0]
    side = "south" if (-uy) * (fr[0] - a[0]) + ux * (fr[1] - a[1]) < 0 else "north"   # (which long wall faces it)
    w_lo, w_hi = p.get("width_m") or [6.0, 11.0]
    h_lo, h_hi = p.get("height_m") or [5.5, 8.5]
    d = float(p.get("depth_m") or 9.0)
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(p["name"])) or 1

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) % 2147483648
        return seed / 2147483648.0
    walls, roofs = p.get("walls") or ["plaster", "brick", "stone"], p.get("roofs") or ["slate", "tile"]
    t, i = 0.0, 0
    while t < ln - w_lo * 0.6 and i < 60:
        w = min(w_lo + (w_hi - w_lo) * rnd(), ln - t)
        h = h_lo + (h_hi - h_lo) * rnd()
        m = t + w / 2
        off = d / 2 * (1 if side == "south" else -1)       # (the front wall on the line, the house behind it)
        cx, cy = a[0] + ux * m - uy * off, a[1] + uy * m + ux * off
        building(o, {"kind": "building", "name": f"{p['name']}_{i}", "at": [cx, cy, a[2] + (b[2] - a[2]) * m / ln],
                     "yaw_deg": yaw, "size_m": [round(w, 2), d, round(h, 2)], "roof": "gable",
                     "ridge": "x" if rnd() < 0.6 else "y", "pitch_deg": 38 + 14 * rnd(),
                     "wall": walls[int(rnd() * len(walls)) % len(walls)],
                     "roof_material": roofs[int(rnd() * len(roofs)) % len(roofs)],
                     "windows": {"rows": max(1, int(h // 2.8)), "cols": max(2, int(w // 2.5)),
                                 "lit": p.get("lit", 0.4), "sides": [side]}, "door_side": side})
        t += w + (float(p.get("gap_m") or 0.0) if rnd() < 0.7 else 2.5)
        i += 1


KINDS = {"ship": ship, "row": row, "block": block, "deck": deck, "building": building, "hull": hull, "mast": mast, "pole": pole, "lamp": lamp,
         "lights": lights, "hill": hill}


def build(plan: dict) -> dict:
    if plan.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} plan")
    o = _Out()
    for g in plan.get("ground") or []:      # the far ground and water, out to the horizon
        o.structure.append({"kind": "ground", "name": g.get("name", "ground"), "material": o.mat(g["material"]),
                            "extent_m": g.get("extent_m", 6000), "z_m": g.get("z_m", -0.1),
                            **({"center": g["center"]} if g.get("center") else {})})
    for p in plan.get("parts") or []:
        if p.get("kind") not in KINDS:
            raise ValueError(f"unknown part kind {p.get('kind')!r}")
        KINDS[p["kind"]](o, p)
    mats = {m: dict(MATERIALS[m]) for m in sorted(o.used)}
    for m, over in (plan.get("materials") or {}).items():   # (a plan's own colour for a material of the list)
        mats.setdefault(m, dict(MATERIALS.get(over.get("like", m), {})))
        mats[m].update({k: v for k, v in over.items() if k != "like"})
    return {"structure": o.structure, "lights": o.lights + list(plan.get("lights") or []), "materials": mats,
            "solids": o.solids}


def for_place(world: dict, place: str) -> dict | None:
    """A story's world as one of its places sees it: the same parts in that place's frame (the place's origin at
    [0, 0, 0], turned by its "yaw_deg"), with the place's stage and spots. None for a place the world does not hold.

    A story world is a world/1 plan with "places": {name: {"origin": [x, y, z], "yaw_deg", "stage_m": [W, D, H],
    "spots": {piece id: {"at": [x, y], "facing": [dx, dy]}}}}: every place of the story stands in the one world."""
    pl = (world.get("places") or {}).get(place)
    if not pl:
        return None
    ox, oy, oz = (list(pl.get("origin") or [0, 0, 0]) + [0.0])[:3]
    yaw = float(pl.get("yaw_deg") or 0.0)

    def pt(v):
        v = list(v)
        x, y = _rot(v[0] - ox, v[1] - oy, -yaw)
        return [round(x, 4), round(y, 4)] + ([round(v[2] - oz, 4)] if len(v) > 2 else [])
    parts = []
    for q in world.get("parts") or []:
        q = dict(q)
        for k in ("at", "to", "from", "front"):
            if isinstance(q.get(k), list):
                q[k] = pt(q[k] if k == "front" or len(q[k]) > 2 else list(q[k]) + [0.0])
        if isinstance(q.get("stays"), list):
            q["stays"] = [pt(v) for v in q["stays"]]
        if yaw and q.get("kind") not in ("pole", "lights"):
            q["yaw_deg"] = float(q.get("yaw_deg") or 0.0) - yaw
        parts.append(q)
    ground = [dict(g, z_m=round(float(g.get("z_m", -0.1)) - oz, 4)) for g in world.get("ground") or []]
    return {"schema": SCHEMA, "stage_m": list(pl["stage_m"]), "ground": ground, "parts": parts,
            "spots": dict(pl.get("spots") or {}), "night": world.get("night"),
            **({"materials": world["materials"]} if world.get("materials") else {})}

