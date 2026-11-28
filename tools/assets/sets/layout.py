"""Set spec -> explicit scene (primitives, lights, assets, cameras in UE units). Stdlib only; see SETS.md.

Set frame: metres, X east, Y north, Z up, origin at the centre of the floor at floor level (Z = 0 is the floor's
top). UE frame: centimetres, x = 100 X, y = -100 Y, z = 100 Z (the same mirror the asset chain uses for glTF and
for spec cameras). Library assets face -Y in the set frame at facing [0, -1]; `facing` turns that front to a
direction in plan.

Every structural element becomes boxes (the engine's 1 m cube scaled), cylinders or spheres; big boxes are split
into tiles of at most TILE_M so Lumen's surface cache keeps its resolution. Output of `expand(spec, assets)`:
  {"prims": [{"m": cube|cylinder|sphere, "c": [x,y,z] cm, "s": [sx,sy,sz], "x": [..], "z": [..] (axes, optional),
              "yaw": deg (optional), "mat": name, "f": outliner folder}],
   "assets": [{"id", "path", "c", "yaw", "f"}], "lights": [...], "env": {...}, "cameras": [...], "figures": [...]}
"""

import math

TILE_M = 8.0
SIDES = ("north", "south", "east", "west")


# ------------------------------------------------------------------ vector helpers (set frame, metres)
def add(a, b):
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def mul(a, k):
    return [a[0] * k, a[1] * k, a[2] * k]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def norm(a):
    n = math.sqrt(dot(a, a))
    return [a[0] / n, a[1] / n, a[2] / n]


def ue_pt(p):
    return [round(p[0] * 100, 2), round(-p[1] * 100, 2), round(p[2] * 100, 2)]


def ue_dir(d):
    return [round(d[0], 6), round(-d[1], 6), round(d[2], 6)]


def yaw_from_facing(f):
    """Set-frame facing [dx, dy] -> UE yaw (deg) for an asset whose front is -Y (set) at yaw 0."""
    set_yaw = math.degrees(math.atan2(f[1], f[0])) + 90.0      # rotation (CCW, set frame) taking -Y to f
    return -set_yaw                                            # the mirror flips the sense of rotation


class Scene:
    def __init__(self):
        self.prims, self.lights, self.assets = [], [], []
        self.stats = {}

    def _count(self, folder):
        k = folder.split("/")[0]
        self.stats[k] = self.stats.get(k, 0) + 1

    # box from its centre and size (size in the box's own frame, turned by yaw degrees CCW), tiled for Lumen
    def box(self, c, size, mat, folder, yaw=0.0, tile=True):
        n = [max(1, math.ceil(size[i] / TILE_M - 1e-6)) if tile and i < 2 else 1 for i in range(3)]
        if size[2] > TILE_M * 1.5 and tile:
            n[2] = max(1, math.ceil(size[2] / TILE_M - 1e-6))
        step = [size[i] / n[i] for i in range(3)]
        ca, sa = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    lx, ly = -size[0] / 2 + step[0] * (i + 0.5), -size[1] / 2 + step[1] * (j + 0.5)
                    cc = [c[0] + lx * ca - ly * sa, c[1] + lx * sa + ly * ca, c[2] - size[2] / 2 + step[2] * (k + 0.5)]
                    p = {"m": "cube", "c": ue_pt(cc), "s": [round(x, 4) for x in step], "mat": mat, "f": folder}
                    if yaw % 180:
                        p["yaw"] = round(-yaw, 4)
                    self.prims.append(p)
                    self._count(folder)

    def box_from_to(self, lo, hi, mat, folder):
        self.box([(lo[i] + hi[i]) / 2 for i in range(3)], [abs(hi[i] - lo[i]) for i in range(3)], mat, folder)

    # a straight member from p0 to p1: width w across, height h along `up` (projected)
    def member(self, p0, p1, w, h, mat, folder, up=(0, 0, 1)):
        d = sub(p1, p0)
        L = math.sqrt(dot(d, d))
        if L < 1e-4:
            return
        x = mul(d, 1 / L)
        up = list(up)
        if abs(dot(x, up)) > 0.98:
            up = [1, 0, 0] if abs(x[0]) < 0.9 else [0, 1, 0]
        z = norm(sub(up, mul(x, dot(up, x))))
        self.prims.append({"m": "cube", "c": ue_pt(mul(add(p0, p1), 0.5)), "s": [round(L, 4), w, h],
                           "x": ue_dir(x), "z": ue_dir(z), "mat": mat, "f": folder})
        self._count(folder)

    def ibeam(self, p0, p1, depth, width, mat, folder, up=(0, 0, 1), tw=None, tf=None):
        tw = tw or max(0.012, depth * 0.06)
        tf = tf or max(0.016, depth * 0.08)
        d = norm(sub(p1, p0))
        up = list(up)
        if abs(dot(d, up)) > 0.98:
            up = [1, 0, 0] if abs(d[0]) < 0.9 else [0, 1, 0]
        z = norm(sub(up, mul(d, dot(up, d))))
        off = mul(z, depth / 2 - tf / 2)
        self.member(p0, p1, tw, depth - 2 * tf, mat, folder, z)
        self.member(add(p0, off), add(p1, off), width, tf, mat, folder, z)
        self.member(sub(p0, off), sub(p1, off), width, tf, mat, folder, z)

    def cyl(self, p0, p1, d, mat, folder):
        v = sub(p1, p0)
        L = math.sqrt(dot(v, v))
        if L < 1e-4:
            return
        z = mul(v, 1 / L)
        x = norm(cross(z, [0, 0, 1])) if abs(z[2]) < 0.98 else [1, 0, 0]
        self.prims.append({"m": "cylinder", "c": ue_pt(mul(add(p0, p1), 0.5)), "s": [d, d, round(L, 4)],
                           "x": ue_dir(x), "z": ue_dir(z), "mat": mat, "f": folder})
        self._count(folder)

    def sphere(self, c, d, mat, folder):
        self.prims.append({"m": "sphere", "c": ue_pt(c), "s": [d, d, d], "mat": mat, "f": folder})
        self._count(folder)


# ------------------------------------------------------------------ walls
class Wall:
    """Maps wall-local coordinates (u along the wall, z up, o outward from the inner face) to the set frame.
    Room walls: u is the absolute X (north/south) or Y (east/west) coordinate. Partitions: u runs from 0 at
    `from` to the length at `to`, and "outward" is the right-hand side walking from `from` to `to`."""

    def __init__(self, side, W, D, t, a=None, b=None):
        self.side, self.t = side, t
        if side == "partition":
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            self.org, self.dir = [a[0], a[1]], [(b[0] - a[0]) / L, (b[1] - a[1]) / L]
            self.out = [self.dir[1], -self.dir[0]]
            self.u0, self.u1 = 0.0, L
        else:
            self.org, self.dir, self.out = {
                "north": ([0, D / 2], [1, 0], [0, 1]), "south": ([0, -D / 2], [1, 0], [0, -1]),
                "east": ([W / 2, 0], [0, 1], [1, 0]), "west": ([-W / 2, 0], [0, 1], [-1, 0])}[side]
            self.u0, self.u1 = ((-W / 2 - t, W / 2 + t) if side in ("north", "south") else (-D / 2, D / 2))
        self.yaw = math.degrees(math.atan2(self.dir[1], self.dir[0]))
        # local box frame: x along the wall, y = outward. With yaw = angle of dir, local +y = left of dir; flip
        # the outward sign so sizes [du, do, dz] stay symmetric (boxes are symmetric, only centres matter).

    def p(self, u, z, o=0.0):
        return [self.org[0] + self.dir[0] * u + self.out[0] * o, self.org[1] + self.dir[1] * u + self.out[1] * o, z]

    def size(self, du, dz, do):
        return [du, do, dz]

    def inward(self):
        return [-self.out[0], -self.out[1], 0]


def wall_block(S, w, u0, u1, z0, z1, mat, folder, o0=None, o1=None):
    """Solid block of the wall between u0..u1 and z0..z1 (through its thickness unless o0/o1 given)."""
    if u1 - u0 < 1e-3 or z1 - z0 < 1e-3:
        return
    o0 = 0.0 if o0 is None else o0
    o1 = w.t if o1 is None else o1
    S.box(w.p((u0 + u1) / 2, (z0 + z1) / 2, (o0 + o1) / 2), w.size(u1 - u0, z1 - z0, o1 - o0), mat, folder, yaw=w.yaw)


def build_wall(S, spec, wd, W, D, H):
    side = wd.get("side", "partition")
    t = wd.get("thickness_m", 0.3)
    w = Wall(side, W, D, t, wd.get("from"), wd.get("to"))
    mat = wd["material"]
    dado = wd.get("dado")                 # lower band in another material
    folder = f"Walls/{wd.get('name', side)}"
    ops = sorted(wd.get("openings", []), key=lambda o: o["from_m"])
    top = wd.get("height_m", H)

    def solid(u0, u1, z0, z1):
        if dado and z0 < dado["height_m"] < z1:
            wall_block(S, w, u0, u1, z0, dado["height_m"], dado["material"], folder)
            wall_block(S, w, u0, u1, dado["height_m"], z1, mat, folder)
        elif dado and z1 <= dado["height_m"]:
            wall_block(S, w, u0, u1, z0, z1, dado["material"], folder)
        else:
            wall_block(S, w, u0, u1, z0, z1, mat, folder)

    u = w.u0
    for o in ops:
        solid(u, o["from_m"], 0, top)
        solid(o["from_m"], o["to_m"], 0, o.get("sill_m", 0.0))
        solid(o["from_m"], o["to_m"], o["head_m"], top)
        u = o["to_m"]
        kind = o["kind"]
        if kind == "window_band":
            window_band(S, w, o, f"{folder}/windows")
        elif kind == "door":
            door(S, w, o, f"{folder}/door")
        elif kind == "glass_wall":
            window_band(S, w, o, f"{folder}/glass")
    solid(u, w.u1, 0, top)
    for b in wd.get("columns", []):       # pilasters on the inner face at listed u positions
        for uc in b["at_m"]:
            p0, p1 = w.p(uc, 0, -b["depth_m"] / 2), w.p(uc, b.get("height_m", top), -b["depth_m"] / 2)
            S.ibeam(p0, p1, b["depth_m"], b["width_m"], b["material"], f"{folder}/columns", up=w.inward())
    for pl in wd.get("plates", []):         # raised panels (relief) on the inner face, skipping openings
        pw, ph = pl["size_m"]
        gap, dep = pl.get("gap_m", 0.03), pl.get("depth_m", 0.025)
        za, zb = pl["z_m"]
        ua = pl.get("from_m", w.u0 + (t if side in ("north", "south") else 0))
        ub = pl.get("to_m", w.u1 - (t if side in ("north", "south") else 0))
        nu, nz = max(1, int((ub - ua) // pw)), max(1, int((zb - za) // ph))
        cw, chh = (ub - ua) / nu, (zb - za) / nz
        for i in range(nu):
            for k in range(nz):
                u0_, u1_ = ua + i * cw + gap / 2, ua + (i + 1) * cw - gap / 2
                z0_, z1_ = za + k * chh + gap / 2, za + (k + 1) * chh - gap / 2
                if any(o["from_m"] < u1_ and u0_ < o["to_m"] and o.get("sill_m", 0) < z1_ and z0_ < o["head_m"]
                       for o in ops):
                    continue
                mat_ = pl["materials"][(i * 7 + k * 3) % len(pl["materials"])] if "materials" in pl else pl["material"]
                wall_block(S, w, u0_, u1_, z0_, z1_, mat_, f"{folder}/plates", -dep, 0.0)
    for f in wd.get("trim", []):           # horizontal ledges / skirting on the inner face
        a, b2 = f.get("from_m", w.u0 + (t if side in ("north", "south") else 0)), \
            f.get("to_m", w.u1 - (t if side in ("north", "south") else 0))
        wall_block(S, w, a, b2, f["z_m"], f["z_m"] + f["height_m"], f["material"], f"{folder}/trim",
                   -f["depth_m"], 0.0)


def window_band(S, w, o, folder):
    u0, u1, z0, z1 = o["from_m"], o["to_m"], o.get("sill_m", 0.0), o["head_m"]
    t = w.t
    bay = o.get("bay_m", 3.0)
    nb = max(1, round((u1 - u0) / bay))
    bw = (u1 - u0) / nb
    mw, md = o.get("mullion_m", [0.2, t + 0.1])
    fmat, gmat = o["frame_material"], o.get("glass_material")
    cols, rows = o.get("panes", [2, 4])
    bar = o.get("bar_m", 0.05)
    go = t * o.get("glass_at", 0.5)
    # glass, one pane per bay
    for i in range(nb if gmat else 0):          # (no glass material: an open, unglazed window)
        a = u0 + i * bw
        S.box(w.p(a + bw / 2, (z0 + z1) / 2, go), w.size(bw, z1 - z0, 0.02), gmat, folder, tile=False, yaw=w.yaw)
    # mullions (full depth, slightly proud of the inner face)
    for i in range(nb + 1):
        uc = u0 + i * bw
        uc = min(max(uc, u0 + mw / 2), u1 - mw / 2)
        S.box(w.p(uc, (z0 + z1) / 2, t - md / 2), w.size(mw, z1 - z0, md), fmat, folder, tile=False, yaw=w.yaw)
    # head and sill
    S.box(w.p((u0 + u1) / 2, z1 - mw / 2, t - md / 2), w.size(u1 - u0, mw, md), fmat, folder, yaw=w.yaw)
    S.box(w.p((u0 + u1) / 2, z0 + mw / 4, t / 2 - 0.05), w.size(u1 - u0, mw / 2, t + 0.1), fmat, folder, yaw=w.yaw)
    # transoms (whole band) and glazing bars (per bay), on both sides of the glass
    for r in range(1, rows):
        zr = z0 + (z1 - z0) * r / rows
        S.box(w.p((u0 + u1) / 2, zr, go), w.size(u1 - u0, bar * 1.6, 0.12), fmat, folder, yaw=w.yaw)
    for i in range(nb):
        for c in range(1, cols):
            uc = u0 + i * bw + bw * c / cols
            S.box(w.p(uc, (z0 + z1) / 2, go), w.size(bar, z1 - z0, 0.08), fmat, folder, tile=False, yaw=w.yaw)
    # cross bracing inside the listed bays (pairs of diagonals on the inner face)
    bm = o.get("brace_material", fmat)
    bz0, bz1 = o.get("brace_z_m", [z0, z1])
    for i in o.get("brace_bays", []):
        if i >= nb:
            continue
        a, b = u0 + i * bw + mw / 2, u0 + (i + 1) * bw - mw / 2
        oi = -0.08
        S.member(w.p(a, bz0, oi), w.p(b, bz1, oi), 0.08, 0.14, bm, folder, up=w.inward())
        S.member(w.p(b, bz0, oi - 0.1), w.p(a, bz1, oi - 0.1), 0.08, 0.14, bm, folder, up=w.inward())


def door(S, w, o, folder):
    """Large sliding door filling the opening: a plate in the wall plane, a grid of frame members and an X brace
    per cell on the inner face, a header track above."""
    u0, u1, z1 = o["from_m"], o["to_m"], o["head_m"]
    z0 = o.get("sill_m", 0.0)
    t = w.t
    pw, ph = o.get("panel_m", [3.0, 3.0])
    nu, nz = max(1, round((u1 - u0) / pw)), max(1, round((z1 - z0) / ph))
    cw, ch = (u1 - u0) / nu, (z1 - z0) / nz
    fm = o.get("frame_material", o["material"])
    bmat = o.get("brace_material", fm)
    fw = o.get("frame_m", 0.25)
    S.box(w.p((u0 + u1) / 2, (z0 + z1) / 2, t * 0.5), w.size(u1 - u0, z1 - z0, 0.12), o["material"], folder, yaw=w.yaw)
    of = t * 0.5 - 0.06 - fw / 2       # members sit on the plate's inner side
    for i in range(nu + 1):
        uc = min(max(u0 + i * cw, u0 + fw / 2), u1 - fw / 2)
        S.box(w.p(uc, (z0 + z1) / 2, of), w.size(fw, z1 - z0, fw), fm, folder, tile=False, yaw=w.yaw)
    for k in range(nz + 1):
        zc = min(max(z0 + k * ch, z0 + fw / 2), z1 - fw / 2)
        S.box(w.p((u0 + u1) / 2, zc, of), w.size(u1 - u0, fw, fw), fm, folder, yaw=w.yaw)
    if o.get("braced", True):
        bo = of - 0.02
        for i in range(nu):
            for k in range(nz):
                a, b = u0 + i * cw + fw / 2, u0 + (i + 1) * cw - fw / 2
                c, d = z0 + k * ch + fw / 2, z0 + (k + 1) * ch - fw / 2
                S.member(w.p(a, c, bo), w.p(b, d, bo), 0.1, 0.1, bmat, folder, up=w.inward())
                S.member(w.p(b, c, bo - 0.04), w.p(a, d, bo - 0.04), 0.1, 0.1, bmat, folder, up=w.inward())
    hd = o.get("header_m", [0.6, 0.5])
    S.box(w.p((u0 + u1) / 2, z1 + hd[0] / 2, -hd[1] / 2), w.size(u1 - u0 + 1.0, hd[0], hd[1]), fm, folder, yaw=w.yaw)


# ------------------------------------------------------------------ roof and ceilings
def build_trusses(S, td, W, D, H):
    ax = td.get("span_axis", "x")
    span = W if ax == "x" else D
    run = D if ax == "x" else W
    sp = td["spacing_m"]
    n = max(1, round(run / sp))
    zb, zt = td["bottom_m"], td["top_m"]
    rise = td.get("rise_m", 0.0)
    cd, cw = td.get("chord_m", [0.35, 0.25])
    web = td.get("web_m", 0.12)
    panel = td.get("panel_m", 2.5)
    mat = td["material"]
    npnl = max(2, round(span / panel))
    if npnl % 2:
        npnl += 1

    def P(s, r, z):        # s along the span (centred), r along the run
        return [s, r, z] if ax == "x" else [r, s, z]

    def ztop(s):
        return zt + rise * (1 - abs(s) / (span / 2))

    stations = [-span / 2 + span * i / npnl for i in range(npnl + 1)]
    for k in range(n):
        r = -run / 2 + run * (k + 0.5) / n
        f = f"Roof/truss_{k:02d}"
        if td.get("members", True):          # false: trusses placed as library assets; keep columns, purlins, deck
            S.ibeam(P(-span / 2, r, zb), P(span / 2, r, zb), cd, cw, mat, f)
            mid = 0.0
            S.ibeam(P(-span / 2, r, ztop(-span / 2)), P(mid, r, ztop(mid)), cd, cw, mat, f)
            S.ibeam(P(mid, r, ztop(mid)), P(span / 2, r, ztop(span / 2)), cd, cw, mat, f)
            for i, s in enumerate(stations):
                S.member(P(s, r, zb + cd / 2), P(s, r, ztop(s) - cd / 2), web, web, mat, f,
                         up=[1, 0, 0] if ax == "y" else [0, 1, 0])
                if i < npnl:
                    s2 = stations[i + 1]
                    if s2 <= 1e-6:        # Pratt: diagonals fall toward the centre
                        a, b = P(s, r, ztop(s) - cd / 2), P(s2, r, zb + cd / 2)
                    else:
                        a, b = P(s, r, zb + cd / 2), P(s2, r, ztop(s2) - cd / 2)
                    S.member(a, b, web, web, mat, f, up=[0, 0, 1])
        if td.get("columns", True):
            cdep = td.get("column_m", [0.5, 0.35])
            for e in (-1, 1):
                s = e * (span / 2 - cdep[0] / 2)
                S.ibeam(P(s, r, 0), P(s, r, ztop(s)), cdep[0], cdep[1], mat, "Roof/columns",
                        up=[1, 0, 0] if ax == "x" else [0, 1, 0])
    pu = td.get("purlins")
    if pu:
        npu = max(2, round(span / pu["spacing_m"]))
        for i in range(npu + 1):
            s = -span / 2 + span * i / npu
            z = ztop(s) + cd / 2 + pu["size_m"][0] / 2
            S.member(P(s, -run / 2, z), P(s, run / 2, z), pu["size_m"][1], pu["size_m"][0], mat, "Roof/purlins")
    bt = td.get("bottom_ties")
    if bt:
        for s in bt["at_m"]:
            S.member(P(s, -run / 2, zb), P(s, run / 2, zb), 0.15, 0.15, mat, "Roof/ties")
    rb = td.get("roof_bracing")
    if rb:
        for k in range(n - 1):
            if k % rb.get("every", 2):
                continue
            r0 = -run / 2 + run * (k + 0.5) / n
            r1 = -run / 2 + run * (k + 1.5) / n
            for e in (-1, 1):
                s0, s1 = 0.0, e * span / 2
                z0, z1 = ztop(s0) + cd / 2, ztop(s1) + cd / 2
                S.member(P(s0, r0, z0), P(s1, r1, z1), 0.08, 0.08, mat, "Roof/bracing")
                S.member(P(s0, r1, z0), P(s1, r0, z1), 0.08, 0.08, mat, "Roof/bracing")
    deck = td.get("deck")
    if deck:
        z = zt + rise + cd / 2 + (pu["size_m"][0] if pu else 0)
        th = deck.get("thickness_m", 0.2)
        # flat deck at the ridge height covering the whole plan (pitched decks: later)
        S.box([0, 0, z + th / 2], [W + 1.0, D + 1.0, th], deck["material"], "Roof/deck")


def build_grid_ceiling(S, g, W, D):
    z = g["height_m"]
    cx, cy = g["cell_m"]
    bw, bd = g.get("bar_m", [0.05, 0.08])
    x0, x1 = g.get("x_m", [-W / 2, W / 2])
    y0, y1 = g.get("y_m", [-D / 2, D / 2])
    S.box([(x0 + x1) / 2, (y0 + y1) / 2, z + bd + 0.03], [x1 - x0, y1 - y0, 0.06], g["panel_material"], "Ceiling/panels")
    nx, ny = round((x1 - x0) / cx), round((y1 - y0) / cy)
    for i in range(nx + 1):
        x = x0 + (x1 - x0) * i / nx
        S.box([x, (y0 + y1) / 2, z + bd / 2], [bw, y1 - y0, bd], g["bar_material"], "Ceiling/grid")
    for j in range(ny + 1):
        y = y0 + (y1 - y0) * j / ny
        S.box([(x0 + x1) / 2, y, z + bd / 2], [x1 - x0, bw, bd], g["bar_material"], "Ceiling/grid")
    above = g.get("plenum")
    if above:      # closed plenum box above the grid so nothing reads as open sky through gaps
        th = 0.2
        S.box([(x0 + x1) / 2, (y0 + y1) / 2, above["top_m"] + th / 2], [x1 - x0 + 1, y1 - y0 + 1, th],
              above["material"], "Ceiling/plenum")


# ------------------------------------------------------------------ platforms, runs, lines
def build_platform(S, p):
    (xa, ya), (xb, yb) = p["from"], p["to"]
    x0, x1, y0, y1 = min(xa, xb), max(xa, xb), min(ya, yb), max(ya, yb)
    z = p["height_m"]
    th = p.get("thickness_m", 0.3)
    f = f"Platforms/{p.get('name', 'platform')}"
    S.box([(x0 + x1) / 2, (y0 + y1) / 2, z - th / 2], [x1 - x0, y1 - y0, th], p["material"], f)
    # edge beams
    em = p.get("edge_material", p["material"])
    for (a, b) in (([x0, y0], [x1, y0]), ([x0, y1], [x1, y1]), ([x0, y0], [x0, y1]), ([x1, y0], [x1, y1])):
        S.member([a[0], a[1], z - th - 0.15], [b[0], b[1], z - th - 0.15], 0.15, 0.3, em, f)
    sup = p.get("supports")
    if sup:
        sp = sup["spacing_m"]
        along_x = (x1 - x0) >= (y1 - y0)
        L = (x1 - x0) if along_x else (y1 - y0)
        n = max(1, round(L / sp))
        lines = sup.get("lines", ["outer"])
        for i in range(n + 1):
            s = (x0 if along_x else y0) + L * i / n
            for ln in lines:
                if along_x:
                    yy = {"outer": y0 + 0.15 if sup.get("outer_side") == "south" else y1 - 0.15,
                          "south": y0 + 0.15, "north": y1 - 0.15}[ln]
                    pt = [min(max(s, x0 + 0.15), x1 - 0.15), yy]
                else:
                    xx = {"outer": x0 + 0.15 if sup.get("outer_side") == "west" else x1 - 0.15,
                          "west": x0 + 0.15, "east": x1 - 0.15}[ln]
                    pt = [xx, min(max(s, y0 + 0.15), y1 - 0.15)]
                S.ibeam([pt[0], pt[1], 0], [pt[0], pt[1], z - th], sup.get("size_m", 0.3), sup.get("size_m", 0.3) * 0.7,
                        sup["material"], f, up=[1, 0, 0])
    rl = p.get("railing")
    if rl:
        h = rl.get("height_m", 1.1)
        rm = rl["material"]
        ps = rl.get("post_spacing_m", 1.5)
        edges = {"south": ([x0, y0], [x1, y0]), "north": ([x0, y1], [x1, y1]),
                 "west": ([x0, y0], [x0, y1]), "east": ([x1, y0], [x1, y1])}
        for side in rl["sides"]:
            a, b = edges[side]
            L = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, round(L / ps))
            for i in range(n + 1):
                t = i / n
                x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                S.box([x, y, z + h / 2], [0.05, 0.05, h], rm, f, tile=False)
            for zz, sz in ((z + h, 0.06), (z + h * 0.5, 0.04)):
                S.member([a[0], a[1], zz], [b[0], b[1], zz], sz, sz, rm, f)
            S.member([a[0], a[1], z + 0.075], [b[0], b[1], z + 0.075], 0.02, 0.15, rm, f)


def build_run(S, r, kind):
    pts = r["points"]
    f = f"Services/{r.get('name', kind)}"
    if kind == "pipe_run":
        d = r["diameter_m"]
        for a, b in zip(pts, pts[1:]):
            S.cyl(a, b, d, r["material"], f)
        for p in pts[1:-1]:
            S.sphere(p, d * 1.02, r["material"], f)
        dr = r.get("drops")            # vertical branches from the run down (or up) to to_z_m at listed points
        if dr:
            for (x, y) in dr["at"]:
                z = pts[0][2]
                S.cyl([x, y, z], [x, y, dr["to_z_m"]], dr.get("diameter_m", d * 0.6), r["material"], f)
                S.box([x, y, dr["to_z_m"] + 0.02], [dr.get("diameter_m", d * 0.6) + 0.06] * 2 + [0.04],
                      r["material"], f, tile=False)
        hang = r.get("hangers")
        if hang:
            for a, b in zip(pts, pts[1:]):
                L = math.dist(a, b)
                n = int(L // hang["spacing_m"])
                for i in range(1, n + 1):
                    t = i * hang["spacing_m"] / L
                    p = [a[k] + (b[k] - a[k]) * t for k in range(3)]
                    S.box([p[0], p[1], (p[2] + hang["to_z_m"]) / 2], [0.03, 0.03, abs(hang["to_z_m"] - p[2])],
                          hang.get("material", r["material"]), f, tile=False)
                    S.member([p[0], p[1], p[2] - d / 2 - 0.02], [p[0], p[1], p[2] - d / 2 - 0.02 + 1e-3],
                             d + 0.08, 0.04, hang.get("material", r["material"]), f)
    else:
        w, h = r["size_m"]
        for a, b in zip(pts, pts[1:]):
            S.member(a, b, w, h, r["material"], f)
        for p in pts[1:-1]:
            S.box(p, [w, w, h], r["material"], f, tile=False)
        for a, b in zip(pts, pts[1:]):          # seams every 1.5 m: slightly bigger collars
            L = math.dist(a, b)
            n = int(L // 1.5)
            for i in range(1, n + 1):
                t = i * 1.5 / L
                p = [a[k] + (b[k] - a[k]) * t for k in range(3)]
                e = [p[k] + (b[k] - a[k]) / L * 0.04 for k in range(3)]
                S.member(p, e, w + 0.04, h + 0.04, r.get("seam_material", r["material"]), f)


def build_lines(S, ld):
    for i, ln in enumerate(ld["lines"]):
        pts = ln["points"]
        w = ln["width_m"]
        z = ln.get("z_m", 0.0)
        th = ln.get("thickness_m", 0.006)
        f = "Floor/lines"
        segs = list(zip(pts, pts[1:]))
        if ln.get("closed"):
            segs.append((pts[-1], pts[0]))
        for a, b in segs:
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            ex = w / 2 / d if d else 0
            a2 = [a[0] - (b[0] - a[0]) * ex, a[1] - (b[1] - a[1]) * ex, z + th / 2]
            b2 = [b[0] + (b[0] - a[0]) * ex, b[1] + (b[1] - a[1]) * ex, z + th / 2]
            S.member(a2, b2, w, th, ln["material"], f)


# ------------------------------------------------------------------ lights
def strip_lights(S, sl, idx):
    L = sl.get("length_m", 1.5)
    z = sl["z_m"]
    hm, tm = sl["fixture_material"], sl["tube_material"]
    pts = []
    for row in sl["rows"]:
        (xa, ya), (xb, yb) = row["from"], row["to"]
        n = row["count"]
        for i in range(n):
            t = 0.5 if n == 1 else i / (n - 1)
            pts.append(([xa + (xb - xa) * t, ya + (yb - ya) * t], row.get("along", sl.get("along", "y"))))
    f = f"Lights/strips_{idx}"
    for (x, y), along in pts:
        if along == "x":
            hs, ts = [L, 0.3, 0.1], [L - 0.06, 0.16, 0.03]
        else:
            hs, ts = [0.3, L, 0.1], [0.16, L - 0.06, 0.03]
        S.box([x, y, z + 0.05], hs, hm, f, tile=False)
        S.box([x, y, z - 0.015], ts, tm, f, tile=False)
        hang = sl.get("hang_to_m")
        if hang:
            for e in (-1, 1):
                o = [e * L * 0.4, 0] if along == "x" else [0, e * L * 0.4]
                S.box([x + o[0], y + o[1], (z + 0.1 + hang) / 2], [0.015, 0.015, hang - z - 0.1], hm, f, tile=False)
        S.lights.append({"type": "rect", "c": ue_pt([x, y, z - 0.035]), "dir": [0, 0, -1],
                         "w_cm": (L - 0.06) * 100 if along == "x" else 16, "h_cm": 16 if along == "x" else (L - 0.06) * 100,
                         "lumens": sl["lumens"], "kelvin": sl.get("kelvin", 5000), "radius_m": sl.get("radius_m", 30),
                         "shadows": sl.get("shadows", True), "barn_door_deg": sl.get("barn_door_deg", 88),
                         "f": f, **({"color": sl["color"]} if "color" in sl else {})})


def light_items(S, lights):
    env = {}
    for i, l in enumerate(lights):
        t = l["type"]
        if t == "strip_lights":
            strip_lights(S, l, i)
        elif t in ("sun", "sky", "fog"):
            env[t] = dict(l)
            if t == "sun":
                env[t]["toward_sun_ue"] = ue_dir(norm(l["toward_sun"]))
        elif t in ("point", "spot", "rect"):
            d = dict(l)
            d["c"] = ue_pt(l["position"])
            if "toward" in l:
                d["dir"] = ue_dir(norm(sub(l["toward"], l["position"])))
            if t == "rect":
                d["w_cm"], d["h_cm"] = l["size_m"][0] * 100, l["size_m"][1] * 100
            d["f"] = "Lights/practicals"
            S.lights.append(d)
        else:
            raise ValueError(f"unknown light type {t}")
    return env


# ------------------------------------------------------------------ assets
def asset_items(S, spec, assets, key):
    out = []
    for it in spec.get(key, []):
        if "row" in it:
            r = it["row"]
            (xa, ya), (xb, yb) = r["from"], r["to"]
            n = r["count"]
            for i in range(n):
                t = 0.5 if n == 1 else i / (n - 1)
                out.append({"asset": it["asset"], "position": [xa + (xb - xa) * t, ya + (yb - ya) * t, r.get("z_m", 0.0)],
                            "facing": it.get("facing", [0, -1]), "name": f"{it.get('name', it['asset'])}_{i:02d}"})
        else:
            p = list(it["position"]) + ([0.0] if len(it["position"]) == 2 else [])
            if "on" in it:                      # standing on another placed item's surface at on_z_m
                p[2] = it["on_z_m"]
            out.append({"asset": it["asset"], "position": p, "facing": it.get("facing", [0, -1]),
                        "name": it.get("name", it["asset"]), "on": it.get("on"), "hang": bool(it.get("hang")),
                        "outside": bool(it.get("outside")), "world": bool(it.get("world"))})
    for o in out:
        a = assets[o["asset"]]
        yaw = yaw_from_facing(o["facing"])
        if a.get("front"):                      # a mesh whose real front is not -Y: turn that front onto the facing
            yaw += math.degrees(math.atan2(a["front"][1], a["front"][0])) + 90.0
        S.assets.append({"id": o["asset"], "name": o["name"], "path": a["ue_path"], "c": ue_pt(o["position"]),
                         "yaw": round(yaw, 3), "f": f"{key.capitalize()}",
                         **({"glb": a["glb"]} if a.get("glb") else {}),
                         # a place's world (its land and the sky dome round it) casts no shadow: it enclosed the
                         # stage, and the per-shot key from the camera's side never reached a face (every talk of an
                         # outdoor place was a black silhouette against the sky, 11-17)
                         **({"no_shadow": True} if o.get("world") else {})})
    return out


def footprint(item, assets):
    """Plan rectangle (x0, y0, x1, y1) and z range of a placed asset, rotated by its facing (set frame)."""
    b = assets[item["asset"]]["bounds_cm"]
    hx, hy = (b["max"][0] - b["min"][0]) / 200, (b["max"][1] - b["min"][1]) / 200
    a = math.radians(math.degrees(math.atan2(item["facing"][1], item["facing"][0])) + 90.0)
    ex = abs(hx * math.cos(a)) + abs(hy * math.sin(a))
    ey = abs(hx * math.sin(a)) + abs(hy * math.cos(a))
    x, y, z = item["position"]
    return (x - ex, y - ey, x + ex, y + ey), (z, z + b["max"][2] / 100)


# ------------------------------------------------------------------ top level
def expand(spec, assets):
    W, D, H = spec["size_m"]
    S = Scene()
    for el in spec["structure"]:
        k = el["kind"]
        if k == "floor":
            th = el.get("thickness_m", 0.4)
            ext = el.get("extend_m", 1.0)
            S.box([0, 0, -th / 2], [W + 2 * ext, D + 2 * ext, th], el["material"], "Floor/slab")
        elif k in ("wall", "partition"):
            build_wall(S, spec, el, W, D, H)
        elif k == "trusses":
            build_trusses(S, el, W, D, H)
        elif k == "ceiling":
            th = el.get("thickness_m", 0.3)
            S.box([0, 0, el["height_m"] + th / 2], [W + 1.0, D + 1.0, th], el["material"], "Ceiling/slab")
        elif k == "grid_ceiling":
            build_grid_ceiling(S, el, W, D)
        elif k == "platform":
            build_platform(S, el)
        elif k in ("pipe_run", "duct_run"):
            build_run(S, el, k)
        elif k == "floor_lines":
            build_lines(S, el)
        elif k == "box":
            b = el["base"]
            sz = el["size_m"]
            S.box([b[0], b[1], b[2] + sz[2] / 2], sz, el["material"], f"Boxes/{el.get('name', 'box')}",
                  yaw=el.get("yaw_deg", 0.0), tile=el.get("tile", True))
        elif k == "ground":
            # (a world's far ground or water: its top at z_m, round `center`, out to the horizon)
            ext = el.get("extent_m", 400)
            cx, cy = el.get("center", [0, 0])
            S.box([cx, cy, el.get("z_m", -0.1) - 0.5], [ext, ext, 1.0], el["material"],
                  f"Outside/{el.get('name', 'ground')}", tile=False)
        elif k == "beam":           # a straight slab between two points (a roof's slope, a gangway, a spar)
            S.member(el["from"], el["to"], el["size_m"][0], el["size_m"][1], el["material"],
                     f"Solids/{el.get('name', 'beam')}", up=tuple(el.get("up") or (0, 0, 1)))
        elif k == "pole":           # a round bar between two points (a mast, a post, a rope)
            S.cyl(el["from"], el["to"], el["diameter_m"], el["material"], f"Solids/{el.get('name', 'pole')}")
        elif k == "ball":
            S.sphere(el["center"], el["diameter_m"], el["material"], f"Solids/{el.get('name', 'ball')}")
        else:
            raise ValueError(f"unknown structure kind {k}")
    env = light_items(S, spec.get("lights", []))
    env["exposure_ev100"] = spec.get("exposure_ev100", 8.0)
    env["post"] = spec.get("post", {})
    placed = asset_items(S, spec, assets, "fixtures") + asset_items(S, spec, assets, "props")
    cams = [{"name": c["name"], "c": ue_pt(c["position"]), "t": ue_pt(c["target"]), "focal": c.get("focal_mm", 24)}
            for c in spec.get("cameras", [])]
    figs = [{"c": ue_pt(list(f["position"]) + [0.0] if len(f["position"]) == 2 else f["position"]),
             "yaw": round(yaw_from_facing(f.get("facing", [0, -1])), 3)}       # the mannequin mesh faces UE +Y too
            for f in spec.get("validation", {}).get("figures", [])]
    return {"prims": S.prims, "assets": S.assets, "lights": S.lights, "env": env, "cameras": cams, "figures": figs,
            "counts": S.stats, "placed": placed}


def hung_ok(spec, fp, tol=0.05):
    """A hung item's plan rectangle touches a wall's inner face or a side of a structure box (a chimney breast)."""
    W, D, _ = spec["size_m"]
    x0, y0, x1, y1 = fp
    if x0 < -W / 2 + tol or x1 > W / 2 - tol or y0 < -D / 2 + tol or y1 > D / 2 - tol:
        return True
    for el in spec.get("structure", []):
        if el.get("kind") != "box":
            continue
        b, s = el["base"], el["size_m"]
        bx0, bx1, by0, by1 = b[0] - s[0] / 2, b[0] + s[0] / 2, b[1] - s[1] / 2, b[1] + s[1] / 2
        near_x = min(abs(x1 - bx0), abs(x0 - bx1)) < tol and y1 > by0 and y0 < by1
        near_y = min(abs(y1 - by0), abs(y0 - by1)) < tol and x1 > bx0 and x0 < bx1
        if near_x or near_y:
            return True
    return False


def check(spec, assets, placed):
    """Plan checks a writer (person or model) can act on: in bounds, no overlaps, nothing floating."""
    W, D, H = spec["size_m"]
    probs = []
    fps = [(p, *footprint(p, assets)) for p in placed]
    for p, (x0, y0, x1, y1), (z0, z1) in fps:
        if p.get("world"):                      # a place in the open: its land holds the whole stage
            continue
        if p.get("outside"):                    # beyond the walls on purpose (a backdrop seen through a window)
            if not (x1 < -W / 2 or x0 > W / 2 or y1 < -D / 2 or y0 > D / 2):
                probs.append(f"{p['name']}: marked outside, but it reaches into the room")
            continue
        if x0 < -W / 2 - 1e-3 or x1 > W / 2 + 1e-3 or y0 < -D / 2 - 1e-3 or y1 > D / 2 + 1e-3:
            probs.append(f"{p['name']}: outside the room ({x0:.2f},{y0:.2f})-({x1:.2f},{y1:.2f})")
        if z1 > H:
            probs.append(f"{p['name']}: taller than the room")
        if p.get("on"):
            hosts = [(q, r) for q, r, _ in fps if q["name"] == p["on"]]
            if not hosts:
                probs.append(f"{p['name']}: 'on' names {p['on']!r}, which is not placed")
            else:
                r = hosts[0][1]
                if not (r[0] - 1e-3 <= x0 and x1 <= r[2] + 1e-3 and r[1] - 1e-3 <= y0 and y1 <= r[3] + 1e-3):
                    probs.append(f"{p['name']}: overhangs {p['on']} (not fully on its surface)")
        elif p.get("hang"):                     # hung on a wall: its back must touch a wall face or a structure box
            if not hung_ok(spec, (x0, y0, x1, y1)):
                probs.append(f"{p['name']}: hung, but its back touches no wall (within 5 cm)")
        elif z0 > 0.005:
            sup =[q for q, (a0, b0, a1, b1), (c0, c1) in fps if q is not p and abs(c1 - z0) < 0.03
                   and a0 <= p["position"][0] <= a1 and b0 <= p["position"][1] <= b1]
            ok_surf = any(abs(s.get("z_m", -1) - z0) < 0.03 for s in spec.get("support_surfaces", []))
            if not sup and not ok_surf:
                probs.append(f"{p['name']}: floats at z={z0:.2f} m (nothing under it within 3 cm)")
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            (p, a, za), (q, b, zb) = fps[i], fps[j]
            if p.get("on") == q["name"] or q.get("on") == p["name"] or p.get("outside") or q.get("outside"):
                continue
            ox = min(a[2], b[2]) - max(a[0], b[0])
            oy = min(a[3], b[3]) - max(a[1], b[1])
            oz = min(za[1], zb[1]) - max(za[0], zb[0])
            if ox > 0.01 and oy > 0.01 and oz > 0.01:
                probs.append(f"{p['name']} overlaps {q['name']} by {ox:.2f} x {oy:.2f} m")
    return probs
