"""Shared helpers for the smithy pieces (model scripts under tools/assets/models/<id>/): coopered vessels, forged
bars, and the smithy's procedural materials. Builds on period_kit (imported, never edited).

    import os, sys
    sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import forge_kit as F            # F.K is period_kit

coopered(prefix, h, r_fn, staves, thick, ...)  staves of a tub or barrel (radius profile r_fn(z)), with a bottom
hoops(prefix, r_fn, zs, ...)                   iron hoops seated on the staves
water(name, r, z)                               a still water surface disc
bar(name, path, w, h, ...)                      a forged square/flat bar along a centreline (section w x h, twist)
forged_iron(...), dark_water(), smithy_wood(...) materials
"""

import math
import random

import bmesh
import bpy
from mathutils import Vector, noise

import period_kit as K

TAU = 2 * math.pi


def coopered(prefix, h, r_fn, staves=22, thick=0.022, gap=0.0012, nz=16, na=3, top_jitter=0.006, bottom_z=0.04,
             seed=1, inner=True):
    """Staves of an open coopered vessel from z = 0 to ~h (outer radius r_fn(z)); each stave is its own closed slab
    (so the joints read), with slightly uneven tops. Returns (staves list, bottom disc)."""
    rng = random.Random(seed)
    out = []
    da = TAU / staves
    for s in range(staves):
        a0 = s * da + gap / max(r_fn(0.0), 0.05)
        a1 = (s + 1) * da - gap / max(r_fn(0.0), 0.05)
        top = h - rng.uniform(0, top_jitter)
        w_off = rng.uniform(-0.002, 0.002)          # staves sit a hair in or out
        bm = bmesh.new()
        grid = []
        for side in (0, 1):                          # 0 outer, 1 inner
            rows = []
            for k in range(nz + 1):
                z = top * k / nz
                r = r_fn(z) + w_off - (thick if side else 0.0)
                rows.append([bm.verts.new((r * math.cos(K.lerp(a0, a1, j / na)), r * math.sin(K.lerp(a0, a1, j / na)), z))
                             for j in range(na + 1)])
            grid.append(rows)
        o, i = grid
        for k in range(nz):
            for j in range(na):
                bm.faces.new((o[k][j], o[k][j + 1], o[k + 1][j + 1], o[k + 1][j]))
                bm.faces.new((i[k][j], i[k + 1][j], i[k + 1][j + 1], i[k][j + 1]))
        for j in range(na):
            bm.faces.new((o[nz][j], o[nz][j + 1], i[nz][j + 1], i[nz][j]))
            bm.faces.new((o[0][j], i[0][j], i[0][j + 1], o[0][j + 1]))
        for k in range(nz):
            bm.faces.new((o[k][0], o[k + 1][0], i[k + 1][0], i[k][0]))
            bm.faces.new((o[k][na], i[k][na], i[k + 1][na], o[k + 1][na]))
        ob = K.from_bm(f"{prefix}stave{s}", bm)
        K.shade(ob, 40)
        out.append(ob)
    rb = r_fn(bottom_z) - thick + 0.004
    bottom = K.lathe(f"{prefix}bottom", [(0.0, bottom_z - 0.01), (rb, bottom_z - 0.01), (rb, bottom_z + 0.012),
                                         (0.0, bottom_z + 0.012)], seg=48)
    return out, bottom


def hoops(prefix, r_fn, zs, width=0.035, thick=0.004, seg=96):
    out = []
    for n, z in enumerate(zs):
        z0, z1 = z - width / 2, z + width / 2
        r0, r1 = r_fn(z0) - 0.001, r_fn(z1) - 0.001
        out.append(K.lathe(f"{prefix}hoop{n}", [(r0 - 0.002, z0), (r0 + thick, z0), (K.lerp(r0, r1, 0.5) + thick * 1.2, z),
                                               (r1 + thick, z1), (r1 - 0.002, z1), (r0 - 0.002, z0)],
                           seg=seg, cap0=False, cap1=False))
    return out


def water(name, r, z, seg=64, rings=10, ripple=0.0015, seed=2):
    """A still water surface: a disc of radius r at height z with faint ripples, closed underneath."""
    bm = bmesh.new()
    off = Vector((seed * 3.1, seed * 1.7, 0))
    centre = bm.verts.new((0, 0, z))
    prev = [centre]
    rings_v = []
    for j in range(1, rings + 1):
        rr = r * j / rings
        ring = []
        for i in range(seg):
            a = TAU * i / seg
            p = Vector((rr * math.cos(a), rr * math.sin(a), 0))
            dz = ripple * noise.noise(p * 12 + off) if j < rings else 0.0
            ring.append(bm.verts.new((p.x, p.y, z + dz)))
        rings_v.append(ring)
    for i in range(seg):
        bm.faces.new((centre, rings_v[0][i], rings_v[0][(i + 1) % seg]))
    for a, b in zip(rings_v, rings_v[1:]):
        for i in range(seg):
            bm.faces.new((a[i], b[i], b[(i + 1) % seg], a[(i + 1) % seg]))
    under = [bm.verts.new((v.co.x, v.co.y, z - 0.01)) for v in rings_v[-1]]
    for i in range(seg):
        bm.faces.new((rings_v[-1][i], under[i], under[(i + 1) % seg], rings_v[-1][(i + 1) % seg]))
    bm.faces.new(list(reversed(under)))
    ob = K.from_bm(name, bm)
    K.shade(ob, 80)
    return ob


def bar(name, path, w, h=None, twist=0.0, n=None, rnd=0.25, seg=16, w_fn=None):
    """A forged bar along a centreline: rounded-square section w x h (rnd 0 sharp .. 1 round), twisted `twist` turns
    over its length; w_fn(t) -> scale of the section along it (tapers)."""
    h = h or w
    path = [Vector(p) for p in path]
    secs = []
    nexp = K.lerp(10.0, 2.0, rnd)
    for i in range(len(path)):
        t = i / max(1, len(path) - 1)
        s = w_fn(t) if w_fn else 1.0
        secs.append(K.superellipse2d(w / 2 * s, h / 2 * s, nexp, twist * TAU * t, seg))
    return K.sweep(name, path, secs)


# ------------------------------------------------------------------ materials
def forged_iron(name="forged_iron", scale_amt=0.7, rust=0.35, polish=0.35, soot=0.0, hot=None):
    """Black hand-forged iron: hammer facets, blue-black scale flaking to grey metal, rust in pits, worn bright where
    handled. hot: (axis, lo, hi) object coordinate range glowing (lo hot end) for work fresh from the fire."""
    k = K.NK(name)
    big = k.noise(k.obj, 6.0, 3.0, 0.5)
    mid = k.noise(k.obj, 40.0, 4.0, 0.55)
    fine = k.noise(k.obj, 350.0, 3.0, 0.6)
    ham = k.voronoi(k.obj, 55.0, "F1", "Distance")       # hammer-blow facets
    e = k.edge(0.0015)
    cv = k.cavity(0.012, 0.9, 0.45)
    scale_m = k.m("MULTIPLY", k.mr(mid, 0.35, 0.6), scale_amt, clamp=True)
    col = k.mix(scale_m, (0.075, 0.074, 0.074), (0.020, 0.021, 0.025))
    col = k.mix(k.mr(ham, 0.0, 0.5, 0.25, 0.0), col, (0.11, 0.11, 0.115))
    pol = k.m("MULTIPLY", e, k.mr(mid, 0.3, 0.6, 0.2, 1.0))
    pol = k.m("MULTIPLY", pol, polish, clamp=True)
    col = k.mix(pol, col, (0.30, 0.295, 0.29))
    rm = k.m("MULTIPLY", k.m("ADD", k.m("MULTIPLY", cv, 0.8), k.mr(big, 0.6, 0.78, 0, 0.6)), rust, clamp=True)
    col = k.mix(rm, col, k.mix(fine, (0.10, 0.04, 0.015, 1), (0.22, 0.09, 0.03, 1)))
    rough = k.mr(mid, 0.3, 0.7, 0.45, 0.72)
    rough = k.mixf(pol, rough, 0.28)
    rough = k.mixf(rm, rough, 0.9)
    met = k.mixf(scale_m, 0.8, 0.45)
    met = k.mixf(pol, met, 1.0)
    met = k.mixf(rm, met, 0.0)
    if soot:
        col = k.mix(k.m("MULTIPLY", k.mr(big, 0.3, 0.6, 0.5, 1.0), soot, clamp=True), col, (0.01, 0.009, 0.008))
    h = k.m("ADD", k.m("MULTIPLY", fine, 0.3), k.m("MULTIPLY", ham, 0.6))
    nrm = k.bump(h, 0.0004, 0.4)
    if hot:
        ax, lo, hi = hot
        s = k.N.new("ShaderNodeSeparateXYZ")
        k.L.new(k.obj, s.inputs[0])
        heat = k.mr(s.outputs[ax], hi, lo)
        ecol = k.ramp(heat, [(0.0, (0, 0, 0)), (0.25, (0.5, 0.03, 0.0)), (0.6, (1.0, 0.25, 0.02)),
                             (1.0, (1.0, 0.75, 0.35))])
        return k.out(col, rough, met, nrm, ecol, 10.0)
    return k.out(col, rough, met, nrm)


def dark_water(name="dark_water", murk=(0.012, 0.014, 0.013), scum=0.3):
    """Opaque still water in a vessel: near-black, glossy, a little floating scale and ash at the edges."""
    k = K.NK(name)
    mid = k.noise(k.obj, 20.0, 3.0, 0.5)
    fine = k.noise(k.obj, 120.0, 3.0, 0.5)
    s = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.obj, s.inputs[0])
    d = k.m("SQRT", k.m("ADD", k.m("MULTIPLY", s.outputs["X"], s.outputs["X"]),
                        k.m("MULTIPLY", s.outputs["Y"], s.outputs["Y"])))
    edge = k.mr(d, 0.20, 0.33)
    film = k.m("MULTIPLY", k.m("MULTIPLY", edge, k.mr(mid, 0.45, 0.7)), scum, clamp=True)
    col = k.mix(film, murk, (0.08, 0.07, 0.06))
    col = k.mix(k.m("MULTIPLY", k.mr(fine, 0.6, 0.75), film), col, (0.14, 0.09, 0.05))
    rough = k.mixf(film, 0.03, 0.5)
    return k.out(col, rough, 0.0, k.bump(fine, 0.0002, 0.1))


def smithy_wood(name="smithy_wood", axis="Z", base=(0.16, 0.10, 0.055), grey=0.4, wet=0.0, soot=0.2, wear=0.4,
                burns=0.0):
    """Rough, unvarnished workshop timber: open grain, weathered grey, dirt and soot, worn pale where handled;
    wet: dark wet band toward z = 0 (object); burns: black scorch blotches (bench tops, stumps)."""
    k = K.NK(name)
    rot = {"Z": (0, 0, 0), "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}[axis]
    v = k.vec(k.obj, rot=rot)
    streak = k.noise(v, 1.0, 5.0, 0.6, stretch=(70.0, 70.0, 2.0))
    fine = k.noise(v, 1.0, 3.0, 0.6, stretch=(600.0, 600.0, 18.0))
    blot = k.noise(k.obj, 5.0, 3.0, 0.5)
    mid = k.noise(k.obj, 30.0, 4.0, 0.55)
    dark = tuple(c * 0.45 for c in base)
    light = tuple(min(1, c * 1.5) for c in base)
    col = k.ramp(k.m("ADD", k.m("MULTIPLY", streak, 0.8), k.m("MULTIPLY", blot, 0.4)),
                 [(0.25, dark), (0.6, base), (0.9, light)])
    col = k.mix(k.m("MULTIPLY", k.mr(fine, 0.55, 0.75), 0.6), col, tuple(c * 0.35 for c in base))
    col = k.mix(k.m("MULTIPLY", k.mr(blot, 0.3, 0.7, 0.3, 1.0), grey), col, (0.16, 0.15, 0.14))
    e = k.edge(0.004)
    cv = k.cavity(0.03)
    col = k.mix(k.m("MULTIPLY", e, k.mr(mid, 0.3, 0.6, 0.0, wear)), col, tuple(min(1, c * 1.9 + 0.03) for c in base))
    col = k.mix(k.m("MULTIPLY", cv, 0.75), col, (0.02, 0.016, 0.012))
    if soot:
        col = k.mix(k.m("MULTIPLY", k.mr(mid, 0.4, 0.7, 0.2, 1.0), soot, clamp=True), col, (0.025, 0.022, 0.02))
    rough = k.mr(mid, 0.3, 0.7, 0.72, 0.92)
    if wet:
        s = k.N.new("ShaderNodeSeparateXYZ")
        k.L.new(k.obj, s.inputs[0])
        wm = k.m("MULTIPLY", k.mr(s.outputs["Z"], wet, 0.0), k.mr(mid, 0.2, 0.6, 0.5, 1.0), clamp=True)
        col = k.mix(wm, col, k.mul_col(col, (0.45, 0.42, 0.4)))
        rough = k.mixf(wm, rough, 0.45)
    if burns:
        bn = k.noise(k.obj, 7.0, 4.0, 0.6)
        bm_ = k.m("MULTIPLY", k.mr(bn, 0.6, 0.72), burns, clamp=True)
        col = k.mix(bm_, col, (0.012, 0.009, 0.007))
        rough = k.mixf(bm_, rough, 0.95)
    h = k.m("ADD", k.m("MULTIPLY", fine, -0.6), k.m("MULTIPLY", streak, 0.4))
    return k.out(col, rough, 0.0, k.bump(h, 0.0006, 0.5))


# ------------------------------------------------------------------ rubble masonry
class Rubble:
    """Rough fieldstone masonry: every stone a rounded, noise-displaced block with a per-stone colour tint (corner
    attribute "pcol"), laid in rough courses on axis-aligned faces over a recessed mortar core (the joints are real
    relief). rng: a random.Random; joint: mortar gap; amp: face roughness."""

    TINTS = [(0.115, 0.110, 0.100), (0.140, 0.130, 0.115), (0.095, 0.095, 0.090), (0.125, 0.110, 0.090),
             (0.165, 0.155, 0.140), (0.105, 0.095, 0.080)]

    def __init__(self, rng, joint=0.018, amp=0.012, spacing=0.11):
        self.rng, self.joint, self.amp, self.spacing = rng, joint, amp, spacing
        self.stones = []

    def stone(self, cx, cy, cz, sx, sy, sz, amp=None):
        rng = self.rng
        amp = self.amp if amp is None else amp
        r = min(0.03, sx * 0.3, sy * 0.3, sz * 0.3)
        ob = K.rbox(f"st{len(self.stones)}", (sx, sy, sz), r=r, spacing=self.spacing,
                    bulge=(rng.uniform(0.002, 0.01), rng.uniform(0.002, 0.01), rng.uniform(0.002, 0.008)))
        inner = [sx / 2 - r, sy / 2 - r, sz / 2 - r]
        off = Vector((rng.uniform(0, 50), rng.uniform(0, 50), rng.uniform(0, 50)))
        tilt = (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
        for v in ob.data.vertices:
            p = v.co.copy()
            q = Vector([max(-inner[i], min(inner[i], p[i])) for i in range(3)])
            d = p - q
            n = d.normalized() if d.length > 1e-9 else Vector((0, 0, 1))
            w = Vector((cx, cy, cz)) + p
            disp = amp * (noise.noise(w * 8.0 + off) * 0.75 + noise.noise(w * 21.0 + off) * 0.35)
            disp += amp * 0.6 * abs(noise.noise(w * 3.5 + off))
            p += n * disp
            p.z += tilt[0] * p.x + tilt[1] * p.y
            v.co = p
        ob.data.update()
        ob.location = (cx, cy, cz)
        tint = rng.choice(self.TINTS)
        j = rng.uniform(0.85, 1.12)
        K.set_color(ob, tuple(c * j for c in tint))
        self.stones.append(ob)
        return ob

    def lay(self, axis, sign, plane, u0, u1, courses, depth, skip=(), lmin=0.22, lmax=0.5, out=0.0, ragged=0.0):
        """Coursed stones on an axis-aligned face ('y': normal along Y, u = X; 'x': normal along X, u = Y). sign: the
        outward direction; plane: the face's outer coordinate; courses: z boundaries; skip: (u0, u1, z0, z1) holes;
        ragged: up to this much off a stone's height (rubble, not ashlar)."""
        rng, J = self.rng, self.joint
        for ci, (z0, z1) in enumerate(zip(courses, courses[1:])):
            u = u0 + (rng.uniform(0, lmin) if ci % 2 else 0.0)
            segs = [(u0, u)] if u > u0 else []
            while u < u1 - 1e-6:
                L = rng.uniform(lmin, lmax)
                if u1 - (u + L) < lmin * 0.6:
                    L = u1 - u
                segs.append((u, min(u1, u + L)))
                u += L
            for a, b in segs:
                pieces = [(a, b)]
                for (s0, s1, sz0, sz1) in skip:
                    if z1 > sz0 + 1e-6 and z0 < sz1 - 1e-6:
                        nxt = []
                        for (pa, pb) in pieces:
                            if pb <= s0 or pa >= s1:
                                nxt.append((pa, pb))
                            else:
                                if pa < s0:
                                    nxt.append((pa, s0))
                                if pb > s1:
                                    nxt.append((s1, pb))
                        pieces = nxt
                for (pa, pb) in pieces:
                    if pb - pa < 0.06:
                        continue
                    L = pb - pa - J
                    cut = rng.uniform(0, ragged)
                    H = z1 - z0 - J - cut
                    proud = out + rng.uniform(-0.004, 0.012)
                    dd = depth * rng.uniform(0.85, 1.0)
                    c_n = plane - sign * dd / 2 + sign * proud
                    cu = (pa + pb) / 2
                    cz = (z0 + z1) / 2 + rng.uniform(-0.5, 0.5) * cut
                    if axis == "y":
                        self.stone(cu, c_n, cz, L, dd, H)
                    else:
                        self.stone(c_n, cu, cz, dd, L, H)

    @staticmethod
    def courses(z0, z1, rng, lo=0.16, hi=0.28):
        cs = [z0]
        while cs[-1] < z1 - 1e-6:
            z = cs[-1] + rng.uniform(lo, hi)
            cs.append(z if z1 - z > lo * 0.6 else z1)
        cs[-1] = z1
        return cs


def stone_material(name, kind="stone", soot_z=None, soot=0.5, grime_base=0.55):
    """Fieldstone ('stone', tinted by pcol) or lime mortar ('mortar'): mottling, flecks, iron-stain patches, worn
    edges, dark crevices, ash dust on ledges, soot rising with height (soot_z = (z0, z1) world heights, fading in),
    floor grime at the foot."""
    k = K.NK(name)
    pos = k.geo.outputs["Position"]
    s = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(pos, s.inputs[0])
    big = k.noise(pos, 3.0, 4.0, 0.55)
    mid = k.noise(pos, 14.0, 5.0, 0.6)
    fine = k.noise(pos, 90.0, 4.0, 0.6)
    vor = k.voronoi(pos, 38.0, "F1", "Distance")
    if kind == "stone":
        base = k.pcol()
        col = k.mix(k.mr(mid, 0.3, 0.7, 0.0, 0.45), base, k.mul_col(base, (0.62, 0.60, 0.58)))
        col = k.mix(k.mr(fine, 0.55, 0.75, 0.0, 0.35), col, k.mul_col(base, (1.3, 1.28, 1.22)))
        col = k.mix(k.mr(vor, 0.08, 0.0, 0.0, 0.5), col, (0.05, 0.048, 0.045))
        col = k.mix(k.mr(big, 0.55, 0.75, 0.0, 0.4), col, (0.16, 0.11, 0.07))
    else:
        col = k.mix(k.mr(mid, 0.3, 0.7), (0.15, 0.14, 0.125), (0.22, 0.21, 0.19))
        col = k.mix(k.mr(fine, 0.5, 0.7, 0.0, 0.5), col, (0.10, 0.095, 0.085))
    e = k.edge(0.008, 0.995, 0.93)
    cv = k.cavity(0.05, 0.92, 0.45)
    col = k.mix(k.m("MULTIPLY", e, 0.35), col, k.mul_col(col, (1.35, 1.33, 1.3)))
    col = k.mix(k.m("MULTIPLY", cv, 0.6), col, k.mul_col(col, (0.45, 0.43, 0.42)))
    sm = k.m("MULTIPLY", k.m("ADD", k.m("MULTIPLY", cv, 0.3), k.mr(big, 0.25, 0.65, 0.4, 1.0)), soot, clamp=True)
    if soot_z:
        sm = k.m("MULTIPLY", sm, k.mr(s.outputs["Z"], soot_z[0], soot_z[1]))
    sootc = k.mix(k.mr(fine, 0.3, 0.7), (0.016, 0.015, 0.014), (0.045, 0.040, 0.036))
    col = k.mix(sm, col, sootc)
    nz = k.N.new("ShaderNodeSeparateXYZ")
    k.L.new(k.geo.outputs["Normal"], nz.inputs[0])
    dust = k.m("MULTIPLY", k.mr(nz.outputs["Z"], 0.6, 0.95), k.mr(mid, 0.35, 0.7, 0.1, 0.5))
    col = k.mix(dust, col, (0.22, 0.21, 0.19))
    col = k.mix(k.m("MULTIPLY", k.mr(s.outputs["Z"], 0.3, 0.0), k.mr(big, 0.3, 0.7, 0.4 * grime_base, grime_base)),
                col, (0.06, 0.05, 0.04))
    rough = k.mixf(sm, k.mr(fine, 0.3, 0.7, 0.78, 0.95), 0.9)
    h = k.m("ADD", k.m("MULTIPLY", fine, 0.6), k.m("MULTIPLY", mid, 0.5))
    h = k.m("ADD", h, k.mr(vor, 0.0, 0.1, -0.3, 0.0))
    return k.out(col, rough, 0.0, k.bump(h, 0.0015, 0.55 if kind == "stone" else 0.4))
