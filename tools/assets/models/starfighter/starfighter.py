"""Model script: single-seat twin-engine starfighter (original design), ~12 m long (Y), ~9 m span (X), ~3.6 m tall.

Front faces -Y (nose), engines exhaust to +Y. Parts are built for the right side (x >= 0) where the ship is
symmetric and mirrored at the end. Every part is its own mesh object (the harness joins them); the parts intersect
on purpose (fuselage, nacelles, wings, fins) and nothing floats.

Parts: fuselage (chined loft), canopy + frame bows, dorsal spine, intake housings with fan faces, nacelles,
nozzles with hot cavities, wings (anhedral, swept), canards, canted fins, gear (nose + twin-wheel mains, doors),
weapons (wingtip rails and missiles, underwing pylons and stores, wing-root cannons), greebles.
Materials: painted hull (two-tone, panel lines, decal blocks, edge wear, grime), dark mechanical metal, tyre
rubber, tinted canopy glass, hot engine interior (base colour carries the glow; emission is not baked).
"""

import math

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

# ------------------------------------------------------------------ parameters (metres)
NOSE_Y, TAIL_Y = -5.75, 5.0          # fuselage tip and tail
NAC_X, NAC_Z = 1.38, 1.72            # nacelle centre line
NAC_HW, NAC_HH = 0.60, 0.55          # nacelle half width / half height
NOZ_Y0, NOZ_Y1 = 5.25, 5.95          # nozzle start / exit
INTAKE_Y0, INTAKE_RAKE = -2.15, 0.25
WING = dict(root_x=1.2, root_le=-0.4, root_chord=4.6, root_z=1.50, root_t=0.26,
            tip_x=4.35, tip_le=3.0, tip_chord=1.5, tip_t=0.07, anhedral_deg=4.0)
FIN = dict(root=(1.45, 3.0, 2.12), root_chord=2.6, root_t=0.16, height=1.55, cant_deg=18.0,
           tip_le_y=4.9, tip_chord=1.1, tip_t=0.05)
CANARD = dict(root=(0.45, -4.35, 1.66), root_chord=1.2, root_t=0.07, tip=(1.45, -3.55, 1.74), tip_chord=0.38,
              tip_t=0.03)
NOSE_GEAR_Y, MAIN_GEAR_Y = -3.9, 1.9
SEG = 56                              # ring resolution of lofts


# ------------------------------------------------------------------ small maths
def spline(keys, x):
    """Catmull-Rom (non-uniform) through keys [(x, v), ...]; v may be a tuple. Clamped at the ends."""
    xs = [k[0] for k in keys]
    if x <= xs[0]:
        return keys[0][1]
    if x >= xs[-1]:
        return keys[-1][1]
    i = max(j for j in range(len(xs) - 1) if xs[j] <= x)
    x0, x1 = xs[i], xs[i + 1]
    h = x1 - x0
    t = (x - x0) / h

    def comp(c):
        v = [k[1][c] if isinstance(k[1], tuple) else k[1] for k in keys]
        m0 = (v[i + 1] - v[i - 1]) / (xs[i + 1] - xs[i - 1]) if i > 0 else (v[1] - v[0]) / (xs[1] - xs[0])
        m1 = ((v[i + 2] - v[i]) / (xs[i + 2] - xs[i]) if i + 2 < len(xs)
              else (v[-1] - v[-2]) / (xs[-1] - xs[-2]))
        t2, t3 = t * t, t * t * t
        return ((2 * t3 - 3 * t2 + 1) * v[i] + (t3 - 2 * t2 + t) * h * m0 + (-2 * t3 + 3 * t2) * v[i + 1]
                + (t3 - t2) * h * m1)

    if isinstance(keys[0][1], tuple):
        return tuple(comp(c) for c in range(len(keys[0][1])))
    return comp(0)


def sgnpow(v, p):
    return math.copysign(abs(v) ** p, v)


def section(cx, cz, hw, ht, hb, nxt=2.0, nzt=2.0, nxb=2.0, nzb=2.0, seg=SEG):
    """Closed section in the XZ plane: superellipse halves with separate exponents for the top and bottom.
    nz < 2 gives a chine (pointed side), nx > 2 a flat top/bottom."""
    pts = []
    for k in range(seg):
        a = 2 * math.pi * (k + 0.5) / seg
        c, s = math.cos(a), math.sin(a)
        if s >= 0:
            pts.append((cx + hw * sgnpow(c, 2 / nxt), cz + ht * sgnpow(s, 2 / nzt)))
        else:
            pts.append((cx + hw * sgnpow(c, 2 / nxb), cz + hb * sgnpow(s, 2 / nzb)))
    return pts


# ------------------------------------------------------------------ object helpers
MATS = {}
PARTS = []


def link(name, bm, material, smooth=35, mirror=True):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(MATS[material])
    ob["smooth"], ob["mirror"] = smooth, mirror
    PARTS.append(ob)
    return ob


def loft(name, rings, material, smooth=35, mirror=True):
    """rings: list of 3D point lists (same length), or a single point (a pointed tip)."""
    bm = bmesh.new()
    vs = [[bm.verts.new(p) for p in r] for r in rings]
    n = max(len(r) for r in vs)
    for a, b in zip(vs, vs[1:]):
        if len(a) == 1:
            for i in range(n):
                bm.faces.new((a[0], b[(i + 1) % n], b[i]))
        elif len(b) == 1:
            for i in range(n):
                bm.faces.new((a[i], a[(i + 1) % n], b[0]))
        else:
            for i in range(n):
                bm.faces.new((a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]))
    if len(vs[0]) > 1:
        bm.faces.new(list(reversed(vs[0])))
    if len(vs[-1]) > 1:
        bm.faces.new(vs[-1])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return link(name, bm, material, smooth, mirror)


def y_loft(name, ys, sec_fn, material, smooth=35, mirror=True, yfn=None):
    """Loft along Y; sec_fn(y) -> list of (x, z); yfn(y, x, z) -> y (rake)."""
    rings = []
    for y in ys:
        r = sec_fn(y)
        if len(r) == 1:
            x, z = r[0]
            rings.append([(x, y, z)])
        else:
            rings.append([(x, yfn(y, x, z) if yfn else y, z) for (x, z) in r])
    return loft(name, rings, material, smooth, mirror)


def frame_from(p0, p1):
    """Matrix taking +Z of a primitive at the origin onto the segment p0 -> p1 (centred)."""
    p0, p1 = Vector(p0), Vector(p1)
    d = p1 - p0
    q = d.normalized().to_track_quat("Z", "Y")
    return Matrix.Translation((p0 + p1) / 2) @ q.to_matrix().to_4x4(), d.length


def cyl(name, p0, p1, r, material, r2=None, seg=24, smooth=35, mirror=True, bevel=None):
    m, length = frame_from(p0, p1)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=seg, radius1=r,
                          radius2=r if r2 is None else r2, depth=length, matrix=m)
    ob = link(name, bm, material, smooth, mirror)
    if bevel:
        add_bevel(ob, *bevel)
    return ob


def box(name, center, size, material, rot=(0, 0, 0), smooth=35, mirror=True, bevel=(0.01, 2)):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    m = (Matrix.Translation(center) @ Euler(rot).to_matrix().to_4x4()
         @ Matrix.Diagonal((*size, 1)))
    bmesh.ops.transform(bm, matrix=m, verts=bm.verts)
    ob = link(name, bm, material, smooth, mirror)
    if bevel:
        add_bevel(ob, *bevel)
    return ob


def sphere(name, center, radii, material, smooth=80, mirror=True):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
    bmesh.ops.transform(bm, matrix=Matrix.Translation(center) @ Matrix.Diagonal((*radii, 1)), verts=bm.verts)
    return link(name, bm, material, smooth, mirror)


def add_bevel(ob, width, segments=2, angle=40):
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width, m.segments, m.limit_method, m.angle_limit = width, segments, "ANGLE", math.radians(angle)
    m.miter_outer = "MITER_ARC"
    return m


def cut(ob, cutter, transfer=True):
    m = ob.modifiers.new("cut_" + cutter.name, "BOOLEAN")
    m.operation, m.object, m.solver = "DIFFERENCE", cutter, "EXACT"
    if transfer:
        m.material_mode = "TRANSFER"
    cutter["cutter"] = True
    return m


def airfoil(le, chord_dir, chord, thick_dir, thick, n=18):
    """Symmetric NACA-like section: upper TE->LE then lower LE->TE. thick = full max thickness (m)."""
    le, cd, td = Vector(le), Vector(chord_dir).normalized(), Vector(thick_dir).normalized()
    us = [0.5 * (1 - math.cos(math.pi * i / n)) for i in range(n + 1)]   # 0..1, dense at LE and TE

    def yt(u):
        return 5 * thick * (0.2969 * math.sqrt(u) - 0.126 * u - 0.3516 * u ** 2 + 0.2843 * u ** 3
                            - 0.1015 * u ** 4) + 0.004
    pts = [le + cd * chord * u + td * yt(u) for u in reversed(us[1:])]
    pts.append(le + cd * chord * 0.0)
    pts += [le + cd * chord * u - td * yt(u) for u in us[1:]]
    return [tuple(p) for p in pts]


def surface(name, sections, material, steps=6, mirror=True, smooth=40):
    """Loft between airfoil sections given as dicts (le, chord_dir, chord, thick_dir, thick), linearly
    interpolated with `steps` rings between consecutive sections; the tip ring shrinks slightly for a rounded end."""
    rings = []
    for a, b in zip(sections, sections[1:]):
        for i in range(steps):
            t = i / steps
            mix = {k: (Vector(a[k]).lerp(Vector(b[k]), t) if isinstance(a[k], tuple) else a[k] + (b[k] - a[k]) * t)
                   for k in a}
            rings.append(airfoil(tuple(mix["le"]), tuple(mix["chord_dir"]), mix["chord"], tuple(mix["thick_dir"]),
                                 mix["thick"]))
    rings.append(airfoil(**{k: tuple(v) if isinstance(v, Vector) else v for k, v in sections[-1].items()}))
    return loft(name, rings, material, smooth, mirror)


# ------------------------------------------------------------------ materials
class NodeKit:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.N, self.L = self.nt.nodes, self.nt.links
        self.pr = self.N["Principled BSDF"]
        self.geo = self.N.new("ShaderNodeNewGeometry")
        sp = self.N.new("ShaderNodeSeparateXYZ")
        self.L.new(self.geo.outputs["Position"], sp.inputs["Vector"])
        self.px, self.py, self.pz = sp.outputs["X"], sp.outputs["Y"], sp.outputs["Z"]
        sn = self.N.new("ShaderNodeSeparateXYZ")
        self.L.new(self.geo.outputs["Normal"], sn.inputs["Vector"])
        self.nx, self.ny, self.nz = sn.outputs["X"], sn.outputs["Y"], sn.outputs["Z"]
        self.ax = self.m("ABSOLUTE", self.px)

    def node(self, t, **inputs):
        n = self.N.new(t)
        for k, v in inputs.items():
            self.set(n.inputs[k], v)
        return n

    def set(self, sock, v):
        if isinstance(v, bpy.types.NodeSocket):
            self.L.new(v, sock)
        else:
            sock.default_value = v

    def m(self, op, a, b=None, clamp=False):
        n = self.N.new("ShaderNodeMath")
        n.operation, n.use_clamp = op, clamp
        for i, x in enumerate((a, b)):
            if x is not None:
                self.set(n.inputs[i], x)
        return n.outputs[0]

    def mr(self, x, fmin, fmax, tmin=0.0, tmax=1.0):
        n = self.N.new("ShaderNodeMapRange")
        n.clamp = True
        for k, v in (("From Min", fmin), ("From Max", fmax), ("To Min", tmin), ("To Max", tmax), ("Value", x)):
            self.set(n.inputs[k], v)
        return n.outputs["Result"]

    def band(self, x, lo, hi, soft=0.006):
        return self.m("MULTIPLY", self.mr(x, lo - soft, lo + soft), self.mr(x, hi + soft, hi - soft))

    def mul(self, *xs):
        out = xs[0]
        for x in xs[1:]:
            out = self.m("MULTIPLY", out, x, clamp=True)
        return out

    def add(self, *xs):
        out = xs[0]
        for x in xs[1:]:
            out = self.m("ADD", out, x, clamp=True)
        return out

    def mix(self, fac, a, b):
        n = self.N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        self.set(n.inputs["Factor"], fac)
        self.set(n.inputs[6], a if not isinstance(a, tuple) or len(a) == 4 else (*a, 1))
        self.set(n.inputs[7], b if not isinstance(b, tuple) or len(b) == 4 else (*b, 1))
        return n.outputs[2]

    def fmix(self, fac, a, b):
        n = self.N.new("ShaderNodeMix")
        self.set(n.inputs["Factor"], fac)
        self.set(n.inputs[2], a)
        self.set(n.inputs[3], b)
        return n.outputs[0]

    def vec(self, x, y, z):
        n = self.N.new("ShaderNodeCombineXYZ")
        for k, v in zip("XYZ", (x, y, z)):
            self.set(n.inputs[k], v)
        return n.outputs[0]

    def noise(self, scale, detail=4.0, rough=0.5, vec=None, stretch=None):
        n = self.node("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough)
        v = vec or self.geo.outputs["Position"]
        if stretch:
            mp = self.N.new("ShaderNodeMapping")
            mp.inputs["Scale"].default_value = stretch
            self.L.new(v, mp.inputs["Vector"])
            v = mp.outputs[0]
        self.L.new(v, n.inputs["Vector"])
        return n.outputs["Fac"]

    def edge(self, radius, lo=0.995, hi=0.93):
        bev = self.node("ShaderNodeBevel", Radius=radius)
        bev.samples = 8
        d = self.N.new("ShaderNodeVectorMath")
        d.operation = "DOT_PRODUCT"
        self.L.new(bev.outputs["Normal"], d.inputs[0])
        self.L.new(self.geo.outputs["Normal"], d.inputs[1])
        return self.mr(d.outputs["Value"], lo, hi), bev

    def ao(self, dist, lo=0.9, hi=0.4):
        a = self.N.new("ShaderNodeAmbientOcclusion")
        a.samples = 16
        a.inputs["Distance"].default_value = dist
        return self.mr(a.outputs["AO"], lo, hi)

    def triplanar(self, fn):
        """fn(vector) -> (mask, rand); picks the projection by the dominant normal axis."""
        anx, any_, anz = (self.m("ABSOLUTE", c) for c in (self.nx, self.ny, self.nz))
        wz = self.m("MULTIPLY", self.m("GREATER_THAN", anz, any_), self.m("GREATER_THAN", anz, anx))
        wx = self.m("MULTIPLY", self.m("GREATER_THAN", anx, any_), self.m("GREATER_THAN", anx, anz))
        wy = self.m("SUBTRACT", 1.0, self.add(wz, wx))
        outs = [fn(self.vec(self.px, self.py, 0.0)), fn(self.vec(self.py, self.pz, 0.0)),
                fn(self.vec(self.px, self.pz, 0.0))]
        res = []
        for k in range(2):
            res.append(self.add(self.m("MULTIPLY", wz, outs[0][k]), self.m("MULTIPLY", wx, outs[1][k]),
                                self.m("MULTIPLY", wy, outs[2][k])))
        return res

    def brick(self, v, w, h, mortar, offset=0.5):
        b = self.node("ShaderNodeTexBrick", Scale=1.0)
        b.offset, b.offset_frequency, b.squash, b.squash_frequency = offset, 2, 0.75, 3
        b.inputs["Mortar Size"].default_value = mortar
        b.inputs["Mortar Smooth"].default_value = 0.2
        b.inputs["Brick Width"].default_value = w
        b.inputs["Row Height"].default_value = h
        b.inputs["Bias"].default_value = 0.0
        b.inputs["Color1"].default_value = (0, 0, 0, 1)
        b.inputs["Color2"].default_value = (1, 1, 1, 1)
        b.inputs["Mortar"].default_value = (1, 1, 1, 1)
        self.L.new(v, b.inputs["Vector"])
        rgb = self.m("ADD", b.outputs["Color"], 0.0)     # colour -> float (grey)
        return b.outputs["Fac"], rgb


def paint_material(name, upper, lower, panels=True, decals=True, wear_amt=1.0):
    mat = bpy.data.materials.new(name)
    k = NodeKit(mat)
    # panels: big panels everywhere, small hatches inside a quarter of them
    if panels:
        line_big, rand = k.triplanar(lambda v: k.brick(v, 1.9, 0.95, 0.010))
        line_small, _ = k.triplanar(lambda v: k.brick(v, 0.62, 0.41, 0.008, 0.3))
        hatch = k.mr(rand, 0.72, 0.74)
        lines = k.add(line_big, k.mul(line_small, hatch))
    else:
        lines, rand = 0.0, 0.5
    # region colours
    belly = k.mr(k.add(k.nz, k.m("MULTIPLY", k.noise(3.0), 0.08)), -0.18, -0.26)
    col = k.mix(belly, upper, lower)
    if decals:
        ay, z = k.py, k.pz
        accent = k.add(
            k.band(ay, -5.05, -4.80),                                              # nose band
            k.mul(k.band(k.ax, 3.62, 3.88), k.mr(z, 1.7, 1.6)),                    # wingtip bands
            k.mr(z, 3.30, 3.32),                                                   # fin tips
        )
        col = k.mix(accent, col, (0.60, 0.15, 0.025))
        blue = k.mul(k.band(z, 2.75, 3.12), k.band(ay, 4.55, 5.30), k.mr(k.ax, 1.45, 1.5))
        col = k.mix(blue, col, (0.025, 0.05, 0.13))
        glare = k.mul(k.band(ay, -5.0, -3.95), k.mr(k.nz, 0.55, 0.7))
        col = k.mix(glare, col, (0.02, 0.022, 0.025))
        stencil = k.mul(k.band(ay, -3.55, -3.05), k.band(z, 1.80, 1.90), k.mr(k.ax, 0.4, 0.45))
        col = k.mix(stencil, col, (0.62, 0.48, 0.06))
        text = k.mul(k.band(ay, -3.50, -3.10), k.band(z, 1.825, 1.875), k.mr(k.ax, 0.4, 0.45),
                     k.m("GREATER_THAN", k.m("FRACT", k.m("MULTIPLY", ay, 16.0)), 0.45))
        col = k.mix(text, col, (0.02, 0.02, 0.02))
        yr = k.m("ADD", ay, k.m("MULTIPLY", k.m("SUBTRACT", z, NAC_Z), INTAKE_RAKE / NAC_HH))
        hz = k.mul(k.band(yr, INTAKE_Y0 - 0.02, INTAKE_Y0 + 0.2), k.mr(k.ax, 0.85, 0.9), k.mr(z, 2.4, 2.35))
        stripes = k.m("GREATER_THAN", k.m("FRACT", k.m("MULTIPLY", k.add(k.ax, z), 10.0)), 0.5)
        col = k.mix(hz, col, k.mix(stripes, (0.02, 0.02, 0.02), (0.62, 0.45, 0.04)))
        walk_o = k.mul(k.band(k.ax, 1.95, 2.75), k.band(ay, 1.2, 2.8), k.mr(k.nz, 0.7, 0.8))
        walk_i = k.mul(k.band(k.ax, 1.99, 2.71), k.band(ay, 1.24, 2.76))
        col = k.mix(k.m("SUBTRACT", walk_o, walk_i, clamp=True), col, (0.03, 0.03, 0.03))
        # control surfaces: flap/aileron hinge line at 78 % chord, split at 2.95 m and 3.6 m
        w = WING
        span = w["tip_x"] - w["root_x"]
        f = k.m("DIVIDE", k.m("SUBTRACT", k.ax, w["root_x"]), span)
        le = k.m("ADD", w["root_le"], k.m("MULTIPLY", f, w["tip_le"] - w["root_le"]))
        ch = k.m("ADD", w["root_chord"], k.m("MULTIPLY", f, w["tip_chord"] - w["root_chord"]))
        yl = k.m("ADD", le, k.m("MULTIPLY", ch, 0.76))
        onwing = k.mul(k.band(k.ax, 2.05, 4.2), k.mr(k.m("ABSOLUTE", k.nz), 0.75, 0.85))
        hinge = k.mr(k.m("ABSOLUTE", k.m("SUBTRACT", ay, yl)), 0.009, 0.004)
        aft = k.mr(k.m("SUBTRACT", ay, yl), 0.0, 0.01)
        splits = k.add(k.mr(k.m("ABSOLUTE", k.m("SUBTRACT", k.ax, 2.95)), 0.009, 0.004),
                       k.mr(k.m("ABSOLUTE", k.m("SUBTRACT", k.ax, 3.6)), 0.009, 0.004))
        lines = k.add(lines, k.mul(onwing, k.add(hinge, k.mul(splits, aft))))
    # per-panel tone
    col = k.mix(k.m("MULTIPLY", rand, 0.18), col, (0.0, 0.0, 0.0))
    mottle = k.mr(k.noise(0.9, 4, 0.6), 0.35, 0.65)                          # faded / touched-up paint
    col = k.mix(k.m("MULTIPLY", mottle, 0.25), col, (0.10, 0.10, 0.095))
    # wear + grime
    edge, _ = k.edge(0.02)
    wear = k.mul(edge, k.mr(k.noise(7.0, 6.0, 0.6), 0.42, 0.56), wear_amt)
    ao = k.ao(0.6)
    streak = k.mul(k.mr(k.noise(2.5, 5, 0.55, stretch=(5.0, 5.0, 0.35)), 0.52, 0.75),
                   k.m("SUBTRACT", 1.0, k.m("ABSOLUTE", k.nz)), 0.9)
    soot = k.mul(k.mr(k.py, 3.8, 6.0), k.mr(k.py, 3.8, 6.0), k.mr(k.noise(4.0), 0.3, 0.6, 0.6, 1.0))
    grime = k.add(k.m("MULTIPLY", ao, 0.55), streak, soot, k.m("MULTIPLY", belly, 0.12))
    col = k.mix(k.m("MULTIPLY", grime, 0.9), col, (0.025, 0.023, 0.02))
    col = k.mix(k.m("MULTIPLY", lines, 0.45), col, (0.02, 0.02, 0.02))
    col = k.mix(wear, col, (0.50, 0.50, 0.50))
    k.L.new(col, k.pr.inputs["Base Color"])
    rough = k.add(k.mr(k.noise(30.0, 6), 0.3, 0.7, 0.46, 0.6), k.m("MULTIPLY", rand, 0.08),
                  k.m("MULTIPLY", grime, 0.25), k.m("MULTIPLY", lines, 0.25))
    k.L.new(k.fmix(wear, rough, 0.3), k.pr.inputs["Roughness"])
    k.L.new(wear, k.pr.inputs["Metallic"])
    _, bev = k.edge(0.008)
    bump = k.node("ShaderNodeBump", Strength=1.0, Distance=0.004)
    k.L.new(bev.outputs["Normal"], bump.inputs["Normal"])
    k.L.new(k.add(k.m("MULTIPLY", k.m("SUBTRACT", 1.0, lines), 1.0), k.m("MULTIPLY", k.noise(60.0, 3), 0.04)),
            bump.inputs["Height"])
    k.L.new(bump.outputs["Normal"], k.pr.inputs["Normal"])
    return mat


def mech_material(name, base=(0.045, 0.047, 0.05), heat=True):
    mat = bpy.data.materials.new(name)
    k = NodeKit(mat)
    col = k.mix(k.mr(k.noise(6.0, 5), 0.3, 0.7), base, tuple(c * 1.6 for c in base))
    if heat:
        ramp = k.N.new("ShaderNodeValToRGB")
        cr = ramp.color_ramp
        cr.elements[0].position, cr.elements[0].color = 0.0, (*base, 1)
        cr.elements[1].position, cr.elements[1].color = 1.0, (0.035, 0.04, 0.075, 1)
        e = cr.elements.new(0.5)
        e.color = (0.06, 0.055, 0.05, 1)
        k.L.new(k.mr(k.py, 5.45, 5.95), ramp.inputs["Fac"])
        col = k.mix(k.mr(k.py, 5.4, 5.55), col, ramp.outputs["Color"])
    edge, _ = k.edge(0.012)
    wear = k.mul(edge, k.mr(k.noise(9.0, 5), 0.45, 0.6))
    ao = k.ao(0.4)
    col = k.mix(k.m("MULTIPLY", ao, 0.7), col, (0.012, 0.011, 0.01))
    col = k.mix(wear, col, (0.35, 0.35, 0.36))
    k.L.new(col, k.pr.inputs["Base Color"])
    k.L.new(k.add(k.mr(k.noise(25.0, 6), 0.3, 0.7, 0.32, 0.55), k.m("MULTIPLY", ao, 0.2)), k.pr.inputs["Roughness"])
    k.pr.inputs["Metallic"].default_value = 0.85
    _, bev = k.edge(0.006)
    k.L.new(bev.outputs["Normal"], k.pr.inputs["Normal"])
    return mat


def simple_material(name, color, rough, metal=0.0, emit=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    pr = mat.node_tree.nodes["Principled BSDF"]
    pr.inputs["Base Color"].default_value = (*color, 1)
    pr.inputs["Roughness"].default_value = rough
    pr.inputs["Metallic"].default_value = metal
    if emit:
        pr.inputs["Emission Color"].default_value = (*color, 1)
        pr.inputs["Emission Strength"].default_value = emit
    return mat


def build_materials():
    MATS["paint"] = paint_material("hull_paint", (0.19, 0.205, 0.22), (0.06, 0.065, 0.075))
    MATS["ordnance"] = paint_material("ordnance", (0.55, 0.55, 0.52), (0.45, 0.45, 0.43), panels=False,
                                      decals=False, wear_amt=0.5)
    MATS["accent"] = paint_material("accent", (0.60, 0.15, 0.025), (0.60, 0.15, 0.025), panels=False,
                                    decals=False, wear_amt=0.6)
    MATS["mech"] = mech_material("mechanical")
    MATS["tyre"] = simple_material("tyre", (0.018, 0.018, 0.018), 0.85)
    MATS["glass"] = simple_material("canopy_glass", (0.012, 0.018, 0.022), 0.04)
    MATS["hot"] = simple_material("engine_hot", (1.0, 0.40, 0.07), 0.5, emit=6.0)
    MATS["seeker"] = simple_material("seeker", (0.02, 0.025, 0.03), 0.1, metal=0.3)


# ------------------------------------------------------------------ fuselage, canopy, spine
FUSE_KEYS = [  # y: (half width, chine z, height above chine, depth below chine)
    (NOSE_Y, (0.015, 1.72, 0.012, 0.012)),
    (-5.6, (0.16, 1.70, 0.10, 0.10)),
    (-5.2, (0.36, 1.68, 0.24, 0.20)),
    (-4.6, (0.56, 1.66, 0.42, 0.30)),
    (-3.8, (0.70, 1.66, 0.60, 0.38)),
    (-2.6, (0.78, 1.68, 0.74, 0.44)),
    (-1.2, (0.82, 1.70, 0.78, 0.48)),
    (0.5, (0.80, 1.72, 0.76, 0.50)),
    (2.5, (0.70, 1.74, 0.66, 0.46)),
    (4.0, (0.50, 1.76, 0.50, 0.36)),
    (4.8, (0.28, 1.78, 0.30, 0.22)),
    (TAIL_Y, (0.05, 1.78, 0.06, 0.05)),
]


def fuse_at(y):
    return spline(FUSE_KEYS, y)


def fuse_top(y):
    hw, zc, ht, hb = fuse_at(y)
    return zc + ht


def fuse_bottom(y):
    hw, zc, ht, hb = fuse_at(y)
    return zc - hb


def ys_between(a, b, n, bias=1.0):
    return [a + (b - a) * (i / n) ** bias for i in range(n + 1)]


def build_fuselage():
    ys = [NOSE_Y] + ys_between(-5.70, TAIL_Y - 0.02, 54)

    def sec(y):
        hw, zc, ht, hb = fuse_at(y)
        return section(0.0, zc, hw, ht, hb, nxt=2.2, nzt=1.35, nxb=3.2, nzb=1.25)
    fz = y_loft("fuselage", ys, sec, "paint", smooth=50, mirror=False)
    bay = box("nose_bay", (0, NOSE_GEAR_Y, fuse_bottom(NOSE_GEAR_Y)), (0.36, 0.95, 0.26), "mech", mirror=False,
              bevel=None)
    cut(fz, bay)


CAN_Y0, CAN_Y1 = -4.05, -1.0


def canopy_sec(y, grow=0.0):
    t = min(max((y - CAN_Y0) / (CAN_Y1 - CAN_Y0), 0.0), 1.0)
    h = 0.60 * math.sin(math.pi * t ** 0.72) ** 0.7
    hw = 0.47 * math.sin(math.pi * t ** 0.8) ** 0.55
    base = fuse_top(y)
    return section(0.0, base - 0.12, hw + grow, h + 0.12 + grow, 0.25, nxt=2.0, nzt=2.0)


def build_canopy():
    ys = ys_between(CAN_Y0 + 0.01, CAN_Y1 - 0.01, 40)
    y_loft("canopy", ys, canopy_sec, "glass", smooth=60, mirror=False)
    for name, t in (("bow_front", 0.17), ("bow_rear", 0.70)):
        y = CAN_Y0 + t * (CAN_Y1 - CAN_Y0)
        pts = canopy_sec(y, grow=0.012)
        top = [p for p in pts if p[1] > fuse_top(y) - 0.08]
        cu = bpy.data.curves.new(name, "CURVE")
        cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.03, 2
        cu.use_fill_caps = True
        sp = cu.splines.new("POLY")
        sp.points.add(len(top) - 1)
        for p, (x, z) in zip(sp.points, top):
            p.co = (x, y, z, 1)
        ob = bpy.data.objects.new(name, cu)
        bpy.context.scene.collection.objects.link(ob)
        ob.data.materials.append(MATS["mech"])
        ob["smooth"], ob["mirror"] = 60, False
        PARTS.append(ob)
    # sill rails along the canopy base, both sides
    for side in (1,):
        ys2 = ys_between(CAN_Y0 + 0.25, CAN_Y1 - 0.2, 24)
        pts = []
        for y in ys2:
            sec = canopy_sec(y)
            xz = max((p for p in sec if p[0] > 0), key=lambda p: -abs(p[1] - (fuse_top(y) - 0.03)))
            pts.append((xz[0] + 0.01, y, xz[1]))
        cu = bpy.data.curves.new("sill", "CURVE")
        cu.dimensions, cu.bevel_depth, cu.bevel_resolution = "3D", 0.028, 2
        cu.use_fill_caps = True
        sp = cu.splines.new("POLY")
        sp.points.add(len(pts) - 1)
        for p, c in zip(sp.points, pts):
            p.co = (*c, 1)
        ob = bpy.data.objects.new("sill", cu)
        bpy.context.scene.collection.objects.link(ob)
        ob.data.materials.append(MATS["mech"])
        ob["smooth"], ob["mirror"] = 60, True
        PARTS.append(ob)


def build_body():
    """Flat blended body between fuselage and nacelles: one lifting surface instead of three tubes."""
    y0, y1 = -1.7, 4.95

    def sec(y):
        t = (y - y0) / (y1 - y0)
        hw = spline([(0.0, 0.75), (0.25, 2.15), (0.78, 2.15), (1.0, NAC_X)], t)
        zt = spline([(0.0, 1.95), (0.25, 2.22), (0.8, 2.18), (1.0, 2.0)], t)
        zb = spline([(0.0, 1.45), (0.25, 1.30), (0.8, 1.32), (1.0, 1.50)], t)
        zc = 1.62
        return section(0.0, zc, hw, zt - zc, zc - zb, nxt=3.2, nzt=1.5, nxb=4.0, nzb=1.3, seg=72)
    y_loft("body", ys_between(y0, y1, 40), sec, "paint", smooth=50, mirror=False)


def build_spine():
    y0, y1 = -1.5, 4.7

    def sec(y):
        t = (y - y0) / (y1 - y0)
        h = 0.24 * math.sin(math.pi * min(max(t, 0), 1) ** 0.6) ** 0.8
        hw = 0.30 * math.sin(math.pi * min(max(t, 0), 1) ** 0.5) ** 0.4 + 0.02
        return section(0.0, fuse_top(y) - 0.12, hw, h + 0.12, 0.2, nxt=2.6, nzt=1.8)
    y_loft("spine", ys_between(y0 + 0.02, y1 - 0.02, 36), sec, "paint", smooth=50, mirror=False)
    # louvres on the spine
    for i in range(5):
        y = 0.25 + i * 0.2
        box(f"louvre_{i}", (0.0, y, fuse_top(y) + 0.20), (0.36, 0.07, 0.07), "mech", rot=(math.radians(-20), 0, 0),
            mirror=False, bevel=(0.008, 2))
    # antenna blade
    y = 2.6
    zt = fuse_top(y) + 0.15
    surface("antenna", [dict(le=(0, y, zt - 0.05), chord_dir=(0, 1, 0), chord=0.5, thick_dir=(1, 0, 0), thick=0.04),
                        dict(le=(0, y + 0.35, zt + 0.32), chord_dir=(0, 1, 0), chord=0.2, thick_dir=(1, 0, 0),
                             thick=0.02)], "paint", steps=3, mirror=False)
    # pitot, chin sensor, tail stinger
    cyl("pitot", (0, NOSE_Y + 0.05, 1.72), (0, NOSE_Y - 0.30, 1.72), 0.018, "mech", r2=0.01, seg=12, mirror=False)
    sphere("chin_sensor", (0, -4.75, fuse_bottom(-4.75) + 0.02), (0.14, 0.24, 0.10), "seeker", mirror=False)
    cyl("stinger", (0, 4.5, 1.78), (0, 5.75, 1.78), 0.11, "mech", r2=0.07, seg=20, mirror=False)
    sphere("tail_light", (0, 5.76, 1.78), (0.06, 0.06, 0.06), "hot", mirror=False)


# ------------------------------------------------------------------ engines
def nac_sec(y):
    hw = spline([(-1.0, 0.50), (-0.2, NAC_HW), (4.0, NAC_HW), (NOZ_Y0 + 0.05, 0.57)], y)
    hh = spline([(-1.0, 0.48), (-0.2, NAC_HH), (4.0, NAC_HH), (NOZ_Y0 + 0.05, 0.56)], y)
    return section(NAC_X, NAC_Z, hw, hh, hh, nxt=2.6, nzt=2.6, nxb=2.6, nzb=2.6, seg=48)


def build_engines():
    nac = y_loft("nacelle", ys_between(-1.0, NOZ_Y0 + 0.05, 40), nac_sec, "paint", smooth=50)
    bay = box("main_bay", (NAC_X, MAIN_GEAR_Y - 0.1, NAC_Z - NAC_HH), (0.5, 1.4, 0.3), "mech", bevel=None)
    cut(nac, bay)

    # intake housing (raked), cavity cut into it, fan face at the back
    def rake(y, x, z):
        return y - INTAKE_RAKE * (z - NAC_Z) / NAC_HH if y < INTAKE_Y0 + 0.3 else y

    def hsec(y):
        return section(NAC_X + 0.02, NAC_Z, 0.56, 0.54, 0.52, nxt=4.0, nzt=4.0, nxb=4.0, nzb=4.0, seg=48)
    housing = y_loft("intake_housing", [INTAKE_Y0, INTAKE_Y0 + 0.03, INTAKE_Y0 + 0.08] + ys_between(
        INTAKE_Y0 + 0.2, -0.3, 6), hsec, "paint", smooth=40, yfn=rake)
    bev = add_bevel(housing, 0.025, 3, 50)

    def csec(y):
        return section(NAC_X + 0.02, NAC_Z, 0.44, 0.42, 0.40, nxt=4.0, nzt=4.0, nxb=4.0, nzb=4.0, seg=48)
    cutter = y_loft("intake_cut", [INTAKE_Y0 - 0.5, INTAKE_Y0 + 0.65], csec, "mech", smooth=40)
    housing.modifiers.remove(bev)
    cut(housing, cutter)
    add_bevel(housing, 0.02, 3, 50)
    yb = INTAKE_Y0 + 0.65
    cyl("fan_hub", (NAC_X + 0.02, yb + 0.01, NAC_Z), (NAC_X + 0.02, yb - 0.22, NAC_Z), 0.15, "mech", r2=0.03)
    for i in range(14):
        a = 2 * math.pi * i / 14
        r = 0.27
        c = (NAC_X + 0.02 + r * math.cos(a), yb - 0.03, NAC_Z + r * math.sin(a))
        box(f"fan_blade_{i}", c, (0.30, 0.05, 0.015), "mech", rot=(0, -a, 0), bevel=None)

    # nozzle: tapered ring with a hot cavity and an exhaust cone, petals outside
    def noz_r(y):
        return spline([(NOZ_Y0, 0.555), (NOZ_Y0 + 0.15, 0.55), (NOZ_Y1, 0.48)], y)
    noz = y_loft("nozzle", ys_between(NOZ_Y0, NOZ_Y1, 6),
                 lambda y: section(NAC_X, NAC_Z, noz_r(y), noz_r(y), noz_r(y), seg=48), "mech", smooth=50)
    hole = cyl("nozzle_cut", (NAC_X, NOZ_Y0 + 0.25, NAC_Z), (NAC_X, NOZ_Y1 + 0.3, NAC_Z), 0.42, "mech", r2=0.43,
               seg=48)
    cut(noz, hole)
    add_bevel(noz, 0.012, 2, 50)
    yb = NOZ_Y0 + 0.25
    cyl("burner_plate", (NAC_X, yb - 0.01, NAC_Z), (NAC_X, yb + 0.02, NAC_Z), 0.405, "hot", seg=48)
    cyl("exhaust_cone", (NAC_X, yb + 0.01, NAC_Z), (NAC_X, yb + 0.33, NAC_Z), 0.20, "mech", r2=0.04, seg=32)
    for i in range(8):   # flame-holder spokes
        a = 2 * math.pi * i / 8
        ob = box(f"spoke_{i}", (0, 0, 0), (0.025, 0.04, 0.24), "mech", bevel=None)
        ob.data.transform(Matrix.Translation((NAC_X, yb + 0.06, NAC_Z)) @ Matrix.Rotation(a, 4, "Y")
                          @ Matrix.Translation((0, 0, 0.29)))
    for i in range(12):
        a = 2 * math.pi * (i + 0.5) / 12
        r = 0.522
        ob = box(f"petal_{i}", (0, 0, 0), (0.17, 0.46, 0.025), "mech", bevel=(0.006, 2))
        m = (Matrix.Translation((NAC_X, (NOZ_Y0 + NOZ_Y1) / 2 + 0.1, NAC_Z)) @ Matrix.Rotation(a, 4, "Y")
             @ Matrix.Translation((0, 0, r)) @ Matrix.Rotation(math.radians(-6), 4, "X"))
        ob.data.transform(m)
    for i in range(7):
        y = 3.2 + i * 0.16
        box(f"vent_{i}", (NAC_X, y, NAC_Z + NAC_HH - 0.01), (0.42, 0.07, 0.05), "mech",
            rot=(math.radians(-25), 0, 0), bevel=(0.006, 2))
    box("vent_frame", (NAC_X, 3.68, NAC_Z + NAC_HH - 0.03), (0.52, 1.25, 0.06), "paint", bevel=(0.015, 2))
    cyl("nozzle_ring", (NAC_X, NOZ_Y0 - 0.02, NAC_Z), (NAC_X, NOZ_Y0 + 0.08, NAC_Z), 0.585, "mech", seg=48,
        bevel=(0.01, 2))


# ------------------------------------------------------------------ flying surfaces
def build_wings():
    w = WING
    dz = (w["tip_x"] - w["root_x"]) * math.tan(math.radians(w["anhedral_deg"]))
    root = dict(le=(w["root_x"], w["root_le"], w["root_z"]), chord_dir=(0, 1, 0), chord=w["root_chord"],
                thick_dir=(0, 0, 1), thick=w["root_t"])
    tip = dict(le=(w["tip_x"], w["tip_le"], w["root_z"] - dz), chord_dir=(0, 1, 0), chord=w["tip_chord"],
               thick_dir=(0, 0, 1), thick=w["tip_t"])
    surface("wing", [root, tip], "paint", steps=10)
    f = FIN
    c, s = math.cos(math.radians(f["cant_deg"])), math.sin(math.radians(f["cant_deg"]))
    rx, ry, rz = f["root"]
    root = dict(le=(rx, ry, rz), chord_dir=(0, 1, 0), chord=f["root_chord"], thick_dir=(c, 0, -s), thick=f["root_t"])
    tip = dict(le=(rx + f["height"] * s, f["tip_le_y"], rz + f["height"] * c), chord_dir=(0, 1, 0),
               chord=f["tip_chord"], thick_dir=(c, 0, -s), thick=f["tip_t"])
    surface("fin", [root, tip], "paint", steps=8)
    cn = CANARD
    surface("canard", [dict(le=cn["root"], chord_dir=(0, 1, 0), chord=cn["root_chord"], thick_dir=(0, 0, 1),
                            thick=cn["root_t"]),
                       dict(le=cn["tip"], chord_dir=(0, 1, 0), chord=cn["tip_chord"], thick_dir=(0, 0, 1),
                            thick=cn["tip_t"])], "paint", steps=5)


def wing_z(x):
    w = WING
    t = (x - w["root_x"]) / (w["tip_x"] - w["root_x"])
    return w["root_z"] - t * (w["tip_x"] - w["root_x"]) * math.tan(math.radians(w["anhedral_deg"]))


def wing_le(x):
    w = WING
    t = (x - w["root_x"]) / (w["tip_x"] - w["root_x"])
    return w["root_le"] + t * (w["tip_le"] - w["root_le"])


def wing_half_t(x):
    w = WING
    t = (x - w["root_x"]) / (w["tip_x"] - w["root_x"])
    return 0.5 * (w["root_t"] + t * (w["tip_t"] - w["root_t"]))


# ------------------------------------------------------------------ gear
def wheel(name, center, r, width, seg=32):
    x, y, z = center
    t = cyl(name, (x - width / 2, y, z), (x + width / 2, y, z), r, "tyre", seg=seg, smooth=40,
            bevel=(width * 0.32, 4, 30))
    cyl(name + "_hub", (x - width / 2 - 0.012, y, z), (x + width / 2 + 0.012, y, z), r * 0.55, "mech", seg=seg,
        bevel=(0.01, 2))
    return t


def build_gear():
    # nose gear (on the centre line)
    y, zb = NOSE_GEAR_Y, fuse_bottom(NOSE_GEAR_Y)
    r = 0.30
    cyl("ng_strut", (0, y, 0.64), (0, y, zb + 0.15), 0.05, "mech", seg=16, mirror=False)
    cyl("ng_sleeve", (0, y, 1.0), (0, y, zb + 0.15), 0.08, "mech", seg=20, mirror=False, bevel=(0.01, 2))
    box("ng_yoke", (0, y, 0.64), (0.36, 0.12, 0.08), "mech", mirror=False)
    box("ng_fork", (0.14, y, 0.46), (0.035, 0.09, 0.36), "mech")
    cyl("ng_axle", (-0.17, y, r), (0.17, y, r), 0.03, "mech", seg=12, mirror=False)
    wheel("ng_wheel", (0, y, r), r, 0.15)
    box("ng_door", (0.195, y, zb - 0.18), (0.02, 0.9, 0.46), "paint", bevel=(0.006, 2))
    # main gear under each nacelle, twin wheels
    y = MAIN_GEAR_Y
    zb = NAC_Z - NAC_HH
    r = 0.38
    cyl("mg_strut", (NAC_X, y, r), (NAC_X, y, zb + 0.15), 0.07, "mech", seg=16)
    cyl("mg_sleeve", (NAC_X, y, 0.85), (NAC_X, y, zb + 0.15), 0.10, "mech", seg=20, bevel=(0.01, 2))
    cyl("mg_brace", (NAC_X, y, 0.95), (NAC_X, y - 1.0, zb + 0.08), 0.04, "mech", seg=12)
    box("mg_link", (NAC_X, y - 0.1, 0.62), (0.05, 0.22, 0.05), "mech", rot=(math.radians(35), 0, 0))
    cyl("mg_axle", (NAC_X - 0.30, y, r), (NAC_X + 0.30, y, r), 0.04, "mech", seg=12)
    wheel("mg_wheel_in", (NAC_X - 0.19, y, r), r, 0.17)
    wheel("mg_wheel_out", (NAC_X + 0.19, y, r), r, 0.17)
    box("mg_door", (NAC_X - 0.27, y - 0.1, zb - 0.16), (0.02, 1.25, 0.52), "paint", bevel=(0.006, 2))


# ------------------------------------------------------------------ weapons
def missile(name, y0, y1, x, z, r):
    cyl(name, (x, y0 + 0.35, z), (x, y1, z), r, "ordnance", seg=20)
    cyl(name + "_nose", (x, y0 + 0.36, z), (x, y0, z), r, "seeker", r2=0.01, seg=20, smooth=60)
    cyl(name + "_band", (x, y0 + 0.45, z), (x, y0 + 0.6, z), r * 1.02, "accent", seg=20)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        for (yc, ln, sp) in ((y1 - 0.22, 0.4, 0.17), (y0 + 0.85, 0.22, 0.09)):
            ob = box(f"{name}_fin_{k}_{yc:.2f}", (0, 0, 0), (0.012, ln, sp), "mech", bevel=None)
            ob.data.transform(Matrix.Translation((x, yc, z)) @ Matrix.Rotation(a, 4, "Y")
                              @ Matrix.Translation((0, 0, r + sp / 2 - 0.01)))


def build_weapons():
    w = WING
    # wingtip rail + missile
    xt = w["tip_x"]
    zt = wing_z(xt)
    box("tip_rail", (xt + 0.02, w["tip_le"] + 0.75, zt - 0.06), (0.08, 1.3, 0.12), "mech", bevel=(0.01, 2))
    missile("tip_missile", w["tip_le"] - 1.1, w["tip_le"] + 1.45, xt + 0.03, zt - 0.20, 0.085)
    # underwing pylon + store
    xp = 2.85
    zp = wing_z(xp) - wing_half_t(xp) * 0.7
    le = wing_le(xp)
    box("pylon", (xp, le + 1.2, zp - 0.12), (0.06, 1.7, 0.30), "paint", bevel=(0.012, 2))
    missile("pylon_store", le - 0.4, le + 2.5, xp, zp - 0.40, 0.14)
    # wing-root cannon: fairing + barrel
    xc = 2.05
    zc = wing_z(xc)
    lec = wing_le(xc)
    cyl("gun_fairing", (xc, lec - 0.6, zc), (xc, lec + 1.6, zc), 0.10, "paint", seg=24)
    sphere("gun_fairing_nose", (xc, lec - 0.6, zc), (0.10, 0.25, 0.10), "paint")
    cyl("gun_barrel", (xc, lec - 1.5, zc), (xc, lec - 0.65, zc), 0.032, "mech", seg=16)
    cyl("gun_muzzle", (xc, lec - 1.55, zc), (xc, lec - 1.38, zc), 0.045, "mech", seg=16)


# ------------------------------------------------------------------ finish: apply, smooth, mirror
def finish():
    sc = bpy.context.scene
    cutters = [o for o in sc.objects if o.get("cutter")]
    for ob in [o for o in PARTS if not o.get("cutter")]:
        bpy.ops.object.select_all(action="DESELECT")
        ob.select_set(True)
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.convert(target="MESH")     # applies modifiers, curves -> mesh
        ob = bpy.context.view_layer.objects.active
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        if ob["smooth"]:
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(ob["smooth"]))
        else:
            bpy.ops.object.shade_flat()
    for c in cutters:
        bpy.data.objects.remove(c, do_unlink=True)
    for ob in [o for o in sc.objects if o.type == "MESH" and o.get("mirror")]:
        me = ob.data.copy()
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.scale(bm, vec=(-1, 1, 1), verts=bm.verts)
        bmesh.ops.reverse_faces(bm, faces=bm.faces)
        bm.to_mesh(me)
        bm.free()
        cp = bpy.data.objects.new(ob.name + "_L", me)
        sc.collection.objects.link(cp)


def build(spec=None):
    build_materials()
    build_fuselage()
    build_canopy()
    build_spine()
    build_body()
    build_engines()
    build_wings()
    build_gear()
    build_weapons()
    finish()


build()
