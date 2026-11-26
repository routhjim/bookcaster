"""Shared helpers for period furniture and hand props (model scripts under tools/assets/models/<id>/).

A model script imports this module (it is not on Blender's path, so the script adds this folder first):

    import os, sys
    sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import period_kit as K

Geometry (metres, front -Y, base on Z = 0; every builder returns a linked object):
  lathe(...)        turned part from a (radius, z) profile; keys(), bead(), cove() build profiles
  box(...)          bevelled box, built at true size (object coordinates = metres, so grain is not stretched)
  rbox(...)         rounded box on a fine grid: cushions, upholstery, pillows; per-axis bulge and diamond
                    button tufting on one face (returns the button positions)
  sweep(...)        a centreline with one 2D section per point (parallel-transport frame); tube() for round ones
  moulding(...)     a straight run of a 2D profile with mitred ends (cornices, skirting, frames)
  fielded_panel(...) raised and fielded panel; frame_panel() a door/carcass panel with stiles and rails
  cabriole(...)     a cabriole leg (knee, ankle, pad foot)
  place(), shade(), bevel(), set_color() (per-part colour attribute "pcol" read by the materials)
  picture_uvs()     UVs over the whole texture for a flat picture part (image_material; spec "keep_uvs")

Materials (procedural Principled; the harness bakes them). Wood grain follows the part's OBJECT axis given by
`axis` (build turned parts along local Z and rotate the object; boxes take the axis of their long side):
  wood(species, axis, ...)   mahogany | walnut | oak | pine | rosewood | ebonised | painted_*; varnish, pores,
                             wear on edges, grime in recesses
  fabric(kind, colour, ...)  leather (buttoned, creased, worn shine) | velvet | damask | linen | wool | rope
  metal(kind, ...)           brass | iron (sooted, black-leaded) | steel (blued, worn) | nickel | copper | gold
  glass(...), ceramic(...), paper(...), emissive(...), plaster(...)
Materials are cached by their arguments, so parts share slots.
"""

import math
import random

import bmesh
import bpy
from mathutils import Matrix, Vector

TAU = 2 * math.pi


# ================================================================== basics
def lerp(a, b, t):
    return a + (b - a) * t


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def link(ob):
    bpy.context.scene.collection.objects.link(ob)
    return ob


def from_bm(name, bm, recalc=True):
    if recalc:
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    return link(bpy.data.objects.new(name, me))


def place(ob, loc=(0, 0, 0), rot=(0, 0, 0), mat=None):
    """Object transform (the harness applies it after storing object coordinates)."""
    ob.location = loc
    ob.rotation_euler = rot
    if mat is not None:
        assign(ob, mat)
    return ob


def assign(ob, mat):
    if mat.name not in [m.name for m in ob.data.materials if m]:
        ob.data.materials.append(mat)
    idx = [m.name if m else None for m in ob.data.materials].index(mat.name)
    for p in ob.data.polygons:
        p.material_index = idx
    return ob


def shade(ob, angle=35):
    me = ob.data
    for p in me.polygons:
        p.use_smooth = True
    try:
        me.set_sharp_from_angle(angle=math.radians(angle))
    except AttributeError:
        pass
    return ob


def picture_uvs(ob):
    """UVs that lay the part's Generated X / Y over the whole 0-1 square (image_material's own mapping): a flat
    picture part bakes to the whole texture, its face and its back on the same texels, its thin rims on its edges.
    The harness keeps them when the spec says "keep_uvs" (its own unwrap packed a 21.6 x 8 m picture into a strip
    38 texels high, 11-20)."""
    me = ob.data
    xs = [v.co.x for v in me.vertices]
    ys = [v.co.y for v in me.vertices]
    x0, y0 = min(xs), min(ys)
    sx, sy = max(max(xs) - x0, 1e-9), max(max(ys) - y0, 1e-9)
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    lay = me.uv_layers.new(name="UVMap")
    for lp in me.loops:
        co = me.vertices[lp.vertex_index].co
        lay.data[lp.index].uv = ((co.x - x0) / sx, (co.y - y0) / sy)
    return ob


def bevel(ob, width=0.002, segments=2, angle=40):
    m = ob.modifiers.new("bevel", "BEVEL")
    m.width, m.segments = width, segments
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(angle)
    m.harden_normals = False
    return ob


def warp(ob, fn):
    """Move every vertex: fn(Vector local position) -> new position (tapers, bends, sags)."""
    for v in ob.data.vertices:
        v.co = Vector(fn(v.co.copy()))
    ob.data.update()
    return ob


def nails(prefix, pts, r=0.0055, mat=None, normal=(0, -1, 0)):
    """Upholstery nail heads (or rivets, bolt heads): small domes at world points, facing `normal` (or a list)."""
    out = []
    for k, p in enumerate(pts):
        n = Vector(normal[k] if isinstance(normal, list) else normal)
        b = lathe(f"{prefix}{k}", [(0.0001, -r * 0.3), (r, -r * 0.3), (r, 0.0), (r * 0.75, r * 0.4), (0.0, r * 0.5)],
                  seg=10, cap1=False, mat=mat)
        q = Vector((0, 0, 1)).rotation_difference(n.normalized())
        b.matrix_world = Matrix.Translation(Vector(p)) @ q.to_matrix().to_4x4()
        out.append(b)
    return out


def set_color(ob, rgb, name="pcol"):
    """Per-part colour as a corner attribute (survives the join); materials read it with use_pcol."""
    me = ob.data
    a = me.color_attributes.get(name) or me.color_attributes.new(name, "FLOAT_COLOR", "CORNER")
    col = (rgb[0], rgb[1], rgb[2], 1.0)
    for d in a.data:
        d.color = col
    return ob


def join(objs, name):
    """Join parts (already transformed) into one object; keeps the first object's transform."""
    ctx = bpy.context
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    objs[0].name = name
    return objs[0]


# ================================================================== profiles for turned work
def keys(pts, step=0.003):
    """Profile from (z, r) keys, smoothstep between keys (swells and coves); returns (r, z) points bottom -> top.
    A key given as (z, r, 'l') joins the previous key in a straight line; (z, r, 'c') makes a hard corner."""
    out = []
    for i in range(len(pts) - 1):
        z0, r0 = pts[i][0], pts[i][1]
        z1, r1 = pts[i + 1][0], pts[i + 1][1]
        mode = pts[i + 1][2] if len(pts[i + 1]) > 2 else "s"
        n = max(1, int(math.ceil(max(abs(z1 - z0), abs(r1 - r0)) / step)))
        if mode in ("l", "c"):
            n = 1 if mode == "c" else max(1, n // 2)
        for k in range(n):
            t = k / n
            f = t if mode in ("l", "c") else smooth(t)
            out.append((lerp(r0, r1, f), lerp(z0, z1, t if mode != "s" else t)))
    out.append((pts[-1][1], pts[-1][0]))
    return out


def bead(z, r, h, n=8):
    """Keys for a half-round bead of height h centred at z, standing out to radius r from r - h/2."""
    return [(z - h / 2 + h * (1 - math.cos(math.pi * k / n)) / 2, r - h / 2 + (h / 2) * math.sin(math.pi * k / n), "l")
            for k in range(n + 1)]


def lathe(name, prof, seg=32, cap0=True, cap1=True, mat=None):
    """Surface of revolution about local Z. prof: (r, z) points bottom -> top; r = 0 closes to a point."""
    bm = bmesh.new()
    rings = []
    for r, z in prof:
        if r < 1e-5:
            rings.append([bm.verts.new((0, 0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(TAU * k / seg), r * math.sin(TAU * k / seg), z)) for k in range(seg)])
    for a, b in zip(rings, rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            for k in range(seg):
                bm.faces.new((a[0], b[k], b[(k + 1) % seg]))
        elif len(b) == 1:
            for k in range(seg):
                bm.faces.new((a[k], b[0], a[(k + 1) % seg]))
        else:
            for k in range(seg):
                bm.faces.new((a[k], a[(k + 1) % seg], b[(k + 1) % seg], b[k]))
    if cap0 and len(rings[0]) > 1:
        bm.faces.new(list(reversed(rings[0])))
    if cap1 and len(rings[-1]) > 1:
        bm.faces.new(rings[-1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    ob = from_bm(name, bm)
    shade(ob, 40)
    if mat is not None:
        assign(ob, mat)
    return ob


# ================================================================== boxes and upholstery
def box(name, size, loc=(0, 0, 0), rot=(0, 0, 0), bev=0.002, seg=2, mat=None):
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * size[0], v.co.y * size[1], v.co.z * size[2]))
    ob = from_bm(name, bm)
    place(ob, loc, rot, mat)
    if bev > 0:
        bevel(ob, min(bev, hx * 0.9, hy * 0.9, hz * 0.9), seg)
    shade(ob, 35)
    return ob


def _ticks(h, r, spacing):
    """Grid coordinates on [-h, h]: uniform inside, finer within r of each edge (for rounding)."""
    r = min(r, h)
    t = set()
    n_in = max(1, int(math.ceil(2 * (h - r) / spacing)))
    for k in range(n_in + 1):
        t.add(round(-(h - r) + 2 * (h - r) * k / n_in, 6))
    n_e = max(2, int(math.ceil(r / (spacing * 0.4))))
    for k in range(n_e + 1):
        d = (h - r) + r * math.sin(math.pi / 2 * k / n_e)
        t.add(round(d, 6))
        t.add(round(-d, 6))
    return sorted(t)


def rbox(name, size, r=0.03, spacing=0.02, loc=(0, 0, 0), rot=(0, 0, 0), bulge=(0, 0, 0), tuft=None, mat=None,
         taper=None, sag=0.0):
    """Rounded box (sphere-swept edges of radius r) on a grid, for cushions and upholstered bodies.

    bulge (bx, by, bz): outward swell of each face pair at its middle.
    taper (tx, ty): top (+Z) of the box scaled in X / Y by these factors (1 = none), for raked backs and seats.
    sag: the +Z face dips by this much in the middle (sat-in seat).
    tuft: {"face": "-y", "cw": cell width, "ch": cell height, "depth": m, "u0": , "v0": offsets, "margin": m}
          diamond button tufting on that face (u = the face's first other axis, v = the second). Returns
          (object, buttons) when tuft is set: buttons = list of (position, outward normal) in local coordinates.
    """
    h = [size[0] / 2, size[1] / 2, size[2] / 2]
    r = min(r, *[x * 0.98 for x in h])
    T = [_ticks(h[a], r, spacing) for a in range(3)]
    bm = bmesh.new()
    for a in range(3):
        b, c = [x for x in range(3) if x != a]
        for s in (-1, 1):
            grid = []
            for ub in T[b]:
                row = []
                for vc in T[c]:
                    p = [0, 0, 0]
                    p[a], p[b], p[c] = s * h[a], ub, vc
                    row.append(bm.verts.new(p))
                grid.append(row)
            for i in range(len(T[b]) - 1):
                for j in range(len(T[c]) - 1):
                    bm.faces.new((grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
    inner = [x - r for x in h]
    buttons = []
    tf = None
    if tuft:
        fa = {"x": 0, "y": 1, "z": 2}[tuft["face"][-1]]
        fs = -1 if tuft["face"].startswith("-") else 1
        fb, fc = [x for x in range(3) if x != fa]
        cw, ch, dep = tuft["cw"], tuft["ch"], tuft["depth"]
        u0, v0 = tuft.get("u0", 0.0), tuft.get("v0", 0.0)
        margin = tuft.get("margin", r + cw * 0.3)
        tf = (fa, fs, fb, fc, cw, ch, dep, u0, v0, margin)
    for v in bm.verts:
        p = list(v.co)
        q = [max(-inner[i], min(inner[i], p[i])) for i in range(3)]
        d = Vector([p[i] - q[i] for i in range(3)])
        if d.length > 1e-9:
            d.normalize()
            p = [q[i] + d[i] * r for i in range(3)]
        nrm = d if d.length > 0.5 else Vector((0, 0, 1))
        base = list(p)
        for a in range(3):
            if bulge[a]:
                o1, o2 = [x for x in range(3) if x != a]
                w = max(0.0, 1 - (base[o1] / h[o1]) ** 2) * max(0.0, 1 - (base[o2] / h[o2]) ** 2)
                p[a] += bulge[a] * math.copysign(1, base[a]) * w * abs(nrm[a])
        if sag and base[2] > 0:
            w = max(0.0, 1 - (base[0] / h[0]) ** 2) * max(0.0, 1 - (base[1] / h[1]) ** 2)
            p[2] -= sag * w * max(0.0, nrm[2])
        if tf:
            fa, fs, fb, fc, cw, ch, dep, u0, v0, margin = tf
            facing = max(0.0, nrm[fa] * fs)
            if facing > 0.3:
                u, vv = base[fb] - u0, base[fc] - v0
                A, B = u / cw + vv / ch, u / cw - vv / ch
                g = (abs(math.sin(math.pi * A)) * abs(math.sin(math.pi * B))) ** 0.45
                eu = smooth((h[fb] - abs(base[fb])) / margin)
                ev = smooth((h[fc] - abs(base[fc])) / margin)
                p[fa] += fs * dep * (g - 1.0) * eu * ev * smooth((facing - 0.3) / 0.5)
        if taper:
            z01 = (base[2] + h[2]) / (2 * h[2])
            p[0] *= lerp(1.0, taper[0], z01)
            p[1] *= lerp(1.0, taper[1], z01)
        v.co = p
    if tf:
        fa, fs, fb, fc, cw, ch, dep, u0, v0, margin = tf
        imax = int(h[fb] / cw * 2) + 2
        for i in range(-imax, imax + 1):
            for j in range(-imax, imax + 1):
                u = u0 + cw * (i + j) / 2
                vv = v0 + ch * (i - j) / 2
                if abs(u) <= h[fb] - margin * 0.999 and abs(vv) <= h[fc] - margin * 0.999:
                    p = [0, 0, 0]
                    p[fb], p[fc] = u, vv
                    p[fa] = fs * (h[fa] - dep)
                    if bulge[fa]:
                        w = max(0.0, 1 - (u / h[fb]) ** 2) * max(0.0, 1 - (vv / h[fc]) ** 2)
                        p[fa] += fs * bulge[fa] * w
                    if taper:
                        z01 = (p[2] + h[2]) / (2 * h[2])
                        p[0] *= lerp(1.0, taper[0], z01)
                        p[1] *= lerp(1.0, taper[1], z01)
                    n = [0, 0, 0]
                    n[fa] = fs
                    buttons.append((Vector(p), Vector(n)))
    ob = from_bm(name, bm)
    shade(ob, 60)
    place(ob, loc, rot, mat)
    if tuft is not None:
        return ob, buttons
    return ob


def add_buttons(prefix, buttons, ob, r=0.009, mat=None, sink=0.3):
    """Small domed buttons at rbox() button positions, transformed by the cushion object's placement."""
    out = []
    M = ob.matrix_basis.copy()
    for k, (p, n) in enumerate(buttons):
        prof = [(0.0001, 0.0), (r * 0.98, 0.0), (r, r * 0.25), (r * 0.8, r * 0.55), (0.0, r * 0.65)]
        b = lathe(f"{prefix}{k}", [(a, z) for a, z in prof], seg=10, cap1=False)
        b.data.transform(Matrix.Translation((0, 0, -r * 0.65 * sink)))
        rotq = Vector((0, 0, 1)).rotation_difference(M.to_3x3() @ n)
        b.matrix_world = Matrix.Translation(M @ p) @ rotq.to_matrix().to_4x4()
        if mat is not None:
            assign(b, mat)
        out.append(b)
    return out


# ================================================================== sweeps and mouldings
def superellipse2d(hw, hh, n=2.0, rot=0.0, seg=16):
    pts = []
    for k in range(seg):
        a = TAU * (k + 0.5) / seg
        c, s = math.cos(a), math.sin(a)
        u = hw * math.copysign(abs(c) ** (2 / n), c)
        v = hh * math.copysign(abs(s) ** (2 / n), s)
        cr, sr = math.cos(rot), math.sin(rot)
        pts.append((u * cr - v * sr, u * sr + v * cr))
    return pts


def sweep(name, path, sections, cap0=True, cap1=True, closed=False, up=None, mat=None, angle=50):
    """path: centreline points; sections: one 2D point list per path point (same count), placed in a
    parallel-transport frame (N, B) around the tangent; `up` seeds N (default world Z, or X for vertical paths)."""
    bm = bmesh.new()
    path = [Vector(p) for p in path]
    m = len(path)
    tang = []
    for i in range(m):
        if closed:
            a, b = path[(i - 1) % m], path[(i + 1) % m]
        else:
            a, b = path[max(i - 1, 0)], path[min(i + 1, m - 1)]
        tang.append((b - a).normalized())
    if up is None:
        up = Vector((0, 0, 1)) if abs(tang[0].z) < 0.9 else Vector((1, 0, 0))
    up = Vector(up)
    n = (up - tang[0] * up.dot(tang[0])).normalized()
    rows = []
    for i, (p, t) in enumerate(zip(path, tang)):
        if i:
            n = (n - t * n.dot(t)).normalized()
        b = t.cross(n)
        rows.append([bm.verts.new(p + n * v + b * u) for (u, v) in sections[i]])
    k = len(rows[0])
    pairs = list(zip(rows, rows[1:])) + ([(rows[-1], rows[0])] if closed else [])
    for r0, r1 in pairs:
        for j in range(k):
            bm.faces.new((r0[j], r0[(j + 1) % k], r1[(j + 1) % k], r1[j]))
    if not closed:
        if cap0:
            bm.faces.new(list(reversed(rows[0])))
        if cap1:
            bm.faces.new(rows[-1])
    ob = from_bm(name, bm)
    shade(ob, angle)
    if mat is not None:
        assign(ob, mat)
    return ob


def tube(name, path, r, seg=10, closed=False, mat=None, r_fn=None):
    secs = []
    for i in range(len(path)):
        rr = r_fn(i / max(1, len(path) - 1)) if r_fn else r
        secs.append(superellipse2d(rr, rr, 2.0, 0.0, seg))
    return sweep(name, path, secs, closed=closed, mat=mat)


def bezier(p0, p1, p2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append(tuple(u ** 3 * a + 3 * u * u * t * b + 3 * u * t * t * c + t ** 3 * d for a, b, c, d in zip(p0, p1, p2, p3)))
    return out


def moulding(name, prof, x0, x1, mitre0=0, mitre1=0, mat=None, angle=40):
    """Straight run of a 2D profile along X from x0 to x1. prof: closed list of (y, z) with y <= 0 pointing out of
    the wall (front -Y). mitre0/1 = +1 / -1 / 0: at that end, x shifts by -y (+1) or +y (-1) so two runs meet at 45
    degrees (+1 when the return goes back toward +Y on that side). Rotate the object for side runs."""
    bm = bmesh.new()
    r0 = [bm.verts.new((x0 + mitre0 * y, y, z)) for (y, z) in prof]
    r1 = [bm.verts.new((x1 - mitre1 * y, y, z)) for (y, z) in prof]
    k = len(prof)
    for j in range(k):
        bm.faces.new((r0[j], r0[(j + 1) % k], r1[(j + 1) % k], r1[j]))
    bm.faces.new(list(reversed(r0)))
    bm.faces.new(r1)
    ob = from_bm(name, bm)
    shade(ob, angle)
    if mat is not None:
        assign(ob, mat)
    return ob


def ogee(depth, height, n=10, base=0.0):
    """A closed cornice profile (y <= 0 out from the wall, z up): flat back, ogee (cyma) face from the bottom
    (depth out ~0.25) to the top (full depth), flat top. Returns (y, z) list."""
    pts = [(0.0, base)]
    for k in range(n + 1):
        t = k / n
        y = -depth * (0.25 + 0.75 * (0.5 - 0.5 * math.cos(math.pi * t)) + 0.08 * math.sin(TAU * t))
        pts.append((y, base + height * t * 0.88))
    pts.append((-depth, base + height))
    pts.append((0.0, base + height))
    return pts


def fielded_panel(name, w, h, t, raise_w=0.03, raise_h=0.004, loc=(0, 0, 0), rot=(0, 0, 0), mat=None, face="-y"):
    """A raised and fielded panel lying in XZ (front -Y): a slab of thickness t whose front has a bevelled field
    standing raise_h proud, the bevel raise_w wide."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * w, v.co.y * t, v.co.z * h))
    bm.faces.ensure_lookup_table()
    front = [f for f in bm.faces if f.normal.y < -0.9][0]
    res = bmesh.ops.inset_individual(bm, faces=[front], thickness=raise_w, depth=0.0)
    for v in front.verts:
        v.co.y -= raise_h
    bm.normal_update()
    ob = from_bm(name, bm)
    place(ob, loc, rot, mat)
    bevel(ob, 0.0012, 1, 30)
    shade(ob, 30)
    return ob


def frame_panel(prefix, w, h, t, loc, mat_v, mat_h, mat_panel=None, cols=1, rows=1, stile=0.06, rail=0.07,
                rail_bottom=None, mid=None, row_h=None, raise_w=0.03, bead=True):
    """Framed-and-panelled leaf in XZ (front -Y) centred at loc (x, y, z of the leaf's bottom centre): stiles, top /
    bottom / intermediate rails, muntins between columns, a fielded panel in each opening and a small bead moulding
    round each opening. row_h: list of opening heights (bottom -> top) or equal split. Returns the parts."""
    x0, y0, z0 = loc
    rb = rail_bottom or rail * 1.4
    mid = mid or rail
    parts = []
    parts.append(box(f"{prefix}_stl", (stile, t, h), (x0 - w / 2 + stile / 2, y0, z0 + h / 2), bev=0.002, mat=mat_v))
    parts.append(box(f"{prefix}_str", (stile, t, h), (x0 + w / 2 - stile / 2, y0, z0 + h / 2), bev=0.002, mat=mat_v))
    inner_w = w - 2 * stile
    open_h_total = h - rb - rail - (rows - 1) * mid
    hs = row_h or [open_h_total / rows] * rows
    s = sum(hs)
    hs = [x * open_h_total / s for x in hs]
    z = z0
    parts.append(box(f"{prefix}_rb", (inner_w, t, rb), (x0, y0, z + rb / 2), bev=0.002, mat=mat_h))
    z += rb
    col_w = (inner_w - (cols - 1) * stile * 0.9) / cols
    for r, oh in enumerate(hs):
        for c in range(cols):
            cx = x0 - inner_w / 2 + col_w / 2 + c * (col_w + stile * 0.9)
            parts.append(fielded_panel(f"{prefix}_p{r}{c}", col_w + 0.012, oh + 0.012, t * 0.55,
                                       raise_w=min(raise_w, col_w * 0.2), loc=(cx, y0 + t * 0.1, z + oh / 2),
                                       mat=mat_panel or mat_v))
            if bead:
                for (bx, bz, bw, bh) in ((cx, z + 0.004, col_w, 0.008), (cx, z + oh - 0.004, col_w, 0.008),
                                         (cx - col_w / 2 + 0.004, z + oh / 2, 0.008, oh),
                                         (cx + col_w / 2 - 0.004, z + oh / 2, 0.008, oh)):
                    parts.append(box(f"{prefix}_bd{r}{c}{len(parts)}", (bw, 0.008, bh), (bx, y0 - t / 2 + 0.006, bz),
                                     bev=0.0035, seg=2, mat=mat_h))
            if c < cols - 1:
                parts.append(box(f"{prefix}_mu{r}{c}", (stile * 0.9, t, oh), (cx + col_w / 2 + stile * 0.45, y0, z + oh / 2),
                                 bev=0.002, mat=mat_v))
        z += oh
        rh = mid if r < rows - 1 else rail
        parts.append(box(f"{prefix}_r{r}", (inner_w, t, rh), (x0, y0, z + rh / 2), bev=0.002, mat=mat_h))
        z += rh
    return parts


def cabriole(name, height, knee=0.035, top=0.026, ankle=0.013, foot=0.024, out=0.028, mat=None, seg=12):
    """Cabriole leg along local +Z from the floor (pad foot) to `height`; the knee swells toward local -Y (front).
    Rounded-square section easing to round at the ankle."""
    zs = [height * k / 48 for k in range(49)]
    path, secs = [], []
    for z in zs:
        t = z / height
        # S curve: foot forward, ankle back, knee forward, top back to centre
        y = -out * (0.55 * math.exp(-((t - 0.04) / 0.06) ** 2) - 0.35 * math.exp(-((t - 0.22) / 0.12) ** 2)
                    + 0.9 * math.exp(-((t - 0.72) / 0.13) ** 2))
        path.append((0, y, z))
        if t < 0.06:
            r = lerp(foot, foot * 0.85, smooth(t / 0.06)) if t > 0.015 else foot * (0.85 + 0.15 * t / 0.015)
        elif t < 0.22:
            r = lerp(foot * 0.85, ankle, smooth((t - 0.06) / 0.16))
        elif t < 0.72:
            r = lerp(ankle, knee, smooth((t - 0.22) / 0.5))
        else:
            r = lerp(knee, top, smooth((t - 0.72) / 0.28))
        n = lerp(2.0, 3.2, smooth((t - 0.25) / 0.4))
        secs.append(superellipse2d(r, r, n, math.pi / 4, seg))
    path[0] = (0, path[0][1], 0.0)
    ob = sweep(name, path, secs, up=(0, -1, 0), mat=mat, angle=60)
    return ob


# ================================================================== material node kit
_CACHE = {}


class NK:
    """Small node-building kit on one material's tree."""

    def __init__(self, name):
        self.mat = bpy.data.materials.new(name)
        self.mat.use_nodes = True
        self.nt = self.mat.node_tree
        self.N, self.L = self.nt.nodes, self.nt.links
        self.pr = self.N["Principled BSDF"]
        tc = self.N.new("ShaderNodeTexCoord")
        self.obj = tc.outputs["Object"]
        self.geo = self.N.new("ShaderNodeNewGeometry")

    def node(self, t, **kw):
        n = self.N.new(t)
        for k, v in kw.items():
            n.inputs[k].default_value = v
        return n

    def _in(self, sock, x):
        if isinstance(x, (int, float)):
            sock.default_value = x
        elif isinstance(x, tuple):
            n = len(sock.default_value)
            sock.default_value = tuple(x) + (1.0,) * (n - len(x)) if len(x) < n else x[:n]
        else:
            self.L.new(x, sock)

    def m(self, op, a, b=None, clamp=False):
        n = self.N.new("ShaderNodeMath")
        n.operation, n.use_clamp = op, clamp
        for i, x in enumerate((a, b)):
            if x is not None:
                self._in(n.inputs[i], x)
        return n.outputs[0]

    def mr(self, x, a, b, c=0.0, d=1.0, clamp=True):
        n = self.node("ShaderNodeMapRange")
        n.clamp = clamp
        n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = a, b
        n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = c, d
        self.L.new(x, n.inputs["Value"])
        return n.outputs["Result"]

    def mix(self, f, a, b):
        n = self.N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        self._in(n.inputs[6], a)
        self._in(n.inputs[7], b)
        self._in(n.inputs["Factor"], f)
        return n.outputs[2]

    def mixf(self, f, a, b):
        n = self.N.new("ShaderNodeMix")
        self._in(n.inputs[2], a)
        self._in(n.inputs[3], b)
        self._in(n.inputs["Factor"], f)
        return n.outputs[0]

    def mul_col(self, a, f):
        n = self.N.new("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = "MULTIPLY"
        n.inputs["Factor"].default_value = 1.0
        self._in(n.inputs[6], a)
        self._in(n.inputs[7], f)
        return n.outputs[2]

    def vec(self, v, scale=(1, 1, 1), rot=(0, 0, 0), loc=(0, 0, 0)):
        mp = self.N.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = scale
        mp.inputs["Rotation"].default_value = rot
        mp.inputs["Location"].default_value = loc
        self.L.new(v, mp.inputs["Vector"])
        return mp.outputs["Vector"]

    def noise(self, vec, scale, detail=4.0, rough=0.5, stretch=None, dist=0.0):
        if stretch:
            vec = self.vec(vec, stretch)
        n = self.node("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough, Distortion=dist)
        self.L.new(vec, n.inputs["Vector"])
        return n.outputs["Fac"]

    def voronoi(self, vec, scale, feature="F1", out="Distance", rand=1.0):
        n = self.node("ShaderNodeTexVoronoi", Scale=scale, Randomness=rand)
        n.feature = feature
        self.L.new(vec, n.inputs["Vector"])
        return n.outputs[out]

    def ramp(self, f, stops):
        n = self.N.new("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.elements[0].position, cr.elements[0].color = stops[0][0], (*stops[0][1], 1)
        cr.elements[1].position, cr.elements[1].color = stops[-1][0], (*stops[-1][1], 1)
        for p, c in stops[1:-1]:
            e = cr.elements.new(p)
            e.color = (*c, 1)
        self.L.new(f, n.inputs["Fac"])
        return n.outputs["Color"]

    def edge(self, radius=0.002, lo=0.995, hi=0.95):
        bev = self.node("ShaderNodeBevel", Radius=radius)
        bev.samples = 8
        dp = self.N.new("ShaderNodeVectorMath")
        dp.operation = "DOT_PRODUCT"
        self.L.new(bev.outputs["Normal"], dp.inputs[0])
        self.L.new(self.geo.outputs["Normal"], dp.inputs[1])
        return self.mr(dp.outputs["Value"], lo, hi)

    def cavity(self, dist=0.02, lo=0.9, hi=0.4):
        ao = self.N.new("ShaderNodeAmbientOcclusion")
        ao.samples = 8
        ao.inputs["Distance"].default_value = dist
        return self.mr(ao.outputs["AO"], lo, hi)

    def pcol(self, name="pcol"):
        a = self.N.new("ShaderNodeAttribute")
        a.attribute_type = "GEOMETRY"
        a.attribute_name = name
        return a.outputs["Color"]

    def bump(self, h, dist=0.0005, strength=0.5, normal=None):
        b = self.node("ShaderNodeBump", Distance=dist, Strength=strength)
        self.L.new(h, b.inputs["Height"])
        if normal is not None:
            self.L.new(normal, b.inputs["Normal"])
        return b.outputs["Normal"]

    def out(self, base, rough, metal=0.0, normal=None, emit=None, emit_strength=0.0):
        self._in(self.pr.inputs["Base Color"], base)
        self._in(self.pr.inputs["Roughness"], rough)
        self._in(self.pr.inputs["Metallic"], metal)
        if normal is not None:
            self.L.new(normal, self.pr.inputs["Normal"])
        if emit is not None:
            self._in(self.pr.inputs["Emission Color"], emit)
            self.pr.inputs["Emission Strength"].default_value = emit_strength
        return self.mat


def _cached(key, fn):
    m = _CACHE.get(key)
    if m is None or m.name not in bpy.data.materials:
        m = fn()
        _CACHE[key] = m
    return m


# ------------------------------------------------------------------ wood
WOODS = {
    #            dark (linear)            light                    figure  ring_scale  pores
    "mahogany": ((0.042, 0.012, 0.006), (0.135, 0.046, 0.021), 0.55, 55.0, 0.6),
    "walnut": ((0.045, 0.024, 0.013), (0.160, 0.088, 0.048), 0.65, 45.0, 0.5),
    "oak": ((0.150, 0.088, 0.040), (0.360, 0.230, 0.115), 0.75, 35.0, 1.0),
    "dark_oak": ((0.055, 0.030, 0.014), (0.150, 0.088, 0.044), 0.75, 35.0, 1.0),
    "pine": ((0.300, 0.165, 0.070), (0.560, 0.380, 0.200), 0.9, 22.0, 0.15),
    "rosewood": ((0.035, 0.010, 0.006), (0.120, 0.040, 0.020), 0.8, 60.0, 0.4),
    "ebonised": ((0.010, 0.008, 0.007), (0.030, 0.022, 0.018), 0.4, 55.0, 0.3),
    "satinwood": ((0.330, 0.200, 0.080), (0.620, 0.430, 0.200), 0.5, 70.0, 0.2),
}


def wood(species="mahogany", axis="Z", wear=0.5, gloss=0.5, grime=0.5, offset=0.0, name=None):
    """Polished/varnished hardwood. Grain (rings) run along the part's object `axis`. wear: edge wear showing
    paler timber; gloss: varnish (0 dull, 1 French polish); grime: dark dirt in recesses."""
    key = ("wood", species, axis, wear, gloss, grime, offset)

    def make():
        k = NK(name or f"wood_{species}_{axis}")
        dark, light, fig, ring_sc, pores = WOODS[species]
        rot = {"Z": (0, 0, 0), "X": (0, math.pi / 2, 0), "Y": (math.pi / 2, 0, 0)}[axis]
        v = k.vec(k.obj, rot=rot, loc=(0.37 + offset, -0.21 - offset, 0))
        # rotate first so the grain is along local Z of v
        w = k.node("ShaderNodeTexWave", Scale=ring_sc, Distortion=1.5, Detail=2.0)
        w.wave_type, w.rings_direction = "RINGS", "Z"
        w.inputs["Detail Scale"].default_value = 1.2
        # distort coordinates with low-frequency noise so rings wander
        dn = k.node("ShaderNodeTexNoise", Scale=1.2, Detail=1.0)
        k.L.new(v, dn.inputs["Vector"])
        vm = k.N.new("ShaderNodeVectorMath")
        vm.operation = "MULTIPLY_ADD"
        k.L.new(dn.outputs["Color"], vm.inputs[0])
        vm.inputs[1].default_value = (0.02, 0.02, 0.0)
        k.L.new(v, vm.inputs[2])
        k.L.new(vm.outputs[0], w.inputs["Vector"])
        rings = w.outputs["Fac"]
        streak = k.noise(v, 1.0, 5.0, 0.6, stretch=(60.0, 60.0, 2.5))
        pore = k.noise(v, 1.0, 2.0, 0.5, stretch=(900.0, 900.0, 25.0))
        blot = k.noise(v, 6.0, 3.0, 0.5)
        g = k.m("ADD", k.m("MULTIPLY", rings, 0.55 * fig), k.m("MULTIPLY", streak, 0.6))
        g = k.m("ADD", g, k.m("MULTIPLY", blot, 0.3))
        col = k.ramp(k.mr(g, 0.35, 1.25), [(0.0, dark), (0.55, tuple(lerp(dark[i], light[i], 0.55) for i in range(3))),
                                         (1.0, light)])
        pdark = k.mr(pore, 0.55, 0.75, 0.0, pores)
        col = k.mix(k.m("MULTIPLY", pdark, 0.45), col, tuple(x * 0.4 for x in dark))
        e = k.edge(0.0025)
        cv = k.cavity(0.03)
        wn = k.noise(k.obj, 30.0, 4.0, 0.6)
        wearm = k.m("MULTIPLY", e, k.mr(wn, 0.35, 0.65, 0.0, 1.0))
        wearm = k.m("MULTIPLY", wearm, wear)
        pale = tuple(min(1.0, x * 1.7 + 0.03) for x in light)
        col = k.mix(wearm, col, pale)
        col = k.mix(k.m("MULTIPLY", cv, 0.7 * grime), col, tuple(x * 0.25 for x in dark))
        rough = k.mr(blot, 0.3, 0.7, lerp(0.62, 0.22, gloss), lerp(0.75, 0.38, gloss))
        rough = k.mixf(wearm, rough, 0.7)
        rough = k.mixf(k.m("MULTIPLY", cv, grime), rough, 0.8)
        h = k.m("ADD", k.m("MULTIPLY", pore, -0.6), k.m("MULTIPLY", rings, 0.15))
        nrm = k.bump(h, 0.0003, 0.35)
        return k.out(col, rough, 0.0, nrm)

    return _cached(key, make)


def painted_wood(colour, wear=0.4, gloss=0.4, name=None):
    """Painted joinery (doors, window frames, skirting): paint over wood, chipped on edges showing the timber."""
    key = ("pwood", tuple(colour), wear, gloss)

    def make():
        k = NK(name or "painted_wood")
        n = k.noise(k.obj, 9.0, 4.0, 0.55)
        f = k.noise(k.obj, 200.0, 3.0, 0.5)
        brush = k.noise(k.obj, 1.0, 3.0, 0.5, stretch=(4.0, 4.0, 300.0))
        col = k.mix(k.mr(n, 0.3, 0.7, 0.0, 0.35), colour, tuple(x * 0.82 for x in colour))
        e = k.edge(0.003)
        cv = k.cavity(0.04)
        chip = k.m("MULTIPLY", e, k.mr(f, 0.55, 0.62))
        chip = k.m("MULTIPLY", chip, wear * 1.5)
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, tuple(x * 0.45 for x in colour))
        col = k.mix(k.m("MULTIPLY", e, 0.3 * wear), col, tuple(min(1, x * 1.15 + 0.02) for x in colour))
        col = k.mix(chip, col, (0.16, 0.10, 0.05))
        rough = k.mr(n, 0.3, 0.7, lerp(0.7, 0.3, gloss), lerp(0.8, 0.45, gloss))
        rough = k.mixf(chip, rough, 0.75)
        h = k.m("ADD", k.m("MULTIPLY", brush, 0.3), k.m("MULTIPLY", chip, -0.6))
        return k.out(col, rough, 0.0, k.bump(h, 0.0003, 0.3))

    return _cached(key, make)


# ------------------------------------------------------------------ fabrics and leather
def fabric(kind="leather", colour=(0.10, 0.035, 0.018), wear=0.5, figure=None, name=None, scale=1.0):
    """Upholstery and textiles. kind: leather | velvet | damask | linen | wool | rope | cotton.
    figure: second colour for damask/woven figures (else a darker shade of colour)."""
    key = ("fab", kind, tuple(colour), wear, tuple(figure) if figure else None, scale)

    def make():
        k = NK(name or f"fabric_{kind}")
        c = colour
        fig = figure or tuple(x * 0.7 for x in c)
        big = k.noise(k.obj, 4.0 * scale, 3.0, 0.5)
        mid = k.noise(k.obj, 28.0 * scale, 4.0, 0.55)
        fine = k.noise(k.obj, 260.0 * scale, 3.0, 0.6)
        e = k.edge(0.006, 0.99, 0.88)
        cv = k.cavity(0.035, 0.92, 0.45)
        if kind == "leather":
            cracks = k.voronoi(k.obj, 140.0 * scale, "DISTANCE_TO_EDGE")
            crk = k.mr(cracks, 0.0, 0.06, 1.0, 0.0)
            col = k.mix(k.mr(big, 0.3, 0.7), tuple(x * 0.75 for x in c), tuple(min(1, x * 1.25) for x in c))
            col = k.mix(k.mr(mid, 0.4, 0.7, 0, 0.5), col, tuple(x * 0.6 for x in c))
            shine = k.m("MULTIPLY", e, k.mr(mid, 0.3, 0.6, 0.3, 1.0))
            shine = k.m("MULTIPLY", shine, wear)
            col = k.mix(shine, col, tuple(min(1, x * 1.9 + 0.025) for x in c))
            col = k.mix(k.m("MULTIPLY", crk, 0.35 * wear), col, tuple(min(1, x * 1.6 + 0.04) for x in c))
            col = k.mix(k.m("MULTIPLY", cv, 0.75), col, tuple(x * 0.3 for x in c))
            rough = k.mr(mid, 0.3, 0.7, 0.38, 0.55)
            rough = k.mixf(shine, rough, 0.25)
            rough = k.mixf(cv, rough, 0.7)
            h = k.m("ADD", k.m("MULTIPLY", crk, -0.5), k.m("MULTIPLY", fine, 0.4))
            nrm = k.bump(h, 0.0003, 0.4)
            return k.out(col, rough, 0.0, nrm)
        if kind in ("velvet", "wool"):
            col = k.mix(k.mr(big, 0.25, 0.75), tuple(x * 0.7 for x in c), tuple(min(1, x * 1.2) for x in c))
            # velvet: pile darkens facing, lighter at grazing (sheen) -> lighter on edges, crushed patches
            crush = k.mr(mid, 0.45, 0.7, 0.0, 1.0)
            col = k.mix(k.m("MULTIPLY", crush, 0.4), col, tuple(min(1, x * 1.5 + 0.02) for x in c))
            col = k.mix(k.m("MULTIPLY", e, 0.6 * wear), col, tuple(min(1, x * 1.7 + 0.03) for x in c))
            col = k.mix(k.m("MULTIPLY", cv, 0.8), col, tuple(x * 0.3 for x in c))
            rough = k.mr(mid, 0.3, 0.7, 0.85, 0.95)
            nrm = k.bump(k.m("ADD", fine, k.m("MULTIPLY", crush, 0.3)), 0.0004, 0.35)
            return k.out(col, rough, 0.0, nrm)
        if kind == "damask":
            # faint woven figure: lobed repeat from two crossed sine fields + a voronoi rosette
            v = k.vec(k.obj, (14.0 * scale, 14.0 * scale, 14.0 * scale))
            # warp the lattice with noise so the motif reads as scrolling foliage, not a grid of dots
            wn = k.node("ShaderNodeTexNoise", Scale=1.3, Detail=3.0, Roughness=0.5)
            k.L.new(v, wn.inputs["Vector"])
            vw = k.N.new("ShaderNodeVectorMath")
            vw.operation = "MULTIPLY_ADD"
            k.L.new(wn.outputs["Color"], vw.inputs[0])
            vw.inputs[1].default_value = (1.6, 1.6, 1.6)
            k.L.new(v, vw.inputs[2])
            v = vw.outputs[0]
            sx = k.N.new("ShaderNodeSeparateXYZ")
            k.L.new(v, sx.inputs[0])
            a = k.m("SINE", k.m("ADD", sx.outputs["X"], sx.outputs["Z"]))
            b = k.m("SINE", k.m("SUBTRACT", sx.outputs["X"], sx.outputs["Z"]))
            c2 = k.m("SINE", k.m("MULTIPLY", sx.outputs["Y"], 2.0))
            pat = k.m("MULTIPLY", k.m("ADD", a, b), k.m("ADD", 1.2, k.m("MULTIPLY", c2, 0.4)))
            pat = k.m("ADD", pat, k.m("MULTIPLY", k.noise(k.obj, 60.0 * scale, 2.0, 0.5), 0.6))
            figm = k.mr(pat, 0.2, 1.1)
            col = k.mix(k.m("MULTIPLY", figm, 0.55), c, fig)
            col = k.mix(k.mr(big, 0.3, 0.7, 0, 0.35), col, tuple(x * 0.75 for x in c))
            col = k.mix(k.m("MULTIPLY", e, 0.5 * wear), col, tuple(min(1, x * 1.35 + 0.02) for x in c))
            col = k.mix(k.m("MULTIPLY", cv, 0.75), col, tuple(x * 0.35 for x in c))
            rough = k.mixf(figm, 0.82, 0.62)
            nrm = k.bump(k.m("ADD", k.m("MULTIPLY", figm, 0.4), fine), 0.0003, 0.3)
            return k.out(col, rough, 0.0, nrm)
        if kind == "rope":
            v = k.obj
            st = k.noise(v, 1.0, 2.0, 0.5, stretch=(300.0, 300.0, 300.0))
            col = k.mix(k.mr(st, 0.3, 0.7), tuple(x * 0.7 for x in c), c)
            col = k.mix(k.m("MULTIPLY", cv, 0.8), col, tuple(x * 0.35 for x in c))
            return k.out(col, 0.85, 0.0, k.bump(st, 0.0006, 0.6))
        # linen / cotton: pale, soft folds dirt in creases
        col = k.mix(k.mr(big, 0.3, 0.7), tuple(x * 0.9 for x in c), c)
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, tuple(x * 0.55 for x in c))
        col = k.mix(k.mr(mid, 0.55, 0.75, 0.0, 0.25 * wear), col, tuple(x * 0.8 for x in c))
        return k.out(col, 0.9, 0.0, k.bump(fine, 0.0003, 0.25))

    return _cached(key, make)


# ------------------------------------------------------------------ metals
METALS = {
    #          base (linear)            tarnish                 rough lo/hi
    "brass": ((0.62, 0.42, 0.16), (0.13, 0.09, 0.04), 0.22, 0.45),
    "copper": ((0.70, 0.33, 0.20), (0.12, 0.07, 0.05), 0.25, 0.5),
    "nickel": ((0.55, 0.53, 0.50), (0.18, 0.17, 0.16), 0.18, 0.35),
    "steel": ((0.42, 0.42, 0.42), (0.10, 0.10, 0.11), 0.25, 0.45),
    "blued": ((0.10, 0.11, 0.14), (0.035, 0.035, 0.04), 0.22, 0.4),
    "gold": ((0.85, 0.62, 0.25), (0.25, 0.17, 0.06), 0.2, 0.35),
    "iron": ((0.040, 0.038, 0.036), (0.020, 0.019, 0.018), 0.45, 0.75),
    "cast_iron": ((0.030, 0.029, 0.028), (0.015, 0.014, 0.014), 0.5, 0.8),
}


def metal(kind="brass", tarnish=0.5, soot=0.0, rust=0.0, polish=0.5, name=None, soot_z=None):
    """Metals. tarnish: darkening in recesses and broad patches; polish: bright wear on edges; soot: black soot
    (soot_z = (z_lo, z_hi) object heights: soot fades from z_lo up to z_hi); rust: red-brown in pits (iron)."""
    key = ("metal", kind, tarnish, soot, rust, polish, soot_z)

    def make():
        k = NK(name or f"metal_{kind}")
        base, tarn, rlo, rhi = METALS[kind]
        ferrous = kind in ("iron", "cast_iron", "steel", "blued")
        big = k.noise(k.obj, 7.0, 3.0, 0.5)
        mid = k.noise(k.obj, 45.0, 4.0, 0.55)
        fine = k.noise(k.obj, 400.0, 3.0, 0.6)
        e = k.edge(0.0015)
        cv = k.cavity(0.015, 0.9, 0.45)
        tm = k.m("ADD", k.m("MULTIPLY", cv, 0.9), k.mr(big, 0.4, 0.75, 0.0, 0.6))
        tm = k.m("MULTIPLY", tm, tarnish, clamp=True)
        col = k.mix(k.mr(mid, 0.3, 0.7, 0.0, 0.3), base, tuple(x * 0.8 for x in base))
        col = k.mix(tm, col, tarn)
        pol = k.m("MULTIPLY", e, k.mr(mid, 0.3, 0.6, 0.2, 1.0))
        pol = k.m("MULTIPLY", pol, polish, clamp=True)
        if kind in ("iron", "cast_iron"):
            bright = (0.20, 0.195, 0.19)
        else:
            bright = tuple(min(1, x * 1.2 + 0.05) for x in base)
        col = k.mix(pol, col, bright)
        rough = k.mr(mid, 0.3, 0.7, rlo, rhi)
        rough = k.mixf(tm, rough, min(0.95, rhi + 0.2))
        rough = k.mixf(pol, rough, rlo * 0.6)
        met = 1.0 if not ferrous or kind == "steel" else 0.75
        metm = k.mixf(tm, met, 0.5 if not ferrous else 0.35)
        metm = k.mixf(pol, metm, 1.0)
        if rust > 0:
            rm = k.m("MULTIPLY", k.m("ADD", k.m("MULTIPLY", cv, 0.8), k.mr(mid, 0.6, 0.75, 0, 0.6)), rust, clamp=True)
            rm = k.m("MULTIPLY", rm, k.m("SUBTRACT", 1.0, pol), clamp=True)
            col = k.mix(rm, col, k.mix(fine, (0.12, 0.05, 0.02, 1), (0.22, 0.10, 0.04, 1)))
            rough = k.mixf(rm, rough, 0.9)
            metm = k.mixf(rm, metm, 0.0)
        if soot > 0:
            sm = k.mr(big, 0.3, 0.6, 0.6, 1.0)
            if soot_z:
                sx = k.N.new("ShaderNodeSeparateXYZ")
                k.L.new(k.geo.outputs["Position"], sx.inputs[0])
                sm = k.m("MULTIPLY", sm, k.mr(sx.outputs["Z"], soot_z[1], soot_z[0]))
            sm = k.m("MULTIPLY", sm, k.m("SUBTRACT", 1.0, k.m("MULTIPLY", pol, 0.8)), clamp=True)
            sm = k.m("MULTIPLY", sm, soot, clamp=True)
            col = k.mix(sm, col, (0.008, 0.0075, 0.007, 1))
            rough = k.mixf(sm, rough, 0.92)
            metm = k.mixf(sm, metm, 0.0)
        h = k.m("ADD", k.m("MULTIPLY", fine, 0.3), k.m("MULTIPLY", mid, 0.2))
        if kind == "cast_iron":
            h = k.m("ADD", h, k.m("MULTIPLY", k.noise(k.obj, 900.0, 2.0, 0.5), 0.5))
        nrm = k.bump(h, 0.0002, 0.35)
        return k.out(col, rough, metm, nrm)

    return _cached(key, make)


def glass(tint=(0.92, 0.95, 0.93), rough=0.03, ior=1.5, name="glass"):
    key = ("glass", tuple(tint), rough, ior, name)

    def make():
        m = bpy.data.materials.new(name)
        m.use_nodes = True
        pr = m.node_tree.nodes["Principled BSDF"]
        pr.inputs["Base Color"].default_value = (*tint, 1)
        pr.inputs["Roughness"].default_value = rough
        pr.inputs["IOR"].default_value = ior
        pr.inputs["Transmission Weight"].default_value = 1.0
        m["asset_glass"] = True
        return m

    return _cached(key, make)


def ceramic(colour=(0.85, 0.84, 0.80), glaze=0.8, pattern=None, craze=0.3, name=None):
    """Glazed earthenware/porcelain. pattern: (colour, band z ranges list) painted bands in object Z."""
    key = ("cer", tuple(colour), glaze, repr(pattern), craze)

    def make():
        k = NK(name or "ceramic")
        big = k.noise(k.obj, 10.0, 3.0, 0.5)
        cr = k.voronoi(k.obj, 90.0, "DISTANCE_TO_EDGE")
        crk = k.mr(cr, 0.0, 0.02, craze, 0.0)
        col = k.mix(k.mr(big, 0.3, 0.7, 0, 0.3), colour, tuple(x * 0.9 for x in colour))
        if pattern:
            pc, bands = pattern
            sx = k.N.new("ShaderNodeSeparateXYZ")
            k.L.new(k.obj, sx.inputs[0])
            z = sx.outputs["Z"]
            msk = None
            for (a, b) in bands:
                mb = k.m("MULTIPLY", k.mr(z, a - 0.0005, a + 0.0005), k.mr(z, b + 0.0005, b - 0.0005))
                msk = mb if msk is None else k.m("MAXIMUM", msk, mb)
            col = k.mix(msk, col, pc)
        cv = k.cavity(0.01)
        col = k.mix(k.m("MULTIPLY", crk, 0.5), col, tuple(x * 0.55 for x in colour))
        col = k.mix(k.m("MULTIPLY", cv, 0.4), col, tuple(x * 0.6 for x in colour))
        rough = lerp(0.5, 0.06, glaze)
        return k.out(col, rough, 0.0, k.bump(k.m("MULTIPLY", crk, -1.0), 0.0001, 0.2))

    return _cached(key, make)


def paper(colour=(0.70, 0.64, 0.50), name="paper"):
    key = ("paper", tuple(colour))

    def make():
        k = NK(name)
        n = k.noise(k.obj, 30.0, 4.0, 0.5)
        cv = k.cavity(0.01)
        col = k.mix(k.mr(n, 0.3, 0.7, 0, 0.4), colour, tuple(x * 0.8 for x in colour))
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, tuple(x * 0.4 for x in colour))
        return k.out(col, 0.8, 0.0)

    return _cached(key, make)


def emissive(colour=(1.0, 0.62, 0.25), strength=8.0, base=None, name="emissive"):
    key = ("emit", tuple(colour), strength, base, name)

    def make():
        k = NK(name)
        return k.out(base or colour, 0.5, 0.0, None, (*colour, 1), strength)

    return _cached(key, make)


def plaster(colour=(0.62, 0.57, 0.46), name="plaster"):
    key = ("plaster", tuple(colour))

    def make():
        k = NK(name)
        n = k.noise(k.obj, 8.0, 5.0, 0.6)
        f = k.noise(k.obj, 120.0, 3.0, 0.5)
        col = k.mix(k.mr(n, 0.3, 0.7, 0, 0.5), colour, tuple(x * 0.75 for x in colour))
        cv = k.cavity(0.05)
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, tuple(x * 0.4 for x in colour))
        return k.out(col, 0.9, 0.0, k.bump(f, 0.0004, 0.3))

    return _cached(key, make)


def stone(colour=(0.30, 0.28, 0.25), vein=None, polish=0.3, name="stone"):
    """Marble / slate / stone. vein: colour of veining (marble) or None (plain grain)."""
    key = ("stone", tuple(colour), tuple(vein) if vein else None, polish)

    def make():
        k = NK(name)
        n = k.noise(k.obj, 3.0, 6.0, 0.6, dist=2.0)
        f = k.noise(k.obj, 80.0, 4.0, 0.6)
        col = k.mix(k.mr(f, 0.3, 0.7, 0, 0.4), colour, tuple(x * 0.8 for x in colour))
        if vein:
            vw = k.node("ShaderNodeTexWave", Scale=3.0, Distortion=12.0, Detail=6.0)
            k.L.new(k.obj, vw.inputs["Vector"])
            vm = k.mr(vw.outputs["Fac"], 0.0, 0.08, 1.0, 0.0)
            col = k.mix(k.m("MULTIPLY", vm, 0.8), col, vein)
        col = k.mix(k.mr(n, 0.5, 0.8, 0, 0.3), col, tuple(x * 0.6 for x in colour))
        e = k.edge(0.003)
        cv = k.cavity(0.03)
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, tuple(x * 0.4 for x in colour))
        col = k.mix(k.m("MULTIPLY", e, 0.3), col, tuple(min(1, x * 1.3) for x in colour))
        rough = k.mr(f, 0.3, 0.7, lerp(0.75, 0.15, polish), lerp(0.85, 0.3, polish))
        return k.out(col, rough, 0.0, k.bump(f, 0.0003, 0.3))

    return _cached(key, make)


# ================================================================== misc
def rng(seed):
    return random.Random(seed)


def book_cloth(name="book_cloth"):
    """Book covers coloured per book by the "pcol" attribute (set_color), with gilt bands and a title panel on the
    spine (the part's -Y face, by Generated coordinates) and paper-coloured page edges on the other faces' insides.
    Build each book as one box with its spine to -Y."""
    key = ("book", name)

    def make():
        k = NK(name)
        tc = k.N["Texture Coordinate"]
        sx = k.N.new("ShaderNodeSeparateXYZ")
        k.L.new(tc.outputs["Generated"], sx.inputs[0])
        gy, gz, gx = sx.outputs["Y"], sx.outputs["Z"], sx.outputs["X"]
        base = k.pcol()
        big = k.noise(k.obj, 40.0, 3.0, 0.5)
        fine = k.noise(k.obj, 300.0, 3.0, 0.6)
        e = k.edge(0.002, 0.995, 0.9)
        cv = k.cavity(0.02)
        spine = k.mr(gy, 0.04, 0.02)
        band = None
        for a, b in ((0.06, 0.075), (0.09, 0.10), (0.84, 0.855), (0.90, 0.915)):
            mb = k.m("MULTIPLY", k.mr(gz, a - 0.004, a), k.mr(gz, b + 0.004, b))
            band = mb if band is None else k.m("MAXIMUM", band, mb)
        label = k.m("MULTIPLY", k.mr(gz, 0.66, 0.67), k.mr(gz, 0.78, 0.77))
        label = k.m("MULTIPLY", label, k.m("MULTIPLY", k.mr(gx, 0.12, 0.18), k.mr(gx, 0.88, 0.82)))
        gilt = k.m("MULTIPLY", spine, band)
        lab = k.m("MULTIPLY", spine, label)
        # pages: faces whose normal is not along Y and not the side boards (X), seen at the top / fore-edge
        nz = k.N.new("ShaderNodeSeparateXYZ")
        k.L.new(k.geo.outputs["Normal"], nz.inputs[0])
        pages = k.m("MULTIPLY", k.mr(k.m("ABSOLUTE", nz.outputs["Z"]), 0.6, 0.8), k.mr(gy, 0.05, 0.08))
        pages = k.m("MULTIPLY", pages, k.m("MULTIPLY", k.mr(gx, 0.03, 0.08), k.mr(gx, 0.97, 0.92)))
        col = k.mix(k.mr(big, 0.3, 0.7, 0.0, 0.4), base, k.mul_col(base, (0.6, 0.6, 0.6, 1)))
        col = k.mix(k.m("MULTIPLY", e, 0.5), col, k.mix(0.5, base, (0.25, 0.2, 0.15, 1)))
        col = k.mix(lab, col, k.mul_col(base, (0.35, 0.35, 0.35, 1)))
        col = k.mix(gilt, col, k.mix(fine, (0.45, 0.32, 0.10, 1), (0.30, 0.20, 0.07, 1)))
        col = k.mix(pages, col, k.mix(fine, (0.55, 0.50, 0.38, 1), (0.42, 0.37, 0.27, 1)))
        col = k.mix(k.m("MULTIPLY", cv, 0.6), col, k.mul_col(base, (0.3, 0.3, 0.3, 1)))
        rough = k.mixf(gilt, k.mr(big, 0.3, 0.7, 0.55, 0.75), 0.3)
        rough = k.mixf(pages, rough, 0.85)
        metal_v = k.m("MULTIPLY", gilt, 0.9)
        h = k.m("ADD", k.m("MULTIPLY", fine, 0.3), k.m("MULTIPLY", gilt, 0.4))
        return k.out(col, rough, metal_v, k.bump(h, 0.0003, 0.3))

    return _cached(key, make)


BOOK_COLOURS = [(0.09, 0.015, 0.012), (0.035, 0.05, 0.025), (0.02, 0.025, 0.06), (0.10, 0.05, 0.02),
                (0.05, 0.03, 0.02), (0.14, 0.09, 0.04), (0.06, 0.008, 0.012), (0.018, 0.035, 0.03),
                (0.16, 0.11, 0.06), (0.03, 0.02, 0.015)]


def book_row(prefix, x0, x1, y_back, z, depth_max, h_range=(0.17, 0.27), seed=1, lean_end=True, gaps=0.06,
             stacks=1, mat=None):
    """Fill a shelf from x0 to x1 with books standing on z, backs against y_back (spines to -Y). Returns parts."""
    R = random.Random(seed)
    mat = mat or book_cloth()
    parts = []
    last_dims = None
    x = x0 + R.uniform(0.0, 0.02)
    stack_at = [R.uniform(x0 + 0.15, x1 - 0.25) for _ in range(stacks)]
    run_h = R.uniform(*h_range)
    col = R.choice(BOOK_COLOURS)
    while x < x1 - 0.03:
        if stack_at and x > stack_at[0]:           # a small stack lying flat
            stack_at.pop(0)
            zz = z
            for j in range(R.randint(2, 4)):
                bw, bd, bt = R.uniform(0.15, 0.22), R.uniform(0.12, 0.17), R.uniform(0.025, 0.045)
                if x + bw > x1:
                    break
                # built standing (thickness X, height Z) so the spine bands follow the book; laid flat by
                # the object rotation (local Z -> world X, thickness -> world Z)
                b = box(f"{prefix}_s{len(parts)}", (bt, bd, bw), (x + bw / 2, y_back - bd / 2 - 0.01, zz + bt / 2),
                        (0, math.pi / 2, R.uniform(-0.06, 0.06)), bev=0.002, seg=1)
                set_color(b, R.choice(BOOK_COLOURS))
                assign(b, mat)
                parts.append(b)
                zz += bt
            x += 0.24
            continue
        if R.random() < 0.25:                       # runs of a set: same height and colour
            run_h = R.uniform(*h_range)
            col = R.choice(BOOK_COLOURS)
        elif R.random() < 0.5:
            col = R.choice(BOOK_COLOURS)
        bt = R.uniform(0.018, 0.055)
        bh = min(h_range[1], max(h_range[0], run_h + R.uniform(-0.01, 0.01)))
        bd = min(depth_max, bh * R.uniform(0.62, 0.78))
        if x + bt > x1:
            break
        b = box(f"{prefix}_{len(parts)}", (bt, bd, bh), (x + bt / 2, y_back - bd / 2 - R.uniform(0.0, 0.02), z + bh / 2),
                bev=0.0025, seg=2)
        set_color(b, tuple(c * R.uniform(0.8, 1.2) for c in col))
        assign(b, mat)
        parts.append(b)
        last_dims = (bt, bh)
        x += bt + (R.uniform(0.0, 0.004))
        if R.random() < gaps * 0.5 and x < x1 - 0.2:
            x += R.uniform(0.03, 0.08)
    if lean_end and last_dims and len(parts) > 1 and parts[-1].rotation_euler.y == 0:
        # the last book leans back against the row (top toward -X), its foot on the shelf
        last = parts[-1]
        (bt, bh), ang = last_dims, 0.24
        xp = last.location.x - bt / 2 - 0.004               # right edge of the book before it
        ca, sa = math.cos(ang), math.sin(ang)
        cx = min(xp + bt / 2 * ca + bh / 2 * sa, x1 - (bt / 2 * ca + bh / 2 * sa))
        last.rotation_euler = (0, -ang, 0)
        last.location.x = cx
        last.location.z = z + bh / 2 * ca + bt / 2 * sa
    return parts


def rrect_path(w, h, r, n_corner=6, plane="xz", centre=(0, 0, 0), top_r=None):
    """Closed rounded-rectangle centreline (for piping) in plane "xz" (y fixed) or "xy" (z fixed) or "yz".
    top_r: a different corner radius on the top two corners (an arched top when ~w/2)."""
    tr = top_r if top_r is not None else r
    pts = []
    corners = [(w / 2 - tr, h / 2 - tr, tr, 0.0), (-w / 2 + tr, h / 2 - tr, tr, math.pi / 2),
               (-w / 2 + r, -h / 2 + r, r, math.pi), (w / 2 - r, -h / 2 + r, r, 1.5 * math.pi)]
    for cx, cy, rr, a0 in corners:
        for k in range(n_corner + 1):
            a = a0 + (math.pi / 2) * k / n_corner
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    out = []
    for u, v in pts:
        if plane == "xz":
            out.append((centre[0] + u, centre[1], centre[2] + v))
        elif plane == "xy":
            out.append((centre[0] + u, centre[1] + v, centre[2]))
        else:
            out.append((centre[0], centre[1] + u, centre[2] + v))
    # drop consecutive duplicates (zero-radius corners)
    ded = [out[0]]
    for p in out[1:]:
        if (Vector(p) - Vector(ded[-1])).length > 1e-5:
            ded.append(p)
    if (Vector(ded[0]) - Vector(ded[-1])).length < 1e-5:
        ded.pop()
    return ded


def image_material(path, name, fallback=(0.45, 0.40, 0.30), varnish=0.0, emit=0.0):
    """A picture/print surface: the image at `path` mapped by the part's Generated X / Y (so a flat box part shows it
    across its face), with a little paper/varnish noise. Falls back to a plain colour when the file is missing."""
    import os as _os
    key = ("img", path, name, emit)

    def make():
        k = NK(name)
        tc = k.N["Texture Coordinate"]
        if path and _os.path.exists(path):
            im = k.N.new("ShaderNodeTexImage")
            im.image = bpy.data.images.load(path)
            im.extension = "EXTEND"
            k.L.new(tc.outputs["Generated"], im.inputs["Vector"])
            if emit:                      # a picture that gives its own light (a view of the sky, of space)
                return k.out((0.0, 0.0, 0.0, 1), 1.0, 0.0, None, emit=im.outputs["Color"], emit_strength=emit)
            col = k.mul_col(im.outputs["Color"], (0.82, 0.80, 0.74, 1))
        else:
            col = fallback
        n = k.noise(k.obj, 60.0, 3.0, 0.5)
        col = k.mix(k.mr(n, 0.4, 0.7, 0.0, 0.15), col, (0.30, 0.24, 0.14, 1))
        rough = lerp(0.85, 0.35, varnish)
        return k.out(col, rough, 0.0, k.bump(n, 0.0002, 0.1))

    return _cached(key, make)


def gilt(name="gilt"):
    """Old gilding on gesso: gold with the red bole showing on worn high points, dark grime in the hollows."""
    key = ("gilt", name)

    def make():
        k = NK(name)
        big = k.noise(k.obj, 12.0, 3.0, 0.5)
        mid = k.noise(k.obj, 70.0, 4.0, 0.55)
        e = k.edge(0.0015)
        cv = k.cavity(0.012, 0.9, 0.45)
        col = k.mix(k.mr(big, 0.3, 0.7), (0.42, 0.29, 0.10, 1), (0.55, 0.40, 0.15, 1))
        bole = k.m("MULTIPLY", e, k.mr(mid, 0.55, 0.7))
        col = k.mix(bole, col, (0.20, 0.07, 0.03, 1))
        col = k.mix(k.m("MULTIPLY", cv, 0.85), col, (0.05, 0.035, 0.015, 1))
        metal_v = k.mixf(k.m("ADD", bole, k.m("MULTIPLY", cv, 0.8)), 0.85, 0.1)
        rough = k.mixf(cv, k.mr(mid, 0.3, 0.7, 0.28, 0.45), 0.8)
        return k.out(col, rough, metal_v, k.bump(mid, 0.0002, 0.25))

    return _cached(key, make)


def framed_picture(pid, img_w, img_h, frame_w=0.065, mount_w=0.06, depth=0.045, picture_path=None, oil=False):
    """A wall picture: gilt moulded frame (four mitred runs), pale mount (none for an oil), the picture, a backboard.
    Built flat (front +Z) then turned upright: front -Y, back flat at the back of the bounds."""
    parts = []
    mw = 0.0 if oil else mount_w
    ow, oh = img_w + 2 * mw, img_h + 2 * mw          # frame opening
    fw = frame_w
    prof = [(0.0, 0.0), (0.0, 0.010), (-0.004, 0.012), (-0.010 * fw / 0.065, depth * 0.40),
            (-0.018 * fw / 0.065, depth * 0.48)]
    for k_ in range(7):                                # bead and cove up to the outer roll
        t = k_ / 6
        prof.append((-(0.022 + 0.026 * t) * fw / 0.065, depth * (0.52 + 0.40 * t ** 1.6 + 0.04 * math.sin(math.pi * t * 2))))
    for k_ in range(6):                                # outer roll
        a = math.pi / 2 * k_ / 5
        prof.append((-(0.050 + 0.015 * math.sin(a)) * fw / 0.065, depth * (0.92 + 0.08 * math.cos(a)) - depth * 0.35 * (1 - math.cos(a))))
    prof += [(-fw, depth * 0.25), (-fw, 0.0)]
    gm = gilt()
    runs = ((0, -oh / 2, 0.0, ow), (0, oh / 2, math.pi, ow), (ow / 2, 0, math.pi / 2, oh), (-ow / 2, 0, -math.pi / 2, oh))
    for i, (x, y, rz, L) in enumerate(runs):
        m = moulding(f"{pid}_fr{i}", prof, -L / 2, L / 2, 1, 1, gm)
        m.rotation_euler = (0, 0, rz)
        m.location = (x, y, 0)
        parts.append(m)
    back = box(f"{pid}_back", (ow + 0.01, oh + 0.01, 0.006), (0, 0, 0.003), bev=0.0,
               mat=wood("pine", "X", wear=0.0, gloss=0.0, grime=1.0))
    parts.append(back)
    pic = box(f"{pid}_pic", (img_w + 0.004, img_h + 0.004, 0.002), (0, 0, 0.007), bev=0.0,
              mat=image_material(picture_path, f"{pid}_print", varnish=0.8 if oil else 0.0))
    parts.append(pic)
    if mw > 0:
        mount = paper((0.55, 0.51, 0.40), name="mount")
        for (x, y, sx, sy) in ((0, -(img_h / 2 + mw / 2), ow, mw), (0, img_h / 2 + mw / 2, ow, mw),
                               (-(img_w / 2 + mw / 2), 0, mw, img_h), (img_w / 2 + mw / 2, 0, mw, img_h)):
            parts.append(box(f"{pid}_mt{x:.2f}{y:.2f}", (sx, sy, 0.003), (x, y, 0.0095), bev=0.0006, seg=1, mat=mount))
    # glass is left out on purpose: a glazed print reflects the room and hides the picture in keyframes
    for o in parts:
        mw_ = Matrix.Rotation(math.pi / 2, 4, "X")
        o.matrix_world = mw_ @ o.matrix_basis
    # hanging wire hooks are behind (not modelled)
    return parts


def plate(name, outline, thick, y=0.0, mat=None, bev=0.0015, seg=2):
    """A flat part cut from a 2D outline in the XZ plane (list of (x, z), counter-clockwise seen from -Y), extruded
    `thick` along Y centred on y. Bevelled edges."""
    bm = bmesh.new()
    f = [bm.verts.new((x, y - thick / 2, z)) for x, z in outline]
    b = [bm.verts.new((x, y + thick / 2, z)) for x, z in outline]
    bm.faces.new(f)
    bm.faces.new(list(reversed(b)))
    n = len(outline)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((f[j], f[i], b[i], b[j]))
    bmesh.ops.triangulate(bm, faces=[fc for fc in bm.faces if len(fc.verts) > 4])
    ob = from_bm(name, bm)
    if bev:
        bevel(ob, bev, seg, 30)
    shade(ob, 30)
    if mat is not None:
        assign(ob, mat)
    return ob


def bail_handle(prefix, x, y, z, w=0.075, mat=None):
    """Brass swan-neck bail drawer handle on a face at y (front -Y): two oval backplates, two posts and a hanging
    curved bail. Returns the parts."""
    parts = []
    for s in (-1, 1):
        bp = lathe(f"{prefix}_bp{s}", [(0.0001, 0.0), (0.011, 0.0), (0.010, 0.002), (0.0001, 0.0025)], seg=20, mat=mat)
        bp.data.transform(Matrix.Diagonal((1.0, 1.45, 1.0, 1.0)))
        bp.location, bp.rotation_euler = (x + s * w / 2, y, z), (math.pi / 2, 0, 0)
        po = lathe(f"{prefix}_po{s}", [(0.0001, 0.0), (0.0035, 0.0), (0.003, 0.012), (0.0045, 0.014), (0.0001, 0.017)],
                   seg=12, mat=mat)
        po.location, po.rotation_euler = (x + s * w / 2, y, z), (math.pi / 2, 0, 0)
        parts += [bp, po]
    pts = []
    for k in range(21):
        t = k / 20
        xx = x + (t - 0.5) * w
        sag = 0.020 * math.sin(math.pi * t) + 0.004 * math.sin(3 * math.pi * t)
        pts.append((xx, y - 0.015, z - sag))
    parts.append(tube(f"{prefix}_bail", pts, 0.0028, seg=8, mat=mat))
    return parts
