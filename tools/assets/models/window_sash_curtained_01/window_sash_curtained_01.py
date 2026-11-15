"""Model script: six-over-six sash window with tied-back damask curtains, front -Y (room side).

Wall face at y = 0 (room toward -Y); opening x +-0.53, z 0.75..2.61; reveal linings, box frame, sashes and glass sit
behind the wall face (y > 0), the architrave, window board, pole and curtains in front. Curtains are grids folded
with sines, gathered to the tie-back and flared to the floor, given thickness by a Solidify modifier."""

import math
import os
import sys

import bmesh
import bpy

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

OX, Z0, Z1 = 0.53, 0.75, 2.61
paint = K.painted_wood((0.40, 0.35, 0.26), wear=0.5, gloss=0.45)
paint_in = K.painted_wood((0.36, 0.32, 0.24), wear=0.3, gloss=0.4)
glass = K.glass((0.86, 0.92, 0.88), 0.03, name="window_glass")
brass = K.metal("brass", tarnish=0.7, polish=0.5)
drape = K.fabric("damask", (0.085, 0.040, 0.016), wear=0.4, figure=(0.030, 0.012, 0.007), scale=3.0)
cord = K.fabric("rope", (0.12, 0.06, 0.02))
parts = []

# ---- reveal linings and box frame (behind the wall face)
for s in (-1, 1):
    parts.append(K.box(f"lining{s}", (0.02, 0.22, Z1 - Z0), (s * (OX + 0.01), 0.11, (Z0 + Z1) / 2), bev=0.001, mat=paint_in))
parts.append(K.box("lining_top", (2 * OX + 0.04, 0.22, 0.02), (0, 0.11, Z1 + 0.01), bev=0.001, mat=paint_in))
FY0, FY1 = 0.12, 0.22
for s in (-1, 1):
    parts.append(K.box(f"frame{s}", (0.05, FY1 - FY0, Z1 - Z0), (s * (OX - 0.025), (FY0 + FY1) / 2, (Z0 + Z1) / 2), bev=0.002, mat=paint))
    parts.append(K.box(f"bead{s}", (0.012, 0.012, Z1 - Z0 - 0.05), (s * (OX - 0.056), FY0 + 0.012, (Z0 + Z1) / 2), bev=0.004, seg=2, mat=paint))
parts.append(K.box("frame_head", (2 * OX, FY1 - FY0, 0.05), (0, (FY0 + FY1) / 2, Z1 - 0.025), bev=0.002, mat=paint))
parts.append(K.box("frame_sill", (2 * OX, FY1 - FY0 + 0.03, 0.04), (0, (FY0 + FY1) / 2 + 0.015, Z0 + 0.02), bev=0.002, mat=paint))

# ---- two sashes, 3 x 2 panes each; lower sash in the inner track, upper in the outer
SW = 2 * OX - 0.10
SH = (Z1 - Z0 - 0.09) / 2 + 0.02


def sash(tag, y, zb, bottom_rail, top_rail):
    out = []
    t = 0.035
    for s in (-1, 1):
        out.append(K.box(f"{tag}_st{s}", (0.045, t, SH), (s * (SW / 2 - 0.0225), y, zb + SH / 2), bev=0.002, mat=paint))
    out.append(K.box(f"{tag}_rb", (SW - 0.09, t, bottom_rail), (0, y, zb + bottom_rail / 2), bev=0.002, mat=paint))
    out.append(K.box(f"{tag}_rt", (SW - 0.09, t, top_rail), (0, y, zb + SH - top_rail / 2), bev=0.002, mat=paint))
    gx0, gx1 = -SW / 2 + 0.045, SW / 2 - 0.045
    gz0, gz1 = zb + bottom_rail, zb + SH - top_rail
    for k in (1, 2):
        x = K.lerp(gx0, gx1, k / 3)
        out.append(K.box(f"{tag}_gv{k}", (0.018, t * 0.8, gz1 - gz0), (x, y, (gz0 + gz1) / 2), bev=0.003, seg=2, mat=paint))
    zm = (gz0 + gz1) / 2
    out.append(K.box(f"{tag}_gh", (gx1 - gx0, t * 0.8, 0.018), (0, y, zm), bev=0.003, seg=2, mat=paint))
    out.append(K.box(f"{tag}_glass", (gx1 - gx0 + 0.01, 0.004, gz1 - gz0 + 0.01), (0, y + 0.004, (gz0 + gz1) / 2), bev=0.0, mat=glass))
    return out


parts += sash("lower", FY0 + 0.035, Z0 + 0.04, 0.075, 0.035)
parts += sash("upper", FY0 + 0.075, Z1 - 0.05 - SH, 0.035, 0.05)
# sash lift and catch (brass)
for s in (-1, 1):
    parts.append(K.box(f"lift{s}", (0.03, 0.015, 0.012), (s * 0.3, FY0 + 0.012, Z0 + 0.075), bev=0.003, seg=2, mat=brass))
parts.append(K.box("catch", (0.05, 0.02, 0.012), (0, FY0 + 0.05, Z0 + 0.04 + SH + 0.006), bev=0.003, seg=2, mat=brass))

# ---- window board, apron, architrave with corner blocks (room side of the wall face)
parts.append(K.box("board", (2 * OX + 0.16, 0.20, 0.03), (0, 0.04, Z0 - 0.015), bev=0.004, seg=3, mat=paint))
parts.append(K.box("apron", (2 * OX + 0.10, 0.02, 0.10), (0, -0.01, Z0 - 0.08), bev=0.002, mat=paint))
arch = [(0.0, 0.0), (-0.010, 0.0), (-0.016, 0.02), (-0.021, 0.045), (-0.018, 0.07), (-0.012, 0.08), (-0.012, 0.09),
        (0.0, 0.09)]
for s in (-1, 1):
    a = K.moulding(f"arch{s}", arch, Z0, Z1, 0, 0, paint)
    # run along X -> along Z; the profile's width (its z) -> -X, so the right one starts 0.09 out
    a.rotation_euler = (0, -math.pi / 2, 0)
    a.location = (OX + 0.09 if s > 0 else -OX, 0.0, 0.0)
    parts.append(a)
top = K.moulding("arch_top", arch, -OX, OX, 0, 0, paint)
top.location = (0, 0, Z1)
parts.append(top)
for s in (-1, 1):
    cb = K.box(f"cblock{s}", (0.10, 0.026, 0.10), (s * (OX + 0.045), -0.013, Z1 + 0.045), bev=0.003, mat=paint)
    ro = K.lathe(f"rosette{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.035, "c"), (0.004, 0.034), (0.008, 0.024),
                                         (0.011, 0.016), (0.016, 0.006), (0.018, 0.0001)], 0.0015), seg=24, mat=paint)
    ro.location, ro.rotation_euler = (s * (OX + 0.045), -0.026, Z1 + 0.045), (math.pi / 2, 0, 0)
    parts += [cb, ro]

# ---- curtain pole, finials, brackets
PZ, PY, PX = 2.70, -0.12, 0.80
pole = K.lathe("pole", [(0.0001, 0.0), (0.016, 0.0), (0.016, 2 * PX), (0.0001, 2 * PX)], seg=20, mat=brass)
pole.location, pole.rotation_euler = (-PX, PY, PZ), (0, math.pi / 2, 0)
parts.append(pole)
for s in (-1, 1):
    fin = K.lathe(f"finial{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.02, "c"), (0.01, 0.012), *K.bead(0.02, 0.02, 0.01),
                                         (0.03, 0.012), (0.045, 0.034), (0.07, 0.034), (0.095, 0.012), (0.10, 0.0001)], 0.002),
                   seg=24, mat=brass)
    fin.location, fin.rotation_euler = (s * PX, PY, PZ), (0, s * math.pi / 2, 0)
    br = K.tube(f"bracket{s}", [(s * (PX - 0.06), PY, PZ), (s * (PX - 0.06), PY + 0.05, PZ + 0.02),
                                (s * (PX - 0.06), -0.004, PZ + 0.03)], 0.007, mat=brass)
    plate = K.lathe(f"plate{s}", [(0.0001, 0.0), (0.025, 0.0), (0.025, 0.006), (0.018, 0.01), (0.0001, 0.01)], seg=20, mat=brass)
    plate.location, plate.rotation_euler = (s * (PX - 0.06), 0.0, PZ + 0.03), (math.pi / 2, 0, 0)
    parts += [fin, br, plate]

# ---- curtains
def curtain(s, seed):
    """Left (s = -1) or right (s = +1) curtain. u: 0 at the outer edge (wall side) -> 1 at the leading edge."""
    nu, nz = 70, 90
    TOPZ, TIEZ = PZ - 0.035, 1.05
    out_x = s * (PX - 0.04)
    folds = 7.0
    bm = bmesh.new()
    grid = []
    for j in range(nz + 1):
        z = K.lerp(0.0, TOPZ, j / nz)
        if z >= TIEZ:
            t = (z - TIEZ) / (TOPZ - TIEZ)
            w = K.lerp(0.13, 0.62, t ** 1.6)
            amp = K.lerp(0.040, 0.028, t)
        else:
            t = (TIEZ - z) / TIEZ
            w = K.lerp(0.13, 0.42, K.smooth(t) ** 0.8)
            amp = K.lerp(0.040, 0.050, t)
        row = []
        for i in range(nu + 1):
            u = i / nu
            ph = folds * u + 0.15 * math.sin(z * 3.0 + seed)
            fy = amp * (abs(math.sin(math.pi * ph)) * 2 - 1)
            x = out_x - s * u * w
            y = PY + 0.045 + fy * (0.6 + 0.4 * u) - 0.02 * u * (1 - abs(z - TIEZ) / 1.7)
            if z < 0.04:                               # hem just clears the floor, slight break
                y -= (0.04 - z) * 0.5
            row.append(bm.verts.new((x, y, max(z, 0.005))))
        grid.append(row)
    for j in range(nz):
        for i in range(nu):
            bm.faces.new((grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]))
    ob = K.from_bm(f"curtain{s}", bm)
    if s > 0:
        bmx = bmesh.new(); bmx.from_mesh(ob.data); bmesh.ops.reverse_faces(bmx, faces=bmx.faces); bmx.to_mesh(ob.data); bmx.free()
    so = ob.modifiers.new("solid", "SOLIDIFY")
    so.thickness, so.offset = 0.006, 0.0
    K.shade(ob, 80)
    K.assign(ob, drape)
    return ob


for s, seed in ((-1, 0.3), (1, 1.7)):
    c = curtain(s, seed)
    parts.append(c)
    # rings along the top
    for k in range(9):
        u = k / 8
        w = 0.62
        rx = s * (PX - 0.04) - s * u * w
        bpy.ops.mesh.primitive_torus_add(major_radius=0.024, minor_radius=0.0035, major_segments=20, minor_segments=6,
                                         location=(rx, PY, PZ), rotation=(0, math.pi / 2, 0))
        r = bpy.context.active_object
        r.name = f"ring{s}_{k}"
        K.assign(r, brass)
        parts.append(r)
    # tie-back: a cord round the gather, with a tassel
    cx = s * (PX - 0.04) - s * 0.065
    loop = [(cx + 0.085 * math.cos(a), PY + 0.045 + 0.06 * math.sin(a), 1.05 + 0.01 * math.sin(2 * a))
            for a in [math.tau * k / 24 for k in range(24)]]
    parts.append(K.tube(f"tie{s}", loop, 0.007, seg=8, closed=True, mat=cord))
    tx = cx - s * 0.085
    hook = K.lathe(f"tiehook{s}", [(0.0001, 0.0), (0.012, 0.0), (0.012, 0.012), (0.008, 0.016), (0.0001, 0.016)], seg=16, mat=brass)
    hook.location, hook.rotation_euler = (tx, -0.0, 1.05), (math.pi / 2, 0, 0)
    tas = K.lathe(f"tassel{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.02, "c"), (0.04, 0.024), (0.07, 0.016),
                                         (0.08, 0.009), (0.095, 0.012), (0.11, 0.008), (0.12, 0.0001)], 0.003),
                  seg=20, mat=cord)
    tas.location = (cx + 0.02 * s, PY + 0.045 - 0.065, 1.05 - 0.16)
    tcord = K.tube(f"tascord{s}", [(cx + 0.02 * s, PY + 0.045 - 0.062, 1.05 - 0.008),
                                   (cx + 0.02 * s, PY + 0.045 - 0.065, 1.05 - 0.04)], 0.004, mat=cord)
    parts += [hook, tas, tcord]
