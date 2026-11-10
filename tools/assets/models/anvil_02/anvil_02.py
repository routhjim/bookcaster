"""Model script: a blacksmith's anvil (London pattern), ~0.62 m long (X), 0.205 m wide at the feet (Y), 0.28 m tall (Z).

Front faces -Y, horn points +X, heel -X.

The shape is a signed distance field on a dense voxel grid (numpy), meshed with OpenVDB (bundled with Blender):
- body, step, waist and base: a side profile (XZ polygon with filleted corners) intersected with a front profile
  (half-width by height, narrower under the step), smooth-max so the edges are dressed, not razor sharp;
- horn: a tapering superellipse section (boxy at the step, round at the tip, flat-topped), smooth-unioned into the
  step and the chest so it flows out of the body instead of sitting on it;
- arched cut-outs between the four feet, square hardy and round pritchel holes;
- forging detail in the field: low-frequency hand-forged unevenness, hammer dents on body, waist and horn, chipped
  face edges, a slightly mushroomed heel end, a shallow sway in the face.
Masks for the material (convex edges, cavities, wear zones) are sampled from the field onto the vertices as
attributes; the material reads them, so the bake does not depend on noisy bevel or AO shader nodes.

ANVIL_VOX_MM (default 1.0) sets the voxel size.
"""

import math
import os
import time

import bpy
import numpy as np
import openvdb as vdb

T0 = time.time()
RNG = np.random.default_rng(1702)
H = float(os.environ.get("ANVIL_VOX_MM", "1.0")) / 1000.0
ORIG = np.array([-0.325, -0.115, -0.004])
EXT = np.array([0.650, 0.230, 0.292])
NX, NY, NZ = (np.ceil(EXT / H).astype(int) + 1).tolist()
xs = (ORIG[0] + H * np.arange(NX)).astype(np.float32)
ys = (ORIG[1] + H * np.arange(NY)).astype(np.float32)
zs = (ORIG[2] + H * np.arange(NZ)).astype(np.float32)
X, Y, Z = xs[:, None, None], ys[None, :, None], zs[None, None, :]

# ------------------------------------------------------------------ dimensions (m)
TOP = 0.280            # striking face
STEP_TOP = 0.2595      # table, ~20 mm below the face
X_HEEL = -0.310
X_STEP = 0.055         # face ends, table begins
X_STEP_END = 0.128     # table ends, horn begins
X_TIP = 0.312
FACE_HW = 0.0505       # face half-width
STEP_HW = 0.0425
HARDY = (-0.266, 0.0112)
PRITCHEL = (-0.222, 0.0058)


# ------------------------------------------------------------------ SDF helpers
def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def fillet_polygon(pts):
    """pts: closed list of (x, z, r). Each corner with r > 0 becomes a tangent arc."""
    out = []
    n = len(pts)
    for i in range(n):
        ax, az, _ = pts[i - 1]
        px, pz, r = pts[i]
        bx, bz, _ = pts[(i + 1) % n]
        if r <= 0:
            out.append((px, pz))
            continue
        u1 = np.array([ax - px, az - pz])
        u2 = np.array([bx - px, bz - pz])
        l1, l2 = np.linalg.norm(u1), np.linalg.norm(u2)
        u1, u2 = u1 / l1, u2 / l2
        th = math.acos(max(-1.0, min(1.0, float(u1 @ u2))))
        t = min(r / math.tan(th / 2), 0.49 * l1, 0.49 * l2)
        r = t * math.tan(th / 2)
        t1 = np.array([px, pz]) + u1 * t
        t2 = np.array([px, pz]) + u2 * t
        bis = (u1 + u2) / np.linalg.norm(u1 + u2)
        c = np.array([px, pz]) + bis * (r / math.sin(th / 2))
        a1 = math.atan2(*(t1 - c)[::-1])
        a2 = math.atan2(*(t2 - c)[::-1])
        da = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        m = max(3, int(abs(da) * r / 0.0007))
        for k in range(m + 1):
            a = a1 + da * k / m
            out.append((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
    return np.array(out, dtype=np.float64)


def polygon_sdf(poly, U, V):
    """Signed distance (negative inside) from grid points (U along axis 0, V along axis 1) to a closed polygon."""
    P = np.stack(np.broadcast_arrays(U[:, None], V[None, :]), -1).astype(np.float64)
    d = np.full(P.shape[:2], np.inf)
    s = np.ones(P.shape[:2])
    n = len(poly)
    for i in range(n):
        vi, vj = poly[i], poly[i - 1]
        e = vj - vi
        w = P - vi
        tt = np.clip((w @ e) / (e @ e), 0.0, 1.0)
        b = w - e * tt[..., None]
        d = np.minimum(d, (b * b).sum(-1))
        c1 = P[..., 1] >= vi[1]
        c2 = P[..., 1] < vj[1]
        c3 = e[0] * w[..., 1] > e[1] * w[..., 0]
        flip = (c1 & c2 & c3) | (~c1 & ~c2 & ~c3)
        s[flip] *= -1
    return (s * np.sqrt(d)).astype(np.float32)


def pchip(xk, yk, x):
    """Monotone cubic (Fritsch-Carlson) through (xk, yk); returns value and derivative at x."""
    xk, yk = np.asarray(xk, float), np.asarray(yk, float)
    hk = np.diff(xk)
    dk = np.diff(yk) / hk
    m = np.zeros_like(yk)
    for i in range(1, len(yk) - 1):
        if dk[i - 1] * dk[i] > 0:
            w1, w2 = 2 * hk[i] + hk[i - 1], hk[i] + 2 * hk[i - 1]
            m[i] = (w1 + w2) / (w1 / dk[i - 1] + w2 / dk[i])
    m[0], m[-1] = dk[0], dk[-1]
    i = np.clip(np.searchsorted(xk, x) - 1, 0, len(xk) - 2)
    t = (x - xk[i]) / hk[i]
    t = np.clip(t, 0, 1)
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    y = h00 * yk[i] + h10 * hk[i] * m[i] + h01 * yk[i + 1] + h11 * hk[i] * m[i + 1]
    dy = ((6 * t**2 - 6 * t) * yk[i] + (3 * t**2 - 4 * t + 1) * hk[i] * m[i] + (-6 * t**2 + 6 * t) * yk[i + 1]
          + (3 * t**2 - 2 * t) * hk[i] * m[i + 1]) / hk[i]
    return y, dy


def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bspline_weights(n, origin, h, cell):
    """(n, m) cubic B-spline weights mapping m coarse knots (spacing `cell`) onto n grid samples."""
    u = (origin + h * np.arange(n)) / cell
    u = u - np.floor(u.min()) + 1.0
    i = np.floor(u).astype(int)
    f = u - i
    W = np.zeros((n, int(i.max()) + 3), np.float32)
    ws = [(1 - f) ** 3 / 6, (3 * f**3 - 6 * f**2 + 4) / 6, (-3 * f**3 + 3 * f**2 + 3 * f + 1) / 6, f**3 / 6]
    for k, wk in enumerate(ws):
        W[np.arange(n), i - 1 + k] += wk
    return W


def vnoise(cell, seed):
    """Smooth value noise (cubic B-spline of random knots), std ~1, over the full grid."""
    rng = np.random.default_rng(seed)
    Wx = bspline_weights(NX, ORIG[0], H, cell)
    Wy = bspline_weights(NY, ORIG[1], H, cell)
    Wz = bspline_weights(NZ, ORIG[2], H, cell)
    G = rng.standard_normal((Wx.shape[1], Wy.shape[1], Wz.shape[1])).astype(np.float32)
    A = np.tensordot(Wx, G, axes=(1, 0))
    A = np.tensordot(A, Wy, axes=(1, 1))            # (nx, cz, ny)
    A = np.tensordot(A, Wz, axes=(1, 1))            # (nx, ny, nz)
    return A / (A.std() + 1e-9)


def box_blur(a, r):
    """Separable box blur, radius r voxels, edge-clamped, applied twice (tent-like)."""
    for _ in range(2):
        for ax in range(3):
            pad = [(0, 0)] * 3
            pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(a, pad, mode="edge"), axis=ax, dtype=np.float32)
            n = a.shape[ax]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax)
            lo = np.take(c, np.arange(0, n), axis=ax)
            a = (hi - lo) / (2 * r + 1)
    return a


def idx(lo, hi, axis_vals):
    return slice(max(0, int(np.searchsorted(axis_vals, lo))), int(np.searchsorted(axis_vals, hi)))


# ------------------------------------------------------------------ 1. side profile (XZ), filleted
# (x, z, fillet radius); goes heel top -> face -> step -> chest -> horn-side foot -> floor -> heel-side foot -> heel
SIDE = [
    (X_HEEL, TOP, 0.0035),
    (X_STEP, TOP, 0.0025),
    (X_STEP, 0.205, 0.0),            # under the table: the table and horn are one sweep (section 3)
    (X_STEP_END - 0.012, 0.205, 0.02),
    (0.062, 0.158, 0.045),           # chest curving down into the waist
    (0.050, 0.120, 0.08),            # waist, horn side
    (0.138, 0.036, 0.016),           # foot top, horn side
    (0.144, 0.0, 0.003),             # toe
    (-0.180, 0.0, 0.003),
    (-0.174, 0.036, 0.016),          # foot top, heel side
    (-0.088, 0.120, 0.08),           # waist, heel side
    (-0.094, 0.176, 0.05),           # under the heel, into the waist
    (-0.205, 0.218, 0.12),           # heel underside, a long shallow curve
    (X_HEEL, 0.235, 0.004),          # heel end, bottom corner
]
t_ = time.time()
side_poly = fillet_polygon(SIDE)
d_side = polygon_sdf(side_poly, xs, zs)[:, None, :]           # (nx, 1, nz)

# ------------------------------------------------------------------ 2. front profile: half-width by height
HW_Z = [0.000, 0.012, 0.030, 0.050, 0.075, 0.100, 0.130, 0.160, 0.185, 0.205, 0.216, 0.300]
HW_V = [0.1025, 0.1015, 0.092, 0.077, 0.062, 0.0525, 0.0455, 0.0450, 0.0470, 0.0497, 0.0505, 0.0505]
hw_z, dhw_z = pchip(HW_Z, HW_V, zs.astype(float))
hw_z, dhw_z = hw_z.astype(np.float32)[None, None, :], dhw_z.astype(np.float32)[None, None, :]
# the table and horn root are narrower than the face (a small shoulder at X_STEP); not the waist or the feet
narrow = sstep(X_STEP - 0.001, X_STEP + 0.007, X) * sstep(0.168, 0.205, Z)
hw = hw_z * (1.0 - (1.0 - STEP_HW / FACE_HW) * narrow)
d_front = (np.abs(Y) - hw) / np.sqrt(1.0 + dhw_z ** 2)

# edge radius by height: dressed 3 mm at the face, softer forged edges lower down
k_edge = (0.0024 + 0.010 * (1 - sstep(0.20, 0.25, Z)) * sstep(0.03, 0.10, Z)
          + 0.006 * (1 - sstep(0.0, 0.04, Z))).astype(np.float32)
d = smax(d_side, d_front, k_edge).astype(np.float32)
del d_front
print(f"[anvil] grid {NX}x{NY}x{NZ} h={H*1000:.2f}mm, body {time.time()-t_:.1f}s")

# ------------------------------------------------------------------ 3. table + horn: one sweep along X
# The table is the sweep's first part (constant boxy section); from X_STEP_END the section shrinks, drops a little
# and rounds off toward the tip, so the horn grows out of the table instead of being stuck on it.
t_ = time.time()
sx, sz = idx(X_STEP - 0.012, X_TIP + 0.01, xs), idx(0.13, 0.275, zs)
Xh, Zh = X[sx], Z[:, :, sz]
t = np.clip((Xh - X_STEP_END) / (X_TIP - X_STEP_END), 0.0, 1.0)
H0 = STEP_TOP - 0.190
ztop = STEP_TOP - 0.015 * t ** 1.6
hh = (H0 - 0.0072) * (1 - t) ** 1.15 + 0.0072
ww = (2 * STEP_HW - 0.0072) * (1 - t) ** 1.12 + 0.0072
ex = 2.05 + 4.95 * (1 - sstep(0.0, 0.38, t))       # boxy table, round from mid-horn on
zc = ztop - hh / 2
a = hh / 2
rel = np.clip((Zh - zc) / a, -1.0, 1.0)
b = ww / 2 * (1.0 + 0.10 * rel * sstep(0.0, 0.15, t))     # pear: a little fuller toward the top, on the horn
q = (np.abs(Y) / b) ** ex + (np.abs(Zh - zc) / a) ** ex
dh = (q ** (1.0 / ex) - 1.0) * np.minimum(a, ww / 2)
# shallow lengthwise facets left by forging, on the horn only, fading out toward the tip
ang = np.arctan2(Zh - zc, Y)
dh = dh + 0.00022 * np.cos(7 * ang + 14 * Xh) * sstep(0.0, 0.1, t) * (1 - t)
dh = smax(dh, Xh - X_TIP, 0.004)
dh = np.maximum(dh, (X_STEP - 0.004) - Xh)
k_h = 0.0025 + 0.016 * (1 - sstep(0.215, 0.248, Zh))      # tight at the table top, wide under the horn
d[sx, :, sz] = smin(d[sx, :, sz], dh.astype(np.float32), k_h)
del dh, q, b, rel, ang
print(f"[anvil] horn {time.time()-t_:.1f}s")

# ------------------------------------------------------------------ 4. arches between the feet, floor
t_ = time.time()
sz = idx(-0.01, 0.06, zs)
Zs = Z[:, :, sz]
ax_, az_ = 0.070, 0.030          # arch seen from the front: under the waist between the two feet
cx = -0.0175
e1 = (np.sqrt(((X - cx) / ax_) ** 2 + (Zs / az_) ** 2) - 1.0) * min(ax_, az_)
by_, bz_ = 0.046, 0.026          # arch seen from the end
e2 = (np.sqrt((Y / by_) ** 2 + (Zs / bz_) ** 2) - 1.0) * min(by_, bz_)
sub = d[:, :, sz]
sub = smax(sub, -np.broadcast_to(e1, sub.shape), 0.008)
sub = smax(sub, -np.broadcast_to(e2, sub.shape), 0.008)
d[:, :, sz] = sub
d = smax(d, -Z, 0.0015).astype(np.float32)
print(f"[anvil] arches {time.time()-t_:.1f}s")

# ------------------------------------------------------------------ 5. hardy and pritchel holes
sx = idx(-0.30, -0.20, xs)
Xs = X[sx]
qx, qy = np.abs(Xs - HARDY[0]) - HARDY[1] + 0.0015, np.abs(Y) - HARDY[1] + 0.0015
hardy = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - 0.0015
prit = np.sqrt((Xs - PRITCHEL[0]) ** 2 + Y ** 2) - PRITCHEL[1]
holes = np.minimum(hardy, prit)
d[sx] = smax(d[sx], -np.broadcast_to(holes, d[sx].shape), 0.0016)

# ------------------------------------------------------------------ 6. hand-forged unevenness (not on the face)
t_ = time.time()
no_face = (1.0 - sstep(0.262, 0.277, Z) * (1 - sstep(X_STEP + 0.01, X_STEP + 0.03, X))).astype(np.float32)
no_floor = sstep(0.0, 0.01, Z).astype(np.float32)
wob = vnoise(0.050, 11) * 0.00055 + vnoise(0.018, 12) * 0.00022
d += wob * no_face * no_floor
del wob
# sway: the face is worn a little hollow toward its middle
d += (0.0006 * np.exp(-((X + 0.11) / 0.12) ** 2) * sstep(0.250, 0.279, Z) * (1 - sstep(X_STEP - 0.03, X_STEP, X))
      ).astype(np.float32)
print(f"[anvil] unevenness {time.time()-t_:.1f}s")


# ------------------------------------------------------------------ 7. hammer dents, chips, mushroomed heel
def window(c, R):
    i = np.round((np.asarray(c) - ORIG) / H).astype(int)
    rv = int(np.ceil(R / H)) + 1
    sl = tuple(slice(max(0, i[k] - rv), min((NX, NY, NZ)[k], i[k] + rv + 1)) for k in range(3))
    return sl


def dent(c, R, depth, stretch=(1.0, 1.0, 1.0)):
    sl = window(c, R * max(stretch))
    dx = (xs[sl[0]][:, None, None] - c[0]) / stretch[0]
    dy = (ys[sl[1]][None, :, None] - c[1]) / stretch[1]
    dz = (zs[sl[2]][None, None, :] - c[2]) / stretch[2]
    r2 = (dx * dx + dy * dy + dz * dz) / (R * R)
    d[sl] += (depth * np.clip(1 - r2, 0, 1) ** 2).astype(np.float32)


def chip(c, T, n1, n2, L, depth, w):
    """A spall off the edge at c (edge direction T, face normals n1, n2): two fracture facets bounded by a lens."""
    T, n1, n2 = (np.asarray(v, float) / np.linalg.norm(v) for v in (T, n1, n2))
    sl = window(c, max(L, w) + 0.002)
    px_ = xs[sl[0]][:, None, None] - c[0]
    py_ = ys[sl[1]][None, :, None] - c[1]
    pz_ = zs[sl[2]][None, None, :] - c[2]
    hs = []
    for k in range(2):
        tilt = RNG.uniform(0.25, 0.75)
        m = tilt * n1 + (1 - tilt) * n2 + T * RNG.uniform(-0.25, 0.25)
        m /= np.linalg.norm(m)
        dk = depth * RNG.uniform(0.6, 1.0)
        hs.append(-(m[0] * px_ + m[1] * py_ + m[2] * pz_) - dk)
    u = (T[0] * px_ + T[1] * py_ + T[2] * pz_) / L
    rest2 = (px_ ** 2 + py_ ** 2 + pz_ ** 2) - (u * L) ** 2
    lens = (np.sqrt(u ** 2 + np.maximum(rest2, 0) / w ** 2) - 1.0) * min(L, w)
    reg = np.maximum(np.minimum(hs[0], hs[1]), lens)
    d[sl] = np.maximum(d[sl], -reg.astype(np.float32))


t_ = time.time()
surf = np.argwhere(np.abs(d) < 0.55 * H)
P = ORIG + surf * H
body_pts = P[(P[:, 2] > 0.012) & (P[:, 2] < 0.262) & (P[:, 0] < X_STEP_END)]
horn_pts = P[(P[:, 0] > X_STEP_END + 0.005) & (P[:, 0] < X_TIP - 0.02)]
for c in body_pts[RNG.choice(len(body_pts), 260, replace=False)]:
    dent(c, RNG.uniform(0.006, 0.016), RNG.uniform(0.00025, 0.0008))
for c in horn_pts[RNG.choice(len(horn_pts), 70, replace=False)]:
    # hammer blows across the horn: short along the horn, wider around it
    dent(c, RNG.uniform(0.004, 0.009), RNG.uniform(0.00015, 0.00035), (0.7, 1.3, 1.3))
# chips along the face edges: more toward the heel end and beside the step, where work hangs off
for side in (-1, 1):
    n = 7 if side < 0 else 5
    xc = np.concatenate([RNG.uniform(X_HEEL + 0.006, X_HEEL + 0.08, n // 2),
                         RNG.uniform(X_STEP - 0.11, X_STEP - 0.006, n - n // 2)])
    for x in xc:
        chip((x, side * FACE_HW, TOP), (1, 0, 0), (0, 0, 1), (0, side, 0),
             RNG.uniform(0.003, 0.010), RNG.uniform(0.0015, 0.0035), 0.006)
for y in RNG.uniform(-FACE_HW + 0.006, FACE_HW - 0.006, 3):
    chip((X_HEEL, y, TOP), (0, 1, 0), (0, 0, 1), (-1, 0, 0), RNG.uniform(0.003, 0.007), RNG.uniform(0.0012, 0.0025), 0.005)
for y in (-0.028, 0.02):
    chip((X_STEP, y, TOP), (0, 1, 0), (0, 0, 1), (1, 0, 0), RNG.uniform(0.004, 0.008), RNG.uniform(0.0015, 0.003), 0.005)
# a slight lip where the heel end has been struck over years (mushrooming)
sl = window((X_HEEL, 0.0, TOP - 0.004), 0.056)
Xl, Yl, Zl = xs[sl[0]][:, None, None], ys[sl[1]][None, :, None], zs[sl[2]][None, None, :]
qx, qy, qz = np.abs(Xl - (X_HEEL + 0.002)) - 0.003, np.abs(Yl) - (FACE_HW - 0.003), np.abs(Zl - (TOP - 0.0045)) - 0.0035
lip = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2 + np.maximum(qz, 0) ** 2) - 0.0018
d[sl] = smin(d[sl], lip.astype(np.float32), 0.003)
print(f"[anvil] dents/chips {time.time()-t_:.1f}s")

# ------------------------------------------------------------------ 8. mesh
t_ = time.time()
band = 3 * H
g = vdb.FloatGrid(background=band)
g.copyFromArray(np.clip(d, -band, band))
pts, tris, quads = g.convertToPolygons(isovalue=0.0, adaptivity=float(os.environ.get("ANVIL_ADAPT", "0.02")))
pts = ORIG + pts.astype(np.float64) * H
faces = [tuple(f) for f in tris.tolist()] + [tuple(f) for f in quads.tolist()]
me = bpy.data.meshes.new("anvil")
me.from_pydata(pts.tolist(), [], faces)
me.validate()
# outward winding (OpenVDB's order depends on the sign convention)
co = np.array([v.co[:] for v in me.vertices])
vol = 0.0
for f in me.polygons:
    v = [co[i] for i in f.vertices]
    for j in range(1, len(v) - 1):
        vol += np.dot(v[0], np.cross(v[j], v[j + 1])) / 6
if vol < 0:
    me.flip_normals()
ob = bpy.data.objects.new("anvil", me)
bpy.context.scene.collection.objects.link(ob)
print(f"[anvil] mesh {len(me.vertices)} verts {len(me.polygons)} faces {time.time()-t_:.1f}s")


# ------------------------------------------------------------------ 9. masks sampled from the field
def trilinear(F, p):
    u = (p - ORIG) / H
    i = np.clip(np.floor(u).astype(int), 0, np.array(F.shape) - 2)
    f = np.clip(u - i, 0, 1)
    out = np.zeros(len(p), np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out += w * F[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
    return out


t_ = time.time()
dc = np.clip(d, -0.02, 0.02)
small = trilinear(box_blur(dc, max(1, int(round(0.0025 / H)))) - dc, pts)
large = trilinear(box_blur(dc, max(2, int(round(0.009 / H)))) - dc, pts)
del dc
me.update()
nrm = np.array([v.normal[:] for v in me.vertices])
px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]
cvx = np.clip(small / 0.0006, 0, 1)                             # convex edges
cav = np.clip(-large / 0.0018, 0, 1)                            # recesses, fillets, under the horn
in_hole = (np.abs(px - HARDY[0]) < HARDY[1] + 0.001) & (np.abs(py) < HARDY[1] + 0.001)
in_hole |= np.hypot(px - PRITCHEL[0], py) < PRITCHEL[1] + 0.001
cav = np.where(in_hole & (pz < TOP - 0.002), np.maximum(cav, 0.85), cav)
face = ((pz > TOP - 0.0022) & (nrm[:, 2] > 0.96) & (px < X_STEP)).astype(np.float32)
# where hands, work and hammers touch: the face and its edges, the table, the top and tip of the horn, the heel top
upper = np.clip((pz - 0.245) / 0.03, 0, 1)
horn_top = ((px > X_STEP_END) & (nrm[:, 2] > 0.35)).astype(np.float32) * np.clip((px - X_STEP_END) / 0.06, 0, 1)
wear = np.maximum.reduce([face, cvx * upper * (px < X_STEP_END + 0.01), horn_top * np.clip(nrm[:, 2], 0, 1),
                          ((pz > STEP_TOP - 0.002) & (nrm[:, 2] > 0.9) & (px > X_STEP)).astype(np.float32) * 0.7])
# the face's margin (pitted, darker where the hammer rarely polishes it)
fd = np.minimum.reduce([FACE_HW - np.abs(py), px - X_HEEL, X_STEP - px])
fedge = np.clip(1.0 - fd / 0.012, 0, 1) * face
for name, val in (("cvx", cvx), ("cav", cav), ("wear", wear), ("face", face), ("fedge", fedge)):
    at = me.attributes.new(name, "FLOAT", "POINT")
    at.data.foreach_set("value", val.astype(np.float32))
del d
print(f"[anvil] masks {time.time()-t_:.1f}s")

for p in me.polygons:
    p.use_smooth = True

# ------------------------------------------------------------------ 10. material
mat = bpy.data.materials.new("anvil_steel")
mat.use_nodes = True
nt = mat.node_tree
N, L_ = nt.nodes, nt.links
pr = N["Principled BSDF"]


def node(t, **inputs):
    n = N.new(t)
    for k, v in inputs.items():
        n.inputs[k].default_value = v
    return n


def link(a, b):
    L_.new(a, b)


def m_(op, a, b=None, clamp=True):
    n = N.new("ShaderNodeMath")
    n.operation, n.use_clamp = op, clamp
    for i, x in enumerate((a, b)):
        if x is None:
            continue
        if isinstance(x, (int, float)):
            n.inputs[i].default_value = x
        else:
            link(x, n.inputs[i])
    return n.outputs[0]


def mr(x, fmin, fmax, tmin=0.0, tmax=1.0, smooth=False):
    n = node("ShaderNodeMapRange")
    if smooth:
        n.interpolation_type = "SMOOTHSTEP"
    n.inputs["From Min"].default_value, n.inputs["From Max"].default_value = fmin, fmax
    n.inputs["To Min"].default_value, n.inputs["To Max"].default_value = tmin, tmax
    link(x, n.inputs["Value"])
    return n.outputs["Result"]


def attr(name):
    n = N.new("ShaderNodeAttribute")
    n.attribute_type, n.attribute_name = "GEOMETRY", name
    return n.outputs["Fac"]


def noise(vec, scale, detail=4.0, rough=0.55, dist=0.0):
    n = node("ShaderNodeTexNoise", Scale=scale, Detail=detail, Roughness=rough, Distortion=dist)
    link(vec, n.inputs["Vector"])
    return n.outputs["Fac"]


def mix_col(fac, a, b):
    n = N.new("ShaderNodeMix")
    n.data_type = "RGBA"
    link(fac, n.inputs["Factor"])
    for sock, v in ((n.inputs[6], a), (n.inputs[7], b)):
        if isinstance(v, tuple):
            sock.default_value = v
        else:
            link(v, sock)
    return n.outputs[2]


def mix_f(fac, a, b):
    n = N.new("ShaderNodeMix")
    link(fac, n.inputs["Factor"])
    for sock, v in ((n.inputs[2], a), (n.inputs[3], b)):
        if isinstance(v, (int, float)):
            sock.default_value = v
        else:
            link(v, sock)
    return n.outputs[0]


tc = N.new("ShaderNodeTexCoord")
P_ = tc.outputs["Object"]
sep = N.new("ShaderNodeSeparateXYZ")
link(P_, sep.inputs["Vector"])
A_cvx, A_cav, A_wear, A_face = attr("cvx"), attr("cav"), attr("wear"), attr("face")

n_patch = noise(P_, 7.0, 5.0, 0.6)            # scale patches
n_mid = noise(P_, 22.0, 6.0, 0.6)
n_fine = noise(P_, 160.0, 4.0, 0.5)
n_rust = noise(P_, 14.0, 8.0, 0.65, 0.3)
A_fedge = attr("fedge")


def voronoi(scale, feature="F1", warp=0.0):
    v = node("ShaderNodeTexVoronoi", Scale=scale, Randomness=1.0)
    v.feature = feature
    vec = P_
    if warp:   # wobble the cell borders so the facets are not clean polygons
        wn = node("ShaderNodeTexNoise", Scale=180.0, Detail=2.0)
        link(P_, wn.inputs["Vector"])
        off = node("ShaderNodeVectorMath")
        off.operation = "MULTIPLY_ADD"
        link(wn.outputs["Color"], off.inputs[0])
        off.inputs[1].default_value = (warp, warp, warp)
        link(P_, off.inputs[2])
        vec = off.outputs["Vector"]
    link(vec, v.inputs["Vector"])
    return v


def stretched(scale_xyz, rot_z, nscale=1.0):
    mp = node("ShaderNodeMapping")
    mp.inputs["Rotation"].default_value = (0.0, 0.0, rot_z)
    mp.inputs["Scale"].default_value = scale_xyz
    link(P_, mp.inputs["Vector"])
    return noise(mp.outputs["Vector"], nscale, 2.0, 0.5)


# hammer facets: overlapping shallow cups from forging blows (two sizes); rims catch light, cups hold scale
ham1 = voronoi(120.0, warp=0.004)
ham2 = voronoi(45.0, warp=0.008)
ham_h = m_("ADD", m_("MULTIPLY", m_("POWER", ham1.outputs["Distance"], 2.0, clamp=False), 0.6, clamp=False),
           m_("MULTIPLY", m_("POWER", ham2.outputs["Distance"], 2.0, clamp=False), 0.8, clamp=False), clamp=False)
ham_h = m_("MULTIPLY", ham_h, mr(n_patch, 0.25, 0.75, 0.35, 1.0), clamp=False)   # facets show in places
rim = mr(ham_h, 0.3, 0.8)
# rust: only in recesses and a few small blooms; brown tints the scale elsewhere
rust = m_("MULTIPLY", mr(A_cav, 0.35, 0.9), mr(n_rust, 0.40, 0.60))
rust = m_("MAXIMUM", rust, m_("MULTIPLY", mr(n_rust, 0.70, 0.76), 0.55))
rust = m_("MULTIPLY", rust, m_("SUBTRACT", 1.0, A_wear))
# polished wear: the face, horn top and working edges, broken up
wear = m_("MULTIPLY", A_wear, mr(n_mid, 0.32, 0.52))
wear = m_("MAXIMUM", wear, A_face)
# colours (linear): black scale with brown and grey patches, lighter hammer rims, rust, steel
scale_c = mix_col(mr(n_patch, 0.35, 0.65, smooth=True), (0.017, 0.017, 0.018, 1), (0.034, 0.028, 0.023, 1))
scale_c = mix_col(m_("MULTIPLY", rim, 0.7), scale_c, (0.050, 0.049, 0.048, 1))
rust_c = mix_col(n_mid, (0.11, 0.047, 0.020, 1), (0.065, 0.032, 0.016, 1))
# face: satin steel with old staining; its margin is pitted and darker; the horn top and edges are darker steel
face_c = mix_col(mr(n_fine, 0.3, 0.7), (0.56, 0.56, 0.57, 1), (0.44, 0.44, 0.45, 1))
face_c = mix_col(mr(n_patch, 0.45, 0.72, 0, 0.45), face_c, (0.26, 0.26, 0.27, 1))
edge_pit = m_("MULTIPLY", A_fedge, mr(noise(P_, 90.0, 6.0, 0.7), 0.35, 0.6))
face_c = mix_col(edge_pit, face_c, (0.12, 0.115, 0.11, 1))
work_c = mix_col(mr(n_mid, 0.3, 0.7), (0.36, 0.36, 0.37, 1), (0.24, 0.24, 0.25, 1))
steel_c = mix_col(A_face, work_c, face_c)
# fine scratches on the face in two working directions
scr1 = mr(stretched((3.0, 1400.0, 1400.0), 0.15), 0.70, 0.80)
scr2 = mr(stretched((3.0, 1400.0, 1400.0), -0.9), 0.72, 0.82)
scr = m_("MULTIPLY", m_("MAXIMUM", scr1, scr2), A_face)
steel_c = mix_col(m_("MULTIPLY", scr, 0.5), steel_c, (0.70, 0.70, 0.71, 1))
col = mix_col(rust, scale_c, rust_c)
col = mix_col(wear, col, steel_c)
link(col, pr.inputs["Base Color"])
rough = mix_f(m_("MULTIPLY", rim, 0.8), mr(n_mid, 0.3, 0.7, 0.55, 0.72), 0.44)
rough = mix_f(rust, rough, 0.88)
r_steel = mix_f(A_face, mr(n_fine, 0.3, 0.7, 0.32, 0.48), mr(n_patch, 0.3, 0.7, 0.34, 0.50))
r_steel = mix_f(edge_pit, r_steel, 0.6)
r_steel = mix_f(scr, r_steel, 0.2)
rough = mix_f(wear, rough, r_steel)
link(rough, pr.inputs["Roughness"])
metal = mix_f(rust, mix_f(rim, 0.45, 0.65), 0.0)
metal = mix_f(wear, metal, 1.0)
link(metal, pr.inputs["Metallic"])

# bump: hammer facets and pitting on the body; scratches, dings and a pitted margin on the face; the weld seam
pits = m_("MULTIPLY", mr(noise(P_, 320.0, 3.0, 0.6, 0.4), 0.71, 0.77), mr(n_mid, 0.45, 0.6))
grain = m_("ADD", m_("MULTIPLY", n_fine, 0.6), m_("MULTIPLY", noise(P_, 600.0, 3.0, 0.5), 0.4), clamp=False)
body_h = m_("ADD", m_("MULTIPLY", grain, 0.35, clamp=False), m_("MULTIPLY", ham_h, 1.4, clamp=False), clamp=False)
body_h = m_("SUBTRACT", body_h, pits, clamp=False)
vd = voronoi(90.0)
dings = m_("MULTIPLY", mr(vd.outputs["Distance"], 0.0, 0.12, 1.0, 0.0, smooth=True), mr(noise(P_, 40.0), 0.5, 0.62))
vp = voronoi(700.0)
margin_pits = m_("MULTIPLY", mr(vp.outputs["Distance"], 0.0, 0.35, 1.0, 0.0, smooth=True), edge_pit)
face_h = m_("SUBTRACT", m_("MULTIPLY", scr, -0.3, clamp=False), m_("ADD", dings, margin_pits), clamp=False)
seam = m_("MULTIPLY", mr(sep.outputs["Z"], 0.2595, 0.2610, 0, 1, smooth=True),
          mr(sep.outputs["Z"], 0.2620, 0.2610, 0, 1, smooth=True))
seam = m_("MULTIPLY", seam, mr(sep.outputs["X"], X_STEP - 0.004, X_STEP - 0.012))
height = mix_f(A_face, body_h, face_h)
height = m_("SUBTRACT", height, m_("MULTIPLY", seam, 1.5, clamp=False), clamp=False)
bump = node("ShaderNodeBump", Distance=0.0012)
link(mix_f(A_face, mix_f(A_wear, 0.7, 0.35), 0.12), bump.inputs["Strength"])
link(height, bump.inputs["Height"])
link(bump.outputs["Normal"], pr.inputs["Normal"])

ob.data.materials.append(mat)
print(f"[anvil] total {time.time()-T0:.1f}s")
