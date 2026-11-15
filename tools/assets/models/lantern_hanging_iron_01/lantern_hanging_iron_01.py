"""Model script: an early medieval hanging lantern, lit, matched to its reference drawing: a round cage of
hammered, pitted black iron ~0.16 m across and ~0.36 m tall to the top of its hanging ring: a dished base with a
riveted band, eight upright straps, a riveted top band, thin amber glass panes with a ripple (real glass: the
candle is seen through them), a conical cap pierced with quatrefoils, a ring and link to hang it by; inside, a
pillar candle with a wick and a flame (the flame is the only thing that glows). Origin at the base.

SPEC["parts"] (optional): build only these groups ("cage", "glass", "candle", "flame") so a hybrid can take the
exact parts from here (glass, candle, flame) and the rest from elsewhere.
"""

import math
import os
import random
import sys

import bmesh
from mathutils import Vector, noise

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import forge_kit as F  # noqa: E402

K = F.K
rng = random.Random(8)
PARTS = set(SPEC.get("parts") or ("cage", "glass", "candle", "flame"))  # noqa: F821
R = 0.075
Z0, Z1 = 0.024, 0.245
NS = 8                                               # upright straps
irons, glass, wax = [], [], []


def hammer(ob, amp=0.0006, scale=70.0, seed=0.0):
    """Hammered, pitted surface as geometry: small radial dents (keeps the part's placement)."""
    for v in ob.data.vertices:
        p = v.co
        n = noise.noise(Vector((p.x * scale + seed, p.y * scale, p.z * scale)))
        r = math.hypot(p.x, p.y)
        if r > 1e-4:
            v.co.x += p.x / r * amp * n
            v.co.y += p.y / r * amp * n
    ob.data.update()
    return ob


def ring_band(name, z, h, r_in, r_out, seg=96):
    b = K.lathe(name, [(r_in, z - h / 2), (r_out, z - h / 2), (r_out + 0.0005, z), (r_out, z + h / 2),
                       (r_in, z + h / 2), (r_in, z - h / 2)], seg=seg, cap0=False, cap1=False)
    return hammer(b, 0.0005, 90.0, z * 50)


if "cage" in PARTS:
    base = K.lathe("base", [(0.0001, 0.0), (0.05, 0.0), (0.066, 0.005), (R + 0.006, 0.011), (R + 0.008, Z0),
                            (R - 0.004, Z0), (R - 0.006, 0.014), (0.0, 0.014)], seg=64, cap1=False)
    irons.append(hammer(base, 0.0004, 60.0, 3.0))
    irons.append(ring_band("band_low", Z0 + 0.004, 0.016, R - 0.003, R + 0.005))
    irons.append(ring_band("band_mid", 0.135, 0.008, R - 0.002, R + 0.004))
    irons.append(ring_band("band_top", Z1 - 0.004, 0.018, R - 0.003, R + 0.006))
    for i in range(NS):
        a = math.tau * (i + 0.5) / NS
        s = K.box("strap", (0.010, 0.0045, Z1 - Z0 + 0.008),
                  loc=(math.cos(a) * (R + 0.0035), math.sin(a) * (R + 0.0035), (Z0 + Z1) / 2), rot=(0, 0, a + math.pi / 2),
                  bev=0.0012, seg=2)
        bm = bmesh.new()                               # more rows so the hammering shows along the strap
        bm.from_mesh(s.data)
        bmesh.ops.subdivide_edges(bm, edges=[e for e in bm.edges if abs(e.verts[0].co.z - e.verts[1].co.z) > 0.05],
                                  cuts=24)
        bm.to_mesh(s.data)
        bm.free()
        for v in s.data.vertices:                      # local dents across the strap face
            v.co.y += 0.0005 * noise.noise(Vector((v.co.x * 120, i * 7.0, v.co.z * 60)))
        s.data.update()
        irons.append(s)
    # rivets on the bands, one at each strap and one between
    pts, nrm = [], []
    for z in (Z0 + 0.004, 0.135, Z1 - 0.004):
        for i in range(NS * 2):
            a = math.tau * (i + 0.5) / (NS * 2) if z != 0.135 else math.tau * (i // 2 * 2 + 0.5) / (NS * 2)
            if z == 0.135 and i % 2:
                continue
            rr = R + (0.0055 if z != 0.135 else 0.0045) + (0.0045 if i % 2 == 0 else 0.0)
            pts.append((rr * math.cos(a), rr * math.sin(a), z))
            nrm.append((math.cos(a), math.sin(a), 0.0))
    irons += K.nails("rivet", pts, r=0.0042, normal=nrm)

    # conical cap pierced with quatrefoils, a rim with rivets, a knop, ring and link
    CZ = Z1 + 0.006
    top = K.lathe("cone", [(R + 0.013, Z1), (R + 0.013, CZ), (0.024, Z1 + 0.078), (0.012, Z1 + 0.084),
                           (0.0, Z1 + 0.086)], seg=96, cap0=False)
    sol = top.modifiers.new("t", "SOLIDIFY")
    sol.thickness, sol.offset = 0.0016, -1.0
    slope = math.atan2(0.072, R + 0.013 - 0.024)
    for i in range(8):
        a = math.tau * (i + 0.25) / 8
        rr = 0.058
        zc = CZ + (R + 0.013 - rr) / (R + 0.013 - 0.024) * 0.072
        bm = bmesh.new()                              # a quatrefoil prism: four lobes in polar form
        outline = []
        for k in range(48):
            t = math.tau * k / 48
            rq = 0.0064 * (0.62 + 0.38 * abs(math.cos(2 * t)) ** 0.6)
            outline.append((rq * math.cos(t), rq * math.sin(t)))
        lo = [bm.verts.new((x, y, -0.02)) for x, y in outline]
        hi = [bm.verts.new((x, y, 0.02)) for x, y in outline]
        bm.faces.new(list(reversed(lo)))
        bm.faces.new(hi)
        for k in range(48):
            bm.faces.new((lo[k], lo[(k + 1) % 48], hi[(k + 1) % 48], hi[k]))
        c = K.from_bm("quatrefoil", bm)
        c.location = (rr * math.cos(a), rr * math.sin(a), zc)
        nrm = Vector((math.cos(a) * math.sin(slope), math.sin(a) * math.sin(slope), math.cos(slope)))  # cone normal
        c.rotation_mode = "QUATERNION"
        c.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(nrm)
        c.hide_set(True)
        m = top.modifiers.new("hole", "BOOLEAN")
        m.operation, m.object, m.solver = "DIFFERENCE", c, "EXACT"
    irons.append(top)
    irons.append(ring_band("cap_rim", Z1 + 0.003, 0.008, R + 0.004, R + 0.015))
    pts = [((R + 0.016) * math.cos(math.tau * i / 16), (R + 0.016) * math.sin(math.tau * i / 16), Z1 + 0.003)
           for i in range(16)]
    irons += K.nails("cap_rivet", pts, r=0.0035, normal=[(math.cos(math.tau * i / 16), math.sin(math.tau * i / 16), 0)
                                                          for i in range(16)])
    irons.append(K.lathe("knop", [(0.0001, Z1 + 0.082), (0.012, Z1 + 0.082), (0.010, Z1 + 0.090), (0.006, Z1 + 0.094),
                                  (0.0, Z1 + 0.095)], seg=24))
    LZ = Z1 + 0.095
    irons.append(K.tube("link", [(0.0, 0.0065 * math.cos(math.tau * i / 16), LZ + 0.008 + 0.009 * math.sin(math.tau * i / 16))
                                 for i in range(16)], 0.0028, seg=8, closed=True))
    irons.append(hammer(K.tube("ring", [(0.022 * math.cos(math.tau * i / 32), 0.0,
                                         LZ + 0.038 + 0.022 * math.sin(math.tau * i / 32)) for i in range(32)],
                               0.0038, seg=10, closed=True), 0.0004, 80.0, 9.0))

if "glass" in PARTS:
    # thin amber glass, one rippled cylinder inside the straps (ripple as geometry: the glass is not baked)
    bm = bmesh.new()
    NA, NZ = 128, 28
    rows = []
    for j in range(NZ + 1):
        z = K.lerp(Z0 + 0.003, Z1 - 0.003, j / NZ)
        row = []
        for k in range(NA):
            a = math.tau * k / NA
            rp = R - 0.0005 + 0.0007 * noise.noise(Vector((math.cos(a) * 9, math.sin(a) * 9, z * 40)))
            row.append(bm.verts.new((rp * math.cos(a), rp * math.sin(a), z)))
        rows.append(row)
    for j in range(NZ):
        for k in range(NA):
            bm.faces.new((rows[j][k], rows[j][(k + 1) % NA], rows[j + 1][(k + 1) % NA], rows[j + 1][k]))
    g = K.from_bm("glass", bm)
    sm = g.modifiers.new("t", "SOLIDIFY")
    sm.thickness, sm.offset = 0.0018, 0.0
    K.shade(g, 60)
    glass.append(g)

if "candle" in PARTS:
    CR, CT = 0.021, 0.128                              # a pillar candle on the base, half burnt
    wax.append(K.lathe("candle", K.keys([(0.014, 0.0001, "c"), (0.014, CR, "c"), (CT - 0.006, CR, "l"),
                                         (CT - 0.002, CR * 0.97), (CT, CR * 0.88), (CT - 0.005, CR * 0.55),
                                         (CT - 0.008, 0.0001)], 0.002), seg=40))
    for k in range(5):                               # wax runs over the rim
        a = rng.uniform(0, math.tau)
        L = rng.uniform(0.015, 0.05)
        p = [((CR + 0.0003) * math.cos(a), (CR + 0.0003) * math.sin(a), CT - 0.002 - L * t / 8) for t in range(9)]
        wax.append(K.tube(f"run{k}", p, 0.0026, seg=8, r_fn=lambda t: 0.0026 * (1 - 0.4 * t)))
    wick = K.tube("wick", [(0, 0, CT - 0.009), (0.0, 0.0, CT + 0.003), (0.0015, 0.0, CT + 0.008)], 0.0009, seg=6)
    K.assign(wick, K.metal("iron", tarnish=1.0, soot=1.0, name="lantern_wick"))

if "flame" in PARTS:
    CT = 0.128
    fl = K.lathe("flame", K.keys([(CT + 0.002, 0.0001, "c"), (CT + 0.006, 0.0045), (CT + 0.012, 0.0058),
                                  (CT + 0.026, 0.0030), (CT + 0.040, 0.0001)], 0.001), seg=20)
    # hot white core, orange body, a blue foot: the flame is the light source (emissive), not the panes
    kf = K.NK("lantern_flame")
    sz = kf.N.new("ShaderNodeSeparateXYZ")
    kf.L.new(kf.obj, sz.inputs[0])
    t = kf.mr(sz.outputs["Z"], CT + 0.003, CT + 0.03)
    col = kf.mix(t, (0.35, 0.45, 1.0), (1.0, 0.55, 0.16))
    col = kf.mix(kf.mr(sz.outputs["Z"], CT + 0.006, CT + 0.016, 0.0, 1.0), col, (1.0, 0.88, 0.62))
    M_FLAME = kf.out(col, 0.5, 0.0, None, col, 60.0)
    K.assign(fl, M_FLAME)

M_IRON = F.forged_iron("lantern_iron", scale_amt=0.75, rust=0.3, polish=0.25, soot=0.45)
M_GLASS = K.glass((0.86, 0.52, 0.16), 0.06, ior=1.52, name="lantern_amber_glass")
M_WAX = K.ceramic((0.80, 0.70, 0.48), glaze=0.35, craze=0.0, name="lantern_wax")
for o in irons:
    K.assign(o, M_IRON)
for o in glass:
    K.assign(o, M_GLASS)
for o in wax:
    K.assign(o, M_WAX)
