"""Model script: black slate temple mantel clock with columns, enamel dial, bezel and hands, front -Y."""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

slate = K.stone((0.018, 0.018, 0.02), vein=None, polish=0.7, name="slate")
gm = K.gilt()
brass = K.metal("brass", tarnish=0.5, polish=0.7)
enamel = K.ceramic((0.66, 0.64, 0.58), glaze=0.9, craze=0.25, name="enamel")
steel = K.metal("blued", tarnish=0.2, polish=0.4)
glassm = K.glass((0.95, 0.95, 0.93), 0.01, name="dial_glass")
W, D = 0.30, 0.13
K.box("plinth", (W, D, 0.025), (0, 0, 0.0125), bev=0.004, seg=2, mat=slate)
K.box("plinth2", (W - 0.02, D - 0.015, 0.018), (0, 0, 0.034), bev=0.003, seg=2, mat=slate)
K.box("body", (0.21, 0.10, 0.19), (0, 0.005, 0.043 + 0.095), bev=0.003, seg=2, mat=slate)
K.box("entab", (0.27, 0.115, 0.022), (0, 0, 0.233 + 0.011), bev=0.003, seg=2, mat=slate)
# low pediment: a triangular prism
import bmesh
bm = bmesh.new()
pts = [(-0.13, 0.255), (0.13, 0.255), (0.0, 0.30)]
vf = [bm.verts.new((x, -0.055, z)) for x, z in pts]
vb = [bm.verts.new((x, 0.055, z)) for x, z in pts]
bm.faces.new(vf[::-1]); bm.faces.new(vb)
for i in range(3):
    j = (i + 1) % 3
    bm.faces.new((vf[i], vf[j], vb[j], vb[i]))
pd = K.from_bm("pediment", bm)
K.assign(pd, slate)
K.bevel(pd, 0.002, 1)
for s in (-1, 1):
    col = K.lathe(f"col{s}", K.keys([(0.043, 0.0001, "c"), (0.043, 0.017, "c"), (0.050, 0.017, "l"), (0.054, 0.013),
                                      (0.20, 0.011), (0.215, 0.013), (0.226, 0.017), (0.233, 0.017, "l"),
                                      (0.233, 0.0001, "c")], 0.003), seg=24, mat=slate)
    col.location = (s * 0.115, -0.04, 0.0)
    for zc in (0.047, 0.222):
        cap = K.lathe(f"cap{s}{zc}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.018, "c"), (0.008, 0.018, "l"),
                                             (0.008, 0.0001, "c")], 0.002), seg=24, mat=gm)
        cap.location = (s * 0.115, -0.04, zc - 0.004)
DZ, DY = 0.15, -0.045
bez = K.lathe("bezel", K.keys([(0.0, 0.0001, "c"), (0.0, 0.058, "c"), (0.004, 0.060), (0.009, 0.056), (0.011, 0.049),
                               (0.011, 0.0001, "c")], 0.0015), seg=48, mat=brass)
bez.location, bez.rotation_euler = (0, DY, DZ), (math.pi / 2, 0, 0)
dial = K.lathe("dial", [(0.0001, 0.0), (0.050, 0.0), (0.050, 0.004), (0.0001, 0.004)], seg=48, mat=enamel)
dial.location, dial.rotation_euler = (0, DY - 0.004, DZ), (math.pi / 2, 0, 0)
for h in range(12):
    a = math.tau * h / 12
    L = 0.010 if h % 3 == 0 else 0.006
    m = K.box(f"mark{h}", (0.0025 if h % 3 else 0.0035, 0.001, L), (0.041 * math.sin(a), DY - 0.0085, DZ + 0.041 * math.cos(a)),
              (0, a, 0), bev=0.0, mat=steel)
for (name, L, a, w) in (("hour", 0.026, 2.1, 0.0035), ("minute", 0.038, -0.6, 0.0022)):
    hd = K.box(name, (w, 0.001, L), (L / 2 * math.sin(a), DY - 0.0095 - (0.001 if name == "minute" else 0), DZ + L / 2 * math.cos(a)),
               (0, a, 0), bev=0.0, mat=steel)
K.lathe("arbor", [(0.0001, 0.0), (0.0025, 0.0), (0.0025, 0.006), (0.0001, 0.006)], seg=12, mat=brass).rotation_euler = (math.pi / 2, 0, 0)
K.bpy.data.objects["arbor"].location = (0, DY - 0.005, DZ)
gl = K.lathe("glass", [(0.0001, 0.0), (0.050, 0.0), (0.040, 0.004), (0.0001, 0.0055)], seg=48, mat=glassm, cap0=True, cap1=False)
gl.location, gl.rotation_euler = (0, DY - 0.010, DZ), (math.pi / 2, 0, 0)
# incised gilt lines on the plinth front and entablature
for z in (0.0125, 0.244):
    K.box(f"gline{z}", (0.24 if z < 0.1 else 0.25, 0.001, 0.0025), (0, -D / 2 - 0.0002 if z < 0.1 else -0.058, z), bev=0.0, mat=gm)
