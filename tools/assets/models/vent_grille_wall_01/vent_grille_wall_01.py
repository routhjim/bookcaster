"""Model script: cast-iron wall air grille, front -Y, wall face at y = 0; a dark cavity box behind (in the wall)."""

import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

W, H, T = 0.24, 0.16, 0.012
iron = K.metal("cast_iron", tarnish=0.6, polish=0.3, rust=0.4, soot=0.5)
dark = K.fabric("linen", (0.008, 0.007, 0.006), wear=0.0)
parts = []
B = 0.022
for (x, z, w, h) in ((0, H - B / 2, W, B), (0, B / 2, W, B), (-W / 2 + B / 2, H / 2, B, H - 2 * B), (W / 2 - B / 2, H / 2, B, H - 2 * B)):
    parts.append(K.box(f"border{x:.2f}{z:.2f}", (w, T, h), (x, -T / 2, z), bev=0.004, seg=2, mat=iron))
# lattice: bars both ways inside the border
nx, nz = 8, 5
iw, ih = W - 2 * B, H - 2 * B
for k in range(1, nx):
    x = -iw / 2 + iw * k / nx
    parts.append(K.box(f"vb{k}", (0.006, T * 0.7, ih + 0.002), (x, -T * 0.4, H / 2), bev=0.0015, seg=1, mat=iron))
for k in range(1, nz):
    z = B + ih * k / nz
    parts.append(K.box(f"hb{k}", (iw + 0.002, T * 0.7, 0.006), (0, -T * 0.4, z), bev=0.0015, seg=1, mat=iron))
parts += K.nails("screw", [(-W / 2 + B / 2, -T - 0.0005, H / 2), (W / 2 - B / 2, -T - 0.0005, H / 2)], 0.0045, iron, (0, -1, 0))
# cavity box behind (inside the wall), open toward the grille
parts.append(K.box("cavity", (iw + 0.004, 0.035, ih + 0.004), (0, 0.0175, H / 2), bev=0.0, mat=dark))
