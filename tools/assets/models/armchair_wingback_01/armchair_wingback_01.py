"""Model script: Victorian wingback armchair in worn gold damask (deep-buttoned leather with TUFT = True), front -Y (0.85 x 0.85 x 1.15 m).

Footprint kept for scene placement: flat seat-cushion top at 0.45 m, cushion front face at y = -0.43, 0.54 m between
the arms, arm tops ~0.67 m. Upholstered parts are period_kit.rbox bodies (rounded, bulged); the inner back and the
inner wing faces carry diamond tufting with domed buttons; piping is swept tube along the cushion, arm-front panels
and back edges; brass nail heads run along the seat rail and round the arm fronts."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

# muted gold silk damask (the room's set picture); the woven figure a darker olive-gold
LEATHER = K.fabric("damask", (0.150, 0.100, 0.036), wear=0.8, figure=(0.085, 0.058, 0.020), scale=4.0)
BUTTON = LEATHER
TUFT = False          # the period picture shows plain (not buttoned) damask; True gives the deep-buttoned version
legw = K.wood("mahogany", "Z", wear=0.6, gloss=0.6, grime=0.7)
brass = K.metal("brass", tarnish=0.75, polish=0.5)
parts = []

SEAT_Z, FRONT = 0.45, -0.43
BASE_Z0, BASE_Z1 = 0.20, 0.355
ARM_X0, ARM_X1 = 0.27, 0.41           # inner / outer face of each arm body
BACK_Y = 0.24                         # front of the inner back at seat height


def pipe(name, path, r=0.0045, closed=True):
    return K.tube(name, path, r, seg=8, closed=closed, mat=LEATHER)


# ---- seat base (upholstered box on the frame) with a nail row along the bottom edge
base = K.rbox("base", (0.84, 0.80, BASE_Z1 - BASE_Z0), r=0.03, spacing=0.03, bulge=(0.006, 0.01, 0.0),
              loc=(0, 0.0, (BASE_Z0 + BASE_Z1) / 2), mat=LEATHER)
parts.append(base)
pts, nrm = [], []
for k in range(27):
    pts.append((K.lerp(-0.39, 0.39, k / 26), -0.4105, BASE_Z0 + 0.018)); nrm.append((0, -1, 0))
for s in (-1, 1):
    for k in range(1, 24):
        pts.append((s * 0.4255, K.lerp(-0.39, 0.37, k / 23), BASE_Z0 + 0.018)); nrm.append((s, 0, 0))
if TUFT:
    parts += K.nails("nail_b", pts, 0.0058, brass, nrm)

# ---- loose seat cushion, piped top and bottom edges
CW, CD, CT = 0.54, 0.67, 0.10
cy = FRONT + CD / 2
cush = K.rbox("cushion", (CW, CD, CT), r=0.022, spacing=0.02, bulge=(0.004, 0.006, 0.012), sag=0.008,
              loc=(0, cy, SEAT_Z - CT / 2 - 0.012), mat=LEATHER)
parts.append(cush)
for j, z in enumerate((SEAT_Z - 0.012 - 0.003, SEAT_Z - CT - 0.012 + 0.006)):
    parts.append(pipe(f"cush_pipe{j}", K.rrect_path(CW - 0.004, CD - 0.004, 0.022, 5, "xy", (0, cy, z))))

# ---- arms: body, rolled top overhanging outward, scroll front panel with piping and nails
for s in (-1, 1):
    ax = s * (ARM_X0 + ARM_X1) / 2
    body = K.rbox(f"arm{s}", (ARM_X1 - ARM_X0, 0.74, 0.30), r=0.025, spacing=0.025, bulge=(0.012, 0.0, 0.0),
                  loc=(ax, -0.03, BASE_Z1 + 0.15 - 0.01), mat=LEATHER)
    # rolled arm: a fat roll overhanging outward (the period scroll arm)
    roll = K.rbox(f"roll{s}", (0.19, 0.76, 0.125), r=0.06, spacing=0.018, bulge=(0.006, 0.0, 0.008),
                  loc=(ax + s * 0.022, -0.035, 0.6075), mat=LEATHER)
    panel = K.rbox(f"armfront{s}", (0.165, 0.035, 0.40), r=0.012, spacing=0.02, bulge=(0.0, 0.006, 0.0),
                   loc=(ax + s * 0.008, -0.4125, 0.465), mat=LEATHER)
    K.warp(panel, lambda c: (c.x, c.y, c.z + (0.012 if c.z > 0.19 else 0.0) * 0.0))
    parts += [body, roll, panel]
    prr = K.rrect_path(0.148, 0.39, 0.012, 6, "xz", (ax + s * 0.008, -0.432, 0.468), top_r=0.07)
    parts.append(pipe(f"arm_pipe{s}", prr))
    npts = K.rrect_path(0.118, 0.35, 0.01, 4, "xz", (ax + s * 0.008, -0.4335, 0.462), top_r=0.055)
    if TUFT:
        parts += K.nails(f"nail_a{s}", npts[::2], 0.0050, brass, (0, -1, 0))

# ---- outer back shell with a humped top, piping round its front edge
shell = K.rbox("back_shell", (0.84, 0.13, 0.95), r=0.03, spacing=0.03, bulge=(0.0, 0.01, 0.0),
               loc=(0, 0.355, BASE_Z0 + 0.475), mat=LEATHER)
K.warp(shell, lambda c: (c.x, c.y, c.z + (0.06 * max(0.0, 1 - (c.x / 0.42) ** 2) * K.smooth((c.z + 0.2) / 0.67))))
parts.append(shell)

# ---- inner back: deep-buttoned, raked back
IB_H = 0.72
ib, btn = K.rbox("inner_back", (0.56, 0.14, IB_H), r=0.035, spacing=0.012, bulge=(0.0, 0.03, 0.0),
                 tuft={"face": "-y", "cw": 0.112, "ch": 0.13, "depth": 0.028 if TUFT else 0.0, "u0": 0.0, "v0": 0.03, "margin": 0.05},
                 loc=(0, 0.27, SEAT_Z - 0.04 + IB_H / 2), rot=(-0.10, 0, 0), mat=LEATHER)
K.warp(ib, lambda c: (c.x, c.y, c.z + 0.05 * max(0.0, 1 - (c.x / 0.28) ** 2) * K.smooth((c.z + 0.1) / 0.46)))
parts.append(ib)
if TUFT:
    parts += K.add_buttons("btn_b", btn, ib, r=0.011, mat=BUTTON)

# ---- wings: tufted inner faces, front edge sweeping back down to the arm roll, piped front edge
for s in (-1, 1):
    wg, wb = K.rbox(f"wing{s}", (0.10, 0.40, 0.50), r=0.03, spacing=0.016, bulge=(0.02, 0.0, 0.0),
                    tuft={"face": ("-x" if s > 0 else "+x"), "cw": 0.10, "ch": 0.12, "depth": 0.022 if TUFT else 0.0, "margin": 0.055},
                    loc=(s * 0.36, 0.16, 0.92), mat=LEATHER)

    def wf(c, s=s):
        f = K.smooth((-c.y) / 0.2) * K.smooth((-c.z + 0.05) / 0.3)
        # wings flare outward toward the top and front
        flare = 0.05 * K.smooth((c.z + 0.25) / 0.5) * K.smooth((0.2 - c.y) / 0.4)
        return (c.x + s * flare, c.y + 0.13 * f, c.z)

    K.warp(wg, wf)
    btn_w = [(K.Vector(wf(p)), n) for p, n in wb]
    parts.append(wg)
    if TUFT:
        parts += K.add_buttons(f"btn_w{s}", btn_w, wg, r=0.010, mat=BUTTON)
    # piping down the wing's front edge (follows the warped outline)
    wp = []
    for k in range(20):
        z = K.lerp(0.24, -0.20, k / 19)
        y = -0.20 + 0.012
        x, yy, zz = wf(K.Vector((0.0, y, z)))
        wp.append((s * 0.36 + x, 0.16 + yy - 0.012, 0.92 + zz))
    parts.append(pipe(f"wing_pipe{s}", wp, 0.005, closed=False))

# ---- legs: cabriole front, turned raked back, brass castor cups
for s in (-1, 1):
    fl = K.cabriole(f"fleg{s}", BASE_Z0 - 0.025, knee=0.028, top=0.026, ankle=0.016, foot=0.022, out=0.02, mat=legw)
    fl.location = (s * 0.33, -0.33, 0.025)
    cup = K.lathe(f"fcup{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.020, "c"), (0.004, 0.024), (0.026, 0.024, "l"),
                                       (0.028, 0.0001, "c")], 0.002), seg=20, mat=brass)
    cup.location = (s * 0.33, -0.33 - 0.004, 0.0)
    bl = K.lathe(f"bleg{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.016, "c"), (0.03, 0.017), (0.06, 0.022),
                                      *K.bead(0.08, 0.026, 0.012), (0.10, 0.021), (0.15, 0.025), (0.18, 0.027),
                                      (0.20, 0.0001, "c")], 0.003), seg=20, mat=legw)
    bl.location, bl.rotation_euler = (s * 0.33, 0.34, 0.012), (0.12, 0, 0)
    bcup = K.lathe(f"bcup{s}", K.keys([(0.0, 0.0001, "c"), (0.0, 0.018, "c"), (0.016, 0.019, "l"),
                                        (0.018, 0.0001, "c")], 0.002), seg=20, mat=brass)
    bcup.location = (s * 0.33, 0.34 + 0.0, 0.0)
    parts += [fl, cup, bl, bcup]
