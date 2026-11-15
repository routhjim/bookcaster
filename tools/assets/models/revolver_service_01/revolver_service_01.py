"""Model script: late-Victorian break-top service revolver, built standing (X = barrel axis toward the muzzle,
Z up, Y across), then laid on its right side (rotated about X) so it rests flat on the floor.

Frame, hammer, trigger and grip are plates cut from 2D outlines (period_kit.plate); barrel and cylinder are turned
along X; flutes and chambers are dark inset parts; the guard and lanyard ring are tubes."""

import math
import os
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

steel = K.metal("blued", tarnish=0.3, polish=0.8)
dark = K.metal("iron", tarnish=0.9, polish=0.0)
walnut = K.wood("walnut", "Z", wear=0.6, gloss=0.4, grime=0.8)
BZ = 0.0                                   # bore axis height in the standing frame
parts = []


def along_x(ob, x, z=BZ, y=0.0):
    ob.location, ob.rotation_euler = (x, y, z), (0, math.pi / 2, 0)
    return ob


# barrel (x 0.045 -> 0.150), rib on top, foresight
bar = K.lathe("barrel", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0105, "c"), (0.012, 0.0100), (0.095, 0.0088, "l"),
                                (0.104, 0.0088, "l"), (0.105, 0.0075, "c"), (0.105, 0.0001, "c")], 0.004), seg=32, mat=steel)
parts.append(along_x(bar, 0.045))
muz = K.lathe("bore", [(0.0001, 0.0), (0.0055, 0.0), (0.0055, 0.002), (0.0001, 0.002)], seg=20, mat=dark)
parts.append(along_x(muz, 0.1495))
parts.append(K.box("rib", (0.102, 0.008, 0.006), (0.098, 0, BZ + 0.0105), bev=0.0015, mat=steel))
parts.append(K.plate("foresight", [(0.140, BZ + 0.012), (0.148, BZ + 0.012), (0.146, BZ + 0.019), (0.141, BZ + 0.019)],
                     0.003, mat=steel, bev=0.0005, seg=1))
# cylinder (x 0.004 -> 0.042)
cyl = K.lathe("cylinder", K.keys([(0.0, 0.0001, "c"), (0.0, 0.0175, "c"), (0.003, 0.0195), (0.035, 0.0195, "l"),
                                  (0.038, 0.0180), (0.038, 0.0001, "c")], 0.002), seg=48, mat=steel)
parts.append(along_x(cyl, 0.004))
for k in range(6):
    a = math.tau * k / 6 + math.pi / 6
    fl = K.box(f"flute{k}", (0.022, 0.0035, 0.004), (0.022, 0.0196 * math.cos(a), BZ + 0.0196 * math.sin(a)),
               (-a, 0, 0), bev=0.0012, seg=2, mat=dark)
    parts.append(fl)
    ch = K.lathe(f"chamber{k}", [(0.0001, 0.0), (0.0048, 0.0), (0.0048, 0.0015), (0.0001, 0.0015)], seg=14, mat=dark)
    parts.append(along_x(ch, 0.0408, BZ + 0.0115 * math.sin(a + math.pi / 6), 0.0115 * math.cos(a + math.pi / 6)))
# frame: top strap over the cylinder, recoil shield, standing breech, lower frame to the grip
frame_o = [(0.055, BZ - 0.012), (0.050, BZ - 0.020), (0.010, BZ - 0.024), (-0.004, BZ - 0.030), (-0.020, BZ - 0.034),
           (-0.034, BZ - 0.030), (-0.040, BZ - 0.016), (-0.036, BZ + 0.004), (-0.020, BZ + 0.014), (-0.004, BZ + 0.019),
           (0.000, BZ + 0.024), (0.050, BZ + 0.024), (0.052, BZ + 0.012), (0.044, BZ + 0.012), (0.044, BZ + 0.020),
           (0.0035, BZ + 0.020), (0.0035, BZ - 0.020), (0.044, BZ - 0.020), (0.044, BZ - 0.012)]
parts.append(K.plate("frame", frame_o, 0.024, mat=steel, bev=0.0015))
# grip frame + walnut panels (bird's head), lanyard ring
grip_o = [(-0.010, BZ - 0.028), (-0.012, BZ - 0.050), (-0.022, BZ - 0.080), (-0.034, BZ - 0.100), (-0.052, BZ - 0.108),
          (-0.066, BZ - 0.100), (-0.068, BZ - 0.086), (-0.058, BZ - 0.066), (-0.048, BZ - 0.040), (-0.040, BZ - 0.020),
          (-0.024, BZ - 0.030)]
parts.append(K.plate("gripframe", grip_o, 0.016, mat=steel, bev=0.0015))
panel_o = [(-0.016, BZ - 0.036), (-0.018, BZ - 0.052), (-0.026, BZ - 0.078), (-0.036, BZ - 0.094), (-0.050, BZ - 0.100),
           (-0.060, BZ - 0.094), (-0.060, BZ - 0.084), (-0.052, BZ - 0.066), (-0.043, BZ - 0.042), (-0.034, BZ - 0.030)]
for s in (-1, 1):
    parts.append(K.plate(f"grip{s}", panel_o, 0.006, y=s * 0.0105, mat=walnut, bev=0.0018, seg=3))
    parts.append(K.lathe(f"screw{s}", [(0.0001, 0.0), (0.003, 0.0), (0.0028, 0.0012), (0.0001, 0.0018)], seg=12, mat=steel))
    sc = parts[-1]
    sc.location, sc.rotation_euler = (-0.038, s * 0.0135, BZ - 0.066), (-s * math.pi / 2, 0, 0)
lan = K.tube("lanyard", [(-0.056 + 0.008 * math.cos(a), 0.0, BZ - 0.112 + 0.008 * math.sin(a)) for a in [math.tau * i / 16 for i in range(16)]],
             0.0016, seg=6, closed=True, mat=steel)
parts.append(lan)
# hammer with a chequered spur
ham_o = [(-0.020, BZ + 0.002), (-0.006, BZ + 0.014), (-0.010, BZ + 0.024), (-0.024, BZ + 0.032), (-0.036, BZ + 0.033),
         (-0.038, BZ + 0.028), (-0.028, BZ + 0.024), (-0.024, BZ + 0.016), (-0.032, BZ + 0.004)]
parts.append(K.plate("hammer", ham_o, 0.008, mat=steel, bev=0.001))
# trigger and round guard
trig = K.sweep("trigger", K.bezier((0.002, 0.0, BZ - 0.026), (0.004, 0.0, BZ - 0.034), (-0.000, 0.0, BZ - 0.042),
                                   (-0.010, 0.0, BZ - 0.048), 12),
               [K.superellipse2d(0.0028 * (1 - 0.3 * k / 12), 0.0022, 3.0, 0.0, 10) for k in range(13)], up=(1, 0, 0), mat=steel)
parts.append(trig)
guard = K.tube("guard", 
               [(-0.004 + 0.020 * math.cos(a), 0.0, BZ - 0.030 + 0.020 * math.sin(a) * 0.9) for a in
                [math.radians(d) for d in range(-10, -195, -8)]], 0.0022, seg=8, mat=steel)
parts.append(guard)
# latch on the top strap
parts.append(K.plate("latch", [(-0.012, BZ + 0.018), (-0.002, BZ + 0.024), (-0.002, BZ + 0.030), (-0.012, BZ + 0.026)],
                     0.026, mat=steel, bev=0.001, seg=1))
# lay it on its right side: rotate about X so +Y (its left side) faces up; the harness puts it on the floor
import bpy  # noqa: E402
from mathutils import Matrix  # noqa: E402
R = Matrix.Rotation(math.pi / 2, 4, "X")
bpy.context.view_layer.update()            # parts placed by .location / .rotation_euler: refresh matrix_world first
for o in parts:
    o.matrix_world = R @ o.matrix_world
