"""Pose library + OpenPose skeleton renderer for ControlNet pose control.

A library of named body poses (tools/poses.json) and a renderer that draws
them as an OpenPose/DWPose skeleton picture (black background, standard
limb colours) for a pose ControlNet, e.g. Qwen-Image-2.1-Fun-Controlnet-Union.
CPU only; needs numpy + opencv + PIL (rtmlib + onnxruntime for `extract`).

    ~/tts-ab/.venv-pose/bin/python -I tools/poses.py list [--json]
    ~/tts-ab/.venv-pose/bin/python -I tools/poses.py render '<figures json>' out.png [--size 1536x864]
    ~/tts-ab/.venv-pose/bin/python -I tools/poses.py sheet out.png
    ~/tts-ab/.venv-pose/bin/python -I tools/poses.py extract image.png name [--desc ...] [--anchor bottom|center]
    ~/tts-ab/.venv-pose/bin/python -I tools/poses.py build      (re-author the built-in poses into poses.json)

Python: `from poses import render_skeleton, list_poses`;
render_skeleton(figures, size=(1536, 864)) -> PIL.Image (RGB).

HOW THE PLANNER CHOOSES A FIGURE
--------------------------------
One figure per visible person:
    {"pose": "sword_thrust_lunge", "box": [x0, y0, x1, y1], "facing": "right",
     "rotate": 0}                                    # rotate optional, degrees
- pose: a name from `list` (each has a one-line description). Pick the pose of
  the beat's key moment (the strike, not the wind-up), from the character's
  own action in the text. Unknown names raise KeyError.
- box: where the person stands, as fractions of the frame (0..1, y down).
  Make it as tall as the person should appear; the skeleton keeps its own
  proportions and is scaled to FIT the box (no stretching), centred
  horizontally. Grounded poses sit on the box's bottom edge (so y1 is the
  floor line); airborne ones (anchor "center": leaping, falling, carried,
  hanging, rolling) are centred in the box. `list` prints each pose's
  aspect (width / height) so the box can match it: a lying pose is ~3x wider
  than tall, a standing one ~0.4.
- facing: which way the person looks/moves on screen. All library poses are
  authored facing right, three-quarter view toward the camera (the near
  side is the person's RIGHT side). "left" mirrors the picture AND swaps the
  left/right joint labels, so the figure still faces the camera (without
  the swap a mirrored skeleton reads as seen from behind). Handedness flips
  with it: a right-handed sword arm becomes the left arm. Two people facing
  each other: the left one "right", the right one "left".
- rotate: optional extra rotation (degrees, clockwise on screen) about the
  box centre, for tumbling/rolling/falling at odd angles.
- Two-person contact (embrace_a + embrace_b): two figures, boxes overlapping
  by ~30-40% of a box width, facings toward each other.

LIMITS
- One person per figure, body only (COCO-18: no hands, feet or face
  landmarks). Props (sword, hammer, helm, cup) are NOT drawn; the prompt and
  references supply them, the skeleton only puts the hands where they go.
- Creatures (the dragon, horses) have no skeleton; give them a depth or
  lineart control or a reference sheet, and pose the PERSON relative to
  them (e.g. carried_limp at the claw box).
- Authored poses are hand-built from a simple 2D body model (standard adult
  proportions); children need a smaller box, not a different pose.
- Extremely foreshortened limbs (pointing straight at the camera) cannot be
  shown in 2D keypoints; pick the 3/4 alternative.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "poses.json")

# OpenPose COCO-18 order
KP_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder",
            "l_elbow", "l_wrist", "r_hip", "r_knee", "r_ankle", "l_hip",
            "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear"]
# left<->right swap used when mirroring
FLIP_IDX = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]
# OpenPose limb sequence (1-based in the reference code), first 17 are drawn
LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
          [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
          [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
          [255, 0, 170], [255, 0, 85]]


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------

_LIB: dict | None = None


def load_library(path: str = LIB_PATH) -> dict:
    global _LIB
    if _LIB is None or path != LIB_PATH:
        with open(path) as f:
            lib = json.load(f)
        if path != LIB_PATH:
            return lib
        _LIB = lib
    return _LIB


def list_poses() -> list[dict]:
    lib = load_library()
    return [{"name": k, "desc": v.get("desc", ""), "aspect": v["aspect"],
             "anchor": v.get("anchor", "bottom")} for k, v in lib.items()]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _place(entry: dict, box, facing: str, rotate: float, W: int, H: int):
    """Unit-box keypoints -> pixel keypoints (list of (x, y) or None)."""
    kps = entry["kps"]
    aw, ah = entry["extent"]               # extent of the unit coords (max = 1)
    pts = []
    for p in kps:
        pts.append(None if p is None else [float(p[0]), float(p[1])])
    if facing == "left":
        pts = [None if pts[FLIP_IDX[i]] is None
               else [aw - pts[FLIP_IDX[i]][0], pts[FLIP_IDX[i]][1]]
               for i in range(18)]
    x0, y0, x1, y1 = box
    bx0, by0, bw, bh = x0 * W, y0 * H, (x1 - x0) * W, (y1 - y0) * H
    if bw <= 0 or bh <= 0:
        raise ValueError(f"empty box {box}")
    s = min(bw / aw, bh / ah)
    ox = bx0 + (bw - aw * s) / 2
    if entry.get("anchor", "bottom") == "bottom":
        oy = by0 + bh - ah * s
    else:
        oy = by0 + (bh - ah * s) / 2
    out = [None if p is None else [ox + p[0] * s, oy + p[1] * s] for p in pts]
    if rotate:
        cx, cy = bx0 + bw / 2, by0 + bh / 2
        c, sn = math.cos(math.radians(rotate)), math.sin(math.radians(rotate))
        out = [None if p is None else
               [cx + (p[0] - cx) * c - (p[1] - cy) * sn,
                cy + (p[0] - cx) * sn + (p[1] - cy) * c] for p in out]
    return out, s


def draw_people(people: list, W: int, H: int, stickwidth: int | None = None,
                radius: int | None = None) -> np.ndarray:
    """DWPose-style drawing (controlnet_aux dwpose/util.draw_bodypose):
    limbs as filled ellipses in full colour, whole canvas * 0.6, then joint
    dots in full colour. RGB uint8 array."""
    import cv2
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    # DWPose draws at stickwidth 4 / radius 4 on ~512 px; scale with the frame
    k = min(W, H) / 512.0
    sw = stickwidth or max(2, round(4 * k))
    rad = radius or max(2, round(4 * k))
    for pts in people:
        for i in range(17):
            a, b = LIMB_SEQ[i][0] - 1, LIMB_SEQ[i][1] - 1
            if pts[a] is None or pts[b] is None:
                continue
            (xa, ya), (xb, yb) = pts[a], pts[b]
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            length = math.hypot(xa - xb, ya - yb)
            angle = math.degrees(math.atan2(ya - yb, xa - xb))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), sw),
                                    int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, COLORS[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for pts in people:
        for i in range(18):
            if pts[i] is None:
                continue
            x, y = pts[i]
            cv2.circle(canvas, (int(x), int(y)), rad, COLORS[i], thickness=-1)
    return canvas


def figures_to_points(figures: list, size=(1536, 864)) -> list:
    W, H = size
    lib = load_library()
    people = []
    for f in figures:
        name = f["pose"]
        if name not in lib:
            raise KeyError(f"unknown pose {name!r}; see `poses.py list`")
        facing = f.get("facing", "right")
        if facing in ("L", "l"):
            facing = "left"
        pts, _ = _place(lib[name], f.get("box", [0.3, 0.1, 0.7, 0.95]),
                        facing, float(f.get("rotate", 0) or 0), W, H)
        people.append(pts)
    return people


def render_skeleton(figures: list, size=(1536, 864)):
    """figures: [{"pose": name, "box": [x0,y0,x1,y1] fractions,
    "facing": "left"|"right", "rotate": deg (optional)}] -> PIL RGB image."""
    from PIL import Image
    W, H = size
    return Image.fromarray(draw_people(figures_to_points(figures, size), W, H))


def contact_sheet(out_path: str, cols: int = 8, cell=(240, 260)):
    from PIL import Image, ImageDraw, ImageFont
    lib = load_library()
    names = list(lib)
    rows = (len(names) + cols - 1) // cols
    cw, ch = cell
    lab = 22
    sheet = Image.new("RGB", (cols * cw, rows * (ch + lab)), (0, 0, 0))
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(sheet)
    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        pts, _ = _place(lib[n], [0.06, 0.06, 0.94, 0.94], "right", 0, cw, ch)
        img = draw_people([pts], cw, ch, stickwidth=3, radius=3)
        tile = Image.fromarray(img)
        x, y = c * cw, r * (ch + lab)
        sheet.paste(tile, (x, y + lab))
        d.rectangle([x, y, x + cw - 1, y + ch + lab - 1], outline=(70, 70, 70))
        d.text((x + 4, y + 3), f"{i + 1}. {n}", fill=(255, 255, 255), font=font)
    sheet.save(out_path)
    return out_path


# --------------------------------------------------------------------------
# Authoring: a small 2D body model -> keypoints
# --------------------------------------------------------------------------
# Angles are screen directions in degrees: 0 = forward (right), 90 = down,
# 180 = back (left), 270 = up. Proportions: standing height ~1.0.

# calibrated against rtmlib DWPose output on a full-body figure
BODY = dict(torso=0.33, shoulder=0.12, hip=0.065, uarm=0.18, farm=0.15,
            thigh=0.25, shin=0.235, head=0.115)


def _v(a):
    r = math.radians(a)
    return np.array([math.cos(r), math.sin(r)])


def _rot(v, a):
    r = math.radians(a)
    c, s = math.cos(r), math.sin(r)
    return np.array([v[0] * c - v[1] * s, v[0] * s + v[1] * c])


def build_pose(spec: dict):
    """spec keys: t (neck->pelvis angle, 90 upright), yaw (body turn toward
    facing, 0 = front, 45 = 3/4), hyaw (head yaw, default yaw), tilt (head
    bend: + = chin down/forward, - = look up), ra/la/rl/ll = [upper, lower]
    angles (+ optional [.., k_upper, k_lower] length factors), omit = names."""
    B = BODY
    yaw = spec.get("yaw", 45)
    cy = math.cos(math.radians(yaw))
    t = spec.get("t", 90)
    N = np.zeros(2)
    P = N + _v(t) * B["torso"] * spec.get("tk", 1.0)
    u = -_v(t)
    s = np.array([-u[1], u[0]])
    k = [None] * 18
    k[1] = N
    k[2] = N - s * B["shoulder"] * cy
    k[5] = N + s * B["shoulder"] * cy
    k[8] = P - s * B["hip"] * cy
    k[11] = P + s * B["hip"] * cy

    def limb(root, seg, l1, l2):
        a1, a2 = seg[0], seg[1]
        f1 = seg[2] if len(seg) > 2 else 1.0
        f2 = seg[3] if len(seg) > 3 else 1.0
        j1 = root + _v(a1) * l1 * f1
        j2 = j1 + _v(a2) * l2 * f2
        return j1, j2

    k[3], k[4] = limb(k[2], spec.get("ra", [95, 92]), B["uarm"], B["farm"])
    k[6], k[7] = limb(k[5], spec.get("la", [85, 88]), B["uarm"], B["farm"])
    k[9], k[10] = limb(k[8], spec.get("rl", [92, 90]), B["thigh"], B["shin"])
    k[12], k[13] = limb(k[11], spec.get("ll", [88, 90]), B["thigh"], B["shin"])

    hy = spec.get("hyaw", yaw)
    hu = _rot(u, spec.get("tilt", 0))
    hs = np.array([-hu[1], hu[0]])
    c, sn = math.cos(math.radians(hy)), math.sin(math.radians(hy))
    C = N + hu * B["head"]
    k[0] = C + hs * 0.06 * sn
    E = C + hu * 0.022
    k[14] = E + hs * (-0.026 * c + 0.04 * sn)
    k[15] = E + hs * (0.026 * c + 0.04 * sn)
    k[16] = C + hu * 0.012 + hs * (-0.055 * c - 0.02 * sn)
    k[17] = C + hu * 0.012 + hs * (0.055 * c - 0.02 * sn)
    if hy > 30:
        k[17] = None
    if hy < -30:
        k[16] = None
    for name in spec.get("omit", []):
        k[KP_NAMES.index(name)] = None
    extra = [C + hu * 0.09]                       # top of head
    for i in (10, 13):                              # soles below the ankles
        extra.append(k[i] + np.array([0, 0.03]))
    return k, extra


def normalise(kps, extra=(), margin=0.02):
    """Keypoints -> unit coords: min corner at 0, larger extent = 1."""
    pts = [p for p in kps if p is not None] + list(extra)
    arr = np.array(pts, dtype=float)
    lo = arr.min(0) - margin
    hi = arr.max(0) + margin
    ext = hi - lo
    sc = float(ext.max())
    out = [None if p is None else [round(float((p[0] - lo[0]) / sc), 4),
                                   round(float((p[1] - lo[1]) / sc), 4)]
           for p in kps]
    extent = [round(float(ext[0] / sc), 4), round(float(ext[1] / sc), 4)]
    return out, extent


# name: (description, anchor, spec)
SPECS: dict[str, tuple[str, str, dict]] = {
    # --- basic stances and movement
    "standing": ("standing at ease, arms at sides", "bottom",
                 dict(ra=[95, 92], la=[84, 88], rl=[91, 90], ll=[88, 90])),
    "standing_front": ("standing facing the camera, arms at sides", "bottom",
                       dict(yaw=0, ra=[100, 95], la=[80, 85], rl=[93, 90], ll=[87, 90])),
    "walking": ("walking, mid-stride", "bottom",
                dict(yaw=60, ra=[110, 100], la=[72, 62], rl=[72, 98], ll=[106, 122])),
    "running": ("running, leaning forward, knee high", "bottom",
                dict(yaw=60, t=102, tilt=-12, ra=[135, 60], la=[60, 290],
                     rl=[25, 100], ll=[120, 160])),
    "kneeling": ("kneeling on one knee, the other knee up", "bottom",
                 dict(yaw=50, ra=[85, 80], la=[70, 60], rl=[90, 180], ll=[2, 90])),
    "kneeling_both": ("kneeling on both knees, upright, hands on thighs", "bottom",
                      dict(yaw=50, ra=[95, 45], la=[85, 40], rl=[92, 180], ll=[88, 178])),
    "crouching": ("crouching low on both feet", "bottom",
                  dict(yaw=50, t=100, tilt=-10, ra=[75, 70], la=[65, 75],
                       rl=[348, 108], ll=[355, 102])),
    "sitting_chair": ("sitting on a chair or rock, hands on thighs", "bottom",
                      dict(yaw=50, t=88, ra=[88, 20], la=[80, 10], rl=[2, 90], ll=[-6, 86])),
    "sitting_ground": ("sitting on the ground, legs out, leaning back on hands", "bottom",
                       dict(yaw=55, t=80, ra=[115, 100], la=[105, 95], rl=[0, 3], ll=[-6, 8])),
    "sitting_by_fire": ("sitting on the ground, knees up, hands held out to a fire", "bottom",
                        dict(yaw=50, t=92, tilt=8, ra=[50, 5], la=[40, 0],
                             rl=[336, 88], ll=[342, 90])),
    "slumped": ("slumped against a wall, legs out, head hanging (hurt, exhausted)", "bottom",
                dict(yaw=40, t=82, tilt=40, ra=[95, 70], la=[85, 60], rl=[2, 10], ll=[-8, 30])),
    "lying_back": ("lying flat on the back, head to the right", "bottom",
                   dict(yaw=70, t=180, hyaw=-70, ra=[172, 180], la=[188, 182],
                        rl=[180, 180], ll=[176, 182])),
    "sleeping": ("asleep on the side, knees drawn up, head right", "bottom",
                 dict(yaw=60, t=180, hyaw=50, tilt=10, ra=[60, 0], la=[90, 20],
                      rl=[130, 190], ll=[120, 180])),
    "falling_back": ("falling backward, arms flung forward", "center",
                     dict(yaw=50, t=45, tilt=30, ra=[330, 300], la=[290, 260],
                          rl=[15, 50], ll=[60, 100])),
    "falling_forward": ("tripping, falling forward, arms out to catch", "center",
                        dict(yaw=55, t=160, tilt=-35, ra=[40, 50], la=[25, 40],
                             rl=[110, 100], ll=[150, 175])),
    "rolling_tuck": ("tucked in a ball, mid forward roll", "center",
                     dict(yaw=60, t=130, tilt=45, ra=[60, 120], la=[50, 110],
                          rl=[330, 120], ll=[340, 125])),
    "crouch_dodge": ("ducking low to dodge, legs wide, guard up", "bottom",
                     dict(yaw=55, t=125, tilt=-30, ra=[50, 320], la=[160, 130],
                          rl=[20, 95], ll=[140, 125])),
    "leaping": ("leaping through the air, legs split", "center",
                dict(yaw=55, t=105, tilt=-10, ra=[300, 290], la=[190, 200],
                     rl=[8, 75], ll=[140, 170])),
    "crawling": ("crawling on hands and knees", "bottom",
                 dict(yaw=60, t=165, hyaw=40, tilt=-20, ra=[92, 90], la=[86, 88],
                      rl=[90, 180], ll=[98, 178])),
    "climbing": ("climbing a ladder or wall, one hand high, one knee up", "bottom",
                 dict(yaw=60, tilt=-20, ra=[280, 275], la=[340, 300],
                      rl=[335, 95], ll=[95, 95])),
    "pulling_rope": ("leaning back hauling on a rope, arms forward", "bottom",
                     dict(yaw=55, t=68, tilt=10, ra=[8, 2], la=[2, 356],
                          rl=[55, 82], ll=[85, 100])),
    # --- smithy
    "hammer_raised": ("smith with hammer raised overhead, tongs hand forward", "bottom",
                      dict(yaw=45, t=95, ra=[255, 205], la=[70, 10], rl=[100, 90], ll=[76, 90])),
    "hammer_strike": ("smith striking down on the anvil, leaning in", "bottom",
                      dict(yaw=45, t=105, tilt=25, ra=[30, 70], la=[72, 8],
                           rl=[100, 92], ll=[76, 92])),
    "tongs_over_anvil": ("holding a blade in tongs over the anvil, looking down at it", "bottom",
                         dict(yaw=45, t=100, tilt=28, ra=[85, 30], la=[65, 5],
                              rl=[98, 90], ll=[80, 90])),
    "quenching": ("bent forward, plunging a blade down into a trough", "bottom",
                  dict(yaw=50, t=125, tilt=25, ra=[70, 88], la=[62, 84],
                       rl=[98, 90], ll=[78, 92])),
    # --- swords and shields
    "sword_draw": ("drawing a sword from the left hip", "bottom",
                   dict(yaw=40, ra=[65, 30], la=[95, 60], rl=[95, 90], ll=[80, 90])),
    "sword_ready": ("en garde, sword held forward in both hands at the waist", "bottom",
                    dict(yaw=45, t=96, ra=[75, 335], la=[65, 345], rl=[105, 95], ll=[72, 92])),
    "sword_raised_overhead": ("sword raised overhead in both hands", "bottom",
                              dict(yaw=40, tilt=-10, ra=[262, 280], la=[245, 265],
                                   rl=[100, 90], ll=[80, 90])),
    "sword_thrust_lunge": ("deep lunge, sword arm thrust straight forward", "bottom",
                           dict(yaw=55, t=100, ra=[356, 0], la=[200, 215],
                                rl=[22, 95], ll=[138, 150])),
    "sword_slash": ("slashing follow-through, sword arm swept down and across", "bottom",
                    dict(yaw=45, t=100, ra=[25, 45], la=[160, 130],
                         rl=[72, 95], ll=[115, 100])),
    "sword_parry_high": ("parrying high, blade held above the head", "bottom",
                         dict(yaw=45, tilt=-15, ra=[300, 255], la=[255, 310],
                              rl=[80, 95], ll=[112, 95])),
    "sword_block": ("blocking, sword held upright in front of the chest", "bottom",
                    dict(yaw=40, ra=[70, 300], la=[85, 320], rl=[100, 92], ll=[75, 92])),
    "shield_raised": ("shield arm raised before the face, sword cocked low", "bottom",
                      dict(yaw=45, t=100, tilt=-5, ra=[130, 60], la=[10, 285, 1, 0.8],
                           rl=[108, 95], ll=[70, 92])),
    "kneel_with_sword": ("kneeling on one knee, sword point planted, hands on the hilt", "bottom",
                         dict(yaw=50, tilt=20, ra=[72, 335], la=[82, 325],
                              rl=[90, 180], ll=[2, 90])),
    # --- helm
    "helm_put_on": ("lifting a helm down onto the head with both hands", "bottom",
                    dict(yaw=40, ra=[225, 356], la=[315, 186, 1, 0.65])),
    "helm_remove": ("helm just removed, held in both hands in front, looking at it", "bottom",
                    dict(yaw=40, tilt=18, ra=[95, 338], la=[85, 306, 1, 0.65])),
    "helm_under_arm": ("standing with a helm tucked under the near arm", "bottom",
                       dict(yaw=45, ra=[100, 15], la=[84, 88])),
    # --- carried / held
    "carried_limp": ("limp body carried horizontally (in claws or arms), head and limbs hanging", "center",
                     dict(yaw=55, t=180, hyaw=50, tilt=45, ra=[100, 95], la=[80, 90],
                          rl=[170, 120], ll=[165, 110])),
    "hanging_from_claws": ("gripped at the shoulders and dangling, arms and legs hanging", "center",
                           dict(yaw=40, tilt=-10, ra=[100, 95], la=[80, 85],
                                rl=[96, 100], ll=[84, 95])),
    "hanging_by_arms": ("hanging from a ledge or a grip by both raised arms", "center",
                        dict(yaw=40, tilt=-15, ra=[268, 272], la=[262, 268],
                             rl=[95, 100], ll=[85, 95])),
    # --- gestures
    "reaching_up": ("reaching up high with one arm", "bottom",
                    dict(yaw=45, tilt=-25, ra=[275, 272], la=[88, 88], rl=[92, 88], ll=[86, 88])),
    "pointing": ("pointing forward with the arm straight", "bottom",
                 dict(yaw=50, ra=[356, 358], la=[85, 90])),
    "shouting": ("shouting, one hand cupped at the mouth", "bottom",
                 dict(yaw=40, tilt=-15, ra=[40, 273], la=[60, 70], rl=[95, 90], ll=[82, 90])),
    "shading_eyes": ("looking into the distance, hand shading the eyes", "bottom",
                     dict(yaw=45, tilt=-5, ra=[0, 246, 1, 0.85], la=[85, 90])),
    "arms_crossed": ("standing facing the camera, arms crossed", "bottom",
                     dict(yaw=0, ra=[97, 345], la=[83, 195], rl=[93, 90], ll=[87, 90])),
    "praying": ("kneeling on both knees, hands together at the chest, head bowed", "bottom",
                dict(yaw=45, tilt=22, ra=[100, 327], la=[80, 228, 1, 0.75],
                     rl=[92, 180], ll=[88, 178])),
    "bowing": ("bowing from the waist, one hand across the belly", "bottom",
               dict(yaw=50, t=140, tilt=10, ra=[70, 8], la=[100, 95], rl=[95, 90], ll=[84, 90])),
    "cheering": ("cheering, both arms up in a V, facing the camera", "bottom",
                 dict(yaw=0, tilt=-15, ra=[235, 250], la=[305, 290], rl=[96, 92], ll=[84, 88])),
    "raise_cup": ("celebrating, cup raised high in one hand", "bottom",
                  dict(yaw=30, tilt=-10, ra=[290, 275], la=[60, 30], rl=[95, 90], ll=[84, 90])),
    "dancing": ("dancing, one arm up, one out, one knee raised", "bottom",
                dict(yaw=30, tilt=-5, ra=[250, 280], la=[10, 340], rl=[20, 100], ll=[90, 90])),
    "embrace_a": ("embracing (left person of a pair; use with embrace_b facing left)", "bottom",
                  dict(yaw=40, t=95, tilt=10, ra=[25, 345], la=[15, 330], rl=[95, 90], ll=[84, 90])),
    "embrace_b": ("embracing (right person of a pair; place facing left)", "bottom",
                  dict(yaw=40, t=95, tilt=15, ra=[50, 5], la=[35, 355], rl=[95, 90], ll=[84, 90])),
}


def build_library(path: str = LIB_PATH):
    """Re-author SPECS into poses.json, keeping extracted (non-authored) poses."""
    old = {}
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)
    lib = {}
    for name, (desc, anchor, spec) in SPECS.items():
        kps, extra = build_pose(spec)
        norm, extent = normalise(kps, extra)
        lib[name] = {"desc": desc, "anchor": anchor, "source": "authored",
                     "aspect": round(extent[0] / extent[1], 3),
                     "extent": extent, "kps": norm}
    for name, e in old.items():
        if e.get("source") != "authored" and name not in lib:
            lib[name] = e
    with open(path, "w") as f:
        f.write("{\n")
        items = list(lib.items())
        for i, (name, e) in enumerate(items):
            f.write(f"  {json.dumps(name)}: {json.dumps(e)}")
            f.write(",\n" if i < len(items) - 1 else "\n")
        f.write("}\n")
    global _LIB
    _LIB = None
    return lib


# --------------------------------------------------------------------------
# Extraction (rtmlib DWPose-family body model on CPU)
# --------------------------------------------------------------------------

def extract_pose(image_path: str, thr: float = 0.3, mode: str = "balanced"):
    """Largest person in the picture -> (pixel keypoints list, scores)."""
    import cv2
    from rtmlib import Body
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    body = Body(mode=mode, to_openpose=True, backend="onnxruntime", device="cpu")
    kps, scores = body(img)
    if len(kps) == 0:
        raise RuntimeError("no person found")
    best, best_area = 0, -1.0
    for i in range(len(kps)):
        vis = kps[i][scores[i] > thr]
        if len(vis) < 2:
            continue
        # prefer the most complete body (a full figure beats a close-up
        # portrait on a reference sheet), then the biggest
        area = len(vis) * 1e9 + float(np.ptp(vis[:, 0]) * np.ptp(vis[:, 1]))
        if area > best_area:
            best, best_area = i, area
    pts = [[float(x), float(y)] if sc > thr else None
           for (x, y), sc in zip(kps[best], scores[best])]
    return pts, [float(x) for x in scores[best]]


def add_extracted(image_path: str, name: str, desc: str = "",
                  anchor: str = "bottom", facing: str | None = None,
                  path: str = LIB_PATH):
    pts, _ = extract_pose(image_path)
    if pts[1] is None:
        raise RuntimeError("neck/shoulders not found; cannot use this picture")
    # canonical = facing right: decide from the nose (or eyes) vs the neck
    if facing is None:
        head = [p for p in (pts[0], pts[14], pts[15]) if p is not None]
        facing = "right" if not head or np.mean([p[0] for p in head]) >= pts[1][0] else "left"
    if facing == "left":
        xs = [p[0] for p in pts if p is not None]
        m = max(xs) + min(xs)
        pts = [None if pts[FLIP_IDX[i]] is None
               else [m - pts[FLIP_IDX[i]][0], pts[FLIP_IDX[i]][1]] for i in range(18)]
    arr = [None if p is None else np.array(p) for p in pts]
    # margin ~ a head-top allowance in body units
    vis = np.array([p for p in pts if p is not None])
    hgt = float(np.ptp(vis[:, 1])) or 1.0
    extra = []
    if arr[0] is not None and arr[1] is not None:
        extra.append(arr[0] + (arr[0] - arr[1]) * 0.8)
    norm, extent = normalise(arr, extra, margin=0.02 * hgt)
    # normalise() put margin in pixel units; renormalise is fine as-is
    with open(path) as f:
        lib = json.load(f)
    lib[name] = {"desc": desc or f"extracted from {os.path.basename(image_path)}",
                 "anchor": anchor, "source": "extracted",
                 "aspect": round(extent[0] / extent[1], 3), "extent": extent,
                 "kps": norm}
    with open(path, "w") as f:
        f.write("{\n")
        items = list(lib.items())
        for i, (n, e) in enumerate(items):
            f.write(f"  {json.dumps(n)}: {json.dumps(e)}")
            f.write(",\n" if i < len(items) - 1 else "\n")
        f.write("}\n")
    return lib[name]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _size(s: str):
    w, h = s.lower().split("x")
    return int(w), int(h)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("render")
    p.add_argument("figures", help="JSON list of figures (or @file.json)")
    p.add_argument("out")
    p.add_argument("--size", default="1536x864", type=_size)
    p = sub.add_parser("sheet")
    p.add_argument("out")
    p.add_argument("--cols", type=int, default=8)
    sub.add_parser("build")
    p = sub.add_parser("extract")
    p.add_argument("image")
    p.add_argument("name")
    p.add_argument("--desc", default="")
    p.add_argument("--anchor", default="bottom", choices=["bottom", "center"])
    p.add_argument("--facing", choices=["left", "right"],
                   help="which way the person in the picture faces (default: guess)")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        poses = list_poses()
        if a.json:
            print(json.dumps(poses, indent=1))
        else:
            for q in poses:
                print(f"{q['name']:24s} aspect {q['aspect']:5.2f}  {q['anchor']:6s}  {q['desc']}")
            print(f"{len(poses)} poses", file=sys.stderr)
    elif a.cmd == "render":
        src = a.figures
        if src.startswith("@"):
            with open(src[1:]) as f:
                src = f.read()
        figs = json.loads(src)
        if isinstance(figs, dict):
            figs = [figs]
        render_skeleton(figs, a.size).save(a.out)
        print(a.out)
    elif a.cmd == "sheet":
        print(contact_sheet(a.out, cols=a.cols))
    elif a.cmd == "build":
        lib = build_library()
        print(f"{len(lib)} poses -> {LIB_PATH}")
    elif a.cmd == "extract":
        e = add_extracted(a.image, a.name, a.desc, a.anchor, a.facing)
        print(json.dumps({a.name: {k: e[k] for k in ("desc", "aspect", "anchor")}}))


if __name__ == "__main__":
    main()
