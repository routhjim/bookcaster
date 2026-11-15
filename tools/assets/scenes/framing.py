"""Camera framing from a character's measured body (stdlib; used by the editor job and by the tests).

Frame: the set frame of SETS.md (metres, X east, Y north, Z up, floor at Z = 0). A camera is a position, a look-at
target and a focal length on a 36 mm wide sensor whose height follows the still's aspect (1152x640: 20 mm).

A subject is what the editor measured on the posed character (all set-frame metres):
  {"feet_z", "eye_z", "head_top_z", "chin_z", "waist_z", "eye_xy", "chest_xy", "pelvis_xy", "facing": [dx, dy]}

Shots computed from it (the subject's own size decides distance and height, nothing is guessed):
  close   head and collar: frame height = CLOSE_HEADS x head height (head top - chin), eye line on the upper third
  medium  waist up: frame bottom just under the waist, eye line at MEDIUM_EYE_LINE from the top
  full    the whole figure: feet to head top with a margin
For each, `az` turns the camera around the subject from straight in front (0; positive = toward the subject's own
left), `el` raises it (degrees above the eye line, seen from the subject), `screen_x` puts the subject off centre
(0.5 = centre; < 0.5 = left of frame), and `focal_mm` picks the lens; the distance follows from the lens.
"""

import math

SENSOR_W = 36.0
CLOSE_HEADS = 2.1          # frame height in head heights for a close-up (head + collar)
CLOSE_EYE_LINE = 1 / 3     # eye line from the top of frame
MEDIUM_EYE_LINE = 0.28
MEDIUM_BELOW_WAIST = 0.06  # metres of frame under the waist line
FULL_MARGIN = 0.12         # fraction of the figure's height added above and below


def sensor_h(res):
    return SENSOR_W * res[1] / res[0]


def vfov(focal, res):
    return 2 * math.atan(sensor_h(res) / 2 / focal)


def hfov(focal):
    return 2 * math.atan(SENSOR_W / 2 / focal)


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _facing_angle(f):
    return math.atan2(f[1], f[0])


CABIN_AZ_DEG = {"close": 28.0, "medium": 30.0}      # how far round from the sitter's front a cabin shot stands (at 40 the face check read 58%)
LIE_EL_DEG = 18.0                                  # a lying person is seen from above their side, never level
LIE_CLOSE_EL_DEG = 50.0                            # a close on one: from well above, the face turned up to the camera
LIE_CLOSE_UP_M = 1.4                               # ... and no higher over the head than a room's ceiling leaves
LIE_FRAME_H_M = {"close": 0.65, "medium": 1.25}     # frame height: head and shoulders; head to the hips


def lay(subject, lie, toward=None):
    """A lying person's measured block as the camera reads it. The body lies along the bed, the face up: the side
    the camera comes from (the block's "facing" and the head's forward in plan) is the bed's side nearer `toward`
    (whoever is with them, else the room's middle), square to the body; "lying" keeps the body's own axis (head
    to foot) and length. The heights stay the measured ones (they are the bed's, not a standing person's)."""
    hx, hy = lie["head_xy"]
    ax = _unit([lie["foot_xy"][0] - hx, lie["foot_xy"][1] - hy, 0.0])
    side = [-ax[1], ax[0]]
    ref = toward or [0.0, 0.0]
    if (ref[0] - hx) * side[0] + (ref[1] - hy) * side[1] < 0:
        side = [-side[0], -side[1]]
    return dict(subject, facing=side, head_fwd_xy=side,
                lying={"axis": [ax[0], ax[1]], "length_m": round(math.dist(lie["head_xy"], lie["foot_xy"]), 3)})


def shot(subject, kind, focal_mm, res=(1152, 640), az=0.0, el=0.0, screen_x=0.5):
    """Camera {position, target, focal_mm, distance_m, frame_h_m} for a close / medium / full shot of a subject."""
    head_h = subject["head_top_z"] - subject["chin_z"]
    eye = subject["eye_z"]
    if subject.get("inside") and (kind == "full" or (kind in ("close", "medium") and not subject.get("open_walls"))):
        # seated inside a cabin: there is no room to stand back. A close or medium is taken from just in front of
        # the face with a wide lens (between the sitter and the controls); a full shot from behind the shoulder,
        # the controls and the windscreen ahead of the sitter in view. In a shell whose walls have no back
        # (img3d/pano_shell.py) a close or medium stands back like any other: it sees in through the wall
        f = _unit(list(subject["facing"]) + [0.0])
        ex, ey = subject["eye_xy"]
        left = [-f[1], f[0]]
        if kind == "full":
            pos = [ex - f[0] * 0.9 - left[0] * 0.35, ey - f[1] * 0.9 - left[1] * 0.35, eye + 0.3]
            tgt = [ex + f[0] * 1.5 + left[0] * 0.2, ey + f[1] * 1.5 + left[1] * 0.2, eye - 0.15]
            fl = 24
        else:
            # (measured from the eyes: 0.45 m from the hips was clear, 0.6 m was inside the instrument panel)
            d_ = 0.30 if kind == "close" else 0.34
            pos = [ex + f[0] * d_ + left[0] * 0.05, ey + f[1] * d_ + left[1] * 0.05, eye + 0.03]
            tgt = [ex - f[0] * 0.1, ey - f[1] * 0.1, eye - 0.03]
            fl = 16 if kind == "close" else 13
        return {"position": [round(x, 4) for x in pos], "target": [round(x, 4) for x in tgt], "focal_mm": fl,
                "distance_m": round(math.dist(pos, tgt), 3), "frame_h_m": 0.6}
    if subject.get("open_walls") and kind in ("close", "medium") and not az:
        # in a cabin a camera sees into, a shot of the sitter is taken from the side of their front: the face
        # three-quarter on, the side glass and what is outside behind them (square on, a wall stood behind the
        # head in every shot and the outside was never in a frame with the face, 11-15)
        az = CABIN_AZ_DEG[kind]
    if subject.get("lying") and kind in ("close", "medium", "full"):
        # lying: the frame is sized in metres along the bed (a head's height in z says nothing here), aimed at the
        # head, the chest or the body's middle, from above the bed's side
        ex, cx, px = subject["eye_xy"], subject["chest_xy"], subject["pelvis_xy"]
        if kind == "close":
            fh, xy = LIE_FRAME_H_M["close"], ex
        elif kind == "medium":
            fh, xy = LIE_FRAME_H_M["medium"], [(ex[0] + cx[0]) / 2, (ex[1] + cx[1]) / 2]
        else:
            fh, xy = subject["lying"]["length_m"] * (1 + 2 * FULL_MARGIN) * res[1] / res[0] * 1.3, px
        cz = subject["waist_z"]
        # (from 18 degrees a close showed a profile behind the cover's hem, a third of the face: every close on a
        # sleeper failed its face check, 11-14)
        el = max(el, LIE_CLOSE_EL_DEG if kind == "close" else LIE_EL_DEG)
    elif kind == "close":
        fh = CLOSE_HEADS * head_h
        cz = eye - fh * (0.5 - CLOSE_EYE_LINE)
        xy = subject["eye_xy"]
    elif kind == "medium":
        bottom = subject["waist_z"] - MEDIUM_BELOW_WAIST
        fh = (eye - bottom) / (1 - MEDIUM_EYE_LINE)
        cz = bottom + fh / 2
        xy = subject["chest_xy"]
    elif kind == "full":
        h = subject["head_top_z"] - subject["feet_z"]
        fh = h * (1 + 2 * FULL_MARGIN)
        cz = subject["feet_z"] + h / 2
        xy = subject["pelvis_xy"]
    else:
        raise ValueError(f"unknown shot kind {kind!r}")
    dist = fh / 2 / math.tan(vfov(focal_mm, res) / 2)
    if subject.get("lying") and kind == "close" and dist * math.sin(math.radians(el)) > LIE_CLOSE_UP_M:
        near = LIE_CLOSE_UP_M / math.sin(math.radians(el))      # nearer with a wider lens, the frame the same
        focal_mm, dist = round(focal_mm * near / dist, 1), near
    # direction from the subject to the camera: its facing turned by az, counter-clockwise in plan (the frame is
    # right-handed with Z up, so positive az swings the camera toward the subject's own left)
    a = _facing_angle(subject["facing"]) + math.radians(az)
    e = math.radians(el)
    d = [math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]
    target = [xy[0], xy[1], cz]
    # off-centre placement: move the aim sideways so the subject lands at screen_x
    if abs(screen_x - 0.5) > 1e-6:
        fw = fh * res[0] / res[1]
        right = [-d[1], d[0], 0.0]          # camera right (the camera looks along -d)
        right = _unit(right) if any(right) else [1.0, 0.0, 0.0]
        off = (0.5 - screen_x) * fw
        target = [target[0] + right[0] * off, target[1] + right[1] * off, target[2]]
    pos = [target[0] + d[0] * dist, target[1] + d[1] * dist, target[2] + d[2] * dist]
    return {"position": [round(x, 4) for x in pos], "target": [round(x, 4) for x in target],
            "focal_mm": focal_mm, "distance_m": round(dist, 3), "frame_h_m": round(fh, 3)}


def project(cam, p, res=(1152, 640)):
    """Set-frame point -> (u, v, depth): u, v in 0..1 from the frame's top-left; depth along the view (m)."""
    d0 = [cam["target"][i] - cam["position"][i] for i in range(3)]
    if not any(abs(x) > 1e-9 for x in d0):
        return None                          # a camera on its own aim point: no view
    f = _unit(d0)
    up0 = [0.0, 0.0, 1.0] if abs(f[2]) < 0.999 else [0.0, 1.0, 0.0]     # straight up or down: another up
    r = _unit([f[1] * up0[2] - f[2] * up0[1], f[2] * up0[0] - f[0] * up0[2], f[0] * up0[1] - f[1] * up0[0]])
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    v = [p[i] - cam["position"][i] for i in range(3)]
    z = sum(v[i] * f[i] for i in range(3))
    if z <= 1e-6:
        return None
    x = sum(v[i] * r[i] for i in range(3))
    y = sum(v[i] * u[i] for i in range(3))
    sw, sh = SENSOR_W, sensor_h(res)
    fx = cam["focal_mm"] * x / z
    fy = cam["focal_mm"] * y / z
    return (0.5 + fx / sw, 0.5 - fy / sh, z)


def check(cam, subject, kind, res=(1152, 640)):
    """Where the subject's key points land in the frame, and the problems a viewer would notice first."""
    xy = subject["pelvis_xy"]
    pts = {"head_top": [subject["eye_xy"][0], subject["eye_xy"][1], subject["head_top_z"]],
           "eyes": [subject["eye_xy"][0], subject["eye_xy"][1], subject["eye_z"]],
           "chin": [subject["eye_xy"][0], subject["eye_xy"][1], subject["chin_z"]],
           "waist": [xy[0], xy[1], subject["waist_z"]],
           "feet": [xy[0], xy[1], subject["feet_z"]]}
    out = {k: (None if project(cam, p, res) is None else [round(c, 3) for c in project(cam, p, res)])
           for k, p in pts.items()}
    probs = []

    def inside(k):
        q = out[k]
        return q is not None and 0 <= q[0] <= 1 and 0 <= q[1] <= 1
    if kind in ("wide", "full"):
        for k in ("head_top", "feet"):
            if not inside(k):
                probs.append(f"{k} out of frame")
    if kind in ("close", "medium"):
        if not inside("eyes"):
            probs.append("eyes out of frame")
        if not inside("head_top"):
            probs.append("head top cut")
    if out["head_top"] and out["feet"]:
        out["figure_h_frac"] = round(out["feet"][1] - out["head_top"][1], 3)
    # facing: the angle between the subject's facing and the direction to the camera (0 = straight at camera)
    to_cam = [cam["position"][0] - xy[0], cam["position"][1] - xy[1]]
    fa = math.degrees(math.atan2(to_cam[1], to_cam[0]) - _facing_angle(subject["facing"]))
    fa = (fa + 180) % 360 - 180
    out["camera_off_facing_deg"] = round(fa, 1)
    if abs(fa) > 100:
        probs.append(f"subject faces away from the camera ({fa:.0f} deg)")
    out["problems"] = probs
    return out
