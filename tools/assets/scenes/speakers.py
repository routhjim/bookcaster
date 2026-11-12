"""Speaker cameras: every shot shows the face of whoever is talking in it. Plan, check from the engine's own facts,
re-frame, and only then let a shot go on to the image model.

    python tools/assets/scenes/speakers.py SCENE.json [--rounds 3]

SCENE.json is the scene tool's spec (README.md) with, per camera:
    "speaker": "<character key>" | null      who talks in this shot (null: nobody; any side may be seen)
    "ots": "<listener key>"                  an over-the-shoulder shot: over the LISTENER's shoulder onto the speaker
and optionally "line": [key, key]: the two people whose line the cameras keep to one side of (default: the first
two characters). Stdlib only; the stills come from scene.py (the engine, through the line).

Rules (general; no per-shot wording):
- Camera placement. A speaker's head must face the camera within MAX_HEAD_DEG, the face must be seen (not covered by
  another person or by furniture) and big enough for the shot's distance. An over-the-shoulder camera stands behind
  the listener's head, offset toward the side of the line the scene has established, aimed at the speaker's eyes.
- Screen direction. The first wide shot sets the side of the line between the two people; every later shot that is
  not a wide stays on that side (a cut across it flips who looks left and who looks right).
- Framing (11-11, the shots a plain key won): in a talk shot that is not a close the LISTENER stays in view (their
  head in the frame; in an over-the-shoulder their head or near shoulder), the speaker's face is at least
  TALK_FACE_H of the frame height for the shot's kind, and nobody is cut by the frame's left, right or top edge (a
  person half in the frame) except the listener of an over-the-shoulder. A shot without a speaker keeps the people
  the director names for it ("show": [keys]) in the frame, uncut. A two-shot favours the speaker: the listener's
  head is not bigger in the frame than the speaker's. A failure is re-framed by a placed camera that
  frames the speaker and the listener (aimed between their faces, nearer the speaker: a two-shot favouring the
  speaker), the passing standpoint nearest the old one.
- Seating (seating.py): a seated person with "seat" is placed from the chair's own seat box before round 0 and
  checked from their measured joints after every round; a failure moves them (the measured error off) and renders
  every camera again.
- The check uses the engine's own facts for each still: the speaker's head angle to the camera (the scene tool's
  measured truth), the visible fraction of points over the speaker's face (each point is seen where the speaker's own
  depth pass is drawn and is what the scene's depth sees there: nothing nearer covers it), and the face's height in
  the frame (head top to chin).
- A shot that fails is re-framed: a camera aimed by the scene tool (close / medium) swings along its arc around the
  speaker toward their face; a placed camera (wide) orbits the speaker; an over-the-shoulder camera is placed again
  from the measured heads. Candidates that cross the line (not wides), leave the room or put another person between
  the camera and the face are dropped. Only the re-framed cameras are rendered again (scene.py --cameras). A shot
  that still fails after the last round is marked "reject" and is not sent to the image model.

Writes <out>/<scene id>/speakers.json (per shot and round: the check, the camera, why it was moved) and puts
"speaker", "speaker_check" and, for a failed shot, "reject" into that scene's result.json shots.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import framing  # noqa: E402
import seating  # noqa: E402

MAX_HEAD_DEG = 45.0
MIN_FACE_VISIBLE = 0.6
MIN_FACE_H = {"wide": 0.03, "full": 0.045, "medium": 0.07, "close": 0.18}
TALK_FACE_H = {"wide": 0.10, "full": 0.10, "medium": 0.10, "close": 0.18}    # a speaker's face in a talk shot
EDGE = 0.01                  # a point this near the frame's edge counts as outside
SIDE_EDGE = 0.03             # left and right: a joint this near the side has the body's outer side past it
ROOM_MARGIN_M = 0.3
PERSON_R_M = 0.22            # a person, for the planner's "is someone between the camera and the face" test


def log(m):
    print(f"[speakers {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ depth pixels (uncompressed scanline EXR, R)
class Exr:
    """The R channel (UE depth, cm) of an uncompressed scanline EXR at chosen pixels, reading only their rows."""

    def __init__(self, path):
        self.b = Path(path).read_bytes()
        b = self.b
        i, hdr = 8, {}
        while b[i] != 0:
            j = b.index(b"\0", i)
            name = b[i:j].decode()
            k = b.index(b"\0", j + 1)
            size = struct.unpack_from("<i", b, k + 1)[0]
            hdr[name] = b[k + 5:k + 5 + size]
            i = k + 5 + size
        i += 1
        v, q, self.ch = hdr["channels"], 0, []
        while v[q] != 0:
            e = v.index(b"\0", q)
            self.ch.append((v[q:e].decode(), struct.unpack_from("<i", v, e + 1)[0]))
            q = e + 17
        if hdr["compression"][0] != 0:
            raise RuntimeError(f"{path}: compressed EXR")
        x0, y0, x1, y1 = struct.unpack("<4i", hdr["dataWindow"])
        self.w, self.h, self.y0 = x1 - x0 + 1, y1 - y0 + 1, y0
        self.offs = struct.unpack_from(f"<{self.h}Q", b, i)
        self.size = {1: 2, 2: 4, 0: 4}

    def at(self, u, v):
        """Depth (cm) at frame position u, v in 0..1, or None outside."""
        if not (0 <= u < 1 and 0 <= v < 1):
            return None
        x, y = int(u * self.w), int(v * self.h)
        o = self.offs[y] + 8
        for name, t in self.ch:
            n = self.size[t]
            if name == "R":
                fmt = "<e" if t == 1 else "<f"
                return struct.unpack_from(fmt, self.b, o + x * n)[0]
            o += self.w * n
        return None


# ------------------------------------------------------------------ geometry
def unit2(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return [v[0] / n, v[1] / n]


def plan_angle(fwd, frm, to):
    d = [to[0] - frm[0], to[1] - frm[1]]
    a = math.degrees(math.atan2(d[1], d[0]) - math.atan2(fwd[1], fwd[0]))
    return abs((a + 180) % 360 - 180)


def side(line, p):
    """+1 / -1: which side of the line a -> b the plan point p is on (0 on it)."""
    (ax, ay), (bx, by) = line
    c = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
    return 0 if abs(c) < 1e-6 else (1 if c > 0 else -1)


def face_points(m):
    """Set-frame points over a measured head's face: three across, four down from the brow to the chin."""
    f = unit2(m["head_fwd_xy"])
    r = [f[1], -f[0]]
    e, cz = m["eye_xy"], m["chin_z"]
    c = [e[0] + f[0] * 0.03, e[1] + f[1] * 0.03]
    zs = [m["eye_z"] + 0.025, m["eye_z"] - 0.02, (m["eye_z"] + cz) / 2 - 0.01, cz + 0.025]
    return [[c[0] + r[0] * s, c[1] + r[1] * s, z] for s in (-0.045, 0.0, 0.045) for z in zs]


def blocked(cam_pos, eye, others):
    """Another person's body (a vertical cylinder from the floor to their head top) between the camera and the eyes."""
    for o in others:
        p, top = o["pelvis_xy"], o["head_top_z"]
        dx, dy = eye[0] - cam_pos[0], eye[1] - cam_pos[1]
        L2 = dx * dx + dy * dy or 1e-9
        t = ((p[0] - cam_pos[0]) * dx + (p[1] - cam_pos[1]) * dy) / L2
        if not 0.05 < t < 0.97:
            continue
        q = [cam_pos[0] + dx * t, cam_pos[1] + dy * t]
        z = cam_pos[2] + (eye[2] - cam_pos[2]) * t
        if math.hypot(q[0] - p[0], q[1] - p[1]) < PERSON_R_M and z < top:
            return True
    return False


def all_in_frame(cam, subjects, res, margin=0.03):
    """Every person's head top and feet inside the frame."""
    for v in subjects.values():
        m = v["measured"]
        for p in ([m["eye_xy"][0], m["eye_xy"][1], m["head_top_z"]], [m["pelvis_xy"][0], m["pelvis_xy"][1], m["feet_z"]]):
            pr = framing.project(cam, p, res)
            if not pr or not (margin < pr[0] < 1 - margin and margin < pr[1] < 1 - margin):
                return False
    return True


def body_points(m):
    """Set-frame points over a measured person: head top, eyes, chin, and the joints when measured (shoulders,
    elbows, hands, hips, knees, feet); a plain-mesh figure has only the head and its pelvis and feet."""
    e = m["eye_xy"]
    pts = {"head_top": [e[0], e[1], m["head_top_z"]], "eyes": [e[0], e[1], m["eye_z"]], "chin": [e[0], e[1], m["chin_z"]]}
    f = unit2(m.get("head_fwd_xy", [0, -1]))          # the head's sides: a face half in the frame is cut
    pts["head_l"] = [e[0] - f[1] * 0.1, e[1] + f[0] * 0.1, m["eye_z"]]
    pts["head_r"] = [e[0] + f[1] * 0.1, e[1] - f[0] * 0.1, m["eye_z"]]
    j = m.get("joints") or {}
    for k in ("upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r", "hand_l", "hand_r", "thigh_l", "thigh_r",
              "calf_l", "calf_r", "foot_l", "foot_r"):
        if k in j:
            pts[k] = j[k]
    if not j:
        pts["pelvis"] = [m["pelvis_xy"][0], m["pelvis_xy"][1], m.get("waist_z", m["feet_z"] + 0.9)]
        pts["feet"] = [m["pelvis_xy"][0], m["pelvis_xy"][1], m["feet_z"]]
    return pts


def frame_facts(cam, m, res):
    """Where a person lands in the frame: head in (eyes and chin inside), how much of them is inside, and whether the
    frame's left, right or top edge cuts them (some points in, some out that way)."""
    pr = {k: framing.project(cam, p, res) for k, p in body_points(m).items()}
    inside = {k: bool(q) and SIDE_EDGE < q[0] < 1 - SIDE_EDGE and EDGE < q[1] < 1 - EDGE for k, q in pr.items()}
    side_in = [q for k, q in pr.items() if q and SIDE_EDGE < q[0] < 1 - SIDE_EDGE and q[1] < 1 - EDGE]
    left = any(q and q[0] <= SIDE_EDGE for q in pr.values())
    right = any(q and q[0] >= 1 - SIDE_EDGE for q in pr.values())
    top = bool(pr["head_top"]) and pr["head_top"][1] <= EDGE and bool(pr["chin"]) and pr["chin"][1] > EDGE
    cut = bool(side_in) and (left or right) or top
    near_sh = [k for k in ("upperarm_l", "upperarm_r") if inside.get(k)]
    return {"head_in": inside["eyes"] and inside["chin"], "in_frac": round(sum(inside.values()) / len(inside), 2),
            "head_any": any(inside[k] for k in ("head_top", "eyes", "chin", "head_l", "head_r")),
            "cut": cut, "any_in": any(inside.values()), "shoulder_in": bool(near_sh),
            "cut_edges": [n for n, b in (("left", left and bool(side_in)), ("right", right and bool(side_in)),
                                         ("top", top)) if b]}


def mask_facts(shot, key, nx=72, ny=40):
    """From a rendered still's own depth passes: the share of the frame where this person is what the camera sees,
    and which frame edges their visible part touches. Joints alone missed a coat's side or a skirt past the edge,
    and a listener hidden behind a high chair back (11-11)."""
    g = shot.get("guides") or {}
    if not (g.get("depth") and g.get(f"depth_{key}")):
        return None
    sc, sp = Exr(g["depth"]), Exr(g[f"depth_{key}"])

    def vis(u, v):
        ds, dp = sc.at(u, v), sp.at(u, v)
        return ds is not None and dp is not None and dp < 5e4 and abs(dp - ds) <= max(3.0, 0.015 * ds)
    n = sum(vis((i + 0.5) / nx, (j + 0.5) / ny) for i in range(nx) for j in range(ny))
    edge = lambda pts: any(vis(u, v) for u, v in pts)
    rows = [(j + 0.5) / ny for j in range(ny)]
    cols = [(i + 0.5) / nx for i in range(nx)]
    return {"area": round(n / (nx * ny), 4), "left": edge([(0.002, v) for v in rows]),
            "right": edge([(0.998, v) for v in rows]), "top": edge([(u, 0.002) for u in cols])}


def face_seen(shot, key, m, res, covered=False):
    """The share of a person's face points where the still's depth shows them (None: no depth passes)."""
    g = shot.get("guides") or {}
    if not (g.get("depth") and g.get(f"depth_{key}")):
        return None
    sc, sp = Exr(g["depth"]), Exr(g[f"depth_{key}"])
    vis = n = 0
    for p in face_points(m):
        n += 1
        pr = framing.project(shot["camera"], p, res)
        if not pr:
            continue
        ds, dp = sc.at(pr[0], pr[1]), sp.at(pr[0], pr[1])
        if ds is None or dp is None or dp > 5e4:
            continue
        if abs(dp - ds) <= max(3.0, 0.015 * ds) or (covered and 0 <= dp - ds <= COVER_CM):
            vis += 1
    return round(vis / n, 2) if n else 0.0


def joint_seen(shot, key, q):
    """Is the person's own surface what the camera sees at a projected joint (within 4 px)."""
    g = shot.get("guides") or {}
    if not (g.get("depth") and g.get(f"depth_{key}")):
        return True
    sc, sp = Exr(g["depth"]), Exr(g[f"depth_{key}"])
    for du in (-0.004, 0.0, 0.004):
        for dv in (-0.006, 0.0, 0.006):
            ds, dp = sc.at(q[0] + du, q[1] + dv), sp.at(q[0] + du, q[1] + dv)
            if ds is not None and dp is not None and dp < 5e4 and abs(dp - ds) <= max(3.0, 0.015 * ds):
                return True
    return False


LISTENER_AREA = {"two": 0.01, "ots": 0.03}      # a listener's visible share of the frame
HEADROOM = {"close": 0.0, "medium": 0.03, "wide": 0.03, "full": 0.03}   # frame share above the hair in a talk shot
HEAD_HAIR_M = 0.04           # hair and its volume above the measured head top
HEAD_HALF_M = 0.11           # half the head's width, ears and hair


READ_DEG = 65.0              # a named person's head turned at most this far from the camera: their face reads (at
                             # 80 a profile turned to the fire passed, and the redraw drew the back of her head)
READ_FACE = 0.5              # and this share of their face points seen on the still (a three-quarter face reads;
                             # the speaker's own bar is MIN_FACE_VISIBLE)
FG_AREA = 0.10               # a person turned away from the camera covers at most this share of the frame
SEATED = ("sit", "seat")


def head_h(cam, m, res):
    top = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["head_top_z"]], res)
    chin = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]], res)
    return (chin[1] - top[1]) if top and chin else 0.0


def est_area(cam, m, res):
    """A person's share of the frame from the measured body alone (before a still exists): the clipped box over
    the head, shoulders (0.3 m out from the head's line) and feet, 90 % filled (a long coat or skirt fills it)."""
    e, p = m["eye_xy"], m["pelvis_xy"]
    f = unit2(m.get("head_fwd_xy", [0, -1]))
    r = [f[1], -f[0]]
    pts = [[e[0], e[1], m["head_top_z"]], [p[0], p[1], m["feet_z"]],
           [e[0] + r[0] * 0.3, e[1] + r[1] * 0.3, m["chin_z"] - 0.1], [e[0] - r[0] * 0.3, e[1] - r[1] * 0.3, m["chin_z"] - 0.1]]
    pr = [framing.project(cam, q, res) for q in pts]
    pr = [q for q in pr if q]
    if len(pr) < 2:
        return 0.0
    x0, x1 = max(0.0, min(q[0] for q in pr)), min(1.0, max(q[0] for q in pr))
    y0, y1 = max(0.0, min(q[1] for q in pr)), min(1.0, max(q[1] for q in pr))
    return max(0.0, x1 - x0) * max(0.0, y1 - y0) * 0.9


def area_scale(shots, subjects, res, scale=None):
    """Per person, how much bigger their rendered share of the frame is than est_area's box (a coat's skirt, the
    arms out): the largest ratio over the rendered stills, 0.5-3. A candidate camera's estimate for a person turned
    away read 10 % where the render showed 14 % and the re-framing went round five rounds (11-11)."""
    scale = dict(scale or {})
    for sh in shots:
        ff = (sh.get("speaker_check") or {}).get("frame_facts") or {}
        for k, f in ff.items():
            ma = (f.get("mask") or {}).get("area")
            if ma is None or k not in subjects:
                continue
            e = est_area(sh["camera"], subjects[k]["measured"], res)
            if e > 0.02 and ma > 0.02:
                scale[k] = round(max(scale.get(k, 0.0), min(3.0, max(0.5, ma / e))), 3)
    return scale


def area_excess(cam, subjects, res, keep, opts=None):
    """How far people turned away from a candidate camera go past FG_AREA by the body estimate (scaled by what the
    renders showed): orders candidates, never rejects one."""
    sc = (opts or {}).get("area_scale") or {}
    return sum(max(0.0, est_area(cam, v["measured"], res) * sc.get(k, 1.0) - FG_AREA)
               for k, v in subjects.items() if k not in keep and turned_away(cam, v["measured"]))


def turned_away(cam, m):
    return plan_angle(m["head_fwd_xy"], m["eye_xy"], cam["position"]) > 100


def framing_why(cam, kind, speaker, listener, ots, show, subjects, res, shot=None, opts=None):
    """The framing rules' failures for one camera (empty: it passes), from the measured people; with a rendered
    shot, its own depth passes decide edge cuts and whether the listener is really seen.
    opts: {"seated": {keys seated}, "subject": key, "joints": [[key, joint], ...] that must be in view}."""
    opts = opts or {}
    why = []
    facts = {k: frame_facts(cam, v["measured"], res) for k, v in subjects.items()}
    if shot is not None:
        for k, f in facts.items():
            mf = mask_facts(shot, k)
            if mf is None:
                continue
            f["mask"] = mf
            # cut: the person's visible part reaches the left or right edge (any of it), or the top edge with the
            # head (a medium's waist at the bottom edge is the shot, not a cut)
            f["cut"] = mf["area"] > 0 and (mf["left"] or mf["right"] or (mf["top"] and f["head_any"]))
            f["cut_edges"] = [e for e in ("left", "right", "top") if mf[e]]
            f["head_proj"] = f["head_any"]   # the head inside the frame's bounds (behind someone or not)
            if mf["area"] < LISTENER_AREA["two"] * 0.3:
                f["head_in"] = f["head_any"] = f["shoulder_in"] = False      # hidden (behind a chair, a person)
    seated_lis = bool(listener) and listener in (opts.get("seated") or ())
    if seated_lis and ots:
        why.append("over the shoulder of a listener seated in a high-backed chair (the chair back fills the frame)")
    # every head in a talk shot is smaller than the speaker's; a person turned away is a modest share of the frame
    # with the head inside it (a recognisable outline, not a dark mass merged with the frame edge)
    # With no speaker the person the shot's line is about leads the same way (11-12: "she gives a start" with her
    # small at the side and a man's back in the middle - the video gave the start to the man, in all three takes).
    lead = speaker or (opts.get("line_subject") if opts.get("line_subject") in subjects else None)
    role = "speaker" if speaker else "person the shot is about"
    sh_spk = head_h(cam, subjects[lead]["measured"], res) if lead else 0.0
    keep = {lead, opts.get("subject")}
    for k, v in subjects.items():
        if k == lead or not facts[k]["any_in"]:
            continue
        h = head_h(cam, v["measured"], res)
        if lead and facts[k]["head_any"] and h > sh_spk * 1.02:
            why.append(f"{k}'s head ({h:.3f}) is bigger than the {role}'s ({sh_spk:.3f})")
        if lead and k not in keep and turned_away(cam, v["measured"]):
            # a foreground back so large it takes the speaker's shot (a back alone is a staging choice: 11-12)
            ma = (facts[k].get("mask") or {}).get("area")
            if ma is None:                   # no still yet (a candidate camera): the area from the body only orders
                continue                     # candidates (area_excess); the render's own mask decides (11-11)
            if ma > FG_AREA:
                why.append(f"{k}, turned away, covers {ma:.0%} of the frame")
    for q3 in opts.get("points") or []:      # a move's start, path and end place: all inside the one frame
        q = framing.project(cam, q3, res)
        if not q or not (0.03 < q[0] < 0.97 and 0.03 < q[1] < 0.97):
            why.append(f"the move's place at {[round(v, 2) for v in q3]} is out of the frame")
    lm = opts.get("landmark")
    if lm and not opts.get("points"):        # a thing the subject turns toward: in the frame (not only its front)
        for q3 in ([lm["center"][0], lm["center"][1], lm["top"]], list(lm["center"])):
            q = framing.project(cam, q3, res)
            if not q or not (0.03 < q[0] < 0.97 and 0.03 < q[1] < 0.97):
                why.append(f"the {lm['name']} (what the shot turns toward) is out of the frame")
                break
    if lm:                                   # the end place's landmark seen from its front, not from behind it
        d = [cam["position"][0] - lm["center"][0], cam["position"][1] - lm["center"][1]]
        if d[0] * lm["front"][0] + d[1] * lm["front"][1] < 0.3 * math.hypot(*d):
            why.append(f"the end place's {lm['name']} is seen from its side or back")
    mv = opts.get("mover")
    ep = opts.get("end_place")
    if mv and ep and mv in subjects:         # the end place as prominent as the other people, the mover nearest it
        top = framing.project(cam, [ep["center"][0], ep["center"][1], ep["top"]], res)
        bot = framing.project(cam, [ep["center"][0], ep["center"][1], 0.0], res)
        eh = (bot[1] - top[1]) if top and bot else 0.0
        oth = [head_h(cam, v["measured"], res) * 7.5 * 0.55 for k, v in subjects.items() if k != mv and facts[k]["any_in"]]
        if oth and eh < min(oth) * 0.8:
            why.append(f"the end place is small in the frame ({eh:.2f} of its height)")
    if mv and mv in subjects:                # the mover is the shot: clearly in view, the face readable at the start
        mm = subjects[mv]["measured"]
        mma = (facts[mv].get("mask") or {}).get("area")
        if plan_angle(mm["head_fwd_xy"], mm["eye_xy"], cam["position"]) > 80 or not facts[mv]["head_in"]:
            why.append(f"{mv}'s face (the one who moves) is not readable")
        elif shot is not None and mma is not None and mma < MOVER_AREA:
            why.append(f"{mv} (the one who moves) is hidden: {mma:.1%} of the frame")
        elif shot is not None:
            fv = face_seen(dict(shot, camera=cam), mv, mm, res, mv in (opts.get("covered") or ()))
            if fv is not None and fv < 0.5:
                why.append(f"{mv}'s face (the one who moves) {fv:.0%} visible")
        elif blocked(cam["position"], [mm["eye_xy"][0], mm["eye_xy"][1], mm["eye_z"]],
                     [v["measured"] for k, v in subjects.items() if k != mv]):
            why.append(f"{mv} (the one who moves) is behind someone")
    for k, jn in opts.get("joints") or []:
        j = ((subjects.get(k) or {}).get("measured") or {}).get("joints") or {}
        q = framing.project(cam, j[jn], res) if jn in j else None
        if not q or not (0.05 < q[0] < 0.95 and 0.05 < q[1] < 0.95):
            why.append(f"{k}'s {jn} (the action) is not in the frame")
        elif shot is not None and not joint_seen(shot, k, q):
            why.append(f"{k}'s {jn} (the action) is hidden")
    # the speaker's whole head and its headroom inside any talk camera (the speaker only: people who are not
    # speaking may be partly hidden, in profile, seen from behind or cut by the frame; that is how a room looks)
    for hk in ([speaker] if speaker else []):
        hm = subjects[hk]["measured"]
        hf = unit2(hm.get("head_fwd_xy", [0, -1]))
        he = hm["eye_xy"]
        pts = [[he[0], he[1], hm["head_top_z"] + HEAD_HAIR_M], [he[0], he[1], hm["chin_z"]]] + \
              [[he[0] + hf[1] * s_ * HEAD_HALF_M, he[1] - hf[0] * s_ * HEAD_HALF_M, hm["eye_z"]] for s_ in (-1, 1)]
        pr = [framing.project(cam, q, res) for q in pts]
        room_top = HEADROOM.get(kind, 0.03)
        if not all(pr) or pr[0][1] < room_top or pr[1][1] > 0.98 or min(q[0] for q in pr) < 0.02 \
                or max(q[0] for q in pr) > 0.98:
            why.append(f"{'the speaker' if hk == speaker else hk}'s head or its headroom is outside the frame")
    if speaker and kind != "close":
        m = subjects[speaker]["measured"]
        top = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["head_top_z"]], res)
        chin = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]], res)
        fh = (chin[1] - top[1]) if top and chin else 0.0
        if fh < TALK_FACE_H.get(kind, 0.09):
            why.append(f"speaker's face {fh:.3f} of the frame (talk {kind}: {TALK_FACE_H.get(kind, 0.09)})")
        if listener and not ots:             # a two-shot favours the speaker: the listener is not the bigger head
            lm = subjects[listener]["measured"]
            lt = framing.project(cam, [lm["eye_xy"][0], lm["eye_xy"][1], lm["head_top_z"]], res)
            lc = framing.project(cam, [lm["eye_xy"][0], lm["eye_xy"][1], lm["chin_z"]], res)
            if lt and lc and (lc[1] - lt[1]) > fh * 1.05 and facts[listener]["any_in"]:
                why.append(f"the listener's head ({lc[1] - lt[1]:.3f}) is bigger than the speaker's ({fh:.3f})")
        if listener and not seated_lis:     # a seated listener may be left out: a clean single of the speaker
            f = facts[listener]
            ma = (f.get("mask") or {}).get("area")
            if ots and (not (f["shoulder_in"] or f["head_any"]) or (ma is not None and ma < LISTENER_AREA["ots"])):
                why.append("the listener's shoulder is not in the over-the-shoulder frame")
            elif ots:                        # the listener's shoulder at one side, the speaker toward the middle
                lm = subjects[listener]["measured"]
                le = framing.project(cam, [lm["eye_xy"][0], lm["eye_xy"][1], lm["eye_z"]], res)
                se = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]], res)
                if le and 0.25 < le[0] < 0.75:
                    why.append(f"the listener's head is in the middle of the frame ({le[0]:.2f}), not at its side")
                if se and not 0.25 < se[0] < 0.75:
                    why.append(f"the speaker's face is at the frame's side ({se[0]:.2f})")
            elif not ots and not in_picture(f):
                why.append("the listener is not in view")
    for k, sp_ in (opts.get("seat_xy") or {}).items():   # a seat in view with its sitter hidden reads as empty
        if k in facts and k in subjects and not in_picture(facts[k]):
            q = framing.project(cam, [sp_[0], sp_[1], 0.6], res)
            if q and 0.05 < q[0] < 0.95 and 0.05 < q[1] < 0.95:
                why.append(f"{k}'s seat is in view with {k} hidden (it reads as empty)")
    for k in show or []:                     # a person the director names: in the picture (any of them), no more
        hs = head_share(cam, subjects[k]["measured"], res) if k in subjects else None
        if k in facts and (not in_picture(facts[k]) or (hs is not None and hs < SLIVER)):
            why.append(f"{k} (named for the shot) is not in the picture")
    sk = opts.get("subject")                 # the plan's shot class: a close is a close
    if kind == "close" and sk in subjects and head_h(cam, subjects[sk]["measured"], res) < MIN_FACE_H["close"]:
        why.append(f"{sk}'s face {head_h(cam, subjects[sk]['measured'], res):.3f} of the frame: not a close")
    if opts.get("_single"):                  # a clean single: everyone else clearly in or clearly out
        for k, v in subjects.items():
            if k == speaker:
                continue
            pr = [framing.project(cam, q, res) for q in body_points(v["measured"]).values()]
            xs = [q[0] for q in pr if q and q[1] < 1.0]
            if xs and not (min(xs) > 1.10 or max(xs) < -0.10 or (min(xs) > 0.05 and max(xs) < 0.95)):
                why.append(f"{k} near the edge of a single")
    if not speaker and shot is not None:     # no one speaks: the people are the picture, not its two edges
        xs = []
        for k, f in facts.items():
            if not in_picture(f):
                continue
            em = subjects[k]["measured"]
            q = framing.project(cam, [em["eye_xy"][0], em["eye_xy"][1], em["eye_z"]], res)
            if q:
                xs.append(q[0])
        if len(xs) >= 2 and not any(MIDDLE[0] <= x <= MIDDLE[1] for x in xs):
            why.append("nobody in the middle of the frame: the people are at its sides ("
                       + ", ".join(f"{x:.2f}" for x in sorted(xs)) + ")")
    for k, f in facts.items():
        # a face cut by the frame's edge to a sliver (a person who is not speaking may be cut otherwise); the
        # speaker's own cut is the speaker's head rule above, and the body cut in a talk shot here
        em = subjects[k]["measured"]
        sl = face_sliver(cam, em, res)
        if k != speaker and sl and not turned_away(cam, em):
            why.append(f"PARTIAL FACE: {k}'s face cut by the frame's edge to a sliver ({sl:.0%} of it in)")
        elif k == speaker and f["cut"] and kind != "close":
            why.append(f"{k} cut by the frame's {'/'.join(f['cut_edges'])} edge")
    return why, facts


def subject_face_why(shot, k, subjects, res, covered=False) -> list:
    """The face of the person a shot is about, shown as a speaker's is: turned toward the camera and seen on the
    still (a woman whose line it was showed only the back of her head in the foreground, 11-12)."""
    m = subjects[k]["measured"]
    cam = shot["camera"]
    if plan_angle(m["head_fwd_xy"], m["eye_xy"], cam["position"]) > MAX_HEAD_DEG + 15:
        return [f"{k}'s face (the shot is about {k}) is turned from the camera"]
    fv = face_seen(shot, k, m, res, covered)
    if fv is not None and fv < MIN_FACE_VISIBLE:
        return [f"{k}'s face (the shot is about {k}) {fv:.0%} visible"]
    return []


def placement_why(cam, subjects, room) -> list:
    """A camera stands inside the room and outside everyone's body: framed on a face turned to the window, the
    camera landed behind the glass and every check passed through it; another sat inside a man and drew black
    (11-12)."""
    p = cam["position"]
    why = []
    if not in_room(p, room):
        why.append("the camera is outside the room")
    for k, v in subjects.items():
        m = v["measured"]
        if math.dist(p[:2], m["pelvis_xy"]) < PERSON_R_M + 0.08 and p[2] < m["head_top_z"] + 0.05:
            why.append(f"the camera is inside {k}")
    return why


def in_picture(f: dict) -> bool:
    """The person is in the picture: their head inside the frame's bounds (it may be behind someone: that is how a
    room looks), and on the still some of their own pixels; a head off the frame's edge with a sliver of body in
    view is not (the redraw dropped her altogether, 11-12)."""
    ma = (f.get("mask") or {}).get("area")
    head = f.get("head_proj", f.get("head_any"))
    return bool(head) and ((ma >= 0.002) if ma is not None else bool(f.get("any_in")))


MIDDLE = (0.25, 0.75)        # a shot with no speaker and two or more people: some head across this middle half
MOVER_AREA = 0.01            # the mover's own visible share of the frame at least
SLIVER = 0.35                # a face with less than this share of its width inside the frame (and some of it in)


def face_sliver(cam, m, res) -> float | None:
    """The share of a face's width inside the frame when it is a sliver (some in, under SLIVER), else None."""
    sh = head_share(cam, m, res)
    return sh if sh is not None and 0.0 < sh < SLIVER else None


def head_share(cam, m, res) -> float | None:
    """The share of a head inside the frame (width and height, the smaller), whichever way it faces."""
    f = unit2(m.get("head_fwd_xy", [0, -1]))
    e = m["eye_xy"]
    pl = framing.project(cam, [e[0] - f[1] * 0.09, e[1] + f[0] * 0.09, m["eye_z"]], res)
    pr = framing.project(cam, [e[0] + f[1] * 0.09, e[1] - f[0] * 0.09, m["eye_z"]], res)
    pe = framing.project(cam, [e[0], e[1], m["eye_z"]], res)
    if not (pl and pr and pe):
        return 0.0                           # behind the camera
    x0, x1 = min(pl[0], pr[0]), max(pl[0], pr[0])
    w = max(x1 - x0, 1e-4)
    share = max(0.0, min(1.0, x1) - max(0.0, x0)) / w
    top = framing.project(cam, [e[0], e[1], m["head_top_z"]], res)
    chin = framing.project(cam, [e[0], e[1], m["chin_z"]], res)
    if top and chin and chin[1] > top[1]:      # cut at the top or bottom too
        share = min(share, max(0.0, min(1.0, chin[1]) - max(0.0, top[1])) / (chin[1] - top[1]))
    return share


LOOK_MAX_DEG = 75.0           # a listener's head turns this far toward the speaker; the chest takes the rest


def look_at(spec, speaker):
    """People who listen face the speaker, head and eyes (a character with "look": false keeps its own; one with
    "look_at": key looks at that person instead): each looker's head and neck turned from the body's facing toward
    the target's head, up or down to it, as pose steps (about z: toward the character's right is positive; about x:
    up). Before this, a listener in a talk shot looked wherever the body pointed, often at the camera (11-11)."""
    ch = spec["characters"]
    eye_z = lambda c: 1.15 if any(w in ((c.get("body_pose") or {}).get("pose") or "") for w in SEATED) else 1.6
    for k, c in list(ch.items()):
        tgt_k = c.get("look_at") if c.get("look_at") in ch and c.get("look_at") != k else speaker
        if tgt_k is None or k == tgt_k or tgt_k not in ch or c.get("asset") or c.get("look") is False or "_look" in c:
            continue
        sp = ch[tgt_k]
        f = unit2(c.get("facing", [0, -1]))
        d = [sp["position"][0] - c["position"][0], sp["position"][1] - c["position"][1]]
        dist = math.hypot(*d) or 1.0
        right = [f[1], -f[0]]
        yaw = math.degrees(math.atan2(d[0] * right[0] + d[1] * right[1], d[0] * f[0] + d[1] * f[1]))
        chest = max(0.0, abs(yaw) - LOOK_MAX_DEG) * (1 if yaw > 0 else -1)
        yaw -= chest
        pitch = math.degrees(math.atan2(eye_z(sp) - eye_z(c), dist))
        bp = dict(c.get("body_pose") or {"pose": "stand", "params": {}})
        pr = dict(bp.get("params") or {})
        if bp.get("pose") == "stand_relaxed":
            pr["head_deg"] = 0.0
            pr["head_tilt"] = 0.0
        steps = [["rot", "spine_03", [0, 0, 1], chest]] if chest else []
        steps += [["rot", "neck_01", [0, 0, 1], round(yaw * 0.4, 2)], ["rot", "head", [0, 0, 1], round(yaw * 0.6, 2)],
                  ["rot", "neck_01", [1, 0, 0], round(pitch * 0.4, 2)], ["rot", "head", [1, 0, 0], round(pitch * 0.6, 2)]]
        pr["steps"] = list(pr.get("steps") or []) + steps
        bp["params"] = pr
        ch[k] = dict(c, body_pose=bp, _look={"at": tgt_k, "yaw": round(yaw + chest, 1), "pitch": round(pitch, 1),
                                             "steps": len(steps)})
    return spec


ACTION_HANDS_M = {"veil_lift": (0.22, all), "look_at_sleeve": (0.38, any)}   # an action's start: the hands this
                                                                             # near the chin (both / either), measured


def assets_action_steps(spec):
    """Start actions the assets' pose solver writes without a measured body (characters/poses_extra): the look
    at a sleeve (one forearm lifted, turned up, the head down on it), put first in the person's pose steps."""
    tools = os.environ.get("SCENES_CHAR_TOOLS")
    if not tools:
        return
    sys.path.insert(0, tools)
    import poses_extra
    for k, c in spec["characters"].items():
        pr = (c.get("body_pose") or {}).get("params") or {}
        if pr.get("action") == "look_at_sleeve" and not pr.get("_action_steps"):
            got = poses_extra.look_at_sleeve("l", pr.get("seat_z_m"))
            pr["steps"] = got["steps"] + list(pr.get("steps") or [])
            pr["_action_steps"] = len(got["steps"])
            c["look"] = False                # the head is on the sleeve, not on the speaker
REACH_M = 0.12               # a reach's hand this near the top of the other person's forearm (measured)
SLEEVE_R_M, PALM_HALF_M, PALM_LEN_M = 0.05, 0.015, 0.09


def solve_reach(spec, subjects):
    """A person with "reach": {"to": key, "her_side": l|r} lays the hand nearer the other person on the top of
    their forearm's sleeve: the steps from the assets' pose solver (characters/poses_extra.reach_forearm) on the
    two measured skeletons, added to his pose. Returns {key: solver facts} or {}."""
    tools = os.environ.get("SCENES_CHAR_TOOLS")
    if not tools:
        return {}
    sys.path.insert(0, tools)
    import poses_extra
    out = {}
    for k, c in spec["characters"].items():
        r = c.get("reach")
        if not r or r.get("to") not in subjects or k not in subjects:
            continue
        hj = subjects[k]["measured"].get("joints") or {}
        sj = subjects[r["to"]]["measured"].get("joints") or {}
        if not all(j in hj for j in ("upperarm_l", "upperarm_r")) or not all(
                j in sj for j in (f"lowerarm_{r['her_side']}", f"hand_{r['her_side']}")):
            continue
        top, _, _ = poses_extra.forearm_contact(sj, r["her_side"], SLEEVE_R_M)
        side = "l" if poses_extra.to_actor(top, c["position"], c["facing"])[0] > 0 else "r"
        steps, facts = poses_extra.reach_forearm(hj, c["position"], c["facing"], sj, r["her_side"], SLEEVE_R_M,
                                                 PALM_HALF_M, PALM_LEN_M, side=side)
        bp = c["body_pose"]
        bp.setdefault("params", {})["steps"] = steps + list(bp["params"].get("steps") or [])
        r.update(side=side, contact=facts["palm_centre_set_m"])
        for cam in spec["cameras"]:
            if cam.get("show_joints"):
                cam["show_joints"] = [[k, f"hand_{side}"], [r["to"], f"lowerarm_{r['her_side']}"]]
        out[k] = dict(facts, side=side)
    return out


def staged_facts(spec, subjects):
    """What the plan asks of each person that the engine still must hold: the worn state's figure (the variant
    resolved, its words carried for the redraw) and the action's start pose (measured hands). {key: [failures]}."""
    out = {}
    for k, c in spec["characters"].items():
        bad = []
        w = c.get("worn") or c.get("variant")
        if w and w != "usual":
            if not (c.get("id") or "").endswith("_" + w):
                bad.append(f"worn '{w}': the staged figure is {c.get('id')}")
            if not c.get("wear_add"):
                bad.append(f"worn '{w}': no words for it reach the redraw")
        t = c.get("facing_to")
        m_ = (subjects.get(k) or {}).get("measured") or {}
        if t in subjects and t != k and m_.get("facing") and not c.get("reach"):
            tp = subjects[t]["measured"]["pelvis_xy"]
            ang = plan_angle(m_["facing"], m_["pelvis_xy"], tp)
            if ang > FACE_TURN_DEG:
                bad.append(f"faces {ang:.0f} deg away from {t}, whom the plan has them face")
        act = ((c.get("body_pose") or {}).get("params") or {}).get("action")
        if c.get("reach"):                   # the hand on the other's forearm, measured
            m = (subjects.get(k) or {}).get("measured") or {}
            hand = ((m.get("joints") or {}).get(f"hand_{c['reach'].get('side', 'r')}"))
            ct = c["reach"].get("contact")
            if not (hand and ct):
                bad.append("reach: not solved (no measured hand or contact)")
            elif math.dist(hand[:3], ct) > REACH_M + 0.1:          # the wrist joint sits a palm short of it
                bad.append(f"reach: hand {math.dist(hand[:3], ct):.2f} m from her forearm")
            act = None
        if act:
            m = (subjects.get(k) or {}).get("measured") or {}
            j = m.get("joints") or {}
            if act not in ACTION_HANDS_M:
                bad.append(f"action '{act}': no start pose for it")
            elif not all(h in j for h in ("hand_l", "hand_r")):
                bad.append(f"action '{act}': hands not measured")
            else:
                chin = [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]]
                lim, how = ACTION_HANDS_M[act]
                ds = [math.dist(j[h][:3], chin) for h in ("hand_l", "hand_r")]
                if not how(d <= lim for d in ds):
                    bad.append(f"action '{act}': hands {min(ds):.2f}-{max(ds):.2f} m from the chin")
        out[k] = bad
    return out


FACE_TURN_DEG = 90.0         # a person's body within a quarter turn of whom they face (checked on the engine)


def aim_bodies(spec):
    """Standing people turned to whom they face ("facing_to", from the plan's facing words or what the shot is
    about), from where everyone now stands: after a walk's start or a mark moved someone, the facings computed
    from the plan's places pointed at the window (11-12). Returns the keys re-aimed."""
    ch, out = spec["characters"], []
    for k, c in ch.items():
        t = c.get("facing_to")
        if not t or t not in ch or t == k or any(w in ((c.get("body_pose") or {}).get("pose") or "") for w in SEATED):
            continue
        d = unit2([ch[t]["position"][0] - c["position"][0], ch[t]["position"][1] - c["position"][1]])
        if math.degrees(math.acos(max(-1.0, min(1.0, d[0] * unit2(c["facing"])[0] + d[1] * unit2(c["facing"])[1])))) > 2:
            c["facing"] = [round(d[0], 4), round(d[1], 4)]
            out.append(k)
    return out


def unlook(c):
    """A character without the head turns look_at added (they are the last pose steps)."""
    lk = c.get("_look")
    if not lk:
        return c
    c = {kk: vv for kk, vv in c.items() if kk != "_look"}
    bp = dict(c.get("body_pose") or {})
    pr = dict(bp.get("params") or {})
    st = list(pr.get("steps") or [])
    pr["steps"] = st[:len(st) - lk.get("steps", 0)]
    bp["params"] = pr
    c["body_pose"] = bp
    return c


# ------------------------------------------------------------------ marks
MARK_CLEAR_M = 0.55          # a mark this far from anyone else's place
TRY_SAME, TRY_SINGLE, TRY_MARK = 2, 2, 3      # a failing talk shot's tries, in this order, each one rendered
MAX_TRIES = 10
CAM_KEYS = ("position", "target", "focal_mm", "az", "el", "screen_x", "ots", "listener", "single", "show", "tight")


def turn2(v, deg):
    a = math.radians(deg)
    return [v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)]


def mark_ref(spec, k, speaker):
    """Whom a person on a vague mark is turned toward: their look target, else the other of the line, else the
    speaker; keeps their facing relative to that person when they change marks."""
    ch = spec["characters"]
    r = ch[k].get("look_at") or (ch[k].get("_look") or {}).get("at")
    if not r:
        ln = [x for x in (spec.get("line") or []) if x != k and x in ch]
        r = ln[0] if k in (spec.get("line") or []) and ln else speaker
    return r if r in ch and r != k else None


def mark_turn(spec, k, p, ref_pos):
    """Degrees the person turns on moving from their place to p, keeping the same angle to the reference person."""
    if not ref_pos:
        return 0.0
    o = spec["characters"][k]["position"]
    a0 = math.atan2(ref_pos[1] - o[1], ref_pos[0] - o[0])
    a1 = math.atan2(ref_pos[1] - p[1], ref_pos[0] - p[0])
    return math.degrees(a1 - a0)


def shifted(subjects, spec, moves, speaker):
    """The measured people as they would stand with `moves` ({key: mark}): bodies carried to the mark and turned
    by the same angle toward their reference person, listeners' heads re-aimed at whom they look at."""
    out = json.loads(json.dumps(subjects))
    for k, p in moves.items():
        m = out[k]["measured"]
        o = spec["characters"][k]["position"]
        ref = mark_ref(spec, k, speaker)
        rp = (moves.get(ref) or spec["characters"][ref]["position"]) if ref else None
        deg = mark_turn(spec, k, p, rp)
        d = [p[0] - o[0], p[1] - o[1]]
        mv = lambda q: [o[0] + turn2([q[0] - o[0], q[1] - o[1]], deg)[0] + d[0],
                        o[1] + turn2([q[0] - o[0], q[1] - o[1]], deg)[1] + d[1]]
        for f in ("eye_xy", "pelvis_xy", "chest_xy"):
            if f in m:
                m[f] = mv(m[f])
        for f in ("head_fwd_xy", "facing"):
            if f in m:
                m[f] = turn2(m[f], deg)
        for j, v in (m.get("joints") or {}).items():
            m["joints"][j] = mv(v) + list(v[2:])
    for k, c in spec["characters"].items():
        at = (c.get("_look") or {}).get("at")
        if at and k in out and at in out and (k in moves or at in moves):
            m = out[k]["measured"]
            t = out[at]["measured"]["eye_xy"]
            m["head_fwd_xy"] = unit2([t[0] - m["eye_xy"][0], t[1] - m["eye_xy"][1]])
    return out


def apply_marks(spec, moves, speaker, relook):
    """Move people to their chosen marks (turned to keep their angle to their reference person), heads re-aimed."""
    ch = spec["characters"]
    new = {}
    for k, p in moves.items():
        ref = mark_ref(spec, k, speaker)
        rp = (moves.get(ref) or ch[ref]["position"]) if ref else None
        deg = mark_turn(spec, k, p, rp)
        new[k] = dict(ch[k], position=[round(p[0], 3), round(p[1], 3)],
                      facing=[round(v, 4) for v in unit2(turn2(ch[k].get("facing", [0, -1]), deg))])
    ch.update(new)
    if relook:
        for k in list(ch):
            ch[k] = unlook(ch[k])
        look_at(spec, relook)


PATH_CLEAR_M = 0.45          # a walk's straight floor path: nobody else's mark within this of it (a body's width)


def end_thing(spec, c):
    sys.path.insert(0, str(HERE.parent / "redraw"))
    import setfacts
    return next((x for x in setfacts.things(Path(C.LIBRARY) / "set" / spec["set"]) if x["name"] == c.get("move_to")),
                None)


def similar_seats(spec, t):
    """The room's other seats like the end place (the same kind by name: "armchair", "armchair_2"): a mover with one
    of them nearer than the end place sat in the nearer one in two of three video takes (11-12)."""
    import re
    sys.path.insert(0, str(HERE.parent / "redraw"))
    import setfacts
    base = re.sub(r"_\d+$", "", t["name"])
    return [x for x in setfacts.things(Path(C.LIBRARY) / "set" / spec["set"])
            if x["name"] != t["name"] and re.sub(r"_\d+$", "", x["name"]) == base]


def seg_dist(p, a, b):
    d = [b[0] - a[0], b[1] - a[1]]
    L2 = d[0] * d[0] + d[1] * d[1] or 1e-9
    t = max(0.0, min(1.0, ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / L2))
    return math.dist(p, [a[0] + d[0] * t, a[1] + d[1] * t])


def clear_walks(spec, seat_names=()):
    """Move shots, as mechanics from the scene's facts, before anything renders: the mover starts on the allowed
    mark nearest the end place (when her start is a region, not a stated spot), and the straight floor path from
    there to the end place is clear: a vague-place person on it takes their nearest allowed mark off it (a move the
    shot's action needs); a person with an exact place on it is reported, as no camera can fix it. In a walk staged
    with the two men between her and the chair, the video had her sink to the floor beside them (11-12).
    Returns {"moved": {key: [from, to]}, "blocked": [keys], "shot_facts": {camera: {...}}}."""
    ch = spec["characters"]
    out = {"moved": {}, "blocked": [], "shot_facts": {}}
    for c in spec["cameras"]:
        mv, t = c.get("mover"), end_thing(spec, c)
        if mv not in ch or not t:
            continue
        end = t["center"][:2]
        m = ch[mv]
        sim = [x["center"][:2] for x in similar_seats(spec, t)]
        if m.get("marks"):                   # nearest the end place, and no seat like it nearer than the end place
            best = min(m["marks"], key=lambda q: (any(math.dist(q, x) < math.dist(q, end) for x in sim),
                                                  math.dist(q, end)))
            if math.dist(best, m["position"]) > 0.05 and not any(
                    math.dist(best, ch[o]["position"]) < MARK_CLEAR_M for o in ch if o != mv):
                out["moved"][mv] = [list(m["position"]), list(best)]
                m["position"] = list(best)
        start = m["position"]
        blocked_here = []
        for o, v in ch.items():
            if o == mv or seg_dist(v["position"], start, end) >= PATH_CLEAR_M:
                continue
            seated = any(w in ((v.get("body_pose") or {}).get("pose") or "") for w in SEATED)
            offs = [q for q in (v.get("marks") or []) if seg_dist(q, start, end) >= PATH_CLEAR_M and not any(
                math.dist(q, ch[x]["position"]) < MARK_CLEAR_M for x in ch if x != o)]
            if offs and not seated:
                q = min(offs, key=lambda q: math.dist(q, v["position"]))
                out["moved"][o] = [list(v["position"]), list(q)]
                v["position"] = list(q)
            else:
                out["blocked"].append(o)
                blocked_here.append(o)
        words = t["words"]
        out["shot_facts"][c["name"]] = {
            "moves": {"who": mv, "to": t["name"], "to_words": words,
                      "posture_end": "sitting" if t["name"] in seat_names else "standing"},
            "fixed": {o: ("sitting" if any(w in ((v.get("body_pose") or {}).get("pose") or "") for w in SEATED)
                          else "standing") for o, v in ch.items() if o != mv},
            "path_blocked_by": list(blocked_here)}
    return out


def move_points(spec, c):
    """A move shot ("mover": key, "move_to": set item name): the mover's start (head and feet), the end place and
    the path between, as set-frame points the key's one frame must hold (the camera then need not move: framed on
    the start alone, a video model had to swing the camera and invent the rest of the room, 11-11)."""
    if not (c.get("mover") and c.get("move_to")):
        return []
    sys.path.insert(0, str(HERE.parent / "redraw"))
    import setfacts
    t = next((x for x in setfacts.things(Path(C.LIBRARY) / "set" / spec["set"]) if x["name"] == c["move_to"]), None)
    m = spec["characters"].get(c["mover"])
    if not t or not m:
        return []
    a = m["position"][:2]
    b = t["center"][:2]
    pts = [[a[0], a[1], 1.65], [a[0], a[1], 0.05], [b[0], b[1], t["center"][2] + t["half"][2]],
           [b[0], b[1], 0.05], [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.8]]
    lm = landmark(spec, c)
    if lm:                                   # the end place's landmark: its middle and its top
        pts += [list(lm["center"]), [lm["center"][0], lm["center"][1], lm["top"]]]
    return pts


def landmark(spec, c):
    """The thing a move's end place is named by ("by the fire": the fireplace): its middle, top and the way its
    front faces (into the room, off the wall it stands nearest)."""
    if not c.get("move_landmark"):
        return None
    sys.path.insert(0, str(HERE.parent / "redraw"))
    import setfacts
    sd = Path(C.LIBRARY) / "set" / spec["set"]
    t = next((x for x in setfacts.things(sd) if x["name"] == c["move_landmark"]), None)
    if not t:
        return None
    W, D = json.loads((sd / "spec.json").read_text())["size_m"][:2]
    x, y = t["center"][:2]
    walls = [(W / 2 - x, [-1.0, 0.0]), (W / 2 + x, [1.0, 0.0]), (D / 2 - y, [0.0, -1.0]), (D / 2 + y, [0.0, 1.0])]
    front = min(walls, key=lambda w: w[0])[1]
    return {"name": t["name"], "center": list(t["center"]), "top": t["center"][2] + t["half"][2], "front": front}


def cam_opts(spec, c):
    """The framing rules' per-camera facts: who sits (no shots over a seated listener's chair back), the camera's
    subject, the action's joints that must be seen ("show_joints": [[key, joint], ...]) and a move's places."""
    seated = {k for k, v in spec["characters"].items()
              if any(w in ((v.get("body_pose") or {}).get("pose") or "") for w in SEATED)}
    covered = {k for k, v in spec["characters"].items() if v.get("face_covered")}
    return {"seated": seated, "subject": c.get("subject"), "line_subject": c.get("line_subject"), "joints": c.get("show_joints") or [], "covered": covered,
            "landmark": c.get("_landmark"), "prev_cam": spec.get("_prev_cam"), "end_place": c.get("_end_place"),
            "seat_xy": {k: v["position"] for k, v in spec["characters"].items()
                        if any(w in ((v.get("body_pose") or {}).get("pose") or "") for w in SEATED)},
            "points": c.get("_points") or [], "mover": c.get("mover"), "area_scale": spec.get("_area_scale") or {}}


def listener_of(c, spec_line, speaker):
    if c.get("single"):
        return None
    if c.get("ots"):
        return c["ots"]
    if c.get("listener"):
        return c["listener"]
    if speaker and spec_line and speaker in spec_line:
        return [k for k in spec_line if k != speaker][0]
    return None


def two_shot_candidates(c, shot, spk, lis, subjects, res, room, line, est_side, tried, opts=None):
    """Placed cameras that keep the speaker and the listener (or the named people) in view, each checked by the
    framing rules: a two-shot aimed between their faces nearer the speaker, or an over-the-shoulder from behind the
    listener aimed at the speaker (when two people face each other no lens holds both faces). Returns
    [(camera, "two" | "ots")], the passing standpoints nearest the old camera first."""
    cam0 = shot["camera"]
    kind = c["kind"]
    keys = [k for k in ([spk, lis] if spk else (c.get("show") or [])) if k]
    if not keys:
        return []
    ms = [subjects[k]["measured"] for k in keys]
    w = [0.6, 0.4] if spk and lis else [1.0 / len(ms)] * len(ms)
    aims = {"two": [sum(wi * m["eye_xy"][0] for wi, m in zip(w, ms)), sum(wi * m["eye_xy"][1] for wi, m in zip(w, ms)),
                    sum(wi * (m["eye_z"] - 0.12) for wi, m in zip(w, ms))]}
    opts = opts or {}
    seated_lis = bool(lis) and lis in (opts.get("seated") or ())
    if spk and seated_lis:                   # a clean single of the speaker (never over a high chair's back)
        sm = ms[0]
        aims["single"] = [sm["eye_xy"][0], sm["eye_xy"][1], sm["eye_z"] - 0.15]
    if opts.get("points"):                   # a move: aimed at the middle of its places
        pts = opts["points"]
        aims["move"] = [sum(p_[i] for p_ in pts) / len(pts) for i in range(2)] + [1.0]
    if opts.get("joints"):                   # the action's joints in the middle of the aim
        pts = [subjects[k]["measured"]["joints"][j] for k, j in opts["joints"]
               if j in (subjects[k]["measured"].get("joints") or {})]
        if pts:
            aims["action"] = [sum(p_[i] for p_ in pts + [aims["two"]]) / (len(pts) + 1) for i in range(3)]
    if spk and lis and not seated_lis:
        sm, lm = ms[0], ms[1]
        aims["ots"] = [sm["eye_xy"][0] * 0.85 + lm["eye_xy"][0] * 0.15, sm["eye_xy"][1] * 0.85 + lm["eye_xy"][1] * 0.15,
                       sm["eye_z"] - 0.1]
    W, D, _ = room
    out = []
    m0 = subjects[spk]["measured"] if spk else ms[0]
    for x in range(-int(W * 5), int(W * 5) + 1):
        for y in range(-int(D * 5), int(D * 5) + 1):
            for z in sorted({1.45, 1.6} | ({round(subjects[lis]["measured"]["eye_z"] + 0.12, 2)} if lis else set())):
                p = [x / 10.0, y / 10.0, z]
                if not in_room(p, room) or any(math.dist(p, t) < 0.15 for t in tried):
                    continue
                if spk and plan_angle(m0["head_fwd_xy"], m0["eye_xy"], p) > MAX_HEAD_DEG - 5:
                    continue
                if est_side and kind not in ("wide", "full") and side(line, p) not in (0, est_side):
                    continue
                for mode, aim in aims.items():
                    others = [v["measured"] for k, v in subjects.items() if k != spk and not (mode == "ots" and k == lis)]
                    if spk and blocked(p, [m0["eye_xy"][0], m0["eye_xy"][1], m0["eye_z"]], others):
                        continue
                    if opts.get("_lens"):
                        lens = opts["_lens"]
                    elif opts.get("points"):
                        lens = (20, 24, 28)
                    elif kind in ("wide", "full"):
                        lens = (20, 24, 28, 35)
                    elif kind == "close":
                        lens = (50, 65, 85, 100)
                    elif mode == "single" or opts.get("_single"):
                        lens = (35, 50, 65, 85)
                    else:
                        lens = (28, 35, 40, 50)
                    for f in lens:
                        q = {"position": [round(v, 3) for v in p], "target": [round(v, 3) for v in aim], "focal_mm": f}
                        why, _ = framing_why(q, kind, spk, lis, mode == "ots", c.get("show"), subjects, res,
                                             opts=opts)
                        pc = opts.get("prev_cam") if opts.get("points") else None
                        sim_pen = 0.0
                        ep = opts.get("end_place")
                        if ep and ep.get("similar") and opts.get("mover") in subjects:
                            # a seat like the end place nearer the mover in the frame than the end place, in view
                            mm = subjects[opts["mover"]]["measured"]["pelvis_xy"]
                            mq = framing.project(q, [mm[0], mm[1], 0.5], res)
                            eq = framing.project(q, [ep["center"][0], ep["center"][1], 0.5], res)
                            for sc_ in ep["similar"]:
                                sq = framing.project(q, [sc_[0], sc_[1], 0.5], res)
                                if mq and eq and sq and 0 < sq[0] < 1 and 0 < sq[1] < 1 and \
                                        math.dist(mq, sq) < math.dist(mq, eq):
                                    sim_pen += 3.0
                        if not why:          # a move keeps the shot before's side of the room when it can
                            out.append((math.dist(p, cam0["position"])
                                        + (3.0 if pc and p[0] * pc[0] + p[1] * pc[1] < 0 else 0.0) + sim_pen
                                        + (0.0 if mode == ("ots" if c.get("ots") else "two") else 0.5)
                                        + 10.0 * area_excess(q, subjects, res, {spk, lis, opts.get("subject")}, opts),
                                        q, mode))
                            break
    if not out and spk and lis and not opts.get("_single"):
        # no two-shot or over-the-shoulder passes (two people facing each other, one of them near): a clean single
        # of the speaker, the listener left to the cut
        return two_shot_candidates(dict(c, show=None), shot, spk, None, subjects, res, room, line, est_side, tried,
                                   dict(opts, _single=True))
    return [(q, mode if not opts.get("_single") else "single") for _, q, mode in sorted(out, key=lambda t: t[0])]


def in_room(p, room):
    W, D, H = room
    return abs(p[0]) < W / 2 - ROOM_MARGIN_M and abs(p[1]) < D / 2 - ROOM_MARGIN_M and 0.3 < p[2] < H - 0.2


# ------------------------------------------------------------------ the check
COVER_CM = 6.0               # a veil over the face: the face counts as shown through it, this far in front of it


def check_shot(shot, speaker, subjects, res, kind, line, est_side, listener=None, ots=False, show=None,
               covered=False, opts=None):
    if not speaker:                          # a shot without a speaker: only the framing rules
        why, facts = framing_why(shot["camera"], kind, None, None, False, show, subjects, res, shot, opts)
        return {"speaker": None, "ok": not why, "why": why, "framing": why, "frame_facts": facts}
    m = subjects[speaker]["measured"]
    cam = shot["camera"]
    g = shot.get("guides") or {}
    out = {"speaker": speaker, "head_deg": (shot.get("truth") or {}).get(speaker, {}).get("head_to_camera_deg")}
    if out["head_deg"] is None:
        out["head_deg"] = round(plan_angle(m["head_fwd_xy"], m["eye_xy"], cam["position"]), 1)
    vis = n = 0
    if g.get("depth") and g.get(f"depth_{speaker}"):
        sc, sp = Exr(g["depth"]), Exr(g[f"depth_{speaker}"])
        for p in face_points(m):
            n += 1
            pr = framing.project(cam, p, res)
            if not pr:
                continue
            ds, dp = sc.at(pr[0], pr[1]), sp.at(pr[0], pr[1])
            if ds is None or dp is None or dp > 5e4:
                continue                     # off the frame, or the speaker is not drawn there (the face turned away)
            if abs(dp - ds) <= max(3.0, 0.015 * ds) or (covered and 0 <= dp - ds <= COVER_CM):
                vis += 1                     # what the camera sees there is the speaker (or the veil over her face)
        out["face_visible"] = round(vis / n, 2) if n else 0.0
    else:
        out["face_visible"] = None
    top = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["head_top_z"]], res)
    chin = framing.project(cam, [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]], res)
    out["face_h"] = round(chin[1] - top[1], 3) if top and chin else 0.0
    out["side"] = side(line, cam["position"]) if line else 0
    why = []
    if out["head_deg"] > MAX_HEAD_DEG:
        why.append(f"head {out['head_deg']:.0f} deg from the camera")
    if out["face_visible"] is not None and out["face_visible"] < MIN_FACE_VISIBLE:
        why.append(f"face {out['face_visible']:.0%} visible")
    if out["face_h"] < MIN_FACE_H.get(kind, 0.03):
        why.append(f"face {out['face_h']:.3f} of the frame height")
    if est_side and kind not in ("wide", "full") and out["side"] and out["side"] != est_side:
        why.append("across the line")
    fw, facts = framing_why(cam, kind, speaker, listener, ots, show, subjects, res, shot, opts)
    out["framing"] = fw
    out["frame_facts"] = facts
    why += fw
    out["ok"] = not why
    out["why"] = why
    return out


# ------------------------------------------------------------------ re-framing
def ots_camera(spk, lis, est_side, line, back=0.55, lateral=0.38, focal=40):
    """Behind the listener's head, toward the established side of the line, aimed at the speaker's eyes."""
    d = unit2([lis["eye_xy"][0] - spk["eye_xy"][0], lis["eye_xy"][1] - spk["eye_xy"][1]])
    n = [-d[1], d[0]]
    if est_side and side(line, [lis["eye_xy"][0] + n[0], lis["eye_xy"][1] + n[1]]) != est_side:
        n = [-n[0], -n[1]]
    pos = [lis["eye_xy"][0] + d[0] * back + n[0] * lateral, lis["eye_xy"][1] + d[1] * back + n[1] * lateral,
           lis["eye_z"] + 0.06]
    tgt = [spk["eye_xy"][0] - n[0] * 0.12, spk["eye_xy"][1] - n[1] * 0.12, spk["eye_z"] - 0.12]
    return {"position": [round(x, 3) for x in pos], "target": [round(x, 3) for x in tgt], "focal_mm": focal}


def candidates(c, shot, spk_key, subjects, res, room, line, est_side, tried):
    """New placed cameras for a failed speaker shot, best first."""
    m = subjects[spk_key]["measured"]
    others = [v["measured"] for k, v in subjects.items() if k != spk_key]
    kind = c["kind"]
    eye = [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]]
    out = []
    if c.get("ots"):
        lis = subjects[c["ots"]]["measured"]
        for back, lat, foc in ((0.55, 0.38, 40), (0.7, 0.48, 40), (0.45, 0.55, 35), (0.85, 0.6, 45)):
            out.append(ots_camera(m, lis, est_side, line, back, lat, foc))
        others = []                          # the listener's shoulder is meant to be in the frame
    elif kind in ("close", "medium", "full"):
        head = math.degrees(math.atan2(m["head_fwd_xy"][1], m["head_fwd_xy"][0]) -
                            math.atan2(m["facing"][1], m["facing"][0]))
        az0 = c.get("az", 0.0)
        azs = sorted({round(head + k) for k in range(-40, 41, 10)}, key=lambda a: abs(a - az0))
        for az in azs:
            cs = framing.shot(m, kind, c.get("focal_mm", 50), res, az, c.get("el", 0.0), c.get("screen_x", 0.5))
            out.append({"position": cs["position"], "target": cs["target"], "focal_mm": cs["focal_mm"]})
    else:                                    # a placed (wide) camera: another standpoint aimed at the same point,
        cam = shot["camera"]                 # nearest first, that keeps everyone in the frame
        W, D, _ = room
        z = cam["position"][2]
        grid = [[x / 10.0, y / 10.0, z] for x in range(-int(W * 5), int(W * 5) + 1, 2)
                for y in range(-int(D * 5), int(D * 5) + 1, 2)]
        ms = [v["measured"] for v in subjects.values()]
        mid = [sum(x["pelvis_xy"][0] for x in ms) / len(ms), sum(x["pelvis_xy"][1] for x in ms) / len(ms),
               sum((x["feet_z"] + x["head_top_z"]) / 2 for x in ms) / len(ms)]
        f0 = cam["focal_mm"]
        for p in sorted(grid, key=lambda p: math.dist(p, cam["position"])):
            for tgt in (cam["target"], [round(v, 3) for v in mid]):   # the same aim, else the people's middle
                for f in sorted({f0, max(18.0, round(f0 * 0.8)), max(18.0, round(f0 * 0.65))}, reverse=True):
                    q = {"position": [round(v, 3) for v in p], "target": tgt, "focal_mm": f}
                    if all_in_frame(q, subjects, res):
                        out.append(q)
                        break
                else:
                    continue
                break
    keep = []
    for q in out:
        p = q["position"]
        if not in_room(p, room):
            continue
        if any(math.dist(p, t) < 0.15 for t in tried):
            continue
        if plan_angle(m["head_fwd_xy"], m["eye_xy"], p) > MAX_HEAD_DEG - 5:
            continue
        if est_side and kind not in ("wide", "full") and side(line, p) not in (0, est_side):
            continue
        if blocked(p, eye, others):
            continue
        keep.append(q)
    return keep


# ------------------------------------------------------------------ driver
def run_scene(spec_path, cams=None):
    cmd = [sys.executable, str(HERE / "scene.py"), str(spec_path)] + (["--cameras", ",".join(cams)] if cams else [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    log(p.stdout.strip().splitlines()[0] if p.stdout.strip() else p.stderr[-400:])
    if p.returncode:
        raise SystemExit(p.stdout[-2000:] + p.stderr[-2000:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--rounds", type=int, default=14, help="render rounds at most (each shot: MAX_TRIES)")
    ap.add_argument("--no-split", action="store_true", help="stage all cameras in one scene (no per-speaker looks)")
    ap.add_argument("--prev", help="the shot before's staged.json (its people's places): a vague mark keeps it")
    a = ap.parse_args()
    t0 = time.time()
    spec = json.loads(Path(a.spec).read_text())
    odir = Path(os.path.expanduser(os.environ.get("SCENES_OUT", str(Path(C.UE_HOME) / "lab" / "scenes" / "out")))) / spec["id"]
    odir.mkdir(parents=True, exist_ok=True)
    room = json.loads((Path(C.LIBRARY) / "set" / spec["set"] / "spec.json").read_text())["size_m"]
    res = spec.get("res", [1152, 640])
    spec["guides"] = True
    # listeners face the speaker: one staging per speaker (a quiet group keeps its poses); several groups run as
    # their own scenes "<id>_<speaker>" / "<id>_q", each with its listeners' heads turned
    groups = {}
    for c in spec["cameras"]:
        groups.setdefault(c.get("speaker"), []).append(c)
    if len(groups) > 1 and not a.no_split:
        rc = 0
        prev = a.prev
        for spk, cams in groups.items():          # in turn: each group keeps the places the one before chose
            sub = dict(spec, id=f"{spec['id']}_{spk or 'q'}", cameras=cams)
            sp_path = odir.parent / f"{sub['id']}.spec.json"
            sp_path.write_text(json.dumps(sub, indent=1))
            log(f"group {sub['id']}: {[c['name'] for c in cams]}")
            rc |= subprocess.run([sys.executable, __file__, str(sp_path), "--rounds", str(a.rounds), "--no-split"]
                                 + (["--prev", prev] if prev else [])).returncode
            st = odir.parent / sub["id"] / "staged.json"
            if st.exists():
                prev = str(st)
        if prev and prev != a.prev:
            (odir / "staged.json").write_text(Path(prev).read_text())
        return rc
    prev = json.loads(Path(a.prev).read_text()).get("positions") if a.prev and Path(a.prev).exists() else None
    relook = list(groups)[0] if len(groups) == 1 else None
    if relook:
        look_at(spec, relook)
    keep = {}                                # a vague place keeps the shot before's place when it is allowed
    for k, v in spec["characters"].items():
        pv = (prev or {}).get(k)
        if v.get("marks") and pv and math.dist(pv, v["position"]) > 0.05 and \
                any(math.dist(pv, m) < 0.05 for m in v["marks"]):
            keep[k] = pv
    if keep:
        apply_marks(spec, keep, relook, relook)
    seat_plan = seating.plan(spec, Path(C.LIBRARY))           # seated people placed from their chairs' seat boxes
    assets_action_steps(spec)
    walks = clear_walks(spec, set(seat_plan.get("seats") or {}))   # before any render: clear floor paths for walks
    aimed = aim_bodies(spec)                 # bodies turned to whom they face, from where everyone now stands
    if aimed:
        log(f"bodies re-aimed: {aimed}")
    if (walks["moved"] or aimed) and relook:   # heads re-aimed at the speaker from the new places
        for k in list(spec["characters"]):
            spec["characters"][k] = unlook(spec["characters"][k])
        look_at(spec, relook)
    for c in spec["cameras"]:
        c["_points"] = move_points(spec, c)
        c["_landmark"] = landmark(spec, c) or (landmark(spec, {"move_landmark": c["face_thing"]})
                                               if c.get("face_thing") else None)
        t = end_thing(spec, c) if c.get("mover") else None
        c["_end_place"] = ({"center": t["center"], "top": t["center"][2] + t["half"][2],
                            "similar": [x["center"] for x in similar_seats(spec, t)]} if t else None)
    spec["_prev_cam"] = (json.loads(Path(a.prev).read_text()).get("camera") if a.prev and Path(a.prev).exists()
                         else None)
    keys = spec.get("line") or list(spec["characters"])[:2]
    line = [spec["characters"][k]["position"][:2] for k in keys] if len(keys) == 2 else None
    est_side = None
    for c in spec["cameras"]:                # a wide sets the side of the line; with none, the first camera that
        if line and c.get("position") and c["kind"] in ("wide", "full"):     # passes does (in the rounds below):
            est_side = side(line, c["position"]) or est_side                 # a failing first camera set it and
            break                                                            # refused a good one "across the line"
    # first pass: over-the-shoulder cameras from the spec's marks (eye height assumed until measured)
    auto_ots = set()
    for c in spec["cameras"]:
        if c.get("ots") and c.get("speaker") and not c.get("position"):
            auto_ots.add(c["name"])
            ch = spec["characters"]
            guess = lambda k: {"eye_xy": ch[k]["position"][:2], "eye_z": 1.6}
            c.update(ots_camera(guess(c["speaker"]), guess(c["ots"]), est_side, line))
            c["kind"] = c.get("kind", "medium")
    report = {"scene": spec["id"], "line": keys, "established_side": est_side, "rounds": [], "shots": {},
              "seating": {}, "marks": {}, "walks": walks}
    if walks["moved"] or walks["blocked"]:
        log(f"walks: moved {walks['moved']} blocked {walks['blocked']}")
    todo = [c["name"] for c in spec["cameras"]]
    tried = {c["name"]: [] for c in spec["cameras"]}
    tries = {c["name"]: {"same": 0, "single": 0, "mark": 0, "wide": 0, "n": 0} for c in spec["cameras"]}
    tried_marks, pending, master = {}, None, None
    reach_done = not any(v.get("reach") for v in spec["characters"].values())
    least_bad = []                           # shots whose least-bad try is not the last render: rendered again
    start_pos = {k: list(v["position"]) for k, v in spec["characters"].items()}
    tries_dir = odir / "tries"

    def move_to_mark(k, mk):
        """One person to a mark: turned to keep their angle to their reference person, heads re-aimed, the
        over-the-shoulder cameras set from the old places set again; every camera renders again."""
        nonlocal line
        subj2 = shifted(subjects, spec, {k: mk}, relook)
        apply_marks(spec, {k: mk}, relook, None)
        aim_bodies(spec)                     # whoever faces the moved person turns with them
        if relook:
            for kk in list(spec["characters"]):
                spec["characters"][kk] = unlook(spec["characters"][kk])
            look_at(spec, relook)
        for c_ in spec["cameras"]:
            if c_["name"] in auto_ots and not c_.get("single"):
                c_.update(ots_camera(subj2[c_["speaker"]]["measured"], subj2[c_["ots"]]["measured"], est_side, line))
            c_["_points"] = move_points(spec, c_)
        if line:
            line = [spec["characters"][kk]["position"][:2] for kk in keys]
        return subj2

    def reframe_at(c, subj2, tried_c=()):
        """A camera for the shot with the people at their new marks (the old one looked past the moved person at
        the same blocker): its own kind first, else a clean single; None keeps the camera."""
        spk = c.get("speaker")
        lis = listener_of(c, keys, spk)
        tc = two_shot_candidates(c, shots[c["name"]], spk, lis, subj2, res, room, line, est_side, list(tried_c),
                                 cam_opts(spec, c))
        if tc:
            return tc[0]
        if not spk and len(c.get("show") or []) >= 2:   # everyone the shot names stays in it
            tc = two_shot_candidates(c, shots[c["name"]], None, None, subj2, res, room, line, est_side, list(tried_c),
                                     cam_opts(spec, c))
            return (tc[0][0], "two") if tc else None
        if not spk:                          # no speaker, one person named: a tighter frame on the shot's subject
            tc = two_shot_candidates(dict(c, show=[c.get("subject")]), shots[c["name"]], None, None, subj2, res,
                                     room, line, est_side, list(tried_c), cam_opts(spec, c)) if c.get("subject") else []
            return (tc[0][0], "tight") if tc else None
        tc = two_shot_candidates(dict(c, show=None), shots[c["name"]], spk, None, subj2, res, room, line, est_side,
                                 list(tried_c), dict(cam_opts(spec, c), _single=True))
        return (tc[0][0], "single") if tc else None

    def in_way(c, spk):
        """The people the shot's failures name or whose bodies stood between its cameras and the speaker's face,
        most often first; the speaker themselves last."""
        ch = spec["characters"]
        cnt = {}
        for h in report["shots"].get(c["name"], {}).get("history", []):
            for w in h.get("why") or []:
                for k in ch:
                    if k != spk and (w.startswith(f"{k} ") or w.startswith(f"{k},") or f"{k}'s" in w
                                     or f": {k}'s" in w):
                        cnt[k] = cnt.get(k, 0) + 1
            if spk:
                m = subjects[spk]["measured"]
                for k, v in subjects.items():
                    if k != spk and blocked(h["camera"]["position"], [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]],
                                            [v["measured"]]):
                        cnt[k] = cnt.get(k, 0) + 1
        order = sorted(cnt, key=lambda k: -cnt[k])
        return order + ([spk] if spk else [])

    def next_mark(c, spk):
        """The next allowed mark for a vague-place person in the way, nearest the previous shot's place first."""
        ch = spec["characters"]
        for k in in_way(c, spk):
            v = ch.get(k) or {}
            if not v.get("marks") or any(w in ((v.get("body_pose") or {}).get("pose") or "") for w in SEATED):
                continue                     # an exact place (a seat, a stated spot) never moves for the camera
            ref = (prev or {}).get(k) or start_pos[k]
            others = [ch[o]["position"] for o in ch if o != k]
            for mk in sorted(v["marks"], key=lambda m: math.dist(m, ref)):
                if math.dist(mk, v["position"]) < 0.05 or any(math.dist(mk, t_) < 0.05 for t_ in tried_marks.get(k, [])):
                    continue
                if any(math.dist(mk, o) < MARK_CLEAR_M for o in others):
                    continue
                return k, list(mk)
        return None

    def badness(h):
        """A rendered try's misses, worst first: the speaker's face, then named people (missing, unreadable, a face
        cut), then how many misses in all."""
        why = h.get("why") or []
        face = [w for w in why if w not in (h.get("framing") or [])]
        named = [w for w in why if "(named for the shot)" in w or w.startswith("PARTIAL FACE")]
        return (len(face), len(named), len(why))

    def give_up(c):
        """A shot whose tries ran out: every shot ends with a key (a dropped shot leaves a hole in the video), so
        the least-bad rendered try becomes it, labelled with its misses and counted as not shippable."""
        n_ = c["name"]
        hist = report["shots"].get(n_, {}).get("history") or []
        best = min(hist, key=badness) if hist else None
        chk = shots[n_]["speaker_check"]
        why = (best or chk)["why"]
        shots[n_]["framing_miss"] = why
        shots[n_]["least_bad"] = {"round": (best or {}).get("round"), "misses": why}
        shots[n_]["tried_stills"] = sorted(str(p_) for p_ in tries_dir.glob(f"{n_}_r*.png"))
        if best and math.dist(best["camera"]["position"], shots[n_]["camera"]["position"]) > 1e-3:
            c.update(position=best["camera"]["position"], target=best["camera"]["target"],
                     focal_mm=best["camera"]["focal_mm"])
            for kk in ("az", "el", "screen_x"):
                c.pop(kk, None)
            least_bad.append(n_)
    for rnd in range(a.rounds + 1):
        sp = odir / f"spec_round{rnd}.json"
        sp.write_text(json.dumps(spec, indent=1))
        run_scene(sp, todo if todo != [c["name"] for c in spec["cameras"]] else None)
        res_j = json.loads((odir / "result.json").read_text())
        if master is None or set(todo) == {c["name"] for c in spec["cameras"]}:
            master = res_j
        else:                                # merge the re-rendered shots into the full record
            new = {s["name"]: s for s in res_j["shots"]}
            master["shots"] = [new.get(s["name"], s) for s in master["shots"]]
        subjects = master["subjects"]
        shots = {s["name"]: s for s in master["shots"]}
        # seating first: a person not sitting in their chair is moved and every camera rendered again
        moved = False
        for k, n in seat_plan["taken"].items():
            st = spec["characters"][k]["_seat"]
            chk = seating.check(subjects[k]["measured"], seat_plan["seats"][n], st["fit"], st["x"])
            report["seating"].setdefault(k, []).append(dict(chk, round=rnd, seat=n, note=st.get("note")))
            log(f"seat {k} in {n}: ok={chk['ok']} {chk['why']} hips(seat frame)={chk.get('hips_seat_frame')}")
            if not chk["ok"] and rnd < a.rounds and max(abs(e) for e in chk.get("error_m", [0])) > 0.01:
                nud = [st["nudge"][0] + chk["error_m"][0], st["nudge"][1] + chk["error_m"][1]]
                c0 = {kk: vv for kk, vv in spec["characters"][k].items() if kk != "_seat"}
                spec["characters"][k] = seating.place(c0, seat_plan["seats"][n], st["fit"], st["x"], nud)
                spec["characters"][k]["_seat"]["fit"] = st["fit"]
                moved = True
        if moved:
            todo = [c["name"] for c in spec["cameras"]]
            report["rounds"].append({"round": rnd, "rendered": "all", "re-seated": True})
            continue
        if not reach_done and rnd < a.rounds:  # a hand on a forearm: solved on the bodies as measured
            reach_done = True
            got = solve_reach(spec, subjects)
            if got:
                report["reach"] = got
                todo = [c["name"] for c in spec["cameras"]]
                report["rounds"].append({"round": rnd, "rendered": "all", "reach": True})
                continue
        fails = []
        for c in spec["cameras"]:
            if c["name"] not in todo:
                continue
            sh = shots[c["name"]]
            sh["speaker"] = c.get("speaker")
            tried[c["name"]].append(sh["camera"]["position"])
            lis = listener_of(c, keys, c.get("speaker"))
            chk = check_shot(sh, c.get("speaker"), subjects, res, c["kind"], line, est_side, lis, bool(c.get("ots")),
                             c.get("show"),
                             bool((spec["characters"].get(c.get("speaker") or "") or {}).get("face_covered")),
                             cam_opts(spec, c))
            pw = [w for w in placement_why(sh["camera"], subjects, room)
                  if not w.startswith("the camera is inside ") or
                  ((chk.get("frame_facts") or {}).get(w.rsplit(" ", 1)[1], {}).get("mask") or {}).get("area", 0) > 0.3]
            # the subject of an action or quiet shot (whom the plan's line is about) has a face shown, as a speaker
            if not c.get("speaker") and c.get("line_subject") in subjects:
                pw += subject_face_why(sh, c["line_subject"], subjects, res,
                                       c["line_subject"] in (cam_opts(spec, c).get("covered") or ()))
            if pw:                           # outside the room, a body filling the frame, the subject's face unseen
                chk["why"] = pw + chk["why"]
                chk["framing"] = pw + chk.get("framing", [])
                chk["ok"] = False
            sh["speaker_check"] = chk
            report["shots"].setdefault(c["name"], {"history": []})
            report["shots"][c["name"]]["history"].append(dict(chk, round=rnd, camera=sh["camera"]))
            report["shots"][c["name"]].update(speaker=c.get("speaker"), ok=chk["ok"], rounds=rnd)
            if not chk["ok"]:
                fails.append(c)
                tries_dir.mkdir(exist_ok=True)
                try:
                    import shutil
                    shutil.copy(sh["path"], tries_dir / f"{c['name']}_r{rnd}.png")
                except OSError:
                    pass
        if line and est_side is None:        # the side of the line from the first camera that passed
            for c in spec["cameras"]:
                sh_ = shots.get(c["name"]) or {}
                if (sh_.get("speaker_check") or {}).get("ok") and side(line, sh_["camera"]["position"]):
                    est_side = side(line, sh_["camera"]["position"])
                    report["established_side"] = est_side
                    break
        spec["_area_scale"] = area_scale([shots[n] for n in todo], subjects, res, spec.get("_area_scale"))
        report["rounds"].append({"round": rnd, "rendered": todo, "failed": [c["name"] for c in fails],
                                 "area_scale": spec["_area_scale"]})
        log(f"round {rnd}: rendered {len(todo)}, failed {[c['name'] + ': ' + '; '.join(shots[c['name']]['speaker_check']['why']) for c in fails]}")
        passing = {n for n, s_ in shots.items() if (s_.get("speaker_check") or {}).get("ok")}
        if pending:                          # a mark tried for a talk shot: kept only if the renders pass more
            better = len(passing) > pending["passing"] or (len(passing) == pending["passing"]
                                                           and pending["shot"] in passing)
            rec = dict(pending, passing_after=len(passing), kept=better, round=rnd)
            report["marks"].setdefault("tries", []).append(rec)
            log(f"mark {pending['k']} -> {pending['to']} for {pending['shot']}: passing {pending['passing']} -> "
                f"{len(passing)}, {'kept' if better else 'back'}")
            tc_ = None
            if not better and rnd < a.rounds and pending.get("cam_tries", 0) < 1:
                # the scene is now measured at the new mark: one more camera for the shot there before undoing it
                c_t = next(c_ for c_ in spec["cameras"] if c_["name"] == pending["shot"])
                tried[c_t["name"]].append(shots[c_t["name"]]["camera"]["position"])
                tc_ = reframe_at(c_t, subjects, tried[c_t["name"]])
            if tc_:
                q, md = tc_
                c_t.update(position=q["position"], target=q["target"], focal_mm=q["focal_mm"])
                for kk in ("az", "el", "screen_x"):
                    c_t.pop(kk, None)
                if md == "single":
                    c_t["listener"], c_t["single"] = None, True
                    c_t.pop("ots", None)
                elif md == "tight":
                    c_t["show"], c_t["tight"] = [c_t.get("subject")], True
                elif md == "ots":
                    c_t["ots"] = listener_of(c_t, keys, c_t.get("speaker"))
                pending["cam_tries"] = pending.get("cam_tries", 0) + 1
                report["marks"]["tries"][-1]["kept"] = "camera tried again"
                todo = [c_t["name"]]
                continue
            if not better and rnd < a.rounds:
                move_to_mark(pending["k"], pending["from"])
                cam_back = next(c_ for c_ in spec["cameras"] if c_["name"] == pending["shot"])
                for kk in CAM_KEYS:
                    cam_back.pop(kk, None)
                cam_back.update(pending["camera"])
                pending = None
                todo = [c["name"] for c in spec["cameras"]]
                continue
            pending = None
        if not fails or rnd == a.rounds:
            for c in fails:
                give_up(c)
            break
        todo, mark_req = [], None
        for c in fails:
            n_ = c["name"]
            t = tries[n_]
            chk = shots[n_]["speaker_check"]
            face_fail = [w for w in chk["why"] if w not in chk.get("framing", [])]
            # the person the shot is about is searched for as a speaker in every try (only on the tries that
            # failed for their face, the search fell back to cameras behind her on the others, 11-12)
            spk = c.get("speaker") or (c.get("line_subject") if c.get("line_subject") in subjects else None)
            lis = listener_of(c, keys, spk)
            if t["n"] >= MAX_TRIES:
                give_up(c)
                continue
            cands, stage = [], None
            if t["same"] < (TRY_SAME if spk else TRY_SAME + 1):   # 1: another camera of the same kind
                if (face_fail or c.get("ots")) and spk:
                    cc = candidates(c, shots[n_], spk, subjects, res, room, line, est_side, tried[n_])
                    good = [q for q in cc if not framing_why(q, c["kind"], spk, lis, bool(c.get("ots")),
                                                             c.get("show"), subjects, res, opts=cam_opts(spec, c))[0]]
                    cands = [(q, "same") for q in good]
                if not cands and (chk.get("framing") or face_fail):
                    partial = any(w.startswith("PARTIAL FACE") for w in chk.get("framing") or [])
                    if not (partial and spk):
                        tc = two_shot_candidates(c, shots[n_], spk, lis, subjects, res, room, line, est_side,
                                                 tried[n_], cam_opts(spec, c))
                        cands = [(q, md) for q, md in tc[:1]]
            if not cands and spk and t["single"] < TRY_SINGLE:     # 2: a clean single of the speaker
                tc = two_shot_candidates(dict(c, show=None), shots[n_], spk, None, subjects, res, room, line, est_side,
                                         tried[n_], dict(cam_opts(spec, c), _single=True))
                cands = [(q, "single") for q, _ in tc[:1]]
            subj_k = spk or c.get("subject")
            named = [k for k in (c.get("show") or []) if k in subjects]
            many = not spk and len(named) >= 2       # a shot naming two or more: every one of them stays in it
            if not cands and many and t["wide"] < TRY_SINGLE:      # 2: a wider frame holding everyone named
                tc = two_shot_candidates(dict(c, kind="full"), shots[n_], None, None, subjects, res, room, line,
                                         est_side, tried[n_], cam_opts(spec, c))
                cands = [(q, "wide") for q, _ in tc[:1]]
            no_mark = t["mark"] >= TRY_MARK or (mark_req is not None or next_mark(c, subj_k) is None)
            if not cands and many and no_mark and t["single"] < TRY_SINGLE:
                # 4: last, a tighter frame that still holds everyone named
                tc = two_shot_candidates(c, shots[n_], None, None, subjects, res, room, line, est_side, tried[n_],
                                         dict(cam_opts(spec, c), _lens=(50, 65, 85)))
                cands = [(q, "tighter") for q, _ in tc[:1]]
            if not cands and not spk and not many and subj_k in subjects and t["single"] < TRY_SINGLE:
                # 2 (no speaker): a tighter frame on the shot's subject, the others clearly in or out
                tc = two_shot_candidates(dict(c, show=[subj_k]), shots[n_], None, None, subjects, res, room, line,
                                         est_side, tried[n_], cam_opts(spec, c))
                cands = [(q, "tight") for q, _ in tc[:1]]
            if not cands and subj_k and t["mark"] < TRY_MARK and mark_req is None:   # 3: a vague-place person moves
                pick = next_mark(c, subj_k)
                if pick:
                    mark_req = (c, pick)
                    continue
            if not cands:
                give_up(c)
                continue
            q, md = cands[0]
            if md == "single":           # a clean single: the listener is no longer required in it
                c["listener"] = None
                c["single"] = True
                c.pop("ots", None)
                t["single"] += 1
            elif md == "tight":          # the shot's subject alone named: the others need not be in it
                c["show"] = [c.get("subject")]
                c["tight"] = True
                t["single"] += 1
            elif md == "wide":
                c["kind"] = "full"
                t["wide"] += 1
            elif md == "tighter":
                t["single"] += 1
            else:
                if md == "ots":
                    c["ots"] = lis
                elif md in ("two", "action", "move"):
                    c.pop("ots", None)
                t["same"] += 1
            t["n"] += 1
            c.update(position=q["position"], target=q["target"], focal_mm=q["focal_mm"])
            for k in ("az", "el", "screen_x"):
                c.pop(k, None)
            todo.append(n_)
        if mark_req:
            c, (k, mk) = mark_req
            t = tries[c["name"]]
            t["mark"] += 1
            t["n"] += 1
            t["same"] = min(t["same"], TRY_SAME - 1)        # one more camera search, one more single, at the mark
            t["single"] = min(t["single"], TRY_SINGLE - 1)
            pending = {"k": k, "from": list(spec["characters"][k]["position"]), "to": mk, "shot": c["name"],
                       "passing": len(passing), "camera": {kk: c[kk] for kk in CAM_KEYS if kk in c}}
            tried_marks.setdefault(k, []).append(mk)
            subj2 = move_to_mark(k, mk)
            rf = reframe_at(c, subj2)        # the move and a camera found for it, rendered together
            if rf:
                q, md = rf
                c.update(position=q["position"], target=q["target"], focal_mm=q["focal_mm"])
                for kk in ("az", "el", "screen_x"):
                    c.pop(kk, None)
                if md == "single":
                    c["listener"], c["single"] = None, True
                    c.pop("ots", None)
                elif md == "tight":
                    c["show"], c["tight"] = [c.get("subject")], True
                elif md == "ots":
                    c["ots"] = listener_of(c, keys, c.get("speaker"))
                elif md in ("two", "action", "move"):
                    c.pop("ots", None)
                pending["new_camera"] = dict(q, mode=md)
            todo = [c_["name"] for c_ in spec["cameras"]]
        if not todo:
            break
    if least_bad:                            # the least-bad tries rendered again as the keys' engine stills
        sp = odir / "spec_final.json"
        sp.write_text(json.dumps(spec, indent=1))
        run_scene(sp, least_bad)
        new = {s_["name"]: s_ for s_ in json.loads((odir / "result.json").read_text())["shots"]}
        for i_, s_ in enumerate(master["shots"]):
            if s_["name"] in new:
                keep = {kk: s_.get(kk) for kk in ("framing_miss", "least_bad", "tried_stills", "speaker",
                                                   "speaker_check")}
                master["shots"][i_] = dict(new[s_["name"]], **keep)
        log(f"least-bad tries as keys: {least_bad}")
    report["staged_facts"] = staged_facts(spec, master["subjects"])
    for sh in master["shots"]:
        sh["staged_facts"] = {k: v for k, v in report["staged_facts"].items() if v}
    log(f"staged facts: {report['staged_facts']}")
    (odir / "result.json").write_text(json.dumps(master, indent=1))
    last = {s_["name"]: s_ for s_ in master["shots"]}.get(spec["cameras"][-1]["name"]) or {}
    (odir / "staged.json").write_text(json.dumps({"positions": {k: v["position"] for k, v in
                                                                spec["characters"].items()},
                                                  "camera": (last.get("camera") or {}).get("position"),
                                                  "shot_facts": walks["shot_facts"]}, indent=1))
    report["seconds"] = round(time.time() - t0, 1)
    (odir / "speakers.json").write_text(json.dumps(report, indent=1))
    for n, s in report["shots"].items():
        last = (s.get("history") or [{}])[-1]
        log(f"  {n}: speaker {s.get('speaker')} ok={s['ok']} rounds={s['rounds']} head={last.get('head_deg')} "
            f"face={last.get('face_visible')} size={last.get('face_h')} {last.get('why') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
