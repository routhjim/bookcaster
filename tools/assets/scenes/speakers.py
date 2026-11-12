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
            if mf["area"] < LISTENER_AREA["two"] * 0.3:
                f["head_in"] = f["head_any"] = f["shoulder_in"] = False      # hidden (behind a chair, a person)
    seated_lis = bool(listener) and listener in (opts.get("seated") or ())
    if seated_lis and ots:
        why.append("over the shoulder of a listener seated in a high-backed chair (the chair back fills the frame)")
    # every head in a talk shot is smaller than the speaker's; a person turned away is a modest share of the frame
    # with the head inside it (a recognisable outline, not a dark mass merged with the frame edge)
    sh_spk = head_h(cam, subjects[speaker]["measured"], res) if speaker else 0.0
    keep = {speaker, opts.get("subject")}
    for k, v in subjects.items():
        if k == speaker or not facts[k]["any_in"]:
            continue
        h = head_h(cam, v["measured"], res)
        if speaker and facts[k]["head_any"] and h > sh_spk * 1.02:
            why.append(f"{k}'s head ({h:.3f}) is bigger than the speaker's ({sh_spk:.3f})")
        if k not in keep and turned_away(cam, v["measured"]):
            ma = (facts[k].get("mask") or {}).get("area")
            if ma is None:                   # no still yet (a candidate camera): the area from the body
                ma = est_area(cam, v["measured"], res)
            if ma > FG_AREA:
                why.append(f"{k}, turned away, covers {ma:.0%} of the frame")
            elif ma is not None and ma > 0.02 and not facts[k]["head_in"]:
                why.append(f"{k}, turned away, is a body without its head in the frame")
    for q3 in opts.get("points") or []:      # a move's start, path and end place: all inside the one frame
        q = framing.project(cam, q3, res)
        if not q or not (0.03 < q[0] < 0.97 and 0.03 < q[1] < 0.97):
            why.append(f"the move's place at {[round(v, 2) for v in q3]} is out of the frame")
    mv = opts.get("mover")
    if mv and mv in subjects:                # the mover's face readable at the move's start
        mm = subjects[mv]["measured"]
        if plan_angle(mm["head_fwd_xy"], mm["eye_xy"], cam["position"]) > 80 or not facts[mv]["head_in"]:
            why.append(f"{mv}'s face (the one who moves) is not readable")
    for k, jn in opts.get("joints") or []:
        j = ((subjects.get(k) or {}).get("measured") or {}).get("joints") or {}
        q = framing.project(cam, j[jn], res) if jn in j else None
        if not q or not (0.05 < q[0] < 0.95 and 0.05 < q[1] < 0.95):
            why.append(f"{k}'s {jn} (the action) is not in the frame")
        elif shot is not None and not joint_seen(shot, k, q):
            why.append(f"{k}'s {jn} (the action) is hidden")
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
            elif not ots and (not f["head_in"] or (ma is not None and ma < LISTENER_AREA["two"])):
                why.append("the listener is not in view")
    for k in show or []:
        if k in facts and not facts[k]["head_in"]:
            why.append(f"{k} (named for the shot) is not in view")
    if opts.get("_single"):                  # a clean single: everyone else clearly in or clearly out
        for k, v in subjects.items():
            if k == speaker:
                continue
            pr = [framing.project(cam, q, res) for q in body_points(v["measured"]).values()]
            xs = [q[0] for q in pr if q and q[1] < 1.0]
            if xs and not (min(xs) > 1.10 or max(xs) < -0.10 or (min(xs) > 0.05 and max(xs) < 0.95)):
                why.append(f"{k} near the edge of a single")
    for k, f in facts.items():
        # a face cut by the frame's edge is always a miss (any shot kind): the run rejects it if no camera clears it
        em = subjects[k]["measured"]
        ep = framing.project(cam, [em["eye_xy"][0], em["eye_xy"][1], em["eye_z"]], res)
        near_edge = bool(ep) and -0.08 < ep[0] < 1.08 and -0.05 < ep[1] < 1.0      # the head at or just past an edge
        if f["cut"] and (f["head_any"] or near_edge) and not turned_away(cam, em) and k != speaker:
            why.append(f"PARTIAL FACE: {k}'s face cut by the frame's {'/'.join(f['cut_edges'])} edge")
        elif f["cut"] and not (ots and k == listener) and kind != "close":
            why.append(f"{k} cut by the frame's {'/'.join(f['cut_edges'])} edge")
    return why, facts


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
        if tgt_k is None or k == tgt_k or tgt_k not in ch or c.get("asset") or c.get("look") is False:
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
        ch[k] = dict(c, body_pose=bp, _look={"at": tgt_k, "yaw": round(yaw + chest, 1), "pitch": round(pitch, 1)})
    return spec


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
    return [[a[0], a[1], 1.65], [a[0], a[1], 0.05], [b[0], b[1], t["center"][2] + t["half"][2]],
            [b[0], b[1], 0.05], [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.8]]


def cam_opts(spec, c):
    """The framing rules' per-camera facts: who sits (no shots over a seated listener's chair back), the camera's
    subject, the action's joints that must be seen ("show_joints": [[key, joint], ...]) and a move's places."""
    seated = {k for k, v in spec["characters"].items()
              if any(w in ((v.get("body_pose") or {}).get("pose") or "") for w in SEATED)}
    return {"seated": seated, "subject": c.get("subject"), "joints": c.get("show_joints") or [],
            "points": c.get("_points") or [], "mover": c.get("mover")}


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
                    for f in ((20, 24, 28) if opts.get("points") else (28, 35, 40, 50)):
                        q = {"position": [round(v, 3) for v in p], "target": [round(v, 3) for v in aim], "focal_mm": f}
                        why, _ = framing_why(q, kind, spk, lis, mode == "ots", c.get("show"), subjects, res,
                                             opts=opts)
                        if not why:
                            out.append((math.dist(p, cam0["position"]) + (0.0 if mode == ("ots" if c.get("ots") else "two") else 0.5), q, mode))
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
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--no-split", action="store_true", help="stage all cameras in one scene (no per-speaker looks)")
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
        for spk, cams in groups.items():
            sub = dict(spec, id=f"{spec['id']}_{spk or 'q'}", cameras=cams)
            sp_path = odir.parent / f"{sub['id']}.spec.json"
            sp_path.write_text(json.dumps(sub, indent=1))
            log(f"group {sub['id']}: {[c['name'] for c in cams]}")
            rc |= subprocess.run([sys.executable, __file__, str(sp_path), "--rounds", str(a.rounds), "--no-split"]).returncode
        return rc
    if len(groups) == 1:
        look_at(spec, list(groups)[0])
    seat_plan = seating.plan(spec, Path(C.LIBRARY))           # seated people placed from their chairs' seat boxes
    for c in spec["cameras"]:
        c["_points"] = move_points(spec, c)
    keys = spec.get("line") or list(spec["characters"])[:2]
    line = [spec["characters"][k]["position"][:2] for k in keys] if len(keys) == 2 else None
    est_side = None
    for c in spec["cameras"]:                # the first wide (or the first camera) sets the side of the line
        if line and c.get("position") and (c["kind"] in ("wide", "full") or est_side is None):
            est_side = side(line, c["position"]) or est_side
            if c["kind"] in ("wide", "full"):
                break
    # first pass: over-the-shoulder cameras from the spec's marks (eye height assumed until measured)
    for c in spec["cameras"]:
        if c.get("ots") and c.get("speaker") and not c.get("position"):
            ch = spec["characters"]
            guess = lambda k: {"eye_xy": ch[k]["position"][:2], "eye_z": 1.6}
            c.update(ots_camera(guess(c["speaker"]), guess(c["ots"]), est_side, line))
            c["kind"] = c.get("kind", "medium")
    report = {"scene": spec["id"], "line": keys, "established_side": est_side, "rounds": [], "shots": {},
              "seating": {}}
    todo = [c["name"] for c in spec["cameras"]]
    tried = {c["name"]: [] for c in spec["cameras"]}
    master = None
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
            sh["speaker_check"] = chk
            report["shots"].setdefault(c["name"], {"history": []})
            report["shots"][c["name"]]["history"].append(dict(chk, round=rnd, camera=sh["camera"]))
            report["shots"][c["name"]].update(speaker=c.get("speaker"), ok=chk["ok"], rounds=rnd)
            if not chk["ok"]:
                fails.append(c)
        report["rounds"].append({"round": rnd, "rendered": todo, "failed": [c["name"] for c in fails]})
        log(f"round {rnd}: rendered {len(todo)}, failed {[c['name'] + ': ' + '; '.join(shots[c['name']]['speaker_check']['why']) for c in fails]}")
        if not fails or rnd == a.rounds:
            for c in fails:                  # only a speaker's unseen face rejects; a framing miss is reported
                chk = shots[c["name"]]["speaker_check"]
                face = [w for w in chk["why"] if w not in chk.get("framing", [])]
                partial = [w for w in chk.get("framing", []) if w.startswith("PARTIAL FACE")]
                if face:
                    shots[c["name"]]["reject"] = "speaker's face not shown: " + "; ".join(face)
                elif partial:
                    shots[c["name"]]["reject"] = "; ".join(partial)
                else:
                    shots[c["name"]]["framing_miss"] = chk["framing"]
            break
        todo = []
        for c in fails:
            chk = shots[c["name"]]["speaker_check"]
            face_fail = [w for w in chk["why"] if w not in chk.get("framing", [])]
            lis = listener_of(c, keys, c.get("speaker"))
            cands = []
            if (face_fail or c.get("ots")) and c.get("speaker"):
                cands = candidates(c, shots[c["name"]], c["speaker"], subjects, res, room, line, est_side,
                                   tried[c["name"]])
                good = [q for q in cands if not framing_why(q, c["kind"], c["speaker"], lis, bool(c.get("ots")),
                                                            c.get("show"), subjects, res, opts=cam_opts(spec, c))[0]]
                cands = good or (cands if face_fail else [])
            if not cands and chk.get("framing"):
                partial = any(w.startswith("PARTIAL FACE") for w in chk["framing"])
                if partial and c.get("speaker"):  # nobody else can be fully in or out: a clean single at once
                    tc = two_shot_candidates(dict(c, show=None), shots[c["name"]], c["speaker"], None, subjects, res,
                                             room, line, est_side, tried[c["name"]], dict(cam_opts(spec, c), _single=True))
                else:
                    tc = two_shot_candidates(c, shots[c["name"]], c.get("speaker"), lis, subjects, res, room, line,
                                             est_side, tried[c["name"]], cam_opts(spec, c))
                if tc:
                    if tc[0][1] == "single":     # a clean single: the listener is no longer required in it
                        c["listener"] = None
                        c["single"] = True
                    if tc[0][1] == "ots":        # framed again over the listener's shoulder, or as a two-shot / single
                        c["ots"] = lis
                    else:
                        c.pop("ots", None)
                    cands = [tc[0][0]]
            if not cands:
                if face_fail:
                    shots[c["name"]]["reject"] = ("no camera left that shows the speaker's face: " + "; ".join(face_fail))
                else:
                    shots[c["name"]]["framing_miss"] = chk.get("framing")
                continue
            q = cands[0]
            c.update(position=q["position"], target=q["target"], focal_mm=q["focal_mm"])
            for k in ("az", "el", "screen_x"):
                c.pop(k, None)
            todo.append(c["name"])
        if not todo:
            break
    (odir / "result.json").write_text(json.dumps(master, indent=1))
    report["seconds"] = round(time.time() - t0, 1)
    (odir / "speakers.json").write_text(json.dumps(report, indent=1))
    for n, s in report["shots"].items():
        last = (s.get("history") or [{}])[-1]
        log(f"  {n}: speaker {s.get('speaker')} ok={s['ok']} rounds={s['rounds']} head={last.get('head_deg')} "
            f"face={last.get('face_visible')} size={last.get('face_h')} {last.get('why') or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
