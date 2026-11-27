"""The camera search's first sieve, all its cameras at once (11-27).

speakers.two_shot_candidates tries some tens of thousands of cameras for one shot, each through framing_why: a few
hundred points projected one at a time. Nearly all of them fail one of a few plain rules of the frame, and those
three can be worked out for every camera together:

  - someone the shot's plan does not list is in the picture,
  - someone's head is bigger than the head of the one the shot is about,
  - someone the shot names is not in the picture,
  - in a full or wide shot of several named people, one of them is behind another, or it is from behind its lead.

sure_fail() says, for each camera, whether one of them SURELY fails it: by more than a margin far wider than any
difference between this arithmetic and framing_why's own. A camera within the margin of any threshold, or with a
point at the camera's own plane, is never judged here. The cameras not refused go through framing_why as before, in
the same order: the search returns the same cameras."""
from __future__ import annotations

import math

import numpy as np

import framing

EPS = 1e-7                   # the margin: a rule fails surely only when it fails by more than this
Z_MIN = 1e-6                 # framing.project's own: a point no further than this before the camera has no place


def project_many(pos, tgt, foc, pts, res):
    """framing.project for N cameras and M points at once -> (u, v, z) each (N, M), and per camera whether it has a
    view at all (not on its own aim point)."""
    d0 = tgt - pos
    has = (np.abs(d0) > 1e-9).any(1)
    n = np.linalg.norm(d0, axis=1)
    f = d0 / np.where(n > 0, n, 1.0)[:, None]
    steep = np.abs(f[:, 2]) >= 0.999                       # straight up or down: another up
    zero = np.zeros(len(f))
    r = np.where(steep[:, None], np.stack([-f[:, 2], zero, f[:, 0]], 1), np.stack([f[:, 1], -f[:, 0], zero], 1))
    rn = np.linalg.norm(r, axis=1)
    r = r / np.where(rn > 0, rn, 1.0)[:, None]
    u = np.cross(r, f)
    v = pts[None, :, :] - pos[:, None, :]
    z = np.einsum("nmk,nk->nm", v, f)
    x = np.einsum("nmk,nk->nm", v, r)
    y = np.einsum("nmk,nk->nm", v, u)
    zs = np.where(np.abs(z) > 1e-12, z, 1e-12)
    sw, sh = framing.SENSOR_W, framing.sensor_h(res)
    return 0.5 + foc[:, None] * x / zs / sw, 0.5 - foc[:, None] * y / zs / sh, z, has


def _head_points(m):
    """A measured person's head points as framing_why's measures take them: top, eyes, chin, the head's two sides
    (frame_facts: 10 cm out) and the two a head's share is taken between (head_share: 9 cm out)."""
    e = m["eye_xy"]
    hf = m.get("head_fwd_xy", [0, -1])
    n = math.hypot(hf[0], hf[1]) or 1.0
    f = [hf[0] / n, hf[1] / n]
    return [[e[0], e[1], m["head_top_z"]], [e[0], e[1], m["eye_z"]], [e[0], e[1], m["chin_z"]],
            [e[0] - f[1] * 0.1, e[1] + f[0] * 0.1, m["eye_z"]], [e[0] + f[1] * 0.1, e[1] - f[0] * 0.1, m["eye_z"]],
            [e[0] - f[1] * 0.09, e[1] + f[0] * 0.09, m["eye_z"]], [e[0] + f[1] * 0.09, e[1] - f[0] * 0.09, m["eye_z"]]]


def sure_fail(pos, tgt, foc, kind, speaker, listener, show, subjects, res, opts, limits) -> np.ndarray:
    """Per camera (rows of pos, tgt, foc): True where framing_why would surely refuse it (see the module's words).
    limits: speakers' own numbers {"EDGE", "SIDE_EDGE", "SLIVER", "FULL_LEAD_K", "BEHIND_M"}."""
    opts = opts or {}
    pos, tgt, foc = np.asarray(pos, float), np.asarray(tgt, float), np.asarray(foc, float)
    N = len(pos)
    keys = list(subjects)
    if not N or not keys:
        return np.zeros(N, bool)
    pts = np.array([q for k in keys for q in _head_points(subjects[k]["measured"])], float)
    u, v, z, has = project_many(pos, tgt, foc, pts, res)
    shaky = ~has | (np.abs(z - Z_MIN) < Z_MIN).any(1)       # (a point at the camera's own plane: not judged here)
    ok = z > Z_MIN
    E, SE, SL = limits["EDGE"], limits["SIDE_EDGE"], limits["SLIVER"]

    def inside(m):
        return ok & (u > SE + m) & (u < 1 - SE - m) & (v > E + m) & (v < 1 - E - m)
    in_sure, in_maybe = inside(EPS), inside(-EPS)
    fail = np.zeros(N, bool)
    head_sure, head_not, share, hh = {}, {}, {}, {}
    for i, k in enumerate(keys):
        c = slice(7 * i, 7 * i + 5)
        head_sure[k] = in_sure[:, c].any(1)                  # (some point of the head surely in the frame)
        head_not[k] = ~in_maybe[:, c].any(1)                 # (surely none of them)
        top, eye, chin, sl, sr = (7 * i + j for j in (0, 1, 2, 5, 6))
        x0, x1 = np.minimum(u[:, sl], u[:, sr]), np.maximum(u[:, sl], u[:, sr])
        w = np.maximum(x1 - x0, 1e-4)
        s = np.maximum(0.0, np.minimum(1.0, x1) - np.maximum(0.0, x0)) / w
        dv = v[:, chin] - v[:, top]
        tall = ok[:, top] & ok[:, chin] & (dv > 0)
        s = np.where(tall, np.minimum(s, np.maximum(0.0, np.minimum(1.0, v[:, chin]) - np.maximum(0.0, v[:, top]))
                                      / np.where(tall, dv, 1.0)), s)
        share[k] = np.where(ok[:, sl] & ok[:, sr] & ok[:, eye], s, 0.0)
        shaky |= ok[:, top] & ok[:, chin] & (np.abs(dv) < 1e-9)
        hh[k] = np.where(ok[:, top] & ok[:, chin], dv, 0.0)
    # someone the plan does not list for the shot is in the picture
    listed = opts.get("listed")
    for k in (keys if listed else ()):
        if k in listed or k in (speaker, listener):
            continue
        fail |= head_sure[k] & (share[k] > SL + EPS)
    # someone the director names is not in the picture
    for k in show or []:
        if k in subjects:
            fail |= head_not[k] | (share[k] < SL - EPS)
    # a head bigger than the head of the one the shot is about
    lead = speaker or (opts.get("line_subject") if opts.get("line_subject") in subjects else None)
    if lead and not subjects[lead]["measured"].get("creature"):
        for k in keys:
            if k == lead or subjects[k]["measured"].get("creature"):
                continue
            group = not speaker and kind in ("full", "wide") and lead in (show or ()) and k in (show or ())
            fail |= head_sure[k] & (hh[k] > hh[lead] * (limits["FULL_LEAD_K"] if group else 1.02) + EPS)
    # a full or wide shot of several named people: none of them behind another, and not from behind its lead
    named = [k for k in show or [] if k in subjects]
    if kind in ("full", "wide") and len(named) > 1:
        seen = [k for k in named if not subjects[k]["measured"].get("creature")]
        cxy = pos[:, :2]
        exy = {k: np.array(subjects[k]["measured"]["eye_xy"][:2], float) for k in seen}
        dist = {k: np.linalg.norm(cxy - exy[k], axis=1) for k in seen}
        for k in seen:
            ab = exy[k][None, :] - cxy
            l2 = (ab * ab).sum(1)
            l2 = np.where(l2 > 0, l2, 1e-12)
            for o in seen:
                if o == k:
                    continue
                t = np.clip(((exy[o][None, :] - cxy) * ab).sum(1) / l2, 0.0, 1.0)
                off = np.linalg.norm(exy[o][None, :] - (cxy + ab * t[:, None]), axis=1)
                fail |= head_sure[k] & head_sure[o] & (dist[o] < dist[k] - 0.3 - EPS) & (off < limits["BEHIND_M"] - EPS)
        if lead and not speaker and lead in seen:
            m = subjects[lead]["measured"]
            d = cxy - exy[lead][None, :]
            a = np.degrees(np.arctan2(d[:, 1], d[:, 0]) - math.atan2(m["head_fwd_xy"][1], m["head_fwd_xy"][0]))
            fail |= head_sure[lead] & (np.abs((a + 180) % 360 - 180) > 100 + 1e-6)
    return fail & ~shaky
