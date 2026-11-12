"""Checks on a rendered take that the vision judge does not make (11-12): the camera or room drifting away from the
opening picture, a jump where two pieces of a long line meet, and the people changing over the clip (an extra head,
a head gone, a seated person rising, a head becoming another's). Mechanics only: numpy on small frames; no model.

  python tools/take_checks.py CLIPS_DIR [--out table.json]      # the table for a folder of clips

A fault is a plain sentence, as the judge's are, so a faulted take is taken again the same way (shot_flow._next_take).
"""
from __future__ import annotations

import json
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---- thresholds (fractions of the frame unless said; env overrides for an evaluation run) -------------------------
def _f(name: str, default: float) -> float:
    return float(os.environ.get(f"TAKE_CHECKS_{name}", default))


SAMPLE_FPS = _f("FPS", 4.0)             # frames looked at per second of clip
SMALL_W = 192                           # the luma frame's width (height follows the clip's shape, made even)
TILES = (4, 6)                          # rows, columns of tiles whose own shifts vote on the camera's move
PAN_MAX = _f("PAN", 0.05)               # the background moved this share of the frame's width or more = a pan
ZOOM_MAX = _f("ZOOM", 0.09)             # the background scaled by this much or more = a reframe (11-12, one 30-shot
#                                         cut: slow push-ins of 6-8% over a clip went unremarked by the reviewer; the
#                                         reframes that were faults read 10% and up)
MATCH_MIN = _f("MATCH", 0.3)            # under this share of the textured tiles still line up with the key = lost
HOLD_N = int(_f("HOLD_N", 2))           # a fault must hold this many sampled frames in a row (not one odd frame)
DIFF_T = _f("DIFF", 0.12)               # a pixel "changed" when its smoothed luma moved this far (0-1)
STILL_T = _f("STILL", 0.035)            # ... and "came to rest" when it moves less than this between late samples
APPEAR_MAX = _f("APPEAR", 0.06)         # a still shot: this share of the frame changed and came to rest = a new thing
SEAM_PAN = _f("SEAM_PAN", 0.02)         # at a seam between pieces: a shift or a scale above these = a framing jump
SEAM_ZOOM = _f("SEAM_ZOOM", 0.03)
SEAM_OVER = _f("SEAM_OVER", 4.0)        # ... and at least this many times the clip's ordinary frame-to-frame move
HEAD_MIN = _f("HEAD_MIN", 0.5)          # a head counts when at least this share of the key's smallest head's height
EXTRA_S = _f("EXTRA_S", 0.4)            # an extra head for this long = an extra person
GONE_S = _f("GONE_S", 0.75)             # a key person's head missing this long (to the clip's end or not) = gone
RISE_HEADS = _f("RISE", 1.2)            # a seated person's head up by this many head heights = they stood
LOOK_MAX = _f("LOOK", 0.45)             # a head's colour histogram this far (0-1) from its own in the key = another


# ---- frames --------------------------------------------------------------------------------------------------------
def frames(mp4: Path, fps: float = SAMPLE_FPS, width: int = SMALL_W, rgb: bool = False) -> np.ndarray:
    """The clip's frames at `fps`, small, luma 0-1 (or RGB): [n, h, w(, 3)] float32. ffmpeg decodes; about 0.2 s of
    CPU per clip second."""
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", str(mp4)], capture_output=True, text=True).stdout.strip().split(",")
    w0, h0 = int(pr[0]), int(pr[1])
    h = max(2, int(round(width * h0 / w0 / 2)) * 2)
    ch = 3 if rgb else 1
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"fps={fps},scale={width}:{h}", "-pix_fmt",
                          "rgb24" if rgb else "gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    n = len(raw) // (width * h * ch)
    a = np.frombuffer(raw[:n * width * h * ch], np.uint8).astype(np.float32) / 255.0
    return a.reshape((n, h, width, 3) if rgb else (n, h, width))


def _blur(a: np.ndarray, k: int = 2) -> np.ndarray:
    """A small box blur (separable, edges held)."""
    for ax in (-2, -1):
        p = np.concatenate([np.repeat(np.take(a, [0], ax), k, ax), a, np.repeat(np.take(a, [-1], ax), k, ax)], ax)
        c = np.cumsum(p, ax, dtype=np.float64)
        z = np.zeros_like(np.take(c, [0], ax))
        c = np.concatenate([z, c], ax)
        n = a.shape[ax]
        a = ((np.take(c, range(2 * k + 1, 2 * k + 1 + n), ax) - np.take(c, range(0, n), ax)) / (2 * k + 1)).astype(np.float32)
    return a


# ---- the camera's move between two frames -------------------------------------------------------------------------
def _shift(a: np.ndarray, b: np.ndarray) -> tuple:
    """How far tile b sits from tile a (dy, dx in pixels, sub-pixel) and how sure (the peak over the mean)."""
    wy, wx = np.hanning(a.shape[0])[:, None], np.hanning(a.shape[1])[None, :]
    fa, fb = np.fft.rfft2((a - a.mean()) * wy * wx), np.fft.rfft2((b - b.mean()) * wy * wx)
    r = fa.conj() * fb
    r /= np.abs(r) + 1e-9
    c = np.fft.irfft2(r, a.shape)
    py, px = np.unravel_index(int(np.argmax(c)), c.shape)
    peak = float(c[py, px])

    def sub(v_m, v_0, v_p):                     # the parabola through the peak and its neighbours
        d = v_m - 2 * v_0 + v_p
        return 0.0 if abs(d) < 1e-9 else float(np.clip(0.5 * (v_m - v_p) / d, -0.5, 0.5))
    dy = py + sub(c[py - 1, px], c[py, px], c[(py + 1) % c.shape[0], px])
    dx = px + sub(c[py, px - 1], c[py, px], c[py, (px + 1) % c.shape[1]])
    dy = dy - c.shape[0] if dy > c.shape[0] / 2 else dy
    dx = dx - c.shape[1] if dx > c.shape[1] / 2 else dx
    return dy, dx, peak / (float(np.abs(c).mean()) + 1e-9)


TILE_SURE = 9.0                         # a tile's match counts above this peak-to-mean (measured 11-12 on rendered clips:
#                                         frames of one clip 8-44, median 14-17; frames of different clips 5-8)
SCALES = (0.7, 0.78, 0.86, 0.93, 1.08, 1.16, 1.25, 1.35, 1.45)


def _rescale(a: np.ndarray, s: float) -> np.ndarray:
    """The frame scaled by s about its middle (nearest pixel; the frame keeps its size)."""
    h, w = a.shape
    ys = np.clip(((np.arange(h) - h / 2) / s + h / 2).round().astype(int), 0, h - 1)
    xs = np.clip(((np.arange(w) - w / 2) / s + w / 2).round().astype(int), 0, w - 1)
    return a[ys][:, xs]


def _tile_fit(a: np.ndarray, b: np.ndarray, tiles: tuple) -> tuple:
    """Each tile's own shift from a to b, and the shift + scale most of them agree on. -> (t [dy, dx], s, agreeing
    tiles, textured tiles)."""
    h, w = a.shape
    th, tw = h // tiles[0], w // tiles[1]
    cs, ds = [], []
    tex = 0
    for i in range(tiles[0]):
        for j in range(tiles[1]):
            ta, tb = a[i * th:(i + 1) * th, j * tw:(j + 1) * tw], b[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
            if ta.std() < 0.02 or tb.std() < 0.02:              # a flat tile (a plain wall) says nothing
                continue
            tex += 1
            dy, dx, sure = _shift(ta, tb)
            if sure > TILE_SURE:
                cs.append(((i + 0.5) * th - h / 2, (j + 0.5) * tw - w / 2))
                ds.append((dy, dx))
    if len(ds) < 4:
        return np.zeros(2), 0.0, 0, tex
    c, d = np.array(cs), np.array(ds)
    keep = np.ones(len(d), bool)
    t, s = np.median(d, 0), 0.0
    for _ in range(4):                          # shift = t + s * centre, on the tiles that agree (the rest are people)
        cc, dd = c[keep], d[keep]
        A = np.zeros((2 * len(cc), 3))
        A[0::2, 0], A[0::2, 2] = 1, cc[:, 0]
        A[1::2, 1], A[1::2, 2] = 1, cc[:, 1]
        sol = np.linalg.lstsq(A, dd.reshape(-1), rcond=None)[0]
        t, s = sol[:2], float(sol[2])
        err = np.linalg.norm(d - (t + s * c), axis=1)
        new = err <= max(1.0, min(3.0, float(np.median(err[keep])) * 2.5))
        if new.sum() < 4 or (new == keep).all():
            keep = new if new.sum() >= 4 else keep
            break
        keep = new
    err = np.linalg.norm(d - (t + s * c), axis=1)
    return t, s, int((err <= 3.0).sum()), tex


def camera_move(a: np.ndarray, b: np.ndarray, tiles: tuple = TILES, scales: tuple = SCALES) -> dict:
    """The background's move from frame a to frame b: each tile's own shift (phase correlation), then the shift and
    scale that most tiles agree on (people moving are a few tiles; the room is the rest). A pan wider than a tile or
    a reframe too large for tiles is found on the whole frame first (b slid, or tried at a few scales), and that
    reading is kept only when more tiles then agree than without it (a large person crossing the frame can win the
    whole-frame match; the tiles cannot be fooled the same way). -> pan (share of the width), zoom (scale - 1), dx,
    dy (pixels), n (tiles that agree), match (their share of the textured tiles)."""
    h, w = a.shape
    cands = [(1.0, 0, 0)]
    gy, gx, gsure = _shift(a, b)
    if gsure > TILE_SURE and (abs(gy) >= 2 or abs(gx) >= 2):
        cands.append((1.0, int(round(gy)), int(round(gx))))
    best = (gsure, 1.0, gy, gx)
    for sc in scales or ():
        y_, x_, su = _shift(a, _rescale(b, 1.0 / sc))
        if su > best[0]:
            best = (su, sc, y_, x_)
    if best[1] != 1.0 and best[0] >= 1.3 * gsure:
        cands.append((best[1], int(round(best[2])), int(round(best[3]))))
    out = None
    for s0, py, px in cands:
        bb = _rescale(b, 1.0 / s0) if s0 != 1.0 else b
        if py or px:
            bb = np.roll(bb, (-py, -px), (0, 1))
        t, s, n, tex = _tile_fit(a, bb, tiles)
        if out is None or n > out["n"]:
            t = t + np.array([py, px])
            out = {"pan": float(np.hypot(*t)) / w, "zoom": (1.0 + s) * s0 - 1.0, "dx": float(t[1]), "dy": float(t[0]),
                   "n": n, "match": n / max(tex, 1)}
    return out


def _run(flags: list, n: int) -> int | None:
    """The first index where `flags` is true `n` times in a row."""
    k = 0
    for i, f in enumerate(flags):
        k = k + 1 if f else 0
        if k >= n:
            return i - n + 1
    return None


def _boxes_mask(shape: tuple, boxes: list, grow: float = 0.35) -> np.ndarray:
    """True outside the people's boxes ((x, y, w, h) as shares of the frame), each grown by `grow` of its size."""
    m = np.ones(shape, bool)
    h, w = shape
    for x, y, bw, bh in boxes or []:
        x0, x1 = int((x - bw * grow) * w), int((x + bw * (1 + grow)) * w) + 1
        y0, y1 = int((y - bh * grow) * h), int((y + bh * (1 + grow)) * h) + 1
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = False
    return m


def drift(fr: np.ndarray, fps: float = SAMPLE_FPS, still: bool = False, people: list | None = None) -> dict:
    """The clip's frames [n, h, w] against its first (the opening picture): the largest pan and zoom that hold, and
    the share of the frame that changed and came to rest (a new thing in the room, or the room redrawn). `still`: a
    shot in which nobody is meant to move about (talk, quiet), where a change that rests is a fault; in an action
    shot it is reported only (the action itself changes the picture). `people`: per frame, the people's boxes, left
    out of the changed area. -> {"pan", "zoom", "appeared", "at", "faults": [...]}"""
    n = len(fr)
    if n < 3:
        return {"pan": 0.0, "zoom": 0.0, "appeared": 0.0, "faults": []}
    key = fr[0]
    mv = [camera_move(key, fr[i]) for i in range(1, n)]
    pans, zooms = [m["pan"] for m in mv], [abs(m["zoom"]) for m in mv]
    faults = []
    i = _run([p >= PAN_MAX for p in pans], HOLD_N)
    if i is not None:
        faults.append(f"the camera moves from {(i + 1) / fps:.1f} s (the background shifts "
                      f"{max(pans) * 100:.0f}% of the frame)")
    j = _run([z >= ZOOM_MAX for z in zooms], HOLD_N)
    if j is not None:
        faults.append(f"the camera reframes from {(j + 1) / fps:.1f} s (the background scales "
                      f"{max(zooms) * 100:.0f}%)")
    k = _run([m.get("match", 1.0) < MATCH_MIN for m in mv], HOLD_N + 1)
    if k is not None and i is None and j is None and still:
        faults.append(f"the picture stops lining up with the opening picture from {(k + 1) / fps:.1f} s "
                      f"(under {MATCH_MIN * 100:.0f}% of it still matches)")
    # what changed and came to rest: the late frames against the key, where the late frames agree with each other
    kb = _blur(key)
    late = [_blur(fr[i]) for i in range(max(1, n - max(2, n // 4)), n)]
    moved = np.minimum.reduce([np.abs(x - kb) for x in late]) >= DIFF_T
    rest = np.maximum.reduce([np.abs(late[i] - late[i - 1]) for i in range(1, len(late))] or
                             [np.zeros_like(kb)]) <= STILL_T
    m = moved & rest
    if people:
        m &= _boxes_mask(m.shape, [b for f_ in people[-len(late):] for b in f_]) & _boxes_mask(m.shape, people[0])
    appeared = float(m.mean())
    if still and appeared >= APPEAR_MAX and i is None and j is None:
        faults.append(f"the room changes: {appeared * 100:.0f}% of the picture is different at the end and stays so")
    return {"pan": max(pans), "zoom": max(zooms), "appeared": appeared, "moves": mv, "faults": faults,
            "match": min(m.get("match", 1.0) for m in mv)}


def seam_jump(before: np.ndarray, after: np.ndarray, usual: float | None = None) -> dict:
    """The framing's jump where one piece ends and the next begins (its last frame, the next's first): the
    background's shift and scale across the cut. `usual`: the clip's ordinary frame-to-frame pan (the jump must also
    stand well above it, so a hand-held feel is not a jump). -> {"pan", "zoom", "fault": str | None}"""
    m = camera_move(before, after)
    over = usual is None or m["pan"] >= SEAM_OVER * max(usual, 1e-4) or abs(m["zoom"]) >= SEAM_ZOOM
    bad = (m["pan"] >= SEAM_PAN or abs(m["zoom"]) >= SEAM_ZOOM) and over
    return {"pan": m["pan"], "zoom": m["zoom"],
            "fault": (f"the framing jumps between pieces (shift {m['pan'] * 100:.1f}% of the frame, scale "
                      f"{m['zoom'] * 100:+.1f}%)") if bad else None}


SEAM_AFTER = 25                         # frames after a seam within which the picture is compared (one second)
SEAM_PAN_S = _f("SEAM_PAN_S", 0.05)     # over that second: the background shifted or scaled this much ...
SEAM_ZOOM_S = _f("SEAM_ZOOM_S", 0.05)
SEAM_RATIO = _f("SEAM_RATIO", 3.0)      # ... and at least this many times what it did in the second before the seam


def seam_window(lu: np.ndarray, f: int, rgb: np.ndarray | None = None) -> dict:
    """A seam at frame f of a clip read at the frame rate (lu [n, h, w]; f = the later piece's first frame). A later
    piece STARTS on the frame before it, so the two frames at the seam always agree; what goes wrong goes wrong in
    the first second after it (the new piece leaves its opening frame at once). So: the picture one second after the
    seam against the frame before it, and against how much the picture moved in the second before the seam.
    -> {"frame", "pan", "zoom", "before", "fault", "more"}"""
    n = len(lu)
    a0, a1 = max(0, f - 1 - SEAM_AFTER), f - 1
    b1 = min(n - 1, f + SEAM_AFTER)
    b05 = min(n - 1, f + SEAM_AFTER // 2)
    if a1 <= a0 or b1 <= f:
        return {"frame": f, "pan": 0.0, "zoom": 0.0, "before": 0.0, "fault": None, "more": []}
    bef = camera_move(lu[a0], lu[a1])
    afts = [camera_move(lu[a1], lu[b05]), camera_move(lu[a1], lu[b1])]
    aft = max(afts, key=lambda m: max(m["pan"] / SEAM_PAN_S, abs(m["zoom"]) / SEAM_ZOOM_S))
    usual = max(bef["pan"] / SEAM_PAN_S, abs(bef["zoom"]) / SEAM_ZOOM_S)
    size = max(aft["pan"] / SEAM_PAN_S, abs(aft["zoom"]) / SEAM_ZOOM_S)
    bad = size >= 1.0 and size >= SEAM_RATIO * usual
    more = seam_people(lu[a1], lu[b1], rgb_b=None if rgb is None else rgb[a1], rgb_a=None if rgb is None else rgb[b1])
    return {"frame": f, "pan": aft["pan"], "zoom": aft["zoom"], "before": bef["pan"],
            "fault": (f"the framing changes within a second of a seam at {f / 25.0:.1f} s (shift "
                      f"{aft['pan'] * 100:.0f}% of the frame, scale {aft['zoom'] * 100:+.0f}%)") if bad else None,
            "more": [f"{x} (the seam at {f / 25.0:.1f} s)" for x in more]}


def seams(pieces: list) -> list:
    """Each seam of a line rendered in pieces (their clips in order), read at the frame rate (seam_window)."""
    out = []
    ends = [frames(Path(p), fps=25.0, rgb=True) for p in pieces]
    t = 0
    for a, b in zip(ends, ends[1:]):
        if len(a) >= 3 and len(b) >= 3:
            both = np.concatenate([a[-(SEAM_AFTER + 1):], b[1:SEAM_AFTER + 2]])   # (b's first frame is a's last)
            s = seam_window(both.mean(-1), min(len(a), SEAM_AFTER + 1), both)
            s["frame"] = t + len(a)
            out.append(s)
        t += len(a) - 1
    return out


def seams_in(mp4: Path, at: list) -> list:
    """The seams inside a joined clip at frame numbers `at` (the first frame of each later piece)."""
    fr = frames(Path(mp4), fps=25.0, rgb=True)
    lu = fr.mean(-1)
    return [seam_window(lu, f, fr) for f in at if 3 <= f < len(fr) - 2]


# ---- the people over the clip --------------------------------------------------------------------------------------
def _centre(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def people_over_time(heads: list, fps: float, key: list | None = None, looks: list | None = None) -> list:
    """Faults from the heads found in each sampled frame. heads: per frame, [(x, y, w, h), ...] as shares of the
    frame. key: the opening picture's people, each {"box": (x, y, w, h), "seated": bool, "stays": bool} ("stays":
    the shot does not have them stand or leave); default: the first frame's heads, nobody seated. looks: per frame,
    per head, a colour histogram (head_look) to tell one head from another's. Each person is followed by nearness
    from frame to frame; a head no person owns is an extra once it is big enough and lasts."""
    if not heads:
        return []
    key = key if key is not None else [{"box": b, "seated": False, "stays": False} for b in heads[0]]
    if not key:
        return [f"{len(heads[-1])} people at the end, none in the opening picture"] if heads[-1] else []
    hmin = HEAD_MIN * min(p["box"][3] for p in key)
    last = [p["box"] for p in key]
    look0 = [None] * len(key)
    seen, extra, rose, other = [[] for _ in key], [], [[] for _ in key], [[] for _ in key]
    for fi, hs in enumerate(heads):
        free = list(range(len(hs)))
        got = {}
        order = sorted(((np.hypot(_centre(last[pi])[0] - _centre(hs[hi])[0],
                                  _centre(last[pi])[1] - _centre(hs[hi])[1]), pi, hi)
                        for pi in range(len(key)) for hi in free), key=lambda x: x[0])
        for d, pi, hi in order:                             # nearest pairs first; a head too far is not that person
            if pi in got or hi not in free or d > 2.5 * max(last[pi][3], hs[hi][3]):
                continue
            got[pi] = hi
            free.remove(hi)
        for pi, p in enumerate(key):
            hi = got.get(pi)
            seen[pi].append(hi is not None)
            if hi is None:
                rose[pi].append(False)
                other[pi].append(False)
                continue
            last[pi] = hs[hi]
            up = (_centre(p["box"])[1] - _centre(hs[hi])[1]) / max(p["box"][3], 1e-6)
            rose[pi].append(bool(p.get("seated") and p.get("stays", True) and up >= RISE_HEADS))
            lk = looks[fi][hi] if looks and fi < len(looks) and hi < len(looks[fi]) else None
            if lk is not None and look0[pi] is None:
                look0[pi] = lk
            other[pi].append(lk is not None and look0[pi] is not None and look_distance(look0[pi], lk) >= LOOK_MAX)
        extra.append(sum(1 for hi in free if hs[hi][3] >= hmin))
    faults = []
    n_extra, n_gone = max(2, int(round(EXTRA_S * fps))), max(2, int(round(GONE_S * fps)))
    i = _run([e > 0 for e in extra], n_extra)
    if i is not None:
        faults.append(f"{len(key) + max(extra[i:i + n_extra])} people, not {len(key)}, from {i / fps:.1f} s")
    for pi, p in enumerate(key):
        g = _run([not s for s in seen[pi]], n_gone)
        if g is not None and not p.get("leaves"):
            faults.append(f"person {pi + 1} of {len(key)} is gone from {g / fps:.1f} s")
        r = _run(rose[pi], n_extra)
        if r is not None:
            faults.append(f"person {pi + 1} of {len(key)} stands up from {r / fps:.1f} s (the shot keeps them seated)")
        o = _run(other[pi], n_extra)
        if o is not None:
            faults.append(f"person {pi + 1} of {len(key)} looks like someone else from {o / fps:.1f} s")
    return faults


def head_look(rgb: np.ndarray, box: tuple, bins: int = 6) -> np.ndarray:
    """A head's colours (hair, skin) as a small histogram: the box on an RGB frame [h, w, 3] 0-1."""
    h, w = rgb.shape[:2]
    x0, y0 = int(box[0] * w), int(box[1] * h)
    x1, y1 = max(x0 + 1, int((box[0] + box[2]) * w)), max(y0 + 1, int((box[1] + box[3]) * h))
    px = rgb[max(0, y0):y1, max(0, x0):x1].reshape(-1, 3)
    if not len(px):
        return np.zeros(bins ** 3, np.float32)
    q = np.minimum((px * bins).astype(int), bins - 1)
    hist = np.bincount(q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2], minlength=bins ** 3).astype(np.float32)
    return hist / hist.sum()


def look_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0 (the same colours) to 1 (none shared)."""
    return float(1.0 - np.minimum(a, b).sum())


def count_faults(counts: dict, people: int, fps_hint: float | None = None) -> list:
    """Faults from head COUNTS alone (the vision judge's answer per sampled time, {seconds: n}), for shots with no
    head boxes: more than `people` at two sampled times in a row (one odd answer is not a person), or fewer at the
    last two (someone gone by the end)."""
    ts = sorted(counts)
    ns = [counts[t] for t in ts]
    out = []
    i = _run([n is not None and n > people for n in ns], 2)
    if i is not None:
        out.append(f"{max(n for n in ns[i:i + 2])} people, not {people}, at {ts[i]} s")
    if len(ns) >= 2 and all(n is not None and n < people for n in ns[-2:]):
        out.append(f"{ns[-1]} people, not {people}, at the end (someone is gone)")
    return out


def count_times(dur: float, per_s: float = 2.0, lo: int = 4, hi: int = 12) -> tuple:
    """When to ask the head count: `per_s` a second across the whole clip (4 to 12 asks), the last near the end."""
    n = int(min(hi, max(lo, round(dur * per_s))))
    return tuple(round(dur * (0.08 + 0.87 * i / (n - 1)), 1) for i in range(n))


def heads_from_face_tracks(pckl: Path, width: int, height: int, n_frames: int, fps_in: float = 25.0,
                           fps: float = SAMPLE_FPS) -> list:
    """The lip-sync step's own face tracks (its tracks.pckl: each {"track": {"frame": [...], "bbox": [[x0, y0, x1,
    y1], ...]}}) as heads per sampled frame: a talk shot's people over time at no extra cost."""
    with open(pckl, "rb") as fh:
        tracks = pickle.load(fh)
    step = fps_in / fps
    out = [[] for _ in range(int(n_frames / step) + 1)]
    for t in tracks:
        tr = t.get("track", t)
        by = {int(f): b for f, b in zip(tr["frame"], tr["bbox"])}
        for i in range(len(out)):
            b = by.get(int(round(i * step)))
            if b is not None:
                out[i].append((float(b[0]) / width, float(b[1]) / height, float(b[2] - b[0]) / width,
                               float(b[3] - b[1]) / height))
    return out


# ---- the picture across a seam, the end state, the speaker's turn, a contact ---------------------------------------
SEAM_DIFF = _f("SEAM_DIFF", 0.22)       # across a seam (after lining the frames up) this share of the picture differs
POS_MAX = _f("POS", 1.0)                # a person's head this many head heights from where it was = moved at a seam
COAT_MAX = _f("COAT", 0.18)             # the body under a head lighter or darker by this much (0-1) = other clothes
LIT_MIN = _f("LIT_MIN", 0.003)          # a lit source: this share of the picture bright and warm
LIT_DROP = _f("LIT_DROP", 0.25)         # ... and one side of a seam with under this share of the other's = it went out
YAW_MAX = _f("YAW", 55.0)               # a speaker turned this many degrees from the camera shows no mouth to read
YAW_SHARE = _f("YAW_SHARE", 0.3)        # ... for this share of the voiced frames = a retake before lip sync is scored
CONTACT_GROW = _f("CONTACT_GROW", 0.5)  # the staged contact region grown by this share of its size (a coarse test)


def _aligned_diff(a: np.ndarray, b: np.ndarray, mv: dict) -> float:
    """The share of the picture that differs between two frames once b is slid by the camera's move."""
    dy, dx = int(round(mv["dy"])), int(round(mv["dx"]))
    h, w = a.shape
    ya, yb = slice(max(0, -dy), h - max(0, dy)), slice(max(0, dy), h - max(0, -dy))
    xa, xb = slice(max(0, -dx), w - max(0, dx)), slice(max(0, dx), w - max(0, -dx))
    if a[ya, xa].size < a.size // 4:
        return 1.0
    return float((np.abs(_blur(a)[ya, xa] - _blur(b)[yb, xb]) >= DIFF_T).mean())


def lit_share(rgb: np.ndarray) -> float:
    """The share of an RGB frame [h, w, 3] that is bright and warm (a fire, a lamp)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return float(((r >= 0.72) & (r - b >= 0.18) & (g >= 0.35)).mean())


def coat_light(rgb_or_luma: np.ndarray, box: tuple) -> float | None:
    """The mean lightness of the body under a head box: a strip one head wide, two head heights down from the chin."""
    a = rgb_or_luma.mean(-1) if rgb_or_luma.ndim == 3 else rgb_or_luma
    h, w = a.shape
    x0, x1 = int(box[0] * w), max(int(box[0] * w) + 1, int((box[0] + box[2]) * w))
    y0, y1 = int((box[1] + 1.3 * box[3]) * h), int((box[1] + 3.3 * box[3]) * h)
    px = a[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    return float(px.mean()) if px.size >= 4 else None


def seam_people(before: np.ndarray, after: np.ndarray, heads_b: list | None = None, heads_a: list | None = None,
                rgb_b: np.ndarray | None = None, rgb_a: np.ndarray | None = None) -> list:
    """What changes across a seam besides the framing (the last frame of one piece, the first of the next): the
    picture as a whole, the lit source, and with heads on both sides the people: their count, each one's place,
    their head's colours and the lightness of what they wear. -> faults."""
    out = []
    mv = camera_move(before, after)
    d = _aligned_diff(before, after, mv)
    if d >= SEAM_DIFF:
        out.append(f"the picture changes between pieces ({d * 100:.0f}% of it differs)")
    if rgb_b is not None and rgb_a is not None:
        lb, la = lit_share(rgb_b), lit_share(rgb_a)
        if max(lb, la) >= LIT_MIN and min(lb, la) < LIT_DROP * max(lb, la):
            out.append("a lit source " + ("goes out" if la < lb else "appears") + " between pieces")
    if heads_b is not None and heads_a is not None:
        if len(heads_b) != len(heads_a):
            out.append(f"{len(heads_a)} people after the seam, {len(heads_b)} before")
        used = set()
        for i, hb in enumerate(heads_b):
            cand = sorted(((np.hypot(_centre(hb)[0] - _centre(ha)[0], _centre(hb)[1] - _centre(ha)[1]), j)
                           for j, ha in enumerate(heads_a) if j not in used))
            if not cand:
                continue
            dist, j = cand[0]
            used.add(j)
            ha = heads_a[j]
            if dist >= POS_MAX * max(hb[3], ha[3]):
                out.append(f"person {i + 1} is somewhere else after the seam")
                continue
            if rgb_b is not None and rgb_a is not None:
                if look_distance(head_look(rgb_b, hb), head_look(rgb_a, ha)) >= LOOK_MAX:
                    out.append(f"person {i + 1} looks like someone else after the seam")
                cb, ca = coat_light(rgb_b, hb), coat_light(rgb_a, ha)
                if cb is not None and ca is not None and abs(cb - ca) >= COAT_MAX:
                    out.append(f"person {i + 1}'s clothes change lightness after the seam")
    return out


def posture_at_end(heads: list, fps: float, key: list, last_s: float = 0.75) -> list:
    """Each key person's posture over the clip's last `last_s` seconds, read from their head's height against the
    opening picture: "standing" / "sitting" / None (not found). A seated person whose head is RISE_HEADS head heights
    up has stood; a standing person whose head is that far down has sat. key: as people_over_time, each with
    "seated"."""
    n = max(2, int(round(last_s * fps)))
    last = [p["box"] for p in key]
    ups = [[] for _ in key]
    for fi, hs in enumerate(heads):
        free = list(range(len(hs)))
        for pi in range(len(key)):
            cand = sorted((np.hypot(_centre(last[pi])[0] - _centre(hs[hi])[0],
                                    _centre(last[pi])[1] - _centre(hs[hi])[1]), hi) for hi in free)
            if not cand or cand[0][0] > 2.5 * max(last[pi][3], hs[cand[0][1]][3]):
                continue
            hi = cand[0][1]
            free.remove(hi)
            last[pi] = hs[hi]
            if fi >= len(heads) - n:
                ups[pi].append((_centre(key[pi]["box"])[1] - _centre(hs[hi])[1]) / max(key[pi]["box"][3], 1e-6))
    out = []
    for p, u in zip(key, ups):
        if not u:
            out.append(None)
            continue
        m = float(np.median(u))
        seated = bool(p.get("seated"))
        out.append(("standing" if m >= RISE_HEADS else "sitting") if seated else
                   ("sitting" if m <= -RISE_HEADS else "standing"))
    return out


def end_state_faults(heads: list, fps: float, key: list, plan_end: list | None = None,
                     next_start: list | None = None) -> list:
    """The take's end against what the shot and the next shot need: each person's posture at the end against the
    plan's end posture for the shot (plan_end) and against the next key's start posture (next_start); both lists
    run in the key's order, entries "standing" / "sitting" / None (unknown: not checked)."""
    got = posture_at_end(heads, fps, key)
    out = []
    for i, g in enumerate(got):
        if g is None:
            continue
        want = plan_end[i] if plan_end and i < len(plan_end) else None
        nxt = next_start[i] if next_start and i < len(next_start) else None
        if want and want != g:
            out.append(f"person {i + 1} of {len(key)} ends {g}, the shot ends with them {want}")
        elif nxt and nxt != g:
            out.append(f"person {i + 1} of {len(key)} ends {g}, the next shot opens with them {nxt}")
    return out


def yaw_from_points(nose: tuple, left_eye: tuple, right_eye: tuple, left_ear: tuple | None = None,
                    right_ear: tuple | None = None) -> float:
    """A head's turn from the camera in degrees (0 facing it, about 90 in profile) from face points in the picture:
    where the nose sits between the ears (or, with no ears found, between the eyes, which is coarser)."""
    a, b = (left_ear, right_ear) if left_ear is not None and right_ear is not None else (left_eye, right_eye)
    half = abs(a[0] - b[0]) / 2.0
    if half < 1e-6:
        return 90.0
    off = (nose[0] - (a[0] + b[0]) / 2.0) / half
    return float(np.degrees(np.arcsin(np.clip(off, -1.0, 1.0)))) if left_ear is not None and right_ear is not None \
        else float(np.clip(abs(off) * 90.0, 0.0, 90.0)) * (1 if off >= 0 else -1)


def speaker_turn_fault(yaws: list, voiced: list) -> str | None:
    """A talk take whose speaker turns away while speaking: yaws per sampled frame (degrees, None = no face found),
    voiced per sampled frame (the line's voice sounds). A profile or a lost face for YAW_SHARE of the voiced frames
    is the fault (lip sync reads a mouth, and a profile shows none)."""
    v = [y for y, on in zip(yaws, voiced) if on]
    if len(v) < 4:
        return None
    away = sum(1 for y in v if y is None or abs(y) >= YAW_MAX)
    if away / len(v) >= YAW_SHARE:
        return f"the speaker is turned away for {100 * away / len(v):.0f}% of the spoken frames"
    return None


def contact_fault(hand: list, target: tuple, fps: float, last_s: float = 0.75) -> str | None:
    """An action staged with a contact (a hand to a place on someone): the hand's point per sampled frame ((x, y) as
    shares of the frame, None = not found) against the staged target region (x, y, w, h), grown by CONTACT_GROW: the
    hand must be inside it for most of the clip's last `last_s` seconds. A coarse test of where the action ends."""
    n = max(2, int(round(last_s * fps)))
    pts = [p for p in hand[-n:] if p is not None]
    if not pts:
        return "the acting hand is not found at the end"
    x, y, w, h = target
    g = CONTACT_GROW
    inside = sum(1 for px, py in pts if x - w * g <= px <= x + w * (1 + g) and y - h * g <= py <= y + h * (1 + g))
    return None if inside * 2 >= len(pts) else "the action does not end where it was staged (the hand is elsewhere)"


# ---- one clip, a folder of clips ----------------------------------------------------------------------------------
def kind_of(name: str) -> str:
    m = re.match(r"(talk|quiet|action)", name)
    return m.group(1) if m else "action"


def clip_faults(mp4: Path, kind: str | None = None, still: bool | None = None, pieces: list | None = None,
                heads: list | None = None, key_people: list | None = None, seam_frames: list | None = None,
                plan_end: list | None = None, next_start: list | None = None) -> dict:
    """One take's faults from these checks. kind: talk | quiet | action (from the name when not given); still:
    nobody moves about (default: talk and quiet). pieces: the line's piece clips in order (their seams are read), or
    seam_frames: the seams' frame numbers inside the joined clip (a chain's lines). heads / key_people: see
    people_over_time; with them plan_end / next_start: see end_state_faults. -> {"faults": [...], "pan", "zoom", "appeared", "seams": [...]}"""
    mp4 = Path(mp4)
    kind = kind or kind_of(mp4.name)
    still = kind in ("talk", "quiet") if still is None else still
    fr = frames(mp4)
    d = drift(fr, still=still, people=heads)
    worst = int(np.argmax([max(m["pan"] / PAN_MAX, abs(m["zoom"]) / ZOOM_MAX) for m in d.get("moves") or [{"pan": 0, "zoom": 0}]])) + 1
    out = {"worst_s": round(worst / SAMPLE_FPS, 2), "match": round(d.get("match", 1.0), 2), "kind": kind, "pan": round(d["pan"], 4), "zoom": round(d["zoom"], 4), "appeared": round(d["appeared"], 4),
           "faults": list(d["faults"]), "seams": []}
    if pieces and len(pieces) > 1 and not seam_frames:      # the joined clip's seams, from its pieces' lengths
        seam_frames, t = [], 0
        for pc in pieces[:-1]:
            t += max(0, _n_frames(pc) - 1)
            seam_frames.append(t + 1)
    for s in seams_in(mp4, seam_frames) if seam_frames else []:
        out["seams"].append({"frame": s.get("frame"), "pan": round(s["pan"], 4), "zoom": round(s["zoom"], 4)})
        if s["fault"]:
            out["faults"].append(s["fault"])
        out["faults"] += s.get("more") or []
    if heads:
        out["faults"] += people_over_time(heads, SAMPLE_FPS, key_people)
        if key_people and (plan_end or next_start):
            out["faults"] += end_state_faults(heads, SAMPLE_FPS, key_people, plan_end, next_start)
    return out


CHAIN_LINE = 97                         # a chain line's frames; each later line's first frame is dropped in the join


def _n_frames(mp4: Path) -> int:
    try:
        return int(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                                   "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(mp4)],
                                  capture_output=True, text=True).stdout.strip() or 0)
    except ValueError:
        return 0


def chain_seams(mp4: Path) -> list:
    """A joined chain's seam frames, from its length: lines of CHAIN_LINE frames overlapped by one (a clip that is
    not such a join has none)."""
    n = _n_frames(mp4)
    k = (n - 1) / (CHAIN_LINE - 1)
    return [i * (CHAIN_LINE - 1) + 1 for i in range(1, int(round(k)))] if n > CHAIN_LINE and abs(k - round(k)) < 1e-6 else []


def table(folder: Path, log=print) -> list:
    """Every shot clip in a folder (takes `_t<k>` and chain takes too; a pieced line's `_c<i>` clips are its seams,
    not rows): the drift numbers and faults, as a table."""
    folder = Path(folder)
    clips = sorted(p for p in folder.glob("*.mp4") if not re.search(r"_c\d+$|\.part$|\.swap$", p.stem))
    rows = []
    log(f"{'clip':46} {'kind':6} {'pan%':>5} {'zoom%':>6} {'new%':>5} {'seams':>6}  faults")
    for p in clips:
        pcs = sorted(folder.glob(f"{p.stem}_c*.mp4"), key=lambda x: int(re.search(r"_c(\d+)$", x.stem).group(1)))
        try:
            r = clip_faults(p, pieces=pcs, seam_frames=None if pcs else chain_seams(p))
        except Exception as e:              # noqa: BLE001  (one unreadable clip does not stop the table)
            r = {"kind": kind_of(p.name), "pan": 0, "zoom": 0, "appeared": 0, "seams": [], "faults": [f"unread: {e!r}"]}
        r["clip"] = p.name
        rows.append(r)
        log(f"{p.stem[:46]:46} {r['kind']:6} {r['pan'] * 100:5.1f} {r['zoom'] * 100:6.1f} {r['appeared'] * 100:5.1f} "
            f"{len(r['seams']):>6}  {'; '.join(r['faults']) or '-'}")
    return rows


def review_agreement(logs: list) -> dict:
    """The reviewer against the judge over review logs (shot_flow's review/log.jsonl): of the cards a reviewer
    answered with a pick, how often it was the judge's. -> {"cards", "picked", "agree", "rate", "overrides",
    "timeouts", "disagreements": [...]}"""
    cards = picked = agree = over = tout = 0
    dis = []
    for lg in logs:
        for line in Path(lg).read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            cards += 1
            c = r.get("choice") or {}
            if c.get("timeout"):
                tout += 1
            elif isinstance(c.get("override"), str) and c["override"].strip():
                over += 1
            elif str(c.get("pick", "")).upper() in ("A", "B", "C") or c.get("accept"):
                picked += 1
                pk = str(c.get("pick") or r.get("judge_pick")).upper()
                if pk == str(r.get("judge_pick")).upper():
                    agree += 1
                else:
                    dis.append({"chain": r.get("chain"), "line_no": r.get("line_no"), "judge": r.get("judge_pick"),
                                "reviewer": pk, "why": c.get("why") or ""})
    return {"cards": cards, "picked": picked, "agree": agree, "rate": round(agree / picked, 3) if picked else None,
            "overrides": over, "timeouts": tout, "disagreements": dis}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="drift, seam and people checks over a folder of rendered takes")
    ap.add_argument("folder", type=Path, nargs="?")
    ap.add_argument("--cut", type=Path, help="a joined cut's list (JSON: [{shot, kind, filled_by: 'clip <path>'}]): "
                                             "its clips, in order, in place of a folder")
    ap.add_argument("--pieces", type=Path, help="with --cut: the folder holding pieced lines' <shot>_c<i>.mp4")
    ap.add_argument("--out", type=Path, help="write the rows as JSON")
    ap.add_argument("--reviews", type=Path, nargs="*", help="review log.jsonl files: the reviewer against the judge")
    a = ap.parse_args()
    if a.cut:
        rows = []
        print(f"{'clip':46} {'kind':6} {'pan%':>5} {'zoom%':>6} {'new%':>5} {'match':>5} {'seams':>5}  faults")
        for e in json.loads(a.cut.read_text()):
            m = re.search(r"clip (\S+\.mp4)", str(e.get("filled_by") or ""))
            if not m:
                continue
            pcs = sorted(a.pieces.glob(f"{e['shot']}_c[0-9]*.mp4"),
                         key=lambda x: int(re.search(r"_c(\d+)$", x.stem).group(1))) if a.pieces else []
            r = clip_faults(Path(m.group(1)), kind=e.get("kind"), pieces=pcs if len(pcs) > 1 else None,
                            seam_frames=None if len(pcs) > 1 else chain_seams(Path(m.group(1))))
            r["clip"] = e["shot"]
            rows.append(r)
            print(f"{e['shot'][:46]:46} {r['kind']:6} {r['pan'] * 100:5.1f} {r['zoom'] * 100:6.1f} "
                  f"{r['appeared'] * 100:5.1f} {r['match']:5.2f} {len(r['seams']):>5}  {'; '.join(r['faults']) or '-'}")
    elif a.folder:
        rows = table(a.folder)
    else:
        ap.error("a folder of clips, or --cut")
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))
    if a.reviews:
        print(json.dumps(review_agreement(a.reviews), indent=1))
    sys.exit(0)
