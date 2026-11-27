"""11-27: the camera search's two inner steps made cheaper, with the same answers. A place's staging spent most of
its hour and a half in them: a point projected through a camera worked the camera's axes out again every time, and a
line of sight asked every thing of the set at every point along it. Made-up data only."""
import math
import random
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("scenes", ""):
    sys.path.insert(0, str(A / sub))
import framing as F  # noqa: E402
import speakers as S  # noqa: E402


def project_plain(cam, p, res=(1152, 640)):
    """The projection as it was written before."""
    d0 = [cam["target"][i] - cam["position"][i] for i in range(3)]
    if not any(abs(x) > 1e-9 for x in d0):
        return None
    f = F._unit(d0)
    up0 = [0.0, 0.0, 1.0] if abs(f[2]) < 0.999 else [0.0, 1.0, 0.0]
    r = F._unit([f[1] * up0[2] - f[2] * up0[1], f[2] * up0[0] - f[0] * up0[2], f[0] * up0[1] - f[1] * up0[0]])
    u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
    v = [p[i] - cam["position"][i] for i in range(3)]
    z = sum(v[i] * f[i] for i in range(3))
    if z <= 1e-6:
        return None
    x = sum(v[i] * r[i] for i in range(3))
    y = sum(v[i] * u[i] for i in range(3))
    return (0.5 + cam["focal_mm"] * x / z / F.SENSOR_W, 0.5 - cam["focal_mm"] * y / z / F.sensor_h(res), z)


def test_a_point_lands_where_it_did_to_the_last_bit():
    rnd = random.Random(7)
    cams = [{"position": [rnd.uniform(-9, 9), rnd.uniform(-9, 9), rnd.uniform(0.3, 6)],
             "target": [rnd.uniform(-9, 9), rnd.uniform(-9, 9), rnd.uniform(0, 3)], "focal_mm": f}
            for f in (20, 28, 50, 85) for _ in range(40)]
    cams += [{"position": [1.0, 2.0, 5.0], "target": [1.0, 2.0, 0.0], "focal_mm": 35},       # straight down
             {"position": [1.0, 2.0, 1.5], "target": [1.0, 2.0, 1.5], "focal_mm": 35}]       # on its own aim
    for _ in range(3):                                   # (asked again: from the kept axes)
        for cam in cams:
            for _ in range(12):
                p = [rnd.uniform(-10, 10), rnd.uniform(-10, 10), rnd.uniform(0, 3)]
                assert F.project(cam, p) == project_plain(cam, p)
    cam = {"position": [0.0, -4.0, 1.6], "target": [0.0, 0.0, 1.5], "focal_mm": 35}
    was = F.project(cam, [0.5, 0.0, 1.0])
    cam["position"] = [3.0, -4.0, 1.6]                   # (the same camera moved: its axes are not the old ones)
    assert F.project(cam, [0.5, 0.0, 1.0]) == project_plain(cam, [0.5, 0.0, 1.0]) != was


def between_plain(p, q, n=24):
    for i in range(1, n):
        t = i / n
        if t > 0.85:
            break
        pt = [p[j] + (q[j] - p[j]) * t for j in range(3)]
        if any(S._in_box(pt, b) for b in S.SOLIDS):
            return True
    return False


def test_a_line_of_sight_is_blocked_by_the_same_things(monkeypatch):
    rnd = random.Random(3)
    boxes = []
    for _ in range(30):
        a = rnd.uniform(0, 2 * math.pi)
        boxes.append([rnd.uniform(-8, 8), rnd.uniform(-8, 8), rnd.uniform(0.1, 2.0), rnd.uniform(0.1, 2.0),
                      rnd.uniform(0.3, 4.0), math.cos(a), math.sin(a)])
    monkeypatch.setattr(S, "SOLIDS", boxes)
    monkeypatch.setattr(S, "HEADS", [])
    monkeypatch.setattr(S, "NECKS", [])
    hit = 0
    for _ in range(4000):
        p = [rnd.uniform(-10, 10), rnd.uniform(-10, 10), rnd.uniform(0.3, 6)]
        q = [rnd.uniform(-10, 10), rnd.uniform(-10, 10), rnd.uniform(0.3, 2)]
        got = S.solid_between(p, q)
        assert got == between_plain(p, q), (p, q)
        hit += got
    assert 400 < hit < 3600                              # (both answers are met with)


def test_the_first_sieve_refuses_no_camera_the_rules_would_take():
    """Every camera the sieve refuses for all cameras at once is one framing_why refuses too, over made-up people
    in every kind of shot; and it does refuse most of those the rules refuse (it is worth its work)."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import staging_synth as SY
    import search_fast as SF
    rnd = random.Random(11)
    lim = {"EDGE": S.EDGE, "SIDE_EDGE": S.SIDE_EDGE, "SLIVER": S.SLIVER, "FULL_LEAD_K": S.FULL_LEAD_K,
           "BEHIND_M": S.BEHIND_M}
    res = [1152, 640]
    refused = failed = tried = 0
    for trial in range(60):
        n = rnd.choice((2, 3, 4))
        names = list("abcd")[:n]
        subj = {}
        for k in names:
            a = rnd.uniform(0, 2 * math.pi)
            subj[k] = SY.person([rnd.uniform(-2.5, 2.5), rnd.uniform(-2.5, 2.5)], (math.cos(a), math.sin(a)),
                                seated=rnd.random() < 0.2)
        kind = rnd.choice(("full", "wide", "medium", "close"))
        spk = rnd.choice((None, None, "a"))
        lis = "b" if spk and rnd.random() < 0.5 else None
        show = rnd.choice((None, names, names[:1], names[:2]))
        opts = {"line_subject": rnd.choice(("a", names[-1], None)), "subject": "a",
                "listed": rnd.choice((None, names[:1], names[:2], names))}
        qs = []
        for _ in range(250):
            t = subj[rnd.choice(names)]["measured"]
            qs.append({"position": [round(rnd.uniform(-7, 7), 3), round(rnd.uniform(-7, 7), 3), rnd.choice((1.45, 1.6, 2.4))],
                       "target": [round(t["eye_xy"][0] + rnd.uniform(-0.6, 0.6), 3),
                                  round(t["eye_xy"][1] + rnd.uniform(-0.6, 0.6), 3), round(rnd.uniform(1.0, 1.7), 3)],
                       "focal_mm": rnd.choice((20, 24, 28, 35, 50, 85))})
        qs.append({"position": [0.0, 0.0, 5.0], "target": [0.0, 0.0, 0.0], "focal_mm": 28})      # (straight down)
        qs.append({"position": [1.0, 1.0, 1.5], "target": [1.0, 1.0, 1.5], "focal_mm": 28})      # (on its own aim)
        got = SF.sure_fail(np.array([q["position"] for q in qs], float), np.array([q["target"] for q in qs], float),
                           np.array([q["focal_mm"] for q in qs], float), kind, spk, lis, show, subj, res, opts, lim)
        for q, x in zip(qs, got):
            why = S.framing_why(q, kind, spk, lis, False, show, subj, res, opts=opts)[0]
            tried += 1
            failed += bool(why)
            refused += bool(x)
            assert why or not x, (trial, q, kind, spk, lis, show, opts)
    assert refused > 0.5 * failed > 0.2 * tried, (refused, failed, tried)


def test_the_search_is_the_same_with_and_without_its_sieve(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import staging_synth as SY
    monkeypatch.setattr(S, "SOLIDS", [])
    monkeypatch.setattr(S, "OPEN_GROUND", [])
    monkeypatch.setattr(S, "PICTURE_SIDE", {})
    subj = {"a": SY.person([0.0, 1.0], (0.3, -1.0)), "b": SY.person([1.3, 0.8], (-0.6, -1.0)),
            "c": SY.person([-1.4, 1.4], (0.5, -1.0))}
    room = [8.0, 8.0, 3.0]
    for c, spk, lis in (({"kind": "full", "show": ["a", "b", "c"], "listed": ["a", "b", "c"], "line_subject": "a"}, None, None),
                        ({"kind": "medium", "show": ["a"], "listed": ["a"], "line_subject": "a"}, None, None),
                        ({"kind": "medium"}, "a", "b")):
        shot = {"camera": SY.cam([0.0, -3.0, 1.6], [0.0, 1.0, 1.5], 35)}
        opts = {k: c[k] for k in ("listed", "line_subject") if k in c}
        monkeypatch.delenv("SPEAKERS_PLAIN_SEARCH", raising=False)
        fast = S.two_shot_candidates(c, shot, spk, lis, subj, [1152, 640], room, None, None, [], dict(opts))
        monkeypatch.setenv("SPEAKERS_PLAIN_SEARCH", "1")
        plain = S.two_shot_candidates(c, shot, spk, lis, subj, [1152, 640], room, None, None, [], dict(opts))
        assert fast == plain
    assert fast                                  # (the talk shot finds cameras)
