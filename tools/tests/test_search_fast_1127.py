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
