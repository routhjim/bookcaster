"""A seat made standing on its own plinth is measured above the plinth: its front is told from the seat's own back,
and its width is the room between its arms (measured with the plinth in, the front could not be told, the seat
came out a quarter of a metre wide, and nobody could be sat on it)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
import seatbox as SB  # noqa: E402


def block(x0, x1, y0, y1, z0, z1, n=6000, seed=0):
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(x0, x1, n), r.uniform(y0, y1, n), r.uniform(z0, z1, n)], 1)


def armed_seat(back_at=1.0, plinth=True):
    """An armed seat 0.9 m wide, its back toward back_at along y, on a slab 0.15 m high that reaches past it."""
    up = 0.15 if plinth else 0.0
    parts = [block(-0.45, 0.45, -0.35, 0.35, up + 0.35, up + 0.5, seed=1),                 # the seat
             block(-0.45, 0.45, 0.25, 0.35, up + 0.5, up + 1.5, seed=2),                   # its back
             block(-0.45, -0.33, -0.35, 0.25, up + 0.5, up + 0.75, seed=3),                # the arms
             block(0.33, 0.45, -0.35, 0.25, up + 0.5, up + 0.75, seed=4),
             block(-0.45, -0.37, -0.35, 0.35, up, up + 0.35, 2000, seed=5),                # the legs' sides
             block(0.37, 0.45, -0.35, 0.35, up, up + 0.35, 2000, seed=6)]
    if plinth:
        parts.append(block(-0.6, 0.6, -0.55, 0.55, 0.0, 0.15, 12000, seed=7))
    P = np.concatenate(parts)
    P[:, 1] *= back_at
    return P


def measure(P):
    P = SB.above_plinth(P)
    f = SB.front_of(P)
    yaw = np.arctan2(f[1], f[0]) - np.arctan2(-1.0, 0.0)
    c, s = np.cos(-yaw), np.sin(-yaw)
    return f, SB.seat_box(np.stack([P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c, P[:, 2]], 1))


def test_plinth_does_not_hide_the_front():
    for back_at in (1.0, -1.0):
        f, box = measure(armed_seat(back_at))
        assert f == [0.0, -back_at], (back_at, f)
        assert 0.55 < box["width_m"] < 0.75, box
        assert 0.45 < box["seat_z_m"] < 0.7, box


def test_no_plinth_is_left_whole():
    P = armed_seat(plinth=False)
    assert len(SB.above_plinth(P)) == len(P)
    f, box = measure(P)
    assert f == [0.0, -1.0] and 0.55 < box["width_m"] < 0.75, (f, box)


def test_nothing_at_the_seat_level_does_not_stop_the_measure():
    P = np.concatenate([block(-0.3, 0.3, -0.3, 0.0, 0.3, 0.4, seed=1), block(-0.3, 0.3, 0.25, 0.3, 0.4, 1.2, seed=2)])
    assert SB.seat_box(P)["width_m"] > 0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v()
    print("all passed")
