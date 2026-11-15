"""Rules of the fit's bare-limb step (img3d/bl_combo.py), run as their source on synthetic arrays without Blender:
the convex-hull girth of a slice (a 2-D cross written out, numpy 2) and the cuff found at the steepest drop of a
plateau of windowed steps, not the first station of the plateau."""

import ast
import math
import unittest
from pathlib import Path

import numpy as np

COMBO = Path(__file__).resolve().parents[1] / "assets" / "img3d" / "bl_combo.py"
SRC = COMBO.read_text()
TREE = ast.parse(SRC)


def function(name):
    for n in TREE.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(SRC, n)
    raise AssertionError(f"bl_combo.py: no function {name}")


class HullPerimeter(unittest.TestCase):
    def setUp(self):
        self.ns = {"np": np}
        exec(function("_hull_perimeter"), self.ns)

    def test_circle_girth(self):
        a = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        r = 0.05
        pts = np.c_[r * np.cos(a), r * np.sin(a)]
        inner = 0.5 * pts                                   # points inside do not change the hull
        g = self.ns["_hull_perimeter"](np.r_[pts, inner])
        self.assertAlmostEqual(g, 2 * math.pi * r, delta=0.001)

    def test_square_and_too_few(self):
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
        self.assertAlmostEqual(self.ns["_hull_perimeter"](sq), 4.0, places=6)
        self.assertIsNone(self.ns["_hull_perimeter"](np.array([[0, 0], [1, 1]], float)))


class CuffStation(unittest.TestCase):
    """an arm along +X: a sleeve of radius 12 cm to 0.40 m, a bare forearm of 7.5 cm beyond; the cuff must land
    where the radius drops, not at the first station of the flat run of windowed steps"""

    def run_edge(self, noise_seed):
        rng = np.random.default_rng(noise_seed)
        pts, sk = [], []
        for x in np.arange(0.0, 0.62, 0.002):
            r = 0.12 if x < 0.40 else 0.075
            a = rng.uniform(0, 2 * np.pi, 30)
            rr = r * (1 + rng.normal(0, 0.01, 30))
            pts.append(np.c_[np.full(30, x), rr * np.cos(a), rr * np.sin(a) + 1.0])
            sk.append(np.full(30, x >= 0.40))
        co = np.vstack(pts)
        ns = {"np": np, "co": co, "SKL": np.concatenate(sk),
              "J": lambda n: type("P", (), {"x": -1.0})()}
        exec(function("_limb_edge"), ns)
        S0, E0 = np.array([0.0, 0.0, 1.0]), np.array([0.6, 0.0, 1.0])
        c, *_ = ns["_limb_edge"](S0, E0, 1, 0.14, 0.012, 1.3, 0.3, 1.0)
        return c

    def test_cuff_at_the_drop(self):
        for seed in range(5):
            c = self.run_edge(seed)
            self.assertIsNotNone(c)
            self.assertLess(abs(c - 0.40), 0.015, f"seed {seed}: cuff at {c:.3f} m, the drop is at 0.400 m")


if __name__ == "__main__":
    unittest.main()
