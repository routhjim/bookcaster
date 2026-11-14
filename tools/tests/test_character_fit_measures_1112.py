"""The fit's own measures (img3d/bl_combo.py, bl_settle.py), on small synthetic point sets, without Blender.

bl_combo.py and bl_settle.py are Blender scripts (they import bpy and run top to bottom on a job), so they cannot be
imported here. The functions they define, and one self-contained inline block (the neckline measured on the
generated figure), are taken from their source by name and run against synthetic stand-ins for the script's globals
(the figure's vertex positions and colours, the body's joints). A function or block that is renamed or removed fails
its test by name; the constants the script passes to the cuff finder are read from its call sites.

Covered: the fixed skin-colour test and why it is not enough (tan cloth passes it); the colour vote measured against
the figure's own hands and chest cloth; the cuff finder on sleeves that end mid-forearm, short of the wrist, at the
wrist and not at all, and the hem on legs; a slice's girth; the bare limb's girth capped under the sleeve's opening;
the neckline class from the ring at mid-neck and the chest below the shoulder line (collar, base, open; a beard and
long hair round an open neck); Taubin smoothing by weight; the settle's capsule distance."""

import ast
import math
import re
import unittest
from pathlib import Path

import numpy as np

IMG3D = Path(__file__).resolve().parents[1] / "assets" / "img3d"
COMBO = IMG3D / "bl_combo.py"
SETTLE = IMG3D / "bl_settle.py"


def functions(path, names):
    """source of the named top-level functions of a script, in file order"""
    src = path.read_text()
    tree = ast.parse(src)
    got = {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names}
    missing = [n for n in names if n not in got]
    assert not missing, f"{path.name}: function(s) not found: {missing}"
    return "\n\n".join(got[n] for n in names)


def block(path, marker, kind=ast.If):
    """source of the top-level statement of KIND whose text contains MARKER (one inline block of a script)"""
    src = path.read_text()
    hits = [ast.get_source_segment(src, n) for n in ast.parse(src).body
            if isinstance(n, kind) and marker in (ast.get_source_segment(src, n) or "")]
    assert len(hits) == 1, f"{path.name}: {len(hits)} top-level blocks contain {marker!r} (expected 1)"
    return hits[0]


class _Numpy:
    """numpy as Blender's own Python has it (1.26): np.cross of two 2-D vectors gives the scalar z component, which
    numpy 2.x no longer does (bl_combo._hull_perimeter relies on it)"""

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def cross(a, b, *args, **kw):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape[-1] == 2 and b.shape[-1] == 2:
            return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
        return np.cross(a, b, *args, **kw)


NP = _Numpy()


class V(tuple):
    """a joint position with .x .y .z, as the script's mathutils vectors"""
    x = property(lambda s: s[0])
    y = property(lambda s: s[1])
    z = property(lambda s: s[2])


def ring(c, axis, r, n=24):
    """n points on a circle of radius r round AXIS at centre c"""
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
    v = np.cross(axis, u)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(c)[None, :] + r * (np.cos(a)[:, None] * u[None, :] + np.sin(a)[:, None] * v[None, :])


def tube(S, E, f0, f1, radius, step=0.004):
    """rings round the axis S->E from fraction f0 to f1 of its length; radius a number or a function of the fraction"""
    S, E = np.asarray(S, float), np.asarray(E, float)
    L = np.linalg.norm(E - S)
    out = []
    for p in np.arange(f0 * L, f1 * L, step):
        r = radius(p / L) if callable(radius) else radius
        out.append(ring(S + (E - S) / L * p, E - S, r))
    return np.vstack(out) if out else np.zeros((0, 3))


SKIN_C = np.array([0.80, 0.58, 0.46])
TAN_C = np.array([0.56, 0.44, 0.30])        # a tan tunic: passes the fixed skin test
BLACK_C = np.array([0.06, 0.06, 0.07])
WHITE_C = np.array([0.90, 0.90, 0.88])

JOINTS = {"pelvis": V((0.0, 0.0, 0.95)), "spine_03": V((0.0, 0.0, 1.20)), "spine_05": V((0.0, 0.0, 1.38)),
          "upperarm_l": V((0.18, 0.0, 1.42)), "hand_l": V((0.60, 0.0, 1.07)), "lowerarm_l": V((0.39, 0.0, 1.245)),
          "upperarm_r": V((-0.18, 0.0, 1.42)), "hand_r": V((-0.60, 0.0, 1.07)), "lowerarm_r": V((-0.39, 0.0, 1.245)),
          "thigh_l": V((0.09, 0.0, 0.93)), "foot_l": V((0.10, 0.0, 0.08)), "calf_l": V((0.095, 0.0, 0.50)),
          "thigh_r": V((-0.09, 0.0, 0.93)), "foot_r": V((-0.10, 0.0, 0.08)), "calf_r": V((-0.095, 0.0, 0.50))}


def combo_ns(co, col=None, skl=None, job=None, extra=()):
    """the fit script's functions over a synthetic figure"""
    ns = {"np": NP, "math": math, "co": co, "col": col, "JOB": job or {}, "J": lambda n: JOINTS[n],
          "pose_head": lambda n: JOINTS[n], "SKL": skl, "SKL_SURE": None, "LIMB_GIRTH": {}}
    exec(functions(COMBO, ["is_skin", "_skin_like", "_limb_edge", "_hull_perimeter", "_girth_at", "_limb_girths"]), ns)
    if col is not None:
        ns["skin"] = ns["is_skin"](col)
    ns.update(dict(extra))
    return ns


def cuff_args(which):
    """the constants bl_combo passes to _limb_edge for the arms (S0, E0) and the legs (S1, E1)"""
    m = re.search(r"_limb_edge\(" + which + r", sg, ([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+), ([0-9.]+)\)",
                  COMBO.read_text())
    assert m, f"the _limb_edge call for {which} was not found in bl_combo.py"
    return [float(x) for x in m.groups()]


ARM = cuff_args("S0, E0")
LEG = cuff_args("S1, E1")
# the cuff's station is read from medians over 8 one-centimetre stations each side, so it is known to about 4 cm
CUFF_TOL_M = 0.04
S_ARM, E_ARM = np.array(JOINTS["upperarm_l"]), np.array(JOINTS["hand_l"])
S_LEG, E_LEG = np.array(JOINTS["thigh_l"]), np.array(JOINTS["foot_l"])


def arm(sleeve_to, sleeve_r=0.065, limb_r=0.035, limb_skin=True, after=None):
    """a left arm: a sleeve from the shoulder to SLEEVE_TO of the arm's length, the bare limb beyond it"""
    sl = tube(S_ARM, E_ARM, 0.0, sleeve_to, sleeve_r)
    lb = tube(S_ARM, E_ARM, sleeve_to, 1.08, after or limb_r)
    co = np.vstack([sl, lb])
    skl = np.r_[np.zeros(len(sl), bool), np.full(len(lb), limb_skin)]
    return co, skl


class SkinColour(unittest.TestCase):
    def test_fixed_test_takes_skin_and_refuses_black_and_white(self):
        ns = combo_ns(np.zeros((3, 3)))
        got = ns["is_skin"](np.array([SKIN_C, BLACK_C, WHITE_C]))
        self.assertEqual(list(got), [True, False, False])

    def test_tan_cloth_passes_the_fixed_test(self):
        # the reason for the measured vote: a fixed skin test cannot tell a tan garment from skin
        ns = combo_ns(np.zeros((1, 3)))
        self.assertTrue(ns["is_skin"](TAN_C[None, :])[0])

    def figure(self, hands=True, hand_c=SKIN_C, cloth_c=TAN_C):
        parts, cols = [], []
        chest = np.array([[x, y, z] for x in (-0.10, -0.07, 0.07, 0.10) for y in (-0.10, 0.10)
                          for z in np.linspace(1.22, 1.36, 8)])
        parts.append(chest)
        cols.append(np.tile(cloth_c, (len(chest), 1)))
        for s in ("l", "r"):
            S, E = np.array(JOINTS[f"upperarm_{s}"]), np.array(JOINTS[f"hand_{s}"])
            sl = tube(S, E, 0.0, 0.7, 0.06)
            parts.append(sl)
            cols.append(np.tile(cloth_c, (len(sl), 1)))
            if hands:
                hd = tube(S, E, 1.06, 1.20, 0.03)
                parts.append(hd)
                cols.append(np.tile(hand_c, (len(hd), 1)))
        return np.vstack(parts), np.vstack(cols)

    def test_vote_is_measured_against_the_figures_own_hands_and_cloth(self):
        co, col = self.figure()
        ns = combo_ns(co, col)
        like, refs = ns["_skin_like"]()
        hand = np.linalg.norm(col - SKIN_C, axis=1) < 1e-6
        self.assertTrue(like[hand].all())
        self.assertFalse(like[~hand].any())              # the tan cloth is cloth, though the fixed test calls it skin
        self.assertTrue(ns["skin"][~hand].all())
        self.assertAlmostEqual(refs["apart"], float(np.linalg.norm(SKIN_C - TAN_C)), places=2)

    def test_sure_skin_is_stricter_than_the_vote(self):
        co, col = self.figure()
        mid = 0.55 * SKIN_C + 0.45 * TAN_C            # between the two: nearer skin, not clearly skin
        co = np.vstack([co, [[0.0, 0.3, 1.0]] * 5])
        col = np.vstack([col, np.tile(mid, (5, 1))])
        ns = combo_ns(co, col)
        like, _ = ns["_skin_like"]()
        self.assertTrue(like[-5:].all())
        self.assertFalse(ns["SKL_SURE"][-5:].any())

    def test_without_hands_it_falls_back_to_the_fixed_test(self):
        co, col = self.figure(hands=False)
        ns = combo_ns(co, col)
        like, refs = ns["_skin_like"]()
        self.assertIsNone(refs)
        self.assertTrue((like == ns["skin"]).all())


class CuffFinder(unittest.TestCase):
    def edge(self, co, skl, args=ARM, S=S_ARM, E=E_ARM):
        ns = combo_ns(co, skl=skl)
        return ns["_limb_edge"](S, E, 1, *args)

    def test_sleeve_ending_mid_forearm_is_found_at_its_end(self):
        co, skl = arm(0.62)
        c, p, rad, side, rep = self.edge(co, skl)
        L = np.linalg.norm(E_ARM - S_ARM)
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c, 0.62 * L, delta=CUFF_TOL_M)
        self.assertAlmostEqual(rep["above_cm"], 6.5, delta=0.5)
        self.assertAlmostEqual(rep["below_cm"], 3.5, delta=0.5)
        self.assertGreaterEqual(rep["step_cm"] / 100, ARM[1])

    def test_short_sleeve_above_the_elbow(self):
        co, skl = arm(0.45)
        c = self.edge(co, skl)[0]
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c, 0.45 * np.linalg.norm(E_ARM - S_ARM), delta=CUFF_TOL_M)

    def test_sleeve_ending_short_of_the_wrist(self):
        co, skl = arm(0.85)
        c = self.edge(co, skl)[0]
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c, 0.85 * np.linalg.norm(E_ARM - S_ARM), delta=CUFF_TOL_M)

    def test_sleeve_to_the_wrist_has_no_cuff(self):
        # a long sleeve: only the hand is bare, which the hand cut takes; nothing changes on the limb
        co, skl = arm(0.99)
        self.assertIsNone(self.edge(co, skl)[0])

    def test_no_sleeve_at_all_has_no_cuff(self):
        co = tube(S_ARM, E_ARM, 0.0, 1.08, 0.035)
        self.assertIsNone(self.edge(co, np.ones(len(co), bool))[0])

    def test_a_step_smaller_than_the_minimum_is_not_a_cuff(self):
        co, skl = arm(0.62, sleeve_r=0.035 + 0.8 * ARM[1], limb_r=0.035)
        self.assertIsNone(self.edge(co, skl)[0])

    def test_a_step_under_the_ratio_is_not_a_cuff(self):
        # a thick limb in a sleeve only a little wider: the step is large enough in cm but under the ratio
        limb = 0.07
        co, skl = arm(0.62, sleeve_r=limb * (ARM[2] - 0.12), limb_r=limb)
        self.assertGreater(limb * (ARM[2] - 0.12) - limb, ARM[1])
        self.assertIsNone(self.edge(co, skl)[0])

    def test_cloth_that_widens_again_below_is_not_a_cuff(self):
        # a sleeve gathered by a band and full again below it: the limb does not stay thin to its end
        co, skl = arm(0.50, after=lambda f: 0.035 if f < 0.62 else 0.07)
        self.assertIsNone(self.edge(co, skl)[0])

    def test_thin_cloth_below_the_step_is_not_a_bare_limb(self):
        # a close under-sleeve of cloth below a wide over-sleeve: thin, but not the figure's skin
        co, skl = arm(0.62, limb_skin=False)
        self.assertIsNone(self.edge(co, skl)[0])

    def test_only_this_side_of_the_body_is_read(self):
        co, skl = arm(0.62)
        mirror = co * np.array([-1.0, 1.0, 1.0])
        ns = combo_ns(mirror, skl=skl)
        self.assertIsNone(ns["_limb_edge"](S_ARM, E_ARM, 1, *ARM)[0])

    def test_hem_on_a_leg_and_trousers_to_the_ankle(self):
        hem = np.vstack([tube(S_LEG, E_LEG, 0.0, 0.60, 0.11), tube(S_LEG, E_LEG, 0.60, 1.0, 0.05)])
        skl = np.r_[np.zeros(len(tube(S_LEG, E_LEG, 0.0, 0.60, 0.11)), bool),
                    np.ones(len(tube(S_LEG, E_LEG, 0.60, 1.0, 0.05)), bool)]
        c = self.edge(hem, skl, LEG, S_LEG, E_LEG)[0]
        self.assertIsNotNone(c)
        self.assertAlmostEqual(c, 0.60 * np.linalg.norm(E_LEG - S_LEG), delta=CUFF_TOL_M)
        trousers = tube(S_LEG, E_LEG, 0.0, 1.0, 0.09)
        self.assertIsNone(self.edge(trousers, np.zeros(len(trousers), bool), LEG, S_LEG, E_LEG)[0])


class Girth(unittest.TestCase):
    def test_hull_perimeter(self):
        ns = combo_ns(np.zeros((1, 3)))
        sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]], float)
        self.assertAlmostEqual(ns["_hull_perimeter"](sq), 4.0, places=6)
        self.assertAlmostEqual(ns["_hull_perimeter"](sq + 7.3), 4.0, places=6)       # the offset does not matter
        self.assertIsNone(ns["_hull_perimeter"](sq[:2]))
        a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        self.assertAlmostEqual(ns["_hull_perimeter"](0.04 * np.c_[np.cos(a), np.sin(a)]), 2 * np.pi * 0.04, places=3)

    def test_girth_of_a_limb_at_a_station(self):
        co = tube(S_ARM, E_ARM, 0.0, 1.0, 0.04)
        ns = combo_ns(co)
        ax = (E_ARM - S_ARM) / np.linalg.norm(E_ARM - S_ARM)
        g = ns["_girth_at"](co, S_ARM, ax, 0.30, 1)
        self.assertAlmostEqual(g, 100 * 2 * np.pi * 0.04, delta=0.4)
        self.assertIsNone(ns["_girth_at"](co, S_ARM, ax, 0.30, -1))                # the other side of the body
        self.assertIsNone(ns["_girth_at"](co[:8], S_ARM, ax, 0.30, 1))             # too few points for a slice

    def limb_girths(self, sleeve_r, limb_r, body_r, job=None):
        """the figure: a sleeve to 0.45 of the arm, its bare limb beyond; the body: a thinner arm"""
        fig = np.vstack([tube(S_ARM, E_ARM, 0.0, 0.45, sleeve_r), tube(S_ARM, E_ARM, 0.45, 1.1, limb_r)])
        bodyv = tube(S_ARM, E_ARM, 0.0, 1.1, body_r)

        class _Co:
            def __init__(self, p):
                self.co = tuple(p)

        class _Body:
            def evaluated_get(self, _dg):
                return self
            data = type("D", (), {"vertices": [_Co(p) for p in bodyv]})()

        class _Bpy:
            context = type("C", (), {"evaluated_depsgraph_get": staticmethod(lambda: None)})()
        ns = combo_ns(fig, job=job, extra={"body": _Body(), "bpy": _Bpy()})
        L = float(np.linalg.norm(E_ARM - S_ARM))
        return ns["_limb_girths"]("arm", "l", 1, S_ARM, E_ARM, 0.45 * L)

    def test_bare_forearm_is_sized_from_the_figure(self):
        out = self.limb_girths(sleeve_r=0.075, limb_r=0.045, body_r=0.030)
        f = out["Forearm"]
        self.assertAlmostEqual(f["figure_cm"], 100 * 2 * np.pi * 0.045, delta=0.6)
        self.assertAlmostEqual(f["body_cm"], 100 * 2 * np.pi * 0.030, delta=0.6)
        self.assertAlmostEqual(f["delta_cm"], f["figure_cm"] - f["body_cm"], delta=0.3)     # the body grows to the figure
        self.assertLessEqual(f["capped_cm"], out["_cap_cm"])

    def test_the_limb_is_capped_under_the_sleeves_opening(self):
        # a drawn limb nearly as wide as the sleeve it comes out of: the body's limb stays under the opening
        out = self.limb_girths(sleeve_r=0.050, limb_r=0.049, body_r=0.030)
        frac = 0.9                                         # bl_combo's default "limb_opening_frac"
        self.assertRegex(COMBO.read_text(), r'JOB\.get\("limb_opening_frac", 0\.9\)')
        self.assertAlmostEqual(out["_cap_cm"], frac * out["_opening_cm"], delta=0.11)
        self.assertLess(out["_cap_cm"], out["_opening_cm"])
        f = out["Forearm"]
        self.assertLess(f["capped_cm"], f["figure_cm"])
        self.assertAlmostEqual(f["capped_cm"], out["_cap_cm"], delta=0.11)

    def test_the_cap_fraction_is_the_jobs(self):
        out = self.limb_girths(sleeve_r=0.050, limb_r=0.049, body_r=0.030, job={"limb_opening_frac": 0.8})
        self.assertAlmostEqual(out["_cap_cm"], 0.8 * out["_opening_cm"], delta=0.11)


class NecklineOnTheFigure(unittest.TestCase):
    """The inline block that classes the neckline on the generated figure before any cut or push."""

    NECK = V((0.0, 0.0, 1.50))
    JN = {"neck_01": NECK, "clavicle_l": V((0.05, 0.0, 1.47)), "head": V((0.0, 0.0, 1.62))}
    R = 0.055                                              # the head mesh's neck radius

    def classify(self, parts, groups=None):
        """parts: list of (points, is_skin); groups: the parser's label group of each part (None: no labels)"""
        src = block(COMBO, 'REP["neckline_figure"]')
        co = np.vstack([p for p, _ in parts])
        skin = np.concatenate([np.full(len(p), s) for p, s in parts])
        head_ring = ring(np.array(self.NECK), [0, 0, 1], self.R, 40)

        class _Vt:
            def __init__(self, p):
                self.co = np.asarray(p)

        class _M:
            def __matmul__(self, v):
                return v
        HEAD = type("H", (), {"matrix_world": _M(), "data": type("D", (), {"vertices": [_Vt(p) for p in head_ring]})()})()
        ns = {"np": np, "math": math, "STATUE": False, "HEAD": HEAD, "J": lambda n: self.JN[n], "co": co,
              "skin": skin, "REP": {},
              "LAB_GROUP": None if groups is None else np.concatenate([np.full(len(p), g) for (p, _), g in zip(parts, groups)])}
        exec(src, ns)
        rep = ns["REP"]["neckline_figure"]
        self.assertNotIn("error", rep, rep)
        return rep

    def collar_ring(self, sectors=range(16), r=0.075):
        """points just outside the neck at mid-neck height, in the given sixteenths of the circle"""
        zmid = 1.47 + 0.5 * ((1.62 - 0.04) - 1.47)
        pts = []
        for k in sectors:
            for a in np.linspace(-np.pi + (k + 0.15) * np.pi / 8, -np.pi + (k + 0.85) * np.pi / 8, 5):
                for dz in (-0.006, 0.0, 0.006):
                    pts.append([r * np.cos(a), r * np.sin(a), zmid + dz])
        return np.array(pts)

    def chest_front(self):
        """points in front of the neck's axis just under the shoulder line"""
        return np.array([[x, y, 1.44 + dz] for x in np.linspace(-0.06, 0.06, 9) for y in (-0.08, -0.06, -0.04)
                         for dz in (-0.006, 0.006)])

    def test_cloth_all_round_the_neck_is_a_collar_to_the_chin(self):
        rep = self.classify([(self.collar_ring(), False), (self.chest_front(), False)])
        self.assertEqual(rep["class"], "chin")
        self.assertEqual(rep["cloth_sectors_at_mid_neck"], 16)

    def test_cloth_on_the_chest_and_a_bare_neck_is_a_neckline_at_the_base(self):
        rep = self.classify([(self.collar_ring(), True), (self.chest_front(), False)])
        self.assertEqual(rep["class"], "base")
        self.assertEqual(rep["cloth_sectors_at_mid_neck"], 0)

    def test_skin_on_the_chest_is_an_open_neckline(self):
        rep = self.classify([(self.collar_ring(), True), (self.chest_front(), True)])
        self.assertEqual(rep["class"], "open")

    def test_hair_all_round_the_neck_is_no_collar_when_the_figure_is_labelled(self):
        # long hair worn down, not skin-coloured: by colour a collar to the chin, by the parser's labels (1 = hair,
        # 4 = garment) a neckline at the base
        parts = [(self.collar_ring(), False), (self.chest_front(), False)]
        self.assertEqual(self.classify(parts)["class"], "chin")
        rep = self.classify(parts, groups=[1, 4])
        self.assertEqual(rep["cloth_sectors_at_mid_neck"], 0)
        self.assertEqual(rep["class"], "base")

    def front_sectors(self):
        """the sixteenths of the ring in front of the neck (the block's front is -y)"""
        return [k for k in range(16) if math.sin(-math.pi + (k + 0.5) * math.pi / 8) < -0.3]

    def test_a_beard_in_front_of_a_base_neckline_is_not_a_collar(self):
        beard = self.collar_ring(self.front_sectors())
        rep = self.classify([(beard, False), (self.chest_front(), False)])
        self.assertLess(rep["cloth_sectors_at_mid_neck"], 10)
        self.assertEqual(rep["class"], "base")

    def test_a_beard_and_long_hair_round_an_open_neck_are_still_open(self):
        # hair behind and beside the neck and a beard in front fill almost every sector: the chest's skin decides
        ringed = self.collar_ring([k for k in range(16) if k != 3])
        rep = self.classify([(ringed, False), (self.chest_front(), True)])
        self.assertGreaterEqual(rep["cloth_sectors_at_mid_neck"], 10)
        self.assertEqual(rep["class"], "open")


class Smoothing(unittest.TestCase):
    def test_taubin_by_weight(self):
        n = 60
        rng = np.random.default_rng(3)
        P = np.c_[np.linspace(0, 1, n), 0.02 * rng.standard_normal(n), np.zeros(n)]
        ED = np.c_[np.arange(n - 1), np.arange(1, n)]
        DEG = np.bincount(ED.ravel(), minlength=n)
        ns = {"np": np, "ED": ED, "DEG": DEG}
        exec(functions(COMBO, ["smooth_pos"]), ns)
        w = np.ones(n)
        w[:20] = 0.0
        Q = ns["smooth_pos"](P.copy(), w, 10)
        self.assertTrue(np.allclose(Q[:20], P[:20]))                             # weight 0: fixed
        rough = lambda X: float(np.abs(np.diff(X[:, 1], 2)).mean())            # noqa: E731
        self.assertLess(rough(Q[22:]), 0.5 * rough(P[22:]))
        self.assertAlmostEqual(float(Q[-1, 0] - Q[21, 0]), float(P[-1, 0] - P[21, 0]), delta=0.03)   # no shrink


class SettleCapsule(unittest.TestCase):
    def test_distance_to_a_tapered_capsule(self):
        ns = {"np": np}
        exec(functions(SETTLE, ["cap_dist"]), ns)
        a, b = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
        X = np.array([[0.2, 0.0, 0.5], [0.05, 0.0, 0.0], [0.0, 0.0, 1.5], [0.02, 0.0, 1.0]])
        sd, nr, c, r, t = ns["cap_dist"](X, a, b, 0.10, 0.04)
        self.assertAlmostEqual(sd[0], 0.2 - 0.07, places=6)                       # outside, radius between the ends'
        self.assertLess(sd[1], 0)                                                 # inside the thick end
        self.assertAlmostEqual(sd[2], 0.5 - 0.04, places=6)                       # beyond the thin end
        self.assertLess(sd[3], 0)
        self.assertTrue(np.allclose(nr[0], [1, 0, 0]))
        self.assertTrue(np.allclose(t, [0.5, 0.0, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
