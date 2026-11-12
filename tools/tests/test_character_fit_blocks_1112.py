"""Inline rules of the fit script (img3d/bl_combo.py), each run as the block of source that holds it, on
synthetic figures and without Blender:

  - the hands found along the drawing's own arm angle (a skin-coloured garment made the far end of the "skin" a
    wrist at the hem; the body is posed to these wrists, so this is what binds a figure in its drawing's pose);
  - the sleeve tube measured from the sleeve itself (a puffed sleeve keeps its puff, a coat sleeve stays narrow,
    the tube is capped short of the torso near the shoulder);
  - the collar lift as one ramp over the whole garment from the hip line to the collar, a sleeve following its
    shoulder end.

Also the figure's neck line, searched only near the body's own neck height (long loose hair made the top of the
head the narrowest slice, and the figure's face stayed on the garment).

A block is found by the words that only it contains; a rule that is moved into a function, renamed or removed fails
here by name (then test the function instead). The script's globals are plain arrays and small stand-ins."""

import ast
import math
import textwrap
import unittest
from pathlib import Path

import numpy as np

COMBO = Path(__file__).resolve().parents[1] / "assets" / "img3d" / "bl_combo.py"
SRC = COMBO.read_text()
TREE = ast.parse(SRC)


def block(kind, *contains, top_level=False):
    """the smallest statement of KIND whose source contains every string in CONTAINS"""
    nodes = TREE.body if top_level else ast.walk(TREE)
    hits = sorted(((len(ast.get_source_segment(SRC, n) or ""), n) for n in nodes if isinstance(n, kind)
                   and all(c in (ast.get_source_segment(SRC, n) or "") for c in contains)), key=lambda h: h[0])
    assert hits, f"bl_combo.py: no {kind.__name__} block contains {contains}"
    assert len(hits) == 1 or hits[0][0] < hits[1][0], f"bl_combo.py: two equal blocks contain {contains}"
    n = hits[0][1]
    return textwrap.dedent(" " * n.col_offset + ast.get_source_segment(SRC, n))


def function(name):
    for n in TREE.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(SRC, n)
    raise AssertionError(f"bl_combo.py: no function {name}")


class V(tuple):
    x = property(lambda s: s[0])
    y = property(lambda s: s[1])
    z = property(lambda s: s[2])


def ring(c, axis, r, n=24):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(axis, [0.0, 1.0, 0.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0.0, 0.0])
    v = np.cross(axis, u)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.asarray(c)[None, :] + r * (np.cos(a)[:, None] * u[None, :] + np.sin(a)[:, None] * v[None, :])


def tube(S, E, f0, f1, radius, step=0.004):
    S, E = np.asarray(S, float), np.asarray(E, float)
    L = np.linalg.norm(E - S)
    out = [ring(S + (E - S) / L * p, E - S, radius(p / L) if callable(radius) else radius)
           for p in np.arange(f0 * L, f1 * L, step)]
    return np.vstack(out) if out else np.zeros((0, 3))


SKIN_C = np.array([0.80, 0.58, 0.46])
TAN_C = np.array([0.56, 0.44, 0.30])
BLACK_C = np.array([0.06, 0.06, 0.07])
J = {"pelvis": V((0.0, 0.0, 0.95)), "upperarm_l": V((0.18, 0.0, 1.42)), "upperarm_r": V((-0.18, 0.0, 1.42)),
     "hand_l": V((0.60, 0.0, 1.07)), "hand_r": V((-0.60, 0.0, 1.07)), "foot_l": V((0.10, 0.0, 0.08)),
     "foot_r": V((-0.10, 0.0, 0.08)), "clavicle_l": V((0.05, 0.0, 1.47))}
ARM_L = float(np.linalg.norm(np.array(J["hand_l"]) - np.array(J["upperarm_l"])))
TOP_Z = 1.75


class HandsAlongTheDrawingsAngle(unittest.TestCase):
    def figure(self, deg, cloth):
        """a figure in a long flared garment of colour CLOTH, its arms DEG from vertical, bare hands beyond the sleeves"""
        parts, cols = [], []

        def add(P, c):
            parts.append(np.asarray(P, float))
            cols.append(np.tile(c, (len(P), 1)))
        add([[x, y, z] for x in np.linspace(-0.15, 0.15, 9) for y in (-0.1, 0.1) for z in np.linspace(0.9, 1.45, 20)], cloth)
        add([[x, y, z] for x in np.linspace(-0.42, 0.42, 25) for y in (-0.15, 0.15) for z in np.linspace(0.25, 0.9, 20)], cloth)
        add([[s * 0.1, 0.0, 0.03 + 0.005 * k] for s in (-1, 1) for k in range(8)], BLACK_C)          # shoes
        th = math.radians(deg)
        for sg in (1, -1):
            S = np.array([sg * 0.18, 0.0, 1.42])
            E = S + ARM_L * np.array([sg * math.sin(th), 0.0, -math.cos(th)])
            add(tube(S, E, 0.0, 0.8, 0.05), cloth)
            add(tube(S, E, 1.0, 1.25, 0.03), SKIN_C)
        return np.vstack(parts), np.vstack(cols)

    def wrists(self, deg, cloth, draw_deg):
        co, col = self.figure(deg, cloth)
        ns = {"np": np, "math": math}
        exec(function("is_skin"), ns)
        ns.update(co=co, skin=ns["is_skin"](col), sh={"l": J["upperarm_l"], "r": J["upperarm_r"]},
                  JOB={} if draw_deg is None else {"draw_deg": draw_deg}, FIT_POSE=False, J=lambda n: J[n], REP={}, LM={},
                  top_z=TOP_Z)
        with np.errstate(all="ignore"):
            exec(block(ast.For, "hand_search", 'LM[f"wrist_{side}"]', top_level=True), ns)
        return ns["LM"], ns["REP"]

    def expected(self, deg, sg):
        th = math.radians(deg)
        return np.array([sg * 0.18, 0.0, 1.42]) + ARM_L * np.array([sg * math.sin(th), 0.0, -math.cos(th)])

    def test_level_arms_in_a_skin_coloured_garment(self):
        LM, REP = self.wrists(90, TAN_C, 90)
        for side, sg in (("l", 1), ("r", -1)):
            w = LM[f"wrist_{side}"]
            self.assertAlmostEqual(w[2], 1.42, delta=0.06)                    # at shoulder height, where it was drawn
            self.assertGreater(sg * w[0], 0.5)
            self.assertLess(np.linalg.norm((w - self.expected(90, sg))[[0, 2]]), 0.13)
            self.assertIn(side + "_expected", REP["hand_search"])

    def test_without_the_drawings_angle_the_garments_hem_was_taken_for_the_hand(self):
        # the failure the rule answers: tan cloth passes the skin test, and the far end of that "skin" is the hem
        LM, _ = self.wrists(90, TAN_C, None)
        self.assertLess(LM["wrist_l"][2], 1.0)
        self.assertLess(LM["wrist_r"][2], 1.0)

    def test_the_angle_is_followed(self):
        for deg in (45, 66):
            LM, _ = self.wrists(deg, TAN_C, deg)
            for side, sg in (("l", 1), ("r", -1)):
                self.assertLess(np.linalg.norm((LM[f"wrist_{side}"] - self.expected(deg, sg))[[0, 2]]), 0.13,
                                f"{deg} deg {side}")

    def test_dark_cloth_gives_the_same_hands_with_or_without_the_angle(self):
        a, _ = self.wrists(45, BLACK_C, 45)
        b, _ = self.wrists(45, BLACK_C, None)
        for side in ("l", "r"):
            self.assertLess(np.linalg.norm(a[f"wrist_{side}"] - b[f"wrist_{side}"]), 0.08)

    def test_the_palm_is_beyond_the_wrist(self):
        LM, _ = self.wrists(90, TAN_C, 90)
        self.assertGreater(LM["palm_l"][0], LM["wrist_l"][0])
        self.assertLess(LM["palm_r"][0], LM["wrist_r"][0])


class NeckLineOfTheFigure(unittest.TestCase):
    """The figure's neck (where it is cut and scaled to the body) is looked for only near where the body's own neck
    sits as a share of its height: the drawing was made from this body's pose map."""
    TOP = 1.75
    JN = {"neck_01": V((0.0, 0.0, 1.49)), "neck_02": V((0.0, 0.0, 1.535))}       # mid-neck at 0.864 of the height

    def width(self, f):
        """the figure's width at height fraction f: long loose hair round the head, a crown that tapers to a point"""
        if f < 0.80:
            return 0.40
        if f < 0.83:
            return 0.20
        if f < 0.87:
            return 0.11            # the neck, between the hair's strands
        if f < 0.95:
            return 0.19            # the head inside its hair
        return 0.19 - (f - 0.95) / 0.05 * 0.17          # the crown: narrower than the neck at the very top

    def match(self, job=None, scale=1.0):
        zs = np.arange(0.0, 1.0005, 0.001)
        co = np.array([[x * scale, 0.0, f * self.TOP * scale] for f in zs
                       for x in np.linspace(-self.width(f) / 2, self.width(f) / 2, 21)])
        ns = {"np": np, "match": "neck", "co": co, "zmin": co[:, 2].min(), "zmax": co[:, 2].max(), "top_z": self.TOP,
              "J": lambda n: self.JN[n], "JOB": job or {}, "REP": {}, "white": np.zeros(len(co), bool)}
        exec(block(ast.If, 'REP["neck_match"]', "neck_search_below", top_level=True), ns)
        return ns["REP"]["neck_match"], ns["s"]

    def test_the_crown_is_narrower_than_the_neck_in_this_figure(self):
        self.assertLess(self.width(0.995), self.width(0.85))               # the trap the window avoids

    def test_the_neck_is_found_in_the_window_round_the_bodys_own(self):
        rep, s = self.match()
        self.assertAlmostEqual(rep["expected_frac"], 0.864, delta=0.003)
        self.assertGreaterEqual(rep["figure_neck_frac"], 0.83)
        self.assertLess(rep["figure_neck_frac"], 0.87)
        self.assertLessEqual(rep["window"][1], 0.95)                        # never up into the top of the head
        self.assertGreaterEqual(rep["window"][0], 0.74)
        self.assertAlmostEqual(rep["width_frac"], 0.11 / self.TOP, delta=0.005)

    def test_the_figure_is_scaled_to_put_its_neck_at_the_bodys(self):
        rep, s = self.match(scale=0.6)                                      # a model comes out at any size
        mid = 0.5 * (self.JN["neck_01"].z + self.JN["neck_02"].z)
        self.assertAlmostEqual(s * rep["figure_neck_frac"] * self.TOP * 0.6, mid, delta=0.003)
        self.assertAlmostEqual(s, 1 / 0.6 * mid / (rep["figure_neck_frac"] * self.TOP), delta=0.01)

    def test_the_window_is_the_jobs(self):
        rep, _ = self.match(job={"neck_search_below": 0.02, "neck_search_above": 0.01})
        self.assertAlmostEqual(rep["window"][0], rep["expected_frac"] - 0.02, delta=0.002)
        self.assertAlmostEqual(rep["window"][1], rep["expected_frac"] + 0.01, delta=0.002)


class MeasuredSleeveTube(unittest.TestCase):
    S, E = np.array(J["upperarm_l"]), np.array(J["hand_l"])
    FLOOR = 0.10

    def run_tube(self, sleeve_radius, extra=(), torso=None, job=None):
        """sleeve_radius: number or function of the fraction along the arm. extra: other cloth (not arm-weighted).
        torso: the body's torso vertices (default: none near the arm). Returns (in tube per vertex, report, n sleeve)."""
        sl = tube(self.S, self.E, 0.02, 1.0, sleeve_radius)
        ex = np.asarray(extra, float).reshape(-1, 3)
        fco = np.vstack([sl, ex])
        armw = np.r_[np.ones(len(sl)), np.zeros(len(ex))]
        tv = np.asarray(torso if torso is not None else [[-0.6, 0.0, 1.9]], float)

        class _Co:
            def __init__(self, p):
                self.co = tuple(p)

        class _Torso:
            def evaluated_get(self, _):
                return self
            data = type("D", (), {"vertices": [_Co(p) for p in tv]})()
        bpy = type("B", (), {"context": type("C", (), {"evaluated_depsgraph_get": staticmethod(lambda: None)})()})()
        j = {"sleeve_radius_m": self.FLOOR}
        j.update(job or {})
        ns = {"np": np, "fco": fco, "armw": armw, "torso": _Torso(), "bpy": bpy, "JOB": j, "REP": {},
              "SLEEVE_R": float(j["sleeve_radius_m"]), "tube": np.zeros(len(fco), bool),
              "pose_head": lambda n: {"upperarm": V(tuple(self.S)), "hand": V(tuple(self.E))}[n.rsplit("_", 1)[0]]}
        src = block(ast.For, "sleeve_tube_m", "rprof", top_level=True).replace('for side in ("l", "r")', 'for side in ("l",)')
        self.assertIn('for side in ("l",)', src, "the sleeve loop no longer runs over the two sides by name")
        exec(src, ns)
        return ns["tube"], ns["REP"].get("sleeve_tube_m", {}).get("l"), len(sl)

    def off_axis(self, frac, r):
        """a point R from the arm's axis at FRAC of its length"""
        ax = (self.E - self.S) / np.linalg.norm(self.E - self.S)
        n = np.cross(ax, [0.0, 1.0, 0.0])
        return self.S + ax * frac * ARM_L + n / np.linalg.norm(n) * r

    def puff(self, f):
        return 0.11 if 0.05 < f < 0.30 else 0.05

    def test_a_puffed_sleeve_keeps_its_puff(self):
        tb, rep, n = self.run_tube(self.puff)
        self.assertGreater(tb[:n].mean(), 0.97)                     # the sleeve is all in its tube
        self.assertGreaterEqual(rep["max"], 0.11)
        self.assertLess(rep["max"], 0.16)

    def test_a_narrow_sleeve_gets_a_narrow_tube_at_the_upper_arm(self):
        side = [self.off_axis(0.2, 0.09)]                           # the garment's side beside the upper arm
        tb_n, rep_n, n = self.run_tube(0.05, extra=side)
        tb_p, rep_p, m = self.run_tube(self.puff, extra=side)
        self.assertFalse(tb_n[n])                                    # not the coat's side (no cape under a raised arm)
        self.assertTrue(tb_p[m])                                     # inside a puff's own width
        self.assertLess(rep_n["min"], rep_p["max"])
        self.assertGreater(tb_n[:n].mean(), 0.97)

    def test_cloth_far_from_the_arm_is_never_in_the_tube(self):
        far = [self.off_axis(f, 0.20) for f in (0.2, 0.5, 0.8)]
        for radius in (0.05, self.puff):
            tb, _, n = self.run_tube(radius, extra=far)
            self.assertFalse(tb[n:].any())

    def test_along_the_forearm_the_tube_is_never_under_the_jobs_floor(self):
        near = [self.off_axis(0.7, self.FLOOR - 0.015)]
        tb, rep, n = self.run_tube(0.04, extra=near)
        self.assertTrue(tb[n])
        self.assertGreaterEqual(rep["max"], self.FLOOR)

    def test_near_the_shoulder_the_tube_stops_short_of_the_torso(self):
        # the body's side 6 cm from the arm's axis along the upper arm: the puff beyond it is not the sleeve's
        torso = [self.off_axis(f, 0.06) for f in np.arange(0.0, 0.36, 0.01)]
        tb, rep, n = self.run_tube(self.puff, torso=torso)
        self.assertLessEqual(rep["min"], 0.051)
        sl = tube(self.S, self.E, 0.02, 1.0, self.puff)
        ax = (self.E - self.S) / ARM_L
        p = (sl - self.S) @ ax
        puff_part = (p > 0.10 * ARM_L) & (p < 0.25 * ARM_L)
        self.assertLess(tb[:n][puff_part].mean(), 0.05)

    def test_the_fixed_tube_when_measuring_is_switched_off(self):
        side = [self.off_axis(0.2, 0.13), self.off_axis(0.2, 0.17)]
        tb, rep, n = self.run_tube(0.05, extra=side, job={"sleeve_tube_measured": False, "sleeve_radius_m": 0.15})
        self.assertIsNone(rep)
        self.assertTrue(tb[n])
        self.assertFalse(tb[n + 1])


class CollarLiftRamp(unittest.TestCase):
    """The whole garment rises by one smooth ramp from the hip line to the collar; a sleeve follows its shoulder end."""
    TOP0 = 1.50

    def lift(self, lift, P, groups, z_sh=1.45):
        names = ["spine_03", "upperarm_l", "lowerarm_l", "clavicle_l", "upperarm_r"]

        class _G:
            def __init__(self, i, w):
                self.group, self.weight = i, w

        class _Vt:
            def __init__(self, i, gs):
                self.index, self.groups = i, [_G(names.index(n), w) for n, w in gs]
        fig = type("F", (), {"vertex_groups": [type("VG", (), {"index": i, "name": n})() for i, n in enumerate(names)],
                             "data": type("D", (), {"vertices": [_Vt(i, g) for i, g in enumerate(groups)]})()})()
        disp = np.zeros((len(P), 3))
        ns = {"np": np, "P_": np.asarray(P, float), "lift": lift, "top0": self.TOP0, "z_sh": z_sh,
              "J": lambda n: J[n], "disp": disp, "REP": {}, "fig": fig}
        exec(block(ast.If, "dz_arm", "lift_shape", "armw_"), ns)
        return disp, ns["REP"]

    def test_torso_cloth_rises_by_a_smooth_ramp_from_the_hip_line(self):
        zs = np.linspace(0.60, self.TOP0, 46)
        P = [[0.0, -0.1, z] for z in zs]
        disp, rep = self.lift(0.03, P, [[("spine_03", 1.0)]] * len(P))
        dz = disp[:, 2]
        self.assertTrue((disp[:, :2] == 0).all())                         # straight up only
        self.assertTrue((dz[zs <= J["pelvis"].z] == 0).all())             # nothing at or below the hip line
        self.assertAlmostEqual(dz[-1], 0.03, places=6)                    # the full lift at the collar's top
        self.assertTrue((np.diff(dz) >= -1e-12).all())                    # one ramp, never back down
        mid = np.argmin(np.abs(zs - (J["pelvis"].z + self.TOP0) / 2))
        self.assertAlmostEqual(dz[mid], 0.015, delta=0.003)
        self.assertLess(np.diff(dz).max() / (zs[1] - zs[0]), 0.03 / (self.TOP0 - J["pelvis"].z) * 1.6)     # no step
        self.assertEqual(rep["collar_height"]["lift_shape"], "torso")

    def test_a_sleeve_follows_its_shoulder_end_and_fades_to_the_cuff(self):
        S, H = np.array(J["upperarm_l"]), np.array(J["hand_l"])
        fr = np.linspace(0.0, 1.0, 11)
        P = [S + f * (H - S) for f in fr]
        disp, _ = self.lift(0.03, P, [[("upperarm_l", 1.0)]] * len(P))
        dz = disp[:, 2]
        torso_at_shoulder, _ = self.lift(0.03, [[0.0, 0.0, S[2]]], [[("spine_03", 1.0)]])
        self.assertAlmostEqual(dz[0], torso_at_shoulder[0, 2], places=6)  # with the torso at the armhole: no tear
        self.assertAlmostEqual(dz[-1], 0.0, places=9)                     # nothing at the cuff
        self.assertTrue((np.diff(dz) <= 1e-12).all())
        # by the torso's ramp alone the cuff (well above the hip line here) would have risen
        t_only, _ = self.lift(0.03, [P[-1]], [[("spine_03", 1.0)]])
        self.assertGreater(t_only[0, 2], 0.001)

    def test_a_clavicle_weight_is_torso_and_a_mixed_vertex_blends(self):
        p = [0.12, 0.0, 1.44]
        clav, _ = self.lift(0.03, [p], [[("clavicle_l", 1.0)]])
        tors, _ = self.lift(0.03, [p], [[("spine_03", 1.0)]])
        arm, _ = self.lift(0.03, [p], [[("upperarm_l", 1.0)]])
        half, _ = self.lift(0.03, [p], [[("spine_03", 0.5), ("upperarm_l", 0.5)]])
        self.assertAlmostEqual(clav[0, 2], tors[0, 2], places=9)
        self.assertAlmostEqual(half[0, 2], 0.5 * (tors[0, 2] + arm[0, 2]), places=9)

    def test_both_sides_are_alike(self):
        S, H = np.array(J["upperarm_l"]), np.array(J["hand_l"])
        p_l = S + 0.4 * (H - S)
        p_r = p_l * np.array([-1.0, 1.0, 1.0])
        left, _ = self.lift(0.03, [p_l], [[("upperarm_l", 1.0)]])
        right, _ = self.lift(0.03, [p_r], [[("upperarm_r", 1.0)]])
        self.assertAlmostEqual(left[0, 2], right[0, 2], places=9)

    def test_no_lift_worth_making_moves_nothing(self):
        P = [[0.0, -0.1, 1.3]]
        disp, rep = self.lift(0.002, P, [[("spine_03", 1.0)]])
        self.assertTrue((disp == 0).all())
        self.assertNotIn("collar_height", rep)
        # a "collar" whose top is at the shoulder line is a neckline, not a collar: not lifted
        disp, _ = self.lift(0.03, P, [[("spine_03", 1.0)]], z_sh=self.TOP0)
        self.assertTrue((disp == 0).all())


if __name__ == "__main__":
    unittest.main()
