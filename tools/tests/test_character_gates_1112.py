"""The character build's two gates and the drawing's legs measure, on synthetic figures (no GPU, no Blender, no
engine): img3d/pose_gate.py (the 2D gate: the row method, and the column method for arms near horizontal, with the
hand inside the picture and at its edge), img3d/mesh_gate.py (arms free of the body and the air at the elbow, with
and without a web of cloth under the arm; the column method), and cast_char.legs_apart (trousers or a long skirt).

A figure is drawn from the same proportions as img3d/apose_map.py (shoulders at 0.815 of the height, 0.112 out),
as a filled silhouette: the 2D gate reads it as a picture, the mesh gate as the points of its filled cells.
The thresholds are the gates' own command-line defaults, read from their parsers' source, so a deliberate change of
a threshold shows as these tests' expectations moving with it and a broken measure as a named failure."""

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ASSETS = Path(__file__).resolve().parents[1] / "assets"
TMP = tempfile.mkdtemp(prefix="char_gates_test_")
os.environ.setdefault("ASSETS_SCHED_DIR", os.path.join(TMP, "sched"))
os.environ.setdefault("ASSETS_LIBRARY", os.path.join(TMP, "lib"))
sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(ASSETS / "img3d"))

import pose_gate as PG  # noqa: E402
import mesh_gate as MG  # noqa: E402


def cli_default(module, flag):
    """a gate's own default for a command-line flag, from its parser's source"""
    src = Path(module.__file__).read_text()
    m = re.search(r'add_argument\("' + re.escape(flag) + r'"[^)]*default=([0-9.]+)', src)
    assert m, f"{flag} has no default in {module.__name__}"
    return float(m.group(1))


MIN_ANGLE = cli_default(PG, "--min-angle")
MIN_ARMPIT = cli_default(PG, "--min-armpit")
KEY = cli_default(PG, "--key")
MIN_FREE = cli_default(MG, "--min-free")
MIN_GAP = cli_default(MG, "--min-gap")
ELBOW = cli_default(MG, "--elbow")

BG = (200, 200, 200)
FG = (60, 50, 45)


def figure(deg, w=896, h=1344, fig=1000, web_to=None, arm_len=0.335, arm_w=0.055, skirt=False):
    """A front-view figure in a spread pose with the arms DEG from vertical. web_to: a sheet of cloth from the arm's
    underside to the body's side, down to that height fraction (None: none). skirt: one block to the floor in place
    of two legs. Returns the RGB picture."""
    im = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(im)
    cx, top = w / 2, (h - fig) / 2

    def P(x, z):
        return (cx + x * fig, top + (1 - z) * fig)
    d.ellipse([P(-0.045, 1.0), P(0.045, 0.875)], fill=FG)                      # head
    d.rectangle([P(-0.022, 0.88), P(0.022, 0.83)], fill=FG)                    # neck
    d.rectangle([P(-0.112, 0.835), P(0.112, 0.50)], fill=FG)                   # torso
    if skirt:
        d.polygon([P(-0.112, 0.52), P(0.112, 0.52), P(0.17, 0.0), P(-0.17, 0.0)], fill=FG)
    else:
        d.rectangle([P(-0.105, 0.52), P(-0.02, 0.0)], fill=FG)                 # legs, air between them
        d.rectangle([P(0.02, 0.52), P(0.105, 0.0)], fill=FG)
    th = np.radians(deg)
    for s in (-1, 1):
        sx, sz = s * 0.112, 0.815
        ex, ez = sx + s * arm_len * np.sin(th), sz - arm_len * np.cos(th)
        d.line([P(sx, sz), P(ex, ez)], fill=FG, width=int(arm_w * fig))
        d.ellipse([P(ex - 0.03, ez + 0.03), P(ex + 0.03, ez - 0.03)], fill=FG)  # the hand
        if web_to is not None:
            # cloth between the arm's underside and the body's side, from the armpit down to web_to
            t = (sz - web_to) / max(np.cos(th), 1e-6) / arm_len
            wx, wz = sx + s * min(t, 1.0) * arm_len * np.sin(th), sz - min(t, 1.0) * arm_len * np.cos(th)
            d.polygon([P(sx, sz), P(wx, wz), P(sx, web_to)], fill=FG)
    return im


def save(im, name):
    p = os.path.join(TMP, name)
    im.save(p)
    return p


def points(im):
    """the figure's filled cells as (x, z) points with z up, as the mesh gate reads a model's front projection"""
    a = np.asarray(im).astype(np.float32)
    fg = np.linalg.norm(a - np.array(BG, np.float32), axis=2) > 40
    ys, xs = np.nonzero(fg[::2, ::2])
    return np.c_[xs * 2.0, -(ys * 2.0)]


class PoseGateRows(unittest.TestCase):
    def test_arms_at_45_pass_by_the_row_method(self):
        r = PG.gate(save(figure(45), "a45.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertTrue(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertNotIn("by", r[side])                       # rows, not the column fallback
            self.assertAlmostEqual(r[side]["angle_deg"], 45, delta=6)
            self.assertGreaterEqual(r[side]["armpit"], MIN_ARMPIT)
            self.assertLess(r[side]["armpit"], 0.84)             # under the shoulder line (0.815 + the arm's width)

    def test_arms_hanging_close_fail_the_angle(self):
        r = PG.gate(save(figure(MIN_ANGLE - 15), "a_low.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertFalse(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            if r[side]["angle_deg"] is not None:
                self.assertLess(r[side]["angle_deg"], MIN_ANGLE)

    def test_a_web_under_the_arm_lowers_the_armpit_and_fails(self):
        web = MIN_ARMPIT - 0.04
        r = PG.gate(save(figure(45, web_to=web), "a_web.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertFalse(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertAlmostEqual(r[side]["armpit"], web, delta=0.015)     # the armpit is where the web ends
            self.assertLess(r[side]["armpit"], MIN_ARMPIT)

    def test_a_web_that_ends_above_the_bar_passes_and_is_measured(self):
        web = MIN_ARMPIT + 0.03
        r = PG.gate(save(figure(45, web_to=web), "a_web_hi.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertTrue(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertAlmostEqual(r[side]["armpit"], web, delta=0.015)

    def test_a_web_down_to_the_hand_never_passes(self):
        r = PG.gate(save(figure(45, web_to=0.58), "a_web_deep.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertFalse(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertTrue(r[side]["armpit"] is None or r[side]["armpit"] < MIN_ARMPIT, r)

    def test_both_sides_are_judged_one_bad_side_fails(self):
        im = figure(45)
        d = ImageDraw.Draw(im)
        w, h = im.size
        cx, top, fig = w / 2, (h - 1000) / 2, 1000
        # a web on the picture's right side only
        d.polygon([(cx + 0.112 * fig, top + 0.185 * fig), (cx + 0.30 * fig, top + 0.40 * fig),
                   (cx + 0.112 * fig, top + 0.42 * fig)], fill=FG)
        r = PG.gate(save(im, "a_one_side.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertFalse(r["pass"], r)
        self.assertTrue(r["left_of_picture"]["pass"], r)
        self.assertFalse(r["right_of_picture"]["pass"], r)


class PoseGateColumns(unittest.TestCase):
    """Arms near horizontal: no row crosses the arm apart from the body, so the gate measures by columns."""

    def test_arms_near_horizontal_pass_by_columns(self):
        r = PG.gate(save(figure(84, w=1152, fig=920), "a84.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertTrue(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertEqual(r[side].get("by"), "columns", r)
            self.assertGreaterEqual(r[side]["armpit"], MIN_ARMPIT)
            self.assertGreater(r[side]["angle_deg"], 70)

    def test_hand_at_the_picture_edge_still_finds_the_armpit(self):
        # the outstretched hands reach the picture's edge (a 896-wide canvas at these angles): the pocket under the
        # arm is closed at the hand's reach, so the armpit is still measured
        im = figure(86, w=896, fig=1000, arm_len=0.345)
        a = np.asarray(im).astype(np.float32)
        fg = np.linalg.norm(a - np.array(BG, np.float32), axis=2) > 40
        self.assertTrue(fg[:, 0].any() and fg[:, -1].any(), "the test figure's hands must touch both edges")
        r = PG.gate(save(im, "a_edge.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertIsNotNone(r[side]["armpit"], r)
            self.assertGreaterEqual(r[side]["armpit"], MIN_ARMPIT, r)

    def test_cloth_filling_the_space_under_level_arms_fails(self):
        # an outer layer hanging from the arms to the knees: no air under the arm at all
        im = figure(84, w=1152, fig=920)
        d = ImageDraw.Draw(im)
        w, h = im.size
        cx, top, fig = w / 2, (h - 920) / 2, 920
        for s in (-1, 1):
            d.polygon([(cx + s * 0.112 * fig, top + 0.185 * fig), (cx + s * 0.44 * fig, top + 0.16 * fig),
                       (cx + s * 0.44 * fig, top + 0.75 * fig), (cx + s * 0.112 * fig, top + 0.75 * fig)], fill=FG)
        r = PG.gate(save(im, "a_cloak.png"), KEY, MIN_ANGLE, MIN_ARMPIT)
        self.assertFalse(r["pass"], r)

    def test_runs_drop_specks(self):
        row = np.zeros(100, bool)
        row[10:13] = True          # a speck
        row[40:60] = True
        self.assertEqual([(int(a), int(b)) for a, b in PG.runs(row)], [(40, 60)])


class MeshGate(unittest.TestCase):
    def test_free_arms_pass(self):
        r = MG.gate(points(figure(45)), MIN_FREE, MIN_GAP, ELBOW)
        self.assertTrue(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertGreaterEqual(r[side]["free"], MIN_FREE)
            self.assertGreaterEqual(r[side]["gap_at_elbow"], MIN_GAP)

    def test_a_web_fails_free_and_gap(self):
        r = MG.gate(points(figure(45, web_to=0.60)), MIN_FREE, MIN_GAP, ELBOW)
        self.assertFalse(r["pass"], r)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertTrue(r[side]["free"] is None or r[side]["free"] < MIN_FREE, r)
            self.assertLess(r[side]["gap_at_elbow"], MIN_GAP, r)

    def test_free_height_follows_the_web(self):
        hi = MG.gate(points(figure(45, web_to=0.74)), MIN_FREE, MIN_GAP, ELBOW)
        lo = MG.gate(points(figure(45, web_to=0.64)), MIN_FREE, MIN_GAP, ELBOW)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertGreater(hi[side]["free"], lo[side]["free"])
            self.assertAlmostEqual(lo[side]["free"], 0.64, delta=0.03)

    def test_level_arms_are_measured_by_columns(self):
        r = MG.gate(points(figure(86, w=1152, fig=920)), MIN_FREE, MIN_GAP, ELBOW)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertEqual(r[side].get("by"), "columns", r)
            self.assertGreaterEqual(r[side]["free"], MIN_FREE, r)
            self.assertGreaterEqual(r[side]["gap_at_elbow"], MIN_GAP, r)
        self.assertTrue(r["pass"], r)

    def test_scale_does_not_matter(self):
        P = points(figure(45))
        a, b = MG.gate(P, MIN_FREE, MIN_GAP, ELBOW), MG.gate(P * 0.0018, MIN_FREE, MIN_GAP, ELBOW)
        for side in ("left_of_picture", "right_of_picture"):
            self.assertAlmostEqual(a[side]["free"], b[side]["free"], delta=0.01)
            self.assertEqual(a[side]["pass"], b[side]["pass"])


class LegsApart(unittest.TestCase):
    """cast_char.legs_apart: the drawing's fallback measure for a garment to the ankles."""

    @classmethod
    def setUpClass(cls):
        import cast_char as CC
        cls.CC = CC

    def test_two_legs_read_as_trousers(self):
        m = self.CC.legs_apart(save(figure(45), "legs.png"))
        self.assertEqual(m["shin_runs"], 2)
        self.assertFalse(m["long_skirt"])

    def test_one_block_to_the_floor_reads_as_a_long_skirt(self):
        m = self.CC.legs_apart(save(figure(45, skirt=True), "skirt.png"))
        self.assertEqual(m["shin_runs"], 1)
        self.assertTrue(m["long_skirt"])

    def test_two_shoes_under_a_hem_read_as_trousers_which_is_why_the_text_fact_decides(self):
        # the known weakness of the drawing's measure (two shoes under a gown's hem at shin height): the build must
        # take the text-first fact over it; this pins the measure's behaviour that makes that rule necessary
        im = figure(45, skirt=True)
        d = ImageDraw.Draw(im)
        w, h = im.size
        cx, top, fig = w / 2, (h - 1000) / 2, 1000
        d.rectangle([cx - 0.2 * fig, top + 0.78 * fig, cx + 0.2 * fig, top + fig + 2], fill=BG)   # the hem at 0.22
        d.rectangle([cx - 0.09 * fig, top + 0.78 * fig, cx - 0.03 * fig, top + fig], fill=FG)     # two ankles
        d.rectangle([cx + 0.03 * fig, top + 0.78 * fig, cx + 0.09 * fig, top + fig], fill=FG)
        m = self.CC.legs_apart(save(im, "hem.png"))
        self.assertFalse(m["long_skirt"])


if __name__ == "__main__":
    unittest.main()
