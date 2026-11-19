"""A whole figure's jointed hands (img3d/statue_hands.py, used by bl_combo.py in statue mode), without Blender:

  - the cut across the forearm: the narrowest section near the wrist, so a plain sleeve or a bare forearm is cut at
    the wrist and a wide cuff or bracer that overlaps the wrist stays whole (the cut lands past its end, where the
    hand comes out of it); the section measured round its own centre, not the arm's axis line;
  - covering: bare hands take their skin colour, gloves and gauntlets (not skin-coloured, or far from the face's
    skin) take the covering's colour;
  - the fallback to the old rigid statue hand, with its reason, when the body's hand cannot be fitted;
  - bl_combo keeps it behind the job's data, on by default for statues, and colours the hands in the statue's own
    material slot (never a "skin" slot, which the engine paints with the MetaHuman's skin).
"""
import sys
import unittest
from pathlib import Path

import numpy as np

IMG3D = Path(__file__).resolve().parents[1] / "assets" / "img3d"
sys.path.insert(0, str(IMG3D))
import statue_hands as SH  # noqa: E402

AX = np.array([1.0, 0.0, 0.0])


def tube(profile, offset=(0.0, 0.0, 0.0), n_ring=24, step=0.0025):
    """Points on rings along +x: profile(station) -> radius; rings centred at offset (y, z) off the axis line."""
    pts, st = [], []
    for s in np.arange(-0.10, 0.20, step):
        r = profile(s)
        for k in range(n_ring):
            a = 2 * np.pi * k / n_ring
            pts.append([s, offset[1] + r * np.cos(a), offset[2] + r * np.sin(a)])
            st.append(s)
    return np.array(pts), np.array(st)


STATIONS = np.round(np.arange(-0.08, 0.10 + 1e-9, 0.005), 4)


def sleeve_and_hand(s):
    """A forearm narrowing to the wrist at 0, the palm widening past it, the fingers narrowing again."""
    if s < 0:
        return 0.035 - 0.08 * s * 0.1          # 3.5 cm at the wrist, wider up the arm
    if s < 0.08:
        return 0.035 + 0.25 * s                # the palm widens
    return 0.055 - 0.2 * (s - 0.08)


def cuffed(s):
    """A wide cuff from 2 cm before the wrist to 3 cm past it, the hand coming out of it narrower."""
    if -0.02 <= s <= 0.03:
        return 0.06
    return sleeve_and_hand(s - 0.035 if s > 0.03 else s)


class CutChoice(unittest.TestCase):
    def profile(self, prof, offset=(0, 0, 0)):
        P, st = tube(prof, offset)
        return SH.radius_profile(P, st, AX, STATIONS)

    def test_plain_wrist_is_cut_at_the_wrist(self):
        cut, why = SH.choose_cut(STATIONS, self.profile(sleeve_and_hand))
        self.assertIsNotNone(cut, why)
        self.assertLessEqual(abs(cut), 0.005, why)

    def test_wide_cuff_over_the_wrist_survives(self):
        cut, why = SH.choose_cut(STATIONS, self.profile(cuffed))
        self.assertIsNotNone(cut, why)
        self.assertGreater(cut, 0.03, f"the cut must land past the cuff's end (0.03): {why}")
        self.assertLessEqual(cut, SH.CUT_HI_M)

    def test_profile_is_round_the_section_not_the_axis(self):
        # the same sleeve 4 cm off the arm's axis line measures the same radii
        a = self.profile(sleeve_and_hand)
        b = self.profile(sleeve_and_hand, offset=(0, 0.04, -0.02))
        ok = np.isfinite(a) & np.isfinite(b)
        self.assertTrue(ok.sum() > 20)
        self.assertLess(float(np.max(np.abs(a[ok] - b[ok]))), 1e-3)

    def test_ties_go_to_the_section_nearest_the_wrist(self):
        r = np.full(len(STATIONS), 0.05)
        r[(STATIONS >= -0.02) & (STATIONS <= 0.04)] = 0.03
        cut, _ = SH.choose_cut(STATIONS, r)
        self.assertEqual(cut, 0.0)

    def test_no_profile_no_cut(self):
        cut, why = SH.choose_cut(STATIONS, np.full(len(STATIONS), np.nan))
        self.assertIsNone(cut)
        self.assertIn("no forearm profile", why)

    def test_cut_never_outside_the_window(self):
        r = np.linspace(0.02, 0.06, len(STATIONS))      # narrowest far up the arm
        cut, _ = SH.choose_cut(STATIONS, r)
        self.assertGreaterEqual(cut, SH.CUT_LO_M - 1e-9)


SKIN = np.array([0.60, 0.45, 0.36])


def noisy(c, n=200, sd=0.02, seed=0):
    return np.clip(np.asarray(c) + np.random.default_rng(seed).normal(0, sd, (n, 3)), 0, 1)


def is_skin(c):          # bl_combo.is_skin
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    return (r > g) & (g > b) & (r - b > 0.10) & (r > 0.25)


class Covering(unittest.TestCase):
    def test_bare_hand_takes_its_skin(self):
        h = noisy(SKIN * 0.92)                         # hands a little darker than the face
        h[:10] = [0.9, 0.9, 0.9]                       # a white cuff's edge caught in the cut
        c = SH.hand_colour(h, is_skin(h), noisy(SKIN, seed=1))
        self.assertFalse(c["covered"], c["why"])
        self.assertLess(np.linalg.norm(np.array(c["rgb"]) - SKIN * 0.92), 0.03)

    def test_grey_gauntlet_is_a_covering(self):
        h = noisy([0.42, 0.43, 0.45])
        c = SH.hand_colour(h, is_skin(h), noisy(SKIN, seed=1))
        self.assertTrue(c["covered"], c["why"])
        self.assertLess(np.linalg.norm(np.array(c["rgb"]) - [0.42, 0.43, 0.45]), 0.03)

    def test_dark_glove_far_from_the_face_is_a_covering(self):
        h = noisy([0.30, 0.18, 0.10])                  # brown leather passes the skin-colour test
        self.assertGreater(is_skin(h).mean(), 0.9)
        c = SH.hand_colour(h, is_skin(h), noisy(SKIN, seed=1))
        self.assertTrue(c["covered"], c["why"])

    def test_without_a_face_skin_share_decides(self):
        h = noisy(SKIN)
        self.assertFalse(SH.hand_colour(h, is_skin(h), None)["covered"])
        g = noisy([0.2, 0.2, 0.25])
        self.assertTrue(SH.hand_colour(g, is_skin(g), None)["covered"])


class Fallback(unittest.TestCase):
    def test_fitted(self):
        self.assertEqual(SH.fit_decision(True, 0.0, 800, 5000, 1.1, 20.0), (True, "fitted"))

    def test_reasons(self):
        cases = [((False, 0.0, 800, 5000, 1.1, 20.0), "off in the job"),
                 ((True, None, 800, 5000, 1.1, 20.0, "no forearm profile at the wrist"), "no forearm profile"),
                 ((True, 0.0, 10, 5000, 1.1, 20.0), "no statue hand"),
                 ((True, 0.0, 800, 0, None, 20.0), "body has no hand"),
                 ((True, 0.0, 800, 5000, 3.0, 20.0), "length"),
                 ((True, 0.0, 800, 5000, None, 20.0), "length"),
                 ((True, 0.0, 800, 5000, 1.1, 120.0), "points")]
        for args, word in cases:
            ok, why = SH.fit_decision(*args)
            self.assertFalse(ok, args)
            self.assertIn(word, why, args)

    def test_angle(self):
        self.assertAlmostEqual(SH.angle_deg([1, 0, 0], [0, 1, 0]), 90.0)
        self.assertEqual(SH.angle_deg([0, 0, 0], [1, 0, 0]), 180.0)


class ComboWiring(unittest.TestCase):
    SRC = (IMG3D / "bl_combo.py").read_text()

    def test_default_on_for_statues_behind_the_job(self):
        self.assertIn('JOB.get("statue_body_hands", True)', self.SRC)

    def test_hands_in_the_statue_slot_not_a_skin_slot(self):
        # the engine (char_render import_outfit / skin_slots) paints any slot named "skin" with the MetaHuman's own
        # skin material; the statue's hands must wear the statue's colour, so they join its one material
        self.assertIn("hands.data.materials.append(mo)", self.SRC)
        self.assertIn("free_swatches(fig, img, [JOINTED[s_][\"rgb\"]", self.SRC)

    def test_rigid_fallback_kept(self):
        self.assertIn('rig[f"hand_{side}"] = np.nonzero(old_rigid)[0]', self.SRC)


if __name__ == "__main__":
    unittest.main()
