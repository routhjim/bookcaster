"""A whole figure's head weights (img3d/statue_head.py, used by bl_combo.py in statue mode), without Blender:

  - the head is the head, not the collar line: ear lobes, jaw corners and nape hair under the old tilted collar line
    are rigid on the head bone (they had the shoulders' clavicle and upper-arm weights, and the ears stood out as
    points when the arms came down from the spread bind pose);
  - big hair standing well off the head is head too, above the line; the shoulders' cloth is never head;
  - the neck under the line blends from the head to the body's own weights, smoothly, without arm weights;
  - everything the collar line gave the head stays head;
  - bl_combo keeps it behind the job's data, on by default for statues.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

IMG3D = Path(__file__).resolve().parents[1] / "assets" / "img3d"
sys.path.insert(0, str(IMG3D))
import statue_head as SH  # noqa: E402

# a standing figure's neck (metres, z up, the face toward -y): neck_01, the neck's upper joint, the head joint
NECK_LO = np.array([0.0, 0.024, 1.541])
NECK_TOP_Z = 1.587
HEAD_J = np.array([0.0, 0.004, 1.635])


def collar_line(P, cut_m=0.03, tilt_deg=20.0):
    """bl_combo's old head region: above a plane through neck_01's y at the upper joint + cut, higher at the back."""
    t = np.radians(tilt_deg)
    return (P[:, 2] - (NECK_TOP_Z + cut_m)) * np.cos(t) - (P[:, 1] - NECK_LO[1]) * np.sin(t) > 0


def follow(P):
    P = np.asarray(P, float)
    return SH.head_follow(P, NECK_LO, HEAD_J, NECK_TOP_Z, rigid=collar_line(P))


class HeadRegion(unittest.TestCase):
    def test_ear_lobe_under_the_collar_line_is_head(self):
        # an ear: 8 cm to the side, 4 cm behind the neck's axis, its lobe 2 cm over the neck's upper joint
        ear = np.array([[0.08, 0.04, NECK_TOP_Z + 0.02], [0.085, 0.045, NECK_TOP_Z + 0.05]])
        self.assertFalse(collar_line(ear)[0])          # the old rule left the lobe to the shoulders
        np.testing.assert_allclose(follow(ear), 1.0)

    def test_nape_hair_and_big_hair_are_head(self):
        nape = np.array([[0.0, 0.10, NECK_TOP_Z + 0.015]])
        mop = np.array([[-0.06, 0.125, HEAD_J[2] + 0.02]])   # hair standing ~14 cm behind the head joint
        self.assertEqual(float(follow(nape)[0]), 1.0)
        self.assertEqual(float(follow(mop)[0]), 1.0)

    def test_shoulders_are_never_head(self):
        sh = np.array([[0.18, 0.02, NECK_TOP_Z - 0.03], [0.16, 0.0, NECK_TOP_Z - 0.10], [0.25, 0.0, NECK_TOP_Z + 0.02]])
        np.testing.assert_allclose(follow(sh), 0.0)

    def test_neck_blends_smoothly_down_to_the_body(self):
        z = np.linspace(NECK_TOP_Z - 0.08, NECK_TOP_Z + 0.03, 45)
        P = np.c_[np.full_like(z, 0.05), np.full_like(z, 0.03), z]      # the side of the neck
        w = follow(P)
        self.assertTrue(np.all(np.diff(w) >= -1e-9))                    # never less head higher up
        self.assertEqual(float(w[0]), 0.0)
        self.assertEqual(float(w[-1]), 1.0)
        self.assertLess(float(np.max(np.diff(w))), 0.12)               # no step: a blend over ~5 cm
        mid = (z > NECK_TOP_Z + SH.HEAD_LINE_M - SH.BLEND_M + 0.005) & (z < NECK_TOP_Z + SH.HEAD_LINE_M - 0.005)
        self.assertTrue(np.all((w[mid] > 0) & (w[mid] < 1)))

    def test_collar_line_head_stays_head(self):
        rng = np.random.default_rng(3)
        P = np.c_[rng.uniform(-0.2, 0.2, 4000), rng.uniform(-0.2, 0.2, 4000), rng.uniform(1.5, 1.8, 4000)]
        old = collar_line(P)
        self.assertTrue(np.all(follow(P)[old] == 1.0))

    def test_tilted_neck_axis(self):
        # the distance is measured from the neck's own (forward-leaning) axis at each height
        d = SH.axis_distance(np.array([[0.0, HEAD_J[1], HEAD_J[2]], [0.0, NECK_LO[1], NECK_LO[2]]]), NECK_LO, HEAD_J)
        np.testing.assert_allclose(d, 0.0, atol=1e-9)


class BlendWeights(unittest.TestCase):
    def test_arms_dropped_and_head_share_added(self):
        g = {"spine_04": 0.4, "clavicle_r": 0.3, "clavicle_scap_r": 0.2, "upperarm_out_r": 0.1}
        out = SH.blend_weights(g, 0.25)
        self.assertAlmostEqual(sum(out.values()), 1.0)
        self.assertFalse(any(SH.is_arm(k) for k in out))
        self.assertAlmostEqual(out["head"], 0.25)
        self.assertAlmostEqual(out["spine_04"], 0.75)

    def test_only_arm_weights_fall_back_to_the_neck(self):
        out = SH.blend_weights({"clavicle_l": 0.6, "upperarm_bck_l": 0.4}, 0.5)
        self.assertEqual(set(out), {"head", SH.NECK_BONE})
        self.assertAlmostEqual(out["head"], 0.5)

    def test_rigid_and_influence_cap(self):
        self.assertEqual(SH.blend_weights({"spine_04": 1.0}, 1.0), {"head": 1.0})
        g = {"spine_03": 0.3, "spine_04": 0.3, "spine_05": 0.2, "neck_01": 0.15, "pelvis": 0.05}
        out = SH.blend_weights(g, 0.4)
        self.assertLessEqual(len(out), SH.MAX_INFLUENCES)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_toes_are_not_arms(self):
        self.assertFalse(SH.is_arm("index_01_l_toe"))   # (a foot's toes share the fingers' names)
        self.assertTrue(SH.is_arm("clavicle_scap_l"))


def shell(c, half, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    return np.asarray(c) + u * np.asarray(half)


class WornThing(unittest.TestCase):
    """A visible head wider, deeper and set back from the hidden one (a drawn mop of hair) and how a helm sized for
    the hidden head must change to hold it."""
    HJ = np.array([0.0, 0.0, 1.63])

    def box(self):
        hidden = shell(self.HJ + [0, -0.03, 0.06], [0.095, 0.10, 0.10])
        statue = shell(self.HJ + [0, 0.01, 0.05], [0.11, 0.13, 0.09], seed=1)
        return SH.head_box(statue, hidden, self.HJ, self.HJ[2] - 0.03)

    def test_box_measures_the_difference(self):
        b = self.box()
        self.assertGreater(b["width_ratio"], 1.1)
        self.assertGreater(b["depth_ratio"], 1.2)
        self.assertLess(b["top_diff_m"], 0.0)                     # the visible head is lower
        self.assertGreater(b["centre_offset_m"][1], 0.03)          # ... and set back

    def test_fit_holds_the_visible_head(self):
        b = self.box()
        f = SH.worn_fit(b, ease_m=0.01)
        dw = (b["statue"]["hi"][1] - b["statue"]["lo"][1] + 0.02) / (b["hidden"]["hi"][1] - b["hidden"]["lo"][1])
        self.assertAlmostEqual(f["scale"], round(dw, 3), places=3)  # the depth decides here
        self.assertEqual(f["up_m"], 0.0)                           # never lowered into the head
        self.assertGreater(f["shift_m"][1], 0.03)

    def test_no_box_no_change_and_never_smaller(self):
        self.assertEqual(SH.worn_fit(None), {"scale": 1.0, "shift_m": [0.0, 0.0], "up_m": 0.0})
        small = SH.head_box(shell(self.HJ, [0.05, 0.05, 0.05]), shell(self.HJ, [0.1, 0.1, 0.1], seed=2), self.HJ,
                            self.HJ[2] - 0.2)
        self.assertEqual(SH.worn_fit(small)["scale"], 1.0)
        self.assertIsNone(SH.head_box(np.zeros((0, 3)), shell(self.HJ, [0.1] * 3), self.HJ, 0.0))


class ComboWiring(unittest.TestCase):
    SRC = (IMG3D / "bl_combo.py").read_text()

    def test_default_on_for_statues_behind_the_job(self):
        self.assertIn('JOB.get("statue_head_follow", True)', self.SRC)
        self.assertIn("SHD.head_follow(", self.SRC)
        self.assertIn("SHD.blend_weights(", self.SRC)
        self.assertIn('REP["statue_head"]', self.SRC)


if __name__ == "__main__":
    unittest.main()
