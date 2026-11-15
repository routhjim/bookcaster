"""Scene tool camera framing (tools/assets/scenes/framing.py): shots computed from a measured subject land the eye
line, head and waist where the shot kind says, and the frame check catches a cut head or a subject facing away."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
import framing  # noqa: E402

SUBJ = {"feet_z": 0.0, "eye_z": 1.76, "head_top_z": 1.89, "chin_z": 1.635, "waist_z": 1.02,
        "eye_xy": [0.35, 0.82], "chest_xy": [0.35, 0.72], "pelvis_xy": [0.35, 0.7], "facing": [0.0, -1.0]}


class FramingTest(unittest.TestCase):
    def test_close_puts_eyes_on_upper_third(self):
        cam = framing.shot(SUBJ, "close", 85)
        chk = framing.check(cam, SUBJ, "close")
        self.assertAlmostEqual(chk["eyes"][1], framing.CLOSE_EYE_LINE, delta=0.02)
        self.assertGreater(chk["head_top"][1], 0.0)
        self.assertEqual(chk["problems"], [])
        self.assertLess(cam["position"][1], SUBJ["eye_xy"][1])        # in front of a subject facing -Y

    def test_medium_keeps_waist_in_and_head_whole(self):
        cam = framing.shot(SUBJ, "medium", 50, az=20)
        chk = framing.check(cam, SUBJ, "medium")
        self.assertLess(chk["waist"][1], 1.0)
        self.assertGreater(chk["waist"][1], 0.85)
        self.assertEqual(chk["problems"], [])
        self.assertAlmostEqual(chk["camera_off_facing_deg"], 20, delta=3)

    def test_distance_follows_lens(self):
        a = framing.shot(SUBJ, "close", 50)["distance_m"]
        b = framing.shot(SUBJ, "close", 100)["distance_m"]
        self.assertAlmostEqual(b / a, 2.0, places=2)

    def test_screen_x_moves_subject(self):
        cam = framing.shot(SUBJ, "close", 85, screen_x=0.4)
        self.assertAlmostEqual(framing.check(cam, SUBJ, "close")["eyes"][0], 0.4, delta=0.01)

    def test_check_flags_back_view_and_cut_head(self):
        cam = framing.shot(SUBJ, "close", 85, az=180)
        self.assertTrue(any("faces away" in p for p in framing.check(cam, SUBJ, "close")["problems"]))
        low = dict(framing.shot(SUBJ, "close", 85))
        low["target"] = [low["target"][0], low["target"][1], low["target"][2] - 0.4]
        low["position"] = [low["position"][0], low["position"][1], low["position"][2] - 0.4]
        self.assertIn("head top cut", framing.check(low, SUBJ, "close")["problems"])

    def test_wide_checks_whole_figure(self):
        cam = {"position": [1.2, -2.45, 1.55], "target": [-0.35, 2.0, 1.2], "focal_mm": 22}
        chk = framing.check(cam, SUBJ, "wide")
        self.assertEqual(chk["problems"], [])
        self.assertGreater(chk["figure_h_frac"], 0.3)


if __name__ == "__main__":
    unittest.main()
