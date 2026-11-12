"""The arms measure behind the character build's arm rules (characters/arm_gap.py), on synthetic guides: is each
forearm separated from the body (air in the person's mask, or a jump in depth), row by row, and is the lower half
(toward the wrist) separated: the measure the loose-sleeve rule reads. No engine, no guides model: the mask is a
small picture, the depth maps are arrays handed to the reader, the camera is a flat stand-in."""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

CHARS = Path(__file__).resolve().parents[1] / "assets" / "characters"
sys.path.insert(0, str(CHARS))
import arm_gap as AG  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="char_arm_test_"))
W = H = 240


class FlatCamera:
    """x across (-1.2 .. 1.2 m), z up (0 .. 2.4 m), everything 5 m away"""

    @staticmethod
    def project(cam, p):
        return (p[0] + 1.2) / 2.4, 1.0 - p[2] / 2.4, 5.0


def px(x, z):
    return int((x + 1.2) / 2.4 * W), int((1.0 - z / 2.4) * H)


J = {"thigh_l": [0.09, 0, 0.9], "thigh_r": [-0.09, 0, 0.9], "neck_01": [0, 0, 1.5],
     "upperarm_l": [0.2, 0, 1.45], "lowerarm_l": [0.42, 0, 1.30], "hand_l": [0.50, 0, 1.00],
     "upperarm_r": [-0.2, 0, 1.45], "lowerarm_r": [-0.42, 0, 1.30], "hand_r": [-0.50, 0, 1.00]}


def scene(joined_above=None, arm_depth=500.0, fill=False, hands=True):
    """A person seen from the front: a torso and two forearms hanging out to the sides.
    joined_above: cloth joins each forearm to the body above this height (None: air all the way).
    fill: the mask is one block (arm against the body); arm_depth: the forearms' depth in cm (the body's is 500)."""
    m = np.zeros((H, W), bool)
    dep = np.full((H, W), 500.0, np.float32)

    def box(x0, x1, z0, z1, val=True, depth=None):
        (c0, r1), (c1, r0) = px(x0, z0), px(x1, z1)
        m[r0:r1 + 1, c0:c1 + 1] = val
        if depth is not None:
            dep[r0:r1 + 1, c0:c1 + 1] = depth
    box(-0.17, 0.17, 0.85, 1.55)                                     # the body
    for s in (1, -1):
        lo, hi = sorted((s * 0.36, s * 0.56))
        box(lo, hi, 0.95 if hands else 1.12, 1.36, depth=arm_depth)  # a forearm (and the hand below the wrist)
        if fill or joined_above is not None:
            a, b = sorted((s * 0.17, s * 0.36))
            box(a, b, 0.95 if fill else joined_above, 1.36)
    d = Path(tempfile.mkdtemp(dir=TMP))
    (d / "guides").mkdir()
    Image.fromarray((m * 255).astype(np.uint8)).save(d / "guides" / "wide_mask_a.png")
    (d / "result.json").write_text(json.dumps({"subjects": {"a": {"measured": {"joints": J}}},
                                               "shots": [{"name": "wide", "camera": {}}]}))
    return d, dep


def measure(**kw):
    d, dep = scene(**kw)
    old_exr, old_fr = AG.read_exr, getattr(AG, "framing", None)
    AG.read_exr = lambda p: dep
    AG.framing = FlatCamera
    try:
        return AG.measure(d, "a")["wide"]
    finally:
        AG.read_exr = old_exr
        AG.framing = old_fr


class ForearmSeparation(unittest.TestCase):
    def test_air_between_forearm_and_body_separates_every_row(self):
        r = measure()
        for s in ("l", "r"):
            self.assertEqual(r[s]["rows"], 10)
            self.assertEqual(r[s]["separated"], 10)
            self.assertEqual(r[s]["mask_gap"], 10)
            self.assertEqual(r[s]["lower_rows"], 5)
            self.assertEqual(r[s]["lower_separated"], 5)

    def test_a_forearm_against_the_body_at_the_same_depth_is_not_separated(self):
        r = measure(fill=True)
        for s in ("l", "r"):
            self.assertEqual(r[s]["separated"], 0)
            self.assertEqual(r[s]["lower_separated"], 0)

    def test_a_forearm_in_front_of_the_body_is_separated_by_depth(self):
        r = measure(fill=True, arm_depth=470.0)          # 30 cm nearer: no air in the mask, a jump in depth
        for s in ("l", "r"):
            self.assertEqual(r[s]["mask_gap"], 0)
            self.assertEqual(r[s]["separated"], 10)

    def test_a_small_depth_difference_is_not_a_separation(self):
        r = measure(fill=True, arm_depth=498.0)
        self.assertEqual(r["l"]["separated"], 0)

    def test_a_sleeve_joined_to_the_body_down_to_the_elbow_leaves_the_lower_forearm_clear(self):
        # the case of the loose-sleeve rule: the upper rows are joined by cloth, the lower forearm and hand are free
        r = measure(joined_above=1.12)
        for s in ("l", "r"):
            self.assertLess(r[s]["separated"], 8)                # the general rule (8 of 10 rows) would fail it
            self.assertGreaterEqual(r[s]["lower_separated"], 4)  # the loose-sleeve rule (4 of the lower 5) passes it
            self.assertEqual(r[s]["lower_rows"], 5)

    def test_the_hand_is_seen_when_the_person_is_there_at_its_depth(self):
        self.assertTrue(measure()["l"]["hand_visible"])
        self.assertFalse(measure(hands=False)["l"]["hand_visible"])
        self.assertFalse(measure(arm_depth=560.0)["l"]["hand_visible"])     # something else is at the palm's place
        self.assertIsNone(measure()["l_upper"]["hand_visible"])

    def test_the_upper_arm_is_measured_too(self):
        r = measure()
        for k in ("l_upper", "r_upper"):
            self.assertIn("separated", r[k])
            self.assertGreater(r[k]["rows"], 0)


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
