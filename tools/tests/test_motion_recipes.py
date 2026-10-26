"""Generic motion recipes: matching by action + affordances, and no story names in the motion tools (no GPU).

    ~/tts-ab/.venv-pose/bin/python -I tools/tests/test_motion_recipes.py
"""
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import motion_batch as MB  # noqa: E402

CASES = [  # (figure's own sentences, holds, attached, expected recipe)
    (["Mira brings the hammer down onto the horseshoe."], {"right": "hammer"}, False, "strike_face"),
    (["Tomas strikes the glowing blade on the anvil."], {"right": "hammer", "left": "blank"}, False, "strike_face_on_flat_work"),
    (["She holds the hammer steady in the air, preparing to bring it down."], {"right": "hammer"}, False, "strike_tool_held_raised"),
    (["The guard thrusts his spear forward."], {"right": "sword"}, False, "thrust_with_tip"),
    (["He raises his sword to block the falling axe."], {"right": "sword"}, False, "overhead_block_with_edge"),
    (["She holds the blade steady above the water."], {"right": "sword"}, False, "quench_point_down"),
    (["The old knight looks at the sword."], {"right": "sword"}, False, "inspect_blade"),
    (["The boy dangles below the giant bird."], {}, True, "carried_by_creature"),
    (["He brings the hammer down."], {}, False, None),                 # no striking prop held: no strike recipe
    (["She thrusts her hand into the bag."], {"right": "lantern"}, False, None),   # lantern has no tip
    (["He walks to the door."], {"right": "sword"}, False, None),
]
STORY_WORDS = r"\b(vorgrath|edric|maren|hulda|knight_forged|sherlock|watson)\b"


def test_recipes():
    bad = []
    for sents, holds, att, want in CASES:
        r = MB.match_recipe(sents, holds, attached=att)
        got = r["name"] if r else None
        if got != want:
            bad.append((sents[0], holds, want, got))
    return bad


def test_no_story_names():
    hits = []
    for f in glob.glob(os.path.join(HERE, "motion_*.py")) + [os.path.join(HERE, "stage_templates", "motion_recipes.json")]:
        for i, ln in enumerate(open(f), 1):
            if re.search(STORY_WORDS, ln, re.I):
                hits.append(f"{os.path.basename(f)}:{i}: {ln.strip()[:80]}")
    return hits

def test_ground_pass_lifts_appendages_keeps_lengths():
    import numpy as np
    from motion_composite import ground_pass, _on_ground
    q = _on_ground(np.array([0, 0, 1.0]), np.array([0, 0, -2.0]), 3.0)   # straight down through the floor
    assert abs(np.linalg.norm(q - [0, 0, 1.0]) - 3.0) < 1e-6 and q[2] >= 0.05 - 1e-9
    par = {"a": "root", "b": "a"}
    seq = [{"root": np.array([0, 0, 1.0]), "a": np.array([1.0, 0, -1.0]), "b": np.array([2.0, 0, -3.0])}]
    out, n, mz = ground_pass(seq, par)
    J = out[0]
    assert n == 2 and mz >= 0.05 - 1e-6
    for c, p in par.items():
        assert abs(np.linalg.norm(J[c] - J[p]) - np.linalg.norm(seq[0][c] - seq[0][p])) < 1e-6

def test_camera_occlusion_and_own_head():
    import numpy as np
    from motion_camera import seg_hits
    p = np.array([0.0, 0, 1.6])
    K = np.array([[10.0, 0, 1.6]])
    C = np.array([[5.0, 0, 1.6]]); R = np.array([0.5])
    assert seg_hits(p, K, C, R, np.zeros((0, 3, 3)))[0]                       # a sphere in the way
    assert not seg_hits(p, K, C + [0, 2, 0], R, np.zeros((0, 3, 3)))[0]        # beside the ray
    T = np.array([[[5.0, -1, 0], [5.0, 1, 0], [5.0, 0, 3]]])                    # a wing membrane across the ray
    assert seg_hits(p, K, np.zeros((0, 3)), np.zeros(0), T)[0]
    own = np.array([[10.0, 0, 1.6], [9.6, 0, 1.6]]); orr = np.array([0.3, 0.3])  # the head's own skull / snout
    assert not seg_hits(p, K, own, orr, np.zeros((0, 3, 3)), own=True, head_r=0.6)[0]



if __name__ == "__main__":
    b = test_recipes()
    h = test_no_story_names()
    for x in b:
        print("RECIPE FAIL", x)
    for x in h:
        print("STORY NAME", x)
    test_ground_pass_lifts_appendages_keeps_lengths()
    print("ground pass ok")
    test_camera_occlusion_and_own_head()
    print("camera occlusion ok")
    print(f"recipes {len(CASES) - len(b)}/{len(CASES)} ok; story names in motion tools: {len(h)}")
    sys.exit(1 if b or h else 0)
