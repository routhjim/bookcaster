"""11-27: in a full or wide shot of several named people each of them is whole from side to side with the head in
the frame (scenes/speakers.py framing_why). One of four stood half out of a wide's edge: the picture drawn from it
had him headless and the shot got no key. Made-up scene only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import speakers as SP  # noqa: E402

SUBJ = {"a": S.person([-1.2, 2.0], facing=[0, -1]), "b": S.person([0.0, 2.0], facing=[0, -1]),
        "c": S.person([1.2, 2.0], facing=[0, -1])}
CUT = "(named for the shot) is cut by the frame's"


def whys(cam, kind="wide", show=("a", "b", "c")):
    return SP.framing_why(cam, kind, None, None, False, list(show), SUBJ, S.RES, opts={"seated": set()})[0]


def test_all_three_whole_in_the_frame_pass_and_one_on_the_edge_fails():
    assert not [w for w in whys(S.cam([0.0, -4.0, 1.5], [0.0, 2.0, 1.2], 24)) if CUT in w]
    # the same camera turned until the left one's middle is on the frame's left edge
    cam = S.cam([0.0, -4.0, 1.5], [0.0, 2.0, 1.2], 24)
    for dx in (x / 10 for x in range(1, 40)):
        cam = S.cam([0.0, -4.0, 1.5], [dx, 2.0, 1.2], 24)
        f = SP.frame_facts(cam, SUBJ["a"]["measured"], S.RES)
        if f["any_in"] and "left" in f["cut_edges"]:
            break
    else:
        raise AssertionError("no camera cut the left person")
    got = [w for w in whys(cam) if CUT in w]
    assert got and got[0].startswith("a ") and "left" in got[0]


def test_a_medium_or_a_shot_naming_one_person_is_not_asked_this():
    cam = S.cam([0.0, -4.0, 1.5], [1.6, 2.0, 1.2], 24)
    assert not [w for w in whys(cam, kind="medium") if CUT in w]
    assert not [w for w in whys(cam, show=("b",)) if CUT in w]
