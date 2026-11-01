"""11-01: a creature's look in the keys loses its stance, never its wings or tail."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mannequin_flow as MF


def test_wings_stay():
    d = ("a massive dragon with scales like black iron, glowing golden eyes, smoke curling from its nostrils, "
         "reptilian face, no clothing, standing on four legs, tail extended, wings folded")
    a = MF._appearance(d)
    assert a == ("a four-legged massive dragon with scales like black iron, glowing golden eyes, smoke curling from "
                 "its nostrils, reptilian face, no clothing, a long tail, great wings.")


def test_person_unchanged():
    assert MF._appearance("a man of about 50, grey beard, standing tall") == "a man of about 50, grey beard."
