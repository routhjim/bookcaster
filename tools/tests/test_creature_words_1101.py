"""11-01: a creature's posed training pictures are drawn from words (no reference), and its stance is not its look."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import charlora as C


def test_stance_words():
    d = "a massive dragon with scales like black iron, glowing golden eyes, standing on four legs, tail extended, wings folded"
    kept = ", ".join(x for x in d.split(", ") if not C.STANCE.search(x))
    assert kept == "a massive dragon with scales like black iron, glowing golden eyes"
    assert not C.STANCE.search("all wearing plain off-white linen tunics and dark sashes")
