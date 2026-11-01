"""11-01: a creature's posed training pictures are drawn from words, and its stance is not its look (its wings are)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import charlora as C


def test_stance_words():
    d = "a massive dragon with scales like black iron, glowing golden eyes, standing on four legs, tail extended, wings folded"
    assert C.drop_stance(d) == "a massive dragon with scales like black iron, glowing golden eyes, a long tail, great wings"
    a = "a group of twelve men, all wearing plain off-white linen tunics and dark sashes"
    assert C.drop_stance(a) == a
