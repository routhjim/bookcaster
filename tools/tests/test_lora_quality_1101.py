"""LoRA training-set audit fixes (11-01): captions from what was seen, face-covering outfits, pose filtering."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import charlora as C

H = SimpleNamespace(kind="human")
D = SimpleNamespace(kind="creature")


def test_caption_from_what_was_seen():
    cd = {"src": "pose", "framing": "close-up portrait, head and shoulders, profile", "pose": "kneeling on one knee"}
    r = {"feet": True, "head_frac": 0.12, "view": "front", "pose_ok": False}
    s = C._seen(H, cd, r)
    assert s.startswith("full body") and "facing the viewer" in s and "kneeling" not in s and "profile" not in s
    r = {"feet": False, "head_frac": 0.4, "view": "side", "pose_ok": True}
    assert C._seen(H, cd, r) == "close-up portrait, head and shoulders, seen in profile, kneeling on one knee"
    assert C._seen(D, {"src": "pose"}, {"feet": True, "view": "away"}).startswith("the whole body from head to tail")


def test_sheet_and_old_judgements_keep_their_words():
    assert C._seen(H, {"src": "sheet", "framing": "full body, standing"}, {"feet": True}) == "full body, standing"
    assert C._seen(H, {"src": "book"}, {}) == "a scene from the story"


def test_face_covering_outfits():
    assert C.covers_face("star-iron armour, a breastplate, a helm with a narrow eye slit")
    assert not C.covers_face("star-iron armour with the helm removed")
    assert C.covers_face("face covered by a thick black veil")
    assert not C.covers_face("a worn brown leather jerkin")


def test_pose_props_filtered_by_words():
    assert {"anvil", "cup", "blade", "tongs"} <= set(C.PROPS)
    assert "sitting_chair" in C.SEAT_POSES and "dancing" in C.POSE_SKIP
