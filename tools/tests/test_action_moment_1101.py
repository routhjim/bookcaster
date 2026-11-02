"""11-01: an action shot's key is its MOMENT (the action under way), people by look only, no layout or body-plan words;
other shots keep the layout. And a lone adjective is not a garment ("the man in the dull")."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as F
import registry as RG

BOOK = Path.home() / "audiobooks" / "knight_forged_v3"
SHOT = {"id": "action-05m16s-003162_s0", "kind": "action", "framing": "close", "place": "The Burnt Mountain",
        "who": ["Vorgrath", "Edric"], "pose": {"Vorgrath": "standing", "Edric": "standing"},
        "picture": "Vorgrath stands before the cave on the left, facing right; Edric stands on the right, holding sword.",
        "moment": "Vorgrath's spiked tail sweeps in low from the left as Edric throws himself into a roll beneath it, "
                  "sword in hand.", "action": "Vorgrath's tail sweeps over Edric, who rolls clear."}


def test_moment_only_for_action():
    assert F.moment(SHOT).startswith("Vorgrath's spiked tail")
    assert F.moment(dict(SHOT, kind="quiet")) == ""
    assert F.moment(dict(SHOT, moment="")) == ""


def test_key_prompt_uses_moment():
    if not BOOK.exists():
        return
    p = F._key_prompt(BOOK, "01_chapter_1", "action-05m16s-003162", SHOT)
    assert "tail sweeps in low" in p and "stands before the cave" not in p
    assert "on the left" not in p.split("tail sweeps")[1] and "never upright" not in p
    q = F._key_prompt(BOOK, "01_chapter_1", "action-05m16s-003162", dict(SHOT, moment=""))
    assert "stands before the cave" in q


def test_describe_lone_adjective():
    if not BOOK.exists():
        return
    d = {"Edric": "Dull, unpolished star-iron armour consisting of a breastplate, worn over a jerkin"}
    assert F.describe(RG.Book(BOOK), "Edric", d) == "the man in the dull unpolished star-iron armour"
    assert F.describe(RG.Book(BOOK), "Princess Maren").endswith("stained with ash")




def test_fill_moments():
    import json as _j
    reply = _j.dumps({"shots": [{"kind": "action", "action": "Vorgrath's tail sweeps over Edric.",
                                 "people": [{"name": "Vorgrath"}, {"name": "Edric"}]},
                                {"kind": "action", "action": "x", "moment": "already here"},
                                {"kind": "quiet", "action": "Edric breathes."}]})
    asked = []
    r, n = F._fill_moments(reply, lambda q: asked.append(q) or "Vorgrath's tail sweeps in low as Edric dives into a roll.")
    assert n == 1 and len(asked) == 1 and "Vorgrath, Edric" in asked[0]
    sh = _j.loads(r)["shots"]
    assert sh[0]["moment"].startswith("Vorgrath's tail") and sh[1]["moment"] == "already here" and "moment" not in sh[2]




def test_write_keys():
    """one question for the kind, then the prompt for that kind, seeded by the moment; a shot with a keyshot is kept"""
    import json as _j
    reply = _j.dumps({"shots": [{"kind": "action", "place": "The Burnt Mountain", "action": "Vorgrath's tail sweeps over Edric.",
                                 "framing": "close", "moment": "Vorgrath's tail sweeps in low as Edric dives into a roll.",
                                 "people": [{"name": "Vorgrath", "pose": "standing"}, {"name": "Edric", "pose": "standing", "at": "downstage-centre"}]},
                                {"kind": "action", "action": "Hulda sits.", "moment": "Hulda sits down on the stool.", "keyshot": "kept"},
                                {"kind": "talk", "speaker": "Edric"}]})
    asked = []
    def ask(q):
        asked.append(q)
        return "fight" if len(asked) == 1 else "Vorgrath the dragon's spiked tail sweeps low over Edric as he throws himself into a roll."
    r, n = F._write_keys(reply, ask, lambda w: "a dragon" if w == "Vorgrath" else "a man")
    assert n == 1 and len(asked) == 2 and "fight, craft or other" in asked[0]
    assert "Vorgrath (a dragon), standing" in asked[1] and "Both bodies" in asked[1] and "Vorgrath's tail sweeps in low" in asked[1]
    sh = _j.loads(r)["shots"]
    assert sh[0]["keyshot"].startswith("Close shot. Vorgrath the dragon") and sh[0]["key_kind"] == "fight"
    assert sh[1]["keyshot"] == "kept" and "keyshot" not in sh[2]
    asked.clear()
    craft = _j.dumps({"shots": [{"kind": "action", "action": "Edric hammers the bar.", "moment": "Edric raises the hammer over the bar.",
                                 "people": [{"name": "Edric"}, {"name": "Hulda", "pose": "sitting", "at": "on a stool"}]}]})
    F._write_keys(craft, lambda q: asked.append(q) or ("craft" if len(asked) == 1 else "Edric raises the hammer over the glowing bar; Hulda sits on her stool, watching."), lambda w: "a man")
    assert "sitting stays sitting" in asked[1] and "Hulda (a man), sitting on a stool" in asked[1]
    s = {"kind": "action", "keyshot": "Close shot. X.", "moment": "m"}
    assert F.moment(s) == "Close shot. X"


if __name__ == "__main__":
    test_moment_only_for_action(); test_key_prompt_uses_moment(); test_describe_lone_adjective(); test_fill_moments(); test_write_keys(); print("ok")
