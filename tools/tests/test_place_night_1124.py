"""11-24: a NEW picture of a place whose first scene opens at night is drawn at night; a picture already drawn stays
whichever hour its words had (no redraw, so no re-plan of the shots there)."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import mannequin_flow as MF

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = book / "work" / "video"
    (vd / "01_one" / "shots" / "scenes").mkdir(parents=True)
    (vd / "look.json").write_text(json.dumps({"characters": {}, "places": {
        "Mill Hall": "Inside: a long hall with a timber roof and tall windows",
        "Mill Yard": "Outside: a cobbled yard with a pump",
        "Mill Loft": "Inside: a low loft with sacks"}}))
    (vd / "01_one" / "shots" / "scenes" / "breakdown.json").write_text(json.dumps({"scenes": [
        {"start": 0, "place": "Mill Hall", "opens": "Evening, the hall lit by candles. Orla stands by the door."},
        {"start": 50, "place": "the Mill Yard (outside)", "opens": "Night. Tam waits by the pump."},
        {"start": 90, "place": "Mill Loft", "opens": "Morning light through the hatch. Orla sits on a sack."},
        {"start": 120, "place": "Mill Hall", "opens": "Noon, bright daylight. The hall is empty."}]}))
    check("the first scene at a place decides: evening is night", MF.place_night(book, "Mill Hall"))
    check("any spelling of the place", MF.place_night(book, "Mill Yard (outside)"))
    check("a morning scene is not night", not MF.place_night(book, "Mill Loft"))
    check("a place in no scene is not night", not MF.place_night(book, "Mill Cellar"))

    st = MF._hero_state(book, "Mill Hall")
    check("a new inside picture is asked for at night", isinstance(st, dict) and MF.NIGHT_INSIDE in st["prompt"], st)
    so = MF._hero_state(book, "Mill Yard (outside)")
    check("a new outside picture has a dark sky", isinstance(so, dict) and MF.NIGHT_OUTSIDE in so["prompt"], so)
    sl = MF._hero_state(book, "Mill Loft")
    check("a day place's words carry no hour", isinstance(sl, dict) and "It is night" not in sl["prompt"], sl)

    # a picture drawn before the rule (its job file holds the words without the hour) is settled, not redrawn
    plain = st["prompt"].replace(MF.NIGHT_INSIDE + " ", "")
    check("the fixture differs only by the hour", plain != st["prompt"] and "It is night" not in plain, plain)
    st["out"].write_bytes(b"x")
    st["jp"].write_text(json.dumps([{"prompt": plain}]))
    st["ok"].write_text("empty")
    got = MF._hero_state(book, "Mill Hall")
    check("an older picture without the hour stays", got == st["out"] and st["out"].exists(), got)
    # and one drawn at night stays when a later chapter's breakdown would say otherwise
    st["jp"].write_text(json.dumps([{"prompt": st["prompt"]}]))
    (vd / "01_one" / "shots" / "scenes" / "breakdown.json").write_text(json.dumps({"scenes": [
        {"start": 0, "place": "Mill Hall", "opens": "Noon, bright daylight."}]}))
    got = MF._hero_state(book, "Mill Hall")
    check("a night picture stays when the scene list changes", got == st["out"] and st["out"].exists(), got)

import shot_flow as SF
class _R:
    def look(self, w):
        return "a woman of about 30, red hair, wearing a grey wool cloak"
d = SF.describe(_R(), "Orla", {"Orla": "green, a long linen gown with a sash"})
check("an article inside the clothes is dropped from the description", d == "the woman in the green long linen gown", d)
check("plain clothes read as before", SF.describe(_R(), "Orla") == "the woman in the grey wool cloak", SF.describe(_R(), "Orla"))

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
