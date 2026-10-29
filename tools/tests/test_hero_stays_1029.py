"""10-29: a settled hero is kept when the set sheet's fixture wording moves (it was redrawn every run, its layout
re-asked and the director re-planned against a room the keys were not drawn on)."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import mannequin_flow as MF

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

b = Path(tempfile.mkdtemp(prefix="hero_"))
(b / "work" / "video").mkdir(parents=True)
(b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {}, "places": {
    "Baker Street Bedroom": "Inside: a Victorian bedroom with an iron bed, a fireplace and a sash window"}}))
first = MF._hero_state(b, "Baker Street Bedroom", lambda m: None, ("four-poster bed", "mantel clock"))
check("an undrawn hero is to be drawn", isinstance(first, dict), first)
first["out"].write_bytes(b"png"); first["jp"].write_text(json.dumps([{"prompt": first["prompt"]}]))
first["tried"].write_text(first["mk"])                    # every seed failed its fixtures: kept
again = MF._hero_state(b, "Baker Street Bedroom", lambda m: None, ("Four-poster bed", "Wooden mantelpiece"))
check("all seeds failed, new fixture words: kept, not redrawn", again == first["out"], again)
first["tried"].unlink(); first["ok"].write_text("3\n" + first["mk"])     # judged good with the old words
again = MF._hero_state(b, "Baker Street Bedroom", lambda m: None, ("Heavy drapes",))
check("judged good, new fixture words: kept", again == first["out"] and first["ok"].exists(), again)
(b / "work" / "video" / "look.json").write_text(json.dumps({"characters": {}, "places": {
    "Baker Street Bedroom": "Inside: a Victorian bedroom with a brass bed"}}))
again = MF._hero_state(b, "Baker Street Bedroom", lambda m: None, ("Heavy drapes",))
check("its own description changed: drawn again", isinstance(again, dict) and not first["out"].exists(), again)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
