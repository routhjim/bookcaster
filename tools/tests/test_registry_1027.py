"""10-27 audit: one owner for who and where (tools/registry.py), on a synthetic book."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import registry as RG

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

with tempfile.TemporaryDirectory() as d:
    b = Path(d)
    (b / "work" / "video" / "charlora" / "doctor_watson").mkdir(parents=True)
    (b / "work" / "video" / "look.json").write_text(json.dumps({
        "characters": {"Sherlock Holmes": "a man of about 40", "Doctor Watson": "a man with a moustache",
                       "Helen Stoner": "a woman of about 30", "Julia Stoner": "a young woman",
                       "Dr. Watson": "a clean-shaven man (an alias stored later)",
                       "Doctor Grimesby Roylott": "clean-shaven", "Dr. Grimesby Roylott": "full grey beard"},
        "places": {"Baker Street Bedroom (inside)": "a bedroom with a bed", "Baker Street Bedroom": "no bed here",
                   "The Crown Inn Room": "an inn room"}}))
    (b / "work" / "video" / "charlora" / "doctor_watson" / "build.json").write_text(json.dumps(
        {"lora": "w.safetensors", "trigger": "chr_w", "hash": "h1"}))
    (b / "book.cast.json").write_text(json.dumps({"characters": {
        "Holmes": {"same_as": ""}, "Stoner": {"same_as": ""}, "Julia": {"same_as": ""},
        "Watson": {"same_as": "narrator"}, "Roylott": {"same_as": ""}, "Driver": {"same_as": ""}}}))
    R = RG.Book(b)
    check("one name per person: aliases merged, first-stored spelling", R.people() ==
          ["Sherlock Holmes", "Doctor Watson", "Helen Stoner", "Julia Stoner", "Doctor Grimesby Roylott"], R.people())
    check("an alias gets the first-stored look, never its own", R.look("Dr. Watson") == "a man with a moustache"
          and R.look("Dr. Grimesby Roylott") == "clean-shaven")
    check("voice labels resolve together: Julia -> Julia Stoner, so Stoner -> Helen Stoner",
          (R.person("Stoner"), R.person("Julia")) == ("Helen Stoner", "Julia Stoner"))
    check("the narrator label is the narrator character", R.person("Narrator") == "Doctor Watson" == R.narrator())
    check("a label with no look stays unknown (no guess)", R.person("Driver") is None and R.person("a woman") is None)
    check("surname/first name/title spellings", (R.person("Holmes"), R.person("Mr. Holmes"), R.person("doctor_watson"))
          == ("Sherlock Holmes", "Sherlock Holmes", "Doctor Watson"))
    check("a LoRA is found under any spelling", R.lora("Dr. Watson") == ("w.safetensors", "chr_w", "h1"))
    check("one place, one description (the first stored)", R.place("Baker Street Bedroom (inside)") ==
          R.place("Baker Street Bedroom") and R.place_look("Baker Street Bedroom") == "a bedroom with a bed")
    check("'The Crown Inn Room' = 'Crown Inn Room (inside)'", R.place_look("Crown Inn Room (inside)") == "an inn room")
    check("the style: the default when nothing is set", "pen and ink" in R.style())
    (b / "work" / "video" / "09_x").mkdir()
    (b / "work" / "video" / "09_x" / "style.txt").write_text("oil painting")
    check("the style: the chapter's style.txt when no art_style.txt", RG.Book(b).style("09_x") == "oil painting")
    (b / "art_style.txt").write_text("woodcut")
    check("the style: `bookcaster art` wins", RG.Book(b).style("09_x") == "woodcut")
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
