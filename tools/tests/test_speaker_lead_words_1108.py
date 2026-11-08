"""E26 (11-08): '"...what shall we do?" And Peter said to them, "Repent..."' made the voice label "And Peter" (the
name pattern takes two capitalised words), the timeline carried it, the registry knew no "And Peter", and the talk
shot of Peter's 16 s line was dropped for a quiet shot (Acts ch.2, 240-256 s). Where the label is made
(tts_reader/speakers.py, re-derived on every load so a cached label is mended too) and where a timeline's label is
matched (registry.Book.person): leading words the book uses as common words are not part of a name."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import fakebook
import registry as RG
from tts_reader import speakers as SP

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


TEXT = ("Now when they heard this they were cut to the heart, and said to Peter and the rest of the apostles, "
        "“Brothers, what shall we do?” And Peter said to them, “Repent and be baptized.”\n\n"
        "But Peter said, “I have no silver and gold.” And he took him by the hand, and Rose Hale watched. "
        "The rose was red. Old Brannoc set down the wrench, and Old Brannoc said, “What did you find?”\n\n"
        "Suddenly Julia Stoner cried, “Helen!” and then Julia fell, but suddenly the old house was still.\n\n")
common = SP.common_words(TEXT)
check("common words: written in lower case, capitalised only at a sentence's start", {"and", "but", "suddenly"} <= common
      and not {"peter", "julia", "rose"} & common, sorted(common & {"and", "but", "suddenly", "peter", "julia", "rose"}))
check("drop_lead_words: the narration's lead word goes, a name stays",
      [SP.drop_lead_words(n, common) for n in ("And Peter", "But Peter", "Suddenly Julia", "Old Brannoc", "Rose Hale",
                                                "Peter", "And")] ==
      ["Peter", "Peter", "Julia", "Old Brannoc", "Rose Hale", "Peter", "And"])

segs = SP.segment_dialogue(TEXT)
raw = [s.speaker for s in segs if s.kind == "quote"]
check("the heuristic still takes the lead word in (the raw label)", "And Peter" in raw and "But Peter" in raw, raw)
SP.attribute_segments(segs, attributor=None, cache_path=None, log=None)
got = [s.speaker for s in segs if s.kind == "quote"]
check("attribute_segments: the labels are the names", got[1] == "Peter" and got[2] == "Peter" and
      not any(str(g).startswith(("And ", "But ", "Suddenly ")) for g in got), got)

# a cached raw label from an earlier run is mended on load too (the cache keeps raw names)
segs = SP.segment_dialogue(TEXT)
cp = fakebook.book({}).book / "x.speakers.json"
keys = {SP._segment_key(segs, i): "And Peter" for i, s in enumerate(segs) if s.kind == "quote" and "Repent" in s.text}
cp.write_text(json.dumps({"version": 1, "speakers": keys}))
for s in segs:
    s.speaker = None
SP.attribute_segments(segs, attributor=None, cache_path=cp, log=None)
check("a cached 'And Peter' is Peter after the load", next(s.speaker for s in segs if "Repent" in s.text) == "Peter",
      next(s.speaker for s in segs if "Repent" in s.text))

# the registry: a timeline made before the fix still says "And Peter"
fb = fakebook.book({"Peter": "a man of about 40, in a brown robe", "Helen Stoner": "a woman of about 30",
                    "Old Brannoc": "an old man in coveralls"}, cast_labels={"Peter": {}, "But Peter": {"same_as": "Peter"}})
(fb.book / "book.txt").write_text(TEXT)
R = RG.Book(fb.book)
check("registry: 'And Peter' and 'But Peter' are Peter", (R.person("And Peter"), R.person("But Peter")) == ("Peter", "Peter"),
      (R.person("And Peter"), R.person("But Peter")))
check("registry: a name's own first word is never dropped ('Julia Stoner' is not Helen Stoner)",
      R.person("Julia Stoner") is None, R.person("Julia Stoner"))
check("registry: a cast name's word is kept ('Old Brannoc', 'Old Peter' stays unknown)",
      R.person("Old Brannoc") == "Old Brannoc" and R.person("Old Peter") is None, (R.person("Old Brannoc"), R.person("Old Peter")))
check("registry: a lone common word is no one", R.person("And") is None, R.person("And"))
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
