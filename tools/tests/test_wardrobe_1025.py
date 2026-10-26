"""10-25: outfits are clothes only (no fire on heads, no "No longer visible"), a repeat of the usual look is no outfit,
and `bookcaster look --do` updates each chapter plan's copy of the look and re-plans that person's outfits.
CPU only. Run with bookcaster's python: .venv/bin/python3 tools/tests/test_wardrobe_1025.py"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_reader import bookcaster as BC   # noqa: E402
from tts_reader import video              # noqa: E402

fails = []


def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  ({got})" if got and not ok else ""))
    if not ok:
        fails.append(what)


check(video.worn_only("rough undyed wool tunic, dark brown leather belt, divided tongues as of fire resting on head")
      == "rough undyed wool tunic, dark brown leather belt", "worn: the fire on his head is not clothing")
check(video.worn_only("No longer visible") == "", "worn: gone is not an outfit")
check(video.worn_only("light blue linen mantle, sandals") == "light blue linen mantle, sandals", "worn: clothes kept")
peter = "a man of about 40, thick beard, wearing a rough undyed wool tunic and a dark brown leather belt"
reply = json.dumps({"Peter": [{"from": 3, "id": "bound", "dress": "rough undyed wool tunic, iron chains"},
                              {"from": 1, "id": "tunic and belt", "dress": "rough undyed wool tunic, dark brown leather belt"},
                              {"from": 5, "id": "freed", "dress": "rough undyed wool tunic, dark brown leather belt"}]})
video._chat = lambda *a, **k: reply
w = video.plan_wardrobe(["p"] * 8, {"Peter": peter}, "", "")
check([r["id"] for r in w.get("Peter", [])] == ["bound", "freed"],
      "wardrobe: the usual clothes first are no outfit; going back to them later is", w)

book = Path(tempfile.mkdtemp())
vd = book / "work" / "video"
(vd / "02_chapter_2").mkdir(parents=True)
(vd / "look.json").write_text(json.dumps({"characters": {"Disciple": "a man, wearing white and blue", "Peter": peter}}))
plan = {"look": {"characters": {"Disciple": "a man, wearing white and blue", "Peter": peter}},
        "wardrobe": {"Disciple": [{"from": 1, "id": "x", "dress": "white tunic, blue mantle"}],
                     "Peter": [{"from": 3, "id": "bound", "dress": "iron chains"}]}, "wardrobe_k": "abc"}
(vd / "02_chapter_2" / "plan.json").write_text(json.dumps(plan))
BC.set_look(book, "Disciple", "a man, wearing brown and ochre", pinned=True)
p = json.loads((vd / "02_chapter_2" / "plan.json").read_text())
check(p["look"]["characters"]["Disciple"] == "a man, wearing brown and ochre", "look --do: the chapter's copy follows")
check(list(p["wardrobe"]) == ["Peter"] and "wardrobe_k" not in p,
      "look --do: their outfits are planned again, others' kept", p.get("wardrobe"))

# 10-25 21:40: an answer that filters down to nobody is "nobody changes", not "no answer"
old = {"look": {"characters": {"Peter": peter}}, "wardrobe_k": "old",
       "wardrobe": {"Peter": [{"from": 1, "id": "tongues of fire", "dress": "rough undyed wool tunic, dark brown leather belt"}]}}
video._chat = lambda *a, **k: json.dumps({"Peter": [{"from": 1, "id": "tongues of fire",
                                                      "dress": "rough undyed wool tunic, dark brown leather belt, tongues of fire"}]})
check(video.ensure_wardrobe(old, "p\n\np", "", "", log=lambda *a: None) and old["wardrobe"] == {},
      "wardrobe: an answer filtered to nobody replaces the stale outfits", old.get("wardrobe"))
def boom(*a, **k):
    raise OSError("down")
video._chat = boom
old = {"look": {"characters": {"Peter": peter}}, "wardrobe_k": "old",
       "wardrobe": {"Peter": [{"from": 3, "id": "bound", "dress": "iron chains"}]}}
check(not video.ensure_wardrobe(old, "p", "", "", log=lambda *a: None) and list(old["wardrobe"]) == ["Peter"],
      "wardrobe: no answer keeps the old outfits")

# a person's sheet is the current look's or none (never the newest older file), from a plan that casts them
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import charlora                              # noqa: E402
(vd / "01_chapter_1").mkdir(exist_ok=True)
(vd / "01_chapter_1" / "plan.json").write_text(json.dumps({"look": {"characters": {"Peter": peter}}, "pictures": []}))
p2 = json.loads((vd / "02_chapter_2" / "plan.json").read_text())
p2.update(pictures=[], sheet_cast=["Disciple", "Peter"])
(vd / "02_chapter_2" / "plan.json").write_text(json.dumps(p2))
(vd / "refs").mkdir(exist_ok=True)
(vd / "refs" / "disciple-00000000.png").write_bytes(b"old")
c = charlora.Char(str(book), "Disciple")
check(c.sheet(None) is None, "sheet: an older look's sheet is not used", c.sheet(None))
want = video.ref_path(vd / "refs", "Disciple", video.ref_prompts(json.loads((vd / "02_chapter_2" / "plan.json").read_text()), "")["Disciple"][0])
want.write_bytes(b"new")
check(c.sheet(None) == want, "sheet: the current look's sheet, from the chapter that casts them", c.sheet(None))

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
