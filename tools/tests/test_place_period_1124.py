"""11-24: a place's reference picture is drawn in the book's period; a book with no period, and every person's
sheet, keep their prompts (so nothing of theirs is redrawn)."""
import sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import video

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

plan = {"look": {"characters": {"Orla": "a broad woman of about 50, grey hair"},
                 "places": {"Mill Loft": "Inside: a low loft under the rafters, sacks along one wall"}},
        "pictures": [{"id": "p1", "who": ["Orla"], "where": "Mill Loft", "prompt": "Orla in the loft"}]}
with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    (book / "work" / "video" / "refs").mkdir(parents=True)
    (book / "chapters").mkdir()
    R = video.registry_at(book / "work" / "video" / "refs")
    check("the fixture book has a registry", R is not None)
    before = video.ref_prompts(plan, "plain ink drawing", R)
    check("no period: no period words", "Set in" not in before["Mill Loft"][0], before["Mill Loft"][0])
    check("no period: the prompt as without a registry", before["Mill Loft"][0] == video.ref_prompts(plan, "plain ink drawing")["Mill Loft"][0])
    (book / "period.txt").write_text("a mill town, around 1650")
    after = video.ref_prompts(plan, "plain ink drawing", R)
    check("the place carries the period", "Set in a mill town, around 1650" in after["Mill Loft"][0]
          and "nothing modern" in after["Mill Loft"][0], after["Mill Loft"][0])
    check("before the style", after["Mill Loft"][0].index("Set in") < after["Mill Loft"][0].index("plain ink drawing"))
    check("a person's sheet is unchanged", after["Orla"] == before["Orla"])

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
