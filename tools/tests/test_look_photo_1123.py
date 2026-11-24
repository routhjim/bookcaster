"""11-23: `bookcaster look --photo NAME FILE...`: a character drawn from photographs of a real person. The pictures
live in work/video/faces/<person>/, that person's sheet has its own words and carries the photographs as refs;
everyone else's sheet is unchanged."""
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from PIL import Image
from tts_reader import video as V

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

plain = V.sheet_prompt("a tall man in a grey coat", "ink drawing")
check("a sheet without photographs keeps its words", plain.startswith("Character sheet of one person, two views")
      and "photograph" not in plain, plain)
ph = V.sheet_prompt("a tall man in a grey coat", "ink drawing", photos=True)
check("a photo sheet names the photographs and still says what they wear",
      "reference photographs" in ph and "a tall man in a grey coat" in ph and "not their clothes, hat" in ph, ph)

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = book / "work" / "video"
    refs = vd / "refs"
    refs.mkdir(parents=True)
    chars = {"Tam": "a tall man in a grey coat", "Orla": "a small woman in a red cloak"}
    (vd / "look.json").write_text(json.dumps({"characters": chars, "places": {}}))
    plan = {"look": {"characters": chars, "places": {}}, "sheet_cast": ["Tam", "Orla"], "pictures": []}
    before = {j["id"]: j for j in V.reference_jobs(plan, "ink drawing", refs)}
    check("no photographs: no refs on any sheet", all("refs" not in j for j in before.values()), before)
    R = V.registry_at(refs)
    d = vd / "faces" / V.slug(R.person("Tam"))
    d.mkdir(parents=True)
    for i in range(4):
        Image.new("RGB", (8, 8)).save(d / f"photo{i}.png")
    after = {j["id"]: j for j in V.reference_jobs(plan, "ink drawing", refs)}
    t = after["ref_tam"]
    check("their sheet carries the photographs (at most three)", len(t.get("refs", [])) == 3
          and all(str(d) in r for r in t["refs"]), t)
    check("their sheet is a new file with the photo words", t["path"] != before["ref_tam"]["path"]
          and "reference photographs" in t["prompt"], t)
    check("everyone else's sheet is unchanged", after["ref_orla"] == before["ref_orla"], after["ref_orla"])

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
