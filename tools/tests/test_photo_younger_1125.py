"""11-25: a person whose look is younger than their photographs is drawn from the photographs shown at the look's
age: the age each photograph reads is asked once, a gap over the bar plans one drawing per photograph ahead of the
sheet, the sheet (and its outfit sheets) drawn from the photographs as taken are removed, and everything that asks
for the person's photographs gets the younger ones once they exist."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from PIL import Image
from tts_reader import video as V
import flow3d as F

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

check("the age asked for is well under the age wanted", "a 20-year-old" in V.younger_words(28) and "a 16-year-old" in V.younger_words(18))

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = book / "work" / "video"
    refs = vd / "refs"
    refs.mkdir(parents=True)
    (vd / "look.json").write_text(json.dumps({"characters": {
        "Orla": "a slim woman of 25, dark hair, wearing a green gown",
        "Tam": "a man of about 60, grey hair, wearing a brown coat"}}))
    for who, n in (("orla", 2), ("tam", 1)):
        d = vd / "faces" / who
        d.mkdir(parents=True)
        for i in range(1, n + 1):
            Image.new("RGB", (300, 400), "grey").save(d / f"photo{i}.png")
    (vd / "faces" / "orla" / "views.json").write_text(json.dumps({"photo1.png": "front", "photo2.png": "profile left"}))
    plan = {"look": {"characters": {"Orla": "a slim woman of 25, dark hair, wearing a green gown",
                                    "Tam": "a man of about 60, grey hair, wearing a brown coat"}, "places": {}},
            "pictures": [], "sheet_cast": ["Orla", "Tam"],
            "wardrobe": {"Orla": [{"id": "cloak", "dress": "a grey travelling cloak"}]}}
    R = V.registry_at(refs)
    check("fixture: both are drawn from photographs", V.face_photos(R, "Orla") and V.face_photos(R, "Tam"), (V.face_photos(R, "Orla"), R.look("Orla")))
    check("no age read yet: drawn as photographed", V.younger_dir(R, "Orla") is None and not V.younger_jobs(R, "Orla"))
    st = "photograph"
    sheets = lambda: {j["id"]: j for j in V.reference_jobs(plan, st, refs) + V.outfit_jobs(plan, st, refs)}  # noqa: E731
    for j in sheets().values():
        Path(j["path"]).write_text("drawn from the photographs as taken")
    asked = []
    def judge(q):
        asked.extend(q)
        return {x["id"]: {"age": "About 47." if "orla" in x["image"] else "62"} for x in q}
    out = F.photo_ages(plan, st, refs, log=lambda *a: None, ask=judge)
    R = V.registry_at(refs)
    check("every photograph's age is asked once", len(asked) == 3 and V.photo_ages(vd / "faces" / "orla") == {"photo1.png": 47, "photo2.png": 47})
    check("the look younger than the photographs: that person only", out == ["Orla"], out)
    js = sheets()
    check("... their sheet and outfit sheet are removed, the others stay",
          not Path(js["ref_orla"]["path"]).exists() and not Path(js["ref_orla_cloak"]["path"]).exists() and Path(js["ref_tam"]["path"]).exists())
    yd = vd / "faces" / "orla" / "age25"
    check("the younger photographs have their own folder by age", V.younger_dir(R, "Orla") == yd and V.younger_dir(R, "Tam") is None)
    jobs = V.reference_jobs(plan, st, refs)
    ids = [j["id"] for j in jobs]
    check("one drawing per photograph, ahead of the sheet",
          ids.index("age_orla_photo1") < ids.index("ref_orla") and "age_orla_photo2" in ids and not any(i.startswith("age_tam") for i in ids), ids)
    yj = next(j for j in jobs if j["id"] == "age_orla_photo2")
    check("... each from its own photograph, with the words of the younger age",
          yj["refs"] == [str(vd / "faces" / "orla" / "photo2.png")] and yj["path"] == str(yd / "photo2.png") and "18-year-old" in yj["prompt"], yj)
    check("the sheet is drawn from the younger photograph facing the camera", js["ref_orla"]["refs"] == [str(yd / "photo1.png")], js["ref_orla"].get("refs"))
    check("until they are drawn everything else still gets the photographs as taken",
          V.face_photos(R, "Orla") == [str(vd / "faces" / "orla" / "photo1.png")])
    yd.mkdir(exist_ok=True)
    for i in (1, 2):
        Image.new("RGB", (64, 64)).save(yd / f"photo{i}.png")
    check("once drawn, every user of the photographs gets the younger ones, with their views",
          V.photos_by_view(R, "Orla") == [(str(yd / "photo1.png"), "front"), (str(yd / "photo2.png"), "profile left")])
    asked.clear()
    Path(js["ref_orla"]["path"]).write_text("drawn from the younger photographs")
    check("a second run asks nothing and removes nothing",
          F.photo_ages(plan, st, refs, log=lambda *a: None, ask=judge) == [] and not asked and Path(js["ref_orla"]["path"]).exists())
    facts = {f["name"]: f["age"] for j, f in F._sheet_facts(plan, st, refs)}
    check("the sheet check holds them to the look's age, and nobody drawn as photographed", facts.get("Orla") == 25 and facts.get("Tam") is None, facts)

# --- the age is read when photographs are stored, and for ones stored before -----------------------------------------
from tts_reader import bookcaster as BC
with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = book / "work" / "video"
    (vd / "refs").mkdir(parents=True)
    (vd / "look.json").write_text(json.dumps({"characters": {"Orla": "a slim woman of 25, dark hair, wearing a green gown"}}))
    d = vd / "faces" / "orla"
    d.mkdir(parents=True)
    Image.new("RGB", (300, 400), "grey").save(d / "photo1.png")
    sheet = vd / "refs" / "orla-0123abcd.png"
    Image.new("RGB", (8, 8)).save(sheet)
    asked = []
    def look_at(url, model, png, q):
        asked.append(q)
        return "front" if q is V.VIEW_ASK else "She looks about 46." if q is V.PHOTO_AGE_ASK else "no box"
    V._look_at = look_at
    who, redraw = BC.add_photos(book, "Orla", [], log=lambda m: None)
    check("stored photographs are read again with no new one: the view and the age",
          V.photo_ages(d) == {"photo1.png": 46} and V.photo_views(d) == {"photo1.png": "front"}, (V.photo_ages(d), asked))
    check("the look is younger than they read: the sheet is drawn again", redraw and not sheet.exists())
    n = len(asked)
    Image.new("RGB", (8, 8)).save(sheet)
    (d / "age25").mkdir()
    Image.new("RGB", (8, 8)).save(d / "age25" / "photo1.png")
    who, redraw = BC.add_photos(book, "Orla", [], log=lambda m: None)
    check("once they are shown at the look's age nothing is asked or drawn again", not redraw and sheet.exists() and len(asked) == n, (redraw, len(asked), n))

print("all passed" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
