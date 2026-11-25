"""11-25: a person drawn from photographs holds their likeness from the side. Their photographs are classed by view
(front, three-quarter, profile, and the side); the sheet is drawn from the ones facing the camera, as before; the
LoRA's training set gets drawings of the face turned, each from the photograph of THAT side (mirrored for the other
side), judged against that photograph, and a floor on the share the judge sees in profile. Photographs added later
(`bookcaster look --photo-add`) retrain the LoRA; in profile they leave the sheet alone."""
import json, random, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from PIL import Image
from tts_reader import video as V
from tts_reader import bookcaster as BC
import charlora as CL

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# --- the view a photograph shows -----------------------------------------------------------------------------------
check("answers are read as one of the five views",
      [V.photo_view(a) for a in ("front", "Profile left.", "three-quarter right", "3/4 left", "profile, right")]
      == ["front", "profile left", "three-quarter right", "three-quarter left", "profile right"])
check("an answer without a kind or a side is a front view", V.photo_view("turned") == "front"
      and V.photo_view("profile") == "front" and V.photo_view("") == "front")

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = book / "work" / "video"
    refs = vd / "refs"
    refs.mkdir(parents=True)
    chars = {"Tam": "a tall man of about 40 in a grey coat", "Orla": "a small woman in a red cloak"}
    (vd / "look.json").write_text(json.dumps({"characters": chars, "places": {"the Hall": "a long stone room"}}))
    R = V.registry_at(refs)
    d = vd / "faces" / V.slug(R.person("Tam"))
    d.mkdir(parents=True)
    for i in (1, 2):
        Image.new("RGB", (40, 50), (10 * i, 0, 0)).save(d / f"photo{i}.png")
    check("photographs nobody classed are all the sheet's, as before",
          [Path(f).name for f in V.face_photos(R, "Tam")] == ["photo1.png", "photo2.png"]
          and [v for _, v in V.photos_by_view(R, "Tam")] == ["front", "front"])
    check("someone without photographs has none", V.face_photos(R, "Orla") == [] and V.photos_by_view(R, "Orla") == [])

    # --- photographs added later -------------------------------------------------------------------------------
    sheet = refs / f"{V.slug('Tam')}-0123abcd.png"
    Image.new("RGB", (8, 8)).save(sheet)
    asked = []
    answers = {"photo1.png": "front", "photo2.png": "three-quarter left", "photo3.png": "profile left",
               "photo4.png": "profile right", "photo5.png": "front"}
    def look_at(url, model, png, q):
        asked.append((Path(png).name, q is V.VIEW_ASK))
        return answers.get(Path(png).name, "none") if q is V.VIEW_ASK else "no box"
    V._look_at = look_at
    src = Path(td) / "new"
    src.mkdir()
    for n in ("a", "b", "c"):
        Image.new("RGB", (60, 80), (0, 90, 0)).save(src / f"{n}.jpg")
    who, redraw = BC.add_photos(book, "Tam", [str(src / "a.jpg"), str(src / "b.jpg")], log=lambda m: None)
    check("two profiles added: stored, every photograph classed once, the old ones too",
          who == "Tam" and V.photo_views(d) == {k: answers[k] for k in list(answers)[:4]}
          and sum(1 for _, v in asked if v) == 4, (who, V.photo_views(d), asked))
    check("profiles are not the sheet's: it is drawn from the same photographs and stays",
          not redraw and sheet.exists()
          and [Path(f).name for f in V.face_photos(R, "Tam")] == ["photo1.png", "photo2.png"], redraw)
    check("every photograph is there with its view",
          [(Path(f).name, v) for f, v in V.photos_by_view(R, "Tam")] == list(answers.items())[:4])
    who, redraw = BC.add_photos(book, "Tam", [str(src / "c.jpg")], log=lambda m: None)
    check("one facing the camera, with room on the sheet: the sheet is drawn again from it",
          redraw and not sheet.exists()
          and [Path(f).name for f in V.face_photos(R, "Tam")] == ["photo1.png", "photo2.png", "photo5.png"])
    check("classed photographs are not asked again", sum(1 for _, v in asked if v) == 5, asked)
    try:
        BC.add_photos(book, "Orla", [str(src / "a.jpg")], log=lambda m: None)
        check("someone not drawn from photographs: --photo first", False)
    except SystemExit as e:
        check("someone not drawn from photographs: --photo first", "--photo first" in str(e), str(e))

    # --- the training set: drawings of the face turned ---------------------------------------------------------
    CL.cutout = lambda s, dst: s
    out = Path(td) / "lora"
    (out / "refs").mkdir(parents=True)
    turned = {v: f for f, v in V.photos_by_view(R, "Tam") if v != "front"}
    c = types.SimpleNamespace(name="Tam", desc=chars["Tam"], face="a tall man of about 40", out=out, turned=turned,
                              look=json.loads((vd / "look.json").read_text()), outfits={None: {"id": None}},
                              style="ink drawing.", period_s="", kind="human", V=V,
                              text=lambda oid: chars["Tam"], pick=lambda m, rng, p=0.75: None)
    plan = CL.turn_plan(turned, 16)
    check("profiles twice as often as three-quarter views, both sides in turn",
          len(plan) == 16 and sum(1 for k, _ in plan if k == "profile") == 11
          and {s for k, s in plan if k == "profile"} == {"left", "right"}
          and sum(1 for k, s in plan if (k, s) == ("profile", "left")) == 6, plan)
    check("no photograph of the face turned: no such drawings", CL.turn_plan({}, 16) == []
          and CL.turn_plan({"profile left": "x"}, 0) == [])
    check("only profiles: only profiles asked", {k for k, _ in CL.turn_plan({"profile left": "x"}, 6)} == {"profile"})
    jobs = CL.gen_jobs_side(c, 16, None, random.Random(3))
    check("one job per planned view, the photograph of that side its only reference",
          len(jobs) == 16 and all(len(j["refs"]) == 1 and m["photo"] == j["refs"][0] for j, m in jobs), len(jobs))
    by = {}
    for j, m in jobs:
        by.setdefault(Path(j["refs"][0]).name, []).append(j["prompt"])
    check("each view's words go with that view's photograph",
          all("left edge" in p and "in profile" in p for p in by["photo_profile_left.png"])
          and all("right edge" in p and "in profile" in p for p in by["photo_profile_right.png"])
          and all("three-quarter" in p and "left edge" in p for p in by["photo_three-quarter_left.png"]), list(by))
    a = Image.open(out / "refs" / "photo_three-quarter_right.png")
    check("the side nobody photographed is the other side's photograph mirrored",
          "photo_three-quarter_right.png" in by and a.size == (40, 50), list(by))
    (out / "m" / "refs").mkdir(parents=True)
    px = Image.new("RGB", (4, 2)); px.putpixel((0, 0), (255, 0, 0)); px.save(d / "lr.png")
    c2 = types.SimpleNamespace(out=out / "m", turned={"profile left": str(d / "lr.png")})
    m_ = Image.open(CL.turned_ref(c2, "profile", "right")).convert("RGB")
    check("mirrored left to right", m_.getpixel((3, 0)) == (255, 0, 0) and m_.getpixel((0, 0)) == (0, 0, 0))
    (d / "lr.png").unlink()
    check("no photograph of either side of a view: not drawn from another",
          CL.turned_ref(types.SimpleNamespace(out=out, turned={"profile left": "x"}), "three-quarter", "left") is None)
    check("the drawings are pose pictures to the judge and keep their words for the caption",
          all(m["src"] == "pose" and m["turn"] in ("profile", "three-quarter") and "framing" in m for _, m in jobs))
    check("the person's name, look and clothes-free reference wording", all(
        p.startswith("Tam, ") and "exactly the face, age and hair of the person in image 1" in p
        and "same side as in image 1" in p and "image 2" not in p for ps in by.values() for p in ps))

    # --- the floor on profiles -----------------------------------------------------------------------------------
    def rows(front, side, sheets=2):
        return ([{"id": f"s{i}", "src": "sheet", "view": "front", "like": 1.0} for i in range(sheets)] +
                [{"id": f"f{i}", "src": "pose", "view": "front", "like": 0.5 + i / 100} for i in range(front)] +
                [{"id": f"p{i}", "src": "pose", "view": "side", "like": 0.6} for i in range(side)])
    check("enough profiles: nothing left out", CL.front_surplus(rows(20, 6)) == [])
    cut = CL.front_surplus(rows(30, 5))
    check("too few: the least-like front drawings left out until profiles are a fifth",
          cut == [f"f{i}" for i in range(12)], cut)
    check("never below what a build needs", len(CL.front_surplus(rows(20, 1))) == 23 - CL.MIN_KEPT)
    check("no profile passed: the set is left whole (and said so in the log)", CL.front_surplus(rows(20, 0)) == [])
    check("sheets and profiles are never the ones left out",
          all(i.startswith("f") for i in CL.front_surplus(rows(3, 1, sheets=20))))
    data = Path(td) / "data"
    data.mkdir()
    kept = []
    for n, i in enumerate(("s0", "f0", "f1", "p0")):
        Image.new("RGB", (4, 4)).save(data / f"{n:03d}.png")
        (data / f"{n:03d}.txt").write_text(i)
        kept.append({"n": n, "id": i})
    rej = []
    k2 = CL.leave_out(data, kept, rej, ["f0"])
    check("a left-out picture's files go, the rest are numbered on with their captions",
          [k["id"] for k in k2] == ["s0", "f1", "p0"] and [k["n"] for k in k2] == [0, 1, 2]
          and sorted(f.name for f in data.glob("*.png")) == ["000.png", "001.png", "002.png"]
          and [(data / f"{n:03d}.txt").read_text() for n in range(3)] == ["s0", "f1", "p0"]
          and rej and rej[0][0] == "f0", (k2, rej))
    check("nothing to leave out: untouched", CL.leave_out(data, k2, rej, []) is k2)

    # --- a LoRA follows its photographs ------------------------------------------------------------------------
    base = dict(desc="d", style="s", kind="human", outfits={}, sheet=lambda o: None, model=lambda o: None)
    plain = types.SimpleNamespace(**base, turned={})
    old = types.SimpleNamespace(**base)                      # (a Char made before there were views)
    side = types.SimpleNamespace(**base, turned={"profile left": str(d / "photo3.png")})
    check("no side photographs: the LoRA's hash is what it was (nobody else retrains)",
          CL.input_hash(plain, 400) == CL.input_hash(old, 400) and CL._turned_sig(plain) == [])
    check("a side photograph is part of what the LoRA is made from: hash, shape hash and data hash change",
          CL.input_hash(side, 400) != CL.input_hash(plain, 400) and CL.shape_hash(side, 400) != CL.shape_hash(plain, 400)
          and CL.data_hash(side) != CL.data_hash(plain))

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
