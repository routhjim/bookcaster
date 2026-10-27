"""10-27: the registry (tools/registry.py) wired into the shared tools: a person's sheet, LoRA, look, wardrobe and
voice label under any spelling; one description per place; one style; set_look converging on one key per person and
keeping hand-set outfits; no story band-aids. Synthetic books in temp dirs, CPU only, no models."""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import registry as RG                       # noqa: E402
from tts_reader import video as V           # noqa: E402
from tts_reader import bookcaster as BC     # noqa: E402
from tts_reader import animate as AN        # noqa: E402

fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    if not ok:
        fails.append(name)


A_MAN = "a man of about 35, brown moustache, short brown hair, wearing a tweed jacket and dark trousers"
B_MAN = "a man of about 40, clean-shaven, black hair, wearing a black frock coat"
A_WOM = "a woman of about 30, dark hair, wearing a grey day dress"
B_WOM = "a woman of about 25, dark hair, wearing a white nightgown"
BEAST = "a massive dragon with black scales, golden eyes"


def make_book(d: Path, chars: dict, places: dict | None = None, cast: dict | None = None, plans: dict | None = None,
              style: str = "ink wash") -> Path:
    vd = d / "work" / "video"
    vd.mkdir(parents=True, exist_ok=True)
    (d / "chapters").mkdir(exist_ok=True)
    (vd / "look.json").write_text(json.dumps({"version": V.LOOK_VERSION, "characters": chars, "places": places or {}}))
    if cast is not None:
        (d / "book.cast.json").write_text(json.dumps({"characters": cast}))
    for stem, plan in (plans or {}).items():
        (vd / stem).mkdir(exist_ok=True)
        (vd / stem / "plan.json").write_text(json.dumps(plan))
        (vd / stem / "style.txt").write_text(style)
    (vd / "refs").mkdir(exist_ok=True)
    return d


# --- 1. charlora: a person's sheet and outfits under the plan's spelling ------------------------------------------
import charlora as CL   # noqa: E402
with tempfile.TemporaryDirectory() as d:
    plan = {"look": {"characters": {"Dr. Marsh": "an old copy", "Arthur Penhale": B_MAN}},
            "pictures": [{"id": "p001", "who": ["Dr. Marsh", "Arthur Penhale"], "where": "", "prompt": "x", "start": 0}],
            "wardrobe": {"Dr. Marsh": [{"from": 2, "id": "greatcoat", "dress": "a heavy grey greatcoat"}]}}
    b = make_book(Path(d), {"Doctor Marsh": A_MAN, "Arthur Penhale": B_MAN, "Dr. Marsh": "a clean-shaven alias"},
                  plans={"01_one": plan})
    c = CL.Char(str(b), "Dr. Marsh")
    check("charlora: any spelling is the first-stored person, with their look", (c.name, c.desc) == ("Doctor Marsh", A_MAN),
          (c.name, c.desc))
    want = V.ref_path(b / "work/video/refs", "Dr. Marsh", V.sheet_prompt(A_MAN, "ink wash"))
    want.write_bytes(b"x")
    check("charlora: Marsh's sheet is found under the plan's 'Dr. Marsh' (10-27: KeyError -> no LoRA)",
          c.sheet(None) == want, c.sheet(None))
    check("charlora: the wardrobe row under 'Dr. Marsh' is his outfit", list(c.outfits) == [None, "greatcoat"],
          list(c.outfits))
    ow = V.outfit_ref(b / "work/video/refs", plan, "ink wash", "Doctor Marsh", plan["wardrobe"]["Dr. Marsh"][0])
    check("video.outfit_ref: the outfit sheet is named by the wardrobe's spelling, drawn from the registry look",
          ow.name.startswith("dr_marsh_greatcoat-") and ow == V.outfit_ref(b / "work/video/refs", plan, "ink wash",
                                                                          "Dr. Marsh", plan["wardrobe"]["Dr. Marsh"][0]), ow)
    ow.write_bytes(b"y")
    check("charlora: the outfit sheet resolves too", c.sheet("greatcoat") == ow, c.sheet("greatcoat"))
    check("charlora: the style is the book's one style (the chapter's style.txt here)", c.draw_style == "ink wash")
    pr = V.ref_prompts(plan, "ink wash", RG.Book(b))
    check("video.ref_prompts with the registry: the plan's names, the registry's look",
          A_MAN in pr["Dr. Marsh"][0] and "old copy" not in pr["Dr. Marsh"][0], pr.get("Dr. Marsh"))
    jobs = {j["id"]: j for j in V.reference_jobs(plan, "ink wash", b / "work/video/refs")}
    check("video.reference_jobs finds the book's registry from refs_dir", A_MAN in jobs["ref_dr_marsh"]["prompt"])
    bj = b / "work/video/charlora/doctor_marsh/build.json"
    bj.parent.mkdir(parents=True)
    bj.write_text(json.dumps({"hash": CL.input_hash(c, CL.STEPS)}))
    check("charlora.stamp_shape names the folder's person by the registry", CL.stamp_shape(b) == ["Doctor Marsh"],
          CL.stamp_shape(b))

# --- 2. flow3d: cast, LoRA todo, audit passages, pins ------------------------------------------------------------
import flow3d as F   # noqa: E402
from tts_reader import score as SC   # noqa: E402
with tempfile.TemporaryDirectory() as d:
    cast = {"Penhale": {"same_as": ""}, "Vane": {"same_as": ""}, "Ada": {"same_as": ""}, "Marsh": {"same_as": "narrator"}}
    b = make_book(Path(d), {"Arthur Penhale": B_MAN, "Clara Vane": A_WOM, "Ada Vane": B_WOM, "Doctor Marsh": A_MAN,
                            "Dr. Marsh": "alias"}, cast=cast)
    check("flow3d.cast_names: one per person", F.cast_names(b) == ["Arthur Penhale", "Clara Vane", "Ada Vane",
                                                                   "Doctor Marsh"], F.cast_names(b))
    check("flow3d.person_key is the registry's", F.person_key("Dr. Marsh") == RG.person_key("doctor_marsh"))
    check("flow3d.lora_todo: a chapter's 'Dr. Marsh' and voice label 'Vane' name their people",
          sorted(F.lora_todo(b, ["Dr. Marsh", "Vane"])) == ["Clara Vane", "Doctor Marsh"], F.lora_todo(b, ["Dr. Marsh", "Vane"]))
    text = "Miss Vane wept.\n\nAda was dead.\n\nPenhale smoked."
    plan = {"look": {"characters": {"Ada Vane": B_WOM, "Clara Vane": A_WOM}}, "pictures": []}
    asked = []

    def fake_chat(url, model, prompt, max_tokens=0):
        asked.append(prompt)
        return json.dumps({"ok": False, "why": "same", "look": A_WOM if "Clara" in prompt.split("\n")[0] else B_WOM})
    SC._chat = fake_chat
    fx = F.audit_looks(b, text, plan, log=lambda *a: None)
    jq = next((q for q in asked if q.startswith("A story character, Ada Vane")), "")
    check("flow3d.audit_looks: Ada is audited on HER passages, not on 'Miss Vane' (Clara's voice label)",
          "Ada was dead" in jq and "Miss Vane wept" not in jq, jq[-200:])
    check("flow3d.audit_looks: an answer that repeats the look is no fix (10-27 no-op fixes)", fx == {}, fx)
    lj = json.loads((b / "work/video/look.json").read_text())
    lj["_pinned"] = {"Dr. Marsh": "a pinned look"}
    (b / "work/video/look.json").write_text(json.dumps(lj))
    plan = {"look": {"characters": {"Doctor Marsh": A_MAN}}, "pictures": []}
    fx = F.audit_looks(b, text, plan, log=lambda *a: None)
    check("flow3d.audit_looks: a pin under another spelling of them is kept", plan["look"]["characters"]["Doctor Marsh"]
          == "a pinned look" and "Doctor Marsh" in fx, (plan["look"], fx))

# --- 3. mannequin_flow: names, looks, style, period, heroes ------------------------------------------------------
import mannequin_flow as MF   # noqa: E402
with tempfile.TemporaryDirectory() as d:
    b = make_book(Path(d), {"Tomas": "a man of about 18, short dark hair, wearing a leather jerkin", "Liese": A_WOM,
                            "Greta": "a woman of about 70, grey hair", "Skarn": BEAST},
                  places={"The Old Forge": "a low stone building at the edge of the village under an open sky",
                          "Kitchen (inside)": "Inside: a warm kitchen with a hearth"},
                  plans={"01_one": {"look": {"characters": {}}, "pictures": []}}, style="chapter ink")
    R = RG.Book(b)
    two = ["Tomas", "Liese"]
    check("_cast_name: 'a woman' is the one woman of the shot", MF._cast_name("a woman", two, [], R) == "Liese")
    check("_cast_name: 'a woman' among two women is no one (never the first person)",
          MF._cast_name("a woman", ["Liese", "Greta"], [], R) is None)
    check("_cast_name: 'the young man' is the one man", MF._cast_name("the young man", ["Liese", "Tomas"], [], R) == "Tomas")
    check("_cast_name: 'the dragon' is the cast member whose look is a dragon",
          MF._cast_name("the dragon", ["Tomas", "Skarn"], ["Skarn"], R) == "Skarn")
    check("_cast_name: a creature word the looks don't name is no one", MF._cast_name("the wolf", ["Tomas", "Skarn"],
                                                                                      ["Skarn"], R) is None)
    check("_cast_name: story words name no one ('the knight', 'the smith')",
          MF._cast_name("the knight", two, [], R) is None and MF._cast_name("the smith", two, [], R) is None)
    check("_cast_name: a substring is not a person ('Tom')", MF._cast_name("Tom", two, [], R) is None)
    check("_cast_name: the full name inside a phrase", MF._cast_name("Tomas, sword raised", two, [], R) == "Tomas")
    (b / "work/video/charlora/tomas").mkdir(parents=True)
    (b / "work/video/charlora/tomas/build.json").write_text(json.dumps({"lora": "e.safetensors", "trigger": "chr_tomas"}))
    check("MF._lora: registry.lora", MF._lora(b, "tomas") == ("e.safetensors", "chr_tomas"), MF._lora(b, "tomas"))
    check("MF.style_of: the chapter's style.txt when no art_style.txt", MF.style_of(b, "01_one") == "chapter ink")
    (b / "art_style.txt").write_text("woodcut")
    RG._AT.clear()
    check("MF.style_of: art_style.txt wins", MF.style_of(b, "01_one") == "woodcut")
    buf = io.StringIO()
    MF._SAID.clear()
    with contextlib.redirect_stdout(buf):
        MF.period_line(b)
        MF.period_line(b)
    check("MF.period_line: a missing period.txt is said once, loudly", buf.getvalue().count("!!! no period") == 1, buf.getvalue())
    st = MF._hero_state(b, "The Old Forge", log=lambda *a: None)
    check("MF hero: a forge described from outside is drawn from outside (no 'forge|hall|shed' guess)",
          "a view of The Old Forge" in st["prompt"], st["prompt"][:120])
    st = MF._hero_state(b, "Kitchen", log=lambda *a: None)
    check("MF hero: the place's description under its one name ('Kitchen' = 'Kitchen (inside)')",
          "warm kitchen" in st["prompt"] and "the inside of" in st["prompt"], st["prompt"][:120])
    check("MF._look: any spelling", MF._look(b, "tomas") .startswith("a man of about 18"))

# --- 4. remake_chapter.Ctx: one look per person, any spelling ----------------------------------------------------
import remake_chapter as RC   # noqa: E402
with tempfile.TemporaryDirectory() as d:
    plan = {"look": {"characters": {"Dr. Marsh": "an old copy"}}, "pictures": [],
            "wardrobe": {"Dr. Marsh": [{"from": 0, "id": "coat", "dress": "a grey greatcoat"}]}}
    b = make_book(Path(d), {"Doctor Marsh": A_MAN, "Dr. Marsh": "a clean-shaven alias", "Clara Vane": A_WOM},
                  cast={"Marsh": {"same_as": "narrator"}, "Vane": {"same_as": ""}}, plans={"01_one": plan})
    (b / "work/video/01_one/anim").mkdir()
    (b / "work/video/01_one/anim/manifest.json").write_text("[]")
    (b / "work/01_one.timeline.json").write_text("[]")
    cx = RC.Ctx(b, "01_one", Path(d) / "out")
    check("Ctx.chars: every spelling is the first-stored look", cx.chars.get("Dr. Marsh") == A_MAN ==
          cx.chars.get("Marsh") == cx.chars["doctor_marsh"], cx.chars.get("Dr. Marsh"))
    check("Ctx.chars: a voice label is the person", cx.chars.get("Vane") == A_WOM)
    check("Ctx.speaker label -> cast name (registry)", AN.speaker_key("Vane", cx.chars) == "Clara Vane")
    check("Ctx.cast_key: a speaker in the chapter's own spelling", cx.cast_key(AN.speaker_key("Marsh", cx.chars))
          == "Dr. Marsh", cx.cast_key(AN.speaker_key("Marsh", cx.chars) or "Marsh"))
    sh = V.ref_path(b / "work/video/refs", "Dr. Marsh", V.sheet_prompt(A_MAN, RG.Book(b).style()))
    sh.write_bytes(b"x")
    (b / "work/video/refs" / "doctor_marsh-00000000.png").write_bytes(b"newer, an older look")
    check("remake_chapter.ref_sheet: the current look's sheet under any spelling, not the newest file",
          RC.ref_sheet(b / "work/video/refs", "Marsh") == sh, RC.ref_sheet(b / "work/video/refs", "Marsh"))
    check("remake_chapter.pslug: the LoRA folder is the person's", RC.pslug("Dr. Marsh") == "doctor_marsh")

# --- 5. video: places, keep_looks, generic fixes -----------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    b = make_book(Path(d), {"Doctor Marsh": A_MAN},
                  places={"The Mill (inside)": "Inside: a dusty mill room with a grindstone"})
    plan = {"look": {"characters": {"Dr. Marsh": "x"},
                     "places": {"Mill (outside)": "Outside: a stone mill by a stream", "Barn (inside)": "Inside: a barn",
                                "The Mill": "a duplicate"}}, "pictures": []}
    V.keep_looks(plan, b / "work/video/look.json")
    reg = json.loads((b / "work/video/look.json").read_text())
    check("keep_looks: an alias gets the first-stored look", plan["look"]["characters"]["Dr. Marsh"] == A_MAN)
    check("keep_looks: a new place is stored under its one name, no '(inside)'", reg["places"].get("Barn") == "Inside: a barn",
          reg["places"])
    check("keep_looks: the other side of a described place stays a place of its own",
          reg["places"].get("Mill (outside)", "").startswith("Outside:")
          and plan["look"]["places"]["Mill (outside)"].startswith("Outside:"), reg["places"])
    check("keep_looks: a second spelling of a described place takes its description",
          plan["look"]["places"]["The Mill"].startswith("Inside: a dusty mill") and "The Mill" not in reg["places"], reg["places"])
    R = RG.Book(b)
    check("registry.place_look: each side its own description", R.place_look("Mill (outside)").startswith("Outside:")
          and R.place_look("The Mill (inside)").startswith("Inside:"))
    check("registry.write_place: under the one name, never overwriting", RG.write_place(b, "the mill", "new words")
          == "The Mill (inside)" and RG.write_place(b, "Granary (inside)", "Inside: grain") == "Granary")
    check("video.outfit: a wardrobe row under another spelling", V.outfit(
        {"wardrobe": {"Doctor Marsh": [{"from": 0, "id": "c", "dress": "coat"}]}}, "Dr. Marsh", 3)["id"] == "c")
check("video.literal_safe: a veil in a look is the wardrobe's business, not stripped",
      "veil" in V.literal_safe("a woman of about 30, wearing a black dress and a heavy black veil"))
fp = V.full_prompt("Clara arrives in a dog-cart, she raises her veil.", {"characters": {}, "places": {}}, "ink", [], "")
check("video.SCENE_FIXES: no era words, no veil stripping", "19th century" not in fp and "raises her veil" in fp
      and "horse-drawn dog-cart" in fp, fp)

# --- 6. bookcaster: set_look / set_outfit / swap / narrator ------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    p1 = {"look": {"characters": {"Dr. Marsh": "x", "Doctor Marsh": "x", "Clara Vane": A_WOM}},
          "pictures": [{"id": "p1", "who": ["Dr. Marsh"], "prompt": "", "start": 0}],
          "wardrobe": {"Dr. Marsh": [{"from": 1, "id": "coat", "dress": "a grey greatcoat"},
                                      {"from": 4, "id": "robe", "dress": "a red robe"}],
                       "Clara Vane": [{"from": 2, "id": "veil", "dress": "a black veil"}]}, "wardrobe_k": "k1"}
    b = make_book(Path(d), {"Doctor Marsh": A_MAN, "Clara Vane": A_WOM, "Dr. Marsh": "a clean-shaven alias"},
                  cast={"Marsh": {"same_as": "narrator"}, "Vane": {"same_as": ""}}, plans={"01_one": p1})
    pp = b / "work/video/01_one/plan.json"
    face = A_MAN.replace("brown moustache", "grey moustache")
    key = BC.set_look(b, "Dr. Marsh", face)
    lj, pl = json.loads((b / "work/video/look.json").read_text()), json.loads(pp.read_text())
    check("set_look: written under the registry's spelling, the alias removed from look.json",
          key == "Doctor Marsh" and list(lj["characters"]) == ["Doctor Marsh", "Clara Vane"], lj["characters"])
    check("set_look: the plan's unused alias spelling removed, the used one follows",
          pl["look"]["characters"] == {"Dr. Marsh": face, "Clara Vane": A_WOM}, pl["look"]["characters"])
    check("set_look: a face-only change keeps the outfits (10-27: every outfit dropped)",
          [r["id"] for r in pl["wardrobe"]["Dr. Marsh"]] == ["coat", "robe"] and pl.get("wardrobe_k") == "k1", pl["wardrobe"])
    check("set_outfit: a hand-set outfit is pinned", BC.set_outfit(b, "Marsh", "coat", "a blue greatcoat") ==
          "Dr. Marsh (coat)" and json.loads(pp.read_text())["_pinned_outfits"] == [["Dr. Marsh", "coat"]])
    BC.set_look(b, "Marsh", face.replace("tweed jacket", "naval uniform"))
    pl = json.loads(pp.read_text())
    check("set_look: a clothes change re-plans their outfits but keeps the pinned one, others' untouched",
          [r["id"] for r in pl["wardrobe"]["Dr. Marsh"]] == ["coat"] and "wardrobe_k" not in pl
          and pl["wardrobe"]["Dr. Marsh"][0]["dress"] == "a blue greatcoat" and "Clara Vane" in pl["wardrobe"], pl["wardrobe"])
    SC._chat = lambda *a, **k: json.dumps({"Dr. Marsh": [{"from": 5, "id": "uniform", "dress": "a naval uniform"}]})
    V._chat = SC._chat
    V.ensure_wardrobe(pl, "a\n\nb\n\nc\n\nd\n\ne\n\nf", "", "", log=lambda *a: None)
    check("ensure_wardrobe: a re-plan keeps the hand-set outfit word for word",
          [(r["id"], r["dress"]) for r in pl["wardrobe"].get("Dr. Marsh", [])][:1] == [("coat", "a blue greatcoat")],
          pl["wardrobe"])
    check("narrator_character agrees with the registry", BC.narrator_character(b) == RG.Book(b).narrator() == "Doctor Marsh",
          BC.narrator_character(b))
    BC.swap_looks(b, "Marsh", "Vane", log=lambda *a: None)
    pl = json.loads(pp.read_text())
    lj = json.loads((b / "work/video/look.json").read_text())
    check("swap_looks: the plan copies end swapped too (10-27: a second swap put them back)",
          pl["look"]["characters"]["Dr. Marsh"] == A_WOM == lj["characters"]["Doctor Marsh"], pl["look"]["characters"])

# --- 7. animate.speaker_key: no story's bit parts ---------------------------------------------------------------
check("speaker_key: a 'Driver' with a look is that person (no hard-coded 'driver')",
      AN.speaker_key("Driver", {"The Driver": "a man of about 50"}) == "The Driver")
check("speaker_key: a label with no look is no one", AN.speaker_key("Driver", {"Arthur Penhale": B_MAN}) is None)

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
