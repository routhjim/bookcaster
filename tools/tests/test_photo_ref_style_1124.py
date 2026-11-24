"""11-24: when nobody chose a picture style for a book its pictures are photographic, and so is everything they are
made from: sheets, outfit sheets, LoRA captions, the 3D figure's views, prop pictures, the 3D flow's keys. A book
with a chosen style (art_style.txt, or a chapter style.txt that is not the old default) is as it was."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import registry as RG
import characters as CH
import props_build as PB
import remake_chapter as RC
import mannequin_flow as MF
from tts_reader import video
from tts_reader import bookcaster as BC

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

PHOTO, OLD = RG.PHOTO_STYLE, RG.DEFAULT_STYLE
LOOK = "a broad woman of about 50, grey hair, wearing a brown apron"
PLAN = {"version": video.PLAN_VERSION, "look": {"characters": {"Orla": LOOK}, "places": {"The Mill": "a stone mill"}},
        "pictures": [{"id": "p001", "seed": 1000, "prompt": "Orla at the mill", "who": ["Orla"], "where": "The Mill"}],
        "wardrobe": {"Orla": [{"id": "cloak", "dress": "a grey travelling cloak"}]}}


def book(td, name, chapter_style=None, art=None, period=None):
    b = Path(td) / name
    vd = b / "work" / "video"
    (vd / "01_one").mkdir(parents=True)
    (vd / "refs").mkdir()
    (vd / "look.json").write_text(json.dumps({"version": video.LOOK_VERSION, "characters": {"Orla": LOOK},
                                              "places": {"The Mill": "a stone mill"}}))
    (vd / "01_one" / "plan.json").write_text(json.dumps(PLAN))
    if chapter_style is not None:
        (vd / "01_one" / "style.txt").write_text(chapter_style)
    if art is not None:
        (b / "art_style.txt").write_text(art + "\n")
    if period is not None:
        (b / "period.txt").write_text(period + "\n")
    return b


check("the photographic words name a technique and no drawing",
      "photograph" in PHOTO and not any(w in PHOTO.lower() for w in ("drawn", "illustration", "sketch", "ink")), PHOTO)
check("unchosen: nothing, the old default (with or without its full stop), the photographic words",
      all(RG.unchosen(s) for s in ("", "  ", OLD, OLD + ".", " " + OLD + " \n", PHOTO, PHOTO + "."))
      and not RG.unchosen("oil painting") and not RG.unchosen(OLD + ", sepia"))
check("is_photo: the photographic words only", RG.is_photo(PHOTO) and RG.is_photo(PHOTO + ".") and not RG.is_photo(OLD)
      and not RG.is_photo("") and not RG.is_photo("oil painting"))

with tempfile.TemporaryDirectory() as td:
    # --- the registry ---------------------------------------------------------------------------------------------
    none, dflt = RG.Book(book(td, "none")), RG.Book(book(td, "dflt", OLD))
    for tag, R in (("no style at all", none), ("the old default in the chapter's style.txt", dflt)):
        check(f"{tag}: nothing chosen, photographic", R.chosen_style("01_one") == "" and R.photographic("01_one")
              and R.photographic())
        check(f"{tag}: the reference style is photographic", R.ref_style("01_one") == PHOTO and R.ref_style() == PHOTO,
              R.ref_style("01_one"))
        check(f"{tag}: the one style is photographic", R.style("01_one") == PHOTO, R.style("01_one"))
        check(f"{tag}: the shots flow's picture style stays empty", R.picture_style() == "")
    check("the old default with a full stop is still nobody's choice",
          RG.Book(book(td, "dot", OLD + ".\n")).ref_style("01_one") == PHOTO)
    art = RG.Book(book(td, "art", "woodcut", art="woodcut"))
    check("art_style.txt: chosen, the reference style is the chapter's style.txt as before",
          art.chosen_style("01_one") == "woodcut" and not art.photographic("01_one") and art.ref_style("01_one") ==
          "woodcut" and art.style("01_one") == "woodcut" and art.picture_style() == "woodcut")
    art2 = RG.Book(book(td, "art2", OLD, art="woodcut"))     # planned before the style was chosen: as before
    check("art_style.txt over a chapter still holding the old default: sheets as before (the old default)",
          art2.ref_style("01_one") == OLD and art2.style("01_one") == "woodcut" and not art2.photographic("01_one"),
          art2.ref_style("01_one"))
    art3 = RG.Book(book(td, "art3", art="woodcut"))          # not planned yet: as before
    check("art_style.txt and no chapter style.txt: as before", art3.ref_style("01_one") == OLD and art3.style() == "woodcut")
    oil = RG.Book(book(td, "oil", "oil painting: thick paint, stormy skies"))
    check("a chapter style.txt that is not the default: chosen, unchanged",
          oil.chosen_style("01_one") == oil.ref_style("01_one") == oil.style("01_one") ==
          "oil painting: thick paint, stormy skies" and not oil.photographic("01_one") and oil.picture_style() == "")

    # --- sheets, outfit sheets, place pictures --------------------------------------------------------------------
    pr = video.ref_prompts(PLAN, None, none)
    check("a sheet with no style given: the registry's reference style (photographic)", PHOTO in pr["Orla"][0]
          and "pen and ink" not in pr["Orla"][0], pr["Orla"][0])
    check("the place picture too", PHOTO in pr["The Mill"][0], pr["The Mill"][0])
    check("without a registry the sheet is photographic too", PHOTO in video.ref_prompts(PLAN, None)["Orla"][0])
    a, b_ = video.sheet_prompt(LOOK, PHOTO), video.sheet_prompt(LOOK, OLD)
    check("a new style = a new sheet file (ref_path hashes the prompt)",
          video.ref_path(Path("r"), "Orla", a) != video.ref_path(Path("r"), "Orla", b_))
    check("no sheet prompt of a photographic book asks for a drawing",
          not any(w in p.lower() for p in (a, video.sheet_prompt(LOOK, PHOTO, True),
                                           video.outfit_prompt("Orla", LOOK, PLAN["wardrobe"]["Orla"][0], PHOTO))
                  for w in ("drawn", "drawing", "illustration", "sketch", "painted")))
    check("a chosen style's photograph-sheet words are as they were",
          video.sheet_prompt(LOOK, "woodcut", True) == video.PHOTO_SHEET.format(desc=LOOK, style="woodcut"))
    refs = none.vd / "refs"
    jobs = video.reference_jobs(PLAN, none.ref_style("01_one"), refs) + video.outfit_jobs(PLAN, none.ref_style("01_one"), refs)
    check("every reference job of a no-style book is photographic", len(jobs) == 3 and all(PHOTO in j["prompt"] for j in jobs),
          [j["prompt"][-120:] for j in jobs])

    # --- style_guard: the plan survives -----------------------------------------------------------------------------
    w = dflt.vd / "01_one"
    (w / "img").mkdir()
    (w / "img" / "p001.png").write_bytes(b"x")
    said = []
    video.style_guard(w, PHOTO, said.append)
    check("style_guard: the old default giving way to photographic keeps the plan and wipes nothing",
          (w / "plan.json").exists() and (w / "img" / "p001.png").exists() and (w / "style.txt").read_text() == PHOTO, said)
    check("... and the book is still nobody's choice afterwards", RG.Book(dflt.book).photographic("01_one")
          and RG.Book(dflt.book).ref_style("01_one") == PHOTO)
    video.style_guard(w, PHOTO, said.append)
    check("style_guard: the same style again: nothing", (w / "plan.json").exists())
    video.style_guard(w, "woodcut", said.append)
    check("style_guard: a chosen style still redoes the plan and its pictures",
          not (w / "plan.json").exists() and not (w / "img" / "p001.png").exists())
    w2 = oil.vd / "01_one"
    video.style_guard(w2, PHOTO, said.append)
    check("style_guard: a chosen style given up for none redoes the plan (a real change)", not (w2 / "plan.json").exists())

    # --- bookcaster ---------------------------------------------------------------------------------------------------
    check("bookcaster.book_style: photographic when nobody chose", BC.book_style(none.book) == PHOTO)
    check("bookcaster.book_style: `bookcaster art` as before", BC.book_style(art.book) == "woodcut")
    src = (HERE.parent.parent / "tts_reader" / "bookcaster.py").read_text()
    check("bookcaster: no reader falls back to the pen-and-ink default", "video.DEFAULT_STYLE" not in src
          and src.count("book_style(book)") >= 6)
    check("bookcaster: the sheets the LoRAs learn from are in the registry's reference style",
          "style=RG.Book(book).ref_style(vd.name))" in src)

    # --- the 3D figure's views, prop pictures ---------------------------------------------------------------------
    RG._AT.clear()
    check("figure views: photographic for a no-style book", CH.figure_style(none.vd) == PHOTO + ". ", CH.figure_style(none.vd))
    check("figure views: photographic over the old default", CH.figure_style(RG.Book(book(td, "dflt2", OLD)).vd) == PHOTO + ". ")
    oil2 = RG.Book(book(td, "oil2", "oil painting: thick paint, stormy skies"))
    check("figure views: a chosen style's technique, cut at ':' as before (own book)", CH.figure_style(oil2.vd) == "oil painting. ",
          CH.figure_style(oil2.vd))
    check("prop pictures: photographic for a no-style book", PB.prop_style(none.vd) == PHOTO + ". ")
    check("prop pictures: a chosen style cut at ':' as before", PB.prop_style(oil2.vd) == "oil painting. ")
    o = {"text": LOOK, "key": "orla"}
    pa, pb = CH.apose_prompt(o, "Orla", "front", "", CH.figure_style(none.vd)), CH.apose_prompt(o, "Orla", "front", "", OLD + ". ")
    check("the A-pose prompt carries the photographic words and no pen and ink", PHOTO in pa and "pen and ink" not in pa, pa)
    check("the figure's prompt.sig words change with the style (sheet_guard sets the old figure aside)", pa != pb)
    csrc = (HERE.parent / "characters.py").read_text()
    check("characters: sheet_guard's prompt.sig is made from the styled A-pose prompt",
          'cur = apose_prompt(o, name, "front", "", style)' in csrc and "style = figure_style(vd)" in csrc)
    check("no direct reader of a chapter's style.txt is left in the figure and prop builders",
          'glob("*/style.txt")' not in csrc and 'glob("*/style.txt")' not in (HERE.parent / "props_build.py").read_text())

    # --- the 3D flow's keys and LTX prompts ----------------------------------------------------------------------------
    per = video.period_line(period="the 1850s")
    check("3D keys, nothing chosen: no style words, the period line alone (as the shots flow)",
          RC.key_style(none, "01_one", "the 1850s") == per.strip(), RC.key_style(none, "01_one", "the 1850s"))
    check("3D keys, the old default in style.txt: the same", RC.key_style(RG.Book(book(td, "dflt3", OLD)), "01_one", "the 1850s") == per.strip())
    check("3D keys, nothing chosen and no period: the photographic words", RC.key_style(none, "01_one", "") == PHOTO)
    check("3D keys, a chosen chapter style: as before", RC.key_style(oil2, "01_one", "the 1850s") ==
          "oil painting: thick paint, stormy skies." + per, RC.key_style(oil2, "01_one", "the 1850s"))
    check("3D keys, art_style.txt: the chapter's style.txt as before", RC.key_style(art, "01_one", "") == "woodcut.")
    rsrc = (HERE.parent / "remake_chapter.py").read_text()
    check("remake_chapter: the context reads the style from the registry", "self.style = key_style(self.R, stem, self.period)" in rsrc
          and 'self.V / "style.txt"' not in rsrc)

    # --- the mannequin flow, the place builder, the sheets of the 3D plan, the LoRA -----------------------------------
    check("mannequin flow: photographic when nobody chose", MF.style_of(none.book, "01_one") == PHOTO)
    check("mannequin flow: a chosen chapter style as before", MF.style_of(oil2.book, "01_one") == oil2.style("01_one"))
    fsrc = (HERE.parent / "flow3d.py").read_text()
    check("flow3d: the plan's sheets are drawn and checked in the registry's reference style",
          "st = _reg(book).ref_style(Path(work).name)" in fsrc and 'r["style"]' not in fsrc)
    check("world: the place pictures ask the registry for the picture style",
          "a.style = RG.Book(book).picture_style()" in (HERE.parent / "world.py").read_text())
    lsrc = (HERE.parent / "charlora.py").read_text()
    check("charlora: the style is the registry's reference style, in the input hash and the captions",
          "self.draw_style = R.ref_style()" in lsrc and "parts = [TOOL_VERSION, steps, c.desc, c.style, c.kind]" in lsrc
          and 'c.style.split(":")[0]' in lsrc)
    check("charlora: a dataset picture of a photographic book is not asked to be a painted illustration",
          'if RG.is_photo(self.draw_style) else "A painted illustration"' in lsrc and lsrc.count("A painted illustration") == 1)

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
