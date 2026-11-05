"""11-05 user: "choose the style from the story". A book with no picture style gets one from its story, asked of the
planner's language model (faked here) before the plan draws anything, the way the period is; a book with a style
keeps it; a failed ask leaves the default. And the outfit-sheet prompt no longer carries the dated note the drawer
read as description (a sci-fi pilot in a top hat), while a sheet already drawn from the old words is kept.
CPU only, a SYNTHETIC book, no service is called (the language model is faked; nothing here starts or stops one).
    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_story_style_1105.py"""
import json
import re
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flow3d as F                       # noqa: E402
from tts_reader import score as SC       # noqa: E402
from tts_reader import video as V        # noqa: E402
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


asked = []
SC._chat = lambda url, model, prompt, max_tokens=0: (asked.append(prompt), '"gritty painted concept art, cold steel light."\nmore')[1]
logs = []
with tempfile.TemporaryDirectory() as td:
    b = Path(td) / "book"
    (b / "work" / "video" / "01_x").mkdir(parents=True)
    a = F.infer_art_style(b, "The ship drifted. " * 50, logs.append)
    check("no style yet: the story's style is chosen and written (one line, quotes and the last stop dropped)",
          a == "gritty painted concept art, cold steel light" and (b / "art_style.txt").read_text().strip() == a, a)
    check("it is said in the plan's log", any("plan: art style = " in m and "bookcaster art" in m for m in logs), logs)
    ask = asked[0][:len(F.ART_STYLE_ASK)]
    check("the ask names no example (no quoted style, no e.g.)", not re.search(r"e\.g\.|for example|such as|[\"“”]", ask), ask)
    asked.clear()
    check("a book with a style keeps it: no ask", F.infer_art_style(b, "x", logs.append) == "" and not asked
          and (b / "art_style.txt").read_text().strip() == a)
    (b / "art_style.txt").unlink()
    (b / "work" / "video" / "01_x" / "style.txt").write_text("pen and ink")
    check("a book already planned in a style (a chapter's style.txt) keeps it: no ask",
          F.infer_art_style(b, "x", logs.append) == "" and not asked and not (b / "art_style.txt").exists())
    (b / "work" / "video" / "01_x" / "style.txt").unlink()

    def boom(*a, **k):
        raise OSError("down")
    SC._chat = boom
    check("the ask fails: nothing written (the default style stays the fallback)",
          F.infer_art_style(b, "x", logs.append) == "" and not (b / "art_style.txt").exists())
    SC._chat = lambda url, model, prompt, max_tokens=0: (asked.append(prompt), "muted painted space opera")[1]
    (b / "work" / "video" / "01_x" / "style.txt").write_text("pen and ink")
    asked.clear()
    check("forced (`bookcaster art --auto`): asked again although a chapter has a style",
          F.infer_art_style(b, "x", logs.append, force=True) == "muted painted space opera" and len(asked) == 1
          and (b / "art_style.txt").read_text().strip() == "muted painted space opera")
    # the CLI: `art --auto` starts the planner's model the plan's own way (faked) and asks with the book's text
    from tts_reader import bookcaster as BC
    from tts_reader import animate as AN
    started = []
    AN.igpu = lambda tenant, log=print: started.append(tenant)
    BC.chapter_files = lambda book: [Path("c1"), Path("c2")]
    BC._chapter_text = lambda f: f"text of {f}"
    asked.clear()
    got = BC.art_auto(b, log=lambda m: None)
    check("art --auto: the language model brought up (faked), both chapters' text asked, the style written",
          started == ["strata"] and "text of c1" in asked[0] and "text of c2" in asked[0] and got == "muted painted space opera",
          (started, got))
src = (HERE / "flow3d.py").read_text()
pl = src[src.index("def plan(book, chapter"):]
check("plan: the style is chosen with the language model up, before the pictures are planned or drawn",
      pl.index('animate.igpu("strata", log)') < pl.index("infer_art_style(") < pl.index("BC.plan_pictures_for("))

# the outfit sheet's prompt: no dated note, no example; an old sheet is still found
p = V.outfit_prompt("Ada Brook", "a woman of about 30", {"id": "coat", "dress": "a long coat"}, "ink")
check("outfit prompt: no dated note or example in it", not re.search(r"\(1\d-\d\d|top hat|dressing gown|nightshirt", p), p)
check("outfit prompt: the rule kept in plain words", "headwear only if written here" in p)
with tempfile.TemporaryDirectory() as td:
    refs = Path(td)
    row = {"id": "coat", "dress": "a long coat"}
    new = V.outfit_path(refs, "Ada Brook", "a woman of about 30", row, "ink")
    old = V.ref_path(refs, "Ada Brook coat", p.replace(V._OUTFIT_HEADWEAR, V._OUTFIT_HEADWEAR_1027, 1))
    check("no sheet yet: the new prompt's file", new != old and new == V.ref_path(refs, "Ada Brook coat", p))
    old.write_bytes(b"png")
    check("a sheet drawn from the old words is kept (its LoRA is not retrained)",
          V.outfit_path(refs, "Ada Brook", "a woman of about 30", row, "ink") == old)
print({"fails": fails})
sys.exit(1 if fails else 0)
