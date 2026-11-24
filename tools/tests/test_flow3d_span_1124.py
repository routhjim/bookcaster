"""11-24: `bookcaster assets | stage | video --flow 3d` on a time window (--from/--to) of one chapter: the window is the
moments that START in it; assets builds only their people (in the outfits worn), LoRAs, props and places; the video
is cut from the first moment's start to the last one's end and its name carries the window. No window = everything,
unchanged. Synthetic, CPU only (two short ffmpeg encodes).

    ~/tts_reader/.venv/bin/python3 tools/tests/test_flow3d_span_1124.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import flow3d as F
import props_build as PB
import world as Wd

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# ---- one made-up chapter: 6 moments over 60 s, a scene change at 30 s
MOMENTS = [{"id": "a-000", "kind": "action", "t": 0.0, "dur": 9.5, "pic": "p0"},
           {"id": "t-010", "kind": "talk", "t": 10.0, "dur": 8.0, "pic": "p0"},
           {"id": "a-020", "kind": "action", "t": 20.0, "dur": 9.0, "pic": "p0"},
           {"id": "a-030", "kind": "action", "t": 30.0, "dur": 10.0, "pic": "p1"},
           {"id": "t-041", "kind": "talk", "t": 41.0, "dur": 7.0, "pic": "p1"},
           {"id": "a-050", "kind": "action", "t": 50.0, "dur": 10.0, "pic": "p1"}]
FRAMES = {"a-000": {"shot": "a-000", "who": ["Orla"], "where": "The Mill", "props": ["Oil Lantern"]},
          "a-020": {"shot": "a-020", "who": ["Orla", "Tam"], "where": "The Mill", "props": []},
          "a-030": {"shot": "a-030", "who": ["Tam"], "where": "Weir below the mill", "props": ["Wooden Bucket"]},
          "a-050": {"shot": "a-050", "who": [], "where": "Weir below the mill", "props": ["Wooden Bucket"]}}
PLAN = {"pictures": [{"id": "p0", "start": 0, "who": ["Orla"], "where": "The Mill"},
                     {"id": "p1", "start": 2, "who": ["Tam", "Bren"], "where": "Weir below the mill"}],
        "scenes": [{"id": "s1", "t0": 0.0, "where": "The Mill"}, {"id": "s2", "t0": 30.0, "where": "Weir below the mill"}]}
TL = [{"t": 0.0, "kind": "narration", "speaker": "narrator", "text": "The wheel turned."},
      {"t": 10.0, "kind": "quote", "speaker": "Tam", "text": "Is the sluice shut?"},
      {"t": 18.0, "kind": "narration", "speaker": "narrator", "text": "Nobody answered him."},
      {"t": 41.0, "kind": "narration", "speaker": "narrator", "text": "The water was low that week."}]

# ---- window -> moment ids
check("a span string is two numbers", F.parse_span("148.0:432.5") == (148.0, 432.5), F.parse_span("148.0:432.5"))
check("no span is no window", F.parse_span(None) is None and F.parse_span("") is None)
check("no window selects nothing in particular (None = every moment)", F.span_ids(MOMENTS, None) is None)
ids = F.span_ids(MOMENTS, "10.0:30.0")
check("the moments that START in [t0, t1)", ids == {"t-010", "a-020"}, ids)
check("a moment already running at t0 is out, one starting just before t1 is in",
      F.span_ids(MOMENTS, "5:21") == {"t-010", "a-020"}, F.span_ids(MOMENTS, "5:21"))
check("an open end takes the rest", F.span_ids(MOMENTS, "30.0:1000000000.0") == {"a-030", "t-041", "a-050"})
check("a window past the chapter is empty", F.span_ids(MOMENTS, "90:120") == set())

# ---- what a window uses
u = F.span_use(MOMENTS, ids, FRAMES, PLAN, TL)
check("a window's cast: the keyframes' people and a talk moment's speaker", u["who"] == ["Tam", "Orla"], u["who"])
check("a window's places are only its own", u["where"] == ["The Mill"], u["where"])
check("a window without props lists none", u["props"] == [], u["props"])
check("each person with the time of each of their moments", u["wear"] == [("Tam", 10.0), ("Orla", 10.0), ("Orla", 20.0),
                                                                              ("Tam", 20.0)], u["wear"])
u2 = F.span_use(MOMENTS, F.span_ids(MOMENTS, "30:60"), FRAMES, PLAN, TL)
check("a talk moment during narration shows its picture's people", u2["who"] == ["Tam", "Bren"], u2["who"])
check("the second half's place and prop", (u2["where"], u2["props"]) == (["Weir below the mill"], ["Wooden Bucket"]), u2)
check("an empty keyframe does not borrow its picture's people",
      F.span_use(MOMENTS, {"a-050"}, FRAMES, PLAN, TL)["who"] == [])
uall = F.span_use(MOMENTS, None, FRAMES, PLAN, TL)
check("no window = the whole chapter's cast, places and props",
      (set(uall["who"]), uall["where"], uall["props"]) == ({"Orla", "Tam", "Bren"}, ["The Mill", "Weir below the mill"],
                                                             ["Oil Lantern", "Wooden Bucket"]), uall)
late = {**FRAMES, "a-020": dict(FRAMES["a-020"], where="Mill Yard")}
check("a moment's own place and its scene's place are both the window's",
      F.span_use(MOMENTS, {"a-020"}, late, PLAN, TL)["where"] == ["Mill Yard", "The Mill"])

# ---- the models to build: the outfit worn, an alias's own body, a creature by name
INDEX = {"characters": {
    "Orla": {"kind": "human", "outfits": {"": {"id": None, "key": "orla"}, "cloak": {"id": "cloak", "key": "orla__cloak"},
                                          "cloak2": {"id": "cloak2", "alias_of": "cloak"},
                                          "plain": {"id": "plain", "alias_of": ""}}},
    "Tam": {"kind": "human", "outfits": {"": {"id": None, "key": "tam"}}},
    "The Heron": {"kind": "creature", "outfits": {"": {"id": None}}}}}
got = F.span_models(INDEX, [("Orla", "cloak"), ("Tam", None), ("Orla", "cloak"), ("The Heron", None), ("Bren", None)])
check("one name per model worn; someone without a model entry is left to the plan's own listing",
      got == ["Orla:cloak", "Tam:", "The Heron"], got)
check("an alias is its own body's model", F.span_models(INDEX, [("Orla", "cloak2"), ("Orla", "plain")]) == ["Orla:cloak", "Orla:"],
      F.span_models(INDEX, [("Orla", "cloak2"), ("Orla", "plain")]))
check("an outfit with no model of its own is the usual look", F.span_models(INDEX, [("Tam", "apron")]) == ["Tam:"])
check("nobody in the window = no name (the caller skips the build)", F.span_models(INDEX, []) == [])

# ---- the review items of a window
RG = SimpleNamespace(same=lambda a, b: a is not None and str(a).split()[-1] == str(b).split()[-1])
win = {"where": ["The Mill"], "bases": {"lantern"}, "cast": ["Orla"]}
items = {"set:The Mill": {"name": "The Mill"}, "set:Weir below the mill": {"name": "Weir below the mill"},
         "prop:lantern__lit": {"name": "lantern__lit"}, "prop:bucket": {"name": "bucket"},
         "character:orla__cloak": {"name": "Orla"}, "character:tam": {"name": "Tam"}, "lora:Miss Orla": {"name": "Miss Orla"}}
mine = sorted(k for k, e in items.items() if F.in_window(RG, k, e, win))
check("only the window's sets, prop families, models and LoRAs are its review items",
      mine == ["character:orla__cloak", "lora:Miss Orla", "prop:lantern__lit", "set:The Mill"], mine)

# ---- the cut and the output's name
part = [m for m in MOMENTS if m["id"] in ids]
check("the cut runs from the first moment's start to the last one's end", F.span_cut(part) == (10.0, 29.0), F.span_cut(part))
check("never past the recording", F.span_cut(MOMENTS[3:], total=58.5) == (30.0, 58.5), F.span_cut(MOMENTS[3:], 58.5))
check("the window in the name, as the shots flow writes it", F.span_tag("148.0:432.9") == "_148-432", F.span_tag("148.0:432.9"))
check("no window, no tag", F.span_tag(None) == "" and F.span_tag("0.0:1000000000.0") == "")
check("an open end is named by where the cut ends", F.span_tag("30.0:1000000000.0", 60.0) == "_30-60", F.span_tag("30.0:1e9", 60.0))

# ---- the plan: whole without a window, the window's entries with one; plan.json is always the whole chapter's
calls = []
class _R:
    @staticmethod
    def plan(cx, only=None):
        calls.append(only)
        return {"stem": "01_one", "entries": [dict(m) for m in MOMENTS]}
P0 = F._plan(_R, None)
check("no window: the plan as remake_chapter made it", [e["id"] for e in P0["entries"]] == [m["id"] for m in MOMENTS])
P1 = F._plan(_R, None, ids)
check("a window: its entries in time order", [e["id"] for e in P1["entries"]] == ["t-010", "a-020"], P1["entries"])
check("the plan on disk is never narrowed to the window", calls == [None, None], calls)

# ---- a place named after another needs that one's picture
names = ["The Mill", "Barn Loft", "Weir below the mill"]
check("a place named after another", Wd.named_after("Weir below the mill", names) == "The Mill")
check("a place of its own", Wd.named_after("Barn Loft", names) is None and Wd.named_after("The Mill", names) is None)

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    cd = book / "work" / "video" / "01_one"
    (cd / "anim").mkdir(parents=True)
    (cd / "stage" / "shots").mkdir(parents=True)
    (book / "chapters").mkdir()
    (book / "chapters" / "01_one.txt").write_text("The wheel turned.\n\nNobody answered him.\n\nThe water was low that week.\n")
    (book / "work" / "video" / "look.json").write_text(json.dumps({
        "characters": {"Orla": "a woman of about 30, red hair", "Tam Reed": "a man of about 50, grey beard",
                       "Bren": "a boy of about 12"},
        "places": {"The Mill": "Inside: a timber mill with a great wheel", "Weir below the mill": "Outside: a low stone weir"}}))
    (cd / "anim" / "manifest.json").write_text(json.dumps(MOMENTS[:5]))
    (cd / "stage" / "extra_moments.json").write_text(json.dumps([dict(MOMENTS[5], extra=True), dict(MOMENTS[0], extra=True)]))
    kf = {"props": {"Oil Lantern": "a brass lantern", "Wooden Bucket": "an oak bucket"}, "frames": list(FRAMES.values())}
    (cd / "anim" / "keyframes.json").write_text(json.dumps(kf))
    (cd / "plan.json").write_text(json.dumps(PLAN))
    (book / "work" / "01_one.timeline.json").write_text(json.dumps(TL))
    (cd / "stage" / "shots" / "a-020.json").write_text(json.dumps({
        "set": {"props": [{"prop": "hearth", "name": "hearth"}, {"prop": "box", "name": "crate"}]},
        "figures": [{"who": "Orla", "holds": {"right": "tongs"}}], "props_state": {"props": {"lantern": {}}}}))

    ms = F.chapter_moments(book, "01_one")
    check("the chapter's moments: the manifest + the scene layer's added ones, once each, in time order",
          [m["id"] for m in ms] == [m["id"] for m in MOMENTS], [m["id"] for m in ms])
    check("no span, no window", F.window(book, "01_one", None) is None)
    w = F.window(book, "01_one", "10.0:30.0")
    check("the window's ids", w["ids"] == {"t-010", "a-020"}, w["ids"])
    check("the cast under the registry's names (a voice label is the person)", w["cast"] == ["Tam Reed", "Orla"], w["cast"])
    check("the prop families of its staged shots", w["bases"] == {"hearth", "tongs", "lantern"}, w["bases"])
    check("the families of the other half are the keyframes' own", PB.moment_bases(cd, {"a-030", "a-050"}) == {"bucket"},
          PB.moment_bases(cd, {"a-030", "a-050"}))
    check("a moment with no prop has no family", PB.moment_bases(cd, {"t-041"}) == set())
    try:
        F.window(book, "01_one", "90.0:120.0")
        check("a window with no moment stops with the reason", False, "no SystemExit")
    except SystemExit as ex:
        check("a window with no moment stops with the reason", "no moment" in str(ex), ex)

    # the stage step's place check looks only at the window's moments: a stray name outside it stays as planned
    kf["frames"][3] = dict(FRAMES["a-050"], where="The Weir: a low stone weir under willows", prompt="Wide shot of the mill wheel.")
    (cd / "anim" / "keyframes.json").write_text(json.dumps(kf))
    before = (cd / "anim" / "keyframes.json").read_text()
    ch = F.normalize_wheres(book, cd, log=lambda *a: None, sets={"The Mill"}, only=ids)
    check("a stray place outside the window is not re-homed to the window's set",
          ch == [] and (cd / "anim" / "keyframes.json").read_text() == before, ch)
    ch = F.normalize_wheres(book, cd, log=lambda *a: None, sets={"The Mill"})
    check("without a window it is, as before", [(i, b) for i, a, b in ch] == [(3, "The Mill")], ch)

    # ---- the assembled cut: only the window's moments, on that stretch of the sound, the window in the name
    os.environ["REMAKE_FULL3D"] = "1"
    import remake_chapter as R
    from tts_reader import video as V
    out = cd / "flow3d"
    (out / "ltx" / "out").mkdir(parents=True)
    (cd / "img").mkdir()
    ff = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi"]
    subprocess.run([*ff, "-i", "sine=frequency=330:duration=60", str(out / "mix.mp3")], check=True)
    for p in ("p0", "p1"):
        subprocess.run([*ff, "-i", "color=c=gray:s=320x180", "-frames:v", "1", str(cd / "img" / f"{p}.png")], check=True)
    for m in MOMENTS:
        if m["id"] != "a-030":                         # one moment outside the window has no clip at all
            (out / "stage" / m["id"]).mkdir(parents=True)
            (out / "stage" / m["id"] / "key.png").write_bytes(b"k")
            subprocess.run([*ff, "-i", f"color=c=blue:s=320x180:r=24:d={m['dur']}", str(out / "ltx" / "out" / f"r_{m['id']}.mp4")],
                           check=True)
    pics = [{"id": "p0", "t0": 0.0, "t1": 30.0}, {"id": "p1", "t0": 30.0, "t1": 60.0}]
    cx = SimpleNamespace(video=SimpleNamespace(audio_seconds=V.audio_seconds, timed=lambda *a: pics, build=V.build,
                                               restore_text=V.restore_text),
                         manifest=[dict(m, out=str(cd / "anim" / f"{m['id']}.mp4")) for m in MOMENTS], OUT=out, A=cd / "anim",
                         W=book / "work", V=cd, book=book, stem="01_one", plan=PLAN, text="", timeline=TL)
    os.environ["REMAKE_AUDIO"], os.environ["REMAKE_SUFFIX"] = str(out / "mix.mp3"), "3d"
    Pw = {"entries": part}
    cut = F.span_cut(Pw["entries"])
    mp4 = R.assemble(cx, Pw, cut=cut, tag=F.span_tag("10.0:30.0", cut[1]))
    check("the name: <book>_<chapter>_<stamp>_10-30_3d.mp4, complete", mp4.name.startswith("book_01_one_")
          and mp4.name.endswith("_10-30_3d.mp4"), mp4.name)
    dur = V.audio_seconds(mp4)
    check("the video is the stretch (19 s), not the chapter (60 s)", abs(dur - 19.0) < 0.5, dur)
    man = json.loads((out / "anim" / "manifest.json").read_text())
    check("only the window's moments are in it", [m["id"] for m in man] == ["t-010", "a-020"], [m["id"] for m in man])
    check("a moment outside the window is not counted missing", json.loads((out / "MISSING.json").read_text()) == [])
    (out / "ltx" / "out" / "r_a-020.mp4").unlink()
    mp4 = R.assemble(cx, Pw, cut=cut, tag="_10-30")
    check("a window moment without a clip still makes it incomplete", mp4.name.endswith("_10-30_3d_INCOMPLETE.mp4")
          and json.loads((out / "MISSING.json").read_text()) == ["a-020"], mp4.name)
    os.environ.pop("REMAKE_AUDIO"), os.environ.pop("REMAKE_SUFFIX")

# ---- the CLI: one parser for --from/--to on all three commands
from tts_reader import bookcaster as BC
check("seconds or m:ss", (BC._secs("2:28", 0.0), BC._secs("432.5", 0.0), BC._secs(None, 1e9)) == (148.0, 432.5, 1e9))
a = SimpleNamespace(t_from="2:28", t_to="7:12")
check("the job's span", BC._span(a) == "148.0:432.0", BC._span(a))
check("neither = no span (the whole chapter, as before)", BC._span(SimpleNamespace(t_from=None, t_to=None)) is None)
check("an open end", F.parse_span(BC._span(SimpleNamespace(t_from="30", t_to=None))) == (30.0, 1e9))

print("FAILED: " + ", ".join(fails) if fails else "all passed")
sys.exit(1 if fails else 0)
