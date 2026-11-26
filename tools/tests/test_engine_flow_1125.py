"""11-25: the 3D flow is one job: the director's plan (nothing drawn), its shots keyed in the engine, the shots flow's
own render from those keys. The October 3D flow's commands are retired and start nothing."""
import json, subprocess, sys, tempfile, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
from PIL import Image
import engine_flow as E

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

check("a book's library prefix is its words' first letters", E.prefix_of(Path("/x/the_tale_of_orla")) == "ttoo" and E.prefix_of(Path("orla")) == "orla")

with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    book = td / "the_hall_story"
    shots_d = book / "work" / "video" / "01_chapter_1" / "shots"
    shots_d.mkdir(parents=True)
    plan = shots_d / "plan_x.json"
    sh = lambda i, k: {"id": i, "kind": k, "place": "the Hall"}  # noqa: E731
    # written out of story order: _es is the order
    plan.write_text(json.dumps({"_sig": "s", "_es": [{"id": "m1", "t": 0}, {"id": "m2", "t": 5}],
                                "m2": [sh("m2_s1", "action")], "m1": [sh("m1_s1", "talk"), sh("m1_s2", "quiet")]}))
    shots = E.plan_shots(plan)
    check("the plan's shots come in story order", [s["id"] for s in shots] == ["m1_s1", "m1_s2", "m2_s1"])
    keys = E.keys_dir(shots_d, plan)
    check("a chapter's engine keys are beside its shots, by plan", keys == shots_d / "engine_keys" / "plan_x")
    check("the book's figures and sets are the book's, not a chapter's", E.engine_dir(book) == book / "work" / "video" / "engine")

    calls = []
    SF = types.ModuleType("shot_flow")
    def sf_run(book_, stem, t0, t1, hero, mix, log=print, until=None, render=None, **k):
        calls.append(("plan", until, render))
        return plan
    SF.run = sf_run
    SF.shot_dir = lambda b, s: shots_d
    SF._step = lambda b, t: calls.append(("step", t.split(":")[0]))
    SF.out_path = lambda b, s, t0, t1: book / "out" / f"cut_{t0:.0f}_{t1:.0f}_shots.mp4"
    def sf_supplied(book_, stem, plan_, keys_, ids, log=print, **k):
        calls.append(("render", ids, k))
        return []
    SF.supplied_keys = sf_supplied
    SF.assemble_stretch = lambda b, s, pl, a, z, out, log=print: calls.append(("cut", a, z)) or out
    sys.modules["shot_flow"] = SF
    made = {"ids": ["m1_s1", "m2_s1"], "rc": 0}
    def fake_keys(book_, plan_, keys_, log=print):
        calls.append(("engine", Path(plan_).name))
        keys_.mkdir(parents=True, exist_ok=True)
        for i in made["ids"]:
            Image.new("RGB", (64, 36), "grey").save(keys_ / f"{i}.png")
        return made["rc"]
    E.engine_keys = fake_keys
    E.line_up = lambda log=print: calls.append(("line",))
    real_run = subprocess.run
    E.subprocess = types.SimpleNamespace(run=lambda cmd, **k: calls.append(("sub", cmd[-1])) or types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    said = []
    out = E.run(book, "01_chapter_1", 0, 10, None, book / "mix.mp3", said.append, until="keys")
    check("keys only: the plan is asked for with nothing drawn, the line is up before the engine works",
          calls[0] == ("plan", "plan", None) and [c[0] for c in calls].index("line") < [c[0] for c in calls].index("engine"), calls)
    check("... and it ends at the sheet of the keys: no render, no cut",
          out.name == "cut_0_10_3d_keys.jpg" and out.exists() and not any(c[0] in ("render", "cut") for c in calls), (out, calls))
    check("a shot the engine gave no key is told", any("1 of 3 shots have no engine key" in m and "m1_s2" in m for m in said), said)
    calls.clear()
    out = E.run(book, "01_chapter_1", 0, 10, None, book / "mix.mp3", said.append, review="user", render=(2.0, 8.0))
    names = [c[0] for c in calls]
    check("the whole job: plan, engine keys, the engine's editor stopped, then the render and the cut",
          names.index("engine") < names.index("sub") < names.index("render") < names.index("cut")
          and ("sub", "--stop-editor") in calls, calls)
    r = next(c for c in calls if c[0] == "render")
    check("the render is the shots flow's own, of the keyed shots, in the book, each key saying who it shows",
          r[1] == ["m1_s1", "m2_s1"] and r[2] == {"review": "user", "hold": True, "key_people_": True, "in_place": True}, r)
    check("the cut runs from the first keyed shot to the last, named as the 3D flow's, for the window asked",
          ("cut", "m1_s1", "m2_s1") in calls and out.name == "cut_2_8_3d.mp4" and calls[0] == ("plan", "plan", (2.0, 8.0)), (out, calls[0]))
    made.update(ids=[], rc=3)
    for f in keys.glob("*.png"):
        f.unlink()
    try:
        E.run(book, "01_chapter_1", 0, 10, None, book / "mix.mp3", said.append)
        check("no key at all stops the job", False)
    except RuntimeError as e:
        check("no key at all stops the job, with where to look", "no key" in str(e) and "keys.log" in str(e), str(e))
    E.subprocess = subprocess
    del sys.modules["shot_flow"]

# --- the plan file ---------------------------------------------------------------------------------------------------
import shot_flow as SF2
with tempfile.TemporaryDirectory() as td:
    fz = Path(td) / "plan.json"
    SF2._save_plan(fz, "sig1", [{"id": "m1"}, {"id": "m2"}], [[1, 2]],
                   [{"id": "m1_s1"}, {"id": "m1_s2"}, {"id": "m2_s1"}])
    d = json.loads(fz.read_text())
    check("the saved plan holds each moment's shots, the moments, the pauses and what it was planned with",
          d == {"_sig": "sig1", "_es": [{"id": "m1"}, {"id": "m2"}], "_pauses": [[1, 2]],
                "m1": [{"id": "m1_s1"}, {"id": "m1_s2"}], "m2": [{"id": "m2_s1"}]}, d)
    check("... and reads back as the engine flow's shots", [s["id"] for s in E.plan_shots(fz)] == ["m1_s1", "m1_s2", "m2_s1"])

# --- only the stretch's people's LoRAs are built ---------------------------------------------------------------------
td_ = {"Orla": "stale", "Tam": "stale", "Wren": "picked again"}
check("the LoRAs of people in none of the stretch's moments wait", SF2.stretch_todo(td_, {"Orla": 3, "Tam": 0, "Wren": 1}) == ({"Orla": "stale", "Wren": "picked again"}, ["Tam"]))
check("moments that name nobody: everyone's is built, as before", SF2.stretch_todo(td_, {"Orla": 0, "Tam": 0}) == (td_, []))

# --- the retired commands --------------------------------------------------------------------------------------------
from tts_reader import bookcaster as BC
for argv in (["assets", "orla", "--do"], ["stage", "orla", "--chapter", "1", "--do"], ["review", "orla"],
             ["approve", "orla"], ["video", "orla", "--flow", "mannequin"]):
    try:
        BC.main(argv)
        check(f"`{' '.join(argv[:1])}` is retired", False)
    except SystemExit as e:
        check(f"`{argv[0]}{' --flow mannequin' if argv[0] == 'video' else ''}` starts nothing and names the one command",
              "retired" in str(e.code) and "video <book> --flow 3d" in str(e.code), e.code)

print("all passed" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
