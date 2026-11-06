"""11-06: fewer language-model (Strata) reloads on the shot flow, and no acts pass for shot-flow chapters.
- the plan review skips its acts/beats pass for a shot-flow chapter: nothing the shot flow runs reads them (checked
  here over the call graph of what shot_flow uses), and bookcaster.shots_ready asks flow3d.plan for it
- flow3d.plan leaves the model up after the stills for the shot flow (its wardrobe asks next)
- shot_flow.run: the room layouts (vision, the model down) before the cover/wardrobe/director (the model up once);
  room_layouts doesn't stop the model when every hero and layout is cached; place_sheets leaves it up and the run
  stops it once, after describe_places
No real services: subprocess.run is faked in the modules under test (systemctl never runs)."""
import ast
import hashlib
import inspect
import json
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S          # noqa: E402
import mannequin_flow as MF    # noqa: E402
import flow3d                  # noqa: E402
import plan_review             # noqa: E402
from tts_reader import bookcaster as BC  # noqa: E402

fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    if not ok:
        fails.append(name)


calls = []


def fake_run(cmd, **k):
    calls.append(" ".join(map(str, cmd)))
    return types.SimpleNamespace(returncode=3, stdout="", stderr="")    # is-active: down


S.subprocess = types.SimpleNamespace(run=fake_run, Popen=None)
stops = lambda: [c for c in calls if c.startswith("systemctl --user stop")]   # noqa: E731

# --- 1. nothing the shot flow runs reads a frame's acts / beats / direction ---------------------------------------
def defs(path):
    return {n.name: n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef)}


def reach(mod, starts, meth=None):
    """Module functions (by name) and, with `meth`, a class's methods (self.<name>) reachable from `starts`."""
    meth = meth or {}
    seen, st = set(), [("m" if n in meth else "f", n) for n in starts]
    while st:
        k, f = st.pop()
        node = (meth if k == "m" else mod).get(f)
        if (k, f) in seen or node is None:
            continue
        seen.add((k, f))
        st += [("f", c.id) for c in ast.walk(node) if isinstance(c, ast.Name) and c.id in mod]
        st += [("m", c.attr) for c in ast.walk(node) if isinstance(c, ast.Attribute) and c.attr in meth
               and isinstance(c.value, ast.Name) and c.value.id == "self"]
    return {(k, f, (meth if k == "m" else mod)[f]) for k, f in seen}


def reads_acts(fn):
    for c in ast.walk(fn):
        if isinstance(c, ast.Constant) and c.value in ("acts", "beats", "direction"):
            return True
    return False


src = Path(S.__file__).read_text()
used = {}
for alias, path in (("MF", MF.__file__), ("F", flow3d.__file__), ("F3", flow3d.__file__)):
    used.setdefault(path, set()).update(__import__("re").findall(rf"\b{alias}\.([A-Za-z_]\w*)\(", src))
R_used = set(__import__("re").findall(r"\bR\.([A-Za-z_]\w*)\(", src)) | {"dress", "look"}
used.setdefault(str(HERE.parent / "remake_chapter.py"), set()).update(R_used)
bad = []
for path, names in used.items():
    mod, meth = defs(path), {}
    if path.endswith("remake_chapter.py"):          # Ctx methods (cx.dress, cx.look)
        ctx = next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Ctx")
        meth = {n.name: n for n in ctx.body if isinstance(n, ast.FunctionDef)}
    bad += [f"{Path(path).name}:{f}" for k, f, node in reach(mod, names, meth) if reads_acts(node)]
own = defs(S.__file__)
bad += [f"shot_flow:{f}" for f in own if reads_acts(own[f])]
check("no function the shot flow runs reads acts/beats/direction", not bad, bad)
check("review(acts=False) skips the acts pass",
      "if acts:\n        rep[\"acts\"] = acts_pass(" in inspect.getsource(plan_review.review))
check("flow3d.plan: shot_flow -> review without acts", "acts=not shot_flow" in inspect.getsource(flow3d.plan))
check("shots_ready plans with shot_flow=True", "flow3d.plan(book, stem, log=log, shot_flow=True)"
      in inspect.getsource(BC.shots_ready))
pd = inspect.getsource(flow3d._plan_draw)
check("flow3d stills: the model stays up for the shot flow (stopped on a failure)",
      "if not (ok and shot_flow):" in pd and "shot_flow" in inspect.signature(flow3d._plan_draw).parameters, pd[-400:])

# --- 2. room_layouts: no stop when everything is cached; a stop when there is vision work ----------------------
TMP = Path(tempfile.mkdtemp(prefix="sf-reloads-"))
hero = TMP / "kestrel_cockpit.png"
hero.write_bytes(b"png bytes")
S.scene_list = lambda book, stem, log=print: [{"place": "Kestrel Cockpit"}]
S.shot_dir = lambda book, stem: TMP / "shots"
S.RG.Book = lambda book: types.SimpleNamespace(place_look=lambda w: "a cockpit")
MF._hero_state = lambda book, w, log=print, must=(): hero
drawn = []
MF.heroes_for = lambda book, places, log=print: drawn.append(dict(places)) or {w: hero for w in places}
import characters as CH  # noqa: E402
CH.vl_ask = lambda jobs: {j["id"]: {"layout": "the seat in the middle"} for j in jobs}
CH.vl_close = lambda: None

calls.clear()
got = S.room_layouts(TMP, "01", log=lambda *a: None)
check("layout not cached: the model is stopped for the vision judge", len(stops()) == 1, calls)
check("layout asked and cached", got == {"Kestrel Cockpit": "the seat in the middle"}, got)
calls.clear()
drawn.clear()
got = S.room_layouts(TMP, "01", log=lambda *a: None)
check("all cached: no stop", stops() == [], calls)
check("all cached: same layout, no hero drawing", got == {"Kestrel Cockpit": "the seat in the middle"} and not drawn,
      (got, drawn))
MF._hero_state = lambda book, w, log=print, must=(): {"where": w}       # a hero still to draw / judge
calls.clear()
S.room_layouts(TMP, "01", log=lambda *a: None)
check("a hero to judge: the model is stopped", len(stops()) == 1, calls)
hero.write_bytes(b"other bytes")
MF._hero_state = lambda book, w, log=print, must=(): hero
check("a changed hero picture: not cached", S._layouts_cached(TMP, {"Kestrel Cockpit": ()}) is None)

# --- 3. run(): order of the steps and the model's stops ----------------------------------------------------------
order = []


class Stop(Exception):
    pass


S.MF.entries = lambda *a, **k: [{"id": "m1", "t": 0.0, "dur": 2.0}]
S.out_path = lambda book, stem, a, b: TMP / "out.mp4"
S._plan_sig = lambda book, stem: "sig"
S.room_layouts = lambda *a, **k: order.append("layouts") or {}
S.MF.cover = lambda *a, **k: order.append("cover") or False
S._step = lambda book, text: None
S.wardrobe = lambda *a, **k: order.append("wardrobe")
S.loras_beside = lambda *a, **k: None          # (#139: the LoRAs beside the director; nothing to train here)
S.direct = lambda book, stem, es, log, **k: order.append(("direct", k.get("keep_llm"))) or (es, {}, {})
S.sfx_plan = lambda *a, **k: order.append("sfx")
S.place_sheets = lambda *a, **k: order.append(("sheets", k.get("keep_llm"))) or {}
S.apply_chains = lambda es, by, log: es
S._migrate_ids = lambda *a, **k: None
S.describe_places = lambda *a, **k: order.append(("describe", len(stops())))


def lg(*a):
    if a and str(a[0]).startswith("rendering"):
        raise Stop
calls.clear()
try:
    S.run(TMP, "01", 0, 10, None, TMP / "mix.mp3", log=lg, render=(0, 10))
except Stop:
    pass
check("layouts before cover, wardrobe and the director",
      order[:4] == ["layouts", "cover", "wardrobe", ("direct", True)], order)
check("place_sheets keeps the model up", ("sheets", True) in order, order)
check("describe_places runs before the run's stop", ("describe", 0) in order, order)
check("the run stops the model once after describe_places", len(stops()) == 1, calls)

print(f"\n{len(fails)} failure(s)" if fails else "\nall passed")
sys.exit(1 if fails else 0)
