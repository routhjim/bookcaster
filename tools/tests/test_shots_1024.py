"""10-24: the shot flow's queued drawing + judging (keys_judged, heroes_for), the take judge while rendering, one
take + retakes, the saved-plan resume, 3-4 s talk pieces, the cached vision answers. CPU only: a stand-in drawer
(a --serve process writing flat PNGs whose red value is the seed) and a stand-in vision judge.
Run with bookcaster's python: .venv/bin/python3 tools/tests/test_shots_1024.py"""
import json
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import characters as CH          # noqa: E402
import mannequin_flow as MF      # noqa: E402
import remake_chapter as R       # noqa: E402
import shot_flow as SF           # noqa: E402

fails = []


def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  ({got})" if got and not ok else ""))
    if not ok:
        fails.append(what)


td = Path(tempfile.mkdtemp())
fake = td / "fake_drawer.py"
fake.write_text(textwrap.dedent('''
    import sys, json
    from PIL import Image
    print(json.dumps({"phase": "load"}), flush=True)
    for ln in sys.stdin:
        j = json.loads(ln)
        Image.new("RGB", (64, 64), (j["seed"], 0, 0)).save(j["path"])
        print(json.dumps({"id": j["id"]}), flush=True)
'''))
_popen = subprocess.Popen
drawn = []


def popen(cmd, *a, **k):
    if "--serve" in cmd:
        drawn.append(cmd)
        cmd = [sys.executable, str(fake)]
    return _popen(cmd, *a, **k)


SF.subprocess.Popen = MF.subprocess.Popen = popen
MF.QWEN_PY = sys.executable
SF.shot_dir = lambda b, s: td / "shots"


class Cx:
    @staticmethod
    def look(w, t):
        return "a young man"


SF._ctx = lambda b, s: (None, Cx())
MF._lora = lambda b, w: (None,)
MF._blank_edges = lambda k: False
SF._cast = lambda b, s, e: ([], [])
SF._key_prompt = lambda *a: "prompt"
judged = []


def ask(items):
    from PIL import Image
    out = {}
    for it in items:
        seed = Image.open(it["image"]).getpixel((0, 0))[0]
        judged.append((Path(it["image"]).parent.name, seed))
        bad = ("B-1" in it["image"] and seed in (7, 11)) or "C-1" in it["image"] or \
            ("place_b" in it["image"] and seed == 3)
        out[it["id"]] = {"text": "no", "count": "2" if bad else "1", "people": "yes" if bad else "no"}
    return out


CH.vl_ask = ask
book = td / "book"
(book / "work" / "video").mkdir(parents=True)
(book / "work" / "video" / "look.json").write_text(json.dumps({"places": {"Place A": "Inside: a hall.",
                                                                          "Place B": "Outside: a field."}}))
shots = [{"id": f"{x}-1_s0", "who": ["Edric"], "picture": "x", "place": "Place " + ("B" if x == "B" else "A")}
         for x in "ABCDE"]
dirty = SF.keys_judged(book, "s", shots, print)
keys = {s["id"]: (SF.shot_dir(0, 0) / s["id"].rsplit("_s", 1)[0] / s["id"] / "key.png") for s in shots}
from PIL import Image            # noqa: E402
seed_of = {k: Image.open(v).getpixel((0, 0))[0] for k, v in keys.items()}
check(seed_of["A-1_s0"] == 7 and seed_of["B-1_s0"] == 13 and [s["id"] for s in dirty] == ["C-1_s0"],
      "keys: one seed at a time, the next only after a failed check, all-failed reported", (seed_of, dirty))
check(not any((SF.shot_dir(0, 0) / "A-1" / "A-1_s0" / f"key_{s}.png").exists() for s in (11, 13)),
      "keys: a shot whose first seed passed draws no second seed")
heroes = book / "work" / "video" / "heroes"
check((heroes / "place_a.empty").exists() and (heroes / "place_b.empty").read_text().startswith("5"),
      "places: drawn and judged as they land, a failed seed redrawn")
check(len(drawn) <= 4, "drawers: one kept-loaded drawer per card per step", len(drawn))
judged.clear()
drawn.clear()
SF.keys_judged(book, "s", shots, print)
check(judged == [] and drawn == [], "keys + places: a rerun draws and judges nothing", (judged, drawn))

# the take judge while rendering: a faulted first take's retake joins the loaded runner's queue
o = td / "out"
o.mkdir()
more, done = td / "m.json", td / "m.done"
more.write_text("[]")
MF.fight_clip_ok = lambda p: ["deformed body at 1.5 s"] if p.name.startswith("B") else []
SF.subprocess.run = lambda *a, **k: None
stop = threading.Event()
th = threading.Thread(target=SF._judge_as_they_land, args=(o, [{"name": "A_s0", "seed": 42},
                                                                {"name": "B_s0", "seed": 42}], more, done, stop, print))
th.start()
(o / "A_s0.mp4").write_text("x")
time.sleep(11)
(o / "B_s0.mp4").write_text("x")
th.join(40)
check(json.loads(more.read_text()) == [{"name": "B_s0_t2", "seed": SF.RETAKE}] and done.exists()
      and (o / "A_s0.judged").exists(), "takes: judged as they land, the faulted one's retake queued")
(o / "B_s0_t2.mp4").write_text("y")
SF.shot_dir = lambda b, s: td
(td / "ltx" / "out").mkdir(parents=True)
for f in ("B_s0.mp4", "B_s0_t2.mp4"):
    (td / "ltx" / "out" / f).write_text((o / f).read_text())
MF.fight_clip_ok = lambda p: ["deformed body at 1.5 s"] if p.name == "B_s0.mp4" else []
again = SF.judge_takes(book, "s", [{"id": "B_s0", "kind": "action"}], print)
check(not again and (td / "ltx" / "out" / "B_s0.mp4").read_text() == "y", "takes: the cleaner retake is kept")
check(SF.TAKES == (42,), "takes: one take per action shot")

# talk lines in 3-4 s pieces
import numpy as np               # noqa: E402
c = R.talk_chunks(360, np.random.rand(600), SF.TALK_PIECE)
check(all(n <= SF.TALK_PIECE for _, n in c) and all(n >= 73 for _, n in c[:-1]) and c[0][0] == 0
      and all(c[i + 1][0] < c[i][0] + c[i][1] for i in range(len(c) - 1)),
      "talk: 3-4 s pieces that overlap", c)

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
