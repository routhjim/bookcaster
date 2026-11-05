"""11-05: a craft in the cast failed its LoRA (2-3 of 80 pictures passed the face checks) and stopped the whole job.
A failed LoRA for a cast member who is not a person (charlora's own kind) is said and left out; a person's still
stops the run. And the cast ask says objects and craft are props. A SYNTHETIC book, CPU only, every build faked
(no card, no service).
    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_nonperson_lora_1105.py"""
import json
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flow3d as F                  # noqa: E402
from tts_reader import video as V   # noqa: E402
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = F.vdir(book)
    vd.mkdir(parents=True)
    looks = {"Ada": "a woman of about 30, short dark hair", "Kite": "a sleek grey single-seat fighter craft with swept wings",
             "Wyrm": "a great black dragon with leathery wings"}
    (vd / "look.json").write_text(json.dumps({"characters": looks}))
    calls, logs = [], []
    F.run = lambda cmd, *a, **k: (calls.append(list(map(str, cmd))), 1)[1]       # every build fails
    left = F.ensure_loras(book, log=logs.append)
    check("every build was tried", {c[c.index("build") + 2] for c in calls} == {"Ada", "Kite", "Wyrm"}, calls)
    check("a person's failed LoRA still stops the run (it is in what is left)",
          len(left) == 1 and "Ada" in left[0], left)
    check("a craft's and a creature's failed LoRA are said, and the run goes on without them",
          sum("not a person: going on" in m for m in logs) == 2 and not any("Kite" in ln or "Wyrm" in ln for ln in left),
          logs)
ask = V.plan_pictures.__code__.co_consts
txt = " ".join(c for c in ask if isinstance(c, str))
check("the cast ask: characters are living people and creatures; a craft or object is a prop",
      "only the living people and creatures" in txt and "never a character: it is a prop" in txt, txt[:200])
print({"fails": fails})
sys.exit(1 if fails else 0)
