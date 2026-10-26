"""The LoRA gate on a SYNTHETIC book (no story names, CPU only, seconds).

    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_lora_gate.py

User 10-20: every character gets a LoRA, and a key drawn without one is never silent. Checks: the missing cast is
listed (preflight / status lines), the video step trains it before any key (run() replaced), anyone still missing
is shouted, and every moment drawn without a person's LoRA lands in FALLBACKS (creatures are not listed).
Prints PASS/FAIL per check, exit 1 on any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flow3d as F  # noqa: E402
import remake_chapter as R  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def lora(vd: Path, who: str):
    d = vd / "charlora" / R.slug(who) / "lora"
    d.mkdir(parents=True, exist_ok=True)
    (d / "step_00200.safetensors").write_bytes(b"x")
    (d.parent / "build.json").write_text("{}")             # a FINISHED build (10-22 rule: build.json = done)


with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = F.vdir(book)
    vd.mkdir(parents=True)
    chars = {"Ada": "a woman of 30", "Bo": "a man of 60", "Wyrm": "a great dragon with wings"}
    (vd / "look.json").write_text(json.dumps({"characters": chars}))
    lora(vd, "Ada")

    todo = F.lora_todo(book)
    check("missing cast listed", set(todo) == {"Bo", "Wyrm"}, str(todo))
    check("preflight line names the missing", any("Bo" in ln for ln in F.lora_lines(book)))

    calls = []
    real_run = F.run
    F.run = lambda cmd, *a, **k: (calls.append(list(map(str, cmd))), lora(vd, "Bo") if "Bo" in map(str, cmd) else None, 0)[2]
    left = F.ensure_loras(book, log=lambda *a: None)   # Wyrm's build "fails": it stays missing
    F.run = real_run
    check("video step trains before the keys", len(calls) == 3 and all("build" in c for c in calls))
    check("a failed LoRA is retried once at once, with more drawings (10-21)",
          sum("Wyrm" in c for c in calls) == 2 and any("Wyrm" in c and "80" in c for c in calls), str(calls))
    cards = sorted({c[c.index("--card") + 1] for c in calls})
    check("one character per card, in parallel, under the card's flock",
          len(set(cards)) == 2 and all(c[0] == "flock" and f"/tmp/gpu-{c[c.index('--card') + 1]}.lock" in c for c in calls), str(cards))
    check("still-missing cast is shouted", left == [ln for ln in F.lora_lines(book)] and any("Wyrm" in ln for ln in left))

    calls.clear()
    F.run = lambda cmd, *a, **k: (calls.append(cmd), 0)[1]
    lora(vd, "Wyrm")
    F.ensure_loras(book, log=lambda *a: None)
    F.run = real_run
    check("nothing to train = no training run", not calls)

    (vd / "charlora" / R.slug("Bo")).rename(vd / "charlora" / "gone")
    cx = SimpleNamespace(W=book / "work", chars=chars)
    P = {"entries": [{"id": "talk-1", "who": [], "speaker": "Bo"},
                     {"id": "action-2", "who": ["Ada", "Wyrm"]},
                     {"id": "action-3", "who": ["Ada", "Bo"]}]}
    fb = R.lora_fallbacks(cx, P)
    check("moments without a LoRA are fallbacks", [x["id"] for x in fb] == ["talk-1", "action-3"], str(fb))
    check("creatures are not listed", all("Wyrm" not in x["why"] for x in fb))

print(f"{len(FAILS)} failure(s)" if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
