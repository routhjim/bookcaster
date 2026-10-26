"""The 3D render stages each moment from the SCENE LAYER's shot when one exists (synthetic, CPU only, seconds).

    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_scene_layer_shots.py

10-21: remake_chapter.stage_status only looked in the book-level stage/shots, so every knight ch1 moment fell through
to the thin text staging (no work on the anvil, no left hand, raw prop boxes) while <chapter>/stage/shots held the
recipe-staged shots. Checks: the chapter's scene-layer shot wins, the book-level one is the fallback, and the
preflight line lists any moment still on thin staging. Prints PASS/FAIL per check, exit 1 on any failure.
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


with tempfile.TemporaryDirectory() as td:
    V = Path(td) / "video" / "ch"
    book_shots = Path(td) / "video" / "stage" / "shots"
    for d in (V / "stage" / "shots", book_shots):
        d.mkdir(parents=True)
    (V / "stage" / "shots" / "m1.json").write_text(json.dumps({"recipe": "strike"}))
    (book_shots / "m2.json").write_text(json.dumps({"figures": []}))
    cx = SimpleNamespace(V=V, shots_dir=book_shots, sets={"Forge": {}}, OUT=Path(td) / "out", chars={})

    ok, why, shot, _ = R.stage_status(cx, {"shot": "m1", "who": ["A"], "where": "Forge"})
    check("scene-layer shot wins", shot == str(V / "stage" / "shots" / "m1.json"), str(shot))
    ok, why, shot, _ = R.stage_status(cx, {"shot": "m2", "who": ["A"], "where": "Forge"})
    check("book-level shot is the fallback", shot == str(book_shots / "m2.json"), str(shot))

    P = {"entries": [{"id": "m1", "route": "stage", "shot": str(V / "stage" / "shots" / "m1.json")},
                     {"id": "m3", "route": "stage", "shot": str(cx.OUT / "stage" / "shots" / "m3.json")}]}
    lines = F.thin_lines(None, cx, P)
    check("thin staging is listed loudly", len(lines) == 1 and "m3" in lines[0] and "m1" not in lines[0], str(lines))

print(f"{len(FAILS)} failure(s)" if FAILS else "all passed")
sys.exit(1 if FAILS else 0)
