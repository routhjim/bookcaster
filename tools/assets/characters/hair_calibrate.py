"""Hair colour words to groom parameters by measurement: a colour word's whiteness (the grooms' grey) is the value
whose rendered hair the vision model names with that word, asked the same question it answers for a sheet.

    python -I hair_calibrate.py portraits DIR --word grey-brown --values 0.3 0.5 0.7 0.85 1.0 \
        --out MAP.json [--vl-python PY]

DIR holds <word>_<value>.png: neutral studio portraits of one preset wearing the word's hair at each whiteness
(char_render make + close shot). The mapping is merged into MAP.json ({word: {"whiteness": v, "answers": {...}}}),
which choose_mh.py reads over its built-in values. cast_char.py runs this on its own built character for a word the
library map lacks (calibrate_hair, 11-18), with --sheet the drawn head.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["portraits"])
    ap.add_argument("dir")
    ap.add_argument("--word", required=True)
    ap.add_argument("--values", nargs="+", type=float, default=[])
    ap.add_argument("--grid", nargs="+", default=None, help="melanin,redness,whiteness entries (searched together)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sheet", default=None, help="a sheet head whose hair the text calls WORD (the target answer)")
    ap.add_argument("--vl-python", default=str(Path("~/tts-ab/.venv-qwenimage/bin/python").expanduser()))
    a = ap.parse_args()
    import choose_mh as CM
    words = list(CM.COLOURS)
    question = ("What colour is this person's hair? Reply with the number only.\n"
                + "\n".join(f"{i}. {c} hair" for i, c in enumerate(words)))
    if a.sheet:
        # the target is the vision model's own word for the drawn hair (the same question on the sheet's head),
        # so the render and the drawing are named on one scale; the text's word keys the mapping
        qs0 = [{"id": "__sheet__", "image": a.sheet, "max_new_tokens": 4, "ask": {"colour": question}}]
    else:
        qs0 = []
    if a.grid:
        # several parameters at once: each entry "melanin,redness,whiteness", its portrait <word>_<entry>.png
        grid = [g for g in a.grid]
        qs = qs0 + [{"id": g, "image": str(Path(a.dir) / f"{a.word}_{g}.png"), "max_new_tokens": 4,
                     "ask": {"colour": question}} for g in grid]
        a.values = grid
    else:
        qs = None
    qs = qs or qs0 + [{"id": str(v), "image": str(Path(a.dir) / f"{a.word}_{v:g}.png" if (Path(a.dir) / f"{a.word}_{v:g}.png").exists() else Path(a.dir) / f"{a.word}_{v}.png"), "max_new_tokens": 4,
           "ask": {"colour": question}} for v in a.values]
    vl = subprocess.run([a.vl_python, "-I", str(Path(__file__).resolve().parents[2] / "vl_judge.py")],
                        input="".join(json.dumps(q) + "\n" for q in qs), capture_output=True, text=True)
    ans = {}
    for ln in vl.stdout.splitlines():
        if ln.startswith("{"):
            r = json.loads(ln)
            raw = (r["answers"].get("colour") or "").strip()
            digits = "".join(ch for ch in raw if ch.isdigit())
            ans[r["id"]] = words[min(int(digits), len(words) - 1)] if digits else raw
    target = ans.pop("__sheet__", None) or a.word
    hits = [v for v in a.values if ans.get(str(v)) == target]
    if a.grid and hits:
        # the middle of the region the vision model names with the target word (not its first corner)
        P = [[float(x) for x in h.split(",")] for h in hits]
        c = [sum(col) / len(P) for col in zip(*P)]
        pick = min(hits, key=lambda h: sum((float(x) - y) ** 2 for x, y in zip(h.split(","), c)))
    else:
        pick = min(hits) if hits else None
    mp = json.loads(Path(a.out).read_text()) if Path(a.out).exists() else {}
    if a.grid and pick:
        m_, r_, w_ = (float(x) for x in pick.split(","))
        mp[a.word] = {"melanin": m_, "redness": r_, "whiteness": w_, "target_answer": target, "answers": ans}
    else:
        mp[a.word] = {"whiteness": pick, "target_answer": target, "answers": ans}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(mp, indent=1))
    print(json.dumps(mp[a.word]))


if __name__ == "__main__":
    main()
