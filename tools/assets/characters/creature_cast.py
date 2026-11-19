"""Cast a creature of the story as a library creature: the rigged model made for its look, chosen by the directing
model (stdlib).

    python tools/assets/characters/creature_cast.py --look "LOOK LINE" --out OUT.json
           [--url http://127.0.0.1:18100/v1/chat/completions] [--cache DIR] [--repeats 3]

The candidates are the library's rigged creatures (scenes/creature.py: a creature entry whose rig names its engine
mesh, clip and poses). The model is asked which of them was made as this very creature, from the look line and each
candidate's own description; an answer is kept only when every repeat gives the same id. OUT.json:
{"asset": "creature/<id>" | null, "candidates": [ids], "answers": [raw answers]}. With no rigged candidate the model
is not asked. A creature with no match has no model yet: making one (drawing, picture-to-3D, rig) is not done here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "scenes"))
import config as C  # noqa: E402
import creature  # noqa: E402

NONE = "none"
SEED = 7


def prompt(look: str, cands: list) -> str:
    """The closed question: which candidate was made as the creature the look line describes, or none."""
    return ("You cast a film of a story from a library of ready-made 3D models. A creature of the story is "
            "described:\n" + look.strip() + "\nThe library holds these rigged creature models, each with its id and "
            "its own description:\n" + "\n".join(f"- {c['asset'].split('/', 1)[1]}: {c['description']}" for c in cands)
            + "\nQuestion: which of these models was made as this very creature? It must be the same kind of "
            "creature with the same build, colouring and features as the description; a model of the same kind that "
            "differs in colour, build or features is a different creature. A pose the model holds does not matter. "
            "Answer with the model's id only, or " + NONE + " if no model is this creature.")


def parse(answer: str, cands: list) -> str | None:
    """The candidate an answer names (its id as a whole word), None for none or anything else."""
    a = (answer or "").strip().strip('."\'`').lower()
    ids = [c["asset"].split("/", 1)[1] for c in cands]
    if a in ids:
        return f"creature/{a}"
    hit = [i for i in ids if re.search(r"(?<![a-z0-9_])" + re.escape(i) + r"(?![a-z0-9_])", a)]
    return f"creature/{hit[0]}" if len(hit) == 1 else None


def ask(url: str, text: str, rep: int, cache: Path | None) -> str:
    key = hashlib.sha256(f"{rep}\n{text}".encode()).hexdigest()[:24]
    f = cache / f"{key}.txt" if cache else None
    if f and f.exists():
        return f.read_text()
    body = {"model": "default", "messages": [{"role": "user", "content": text}], "temperature": 0.0, "seed": SEED + rep,
            "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False}}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    if f:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:18100/v1/chat/completions")
    ap.add_argument("--cache")
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    cands = creature.rigged(Path(C.LIBRARY))
    out = {"asset": None, "candidates": [c["asset"] for c in cands], "answers": []}
    if cands:
        text = prompt(a.look, cands)
        out["answers"] = [ask(a.url, text, r, Path(a.cache) if a.cache else None) for r in range(a.repeats)]
        got = {parse(x, cands) for x in out["answers"]}
        out["asset"] = got.pop() if len(got) == 1 else None
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"asset": out["asset"], "candidates": len(cands)}))


if __name__ == "__main__":
    main()
