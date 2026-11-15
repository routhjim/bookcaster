"""The face skin's age: the MetaHuman's face texture index chosen by the drawn face's apparent age.

The character's skin has no age control; its only age-bearing choice is `skin.face_texture_index`, one of the shipped
face textures (wrinkles, pores, stubble shading and marks, and the matching high-frequency face detail). The engine
ships a table of what each index is (texture_attributes.json beside the texture model: wrinkles low/medium/high,
stubble none/low/medium/high, marks low/medium/high). This module

  - reads that table (attributes(): which indices may be used for a character: no stubble shading unless the
    character's own beard is stubble),
  - holds the index -> apparent age table measured once by the vision model on neutral portraits (ages subcommand;
    the same question face_pick.py asks of the drawn head and of the candidate faces, so all three are on one scale),
  - picks the allowed index whose measured age is nearest the drawn face's (pick()).

    python -I face_skin.py attributes [--engine DIR]
    python -I face_skin.py ages DIR --preset NAME --out TABLE.json [--vl-python PY]
        DIR holds idx_<i>.png: neutral studio portraits of one preset wearing each face texture index
        (skin_age_calibrate.py renders them); merges {"presets": {NAME: {"ages": {"<i>": years}}}} into TABLE.json
    python -I face_skin.py pick --age N --table TABLE.json [--preset NAME] [--stubble] [--engine DIR]

The index must be set before the face is rigged: it also selects the face state's high-frequency variant
(MetaHumanCharacterEditorSubsystem.cpp, SetHighFrequencyVariant on opening and ApplySkinPropertiesToFaceState on a
skin commit), and the editor's own skin tool locks it on a rigged character ("Face and body texture selection are
locked while the character is rigged"). char_make.py sets it in the make step, which runs before the cloud rig.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

AGE_QUESTION = "About how old is the person in this picture? Answer with a number only."   # face_pick.py's question
TS_GLOB = "Plugins/MetaHuman/MetaHumanCharacter/Content/Optional/TextureSynthesis/*/texture_attributes.json"
MIN_SPREAD_Y = 8.0      # a table whose measured ages span less than this says the index does not change the age read


def attributes(engine):
    """The shipped face texture attribute table: {"n": count, "attrs": {name: {"names": [...], "values": [...]}}}."""
    hits = sorted(Path(engine).glob("Engine/" + TS_GLOB)) or sorted(Path(engine).glob(TS_GLOB))
    if not hits:
        raise FileNotFoundError(f"no texture_attributes.json under {engine}")
    d = json.loads(hits[-1].read_text())
    attrs = {k: {"names": list(v["names"]), "values": [int(x) for x in v["values"]]} for k, v in d["attributes"].items()}
    n = {len(v["values"]) for v in attrs.values()}
    if len(n) != 1:
        raise ValueError(f"attribute lists differ in length: {sorted(n)}")
    return {"n": n.pop(), "attrs": attrs, "file": str(hits[-1])}


def _attr(table, word):
    for k, v in table["attrs"].items():
        if word in k.lower():
            return v
    return None


def allowed(table, stubble=False):
    """Indices a character may wear: without a stubble beard of its own, only textures with no stubble shading."""
    idx = list(range(table["n"]))
    st = _attr(table, "stubble")
    if st and not stubble:
        none = st["names"].index("None") if "None" in st["names"] else 0
        idx = [i for i in idx if st["values"][i] == none]
    return idx


def preset_ages(tab, preset=None):
    """{index: years} for a preset from the measured table; a preset that was not measured gets the median over the
    measured presets (the texture, not the face under it, carries most of the age read)."""
    ps = (tab or {}).get("presets") or {}
    if preset in ps:
        return {int(k): float(v) for k, v in ps[preset]["ages"].items()}
    out = {}
    for i in {k for p in ps.values() for k in p["ages"]}:
        vals = [float(p["ages"][i]) for p in ps.values() if i in p["ages"]]
        out[int(i)] = statistics.median(vals)
    return out


def spread(ages):
    v = sorted(ages.values())
    return (v[-1] - v[0]) if v else 0.0


def pick(age, ages, ok, table=None):
    """The allowed index whose measured age is nearest AGE; ties to fewer marks, then the lower index.
    Raises when there is nothing to choose from or the measured ages do not vary (no effect to use)."""
    cand = [i for i in ok if i in ages]
    if not cand:
        raise ValueError("SKIN AGE: no measured index among the allowed ones (run skin_age_calibrate.py)")
    if spread({i: ages[i] for i in cand}) < MIN_SPREAD_Y:
        raise ValueError(f"SKIN AGE HAD NO EFFECT: measured ages of the allowed indices span under {MIN_SPREAD_Y:g} years")
    marks = (_attr(table, "marks") or {}).get("values") if table else None
    return min(cand, key=lambda i: (abs(ages[i] - float(age)), marks[i] if marks else 0, i))


def read_ages(folder, vl_python):
    """Ask the vision model the age of each idx_<i>.png in FOLDER. Runs the image model: a card job, not for CPU."""
    qs = []
    for p in sorted(Path(folder).glob("idx_*.png")):
        m = re.fullmatch(r"idx_(\d+)", p.stem)
        if m:
            qs.append({"id": m.group(1), "image": str(p), "max_new_tokens": 4, "ask": {"age": AGE_QUESTION}})
    if not qs:
        raise SystemExit(f"no idx_<i>.png in {folder}")
    vl = subprocess.run([vl_python, "-I", str(Path(__file__).resolve().parents[2] / "vl_judge.py")],
                        input="".join(json.dumps(q) + "\n" for q in qs), capture_output=True, text=True)
    ages = {}
    for ln in vl.stdout.splitlines():
        if ln.startswith("{"):
            r = json.loads(ln)
            m = re.search(r"\d+", r["answers"].get("age") or "")
            if m:
                ages[r["id"]] = int(m.group())
    if not ages:
        raise SystemExit(f"vision model gave no ages: {vl.stderr[-800:]}")
    return ages


def merge_table(path, preset, ages):
    p = Path(path)
    tab = json.loads(p.read_text()) if p.exists() else {}
    tab.setdefault("question", AGE_QUESTION)
    tab.setdefault("presets", {}).setdefault(preset, {"ages": {}})["ages"].update({str(k): v for k, v in ages.items()})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tab, indent=1))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["attributes", "ages", "pick"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--stubble", action="store_true")
    ap.add_argument("--vl-python", default=os.environ.get("ASSETS_QWEN_PY"),
                    help="the image model's Python (default: $ASSETS_QWEN_PY)")
    a = ap.parse_args()
    if a.cmd == "ages" and not a.vl_python:
        raise SystemExit("ages: give --vl-python or set ASSETS_QWEN_PY")
    if a.engine is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import config as C
        a.engine = str(C.UE_ENGINE)
    if a.cmd == "attributes":
        t = attributes(a.engine)
        print(json.dumps({"n": t["n"], "file": t["file"],
                          "attrs": {k: {nm: v["values"].count(i) for i, nm in enumerate(v["names"])}
                                    for k, v in t["attrs"].items()}}, indent=1))
    elif a.cmd == "ages":
        ages = read_ages(a.dir, a.vl_python)
        tab = merge_table(a.out, a.preset, ages)
        got = preset_ages(tab, a.preset)
        print(json.dumps({"preset": a.preset, "measured": len(ages), "span_years": spread(got)}))
        if spread(got) < MIN_SPREAD_Y:
            raise SystemExit(f"SKIN AGE HAD NO EFFECT: the measured ages span {spread(got):g} years")
    else:
        t = attributes(a.engine)
        ages = preset_ages(json.loads(Path(a.table).read_text()), a.preset)
        i = pick(a.age, ages, allowed(t, a.stubble), t)
        print(json.dumps({"face_texture_index": i, "age": ages[i],
                          "attributes": {k: v["names"][v["values"][i]] for k, v in t["attrs"].items()}}))


if __name__ == "__main__":
    main()
