"""Once per preset: which apparent age each face texture index gives, measured by the vision model.

A MetaHuman's skin has no age control; the face texture index is the only age-bearing choice (face_skin.py). This
makes one character per index from a shipped preset (char_make.py: a copy of the preset with
skin.face_texture_index set, no facial hair), renders each as the same neutral studio portrait the face comparison
uses (char_render.py, close framing), asks the vision model the age of each (face_skin.py ages: the question
face_pick.py asks of the drawn head), and merges the answers into <library>/presets/face_texture_age.json.

    python tools/assets/characters/skin_age_calibrate.py --preset NAME --work DIR [--sample N | --indices I ...]
                                                         [--batch 6] [--keep-stubble]

Everything on a device goes through the line: engine jobs (--ue-job) for the makes and stills, one image-station
job for the vision model. Nothing here runs a model or the engine itself.

--sample N takes N indices spread evenly over the wrinkle classes of the shipped attribute table (low, medium,
high) instead of every index: a first table in a fraction of the time. Indices with stubble shading are left out
unless --keep-stubble (they are only worn by a character whose own beard is stubble).

A table whose ages span under face_skin.MIN_SPREAD_Y years stops with "SKIN AGE HAD NO EFFECT": the index then
does not move the age read on this preset and the build step must not rely on it.

Not verified at run time (written without starting the editor, 11-12): that set_editor_property accepts an index
above the property's declared UI range of 0..1 (the shipped example example_modify_skin.py sets 3, and the skin
tool raises the range to the texture count at run time), and that a copy of a preset renders with the new texture
without a cloud texture request (the editable preview material is set by the make).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(ASSETS / "img3d"))
sys.path.insert(0, str(HERE))

import config as C  # noqa: E402
import face_skin as FS  # noqa: E402


def sample_indices(table, n, stubble=False):
    """N indices of the allowed ones, spread evenly over the wrinkle classes (each class's indices in order)."""
    ok = FS.allowed(table, stubble)
    wr = FS._attr(table, "wrinkle")
    if not wr or n >= len(ok):
        return ok
    by = {}
    for i in ok:
        by.setdefault(wr["values"][i], []).append(i)
    out, k = [], 0
    classes = sorted(by)
    while len(out) < n and any(by.values()):
        c = classes[k % len(classes)]
        if by[c]:
            # the middle of what is left of the class first, then outward: an even spread at any N
            out.append(by[c].pop(len(by[c]) // 2))
        k += 1
    return sorted(out)


def batches(idx, size):
    return [idx[i:i + size] for i in range(0, len(idx), size)]


def make_job(preset, part, out_dir, tag):
    """The char_render job for one batch: a character per index, one close still each (named idx_<i>)."""
    names = {i: f"MHC_SKIN_{tag}_{i}" for i in part}
    return {"map": "/Engine/Maps/Templates/Template_Default", "out": str(out_dir), "warm_frames": 120,
            "settle_frames": 30, "key_cd": 22, "fill_cd": 22, "rim_cd": 8, "exposure_bias": -0.5,
            "make": [{"name": names[i], "folder": "/Game/Char/Cand", "base": preset,
                      "clear_slots": ["Beard", "Mustache"], "skin": {"face_texture_index": int(i)}} for i in part],
            "characters": {f"idx_{i}": {"path": f"/Game/Char/Cand/{names[i]}"} for i in part},
            "shots": [{"name": f"idx_{i}", "char": f"idx_{i}", "frame": "close", "az": 0} for i in part]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--indices", nargs="+", type=int, default=None)
    ap.add_argument("--batch", type=int, default=6, help="characters per engine job (memory)")
    ap.add_argument("--keep-stubble", action="store_true")
    ap.add_argument("--table", default=str(Path(C.LIBRARY) / "presets" / "face_texture_age.json"))
    ap.add_argument("--plan", action="store_true", help="print the indices and the job count, submit nothing")
    a = ap.parse_args()
    table = FS.attributes(C.UE_ENGINE)
    idx = a.indices or (sample_indices(table, a.sample, a.keep_stubble) if a.sample else
                        FS.allowed(table, a.keep_stubble))
    bad = [i for i in idx if not 0 <= i < table["n"]]
    if bad:
        raise SystemExit(f"indices outside 0..{table['n'] - 1}: {bad}")
    W = Path(a.work)
    out = W / "portraits"
    todo = [i for i in idx if not (out / f"idx_{i}.png").exists()]
    parts = batches(todo, a.batch)
    if a.plan:
        print(json.dumps({"preset": a.preset, "indices": idx, "to_render": len(todo), "engine_jobs": len(parts)}))
        return
    import cast_char as CC          # the one command's line helpers (engine job with the memory wait, line entry)
    out.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch for ch in a.preset.upper() if ch.isalnum())[:10]
    for k, part in enumerate(parts):
        r = CC.ue_job(W, f"skin_{k}", str(HERE / "char_render.py"), make_job(a.preset, part, out, tag), 2400)
        made = r.get("make") or {}
        for i in part:
            got = (made.get(f"MHC_SKIN_{tag}_{i}") or {}).get("face_texture_index")
            if got != i:
                raise SystemExit(f"SKIN INDEX NOT SET: asked for {i}, the make reports {got}")
    p = CC.line(["--line-cmd", "image", "choose", "--id", f"skin_age_{tag}", "--", "bash", CC.CARD, "05", CC.HY_PY,
                 "-I", str(HERE / "face_skin.py"), "ages", str(out), "--preset", a.preset, "--out", a.table,
                 "--vl-python", CC.QWEN_PY], W / "ages.log")
    if p.returncode:
        raise SystemExit(f"ages failed (a table that does not vary stops here too): {p.stdout[-800:]}")
    ages = FS.preset_ages(json.loads(Path(a.table).read_text()), a.preset)
    print(json.dumps({"preset": a.preset, "measured": len(ages), "span_years": FS.spread(ages), "table": a.table}))


if __name__ == "__main__":
    main()
