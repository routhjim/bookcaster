"""The 3D flow (11-25): a chapter's stretch from the director's shot plan, through keys staged in the 3D engine, to
video. One run() with the shots flow's contract, so `bookcaster video --flow 3d` is one background job:

  plan     shot_flow.run(until="plan"): wardrobe, the cast's LoRAs, the director's shots, the place sheets; the saved
           plan (shots/plan_*.json), no key drawn
  keys     tools/assets/story_keys.py on that plan: each person's figure from their sheet (picture to 3D), each
           place's set, every shot staged in the engine, the stills redrawn into keys and checked; all of its card
           work goes through the asset line, which this starts if it is not running
  [until="keys": the sheet of the keys, and stop]
  video    shot_flow.supplied_keys on those keys (the shots flow's own LTX render, judges and review cards), then
           the flow's cut of the stretch

The figures and sets are the book's (work/video/engine: cast.json, places.json and each place's folder), made once
and used by every chapter; a chapter's keys are in its shots/engine_keys/. Nothing here decides how anything looks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HOME = Path.home()
# what the engine tools read from the environment, where this box keeps them (used only when present and not set)
ENV_DEFAULTS = {
    "ASSETS_TRELLIS_OMP": HOME / "tts-ab" / ".venv-hyworld" / "lib" / "python3.12" / "site-packages" / "_rocm_sdk_core" / "lib" / "llvm" / "lib",
    "ASSETS_SHELL_PY": HOME / "tts-ab" / ".venv-hyworld" / "bin" / "python",
}
KEYS_JOBS = int(os.environ.get("BOOKCASTER_ENGINE_JOBS", "2"))     # places worked side by side (their card work queues)


def engine_env() -> dict:
    env = dict(os.environ)
    for k, p in ENV_DEFAULTS.items():
        if k not in env and Path(p).exists():
            env[k] = str(p)
    env.setdefault("MH_EXTRA_ARGS", "-NoMetaHumanAccountPortalLoginFallback")   # an unattended run never opens a sign-in
    env["PYTHONUNBUFFERED"] = "1"
    return env


def prefix_of(book: Path) -> str:
    """The short id the book's figures and sets are named with in the asset library: its words' first letters."""
    import re
    w = re.findall(r"[a-z0-9]+", Path(book).name.lower())
    return ("".join(x[0] for x in w) if len(w) > 1 else (w[0][:4] if w else "bk"))[:6]


def engine_dir(book: Path) -> Path:
    return Path(book) / "work" / "video" / "engine"


def keys_dir(shots: Path, plan: Path) -> Path:
    return Path(shots) / "engine_keys" / Path(plan).stem


def plan_shots(plan: Path) -> list:
    """The plan's shots, in story order."""
    by = json.loads(Path(plan).read_text())
    es = by.get("_es") or []
    order = [e["id"] for e in es] or [k for k in by if not k.startswith("_")]
    return [s for k in order for s in (by.get(k) or []) if isinstance(s, dict) and s.get("id")]


def keyed(shots: list, keys: Path) -> tuple:
    """(the shots with a key, the ids without one)"""
    have = [s for s in shots if (Path(keys) / f"{s['id']}.png").exists()]
    return have, [s["id"] for s in shots if s not in have]


def line_up(log=print) -> None:
    """The asset line runs (it owns the cards' locks for every engine, picture-to-3D and drawing job)."""
    ra = [sys.executable, str(HERE / "assets" / "run_asset.py")]
    st = subprocess.run(ra + ["--status"], capture_output=True, text=True)
    if "not running" in (st.stdout + st.stderr):
        log("3d: starting the asset line")
        r = subprocess.run(ra + ["--start"], capture_output=True, text=True, env=engine_env())
        if r.returncode:
            raise RuntimeError(f"the asset line did not start: {(r.stdout + r.stderr).strip()[-300:]}")


def engine_keys(book: Path, plan: Path, keys: Path, log=print) -> int:
    """story_keys.py on the plan, its lines told as they come (and kept in work/video/engine/keys.log). -> its exit code"""
    work = engine_dir(book)
    work.mkdir(parents=True, exist_ok=True)
    keys.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(HERE / "assets" / "story_keys.py"), str(plan), str(book), str(work), str(keys),
           "--prefix", prefix_of(book), "--jobs", str(KEYS_JOBS)]
    with open(work / "keys.log", "a") as f:
        f.write(f"\n=== {time.strftime('%m-%d %H:%M')} {' '.join(cmd)}\n")
        p = subprocess.Popen(cmd, cwd=str(HERE.parent), env=engine_env(), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        for ln in p.stdout:
            f.write(ln)
            f.flush()
            if ln.strip():
                log("3d keys: " + ln.rstrip()[:300])
        return p.wait()


def key_sheet(shots: list, keys: Path, out: Path, cols: int = 4, w: int = 480) -> Path:
    """The keys in story order on one picture, each with its shot id (a shot without a key: an empty tile)."""
    from PIL import Image, ImageDraw
    h = w * 9 // 16
    rows = max(1, -(-len(shots) // cols))
    sh = Image.new("RGB", (cols * w, rows * h), (16, 16, 16))
    d = ImageDraw.Draw(sh)
    for i, s in enumerate(shots):
        x, y = (i % cols) * w, (i // cols) * h
        f = Path(keys) / f"{s['id']}.png"
        if f.exists():
            sh.paste(Image.open(f).convert("RGB").resize((w, h)), (x, y))
        d.rectangle([x, y, x + 8 * len(s["id"]) + 8, y + 16], fill=(0, 0, 0))
        d.text((x + 4, y + 2), s["id"] + ("" if f.exists() else "  (no key)"), fill=(255, 220, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    sh.save(out, quality=88)
    return out


def run(book: Path, stem: str, t0: float, t1: float, hero, mix: Path, log=print, until: str | None = None,
        guide: str | None = None, render: tuple | None = None, review: str | None = None) -> Path:
    import shot_flow as SF
    plan = SF.run(book, stem, t0, t1, hero, mix, log, until="plan", render=render)
    shots = plan_shots(plan)
    keys = keys_dir(SF.shot_dir(book, stem), plan)
    SF._step(book, f"3d keys: figures, sets and staging in the engine ({len(shots)} shots)")
    line_up(log)
    rc = engine_keys(book, plan, keys, log)
    have, lost = keyed(shots, keys)
    if lost:
        log(f"!!! 3d: {len(lost)} of {len(shots)} shots have no engine key: {lost[:12]}")
    if not have:
        raise RuntimeError(f"3d: the engine made no key (story_keys exit {rc}; see {engine_dir(book) / 'keys.log'})")
    op = SF.out_path(book, stem, *(render or (t0, t1)))
    op = op.with_name(op.name.replace("_shots", "_3d"))
    if until == "keys":
        return key_sheet(shots, keys, op.with_name(op.stem + "_keys.jpg"))
    # the engine's editor gives its card back before the video model takes both
    subprocess.run([sys.executable, str(HERE / "assets" / "run_asset.py"), "--stop-editor"], capture_output=True,
                   text=True, env=engine_env())
    SF._step(book, f"3d video: {len(have)} shots from their engine keys")
    SF.supplied_keys(book, stem, plan, keys, [s["id"] for s in have], log, review=review or "agent", hold=True,
                     key_people_=True, in_place=True)
    return SF.assemble_stretch(book, stem, plan, have[0]["id"], have[-1]["id"], op, log)
