"""The LTX flow as a library for bookcaster (any book): render a chapter's animated shots in its OWN anim/ dir.

    from ltx_flow import render_ltx
    render_ltx(shots, pics, plan, style, work, audio, log)      # work = <book>/work/video/<chapter stem>

shots: animate.render()'s todo list ({id, kind: talk|action, t, dur, pic, out, image (talk portrait), refs, mode}).
Every shot whose s["out"] does not exist yet is made; finished ones are skipped (resumable: the routes' own
outputs live in <work>/ltx_flow/ and are kept between runs).

Routes (tools/remake_chapter.py does the work; this only feeds it bookcaster's shots):
  talk   -> face/age check of the portrait (redraw: headshot first + age wording; still failing: the original
            frame with only the failing heads redrawn), then LTX A2Vid in <= 5 s chunks with the camera lock,
            creature speakers with jaw wording + modality 5, stitched.
  action -> LTX I2V from the (checked) keyframe, the shot prompt + motion; "continue" moments chain from the
            previous clip's last shown frame (their framing words dropped). DEFAULT since 10-20.
            With BOOKCASTER_LTX_STAGE=1: the full chain (motion_batch -> render -> keyframe_draw -> LTX Mode 2,
            depth 0.35) when the planner can stage the shot cleanly (own rig for every person) in a HY-World set.
With staging on, HY-World sets for locations without one are built first (tools/world.py build-book, its own GPU
locks), unless BOOKCASTER_LTX_SETS=0. GPU: both XTXs, one long hold per phase. The caller must free both cards first:
animate.render() calls _servers("down") before this (pauses the Fish voice and ACE music servers; LTX needs both
XTXs) and _servers("up") after; standalone callers do the same. Kernel log
watched every minute during a hold: a hit stops the LTX process and the run (SystemExit).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import remake_chapter as R  # noqa: E402

HOLD_MINUTES = float(os.environ.get("BOOKCASTER_LTX_HOLD_MIN", "120"))
# the full staging chain is OPT-IN (coordinator 10-20, after the knight ch1 remake: the staged keyframes lost the
# drama of the picture keyframes - whole-body wide framing, missing armour, stand-in bodies); default = LTX I2V
# from the checked keyframes. BOOKCASTER_LTX_STAGE=1 turns staging (and the HY-World set builds) back on.
STAGE = os.environ.get("BOOKCASTER_LTX_STAGE", "0") == "1"
# BOOKCASTER_LTX_FULL3D=1: EVERY moment through the scene layer's staged shot (<chapter>/stage/shots/<id>.json):
# staged talk stills -> keyframes (checks: outfit/age/beard/hair/location, bg-hold) -> A2Vid; action -> Mode 2;
# no silent I2V (a moment without a staged shot, or with an unapproved prop/rig/model, blocks the render)
FULL3D = os.environ.get("BOOKCASTER_LTX_FULL3D", "0") == "1"
if FULL3D:
    STAGE = True
    os.environ["REMAKE_FULL3D"] = "1"
BUILD_SETS = STAGE and os.environ.get("BOOKCASTER_LTX_SETS", "1") != "0"


def _first_frame(s: dict, anim: Path) -> str | None:
    if s["kind"] == "talk":
        return s.get("image")
    k = anim / "keyframes" / f"{s['id']}.png"
    if k.exists():
        return str(k)
    refs = s.get("refs") or []
    if refs or s.get("still_image"):
        return refs[-1] if refs else s.get("still_image")
    # a shot the keyframes don't cover (a hand-written keyframes.json for part of the chapter): its picture
    pic = anim.parent / "img" / f"{s['pic']}.png"
    return str(pic) if pic.exists() else None


def _talk_wav(s: dict, audio: Path, anim: Path) -> Path:
    wav = anim / f"{s['id']}.wav"
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s['t']:.2f}", "-t", f"{s['dur']:.2f}",
                        "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav)], check=True)
    return wav


class _KernelWatch:
    """During a long hold: journalctl -k every minute; a hit stops the LTX process (pgrep on its full path)."""

    def __init__(self, out: Path):
        self.out, self.stop, self.hit = out, threading.Event(), None
        self.t = threading.Thread(target=self.run, daemon=True)

    def run(self):
        pat = re.compile(r"\b(bug|oops|lockup|reset|ring|fault)\b", re.I)
        while not self.stop.wait(60):
            k = subprocess.run(["journalctl", "-k", "--since", "-3min"], capture_output=True, text=True).stdout
            hits = [ln for ln in k.splitlines() if pat.search(ln)]
            if hits:
                self.hit = hits
                (self.out / "KERNEL_HIT.txt").write_text("\n".join(hits))
                me = str(self.out / "ltx" / "ltx_split.py")
                for p in subprocess.run(["pgrep", "-f", f"^{R.LTX_PY} {re.escape(me)}"], capture_output=True,
                                        text=True).stdout.split():
                    subprocess.run(["kill", p])
                return

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


def render_ltx(shots: list[dict], pics: list[dict], plan: dict, style: str, work: Path, audio: Path,
               log=print, dry_run: bool = False) -> dict:
    """Make every shot's s["out"] that is missing. Returns the route plan ({"entries": [...]})."""
    work = Path(work)
    anim = work / "anim"
    book, stem = work.parents[2], work.name
    # the chapter is the voices file's (bookcaster passes <book>/work/<stem>.mp3); the work dir may be a scratch copy
    if (book / "work" / f"{Path(audio).stem}.timeline.json").exists():
        stem = Path(audio).stem
    todo = [s for s in shots if not Path(s["out"]).exists() and s.get("mode") != "still"]
    man = []
    for s in shots:
        m = {k: s[k] for k in ("id", "kind", "t", "dur", "pic", "out")}
        if (f := _first_frame(s, anim)):
            m["first"] = str(f)
        if s["kind"] == "talk":
            m["listener"] = s.get("listener")    # who the portrait shows besides the speaker (face checks)
        if s["kind"] == "talk" and not dry_run:
            m["wav"] = str(_talk_wav(s, audio, anim))
        man.append(m)
    out = work / "ltx_flow"
    if BUILD_SETS and not dry_run:
        r = subprocess.run(["python3", str(HERE / "world.py"), "build-book", str(book)], capture_output=True, text=True)
        (out.mkdir(exist_ok=True), (out / "build_sets.log").write_text(r.stdout + r.stderr))
        log(f"  HY-World sets: build-book rc={r.returncode}")
    if not STAGE:
        os.environ["REMAKE_NO_STAGE"] = "1"
    os.environ.setdefault("REMAKE_I2V_STRIP_FRAMING", "1")   # the keyframe sets the framing; the words only zoom
    cx = R.Ctx(book, stem, out, manifest=man, work=work)
    P = R.plan(cx)
    ids = {s["id"] for s in todo}
    P["entries"] = [e for e in P["entries"] if e["id"] in ids]
    from collections import Counter
    log(f"  LTX routes: {dict(Counter(e['route'] for e in P['entries']))} "
        f"(estimate {R.estimate(P)['gpu_min']})")
    if dry_run or not P["entries"]:
        return P
    # a continuation whose source clip already exists (made earlier, maybe by another engine): give the chain it
    (out / "ltx" / "out").mkdir(parents=True, exist_ok=True)
    by_id = {s["id"]: s for s in shots}
    for e in P["entries"]:
        src = e.get("chain_from")
        if src and src not in ids and Path(by_id[src]["out"]).exists():
            dst = out / "ltx" / "out" / f"r_{src}.mp4"
            if not dst.exists():
                shutil.copy(by_id[src]["out"], dst)
    R.HOLD_TIMEOUT = int(HOLD_MINUTES * 60)
    R.HOLD_BUDGET.update(talk=R.HOLD_TIMEOUT - 420.0, act=R.HOLD_TIMEOUT - 300.0)
    t0 = time.time()
    if FULL3D:      # keyframes come from the staged scene: stage first, then hold the set and check every key
        R.stage(cx, P, workers=3, seeds=3)
        R.bg_hold(cx, P)
        R.talk_checks(cx, P)
    R.check(cx, P)
    R.redraw(cx, P)
    # still failing after the redraw: the original frame, only the failing heads redrawn
    failed = set()
    for ln in (out / "log.jsonl").read_text().splitlines() if (out / "log.jsonl").exists() else []:
        d = json.loads(ln)
        if d.get("phase") == "redraw" and d.get("passed") is False:
            failed.update(d.get("ids", []))
    if failed:
        os.environ["REMAKE_STRICT_HEADONLY"] = "1"
        try:
            R.redraw_strict(cx, P, failed)
        finally:
            os.environ.pop("REMAKE_STRICT_HEADONLY", None)
    if not FULL3D:
        R.stage(cx, P, workers=3, seeds=3)
    log(f"  LTX staging + keyframes done in {(time.time() - t0) / 60:.0f} min")
    with _KernelWatch(out) as kw:
        R.ltx(cx, P)
        # video QA (cuts/zooms vs the camera path, hands, left/right, identity, lip-sync per chunk, on-screen text,
        # location/outfit on sampled frames): failing clips re-rolled once with a new seed, then listed
        summ = R.qa(cx, P, reroll=True)
        if summ.get("fail"):
            R.ltx(cx, P)
            summ = R.qa(cx, P)
    if kw.hit:
        raise SystemExit("kernel log hit during the LTX hold: stopped (see ltx_flow/KERNEL_HIT.txt)")
    if summ.get("fail"):
        log(f"  LTX QA: {len(summ['fail'])} clip(s) still flagged after one re-roll: {sorted(summ['fail'])}")
    n = 0
    for e in P["entries"]:
        c = R.new_clip(cx, e)
        if c and not Path(e["old"]).exists():
            shutil.copy(c, e["old"])
            n += 1
    log(f"  LTX: {n} clip(s) written in {(time.time() - t0) / 60:.0f} min")
    return P
