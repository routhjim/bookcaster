"""Does the lip-sync measure fail what it should? Two synthetic checks on talk clips that pass it, made from the
clips themselves (nothing is rendered; the clips and their folders are only read; everything written goes to --out).

1. Shifted sound: each clip's own sound moved later by 5, 10 and 20 frames (25 fps). The measure must follow a
   shift it can see (5, 10: a lock whose offset moved by the shift, within a frame) and must not pass one it cannot
   (20: beyond the search, so no lock or a reading under the bar).
2. The wrong voice: each clip's picture with another clip's sound (the next in the list) - a mouth moving to speech
   that is not its own. The measure must fail it. The same pairs are scored again with the voiced span cut to its
   first 10, 15, 20, 30, 44 and 60 windows: the share that still reads over the bar (and over the shift's
   confidence) per span length is what sets SYNC_ROWS_SURE from data (a few windows of an unsynced face read high
   by chance).

    python tools/sync_null_check.py --out DIR CLIP.mp4 [CLIP.mp4 ...]      (or --list FILE, one clip a line)

Each clip costs 1 + 3 + 1 SyncNet runs on the iGPU: run it there, through whatever queues iGPU work. Prints the
tables; the readings are kept in DIR/readings.json. Resumable: a reading already in DIR is not made again."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import shot_flow as S          # noqa: E402

SHIFTS = (5, 10, 20)                    # frames at 25 fps
SPANS = (10, 15, 20, 30, 44, 60)        # voiced windows kept for the wrong-voice pairs
BUCKETS = ((0, 30), (30, 60), (60, 10 ** 6))
LATER_MADE = 2                          # how the shifted clips are made (1: a timestamp offset, lost in the wav)


def _env() -> dict:
    ig = subprocess.run([str(Path.home() / "bin" / "hip-index"), "igpu"], capture_output=True, text=True).stdout.strip()
    return {**os.environ, "PYTHONPATH": str(S.SYNCNET / "_deps"), "HIP_VISIBLE_DEVICES": ig, "HSA_CU_MASK": f"{ig}:0-38"}


def _frames25(clip: Path) -> int:
    try:
        return int(float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                         "csv=p=0", str(clip)], capture_output=True, text=True).stdout) * 25) + 1
    except ValueError:
        return 0


def later(clip: Path, frames: int, out: Path) -> Path:
    """The clip with its own sound `frames` (25 fps) later; the picture is copied. The sound is really delayed
    (silence put before it): a timestamp offset alone is dropped when the sound is taken out as a wav file, so the
    shift would not reach the measure."""
    ms = int(round(frames * 1000 / 25))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-map", "0:v", "-map", "0:a", "-c:v", "copy",
                    "-af", f"adelay={ms}:all=1", "-c:a", "aac", "-shortest", str(out)], capture_output=True)
    return out


def other_voice(clip: Path, voice: Path, out: Path) -> Path:
    """The clip's picture with another clip's sound."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-i", str(voice), "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", "-shortest", str(out)], capture_output=True)
    return out


def shifted(voiced: list, frames: int) -> list:
    """A clip's voiced frames when its sound is `frames` later."""
    return ([False] * frames + list(voiced))[: len(voiced)]


def first_span(voiced: list, windows: int) -> list:
    """The voiced frames cut to the first `windows` of them."""
    out, left = [], windows
    for v in voiced:
        out.append(bool(v) and left > 0)
        left -= bool(v)
    return out


def fit(voiced: list, n: int) -> list:
    return (list(voiced) + [False] * n)[:n]


def read(clip: Path, bx, voiced: list, work: Path, ref: str, env: dict, at) -> dict:
    """One SyncNet run of `clip` in the zone bx, read by the measure over `voiced`. The run's tracks and distances
    stay in work for further readings of the same run (reread)."""
    S._syncnet_run(clip, bx, work, ref, env)
    return reread(bx, voiced, work, ref, at)


def reread(bx, voiced: list, work: Path, ref: str, at) -> dict:
    r = S.sync_read(work, ref, voiced, S.W * ((bx[2] - bx[0]) if bx else 1.0), at)
    return {k: r.get(k) for k in ("conf", "conf_raw", "offset", "lock", "basis", "why", "voiced_rows", "rows",
                                  "voiced", "whole", "track", "faces")}


def bucket(rows) -> str:
    rows = rows or 0
    return next(f"{a}-{b - 1}" if b < 10 ** 6 else f"{a}+" for a, b in BUCKETS if a <= rows < b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clips", nargs="*", type=Path, help="talk clips that pass the measure (out/<name>.mp4)")
    ap.add_argument("--list", type=Path, help="a file of clip paths, one a line")
    ap.add_argument("--out", type=Path, required=True, help="where the made clips, SyncNet's work and the readings go")
    a = ap.parse_args()
    clips = list(a.clips) + ([Path(x) for x in a.list.read_text().split()] if a.list else [])
    if len(clips) < 2:
        ap.error("two clips at least (the wrong-voice check pairs each with the next)")
    a.out.mkdir(parents=True, exist_ok=True)
    rf = a.out / "readings.json"
    R = json.loads(rf.read_text()) if rf.exists() else {}
    env, bar = _env(), S.TALK_SYNC_BAR

    def keep():
        rf.write_text(json.dumps(R, indent=1, allow_nan=False))

    for i, c in enumerate(clips):
        key = str(c)
        if R.get(key, {}).get("done") and R[key].get("later_made") == LATER_MADE:
            continue
        if R.get(key, {}).get("done"):                 # the shifted clips were made another way: only those again
            L, name = c.parent.parent, c.stem
            bx, n = S._job_speaker_box(L, name), _frames25(c)
            at = S._job_speaker_at(L, name, bx)
            voiced, _ = S._clip_voiced(L, name, c, n) if n else ([], "none")
            work = a.out / "work" / f"{i:03d}"
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            for k in SHIFTS:
                v = later(c, k, work / f"later{k}.mp4")
                R[key]["later"][str(k)] = read(v, bx, shifted(voiced, k), work, f"later{k}", env, at)
            R[key]["later_made"] = LATER_MADE
            keep()
            shutil.rmtree(work, ignore_errors=True)
            print(f"{name}: later again", flush=True)
            continue
        L, name = c.parent.parent, c.stem
        bx, n = S._job_speaker_box(L, name), _frames25(c)
        at = S._job_speaker_at(L, name, bx)
        voiced, how = S._clip_voiced(L, name, c, n) if n else ([], "none")
        work = a.out / "work" / f"{i:03d}"
        shutil.rmtree(work, ignore_errors=True)
        rec = {"clip": key, "voiced_from": how, "own": read(c, bx, voiced, work, "own", env, at), "later": {},
               "null": {}}
        for k in SHIFTS:
            v = later(c, k, work / f"later{k}.mp4")
            rec["later"][str(k)] = read(v, bx, shifted(voiced, k), work, f"later{k}", env, at)
        o = clips[(i + 1) % len(clips)]
        ov, _ = S._clip_voiced(o.parent.parent, o.stem, o, _frames25(o))
        ov = fit(ov, n)
        w = other_voice(c, o, work / "null.mp4")
        rec["null"]["voice_of"] = str(o)
        rec["null"]["all"] = read(w, bx, ov, work, "null", env, at)
        rec["null"]["spans"] = {str(k): reread(bx, first_span(ov, k), work, "null", at) for k in SPANS}
        rec["done"] = True
        rec["later_made"] = LATER_MADE
        R[key] = rec
        keep()
        shutil.rmtree(work, ignore_errors=True)
        print(f"{name}: own {rec['own']['conf']} at {rec['own']['offset']}", flush=True)

    done = [R[str(c)] for c in clips if R.get(str(c), {}).get("done")]
    strong = [r for r in done if (r["own"]["conf"] or 0) >= bar]
    print(f"\n{len(done)} clip(s) read, {len(strong)} pass the bar ({bar}) with their own sound; the checks count those")
    print("\n1. the clip's own sound, later (must follow 5 and 10, must not pass 20)")
    print(f"{'frames':>7} {'clips':>6} {'locks':>6} {'pass bar':>9} {'offset moved by the shift (+-1)':>32} {'moved, median':>14}")
    bad = []
    for k in SHIFTS:
        rs = [(r, r["later"][str(k)]) for r in strong]
        moved = [abs((x["offset"] or 0) - (r["own"]["offset"] or 0)) for r, x in rs if x["lock"]]
        follow = sum(1 for m in moved if abs(m - k) <= 1)
        passed = sum(1 for _, x in rs if (x["conf"] or 0) >= bar)
        med = sorted(moved)[len(moved) // 2] if moved else None
        print(f"{k:>7} {len(rs):>6} {sum(1 for _, x in rs if x['lock']):>6} {passed:>9} {follow:>32} {str(med):>14}")
        if k < S.SYNC_VSHIFT - S.SYNC_LOCK_EDGE and follow < len(rs):
            bad.append(f"a {k}-frame shift was not followed in {len(rs) - follow} of {len(rs)} clip(s)")
        if k >= S.SYNC_VSHIFT and passed:
            bad.append(f"a {k}-frame shift (beyond the search) passed the bar in {passed} clip(s)")
    print("\n2. the wrong voice (must fail)")
    print(f"{'voiced windows':>15} {'pairs':>6} {'read >= bar':>12} {'read >= ' + str(S.SYNC_MIN_CONF):>12} {'believed as voiced':>19}")
    rows = [("the whole span, " + b, [r["null"]["all"] for r in strong if bucket(r["null"]["all"]["voiced_rows"]) == b])
            for b in (bucket(x) for x, _ in BUCKETS)]
    rows += [(f"first {k}", [r["null"]["spans"][str(k)] for r in strong
                             if (r["null"]["spans"][str(k)]["voiced_rows"] or 0) >= min(k, S.SYNC_MIN_VOICED)])
             for k in SPANS]
    for label, xs in rows:
        hi = sum(1 for x in xs if (x["conf"] or 0) >= bar)
        mid = sum(1 for x in xs if (x["conf"] or 0) >= S.SYNC_MIN_CONF)
        vo = sum(1 for x in xs if x["basis"] == "voiced")
        print(f"{label:>15} {len(xs):>6} {hi:>12} {mid:>12} {vo:>19}")
        if hi and label.startswith("the whole"):
            bad.append(f"{hi} wrong-voice pair(s) passed the bar ({label})")
    raw = {k: [((r["null"]["spans"][str(k)].get("voiced") or {}).get("conf") or 0) for r in strong] for k in SPANS}
    print("\n   the voiced reading alone, before the agreement and offset rules (what a floor on the span must keep out):")
    for k in SPANS:
        xs = raw[k]
        print(f"{'first ' + str(k):>15} {len(xs):>6} {sum(x >= bar for x in xs):>12} {sum(x >= S.SYNC_MIN_CONF for x in xs):>12}")
    print(f"\n   SYNC_ROWS_SURE is {S.SYNC_ROWS_SURE}: it should be the shortest span at which the voiced reading "
          f"alone no longer reads over the bar here")
    print("\nFAILS:\n  " + "\n  ".join(bad) if bad else "\nboth checks hold")
    keep()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
