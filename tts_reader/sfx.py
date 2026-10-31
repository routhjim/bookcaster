"""Sound effects under a chapter: ambience beds for stretches of the story
(a coal fire and a clock, a London street, wind on the moor) and spot effects
at the moment the text says they happen (a door dashed open, hooves on
gravel, a low whistle in the night).

The language model reads the recording's timeline and marks them;
MOSS-SoundEffect v2.0 (Apache 2.0; tools/sfx_gen.py, on the music card beside
ACE) generates them; they're mixed into the finished chapter, the beds ducking
whenever someone speaks (as the music does). Cached in work/sfx/<chapter>/.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from . import video

MOSS_PY = Path(os.environ.get("BOOKCASTER_SFX_PYTHON",
                              Path.home() / "src" / "MOSS-TTS" / "moss_soundeffect_v2" / ".venv" / "bin" / "python"))
GEN = Path(__file__).resolve().parents[1] / "tools" / "sfx_gen.py"
SFX_GPU_BUS = os.environ.get("BOOKCASTER_SFX_BUS", "65")      # the music card (ACE + the drawer live there)
PLAN_VERSION = 2       # 2 (10-31): each spot's cue words and its line's end, for word-level placement in a video
# a sound that lands on one frame (a video can move it onto the picture's impact, tools/shot_flow.py)
IMPACT = re.compile(r"\b(knock|knocks|knocking|slam|slams|slammed|bang|bangs|clang|clangs|crash|crashes|thud|thuds|"
                    r"smash|shatter|shatters|gunshot|shot|strike|strikes|struck|snap|snaps|slap|thump|dashed)\b", re.I)
SPOT_DB = {"soft": -8.0, "medium": -3.0, "loud": 1.0}        # after normalising each effect to -22 LUFS
BED_DB = -12.0                                               # after normalising each bed to -28 LUFS; ducked
MAX_SPOT_S, MAX_BED_S = 8.0, 30.0


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def plan_sfx(timeline: list[dict], total: float, llm_url: str, model: str) -> dict:
    lines = []
    for n, s in enumerate(timeline):
        text = re.sub(r"\[[^\]]*\]", "", str(s.get("text", ""))).strip()
        if text:
            lines.append(f"[{n}] {_mmss(float(s['t']))} {s.get('speaker', 'narrator')}: {text[:220]}")
    prompt = (
        "You are the sound designer for a full-cast audiobook. Below is the recording, one numbered "
        "line per spoken segment (number, time, speaker, words). Add sound effects under it.\n\n"
        "1. beds: the continuous surroundings of each stretch of the story where the place stays the "
        "same (indoors by a coal fire with a ticking clock; a busy London street; a carriage ride on a "
        "country road; wind over open fields; a silent house at night). Give the first and last line "
        "of the stretch. Leave a stretch without a bed if nothing fits. No music, no voices.\n"
        "2. spots: single sounds the story says HAPPEN at that moment (a door dashed open, footsteps "
        "on stairs, a knock, hooves and wheels arriving, a clang of metal, a whistle, a scream, a match "
        "struck, a clock striking). Give the line where it happens. Only real, present sounds: not "
        "remembered or imagined ones, not figures of speech, not anything inside a quotation that is "
        "only being told about. At most about one per 30 seconds, only where it helps. With each spot, "
        "the 2-5 words of that line where the sound happens, copied exactly from it.\n"
        "Describe every sound concretely, as heard (what makes it, the surface, the space), 4-14 words.\n\n"
        'Reply with ONLY JSON: {"beds": [{"from": 3, "to": 40, "sound": "..."}], '
        '"spots": [{"at": 12, "words": "...", "sound": "...", "seconds": 2.5, "loud": "soft|medium|loud"}]}\n\n'
        + "\n".join(lines))
    from .score import _chat
    reply = _chat(llm_url, model, prompt, max_tokens=6000)
    m = re.search(r"\{.*\}", reply, re.S)
    data = json.loads(m.group(0)) if m else {}
    times = [float(s["t"]) for s in timeline] + [total]

    def at(i) -> float | None:
        try:
            i = int(i)
        except (TypeError, ValueError):
            return None
        return times[i] if 0 <= i < len(timeline) else None

    beds, spots = [], []
    for b in data.get("beds", []):
        t0, t1 = at(b.get("from")), at(b.get("to"))
        if t0 is None or t1 is None or not str(b.get("sound", "")).strip():
            continue
        t1 = times[int(b["to"]) + 1]                       # to the end of the last line
        if t1 - t0 >= 8:
            beds.append({"t0": round(t0, 2), "t1": round(t1, 2), "sound": str(b["sound"]).strip()})
    for s in data.get("spots", []):
        t = at(s.get("at"))
        if t is None or not str(s.get("sound", "")).strip():
            continue
        secs = max(0.8, min(MAX_SPOT_S, float(s.get("seconds") or 2.5)))
        loud = s.get("loud") if s.get("loud") in SPOT_DB else "medium"
        spots.append({"t": round(t, 2), "t_end": round(times[int(s["at"]) + 1], 2),
                      "words": str(s.get("words") or "").strip(), "seconds": round(secs, 1),
                      "sound": str(s["sound"]).strip(), "loud": loud})
    beds.sort(key=lambda b: b["t0"])
    for a, b in zip(beds, beds[1:]):                       # no overlapping beds
        a["t1"] = min(a["t1"], b["t0"])
    return {"version": PLAN_VERSION, "beds": [b for b in beds if b["t1"] - b["t0"] >= 8],
            "spots": sorted(spots, key=lambda s: s["t"])}


def ensure_plan(work: Path, timeline_path: Path, total: float, llm_url: str, model: str, log) -> dict:
    p = work / "plan.json"
    if p.exists():
        plan = json.loads(p.read_text())
        if plan.get("version") == PLAN_VERSION:
            return plan
    log("  planning sound effects...")
    plan = plan_sfx(json.loads(timeline_path.read_text()), total, llm_url, model)
    work.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    log(f"  planned {len(plan['beds'])} ambience bed(s) and {len(plan['spots'])} spot effect(s)")
    return plan


def _fx_name(kind: str, n: int, sound: str, secs: float) -> str:
    """A file named by what it holds (10-31: named by its index, a re-planned chapter played the old plan's sound
    for the new one's spot n)."""
    import hashlib
    return f"{kind}{n:03d}_{hashlib.sha1(f'{sound}|{secs:.1f}'.encode()).hexdigest()[:8]}.wav"


def generate(plan: dict, work: Path, log) -> None:
    jobs = []
    for n, b in enumerate(plan["beds"]):
        b["file"] = str(work / _fx_name("bed", n, b["sound"], min(MAX_BED_S, b["t1"] - b["t0"])))
        jobs.append({"id": f"bed{n:03d}", "prompt": b["sound"], "path": b["file"],
                     "seconds": min(MAX_BED_S, b["t1"] - b["t0"]), "seed": 11 + n})
    for n, s in enumerate(plan["spots"]):
        s["file"] = str(work / _fx_name("spot", n, s["sound"], s["seconds"]))
        jobs.append({"id": f"spot{n:03d}", "prompt": s["sound"], "path": s["file"],
                     "seconds": s["seconds"], "seed": 101 + n})
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    if not todo:
        return
    jf = work / "jobs.json"
    jf.write_text(json.dumps(todo, indent=1, ensure_ascii=False))
    log(f"  generating {len(todo)} sound effect(s)...")
    t = time.time()
    env = {**os.environ, "HIP_VISIBLE_DEVICES": video.hip_index(SFX_GPU_BUS),
           "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True"}
    # its own memory scope: loading the model in the job's 8 GB unit got the whole job OOM-killed (10-17)
    avail = video._avail_gb()
    scope = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={max(8.0, min(16.0, avail - 2.0)):.1f}G",
             "-p", "MemorySwapMax=0"]
    with open(work / "gen.log", "w") as err, video._music_paused(log):   # ACE holds 13 GB of that card
        r = subprocess.run([*scope, str(MOSS_PY), "-I", str(GEN), str(jf)], env=env, stdout=err,
                           stderr=subprocess.STDOUT)
    left = sum(not Path(j["path"]).exists() for j in todo)
    if r.returncode or left:
        tail = (work / "gen.log").read_text(errors="replace").strip().splitlines()[-3:]
        raise RuntimeError(f"sound effects: {left} not generated: " + " | ".join(tail))
    log(f"  generated {len(todo)} sound effect(s) in {time.time() - t:.0f} s")


def mix(base: Path, voice: Path, plan: dict, out: Path, bitrate: str = "192k") -> None:
    """The finished chapter (*base*) plus the effects: spots at fixed levels,
    beds looped over their stretch and ducked under the voice track."""
    items = [("bed", b) for b in plan["beds"] if Path(b.get("file", "")).exists()] + \
            [("spot", s) for s in plan["spots"] if Path(s.get("file", "")).exists()]
    if not items:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(base), "-c", "copy", str(out)], check=True)
        return
    inputs = ["-i", str(base), "-i", str(voice)]
    chains, beds, spots = [], [], []
    for n, (kind, x) in enumerate(items, 2):
        inputs += ["-i", x["file"]]
        if kind == "bed":
            length = x["t1"] - x["t0"]
            d = int(x["t0"] * 1000)
            chains.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=mono,loudnorm=I=-28:TP=-3,"
                          f"aloop=loop=-1:size=2147483647,atrim=0:{length:.2f},asetpts=N/SR/TB,"
                          f"afade=t=in:d=2.5,afade=t=out:st={max(0.0, length - 3):.2f}:d=3,"
                          f"volume={BED_DB}dB,pan=stereo|c0=c0|c1=c0,adelay={d}|{d}[e{n}]")
            beds.append(f"[e{n}]")
        else:
            d = int(x["t"] * 1000)
            chains.append(f"[{n}:a]aresample=48000,aformat=channel_layouts=mono,loudnorm=I=-22:TP=-2,"
                          f"afade=t=out:st={max(0.0, x['seconds'] - 0.4):.2f}:d=0.4,"
                          f"volume={SPOT_DB[x['loud']]}dB,pan=stereo|c0=c0|c1=c0,adelay={d}|{d}[e{n}]")
            spots.append(f"[e{n}]")
    graph = ";".join(chains)
    layers = ["[0:a]aresample=48000,aformat=channel_layouts=stereo[base]"]
    mixed = ["[base]"]
    if beds:
        layers.append(f"{''.join(beds)}amix=inputs={len(beds)}:normalize=0:dropout_transition=0[beds]")
        layers.append("[1:a]aresample=48000,aformat=channel_layouts=mono,pan=stereo|c0=c0|c1=c0[key]")
        layers.append("[beds][key]sidechaincompress=threshold=0.02:ratio=8:attack=60:release=1200[ducked]")
        mixed.append("[ducked]")
    if spots:
        layers.append(f"{''.join(spots)}amix=inputs={len(spots)}:normalize=0:dropout_transition=0[spots]")
        mixed.append("[spots]")
    layers.append(f"{''.join(mixed)}amix=inputs={len(mixed)}:normalize=0:duration=first[out]")
    graph += ";" + ";".join(layers)
    tmp = out.with_name("." + out.name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[out]",
                    "-c:a", "libmp3lame", "-b:a", bitrate, "-f", "mp3", str(tmp)], check=True)
    tmp.replace(out)


def add_sfx(final: Path, voice: Path, timeline_path: Path, work: Path, llm_url: str, model: str,
            log=print) -> dict:
    """Plan (cached), generate (what's missing) and mix effects into *final*
    in place. The pre-effects mix is kept as <final>.nosfx.mp3 so a re-run
    starts from it rather than stacking effects twice."""
    total = video.audio_seconds(voice)
    plan = ensure_plan(work, timeline_path, total, llm_url, model, log)
    generate(plan, work, log)
    clean = final.with_name(final.stem + ".nosfx.mp3")
    if not clean.exists() or final.stat().st_mtime > clean.stat().st_mtime + 1:
        final.replace(clean)              # a fresh mix (or the first run): it's the clean base
    mix(clean, voice, plan, final)
    log(f"  mixed {len(plan['beds'])} bed(s) and {len(plan['spots'])} spot effect(s) into {final.name}")
    return plan
