"""Background score for a rendered chapter: mood music under key scenes.

1. plan   - an LLM reads the chapter (paragraph-numbered) and marks the key
            scenes: where each starts/ends, its mood, and an instrumental
            music brief (instruments, tempo, key, intensity).
2. place  - paragraphs are matched to the render's <name>.timeline.json
            (written by the cast renderer) to get each cue's start/end time.
3. compose- each cue is generated at its scene's length by an ACE-Step 1.5
            server (instrumental, text-to-music).
4. mix    - cues fade in/out at scene boundaries, sit well under the voice,
            and duck further whenever someone is speaking (sidechain).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen

from .textprep import prepare_for_speech

DEFAULT_ACE_URL = "http://127.0.0.1:8001"


def paragraphs(text: str) -> list[str]:
    """The chapter's paragraphs, in the same speech-prepped form the renderer used."""
    return [p.strip() for p in re.split(r"\n{2,}", prepare_for_speech(text)) if p.strip()]


def _chat(url: str, model: str, prompt: str, max_tokens: int = 3000) -> str:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=1800) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def plan_scenes(paras: list[str], llm_url: str, model: str = "default",
                max_cues: int = 9) -> list[dict]:
    numbered = "\n\n".join(f"[P{i}] {p}" for i, p in enumerate(paras))
    prompt = (
        "You are the music supervisor for an audiobook. Below is one chapter, "
        "paragraph-numbered. Mark the KEY SCENES that deserve background "
        f"score (at most {max_cues}; leave ordinary stretches unscored — not "
        "every paragraph needs music). Scenes must not overlap and must be in "
        "order. For each give:\n"
        "- start, end: first and last paragraph numbers (inclusive)\n"
        "- scene: a few words naming it\n"
        "- mood: a few words (e.g. 'creeping dread', 'warm domestic calm')\n"
        "- music: an INSTRUMENTAL music description for a text-to-music model: "
        "genre/style, instruments, texture, dynamics — subtle underscore that "
        "sits under narration, no vocals, no drums unless the scene is action "
        "(e.g. 'sparse cinematic underscore, low cello drone, distant pizzicato, "
        "tense, slowly building').\n"
        "- bpm: tempo (40-140)\n"
        "- key: e.g. 'D minor'\n"
        "- intensity: 0.0-1.0 (how present the music should be)\n\n"
        'Reply with ONLY JSON: {"cues": [{"start": 0, "end": 3, "scene": "...", '
        '"mood": "...", "music": "...", "bpm": 70, "key": "D minor", '
        '"intensity": 0.4}]}\n\n'
        f"Chapter:\n{numbered}"
    )
    reply = _chat(llm_url, model, prompt)
    m = re.search(r"\{.*\}", reply, re.S)
    cues = json.loads(m.group(0)).get("cues", []) if m else []
    clean, last = [], -1
    for c in cues:
        try:
            s, e = int(c["start"]), int(c["end"])
        except (KeyError, TypeError, ValueError):
            continue
        s, e = max(s, last + 1), min(e, len(paras) - 1)
        if s > e:
            continue
        clean.append({**c, "start": s, "end": e})
        last = e
    return clean


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def paragraph_times(paras: list[str], timeline: list[dict]) -> list[float]:
    """Start time of every paragraph: walk the timeline in order and match
    each paragraph's opening words to the segment that speaks them."""
    times, j = [], 0
    for p in paras:
        head = _norm(p)[:40]
        found = None
        for k in range(j, min(j + 400, len(timeline))):
            seg = _norm(timeline[k]["text"])
            if seg and (head.startswith(seg[:25]) or seg.startswith(head[:25])):
                found = k
                break
        if found is None:   # quote-first paragraph etc.: take the next segment
            found = min(j, len(timeline) - 1)
        times.append(float(timeline[found]["t"]))
        j = found + 1
    return times


def place_cues(cues: list[dict], paras: list[str], timeline: list[dict],
               total: float) -> list[dict]:
    starts = paragraph_times(paras, timeline)
    out = []
    for c in cues:
        t0 = starts[c["start"]]
        t1 = starts[c["end"] + 1] if c["end"] + 1 < len(starts) else total
        if t1 - t0 >= 10:
            out.append({**c, "t0": round(t0, 2), "t1": round(t1, 2)})
    return out


def estimated_starts(paras: list[str], chars_per_sec: float,
                     pause_s: float = 0.65) -> tuple[list[float], float]:
    """Paragraph start times guessed from text length alone (for composing
    music before the narration exists): (starts, total seconds)."""
    starts, t = [], 0.0
    for p in paras:
        starts.append(t)
        t += len(p) / chars_per_sec + pause_s
    return starts, t


def plan_only(text: str, work: Path, llm_url: str, llm_model: str = "default",
              log=print) -> list[dict]:
    """Plan the scenes now (the LLM's part) and cache them in *work*; the
    later --ahead / score runs restore the plan and need no LLM."""
    work.mkdir(parents=True, exist_ok=True)
    plan_path = work / "plan.json"
    if plan_path.exists():
        scenes = json.loads(plan_path.read_text())
        log(f"  score plan already cached ({len(scenes)} cue(s))")
        return scenes
    log("  planning scenes...")
    scenes = plan_scenes(paragraphs(text), llm_url, llm_model)
    plan_path.write_text(json.dumps(scenes, indent=1))
    log(f"  planned {len(scenes)} cue(s)")
    return scenes


def compose_ahead(text: str, work: Path, llm_url: str, llm_model: str = "default",
                  ace_url: str = DEFAULT_ACE_URL, chars_per_sec: float = 15.0,
                  margin: float = 1.2, log=print) -> list[dict]:
    """Compose every cue from ESTIMATED scene lengths (x *margin*) so the
    music can be generated on its own GPU while the voices render on
    another. The normal score run afterwards re-times the cues to the real
    timeline, reuses each cue that is long enough (the mix trims and fades
    the rest) and recomposes only one that came out too short."""
    work.mkdir(parents=True, exist_ok=True)
    paras = paragraphs(text)
    plan_path = work / "plan.json"
    if plan_path.exists():
        scenes = [{k: v for k, v in c.items() if k not in ("t0", "t1")}
                  for c in json.loads(plan_path.read_text())]
        log(f"  score plan restored ({len(scenes)} cue(s))")
    else:
        log("  planning scenes...")
        scenes = plan_scenes(paras, llm_url, llm_model)
    starts, total = estimated_starts(paras, chars_per_sec)
    cues = []
    for c in scenes:
        t0 = starts[c["start"]]
        t1 = starts[c["end"] + 1] if c["end"] + 1 < len(starts) else total
        if t1 - t0 >= 10:
            cues.append({**c, "t0": round(t0, 2), "t1": round(t0 + (t1 - t0) * margin, 2)})
    plan_path.write_text(json.dumps(cues, indent=1))
    for n, c in enumerate(cues, 1):
        f = work / f"cue{n:02d}.wav"
        if f.exists() and audio_seconds(f) >= c["t1"] - c["t0"] + 3:
            continue
        t = time.time()
        compose(ace_url, c, f)
        log(f"  composed cue {n} ahead ({c['t1'] - c['t0']:.0f} s, est.) in {time.time() - t:.0f} s")
    log(f"  {len(cues)} cue(s) ready ahead of the render (estimated {total / 60:.1f} min)")
    return cues


def compose(ace_url: str, cue: dict, out: Path, seed: int = 7, timeout: float = 1800) -> Path:
    """One instrumental cue at the scene's length from an ACE-Step server."""
    duration = max(10.0, min(600.0, cue["t1"] - cue["t0"] + 4))   # +4 s for the fade-out
    body = {"prompt": cue["music"], "lyrics": "[Instrumental]", "instrumental": True,
            "audio_duration": duration, "bpm": int(cue.get("bpm") or 0) or None,
            "key_scale": cue.get("key") or "", "batch_size": 1,
            "use_random_seed": False, "seed": seed, "thinking": False,
            "use_cot_caption": False, "audio_format": "wav"}
    req = Request(f"{ace_url}/release_task", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    task = json.loads(urlopen(req, timeout=120).read())
    task_id = (task.get("data") or {}).get("task_id") or task.get("task_id")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        q = Request(f"{ace_url}/query_result", data=json.dumps({"task_id_list": [task_id]}).encode(),
                    headers={"Content-Type": "application/json"})
        item = json.loads(urlopen(q, timeout=60).read())["data"][0]
        if item["status"] == 2:
            raise RuntimeError(f"ACE-Step failed: {item.get('result')}")
        if item["status"] == 1:
            result = json.loads(item["result"])[0]
            audio = urlopen(f"{ace_url}{result['file']}", timeout=300).read()
            out.write_bytes(audio)
            return out
    raise RuntimeError("ACE-Step timed out")


def mix(voice: Path, cues: list[dict], out: Path, level_db: float = -20.0,
        bitrate: str = "192k") -> None:
    """Lay cues under the narration: fades at scene edges, a base level well
    below the voice (scaled by each cue's intensity), and sidechain ducking."""
    inputs = ["-i", str(voice)]
    chains, labels = [], []
    for n, c in enumerate(cues, 1):
        inputs += ["-i", str(c["file"])]
        length = c["t1"] - c["t0"]
        gain = level_db + 6 * (float(c.get("intensity", 0.5)) - 0.5)   # ±3 dB by intensity
        delay = int(c["t0"] * 1000)
        chains.append(
            f"[{n}:a]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"atrim=0:{length + 3:.2f},afade=t=in:d=3,afade=t=out:st={max(0, length - 1):.2f}:d=4,"
            f"volume={gain:.1f}dB,adelay={delay}|{delay}[m{n}]")
        labels.append(f"[m{n}]")
    graph = ";".join(chains)
    graph += (f";{''.join(labels)}amix=inputs={len(labels)}:normalize=0:dropout_transition=0[music]"
              # pan, not aformat: a mono voice is copied to both channels at
              # unity (the default upmix drops it 3 dB)
              ";[0:a]aresample=48000,aformat=channel_layouts=mono,pan=stereo|c0=c0|c1=c0,"
              "asplit=2[v1][v2]"
              ";[music][v1]sidechaincompress=threshold=0.03:ratio=6:attack=80:release=900[ducked]"
              ";[v2][ducked]amix=inputs=2:normalize=0:duration=first[out]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph,
                    "-map", "[out]", "-c:a", "libmp3lame", "-b:a", bitrate, str(out)], check=True)


def audio_seconds(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def score_chapter(text: str, audio: Path, timeline_path: Path, out: Path, llm_url: str,
                  llm_model: str = "default", ace_url: str = DEFAULT_ACE_URL,
                  work_dir: Path | None = None, log=print) -> list[dict]:
    """Plan, compose and mix; caches the plan and cues in *work_dir*."""
    work = work_dir or out.with_suffix(".score")
    work.mkdir(parents=True, exist_ok=True)
    paras = paragraphs(text)
    plan_path = work / "plan.json"
    timeline = json.loads(timeline_path.read_text())
    if plan_path.exists():
        # The scenes are paragraph ranges; their times come from this
        # render's timeline (a re-render with new pacing moves every cue).
        scenes = [{k: v for k, v in c.items() if k not in ("t0", "t1")}
                  for c in json.loads(plan_path.read_text())]
        log(f"  score plan restored ({len(scenes)} cue(s))")
    else:
        log("  planning scenes...")
        scenes = plan_scenes(paras, llm_url, llm_model)
    cues = place_cues(scenes, paras, timeline, audio_seconds(audio))
    plan_path.write_text(json.dumps(cues, indent=1))
    for n, c in enumerate(cues, 1):
        log(f"  cue {n}: {c['t0']/60:5.1f}-{c['t1']/60:5.1f} min  {c['scene']} — {c['mood']}")
    for n, c in enumerate(cues, 1):
        f = work / f"cue{n:02d}.wav"
        # A cached cue is reused if it still covers its scene (+3 s tail
        # the mix fades out over); a longer scene needs a longer cue.
        if f.exists() and audio_seconds(f) < c["t1"] - c["t0"] + 3:
            log(f"  cue {n} is shorter than its scene now; recomposing")
            f.unlink()
        if not f.exists():
            t = time.time()
            compose(ace_url, c, f)
            log(f"  composed cue {n} ({c['t1'] - c['t0']:.0f} s) in {time.time() - t:.0f} s")
        c["file"] = str(f)
    mix(audio, cues, out)
    log(f"  wrote {out}")
    return cues
