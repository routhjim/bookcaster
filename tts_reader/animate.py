"""Animated shots for a chapter's picture video (levels "moments" and "animated").

Two SkyReels-V3 video models do the animating, run by their runners on the
two 7900 XTX cards (BOOKCASTER_SKYREELS_DIR, default ~/src/SkyReels-V3):

- talking shots (run_a2v.py on the bus-05 card, its VAE on the iGPU via
  vae_server.py, all shots in one batch): a character's
  spoken line, lip-synced to their recorded voice, starting from a picture
  that shows only them (one of the chapter's pictures, or a portrait of them
  in that place drawn for the purpose);
- action shots (run_split.py --fp8 --jobs, one batch per card): 5-second
  moving scenes made from a picture's characters and place references.

"moments" animates a few of each; "animated" covers the whole chapter: every
run of one character's speech becomes a talking shot and everything else
becomes chained action clips of the picture on screen.

Shots are kept in work/video/<chapter>/anim/ (one .mp4 each + manifest.json),
named by kind and start time, so a long job resumes where it stopped;
video.build splices them over the stills.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import video

SKYREELS = Path(os.environ.get("BOOKCASTER_SKYREELS_DIR", Path.home() / "src" / "SkyReels-V3"))
SKYREELS_PY = Path(os.environ.get("BOOKCASTER_SKYREELS_PYTHON", SKYREELS / ".venv" / "bin" / "python"))
HIP_INDEX = Path(os.environ.get("BOOKCASTER_HIP_INDEX", Path.home() / "bin" / "hip-index"))
STACK = Path(__file__).resolve().parents[1] / "tools" / "fish_stack.sh"
INDUCTOR_CACHE = Path.home() / ".cache" / "torchinductor-skyreels"
VAE_SOCK = Path(os.environ.get("BOOKCASTER_VAE_SOCK", "/tmp/skyreels-vae.sock"))
VAE_UNIT = "vae-igpu-server"

# Measured 2026-10-16 (480p): render minutes per second of speech for a
# talking shot, and per 5-second action clip on one card.
# 10-18: x0.81 with 1:1 sparse attention (window 148 -> 120 s)
TALK_MIN_PER_S = float(os.environ.get("BOOKCASTER_TALK_MIN_PER_S", "0.73"))
# action shots: "i2v" = Wan2.1 I2V 14B 480P, LightX2V 4-step distilled, from the keyframe (run_i2v.py,
# ~2.5 min a clip incl. RIFE to 24 fps; chosen 10-17 over the reference model, which re-imagined
# scenes, and the V2V extension model, 10.5 min); "r2v" = the old reference-to-video runner
# "i2v22" = Wan2.2 I2V A14B (LightX2V 4-step), its two experts pipelined across the cards (run_i2v22.py),
# from a start and, where planned, an end keyframe: the 2.1 claw-block clip became another dragon and a
# jumble, 2.2 played the parry; ~41 s a clip with 1:1 sparse (10-19)
# default i2v22 since 10-19 (the engine every knight sample ran on)
# "ltx" (10-20, default) = LTX-2.5 on both XTXs (tools/ltx_flow.py): LTX I2V from the checked keyframe; the staging
# chain (motion -> blockout render -> keyframe -> Mode 2 depth 0.35) only with BOOKCASTER_LTX_STAGE=1
ACTION_ENGINE = os.environ.get("BOOKCASTER_ACTION_ENGINE", "ltx")
# talking shots: "infinitetalk" (run_infinitetalk.py, 10-17) or "a2v" (SkyReels A2V 19B: intermittent NaNs)
# or "ltx" (10-20, default: LTX A2Vid, <= 5 s chunks, camera lock, face/age-checked first frame)
TALK_ENGINE = os.environ.get("BOOKCASTER_TALK_ENGINE", "ltx")
if TALK_ENGINE == "ltx" and "BOOKCASTER_TALK_MIN_PER_S" not in os.environ:
    TALK_MIN_PER_S = 0.8          # ~190 s per <= 5 s chunk, one job on both cards (10-20 knight ch1)
# action shots' attention: "D:S" = Radial sparse self-attention dense within +-D latent frames, the first
# S of the 4 steps dense (they set the composition: the clip keeps dense's picture); "0" = dense. 1:1
# chosen by the user 10-18 over pure D1 (motion wandered: John left the frame) - step 17.3 -> ~19 s avg
ACTION_SPARSE = os.environ.get("BOOKCASTER_ACTION_SPARSE", "1:1")
# the same for InfiniteTalk's talking shots (the pinned motion frames stay global): 10-18 D1 window
# 150 -> 113 s, lip shapes frame-for-frame with dense; 1:1 like the action shots
TALK_SPARSE = os.environ.get("BOOKCASTER_TALK_SPARSE", "1:1")


def _sparse_args(spec: str) -> list[str]:
    """ "D:S" -> the runners' --sparse D --dense-steps S ("0" or "" = dense)."""
    d, _, steps = spec.partition(":")
    return ["--sparse", d.strip(), "--dense-steps", steps.strip() or "0"] if d.strip() not in ("", "0") else []
# per clip with 1:1 sparse + the next clip's conditioning made during the render (10-18: ~83 s render,
# the 20-45 s CLIP + encode hidden; was 2.5 dense with the card idle during conditioning)
ACTION_MIN_PER_CLIP = float(os.environ.get("BOOKCASTER_ACTION_MIN_PER_CLIP",
                                           "1.5" if ACTION_ENGINE in ("i2v", "i2v22") else
                                           "1.4" if ACTION_ENGINE == "ltx" else "7.8"))   # ltx I2V: ~80 s a clip
ACTION_S = 5.0
MOMENTS_TALK = int(os.environ.get("BOOKCASTER_MOMENTS_TALK", "4"))
MOMENTS_ACTION = int(os.environ.get("BOOKCASTER_MOMENTS_ACTION", "4"))
# InfiniteTalk's first window costs the same for 1 s as for 3 s; short lines ("You see it, Watson?")
# left as stills meant nobody's lips moved (user, 10-17)
TALK_MIN_S, TALK_MAX_S = float(os.environ.get("BOOKCASTER_TALK_MIN_S", "1.0")), 30.0
MIN_GAP_S = 2.0                        # a hole shorter than this keeps the still

# Words the video model draws as floating particles (one test clip "snowed":
# "firelight flickers" became drifting embers).
PARTICLES = re.compile(r"[^.;]*\b(flicker\w*|ember\w*|spark\w*|crackl\w*|dust|smoke|smok\w+|ash|"
                       r"snow\w*|mist\w*|haze|glitter\w*)\b[^.;]*[.;]?", re.I)
# Only what the scene describes: "the people move naturally" made a dead man
# move and turned "gazing at each other" into a stiff line-up facing the camera.
# The image-to-video model starts from the drawn keyframe, so the old "slow and small, the camera holds
# still" (which reined in the reference model's inventions) only damped it (10-17)
# a static camera: told "the cloud drifts higher", the Ascension shot tilted up into empty sky and the
# two shots continued from it invented a giant blue face in the clouds (10-17)
MOTION = ("The camera stays completely still: no pan, tilt or zoom; everyone stays in the frame. "
          "Anyone dead, asleep or unconscious stays completely still. Still, clear air; steady light. "
          "Each person has exactly two arms, two hands and two legs. A sword or tool stays ONE object in the "
          "same one hand throughout; it never doubles or moves to both hands.")
# the sword: Wan2.2's parry ended two-handed with a second blade in the other hand (user, 10-19)


# --- which shots -------------------------------------------------------------------

def speaker_key(speaker: str, chars: dict, labels=()) -> str | None:
    """The LOOK character the timeline's speaker label names; None for the
    narrator or anyone without a look. A label two characters share (the
    recording calls Helen 'Stoner'; Julia Stoner has her own label 'Julia')
    goes to the one no other label claims."""
    def hits(label):
        s = label.lower().strip()
        return [k for k in chars if s == k.lower() or s in {n.lower() for n in video._names(k)}]

    if not speaker.strip() or speaker.lower() in ("narrator", "driver"):
        return None
    found = hits(speaker)
    if len(found) > 1:
        claimed = {h[0] for lb in labels if lb != speaker and len(h := hits(lb)) == 1}
        found = [k for k in found if k not in claimed]
    return found[0] if len(found) == 1 else None


def dialogue_runs(timeline: list[dict], total: float) -> list[dict]:
    """Maximal runs of one speaker's quoted lines with nobody else between."""
    runs: list[dict] = []
    for a, b in zip(timeline, timeline[1:] + [None]):
        t1 = float(b["t"]) if b else total
        if a.get("kind") != "quote" or a.get("speaker", "narrator") == "narrator":
            continue
        if runs and runs[-1]["speaker"] == a["speaker"] and abs(runs[-1]["t1"] - float(a["t"])) < 0.05:
            runs[-1]["t1"] = t1
            runs[-1]["text"] += " " + str(a.get("text", ""))
        else:
            runs.append({"speaker": a["speaker"], "t0": float(a["t"]), "t1": t1,
                         "text": str(a.get("text", ""))})
    return runs


def picture_at(pics: list[dict], t: float) -> dict | None:
    for p in pics:
        if p["t0"] <= t < p["t1"]:
            return p
    return None


def listener(speaker: str, t: float, pics: list[dict], timeline: list[dict], chars: dict,
             labels=()) -> str | None:
    """Who the speaker is talking to: the person most recently pictured with
    them (the scene picture itself first, back up to 5 minutes), preferring
    one with a line nearby. Not simply the others in the scene picture: in
    the Speckled Band's climax that is the dead Roylott, while Holmes is
    talking to Watson (whose lines there are labelled Narrator)."""
    # turn-taking first: the nearest line by someone else, within a minute either way, is who they are
    # talking to. Searching back through the pictures, Edric on the mountain "spoke to" Hulda from the
    # forge scene and the dragon and the princess to each other, all three to Edric really (10-18).
    turns = sorted((abs(float(e["t"]) - t), speaker_key(str(e.get("speaker", "")), chars, labels))
                   for e in timeline if e.get("kind") == "quote" and abs(float(e["t"]) - t) < 20)
    for _, who in turns:
        if who and who != speaker and who in chars:
            return who
    near = {speaker_key(str(e.get("speaker", "")), chars, labels) for e in timeline
            if e.get("kind") == "quote" and abs(float(e["t"]) - t) < 90}
    here = picture_at(pics, t) or {}
    for p in reversed([p for p in pics if p["t0"] <= t]):
        if p["t1"] < t - 300 or (here.get("where") and p.get("where") != here.get("where")):
            break                          # not past a change of place: Hulda's forge isn't the mountain
        others = [n for n in p.get("who", []) if n != speaker]
        if speaker in p.get("who", []) and others:
            # only someone who talks with them: a speech to a crowd is a close-up, not a two-shot with
            # whoever was pictured beside the speaker (Peter to the 120 had a silent John, 10-18)
            # ... the one who spoke closest in time: Maren's shout from the ledge is to Edric, who spoke
            # just before, not to the dragon (10-18)
            spoke = {n: min((abs(float(e["t"]) - t) for e in timeline if e.get("kind") == "quote" and
                             speaker_key(str(e.get("speaker", "")), chars, labels) == n), default=1e9)
                     for n in others if n in near}
            return min(spoke, key=spoke.get) if spoke else None
    return None


def ask_listeners(shots: list[dict], timeline: list[dict], total: float, chars: dict, work: Path,
                  llm_url: str | None, model: str, log=print) -> None:
    """Who each talking shot's speaker addresses, read from the passage by the language model (the text
    usually says: "he told the king", "shouted from the ledge" to the knight fighting below); cached in
    anim/listeners.json. Turn-taking and the scene picture guessed the knight was talking to Edric, Hulda
    to the dragon and the princess to the dragon (10-18). Without the model the guesses stay."""
    from .score import _chat
    f = work / "anim" / "listeners.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    names = sorted(chars)
    changed = False
    for s in shots:
        if s["kind"] != "talk":
            continue
        ctx = _shot_text(max(0.0, s["t"] - 40), s["t"] + s["dur"] + 20, timeline, total)
        key = f"{s['id']}|{s['speaker']}|{_hash(ctx)}"
        if key not in cache and llm_url:
            q = (f"Passage from a story:\n{ctx}\n\nIn it, {s['speaker']} speaks the words heard around "
                 f"{s['t']:.0f} s. Who is {s['speaker']} speaking TO in those words? Choose one name from: "
                 f"{', '.join(n for n in names if n != s['speaker'])}; or \"none\" if they speak to a crowd, to "
                 f"themselves, or to someone not in that list. Reply with only JSON: {{\"to\": \"<name or none>\"}}")
            try:
                r = _chat(llm_url, model, q, max_tokens=60) or ""
                m = re.findall(r"\{[^{}]*\}", r)
                to = json.loads(m[-1]).get("to") if m else None
                cache[key] = to if to in chars else "none"
                changed = True
            except Exception:
                pass
        if key in cache:
            s["listener"] = None if cache[key] == "none" else cache[key]
    if changed:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(cache, indent=1))
        log(f"  asked who each of {sum(s['kind'] == 'talk' for s in shots)} talking shots speaks to")


def _motion_prompt(p: dict, style: str) -> str:
    body = PARTICLES.sub(" ", p["prompt"].replace(style, ""))
    body = re.sub(r"\s{2,}", " ", body).strip(" .,;")
    return f"{body}. {MOTION} {style}."


def plan_shots(level: str, pics: list[dict], timeline: list[dict], total: float, plan: dict,
               style: str) -> list[dict]:
    """The shots to animate, in time order, each {id, kind, t, dur, pic, ...}."""
    chars, _ = video.looks(plan)
    labels = sorted({str(x.get("speaker", "")) for x in timeline})
    talks = []
    for r in dialogue_runs(timeline, total):
        key = speaker_key(r["speaker"], chars, labels)
        p = picture_at(pics, r["t0"])
        if key is None or p is None or r["t1"] - r["t0"] < TALK_MIN_S:
            continue
        t = r["t0"]
        while r["t1"] - t >= TALK_MIN_S:            # long speeches: several shots
            dur = min(TALK_MAX_S, r["t1"] - t)
            p = picture_at(pics, t) or p
            talks.append({"kind": "talk", "t": round(t, 2), "dur": round(dur, 2), "pic": p["id"],
                          "speaker": key, "where": p.get("where", ""),
                          "listener": listener(key, t, pics, timeline, chars, labels),
                          "own_picture": p.get("who") == [key]})
            t += dur
    if level == "moments":
        own = sorted((s for s in talks if s["own_picture"] and s["dur"] <= 12),
                     key=lambda s: -s["dur"])
        talks = _spread(own, MOMENTS_TALK, total)
        people = [p for p in pics if p.get("who")]
        actions = [{"kind": "action", "t": p["t0"], "dur": ACTION_S, "pic": p["id"]}
                   for p in _spread(people, MOMENTS_ACTION, total, key="t0")]
    else:
        actions, busy = [], [(s["t"], s["t"] + s["dur"]) for s in talks]
        for p in pics:            # everything not covered by a talking shot
            for a, b in _holes(p["t0"], p["t1"], busy):
                # an empty room needs one moving shot; its still pans for the rest
                n = 1 if not p.get("who") else math.ceil((b - a) / ACTION_S)
                for k in range(n):
                    t = a + k * ACTION_S
                    if b - t >= MIN_GAP_S:
                        actions.append({"kind": "action", "t": round(t, 2),
                                        "dur": round(min(ACTION_S, b - t), 2), "pic": p["id"]})
    by_id = {p["id"]: p for p in pics}
    shots = sorted(talks + actions, key=lambda s: s["t"])
    for s in shots:
        s["id"] = f"{s['kind']}-{int(round(s['t'] * 10)):06d}"
        p = by_id[s["pic"]]
        if s["kind"] == "action":
            s["prompt"] = _motion_prompt(p, style)
            s["seed"] = int(p.get("seed", 1)) + int(s["t"] * 10)
        else:
            s["prompt"] = (f"{s['speaker']} speaks, looking slightly to one side, hands still. "
                           f"Still, clear air; steady light. Static camera. {style}.")
    return shots


def _spread(items: list[dict], n: int, total: float, key: str = "t") -> list[dict]:
    """Up to n items, one from each of n equal slices of the chapter."""
    picked = []
    for k in range(n):
        lo, hi = total * k / n, total * (k + 1) / n
        inside = [i for i in items if lo <= i[key] < hi and i not in picked]
        if inside:
            picked.append(inside[0])
    return sorted(picked, key=lambda i: i[key])


def _holes(a: float, b: float, busy: list[tuple]) -> list[tuple]:
    out, t = [], a
    for s, e in sorted(busy):
        if e <= t or s >= b:
            continue
        if s - t >= MIN_GAP_S:
            out.append((t, s))
        t = max(t, e)
    if b - t >= MIN_GAP_S:
        out.append((t, b))
    return out


def estimate_minutes(level: str, total: float, dialogue_s: float, n_pics: int) -> float:
    """Box minutes for the animation alone (two cards)."""
    if level == "moments":
        talk_s = MOMENTS_TALK * 8.0
        return talk_s * TALK_MIN_PER_S + math.ceil(MOMENTS_ACTION / 2) * ACTION_MIN_PER_CLIP + 6
    if level == "moments" and ACTION_ENGINE == "ltx":
        # one LTX job holds both cards: talk seconds + action clips + one model load per engine (~6 min each)
        return MOMENTS_TALK * 8.0 * TALK_MIN_PER_S + MOMENTS_ACTION * ACTION_MIN_PER_CLIP + 12
    if level == "animated" and ACTION_ENGINE == "ltx":
        # one LTX job holds both cards (no split): talk + action + ~6 min per model load
        narration_clips = max(0.0, total - dialogue_s) / ACTION_S
        return dialogue_s * TALK_MIN_PER_S + narration_clips * ACTION_MIN_PER_CLIP + 20
    if level == "animated":
        narration_clips = max(0.0, total - dialogue_s) / ACTION_S
        # both cards share the talking and the action shots (10-18)
        return (dialogue_s * TALK_MIN_PER_S + narration_clips * ACTION_MIN_PER_CLIP) / 2 + 10
    return 0.0


# --- inputs --------------------------------------------------------------------------

def _figure(sheet: Path, out_dir: Path) -> Path:
    """The full-length half of a character sheet: a sheet shows the person
    twice, which a video model can take for two people."""
    out = out_dir / f"fig-{sheet.name}"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(video.ART_PYTHON), "-I", "-c",
                        "import sys\nfrom PIL import Image\nim = Image.open(sys.argv[1])\n"
                        "w, h = im.size\nim.crop((w // 2, 0, w, h)).save(sys.argv[2])",
                        str(sheet), str(out)], check=True)
    return out


LIGHT_SOURCE = re.compile(r"[^.]*\b(lamp\w*|lantern\w*|candle\w*|match\w*|fire\w*|moon\w*|sun\w*|"
                          r"daylight|dawn|dusk|window light)\b[^.]*\.?", re.I)
DARK = re.compile(r"\b(dark\w*|gloom\w*|night\w*|shadow\w*)\b", re.I)


def _lighting(prompt: str) -> str:
    """The scene still's light, for a portrait drawn for that moment (a
    portrait without it came out in daylight in a night scene, 10-16)."""
    lit = " ".join(m.group(0).strip() for m in LIGHT_SOURCE.finditer(prompt))
    if lit:
        return lit
    if DARK.search(prompt):
        return "It is night: the room is in near darkness, faces dimly lit."
    return ""


def _hair(look: str) -> str:
    m = re.search(r"[^,]*\bhair\b[^,]*", look)
    return m.group(0).strip() if m else "their face"


def _from_behind(look: str, dress: str | None = None) -> str:
    """Hair and clothes from a look (what tells two men apart, and what shows
    from behind); *dress* replaces the clothes (Roylott in his dressing-gown)."""
    hair = re.search(r"[^,]*\bhair\b[^,]*", look)
    if dress is None:
        m = re.search(r"wearing ([^.]*)", look)
        dress = re.split(r",\s*(?:looking|carrying|with|holding)\b", m.group(1))[0].strip() if m else ""
    return ", ".join(x for x in (hair and hair.group(0).strip(), dress) if x)


def _portraits(shots: list[dict], by_id: dict, plan: dict, style: str, work: Path, log,
               kf: dict | None = None, llm_url: str | None = None, model: str = "default") -> None:
    """The opening picture of each talking shot: a TWO-SHOT of just the speaker
    and the person spoken to, whoever else is in the scene (user, 10-17) -
    listener on the left turned to the speaker in three-quarter profile, speaker
    on the right facing us - so the two-person lip-sync can tie the voice to the
    right part and give the listener silence (s["bbox"]); a close-up when there is
    nobody to speak to. Faces from the scene's anchors and the room from its set
    anchor when the keyframes have a continuity bible; checked like keyframes
    (llm_url). "On the RIGHT half ... on the LEFT half" was drawn as two separate
    panels with Watson twice (10-17): one single picture, said outright."""
    chars, places = video.looks(plan)
    known = video.ref_prompts(plan, style)
    refs_dir, pdir = work.parent / "refs", work / "anim" / "portraits"
    props = {n: _prop_job(n, d, style, refs_dir) for n, d in plan["look"].get("props", {}).items()}
    anchors = anchor_jobs(kf, plan, style, work, props) if kf else {}
    jobs: dict[str, dict] = {}
    # each pair keeps its sides for the whole scene (the 180-degree rule): with the speaker always on the
    # right, Edric and Maren swapped places on every reply (user, 10-19). The first exchange sets them.
    sides: dict[tuple, dict[str, str]] = {}
    for s in shots:
        if s["kind"] != "talk":
            continue
        # with keyframes, always a portrait: the scene still of the Ascension had Jesus already rising,
        # without his mantle, for the words he says before it (10-17)
        if s["own_picture"] and not anchors and not kf:
            s["image"] = str(work / "img" / f"{s['pic']}.png")
            continue
        pic = by_id[s["pic"]]
        me, other = s["speaker"], s.get("listener")
        who = [me] + ([other] if other else [])
        if other:
            pair = sides.setdefault((s.get("where"), *sorted((me, other))), {other: "left", me: "right"})
            left, right = (other, me) if pair[me] == "right" else (me, other)
            halves = {"right": [0.52, 0.0, 1.0, 1.0], "left": [0.0, 0.0, 0.48, 1.0]}
            s["bbox"] = [halves[pair[me]], halves[pair[other]]]                  # speaker, listener
            s["prompt"] = (f"{me} speaks to {other}, who listens in silence with his mouth closed. "
                           f"Static camera. {style}.")
            beast = [w for w in (me, other) if _hair(chars.get(w, "")) == "their face"]
            if beast:
                # a creature in the two-shot: written for two people, the drawer painted the dragon as a
                # second princess (10-18). Its own staging: the creature looms, head and neck in view.
                b, h = beast[0], (other if beast[0] == me else me)
                side = {w: f"on the {pair[w]}" for w in (me, other)}
                body = (f"ONE single continuous illustration, NOT divided into panels. {h} {side[h]}, seen from the "
                        f"waist up, faces {b}; {b} {side[b]} is a huge creature, its head, neck and shoulders in "
                        f"view, turned toward {h}. {other} listens, mouth closed; {me} faces the viewer, calm, "
                        f"mouth closed. Only ONE person and ONE creature, each appearing exactly once: {h} is the "
                        f"only human")
            else:
                body = (f"ONE single continuous illustration of one room, NOT divided into panels, frames or "
                        f"halves. {left} and {right} stand facing each other, both seen from the waist up, a little "
                        f"apart: {left} on the left, {right} on the right. {other} is turned toward {me} in "
                        f"three-quarter profile, listening, mouth closed; {me} faces the viewer in three-quarter "
                        f"view, calm, mouth closed, hands still. Only these two people, each appearing exactly ONCE")
        else:
            body = (f"ONE single illustration, not divided into panels: a close-up of {me} alone, head and "
                    f"shoulders, facing slightly toward the viewer, calm, mouth closed")
        night = re.search(r"It is [^.]*\.(?:[^.]*light[^.]*\.)?", s.get("scene_prompt", ""))
        light = night.group(0) if night else _lighting(pic["prompt"])
        worn = {w: r for w in who if (r := video.outfit(plan, w, s.get("para")))}
        # the place first, in words: the character sheets' own backgrounds (a Gothic hall) pulled Maren's
        # shout from the ledge of the burnt mountain indoors (10-19)
        if s.get("where"):
            place = str(places.get(s["where"], "")).split(".")[0].strip()
            body = f"The scene is {s['where']}" + (f" ({place})" if place else "") + ", not indoors elsewhere. " + body
        # lip-sync needs the mouth: a knight's helm with an eye slit would hide it (10-18)
        helmed = [w for w in who if w in worn and HELM_ON.search(worn[w]["dress"])]
        if helmed:
            # the helm stays on through the fight (user, 10-19), open-faced so the lip-sync sees the mouth
            body += (f". {' and '.join(helmed)} keep{'s' if len(helmed) == 1 else ''} the helm on, open-faced: "
                     f"every face and mouth fully visible, no visor or mask over it")
        else:
            body += ". Every face fully visible: no helm, visor, hood or mask over it (a helm is off or carried)"
        refs, parts = [], [_tag_people(body, who, chars, {w: r["dress"] for w, r in worn.items()}),
                           light.rstrip(". ") if light else ""]
        for w in who:
            sheet = _outfit_sheet(plan, style, refs_dir, w, worn[w]["id"]) if w in worn else None
            if sheet:
                # the outfit of this moment: the knight speaks to the dragon in the armor he forged (10-18)
                refs.append(str(_figure(sheet, work / "anim" / "refs")))
                parts.append(f"{w} has exactly the face, hair and clothes of the person in image {len(refs)} "
                             f"({worn[w]['dress']})")
            elif f"char:{w}" in anchors:
                refs.append(anchors[f"char:{w}"]["path"])
                parts.append(f"{w} has exactly the face, age and hair of the person in image {len(refs)}")
            elif w in known:
                # the one-figure half of the sheet: given the whole sheet, every solo close-up of
                # Jesus and Peter came back as the sheet itself, the man twice, 4 seeds of 4 (10-17)
                refs.append(str(_figure(video.ref_path(refs_dir, w, known[w][0]), work / "anim" / "refs")))
                if _hair(chars.get(w, "")) == "their face":
                    parts.append(f"{w} is exactly the creature in image {len(refs)}: the same head, horns, body "
                                 f"and scale colour (a creature, not a person)")
                else:
                    parts.append(f"{w} has exactly the face, hair and clothes of the person in image {len(refs)}")
        set_anchor = anchors.get(f"set:{s['where']}")
        if set_anchor:
            refs.append(set_anchor["path"])
            parts.append(f"The room is the one in image {len(refs)}, seen behind them")
        elif s["where"] in known:
            refs.append(str(video.ref_path(refs_dir, s["where"], known[s["where"]][0])))
            parts.append(f"The setting is {s['where']}, as in image {len(refs)}")
        elif s["where"] in places:
            # said in words when there is no picture of it: left out, the portraits on the burnt mountain
            # came out in a candlelit hall (10-18)
            parts.append(f"The setting is {s['where']}: {str(places[s['where']]).rstrip('.')}")
        parts += [video.ANATOMY, style, "No text, no lettering, no signature"]
        prompt = ". ".join(x for x in parts if x) + "."
        pid = f"talk-{video.slug(me)}-{s['pic']}-{_hash(prompt)}"
        s["image"] = str(pdir / f"{pid}.png")
        if pid not in jobs:
            pad = lambda b: [max(0.0, b[0] - 0.05), b[1], min(1.0, b[2] + 0.05), b[3]]  # noqa: E731
            prompt, loras = _char_loras(prompt, who, chars, work,
                                        {w: pad(b) for w, b in zip(who, s.get("bbox") or [])})
            jobs[pid] = {"id": pid, "prompt": prompt, "seed": 4242, "path": s["image"], "refs": refs,
                         **({"loras": loras} if loras else {}),
                         # the outfit of the moment for the check too: judged against his usual leathers, every
                         # armored portrait of the knight failed three times (10-18)
                         "frame": {"who": who, "props": [], "dress": {w: r["dress"] for w, r in worn.items()},
                                   "must": [
                             "one single picture of one room, not split into panels or halves",
                             f"exactly {len(who)} {'person' if len(who) == 1 else 'people'}, each appearing once"]
                             + [f"{WARDROBE_MUST}{w} wears {_gist(r['dress'])}" for w, r in worn.items()]
                         + [h for w, r in worn.items() if (h := _head(w, r["dress"]))]},
                         "anchors": {w: anchors[f"char:{w}"]["path"] for w in who if f"char:{w}" in anchors},
                         "set": None}
    if not jobs:
        return
    state_f = work / "anim" / "portrait_checks.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"seeds": {}, "passed": {}, "log": []}
    for j in jobs.values():
        j["seed"] = state["seeds"].get(j["id"], j["seed"])
    keep = DRAW_KEEP
    for rnd in range(KEYFRAME_ROUNDS + 1):
        todo = [j for j in jobs.values() if not Path(j["path"]).exists()]
        if todo:
            log(f"  drawing {len(todo)} talking portrait(s)")
            video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in
                             [j for j in anchors.values() if not Path(j["path"]).exists()] + todo], pdir, log)
        if not llm_url:
            break
        # in parallel, like the keyframes (one at a time, 12 portraits took ~15 min, 10-18)
        bad = set()
        with ThreadPoolExecutor(video.checker_workers()) as ex:
            for r in ex.map(lambda j: check_keyframes([j], state, chars, {}, llm_url, model, log,
                                                      rnd == KEYFRAME_ROUNDS, True), list(jobs.values())):
                bad |= r
        for j in jobs.values():
            if j["id"] in bad:
                state["seeds"][j["id"]] = j["seed"] = int(j["seed"]) + 7919
                Path(j["path"]).unlink(missing_ok=True)
        state_f.write_text(json.dumps(state, indent=1))
        if not bad:
            break


# What sets the scene for an action shot: "still" = the chapter's own picture
# of that moment (its style, faces and composition carry over; the first
# clips, made from the empty-room reference, looked painterly-photographic
# next to the pen-and-ink stills and Holmes changed face), "place" = the
# empty-room reference.
ACTION_SCENE_REF = os.environ.get("BOOKCASTER_ACTION_SCENE_REF", "still")


def _action_refs(s: dict, pic: dict, plan: dict, style: str, work: Path) -> list[str]:
    prompts = video.ref_prompts(plan, style)
    refs_dir = work.parent / "refs"
    out_dir = work / "anim" / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = [str(_figure(video.ref_path(refs_dir, n, prompts[n][0]), out_dir))
            for n in pic.get("who", [])[:3] if n in prompts]
    still = work / "img" / f"{pic['id']}.png"
    if ACTION_SCENE_REF == "still" and still.exists():
        refs.append(str(still))
    elif pic.get("where") in prompts:
        refs.append(str(video.ref_path(refs_dir, pic["where"], prompts[pic["where"]][0])))
    return refs


# --- keyframes ------------------------------------------------------------------------
DRAW_KEEP = ("id", "prompt", "seed", "path", "refs", "size", "control", "loras")
# pose skeletons (tools/poses.py) as the drawer's control picture, with Qwen-Image: given a lunge, the
# sword went into the dragon's chest; without, both drawers held it beside the dragon (10-19)
POSES = Path(__file__).resolve().parent.parent / "tools" / "poses.py"
POSE_PYTHON = Path(os.environ.get("BOOKCASTER_POSE_PYTHON", Path.home() / "tts-ab" / ".venv-pose" / "bin" / "python"))
# 0.8 with references pulled the sample's figures onto plain backgrounds in stiff sheet poses (10-19)
POSE_SCALE = float(os.environ.get("BOOKCASTER_POSE_SCALE", "0.5"))
CUTOUT = POSES.with_name("cutout.py")


def _cut(ref: str, out_dir: Path) -> str:
    """For Qwen-Image, a reference cut out onto a plain grey card (tools/cutout.py): it copies a reference's
    background into the scene - the knight on blank white beside a lone table, a sword floating, from the
    prop and creature sheets drawn on white (10-19). Other drawers get the reference as it is."""
    if video.DRAWER != "qwen":
        return ref
    out = out_dir / f"cut-{Path(ref).name}"
    if not out.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([str(POSE_PYTHON), "-I", str(CUTOUT), ref, str(out)], capture_output=True)
        if r.returncode or not out.exists():
            return ref
    return str(out)


def pose_names() -> list[str]:
    try:
        return list(json.loads(POSES.with_name("poses.json").read_text()))
    except (OSError, ValueError, AttributeError):
        return []


def _pose_control(poses: list, out: Path) -> dict | None:
    """A frame's planned poses drawn as an OpenPose skeleton for the drawer (Qwen-Image only: klein has
    no control input); None without poses or when the renderer fails."""
    if video.DRAWER != "qwen" or not poses:
        return None
    figs = [{k: p[k] for k in ("pose", "box", "facing") if k in p} for p in poses
            if isinstance(p, dict) and p.get("pose") and isinstance(p.get("box"), list) and len(p["box"]) == 4]
    if not figs:
        return None
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([str(POSE_PYTHON), "-I", str(POSES), "render", json.dumps(figs), str(out),
                            "--size", "1536x864"], capture_output=True, text=True)
        if r.returncode or not out.exists():
            return None
    return {"image": str(out), "type": "pose", "scale": POSE_SCALE}
# One drawn picture per action shot (anim/keyframes.json), so each 5 s clip shows its
# own moment instead of 13 variations on one scene still, with the objects in it as
# references in their state (the safe open, the speckled band a snake) (user, 10-16).
# A frame is "key" (drawn, animated from it), "continue" (animated on from the clip
# before: no picture) or "still" (drawn - or the last picture again - and shown with
# a slow push-in: dark waiting, the dead man; ~20 s instead of ~8 min).

def _shot_text(t0: float, t1: float, timeline: list[dict], total: float) -> str:
    """The words spoken over [t0, t1), cut from the timeline's segments by position (an estimate)."""
    out = []
    for i, e in enumerate(timeline):
        a = float(e["t"]); b = float(timeline[i + 1]["t"]) if i + 1 < len(timeline) else total
        if b <= t0 or a >= t1:
            continue
        txt = str(e.get("text", "")); n = len(txt)
        out.append(txt[int(max(0.0, (t0 - a) / (b - a)) * n):int(min(1.0, (t1 - a) / (b - a)) * n)])
    return " ".join(x.strip() for x in out if x.strip())


KEYFRAME_RULES = """Rules (each learned from a failed picture):
- One frame per 5-second shot, in order. "key" = a new picture of that moment; "continue" = the same moment
  carries on from the shot before (no picture; motion only). Use "continue" when nothing new happens.
- "who": every person whose head is in the picture, by the names listed; [] for a room or hands only.
  A dead or unconscious person is still listed. Put only the people the moment needs in a close-up.
- Name people by name in "prompt"; their looks are added automatically. Say who holds what ("Watson holds
  the revolver").
- Props: the objects the moment needs, from the list or new ones you define in "props". Describe an object by
  its shape and parts, never by its light ("a small dark square hole in the wall", not "a glow"). A prop in a
  changed state is its own prop named "<Name>, <state>" ("Iron safe, open") with its own description.
  Props are things, never people: a person's posture or state goes in "prompt" and "must" ("Lame Man,
  sitting" as a prop kept the healed man sitting, 10-18).
- Say the light every time ("It is night: the window is black; the only light is ..."). Never "flicker",
  "ember", "spark", "dust", "smoke", "mist" (they become floating particles).
- "must": 1-3 short checkable facts the picture has to show (states, who holds what, what must NOT be there).
- "dress": a person's clothes when they differ from their usual look ({"Roylott": "a long grey dressing-gown"}).
- "motion": what moves during the 5 seconds, plainly, as one or two sentences.
- "carry": true only when this picture is the same framing a moment after the previous key picture.
- Start every "prompt" with its framing: "Wide shot", "Medium shot", "Close-up of <what>", "Low angle".
  Consecutive key frames of a scene never share a framing: cut between wide, medium and close-ups of
  hands, tools, objects and faces, as a film editor would. Eight medium shots of a man at an anvil
  read as one picture eight times (10-18).
- A montage (time passing, work done step by step: "first the breastplate, then the helm, then ...",
  "forty days") is one "key" frame per step, each showing that step's own result or action, never
  "continue". Show the work in progress: the glowing metal, the tool, the half-made thing.
- "continue" only when the motion of the shot before simply goes on. Aim for about half the shots as "key":
  a new moment, a new framing, a step of a montage; a moment that goes on is "continue" (every key frame
  is drawn several times and checked; 56 keys for 69 shots took hours, 10-18).
- "checks": 2-6 atomic yes/no questions a viewer could answer from the picture alone, each about ONE thing:
  who is there ("Is the dragon in the picture?"), what someone wears or holds ("Is Edric holding a hammer?"),
  a simple relation ("Is Maren on the ledge above the dragon?"), a setting detail. Never framing, motion,
  sound, heat, feelings or what happened before. Give each a "type" (entity, outfit, relation, prop,
  setting, count) and, if it only makes sense when another is yes, "parent": that question's index.
- Action (a hammer blow, a thrust, a parry, a roll, a creature seizing or carrying someone) is drawn at its
  clearest instant as a physical pose: who, which hand, the tool or weapon, and WHERE it touches what ("Close-up
  of Edric's right hand bringing the hammer down onto the orange-hot blade on the anvil, the hammer head on the
  blade"; "Vorgrath's foreclaws close around Maren's waist, she hangs below its chest, its wings spread"). Never
  a vague "fighting", "battle" or "struggle": the dragon fight drawn as two figures facing each other did not
  read (user, 10-19). Hot metal is "glowing orange-hot" (a colour on the metal, never sparks).
  A weapon or tool is held in ONE named hand ("the sword in his right hand, his left hand free"): a sword
  drawn in both hands came out as two blades (user, 10-19); its "must" says so.
- A fight or a piece of work is a run of key frames, ONE move each (strike, parry, roll, thrust; heat, hammer,
  quench), each its own framing; a new move is never "continue". Its "motion" is that one movement, its
  direction and its end ("he swings the sword down and the dragon's claw blocks it"); its "must" and a
  "relation" check name the contact ("Is the hammer head touching the blade?").
- "poses": for each person in a key frame whose body matters to the moment, a pose from the pose list,
  the box their body fills ([x0, y0, x1, y1], fractions of the frame: 0,0 top left; a standing person in a
  medium shot about 0.45 wide and 0.15-1.0 tall) and the way they face ("left"/"right"). People facing each
  other face opposite ways. A creature has no pose: leave it out and give it the free part of the frame.
- "end": for a key frame that is ONE move (a strike, a thrust, a parry, a roll, a creature seizing someone,
  a helm put on or taken off), the same picture at the END of the move: {"prompt": "<what has changed: the
  hammer head on the blade>", "poses": [... the end poses ...]}. Same framing, people and place; only the
  move changes. The clip is the motion between the two. Leave "end" out for a moment without a move. The end
  "prompt" says ONLY what is different ("the sword now blocks the claw above his head"), never the start again.
- Show what the narration tells, including what it reports only in passing or by its result ("Seven rode out ...
  Three days later one came back, on foot, his shield melted"): the knights riding out, the dragon's fire breaking
  over them, the one who walks back. Narration over an empty place shows the event it tells of, not the place.
- One moment, one picture: never a split screen, panels, an inset or "two areas" (drawn as two pictures
  in one, 10-18). Someone's thoughts or memories get their own key frame, or stay unseen.
- An object the story has changed is described as changed from then on, never by its old description
  (the star-iron's crack closes under the hammer: the finished sword has no crack).
- The camera never moves. "motion" is what the PEOPLE do; nothing that would lead the eye out of the frame
  ("the cloud rises higher", "a bird flies off"). When something leaves the frame or the people would no
  longer be in view, the NEXT shot is a new "key" frame, never "continue".
- Only people physically there at that moment: someone the narrator addresses ("O Theophilus") or who is only
  spoken about is never in the picture.
- "crowd": the unnamed people around the named ones when the story has a group there (the disciples, a
  crowd): how many, how dressed, where ("a dozen men in plain robes seated around him"); "" for none. Never
  write "alone" for someone the story has among others.
- The same people throughout a scene: every key frame of it has the same "who" and the same "crowd" (same
  words, same count) unless the narration has someone arrive or leave; one who leaves (Jesus taken up) is
  dropped, one who arrives (two men in white) is added under their look name ("Angels"). Use the look names
  above for groups too.
- Posture: a person keeps their posture (standing stays standing) unless the narration changes it; say so in
  "motion" ("Peter stays standing, ..."); "stood up" means rising to stand and staying up. No bench, chair or
  stool right behind a standing person (the video model sits them down).
- A change lasts: once the narration changes someone (healed, rises, is bound, puts something on), every
  later frame shows them changed, in the prompt and "must" alike: the lame man, healed and clinging to Peter,
  was drawn sitting on the steps again in the next scene (10-18).
- Flashback: when the narration under a shot tells of an earlier event somewhere else (an aside, "Now this
  man acquired a field ..."), show THAT event with its people in a key frame (Judas at the field he bought),
  with its own "where", then go back to the scene; an empty place alone is not a flashback. Never show a death, a wound, blood or a body: show the moment before, or the empty place after. A blow the story strikes is shown, without blood (the sword driven into the dragon's chest between its scales)."""


def plan_keyframes(shots: list[dict], pics: list[dict], timeline: list[dict], total: float, plan: dict,
                   style: str, llm_url: str, model: str, log=print, starts: list[float] | None = None) -> dict:
    """keyframes.json for these shots: one language-model call per scene picture, given the picture's
    description, the narration under each of its action shots, the cast and the props so far."""
    from .score import _chat

    chars, places = video.looks(plan)
    props = dict(plan["look"].get("props", {}))
    frames = []
    by_pic: dict[str, list[dict]] = {}
    for sh in sorted((x for x in shots if x["kind"] == "action"), key=lambda x: x["t"]):
        by_pic.setdefault(sh["pic"], []).append(sh)
    pic_of = {p["id"]: p for p in pics}
    for n, (pid, group) in enumerate(by_pic.items(), 1):
        pic = pic_of[pid]
        lines = "\n".join(f'{sh["id"]}: "{_shot_text(sh["t"], sh["t"] + sh["dur"], timeline, total)}"'
                          for sh in group)
        worn = []
        for name in chars:
            seen = []
            for sh in group:
                row = video.outfit(plan, name, video.para_at(starts, sh["t"])) if starts else None
                d = row["dress"] if row else "their usual clothes (as in their look)"
                if not seen or seen[-1][1] != d:
                    seen.append((sh["id"], d))
            if any(d != "their usual clothes (as in their look)" for _, d in seen):
                worn.append(f"- {name}: " + "; ".join(f"from {i}: {d}" for i, d in seen))
        wardrobe = ("What people wear here (fixed by the story's wardrobe and added automatically; never "
                    "describe other clothes for them, and leave them out of \"dress\"):\n" + "\n".join(worn) + "\n"
                    if worn else "")
        prompt = (
            "You plan the animation of an illustrated audiobook, one drawn keyframe per 5-second shot.\n"
            f"Scene picture: {pic['prompt']}\nPlace: {pic.get('where', '')}: {places.get(pic.get('where', ''), '')}\n"
            + (f"Group there: {pic['crowd']}\n" if pic.get("crowd") else "") +
            f"People (use these names): {', '.join(chars)}\n"
            f"Props so far: {json.dumps(props, ensure_ascii=False)}\n" + wardrobe + "\n"
            f"The shots and the narration heard during each:\n{lines}\n\n{KEYFRAME_RULES}\n\n"
            f"The pose list: {', '.join(pose_names())}\n\n"
            'Reply with ONLY JSON: {"props": {"<new prop>": "<description>"}, "frames": [{"shot": "<id>", '
            '"mode": "key|continue", "who": [...], "crowd": "", "where": "<place>", "props": [...], "dress": {}, "must": [...], '
            '"checks": [{"q": "<yes/no question>", "type": "entity|outfit|relation|prop|setting|count", "parent": null}], '
            '"prompt": "<the picture, 2-4 sentences>", "motion": "<what moves>", "carry": false, '
            '"poses": [{"who": "<name>", "pose": "<from the list>", "box": [x0, y0, x1, y1], "facing": "right"}], '
            '"end": {"prompt": "<what has changed>", "poses": [...]}}]}')
        reply = _chat(llm_url, model, prompt, max_tokens=6000)
        m = re.search(r"\{.*\}", reply or "", re.S)
        try:
            data = json.loads(m.group(0)) if m else {}
        except ValueError:
            data = {}
        props.update({k: str(v) for k, v in (data.get("props") or {}).items() if k not in props})
        got = {f.get("shot"): f for f in data.get("frames", []) if isinstance(f, dict)}
        for k, sh in enumerate(group):
            f = got.get(sh["id"]) or {"mode": "continue" if k else "key", "who": pic.get("who", []),
                                      "crowd": pic.get("crowd", ""),
                                      "where": pic.get("where", ""), "prompt": pic["prompt"],
                                      "motion": _motion_prompt(pic, "")}
            f["shot"] = sh["id"]
            f["pic"] = sh["pic"]
            if k == 0 or not f.get("mode"):
                f["mode"] = "key" if k == 0 else f.get("mode", "key")
            if f["mode"] == "key" and not f.get("prompt"):
                f["prompt"] = pic["prompt"]
            frames.append(f)
        log(f"    planned keyframes for {pid} ({n}/{len(by_pic)}): "
            f"{sum(f['mode'] == 'key' for f in frames[-len(group):])} key / {len(group)}")
    _cap_who(frames)
    _merge_near_duplicates(frames, log)
    _break_chains(frames)
    return {"version": 1, "props": props, "frames": frames}


def _merge_near_duplicates(frames: list[dict], log=print) -> None:
    """A key frame whose picture is nearly the one before it (same people, same scene, the same words
    give or take the framing) is a "continue": drawn and checked again for nothing otherwise."""
    import difflib
    strip = lambda p: re.sub(r"^\s*(extreme close-up|close-up|medium shot|wide shot|low angle|high angle)[^.]*\.",
                             "", str(p or ""), flags=re.I).strip().lower()
    last, n = None, 0
    for f in frames:
        if f.get("mode") != "key":
            continue
        if (last and f.get("pic") == last.get("pic") and f.get("who") == last.get("who")
                and difflib.SequenceMatcher(None, strip(f.get("prompt")), strip(last.get("prompt"))).ratio() > 0.85):
            f["mode"], n = "continue", n + 1
            continue
        last = f
    if n:
        log(f"    {n} near-duplicate key frame(s) made continuous shots")


def _drop_people_props(kf: dict, plan: dict) -> None:
    """A "prop" named after a person ("Lame Man, sitting") is a posture, not a
    thing: drawn as an object reference and checked as an item, it held the
    healed man sitting through every later frame (10-18). Drop it everywhere."""
    names = [n.lower() for n in video.looks(plan)[0]]
    people = lambda p: any(re.search(rf"\b{re.escape(n)}\b", p.lower()) for n in names)
    kf["props"] = {n: d for n, d in (kf.get("props") or {}).items() if not people(n)}
    for f in kf.get("frames", []):
        f["props"] = [x for x in f.get("props") or [] if not people(x)]
    for n in [n for n in plan["look"].get("props", {}) if people(n)]:
        del plan["look"]["props"][n]


MAX_WHO = 3   # references the image model takes per picture


def _add_creatures(frames: list[dict], chars: dict) -> None:
    """A creature the prompt names by its kind ("the dragon's underbelly fills the frame") joins the
    frame's "who", so its sheet keeps it the same creature: the planner listed only Edric for the five
    frames under the dragon's wing, and the dragon came out different each time (10-18)."""
    kinds = {}
    for name, look in chars.items():
        if _hair(look) != "their face":
            continue
        m = re.match(r"\s*(?:an?|the)\s+((?:[\w-]+\s+){0,3}?)([a-z-]+)\b(?=\s*(?:with|,|$|of\b))", look, re.I)
        if m:
            kinds[m.group(2).lower()] = name
    for f in frames:
        said = f"{f.get('prompt', '')} {f.get('motion', '')}".lower()
        for kind, name in kinds.items():
            who = f.setdefault("who", [])
            if name not in who and re.search(rf"\b{re.escape(kind)}\b", said):
                who.append(name)


def _cap_who(frames: list[dict]) -> None:
    """At most MAX_WHO named people a frame: the ones its prompt and motion name first, then those
    already in the shot (the same faces from frame to frame); the rest join its crowd. Cut in list order, the Ascension's
    Angels (named last) were dropped from the frame they speak in (10-17)."""
    before: list[str] = []
    for f in frames:
        who = list(dict.fromkeys(f.get("who") or []))
        if len(who) > MAX_WHO:
            said = f"{f.get('prompt', '')} {f.get('motion', '')}"
            rank = sorted(who, key=lambda w: (not re.search(rf"\b{re.escape(w)}\b", said), w not in before,
                                              who.index(w)))
            keep, rest = rank[:MAX_WHO], rank[MAX_WHO:]
            f["who"] = [w for w in who if w in keep]
            f["crowd"] = ", ".join(x for x in (str(f.get("crowd") or "").rstrip(". "),
                                               f"among them {' and '.join(rest)}") if x)
        if f.get("mode") == "key":
            before = list(f.get("who") or [])


MAX_CONTINUE = 3   # 5-s shots animated on from the clip before, in a row


def _break_chains(frames: list[dict]) -> None:
    """After MAX_CONTINUE "continue" shots a fresh key frame of the same moment (drawn from the scene's
    last key frame, so the same people and room): each continue starts from the last frame of the clip
    before, and drift adds up - three shots on from the Ascension the clouds had grown a giant blue face
    (10-17); chapter 2's plan chained nine."""
    last, run = None, 0
    for f in frames:
        if f.get("mode") == "key":
            last, run = f, 0
            continue
        if f.get("mode") != "continue" or last is None or f.get("pic") is None:
            continue
        run += 1
        if run > MAX_CONTINUE and f.get("pic") == last.get("pic"):
            # cut to another framing of the same moment: the same picture again read as a jump back (10-18)
            closer = _framing(last.get("prompt", "")) in ("wide", "medium", "")
            body = re.sub(r"^\s*(extreme close-up|close-up|medium shot|wide shot|low angle|high angle)[^.]*\.\s*",
                          "", last.get("prompt", ""), flags=re.I)
            f.update(mode="key", carry=False,
                     prompt=("Close-up of the same moment: " if closer else "Wide shot of the same moment: ") + body,
                     where=last.get("where", ""),
                     who=list(last.get("who") or []), crowd=last.get("crowd", ""),
                     props=list(last.get("props") or []), must=list(last.get("must") or []),
                     dress=dict(last.get("dress") or {}))
            last, run = f, 0


def keyframes(work: Path) -> dict | None:
    f = work / "anim" / "keyframes.json"
    if not f.exists():
        return None
    kf = json.loads(f.read_text())
    _cap_who(kf.get("frames", []))
    _break_chains(kf.get("frames", []))
    _carry_dress(kf.get("frames", []))
    return kf


FRAMINGS = (("extreme close-up", "close"), ("close-up", "close"), ("close up", "close"), ("closeup", "close"),
            ("over-the-shoulder", "medium"), ("medium shot", "medium"), ("medium", "medium"),
            ("wide shot", "wide"), ("establishing", "wide"), ("wide", "wide"), ("low angle", "low"),
            ("high angle", "high"), ("from above", "high"), ("from below", "low"))


def _framing(prompt: str) -> str:
    """'close' / 'medium' / 'wide' / 'low' / 'high' from a keyframe prompt's wording ('' = unsaid)."""
    low = prompt.lower()[:160]
    hits = [(low.find(k), v) for k, v in FRAMINGS if k in low]
    return min(hits)[1] if hits else ""


def _drop_stale(jobs: list[dict], kf: dict, state: dict, work: Path, log) -> None:
    """A keyframe whose prompt or references changed since it was drawn (a new look, an outfit from the
    wardrobe, a fixed plan) is drawn again, and its clip - with the clips carried on from it - rendered
    again: they only ever checked that the picture existed, so a corrected look reached the scene
    pictures and never the animation (10-18). A keyframe drawn before this stamp is stamped, not redrawn."""
    stamps = state.setdefault("prompts", {})
    order = [f["shot"] for f in kf.get("frames", [])]
    modes = {f["shot"]: f.get("mode") for f in kf.get("frames", [])}
    stale = []
    for j in jobs:
        h = _hash(j["prompt"] + "|" + "|".join(j["refs"]))
        if stamps.get(j["id"]) not in (None, h) and Path(j["path"]).exists():
            Path(j["path"]).unlink()
            stale.append(j["id"])
            i = order.index(j["id"]) if j["id"] in order else len(order)
            for shot in [j["id"]] + [x for x in order[i + 1:]][:MAX_CONTINUE + 1]:
                if shot != j["id"] and modes.get(shot) != "continue":
                    break
                for x in (work / "anim").glob(f"{shot}*"):
                    if x.suffix in (".mp4", ".png") or x.name.endswith(".latents.pt"):
                        x.unlink()
        stamps[j["id"]] = h
    if stale:
        log(f"  {len(stale)} keyframe(s) changed since drawn (look, outfit or plan): drawing and rendering "
            f"them again")


def _shot_t(shot_id: str) -> float:
    """A shot's start in seconds from its id ("action-000458" = 45.8 s)."""
    try:
        return int(str(shot_id).rsplit("-", 1)[-1]) / 10
    except ValueError:
        return 0.0


WARDROBE_MUST = "Clothes: "


def apply_wardrobe(kf: dict, plan: dict, starts: list[float]) -> None:
    """Each frame's people in the outfit the wardrobe has them in at that moment (video.plan_wardrobe):
    its dress, its reference sheet (f["outfit"]) and a check that they wear it. The wardrobe wins over
    a "dress" the per-scene planner gave: it put the knight back in leather for the fight (10-18)."""
    for f in kf.get("frames", []):
        para = video.para_at(starts, _shot_t(f.get("shot", "")))
        f["must"] = [m for m in f.get("must") or [] if not str(m).startswith(WARDROBE_MUST)]
        f["outfit"] = {}
        for w in f.get("who") or []:
            row = video.outfit(plan, w, para)
            if row:
                f.setdefault("dress", {})[w] = row["dress"]
                f["outfit"][w] = row["id"]
                f["must"].append(f"{WARDROBE_MUST}{w} wears {_gist(row['dress'])}")
                if head := _head(w, row["dress"]):
                    f["must"].append(head)


def _head(w: str, dress: str) -> str | None:
    """The helm as its own checked fact: cut from the gist of the armor, it came and went through the
    dragon fight unchecked (user, 10-19)."""
    if not re.search(r"\b(helm|helmet)\b", dress, re.I):
        return None
    if HELM_ON.search(dress):
        return f"{WARDROBE_MUST}{w} wears a helm on the head"
    return f"{WARDROBE_MUST}{w} wears no helm on the head (bare head; a helm may be carried)"


def _gist(dress: str, n: int = 110) -> str:
    """The first clauses of a dress, short enough for a checkable fact."""
    if len(dress) <= n:
        return dress
    cut = dress[:n].rsplit(",", 1)[0]
    return cut if len(cut) > 30 else dress[:n].rsplit(" ", 1)[0]


def _outfit_sheet(plan: dict, style: str, refs_dir: Path, w: str, oid: str | None) -> Path | None:
    row = next((r for r in (plan.get("wardrobe") or {}).get(w, []) if r["id"] == oid), None) if oid else None
    return video.outfit_ref(refs_dir, plan, style, w, row) if row else None


def _carry_dress(frames: list[dict]) -> None:
    """A costume change lasts: each character keeps the last "dress" a frame gave them until a later
    frame gives another. Frames are planned one scene picture at a time, so the armor the knight forged
    in one scene was gone (blank: back to his stable leathers) in the next, and in the fight (10-18)."""
    last: dict[str, str] = {}
    for f in sorted(frames, key=lambda f: float(str(f.get("shot", "x-0")).rsplit("-", 1)[-1] or 0)):
        dress = f.setdefault("dress", {})
        for name in f.get("who", []):
            if dress.get(name):
                last[name] = dress[name]
            elif name in last:
                dress[name] = last[name]


def _prop_job(name: str, desc: str, style: str, refs_dir: Path) -> dict:
    prompt = (f"Object reference: {name}, {desc.rstrip('.')}. Shown alone, whole and clearly, on a "
              f"plain light background, even light. {style}. No people. No text, no lettering, no signature.")
    path = video.ref_path(refs_dir, f"prop {name}", prompt)
    return {"id": path.stem, "prompt": prompt, "seed": 7, "path": str(path), "size": [1024, 768]}


def _char_loras(prompt: str, who: list[str], chars: dict, work: Path,
                boxes: dict | None = None) -> tuple[str, list]:
    """Character LoRAs (tools/charlora.py: <book>/work/video/charlora/<slug>/lora/USE or step_00800) for a draw
    job: [path, 1.0] in a one-person frame, [path, 1.0, box] (0-1, REGIONAL) with several people - a global
    character LoRA in a two-shot dresses everyone as that person (charlora eval, 10-20), so a person with no
    box gets none. The trigger goes before the person's first mention. -> (prompt, loras)"""
    people = [w for w in who if _hair(chars.get(w, "")) != "their face"]
    loras = []
    for w in people:
        d = work.parent / "charlora" / video.slug(w) / "lora"
        use = d / "USE"
        f = d / use.read_text().strip() if use.exists() else d / "step_00800.safetensors"
        if not f.exists():
            continue
        if len(people) == 1:
            loras.append([str(f), 1.0])
        elif boxes and boxes.get(w):
            loras.append([str(f), 1.0, boxes[w]])
        else:
            continue
        meta = d.parent / "data" / "meta.json"
        trig = (json.loads(meta.read_text()).get("trigger") if meta.exists() else None) or f"chr_{video.slug(w)}"
        i = prompt.find(w)
        if i >= 0 and trig not in prompt:
            prompt = prompt[:i] + trig + " " + prompt[i:]
    return prompt, loras


def _tag_people(prompt: str, who: list[str], chars: dict, dress: dict | None = None) -> str:
    """Each person's first mention gets their distinguishing look inline ("Watson
    (short brown hair, tweed suit) holds the revolver"): with only names, the drawer
    gave Holmes the lamp AND Watson's revolver and swapped their hair and coats (10-17)."""
    for w in who:
        look = _from_behind(chars.get(w, ""), (dress or {}).get(w))
        # the age and face too (10-20): hair + clothes alone never said "a woman of about 70, deep wrinkles",
        # and the drawer painted Hulda at 40 in every two-shot
        short = video._short(chars.get(w, ""))
        if short and look and short not in look:
            look = f"{short}; {look}"
        if not look:
            continue
        for name in sorted(video._names(w), key=len, reverse=True):
            m = re.search(rf"\b{re.escape(name)}\b(?!\s*\(|['’]s)", prompt, re.I)   # not "Roylott's room"
            if m:
                prompt = prompt[:m.end()] + f" ({look})" + prompt[m.end():]
                break
    return prompt


def draw_keyframes(kf: dict, plan: dict, style: str, work: Path, log=print,
                   llm_url: str | None = None, model: str = "default") -> dict[str, str]:
    """Draw the props and the keyframes not drawn yet; returns shot id -> picture
    (a "still" without its own prompt gets the picture before it). With a vision
    model (llm_url), each new keyframe is checked against its frame's plan and a
    failing one drawn again with another seed (check_keyframes)."""
    state_f = work / "anim" / "keyframe_checks.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"seeds": {}, "passed": {}, "log": []}
    state["_path"] = str(state_f)
    refs_dir = work.parent / "refs"
    kdir = work / "anim" / "keyframes"
    props = {n: _prop_job(n, d, style, refs_dir) for n, d in plan["look"].get("props", {}).items()}
    chars, places = video.looks(plan)
    known = video.ref_prompts(plan, style)
    anchors = anchor_jobs(kf, plan, style, work, props)
    cuts = work / "anim" / "refs"
    if video.DRAWER == "qwen":
        # the sheets and props must exist before they can be cut out: draw them first
        first = [j for j in video.reference_jobs(plan, style, refs_dir) + video.outfit_jobs(plan, style, refs_dir)
                 + list(props.values()) if not Path(j["path"]).exists()]
        if first:
            log(f"  drawing {len(first)} reference(s) and prop(s) first (cut out for the drawer)")
            video.draw_jobs([{k: v for k, v in j.items() if k in DRAW_KEEP} for j in first], kdir, log)
    jobs, images, last = [], {}, None
    for n, f in enumerate(kf["frames"]):
        if f.get("mode") == "continue":
            continue
        if not f.get("prompt"):
            if last:
                images[f["shot"]] = last
            continue
        who = [w for w in f.get("who", []) if w in chars][:3]
        st = states_for(kf, f["shot"], who)
        refs, parts = [], [_tag_people(f["prompt"].rstrip(". "), who, chars, f.get("dress"))]
        parts += [x["prompt"].rstrip(". ") for x in st if x.get("prompt")]
        used = {}
        for w in who:
            sheet = _outfit_sheet(plan, style, refs_dir, w, (f.get("outfit") or {}).get(w))
            if sheet:
                # the outfit's own sheet (same face, the clothes of this moment): with the usual sheet
                # and the outfit only in words, the drawer copied the sheet's clothes (10-18)
                refs.append(_cut(str(_figure(sheet, work / "anim" / "refs")), cuts)); used[w] = "outfit"
            elif f"char:{w}" in anchors:
                refs.append(_cut(anchors[f"char:{w}"]["path"], cuts)); used[w] = "anchor"
            elif w in known:
                # the full-length half of the sheet: given the whole sheet (the person twice), the leaping
                # healed man's scenes came out with two Johns, three rounds running (10-18)
                refs.append(_cut(str(_figure(video.ref_path(refs_dir, w, known[w][0]), work / "anim" / "refs")),
                                 cuts))
                used[w] = "sheet"
        cast = [f"{w} ({video._short(chars[w])}, as in image {k})" for k, w in enumerate(who, 1)]
        crowd = str(f.get("crowd") or "").strip()
        if who and crowd:
            parts.append(video.crowd_line(who, cast, crowd))
        else:
            parts.append(f"Exactly {len(who)} {'person' if len(who) == 1 else 'people'} in the picture, each "
                         f"appearing once: " + "; ".join(cast) if who else
                         (f"Unnamed people only: {crowd}" if crowd else "No people in the picture"))
        for k, w in enumerate(who, 1):
            if used.get(w) == "outfit":
                parts.append(f"{w} has exactly the face, hair and clothes of the person in image {k} "
                             f"({f['dress'][w]}), in the pose this moment describes")
            elif used.get(w) == "anchor":
                parts.append(f"{w} has exactly the face, age and hair of the person in image {k}")
            elif used.get(w) == "sheet" and _hair(chars.get(w, "")) == "their face":
                # a creature: its sheet fixes its body plan, whatever pose the moment takes - the dragon
                # was a coiled serpent, an upright wyvern and a giant wall of wing in one fight (10-18)
                parts.append(f"{w} is exactly the creature in image {k}: the same body shape, legs, wings, "
                             f"head, horns and scale colour, in the pose this moment describes")
            elif used.get(w) == "sheet" and (f.get("dress") or {}).get(w):
                # the sheet shows the usual clothes: copying them put the knight back in his stable
                # leathers for the fight he had forged armor for (10-18)
                parts.append(f"{w} has exactly the face and hair of the person in image {k}, but NOT their "
                             f"clothes: here {w} wears {f['dress'][w]}, in the pose this moment describes")
            elif used.get(w) == "sheet":
                parts.append(f"{w} has exactly the face, hair and clothes of the person in image {k}, in "
                             f"the pose this moment describes")
        if who:
            parts.append(video.ANATOMY)
        where = f.get("where", "")
        # the picture before as a reference only where the frame asks ("carry": the same framing a
        # moment later): given to every frame it copied the composition, duplicated people and froze
        # poses (10-16)
        # a later key frame of the same scene in the same place always carries the one before: drawn
        # alone, the three Ascension frames had four disciples, then two, then three (10-17)
        # ... but not past someone leaving: carried, the frame after the Ascension kept a Jesus-like
        # figure standing in the cloud (10-17)
        same_scene = (jobs and jobs[-1].get("pic") and jobs[-1].get("pic") == f.get("pic")
                      and not set(jobs[-1]["frame"].get("who") or []) - set(who))
        prev_same = (f.get("carry") or same_scene) and jobs and jobs[-1]["where"] == where and where
        # ... at the same framing only: carried at every shot of a scene, the forge's eight moments
        # (bellows, tongs, the hammer raised, dropped, the metal like honey) came out as one picture eight
        # times (user, 10-18). A new framing takes the place and light from it, not the composition.
        reframed = prev_same and _framing(f.get("prompt", "")) != _framing(jobs[-1]["frame"].get("prompt", ""))
        set_anchor = anchors.get(f"set:{where}")
        # Qwen-Image draws a person in ANY reference as one more person: given the moment before "for place
        # and light", the forge frame had Edric twice, in leathers and in armor (10-19). It gets the set instead.
        use_prev = prev_same and video.DRAWER != "qwen"
        if use_prev and reframed:
            refs.append(jobs[-1]["path"])
            parts.append(f"The same place and light as image {len(refs)}, seen from this moment's own framing "
                         f"and angle, NOT its composition")
        elif use_prev:                    # the moment before, in this place: carry it on
            refs.append(jobs[-1]["path"])
            parts.append(f"The same place, light and camera side as image {len(refs)}, and anyone still "
                         f"in the scene stands where they were there, except for what this moment changes")
        elif set_anchor:
            refs.append(set_anchor["path"])
            parts.append(f"The room is the one in image {len(refs)}: its furniture and every object in it stay "
                         f"exactly where they are there, the same size and shape, one of each; draw it from this "
                         f"moment's camera position")
        elif where in known and not any("," in n for n in f.get("props", [])):
            refs.append(str(video.ref_path(refs_dir, where, known[where][0])))
            parts.append(f"The setting is {where}, as in image {len(refs)}")
        elif where in places:
            # a prop in a changed state ("Iron safe, open"): the room's sheet shows it in its usual
            # state and the drawer kept both (a shut safe with a separate door standing open, 10-17)
            parts.append(f"The setting is {where}: {str(places[where]).rstrip('.')}")
        for name in f.get("props", []):
            if name in props:
                refs.append(_cut(props[name]["path"], cuts) if Path(props[name]["path"]).exists()
                            else props[name]["path"])
                parts.append(f"The {name.lower()} looks as in image {len(refs)}")
        if video.DRAWER == "qwen" and refs:
            parts.append("The grey cards of the reference images are not part of the scene: only the people, "
                         "creatures and objects on them are")
        parts += [style, "Nothing from a later period than the story: no wristwatches, no clocks, no glasses",
                  "No text, no lettering, no signature"]
        path = str(kdir / f"{f['shot']}.png")
        control = _pose_control(f.get("poses"), kdir / f"{f['shot']}.pose.png")
        if control:
            parts.insert(1, "The people stand exactly in the poses of the control skeleton")
        kprompt, kloras = _char_loras(". ".join(parts) + ".", who, chars, work)   # one person only (no boxes)
        jobs.append({"head": parts[0], "tail": ". ".join(parts[1:]), "who": who,
                     **({"control": control} if control else {}),
                     **({"loras": kloras} if kloras else {}),
                     "parent": jobs[-1]["id"] if prev_same else None,
                     "id": f["shot"], "pic": f.get("pic"), "carried": bool(prev_same),
                     "prompt": kprompt, "path": path,
                     "seed": state["seeds"].get(f["shot"], 5000 + n), "frame": f,
                     "refs": refs, "where": where, "states": st,
                     "anchors": {w: anchors[f"char:{w}"]["path"] for w in who if f"char:{w}" in anchors},
                     "creatures": {w: refs[k - 1] for k, w in enumerate(who, 1)
                                   if used.get(w) == "sheet" and _hair(chars.get(w, "")) == "their face"},
                     "set": set_anchor["path"] if set_anchor and not use_prev else None})
        images[f["shot"]] = last = path
    _drop_stale(jobs, kf, state, work, log)
    if KEYFRAME_MODE == "best" and llm_url:
        # references first (sheets, outfit sheets, props, anchors), then best-of-N per keyframe
        keep = DRAW_KEEP
        first = [j for j in video.reference_jobs(plan, style, refs_dir) + video.outfit_jobs(plan, style, refs_dir)
                 if not Path(j["path"]).exists()]
        first += [j for j in list(props.values()) + list(anchors.values()) if not Path(j["path"]).exists()]
        if first:
            log(f"  drawing {len(first)} reference(s), prop(s) and anchor(s)")
            video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in first], kdir, log)
        _select_keyframes(jobs, state, chars, plan, style, kdir, llm_url, model, log)
        gone = [j for j in jobs if not Path(j["path"]).exists()]
        if gone:      # never on to the render with a keyframe missing: one plain attempt each
            log(f"  {len(gone)} keyframe(s) without a picture: drawing them plainly")
            video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in gone], kdir, log)
        draw_end_frames(jobs, state, chars, plan, style, kdir, llm_url, model, log)
        state_f.write_text(json.dumps({k: v for k, v in state.items() if k != "_path"}, indent=1))
        return images
    for rnd in range(KEYFRAME_ROUNDS + 1):
        todo_props = [j for j in props.values() if not Path(j["path"]).exists()]
        todo_anchors = [j for j in anchors.values() if not Path(j["path"]).exists()]
        todo = [j for j in jobs if not Path(j["path"]).exists()]
        if todo_props or todo_anchors or todo:
            log(f"  drawing {len(todo_props)} prop reference(s), {len(todo_anchors)} anchor(s) and "
                f"{len(todo)} keyframe(s)")
            # sheets first if any are missing, then props, anchors (people use sheets, sets use props),
            # then keyframes in order
        final, bad, checked, futs = rnd == KEYFRAME_ROUNDS, set(), set(), []
        # each keyframe is checked (iGPU) as soon as it is saved, while the next one is drawn (XTX):
        # drawing, then checking, one after the other left each side idle half the phase (10-18)
        pool = ThreadPoolExecutor(video.checker_workers()) if llm_url else None
        by_id = {j["id"]: j for j in jobs}
        look_props = plan["look"].get("props", {})

        tries = state.setdefault("tries", {})

        def recheck(jid):
            """Check a picture just drawn; if it fails and has tries left, send it straight back to the
            drawer with a new seed and what went wrong (user 10-18: redraw on the failed check, not at
            the end of the round). Its last try is checked as final (kept and logged)."""
            j = by_id[jid]
            last = rnd + tries.get(jid, 0) >= KEYFRAME_ROUNDS
            failed = check_keyframes([j], state, chars, look_props, llm_url, model, log, last, True)
            if jid in failed and not last:
                tries[jid] = tries.get(jid, 0) + 1
                _feed_forward(jobs, {jid}, state)
                state["seeds"][jid] = j["seed"] = int(j["seed"]) + 7919
                keep = DRAW_KEEP
                # the old picture stays until the new one overwrites it: deleted first, 21 keyframes were
                # lost when the batch had closed and the round's redraw never came (10-18)
                if video.requeue({k: v for k, v in j.items() if k in keep}):
                    checked.discard(jid)
                    return set()                 # it comes back through on_done and is checked again
            return failed

        def on_done(dj):
            if pool and dj["id"] in by_id:
                checked.add(dj["id"])
                futs.append(pool.submit(recheck, dj["id"]))
        if todo_props or todo_anchors or todo:
            keep = DRAW_KEEP
            video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in
                             [j for j in video.reference_jobs(plan, style, refs_dir)
                              + video.outfit_jobs(plan, style, refs_dir)
                              if not Path(j["path"]).exists()] + todo_props + todo_anchors + todo], kdir, log,
                            on_done=on_done)
        if not llm_url:
            break
        seen = 0
        while seen < len(futs):                 # rechecks of re-queued pictures add futures as they finish
            bad |= futs[seen].result()
            seen += 1
        pool.shutdown()
        # ... and those drawn before (a restarted job) that were never passed: in parallel too (one at a
        # time, 45 left over from a restart took longer than drawing them, 10-18)
        rest = [j for j in jobs if j["id"] not in checked]
        with ThreadPoolExecutor(video.checker_workers()) as ex:
            for r in ex.map(lambda j: check_keyframes([j], state, chars, look_props, llm_url, model, log,
                                                      final, True), rest):
                bad |= r
        log(f"  checked keyframes: {len(bad)} to draw again")
        if rnd == KEYFRAME_ROUNDS:
            _edit_out_of_period(jobs, state, kdir, log)
        _feed_forward(jobs, bad, state)
        for k, j in enumerate(jobs):
            if j["id"] in bad:
                state["seeds"][j["id"]] = j["seed"] = int(j["seed"]) + 7919
                Path(j["path"]).unlink(missing_ok=True)
                for later in jobs[k + 1:]:            # carried pictures were drawn from this one
                    if not later.get("carried"):
                        break
                    Path(later["path"]).unlink(missing_ok=True)
        state_f.write_text(json.dumps(state, indent=1))
        if not bad and not [j for j in jobs if not Path(j["path"]).exists()]:
            break
    # never on to the render with a keyframe missing: draw (and check) whatever is still not on disk
    gone = [j for j in jobs if not Path(j["path"]).exists()]
    if gone:
        log(f"  {len(gone)} keyframe(s) missing after the checks: drawing them")
        keep = DRAW_KEEP
        video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in gone], kdir, log)
        if llm_url:
            with ThreadPoolExecutor(video.checker_workers()) as ex:
                list(ex.map(lambda j: check_keyframes([j], state, chars, look_props, llm_url, model, log,
                                                      True, True), gone))
        state_f.write_text(json.dumps(state, indent=1))
    draw_end_frames(jobs, state, chars, plan, style, kdir, llm_url, model, log)
    state_f.write_text(json.dumps({k: v for k, v in state.items() if k != "_path"}, indent=1))
    return images


def end_path(start: str | Path) -> Path:
    p = Path(start)
    return p.with_name(p.stem + "-end.png")


def draw_end_frames(jobs: list[dict], state: dict, chars: dict, plan: dict, style: str, kdir: Path,
                    llm_url: str | None, model: str, log) -> None:
    """The END picture of each key frame that is one move (the planner's "end"), drawn as an edit of the
    chosen start picture - same framing, people and light, only the move changed - so the video model
    animates between two pictures that agree. Drawn separately, "hammer raised" and "hammer on the blade"
    disagreed in framing and the clip morphed between them (10-19). Checked like keyframes; one redraw,
    then dropped (the clip then runs from its start alone)."""
    todo = []
    for j in jobs:
        end = (j.get("frame") or {}).get("end")
        if not isinstance(end, dict) or not str(end.get("prompt") or "").strip() or not Path(j["path"]).exists():
            continue
        import difflib
        if difflib.SequenceMatcher(None, str(end["prompt"]).lower(),
                                   str(j["frame"].get("prompt", "")).lower()).ratio() > 0.8:
            continue          # the start copied as its "end": nothing moves (the sample's parry, 10-19)
        out, stamp = end_path(j["path"]), end_path(j["path"]).with_suffix(".txt")
        st = Path(j["path"]).stat()
        key = _hash(f"{end['prompt']}|{json.dumps(end.get('poses'))}|{st.st_size}|{st.st_mtime_ns}")
        if out.exists() and stamp.exists() and stamp.read_text() == key:
            continue
        out.unlink(missing_ok=True)
        parts = [f"The same picture as image 1, a moment later: {str(end['prompt']).rstrip('. ')}",
                 "Everything else stays exactly as in image 1: the same framing, camera, light and place, the same "
                 "people each appearing once, with the same faces, hair and clothes, and the same objects"]
        control = _pose_control(end.get("poses"), kdir / f"{j['id']}-end.pose.png")
        if control:
            parts.append("The people stand exactly in the poses of the control skeleton")
        if j.get("who"):
            parts.append(video.ANATOMY)
        parts += [style, "No text, no lettering, no signature"]
        todo.append(({"id": f"{j['id']}-end", "prompt": ". ".join(parts) + ".", "path": str(out),
                      "seed": int(j["seed"]) + 31, "refs": [j["path"]], **({"control": control} if control else {})},
                     j, stamp, key))
    if not todo:
        return
    log(f"  drawing {len(todo)} end frame(s) (the end of each move)")
    look_props = plan["look"].get("props", {})
    for attempt in range(2):
        video.draw_jobs([{k: v for k, v in d.items() if k in DRAW_KEEP} for d, *_ in todo], kdir, log)
        if not llm_url:
            break
        failed = []
        for d, j, stamp, key in todo:
            if not Path(d["path"]).exists():
                continue
            cj = {**j, "id": d["id"], "path": d["path"], "seed": d["seed"],
                  "frame": {**j["frame"], "prompt": j["frame"]["end"]["prompt"]}}
            state.get("passed", {}).pop(d["id"], None)
            if check_keyframes([cj], state, chars, look_props, llm_url, model, log, attempt == 1, True,
                               hard_only=True):
                failed.append((d, j, stamp, key))
            else:
                stamp.write_text(key)
        if not failed:
            break
        for d, *_ in failed:
            Path(d["path"]).unlink(missing_ok=True)
            d["seed"] += 7919
        todo = failed
    else:
        if todo:
            log(f"  {len(todo)} end frame(s) failed twice: those clips run from their start alone")
    if not llm_url:
        for d, j, stamp, key in todo:
            if Path(d["path"]).exists():
                stamp.write_text(key)


# --- best-of-N keyframes (plan 10-18: PLAN-keyframes-2026-10-18.md) ---------------------------------
# The retry loop (draw, check, redraw up to 3 times) spent 4.5 h on the knight's 56 keyframes: 114 of 167
# failed checks were scene facts a reseed fixed 15% of the time. Instead: N candidates up front on both
# cards, a fast yes/no ranking of each, ONE careful check of the winner (the runner-up if it fails), a
# single rewrite of a frame nothing could draw, and the best kept - no serial rounds.
# 10-19 validation (tools/validate_keyframes.py, knight labels): the fast yes/no ranking flagged 44% of good
# frames and 56% of flawed ones - no better than chance - while costing a checker call per candidate. Back to
# "retry": one picture, the careful check as soon as it is saved, a redraw at once on a failure.
KEYFRAME_MODE = os.environ.get("BOOKCASTER_KEYFRAME_MODE", "retry")
KEYFRAME_N = int(os.environ.get("BOOKCASTER_KEYFRAME_N", "3"))
QUESTION_WEIGHT = {"entity": 3.0, "outfit": 3.0, "relation": 2.0, "fact": 1.5, "setting": 1.0,
                   "prop": 1.0, "count": 1.0}


def _frame_questions(j: dict, chars: dict, plan: dict) -> list[dict]:
    """The atomic yes/no questions a candidate is ranked on: the planner's "checks" (typed, with parents)
    where it wrote them, else made from the frame: who is there, what they wear, its "must" facts."""
    f = j["frame"]
    qs = [dict(q) for q in f.get("checks") or [] if isinstance(q, dict) and q.get("q")]
    if not qs:
        for w in j.get("who") or []:
            qs.append({"q": f"Is {w} ({video._short(chars.get(w, ''))}) in the picture?", "type": "entity"})
        for m in f.get("must") or []:
            m = str(m)
            if m.startswith(WARDROBE_MUST):
                qs.append({"q": f"Does {m[len(WARDROBE_MUST):].rstrip('.')}?".replace(" wears ", " wear ", 1),
                           "type": "outfit"})
            else:
                qs.append({"q": f"Is this true of the picture: {m.rstrip('.')}?", "type": "fact"})
    return [q for q in qs if not _unseeable(q["q"])][:8]


UNSEEABLE = re.compile(r"\b(warm|hot|cold|heat|sound|loud|quiet|smell|feel|feels|feeling|thinks?|remembers?|"
                       r"north|south|east|west|walking towards|heading|before|after|then|trembl\w*|"
                       r"camera|angle|close-up|wide shot|medium shot|framing)\b", re.I)


def _unseeable(q: str) -> bool:
    """A question no still picture can answer (heat, sound, a feeling, direction of travel, time) or
    about the framing (a drawing instruction, not a fact): never asked."""
    return bool(UNSEEABLE.search(q))


def _answer(png: Path, questions: list[dict], url: str, model: str) -> list[bool] | None:
    """One fast call: yes/no to each question, no reasoning (ranking only; the careful check reasons)."""
    if not questions:
        return []
    q = ("Look at this illustration and answer each question with yes or no.\n"
         + "\n".join(f"{i}. {x['q']}" for i, x in enumerate(questions, 1)) +
         '\nReply with only a JSON list of answers in order, like ["yes", "no"].')
    try:
        r = video._look_at(url, model, png, q)
    except Exception:
        return None
    m = re.findall(r"\[[^\[\]]*\]", r or "")
    try:
        got = json.loads(m[-1]) if m else []
    except ValueError:
        return None
    ans = [str(a).strip().lower().startswith("y") for a in got][:len(questions)]
    return ans + [False] * (len(questions) - len(ans))


YES_PROB = os.environ.get("BOOKCASTER_YES_PROB", "0") == "1"


def _answer_probs(png: Path, questions: list[dict], url: str, model: str) -> list[float] | None:
    """P(yes) for each question from the answer tokens' probabilities (VQAScore-style: a confident yes
    outranks a hesitant one). None when the server gives no probabilities."""
    import math
    q = ("Look at this illustration and answer each question with yes or no.\n"
         + "\n".join(f"{i}. {x['q']}" for i, x in enumerate(questions, 1)) +
         '\nReply with only a JSON list of answers in order, like ["yes", "no"].')
    try:
        text, toks = video._look_at_probs(url, model, png, q)
    except Exception:
        return None
    probs = []
    for tok, alts in toks:
        w = tok.strip().strip('"').lower()
        if w not in ("yes", "no"):
            continue
        py = sum(math.exp(v) for a, v in alts.items() if a.strip().strip('"').lower() == "yes")
        pn = sum(math.exp(v) for a, v in alts.items() if a.strip().strip('"').lower() == "no")
        probs.append(py / (py + pn) if py + pn else (1.0 if w == "yes" else 0.0))
    if len(probs) < len(questions):
        return None
    return probs[:len(questions)]


def _score(questions: list[dict], answers: list[bool]) -> float:
    """Weighted yes count; a child question counts no when its parent is no (DSG dependencies)."""
    total = 0.0
    for i, (q, a) in enumerate(zip(questions, answers)):
        par = q.get("parent")
        if isinstance(par, int) and 0 <= par < len(answers) and float(answers[par]) < 0.5:
            a = False
        total += QUESTION_WEIGHT.get(q.get("type", "fact"), 1.0) * (float(a) if not isinstance(a, bool) else
                                                                     (1.0 if a else 0.0))
    return total


# --- phase 2: detectors on the CPU (tools/detect.py) ------------------------------------------
# Counting people is where a vision model is weakest (~40% wrong in the studies; it missed a stray head
# under the dragon's wing): Grounding DINO counts them, DINOv2 says how alike each figure is to its sheet.
# Off until calibrated (BOOKCASTER_DETECT=1); thresholds from the validation set.
DETECT_ON = os.environ.get("BOOKCASTER_DETECT", "0") == "1"
SAME_ALARM = float(os.environ.get("BOOKCASTER_SAME_ALARM", "0.55"))


class _Detector:
    """One detect.py process kept loaded (iGPU, else CPU), asked one picture at a time."""

    def __init__(self):
        self.proc, self.lock = None, threading.Lock()

    def ask(self, job: dict) -> dict | None:
        with self.lock:
            try:
                if self.proc is None or self.proc.poll() is not None:
                    # the iGPU (1.1 s a picture; the CPU takes 15 s): while keyframes draw it only holds the
                    # language model, and its venv has the gfx1151 kernels the drawing venv lacks
                    py = SKYREELS / ".venv-igpu" / "bin" / "python"
                    if py.exists():
                        ig = _gpu("igpu")
                        env = {**os.environ, "HIP_VISIBLE_DEVICES": ig, "HSA_CU_MASK": f"{ig}:0-38",
                               "DETECT_DEVICE": "cuda"}
                    else:
                        py, env = video.ART_PYTHON, {**os.environ, "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": ""}
                    self.proc = subprocess.Popen(
                        ["nice", "-n", "10", str(py), "-I",
                         str(Path(__file__).resolve().parent.parent / "tools" / "detect.py")],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
                self.proc.stdin.write(json.dumps(job) + "\n")
                self.proc.stdin.flush()
                while True:
                    line = self.proc.stdout.readline()
                    if not line:
                        return None
                    msg = json.loads(line)
                    if msg.get("id") == job["id"]:
                        return msg
            except Exception:
                return None

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)


DETECTOR = _Detector()


def _detector_faults(png: Path, j: dict, chars: dict) -> tuple[list[str], list[str]]:
    """(hard failures, identity alarms) from the detector: more people than the frame has (no crowd), and
    characters whose best-matching figure is unlike their sheet (checked side by side, then)."""
    if not DETECT_ON:
        return [], []
    who = j.get("who") or []
    humans = [w for w in who if _hair(chars.get(w, "")) != "their face"]
    crowd = str(j["frame"].get("crowd") or "").strip()
    figs = {w: r for w, r in zip(who, j.get("refs", [])) if w in humans}
    res = DETECTOR.ask({"id": png.stem, "image": str(png), "count": ["person", "face"], "same": figs})
    if not res or "error" in res:
        return [], []
    hard, alarms = [], []
    n = res["counts"].get("person", 0)
    if not crowd and humans and n > len(humans):
        hard.append(f"{n} people detected where the frame has {len(humans)}")
    for w, sim in (res.get("same") or {}).items():
        if sim < SAME_ALARM:
            alarms.append(w)
    return hard, alarms


# --- phase 3: fix a region in place --------------------------------------------------------------------
# A frame whose only fault is local (a third hand, a floating hammer, a stray head) needn't be redrawn
# whole: the vision model points at the fault, the drawing model repaints that crop from itself, and the
# patch is feathered back in; checked again, kept only if it passes. Off until validated.
REGION_FIX = os.environ.get("BOOKCASTER_REGION_FIX", "0") == "1"
LOCAL = re.compile(r"\b(arm|arms|hand|hands|finger|fingers|leg|legs|head|heads|face|limb|limbs|floating|"
                   r"merged|hammer|tool)\b", re.I)

_PASTE = r"""
import sys
from PIL import Image, ImageFilter, ImageDraw
src, patch, out, x0, y0, x1, y1 = sys.argv[1:8]
x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
im = Image.open(src).convert("RGB")
p = Image.open(patch).convert("RGB").resize((x1 - x0, y1 - y0), Image.LANCZOS)
m = Image.new("L", p.size, 0)
pad = max(4, min(p.size) // 8)
ImageDraw.Draw(m).rectangle((pad, pad, p.size[0] - pad, p.size[1] - pad), fill=255)
m = m.filter(ImageFilter.GaussianBlur(pad / 2))
im.paste(p, (x0, y0), m)
im.save(out)
"""

_CROP = r"""
import sys
from PIL import Image
src, out, x0, y0, x1, y1 = sys.argv[1:7]
Image.open(src).convert("RGB").crop(tuple(map(int, (x0, y0, x1, y1)))).save(out)
"""


def _locate(png: Path, fault: str, url: str, model: str) -> list[float] | None:
    q = (f"In this illustration, find this fault: {fault}. Reply with only the box around it as fractions of "
         f"the width and height: [x0, y0, x1, y1]. If there is no such fault, reply [].")
    try:
        r = video._look_at(url, model, png, q)
    except Exception:
        return None
    m = re.findall(r"\[([^\[\]]*)\]", r or "")
    try:
        b = [float(x) for x in m[-1].split(",")] if m and m[-1].strip() else []
    except ValueError:
        return None
    if len(b) != 4:
        return None
    if max(b) > 1.5:                     # pixels or 0-1000 (Qwen grounding): to fractions
        scale = 1000.0 if max(b) <= 1000 else None
        if not scale:
            return None
        b = [x / scale for x in b]
    x0, y0, x1, y1 = b
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)] if x1 - x0 > 0.01 and y1 - y0 > 0.01 else None


def _fix_region(cj: dict, j: dict, faults: list[str], chars: dict, look_props: dict, llm_url: str,
                model: str, style: str, log) -> bool:
    """Repaint the region of the first local fault in candidate cj; True if the fixed picture then passes
    the careful check (it replaces the candidate's file)."""
    local = [f for f in faults if LOCAL.search(f)]
    if not REGION_FIX or not local or len(local) != len(faults) or video._SESSION is None:
        return False
    png = Path(cj["path"])
    box = _locate(png, local[0], video.checker_url(llm_url), model)
    if not box:
        return False
    W, H = 1536, 864
    try:
        out = subprocess.run([str(video.ART_PYTHON), "-I", "-c",
                              "import sys;from PIL import Image;print(*Image.open(sys.argv[1]).size)", str(png)],
                             capture_output=True, text=True, check=True).stdout.split()
        W, H = int(out[0]), int(out[1])
    except Exception:
        pass
    cx, cy = (box[0] + box[2]) / 2 * W, (box[1] + box[3]) / 2 * H
    side = max(256, int(max((box[2] - box[0]) * W, (box[3] - box[1]) * H) * 1.6))
    x0, y0 = int(max(0, min(W - side, cx - side / 2))), int(max(0, min(H - side, cy - side / 2)))
    x1, y1 = min(W, x0 + side), min(H, y0 + side)
    crop = png.with_name(png.stem + ".crop.png")
    patch = png.with_name(png.stem + ".patch.png")
    subprocess.run([str(video.ART_PYTHON), "-I", "-c", _CROP, str(png), str(crop), str(x0), str(y0), str(x1), str(y1)],
                   check=True)
    size = [max(256, (x1 - x0) // 16 * 16), max(256, (y1 - y0) // 16 * 16)]
    done = threading.Event()
    video._SESSION.put({"id": png.stem + "-fix", "prompt": (
        f"Exactly the same picture as image 1, the same style, colours, light and everything in it, except "
        f"that this is fixed: {local[0]}. Correct anatomy, every hand belongs to an arm, nothing floating. "
        f"{style}. No text."), "seed": int(cj.get("seed", 1)) + 1, "refs": [str(crop)], "path": str(patch),
        "size": size}, lambda _j: done.set())
    if not done.wait(timeout=600) or not patch.exists():
        return False
    fixed = png.with_name(png.stem + ".fixed.png")
    subprocess.run([str(video.ART_PYTHON), "-I", "-c", _PASTE, str(png), str(patch), str(fixed),
                    str(x0), str(y0), str(x1), str(y1)], check=True)
    st = {"seeds": {}, "passed": {}, "log": []}
    probe = dict(j, path=str(fixed), id=png.stem + "-fixed")
    check_keyframes([probe], st, chars, look_props, llm_url, model, log, False, True, hard_only=True)
    if probe["id"] in st["passed"]:
        fixed.replace(png)
        log(f"    {j['id']}: fixed in place ({local[0][:80]})")
        return True
    return False


def _rewrite_frame(j: dict, failing: list[str], llm_url: str, model: str) -> str | None:
    """The frame's picture description rewritten once so the drawing model can draw it: what no candidate
    showed comes first and plainly, the composition simpler."""
    from .score import _chat
    q = ("An image model drew this keyframe several times and never showed: " + "; ".join(failing[:5]) +
         ".\nThe description it was given:\n" + j["frame"].get("prompt", "") +
         "\nRewrite the description (2-4 sentences) so it can be drawn: put what was missed first and say it "
         "plainly and physically (who holds what, where each person is), simplify the composition, keep the "
         "same people, place and moment, start with the framing. Reply with only the new description.")
    try:
        r = (_chat(llm_url, model, q, max_tokens=400) or "").strip().strip('"')
    except Exception:
        return None
    return r if 40 <= len(r) <= 900 else None


def _select_keyframes(jobs: list[dict], state: dict, chars: dict, plan: dict, style: str, kdir: Path,
                      llm_url: str, model: str, log) -> None:
    import shutil
    cdir = kdir / "cands"
    cdir.mkdir(parents=True, exist_ok=True)
    by_id = {j["id"]: j for j in jobs}
    done_ids = {j["id"] for j in jobs if Path(j["path"]).exists() and j["id"] in state.get("passed", {})}
    todo = [j for j in jobs if j["id"] not in done_ids]
    if not todo:
        return
    chosen = state.setdefault("chosen", {})
    lock, all_done = threading.Lock(), threading.Event()
    left = {j["id"] for j in todo}
    cands: dict[str, list] = {j["id"]: [] for j in todo}        # [(job, score, answers)]
    waiting: dict[str, int] = {}
    rewritten: set = set()
    look_props = plan["look"].get("props", {})
    pool = ThreadPoolExecutor(video.checker_workers() + 1)
    t0 = time.time()
    log(f"  keyframes: best of {KEYFRAME_N} for {len(todo)} frame(s) on {len(video.DRAW_BUSES)} card(s)")

    def submit(j: dict, n: int, salt: int = 0) -> None:
        base = int(state["seeds"].get(j["id"], j["seed"]))
        with lock:
            waiting[j["id"]] = waiting.get(j["id"], 0) + n
        for k in range(n):
            cid = f"{j['id']}~{salt + k}"
            cj = {"id": cid, "prompt": j["prompt"], "seed": base + (salt + k) * 7919, "refs": j["refs"],
                  "path": str(cdir / f"{cid}.png"), **({"control": j["control"]} if j.get("control") else {})}
            video._SESSION.put(cj, on_drawn)

    progress = {"t": time.time()}

    def on_drawn(cj: dict) -> None:
        pool.submit(guarded, rank, cj)

    def guarded(fn, arg):
        try:
            fn(arg)
        except Exception as exc:              # never hang the stage: keep what there is for that frame
            jid = (arg["id"] if isinstance(arg, dict) else arg).split("~")[0]
            log(f"    {jid}: selection error ({type(exc).__name__}: {exc}); keeping the best so far")
            fallback(jid)

    def fallback(jid: str) -> None:
        j = by_id[jid]
        with lock:
            have = sorted((c for c in cands[jid] if Path(c[0]["path"]).exists()), key=lambda c: -c[1])
        if have and not Path(j["path"]).exists():
            shutil.copyfile(have[0][0]["path"], j["path"])
        finish(jid, have[0] if have else None)

    def finish(jid: str, best) -> None:
        j = by_id[jid]
        with lock:
            if jid not in left:
                return
            if best is not None:
                chosen[jid] = {"cand": best[0]["id"], "score": best[1], "of": len(cands[jid])}
                state["seeds"][jid] = best[0]["seed"]
            if Path(j["path"]).exists():
                state["passed"][jid] = Path(j["path"]).stat().st_mtime
            left.discard(jid)
            progress["t"] = time.time()
            kids = [k for k in todo if k.get("parent") == jid]
            finished = not left
        _save_checks(state)
        for k in kids:                                     # drawn from this one: now they can start
            submit(k, KEYFRAME_N)
        if finished:
            all_done.set()

    def rank(cj: dict) -> None:
        jid = cj["id"].split("~")[0]
        j = by_id[jid]
        qs = _frame_questions(j, chars, plan)
        drawn = Path(cj["path"]).exists()
        url = video.checker_url(llm_url)
        probs = _answer_probs(Path(cj["path"]), qs, url, model) if drawn and YES_PROB else None
        ans = ([p >= 0.5 for p in probs] if probs is not None else
               _answer(Path(cj["path"]), qs, url, model) if drawn else None)
        sc = (_score(qs, probs) if probs is not None else
              _score(qs, ans) if ans is not None else (-1.0 if drawn else -9.0))
        with lock:
            cands[jid].append((cj, sc, ans, qs))
            waiting[jid] -= 1
            ready = waiting[jid] == 0
        if ready:
            pool.submit(guarded, decide, jid)

    def careful(cj: dict, j: dict) -> list[str]:
        """The full, reasoned check (arm by arm, identity, outfit), hard items only; [] = passes. With the
        detectors on, a detected extra person fails it outright, and a figure unlike its sheet gets the
        side-by-side identity comparison."""
        hard, alarms = _detector_faults(Path(cj["path"]), j, chars)
        if hard:
            return hard
        probe = dict(j, path=cj["path"], id=cj["id"])
        if alarms:
            probe["anchors"] = {**(j.get("anchors") or {}),
                                **{w: r for w, r in zip(j.get("who") or [], j.get("refs", [])) if w in alarms}}
        st = {"seeds": {}, "passed": {}, "log": []}
        check_keyframes([probe], st, chars, look_props, llm_url, model, log, False, True, hard_only=True)
        return [] if cj["id"] in st["passed"] else (st["log"][-1].get("missing", []) + st["log"][-1].get("problems", [])
                                                   if st["log"] else ["unchecked"])

    def decide(jid: str) -> None:
        j = by_id[jid]
        with lock:
            ranked = sorted((c for c in cands[jid] if Path(c[0]["path"]).exists()), key=lambda c: -c[1])
        if not ranked:                                     # nothing drawn at all (drawer errors)
            log(f"    {jid}: no candidate could be drawn")
            finish(jid, None)
            return
        best, why = None, []
        if jid in rewritten:      # after a rewrite its own candidates are tried first (the old ones failed)
            new = [c for c in ranked if int(c[0]["id"].rsplit("~", 1)[1]) >= KEYFRAME_N]
            ranked = new + [c for c in ranked if c not in new]
        for cj, sc, ans, qs in ranked[:2]:               # the winner, else the runner-up
            fails = careful(cj, j)
            if not fails:
                best = (cj, sc)
                break
            why = why or fails
        if best is None and jid not in rewritten:
            # nothing drew it: unrenderable as asked, not unlucky - ask differently, once
            missed = sorted({q["q"] for cj, sc, ans, qs in ranked for q, a in zip(qs, ans or []) if not a})
            new = _rewrite_frame(j, why + missed, llm_url, model)
            with lock:
                rewritten.add(jid)
            if new:
                j["prompt"] = _tag_people(new.rstrip(". "), j.get("who") or [], chars, j["frame"].get("dress")) + \
                    ". " + j["tail"] + "."
                j["frame"]["prompt"] = new
                log(f"    {jid}: no candidate passed ({'; '.join(why)[:100]}): rewritten, 2 more")
                submit(j, 2, salt=KEYFRAME_N)
                return
        if best is None and why and _fix_region(ranked[0][0], j, why, chars, look_props, llm_url, model,
                                                style, log):
            best = (ranked[0][0], ranked[0][1])        # repaired where it was wrong, then passed
            why = []
        if best is None:                                  # keep the best-ranked, logged
            best = (ranked[0][0], ranked[0][1])
            log(f"    {jid} kept as the best of {len(ranked)}: {'; '.join(why)[:160]}")
            state["log"].append({"id": jid, "seed": best[0]["seed"], "missing": why, "problems": [], "kept": True})
        shutil.copyfile(best[0]["path"], j["path"])
        finish(jid, best)

    for j in todo:
        if not j.get("parent") or j["parent"] in done_ids:
            submit(j, KEYFRAME_N)
    while not all_done.wait(timeout=10):
        if video._SESSION is None or video._SESSION.broken:
            log("  keyframes: the drawing cards stopped; keeping what was chosen")
            break
        if time.time() - progress["t"] > 1200:            # nothing decided for 20 min: don't hang the night
            log(f"  keyframes: no progress for 20 min; keeping the best so far for {len(left)} frame(s)")
            for jid in list(left):
                fallback(jid)
            break
    pool.shutdown(wait=True)
    log(f"  keyframes chosen in {(time.time() - t0) / 60:.0f} min "
        f"({sum(1 for c in chosen.values() if c.get('of')) } picked, "
        f"{sum(1 for e in state['log'] if e.get('kept'))} kept with faults, {len(rewritten)} rewritten)")


def _feed_forward(jobs: list[dict], bad: set, state: dict) -> None:
    """A redraw is told what went wrong, not just given a new seed (the same prompt failed the same way
    again); and what went wrong with someone in one frame (the dragon's shape, the knight's armor, an
    extra head) is said to the scene's other frames being redrawn too (user, 10-18: "feed the fixes
    forward"). The notes go after the prompt as "Make sure: ..."."""
    last, mistakes = {}, {}
    for e in state.get("log", []):
        last[e["id"]] = [m for m in e.get("missing", []) + e.get("problems", []) if m]
        mistakes[e["id"]] = [m for m in e.get("problems", []) if m]     # drawing errors: every frame's concern
    for j in jobs:
        if j["id"] not in bad:
            continue
        own = last.get(j["id"], [])
        who = set(j["frame"].get("who") or [])
        shared = []
        for o in jobs:                     # the same scene's failures about the same people, or drawing errors
            if o is j or o.get("pic") != j.get("pic"):
                continue
            for m in last.get(o["id"], []):
                if m in mistakes.get(o["id"], []) or any(re.search(rf"\b{re.escape(w)}\b", m) for w in who):
                    shared.append(m)
        notes = list(dict.fromkeys(own + shared))[:5]
        base = j.setdefault("base_prompt", j["prompt"])
        j["prompt"] = base + (" Make sure: " + "; ".join(n.rstrip(". ") for n in notes) + "." if notes else "")


KEYFRAME_ROUNDS = int(os.environ.get("BOOKCASTER_KEYFRAME_ROUNDS", "3"))
PERIOD_ITEM = ("nobody wears or holds anything from a later period than the story, "
               "such as a wristwatch or a wall clock")


def _edit_out_of_period(jobs: list[dict], state: dict, kdir: Path, log) -> None:
    """A keyframe kept after its redraws with something from a later period in
    it: the drawing model paints the same picture again, with the keyframe as
    its reference, minus that thing. A disciple's wristwatch came back with the
    same seed on every redraw, and cloning the wrist by hand left a pale band the
    video model turned back into a watch; the whole-picture edit stayed faithful
    and left a bare wrist (10-18)."""
    edits = state.pop("edits", {})
    todo = [j for j in jobs if j["id"] in edits and Path(j["path"]).exists()]
    if not todo:
        return
    log(f"  painting out later-period things in {len(todo)} keyframe(s)")
    out = {j["id"]: kdir / f"{j['id']}-edit.png" for j in todo}
    video.draw_jobs([{"id": f"{j['id']}-edit", "seed": int(j["seed"]) + 1, "path": str(out[j["id"]]),
                      "refs": [j["path"]], **({"size": j["size"]} if j.get("size") else {}),
                      "prompt": "Exactly the same painting as the reference image, unchanged in every "
                                "detail, except that the thing from a later period than the story "
                                f"described here is gone: {edits[j['id']]} Nobody wears or holds it: "
                                "bare skin, cloth or wall in its place."} for j in todo], kdir, log)
    for j in todo:
        if out[j["id"]].exists():
            out[j["id"]].replace(j["path"])
            state["passed"][j["id"]] = Path(j["path"]).stat().st_mtime
            log(f"    {j['id']}: painted out ({edits[j['id']][:120]})")


# --- continuity bible ------------------------------------------------------------------
# Every keyframe drawn on its own gave a different Roylott in each frame, a safe that moved, changed
# shape and doubled, a band on his head in some frames only, a dead man whose eyes opened, and a cane
# fused with the lit match into one burning stick (10-17). keyframes.json can now carry:
#   "anchors": {"chars": {name: {"prompt": how they look IN THIS SCENE}},
#               "sets":  {place: {"prompt": the room's layout, "props": [...]}}}
#     drawn first; every keyframe with that person / in that room takes the anchor as its reference
#     (instead of the generic sheet / empty-room sheet), and the check compares them side by side;
#   "states": [{"who": name?, "from": shot, "to": shot, "prompt": ..., "motion": ..., "must": [...]}]
#     added to every frame in the range (that shows *who*, if given): picture, motion and checks.

def _hash(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def anchor_jobs(kf: dict, plan: dict, style: str, work: Path, props: dict) -> dict[str, dict]:
    """Drawing jobs for the scene's anchors: "char:<name>" and "set:<place>"."""
    chars, places = video.looks(plan)
    known = video.ref_prompts(plan, style)
    refs_dir, adir = work.parent / "refs", work / "anim" / "anchors"
    out = {}
    for name, a in (kf.get("anchors") or {}).get("chars", {}).items():
        refs = [str(video.ref_path(refs_dir, name, known[name][0]))] if name in known else []
        prompt = (f"Portrait of {name}, head, shoulders and upper body, three-quarter view facing the viewer, "
                  f"against a plain dark background: {a['prompt'].rstrip('. ')}. "
                  + (f"Image 1 is a character sheet showing this person twice: draw ONE person, with that face. "
                     if refs else "") + f"{video.ANATOMY}. {style}. No text, no lettering, no signature.")
        out[f"char:{name}"] = {"id": f"anchor-{video.slug(name)}", "prompt": prompt, "seed": 77, "refs": refs,
                               "path": str(adir / f"char-{video.slug(name)}-{_hash(prompt)}.png")}
    for place, a in (kf.get("anchors") or {}).get("sets", {}).items():
        refs, parts = [], [f"Establishing wide view of {place}, no people: {a['prompt'].rstrip('. ')}"]
        if place in known:
            refs.append(str(video.ref_path(refs_dir, place, known[place][0])))
            parts.append(f"The room is as in image {len(refs)}")
        for n in a.get("props", []):
            if n in props:
                refs.append(props[n]["path"])
                parts.append(f"the {n.lower()} looks as in image {len(refs)}, and there is only ONE of it")
        prompt = ". ".join(parts + [style, "No text, no lettering, no signature"]) + "."
        out[f"set:{place}"] = {"id": f"anchor-set-{video.slug(place)}", "prompt": prompt, "seed": 78,
                               "refs": refs, "path": str(adir / f"set-{video.slug(place)}-{_hash(prompt)}.png")}
    if video.DRAWER == "qwen":
        # every place gets an empty picture of itself: Qwen draws people from any reference as more people, so
        # it can't take the moment before for the place and light, and with only character sheets and props
        # it drew the sample's scenes on blank white (10-19)
        for place in dict.fromkeys(f.get("where") for f in kf.get("frames", []) if f.get("where")):
            if f"set:{place}" in out:
                continue
            desc = str(places.get(place, "")).rstrip(". ")
            # inside or out, from the moments set there: drawn as a cottage seen from outside, the forge's
            # place picture would have put every smithing shot out of doors (10-19)
            said = " ".join(str(f.get("prompt", "")) for f in kf.get("frames", []) if f.get("where") == place)
            inside = len(re.findall(r"\b(inside|indoors|interior|in the (?:forge|room|hall|chamber|cell|kitchen|"
                                    r"workshop|smithy|house|castle|church|tavern)|anvil|hearth|fireplace|"
                                    r"table|workbench|ceiling)\b", said, re.I))
            outside = len(re.findall(r"\b(outside|outdoors|sky|field|river|mountain|ledge|valley|road|path|"
                                     r"forest|clouds?)\b", said, re.I))
            view = (f"Interior of {place}, seen from inside" if inside > outside else
                    f"Establishing wide view of {place}")
            prompt = (f"{view}, empty, no people and no creatures"
                      + (f": {desc}" if desc else "") + f". {style}. No text, no lettering, no signature.")
            out[f"set:{place}"] = {"id": f"anchor-set-{video.slug(place)}", "prompt": prompt, "seed": 78, "refs": [],
                                   "path": str(adir / f"set-{video.slug(place)}-{_hash(prompt)}.png")}
    return out


def states_for(kf: dict, shot: str, who: list[str]) -> list[dict]:
    """The states in force at *shot* for the people in it (or for everyone, when no "who")."""
    order = [f["shot"] for f in kf["frames"]]
    if shot not in order:
        return []
    i = order.index(shot)
    out = []
    for st in kf.get("states", []):
        a = order.index(st["from"]) if st.get("from") in order else 0
        b = order.index(st["to"]) if st.get("to") in order else len(order) - 1
        if a <= i <= b and (not st.get("who") or st["who"] in who):
            out.append(st)
    return out


def _side_by_side(left: str, right: Path, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", left, "-i", str(right), "-filter_complex",
                    "[0:v]scale=-2:512[a];[1:v]scale=-2:512[b];[a][b]hstack=inputs=2", str(out)], check=True)
    return out


def _compare(j: dict, png: Path, llm_url: str, model: str) -> list[str]:
    """The keyframe next to its anchors: the same person (face, age, hair)? the room's furniture and
    objects in the same places, same size and shape, one of each? Returns what failed."""
    tmp = png.with_name(f".cmp-{png.stem}.png")
    fails = []
    for w, a in (j.get("anchors") or {}).items():
        q = (f"The LEFT picture shows {w}. Is {w} in the RIGHT picture the same person: the same face, the "
             f"same age and the same hair? Ignore clothes, pose, light and framing. Answer with a final line of "
             f'JSON: {{"same": true/false, "why": "..."}}')
        r = video._look_at(llm_url, model, _side_by_side(a, png, tmp), q)
        m = re.findall(r"\{[^{}]*\"same\"[^{}]*\}", r or "")
        if m and json.loads(m[-1]).get("same") is False:
            fails.append(f"{w} is not the same person as the anchor: {json.loads(m[-1]).get('why', '')[:120]}")
    for w, ref in (j.get("creatures") or {}).items():
        q = (f"The LEFT picture shows {w}. Is the creature in the RIGHT picture the same creature: the same "
             f"body plan (number of legs, wings, neck and tail), the same head and horns, the same scale "
             f"colour? Ignore its pose, size and the light. Answer with a final line of JSON: "
             f'{{"same": true/false, "why": "..."}}')
        r = video._look_at(llm_url, model, _side_by_side(ref, png, tmp), q)
        m = re.findall(r"\{[^{}]*\"same\"[^{}]*\}", r or "")
        if m and json.loads(m[-1]).get("same") is False:
            fails.append(f"{w} is not the same creature as its sheet: {json.loads(m[-1]).get('why', '')[:120]}")
    if j.get("set") and "close-up" not in j["frame"].get("prompt", "").lower():
        q = ("The LEFT picture is a room. The RIGHT picture is the same room at a later moment, maybe from "
             "another angle. Are the furniture and the objects of the room in the same places, the same size "
             "and the same shape, with no object doubled or missing? People do not matter. Answer with a final "
             'line of JSON: {"same": true/false, "why": "..."}')
        r = video._look_at(llm_url, model, _side_by_side(j["set"], png, tmp), q)
        m = re.findall(r"\{[^{}]*\"same\"[^{}]*\}", r or "")
        if m and json.loads(m[-1]).get("same") is False:
            fails.append(f"the room differs from its anchor: {json.loads(m[-1]).get('why', '')[:120]}")
    tmp.unlink(missing_ok=True)
    return fails


HELM_ON = re.compile(r"(?<!\bno )(?<!\bwithout )\b(helm|helmet|visor|hood|mask)\b(?![^,;]*\b(off|removed|under|held|carried|"
                     r"lowered|back)\b)", re.I)


def _recognisable(w: str, look: str, dress: str | None) -> str:
    """The check item that tells *w* apart: hair, and clothes where they show. A covered head (the
    knight's helm) is known by the armor alone - asked for his hair, 15 armored frames failed (10-18);
    a creature with no hair (the dragon) by its look, not by "their face" and empty clothes."""
    hair = _hair(look)
    clothes = _from_behind(look, dress).split(", ", 1)[-1] if (dress or "wearing" in look) else ""
    if hair == "their face":
        # as far as it is in view: the close-up of the dragon's eye failed for its legs (10-18)
        return f"{w} is recognisable as {video._short(look)}, as far as it is in view"
    if dress and HELM_ON.search(dress):
        return f"{w} is recognisable by what he or she wears: {_gist(dress)} (the head may be covered)"
    return f"{w} is recognisable by {hair}" + (f" (and, where their clothes show, {clothes})" if clothes else "")


_CHECKS_LOCK = threading.Lock()


def _save_checks(state: dict) -> None:
    """Keyframe checks saved per picture (a stopped job kept none of a round's checks, 10-18); the
    checks run several at once on the XTX checker, hence the lock."""
    if state.get("_path"):
        with _CHECKS_LOCK:
            for _ in range(5):      # another check may add to the state while it is written out
                try:
                    text = json.dumps(state, indent=1)
                    break
                except RuntimeError:
                    time.sleep(0.05)
            else:
                return
            Path(state["_path"]).write_text(text)


ANATOMY_ITEMS = [
    "every person has exactly two arms, two hands and two legs, and one head (trace each arm from its "
    "shoulder); no extra or floating head, face or limb anywhere",
    "no face or body is merged into furniture, clothing, a creature or another object",
    "it is one single picture: not split into panels, and nobody appears twice"]


def check_keyframes(jobs: list[dict], state: dict, chars: dict, props: dict, llm_url: str,
                    model: str, log, final: bool = False, quiet: bool = False,
                    hard_only: bool = False) -> set[str]:
    """Show each keyframe not yet passed to the vision model with its frame's
    checklist: how many people, each one by hair and clothes, the props it names,
    and the frame's "must" lines (states and who holds what: "the safe's door is
    wide open", "Watson holds the revolver", "a small square ventilator hole high
    in the wall, no lamp"). Plus the usual drawing mistakes (video.check_picture).
    Returns the shot ids to draw again (none on the final round: those are logged
    and kept). A missing vision server skips the check."""
    bad = set()
    for j in jobs:
        png = Path(j["path"])
        if not png.exists() or state["passed"].get(j["id"]) == png.stat().st_mtime:
            continue
        f = j["frame"]
        who = [w for w in f.get("who", []) if w in chars]
        # fair to the framing (10-17: whole looks and whole prop descriptions failed close-ups for
        # slippers that weren't in shot, hands for being "a person", a bell-pull for its tassel)
        crowd = str(f.get("crowd") or "").strip()
        items = ([f"a group of other people besides {' and '.join(who) or 'nobody named'}: {crowd}"] if crowd else
                 [f"exactly {len(who)} {'figure' if len(who) == 1 else 'figures'} with a head in the picture, "
                  f"people or creatures (a dragon or an animal counts as one; count heads, including small or "
                  f"background figures; hands alone do not count)"])
        items += [_recognisable(w, chars[w], (f.get("dress") or {}).get(w)) for w in who]
        # a prop only where the moment's own words have it: "a recognisable hammer" failed the shot of the
        # knight in his finished breastplate, three times over (10-18)
        said = f"{f.get('prompt', '')} {' '.join(f.get('must') or [])}".lower()
        items += [f"a recognisable {n.lower().replace(', ', ' (')}{')' if ', ' in n else ''}: "
                  f"{re.split(r'[,;]', props[n])[0]}" for n in f.get("props", [])
                  if n in props and re.split(r"[ ,(]", n.lower())[0] in said]
        if hard_only:      # best-of-N: scene facts and props rank candidates, they don't fail them
            items = [x for x in items if not x.startswith("a recognisable ")]
            items += [m for m in f.get("must", []) if str(m).startswith(WARDROBE_MUST)]
        else:
            items += list(f.get("must", []))
            items += [m for x in j.get("states", []) for m in x.get("must", [])]
        # a disciple at the Ascension wore a wristwatch (10-17)
        items.append(PERIOD_ITEM)

        q = ("Check this illustration against a list. For each numbered item, look carefully and answer "
             "whether the picture shows it. An item about what a still picture cannot show (heat, sound, "
             "smell, a feeling, the direction someone travels, what happened before) counts as true.\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) +
             '\nAnswer with a final line of JSON: {"items": [true/false, ...], "people": <count>, '
             '"notes": "what is missing or wrong"}')
        url = video.checker_url(llm_url)
        try:
            reply = video._look_at(url, model, png, q)
            found = re.findall(r"\{[^{}]*\"items\"[^{}]*\}", reply or "")
            data = json.loads(found[-1]) if found else {}
            # the drawing mistakes as their own question, reasoned out arm by arm (folded into the list and
            # answered briefly, it would wave extra heads and limbs through; user 10-18: don't rush checks)
            probs = video.check_picture(png, len(who) if who and not crowd else None, url, model)
        except Exception as exc:                 # no vision server: draw unchecked
            log(f"  keyframe check skipped ({exc})")
            return set()
        missing = [items[i] for i, ok in enumerate(data.get("items", [])) if ok is False and i < len(items)]
        # a second, describe-then-judge look at each failed item: in one list the model failed every
        # dragon frame for "2 people" and still frames for "is walking", "is trembling" (10-18)
        missing = [m for m in missing if not _item_holds(png, m, url, model, still=True)]
        try:
            missing += _compare(j, png, url, model)
        except Exception as exc:
            log(f"    comparison skipped for {j['id']} ({exc})")
        if missing or probs:
            state["log"].append({"id": j["id"], "seed": j["seed"], "missing": missing, "problems": probs,
                                 "notes": str(data.get("notes", ""))[:300]})
            if final:
                log(f"    {j['id']} kept after {KEYFRAME_ROUNDS} redraws: {'; '.join(missing + probs)[:200]}")
                if PERIOD_ITEM in missing:
                    state.setdefault("edits", {})[j["id"]] = str(data.get("notes", ""))[:300]
            else:
                bad.add(j["id"])
                _save_checks(state)
                continue
        state["passed"][j["id"]] = png.stat().st_mtime
        _save_checks(state)
    if not quiet:
        log(f"  checked keyframes: {len(bad)} to draw again")
    return bad


# --- checking the rendered clips -------------------------------------------------------
# The keyframes are checked before the render, but the video model can still drift away from a good
# one in 5 seconds: a second knight walks in, armor turns back into cloth, a regrown wristwatch (10-17),
# a face melts. These were only caught by a person looking at contact sheets (user, 10-18: "bookcaster
# should do all of those checks on its own"). Each clip's first, middle and last frames, side by side,
# go to the vision model against the shot's plan; a failing clip is rendered again with another seed
# (CLIP_ROUNDS times), then kept and logged.
CLIP_ROUNDS = int(os.environ.get("BOOKCASTER_CLIP_ROUNDS", "1"))


def _clip_state(work: Path) -> Path:
    return work / "anim" / "clip_checks.json"


def _clip_strip(mp4: Path, out: Path) -> Path:
    """First, middle and last frame of a clip side by side."""
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                            "stream=nb_read_packets", "-of", "csv=p=0", str(mp4)],
                           capture_output=True, text=True).stdout.strip() or 0)
    picks = [0, max(0, n // 2), max(0, n - 1)]
    sel = "+".join(f"eq(n\\,{k})" for k in picks)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf",
                    f"select='{sel}',tile=3x1", "-frames:v", "1", "-vsync", "0", str(out)], check=True)
    return out


def _item_holds(png: Path, item: str, llm_url: str, model: str, still: bool = False) -> bool:
    """Ask again about one failed item, describing first. *still*: one picture, so motion words
    ("walking", "trembling") are judged by pose, and a creature counts as a figure."""
    q = ("Look carefully at this illustration. First describe every figure in it (people and creatures such "
         "as a dragon or an animal; limbs of someone being carried belong to that person), what each wears "
         "and holds, and their pose. "
         + ("It is a single still frame from an animation: judge any movement by the pose (mid-stride counts "
            "as walking). " if still else "") +
         f"Then judge this statement about the picture: \"{item}\". Answer with a final line of JSON: "
         '{"holds": true/false, "why": "..."}')
    try:
        r = video._look_at(llm_url, model, png, q)
    except Exception:
        return True
    m = re.findall(r"\{[^{}]*\"holds\"[^{}]*\}", r or "")
    try:
        return not (m and json.loads(m[-1]).get("holds") is False)
    except ValueError:
        return True


def _clip_item_holds(strip: Path, item: str, llm_url: str, model: str) -> bool:
    """A second, slower look at one failed item: describe each person first, then judge. Asked in one
    list, the model counted four people in a lift where Peter and John carry the healed man - three,
    limbs tangled (10-18): only a confirmed failure re-renders a clip."""
    q = ("These are the first, middle and last frames of one animated shot, side by side. First describe, "
         "frame by frame, every distinct person you see (clothes, hair, what they do); limbs of someone "
         "being carried or held belong to that person. Then judge this statement about the shot: "
         f"\"{item}\". Answer with a final line of JSON: {{\"holds\": true/false, \"why\": \"...\"}}")
    try:
        r = video._look_at(llm_url, model, strip, q)
    except Exception:
        return True
    m = re.findall(r"\{[^{}]*\"holds\"[^{}]*\}", r or "")
    try:
        return not (m and json.loads(m[-1]).get("holds") is False)
    except ValueError:
        return True


def check_clips(shots: list[dict], kf: dict | None, plan: dict, work: Path, llm_url: str | None, model: str,
                log, final: bool = False) -> list[str]:
    """Check each rendered clip not yet passed (by file time); delete the failing ones (and the clips
    carried on from them) and give them a new seed. Returns the failed shot ids (none on the final
    round: those are kept and logged)."""
    if not llm_url:
        return []
    chars, _ = video.looks(plan)
    frames = {f["shot"]: f for f in (kf or {}).get("frames", [])}
    state_f = _clip_state(work)
    state = json.loads(state_f.read_text()) if state_f.exists() else {"seeds": {}, "passed": {}, "log": []}
    order = sorted(shots, key=lambda s: s["t"])
    bad, t0, n = [], time.time(), 0
    # the shot plan each clip is judged on (a "continue" shot: its key frame's people)
    todo, last_key = [], None
    for s in order:
        f = frames.get(s["id"]) or {}
        if f.get("mode") == "key":
            last_key = f
        out = Path(s.get("out") or work / "anim" / f"{s['id']}.mp4")
        if s["kind"] not in ("action", "talk") or not out.exists():
            continue
        if state["passed"].get(s["id"]) == out.stat().st_mtime:
            continue
        todo.append((s, f, last_key, out))
    lock = threading.Lock()

    def judge(arg):
        """One clip: its first, middle and last frames against its plan (run several at once: one at a
        time, 95 clips took ~50 min, 10-18)."""
        s, f, last_key, out = arg
        if s["kind"] == "talk":
            everyone = [w for w in [s.get("speaker"), s.get("listener")] if w]
            crowd = ""
        else:
            src = f if f.get("who") is not None and f.get("mode") != "continue" else (last_key or f)
            everyone = list(src.get("who") or [])
            crowd = str(src.get("crowd") or "").strip()
        # drift only: what the video model changed from the clip's own first frame (its keyframe, already
        # checked). Judged against the plan, 9 of 19 failed clips were Maren's dress colour as the keyframes
        # drew it, and a re-render from the same keyframe can't change that (10-18).
        n_fig = len(everyone)
        items = ["every person and creature in the first frame is still there in the middle and last frames, "
                 "and nobody new has appeared (people walking in or out of the edge of the picture is fine)",
                 "nobody appears twice in any frame, and no person turns into someone else",
                 "everyone wears the same clothes and armor in all three frames as in the first",
                 "no face, body, limb or object melts, splits, stretches or merges into another",
                 PERIOD_ITEM]
        if not crowd and n_fig:
            items.insert(0, f"the first frame shows {n_fig} {'figure' if n_fig == 1 else 'figures'} with a head "
                            f"(people or creatures), and so do the middle and last frames")
        strip = _clip_strip(out, work / "anim" / f".strip-{s['id']}.jpg")
        q = ("These are the first, middle and last frames of one 5-second animated shot, side by side. The "
             "first frame is the drawn keyframe the shot starts from; judge only what CHANGED after it. "
             "Check them against a list; for each numbered item answer whether it holds.\n"
             + "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) +
             '\nAnswer with a final line of JSON: {"items": [true/false, ...], "notes": "what is wrong"}')
        url = video.checker_url(llm_url)
        try:
            reply = video._look_at(url, model, strip, q)
            found = re.findall(r"\{[^{}]*\"items\"[^{}]*\}", reply or "")
            try:
                data = json.loads(found[-1]) if found else {}
            except ValueError:
                data = {}
            missing = [items[i] for i, ok in enumerate(data.get("items", [])) if ok is False and i < len(items)]
            if missing:
                missing = [m for m in missing if not _clip_item_holds(strip, m, url, model)]
        except Exception:
            return s, out, None, ""
        finally:
            strip.unlink(missing_ok=True)
        return s, out, missing, str(data.get("notes", ""))[:300]

    with ThreadPoolExecutor(video.checker_workers()) as ex:
        for s, out, missing, notes in ex.map(judge, todo):
            if missing is None:
                continue                         # no answer from the checker: left for the next look
            n += 1
            if missing and not final:
                state["log"].append({"id": s["id"], "seed": s.get("seed"), "missing": missing, "notes": notes})
                bad.append(s["id"])
                continue
            if missing:
                log(f"    {s['id']} kept after {CLIP_ROUNDS} re-render(s): {'; '.join(missing)[:200]}")
            state["passed"][s["id"]] = out.stat().st_mtime
    # the failed clips go, with every clip carried on from them, and render again with another seed
    gone = set(bad)
    for a, b in zip(order, order[1:]):          # a "continue" shot starts from the clip before it
        if b.get("mode") == "continue" and a["kind"] == "action" and a["id"] in gone:
            gone.add(b["id"])
    for sid in gone:
        state["seeds"][sid] = int(next((x.get("seed", 1) for x in order if x["id"] == sid), 1)) + 7919
        for x in (work / "anim").glob(f"{sid}*"):
            if x.suffix in (".mp4", ".png", ".wav") or x.name.endswith(".latents.pt"):
                x.unlink()
    state_f.write_text(json.dumps(state, indent=1))
    if n:
        log(f"  checked {n} clip(s) in {(time.time() - t0) / 60:.0f} min: {len(gone)} to render again")
    return sorted(gone)


def apply_keyframes(shots: list[dict], kf: dict, images: dict[str, str], plan: dict, style: str,
                    work: Path) -> None:
    """Point each shot at its keyframe: prompt = the frame's motion, refs = the
    people in it and the keyframe (the scene); stills get a picture to push in on;
    talking shots open from the latest keyframe showing the speaker."""
    frames = {f["shot"]: f for f in kf["frames"]}
    chars, _ = video.looks(plan)
    known = video.ref_prompts(plan, style)
    refs_dir, out_dir = work.parent / "refs", work / "anim" / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    scene, people, showing, prompts = None, [], {}, {}
    for s in sorted(shots, key=lambda s: s["t"]):
        f = frames.get(s["id"])
        if s["kind"] == "talk":
            if s["speaker"] in showing:
                s["scene_image"] = showing[s["speaker"]]
                s["scene_prompt"] = prompts.get(s["scene_image"], "")
            continue
        if not f:
            continue
        s["mode"] = f.get("mode", "key")
        if s["id"] in images:
            scene = images[s["id"]]
            people = [w for w in f.get("who", []) if w in chars and w in known][:3]
            showing.update({w: scene for w in f.get("who", [])})
            prompts[scene] = f.get("prompt", "")
        if s["mode"] == "still":
            s["still_image"] = images.get(s["id"]) or scene
            continue
        st = states_for(kf, s["id"], people or f.get("who", []))
        extra = " ".join(x["motion"].rstrip(". ") + "." for x in st if x.get("motion"))
        s["prompt"] = f"{f.get('motion', '').rstrip('. ')}. {extra} {MOTION} {style}.".replace("  ", " ")
        s["refs"] = [str(_figure(video.ref_path(refs_dir, w, known[w][0]), out_dir)) for w in people] + [scene]


def _still_clip(img: str, dur: float, out: str) -> None:
    """A keyframe held for *dur* with a slow push-in, at the video models' size."""
    n = max(1, round(dur * video.FPS))
    vf = (f"scale=1664:960:force_original_aspect_ratio=increase,crop=1664:960,"
          f"zoompan=z='1+0.05*on/{n}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={n}:s=832x480:fps={video.FPS}")
    tmp = Path(out).with_name("." + Path(out).name)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-i", img, "-vf", vf, "-frames:v", str(n),
                    "-c:v", *video.VIDEO_CODEC, "-b:v", "4M", "-pix_fmt", "yuv420p", "-f", "mp4", str(tmp)],
                   check=True)
    tmp.replace(out)


# --- rendering ------------------------------------------------------------------------

def _gpu(bus: str) -> str:
    return subprocess.run([str(HIP_INDEX), bus], capture_output=True, text=True, check=True).stdout.strip()


def _servers(action: str, log) -> bool:
    """Pause ("down") or resume ("up") the voice and music servers, which hold
    both cards. Returns whether they were running (so only those come back)."""
    if action == "down":
        running = any(subprocess.run(["systemctl", "--user", "is-active", "-q", u]).returncode == 0
                      for u in ("tts-fish", "tts-ace"))
        if running:
            log("  pausing the voice and music servers (the video models need both cards)")
            subprocess.run([str(STACK), "down", "--music"], capture_output=True)
        return running
    log("  restarting the voice and music servers")
    subprocess.run([str(STACK), "up", "--music"], capture_output=True)
    return True


def _scope(mem_gb: int, env: dict, cmd: list[str], log_file: Path) -> subprocess.Popen:
    """A runner in its own memory-capped scope (an OOM kills it, not the box)."""
    with open(log_file, "a") as fh:
        return subprocess.Popen(["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={mem_gb}G",
                                 "-p", "MemorySwapMax=0", *[f"--setenv={k}={v}" for k, v in env.items()],
                                 *cmd], stdout=fh, stderr=subprocess.STDOUT, cwd=SKYREELS)


def render(shots: list[dict], pics: list[dict], plan: dict, style: str, work: Path, audio: Path,
           log=print) -> list[dict]:
    """Render what isn't rendered yet; returns the manifest entries of the
    shots that exist."""
    anim = work / "anim"
    anim.mkdir(parents=True, exist_ok=True)
    by_id = {p["id"]: p for p in pics}
    for s in shots:
        s["out"] = str(anim / f"{s['id']}.mp4")
    # what a "continue" shot carries on from: the shot just before it IN THE STORY (after a restart,
    # the one before it in a card's queue was an unrelated shot: a dead man became a cartoon manor, 10-17)
    order = sorted(shots, key=lambda s: s["t"])
    for a, b in zip(order, order[1:]):
        if b.get("mode") == "continue" and a["kind"] == "action":
            b["chain_from"] = a["out"]
    todo = [s for s in shots if not Path(s["out"]).exists()]
    if todo:
        if any(s["kind"] == "talk" and not s.get("image") for s in todo):
            _portraits(todo, by_id, plan, style, work, log)
        resume = _servers("down", log)
        if ACTION_ENGINE != "ltx" or TALK_ENGINE != "ltx":
            igpu("vae", log)                    # talking shots' VAE and action shots' decode (Wan / InfiniteTalk)
        started = True
        for s in [s for s in todo if s.get("mode") == "still" and s.get("still_image")]:
            _still_clip(s["still_image"], s["dur"], s["out"])
        try:
            actions = [s for s in todo if s["kind"] == "action" and s.get("mode") != "still"]
            talks = [s for s in todo if s["kind"] == "talk"]
            # bus 05 renders the talking shots first, so it takes fewer action shots
            talk_min = sum(s["dur"] for s in talks) * TALK_MIN_PER_S
            n05 = min(len(actions), max(0, round((len(actions) * ACTION_MIN_PER_CLIP - talk_min)
                                                 / (2 * ACTION_MIN_PER_CLIP))))
            est = (max(talk_min + n05 * ACTION_MIN_PER_CLIP, (len(actions) - n05) * ACTION_MIN_PER_CLIP)
                   if ACTION_ENGINE not in ("i2v", "i2v22") else
                   max((talk_min + len(actions) * ACTION_MIN_PER_CLIP) / 2,
                       max((s["dur"] * TALK_MIN_PER_S for s in talks), default=0)))
            log(f"  rendering {len(talks)} talking shot(s) and {len(actions)} action shot(s) on both cards "
                f"(~{est:.0f} min)")
            t = time.time()
            if ACTION_ENGINE == "ltx":
                # LTX (tools/ltx_flow.py): the shots of the engines set to "ltx"; talking shots of another
                # engine after it, as before
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
                import ltx_flow
                mine = actions + (talks if TALK_ENGINE == "ltx" else [])
                ltx_flow.render_ltx(mine, pics, plan, style, work, audio, log)
                if talks and TALK_ENGINE != "ltx":
                    talk_jobs = _talk_jobs(talks, audio, work)
                    second = threading.Thread(target=_render_talks, args=(talk_jobs, work, "65", log))
                    second.start()
                    _render_talks(talk_jobs, work, "05", log)
                    second.join()
                p65 = p05 = None
                if talks:
                    log(f"    {sum(Path(s['out']).exists() for s in talks)}/{len(talks)} talking shot(s) ok")
            elif ACTION_ENGINE == "i2v22":
                # Wan2.2's two experts can't share a card: high-noise on 05, low-noise on 65, every action
                # clip passing through both (run_i2v22.py); then the talking shots on both cards
                for c in list(anim.glob(".*.claim")) + list(anim.glob("*.mid.pt")):
                    c.unlink()
                procs = _action_pipeline(actions, by_id, plan, style, work)
                for p in procs:
                    p.wait()
                if TALK_ENGINE == "ltx":      # LTX talks need both cards: after the action pipeline
                    _ltx_talks(talks, pics, plan, style, work, audio, log)
                talk_jobs = _talk_jobs(talks, audio, work) if TALK_ENGINE != "ltx" else []
                second = threading.Thread(target=_render_talks, args=(talk_jobs, work, "65", log))
                second.start()
                _render_talks(talk_jobs, work, "05", log)
                second.join()
                p65 = p05 = None
                if talks:
                    log(f"    {sum(Path(s['out']).exists() for s in talks)}/{len(talks)} talking shot(s) ok")
            elif ACTION_ENGINE == "i2v":
                # both cards get every action shot, in runs (a key shot + the continues after it): bus 65
                # takes the runs from the end, bus 05 (after the talking shots) from the start; each claims a
                # shot as it starts it (run_i2v.py), so they meet wherever they meet - a fixed split left one
                # card idle while the other had a backlog (10-17)
                for c in anim.glob(".*.claim"):
                    c.unlink()                 # a crashed run's claims would block their shots
                runs = []
                for a in actions:
                    if a.get("mode") == "continue" and runs:
                        runs[-1].append(a)
                    else:
                        runs.append([a])
                # ... and the talking shots too: bus 05 starts on them, bus 65 joins once the action shots
                # are gone. Talking shots on bus 05 alone left bus 65 idle 95 of a 151 min render (10-18).
                talk_jobs = _talk_jobs(talks, audio, work) if TALK_ENGINE != "ltx" else []

                def lane65():
                    p = _action_batch("65", [a for r in reversed(runs) for a in r], by_id, plan, style, work)
                    if p:
                        p.wait()
                    _render_talks(talk_jobs, work, "65", log)
                second = threading.Thread(target=lane65)
                second.start()
                _render_talks(talk_jobs, work, "05", log)
                p05 = _action_batch("05", actions, by_id, plan, style, work)
                if p05:
                    p05.wait()
                second.join()
                if TALK_ENGINE == "ltx":      # LTX talks need both cards: after the action batches
                    _ltx_talks(talks, pics, plan, style, work, audio, log)
                p65 = None
                ok = sum(Path(s["out"]).exists() for s in talks)
                if talks:
                    log(f"    {ok}/{len(talks)} talking shot(s) ok")
            else:
                n05 = _scene_cut(actions, n05)
                p65 = _action_batch("65", actions[n05:], by_id, plan, style, work)
                if TALK_ENGINE != "ltx":
                    _render_talks_a2v(_talk_jobs(talks, audio, work), work, log)
                p05 = _action_batch("05", actions[:n05], by_id, plan, style, work)
            for p in (p65, p05):
                if p:
                    p.wait()
            if TALK_ENGINE == "ltx" and ACTION_ENGINE not in ("ltx", "i2v22", "i2v"):
                _ltx_talks(talks, pics, plan, style, work, audio, log)
            log(f"  shots done in {(time.time() - t) / 60:.0f} min")
        finally:
            if started:
                _vae_server("down", log)
            if resume:
                _servers("up", log)
    done = [{k: s[k] for k in ("id", "kind", "t", "dur", "pic", "out")} for s in shots
            if Path(s["out"]).exists()]
    (anim / "manifest.json").write_text(json.dumps(done, indent=1))
    # the newest picture of each moment, for stretches without an animated shot (video.build frames=)
    stills, last = [], None
    for s in sorted(shots, key=lambda s: s["t"]):
        img = s.get("image") if s["kind"] == "talk" else (s.get("refs") or [None])[-1] or s.get("still_image")
        img = img or last
        if img:
            stills.append({"t": s["t"], "pic": s["pic"], "image": str(img)})
            last = img if s["kind"] == "action" else last
    (anim / "stills.json").write_text(json.dumps(stills, indent=1))
    missing = len(shots) - len(done)
    log(f"  {len(done)} animated shot(s) ready" + (f", {missing} failed (stills stay)" if missing else ""))
    return done


def _ltx_talks(talks: list[dict], pics: list[dict], plan: dict, style: str, work: Path, audio: Path, log) -> None:
    """Talking shots through LTX A2Vid (tools/ltx_flow.py) when TALK_ENGINE is "ltx" but the action engine is not."""
    if not talks:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    import ltx_flow
    ltx_flow.render_ltx(talks, pics, plan, style, work, audio, log)


def _scene_cut(actions: list[dict], n: int) -> int:
    """Move the split between the cards' batches to the nearest change of
    picture, so a chained run of one scene's clips stays on one card."""
    cuts = [i for i in range(1, len(actions)) if actions[i].get("mode") == "key"
            or ("mode" not in actions[i] and actions[i]["pic"] != actions[i - 1]["pic"])]
    near = [i for i in cuts if abs(i - n) <= 3]
    return min(near, key=lambda i: abs(i - n)) if near else n


def _action_batch(bus: str, todo: list[dict], by_id: dict, plan: dict, style: str,
                  work: Path) -> subprocess.Popen | None:
    """Start one card's batch of action shots (run_split.py loads the model once).
    A clip that carries on the clip before it (same picture, back to back)
    takes that clip's last frame as its scene, so people keep their places
    across the cut (Holmes and Watson swapped sides, 10-16). Decoded on the
    iGPU VAE server at 832 wide (its tuned shapes), not ~7 min of CPU."""
    if not todo:
        return None
    jobs = []
    for k, s in enumerate(todo):
        prev = todo[k - 1] if k else None
        chain = s.get("chain_from") or (
            prev["out"] if "mode" not in s and prev and prev["pic"] == s["pic"]
            and abs(prev["t"] + prev["dur"] - s["t"]) < 0.5 else None)
        jobs.append({"refs": s.get("refs") or _action_refs(s, by_id[s["pic"]], plan, style, work),
                     "prompt": s["prompt"], "out": s["out"], "seed": s["seed"], "duration": ACTION_S,
                     **({"scene_from": chain} if chain else {})})
    env = {"HIP_VISIBLE_DEVICES": _gpu(bus), "TORCHINDUCTOR_CACHE_DIR": str(INDUCTOR_CACHE)}
    if ACTION_ENGINE == "i2v":
        # the clip starts from its keyframe (the last ref: the scene) or, chained, from the clip before
        jobs = [{"image": j["refs"][-1], "prompt": j["prompt"], "out": j["out"], "seed": j["seed"],
                 **({"scene_from": j["scene_from"]} if j.get("scene_from") else {})} for j in jobs]
        jf = work / "anim" / f"i2v-jobs-{bus}.json"
        jf.write_text(json.dumps(jobs, indent=1))
        return _scope(40, env, [str(SKYREELS_PY), "-I", "-X", "faulthandler", "-u", "run_i2v.py",
                                "--jobs", str(jf), "--vae-server", str(VAE_SOCK), *_sparse_args(ACTION_SPARSE)],
                      work / "anim" / f"i2v-{bus}.log")
    jf = work / "anim" / f"r2v-jobs-{bus}.json"
    jf.write_text(json.dumps(jobs, indent=1))
    return _scope(20, env, [str(SKYREELS_PY), "-I", "-X", "faulthandler", "-u", "run_split.py",
                            "--fp8", "--jobs", str(jf), "--vae-server", str(VAE_SOCK), "--width", "832"],
                  work / "anim" / f"r2v-{bus}.log")


def _action_pipeline(todo: list[dict], by_id: dict, plan: dict, style: str, work: Path) -> list:
    """Wan2.2 (run_i2v22.py): the high-noise expert on bus 05 and the low-noise expert on bus 65 over the
    same job list, each clip from its keyframe and, where one was drawn, to its end frame."""
    if not todo:
        return []
    jobs = []
    for k, s in enumerate(todo):
        prev = todo[k - 1] if k else None
        chain = s.get("chain_from") or (
            prev["out"] if "mode" not in s and prev and prev["pic"] == s["pic"]
            and abs(prev["t"] + prev["dur"] - s["t"]) < 0.5 else None)
        start = (s.get("refs") or _action_refs(s, by_id[s["pic"]], plan, style, work))[-1]
        end = end_path(start)
        jobs.append({"image": start, "prompt": s["prompt"], "out": s["out"], "seed": s["seed"],
                     **({"scene_from": chain} if chain else {}),
                     **({"end_image": str(end)} if end.exists() and not chain else {})})
    jf = work / "anim" / "i2v22-jobs.json"
    jf.write_text(json.dumps(jobs, indent=1))
    procs = []
    for bus, expert in (("05", "high"), ("65", "low")):
        env = {"HIP_VISIBLE_DEVICES": _gpu(bus), "TORCHINDUCTOR_CACHE_DIR": str(INDUCTOR_CACHE)}
        procs.append(_scope(40, env, [str(SKYREELS_PY), "-I", "-X", "faulthandler", "-u", "run_i2v22.py",
                                      "--expert", expert, "--jobs", str(jf), "--vae-server", str(VAE_SOCK),
                                      *_sparse_args(ACTION_SPARSE)], work / "anim" / f"i2v22-{expert}.log"))
    return procs


STRATA_UP = Path(os.environ.get("BOOKCASTER_STRATA_UP", Path.home() / "strata-eval" / "strata-best-up.sh"))
STRATA_UNIT = "strata-eval"


def igpu(tenant: str, log=print) -> None:
    """Give the iGPU to one tenant: "strata" (the language model: planning, and seeing pictures for the
    keyframe checks; ~90 GB of the shared RAM) or "vae" (the VAE server the renders decode on). They
    can't share the memory, so a job does all its language-model work first and switches once (10-17)."""
    def active(u):
        return subprocess.run(["systemctl", "--user", "is-active", "-q", u]).returncode == 0

    if tenant == "strata":
        if active(STRATA_UNIT):
            return
        ch = sys.modules.get("characters")      # the loaded vision judge (10-24) can't share the iGPU with it
        if ch is not None and hasattr(ch, "vl_close"):
            ch.vl_close()
        _vae_server("down", log)
        _flush()
        log("  starting the language model on the iGPU")
        subprocess.run([str(STRATA_UP)], capture_output=True)
        from urllib.request import urlopen
        for _ in range(120):
            try:
                urlopen("http://127.0.0.1:18100/v1/models", timeout=5)
                return
            except OSError:
                time.sleep(5)
        raise RuntimeError("the language model didn't come up (journalctl --user -u strata-eval)")
    if active(STRATA_UNIT):
        log("  stopping the language model (the renders need the iGPU's memory)")
        subprocess.run(["systemctl", "--user", "stop", STRATA_UNIT], capture_output=True)
        time.sleep(5)
    _flush()
    _vae_server("up", log)


def _flush() -> None:
    """The iGPU's memory pool keeps an unloaded model's pages until flushed (10-14)."""
    if os.path.exists(video.FLUSH):
        subprocess.run(["sudo", "-n", video.FLUSH], capture_output=True)


def _vae_server(action: str, log) -> bool:
    """Start ("up") or stop ("down") the iGPU VAE server the talking model
    uses. Its HSA_CU_MASK keeps one of the 40 CUs free: compute holding all
    of them starved GNOME's ring into a reset (10-16). Returns whether this
    call started it."""
    if action == "down":
        # not while another runner still decodes on it (10-17: a job's end cut off a helper batch)
        busy = subprocess.run(["pgrep", "-f", "run_(i2v|a2v|split|v2v).py"], capture_output=True, text=True)
        if busy.stdout.split():
            log("  leaving the VAE server up: another render is still using it")
            return False
        subprocess.run(["systemctl", "--user", "stop", VAE_UNIT], capture_output=True)
        return False
    if subprocess.run(["systemctl", "--user", "is-active", "-q", VAE_UNIT]).returncode == 0:
        return False
    subprocess.run(["systemctl", "--user", "reset-failed", VAE_UNIT], capture_output=True)
    VAE_SOCK.unlink(missing_ok=True)
    ig = _gpu("igpu")
    subprocess.run(["systemd-run", "--user", f"--unit={VAE_UNIT}", "-p", "MemoryMax=24G",
                    "-p", f"WorkingDirectory={SKYREELS}", f"--setenv=HIP_VISIBLE_DEVICES={ig}",
                    f"--setenv=HSA_CU_MASK={ig}:0-38", "--setenv=PYTHONUNBUFFERED=1",
                    "--setenv=MIOPEN_FIND_MODE=FAST", str(SKYREELS / ".venv-igpu" / "bin" / "python"),
                    "-I", "-X", "faulthandler", "vae_server.py", "--sock", str(VAE_SOCK)],
                   check=True, capture_output=True)
    for _ in range(120):
        if VAE_SOCK.exists():
            log("  VAE server up on the iGPU")
            return True
        time.sleep(1)
    raise RuntimeError(f"{VAE_UNIT} did not come up (journalctl --user -u {VAE_UNIT})")


def _talk_jobs(todo: list[dict], audio: Path, work: Path) -> list[dict]:
    """The talking shots' runner jobs, longest first (the card that joins late
    then takes the short ones, so the two finish close together)."""
    jobs = []
    for s in sorted(todo, key=lambda s: -s["dur"]):
        wav = work / "anim" / f"{s['id']}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s['t']:.2f}", "-t", f"{s['dur']:.2f}",
                        "-i", str(audio), "-ac", "1", "-ar", "16000", str(wav)], check=True)
        jobs.append({"image": s["image"], "audio": str(wav), "prompt": s["prompt"], "out": s["out"],
                     "seed": int(s.get("seed", 1)), **({"bbox": s["bbox"]} if s.get("bbox") else {})})
    return jobs


def _render_talks(jobs: list[dict], work: Path, bus: str, log) -> None:
    """The talking shots not rendered yet, on one card (blocks). Both cards run
    the same lists: run_infinitetalk.py claims a shot as it starts it.
    InfiniteTalk on the 4-step I2V base (10-17): no NaN in any test, ~53 s of
    render per second of speech; one speaker = "single" weights, a two-shot =
    "multi" (listener silent). No expandable_segments: with it the XTX's first
    VAE passes came out NaN."""
    if TALK_ENGINE != "infinitetalk":         # run_a2v.py has no claims: bus 05 renders them all
        if bus == "05":
            _render_talks_a2v(jobs, work, log)
        return
    env = {"HIP_VISIBLE_DEVICES": _gpu(bus), "MIOPEN_FIND_MODE": "FAST", "OMP_NUM_THREADS": "8",
           "IT_AUDIO_CFG": "1.0"}
    for variant, group in (("single", [j for j in jobs if not j.get("bbox")]),
                           ("multi", [j for j in jobs if j.get("bbox")])):
        if not [j for j in group if not Path(j["out"]).exists()]:
            continue
        gf = work / "anim" / f"it-jobs-{variant}-{bus}.json"
        gf.write_text(json.dumps(group, indent=1))
        _scope(48, {**env, "IT_VARIANT": variant},
               [str(SKYREELS_PY), "-I", "-X", "faulthandler", "-u", "run_infinitetalk.py", "--jobs", str(gf),
                "--vae-server", str(VAE_SOCK), *_sparse_args(TALK_SPARSE)], work / "anim" / f"it-{bus}.log").wait()


def _render_talks_a2v(jobs: list[dict], work: Path, log) -> None:
    """All talking shots in one run_a2v.py batch on bus 05 (blocks)."""
    if not jobs:
        return
    jf = work / "anim" / "a2v-jobs.json"
    t = time.time()
    # both cards' batches share the iGPU VAE server: a long timeout, not the runner's 240 s
    env = {"HIP_VISIBLE_DEVICES": _gpu("05"), "MIOPEN_FIND_MODE": "FAST",
           "OMP_NUM_THREADS": "8", "A2V_CALL_TIMEOUT": "900"}
    # a pass that goes inf/NaN does so for that input every time: the retry takes another seed AND
    # the fp32 encode (the adder speech's 4th window failed 10 times with the fp16 one, 10-16)
    for seed, fp16 in ((1, True), (2, False)):
        left = [j for j in jobs if not Path(j["out"]).exists()]
        if not left:
            break
        jf.write_text(json.dumps([dict(j, seed=seed) for j in left], indent=1))
        _scope(48, env, [str(SKYREELS_PY), "-I", "-X", "faulthandler", "-u", "run_a2v.py",
                         "--jobs", str(jf), "--vae-server", str(VAE_SOCK)]
               + (["--vae-encode-fp16"] if fp16 else []), work / "anim" / "a2v.log").wait()
    ok = sum(Path(j["out"]).exists() for j in jobs)
    log(f"    {ok}/{len(jobs)} talking shot(s) ok in {(time.time() - t) / 60:.0f} min")


def animate_chapter(level: str, text: str, audio: Path, timeline_path: Path, work: Path,
                    style: str, log=print, span: tuple[float, float] | None = None,
                    narrator: str | None = None, llm_url: str | None = None,
                    llm_model: str = "default", phase: str = "all") -> list[dict]:
    """Plan and render a chapter's animated shots (the pictures must be drawn);
    span = (t0, t1) seconds renders only the shots starting inside it.
    narrator = the character who narrates in the first person (the cast's
    same_as "narrator"): quotes labelled Narrator are that character speaking
    (Watson's "What can it mean?"), so they get talking shots and count as
    nearby lines when choosing who is spoken to. Narration stays narration.
    phase: "plan" = keyframes planned, drawn and checked (the language model holds the iGPU), no render;
    "render" = render only (the VAE server holds the iGPU); "all" = both. A job over several chapters
    plans them all, then renders them all: one switch of the iGPU instead of two per chapter (10-17)."""
    timeline = json.loads(timeline_path.read_text())
    if narrator:
        timeline = [dict(e, speaker=narrator) if e.get("kind") == "quote" and
                    str(e.get("speaker", "")).lower() == "narrator" else e for e in timeline]
    total = video.audio_seconds(audio)
    plan = json.loads((work / "plan.json").read_text())
    pics = video.timed(plan, text, timeline, total)
    shots = plan_shots(level, pics, timeline, total, plan, style)
    if span:
        shots = [s for s in shots if span[0] <= s["t"] < span[1]]
    # the book's look registry (look.json) over the chapter plan: a corrected look
    # (Roylott clean-shaven, 10-16) and the keyframes' props, first stored wins
    # ... plus the book's looks for anyone this chapter names or hears who the picture plan left out:
    # a flashback to Judas, the Angels who speak (10-17); then everyone gets their registry look
    reg_f = work.parent / "look.json"
    if reg_f.exists():
        heard = {str(e.get("speaker", "")) for e in timeline if e.get("kind") == "quote"}
        mine = plan["look"].setdefault("characters", {})
        for name, desc in (json.loads(reg_f.read_text()).get("characters") or {}).items():
            if name not in mine and (name in heard or re.search(rf"\b{re.escape(name)}\b", text)):
                mine[name] = desc
    video.keep_looks(plan, reg_f)
    starts = video.paragraph_times(video.paragraphs(text), timeline)
    if "wardrobe" not in plan and llm_url and phase != "render":
        igpu("strata", log)
        video.ensure_wardrobe(plan, text, llm_url, llm_model, log)
        saved = json.loads((work / "plan.json").read_text())
        saved["wardrobe"] = plan["wardrobe"]
        (work / "plan.json").write_text(json.dumps(saved, indent=1, ensure_ascii=False))
    reseed = json.loads(_clip_state(work).read_text()).get("seeds", {}) if _clip_state(work).exists() else {}
    ask_listeners(shots, timeline, total, video.looks(plan)[0], work,
                  llm_url if phase != "render" else None, llm_model, log)
    for s in shots:
        s["para"] = video.para_at(starts, s["t"])
        if s["id"] in reseed:
            s["seed"] = reseed[s["id"]]
    kf = keyframes(work)
    if phase == "render":
        llm_url = None                       # planned and checked already
    if kf is None and llm_url and ACTION_ENGINE in ("i2v", "i2v22", "ltx"):
        igpu("strata", log)
        log("  planning keyframes...")
        kf = plan_keyframes(shots, pics, timeline, total, plan, style, llm_url, llm_model, log, starts)
        (work / "anim").mkdir(parents=True, exist_ok=True)
        (work / "anim" / "keyframes.json").write_text(json.dumps(kf, indent=1, ensure_ascii=False))
    if kf:
        _drop_people_props(kf, plan)
        _add_creatures(kf.get("frames", []), video.looks(plan)[0])
        _cap_who(kf.get("frames", []))
        apply_wardrobe(kf, plan, starts)
    plan["look"]["props"] = {**(kf or {}).get("props", {}), **plan["look"].get("props", {})}
    video.keep_looks(plan, work.parent / "look.json")
    if phase == "plan-only":     # bookcaster keyframes --sample: the keyframes as a real run would draw them
        return {"kf": kf, "plan": plan, "shots": shots, "pics": pics, "timeline": timeline, "total": total,
                "starts": starts, "style": style}
    if kf and phase.startswith("check"):
        # everything is drawn and checked: only point the shots at their pictures
        images = draw_keyframes(kf, plan, style, work, log, llm_url=None, model=llm_model)
        apply_keyframes(shots, kf, images, plan, style, work)
        _portraits(shots, {p["id"]: p for p in pics}, plan, style, work, log, kf, None, llm_model)
    elif kf:
        if llm_url:
            igpu("strata", log)            # the keyframe checks look at the pictures
        # one drawing model load for the keyframes, their redraws and the portraits (10-18)
        with video.checker_session(log) if llm_url else contextlib.nullcontext(), \
                video.drawing_session(work / "anim" / "draw.log", log):
            # the wardrobe's outfit sheets first: keyframes and portraits crop them as references
            missing = [j for j in video.reference_jobs(plan, style, work.parent / "refs")
                       + video.outfit_jobs(plan, style, work.parent / "refs") if not Path(j["path"]).exists()]
            if missing:
                log(f"  drawing {len(missing)} reference sheet(s) (outfits from the wardrobe)")
                video.draw_jobs(missing, work.parent / "refs", log)
            images = draw_keyframes(kf, plan, style, work, log, llm_url=llm_url, model=llm_model)
            apply_keyframes(shots, kf, images, plan, style, work)
            # the talking shots' opening pictures, drawn and checked while the language model is up
            _portraits(shots, {p["id"]: p for p in pics}, plan, style, work, log, kf, llm_url, llm_model)
    if phase == "plan":
        return []
    if phase.startswith("check"):
        return check_clips(shots, kf, plan, work, llm_url, llm_model, log, final=phase == "check-final")
    talk_s = sum(s["dur"] for s in shots if s["kind"] == "talk")
    log(f"  {level}: {sum(s['kind'] == 'talk' for s in shots)} talking shot(s) ({talk_s / 60:.1f} min), "
        f"{sum(s['kind'] == 'action' for s in shots)} action shot(s)")
    return render(shots, pics, plan, style, work, audio, log)
