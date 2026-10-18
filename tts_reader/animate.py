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

import json
import math
import os
import re
import subprocess
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
ACTION_ENGINE = os.environ.get("BOOKCASTER_ACTION_ENGINE", "i2v")
# talking shots: "infinitetalk" (run_infinitetalk.py, 10-17) or "a2v" (SkyReels A2V 19B: intermittent NaNs)
TALK_ENGINE = os.environ.get("BOOKCASTER_TALK_ENGINE", "infinitetalk")
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
                                           "1.5" if ACTION_ENGINE == "i2v" else "7.8"))
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
          "Each person has exactly two arms, two hands and two legs.")


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
    near = {speaker_key(str(e.get("speaker", "")), chars, labels) for e in timeline
            if e.get("kind") == "quote" and abs(float(e["t"]) - t) < 90}
    for p in reversed([p for p in pics if p["t0"] <= t]):
        if p["t1"] < t - 300:
            break
        others = [n for n in p.get("who", []) if n != speaker]
        if speaker in p.get("who", []) and others:
            # only someone who talks with them: a speech to a crowd is a close-up, not a two-shot with
            # whoever was pictured beside the speaker (Peter to the 120 had a silent John, 10-18)
            return next((n for n in others if n in near), None)
    return None


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
    chars, _ = video.looks(plan)
    known = video.ref_prompts(plan, style)
    refs_dir, pdir = work.parent / "refs", work / "anim" / "portraits"
    props = {n: _prop_job(n, d, style, refs_dir) for n, d in plan["look"].get("props", {}).items()}
    anchors = anchor_jobs(kf, plan, style, work, props) if kf else {}
    jobs: dict[str, dict] = {}
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
            s["bbox"] = [[0.52, 0.0, 1.0, 1.0], [0.0, 0.0, 0.48, 1.0]]          # speaker, listener
            s["prompt"] = (f"{me} speaks to {other}, who listens in silence with his mouth closed. "
                           f"Static camera. {style}.")
            body = (f"ONE single continuous illustration of one room, NOT divided into panels, frames or "
                    f"halves. {other} and {me} stand facing each other, both seen from the waist up, a little "
                    f"apart: {other} on the left, turned toward {me} in three-quarter profile, listening, mouth "
                    f"closed; {me} on the right, facing the viewer in three-quarter view, calm, mouth closed, "
                    f"hands still. Only these two people, each appearing exactly ONCE")
        else:
            body = (f"ONE single illustration, not divided into panels: a close-up of {me} alone, head and "
                    f"shoulders, facing slightly toward the viewer, calm, mouth closed")
        night = re.search(r"It is [^.]*\.(?:[^.]*light[^.]*\.)?", s.get("scene_prompt", ""))
        light = night.group(0) if night else _lighting(pic["prompt"])
        refs, parts = [], [_tag_people(body, who, chars), light.rstrip(". ") if light else ""]
        for w in who:
            if f"char:{w}" in anchors:
                refs.append(anchors[f"char:{w}"]["path"])
                parts.append(f"{w} has exactly the face, age and hair of the person in image {len(refs)}")
            elif w in known:
                # the one-figure half of the sheet: given the whole sheet, every solo close-up of
                # Jesus and Peter came back as the sheet itself, the man twice, 4 seeds of 4 (10-17)
                refs.append(str(_figure(video.ref_path(refs_dir, w, known[w][0]), work / "anim" / "refs")))
                parts.append(f"{w} has exactly the face, hair and clothes of the man in image {len(refs)}")
        set_anchor = anchors.get(f"set:{s['where']}")
        if set_anchor:
            refs.append(set_anchor["path"])
            parts.append(f"The room is the one in image {len(refs)}, seen behind them")
        elif s["where"] in known:
            refs.append(str(video.ref_path(refs_dir, s["where"], known[s["where"]][0])))
            parts.append(f"The setting is {s['where']}, as in image {len(refs)}")
        parts += [video.ANATOMY, style, "No text, no lettering, no signature"]
        prompt = ". ".join(x for x in parts if x) + "."
        pid = f"talk-{video.slug(me)}-{s['pic']}-{_hash(prompt)}"
        s["image"] = str(pdir / f"{pid}.png")
        if pid not in jobs:
            jobs[pid] = {"id": pid, "prompt": prompt, "seed": 4242, "path": s["image"], "refs": refs,
                         "frame": {"who": who, "props": [], "must": [
                             "one single picture of one room, not split into panels or halves",
                             f"exactly {len(who)} {'person' if len(who) == 1 else 'people'}, each appearing once"]},
                         "anchors": {w: anchors[f"char:{w}"]["path"] for w in who if f"char:{w}" in anchors},
                         "set": None}
    if not jobs:
        return
    state_f = work / "anim" / "portrait_checks.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {"seeds": {}, "passed": {}, "log": []}
    for j in jobs.values():
        j["seed"] = state["seeds"].get(j["id"], j["seed"])
    keep = ("id", "prompt", "seed", "path", "refs", "size")
    for rnd in range(KEYFRAME_ROUNDS + 1):
        todo = [j for j in jobs.values() if not Path(j["path"]).exists()]
        if todo:
            log(f"  drawing {len(todo)} talking portrait(s)")
            video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in
                             [j for j in anchors.values() if not Path(j["path"]).exists()] + todo], pdir, log)
        if not llm_url:
            break
        bad = check_keyframes(list(jobs.values()), state, chars, {}, llm_url, model, log,
                              final=rnd == KEYFRAME_ROUNDS)
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
  with its own "where", then go back to the scene; an empty place alone is not a flashback. Never show a death, a wound, blood or a body: show the moment before, or the empty place after."""


def plan_keyframes(shots: list[dict], pics: list[dict], timeline: list[dict], total: float, plan: dict,
                   style: str, llm_url: str, model: str, log=print) -> dict:
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
        prompt = (
            "You plan the animation of an illustrated audiobook, one drawn keyframe per 5-second shot.\n"
            f"Scene picture: {pic['prompt']}\nPlace: {pic.get('where', '')}: {places.get(pic.get('where', ''), '')}\n"
            + (f"Group there: {pic['crowd']}\n" if pic.get("crowd") else "") +
            f"People (use these names): {', '.join(chars)}\n"
            f"Props so far: {json.dumps(props, ensure_ascii=False)}\n\n"
            f"The shots and the narration heard during each:\n{lines}\n\n{KEYFRAME_RULES}\n\n"
            'Reply with ONLY JSON: {"props": {"<new prop>": "<description>"}, "frames": [{"shot": "<id>", '
            '"mode": "key|continue", "who": [...], "crowd": "", "where": "<place>", "props": [...], "dress": {}, "must": [...], '
            '"prompt": "<the picture, 2-4 sentences>", "motion": "<what moves>", "carry": false}]}')
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
    _break_chains(frames)
    return {"version": 1, "props": props, "frames": frames}


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
            f.update(mode="key", carry=True, prompt=last.get("prompt", ""), where=last.get("where", ""),
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
    return kf


def _prop_job(name: str, desc: str, style: str, refs_dir: Path) -> dict:
    prompt = (f"Object reference: {name}, {desc.rstrip('.')}. Shown alone, whole and clearly, on a "
              f"plain light background, even light. {style}. No people. No text, no lettering, no signature.")
    path = video.ref_path(refs_dir, f"prop {name}", prompt)
    return {"id": path.stem, "prompt": prompt, "seed": 7, "path": str(path), "size": [1024, 768]}


def _tag_people(prompt: str, who: list[str], chars: dict, dress: dict | None = None) -> str:
    """Each person's first mention gets their distinguishing look inline ("Watson
    (short brown hair, tweed suit) holds the revolver"): with only names, the drawer
    gave Holmes the lamp AND Watson's revolver and swapped their hair and coats (10-17)."""
    for w in who:
        look = _from_behind(chars.get(w, ""), (dress or {}).get(w))
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
    refs_dir = work.parent / "refs"
    kdir = work / "anim" / "keyframes"
    props = {n: _prop_job(n, d, style, refs_dir) for n, d in plan["look"].get("props", {}).items()}
    chars, places = video.looks(plan)
    known = video.ref_prompts(plan, style)
    anchors = anchor_jobs(kf, plan, style, work, props)
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
            if f"char:{w}" in anchors:
                refs.append(anchors[f"char:{w}"]["path"]); used[w] = "anchor"
            elif w in known:
                # the full-length half of the sheet: given the whole sheet (the person twice), the leaping
                # healed man's scenes came out with two Johns, three rounds running (10-18)
                refs.append(str(_figure(video.ref_path(refs_dir, w, known[w][0]), work / "anim" / "refs")))
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
            if used.get(w) == "anchor":
                parts.append(f"{w} has exactly the face, age and hair of the person in image {k}")
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
        set_anchor = anchors.get(f"set:{where}")
        if prev_same:                     # the moment before, in this place: carry it on
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
                refs.append(props[name]["path"])
                parts.append(f"The {name.lower()} looks as in image {len(refs)}")
        parts += [style, "Nothing from a later period than the story: no wristwatches, no clocks, no glasses",
                  "No text, no lettering, no signature"]
        path = str(kdir / f"{f['shot']}.png")
        jobs.append({"id": f["shot"], "pic": f.get("pic"), "carried": bool(prev_same),
                     "prompt": ". ".join(parts) + ".", "path": path,
                     "seed": state["seeds"].get(f["shot"], 5000 + n), "frame": f,
                     "refs": refs, "where": where, "states": st,
                     "anchors": {w: anchors[f"char:{w}"]["path"] for w in who if f"char:{w}" in anchors},
                     "set": set_anchor["path"] if set_anchor and not prev_same else None})
        images[f["shot"]] = last = path
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
        pool = ThreadPoolExecutor(1) if llm_url else None
        by_id = {j["id"]: j for j in jobs}
        look_props = plan["look"].get("props", {})

        def on_done(dj):
            if pool and dj["id"] in by_id:
                checked.add(dj["id"])
                futs.append(pool.submit(check_keyframes, [by_id[dj["id"]]], state, chars, look_props,
                                        llm_url, model, log, final, True))
        if todo_props or todo_anchors or todo:
            keep = ("id", "prompt", "seed", "path", "refs", "size")
            video.draw_jobs([{k: v for k, v in j.items() if k in keep} for j in
                             [j for j in video.reference_jobs(plan, style, refs_dir)
                              if not Path(j["path"]).exists()] + todo_props + todo_anchors + todo], kdir, log,
                            on_done=on_done)
        if not llm_url:
            break
        for f in futs:
            bad |= f.result()
        pool.shutdown()
        # ... and those drawn before (a restarted job) that were never passed
        bad |= check_keyframes([j for j in jobs if j["id"] not in checked], state, chars, look_props,
                               llm_url, model, log, final, True)
        log(f"  checked keyframes: {len(bad)} to draw again")
        if rnd == KEYFRAME_ROUNDS:
            _edit_out_of_period(jobs, state, kdir, log)
        for k, j in enumerate(jobs):
            if j["id"] in bad:
                state["seeds"][j["id"]] = j["seed"] = int(j["seed"]) + 7919
                Path(j["path"]).unlink(missing_ok=True)
                for later in jobs[k + 1:]:            # carried pictures were drawn from this one
                    if not later.get("carried"):
                        break
                    Path(later["path"]).unlink(missing_ok=True)
        state_f.write_text(json.dumps(state, indent=1))
        if not bad:
            break
    return images


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


def check_keyframes(jobs: list[dict], state: dict, chars: dict, props: dict, llm_url: str,
                    model: str, log, final: bool = False, quiet: bool = False) -> set[str]:
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
                 [f"exactly {len(who)} {'person' if len(who) == 1 else 'people'} with a head in the picture "
                  f"(count heads, including small or background figures; hands alone do not count)"])
        items += [f"{w} is recognisable by {_hair(chars[w])} (and, where their clothes show, "
                  f"{_from_behind(chars[w], (f.get('dress') or {}).get(w)).split(', ', 1)[-1]})" for w in who]
        items += [f"a recognisable {n.lower().replace(', ', ' (')}{')' if ', ' in n else ''}: "
                  f"{re.split(r'[,;]', props[n])[0]}" for n in f.get("props", []) if n in props]
        items += list(f.get("must", []))
        items += [m for x in j.get("states", []) for m in x.get("must", [])]
        # a disciple at the Ascension wore a wristwatch (10-17)
        items.append(PERIOD_ITEM)
        q = ("Check this illustration against a list. For each numbered item, look carefully and answer "
             "whether the picture shows it.\n" + "\n".join(f"{i}. {x}" for i, x in enumerate(items, 1)) +
             '\nAnswer with a final line of JSON: {"items": [true/false, ...], "people": <count>, '
             '"notes": "what is missing or wrong"}')
        try:
            reply = video._look_at(llm_url, model, png, q)
            found = re.findall(r"\{[^{}]*\"items\"[^{}]*\}", reply or "")
            data = json.loads(found[-1]) if found else {}
            # a frame with nobody planned (a close-up of hands, a room): the anatomy check's own people
            # count reads hands as people (10-17), so only its drawing-mistake flags count there
            probs = video.check_picture(png, len(who) if who and not crowd else None, llm_url, model)
        except Exception as exc:                 # no vision server: draw unchecked
            log(f"  keyframe check skipped ({exc})")
            return set()
        missing = [items[i] for i, ok in enumerate(data.get("items", [])) if ok is False and i < len(items)]
        try:
            missing += _compare(j, png, llm_url, model)
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
                continue
        state["passed"][j["id"]] = png.stat().st_mtime
    if not quiet:
        log(f"  checked keyframes: {len(bad)} to draw again")
    return bad


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
        igpu("vae", log)                        # talking shots' VAE and action shots' decode
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
                   if ACTION_ENGINE != "i2v" else
                   max((talk_min + len(actions) * ACTION_MIN_PER_CLIP) / 2,
                       max((s["dur"] * TALK_MIN_PER_S for s in talks), default=0)))
            log(f"  rendering {len(talks)} talking shot(s) and {len(actions)} action shot(s) on both cards "
                f"(~{est:.0f} min)")
            t = time.time()
            if ACTION_ENGINE == "i2v":
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
                talk_jobs = _talk_jobs(talks, audio, work)

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
                p65 = None
                ok = sum(Path(s["out"]).exists() for s in talks)
                if talks:
                    log(f"    {ok}/{len(talks)} talking shot(s) ok")
            else:
                n05 = _scene_cut(actions, n05)
                p65 = _action_batch("65", actions[n05:], by_id, plan, style, work)
                _render_talks_a2v(_talk_jobs(talks, audio, work), work, log)
                p05 = _action_batch("05", actions[:n05], by_id, plan, style, work)
            for p in (p65, p05):
                if p:
                    p.wait()
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
                     **({"bbox": s["bbox"]} if s.get("bbox") else {})})
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
    kf = keyframes(work)
    if phase == "render":
        llm_url = None                       # planned and checked already
    if kf is None and llm_url and ACTION_ENGINE == "i2v":
        igpu("strata", log)
        log("  planning keyframes...")
        kf = plan_keyframes(shots, pics, timeline, total, plan, style, llm_url, llm_model, log)
        (work / "anim").mkdir(parents=True, exist_ok=True)
        (work / "anim" / "keyframes.json").write_text(json.dumps(kf, indent=1, ensure_ascii=False))
    if kf:
        _drop_people_props(kf, plan)
    plan["look"]["props"] = {**(kf or {}).get("props", {}), **plan["look"].get("props", {})}
    video.keep_looks(plan, work.parent / "look.json")
    if kf:
        if llm_url:
            igpu("strata", log)            # the keyframe checks look at the pictures
        # one drawing model load for the keyframes, their redraws and the portraits (10-18)
        with video.drawing_session(work / "anim" / "draw.log", log):
            images = draw_keyframes(kf, plan, style, work, log, llm_url=llm_url, model=llm_model)
            apply_keyframes(shots, kf, images, plan, style, work)
            # the talking shots' opening pictures, drawn and checked while the language model is up
            _portraits(shots, {p["id"]: p for p in pics}, plan, style, work, log, kf, llm_url, llm_model)
    if phase == "plan":
        return []
    talk_s = sum(s["dur"] for s in shots if s["kind"] == "talk")
    log(f"  {level}: {sum(s['kind'] == 'talk' for s in shots)} talking shot(s) ({talk_s / 60:.1f} min), "
        f"{sum(s['kind'] == 'action' for s in shots)} action shot(s)")
    return render(shots, pics, plan, style, work, audio, log)
