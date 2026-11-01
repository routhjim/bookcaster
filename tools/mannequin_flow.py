"""MANNEQUIN FLOW (user 10-22 "Bookcaster Video Director: Mannequin-Based Keyframe Workflow"): 3D only for position and
motion. Rigged capsule mannequins + primitive / modelled props are staged and rendered ONLY as control maps (pose
skeleton from the rig's joints, set+props depth without the people); each keyframe is drawn from scratch (full denoise,
no img2img) with pose + depth ControlNet ending part-way, the character LoRAs (regional when several people) and the
place's hero image as reference; LTX animates from there (IC-LoRA with the pose/depth video for physical action).

    ~/tts_reader/.venv/bin/python3 tools/mannequin_flow.py <book> <chapter stem> stage <moment ids...>
    ~/tts_reader/.venv/bin/python3 tools/mannequin_flow.py <book> <chapter stem> keys  <moment ids...> --hero <png>

Output: <book>/work/video/<stem>/mannequin/<id>/ (maps/: pose.mp4, depth_bg.mp4, frames/p_*, dbg_*; key.png).
Resolution: 1152x640 everywhere (the LTX IC-LoRA's half-size control needs multiples of 128).
"""

from __future__ import annotations

import argparse
import filecmp
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
POSE_PY = os.path.expanduser("~/tts-ab/.venv-pose/bin/python")
QWEN_PY = os.path.expanduser("~/tts-ab/.venv-qwenimage/bin/python")
W, H = 1152, 640
LAYERS = "bg=set+creature+prop,people=figure,creature=creature"     # creature alone: its region in the key (10-23)


_REG: list = [None]     # the registry of the book last opened here (paths/_reg): _cast_name's looks


def _reg(book: Path):
    """The book's registry (tools/registry.py, cached until look.json changes): the one owner of who and where."""
    sys.path.insert(0, str(HERE))
    import registry as RG
    R = RG.at(book) or RG.Book(book)
    _REG[0] = R
    return R


def paths(book: Path, stem: str):
    vd = book / "work" / "video" / stem
    try:
        _reg(book)
    except (OSError, ValueError):
        pass
    return vd, vd / "flow3d" / "stage", vd / "mannequin"


def style_of(book: Path, stem: str | None = None) -> str:
    """The book's one art style (registry.style: art_style.txt > the chapter's style.txt > the default; 10-27: the
    keys read art_style.txt only, '' for the Speckled Band, while its sheets and LoRAs were pen-and-ink)."""
    return _reg(book).style(stem)


def direct(book: Path, stem: str, ids: list, log=print) -> None:
    """The DIRECTOR (the language model, plan_review.direct): every scene of the stretch whose moments are not yet
    directed gets its acts (both hands of everyone handling anything; strike, block, operate...) and its beats (the
    mechanics of the action, in order), from the story and the book's director's notes. Staging compiles the acts;
    the clip prompts follow the beats. The language model is stopped after (staging needs the RAM)."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    import plan_review as PRV
    animate.igpu("strata", log)
    try:
        rep = PRV.direct(book, stem, ids, log)
    finally:
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    if rep.get("undirected"):
        log(f"!!! direction: {len(rep['undirected'])} hand(s) left undirected: {rep['undirected'][:6]}")


STAGE_WORKERS = int(os.environ.get("MANNEQUIN_STAGE_WORKERS", "6"))
MAPS_VERSION = "2"                          # 2 (10-23): face points only where the face turns to the camera
CLOSE_FRAC = 0.12                           # people covering this share of the frame = a close framing (keys)
COVER_MIN_S = 4.0                           # a narrated stretch this long with no moment gets its own moments


def cover(book: Path, stem: str, t0: float, t1: float, log=print) -> list:
    """Moments for every narrated stretch of [t0, t1) that has none (10-23: the opening's 29 s held one frame - the
    shot planner gives a picture without people ONE moving shot and lets "its still pan for the rest", and this flow
    has no stills - and nobody showed the knights riding out or burned). One action moment per 5 s of the stretch,
    their keyframes planned by bookcaster's keyframe planner from the narration under them (events it only implies
    included), added as the scene layer's extra moments so staging, the director and the keys treat them as any
    other. -> the new moment ids."""
    import math
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate, video
    from tts_reader import bookcaster as BC
    vd, _, _ = paths(book, stem)
    f = next(x for x in BC.chapter_files(book) if x.stem == stem)
    text = BC._chapter_text(f)
    tl = json.loads((book / "work" / f"{stem}.timeline.json").read_text())
    total = video.audio_seconds(book / "work" / "final" / f"{stem}.mp3")
    plan = json.loads((vd / "plan.json").read_text())
    xp = vd / "stage" / "extra_moments.json"
    extra = json.loads(xp.read_text()) if xp.exists() else []
    moments = sorted(json.loads((vd / "anim" / "manifest.json").read_text()) + extra, key=lambda m: m["t"])
    have = {m["id"] for m in moments}
    edges = [(0.0, 0.0)] + [(float(m["t"]), float(m["t"]) + float(m["dur"])) for m in moments] + [(total, total)]
    holes, end = [], 0.0
    for a_, b_ in edges:
        if a_ - end >= COVER_MIN_S and a_ > t0 and end < t1:
            holes.append((max(end, t0), min(a_, t1)))
        end = max(end, b_)
    pics = video.timed(plan, text, tl, total)
    new = []
    for a_, b_ in holes:
        n = max(1, round((b_ - a_) / animate.ACTION_S))
        d = (b_ - a_) / n
        for k in range(n):
            t = round(a_ + k * d, 2)
            mid = f"action-{int(round(t * 10)):06d}"
            p = animate.picture_at(pics, t) or (pics[0] if pics else None)
            if mid in have or p is None:
                continue
            new.append({"id": mid, "kind": "action", "t": t, "dur": round(d, 2), "pic": p["id"], "extra": True})
    if not new:
        return []
    log(f"cover: {len(new)} moment(s) for {len(holes)} stretch(es) without one: "
        + ", ".join(f"{a_:.0f}-{b_:.0f}s" for a_, b_ in holes))
    animate.igpu("strata", log)
    starts = video.paragraph_times(video.paragraphs(text), tl)
    kf_new = animate.plan_keyframes(new, pics, tl, total, plan, style_of(book, stem),
                                    BC.LLM_URL, BC.LLM_MODEL, log, starts)
    for fr in kf_new["frames"]:             # a stretch of its own: every one a new picture
        fr["mode"] = "key"
    animate.apply_wardrobe(kf_new, plan, starts)
    kp = vd / "anim" / "keyframes.json"
    kf = _keyframes(vd)                     # (10-29: a pace change deletes anim/ - the moments kept, this file not)
    kf["frames"] = sorted([x for x in kf["frames"] if x.get("shot") not in {m["id"] for m in new}] + kf_new["frames"],
                          key=lambda x: animate._shot_t(x.get("shot", "")))
    kf.setdefault("props", {}).update({k: v for k, v in (kf_new.get("props") or {}).items() if k not in kf["props"]})
    kp.write_text(json.dumps(kf, indent=1, ensure_ascii=False))
    xp.parent.mkdir(parents=True, exist_ok=True)
    xp.write_text(json.dumps(extra + new, indent=1))
    import flow3d as F
    F.scene_frames(book, f, vd, moments + new, log)          # each new moment into its scene (place, who is there)
    return [m["id"] for m in new]


def scene(book: Path, stem: str, log=print) -> None:
    """The scene layer (scene_stage.py: places, marks, cameras, contact actions incl. the book's approved recipes) and
    the solved motion of every moment whose shot changed (remake_chapter.stage, motion only: no 3D beauty renders,
    which this flow never uses)."""
    sys.path.insert(0, str(HERE))
    import flow3d as F
    vd, _, _ = paths(book, stem)
    lg = vd / "stage" / "mannequin_scene.log"
    rc = F.run([F.POSE_PY, "-I", HERE / "scene_stage.py", book, stem, "--sheets", "--no-render"], lg)
    if rc:
        raise SystemExit(f"scene layer failed (rc {rc}): see {lg}")
    for sub in ("plan", "build"):            # a prop the scene layer added needs its mesh
        rc = F.run([F.POSE_PY, "-I", HERE / "props_build.py", sub, book, stem, *(["--card", "any"] if sub == "build" else [])], lg)
        if rc:
            raise SystemExit(f"props {sub} after the scene layer failed (rc {rc}): see {lg}")
    R, cx = F._ctx(book, stem)
    R.stage(cx, R.plan(cx), steps="motion")
    log("scene layer + motion: done")


def stage(book: Path, stem: str, ids: list, log=print) -> dict:
    """Each moment staged with mannequins: an action moment with its solved motion (motion_batch --render of its motion
    dir), a talk moment from its talk shot; the renderer's --mannequin draws the people as capsules from the same joints."""
    vd, sd, md = paths(book, stem)
    out = {}
    for mid in ids:
        od = md / mid
        maps = od / "maps"
        src_sig = sd / "motion" / mid / "input.sig"
        vf = maps / "version"
        if (maps / "pose.mp4").exists() and (not vf.exists() or vf.read_text().strip() != MAPS_VERSION):
            shutil.rmtree(maps)                   # drawn by an older staging renderer
            log(f"{mid}: maps redrawn (renderer v{MAPS_VERSION})")
        if (maps / "pose.mp4").exists() and _creatures(md, mid) and \
                not (maps / "frames" / "depth_0000_creature.npy").exists():
            shutil.rmtree(maps)                   # drawn before the creature layer (10-23)
            log(f"{mid}: maps redrawn with the creature layer")
        if (maps / "pose.mp4").exists():
            got = od / "motion" / "input.sig"     # the scene step re-staged it (10-22: the approved recipe changed
            same = not (src_sig.exists() and got.exists()) or src_sig.read_text() == got.read_text()   # 5 strikes)
            # ... or rebuilt the same input's motion (10-23: the killing thrust's failed build was retried - same
            # shot and beats, new joints - and its maps kept the old upright dragon)
            new = [j for j in (sd / "motion" / mid).glob("joints_*.npz") if j.stat().st_mtime > (maps / "pose.mp4").stat().st_mtime]
            if same and not new:
                out[mid] = "cached"
                continue
            shutil.rmtree(maps)
            log(f"{mid}: re-staged, maps redrawn")
        od.mkdir(parents=True, exist_ok=True)
        extra = ["--target", "ltx", "--all-passes", "--mannequin", "--keep-frames", "--layers", LAYERS]
        mdir = sd / "motion" / mid
        if (mdir / "render.json").exists():
            work = od / "motion"
            if work.exists():
                shutil.rmtree(work)
            shutil.copytree(mdir, work)
            cmd = [POSE_PY, "-I", str(HERE / "motion_batch.py"), "--render", str(work), str(maps), "--", *extra]
        elif (sd / mid / "shot_talk.json").exists():
            cmd = [POSE_PY, "-I", str(HERE / "blockout_video.py"), str(sd / mid / "shot_talk.json"), str(maps),
                   "--frames", "25", *extra]
        else:
            out[mid] = "no motion or talk shot"
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if "--render" in cmd and (maps / "frames" / "p_0000.png").exists() and not _visible(maps, work):
            # nobody in the picture (10-23: the killing thrust's camera stood inside the rock): the camera pulled
            # back to a wide on everyone, rendered once more
            rj = json.loads((work / "render.json").read_text())
            sh = json.loads(Path(rj["shot"]).read_text())
            sh["camera"] = {k: v for k, v in (sh.get("camera") or {}).items() if k in ("line",)}
            hu = [f["who"] for f in sh.get("figures") or [] if f.get("kind", "human") != "creature" and not f.get("attach")]
            if hu and any(f.get("kind") == "creature" for f in sh.get("figures") or []):
                sh["camera"].update(framing="medium", target=hu[0])     # the over-the-shoulder up at the creature
            else:                                                       # (a wide left the fighters tiny, 10-23)
                sh["camera"].update(framing="wide", target=[f["who"] for f in sh.get("figures") or []])
            (work / "shot_render.json").write_text(json.dumps(sh, indent=1))
            rj["shot"] = str(work / "shot_render.json")
            (work / "render.json").write_text(json.dumps(rj, indent=1))
            shutil.rmtree(maps)
            log(f"{mid}: a fighter not visible in the staged camera: re-rendered ({sh['camera']})")
            r = subprocess.run(cmd, capture_output=True, text=True)
        (od / "stage.log").write_text(r.stdout[-6000:] + r.stderr[-6000:])
        if maps.exists():
            (maps / "version").write_text(MAPS_VERSION)
        # rc 2 = grip/contact violations over the clip (informational, as at staging): the maps are what count
        out[mid] = ("ok" if r.returncode == 0 else "ok (check violations, see maps/checks.json)") \
            if (maps / "frames" / "p_0000.png").exists() else f"FAILED rc {r.returncode}"
        log(f"{mid}: stage {out[mid]}")
    return out


def _keyframes(vd: Path) -> dict:
    """The planner's anim/keyframes.json, or an empty one when it is gone (a pace change deletes anim/)."""
    kp = vd / "anim" / "keyframes.json"
    try:
        kf = json.loads(kp.read_text())
    except (OSError, ValueError):
        kf = {}
    kf.setdefault("frames", [])
    kf.setdefault("props", {})
    return kf


def _frame(book: Path, stem: str, mid: str) -> dict:
    vd, _, _ = paths(book, stem)
    kf = _keyframes(vd)
    return next((f for f in kf["frames"] if f.get("shot") == mid), {})


def _who(book: Path, stem: str, mid: str) -> list:
    vd, sd, _ = paths(book, stem)
    for p in (vd / "stage" / "shots" / f"{mid}.json", sd / mid / "shot_talk.json"):
        if p.exists():
            return [f["who"] for f in json.loads(p.read_text()).get("figures", [])]
    return []


def _view(md: Path, mid: str, who: str):
    """How the staged camera sees `who` at frame 0 -> (words, lora weight factor): "seen from behind, facing
    Vorgrath" etc. (10-23: an over-the-shoulder from behind Edric was drawn with him facing the viewer - the skeleton
    hardly says front from back, and his LoRA learned his face from front views - so the words say it and the LoRA
    is lighter when his face is away)."""
    import numpy as np
    jp, sj = md / mid / "motion" / f"joints_{who}.npz", md / mid / "maps" / "scene.json"
    if not (jp.exists() and sj.exists()):
        return "", 1.0
    try:
        z = np.load(jp)
        names = [str(n) for n in z["names"]]
        P = z["pos"][0]
        hip = P[names.index("l_hip")] - P[names.index("r_hip")]
        fwd = np.array([hip[1], -hip[0]])                          # left hip to the left: forward (the toes' way)
        cam = json.loads(sj.read_text())["frames"][0]["camera"]
        to = np.asarray(cam["pos"][:2], float) - P[names.index("pelvis")][:2]
        right = np.asarray(cam["R"], float)[:2, 0]
    except (OSError, ValueError, KeyError, IndexError):
        return "", 1.0
    c = float(np.dot(fwd, to) / (np.linalg.norm(fwd) * np.linalg.norm(to) + 1e-9))
    ang = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    face = ""
    try:
        f = next((g for g in json.loads((md / mid / "motion" / "shot.json").read_text())["figures"] if g["who"] == who), {})
        face = f" facing {f['facing']}" if isinstance(f.get("facing"), str) else ""
    except (OSError, ValueError, KeyError, StopIteration):
        pass
    if ang < 55:
        return (f"facing the viewer{face.replace(' facing', ', toward') if face else ''}", 1.0)
    if ang > 125:
        return (f"seen from behind, back to the viewer{face and ','}{face}", 0.6)
    side = "right" if float(np.dot(fwd, right)) > 0 else "left"
    return (f"seen in profile, facing screen {side}{face and ','}{face}", 0.8)


def _guide(md: Path, mid: str):
    """The control video that steers a moment's clip, or None (prompt + key only): a staged action that LANDS -
    a hammer landing (impacts) or a contact of a blow, block or claw on a figure (10-23: the fight's block, tail
    sweep and thrust went prompt-only and never connected). The skeleton video carries only the people, so a shot
    with a creature is steered by the DEPTH video (both bodies; the union IC-LoRA reads either)."""
    maps = md / mid / "maps"
    pose, depth = maps / "pose.mp4", maps / "depth.mp4"
    try:
        sh = json.loads((md / mid / "motion" / "shot.json").read_text())
    except (OSError, ValueError):
        sh = {}
    figs = {f["who"]: f for f in sh.get("figures") or []}
    creature = any(f.get("kind") == "creature" for f in figs.values())
    import interaction as IA
    beats = []
    try:                                    # the director's beats: a blow, block or sweep that LANDS on someone
        kf = json.loads((md.parent / "anim" / "keyframes.json").read_text())
        beats = next((f.get("beats") or [] for f in kf["frames"] if f.get("shot") == mid), [])
    except (OSError, ValueError, KeyError):
        pass
    lands = bool(impacts(maps / "anchors.json")) or any(
        str(c.get("to", "")).split(":")[0] in figs or c.get("kind") in ("block", "strike")
        for c in sh.get("contacts") or []) or (len(figs) > 1 and any(IA.RESULT.search(str(b_)) for b_ in beats))
    if not lands:
        return None
    if creature and depth.exists():
        return depth
    return pose if pose.exists() else None


GUIDE = "depth"                             # how a landing action's staging steers its clip (run(guide=...))
GUIDE_ATTN = {"depth": 0.35, "full": 1.0}   # the depth video's attention strength
PIN_S = 0.9                                 # a pin key's strength at its frame


# FIGHTS (10-23, the user watched every route side by side): a moment with a creature is drawn from an OPENING
# layout key (both whole, apart, facing) and the director's one action sentence on the two-stage LTX (dev + CFG/STG +
# a negative prompt), with no staging guide. The staged guide, the pins and the depth video fused the bodies and lost
# the action; the fast model melted the dragon's head at the contact; "magic"/"enchanted" words and an unsourced
# "fire breathing" put the fire in the knight's hands.
FIGHT_RULES = """You direct one shot of a fight in a video made from a book. The shot lasts about {sec:.0f} seconds. A video
model starts from ONE painted picture of the shot's opening and animates it from your ACTION words alone, with no other
guide, so the action must be complete and unambiguous.

The narration over this shot: {text}
The director's beats for this moment: {beats}
Who is in the shot: {people}
The place: {place}
The previous fight shot's action: {prev}

Write:
- "left" and "right": exactly one of these names each - {names} - for who is on the left and who is on the right of
  the opening picture (only one in the shot: that name in "left", "right" empty). Names go ONLY here.
- "framing": "wide" or "medium" ("wide" when the creature's tail, wings or whole body act).
- "layout": one sentence for the OPENING picture only: where each one is, which way each faces, how far apart they are,
  what each holds, and each body's posture: a fighter is in a READY fighting stance (weight forward, weapon raised
  toward the opponent, leaning in), never standing at rest with arms down; a four-legged creature crouches, stands or
  lies on all four legs; anyone knocked down but alive is on the ground propped on an elbow, eyes open, still holding
  the weapon; anyone dead lies flat on the ground, eyes closed, motionless. Every body part the action will use is in
  the picture (a tail that sweeps, a claw that strikes, the wound a blade goes into). Both bodies whole and clearly
  apart, no contact yet, nothing in mid-action.
- "action": what happens in the shot, in order, in plain physical verbs, each action played through to its result (the
  blade goes in, the fire hits) - the narration's result happens in THIS shot when the narration says it (it says
  "drove the sword in": the blade goes in, not a press or an aim). Every effect names its source ("a stream of fire from the dragon's mouth"); say what
  each person's hands hold ("his hands grip only the sword"); call each one by a plain description (a young man in
  steel plate armour, a huge black dragon), never by name; no magic words (magic, enchanted, spell, glowing) and no
  camera directions. At most 70 words.
{fix}Reply with JSON only: {{"left": "...", "right": "...", "framing": "...", "layout": "...", "action": "..."}}"""
FIGHT_NEG = (", deformed anatomy, melting, morphing, bloated swollen body, egg-shaped head, distorted creature, extra "
             "limbs, fused bodies, fire from hands, magic spell, energy blast, glowing hands, explosion from a sword")
MAGIC = re.compile(r"\b(magic\w*|enchant\w*|spell\w*|sorcer\w*|glow\w*)\b", re.I)
EFFECT = re.compile(r"\b(fire|flames?|smoke|sparks?|lightning|embers?)\b", re.I)
LEFT_BOX, RIGHT_BOX = [0.02, 0.1, 0.5, 1.0], [0.42, 0.0, 1.0, 1.0]


def _fight(md: Path, e: dict) -> bool:
    """A moment on the fight route: an action moment with a creature in its shot."""
    return e.get("route") != "talk" and bool(_creatures(md, e["id"]))


MALE = re.compile(r"\b(man|boy|youth|lad|gentleman)\b", re.I)
FEMALE = re.compile(r"\b(woman|girl|lady)\b", re.I)


def _cast_name(v, cast: list, crs: list, R=None):
    """The director's left/right (or a speaker, a listener) as a cast name, or None (never a guess). Any spelling of a
    cast member's name (the registry: Dr./Doctor, a unique surname, a voice label), or the full name inside a longer
    phrase; else a DESCRIPTION that fits exactly one cast member's look (10-23: it wrote "the young man" there too, all
    three tries): the creature kind their look names ("the dragon"), or a man / a woman. 10-27: substring matching and
    story words (knight, princess, smith, king) gave "a woman" to the first person of the cast, whoever it was."""
    v = str(v or "").strip()
    if not v or v.lower() in ("none", "null", "empty"):
        return None
    R = R or _REG[0]
    cast = list(cast or [])
    if v in cast:
        return v
    if R is not None and (k := R.key_in(v, cast)) is not None:
        return k
    sys.path.insert(0, str(HERE))
    import registry as RG
    vw = {x.lower() for x in RG._words(v)}       # "Holmes", "Miss Stoner": the one cast member with those words
    for k in ([w for w in cast if RG.person_key(w) == RG.person_key(v)],
              [w for w in cast if vw and vw <= {x.lower() for x in RG._words(w)}],
              [w for w in cast if re.search(rf"(?<![\w-]){re.escape(w)}(?![\w-])", v, re.I)]):
        if len(k) == 1:
            return k[0]
        if k:
            return None
    if R is None:
        return None
    from tts_reader import video
    lk = {w: R.look(w) for w in cast}
    kinds = video.creature_kinds(lk)
    named = [w for w, kd in kinds.items() if re.search(rf"\b{kd}s?\b", v, re.I)]
    if named:
        return named[0] if len(named) == 1 else None
    head = {w: " ".join(re.findall(r"[a-z]+", (d or "").lower())[:6]) for w, d in lk.items() if w not in kinds}
    for rx in (MALE, FEMALE):
        if rx.search(v):
            fit = [w for w, h in head.items() if rx.search(h)]
            return fit[0] if len(fit) == 1 else None
    return None


def _action_faults(a: dict, cast: list) -> list:
    """What breaks the fight rules in a director's answer."""
    act = str(a.get("action", ""))
    out = []
    named = [w for w in cast if re.search(rf"\b{re.escape(w.split()[-1])}\b", act)]
    if named:
        out.append(f"it names {', '.join(named)} (use a plain description)")
    if MAGIC.search(act):
        out.append(f"it uses the word '{MAGIC.search(act).group(0)}'")
    for sent in re.split(r"(?<=[.;])\s+", act):
        if EFFECT.search(sent) and not re.search(r"\bfrom\b", sent):
            out.append(f"'{EFFECT.search(sent).group(0)}' does not say where it comes from")
    if len(cast) == 1:
        if a.get("left") != cast[0]:
            out.append(f"left must be {cast[0]}")
    elif a.get("left") not in cast or a.get("right") not in cast or a.get("left") == a.get("right"):
        out.append(f"left/right must be two of: {', '.join(cast)}")
    if len(act.split()) > 90 or not a.get("layout"):
        out.append("the action is over 70 words or the layout is missing")
    return out


def fight_direct(book: Path, stem: str, es: list, log=print) -> dict:
    """{mid: {left, right, framing, layout, action}} for every fight moment, from the language model (iGPU), checked
    against the rules (asked again with the faults, twice); cached by input in fight.json beside the moment."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    import flow3d as F
    R, cx = F._ctx(book, stem)
    _, _, md = paths(book, stem)
    out, asked, prev, here = {}, False, "(none: the first shot of the fight)", ""
    tlp = book / "work" / f"{stem}.timeline.json"
    tl = json.loads(tlp.read_text()) if tlp.exists() else []
    for e in es:
        here = e.get("where") or here
        if not _fight(md, e):
            continue
        f = _frame(book, stem, e["id"])
        hum = [w for w in _who(book, stem, e["id"]) if w not in _creatures(md, e["id"])]
        cast = (hum[:1] + _creatures(md, e["id"])[:1]) if hum else _creatures(md, e["id"])[:2]
        # the narration spoken over the moment (10-23: the frames carried no text, and the thrust at 6:01 - "drove the
        # sword in" - was directed as a press that never went in)
        sec = float(e.get("frames") or 121) / 24.0
        # ... up to the NEXT moment's start (10-23: 6:01 also played the scream and the fall of 6:06 and 6:11)
        nxt = min([float(x["t"]) for x in es if float(x["t"]) > float(e["t"])] + [float(e["t"]) + sec])
        said = " ".join(x["text"] for x in tl if float(e["t"]) - 1.0 <= x["t"] < nxt - 1.0)
        looks = {w: _brief(_appearance(cx.look(w, e["t"]) or "")) for w in cast}
        q0 = FIGHT_RULES.format(sec=sec, text=(said or f.get("text") or "")[:700],
                                beats=" / ".join(f.get("beats") or []) or "(none)", place=here, prev=prev,
                                people="; ".join(f"{w} ({looks[w]})" for w in cast), fix="{fix}",
                                names=", ".join(cast))
        k = _h(q0)
        cp = md / e["id"] / "fight.json"
        if cp.exists() and json.loads(cp.read_text()).get("k") == k:
            out[e["id"]] = json.loads(cp.read_text())
            prev = out[e["id"]]["action"]
            continue
        if not asked:
            animate.igpu("strata", log)
            asked = True
        a, faults = {}, []
        for n in range(3):
            fix = ("Your last answer broke the rules: " + "; ".join(faults) + ". Write it again.\n") if faults else ""
            reply = _chat(BC.LLM_URL, BC.LLM_MODEL, q0.replace("{fix}", fix), max_tokens=1200)
            m = re.search(r"\{.*\}", reply or "", re.S)
            try:
                a = json.loads(m.group(0)) if m else {}
            except ValueError:
                a = {}
            crs_ = _creatures(md, e["id"])
            a["left"], a["right"] = _cast_name(a.get("left"), cast, crs_, cx.R), _cast_name(a.get("right"), cast, crs_, cx.R)
            faults = _action_faults(a, cast)
            if not faults:
                break
        if not a.get("action") or not a.get("layout") or a.get("left") not in cast or \
                (len(cast) > 1 and a.get("right") not in cast):
            log(f"!!! {e['id']}: no usable fight direction ({faults}): the ordinary action route")
            continue
        if faults:
            log(f"!!! {e['id']}: fight direction still breaks the rules: {faults}")
        a = {**{x: a.get(x) for x in ("left", "framing", "layout", "action")},
             "right": a.get("right") if len(cast) > 1 else None, "k": k}
        (md / e["id"]).mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(a, indent=1))
        out[e["id"]] = a
        prev = a["action"]
        log(f"{e['id']} fight: {a['left']} | {a['right']} ({a['framing']}): {a['layout']} || {a['action']}")
    if asked:
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    return out


DOWN = re.compile(r"\b(lies|lying|lie|sprawled|collapsed|knocked (down|flat)|propped on|on his back|on her back)\b", re.I)


def _fight_prompt(book: Path, stem: str, mid: str, lay: dict) -> str:
    """The opening key's words: the layout, both looks (wardrobe in force), the place, period and style."""
    vd, _, _ = paths(book, stem)
    RG = _reg(book)
    style = style_of(book, stem)
    L, Rt = lay["left"], lay["right"]
    dress = _dress(book, stem, mid, [L, Rt])

    _, _, md = paths(book, stem)
    crs = _creatures(md, mid)

    def one(w, side, other):
        lk = _look(book, w)
        if w in dress:
            lk = re.split(r",?\s*\bwearing\b", lk)[0] + ", wearing " + dress[w]
        # (10-23: "seen in profile, not facing the viewer" was ignored and no checker can tell the difference; the good
        # fight key had Edric in a fighting stance, sword raised toward the dragon, and LTX turned him in the action)
        # (10-23: a layout with the knight knocked flat + this stance = TWO Edrics, one lying, one standing)
        turn = (f"in a fighting stance, weapon raised toward {other}, leaning in" if w not in crs and not
                DOWN.search(lay.get("layout") or "") else f"facing {other}") if other else ""
        return (f"{_lora(book, w)[1] or ''} {w}, {'on the ' + side + ', ' if other else ''}whole body visible"
                f"{', ' + turn if turn else ''}: {lk}").strip()
    place = RG.place(_place_of(book, stem, mid) or "") or ""
    who_ = f"{one(L, 'left', Rt)}. {one(Rt, 'right', L)}. Both bodies whole and clearly apart." if Rt else f"{one(L, '', None)}."
    return (f"{str(lay.get('framing') or 'wide').capitalize()} shot{' in ' + place if place else ''}. {lay['layout']} "
            f"{who_} {period_line(book)} {style} "
            f"No text.").replace("..", ".").replace("  ", " ")


def fight_clip_ok(mp4: Path, people: int | None = None) -> list:
    """The clip's faults from frames across it (the vision judge): a melted/swollen/merged body, and with `people`
    more people than the shot has (10-27: a second Watson grew in the bed; the "fire from hands" question was the
    forge's alone and is gone, user: "these checks seem to be too context specific")."""
    import characters as CH
    items = []
    try:                                    # a short clip (a ~2.5 s quiet shot) is sampled at 30% and 75% of it, not
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                    str(mp4)], capture_output=True, text=True).stdout.strip())   # once at 1.5 s (10-26)
    except ValueError:
        dur = 5.0
    # 10-30: with a head count, 4 points to the clip's end (a second Helen walked through between 2.3 s and the end)
    pts = (tuple(round(dur * f, 1) for f in (0.2, 0.45, 0.7, 0.95)) if people else
           (1.5, 3.0, 4.5) if dur >= 3.2 else (round(dur * 0.3, 1), round(dur * 0.75, 1)))
    for t in pts:
        png = mp4.with_name(f"{mp4.stem}_chk{int(t * 10)}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(mp4), "-frames:v", "1", str(png)])
        if png.exists():
            ask = {"whole": "Answer yes or no: is every person and creature in this picture whole and natural, with a "
                            "normal head and body, each person with two arms and two hands (no extra limbs or hands; "
                            "nothing melted, swollen, merged or deformed)?"}
            if people:
                ask["count"] = "How many people are in this picture? Answer with a number."
            items.append({"id": f"{t}", "image": str(png), "ask": ask})
    ans = CH.vl_ask(items) if items else {}
    bad = []
    for it in items:
        a_ = ans.get(it["id"]) or {}
        if not CH._yes(a_.get("whole")):
            bad.append(f"deformed body at {it['id']} s")
        m_ = re.search(r"\d+", str(a_.get("count") or ""))
        if people and m_ and int(m_.group()) > people:
            bad.append(f"{m_.group()} people, not {people}, at {it['id']} s")
    return bad


def pin_frames(md: Path, mid: str, n: int) -> list:
    """The frames a landing action is pinned at besides 0: its HIT (the fastest-moving staged frame, hands and
    claws included, kept a second from either end) and its last frame - the guided clips (10-23 fight) followed
    the staged motion only loosely and fused the bodies where the key and the depth video disagreed."""
    import numpy as np
    last = n - 1
    v = None
    for jp in sorted((md / mid / "motion").glob("joints_*.npz")):
        pos = np.load(jp)["pos"][:n]
        sp = np.linalg.norm(np.diff(pos, axis=0), axis=2).mean(axis=1)
        v = sp if v is None else v[:len(sp)] + sp[:len(v)]
    if v is None or len(v) < 3:
        return [last]
    v = _smooth(v, 9)
    lo, hi = min(24, last // 3), max(min(24, last // 3), last - 24)
    hit = int(lo + np.argmax(v[lo:hi + 1])) if hi > lo else last // 2
    return [hit, last]


def pins(md: Path, mid: str) -> list:
    """[(frame, pin key)] drawn for a moment (pin_<frame>.png)."""
    return sorted(((int(k.stem.split("_")[1]), k) for k in (md / mid).glob("pin_*.png") if len(k.stem.split("_")) == 2),
                  key=lambda x: x[0])


def pin_keys(book: Path, stem: str, es: list, hero_of: dict, log=print) -> dict:
    """guide=pins: each landing action's hit and end keys, drawn on the staged pose + depth of those frames with the
    moment's own key as the first reference (so the people, the creature and the place stay the ones in it); the
    first seed with no text or blank edges is kept as pin_<frame>.png."""
    from concurrent.futures import ThreadPoolExecutor
    import characters as CH
    _, _, md = paths(book, stem)
    want = {}
    for e in es:
        m = e["id"]
        if e["route"] == "talk" or _guide(md, m) is None or not (md / m / "key.png").exists():
            continue
        n = min(int(e["frames"]), len(list((md / m / "maps" / "frames").glob("p_*.png"))))
        for f in pin_frames(md, m, n):
            want.setdefault(f, []).append(m)
    for f, mids in want.items():
        by = {}
        for m in mids:
            by.setdefault(str(hero_of[m]), []).append(m)
        for h_, ms in by.items():
            half = [ms[0::2], ms[1::2]]
            with ThreadPoolExecutor(2) as ex:
                list(ex.map(lambda a: keys(book, stem, a[0], Path(h_), a[1], log=log, at=f,
                                           refs={m: md / m / "key.png" for m in a[0]}) if a[0] else None,
                            zip(half, ("05", "65"))))
    out = {}
    for f, mids in want.items():
        ks = [(m, k) for m in mids for k in sorted((md / m).glob(f"pin_{f}_*.png"))]
        ans = CH.vl_ask([{"id": f"{m}|{k.stem}", "image": str(k), "ask": {"text": "Answer yes or no: does this "
                          "picture contain any written text, letters, a page, a panel or a border?"}}
                         for m, k in ks]) if ks else {}
        for m in mids:
            mk = [k for mm, k in ks if mm == m]
            clean = [k for k in mk if not CH._yes((ans.get(f"{m}|{k.stem}") or {}).get("text")) and not _blank_edges(k)]
            best = (clean or mk or [None])[0]
            if best is None:
                log(f"!!! {m}: no pin key at frame {f}")
                continue
            pp = md / m / f"pin_{f}.png"
            if not (pp.exists() and filecmp.cmp(best, pp, shallow=False)):
                shutil.copy(best, pp)
            out.setdefault(m, []).append(f)
    for m, fs in out.items():
        log(f"{m}: pinned at frames {sorted(fs)}")
    return out


def _overlap_factor(shot: list, who: str) -> float:
    """0.65 when `who`'s region is mostly inside another's (10-23: Edric's and the dragon's boxes overlapped at 5:26
    and the two LoRAs fused them into one armoured creature), else 1."""
    me = next((b for w, b in shot if w == who), None)
    if not me:
        return 1.0
    area = max((me[2] - me[0]) * (me[3] - me[1]), 1e-6)
    for w, b in shot:
        if w == who or not b:
            continue
        ix = max(0.0, min(me[2], b[2]) - max(me[0], b[0])) * max(0.0, min(me[3], b[3]) - max(me[1], b[1]))
        if ix / min(area, max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)) > 0.5:
            return 0.65
    return 1.0


def _visible(maps: Path, work: Path, frac: float = 0.005) -> bool:
    """Someone of the staged shot covers some of frame 0 (people / creature depth layers); a close-up may leave the
    others out, a camera inside the rock shows nobody."""
    import numpy as np
    fr = maps / "frames"
    try:
        figs = json.loads((work / "shot.json").read_text()).get("figures") or []
    except (OSError, ValueError):
        return True
    try:
        sh = json.loads((work / "shot.json").read_text())
    except (OSError, ValueError):
        sh = {}
    cr = {f["who"] for f in figs if f.get("kind") == "creature"}
    fight = any(str(c.get("to", "")).split(":")[0] in cr for c in sh.get("contacts") or [])
    cover = {}
    for kind, layer in (("human", "people"), ("creature", "creature")):
        p = fr / f"depth_0000_{layer}.npy"
        if any(f.get("kind", "human") == kind for f in figs) and p.exists():
            cover[kind] = float((np.load(p) < 1e9).mean())
    if not cover:
        return True
    if fight:                               # a blow on a creature: the striker AND the creature (10-23: the killing
        return all(v >= frac for v in cover.values())        # thrust showed the dragon and 0.2 % of Edric)
    return any(v >= frac for v in cover.values())


def _creatures(md: Path, mid: str) -> list:
    """The shot's creatures (kind creature in its shot file)."""
    for p in (md.parent / "stage" / "shots" / f"{mid}.json", md.parent / "flow3d" / "stage" / mid / "shot_talk.json"):
        if p.exists():
            return [f["who"] for f in json.loads(p.read_text()).get("figures", []) if f.get("kind") == "creature"]
    return []


def _in_shot(md: Path, mid: str, fi: int = 0) -> list:
    """[(who, box)] for the people the staged camera actually frames (frame 0 of anchors.json), box = their 0-1 region
    when two or more share the picture, else None. 10-22: the talk shots frame ONE speaker while the shot lists both,
    and the listener's LoRA and look went onto the speaker. Regions: the people-depth mask split into vertical bands
    at the midpoints between the figures' hands, each band's mask bbox padded 6 %."""
    import numpy as np
    a = md / mid / "maps" / "anchors.json"
    if not a.exists():
        return []
    fr_ = json.loads(a.read_text())["frames"]
    figs = fr_[min(fi, len(fr_) - 1)]["figures"]
    # the creature from its own depth layer (10-23: the keys left Vorgrath out - no name, look or LoRA - and he
    # showed up only when the moment's words happened to name him)
    cp = md / mid / "maps" / "frames" / f"depth_{fi:04d}_creature.npy"
    cm = np.load(cp) < 1e9 if cp.exists() else None
    cr = [w for w in _creatures(md, mid)] if cm is not None and cm.mean() > 0.005 else []

    def box(m):
        ys, xs_ = np.nonzero(m)
        bx = [xs_.min() / W, ys.min() / H, xs_.max() / W, ys.max() / H]
        return [round(float(v), 3) for v in (max(0.0, bx[0] - .06), max(0.0, bx[1] - .06), min(1.0, bx[2] + .06),
                                             min(1.0, bx[3] + .06))]
    if cr:
        dp = md / mid / "maps" / "frames" / f"depth_{fi:04d}_people.npy"
        pm = np.load(dp) < 1e9 if dp.exists() else None
        if len(figs) == 1 and pm is not None and pm.any():
            return [(figs[0]["who"], box(pm))] + [(w, box(cm)) for w in cr]
        if not figs:
            return [(w, None) for w in cr]
    if len(figs) < 2:
        return [(f["who"], None) for f in figs]
    def hx(f):                              # a hand off-screen is None (10-23 overnight: action-000744)
        hs = [h[0] for h in (f.get("hands") or {}).values() if h]
        return float(np.mean(hs)) if hs else W / 2
    xs = [hx(f) for f in figs]
    order = sorted(range(len(figs)), key=lambda i: xs[i])
    cuts = [0.0] + [(xs[order[k]] + xs[order[k + 1]]) / 2 for k in range(len(order) - 1)] + [float(W)]
    dp = md / mid / "maps" / "frames" / f"depth_{fi:04d}_people.npy"
    mask = np.load(dp) < 1e9 if dp.exists() else np.ones((H, W), bool)
    out = {}
    for k, i in enumerate(order):
        x0, x1 = int(cuts[k]), int(cuts[k + 1])
        ys, xs_ = np.nonzero(mask[:, x0:x1])
        if len(ys):
            bx = [(x0 + xs_.min()) / W, ys.min() / H, (x0 + xs_.max()) / W, ys.max() / H]
        else:
            bx = [x0 / W, 0.0, x1 / W, 1.0]
        bx = [max(0.0, bx[0] - .06), max(0.0, bx[1] - .06), min(1.0, bx[2] + .06), min(1.0, bx[3] + .06)]
        out[i] = (figs[i]["who"], [round(float(v), 3) for v in bx])
    return [out[i] for i in range(len(figs))] + [(w, box(cm)) for w in cr]


def _person_key(n: str) -> str:
    """One key per person, whatever the spelling ('Dr. Watson' = 'Doctor Watson' = folder doctor_watson)."""
    sys.path.insert(0, str(HERE))
    import registry as RG
    return RG.person_key(n)


def _lora(book: Path, who: str):
    """(LoRA path, trigger) of the person under any spelling (registry.lora), or (None, None)."""
    lo, trig, _ = _reg(book).lora(who)
    return lo, trig


_SAID: set = set()


def _say_once(msg: str, log=print) -> None:
    if msg not in _SAID:
        _SAID.add(msg)
        log(msg)


def _look(book: Path, who: str, log=print) -> str:
    """The person's one look (registry.look), said LOUDLY once when there is none (10-27: looks.get(w, '') drew
    every other spelling of a person from no words at all)."""
    lk = _reg(book).look(who)
    if not lk and who:
        _say_once(f"!!! no look for {who!r} in look.json: drawn from the shot's words alone", log)
    return lk


def period_line(book: Path, what: str = "every object, tool and garment") -> str:
    """The drawing prompts' period sentence from <book>/period.txt (10-26: "early medieval Northern Europe, around
    1000 AD" was hardcoded from the knight book, so Acts (Roman Judea, around 30 AD) was drawn as medieval)."""
    f = Path(book) / "period.txt"
    per = f.read_text().strip().rstrip(".") if f.exists() else ""
    if not per:                                # (10-27: a missing period was silent: modern objects in every key)
        _say_once(f"!!! no period for {Path(book).name} ({f} missing): prompts carry no time period "
                  f"(`bookcaster period` sets it)")
    return f"Set in {per}: {what} as it was then, nothing from a later time." if per else ""


def _place_of(book: Path, stem: str, mid: str) -> str:
    """The moment's place: its own, else the scene it sits in (a talk moment has none of its own)."""
    vd, _, _ = paths(book, stem)
    here = ""
    for e in sorted(json.loads((vd / "flow3d" / "plan.json").read_text())["entries"], key=lambda e: e["t"]):
        here = e.get("where") or here
        if e["id"] == mid:
            return here
    return here


def _prompt(book: Path, stem: str, mid: str, who: list, setting_line: bool = False) -> str:
    """The keyframe's words: the moment's own prompt (framing, action, light), each person's look, the period and the
    book's art style. Never a description of the render (there is none to describe)."""
    vd, _, _ = paths(book, stem)
    f = _frame(book, stem, mid)
    style = style_of(book, stem)
    base = f.get("prompt") or (f"Medium close-up of {who[0]} speaking." if who else "")
    import re
    dress = _dress(book, stem, mid, who)
    # the outfit in force at the moment replaces the look's own clothes (10-23: Edric fought the dragon in his
    # leather jerkin - the keys used the base look and never the wardrobe's star-iron armor)
    _, _, md_ = paths(book, stem)
    view = {w: _view(md_, mid, w)[0] for w in who}
    # a fallen / slain figure is drawn so (10-23: the dragon, dead at 6:11, stood alive beside Edric at 6:16)
    down = re.compile(r"\b(lies|lying) (dead|still|motionless|unmoving|flat)\b|\bunmoving\b|\blifeless\b|\bdead\b|"
                      r"\bcollapses?\b|\bslumps?\b", re.I)
    for w in who:
        if any(str(b_).lower().startswith(w.lower() + ":") and down.search(str(b_)) for b_ in f.get("beats") or []):
            view[w] = ", ".join(x for x in (view.get(w), "lying collapsed and motionless on the ground") if x)
    looks = {w: _appearance(_look(book, w)) for w in who}
    people = " ".join(f"{_lora(book, w)[1] or ''} {w}{', ' + view[w] if view.get(w) else ''}: "
                      f"{re.split(r',?\s*\bwearing\b', looks.get(w, ''))[0] + ', wearing ' + dress[w] if w in dress else looks.get(w, '')}.".strip()
                      for w in who).replace("..", ".")
    place = _reg(book).place(_place_of(book, stem, mid) or f.get("where") or "") or ""
    # the SETTING always: a talk moment's words were only "Medium close-up of Hulda speaking." and 7 of 8 such keys
    # came out as white-background character sheets in every seed (10-23 overnight)
    setting = "" if not setting_line else (f" The scene takes place in {place}, which fills the whole picture behind them to its edges: a "
               f"single painted scene, never a character sheet, never a plain or white background.") if place else \
        " A single painted scene filling the picture to its edges, never a plain or white background."
    # (only on the retry of a moment whose seeds were all sheets: the first draw keeps its cached keys)
    return (f"{base}{setting} {people} {period_line(book)} {style} No text.").replace("  ", " ")


POSTURE = re.compile(r"\b(stand(s|ing)?|sit(s|ting)?|seated|kneel\w*|lying|lies|stance|posture|stooped|upright|"
                     r"wings? (folded|spread|open)|tail (extended|curled|raised)|on (four|two|all) (legs|fours))\b", re.I)


def _appearance(desc: str) -> str:
    """A look without its posture clauses (10-23: "Vorgrath: ... standing on four legs, tail extended, wings folded"
    in every key overrode "lying collapsed" - the slain dragon was drawn standing): what they look like, not how
    they stand; the shot says that."""
    parts = [x.strip() for x in re.split(r",", (desc or "").rstrip("."))]
    # ... and a stance clause's body part stays (11-01: "wings folded" dropped whole: every Vorgrath key since 10-23
    # was a wingless lizard, and his LoRA at 0.5 could not add what the words left out)
    keep = [x if not POSTURE.search(x) else "great wings" if re.search(r"\bwings?\b", x, re.I)
            else "a long tail" if re.search(r"\btail\b", x, re.I) else "" for x in parts if x]
    keep = [x for x in keep if x]
    # ... but the BODY PLAN stays (10-23: with "on four legs" stripped, the fight keys drew the dragon upright on two
    # legs like a man): four-legged is what it is, not how it stands
    legs = re.search(r"\bon (four|all four|4) legs\b|\bon all fours\b", desc or "", re.I)
    if legs and keep and "four-legged" not in keep[0]:
        keep[0] = re.sub(r"^(an?|the)\s+", lambda m_: m_.group(0) + "four-legged ", keep[0], count=1) \
            if re.match(r"^(an?|the)\s+", keep[0]) else "four-legged " + keep[0]
    return ", ".join(keep) + "." if keep else desc


def _dress(book: Path, stem: str, mid: str, who: list) -> dict:
    """{who: the dress the plan's wardrobe has them in at this moment} (video.outfit at the moment's paragraph)."""
    from tts_reader import video
    vd, _, _ = paths(book, stem)
    plan = json.loads((vd / "plan.json").read_text())
    if not plan.get("wardrobe"):
        return {}
    e = next((x for x in json.loads((vd / "flow3d" / "plan.json").read_text())["entries"] if x["id"] == mid), None)
    if e is None and re.fullmatch(r"[a-z]+-(?:\d\dm\d\ds-)?\d{6}", mid):    # a director's own moment: its time is in its id (tenths)
        e = {"t": int(mid.rsplit("-", 1)[1]) / 10}
    tl = book / "work" / f"{stem}.timeline.json"
    if e is None or not tl.exists():
        return {}
    paras = video.paragraphs((book / "chapters" / f"{stem}.txt").read_text())
    starts = video.paragraph_times(paras, json.loads(tl.read_text()))
    para = video.para_at(starts, float(e["t"]))
    # an outfit worn from where the story MAKES it is worn once it's done, as characters.outfit_at has it (11-01:
    # Edric forged in his finished star-iron armour - the keys read the wardrobe row unshifted)
    import characters as CH
    plan = CH.made_outfits_shift(plan, paras)
    R = _reg(book)                              # a wardrobe row under any spelling of the person (10-27)
    return {w: d for w in who if (row := video.outfit(plan, w, para, R)) and (d := video.worn_only(row["dress"]))}


def _bg_init(md: Path, mid: str, hero: Path, strength_bg: float) -> dict | None:
    """Start the key from the PLACE's hero image (10-23: with no set behind them - an empty depth map - Hulda's and
    the King's keys came out as white character sheets in every seed, with or without their LoRAs): the people's
    region (staged people mask, grown) drawn fresh, the rest repainted at `strength_bg` so the place stays."""
    import numpy as np
    from PIL import Image, ImageFilter
    if not Path(hero).exists():
        return None
    d = md / mid
    Image.open(hero).convert("RGB").resize((W, H)).save(d / "init_bg.png")
    dp = d / "maps" / "frames" / "depth_0000_people.npy"
    m = (np.load(dp) < 1e9) if dp.exists() else np.zeros((H, W), bool)
    if not m.any():                         # no people mask: the pose map's drawn pixels
        a = np.asarray(Image.open(d / "maps" / "frames" / "p_0000.png").convert("L"))
        m = a > 20
    mk = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(61)).filter(ImageFilter.GaussianBlur(12))
    mk.save(d / "init_mask.png")
    return {"image": str(d / "init_bg.png"), "strength": 1.0, "mask": str(d / "init_mask.png"),
            "strength_bg": strength_bg}


def keys(book: Path, stem: str, ids: list, hero: Path, card: str = "05", seeds=(7, 11), pose_s=0.85, depth_s=0.6,
         end=0.5, log=print, setting_line: bool = False, lora_w: float = 0.9, bg_init: float = 0.0, at: int = 0,
         refs: dict | None = None, fights: dict | None = None) -> dict:
    """One keyframe per moment (best of the seeds kept as key_<seed>.png; key.png = the first, to be judged):
    pose + set-depth control ending at `end` of the steps, the people's LoRAs, the hero image as the only reference.
    at > 0: a PIN key (pin_<at>_<seed>.png) from the staging's frame `at`, refs[mid] (the moment's own key) first."""
    import numpy as np
    vd, sd, md = paths(book, stem)
    jobs = []
    for mid in ids:
        fr = md / mid / "maps" / "frames"
        lay = (fights or {}).get(mid)
        if lay:                             # a fight: the opening layout alone, no pose or depth (FIGHT_RULES)
            for s in seeds:
                jobs.append({"id": f"{mid}_key_{s}", "seed": s, "size": [W, H], "path": str(md / mid / f"key_{s}.png"),
                             "prompt": _fight_prompt(book, stem, mid, lay), "refs": [str(hero)],
                             "loras": [[_lora(book, w)[0], lora_w * 0.7, bx] for w, bx in
                                       (((lay["left"], LEFT_BOX), (lay["right"], RIGHT_BOX)) if lay.get("right")
                                        else ((lay["left"], [0.0, 0.0, 1.0, 1.0]),)) if _lora(book, w)[0]]})
            continue
        if not (fr / f"p_{at:04d}.png").exists():
            log(f"{mid}: not staged" + (f" at frame {at}" if at else ""))
            continue
        try:
            shot = _in_shot(md, mid, at) or [(w, None) for w in _who(book, stem, mid)]
        except Exception as exc:            # noqa: BLE001 - one moment's regions never stop the keys
            log(f"!!! {mid}: people regions failed ({exc}): the shot's cast, whole picture")
            shot = [(w, None) for w in _who(book, stem, mid)]
        who = [w for w, _ in shot]
        loras = [[_lora(book, w)[0], round(lora_w * _view(md, mid, w)[1] * _overlap_factor(shot, w), 2)]
                 + ([bx] if bx else []) for w, bx in shot if _lora(book, w)[0]] \
            if lora_w > 0 else []
        init = _bg_init(md, mid, hero, bg_init) if bg_init > 0 else None
        dimg = fr / f"dbg_{at:04d}.png"
        try:                                # a close framing: the people's depth too, so they're drawn their staged
            pm = np.load(fr / f"depth_{at:04d}_people.npy") < 1e9     # size (10-23: Maren's close-up drawn as a
            if pm.mean() > CLOSE_FRAC or at:    # small figure lying on the rocks - the set-only depth and a head-
                dimg = fr / f"d_{at:04d}.png"   # and-shoulders skeleton never fixed her scale); a pin: both bodies
        except OSError:
            pass
        for s in seeds:
            name = f"pin_{at}_{s}" if at else f"key_{s}"
            jobs.append({"id": f"{mid}_{name}", "seed": s, "size": [W, H], "path": str(md / mid / f"{name}.png"),
                         **({"init": init} if init else {}),
                         "prompt": _prompt(book, stem, mid, who, setting_line),
                         "refs": ([str((refs or {})[mid])] if (refs or {}).get(mid) else []) + [str(hero)],
                         "control": [{"image": str(fr / f"p_{at:04d}.png"), "type": "pose", "scale": pose_s, "end": end}] +
                                    ([{"image": str(dimg), "type": "depth", "scale": depth_s, "end": end}]
                                     if depth_s > 0 else []),
                         "loras": loras})
    def sig_of(j, content=True):            # content: the control maps' and refs' bytes (a re-staged moment keeps
        d = {k: j.get(k) for k in ("prompt", "refs", "control", "loras", "size")}       # the same file paths)
        if j.get("init"):                   # (only when used: the other keys keep their signatures)
            d["init"] = j["init"]
        if content:
            import hashlib
            d["_bytes"] = [hashlib.sha1(Path(x).read_bytes()).hexdigest()[:12] if Path(x).exists() else None
                           for x in [c["image"] for c in j.get("control") or []] + list(j.get("refs") or [])]
        return _h(json.dumps(d, sort_keys=True))
    last = {}                               # keys drawn before .sig files existed: the job files that drew them
    for f in md.glob("keys_*.json"):
        try:
            last.update({j["path"]: sig_of(j, False) for j in json.loads(f.read_text())})
        except (ValueError, KeyError, TypeError):
            pass
    for j in jobs:                          # a key drawn from other words/LoRAs/maps is stale: drawn again
        sig = sig_of(j)
        sp = Path(j["path"]).with_suffix(".sig")
        old = sp.read_text() if sp.exists() else (sig if last.get(j["path"]) == sig_of(j, False) else None)
        if Path(j["path"]).exists() and old != sig:
            Path(j["path"]).unlink()
        sp.write_text(sig)
    if not jobs:
        return {}
    jp = md / f"keys_{card}.json"           # one per card: both cards draw at once (10-22: a shared name lost half)
    jp.write_text(json.dumps(jobs, indent=1))
    hip = subprocess.run([os.path.expanduser("~/bin/hip-index"), card], capture_output=True, text=True).stdout.strip()
    r = subprocess.run(["flock", f"/tmp/gpu-{card}.lock", QWEN_PY, "-I", str(HERE / "qwen_image.py"), str(jp), str(md)],
                       capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
    (md / f"keys_{card}.log").write_text(r.stdout[-8000:] + r.stderr[-8000:])
    out = {j["id"]: Path(j["path"]).exists() for j in jobs}
    if not all(out.values()):
        log(f"!!! keys on card {card}: {sum(out.values())}/{len(out)} drawn, rc {r.returncode} (see keys_{card}.log)")
    return out


def entries(book: Path, stem: str, t0: float, t1: float) -> list:
    """The planned moments whose start is in [t0, t1) (flow3d/plan.json), in time order."""
    vd, _, _ = paths(book, stem)
    P = json.loads((vd / "flow3d" / "plan.json").read_text())
    return sorted((e for e in P["entries"] if t0 <= float(e["t"]) < t1), key=lambda e: e["t"])


def _seed_keys(d: Path) -> list:
    """key_<seed>.png in SEED order (10-22: a name sort put key_11 before key_7)."""
    return sorted(d.glob("key_*.png"), key=lambda k: int(k.stem.split("_")[1]) if k.stem.split("_")[1].isdigit() else 1e9)


def _blank_edges(k: Path, thr: float = 0.25) -> bool:
    """A picture whose border is mostly near-white or flat = a sheet/panel, not a scene (10-22: three forge keys came
    out as white-background character-sheet poses that the text question passed)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(k).convert("L"), dtype=np.float32) > 215
    b = max(4, a.shape[1] // 40)
    # 10-30: the top edge left out of the mean - a pale overcast sky failed every seed of the Stoke Moran exteriors
    # (27 Speckled Band keys, none a sheet); a sheet still shows white on 3 sides, a side text panel on the others
    sides = [a[:, :b], a[:, -b:], a[-b:]]
    return (float(np.concatenate([s.ravel() for s in sides]).mean()) > thr
            or sum(float(s.mean()) > 0.5 for s in sides + [a[:b]]) >= 3)


def pick_keys(book: Path, stem: str, ids: list, log=print, fights: dict | None = None) -> dict:
    """key.png = the first seed whose picture has NO text or page layout (10-22: the gothic style drew 'book page'
    panels with fake text into 3 of 4 seed-11 keys); the vision judge on the iGPU."""
    sys.path.insert(0, str(HERE))
    import re
    import characters as CH
    import flow3d as F
    _, _, md = paths(book, stem)
    _, cx = F._ctx(book, stem)
    # WHO must be there (10-23: the killing thrust's key had no Edric, the dragon's death had no Maren): each
    # in-shot figure by the first words of its look ("a man of about 18", "a massive dragon"), and the people count
    need = {}
    for m in ids:
        lay = (fights or {}).get(m)
        try:
            shot = [w for w in (lay["left"], lay.get("right")) if w] if lay else [w for w, _ in _in_shot(md, m)]
        except Exception:                   # noqa: BLE001 - no regions: no presence questions
            shot = []
        q = {}
        for i, w in enumerate(shot):
            first = re.split(r",| with | wearing ", (cx.look(w, 0) or "").strip(), maxsplit=1)[0].strip().rstrip(".")
            first = re.sub(r"\s+of about \d+.*$", "", first)     # "a man of about 18": an armoured Edric was "no"
            if first:
                q[f"who{i}"] = f"Answer yes or no: is there {first} in this picture?"
                # and turned the way the staging has them (10-23: Edric drawn facing the viewer in shot after shot
                # while the staging had his back or side to the camera, facing the dragon)
                vw = _view(md, m, w)[0] if w not in _creatures(md, m) and not lay else ""
                if vw.startswith("seen from behind"):
                    q[f"face{i}"] = f"Answer yes or no: is {first} turned away from the viewer, his or her back to us?"
                elif vw.startswith("seen in profile"):
                    q[f"face{i}"] = f"Answer yes or no: is {first} seen from the side, in profile?"
        if lay and len(shot) == 2:          # a fight's opening: facing each other, apart, both whole
            a_, b_ = (re.sub(r"\s+of about \d+.*$", "", re.split(r",| with | wearing ", (cx.look(w, 0) or w).strip(),
                                                                 maxsplit=1)[0].strip().rstrip(".")) for w in shot)
            # (not every creature moment is a duel - a carry, a flight, a death: the director's own layout is the test)
            q["face_f"] = f"Answer yes or no: does this picture show this: {lay['layout']}"
            q["face_apart"] = (f"Answer yes or no: are {a_} and {b_} clearly separate, with a gap between them and "
                               f"neither body merged into the other?")
        for i_, w in enumerate(shot if lay else []):
            if "four-legged" in _appearance(cx.look(w, 0) or ""):   # (10-23: the dragon drawn upright like a man)
                q[f"face_legs{i_}"] = f"Answer yes or no: is the {('dragon' if 'dragon' in (cx.look(w, 0) or '') else 'creature')} on four legs, not upright on two legs like a person?"
        hum = [w for w in shot if w not in _creatures(md, m)]
        if hum:
            q["count"] = "How many people are in this picture? Answer with a number."
        need[m] = (q, len(hum))
    items = [{"id": f"{m}|{k.stem}", "image": str(k), "ask": {"text": "Answer yes or no: does this picture contain any "
              "written text, letters, a page, a panel or a border?", **need[m][0]}}
             for m in ids for k in _seed_keys(md / m)]
    ans = CH.vl_ask(items) if items else {}

    def present(m, k):
        a_ = ans.get(f"{m}|{k.stem}") or {}
        q, n = need[m]
        if any(not CH._yes(a_.get(x)) for x in q if x.startswith(("who", "face"))):
            return False
        c = re.search(r"\d+", str(a_.get("count") or ""))
        # too many counts too (10-23: the 5:26 fight key drew Edric twice, standing and lying)
        return not (n and c and (int(c.group(0)) < n or (fights or {}).get(m) and int(c.group(0)) > n))
    out, dirty = {}, []
    for m in ids:
        ks = _seed_keys(md / m)
        clean = [k for k in ks if not CH._yes((ans.get(f"{m}|{k.stem}") or {}).get("text")) and not _blank_edges(k)
                 and present(m, k)]
        best = (clean or ks or [None])[0]
        kp = md / m / "key.png"
        if best is not None and not (kp.exists() and filecmp.cmp(best, kp, shallow=False)):
            shutil.copy(best, kp)                                   # a NEW key.png makes its clip stale (ltx)
        out[m] = best.name if best else None
        if ks and not clean:
            dirty.append(m)
        log(f"{m}: key = {out[m]} ({len(ks) - len(clean)} of {len(ks)} seeds failed: text, sheet, someone missing "
            f"or turned the wrong way)")
    pick_keys.dirty = dirty                 # moments where NO seed passed (the caller draws more seeds)
    return out


MOTION_RULES = """You write the prompt for an image-to-video model. It animates ONE still picture (already drawn,
already in the right style and place) for {sec:.1f} seconds. Write 2 to 4 plain sentences:
- who is in the picture (only: {people}), in a few words each: age, hair, main garment;
- ONE continuous physical action they perform from start to finish within the {sec:.1f} seconds, as body mechanics:
  which hand, which direction, how far, how often (a rhythm if it repeats), what the tool or object does. {source}
- what each hand holds, exactly as staged (given below): say it, never change or drop it;
- the visible physical effect (sparks, coals flaring, metal glowing), if any;
- end with: The camera holds still.
Never: story or feelings, what something means, what happened before, cracks or changes over time, people who are
not in the picture, camera moves, style or period words, text, sitting, kneeling, crouching or bending the knees unless
the moment says the person sits or kneels. Present tense. No quotes.
Two good prompts:
1. A young blacksmith in a brown leather jerkin stands at an iron anvil inside a dim stone forge lit by the fire of
the hearth. He raises a heavy hammer high over his head with his right hand and brings it down hard on the glowing
metal bar on the anvil, sparks bursting from the strike, his left hand steadying the bar. Firelight flickers on the
stone walls. The camera holds still.
2. A young blacksmith in a brown leather jerkin works a large leather bellows beside a stone hearth inside a dim
forge, pushing the long handle down and pulling it up again in a steady rhythm; with each push the coals in the
hearth flare bright orange and sparks rise. Firelight flickers on the stone walls. The camera holds still.

The moment: {moment}
The director's beats (the mechanics of the action, in order; follow them): {beats}
The picture: {picture}
Place: {place}
{hands}{swing}CROSS-CHECK first: compare the director's beats with the staged motion and the staged hands. {follow}
Reply with JSON only: {{"prompt": "...", "conflicts": ["...", ...]}}"""


def _brief(desc: str) -> str:
    """A look in a few words for LTX (the key already shows it): age, hair, the main garment. 10-22: the full look
    ("stooped posture, plain footwear, dark eyes...") buried the talk prompt's one job, the moving mouth."""
    import re
    parts = [x.strip() for x in re.split(r",|;", desc.rstrip(".")) if x.strip()]
    age = next((x for x in parts if re.search(r"\b(man|woman|boy|girl|person|of about)\b", x, re.I)), "")
    hair = next((x for x in parts if re.search(r"\bhair\b", x, re.I)), "")
    wear = next((x for x in parts if re.search(r"\b(wearing|in a|dressed)\b", x, re.I)), "")
    return ", ".join(x for x in (age, hair, wear) if x) or (parts[0] if parts else "")


def _hands(md: Path, mid: str) -> str:
    """What each person's hands hold in the staged shot ("Edric: right hand the hammer, left hand the sword blank by
    its hilt"), from the motion's shot.json (the approved recipe's work piece included)."""
    p = md / mid / "motion" / "shot.json"
    if not p.exists():
        return ""
    sh = json.loads(p.read_text())
    out = []
    for f in sh.get("figures") or []:
        h = f.get("holds") or {}
        if not h:
            continue
        def nm(x):
            b, _, st = str(x).partition("__")
            return f"the {b.replace('_', ' ')}" + (f" ({st.replace('_', ' ')})" if st else "")
        parts = [f"{side} hand {nm(x)}" + (" by its hilt, the blade lying on the anvil" if "sword" in str(x) and
                 (sh.get("recipe") or {}).get("offhand") == x else "") for side, x in sorted(h.items(), key=lambda kv: kv[0] != "right")]
        out.append(f"{f['who']}: " + ", ".join(parts))
    return "; ".join(out)


def _smooth(x, k=5):
    import numpy as np
    k = max(1, k)
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def _turns(x, prom):
    """indices of the alternating extrema of x whose swing between them exceeds prom (start and end included)."""
    pts, i0, d = [0], 0, 0
    lo = hi = 0
    for i in range(1, len(x)):
        if d >= 0 and x[i] > x[hi]:
            hi = i
        if d <= 0 and x[i] < x[lo]:
            lo = i
        if d >= 0 and x[hi] - x[i] > prom and hi != pts[-1]:
            pts.append(hi); d = -1; lo = i
        elif d <= 0 and x[i] - x[lo] > prom and lo != pts[-1]:
            pts.append(lo); d = 1; hi = i
    if pts[-1] != len(x) - 1:
        pts.append(len(x) - 1)
    return pts


def staged_action(md: Path, mid: str, landings: list | None = None, tool: str = "hammer") -> str:
    """What the STAGED motion does, in plain words, for the LTX prompt writer (user 10-22: describe the action so LTX
    knows what to animate; generic, from the solved joints): each hand's path in body terms (waist, chest, head,
    above the head) with times, repeated strokes, a worked prop's cycle, a held tool landing, walking, turning,
    crouching; "stays nearly still" when nothing moves."""
    import numpy as np
    FPS = 24.0
    mo = md / mid / "motion"
    shp = mo / "shot.json"
    sh = json.loads(shp.read_text()) if shp.exists() else {}
    figs = {f["who"]: f for f in sh.get("figures") or []}
    out = []
    for jp in sorted(mo.glob("joints_*.npz")):
        who = jp.stem[len("joints_"):]
        z = np.load(jp)
        names = [str(n) for n in z["names"]]
        P = z["pos"]
        J = lambda k: P[:, names.index(k)]                       # noqa: E731
        n = len(P)
        holds = (figs.get(who) or {}).get("holds") or {}
        top, neck, mid_, pel = J("head_top")[:, 2], J("neck")[:, 2], J("spine_mid")[:, 2], J("pelvis")[:, 2]

        def level(i, zz):
            if zz > top[i] + 0.02:
                return "above the head"
            if zz > neck[i]:
                return "at head height"
            if zz > mid_[i]:
                return "at chest height"
            if zz > pel[i] - 0.12:
                return "at waist height"
            return "low, by the thigh"
        # the body's forward axis at each frame (perpendicular to the hips, horizontal)
        hip = J("l_hip") - J("r_hip")
        fwd = np.stack([hip[:, 1], -hip[:, 0], np.zeros(n)], 1)     # (10-23: was reversed - the toes point this way)
        fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
        lines = []
        ops = [o for o in sh.get("operate") or [] if o.get("who") == who]
        busy = {h for o in ops for h in o.get("hands") or []}
        for o in ops:                      # a worked prop moves by its own cycle (the hands follow it by IK)
            pr = next((p for p in (sh.get("set") or {}).get("props", []) if (p.get("prop") or p.get("name")) == o["prop"]), None)
            per = None
            try:
                import blockout_props as PR
                _, ent = PR.set_entry(pr) if pr else (None, {})
                jd = next((j for j in ent.get("joints") or [] if j.get("name") == o.get("joint")), {})
                per = float((jd.get("cycle") or {}).get("period_s", 1.4)) if jd.get("cycle") else None
            except Exception:                                      # noqa: BLE001 (a summary, never a failure)
                per = None
            hands_ = " and ".join(o.get("hands") or []) or "both"
            strokes = f": {max(1, round(n / FPS / per))} full strokes, one every {per:.1f} s, in a steady rhythm" if per else \
                " in a steady rhythm"
            lines.append(f"works the {o['prop']} ({o.get('joint', 'its moving part')}) with the {hands_} "
                         f"hand{'s' if ' and ' in hands_ else ''}, pushing it and letting it return{strokes}")
        for side, hn in (("right", "r_hand"), ("left", "l_hand")):
            if side in busy:
                continue
            h = J(hn)
            zz = _smooth(h[:, 2])
            ff = _smooth(np.einsum("ij,ij->i", h - J("pelvis"), fwd))
            item = holds.get(side)
            what = f"the {side} hand" + (f" (holding the {str(item).split('__')[0].replace('_', ' ')})" if item else "")
            rz, rf = np.ptp(zz), np.ptp(ff)
            if max(rz, rf) < 0.12:
                continue
            axis, x = ("up", zz) if rz >= rf * 0.8 else ("forward", ff)
            if axis == "forward" and rf < 0.25:   # arm drift, not an action (10-22: "pulls back then reaches
                continue                          # forward before pulling back again" for a man standing in armour)
            pts = _turns(x, max(0.12, 0.35 * np.ptp(x)))
            segs = [(a, b) for a, b in zip(pts, pts[1:]) if abs(x[b] - x[a]) > (0.1 if axis == "up" else 0.2)]
            if len(segs) >= 4:                                     # a repeated stroke
                k = len(segs) // 2
                lo_, hi_ = int(np.argmin(x)), int(np.argmax(x))
                if axis == "up":
                    lines.append(f"{what} moves down and up {k} times in a steady rhythm, between "
                                 f"{level(lo_, zz[lo_])} and {level(hi_, zz[hi_])}")
                else:
                    lines.append(f"{what} pushes forward and pulls back {k} times in a steady rhythm")
                continue
            for a, b in segs:
                t = f"({a / FPS:.1f}-{b / FPS:.1f} s)"
                if axis == "up":
                    la, lb = level(a, zz[a]), level(b, zz[b])
                    if la == lb:
                        continue
                    land = next((f_ for f_ in landings or [] if a < f_ <= b), None) \
                        if str(item or "").split("__")[0] == tool else None    # only the striking tool's hand lands
                    if x[b] > x[a]:
                        lines.append(f"{what} raises from {la} to {lb} {t}")
                    elif land is not None:
                        lines.append(f"{what} swings down hard from {la} and the {str(item).split('__')[0]} lands on "
                                     f"the work at {land / FPS:.1f} s, then rests there")
                    else:
                        lines.append(f"{what} lowers from {la} to {lb} {t}")
                else:
                    verb = "reaches forward" if x[b] > x[a] else "pulls back"
                    lines.append(f"{what} {verb} {t}")
        root = J("pelvis")[:, :2]
        dist = float(np.linalg.norm(root[-1] - root[0]))
        if dist > 0.35:
            lines.append(f"walks about {dist:.1f} m")
        yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0])))
        if abs(yaw[-1] - yaw[0]) > 70:
            lines.append(f"turns about {abs(yaw[-1] - yaw[0]):.0f} degrees to the {'left' if yaw[-1] > yaw[0] else 'right'}")
        drop = float(pel[0] - pel.min())
        if drop > 0.25:
            lines.append("lowers the body (bends the knees deeply)")
        if not lines:
            lines.append("stays nearly still: breathing and small movements only")
        out.append(f"{who}: " + "; ".join(lines))
    return "\n".join(out)


def motion_prompts(book: Path, stem: str, es: list, log=print) -> dict:
    """The LTX words for every moment, written BEFORE the GPU steps by the language model (iGPU), in the 13-second
    preview's style: one continuous physical action, start to finish, few words of look, no story, no style blurb.
    10-22: bookcaster's 3D-flow prompts (state + story + full look + a style paragraph; "bends his knees" twice; "everyone
    else listens with their mouth closed") gave frozen lips, weak swings, and a knight who sat down. Cached by input."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    import re
    import flow3d as F
    R, cx = F._ctx(book, stem)
    _, _, md = paths(book, stem)
    cache_p = md / "prompts.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    here, out, asked = "", {}, False
    for e in es:
        here = e.get("where") or here
        shot = _in_shot(md, e["id"])
        people = [w for w, _ in shot] or ([e["speaker"]] if e.get("speaker") else [])
        looks = {w: _brief(cx.look(w, e["t"]) or "") for w in people}
        if e["route"] == "talk":
            who = e["speaker"]
            # the speaker under any spelling (10-27: a shot's "Dr. Watson" speaking as "Watson" had no look: "a person")
            who = next((w for w in people if cx.R.same(w, who)), who)
            looks.setdefault(who, _brief(cx.look(who, e["t"]) or ""))
            g = R.gender_of(looks.get(who, ""))
            pron = "her" if g == "woman" else "his" if g == "man" else "their"
            emo = f", sounding {e['emotion']}," if e.get("emotion") else ","
            others = [w for w in people if w != who]
            lst = (" " + " and ".join(others) + " listens with a closed mouth.") if others else ""
            out[e["id"]] = (f"Static camera, the framing does not change. {who}, {looks.get(who) or 'a person'}, inside "
                            f"{here or 'the room'}, speaks{emo} lips and jaw moving clearly in sync with {pron} voice.{lst} "
                            f"No text anywhere in the picture.")
            continue
        f = _frame(book, stem, e["id"])
        sec = float(e.get("frames") or 121) / 24.0
        hits = impacts(md / e["id"] / "maps" / "anchors.json")
        hands = _hands(md, e["id"])
        hands = f"Staged hands (keep exactly): {hands}\n" if hands else ""
        sa = staged_action(md, e["id"], hits)
        if sa and all("stays nearly still" in ln for ln in sa.splitlines()):
            sa = ""                         # no staged body motion: the moment's own action, small and in place
            swing = "The staging has no body motion here: take the action from the moment, small and in place.\n"
        else:
            swing = ""
        # which motion the clip follows: the pose video steers only a staged swing that lands (ltx()); every other clip
        # is drawn from the key and these words alone, so the DIRECTOR's beats are its action (10-23: the fight's
        # staging stood still, the prompts followed it - "Edric ... standing still" - and the fight had no action)
        driven = _guide(md, e["id"]) is not None
        directed = bool(f.get("beats"))
        if sa and (driven or not directed):
            swing = (f"The staged motion, from the 3D staging the clip follows (animate exactly this, in this order; "
                     f"write it as the action, in plain body words, without the times):\n{sa}\n")
        elif sa:
            swing = f"The staged motion (the 3D blocking: a reference only, the clip does not follow it):\n{sa}\n"
        elif directed:
            swing = ""                      # a still staging never shrinks a directed action to "small and in place"
        if driven or not directed:
            source = ("When the staged motion is given, that IS the action: describe every part of it (a raise before "
                      "a swing, each stroke), and nothing it does not do;")
            follow = ("The clip follows the STAGING, so where they disagree (a beat the staging does not do, a staged "
                      "move no beat names, a different item in a hand) the prompt describes the staging, and each "
                      "disagreement is listed in \"conflicts\" in a few words.")
        else:
            source = ("The director's beats ARE the action: describe every beat, in order, each played through to its "
                      "result (the blow lands, the fire hits); a creature's jaws, claws, tail, wings and fire too;")
            follow = ("The clip is drawn from the picture and these words alone, so the prompt follows the BEATS; the "
                      "hands keep what they hold as staged. Each disagreement between the beats and the staging (a beat "
                      "the staging does not do, a staged move no beat names) is listed in \"conflicts\" in a few words.")
        q = MOTION_RULES.format(sec=sec, people=", ".join(f"{w} ({looks[w]})" for w in people) or "nobody",
                                moment=" ".join(x for x in (f.get("motion"), (f.get("text") or "")[:400]) if x),
                                picture=f.get("prompt", ""), place=here, hands=hands, swing=swing,
                                source=source, follow=follow,
                                beats=" / ".join(f.get("beats") or []) or "(none)")
        k = _h(q)
        if cache.get(e["id"], {}).get("k") == k:
            out[e["id"]] = cache[e["id"]]["prompt"]
            continue
        if not asked:
            animate.igpu("strata", log)
            asked = True
        txt, bad = "", re.compile(r"\b(sit|sits|sitting|sat|kneel\w*|knees?|crouch\w*|squat\w*)\b", re.I)
        may_sit = re.search(r"\b(sit|sits|sitting|kneel\w*|knocked|falls?|lies|lying|rolls?|flat)\b",
                            " ".join([f.get("motion", ""), f.get("prompt", "")] + list(f.get("beats") or [])), re.I)
        conf = []
        for n in range(3):
            reply = _chat(BC.LLM_URL, BC.LLM_MODEL, q + ("\nYour last prompt bent the knees or sat the person down: "
                          "they stay standing." if n else ""), max_tokens=1500)
            m = re.search(r"\{.*\}", reply or "", re.S)
            try:
                js = json.loads(m.group(0)) if m else {}
                txt = str(js.get("prompt", "")).strip()
                conf = [str(c) for c in js.get("conflicts") or [] if str(c).strip()]
            except (ValueError, AttributeError):
                txt = ""
            if txt and (may_sit or not bad.search(txt)):
                break
        if txt and not may_sit and bad.search(txt):     # still bending: the clause goes (10-22: the knight sat down)
            t_ = re.sub(r"(,\s*|\s+and\s+)[^,.;]*?" + bad.pattern + r"[^,.;]*", "", txt, flags=re.I)
            if bad.search(t_):              # not a coordinated clause: the whole sentence goes
                t_ = " ".join(x for x in re.split(r"(?<=\.)\s+", t_) if not bad.search(x))
            txt = re.sub(r"\s+([,.])", r"\1", t_).strip()
        if not txt:                         # one moment never stops the night: bookcaster's own words, flagged
            txt = R.i2v_prompt(cx, e, True)
            log(f"!!! {e['id']}: the language model wrote no motion prompt: the plan's words used")
        if "camera holds still" not in txt.lower():
            txt += " The camera holds still."
        out[e["id"]] = txt
        cache[e["id"]] = {"k": k, "prompt": txt, "conflicts": conf}
        log(f"{e['id']} prompt: {txt}")
        for c in conf:
            log(f"  cross-check {e['id']}: direction vs staging: {c}")
    cache_p.write_text(json.dumps(cache, indent=1))
    if asked:                               # it holds ~90 GB of the shared RAM: LTX next needs it (10-22: 19 GB left)
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    return out


def _h(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def ltx(book: Path, stem: str, es: list, log=print, redo: dict | None = None, prompts: dict | None = None,
        fights: dict | None = None) -> dict:
    """The clips: TALK = audio-to-video from the key (start 0.8, NO end lock: 10-22, both ends pinned froze Hulda's
    mouth, motion 0.10 vs 1.09); ACTION whose motion is the point (contacts, recipe, a worked prop, a creature) = the
    IC-LoRA with the mannequin's POSE video; other action = the key + the prompt (bookcaster's i2v/talk prompt words)."""
    sys.path.insert(0, str(HERE))
    import flow3d as F
    R, cx = F._ctx(book, stem)
    _, sd, md = paths(book, stem)
    L = md / "ltx"
    (L / "out").mkdir(parents=True, exist_ok=True)
    (L / "emb").mkdir(exist_ok=True)
    (L / "audio").mkdir(exist_ok=True)
    shutil.copy(HERE / "ltx_runner" / "ltx_split.py", L / "ltx_split.py")
    neg = subprocess.run([str(R.LTX_PY), "-c", "import sys; sys.path.insert(0, '/home/jrouth/src/LTX-2/packages/ltx-pipelines/src');"
                          "from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT as N; print(N)"],
                         capture_output=True, text=True).stdout.strip() + R.NEG_EXTRA
    enc_t, enc_a, talk, act, enc_f, fight = [], [], [], [], [], []
    for e in es:
        key = md / e["id"] / "key.png"
        if not key.exists():
            log(f"{e['id']}: no key")
            continue
        for c in (L / "out").glob(f"r_{e['id']}*.mp4"):      # a clip older than its key (or redrawn) is drawn again
            newest = max([key.stat().st_mtime] + [k.stat().st_mtime for _, k in pins(md, e["id"])] * (GUIDE == "pins"))
            if c.stat().st_mtime < newest or e["id"] in (redo or {}):
                c.unlink()
        if e["route"] == "talk":
            pad = L / "audio" / f"{e['id']}_pad.wav"
            n_all = max(s_ + n_ for s_, n_ in e["chunks"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", e["wav"], "-ac", "2", "-ar", "48000", "-af",
                            f"apad=whole_dur={n_all / 24:.4f}", str(pad)], check=True)
            p_ = (prompts or {}).get(e["id"]) or R.talk_prompt(cx, e)
            emb = f"t_{e['id']}_{_h(p_)}"
            enc_t += [{"name": emb, "prompt": p_}, {"name": emb + "__neg", "prompt": neg}]
            for k, (st, n) in enumerate(e["chunks"]):
                talk.append({"name": f"r_{e['id']}" if len(e["chunks"]) == 1 else f"r_{e['id']}_c{k}", "emb": emb,
                             "prompt": p_, "width": W, "height": H, "fps": 24, "seed": 42, "frames": n, "audio": str(pad),
                             "audio_start": round(st / 24, 4), "line": e["id"], "speaker": e.get("speaker"),
                             "speech_s": e.get("speech_s"), "chunk": [k, len(e["chunks"]), st, n],
                             "creature": e.get("creature", False),
                             "images": [{"path": str(key), "frame": 0, "strength": (redo or {}).get(e["id"], 0.8)}]})
        elif (fights or {}).get(e["id"]):   # the fight route: the action sentence, two-stage, no guide
            p_ = fights[e["id"]]["action"]
            emb = f"f_{e['id']}_{_h(p_ + FIGHT_NEG)}"
            enc_f += [{"name": emb, "prompt": p_}, {"name": emb + "__neg", "prompt": neg + FIGHT_NEG}]
            fight.append({"name": f"r_{e['id']}", "emb": emb, "prompt": p_, "width": W, "height": H, "fps": 24,
                          "seed": 42 + 1000 * (e["id"] in (redo or {})), "frames": e["frames"], "line": e["id"],
                          "neg_extra": FIGHT_NEG, "route": "fight",
                          "images": [{"path": str(key), "frame": 0, "strength": 1.0}]})
        else:
            p_ = (prompts or {}).get(e["id"]) or R.i2v_prompt(cx, e, True)
            emb = f"a_{e['id']}_{_h(p_)}"
            enc_a.append({"name": emb, "prompt": p_})
            j = {"name": f"r_{e['id']}", "emb": emb, "prompt": p_, "width": W, "height": H, "fps": 24,
                 "seed": 7 + 1000 * (e["id"] in (redo or {})),
                 "frames": e["frames"], "line": e["id"], "images": [{"path": str(key), "frame": 0, "strength": 1.0}]}
            # the staged motion steers only an action that LANDS: a hammer on the work, a blow / block / claw on
            # a figure (10-22: the bellows on the pose video pumped worse than the preview's prompt-only clip)
            g = _guide(md, e["id"])
            pn = pins(md, e["id"]) if GUIDE == "pins" else []
            if g and pn:                    # the staging as drawn keys at the hit and the end, no control video
                j["images"] += [{"path": str(k), "frame": min(f, e["frames"] - 1), "strength": PIN_S} for f, k in pn]
                j["route"] = "pins"
            elif g:
                j["control"] = {"path": str(g), "attn": GUIDE_ATTN.get(GUIDE, 0.35)}
                j["route"] = "ic-" + g.stem
            else:
                j["route"] = "i2v"
            act.append(j)
    for j in talk + act + fight:            # a clip made from other words or another control is drawn again
        rj = L / "out" / f"{j['name']}.json"
        if rj.exists():
            old = json.loads(rj.read_text())
            if old.get("prompt") != j["prompt"] or old.get("control") != j.get("control") or \
                    [(i["path"], i["frame"]) for i in old.get("images") or []][1:] != \
                    [(i["path"], i["frame"]) for i in j["images"]][1:]:
                (L / "out" / f"{j['name']}.mp4").unlink(missing_ok=True)
    for nm, x in (("enc_talk", enc_t), ("enc_act", enc_a), ("talk_jobs", talk), ("act_jobs", act),
                  ("enc_fight", enc_f), ("fight_jobs", fight)):
        (L / f"{nm}.json").write_text(json.dumps(x, indent=1))
    # the 3D flow's pipelined runner (10-22: this flow first ran its own one-clip-at-a-time call): ltx_split_next.py
    # --pp --lanes 2 --pp-shared-a for gen, the per-route split pp_fit fits from this stretch's own timing, the gap
    # runner, the RAM floor and the kernel-log check, all from remake_chapter.ltx_step
    from types import SimpleNamespace
    os.environ["REMAKE_LTX_SPLIT_TALK"] = str(F._pp_split(md, "a2vid", 25))
    os.environ["REMAKE_LTX_SPLIT_ACT"] = str(F._pp_split(md, "ic", 17))
    R.PP = True                             # lanes/stagger/pool/decode-b = R.LTX_PP_ARGS (10-26: one default set)
    shutil.copy(HERE / "ltx_runner" / "ltx_split_next.py", L / "ltx_split_next.py")
    cxl = SimpleNamespace(OUT=md)

    def step(args, name):
        rc = R.ltx_step(cxl, name, args)
        log(f"ltx {name}: rc {rc}")
        return rc

    def gen(kind, jobs, args):
        """gen holds until every clip exists: each hold stops starting clips after its budget (the card locks are
        shared with other GPU jobs); three holds in a row without a new clip = give up on the rest, logged."""
        left, stuck = None, 0
        while True:
            m = [j["name"] for j in jobs if not (L / "out" / f"{j['name']}.mp4").exists()]
            if not m:
                return
            if left is not None and len(m) >= left:
                stuck += 1
                if stuck >= 3:
                    log(f"!!! ltx {kind}: no progress in 3 holds, {len(m)} clip(s) not made: {m}")
                    return
            else:
                stuck = 0
            left = len(m)
            step(["gen", f"{kind}_jobs.json", *args, "--out", "out", "--budget", str(R.HOLD_BUDGET.get(kind, 650.0))],
                 f"gen_{kind}")
    if enc_t:
        step(["encode", "enc_talk.json", "--dev"], "enc_talk")
    if enc_a:
        step(["encode", "enc_act.json"], "enc_act")
    if talk:
        gen("talk", talk, ["--a2vid", "--fa-amd", "--blaslt"])
        if any(len(e.get("chunks") or []) > 1 for e in es if e["route"] == "talk"):
            subprocess.run([str(R.LTX_PY), str(HERE / "ltx_runner" / "stitch_talk.py"), "talk_jobs.json", "out"],
                           cwd=str(L), capture_output=True)
    if act:
        gen("act", act, ["--ic-lora", str(R.IC_LORA), "--fa-amd", "--blaslt"])
    if fight:                               # two-stage, pipelined (--ti2v ported to the pp runner 10-23)
        step(["encode", "enc_fight.json", "--dev"], "enc_fight")
        os.environ["REMAKE_LTX_SPLIT_ACT"] = str(F._pp_split(md, "ti2v", 24))
        gen("fight", fight, ["--ti2v", "--fa-amd", "--blaslt"])
    return {j["name"]: (L / "out" / f"{j['name']}.mp4").exists() for j in talk + act + fight}


def _green_clip(mp4: Path, frac: float = 0.01) -> bool:
    """A frame (of 1 in 6) where over `frac` of the picture is saturated pure green: control colours leaking."""
    import numpy as np
    if not mp4.exists():
        return False
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", "fps=4,scale=288:160", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, 160, 288, 3).astype(int)
    if not len(v):
        return False
    g = (v[..., 1] > 170) & (v[..., 0] < 90) & (v[..., 2] < 90)
    return bool((g.reshape(len(v), -1).mean(1) > frac).any())


def mouth_motion(mp4: Path) -> float:
    """Mean frame-to-frame change in the picture's centre band (a talking face moves ~1; 0.1 = frozen)."""
    import numpy as np
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                       capture_output=True)
    f = np.frombuffer(r.stdout, np.uint8).reshape(-1, H, W).astype(float)
    if len(f) < 2:
        return 0.0
    m = f[:, int(H * 0.35):int(H * 0.65), int(W * 0.35):int(W * 0.65)]
    return float(np.abs(np.diff(m, axis=0)).mean())


def impacts(anchors: Path, item: str = "hammer", drop: float = 100.0, still: float = 2.0) -> list:
    """The frames where a staged swing LANDS, from the tool's own screen path (anchors.json grips -> work point):
    the first still frame after a fall of more than `drop` px within half a second. 10-22: render.json's "impact"
    is set for every moment (bellows, armour) and a recipe's strike/hold mode does not say whether it swings."""
    if not anchors.exists():
        return []
    ys = []
    for fr in json.loads(anchors.read_text())["frames"]:
        g = [g for f in fr["figures"] for g in f.get("grips", []) if g.get("item") == item and g.get("work")]
        ys.append(g[0]["work"][1] if g else None)
    out = []
    for i in range(1, len(ys) - 2):
        w = ys[i - 1:i + 3]
        if None in w or abs(w[1] - w[0]) < still or max(abs(w[k + 1] - w[k]) for k in (1, 2)) >= still:
            continue
        back = [y for y in ys[max(0, i - 12):i] if y is not None]
        if back and ys[i] - min(back) > drop:
            out.append(i)
    return out


CLANG = ("A heavy blacksmith's hammer strikes hot iron on an anvil: one sharp ringing metallic clang, "
         "a short bright ring after it, close and dry, no music, no voice.")


def clangs(book: Path, stem: str, log=print) -> list:
    """Two clang takes (alternated per hit), drawn once by bookcaster's sound-effect model on the music card."""
    import fcntl
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import sfx as SFX
    _, _, md = paths(book, stem)
    work = md / "sfx"
    work.mkdir(exist_ok=True)
    plan = {"beds": [], "spots": [{"sound": CLANG, "seconds": 1.5} for _ in range(2)]}
    if not all((work / f"spot{n:03d}.wav").exists() for n in range(2)):
        with open(f"/tmp/gpu-{SFX.SFX_GPU_BUS}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            SFX.generate(plan, work, log)
    return [work / f"spot{n:03d}.wav" for n in range(2) if (work / f"spot{n:03d}.wav").exists()]


def assemble(book: Path, stem: str, es: list, mix: Path, out: Path, log=print) -> Path:
    """The clips in time order, each trimmed to its moment, on the chapter mix from the first moment's start; the
    planned sound effects at their time, plus a clang on every staged hammer landing (impacts())."""
    _, sd, md = paths(book, stem)
    L = md / "ltx"
    segs, t0 = [], float(es[0]["t"])
    # each moment's segment runs to the NEXT moment's start (10-23: the moments leave gaps - 29 s after the opening,
    # 14 in all - and back-to-back segments slid every later shot ahead of its narration)
    span = {e["id"]: (float(es[i + 1]["t"]) - float(e["t"]) if i + 1 < len(es) else float(e["dur"]))
            for i, e in enumerate(es)}
    # whole frames counted from the stretch's start (10-23: each segment cut with -t rounded UP to a frame, and the
    # picture fell 1.0 s behind the voice by the chapter's end)
    edge = [round((float(e["t"]) - t0) * 24) for e in es] + [round((float(es[-1]["t"]) + float(es[-1]["dur"]) - t0) * 24)]
    nf = {e["id"]: max(edge[i + 1] - edge[i], 1) for i, e in enumerate(es)}
    for e in es:
        c = L / "out" / f"r_{e['id']}.mp4"
        sg = L / f"seg_{e['id']}.mp4"
        if not c.exists():                  # its time stays filled (a skipped segment slid every later shot off the
            key = md / e["id"] / "key.png"  # narration): the key held still, else black
            log(f"!!! {e['id']}: no clip: " + ("its key held" if key.exists() else "black"))
            src = ["-loop", "1", "-i", str(key)] if key.exists() else ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=24"]
            subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-frames:v", str(nf[e["id"]]), "-vf",
                            f"scale={W}:{H},format=yuv420p", "-an", "-c:v", "libopenh264", "-b:v", "6M", "-r", "24",
                            str(sg)], check=True)
            segs.append((e, sg))
            continue
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(c), "-vf", "tpad=stop_mode=clone:stop_duration=120",
                        "-frames:v", str(nf[e["id"]]), "-an", "-c:v", "libopenh264", "-b:v", "6M", "-r", "24", str(sg)],
                       check=True)              # a clip shorter than its moment holds its last frame (keeps sync)
        segs.append((e, sg))
    lst = L / "segs.txt"
    lst.write_text("".join(f"file '{sg}'\n" for _, sg in segs))
    vid = L / "video_only.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(vid)],
                   check=True)
    dur = sum(nf[e["id"]] for e, _ in segs) / 24
    got = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_packets", "-of", "csv=p=0", str(vid)], capture_output=True, text=True).stdout or 0)
    if got != round(dur * 24):
        log(f"!!! picture is {got} frames, the timeline {round(dur * 24)}: out of sync by {(got - dur * 24) / 24:+.2f} s")
    # a clang on every staged hammer landing (the text plan names few of them): the clip starts at the moment's t
    hits = [(float(e["t"]) + i / 24.0 - t0) for e, _ in segs
            for i in impacts(md / e["id"] / "maps" / "anchors.json") if i / 24.0 < float(e["dur"])]
    return mux(book, stem, vid, t0, dur, mix, out, hits, len(segs), log)


def mux(book: Path, stem: str, vid: Path, t0: float, dur: float, mix: Path, out: Path, hits: list, n: int,
        log=print, planned: bool = True) -> Path:
    """The picture on the chapter mix from t0, the stretch's planned sound effects, a clang at each hit (s from t0)."""
    # sound effects of the stretch: a spot inside a staged strike moves onto its impact frame
    vd, _, _ = paths(book, stem)
    sfx_plan = book / "work" / "sfx" / stem / "jobs.json"
    spots = []
    if planned and sfx_plan.exists():    # (planned=False: the mix has them already - the shots flow, 10-31)
        jobs = json.loads(sfx_plan.read_text())
        plan = json.loads((book / "work" / "sfx" / stem / "plan.json").read_text())
        pl = plan if isinstance(plan, list) else plan.get("spots", [])
        by_prompt = {x.get("prompt"): x for x in jobs if str(x.get("id", "")).startswith("spot")}
        for pp in pl:
            j = by_prompt.get(pp.get("sound"))
            if j is None or "t" not in pp or not Path(j["path"]).exists():
                continue
            t = float(pp["t"])
            if not (t0 <= t < t0 + dur):
                continue
            spots.append((t - t0, j["path"]))
    cl = clangs(book, stem, log) if hits else []
    spots += [(h, str(cl[k % len(cl)])) for k, h in enumerate(hits)] if cl else []
    if hits and not cl:
        log(f"!!! {len(hits)} hammer landing(s) without a clang (the sound-effect model failed)")
    inputs = ["-i", str(vid), "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(mix)]
    fl, labels = [], ["[1:a]"]
    for k, (dt, path) in enumerate(spots):
        inputs += ["-i", path]
        ms = int(max(dt, 0) * 1000)
        fl.append(f"[{k + 2}:a]adelay={ms}|{ms},volume=1.6[s{k}]")
        labels.append(f"[s{k}]")
    graph = ";".join(fl + [f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=first,alimiter=limit=0.89:level=0[a]"])   # clangs on loud lines clipped (10-22)
    # written beside and swapped in whole (10-27: -y rewrote the cut in place, and work/final/<stem>.mp4 is a hard link
    # to it: the chapter's finished video changed - or broke - under every re-run)
    tmp = Path(out).with_name(f".{Path(out).stem}.{os.getpid()}{Path(out).suffix}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(tmp)], check=True)
    os.replace(tmp, out)
    log(f"assembled {n} clips, {len(spots)} sound effect(s) -> {out}")
    return out


def _hero_state(book: Path, where: str, log=print, must: tuple = ()):
    """A place's hero: its path when it is settled (judged empty, or every seed failed before), else what drawing and
    judging it needs."""
    import re
    vd = book / "work" / "video"
    hd = vd / "heroes"
    hd.mkdir(parents=True, exist_ok=True)
    R = _reg(book)
    # one hero per place and side, under the registry's one name of the place (10-27: "The Beautiful Gate" and
    # "Beautiful Gate" were two heroes); the caller's own spelling stays its key in heroes_for
    sw = re.search(r"\s*\((inside|outside)\)\s*$", where, re.I)
    name = (R.place(where) or where) + (f" ({sw.group(1).lower()})" if sw else "")
    out = hd / (re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".png")
    raw = R.place_look(where)                   # the place's one description, any spelling (10-27: by base name only)
    if not raw:
        _say_once(f"!!! no description for the place {where!r} in look.json: its hero is drawn from its name", log)
    desc = re.sub(r"^\s*(inside|outside)\s*:\s*", "", raw, flags=re.I)
    style = R.picture_style()                  # the same picture style as the keys drawn on it (10-27)
    # the side the place's own words give first (10-23: "Outside Hulda's Forge" matched "forge" and was drawn as
    # the inside of a forge, twice); else its description's words vote (world.classify_kind; 10-27: not a guess from
    # one book's room words "forge|hall|shed")
    import world as WO
    named = re.search(r"\((inside|outside)\)|^\s*(inside|outside)\b", where, re.I) or \
        re.match(r"\s*(inside|outside)\s*:", raw, re.I)
    sd_ = (next(g for g in named.groups() if g) if named else "").lower()
    if not sd_ and desc:
        sd_ = "inside" if WO.classify_kind(desc, [])[0] == "interior" else ""
    side = "the inside of" if sd_ == "inside" else "a view of"
    # the set builder's recipe (world.set_picture_prompt): the style's TECHNIQUE only, its architecture words out
    # (10-23: "Gothic ... cathedral-like stone" made the Burnt Mountain's "cave entrance" a carved church portal with
    # candles behind every shot of the dragon fight, and a "no doorway, no carving" sentence drew them anyway)
    pic = WO.set_picture_prompt(f"Wide shot of {side} {name}: {desc}", style, "interior" if side.startswith("the inside")
                                else "exterior")
    prompt = (f"{pic} {period_line(book, 'every building and object')} "
              f"No people. A full-frame picture with no border, no page and no text.")
    jp = hd / f"job_{out.stem}.json"
    try:
        same = json.loads(jp.read_text())[0]["prompt"] == prompt
    except (OSError, ValueError, KeyError, IndexError):
        same = False
    # an EMPTY place (10-23: Outside Hulda's Forge was drawn with figures in the shed and dogs, and every key there
    # copied them - 15 shots failed their people count in all 5 seeds): judged once, redrawn with other seeds
    ok = out.with_suffix(".empty")
    # the scene sheet's fixtures must be in it too (10-24: The Old Forge came out a tavern with no anvil, bellows or
    # hearth, and every key there copied it); the marker records what it was judged for
    # 10-29: a SETTLED hero stays while its own words stay. The fixtures come from the set sheet, asked again from the
    # director's shots, so their wording moved every run ("mantel clock" -> "wooden mantelpiece"); keyed on them, the
    # bedroom was redrawn, its layout re-asked, the director re-planned against it - and the keys were drawn on a
    # newer room than the one he planned in, every run. The fixtures judge a hero's FIRST drawing only
    mk = json.dumps(sorted(must))
    settled = out.exists() and (same or not jp.exists())
    if ok.exists() and must and not settled and (ok.read_text().split("\n", 1) + [""])[1] != mk:
        ok.unlink()                         # judged before the fixtures were asked for: judged again
    tried = out.with_suffix(".tried")         # every seed failed: kept, not drawn again each run
    if out.exists() and same and tried.exists():
        log(f"hero image for {where}: {out} (!!! all seeds failed earlier: people in it or {list(must)} missing)")
        return out
    if out.exists() and (same or not jp.exists()) and ok.exists():
        return out
    q = {"people": "Answer yes or no: are there any people, figures or animals in this picture?"}
    q.update({f"has{i}": f"Answer yes or no: is there {f} in this picture?" for i, f in enumerate(must)})
    same = same or not jp.exists()          # a hero drawn before job files: its words are taken as current
    if not (out.exists() and same):
        out.unlink(missing_ok=True)         # its words changed: drawn again
    return dict(where=where, out=out, prompt=prompt, jp=jp, q=q, ok=ok, tried=tried, mk=mk, must=must,
                seeds=[3, 5, 9, 13])


HERO_CARDS = ("05", "65")


def heroes_for(book: Path, places: dict, log=print) -> dict:
    """{where: must} -> {where: hero path}. Every place still to settle is drawn on BOTH cards by kept-loaded drawers
    (qwen_image.py --serve) and judged on the iGPU as each picture lands; a failed one's next seed goes straight
    back to a drawer (10-24 user: the vision model kept loaded, checking as the sets get built - before, each seed
    of each place started its own drawer and waited for its check: up to 40 model loads, one card, ~31 min)."""
    import characters as CH
    import queue
    import threading
    res, todo = {}, []
    for w, must in places.items():
        st = _hero_state(book, w, log, tuple(must or ()))
        if isinstance(st, Path):
            res[w] = st
        else:
            todo.append(st)
    if not todo:
        return res
    q_ = queue.Queue()
    for st in todo:
        q_.put(st)
    left = {st["where"] for st in todo}
    lock = threading.Lock()

    def judge(st):
        a_ = CH.vl_ask([{"id": "h", "image": str(st["out"]), "ask": st["q"]}]).get("h") or {}
        good = a_ and not CH._yes(a_.get("people")) and all(CH._yes(a_.get(f"has{i}")) for i in range(len(st["must"])))
        if good:
            st["ok"].write_text(f"{st['seed']}\n{st['mk']}" if st["must"] else str(st["seed"]))
        elif st["seeds"] and a_:
            log(f"hero image for {st['where']}: seed {st['seed']} has people in it or misses {list(st['must'])}: "
                f"drawn again")
            st["out"].unlink()
            q_.put(st)                      # its next seed, to whichever drawer is free
            return
        else:
            st["tried"].write_text(st["mk"])
        log(f"hero image for {st['where']}: {st['out']}" +
            ("" if good else f" (!!! people in it or {list(st['must'])} missing)"))
        with lock:
            res[st["where"]] = st["out"]
            left.discard(st["where"])

    def drawer(card):
        hip = subprocess.run([os.path.expanduser("~/bin/hip-index"), card], capture_output=True, text=True).stdout.strip()
        p = None
        try:
            while True:
                with lock:
                    if not left:
                        return
                try:
                    st = q_.get(timeout=5)
                except queue.Empty:
                    continue
                if st["out"].exists():      # drawn before but not judged yet
                    st.setdefault("seed", 0)
                    jq.put(st)
                    continue
                st["seed"] = st["seeds"].pop(0)
                job = {"id": st["out"].stem, "prompt": st["prompt"], "seed": st["seed"], "path": str(st["out"]),
                       "size": [W, H]}
                st["jp"].write_text(json.dumps([job]))
                if p is None:
                    p = subprocess.Popen(["flock", f"/tmp/gpu-{card}.lock", QWEN_PY, "-I", str(HERE / "qwen_image.py"),
                                          "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                         env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
                p.stdin.write(json.dumps(job) + "\n")
                p.stdin.flush()
                while True:                 # its result line (the load line comes first)
                    ln = p.stdout.readline()
                    if not ln:
                        break
                    try:
                        if json.loads(ln).get("id") == job["id"]:
                            break
                    except ValueError:
                        pass
                if st["out"].exists():
                    jq.put(st)              # judged on the iGPU while this card draws the next one
                else:
                    log(f"hero image for {st['where']}: seed {st['seed']} FAILED to draw")
                    if st["seeds"]:
                        q_.put(st)
                    else:
                        with lock:
                            res[st["where"]] = st["out"]
                            left.discard(st["where"])
        finally:
            if p is not None:
                p.stdin.close()
                try:
                    p.wait(60)
                except subprocess.TimeoutExpired:
                    p.kill()

    jq = queue.Queue()

    def judger():
        while True:
            with lock:
                if not left:
                    return
            try:
                st = jq.get(timeout=5)
            except queue.Empty:
                continue
            judge(st)

    ths = [threading.Thread(target=drawer, args=(c,)) for c in HERO_CARDS[:max(1, min(len(HERO_CARDS), len(todo)))]]
    ths.append(threading.Thread(target=judger))
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return res


def hero_for(book: Path, where: str, card: str = "05", log=print, must: tuple = ()) -> Path:
    """The PLACE's hero image (drawn once, cached in work/video/heroes/): a wide empty view of it, in the book's period
    and art style, from the look's place description - the set reference of every keyframe there."""
    return heroes_for(book, {where: must}, log)[where]


def run(book: Path, stem: str, t0: float, t1: float, hero: Path | None, mix: Path, log=print,
        until: str | None = None, guide: str | None = None) -> Path:
    """The whole stretch: stage every moment (CPU, 3 at once) while the first staged ones' keys draw, then the rest
    of the keys, the text check, LTX (talk batch + action batch), the talk freeze check (a frozen clip redrawn once
    with the start at 0.6), assembly."""
    from concurrent.futures import ThreadPoolExecutor
    es = entries(book, stem, t0, t1)
    ids = [e["id"] for e in es]
    if cover(book, stem, t0, t1, log):       # narrated stretches without a moment get theirs (staged by scene())
        scene(book, stem, log)
        es = entries(book, stem, t0, t1)
        ids = [e["id"] for e in es]
    log(f"{len(ids)} moments {t0}-{t1}: {ids}")
    direct(book, stem, ids, log)
    scene(book, stem, log)
    fights = fight_direct(book, stem, es, log)          # the fight moments' opening layouts + action sentences
    # 6 staging renders at once, each Blender on a 32nd-share of the threads (10-23: 3 at once with 16 threads each
    # left the box idle - a render is ~0.4 s a frame, mostly re-posing on the CPU + short iGPU draws; ~2.8 GB each)
    os.environ.setdefault("BLOCKOUT_THREADS", str(max(2, (os.cpu_count() or 16) // STAGE_WORKERS)))
    with ThreadPoolExecutor(STAGE_WORKERS) as ex:
        list(ex.map(lambda m: stage(book, stem, [m], log), ids))
    # each place's hero image (or the one given), then the keys on BOTH cards (one drawing process per card)
    by_place, here = {}, next((e["where"] for e in es if e.get("where")), "")
    for e in es:                            # a talk moment has no place of its own: the scene it sits in (10-22)
        here = e.get("where") or here
        by_place.setdefault(here, []).append(e["id"])
    for where, mids in by_place.items():
        h_ = hero or hero_for(book, where, log=log)
        half = [mids[0::2], mids[1::2]]
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda a: keys(book, stem, a[0], h_, a[1], log=log, fights=fights) if a[0] else None,
                        zip(half, ("05", "65"))))
    pick_keys(book, stem, ids, log, fights=fights)
    dirty = list(getattr(pick_keys, "dirty", []))
    if dirty:                               # no clean seed: three more seeds, then pick again (10-23 overnight: 8 keys
        log(f"no clean key for {dirty}: 3 more seeds each")      # fell back to a white-sheet / page-panel seed 7)
        here_, wh = "", {}
        for e in es:
            here_ = e.get("where") or here_
            wh[e["id"]] = here_
        groups = {}
        for m in dirty:
            groups.setdefault(wh.get(m, ""), []).append(m)
        for where, mids in groups.items():
            h_ = hero or hero_for(book, where, log=log)
            half = [mids[0::2], mids[1::2]]
            with ThreadPoolExecutor(2) as ex:
                list(ex.map(lambda a: keys(book, stem, a[0], h_, a[1], seeds=(13, 17, 23), log=log,
                                                  setting_line=True, fights=fights) if a[0] else None,
                            zip(half, ("05", "65"))))
        pick_keys(book, stem, dirty, log, fights=fights)
        for m in getattr(pick_keys, "dirty", []):
            log(f"!!! {m}: still no clean key after 5 seeds")
    if until == "keys":                     # the checkpoint before the long video step: the staging + the keys
        return review_sheet(book, stem, es, out_path(book, stem, t0, t1).with_name(
            out_path(book, stem, t0, t1).stem + "_keys.jpg"), log)
    global GUIDE
    GUIDE = guide or GUIDE
    if GUIDE == "pins":
        here_, hero_of = "", {}
        for e in es:
            here_ = e.get("where") or here_
            hero_of[e["id"]] = hero or hero_for(book, here_, log=log)
        pin_keys(book, stem, [e for e in es if e["id"] not in fights], hero_of, log)
    log(f"landing actions guided by: {GUIDE}; fights (two-stage, no guide): {sorted(fights)}")
    prompts = motion_prompts(book, stem, [e for e in es if e["id"] not in fights], log)
    ltx(book, stem, es, log, prompts=prompts, fights=fights)
    _, _, md = paths(book, stem)
    if fights:                              # the fight clips judged (a melted / merged body, fire from hands): one redraw
        sys.path.insert(0, str(HERE.parent))
        from tts_reader import animate
        free_igpu = lambda: subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT],  # noqa: E731
                                           capture_output=True)  # the vision judge loads its own model on the iGPU
        free_igpu()
        try:
            bad = {m: f_ for m in fights if (c := md / "ltx" / "out" / f"r_{m}.mp4").exists() and (f_ := fight_clip_ok(c))}
        finally:
            subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
        if bad:
            log(f"fight clip(s) with faults {bad}: redrawn once with another seed")
            keep = {m: md / "ltx" / "out" / f"r_{m}.mp4" for m in bad}
            for m, c in keep.items():       # the first take kept aside (outside the r_<id>* names a redraw clears)
                shutil.copy(c, c.with_name(f"take1_{c.name}"))
            ltx(book, stem, [e for e in es if e["id"] in bad], log, redo={m: 1.0 for m in bad}, prompts=prompts,
                fights=fights)
            free_igpu()
            try:
                for m, f1 in bad.items():
                    c = keep[m]
                    f2 = fight_clip_ok(c) if c.exists() else ["no clip"]
                    if len(f2) >= len(f1):  # the second take is no better: the first one back
                        shutil.copy(c.with_name(f"take1_{c.name}"), c)
                        log(f"!!! {m}: both takes have faults ({f1} / {f2}): the first kept")
                    elif f2:
                        log(f"!!! {m}: the second take kept ({f1} -> {f2})")
            finally:
                subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    def frozen():
        out_ = {}
        for e in es:
            c = md / "ltx" / "out" / f"r_{e['id']}.mp4"
            if e["route"] == "talk" and c.exists() and mouth_motion(c) < 0.4:
                out_[e["id"]] = mouth_motion(c)
        return out_
    green = [e for e in es if e["route"] != "talk" and _green_clip(md / "ltx" / "out" / f"r_{e['id']}.mp4")]
    if green:                               # the pose map's colours leaked in (10-22: a green block in 001745)
        log(f"clip(s) with a green-screen block {[e['id'] for e in green]}: redrawn once with another seed")
        ltx(book, stem, green, log, redo={e["id"]: 1.0 for e in green}, prompts=prompts, fights=fights)
        for e in green:
            if _green_clip(md / "ltx" / "out" / f"r_{e['id']}.mp4"):
                log(f"!!! {e['id']}: green block again after the redraw")
    fz = frozen()
    if fz:
        log(f"frozen talk clip(s) {fz}: redrawn once with the key at 0.6")
        ltx(book, stem, [e for e in es if e["id"] in fz], log, redo={m: 0.6 for m in fz}, prompts=prompts)
        for m, v in frozen().items():
            log(f"!!! {m}: talk clip still frozen (mouth motion {v:.2f})")
    return assemble(book, stem, es, mix, out_path(book, stem, t0, t1), log)


def review_sheet(book: Path, stem: str, es: list, out: Path, log=print) -> Path:
    """One row per moment: the staging (clay, mid-shot), its key, and the moment's id, time and first beats."""
    from PIL import Image, ImageDraw, ImageFont
    _, _, md = paths(book, stem)
    kf = {f.get("shot"): f for f in _keyframes(paths(book, stem)[0])["frames"]}
    tw, th = 576, 320
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 15)
    except OSError:
        font = None
    rows = [e for e in es if (md / e["id"] / "key.png").exists()]
    S = Image.new("RGB", (tw * 2 + 520, th * max(1, len(rows))), (16, 16, 16))
    d = ImageDraw.Draw(S)
    for r, e in enumerate(rows):
        fr = md / e["id"] / "maps" / "frames"
        clay = sorted(fr.glob("clay_*.png"))
        if clay:
            S.paste(Image.open(clay[len(clay) // 2]).convert("RGB").resize((tw, th)), (0, r * th))
        S.paste(Image.open(md / e["id"] / "key.png").convert("RGB").resize((tw, th)), (tw, r * th))
        t = float(e["t"])
        txt = [f"{e['id']}  {int(t // 60)}:{t % 60:04.1f}"] + [str(b)[:300] for b in (kf.get(e["id"], {}).get("beats") or [])[:4]]
        y = r * th + 8
        for ln in txt:
            import textwrap
            for w_ in textwrap.wrap(ln, 62):
                d.text((tw * 2 + 10, y), w_, fill=(230, 230, 230), font=font)
                y += 18
            y += 4
    out.parent.mkdir(parents=True, exist_ok=True)
    S.save(out, quality=82)
    log(f"keys review sheet: {out}")
    return out


def out_path(book: Path, stem: str, t0: float, t1: float) -> Path:
    span = "" if t0 <= 0 and t1 >= 1e8 else f"_{int(t0)}-{int(t1)}"     # the whole chapter carries no span
    return book / "out" / f"{book.name}_{stem}{span}_mannequin.mp4"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("stem")
    ap.add_argument("cmd", choices=["stage", "keys", "run", "assemble"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--hero")
    ap.add_argument("--card", default="05")
    ap.add_argument("--t0", type=float)
    ap.add_argument("--t1", type=float)
    ap.add_argument("--mix")
    a = ap.parse_args(argv)
    book = Path(a.book).expanduser().resolve()
    if a.cmd == "stage":
        print(json.dumps(stage(book, a.stem, a.ids), indent=1))
    elif a.cmd == "keys":
        print(json.dumps(keys(book, a.stem, a.ids, Path(a.hero), a.card), indent=1))
    elif a.cmd == "assemble":
        es = entries(book, a.stem, a.t0, a.t1)
        print(assemble(book, a.stem, es, Path(a.mix) if a.mix else book / "work" / "final" / f"{a.stem}.mp3",
                       out_path(book, a.stem, a.t0, a.t1)))
    else:
        print(run(book, a.stem, a.t0, a.t1, Path(a.hero) if a.hero else None,
                  Path(a.mix) if a.mix else book / "work" / "final" / f"{a.stem}.mp3"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
