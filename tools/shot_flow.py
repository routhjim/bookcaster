"""The shot-list flow (10-23 night, user: "do it"): what the dragon-fight and forge probes showed works, nothing else.

    narration -> SHOTS (the director, Strata: 2-5 s each, ONE physical action, one character where it can be)
              -> one opening picture per shot (Qwen-Image: the place's hero image + the people's LoRAs + their looks;
                 no pose or depth control - the 3D staging fought the video model and fused the bodies)
              -> LTX: action = two-stage (dev DiT + CFG/STG + negative prompt; the distilled model melted the dragon)
                 in TWO takes, quiet = distilled, talk = A2Vid; every batch pipelined over both XTXs (--pp, 2 lanes)
              -> the takes judged (melted / merged bodies, fire from hands), the better kept (a tie keeps the first)
              -> assembled to the narration (each moment's time split over its shots), clangs where the picture lands

    bookcaster video <book> --flow shots [--from m:ss --to m:ss] [--keys-only]

Research behind it (10-23): the opening frame carries a shot's consistency (arXiv 2512.16954: 7.99 -> 0.55 without
it), one exchange per ~5 s clip, extra actions creep in after ~2.8 s, one LoRA per frame (a second character picks up
the first one's features), fights covered as intent / action / reaction shots.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mannequin_flow as MF          # noqa: E402 - the shared pieces: paths, looks, LoRAs, heroes, judge, mux

W, H = MF.W, MF.H
FPS = 24
SHOT_MIN, SHOT_MAX = 2.0, 5.0
# action shots: one take; a second (seed RETAKE) only for a take the judge faults, the better kept (10-24 user: the
# second take changed 4 of 47 shots for ~45 min of both cards; take 1 was faulted in 8 of 47)
TAKES = (42,)
RETAKE = 1042
PP_POOL = True                      # activations between the cards through reused pinned buffers (runner --pp-pool)
DECODE_B = True                     # quiet clips decode on card 65 (runner --decode-b); on once its A/B test passes
TALK_PIECE = 97                     # talk lines in 3-4 s pieces (73-97 frames), chained (10-24 user; was <= 5 s)
KEY_SEEDS, MORE_SEEDS = (7, 11), (13, 17, 23)
STEPS = int(os.environ.get("SHOTS_STEPS", "20"))      # two-stage stage-1 steps (10-23 test: 20 = 30 by eye)
GUIDER = json.loads(os.environ.get("SHOTS_GUIDER", "{\"modality_scale\": 1.0}"))   # the audio-video pass off (we keep our own sound): 105 -> 64 s a 3 s clip with 2 in flight, same picture by eye (10-23)
JUDGE_BATCH = 60                    # pictures per vision-judge call (each call has a 15 min limit)
DIRECT_WORKERS = 3                  # director requests in flight (Strata serves them in its slots)

SHOT_RULES = """You direct a video made from a book. Split this stretch of the story into SHOTS for a video model.
Each shot starts from ONE painted picture of its opening and the video model animates it from your ACTION words alone,
so every shot must be simple, physical and unambiguous.

The stretch lasts {sec:.1f} seconds.
The narration over it: {text}
The director's beats for it: {beats}
Who is here: {people}
The place: {place}
The room (fixed, always there; keep each thing where it stands): {room}
Things as they are when this stretch starts: {now}
The stretch before: {prev}

Rules:
- {nmin} to {nmax} shots, each {smin:.0f} to {smax:.0f} seconds long; the lengths add up to {sec:.1f}.
- Each shot shows ONE physical action (one move, one blow, one exchange) or a quiet held moment. The narration's events
  happen in its order, each played through to its result in some shot (it says "drove the sword in": a shot where the
  blade goes in).
- Show the characters TOGETHER. When two are here, most shots have both in the picture, one on each side, a few steps
  apart, each turned toward the other. When three or more are here, the stretch opens with a WIDE shot of all of them
  (up to {maxp}), where each stands and which way each faces; then shots of the two or three who act or speak, and
  close reactions. A wide shot holds up to {maxp}; a medium or close shot at most {closemax}. A shot of one alone only
  for a close reaction or a detail.
- "crowd": true when the scene's crowd ({crowd}) is in the picture, in the background, small and still; else false. A fight is covered as intent (wide: both fighters, the attacker
  coils, the other sees the opening), action (both in the picture: the one blow or move and the one it reaches),
  reaction (close: the one hit, or a witness).
- "who": the names in the picture, left to right, only from: {names}.
- "framing": "wide", "medium" or "close".
- "picture": one sentence for the OPENING picture only: where each one is and which way each faces, what each holds,
  each body's posture (a fighter in a READY stance, weapon raised toward the opponent; someone knocked down but alive is
  on the ground propped on an elbow, eyes open; someone dead lies flat, eyes closed; a four-legged creature on its four
  legs). Every body part or object the action will use is in the picture (a tail that sweeps, the blade that is struck).
  Nothing in mid-action. Something being MADE is shown being worked on (on the anvil, in the tongs, in the hands),
  never already worn: "he made the breastplate" is him hammering a breastplate on the anvil, not wearing one.
- When the narration says someone MADE, forged, folded or built something, the shots show the WORK itself (heating
  it, folding it, hammering it on the anvil), never the finished thing being admired: "Last of all, he made the sword.
  He folded the star-iron over itself" is him folding and hammering a glowing blade.
- "action": what moves in the shot, in order, plain physical verbs, played through. Every effect names its source
  ("sparks fly from the hammer striking the blade", "a stream of fire from the dragon's mouth"). A change over time is
  written as visible steps ("the blade's edge glows faint blue, brighter with each blow"). Say what the hands hold.
  Call each one by a plain description (a young man in a leather apron, a huge black dragon), never by name. No magic
  words (magic, enchanted, spell) and no camera directions. At most 50 words.
- "kind": "action" when a body or object moves with force (a strike, a fall, a lunge, fire, a hammer blow), else
  "quiet" (breathing, looking, a few steps).
{fix}Reply with JSON only: {{"shots": [{{"dur": 3.0, "who": ["..."], "framing": "...", "picture": "...", "action": "...", "kind": "...", "crowd": false}}]}}"""

TALK_PICTURE = ("{who}, seen from the chest up and filling much of the picture, turned three-quarters toward someone just "
                "beside the viewer, mid-word")
# an exchange (10-24 user: "having only one person in talking shots in the exact same position that the other one was
# in isn't convincing"; "I don't like that there's only ever one person in each shot"): its first line a two-shot, the
# replies over the listener's shoulder, each one on the side the two-shot gave them
# 10-25 Acts: "a few steps apart" drew them head to toe, faces ~30 px - no lips to sync ("lip sync is off")
TALK_TWO = ("{left} on the left and {right} on the right, seen from the waist up and close to the viewer, both faces large "
            "in the picture, an arm's length apart, facing each other, {who} mid-word")
TALK_OTS = ("over-the-shoulder: {who} on the {side} third of the picture, seen from the chest up, turned toward the "
            "{other} and mid-word; in the {other} foreground, close to the viewer and out of focus, the back of "
            "{listener}'s head and one shoulder, facing {who}")
TALK_CREATURE = ("a close view of {who}'s head and long neck, low to the ground, jaws slightly parted, the body behind it "
                 "on four legs")
# words the image model draws as something else (10-23: "bellows" = a bell, twice)
GLOSS = {r"\bbellows\b": "bellows (two flat wooden boards hinged at the back, joined by pleated leather sides, a short "
                          "iron nozzle pointing into the fire)"}
# a four-legged creature's body plan in the picture (10-23 shots keys: the dragon stood upright like a seahorse in most
# fight keys, its LoRA at 0.8 full frame; at 0.63 in a side box it crouched on all fours) + its LoRA lighter
FOUR_LEGS = ("its body long and horizontal like a giant lizard's, all four feet on the ground, never upright on its "
             "hind legs")
CREATURE_LORA = 0.5
SETTING = (" The scene takes place in {place}, which fills the whole picture behind them to its edges: a single painted "
           "scene, never a character sheet, never a plain or white background.")
SHOT_MAGIC = re.compile(r"\b(magic\w*|enchant\w*|spell\w*|sorcer\w*)\b", re.I)
HAND_GLOW = re.compile(r"\b(glow\w*|light|fire|flames?|sparks?)\b[^.;]*\b(hands?|palms?|fingers?)\b|"
                       r"\b(hands?|palms?|fingers?)\b[^.;]*\b(glow\w*|shin\w*|burn\w*)\b", re.I)
STRIKE = re.compile(r"\b(hammer\w*|anvil|strik\w*|struck|blows?|clang\w*)\b", re.I)


def shot_dir(book: Path, stem: str) -> Path:
    return MF.paths(book, stem)[0] / "shots"


def _frames(sec: float) -> int:
    """An LTX length (8k+1 frames) covering `sec`."""
    return 8 * max(1, math.ceil((sec * FPS - 1) / 8)) + 1


# --- groups (10-24 user: "an up to 6 people framework that appropriately sets up a wide shot for that many people and
# then handles everything from there ... with a crowd possibly in the background") ------------------------------------
MAX_PEOPLE = 6                      # named people in one picture; 3+ = a wide shot; 4+ in two depth rows
CLOSE_MAX = 2                       # medium / close shots: faces and lips matter
LORA_W = {1: 0.8, 2: 0.63, 3: 0.55, 4: 0.5, 5: 0.45, 6: 0.45}   # each LoRA lighter as the boxes shrink (bleed)


STILL_CROWD = " The people in the background stay still."   # (LTX melts a moving crowd's faces)


def group_layout(n: int) -> list:
    """[(box, where-words)] left to right for n people: one row up to 3; 4-6 in two depth rows, zig-zag (each one
    behind stands in the gap between two in front, higher and smaller), as a group is staged for a wide shot."""
    if n <= 1:
        return [([0.0, 0.0, 1.0, 1.0], "")]
    if n == 2:
        return [([0.02, 0.1, 0.5, 1.0], "on the left"), ([0.42, 0.0, 1.0, 1.0], "on the right")]
    if n == 3:
        return [([0.0, 0.1, 0.38, 1.0], "on the left"), ([0.31, 0.05, 0.69, 1.0], "in the centre"),
                ([0.62, 0.1, 1.0, 1.0], "on the right")]
    out = []                        # zig-zag: even places in front (low, big), odd places behind (higher, smaller)
    fw = min(0.5, 2.2 / n)
    for k in range(n):
        c = (k + 0.5) / n
        side = "on the far left" if c < 0.15 else "on the left" if c < 0.4 else "in the centre" if c < 0.6 else \
            "on the right" if c < 0.85 else "on the far right"
        if k % 2 == 0:
            out.append(([max(0.0, c - fw / 2), 0.3, min(1.0, c + fw / 2), 1.0], f"in front, {side}"))
        else:
            out.append(([max(0.0, c - fw * 0.32), 0.0, min(1.0, c + fw * 0.32), 0.62], f"behind, {side}"))
    return sorted(out, key=lambda b: b[0][0] + b[0][2])


def _cast(book: Path, stem: str, e: dict) -> tuple:
    """(names in the moment, creatures among them)."""
    _, _, md = MF.paths(book, stem)
    who = e.get("who") or MF._frame(book, stem, e["id"]).get("who") or MF._who(book, stem, e["id"])
    if isinstance(who, str):
        who = re.findall(r"'([^']+)'", who) or [w.strip() for w in who.split(",") if w.strip()]
    crs = MF._creatures(md, e["id"])
    _, cx = _ctx(book, stem)
    crs += [w for w in who if w not in crs and re.search(r"\b(dragon|beast|creature|wyrm|wolf|horse|monster)\b",
                                                          cx.look(w, 0) or "", re.I)]
    return list(dict.fromkeys(who)), crs


_CTX = {}


def _ctx(book: Path, stem: str):
    import flow3d as F
    k = (str(book), stem)
    if k not in _CTX:
        _CTX[k] = F._ctx(book, stem)
    return _CTX[k]


def _spans(es: list) -> dict:
    """Each moment's time: to the next moment's start (the last: its own length)."""
    return {e["id"]: (float(es[i + 1]["t"]) - float(e["t"]) if i + 1 < len(es) else float(e["dur"]))
            for i, e in enumerate(es)}


def _faults(a: dict, cast: list, sec: float) -> list:
    out = []
    shots = a.get("shots") if isinstance(a.get("shots"), list) else []
    if not shots:
        return ["no shots"]
    for i, s in enumerate(shots, 1):
        act = str(s.get("action", ""))
        if not s.get("picture") or not act:
            out.append(f"shot {i} has no picture or action")
        if (cast and not s.get("who")) or any(w not in cast for w in s.get("who") or []):
            out.append(f"shot {i}: who must be names from {', '.join(cast)}")
        nw = len(s.get("who") or [])
        if nw > MAX_PEOPLE:
            out.append(f"shot {i} has {nw} people: at most {MAX_PEOPLE}")
        elif nw > CLOSE_MAX and s.get("framing") != "wide":
            out.append(f"shot {i} has {nw} people in a {s.get('framing')} shot: {nw} people need a wide shot")
        named = [w for w in cast if re.search(rf"\b{re.escape(w if w.lower().startswith('the ') else w.split()[-1])}\b",
                                              act)]
        if named:
            out.append(f"shot {i} names {', '.join(named)} in the action (use a plain description)")
        if SHOT_MAGIC.search(act):
            out.append(f"shot {i} uses the word '{SHOT_MAGIC.search(act).group(0)}'")
        if HAND_GLOW.search(act):
            out.append(f"shot {i} puts light or fire at someone's hands (give it its real source)")
        for sent in re.split(r"(?<=[.;])\s+", act):
            if MF.EFFECT.search(sent) and not re.search(r"\bfrom\b", sent):
                out.append(f"shot {i}: '{MF.EFFECT.search(sent).group(0)}' does not say where it comes from")
        if len(act.split()) > 70:
            out.append(f"shot {i}'s action is over 50 words")
    tot = sum(float(s.get("dur") or 0) for s in shots)
    if abs(tot - sec) > max(1.0, 0.15 * sec):
        out.append(f"the lengths add up to {tot:.1f} s, not {sec:.1f}")
    return out


def direct(book: Path, stem: str, es: list, log=print, sheets: dict | None = None) -> dict:
    """{mid: [shot]} for every moment: talk = one shot of the speaker; the rest from the director (cached by input in
    shots/<mid>/shots.json; asked again with the faults, twice)."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    _, cx = _ctx(book, stem)
    sd = shot_dir(book, stem)
    tlp = book / "work" / f"{stem}.timeline.json"
    tl = json.loads(tlp.read_text()) if tlp.exists() else []
    span = _spans(es)
    out, todo, here, prev_text = {}, [], "", "(the start of the chapter)"
    sides, pair, last_who = {}, None, ""         # the exchange in progress: who stands where
    # the people the scene's own moments show, per run of one place: a talk shot's listener comes from these first
    # (10-25 Acts: the sheet's "present" kept Jesus in the Upper Room after the ascension and Peter talked to him)
    seen, run_of, w_ = [], {}, None
    for e in es:
        if (e.get("where") or w_) != w_:
            seen = []
        w_ = e.get("where") or w_
        seen.append(_cast(book, stem, e)[0])
        run_of[e["id"]] = seen
    for e in es:
        if (e.get("where") or here) != here:
            sides, pair, last_who = {}, None, ""
        here = e.get("where") or here
        sec = span[e["id"]]
        said = " ".join(x["text"] for x in tl if float(e["t"]) - 0.5 <= x["t"] < float(e["t"]) + sec - 0.5)
        if e["route"] == "talk":
            who = e.get("speaker") or ""
            cast_t, crs_ = _cast(book, stem, e)
            shown = list(dict.fromkeys(w for c in run_of.get(e["id"], []) for w in c))
            cast_t = list(dict.fromkeys(cast_t + shown + ([] if shown else
                                                          ((sheets or {}).get(e["id"]) or {}).get("present", []))))
            others = [w for w in cast_t if w and w != who]
            listener = last_who if last_who in others else (others[0] if others else "")
            sh = {"id": f"{e['id']}_s0", "dur": sec, "who": [who] if who else [], "framing": "medium",
                  "picture": (TALK_CREATURE if who in crs_ else TALK_PICTURE).format(who=who or "the speaker"),
                  "action": "", "kind": "talk", "place": here}
            if who and listener and who not in crs_ and listener not in crs_:
                if who not in sides and listener not in sides:
                    sides.update({who: "left", listener: "right"})
                sides.setdefault(who, "right" if sides.get(listener) == "left" else "left")
                sides.setdefault(listener, "right" if sides[who] == "left" else "left")
                if sides[listener] == sides[who]:       # a third one joins: the listener faces the speaker
                    sides[listener] = "right" if sides[who] == "left" else "left"
                lr = sorted((who, listener), key=lambda w: sides[w] != "left")
                if pair != {who, listener}:
                    sh.update(who=lr, listener=listener, picture=TALK_TWO.format(left=lr[0], right=lr[1], who=who))
                else:
                    other = "right" if sides[who] == "left" else "left"
                    sh.update(who=lr, listener=listener, ots=listener, picture=TALK_OTS.format(
                        who=who, side=sides[who], other=other, listener=listener))
                pair = {who, listener}
            last_who = who or last_who
            out[e["id"]] = [sh]
            prev_text = said or prev_text
            continue
        cast, crs = _cast(book, stem, e)
        sh_ = (sheets or {}).get(e["id"]) or {}
        at = sh_.get("elsewhere") or here     # 10-24: Hulda's forging, narrated in the King's Hall, drawn in the hall
        if sh_.get("elsewhere"):
            sh_ = {}                            # another place: not this room, its people or its crowd
        cast = list(dict.fromkeys(cast + sh_.get("present", [])))      # the silent ones in the room too
        f = MF._frame(book, stem, e["id"])
        looks = {w: MF._brief(MF._appearance(cx.look(w, e["t"]) or "")) for w in cast}
        q0 = SHOT_RULES.format(sec=sec, text=(said or f.get("prompt") or "")[:900],
                               beats=" / ".join(f.get("beats") or []) or "(none)",
                               people="; ".join(f"{w} ({looks.get(w, '')})" for w in cast) or "nobody", place=at,
                               room=sh_.get("room") or "(as the narration says)", now=sh_.get("now") or "(nothing noted)",
                               maxp=MAX_PEOPLE, closemax=CLOSE_MAX, crowd=sh_.get("crowd") or "none here",
                               prev=prev_text[:400], names=", ".join(cast) or "(nobody: the place alone)", nmin=max(1, math.ceil(sec / SHOT_MAX)),
                               nmax=max(1, math.floor(sec / SHOT_MIN)), smin=SHOT_MIN, smax=SHOT_MAX, fix="{fix}")
        prev_text = said or prev_text
        cp = sd / e["id"] / "shots.json"
        k = MF._h(q0)
        got = json.loads(cp.read_text()) if cp.exists() else {}
        # a cached answer a newer check faults is asked again - once: one asked with the checks ("checked") stays
        # (10-24: 570 and 1468 kept their faults, were asked again every run, and redrew their keys and clips)
        if got.get("k") == k and (got.get("checked") or not (
                _worn(got, " ".join(MF._dress(book, stem, e["id"], cast).values())) or _making(got, said))):
            out[e["id"]] = json.loads(cp.read_text())["shots"]
            continue                    # (a cached answer a newer check faults is asked again)
        todo.append((e, q0, k, cast, crs, sec, at, said))
    if todo:
        animate.igpu("strata", log)

        def ask(t):
            e, q0, k, cast, crs, sec, here, said = t
            a, faults = {}, []
            for _ in range(3):
                fix = ("Your last answer broke the rules: " + "; ".join(faults) + ". Write it again.\n") if faults else ""
                reply = _chat(BC.LLM_URL, BC.LLM_MODEL, q0.replace("{fix}", fix), max_tokens=2500)
                m = re.search(r"\{.*\}", reply or "", re.S)
                try:
                    a = json.loads(m.group(0)) if m else {}
                except ValueError:
                    a = {}
                for s in a.get("shots") or []:
                    s["who"] = list(dict.fromkeys(w for w in (MF._cast_name(x, cast, crs) for x in (s.get("who") or []))
                                                  if w))
                faults = _faults(a, cast, sec) + _worn(a, " ".join(MF._dress(book, stem, e["id"], cast).values())) + \
                    _making(a, said)
                if not faults:
                    break
            shots = [s for s in (a.get("shots") or []) if s.get("picture") and s.get("action")]
            if not shots:
                log(f"!!! {e['id']}: no usable shots ({faults}): one quiet shot of the moment")
                shots = [{"dur": sec, "who": cast[:1], "framing": "wide", "kind": "quiet",
                          "picture": MF._frame(book, stem, e["id"]).get("prompt") or "", "action": "Small movements."}]
            elif faults:
                log(f"!!! {e['id']}: shots still break the rules: {faults}")
            tot = sum(float(s.get("dur") or 0) for s in shots) or 1.0
            for i, s in enumerate(shots):   # the lengths scaled to the moment's time exactly
                s.update(id=f"{e['id']}_s{i}", dur=float(s.get("dur") or 0) * sec / tot, place=here,
                         framing=s.get("framing") or "medium",
                         kind=s.get("kind") if s.get("kind") in ("action", "quiet") else "action")
            (sd / e["id"]).mkdir(parents=True, exist_ok=True)
            # a stand-in is not cached (the next run asks again)
            (sd / e["id"] / "shots.json").write_text(json.dumps({"k": k if a.get("shots") else "", "shots": shots,
                                                                 "checked": True}, indent=1))
            for s in shots:
                log(f"{s['id']} ({s['dur']:.1f} s, {s['kind']}, {s['framing']}, {', '.join(s['who'])}): "
                    f"{s['picture']} || {s['action']}")
            return e["id"], shots
        try:
            with ThreadPoolExecutor(DIRECT_WORKERS) as ex:
                for mid, shots in ex.map(ask, todo):
                    out[mid] = shots
        finally:
            subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    return out


PLACE_ASK = """Describe the place "{where}" as a painter would see it {side}, empty: no people and no animals. One sentence
of under 35 words: the ground, the buildings or the landscape, what stands there, the light. Early medieval Northern
Europe, around 1000 AD. Use what this passage says and things that belong in such a place. Only the place itself:
no swords, weapons or other things the story's people make, carry or fight with, and no light that is not the sky's
or a fire's.

Passage: {text}

Answer with the sentence only."""


SCENE_ASK = """You keep CONTINUITY for one scene of a video made from a book: everything that must look the same from
shot to shot, and everything that changes, with the moment it changes.

The place: {place}
The people of the book: {people}
Props the book knows: {props}
The scene, moment by moment ([id] what is narrated or said):
{moments}

Reply with JSON only:
{{"room": "two sentences: the FIXED things of this place and where each stands, as a painter needs them (the hearth
with its chimney hood at the back wall, the bellows on its left, the anvil on a stump in front of it, ...). Never a
person or creature, nothing the story makes or carries, no weather or sky; for a place indoors, nothing seen
outside it.",
 "fixtures": ["2 to 5 short names of things BUILT INTO this place a viewer must see in every wide picture of it
 (a hearth, an anvil, a door, a window): never a prop someone carries, lights or brings, and nothing the story
 makes"],
 "present": ["the people in this place during the scene, also the silent ones, only from the people above"],
 "crowd": "the UNNAMED people always in this place during the scene, as a painter needs them (about twenty courtiers
 in dark robes along both walls), or \"\" when there are none",
 "elsewhere": [{{"id": "<moment id>", "place": "<the place it shows>"}}],
 "states": [{{"from": "<the moment id where it starts>", "thing": "<short name>", "look": "<how it looks from then
 on, in plain painter's words>"}}]}}
"elsewhere": the moments whose narration SHOWS another place than this one - a memory, a story someone tells,
what someone far away is doing - with that place, named as in this list when it is one of them: {places}. [] when none.
Everyone else in this scene is not in those moments.
"states": every visible thing that CHANGES in the scene, one entry each time it changes, in order: the work piece (a
dark bar with a crack across its face -> glowing orange, the crack closing -> a breastplate taking shape on the anvil
-> ...), things made so far and where they lie, a door opened, a fire lit. Something being made lies on the anvil or
in the hands, never worn."""


# 10-24: the scene room hung "a dark blue sword above the door" (the sword the story makes) despite the rule
# and (10-24 King's Hall) "a massive black dragon silhouette" in its windows in every key
STORY_THING = re.compile(r"\b(swords?|blades?|axes?|spears?|shields?|armou?r|breastplates?|helms?|helmets?|gauntlets?|greaves|"
                         r"dragons?|beasts?|monsters?|creatures?|silhouettes?)\b", re.I)


def _unpropped(room: str) -> str:
    """The room without its sentences about weapons or armour: those are the story's, never the set's."""
    return " ".join(x for x in re.split(r"(?<=[.!?])\s+", room.strip()) if not STORY_THING.search(x))


def scene_sheets(book: Path, stem: str, es: list, log=print) -> dict:
    """{mid: {"scene", "room", "fixtures", "present", "now"}}: the director's continuity sheet per scene (a run of
    moments in one place; 10-24 user: "the forge scene stinks, so many things are missing or inconsistent": every key
    was drawn alone, so the bellows, the anvil, the metal's state and Hulda came and went). Cached per scene by input
    in shots/scenes/<scene>.json. "now" = each changing thing as it is at the moment."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    _, cx = _ctx(book, stem)
    vd, _, _ = MF.paths(book, stem)
    plan = json.loads((vd / "plan.json").read_text())
    chars = sorted(((plan.get("look") or {}).get("characters") or {}))
    lp = book / "work" / "video" / "look.json"
    props = (json.loads(lp.read_text()).get("props") or {}) if lp.exists() else {}
    tlp = book / "work" / f"{stem}.timeline.json"
    tl = json.loads(tlp.read_text()) if tlp.exists() else []
    span = _spans(es)
    places = ", ".join(dict.fromkeys(e.get("where") for e in MF.entries(book, stem, 0, 1e9) if e.get("where")))
    runs, here = [], None
    for e in es:
        w = e.get("where") or here or ""
        if not runs or w != here:
            runs.append((w, []))
        here = w
        runs[-1][1].append(e)
    sd = shot_dir(book, stem) / "scenes"
    sd.mkdir(parents=True, exist_ok=True)
    sheets, todo = {}, []
    for n, (w, run) in enumerate(runs):
        lines = []
        for e in run:
            said = " ".join(x["text"] for x in tl if float(e["t"]) - 0.5 <= x["t"] < float(e["t"]) + span[e["id"]] - 0.5)
            lines.append(f"[{e['id']}] " + (f"{e.get('speaker')} says: " if e["route"] == "talk" else "") + said[:600])
        q = SCENE_ASK.format(place=w or "(unnamed)", people=", ".join(chars), places=places,
                             props="; ".join(f"{k}: {v}" for k, v in props.items())[:2500], moments="\n".join(lines))
        sid = f"sc{n:02d}_" + re.sub(r"[^a-z0-9]+", "_", (w or "none").lower()).strip("_")
        cp = sd / f"{sid}.json"
        k = MF._h(q)
        got = json.loads(cp.read_text()) if cp.exists() else {}
        if got.get("k") == k:
            sheets[sid] = (run, got["sheet"])
        else:
            todo.append((sid, run, q, k, cp))
    if todo:
        animate.igpu("strata", log)

        def ask(t):
            sid, run, q, k, cp = t
            a = {}
            for _ in range(2):
                reply = _chat(BC.LLM_URL, BC.LLM_MODEL, q, max_tokens=3000)
                m = re.search(r"\{.*\}", reply or "", re.S)
                try:
                    a = json.loads(m.group(0)) if m else {}
                except ValueError:
                    a = {}
                if a.get("room"):
                    break
            ids = {e["id"] for e in run}
            a = {"room": str(a.get("room") or "").strip(),
                 "fixtures": [str(x) for x in (a.get("fixtures") or [])][:5],
                 "present": [x for x in (a.get("present") or []) if x in chars],
                 "crowd": str(a.get("crowd") or "").strip(),
                 "states": [x for x in (a.get("states") or []) if isinstance(x, dict) and x.get("from") in ids
                            and x.get("thing") and x.get("look")],
                 "elsewhere": {x["id"]: str(x["place"]).strip() for x in (a.get("elsewhere") or [])
                               if isinstance(x, dict) and x.get("id") in ids and str(x.get("place") or "").strip()
                               and str(x["place"]).strip() != w}}
            if a["room"]:
                cp.write_text(json.dumps({"k": k, "sheet": a}, indent=1))
            log(f"scene {sid}: {a['room']} | fixtures {a['fixtures']} | present {a['present']} | "
                + " -> ".join(f"{x['thing']}@{x['from']}: {x['look']}" for x in a["states"]))
            return sid, run, a
        try:
            with ThreadPoolExecutor(DIRECT_WORKERS) as ex:
                for sid, run, a in ex.map(ask, todo):
                    sheets[sid] = (run, a)
        finally:
            subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    out = {}
    for sid, (run, a) in sheets.items():
        a = dict(a, room=_unpropped(a.get("room", "")),
                 fixtures=[f for f in a.get("fixtures") or [] if not STORY_THING.search(f)])
        cur = {}
        for e in run:
            for x in a.get("states") or []:
                if x["from"] == e["id"]:
                    cur[x["thing"]] = x["look"]
            out[e["id"]] = {"scene": sid, "room": a.get("room", ""), "fixtures": a.get("fixtures") or [],
                            "present": a.get("present") or [], "crowd": a.get("crowd") or "",
                            "now": "; ".join(f"{t}: {v}" for t, v in cur.items()),
                            "elsewhere": (a.get("elsewhere") or {}).get(e["id"], "")}
    return out


def wardrobe(book: Path, stem: str, log=print) -> None:
    """The chapter's wardrobe, planned again when its rules changed (video.ensure_wardrobe; 10-24: Edric wore the
    armor he was forging, in the forge). Each key's dress comes from it (MF._dress)."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate, video
    from tts_reader import bookcaster as BC
    vd, _, _ = MF.paths(book, stem)
    pp = vd / "plan.json"
    plan = json.loads(pp.read_text())
    import hashlib
    k = hashlib.sha1(video.WARDROBE_PROMPT.encode()).hexdigest()[:8]
    if plan.get("wardrobe_k") == k or (not plan.get("wardrobe") and "wardrobe_k" in plan):
        return                          # current (or nobody changes): no language model needed
    animate.igpu("strata", log)
    if video.ensure_wardrobe(plan, (book / "chapters" / f"{stem}.txt").read_text(), BC.LLM_URL, BC.LLM_MODEL, log):
        pp.write_text(json.dumps(plan, indent=1, ensure_ascii=False))


def describe_places(book: Path, stem: str, es: list, shots: list, log=print) -> None:
    """A description for every place the shots use that the looks never described (10-24: Outside Hulda's Forge,
    the Barley Fields, the Barren Lands and the Tower had none; their hero pictures were drawn from the name alone and
    filled with people in every seed, and the keys copied them): the director writes one from the passages there,
    kept in look.json. A place whose hero is already judged empty keeps it (its shots stay cached)."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    vd = book / "work" / "video"
    lp = vd / "look.json"
    look = json.loads(lp.read_text()) if lp.exists() else {}
    places = look.setdefault("places", {})
    tlp = book / "work" / f"{stem}.timeline.json"
    tl = json.loads(tlp.read_text()) if tlp.exists() else []
    span, t_of = _spans(es), {e["id"]: float(e["t"]) for e in es}
    ask_k = MF._h(PLACE_ASK)
    # the scene sheet's ROOM is the place's description wherever the looks have none (or only one of ours): the hero
    # is drawn with its fixtures and every key there shares them (10-24: The Old Forge, undescribed, came out a
    # tavern - wall fireplace, glass windows, chandelier - with no anvil, bellows or hearth)
    done = set()
    for s in shots:
        w, room = s.get("place") or "", s.get("room") or ""
        base = re.sub(r"\s*\((inside|outside)\)$", "", w)
        mine = (look.get("_described") or {}).get(base)
        if not w or not room or base in done or (places.get(base) and mine is None):
            continue
        done.add(base)
        for k_, v_ in GLOSS.items():        # (the bellows drawn as a lute in the place picture, 10-24)
            room = re.sub(k_, v_, room, count=1, flags=re.I)
        rk = "room:" + MF._h(room)
        if mine != rk:
            wj = vd / "worlds" / re.sub(r"[^a-z0-9]+", "_", w.lower()).strip("_") / "world.json"
            kind = json.loads(wj.read_text()).get("kind") if wj.exists() else ""
            m = re.search(r"\((inside|outside)\)$|^(outside|inside)\b", w, re.I)
            sd_ = (next(g for g in m.groups() if g).lower() if m else
                   {"interior": "inside", "exterior": "outside"}.get(kind, "outside"))
            places[base] = f"{sd_.capitalize()}: {room}"
            look.setdefault("_described", {})[base] = rk
            log(f"place described (scene sheet): {base}: {places[base]}")
    if done:
        lp.write_text(json.dumps(look, indent=1))
    todo = {}
    for s in shots:
        w = s.get("place") or ""
        base = re.sub(r"\s*\((inside|outside)\)$", "", w)
        slug = re.sub(r"[^a-z0-9]+", "_", w.lower()).strip("_")
        # a description this step wrote with older rules is written again (10-24: "a dark blue sword hanging above
        # the door" of Outside Hulda's Forge put the story's sword in every shot there)
        mine = (look.get("_described") or {}).get(base)
        if not w or base in done or (places.get(base) and mine in (None, ask_k)) or \
                (not places.get(base) and (vd / "heroes" / f"{slug}.empty").exists()):
            continue
        mid = s["id"].rsplit("_s", 1)[0]
        todo.setdefault(w, set()).add(mid)
    if not todo:
        return
    animate.igpu("strata", log)
    try:
        for w, mids in todo.items():
            base = re.sub(r"\s*\((inside|outside)\)$", "", w)
            wj = vd / "worlds" / re.sub(r"[^a-z0-9]+", "_", w.lower()).strip("_") / "world.json"
            kind = json.loads(wj.read_text()).get("kind") if wj.exists() else ""
            m = re.search(r"\((inside|outside)\)$|^(outside|inside)\b", w, re.I)
            sd_ = (next(g for g in m.groups() if g).lower() if m else
                   {"interior": "inside", "exterior": "outside"}.get(kind, "outside"))
            text = " ".join(x["text"] for x in tl for mid in sorted(mids, key=t_of.get)
                            if t_of[mid] - 0.5 <= x["t"] < t_of[mid] + span[mid] - 0.5)[:1500]
            reply = (_chat(BC.LLM_URL, BC.LLM_MODEL, PLACE_ASK.format(where=base, side=f"from {sd_}", text=text),
                           max_tokens=300) or "").strip().strip('"')
            reply = re.sub(r"(?s)<think>.*?</think>", "", reply).strip().splitlines()[0:1]
            if reply and reply[0].strip():
                places[base] = f"{sd_.capitalize()}: {reply[0].strip()}"
                look.setdefault("_described", {})[base] = ask_k
                log(f"place described: {base}: {places[base]}")
    finally:
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    lp.write_text(json.dumps(look, indent=1))


# armor on a body the wardrobe hasn't dressed in it (10-24: "He made the breastplate first, curved to his own chest" =
# Edric pressing it to his chest, then wearing the arm and leg plates, in the forge)
WORN = re.compile(r"\b(wear\w*|put\w* on|tries? on|places? [^.;]{0,30}\bon (his|her|their) head|(against|over|on) "
                  r"(his|her|their) (chest|torso|arms?|legs?|shins?|forearms?|trousers|jerkin|head))\b", re.I)
ARMOR = re.compile(r"\b(armou?r|breastplate|helm\w*|plates?|mail|gauntlets?|greaves)\b", re.I)


MAKING = re.compile(r"\b(made|makes|making|forged|forges|forging|folded|folds|folding|built|crafted|hammered)\b", re.I)
WORK = re.compile(r"\b(hammer\w*|strik\w*|struck|fold\w*|heat\w*|pound\w*|beat\w*|forg\w*|bellows|tongs|anvil)\b",
                  re.I)


def _making(a: dict, said: str) -> list:
    """The narration makes something and no shot shows the work (10-24: "Last of all, he made the sword. He folded the
    star-iron over itself a hundred times" came out as Edric turning a finished sword in his hands)."""
    if not MAKING.search(said or ""):
        return []
    if any(WORK.search(str(s.get("action", ""))) for s in a.get("shots") or []):
        return []
    return ["the narration has something being made, but no shot shows the work (heating, folding, hammering it)"]


def _worn(a: dict, dressed: str) -> list:
    """Shots that put armor on someone whose dress at this moment has none."""
    if ARMOR.search(dressed or ""):
        return []
    out = []
    for i, s in enumerate(a.get("shots") or [], 1):
        for sent in re.split(r"(?<=[.;])\s+", f"{s.get('picture', '')} {s.get('action', '')}"):
            if WORN.search(sent) and ARMOR.search(sent):
                out.append(f"shot {i} puts armor on someone who is not wearing any yet (it is being made: show it "
                           f"on the anvil, in the tongs or held in the hands, away from the body)")
                break
    return out


STYLE_CONTENT = re.compile(r"\b(cathedral|church|chapel|temple|castle|spire|arch\w*|stone|iron|window|sky|skies|"
                           r"candle\w*|ruins?|tower)\b", re.I)


def _key_style(style: str) -> str:
    """The art style's TECHNIQUE (10-23: "Gothic dark fantasy ...: ..., cathedral-like stone and iron" put a carved
    church behind the dragon fight at 4:12, 4:17 and 6:16): architecture words out of its head, content clauses out."""
    head, _, rest = (style or "").partition(":")
    head = re.sub(r"(?i)\b(gothic|baroque|romanesque|victorian|byzantine|rococo|renaissance|medieval)\b\s*", "", head)
    keep = [re.sub(r"(?i)\bgothic\s*", "", c).strip() for c in re.split(r",", rest) if c.strip() and not STYLE_CONTENT.search(c)]
    return (head.strip(" ,.") + (": " + ", ".join(keep) if keep else "")).rstrip(".") + "." if head.strip() else ""


def _key_prompt(book: Path, stem: str, mid: str, s: dict) -> str:
    vd, _, _ = MF.paths(book, stem)
    plan = json.loads((vd / "plan.json").read_text())
    looks = {w: MF._appearance(d) for w, d in ((plan.get("look") or {}).get("characters") or {}).items()}
    style_p = book / "art_style.txt"
    style = _key_style(style_p.read_text().strip()) if style_p.exists() else ""
    dress = MF._dress(book, stem, mid, s["who"])
    people = []
    for w in s["who"]:
        lk = looks.get(w, "")
        if w in dress:
            lk = re.split(r",?\s*\bwearing\b", lk)[0] + ", wearing " + dress[w]
        lay = group_layout(len(s["who"]))
        side = f" {lay[s['who'].index(w)][1]}" if len(s["who"]) > 1 else ""
        legs = ", " + FOUR_LEGS if "four-legged" in lk and not MF.DOWN.search(s["picture"]) else ""
        if w == s.get("ots"):           # over the shoulder: the back of the head, out of focus
            people.append(f"{w}, seen from behind{side}: {re.split(r',', lk)[0]}")
            continue
        people.append(f"{MF._lora(book, w)[1] or ''} {w}{side}"
                      f"{', whole body visible' if s.get('framing') == 'wide' else ''}: {lk.rstrip('.')}{legs}".strip())
    place = re.sub(r"\s*\((inside|outside)\)$", "", s.get("place") or "")
    pic = s["picture"]
    for k_, v_ in GLOSS.items():
        pic = re.sub(k_, v_, pic, count=1, flags=re.I)
    room = s.get("room") or ""
    for k_, v_ in GLOSS.items():
        room = re.sub(k_, v_, room, count=1, flags=re.I)
    crowd = (f" In the background, small and still: {s['crowd'].rstrip('.')}." if s.get("crowd") else "")
    cont = crowd + (f" The room: {room.rstrip('.')}." if room else "") + \
           (f" As it is now: {s['now'].rstrip('.')}." if s.get("now") else "")
    return (f"{str(s.get('framing') or 'medium').capitalize()} shot{' in ' + place if place else ''}. {pic}. "
            f"{'. '.join(people)}.{cont}{SETTING.format(place=place) if place else ''} Set in early medieval Northern Europe, "
            f"around 1000 AD: every object, tool and garment hand-made, nothing modern. {style} No text."
            ).replace("..", ".").replace("  ", " ")


def _talk_prompt(R, cx, e: dict, here: str, s: dict | None = None) -> str:
    """The mannequin flow's talk words (motion_prompts): short, "lips and jaw moving clearly", no style paragraph and
    no "everyone else ... mouth closed" (10-24: R.talk_prompt's long prompt with that sentence froze the lips again in
    most of the chapter's talk clips, as it had on 10-22; a talk shot has one person, so it closed the speaker's mouth)."""
    who = e["speaker"]
    look = MF._brief(cx.look(who, e["t"]) or "")
    emo = f", sounding {e['emotion']}," if e.get("emotion") else ","
    place = re.sub(r"\s*\((inside|outside)\)$", "", here)
    at = f" at {place}" if place else ""
    if e.get("creature"):
        return (f"Static camera, the framing does not change. {who}, {look}{at}, speaks{emo} its great jaws opening and "
                f"closing clearly in sync with its voice, the lower jaw dropping on each syllable. "
                f"No text anywhere in the picture.")
    g = R.gender_of(look)
    pron = "her" if g == "woman" else "his" if g == "man" else "their"
    s = s or {}
    lst = ""
    if s.get("listener") and not s.get("ots"):    # a two-shot: the other face is in the picture
        lst = f" {s['listener']} listens with a closed mouth."
    pos = f" on the {'left' if s['who'][0] == who else 'right'}" if len(s.get("who") or []) == 2 else ""
    return (f"Static camera, the framing does not change. {who}{pos}, {look}{at}, speaks{emo} lips and jaw moving "
            f"clearly in sync with {pron} voice.{lst} No text anywhere in the picture.")


def keys(book: Path, stem: str, shots: list, seeds=KEY_SEEDS, log=print) -> None:
    """Every shot's opening picture in `seeds` (key_<seed>.png), both cards drawing at once (one process per card)."""
    sd = shot_dir(book, stem)
    places = {}
    for s in shots:
        if s.get("place") and s.get("place") not in places:
            places[s.get("place")] = next((tuple(x.get("fixtures") or ()) for x in shots
                                           if x.get("place") == s.get("place") and x.get("fixtures")), ())
    heroes = MF.heroes_for(book, places, log)   # all places at once, both cards, judged as they land (10-24)
    heroes.setdefault(None, None)
    jobs = []
    for s in shots:
        d = sd / s["id"].rsplit("_s", 1)[0] / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        lw = LORA_W.get(len(s["who"]), min(LORA_W.values()))
        boxes = [b for b, _ in group_layout(len(s["who"]))]
        crs = _cast(book, stem, {"id": s["id"].rsplit("_s", 1)[0], "who": s["who"]})[1]
        loras = [[MF._lora(book, w)[0], min(lw, CREATURE_LORA) if w in crs else lw, bx]
                 for w, bx in zip(s["who"], boxes) if MF._lora(book, w)[0] and w != s.get("ots")]
        p = _key_prompt(book, stem, s["id"].rsplit("_s", 1)[0], s)
        hero = heroes.get(s.get("place") or None)
        sig = MF._h(json.dumps([p, loras, str(hero)]))
        # a place picture drawn again (10-24: the people-free hero redraws left every key there cached, so the
        # rerun changed nothing): the keys drawn before it are drawn again
        older = hero is not None and hero.exists() and any(k.stat().st_mtime < hero.stat().st_mtime
                                                          for k in d.glob("key_*.png"))
        if older or ((d / "key.sig").exists() and (d / "key.sig").read_text() != sig):
            for k in d.glob("key*.png"):    # the shot changed: its pictures drawn again
                k.unlink()
        (d / "key.sig").write_text(sig)
        for sd_ in seeds:
            if not (d / f"key_{sd_}.png").exists():
                jobs.append({"id": f"{s['id']}_key_{sd_}", "seed": sd_, "size": [W, H], "path": str(d / f"key_{sd_}.png"),
                             "prompt": p, "refs": [str(hero)] if hero else [], "loras": loras})
    if not jobs:
        return
    half = [jobs[0::2], jobs[1::2]]

    def draw(a):
        js, card = a
        if not js:
            return
        jp = sd / f"keys_{card}.json"
        jp.write_text(json.dumps(js, indent=1))
        hip = subprocess.run([os.path.expanduser("~/bin/hip-index"), card], capture_output=True, text=True).stdout.strip()
        r = subprocess.run(["flock", f"/tmp/gpu-{card}.lock", MF.QWEN_PY, "-I", str(HERE / "qwen_image.py"), str(jp),
                            str(sd)], capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
        (sd / f"keys_{card}.log").write_text(r.stdout[-8000:] + r.stderr[-8000:])
    with ThreadPoolExecutor(2) as ex:
        list(ex.map(draw, zip(half, ("05", "65"))))
    miss = [j["id"] for j in jobs if not Path(j["path"]).exists()]
    if miss:
        log(f"!!! {len(miss)} of {len(jobs)} pictures not drawn (shots/keys_*.log): {miss[:4]}")


def _key_questions(cx, s: dict) -> tuple:
    """The vision questions for a shot's opening picture, and how many people it should hold."""
    # (10-24: no "does it show the words" question - it only chose between near-identical clean seeds)
    q = {"text": "Answer yes or no: does this picture contain any written text, letters, a page, a panel or a border?"}
    hum = 0
    for i, w in enumerate(s["who"]):
        lk = cx.look(w, 0) or ""
        if w == s.get("ots"):           # the back of a head: counted, not asked for by face
            hum += 1
            continue
        first = re.sub(r"\s+of about \d+.*$", "", re.split(r",| with | wearing ", lk.strip(), maxsplit=1)[0]).strip().rstrip(".")
        if first:
            q[f"who{i}"] = f"Answer yes or no: is there {first} in this picture?"
        if "four-legged" in MF._appearance(lk) and not MF.DOWN.search(s["picture"]):
            q[f"legs{i}"] = (f"Answer yes or no: is the {'dragon' if 'dragon' in lk else 'creature'}'s body "
                             f"horizontal like a lizard's, with its front feet on the ground (not rearing upright "
                             f"on its hind legs)?")
        hum += not re.search(r"\b(dragon|beast|creature|wyrm|wolf|horse|monster)\b", lk, re.I)
    if hum and not s.get("crowd"):
        q["count"] = "How many people are in this picture? Answer with a number."
    return q, hum


ALL_SEEDS = KEY_SEEDS[:1] + KEY_SEEDS[1:] + MORE_SEEDS     # tried in this order, one at a time


def keys_judged(book: Path, stem: str, shots: list, log=print) -> list:
    """Each shot's opening picture: ONE seed drawn, judged on the iGPU the moment it lands, the next seed only if it
    fails (up to ALL_SEEDS), by kept-loaded drawers on both cards (qwen_image.py --serve) (10-24 user: one seed, the
    image model kept loaded, checked as it is ready; before, 2 seeds were drawn for every shot, then all judged:
    the first seed passed for 142 of 149). key.png = the first clean seed, else the nearest head count.
    -> the shots with no clean seed."""
    import queue
    import threading
    _, cx = _ctx(book, stem)
    sd = shot_dir(book, stem)
    places = {}
    for s in shots:
        if s.get("place") and s.get("place") not in places:
            places[s.get("place")] = next((tuple(x.get("fixtures") or ()) for x in shots
                                           if x.get("place") == s.get("place") and x.get("fixtures")), ())
    heroes = MF.heroes_for(book, places, log)   # all places at once, both cards, judged as they land (10-24)
    heroes.setdefault(None, None)
    q_, tasks, dirty, lock = queue.Queue(), [], [], threading.Lock()
    for s in shots:
        d = sd / s["id"].rsplit("_s", 1)[0] / s["id"]
        d.mkdir(parents=True, exist_ok=True)
        lw = LORA_W.get(len(s["who"]), min(LORA_W.values()))
        boxes = [b for b, _ in group_layout(len(s["who"]))]
        crs = _cast(book, stem, {"id": s["id"].rsplit("_s", 1)[0], "who": s["who"]})[1]
        loras = [[MF._lora(book, w)[0], min(lw, CREATURE_LORA) if w in crs else lw, bx]
                 for w, bx in zip(s["who"], boxes) if MF._lora(book, w)[0] and w != s.get("ots")]
        p = _key_prompt(book, stem, s["id"].rsplit("_s", 1)[0], s)
        hero = heroes.get(s.get("place") or None)
        sig = MF._h(json.dumps([p, loras, str(hero)]))
        # a place picture drawn again (10-24: the people-free hero redraws left every key there cached, so the
        # rerun changed nothing): the keys drawn before it are drawn again
        older = hero is not None and hero.exists() and any(k.stat().st_mtime < hero.stat().st_mtime
                                                          for k in d.glob("key_*.png"))
        if older or ((d / "key.sig").exists() and (d / "key.sig").read_text() != sig):
            for k in d.glob("key*.png"):    # the shot changed: its pictures drawn again
                k.unlink()
        (d / "key.sig").write_text(sig)
        ks0 = MF._seed_keys(d)
        if ks0 and (d / "key.png").exists() and (d / "key.png").stat().st_mtime >= max(k.stat().st_mtime for k in ks0):
            continue                        # picked before and no seed is newer: kept
        q, need = _key_questions(cx, s)
        t = {"s": s, "d": d, "q": q, "need": need, "seeds": list(ALL_SEEDS), "tried": [],
             "job": {"prompt": p, "refs": [str(hero)] if hero else [], "loras": loras, "size": [W, H]}}
        tasks.append(t)
        q_.put(t)
    left = {t["s"]["id"] for t in tasks}

    def settle(t, best, clean):
        if best is not None and not ((t["d"] / "key.png").exists() and
                                     (t["d"] / "key.png").read_bytes() == best.read_bytes()):
            shutil.copy(best, t["d"] / "key.png")
        with lock:
            if not clean:
                dirty.append(t["s"])
            left.discard(t["s"]["id"])
        log(f"{t['s']['id']}: key = {best.name if best else None} ({len(t['tried']) - bool(clean)} of "
            f"{len(t['tried'])} seeds failed)")

    def judge(t, k):
        """True = settled."""
        ans = {}
        _judge([{"id": f"{t['s']['id']}|{k.stem}", "image": str(k), "ask": t["q"]}], ans, first=False)
        a = ans.get(f"{t['s']['id']}|{k.stem}") or {}
        t["tried"].append((k, a))
        if _clean(a, k, t["need"]):
            settle(t, k, True)
            return True
        while t["seeds"] and (t["d"] / f"key_{t['seeds'][0]}.png").exists():
            if judge(t, t["d"] / f"key_{t['seeds'].pop(0)}.png"):     # drawn before: judged first
                return True
        if t["seeds"]:
            q_.put(t)
            return False

        def off(x):
            c = re.search(r"\d+", str(x[1].get("count") or ""))
            return abs(int(c.group(0)) - t["need"]) if c and t["need"] else 0
        settle(t, min(t["tried"], key=off)[0], False)    # none clean: the nearest head count
        return True

    def drawer(card):
        hip = subprocess.run([os.path.expanduser("~/bin/hip-index"), card], capture_output=True, text=True).stdout.strip()
        proc = None
        try:
            while True:
                with lock:
                    if not left:
                        return
                try:
                    t = q_.get(timeout=5)
                except queue.Empty:
                    continue
                seed = t["seeds"].pop(0)
                k = t["d"] / f"key_{seed}.png"
                if not k.exists():
                    if proc is None:
                        proc = subprocess.Popen(["flock", f"/tmp/gpu-{card}.lock", MF.QWEN_PY, "-I",
                                                 str(HERE / "qwen_image.py"), "--serve"], stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                                bufsize=1, env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
                    job = dict(t["job"], id=f"{t['s']['id']}_key_{seed}", seed=seed, path=str(k))
                    proc.stdin.write(json.dumps(job) + "\n")
                    proc.stdin.flush()
                    while True:
                        ln = proc.stdout.readline()
                        if not ln:
                            break
                        try:
                            if json.loads(ln).get("id") == job["id"]:
                                break
                        except ValueError:
                            pass
                if k.exists():
                    jq.put((t, k))          # judged on the iGPU while this card draws the next one
                elif t["seeds"]:
                    log(f"!!! {t['s']['id']}: seed {seed} not drawn")
                    q_.put(t)
                else:
                    settle(t, (t["tried"] or [(None, {})])[0][0], False)
        finally:
            if proc is not None:
                proc.stdin.close()
                try:
                    proc.wait(60)
                except subprocess.TimeoutExpired:
                    proc.kill()

    jq = queue.Queue()

    def judger():
        while True:
            with lock:
                if not left:
                    return
            try:
                t, k = jq.get(timeout=5)
            except queue.Empty:
                continue
            judge(t, k)

    if tasks:
        ths = [threading.Thread(target=drawer, args=(c,)) for c in ("05", "65")] + [threading.Thread(target=judger)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
    return dirty


def pick(book: Path, stem: str, shots: list, log=print) -> list:
    """key.png = the best seed: no text/sheet/border, everyone there, the right number of people, a four-legged
    creature on four legs; the first clean seed. -> the shots with no clean seed."""
    import characters as CH
    _, cx = _ctx(book, stem)
    sd = shot_dir(book, stem)
    items, need = [], {}
    kept = set()
    for s in shots:
        d = sd / s["id"].rsplit("_s", 1)[0] / s["id"]
        ks0 = MF._seed_keys(d)
        if ks0 and (d / "key.png").exists() and (d / "key.png").stat().st_mtime >= max(k.stat().st_mtime for k in ks0):
            kept.add(s["id"])           # picked by an earlier run and no seed is newer: kept, not judged again (10-24)
            continue
        q, need[s["id"]] = _key_questions(cx, s)
        items += [{"id": f"{s['id']}|{k.stem}", "image": str(k), "ask": q} for k in MF._seed_keys(d)]
    # each picture's answers kept beside it (10-24: every run judged all ~300 seeds again, ~20 min of the iGPU);
    # each shot's first seed first, its others only where that one fails (it passed for 142 of 149 shots)
    ans = {}
    _judge(items, ans, first=True)
    bad1 = set()
    for s in shots:
        if s["id"] in kept:
            continue
        k1 = (MF._seed_keys(sd / s["id"].rsplit("_s", 1)[0] / s["id"]) or [None])[0]
        if k1 is None or not _clean(ans.get(f"{s['id']}|{k1.stem}") or {}, k1, need[s["id"]]):
            bad1.add(s["id"])
    _judge([it for it in items if it["id"].split("|")[0] in bad1], ans, first=False)
    dirty = []
    for s in shots:
        if s["id"] in kept:
            continue
        d = sd / s["id"].rsplit("_s", 1)[0] / s["id"]
        ks = MF._seed_keys(d)
        if s["id"] not in bad1:
            ks = ks[:1]
        clean = [k for k in ks if _clean(ans.get(f"{s['id']}|{k.stem}") or {}, k, need[s["id"]])]
        if not clean:   # none clean: the nearest head count (10-24: 3711_s0 fell back to seed 7 with two Edrics)
            def off(k):
                c = re.search(r"\d+", str((ans.get(f"{s['id']}|{k.stem}") or {}).get("count") or ""))
                return abs(int(c.group(0)) - need[s["id"]]) if c and need[s["id"]] else 0
            ks = sorted(ks, key=off)
        best = (clean or ks or [None])[0]
        if best is not None and not ((d / "key.png").exists() and (d / "key.png").read_bytes() == best.read_bytes()):
            shutil.copy(best, d / "key.png")
        if ks and not clean:
            dirty.append(s)
        log(f"{s['id']}: key = {best.name if best else None} ({len(ks) - len(clean)} of {len(ks)} seeds failed)")
    return dirty


def _clean(a: dict, k: Path, need: int) -> bool:
    """A seed's answers pass: no text, no white sheet at the edges, everyone there, legs right, the head count."""
    import characters as CH
    c = re.search(r"\d+", str(a.get("count") or ""))
    return bool(a) and (not CH._yes(a.get("text")) and not MF._blank_edges(k)
                        and all(CH._yes(a.get(x)) for x in a if x.startswith(("who", "legs")))
                        and not (need and c and int(c.group(0)) != need))


def _judge(items: list, ans: dict, first: bool) -> None:
    """Vision answers for the items not answered yet, kept beside each picture; first=True: each shot's first
    seed only."""
    import characters as CH
    if first:
        seen = set()
        items = [it for it in items if not (it["id"].split("|")[0] in seen or seen.add(it["id"].split("|")[0]))]
    todo = []
    for it in items:
        if it["id"] in ans:
            continue
        jf = Path(it["image"]).with_suffix(".judge.json")
        got = json.loads(jf.read_text()) if jf.exists() else {}
        if got.get("q") == it["ask"] and jf.stat().st_mtime >= Path(it["image"]).stat().st_mtime:
            ans[it["id"]] = got["a"]
        else:
            todo.append(it)
    for k0 in range(0, len(todo), JUDGE_BATCH):   # (10-23: the whole chapter's 300 pictures in one call hit its 15 min)
        got = CH.vl_ask(todo[k0:k0 + JUDGE_BATCH])
        ans.update(got)
        for it in todo[k0:k0 + JUDGE_BATCH]:
            if got.get(it["id"]):
                Path(it["image"]).with_suffix(".judge.json").write_text(json.dumps({"q": it["ask"], "a": got[it["id"]]}))


def sheet(book: Path, stem: str, shots: list, out: Path, takes: bool = False) -> Path:
    """The review sheet: per shot its key (or its two takes' middle frames) with the shot's id and action."""
    from PIL import Image, ImageDraw, ImageFont
    sd = shot_dir(book, stem)
    tw, th = 384, 214
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    cols = 4
    rows = math.ceil(len(shots) / cols)
    img = Image.new("RGB", (cols * tw, rows * (th + 60)), "black")
    dr = ImageDraw.Draw(img)
    for n, s in enumerate(shots):
        d = sd / s["id"].rsplit("_s", 1)[0] / s["id"]
        x, y = (n % cols) * tw, (n // cols) * (th + 60)
        src = d / "key.png"
        if takes and (sd / "ltx" / "out" / f"{s['id']}.mp4").exists():
            fr = d / "mid.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s['dur'] / 2:.2f}", "-i",
                            str(sd / "ltx" / "out" / f"{s['id']}.mp4"), "-frames:v", "1", str(fr)])
            src = fr if fr.exists() else src
        if src.exists():
            img.paste(Image.open(src).convert("RGB").resize((tw, th)), (x, y))
        txt = f"{s['id']} {s['dur']:.1f}s {s['kind']}: {s.get('action') or s['picture']}"
        lines = [txt[i:i + 58] for i in range(0, min(len(txt), 232), 58)]
        dr.multiline_text((x + 4, y + th + 2), "\n".join(lines), fill="white", font=font)
    img.save(out, quality=85)
    return out


def ltx(book: Path, stem: str, es: list, shots: list, log=print, redo: set | None = None,
        retake: set | None = None) -> None:
    """The clips, pipelined over both XTXs: action = two-stage in TAKES, quiet = distilled, talk = A2Vid."""
    import flow3d as F
    R, cx = _ctx(book, stem)
    sd = shot_dir(book, stem)
    L = sd / "ltx"
    for d_ in ("out", "emb", "audio"):
        (L / d_).mkdir(parents=True, exist_ok=True)
    for f in ("ltx_split.py", "ltx_split_next.py"):
        shutil.copy(HERE / "ltx_runner" / f, L / f)
    neg = subprocess.run([str(R.LTX_PY), "-c", "import sys; sys.path.insert(0, '/home/jrouth/src/LTX-2/packages/ltx-pipelines/src');"
                          "from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT as N; print(N)"],
                         capture_output=True, text=True).stdout.strip() + R.NEG_EXTRA
    byid = {e["id"]: e for e in es}
    enc_dev, enc_dist, two, quiet, talk = [], [], [], [], []
    for s in shots:
        mid = s["id"].rsplit("_s", 1)[0]
        key = sd / mid / s["id"] / "key.png"
        if not key.exists():
            log(f"{s['id']}: no key")
            continue
        img = [{"path": str(key), "frame": 0, "strength": 1.0}]
        n = _frames(min(s["dur"], SHOT_MAX + 0.5))
        if s["kind"] == "talk":
            e = dict(byid[mid])
            n_all = max(st + nn for st, nn in e["chunks"])
            e["chunks"] = R.talk_chunks(n_all, R.audio_env(Path(e["wav"]), n_all + TALK_PIECE), TALK_PIECE)
            pad = L / "audio" / f"{mid}_pad.wav"
            n_all = max(st + nn for st, nn in e["chunks"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", e["wav"], "-ac", "2", "-ar", "48000", "-af",
                            f"apad=whole_dur={n_all / 24:.4f}", str(pad)], check=True)
            p_ = _talk_prompt(R, cx, e, s.get("place") or "", s)
            emb = f"t_{mid}_{MF._h(p_)}"
            enc_dev += [{"name": emb, "prompt": p_}, {"name": emb + "__neg", "prompt": neg}]
            for k, (st, nn) in enumerate(e["chunks"]):
                talk.append({"name": s["id"] if len(e["chunks"]) == 1 else f"{s['id']}_c{k}", "emb": emb, "prompt": p_,
                             "width": W, "height": H, "fps": FPS, "frames": nn, "audio": str(pad),
                             "seed": int((sd / mid / s["id"] / "talk_seed").read_text())
                             if (sd / mid / s["id"] / "talk_seed").exists() else 42,
                             "audio_start": round(st / 24, 4), "line": mid, "speaker": e.get("speaker"),
                             "speech_s": e.get("speech_s"), "chunk": [k, len(e["chunks"]), st, nn],
                             "images": [dict(img[0], strength=0.8)]})
                if k:       # 10-24 user: the pieces didn't join - each restarted from the key; now each starts on
                    pst, pn = e["chunks"][k - 1]    # the frame of the piece before where their overlap begins
                    talk[-1].update(chain_from=talk[-2]["name"], chain_frame=st - pst, overlap=pst + pn - st,
                                    seam_tries=0,      # user: chained, no re-renders, no checks
                                    images=[])
        elif s["kind"] == "action":
            act = s["action"] + STILL_CROWD * bool(s.get("crowd"))
            emb = f"x_{s['id']}_{MF._h(act + MF.FIGHT_NEG)}"
            enc_dev += [{"name": emb, "prompt": act}, {"name": emb + "__neg", "prompt": neg + MF.FIGHT_NEG}]
            for t, seed in enumerate(TAKES + ((RETAKE,) if s["id"] in (retake or ()) else ())):
                two.append({"name": s["id"] + ("" if t == 0 else f"_t{t + 1}"), "emb": emb, "prompt": act,
                            "width": W, "height": H, "fps": FPS, "seed": seed + 7 * bool(s["id"] in (redo or ())),
                            "frames": n, "line": s["id"], "neg_extra": MF.FIGHT_NEG, "route": "ti2v", "steps": STEPS,
                            "guider": GUIDER, "aguider": GUIDER, "images": img})
        else:
            p_ = (s["action"] or "Small, natural movements; the camera holds still.") + STILL_CROWD * bool(s.get("crowd"))
            emb = f"q_{s['id']}_{MF._h(p_)}"
            enc_dist.append({"name": emb, "prompt": p_})
            quiet.append({"name": s["id"], "emb": emb, "prompt": p_, "width": W, "height": H, "fps": FPS, "seed": 7,
                          "frames": n, "line": s["id"], "route": "i2v", "images": img})
    for j in talk + two + quiet:            # a clip from another key or other words is drawn again
        c, rj = L / "out" / f"{j['name']}.mp4", L / "out" / f"{j['name']}.json"
        src = L / "out" / f"{j.get('chain_from')}.mp4"
        key = Path(j["images"][0]["path"]) if j["images"] else src
        stale = c.exists() and (not key.exists() or c.stat().st_mtime < key.stat().st_mtime or j["line"] in (redo or ()))
        if rj.exists() and c.exists():
            old = json.loads(rj.read_text())
            stale = stale or old.get("prompt") != j["prompt"] or old.get("seed") != j["seed"] or \
                old.get("frames") != j["frames"] or old.get("chain_from") != j.get("chain_from")
        if stale:
            c.unlink()
    for nm, x in (("enc_dev", enc_dev), ("enc_dist", enc_dist), ("talk_jobs", talk), ("two_jobs", two),
                  ("quiet_jobs", quiet)):
        (L / f"{nm}.json").write_text(json.dumps(x, indent=1))
    from types import SimpleNamespace
    # ONE long hold per batch (10-23: 15-minute holds reloaded the DiT every 3 clips, +35 s a clip); the runner still
    # yields to a waiting GPU job (LTX_YIELD)
    R.HOLD_TIMEOUT = 4 * 3600
    R.PP = True                         # the fast runner set itself (lanes, stagger, pool, decode-b, split) = R.LTX_PP_ARGS
    if not PP_POOL:
        os.environ.setdefault("REMAKE_LTX_PP_ARGS", R.LTX_PP_ARGS.replace(" --pp-pool", ""))
    cxl = SimpleNamespace(OUT=sd)

    def step(args, name):
        rc = R.ltx_step(cxl, name, args)
        log(f"ltx {name}: rc {rc}")
        return rc

    def gen(kind, jobs, args, split):
        os.environ["REMAKE_LTX_SPLIT_TALK"] = os.environ["REMAKE_LTX_SPLIT_ACT"] = str(split)
        for _ in range(3):                  # a hold that died part-way: again for the rest
            if all((L / "out" / f"{j['name']}.mp4").exists() for j in jobs):
                return
            step(["gen", f"{kind}_jobs.json", *args, "--out", "out", "--budget", str(R.HOLD_TIMEOUT - 600)], f"gen_{kind}")
        left = [j["name"] for j in jobs if not (L / "out" / f"{j['name']}.mp4").exists()]
        if left:
            log(f"!!! ltx {kind}: {len(left)} clip(s) not made: {left[:6]}")
    if enc_dev:
        R.encode_chunked(cxl, L / "enc_dev.json", "enc_dev", ["--dev"])
    if enc_dist:
        R.encode_chunked(cxl, L / "enc_dist.json", "enc_dist", [])
    if two:
        # the take judge runs on the iGPU while the XTXs render (10-24 user: batch the re-do work, interleave the
        # checks): each first take is judged as it lands, a faulted one's retake joins the loaded runner's queue
        more, done = L / "two_more.json", L / "two_more.done"
        more.write_text("[]")
        done.unlink(missing_ok=True)
        stop = threading.Event()
        th = threading.Thread(target=_judge_as_they_land, args=(L / "out", two, more, done, stop, log), daemon=True)
        th.start()
        try:
            gen("two", two, ["--ti2v", "--fa-amd", "--blaslt", "--more", more.name, "--more-done", done.name],
                F._pp_split(sd, "ti2v", 24))
        finally:
            stop.set()
            th.join(60)
    if talk:
        gen("talk", talk, ["--a2vid", "--fa-amd", "--blaslt"], F._pp_split(sd, "a2vid", 25))
        if any(len(byid[s["id"].rsplit("_s", 1)[0]].get("chunks") or []) > 1 for s in shots if s["kind"] == "talk"):
            subprocess.run([str(R.LTX_PY), str(HERE / "ltx_runner" / "stitch_talk.py"), "talk_jobs.json", "out"],
                           cwd=str(L), capture_output=True)
    if quiet:
        gen("quiet", quiet, ["--fa-amd", "--blaslt", *(["--decode-b"] if DECODE_B else [])], F._pp_split(sd, "distilled", 17))


def judge_talk(book: Path, stem: str, es: list, shots: list, log=print) -> set:
    """Talk clips whose speaker's mouth never opens (10-24 user: "most of the talking shots don't have lips moving";
    the short talk words fixed 3 of 4, Hulda's 0:90 line stayed shut): three frames inside the speech asked "is the
    mouth open"; none -> the moment is drawn again with another seed (once; talk_seed keeps it). -> those moments."""
    import characters as CH
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)   # the judge's iGPU
    sd = shot_dir(book, stem)
    out_ = sd / "ltx" / "out"
    byid = {e["id"]: e for e in es}
    _, cx = _ctx(book, stem)
    items, judged = [], []
    for s in shots:
        mid = s["id"].rsplit("_s", 1)[0]
        c, mk = out_ / f"{s['id']}.mp4", out_ / f"{s['id']}.talk"
        if s["kind"] != "talk" or mid not in byid or byid[mid].get("creature") or not c.exists() or \
                (mk.exists() and mk.stat().st_mtime >= c.stat().st_mtime):
            continue
        who = byid[mid].get("speaker") or ""
        first = re.split(r",| with | wearing ", (cx.look(who, 0) or "the person").strip(), maxsplit=1)[0].strip()
        side = f" on the {'left' if s['who'][0] == who else 'right'}" if len(s.get("who") or []) == 2 else ""
        sp = float(byid[mid].get("speech_s") or s["dur"])
        for i, f in enumerate((0.3, 0.5, 0.7)):
            png = out_ / f"{s['id']}_m{i}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{sp * f:.2f}", "-i", str(c), "-frames:v", "1",
                            str(png)])
            if png.exists():
                items.append({"id": f"{s['id']}|{i}", "image": str(png), "ask": {
                    "open": f"Answer yes or no: is the mouth of {first}{side} open, lips apart?"}})
        judged.append((s, mid, mk))
    ans = {}
    for k0 in range(0, len(items), JUDGE_BATCH):
        ans.update(CH.vl_ask(items[k0:k0 + JUDGE_BATCH]))
    bad = set()
    for s, mid, mk in judged:
        if any(CH._yes((ans.get(f"{s['id']}|{i}") or {}).get("open")) for i in range(3)):
            mk.write_text("open")
            continue
        seed_p = sd / mid / s["id"] / "talk_seed"
        if seed_p.exists():
            log(f"!!! {s['id']}: the mouth stays shut in the second seed too")
            mk.write_text("shut")
            continue
        seed_p.write_text("49")
        bad.add(mid)
        log(f"{s['id']}: the mouth never opens: drawn again with another seed")
    return bad


def _judge_as_they_land(out_: Path, jobs: list, more: Path, done: Path, stop, log) -> None:
    """Judge each action shot's first take as soon as its clip exists (the vision judge stays loaded); a faulted one
    gets its retake job (seed RETAKE, name <id>_t2) appended to `more`; `done` once every first take is judged (or
    the render stopped). judge_takes() afterwards keeps the better take of each pair."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)   # the judge's iGPU
    try:
        firsts = [j for j in jobs if not re.search(r"_t\d+$", j["name"])]
        names = {j["name"] for j in jobs}
        left = {j["name"]: j for j in firsts}
        while left and not stop.is_set():
            ready = [n for n in left if (out_ / f"{n}.mp4").exists()]
            if not ready:
                time.sleep(10)
                continue
            for n in ready:
                j = left.pop(n)
                if (out_ / f"{n}.judged").exists() or f"{n}_t2" in names:
                    continue
                fa = MF.fight_clip_ok(out_ / f"{n}.mp4")
                if not fa:
                    (out_ / f"{n}.judged").write_text(json.dumps({"take1": fa}))
                    continue
                log(f"{n}: {fa}: taken once more (queued on the loaded model)")
                extra = json.loads(more.read_text())
                extra.append(dict(j, name=f"{n}_t2", seed=RETAKE))
                tmp = more.with_suffix(".tmp")
                tmp.write_text(json.dumps(extra, indent=1))
                os.replace(tmp, more)
    except Exception as e:                  # noqa: BLE001  (the render goes on; judge_takes catches up after)
        log(f"!!! take judge while rendering: {e!r}")
    finally:
        done.write_text("done")


def judge_takes(book: Path, stem: str, shots: list, log=print) -> set:
    """Each action shot: its takes judged (melted / merged bodies, fire from hands); the take with the fewest faults
    becomes <id>.mp4 (a tie keeps the first), the other stays as <id>_t2.mp4. -> the shots whose only take is
    faulted (to be taken once more)."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)   # the judge's iGPU
    out_ = shot_dir(book, stem) / "ltx" / "out"
    again = set()
    for s in shots:
        if s["kind"] != "action":
            continue
        a, b = out_ / f"{s['id']}.mp4", out_ / f"{s['id']}_t2.mp4"
        if not a.exists() or (out_ / f"{s['id']}.judged").exists():
            continue
        fa = MF.fight_clip_ok(a)
        if not b.exists():
            if fa:
                log(f"{s['id']}: {fa}: taken once more")
                again.add(s["id"])
            else:
                (out_ / f"{s['id']}.judged").write_text(json.dumps({"take1": fa}))
            continue
        fb = MF.fight_clip_ok(b)
        if len(fb) < len(fa):               # swap: the better take under the shot's own name
            tmp = a.with_suffix(".swap.mp4")
            os.replace(a, tmp)
            os.replace(b, a)
            os.replace(tmp, b)
        (out_ / f"{s['id']}.judged").write_text(json.dumps({"take1": fa, "take2": fb}))
        log(f"{s['id']}: take {'2' if len(fb) < len(fa) else '1'} kept ({fa} / {fb})")
    return again


def _impacts_in_clip(mp4: Path, fps: int = FPS) -> list:
    """Frames where a fast motion stops dead (a hammer landing): the picture's own frame-to-frame change rises to a
    peak and falls by most of it within two frames."""
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", "scale=96:54,format=gray", "-f", "rawvideo",
                          "-"], capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(-1, 54, 96).astype(np.float32)
    if len(a) < 6:
        return []
    m = np.abs(np.diff(a, axis=0)).mean(axis=(1, 2))
    base = float(np.median(m)) + 1e-3
    hits, last = [], -99
    for i in range(1, len(m) - 2):
        if m[i] >= m[i - 1] and m[i] > 2.5 * base and min(m[i + 1], m[i + 2]) < 0.4 * m[i] and i - last >= fps // 3:
            hits.append(i + 1)
            last = i
    return hits


def assemble(book: Path, stem: str, es: list, shots_by: dict, mix: Path, out: Path, log=print) -> Path:
    """The shots in time order: each moment's whole frames (counted from the stretch's start, so the picture never
    drifts from the voice) split over its shots by their lengths; a missing clip = its key held."""
    sd = shot_dir(book, stem)
    L = sd / "ltx"
    t0 = float(es[0]["t"])
    span = _spans(es)
    edge = [round((float(e["t"]) - t0) * FPS) for e in es] + [round((float(es[-1]["t"]) + span[es[-1]["id"]] - t0) * FPS)]
    segs, hits, f_at = [], [], 0
    for i, e in enumerate(es):
        n_all = max(edge[i + 1] - edge[i], 1)
        shots = shots_by.get(e["id"]) or []
        tot = sum(s["dur"] for s in shots) or 1.0
        cum, prev_end = 0.0, 0
        for s in shots:
            cum += s["dur"]
            end = round(n_all * cum / tot)
            nf = max(end - prev_end, 1)
            prev_end = end
            c = L / "out" / f"{s['id']}.mp4"
            key = sd / e["id"] / s["id"] / "key.png"
            sg = L / f"seg_{s['id']}.mp4"
            if c.exists():
                src, vf = ["-i", str(c)], "tpad=stop_mode=clone:stop_duration=120"
                if s["kind"] != "talk" and STRIKE.search(s.get("action") or ""):
                    hits += [(f_at + h) / FPS for h in _impacts_in_clip(c) if h < nf]
            else:
                log(f"!!! {s['id']}: no clip: " + ("its key held" if key.exists() else "black"))
                src = ["-loop", "1", "-i", str(key)] if key.exists() else \
                    ["-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r=24"]
                vf = f"scale={W}:{H}"
            subprocess.run(["ffmpeg", "-v", "error", "-y", *src, "-vf", vf + ",format=yuv420p", "-frames:v", str(nf),
                            "-an", "-c:v", "libopenh264", "-b:v", "6M", "-r", str(FPS), str(sg)], check=True)
            segs.append(sg)
            f_at += nf
    lst = L / "segs.txt"
    lst.write_text("".join(f"file '{sg}'\n" for sg in segs))
    vid = L / "video_only.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(vid)],
                   check=True)
    got = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                              "stream=nb_read_packets", "-of", "csv=p=0", str(vid)], capture_output=True, text=True).stdout or 0)
    if got != edge[-1]:
        log(f"!!! picture is {got} frames, the timeline {edge[-1]}: out of sync by {(got - edge[-1]) / FPS:+.2f} s")
    return MF.mux(book, stem, vid, t0, edge[-1] / FPS, mix, out, hits, len(segs), log)


def out_path(book: Path, stem: str, t0: float, t1: float) -> Path:
    p = MF.out_path(book, stem, t0, t1)
    return p.with_name(p.name.replace("_mannequin", "_shots"))


def run(book: Path, stem: str, t0: float, t1: float, hero: Path | None, mix: Path, log=print,
        until: str | None = None, guide: str | None = None) -> Path:
    """The whole stretch: shots (director) -> keys (both cards) -> pick (+3 seeds where none passed) -> [until=keys:
    the review sheet] -> LTX (pipelined) -> takes judged -> the takes sheet -> assembled."""
    es = MF.entries(book, stem, t0, t1)
    op = out_path(book, stem, t0, t1)
    # 10-24 user: "you should have gone right back to rendering" - a render that was stopped resumes on the plan it
    # had (shots + keys), nothing planned, drawn or checked again; the plan is dropped once the cut is assembled
    fz = shot_dir(book, stem) / f"plan_{op.stem}.json"
    sig = _plan_sig(book, stem)
    if until != "keys" and fz.exists() and json.loads(fz.read_text()).get("_sig") not in (None, sig):
        log(f"the saved plan is from other looks or LoRAs: planning again")
        fz.unlink()
    if until != "keys" and fz.exists():
        by = json.loads(fz.read_text())
        by.pop("_sig", None)
        shots = [s for e in es for s in by.get(e["id"], [])]
        log(f"resuming the stopped render on its saved plan ({len(shots)} shots; delete {fz} to plan again)")
        return _render(book, stem, es, by, shots, mix, op, fz, log)
    if MF.cover(book, stem, t0, t1, log):  # narrated stretches without a moment get theirs
        es = MF.entries(book, stem, t0, t1)
    wardrobe(book, stem, log)
    sheets = scene_sheets(book, stem, es, log)
    by = direct(book, stem, es, log, sheets)
    shots = [s for e in es for s in by.get(e["id"], [])]
    for s in shots:
        s.update({k: v for k, v in (sheets.get(s["id"].rsplit("_s", 1)[0]) or {}).items()
                  if k in ("room", "now", "fixtures", "scene") and not sheets[s["id"].rsplit("_s", 1)[0]].get("elsewhere")})
        if not s.get("crowd") is True or sheets[s["id"].rsplit("_s", 1)[0]].get("elsewhere"):
            # the director's true/false -> the scene's crowd words, or none
            s["crowd"] = ""
        else:
            s["crowd"] = (sheets.get(s["id"].rsplit("_s", 1)[0]) or {}).get("crowd") or ""
    describe_places(book, stem, es, shots, log)
    log(f"{len(es)} moments -> {len(shots)} shots ({sum(s['kind'] == 'action' for s in shots)} action, "
        f"{sum(s['kind'] == 'quiet' for s in shots)} quiet, {sum(s['kind'] == 'talk' for s in shots)} talk)")
    for s in keys_judged(book, stem, shots, log):
        log(f"!!! {s['id']}: no clean picture in {len(ALL_SEEDS)} seeds")
    ks = sheet(book, stem, shots, op.with_name(op.stem + "_keys.jpg"))
    log(f"key sheet: {ks}")
    if until == "keys":
        return ks
    fz.write_text(json.dumps({"_sig": sig, **{s_["id"].rsplit("_s", 1)[0]: [x for x in shots if x["id"].rsplit("_s", 1)[0] ==
                                                                        s_["id"].rsplit("_s", 1)[0]] for s_ in shots}}))
    return _render(book, stem, es, by, shots, mix, op, fz, log)


def _plan_sig(book: Path, stem: str) -> str:
    """The looks + LoRAs a saved plan's keys were drawn with (10-25: a resume after `look --swap` would have rendered
    the old keys). The chapter plan's own copy of the looks + each LoRA's build hash and time: 10-25 22:26 Acts 2 went
    to resume keys with the white-and-blue Disciple, as look.json had changed first and a retrain keeps the file name."""
    vd = book / "work" / "video"
    lk = json.loads((vd / "look.json").read_text()).get("characters", {}) if (vd / "look.json").exists() else {}
    pp = vd / stem / "plan.json"
    plk = (json.loads(pp.read_text()).get("look") or {}).get("characters", {}) if pp.exists() else {}
    loras = sorted((bj.parent.name, *(json.loads(bj.read_text()).get(k) for k in ("lora", "hash", "built")))
                   for bj in vd.glob("charlora/*/build.json"))
    return MF._h(json.dumps([lk, plk, loras], sort_keys=True))


def _render(book, stem, es, by, shots, mix, op, fz, log) -> Path:
    """LTX (pipelined) -> takes judged -> the takes sheet -> assembled; the saved plan goes with the finished cut."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    # the iGPU's language model holds ~90 GB the LTX memory floor needs (10-25: a resume found it restarted by the
    # stopped job's server restore and refused, "MemAvailable 14 GiB < 15"); the planning steps stop it, a resume didn't
    if subprocess.run(["systemctl", "--user", "is-active", "-q", animate.STRATA_UNIT]).returncode == 0:
        log("stopping the language model (the render needs its memory)")
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
        time.sleep(5)
        animate._flush()
    ltx(book, stem, es, shots, log)
    again = judge_takes(book, stem, shots, log)
    if again:
        ltx(book, stem, es, [s for s in shots if s["id"] in again], log, retake=again)
        judge_takes(book, stem, shots, log)
    # 10-24: judge_talk no longer re-renders - it failed all 7 forge lines and the user judged every one good
    log(f"takes sheet: {sheet(book, stem, shots, op.with_name(op.stem + '_takes.jpg'), takes=True)}")
    out = assemble(book, stem, es, by, mix, op, log)
    fz.unlink(missing_ok=True)
    return out
