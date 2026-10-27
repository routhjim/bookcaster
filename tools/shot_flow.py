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
QUIET_SEED = 7                      # a quiet shot's first take (its retake: RETAKE)
PP_POOL = True                      # activations between the cards through reused pinned buffers (runner --pp-pool)
DECODE_B = True                     # quiet clips decode on card 65 (runner --decode-b); on once its A/B test passes
TALK_PIECE = 97                     # talk lines in 3-4 s pieces (73-97 frames), chained (10-24 user; was <= 5 s)
# 10-26 user ("Peter's sermon just gets further and further out ... the last frame from one 3s clip should be the
# starting frame for the next"): each piece starts ON the previous piece's last frame (lossless, from the render, not
# the mp4) and the stitch hands over there with no crossfade; was a 12-frame overlap = 0.5 s of two blended mouths at
# every seam and an mp4 round trip per piece (sharpness 1.4-5x and drift from the key growing piece after piece)
TALK_CHAIN_OVERLAP = 1
KEY_SEEDS, MORE_SEEDS = (7, 11), (13, 17, 23)
STEPS = int(os.environ.get("SHOTS_STEPS", "20"))      # two-stage stage-1 steps (10-23 test: 20 = 30 by eye)
GUIDER = json.loads(os.environ.get("SHOTS_GUIDER", "{\"modality_scale\": 1.0}"))   # the audio-video pass off (we keep our own sound): 105 -> 64 s a 3 s clip with 2 in flight, same picture by eye (10-23)
JUDGE_BATCH = 60                    # pictures per vision-judge call (each call has a 15 min limit)
# 10-27 user: "try out thinking on medium" (every director ask had thinking off). Measured on the Speckled Band: on
# the shots, 5-6x the time for richer pictures but the same scene mistakes -> off; on for the one scene breakdown
DIRECT_EFFORT = os.environ.get("BOOKCASTER_DIRECT_EFFORT", "")
BREAKDOWN_EFFORT = os.environ.get("BOOKCASTER_BREAKDOWN_EFFORT", "medium")
DIRECT_WORKERS = 3                  # director requests in flight (Strata serves them in its slots)

# 10-27 user ("the whole point of the director is to break up the story into individual shots... it needs more context
# right?"): the director was asked one 5 s moment at a time with only the sentences STARTING in it - 10 of the Speckled
# Band's first 19 moments had none and got a stale camera note instead ("Watson blinks up at Holmes": he lay down
# again; "Roylott walks across the lawn of Stoke Moran" while Holmes sat in Baker Street), the 67 s opening asked for 14
# shots at once and came back empty three times (one frozen title card). Now it gets a whole PASSAGE (the narration
# between two spoken lines, <= PASSAGE_MAX s), every sentence timed and tagged with who says it, the story just before
# and after, where the people really are, and the last shot before it; it splits the passage into shots itself.
# 10-27 user ("couldn't the director just run over the entire passage ... all at once"): the director is ONE running
# conversation through the chapter - these rules once, then a turn per passage with the answer kept - so it sees its own
# earlier shots in full (not an 80-word state) and Strata reads only the new turn (the prefix is cached). One giant ask
# was the frozen-title-card failure; a turn per passage keeps retries and the cache per passage.
SHOT_RULES = """You direct a film of a book. You get the story one PASSAGE at a time, in order, narration and spoken lines
together, every line timed. YOU break each passage into the film's SHOTS: where each cut falls, what each shot shows,
who is in it. Each shot starts from ONE painted picture of its opening and a video model animates it from your ACTION
words alone, so every shot must be simple, physical and unambiguous. Your shots for the earlier passages are above;
carry the film on from them. Every passage belongs to one SCENE of the chapter's scene list: its shots are at
that scene's place, or at a place the scene may cut to.

Rules:
- Cover the whole passage in order with about the number of shots it asks for. Give each shot its "start" (seconds
  from the passage's start): the first starts at 0.0, each later one after the one before, the last before the
  passage's end. A shot starting at second X shows what the narration is telling at second X.
- Every SPOKEN line is covered by "talk" shots of the one who says it: their lips are synced to the voice. Nobody
  speaks in any other shot. A talk shot starts when its line starts and lasts at most 8 seconds: a longer line cuts
  to a close reaction of the listener and back to the speaker. Give "speaker", "listener" (the one spoken to, or "")
  and "framing": "two" (both in the picture, waist up, facing each other: the first line of an exchange),
  "over-the-shoulder" (a reply in a running exchange) or "close". A talk shot is always where the speaker really is,
  never in a place someone describes. No picture or action words.
- Every other shot is 2 to 5 seconds long. Show what the passage says is HAPPENING to the people in it: who does what,
  where, how they react. When it describes a person (their face, clothes, manner), show that person in the moment. Each
  shot shows ONE physical action (one move, one gesture, one exchange) or a quiet held moment, played through to its
  result. A shot may show a thing alone (a clock, a letter, a door) when the passage looks at it.
- CONTINUITY: the first shot of a passage continues from the last shot before: the same people in the same places,
  clothes and postures, unless the passage changes them. Nobody lies down, sits or stands up again unless the passage
  says so.
- Show the characters TOGETHER. When two are here, most shots have both in the picture, one on each side, each turned
  toward the other. When three or more are here, open with a WIDE shot of them (up to {maxp}), where each stands and
  which way each faces; then the two or three who act or speak, and close reactions. A wide shot holds up to {maxp}; a
  medium or close shot at most {closemax}. One person alone only for a close reaction or a detail.
- WHEN: nothing of an event (an arrival, a light, a sound, a change) is in any shot before the second its sentence
  starts.
- "place": where the shot is, one of the passage's places. "framing": "wide", "medium" or "close".
- "people": everyone in the OPENING picture, left to right, only from the passage's names, each with "name", "side"
  ("left", "centre" or "right"), "pose" (standing, sitting, lying, kneeling or crouching: as the passage says, someone
  leaning back in a chair is sitting), "at" (where on the set: "in the armchair by the fire", "at the writing desk",
  "in bed"), "facing" (toward whom or what) and "holds" (what is in their hands, or ""). [] for a thing alone.
- "setting": one sentence for the rest of the OPENING picture: the things and light the shot needs, every object the
  action will use. Never a person's position or posture here (they are in "people"), nothing in mid-action, never
  writing, a title or a caption: the video model can't draw letters.
- "action": what moves in the shot, in order, plain physical verbs, played through. Every effect names its source. A
  change over time is written as visible steps. Say what the hands hold. Call each one by a plain description (a tall
  man in a dark frock coat), never by name. No magic words (magic, enchanted, spell), no camera directions, no
  written words or titles. At most 50 words.
- "pose_end": each person's pose at the shot's END, one word each. A pose carries over from the end of the shot before and changes
  only in a shot whose action changes it (he is pulled to his feet: "pose" sitting, "pose_end" standing). Someone the
  narration says cannot stand or walk stays sitting or lying in every shot until the narration changes that.
- "kind": "talk" (above), "action" when a body or object moves with force (a strike, a fall, a lunge, a hammer blow),
  else "quiet" (breathing, looking, a gesture, a few steps).
- "state_end": how things stand at the END of the passage: each person's clothes (what they wear now), what each
  holds, each one's posture and where they are, and any visible change to the place (a lamp lit, a door open, a letter
  on the table). Plain words, at most 80.
Reply to each passage with JSON only: {{"state_end": "...", "shots": [{{"start": 0.0, "kind": "quiet", "place": "...", "framing": "...", "people": [{{"name": "...", "side": "left", "pose": "sitting", "at": "...", "facing": "...", "holds": ""}}], "setting": "...", "action": "...", "pose_end": {{"<name>": "sitting"}}, "crowd": false}}, {{"start": 4.2, "kind": "talk", "speaker": "...", "listener": "...", "framing": "two"}}]}}"""
PASSAGE_ASK = """{scene}
The story just before: {before}
THE PASSAGE ({sec:.1f} seconds). Each line: [seconds from the passage's start] who: the words. Spoken lines say
"SAYS ALOUD" with their start and end.
{text}
Right after it (only so you know where the story goes: none of it is in this passage's shots): {after}

{narrator}Where the people are now: {home}. Its fixed things (keep each where it stands): {room}
{told}Who may appear: {people}
As things stand when this passage starts (carry them on; change only what the passage changes): {now}
The last shot before this passage: {prev}
About {nmin} to {nmax} shots. Places: {places}. Names: {names}.
{extra}{fix}"""
# the rules for what only some passages have, given only when the passage has it (10-27: forge, dragon and
# tongues-of-fire rules were a third of every Speckled Band prompt)
RULE_TOLD = ("Other places in this passage: {others}. When its words tell of another time or place (someone recounting "
             "what happened, a memory), the non-talk shots may show those events there, with only the people the words "
             "describe, while the talk shots stay with the teller where the people are now. When the people of the story "
             "really go there, they are there.\n")
RULE_EXTRA = [
    (re.compile(r"\b(fire|flames?|blaz\w*|smok\w*|wind|gusts?|gale|lightning|tongues?\s+(?:as\s+)?of)\b", re.I),
     "- A passing effect (fire, flames, smoke, wind, lightning) is in the shots of the sentence that describes it and the\n"
     "  shots after it in THIS passage only; never before it, and not carried in from before.\n"),
    (re.compile(r"\btongues?\b|\bas if\b|\blike a\b", re.I),
     "- Figurative words are drawn as what a viewer would literally see, never the figure of speech: \"tongues of fire\"\n"
     "  are small flames resting above the heads; people \"speaking in tongues\" or in other languages talk normally and\n"
     "  eagerly. Never draw a tongue (or any body part) an idiom names, and no wide-open mouths or working jaws.\n"),
    (re.compile(r"\b(fight\w*|fought|strik\w*|struck|blows?|sword\w*|attack\w*|hit|lunge\w*|slash\w*|stab\w*|claws?|"
                r"jaws|spear\w*|axe|arrow\w*|wound\w*|dead|kill\w*|slain|fell)\b", re.I),
     "- A fight is covered as intent (wide: both fighters, the attacker coils, the other sees the opening), action (both\n"
     "  in the picture: the one blow or move and the one it reaches), reaction (close: the one hit, or a witness). A\n"
     "  fighter's picture has a READY stance, weapon raised toward the opponent; someone knocked down but alive is on the\n"
     "  ground propped on an elbow, eyes open; someone dead lies flat, eyes closed; a four-legged creature on its four legs.\n"),
    (re.compile(r"\b(forg\w*|hammer\w*|anvil|smith\w*|(?:made|built|folded)\s+(?:a|an|the|it|him|her|them)\b)", re.I),
     "- When the narration says someone MADE, forged, folded or built something, the shots show the WORK itself (heating\n"
     "  it, folding it, hammering it), never the finished thing being admired or worn: something being made is shown\n"
     "  being worked on (on the anvil, in the tongs, in the hands).\n"),
]
RULE_CROWD = ("- \"crowd\": true when the scene's crowd ({crowd}) is in the picture, in the background, small and still; "
              "else false.\n")
WRITTEN = re.compile(r"\b(title card|titles?|typography|lettering|letters? reading|text reading|reads? ['\"]|written words?|"
                     r"inscription|caption)\b", re.I)          # 10-27: the title card came out "Foeckled Band"
# 10-27: "camera pushes in" got past the rules
CAMERA = re.compile(r"\b(camera|zoom\w*|dolly|close-?up|push(?:es|ing)? in|pull(?:s|ing)? back to)\b", re.I)
# a planner place that is no place ("Title Card": the Speckled Band's first minute was 17 shots of a dark background)
NO_PLACE = re.compile(r"\b(title|card|credits?|caption|intro\w*)\b", re.I)
SPEAKS = re.compile(r"\b(speak\w*|spoke|talk\w*|says?|said|lips|mouth\w*|tell\w*|explain\w*|recount\w*)\b", re.I)
HISTORY_CHARS = 60000               # the director's conversation (~15k tokens) before it goes on from the last two passages
SHOT_SHORT = 1.5                    # a shot shorter than this is folded into the one before (10-27: 0.9 and 1.0 s shots)
PASSAGE_MAX = 30.0                  # seconds of narration per director request (6-15 shots; 10-27: 67 s came back empty)


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
# figures of speech the director and the video model took literally (10-26 Acts 2: "tongues of fire" drawn as tongues,
# "speaking in tongues" as wide-open mouths with working jaws): the literal sight, in every picture and action prompt
FIGURATIVE = {r"\b(?:divided\s+)?tongues?\s+(?:as\s+)?of\s+(?:fire|flame)s?\b": "small flames",
              r"\btongue-like\s+flames?\b|\bflame-like\s+tongues?\b": "small flames",
              r"\bspeak(?:ing|s)?\s+in\s+(?:other\s+|many\s+|strange\s+)?tongues\b": "talking eagerly"}
TONGUE = re.compile(r"\btongues?\b", re.I)
# passing effects: only from the sentence that describes them, in its own stretch (10-26 Acts 2: flames from 0:00 and on
# Peter's head out in the street)
PASSING = {"fire": re.compile(r"\b(fire|flames?|flame-like|blaz\w*|burning|embers?)\b", re.I),
           "smoke": re.compile(r"\bsmok\w*\b", re.I), "wind": re.compile(r"\b(wind|gusts?|gale)\b", re.I),
           "lightning": re.compile(r"\blightning\b", re.I)}
# a fire that belongs to the place (a forge, a hearth) stays lit; any other passing effect in a scene sheet's states is
# drawn only in the moments whose narration has it (10-26 Acts 2 v3: "Divided Tongues" was a state from the first
# moment on, so flames sat on every head through the sermon and the baptisms)
HELD_EFFECT = re.compile(r"\b(hearth|forge|fireplace|brazier|torch\w*|lamps?|lanterns?|candles?|ovens?|kilns?|stoves?|"
                         r"grates?|chimney|campfire|bonfire|pyre)\b", re.I)


def passing_shown(state: str, said: str) -> bool:
    """Whether a scene state shows in a moment with this narration: a passing effect (PASSING) only where it is said."""
    if HELD_EFFECT.search(state):
        return True
    for g, rx in PASSING.items():
        if rx.search(state) or (g == "fire" and TONGUE.search(state)):
            return bool(rx.search(said) or (g == "fire" and re.search(r"\btongues?\s+(as\s+)?of\b", said, re.I)))
    return True
# body positions, carried shot to shot (10-26 Acts 3: the lame man was drawn up, down and up again before he was
# healed; the keys and clips are told each person's held pose and the key judge checks the not-standing ones)
POSE_CAT = (("lie", re.compile(r"\b(ly\w*|lie\w*|lain|prone|sprawl\w*|flat on)\b", re.I)),
            ("kneel", re.compile(r"\bkneel\w*|\bknelt\b", re.I)),
            ("crouch", re.compile(r"\b(crouch\w*|squat\w*|hunch\w*)\b", re.I)),
            ("sit", re.compile(r"\b(sit\w*|sat|seated)\b", re.I)),
            ("stand", re.compile(r"\b(stand\w*|stood|upright|walk\w*|on (his|her|their) feet)\b", re.I)))
# plain postures: the shot's picture says where (in a chair, on a bed, on the ground). 10-27: "sitting on the ground"
# (from Acts' lame man) put Watson on the floor of Baker Street and failed every seed with him in his chair
POSE_WORDS = {"lie": "lying down", "kneel": "kneeling", "crouch": "crouching low", "sit": "sitting", "stand": "standing"}
# what a look CARRIES ("..., with a revolver", "carrying a hunting crop"): not clothes; in a picture only when the shot
# calls for it (10-27: Watson's revolver in every Baker Street key, and 28 outfit fails for not showing it)
CARRIED = re.compile(r"(?:,\s*|\s+and\s+|\s+)\b(?:carrying|holding|with an?|armed with)\b[^,;]*", re.I)


def uncarried(look: str, picture: str = "") -> str:
    """The look without what it carries, unless the shot's picture names that thing."""
    def keep(m):
        thing = re.sub(r"^\W*(carrying|holding|with an?|armed with)\s+", "", m.group(0).strip(" ,"), flags=re.I)
        head = (thing.split() or [""])[-1]
        return m.group(0) if head and re.search(rf"\b{re.escape(head)}", picture or "", re.I) else ""
    return re.sub(r"\s*,\s*$", "", CARRIED.sub(keep, look or "")).strip()


def no_pose(look: str) -> str:
    """A look without its posture clauses (10-26: the lame man's look ends "sitting on the ground with legs extended",
    still said after he was healed); the shot's held pose is the only posture a picture or clip is told."""
    parts = [x for x in re.split(r",\s*", str(look or "")) if x.strip()]
    return ", ".join(x for x in parts if not (pose_cat(x) and not re.search(r"\bwearing\b", x)))


def video_dress(cx, who: str, t: float) -> str:
    """The look in force at t (the wardrobe's dress if any, clothes only), for the key judge's outfit question."""
    from tts_reader import video
    d = cx.dress(who, t) if hasattr(cx, "dress") else ""
    return f"wearing {video.worn_only(d)}" if d and video.worn_only(d) else ""


def clothes_of(look: str) -> str:
    """The clothing words of a look: what follows "wearing", posture and carried things left out."""
    m = re.search(r"\bwearing\b\s+(.*)", no_pose(uncarried(look)))
    return m.group(1).strip().rstrip(".") if m else ""


def garments(clothes: str) -> str:
    """The first two garments of a clothes description, whole ("a tweed jacket, white shirt with tie")."""
    parts = [x.strip() for x in re.split(r",|;|\band\b", clothes) if x.strip()]
    return ", ".join(parts[:2])


def pose_cat(text) -> str:
    """'sit' / 'stand' / 'lie' / 'kneel' / 'crouch' for a pose word or phrase, '' when it names none."""
    return next((c for c, rx in POSE_CAT if rx.search(str(text or ""))), "")


def _pose_hold(cx, s: dict) -> str:
    """' The man in a ragged brown tunic stays sitting on the ground throughout.' for each person whose held pose is not
    standing and does not change in this shot (the video model's words: a plain description, never a name)."""
    out = []
    for w, c in (s.get("pose_held") or {}).items():
        if c == "stand" or (s.get("pose_after") or {}).get(w, c) != c:
            continue
        lk = cx.look(w, 0) or ""
        first = re.sub(r"\s+of about \d+.*$", "", re.split(r",| with | wearing ", lk.strip(), maxsplit=1)[0]).strip().rstrip(".")
        if first:
            out.append(f" {first[0].upper() + first[1:]} stays {POSE_WORDS[c]} throughout.")
    return "".join(out)


def carry_poses(es: list, out: dict) -> None:
    """Each shot's "held" pose per person: the pose the shot before ended with (a person's first shot: the director's
    opening pose); it changes only where the director has a shot's own action change it (pose != pose_end). A
    director's opening pose that contradicts the carried one (up / down / up) is overruled."""
    cur = {}
    for e in es:
        for s in out.get(e["id"]) or []:
            if s.get("kind") == "talk":
                continue
            p0 = {w: pose_cat(v) for w, v in (s.get("pose") or {}).items()}
            p1 = {w: pose_cat(v) for w, v in (s.get("pose_end") or {}).items()}
            held = {}
            for w in s.get("who") or []:
                c0 = cur.get(w) or p0.get(w) or ""
                if not c0:
                    continue
                held[w] = c0
                cur[w] = p1[w] if (p0.get(w) and p1.get(w) and p1[w] != p0[w]) else c0
            s["pose_held"] = held
            s["pose_after"] = {w: cur[w] for w in held}


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


def figurative(text: str) -> str:
    """A prompt with its figures of speech replaced by what is literally seen (FIGURATIVE)."""
    for k_, v_ in FIGURATIVE.items():
        text = re.sub(k_, v_, text, flags=re.I)
    return text


def timed(tl: list, t0: float, sec: float) -> list:
    """[(seconds from t0, sentence)] for the narration starting in [t0 - 0.5, t0 + sec - 0.5): a timeline entry is a
    paragraph, so each sentence's start is estimated from its word position (the narration's pace is steady)."""
    out = []
    for i, x in enumerate(tl):
        t, nxt = float(x["t"]), float(tl[i + 1]["t"]) if i + 1 < len(tl) else float(x["t"]) + 0.4 * len(x["text"].split())
        sents = [z for z in re.split(r"(?<=[.!?;])\s+", x["text"].strip()) if z]
        n = sum(len(z.split()) for z in sents) or 1
        done = 0
        for z in sents:
            ts = t + (nxt - t) * done / n
            done += len(z.split())
            if t0 - 0.5 <= ts < t0 + sec - 0.5:
                out.append((round(max(0.0, ts - t0), 1), z))
    return out


def _passing(a: dict, tim: list, room: str) -> list:
    """Faults for a passing effect in a shot before its sentence, or with no sentence in this stretch (PASSING)."""
    out, start = [], 0.0
    for i, s in enumerate(a.get("shots") if isinstance(a.get("shots"), list) else [], 1):
        end = start + float(s.get("dur") or 0)
        seen = f"{s.get('picture', '')} {s.get('action', '')}"
        for g, rx in PASSING.items():
            if not rx.search(seen) or rx.search(room or ""):
                continue
            said = [t for t, z in tim if rx.search(z) or (g == "fire" and re.search(r"\btongues?\s+(as\s+)?of\b", z, re.I))]
            if not said:
                out.append(f"shot {i} shows {g}, but the narration of this stretch never has it (leave it out)")
            elif min(said) > end + 0.5:
                out.append(f"shot {i} ({start:.1f}-{end:.1f} s) shows {g} before the narration has it at {min(said):.1f} s")
        start = end
    return out


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
        if s.get("who") and not all(pose_cat((s.get("pose") or {}).get(w)) for w in s["who"]):
            out.append(f"shot {i}: give \"pose\" (and \"pose_end\") for each person: standing, sitting, lying, "
                       f"kneeling or crouching")
        if TONGUE.search(f"{s.get('picture', '')} {act}"):
            out.append(f"shot {i} says 'tongue': draw what is seen (small flames above heads, people talking), never a tongue")
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


def _converse(url: str, model: str, messages: list, max_tokens: int = 5000, effort: str = "") -> tuple:
    """One turn of a conversation: (reply, prompt tokens the server had cached). `effort` ("low", "medium", ...) lets
    the model think first; the thinking stays out of the reply (and so out of the conversation)."""
    from urllib.request import Request, urlopen
    body = {"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens}
    if effort:
        body["reasoning_effort"] = effort
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=1800) as resp:
        r = json.loads(resp.read())
    return (re.sub(r"(?s)^.*</think>", "", r["choices"][0]["message"]["content"] or ""),
            ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0))


SIDE = {"left": 0, "centre": 1, "center": 1, "middle": 1, "right": 2}
POSE_VERB = {"stand": "stands", "sit": "sits", "lie": "lies", "kneel": "kneels", "crouch": "crouches"}


def compose(s: dict, cast: list, crs: list) -> dict:
    """A shot's "who", "pose" and "picture" from the director's "people" (name, side, pose, at, facing, holds) and
    "setting" (10-27 user: "the goal is to prevent the errors more than detect and fix them": who stands where and in
    what posture was written twice - the who list's order and the picture's free words - and they disagreed: Helen
    drawn at the desk opened Holmes's drawer, Holmes "stood" while leaning back in his chair). Now it is said once."""
    ppl = [x for x in s.get("people") or [] if isinstance(x, dict)]
    if not ppl and not s.get("setting"):
        return s                        # an old-style answer: as it is
    rows = []
    for x in ppl:
        w = MF._cast_name(x.get("name"), cast, crs)
        if w and w not in [r[0] for r in rows]:
            rows.append((w, SIDE.get(str(x.get("side") or "").lower(), 1), pose_cat(x.get("pose")) or "stand", x))
    rows.sort(key=lambda r: r[1])       # left to right, stable
    many = len(rows) > 1
    parts = []
    for w, sd_, pz, x in rows:
        where = str(x.get("at") or "").strip().rstrip(".")
        side = f" on the {['left', 'centre', 'right'][sd_]}" if many else ""
        face = str(x.get("facing") or "").strip().rstrip(".")
        hold = str(x.get("holds") or "").strip().rstrip(".")
        parts.append(f"{w} {POSE_VERB[pz]}{' ' + where if where else ''}{side}"
                     f"{', facing ' + re.sub(r'^(facing|toward|towards)\s+', 'toward ', face) if face else ''}"
                     f"{', holding ' + re.sub(r'^holding\s+', '', hold) if hold and hold.lower() not in ('nothing', 'none') else ''}")
    setting = str(s.get("setting") or "").strip().rstrip(".")
    s = dict(s, who=[r[0] for r in rows], pose={r[0]: POSE_WORDS[r[2]].split()[0] for r in rows},
             picture=("; ".join(parts) + (". " if parts and setting else "") + setting).strip() + ".")
    return s


def picture_order(who: list, picture: str) -> list:
    """`who` in the left-to-right order the picture puts them (10-27: "Holmes stands at the desk on the right ... Helen
    on the left" with who [Holmes, Watson, Helen]: the looks are drawn in boxes by who's order, so Helen stood at the
    desk and opened the drawer Holmes's action named). Unchanged unless the picture places at least two of them."""
    pos = {}
    for w in who:
        names = [w] + [x for x in w.split() if len(x) > 2 and x[0].isupper() and x.lower() not in ("the", "sir", "dr.")]
        for clause in re.split(r"[;,]|\bwhile\b|\band\b(?=\s+[A-Z])", picture or ""):
            if any(re.search(rf"\b{re.escape(n)}\b", clause) for n in names):
                m = re.search(r"\b(?:on|at|to) the (left|right|centre|center|middle)\b|\bin the (centre|center|middle)\b",
                              clause, re.I)
                if m:
                    pos[w] = SIDE[(m.group(1) or m.group(2)).lower()]
                    break
    if len(pos) < 2:
        return list(who)
    return sorted(who, key=lambda w: (pos.get(w, 1), who.index(w)))


def _repeats(a: dict, before: list, scene: list) -> list:
    """Shots that bring back a picture of an EARLIER scene, word for word or nearly (10-27: Watson's preamble looped
    desk -> bed -> desk -> bed, copying its own earlier answers). Pictures of the scene going on now may repeat: a
    steady listener, the master shot cut back to between close-ups (10-27 user: someone listening for 30 s stays as
    they are). `before`: the pictures of the scenes before this one; `scene`: this scene's so far."""
    import difflib
    out, scene = [], [e for e in scene if e]
    for i, s in enumerate(a.get("shots") or [], 1):
        pic = str(s.get("picture") or "")
        if s.get("kind") == "talk" or not pic:
            continue
        sim = lambda e: difflib.SequenceMatcher(None, pic.lower(), e.lower()).ratio() > 0.9
        if any(sim(e) for e in before if e) and not any(sim(e) for e in scene):
            out.append(f"shot {i} brings back a picture of an earlier scene: show what THIS passage tells")
        scene.append(pic)
    return out


def sentences(tl: list) -> list:
    """Every sentence of the chapter: (start, end, speaker, kind, words), each sentence's times estimated from its word
    position in its timeline entry (an entry is a paragraph or a spoken line; the pace is steady)."""
    out = []
    for i, x in enumerate(tl):
        t, nxt = float(x["t"]), float(tl[i + 1]["t"]) if i + 1 < len(tl) else float(x["t"]) + 0.4 * len(x["text"].split())
        sents = [z for z in re.split(r"(?<=[.!?;])\s+", x["text"].strip()) if z]
        n = sum(len(z.split()) for z in sents) or 1
        done = 0
        for z in sents:
            ts = t + (nxt - t) * done / n
            done += len(z.split())
            out.append((ts, t + (nxt - t) * done / n, x.get("speaker") or "narrator", x.get("kind") or "narration", z))
    return out


def passages(tl: list, t0: float, t1: float, mx: float = PASSAGE_MAX, forced: tuple = ()) -> list:
    """[(start, end)] covering [t0, t1): cut at timeline entries (a paragraph or a spoken line) once a passage has
    reached 60% of `mx`, and inside an entry at a sentence when one entry alone runs past `mx`; always cut at the
    `forced` times (the scenes' starts: no passage spans two scenes)."""
    forced = sorted(f for f in forced if t0 + 0.5 < f < t1 - 0.5)
    if forced:
        out, a = [], t0
        for f in forced + [t1]:
            out += passages(tl, a, f, mx)
            a = f
        return out
    cuts = sorted({float(x["t"]) for x in tl if t0 < float(x["t"]) < t1})
    marks = sorted(set(cuts) | {ts for ts, _, _, _, _ in sentences(tl) if t0 < ts < t1})
    out, a = [], t0
    while t1 - a > 0.05:
        if t1 - a <= mx:
            out.append((a, t1))
            break
        ok = [c for c in cuts if a + 0.6 * mx <= c <= a + mx] or [c for c in marks if a + 0.6 * mx <= c <= a + mx] or \
             [c for c in marks if a + 2.0 <= c <= a + mx]
        b = max(ok) if ok else a + mx
        out.append((a, b))
        a = b
    return out


def load_timeline(book: Path, stem: str, chars: list) -> list:
    """The chapter's timeline with every speaker as their CAST name ("Stoner" -> "Helen Stoner", "Watson" ->
    "Dr. Watson"; 10-27: Helen's talk prompts said "Stoner speaks ... Helen Stoner listens with a closed mouth" and
    the model moved Holmes's lips). Read through here, nothing after it sees a second name for anyone."""
    tl = json.loads((book / "work" / f"{stem}.timeline.json").read_text())
    for x in tl:
        if x.get("speaker") and x["speaker"] != "narrator":
            x["speaker"] = _who_full(x["speaker"], chars) or x["speaker"]
    return tl


def _who_full(spk: str, names: list) -> str:
    """A timeline speaker ("Holmes", "Stoner") as a cast name ("Sherlock Holmes", "Helen Stoner")."""
    if not spk or spk == "narrator":
        return ""
    return next((w for w in names if w == spk), None) or next(
        (w for w in names if re.search(rf"\b{re.escape(spk)}\b", w, re.I)), spk)


def _shot_faults(a: dict, cast: list, sec: float, lines: list) -> list:
    """The director's passage answer against the rules: starts in order, lengths, talk shots on their own speaker's
    line, and the one-shot checks (_faults) on the rest."""
    shots = a.get("shots") if isinstance(a.get("shots"), list) else []
    if not shots:
        return ["no shots"]
    out = []
    try:
        st = [float(s.get("start")) for s in shots]
    except (TypeError, ValueError):
        return ["every shot needs a number \"start\""]
    if abs(st[0]) > 0.3 or any(b <= a_ for a_, b in zip(st, st[1:])) or st[-1] >= sec:
        out.append(f"the starts must begin at 0.0 and rise, each before {sec:.1f}")
    ends = st[1:] + [sec]
    rest = []
    for i, (s, a_, b) in enumerate(zip(shots, st, ends), 1):
        d = b - a_
        if s.get("kind") == "talk":
            who = MF._cast_name(s.get("speaker"), cast, [])
            mine = [(x0, x1) for x0, x1, w in lines if w == who]
            if not who:
                out.append(f"shot {i} is a talk shot without a \"speaker\" from {', '.join(cast)}")
            elif not any(min(b, x1) - max(a_, x0) >= 0.6 * min(d, x1 - x0) for x0, x1 in mine):
                out.append(f"shot {i} is a talk shot of {who} at {a_:.1f}-{b:.1f} s, but {who} is not speaking then")
            elif d > 8.5:
                out.append(f"shot {i} is a talk shot of {d:.1f} s: at most 8 (cut to the listener and back)")
        else:
            if d > SHOT_MAX + 0.6 or (d < SHOT_MIN - 0.6 and b < sec):
                out.append(f"shot {i} lasts {d:.1f} s: other shots are {SHOT_MIN:.0f} to {SHOT_MAX:.0f} s")
            rest.append(dict(s, dur=d))
    if rest:
        out += [f for f in _faults({"shots": rest}, cast, sum(r["dur"] for r in rest)) if not f.startswith("the lengths")
                and not (f.endswith(f"who must be names from {', '.join(cast)}") and
                         not rest[int(f.split()[1].rstrip(':')) - 1].get("who"))]  # a thing alone: nobody in it
        for i, s in enumerate(shots, 1):
            if s.get("kind") != "talk" and WRITTEN.search(str(s.get("picture", ""))):
                out.append(f"shot {i}'s picture has writing ({WRITTEN.search(s['picture']).group(0)}): the video model "
                           "can't write; show the people or a place instead")
            if s.get("kind") != "talk" and CAMERA.search(str(s.get("action", ""))):
                out.append(f"shot {i}'s action moves the camera ('{CAMERA.search(s['action']).group(0)}'): only the "
                           "people and things move")
            have = {pose_cat(v) for v in (s.get("pose") or {}).values()} - {""}
            said = {c for c, rx in POSE_CAT if rx.search(str(s.get("picture", "")))} & {"sit", "stand", "lie"}
            if s.get("kind") != "talk" and have and said - have:   # 10-27: "Holmes stands by the desk", pose sitting
                out.append(f"shot {i}'s picture has someone {POSE_WORDS[sorted(said - have)[0]].split()[0]} but its "
                           f"\"pose\" has nobody so: the picture shows each one in their pose")
    return out


def direct(book: Path, stem: str, es: list, log=print, sheets: dict | None = None, keep_llm: bool = False) -> tuple:
    """The director makes the film's shots itself, a PASSAGE at a time (10-27 user: "the whole point of the director
    is to break up the story into individual shots"): narration and spoken lines together, every sentence timed with
    who says it, the story before and after, where the people are, how things stand (carried passage to passage) and
    the last shot before. Its shots become the moments: -> (moments, {mid: [shot]}, {mid: sheet}). The planner's
    moments give the spoken lines' voice audio and where each place's sheet is; passages are cached by their question
    in shots/passages/."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    _, cx = _ctx(book, stem)
    sd = shot_dir(book, stem)
    (sd / "passages").mkdir(parents=True, exist_ok=True)
    sheets = sheets or {}
    span = _spans(es)
    t_a, t_b = float(es[0]["t"]), float(es[-1]["t"]) + span[es[-1]["id"]]
    plan = json.loads((MF.paths(book, stem)[0] / "plan.json").read_text())
    chars = sorted((plan.get("look") or {}).get("characters") or {}) or sorted({w for e in es for w in _cast(book, stem, e)[0]})
    tl = load_timeline(book, stem, chars)
    sents = sentences(tl)
    nar = BC.narrator_character(book)
    nar = _who_full(nar, chars) if nar else ""
    talks = [e for e in es if e["route"] == "talk" and e.get("wav")]
    # each place's sheet (room, fixtures, crowd): from a planner moment there that isn't shown "elsewhere"
    place_sheet = {}
    for e in es:
        sh = sheets.get(e["id"]) or {}
        if e.get("where") and not sh.get("elsewhere") and not NO_PLACE.search(e["where"]):
            place_sheet.setdefault(e["where"], sh)
    real = list(dict.fromkeys(e["where"] for e in es if e.get("where") and not NO_PLACE.search(e["where"])))
    lp = book / "work" / "video" / "look.json"     # no sheets yet (they come after): the book's own place looks
    for w, d in ((json.loads(lp.read_text()).get("places") or {}) if lp.exists() else {}).items():
        for w2 in [w] + [x for x in real if re.sub(r"\s*\((inside|outside)\)$", "", x) == w]:
            if isinstance(d, str) and d:
                place_sheet.setdefault(w2, {"room": d})
    # every spoken line's span, by speaker (the voices of the others are silenced in a talk shot's audio)
    spans = {}
    for x, y in zip(tl, tl[1:] + [{"t": float(tl[-1]["t"]) + 0.4 * len(tl[-1]["text"].split())}]):
        if x.get("kind") == "quote" and x.get("speaker") not in (None, "", "narrator"):
            spans.setdefault(_who_full(x["speaker"], chars), []).append((float(x["t"]), float(y["t"]), x))
    voices = book / "work" / f"{stem}.mp3"
    (sd / "audio").mkdir(parents=True, exist_ok=True)

    def look_of(w, dress):              # "the man in the tweed shooting jacket" for a name in an action (10-27: "the
        lk = MF._appearance(cx.look(w, t_a) or "")          # man of about 35" / "of about 40" told two men apart badly)
        noun = (re.search(r"\b(man|woman|boy|girl|creature|dragon|horse|dog)\b", lk, re.I) or [None, "figure"])[1]
        cl = garments(uncarried(dress.get(w) or clothes_of(lk))).split(",")[0].strip()
        cl = re.sub(r"^(a|an|the)\s+", "", cl, flags=re.I)
        return f"the {noun.lower()} in the {cl[0].lower() + cl[1:]}" if cl else f"the {noun.lower()}"

    def unname(txt, cast, dress):       # the video model reads looks, not names: swap them in
        for w in sorted(cast, key=len, reverse=True):
            parts = [w] + [x for x in w.split() if len(x) > 2 and x[0].isupper() and x.lower() not in ("the", "sir", "dr.")]
            txt = re.sub(r"\b(" + "|".join(map(re.escape, parts)) + r")\b", look_of(w, dress), txt)
        return txt

    def talk_moment(mid, who, t, d):    # a talk moment for the line `who` speaks at t (its own cut of the voices)
        mine = [(a_, b_, x) for a_, b_, x in spans.get(who, []) if b_ > t and a_ < t + d]
        src = next((e for e in talks if e.get("speaker") == who), None) or (talks[0] if talks else {})
        wav = sd / "audio" / f"{mid}_speech.wav"
        keep = "+".join(f"between(t,{max(0.0, a_ - t):.3f},{b_ - t:.3f})" for a_, b_, _ in mine) or "0"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{d:.3f}", "-i", str(voices),
                            "-ac", "1", "-ar", "16000", "-af", f"volume=0:enable='not({keep})'", str(wav)], check=True)
        x = mine[0][2] if mine else {}
        return dict(src, id=mid, kind="talk", route="talk", t=t, dur=d, speaker=who, line=x.get("text", ""),
                    emotion=x.get("emotion") or src.get("emotion"), creature=bool(src.get("creature")) and
                    src.get("speaker") == who, speech_s=d, frames=_frames(d), chunks=[[0, _frames(d)]], wav=str(wav))

    def at(t):                          # the planner moment at time t
        return next((e for e in reversed(es) if float(e["t"]) <= t + 1e-6), es[0])
    # the chapter's opening place, unless the planner opened on no place (a title card): then the director picks
    home = "" if es[0].get("where") and NO_PLACE.search(es[0]["where"]) else (real[0] if real else "")
    where_of = {}                       # where each one was last shown: a talk shot is there
    state, prev_shot, people_here = "(the start of the chapter)" if t_a < 1 else "", None, []
    new_es, by, new_sheets = [], {}, {}
    system, talk_ = SHOT_RULES.format(maxp=MAX_PEOPLE, closemax=CLOSE_MAX), []
    pics_before, pics_scene = [], []
    sides, pair = {}, None
    animate_up = False
    known = list(dict.fromkeys(real + [w for w in ((json.loads(lp.read_text()).get("places") or {}) if lp.exists() else {})]))
    scenes = breakdown(book, stem, tl, chars, known, nar, log)
    animate_up = subprocess.run(["systemctl", "--user", "is-active", "-q", animate.STRATA_UNIT]).returncode == 0
    for sc_ in scenes:                  # the breakdown's places have their sheets (the book's looks) where known
        for w in [sc_["place"]] + [c["place"] for c in sc_["cutaways"]]:
            base = re.sub(r"\s*\((inside|outside)\)$", "", w)
            if w not in place_sheet and base in place_sheet:
                place_sheet[w] = place_sheet[base]
    cur = None
    for pa, pb in passages(tl, t_a, t_b, forced=tuple(x["start"] for x in scenes)):
        sec = pb - pa
        si = max([i for i, x in enumerate(scenes) if x["start"] <= pa + 0.5] or [0])
        sc_ = scenes[si] if scenes else {"place": home, "who": [], "what": "", "cutaways": [], "opens": "",
                                         "start": t_a, "end": t_b}
        if si != cur:                   # a new scene: its place; the conversation starts again (no old shots to copy)
            cur, home = si, sc_["place"] or home
            sides, pair, people_here, where_of, talk_ = {}, None, [], {}, []
            pics_before, pics_scene = pics_before + pics_scene, []
            if sc_.get("opens"):
                state = (f"{state} Now, as this scene opens: {sc_['opens']}" if state and not state.startswith("(")
                         else sc_["opens"])[-700:]
        nxt = scenes[si + 1] if si + 1 < len(scenes) else None
        scene_txt = (f"THE SCENE: scene {si + 1} of the chapter ({sc_['start']:.0f}-{sc_['end']:.0f} s of it), at "
                     f"{home}, with {', '.join(sc_['who']) or 'nobody'}. The whole scene, for context (show only what "
                     f"THIS passage tells, never what comes later in it): {sc_.get('what', '')}"
                     + "{CUTS}"
                     + (f"\nThe scene after it (not this one): at {nxt['place']}: {nxt.get('what', '')}" if nxt else ""))
        lines_ = [x for x in sents if x[1] > pa + 0.05 and x[0] < pb - 0.05]
        quotes = [(max(0.0, x0 - pa), min(sec, x1 - pa), _who_full(spk, chars)) for x0, x1, spk, k, _ in lines_
                  if k == "quote" and spk != "narrator"]
        txt = []
        for x0, x1, spk, k, z in lines_:
            who = _who_full(spk, chars) or ("narrator" + (f" ({nar}, telling it as 'I')" if nar else ""))
            cont = " (continuing)" if x0 < pa - 0.3 else ""
            if k == "quote" and spk != "narrator":
                txt.append(f"[{max(0.0, x0 - pa):.1f}-{min(sec, x1 - pa):.1f}] {who} SAYS ALOUD{cont}: {z}")
            else:
                txt.append(f"[{max(0.0, x0 - pa):.1f}] {who}{cont}: {z}")
        before = " ".join(z for x0, x1, spk, k, z in sents if x1 <= pa + 0.05)[-500:] or "(the chapter's start)"
        after = " ".join(z for x0, x1, spk, k, z in sents if x0 >= pb - 0.05)[:250] or "(the chapter's end)"
        ms = [e for e in es if float(e["t"]) < pb - 0.05 and float(e["t"]) + span[e["id"]] > pa + 0.05]
        wheres = list(dict.fromkeys(e.get("where") for e in ms if e.get("where") and not NO_PLACE.search(e["where"])))
        # a cutaway only while its teller is telling it, in this passage (prevention, not a check: no other passage is
        # offered the place at all)
        tellers = {w for _, _, w in quotes}
        open_ = [c for c in sc_["cutaways"] if c["from"] - 2 < pb and c["to"] + 2 > pa and
                 (not c["teller"] or c["teller"] in tellers or (c["teller"] == nar and len(quotes) < len(lines_)))]
        others = list(dict.fromkeys(c["place"] for c in open_ if c["place"] != home)) if scenes else \
            [w for w in wheres or real if w != home]
        scene_txt = scene_txt.replace("{CUTS}", (" In this passage it may cut to: " + "; ".join(
            f"{c['place']} (as {c['teller'] or 'the narration'} tells of it)" for c in open_) + ".") if open_ else
            " In this passage it stays there.")
        cast = list(dict.fromkeys(people_here + sc_["who"] + [w for e in ms for w in _cast(book, stem, e)[0]] +
                                  [w for _, _, w in quotes if w] +
                                  [w for e in ms for w in (sheets.get(e["id"]) or {}).get("present", [])]))
        cast = [w for w in cast if w in chars] or cast
        crs = [w for e in ms for w in _cast(book, stem, e)[1]]
        mid_t = round(pa * 10)
        dress = MF._dress(book, stem, f"action-{mid_t:06d}", cast)
        looks = {w: MF._brief(MF._appearance(cx.look(w, pa) or "")) for w in cast}
        people = "; ".join(f"{w} ({looks.get(w, '')}{'; wears ' + dress[w] if dress.get(w) else ''})" for w in cast)
        hs = place_sheet.get(home) or {}
        narr = " ".join(z for *_, k_, z in lines_ if k_ != "quote")
        extra = "".join(r for rx, r in RULE_EXTRA[:3] if rx.search(" ".join(z for *_, z in lines_))) + \
            "".join(r for rx, r in RULE_EXTRA[3:] if rx.search(narr))     # 10-27: Helen's "made" was no forge
        crowd = " / ".join(dict.fromkeys(c for c in [(place_sheet.get(w) or {}).get("crowd") for w in [home] + others] if c))
        if crowd:
            extra += RULE_CROWD.format(crowd=crowd)
        told = RULE_TOLD.format(others="; ".join(f"{w} (its fixed things: {(place_sheet.get(w) or {}).get('room') or 'as the narration says'})"
                                                  for w in others)) if others else ""
        pv = ("(none: the chapter starts here)" if prev_shot is None else
              f"at {prev_shot.get('place')}: {prev_shot.get('picture') or ', '.join(prev_shot.get('who') or [])}"
              + (f" Postures at its end: {json.dumps(prev_shot.get('pose_end'))}" if prev_shot.get("pose_end") else ""))
        q0 = PASSAGE_ASK.format(scene=scene_txt, before=before, sec=sec, text="\n".join(txt) or "(silence)", after=after,
                               narrator=(f"The narrator is {nar}, telling the story as 'I'. When the narration is {nar} "
                                         f"musing, explaining or judging rather than telling what happens, show {nar} "
                                         f"where the story has them NOW, doing what they are doing there (listening, "
                                         f"watching, getting dressed); only before the story itself begins, {nar} "
                                         f"writing at a desk among old notes. Never an empty background.\n"
                                         if nar else ""),     # 10-27: a mid-story musing cut back to the desk
                               home=home or "(as the narration says)", room=hs.get("room") or "(as the narration says)",
                               told=told, people=people or "nobody", now=state or hs.get("now") or "(nothing noted)",
                               prev=pv, nmin=max(1, math.ceil(sec / SHOT_MAX)), nmax=max(1, math.floor(sec / SHOT_MIN)),
                               extra=extra,
                               places="; ".join([home] + others) if home else "; ".join(others) or "(as the narration says)",
                               names=", ".join(cast) or "(nobody: the place alone)", fix="{fix}")
        cp = sd / "passages" / f"p{mid_t:06d}.json"
        k = MF._h(system + q0 + DIRECT_EFFORT)
        got = json.loads(cp.read_text()) if cp.exists() else {}
        if got.get("k") == k:
            a = got["a"]
            hit = True
        else:
            hit = False
            if not animate_up:
                animate.igpu("strata", log)
                animate_up = True
            a, faults = {}, []
            if sum(len(m_["content"]) for m_ in talk_) > HISTORY_CHARS:
                talk_[:] = talk_[-4:]   # too long (decode slows with context): go on from the last two passages
            turn = [{"role": "user", "content": q0.replace("{fix}", "")}]
            for _ in range(3):
                reply, cached = _converse(BC.LLM_URL, BC.LLM_MODEL, [{"role": "system", "content": system}] + talk_ +
                                          turn, max_tokens=5000 + 12000 * bool(DIRECT_EFFORT), effort=DIRECT_EFFORT)
                log(f"  passage {pa:.1f} s: {cached} of the prompt's tokens cached")
                turn += [{"role": "assistant", "content": reply or ""}]
                m = re.search(r"\{.*\}", reply or "", re.S)
                try:
                    a = json.loads(m.group(0)) if m else {}
                except ValueError:
                    a = {}
                a["shots"] = [compose(s, cast, crs) if isinstance(s, dict) and s.get("kind") != "talk" else s
                              for s in a.get("shots") or []] if isinstance(a.get("shots"), list) else []
                for s in a.get("shots") or []:
                    s["who"] = list(dict.fromkeys(w for w in (MF._cast_name(x, cast, crs) for x in (s.get("who") or []))
                                                  if w))
                    if s.get("action"):
                        s["action"] = unname(str(s["action"]), cast, dress)

                faults = _shot_faults(a, cast, sec, quotes) + _repeats(a, pics_before, pics_scene) + [
                    f"shot {i} is at {x.get('place')}: this scene is at {home}" + (f" and may cut to {', '.join(others)}"
                                                                                   if others else "")
                    for i, x in enumerate(a.get("shots") or [], 1) if x.get("kind") != "talk" and scenes and
                    not any(str(x.get("place") or "").strip().lower() in (w.lower(), re.sub(r"\s*\((inside|outside)\)$",
                            "", w).lower()) for w in [home] + others)]
                rest = [s for s in a.get("shots") or [] if s.get("kind") != "talk"]
                if rest and not faults:
                    said = " ".join(z for *_, z in lines_)
                    tim = [(max(0.0, x0 - pa), z) for x0, _, _, _, z in lines_]
                    st_ = [float(s.get("start") or 0) for s in a["shots"]]
                    durd = [dict(s, dur=b - x) for s, x, b in zip(a["shots"], st_, st_[1:] + [sec]) if s.get("kind") != "talk"]
                    faults = _worn({"shots": durd}, " ".join(dress.values())) + _making({"shots": durd}, narr) + \
                        _passing({"shots": durd}, tim, hs.get("room") or "")
                if not faults:
                    break
                turn.append({"role": "user", "content": "Your last answer broke the rules: " + "; ".join(faults) +
                             ". Write it again, the whole passage."})
            if faults:
                log(f"!!! passage {pa:.1f}-{pb:.1f} s: the director's shots still break the rules: {faults}")
            if a.get("shots"):
                cp.write_text(json.dumps({"k": k if not faults else "", "a": a, "reply": reply}, indent=1))
        if a.get("shots"):               # the conversation goes on with this passage and its answer (the bad tries dropped)
            talk_ += [{"role": "user", "content": q0.replace("{fix}", "")},
                      {"role": "assistant", "content": (got.get("reply") if hit else reply) or json.dumps(a)}]
        shots = a.get("shots") or []
        if not shots:                   # nothing usable: one quiet shot per 5 s of the people here
            log(f"!!! passage {pa:.1f}-{pb:.1f} s: no usable shots: quiet shots of the people here")
            n = max(1, math.ceil(sec / SHOT_MAX))
            shots = [{"start": i * sec / n, "kind": "quiet", "place": home, "who": cast[:2], "framing": "medium",
                      "picture": f"{' and '.join(cast[:2]) or 'The place'} at {home}.", "action": "Small movements."}
                     for i in range(n)]
        st = [float(s.get("start") or 0) for s in shots]
        st[0] = 0.0
        for i in range(1, len(st)):     # starts in order, and at least 1 s apart
            st[i] = min(max(st[i], st[i - 1] + 1.0), sec - 0.5)
        keep_ = [i for i in range(len(st)) if i == 0 or (st[i + 1] if i + 1 < len(st) else sec) - st[i] >= SHOT_SHORT]
        shots, st = [shots[i] for i in keep_], [st[i] for i in keep_]     # a too-short shot: the one before runs on
        cuts_ = []                      # a non-talk shot of someone speaking their line becomes their talk shot
        for s, x, b in zip(shots, st, st[1:] + [sec]):
            if s.get("kind") != "talk" and SPEAKS.search(str(s.get("action", ""))):
                hit = next((w for w in s.get("who") or [] for a_, b_, _ in spans.get(w, [])
                            if min(pa + b, b_) - max(pa + x, a_) >= 0.6 * (b - x)), None)
                if hit:
                    others_ = [w for w in s.get("who") or [] if w != hit]
                    s = {"kind": "talk", "speaker": hit, "listener": others_[0] if others_ else "", "framing": "two"}
            n_ = max(1, math.ceil((b - x) / 8.0)) if s.get("kind") == "talk" else 1
            for j in range(n_):         # a long talk shot: cut into pieces of at most 8 s
                cuts_.append((dict(s, framing=s.get("framing") if j == 0 else "over-the-shoulder"),
                              x + (b - x) * j / n_, x + (b - x) * (j + 1) / n_))
        for s, x, b in cuts_:
            t, d = pa + x, b - x
            if d < 0.4:
                continue
            place = next((w for w in [home] + others if w and str(s.get("place") or "").lower() in (w.lower(), "")), None) \
                or next((w for w in [home] + others if w and w.lower() in str(s.get("place") or "").lower()), home)
            if s.get("kind") == "talk":
                who = MF._cast_name(s.get("speaker"), cast, crs)
                if who and any(min(t + d, b_) - max(t, a_) > 0.3 for a_, b_, _ in spans.get(who, [])):
                    mid = f"talk-{round(t * 10):06d}"
                    e = talk_moment(mid, who, t, d)
                    tp = where_of.get(who) or home or place    # 10-27: talk shots on the title card / back in the bedroom
                    lis = MF._cast_name(s.get("listener"), cast, crs)
                    lis = lis if lis and lis != who else None
                    sh = {"id": f"{mid}_s0", "dur": d, "who": [who], "framing": "medium", "kind": "talk", "place": tp,
                          "action": "", "picture": (TALK_CREATURE if who in crs else TALK_PICTURE).format(who=who)}
                    if lis and who not in crs and lis not in crs:
                        if who not in sides and lis not in sides:
                            sides.update({who: "left", lis: "right"})
                        sides.setdefault(who, "right" if sides.get(lis) == "left" else "left")
                        sides.setdefault(lis, "right" if sides[who] == "left" else "left")
                        if sides[lis] == sides[who]:
                            sides[lis] = "right" if sides[who] == "left" else "left"
                        lr = sorted((who, lis), key=lambda w: sides[w] != "left")
                        fr = str(s.get("framing") or "").lower()
                        if pair != {who, lis} or fr.startswith("two"):
                            sh.update(who=lr, listener=lis, picture=TALK_TWO.format(left=lr[0], right=lr[1], who=who))
                        elif not fr.startswith("close"):
                            other = "right" if sides[who] == "left" else "left"
                            sh.update(who=lr, listener=lis, ots=lis, picture=TALK_OTS.format(
                                who=who, side=sides[who], other=other, listener=lis))
                        pair = {who, lis}
                    new_es.append(e)
                    by[mid] = [sh]
                    new_sheets[mid] = dict(place_sheet.get(tp) or {}, elsewhere="")
                    continue
                s = dict(s, kind="quiet", who=[w for w in [who, MF._cast_name(s.get("listener"), cast, crs)] if w] or cast[:1],
                         picture=s.get("picture") or f"{who or 'The speaker'} speaking at {home}.",
                         action=s.get("action") or "Small movements while listening.")
            mid = f"action-{round(t * 10):06d}"
            new_es.append({"id": mid, "kind": "action", "t": t, "dur": d, "route": "shot", "where": place,
                           "who": s.get("who") or []})
            sh = {k_: v for k_, v in s.items() if k_ not in ("start", "speaker", "listener")}
            sh.update(id=f"{mid}_s0", dur=d, place=place, framing=s.get("framing") if s.get("framing") in
                      ("wide", "medium", "close") else "medium",
                      kind=s.get("kind") if s.get("kind") in ("action", "quiet") else "quiet")
            by[mid] = [sh]
            new_sheets[mid] = dict(place_sheet.get(place) or sheets.get(at(t)["id"]) or {}, elsewhere="")
            prev_shot = sh
            if place and (place == home or not home):
                where_of.update({w: place for w in sh.get("who") or []})
        pics_scene += [by[m_["id"]][0].get("picture", "") for m_ in new_es if float(m_["t"]) >= pa - 1e-6
                       and by[m_["id"]][0].get("kind") != "talk"]
        people_here = list(dict.fromkeys(people_here + [w for m_ in new_es if by.get(m_["id"]) and
                                                         by[m_["id"]][0].get("place") == home
                                                         and float(m_["t"]) >= pa for w in by[m_["id"]][0]["who"]]))
        state = str(a.get("state_end") or state)[:600]
        for m_ in new_es:
            if float(m_["t"]) >= pa - 1e-6:
                s_ = by[m_["id"]][0]
                log(f"{s_['id']} ({s_['dur']:.1f} s, {s_['kind']}, {s_.get('framing')}, {', '.join(s_['who'])}, "
                    f"{s_.get('place')}): {s_.get('picture', '')} || {s_.get('action', '')}")
    if animate_up and not keep_llm:     # (the set sheets come next and ask it too: no reload)
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    for mid, sh in new_sheets.items():   # the place's "as it is now" comes from the director's own state now
        sh.pop("now", None)
    carry_poses(new_es, by)
    return new_es, by, new_sheets


PLACE_ASK = """Describe the place "{where}" as a painter would see it {side}, empty: no people and no animals. One sentence
of under 35 words: the ground, the buildings or the landscape, what stands there, the light. Early medieval Northern
{period}. Use what this passage says and things that belong in such a place. Only the place itself:
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
in the hands, never worn. Never a passing effect (flames over people, a wind, smoke, lightning): those come and go
with the narration, not the scene."""


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
            said = " ".join(z for _, z in timed(tl, float(e["t"]), span[e["id"]]))
            out[e["id"]] = {"scene": sid, "room": a.get("room", ""), "fixtures": a.get("fixtures") or [],
                            "present": a.get("present") or [], "crowd": a.get("crowd") or "",
                            "now": "; ".join(f"{t}: {v}" for t, v in cur.items() if passing_shown(f"{t} {v}", said)),
                            "elsewhere": (a.get("elsewhere") or {}).get(e["id"], "")}
    return out


SET_ASK = """You design the SET of one place in a film made from a book: what a painter needs so that every picture of it
looks the same.

The place: {place}
{period}The people of the book: {people}
What happens there (the director's shots, in order):
{shots}
The story's words there: {said}

Reply with JSON only:
{{"room": "two sentences: the FIXED things of this place and where each stands (the hearth with its chimney hood at
the back wall, the bellows on its left, ...). Never a person or creature, nothing the story's people carry, make or
bring; no weather or sky; for a place indoors, nothing seen outside it.",
 "fixtures": ["2 to 5 short names of things BUILT INTO this place a viewer must see in every wide picture of it"],
 "crowd": "the UNNAMED people always there (about twenty courtiers in dark robes along both walls), or \"\" when none"}}"""


# 10-27 user: "what's the right way to prevent [going back to a shot from a previous scene] in the first place?": a
# SCENE BREAKDOWN first, over the whole chapter (like a screenplay's scene headings), then every passage is directed
# inside its scene. The director's conversation starts again at each scene (an old scene's shots can't be copied in);
# what carries across (clothes, props, how things stand) carries in the breakdown and the passage-to-passage state.
BREAKDOWN_ASK = """You are a film's director, breaking a chapter of a book into its SCENES before you plan any shots,
like the scene headings of a screenplay. A new scene starts when the people of the story go somewhere else, or when
time jumps. Someone TELLING of another place or time (a story, a memory) stays in the scene where they tell it: list
the places the telling may cut to as "cutaways". {narrator}

The places already known (use these names when it is one of them; else name a new place plainly, with "(inside)" or
"(outside)"): {places}
The people of the book: {people}

THE CHAPTER, each line [seconds from its start] who: the words:
{text}

Reply with JSON only: {{"scenes": [{{"start": 0.0, "place": "...", "who": ["the people there, also the silent ones"],
"what": "one sentence: what happens in the scene", "cutaways": [{{"place": "...", "teller": "the one who tells of it",
"from": <the second the telling starts>, "to": <the second it ends>}}],
"opens": "how things stand as it opens, for a painter: each one's clothes, what each holds, postures, the time of
day and the light, at most 60 words"}}]}}
The first scene starts at 0.0; each later one after the one before; scenes in story order."""


def breakdown(book: Path, stem: str, tl: list, chars: list, places: list, nar: str, log=print) -> list:
    """The chapter's scenes [{"start", "end", "place", "who", "what", "cutaways", "opens"}], by the director over the
    whole chapter at once (cached in shots/scenes/breakdown.json by its question)."""
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    end = float(tl[-1]["t"]) + 0.4 * len(tl[-1]["text"].split())
    text = "\n".join(f"[{float(x['t']):.0f}] {_who_full(x.get('speaker') or 'narrator', chars) or 'narrator'}: "
                     f"{x['text']}" for x in tl)
    q = BREAKDOWN_ASK.format(narrator=(f"The narrator, {nar}, tells the story as 'I'; a stretch where {nar} looks "
                                       f"back before the story begins is a scene of its own." if nar else ""),
                             places="; ".join(places) or "(none yet)", people=", ".join(chars), text=text)
    cp = shot_dir(book, stem) / "scenes" / "breakdown.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    k = MF._h(q + BREAKDOWN_EFFORT)
    got = json.loads(cp.read_text()) if cp.exists() else {}
    if got.get("k") == k:
        sc = got["scenes"]
    else:
        animate.igpu("strata", log)
        sc, turn = [], [{"role": "user", "content": q}]
        for _ in range(3):
            reply, _c = _converse(BC.LLM_URL, BC.LLM_MODEL, turn, max_tokens=8000 + 16000 * bool(BREAKDOWN_EFFORT),
                                  effort=BREAKDOWN_EFFORT)
            m = re.search(r"\{.*\}", reply or "", re.S)
            try:
                sc = (json.loads(m.group(0)) if m else {}).get("scenes") or []
            except ValueError:
                sc = []
            faults = _breakdown_faults(sc, end)
            if not faults:
                break
            turn += [{"role": "assistant", "content": reply or ""},
                     {"role": "user", "content": "Your scene list broke the rules: " + "; ".join(faults) +
                      ". Write it again, the whole list."}]
        if faults:
            log(f"!!! the scene breakdown still breaks the rules: {faults}")
        if sc:
            cp.write_text(json.dumps({"k": k if not faults else "", "scenes": sc}, indent=1))
    sc = [x for x in sc if isinstance(x, dict) and x.get("place")]
    sc.sort(key=lambda x: float(x.get("start") or 0))
    starts = sorted(float(x["t"]) for x in tl)
    for x in sc[1:]:                    # a scene starts where a paragraph or a spoken line starts (10-27: a scene cut
        x["start"] = min(starts, key=lambda t: abs(t - float(x.get("start") or 0)))  # at 130.0 split Watson's line)
    sc = [x for i, x in enumerate(sc) if i == 0 or x["start"] > sc[i - 1]["start"]]
    if sc:
        sc[0]["start"] = 0.0
    for a, b in zip(sc, sc[1:] + [{"start": end}]):
        a["start"], a["end"] = float(a.get("start") or 0), float(b["start"])
        a["who"] = [w for w in (MF._cast_name(x, chars, []) for x in a.get("who") or []) if w]
        a["cutaways"] = [c for c in a.get("cutaways") or [] if isinstance(c, dict) and c.get("place")
                         and c["place"] != a["place"]]
        for c in a["cutaways"]:         # a telling: who tells it, and when (10-27: offered for the whole scene, the
            c["teller"] = MF._cast_name(c.get("teller"), chars, []) or ""   # director cut to Stoke Moran while Watson dressed)
            try:
                c["from"], c["to"] = float(c.get("from") or a["start"]), float(c.get("to") or a["end"])
            except (TypeError, ValueError):
                c["from"], c["to"] = a["start"], a["end"]
    for i, a in enumerate(sc, 1):
        log(f"scene {i} {a['start']:.0f}-{a['end']:.0f} s at {a['place']} ({', '.join(a['who'])}): {a.get('what', '')}"
            + (f" | cutaways {[c['place'] for c in a['cutaways']]}" if a["cutaways"] else ""))
    return sc


def _breakdown_faults(sc: list, end: float) -> list:
    if not sc:
        return ["no scenes"]
    out = []
    try:
        st = [float(x.get("start")) for x in sc]
    except (TypeError, ValueError, AttributeError):
        return ["every scene needs a number \"start\""]
    if st[0] > 1.0 or any(b <= a for a, b in zip(st, st[1:])) or st[-1] >= end:
        out.append(f"the starts must begin at 0.0 and rise, each before {end:.0f}")
    out += [f"scene {i} has no place" for i, x in enumerate(sc, 1) if not str(x.get("place") or "").strip()]
    out += [f"scene {i}'s place is no place ({x['place']}): name where the people are" for i, x in enumerate(sc, 1)
            if NO_PLACE.search(str(x.get("place") or ""))]
    return out


def place_sheets(book: Path, stem: str, es: list, by: dict, log=print) -> dict:
    """{mid: {"scene", "room", "fixtures", "crowd"}}: one set sheet per place the DIRECTOR's shots use, made after it
    (10-27 user: should the scenes be an output of the director? Yes: the planner's places and the scene step's
    who's-here / how-things-stand made the title card, the talk back in the bedroom and ch2's lasting flames; the
    director now says where every shot is and carries how things stand itself). Cached by input in shots/scenes/."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    from tts_reader import bookcaster as BC
    from tts_reader.score import _chat
    vd, _, _ = MF.paths(book, stem)
    plan = json.loads((vd / "plan.json").read_text())
    chars = sorted(((plan.get("look") or {}).get("characters") or {}))
    tl = load_timeline(book, stem, chars)
    pf = book / "period.txt"
    period = f"The time and place of the story: {pf.read_text().strip()}\n" if pf.exists() else ""
    at = {}
    for e in es:
        sh = (by.get(e["id"]) or [{}])[0]
        if sh.get("place"):
            at.setdefault(sh["place"], []).append((e, sh))
    sd = shot_dir(book, stem) / "scenes"
    sd.mkdir(parents=True, exist_ok=True)
    got_all, todo = {}, []
    for w, xs in at.items():
        shots = "\n".join(f"- {sh.get('picture') or ', '.join(sh.get('who') or [])}" for _, sh in xs
                          if sh.get("kind") != "talk")[:3000] or "(people talking)"
        said = " ".join(z for e, _ in xs for _, z in timed(tl, float(e["t"]), float(e["dur"])))[:2500]
        q = SET_ASK.format(place=w, period=period, people=", ".join(chars), shots=shots, said=said)
        sid = "set_" + re.sub(r"[^a-z0-9]+", "_", w.lower()).strip("_")
        cp, k = sd / f"{sid}.json", MF._h(q)
        got = json.loads(cp.read_text()) if cp.exists() else {}
        if got.get("k") == k:
            got_all[w] = (sid, got["sheet"])
        else:
            todo.append((w, sid, q, k, cp))
    if todo:
        animate.igpu("strata", log)

        def ask(t):
            w, sid, q, k, cp = t
            a = {}
            for _ in range(2):
                m = re.search(r"\{.*\}", _chat(BC.LLM_URL, BC.LLM_MODEL, q, max_tokens=1500) or "", re.S)
                try:
                    a = json.loads(m.group(0)) if m else {}
                except ValueError:
                    a = {}
                if a.get("room"):
                    break
            a = {"room": str(a.get("room") or "").strip(), "fixtures": [str(x) for x in (a.get("fixtures") or [])][:5],
                 "crowd": str(a.get("crowd") or "").strip()}
            if a["room"]:
                cp.write_text(json.dumps({"k": k, "sheet": a}, indent=1))
            log(f"set {sid}: {a['room']} | fixtures {a['fixtures']} | crowd {a['crowd'] or '-'}")
            return w, sid, a
        try:
            with ThreadPoolExecutor(DIRECT_WORKERS) as ex:
                for w, sid, a in ex.map(ask, todo):
                    got_all[w] = (sid, a)
        finally:
            subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    out = {}
    for w, xs in at.items():
        sid, a = got_all.get(w, ("", {}))
        for e, _ in xs:
            out[e["id"]] = {"scene": sid, "room": _unpropped(a.get("room", "")),
                            "fixtures": [f for f in a.get("fixtures") or [] if not STORY_THING.search(f)],
                            "crowd": a.get("crowd") or "", "elsewhere": ""}
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
            reply = (_chat(BC.LLM_URL, BC.LLM_MODEL, PLACE_ASK.format(where=base, side=f"from {sd_}", text=text,
                                                                        period=MF.period_line(book, "everything there")),
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
    looks = {w: no_pose(MF._appearance(d)) for w, d in ((plan.get("look") or {}).get("characters") or {}).items()}
    style_p = book / "art_style.txt"
    style = _key_style(style_p.read_text().strip()) if style_p.exists() else ""
    dress = MF._dress(book, stem, mid, s["who"])
    people = []
    for w in s["who"]:
        lk = looks.get(w, "")
        if w in dress:
            lk = re.split(r",?\s*\bwearing\b", lk)[0] + ", wearing " + dress[w]
        lk = uncarried(lk, s.get("picture", ""))
        lay = group_layout(len(s["who"]))
        side = f" {lay[s['who'].index(w)][1]}" if len(s["who"]) > 1 else ""
        legs = ", " + FOUR_LEGS if "four-legged" in lk and not MF.DOWN.search(s["picture"]) else ""
        if w == s.get("ots"):           # over the shoulder: the back of the head, out of focus
            people.append(f"{w}, seen from behind{side}: {re.split(r',', lk)[0]}")
            continue
        pz = POSE_WORDS.get((s.get("pose_held") or {}).get(w, ""), "")
        people.append(f"{MF._lora(book, w)[1] or ''} {w}{side}{', ' + pz if pz else ''}"
                      f"{', whole body visible' if s.get('framing') == 'wide' else ''}: {lk.rstrip('.')}{legs}".strip())
    place = re.sub(r"\s*\((inside|outside)\)$", "", s.get("place") or "")
    pic = figurative(s["picture"])
    for k_, v_ in GLOSS.items():
        pic = re.sub(k_, v_, pic, count=1, flags=re.I)
    room = s.get("room") or ""
    for k_, v_ in GLOSS.items():
        room = re.sub(k_, v_, room, count=1, flags=re.I)
    crowd = (f" In the background, small and still: {s['crowd'].rstrip('.')}." if s.get("crowd") else "")
    cont = crowd + (f" The room: {room.rstrip('.')}." if room else "") + \
           (f" As it is now: {s['now'].rstrip('.')}." if s.get("now") else "")
    return (f"{str(s.get('framing') or 'medium').capitalize()} shot{' in ' + place if place else ''}. {pic}. "
            f"{'. '.join(people)}.{cont}{SETTING.format(place=place) if place else ''} "
            f"{MF.period_line(book)} {style} No text."
            ).replace("..", ".").replace("  ", " ")


def _talk_prompt(R, cx, e: dict, here: str, s: dict | None = None) -> str:
    """The mannequin flow's talk words (motion_prompts): short, "lips and jaw moving clearly", no style paragraph and
    no "everyone else ... mouth closed" (10-24: R.talk_prompt's long prompt with that sentence froze the lips again in
    most of the chapter's talk clips, as it had on 10-22; a talk shot has one person, so it closed the speaker's mouth)."""
    s = s or {}
    # 10-27 user: "it's almost as if Holmes and her are switched, his lips move when she talks": the timeline's label
    # ("Stoner") was not the cast name ("Helen Stoner"), so her prompt had no look, the wrong side and "Helen Stoner
    # listens with a closed mouth" - the model moved Holmes's lips. The speaker is the shot's own name for them.
    who = MF._cast_name(e["speaker"], s.get("who") or [], []) or e["speaker"]
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
    lst = ""
    other = next((w for w in s.get("who") or [] if w != who), None)
    if other and not s.get("ots"):                # a two-shot: the other face is in the picture, never the speaker's
        lst = f" {other} listens with a closed mouth."
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
    # 10-27: "...letters, a page, a panel..." said yes to the papers, books and frames of Holmes's study (250 of 276
    # Speckled Band keys failed it): only writing laid OVER the picture
    q = {"text": "Answer yes or no: is there a caption, title, lettering or a border laid over this picture, like text "
                 "added on top of a painting (books, papers and framed pictures in the scene don't count)?"}
    hum = 0
    for i, w in enumerate(s["who"]):
        lk = cx.look(w, 0) or ""
        if w == s.get("ots"):           # the back of a head: counted, not asked for by face
            hum += 1
            continue
        first = re.sub(r"\s+of about \d+.*$", "", re.split(r",| with | wearing ", lk.strip(), maxsplit=1)[0]).strip().rstrip(".")
        if first:
            q[f"who{i}"] = f"Answer yes or no: is there {first} in this picture?"
        m_ = re.search(r"-(\d{4,})", s.get("id") or "")     # the shot's time (ids carry it in tenths of a second)
        cl = clothes_of(video_dress(cx, w, int(m_.group(1)) / 10 if m_ else 0.0) or lk)
        if first and cl and s.get("framing") != "close":    # 10-26 user: "outfit changes between clips"
            q[f"who{i}w"] = f"Answer yes or no: is {first} wearing {garments(cl)}?"   # (10-27: cut at 14 words)
        pz = (s.get("pose_held") or {}).get(w, "")
        if first and pz in ("sit", "lie", "kneel", "crouch"):
            q[f"pose{i}"] = f"Answer yes or no: is {first} {POSE_WORDS[pz]} (not standing)?"
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
                        and all(CH._yes(a.get(x)) for x in a if x.startswith(("who", "legs", "pose")))
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
            e["chunks"] = R.talk_chunks(n_all, R.audio_env(Path(e["wav"]), n_all + TALK_PIECE), TALK_PIECE,
                                        TALK_CHAIN_OVERLAP)
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
            act = figurative(s["action"]) + _pose_hold(cx, s) + STILL_CROWD * bool(s.get("crowd"))
            emb = f"x_{s['id']}_{MF._h(act + MF.FIGHT_NEG)}"
            enc_dev += [{"name": emb, "prompt": act}, {"name": emb + "__neg", "prompt": neg + MF.FIGHT_NEG}]
            for t, seed in enumerate(TAKES + ((RETAKE,) if s["id"] in (retake or ()) else ())):
                two.append({"name": s["id"] + ("" if t == 0 else f"_t{t + 1}"), "emb": emb, "prompt": act,
                            "width": W, "height": H, "fps": FPS, "seed": seed + 7 * bool(s["id"] in (redo or ())),
                            "frames": n, "line": s["id"], "neg_extra": MF.FIGHT_NEG, "route": "ti2v", "steps": STEPS,
                            "guider": GUIDER, "aguider": GUIDER, "images": img,
                            "people": None if s.get("crowd") else len(s.get("who") or [])})
        else:
            p_ = figurative(s["action"] or "Small, natural movements; the camera holds still.") + _pose_hold(cx, s) + \
                STILL_CROWD * bool(s.get("crowd"))
            emb = f"q_{s['id']}_{MF._h(p_)}"
            enc_dist.append({"name": emb, "prompt": p_})
            for t, seed in enumerate((QUIET_SEED,) + ((RETAKE,) if s["id"] in (retake or ()) else ())):
                quiet.append({"name": s["id"] + ("" if t == 0 else f"_t{t + 1}"), "emb": emb, "prompt": p_,
                              "width": W, "height": H, "fps": FPS, "seed": seed, "frames": n, "line": s["id"],
                              "route": "i2v", "images": img,
                              "people": None if s.get("crowd") else len(s.get("who") or [])})
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
        # 10-26 user: quiet clips judged as they land too (the same body checks as action, on the idle iGPU), a
        # faulted one's retake queued on the loaded runner
        more, done = L / "quiet_more.json", L / "quiet_more.done"
        more.write_text("[]")
        done.unlink(missing_ok=True)
        stop = threading.Event()
        th = threading.Thread(target=_judge_as_they_land, args=(L / "out", quiet, more, done, stop, log), daemon=True)
        th.start()
        try:
            gen("quiet", quiet, ["--fa-amd", "--blaslt", *(["--decode-b"] if DECODE_B else []),
                                 "--more", more.name, "--more-done", done.name], F._pp_split(sd, "distilled", 17))
        finally:
            stop.set()
            th.join(60)


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
    """Judge each action or quiet shot's first take as soon as its clip exists (the vision judge stays loaded); a faulted one
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
                fa = MF.fight_clip_ok(out_ / f"{n}.mp4", j.get("people"))
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
    """Each action or quiet shot (quiet too since 10-26): its takes judged (melted / merged bodies, fire from hands); the take with the fewest faults
    becomes <id>.mp4 (a tie keeps the first), the other stays as <id>_t2.mp4. -> the shots whose only take is
    faulted (to be taken once more)."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import animate
    subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)   # the judge's iGPU
    out_ = shot_dir(book, stem) / "ltx" / "out"
    again = set()
    for s in shots:
        if s["kind"] not in ("action", "quiet"):
            continue
        a, b = out_ / f"{s['id']}.mp4", out_ / f"{s['id']}_t2.mp4"
        if not a.exists() or (out_ / f"{s['id']}.judged").exists():
            continue
        n_ = None if s.get("crowd") else len(s.get("who") or [])
        fa = MF.fight_clip_ok(a, n_)
        if not b.exists():
            if fa:
                log(f"{s['id']}: {fa}: taken once more")
                again.add(s["id"])
            else:
                (out_ / f"{s['id']}.judged").write_text(json.dumps({"take1": fa}))
            continue
        fb = MF.fight_clip_ok(b, n_)
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
        es = by.pop("_es", None) or es
        shots = [s for e in es for s in by.get(e["id"], [])]
        log(f"resuming the stopped render on its saved plan ({len(shots)} shots; delete {fz} to plan again)")
        return _render(book, stem, es, by, shots, mix, op, fz, log)
    if MF.cover(book, stem, t0, t1, log):  # narrated stretches without a moment get theirs
        es = MF.entries(book, stem, t0, t1)
    wardrobe(book, stem, log)
    es, by, _ = direct(book, stem, es, log, keep_llm=True)    # the director's shots are the moments from here on
    sheets = place_sheets(book, stem, es, by, log)            # then one set sheet per place it used
    from tts_reader import animate
    subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
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
    fz.write_text(json.dumps({"_sig": sig, "_es": es, **{s_["id"].rsplit("_s", 1)[0]: [x for x in shots if x["id"].rsplit("_s", 1)[0] ==
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
