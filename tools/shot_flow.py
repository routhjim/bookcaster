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
import mannequin_flow as MF
import registry as RG          # noqa: E402 - the shared pieces: paths, looks, LoRAs, heroes, judge, mux

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
TALK_PRE = 12                       # frames of silence before a line (0.5 s): the mouth is live at the first word
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
  result: about one beat for every 2 seconds of the shot. More than that is more shots (he walks to the desk; a
  close shot as he unlocks the drawer and takes out the book). A shot may show a thing alone (a clock, a letter, a door) when the passage looks at it.
- THE NARRATION FIRST: whatever the passage says about where someone is or how they are (sitting, lying, holding,
  wearing; "he had been leaning back in his chair" too) is how the picture shows them, whatever earlier shots had.
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
  action will use, set up for the action's first moment (his hand already at the drawer; a thing the action takes out
  is in sight or in an open drawer). Never a person's position or posture here (they are in "people"), nothing in mid-action, never
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
- "state_end": how things stand at the END of the passage: what each one holds, each one's posture and where they
  are, and any visible change to the place (a lamp lit, a door open, a letter on the table). Not their clothes: the
  wardrobe keeps those. Plain words, at most 80.
Reply to each passage with JSON only: {{"state_end": "...", "shots": [{{"start": 0.0, "kind": "quiet", "place": "...", "framing": "...", "people": [{{"name": "...", "side": "left", "pose": "sitting", "at": "...", "facing": "...", "holds": ""}}], "setting": "...", "action": "...", "pose_end": {{"<name>": "sitting"}}}}, {{"start": 4.2, "kind": "talk", "speaker": "...", "listener": "...", "framing": "two"}}]}}"""
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
# 10-27 audit: only rules any book can need, each given only when its passage has it (the passing-effect rule made a
# lit hearth a fault; the making rule fired on "made a remark"; both were one story's band-aids)
RULE_EXTRA = [
    (re.compile(r"\bas if\b|\blike an? \w+|\btongues? of\b|\bin tongues\b", re.I),
     "- Figurative words are drawn as what a viewer would literally see, never the figure of speech: no body part an\n"
     "  idiom names, no wide-open mouths or working jaws for speech.\n"),
    (re.compile(r"\b(fight\w*|fought|sword\w*|attack\w*|lunge\w*|slash\w*|stab\w*|spear\w*|axe|arrows?|kill\w*|"
                r"slain|duel\w*|grappl\w*|wrestl\w*)\b", re.I),
     "- A fight is covered as intent (wide: both fighters, the attacker coils, the other sees the opening), action (both\n"
     "  in the picture: the one blow or move and the one it reaches), reaction (close: the one hit, or a witness). A\n"
     "  fighter's picture has a READY stance, weapon raised toward the opponent; someone knocked down but alive is on the\n"
     "  ground propped on an elbow, eyes open; someone dead lies flat, eyes closed.\n"),
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


# 10-27 user: "Watson effectively disappears for certain shots... you'd need to zoom in on Holmes more": a one-person
# talk shot is a true CLOSE-UP, so the other one is simply out of frame - and a large face is more mouth to sync
TALK_PICTURE = ("a close-up of {who} alone: head and shoulders filling the frame, no one else and no shoulder or head in "
                "the foreground, face turned three-quarters with the eyes just off the camera, mid-word")
# (10-28: "turned three-quarters toward someone just beside the viewer" drew that someone - a blurred shoulder in the
# foreground, an over-the-shoulder two-shot - and 10 close-ups failed the one-person count in all 5 seeds)
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
# body positions (10-26 Acts 3: the lame man was drawn up, down and up again before he was healed; since 10-27 the
# director's own per-person pose is their one owner, and the keys and clips are told it)
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
    out = CARRIED.sub(keep, look or "")
    # 10-28: "...and trousers, topped with a black top-hat" lost its "with ..." here and the key said "trousers,
    # topped." - a verb left without its object goes too; and "His usual" (any case) is the wardrobe's word, not a look
    out = re.sub(r",?\s*\b(?:topped|finished|trimmed|crowned)\b(?=\s*(?:[,.;]|$))", "", out, flags=re.I)
    out = re.sub(r"\b(?:his|her|their)\s+usual\s+", "", out, flags=re.I)
    return re.sub(r"\s*,\s*$", "", out).strip()


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
        first = describe(RG.Book(cx.book), w, _shot_dress(cx, s))     # the one description (10-27 audit)
        out.append(f" {first[0].upper() + first[1:]} stays {POSE_WORDS[c]} throughout.")
    return "".join(out)


def _shot_dress(cx, s: dict) -> dict:
    """Each person's clothes at the shot's own time (the wardrobe), by its id."""
    mid = str(s.get("id") or "").rsplit("_s", 1)[0]
    try:
        return MF._dress(cx.book, cx.stem, mid, s.get("who") or [])
    except Exception:                                   # noqa: BLE001 - no wardrobe: the looks' own clothes
        return {}


def carry_poses(es: list, out: dict, cuts: tuple = ()) -> None:
    """Each shot's "held" pose per person: the director's opening pose for them, or (none given) the pose the shot
    before ended with; "pose_after" = the shot's pose_end."""
    cur, starts = {}, sorted(cuts)
    for e in es:
        if starts and float(e.get("t", -1)) >= starts[0] - 1e-6:   # a new scene: nothing carried across it
            while starts and float(e.get("t", -1)) >= starts[0] - 1e-6:
                starts.pop(0)
            cur = {}
        for s in out.get(e["id"]) or []:
            if s.get("kind") == "talk":
                continue
            p0 = {w: pose_cat(v) for w, v in (s.get("pose") or {}).items()}
            p1 = {w: pose_cat(v) for w, v in (s.get("pose_end") or {}).items()}
            held = {}
            for w in s.get("who") or []:
                # one owner: the director's own opening pose (given per person, as the narration says) wins; the
                # carried one only fills a gap (10-27: Watson "sitting" in bed was carried into the sitting room,
                # where the picture had him standing - the key prompt and its check said "sitting")
                c0 = p0.get(w) or cur.get(w) or ""
                if not c0:
                    continue
                held[w] = c0
                cur[w] = p1.get(w) or c0
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
def closeup(s: dict) -> bool:
    """A one-person talk shot: a true close-up (its key drops the wide room reference and the posture)."""
    return s.get("kind") == "talk" and len(s.get("who") or []) == 1


SETTING = (" The scene takes place in {place}, which fills the whole picture behind them to its edges: a single painted "
           "scene, never a character sheet, never a plain or white background.")
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
    crs += [w for w in who if w not in crs and re.search(r"^\W*(?:an?|the)?\s*(?:\w+\s+){0,3}(dragon|beast|creature|wyrm|wolf|horse|monster)\b",
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
        for sent in re.split(r"(?<=[.;])\s+", act):
            if MF.EFFECT.search(sent) and not re.search(r"\bfrom\b", sent):
                out.append(f"shot {i}: '{MF.EFFECT.search(sent).group(0)}' does not say where it comes from")
        if len(act.split()) > 70:
            out.append(f"shot {i}'s action is over 50 words")
        beats = _beats(act)
        dur = float(s.get("dur") or 0)
        if dur and beats > max(1, round(dur / 1.25)):    # (10-27: turn, walk, unlock, take out - 4 beats in 3 s;
                                                         # 10-28: two small gestures in 2 s are fine, the director
                                                         # couldn't split those and the passage was asked 3 times)
            out.append(f"shot {i} has {beats} actions in {dur:.1f} s: about one per 2 s; split it into shots")
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


def shot_seed(sid: str) -> int:
    """A shot's own seed offset, from its time (10-27 user: Watson's preamble may repeat its desk shots, but the same
    words and seed drew the same pixels: a repeated setup is a fresh take of it)."""
    m = re.search(r"-(\d{4,})", sid or "")
    return 1000 * (int(m.group(1)) % 997) if m else 0


def compose(s: dict, cast: list, crs: list, sides: dict | None = None, resolve=None) -> dict:
    """A shot's "who", "pose", "placed" and "picture" from the director's "people" (name, side, pose, at, facing,
    holds) and "setting" (10-27 user: "the goal is to prevent the errors more than detect and fix them": who stands
    where and in what posture was written twice - the who list's order and the picture's free words - and they
    disagreed: Helen drawn at the desk opened Holmes's drawer). Said once, here. `sides`: the SCENE's one side map
    (0 left, 1 centre, 2 right), read and extended as shots are composed in order (talk shots read it too);
    `resolve`: name -> cast name (the registry). The position words are the key layout's own (10-27 audit: "on the
    centre"; two people "on the left" while the key's boxes go by order)."""
    ppl = [x for x in s.get("people") or [] if isinstance(x, dict)]
    if not ppl and not s.get("setting"):
        return s                        # an old-style answer: as it is
    resolve = resolve or (lambda v: MF._cast_name(v, cast, crs))
    rows = []
    for x in ppl:
        w = resolve(x.get("name"))
        if w and w not in [r[0] for r in rows]:
            rows.append((w, SIDE.get(str(x.get("side") or "").lower(), 1), pose_cat(x.get("pose")) or "stand", x))
    if sides is not None:               # a scene keeps each one's side (10-27: Holmes right, then left, then right: the
        for i, (w, sd_, pz, x) in enumerate(rows):      # 180-degree rule); the first shot that places someone sets it
            if w in sides:
                rows[i] = (w, sides[w], pz, x)
        for w, sd_, pz, x in rows:
            sides.setdefault(w, sd_)
    rows.sort(key=lambda r: r[1])       # left to right, stable
    if sides is not None:               # the order DRAWN is the truth for the rest of the scene: no two people share a
        prev = None                     # side value, so no later shot can break a tie the other way (10-27 user: Helen
        for w, sd_, pz, x in rows:      # left, Holmes right, then Holmes left in the next two-shot - both had "right")
            if prev is not None and sides[w] <= prev:
                sides[w] = prev + 0.01
            prev = sides[w]
    lay = [l for _, l in group_layout(len(rows))]
    parts, placed = [], []
    for (w, sd_, pz, x), side in zip(rows, lay):
        where = str(x.get("at") or "").strip().rstrip(".")
        face = str(x.get("facing") or "").strip().rstrip(".")
        hold = str(x.get("holds") or "").strip().rstrip(".")
        hold = "" if hold.lower() in ("", "nothing", "none") else re.sub(r"^holding\s+", "", hold)
        parts.append(f"{w} {POSE_VERB[pz]}{' ' + where if where else ''}{' ' + side if side else ''}"
                     f"{', facing ' + re.sub(r'^(facing|toward|towards)\s+', 'toward ', face) if face else ''}"
                     f"{', holding ' + hold if hold else ''}")
        placed.append({"name": w, "pose": pz, "at": where, "facing": face, "holds": hold})
    setting = str(s.get("setting") or "").strip().rstrip(".")
    s = dict(s, who=[r[0] for r in rows], pose={r[0]: POSE_WORDS[r[2]].split()[0] for r in rows}, placed=placed,
             picture=("; ".join(parts) + (". " if parts and setting else "") + setting).strip() + ".")
    return s

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


TAG_WORDS = 8                       # a narration entry this short between two lines of one speaker is a speech tag


def _turn_cuts(tl: list) -> dict:
    """The timeline's entry starts sorted by what a cut there would do: "inside" = inside one speaker's turn (their
    next line after a short tag like "said he", or that tag itself); "change" = where the speaker changes or a
    narration paragraph begins."""
    inside, change = set(), set()
    for i, x in enumerate(tl):
        t = float(x["t"])
        prev = tl[i - 1] if i else None
        if x.get("kind") == "quote":
            p1 = prev
            p2 = tl[i - 2] if i >= 2 else None
            tag = p1 is not None and p1.get("kind") != "quote" and len(str(p1.get("text", "")).split()) <= TAG_WORDS
            if (p1 and p1.get("kind") == "quote" and p1.get("speaker") == x.get("speaker")) or \
                    (tag and p2 and p2.get("kind") == "quote" and p2.get("speaker") == x.get("speaker")):
                inside.add(t)                       # the same speaker again, after at most a short tag
            elif prev is None or prev.get("speaker") != x.get("speaker"):
                change.add(t)
        else:
            nxt = tl[i + 1] if i + 1 < len(tl) else None
            short = len(str(x.get("text", "")).split()) <= TAG_WORDS
            if short and prev and nxt and prev.get("kind") == nxt.get("kind") == "quote" and \
                    prev.get("speaker") == nxt.get("speaker"):
                inside.add(t)                       # the tag between two lines of one speaker
            elif not short:
                change.add(t)
    return {"inside": inside, "change": change}


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
    # 10-28 user: "the passage boundary has to be more flexible to fall at an actual shot boundary" - a cut inside one
    # speaker's turn ("Farintosh," / said he. / "Ah yes...") split one close-up into two shots from two keys. A cut
    # never falls inside a turn; a change of speaker is preferred; a passage may run to 1.2 x mx to reach one
    turn = _turn_cuts(tl)
    cuts = sorted({float(x["t"]) for x in tl if t0 < float(x["t"]) < t1})
    marks = sorted(set(cuts) | {ts for ts, _, _, _, _ in sentences(tl) if t0 < ts < t1})
    free = [c for c in cuts if c not in turn["inside"]]
    out, a = [], t0
    while t1 - a > 0.05:
        if t1 - a <= mx:
            out.append((a, t1))
            break
        ok = [c for c in free if c in turn["change"] and a + 0.6 * mx <= c <= a + 1.2 * mx] or \
             [c for c in free if a + 0.6 * mx <= c <= a + 1.2 * mx] or \
             [c for c in marks if a + 0.6 * mx <= c <= a + mx and c not in turn["inside"]] or \
             [c for c in marks if a + 2.0 <= c <= a + mx]
        b = (max(c for c in ok if c <= a + mx) if any(c <= a + mx for c in ok) else min(ok)) if ok else a + mx
        out.append((a, b))
        a = b
    return out


def load_timeline(book: Path, stem: str, chars: list | None = None) -> list:
    """The chapter's timeline with every speaker as their CAST name, by the registry ("Stoner" -> "Helen Stoner",
    "Watson" and "Narrator" -> "Doctor Watson"; 10-27: Helen's talk prompts said "Stoner speaks ... Helen Stoner
    listens with a closed mouth" and the model moved Holmes's lips). Nothing after it sees a second name for anyone;
    narration stays "narrator"."""
    R_ = RG.Book(book)
    tl = json.loads((book / "work" / f"{stem}.timeline.json").read_text())
    for x in tl:
        spk = x.get("speaker")
        if spk and spk != "narrator":
            x["speaker"] = (R_.person(spk) if x.get("kind") == "quote" else R_.person(spk, narrator=False)) or spk
    return tl


def describe(R_, w: str, dress: dict | None = None) -> str:
    """THE description of a person for the video model's words: "the man in the tweed shooting jacket" - one
    builder for actions, pose holds, talk and checks (10-27 audit: four builders, from different times and two
    sources). Their clothes now (the wardrobe), else their look's; carried things left out."""
    lk = MF._appearance(R_.look(w) or "")
    noun = (re.search(r"\b(man|woman|boy|girl|creature|dragon|horse|dog|wolf|beast)\b", lk, re.I) or [None, "figure"])[1]
    cl = garments(uncarried((dress or {}).get(w) or clothes_of(lk))).split(",")[0].strip()
    cl = re.sub(r"^(?:(?:a|an|the|his|her|their|its)\s+)?(?:usual\s+)?", "", cl, flags=re.I)   # (10-27: "in the her usual")
    return f"the {noun.lower()} in the {cl[0].lower() + cl[1:]}" if cl else f"the {noun.lower()}"


def _problem(book: Path, log, msg: str) -> None:
    """Something the job could only work around: in the log at once and in the job's list of problems, so a finished
    job says it (10-27 audit: fallbacks that only logged, and the job said "done")."""
    log(f"!!! {msg}")
    try:
        sys.path.insert(0, str(HERE.parent))
        from tts_reader import bookcaster as BC
        if hasattr(BC, "degrade"):
            BC.degrade(book, msg)
    except Exception:                                   # noqa: BLE001 - reporting must never stop the job
        pass


def _who_full(spk: str, names: list) -> str:
    """A timeline speaker ("Holmes", "Stoner") as a cast name ("Sherlock Holmes", "Helen Stoner")."""
    if not spk or spk == "narrator":
        return ""
    return next((w for w in names if w == spk), None) or next(
        (w for w in names if re.search(rf"\b{re.escape(spk)}\b", w, re.I)), spk)


PERSON_WORD = re.compile(r"\b(he|she|they|him|her|them|his|hers|their|man|woman|men|women|lady|gentleman|girl|boy|"
                         r"figure|person|people)\b", re.I)


def _people_words(text: str, cast=()) -> str:
    """Only the sentences of a picture that are about a person (a cast name or a person word): "a case-book sits on
    the surface" says nothing about anyone's posture."""
    names = [w for c in cast or () for w in [c, *str(c).split()] if len(w) > 2]
    keep = [z for z in re.split(r"(?<=[.;!?])\s+", str(text or ""))
            if PERSON_WORD.search(z) or any(re.search(rf"\b{re.escape(n)}\b", z, re.I) for n in names)]
    return " ".join(keep)


def _no_place_words(text, places=()) -> str:
    """A picture's words without any place's name - a title can imply a posture or a piece of furniture ("Sitting
    Room", "Bedroom", "Huddle Room"), and a posture check must read the people only (10-28 user: handle whatever a
    setting's title implies, not one room). Longest names first, with and without "(inside)"/"(outside)"."""
    t = str(text or "")
    names = set()
    for p in places or ():
        if not p:
            continue
        base = re.sub(r"\s*\((inside|outside)\)$", "", str(p)).strip()
        ws = base.split()
        names |= {str(p), base} | {" ".join(ws[i:]) for i in range(1, len(ws))}   # its short forms: "the sitting room"
    for w in sorted(names, key=len, reverse=True):
        t = re.sub(r"(?<!\w)" + re.escape(w) + r"(?!\w)", " ", t, flags=re.I)
    return t


def _shot_faults(a: dict, cast: list, sec: float, lines: list, places=()) -> list:
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
            said = {c for c, rx in POSE_CAT if rx.search(_people_words(_no_place_words(
                s.get("picture", ""), [*places, s.get("place")]), cast))} & \
                {"sit", "stand", "lie"}     # (10-28: "Baker Street Sitting Room" read as someone sitting, 3 asks;
                                            #  "a case-book sits on the surface" as someone sitting)
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
    RGB = RG.Book(book)                 # who, their looks and the places: the registry (no planner context needed)
    sd = shot_dir(book, stem)
    (sd / "passages").mkdir(parents=True, exist_ok=True)
    sheets = sheets or {}
    span = _spans(es)
    t_a, t_b = float(es[0]["t"]), float(es[-1]["t"]) + span[es[-1]["id"]]
    plan = json.loads((MF.paths(book, stem)[0] / "plan.json").read_text())
    chars = list(dict.fromkeys(p_ for p_ in (RGB.person(x) for x in ((plan.get("look") or {}).get("characters") or {}))
                               if p_)) or RGB.people()
    tl = load_timeline(book, stem, chars)
    sents = sentences(tl)
    nar = RGB.narrator() or ""
    talks = [e for e in es if e["route"] == "talk" and e.get("wav")]
    real = list(dict.fromkeys(e["where"] for e in es if e.get("where") and not NO_PLACE.search(e["where"])))

    class _Rooms(dict):                 # each place's fixed things: the book's one description of it (the registry;
        def get(self, w, d=None):       # 10-27 audit: the director got "(as the narration says)" for every main place)
            r = RGB.place_look(w) if w else ""
            return {"room": r} if r else d
    place_sheet = _Rooms()

    def cast_of(v, cast):               # a name as one of this passage's cast (the registry), else a creature in it
        p_ = RGB.person(v)
        if p_ in cast:
            return p_
        return MF._cast_name(v, [c for c in cast if c in crs], crs) if crs and v else None

    def tokens(w):                      # the words that name `w` alone ("Holmes", not "Stoner" for Helen or Julia)
        return [w] + [x for x in re.findall(r"[A-Z][a-z'\-]+", w) if len(x) > 2 and RGB.person(x) == w]

    def named_in(text, cast):
        return [w for w in cast if any(re.search(rf"\b{re.escape(x)}\b", str(text or "")) for x in tokens(w))]
    # every spoken line's span, by speaker (the voices of the others are silenced in a talk shot's audio)
    spans = {}
    for x, y in zip(tl, tl[1:] + [{"t": float(tl[-1]["t"]) + 0.4 * len(tl[-1]["text"].split())}]):
        if x.get("kind") == "quote" and x.get("speaker") not in (None, "", "narrator"):
            spans.setdefault(x["speaker"], []).append((float(x["t"]), float(y["t"]), x))
    voices = book / "work" / f"{stem}.mp3"
    (sd / "audio").mkdir(parents=True, exist_ok=True)

    def unname(txt, cast, dress):       # the video model reads looks, not names: the one description of each
        for w in sorted(cast, key=len, reverse=True):
            txt = re.sub(r"\b(" + "|".join(map(re.escape, tokens(w))) + r")\b", describe(RGB, w, dress), txt)
        return txt

    def talk_moment(mid, who, t, d, creature=False):   # a talk moment for the line `who` speaks at t
        mine = [(a_, b_, x) for a_, b_, x in spans.get(who, []) if b_ > t and a_ < t + d]
        keep = "+".join(f"between(t,{max(0.0, a_ - t):.3f},{b_ - t:.3f})" for a_, b_, _ in mine) or "0"
        # the voice clip is named by everything it is cut from (10-27 audit: named by the shot's id alone, 15 of 66
        # talk shots reused an earlier run's cut of another length - their lips would have run out of sync)
        vs = voices.stat()
        wav = sd / "audio" / f"{mid}_{MF._h(f'{vs.st_size}|{vs.st_mtime_ns}|{t:.3f}|{d:.3f}|{keep}')}_speech.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-t", f"{d:.3f}", "-i", str(voices),
                            "-ac", "1", "-ar", "16000", "-af", f"volume=0:enable='not({keep})'", str(wav)], check=True)
        x = mine[0][2] if mine else {}
        # built from scratch: the planner's moment fields (another line's pic/first/emotion) no longer leak in
        return {"id": mid, "kind": "talk", "route": "talk", "t": t, "dur": d, "speaker": who, "line": x.get("text", ""),
                "emotion": x.get("emotion") or "", "creature": bool(creature), "speech_s": d, "frames": _frames(d),
                "chunks": [[0, _frames(d)]], "wav": str(wav)}

    def at(t):                          # the planner moment at time t
        return next((e for e in reversed(es) if float(e["t"]) <= t + 1e-6), es[0])
    # the chapter's opening place, unless the planner opened on no place (a title card): then the director picks
    home = "" if es[0].get("where") and NO_PLACE.search(es[0]["where"]) else (real[0] if real else "")
    where_of = {}                       # where each one was last shown: a talk shot is there
    state, prev_shot, people_here = "(the start of the chapter)" if t_a < 1 else "", None, []
    new_es, by, new_sheets = [], {}, {}
    system, talk_ = SHOT_RULES.format(maxp=MAX_PEOPLE, closemax=CLOSE_MAX), []
    pics_before, pics_scene, screen, last_pose, last_at = [], [], {}, {}, {}
    pair = None
    animate_up = False
    scenes = scene_list(book, stem, log)     # the chapter's scenes (shared, cached: the wardrobe reads them too)
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
            pair, people_here, where_of, talk_, screen, last_pose, last_at = None, [], {}, [], {}, {}, {}
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
        looks = {w: MF._brief(MF._appearance(RGB.look(w))) for w in cast}
        people = "; ".join(f"{w} ({looks.get(w, '')}{'; wears ' + dress[w] if dress.get(w) else ''})" for w in cast)
        hs = place_sheet.get(home) or {}
        narr = " ".join(z for *_, k_, z in lines_ if k_ != "quote")
        extra = "".join(r for rx, r in RULE_EXTRA if rx.search(narr))     # the narration's own events only
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
        k = MF._h(system + q0 + DIRECT_EFFORT + str(BC.LLM_MODEL))

        def process(reply):
            """The director's words -> (answer, faults, its screen sides). Runs on every use, a cached answer too: the
            cache holds the model's words only (10-27 audit: code fixes never reached cached passages)."""
            m = re.search(r"\{.*\}", reply or "", re.S)
            try:
                a = json.loads(m.group(0)) if m else {}
            except ValueError:
                a = {}
            a = a if isinstance(a, dict) else {}
            scr = dict(screen)
            res = (lambda v: cast_of(v, cast))
            a["shots"] = [compose(x, cast, crs, scr, res) if isinstance(x, dict) and x.get("kind") != "talk" else x
                          for x in a.get("shots") or []] if isinstance(a.get("shots"), list) else []
            for x in a["shots"]:
                if isinstance(x, dict):
                    x["who"] = list(dict.fromkeys(w for w in (cast_of(v, cast) for v in (x.get("who") or [])) if w))
                    if x.get("action"):
                        x["action"] = unname(str(x["action"]), cast, dress)
            a["shots"] = [x for x in a["shots"] if isinstance(x, dict)]
            faults = _shot_faults(a, cast, sec, quotes, RGB.places() + [home] + others) + _repeats(a, pics_before, pics_scene) + [
                f"shot {i} is at {x.get('place')}: this scene is at {home}" + (f" and may cut to {', '.join(others)}"
                                                                               if others else "")
                for i, x in enumerate(a["shots"], 1) if x.get("kind") != "talk" and scenes and
                not any(RG.place_key(x.get("place")) == RG.place_key(w) for w in [home] + others)] + [
                f"shot {i}'s setting names {', '.join(n_)}: where people are, their posture and clothes go only in "
                f"\"people\"" for i, x in enumerate(a["shots"], 1)
                if x.get("kind") != "talk" and (n_ := named_in(x.get("setting"), cast))]
            return a, faults, scr

        got = json.loads(cp.read_text()) if cp.exists() else {}
        hit = got.get("k") == k and bool(got.get("reply"))
        if hit:
            reply = got["reply"]
            a, faults, scr = process(reply)
            new = [f for f in faults if f not in (got.get("faults") or [])]
            if new:         # 10-28: a rule added since (the desk shot's three verbs in one sentence) only logged on
                hit = False     # a cached answer; the passage that breaks it now is asked again, the others replay
                log(f"  passage {pa:.1f} s: asked again: its cached shots break a newer rule: {new[:2]}")
        if not hit:
            if got.get("reply") and got.get("k") != k:
                log(f"  passage {pa:.1f} s: asked again: {_what_changed(got.get('ask'), system, q0)}")
            if not animate_up:
                animate.igpu("strata", log)
                animate_up = True
            a, faults, scr, reply = {}, [], dict(screen), ""
            if sum(len(m_["content"]) for m_ in talk_) > HISTORY_CHARS:
                talk_[:] = talk_[-4:]   # too long (decode slows with context): go on from the last two passages
            turn = [{"role": "user", "content": q0.replace("{fix}", "")}]
            for _ in range(3):
                reply, cached = _converse(BC.LLM_URL, BC.LLM_MODEL, [{"role": "system", "content": system}] + talk_ +
                                          turn, max_tokens=5000 + 12000 * bool(DIRECT_EFFORT), effort=DIRECT_EFFORT)
                log(f"  passage {pa:.1f} s: {cached} of the prompt's tokens cached")
                a, faults, scr = process(reply)
                if not faults:
                    break
                turn += [{"role": "assistant", "content": reply or ""},
                         {"role": "user", "content": "Your last answer broke the rules: " + "; ".join(faults) +
                          ". Write it again, the whole passage."}]
            if a.get("shots"):          # kept with its own key, faults and all (10-27 audit: a faulted passage was
                cp.write_text(json.dumps({"k": k, "reply": reply, "faults": faults,   # re-asked every run)
                                          "ask": {"system": system, "q": q0}}, indent=1))   # (what a miss changed)
        if faults:
            _problem(book, log, f"passage {pa:.1f}-{pb:.1f} s: the director's shots still break the rules: {faults}")
            for x in a.get("shots") or []:      # what would contradict "people" in the picture leaves the setting
                if x.get("kind") != "talk" and named_in(x.get("setting"), cast):
                    x["setting"] = " ".join(z for z in re.split(r"(?<=[.;])\s+", str(x["setting"]))
                                            if not named_in(z, cast))
            scr = dict(screen)
            a["shots"] = [compose(x, cast, crs, scr, lambda v: cast_of(v, cast)) if x.get("kind") != "talk" else x
                          for x in a.get("shots") or []]
        screen.update(scr)
        if a.get("shots"):               # the conversation goes on with this passage and its answer (the bad tries dropped)
            talk_ += [{"role": "user", "content": q0.replace("{fix}", "")},
                      {"role": "assistant", "content": reply or json.dumps(a)}]
        shots = a.get("shots") or []
        if not shots:                   # nothing usable at all: said so, and the people here, still
            _problem(book, log, f"passage {pa:.1f}-{pb:.1f} s: no usable shots from the director (quiet shots instead)")
            n = max(1, math.ceil(sec / SHOT_MAX))
            shots = [compose({"people": [dict(name=w, side=["left", "right"][j % 2], pose=last_pose.get(w, "standing"),
                                              at=last_at.get(w, "")) for j, w in enumerate(cast[:2])],
                              "setting": "", "start": i * sec / n, "kind": "quiet", "place": home, "framing": "medium",
                              "action": "They listen and look at one another."}, cast, crs, screen,
                             lambda v: cast_of(v, cast)) for i in range(n)]
        st = [float(x.get("start") or 0) for x in shots]
        st[0] = 0.0
        for i in range(1, len(st)):     # starts in order, and at least 1 s apart
            st[i] = min(max(st[i], st[i - 1] + 1.0), sec - 0.5)
        shots, st = _talk_cuts(shots, st, sec, pa, spans, voices, lambda v: cast_of(v, cast))
        keep_ = [i for i in range(len(st)) if i == 0 or (st[i + 1] if i + 1 < len(st) else sec) - st[i] >= SHOT_SHORT]
        shots, st = [shots[i] for i in keep_], [st[i] for i in keep_]     # a too-short shot: the one before runs on
        cuts_ = []
        for x0, x, b in zip(shots, st, st[1:] + [sec]):
            after = None
            if x0.get("kind") == "talk":    # (10-27: 10 of 74 talk shots ran on into the narration after their line)
                who_ = cast_of(x0.get("speaker"), cast)
                ends = [b_ for a_, b_, _ in spans.get(who_, []) if a_ < pa + b and b_ > pa + x]
                end = min(b, max(ends) - pa) if ends else b
                if b - end >= SHOT_SHORT and end - x >= SHOT_SHORT:     # the rest: the listener's reaction
                    after = ({"kind": "reaction", "of": x0.get("listener") or x0.get("speaker"),
                              "to": x0.get("speaker")}, end, b)
                    b = end
            n_ = max(1, math.ceil((b - x) / 8.0)) if x0.get("kind") == "talk" else 1
            edges = [x + (b - x) * j / n_ for j in range(n_ + 1)]
            if n_ > 1:                  # (10-28: these cut at equal fractions, mid-word; now at the voice's pauses)
                env = _envelope(voices)
                for j in range(1, n_):
                    edges[j] = _pause_near(env, pa + edges[j], pa + edges[j - 1] + SHOT_SHORT,
                                           pa + edges[j + 1] - SHOT_SHORT, reach=1.0) - pa
            for j in range(n_):         # a long line: the speaker, the listener's reaction, the speaker (SHOT_RULES)
                piece = x0
                if j % 2 and x0.get("listener"):
                    piece = {"kind": "reaction", "of": x0.get("listener"), "to": x0.get("speaker")}
                elif j:
                    piece = dict(x0, framing="over-the-shoulder")
                cuts_.append((piece, edges[j], edges[j + 1]))
            if after:
                cuts_.append(after)

        def held(w):                    # how someone is in the scene now: from the last shot that placed them
            pz = last_pose.get(w, "")
            return f"{w} {POSE_VERB[pz]}{' ' + last_at[w] if last_at.get(w) else ''}" if pz else ""

        for x0, x, b in cuts_:
            t, d = pa + x, b - x
            if d < 0.4:
                continue
            place = next((w for w in [home] + others if RG.place_key(w) == RG.place_key(x0.get("place"))), home)
            if x0.get("kind") == "reaction":          # the listener's reaction during a long line
                lis, spk = cast_of(x0.get("of"), cast), cast_of(x0.get("to"), cast)
                if not lis:
                    continue
                x0 = {"kind": "quiet", "framing": "close", "place": where_of.get(lis) or home, "who": [lis],
                      "pose": {lis: POSE_WORDS.get(last_pose.get(lis, "stand"), "standing").split()[0]},
                      "picture": (held(lis) or f"{lis} listens") + (f", facing toward {spk}" if spk else "") + ".",
                      "action": f"{describe(RGB, lis, dress)} listens closely, eyes on the speaker, then gives a "
                                f"small nod."}
                place = x0["place"]
            if x0.get("kind") == "talk":
                who = cast_of(x0.get("speaker"), cast)
                if who and any(min(t + d, b_) - max(t, a_) > 0.3 for a_, b_, _ in spans.get(who, [])):
                    mid = f"talk-{round(t * 10):06d}"
                    e = talk_moment(mid, who, t, d, creature=who in crs)
                    tp = where_of.get(who) or home or place    # 10-27: talk shots on the title card / back in the bedroom
                    lis = cast_of(x0.get("listener"), cast)
                    lis = lis if lis and lis != who else None
                    fr = str(x0.get("framing") or "").lower()
                    sh = {"id": f"{mid}_s0", "dur": d, "who": [who], "framing": "close" if fr.startswith("close")
                          else "medium", "kind": "talk", "place": tp, "action": "",
                          "picture": (TALK_CREATURE if who in crs else TALK_PICTURE).format(who=who)}
                    if lis and who not in crs and lis not in crs and not fr.startswith("close"):
                        # the scene's ONE side map, the same the other shots use (10-27 audit: talk shots kept their
                        # own, and Helen and Holmes swapped sides at every cut between a talk shot and the others)
                        if who not in screen and lis not in screen:
                            screen.update({who: 0, lis: 2})
                        screen.setdefault(who, 2 if screen.get(lis, 0) <= 1 else 0)
                        screen.setdefault(lis, 2 if screen[who] <= 1 else 0)
                        lr = sorted((who, lis), key=lambda w: (screen[w], w != who))
                        side = "left" if lr[0] == who else "right"
                        # (10-27 user: the seated/lying variant without "waist up, close to the viewer" broke the lip
                        # sync - one two-shot template again, the one that worked)
                        if pair != {who, lis} or fr.startswith("two"):
                            sh.update(who=lr, listener=lis, picture=TALK_TWO.format(left=lr[0], right=lr[1], who=who))
                        else:
                            sh.update(who=lr, listener=lis, ots=lis, picture=TALK_OTS.format(
                                who=who, side=side, other="right" if side == "left" else "left", listener=lis))
                        pair = {who, lis}
                    hs_ = [held(w) for w in sh["who"] if held(w)] if len(sh["who"]) > 1 else []   # (a close-up:
                    if hs_:                       # none - "beside the bed" put an empty bed and his whole stance in it)
                        sh["picture"] += ". " + "; ".join(hs_)
                    sh["pose_held"] = {w: last_pose[w] for w in sh["who"] if last_pose.get(w)}
                    new_es.append(e)
                    by[mid] = [sh]
                    new_sheets[mid] = dict(place_sheet.get(tp) or {}, elsewhere="")
                    continue
                # a talk shot when its speaker isn't speaking: the people listening, never "speaking" (said so)
                _problem(book, log, f"{t:.1f} s: a talk shot of {who or x0.get('speaker')} while they are not speaking")
                ppl_ = [w for w in [who, cast_of(x0.get("listener"), cast)] if w] or cast[:1]
                x0 = {"kind": "quiet", "framing": "medium", "who": ppl_, "place": home,
                      "pose": {w: POSE_WORDS.get(last_pose.get(w, "stand"), "standing").split()[0] for w in ppl_},
                      "picture": "; ".join(held(w) or w for w in ppl_) + ".", "action": "They look at one another."}
            mid = f"action-{round(t * 10):06d}"
            new_es.append({"id": mid, "kind": "action", "t": t, "dur": d, "route": "shot", "where": place,
                           "who": x0.get("who") or []})
            sh = {k_: v for k_, v in x0.items() if k_ not in ("start", "speaker", "listener")}
            sh.update(id=f"{mid}_s0", dur=d, place=place, framing=x0.get("framing") if x0.get("framing") in
                      ("wide", "medium", "close") else "medium",
                      kind=x0.get("kind") if x0.get("kind") in ("action", "quiet") else "quiet")
            by[mid] = [sh]
            new_sheets[mid] = dict(place_sheet.get(place) or {}, elsewhere="")
            prev_shot = sh
            for pl in sh.get("placed") or []:  # how each one is now, for the talk and reaction shots that follow
                end = pose_cat((sh.get("pose_end") or {}).get(pl["name"])) or pl["pose"]
                last_pose[pl["name"]] = end
                if pl.get("at"):
                    last_at[pl["name"]] = pl["at"]
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
    # 10-28 user: "Farintosh," and the next line jittered between them - two close-ups of Holmes, each from its own
    # key, because a passage boundary fell between them (the same-speaker merge works inside a passage only). Across
    # the whole stretch now: back-to-back talk shots of one speaker in one framing and place are ONE shot - one key,
    # one chained render; the narration between them is silent in its voice cut, so the lips rest there
    merged = []
    for e in new_es:
        p_ = merged[-1] if merged else None
        if p_ and e.get("kind") == p_.get("kind") == "talk" and e.get("speaker") == p_.get("speaker") and \
                len(by.get(e["id"]) or []) == len(by.get(p_["id"]) or []) == 1:
            sp, se = by[p_["id"]][0], by[e["id"]][0]
            if all(sp.get(k) == se.get(k) for k in ("framing", "place", "who")) and \
                    abs(float(p_["t"]) + float(p_["dur"]) - float(e["t"])) < 0.05:
                d = float(p_["dur"]) + float(e["dur"])
                merged[-1] = talk_moment(p_["id"], p_["speaker"], float(p_["t"]), d, creature=p_.get("creature"))
                sp["dur"] = d
                by.pop(e["id"], None)
                new_sheets.pop(e["id"], None)
                continue
        merged.append(e)
    new_es = merged
    carry_poses(new_es, by, tuple(x["start"] for x in scenes[1:]))
    return new_es, by, new_sheets


PLACE_ASK = """Describe the place "{where}" as a painter would see it {side}, empty: no people and no animals. One sentence
of under 35 words: the ground, the buildings or the landscape, what stands there, the light. {period} Use what this
passage says and things that belong in such a place. Only the place itself: nothing the story's people carry, make
or bring, and no light without its source.

Passage: {text}

Answer with the sentence only."""


# 10-24: the scene room hung "a dark blue sword above the door" (the sword the story makes) despite the rule
# and (10-24 King's Hall) "a massive black dragon silhouette" in its windows in every key
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


def scene_list(book: Path, stem: str, log=print) -> list:
    """The chapter's scene breakdown from the same inputs wherever it is asked for (the director, the wardrobe), so
    it is made once and shared from its cache: the whole chapter's cast, places and timeline, whatever the span."""
    sys.path.insert(0, str(HERE.parent))
    R_ = RG.Book(book)
    plan = json.loads((MF.paths(book, stem)[0] / "plan.json").read_text())
    chars = list(dict.fromkeys(p_ for p_ in (R_.person(x) for x in ((plan.get("look") or {}).get("characters") or {}))
                               if p_)) or R_.people()
    tl = load_timeline(book, stem, chars)
    fp = MF.paths(book, stem)[0] / "flow3d" / "plan.json"       # the planner's places, when it has planned
    real = [e["where"] for e in (MF.entries(book, stem, 0, 1e9) if fp.exists() else [])
            if e.get("where") and not NO_PLACE.search(e["where"])]
    known = list({RG.place_key(w): w for w in R_.places() + real}.values())   # each place once, by its one name
    return breakdown(book, stem, tl, chars, known, R_.narrator() or "", log)


def scene_notes(book: Path, stem: str, log=print) -> str:
    """How each scene opens, by the chapter paragraph it opens in: "[from paragraph 3] Watson in a nightshirt ..."."""
    sys.path.insert(0, str(HERE.parent))
    from tts_reader.score import paragraphs
    paras = [re.sub(r"\s+", " ", p) for p in paragraphs((book / "chapters" / f"{stem}.txt").read_text())]
    tl = load_timeline(book, stem)
    out = []
    for sc in scene_list(book, stem, log):
        x = min(tl, key=lambda y: abs(float(y["t"]) - float(sc["start"])))
        k = re.sub(r"\s+", " ", str(x.get("text") or ""))[:40]
        i = next((j for j, p in enumerate(paras) if k and k in p), None)
        if i is not None and sc.get("opens"):
            out.append(f"[from paragraph {i}] at {sc['place']}: {sc['opens']}")
    return "\n".join(out)


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
            out[e["id"]] = {"scene": sid, "room": a.get("room", ""), "fixtures": list(a.get("fixtures") or []),
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
    text = (book / "chapters" / f"{stem}.txt").read_text()
    R_ = RG.Book(book)                  # the narrator in the plan's own spelling of them (the "I" of the text)
    nar = R_.key_in(R_.narrator(), video.looks(plan)[0]) if R_.narrator() else ""
    notes = scene_notes(book, stem, log)          # the breakdown (cached; it brings up the language model if not)
    if plan.get("wardrobe_k") == video.wardrobe_inputs_key(plan, text, nar or "", notes):
        return                          # current for these rules, looks, chapter and scene notes: nothing to ask
    animate.igpu("strata", log)
    if video.ensure_wardrobe(plan, text, BC.LLM_URL, BC.LLM_MODEL, log, narrator=nar or "", notes=notes):
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
    R_ = RG.Book(book)                  # a place is described ONCE, under its one name (10-27 audit: "Baker Street
    for s in shots:                     # Bedroom (inside)" had a description, so this wrote a second, contradicting one)
        w, room = s.get("place") or "", s.get("room") or ""
        base = R_.place(w) or re.sub(r"\s*\((inside|outside)\)$", "", w)
        mine = (look.get("_described") or {}).get(base)
        if not w or not room or base in done or R_.place_look(w):
            continue            # 10-28: described once, kept (rewritten from each run's set sheet, it changed the
                                # director's passage prompts and his cached answers were asked again)
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
            slot, _ = RG.place_slot(places, w, f"{sd_.capitalize()}: {room}")   # its one entry (10-27 audit)
            places[slot] = f"{sd_.capitalize()}: {room}"
            look.setdefault("_described", {})[base] = rk
            log(f"place described (scene sheet): {slot}: {places[slot]}")
    if done:
        lp.write_text(json.dumps(look, indent=1))
    todo = {}
    R_ = RG.Book(book)
    for s in shots:
        w = s.get("place") or ""
        base = R_.place(w) or re.sub(r"\s*\((inside|outside)\)$", "", w)
        slug = re.sub(r"[^a-z0-9]+", "_", w.lower()).strip("_")
        # a description this step wrote with older rules is written again (10-24: "a dark blue sword hanging above
        # the door" of Outside Hulda's Forge put the story's sword in every shot there)
        mine = (look.get("_described") or {}).get(base)
        if not w or base in done or (R_.place_look(w) and (mine in (None, ask_k) or str(mine).startswith("room:"))) or \
                (not R_.place_look(w) and (vd / "heroes" / f"{slug}.empty").exists()):
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
    R_ = RG.Book(book)                  # the looks and the style have one owner each (10-27 audit: the plan's copies
    looks = {w: no_pose(MF._appearance(R_.look(w))) for w in s["who"]}      # of the looks, art_style.txt only)
    style = _key_style(R_.picture_style())     # the picture's style (10-27: realistic unless `bookcaster art`)
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
            hair = next((x.strip() for x in re.split(r",|;", lk) if re.search(r"\bhair\b", x, re.I)), "")
            cl = garments(uncarried(dress.get(w) or clothes_of(lk)))
            people.append(f"{w}, seen from behind{side}: {re.split(r',', lk)[0]}"   # (10-27: the back of a stranger
                          f"{', ' + hair if hair else ''}{', wearing ' + cl if cl else ''}")   # in green, not Helen)
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
    if closeup(s):                      # 10-27: three close-ups came out medium shots beside an empty bed
        people = [re.sub(r", (standing|sitting|lying down|kneeling|crouching low)(?=[,:])", "", x) for x in people]
        cont, setting = "", (f" Behind the face, softly out of focus, a glimpse of {place}." if place else "")
        held = set((s.get("pose_held") or {}).values())     # 10-28 user: "Holmes is sitting then standing"
        # 10-28 A/B (4 wordings x 6 seeds, chair judged): the posture in the subject took chairs from 3/6 to 1/6
        # (the chair check rejects the rest); leaving the room's name out too gave 0/6 - not a clear difference on 6
        # seeds, and the user keeps the real background: the posture only
        # every held posture stated in the subject (10-28 user: a place's title implies one - "Sitting Room" sat
        # Holmes down, "Bedroom" would lay someone down - so the director's pose is said outright, whatever the room)
        word = {"stand": "standing upright", "sit": "seated", "lie": "lying down", "kneel": "kneeling",
                "crouch": "crouching low"}.get(next(iter(held)) if len(held) == 1 else "")
        if word:
            pic = re.sub(r"^(a close-up of .+? alone)(:)", rf"\1, {word}\2", pic, count=1)
        if held == {"stand"}:
            setting += (" Standing upright, seen from the chest up; behind the shoulders only the bare wall, curtains "
                        "and window, at standing height.")
        elif held == {"sit"}:
            setting += " Seated: the top of a chair back behind the shoulders."
    else:
        setting = SETTING.format(place=place) if place else ""
    return (f"{str(s.get('framing') or 'medium').capitalize()} shot{' in ' + place if place else ''}. {pic}. "
            f"{'. '.join(people)}.{cont}{setting} "
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
    R_ = RG.Book(cx.book)
    who = R_.person(e["speaker"]) or MF._cast_name(e["speaker"], s.get("who") or [], []) or e["speaker"]
    dr = _shot_dress(cx, dict(s, who=[who]))        # the one look, in the clothes of the moment, nothing carried
    lk = R_.look(who) or ""
    if dr.get(who):
        lk = re.split(r",?\s*\bwearing\b", lk)[0] + ", wearing " + dr[who]
    look = MF._brief(uncarried(lk))
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
    held = s.get("pose_held") or {}              # 10-28 user: the place's title can imply a posture ("Sitting Room");
    def posed(w):                                 # the held one is said outright
        return f", {POSE_WORDS[held[w]]}" if held.get(w) in POSE_WORDS else ""
    if other and not s.get("ots"):                # a two-shot: the other face is in the picture, never the speaker's
        lst = f" {other}{posed(other) + ',' if posed(other) else ''} listens with a closed mouth."
    pos = f" on the {'left' if s['who'][0] == who else 'right'}" if len(s.get("who") or []) == 2 else ""
    return (f"Static camera, the framing does not change. {who}{pos}{posed(who)}, {look}{at}, speaks{emo} lips and jaw moving "
            f"clearly in sync with {pron} voice.{lst} No text anywhere in the picture.")


def _key_questions(cx, s: dict) -> tuple:
    """The vision questions for a shot's opening picture, and how many people it should hold."""
    # (10-24: no "does it show the words" question - it only chose between near-identical clean seeds)
    # 10-27: "...letters, a page, a panel..." said yes to the papers, books and frames of Holmes's study (250 of 276
    # Speckled Band keys failed it): only writing laid OVER the picture
    q = {"text": "Answer yes or no: is there a caption, title, lettering or a border laid over this picture, like text "
                 "added on top of a painting (books, papers and framed pictures in the scene don't count)?"}
    hum = 0
    for i, w in enumerate(s["who"]):
        lk = RG.Book(cx.book).look(w)    # the one look (10-27 audit: cx.look at t=0, other spellings)
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
        elif first and pz == "stand" and closeup(s):       # (10-28: close-ups of a standing Holmes read as seated;
            q[f"chair{i}"] = (f"Answer yes or no: is there a chair, sofa or bed right behind or under {first}?")
            # "does he look like he is standing" can't be seen in a head-and-shoulders picture: "No" for every seed)
        if "four-legged" in MF._appearance(lk) and not MF.DOWN.search(s["picture"]):
            q[f"legs{i}"] = (f"Answer yes or no: is the {'dragon' if 'dragon' in lk else 'creature'}'s body "
                             f"horizontal like a lizard's, with its front feet on the ground (not rearing upright "
                             f"on its hind legs)?")
        hum += not re.search(r"^\W*(?:an?|the)?\s*(?:\w+\s+){0,3}(dragon|beast|creature|wyrm|wolf|horse|monster)\b", lk, re.I)
    if hum and not s.get("crowd"):
        q["count"] = "How many people are in this picture? Answer with a number."
    if s.get("kind") == "talk" and len(s.get("who") or []) == 1:   # a one-person talk shot is a close-up (10-27)
        q["close"] = "Answer yes or no: is this a close-up, the person's head and shoulders filling most of the picture?"
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
        R_ = RG.Book(book)
        miss = [w for w in s["who"] if w != s.get("ots") and not R_.lora(w)[0]]
        if miss:                        # never a key drawn from words alone (10-27: Watson had no LoRA, nobody said)
            raise RuntimeError(f"{s['id']}: no LoRA for {miss} (bookcaster trains them before the keys)")
        loras = [[R_.lora(w)[0], min(lw, CREATURE_LORA) if w in crs else lw, bx]
                 for w, bx in zip(s["who"], boxes) if w != s.get("ots")]
        p = _key_prompt(book, stem, s["id"].rsplit("_s", 1)[0], s)
        hero = None if closeup(s) else heroes.get(s.get("place") or None)
        # each LoRA's BUILD is in the sig (10-27 audit: a retrain keeps the file name, so its old keys were kept)
        sig = MF._h(json.dumps([p, loras, [R_.lora(w)[2] for w in s["who"]], str(hero)]))
        # a place picture drawn again (10-24: the people-free hero redraws left every key there cached, so the
        # rerun changed nothing): the keys drawn before it are drawn again
        older = hero is not None and hero.exists() and any(k.stat().st_mtime < hero.stat().st_mtime
                                                          for k in d.glob("key_*.png"))
        if older or ((d / "key.sig").exists() and (d / "key.sig").read_text() != sig):
            for k in d.glob("key*.png"):    # the shot changed: its pictures drawn again
                k.unlink()
        (d / "key.sig").write_text(sig)
        ks0 = MF._seed_keys(d)
        q, need = _key_questions(cx, s)
        qk = MF._h(json.dumps([q, need], sort_keys=True))
        if ks0 and (d / "key.png").exists() and (d / "key.png").stat().st_mtime >= max(k.stat().st_mtime for k in ks0) \
                and (d / "key.q").exists() and (d / "key.q").read_text() == qk:
            continue                        # picked before against the same questions, no seed newer: kept
        (d / "key.q").write_text(qk)        # (10-27 audit: new questions never re-judged an old key)
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
                    job = dict(t["job"], id=f"{t['s']['id']}_key_{seed}", seed=seed + shot_seed(t["s"]["id"]), path=str(k))
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


def _clean(a: dict, k: Path, need: int) -> bool:
    """A seed's answers pass: no text, no white sheet at the edges, everyone there, legs right, the head count."""
    import characters as CH
    c = re.search(r"\d+", str(a.get("count") or ""))
    return bool(a) and (not CH._yes(a.get("text")) and not MF._blank_edges(k)
                        and all(CH._yes(a.get(x)) for x in a if x.startswith(("who", "legs", "pose", "close")))
                        and not any(CH._yes(a.get(x)) for x in a if x.startswith("chair"))      # (must be "no")
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
            # 10-28 user: "Holmes lips do not move at all in the Farintosh line" - a line spoken from the clip's
            # first frame got lips ~0.4 s late (closed for 10 frames, then moving into the silence after). Each talk
            # clip now starts TALK_PRE frames of silence early, the mouth live when the words come; the assembly
            # drops them (talk_pre)
            lead = L / "audio" / f"{mid}_lead.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", e["wav"], "-af",
                            f"adelay={TALK_PRE / 24 * 1000:.0f}:all=1", str(lead)], check=True)
            n_all = max(st + nn for st, nn in e["chunks"]) + TALK_PRE
            e["chunks"] = R.talk_chunks(n_all, R.audio_env(lead, n_all + TALK_PIECE), TALK_PIECE,
                                        TALK_CHAIN_OVERLAP)
            pad = L / "audio" / f"{mid}_pad.wav"
            n_all = max(st + nn for st, nn in e["chunks"])
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(lead), "-ac", "2", "-ar", "48000", "-af",
                            f"apad=whole_dur={n_all / 24:.4f}", str(pad)], check=True)
            p_ = _talk_prompt(R, cx, e, s.get("place") or "", s)
            emb = f"t_{mid}_{MF._h(p_)}"
            enc_dev += [{"name": emb, "prompt": p_}, {"name": emb + "__neg", "prompt": neg}]
            for k, (st, nn) in enumerate(e["chunks"]):
                talk.append({"name": s["id"] if len(e["chunks"]) == 1 else f"{s['id']}_c{k}", "emb": emb, "prompt": p_,
                             "width": W, "height": H, "fps": FPS, "frames": nn, "audio": str(pad),
                             "seed": (int((sd / mid / s["id"] / "talk_seed").read_text())
                                      if (sd / mid / s["id"] / "talk_seed").exists() else 42) + shot_seed(s["id"]),
                             "audio_start": round(st / 24, 4), "line": mid, "speaker": e.get("speaker"),
                             "audio_src": e["wav"], "people": len(s.get("who") or []) or None,
                             "speaker_box": _speaker_box(s, e.get("speaker")),
                             "speech_s": e.get("speech_s"), "chunk": [k, len(e["chunks"]), st, nn], "pre": TALK_PRE,
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
                            "width": W, "height": H, "fps": FPS, "seed": seed + 7 * bool(s["id"] in (redo or ())) + shot_seed(s["id"]),
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
                              "width": W, "height": H, "fps": FPS, "seed": seed + shot_seed(s["id"]), "frames": n, "line": s["id"],
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
                old.get("frames") != j["frames"] or old.get("chain_from") != j.get("chain_from") or \
                old.get("audio_src") != j.get("audio_src")         # 10-27: a new voice cut is a new clip
        if stale:
            c.unlink()
            if not re.search(r"_t\d+$", j["name"]):    # (10-27 audit: an old _t2 or .judged outlived a new take)
                for x in (L / "out" / f"{j['name']}_t2.mp4", L / "out" / f"{j['name']}.judged"):
                    x.unlink(missing_ok=True)
    for nm, x in (("enc_dev", enc_dev), ("enc_dist", enc_dist), ("talk_jobs", talk), ("two_jobs", two),
                  ("quiet_jobs", quiet), ("two_talk_jobs", two + talk)):
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
    if two or talk:
        # 10-28 user: action and talk in ONE hold (the same dev DiT, loaded once). On the iGPU meanwhile: each action
        # take judged as it lands (10-24: a faulted one's retake joins the loaded runner's queue), and each talk shot
        # measured by SyncNet and body-checked as it lands (a faulted one-piece talk shot retaken the same way)
        more, done = L / "two_more.json", L / "two_more.done"
        more.write_text("[]")
        done.unlink(missing_ok=True)
        stop = threading.Event()
        ths = []
        if two:
            ths.append(threading.Thread(target=_judge_as_they_land, daemon=True,
                                        args=(L / "out", two, more, L / "two_more.adone", stop, log)))
        if talk:
            ths.append(threading.Thread(target=_sync_as_they_land, daemon=True,
                                        args=(book, R, L, talk, stop, log, True)))
        for th in ths:
            th.start()

        def closer():                   # no more retakes once every checker is through its first takes
            for th in ths:
                th.join()
            done.write_text("done")
        tc = threading.Thread(target=closer, daemon=True)
        tc.start()
        kind = "two_talk" if two and talk else "two" if two else "talk"
        try:
            gen(kind, two + talk, ["--ti2v"] * bool(two) + ["--a2vid"] * bool(talk) +
                ["--fa-amd", "--blaslt", "--more", more.name, "--more-done", done.name],
                F._pp_split(sd, "a2vid" if talk else "ti2v", 25 if talk else 24))
        finally:
            stop.set()
            for th in ths:
                th.join(600)
        if talk:
            _stitch_talk(book, R, L, talk, log)             # what the thread didn't get to (cached: seconds)
            _sync_offsets(book, L, sorted({j["name"].rsplit("_c", 1)[0] for j in talk}), log)
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
                extra.append(dict(j, name=f"{n}_t2", seed=RETAKE + shot_seed(n)))
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


NOT_VERB = {"his", "hers", "its", "this", "was", "is", "has", "as", "us", "yes", "always", "towards", "less",
            # plural nouns that open a clause ("sits still, hands in her lap")
            "hands", "eyes", "arms", "legs", "lips", "fingers", "shoulders", "papers", "books", "boots", "clothes",
            "curtains", "flames", "shadows", "candles", "windows", "tears", "brows", "knees", "palms", "others"}


def _beats(act: str) -> int:
    """The actions in a shot's action text: its sentences, and within a sentence each clause that starts with a verb
    (10-28: "turns away toward the desk, unlocks the drawer, and takes out a small case-book" was one sentence, so
    one beat by the sentence count, and the director put three actions in 2.5 s)."""
    n = 0
    for sent in re.split(r"(?<=[.!?;])\s+", act.strip()):
        if len(sent.split()) <= 2:
            continue
        parts = re.split(r",\s*(?:and\s+|then\s+)?|\s+(?:and|then)\s+(?:then\s+)?", sent)
        verbs = [p_ for p_ in parts[1:] if _verbish((p_.split() or [""])[0])]
        lead = any(_verbish(w) for w in parts[0].split()[1:])   # "The man ..., with his hat, stands" = one action
        n += max(1, len(verbs) + lead)
    return n


def _verbish(w: str) -> bool:
    """A present-tense verb, roughly: "unlocks", "takes" - not "dress", "glasses", "his"."""
    w = w.lower().strip(",.;:")
    return len(w) > 2 and w.endswith("s") and not w.endswith(("ss", "ous", "'s")) and w not in NOT_VERB


_ENV = {}


def _envelope(wav: Path):
    """The voice track's loudness, 100 values a second (cached per file)."""
    import numpy as np
    k = (str(wav), wav.stat().st_mtime_ns if wav.exists() else 0)
    if k not in _ENV:
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "4000", "-f", "f32le", "-"],
                             capture_output=True).stdout if wav.exists() else b""
        x = np.frombuffer(raw, np.float32)
        x = x[:len(x) // 40 * 40].reshape(-1, 40)
        _ENV.clear()
        _ENV[k] = np.sqrt((x ** 2).mean(1)) if len(x) else np.zeros(0)
    return _ENV[k]


def _pause_near(env, t: float, lo: float, hi: float, reach: float = 0.6) -> float:
    """The middle of the quietest 120 ms within `reach` of t (inside [lo, hi]) when it is a real pause - under a
    third of the voice's typical loudness around it; else t (no pause there: the cut stays put)."""
    import numpy as np
    a, b = int(max(lo, t - reach) * 100), int(min(hi, t + reach) * 100)
    if b - a < 12 or b > len(env):
        return t
    w = np.convolve(env[a:b], np.ones(12) / 12, mode="valid")
    ctx = env[max(0, int((t - 3) * 100)):int((t + 3) * 100)]
    i = int(np.argmin(w))
    return (a + i + 6) / 100 if w[i] < 0.33 * float(np.median(ctx)) else t


def _talk_cuts(shots: list, st: list, sec: float, pa: float, spans: dict, voices: Path, who_of) -> tuple:
    """10-28 user: a line was chopped into four ~2 s close-ups of the same speaker, cut mid-phrase ("opal |
    tee-ar-uh"), and "Watson" got a sliver before the cut. The talk shots' cuts follow the voice now: back-to-back
    shots of one speaker in one framing are one shot (its clip is chained pieces, seamless), and a cut still inside
    someone's line moves to the nearest real pause."""
    def fr(x):
        return "close" if str(x.get("framing") or "").lower().startswith("close") else str(x.get("framing") or "")
    keep = [0]
    for i in range(1, len(shots)):
        p, x = shots[keep[-1]], shots[i]
        if p.get("kind") == x.get("kind") == "talk" and who_of(p.get("speaker")) == who_of(x.get("speaker")) and \
                fr(p) == fr(x):
            continue
        keep.append(i)
    shots, st = [shots[i] for i in keep], [st[i] for i in keep]
    lines = [(a_, b_) for v in spans.values() for a_, b_, _ in v]
    if len(st) > 1 and lines:
        env = _envelope(voices)
        for i in range(1, len(st)):
            t = pa + st[i]
            if any(a_ + 0.2 < t < b_ - 0.2 for a_, b_ in lines):        # inside a spoken line
                lo = pa + st[i - 1] + SHOT_SHORT
                hi = pa + (st[i + 1] if i + 1 < len(st) else sec) - SHOT_SHORT
                st[i] = _pause_near(env, t, lo, hi) - pa
    return shots, st


def _what_changed(old, system: str, q: str) -> str:
    """Why a cached passage missed (10-28: 43 passages were asked again and nothing said why): the first words that
    differ between what it was asked then and now."""
    if not old:
        return "the cache has no record of the old question (written before 10-28)"
    import difflib
    for part, a_, b_ in (("the system prompt", old.get("system") or "", system), ("the passage", old.get("q") or "", q)):
        if a_ != b_:
            sm = difflib.SequenceMatcher(None, a_, b_, autojunk=False)
            op = next(o for o in sm.get_opcodes() if o[0] != "equal")
            return (f"{part} changed: {a_[max(0, op[1] - 40):op[2] + 40]!r} -> {b_[max(0, op[3] - 40):op[4] + 40]!r}")[:400]
    return "the effort or the model changed"


SYNCNET = Path.home() / "src" / "syncnet_python"     # joonson/syncnet_python + its weights, deps in _deps/
SYNCNET_PY = Path.home() / "tts-ab" / ".venv-charlora-igpu" / "bin" / "python"    # ROCm torch for the iGPU
SYNC_MIN_CONF = 3.0             # SyncNet's usual bar; a weaker reading moves nothing (10-28: the two weak ones read
SYNC_MAX_SHIFT = 12             # 0.3-0.5 s, a confident shift was 1-2 frames - the user picked confident-only)
SYNC_VSHIFT = 15                # run_syncnet.py's search range (+-15 frames at 25 fps)


def talk_pre(clip: Path) -> int:
    """The silent lead-in frames a talk clip was rendered with (its job's "pre"; a stitched line's first piece's);
    0 for clips from before the lead-in."""
    for j in (clip.with_suffix(".json"), clip.with_name(clip.stem + "_c0.json")):
        try:
            return int(json.loads(j.read_text()).get("pre") or 0)
        except (OSError, ValueError):
            continue
    return 0


def sync_shift(clip: Path) -> int:
    """The frames (24 fps) to show a talk clip earlier (+) or later (-) so its lips meet its voice: SyncNet's
    measured offset when it is confident, else 0."""
    f = clip.with_suffix(".sync.json")
    try:
        m = json.loads(f.read_text())
    except (OSError, ValueError):
        return 0
    st = clip.stat()
    if m.get("sig") != f"{st.st_size}:{st.st_mtime_ns}" or m.get("conf") is None or m["conf"] < SYNC_MIN_CONF:
        return 0
    if abs(m["offset"]) >= SYNC_VSHIFT - 1:     # at the edge of the search: the real offset may lie beyond it, or
        return 0                                # the match is degenerate (10-28: "Farintosh," read -15 of 15)
    sh = round(m["offset"] * FPS / 25)          # SyncNet counts 25 fps frames; + = the picture lags the sound
    return max(-SYNC_MAX_SHIFT, min(SYNC_MAX_SHIFT, sh))


def _speaker_box(s: dict, speaker) -> list | None:
    """The speaker's zone of the picture [x0, y0, x1, y1] (0-1) when others share the shot: the layout box the key
    drew them in (group_layout, left to right in "who" order) - for any number of people (10-28 user: crop to the
    speaker's own zone, not a half, once talk shots hold more than two). None for a close-up."""
    who = s.get("who") or []
    if len(who) < 2 or speaker not in who:
        return None
    return group_layout(len(who))[who.index(speaker)][0]


def _job_speaker_box(L: Path, name: str) -> list | None:
    """A talk clip's speaker zone from its job (one piece, or a joined line's first piece)."""
    for j in (L / "out" / f"{name}.json", L / "out" / f"{name}_c0.json"):
        try:
            return json.loads(j.read_text()).get("speaker_box")
        except (OSError, ValueError):
            continue
    return None


def _sync_offsets(book: Path, L: Path, names: list, log=print, quiet: bool = False) -> None:
    """SyncNet on each talk clip against its own voice (on the iGPU), kept beside the clip as <name>.sync.json and
    measured again only when the clip changes; the assembly shifts by it (sync_shift)."""
    if not (SYNCNET / "data" / "syncnet_v2.model").exists() or not SYNCNET_PY.exists():
        _problem(book, log, f"lip-sync offsets not measured: no SyncNet at {SYNCNET}")
        return
    ig = subprocess.run([str(Path.home() / "bin" / "hip-index"), "igpu"], capture_output=True, text=True).stdout.strip()
    env = {**os.environ, "PYTHONPATH": str(SYNCNET / "_deps"), "HIP_VISIBLE_DEVICES": ig, "HSA_CU_MASK": f"{ig}:0-38"}
    work = L / "syncnet"
    done = shifted = 0
    for n in names:
        c = L / "out" / f"{n}.mp4"
        if not c.exists():
            continue
        f, st = c.with_suffix(".sync.json"), c.stat()
        sig = f"{st.st_size}:{st.st_mtime_ns}"
        try:
            if json.loads(f.read_text()).get("sig") == sig:
                continue
        except (OSError, ValueError):
            pass
        v = c
        bx = _job_speaker_box(L, n)
        if bx:                  # 10-28 user: SyncNet sees the speaker's own zone only (never a listener's face)
            work.mkdir(parents=True, exist_ok=True)
            v = work / f"zone_{n}.mp4"
            x0, y0, x1, y1 = bx
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(c), "-vf",
                            f"crop=trunc(iw*{x1 - x0:.4f}/2)*2:trunc(ih*{y1 - y0:.4f}/2)*2:iw*{x0:.4f}:ih*{y0:.4f}",
                            "-c:v", "libopenh264", "-b:v", "6M", "-c:a", "copy", str(v)], capture_output=True)
        subprocess.run([str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(v), "--reference", n, "--data_dir",
                        str(work), "--min_track", "25", "--min_face_size", "60", "--overwrite"], cwd=SYNCNET, env=env,
                       capture_output=True)
        r = subprocess.run([str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(v), "--reference", n, "--data_dir",
                            str(work)], cwd=SYNCNET, env=env, capture_output=True, text=True)
        out = r.stdout + r.stderr
        offs = [int(x) for x in re.findall(r"AV offset:\s*(-?\d+)", out)]
        confs = [float(x) for x in re.findall(r"Confidence:\s*([\d.]+)", out)]
        i = max(range(len(confs)), key=confs.__getitem__) if confs and len(confs) == len(offs) else None
        f.write_text(json.dumps({"sig": sig, "offset": offs[i] if i is not None else 0,
                                 "conf": confs[i] if i is not None else None}))
        done += 1
        shifted += bool(sync_shift(c))
    if done and not quiet:
        log(f"lip sync: {done} talk clip(s) measured, {shifted} shifted")
    elif done:
        log(f"lip sync: {names[0]} measured" + (", shifted" if shifted else ""))
    shutil.rmtree(work, ignore_errors=True)


def _sync_as_they_land(book: Path, R, L: Path, talk: list, stop, log=print, check: bool = False) -> None:
    """While the talk clips render on the XTXs: each finished shot (a one-piece clip, or a line whose pieces are all
    there, joined first) is measured by SyncNet on the iGPU at once, so nothing waits for it after the render; and
    body-checked like action and quiet clips (10-28 user: talk clips had no clip check) - a fault is a counted defect
    (a job problem) to remove at its cause, not a retake (lean: fewer defects, not more catching)."""
    shots = {}
    for j in talk:
        shots.setdefault(j["name"].rsplit("_c", 1)[0] if len(j.get("chunk") or []) == 4 and j["chunk"][1] > 1
                         else j["name"], []).append(j)
    left = dict(shots)
    try:
        while left and not stop.is_set():
            ready = [n for n, js in left.items() if all((L / "out" / f"{j['name']}.mp4").exists() for j in js)]
            if not ready:
                time.sleep(10)
                continue
            for n in ready:
                left.pop(n)
                if len(shots[n]) > 1:
                    _stitch_talk(book, R, L, talk, log, only={n}, quiet=True)
                _sync_offsets(book, L, [n], log, quiet=True)
                c = L / "out" / f"{n}.mp4"
                if not check or not c.exists():
                    continue
                fa = MF.fight_clip_ok(c, shots[n][0].get("people"))
                (L / "out" / f"{n}.judged").write_text(json.dumps({"take1": fa}))
                if fa:                  # lean (10-28 user): counted as a defect to fix upstream, not re-rendered
                    _problem(book, log, f"{n}: talk clip defect {fa}")
    except Exception as e:                  # noqa: BLE001  (the render goes on; the pass after it catches up)
        log(f"lip sync as clips land stopped: {e!r}")


def _stitch_talk(book: Path, R, L: Path, talk: list, log=print, only: set | None = None, quiet: bool = False) -> None:
    """Join each pieced talk shot (10-28: the gate read the moment's OLD pieces - the new ones were on a copy - so no
    line was stitched all night, and yesterday's joins of other words were assembled). A join older than any of its
    pieces is dropped and made again; one made from tonight's pieces stays (lines are joined as their pieces land,
    _sync_as_they_land); a line that didn't come out is a problem, not a silent gap. `only`: these shots."""
    pieces = {}
    for j in talk:
        if len(j.get("chunk") or []) == 4 and j["chunk"][1] > 1:
            pieces.setdefault(j["name"].rsplit("_c", 1)[0], []).append(j)
    names = sorted(n for n in pieces if only is None or n in only)
    todo = []
    for n in names:
        c = L / "out" / f"{n}.mp4"
        pm = [L / "out" / f"{j['name']}.mp4" for j in pieces[n]]
        newest = max((p_.stat().st_mtime for p_ in pm if p_.exists()), default=float("inf"))
        if c.exists() and c.stat().st_mtime >= newest and all(p_.exists() for p_ in pm):
            continue
        c.unlink(missing_ok=True)
        todo.append(n)
    if not todo:
        return
    lines = ",".join(sorted({pieces[n][0]["line"] for n in todo}))
    r = subprocess.run([str(R.LTX_PY), str(HERE / "ltx_runner" / "stitch_talk.py"), "talk_jobs.json", "out", lines],
                       cwd=str(L), capture_output=True, text=True)
    names = todo
    bad = [n for n in names if not (L / "out" / f"{n}.mp4").exists()]
    if r.returncode or bad:
        _problem(book, log, f"talk stitch: {len(bad)} of {len(names)} lines not joined ({', '.join(bad[:4])})"
                 f"{': ' + r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ''}")
    elif not quiet:
        log(f"talk stitch: {len(names)} lines joined")


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
                # a talk clip: its silent lead-in dropped (talk_pre), and SyncNet's shift (10-28 user: "version 3
                # is definitely an improvement") trims a little more (lips late) or less (lips early) of it
                st_ = talk_pre(c) + sync_shift(c) if s.get("kind") == "talk" else 0
                if st_ > 0:
                    src = ["-ss", f"{st_ / FPS:.4f}", "-i", str(c)]
                elif st_ < 0:           # (a clip from before the lead-in, lips early: its first frame held)
                    vf = f"tpad=start_mode=clone:start_duration={-st_ / FPS:.4f}," + vf
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


def _step(book: Path, text: str) -> None:
    """The job's status line follows the flow's steps (10-27: it said "character LoRAs" through the whole render)."""
    try:
        sys.path.insert(0, str(HERE.parent))
        from tts_reader import bookcaster as BC
        BC._set_status(book, step=f"shots: {text}")
    except Exception:                                   # noqa: BLE001 - a status line never stops the job
        pass


def run(book: Path, stem: str, t0: float, t1: float, hero: Path | None, mix: Path, log=print,
        until: str | None = None, guide: str | None = None, render: tuple | None = None) -> Path:
    """The whole stretch: shots (director) -> keys (both cards) -> pick (+3 seeds where none passed) -> [until=keys:
    the review sheet] -> LTX (pipelined) -> takes judged -> the takes sheet -> assembled."""
    es = MF.entries(book, stem, t0, t1)
    # --render a:b: the director plans the whole stretch (cached), only this window is drawn, rendered, assembled
    # (10-27 user: "shorten the clip to only a minute for now to increase the learning cycles")
    op = out_path(book, stem, *(render or (t0, t1)))
    # 10-24 user: "you should have gone right back to rendering" - a render that was stopped resumes on the plan it
    # had (shots + keys), nothing planned, drawn or checked again; the plan is dropped once the cut is assembled
    fz = shot_dir(book, stem) / f"plan_{op.stem}.json"
    sig = _plan_sig(book, stem)
    if until != "keys" and fz.exists() and json.loads(fz.read_text()).get("_sig") != sig:   # (no sig: older code)
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
    _step(book, "wardrobe and the director")
    wardrobe(book, stem, log)
    es, by, _ = direct(book, stem, es, log, keep_llm=True)    # the director's shots are the moments from here on
    sheets = place_sheets(book, stem, es, by, log)            # then one set sheet per place it used
    from tts_reader import animate
    subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    shots = [s for e in es for s in by.get(e["id"], [])]
    R_ = RG.Book(book)
    for s in shots:                     # each place's ONE room: the book's description, else its set sheet's (10-27
        sh = sheets.get(s["id"].rsplit("_s", 1)[0]) or {}        # audit: keys used the set sheet, the hero the book's)
        s.update(room=R_.place_look(s.get("place")) or sh.get("room", ""), fixtures=sh.get("fixtures") or [],
                 scene=sh.get("scene", ""),
                 crowd=(sh.get("crowd") or "") if s.get("framing") != "close" else "")   # the set sheet's crowd only
    describe_places(book, stem, es, shots, log)     # an undescribed place gets its set sheet's room in the book
    if render:
        keep = {e["id"] for e in es if render[0] - 1e-6 <= float(e["t"]) < render[1]}
        es = [e for e in es if e["id"] in keep]
        by = {k: v for k, v in by.items() if k in keep}
        shots = [s for s in shots if s["id"].rsplit("_s", 1)[0] in keep]
        log(f"rendering {render[0]:.1f}-{render[1]:.1f} s only: {len(shots)} shots of the directed stretch")
        if not shots:
            raise RuntimeError(f"the render window {render[0]:g}-{render[1]:g} s has no shots of the directed stretch")
    _step(book, f"keyframes ({len(shots)} shots)")
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
    """What a saved plan's shots and keys were made from: this chapter's cast's looks and LoRA builds (10-25: a resume
    after `look --swap` would have rendered the old keys), the timeline and the mix (10-27 audit: a stopped render
    resumed after a pace change would have rebuilt the video on the old timings), and this code."""
    vd = book / "work" / "video"
    R_ = RG.Book(book)
    pp = vd / stem / "plan.json"
    cast = [R_.person(n) or n for n in ((json.loads(pp.read_text()).get("look") or {}).get("characters", {})
                                        if pp.exists() else {})]
    def stamp(f):
        return f"{f.stat().st_size}:{f.stat().st_mtime_ns}" if f.exists() else "-"
    return MF._h(json.dumps([{w: R_.look(w) for w in cast}, [R_.lora(w) for w in cast],
                             stamp(book / "work" / f"{stem}.timeline.json"), stamp(book / "work" / "final" / f"{stem}.mp3"),
                             (book / "pace.txt").read_text().strip() if (book / "pace.txt").exists() else "",
                             _plan_code()], sort_keys=True))


PLAN_ROOTS = ("run", "wardrobe", "direct", "place_sheets", "describe_places", "keys_judged", "sheet")
RENDER_SIDE = {"_render"}       # the render owns its own staleness (key, prompt, seed, frames, voice cut per clip)


def _plan_code() -> str:
    """The code a saved plan came from: what the planning steps reach in this file (functions and the prompts,
    rules and numbers they read), never the render's (10-28: a fix to the talk stitch changed the file's hash,
    the night's plan was dropped, the director asked again and 3 hours of clips were rendered a second time)."""
    import inspect
    g = globals()
    out = [inspect.getsource(g[n]) if callable(g[n]) else f"{n}={g[n]!r}" for n in _plan_reach()]
    return MF._h("\n".join(sorted(out)))


def _plan_reach() -> set:
    """The names of this file the planning steps reach (functions followed through their code, constants kept)."""
    import types as _t
    g, seen, todo, keep = globals(), set(), list(PLAN_ROOTS), set()
    while todo:
        n = todo.pop()
        if n in seen or n in RENDER_SIDE or n not in g:
            continue
        seen.add(n)
        v = g[n]
        if isinstance(v, _t.FunctionType) and v.__module__ == __name__:
            keep.add(n)
            codes = [v.__code__]
            while codes:
                c = codes.pop()
                todo += list(c.co_names)
                codes += [k for k in c.co_consts if isinstance(k, _t.CodeType)]
        elif isinstance(v, (str, int, float, tuple, list, dict, set, frozenset)):
            keep.add(n)
    return keep

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
    _step(book, f"LTX video ({len(shots)} shots)")
    ltx(book, stem, es, shots, log)
    again = judge_takes(book, stem, shots, log)
    if again:
        ltx(book, stem, es, [s for s in shots if s["id"] in again], log, retake=again)
        judge_takes(book, stem, shots, log)
    # 10-24: judge_talk no longer re-renders - it failed all 7 forge lines and the user judged every one good
    log(f"takes sheet: {sheet(book, stem, shots, op.with_name(op.stem + '_takes.jpg'), takes=True)}")
    _step(book, "assembling the cut")
    out = assemble(book, stem, es, by, mix, op, log)
    fz.unlink(missing_ok=True)
    return out
