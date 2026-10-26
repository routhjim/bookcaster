"""Motions for MANY shots in one resident session (humans: HY-Motion, one encoder pass + one DiT load + one GPU hold
for the whole list; creatures: the procedural engine, CPU), written per shot in blockout_video's format.

    ~/tts-ab/.venv-pose/bin/python -I tools/motion_batch.py LIST.json OUT_DIR [--fps 24] [--checks] [--seed 1]

LIST.json: [{"shot": "/abs/shot.json" | null, "frame": "/abs/keyframes.json#<shot id>" | {frame}, "dur": seconds,
             "id": "<shot id>", "motion": "planner motion text (optional; else the frame's)"}]
  * shot: a blockout shot JSON (tools/blockout_plan.py output). Without one, the frame is planned with
    blockout_plan.frame_to_shot; humans the planner could not place (no 2D pose boxes) are staged by default
    rules (DEFAULT STAGING, flagged in the report).
  * dur: the clip length; frames = 1 + 8k >= dur * fps (LTX).
Per figure the planner's motion text is split into sentences and given to the figure it names (pronouns follow
the last named subject). Humans: ACTION MAP (story verb -> a body-only HY-Motion prompt with the hand stated,
an impact mode for motion_shot, a hold for poses the model will not hold). Creatures: CREATURE MAP (-> verbs of
tools/motion_creature.py; a strike aims at the human's hand at the shared impact frame, a recoil starts on it).
OUT_DIR/<id>/: shot.json, <who>.npz/.json (CanonicalMotion), joints_<who>.npz (blockout dict: names, pos (N,K,3)),
motion_shot.json; OUT_DIR/report.json: per-shot prompts, modes, timings, warnings, failures (and --checks: the
renderer's cleanup checks per shot, no Blender).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

MODELS = os.environ.get("BLOCKOUT_MODELS", os.path.expanduser("~/tts_reader/eval/blockout/models"))

# ---------------------------------------------------------------------------------------------------
# story verb -> HY-Motion prompt. Rules learned on the test bed: body-only verb phrases, the hand stated,
# no objects/scene (the model ignores them; describe what the BODY does with them), one action.
# (pattern, prompt without subject, impact mode for motion_shot, hold: None | "top" (freeze at the hands' highest
#  frame) | "end", from: "start" | "windup")
# ---------------------------------------------------------------------------------------------------
ACTION_MAP = [
    (r"\b(brings?|swings?) (the |his |her )?hammer down|\bstrikes?\b.*\b(metal|anvil|ingot|blade|iron)\b|\bhammers?\b",
     # the approved forge recipe (user 10-20/21): the LEFT hand steadies the work on the anvil (the blade's hilt),
     # NO tongs (10-22: "holding tongs" was still in this text)
     "raises a hammer overhead with the right hand and strikes down hard, the left hand low in front steadying the "
     "work on the anvil",
     "surface", None, "windup"),
    (r"\bthrusts?\b.*\bsword\b|\bdriv(es|ing)\b.*\b(sword|blade)\b.*\binto\b|\bstabs?\b",
     "lunges forward with the right leg, thrusts a sword forward with both hands and holds the extended thrust",
     "reach", None, "start"),
    (r"\braises?\b.*\bsword\b.*\b(block|meet)|\bholds?\b.*\bblock\b|\bblocks?\b",
     "raises a sword horizontally above the head with both hands to block a blow from above",
     "none", "top", "start"),
    (r"\b(pump(s|ing)?)\b.*\bbellows\b|\bpush(es)?\b.*\bdown\b.*\bpull(s)?\b.*\bup\b",
     "pushes both hands down hard in front of the waist, elbows close to the body, and pulls them back up, again and again",
     "none", None, "start"),
    (r"\bruns?\b|\brunning\b", "runs forward", "none", None, "start"),
    (r"\bwalks?\b|\bwalking\b|\btakes? a step\b|\bsteps?\b.*\b(forward|firmly|onto)\b",
     "walks forward steadily", "none", None, "start"),
    (r"\brolls?\b.*\bground\b", "rolls sideways on the ground and comes up onto one knee", "none", None, "start"),
    (r"\bscrambles? up\b|\bgets? up\b|\b(rises?|gets?) to (his|her) feet\b",
     "gets up quickly from the ground to standing", "none", None, "start"),
    (r"\bgestures? (towards|toward|at)\b", "points forward with the right hand while talking", "none", None, "start"),
    (r"\b(speaks?|talks?|says)\b.*\bgestur|\bgesturing\b", "talks, gesturing with both hands", "none", None, "start"),
    (r"\badjusts?\b.*\bhelm", "adjusts a helmet on the head with both hands", "none", None, "start"),
    (r"\bmoves?\b.*\barms and legs\b|\btesting\b.*\bmobility\b",
     "swings the arms and lifts the knees one after the other, testing the movement", "none", None, "start"),
    (r"\bleans?\b.*\b(out|forward|closer)\b", "leans the upper body forward and stays there", "none", "end", "start"),
    (r"\bholds?\b.*\b(high|above|up)\b", "raises the right hand high and holds it steady", "none", "top", "start"),
    (r"\bholds?\b.*\bsteady\b|\bgrips?\b", "holds both hands steady in front of the chest", "none", "end", "start"),
    (r"\bbreathes? heavily\b|\bchest rising\b", "stands breathing heavily, chest rising and falling", "none", None, "start"),
    (r"\btrembl|\bshak(es|ing)\b|\bshiver", "stands trembling, shoulders shaking with fear", "none", None, "start"),
    (r"\bsits?\b|\bseated\b", "sits still, breathing slowly", "none", None, "start"),
    (r"\bfeeding\b|\breaches? out\b", "reaches out with the right hand and pulls it back, again and again",
     "none", None, "start"),
    (r"\bshouts?\b|\byells?\b|\bcalls? out\b", "shouts, leaning forward with both arms raised", "none", None, "start"),
    (r"\bbrac(es|ing)\b|\bstands? firm\b", "stands firm with feet apart, bracing the body and leaning forward",
     "none", "end", "start"),
    (r"\bfrozen\b|\bhover", "holds both hands forward over a table in front of the waist, frozen still", "none", "end", "start"),
    (r"\blaugh", "laughs, shoulders shaking, leaning back slightly", "none", None, "start"),
    (r"\blantern\b|\btorch\b|\blamp\b", "holds the left hand up in front of the body, steady", "none", "end", "start"),
    (r"\badjusts?\b", "makes a small adjustment with the left hand in front of the chest", "none", None, "start"),
    (r"\bexhales?\b|\bbreath", "stands breathing heavily, chest rising and falling", "none", None, "start"),
    (r"\bspeaks?\b|\bmouth moves\b|\btalks?\b", "talks, making small gestures with the right hand", "none", None, "start"),
    (r"\blooks?\b|\blisten|\bstands? still\b|\bremains?\b|\bwatch|\bobserv|\bstill\b|\bwaiting\b"
     r"|\bexpression\b|\bsmile|\beyes?\b|\bgaze\b",
     "stands still, breathing calmly and shifting weight slightly", "none", None, "start"),
]
IDLE = "A person stands still, breathing calmly and shifting weight slightly."

CREATURE_MAP = [   # pattern -> verb builder(dur, ctx) ; ctx: impact time, human hand target in creature frame
    (r"\bpreparing to strike\b", "brace"),
    (r"\bclaws?\b.*\bdown\b|(?<!to )\bstrikes?\b|\bswipes?\b|\bslashes?\b|\b(extends?|presses?|thrusts?|"
     r"lunges?)\b.*\b(fore)?claws?\b|\bsnaps?\b|\bbites?\b", "strike"),
    (r"\bscreams?\b|\broars?\b|\bsnarl|\brears? up\b|\bthrows? its head back\b", "roar"),
    (r"\bmouth opens\b.*\bfire\b|\bbreathes? fire\b|\bfire spills\b", "roar_long"),
    (r"\brecoils?\b|\bis hit\b|\bflinch", "recoil"),
    (r"\bshifts? (its )?weight\b|\bpreparing to strike\b|\bcrouch", "brace"),
    (r"\bcarr(y|ies|ying)\b|\bin its claws\b|\bclutch", "carry"),
    (r"\bfl(y|ies|ying)\b|\bglides?\b|\bsoars?\b|\bwheels? overhead\b", "fly"),
    (r"\b(lies|lying) (dead|still|motionless|unmoving)\b|\bunmoving\b|\blifeless\b|\bdead\b", "dead"),
    (r"\bfalls?\b|\bcollapses?\b|\bslumps?\b|\bcrashes? (down|to)\b", "fall"),
    (r"\bwings? (beat|flap)|\bbeats? its\b.*\bwings\b|\bthrash", "flap"),
    (r"\btail sweeps?\b|\bsweeps? its tail\b|\btail (lashes|whips|swings)\b", "tail"),
    (r"\bswings? its head\b", "roar"),
    (r"\bbreathing\b|\bremains?\b|\bstill\b|\bcoiled\b|\bblinks?\b|\beye\b", "idle"),
]

GENERIC_CREATURES = {"dragon", "the dragon", "beast", "creature", "wyrm", "monster"}
PRONOUNS = r"\b(he|she|his|her|him|it|its|they|their)\b"


def creature_anchors(model):
    """{anchor: joint} of a creature rig (rig.json "anchors"), {} when unknown."""
    if not model:
        return {}
    rj = os.path.join(os.path.dirname(model if isinstance(model, str) else model.get("blend")), "rig.json")
    if not os.path.exists(rj):
        return {}
    return {k: v.get("joint") for k, v in (json.load(open(rj)).get("anchors") or {}).items()}


def is_composite(model):
    """A composite creature rig: a humanoid body on the human joint names + extra chains (neck2.., tail, wings)."""
    if not model:
        return False
    rj = os.path.join(os.path.dirname(model if isinstance(model, str) else model.get("blend")), "rig.json")
    if not os.path.exists(rj):
        return False
    J = json.load(open(rj)).get("joints", {})
    return all(k in J for k in ("pelvis", "r_shoulder", "r_hand", "r_toe", "neck2"))


def load_tpl(name):
    from motion_templates import load_template
    return load_template(name)


def resolve_verb(v, ctx, seqs, Rz, at, BL, fps):
    """Choreography placeholders: times "impact" / "impact+0.5" (s); targets "human:<who>:<joint>[+dz]" (the human's
    joint at the verb's hit/start time, in the creature's template frame)."""
    v = dict(v)
    def tval(x):
        if isinstance(x, str) and x.startswith("impact"):
            return ctx["t_impact"] + (float(x[6:]) if len(x) > 6 else 0.0)
        return x
    for k in ("t", "t_hit"):
        if k in v:
            v[k] = tval(v[k])
    tg = v.get("target")
    if isinstance(tg, str) and tg.startswith("human:"):
        _, who, rest = tg.split(":", 2)
        joint, _, dz = rest.partition("+")
        fr = int(round(float(v.get("t_hit", v.get("t", 0.0))) * fps))
        sq = seqs.get(who)
        if sq is None:
            v["target"] = [1.6, 0.0, 1.0]
        else:
            J = sq[min(max(fr, 0), len(sq) - 1)]
            pt = (np.asarray(J["l_hand"], float) + np.asarray(J["r_hand"], float)) / 2 if joint == "hands" \
                else np.asarray(J[joint], float)
            pt = pt + np.array([0, 0, float(dz or 0.0)])
            v["target"] = ((Rz @ (pt - at)) / BL).tolist()
    return v


def frames_for(dur, fps):
    n = int(math.ceil(dur * fps))
    return 1 + 8 * int(math.ceil(max(n - 1, 8) / 8))


def sentences_for(text, who, all_who, creatures=()):
    """Sentences of the planner text that are about `who` (named, or a pronoun after it was the last subject).
    A shot's ONLY creature is also called by generic words ("the dragon", "the beast", ...)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]
    names = {w: [w.lower()] + [p.lower() for p in w.split() if len(p) > 3] for w in all_who}
    cr = [w for w in all_who if w in set(creatures)]
    if len(cr) == 1:
        names[cr[0]] += sorted(GENERIC_CREATURES - {"the dragon"})
    out, last = [], None
    for s in sents:
        low = s.lower()
        pos = {}
        for w in all_who:                                   # first mention of each figure in the sentence
            hits = [m.start() for n in names[w] for m in re.finditer(r"\b" + re.escape(n) + r"\b", low)]
            if hits:
                pos[w] = min(hits)
        named = sorted(pos, key=pos.get)
        # the SUBJECTS: the earliest name, plus names joined to it by "and" ("X and Y walk ...")
        subj = named[:1]
        for w in named[1:]:
            between = low[pos[subj[-1]]:pos[w]]
            if re.fullmatch(r"[\w' ]*\band\b\s*(the\s+)?", between.split(" ", 1)[-1] if " " in between else between) \
                    or re.search(r"^\S+(\s+\S+)?\s+and\s+(the\s+)?$", between):
                subj.append(w)
            else:
                break
        if subj:
            last = subj[0]
        if who in subj or (not named and last == who and re.search(PRONOUNS, low)):
            out.append(s)
    return out


RECIPES = os.path.join(HERE, "stage_templates", "motion_recipes.json")
PROPS = os.path.join(HERE, "stage_props", "props.json")


def book_props(index_path):
    """The book's own prop library (<book>/work/video/stage/props/props.json) beside its cast index."""
    if not index_path:
        return None
    p = os.path.join(os.path.dirname(os.path.dirname(index_path)), "props", "props.json")
    return p if os.path.exists(p) else None


def prop_affordances(name):
    """Affordance names of a prop in the book's library, then the shared one (by key or alias); [] when unknown."""
    if not name:
        return []
    lib = {}
    for path in [PROPS, os.environ.get("BLOCKOUT_PROPS_EXTRA")]:
        if path and os.path.exists(path):
            d_ = json.load(open(path))
            lib.update(d_.get("props", d_))
    n = name.lower()
    ent = lib.get(n) or next((v for k, v in lib.items() if isinstance(v, dict) and
                              n in [a.lower() for a in v.get("aliases", [])]), None)
    return sorted((ent or {}).get("affordances", {}).keys()) if isinstance(ent, dict) else []


def match_recipe(sents, holds, attached=False):
    """The first generic recipe whose verb matches the figure's own sentences AND whose affordance needs are met
    by the props it holds ({"right": prop, "left": prop}). -> recipe dict or None."""
    text = " ".join(sents).lower()
    aff = {h: set(prop_affordances(p)) for h, p in (holds or {}).items()}
    allaff = set().union(*aff.values()) if aff else set()
    for r in json.load(open(RECIPES))["recipes"]:
        if r.get("attached"):
            if attached:
                return r
            continue
        if not re.search(r["verb"], text):
            continue
        if "holds" in r and not all(a in aff.get(h, set()) for h, a in r["holds"].items()):
            continue
        if "holds_any" in r and r["holds_any"] not in allaff:
            continue
        return r
    return None


def human_prompt(sents, who):
    """-> (prompt, mode, hold, from_, matched pattern or None)."""
    text = " ".join(sents)
    low = text.lower()
    for pat, prompt, mode, hold, from_ in ACTION_MAP:
        if re.search(pat, low):
            return "A person " + prompt + ".", mode, hold, from_, pat
    return IDLE, "none", None, "start", None              # unmatched (reported): a calm idle, never raw prose


def beats_of(frame, who):
    """The director's beats for `who` ("Edric: rolls away ..." -> "Edric rolls away ..."): the moment's MECHANICS,
    which the motion follows when given (10-23: the fight's motion came from the planner's one-line text by keywords
    - the tail sweep, the block and the roll were never staged, both figures stood still)."""
    out = []
    for b in frame.get("beats") or []:
        b = str(b).strip()
        if b.lower().startswith(who.lower() + ":"):
            out.append(f"{who} {b.split(':', 1)[1].strip()}")
    return out


def beats_prompt(sents, who):
    """HY-Motion words from a figure's beats when no tuned pattern matched: "A person <the beats>" (body words only)."""
    acts = [re.sub(r"^" + re.escape(who) + r"\s+", "", x).rstrip(".") for x in sents]
    txt = ", then ".join(a for a in acts if a)
    txt = re.sub(r"\b(his|her|their)\b", "the", txt)
    return ("A person " + txt[:220].rsplit(" ", 1)[0] + ".") if txt else None


def creature_verbs(sents, dur, ctx):
    low = " ".join(sents).lower()
    verbs, warn, kinds = [], [], []
    for pat, kind in CREATURE_MAP:
        if re.search(pat, low) and kind not in kinds:
            kinds.append(kind)
            if kind == "brace" and re.search(r"\bpreparing to strike\b", low):
                low = re.sub(r"\bpreparing to strike\b", "", low)
    if not kinds:
        kinds = ["idle"]
    t_imp = ctx["t_impact"]
    for k in kinds:
        if k == "strike":
            if ctx.get("target") is None:
                warn.append("strike without a human target: aimed 1.6 body lengths ahead")
            tgt = ctx.get("target") if ctx.get("target") is not None else np.array([1.6, -0.15, 0.95])
            verbs += [{"verb": "brace", "t": 0.0, "dur": 0.5, "until": max(0.2, t_imp - 1.4)},
                      {"verb": "strike", "limb": "fleg_r", "target": tgt.tolist(), "t": max(0.1, t_imp - 1.4),
                       "t_hit": t_imp, "hold": min(1.0, max(0.2, dur - t_imp - 0.6))}]
        elif k == "roar":
            verbs.append({"verb": "roar", "t": min(ctx.get("t_after", 0.3), dur - 1.0), "dur": max(0.8, min(1.6, dur - 0.5))})
        elif k == "roar_long":
            verbs.append({"verb": "roar", "t": 0.2, "dur": max(1.0, dur - 0.6)})
        elif k == "recoil":
            verbs.append({"verb": "recoil", "t": t_imp, "from": [-1, 0, 0], "strength": 1.0})   # a blow from the front
        elif k == "brace":
            verbs.append({"verb": "brace", "t": 0.2, "dur": 0.8})
        elif k == "idle":
            verbs.append({"verb": "idle", "t": 0.0})
        elif k in ("carry", "fly"):
            if "carry" in kinds and k == "fly":
                continue                                    # carry already flies
            span = 1.2 * dur                                # across the frame, left to right, at the staged height
            verbs.append({"verb": k, "t": 0.0, "dur": dur, "from": [-span, 0, 3.0], "to": [span, 0, 3.2],
                          "glide": bool(re.search(r"\bglid|\bsoar|\bsteadily\b", low))})
        elif k == "dead":                                   # slain earlier: lies still the whole shot (10-23)
            verbs.append({"verb": "dead", "t": 0.0, "dur": dur})
        elif k == "fall":
            verbs.append({"verb": "fall", "t": 0.3, "dur": max(1.2, min(2.6, dur - 0.6))})
        elif k == "flap":
            verbs.append({"verb": "flap", "t": 0.1, "dur": max(1.0, dur - 0.2), "amp": 45})
        elif k == "tail":
            verbs.append({"verb": "tail", "t": 0.3 * dur, "dur": min(1.6, 0.5 * dur), "amp": 80})
        else:
            warn.append(f"creature action '{k}' has no verb yet (idle used)")
            verbs.append({"verb": "idle", "t": 0.0})
    return verbs, kinds, warn


# ---------------------------------------------------------------------------------------------------
# staging fallbacks
# ---------------------------------------------------------------------------------------------------

CAST = {"index": None}
MODEL_OVERRIDE = {}


def find_cast(path):
    """The book's cast index (<book>/work/video/stage/characters/index.json), found upward from a shot/frame path."""
    d = os.path.dirname(os.path.abspath(path))
    for _ in range(8):
        for cand in (os.path.join(d, "stage", "characters", "index.json"),
                     os.path.join(d, "video", "stage", "characters", "index.json")):
            if os.path.exists(cand):
                return cand
        d = os.path.dirname(d)
    return None


def cast_kind(who):
    """"creature" | "human" | None from the book's registry: the cast index's character "kind", else
    stage/assets.json "creatures" (names or entries with "name")."""
    idx = CAST["index"]
    if idx:
        chars = json.load(open(idx)).get("characters", {})
        ent = next((v for k, v in chars.items() if k.lower() == who.lower()), None)
        if ent and ent.get("kind"):
            return "creature" if ent["kind"] == "creature" else "human"
        assets = os.path.join(os.path.dirname(os.path.dirname(idx)), "assets.json")
        if os.path.exists(assets):
            cr = json.load(open(assets)).get("creatures") or []
            names = {(c.get("name") if isinstance(c, dict) else c).lower() for c in cr}
            if who.lower() in names:
                return "creature"
    return None


def rig_for(who, outfit=None):
    """The figure's OWN rigged model from the cast index (the outfit's "model", else its dir/rigged.blend);
    None when the cast has none (never another character's body)."""
    idx = CAST["index"]
    if idx:
        chars = json.load(open(idx))["characters"]
        ent = next((v for k, v in chars.items() if k.lower() == who.lower()), None)
        if ent:
            outs = ent.get("outfits", {})
            o = outs.get(outfit or "", None) or next((v for k, v in outs.items() if outfit and v.get("id") == outfit), None) \
                or outs.get("")
            if o:
                m = o.get("model") or os.path.join(o["dir"], "rigged.blend")
                if os.path.exists(m):
                    return m
    return None


def default_figure(who, i, n):
    # NB "standing" is ALSO a dragon pose in blockout_rig.DRAGON_POSES (-> staged as a creature): use a human one
    f = {"who": who, "kind": "human", "pose": "standing_front", "at": [round((i - (n - 1) / 2) * 1.2, 2), 4.0],
         "height": 1.8, "facing": -60.0 if i % 2 == 0 else -120.0}
    m = rig_for(who)
    if m:
        f["model"] = m
    return f


POSE_FOR = [   # mapped action -> a library start pose for staging (blockout_rig.SPECS; never "standing")
    (r"hammer", "hammer_raised"), (r"thrusts a sword", "sword_ready"), (r"block", "sword_block"),
    (r"walks", "walking"), (r"runs", "running"), (r"\bsits\b", "sitting_chair"), (r"rolls", "crouching"),
    (r"points|gesturing|talks", "pointing"), (r"shouts", "shouting"),
]


def start_pose(prompt):
    for pat, pose in POSE_FOR:
        if re.search(pat, prompt or ""):
            return pose
    return "standing_front"


def pose_boxes_from_image(png, humans, prompts, thr=0.3):
    """People in the drawn keyframe picture (DWPose, rtmlib, CPU) -> planner "poses" entries: normalised
    box from the keypoints (padded), facing from the nose vs the neck, names given LEFT TO RIGHT in the frame's
    "who" order (a heuristic, reported), a library start pose from each figure's mapped action."""
    import cv2
    from rtmlib import Body
    img = cv2.imread(png)
    if img is None:
        return []
    H_, W_ = img.shape[:2]
    kps, scores = Body(mode="balanced", to_openpose=True, backend="onnxruntime", device="cpu")(img)
    found = []
    for k, sc in zip(kps, scores):
        vis = k[sc > thr]
        if len(vis) < 6:
            continue
        x0, y0 = vis.min(0)
        x1, y1 = vis.max(0)
        h = y1 - y0
        x0, x1 = x0 - 0.12 * h, x1 + 0.12 * h
        y0, y1 = y0 - 0.12 * h, y1 + 0.04 * h
        head = [k[i] for i in (0, 14, 15) if sc[i] > thr]
        facing = "right" if not head or sc[1] <= thr or np.mean([p[0] for p in head]) >= k[1][0] else "left"
        found.append({"box": [max(0.0, x0 / W_), max(0.0, y0 / H_), min(1.0, x1 / W_), min(1.0, y1 / H_)],
                      "facing": facing, "area": float((x1 - x0) * (y1 - y0))})
    found = sorted(found, key=lambda d: -d["area"])[:len(humans)]
    found.sort(key=lambda d: d["box"][0])
    return [{"who": w, "pose": start_pose(prompts.get(w, "")), "box": d["box"], "facing": d["facing"]}
            for w, d in zip(humans, found)]


def load_frame(spec):
    if isinstance(spec, dict):
        return spec
    path, _, sid = spec.partition("#")
    d = json.load(open(path))
    fr = d["frames"] if isinstance(d, dict) and "frames" in d else d
    return next(f for f in fr if f.get("shot") == sid)


def is_creature(who, shot):
    f = next((f for f in shot["figures"] if f.get("who") == who), None)
    return bool(f and f.get("kind") == "creature")


# ---------------------------------------------------------------------------------------------------

def run(items, out_dir, fps=24, seed=1, checks=False, fit=True):
    import blockout as B
    import motion_shot as MS
    from blockout_plan import frame_to_shot
    from motion_canonical import CanonicalMotion
    from motion_creature import animate
    from motion_source import hymotion_batch
    T0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    report = {"shots": {}, "timings": {}}
    plans = []
    # ---- 1. plan every shot, collect the human prompts
    for it in items:
        sid = it.get("id") or (it["frame"].split("#")[-1] if isinstance(it.get("frame"), str) else "shot")
        R = report["shots"].setdefault(sid, {"warnings": [], "figures": {}})
        try:
            CAST["index"] = find_cast(it.get("shot") or (it["frame"].split("#")[0] if isinstance(it.get("frame"), str) else "."))
            bp = book_props(CAST["index"])
            if bp and not os.environ.get("BLOCKOUT_PROPS_EXTRA"):      # the book's own props (the cleanup reads it)
                os.environ["BLOCKOUT_PROPS_EXTRA"] = bp
            frame = load_frame(it["frame"]) if it.get("frame") else {}
            text = it.get("motion") or frame.get("motion", "")
            creatures_ = {w for w in frame.get("who", []) if cast_kind(w) == "creature"} or None
            if it.get("shot"):
                shot = json.load(open(it["shot"]))
            else:
                if not frame.get("poses") and isinstance(it.get("frame"), str):
                    # the planner's 2D pose boxes are missing: recover them from the drawn keyframe picture
                    png = os.path.join(os.path.dirname(it["frame"].split("#")[0]), "keyframes", sid + ".png")
                    humans_ = [w for w in frame.get("who", []) if not creatures_ or w not in creatures_]
                    pr = {w: human_prompt(sentences_for(text, w, frame.get("who", [])), w)[0] for w in humans_}
                    if os.path.exists(png) and humans_:
                        frame = dict(frame, poses=pose_boxes_from_image(png, humans_, pr))
                        R["warnings"].append(f"pose boxes from the keyframe picture (DWPose): "
                                             f"{[(p['who'], p['pose'], p['facing']) for p in frame['poses']]}"
                                             + (" -- names given left to right" if len(frame["poses"]) > 1 else ""))
                shot = frame_to_shot(frame, creatures=creatures_)
            for af in it.get("add_figures", []):               # figures the planner omitted (choreography data)
                if af.get("kind") == "creature":               # a generic planner creature ("Dragon") IS this one:
                    gen = next((f for f in shot.setdefault("figures", []) if f.get("kind") == "creature"
                                and f.get("who", "").lower() in GENERIC_CREATURES), None)
                    if gen is not None:                         # take it over, keeping its on-screen placement
                        old_who = gen["who"]
                        keep = {k: gen[k] for k in ("at", "facing") if k in gen}
                        gen.clear()
                        gen.update({k: v for k, v in af.items() if k not in ("near", "dist", "at")}, **keep)
                        R["warnings"].append(f"planner creature '{old_who}' is {af['who']} (choreography)")
                        continue
                if not any(f.get("who") == af["who"] for f in shot.setdefault("figures", [])):
                    af = dict(af)
                    near = next((f for f in shot["figures"] if f.get("who") == af.get("near")), None)
                    if near is not None:                         # in front of that figure, along its facing
                        fa = near.get("facing", 0.0)
                        fa = float(fa) if not isinstance(fa, str) else 0.0
                        d_ = float(af.get("dist", 4.5))
                        af["at"] = [near["at"][0] + d_ * math.cos(math.radians(fa)),
                                    near["at"][1] + d_ * math.sin(math.radians(fa))]
                        af.setdefault("facing", near["who"])
                    af.pop("near", None)
                    af.pop("dist", None)
                    af.setdefault("at", [0.0, 6.0])
                    shot["figures"].append(af)
                    R["warnings"].append(f"added by the choreography: {af['who']} at {np.round(af['at'], 2).tolist()}")
            who_all = list(dict.fromkeys(list(frame.get("who") or []) + [f["who"] for f in shot.get("figures", [])]))
            placed = {f["who"] for f in shot.get("figures", [])}
            humans_missing = [w for w in who_all if w not in placed and cast_kind(w) != "creature"]
            for i, w in enumerate(humans_missing):
                shot.setdefault("figures", []).append(default_figure(w, i, len(humans_missing)))
                R["warnings"].append(f"DEFAULT STAGING for {w} (the planner had no pose box)")
            for f in shot["figures"]:                       # --model overrides (a rig under review, e.g.)
                if f.get("who") in MODEL_OVERRIDE:
                    f["model"] = MODEL_OVERRIDE[f["who"]]
                    f.pop("pose", None) if f.get("kind") == "creature" else None
            for f in shot["figures"]:                       # creatures: the rigged mesh when one exists, so the
                if f.get("kind") == "creature" and not f.get("model"):   # renderer AND the volume checks use the motion
                    rp = rig_for(f["who"])
                    if rp:
                        f["model"] = rp
                        f.pop("pose", None)
                        f.pop("reach", None)
                    else:
                        R["warnings"].append(f"no rig for creature {f['who']}: the static proxy is rendered")
            for f in shot["figures"]:
                if f.get("kind", "human") == "human" and not f.get("model"):
                    m_ = rig_for(f["who"], f.get("outfit"))
                    if m_:
                        f["model"] = m_
                    else:
                        raise RuntimeError(f"no rigged model for {f['who']} in the cast index {CAST['index']} "
                                           f"(no character borrows another's body)")
            N = frames_for(float(it["dur"]), fps)
            impact_at = int(round(0.55 * (N - 1)))
            d = os.path.join(out_dir, sid)
            os.makedirs(d, exist_ok=True)
            shot_path = os.path.join(d, "shot.json")
            figs = {}
            ch = it.get("choreo", {})
            cre_names = [g["who"] for g in shot["figures"] if g.get("kind") == "creature"]
            for f in shot["figures"]:
                w = f["who"]
                directed = beats_of(frame, w)
                sents = directed or sentences_for(text, w, who_all, cre_names)
                cw = ch.get(w, {})
                if f.get("kind") == "creature":
                    figs[w] = {"kind": "creature", "sents": sents, "verbs": cw.get("verbs"), "posture": cw.get("posture"),
                               "anchor": cw.get("anchor", "chest")}
                    if is_composite(f.get("model")):            # humanoid body from HY-Motion + procedural layers
                        from motion_composite import layers_for
                        v0 = cw.get("verbs") or creature_verbs(sents, float(it["dur"]), {"t_impact": 0.55 * float(it["dur"]), "target": None})[0]
                        prompt_, mode_, _, fly_ = layers_for(v0, float(it["dur"]), None)
                        figs[w].update({"composite": True, "prompt": "A person " + prompt_ + ".", "mode": mode_})
                else:
                    prompt, mode, hold, from_, pat = human_prompt(sents, w)
                    auto_att = None                              # staged as carried: an attach field or a hanging pose
                    if isinstance(f.get("attach"), dict) and f["attach"].get("to"):
                        auto_att = {"to": f["attach"]["to"]}
                    elif f.get("pose") in ("carried_limp", "hanging_from_claws"):
                        cr_ = next((g["who"] for g in shot["figures"] if g.get("kind") == "creature"), None)
                        auto_att = {"to": cr_} if cr_ else None
                    rec = match_recipe(sents, f.get("holds"), attached=bool(cw.get("attach") or auto_att))
                    gen_dur, attach = None, cw.get("attach") or auto_att
                    if rec:                                      # a generic recipe (action + affordances)
                        prompt = "A person " + rec["prompt"] + "."
                        mode, hold, from_, pat = rec.get("mode", "none"), rec.get("hold"), rec.get("from", "start"), "recipe"
                        gen_dur = rec.get("gen_dur")
                        if rec.get("attach") and attach:
                            attach = dict(rec["attach"], **attach)
                        R.setdefault("recipes", {})[w] = rec["name"]
                    if rec and rec.get("contact") and not any(c.get("who") == w and c.get("point") == rec["contact"]["point"]
                                                              for c in shot.get("contacts") or []):
                        cr = [g for g in shot["figures"] if g.get("kind") == "creature"]
                        prop_ = next((pp for hh, pp in (f.get("holds") or {}).items()
                                      if rec["contact"]["affordance"] in prop_affordances(pp)), None)
                        if cr and prop_:                         # the contact the recipe's action implies
                            me = np.asarray(f.get("at", [0, 0]), float)
                            tgt = min(cr, key=lambda g: np.linalg.norm(np.asarray(g.get("at", [0, 0]), float) - me))
                            cc = {k: v for k, v in rec["contact"].items() if k not in ("affordance", "to", "anchor")}
                            anc = rec["contact"]["anchor"]
                            anc = anc if isinstance(anc, list) else [anc]
                            have = creature_anchors(tgt.get("model"))
                            anc = next((a for a in anc if not have or a in have), anc[0])
                            cc.update({"who": w, "prop": prop_, "to": f"{tgt['who']}:{anc}"})
                            shot.setdefault("contacts", []).append(cc)
                            R["warnings"].append(f"{w}: recipe {rec['name']} adds contact {cc['prop']}.{cc['point']} -> {cc['to']}")
                    if cw.get("prompt"):                         # choreography data wins
                        prompt = cw["prompt"] if cw["prompt"].lower().startswith("a person") else "A person " + cw["prompt"]
                        prompt = prompt if prompt.endswith(".") else prompt + "."
                        mode, hold, from_, pat = cw.get("mode", "none"), cw.get("hold"), cw.get("from", "start"), "choreo"
                    if directed and beats_prompt(directed, w) and (pat is None or "stands still" in prompt):
                        # the beats themselves, never an idle ("right hand STILL gripping the sword" matched the idle)
                        prompt, pat = beats_prompt(directed, w), "beats"
                    figs[w] = {"kind": "human", "sents": sents, "prompt": prompt, "mode": mode, "hold": hold,
                               "from": from_, "matched": pat is not None, "attach": attach,
                               "gen_dur": cw.get("gen_dur") or gen_dur}
                    if pat is None:
                        R["warnings"].append(f"{w}: no verb pattern matched; idle used")
                R["figures"][w] = {k: v for k, v in figs[w].items() if k != "kind"}
            json.dump(shot, open(shot_path, "w"), indent=1)     # after the recipes' implied contacts
            plans.append({"sid": sid, "shot": shot, "shot_path": shot_path, "N": N, "impact_at": impact_at,
                          "figs": figs, "dir": d, "dur": float(it["dur"])})
            R.update({"frames": N, "impact_frame": impact_at})
        except Exception as e:
            R["failure"] = f"plan: {type(e).__name__}: {e}"
            R["trace"] = traceback.format_exc()[-1500:]
    T1 = time.time()
    # ---- 2. ALL human motions in one HY-Motion session (one encoder pass, one DiT load, one GPU hold)
    reqs, keys = [], []
    for p in plans:
        for w, f in p["figs"].items():
            if f["kind"] == "human" or f.get("composite"):
                # generate a bit longer than the shot when it is a strike (wind-up + recovery), else the shot
                gd = f.get("gen_dur") or (max(p["dur"], 4.0) if f.get("mode", "none") != "none" else max(p["dur"], 2.0))
                reqs.append((f["prompt"], min(12.0, float(gd))))      # gen_dur: reuse a proven clip length
                keys.append((p["sid"], w))
    cms = hymotion_batch(reqs, fps, seed=seed, raw_text=True) if reqs else []
    # T/A-pose leaks (tools/motion_validate.py): trim leading rest-pose frames; a clip that still leaks is
    # regenerated with other seeds (one more session for all of them) and the least-leaking seed is kept
    from motion_validate import tpose_leak
    def _leak(cm_, prompt_):
        seq_ = [{n: cm_.pos[i, k] for k, n in enumerate(JOINTS)} for i in range(cm_.frames)]
        return tpose_leak(seq_, prompt_, min_frames=1)
    from motion_canonical import JOINTS
    fixed, retry = [], []
    for i_, (cm_, (prompt_, d_)) in enumerate(zip(cms, reqs)):
        from motion_validate import arm_abduction
        lead = 0                                     # leading frames still easing out of a rest pose (looser band)
        while lead < cm_.frames // 3:
            Jl = {n: cm_.pos[lead, k] for k, n in enumerate(JOINTS)}
            if all(a > 50.0 and lat > 0.6 for a, lat in (arm_abduction(Jl, s_) for s_ in ("l", "r"))):
                lead += 1
            else:
                break
        if lead:                                     # the clip starts from a rest pose: cut it off
            cm_ = cm_.trim(lead / fps)
            cm_.meta["trimmed_rest_frames"] = lead
        rest = [f for f in _leak(cm_, prompt_)]
        if len(rest) >= 6:
            retry.append(i_)
        fixed.append(cm_)
    if retry:
        for sd in (seed + 1, seed + 2):
            alt = hymotion_batch([reqs[i_] for i_ in retry], fps, seed=sd, raw_text=True)
            for i_, cm2 in zip(retry, alt):
                n_old = len(_leak(fixed[i_], reqs[i_][0]))
                n_new = len(_leak(cm2, reqs[i_][0]))
                if n_new < n_old:
                    cm2.meta["reseeded_for_tpose_leak"] = sd
                    fixed[i_] = cm2
    cms = fixed
    T2 = time.time()
    hum = dict(zip(keys, cms))
    report["tpose"] = {f"{k[0]}:{k[1]}": {"trimmed": c.meta.get("trimmed_rest_frames", 0),
                                         "reseeded": c.meta.get("reseeded_for_tpose_leak"),
                                         "leak_frames_left": len(_leak(c, r[0]))}
                       for k, c, r in zip(keys, cms, reqs) if c.meta.get("trimmed_rest_frames") or
                       c.meta.get("reseeded_for_tpose_leak") or _leak(c, r[0])}
    # ---- 3. per shot: holds, retarget into the shot (motion_shot.build), creatures coupled to the humans
    for p in plans:
        sid, R = p["sid"], report["shots"][p["sid"]]
        t0 = time.time()
        try:
            specs, seqs = {}, {}
            for w, f in p["figs"].items():
                if f["kind"] != "human":
                    continue
                cm = hum[(sid, w)]
                if f["hold"] == "top":
                    import numpy as _np
                    from motion_canonical import JI
                    top = int(_np.argmax(cm.pos[:, JI["l_wrist"], 2] + cm.pos[:, JI["r_wrist"], 2]))
                    cm = cm.hold(top, until_s=max(cm.duration, p["N"] / fps))
                elif f["hold"] == "end":
                    cm = cm.hold(min(cm.frames - 1, int(0.7 * cm.frames)), until_s=max(cm.duration, p["N"] / fps))
                base = os.path.join(p["dir"], w.replace(" ", "_"))
                cm.save(base)
                specs[w] = base + ":" + f["mode"]
            if specs:
                for w, spec in specs.items():         # build per figure (its own from_ rule)
                    shot_, sq, rep = MS.build(p["shot_path"], {w: spec}, p["N"], fps, p["impact_at"],
                                              p["figs"][w]["from"])
                    seqs.update(sq)
                    R["figures"][w]["placement"] = rep[w]
                    # --fit: slide a figure with contacts along its facing to where the cleanup has least to fix
                    if fit and any(c.get("who") == w for c in p["shot"].get("contacts", [])):
                        import blockout_video as V
                        st = B.stage_frames(p["shot"], [("start", copy.deepcopy(p["shot"]["figures"])),
                                                        ("end", B.merge_end(p["shot"]) or copy.deepcopy(p["shot"]["figures"]))])
                        fwd = np.asarray(st[0][4][w]["ex"]["fwd"], float)
                        imp = p["impact_at"] if p["figs"][w].get("mode") != "none" else p["N"] - 1
                        off, nb = MS.fit_offset(p["shot"], {w: seqs[w]}, w, imp, p["N"], V, fwd, span=(-0.3, 0.2),
                                                verbose=False)
                        if nb > 0 and off >= 0.2 - 1e-6:              # best at the forward limit: search further in,
                            off2, nb2 = MS.fit_offset(p["shot"], {w: seqs[w]}, w, imp, p["N"], V, fwd,
                                                      span=(0.25, 0.6), verbose=False)
                            if nb2 < nb:                                # and keep it only if it wins on ALL frames
                                full = [MS.fit_offset(p["shot"], {w: seqs[w]}, w, imp, p["N"], V, fwd, span=(o_, o_),
                                                      verbose=False, all_frames=True)[1] for o_ in (off, off2)]
                                if full[1] < full[0]:
                                    off, nb = off2, nb2
                                R["figures"][w]["fit_full_counts"] = {str(off_): n_ for off_, n_ in zip((round(off, 2), round(off2, 2)), full)}
                        for J in seqs[w]:
                            for q in J:
                                if not q.startswith("_"):
                                    J[q] = np.asarray(J[q], float) + fwd * off
                        R["figures"][w]["fit_offset_m"] = round(off, 3)
            # creatures, coupled: the strike lands on a human hand at the shared impact frame
            for w, f in p["figs"].items():
                if f["kind"] != "creature":
                    continue
                fig = next(x for x in p["shot"]["figures"] if x["who"] == w)
                BL = float(fig.get("height", 4.5)) / 1.75          # template head height ~1.75 body lengths
                at = np.array(fig.get("at", [0, 0]) + [0.0])[:3]
                fac = fig.get("facing", 0)
                if isinstance(fac, str):                            # facing a figure: toward its position
                    other = next((x for x in p["shot"]["figures"] if x["who"] == fac), None)
                    v = (np.array(other["at"]) - at[:2]) if other else np.array([1.0, 0.0])
                    fac = math.degrees(math.atan2(v[1], v[0]))
                ctx = {"t_impact": p["impact_at"] / fps, "target": None}
                Rz = np.array([[math.cos(-math.radians(fac)), -math.sin(-math.radians(fac)), 0],
                               [math.sin(-math.radians(fac)), math.cos(-math.radians(fac)), 0], [0, 0, 1]])
                posture = "quadruped"                               # an upright-built rig -> upright motions
                mb = fig.get("model")
                rjp = os.path.join(os.path.dirname(mb if isinstance(mb, str) else mb.get("blend")), "rig.json") if mb else None
                if rjp and os.path.exists(rjp):
                    rj = json.load(open(rjp))["joints"]
                    tr = np.asarray(rj["chest"], float) - np.asarray(rj["pelvis"], float)
                    if abs(tr[2]) > np.linalg.norm(tr[:2]):
                        posture = "upright"
                        BL = float(np.linalg.norm(tr)) * float(fig.get("scale", 1.0))
                if seqs:
                    hw = next(iter(seqs))
                    J_ = seqs[hw][p["impact_at"]]
                    # the blow lands on the BLADE held above the hands, not on the hands/arms
                    hand = (np.asarray(J_["r_hand"], float) + np.asarray(J_["l_hand"], float)) / 2 + np.array([0, 0, 0.25])
                    Rz = np.array([[math.cos(-math.radians(fac)), -math.sin(-math.radians(fac)), 0],
                                   [math.sin(-math.radians(fac)), math.cos(-math.radians(fac)), 0], [0, 0, 1]])
                    ctx["target"] = (Rz @ (hand - at)) / BL + np.array([0, 0, 0.05])
                    # a striking creature stands at striking distance: the target ~70 % of the fore leg's reach
                    # in front of its hip (the planner's spacing was made for the small proxy)
                    if any(re.search(pat, " ".join(f["sents"]).lower()) for pat, kd in CREATURE_MAP if kd == "strike"):
                        from motion_templates import load_template
                        TR = {k: np.asarray(v, float) for k, v in load_template("winged_quadruped")["rest"].items()}
                        idle = animate("winged_quadruped", [{"verb": "idle", "t": 0}], 0.4, fps, start={"posture": posture})
                        hip = idle.pos[-1, idle.joints.index("fleg_r_hip")]       # the stance's real fore hip
                        reach = sum(np.linalg.norm(TR[f"fleg_r_{b}"] - TR[f"fleg_r_{a}"])
                                    for a, b in (("hip", "knee"), ("knee", "ankle"), ("ankle", "toe")))
                        want_x = hip[0] + 0.7 * reach * math.cos(math.asin(min(0.95, max(-0.95, (ctx["target"][2] - hip[2]) / reach))))
                        dx = ctx["target"][0] - want_x                       # template units along the facing
                        fwd_w = np.array([math.cos(math.radians(fac)), math.sin(math.radians(fac)), 0.0])
                        at = at + fwd_w * dx * BL
                        ctx["target"] = (Rz @ (hand - at)) / BL + np.array([0, 0, 0.05])
                        R["figures"][w]["moved_to_strike_m"] = round(float(dx * BL), 2)
                struck = any(c.get("who") in seqs and str(c.get("to", "")).startswith(w + ":")
                             for c in p["shot"].get("contacts", []))
                if f.get("verbs"):                           # choreography data: resolve the placeholders
                    verbs = [resolve_verb(v, ctx, seqs, Rz, at, BL, fps) for v in f["verbs"]]
                    kinds, warn = ["choreo"], []
                    if f.get("posture"):
                        posture = f["posture"]
                else:
                    verbs, kinds, warn = creature_verbs(f["sents"], p["N"] / fps, ctx)
                if f.get("composite"):                       # humanoid composite: HY body + layers, no template engine
                    from motion_composite import composite_motion, layers_for
                    if struck and not any(v.get("verb") == "recoil" for v in verbs):
                        verbs = verbs + [{"verb": "recoil", "t": ctx["t_impact"], "dur": 1.0}]
                    _, _, Lyr, fly_ = layers_for(verbs, p["N"] / fps, None)
                    if fly_:                                   # the flight path centred on the staging, at its height
                        z0 = float(fig.get("z", 4.0)) + 2.9
                        span = 1.3 * p["N"] / fps
                        fw = np.array([math.cos(math.radians(fac)), math.sin(math.radians(fac)), 0.0])
                        Lyr["root"] = {"from": (at + np.array([0, 0, z0]) - fw * span).tolist(),
                                       "to": (at + np.array([0, 0, z0 + 0.3]) + fw * span).tolist()}
                    rig_j = os.path.join(os.path.dirname(fig["model"] if isinstance(fig["model"], str) else fig["model"]["blend"]), "rig.json")
                    body = hum[(sid, w)].resample(fps)
                    if any(v["verb"] == "strike" for v in verbs):   # the motion's own claw stop -> the shared impact
                        f_imp = MS.impact_of(body, "stop", "r")
                        body, _ = MS.retime(body, f_imp, p["impact_at"], p["N"])
                    cseq, val = composite_motion(rig_j, body, Lyr, fps, p["N"],
                                                 at=(float(at[0]), float(at[1])), facing_deg=float(fac))
                    from motion_composite import lean_to
                    imp = p["impact_at"]
                    if ctx.get("target") is not None and any(v["verb"] == "strike" for v in verbs):
                        # the claw meets the defender's blade: a tall body leans down to it, then centres over it
                        tw = np.asarray(J_["r_hand"], float) * 0.5 + np.asarray(J_["l_hand"], float) * 0.5 + np.array([0, 0, 0.25])
                        cseq, ang, res, crd = lean_to(cseq, "r_hand", tw, imp, fps)
                        val.update({"claw_to_target_m": round(res, 3), "lean_deg": round(ang, 1), "crouch_m": round(crd, 2)})
                    if struck:                                  # bring the struck anchor into the striker's reach
                        cont = next(c for c in p["shot"]["contacts"] if str(c.get("to", "")).startswith(w + ":"))
                        an_ = json.load(open(rig_j)).get("anchors", {}).get(str(cont["to"]).split(":", 1)[1], {})
                        if an_:
                            sz = float(np.asarray(seqs[cont["who"]][imp]["r_shoulder"], float)[2])
                            tz = np.array([0.0, 0.0, sz + 0.35])
                            cseq, ang, res, crd = lean_to(cseq, an_["joint"], tz, imp, fps, offset=an_.get("offset"),
                                                          xy=False, crouch_max=0.7)   # a giant goes down for the blow
                            val.update({"struck_anchor_height_residual_m": round(res, 3), "lean_deg": round(ang, 1),
                                        "crouch_m": round(crd, 2)})
                    from motion_composite import ground_pass
                    cseq, nfix, mz = ground_pass(cseq, json.load(open(rig_j)).get("parents", {}))
                    val.update({"ground_fixed_joint_frames": nfix, "min_joint_z": mz})
                    if mz < -0.05:
                        R["warnings"].append(f"{w}: joints below the floor ({mz} m) after the ground pass")
                    seqs[w] = cseq
                    names = sorted(cseq[0])
                    np.savez_compressed(os.path.join(p["dir"], f"joints_{w}.npz"), names=np.array(names),
                                        pos=np.array([[J2[n] for n in names] for J2 in cseq]))
                    R["figures"][w].update({"verbs": [v["verb"] for v in verbs], "composite": True,
                                            "body_prompt": f["prompt"], "validation": val})
                    continue
                if struck and not any(v.get("verb") == "recoil" for v in verbs):   # generic: recoil_when_struck
                    verbs = verbs + [{"verb": "recoil", "t": ctx["t_impact"], "from": [-1, 0, 0], "strength": 1.0}]
                    R["warnings"].append(f"{w}: recoil_when_struck (a human contact targets it)")
                R["warnings"] += [f"{w}: {x}" for x in warn]
                cm = animate("winged_quadruped", verbs, p["N"] / fps, fps, start={"posture": posture})
                R["figures"][w]["posture"] = posture
                base = os.path.join(p["dir"], w.replace(" ", "_"))
                cm.save(base)
                P, Rb = cm.place(BL, (float(at[0]), float(at[1])), float(fac))
                if fig.get("model"):                        # rigged: the rig's lengths + the moved placement
                    sh2 = copy.deepcopy(p["shot"])
                    for x in sh2["figures"]:
                        if x["who"] == w:
                            x["at"] = [float(at[0]), float(at[1])]
                    free = any(v["verb"] in ("fly", "carry") for v in verbs)
                    # place by the anchor a human contact targets (its staged geometry then holds), else the default
                    hit = next((str(c["to"]).split(":", 1)[1] for c in p["shot"].get("contacts", [])
                                if str(c.get("to", "")).startswith(w + ":") and c.get("who") in seqs), None)
                    cseq, crep = MS.place_creature(sh2, w, base, p["N"], p["impact_at"],
                                                   None if free else (hit or f.get("anchor", "chest")))
                    # a human's blow must be able to land: if the targeted anchor is above what the striker can
                    # reach at the impact (shoulder + 0.35 m), the creature comes down for it (lower_for), once
                    if hit and not free:
                        cont = next(c for c in p["shot"]["contacts"] if str(c.get("to", "")).startswith(w + ":"))
                        hj = cont["who"]
                        aj = load_tpl(cm.template)["anchors"][hit]["joint"]
                        az = float(np.asarray(cseq[p["impact_at"]][aj], float)[2])
                        sz = float(np.asarray(seqs[hj][p["impact_at"]]["r_shoulder"], float)[2])
                        need = az - (sz + 0.35)
                        if need > 0.05:
                            req, az2, base_verbs = need, az, list(verbs)
                            for _pass in range(3):               # secant: scale the request by needed/achieved
                                verbs = base_verbs + [{"verb": "lower_for", "t": p["impact_at"] / fps, "drop": req / BL}]
                                cm = animate("winged_quadruped", verbs, p["N"] / fps, fps, start={"posture": posture})
                                cm.save(base)
                                cseq, crep = MS.place_creature(sh2, w, base, p["N"], p["impact_at"], hit)
                                az2 = float(np.asarray(cseq[p["impact_at"]][aj], float)[2])
                                got = az - az2
                                if az2 - (sz + 0.35) < 0.08 or got < 0.02:
                                    break
                                req = min(req * need / got, 3.0 * need)
                            R["warnings"].append(f"{w}: lower_for {hj}'s blow: {hit} {az:.2f} -> {az2:.2f} m "
                                                 f"(striker shoulder {sz:.2f} m, target {sz + 0.35:.2f})")
                    P = np.array([[J[j] for j in cm.joints] for J in cseq])
                    seqs[w] = cseq
                    R["figures"][w]["placement"] = crep
                np.savez_compressed(os.path.join(p["dir"], f"joints_{w}.npz"), names=np.array(cm.joints), pos=P,
                                    bone_rot=Rb, body_length_m=BL)
                R["figures"][w].update({"verbs": [v["verb"] for v in verbs], "kinds": kinds, "body_length_m": round(BL, 2),
                                        "facing_deg": round(float(fac), 1)})
            for w, f in p["figs"].items():                  # carried figures hang from the creature's claws
                at_ = f.get("attach") if f["kind"] == "human" else None
                if at_ and at_.get("to") in seqs and w in seqs:
                    C_ = seqs[at_["to"]]
                    for i_, J in enumerate(seqs[w]):
                        cj = C_[min(i_, len(C_) - 1)]
                        an_ = creature_anchors(next((g.get("model") for g in p["shot"]["figures"] if g["who"] == at_["to"]), None))
                        cl_, cr_ = an_.get("claw_l", "fleg_l_toe"), an_.get("claw_r", "fleg_r_toe")
                        claw = (np.asarray(cj[cl_], float) + np.asarray(cj[cr_], float)) / 2   # the rig's own claws
                        by = at_.get("by", "hands")              # where the claws hold the carried figure
                        a_, b_ = {"hands": ("l_hand", "r_hand"), "shoulders": ("l_shoulder", "r_shoulder"),
                                  "waist": ("l_hip", "r_hip")}[by]
                        grip = (np.asarray(J[a_], float) + np.asarray(J[b_], float)) / 2
                        # held BELOW the claws (the claws close over the shoulders/waist, the body hangs clear)
                        dv = claw - grip + np.array([0, 0, -0.05 if by == "hands" else -float(at_.get("below", 0.12))])
                        for k_ in J:
                            if not k_.startswith("_"):
                                J[k_] = np.asarray(J[k_], float) + dv
                    R["figures"][w]["attached_to"] = at_["to"]
            sh_r = copy.deepcopy(p["shot"])                # the shot as rendered: creatures at their final spot,
            for w, f in p["figs"].items():                  # their real keyframe joints for the human staging
                if f["kind"] == "creature" and w in seqs:
                    fig_ = next(x for x in sh_r["figures"] if x["who"] == w)
                    fig_["at"] = [float(v) for v in np.asarray(seqs[w][0]["pelvis"], float)[:2]]
                    fig_["pose_joints"] = {k: np.asarray(v, float).tolist() for k, v in seqs[w][0].items()}
                    end = sh_r.setdefault("end", {}).setdefault("figures", [])
                    ef = next((x for x in end if x.get("who") == w), None)
                    if ef is None:
                        ef = {"who": w}
                        end.append(ef)
                    ef["pose_joints"] = {k: np.asarray(v, float).tolist()
                                         for k, v in seqs[w][p["impact_at"]].items()}
                    AN = {"head": "head", "jaw": "jaw", "chest": "chest", "pelvis": "pelvis",
                          "claw_l": "fleg_l_toe", "claw_r": "fleg_r_toe"}     # for the camera layer
                    if "fleg_r_toe" not in seqs[w][0]:          # a humanoid composite: the claws are its hands
                        AN.update({"claw_l": "l_hand", "claw_r": "r_hand"})
                    fig_["anchors_by_frame"] = {str(i_): {a: np.round(np.asarray(J[j], float), 3).tolist()
                                                          for a, j in AN.items() if j in J}
                                                for i_, J in enumerate(seqs[w])}
            json.dump(sh_r, open(os.path.join(p["dir"], "shot_render.json"), "w"), indent=1)
            models_ = {}
            for g in p["shot"]["figures"]:
                mb = g.get("model")
                mb = mb if isinstance(mb, str) else (mb or {}).get("blend")
                if mb and os.path.exists(mb):
                    import hashlib
                    models_[g["who"]] = {"model": mb, "sha1": hashlib.sha1(open(mb, "rb").read()).hexdigest()[:16]}
            R["models"] = models_
            json.dump({"shot": os.path.join(p["dir"], "shot_render.json"), "frames": p["N"], "fps": fps, "models": models_,
                       "impact": p["impact_at"], "figures": {w: f["kind"] for w, f in p["figs"].items()},
                       "impact_figures": [w for w, f in p["figs"].items() if f["kind"] == "human" and f.get("mode") != "none"]},
                      open(os.path.join(p["dir"], "render.json"), "w"), indent=1)
            for w, seq in seqs.items():
                if p["figs"].get(w, {}).get("kind") == "creature":
                    names = list(seq[0])                    # creature: world joints as placed (rig lengths)
                    np.savez_compressed(os.path.join(p["dir"], f"joints_{w}.npz"), names=np.array(names),
                                        pos=np.array([[J[n] for n in names] for J in seq]))
                    continue
                names = sorted(seq[0])
                np.savez_compressed(os.path.join(p["dir"], f"joints_{w}.npz"), names=np.array(names),
                                    pos=np.array([[J[n] for n in names] for J in seq]))
            if checks and seqs:
                import blockout_video as V
                V.IMPACTS.clear()
                V.IMPACTS.update({w: p["impact_at"] for w in seqs
                                  if p["figs"].get(w, {}).get("kind") == "human" and p["figs"][w].get("mode") != "none"})
                import contextlib, io
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = MS.dry_checks(p["shot_path"], p["dir"], seqs, p["N"], V, p["shot"])
                d = json.load(open(os.path.join(p["dir"], "checks_dry.json")))
                from collections import Counter
                R["checks"] = {"violations": d["violations"],
                               "by_check": dict(Counter(v["check"] for f in d["frames"] for v in f["violations"]))}
        except Exception as e:
            R["failure"] = f"build: {type(e).__name__}: {e}"
            R["trace"] = traceback.format_exc()[-1500:]
        R["build_s"] = round(time.time() - t0, 2)
    report["timings"] = {"plan_s": round(T1 - T0, 1), "hymotion_s": round(T2 - T1, 1), "n_human_motions": len(reqs),
                         "per_shot_build_s": {p["sid"]: report["shots"][p["sid"]].get("build_s") for p in plans},
                         "total_s": round(time.time() - T0, 1)}
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1, default=lambda o: np.asarray(o).tolist())
    return report


def render(shot_dir, out, extra):
    """Render one batch shot EXACTLY as the batch solved it: its joints (humans: blockout dicts; creatures: world
    joints on the rig's lengths), its shot_render.json (creatures at their final spot + pose_joints), its impact."""
    import blockout_video as V
    R = json.load(open(os.path.join(shot_dir, "render.json")))
    motion = {}
    for w in R["figures"]:
        z = np.load(os.path.join(shot_dir, f"joints_{w}.npz"))
        names = [str(n) for n in z["names"]]
        motion[w] = [{n: z["pos"][f, k] for k, n in enumerate(names)} for f in range(len(z["pos"]))]
    humans = R.get("impact_figures", [w for w, k in R["figures"].items() if k == "human"])
    shot_file = os.path.join(shot_dir, "shot_render.json")     # always the one beside it (dirs can be moved)
    return V.main([shot_file, out, "--frames", str(R["frames"]), "--fps", str(R["fps"])] + extra,
                  motion=motion, impacts={w: R["impact"] for w in humans})


if __name__ == "__main__":
    if "--render" in sys.argv:                     # motion_batch.py --render OUT/<id> RENDER_OUT [-- blockout_video args]
        i = sys.argv.index("--render")
        extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
        sys.exit(render(sys.argv[i + 1], sys.argv[i + 2], extra))
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--no-fit", action="store_true", help="skip the per-figure placement search (motion_shot --fit)")
    ap.add_argument("--model", action="append", default=[], help="WHO=/abs/rigged.blend: override a figure's model")
    a = ap.parse_args()
    MODEL_OVERRIDE.update(dict(m.split("=", 1) for m in a.model))
    rep = run(json.load(open(a.list)), a.out, a.fps, a.seed, a.checks, not a.no_fit)
    print(json.dumps(rep["timings"], indent=1))
    for sid, r in rep["shots"].items():
        print(sid, "FAIL " + r["failure"] if r.get("failure") else "ok", r.get("checks", ""), r["warnings"][:3])
