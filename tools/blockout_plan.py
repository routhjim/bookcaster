"""Turn one planner keyframe (anim/keyframes.json frame) into a blockout shot JSON for tools/blockout.py.

    ~/tts-ab/.venv-pose/bin/python -I tools/blockout_plan.py FRAME.json [-o shot.json] [--creatures Vorgrath]
                                                             [--llm] [--render OUT_DIR]
    FRAME.json: one frame object, or "keyframes.json#action-003117" to pick a shot from a file.

Python: `from blockout_plan import frame_to_shot`; frame_to_shot(frame, creatures={"Vorgrath"}) -> dict.

DETERMINISTIC LAYOUT (no model needed)
- The framing is read from the start of the prompt ("Wide shot", "Medium shot", "Close-up", "Low angle")
  and sets the lens and the camera height. The camera stands at the origin looking along +y.
- Each posed person: the 2D box gives the size on screen; the pose's real 3D height (blockout_rig) divided
  by that gives the depth (y); the box centre gives x at that depth. "facing": right/left becomes a 3/4
  turn toward the camera, so the 3D figure faces the same way on screen as the 2D plan.
- The first person's feet sit on the box's bottom edge: that fixes the camera height (a box bottom above
  the frame centre means a low camera looking up).
- A creature in "who" but without a pose gets the free side of the frame, a little behind the people, facing
  them; its pose comes from the prompt's verbs (flies/carries -> flying(_carry), claw/strike -> striking,
  rears -> rearing, lunge/charge -> lunging, else standing). A carried_limp / hanging_from_claws person is
  hung from its claws.
- Held items come from the prompt: hammer, sword/blade, tongs, shield; a helm is worn when the prompt has
  a helm on and the pose does not take it off.
- "end": the same mapping for the end poses; tools/blockout.py renders it from the same camera.
--llm asks Strata (http://127.0.0.1:18100, may be busy: 20 s timeout, then the deterministic answer) to
pick the creature pose and the props; nothing else depends on it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
import sys
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockout_rig as rig  # noqa: E402

W, H = 1536, 864
LLM_URL = os.environ.get("BLOCKOUT_LLM", "http://127.0.0.1:18100/v1/chat/completions")
CREATURE_WORDS = ("dragon", "wyrm", "beast", "creature")
FRAMINGS = [("extreme close", "close-up"), ("medium close", "medium close-up"), ("close-up", "close-up"), ("close up", "close-up"),
            ("low angle", "low angle"), ("high angle", "high angle"), ("wide", "wide"),
            ("establishing", "wide"), ("medium close", "medium close-up"), ("medium", "medium")]
LENS = {"wide": 28, "medium": 40, "medium close-up": 50, "close-up": 60, "low angle": 24, "high angle": 35}
_CAST = {}          # the book's character index (tools/characters.py): heights, kinds, models; set per call


def framing_of(prompt: str) -> str:
    p = (prompt or "").lower().strip()
    for key, name in FRAMINGS:
        if p.startswith(key) or p[:40].find(key) >= 0:
            return name
    return "medium"


def creature_pose(text: str, carrying: bool) -> str:
    t = (text or "").lower()
    if carrying or re.search(r"\b(carr|clutch|in its claws|holds? (her|him))", t):
        return "flying_carry" if re.search(r"fl(y|ies|ew|ight)|sky|moon|air", t) or carrying else "striking"
    if re.search(r"\bfl(y|ies|ying|ew)|soar|wings beat|in flight", t):
        return "flying"
    if re.search(r"claw|strike|swipe|descend|slash", t):
        return "striking"
    if re.search(r"\brear", t):
        return "rearing"
    if re.search(r"lunge|charg|snap|bite|lower(s|ed)? (its|his) head|neck", t):
        return "lunging"
    return "standing"


def holds_of(text: str, pose: str) -> tuple[dict, list]:
    t = (text or "").lower()
    holds, wear = {}, []
    if "hammer" in pose or re.search(r"\b(hold|grip|rais|swing|lift|strik|bring)\w*[^.]{0,40}\bhammer", t):
        holds["right"] = "hammer"
        if re.search(r"\btongs\b", t):              # ONLY when the text says tongs (never invented)
            holds["left"] = "tongs"
    elif pose == "tongs_over_anvil" and re.search(r"\btongs\b", t):
        holds["left"] = "tongs"
    elif pose.startswith("sword") or pose in ("kneel_with_sword", "quenching") or re.search(
            r"\b(hold|grip|carr|wield|rais|swing|lift|draw|brandish|clutch|thrust|point)\w*[^.]{0,40}\b(sword|blade)", t):
        holds["right"] = "sword"
    m_ = re.search(r"\b(hold|grip|carr|rais|lift|clutch|adjust)\w*[^.]{0,40}\b(lantern|torch|reins|cup|goblet)", t) \
        or re.search(r"\b(lantern|torch|reins|cup|goblet)\b[^.]{0,30}\b(in (his|her) hand|held)", t)
    if m_:                                   # small held items from the library (a free hand takes them)
        item = {"goblet": "cup"}.get(m_.group(2) if m_.lastindex >= 2 and m_.group(2) in
                                     ("lantern", "torch", "reins", "cup", "goblet") else m_.group(1), None)
        item = item or next(w for w in ("lantern", "torch", "reins", "cup", "goblet") if w in m_.group(0))
        item = {"goblet": "cup"}.get(item, item)
        holds["left" if "right" in holds else "right"] = item
    if re.search(r"\b(long|short)?bows?\b(?! (his|her|their) head)", t) and re.search(
            r"\b(hold|grip|carr|rais|lift|draw|nock|aim|string)\w*[^.]{0,40}\b(long|short)?bow\b|\barrow", t):
        holds["left"] = "bow"                 # the bow hand; a drawing right hand goes to the nock
    if pose == "shield_raised" or re.search(r"\bshield\b", t):
        holds["left"] = "shield"
    if pose == "helm_remove":
        holds = {"right": "helm"}
    elif pose == "helm_put_on" or re.search(r"helm(et)? on|wears? (a |his |the )?helm|visor", t):
        wear.append("helm")
    return holds, wear


# TEXT STAGING (no 2D pose boxes): verb -> pose rules, first match wins, on the sentences about a person
VERB_POSES = [
    (r"hammer[^.]{0,60}\b(rais|lift|hoist)|\b(rais|lift|hoist)\w*[^.]{0,40}hammer", "hammer_raised"),
    (r"hammer|strikes? the (metal|blade|steel|iron)|\bforg(e|es|ing) (the|a) ", "hammer_strike"),
    (r"\btongs\b", "tongs_over_anvil"),
    (r"quench|(sword|blade)[^.]{0,40}(water|river)|(water|river)[^.]{0,40}(sword|blade)", "quenching"),
    (r"thrust|\bstab(s|bed|bing)?\b|drives?[^.]{0,30}(sword|blade)|plunges?", "sword_thrust_lunge"),   # not "stable"
    (r"\bparr(y|ies)", "sword_parry_high"),
    (r"\bblock", "sword_block"),
    (r"(sword|blade)[^.]{0,40}(overhead|above (his|her) head|high)", "sword_raised_overhead"),
    (r"\bslash|swings?[^.]{0,30}(sword|blade)", "sword_slash"),
    (r"draws?[^.]{0,20}(sword|blade)", "sword_draw"),
    (r"(sword|blade)", "sword_ready"),
    (r"\bshield", "shield_raised"),
    (r"(takes?|pulls?|lifts?) off[^.]{0,20}helm|removes?[^.]{0,20}helm", "helm_remove"),
    (r"puts? on[^.]{0,20}helm|adjusts?[^.]{0,20}helm", "helm_put_on"),
    (r"helm[^.]{0,30}under (his|her) arm", "helm_under_arm"),
    (r"\bkneel", "kneeling"),
    (r"scrambl|gets? up|rises? from|push(es)? (himself|herself) up", "crouching"),
    (r"\bcrouch|\bhunch|duck", "crouching"),
    (r"\b(roll|dodge|dive)", "crouch_dodge"),
    (r"\bfall(s|ing)?\b|\bfell\b|knocked (back|down)|thrown", "falling_back"),
    (r"\blies|\blying|sprawl", "lying_back"),
    (r"\bsleep", "sleeping"),
    (r"\bsits?\b|seated|sitting", "sitting_chair"),
    (r"\bclimb", "climbing"),
    (r"\bcrawl", "crawling"),
    (r"\b(run|runs|running|sprint|charg|races?)\b", "running"),
    (r"\b(walk|step|stride|march|approach|leads?)\w*", "walking"),
    (r"\bpull(s|ing)?[^.]{0,20}(rope|chain|bellows)|bellows", "pulling_rope"),
    (r"\b(point|gestur|beckon|waves?)\w*", "pointing"),
    (r"\b(shout|yell|roar|calls? out|cries out|scream)\w*", "shouting"),
    # archery before bowing: "draws his bow" is a held prop (the bow arm extends: 'pointing' until an aim pose exists)
    (r"\b(draw|aim|rais|lift|nock|loos)\w*[^.]{0,30}\b((long|short)?bow|arrow)s?\b", "pointing"),
    (r"\bbow(s|ing)\b(?! (and|an?|the|his|her) arrow)|\bbow (low|deeply|before|to)\b", "bowing"),
    (r"\bpray", "praying"),
    (r"\bcheer", "cheering"),
    (r"embrac|\bhug", "embrace_a"),
    (r"reach(es)? (up|for)|holds? (up|aloft)|raises? (a|the|his|her) (lantern|torch|cup)", "reaching_up"),
    (r"shad(es|ing) (his|her) eyes", "shading_eyes"),
    (r"arms crossed|crosses (his|her) arms", "arms_crossed"),
    (r"\b(slump|collapse|exhaust)\w*", "slumped"),
    (r"\b(stands?|standing|waits?|watch|listen|look|stares?|faces?|holds?|brac|remains?|still|gaze|focus|"
     r"express|process|eyes|tests?|testing|moves? (his|her) (arms|legs)|adjusts?|breath|nods?)\w*", "standing"),
]
DEPTHS = {"close-up": 2.0, "medium close-up": 2.8, "medium": 4.2, "wide": 7.5, "low angle": 3.6, "high angle": 6.0}


def verb_pose(text: str) -> tuple[str, str | None]:
    """(pose, the rule that matched) from the sentences about one person; ("standing", None) if none."""
    t = (text or "").lower()
    for rx, pose in VERB_POSES:
        if re.search(rx, t):
            return pose, rx
    return "standing", None


def _about(text: str, who: str, everyone: list) -> str:
    """The sentences of `text` about `who`: those naming them, plus unnamed ones (he/she) that follow
    one naming them; with one person, everything."""
    if len([w for w in everyone if w]) <= 1 or not who:
        return text
    first = lambda n: (n or "").split()[-1].lower()
    out, cur = [], None
    for sent in re.split(r"(?<=[.!?])\s+", text or ""):
        low = sent.lower()
        named = sorted((low.find(first(w)), w) for w in everyone if w and first(w) in low)
        if named:
            # the sentence's subject: the first name, or "A and B" at its start
            cur = [named[0][1]]
            if len(named) > 1 and re.search(first(named[0][1]) + r"\s+and\s+" + first(named[1][1]), low):
                cur.append(named[1][1])
        if cur and who in cur:
            out.append(sent)
    return " ".join(out)


def _pose_height(pose: str, height: float) -> tuple[float, str]:
    """Vertical extent on screen of a pose (metres) and its anchor."""
    Wj, ex = rig.place_human({"pose": pose, "height": height, "at": [0, 0], "_facing": 0})
    zs = [v[2] for k, v in Wj.items() if not k.startswith("_")]
    return (max(zs) - min(zs)) + 0.12 * ex["scale"], ex["anchor"]


def _height_of(who: str) -> float:
    ent = (_CAST.get("characters") or {}).get(who or "")
    return float(ent["height"]) if ent and ent.get("height") else 1.8


def _text_poses(frame, who, is_creature, creatures, fr, notes):
    """No 2D boxes in the plan: stage from the TEXT. Each named person gets a pose from the verbs of the
    sentences about them (VERB_POSES); people stand side by side across the frame at the framing's depth,
    boxes synthesised so the box-based layout below applies unchanged. A creature named only in the prompt
    ("the dragon") joins under the book's creature name. Writes what it did into notes."""
    text = (frame.get("prompt", "") + " " + frame.get("motion", "")).strip()
    people = [w for w in who if w and not is_creature(w)]
    if not any(is_creature(w) for w in who) and re.search(r"\b(" + "|".join(CREATURE_WORDS) + r")\b", text.lower()):
        cname = next(iter(creatures), None) if creatures else None
        who = list(who) + [cname.title() if cname else "Dragon"]
    n = len(people)
    out = []
    hint = frame.get("pose_hint") or {}
    for i, w in enumerate(people):
        about = _about(text, w, people)
        pose, rx = verb_pose(about or text)
        if hint.get(w) in rig.SPECS:                   # the scene layer knows better (a carried captive)
            pose, rx = hint[w], "hint"
        notes["verbs"][w] = pose
        if rx is None:
            notes["unknown_verb"].append(w)
        # a box at the framing's apparent size: a person of ~1.8 m at DEPTHS[fr] across the frame
        hfrac = min(0.9, 1.75 / (2 * DEPTHS[fr] * math.tan(math.atan((rig.SENSOR_W * H / W) / 2 / LENS[fr]))))
        wfrac = hfrac * 0.45 * H / W
        cx = 0.5 + (i - (n - 1) / 2) * min(0.28, 0.8 / max(n, 1))
        y1 = 0.92 if fr != "close-up" else 1.1
        out.append({"who": w, "pose": pose, "box": [cx - wfrac / 2, y1 - hfrac, cx + wfrac / 2, y1],
                    "facing": "right" if (n > 1 and i == 0) or (n == 1 and i == 0) else "left"})
    return out, who


def set_kind(frame: dict, cast: dict | None = None) -> str:
    """interior / exterior of the frame's place: the book's set data first (stage/assets.json "kind", from the
    world's world.json), the place name's words only as a fallback."""
    where = frame.get("where") or ""
    vd = Path((cast or {}).get("book") or "")
    ap = vd / "stage" / "assets.json" if vd.name else None
    if ap and ap.exists() and where:
        try:
            ent = next((e for e in json.loads(ap.read_text()).get("sets", []) if e.get("where") == where), None)
        except (OSError, ValueError):
            ent = None
        if ent and ent.get("kind") in ("interior", "exterior"):
            return ent["kind"]
    return "interior" if re.search(r"\b(hall|room|inside|interior|chamber|smithy|forge|kitchen|cabin|cellar|"
                                   r"study|parlou?r|bedroom|tavern|inn|church|chapel|shop|house|office)\b",
                                   where.lower()) else "exterior"


def frame_to_shot(frame: dict, creatures: set | None = None, llm: bool = False, cast: dict | None = None,
                  chapter: str | None = None, t: float | None = None) -> dict:
    """cast: the book's character index (tools/characters.py plan/build; default: <book>/work/video/stage/
    characters/index.json when frame["book"] or BLOCKOUT_BOOK names the book). With it every figure gets ITS
    OWN model (the outfit in force at time t of the chapter) and height, and creatures are known by kind."""
    global _CAST
    if cast is None:
        bk = frame.get("book") or os.environ.get("BLOCKOUT_BOOK")
        ip = None
        if bk:
            import characters as CH
            ip = CH.vdir_of(bk) / "stage" / "characters" / "index.json"
        cast = json.loads(ip.read_text()) if ip and ip.exists() else {}
    _CAST = cast or {}
    creatures = {c.lower() for c in (creatures or set())} | {
        n.lower() for n, e in (_CAST.get("characters") or {}).items() if e.get("kind") == "creature"}
    prompt = frame.get("prompt", "")
    fr = framing_of(prompt)
    lens = LENS[fr]
    fpx = lens / rig.SENSOR_W * W
    who = list(frame.get("who") or [])
    poses = [p for p in (frame.get("poses") or []) if isinstance(p, dict) and p.get("pose") in rig.SPECS
             and isinstance(p.get("box"), list) and len(p["box"]) == 4]
    is_creature = lambda n: (n or "").lower() in creatures or any(w in (n or "").lower() for w in CREATURE_WORDS)
    figures, cam_h = [], None
    notes = {"staging": "boxes" if poses else "text", "verbs": {}, "unknown_verb": [], "props": []}
    if not poses:
        poses, who = _text_poses(frame, who, is_creature, creatures, fr, notes)
    carried = None
    for i, p in enumerate(poses):
        x0, y0, x1, y1 = p["box"]
        hgt = _height_of(p.get("who"))
        vh, anchor = _pose_height(p["pose"], hgt)
        lib = rig.SPECS[p["pose"]]
        # the 2D renderer fits the skeleton in the box keeping its aspect: the smaller scale wins
        from poses import load_library
        ent = load_library()[p["pose"]]
        aw, ah = ent["extent"]
        s = min((x1 - x0) * W / aw, (y1 - y0) * H / ah)
        pix_h = max(ah * s, 8.0)
        depth = fpx * vh / pix_h
        cx = (x0 + x1) / 2 * W
        x = (cx - W / 2) * depth / fpx
        f = {"who": p.get("who") or f"person{i}", "kind": "human", "pose": p["pose"], "at": [round(x, 3), round(depth, 3)],
             "height": hgt, "facing": -45.0 if p.get("facing", "right") != "left" else 225.0}
        holds, wear = holds_of(_about(prompt, p.get("who"), [q.get("who") for q in poses]), p["pose"])
        if holds:
            f["holds"] = holds
        if wear:
            f["wear"] = wear
        if p["pose"] in ("carried_limp", "hanging_from_claws"):
            carried = f
        bottom_y = (y0 + (y1 - y0) / 2 + ah * s / H / 2) * H if anchor == "center" else y1 * H
        if cam_h is None and anchor == "bottom":
            cam_h = (bottom_y - H / 2) * depth / fpx       # feet on the box's bottom edge
        figures.append(f)
    cam_h = 1.5 if cam_h is None else float(min(max(cam_h, 0.25), 6.0))
    pitch = 0.0
    if fr == "low angle" and figures:
        # a low camera tilted up so the first person's feet stay on their box's bottom edge
        f0 = next((f for f in figures if f["kind"] == "human"), None)
        b0 = next((p["box"] for p in poses if p.get("who") == f0["who"]), None) if f0 else None
        if b0:
            d0 = f0["at"][1]
            want = math.atan((b0[3] - 0.5) * H / fpx)
            cam_h = 0.4
            pitch = want - math.atan(cam_h / d0)
    # creatures
    people_x = [f["at"][0] for f in figures] or [0.0]
    people_d = [f["at"][1] for f in figures] or [6.0]
    names = [n for n in who if is_creature(n)]
    cpose = None
    if names and llm:
        cpose = _ask_llm(prompt, frame.get("motion", ""))
    for n in names:
        if any(f["who"] == n for f in figures):
            continue
        pose = cpose or creature_pose(prompt + " " + frame.get("motion", ""), carried is not None)
        d = float(np.mean(people_d)) + 2.5
        # the free side of the frame at that depth
        boxes = [p["box"] for p in poses]
        left_free = min([b[0] for b in boxes] or [0.5])
        right_free = 1 - max([b[2] for b in boxes] or [0.5])
        side = 1 if right_free >= left_free else -1
        half_w = d * (W / 2) / fpx
        x = float(np.mean(people_x)) + side * min(half_w * 0.55, 3.5)
        c = {"who": n, "kind": "creature", "pose": pose, "at": [round(x, 3), round(d, 3)], "height": 4.5}
        if figures:
            c["facing"] = figures[0]["who"]
            for f in figures:                              # people face the creature they confront
                if f is not carried:
                    f["facing"] = n if abs(f.get("facing", 0)) < 1e-6 else f["facing"]
        else:
            c["facing"] = 0
        if pose.startswith("flying"):
            c["z"] = 3.0 + d * 0.15
        if carried is not None:
            c["at"] = [carried["at"][0], carried["at"][1]]
            carried["attach"] = {"to": n, "point": "claws",
                                 "by": "hands" if carried["pose"] == "hanging_from_claws" else "pelvis"}
            if not pose.startswith("flying"):
                c["pose"] = "flying_carry"
                c["z"] = 3.0 + d * 0.15
        if pose == "striking" and figures:
            c["reach"] = {"limb": "fore_r", "to": figures[0]["who"], "point": "head", "above": 0.4}
            c["at"] = [figures[0]["at"][0] + side * 2.4, figures[0]["at"][1] + 0.5]
        figures.insert(0, c) if side < 0 else figures.append(c)
    look_d = float(np.mean(people_d)) if figures else 6.0
    pitch_target = cam_h + look_d * math.tan(pitch)
    shot = {"set": {"kind": set_kind(frame, cast),
                    "ground": True, "room": [12, 30, 5], "room_at": [0, look_d]},
            "figures": figures,
            "camera": {"framing": fr, "pos": [0.0, 0.0, cam_h], "look": [0.0, look_d, pitch_target], "lens": lens},
            "source": {"shot": frame.get("shot"), "prompt": prompt}, "plan_notes": notes}
    if frame.get("where"):
        shot["set"]["where"] = frame["where"]            # the HY-World set, when the book has one
    # set dressing named in the text, from the prop library (by alias): a bench/table in front of the first
    # person at hand height, small things (stone, candle, oilcloth) on it, bellows beside the forge side
    f0 = next((f for f in figures if f["kind"] == "human"), None)
    if f0 is not None:
        low = (prompt + " " + frame.get("motion", "")).lower()
        props_ = shot["set"].setdefault("props", [])
        sx = 0.75 if f0.get("facing") == -45.0 else -0.75
        bx, by = f0["at"][0] + sx * 0.8, f0["at"][1] + 0.35
        if re.search(r"\b(workbench|bench|table)\b", low):
            props_.append({"name": "workbench", "prop": "workbench", "box": [bx, by, 0, 1.2, 0.7, 0.9]})
            top = 0.9
            if re.search(r"\bstone\b", low):
                props_.append({"name": "stone", "prop": "stone", "box": [bx, by, top, 0.25, 0.2, 0.12]})
            if re.search(r"oilcloth|cloth", low):
                props_.append({"name": "oilcloth", "prop": "oilcloth", "box": [bx - 0.05, by, top, 0.5, 0.4, 0.01]})
            if re.search(r"\bcandle", low):
                props_.append({"name": "candle", "prop": "candle", "box": [bx + 0.45, by + 0.2, top, 0.04, 0.04, 0.2], "hot": True})
        if re.search(r"\bbellows\b", low):
            props_.append({"name": "bellows", "prop": "bellows", "box": [f0["at"][0] - sx * 0.7, f0["at"][1] + 0.4, 0.5, 0.8, 0.4, 0.3]})
    if re.search(r"anvil", prompt.lower()) and figures:
        f0 = next((f for f in figures if f["kind"] == "human"), None)
        if f0:
            dx = 0.75 if f0["facing"] == -45.0 else -0.75
            shot["set"].setdefault("props", []).append({"name": "anvil", "prop": "anvil",
                                                        "box": [f0["at"][0] + dx, f0["at"][1] - 0.4, 0, 0.7, 0.3, 0.8]})
    end = frame.get("end")
    if isinstance(end, dict) and end.get("poses"):
        ef = []
        for p in end["poses"]:
            if not isinstance(p, dict) or p.get("pose") not in rig.SPECS:
                continue
            e = {"who": p.get("who"), "pose": p["pose"]}
            holds, wear = holds_of(end.get("prompt", "") + " " + prompt, p["pose"])
            if holds:
                e["holds"] = holds
            e["wear"] = wear
            ef.append(e)
        shot["end"] = {"figures": ef, "prompt": end.get("prompt", "")}
    if _CAST:
        attach_models(shot, _CAST, chapter or frame.get("chapter"), t if t is not None else frame.get("t"))
    return shot


def attach_models(shot: dict, cast: dict, chapter: str | None, t: float | None):
    """Every figure wears ITS OWN model from the cast index (the outfit in force at t); a human without a
    model is listed in plan_notes["no_model"] (never silently a stand-in for someone else)."""
    import characters as CH
    miss = []
    for f in shot.get("figures", []):
        ent = (cast.get("characters") or {}).get(f.get("who"))
        if not ent:
            miss.append(f.get("who"))
            continue
        mo = CH.model_at(cast, f["who"], chapter, t) if (chapter and t is not None) else \
            dict(ent["outfits"].get("") or {}, kind=ent["kind"], height=ent.get("height"), outfit=None)
        if mo and mo.get("model"):
            f["model"] = mo["model"]
            f["outfit"] = mo.get("outfit")
            if mo.get("height") and f.get("kind") != "creature":
                f["height"] = mo["height"]
        else:
            miss.append(f.get("who"))
    shot.setdefault("plan_notes", {})["no_model"] = miss


def _ask_llm(prompt: str, motion: str) -> str | None:
    body = {"model": "default", "max_tokens": 20, "temperature": 0,
            "messages": [{"role": "user", "content":
                          f"Picture: {prompt}\nMotion: {motion}\nWhich pose is the creature in? Answer with one word from: "
                          + ", ".join(rig.CREATURE_POSES)}],
            "chat_template_kwargs": {"enable_thinking": False}}
    try:
        req = urllib.request.Request(LLM_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"].strip().lower()
        for p in rig.CREATURE_POSES:
            if p in txt.replace(" ", "_"):
                return p
    except Exception:
        return None
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("frame")
    ap.add_argument("-o", "--out")
    ap.add_argument("--creatures", default="", help="comma-separated names that are creatures")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--render", help="also render it with tools/blockout.py into this dir")
    a = ap.parse_args(argv)
    src = a.frame
    if "#" in src:
        path, shot_id = src.split("#", 1)
        frames = json.load(open(path))["frames"]
        frame = next(f for f in frames if f.get("shot") == shot_id)
    else:
        frame = json.load(open(src))
    shot = frame_to_shot(frame, {c.strip() for c in a.creatures.split(",") if c.strip()}, a.llm)
    txt = json.dumps(shot, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    else:
        print(txt)
    if a.render:
        import blockout
        print(json.dumps(blockout.run(shot, a.render)))


if __name__ == "__main__":
    main()
