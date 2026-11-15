"""Engine still -> image-model redraw: job files for the two image models, with the words built from the scene.

    python tools/assets/redraw/redraw.py RUN.json        (stdlib; writes the job files, prints their paths)

RUN: {"scenes": ["/abs/scene_out", ...]   scene tool outputs with truth.json (tools/assets/redraw/guides.py),
      "out": "/abs/folder",
      "method": "edit" | "image",
      "sheet": "/abs/character sheet.png",    identity picture (both models)
      "lora": "/abs/char.safetensors", "trigger": "chr_x", "lora_w": {"1": 0.8, "2": 0.63},   (method image)
      "cast": {"<scene key>": {"noun": "man"|"woman", "trigger", "lora", "lora_w", "sheet", "wear", "wear_back",
               "back_detail", "face_ref", "seat"}},   (several different people: each their own words, their LoRA on their own
                                              region of the frame (box + lora_pad) and their sheet as a reference;
                                              replaces sheet/lora/trigger/wear/wear_back; face_ref: a crop of their
                                              face from an accepted shot, facecrop.py, given where their face shows;
                                              seat: what they sit in, said when they are seated)
      "look": "...",                          (the closing words on the picture's look; default: real skin, film grain)
      "room_surfaces": "...",                 (the set's walls, floor and drapes in words, said in every shot)
      "set": "/abs/library set dir", "light_words": true,   (say where the light comes from, from the set's lit
                                              things and daylight, the same in every shot)
      "wear": "what the character wears",     (words; the scene tool does not know clothes yet)
      "wear_back": "what shows from behind",  (method image: said instead of "wear" for a person turned away)
      "set_words": "what the room holds",     (the set's library description)
      "strengths": [0.55, 0.7],               (method image: img2img strength of the engine still)
      "control": {"type": "depth", "scale": 0.8, "end": 1.0} | null,    (method image: the engine's own guide)
      "controls": [{"type": "depth"|"pose"|"canny", ...}, ...]          (several: depth, the bones' OpenPose
                                              picture, the outlines)
      "body_words": false,                    (say each body's orientation apart from the head)
      "mask_people": false | true | "front",  (method image: the still's people masks as the per-pixel strength,
                                               "strength_bg" for the room; "front": only people whose body is
                                               within 100 deg of the camera, the others at "strength_bg")
      "set_facts": true,                      (name the set's things the frame holds, lit ones as lit: guides.py --set)
      "room_all": false,                      (the room's contents for every shot; default: wide shots only)
      "extra_refs": ["/abs/anchor.png", ...], (method image: more reference pictures after the sheet, e.g. the
                                               scene's first accepted redraw, so later shots draw the same room)
      "seed": 7, "only": ["scene/cam", ...]}
Outputs: method edit -> edit_jobs.json for tools/assets/qwen_edit.py; method image -> image_jobs.jsonl for
tools/qwen_image.py --serve (one JSON line per picture). Each job id is "<scene>__<cam>[__s<strength>]".

The words follow general rules and the scene's own facts only: the camera distance (the still's camera kind), each
person's place in frame (their engine mask's box) and which way they face relative to the camera (the angle the
scene measured between the head's direction and the camera), what they wear, what the room holds. A person turned
away is said to be seen from behind with no face visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SHOT = {
    "wide": "A wide shot: the whole room in view, the people small and full length, feet on the floor.",
    "medium": "A medium shot: the people seen from about the waist up, the room behind them.",
    "close": "A close shot: head and shoulders filling the frame, the room behind soft and out of focus.",
}


def facing_words(deg: float, pos: str = "his") -> str:
    """How a person is turned relative to the camera, from the scene's measured angle (0 = facing the camera)."""
    if deg < 25:
        return f"faces the camera; {pos} face is seen from the front"
    if deg < 60:
        return f"is turned a little to one side, a three-quarter view; {pos} face is visible"
    if deg < 115:
        return "is seen in profile, side-on to the camera"
    if deg < 150:
        return (f"is turned away from the camera, seen three-quarters from behind: we see the back and side of {pos} "
                f"head, {pos} ear and the edge of {pos} cheek, and the back of {pos} clothes; {pos} face is not visible")
    return (f"is seen from directly behind, facing away from the camera: we see the back of {pos} head, {pos} hair and "
            f"the back of {pos} clothes and shoulders; {pos} face is not visible")


def place_words(box) -> str:
    cx = (box[0] + box[2]) / 2
    side = "on the left" if cx < 0.38 else "on the right" if cx > 0.62 else "in the centre"
    near = "in the foreground, close to the camera" if box[3] - box[1] > 0.95 and box[2] - box[0] > 0.3 else ""
    return side + (", " + near if near else "")


def people(shot: dict, poses: dict | None = None) -> list:
    """[(key, box, angle, posture)] of the people visible in the still, left to right; posture "sitting" when the
    scene posed them seated (pose name), the lying words for a lying stance, else "standing"."""
    poses = poses or {}
    out = [(k, v["box"], v.get("head_to_camera_deg", 0.0),
            "lying in bed on the back, the bedclothes over the body" if "lying" in (poses.get(k) or "") else
            "sitting" if any(w in (poses.get(k) or "") for w in ("sit", "seat")) else "standing",
            v.get("body_to_camera_deg", v.get("head_to_camera_deg", 0.0)), v.get("nearer_shoulder"))
           for k, v in shot["people"].items() if v.get("box") and v.get("area", 0) > 0.002]
    return sorted(out, key=lambda x: (x[1][0] + x[1][2]) / 2)


SEAT_WORDS = ("chair", "sofa", "settee", "stool", "bench", "seat")


def seats_of(ppl: list, shot: dict) -> dict:
    """For each sitting person, the set thing they sit in: the seat whose frame box overlaps the lower half of
    their box most. Said in the person's words ("she sits in the armchair (...)"): with two armchairs named in a
    wide, the redraw sat her in the wrong one (11-11)."""
    out = {}
    for k, b, ang, post, *_ in ppl:
        if post != "sitting":
            continue
        low = [b[0], (b[1] + b[3]) / 2, b[2], b[3]]
        best = (0.0, None)
        for t in shot.get("things") or []:
            if not any(w in t["name"] for w in SEAT_WORDS):
                continue
            tb = t["box"]
            ix = max(0.0, min(low[2], tb[2]) - max(low[0], tb[0]))
            iy = max(0.0, min(low[3], tb[3]) - max(low[1], tb[1]))
            a = ix * iy
            if a > best[0]:
                best = (a, t)
        if best[1] is not None:
            out[k] = best[1]["words"]
    return out


def heads_out(res: dict, cam: str) -> set:
    """People in this still whose head is out of the frame (the scene's frame check: head top and chin), e.g. a
    listener's shoulder at the edge of a close shot. Named as a full person facing the camera, with their LoRA,
    the redraw gave the speaker the listener's moustache (11-11, a close of a clean-shaven man)."""
    out = set()
    for sh in res.get("shots", []):
        if sh.get("name") != cam:
            continue
        checks = dict(sh.get("others_check") or {})
        subj = [k for k in (sh.get("truth") or {}) if k not in checks]
        if subj and sh.get("frame_check"):
            checks[subj[0]] = sh["frame_check"]
        for k, fc in checks.items():
            pts = [fc.get("head_top"), fc.get("chin")]
            if all(pts) and not all(0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0 for q in pts):
                out.add(k)
    return out


def edge_words(edge: list) -> str:
    if not edge:
        return ""
    return (" At the edge of the frame only part of another person's shoulder and clothes is seen, their head out of "
            "the picture; no other face is in the picture.")


def scene_poses(sd: Path) -> dict:
    """Each person's posture word source, from the scene's own job (job.json): a posed character's pose name, a
    plain-mesh figure's library tags ("seated": a figure made sitting). The scene result's subjects do not carry the
    pose name, so before this every person was said to be standing."""
    jp = sd / "job.json"
    if not jp.exists():
        return {}
    j = json.loads(jp.read_text())
    j = j.get("job", j)
    out = {k: ("lying" if ((v.get("body_pose") or {}).get("params") or {}).get("lie") or v.get("_lie")
               else (v.get("body_pose") or {}).get("pose") or "") for k, v in (j.get("characters") or {}).items()}
    lib = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(lib))
    import config as C
    for k, f in (j.get("figures") or {}).items():
        a = C.LIBRARY / f.get("id", "") / "asset.json"
        tags = json.loads(a.read_text()).get("tags", []) if a.exists() else []
        out[k] = "seated" if "seated" in tags else ""
    return out


def scene_wear(sd: Path) -> dict:
    """Per person, this scene's own wardrobe facts from its job (scene spec "wear_add": words for what this scene
    adds, e.g. a veil; "face_covered": the face is behind it, so no face reference and no face words)."""
    jp = sd / "job.json"
    if not jp.exists():
        return {}
    j = json.loads(jp.read_text())
    j = j.get("job", j)
    return {k: {"wear_add": v.get("wear_add"), "face_covered": bool(v.get("face_covered")), "id": v.get("id")}
            for k, v in (j.get("characters") or {}).items()}


LOOK_FACTS = ("hair_colour", "hair_worn", "facial_hair")


def entry_facts(fig: str | None) -> dict:
    """A built figure's look facts from its library entry ("facts": hair colour, how it is worn, facial hair,
    missing layers), the character builder's own record of what the figure has; a stance or wardrobe variant
    without its own takes its base entry's (the longest id prefix that has them)."""
    if not fig:
        return {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as C
    parts = fig.split("_")
    for n in range(len(parts), 1, -1):
        p = Path(C.LIBRARY) / "character" / "_".join(parts[:n]) / "character.json"
        if p.exists():
            f = json.loads(p.read_text()).get("facts")
            if f:
                return f
    return {}


def look_words(f: dict, pos: str) -> tuple:
    """Hair, facial hair and missing layers in words, from the entry's facts -> (words, missing fact names)."""
    miss = [k for k in LOOK_FACTS if not f.get(k)]
    w = []
    col, worn = f.get("hair_colour"), f.get("hair_worn") or ""
    if col and "hair" in worn and " " not in col:
        w.append(worn.replace("hair", f"{col} hair", 1))         # "short hair, slicked back" -> "short black hair, .."
    elif col:
        w.append((worn + f", the hair {col}") if worn else f"{col} hair")
    elif worn:
        w.append(worn)
    if f.get("facial_hair") and f["facial_hair"] != "none":
        w.append(f["facial_hair"])
    elif f.get("facial_hair") == "none" and pos == "his":
        w.append("clean-shaven")
    for m in f.get("missing_layers") or []:
        w.append(m)
    return "; ".join(w), miss


LORA_HEAD = (0.20, 0.36)     # head height in frame: at or under the first, the one-person weight; at or over the
                             # second, the run's weight for the number of people; a straight ramp between


def lora_by_head(lw: float, run: dict, hh: float | None, n: int) -> float:
    """A person's character-model weight in their own region, rising as their head gets smaller in frame: in
    two-person keys at 0.63 his outfit and face held with the head at 0.36-0.41 of the frame and were lost at
    0.16-0.31 (11-12)."""
    if n <= 1 or hh is None:
        return lw
    one = float((run.get("lora_w") or {}).get("1", 0.8))
    lo, hi = LORA_HEAD
    t = min(1.0, max(0.0, (hi - hh) / (hi - lo)))
    return round(lw + (one - lw) * t, 3)


def crowd_words(crowd: str | None, kind: str) -> str:
    """The place's crowd, as the director's set sheet names it: nobody builds or stages them, the picture fills
    them in behind the people of the shot (never in a close shot, where there is no background to hold them)."""
    if not crowd or kind == "close":
        return ""
    return (f" Far behind them, small and out of focus, the place's other people go about their business: {crowd.strip().rstrip('.')}."
            " None of them is near the camera and none has the face of anyone named above.")


def setting_words(sd: Path, cam: str) -> str:
    """The director's own words for what this shot shows of the place (the plan's "setting", carried on the staged
    camera): what a screen shows, what is seen outside, how the light falls. The engine cannot stage these."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return ""
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    vw = c.get("view") or {}
    if vw.get("ahead") and not (c.get("position") and c.get("target")):
        # a camera the scene tool placed has no position in the spec: the staged one is in the scene's truth
        # (without it every cabin shot fell through to the plain setting and no view reached the redraw, 11-14)
        tf = sd / "truth.json"
        tc = ((json.loads(tf.read_text()).get("shots") or {}).get(cam) or {}).get("camera") if tf.exists() else None
        c = {**c, **{k: (tc or {}).get(k) for k in ("position", "target")}}
    if vw.get("ahead") and c.get("position") and c.get("target"):
        # a cabin with a view outside: what the frame shows of it depends on which way the camera looks. Looking
        # the way the cabin faces, the view of this shot is ahead through the canopy; looking back at whoever sits
        # in it, the view is behind the camera and what is behind the person is the cabin's rear and the far dark
        # (told of the view regardless, the redraw painted its mountain behind the pilot's seat, 11-14)
        d = [c["target"][0] - c["position"][0], c["target"][1] - c["position"][1]]
        n = (d[0] ** 2 + d[1] ** 2) ** 0.5 or 1.0
        fwd = (d[0] * vw["ahead"][0] + d[1] * vw["ahead"][1]) / n
        if fwd > 0.25:
            return f" In this shot, ahead through the canopy: {vw['words'].strip()}." if vw.get("words") else ""
        if fwd < -0.25:
            return (" In this shot the camera looks back into the cabin: behind the person is the cabin's rear, and "
                    "through any window only black space with a few faint stars.")
        return (f" In this shot, to one side through the canopy: {vw['words'].strip()}; the rest of the windows show "
                "black space with a few faint stars.") if vw.get("words") else ""
    return f" In this shot: {c['setting'].strip()}" if c.get("setting") else ""


def facing_to_words(sd: Path, ppl: list, cast: dict | None) -> str:
    """Whom each person in the frame is turned toward, from the staged spec ("facing_to"): the redraw kept the
    engine's bodies, but a video model read two men at a window as looking out of it (11-12)."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds or not cast:
        return ""
    ch = json.loads(rounds[-1].read_text()).get("characters") or {}
    inf = {p[0] for p in ppl}
    nm = names(ppl, cast)
    out = []
    for k in [p[0] for p in ppl]:
        t = (ch.get(k) or {}).get("facing_to")
        if t and t in ch and t != k:
            who = nm.get(k, "the person")
            tgt = nm.get(t) if t in inf else (who_of(cast, t).get("noun") and f"the {who_of(cast, t)['noun']}")
            if tgt:
                out.append(f"{who[0].upper() + who[1:]} is turned toward {tgt}.")
    return (" " + " ".join(out)) if out else ""


def kind_of(cam: str, shot: dict) -> str:
    for k in ("close", "medium", "wide"):
        if cam.startswith(k):
            return k
    k = "wide" if shot.get("kind") in ("wide", "full") else shot.get("kind", "medium")
    if k == "wide":       # a placed camera is "wide" to the scene tool; a person filling the frame's height is not
        tall = max(((v.get("box") or [0, 0, 0, 0])[3] - (v.get("box") or [0, 0, 0, 0])[1]
                    for v in shot.get("people", {}).values()), default=0.0)
        if tall > 0.75:
            return "medium"
    return k


def body_words(deg: float, near: str | None, who: str = "His", pos: str = "his", back: str = "") -> str:
    """The body's own orientation, apart from the head (the scene's hips-to-camera angle and nearer shoulder): a
    redraw kept a turned-away head but swung the body to profile or front when only the head was described
    (11-11, three-quarter-away shots). `back`: what of the clothes faces the camera from behind (the cast's words)."""
    sh = f", {pos} {near} shoulder nearer the camera" if near else ""
    if deg < 60:
        return f"{who} body faces the camera."
    if deg <= 100:
        return f"{who} body is side-on to the camera{sh}."
    return (f"{who} body is turned away from the camera: we see {pos} back{sh}; the front of {pos} clothes is not "
            f"visible" + (f", and {back} faces the camera." if back else "."))


NUM = {2: "Two", 3: "Three", 4: "Four"}
DEFAULT_WHO = {"noun": "man", "pos": "his", "subj": "he"}


def who_of(cast: dict, key: str) -> dict:
    """A person's words from the run's cast (noun, pronouns, trigger, clothes); a run without a cast is one man."""
    w = dict(DEFAULT_WHO)
    w.update(cast.get(key) or cast.get("*") or {})
    if w["noun"] in ("woman", "girl", "lady"):
        w.setdefault("pos", "her")
        w.setdefault("subj", "she")
        w["pos"], w["subj"] = ("her", "she") if w.get("pos") == "his" else (w["pos"], w["subj"])
    return w


def names(ppl: list, cast: dict) -> dict:
    """What each person in frame is called in the words: 'the man' when the noun is unique in frame, else with
    their place ('the man on the left')."""
    nouns = [who_of(cast, p[0])["noun"] for p in ppl]
    out = {}
    for p, n in zip(ppl, nouns):
        out[p[0]] = f"the {n}" if nouns.count(n) == 1 else f"the {n} {place_words(p[1]).split(',')[0]}"
    ORD = ("first", "second", "third", "fourth")
    for n in set(nouns):                 # two of a noun on the same side: count them from the left instead
        ks = [p[0] for p, m in zip(ppl, nouns) if m == n]          # ppl is ordered left to right
        if len(ks) > 1 and len({out[k] for k in ks}) < len(ks):
            for i, k in enumerate(ks):
                out[k] = f"the {ORD[min(i, 3)]} {n} from the left"
    return out


def people_words(ppl: list, with_body: bool = False, cast: dict | None = None, speaker: str | None = None) -> str:
    if not ppl:
        return ""
    cast = cast or {}
    n = len(ppl)
    nm = names(ppl, cast)
    ws = [who_of(cast, p[0]) for p in ppl]
    same = len({(w["noun"], w.get("trigger")) for w in ws}) == 1
    if n == 1:
        head = f"One {ws[0]['noun']}, {{T0}}, is in the picture."
    elif same:
        head = f"{NUM.get(n, n)} {ws[0]['noun']}{'e' if ws[0]['noun'].endswith('s') else ''}s, each {{T0}}, are in the picture, dressed alike."
        head = head.replace("mans", "men").replace("womans", "women")
    else:
        head = (f"{NUM.get(n, n)} different people are in the picture: "
                + " and ".join(f"{nm[p[0]]} ({{T{i}}})" for i, p in enumerate(ppl)) + ".")
    parts = [head]
    for i, (k, box, ang, post, bdeg, near) in enumerate(ppl):
        w = ws[i]
        who = nm[k][0].upper() + nm[k][1:]
        placed = nm[k] != f"the {w['noun']}"           # the name already says where they are
        parts.append(f"{who} is {post} and {facing_words(ang, w['pos'])}." if placed else
                     f"{who} is {post} {place_words(box)} and {facing_words(ang, w['pos'])}.")
        if with_body:
            parts.append(body_words(bdeg, near, w["pos"].capitalize(), w["pos"], w.get("back_detail", "")))
        if post == "sitting" and w.get("seat"):         # what they sit in, in the same words as the set's own chair
            parts.append(f"{w['subj'].capitalize() if n == 1 else who} sits in {w['seat']}.")
    out = " ".join(parts)
    if speaker and speaker in nm:
        out += f" {nm[speaker][0].upper() + nm[speaker][1:]} is speaking, mouth slightly open mid-word."
    for i, w in enumerate(ws):
        out = out.replace(f"{{T{i}}}", w.get("trigger") or w["noun"])
    return out


NECKWEAR = ("collar", "cravat", "tie", "stock", "neckcloth", "frill", "ruff", "scarf")


def neck_words(w: dict) -> str:
    """The person's own collar and neckwear words, repeated as a fixed fact where their front shows: drawn from
    the sheet alone, a close redraw opened the man's wing collar into a wide spread collar (11-11)."""
    parts = [x.strip() for x in w.get("wear", "").replace(";", ",").split(",")]
    neck = [x for x in parts if any(n in x for n in NECKWEAR)]
    if not neck:
        return ""
    return (f"At {w['pos']} neck, exactly as on {w['pos']} character sheet and worn the same way: "
            + ", ".join(neck) + ".")


def limb_words(ppl: list, shot: dict, cast: dict) -> str:
    """Each person's arms and hands the engine shows (the guides' joints in the frame and seen), said as a fact:
    with a coat's sides over the arms in the still, the redraw drew a man with no arms (11-11)."""
    nm = names(ppl, cast)
    out = []
    for k, box, ang, post, bdeg, near in ppl:
        if bdeg > 100:
            continue
        j2 = (shot.get("people", {}).get(k) or {}).get("joints2d") or {}
        arms = sum(1 for s in ("l", "r") if any(n in j2 and j2[n][2] for n in (f"lowerarm_{s}", f"hand_{s}")))
        hands = sum(1 for s in ("l", "r") if f"hand_{s}" in j2 and j2[f"hand_{s}"][2])
        if not arms:
            continue
        w = who_of(cast, k)
        who = nm[k][0].upper() + nm[k][1:]
        a = "both of " + w["pos"] + " arms are" if arms == 2 else "one of " + w["pos"] + " arms is"
        h = (", with both hands" if hands == 2 else ", with its hand" if hands == 1 and arms == 1 else
             ", with one hand" if hands == 1 else "")
        out.append(f"{who}: {a} visible in their sleeves{h} in view, as in the engine frame.")
    return " ".join(out)


def any_away(ppl: list) -> bool:
    return any(p[2] >= 115 for p in ppl)


def edit_prompt(kind: str, ppl: list, wear: str, room: str, room_all: bool = False, things: str = "") -> str:
    away = any_away(ppl)
    ident = ("Picture 2 shows who he is: his face, his hair and his clothes"
             + (" (use it for his hair, build and clothes; where he is turned away, keep him turned away exactly "
                "as in Picture 1 and do not show his face)" if away else "") + f": {wear}.")
    return " ".join([
        "Picture 1 is a frame rendered by a game engine. Keep everything about it: the same camera position, lens, "
        "framing and perspective; every person in the same place, at the same size, in the same pose and turned the "
        "same way as in Picture 1; the room's layout unchanged, every wall, window, door, fireplace and piece of "
        "furniture where it is, nothing added, nothing removed, nothing covered.",
        "Change only how it looks: make it a photographed frame from a period film, with real skin and hair, woven "
        "wool and linen with natural folds, real wood, plaster, fabric and brass, the light coming from the same "
        "directions, film grain.",
        SHOT[kind], people_words(ppl), ident,
        room_words(kind, room, room_all) if not things or kind != "wide" else f"{room.split(':')[0]}.", things])


def frame_place(box) -> str:
    cx = (box[0] + box[2]) / 2
    return "on the left" if cx < 0.35 else "on the right" if cx > 0.65 else "in the middle"


PROP_SEEN = 0.3       # share of a free prop's sample points seen in the still
PROP_AREA = 0.004     # share of the frame its box covers
MAX_THINGS = 10       # the biggest things in frame; more words only dilute the prompt


def set_words(shot: dict) -> str:
    """The set's things the engine sees in this frame (guides.py --set), with their place and whether they burn or
    are lit, plus where the daylight comes from. Nothing that is out of frame is named, except the daylight."""
    def area(t):
        b = t["box"]
        return (b[2] - b[0]) * (b[3] - b[1])
    # fixtures when partly seen, lit things, and a free prop (a rug, a picture, a sofa) when most of it is in frame
    # and it covers a fair part of the frame: unnamed, the redraw repainted such things differently in every shot
    th = [t for t in shot.get("things") or [] if t["visible_frac"] >= 0.05 and t["kind"] != "prop" or
          (t.get("lit") and t["visible_frac"] >= 0.02) or
          (t["kind"] == "prop" and t["visible_frac"] >= PROP_SEEN and area(t) >= PROP_AREA)]
    th = [t for t in th if t["name"] != "rug" or t["visible_frac"] >= 0.2]
    big = [t for t in th if t.get("large")]                   # large furniture always: never dropped by the cap
    th = big + [t for t in sorted(th, key=area, reverse=True) if not t.get("large")][:max(0, MAX_THINGS - len(big))]
    parts = []
    for t in sorted(th, key=lambda t: (t["box"][0] + t["box"][2]) / 2):
        # only a lit thing whose flame is out of view loses its flame words (null = not a lit thing: keep its look)
        w = t["words"] if t.get("flame_in_view") is not False else t["words"].split(" with ")[0].split(",")[0]
        if t.get("place"):                 # where it stands in the room, the same words in every shot
            w += f", {t['place']},"
        if t["kind"] == "window" and t.get("view"):
            w += f" showing {t['view']}"
        elif t["kind"] == "window" and shot.get("daylight"):
            w += ", bright with " + shot["daylight"] + " outside"
        parts.append(f"{w} {frame_place(t['box'])}")
    out = ("In the frame: " + "; ".join(parts) + ".") if parts else ""
    if shot.get("daylight") and not any(t["kind"] == "window" for t in th):
        out += f" Soft {shot['daylight']} comes from a window out of frame."
    lit = [t for t in th if (t.get("fire") or t.get("lit")) and t.get("flame_in_view") is not False]
    # a fire whose grate is just out of view but whose fireplace shows: lit all the same (a lit source is a fact of
    # the whole scene); with no words the redraw drew the visible fireplace cold in one key, burning in the next
    edge = [t for t in th if t.get("fire") and t.get("flame_in_view") is False]
    for t in edge:
        out += (f" The fire in {t['words'].split(' with ')[0]} at the frame's edge is burning: its warm firelight "
                "glows on the fireplace, the hearth and the floor in front of it.")
    if lit:
        out += " " + " ".join(
            (f"The fire in {t['words'].split(' with ')[0]} burns brightly with real flames, as in Picture 1."
             if t.get("fire") else f"{t['words'].split(',')[0].capitalize()} is lit, its flame burning, as in Picture 1.")
            for t in lit)
    return out.strip()


def room_words(kind: str, room: str, room_all: bool = False) -> str:
    """The room's contents only for a wide shot: given the list for a closer shot, the edit model widened the
    view to show it all (11-11 run a: every medium and close redraw pulled back and rebuilt the room)."""
    if kind == "wide" or room_all:
        return f"The room: {room}"
    return ("Behind him keep exactly the part of the room that Picture 1 shows, at the same scale and as out of "
            "focus as it is there; do not widen the view, do not show more of the room, add nothing to the walls.")


def wear_words(ppl: list, wear: str, wear_back: str | None, cast: dict | None = None, neck: bool = False) -> str:
    """Clothes for the people who show their front; a person turned away (> 100 deg) is given only what shows
    from behind (wear_back), when the run has it: with the full list a strong redraw put the cravat and waistcoat
    on the back of a man seen from behind (11-11, strengths 0.8-0.95). Each person's own clothes from the cast."""
    cast = cast or {}
    nm = names(ppl, cast)
    ws = {p[0]: who_of(cast, p[0]) for p in ppl}
    same = len({(w["noun"], w.get("trigger")) for w in ws.values()}) == 1
    if not cast and not wear_back:
        return f"He wears {wear}." if len(ppl) <= 1 else f"Each wears {wear}."
    out = []
    for k, box, ang, post, bdeg, near in ppl:
        w = ws[k]
        wr, wb = w.get("wear", wear), w.get("wear_back", wear_back)
        who = w["subj"].capitalize() if len(ppl) == 1 else nm[k][0].upper() + nm[k][1:]
        if same and len(ppl) > 1 and not wb:
            return f"Each wears {wr}."
        out.append(f"{who} is seen from behind: we see only {wb}." if wb and max(ang, bdeg) > 100
                   else f"{who} wears {wr}.")
        if neck and max(ang, bdeg) <= 100 and w.get("wear"):
            out.append(neck_words(w))
    return " ".join(x for x in out if x)


def image_prompt(kind: str, ppl: list, wear: str, room: str, trigger: str, things: str = "",
                 wear_back: str | None = None, with_body: bool = False, cast: dict | None = None,
                 look: str = "", speaker: str | None = None, shot: dict | None = None, facts: bool = False) -> str:
    cast = cast or {"*": {"trigger": trigger}}
    return " ".join(x for x in [
        f"A photographed frame from a period film. {SHOT[kind]}", people_words(ppl, with_body, cast, speaker),
        limb_words(ppl, shot, cast) if facts and shot else "",
        wear_words(ppl, wear, wear_back, cast, neck=facts and kind != "wide"),
        (f"The room: {room}" if not things else f"{room.split(':')[0]}. {things.replace(', as in Picture 1', '')}"),
        look or "Real skin and hair, woven cloth with natural folds, natural light, film grain."] if x)


def faces(ppl: list, cast: dict) -> list:
    """Each person's face reference (cast "face_ref": a crop of their face from the first accepted close shot,
    facecrop.py) for the people in frame whose face shows: beside the LoRA and the sheet, it holds one face from
    shot to shot where the LoRA alone let a man's face drift between a single and an over-the-shoulder shot."""
    out = []
    for k, box, ang, *_ in ppl:
        f = (cast.get(k) or {}).get("face_ref")
        if f and ang < 115 and f not in out:
            out.append(f)
    return out


def anchor_crops(run: dict, out: Path) -> dict:
    """{thing name: crop path} of each large piece of furniture in the run's room anchor (run "anchor": {"image",
    "scene", "cam"}: an accepted key that shows the room), from the engine's own box of it in that shot. Given as a
    reference wherever that piece is in frame (several anchors: each piece from the one that shows it biggest), they
    hold one look per chair across shots (11-11: the leather chair
    drawn brown in one shot, red in another, button-backed or plain)."""
    anchors = run.get("anchor") or []
    anchors = [anchors] if isinstance(anchors, dict) else anchors
    if not anchors:
        return {}
    from PIL import Image
    best = {}                                  # per piece: the anchor that shows it biggest
    for a in anchors:
        sh = json.loads((Path(a["scene"]) / "truth.json").read_text())["shots"][a["cam"]]
        for t in sh.get("things") or []:
            if not t.get("large") or t["visible_frac"] < 0.3 or t["kind"] == "prop":
                continue
            b = t["box"]
            ar = (b[2] - b[0]) * (b[3] - b[1])
            if ar >= 0.01 and ar > best.get(t["name"], (0,))[0]:
                best[t["name"]] = (ar, a["image"], b)
    d = out / "anchor_crops"
    d.mkdir(parents=True, exist_ok=True)
    res = {}
    for n, (ar, img, b) in best.items():
        im = Image.open(img).convert("RGB")
        W, H = im.size
        p = d / f"{n}.png"
        im.crop((int(b[0] * W), int(b[1] * H), int(b[2] * W), int(b[3] * H))).save(p)
        res[n] = str(p)
    return res


def light_words(set_dir: str | None) -> str:
    """Where the light comes from, from the set's own facts (setfacts: lit things, a fire, daylight through a window):
    the same sentence for every shot of the set, so the redraw keeps one light and mood across cuts."""
    if not set_dir:
        return ""
    import setfacts
    th = setfacts.things(set_dir)
    src = []
    for t in th:
        if t.get("fire"):
            src.append("the fire in the grate")
        elif t.get("lit"):
            src.append("the " + " ".join(w for i, w in enumerate(t["name"].split("_")) if i == 0 or len(w) > 2))
    day = setfacts.daylight(set_dir)
    if day and any(t["kind"] == "window" for t in th):
        src.append(f"the {day} from the window")
    if not src:
        return ""
    lst = ", ".join(src[:-1]) + (" and " if len(src) > 1 else "") + src[-1]
    return (f"The room is lit only by {lst}: pools of light near them, deep shadows in the corners and under the "
            f"furniture, the walls away from the light falling into darkness.")


def blocked_still(run: dict, out: Path, scene: str, cam: str, still: str) -> str:
    """The picture the draw starts from: the engine still, or with "blocked_stills": true its colour-blocked copy
    (detail.py block: each large piece pulled toward its own base colour) when one was made. Off by default."""
    p = out / "detail" / "blocked" / f"{scene}__{cam}.png"
    return str(p) if run.get("blocked_stills") and p.exists() else still


def main():
    run = json.loads(Path(sys.argv[1]).read_text())
    # the room in words is the set's own (its builder wrote them from its pieces and its period kit) unless the
    # run gives its own
    if run.get("set") and (Path(run["set"]) / "spec.json").exists():
        ssp = json.loads((Path(run["set"]) / "spec.json").read_text())
        run.setdefault("set_words", ssp.get("words") or ssp.get("description") or "")
        run.setdefault("room_surfaces", ssp.get("surfaces_words") or "")
        run.setdefault("crowd", ssp.get("crowd") or "")
    out = Path(run["out"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(run.get("seed", 7))
    only = set(run.get("only") or [])
    jobs = []
    crops = anchor_crops(run, out)
    for sd in map(Path, run["scenes"]):
        truth = json.loads((sd / "truth.json").read_text())
        res = json.loads((sd / "result.json").read_text())
        room = run["set_words"]
        lights = light_words(run.get("set")) if run.get("light_words") else ""
        for cam, sh in truth["shots"].items():
            sid = f"{sd.name}/{cam}"
            if only and sid not in only:
                continue
            if sh.get("reject"):            # the staging check failed (e.g. the speaker's face is not shown)
                print(f"skip {sid}: {sh['reject']}")
                continue
            kind = kind_of(cam, sh)
            poses = scene_poses(sd)
            ppl = people(sh, poses)
            out_heads = heads_out(res, cam)
            edge = [x for x in ppl if x[0] in out_heads and len(ppl) > 1]
            ppl = [x for x in ppl if x not in edge]
            seats = seats_of(ppl, sh)
            base = {"scene": sd.name, "cam": cam, "kind": kind, "still": sh["still"],
                    "people": [{"key": k, "box": b, "head_to_camera_deg": a, "posture": q, "body_to_camera_deg": bd,
                                "nearer_shoulder": ns} for k, b, a, q, bd, ns in ppl]}
            if run["method"] == "edit":
                jobs.append({"id": f"{sd.name}__{cam}", "prompt": edit_prompt(kind, ppl, run["wear"], room, run.get("room_all", False),
                                                     set_words(sh) if run.get("set_facts") else "", run.get("wear_back"),
                                                run.get("body_words", False)),
                             "images": [sh["still"], run["sheet"]], "seed": seed,
                             "path": str(out / f"{sd.name}__{cam}.png"), "size": [1152, 640], "_meta": base})
            else:
                g = sd / "guides"
                n = max(1, len(ppl))
                lw = float((run.get("lora_w") or {}).get(str(min(n, 2)), 0.8 if n == 1 else 0.63))
                cast = run.get("cast") or {}
                cast = {k: (dict(v, seat=seats[k]) if k in seats else v) for k, v in cast.items()}
                for k, wv in scene_wear(sd).items():       # this scene's wardrobe on top of the cast's
                    if k in cast:
                        c2 = dict(cast[k])
                        # hair, facial hair, missing layers: the built figure's own facts (library entry), never
                        # a hand-written cast line (one said "dark brown-black" of a figure built black and grey)
                        # one fixed wording from the entry's facts in every key (the text is the fact; the hair check
                        # holds the keys together against the person's median and re-draws outliers)
                        # (a stand-in figure's hair is not this person's: their own sheet and words say theirs)
                        lw_, miss = ("", []) if c2.get("stand_in") else look_words(
                            entry_facts(wv.get("id")), "her" if c2.get("noun") in ("woman", "girl") else "his")
                        if lw_:
                            c2["wear"] = c2.get("wear", "") + "; " + lw_
                        if miss:
                            print(f"{sd.name} {k} ({wv.get('id')}): no entry facts for {', '.join(miss)}")
                        if wv.get("wear_add"):
                            c2["wear"] = c2.get("wear", "") + "; " + wv["wear_add"]
                        if wv.get("face_covered"):
                            c2.pop("face_ref", None)
                        cast[k] = c2
                if cast:          # each person's own LoRA on their own region; their sheets as references
                    loras, sheets = [], []
                    for k, b, *_ in ppl:
                        w = who_of(cast, k)
                        if w.get("lora"):
                            pad = run.get("lora_pad", 0.06)
                            reg = (sh["people"].get(k) or {}).get("region")
                            if len(ppl) <= 1:
                                area = [0.0, 0.0, 1.0, 1.0]
                            elif run.get("regions") and reg and Path(reg).exists():
                                area = reg      # the person's own region: nearer them than anyone else (guides.py)
                            else:
                                area = [max(0.0, b[0] - pad), max(0.0, b[1] - pad), min(1.0, b[2] + pad),
                                        min(1.0, b[3] + pad)]
                            j2 = (sh["people"].get(k) or {}).get("joints2d") or {}
                            hh = (j2["chin"][1] - j2["head_top"][1]) if "chin" in j2 and "head_top" in j2 else None
                            loras.append([w["lora"], float(w.get("lora_w", lora_by_head(lw, run, hh, len(ppl)))), area])
                        if w.get("sheet") and w["sheet"] not in sheets:
                            sheets.append(w["sheet"])
                else:
                    loras, sheets = [[run["lora"], lw, [0.0, 0.0, 1.0, 1.0]]], [run["sheet"]]
                for st, seed in [(a, b) for a in run.get("strengths", [0.7]) for b in (run.get("seeds") or [seed])]:
                    tag = f"__s{int(round(st * 100))}" + (f"__r{seed}" if run.get("seeds") else "")
                    j = {"id": f"{sd.name}__{cam}{tag}",
                         "prompt": image_prompt(kind, ppl, run.get("wear", ""), room, run.get("trigger", ""),
                                                set_words(sh) if run.get("set_facts") else "", run.get("wear_back"),
                                                run.get("body_words", False), cast or None,
                                                " ".join(x for x in (run.get("room_surfaces", ""), lights,
                                                                     run.get("look", "")) if x),
                                                sh.get("speaker"), sh, run.get("facts", False)) + edge_words(edge)
                         + setting_words(sd, cam) + crowd_words(run.get("crowd"), kind)
                         + facing_to_words(sd, ppl, cast),
                         "seed": seed, "path": str(out / f"{sd.name}__{cam}{tag}.png"),
                         "size": [1152, 640], "refs": sheets + faces(ppl, cast) + list(run.get("extra_refs") or [])
                         + [crops[t["name"]] for t in sorted(sh.get("things") or [], key=lambda t: -t.get("mask_area", 0))
                            if t["name"] in crops and t.get("mask")][:run.get("max_piece_refs", 2)],
                         "loras": loras,
                         "init": {"image": blocked_still(run, out, sd.name, cam, sh["still"]), "strength": st},
                         "_meta": dict(base, strength=st, seed=seed)}
                    cl = run.get("controls") or ([run["control"]] if run.get("control") else [])
                    pics = {"depth": "depth", "pose": "pose", "canny": "lines"}
                    cj = [{"image": str(g / f"{cam}_{pics[c['type']]}.png"), "type": c["type"],
                           "scale": c.get("scale", 0.8), "end": c.get("end", 1.0), "prepared": True}
                          for c in cl if (g / f"{cam}_{pics[c['type']]}.png").exists()]
                    if cj:
                        j["control"] = cj[0] if len(cj) == 1 else cj
                    if run.get("mask_people") and ppl:      # people redrawn harder than the room
                        mk = run.get("mask_people") if isinstance(run.get("mask_people"), str) else "people"
                        j["init"]["mask"] = str(g / f"{cam}_mask_{mk}.png")
                        j["init"]["strength_bg"] = run.get("strength_bg", 0.6)
                    jobs.append(j)
    if run["method"] == "edit":
        p = out / "edit_jobs.json"
        p.write_text(json.dumps(jobs, indent=1))
    else:
        p = out / "image_jobs.jsonl"
        p.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (out / "meta.json").write_text(json.dumps({j["id"]: dict(j["_meta"], prompt=j["prompt"]) for j in jobs}, indent=1))
    print(p, len(jobs))


if __name__ == "__main__":
    main()
