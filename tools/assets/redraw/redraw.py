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
import math
import re
import sys
from pathlib import Path

SHOT = {
    "wide": "A wide shot: the whole room in view, the people small and full length, feet on the floor.",
    "medium": "A medium shot: the people seen from about the waist up, the room behind them.",
    "close": "A close shot: head and shoulders filling the frame, the room behind soft and out of focus.",
}
# a place in the open (its set built as a world, sets/place_build.py world_stage): said of the open place, not a
# room. Told "the room behind", every close and medium of an outdoor place was drawn as a dark room with windows
# and a stone floor while its wides showed the river and the hills (11-17)
SHOT_OPEN = {
    "wide": "A wide shot out of doors: the whole place in view under the open sky, the people small and full length, "
            "feet on the ground.",
    "medium": "A medium shot out of doors: the people seen from about the waist up, the open land and the sky "
              "behind them.",
    "close": "A close shot out of doors: head and shoulders filling the frame, the open land behind soft and out of "
             "focus.",
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
            "lying on the back on top of the made bed, no bedclothes over the body" if "lying" in (poses.get(k) or "") else
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


EDGE_MIN = 0.03              # someone at the frame's edge, head out, is said when this share of the frame is theirs


def seen_share(sd: Path, cam: str) -> dict:
    """Each person's own share of a camera's still (the staging check's depth-pass mask), {} when not recorded."""
    try:
        h = (json.loads((sd / "speakers.json").read_text()).get("shots") or {}).get(cam, {}).get("history") or []
    except (OSError, ValueError):
        return {}
    ff = (h[-1].get("frame_facts") or {}) if h else {}
    return {k: f["mask"]["area"] for k, f in ff.items() if isinstance(f.get("mask"), dict) and "area" in f["mask"]}


def edge_words(edge: list, cast: dict | None = None) -> str:
    """Someone at the frame's edge with the head out of it: only a shoulder and clothes, said as that person's own
    clothes (told only "clothes", the redraw dressed a man in armour in the velvet of the person beside him)."""
    if not edge:
        return ""
    # (a creature at the edge is said as itself: listed with the people, its part of the frame was "another person's
    # clothes" and was painted as the dress of the woman beside it, 11-18)
    beasts = [str((cast or {}).get(x[0], {}).get("noun") or "") for x in edge
              if not (cast or {}).get(x[0], {}).get("wear") and str((cast or {}).get(x[0], {}).get("noun") or "").lower()
              not in ("", "man", "woman", "boy", "girl", "lady", "person", "child")]
    folk = [x for x in edge if str((cast or {}).get(x[0], {}).get("noun") or "") not in beasts]
    out = ""
    if folk:
        wear = [(cast or {}).get(x[0], {}).get("wear", "") for x in folk]
        wear = [", ".join(q.strip() for q in w.replace(";", ",").split(",")[:2]) for w in wear if w]
        out += (" At the edge of the frame only part of another person's shoulder and clothes is seen"
                + (f" ({'; '.join(wear)})" if wear else "") + ", their head out of the picture; no other face is in the "
                "picture.")
    for n in dict.fromkeys(beasts):
        out += f" At the edge of the frame only part of the {n}'s body is seen, its head out of the picture."
    return out


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
    return {k: {"wear_add": v.get("wear_add"), "face_covered": bool(v.get("face_covered")), "id": v.get("id"),
                "wear_set": v.get("wear_set"), "wear_back_set": v.get("wear_back_set"),
                "worn_sheet": v.get("worn_sheet")}
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


def head_things(sd: Path, cam: str) -> dict:
    """{scene key: [what they have on the head in this shot]} from the staged camera's carried things worn at the
    head, each by its library record's own first words. The figure wears it in the engine's frame, and the key
    said the person's hair and usual clothes first and the thing last in a list: it was drawn bareheaded (11-18)."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return {}
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import config as C
    out = {}
    for k, xs in (c.get("carried") or {}).items():
        for x in xs or []:
            if x.get("at") != "head":
                continue
            try:
                rec = json.loads((Path(C.LIBRARY) / "prop" / str(x.get("asset")) / "asset.json").read_text())
            except (OSError, ValueError):
                continue
            nm = str(rec.get("look") or "").split(";")[0].strip()
            if nm:
                out.setdefault(k, []).append(nm)
    return out


def laid_words(sd: Path, cam: str, ppl: list) -> str:
    """The things the staging laid on a piece of the set in this shot (the staged camera's carried things "on"),
    each by the plan's own name for it: they are in the engine's frame, and unsaid, a key drew another thing
    in their place or none (a blade on an anvil was a piece of armour in one key and gone in the next, 11-18)."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return ""
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    try:                                     # (the plan's own name for each made thing: the place's carried.json)
        made = {v: k for k, v in json.loads((sd.parent.parent / "carried.json").read_text()).items()}
    except (OSError, ValueError):
        made = {}
    inf, names_ = {p[0] for p in ppl}, []
    for k, xs in (c.get("carried") or {}).items():
        for x in xs or []:
            if x.get("at") != "on" or k not in inf:
                continue
            nm = str(made.get(str(x.get("asset"))) or "").strip()
            if nm and x.get("glow"):         # (hot from the fire in this shot: said, or the key drew it cold)
                nm += f", glowing {str(x['glow']).strip()} with its own heat and lighting what is near it"
            if nm and nm not in names_:
                names_.append(nm)
    if not names_:
        return ""
    # (said only as "lying where the picture has it", a blade on the anvil was drawn in the striker's raised hand)
    return (f" In nobody's hands, lying flat on the work in front of them where the picture has it, in its place and "
            f"at its size: {', '.join(names_)}; nothing else lies there.")


def look_words(f: dict, pos: str, covered: bool = False) -> tuple:
    """Hair, facial hair and missing layers in words, from the entry's facts -> (words, missing fact names).
    `covered`: the head has a thing on it in this shot, the hair is not said."""
    miss = [k for k in LOOK_FACTS if not f.get(k)]
    w = []
    col, worn = f.get("hair_colour"), f.get("hair_worn") or ""
    if covered:
        pass
    elif col and "hair" in worn and " " not in col:
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


def drawn_free(sd: Path, cam: str) -> bool:
    """Whether this shot is drawn without the staging's shapes holding it (the staged camera's "free"): a view of
    the whole place with nobody of the cast in it, where the shot's words may name what was never staged - riders,
    a crowd, a herd. The staged still is still where the drawing starts. (Held to the depth of an empty field, a
    shot of riders leaving a tower was drawn as the empty field, 11-20.)"""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return False
    return bool(next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam),
                     {}).get("free"))


def free_start(run: dict, still: str) -> str:
    """The picture a free shot's drawing starts from: the place's own picture (the run's first extra reference)
    when it is the size of the stills, else the engine still. Such a shot is the whole place as its picture sees
    it; the engine's frame of it adds nothing the picture lacks. (From the picture's eye the engine stood at the
    edge of the land made from the picture and half its frame was blank, 11-20.)"""
    ref = next(iter(run.get("extra_refs") or []), None)
    try:
        from PIL import Image
        if ref and Path(ref).exists() and Image.open(ref).size == Image.open(still).size:
            return str(ref)
    except OSError:
        pass
    return still


def setting_words(sd: Path, cam: str) -> str:
    """The director's own words for what this shot shows of the place (the plan's "setting", carried on the staged
    camera): what a screen shows, what is seen outside, how the light falls. The engine cannot stage these."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return ""
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    vw = c.get("view") or {}
    if vw.get("ahead"):
        # a cabin with a view outside: the engine hangs this shot's view where its camera looks (scene_job), so
        # whatever glass is in the frame shows it; the walls and frames between the panes are the cabin's own
        return (f" In this shot, outside the cabin, seen through its glass wherever a pane is in the picture: "
                f"{vw['words'].strip().rstrip('.')}. The glass sits in the cabin's own frame, which closes it all "
                "round; the cabin's walls stay as the picture shows them.") if vw.get("words") else ""
    return f" In this shot: {c['setting'].strip()}" if c.get("setting") else ""


def holds_words(sd: Path, cam: str, ppl: list, cast: dict | None, kind: str) -> str:
    """What each person in the frame holds in this shot, from the staged camera ("holds", the plan's own words):
    the engine stages no hand props. Not in a close shot: the hands are out of it."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds or kind == "close":
        return ""
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    held = c.get("holds") or {}
    nm = names(ppl, cast or {})
    try:                                     # where each hand is, from the staged bodies (the engine's own facts)
        subj = json.loads((sd / "result.json").read_text()).get("subjects") or {}
    except (OSError, ValueError):
        subj = {}
    out = []
    for k in [p[0] for p in ppl]:
        if held.get(k) and str(held[k]).split(" (")[0].strip().lower() not in ("nothing", "none"):
            who = nm.get(k, "the person")
            where = hand_place((subj.get(k) or {}).get("measured") or {})
            # (said without a place, a sword was drawn as a blade floating across the frame behind his head and a
            # smith's tongs as pointed ears, 11-17)
            out.append(f"{who[0].upper() + who[1:]} holds in the hands, not worn: {held[k]}"
                       + (f", {where}" if where else "") + "; it is in the hand and nowhere else, its whole length "
                       "joined to the hand, nothing of it floating or behind the head.")
    return (" " + " ".join(out)) if out else ""


def hand_place(m: dict) -> str:
    """Where the working (right) hand is, in plain words, from a measured body: raised above the head, in front at
    the chest or the waist, or down at the side; "" without joints."""
    j = (m.get("joints") or {}).get("hand_r")
    if not j or "head_top_z" not in m:
        return ""
    f = m.get("facing") or [0.0, -1.0]
    n = (f[0] ** 2 + f[1] ** 2) ** 0.5 or 1.0
    fwd = ((j[0] - m["pelvis_xy"][0]) * f[0] + (j[1] - m["pelvis_xy"][1]) * f[1]) / n
    if j[2] >= m["head_top_z"] - 0.05:
        return "in the right hand raised above the head"
    if fwd >= 0.2:
        return "in the right hand held out in front of the body at " + ("chest" if j[2] > m.get("waist_z", 1.0) + 0.25
                                                                         else "waist") + " height"
    return "in the right hand hanging down at the side, below the waist"


def cam_wears(sd: Path, cam: str) -> dict:
    """{scene key: what they have on in this shot} from the staged camera ("wears"), {} when it says nothing."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return {}
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    return c.get("wears") or {}


def worn_sheet(w: dict, worn: str | None) -> str | None:
    """The person's own sheet in what they have on now: of the cast entry's "worn_sheets" ({what is worn: sheet})
    the one sharing most words with the shot's worn things; None when none shares a word."""
    have = set(re.findall(r"[a-z]+", str(worn or "").lower()))
    best = max(((len(have & set(re.findall(r"[a-z]+", n.lower()))), p) for n, p in (w.get("worn_sheets") or {}).items()),
               default=(0, None))
    return best[1] if best[0] else None


def wears_words(sd: Path, cam: str, ppl: list, cast: dict | None) -> str:
    """What each person in the frame has on in this shot beyond their usual clothes, from the staged camera
    ("wears", the director's account): the figures wear their usual look in every shot."""
    rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
    if not rounds:
        return ""
    c = next((c for c in json.loads(rounds[-1].read_text()).get("cameras") or [] if c.get("name") == cam), {})
    worn = c.get("wears") or {}
    nm = names(ppl, cast or {})
    out = []
    for k in [p[0] for p in ppl]:
        if worn.get(k):
            who = nm.get(k, "the person")
            out.append(f"{who[0].upper() + who[1:]} has on, worn and not held: {worn[k]}.")
        if k in (c.get("eyes_closed") or []):
            who = nm.get(k, "the person")
            out.append(f"{who[0].upper() + who[1:]} has the eyes closed.")
    return (" " + " ".join(out)) if out else ""


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


BEAST_HEAD = (0.45, 0.18)    # a creature's head height in the frame: at or over the first, its head alone is the
                             # picture; at or over the second, its head, neck and upper body; under it, the whole of it
BEAST_PITCH = 15.0           # the camera looking up or down at it by this many degrees is said


def beast_part(v: dict) -> str:
    """How much of a creature the engine's frame shows, from its measured head in the frame: "head", "upper" or
    "whole"."""
    j = (v or {}).get("joints2d") or {}
    if "chin" in j and "head_top" in j:
        hh = j["chin"][1] - j["head_top"][1]
    else:                                   # no measured head: by how much of the frame its box fills
        b = (v or {}).get("box") or [0, 0, 0, 0]
        hh = BEAST_HEAD[1] if (b[3] - b[1]) > 0.95 and (b[2] - b[0]) > 0.5 else 0.0
    return "head" if hh >= BEAST_HEAD[0] else "upper" if hh >= BEAST_HEAD[1] else "whole"


def beast_words(noun: str, v: dict, cam: dict | None) -> str:
    """What of a creature is in the picture, and from where it is seen. A creature said in a person's words
    ("standing", "from the waist up") with only its head in the engine's frame was redrawn as a rock in one key and
    a skull in another (11-18)."""
    part = beast_part(v)
    out = {"head": f"Only the {noun}'s head is in the picture, large and close, filling much of the frame: its whole "
                   f"head with its eyes, snout or muzzle and jaws, drawn as on its character sheet; its body is out "
                   f"of the frame.",
           "upper": f"The {noun}'s head, its neck and the upper part of its body are in the picture, its head drawn "
                    f"as on its character sheet; the rest of its body is out of the frame.",
           "whole": f"The whole {noun} is in the picture, head to tail."}[part]
    pos, tgt = (cam or {}).get("position"), (cam or {}).get("target")
    if pos and tgt:
        pitch = math.degrees(math.atan2(tgt[2] - pos[2], math.hypot(tgt[0] - pos[0], tgt[1] - pos[1]) or 1e-6))
        if pitch >= BEAST_PITCH:
            out += f" It is seen from below, the camera looking up at it: the underside of its jaw and its throat show."
        elif pitch <= -BEAST_PITCH:
            out += f" It is seen from above, the camera looking down on it."
    return out


def beast_shot(ppl: list, cast: dict, shot: dict | None, outdoors: bool) -> str:
    """The shot's opening words when everyone in it is a creature: said by what of the creature shows, never as
    people "from the waist up" or "head and shoulders"."""
    if not ppl or not shot or not all((cast.get(p[0]) or {}).get("creature") for p in ppl):
        return ""
    parts = [beast_part((shot.get("people") or {}).get(p[0])) for p in ppl]
    part = "head" if "head" in parts else "upper" if "upper" in parts else "whole"
    n = who_of(cast, ppl[0][0])["noun"] if len(ppl) == 1 else "creature"
    o, behind = (" out of doors", "the open land and the sky") if outdoors else ("", "the room")
    return {"head": f"A close shot{o} of a {n}'s head, {behind} behind it soft and out of focus.",
            "upper": f"A shot{o} of a {n}'s head, neck and upper body, {behind} behind it.",
            "whole": f"A wide shot{o}: the whole {n} in view, {behind} around it."}[part]


def people_words(ppl: list, with_body: bool = False, cast: dict | None = None, speaker: str | None = None,
                 shot: dict | None = None) -> str:
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
        if w.get("creature"):               # a creature: what of it shows, where, and how it is turned
            parts.append(beast_words(w["noun"], ((shot or {}).get("people") or {}).get(k) or {"box": box},
                                     (shot or {}).get("camera")))
            parts.append(f"It is {place_words(box).split(',')[0]} and {facing_words(ang, w['pos'])}.")
            continue
        parts.append(f"{who} is {post} and {facing_words(ang, w['pos'])}." if placed else
                     f"{who} is {post} {place_words(box)} and {facing_words(ang, w['pos'])}.")
        if with_body:
            parts.append(body_words(bdeg, near, w["pos"].capitalize(), w["pos"], w.get("back_detail", "")))
        if post == "sitting" and w.get("seat"):         # what they sit in, in the same words as the set's own chair
            parts.append(f"{w['subj'].capitalize() if n == 1 else who} sits in {w['seat']}.")
    out = " ".join(parts)
    if speaker and speaker in nm:
        out += (f" {nm[speaker][0].upper() + nm[speaker][1:]} is speaking, "
                + ("its jaws a little parted." if (cast.get(speaker) or {}).get("creature")
                   else "mouth slightly open mid-word."))
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


def open_air_words(text: str) -> str:
    """A place in the open said as one: the concept tells every place's "floor" and "ceiling" as a room's; out of
    doors they are the ground and the sky (told of a "stone floor", the redraw laid flagstones on a river bank and
    walled it in, 11-17)."""
    def keep_case(word):
        return lambda m: word.capitalize() if m.group(0)[0].isupper() else word
    return re.sub(r"\bceiling\b", keep_case("sky"), re.sub(r"\bfloor\b", keep_case("ground"), text, flags=re.I),
                  flags=re.I)


def set_words(shot: dict, outdoors: bool = False) -> str:
    """The set's things the engine sees in this frame (guides.py --set), with their place and whether they burn or
    are lit, plus where the daylight comes from. Nothing that is out of frame is named, except the daylight (in a
    room: from a window; out of doors the place's own words say its light)."""
    def area(t):
        b = t["box"]
        return (b[2] - b[0]) * (b[3] - b[1])
    # fixtures when partly seen, lit things, and a free prop (a rug, a picture, a sofa) when most of it is in frame
    # and it covers a fair part of the frame: unnamed, the redraw repainted such things differently in every shot
    th = [t for t in shot.get("things") or [] if t["visible_frac"] >= 0.05 and t["kind"] != "prop" or
          (t.get("lit") and t["visible_frac"] >= 0.02) or
          (t["kind"] == "prop" and t["visible_frac"] >= PROP_SEEN and area(t) >= PROP_AREA)]
    th = [t for t in th if t["name"] != "rug" or t["visible_frac"] >= 0.2]
    # a place on the open ground is no thing to draw: the place's own words say its land ("the land at5 in the
    # middle; the land at7 on the right" went into every wide's words, 11-19)
    th = [t for t in th if t["kind"] != "mark"]
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
    if shot.get("daylight") and not outdoors and not any(t["kind"] == "window" for t in th):
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
        if not str((wb if wb and max(ang, bdeg) > 100 else wr) or "").strip():
            continue                             # (a creature wears nothing: "It wears ." was said of one)
        out.append(f"{who} is seen from behind: we see only {wb}." if wb and max(ang, bdeg) > 100
                   else f"{who} wears {wr}.")
        if neck and max(ang, bdeg) <= 100 and w.get("wear"):
            out.append(neck_words(w))
    return " ".join(x for x in out if x)


def image_prompt(kind: str, ppl: list, wear: str, room: str, trigger: str, things: str = "",
                 wear_back: str | None = None, with_body: bool = False, cast: dict | None = None,
                 look: str = "", speaker: str | None = None, shot: dict | None = None, facts: bool = False,
                 outdoors: bool = False) -> str:
    cast = cast or {"*": {"trigger": trigger}}
    return " ".join(x for x in [
        f"A photographed frame from a period film. "
        f"{beast_shot(ppl, cast, shot, outdoors) or (SHOT_OPEN if outdoors else SHOT)[kind]}",
        people_words(ppl, with_body, cast, speaker, shot),
        limb_words(ppl, shot, cast) if facts and shot else "",
        wear_words(ppl, wear, wear_back, cast, neck=facts and kind != "wide"),
        # (out of doors the place's words are its land, kept whole: cut at their first colon as a room's list is,
        # only "In the open air, <period>" was left of them; what stands there is named by the frame's things)
        # (with none of them in the frame, what stands in the place is not said at all: a close shot of a man, told
        # of the place's boulder, was drawn with a head of rough stone, 11-18)
        (room.split('; standing there')[0].rstrip('.') + "." if outdoors and not things else f"The room: {room}" if not things
         else f"{room.split('; standing there')[0].rstrip('.')}. {things.replace(', as in Picture 1', '')}" if outdoors
         else f"{room.split(':')[0].rstrip('.')}. {things.replace(', as in Picture 1', '')}"),
        look or "Real skin and hair, woven cloth with natural folds, natural light, film grain."] if x)


FACE_SEEN_DEG = 115          # a head turned less than this from the camera shows its face
FACE_SHEET_HEAD = 0.09       # a face this tall in the frame or taller (crown to chin, of the frame's height) is read


def sheet_ref(w: dict, worn: str | None, with_lora: bool = False) -> str | None:
    """The sheet given as a person's reference picture in a job, or None. Someone drawn by their own LoRA in their
    usual look needs none: the LoRA is their face, build and usual clothes, and every reference picture is paid for
    in each drawing (a job with one took 6.5 s, with two 11.7 s, three 15.3 s, four 19.4 s, and with more than one
    the card had no room to keep the references between steps, 11-19). Their sheet is given when they wear what the
    LoRA never saw (an outfit's own sheet, a sheet of the thing they have on), or when they have no LoRA.
    with_lora: the usual sheet beside the LoRA all the same: asked for the whole run ("sheet_with_lora"), or for
    this person in this shot because their face is turned to the camera and large enough to read (FACE_SHEET_HEAD).
    From her LoRA alone a woman of seventy was drawn as one of fifty in every shot that showed her face; the man
    beside her was himself either way (11-19). Small, turned away or far off, nobody's sheet is paid for."""
    if worn:
        return worn
    if w.get("_outfit_sheet") or not w.get("lora") or with_lora:
        return w.get("sheet")
    return None


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
        run.setdefault("outdoors", bool(ssp.get("world")))
    if run.get("outdoors"):
        run["room_surfaces"] = open_air_words(run.get("room_surfaces") or "")
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
            # (a sliver of someone at the edge is not said: told of a dress at the edge where 1.5% of a woman showed
            # beside a beast's head, the redraw painted the dress over the beast, 11-18)
            seen = seen_share(sd, cam)
            edge = [x for x in edge if seen.get(x[0], 1.0) >= EDGE_MIN]
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
                on_head = head_things(sd, cam)
                for k, wv in scene_wear(sd).items():       # this scene's wardrobe on top of the cast's
                    if k in cast:
                        c2 = dict(cast[k])
                        # a wardrobe variant (the book's change of clothes): its own outfit words and sheet in place
                        # of the usual ones (the figure wears that outfit; added on top, the usual clothes were said too)
                        if wv.get("wear_set"):
                            c2["wear"] = wv["wear_set"]
                            if wv.get("wear_back_set"):
                                c2["wear_back"] = wv["wear_back_set"]
                                c2["back_detail"] = wv["wear_back_set"].split(" and the back of ")[0]
                        if wv.get("worn_sheet"):           # (its sheet, not another of the person's worn sheets)
                            c2["sheet"], c2["worn_sheets"] = wv["worn_sheet"], {}
                            c2["_outfit_sheet"] = True     # (an outfit's own sheet: a reference their LoRA has not)
                        # hair, facial hair, missing layers: the built figure's own facts (library entry), never
                        # a hand-written cast line (one said "dark brown-black" of a figure built black and grey)
                        # one fixed wording from the entry's facts in every key (the text is the fact; the hair check
                        # holds the keys together against the person's median and re-draws outliers)
                        # (a stand-in figure's hair is not this person's: their own sheet and words say theirs)
                        lw_, miss = ("", []) if c2.get("stand_in") else look_words(
                            entry_facts(wv.get("id")), "her" if c2.get("noun") in ("woman", "girl") else "his",
                            covered=bool(on_head.get(k)))
                        if on_head.get(k):   # (said with the person, before the place and the shot)
                            c2["wear"] = (c2.get("wear", "") + "; on the head, worn over the hair and covering it: "
                                          + ", ".join(on_head[k]))
                        if lw_:
                            c2["wear"] = c2.get("wear", "") + "; " + lw_
                        if miss:
                            print(f"{sd.name} {k} ({wv.get('id')}): no entry facts for {', '.join(miss)}")
                        if wv.get("wear_add"):
                            c2["wear"] = c2.get("wear", "") + "; " + wv["wear_add"]
                        if wv.get("face_covered"):
                            c2.pop("face_ref", None)
                        cast[k] = c2
                worn_now = cam_wears(sd, cam)
                if cast:          # each person's own LoRA on their own region; their sheets as references
                    loras, sheets = [], []
                    for k, b, *rest_ in ppl:
                        w = who_of(cast, k)
                        j2_ = (sh["people"].get(k) or {}).get("joints2d") or {}
                        hh_ = (j2_["chin"][1] - j2_["head_top"][1]) if "chin" in j2_ and "head_top" in j2_ else None
                        face_big = (rest_[0] if rest_ else 0) < FACE_SEEN_DEG and (
                            hh_ >= FACE_SHEET_HEAD if hh_ is not None else (b[3] - b[1]) >= 0.6)
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
                        # (wearing something the book has a sheet of them in, that sheet is their reference here:
                        # told in words alone that a pilot had his helmet on, no key drew it, 11-15)
                        ws = sheet_ref(w, worn_sheet(w, worn_now.get(k)), bool(run.get("sheet_with_lora")) or face_big)
                        if ws and ws not in sheets:
                            sheets.append(ws)
                else:
                    loras, sheets = [[run["lora"], lw, [0.0, 0.0, 1.0, 1.0]]], [run["sheet"]]
                for st, seed in [(a, b) for a in run.get("strengths", [0.7]) for b in (run.get("seeds") or [seed])]:
                    tag = f"__s{int(round(st * 100))}" + (f"__r{seed}" if run.get("seeds") else "")
                    j = {"id": f"{sd.name}__{cam}{tag}",
                         "prompt": image_prompt(kind, ppl, run.get("wear", ""), room, run.get("trigger", ""),
                                                set_words(sh, run.get("outdoors", False)) if run.get("set_facts")
                                                else "", run.get("wear_back"), run.get("body_words", False), cast or None,
                                                " ".join(x for x in (run.get("room_surfaces", ""), lights,
                                                                     run.get("look", "")) if x),
                                                sh.get("speaker"), sh, run.get("facts", False),
                                                run.get("outdoors", False)) + edge_words(edge, cast)
                         + setting_words(sd, cam) + crowd_words(run.get("crowd"), kind)
                         + facing_to_words(sd, ppl, cast) + wears_words(sd, cam, ppl, cast) + laid_words(sd, cam, ppl)
                         + holds_words(sd, cam, ppl, cast, kind),
                         "seed": seed, "path": str(out / f"{sd.name}__{cam}{tag}.png"),
                         "size": [1152, 640], "refs": sheets + faces(ppl, cast) + list(run.get("extra_refs") or [])
                         + [crops[t["name"]] for t in sorted(sh.get("things") or [], key=lambda t: -t.get("mask_area", 0))
                            if t["name"] in crops and t.get("mask")][:run.get("max_piece_refs", 2)],
                         "loras": loras,
                         "init": {"image": blocked_still(run, out, sd.name, cam, sh["still"]), "strength": st},
                         "_meta": dict(base, strength=st, seed=seed)}
                    free_ = not ppl and drawn_free(sd, cam)
                    if free_:
                        j["init"]["image"] = free_start(run, sh["still"])
                    cl = run.get("controls") or ([run["control"]] if run.get("control") else [])
                    pics = {"depth": "depth", "pose": "pose", "canny": "lines"}
                    cj = [{"image": str(g / f"{cam}_{pics[c['type']]}.png"), "type": c["type"],
                           "scale": c.get("scale", 0.8), "end": c.get("end", 1.0), "prepared": True}
                          for c in cl if (g / f"{cam}_{pics[c['type']]}.png").exists()]
                    if cj and not free_:
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
