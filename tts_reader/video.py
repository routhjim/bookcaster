"""Picture video for a recorded chapter: generated illustrations that change
with the story, a slow pan/zoom on each, and subtitles, over the audiobook.

1. plan    - an LLM reads the chapter (paragraph-numbered), fixes how the
             recurring characters and places look, and picks the moments to
             illustrate (about one a minute), each with a self-contained
             image prompt in the book's art style.
2. place   - each picture starts with its paragraph, timed from the render's
             <name>.timeline.json (same matching as the music score).
3. draw    - tools/illustrate.py (FLUX.2 klein, in the music model's venv,
             on the music GPU) draws them; finished pictures are cached.
4. build   - ffmpeg renders one clip per picture (Ken Burns move, fades,
             burned-in subtitles) in parallel, joins them frame-exactly and
             lays the chapter's audio under them: <out>.mp4, 1080p.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .score import _chat, audio_seconds, paragraph_times, paragraphs

FPS = 24
W, H = 1920, 1080


def _rg():
    """tools/registry.py: the one owner of who and where (people, looks, places, style)."""
    import sys as _s
    t = str(Path(__file__).resolve().parents[1] / "tools")
    if t not in _s.path:
        _s.path.insert(0, t)
    import registry
    return registry


def registry_at(path):
    """The registry of the book holding *path* (a refs dir, a chapter dir ...), or None outside a book."""
    try:
        return _rg().at(path) if path else None
    except (OSError, ValueError):
        return None


def period_of(path) -> str:
    """The book's time period (<book>/period.txt, `bookcaster period`; the 3D plan writes it from the text when
    missing). Any path inside the book works (the book dir, work/video, a chapter dir). '' = not set."""
    from pathlib import Path as _P
    p = _P(path).expanduser().resolve()
    for d in (p, *p.parents):
        f = d / "period.txt"
        if f.exists():
            return f.read_text(encoding="utf-8").strip()
        if (d / "work").is_dir() and (d / "chapters").is_dir():
            break
    return ""


def period_line(path_or_period: str = "", period: str | None = None) -> str:
    """The sentence every drawing prompt carries (user 10-21: a modern claw hammer in a forge 1000 years ago):
    the period, and everything as made then. '' when no period is set."""
    per = period if period is not None else period_of(path_or_period)
    if not per:
        return ""
    return (f" Set in {per}: every object, tool, garment, building and material exactly as made in that time, "
            f"hand-made, nothing modern.")


def period_question(per: str) -> str:
    """The vision judge's period check (yes = in period)."""
    return (f"Answer yes or no: could everything in this picture (objects, tools, clothes, buildings, materials) "
            f"have existed in {per}? Answer no if anything looks modern or machine-made.")


DEFAULT_STYLE = ("detailed 19th-century book illustration, pen and ink linework with a soft "
                 "watercolour wash, muted period colours")
ART_PYTHON = Path(os.environ.get("BOOKCASTER_ART_PYTHON",
                                 Path.home() / "tts-ab" / ".venv-ace" / "bin" / "python"))
def hip_index(bus: str) -> str:
    """HIP_VISIBLE_DEVICES index of the GPU on PCI bus *bus* ("65"). HIP numbers
    GPUs in KFD topology order, which moves with how the docks come up at boot
    (a reboot turned the music XTX from 2 into 1), so never hard-code it."""
    nodes = Path("/sys/class/kfd/kfd/topology/nodes")
    i = 0
    for n in sorted(nodes.iterdir(), key=lambda d: int(d.name)) if nodes.exists() else []:
        props = dict(line.split() for line in (n / "properties").read_text().splitlines() if line.strip())
        if props.get("simd_count", "0") == "0":
            continue
        if f"{int(props.get('location_id', 0)) >> 8:02x}" == bus.lower():
            return str(i)
        i += 1
    return "2"    # the reference box's old index, if the topology can't be read


# HIP index of the music XTX (the drawing model shares it with ACE)
ART_BUS = os.environ.get("BOOKCASTER_ART_BUS", "65")
ART_GPU = os.environ.get("BOOKCASTER_ART_GPU") or hip_index(ART_BUS)
ILLUSTRATE = Path(__file__).resolve().parents[1] / "tools" / "illustrate.py"
# Which picture model draws: "klein" (FLUX.2 klein 4B, tools/illustrate.py, Apache) or "qwen"
# (Qwen-Image-2.1 + Lightning + ControlNet Union, tools/qwen_image.py, non-commercial licence, its own
# venv). Both speak the same jobs / --serve protocol; a job may carry "control" (qwen only, klein ignores it).
# default qwen since 10-19 (its ControlNet poses + reference edits drew every knight sample)
DRAWER = os.environ.get("BOOKCASTER_DRAWER", "qwen").strip().lower()
DRAW_SCRIPT = (ILLUSTRATE.with_name("qwen_image.py") if DRAWER == "qwen" else ILLUSTRATE)
DRAW_PYTHON = Path(os.environ.get("BOOKCASTER_DRAW_PYTHON") or (
    Path.home() / "tts-ab" / ".venv-qwenimage" / "bin" / "python" if DRAWER == "qwen" else ART_PYTHON))
# Fedora's ffmpeg has no libx264 and its VAAPI drivers have no H.264 encoder;
# OpenH264 (CPU) plays everywhere. Mostly-still pictures need little bitrate.
VIDEO_CODEC = os.environ.get("BOOKCASTER_VIDEO_CODEC", "libopenh264").split()
VIDEO_BITRATE = os.environ.get("BOOKCASTER_VIDEO_BITRATE", "2500k")


# --- 1. plan ---------------------------------------------------------------------

PLAN_VERSION = 4   # bump when the plan's shape changes: older plans are redone


def plan_pictures(paras: list[str], llm_url: str, model: str, style: str, count: int) -> dict:
    numbered = "\n\n".join(f"[P{i}] {p}" for i, p in enumerate(paras))
    prompt = (
        "You are the illustrator of an illustrated audiobook video. Below is one chapter, "
        "paragraph-numbered. Plan the pictures shown while it is read aloud.\n\n"
        "First fix the LOOK of every recurring character and place. A reference picture is "
        "drawn from each description and reused in every scene, so write it for an image "
        "model that takes words literally:\n"
        "- A person: START with sex and apparent age as a number (\"a woman of about 30\"), "
        "then build, face, hair, and clothing true to the story's period and their station, "
        "named the way a costume designer would (\"a Victorian lady's black mourning dress "
        "with a fitted bodice and a black bonnet\"). Never use words that suggest a different "
        "role or era (habit, robe, wimple, cloak-and-hood) and don't make anyone look older "
        "than their age. Complexion in natural words (pale, sallow, ruddy, weathered), never "
        "a colour name like yellow or green.\n"
        "- A place: its key features, as an empty establishing view.\n\n"
        f"Then choose about {count} moments to illustrate, spread through the whole chapter "
        "(the first at P0 or close to it), each starting at the paragraph where it begins. "
        "Pick what a listener is picturing at that point: a face, an arrival, a room, a "
        "landscape, a dramatic action. Vary the shots (wide, medium, close-up). Prefer one or "
        "two named people per picture; never more than three named.\n"
        "When the story puts them among a group (the disciples, a congregation, a crowd, a household), "
        "show the group, above all the first time we see it: name the main people in \"who\" and "
        "describe the others in \"crowd\" (how many, how dressed, where: \"about twenty men and women "
        "in plain robes seated on the floor around him\"), and put them in the prompt too.\n"
        "Only people physically present at that moment: someone the narrator addresses, or who is only "
        "spoken about, is not in the picture. (An event the narration tells of from another time can be "
        "its own picture, a flashback: then its people are present.) Never show a death, a wound, blood "
        "or a body: show the moment before, or the empty place after.\n"
        "The image model draws extra hands, arms and fingers whenever hands are the subject "
        "or people touch, so: no close-ups of hands or objects held in hands, no handshakes, "
        "embraces, or one person gripping another; show a held object at the side or on a "
        "table; let faces, posture and the setting tell the moment, with hands relaxed or "
        "out of frame.\n"
        "For each give:\n"
        "- start: paragraph number (increasing)\n"
        "- who: the LOOK names of the characters visible in it (each exactly once; [] for none). "
        "Leave out drivers, servants and bystanders the story doesn't dwell on, or the image model "
        "adds strangers\n"
        "- crowd: the unnamed group around them, as above, or \"\" for none\n"
        "- where: the LOOK name of the place, or \"\" if it isn't one of them\n"
        "- prompt: 30-70 words: what happens, each person's pose and expression (by name), "
        "composition and camera distance, lighting and mood. Don't describe the characters' "
        "or places' looks (the references carry them). No written words, captions or "
        "signatures.\n\n"
        "Each PLACE is one side of a location, inside OR outside, never both: its description starts with "
        "\"Inside:\" (a room, walls and ceiling around) or \"Outside:\" (seen from outdoors). A location the "
        "story uses both ways is two places, e.g. \"The Mill (inside)\" and \"The Mill (outside)\", and the "
        "outside one describes the building so it can be recognised from anywhere around it. A place "
        "description names no movable objects (tools, weapons, furniture people use): those are drawn separately.\n"
        'Reply with ONLY JSON: {"look": {"characters": {"Name": "description"}, '
        '"places": {"Name": "description"}}, '
        '"pictures": [{"start": 0, "who": ["Name"], "where": "Name", "crowd": "", "prompt": "..."}]}\n\n'
        f"Chapter:\n{numbered}"
    )
    reply = _chat(llm_url, model, prompt, max_tokens=12000)
    m = re.search(r"\{.*\}", reply, re.S)
    data = json.loads(m.group(0)) if m else {}
    look = data.get("look", {}) or {}
    chars = look.get("characters") or {}
    places = look.get("places") or {}
    pics, last = [], -1
    for p in data.get("pictures", []):
        try:
            s = int(p["start"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(p.get("prompt", "")).strip()
        if s <= last or s >= len(paras) or not text:
            continue
        # Anyone the action names is in the picture too: a cast list that left
        # out a named character made the model add an extra, unreferenced figure.
        named = [w for w in (p.get("who") or []) if w in chars] + mentions(text, chars)
        who = present(list(dict.fromkeys(named)), text, chars)[:3]
        where = p.get("where") if p.get("where") in places else ""
        pics.append({"start": s, "who": who, "where": where, "prompt": text,
                     "crowd": str(p.get("crowd") or "").strip()})
        last = s
    if not pics:
        raise ValueError("the planner returned no usable pictures")
    pics[0]["start"] = 0          # a picture from the first second
    chars = check_looks(chars, llm_url, model)
    return {"version": PLAN_VERSION, "look": {"characters": chars, "places": places},
            "pictures": pics}


SCENE_SIDE_RX = re.compile(r"^\s*(inside|outside)\s*:", re.I)


def scene_side(name: str, desc: str) -> str:
    """'inside' / 'outside' / '' of a place, from its name's variant or its description's prefix."""
    m = re.search(r"\((inside|outside)\)\s*$", name or "", re.I)
    if m:
        return m.group(1).lower()
    m = SCENE_SIDE_RX.match(desc or "")
    return m.group(1).lower() if m else ""


def plan_scenes(paras: list[str], look: dict, llm_url: str, model: str) -> dict:
    """The chapter's SCENES from the text, once, before any picture or moment is placed (user 10-22: the forge
    work drifted outside, the throne room only began at "in the corner of the hall", the King was never shown:
    each picture and moment picked its place on its own). A scene = continuous action in ONE place, ONE side
    (inside or outside); everyone physically present (the silent ones too); the props used there.
    -> {"scenes": [{"id", "from", "where", "side", "present", "props", "first_words", "why"}], "new_places": {}}"""
    places = look.get("places") or {}
    chars = look.get("characters") or {}
    numbered = "\n\n".join(f"[P{i}] {p}" for i, p in enumerate(paras))
    plist = "\n".join(f"- {n}: {d[:160]}" for n, d in places.items())
    prompt = (
        "You are the script supervisor of a film of this chapter. Split it into SCENES.\n"
        "A scene is continuous action in ONE place, seen from ONE side: inside it or outside it. A new scene starts "
        "ONLY when the action moves to another place, goes from inside to outside (or back), or jumps in time "
        "(a flashback, 'the next morning'). Narration that describes the place, or a line that mentions another "
        "place, while the people stay where they are, does NOT start a scene. Work done at a forge, a bench or a "
        "hearth is inside unless the text says it is done outdoors. The first scene starts at P0.\n"
        "For each scene give:\n"
        "- from: the number of its first paragraph (increasing)\n"
        "- where: EXACTLY one of the place names below. One location has ONE name: if two names below are the "
        "same room (a king's hall and a throne room), always use the same one. If no name fits, invent a short "
        "name and describe it in new_places\n"
        "- side: \"inside\" or \"outside\"\n"
        "- present: the names (from the characters below) of EVERYONE physically there during the scene, also "
        "those who never speak (a king on his throne listening, a guard at the door). Not people only spoken of\n"
        "- props: the objects people handle or use there (a hammer, bellows, a lantern), [] for none\n"
        "- first_words: the first 6-10 words of its first paragraph, copied exactly\n"
        "- why: 3-8 words, what changes here\n\n"
        f"Places:\n{plist}\n\nCharacters: {', '.join(chars)}\n\n"
        'Reply with ONLY JSON: {"scenes": [{"from": 0, "where": "Name", "side": "inside", "present": ["Name"], '
        '"props": [], "first_words": "...", "why": "..."}], "new_places": {"Name": "Inside: ..."}}\n\n'
        f"Chapter:\n{numbered}")
    reply = _chat(llm_url, model, prompt, max_tokens=6000)
    m = re.search(r"\{.*\}", reply, re.S)
    data = json.loads(m.group(0)) if m else {}
    return clean_scenes(data, paras, look)


def clean_scenes(data: dict, paras: list[str], look: dict) -> dict:
    """The model's scene list made usable: increasing starts from P0, known places (new ones kept with an
    Inside:/Outside: description), the side matched to the place's own side (its "(inside)"/"(outside)"
    variant when the book has one), present = known characters."""
    places = dict(look.get("places") or {})
    chars = look.get("characters") or {}
    new = {n: d for n, d in (data.get("new_places") or {}).items() if n and n not in places and SCENE_SIDE_RX.match(d or "")}
    places.update(new)
    out, last = [], -1
    for sc in data.get("scenes") or []:
        try:
            s = int(sc.get("from"))
        except (TypeError, ValueError):
            continue
        if s <= last or s >= len(paras):
            continue
        w = sc.get("where") or ""
        if w not in places and re.sub(r"\s*\((inside|outside)\)\s*$", "", w) in places:
            places.setdefault(w, "")                  # a side variant of a known place
        if w not in places:
            hit = [n for n in places if n.lower() in w.lower() or w.lower() in n.lower()]
            w = hit[0] if len(hit) == 1 else ""
        side = str(sc.get("side") or "").lower()
        side = side if side in ("inside", "outside") else scene_side(w, places.get(w, ""))
        if w and scene_side(w, places.get(w, "")) not in ("", side):
            base = re.sub(r"\s*\((inside|outside)\)\s*$", "", w)
            alt = next((n for n in places if re.sub(r"\s*\((inside|outside)\)\s*$", "", n) == base
                        and scene_side(n, places[n]) == side), None)
            # the other side of a known place: its "(inside)"/"(outside)" variant (the set builder draws a variant
            # from the base place's words, one side only)
            w = alt or f"{base} ({side})"
            places.setdefault(w, "")
        out.append({"id": f"s{len(out) + 1:02d}", "from": s, "where": w, "side": side,
                    "present": [n for n in dict.fromkeys(sc.get("present") or []) if n in chars],
                    "props": [str(x) for x in sc.get("props") or []][:8],
                    "first_words": str(sc.get("first_words") or paras[s][:60]),
                    "why": str(sc.get("why") or "")})
        last = s
    if not out:
        raise ValueError("the scene planner returned no usable scenes")
    out[0]["from"] = 0
    for a, b in zip(out, out[1:] + [None]):
        a["to"] = (b["from"] - 1) if b else len(paras) - 1
    return {"scenes": out, "new_places": new}


def merge_scenes(scenes: list[dict]) -> list[dict]:
    """Consecutive scenes in the same place and side are ONE scene (10-22: a split and its return landing on the same
    start left s12/s13 "The Barren Lands" twice); the first keeps its start, the present lists join."""
    out = []
    for sc in scenes:
        if out and out[-1].get("where") == sc.get("where") and out[-1].get("side") == sc.get("side"):
            out[-1]["present"] = list(dict.fromkeys((out[-1].get("present") or []) + (sc.get("present") or [])))
            out[-1]["props"] = list(dict.fromkeys((out[-1].get("props") or []) + (sc.get("props") or [])))
            continue
        out.append(sc)
    for i, sc in enumerate(out):
        sc["id"] = f"s{i + 1:02d}"
    return out


def scene_at(scenes: list[dict], para: int) -> dict | None:
    """The scene a paragraph is in."""
    cur = None
    for sc in scenes:
        if sc["from"] <= para:
            cur = sc
    return cur


def scenes_for(plan: dict, text: str, llm_url: str, llm_model: str, log=print, known: dict | None = None) -> list[dict]:
    """The plan's scenes (planned once, kept in plan.json), every picture's `where` set to its scene's place.
    known: more place names the book already has ({name: description}: built sets, side variants)."""
    paras = paragraphs(text)
    if not plan.get("scenes"):
        look = dict(plan.get("look") or {})
        # a BUILT set's side (its world kind) wins over the look's words (10-22: "The River Bank" was built inside)
        look["places"] = {**(known or {}), **(look.get("places") or {}), **{k: v for k, v in (known or {}).items() if v}}
        res = plan_scenes(paras, look, llm_url, llm_model)
        plan["scenes"] = res["scenes"]
        if res["new_places"]:
            plan.setdefault("look", {}).setdefault("places", {}).update(res["new_places"])
            log(f"  scenes: new place(s) {sorted(res['new_places'])}")
    places = {**(known or {}), **((plan.get("look") or {}).get("places") or {}), **{k: v for k, v in (known or {}).items() if v}}
    for sc in plan["scenes"]:                       # a planned scene whose place is built on the other side: the
        w = sc.get("where") or ""                    # place's set on the scene's side (re-checked every plan run)
        if w and sc.get("side") in ("inside", "outside") and scene_side(w, places.get(w, "")) not in ("", sc["side"]):
            base = re.sub(r"\s*\((inside|outside)\)\s*$", "", w)
            alt = next((n for n in places if re.sub(r"\s*\((inside|outside)\)\s*$", "", n) == base
                        and scene_side(n, places[n]) == sc["side"]), None) or f"{base} ({sc['side']})"
            log(f"  scene {sc.get('id')}: {w!r} is built {scene_side(w, places.get(w, ''))}: -> {alt!r}")
            sc["where"] = alt
    for p in plan.get("pictures") or []:
        sc = scene_at(plan["scenes"], int(p.get("start", 0)))
        if sc and sc["where"] and p.get("where") != sc["where"]:
            p["where_text"], p["where"] = p.get("where"), sc["where"]
    return plan["scenes"]


CREATURE_WORDS = ("dragon", "wyvern", "drake", "griffin", "gryphon", "serpent", "wolf", "bear", "horse", "troll",
                  "giant", "eagle", "lion", "beast", "monster", "unicorn", "basilisk", "hydra")


def creature_kinds(chars: dict) -> dict:
    """{creature cast name: its kind word} from the looks ("Vorgrath": "a massive black dragon ..." -> "dragon");
    people (a man, a woman ...) are not creatures."""
    out = {}
    for n, d in (chars or {}).items():
        head = " ".join(re.findall(r"[a-z]+", (d or "").lower())[:14])
        if re.search(r"\b(man|woman|boy|girl|knight|king|queen|princess|prince|smith|person|child|lady|lord)\b", head):
            continue
        k = next((w for w in CREATURE_WORDS if re.search(rf"\b{w}s?\b", head)), None)
        if k:
            out[n] = k
    return out


PLACE_GENERIC = {"inside", "outside", "the", "edge", "behind", "below", "above", "north", "south", "east", "west",
                 "near", "with", "from", "into", "old", "great"}


def place_words(name: str | None) -> set:
    """The distinctive words of a place name ("Outside Hulda's Forge (outside)" -> {hulda, forge})."""
    return {w for w in re.findall(r"[a-z]{4,}", (name or "").lower()) if w not in PLACE_GENERIC}


def place_phrase(name: str) -> str:
    """A place NAMED in a prompt, as a phrase (10-22: "coiled like a river of iron" put the dragon's cave in The River):
    a one-word name needs its article ("the river"), a longer one its words in order ("barley fields")."""
    base = re.sub(r"\s*\((inside|outside)\)\s*$", "", name or "").lower().strip()
    rest = re.sub(r"^(the|a|an)\s+", "", base)
    words = re.findall(r"[a-z']+", rest)
    if len(words) <= 1:
        return rf"\bthe\s+{re.escape(rest)}\b"
    return r"\b" + r"\W+".join(re.escape(w) for w in words) + r"\b"


def scene_starts(plan: dict, text: str, timeline: list[dict]) -> list[tuple[float, dict]]:
    """[(start seconds, scene)] of the plan's scenes in the recording."""
    sc = plan.get("scenes") or []
    if not sc:
        return []
    st = paragraph_times(paragraphs(text), timeline)
    # a scene the plan review split off INSIDE a paragraph starts at its first moment (t_split)
    return [(0.0 if i == 0 else max(st[s["from"]], float(s.get("t_split") or 0.0)), s) for i, s in enumerate(sc)]


def scene_at_t(starts: list, t: float) -> dict | None:
    cur = None
    for t0, s in starts:
        if t0 <= t + 0.05:
            cur = s
    return cur


def apply_scene(f: dict, sc: dict, chars: dict, places: dict | None = None) -> list[str]:
    """One moment's frame made to agree with its scene: the scene's place, the scene id, who is present; a cast
    creature its text names ("a massive black dragon ...") is a figure, not a prop (10-22: the opening's dragon
    was planned as the prop "Dragon Silhouette" and never staged). -> what changed (for the log)."""
    ch = []
    own = f.get("where_text") if f.get("scene") else f.get("where")       # the moment's own place (first planned)
    txt_ = " ".join([f.get("prompt") or "", f.get("motion") or ""]).lower()
    # a known place the PROMPT names (other than the scene's) is the moment's own place (10-22: the dragon over "the
    # Barley Fields" during the throne-room speech: a cutaway, never the hall)
    named = [n for n in (places or {}) if n != sc.get("where") and place_words(n)
             and re.search(place_phrase(n), txt_)
             and not (place_words(n) <= place_words(sc.get("where")))]
    if named:
        own = max(named, key=lambda n: len(place_words(n)))
    ow, sw = place_words(own), place_words(sc.get("where"))
    # the moment's OWN text names its own place and not the scene's: a disagreement for the plan review to
    # settle (a missed scene change: 10-22 Hulda's closing forge shots inside "The King's Hall"), never overwritten
    same = bool(own and sc.get("where") and (own.startswith(sc["where"]) or sc["where"].startswith(own.split(":")[0])))
    if own and sc.get("where") and own != sc["where"] and not same and re.search(place_phrase(own), txt_) \
            and not re.search(place_phrase(sc["where"]), txt_):
        f["scene_conflict"] = {"scene": sc["id"], "scene_where": sc["where"], "own_where": own}
        f["where"] = own
        f["where_text"] = own
        ch.append(f"CONFLICT: its text names {own!r}, the scene is {sc['where']!r} (kept; the plan review decides)")
    elif sc.get("where") and f.get("where") != sc["where"]:
        f.pop("scene_conflict", None)
        ch.append(f"where {f.get('where')!r} -> {sc['where']!r}")
        f["where_text"], f["where"] = own, sc["where"]
    else:
        f.pop("scene_conflict", None)
    f["scene"] = sc["id"]
    f["present"] = list(sc.get("present") or [])
    txt = " ".join([f.get("prompt") or "", f.get("motion") or ""]).lower()     # (not "must": it can be stale)
    for n, k in creature_kinds(chars).items():
        named = re.search(rf"\b{k}s?\b", txt) or any(re.search(rf"\b{k}", str(p).lower()) for p in f.get("props") or [])
        if named and n not in (f.get("who") or []):
            f["who"] = list(f.get("who") or []) + [n]
            ch.append(f"+{n} (the text's {k})")
        if n in (f.get("who") or []):
            keep = [p for p in f.get("props") or [] if not re.search(rf"\b{k}", str(p).lower())]
            if len(keep) != len(f.get("props") or []):
                ch.append(f"{k} prop(s) dropped: {n} is the figure")
                f["props"] = keep
    return ch


# Words an image model takes the wrong way in a character's reference
# portrait ("yellowed skin" drew a yellow man).
# 10-27: the veil/hood/wimple strip is gone: it was one book's band-aid (a veiled Helen drawn as a nun) applied to
# every book, and it fought the wardrobe, which is now the one owner of what is worn when (a veil, a hood).
LITERAL_FIXES = [
    (r"\byellow(?:ed|ish)?(\s+(?:and\s+\w+\s+)?(?:skin|complexion|face))", r"sallow\1"),
    (r"\b(?:green|grey|gray)(?:ish)?(\s+(?:skin|complexion))", r"pale\1"),
]


# Words in a scene that the image model reads in the modern sense: a "trap"
# (a pony cart) came out as a motor car. The era comes from the book's period line, never from here (10-27: "19th
# century" was written into every book's carriages, and the scene veils were stripped, as above).
SCENE_FIXES = [
    (r"\b(?<!horse-drawn )(trap|dog-cart|gig|brougham|landau|hansom(?: cab)?|carriage)\b",
     r"horse-drawn \1 (pulled by a horse)"),
    # a close-up of a wrist or hand drew it with no arm, or a second hand
    (r"\b(?:extreme\s+)?close-?up\s+(?:on|of)\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*)?)'s\s+"
     r"(?:\w+\s+)?(hands?|wrists?|arms?|fingers)\b", r"Medium shot of \1, showing \1's \2"),
]


def literal_safe(desc: str) -> str:
    out = str(desc)
    for pat, rep in LITERAL_FIXES:
        out = re.sub(pat, rep, out, flags=re.I)
    return re.sub(r"\s{2,}", " ", out).strip(" ,;")


def check_looks(chars: dict, llm_url: str, model: str) -> dict:
    """A second, narrow pass over the character descriptions only: the big
    planning prompt's rules about literal-minded image models were ignored
    (a veiled, greying Helen Stoner came out as a nun twice). Then a hard
    filter for the known traps."""
    if not chars:
        return chars
    prompt = (
        "These descriptions are used to draw each character's reference portrait for "
        "an illustrated story; the image model takes every word literally. Rewrite each "
        "(same names) so the portrait shows the person's EVERYDAY look:\n"
        "- begin EXACTLY like 'a man of about 40, clean-shaven, black hair swept back,' : "
        "sex and age as a number, then for a man his facial hair (clean-shaven, or the "
        "moustache/beard and its colour), then hair colour and style, each followed by a comma; then build, "
        "face and eyes; keep what the story says\n"
        "- face uncovered: no veils, hoods, masks; hats and bonnets are fine\n"
        "- grey or white hair only for someone over 50; for a younger person who is "
        "'prematurely grey', say 'a few grey streaks in dark hair'\n"
        "- skin in natural terms (pale, sallow, ruddy, weathered), never a colour name\n"
        "- clothing named precisely for the period and their station (e.g. 'a Victorian "
        "lady's dark day dress with a fitted bodice and a small bonnet'); mourning black is "
        "fine but never anything that reads as religious dress\n"
        "- one sentence, 30-50 words\n\n"
        'Reply with ONLY JSON {"Name": "description"}.\n\n'
        + json.dumps(chars, ensure_ascii=False, indent=1))
    try:
        reply = _chat(llm_url, model, prompt, max_tokens=3000)
        m = re.search(r"\{.*\}", reply, re.S)
        fixed = json.loads(m.group(0)) if m else {}
    except (OSError, ValueError):
        fixed = {}
    out = {k: literal_safe(fixed.get(k) or v) for k, v in chars.items()}
    # the "facial hair" slot is for men: "a woman of about 30, clean-shaven" reads oddly
    return {k: re.sub(r"^((?:a|an)\s+(?:woman|girl|lady)\b[^,]*),\s*clean-shaven,", r"\1,", v, flags=re.I)
            for k, v in out.items()}


def style_guard(work: Path, style: str, log=print) -> None:
    """Plan and pictures were made for one art style; a new style redoes both."""
    work.mkdir(parents=True, exist_ok=True)
    f = work / "style.txt"
    old = f.read_text(encoding="utf-8").strip() if f.exists() else None
    if old is not None and old != style.strip():
        for p in [work / "plan.json", *(work / "img").glob("*.png")]:
            p.unlink(missing_ok=True)
        log(f"  art style changed ({old} -> {style}): redrawing")
    f.write_text(style.strip(), encoding="utf-8")


def ensure_plan(text: str, work: Path, llm_url: str, llm_model: str, style: str,
                total: float, per_minute: float = 1.0, log=print) -> dict:
    """The picture plan (cached in work/plan.json). Needs no recording: *total*
    may be an estimate, so it can run before/while the voices render."""
    style_guard(work, style, log)
    plan_path = work / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()).get("version") != PLAN_VERSION:
        plan_path.unlink()                     # an older kind of plan: plan again
        for f in (work / "img").glob("*.png"):
            f.unlink()
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        for n, p in enumerate(plan["pictures"]):
            p.setdefault("id", f"p{n + 1:03d}")
            p.setdefault("seed", 1000 + n)
            p.setdefault("full_prompt", full_prompt(p["prompt"], plan.get("look", {}), style))
        before = {p["id"]: p["full_prompt"] for p in plan["pictures"]}
        refresh_looks(plan, work.parent / "look.json", llm_url, llm_model, style, log)
        if ensure_wardrobe(plan, text, llm_url, llm_model, log):
            plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
        changed = []
        for p in plan["pictures"]:       # the prompt rules may have changed since: redraw what differs
            if "who" in p:
                p["who"] = present(p["who"], p["prompt"], plan["look"].get("characters", {}))
                p["full_prompt"] = full_prompt(p["prompt"], plan["look"], style, p["who"], p.get("where", ""),
                                               p.get("crowd", ""), scene_dress(plan, p))
            if p["full_prompt"] != before[p["id"]]:
                changed.append(p["id"])
                (work / "img" / f"{p['id']}.png").unlink(missing_ok=True)
        if changed:
            log(f"  {len(changed)} picture prompt(s) changed: redrawing them")
            plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
        log(f"  picture plan restored ({len(plan['pictures'])} picture(s))")
        return plan
    count = max(3, min(90, round(total / 60 * per_minute)))
    log(f"  planning ~{count} pictures...")
    plan = plan_pictures(paragraphs(text), llm_url, llm_model, style, count)
    keep_looks(plan, work.parent / "look.json")
    refresh_looks(plan, work.parent / "look.json", llm_url, llm_model, style, log)
    ensure_wardrobe(plan, text, llm_url, llm_model, log)
    for n, p in enumerate(plan["pictures"]):
        p.update(id=f"p{n + 1:03d}", seed=1000 + n,
                 full_prompt=full_prompt(p["prompt"], plan.get("look", {}), style,
                                         p.get("who"), p.get("where", ""), p.get("crowd", ""),
                                         scene_dress(plan, p)))
    plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    log(f"  planned {len(plan['pictures'])} picture(s)")
    return plan


LOOK_VERSION = 2   # bump when check_looks' rules change: stored looks are redone once


TITLES = {"doctor": "dr", "mister": "mr", "missus": "mrs", "saint": "st", "professor": "prof",
          "captain": "capt", "colonel": "col"}


def canon(name: str) -> str:
    """One spelling per person: 'Doctor Grimesby Roylott' and 'Dr. Grimesby
    Roylott' were stored as two characters with different looks (one bearded)."""
    words = re.findall(r"[a-z0-9']+", name.lower())
    return " ".join(TITLES.get(w, w) for w in words)


def registry_entry(reg: dict, kind: str, name: str, R=None) -> str | None:
    """The stored look for *name* under any spelling; the first stored wins. With the book's registry *R* a person
    is matched as the registry matches them (a unique surname, a voice label: 10-27) and a place by its place key
    (without "(inside)"/"(outside)" or "The")."""
    if R is not None and kind == "places":        # the same place (and side): registry.place_slot
        k, known = _rg().place_slot(reg.get(kind, {}), name)
        return k if known else None
    c = canon(name)
    for k, v in reg.get(kind, {}).items():
        if canon(k) == c:
            return k
    if R is not None and kind == "characters":
        p = R.person(name)
        return next((k for k in reg.get(kind, {}) if p and R.person(k) == p), None)
    return None


def refresh_looks(plan: dict, registry: Path, llm_url: str, model: str, style: str,
                  log=print) -> bool:
    """Re-run check_looks over a book's stored characters if they were written
    under older rules, keeping the scenes; True if the plan changed."""
    reg = json.loads(registry.read_text()) if registry.exists() else None
    if reg is not None and reg.get("version") != LOOK_VERSION and reg.get("characters"):
        reg["characters"] = check_looks(reg["characters"], llm_url, model)
        reg["version"] = LOOK_VERSION
        registry.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
        log("  character looks rewritten (age, beard, hair spelled out)")
    if reg is None:
        return False
    R = registry_at(registry.parent)
    mine = plan.setdefault("look", {}).setdefault("characters", {})
    changed = False
    for name in mine:
        key = registry_entry(reg, "characters", name, R)
        if key is not None and mine[name] != reg["characters"][key]:
            mine[name] = reg["characters"][key]
            changed = True
    if changed:
        for p in plan["pictures"]:
            if "who" in p:
                p["full_prompt"] = full_prompt(p["prompt"], plan["look"], style,
                                               p["who"], p.get("where", ""), p.get("crowd", ""),
                                               scene_dress(plan, p))
    return changed


def keep_looks(plan: dict, registry: Path) -> None:
    """A character or place keeps the look it was first given in this book:
    a later chapter's plan reuses it (so its reference picture is reused too).
    10-27: matched as the registry matches (tools/registry.py); a new place is stored under its one name (no
    "(inside)"/"(outside)"), never as another spelling of a place already described."""
    reg = json.loads(registry.read_text()) if registry.exists() else {
        "version": LOOK_VERSION, "characters": {}, "places": {}}
    R = registry_at(registry.parent)
    for kind in ("characters", "places", "props"):       # props: keyframes' objects (the safe, the band)
        reg.setdefault(kind, {})
        mine = plan["look"].setdefault(kind, {})
        for name in list(mine):
            key = registry_entry(reg, kind, name, R)
            if key is not None:
                mine[name] = reg[kind][key]
            elif kind == "places" and R is not None:
                reg[kind][_rg().place_slot(reg[kind], name, mine[name])[0]] = mine[name]
            else:
                reg[kind][name] = mine[name]
    registry.write_text(json.dumps(reg, indent=1, ensure_ascii=False))


def timed(plan: dict, text: str, timeline: list[dict], total: float) -> list[dict]:
    """Each picture's start/end in the recording: its paragraph's start time;
    one shown under 4 s is dropped (the previous one stays up)."""
    starts = paragraph_times(paragraphs(text), timeline)
    pics = []
    for n, p in enumerate(plan["pictures"]):
        t0 = 0.0 if n == 0 else starts[p["start"]]
        if pics and t0 - pics[-1]["t0"] < 4:
            continue
        pics.append({**p, "t0": round(t0, 2)})
    for a, b in zip(pics, pics[1:] + [None]):
        a["t1"] = b["t0"] if b else round(total, 2)
    return pics


GENERIC = {"doctor", "dr", "mr", "mrs", "miss", "sir", "lady", "lord", "the", "room", "house",
           "manor", "street", "sitting", "bedroom", "hall", "hotel", "inn", "station", "old",
           "young", "little", "great", "village", "town", "farm", "church", "garden", "road"}


def _names(key: str) -> set[str]:
    """Ways a prompt may mention a LOOK entry: 'Doctor Grimesby Roylott' ->
    the full name, 'Grimesby', 'Roylott'; 'Stoke Moran Manor' -> 'Stoke Moran'."""
    words = re.findall(r"[A-Za-z][\w'-]*", key)
    core = list(words)
    while len(core) > 1 and core[-1].lower() in GENERIC:
        core.pop()
    out = {key.lower()}
    if not core[-1].endswith("'s"):
        out.add(" ".join(core).lower())
    for w in (core[0], core[-1]):
        if w.endswith("'s"):         # "Roylott's Bedroom" is a place, not Roylott
            continue
        if len(w) > 3 and w.lower() not in GENERIC and w[:1].isupper():
            out.add(w.lower())
    return out


PERSON = re.compile(r"\b(man|woman|girl|boy|lady|gentleman|child|he|she|his|her|aged?|years|"
                    r"wears|hair|eyes|face)\b", re.I)


def looks(plan: dict, R=None) -> tuple[dict, dict]:
    """(characters, places) from a plan; older plans had one flat dict. With the book's registry *R* the plan
    gives only the NAMES (the chapter's cast and places) and every look and place description is the registry's
    (10-27: plan copies disagreed with look.json - the climax drew a second, bearded Roylott); the characters
    then answer any spelling of a person (registry.Looks)."""
    look = plan.get("look", {}) or {}
    if "characters" in look or "places" in look:
        chars, places = dict(look.get("characters") or {}), dict(look.get("places") or {})
    else:
        chars = {k: v for k, v in look.items() if PERSON.search(str(v))}
        places = {k: v for k, v in look.items() if k not in chars}
    if R is None:
        return chars, places
    return R.looks(list(chars), chars), {n: R.place_look(n) or d for n, d in places.items()}


def mentions(prompt: str, look: dict) -> list[str]:
    """Entries of *look* the text names. A word two entries share (the
    Stoner sisters' surname) names neither: only 'Helen' or the full name
    counts, or every Helen scene would also cast Julia."""
    low = prompt.lower()
    names = {k: _names(k) for k in look}
    counts: dict[str, int] = {}
    for ns in names.values():
        for n in ns:
            counts[n] = counts.get(n, 0) + 1
    return [k for k, ns in names.items()
            if any(counts[n] == 1 and re.search(rf"\b{re.escape(n)}\b", low) for n in ns)]


# Words that make a name in a scene prompt someone talked about, not someone
# in the room: "Helen speaks of her sister Julia" drew Julia beside her.
TALKED_ABOUT = re.compile(
    r"\b(?:(?:speak|talk|think|tell|dream|grieve|worr(?:y|ie))\w*\s+(?:\w+\s+)?(?:of|about)"
    r"|recount\w*|describ\w*|mention\w*|remember\w*|recall\w*|mourn\w*|grief\s+for"
    r"|(?:memory|memories|thoughts?|death|fate|story|ghost|portrait|photograph|letter)\s+of)\b", re.I)


def talked_about(prompt: str, key: str, others=()) -> bool:
    """True when every mention of LOOK entry *key* in *prompt* comes after a
    talking/remembering word in the same clause. Name parts *others* share
    (the Stoner sisters' surname) don't count."""
    shared = set().union(*(_names(o) for o in others if o != key)) if others else set()
    hits = 0
    for clause in re.split(r"[.;:!?]", prompt):
        for n in _names(key) - shared:
            for m in re.finditer(rf"\b{re.escape(n)}\b", clause, re.I):
                hits += 1
                if not TALKED_ABOUT.search(clause[:m.start()]):
                    return False
    return hits > 0


def present(who: list[str], prompt: str, chars=()) -> list[str]:
    return [n for n in who if not talked_about(prompt, n, chars)]


def _short(desc: str) -> str:
    """'a man of about 40, clean-shaven, black hair swept back, lean ...' ->
    'a man of about 40, clean-shaven, black hair swept back': age, beard and
    hair, which a close-up otherwise invents afresh each time."""
    head = re.split(r"[;.]", str(desc), maxsplit=1)[0]
    return ",".join(head.split(",")[:3]).strip()


# A third hand turned up in a scene (10-16): said outright on every picture with people.
# Wording lowers the odds; the vision check (check_picture) is what catches the rest.
ANATOMY = "Each person has exactly two arms, two hands and two legs, natural anatomy"


def full_prompt(prompt: str, look: dict, style: str, who: list[str] | None = None,
                where: str = "", crowd: str = "", dress: dict | None = None) -> str:
    """The scene prompt for the image model. With references (who/where
    known), each person is tied to their reference image and said to appear
    once: writing out their full look as well made the model draw a second
    copy. Without, the looks are written in (older plans)."""
    chars = look.get("characters") if "characters" in look else look
    places = look.get("places") if "places" in look else {}
    body = prompt.replace(style, "").strip().rstrip(".,; ")
    for pat, rep in SCENE_FIXES:
        body = re.sub(pat, rep, body, count=1 if "horse" in rep else 0, flags=re.I)
    if who is None:
        merged = {**(chars or {}), **(places or {})}
        seen = [f"{k}: {str(v).strip().rstrip('.')}" for k, v in merged.items()
                if k in mentions(body, merged)]
        return ". ".join([body] + seen + [style]) + "."
    parts = [body]
    if who:
        cast = [f"{n} ({_short(chars.get(n, ''))}, as in image {i})" for i, n in enumerate(who, 1)]
        parts.append(crowd_line(who, cast, crowd) if crowd else
                     f"Exactly {len(who)} {'person' if len(who) == 1 else 'people'} in the "
                     f"picture, each appearing once: " + "; ".join(cast))
        parts.append(ANATOMY)
        for n in who:
            if (dress or {}).get(n):
                parts.append(f"{n} wears {dress[n]}, as in image {who.index(n) + 1}")
    elif crowd:
        parts.append(f"Unnamed people only: {crowd}")
    else:
        parts.append("No people in the picture")
    if where:
        parts.append(f"The setting is {where}, as in image {len(who) + 1}")
    parts.append(style)
    parts.append("No text, no lettering, no signature")
    return ". ".join(parts) + "."


def crowd_line(who: list[str], cast: list[str], crowd: str) -> str:
    """The cast among an unnamed group (the disciples in the upper room: Peter alone there read
    as Peter alone, 10-17). Each named person still once; the group gets no references."""
    return (f"In front: {'; '.join(cast)}, each appearing exactly once. Around and behind them: "
            f"{crowd.rstrip('. ')}, ordinary unnamed people who look nothing like {' or '.join(who)}")


# --- 3. draw ---

FLUSH = "/usr/local/sbin/gpu-pool-flush"
DRAW_NEED_GB = float(os.environ.get("BOOKCASTER_DRAW_NEED_GB", "4"))


def _avail_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1048576
    return 0.0


def memory_scope() -> list[str]:
    """Run the drawer in its own systemd scope capped BELOW the memory that is
    actually free, so if it overreaches the kernel kills it, not a server
    (on this box freed GPU pages sit in a pool the OOM killer doesn't wait
    for, and a global OOM took the Fish engine once). Flushes that pool
    first and refuses to start without room."""
    if _avail_gb() < DRAW_NEED_GB + 2 and os.path.exists(FLUSH):
        subprocess.run(["sudo", "-n", FLUSH], capture_output=True)
    avail = _avail_gb()
    if avail < DRAW_NEED_GB + 1:
        raise RuntimeError(f"not enough free memory to draw ({avail:.1f} GiB free, "
                           f"needs {DRAW_NEED_GB + 1:.0f})")
    cap = max(DRAW_NEED_GB, min(8.0, avail - 2.0))
    return ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={cap:.1f}G",
            "-p", "MemorySwapMax=0"]


# ------------------------------------------------------------------

def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:50] or "x"


def ref_path(refs_dir: Path, name: str, prompt: str) -> Path:
    """A reference's file is named by what drew it: a changed look or art
    style draws a new one instead of reusing a stale picture."""
    import hashlib
    return refs_dir / f"{slug(name)}-{hashlib.sha1(prompt.encode()).hexdigest()[:8]}.png"


INSIDE_RX = re.compile(r"(?i)^\s*inside\b|\((inside)\)\s*$")
OUTSIDE_RX = re.compile(r"(?i)^\s*outside\b|\((outside)\)\s*$")


def place_side(plan: dict, name: str, desc: str) -> str:
    """'Show the inside of X: ...' / 'Show the outside of X: ...' for a place picture (user 10-21: the forge's
    picture showed the building from outside though the story works inside it). The place's own name/description
    says it ("(inside)", "Inside:"); else its scenes' words vote (tools/world.py classify_kind). '' if no vote.
    NOTE: changes the prompt (and so the file name) of places that get a side: their pictures are redrawn."""
    if INSIDE_RX.search(name) or INSIDE_RX.search(desc or ""):
        kind = "interior"
    elif OUTSIDE_RX.search(name) or OUTSIDE_RX.search(desc or ""):
        kind = "exterior"
    else:
        import sys as _s
        _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
        import world as _W
        prompts = [p.get("prompt", "") for p in plan.get("pictures", []) if p.get("where") == name]
        kind, why = _W.classify_kind(desc or "", prompts)
        if not (why["pictures_interior"] or why["pictures_exterior"] or why["description_score"]):
            return ""
    base = re.sub(r"\s*\((inside|outside)\)\s*$", "", name)
    return (f"Show the INSIDE of {base}: an interior room, walls, floor and ceiling around, no sky. "
            if kind == "interior" else f"Show the OUTSIDE of {base}, seen from outdoors. ")


def place_desc(plan: dict, name: str, desc: str) -> str:
    """The description without the other side's clauses or story props (tools/world.py clean_description):
    "anvil outside" has no place in a picture of the forge's inside."""
    side = place_side(plan, name, desc)
    if not side:
        return str(desc)
    import sys as _s
    _s.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import world as _W
    kind = "interior" if "INSIDE" in side else "exterior"
    props = list(((plan.get("look") or {}).get("props") or {}).keys()) if isinstance(plan.get("look"), dict) else []
    return _W.clean_description(re.sub(r"(?i)^\s*(inside|outside)\s*:\s*", "", str(desc)), kind,
                                _W.prop_names({"props": {p: "" for p in props}}))


def sheet_prompt(desc: str, style: str) -> str:
    """A person's reference sheet words (its file is named by their hash: ref_path)."""
    # A full-length figure alone left the face ~20 px tall once shrunk
    # to a reference, so every close-up invented a new face.
    return (f"Character sheet of one person, two views side by side on a plain "
            f"light background: on the left a large head-and-shoulders portrait "
            f"facing the viewer, the face filling the frame; on the right the same "
            f"person full length, standing. {str(desc).rstrip('.')}. Even light. "
            f"{style}. No text, no lettering, no signature.")


def ref_prompts(plan: dict, style: str | None, R=None) -> dict[str, tuple[str, list[int]]]:
    """name -> (prompt, [w, h]) for every reference the plan's scenes use. With the book's registry *R* every
    look, place description and (when *style* is None) the style are the registry's (10-27: a sheet, its LoRA's
    captions and the keys from one look and one style); the names stay the plan's, so a sheet's file name changes
    only when its words do."""
    chars, places = looks(plan, R)
    if style is None:
        style = R.ref_style() if R is not None else DEFAULT_STYLE   # sheets: the reference style
    # the 3D flow builds a model + LoRA for the whole cast, so it lists them all (10-22 knight_forged_v3: the King
    # was in no still picture -> no sheet -> his LoRA failed with no reference face)
    used = set(plan.get("sheet_cast") or [])
    for p in plan["pictures"]:
        if "who" in p:
            used.update(p["who"] + ([p["where"]] if p.get("where") else []))
        else:
            used.update(mentions(p["prompt"], {**chars, **places}))
    if R is not None:                    # a sheet_cast / who spelling of the same person or place counts
        used |= {n for n in chars if any(R.same(n, u) for u in used)}
        used |= {n for n in places if any(_rg().place_key(n) == _rg().place_key(u) for u in used)}
    out = {}
    for name, desc in chars.items():
        if name in used:
            out[name] = (sheet_prompt(desc, style), [1280, 896])
    for name, desc in places.items():
        if name in used:
            out[name] = (f"Establishing view of {name}, no people. {place_side(plan, name, desc)}"
                         f"{place_desc(plan, name, desc).rstrip('.')}. "
                         f"{style}. No text, no lettering, no signature.", [1024, 576])
    return out


# --- the wardrobe ------------------------------------------------------------------
# A look is one description per person, but people change in a story: the knight forges his armor
# and fights in it, the princess's dress is torn and grey with ash, a lame man walks. Planned one scene
# at a time, the pictures forgot every change at the next scene and the reference sheet (the usual
# clothes) pulled the knight back into his stable leathers for the fight (10-18). So one pass over the
# whole chapter lists every lasting change and the paragraph it starts at; each change gets its own
# reference sheet, drawn from the person's sheet (same face); every picture, keyframe and portrait
# uses the outfit in force at its moment, and the checks look for it.

# 10-27 audit: the prompt asked only for "LASTING" changes and its examples were one story's (armor a smith forges),
# so a chapter that opens with Watson woken in bed had no nightclothes and he was drawn in bed in his tweed jacket
WARDROBE_PROMPT = """You keep the wardrobe for an illustrated audiobook. Below are the people with their
usual look, then the chapter in numbered paragraphs.

List every change to what a person WEARS, in story order, whenever the text states or plainly implies they are
dressed differently from their usual look: night clothes while they are in bed or have just woken, a coat, cloak or
hat put on to go out (and off again indoors), a disguise, clothes soaked, torn, stained or burnt, a garment given
away or picked up, a bandage. Getting dressed again is its own entry with their usual clothes.
POSTURE AND MOVEMENT ARE NOT CLOTHES: sitting, standing, lying, kneeling, walking or leaping, being healed or able to
walk, and where someone is are never a wardrobe entry and never part of a "dress" (the shots show them). Only what the
text states or plainly implies.
Something being made, bought or carried is not worn: it is worn from the first paragraph where the text has them put
it on or wear it.

For each change: "from" = the number of the first paragraph where it is already true, "id" = 2-4 words
naming the look ("night clothes", "travelling cloak"), "dress" = EVERYTHING the person wears from then on, described
so a painter can draw it with no other context (materials, colors, what covers what; a hat on or off).
Never describe the face, hair or age (those never change here). People who never change are left out.
Only things WORN: fire, light, a glow, a halo, a cloud or anything resting on or hovering over someone is a moment of
the story, not clothing, and is never a wardrobe entry; nor is being unseen or gone.

{notes}People:
{people}

Chapter:
{paras}

Reply with ONLY JSON: {{"<name>": [{{"from": <n>, "id": "<look>", "dress": "<everything worn>"}}]}}, or {{}}."""


NOT_WORN = re.compile(r"\b(fire|flames?|tongues|glow\w*|halo|aura|hover\w*|resting on|sparks?|no longer|"
                      r"vanish\w*|ascend\w*|"
                      # posture / movement / healing (10-26 Acts 3: the lame man's "standing upright with legs bearing
                      # weight" as his outfit from the begging paragraph on drew him standing before he was healed)
                      r"stand\w*|sit|sits|sitting|seated|lying|lies|kneel\w*|crouch\w*|walk\w*|leap\w*|jump\w*|"
                      r"upright|bearing weight|posture|heal\w*|lame|crippled)\b", re.I)


def worn_only(dress: str) -> str:
    """The parts of an outfit that are clothes (10-25: Acts 2 dressed Peter in "divided tongues as of fire resting
    on head" for the whole chapter, and Acts 1 dressed the ascended Jesus in "No longer visible")."""
    return ", ".join(p.strip() for p in dress.strip().rstrip(".").split(",") if p.strip() and not NOT_WORN.search(p))


def changes_nothing(dress: str, look: str) -> bool:
    """True when every word of an outfit is already in the person's usual look: no outfit, only a repeat of it
    (10-25: Acts 2 gave Peter "tunic and belt" = his usual tunic and belt, and an outfit costs a sheet + LoRA retrain)."""
    words = set(re.findall(r"[a-z]{4,}", dress.lower())) - {"with", "over", "under", "wearing", "worn"}
    return bool(words) and words <= set(re.findall(r"[a-z]{4,}", str(look).lower()))


def plan_wardrobe(paras: list[str], chars: dict, llm_url: str, model: str, narrator: str = "",
                   notes: str = "") -> dict | None:
    """{name: [{"from": paragraph, "id", "dress"}, ...]} (in story order), from one language-model
    pass over the whole chapter; {} when nobody changes, None when the server gave no usable answer
    (10-25: Acts 2's answer filtered down to {} read as "no answer", and the stale fire outfits stayed)."""
    if not chars:
        return {}
    # the narrator is "I" in the text, never named where they wake or dress (10-27: Watson got no night clothes)
    people = "\n".join(f"- {n}: {str(d).strip()}" + (" (the narrator: the \"I\" of the chapter is this person)"
                                                     if narrator and n == narrator else "") for n, d in chars.items())
    numbered = "\n".join(f"[{i}] {p}" for i, p in enumerate(paras))
    try:
        # the director's notes on how each scene opens are evidence of clothes (10-27: its breakdown had "Watson in a
        # nightshirt" for the waking scene while the wardrobe, reading "I", gave him none)
        # 10-28: a re-asked breakdown put Holmes "in a dressing-gown" the text never gives him (Watson: "fully
        # dressed"), and the wardrobe took it as evidence - the notes are hints, the chapter's words decide
        nt = (f"The film director's notes on how each scene opens (hints only; the chapter's own words decide - an "
              f"outfit the chapter does not describe or clearly imply is not an entry):\n{notes}\n\n"
              if notes else "")
        reply = _chat(llm_url, model, WARDROBE_PROMPT.format(notes=nt, people=people, paras=numbered), max_tokens=4000)
    except Exception:
        return None
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        data = json.loads(m.group(0)) if m else None
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    out = {}
    for name, changes in (data or {}).items():
        if name not in chars or not isinstance(changes, list):
            continue
        rows = []
        for c in changes:
            try:
                n = max(0, min(len(paras) - 1, int(c.get("from"))))
            except (TypeError, ValueError, AttributeError):
                continue
            dress = worn_only(str(c.get("dress") or ""))
            if dress:
                rows.append({"from": n, "id": str(c.get("id") or "changed").strip()[:40], "dress": dress})
        rows.sort(key=lambda r: r["from"])
        while rows and changes_nothing(rows[0]["dress"], chars[name]):
            rows.pop(0)                 # the usual clothes before any change (back to them later is a change)
        if rows:
            out[name] = rows
    return out


def wardrobe_inputs_key(plan: dict, text: str, narrator: str = "", notes: str = "") -> str:
    """What a chapter's wardrobe is planned from: its rules, the cast's looks and the chapter (10-27 audit: the key was
    the prompt alone, so a look change or a fixed cast never re-planned it)."""
    import hashlib
    chars, _ = looks(plan)
    return hashlib.sha1(json.dumps([WARDROBE_PROMPT, sorted(chars.items()), hashlib.sha1(text.encode()).hexdigest(),
                                    narrator, notes],
                                   sort_keys=True).encode()).hexdigest()[:8]


def ensure_wardrobe(plan: dict, text: str, llm_url: str, model: str, log=print, narrator: str = "",
                    notes: str = "") -> bool:
    """Plan the wardrobe into plan["wardrobe"] once (again when its rules change: 10-24, the armor Edric forges was
    worn from the forging paragraph on, in the forge); True if it was planned now. A re-plan keeps each outfit's
    words (the sheets, and any `bookcaster look` edit): only when each starts and ends changes."""
    k = wardrobe_inputs_key(plan, text, narrator, notes)
    if "wardrobe" in plan and plan.get("wardrobe_k", k if not plan["wardrobe"] else "") == k:
        return False
    old = {(n, r["id"].lower()): r["dress"] for n, rows in (plan.get("wardrobe") or {}).items() for r in rows}
    chars, _ = looks(plan)
    new = plan_wardrobe(paragraphs(text), chars, llm_url, model, narrator, notes)
    if new is None:
        log("  wardrobe: the language model gave no usable answer; the old outfits stay")
        return False
    for n, rows in new.items():
        for r in rows:
            r["dress"] = worn_only(old.get((n, r["id"].lower()), r["dress"])) or r["dress"]
    # an outfit set by hand (bookcaster set_outfit: "_pinned_outfits") stays, word for word, even if the re-plan
    # dropped it (10-27)
    olds = {(n, r["id"].lower()): r for n, rows in (plan.get("wardrobe") or {}).items() for r in rows}
    for n, oid in plan.get("_pinned_outfits") or []:
        r = olds.get((n, str(oid).lower()))
        if r is None:
            continue
        nk = wardrobe_key({"wardrobe": new}, n) or n
        rows = new.setdefault(nk, [])
        hit = next((x for x in rows if x["id"].lower() == r["id"].lower()), None)
        if hit is not None:
            hit["dress"] = r["dress"]
        else:
            rows.append(dict(r))
            rows.sort(key=lambda x: x["from"])
    plan["wardrobe"], plan["wardrobe_k"] = new, k
    for name, rows in plan["wardrobe"].items():
        log(f"  wardrobe: {name}: " + " -> ".join(f"{r['id']} (from paragraph {r['from']})" for r in rows))
    if not plan["wardrobe"]:
        log("  wardrobe: nobody changes")
    return True


def scene_dress(plan: dict, p: dict) -> dict:
    """{name: dress} for the people of scene picture *p* whose outfit isn't their usual one."""
    return {n: r["dress"] for n in p.get("who") or [] if (r := outfit(plan, n, p.get("start")))}


def wardrobe_key(plan: dict, name: str, R=None) -> str | None:
    """The plan's wardrobe key for *name* under any spelling ("Dr. Watson" = "Doctor Watson"; with the registry *R*
    also a unique surname or voice label), or None (10-27: rows were matched by the exact name only)."""
    w = plan.get("wardrobe") or {}
    if name in w:
        return name
    if R is not None:
        return R.key_in(name, list(w))
    pk = _rg().person_key(name)
    hits = [k for k in w if _rg().person_key(k) == pk]
    return hits[0] if len(hits) == 1 else None


def outfit(plan: dict, name: str, para: int | None, R=None) -> dict | None:
    """The wardrobe entry in force for *name* at paragraph *para*: None = their usual look."""
    if para is None:
        return None
    k = wardrobe_key(plan, name, R)
    rows = [r for r in (plan.get("wardrobe") or {}).get(k, []) if r["from"] <= para] if k else []
    return rows[-1] if rows else None


def para_at(starts: list[float], t: float) -> int:
    """The paragraph being read at *t* seconds (*starts*: each paragraph's start time)."""
    n = 0
    for i, s in enumerate(starts):
        if s <= t + 0.05:
            n = i
    return n


def outfit_prompt(name: str, desc: str, row: dict, style: str) -> str:
    return (f"Character sheet of one person, two views side by side on a plain light background: on the "
            f"left a large head-and-shoulders portrait facing the viewer, the face filling the frame; on the "
            f"right the same person full length, standing. The same person as in image 1: exactly the same "
            f"face, hair, age and build ({_short(desc)}), but now wearing {row['dress']}, in BOTH views (the portrait "
            f"shows these clothes too). Only the face and body come from image 1: everything worn or held is as "
            f"written here, not as in image 1; a hat or other headwear only if written here (10-27: the base "
            f"look's top hat on a dressing gown, its tweed jacket over a nightshirt). Even light. "
            f"{style}. No text, no lettering, no signature.")


def outfit_ref(refs_dir: Path, plan: dict, style: str, name: str, row: dict, R=None) -> Path:
    """The outfit sheet's file: named by the wardrobe's own spelling of the person, drawn from the registry's look
    (*R*, else the book's registry found from *refs_dir*)."""
    R = R if R is not None else registry_at(refs_dir)
    chars, _ = looks(plan, R)
    name = wardrobe_key(plan, name, R) or name
    return ref_path(refs_dir, f"{name} {row['id']}", outfit_prompt(name, chars.get(name, ""), row, style))


def outfit_jobs(plan: dict, style: str, refs_dir: Path) -> list[dict]:
    """One reference sheet per wardrobe change, drawn from the person's own sheet (image 1), so the
    face stays theirs. Listed after reference_jobs: a drawing batch runs in order."""
    R = registry_at(refs_dir)
    chars, _ = looks(plan, R)
    known = ref_prompts(plan, style, R)
    jobs = []
    for name, rows in (plan.get("wardrobe") or {}).items():
        if name not in chars:
            continue
        kn = name if name in known else next((k for k in known if R is not None and R.same(k, name)), None)
        base = [str(ref_path(refs_dir, kn, known[kn][0]))] if kn else []
        for row in rows:
            prompt = outfit_prompt(name, chars[name], row, style)
            jobs.append({"id": f"ref_{slug(name)}_{slug(row['id'])}", "size": [1280, 896], "seed": 7,
                         "prompt": prompt, "refs": base, "path": str(ref_path(refs_dir, f"{name} {row['id']}", prompt))})
    return jobs


def reference_jobs(plan: dict, style: str, refs_dir: Path) -> list[dict]:
    """One picture per recurring character (full length, plain background)
    and place (empty establishing view) that the scenes use: drawn first,
    then handed to every scene they appear in, so faces, clothes and rooms
    stay the same. Kept per BOOK (refs_dir), reused across chapters."""
    return [{"id": f"ref_{slug(n)}", "size": size, "seed": 7, "prompt": prompt,
             "path": str(ref_path(refs_dir, n, prompt))}
            for n, (prompt, size) in ref_prompts(plan, style, registry_at(refs_dir)).items()]


def scene_jobs(pics: list[dict], plan: dict, img_dir: Path, refs_dir: Path,
               max_refs: int = 3, style: str = "") -> list[dict]:
    """Each scene with the references of who and where it shows: characters
    first (up to *max_refs*), then one place if there's room."""
    R = registry_at(refs_dir)
    chars, places = looks(plan, R)
    jobs = []
    for p in pics:
        if "who" in p:          # the plan names them (and the prompt numbers the images)
            who, where = p["who"][:max_refs], [p["where"]] if p.get("where") else []
        else:
            who = mentions(p["prompt"], chars)[:max_refs]
            where = mentions(p["prompt"], places)[:max(0, max_refs - len(who))][:1]
        prompts = ref_prompts(plan, style, R)
        refs = []
        for n in who + where:
            row = outfit(plan, n, p.get("start"), R) if n in chars else None
            kn = n if n in prompts else next((k for k in prompts if R is not None and R.same(k, n)), None)
            if row:
                refs.append(str(outfit_ref(refs_dir, plan, style, n, row, R)))
            elif kn:
                refs.append(str(ref_path(refs_dir, kn, prompts[kn][0])))
        refs += p.get("extra_refs", [])          # e.g. a talking portrait's scene still
        jobs.append({"id": p["id"], "prompt": p["full_prompt"], "seed": p["seed"],
                     "path": str(img_dir / f"{p['id']}.png"), "refs": refs})
    return jobs


# The dock-attached XTX sometimes faults mid-run (corrupted copies, shader
# errors) and then waits forever on the GPU: no picture for this long means
# the run is dead.
DRAW_STALL_S = float(os.environ.get("BOOKCASTER_DRAW_STALL", "300"))


def _run_drawer(jobs_file: Path, img_dir: Path, n_jobs: int, log) -> str:
    """One illustrate.py run: 'ok', 'failed' or 'stalled' (killed)."""
    with open(img_dir.parent / "draw.log", "w") as err:   # a file: a full stderr pipe would block it
        proc = subprocess.Popen(
            [*memory_scope(), str(DRAW_PYTHON), "-I", str(DRAW_SCRIPT), str(jobs_file), str(img_dir)],
            stdout=subprocess.PIPE, stderr=err, text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": ART_GPU,
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        last, stalled = [time.time()], threading.Event()

        def watch():
            while proc.poll() is None:
                if time.time() - last[0] > DRAW_STALL_S:
                    stalled.set()
                    proc.kill()
                    return
                time.sleep(5)
        threading.Thread(target=watch, daemon=True).start()
        done = 0
        for line in proc.stdout:
            last[0] = time.time()
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "phase" in msg:
                log(f"    {msg['phase']}: {msg.get('seconds', '?')} s")
            if "id" in msg and msg.get("error"):          # a per-job error: loud, never a KeyError (10-21)
                log(f"    !!! {msg['id']}: {msg['error'][:300]}")
            elif "id" in msg:
                done += 1
                if done == 1 or done % 10 == 0 or done == n_jobs:
                    log(f"    drew {done}/{n_jobs} ({msg.get('seconds', '?')} s each)")
        rc = proc.wait()
    if stalled.is_set():
        return "stalled"
    return "failed" if rc else "ok"



def draw(pics: list[dict], img_dir: Path, log=print, plan: dict | None = None,
         style: str = "", refs_dir: Path | None = None) -> None:
    """Draw what isn't drawn yet: the references (if a plan is given), then
    the scenes, in one model load."""
    refs_dir = refs_dir or img_dir.parent.parent / "refs"
    if plan is not None:
        jobs = (reference_jobs(plan, style, refs_dir) + outfit_jobs(plan, style, refs_dir)
                + scene_jobs(pics, plan, img_dir, refs_dir, style=style))
    else:
        jobs = [{"id": p["id"], "prompt": p["full_prompt"], "seed": p["seed"],
                 "path": str(img_dir / f"{p['id']}.png")} for p in pics]
    jobs = [j for j in jobs if not Path(j["path"]).exists()]
    if not jobs:
        log(f"  all {len(pics)} pictures already drawn")
        return
    draw_jobs(jobs, img_dir, log)


def draw_jobs(jobs: list[dict], img_dir: Path, log=print, on_done=None) -> None:
    """Run ready-made drawing jobs ({id, prompt, seed, path, refs?, size?}) in
    one model load, in order (a job may use an earlier job's picture as a
    reference: illustrate.py opens references as it reaches each job).
    Inside drawing_session() the session's loaded model draws them, and
    on_done(job) is called as each picture is saved (a check can look at it
    while the next one is drawn)."""
    jobs = [j for j in jobs if not Path(j["path"]).exists()]
    if not jobs:
        return
    img_dir.mkdir(parents=True, exist_ok=True)
    log(f"  drawing {len(jobs)} picture(s)...")
    t = time.time()
    if _SESSION and not _SESSION.broken:
        _SESSION.draw(jobs, log, on_done)
    left = [j for j in jobs if not Path(j["path"]).exists()]
    if left and (not _SESSION or _SESSION.broken):
        jobs = left
        jobs_file = img_dir.parent / "jobs.json"
        jobs_file.write_text(json.dumps(jobs, ensure_ascii=False))
        with _music_paused(log):
            _draw_attempts(jobs, jobs_file, img_dir, log)
        for j in jobs:
            if on_done and Path(j["path"]).exists():
                on_done(j)
    log(f"  drew {len(jobs)} picture(s) in {time.time() - t:.0f} s")


_SESSION = None

# The cards that draw during a keyframe stage: the music card and the voice server's card (both servers
# are paused meanwhile). Best-of-N draws ~3 candidates a keyframe; two cards halve that (10-18 plan).
DRAW_BUSES = [b for b in os.environ.get("BOOKCASTER_DRAW_BUSES", "65,05").split(",") if b.strip()]


class DrawPool:
    """One illustrate.py --serve process per card, all taking jobs from one queue: pictures can be added
    at any time (a failed keyframe's redraw, the next keyframe once the one it is drawn from is chosen) and
    draw on whichever card is free. A job waits while a reference it uses is still being drawn by another
    job (an outfit sheet needs its base sheet: with two cards, order alone no longer guarantees that). A
    card that stalls or dies twice drops out; its jobs go to the others."""

    def __init__(self, log_file: Path, buses: list[str], log=print):
        import queue
        self.log_file, self.log = log_file, log
        self.q, self.cv = queue.Queue(), threading.Condition()
        self.pending, self.outputs, self.alive, self.broken = 0, set(), 0, False
        self.on_done = None
        self.threads = []
        for bus in buses:
            t = threading.Thread(target=self._worker, args=(bus,), daemon=True)
            self.threads.append(t)
            self.alive += 1
            t.start()

    # --- the draw_jobs / requeue interface -------------------------------------------------
    def put(self, job: dict, on_done=None) -> None:
        with self.cv:
            self.pending += 1
            self.outputs.add(str(job["path"]))
        self.q.put((job, on_done or self.on_done))

    def wait(self) -> None:
        with self.cv:
            while self.pending and self.alive:
                self.cv.wait(timeout=5)

    def draw(self, jobs: list[dict], log, on_done=None) -> None:
        self.on_done = on_done
        for j in jobs:
            self.put(j, on_done)
        self.wait()

    def add(self, job: dict) -> bool:
        if not self.alive:
            return False
        self.put(job)
        return True

    def stop(self) -> None:
        for _ in self.threads:
            self.q.put((None, None))
        for t in self.threads:
            t.join(timeout=90)

    # --- one card -----------------------------------------------------------------------------
    def _start(self, bus: str):
        import queue
        err = open(self.log_file.with_name(f"{self.log_file.stem}-{bus}.log"), "a")
        proc = subprocess.Popen(
            [*memory_scope(), str(DRAW_PYTHON), "-I", str(DRAW_SCRIPT), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": hip_index(bus),
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        err.close()
        lines = queue.Queue()

        def read():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=read, daemon=True).start()
        return proc, lines

    def _ready(self, job: dict) -> bool:
        """No reference of this job is still waiting to be drawn."""
        with self.cv:
            return not any(str(r) in self.outputs and not Path(r).exists() for r in job.get("refs", []))

    def _done(self, job: dict) -> None:
        with self.cv:
            self.pending -= 1
            self.outputs.discard(str(job["path"]))
            self.cv.notify_all()

    def _worker(self, bus: str) -> None:
        import queue
        keep = ("id", "prompt", "seed", "path", "refs", "size", "control")
        proc, lines, first, fails = None, None, True, 0
        while True:
            job, cb = self.q.get()
            if job is None:
                break
            if not self._ready(job):                 # its reference is still being drawn: later
                self.q.put((job, cb))
                time.sleep(0.5)
                continue
            if Path(job["path"]).exists():
                self._done(job)
                if cb:
                    cb(job)
                continue
            if proc is None or proc.poll() is not None:
                proc, lines = self._start(bus)
                first = True
            proc.stdin.write(json.dumps({k: v for k, v in job.items() if k in keep}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            status = None
            while status is None:
                try:     # the first wait includes the model load
                    line = lines.get(timeout=DRAW_STALL_S + (120 if first else 0))
                except queue.Empty:
                    status = "stalled"
                    break
                if line is None:
                    status = "died"
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "phase" in msg:
                    self.log(f"    [{bus}] {msg['phase']}: {msg['seconds']} s")
                elif msg.get("id") == job["id"]:
                    status = "error" if "error" in msg else "ok"
                    if "error" in msg:
                        self.log(f"    [{bus}] {job['id']}: {msg['error']}")
            first = False
            if status in ("ok", "error"):
                self._done(job)
                if cb and (Path(job["path"]).exists() or job.get("_report_errors", True)):
                    try:
                        cb(job)             # also on an error: a caller counting its jobs must not wait forever
                    except Exception as exc:
                        self.log(f"    [{bus}] callback for {job['id']}: {exc}")
                continue
            fails += 1                                  # stalled or died: the job goes back for any card
            proc.kill()
            proc.wait()
            proc = None
            self.q.put((job, cb))
            if fails >= 2:
                self.log(f"    [{bus}] drawing card dropped out ({status} twice)")
                break
        if proc and proc.poll() is None:
            try:
                proc.stdin.close()
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
        with self.cv:
            self.alive -= 1
            self.broken = self.alive == 0
            self.cv.notify_all()


class _DrawSession:
    """illustrate.py --serve for a run of draw_jobs calls: the model loads once.
    A run that stalls (no picture for DRAW_STALL_S) or dies is started again
    once, carrying on with the pictures not saved yet; if that one fails too,
    the session is marked broken and draw_jobs goes back to one run per call
    (the old way, which frees the text half before loading the picture half)."""

    def __init__(self, log_file: Path):
        self.log_file, self.proc, self.lines, self.broken = log_file, None, None, False
        self.lock, self.batch = threading.Lock(), None     # batch: {"expected", "jobs"} while draw() runs

    def add(self, job: dict) -> bool:
        """Queue one more picture into the batch being drawn (a keyframe that just failed its check goes
        straight back, user 10-18). False when no batch is running: the caller redraws it later."""
        keep = ("id", "prompt", "seed", "path", "refs", "size", "control")
        with self.lock:
            if self.batch is None or self.proc is None or self.proc.poll() is not None:
                return False
            try:
                self.proc.stdin.write(json.dumps({k: v for k, v in job.items() if k in keep},
                                                 ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
            except OSError:
                return False
            self.batch["expected"] += 1
            self.batch["jobs"][job["id"]] = job
            return True

    def _start(self, log):
        import queue
        err = open(self.log_file, "a")
        self.proc = subprocess.Popen(
            [*memory_scope(), str(DRAW_PYTHON), "-I", str(DRAW_SCRIPT), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
            env={**os.environ, "HIP_VISIBLE_DEVICES": ART_GPU,
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        err.close()
        self.lines = queue.Queue()

        def read(proc, q):
            for line in proc.stdout:
                q.put(line)
            q.put(None)
        threading.Thread(target=read, args=(self.proc, self.lines), daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=60)
            except Exception:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def draw(self, jobs: list[dict], log, on_done=None) -> None:
        import queue
        for attempt in (1, 2):
            left = [j for j in jobs if not Path(j["path"]).exists()]
            if not left:
                return
            if self.proc is None or self.proc.poll() is not None:
                self._start(log)
            keep = ("id", "prompt", "seed", "path", "refs", "size", "control")
            with self.lock:
                for j in left:
                    self.proc.stdin.write(json.dumps({k: v for k, v in j.items() if k in keep},
                                                     ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                self.batch = {"expected": len(left), "jobs": {j["id"]: j for j in left}}
            by_id, done, status = self.batch["jobs"], 0, "ok"
            while True:
                with self.lock:
                    if done >= self.batch["expected"]:
                        self.batch = None
                        break
                try:     # the first wait includes the model load
                    line = self.lines.get(timeout=DRAW_STALL_S + (120 if done == 0 else 0))
                except queue.Empty:
                    status = "stalled"
                    break
                if line is None:
                    status = "failed"
                    break
                try:
                    msg = json.loads(line)
                except ValueError:
                    continue
                if "phase" in msg:
                    log(f"    {msg['phase']}: {msg['seconds']} s")
                elif "id" in msg:
                    done += 1
                    if "error" in msg:
                        log(f"    {msg['id']}: {msg['error']}")
                    elif on_done and msg["id"] in by_id:
                        on_done(by_id[msg["id"]])     # a re-queued picture comes back through here too
                    if done == 1 or done % 10 == 0 or done == self.batch["expected"]:
                        log(f"    drew {done}/{self.batch['expected']} ({msg.get('seconds', '?')} s each)")
            with self.lock:
                self.batch = None
            if status == "ok":
                return
            self.proc.kill()
            self.proc.wait()
            self.proc = None
            n = sum(not Path(j["path"]).exists() for j in jobs)
            tail = [l for l in self.log_file.read_text(errors="replace").splitlines() if l.strip()][-3:]
            if attempt == 2:
                self.broken = True
                log(f"    the kept drawing model failed ({status}: {' | '.join(tail)[-200:]}): "
                    f"back to one model load per batch")
                return
            log(f"    drawing {status}, {n} picture(s) left: starting the drawer again")


@contextlib.contextmanager
def requeue(job: dict) -> bool:
    """Put a picture back into the drawing batch now running (see _DrawSession.add)."""
    return bool(_SESSION and not _SESSION.broken and _SESSION.add(job))


@contextlib.contextmanager
def drawing_session(log_file: Path, log=print):
    """Keep the drawing model loaded across the draw_jobs calls inside (a keyframe phase: draw, check,
    redraw, then the talking portraits), on every card in DRAW_BUSES. The music server shares the first
    card and the voice server the second: both stay stopped for the whole session."""
    global _SESSION
    if _SESSION:                     # nested: the outer session serves
        yield
        return
    voice_card = any(b != ART_BUS for b in DRAW_BUSES)
    fish = voice_card and subprocess.run(["systemctl", "--user", "is-active", "-q", "tts-fish"]).returncode == 0
    if fish:
        log("    pausing the voice server: its card draws too")
        subprocess.run(["systemctl", "--user", "stop", "tts-fish-adapter", "tts-fish"], capture_output=True)
    # the session HOLDS its cards' locks while its drawers are loaded (10-21 v3 plan: a resident drawer on 05 without
    # the lock; the sheet check's judge took 05 as "free" and ran out of memory): other GPU users see them busy
    import fcntl
    locks = []
    for b in DRAW_BUSES:
        lf = open(f"/tmp/gpu-{b}.lock", "a")
        fcntl.flock(lf, fcntl.LOCK_EX)
        locks.append(lf)
    with _music_paused(log):
        _SESSION = DrawPool(log_file, DRAW_BUSES, log)
        try:
            yield
        finally:
            _SESSION.stop()
            _SESSION = None
            for lf in locks:
                lf.close()
            if fish:
                subprocess.run(["bash", str(Path(__file__).resolve().parent.parent / "tools" / "fish_stack.sh"), "up"],
                               capture_output=True)


@contextlib.contextmanager
def _music_paused(log):
    """The drawing model shares the music XTX with ACE-Step, which keeps ~13 GB after its first
    piece of music: the drawer then ran out of memory (10-17 storyboard). ACE has no unload call,
    so it is stopped for the drawing and started again after (it is back in well under a minute)."""
    unit = "tts-ace"
    up = subprocess.run(["systemctl", "--user", "is-active", "-q", unit]).returncode == 0
    if up:
        log("    pausing the music server while drawing (they share a card)")
        subprocess.run(["systemctl", "--user", "stop", unit], capture_output=True)
    try:
        yield
    finally:
        if up:   # a transient unit: fish_stack.sh makes it again (the voice server, already up, is left alone)
            subprocess.run([str(Path(__file__).resolve().parents[1] / "tools" / "fish_stack.sh"), "up", "--music"],
                           capture_output=True)


def _draw_attempts(jobs, jobs_file, img_dir, log) -> None:
    for attempt in (1, 2):
        # illustrate.py skips pictures already saved, so a retry carries on
        status = _run_drawer(jobs_file, img_dir, len(jobs), log)
        if status == "ok":
            break
        left = sum(not Path(j["path"]).exists() for j in jobs)
        err = (img_dir.parent / "draw.log").read_text(errors="replace")
        tail = [l for l in err.splitlines() if l.strip()][-3:]
        if attempt == 2 or left == 0:
            raise RuntimeError(f"drawing failed ({status}): " + " | ".join(tail))
        log(f"    drawing {status}, {left} picture(s) left: retrying once")



# --- 3b. check (vision) -------------------------------------------------------------
# The image model draws extra hands, arms or a second copy of a person now
# and then. The language model can look at pictures (Strata --vision): each
# scene is shown to it, and a faulty one is drawn again with another seed.

CHECK_ROUNDS = int(os.environ.get("BOOKCASTER_CHECK_ROUNDS", "2"))


# --- the picture checker on an XTX -------------------------------------------------------
# Keyframe checks ran on the language model on the iGPU, one at a time, ~40 s a picture: three times
# slower than drawing, so the keyframe stage was the checks (45+ min for 56 pictures, user 10-18).
# During the keyframe stage the voice server's XTX sits idle: a vision model there (Qwen3.8-27B Q4 +
# its projector, ~17 GB) checks CHECKER_SLOTS pictures at once. Small context, no flash attention
# (the XTX reset its ring at 8k + fa, 09-12). Without it, checks go to the iGPU as before.
CHECKER_BIN = Path(os.environ.get("BOOKCASTER_CHECKER_BIN", Path.home() / "llama.cpp-merge/build/bin/llama-server"))
CHECKER_MODEL = Path(os.environ.get("BOOKCASTER_CHECKER_MODEL", Path.home() / "models/Qwen3.8-27B-Q4_K_S.gguf"))
CHECKER_MMPROJ = Path(os.environ.get("BOOKCASTER_CHECKER_MMPROJ", Path.home() / "models/Qwen3.8-27B-mmproj-F16.gguf"))
CHECKER_PORT = int(os.environ.get("BOOKCASTER_CHECKER_PORT", "18111"))
CHECKER_SLOTS = int(os.environ.get("BOOKCASTER_CHECKER_SLOTS", "4"))
CHECKER_BUS = os.environ.get("BOOKCASTER_CHECKER_BUS", "05")
# On llama.cpp-merge with hand-picked flags the 27B decoded 13 tok/s (2 each at 4 slots); through the
# production launcher ~30 per stream at 2 slots (10-18). It checks beside the iGPU model, not instead.
CHECKER_ON = os.environ.get("BOOKCASTER_CHECKER", "1") == "1"
CHECKER_LAUNCHER = Path(os.environ.get("BOOKCASTER_CHECKER_LAUNCHER", Path.home() / "models/run-qwen38.sh"))
CHECKER_UNIT = "bookcaster-checker"
CHECKER: dict = {"url": None}        # set while checker_session() runs


def _vulkan_index(bus: str) -> str | None:
    """ggml's Vulkan index (GGML_VK_VISIBLE_DEVICES) of the GPU on PCI bus *bus* (hex). ggml lists the
    discrete GPUs first, then the integrated one, each in Vulkan's order: vulkaninfo's own index (which
    counts the iGPU first here) put the checker on the drawing model's card - two models on one 24 GB
    card spilled into system RAM (1 GB left) and reset its ring (10-18)."""
    out = subprocess.run(["vulkaninfo"], capture_output=True, text=True).stdout
    devs, cur = [], None
    for line in out.splitlines():
        if re.match(r"GPU\d+:", line.strip()):
            cur = {"bus": None, "type": ""}
            devs.append(cur)
        elif cur is not None:
            m = re.search(r"pciBus\s*=\s*(\d+)", line)
            if m and cur["bus"] is None:
                cur["bus"] = int(m.group(1))
            m = re.search(r"deviceType\s*=\s*PHYSICAL_DEVICE_TYPE_(\w+)", line)
            if m and not cur["type"]:
                cur["type"] = m.group(1)
    gpus = [d for d in devs if d["type"] in ("DISCRETE_GPU", "INTEGRATED_GPU")]
    order = [d for d in gpus if d["type"] == "DISCRETE_GPU"] + [d for d in gpus if d["type"] == "INTEGRATED_GPU"]
    for i, d in enumerate(order):
        if d["bus"] == int(bus, 16):
            return str(i)
    return None


def _vram_used_gb(bus: str) -> float:
    f = Path(f"/sys/bus/pci/devices/0000:{bus}:00.0/mem_info_vram_used")
    try:
        return int(f.read_text()) / 2**30
    except (OSError, ValueError):
        return -1.0


@contextlib.contextmanager
def checker_session(log=print):
    """The vision checker on the voice server's XTX for the length of the block (the voice server is
    paused and started again after). Yields its URL, or None when it can't run (checks then use the
    iGPU language model)."""
    # never beside a drawer: the checker and the second drawing card both took bus 05 and the 24 GB
    # card spilled into system RAM (1 GB left, 10-19). When its card draws, Strata checks.
    if CHECKER["url"] or not CHECKER_ON or not CHECKER_LAUNCHER.exists() or CHECKER_BUS in DRAW_BUSES:
        yield CHECKER["url"]
        return
    vk = _vulkan_index(CHECKER_BUS)
    if vk is None:
        yield None
        return
    fish = subprocess.run(["systemctl", "--user", "is-active", "-q", "tts-fish"]).returncode == 0
    if fish:
        log("    pausing the voice server: its card checks the pictures")
        subprocess.run(["systemctl", "--user", "stop", "tts-fish-adapter", "tts-fish"], capture_output=True)
    subprocess.run(["systemctl", "--user", "reset-failed", CHECKER_UNIT], capture_output=True)
    # the production launcher (tuned build snapshot, fa on, q8 KV, draft model): hand-picked flags on
    # another build decoded 13 tok/s; this one ~30 per stream (10-18). GPU=egpu = the first XTX by name.
    subprocess.run(["systemd-run", "--user", f"--unit={CHECKER_UNIT}", "-p", "MemoryMax=24G",
                    "--setenv=GPU=egpu", "--setenv=VISION=1", f"--setenv=NP_EGPU={CHECKER_SLOTS}",
                    f"--setenv=CTX_OVERRIDE={16384 * CHECKER_SLOTS}", "--setenv=MODEL_RUN=0",
                    f"--setenv=PORT={CHECKER_PORT}", "bash", str(CHECKER_LAUNCHER)], capture_output=True)
    url = f"http://127.0.0.1:{CHECKER_PORT}/v1/chat/completions"
    from urllib.request import urlopen
    ok = False
    for _ in range(180):
        try:
            urlopen(f"http://127.0.0.1:{CHECKER_PORT}/health", timeout=3)
            ok = True
            break
        except OSError:
            if subprocess.run(["systemctl", "--user", "is-active", "-q", CHECKER_UNIT]).returncode != 0:
                break
            time.sleep(2)
    if ok and _vram_used_gb(CHECKER_BUS) < 12:     # loaded somewhere else: never beside the drawer
        log(f"    picture checker isn't on the bus-{CHECKER_BUS} card: stopped, checking on the language model")
        subprocess.run(["systemctl", "--user", "stop", CHECKER_UNIT], capture_output=True)
        ok = False
    if ok:
        log(f"    picture checker up on the bus-{CHECKER_BUS} XTX ({CHECKER_SLOTS} at a time)")
        CHECKER["url"] = url
    else:
        log("    picture checker didn't start: checking on the language model instead")
    try:
        yield CHECKER["url"]
    finally:
        CHECKER["url"] = None
        subprocess.run(["systemctl", "--user", "stop", CHECKER_UNIT], capture_output=True)
        if fish:
            subprocess.run(["bash", str(Path(__file__).resolve().parent.parent / "tools" / "fish_stack.sh"), "up"],
                           capture_output=True)


_ROTA = {"n": 0}
_ROTA_LOCK = threading.Lock()


def checker_url(llm_url: str | None) -> str | None:
    """Where a picture check goes: in turn the XTX checker's slots and the iGPU language model while the
    checker runs (both at once), else the language model."""
    if not CHECKER["url"] or not llm_url:
        return CHECKER["url"] or llm_url
    pool = [CHECKER["url"]] * CHECKER_SLOTS + [llm_url]
    with _ROTA_LOCK:
        _ROTA["n"] += 1
        return pool[_ROTA["n"] % len(pool)]


def checker_workers() -> int:
    return CHECKER_SLOTS + 1 if CHECKER["url"] else 1


def _look_at_probs(llm_url: str, model: str, png: Path, question: str, max_tokens: int = 200):
    """_look_at with the answer's token log-probabilities: (text, [(token, {alt: logprob})]), or the text
    and [] where the server gives none."""
    import base64
    from urllib.request import Request, urlopen
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "logprobs": True, "top_logprobs": 5,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()}},
                {"type": "text", "text": question}]}]}
    req = Request(llm_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as r:
        ch = json.loads(r.read())["choices"][0]
    toks = []
    for t in ((ch.get("logprobs") or {}).get("content") or []):
        alts = {a["token"]: a["logprob"] for a in t.get("top_logprobs") or []}
        alts.setdefault(t.get("token", ""), t.get("logprob", 0.0))
        toks.append((t.get("token", ""), alts))
    return ch["message"]["content"], toks


def _look_at(llm_url: str, model: str, png: Path, question: str) -> str:
    import base64
    from urllib.request import Request, urlopen
    body = {"model": model, "temperature": 0, "max_tokens": 1500,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()}},
                {"type": "text", "text": question}]}]}
    req = Request(llm_url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


CHECK_VERSION = 2   # bump when check_picture changes: pictures that passed are checked again


def check_picture(png: Path, people: int | None, llm_url: str, model: str,
                  dress: dict | None = None) -> list[str]:
    """What is wrong with one drawing ([] = fine). *people*: how many should be
    in it. Asked for "problems" the model invented some (two of every three
    flags were wrong); asked for counts it missed a four-armed man. Tracing
    each arm from its shoulder, then yes/no questions, got the known cases
    right. It cannot see a face merged into a chair, even cropped: that needs
    `bookcaster redraw`."""
    q = ("Inspect this illustration for mistakes typical of AI image generators. For each person, "
         "trace every arm from the shoulder to the hand and say which shoulder it comes from; count "
         "them. Then check: is any face or body merged into furniture, clothing or another object? "
         "Are there arms, hands or legs whose body is not in the picture (a close-up of limbs "
         "alone)? Is the picture split into two or more panels, or is the same person drawn twice? "
         "Is there a second head? Then answer with a final line of JSON: "
         '{"people": <number of people>, "extra_limbs": true/false, "merged": true/false, '
         '"limbs_without_body": true/false, "panels_or_duplicate": true/false, "details": "..."}')
    reply = _look_at(llm_url, model, png, q)       # OSError (no server / no vision) goes to the caller
    found = re.findall(r"\{[^{}]*\"extra_limbs\"[^{}]*\}", reply or "")
    try:
        data = json.loads(found[-1]) if found else {}
    except ValueError:
        return []
    probs = []
    if any(data.get(k) is True for k in ("extra_limbs", "merged", "limbs_without_body",
                                          "panels_or_duplicate", "duplicate_person")):
        probs.append(str(data.get("details") or "drawing mistake")[:200])
    try:
        n = int(data.get("people"))
    except (TypeError, ValueError):
        n = None
    if people is not None and n is not None and n > people:
        probs.append(f"{n} people instead of {people}")
    for name, worn in (dress or {}).items():
        # the wardrobe's outfit of this moment: a scene of the fight had the knight in his stable leathers
        q = (f"Look at {name.split()[0]} in this illustration ({_short(str(worn))}...). Is that person wearing "
             f"this: {worn}? Small differences in detail are fine; a different kind of clothing is not. "
             'Answer with a final line of JSON: {"wears": true/false, "why": "..."}')
        r = _look_at(llm_url, model, png, q)
        m = re.findall(r"\{[^{}]*\"wears\"[^{}]*\}", r or "")
        try:
            if m and json.loads(m[-1]).get("wears") is False:
                probs.append(f"{name} is not wearing {worn[:80]}: {json.loads(m[-1]).get('why', '')[:100]}")
        except ValueError:
            pass
    return probs


def _checks_path(img_dir: Path) -> Path:
    return img_dir.parent / "checks.json"


def apply_reseeds(pics: list[dict], img_dir: Path) -> None:
    """Pictures redrawn after a failed check keep the seed that drew them."""
    f = _checks_path(img_dir)
    seeds = json.loads(f.read_text()).get("seeds", {}) if f.exists() else {}
    for p in pics:
        if p["id"] in seeds:
            p["seed"] = seeds[p["id"]]


def reject(pics: list[dict], img_dir: Path, which: list[str]) -> list[str]:
    """Pictures the user says are wrong, by id (p021) or by when they are on
    screen (12:30 or 1:02:03): drawn again with a new seed at the next build.
    Returns the ids."""
    def secs(t: str) -> float:
        out = 0.0
        for part in t.split(":"):
            out = out * 60 + float(part)
        return out
    ids = []
    for w in which:
        if re.fullmatch(r"p\d+", w):
            hit = [p for p in pics if p["id"] == w]
        else:
            t = secs(w)
            hit = [p for p in pics if p["t0"] <= t < p["t1"]]
        if not hit:
            raise ValueError(f"no picture {w}")
        ids.append(hit[0]["id"])
    f = _checks_path(img_dir)
    state = json.loads(f.read_text()) if f.exists() else {}
    state.setdefault("seeds", {})
    for p in pics:
        if p["id"] in ids:
            state["seeds"][p["id"]] = int(state["seeds"].get(p["id"], p["seed"])) + 104729
            (img_dir / f"{p['id']}.png").unlink(missing_ok=True)
            state.setdefault("passed", {}).pop(p["id"], None)
            state.setdefault("log", []).append({"id": p["id"], "problems": ["rejected by the user"]})
    f.write_text(json.dumps(state, indent=1))
    return ids


def draw_checked(pics: list[dict], img_dir: Path, plan: dict, style: str, llm_url: str,
                 model: str, log=print) -> None:
    """draw(), then show each new scene to the vision model; a faulty one is
    redrawn with a new seed (CHECK_ROUNDS times at most). A picture that
    passed (same file) isn't checked again; with no vision server it's skipped."""
    apply_reseeds(pics, img_dir)
    draw(pics, img_dir, log=log, plan=plan, style=style)
    f = _checks_path(img_dir)
    state = json.loads(f.read_text()) if f.exists() else {}
    state.setdefault("seeds", {})
    if state.get("version") != CHECK_VERSION:
        state["passed"], state["version"] = {}, CHECK_VERSION
    state.setdefault("passed", {})
    for rnd in range(CHECK_ROUNDS + 1):
        todo = [p for p in pics if (img_dir / f"{p['id']}.png").exists()
                and state["passed"].get(p["id"]) != (img_dir / f"{p['id']}.png").stat().st_mtime]
        if not todo:
            break
        t, bad = time.time(), []
        for p in todo:
            png = img_dir / f"{p['id']}.png"
            try:
                # a crowd has no fixed count: only the drawing-mistake flags count there
                probs = check_picture(png, len(p["who"]) if "who" in p and not p.get("crowd") else None,
                                      llm_url, model, scene_dress(plan, p))
            except Exception as exc:          # the server can't see pictures, or is down
                log(f"  picture check skipped ({exc})")
                return
            if probs and rnd < CHECK_ROUNDS:
                bad.append(p)
                state.setdefault("log", []).append({"id": p["id"], "seed": p["seed"], "problems": probs})
            else:
                state["passed"][p["id"]] = png.stat().st_mtime
                if probs:
                    log(f"    {p['id']} kept after {CHECK_ROUNDS} redraws: {'; '.join(probs)[:150]}")
        log(f"  checked {len(todo)} picture(s) in {time.time() - t:.0f} s: {len(bad)} to redraw")
        for p in bad:
            p["seed"] = int(p["seed"]) + 7919
            state["seeds"][p["id"]] = p["seed"]
            (img_dir / f"{p['id']}.png").unlink()
        f.write_text(json.dumps(state, indent=1))
        if not bad:
            break
        draw(pics, img_dir, log=log, plan=plan, style=style)
    f.write_text(json.dumps(state, indent=1))

# --- subtitles ---------------------------------------------------------------------

def _norm_map(s: str) -> tuple[str, list[int]]:
    """Lower-case letters/digits with single spaces, plus each kept
    character's index in *s*."""
    out, idx = [], []
    for i, ch in enumerate(s.lower()):
        if ch.isalnum():
            out.append(ch)
            idx.append(i)
        elif out and out[-1] != " ":
            out.append(" ")
            idx.append(i)
    return "".join(out), idx


SPOKEN_ABBR = {"doctor": "dr", "missus": "mrs", "mister": "mr", "saint": "st"}


def restore_text(timeline: list[dict], text: str, cut: int = 160) -> list[dict]:
    """Recordings made before engine.py kept the whole line stored only its
    first 160 characters, so a long paragraph got one caption and then
    nothing for a minute or more. Refill each cut line from the chapter: from
    where it starts to where the next line starts."""
    if not any(len(str(e.get("text", ""))) >= cut for e in timeline):
        return timeline
    norm, idx = _norm_map(text)
    starts, cur = [], 0
    for e in timeline:
        head = _norm_map(re.sub(r"\[[^\]]*\]", "", str(e.get("text", ""))))[0].strip()
        pos = -1
        # the narration says "Doctor" / "Missus" where the book prints "Dr." / "Mrs."
        book_way = re.sub(r"\b(doctor|missus|mister|saint)\b",
                          lambda m: SPOKEN_ABBR[m.group(1)], head)
        for h in dict.fromkeys((head, book_way)):
            for n in (40, 20):
                if pos < 0 and len(h) >= 8:
                    pos = norm.find(h[:n], cur, cur + 20000)
        starts.append(pos)
        if pos >= 0:
            cur = pos + 1
    out = []
    for k, e in enumerate(timeline):
        t = str(e.get("text", ""))
        nxt = next((s for s in starts[k + 1:] if s >= 0), len(norm))
        if len(t) >= cut and starts[k] >= 0 and nxt > starts[k]:
            end = idx[nxt] if nxt < len(idx) else len(text)
            full = re.sub(r"\s+", " ", text[idx[starts[k]]:end]).strip().strip("\"“”‘’ ,")
            full = re.sub(r"[\s\"“”‘’(]+$", "", full)    # not the next line's opening bracket: 'ministry.” (
            if len(full) > len(t) * 0.9:
                e = {**e, "text": full}
        out.append(e)
    return out


def subtitle_cues(timeline: list[dict], total: float, width: int = 84) -> list[tuple]:
    """(start, end, text) per caption: each timeline line, split into
    chunks of at most two lines, timed by their share of the characters."""
    cues = []
    for a, b in zip(timeline, timeline[1:] + [None]):
        text = re.sub(r"\[[^\]]*\]", "", str(a.get("text", ""))).strip()
        if not text:
            continue
        t0 = float(a["t"])
        t1 = min(float(b["t"]) if b else total, t0 + len(text) / 9 + 2.5) - 0.05
        if t1 - t0 < 0.3:
            continue
        chunks = []
        for sent in re.split(r"(?<=[.!?;])\s+", text):   # a caption never runs across a sentence
            cur = ""
            for w in sent.split():
                if cur and len(cur) + 1 + len(w) > width:
                    chunks.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                # a short sentence joins the previous caption if both fit
                if chunks and len(chunks[-1]) + 1 + len(cur) <= width // 2:
                    chunks[-1] += " " + cur
                else:
                    chunks.append(cur)
        # libass wraps each chunk onto at most two lines
        total_chars = sum(len(c) for c in chunks)
        t = t0
        for c in chunks:
            d = (t1 - t0) * len(c) / total_chars
            cues.append((t, t + d, c))
            t += d
    return cues


def _ts(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}".replace(".", ",")


def write_srt(cues: list[tuple], path: Path, offset: float = 0.0, until: float = 1e9) -> None:
    lines, n = [], 0
    for t0, t1, text in cues:
        if t1 <= offset or t0 >= until:
            continue
        n += 1
        lines += [str(n), f"{_ts(max(t0, offset) - offset)} --> {_ts(min(t1, until) - offset)}",
                  text, ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --- 4. build ----------------------------------------------------------------------

SUB_STYLE = ("FontName=Noto Sans,FontSize=17,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
             "BackColour=&H80000000,BorderStyle=1,Outline=1.6,Shadow=0.8,MarginV=26,WrapStyle=0")


def _clip(n: int, img: Path, frames: int, srt: Path | None, out: Path,
          fade_in: bool = True, fade_out: bool = True) -> None:
    """One picture: a slow zoom (in or out, alternating), panning toward a
    side, fading in/out over black; subtitles burned in. A picture split
    around an animated shot fades only at its own start and end."""
    d = frames
    zin = n % 2 == 0
    # 1.06-1.18, never 1.0: the edges (where image models sign) stay cropped
    z = f"1.06+0.12*on/{d}" if zin else f"1.18-0.12*on/{d}"
    side = (n // 2) % 3            # 0 centre, 1 drift left, 2 drift right
    x = {0: "iw/2-(iw/zoom/2)", 1: f"(iw-iw/zoom)*(0.5-0.5*on/{d})",
         2: f"(iw-iw/zoom)*(0.5+0.5*on/{d})"}[side]
    fade = min(12, d // 4)
    vf = (f"scale=3840:2160:flags=lanczos,setsar=1,"
          f"zoompan=z='{z}':x='{x}':y='ih/2-(ih/zoom/2)':d={d}:s={W}x{H}:fps={FPS}")
    if fade_in:
        vf += f",fade=t=in:st=0:n={fade}"
    if fade_out:
        vf += f",fade=t=out:s={d - fade}:n={fade}"
    if srt is not None:
        vf += f",subtitles='{srt}':force_style='{SUB_STYLE}'"
    vf += ",format=yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(img), "-vf", vf,
                    "-frames:v", str(d), "-r", str(FPS), "-c:v", *VIDEO_CODEC,
                    "-b:v", VIDEO_BITRATE, "-threads", "4", str(out)], check=True)


def _anim_clip(src: Path, frames: int, srt: Path | None, out: Path) -> None:
    """An animated shot at the video's size and rate, cut or held (last frame)
    to exactly *frames*, subtitles burned in."""
    # fill and crop, not stretch: the video models make 832x480 (2.6% narrower than 16:9)
    vf = (f"fps={FPS},scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1,"
          f"tpad=stop_mode=clone:stop_duration={frames / FPS:.3f}")
    if srt is not None:
        vf += f",subtitles='{srt}':force_style='{SUB_STYLE}'"
    vf += ",format=yuv420p"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-an", "-vf", vf,
                    "-frames:v", str(frames), "-r", str(FPS), "-c:v", *VIDEO_CODEC,
                    "-b:v", VIDEO_BITRATE, "-threads", "4", str(out)], check=True)


def _segments(p: dict, b0: int, b1: int, anims: list[dict]) -> list[tuple]:
    """A picture's frames [b0, b1) as ('still'|'anim', f0, f1, shot) pieces:
    its animated shots where they fall, the still around them."""
    out, f = [], b0
    # by time as well as by picture: Peter's talking shot started 0.01 s before the next picture (the
    # paragraph it opens) and was dropped by both, leaving that picture's old still for 16 s (10-17)
    mine = [a for a in anims if a["pic"] == p["id"] or b0 <= round(a["t"] * FPS) < b1]
    for a in sorted(mine, key=lambda a: a["t"]):
        a0 = max(b0, round(a["t"] * FPS))
        a1 = min(b1, round((a["t"] + a["dur"]) * FPS))
        if a1 - a0 < FPS // 2 or a0 < f:
            continue
        if a0 > f:
            out.append(("still", f, a0, None))
        out.append(("anim", a0, a1, a))
        f = a1
    if f < b1:
        out.append(("still", f, b1, None))
    # a sliver of still between shots (< 1 s) becomes part of the shot before
    merged = []
    for seg in out:
        if merged and seg[0] == "still" and seg[2] - seg[1] < FPS and merged[-1][0] == "anim":
            k, f0, _, a = merged[-1]
            merged[-1] = (k, f0, seg[2], a)
        else:
            merged.append(seg)
    return merged


def build(pics: list[dict], img_dir: Path, audio: Path, timeline: list[dict], out: Path,
          work: Path, subtitles: bool = True, workers: int = 6, log=print,
          anims: list[dict] | None = None, frames: list[dict] | None = None) -> Path:
    """*frames*: [{t, pic, image}] - the newer pictures of moments (keyframes, talking portraits): a stretch
    of a picture with no animated shot shows the latest of these at or before it instead of the old scene
    still (10-17: the gaps showed two-day-old pictures, a bearded Roylott)."""
    clips = work / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    total = audio_seconds(audio)
    cues = subtitle_cues(timeline, total) if subtitles else []
    # Frame-exact: each clip ends where the next starts, so 50 clips don't drift.
    bounds = [round(p["t0"] * FPS) for p in pics] + [round(total * FPS)]
    # a shot running across a picture change plays to its end and the next picture starts after it: Peter's
    # talking shot was cut 2 s in at the paragraph that brings in Matthias, whose still then took his
    # words for 14 s (10-17)
    for a in anims or []:
        a0, a1 = round(a["t"] * FPS), round((a["t"] + a["dur"]) * FPS)
        for n in range(1, len(pics)):
            if a0 < bounds[n] < a1 and a1 < bounds[n + 1]:
                bounds[n] = a1
    tasks, n_anim = [], 0
    for n, p in enumerate(pics):
        if bounds[n + 1] - bounds[n] <= 0:
            continue
        segs = _segments(p, bounds[n], bounds[n + 1], anims or [])
        for k, (kind, f0, f1, a) in enumerate(segs):
            name = f"{p['id']}-{k}"
            srt = None
            if subtitles:
                srt = clips / f"{name}.srt"
                write_srt(cues, srt, f0 / FPS, f1 / FPS)
            if kind == "anim":
                n_anim += 1
                tasks.append((_anim_clip, (Path(a["out"]), f1 - f0, srt, clips / f"{name}.mp4")))
            else:
                img = img_dir / f"{p['id']}.png"
                newer = [x for x in (frames or []) if x["pic"] == p["id"] and x["t"] * FPS <= f0 + FPS // 2
                         and Path(x["image"]).exists()]
                if newer:
                    img = Path(max(newer, key=lambda x: x["t"])["image"])
                tasks.append((_clip, (n, img, f1 - f0, srt, clips / f"{name}.mp4",
                                      k == 0, k == len(segs) - 1)))
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for f in [ex.submit(fn, *args) for fn, args in tasks]:
            f.result()
    log(f"  rendered {len(tasks)} clip(s)" + (f" ({n_anim} animated)" if n_anim else "")
        + f" in {time.time() - t:.0f} s")
    outs = [args[-3] if fn is _clip else args[-1] for fn, args in tasks]
    lst = clips / "list.txt"
    lst.write_text("".join(f"file '{o}'\n" for o in outs))
    tmp = out.with_name(f".{out.name}")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                    "-f", "mp4", str(tmp)], check=True)
    tmp.replace(out)
    for o in outs:
        o.unlink(missing_ok=True)
    log(f"  wrote {out}")
    return out


def make_video(text: str, audio: Path, timeline_path: Path, out: Path, work: Path,
               llm_url: str, llm_model: str = "default", style: str = "",
               subtitles: bool = True, log=print, animated: bool = False) -> Path:
    """Plan (if not yet), draw (what's missing), build."""
    timeline = json.loads(timeline_path.read_text())
    total = audio_seconds(audio)
    plan = ensure_plan(text, work, llm_url, llm_model, style or DEFAULT_STYLE, total, log=log)
    pics = timed(plan, text, timeline, total)
    draw_checked(pics, work / "img", plan, style or DEFAULT_STYLE, llm_url, llm_model, log=log)
    manifest = work / "anim" / "manifest.json"
    anims = json.loads(manifest.read_text()) if animated and manifest.exists() else []
    stills = work / "anim" / "stills.json"
    frames = json.loads(stills.read_text()) if animated and stills.exists() else None
    return build(pics, work / "img", audio, restore_text(timeline, text), out, work,
                 subtitles=subtitles, log=log, anims=anims, frames=frames)
