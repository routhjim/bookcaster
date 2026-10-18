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
ART_GPU = os.environ.get("BOOKCASTER_ART_GPU") or hip_index(os.environ.get("BOOKCASTER_ART_BUS", "65"))
ILLUSTRATE = Path(__file__).resolve().parents[1] / "tools" / "illustrate.py"
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


# Words an image model takes the wrong way in a character's reference
# portrait (seen: a black veil + greying hair drew a nun; "yellowed skin"
# drew a yellow man).
LITERAL_FIXES = [
    (r"\byellow(?:ed|ish)?(\s+(?:and\s+\w+\s+)?(?:skin|complexion|face))", r"sallow\1"),
    (r"\b(?:green|grey|gray)(?:ish)?(\s+(?:skin|complexion))", r"pale\1"),
    (r",?\s*(?:with\s+|and\s+)?(?:a\s+)?(?:heavy\s+|thick\s+|long\s+)?(?:black\s+)?"
     r"(?:veil|hood|wimple|cowl)[^,;.]*", ""),
]


# Words in a scene that the image model reads in the modern sense: a "trap"
# (a pony cart) came out as a motor car.
SCENE_FIXES = [
    (r"\b(?<!horse-drawn )(trap|dog-cart|gig|brougham|landau|hansom(?: cab)?|carriage)\b",
     r"horse-drawn \1 (pulled by a horse, 19th century)"),
    # "Helen raises her veil" drew her veiled like a nun again
    (r"\b(?:raises|lifts|lowers|throws back|pushes back|draws back|removes)\s+(?:her|his|the)\s+"
     r"(?:\w+\s+)?veil,?\s*(?:revealing|to reveal|showing)?\s*", "shows "),
    # a close-up of a wrist or hand drew it with no arm, or a second hand
    (r"\b(?:extreme\s+)?close-?up\s+(?:on|of)\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*)?)'s\s+"
     r"(?:\w+\s+)?(hands?|wrists?|arms?|fingers)\b", r"Medium shot of \1, showing \1's \2"),
    (r"\b(?:heavily\s+|thickly\s+)?veiled\b(?:\s+in\s+black)?", "dressed in black"),
    (r",?\s*(?:with\s+|and\s+|in\s+)?(?:a\s+|her\s+|the\s+)?(?:\w+\s+)?(?:veil|wimple|cowl)\b", ""),
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
        changed = []
        for p in plan["pictures"]:       # the prompt rules may have changed since: redraw what differs
            if "who" in p:
                p["who"] = present(p["who"], p["prompt"], plan["look"].get("characters", {}))
                p["full_prompt"] = full_prompt(p["prompt"], plan["look"], style, p["who"], p.get("where", ""),
                                               p.get("crowd", ""))
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
    for n, p in enumerate(plan["pictures"]):
        p.update(id=f"p{n + 1:03d}", seed=1000 + n,
                 full_prompt=full_prompt(p["prompt"], plan.get("look", {}), style,
                                         p.get("who"), p.get("where", ""), p.get("crowd", "")))
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


def registry_entry(reg: dict, kind: str, name: str) -> str | None:
    """The stored look for *name* under any spelling; the first stored wins."""
    c = canon(name)
    for k, v in reg.get(kind, {}).items():
        if canon(k) == c:
            return k
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
    mine = plan.setdefault("look", {}).setdefault("characters", {})
    changed = False
    for name in mine:
        key = registry_entry(reg, "characters", name)
        if key is not None and mine[name] != reg["characters"][key]:
            mine[name] = reg["characters"][key]
            changed = True
    if changed:
        for p in plan["pictures"]:
            if "who" in p:
                p["full_prompt"] = full_prompt(p["prompt"], plan["look"], style,
                                               p["who"], p.get("where", ""), p.get("crowd", ""))
    return changed


def keep_looks(plan: dict, registry: Path) -> None:
    """A character or place keeps the look it was first given in this book:
    a later chapter's plan reuses it (so its reference picture is reused too)."""
    reg = json.loads(registry.read_text()) if registry.exists() else {
        "version": LOOK_VERSION, "characters": {}, "places": {}}
    for kind in ("characters", "places", "props"):       # props: keyframes' objects (the safe, the band)
        reg.setdefault(kind, {})
        mine = plan["look"].setdefault(kind, {})
        for name in list(mine):
            key = registry_entry(reg, kind, name)
            if key is not None:
                mine[name] = reg[kind][key]
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


def looks(plan: dict) -> tuple[dict, dict]:
    """(characters, places) from a plan; older plans had one flat dict."""
    look = plan.get("look", {}) or {}
    if "characters" in look or "places" in look:
        return dict(look.get("characters") or {}), dict(look.get("places") or {})
    chars = {k: v for k, v in look.items() if PERSON.search(str(v))}
    return chars, {k: v for k, v in look.items() if k not in chars}


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
                where: str = "", crowd: str = "") -> str:
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


def ref_prompts(plan: dict, style: str) -> dict[str, tuple[str, list[int]]]:
    """name -> (prompt, [w, h]) for every reference the plan's scenes use."""
    chars, places = looks(plan)
    used = set()
    for p in plan["pictures"]:
        if "who" in p:
            used.update(p["who"] + ([p["where"]] if p.get("where") else []))
        else:
            used.update(mentions(p["prompt"], {**chars, **places}))
    out = {}
    for name, desc in chars.items():
        if name in used:
            # A full-length figure alone left the face ~20 px tall once shrunk
            # to a reference, so every close-up invented a new face.
            out[name] = (f"Character sheet of one person, two views side by side on a plain "
                         f"light background: on the left a large head-and-shoulders portrait "
                         f"facing the viewer, the face filling the frame; on the right the same "
                         f"person full length, standing. {str(desc).rstrip('.')}. Even light. "
                         f"{style}. No text, no lettering, no signature.", [1280, 896])
    for name, desc in places.items():
        if name in used:
            out[name] = (f"Establishing view of {name}, no people. {str(desc).rstrip('.')}. "
                         f"{style}. No text, no lettering, no signature.", [1024, 576])
    return out


def reference_jobs(plan: dict, style: str, refs_dir: Path) -> list[dict]:
    """One picture per recurring character (full length, plain background)
    and place (empty establishing view) that the scenes use: drawn first,
    then handed to every scene they appear in, so faces, clothes and rooms
    stay the same. Kept per BOOK (refs_dir), reused across chapters."""
    return [{"id": f"ref_{slug(n)}", "size": size, "seed": 7, "prompt": prompt,
             "path": str(ref_path(refs_dir, n, prompt))}
            for n, (prompt, size) in ref_prompts(plan, style).items()]


def scene_jobs(pics: list[dict], plan: dict, img_dir: Path, refs_dir: Path,
               max_refs: int = 3, style: str = "") -> list[dict]:
    """Each scene with the references of who and where it shows: characters
    first (up to *max_refs*), then one place if there's room."""
    chars, places = looks(plan)
    jobs = []
    for p in pics:
        if "who" in p:          # the plan names them (and the prompt numbers the images)
            who, where = p["who"][:max_refs], [p["where"]] if p.get("where") else []
        else:
            who = mentions(p["prompt"], chars)[:max_refs]
            where = mentions(p["prompt"], places)[:max(0, max_refs - len(who))][:1]
        prompts = ref_prompts(plan, style)
        refs = [str(ref_path(refs_dir, n, prompts[n][0])) for n in who + where if n in prompts]
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
            [*memory_scope(), str(ART_PYTHON), "-I", str(ILLUSTRATE), str(jobs_file), str(img_dir)],
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
                log(f"    {msg['phase']}: {msg['seconds']} s")
            if "id" in msg:
                done += 1
                if done == 1 or done % 10 == 0 or done == n_jobs:
                    log(f"    drew {done}/{n_jobs} ({msg['seconds']} s each)")
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
        jobs = (reference_jobs(plan, style, refs_dir)
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


class _DrawSession:
    """illustrate.py --serve for a run of draw_jobs calls: the model loads once.
    A run that stalls (no picture for DRAW_STALL_S) or dies is started again
    once, carrying on with the pictures not saved yet; if that one fails too,
    the session is marked broken and draw_jobs goes back to one run per call
    (the old way, which frees the text half before loading the picture half)."""

    def __init__(self, log_file: Path):
        self.log_file, self.proc, self.lines, self.broken = log_file, None, None, False

    def _start(self, log):
        import queue
        err = open(self.log_file, "a")
        self.proc = subprocess.Popen(
            [*memory_scope(), str(ART_PYTHON), "-I", str(ILLUSTRATE), "--serve"],
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
            keep = ("id", "prompt", "seed", "path", "refs", "size")
            for j in left:
                self.proc.stdin.write(json.dumps({k: v for k, v in j.items() if k in keep},
                                                 ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            by_id, done, status = {j["id"]: j for j in left}, 0, "ok"
            while done < len(left):
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
                        on_done(by_id[msg["id"]])
                    if done == 1 or done % 10 == 0 or done == len(left):
                        log(f"    drew {done}/{len(left)} ({msg.get('seconds', '?')} s each)")
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
def drawing_session(log_file: Path, log=print):
    """Keep the drawing model loaded across the draw_jobs calls inside (a
    keyframe phase: draw, check, redraw, then the talking portraits). The music
    server shares its card and stays stopped for the whole session."""
    global _SESSION
    if _SESSION:                     # nested: the outer session serves
        yield
        return
    with _music_paused(log):
        _SESSION = _DrawSession(log_file)
        try:
            yield
        finally:
            _SESSION.stop()
            _SESSION = None


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


def check_picture(png: Path, people: int | None, llm_url: str, model: str) -> list[str]:
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
                                      llm_url, model)
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
