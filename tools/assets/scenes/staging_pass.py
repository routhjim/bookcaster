"""Staging pass: the facts a director's plan leaves out, asked of the language model from the passage (stdlib).

    python tools/assets/scenes/staging_pass.py PLAN.json SHOTS.txt CAST.json CHAPTER.txt OUT_ANSWERS.json
           [--cache DIR] [--url http://127.0.0.1:18100/v1/chat/completions] [--set-dir LIBRARY_SET]

One call per scene (the set; who is in the room at the start; each person's place, posture and worn state), then one
call per shot only where something changes or is missing (an action, a talk shot, a shot whose listed people differ
from the carried state): who is in the room, who moved and where the move ends, worn state, the start pose from the
pose list, what the first picture must show, who looks at whom, the talk camera kind. The model answers from the
passage and the lists only, in a fixed JSON schema; where the passage does not say, it answers "not stated" and
from_plan.py keeps the previous state. Prompts are general: the scene's own text and lists are the only specifics,
passed as data. Answers are cached on disk keyed by the prompt (re-runs are free; --cache to a new dir is a cold run).
Output: {"scene": {...}, "shots": {shot id: {...}}, "_calls": n, "_cached": n} for from_plan.py --pass.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "redraw"))
import config as C  # noqa: E402
import seating  # noqa: E402
import setfacts  # noqa: E402

POSES = ["stand", "stand_relaxed", "stand_hand_on_mantel", "gesture_point", "seated", "sit_armchair_relaxed",
         "veil_lift", "look_at_sleeve", "reach_to_forearm", "look"]
CAMERAS = ["wide", "two_shot", "over_the_shoulder", "single", "close"]
NOT = "not stated"

RULES = (
    "Answer only from the passage and the lists given. Where the passage does not say something, answer \"" + NOT
    + "\" for it; do not invent. Use only the names, places, poses and choices exactly as they appear in the lists. "
    "Answer with the JSON object only.")


def passage(chapter: str, plan: dict, ids: list) -> str:
    """The chapter's text from a little before the scene's first spoken line to just after its last one: the first
    words of the first line and the last words of the last line, found in the chapter by their letters alone (the
    spoken lines leave out the narration between their parts)."""
    es = {e["id"]: e for e in plan.get("_es", [])}
    pos, letters = [], []
    for i, ch in enumerate(chapter):         # letters-and-digits only, with each one's place in the chapter
        if ch.isalnum():
            letters.append(ch.lower())
            pos.append(i)
    flat = "".join(letters)
    key = lambda s, n, last=False: "".join(c.lower() for c in " ".join(
        (re.findall(r"\w+", s)[-n:] if last else re.findall(r"\w+", s)[:n])) if c.isalnum())
    talk = [es[i].get("words") for i in ids if es.get(i, {}).get("words")]
    if not talk:
        raise SystemExit("no spoken line in the scene's shots: the passage cannot be found")
    a = b = -1
    for n in (6, 4, 3):
        a = flat.find(key(talk[0], n)) if a < 0 else a
        k_end = key(talk[-1], n, last=True)
        b = flat.find(k_end, max(a, 0)) + len(k_end) if b < 0 and flat.find(k_end, max(a, 0)) >= 0 else b
    if a < 0 or b < 0:
        raise SystemExit("the scene's spoken lines were not found in the chapter")
    start = chapter.rfind("\n\n", 0, max(0, pos[a] - 1200)) + 2
    end = chapter.find("\n\n", pos[min(b, len(pos) - 1)])
    return chapter[max(0, start):end if end > 0 else len(chapter)]


def set_words(set_dir: Path, library: Path) -> tuple:
    things = [t for t in setfacts.things(set_dir) if t["kind"] in ("fixture", "window", "door")]
    seats = seating.seats(library, set_dir)
    lines = []
    for t in things:
        w = t["words"]
        lines.append(f"{t['name']}: {w}" + (f", {t['place']}" if t.get("place") else "")
                     + (" (a seat)" if t["name"] in seats else ""))
    return things, seats, lines


def cast_words(cast: dict, library: Path) -> list:
    out = []
    for nm, c in cast.items():
        base = c.get("base") or c.get("id")
        variants = sorted({d.name[len(base) + 1:].replace("seated_", "").split("_", 1)[-1]
                           for d in (library / "character").glob(f"{base}_*")
                           if "veil" in d.name or d.name.endswith(("_raised", "_down"))} - {""})
        out.append(f"{nm}: usual clothes" + (f"; worn-state variants: {', '.join(sorted(set(variants)))}" if variants else ""))
    return out


def scene_prompt(psg: str, set_lines: list, sets: list, people: list, shot0: dict) -> str:
    return (
        "You are staging a scene of a story for a film. The passage of the story for this scene:\n<<<\n" + psg
        + "\n>>>\nThe director's description of the place: " + json.dumps({k: shot0.get(k) for k in ("place", "room")})
        + "\nThe sets available (id: description):\n" + "\n".join(sets)
        + "\nThe chosen set's named things (name: what it is, where it stands):\n" + "\n".join(set_lines)
        + "\nThe people (name: worn-state choices):\n" + "\n".join(people)
        + "\nFor the moment the scene begins, give: the set id; for each person whether they are in the room; for "
        "each person in the room, the named thing they are at (a name from the list), whether they sit in it (only if "
        "it is a seat), the side of it they are on (one of: front, left, right, behind, in), the distance (one of: "
        "touching, near, a few steps), their posture (sitting or standing), and their worn state (usual, or one of "
        "their worn-state choices).\n" + RULES + "\nSchema: {\"set\": id, \"people\": {name: {\"in_room\": true or "
        "false or \"" + NOT + "\", \"thing\": name or \"" + NOT + "\", \"sitting\": true or false, \"side\": word, "
        "\"distance\": word, \"posture\": word, \"worn\": word}}}")


def shot_prompt(psg: str, set_lines: list, people: list, shot: dict, state: dict, n: int) -> str:
    return (
        "You are staging a scene of a story for a film. The passage of the story for this scene:\n<<<\n" + psg
        + "\n>>>\nThe set's named things (name: what it is, where it stands):\n" + "\n".join(set_lines)
        + "\nThe people (name: worn-state choices):\n" + "\n".join(people)
        + "\nThe pose list: " + ", ".join(POSES) + "\nThe camera kinds: " + ", ".join(CAMERAS)
        + f"\nShot {n} of the scene, as the director wrote it: "
        + json.dumps({k: shot.get(k) for k in ("kind", "framing", "speaker", "listener", "action", "people", "line",
                                                "things")
                      if shot.get(k)}, ensure_ascii=False)
        + "\nWhere everyone was just before this shot: " + json.dumps(state, ensure_ascii=False)
        + "\nFor the FIRST picture of this shot (the moment before its action happens) give: who is in the room; who "
        "has moved since the shot before and where they are now (a named thing from the list, sitting or not, side, "
        "distance, posture); for a move that happens during this shot, where it ends; each person's worn state; for "
        "each person who acts, the pose from the pose list at the start; which body parts of which people must be "
        "visible for the action to read; whom each person looks at (a name, or the camera); and for a shot with a "
        "speaker, the camera kind from the list.\n" + RULES + "\nSchema: {\"in_room\": [names] or \"" + NOT + "\", "
        "\"moved\": {name: {\"thing\": name, \"sitting\": bool, \"side\": word, \"distance\": word, \"posture\": "
        "word}}, \"move_ends\": {name: {\"thing\": name, \"sitting\": bool, \"side\": word, \"distance\": word, "
        "\"posture\": word}}, \"worn\": {name: word}, \"pose\": {name: pose}, \"must_show\": [[name, body part]], "
        "\"looks_at\": {name: name or \"camera\" or \"" + NOT + "\"}, \"camera\": kind or \"" + NOT + "\"}")


def ask(url: str, prompt: str, cache: Path, max_tokens: int = 900) -> tuple:
    key = hashlib.sha256(prompt.encode()).hexdigest()[:24]
    f = cache / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text()), True
    body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "temperature": 0.3,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": False}}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=1200) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        ans = json.loads(m.group(0)) if m else {"_raw": txt}
    except json.JSONDecodeError:
        ans = {"_raw": txt}
    cache.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(ans, indent=1))
    return ans, False


def needs_call(sh: dict, state_names: set) -> bool:
    listed = {p["name"] for p in sh.get("people") or []}
    return sh.get("kind") in ("action", "talk") or not listed or bool(listed - state_names)


def main():
    a = sys.argv[1:]
    plan = json.loads(Path(a[0]).read_text())
    ids = [x.strip() for x in Path(a[1]).read_text().splitlines() if x.strip()]
    cast = json.loads(Path(a[2]).read_text())
    chapter = Path(a[3]).read_text()
    out = Path(a[4])
    cache = Path(a[a.index("--cache") + 1]) if "--cache" in a else out.parent / "llm_cache"
    url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:18100/v1/chat/completions"
    library = Path(C.LIBRARY)
    set_dir = Path(a[a.index("--set-dir") + 1]) if "--set-dir" in a else None
    sets = []
    for d in sorted((library / "set").iterdir()):
        if (d / "asset.json").exists():
            aj = json.loads((d / "asset.json").read_text())
            sets.append(f"{aj['id']}: {aj.get('description', '')[:300]}")
    psg = passage(chapter, plan, ids)
    people = cast_words(cast, library)
    first = plan[ids[0]][0]
    calls = cached = 0
    t0 = time.time()
    # the set first (its own call when not given), then the scene's start
    if set_dir is None:
        ans, hit = ask(url, scene_prompt(psg, [], sets, people, first), cache)
        calls += 1
        cached += hit
        set_dir = library / "set" / (ans.get("set") if isinstance(ans.get("set"), str) else "")
        if not (set_dir / "spec.json").exists():
            set_dir = library / "set" / sorted(d.name for d in (library / "set").iterdir())[0]
    _, _, set_lines = set_words(set_dir, library)
    scene, hit = ask(url, scene_prompt(psg, set_lines, sets, people, first), cache)
    calls += 1
    cached += hit
    state = {nm: v for nm, v in (scene.get("people") or {}).items() if isinstance(v, dict) and v.get("in_room") is not False}
    shots = {}
    for n, sid in enumerate(ids, 1):
        sh = plan[sid][0]
        if not needs_call(sh, set(state)):
            continue
        ans, hit = ask(url, shot_prompt(psg, set_lines, people, sh, state, n), cache)
        calls += 1
        cached += hit
        shots[sid] = ans
        for nm, mv in list((ans.get("moved") or {}).items()) + list((ans.get("move_ends") or {}).items()):
            if isinstance(mv, dict) and mv.get("thing") not in (None, NOT):
                state[nm] = dict(state.get(nm) or {}, **mv)
        for nm, w in (ans.get("worn") or {}).items():
            if w and w != NOT and nm in state:
                state[nm]["worn"] = w
    res = {"set": set_dir.name, "scene": scene, "shots": shots, "_calls": calls, "_cached": cached,
           "_s": round(time.time() - t0, 1)}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("set", "_calls", "_cached", "_s")}))


if __name__ == "__main__":
    main()
