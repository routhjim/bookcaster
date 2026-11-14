"""A place's spec written by the chapter's directing model (stdlib): what kind of room, its openings, the pieces every
plan phrase needs, their looks and stated relations, in the fixed schema place_spec/1; validated by code and re-asked
with the faults stated (never patched by hand); cached by the prompt.

    python tools/assets/scenes/place_spec.py PLAN SHOTS TIMELINE PLACE OUT.json [--cache DIR] [--url URL]

PLAN: the director's plan; SHOTS: the shot ids (one per line) of the stretch; TIMELINE: the chapter's timeline
(lines with times); PLACE: the plan's place name. Writes OUT.json (the validated spec, or the last answer with its
faults under "_faults") and OUT.prompt.txt.

Schema place_spec/1 (lists in place_kinds.json):
 {"schema", "place", "kind", "period", "size",
  "openings": [{"id", "kind": window|door|fireplace, "wall": north|south|east|west|not stated, "words"}],
  "pieces": [{"id", "kind", "look", "phrases": [plan phrases verbatim], "against": wall|opening:<id>|free|not stated}],
  "relations": [{"a", "rel", "b", "c"?}],
  "named_places": {plan phrase: piece or opening id}}
The model only judges what the text gives; sizes, positions and facings are code's (the layout), defaults of the
room kind are data.

An EXTENSION of the director's set sheet (tools/shot_flow.py SET_ASK / place_sheets on main: {"room", "fixtures",
"crowd"} in shots/scenes/set_<place>.json, read by the keyframe prompts and the hero picture): the same one call
answers those three fields as before plus the place_spec/1 fields, so the sheet's readers see no change; the sheet
written is {"k": <prompt key>, "sheet": {room, fixtures, crowd, ...place_spec/1}}.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
KINDS = json.loads((HERE / "place_kinds.json").read_text())
SEED = 7
MAX_ASKS = 3                                 # the first answer and two re-asks with its faults
NOT = "not stated"


def place_shots(plan: dict, ids: list, place: str) -> list:
    return [s for s in ids if (plan.get(s) or [{}])[0].get("place") == place]


def phrases(plan: dict, ids: list, names: set) -> list:
    """Every place and facing phrase the shots use for a thing (not a person, not the camera), verbatim, in order."""
    out = []
    for s in ids:
        for p in plan[s][0].get("people") or []:
            for ph in (p.get("at"), p.get("facing")):
                if not ph or ph.strip().lower() in ("the camera", "camera"):
                    continue
                words = set(re.findall(r"[a-z]+", ph.lower()))
                if words & {w for n in names for w in re.findall(r"[a-z]+", n.lower()) if len(w) > 3}:
                    continue                     # a person's name: a relation to someone, not a place
                if ph not in out:
                    out.append(ph)
    return out


def passage(timeline: list, plan: dict, ids: list, all_ids: list) -> str:
    """The chapter's lines spanning the place's shots (from the first shot's time to the next shot after the last)."""
    sys.path.insert(0, str(HERE))
    import staging_pass as SP
    es = {e["id"]: e for e in plan.get("_es", [])}
    at = SP.audio_times(timeline, es, all_ids)
    t0 = min(at[s] for s in ids)
    nxt = [at[s] for s in all_ids if at[s] > max(at[x] for x in ids)]
    t1 = min(nxt) if nxt else max(at[s] for s in ids) + 10.0
    return SP.span(timeline, t0 - 2.0, t1)


def prompt(place: str, phr: list, actions: list, psg: str) -> str:
    k = KINDS
    return (
        "You describe one place of a story so that it can be built as a film set. Answer from the passage and the "
        "shot lines only; where they do not say, write \"not stated\" (never invent story facts).\n"
        f"Place name: {place}\n"
        "Passage set in this place:\n<<<\n" + psg + "\n>>>\n"
        "The film's shot lines in this place:\n" + "\n".join(f"- {a}" for a in actions) + "\n"
        "Place phrases the shots use (each must be answered by one piece or opening in named_places, verbatim):\n"
        + "\n".join(f"- {p}" for p in phr) + "\n"
        "Write only a JSON object of this shape:\n"
        '{"room": <two sentences: the FIXED things of this place and where each stands; never a person, nothing the '
        'story\'s people carry; no weather or sky; nothing seen outside it>, '
        '"fixtures": [<2 to 5 short names of things built into this place a viewer must see in every wide picture>], '
        '"crowd": <the unnamed people always there, or "" when none>, '
        '"schema": "place_spec/1", "place": <the place name>, "kind": <one of ROOM_KINDS>, '
        '"period": <one of PERIODS or "not stated">, "size": <one of SIZES>, '
        '"openings": [{"id": "o1", "kind": <window|door|fireplace>, "wall": <north|south|east|west|not stated>, '
        '"words": <the text\'s words for it or "">}], '
        '"pieces": [{"id": "p1", "kind": <one of PIECE_KINDS>, "look": <one line: material, colour, condition, style>, '
        '"phrases": [<the place phrases this piece answers to, verbatim>], "against": <wall|opening:<id>|free|not stated>}], '
        '"relations": [{"a": <id>, "rel": <one of RELATIONS>, "b": <id>}], '
        '"named_places": {<each place phrase verbatim>: <a piece or opening id, or "floor">}}\n'
        f"ROOM_KINDS: {', '.join(k['ROOM_KINDS'])}\nPERIODS: {', '.join(k['PERIODS'])}\nSIZES: {', '.join(k['SIZES'])}\n"
        f"PIECE_KINDS: {', '.join(k['PIECE_KINDS'])}\nRELATIONS: {', '.join(k['RELATIONS'])}\n"
        "Include every piece a place phrase or shot line names, and the usual furnishing of such a room. Relations "
        "only where the text states them. A phrase that names the open floor itself and no thing (the middle of "
        "the room, the centre of the floor) has the named_places value \"floor\": no piece is ever made up to stand "
        "for a part of the floor. Otherwise a named_places value is always a piece or opening id, never \"not stated\": "
        "when a phrase names a thing whose position the text does not give, list that thing as an opening or piece "
        "anyway and write \"not stated\" only in its wall or against field. Write the JSON only, nothing else.")


def validate(spec, phr: list) -> list:
    """The faults of an answer, in plain words (empty: valid)."""
    if not isinstance(spec, dict):
        return ["the answer is not a JSON object"]
    f = []
    # the set sheet's own fields (shot_flow.place_sheets: room, fixtures, crowd), which the keyframes read as now
    if not str(spec.get("room") or "").strip():
        f.append('"room" (two sentences on the fixed things and where each stands) is missing')
    fx = spec.get("fixtures")
    if not (isinstance(fx, list) and 2 <= len(fx) <= 5 and all(isinstance(x, str) for x in fx)):
        f.append('"fixtures" must be a list of 2 to 5 short names')
    if not isinstance(spec.get("crowd", ""), str):
        f.append('"crowd" must be a string ("" when none)')
    for key, lst in (("kind", KINDS["ROOM_KINDS"]), ("size", KINDS["SIZES"])):
        if spec.get(key) not in lst:
            f.append(f'"{key}" must be one of: {", ".join(lst)}')
    if spec.get("period") not in KINDS["PERIODS"] + [NOT]:
        f.append('"period" must be one of PERIODS or "not stated"')
    ids = {}
    for o in spec.get("openings") or []:
        if not isinstance(o, dict) or o.get("kind") not in KINDS["OPENINGS"]:
            f.append(f"opening {o!r}: kind must be window, door or fireplace")
            continue
        if o.get("wall", NOT) not in KINDS["WALLS"]:
            f.append(f'opening {o.get("id")}: wall must be one of {", ".join(KINDS["WALLS"])}')
        ids[o.get("id")] = "opening"
    for p in spec.get("pieces") or []:
        if not isinstance(p, dict) or p.get("kind") not in KINDS["PIECE_KINDS"]:
            f.append(f"piece {p.get('id') if isinstance(p, dict) else p!r}: kind {p.get('kind') if isinstance(p, dict) else ''!r} "
                     "is not in PIECE_KINDS")
            continue
        if not (p.get("look") or "").strip():
            f.append(f"piece {p.get('id')}: no look line")
        a = p.get("against", NOT)
        if not (a in KINDS["AGAINST"] or (isinstance(a, str) and a.startswith("opening:"))):
            f.append(f"piece {p.get('id')}: against must be wall, opening:<id>, free or not stated")
        if p.get("id") in ids:
            f.append(f"id {p.get('id')} is used twice")
        ids[p.get("id")] = "piece"
    for r in spec.get("relations") or []:
        if not isinstance(r, dict) or r.get("rel") not in KINDS["RELATIONS"]:
            f.append(f"relation {r!r}: rel must be one of RELATIONS")
            continue
        for x in ("a", "b") + (("c",) if r.get("rel") == "between" else ()):
            if r.get(x) not in ids:
                f.append(f"relation {r!r}: {x} is not a known piece or opening id")
    np = spec.get("named_places") or {}
    for ph in phr:
        if ph not in np:
            f.append(f'the place phrase "{ph}" is missing from named_places')
        elif np[ph] not in ids and np[ph] != "floor":
            f.append(f'named_places["{ph}"] is {np[ph]!r}, not a known piece or opening id: list the thing this '
                     'phrase names as an opening or piece (its position may be "not stated") and map the phrase to its id')
    return f


def usual_missing(spec: dict) -> list:
    """The usual pieces of the spec's room kind that it does not have (the layout helper's dressing list)."""
    sys.path.insert(0, str(HERE.parent / "sets"))
    try:
        import place_layout as L
        return L.missing_dressing(spec)
    except (KeyError, ImportError, OSError, ValueError):
        return []


def ask(url: str, text: str, cache: Path) -> str:
    key = hashlib.sha256(f"place_spec/1\n{SEED}\n{text}".encode()).hexdigest()[:24]
    fp = cache / f"{key}.txt"
    if fp.exists():
        return fp.read_text()
    body = {"model": "default", "messages": [{"role": "user", "content": text}], "temperature": 0.0, "seed": SEED,
            "max_tokens": 2500, "chat_template_kwargs": {"enable_thinking": False}}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=900) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    cache.mkdir(parents=True, exist_ok=True)
    fp.write_text(txt)
    return txt


def parse(txt: str):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


def write(plan: dict, all_ids: list, timeline: list, place: str, url: str, cache: Path) -> dict:
    ids = place_shots(plan, all_ids, place)
    names = {p["name"] for s in all_ids for p in plan[s][0].get("people") or []}
    phr = phrases(plan, ids, names)
    acts = [plan[s][0].get("action") or "" for s in ids if plan[s][0].get("action")]
    base = prompt(place, phr, acts, passage(timeline, plan, ids, all_ids))
    text, spec, faults, asks, advised = base, None, ["no answer"], 0, False
    while asks < MAX_ASKS:
        asks += 1
        spec = parse(ask(url, text, cache))
        faults = validate(spec, phr)
        usual = [] if advised or not isinstance(spec, dict) else usual_missing(spec)
        if not faults and not usual:
            break
        # the director re-ask: the same prompt, the faults in plain words; once, the room kind's usual pieces it left out
        text = base + "\n\nYour previous answer:\n" + json.dumps(spec) + "\n"
        if faults:
            text += "It had these faults; write the whole JSON again with them fixed:\n" + "\n".join(f"- {x}" for x in faults) + "\n"
        if usual:
            advised = True
            text += ("A room of this kind usually also has: " + ", ".join(usual) + ". Add those that suit this place "
                     "(as pieces with a look and against, no phrases needed) and write the whole JSON again.\n")
    out = dict(spec or {}, _prompt_sha=hashlib.sha256(base.encode()).hexdigest()[:16], _asks=asks,
               _phrases=phr, _shots=ids)
    # the director's own set-sheet line for the place, as the plan carries it (what lies outside a vehicle's cabin
    # is said there and nowhere else)
    out["_director_room"] = next((plan[i][0].get("room") for i in ids if plan[i][0].get("room")), "")
    out["_director_crowd"] = next((plan[i][0].get("crowd") for i in ids if plan[i][0].get("crowd")), "")
    if faults:
        out["_faults"] = faults
    return out, base


def main():
    a = sys.argv[1:]
    plan = json.loads(Path(a[0]).read_text())
    ids = [x.strip() for x in Path(a[1]).read_text().splitlines() if x.strip()]
    tl = json.loads(Path(a[2]).read_text())
    tl = tl if isinstance(tl, list) else tl.get("segments") or tl.get("lines") or []
    place, out = a[3], Path(a[4])
    cache = Path(a[a.index("--cache") + 1]) if "--cache" in a else out.parent / "place_cache"
    url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:18100/v1/chat/completions"
    spec, base = write(plan, ids, tl, place, url, cache)
    out.write_text(json.dumps(spec, indent=1))
    out.with_suffix(".prompt.txt").write_text(base)
    print(json.dumps({"place": place, "asks": spec["_asks"], "faults": spec.get("_faults", []),
                      "pieces": len(spec.get("pieces") or []), "phrases": len(spec["_phrases"])}))
    return 0 if not spec.get("_faults") else 1


if __name__ == "__main__":
    sys.exit(main())
