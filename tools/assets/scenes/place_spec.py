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
  "pieces": [{"id", "kind", "look", "phrases": [plan phrases verbatim], "against": wall|opening:<id>|free|not stated,
              "gives_light"?: true (it burns or glows), "longest_m"?: m (a landmark, a work_piece)}],
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
        '"outside": <one phrase: what is seen outside this place, through its windows or canopy, as the text and the '
        'shot lines tell it; "" when they say nothing of it>, '
        '"schema": "place_spec/1", "place": <the place name>, "kind": <one of ROOM_KINDS>, '
        '"period": <one of PERIODS or "not stated">, "size": <one of SIZES>, '
        '"openings": [{"id": "o1", "kind": <window|door|fireplace>, "wall": <north|south|east|west|not stated>, '
        '"words": <the text\'s words for it or "">}], '
        '"pieces": [{"id": "p1", "kind": <one of PIECE_KINDS>, "look": <one line: material, colour, condition, style>, '
        '"phrases": [<the place phrases this piece answers to, verbatim>], "against": <wall|opening:<id>|free|not stated>, '
        '"gives_light"?: true, "longest_m"?: <metres>, "hangs_m"?: <metres>}], '
        '"relations": [{"a": <id>, "rel": <one of RELATIONS>, "b": <id>}], '
        '"named_places": {<each place phrase verbatim>: <a piece or opening id, or "floor">}}\n'
        f"ROOM_KINDS: {', '.join(k['ROOM_KINDS'])}\nPERIODS: {', '.join(k['PERIODS'])}\nSIZES: {', '.join(k['SIZES'])}\n"
        f"PIECE_KINDS: {', '.join(k['PIECE_KINDS'])}\nRELATIONS: {', '.join(k['RELATIONS'])}\n"
        "Include every piece a place phrase or shot line names, and the usual furnishing of such a room. A piece is "
        "the very thing the words name, never a piece made for another use standing in for it: a thing of the place "
        "that the shot lines have people work on, at or with (what the work the shots show is done on, at or in), "
        "when no kind of PIECE_KINDS is that thing itself, has the kind \"work_piece\", with \"longest_m\": its "
        "longest side in metres, and its look begins with the thing's own plain name (the word the shot lines use "
        "for it), then what it is and what it is used for in a few plain words, then the shape it truly has in that "
        "period, told as one who has worked with such a thing would tell someone who must build it having never seen "
        "one (its real outline, its working parts and how they join, how it stands or hangs where it is used; never "
        "a plain box unless it is one), then its material, colour and condition. The loose thing that work is done on (the stuff or object the work changes: held, struck, heated, "
        "cut or carried, whatever the text calls it) is "
        "never a piece: a phrase that names it has the named_places value of the piece it lies on. A fire that work "
        "is done at (where things are heated, cooked or made) is a piece of kind \"work_piece\" standing where it is "
        "used, never an opening: the opening \"fireplace\" is only a room's own fire for warmth, built into its wall. A piece or opening with a fire or a flame burning in it, or a "
        "lamp, has \"gives_light\": true; a thing that only shines, gleams or reflects light does not. A thing fixed up on a wall or "
        "over an opening, that does not stand on the floor, has \"hangs_m\": the height of its lowest point above "
        "the floor in metres, and \"against\": the wall or the opening it is on or over. Relations "
        "only where the text states them. A phrase that names the open floor itself and no thing (the middle of "
        "the room, the centre of the floor) has the named_places value \"floor\": no piece is ever made up to stand "
        "for a part of the floor. Otherwise a named_places value is always a piece or opening id, never \"not stated\": "
        "when a phrase names a thing whose position the text does not give, list that thing as an opening or piece "
        "anyway and write \"not stated\" only in its wall or against field. "
        "A place in the open air (no walls round it, the sky overhead) has the kind \"outdoors\"; then: \"openings\" "
        "is []; \"outside\" is \"\"; \"room\" tells the ground and the land round it; every piece's against is "
        "\"free\"; the pieces are only the separate solid things standing on the ground near the people that a place "
        "phrase or shot line names (the kind landmark, with \"longest_m\": its longest side in metres, for a thing "
        "nobody sits or lies on), never furnishing; the ground itself, water, slopes, paths, fields, the sky and "
        "anything far off are the land, not pieces, and a phrase that names one of them has the named_places value "
        "\"floor\", as has a phrase that only gives a direction or a side of the picture; when such a phrase names a particular part of the land (not the open ground in general, not a "
        "direction), add it to \"land_places\": {<the phrase verbatim>: <a few plain words for that part of the land "
        "as it would be seen in a picture>}. Write the JSON only, nothing else.")


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
    if spec.get("kind") == "outdoors":              # a place in the open: no walls, nothing against one
        if spec.get("openings"):
            f.append('a place of the kind "outdoors" has no openings: "openings" must be []')
        for p in spec.get("pieces") or []:
            if isinstance(p, dict) and p.get("against") not in ("free", NOT):
                f.append(f'piece {p.get("id")}: a place of the kind "outdoors" has no walls: against must be "free"')
    lp = spec.get("land_places", {})
    if not isinstance(lp, dict) or not all(isinstance(v, str) and v.strip() for v in lp.values()):
        f.append('"land_places" must be an object {phrase: a few words for that part of the land}')
    else:
        for ph in lp:
            if (spec.get("named_places") or {}).get(ph) != "floor":
                f.append(f'land_places["{ph}"]: only a phrase whose named_places value is "floor" is a part of the land')
    for p in spec.get("pieces") or []:
        if isinstance(p, dict) and p.get("kind") in ("landmark", "work_piece"):
            lm = p.get("longest_m")
            if not (isinstance(lm, (int, float)) and 0.3 <= lm <= 80):
                f.append(f'piece {p.get("id")}: a {p["kind"]} needs "longest_m", its longest side in metres (0.3 to 80)')
    for p in spec.get("pieces") or []:
        if isinstance(p, dict) and "hangs_m" in p and not (isinstance(p["hangs_m"], (int, float)) and 0 <= p["hangs_m"] <= 12):
            f.append(f'piece {p.get("id")}: "hangs_m" is a height in metres (0 to 12), or left out')
    for x in (spec.get("pieces") or []) + (spec.get("openings") or []):
        if isinstance(x, dict) and "gives_light" in x and not isinstance(x["gives_light"], bool):
            f.append(f'{x.get("id")}: "gives_light" is true or left out')
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


VIEWS_ASK = """A story has these shots, in order, in the place "{place}". What is seen outside this place, through its
windows or canopy, is in general: {outside}.

For each shot write what is seen outside AT THAT MOMENT of the story, as the director's words for the shots tell it.
Each description is given, alone, to a painter who knows nothing else of the story and paints one picture from it,
so it must be whole in itself: first what surrounds the place there (the kind of surroundings it is in, and what
fills the rest of the view: its sky or its dark, its ground or the lack of any), then what is in view, how near it
has come and how large it stands, and the light and its colour. Where the place is not on a ground (in the air, on
open water, in space) say so in plain words: nothing lies under the things in view and there is no horizon. Say every thing by what it is and looks like, never
by a name the story gave it or by a likeness to something else that the painter would paint instead. 20 to 40 words
each, nothing about the people or the inside of the place or its windows. When nothing outside has changed since the
shot before, repeat that shot's words exactly, letter for letter. Use at most {most} different descriptions in all.

The shots:
{shots}

Answer with one JSON object and nothing else: {{"<shot id>": "<what is seen outside>", ...}} with every shot id."""
VIEWS_MOST = 8


def views(plan: dict, ids: list, place: str, outside: str, url: str, cache: Path) -> dict:
    """{shot id: what is seen outside in that shot}: the director's own account of how the view changes over the
    story (a ship's canopy showed one fixed picture from the first shot to the last, 11-14). Asked once more with
    its faults when a shot is missing or the descriptions are too many; {} when it still is not usable."""
    lines = "\n".join(f"{i}: {(plan[i][0].get('setting') or plan[i][0].get('action') or '').strip() or '(no words)'}"
                      for i in ids)
    base = VIEWS_ASK.format(place=place, outside=outside, most=VIEWS_MOST, shots=lines)
    text = base
    for n in range(2):
        got = parse(ask(url, text, cache))
        bad, many = [], False
        if not isinstance(got, dict):
            bad.append("the answer was not one JSON object")
        else:
            miss = [i for i in ids if not str(got.get(i) or "").strip()]
            if miss:
                bad.append("these shot ids have no description: " + ", ".join(miss[:8]))
            many = len({str(v).strip() for v in got.values()}) > VIEWS_MOST
        # (too many descriptions is asked about once, then taken as it is: a whole answer one view over the count
        # was thrown away twice and the canopy kept its single picture, 11-14)
        if not bad and (not many or n):
            return {i: str(got[i]).strip().rstrip(".") for i in ids}
        if many:
            bad.append(f"more than {VIEWS_MOST} different descriptions")
        text = base + "\n\nYour previous answer:\n" + json.dumps(got) + "\nIt had these faults; answer again:\n" + \
            "\n".join(f"- {b}" for b in bad)
    return {}


THINGS_ASK = """A story is told below, whole. Then come its shots in the place "{place}", in order, each with the
people in it, what happens, and what the shot's plan has each person holding.

For each shot and each person in it, list what that person has ON or CARRIES at that moment beyond their everyday
clothes: anything worn on the head or face, on the hands or feet, or put on over the clothes for what they are
doing, anything they hold in a hand and could carry away with them, and the loose thing their work in the shot is
done on (lying in front of them on what they work at, or gripped in a tool), as the story has it at that moment. A
part of the place itself that they grip or work (a lever, a wheel, a rail, a panel) is not listed. Take it from what the story tells and from what follows
from it: a thing the story has someone take off was on them until then, a thing it has them put on is on them from
then. For each thing give:
  "what": one short plain description of the thing, in the same words every time it is the same thing;
  "on": exactly one of head, face, hands, feet, body (where it is worn), held (carried in a hand) or worked (the
  thing being worked on, lying on what they work at or gripped in a tool);
  "goes_on": for a thing made to be worn, where it is worn when it is on (head, face, hands, feet or body), even
  while it is only being carried; none for a thing that is never worn;
  "size_cm": its longest side in centimetres, a number.
An empty list when the person has nothing on or in hand beyond their everyday clothes.

The story:
<<<
{story}
>>>

The shots:
{shots}

Answer with one JSON object and nothing else:
{{"<shot id>": {{"<person>": [{{"what": "...", "on": "...", "goes_on": "...", "size_cm": 0}}, ...]}}, ...}} with every
shot id."""
THING_ON = ("head", "face", "hands", "feet", "body", "held", "worked")
PART_ASK = ("In a scene this happens: {act}\nA prop list for it names this as a thing in a person's hand: {what}\n"
            "Is it a whole thing of its own that a person can pick up and carry away; or is it a part of something "
            "bigger that stays in its place? Answer with exactly one word: whole or part.")
LOOK_ASK = """A story is told below, whole.

The story:
<<<
{story}
>>>

A prop maker must make this one thing from the story as an object on its own: {what}
Describe in one sentence how it looks as the story has it in the place "{place}": its shape and build, what it is
made of, its colours and its marks. Describe the thing alone: never what it lies, rests, stands or hangs on, never
who holds it or what is done with it, never the place around it. Use plain words for what is to be seen (a name
the story itself gives a material or a thing says nothing to a prop maker). Answer with the sentence and nothing else."""


def things(plan: dict, ids: list, place: str, timeline: list, url: str, cache: Path) -> dict:
    """{shot id: {person: [{"what", "on", "size_cm"}]}}: the director's account, from the whole story, of what each
    person has on or carries beyond their everyday clothes (a pilot flew bare-headed in every shot: the story only
    says he pulled his helmet off after landing). {} when the answer is not usable."""
    sys.path.insert(0, str(HERE))
    import staging_pass as SP
    lines = "\n".join(
        f"{i}: " + "; ".join(p["name"] + (f" (holding: {p['holds']})" if str(p.get("holds") or "").strip() else "")
                             for p in plan[i][0].get("people") or [])
        + f" | {(plan[i][0].get('action') or '').strip() or '(no words)'}" for i in ids)
    try:
        got = parse(ask(url, THINGS_ASK.format(place=place, story=SP.span(timeline, -1.0, 1e9), shots=lines), cache))
    except OSError:                         # (no director to ask: no account, and the figures' usual look stands)
        return {}
    if not isinstance(got, dict):
        return {}
    out = {}
    for i in ids:
        row = got.get(i) if isinstance(got.get(i), dict) else {}
        who = {p["name"] for p in plan[i][0].get("people") or []}
        out[i] = {}
        for n, v in row.items():
            ok = [{"what": str(t["what"]).strip(), "on": str(t.get("on")).strip().lower(),
                   "goes_on": str(t.get("goes_on") or "").strip().lower() if str(t.get("goes_on") or "").strip().lower()
                   in THING_ON[:5] else (str(t.get("on")).strip().lower() if str(t.get("on")).strip().lower() in THING_ON[:5] else None),
                   "size_cm": float(t["size_cm"]) if isinstance(t.get("size_cm"), (int, float)) and 2 <= t["size_cm"] <= 200
                   else None}
                  for t in (v if isinstance(v, list) else []) if isinstance(t, dict) and str(t.get("what") or "").strip()]
            ok = [t for t in ok if t["on"] in THING_ON]
            if n in who and ok:
                out[i][n] = ok
    # a held thing is asked about on its own: a part of something fixed is not a prop (the prompt rules it out and
    # a lever's handle was listed all the same, and made as a bell, 11-18); and how the thing looks alone, from the
    # story (a look that named what the thing lay on was modelled with the block under it)
    act = {}
    for i in ids:
        for ts in out[i].values():
            for t in ts:
                if t["on"] in ("held", "worked"):
                    act.setdefault(t["what"], (plan[i][0].get("action") or "").strip())
    try:
        part = {w for w, a in act.items() if (lambda r: "part" in r and "whole" not in r)(
            ask(url, PART_ASK.format(act=a or "(no words)", what=w), cache).lower())}
        look = {w: ask(url, LOOK_ASK.format(story=SP.span(timeline, -1.0, 1e9), what=w, place=place), cache).strip()
                for w in act if w not in part}
        look = {w: v for w, v in look.items() if 8 <= len(v) <= 400 and not set("{}[]\n") & set(v)}   # (a sentence)
    except OSError:
        part, look = set(), {}
    for i in ids:
        for n in list(out[i]):
            out[i][n] = [dict(t, **({"look": look[t["what"]]} if look.get(t["what"]) else {}))
                         for t in out[i][n] if t["what"] not in part]
            if not out[i][n]:
                del out[i][n]
    # a thing a person has in the shot before and in the shot after, the same way, they have in between too (the
    # account left a helmet off for one shot of a flight)
    for a_, b_, c_ in zip(ids, ids[1:], ids[2:]):
        for n in set(out[a_]) & set(out[c_]) & {p["name"] for p in plan[b_][0].get("people") or []}:
            for t in out[a_][n]:
                if any(u["what"] == t["what"] and u["on"] == t["on"] for u in out[c_][n]) and \
                        not any(u["what"] == t["what"] for u in out[b_].get(n) or []):
                    out[b_].setdefault(n, []).append(dict(t))
    return out


def parse(txt: str):
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None


LOOSE_ASK = ("In a scene these things happen: {acts}\nA set builder lists this as a thing of the place: {look}\n"
             "Is it a fixed thing of the place, one that stays where it stands while people work at, on or with it; or "
             "is it the loose thing the work is done to, which is held, moved, changed by the work or carried away? "
             "Answer with exactly one word: fixed or loose.")


def loose(spec: dict, acts: list, url: str, cache: Path) -> list:
    """The work pieces that are the thing worked on, not a thing of the place (the prompt rules it out and the
    director listed it all the same: the smith's bar became a piece standing on the forge floor, 11-17), each asked
    about on its own; -> the faults to put to the director again."""
    out = []
    for p in spec.get("pieces") or []:
        if p.get("kind") != "work_piece":
            continue
        ans = ask(url, LOOSE_ASK.format(acts=" ".join(dict.fromkeys(acts)), look=str(p.get("look") or "").strip()), cache)
        if re.search(r"\bloose\b", ans.lower()) and not re.search(r"\bfixed\b", ans.lower()):
            out.append(f'piece {p["id"]} ({str(p.get("look") or "").split(",")[0].strip()}) is the loose thing work is '
                       "done on, not a thing of the place: leave it out of \"pieces\" and \"relations\", and give each "
                       "phrase that names it the named_places value of the piece it lies on")
    return out


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
        if not faults:
            faults = loose(spec, acts, url, cache)
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
    if str(out.get("outside") or "").strip() and len(ids) > 1:
        out["_views"] = views(plan, ids, place, str(out["outside"]).strip(), url, cache)
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
