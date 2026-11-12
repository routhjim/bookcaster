"""Director's plan -> scene specs that speakers.py stages (stdlib). Mechanics only; judgment is read from answers.

    python tools/assets/scenes/from_plan.py PLAN.json SHOTS.txt CAST.json OUT_DIR [--answers ANSWERS.json]

PLAN.json   the director's plan for a chapter: {shot id: [shot record, ...], ...} (the first record is the shot)
SHOTS.txt   the shot ids of one scene, in story order, one per line
CAST.json   {person name as the plan writes it: {"key": "<scene key>", "id": "<library character id>"} |
             {"key", "base": "<library base id>"} (a seated stance is then chosen per seat by seating.py)}
ANSWERS     the language model's answers to the questions this tool writes (questions.json), keyed by question id;
            where an answer is missing the mechanical default below is used and the shot is marked "default".

What the plan carries per shot (read here): kind, framing (wide / medium / close), the people in frame at the
shot's start with side, posture (sitting / standing), place words ("at") and facing words, the speaker and the
listener of a talk shot, the action words, "tight", and the scene's place and room words.
What it does not carry (questions.json, one per case, general wording, no names or examples): which library set is
the place; which named thing of the set each place's words mean (and which seat); who is in the room but out of the
shot; the people's places in a talk shot (it lists none: they are carried from the shot before); a pose from the
pose list for an action's start; what the key must show (the action's joints); a worn state that differs from the
usual clothes; which library figure each person is.

Mechanical defaults (no answer): the set whose tags and description share most words with the plan's room words;
a place's thing by its own name in the place words (window, desk, fire, door, chair, sofa, a person's name); a
sitting person in the set seat nearest that thing; a standing person on a free floor point near it (0.6-1.6 m,
clear of furniture and of other people, toward the room's middle), facing the named person or thing; everyone who
has appeared in the scene stays where they were last placed; cameras: the plan's framing class on the shot's
subject (the speaker, else the first person), "show" = the people the shot lists; the framing rules re-frame.
Output: <OUT>/<scene>_gNN.json (one staging per run of shots whose people and postures do not change),
<OUT>/plan_facts.json (per shot: the facts read, the defaults used) and <OUT>/questions.json.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "redraw"))
import config as C  # noqa: E402
import seating  # noqa: E402
import setfacts  # noqa: E402

FRAMING = {"wide": "full", "full": "full", "medium": "medium", "close": "close", "two": "medium",
           "over-the-shoulder": "medium"}
FOCAL = {"full": 28, "medium": 50, "close": 85}
STAND_POSE = {"pose": "stand_relaxed", "params": {"weight_on": "l", "chest_deg": 0, "head_deg": 0, "head_tilt": 0,
                                                    "hand_at_lapel": False}}
SEAT_POSE = {"pose": "seated", "params": {"lean_deg": 4, "chin_deg": 2, "knee_spread_m": 0.03, "shin_fwd": 0.12,
                                          "hands": "lap"}}
THING_WORDS = {"window": ("window",), "desk": ("desk",), "fireplace": ("fire", "fireplace", "hearth", "grate"),
               "door": ("door",), "sofa": ("sofa", "settee", "couch"), "bookcase": ("bookcase", "bookshelf", "books"),
               "armchair": ("chair", "armchair")}


# ------------------------------------------------------------------ questions (asked by the scheduler, not here)
def q_set(plan_shot: dict, sets: list) -> dict:
    return {"id": "set", "ask": (
        "A scene of a story takes place here: " + plan_shot.get("place", "") + ". The director describes the room: "
        + plan_shot.get("room", "") + " Its fixed features: " + ", ".join(plan_shot.get("fixtures") or []) + ". "
        "These are the sets the library holds, each with its id and description:\n"
        + "\n".join(f"- {s['id']}: {s['description']}" for s in sets)
        + "\nWhich set fits the scene best? Answer with the set's id only."), "answer": "set id"}


def q_place(qid: str, person_words: str, set_things: list, people: list) -> dict:
    return {"id": qid, "ask": (
        "A room holds these named things, each with where it stands: "
        + "; ".join(f"{t['name']} ({t.get('place') or t['kind']})" for t in set_things)
        + ". The director placed a person: " + person_words + ". Other people placed: " + (", ".join(people) or "none")
        + ". Which named thing is this person at? Are they sitting in it (if it is a seat) or standing beside it, and "
        "on which side of it (front, left, right, behind, as seen from the room's middle)? Answer as JSON: "
        '{"thing": name, "sitting": true or false, "side": word}.'), "answer": "json"}


def q_present(qid: str, shot_words: str, people: list) -> dict:
    return {"id": qid, "ask": (
        "In this shot of a scene: " + shot_words + " These people have appeared in the scene so far: "
        + ", ".join(people) + ". Which of them are still in the room during this shot, shown in it or not? Answer as "
        'a JSON list of their names.'), "answer": "json list"}


def q_change(qid: str, shot: dict, before: dict, poses: list) -> dict:
    return {"id": qid, "ask": (
        f"A shot of a scene: kind {shot.get('kind')}, framing {shot.get('framing')}; the action at the shot: "
        + (shot.get("action") or "(none written)") + " Before this shot the people were: "
        + "; ".join(f"{k}: {v['pose']} {v['at']}" for k, v in before.items())
        + ". At the START of this shot (the moment before the action happens): who has moved, and where to; who "
        "speaks; each person's posture; and which pose from this list fits each person: " + ", ".join(poses)
        + ". Which parts of which person must be visible in the first picture for the action to read? Answer as "
        'JSON: {"moved": {name: place words}, "speaker": name or null, "pose": {name: pose}, '
        '"must_show": [[name, body part], ...]}.'), "answer": "json"}


def q_worn(qid: str, shot: dict, names: list) -> dict:
    return {"id": qid, "ask": (
        "A shot of a scene; the action: " + (shot.get("action") or "(none written)") + " The things named in it: "
        + json.dumps(shot.get("things") or {}) + ". For each of these people: " + ", ".join(names) + ", does what "
        "they wear at the shot's start differ from their usual clothes? Answer as JSON {name: the difference in a "
        "few words, or null}."), "answer": "json"}


def q_cast(qid: str, name_words: str, figures: list) -> dict:
    return {"id": qid, "ask": (
        "A person of the story is described: " + name_words + ". The library holds these figures, each with its "
        "id and description:\n" + "\n".join(f"- {f['id']}: {f['description']}" for f in figures)
        + "\nWhich figure is this person? Answer with the id only."), "answer": "id"}


# ------------------------------------------------------------------ mechanics
def words(s: str) -> set:
    return set(re.findall(r"[a-z]+", (s or "").lower()))


def pick_set(shot: dict, library: Path) -> tuple:
    best = (-1, None)
    sets = []
    for d in sorted((library / "set").iterdir()):
        a = d / "asset.json"
        if not a.exists():
            continue
        aj = json.loads(a.read_text())
        sets.append({"id": aj["id"], "description": aj.get("description", "")})
        w = words(" ".join(aj.get("tags") or []) + " " + aj.get("description", ""))
        score = len(w & words(" ".join([shot.get("place", ""), shot.get("room", "")] + (shot.get("fixtures") or []))))
        if score > best[0]:
            best = (score, aj["id"])
    return best[1], sets


def thing_for(at: str, things: list, names: dict) -> tuple:
    """(set thing or None, person key or None) the place words name: a person ("beside X"), else a thing by its
    own name; "the chair by the X" names X (the seat is then the nearest to X)."""
    w = words(at)
    for nm, k in names.items():
        if words(nm) & w and len(words(nm) & w) >= 1 and any(len(x) > 3 for x in words(nm) & w):
            return None, k
    m = re.search(r"by the (\w+)", at or "")
    order = [m.group(1)] if m else []
    order += [x for x in re.findall(r"[a-z]+", (at or "").lower())]
    for x in order:
        for base, syn in THING_WORDS.items():
            if x in syn and base != "armchair":
                cand = [t for t in things if t["name"] == base or t["name"].startswith(base + "_")
                        or (base == "window" and t["kind"] == "window") or (base == "door" and t["kind"] == "door")]
                cand.sort(key=lambda t: (t["name"] != base, t["name"]))     # the thing of that very name first
                if cand:
                    return cand[0], None
    return None, None


def free_mark(target: list, others: list, things: list, room: list, near=(0.6, 1.6)) -> list:
    """The nearest free floor point to the target thing's middle, between near[0] and near[1] m from it, clear of
    fixtures (their footprint + 0.25 m) and other people (0.55 m), inside the room, toward the room's middle."""
    W, D = room[0] / 2 - 0.35, room[1] / 2 - 0.35
    best = None
    for r10 in range(int(near[0] * 10), int(max(near[1], 3.0) * 10) + 1):     # widened until a point is free
        r = r10 / 10
        for a in range(0, 360, 10):
            p = [target[0] + r * math.cos(math.radians(a)), target[1] + r * math.sin(math.radians(a))]
            if abs(p[0]) > W or abs(p[1]) > D:
                continue
            if any(math.dist(p, o) < 0.55 for o in others):
                continue
            if any(t["kind"] == "fixture" and abs(p[0] - t["center"][0]) < t["half"][0] + 0.25
                   and abs(p[1] - t["center"][1]) < t["half"][1] + 0.25 for t in things):
                continue
            score = r + 0.3 * math.hypot(*p) / max(W, D)          # nearer the thing, then nearer the middle
            if best is None or score < best[0]:
                best = (score, p)
        if best:
            break
    return [round(best[1][0], 3), round(best[1][1], 3)] if best else [round(target[0] * 0.6, 3), round(target[1] * 0.6, 3)]


def main():
    a = sys.argv[1:]
    plan = json.loads(Path(a[0]).read_text())
    ids = [x.strip() for x in Path(a[1]).read_text().splitlines() if x.strip()]
    cast = json.loads(Path(a[2]).read_text())
    out = Path(a[3])
    out.mkdir(parents=True, exist_ok=True)
    answers = json.loads(Path(a[a.index("--answers") + 1]).read_text()) if "--answers" in a else {}
    library = Path(C.LIBRARY)
    first = plan[ids[0]][0]
    questions, facts = [], {}
    set_id, sets = pick_set(first, library)
    questions.append(q_set(first, sets))
    set_id = answers.get("set", set_id)
    set_dir = library / "set" / set_id
    things = setfacts.things(set_dir)
    room = json.loads((set_dir / "spec.json").read_text())["size_m"]
    seats = seating.seats(library, set_dir)
    for n, s_ in seats.items():
        s_["name"] = n
    names = {nm: v["key"] for nm, v in cast.items()}
    named = [t for t in things if t["kind"] in ("fixture", "window", "door")]
    state: dict = {}            # key -> {"at", "pose", "position", "facing", "seat", "facing_words"}
    groups, cur, cur_sig = [], [], None
    for n, sid in enumerate(ids, 1):
        sh = plan[sid][0]
        f = {"id": sid, "kind": sh.get("kind"), "framing": sh.get("framing"), "speaker": sh.get("speaker"),
             "listener": sh.get("listener"), "people_listed": [p["name"] for p in sh.get("people") or []],
             "tight": sh.get("tight"), "defaults": [], "missing": []}
        if not sh.get("people"):
            f["missing"].append("people and places (carried from the shot before)")
        questions.append(q_present(f"present:{sid}", (sh.get("action") or sh.get("picture") or ""), list(state)))
        if sh.get("kind") == "action":
            questions.append(q_change(f"change:{sid}", sh, {k: v for k, v in state.items()},
                                      ["stand", "stand_relaxed", "stand_hand_on_mantel", "gesture_point", "seated",
                                       "sit_armchair_relaxed", "veil_lift", "look_at_sleeve", "reach_forearm", "look"]))
            f["missing"].append("pose for the action's start; what the key must show")
        questions.append(q_worn(f"worn:{sid}", sh, [p["name"] for p in sh.get("people") or []]))
        placed_now = []
        for p in sh.get("people") or []:
            k = names.get(p["name"])
            if not k:
                f["missing"].append(f"library figure for a person ({p['name'][:1]}.)")
                continue
            prev = state.get(k)
            if prev and prev["at"] == p.get("at") and prev["pose"] == p.get("pose"):
                prev["facing_words"] = p.get("facing")
                continue
            qid = f"place:{sid}:{k}"
            questions.append(q_place(qid, f"{p.get('pose')} {p.get('at')}, facing {p.get('facing')}",
                                     named, [f"{v['pose']} {v['at']}" for kk, v in state.items() if kk != k]))
            ans = answers.get(qid)
            thing, rel = thing_for(p.get("at", ""), things, {nm: kk for nm, kk in names.items() if kk != k})
            if ans:
                thing = next((t for t in things if t["name"] == ans.get("thing")), thing)
            else:
                f["defaults"].append(f"{k}: place by name match ({(thing or {}).get('name') or rel})")
            entry = {"at": p.get("at"), "pose": p.get("pose"), "facing_words": p.get("facing")}
            if p.get("pose") == "sitting":
                anchor = (thing or {}).get("center") or (state.get(rel) or {}).get("position") or [0, 0]
                taken = {v.get("seat") for kk, v in state.items() if kk != k}
                cands = [s_ for s_ in seats.values() if s_["name"] not in taken]
                seat = min(cands, key=lambda s_: math.dist(s_["pos"], anchor[:2])) if cands else None
                entry.update(seat=seat["name"] if seat else None, position=list(seat["pos"]) if seat else anchor[:2])
            else:
                if rel and rel in state:
                    tgt = state[rel]["position"]
                    near = (0.5, 0.9)
                else:
                    tgt = (thing or {}).get("center") or [0, 0]
                    near = (0.6, 1.6)
                others = [v["position"] for kk, v in state.items() if kk != k] + [e["position"] for e in placed_now]
                entry.update(seat=None, position=free_mark(tgt, others, things, room, near))
            state[k] = entry
            placed_now.append(entry)
        # facing from the words: a person's name, else a named thing, else the room's middle
        for k, v in state.items():
            fw = v.get("facing_words") or ""
            _, rel = thing_for(fw, things, {nm: kk for nm, kk in names.items() if kk != k})
            thing, _ = thing_for(fw, things, {})
            tgt = (state.get(rel) or {}).get("position") if rel else (thing or {}).get("center")
            if tgt is None and len(state) > 1:                # "the camera", a group: toward the others' middle
                o = [vv["position"] for kk, vv in state.items() if kk != k]
                tgt = [sum(x[0] for x in o) / len(o), sum(x[1] for x in o) / len(o)]
            tgt = tgt or [0.0, 0.0]
            d = [tgt[0] - v["position"][0], tgt[1] - v["position"][1]]
            nd = math.hypot(*d) or 1.0
            v["facing"] = [round(d[0] / nd, 4), round(d[1] / nd, 4)]
        spk = names.get(sh.get("speaker")) if sh.get("speaker") else None
        lis = names.get(sh.get("listener")) if sh.get("listener") else None
        listed = [names[p["name"]] for p in sh.get("people") or [] if p["name"] in names]
        subj = spk or (listed[0] if listed else next(iter(state)))
        kind = FRAMING.get(sh.get("framing") or "medium", "medium")
        if sh.get("tight") and kind != "close":
            kind = "close"
        cam = {"name": f"s{n:02d}_{sid.split('-')[0]}", "plan_id": sid, "kind": kind, "subject": subj,
               "speaker": spk, "focal_mm": FOCAL[kind], "show": listed or None}
        if lis:
            cam["listener"] = lis
        f.update(camera=dict(cam), staged={k: {kk: v[kk] for kk in ("pose", "at", "seat", "position", "facing")}
                                           for k, v in state.items()})
        facts[sid] = f
        sig = json.dumps({k: (v["position"], v["pose"], v.get("seat")) for k, v in state.items()}, sort_keys=True)
        if sig != cur_sig and cur:
            groups.append(cur)
            cur = []
        cur_sig = sig
        cur.append((cam, json.loads(json.dumps(state))))
    if cur:
        groups.append(cur)
    specs = []
    for gi, g in enumerate(groups, 1):
        st = g[0][1]
        chars = {}
        for k, v in st.items():
            cv = next(c for c in cast.values() if c["key"] == k)
            if v["pose"] == "sitting":
                c = {"seated_from": cv.get("base") or cv["id"], "seat": v.get("seat") or True,
                     "position": v["position"], "facing": v["facing"], "body_pose": json.loads(json.dumps(SEAT_POSE))}
            else:
                c = {"id": cv.get("id") or cv["base"], "position": v["position"], "facing": v["facing"],
                     "body_pose": json.loads(json.dumps(STAND_POSE))}
            chars[k] = c
        line = [k for k in ("a", "b") if k in chars] or list(chars)[:2]
        spec = {"id": f"pl_g{gi:02d}", "set": set_id, "guides": True, "characters": chars, "line": line,
                "exposure_ev100": 5.4, "cameras": [c for c, _ in g]}
        p = out / f"pl_g{gi:02d}.json"
        p.write_text(json.dumps(spec, indent=1))
        specs.append(str(p))
    (out / "plan_facts.json").write_text(json.dumps(facts, indent=1))
    (out / "questions.json").write_text(json.dumps(questions, indent=1))
    print(json.dumps({"set": set_id, "groups": len(groups), "specs": specs, "questions": len(questions)}))


if __name__ == "__main__":
    main()
