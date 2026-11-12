"""Director's plan -> scene specs that speakers.py stages (stdlib). Mechanics only; judgment is read from answers.

    python tools/assets/scenes/from_plan.py PLAN.json SHOTS.txt CAST.json OUT_DIR [--answers ANSWERS.json]
           [--pass PASS.json]   (staging_pass.py's answers: the set, who is in the room, places, moves and where
                                 they end, worn state, poses, what the key must show, looks, the talk camera kind)

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


DIST = {"touching": 0.45, "near": 0.8, "a few steps": 1.5}
NOTS = (None, "", "not stated")


def place_from(ans: dict, things: list, seats: dict, room: list, others: list, taken: set) -> dict | None:
    """A place from the staging pass's words (a named thing, sitting or not, side, distance) -> position / seat;
    None when it is not stated or names nothing the set has."""
    if not isinstance(ans, dict) or ans.get("thing") in NOTS:
        return None
    t = next((x for x in things if x["name"] == ans["thing"]), None)
    if t is None:
        return None
    sitting = bool(ans.get("sitting")) or ans.get("posture") == "sitting"
    if sitting:
        cands = [s_ for n_, s_ in seats.items() if n_ not in taken]
        seat = seats.get(t["name"]) if t["name"] in seats and t["name"] not in taken else (
            min(cands, key=lambda s_: math.dist(s_["pos"], t["center"][:2])) if cands else None)
        if seat:
            return {"pose": "sitting", "seat": seat["name"], "position": list(seat["pos"])}
    c = t["center"][:2]
    mid = [-c[0], -c[1]]                       # toward the room's middle = the thing's front, as seen from there
    n = math.hypot(*mid) or 1.0
    fr = [mid[0] / n, mid[1] / n]
    rt = [fr[1], -fr[0]]
    d = DIST.get(ans.get("distance"), 0.8) + max(t["half"][0], t["half"][1]) * 0.5
    side = ans.get("side") or "front"
    v = {"front": fr, "behind": [-fr[0], -fr[1]], "left": [-rt[0], -rt[1]], "right": rt}.get(side, fr)
    tgt = [c[0] + v[0] * d, c[1] + v[1] * d]
    return {"pose": "standing", "seat": None, "position": free_mark(tgt, others, things, room, (0.0, 0.8))}


VARIANT = {"usual": None}


def main():
    a = sys.argv[1:]
    plan = json.loads(Path(a[0]).read_text())
    ids = [x.strip() for x in Path(a[1]).read_text().splitlines() if x.strip()]
    cast = json.loads(Path(a[2]).read_text())
    out = Path(a[3])
    out.mkdir(parents=True, exist_ok=True)
    answers = json.loads(Path(a[a.index("--answers") + 1]).read_text()) if "--answers" in a else {}
    sp = json.loads(Path(a[a.index("--pass") + 1]).read_text()) if "--pass" in a else None    # staging_pass.py
    library = Path(C.LIBRARY)
    first = plan[ids[0]][0]
    questions, facts = [], {}
    set_id, sets = pick_set(first, library)
    questions.append(q_set(first, sets))
    set_id = (sp or {}).get("set") or answers.get("set", set_id)
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
    worn: dict = {}

    def apply_place(k, pa, why):
        others = [v["position"] for kk, v in state.items() if kk != k]
        taken = {v.get("seat") for kk, v in state.items() if kk != k}
        pl = place_from(pa, things, seats, room, others, taken)
        if pl:
            state[k] = dict(state.get(k) or {"facing_words": None}, at=f"{pa.get('side')} {pa['thing']}", **pl)
            return why
        return None

    if sp:
        for nm, v in ((sp.get("scene") or {}).get("people") or {}).items():
            k = names.get(nm)
            if k and isinstance(v, dict) and v.get("in_room") is not False:
                apply_place(k, v, "pass")
                if v.get("worn") not in NOTS:
                    worn[k] = v["worn"]
    for n, sid in enumerate(ids, 1):
        sh = plan[sid][0]
        pa = (sp or {}).get("shots", {}).get(sid) or {}
        used = []
        if isinstance(pa.get("in_room"), list):          # who is in the room: those not named leave
            inroom = {names[x] for x in pa["in_room"] if x in names}
            for k in list(state):
                if k not in inroom:
                    state.pop(k)
                    used.append(f"{k} left the room")
        for nm, mv in (pa.get("moved") or {}).items():
            if nm in names and apply_place(names[nm], mv, "pass"):
                used.append(f"{names[nm]} moved to {mv.get('thing')}")
        for nm, w in (pa.get("worn") or {}).items():
            if nm in names and w not in NOTS:
                worn[names[nm]] = w
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
        moved_now = {names[nm] for nm in (pa.get("moved") or {}) if nm in names}
        for p in sh.get("people") or []:
            k = names.get(p["name"])
            if not k:
                f["missing"].append(f"library figure for a person ({p['name'][:1]}.)")
                continue
            if sp and (k in moved_now or (k in state and state[k].get("at", "").split(" ")[-1] in
                                          {t["name"] for t in things})):
                state[k]["facing_words"] = p.get("facing")        # the pass placed this person: keep it
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
        ck = pa.get("camera")
        if spk and ck not in NOTS:
            kind = {"close": "close", "wide": "full"}.get(ck, "medium" if kind == "close" and ck != "close" else kind)
        must = [names[x[0]] for x in (pa.get("must_show") or []) if isinstance(x, list) and x and x[0] in names]
        cam = {"name": f"s{n:02d}_{sid.split('-')[0]}", "plan_id": sid, "kind": kind, "subject": subj,
               "speaker": spk, "focal_mm": FOCAL[kind], "show": sorted(set(listed + must)) or None}
        if lis:
            cam["listener"] = lis
        if spk and ck == "over_the_shoulder" and lis:
            cam["ots"] = lis
        if spk and ck == "single":
            cam["single"] = True
            cam.pop("listener", None)
        looks = {names[a_]: (names.get(b_) or b_) for a_, b_ in (pa.get("looks_at") or {}).items()
                 if a_ in names and b_ not in NOTS}
        poses = {names[a_]: b_ for a_, b_ in (pa.get("pose") or {}).items() if a_ in names and b_ not in NOTS}
        f["pass"] = {"used": used, "camera": ck, "looks": looks, "pose": poses, "must_show": pa.get("must_show"),
                     "worn": dict(worn)}
        cam["_looks"] = looks
        cam["_poses"] = poses
        cam["_worn"] = dict(worn)
        f.update(camera=dict(cam), staged={k: {kk: v[kk] for kk in ("pose", "at", "seat", "position", "facing")}
                                           for k, v in state.items()})
        facts[sid] = f
        sig = json.dumps({k: (v["position"], v["pose"], v.get("seat")) for k, v in state.items()}, sort_keys=True) \
            + json.dumps([cam.get("_poses"), cam.get("_worn")], sort_keys=True)
        if sig != cur_sig and cur:
            groups.append(cur)
            cur = []
        cur_sig = sig
        cur.append((cam, json.loads(json.dumps(state))))
        for nm, mv in (pa.get("move_ends") or {}).items():       # where a move during this shot ends
            if nm in names:
                apply_place(names[nm], mv, "pass")
    if cur:
        groups.append(cur)
    specs = []
    for gi, g in enumerate(groups, 1):
        st = g[0][1]
        chars = {}
        cam0 = g[0][0]
        for k, v in st.items():
            cv = next(c for c in cast.values() if c["key"] == k)
            w = (cam0.get("_worn") or {}).get(k)
            var = None if w in (None, "usual") else w.replace(" ", "_")
            if v["pose"] == "sitting":
                c = {"seated_from": cv.get("base") or cv["id"], "seat": v.get("seat") or True,
                     "position": v["position"], "facing": v["facing"], "body_pose": json.loads(json.dumps(SEAT_POSE))}
                if var:
                    c["variant"] = var
            else:
                base = cv.get("id") or cv["base"]
                vid = f"{base}_{var}" if var and (library / "character" / f"{base}_{var}").exists() else base
                c = {"id": vid, "position": v["position"], "facing": v["facing"],
                     "body_pose": json.loads(json.dumps(STAND_POSE))}
            lk = (cam0.get("_looks") or {}).get(k)
            if lk in st:
                c["look_at"] = lk
            pz = (cam0.get("_poses") or {}).get(k)
            if pz:
                c["_pose_choice"] = pz                       # recorded; the pose list's extras need a solve
            chars[k] = c
        for c_, _ in g:
            for kk in ("_looks", "_poses", "_worn"):
                c_.pop(kk, None)
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
