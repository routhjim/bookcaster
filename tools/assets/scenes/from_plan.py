"""Director's plan -> scene specs that speakers.py stages (stdlib). Mechanics only; judgment is read from answers.

    python tools/assets/scenes/from_plan.py PLAN.json SHOTS.txt CAST.json OUT_DIR [--answers ANSWERS.json]
           [--pass PASS.json]   (staging_pass.py's closed answers for the plan's gaps: who is in the room where the
                                 plan lists nobody, where a move ends, worn state at the start and at a shot's end,
                                 the action's start pose, the talk camera kind; never a fact the plan gives)

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
import lying  # noqa: E402
import stature  # noqa: E402
import creature  # noqa: E402
import setfacts  # noqa: E402
import framing  # noqa: E402
sys.path.insert(0, str(HERE.parent / "places"))
import ground_map as GM  # noqa: E402

FRAMING = {"wide": "full", "full": "full", "medium": "medium", "close": "close", "two": "medium",
           "over-the-shoulder": "medium"}
FOCAL = {"full": 28, "medium": 50, "close": 85}
CARRIED_ON = ("head", "held")           # the things staged as objects: worn on the head, carried in a hand
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


def said_look(t: dict) -> str:
    """What a thing is, for a thing whose name does not say it (a kind that holds any thing: a landmark, a work
    piece, named by its kind): asked where a person "in the doorway" stood among "landmark" and "landmark_2", the
    answer put her behind the building, 11-18."""
    if str(t.get("short") or "").replace("_", " ") not in ("landmark", "work piece") or not str(t.get("look") or "").strip():
        return ""
    return ": " + str(t["look"]).split(";")[0].strip()


def q_place(qid: str, person_words: str, set_things: list, people: list) -> dict:
    return {"id": qid, "ask": (
        "A room holds these named things, each with where it stands: "
        + "; ".join(f"{t['name']} ({t.get('place') or t['kind']}{said_look(t)})" for t in set_things)
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
        sp = d / "spec.json"                 # a set built for this very place (place_build.py) is the place
        if shot.get("place") and sp.exists() and json.loads(sp.read_text()).get("place") == shot["place"]:
            best = (10 ** 6, aj["id"])
            continue
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


PERSON_GAP_M = 1.1        # standing people keep this between them (at 0.7 m, nose to nose, one camera in four could
                          # show a face past the other's head; at 1.1 m, four in six: measured on a failed pair)


def too_near(p: list, others: list, own_r: float = 0.0) -> bool:
    """Closer to another person than PERSON_GAP_M; an other given as [x, y, r] (a creature) keeps r more clear of
    its middle, and a placed creature (own_r) its own reach more."""
    return any(math.dist(p, o[:2]) < PERSON_GAP_M + own_r + (o[2] if len(o) > 2 else 0.0) for o in others)


def held_on(in_hand: dict, rows: dict, said: set, present: set) -> dict:
    """A shot's carried things with what people still have in their hands from the shots before: the pass speaks of
    a shot's own people only, so someone it says nothing of in this shot (a line spoken by another, with them in
    the background) keeps what they held; someone it does speak of has what it says, nothing when it says nothing;
    someone gone from the place has let go. in_hand is kept for the next shot. (A player's instrument was in his
    hand in the ten shots that named him and gone from his hand in every shot between them, 11-27.)"""
    for k in [k for k in in_hand if k not in present]:
        del in_hand[k]
    out = {k: list(v) for k, v in rows.items()}
    for k in said:
        now = [dict(r) for r in rows.get(k) or [] if r.get("at") == "hand"]
        was = in_hand.get(k) or []
        if now and len(now) < len(was) and {r["asset"] for r in now} < {r["asset"] for r in was}:
            # (still holding one of the things they held: the shot's line names that one, and the other is in the
            # other hand yet. A bow named once with its fiddle was gone from the hand in every shot after, 11-27)
            out[k] = [r for r in out[k] if r.get("at") != "hand"] + [dict(r) for r in was]
            now = [dict(r) for r in was]
        in_hand[k] = now
    for k, kept in in_hand.items():
        if k not in said and kept:
            out.setdefault(k, []).extend(dict(r) for r in kept)
    return out


STAGE_DIR: list = []       # a place built from ONE picture, its cameras held to that picture's side (the set's
#                            "picture_dir", not "all_round"; speakers.py picture_side): the way the picture looks, in
#                            plan. main() fills it; empty in a room and in a place with a true panorama all round
STAGE_ACROSS_DEG = 55.0    # there the line between two people of a group runs at least this far off the picture's view
STAGE_GROUP_M = 4.0        # ... two people of a group: no further apart than this
STAGE_REACH_M = 1.5        # a mark across the stage is looked for this far past the place's own reach, no further
STAGE_OPEN_DEG = 30.0      # someone standing turned to a person or the group is turned this far toward the cameras


def stage_open(facing: list, standing: bool) -> list:
    """A standing person's facing toward a person or their group, on a stage of one picture: turned STAGE_OPEN_DEG
    toward the side its cameras stand on, as on any stage; the facing itself anywhere else. (Four people across
    such a stage faced each other along it, and no camera of its side saw more than a quarter of a face, 11-27.)"""
    if not STAGE_DIR or not standing or math.hypot(facing[0], facing[1]) < 1e-6:
        return facing
    return framing.turned_toward(facing, [-STAGE_DIR[0], -STAGE_DIR[1]], STAGE_OPEN_DEG)


def stage_dir(set_spec: dict) -> list:
    """The way a set's one picture looks when its cameras are held to it (speakers.py picture_side), else []."""
    d = set_spec.get("picture_dir")
    if not d or set_spec.get("all_round") or math.hypot(d[0], d[1]) < 1e-6:
        return []
    n = math.hypot(d[0], d[1])
    return [d[0] / n, d[1] / n]


def across(p: list, others: list) -> bool:
    """Whether someone at p stands across the stage from the people near them, not before or behind them: every
    camera of a place built from one picture looks one way, so of two people in a line along that way the nearer
    hides the other and, turned to them, shows the camera their back. True in any other place, and of a creature
    ([x, y, reach]: its own places are its reach's). (Four people placed in a ring on such a stage: one stood
    straight between the camera's side and another, back to every camera; the group's wide had no key, 11-27.)"""
    if not STAGE_DIR:
        return True
    lim = math.cos(math.radians(STAGE_ACROSS_DEG))
    for o in others:
        d = [o[0] - p[0], o[1] - p[1]]
        n = math.hypot(*d)
        if len(o) > 2 or n < 1e-6 or n > STAGE_GROUP_M:
            continue
        if abs(d[0] * STAGE_DIR[0] + d[1] * STAGE_DIR[1]) / n > lim:
            return False
    return True


def bounds(room: list, own_r: float = 0.0, walls: bool = True) -> tuple:
    """How far from the middle a placed middle may go: inside the walls by 0.35 m and, with walls, by a creature's
    reach too. A place in the open has no walls: a creature's body may reach past the stage's edge onto the ground
    round it (on a 13 m stage a 5 m reach left it no free point, and it was stood on a fallback beside a man, 11-16)."""
    return room[0] / 2 - 0.35 - (own_r if walls else 0.0), room[1] / 2 - 0.35 - (own_r if walls else 0.0)


def dry(p: list, things: list) -> bool:
    """Not on the water of a place in the open (a "water" thing: the set's ground-map water, places/ground_map.py).
    People are never stood in it unless their place is in the water (the caller then leaves the water out)."""
    return not any(t["kind"] == "water" and GM.in_water(t.get("grid"), p[0], p[1]) for t in things)


def water_things(set_spec: dict) -> list:
    """The set's water as a thing of the place (no name a plan phrase can land on; its middle under the ground)."""
    # (and where its land rises, sets/place_build.py land_high: nobody is stood inside a bank or a slope either)
    return [{"name": n_, "asset": None, "kind": "water", "grid": w, "center": [0.0, 0.0, -0.05],
             "half": [0.0, 0.0, 0.0], "on": None, "lit": False, "fire": False}
            for n_, w in (("_water", set_spec.get("water")), ("_high_land", set_spec.get("high_land")))
            if w and w.get("runs")]


def clear_of_fixtures(p: list, things: list, r: float = 0.0) -> bool:
    return not any(t["kind"] == "fixture" and abs(p[0] - t["center"][0]) < t["half"][0] + 0.25 + r
                   and abs(p[1] - t["center"][1]) < t["half"][1] + 0.25 + r for t in things)


TALL_M = 1.8              # a thing whose top is over this hides whoever stands behind it


def front_of(thing: dict | None):
    """For a tall thing of the set (a building, a wall, a great piece): the unit way from its middle to the room's
    middle, its front as the people and cameras see it; None for a thing people are seen over or a place on the
    floor. People sent to such a thing stand before it: of two sent to a forge's front, the second was stood
    behind the building, hidden from every camera, 11-18."""
    c = (thing or {}).get("center") or []
    hf = (thing or {}).get("half") or []
    if len(c) < 3 or len(hf) < 3 or (thing or {}).get("kind") in ("floor", "mark") or c[2] + hf[2] < TALL_M:
        return None
    n = math.hypot(c[0], c[1])
    return [-c[0] / n, -c[1] / n] if n > 0.5 else None


STAND_CLEAR = (1 / 3, 2.0)   # of a tall thing's height, and at most this many metres


def stand_clear(thing: dict | None) -> float:
    """How far clear of a tall thing's foot people sent to it stand: a third of its height, 2 m at most. Stood at a
    building's very wall they were under its eaves, in its shadow, and every close shot of them was black (11-18)."""
    c, hf = (thing or {}).get("center") or [], (thing or {}).get("half") or []
    return round(min(STAND_CLEAR[1], STAND_CLEAR[0] * (c[2] + hf[2])), 2) if len(c) >= 3 and len(hf) >= 3 else 0.0


def before(p: list, target: list, front, off: float = 0.0) -> bool:
    """In front of the target (more than off metres out along front), or anywhere with no front."""
    return front is None or (p[0] - target[0]) * front[0] + (p[1] - target[1]) * front[1] > off


def free_marks(target: list, others: list, things: list, room: list, near=(0.6, 1.6), n: int = 12,
               own_r: float = 0.0, fix_r: float | None = None, walls: bool = True, front=None,
               front_off: float = 0.0) -> list:
    """Every allowed mark for a place given only as a relation or a region ("beside him", "near the desk"): free
    floor points within the place's reach, best first (free_mark's order); the camera rules choose among them."""
    W, D = bounds(room, own_r, walls)
    fr = own_r if fix_r is None else fix_r
    out = []
    for r10 in range(int(near[0] * 10), int(near[1] * 10) + 1, 2):
        r = r10 / 10
        for a in range(0, 360, 20):
            p = [target[0] + r * math.cos(math.radians(a)), target[1] + r * math.sin(math.radians(a))]
            if abs(p[0]) > W or abs(p[1]) > D or too_near(p, others, own_r) or not before(p, target, front, front_off):
                continue
            if not clear_of_fixtures(p, things, fr) or not dry(p, things):
                continue
            out.append((r + 0.3 * math.hypot(*p) / max(W, D), [round(p[0], 3), round(p[1], 3)]))
    if not own_r and any(across(p, others) for _, p in out):   # (a stage of one picture: its marks across it,
        out = [x for x in out if across(x[1], others)]         # where there are any)
    out.sort(key=lambda x: x[0])
    picked = []
    for _, p in out:                       # spread out: no two marks closer than 0.35 m
        if all(math.dist(p, q) >= 0.35 for q in picked):
            picked.append(p)
        if len(picked) >= n:
            break
    return picked


def free_mark(target: list, others: list, things: list, room: list, near=(0.6, 1.6), own_r: float = 0.0,
              fix_r: float | None = None, walls: bool = True, front=None, front_off: float = 0.0,
              _stage: bool | None = True) -> list:
    """The nearest free floor point to the target thing's middle, between near[0] and near[1] m from it, clear of
    fixtures (their footprint + 0.25 m) and other people (too_near), inside the room, toward the room's middle; a
    creature (own_r: its reach round its middle) keeps that much more clear of walls and people, and fix_r (by
    default the same) more clear of fixtures."""
    W, D = bounds(room, own_r, walls)
    fr = own_r if fix_r is None else fix_r
    if _stage and STAGE_DIR and not own_r and others:
        # a stage of one picture: across it from the people already there (across()), within a little of the
        # place's reach; where no such point is free, the place as anywhere else
        got = free_mark(target, others, things, room, near, own_r, fix_r, walls, front, front_off, _stage=None)
        if got:
            return got
    best = None
    # widened until a point is free: as far as the room goes (a thing bigger than the old 3 m reach, a craft in a
    # hangar, left no free point and everyone sent to it was stood on one fallback spot, 11-13)
    r_far = max(near[1], 3.0, math.hypot(room[0], room[1])) if _stage is not None else near[1] + STAGE_REACH_M
    for r10 in range(int(near[0] * 10), int(r_far * 10) + 1):
        r = r10 / 10
        for a in range(0, 360, 10):
            p = [target[0] + r * math.cos(math.radians(a)), target[1] + r * math.sin(math.radians(a))]
            if abs(p[0]) > W or abs(p[1]) > D or not before(p, target, front, front_off):
                continue
            if too_near(p, others, own_r) or (_stage is None and not across(p, others)):
                continue
            if not clear_of_fixtures(p, things, fr) or not dry(p, things):
                continue
            # nearer the thing, then nearer the middle; and with whoever is already at the place rather than across
            # it (two men sent to one craft stood on opposite sides of it, the craft between them)
            score = r + 0.3 * math.hypot(*p) / max(W, D, 0.1) + 0.15 * min((math.dist(p, o[:2]) for o in others),
                                                                         default=0.0)
            if best is None or score < best[0]:
                best = (score, p, r)
        if best and r >= best[2] + 1.0:                           # a metre past the first free ring, then the best
            break
    if best:
        return [round(best[1][0], 3), round(best[1][1], 3)]
    if _stage is None:
        return None
    # nowhere is free. The people's clearance is the last thing given up (a body standing in a body is no staging):
    # first a creature's wider berth from the set's things (its body's breadth; its footprint still kept), then
    # the point with the most room from the others. (The old fallback, 0.6 of the way to the target, stood a 10 m
    # beast in the middle of the two people it was placed near, its neck through one of them, 11-18)
    if fr > 0.0:
        return free_mark(target, others, things, room, near, own_r, 0.0, walls, _stage=False)
    best = None
    for r10 in range(int(near[0] * 10), int(max(near[1], 3.0, math.hypot(room[0], room[1])) * 10) + 1, 2):
        r = r10 / 10
        for a in range(0, 360, 10):
            p = [target[0] + r * math.cos(math.radians(a)), target[1] + r * math.sin(math.radians(a))]
            if abs(p[0]) > W or abs(p[1]) > D or not clear_of_fixtures(p, things) or not dry(p, things):
                continue
            room_ = min((math.dist(p, o[:2]) - PERSON_GAP_M - own_r - (o[2] if len(o) > 2 else 0.0) for o in others),
                        default=0.0)
            if best is None or room_ > best[0] + 1e-9:
                best = (room_, p)
    if best:
        return [round(best[1][0], 3), round(best[1][1], 3)]
    return [round(target[0] * 0.6, 3), round(target[1] * 0.6, 3)]


BASE_STAND_M = (0.4, 1.2)  # someone sent to a thing on a low base stands this far from the middle of its front edge,
BASE_CLEAR_M = 0.35        # ... and at least this far out in front of that edge (the body clear of the base)
ONLOOK_M = (3.6, 7.0)     # someone looking on at two who are turned to each other stands this far from them


def looks_on(p: dict, people: list) -> bool:
    """Someone of a shot who watches two others that are turned to each other, nobody turned to them: not one of
    the exchange, they stand apart from it, behind it in the picture. (A third man on the open floor was stood at
    the elbow of one of two who spoke, a party of three, where the plan had him off at the side, 11-20.)"""
    face = {q.get("name"): q.get("facing") for q in people if q.get("name")}
    me, to = p.get("name"), p.get("facing")
    return bool(me and to in face and to != me and me not in face.values()
                and face.get(to) in face and face.get(face[to]) == to and face[to] != me)


JOIN_REACH_M = 0.62       # how far ahead of the body a joined hand is held at most (an arm not quite straight)
# how far apart two stand by how near the staging pass has them (m, between their middles); the others keep the
# distance their placing gave them
NEAR_GAP_M = {"dancing together": 0.38, "touching": 0.7, "close": 0.8}


NEAR_GOES_ON = ("dancing together",)     # what two go on doing through the shots that do not say (a dance); the
                                         # other bands are of the shots that say them (someone who steps up to a
                                         # pair to speak was kept at their elbow for the rest of the dance, 11-26)
NEAR_CLEAR_M = 0.6                       # one who comes near a person stays this clear of everyone else


def open_with_people(plan: dict, ids: list, visit: list) -> dict:
    """A scene that opens on shots which list nobody (a spoken line: its people are carried from the shot before, and
    there is none): those shots take the people, with their places, of the first shot of the same visit that lists
    them - where they are first described is where they already stood. Written into the plan's records here.
    -> {shot id: the shot its people came from}. (11-26: a stretch that began on a line had an empty room, the
    speaker was framed as nobody and the place's whole staging stopped.)"""
    out = {}
    for v in dict.fromkeys(visit):
        mine = [s for s, vv in zip(ids, visit) if vv == v]
        src = next((s for s in mine if plan[s][0].get("people")), None)
        if src is None or (v and any(plan[s][0].get("people") for s, vv in zip(ids, visit) if vv < v)):
            continue                      # (a later visit to the place has the people it had before)
        for s in mine[:mine.index(src)]:
            plan[s][0] = dict(plan[s][0], people=json.loads(json.dumps(plan[src][0]["people"])))
            out[s] = src
    return out


def _near_rows(name: str, rows: list) -> tuple:
    """The nearest band this person is in among one shot's pairs (a dancer whom a third steps up to is still in the
    dance). -> (whom, how) or (None, None)"""
    band = list(NEAR_GAP_M)
    best = None
    for row in rows or []:
        a, b, how = (list(row) + [None, None, None])[:3]
        if name in (a, b) and a != b:
            rank = band.index(how) if how in band else len(band)
            if best is None or rank < best[0]:
                best = (rank, b if a == name else a, how)
    return (best[1], best[2]) if best else (None, None)


def near_of(name: str, shot_ids: list, sp: dict, order: list | None = None) -> tuple:
    """(whom this person is near in any of these shots, how), from the staging pass's "near": the nearest band the
    shots give them. With none of their own, what the latest earlier shot (ORDER: every shot in story order) has
    them going on doing with someone (NEAR_GOES_ON), unless a shot since has those two in another band: two who
    dance go on dancing through the shots that do not say. (None, None) otherwise."""
    near = (sp or {}).get("near") or {}
    own = _near_rows(name, [r for sid in shot_ids for r in near.get(sid) or []])
    if own[0]:
        return own
    first = min((order.index(x) for x in shot_ids if x in order), default=0) if order else 0
    since = set()                        # whom a later shot has them with in another band (that ended it with them;
    for sid in (order[:first][::-1] if order else []):       # a third who stepped up ended nothing)
        for row in near.get(sid) or []:
            a_, b_, how = (list(row) + [None, None, None])[:3]
            if name in (a_, b_) and a_ != b_:
                who = b_ if a_ == name else a_
                if how in NEAR_GOES_ON:
                    return (None, None) if who in since else (who, how)
                since.add(who)
    return None, None


def near_position(pos: list, partner_pos: list, how: str, others: list | None = None) -> list:
    """Where the one of a near pair who is moved stands: at the band's distance from the partner, on the line to
    where they stood (nearer or farther than they were placed); when that is within NEAR_CLEAR_M of someone else
    (OTHERS: positions), round the partner to the nearest side that is clear."""
    gap = NEAR_GAP_M.get(how)
    d = math.dist(pos[:2], partner_pos[:2])
    if not gap or d < 1e-6:
        return list(pos)
    a0 = math.atan2(pos[1] - partner_pos[1], pos[0] - partner_pos[0])
    at = lambda a: [partner_pos[0] + math.cos(a) * gap, partner_pos[1] + math.sin(a) * gap]   # noqa: E731
    clear = lambda q: all(math.dist(q, o[:2]) >= NEAR_CLEAR_M for o in others or [])        # noqa: E731
    turns = [0.0] + [s * math.radians(k) for k in range(15, 181, 15) for s in (1, -1)]
    q = next((at(a0 + t_) for t_ in turns if clear(at(a0 + t_))), at(a0))
    return [round(q[0], 3), round(q[1], 3)] + list(pos[2:])


def near_pose(how: str, leads: bool) -> dict | None:
    """The stance of a band: a dance's closed hold (one leads), a joined hand, else none of its own."""
    if how == "dancing together":
        return {"pose": "dance_hold", "params": {"role": "lead" if leads else "follow",
                                                  "gap_m": NEAR_GAP_M[how], "weight_on": "l"}}
    if how == "touching":
        return {"pose": "hand_joined", "params": {"hand": "r", "weight_on": "l",
                                                   "reach_m": round(min(JOIN_REACH_M, NEAR_GAP_M[how] / 2), 3)}}
    return None


TALK_M = (1.1, 1.6)       # two on the open floor who are turned to each other stand this far apart (PERSON_GAP_M up)


def turned_to(name: str, faces: dict, others: list) -> bool:
    """Whether the staging pass has this person facing one of the shot's other people, or one of them facing this
    person: the two are with each other, not merely in the same place. (Two who spoke in a ballroom were stood by the
    open floor's rule for someone a few steps off, 2 m and more apart, and the picture of them both had one at each
    edge of the frame, 11-26.)"""
    faces = faces or {}
    return bool(name) and (faces.get(name) in others or any(faces.get(o) == name for o in others))


def floor_near(thing: dict) -> tuple:
    """How far from the others (or from the middle) someone on the open floor stands."""
    if thing.get("apart"):
        return ONLOOK_M
    if thing.get("with"):
        return TALK_M if thing.get("turned") else (2.0, 5.0)
    return (0.0, 1.6)


AT_THING = "the thing they are at"     # the pass's answer for someone facing what they work at (staging_pass)
TALL_THING_M = 1.5        # a thing whose top is above a standing person's eyes: a wall of it, nothing to work on top of
WALL_NEAR_M = 1.2         # standing this near its footprint, turned to it, a person's face is a hand from its face
WALL_OPEN_DEG = 60.0      # ... so they stand open to the floor, turned this far off it


def open_from_wall(pos: list, facing: list, thing: dict, others: list) -> list:
    """A standing person turned to a tall thing they stand at (a rock face, a cliff, a cabinet: its top above their
    eyes) stands open to the floor, turned WALL_OPEN_DEG off it toward the others (else the place's middle): squared
    up to it from a step away, every camera on their face stood inside it. (A woman on a ledge "at" a 3 m rock, turned
    to it 0.3 m from its face, showed only her back or a wall of rock in every try, 11-18.) A low thing (a desk, an
    anvil) or a far one -> the facing as it is."""
    c, h = thing.get("center") or [], thing.get("half") or []
    if len(c) < 3 or len(h) < 3 or c[2] + h[2] < TALL_THING_M:
        return facing
    gap = math.hypot(max(0.0, abs(pos[0] - c[0]) - h[0]), max(0.0, abs(pos[1] - c[1]) - h[1]))
    if gap > WALL_NEAR_M:
        return facing
    to = [sum(o[0] for o in others) / len(others), sum(o[1] for o in others) / len(others)] if others else [0.0, 0.0]
    cross = facing[0] * (to[1] - pos[1]) - facing[1] * (to[0] - pos[0])
    a = math.radians(WALL_OPEN_DEG) * (1.0 if cross >= 0 else -1.0)
    return [round(facing[0] * math.cos(a) - facing[1] * math.sin(a), 4),
            round(facing[0] * math.sin(a) + facing[1] * math.cos(a), 4)]
STRIKE_FAR_M = 2.5        # past a creature's reach: whoever it fights stands at most this much further off (a step
                          # or two from its snout), at least PERSON_GAP_M
FIGHT_CLEAR_M = 3.0       # and, where the floor allows, this far from anyone else and from the set's things (a camera
                          # on the fight can then leave out someone the shot does not list, and stand behind him: with
                          # a cave mouth at his back every camera on the beast's face was framed past the rock, 11-17)


WITH_M = 2.5                   # someone this near another is with them already


def stays(prev: dict, pose: str | None, where: list | None, state: dict, crt: dict, beds: bool) -> bool:
    """Whether someone already placed (prev) stays where they are in a shot whose words give them this pose and
    send them `where` (["thing", name] | ["with", person] | None). They move for another thing of the set, for
    someone they are not with, to sit down or get up from a seat or a bed; not for words that name no place (who
    lies down by such words lies where they stand: knocked flat "on the ground" in a place with a bench, a man was
    laid 12 m off, and stood up 13 m from there, 11-19)."""
    if not prev.get("position"):
        return False
    if pose != prev.get("pose") and (pose == "sitting" or prev.get("pose") == "sitting" or prev.get("bed")):
        return False
    if where is None or where == prev.get("where"):
        return True
    if where[0] == "with" and where[1] in state and state[where[1]].get("position"):
        near = WITH_M + (crt.get(where[1]) or {}).get("radius_m", 0.0) + (STRIKE_FAR_M if where[1] in crt else 0.0)
        return prev.get("foe") == where[1] or math.dist(prev["position"][:2], state[where[1]]["position"][:2]) <= near
    return False


STRIKE_ACROSS_DEG = 30    # in the open a fighter stands across the picture's view from the creature, within this


def strike_place(cpos: list, reach: float, ppos: list, others: list, things: list, room: list,
                 walls: bool = True, across: bool = False) -> list | None:
    """Where a person who fights a creature stands: at striking distance from its middle (its reach, from its record,
    and a gap past it: PERSON_GAP_M to STRIKE_FAR_M), on the side they are already on as near as the floor allows,
    clear of fixtures and of the others (FIGHT_CLEAR_M where the floor allows), inside the place. None when they
    already stand so (or nowhere is free). A man was jammed 1.6 m from a beast with a 5 m reach, and another stood
    10 m off (11-16). across (a place in the open, its picture looking along +y): beside the creature across that
    view, so a camera side-on to the two looks the way the picture does and has the picture's land behind them. (A
    man stood before a beast along the view; the side-on camera looked along a cliff made from the picture, edge
    on, and the cliff's flank was drawn as a stone arch in nine keys, 11-19.)"""
    lo, hi = reach + PERSON_GAP_M, reach + STRIKE_FAR_M
    d0 = math.dist(cpos, ppos)
    lies = not across or d0 < 1e-6 or \
        abs(ppos[1] - cpos[1]) <= d0 * math.sin(math.radians(STRIKE_ACROSS_DEG)) + 1e-6
    if lies and lo - 1e-6 <= d0 <= hi + 1e-6 and all(math.dist(ppos, o[:2]) >= FIGHT_CLEAR_M for o in others) and \
            clear_of_fixtures(ppos, things, FIGHT_CLEAR_M):
        return None
    W, D = bounds(room, 0.0, walls)
    a0 = math.atan2(ppos[1] - cpos[1], ppos[0] - cpos[0]) if d0 > 1e-6 else -math.pi / 2
    if across:                               # (from the side of the creature they are on, straight across)
        a0 = 0.0 if math.cos(a0) >= 0 else math.pi
    best = None
    for r in (lo + 0.1, (lo + hi) / 2, hi - 0.1):
        for da in range(0, 181, 10):
            for sgn in ((1,) if da in (0, 180) else (1, -1)):
                a = a0 + sgn * math.radians(da)
                q = [cpos[0] + r * math.cos(a), cpos[1] + r * math.sin(a)]
                if abs(q[0]) > W or abs(q[1]) > D or not clear_of_fixtures(q, things) or not dry(q, things):
                    continue
                if too_near(q, others):
                    continue
                # clear of the others first, then the side they are on, then near the reach
                crowd = sum(1 for o in others if math.dist(q, o[:2]) < FIGHT_CLEAR_M) + \
                    (0 if clear_of_fixtures(q, things, FIGHT_CLEAR_M) else 1)    # (room behind him for a camera)
                turn = 3 * min(da, 180 - da) + 0.1 * da if across else da
                score = 1000 * crowd + turn + 2.0 * abs(r - (lo + 0.6))
                if best is None or score < best[0]:
                    best = (score, [round(q[0], 3), round(q[1], 3)])
    return best[1] if best else None


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
    sitting = bool(ans.get("sitting")) is True or ans.get("posture") in ("sitting", "seated")
    if sitting:
        cands = [s_ for n_, s_ in seats.items() if n_ not in taken]
        seat = seats.get(t["name"]) if t["name"] in seats and t["name"] not in taken else (
            min(cands, key=lambda s_: math.dist(s_["pos"], t["center"][:2])) if cands else None)
        if seat:
            return {"pose": "sitting", "seat": seat["name"], "position": list(seat["pos"])}
    if t.get("base") and (ans.get("side") or "front") == "front":     # on a low base: before its front edge
        return {"pose": "standing", "seat": None,
                "position": free_mark(list(t["base"]["edge"]), others, things, room, BASE_STAND_M,
                                      front=list(t["base"]["front"]), front_off=BASE_CLEAR_M)}
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


WORK_STAND_M = 0.38       # a person working at a piece stands this far from its face (a smith at the anvil)
WORK_REACH_M = 0.62       # and their hands reach at most this far forward of their hips


WORK_TOP_M = (0.35, 1.3)  # a piece whose top lies in this band is one a standing person works on with their hands
WORK_NEAR_M = 1.2         # with no piece named, the work is the nearest such piece within this of where they stand


def work_thing(v: dict, things: list) -> dict | None:
    """The piece of the set a person at work works at: the thing they face, else the thing their place names (the
    pass's "<side> <thing>"), else the nearest piece with a top at working height."""
    tall = [x for x in things if len(x.get("center") or []) > 2 and len(x.get("half") or []) > 2
            and WORK_TOP_M[0] <= x["center"][2] + x["half"][2] <= WORK_TOP_M[1]]
    at = str(v.get("at") or "")
    for x in tall:
        if x["name"] == v.get("facing_thing") or at == x["name"] or at.endswith(" " + x["name"]):
            return x
    near = [(math.dist(v["position"], x["center"][:2]) - max(x["half"][:2]), i) for i, x in enumerate(tall)]
    near = [n for n in near if n[0] <= WORK_NEAR_M]
    return tall[min(near)[1]] if near else None


def work_place(t: dict, his: list, others: list, things: list, room: list) -> dict | None:
    """Where a person stands to work at a piece of the set with their hands (a start action of WORK_ACTIONS): at
    one of its faces, WORK_STAND_M off it and turned to it, the face nearest where they were that is inside the room,
    clear of the other pieces and people. -> {"position", "facing", "top_cm" (the piece's top above the floor),
    "fwd_cm" (how far forward of their hips the work lies), "near_cm" (how far its near face is)}; None when the
    piece has no free face or no height."""
    if len(t.get("center") or []) < 3 or len(t.get("half") or []) < 3:
        return None
    c, h = t["center"], t["half"]
    W, D = bounds(room)
    rest = [x for x in things if x is not t]
    best = None
    for ax, sg in ((0, 1), (0, -1), (1, 1), (1, -1)):
        d = h[ax] + WORK_STAND_M
        q = [c[0] + (sg * d if ax == 0 else 0.0), c[1] + (sg * d if ax == 1 else 0.0)]
        if abs(q[0]) > W or abs(q[1]) > D or not clear_of_fixtures(q, rest, -0.1) or too_near(q, others):
            continue
        # at its long side before an end (the work lies across in front of both hands), then the nearest
        rank = (h[ax] > h[1 - ax] * 1.15, math.dist(q, his))
        if best is None or rank < best[4]:
            best = (q, ax, sg, d, rank)
    if best is None:
        return None
    q, ax, sg, d, _ = best
    return {"position": [round(q[0], 3), round(q[1], 3)], "facing": [-float(sg) if ax == 0 else 0.0, -float(sg) if ax == 1 else 0.0],
            # (11-18: fwd_cm was capped at WORK_REACH_M, so the pose aimed short of a work lying farther; the true
            # distance goes to the pose, which leans to it, and a worked thing's own top point overrides it there)
            "top_cm": round((c[2] + h[2]) * 100, 1), "fwd_cm": round(d * 100, 1),
            "near_cm": round(WORK_STAND_M * 100, 1)}


def give_way(spot: list, t: dict, others: dict) -> dict | None:
    """Where the people standing by go when a person takes spot to work at piece t: each one nearer the spot than
    PERSON_GAP_M to their nearest mark that is clear of it and no nearer the piece's middle than the worker (not
    between them and the work). -> {key: mark}, {} when nobody is in the way, None when one of them has nowhere."""
    out, taken = {}, [list(spot)]
    reach = math.dist(spot, t["center"][:2])
    for k, v in others.items():
        if not too_near(v["position"], [spot]):
            taken.append(list(v["position"]))
            continue
        ok = [m for m in v.get("marks") or [] if not too_near(m, taken) and math.dist(m, t["center"][:2]) >= reach]
        if not ok:
            return None
        out[k] = min(ok, key=lambda m: math.dist(m, v["position"]))
        taken.append(list(out[k]))
    return out


WORK_VIEW_DEG = 70        # the side of a piece its work is seen from: this far either way of the line from the worker
WORK_VIEW_M = 2.6         # ... on past the piece, out to here
WORK_ASIDE_M = (1.6, 2.6)   # someone standing by steps to the piece's side, this far from its middle
WORK_ASIDE_DEG = (80, 125)  # ... this far round from that line (level with the piece to a little behind the worker)


def stand_aside(spot: list, t: dict, others: dict, things: list, room: list, walls: bool = True) -> dict:
    """Where the people standing by go when a person works at piece t from spot: whoever stands across the piece
    from the worker stands where the work is seen from (work_camera), and steps to its side, to the free mark
    nearest where they stood. -> {key: mark} (those with no free mark at the side stay). (A woman sent to the same
    piece stood half a metre beyond it, facing the worker over it: no camera had a place on the work, and of nine
    shots of the work eight were framed on the worker's chest with the piece out of frame, 11-20.)"""
    c = t["center"][:2]
    d = math.dist(spot[:2], c)
    if d < 0.2:
        return {}
    u = [(c[0] - spot[0]) / d, (c[1] - spot[1]) / d]

    def off(p):                                  # degrees round from the worker-to-piece line, seen from the piece
        w = [p[0] - c[0], p[1] - c[1]]
        n = math.hypot(*w)
        return 0.0 if n < 1e-6 else math.degrees(math.acos(max(-1.0, min(1.0, (w[0] * u[0] + w[1] * u[1]) / n))))
    out, taken = {}, [list(spot[:2])] + [list(v["position"][:2]) for v in others.values()]
    for k, v in others.items():
        p = v["position"][:2]
        if math.dist(p, c) > WORK_VIEW_M or off(p) > WORK_VIEW_DEG:
            continue
        marks = [m for m in free_marks(c, [q for q in taken if q != list(p)], things, room, WORK_ASIDE_M, n=60,
                                       walls=walls) if WORK_ASIDE_DEG[0] <= off(m) <= WORK_ASIDE_DEG[1]]
        if marks:
            out[k] = min(marks, key=lambda m: math.dist(m, p))
            taken[taken.index(list(p))] = list(out[k])
    return out


VARIANT = {"usual": None}


STAND_ACTIONS = ("stand_hand_on_mantel", "gesture_point")     # standing poses of their own (scenes/ue/pose.py)
WORK_POSE = "hands_at_work"      # someone with a tool in the hand and a thing lying before them, no pose named: the
                                 # tool down at the work (raised by default, a pair of tongs was held up by the
                                 # head, the work out of frame; the strike itself is the clip's, 11-20)
WORK_ACTIONS = ("tool_raised", "striking_down", "hands_at_work")      # standing work over what is in front (pose.py)
READY_ACTIONS = ("weapon_ready",)     # a fighter's guard (pose.py): placed by the fight, not at a piece
BLOW_POSES = ("tool_raised", "striking_down")    # a blow the pass may name for a fighter: the clip's, not its start
PLAY_ACTIONS = ("play_at_shoulder",)  # an instrument played (pose.py): gone on with shot after shot (played_on)
ACTION_NAMES = ("stand_hand_on_mantel", "gesture_point", "veil_lift", "look_at_sleeve", "reach_to_forearm") + \
    WORK_ACTIONS + READY_ACTIONS + PLAY_ACTIONS


def played_on(playing: dict, poses: dict, present: set, holding: set) -> dict:
    """A shot's poses with the playing of those who went on playing: someone who plays an instrument in one shot
    plays it in the shots after, whatever the shot is about, until a shot names another action for them, their hands
    are empty or they have left. The pass names an action where a shot's own line states it: the music under a
    whole scene is stated once. playing is kept for the next shot."""
    out = dict(poses)
    for k in [k for k in playing if k not in present or k not in holding
              or (out.get(k) in ACTION_NAMES and out[k] not in PLAY_ACTIONS)]:
        del playing[k]
    for k, pz in out.items():
        if pz in PLAY_ACTIONS and k in holding and k in present:
            playing[k] = pz
    for k, pz in playing.items():
        out[k] = pz
    return {k: v for k, v in out.items() if v not in PLAY_ACTIONS or k in holding}


def play_hands(rows: list) -> list:
    """A player's carried things with their hands as the instrument is played: the first thing in the hand (the
    instrument: the pass's first) in the left, the second in the right."""
    hands, out = iter(("l", "r")), []
    for r in rows:
        out.append(dict(r, hand=next(hands, r.get("hand"))) if r.get("at") == "hand" else r)
    return out


WORN_KEYS = ("wear_add", "face_covered", "wear_set", "wear_back_set", "worn_sheet")


FIGHT_POSE = "weapon_ready"    # the stance of someone who fights with a thing in the hand: a guard, the thing in
#                                front pointed at the foe (held up beside the head it read as a stick held up like a
#                                torch, and close in it lay behind the head and was redrawn as a club, 11-20)
FIGHT_BACK_M = (1.0, 2.6)      # the fight camera on the creature alone: this far behind the fighter and to their side
#                                (from beside them, the fighter out of its frame: 2.4 m behind and 0.9 m aside it was
#                                a back in the foreground, and what the place has behind the creature, 11-20)
FIGHT_SIDE_M = 5.0             # the side-on fight camera: at least this far off the line between the two
FIGHT_EYE_M = 1.7


def fights(k: str, v: dict, cam: dict, crt: dict) -> bool:
    """Whether person k fights a creature in this shot: an action shot, k standing with a thing in a hand and facing
    a creature of the scene (the pass's answer for whom they face); or any shot but a talk shot once the two have
    squared up (a fighter's guard does not drop between the blows: in the quiet shots of a fight his sword hung
    at his side, 11-20)."""
    return (k not in crt and v.get("pose") == "standing"
            and (cam.get("shot_kind") == "action" or (v.get("foe") in crt and cam.get("shot_kind") != "talk"))
            and (v.get("face_target") in crt or v.get("foe") in crt)
            and any(r.get("at") == "hand" for r in (cam.get("carried") or {}).get(k) or []))


def foe_aim(v: dict, st: dict, crt: dict) -> list | None:
    """Where a fighter's guard points (plan metres): the near part of the creature they face, its head toward them,
    at 0.6 of its height (its head to its middle); None with no creature in the staging."""
    kc = v.get("face_target") if v.get("face_target") in crt else v.get("foe")
    if kc not in crt or kc not in st or not v.get("position"):
        return None
    p, q = v["position"], st[kc]["position"]
    d = math.dist(p[:2], q[:2])
    r = float(crt[kc].get("radius_m") or 2.0)
    u = [(q[0] - p[0]) / d, (q[1] - p[1]) / d] if d > 1e-6 else [0.0, 0.0]
    k = min(0.5 * r, 0.5 * d)
    hz = 0.6 * float((crt[kc].get("size_m") or [0, 0, 3.0])[2])
    return [round(q[0] - u[0] * k, 3), round(q[1] - u[1] * k, 3), round(max(hz, 0.5), 3)]


def standoff_view(cam: dict, kp: str, kc: str) -> bool | None:
    """How a shot of a standoff between person kp and creature kc is seen. False: side-on, the two facing each
    other across the frame (a shot that shows both, or names neither). True: over the person's shoulder at the
    creature (a shot of the creature alone: what it does, it does at them). None: the shot keeps its own camera (a
    talk shot, a close shot of the person alone, a shot of someone else)."""
    shown = cam.get("show") or cam.get("listed") or [x for x in (cam.get("subject"),) if x]
    if cam.get("shot_kind") == "talk" or (shown and kp not in shown and (kc not in shown or len(shown) > 1)):
        return None                          # (... or of the creature with someone else)
    if shown and kc not in shown:
        return None if cam.get("kind") == "close" else False
    return bool(shown) and kp not in shown


FIGHT_SWING_DEG = 32.0          # the side-on fight cameras swing this far round the two, either way


def fight_camera(ppos: list, cpos: list, rig: dict, over_shoulder: bool, take: int = 0) -> dict | None:
    """A camera on a fight between a person (at ppos) and a far larger creature (at cpos; its rig's size), at the
    person's scale: over their shoulder toward the creature's head, the fighter's back and raised arm on one side
    and the creature filling the rest; or side-on, the fighter on one side and the creature's head and forequarters
    on the other. (A camera that held the whole creature made the fighter a speck; one behind him showed his back
    and a small beast far off: of seven fight shots one could be drawn as a fight, 11-19.) The side-on shots of one
    fight are not one camera: by take they are square on the two, swung round toward the fighter's back (the
    creature turned to the camera), swung round toward the creature's (the fighter's face seen), all from the
    same side of the line between them. (Nobody
    moved between shots and every side-on shot had the same camera: nine keys were one frame, the fighter a
    quarter of its height, 11-20.)"""
    d = math.dist(ppos[:2], cpos[:2])
    if d < 1.0:
        return None
    u = [(cpos[0] - ppos[0]) / d, (cpos[1] - ppos[1]) / d]
    n = [-u[1], u[0]]
    if n[1] > 0:                             # (from the nearer side of the place, y low: a place in the open's
        n = [-n[0], -n[1]]                   # picture is seen from there, and what it shows lies beyond)
    r = float(rig.get("radius_m") or 2.0)
    hz = max(2.0, 0.7 * float((rig.get("size_m") or [0, 0, 3.0])[2]))
    head = [cpos[0] - u[0] * 0.5 * r, cpos[1] - u[1] * 0.5 * r, hz]       # (its head: toward whom it fights)
    rnd = lambda q: [round(x, 3) for x in q]
    if over_shoulder:
        p = [ppos[0] - u[0] * FIGHT_BACK_M[0] + n[0] * FIGHT_BACK_M[1],
             ppos[1] - u[1] * FIGHT_BACK_M[0] + n[1] * FIGHT_BACK_M[1], FIGHT_EYE_M]
        return {"position": rnd(p), "target": rnd(head), "focal_mm": 28, "fight_view": "shoulder"}
    # (every one of them has the two whole and apart in it, the creature at its size: a camera in close on the
    # fighter cut the creature at the chin, one low behind him looked along the ground at a wall, 11-20. The
    # picture a fight clip starts from is the two squared up, seen whole; the clip plays the blow.)
    # (not from round toward the creature's back: there its dark flank filled the near half of the frame and the
    # fighter was small beyond it, twice, 11-20; the third is the square view from further off)
    view = ("wide", "fighter", "far")[take % 3]
    off = max(FIGHT_SIDE_M, 0.9 * d)
    mid = [(ppos[0] + head[0]) / 2, (ppos[1] + head[1]) / 2, (1.4 + hz) / 2]
    a = math.radians({"wide": 0.0, "fighter": FIGHT_SWING_DEG, "far": 0.4 * FIGHT_SWING_DEG}[view])
    v = [n[0] * math.cos(a) - u[0] * math.sin(a), n[1] * math.cos(a) - u[1] * math.sin(a)]
    k = {"wide": 1.0, "fighter": 1.1, "far": 1.4}[view]
    p = [mid[0] + v[0] * off * k, mid[1] + v[1] * off * k, 1.5 if view != "fighter" else 1.7]
    return {"position": rnd(p), "target": rnd(mid), "focal_mm": 35, "fight_view": view}


WORK_CAM_M = {"close": 1.6, "medium": 2.3}      # the work camera's distance from the piece, by the shot's framing
WORK_CAM_FAR_M = 3.2                            # ... for any wider framing
WORK_CAM_DEG = (40, -40, 65, -65, 20, -20)      # ... its turns off the line worker -> piece, first choice by take
WORK_CAM_Z_M = 1.55
WORK_FORE, WORK_FORE_TOP_M = 0.5, 0.3            # the camera's part of its line to the work: nothing over this high
WORK_FRAME_DEG = 38.0                           # half the work camera's frame, for who stands in its foreground
WORK_HEAD_M = 1.55                              # a standing worker's head, for the camera's line to it


def work_camera(wpos: list, thing: dict, kind: str, others: list, things: list, room: list, walls: bool = True,
                take: int = 0) -> dict | None:
    """A camera on someone at work at a piece of the set (at wpos, turned to it): from beyond the piece, to one
    side of the line from them to it, looking back over the piece's top at them - their face, their hands, the
    tool and what lies on the piece in one frame. Inside the place, clear of the pieces, nobody else between it and
    the work; the side changes with take. None when no such place is free (the shot keeps its own camera). (A
    work shot was framed as a shot of the person: of a dozen shots of a smith forging, the piece he struck was in
    frame in two, 11-20.)"""
    if len(thing.get("center") or []) < 3 or len(thing.get("half") or []) < 3:
        return None
    c, top = thing["center"], thing["center"][2] + thing["half"][2]
    d = math.dist(wpos[:2], c[:2])
    if d < 0.2:
        return None
    u = [(c[0] - wpos[0]) / d, (c[1] - wpos[1]) / d]
    # the work shots of one place are not one camera: by take, over the piece at the worker; close on the work
    # itself (hands, tool, what lies on the piece); from the side, worker and piece in profile. A full or wide
    # framing keeps the first, from further off.
    view = take % 3 if kind in WORK_CAM_M else 0
    dist = WORK_CAM_M.get(kind, WORK_CAM_FAR_M) + max(thing["half"][:2])
    W, D = bounds(room, 0.0, walls)
    head = [wpos[0], wpos[1], WORK_HEAD_M]
    work = [c[0], c[1], top + 0.1]
    # (every view has the worker and the piece in it: the close one leans to the work, it does not leave them out)
    tgt = [(c[0] + wpos[0]) / 2, (c[1] + wpos[1]) / 2, (top + 1.35) / 2]
    turns, z, focal = WORK_CAM_DEG, WORK_CAM_Z_M, 35
    if view == 1:
        dist = WORK_CAM_M["close"] + max(thing["half"][:2])
        tgt = [c[0] * 0.65 + wpos[0] * 0.35, c[1] * 0.65 + wpos[1] * 0.35, (top + 1.2) / 2]
    elif view == 2:
        turns = (80, -80, 65, -65, 90, -90)      # (never past square: from behind their shoulder it is their back)
    if (take // 3) % 2:
        turns = tuple(-x for x in turns)
    rest = [t for t in things if t is not thing and t.get("kind") == "fixture"]

    def sees(q3, p3):
        """Nothing of the set stands in the straight line from q3 to p3. Near the work a piece under the line does
        not; in the camera's half of the way any piece above WORK_FORE_TOP_M does (it fills the foreground)."""
        for i in range(1, 12):
            f_ = i / 12
            x, y, h = (q3[j] + (p3[j] - q3[j]) * f_ for j in range(3))
            for t in rest:
                if abs(x - t["center"][0]) < t["half"][0] + 0.1 and abs(y - t["center"][1]) < t["half"][1] + 0.1 \
                        and t["center"][2] + t["half"][2] > (h - 0.1 if f_ > WORK_FORE else WORK_FORE_TOP_M):
                    return False
        return True
    # (in a small room the full distance is past the wall or behind another piece: then nearer; and no other piece
    # between the camera and the worker's head or the work, the pieces right beside the work too - one stood with a
    # hearth filling its frame, the work behind it, 11-20; again from behind a hearth a hand from the work, 11-20)
    for deg, k_ in [(g, k) for k in (1.0, 0.8, 0.65) for g in turns]:
        a = math.radians(deg)
        v = [u[0] * math.cos(a) - u[1] * math.sin(a), u[0] * math.sin(a) + u[1] * math.cos(a)]
        q = [c[0] + v[0] * dist * k_, c[1] + v[1] * dist * k_]
        if (walls and (abs(q[0]) > W or abs(q[1]) > D)) or not clear_of_fixtures(q, rest):
            continue
        if not (sees(q + [z], head) and sees(q + [z], work)):
            continue
        L = math.dist(q, tgt[:2])
        hid = False
        for o in others:                         # (nobody else between the camera and the work)
            t_ = max(0.0, min(1.0, ((o[0] - q[0]) * (tgt[0] - q[0]) + (o[1] - q[1]) * (tgt[1] - q[1])) / (L * L)))
            if math.dist(o[:2], [q[0] + (tgt[0] - q[0]) * t_, q[1] + (tgt[1] - q[1]) * t_]) < 0.6:
                hid = True
            # (nor nearer the camera than the work and inside its frame: a head filled the frame's corner, 11-20)
            do = math.dist(o[:2], q)
            if 0.05 < do < L and ((o[0] - q[0]) * (tgt[0] - q[0]) + (o[1] - q[1]) * (tgt[1] - q[1])) / (do * L) > \
                    math.cos(math.radians(WORK_FRAME_DEG)):
                hid = True
        if hid:
            continue
        return {"position": [round(q[0], 3), round(q[1], 3), z], "target": [round(x, 3) for x in tgt],
                "focal_mm": focal}
    return None


def not_on_head(cam: dict, key: str) -> None:
    """A shot's carried things without what this person has on the head (their figure is made wearing it)."""
    rows = (cam.get("carried") or {}).get(key)
    if rows:
        cam["carried"][key] = [r for r in rows if r.get("at") != "head"]
        if not cam["carried"][key]:
            del cam["carried"][key]


def own_variant(library: Path, base: str, var: str | None) -> str | None:
    """The library figure built for this worn state of base (recorded "variant_of": base), else None."""
    if not var:
        return None
    p = library / "character" / f"{base}_{var}" / "character.json"
    try:
        return p.parent.name if p.exists() and json.loads(p.read_text()).get("variant_of") == base else None
    except ValueError:
        return None


def action_misfit(pz: str, k: str, v: dict, st: dict, worn: str | None, things: list,
                  carried: list | None = None) -> str | None:
    """Why a start action contradicts the staged scene's facts (None: it fits): a standing action for a person
    sitting, a hand on the mantel away from the fireplace, a veil lifted that is not down, a reach with nobody
    seated in the shot to reach, a guard with nothing in the hand (carried: the shot's carried rows for k)."""
    sitting = v.get("pose") == "sitting"
    if pz in ("stand_hand_on_mantel", "gesture_point", "reach_to_forearm") + WORK_ACTIONS + READY_ACTIONS and sitting:
        return f"{k} is sitting"
    if pz in READY_ACTIONS + PLAY_ACTIONS and not any(r.get("at") == "hand" for r in carried or []):
        return f"{k} has nothing in the hand"
    if pz == "stand_hand_on_mantel":
        fire = [t for t in things if t.get("fire")]
        if not fire or min(math.dist(v["position"], t["center"][:2]) - max(t["half"][:2]) for t in fire) > 0.6:
            return f"{k} is not at the fireplace"
    if pz == "veil_lift" and worn != "veiled":
        return f"{k}'s veil is not down ({worn or 'no veil'})"
    if pz == "reach_to_forearm" and not any(kk != k and vv.get("pose") == "sitting" for kk, vv in st.items()):
        return "nobody in the shot sits to be reached"
    return None


def reach_place(tgt: dict, his: list, things: list, room: list, gap: float = 0.5) -> tuple:
    """Where a standing person stands to lay a hand on a (seated) person's forearm: beside them, at the side
    nearer where he was, a little forward of their hips; clear of fixtures and inside the room. -> (pos, side)."""
    f = tgt.get("facing") or [0.0, -1.0]
    n = math.hypot(*f) or 1.0
    f = [f[0] / n, f[1] / n]
    left = [-f[1], f[0]]
    W, D = room[0] / 2 - 0.3, room[1] / 2 - 0.3
    best = None
    for side, sx in (("l", 1.0), ("r", -1.0)):
        p = [tgt["position"][0] + sx * left[0] * gap + f[0] * 0.3, tgt["position"][1] + sx * left[1] * gap + f[1] * 0.3]
        bad = abs(p[0]) > W or abs(p[1]) > D or any(
            t["kind"] == "fixture" and abs(p[0] - t["center"][0]) < t["half"][0] + 0.2
            and abs(p[1] - t["center"][1]) < t["half"][1] + 0.2 for t in things)
        cand = (bad, math.dist(p, his), [round(p[0], 3), round(p[1], 3)], side)
        best = cand if best is None or cand[:2] < best[:2] else best
    return best[2], best[3]


def acts_in(sh: dict, name: str, looks: dict, cast: dict) -> bool:
    """The shot's own action line names this person (by name, or by the look phrase the director's lines use)."""
    from shot_people import phrase as look_phrase_of
    line = str(sh.get("action") or "").lower()
    ph = (look_phrase_of(looks.get(name, ""), cast.get(name)) or "").lower()
    return bool(line) and (name.lower() in line or bool(ph and ph in line))


INSERT_DIST = {"close": 1.6, "medium": 2.6, "full": 4.0, "wide": 5.0}
VARY = [(0.0, 0.0), (40.0, 0.0), (-40.0, -8.0), (70.0, 5.0), (-70.0, 0.0)]   # (az, el) deg for action shots in a row


INSERT_CLEAR_M = 1.0      # an empty shot's camera stands this far clear of the standing things it is not on


def insert_camera(sh: dict, things: list, set_named: dict, room: list, name: str, sid: str,
                  walls: bool = True) -> dict:
    """A camera for a shot with nobody of the cast in it: on the set's thing the shot's words name (its own name,
    or a place phrase of the set that shares the most words with them), else the room's middle; from the room's
    side of it, at standing eye height, kept inside the room. -> camera (with "_on": what it is on). A place in
    the open whose things the words do not name, framed wide: the place as its picture sees it, from the stage's
    near edge (y low) looking the picture's way, and "free" - drawn from that view without the staging's shapes
    holding it, since what such a shot is of (riders, a crowd) is nobody the set or the cast has (redraw.drawn_free).
    (Seven riders leaving a tower: the camera stood five metres from the tower's wall, and the key was the wall,
    11-20.)"""
    w = words(" ".join(str(sh.get(k) or "") for k in ("action", "picture", "keyshot", "setting")))
    best, on = None, "the room's middle"
    for ph, item in (set_named or {}).items():
        t = next((x for x in things if x["name"] == item), None)
        sc = len(words(ph) & w - {"the", "a", "an", "at", "on", "in", "of", "to", "by", "toward", "left", "right"})
        if t and sc and (best is None or sc > best[0]):
            best = (sc, t)
    if best is None:
        t_, _ = thing_for(" ".join(sorted(w)), things, {})
        best = (1, t_) if t_ else None
    kind = FRAMING.get(sh.get("framing") or "medium", "medium")
    if best is None and not walls and kind in ("wide", "full"):
        return {"name": name, "plan_id": sid, "kind": "wide", "subject": None, "speaker": None, "show": None,
                "listed": None, "shot_kind": sh.get("kind"), "insert": True, "free": True,
                "position": [0.0, round(-(room[1] / 2 - 0.4), 3), 1.6], "target": [0.0, round(room[1] / 2, 3), 1.4],
                "focal_mm": 24, "_on": "the place as its picture sees it", "_looks": {}, "_poses": {}, "_worn": {},
                **({"moment": str(sh.get("moment") or sh.get("action")).strip()}
                   if str(sh.get("moment") or sh.get("action") or "").strip() else {}),
                **({"setting": sh["setting"]} if sh.get("setting") else {})}
    if best:
        t = best[1]
        c, top = list(t["center"][:2]), float(t["center"][2] + t["half"][2]) if len(t["center"]) > 2 else 1.0
        on = t["name"]
    else:
        c, top = [0.0, 0.0], 1.2
    v = [-c[0], -c[1]]
    n_ = math.hypot(*v)
    v = [v[0] / n_, v[1] / n_] if n_ > 0.3 else [0.0, -1.0]
    W, D = room[0] / 2 - 0.4, room[1] / 2 - 0.4
    d = INSERT_DIST.get(kind, 2.6)
    p = None
    # the standpoint: on the room's side of the thing, turned a little at a time and stepped back, the first one
    # inside the room and clear of every standing thing (one set inside a hearth photographed its inside, 11-17)
    # ... first a standpoint well clear of them with an open line to what it is on (a pace from a tower's wall,
    # the wall was half the picture, 11-19), then any clear one
    rest = [t for t in things if t is not (best or [0, None])[1]]
    for gap in (INSERT_CLEAR_M, 0.05):
        for dd in (d, d * 1.3, d * 0.75, d * 1.7):
            for da in (0, 20, -20, 40, -40, 60, -60, 90, -90):
                ca, sa = math.cos(math.radians(da)), math.sin(math.radians(da))
                u = [v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca]
                q = [c[0] + u[0] * dd, c[1] + u[1] * dd]
                if abs(q[0]) > W or abs(q[1]) > D or not clear_of_fixtures(q, rest, gap):
                    continue
                if gap > 0.05 and not all(clear_of_fixtures([q[0] + (c[0] - q[0]) * k / 8, q[1] + (c[1] - q[1]) * k / 8],
                                                            rest, 0.05) for k in range(1, 8)):
                    continue
                p = [q[0], q[1], 1.5]
                break
            if p:
                break
        if p:
            break
    p = p or [max(-W, min(W, c[0] + v[0] * d)), max(-D, min(D, c[1] + v[1] * d)), 1.5]
    return {"name": name, "plan_id": sid, "kind": kind, "subject": None, "speaker": None, "show": None,
            "listed": None, "shot_kind": sh.get("kind"), "insert": True, "position": [round(x, 3) for x in p],
            "target": [round(c[0], 3), round(c[1], 3), round(max(0.5, top * 0.7), 3)],
            "focal_mm": FOCAL.get(kind, 50), "_on": on, "_looks": {}, "_poses": {}, "_worn": {},
            # the director's own words for what the shot shows (the redraw draws what the set has no piece for)
            **({"setting": sh["setting"]} if sh.get("setting") else {})}


def mark_move(cam: dict, f: dict, k: str, st: dict, thing: str | None = None, landmark: str | None = None):
    """A move during the shot: its camera holds the mover's start, path and end place in one frame (speakers.py
    move_points), and the thing the end place is named by (the place words' landmark: "by the fire") in view from
    its front; without it a plan move shot was framed on the start alone, then from behind the fire (11-12)."""
    to = thing or st.get("seat")
    if not to:
        return
    cam["mover"], cam["move_to"] = k, to
    f["camera"].update(mover=k, move_to=to)
    if landmark and landmark != to:
        cam["move_landmark"] = landmark
        f["camera"]["move_landmark"] = landmark


def main():
    a = sys.argv[1:]
    plan = json.loads(Path(a[0]).read_text())
    # who speaks in a talk shot is the audio's own fact (the plan's event): a shot the plan wrote without its
    # speaker (its over-the-shoulder shots) is still that person's line, and the camera's face rules are theirs
    # (28 such shots were staged with no speaker: the back of a head filled the frame and hid the one speaking)
    for e_ in plan.get("_es") or []:
        for sh_ in plan.get(e_.get("id")) or []:
            if isinstance(sh_, dict) and sh_.get("kind") == "talk" and not sh_.get("speaker") and e_.get("speaker"):
                sh_["speaker"] = e_["speaker"]
    ids = [x.strip() for x in Path(a[1]).read_text().splitlines() if x.strip()]
    cast = json.loads(Path(a[2]).read_text())
    out = Path(a[3])
    out.mkdir(parents=True, exist_ok=True)
    answers = json.loads(Path(a[a.index("--answers") + 1]).read_text()) if "--answers" in a else {}
    sp = json.loads(Path(a[a.index("--pass") + 1]).read_text()) if "--pass" in a else None    # staging_pass.py
    # the made objects of what people wear on the head or carry in a hand: {the thing's words: library prop id}
    made_ = json.loads(Path(a[a.index("--carried") + 1]).read_text()) if "--carried" in a else {}
    # the book's look file: each person's look phrase, as the director's lines name people ("the man in the ...")
    look_txt = (json.loads(Path(a[a.index("--looks") + 1]).read_text()).get("characters") or {}) if "--looks" in a else {}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from shot_people import phrase as look_phrase_of
    library = Path(C.LIBRARY)
    first = plan[ids[0]][0]
    questions, facts = [], {}
    set_id, sets = pick_set(first, library)
    questions.append(q_set(first, sets))
    set_id = answers.get("set", set_id)
    set_dir = library / "set" / set_id
    things = setfacts.things(set_dir)
    # a set built from a place spec maps every plan phrase to its item (place_build.py); a phrase found there is
    # that item, no name matching
    np_path = set_dir / "named_places.json"
    set_named = json.loads(np_path.read_text()) if np_path.exists() else {}
    set_spec_ = json.loads((set_dir / "spec.json").read_text())
    room = set_spec_["size_m"]
    STAGE_DIR[:] = stage_dir(set_spec_)            # (a place built from one picture: its people stand across it)
    things += water_things(set_spec_)              # (a place in the open's water: nobody is stood in it)
    walled = bool(set_spec_.get("structure"))       # (a place in the open has none: no walls round its stage)
    views_ = set_spec_.get("views") or {}          # a cabin's backdrops, one for each thing the story sees outside
    seats = seating.seats(library, set_dir)
    beds = lying.beds(library, set_dir)
    for n, s_ in seats.items():
        s_["name"] = n
    names = {nm: v["key"] for nm, v in cast.items()}
    # the cast's creatures (a library creature with its rig, creature.py): staged standing on the ground in a pose
    # of their own clip, never seated or bedded; everyone keeps their reach clear of them
    crt = {v["key"]: r_ for v in cast.values() if v.get("creature") and (r_ := creature.rig(library, v["id"]))}
    cpose = {}                  # creature key -> the pose the pass chose last (carried while no shot says another)
    run_ = [None, 0]            # the people of the last action shot, and how many action shots in a row had them

    def ring(k, placed=()):
        """Where the others stand, a creature as [x, y, its reach] (too_near)."""
        out_ = [list(v["position"][:2]) + ([crt[kk]["radius_m"]] if kk in crt else [])
                for kk, v in state.items() if kk != k]
        return out_ + [list(e["position"][:2]) + ([e["_r"]] if e.get("_r") else []) for e in placed]
    named = [t for t in things if t["kind"] in ("fixture", "window", "door")]
    state: dict = {}            # key -> {"at", "pose", "position", "facing", "seat", "facing_words"}
    groups, cur, cur_sig = [], [], None
    worn: dict = {}
    in_hand: dict = {}          # key -> what they last had in their hands (held_on)
    playing: dict = {}          # key -> the instrument pose they are in (played_on)

    def apply_place(k, pa, why):
        others = ring(k)
        taken = {v.get("seat") for kk, v in state.items() if kk != k}
        pl = place_from(pa, things, seats, room, others, taken)
        if pl:
            state[k] = dict(state.get(k) or {"facing_words": None}, at=f"{pa.get('side')} {pa['thing']}", **pl)
            return why
        return None

    if sp:                                       # worn state at the scene's start (the pass's closed answer)
        for nm, w in (sp.get("worn_start") or {}).items():
            if nm in names and w not in NOTS:
                worn[names[nm]] = w
    away = {}                                    # people out of the room: their last state, for their return
    # someone who goes out during a shot is in that shot's opening picture and gone from a later one: the first
    # after it, or after the last shot before the next spoken line whose plan still shows them (the passage after
    # a line covers the pictures up to the next line: a man was gone from the very shot of his striding out)
    gone_from = {}
    for sid_, who_ in ((sp or {}).get("exits") or {}).items():
        if sid_ not in ids:
            continue
        for x in who_:
            j = k = ids.index(sid_)
            while k + 1 < len(ids) and plan[ids[k + 1]][0].get("kind") != "talk":
                k += 1
                if any(q.get("name") == x for q in plan[ids[k]][0].get("people") or []):
                    j = k
            gone_from.setdefault(j + 1, []).append(x)
    # a return to this place after the story was elsewhere is a new visit: only the people its own shots name are
    # there (11-16: a mother seated on a bed in the evening scene was still staged there in the night scene hours
    # later, with only the father and daughter in it, covering a third of the frame)
    order_ = [k_ for k_ in plan if not str(k_).startswith("_")]
    pos_ = {k_: i_ for i_, k_ in enumerate(order_)}
    visit, v_ = [], 0
    for i_, sid_ in enumerate(ids):
        if i_ and sid_ in pos_ and ids[i_ - 1] in pos_ and any(
                (plan[order_[j_]][0].get("place") or "") != (plan[sid_][0].get("place") or "")
                for j_ in range(pos_[ids[i_ - 1]] + 1, pos_[sid_])):
            v_ += 1
        visit.append(v_)
    opened = open_with_people(plan, ids, visit)
    for s_, src_ in opened.items():      # (and how near they are to each other there)
        if isinstance(sp, dict) and (sp.get("near") or {}).get(src_) and s_ not in sp["near"]:
            sp["near"][s_] = sp["near"][src_]
    named_in = {}
    for i_, sid_ in enumerate(ids):
        named_in.setdefault(visit[i_], set()).update(
            names[q["name"]] for q in plan[sid_][0].get("people") or [] if q.get("name") in names)
    for n, sid in enumerate(ids, 1):
        sh = plan[sid][0]
        sp = sp or {}
        for nm, w in ((sp.get("worn_plan") or {}).get(sid) or {}).items():     # the book's wardrobe changes here
            if nm in names and w not in NOTS:
                worn[names[nm]] = w
        if n > 1 and visit[n - 1] != visit[n - 2]:
            for k_ in [k_ for k_ in state if k_ not in named_in[visit[n - 1]]]:
                state.pop(k_)
        pa = {"camera": (sp.get("camera") or {}).get(sid), "pose": (sp.get("pose") or {}).get(sid) or {}}
        for nm_, pz_ in ((sp.get("creature_pose") or {}).get(sid) or {}).items():     # a creature's pose, carried
            if names.get(nm_) in crt and pz_ in crt[names[nm_]]["poses"]:
                cpose[names[nm_]] = pz_
        if cpose:
            pa["pose"] = dict(pa["pose"], **{nm_: cpose[k_] for nm_, k_ in names.items() if k_ in cpose})
        used = []
        # presence: a person leaves only on an exit the passage states in this shot's own span, and comes back on a
        # stated entrance, at the place they left from; a pass without exit answers (only "who is in the room")
        # takes nobody out (not being mentioned is not leaving, 11-12)
        for x in gone_from.get(n - 1) or []:
            if x in names and names[x] in state:
                away[names[x]] = state.pop(names[x])
                used.append(f"{names[x]} goes out of the room")
        for x in (sp.get("enters") or {}).get(sid) or []:
            if x in names and names[x] not in state and names[x] in away:
                state[names[x]] = away.pop(names[x])
                used.append(f"{names[x]} comes back into the room")
        f = {"id": sid, "kind": sh.get("kind"), "framing": sh.get("framing"), "speaker": sh.get("speaker"),
             "listener": sh.get("listener"), "people_listed": [p["name"] for p in sh.get("people") or []],
             "tight": sh.get("tight"), "defaults": [], "missing": []}
        if sid in opened:
            f["missing"].append(f"people and places (the scene opens on this shot: as shot {opened[sid]} first has them)")
        elif not sh.get("people"):
            f["missing"].append("people and places (carried from the shot before)")
        questions.append(q_present(f"present:{sid}", (sh.get("action") or sh.get("picture") or ""), list(state)))
        if sh.get("kind") == "action":
            questions.append(q_change(f"change:{sid}", sh, {k: v for k, v in state.items()},
                                      ["stand", "stand_relaxed", "stand_hand_on_mantel", "gesture_point", "seated",
                                       "sit_armchair_relaxed", "veil_lift", "look_at_sleeve", "reach_forearm", "look"]))
            f["missing"].append("pose for the action's start; what the key must show")
        questions.append(q_worn(f"worn:{sid}", sh, [p["name"] for p in sh.get("people") or []]))
        placed_now = []
        if n == 1:                               # the scene's start: people the passage has already in the place
            for nm, th in ((sp or {}).get("present_at_start") or {}).items():
                k = names.get(nm)
                if not k or k in state or th in NOTS or any(q["name"] == nm for q in sh.get("people") or []):
                    continue
                sit_ = th in seats and k not in crt
                if apply_place(k, {"thing": th, "sitting": sit_, "side": "front", "distance": "near"}, "pass"):
                    state[k]["pose"] = "sitting" if sit_ else "standing"
                    state[k].setdefault("facing_words", None)
                    used.append(f"{k} is already in the place when the scene opens ({th}, the passage)")
        for p in sh.get("people") or []:
            k = names.get(p["name"])
            if not k:
                f["missing"].append(f"library figure for a person ({p['name'][:1]}.)")
                continue
            # a person the passage sent out stays out until it brings them back, whatever the plan's list says
            # (the plan kept a woman by the fire through the scene after she had left; the pass never sends out
            # someone who still has a line to speak in the scene)
            # ... unless the shot's own action line has them act in it (by name or look phrase): then they are there,
            # back by an entrance the passage left unsaid (a smith sent out by one line was kept out of the six
            # shots of him at the anvil that followed, and none of them could be staged, 11-16)
            if k in away and not acts_in(sh, p["name"], look_txt, cast):
                used.append(f"{k} is listed by the plan but has gone out (the passage)")
                continue
            if k in away:
                away.pop(k)
                used.append(f"{k} is back: the shot's action has them in it")
            if k in crt:                         # a creature: on the ground, its posture is its pose (the pass's)
                p = dict(p, pose="standing")
            prev = state.get(k)
            # lying where another person of this shot lies by the same words ("in the bunk bed", both): that
            # person's bed, shared (a bed taken by one sent the other to the next bed, 2 m away, 11-16)
            mate = next((names[q["name"]] for q in sh.get("people") or [] if q is not p and q.get("name") in names
                         and names[q["name"]] != k and q.get("pose") == "lying" == p.get("pose")
                         and (q.get("at") or "").strip().lower() == (p.get("at") or "").strip().lower()
                         and (state.get(names[q["name"]]) or {}).get("bed")), None)
            if prev and mate and prev.get("bed") != state[mate]["bed"]:
                prev = None                      # (placed again: into the mate's bed)
            if prev and prev["at"] == p.get("at") and prev["pose"] == p.get("pose"):
                prev["facing_words"] = p.get("facing")
                continue
            start = (sh.get("pose") or {}).get(p["name"])
            if (prev and sh.get("kind") == "action" and start and start != p.get("pose")
                    and (start == "sitting") == (prev["pose"] == "sitting")):
                # an action key shows the action's start: the plan's start posture is the person's current one (the
                # people list gave the end: "stands up from the armchair" staged standing beside it, 11-12)
                prev["facing_words"] = p.get("facing")
                continue
            qid = f"place:{sid}:{k}"
            questions.append(q_place(qid, f"{p.get('pose')} {p.get('at')}, facing {p.get('facing')}",
                                     named, [f"{v['pose']} {v['at']}" for kk, v in state.items() if kk != k]))
            ans = answers.get(qid)
            thing, rel = thing_for(p.get("at", ""), things, {nm: kk for nm, kk in names.items() if kk != k})
            stated = set_named.get(p.get("at") or "")
            # where the words send them: a thing of the set, or beside someone; None when they name neither (screen
            # words, "lower-left foreground"; a state, "reared up", "on the ground")
            t_name = ans.get("thing") if ans and ans.get("thing") not in NOTS else \
                stated if stated and stated != "floor" else (thing or {}).get("name")
            # (a part of the open ground the set marked for a phrase, sets/place_build.py land_marks, is where
            # someone coming into the place is stood; it sends nobody already there anywhere: the director gave
            # "lower-left foreground", a posture and "on the ground" each as a part of the land, the set a spot on
            # the ground for each, and a man was walked between them shot by shot, 11-19)
            if t_name and next((t_.get("kind") for t_ in things if t_["name"] == t_name), None) == "mark":
                t_name = None
            where = ["thing", t_name] if t_name else ["with", rel] if rel in state else None
            if prev and stays(prev, p.get("pose"), where, state, crt, bool(beds)):
                # people stay where they are from shot to shot: someone already placed moves only when the words
                # send them to another thing of the set or to someone they are not with (through one fight a man's
                # words were "lower-left foreground", "in the fading fire", "on the ground": placed anew for each,
                # he stood in five places up to 23 m apart, and the beast he fought in three, 11-19)
                prev.update(at=p.get("at"), pose=p.get("pose"), facing_words=p.get("facing"))
                f["defaults"].append(f"{k}: stays where they are")
                continue
            if ans:
                thing = next((t for t in things if t["name"] == ans.get("thing")), thing)
            elif stated == "floor":
                # the open floor: a few steps from where the shot's other people are (their own named things, or
                # where they already stand), else the room's middle. In a hangar the middle of the floor was 23 m
                # from the man at the workbench, and no wide shot held the two of them
                oth = [t_["center"][:2] for q_ in sh.get("people") or [] if q_.get("name") != p.get("name")
                       for t_ in things if t_["name"] == set_named.get(q_.get("at") or "")]
                oth += [v_["position"][:2] for kk, v_ in state.items() if kk != k and v_.get("position")]
                c_ = [sum(o[0] for o in oth) / len(oth), sum(o[1] for o in oth) / len(oth)] if oth else [0.0, 0.0]
                thing = {"name": "floor", "kind": "floor", "center": c_, "half": [0.0, 0.0], "with": bool(oth),
                         "apart": bool(oth) and looks_on(p, sh.get("people") or []),
                         "turned": bool(oth) and turned_to(p.get("name"), (sp or {}).get("face", {}).get(sid) or {},
                                                           [q_.get("name") for q_ in sh.get("people") or []
                                                            if q_.get("name") != p.get("name")]),
                         "line": [o for i_, o in enumerate(oth) if all(math.dist(o, q_) > 0.3 for q_ in oth[:i_])][:2]}
                f["defaults"].append(f"{k}: on the open floor ({'near the others' if oth else 'the middle of the room'})")
            elif stated:
                thing = next((t for t in things if t["name"] == stated), thing)
                f["defaults"].append(f"{k}: place from the set's named places ({stated})")
            else:
                f["defaults"].append(f"{k}: place by name match ({(thing or {}).get('name') or rel})")
            entry = {"at": p.get("at"), "pose": p.get("pose"), "facing_words": p.get("facing"),
                     # placed at a named thing of the set (not the open floor, not beside someone): an exact place
                     "named_at": bool(ans or (stated and stated != "floor")), "where": where}
            if p.get("pose") == "sitting":
                anchor = (thing or {}).get("center") or (state.get(rel) or {}).get("position") or [0, 0]
                # (a seat with several places, a sofa or a bed's edge, is free until every place is sat on)
                sat_n = {}
                for kk, v in state.items():
                    if kk != k and v.get("seat"):
                        sat_n[v["seat"]] = sat_n.get(v["seat"], 0) + 1
                cands = [s_ for s_ in seats.values()
                         if sat_n.get(s_["name"], 0) < len(s_["box"].get("places") or [0.0])]
                own = [s_ for s_ in cands if thing and s_["name"] == thing.get("name")]   # the named thing is a seat
                seat = own[0] if own else min(cands, key=lambda s_: math.dist(s_["pos"], anchor[:2])) if cands else None
                if seat:
                    entry.update(seat=seat["name"], position=list(seat["pos"]))
                    if thing and thing.get("name") != seat["name"]:
                        entry["at_thing"] = thing["name"]    # seated AT a thing (a desk, a table): see the cameras
                else:
                    # no seat is free in the place: they stand by it (reported), the scene is not lost for it (two
                    # sisters "sitting" in a bedroom with one chair stopped every shot of their talk)
                    others = ring(k, placed_now)
                    entry.update(seat=None, pose="standing", position=free_mark(anchor[:2], others, things, room))
                    f["defaults"].append(f"{k}: no free seat in the place; standing")
            elif p.get("pose") == "lying" and beds:
                # lying: in the bed the place words name, else the nearest one nobody lies in
                anchor = (thing or {}).get("center") or (state.get(rel) or {}).get("position") or [0, 0]
                taken_b = {v.get("bed") for kk, v in state.items() if kk != k}
                cands = [b_ for b_ in beds.values() if b_["name"] not in taken_b] or list(beds.values())
                own = [b_ for b_ in cands if thing and b_["name"] == thing.get("name")]
                # placed against someone who lies in a bed ("tucked against her father"): that same bed, beside
                # them (lying.plan lays them side by side, or gives the next bed when it is too narrow); one bed
                # each put a girl 2 m from the father she lay against, 11-16
                shared = beds.get((state.get(rel) or {}).get("bed") or "") if rel else None
                shared = shared or (beds.get(state[mate]["bed"]) if mate else None)
                bed = shared or (own[0] if own else min(cands, key=lambda b_: math.dist(b_["pos"], anchor[:2])))
                entry.update(seat=None, bed=bed["name"], position=[round(x, 3) for x in bed["frame"]["centre_xy"]])
            else:
                if rel and rel in state:
                    tgt = state[rel]["position"]
                    near = (0.5, 0.9)
                else:
                    tgt = (thing or {}).get("center") or [0, 0]
                    near = floor_near(thing) if (thing or {}).get("kind") == "floor" else (0.6, 1.6)
                others = ring(k, placed_now)
                own_r = (crt.get(k) or {}).get("radius_m", 0.0)
                # (a creature keeps its reach from people; from fixtures its body's half breadth: a rock or a cave
                # mouth may stand by its flank, and by its whole reach a 13 m stage with two rocks held no place)
                fix_r = 0.5 * min(crt[k]["size_m"][:2]) if k in crt else None
                if own_r:
                    entry["_r"] = own_r
                # (a creature too stands before a thing larger than itself, a rock face with its cave: placed by
                # its middle, a beast "at the cave mouth" stood behind the rock, 11-19)
                big_ = k in crt and thing and min((thing.get("half") or [0, 0])[:2]) > own_r
                fro = None if (rel and rel in state) or (k in crt and not big_) else front_of(thing)
                fro_off = 0.0
                if fro and fix_r is None:
                    fix_r = stand_clear(thing)
                if (thing or {}).get("base") and not (rel and rel in state) and k not in crt:
                    # a thing on a low base (a seat on a raised floor): stood on the floor before the base's front
                    # edge, not on the base (the figure would stand in it) nor beside the thing
                    tgt, near, fro = list(thing["base"]["edge"]), BASE_STAND_M, list(thing["base"]["front"])
                    fro_off = BASE_CLEAR_M
                # (a part of the land in the water itself, a ford, by the ground map's judge: the water is no bar)
                th_ = [t_ for t_ in things if t_["kind"] != "water"] if (thing or {}).get("wet") else things
                entry.update(seat=None, position=free_mark(tgt, others, th_, room, near, own_r, fix_r, walled, fro,
                                                           fro_off))
                ln_ = (thing or {}).get("line") or []
                if (thing or {}).get("apart") and len(ln_) == 2:
                    # (off the line of the two they watch, to its side: behind one of them they are hidden from
                    # the other, and from any camera that has both)
                    ux, uy = ln_[1][0] - ln_[0][0], ln_[1][1] - ln_[0][1]
                    L_ = math.hypot(ux, uy)
                    off_ = [(abs((m[0] - ln_[0][0]) * uy - (m[1] - ln_[0][1]) * ux) / L_, m) for m in
                            free_marks(tgt, others, th_, room, near, n=40, own_r=own_r, fix_r=fix_r, walls=walled,
                                       front=fro, front_off=fro_off)]
                    if off_:
                        entry["position"] = max(off_, key=lambda x: x[0])[1]
                # a vague place (a relation or a region): every allowed mark, the chosen one first; the staging's
                # camera rules may move the person among them (an exact place, a seat, never moves for the camera)
                entry["marks"] = [entry["position"]] + [m for m in free_marks(tgt, others, th_, room, near,
                                                                              own_r=own_r, fix_r=fix_r, walls=walled,
                                                                              front=fro, front_off=fro_off)
                                                        if math.dist(m, entry["position"]) > 0.2]
            old_ = state.get(k) or {}
            if old_.get("foe") in state and old_.get("position") and entry.get("pose") == old_.get("pose") == "standing":
                # squared up to a creature (or, the creature, to whoever fights it): they hold that ground, shot
                # after shot, wherever the plan's words for this shot would put them (between the shots of one
                # fight a man stood beside the beast, then 25 m off, then beside it, and the beast in three
                # places, 11-19)
                entry.update(position=list(old_["position"]), foe=old_["foe"], named_at=False, seat=None)
                entry.pop("marks", None)
                f["defaults"].append(f"{k}: holds their ground against {old_['foe']}")
            state[k] = entry
            placed_now.append(entry)
        # facing from the words: a person's name, else a named thing; with no words, what the shot is about (its
        # first listed person: the one who acts, or whom the line is about); a place word ("by the window") says
        # where someone stands, never which way they face. A person faced is kept as "facing_to": the staging
        # re-aims the body when anyone moves (11-12: the men turned to the glass after she was moved)
        focus = next((names[q["name"]] for q in sh.get("people") or [] if q["name"] in names), None)
        # whom each listed person faces, the pass's answer by name (the plan's facing words are screen directions);
        # kept for the shots after that do not list them, until a shot that lists them says otherwise
        face_ = {names[a_]: (names[b_] if b_ in names else AT_THING) for a_, b_ in
                 ((sp.get("face") or {}).get(sid) or {}).items() if a_ in names and (b_ in names or b_ == AT_THING)
                 and a_ != b_}
        for q in sh.get("people") or []:
            if names.get(q.get("name")) in state:
                state[names[q["name"]]]["face_target"] = face_.get(names[q["name"]])
        # a creature and whoever squares up to it in this shot (both listed, each facing the other): at striking
        # distance by its reach (the person moves; the creature holds its place). Only someone placed on the open
        # floor or by a relation moves: a person the plan puts at a named thing (a ledge) stays there
        here_ = {names[q["name"]] for q in sh.get("people") or [] if q.get("name") in names}
        for kp, v in state.items():          # (those squared up already face each other still, listed or not)
            if kp not in crt and v.get("foe") in crt and v["foe"] in state and state[v["foe"]].get("foe") == kp:
                v["face_target"], state[v["foe"]]["face_target"] = v["foe"], kp
        for kc in [x for x in state if x in crt and x in here_]:
            for kp, v in state.items():
                # (in an action shot the fight takes him off the mark the plan named: left at "the cave mouth" a man
                # with his sword out stood sixteen metres from the beast he fought, 11-19)
                # (... and there one of them facing the other is enough when the person has a thing in the hand: the
                # pass had a man raising his sword at a beast face "the thing he is at", and a beast he faced turn
                # to someone else; he stood twenty metres off in both)
                armed_ = sh.get("kind") == "action" and any(
                    t_.get("on") == "held" for nm_, ts_ in (((sp or {}).get("things") or {}).get(sid) or {}).items()
                    if names.get(nm_) == kp for t_ in ts_ or [])
                each_ = v.get("face_target") == kc and state[kc].get("face_target") == kp
                one_ = armed_ and (v.get("face_target") == kc or state[kc].get("face_target") == kp)
                if kp in crt or kp not in here_ or v.get("pose") != "standing" or \
                        (v.get("named_at") and sh.get("kind") != "action") or not (each_ or one_):
                    continue
                if one_ and not each_:       # they square up: each faces the other from here on
                    v["face_target"], state[kc]["face_target"] = kc, kp
                v["foe"], state[kc]["foe"] = kc, kp
                q = strike_place(state[kc]["position"], crt[kc]["radius_m"], v["position"],
                                 [list(o["position"][:2]) for kk, o in state.items() if kk not in (kp, kc)],
                                 things, room, walled, across=not walled)
                if q:
                    used.append(f"{kp} at striking distance from {kc} ({math.dist(q, state[kc]['position']):.1f} m)")
                    v["position"] = q
                    v.pop("marks", None)
        for k, v in state.items():
            fw = v.get("facing_words") or ""
            _, rel = thing_for(fw, things, {nm: kk for nm, kk in names.items() if kk != k})
            thing, _ = thing_for(fw, things, {})
            if v.get("face_target") in state and v["face_target"] != k:      # the pass's answer: whom, by name
                rel, thing = v["face_target"], None
            elif v.get("face_target") == AT_THING:                            # ... or the thing they work at
                at_t = next((t for t in things if t["name"] == set_named.get(v.get("at") or "")), None) or \
                    thing_for(v.get("at") or "", things, {})[0]
                if at_t:
                    rel, thing = None, at_t
            if not fw.strip() and not rel and focus and focus != k and focus in state:
                rel = focus
            v["facing_to"] = rel if rel in state and rel != k else None
            # (turned to the thing they work at is no turn of the shot: that thing is framed as where they work)
            v["facing_thing"] = thing["name"] if (thing and not v["facing_to"] and fw.strip()
                                                  and v.get("face_target") != AT_THING) else None
            tgt = (state.get(rel) or {}).get("position") if rel else (thing or {}).get("center")
            if tgt is None and len(state) > 1:                # "the camera", a group: toward the others' middle
                o = [vv["position"] for kk, vv in state.items() if kk != k]
                tgt = [sum(x[0] for x in o) / len(o), sum(x[1] for x in o) / len(o)]
            tgt = tgt or [0.0, 0.0]
            d = [tgt[0] - v["position"][0], tgt[1] - v["position"][1]]
            nd = math.hypot(*d)
            # (standing on what it would face, a facing is kept from before, else toward the room's front)
            v["facing"] = [round(d[0] / nd, 4), round(d[1] / nd, 4)] if nd > 1e-6 else \
                (v.get("facing") if any(v.get("facing") or []) else [0.0, -1.0])
            if not rel and thing and v.get("pose") != "sitting":
                v["facing"] = open_from_wall(v["position"], v["facing"], thing,
                                             [vv["position"] for kk, vv in state.items() if kk != k])
            elif nd > 1e-6 and k not in crt and rel not in crt:
                v["facing"] = stage_open(v["facing"], v.get("pose") == "standing")
        spk = names.get(sh.get("speaker")) if sh.get("speaker") else None
        lis = names.get(sh.get("listener")) if sh.get("listener") else None
        if sh.get("kind") == "talk" and spk in state:
            # the speaker faces whom he speaks to, from where everyone is now: the listener, else the middle of the
            # others; a place word or an earlier turn toward a thing does not hold him once he speaks (11-12: he
            # talked toward the window with her by the fire behind him)
            v = state[spk]
            if lis in state and lis != spk:
                tgt, v["facing_to"] = state[lis]["position"], lis
            else:
                o = [vv["position"] for kk, vv in state.items() if kk != spk]
                tgt, v["facing_to"] = ([sum(x[0] for x in o) / len(o), sum(x[1] for x in o) / len(o)] if o else None), None
            if tgt is not None:
                v["facing_thing"] = None
                d = [tgt[0] - v["position"][0], tgt[1] - v["position"][1]]
                nd = math.hypot(*d)
                if nd > 1e-6:
                    v["facing"] = [round(d[0] / nd, 4), round(d[1] / nd, 4)]
                    if spk not in crt and lis not in crt:
                        v["facing"] = stage_open(v["facing"], v.get("pose") == "standing")
        listed = [names[p["name"]] for p in sh.get("people") or [] if p["name"] in names]
        # (who is in the room now: someone the plan lists or has speaking who has left it, or whom the staging never
        # brought in, is not in the picture; with nobody at all in the room the shot is an insert too. 11-26: a
        # speaker the plan named with the room empty left no one to frame, and the whole place's staging stopped)
        if not state:
            spk, listed = None, []
        if not (spk or listed or state):
            # nobody of the cast is in this shot (a place or a thing seen alone, someone the book has no figure
            # for): an insert of the set, its camera on the set's thing the shot's words name, staged with nobody
            # (every shot ends with an engine key, 11-17; it used to get no camera and no key)
            ins = insert_camera(sh, things, set_named, room, f"s{n:02d}_{sid.split('-')[0]}", sid, walled)
            f["missing"].append(f"nobody of the cast in the shot: an insert of the set ({ins['_on']})")
            ins.pop("_on")
            facts[sid] = dict(f, camera=dict(ins))
            if cur:
                groups.append(cur)
            groups.append([(ins, {})])
            cur, cur_sig = [], None
            continue
        listed = [x for x in listed if x in state]          # (someone the plan lists who is out of the room)
        if spk not in state:
            spk = None
        if lis not in state:                 # (a listener out of the room is nobody's listener here)
            lis = None
        subj = spk or (listed[0] if listed else next(iter(state)))
        kind = FRAMING.get(sh.get("framing") or "medium", "medium")
        if sh.get("tight") and kind != "close":
            kind = "close"
        ck = pa.get("camera")
        if spk and ck not in NOTS:
            kind = {"close": "close", "wide": "full"}.get(ck, "medium" if kind == "close" and ck != "close" else kind)
        must = [names[x[0]] for x in (pa.get("must_show") or []) if isinstance(x, list) and x and x[0] in names]
        # whom the shot's line is about: the person named (by name or look phrase) first in its action line
        line = (sh.get("action") or "").lower()
        at = {k_: min([i for i in (line.find(nm.lower()),
                                   line.find((look_phrase_of(look_txt.get(nm, ""), cast.get(nm)) or "\0").lower()))
                       if i >= 0] or [10 ** 6]) for nm, k_ in names.items() if k_ in state}
        line_subj = min(at, key=at.get) if at and min(at.values()) < 10 ** 6 else None
        # an action shot's subject turning toward a named thing: that thing is in the frame (04:58: he turned to
        # the desk with the desk behind the camera and a round table in view; the video walked him to the table)
        ft = (state.get(line_subj) or {}).get("facing_thing") if sh.get("kind") == "action" and line_subj else None
        # someone seated at a desk or a table: it is in the frame of any shot wider than a close (11-12: 23 shots of a
        # man at his desk and no desk in any of them)
        # someone at work on a thing of the set with a tool in the hand, in an action shot the plan frames close:
        # the shot is of the work, so it is framed as a medium one with the thing in it. Framed on the head, a close
        # of a smith striking at his anvil was the back of his bowed head and a raised hammer, no anvil (11-18)
        st0 = state.get(subj) or {}
        if kind == "close" and sh.get("kind") == "action" and st0.get("named_at") and st0.get("face_target") == AT_THING \
                and any(str(p_.get("holds") or "").strip().lower() not in ("", "nothing", "none")
                        for p_ in sh.get("people") or [] if names.get(p_.get("name")) == subj):
            kind = "medium"
        at_thing = (state.get(subj) or {}).get("at_thing") if kind != "close" else None
        # someone standing at a named thing of the set in an action shot, turned to it (the pass: the thing they
        # work at, not a cave a beast lies before): it is in the
        # frame of any shot wider than a close (six shots of a smith striking at his anvil showed a wall, 11-17)
        if not at_thing and kind != "close" and sh.get("kind") == "action" and (state.get(subj) or {}).get("named_at") \
                and (state.get(subj) or {}).get("face_target") == AT_THING:    # (the pass: they work at it)
            st_ = state[subj]
            th_ = next((t for t in things if t["kind"] == "fixture" and t["name"] == set_named.get(st_.get("at") or "")), None)
            at_thing = th_["name"] if th_ else None
        cam = {"name": f"s{n:02d}_{sid.split('-')[0]}", "plan_id": sid, "kind": kind, "subject": subj,
               "line_subject": line_subj, "face_thing": ft, "at_thing": at_thing,
               "speaker": spk, "focal_mm": FOCAL[kind], "show": sorted(set(listed + must)) or None,
               # the people the plan lists for the shot: anyone else present stays out of its frame (speakers.py)
               "listed": sorted(set(listed + must)) if listed else None,
               "shot_kind": sh.get("kind")}
        # action shots in a row of the same people: each from another side (the same wide five times over, 11-17);
        # the scene tool's aimed camera turns round its subject by az and looks from el
        if sh.get("kind") == "action" and not spk:
            sig_ = tuple(sorted(listed))
            run_ = [sig_, run_[1] + 1] if run_[0] == sig_ else [sig_, 0]
            az_, el_ = VARY[run_[1] % len(VARY)]
            if az_ or el_:
                cam.update(az=az_, el=el_)
        if sh.get("setting"):                 # the director's own words for what this shot shows of the place
            cam["setting"] = sh["setting"]
        if views_.get("words"):               # what is outside in this shot: its backdrop, its words, the way ahead
            vn = (views_.get("by_shot") or {}).get(sid) or "outside_view"
            cam["view"] = {"name": vn, "words": views_["words"].get(vn, ""), "ahead": views_.get("ahead"),
                           "beyond_m": views_.get("beyond_m"), "width_m": views_.get("width_m"),
                           "all": sorted(views_["words"])}
        # what each person has on in this shot beyond their usual clothes (the pass's "wears": the director's
        # account from the whole story); the figures wear their usual look, so the redraw is told
        worn_ = {names[n_]: ", ".join(t_) for n_, t_ in (((sp or {}).get("wears") or {}).get(sid) or {}).items()
                 if n_ in names and t_}
        if worn_:
            cam["wears"] = worn_
        # each thing in a hand gets its own hand: the first the working (right) hand, a second the other; no third
        # (11-18: the hammer and the tongs were both put in one palm). The plan's name rides along for the staged
        # check's words ("the hammer's face is 9 cm from the work")
        obj_ = {}
        for n_, ts_ in (((sp or {}).get("things") or {}).get(sid) or {}).items():
            if n_ not in names:
                continue
            hands_ = iter(("r", "l"))
            for t_ in ts_:
                if t_["on"] not in CARRIED_ON or t_["what"] not in made_:
                    continue
                if t_["on"] == "head":
                    obj_.setdefault(names[n_], []).append({"asset": made_[t_["what"]], "at": "head", "name": t_["what"]})
                elif (h_ := next(hands_, None)):
                    obj_.setdefault(names[n_], []).append({"asset": made_[t_["what"]], "at": "hand", "hand": h_,
                                                           "name": t_["what"],
                                                           **({"glow": t_["glow"]} if t_.get("glow") else {})})
        # the thing being worked on (the pass's "worked"): a made object lying on the top of the named thing its
        # worker stands at (a smith's bar on his anvil; drawn as a piece of the set, it came out a star, 11-17)
        for n_, ts_ in (((sp or {}).get("things") or {}).get(sid) or {}).items():
            k_ = names.get(n_)
            at_ = next((t for t in things if t["kind"] == "fixture"
                        and t["name"] == set_named.get(str((state.get(k_) or {}).get("at") or ""))), None)
            for t_ in ts_:
                if t_["on"] == "worked" and t_["what"] in made_ and at_ and k_ in state:
                    obj_.setdefault(k_, []).append({"asset": made_[t_["what"]], "at": "on", "name": t_["what"], "xyz": [
                        at_["center"][0], at_["center"][1], round(at_["center"][2] + at_["half"][2], 3)],
                        **({"glow": t_["glow"]} if t_.get("glow") else {})})
        obj_ = held_on(in_hand, obj_, {names[n_] for n_ in ((sp or {}).get("things") or {}).get(sid) or {}
                                       if n_ in names}, set(state))
        obj_ = {k_: v_ for k_, v_ in obj_.items() if v_}
        if obj_:                              # (scene_job puts each on the person's head or in their hand)
            cam["carried"] = obj_
        shut_ = [names[n_] for n_ in ((sp or {}).get("eyes") or {}).get(sid) or {} if n_ in names]
        if shut_:                             # whose eyes are closed as the shot opens (the pass's closed answer)
            cam["eyes_closed"] = shut_
        # what each person holds in this shot (the plan's "holds", with the thing's look when the plan gives one):
        # the engine stages no hand props, so the redraw is told (47 shots held a helmet and no key showed one;
        # the video then invented it, on his head in one shot and in his hand in the next, 11-14)
        held = {names[p_["name"]]: (str(p_["holds"]).strip() + (f" ({lk})" if (lk := next(
                    (v_ for t_, v_ in (sh.get("things") or {}).items() if t_.lower() in str(p_["holds"]).lower()), "")) else ""))
                for p_ in sh.get("people") or [] if p_.get("name") in names and str(p_.get("holds") or "").strip()}
        # (a held thing that glows of itself in this shot, the pass's "glow": said, or the key drew it cold)
        for n_, ts_ in (((sp or {}).get("things") or {}).get(sid) or {}).items():
            g_ = next((t_["glow"] for t_ in ts_ if t_["on"] == "held" and t_.get("glow")), None)
            if g_ and names.get(n_) in held:
                held[names[n_]] += f", glowing {g_} along its whole length with its own heat and lighting what is near it"
        if held:
            cam["holds"] = held
        if lis:
            cam["listener"] = lis
            # the director's own picture of the shot has the listener in it (by name or look phrase): the camera
            # keeps them, never trading them for a clean single of the speaker (14 of 17 two-person talk keys showed
            # the speaker alone, the picture asking for the listener beyond, 11-17)
            pic = str(sh.get("picture") or sh.get("keyshot") or "").lower()
            lnm = sh.get("listener") or ""
            lph = (look_phrase_of(look_txt.get(lnm, ""), cast.get(lnm)) or "\0").lower()
            if lnm and (lnm.lower() in pic or lph in pic):
                cam["keep_listener"] = True
        if spk and ck == "over_the_shoulder" and lis:
            cam["ots"] = lis
        if spk and ck == "single" and not cam.get("keep_listener"):   # (the director's picture has them both)
            cam["single"] = True
            cam.pop("listener", None)
        looks = {}
        poses = {names[a_]: b_ for a_, b_ in (pa.get("pose") or {}).items() if a_ in names and b_ not in NOTS}
        poses = played_on(playing, poses, set(state), {k_ for k_, rows_ in (cam.get("carried") or {}).items()
                                                       if any(r_.get("at") == "hand" for r_ in rows_)})
        for k_ in [k_ for k_, pz_ in poses.items() if pz_ in PLAY_ACTIONS and (cam.get("carried") or {}).get(k_)]:
            cam["carried"][k_] = play_hands(cam["carried"][k_])
        # someone at work or holding a tool in an action shot: their working hand is in the frame and seen (in a
        # close of a smith the hammer and tongs the redraw was told of had nowhere to go but his head: "ears", a red
        # mask, 11-17)
        if sh.get("kind") == "action":
            hold_ = {k_ for k_, v_ in held.items() if v_.split(" (")[0].strip().lower() not in ("nothing", "none", "")}
            hj_ = [[k_, "hand_r"] for k_ in sorted(hold_ | {k_ for k_, pz_ in poses.items() if pz_ in WORK_ACTIONS})
                   if k_ in state and k_ not in crt]          # (a creature has no hand: what it holds is in its claws)
            if hj_:
                cam["show_joints"] = hj_
                # (a hand that only carries: wanted in the frame, given up where no camera holds it; a working
                # hand is the shot)
                if not any(poses.get(k_) in WORK_ACTIONS for k_, _ in hj_):
                    cam["joints_soft"] = True
        f["pass"] = {"used": used, "camera": ck, "pose": poses, "worn": dict(worn)}
        cam["_looks"] = looks
        cam["_poses"] = poses
        cam["_worn"] = dict(worn)
        f.update(camera=dict(cam), staged={k: {kk: v[kk] for kk in ("pose", "at", "seat", "position", "facing")}
                                           for k, v in state.items()})
        facts[sid] = f
        sig = json.dumps({k: (v["position"], v["pose"], v.get("seat")) for k, v in state.items()}, sort_keys=True) \
            + json.dumps([cam.get("_poses"), cam.get("_worn")], sort_keys=True)
        # a standing speaker who turns to whom he now speaks is a new staging (the group's people are posed once)
        turned = bool(cur) and sh.get("kind") == "talk" and spk in state and state[spk]["pose"] != "sitting" and \
            (cur[0][1].get(spk) or {}).get("facing_to") != state[spk].get("facing_to")
        if (sig != cur_sig or turned) and cur:
            groups.append(cur)
            cur = []
        cur_sig = sig
        cur.append((cam, json.loads(json.dumps(state))))
        pe = sh.get("pose_end") or {}
        # where a move ends, from the plan itself: a person whose posture changes over an action shot is placed, for
        # the shots that list nobody until then, where the plan next lists them (the plan's own later fact)
        if "--no-lookahead" not in a and sh.get("kind") == "action":
            for p in sh.get("people") or []:
                nm, k = p["name"], names.get(p["name"])
                if not k or k in crt or pe.get(nm) in (None, p.get("pose")):
                    continue
                nxt = next((q for sid2 in ids[n:] for q in plan[sid2][0].get("people") or [] if q["name"] == nm), None)
                if nxt and nxt.get("pose") == pe.get(nm):
                    thing, rel = thing_for(nxt.get("at", ""), things, {})
                    if thing:
                        if apply_place(k, {"thing": thing["name"] if thing["name"] not in seats else thing["name"],
                                           "sitting": nxt.get("pose") == "sitting", "side": "front", "distance": "near"},
                                       "plan"):
                            if nxt.get("pose") == "sitting" and thing["name"] not in seats:
                                anchor = thing["center"][:2]
                                taken = {v.get("seat") for kk, v in state.items() if kk != k}
                                cands = [s_ for s_ in seats.values() if s_["name"] not in taken]
                                if cands:
                                    seat = min(cands, key=lambda s_: math.dist(s_["pos"], anchor))
                                    state[k].update(seat=seat["name"], position=list(seat["pos"]))
                            state[k]["pose"] = nxt.get("pose")
                            state[k]["at"] = nxt.get("at")
                            mark_move(cam, f, k, state[k], landmark=thing["name"])
                            f.setdefault("pass", {}).setdefault("used", []).append(f"{k} ends where the plan next "
                                                                                    f"lists them: {nxt.get('at')}")
        for nm, th in ((sp.get("move_end") or {}).get(sid) or {}).items():     # where a move during this shot ends
            if nm in names and names[nm] not in crt:
                k = names[nm]
                was = (state.get(k) or {}).get("seat") or ""
                sitting = (pe.get(nm) or (state.get(k) or {}).get("pose")) == "sitting"
                if apply_place(k, {"thing": th, "sitting": sitting, "side": "front", "distance": "near"}, "pass"):
                    state[k]["pose"] = "sitting" if sitting else "standing"
                    if state[k].get("seat") != was:
                        f.setdefault("pass", {}).setdefault("used", []).append(f"{k} ends at {th}")
                        mark_move(cam, f, k, state[k], th, landmark=th)
        for nm, w in ((sp.get("worn_end") or {}).get(sid) or {}).items():      # worn state when the shot ends
            if nm in names and w not in NOTS:
                worn[names[nm]] = w
    if cur:
        groups.append(cur)
    specs = []
    at_work = []                             # the place's work shots so far (work_camera's take)
    side_on = []                             # the place's side-on fight shots so far (fight_camera's take)
    reached = {}                     # key -> (where the plan had them, where a reach put them, facing there)
    for gi, g in enumerate(groups, 1):
        st = g[0][1]
        chars, way, turned = {}, {}, {}
        cam0 = g[0][0]
        for k in list(reached):      # still where the plan had them before the reach, standing: they did not move
            if k in st and st[k]["position"] == reached[k][0] and st[k]["pose"] != "sitting" and \
                    (st[k].get("at"), st[k].get("facing_thing")) == reached[k][3]:    # (new place words or a turn
                # toward a thing are the plan moving them)
                for _, s_ in g:
                    if k in s_ and s_[k]["position"] == reached[k][0]:
                        s_[k]["position"] = list(reached[k][1])
                        s_[k].pop("marks", None)     # (the plan's marks are round the place they left)
                        to = s_.get(s_[k].get("facing_to") or "")
                        if to:                   # whom he faces, from where he now stands
                            d_ = [to["position"][0] - reached[k][1][0], to["position"][1] - reached[k][1][1]]
                            n__ = math.hypot(*d_) or 1.0
                            s_[k]["facing"] = [round(d_[0] / n__, 4), round(d_[1] / n__, 4)]
                        else:
                            s_[k]["facing"] = list(reached[k][2])
            else:
                del reached[k]
        for k, v in st.items():
            cv = next(c for c in cast.values() if c["key"] == k)
            w = (cam0.get("_worn") or {}).get(k)
            var = None if w in (None, "usual") else w.replace(" ", "_")
            if k in crt:                 # a creature: its library model in the pose of its clip the pass chose
                pz = (cam0.get("_poses") or {}).get(k)
                chars[k] = {"asset": crt[k]["asset"], "creature": True, "position": v["position"],
                            "facing": v["facing"], "facing_to": v.get("facing_to"),
                            "pose": pz if pz in crt[k]["poses"] else crt[k]["poses"][0]}
                continue
            # a wardrobe variant built as its own figure (its own stances made from it): staged from that figure
            vfig = own_variant(library, cv.get("base") or cv["id"], var)
            if var and ((cv.get("worn") or {}).get(var) or {}).get("head_on"):
                # the figure of this outfit wears its headwear already: no separate thing is set on its head
                for c_, _ in g:
                    not_on_head(c_, k)
            if v["pose"] == "sitting":
                c = {"seated_from": vfig or cv.get("base") or cv["id"], "seat": v.get("seat") or True,
                     "position": v["position"], "facing": v["facing"], "body_pose": json.loads(json.dumps(SEAT_POSE))}
                if var and not vfig:
                    c["variant"] = var
            elif v["pose"] == "lying" and v.get("bed"):
                c = {"lying_from": vfig or cv.get("base") or cv["id"], "bed": v["bed"], "position": v["position"],
                     "facing": v["facing"], "body_pose": {"pose": "lying", "params": {}}}   # (lying.py: the stance's)
            else:
                base = cv.get("id") or cv["base"]
                vid = f"{base}_{var}" if var and (library / "character" / f"{base}_{var}").exists() else base
                c = {"id": vid, "position": v["position"], "facing": v["facing"], "facing_to": v.get("facing_to"),
                     "body_pose": json.loads(json.dumps(STAND_POSE))}
                rev_ = {kk_: nn_ for nn_, kk_ in names.items()}
                nw_, how_ = near_of(rev_.get(k, ""), [c_["plan_id"] for c_, _ in g], sp, ids)
                jw_ = names.get(nw_ or "")
                if jw_ and how_ in NEAR_GAP_M and jw_ in st and st[jw_].get("position") and \
                        st[jw_].get("pose") != "sitting" and k not in crt and jw_ not in crt:
                    # (a hold or a joined hand takes two: only when the other is with this one too)
                    back_ = names.get(near_of(nw_, [c_["plan_id"] for c_, _ in g], sp, ids)[0] or "")
                    if back_ != k and how_ in ("dancing together", "touching"):
                        how_ = "close"
                    if k > jw_ or back_ != k:    # (the later of the two comes to the other; both then stay put)
                        v["position"] = near_position(v["position"], st[jw_]["position"], how_,
                                                      [x_["position"] for kk_, x_ in st.items()
                                                       if kk_ not in (k, jw_) and x_.get("position")])
                        c["position"] = v["position"]
                    for x_ in (v, st[jw_]):
                        x_.pop("marks", None)
                    man_ = lambda kk_: str((cast.get(rev_.get(kk_, "")) or {}).get("id") or "").split("_")[0] not in ("woman", "girl")  # noqa: E731
                    np_ = near_pose(how_, leads=(man_(k) and not man_(jw_)) or (man_(k) == man_(jw_) and k < jw_))
                    if np_:
                        c["body_pose"] = np_
                    c["near_to"], c["near_how"] = jw_, how_
                if v.get("marks"):
                    # the previous shot's mark first: it is kept whenever it still passes (no unmotivated move)
                    prev = (groups[gi - 2][-1][1].get(k) or {}).get("position") if gi > 1 else None
                    mk = [prev] if prev and any(math.dist(prev, m) < 0.05 for m in v["marks"]) else []
                    c["marks"] = mk + [m for m in v["marks"] if not mk or math.dist(m, mk[0]) >= 0.05]
            sc_ = stature.scale(look_txt.get(next((n for n, c_ in cast.items() if c_["key"] == k), "")))
            if sc_ < 1.0:                                    # a child: staged at their age's height (stature.py)
                c["scale"] = sc_
            lk = (cam0.get("_looks") or {}).get(k)
            if lk in st:
                c["look_at"] = lk
            pz = (cam0.get("_poses") or {}).get(k)
            bad = action_misfit(pz, k, v, st, w, things, (cam0.get("carried") or {}).get(k)) \
                if pz in ACTION_NAMES else None
            if bad:                                          # dropped by the scene's own facts, reported
                for c_, _ in g:
                    facts[c_["plan_id"]].setdefault("pass", {}).setdefault("dropped", []).append(f"{k} {pz}: {bad}")
                pz = None
            if (pz not in ACTION_NAMES or pz in BLOW_POSES) and fights(k, v, cam0, crt):
                # a fight: someone standing with a thing in the hand, squared up to a creature in an action shot,
                # holds it up against it (the pass names a pose only where the shot's line names the act: in six of
                # seven shots of a fight the man stood at ease, his sword hanging, 11-19). A blow the pass names
                # starts from the guard too: the clip plays the blow
                why_ = "a fight" if pz not in ACTION_NAMES else f"a fight (the pass's {pz} is the clip's blow)"
                pz = FIGHT_POSE
                for c_, _ in g:
                    facts[c_["plan_id"]].setdefault("pass", {}).setdefault("used", []).append(f"{k} {pz}: {why_}")
            if pz not in ACTION_NAMES and v.get("pose") == "standing" and cam0.get("shot_kind") == "action":
                # at work: a tool in the hand and a thing lying before them (the pass's "on") - the tool is up over
                # it (the pass names a pose only where the line names the act: a smith with a hammer and the iron on
                # the anvil stood at ease in four shots of the forging, 11-20)
                rows_ = (cam0.get("carried") or {}).get(k) or []
                if any(r_.get("at") == "hand" for r_ in rows_) and any(r_.get("at") == "on" for r_ in rows_):
                    pz = WORK_POSE
                    for c_, _ in g:
                        facts[c_["plan_id"]].setdefault("pass", {}).setdefault("used", []).append(f"{k} {pz}: at work")
            if pz in ACTION_NAMES:                           # what they begin to do as the shot starts
                c["_pose_choice"] = pz
                if pz in STAND_ACTIONS and v["pose"] != "sitting":
                    c["body_pose"]["pose"] = pz
                else:
                    c["body_pose"].setdefault("params", {})["action"] = pz
                if pz in READY_ACTIONS:                      # the guard points at the foe (scene.py: UE cm)
                    aim_ = foe_aim(v, st, crt)
                    if aim_:
                        c["body_pose"]["params"]["aim_at_m"] = aim_
                if pz in WORK_ACTIONS:
                    # the work places them: at the piece they work at, a step off its face and turned to it, their
                    # hands on its top (11-17: the smith stood a metre off the anvil, upright, hands in the air)
                    wt = work_thing(v, things)
                    oth = [st[kk]["position"] for kk in st if kk != k]
                    wp = work_place(wt, v["position"], oth, things, room) if wt else None
                    # the work takes its place first: a person standing by who has another mark steps to it (an
                    # onlooker's mark at the anvil's long side put the smith at its end, the work 77 cm off, and he
                    # folded 34 deg to reach it, 11-19)
                    pzs = cam0.get("_poses") or {}
                    stay = [kk for kk in st if kk != k and (pzs.get(kk) in WORK_ACTIONS or st[kk]["pose"] == "sitting"
                                                           or not st[kk].get("marks"))]
                    wp_free = work_place(wt, v["position"], [st[kk]["position"] for kk in stay], things, room) \
                        if wt and len(stay) < len(st) - 1 else None
                    if wp_free and (not wp or wp_free["fwd_cm"] < wp["fwd_cm"] - 1.0):
                        step = give_way(wp_free["position"], wt, {kk: st[kk] for kk in st if kk != k and kk not in stay})
                        if step is not None:
                            wp = wp_free
                            way.update(step)
                    if wp:
                        c.update(position=wp["position"], facing=wp["facing"], work_at=wt["name"])
                        c["body_pose"]["params"].update(work_top_cm=wp["top_cm"], work_fwd_cm=wp["fwd_cm"],
                                                        work_near_cm=wp["near_cm"])
                        c.pop("marks", None)
                        c["facing_to"] = None
                        # (and whoever stands by across the piece steps to its side: the work is seen from there)
                        side = stand_aside(wp["position"], wt, {
                            kk: dict(st[kk], position=way.get(kk, st[kk]["position"])) for kk in st
                            if kk != k and st[kk]["pose"] != "sitting" and pzs.get(kk) not in WORK_ACTIONS
                            and kk not in crt}, things, room, walled)
                        way.update(side)
                        turned.update({kk: wt["center"][:2] for kk in side})     # (they watch the work from there)
                if pz == "reach_to_forearm" and v["pose"] != "sitting":
                    # the reach places him: beside the person he reaches, at the forearm nearer him (the action's own
                    # place, not a camera move); the arm is solved on the measured bodies (speakers.py)
                    who = [kk for kk in (cam0.get("show") or []) if kk != k and kk in st] or \
                          sorted((kk for kk in st if kk != k), key=lambda kk: math.dist(st[kk]["position"], v["position"]))
                    if who:
                        pos, side = reach_place(st[who[0]], v["position"], things, room)
                        d = [st[who[0]]["position"][0] - pos[0], st[who[0]]["position"][1] - pos[1]]
                        n_ = math.hypot(*d) or 1.0
                        c.update(position=pos, facing=[round(d[0] / n_, 4), round(d[1] / n_, 4)],
                                 reach={"to": who[0], "her_side": side})
                        # he is there now: the next stagings find him beside her until the plan moves him (11-12:
                        # back at the desk in the next shot, and no camera could then hold her and him together)
                        reached[k] = (list(v["position"]), list(pos), list(c["facing"]), (v.get("at"), v.get("facing_thing")))
                        c.pop("marks", None)
                        cam0["show_joints"] = [[k, "hand_r"], [who[0], f"lowerarm_{side}"]]
            if var:
                c["worn"] = var                          # checked on the staged figure (speakers.staged_facts)
            wd = ((cv.get("worn") or {}).get(w) or {}) if w else {}
            for kk in WORN_KEYS:                         # the worn state's own words (the cast file's)
                if kk in wd:
                    c[kk] = wd[kk]
            chars[k] = c
        for kk, m in way.items():                        # those who stepped aside for the work, at their other mark
            if kk in chars and "work_at" not in chars[kk]:
                chars[kk]["position"] = list(m)
                if kk in turned and math.dist(m, turned[kk]) > 0.1:
                    n_ = math.dist(m, turned[kk])
                    chars[kk]["facing"] = [round((turned[kk][0] - m[0]) / n_, 4), round((turned[kk][1] - m[1]) / n_, 4)]
                if chars[kk].get("marks"):
                    chars[kk]["marks"] = [list(m)] + [q for q in chars[kk]["marks"] if math.dist(q, m) >= 0.05]
        for n_, (c_, s_) in enumerate(g):                # a fight shot's camera: at the fighter's scale (fight_camera)
            kp = next((k for k in chars if k in s_ and fights(k, s_[k], c_, crt)), None) or \
                next((k for k in chars if k in s_ and k not in crt and s_[k].get("foe") in chars
                      and s_[k].get("foe") in crt and s_[k].get("pose") == "standing"), None)
            if kp and not c_.get("insert"):
                kc = s_[kp]["face_target"] if s_[kp].get("face_target") in crt else s_[kp]["foe"]
                over = standoff_view(c_, kp, kc)
                fc = None if over is None else fight_camera(chars[kp]["position"], chars[kc]["position"], crt[kc],
                                                            over_shoulder=over, take=len(side_on))
                if fc:
                    if fc.pop("fight_view") != "shoulder":
                        side_on.append(c_.get("name"))
                    c_.update(fc, fight={"person": kp, "creature": kc})
        for c_, _ in g:                                  # a work shot's camera: on the work (work_camera)
            kw = next((k for k in chars if chars[k].get("work_at")), None)
            shown_ = c_.get("show") or c_.get("listed") or []
            if not kw or c_.get("insert") or c_.get("fight") or c_.get("shot_kind") != "action" or \
                    (shown_ and kw not in shown_):
                continue
            wt_ = next((t for t in things if t["name"] == chars[kw]["work_at"]), None)
            wc = wt_ and work_camera(chars[kw]["position"], wt_, c_.get("kind"),
                                     [chars[k]["position"] for k in chars if k != kw and chars[k].get("position")],
                                     things, room, walled, take=len(at_work))
            if wc:
                at_work.append(c_.get("name"))
                c_.update(wc, work={"person": kw, "thing": wt_["name"]})
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
