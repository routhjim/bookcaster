"""The unnamed people of a place (11-25: a ball of four people in an empty hall). A scene whose keyframes name a
"crowd" gets guests standing behind its cast: a few plain models built once per place (characters.py extras), each
used several times, placed in pairs on the free floor away from the cast's marks. They are part of the SET: no
motion, no checks, no questions asked of them; the renderers draw them (models()) from shot["crowd"].

    looks(where, period)            the guests to build for a place: [{"who", "key", "kind", "desc"}]
    count(text, area)               how many stand there
    place(poly, keep, n, seed)      where: [{"at", "_facing", "pose"}]
    stage(...)                      shot["crowd"] for every shot of a scene
    models(shot, cam_pos)           the posed model entries of a frame
"""
import hashlib
import json
import math
import os
import random
import re

VARIANTS = (("man", 30, "short dark hair, medium build"), ("woman", 25, "fair hair pinned up, slender"),
            ("man", 50, "grey hair, heavy build"), ("woman", 45, "brown hair pinned up, medium build"),
            ("man", 22, "light brown hair, slight build"), ("woman", 60, "grey hair pinned up, stout"))
HEIGHT = {"man": 1.76, "woman": 1.62}
MIN_N, MAX_N = 6, 24
AREA_EACH = 5.0          # m2 of free floor a guest, when the text gives no number
CLEAR = 2.0              # m from every mark of the cast
APART = 1.9              # m between two groups
PAIR = 0.85              # m between the two of a pair
WALL = 0.5               # m of free floor around a group
REACH = 8.0              # m around the middle of the cast's marks: where the cameras look
AHEAD = 0.5              # a guest nearer the camera than the nearest of the cast (+ this) is not drawn
WORDS = {"hundred": 100, "fifty": 50, "forty": 40, "thirty": 30, "twenty": 20, "dozen": 12, "ten": 10, "eight": 8,
         "six": 6}


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")[:50] or "x"


def looks(where: str, period: str = "") -> list:
    """The plain guests of a place: men and women of several ages, dressed for it as its time dressed."""
    when = f" in {period.strip().rstrip('.')}" if period.strip() else ""
    out = []
    for i, (kind, age, words) in enumerate(VARIANTS):
        out.append({"who": f"guest {i + 1}", "key": f"{slug(where)}__guest{i + 1}", "kind": kind,
                    "height": HEIGHT[kind],
                    "desc": f"a {kind} of about {age}, {words}, an ordinary plain face, wearing the clothes a {kind} "
                            f"of that age wears to {where}{when}, ordinary and unremarkable"})
    return out


def count(text: str, area: float) -> int:
    """How many guests: the number the crowd's words give, else one to AREA_EACH m2 of free floor."""
    low = str(text or "").lower()
    m = re.search(r"\b(\d{1,3})\b", low)
    n = int(m.group(1)) if m else next((v for k, v in WORDS.items() if re.search(rf"\b{k}\b", low)), None)
    if n is None:
        n = int(round(area / AREA_EACH))
    return max(MIN_N, min(MAX_N, n))


def area(poly) -> float:
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                   for i in range(len(poly)))) / 2.0


def _in_poly(poly, x, y) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i][:2]
        x2, y2 = poly[(i + 1) % n][:2]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            inside = not inside
    return inside


def seed_of(where: str) -> int:
    return int(hashlib.sha1(slug(where).encode()).hexdigest()[:8], 16)


def place(poly, keep, n: int, seed: int) -> list:
    """n guests in pairs on the free floor, within REACH of the middle of keep while there is room: each group WALL clear of the floor's edge, CLEAR from every point of
    keep (the cast's marks), APART from the other groups; the two of a pair face each other. The same seed gives
    the same places. Fewer than n when the floor has no more room. They STAND (11-25: the rig's one dancing pose, an
    arm in the air, made a hall of statues mid-leap; a still couple reads as a couple talking or about to dance).
    -> [{"at", "_facing", "pose"}]"""
    rnd = random.Random(seed)
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    free = lambda x, y: all(_in_poly(poly, x + dx * WALL, y + dy * WALL)       # noqa: E731
                            for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)))
    centres = []
    mid = [sum(k[0] for k in keep) / len(keep), sum(k[1] for k in keep) / len(keep)] if keep else None
    for k_ in range(9000):
        if len(centres) >= (n + 1) // 2:
            break
        x, y = rnd.uniform(min(xs), max(xs)), rnd.uniform(min(ys), max(ys))
        if mid and k_ < 6000 and math.hypot(x - mid[0], y - mid[1]) > REACH:
            continue                               # around the cast first; the rest of the floor if that is full
        if not free(x, y) or any(math.hypot(x - k[0], y - k[1]) < CLEAR for k in keep) \
                or any(math.hypot(x - c[0], y - c[1]) < APART for c in centres):
            continue
        centres.append((x, y))
    out = []
    for x, y in centres:
        a = rnd.uniform(0, 2 * math.pi)
        for j in range(min(2, n - len(out))):
            ang = a + j * math.pi
            out.append({"at": [round(x + math.cos(ang) * PAIR / 2, 3), round(y + math.sin(ang) * PAIR / 2, 3)],
                        "_facing": round(math.degrees(ang + math.pi) % 360.0, 1), "pose": "standing"})
    return out


def index_path(vd) -> str:
    return os.path.join(str(vd), "stage", "extras", "index.json")


def guests(vd, where: str) -> list:
    """The built guests of a place (those whose model file exists): [{"who", "model", "height", "kind"}]."""
    try:
        with open(index_path(vd)) as fh:
            ent = json.load(fh).get(slug(where)) or {}
    except (OSError, ValueError):
        return []
    return [p for p in ent.get("people", []) if p.get("model") and os.path.exists(p["model"])]


def crowd_text(frames: dict, moments) -> str:
    """The scene's unnamed people: the first of its moments' keyframes that names any."""
    return next((c for m in moments if (c := str((frames.get(m) or {}).get("crowd") or "").strip())), "")


def stage(where: str, moments, frames: dict, shots: dict, room, marks: dict, vd) -> list:
    """shot["crowd"] for every shot of a scene whose keyframes name a crowd, when the place's guests are built and
    its free floor is known: the same guests at the same spots in every shot. -> [log lines]"""
    text = crowd_text(frames, moments)
    if not text:
        return []
    people = guests(vd, where)
    poly = (room or {}).get("floor_polygon_xy")
    if not people or not poly:
        return [f"crowd at {where!r} ({text}): not staged ("
                + ("no guest models: `characters.py extras`" if not people else "the place has no measured floor") + ")"]
    keep = [list(mk["at"])[:2] for mk in (marks or {}).values() if mk.get("at")]
    keep += [list(f["at"])[:2] for s in shots.values() for f in s.get("figures", []) if f.get("at")]
    spots = place(poly, keep, count(text, area(poly)), seed_of(where))
    crowd = [{**sp, "who": f"{people[i % len(people)]['who']} #{i}", "model": people[i % len(people)]["model"],
              "height": people[i % len(people)].get("height", 1.7)} for i, sp in enumerate(spots)]
    for s in shots.values():
        s["crowd"] = [dict(g) for g in crowd]
    return [f"crowd at {where!r} ({text}): {len(crowd)} guest(s) of {len(people)} model(s) in {len(shots)} shot(s)"]


def visible(crowd, cam_pos, cast_xy) -> list:
    """The guests a camera draws: none nearer to it than the nearest of the cast (+ AHEAD), so nobody unnamed
    stands in front of the people the shot is of; with no cast in the shot, none within 2 m of the camera."""
    cx, cy = float(cam_pos[0]), float(cam_pos[1])
    near = min((math.hypot(p[0] - cx, p[1] - cy) for p in cast_xy), default=None)
    lim = near + AHEAD if near is not None else 2.0
    return [g for g in crowd or [] if math.hypot(g["at"][0] - cx, g["at"][1] - cy) >= lim]


_RIGS = {}


def models(shot: dict, cam_pos) -> list:
    """The frame's guests as posed model entries (blockout_scene.import_model): each on its own rig's bone lengths,
    in the set's layer."""
    crowd = shot.get("crowd") or []
    if not crowd:
        return []
    import numpy as np
    import blockout_rig as rig
    import blockout_contact as contact
    cast = [f["at"][:2] for f in shot.get("figures", []) if f.get("at")]
    out = []
    for g in visible(crowd, cam_pos, cast):
        rj = os.path.join(os.path.dirname(g["model"]), "rig.json")
        if not os.path.exists(g["model"]) or not os.path.exists(rj):
            continue
        if rj not in _RIGS:
            with open(rj) as fh:
                _RIGS[rj] = json.load(fh)["joints"]
        h = float(g.get("height", 1.7))
        Wj, _ = rig.place_human({"pose": g.get("pose", "standing"), "at": g["at"], "_facing": g["_facing"], "height": h})
        Wj = contact.fk_model(Wj, _RIGS[rj], h)
        out.append({"blend": g["model"], "who": g["who"], "height": h, "ground": False, "fist": [], "layer": "set",
                    "targets": {k: np.asarray(v).tolist() for k, v in Wj.items() if not k.startswith("_")}})
    return out
