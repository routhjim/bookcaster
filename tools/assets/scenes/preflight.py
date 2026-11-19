"""What is wrong with a place's scenes before anything is staged or drawn (stdlib; no GPU, no engine).

    python tools/assets/scenes/preflight.py SPECS_DIR SET_DIR [PASS.json] [--people NAME ...]

story_keys.py runs check() after the pass step and tells each finding in its log (nothing is stopped: these are
the faults that were found only on a finished sheet, after the engine and the image model had run, 11-19):

  words    a set's thing that would be named by its kind and not by what it is ("the work piece", "the landmark");
           a place on the open ground that would be named as a thing; a sentence of the set's words that says what
           the place lacks (a picture model draws what it is told of)
  work     a person placed to work at a piece with the work farther off than their hands reach standing
  people   two people of a scene on one spot; a person standing inside a standing thing
  things   a carried or worn thing that is one of the story's people; a worked thing that glows in the shots
           before and after a shot of the same scene run but not in it
  camera   a placed camera (a shot with nobody in it) against a standing thing, or with one across its line
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "redraw"))
sys.path.insert(0, str(HERE.parent / "sets"))

KIND_NAMES = ("work piece", "landmark", "group", "piece")     # a kind that holds any thing: no name to draw from
WORK_FAR_CM = 70.0         # the work this far forward of a standing person's hips is past their hands (pose.py folds
#                            the body to it: 77 cm gave 34 degrees)
SAME_SPOT_M = 0.3
CAMERA_CLEAR_M = 0.6


def _things(set_dir) -> list:
    import setfacts
    return setfacts.things(set_dir)


def words_why(set_dir, things: list) -> list:
    out = []
    try:
        import place_build as PB
        spec = json.loads((Path(set_dir) / "spec.json").read_text())
        w = str(spec.get("words") or "")
        gone = [x for x in re.split(r"(?<=[.!?])\s+", w) if x and PB.ABSENT.match(x)]
        if gone:
            out.append(f"the set's words say what is absent ({len(gone)} sentence(s)): \"{gone[0][:80]}\"")
    except (OSError, ValueError, ImportError):
        pass
    kinds = [t for t in things if t.get("kind") in ("fixture", "prop") and not t.get("called")
             and re.sub(r"[ _]\w?\d+$", "", str(t.get("name") or "").replace("_", " ")).strip() in KIND_NAMES]
    if kinds:
        out.append(f"{len(kinds)} piece(s) of the set have no name of their own and would be drawn as "
                   f"\"the {str(kinds[0]['name']).replace('_', ' ')}\": build the set again")
    return out


def _in_box(p, t, m=0.0) -> bool:
    c, h = t["center"], t["half"]
    return abs(p[0] - c[0]) < h[0] + m and abs(p[1] - c[1]) < h[1] + m


def spec_why(spec: dict, things: list, people: set) -> list:
    out = []
    gid = spec.get("id", "?")
    solid = [t for t in things if t.get("kind") == "fixture" and len(t.get("half") or []) > 2 and t["half"][2] * 2 > 0.5]
    ch = spec.get("characters") or {}
    for k, c in ch.items():
        pr = (c.get("body_pose") or {}).get("params") or {}
        if pr.get("work_fwd_cm") and float(pr["work_fwd_cm"]) > WORK_FAR_CM:
            out.append(f"{gid}: {k} works at {c.get('work_at') or 'a piece'} from {float(pr['work_fwd_cm']):.0f} cm "
                       f"off (past {WORK_FAR_CM:.0f} cm the body folds to reach)")
        if c.get("position") and not c.get("work_at") and not str((c.get("body_pose") or {}).get("pose") or "").startswith(("sit", "ly")):
            inside = next((t for t in solid if _in_box(c["position"], t, -0.05)), None)
            if inside:
                out.append(f"{gid}: {k} stands inside {str(inside['name']).replace('_', ' ')}")
    ks = [k for k in ch if ch[k].get("position")]
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            if math.dist(ch[a]["position"][:2], ch[b]["position"][:2]) < SAME_SPOT_M:
                out.append(f"{gid}: {a} and {b} stand on one spot")
    low = {re.sub(r"^(the|a|an) ", "", re.sub(r"[^a-z ]", "", n.lower()).strip()) for n in people}
    for cam in spec.get("cameras") or []:
        for k, rows in (cam.get("carried") or {}).items():
            for r in rows or []:
                nm = re.sub(r"^(the|a|an) ", "", re.sub(r"[^a-z ]", "", str(r.get("name") or "").lower()).strip())
                if nm and nm in low:
                    out.append(f"{gid}: {cam.get('plan_id')}: {k} carries \"{r.get('name')}\", one of the story's people, as a thing")
        if cam.get("insert") and cam.get("position") and cam.get("target"):
            p, c = cam["position"], cam["target"]
            on = cam.get("_on")
            rest = [t for t in solid if t.get("name") != on]
            near = next((t for t in rest if _in_box(p, t, CAMERA_CLEAR_M)), None)
            if near:
                out.append(f"{gid}: {cam.get('plan_id')}: the camera stands against {str(near['name']).replace('_', ' ')}")
            else:
                cut = next((t for t in rest for j in range(1, 8)
                            if _in_box([p[0] + (c[0] - p[0]) * j / 8, p[1] + (c[1] - p[1]) * j / 8], t)
                            and t["center"][2] + t["half"][2] > 1.2), None)
                if cut:
                    out.append(f"{gid}: {cam.get('plan_id')}: {str(cut['name']).replace('_', ' ')} stands across the camera's line")
    return out


def glow_why(things_by_shot: dict) -> list:
    """A thing that glows in the shot before and the shot after but not between (asked shot by shot)."""
    out = []
    ids = list(things_by_shot)
    state = []
    for i in ids:
        g = {}
        for rows in (things_by_shot[i] or {}).values():
            for t in rows or []:
                if t.get("on") in ("worked", "held"):
                    g[t.get("what")] = t.get("glow")
        state.append(g)
    for n in range(1, len(ids) - 1):
        for what, gl in state[n].items():
            if not gl and state[n - 1].get(what) and state[n + 1].get(what):
                out.append(f"{ids[n]}: \"{what}\" does not glow here but glows in the shots before and after")
    return out


def check(specs_dir, set_dir, pass_p=None, people=()) -> list:
    out = []
    try:
        things = _things(set_dir)
    except (OSError, ValueError, KeyError):
        things = []
        out.append("the set cannot be read: nothing checked against it")
    out += words_why(set_dir, things)
    for f in sorted(Path(specs_dir).glob("pl_g*.json")):
        try:
            out += spec_why(json.loads(f.read_text()), things, set(people))
        except (OSError, ValueError):
            out.append(f"{f.name} cannot be read")
    if pass_p and Path(pass_p).exists():
        try:
            out += glow_why(json.loads(Path(pass_p).read_text()).get("things") or {})
        except (OSError, ValueError):
            pass
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    ppl = a[a.index("--people") + 1:] if "--people" in a else []
    pos = a[:a.index("--people")] if "--people" in a else a
    for w in check(pos[0], pos[1], pos[2] if len(pos) > 2 else None, ppl):
        print(w)
