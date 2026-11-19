"""The people in a key, from its staged scene (stdlib): who is visible, by the plan's own look phrase, how many, and
for a move shot who moves where. Written beside the key for the video flow, never by hand.

    python tools/assets/scenes/shot_people.py SCENE_DIR CAMERA LOOK_JSON CAST_JSON OUT_STEM

SCENE_DIR: a staged scene (speakers.py: truth.json, staged.json, spec_round*.json). CAMERA: its camera name.
LOOK_JSON: the book's look file ({"characters": {name: look text}}). CAST_JSON: {name: {"key": scene key}}.
OUT_STEM: <keys folder>/<shot id>; writes OUT_STEM.shot_facts.json and OUT_STEM.shot_facts.txt.

Visible: a person whose own pixels in the engine still are at least VISIBLE_AREA of the frame, seen from the front,
the side or the back (a back is a person in the picture; a sliver at the edge or a hand behind someone is not).
The look phrase is the one the director's lines use: "the <man|woman> in <the first thing worn>", from the look
text ("a man of about 40, ..., wearing a dark Victorian frock coat, waistcoat, ..." -> "the man in the dark
Victorian frock coat").
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

VISIBLE_AREA = 0.01         # people in the keys read 1.3 % and up; slivers behind the edge 0.3-0.4 %


def look_phrase(look: str) -> str | None:
    m = re.search(r"\b(man|woman|boy|girl)\b", look)
    w = re.search(r"wearing (?:an? |the )?(.+?)(?:,|;|\.| and |$)", look)
    if not (m and w):
        return None
    return f"the {m.group(1)} in the {w.group(1).strip()}"


def creature_phrase(look: str) -> str | None:
    """A creature's phrase as the director's lines name it: "the" and the last word of its look's head phrase (the
    words before "with", "of", "that", "who" or the first comma)."""
    head = re.split(r"\s+(?:with|of|that|who)\s+|,", re.sub(r"^\s*an?\s+", "", (look or "").strip()))[0].split()
    return f"the {head[-1].lower()}" if head else None


def phrase(look: str, entry: dict | None) -> str | None:
    """The look phrase of a cast entry: a person's, or a creature's (cast "creature": true)."""
    return creature_phrase(look) if (entry or {}).get("creature") else look_phrase(look)


def heads_out(scene: Path, cam: str) -> set:
    """The people of this still whose head (head top or chin, the scene's frame check) is outside the frame."""
    rp = scene / "result.json"
    out = set()
    if not rp.exists():
        return out
    for sh in json.loads(rp.read_text()).get("shots", []):
        if sh.get("name") != cam:
            continue
        checks = dict(sh.get("others_check") or {})
        subj = [k for k in (sh.get("truth") or {}) if k not in checks]
        if subj and sh.get("frame_check"):
            checks[subj[0]] = sh["frame_check"]
        for k, fc in checks.items():
            pts = [fc.get("head_top"), fc.get("chin")]
            if all(pts) and not all(0.0 <= q[0] <= 1.0 and 0.0 <= q[1] <= 1.0 for q in pts):
                out.add(k)
    return out


def people(scene: Path, cam: str, looks: dict, cast: dict) -> dict:
    truth = json.loads((scene / "truth.json").read_text())["shots"][cam]
    key_name = {v["key"]: nm for nm, v in cast.items()}
    vis = []
    for k, v in (truth.get("people") or {}).items():
        if (v.get("area") or 0.0) < VISIBLE_AREA:
            continue
        nm = key_name.get(k, k)
        deg = v.get("head_to_camera_deg") or 0.0
        vis.append({"key": k, "name": nm, "look": phrase(looks.get(nm, ""), cast.get(nm)) or nm,
                    "area": round(v["area"], 4), "seen": "front" if deg < 60 else "side" if deg < 115 else "back",
                    "box": v.get("box")})
    # a person whose head is out of the frame (a shoulder or a back at the picture's edge) is not one of the
    # picture's people: named to the video model as a person in the picture, they walked in whole (a listener's
    # back at the left edge became a man standing in the room, 11-13). The redraw leaves them unnamed the same way.
    out_h = heads_out(scene, cam)
    kept = [p for p in vis if p["key"] not in out_h]
    if kept:
        vis = kept
    vis.sort(key=lambda p: ((p["box"] or [0, 0, 0, 0])[0] + (p["box"] or [0, 0, 0, 0])[2]) / 2)
    out = {"camera": cam, "visible": [dict({x: p[x] for x in ("name", "look", "seen", "area")}, **box_of(p["box"]))
                                      for p in vis],
           "count": len(vis), "move": None}
    st = scene / "staged.json"
    sf = (json.loads(st.read_text()).get("shot_facts") or {}).get(cam) if st.exists() else None
    if sf:
        m = sf["moves"]
        out["move"] = {"who": look_phrase(looks.get(key_name.get(m["who"], ""), "")) or key_name.get(m["who"]),
                       "to": m["to_words"], "ends": m["posture_end"],
                       "fixed": {look_phrase(looks.get(key_name.get(k, ""), "")) or k: p for k, p in sf["fixed"].items()}}
    return out


def box_of(b) -> dict:
    """The person's visible body in the key as [left, top, right, bottom], 0-1 of the picture (the engine mask's
    bounds, head included), or nothing when the scene has no box for them."""
    if not b or len(b) != 4:
        return {}
    l, t, r, bt = (max(0.0, min(1.0, float(v))) for v in b)
    return {"box": [round(l, 4), round(t, 4), round(r, 4), round(bt, 4)]} if l < r and t < bt else {}


def lines(f: dict) -> str:
    def side(i, n):
        return "" if n == 1 else (" on the left" if i == 0 else " on the right" if i == n - 1 else " in the middle")
    n = f["count"]
    ppl = [p["look"] + side(i, n) + (" (seen from behind)" if p["seen"] == "back" else "") for i, p in enumerate(f["visible"])]
    out = [f"- In the picture: {n} {'person' if n == 1 else 'people'}" + (": " + "; ".join(ppl) if ppl else "") + "."]
    if f.get("move"):
        m = f["move"]
        out.append(f"- In this shot {m['who']} walks to {m['to']} and ends "
                   f"{'sitting in it' if m['ends'] == 'sitting' else 'standing by it'}.")
        out.append("- Everyone else keeps their place and posture for the whole shot: "
                   + ", ".join(f"{k} {p}" for k, p in m["fixed"].items()) + ".")
    return "\n".join(out) + "\n"


def main():
    scene, cam, look_p, cast_p, stem = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    looks = json.loads(Path(look_p).read_text()).get("characters") or {}
    f = people(scene, cam, looks, json.loads(Path(cast_p).read_text()))
    Path(stem + ".shot_facts.json").write_text(json.dumps(f, indent=1))
    Path(stem + ".shot_facts.txt").write_text(lines(f))
    print(stem, f["count"], [p["look"] for p in f["visible"]])


if __name__ == "__main__":
    main()
