"""Hand-authored staging v plan-generated staging, shot by shot (stdlib).

    python tools/assets/scenes/plan_diff.py MAP.json OUT.json [SHEET.json [SET_DIR]]

MAP: [{"shot": "<plan shot id>", "label": "...", "hand": ["/abs/scene_out", "<camera>"],
       "plan": ["/abs/scene_out", "<camera>"] | null}, ...]   (scene outputs of speakers.py: result.json, the still)
Per shot: who is in the room (keys) and who is in the frame (their visible part), each person's place (the seat they
sit in, else their floor mark), their posture (sitting / standing), the speaker, the camera class (close / medium /
wide by the speaker's or subject's face size), and whether the speaker's face passes the rule; the differences;
counts: shots whose people and places match (same people, the same seat or marks within PLACE_TOL), shots whose
speaker's face passes. SHEET (optional): a sheet spec for redraw/sheet.py, hand still | plan still per row.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PLACE_TOL = 0.6          # metres between two floor marks still "the same place"


def load(sd: str, cam: str) -> dict | None:
    if not sd:
        return None
    r = json.loads((Path(sd) / "result.json").read_text())
    sh = next((s for s in r["shots"] if s["name"] == cam), None)
    if sh is None:
        return None
    spec = {}
    for p in sorted(Path(sd).glob("spec_round*.json")):
        spec = json.loads(p.read_text())
    people = {}
    for k, v in r["subjects"].items():
        m = v.get("measured") or {}
        c = (spec.get("characters") or {}).get(k) or {}
        pose = ((c.get("body_pose") or {}).get("pose") or "")
        people[k] = {"pos": [round(x, 2) for x in m.get("pelvis_xy", c.get("position", [0, 0]))],
                     "seat": c.get("seat") if any(w in pose for w in ("sit", "seat")) else None,
                     "posture": "sitting" if any(w in pose for w in ("sit", "seat")) else "standing",
                     "in_frame": bool(((sh.get("speaker_check") or {}).get("frame_facts") or {}).get(k, {}).get("any_in", True))}
    chk = sh.get("speaker_check") or {}
    face = None
    spk = sh.get("speaker")
    if spk:
        top = chk.get("face_h") or 0
        face = 0.25 if top >= 0.25 else 0.1 if top >= 0.08 else 0.0
    kind = sh.get("kind") or "wide"
    cls = {"close": "close", "medium": "medium"}.get(kind, "wide")
    if spk and chk.get("face_h"):
        cls = "close" if chk["face_h"] >= 0.25 else "medium" if chk["face_h"] >= 0.09 else "wide"
    return {"still": sh["path"], "people": people, "speaker": spk, "class": cls,
            "speaker_ok": (bool(chk.get("ok")) and not sh.get("reject")) if spk else None,
            "reject": sh.get("reject"), "framing_miss": sh.get("framing_miss")}


_THINGS = []


def anchor(pos) -> str:
    """The named set thing a floor mark is nearest (furniture, fireplace, window, door): "the same place" in the
    room's own terms, where two hand-placed marks for "near the desk" differ by most of a metre."""
    if not _THINGS:
        return ""
    t = min(_THINGS, key=lambda t: math.dist(t["center"][:2], pos))
    return t["name"]


def diff(h: dict, p: dict) -> list:
    out = []
    if set(h["people"]) != set(p["people"]):
        out.append(f"people: hand {sorted(h['people'])} plan {sorted(p['people'])}")
    for k in sorted(set(h["people"]) & set(p["people"])):
        a, b = h["people"][k], p["people"][k]
        if a["posture"] != b["posture"]:
            out.append(f"{k}: {a['posture']} v {b['posture']}")
        if a["seat"] or b["seat"]:
            if a["seat"] != b["seat"]:
                out.append(f"{k}: seat {a['seat']} v {b['seat']}")
        elif math.dist(a["pos"], b["pos"]) > PLACE_TOL:
            same = anchor(a["pos"]) == anchor(b["pos"])
            out.append(f"{k}: mark {a['pos']} v {b['pos']} ({math.dist(a['pos'], b['pos']):.1f} m"
                       + (f", both by the {anchor(a['pos'])})" if same else
                          f", by the {anchor(a['pos'])} v by the {anchor(b['pos'])})"))
    if h["speaker"] != p["speaker"]:
        out.append(f"speaker {h['speaker']} v {p['speaker']}")
    if h["class"] != p["class"]:
        out.append(f"camera {h['class']} v {p['class']}")
    return out


def main():
    rows = json.loads(Path(sys.argv[1]).read_text())
    res, tiles = [], []
    n_place = n_named = n_face = n_talk = n_face_hand = 0
    if len(sys.argv) > 4:                    # the set, for "the same named place"
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "redraw"))
        import setfacts
        _THINGS.extend(t for t in setfacts.things(sys.argv[4]) if t["kind"] in ("fixture", "window", "door"))
    for i, r in enumerate(rows, 1):
        h = load(*r["hand"])
        p = load(*r["plan"]) if r.get("plan") else None
        d = diff(h, p) if (h and p) else ["no plan staging"]
        place_ok = bool(p) and not [x for x in d if x.startswith(("people", )) or ": seat" in x or ": mark" in x
                                    or " v sitting" in x or " v standing" in x]
        named_ok = bool(p) and not [x for x in d if x.startswith(("people", )) or ": seat" in x
                                    or (": mark" in x and "both by" not in x) or " v sitting" in x or " v standing" in x]
        n_place += place_ok
        n_named += named_ok
        if p and p["speaker"]:
            n_talk += 1
            n_face += bool(p["speaker_ok"])
        if h and h["speaker"]:
            n_face_hand += bool(h["speaker_ok"])
        res.append({"shot": r["shot"], "label": r.get("label"), "hand": h, "plan": p, "diff": d,
                    "people_places_match": place_ok, "same_named_places": named_ok})
        tiles.append({"image": h["still"] if h else None, "label": f"{i} hand-authored | {r.get('label', '')}"})
        tiles.append({"image": p["still"] if p else None,
                      "label": f"{i} from the plan | " + ("match" if place_ok else "same named places" if named_ok
                                                          else "; ".join(d)[:150])
                      + ("" if not p or p["speaker_ok"] is None else f" | speaker face {'ok' if p['speaker_ok'] else 'FAIL'}")})
    summary = {"shots": len(rows), "people_places_match": n_place, "same_named_places": n_named, "talk_shots": n_talk,
               "speaker_face_ok_plan": n_face, "speaker_face_ok_hand": n_face_hand}
    Path(sys.argv[2]).write_text(json.dumps({"summary": summary, "shots": res}, indent=1))
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps({"out": json.loads(Path(sys.argv[3]).read_text()).get("out")
                                                 if Path(sys.argv[3]).exists() else "sheet.jpg",
                                                 "cols": 2, "tile_w": 640, "tiles": tiles}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
