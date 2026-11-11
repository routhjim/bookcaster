"""Counts from measure.py's metrics (stdlib): per method, per grid cell and in all.

    python tools/assets/redraw/tally.py METRICS.json [METRICS.json ...] [--json OUT]

Per person in a redraw: place held (person IoU >= 0.5 with the engine's visible box), distance held (box height
0.8-1.25 of the engine's), facing held (the judge's front/side/back = the scene's angle class), and for a person
the scene turns away (>= 115 deg): no face found in their box. Per picture: room same (the judge, both pictures),
room things held (window/fireplace/armchair boxes that the engine still has), lit sources held (each lit lamp or fire
whose source is in frame judged lit), light side held, look = photo; identity: mean face similarity to the sheet of
people the scene has facing the camera (< 60 deg). The judge's own error is shown by the same checks on the engine
stills (facing on the engine's own picture).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def cls(deg):
    return "front" if deg < 60 else "side" if deg < 115 else "back"


def tally(m: dict) -> dict:
    T = defaultdict(lambda: defaultdict(int))
    sims = defaultdict(list)
    fails = defaultdict(list)
    for iid, rec in m.items():
        r, e = rec["redraw"], rec.get("engine") or {}
        parts = iid.split("__")
        cell_dist = parts[1]
        for k, q in r["people"].items():
            c = cls(q["truth_deg"])
            for cell in (f"{cell_dist}/{c}", f"all/{c}", "all/all"):
                t = T[cell]
                t["people"] += 1
                t["place"] += int(q.get("iou", 0) >= 0.5)
                t["distance"] += int(0.8 <= q.get("h_ratio", 0) <= 1.25 and 0.75 <= q.get("w_ratio", 1.0) <= 1.33)
                t["facing"] += int(q.get("judge_facing") == c)
                eq = (e.get("people") or {}).get(k, {})
                t["facing_judge_on_engine"] += int(eq.get("judge_facing") == c)
                if c == "back":
                    t["no_face"] += int(q.get("face_boxes", 0) == 0)
                    t["no_face_engine"] += int(eq.get("face_boxes", 0) == 0)
                if q.get("light_side") is not None and eq.get("light_side") is not None:
                    t["light_n"] += 1
                    t["light_held"] += int(q["light_side"] == eq["light_side"])
            if c == "back" and (q.get("face_boxes", 0) or q.get("judge_facing") != "back"):
                fails["away_turned"].append(f"{iid}/{k} judge={q.get('judge_facing')} faces={q.get('face_boxes')}")
            if q.get("iou", 0) < 0.5:
                fails["moved"].append(f"{iid}/{k} iou={q.get('iou')}")
            if c == "front" and q.get("face_sim") is not None:
                sims[iid.split("__")[0] + "__" + parts[1]].append(q["face_sim"])
                sims["all"].append(q["face_sim"])
        t = T[f"{cell_dist}/pictures"]
        for cell in (f"{cell_dist}/pictures", "all/pictures"):
            t = T[cell]
            t["pictures"] += 1
            t["room_same"] += int(str(r.get("room_judge", "")).strip().lower().startswith("same"))
            t["frame_same"] += int(str(r.get("frame_judge", "")).strip().lower().startswith("yes"))
            rh = rec.get("room_held") or {}
            t["room_things"] += len(rh)
            t["room_things_held"] += sum(bool(v) for v in rh.values())
            t["photo"] += int(r.get("look") == "photo")
            t["photo_engine"] += int(e.get("look") == "photo")
            for name, v in (r.get("lit") or {}).items():
                t["lit_n"] += 1
                t["lit_held"] += int(v == "yes")
                t["lit_engine"] += int((e.get("lit") or {}).get(name) == "yes")
        if not str(r.get("room_judge", "")).lower().startswith("same"):
            fails["room"].append(f"{iid}: {r.get('room_judge')}")
        for name, v in (r.get("lit") or {}).items():
            if v != "yes":
                fails["unlit"].append(f"{iid}: {name}")
    out = {c: dict(v) for c, v in sorted(T.items())}
    out["identity_face_sim_mean"] = round(sum(sims["all"]) / len(sims["all"]), 3) if sims["all"] else None
    out["identity_n"] = len(sims["all"])
    out["failures"] = {k: v for k, v in fails.items()}
    return out


def by_strength(m: dict) -> dict:
    g = defaultdict(dict)
    for iid, rec in m.items():
        s = iid.rsplit("__", 1)[1] if iid.count("__") >= 2 else "s"
        g[s][iid] = rec
    return g


def main():
    args = [a for a in sys.argv[1:] if a != "--json"]
    jo = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if jo:
        args.remove(jo)
    res = {}
    for a in args:
        m = json.loads(Path(a).read_text())
        for s, sub in by_strength(m).items():
            res[f"{Path(a).parent.name}/{s}"] = tally(sub)
    for k, v in res.items():
        a = v.get("all/all", {})
        b = v.get("all/back", {})
        p = v.get("all/pictures", {})
        print(f"{k}: people {a.get('people')} place {a.get('place')} distance {a.get('distance')} facing {a.get('facing')} "
              f"(judge on engine {a.get('facing_judge_on_engine')}) | away {b.get('people')}: judged back {b.get('facing')}, "
              f"no face {b.get('no_face')} (engine {b.get('no_face_engine')}) | pictures {p.get('pictures')}: room same "
              f"{p.get('room_same')}, frame same {p.get('frame_same')}, room things {p.get('room_things_held')}/{p.get('room_things')}, "
              f"lit {p.get('lit_held')}/{p.get('lit_n')} (engine {p.get('lit_engine')}), light side "
              f"{a.get('light_held')}/{a.get('light_n')}, photo {p.get('photo')} (engine {p.get('photo_engine')}) | "
              f"face sim {v['identity_face_sim_mean']} n {v['identity_n']}")
    if jo:
        Path(jo).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
