"""Accepting redrawn keys: build the key checks from a redraw run, write one re-draw for each failure, pick (stdlib +
Pillow for reading masks).

    python tools/assets/redraw/accept.py checks RUN.json CHECK.json [--ids-from JOBS.jsonl] [--pose-model P]
    python tools/assets/redraw/accept.py retry  RUN.json CHECKED.json RETRY_JOBS.jsonl
    python tools/assets/redraw/accept.py pick   RUN.json CHECKED.json [RETRY_CHECKED.json] ACCEPTED.json

checks: one item per redraw in the run's meta.json (or only the ids of a jobs file): the redraw, its engine still and
        depth picture, and per person the scene's truth (box, angles, the joints in the frame and whether the engine
        sees them: guides.py "joints2d", else computed here from the scene's measured joints and the person mask)
        with the cast's clothes words and sheet. keycheck.py runs it at a card station.
retry:  for each key that fails a check (arms, hands, neck, outfit), the same job with another seed and the failed
        fact said once more (the cast's own words; nothing per shot), as "<id>__x".
pick:   per key, the first that passes every check (the draw, else its re-draw), else the one with fewer failures.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "scenes"))
import framing  # noqa: E402
import redraw as RD  # noqa: E402

JOINTS = ("head", "FACIAL_L_Eye", "FACIAL_R_Eye", "neck_01", "upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r",
          "hand_l", "hand_r", "thigh_l", "thigh_r")


def joints2d_from(res: dict, key: str, cam: dict, mask_path: Path) -> dict:
    from PIL import Image
    m = (res.get("subjects", {}).get(key) or {}).get("measured") or {}
    j = dict(m.get("joints") or {})
    if m.get("eye_xy"):
        j["chin"] = [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]]
        j["head_top"] = [m["eye_xy"][0], m["eye_xy"][1], m["head_top_z"]]
    mk = Image.open(mask_path).convert("L") if mask_path.exists() else None
    out = {}
    for n in JOINTS + ("chin", "head_top"):
        if n not in j:
            continue
        q = framing.project(cam, j[n], (1152, 640))
        if not q:
            continue
        seen = False
        if 0 <= q[0] < 1 and 0 <= q[1] < 1 and mk is not None:
            x, y = int(q[0] * mk.width), int(q[1] * mk.height)
            seen = mk.crop((max(0, x - 2), max(0, y - 2), x + 3, y + 3)).getextrema()[1] > 127
        out[n] = [round(q[0], 4), round(q[1], 4), seen]
    return out


def failures(v: dict) -> list:
    r = v.get("redraw", {})
    f = []
    if not r.get("arms_ok", True):
        f.append("arms")
    if not r.get("hands_ok", True):
        f.append("hands")
    if v.get("neck_ok") is False:
        f.append("neck")
    if v.get("outfit_ok") is False:
        f.append("outfit")
    return f


def item_fails(rec: dict) -> dict:
    f = {k: failures(v) for k, v in rec.get("people", {}).items() if failures(v)}
    for n, ok in (rec.get("things_ok") or {}).items():
        if ok is False:
            f.setdefault("set", []).append(f"colour {n}")
    return f


LAB_L, LAB_HUE, LAB_C = 15.0, 35.0, 18.0     # lightness, hue angle (deg, both coloured) and chroma tolerances


def colour_eval(*results: dict, ref_id: str | None = None) -> dict:
    """Each large piece's look over the run's keys: its colour (median Lab in its engine mask) against the run's
    reference for it (the anchor key's, else the median over the keys): lightness, hue and chroma within
    tolerance. Returns the references."""
    import statistics
    vals = {}
    for res in results:
        for iid, rec in res.items():
            for n, v in (rec.get("things") or {}).items():
                if v and v["px"] > 400:
                    vals.setdefault(n, []).append((iid, v))
    ref = {}
    for n, vs in vals.items():
        an = [v for iid, v in vs if ref_id and iid == ref_id]
        ref[n] = an[0] if an else {k: statistics.median(v[k] for _, v in vs) for k in ("L", "a", "b")}
    for res in results:
        for rec in res.values():
            ok = {}
            for n, v in (rec.get("things") or {}).items():
                if not v or v["px"] <= 400 or n not in ref:
                    continue
                r = ref[n]
                c1, c2 = math.hypot(v["a"], v["b"]), math.hypot(r["a"], r["b"])
                h = math.degrees(math.atan2(v["b"], v["a"]) - math.atan2(r["b"], r["a"]))
                dh = abs((h + 180) % 360 - 180) if min(c1, c2) > 8 else 0.0
                ok[n] = abs(v["L"] - r["L"]) <= LAB_L and dh <= LAB_HUE and abs(c1 - c2) <= LAB_C
            rec["things_ok"] = ok
    return ref


def cmd_checks(run: dict, out_p: Path, ids_from: str | None, pose_model: str):
    out = Path(run["out"])
    meta = json.loads((out / "meta.json").read_text())
    ids = None
    if ids_from:
        ids = [json.loads(ln)["id"] for ln in Path(ids_from).read_text().splitlines() if ln.strip()]
        jobs = {json.loads(ln)["id"]: json.loads(ln) for ln in Path(ids_from).read_text().splitlines() if ln.strip()}
    scenes = {Path(s).name: Path(s) for s in run["scenes"]}
    cast = run.get("cast") or {}
    items = []
    for iid in (ids if ids is not None else list(meta)):
        m = meta[iid[:-3] if iid.endswith("__x") else iid]
        sd = scenes[m["scene"]]
        truth = json.loads((sd / "truth.json").read_text())["shots"][m["cam"]]
        res = json.loads((sd / "result.json").read_text())
        ppl = []
        for p in m["people"]:
            tp = truth["people"].get(p["key"]) or {}
            j2 = tp.get("joints2d") or joints2d_from(res, p["key"], truth["camera"],
                                                      sd / "guides" / f"{m['cam']}_mask_{p['key']}.png")
            c = cast.get(p["key"]) or {}
            ppl.append(dict(p, joints2d=j2, wear=c.get("wear"), sheet=c.get("sheet"), noun=c.get("noun")))
        path = jobs[iid]["path"] if ids is not None else str(out / f"{iid}.png")
        things = [{"name": t["name"], "mask": t["mask"], "words": t["words"], "place": t.get("place")}
                  for t in truth.get("things") or [] if t.get("mask")]
        items.append({"id": iid, "redraw": path, "engine": m["still"],
                      "depth": str(sd / "guides" / f"{m['cam']}_depth.png"), "people": ppl, "things": things})
    out_p.write_text(json.dumps({"out": str(out_p.with_suffix("")) + "_result.json", "pose_model": pose_model,
                                 "judge": True, "items": items}, indent=1))
    print(out_p, len(items))


FIX = {"arms": "{Who}'s arms hang in their sleeves clear of the body, both sleeves and hands drawn.",
       "hands": "{Who}'s hands are drawn where the engine frame shows them.",
       "neck": "{Who}'s neck is of natural length, the collar standing close under the jaw.",
       "outfit": ""}


def cmd_retry(run: dict, checked: dict, out_jobs: Path):
    out = Path(run["out"])
    neck_eval(checked)
    colour_eval(checked, ref_id=(run.get("anchor") or {}).get("id"))
    jobs = {json.loads(ln)["id"]: json.loads(ln) for ln in (out / "image_jobs.jsonl").read_text().splitlines() if ln.strip()}
    meta = json.loads((out / "meta.json").read_text())
    cast = run.get("cast") or {}
    lines = []
    for iid, rec in checked.items():
        f = item_fails(rec)
        if not f or iid not in jobs:
            continue
        j = dict(jobs[iid])
        ppl = [(p["key"], p["box"], p["head_to_camera_deg"], p["posture"], p["body_to_camera_deg"], p["nearer_shoulder"])
               for p in meta[iid]["people"]]
        nm = RD.names(ppl, cast)
        add = []
        tw = {t["name"]: t for t in json.loads((Path([s for s in run["scenes"] if Path(s).name == meta[iid]["scene"]][0])
                                                 / "truth.json").read_text())["shots"][meta[iid]["cam"]].get("things") or []}
        for x in f.pop("set", []):
            t = tw.get(x.split(" ", 1)[1])
            if t:
                add.append(f"{t['words'].split(' (')[0].capitalize()} {('(' + t['words'].split(' (', 1)[1]) if ' (' in t['words'] else ''}"
                           f"{', ' + t['place'] if t.get('place') else ''} keeps exactly the colour and shape of its "
                           f"reference picture.")
        for k, fs in f.items():
            who = nm.get(k, "the person")
            w = RD.who_of(cast, k)
            for x in fs:
                if x == "outfit":
                    add.append(RD.neck_words(w))
                else:
                    add.append(FIX[x].format(Who=who[0].upper() + who[1:]))
        j["prompt"] = j["prompt"] + " " + " ".join(a for a in add if a)
        j["seed"] = int(j["seed"]) + 1000
        j["id"] = iid + "__x"
        j["path"] = str(Path(j["path"]).with_name(j["id"] + ".png"))
        j["_fails"] = f
        lines.append(json.dumps(j))
    out_jobs.write_text("".join(x + "\n" for x in lines))
    print(out_jobs, len(lines))


NECK_TOL = 1.2


def neck_eval(*results: dict) -> None:
    """Neck verdicts over all keys of the run together (first draws and re-draws): longer than NECK_TOL times the
    character's median fails, and than NECK_TOL times the engine's own reading when that is the longer reference
    (the staging figures' necks differ from people's: the man's engine neck reads 1.3-1.6, the woman's 0.7-0.8,
    against 0.9 drawn; judged on the engine alone, every key of her failed and his long-neck close passed)."""
    import statistics
    by = {}
    for res in results:
        for rec in res.values():
            for k, v in rec.get("people", {}).items():
                n = v.get("redraw", {}).get("neck_ratio")
                if n:
                    by.setdefault(k, []).append(n)
    med = {k: statistics.median(v) for k, v in by.items()}
    for res in results:
        for rec in res.values():
            for k, v in rec.get("people", {}).items():
                n = v.get("redraw", {}).get("neck_ratio")
                e = v.get("engine", {}).get("neck_ratio")
                if not n:
                    v["neck_ok"] = None
                    continue
                v["neck_ok"] = n <= NECK_TOL * med[k] and n <= NECK_TOL * max(e or 0.0, med[k])
                v["neck_refs"] = {"engine": e, "character_median": round(med[k], 3)}


def cmd_pick(run: dict, checked: dict, retry: dict, out_p: Path):
    out = Path(run["out"])
    neck_eval(checked, retry)
    colour_eval(checked, retry, ref_id=(run.get("anchor") or {}).get("id"))
    acc = {}
    for iid, rec in checked.items():
        f0 = item_fails(rec)
        x = retry.get(iid + "__x")
        choice = (iid, f0)
        if f0 and x is not None:
            f1 = item_fails(x)
            n0 = sum(len(v) for v in f0.values())
            n1 = sum(len(v) for v in f1.values())
            if n1 < n0:
                choice = (iid + "__x", f1)
        acc[iid] = {"path": str(out / f"{choice[0]}.png"), "id": choice[0], "fails": choice[1],
                    "first_fails": f0, "redrawn": iid + "__x" in retry}
    out_p.write_text(json.dumps(acc, indent=1))
    print(out_p, sum(1 for v in acc.values() if not v["fails"]), "of", len(acc), "pass")


def main():
    a = sys.argv[1:]
    run = json.loads(Path(a[1]).read_text())
    if a[0] == "checks":
        ids = a[a.index("--ids-from") + 1] if "--ids-from" in a else None
        pm = a[a.index("--pose-model") + 1] if "--pose-model" in a else str(Path(run["out"]).parent / "models" / "rtmw_x_256x192.pt")
        cmd_checks(run, Path(a[2]), ids, pm)
    elif a[0] == "retry":
        cmd_retry(run, json.loads(Path(a[2]).read_text()), Path(a[3]))
    elif a[0] == "pick":
        retry = json.loads(Path(a[3]).read_text()) if len(a) > 4 and Path(a[3]).exists() else {}
        cmd_pick(run, json.loads(Path(a[2]).read_text()), retry, Path(a[-1]))


if __name__ == "__main__":
    main()
