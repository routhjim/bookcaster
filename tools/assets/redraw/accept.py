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
    if v.get("face_ok") is False:
        f.append("face")
    pf = v.get("person_found") or {}
    if pf.get("engine", 0) >= PERSON_FOUND and pf.get("redraw", 1.0) < PERSON_FOUND:
        f.append("missing")              # in the engine still, not in the redraw
    if v.get("action_ok") is False:
        f.append("action")
    return f


def item_fails(rec: dict) -> dict:
    f = {k: failures(v) for k, v in rec.get("people", {}).items() if failures(v)}
    for n, ok in (rec.get("things_ok") or {}).items():
        if ok is False:
            f.setdefault("set", []).append(f"colour {n}")
    for n, ok in (rec.get("lit_ok") or {}).items():
        if ok is False:
            f.setdefault("set", []).append(f"lit {n}")
    for n, got in (rec.get("look") or {}).items():
        if got.get("verdict", got.get("ok")) is False:
            f.setdefault("set", []).append(f"look {n}")
    for k, ok in (rec.get("wear_ok") or {}).items():
        if ok is False:
            f.setdefault(k, []).append("clothes colour")
    for k, ok in (rec.get("hair_ok") or {}).items():
        if ok is False:
            f.setdefault(k, []).append("hair")
    return f


# hue angle (deg), judged when both are coloured (chroma > LAB_MIN_C). Lightness and chroma are not judged: they
# follow each shot's light (the same gold chair reads L 4, chroma 10 in the window's shade and L 20, chroma 21 by
# the fire); the gold chair reads hue 65-71 in every shot that keeps it gold, drawn as red leather 45-55: 10 deg
LAB_HUE, LAB_MIN_C = 10.0, 10.0
LAB_C_LO, LAB_C_HI = 0.55, 1.8   # chroma ratio (redraw over engine) within these factors of the piece's median


LIT_L, LIT_B = 0.55, 0.45    # a lit source in the redraw: at least this share of the engine's lightness and warmth


def lit_eval(*results: dict) -> None:
    """Every lit source the engine shows (even a sliver) is bright and warm in the redraw inside its own mask,
    relative to the engine still: a fire drawn dark in one key and burning in the next (11-11)."""
    for res in results:
        for rec in res.values():
            ok = {}
            for n, v in (rec.get("lit") or {}).items():
                r, e = v.get("redraw"), v.get("engine")
                # judged on the brightest tenth of the mask (a fire drawn as flames in a dark grate is lit; the
                # engine's grate glows as a whole, so its median overstated what a lit redraw must reach)
                rl, el = r.get("L90", r["L"]) if r else 0, e.get("L90", e["L"]) if e else 0
                rb, eb = r.get("b90", r["b"]) if r else 0, e.get("b90", e["b"]) if e else 0
                if r and e and el > 25:
                    # warmth from either the median or the brightest tenth (a flame's white core has little yellow)
                    warm = max(r["b"] / max(e["b"], 1.0), rb / max(eb, 1.0))
                    ok[n] = rl >= LIT_L * el and warm >= LIT_B
            rec["lit_ok"] = ok


LOOK_FILL = 0.45         # a piece judged for its look when its own pixels fill this much of its crop (a bureau
                         # half behind the leather armchair was "named" as the armchair)


def look_eval(run: dict, *results: dict) -> None:
    """The look answers judged: an answer naming a piece with the same words (two identical bookcases) is right;
    a piece whose crop is mostly something else in front of it is not judged."""
    opts = run.get("_look_opts") or []
    if not opts and run.get("set"):
        import setfacts
        opts = [{"name": t["name"], "words": t["words"]} for t in setfacts.things(run["set"])
                if t.get("large") and t["kind"] != "window" and t["name"] != "rug"]
    for res in results:
        for rec in res.values():
            for n, v in (rec.get("look") or {}).items():
                if v.get("fill") is not None and v["fill"] < LOOK_FILL:
                    v["verdict"] = None
                    continue
                g = v.get("got") or int("".join(ch for ch in str(v.get("answer", "")) if ch.isdigit()) or 0)
                same = 0 < g <= len(opts) and opts[g - 1]["words"] == opts[v["want"] - 1]["words"]
                v["verdict"] = g == v["want"] or same


PERSON_FOUND = 0.15      # a "person" box at least this share of the person's own engine box
FACE_MIN = 0.45          # a face the pose model finds: nose and an eye above this score, the contour's median too
YAW_TURN = 0.45          # the nose this far (in eye spacings) off the eyes' middle: the head clearly turned


YAW_PROFILE = 1.1        # the nose past this many eye spacings off the eyes' middle: a profile or the head's back
                         # (the pose model scores face points high even on the back of a head: only the turn tells)


def face_seen(q: dict) -> bool:
    fs = q.get("face_scores") or [0, 0, 0]
    return (fs[0] >= FACE_MIN and max(fs[1], fs[2]) >= FACE_MIN and (q.get("face_contour") or 0) >= FACE_MIN
            and abs(q.get("yaw") or 0.0) <= YAW_PROFILE)


def face_eval(*results: dict) -> None:
    """The speaker readable in the redraw where the engine still shows their face: the face found by the pose model
    in the redraw, turned the same way as in the engine."""
    for res in results:
        for rec in res.values():
            for k, v in rec.get("people", {}).items():
                if not v.get("named"):
                    continue
                r, e = v.get("redraw") or {}, v.get("engine") or {}
                if not e or not face_seen(e):
                    continue
                ok = face_seen(r)
                ey, ry = e.get("yaw"), r.get("yaw")
                if ok and ey is not None and ry is not None and abs(ey) > YAW_TURN and abs(ry) > YAW_TURN and ey * ry < 0:
                    ok = False               # turned the other way
                v["face_ok"] = ok


ACTION_SEEN = ("veil_lift", "look_at_sleeve", "reach_to_forearm")
HEAD_BOW = 0.19          # nose below the eyes over the face height: a head bowed to look down


def action_there(act: str, q: dict) -> bool | None:
    """The staged start action in one picture's pose estimate (None: not measurable): hands near the chin for the
    veil lift (both, within 1.6 face heights), a forearm raised for the sleeve look (a wrist above its elbow, or a
    hand within 2.2 face heights of the chin), a hand reaching out for the reach (a wrist out past its elbow)."""
    t = q.get("pts") or {}
    eye = [x for x in (t.get("l_eye"), t.get("r_eye")) if x]
    if not (t.get("chin") and eye):
        return None
    fh = max(1e-3, t["chin"][1] - sum(x[1] for x in eye) / len(eye)) * 2.2      # eyes-to-chin ~ half the face
    hands = [t.get("l_hand") or t.get("l_wr"), t.get("r_hand") or t.get("r_wr")]
    d = [((h[0] - t["chin"][0]) ** 2 + (h[1] - t["chin"][1]) ** 2) ** 0.5 / fh if h else None for h in hands]
    if act == "veil_lift":
        return None if None in d else max(d) <= 1.6
    if act == "look_at_sleeve":              # the head bowed onto the sleeve: the nose low between eyes and chin
        if not t.get("nose"):                # (the solver lifts the forearm only a little; the bow is what reads:
            return None                      # level heads read 0.15-0.17, the engine's bowed one 0.22, 11-12)
        ey = sum(x[1] for x in eye) / len(eye)
        return (t["nose"][1] - ey) / fh >= HEAD_BOW
    if act == "reach_to_forearm":
        return any(t.get(f"{s}_wr") and t.get(f"{s}_el") and t.get(f"{s}_sh") and
                   abs(t[f"{s}_wr"][0] - t[f"{s}_sh"][0]) > abs(t[f"{s}_el"][0] - t[f"{s}_sh"][0]) for s in "lr")
    return None


def action_eval(*results: dict) -> None:
    """A start action the engine still shows must also be in the redraw (the same measures on its pose estimate)."""
    for res in results:
        for rec in res.values():
            for k, v in rec.get("people", {}).items():
                act = v.get("action")
                if not act:
                    continue
                e, r = action_there(act, v.get("engine") or {}), action_there(act, v.get("redraw") or {})
                if e:
                    v["action_ok"] = bool(r)


HAIR_L = 12.0            # a person's hair: lightness above the engine's in the same region, v their own median
HAIR_PX = 3000           # judged on a hair region this big (a small head's region takes in the wall behind it)


def hair_eval(*results: dict) -> dict:
    """Each person's hair reads the same in every key: the redraw's lightness minus the engine's in the hair region
    against the median of that person over the run's keys (grey in one key, dark in the next, 11-12)."""
    import statistics
    vals = {}
    for res in results:
        for rec in res.values():
            for k, v in (rec.get("hair") or {}).items():
                if v.get("redraw") and v.get("engine") and v["redraw"]["px"] >= HAIR_PX:
                    vals.setdefault(k, []).append(v["redraw"]["L"] - v["engine"]["L"])
    med = {k: statistics.median(v) for k, v in vals.items() if len(v) >= 3}
    for res in results:
        for rec in res.values():
            ok = {}
            for k, v in (rec.get("hair") or {}).items():
                if v.get("redraw") and v.get("engine") and v["redraw"]["px"] >= HAIR_PX and k in med:
                    ok[k] = abs((v["redraw"]["L"] - v["engine"]["L"]) - med[k]) <= HAIR_L
            rec["hair_ok"] = ok
    return med


WEAR_L = 12.0            # a person's clothes: lightness above the engine's in the same mask, v their usual


def wear_eval(*results: dict) -> dict:
    """Each person's clothes held from key to key: the redraw's lightness minus the engine's in the same mask (the
    shot's light taken out) against the median of that person over the run's keys; a coat drawn grey in one key and
    black in the next differs by 20+ (11-11)."""
    import statistics
    vals = {}
    for res in results:
        for rec in res.values():
            for k, v in (rec.get("wear") or {}).items():
                if v.get("redraw") and v.get("engine") and v["redraw"]["px"] > 400:
                    vals.setdefault(k, []).append(v["redraw"]["L"] - v["engine"]["L"])
    med = {k: statistics.median(v) for k, v in vals.items()}
    for res in results:
        for rec in res.values():
            ok = {}
            for k, v in (rec.get("wear") or {}).items():
                if v.get("redraw") and v.get("engine") and v["redraw"]["px"] > 400 and k in med:
                    ok[k] = abs((v["redraw"]["L"] - v["engine"]["L"]) - med[k]) <= WEAR_L
            rec["wear_ok"] = ok
    return med


def colour_eval(*results: dict, ref_id: str | None = None) -> dict:
    import statistics
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
    # chroma: the redraw's over the engine still's in the same mask, against the piece's median ratio over the keys
    # (the shot's light cancels): a gold chair drawn grey-brown has no hue left to compare (chroma 7 against 23)
    ratios = {}
    for res in results:
        for rec in res.values():
            for n, v in (rec.get("things") or {}).items():
                e = (rec.get("things_engine") or {}).get(n)
                if v and e and v["px"] > 400 and math.hypot(e["a"], e["b"]) > LAB_MIN_C:
                    ratios.setdefault(n, []).append(math.hypot(v["a"], v["b"]) / math.hypot(e["a"], e["b"]))
    med_r = {n: statistics.median(v) for n, v in ratios.items()}
    for res in results:
        for rec in res.values():
            ok = {}
            for n, v in (rec.get("things") or {}).items():
                if not v or v["px"] <= 400 or n not in ref:
                    continue
                r = ref[n]
                c1, c2 = math.hypot(v["a"], v["b"]), math.hypot(r["a"], r["b"])
                h = math.degrees(math.atan2(v["b"], v["a"]) - math.atan2(r["b"], r["a"]))
                dh = abs((h + 180) % 360 - 180) if min(c1, c2) > LAB_MIN_C else 0.0
                good = dh <= LAB_HUE
                e = (rec.get("things_engine") or {}).get(n)
                if e and n in med_r and math.hypot(e["a"], e["b"]) > LAB_MIN_C:
                    ratio = c1 / math.hypot(e["a"], e["b"])
                    good = good and LAB_C_LO * med_r[n] <= ratio <= LAB_C_HI * med_r[n]
                ok[n] = good
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
        rounds = sorted(sd.glob("spec_round*.json"), key=lambda q: int(q.stem.split("round")[1]))
        spec = json.loads(rounds[-1].read_text()) if rounds else {}
        cam = next((c for c in spec.get("cameras") or [] if c["name"] == m["cam"]), {})
        # the face check is for the speaker only: people who are not speaking may be partly hidden, in profile or
        # seen from behind (a person partly behind another is fine)
        named = ({cam["speaker"]} if cam.get("speaker") else
                 {cam["line_subject"]} if cam.get("line_subject") else set())   # the speaker, else whom it is about
        acts = {k: ((v.get("body_pose") or {}).get("params") or {}).get("action") for k, v in
                (spec.get("characters") or {}).items()}
        ppl = []
        for p in m["people"]:
            tp = truth["people"].get(p["key"]) or {}
            j2 = tp.get("joints2d") or joints2d_from(res, p["key"], truth["camera"],
                                                      sd / "guides" / f"{m['cam']}_mask_{p['key']}.png")
            c = cast.get(p["key"]) or {}
            ppl.append(dict(p, joints2d=j2, wear=c.get("wear"), sheet=c.get("sheet"), noun=c.get("noun"),
                            creature=bool(c.get("creature")),
                            mask=str(sd / "guides" / f"{m['cam']}_mask_{p['key']}.png"), named=p["key"] in named,
                            action=acts.get(p["key"]) if acts.get(p["key"]) in ACTION_SEEN else None))
        path = jobs[iid]["path"] if ids is not None else str(out / f"{iid}.png")
        things = [{"name": t["name"], "mask": t["mask"], "words": t["words"], "place": t.get("place"),
                   "area": t.get("mask_area")} for t in truth.get("things") or [] if t.get("mask")]
        lit = [{"name": t["name"], "lit_mask": t["lit_mask"], "words": t["words"]}
               for t in truth.get("things") or [] if t.get("lit_mask")]
        if run.get("set") and "_look_opts" not in run:     # the room's own large pieces: the look question's options
            import setfacts
            run["_look_opts"] = [{"name": t["name"], "words": t["words"]} for t in setfacts.things(run["set"])
                                 if t.get("large") and t["kind"] != "window" and t["name"] != "rug"]
        items.append({"id": iid, "redraw": path, "engine": m["still"], "look_opts": run.get("_look_opts") or [],
                      "depth": str(sd / "guides" / f"{m['cam']}_depth.png"), "people": ppl, "things": things,
                      "lit": lit})
    out_p.write_text(json.dumps({"out": str(out_p.with_suffix("")) + "_result.json", "pose_model": pose_model,
                                 "judge": True, "items": items}, indent=1))
    print(out_p, len(items))


ACTION_WORDS = {
    "veil_lift": "{Who} raises both hands to the hem of {pos} veil under {pos} chin, about to lift it.",
    "look_at_sleeve": "{Who} bows {pos} head and looks down at the sleeve of {pos} raised left forearm.",
    "reach_to_forearm": "{Who} bends forward and lays {pos} hand on the other person's forearm."}

FIX = {"clothes colour": "{Who}'s clothes keep exactly the colours of the character sheet.",
       "arms": "{Who}'s arms hang in their sleeves clear of the body, both sleeves and hands drawn.",
       "hands": "{Who}'s hands are drawn where the engine frame shows them.",
       "neck": "{Who}'s neck is of natural length, the collar standing close under the jaw.",
       "outfit": ""}


def cmd_retry(run: dict, checked: dict, out_jobs: Path):
    out = Path(run["out"])
    neck_eval(checked)
    colour_eval(checked)
    wear_eval(checked)
    lit_eval(checked)
    face_eval(checked)
    action_eval(checked)
    hair_eval(checked)
    look_eval(run, checked)
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
            if t and x.startswith("lit "):
                add.append(f"{t['words'].split(' with ')[0].split(' (')[0].capitalize()} is lit: "
                           + ("the fire in its grate burns brightly with real flames and glowing coals, as in Picture 1."
                              if t.get("fire") else "its flame burns, as in Picture 1."))
            elif t and x.startswith("look "):        # its build and material, from its own words
                add.append(f"{t['words'].split(' (')[0].capitalize()} keeps exactly its build and material: "
                           f"{t['words'].split(' (', 1)[1].rstrip(')') if ' (' in t['words'] else t['words']}"
                           f"{', ' + t['place'] if t.get('place') else ''}; it is not drawn as any other piece of the "
                           "room.")
            elif t:
                add.append(f"{t['words'].split(' (')[0].capitalize()} {('(' + t['words'].split(' (', 1)[1]) if ' (' in t['words'] else ''}"
                           f"{', ' + t['place'] if t.get('place') else ''} keeps exactly the colour and shape of its "
                           f"reference picture.")
        for k, fs in f.items():
            who = nm.get(k, "the person")
            w = RD.who_of(cast, k)
            pm = next((p for p in meta[iid]["people"] if p["key"] == k), {})
            pos = "her" if (w.get("noun") == "woman") else "his"
            Who = who[0].upper() + who[1:]
            for x in fs:
                if x == "outfit":
                    add.append(RD.neck_words(w))
                elif x == "face":                # the facing the engine frame has, stated
                    add.append(f"{Who} {RD.facing_words(pm.get('head_to_camera_deg') or 0.0, pos)}, as in the "
                               f"engine frame; {pos} face is clearly drawn.")
                elif x == "action":              # the staged start action, stated
                    act = (rec.get("people", {}).get(k) or {}).get("action")
                    if act in ACTION_WORDS:
                        add.append(ACTION_WORDS[act].format(Who=Who, pos=pos))
                elif x == "missing":             # present in the engine still: drawn where it shows them
                    add.append(f"{Who} is in the picture where the engine frame shows "
                               f"{'himself' if pos == 'his' else 'herself'}, clearly drawn, not left out.")
                elif x == "hair":
                    add.append(f"{Who}'s hair is exactly as on the character sheet, the same in every shot.")
                else:
                    add.append(FIX[x].format(Who=Who))
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
    colour_eval(checked, retry)
    wear_eval(checked, retry)
    lit_eval(checked, retry)
    face_eval(checked, retry)
    action_eval(checked, retry)
    hair_eval(checked, retry)
    look_eval(run, checked, retry)
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
