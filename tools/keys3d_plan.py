"""The 3D flow's keys handed to the shots flow's LTX stage (11-25 user: the two flows are identical when it comes to
running LTX; the only difference is how the keyframes are made - the 3D flow's own job builder made unsynced lips and
frozen takes). The gated moments of flow3d's plan become a plan as shot_flow.run() saves it (one shot <id>_s0 a
moment) and a keys folder as shot_flow.supplied_keys(key_people_=True) reads it:

    build(R, cx, P, keep, log)   -> (plan file, keys folder, the shot ids to render, the moments without a key)
    blank(plan, mids)            the moments whose clip was not made: black in the cut, never an older clip

Who is in a key is what its 3D still shows (render/start_expect.json: the cast's boxes; the scene's unnamed guests,
tools/crowd.py, are never among them). Nothing here builds an LTX job."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

NO_KEY = "none"         # <id>_snone: the shot of a moment with no key to render - it has no clip and no key anywhere,
#                         so the cut shows it black and says so (a strict delivery never shows an older clip)
# a staged pose that is a posture held, not a movement (the staging's pose names: standing, sitting_chair, ...)
REST = re.compile(r"^(stand|sit|seat|ly|lie|lay|kneel|crouch|squat|lean)", re.I)
CROWD = "other people of the place"     # a still with guests whose keyframes give them no words


def staged(cx, e: dict) -> dict:
    """The shot the moment's still was rendered from: a talk moment's clean single (stage/<id>/shot_talk.json,
    remake_chapter.talk_shot) when it was made, else the scene layer's staged shot."""
    for p in ((cx.OUT / "stage" / e["id"] / "shot_talk.json") if e.get("route") == "talk" else None, e.get("shot")):
        try:
            if p and Path(p).exists():
                return json.loads(Path(p).read_text())
        except (OSError, ValueError):
            pass
    return {}


def visible(cx, e: dict, sh: dict) -> list:
    """[{"name", "box"}] left to right: the cast the 3D still shows, each with its box as 0-1 of the picture (the
    figures of render/start_expect.json; one out of frame is in its "offscreen", not here). Without that file the
    staged shot's figures, no boxes. The guests (shot["crowd"]) are not cast: never listed."""
    guests = {g.get("who") for g in sh.get("crowd") or []}
    ex = cx.OUT / "stage" / e["id"] / "render" / "start_expect.json"
    try:
        d = json.loads(ex.read_text())
    except (OSError, ValueError):
        return [{"name": g["who"]} for g in sh.get("figures", []) if g.get("who") and g["who"] not in guests]
    W, H = (d.get("size") or [1152, 640])[:2]
    out = []
    for f in d.get("figures", []):
        if not f.get("who") or f["who"] in guests or any(v["name"] == f["who"] for v in out):
            continue
        v = {"name": f["who"]}
        b = f.get("bbox")
        if b and len(b) == 4:
            b = [round(min(max(b[0] / W, 0.0), 1.0), 4), round(min(max(b[1] / H, 0.0), 1.0), 4),
                 round(min(max(b[2] / W, 0.0), 1.0), 4), round(min(max(b[3] / H, 0.0), 1.0), 4)]
            if b[0] < b[2] and b[1] < b[3]:         # (a sliver at the frame's edge: no box, the order still says where)
                v["box"] = b
        out.append(v)
    return sorted(out, key=lambda v: (v["box"][0] + v["box"][2]) / 2 if "box" in v else 0.5)


def moves(R, e: dict, sh: dict) -> bool:
    """Does the moment's own staging mark movement? Its motion is the point (remake_chapter.needs_clip: contacts, fx,
    a recipe, a worked prop, a creature, someone carried), its camera has an authored move, or a figure's staged pose
    is not a posture held (walking, bowing). Every keyframe has motion words, so those say nothing either way."""
    if R.needs_clip(e) or R.camera_path(e.get("shot")):
        return True
    return any(isinstance(g.get("pose"), str) and g["pose"] and not REST.match(g["pose"]) for g in sh.get("figures", []))


def framing(sh: dict) -> str:
    f = str((sh.get("camera") or {}).get("framing") or "").lower()
    return "close" if "close" in f else "wide" if re.search(r"wide|establish|long", f) else "medium"


def words(R, S, cx, e: dict, who: list, sides: dict) -> str:
    """The moment's motion words as the shots flow's director writes an action: the keyframe plan's motion (its
    camera sentences and quotation marks out, as remake_chapter.i2v_prompt), every name as that person's one
    description (shot_flow.describe) with the side the key shows them on, then the authored camera move's words."""
    f = cx.frames.get(e["id"]) or {}
    t = R.no_quotes(re.sub(r"(?i)\b(the )?camera\b[^.]*\.", "", f.get("motion", "")).strip())
    Rb = S.RG.Book(cx.book)
    people = Rb.people()
    try:
        dress = S.MF._dress(cx.book, cx.stem, e["id"], people)
    except Exception:                                   # noqa: BLE001 - no wardrobe: the looks' own clothes
        dress = {}
    for w in sorted(people, key=len, reverse=True):
        tok = [w] + [x for x in re.findall(r"[A-Z][a-z'\-]+", w) if len(x) > 2 and Rb.person(x) == w]
        d = S.describe(Rb, w, dress) + (f" {sides[w]}" if w in sides and len(sides) > 1 else "")
        t = re.sub(r"\b(" + "|".join(map(re.escape, tok)) + r")\b", lambda m, d=d: d, t)
    path = R.camera_path(e.get("shot"))
    cam = R.camera_phrase(path) if path else ""
    return " ".join(x for x in (t[:1].upper() + t[1:], "" if cam == R.STATIC_PHRASE else cam) if x)


def shot_of(R, S, cx, e: dict) -> tuple:
    """(the moment's one shot, its key's visible people) in the shots flow's own fields."""
    sh = staged(cx, e)
    vis = visible(cx, e, sh)
    Rb = S.RG.Book(cx.book)
    who = list(dict.fromkeys(Rb.person(v["name"]) or v["name"] for v in vis))
    pl = R.shot_place(cx, e)
    d = cx.OUT / "stage" / e["id"]
    f = cx.frames.get(e["id"]) or {}
    talk = e.get("route") == "talk"
    # the guests behind the cast: the shot says so with its crowd words, and no head is counted against the cast
    crowd = (R.scene_crowd(cx.frames, e["id"]) or CROWD) if sh.get("crowd") else ""
    held = {}
    for g in sh.get("figures", []):         # a posture the staging holds ("stays sitting throughout")
        w = Rb.person(g.get("who")) or g.get("who")
        c = S.pose_cat(str(g.get("pose") or "").replace("_", " ")) if REST.match(str(g.get("pose") or "")) else ""
        if w in who and c:
            held[w] = c
    s = {"id": f"{e['id']}_s0", "dur": float(e["dur"]), "who": who, "framing": framing(sh),
         "kind": "talk" if talk else "action" if moves(R, e, sh) else "quiet",
         "place": pl["where"], "picture": f.get("prompt", ""), "pose_held": held,
         # the place's words as the 3D prompts get them: the ones its set was drawn from, else the book's
         "room": (cx.sets.get(pl["where"]) or {}).get("desc") or pl["desc"] or "", "fixtures": [],
         "scene": str((sh.get("scene") or {}).get("id") or ""), "crowd": crowd, "crowd_behind": bool(crowd)}
    if talk:
        spk = Rb.person(e.get("speaker")) or e.get("speaker")
        ots = ((sh.get("camera") or {}).get("ots") or [None])[0]    # whose back is to the camera, when staged so
        ots = Rb.person(ots) or ots
        s.update(action="", listener=next((w for w in who if w != spk), None),
                 ots=ots if ots in who and ots != spk and len(who) > 1 else None)
    else:
        bx = {(Rb.person(v["name"]) or v["name"]): v["box"] for v in vis if "box" in v}
        s.update(action=words(R, S, cx, e, who, S.key_sides(who, bx) if who else {}), setting="")
        dv = d / "render" / "depth.mp4"
        if dv.exists():                     # the staged motion guides the clip (shot_flow._guided)
            s["control"] = {"video": str(dv), "attn": R.MODE2_ATTN}
            if R.camera_path(e.get("shot")) and (d / "key_end.png").exists():   # the authored move's last camera
                s["control"].update(key_end=str(d / "key_end.png"), key_end_strength=R.END_GUIDE)
    return s, vis


def build(R, cx, P: dict, keep: set, log=print) -> tuple:
    """P's moments as a shots-flow plan + keys folder under <flow3d>/shots_ltx. `keep`: the moments whose key passed
    the gate; one of them with a key (stage/<id>/key.png, no FAILED beside it) gets its shot <id>_s0, its key and
    its shot_facts.json. Every other moment of P keeps its place in the cut as a shot with nothing to render.
    -> (plan.json, keys/, [shot ids to render], [moments without a key])"""
    import shot_flow as S
    from PIL import Image
    out = cx.OUT / "shots_ltx"
    keys = out / "keys"
    if keys.exists():
        shutil.rmtree(keys)                 # (this run's keys only: a dropped moment's old key is not offered again)
    keys.mkdir(parents=True)
    es, by, ids, lost = [], {}, [], []
    for e in sorted(P["entries"], key=lambda e: e["t"]):
        d = cx.OUT / "stage" / e["id"]
        ok = e["id"] in keep and e.get("route") in ("talk", "stage") and (d / "key.png").exists() \
            and not (d / "FAILED").exists()
        if not ok:
            lost.append(e["id"])
            es.append(dict(e, who=list(e.get("who") or []), where=e.get("where") or ""))
            by[e["id"]] = [{"id": f"{e['id']}_s{NO_KEY}", "dur": float(e["dur"]), "who": [], "kind": "quiet",
                            "framing": "medium", "place": "", "action": "", "picture": "", "crowd": ""}]
            continue
        s, vis = shot_of(R, S, cx, e)
        with Image.open(d / "key.png") as im:       # the clip's size, RGB (shot_flow.check_supplied)
            if im.size == (S.W, S.H) and im.mode == "RGB":
                shutil.copyfile(d / "key.png", keys / f"{s['id']}.png")
            else:
                im.convert("RGB").resize((S.W, S.H), Image.LANCZOS).save(keys / f"{s['id']}.png")
        (keys / f"{s['id']}.shot_facts.json").write_text(json.dumps({"visible": vis}, indent=1))
        x = dict(e, who=s["who"], where=s["place"])
        if s["kind"] == "talk":             # the line's words for the talk prompt, as the shots flow quotes them
            x["words"] = re.sub(r"[“”\"‘’]|(?<!\w)'|'(?!\w)", "", str(e.get("line") or "")).strip()
            x["emotion"] = e.get("emotion") or ""
        es.append(x)
        by[e["id"]] = [s]
        ids.append(s["id"])
    plan = out / "plan.json"
    plan.write_text(json.dumps({"_sig": "flow3d", "_es": es, **by}, indent=1))
    log(f"3d keys -> the shots flow's LTX: {len(ids)} shot(s) ("
        f"{sum(by[i.rsplit('_s', 1)[0]][0]['kind'] == 'talk' for i in ids)} talk, "
        f"{sum('control' in by[i.rsplit('_s', 1)[0]][0] for i in ids)} with a motion guide), "
        f"{len(lost)} moment(s) without a key")
    return plan, keys, ids, lost


def blank(plan: Path, mids) -> None:
    """The moments whose clip was not made this run (a talk shot stopped for its speaker, a failed render): their
    shot becomes the no-key one, so the cut never shows a clip of an earlier key."""
    by = json.loads(Path(plan).read_text())
    for m in mids:
        for s in by.get(m) or []:
            s["id"] = f"{m}_s{NO_KEY}"
    Path(plan).write_text(json.dumps(by, indent=1))
