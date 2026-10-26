"""SCENE layer: one persistent 3D blocking per scene, every moment a camera on it (any book, any chapter).

    P=~/tts-ab/.venv-pose/bin/python
    $P -I tools/scene_stage.py <book> <chapter> [--extra-lines "92.1" "110.9" ...] [--sheets] [--only SCENE ...]

Why: per-shot staging re-places people in every shot, so in a dialogue they switch sides between cuts. Here:
  1. MOMENTS: the chapter's anim/manifest.json (action + talk) + stage/extra_moments.json (talk moments for quoted
     lines on still stretches, made with --extra-lines T ...). A talk moment's speaker = the timeline quote at t.
  2. SCENES: consecutive moments in the same place (the plan picture's "where"), gap < 90 s.
  3. BLOCKING (per scene): the reference moment = the one with the most people; its planner staging
     (tools/blockout_plan.frame_to_shot) gives the MARKS (x, y, facing, raised ground) of everyone in it; anyone else
     in the scene gets a mark off the end of the line, facing the group. The scene AXIS = [left, right] = the two
     principal figures in screen order from the reference camera: every camera of the scene stays on that side
     (tools/blockout.solve_camera "line"), so left/right never flips between shots. Set props (anvil, workbench,
     bellows ...) are placed once per scene and persist.
  4. SHOTS (per moment): the moment's own poses / held props / contacts from the planner, the people on their scene
     marks, a camera from the plan's framing on the scene staging: wide = master, medium = two-shot (or a single),
     close-up / medium close-up = single; TALK = a single on the speaker (medium close-up: the face >= 15 % of the
     frame height) or an over-the-shoulder from the listener when two people talk. Camera MOVES the text names
     (push in / pull back / pan / track / crane / orbit) become an authored camera path (camera.path, eased keys,
     start = the plan's framing, end = the move's target). Static otherwise.
Writes <chapter>/stage/shots/<moment>.json, <chapter>/stage/scenes/<scene>.json, <chapter>/stage/extra_moments.json,
and with --sheets a contact sheet per scene (<chapter>/stage/scenes/<scene>.jpg: every shot's staged start frame,
with who is screen-left/right written under it).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import blockout_plan as BP  # noqa: E402
import blockout_rig as rig  # noqa: E402
import characters as CH  # noqa: E402

FPS = 24
MOVES = [  # (regex on the moment's prompt + motion, move kind)
    (r"\b(push(es|ing)? in|zoom(s|ing)? in|dolly in|moves? closer|closes in on)\b", "push_in"),
    (r"\b(pull(s|ing)? back|zoom(s|ing)? out|dolly out|reveal(s|ing)?|widens?)\b", "pull_back"),
    (r"\bpan(s|ning)? (to the )?left\b", "pan_left"),
    (r"\bpan(s|ning)? (to the )?right\b|\bpans?\b", "pan_right"),
    (r"\b(track(s|ing)?|follows?|alongside)\b", "track"),
    (r"\b(crane(s|ing)?|tilt(s|ing)? up|rises? up to)\b", "crane_up"),
    (r"\b(orbit(s|ing)?|circles? around)\b", "orbit"),
]
TIGHTER = {"wide": "medium", "medium": "medium close-up", "medium close-up": "close-up", "close-up": "close-up",
           "two-shot": "medium close-up", "low angle": "medium", "high angle": "medium"}
WIDER = {v: k for k, v in TIGHTER.items() if k != v}
WIDER["close-up"] = "medium close-up"


# ------------------------------------------------------------------------------------------------ moments
def load(book, chapter):
    vd = CH.vdir_of(book)
    cd = vd / chapter
    man = json.loads((cd / "anim" / "manifest.json").read_text())
    kf = {f["shot"]: f for f in json.loads((cd / "anim" / "keyframes.json").read_text())["frames"]}
    plan = json.loads((cd / "plan.json").read_text())
    tl = json.loads((vd.parent / f"{chapter}.timeline.json").read_text())
    cast_p = vd / "stage" / "characters" / "index.json"
    cast = json.loads(cast_p.read_text()) if cast_p.exists() else {}
    return vd, cd, man, kf, plan, tl, cast


def quote_at(tl, t):
    """The timeline segment being spoken at t (a quote: its speaker), else None."""
    seg = None
    for s in tl:
        if s["t"] <= t + 0.05:
            seg = s
        else:
            break
    return seg


def extra_moments(cd, man, tl, plan, lines):
    """Talk moments for quoted lines on still stretches (t values): dur = to the next timeline segment."""
    out = []
    pics = plan["pictures"]
    for t in lines:
        t = float(t)
        i = max(k for k, s in enumerate(tl) if s["t"] <= t + 0.05)
        s = tl[i]
        nxt = tl[i + 1]["t"] if i + 1 < len(tl) else s["t"] + 2.0
        pic = pic_at(man, t)
        out.append({"id": f"talk-{int(round(s['t'] * 10)):06d}", "kind": "talk", "t": round(s["t"], 2),
                    "dur": round(nxt - s["t"], 2), "pic": pic, "speaker": s.get("speaker"), "text": s.get("text"),
                    "extra": True})
    (cd / "stage").mkdir(parents=True, exist_ok=True)
    (cd / "stage" / "extra_moments.json").write_text(json.dumps(out, indent=1))
    return out


def pic_at(man, t):
    prev = None
    for m in sorted(man, key=lambda m: m["t"]):
        if m["t"] <= t:
            prev = m.get("pic")
    return prev


def moment_frame(m, kf, plan, tl):
    """The planner frame of a moment: an action moment's keyframe; a talk moment's synthesised from the
    picture (who, where) and the quote (speaker, line)."""
    pics = {p["id"]: p for p in plan["pictures"]}
    p = pics.get(m.get("pic")) or {}
    if m["kind"] == "action" and m["id"] in kf:
        f = dict(kf[m["id"]])
        f.setdefault("where", p.get("where"))
        return f
    q = quote_at(tl, m["t"]) or {}
    spk = m.get("speaker") or (q.get("speaker") if q.get("kind") == "quote" else None)
    who = list(p.get("who") or [])
    if spk and spk not in who and spk != "narrator":
        who = [spk] + who
    others = [w for w in who if w != spk]
    # the quote stays OUT of the prompt the planner reads ("I want to go after the dragon" is not a dragon here)
    return {"shot": m["id"], "who": who, "where": p.get("where"), "speaker": spk, "line": q.get("text", ""),
            "prompt": (f"Medium close-up. {spk} speaks to {others[0]}." if others else f"Medium close-up. {spk} speaks.")
            if spk else "Medium shot.",
            "motion": f"{spk} talks." if spk else ""}


def _fixed_words():
    """The words of every FIXED prop class (stage_templates/prop_classes.json mobility "fixed"): they live in one
    place (a workbench, a hearth, a door)."""
    import props_build as PB
    return tuple(w for r in PB.CLASSES["physical"] if r.get("mobility") == "fixed" for w in r["words"])


FIXED_WORDS = _fixed_words()
import props_build as _PB  # noqa: E402
ARCH_WORDS = tuple(w for r in _PB.CLASSES["physical"] if r.get("architectural") for w in r["words"])
_ARCH_SIZE = {r["base"]: tuple(r.get("opening_m") or (1.0, 2.1)) for r in _PB.CLASSES["physical"] if r.get("architectural")}



def canon_names(frames, cast):
    """Timeline / keyframe names -> the cast's names ("Maren", "Princess" -> "Princess Maren")."""
    names = list((cast.get("characters") or {}))

    def canon(w):
        if not w or w in names:
            return w
        hits = [n for n in names if w.lower() in n.lower().split() or n.lower() in w.lower()]
        return hits[0] if len(hits) == 1 else w
    for f in frames.values():
        f["who"] = list(dict.fromkeys(canon(w) for w in (f.get("who") or [])))
        if f.get("speaker"):
            f["speaker"] = canon(f["speaker"])


def rehome(frames, order=None):
    """A moment whose FIXED props (a workbench, bellows, an anvil: they live in one place) are seen only at
    another location in this chapter belongs to that location (the keyframe's "where" was wrong)."""
    seen = {}
    for f in frames.values():
        for p in f.get("props") or []:
            for w in FIXED_WORDS:
                if w in p.lower():
                    seen.setdefault(w, {}).setdefault(f.get("where"), 0)
                    seen[w][f.get("where")] += 1
    moved = []
    order = order or list(frames)
    nb = {}
    for i, mid in enumerate(order):                     # the neighbours in time: a scene's continuation
        nb[mid] = {frames[order[j]].get("where") for j in (i - 1, i + 1) if 0 <= j < len(order)}
    for mid, f in frames.items():
        if f.get("scene"):
            continue                                  # the plan's scene set its place (10-22): never re-homed
        votes = {}
        for p in f.get("props") or []:
            for w in FIXED_WORDS:
                if w in p.lower() and w in seen:
                    for loc, n in seen[w].items():
                        votes[loc] = votes.get(loc, 0) + n
        if not votes:
            continue
        txt = (f.get("prompt", "") + " " + f.get("motion", "")).lower()
        wwords = [w for w in re.findall(r"[a-z]{4,}", (f.get("where") or "").lower())
                  if w not in ("behind", "below", "above", "edge", "the", "north", "south", "east", "west")]
        if any(re.search(rf"\b{w}", txt) for w in wwords):
            continue                                   # the text itself names its place: trust it
        own = sum(1 for p in f.get("props") or [] for w in FIXED_WORDS if w in p.lower() and w in seen)
        votes[f.get("where")] = votes.get(f.get("where"), 0) - own      # not its own votes
        best = max(votes, key=votes.get)
        here = votes.get(f.get("where"), 0)
        # only when no OTHER moment shows these fixed props at the stated place (a second workbench there
        # = the place really has one, e.g. a shed scene of two shots)
        if best != f.get("where") and votes[best] >= 3 and here <= 0:
            moved.append((mid, f.get("where"), best))
            f["where_text"] = f.get("where")
            f["where"] = best
    return moved


def world_fixed(where, cast):
    """The location's own fixed objects (tools/world_locate.py -> <world>/fixed_props.json), or {}."""
    vd = Path(cast.get("book") or "")
    ap = vd / "stage" / "assets.json" if vd.name else None
    if not ap or not ap.exists():
        return {}, None
    ent = next((e for e in json.loads(ap.read_text()).get("sets", []) if e.get("where") == where), None)
    if not ent:
        return {}, None
    fp = Path(ent["world_dir"]) / "fixed_props.json"
    return (json.loads(fp.read_text()) if fp.exists() else {}), ent


def anchor_fixed(shot, where, cast):
    """FIXED props belong to the SET: one the world already has (its own workbench, door, bellows) is anchored
    at the world's position and HIDDEN (the world draws it; it stays the support surface + collider); one the
    world lacks is placed by its prop-class "near" hint (the anvil in front of the forge fire) and rendered."""
    wf, ent = world_fixed(where, cast)
    if not wf:
        return
    classify = _families()
    room_c = np.array((ent.get("room") or {}).get("capture_xy") or [0.0, 0.0], float)
    for pr in shot["set"].get("props", []):
        nm = (pr.get("prop") or pr.get("name") or "").lower()
        cls, r = classify(nm)
        if cls != "physical" or r.get("mobility") != "fixed":
            continue
        x, y, z, w, d, h = pr["box"]
        hit = wf.get(nm) or wf.get(r.get("base"))
        # only ARCHITECTURE is the set's own (a door, a hearth): the sets are drawn clean of the staged props
        # (user 10-21), so a work prop "found" in one is a misreading (10-22: the Forge Shed's hearth was also
        # read as an anvil; the anvil model was hidden and the blade lay on the hearth's proxy box)
        if hit and not r.get("architectural"):
            hit = None
        if hit:
            pr["box"] = [hit["at"][0], hit["at"][1], 0.0, w, d, h]
            pr["yaw"] = hit.get("yaw", pr.get("yaw", 0))
            pr["hidden"] = True
            pr["anchored"] = "world"
            continue
        near = r.get("near")
        tgt = (wf.get(near["object"]) or (wf.get("hearth") if re.search(r"fire|forge|hearth", near["object"]) else None)) \
            if near else None                       # the set locator calls the forge fire "hearth"
        if near and tgt:
            o = np.array(tgt["at"], float)
            v = room_c - o
            v = v / max(np.linalg.norm(v), 1e-9)
            p_ = o + v * float(near["dist"])
            if near.get("beside"):                    # BESIDE the object (the bellows by the hearth), not in front
                side_ = np.array([-v[1], v[0]])
                p_ = o + v * float(near["dist"]) * 0.35 + side_ * float(near["beside"])
            room = (ent or {}).get("room")
            if room:                                  # on the room's FREE floor (not inside the hearth)
                import world_room as WR
                p_ = np.array(WR.clamp_into_room(room, [float(p_[0]), float(p_[1]), 0.5])[:2])
            # CLEAR of the props already placed (10-22: the bellows' nozzle nearly touched the anvil): footprints (the
            # modelled size when there is one) at least 0.4 m apart, the prop slid sideways (away) until they are
            import blockout_props as PR_
            def rad(q):
                try:
                    ex_ = PR_.set_entry(q)[1].get("extent_m")
                except Exception:                      # noqa: BLE001
                    ex_ = None
                return 0.5 * float(max(ex_[:2]) if ex_ else max(q["box"][3], q["box"][4]))
            others = [q for q in shot["set"].get("props", []) if q is not pr and q.get("anchored") and not q.get("hidden")]
            side2 = np.array([-v[1], v[0]])
            for _ in range(12):
                hit_ = next((q for q in others if np.linalg.norm(p_ - np.array(q["box"][:2], float)) <
                             rad(q) + rad(dict(pr, box=[0, 0, 0, w, d, h])) + 0.4), None)
                if hit_ is None:
                    break
                away = np.sign(np.dot(p_ - np.array(hit_["box"][:2], float), side2)) or 1.0
                p_ = p_ + side2 * away * 0.25
            pr["box"] = [round(float(p_[0]), 3), round(float(p_[1]), 3), 0.0, w, d, h]
            pr["yaw"] = float(np.degrees(np.arctan2(v[1], v[0]))) + 90.0
            if near.get("aim"):                       # its working end AT the object (the bellows' nozzle at the fire):
                to = o - p_                           # yaw = the prop's +x (its nozzle end) along to the object
                pr["yaw"] = float(np.degrees(np.arctan2(to[1], to[0]))) + float(near.get("aim_yaw", 0.0))
            pr["anchored"] = f"near the world's {near['object']}"


# ------------------------------------------------------------------------------------------------ scenes
def scenes_of(moments, frames, gap=90.0):
    out, cur = [], None
    for m in sorted(moments, key=lambda m: m["t"]):
        w = frames[m["id"]].get("where")
        sid = frames[m["id"]].get("scene")            # the plan's scene (10-22): its id, never a time gap; a moment
        new = (sid != cur.get("plan_scene") or w != cur["where"]) if (cur is not None and sid) else \
            (cur is None or w != cur["where"] or m["t"] - cur["t1"] > gap)   # in its own place (a cutaway) stages there
        if new:
            nid = sid or f"s{len(out) + 1:02d}"
            while any(o["id"] == nid for o in out):
                nid += "b"                            # the scene resumed after a cutaway: s02, s02b
            cur = {"id": nid, "where": w, "moments": [], "t0": m["t"], "t1": m["t"], "plan_scene": sid}
            out.append(cur)
        cur["moments"].append(m["id"])
        cur["t1"] = m["t"] + m.get("dur", 0)
    return out


OPERATE_RX = r"\b(pump|pumps|pumped|pumping|work|works|worked|working|push|pushes|pushed|pushing)\b"


def operate_rule(shot, frame):
    """A person the text has WORKING a jointed set prop (pumps the bellows) operates it (10-22: the bellows were
    never touched): shot["operate"] = [{who, prop, joint}], the person stands at the part's handle (0.42 m out from
    it, away from the hinge) facing it; the clip renderer drives the part through its cycle and locks the hands on
    the moving handle. The person = the one named in the same sentence, else the shot's first person."""
    import blockout_props as P_
    txt = " ".join([frame.get("prompt") or "", frame.get("motion") or "", " ".join(frame.get("must") or [])])
    people = [f for f in shot["figures"] if f.get("kind") != "creature" and not f.get("attach")]
    if not people:
        return
    for pr in shot["set"].get("props", []):
        nm = (pr.get("prop") or pr.get("name") or "")
        try:
            _, e = P_.set_entry(pr)
        except Exception:                          # noqa: BLE001 - an unknown prop has no joints
            continue
        for jd in e.get("joints") or []:
            if not jd.get("cycle") or not jd.get("handle_m"):
                continue
            hit = None
            for sent in re.split(r"(?<=[.!?])\s+", txt):
                if re.search(OPERATE_RX, sent, re.I) and re.search(rf"\b{re.escape(nm.split('__')[0])}", sent, re.I):
                    hit = sent
                    break
            if not hit:
                continue
            who = next((f for f in people if f["who"].split()[0].lower() in hit.lower()), people[0])
            hw, pw = P_.joint_handle(pr, jd["name"], None)
            out = np.array([hw[0] - pw[0], hw[1] - pw[1]])
            out = out / max(np.linalg.norm(out), 1e-6)
            stand = hw[:2] + out * 0.42
            who["at"] = [round(float(stand[0]), 3), round(float(stand[1]), 3)]
            who["facing"] = round(float(np.degrees(np.arctan2(-out[1], -out[0]))), 1)
            who.pop("ground_z", None)
            shot.setdefault("operate", []).append({"who": who["who"], "prop": nm, "joint": jd["name"],
                                                   "hands": ["right", "left"]})
            shot.setdefault("plan_notes", {})["operate"] = f"{who['who']} works the {nm} ({jd['name']})"


def ledge_rule(shot, frame):
    """A figure the text puts ON a ledge / rock / step stands on raised ground: a set prop with that top."""
    txt = (frame.get("prompt", "") + " " + frame.get("motion", "") + " " + " ".join(frame.get("must") or [])).lower()
    for f in shot["figures"]:
        nm = (f.get("who") or "").split()[0].lower()
        if f.get("kind") == "creature" or not nm:
            continue
        if re.search(rf"{re.escape(nm)}\w*[^.]{{0,60}}\b(on|atop|upon) (a|an|the)\b[^.]{{0,25}}\b(ledge|rock|outcrop|step|platform|boulder)", txt):
            h = 2.4 if "ledge" in txt else 1.2
            f["ground_z"] = h
            shot["set"].setdefault("props", []).append(
                {"name": "ledge", "box": [f["at"][0] - 0.9, f["at"][1] - 0.7, 0, 1.8, 1.4, h], "support": True})


def block_scene(sc, moments, frames, cast, chapter, prior=None):
    """The scene's marks + axis from its reference moment (the most people). prior: an earlier scene at the SAME
    place: its marks, axis and set props are kept for the people it had (a location keeps its layout)."""
    ms = {m["id"]: m for m in moments}
    best, best_n = None, -1
    for mid in sc["moments"]:
        f = frames[mid]
        if ms[mid]["kind"] != "action":
            continue
        n = len(f.get("who") or [])
        if n > best_n:
            best, best_n = mid, n
    ref = best or sc["moments"][0]
    shot = BP.frame_to_shot(frames[ref], cast=cast, chapter=chapter, t=ms[ref]["t"])
    for pr in shot["set"].get("props", []):
        pr["fixed"] = True
    ledge_rule(shot, frames[ref])
    marks = {}
    for f in shot["figures"]:
        marks[f["who"]] = {k: copy.deepcopy(f[k]) for k in ("at", "facing", "ground_z", "z", "kind", "height")
                           if k in f}
    # everyone else in the scene: off the end of the line, facing the group
    people = []
    for mid in sc["moments"]:
        for w in frames[mid].get("who") or []:
            if w not in people:
                people.append(w)
        spk = frames[mid].get("speaker")
        if spk and spk not in people:
            people.append(spk)
    if marks:
        xs = [m["at"][0] for m in marks.values()]
        cy = float(np.mean([m["at"][1] for m in marks.values()]))
        for w in people:
            if w not in marks:
                x = max(xs) + 1.4
                xs.append(x)
                kind = (cast.get("characters", {}).get(w) or {}).get("kind", "human")
                marks[w] = {"at": [round(x, 3), round(cy, 3)], "facing": 180.0, "kind": kind}
    # the axis: the two principal figures, screen order from the reference camera (left first)
    cam = shot.get("camera") or {}
    pos = np.asarray(cam.get("pos", [0, 0, 1.6]), float)
    look = np.asarray(cam.get("look", [0, 5, 1.6]), float)
    fwd = look - pos
    fwd[2] = 0
    fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
    right = np.array([fwd[1], -fwd[0], 0.0])
    count = {}
    for mid in sc["moments"]:
        for w in frames[mid].get("who") or []:
            count[w] = count.get(w, 0) + 1
    principal = sorted(marks, key=lambda w: -count.get(w, 0))[:2]
    axis = sorted(principal, key=lambda w: float(np.dot(np.array(marks[w]["at"] + [0.0]) - pos, right)))
    # FIXED set props used anywhere in the scene (an anvil, a workbench, bellows): placed ONCE, at the planner's
    # offset from the person it was set beside, measured from that person's scene mark
    have = {(pr.get("prop") or pr.get("name") or "").lower() for pr in shot["set"].get("props", [])}
    for mid in sc["moments"]:
        if ms[mid]["kind"] != "action":
            continue
        b = BP.frame_to_shot(frames[mid], cast=cast, chapter=chapter, t=ms[mid]["t"])
        f0 = next((f for f in b["figures"] if f.get("kind") == "human" and f["who"] in marks), None)
        if f0 is None:
            continue
        for pr in b["set"].get("props", []):
            key = (pr.get("prop") or pr.get("name") or "").lower()
            if not key or key in have:
                continue
            q = copy.deepcopy(pr)
            dx, dy = q["box"][0] - f0["at"][0], q["box"][1] - f0["at"][1]
            q["box"][0] = round(marks[f0["who"]]["at"][0] + dx, 3)
            q["box"][1] = round(marks[f0["who"]]["at"][1] + dy, 3)
            q["fixed"] = True
            shot["set"].setdefault("props", []).append(q)
            have.add(key)
    anchor_fixed(shot, sc["where"], cast)
    _wf, _ent = world_fixed(sc["where"], cast)
    room_ = (_ent or {}).get("room")
    if prior:
        for w, mk in prior["marks"].items():
            if w in marks:
                marks[w] = copy.deepcopy(mk)
        if all(w in marks for w in prior["axis"]) and len(prior["axis"]) == 2:
            axis = list(prior["axis"])
        new_props = [p for p in shot["set"].get("props", []) if (p.get("prop") or p.get("name")) not in
                     {(q.get("prop") or q.get("name")) for q in prior["set"].get("props", [])}]
        shot["set"] = copy.deepcopy(prior["set"])
        shot["set"].setdefault("props", []).extend(new_props)
        pos = np.asarray(prior["reference_camera"]["pos"], float)
        look = np.asarray(prior["reference_camera"]["look"], float)
        cam = prior["reference_camera"]
    # NEAR THE CAPTURE POINT: a pano set holds up only near where it was captured (farther = stretched streaks).
    # A scene with no world-anchored props is blocked around a spot ~4 m in front of the capture point (the
    # layout keeps its shape; a prior layout at the place is kept as is)
    if _ent and not prior and marks:
        cap_ = np.array(((_ent.get("room") or {}).get("capture_xy")) or [0.0, 0.0], float)
        anchored_ = any(p_.get("anchored") == "world" for p_ in shot["set"].get("props", []))
        cen_ = np.mean([np.asarray(m_["at"][:2], float) for m_ in marks.values()], axis=0)
        if not anchored_ and float(np.linalg.norm(cen_ - cap_)) > 5.0:
            cc = (_ent.get("capture_camera") or {})
            fwd_ = np.asarray(cc.get("look_at", [0, 10, 0])[:2], float) - np.asarray(cc.get("pos", [0, 0, 0])[:2], float)
            fwd_ = fwd_ / max(np.linalg.norm(fwd_), 1e-9)
            delta = cap_ + fwd_ * 4.0 - cen_
            for m_ in marks.values():
                m_["at"] = [round(float(m_["at"][0] + delta[0]), 3), round(float(m_["at"][1] + delta[1]), 3)] + \
                    list(m_["at"][2:])
            for p_ in shot["set"].get("props", []):
                if p_.get("anchored"):
                    continue
                p_["box"][0] = round(float(p_["box"][0] + delta[0]), 3)
                p_["box"][1] = round(float(p_["box"][1] + delta[1]), 3)
            pos = np.asarray(pos, float) + np.array([delta[0], delta[1], 0.0])
            look = np.asarray(look, float) + np.array([delta[0], delta[1], 0.0])
            sys.stderr.write(f"{sc['id']}: blocking moved {np.linalg.norm(delta):.1f} m toward the set's capture point\n")
        # a HUGE creature (taller than ~4 m) stands FARTHER out along the capture look (~9 m): the cameras stay near
        # the capture point (the pano holds there) and see the people in front of it AND the creature whole
        cc = (_ent.get("capture_camera") or {})
        fwd2 = np.asarray(cc.get("look_at", [0, 10, 0])[:2], float) - np.asarray(cc.get("pos", [0, 0, 0])[:2], float)
        fwd2 = fwd2 / max(np.linalg.norm(fwd2), 1e-9)
        for w_, m_ in marks.items():
            if m_.get("kind") == "creature" and float(m_.get("height") or 6.0) >= 2.5 * 1.8 - 1e-6 and not anchored_:
                side_ = np.array([fwd2[1], -fwd2[0]])
                lat = float(np.dot(np.asarray(m_["at"][:2], float) - cap_, side_))
                newp = cap_ + fwd2 * float(_ent.get("creature_dist_m", 9.0)) + side_ * float(np.clip(lat, -3, 3))
                m_["at"] = [round(float(newp[0]), 3), round(float(newp[1]), 3)] + list(m_["at"][2:])
                sys.stderr.write(f"{sc['id']}: {w_} (a huge creature) stands {_ent.get('creature_dist_m', 9.0)} m out along the capture look\n")
    _wj = {}
    if _ent and (Path(_ent["world_dir"]) / "world.json").exists():
        _wj = json.loads((Path(_ent["world_dir"]) / "world.json").read_text())
    return {"id": sc["id"], "where": sc["where"], "ref_moment": ref, "marks": marks, "axis": axis, "room": room_,
            "anchors": _wj.get("anchors") or {}, "world_dir": (_ent or {}).get("world_dir"),
            "world_kind": _wj.get("kind"),
            "capture_xy": ((_ent or {}).get("room") or {}).get("capture_xy", [0.0, 0.0]),
            "set": shot["set"], "reference_camera": {"pos": pos.tolist(), "look": look.tolist(),
                                                     "lens": cam.get("lens", 35)}, "moments": sc["moments"]}


# ------------------------------------------------------------------------------------------------ shots
def camera_for(m, frame, base, scene):
    fr = (base.get("camera") or {}).get("framing", "medium")
    who = [f["who"] for f in base["figures"]]
    humans = [f["who"] for f in base["figures"] if f.get("kind") != "creature"]
    axis = [w for w in scene["axis"] if w in scene["marks"]]
    cam = {"framing": fr}
    if len(axis) == 2:
        cam["line"] = axis
    if m["kind"] == "talk":
        spk = frame.get("speaker")
        if spk in who and spk not in humans:            # a creature speaks: frame its head (the jaw moves)
            cam.update(framing="medium close-up", target=[f"{spk}:head"])
            return cam
        listeners = [w for w in humans if w != spk]
        def _gap(a, b):
            pa, pb = scene["marks"].get(a, {}).get("at"), scene["marks"].get(b, {}).get("at")
            return float(np.hypot(pa[0] - pb[0], pa[1] - pb[1])) if pa and pb else 99.0
        # an over-the-shoulder needs the pair CLOSE (the listener's shoulder in frame with the speaker's face
        # big enough); farther apart -> a single on the speaker
        if spk in who and len(listeners) == 1 and fr != "close-up" and spk in scene["marks"] and \
                listeners[0] in scene["marks"] and _gap(listeners[0], spk) <= 2.2:
            cam = {"ots": [listeners[0], spk], "line": pair_line(scene, listeners[0], spk), "lens": 50}
        elif spk in who:
            cam.update(framing="medium close-up" if fr != "close-up" else "close-up", target=spk)
        return cam
    carried = next((f for f in base["figures"] if f.get("attach")), None)
    if carried is not None:
        # a person CARRIED by a creature: frame the person, the claws and the head together, from below
        # (the body hangs under the forearms: a low camera sees it; a level one has the forearms in front)
        cr = carried["attach"].get("to")
        cam.update(framing="wide", target=[carried["who"], f"{cr}:claw_r", f"{cr}:claw_l", f"{cr}:head"])
        cam["height"] = 0.8
        return cam
    crs = [f["who"] for f in base["figures"] if f.get("kind") == "creature"]
    if crs and humans and fr in ("medium", "two-shot", "low angle"):
        # a fight at a medium: ONE person as the target -> the solver's over-the-shoulder up at the creature's head
        # (blockout.solve_camera); 10-23: a "wide" on both put the fighters far off, tiny on a chunk of the set
        cam.update(framing="medium" if fr == "two-shot" else fr, target=humans[0])
        return cam
    if crs and humans and fr in ("wide", "establishing", "high angle"):
        cam.update(target=humans + [f"{crs[0]}:head"])
        return cam
    if fr in ("close-up", "medium close-up") and who:
        cam["target"] = humans[0] if humans else who[0]
        holder = insert_of(frame.get("prompt") or "", base["figures"])
        if holder:              # "Close-up of Edric's hands holding the sword": the hands + the sword, not his face
            cam.update(framing="insert", target=holder)
    elif fr == "medium" and len(humans) >= 2:
        cam.update(framing="two-shot", target=humans)
    elif fr == "medium" and humans:
        cam["target"] = humans[0]
    return cam


def insert_of(prompt, figures):
    """A close-up whose SUBJECT is hands or a held thing ('Close-up of Edric's hands holding the sheathed sword') ->
    the holder's name (an "insert" framing: hands + held props), else None (10-21: framed on his face, no sword)."""
    m = re.match(r"\s*(?:extreme |tight )?close[- ]?up(?: shot)? (?:of|on) (.{0,100}?)(?:[.;]|$)", prompt, re.I)
    if not m:
        return None
    subj = m.group(1).lower()
    if re.search(r"\b(face|eyes?|head|expression|brow|lips|mouth)\b", subj):
        return None                                   # "Edric's face and the glowing sword": the face leads
    for f in figures:
        held = [str(v).lower() for v in (f.get("holds") or {}).values() if v]
        named = f["who"].lower().split()[-1] in subj
        if re.search(r"\b(hands?|fingers?|grip|fist)\b", subj) and (named or len(figures) == 1) or \
                any(h.split()[-1] in subj.split(" holding")[0] for h in held):
            return f["who"]
    return None


def pair_line(scene, a, b):
    """Any two people of the scene in screen order (left, right) as the scene's reference camera sees their
    marks: every shot of this pair keeps that side of THEIR line."""
    rc = scene["reference_camera"]
    pos, look = np.asarray(rc["pos"], float), np.asarray(rc["look"], float)
    f = look - pos
    f[2] = 0
    f = f / max(np.linalg.norm(f), 1e-9)
    right = np.array([f[1], -f[0], 0.0])
    x = lambda w: float(np.dot(np.array(list(scene["marks"][w]["at"])[:2] + [0.0]) - pos, right))
    return sorted([a, b], key=x)


def move_of(frame):
    txt = (frame.get("prompt", "") + " " + frame.get("motion", "")).lower()
    for rx, mv in MOVES:
        if re.search(rx, txt):
            return mv
    return None


def path_for(shot, mv, n_frames, size=(1152, 640)):
    """An authored camera path for a move: solve the start (the plan's framing) and the end (the move's target
    framing / transform) on the staged scene; keys at 0, mid, N-1 (eased by the renderer)."""
    import blockout as B
    os.environ.setdefault("BLOCKOUT_DRY", "1")
    st = B.stage_frames(shot, [("start", copy.deepcopy(shot["figures"]))])
    _, figs, _, _, info = st[0]
    c0 = B.solve_camera(dict(shot["camera"]), figs, info, size)
    cs = dict(shot["camera"])
    if mv == "push_in":
        cs["framing"] = TIGHTER.get(cs.get("framing", "medium"), "medium close-up")
        c1 = B.solve_camera(cs, figs, info, size)
    elif mv == "pull_back":
        cs["framing"] = WIDER.get(cs.get("framing", "medium"), "wide")
        cs.pop("target", None) if cs["framing"] == "wide" else None
        c1 = B.solve_camera(cs, figs, info, size)
    else:
        p, l = np.asarray(c0["pos"]), np.asarray(c0["look"])
        if mv in ("pan_left", "pan_right"):
            l = p + rig.rodrigues(l - p, rig.UP, 18 if mv == "pan_left" else -18)
        elif mv == "track":
            side = np.cross(rig._n(l - p), rig.UP)
            p, l = p + side * 1.5, l + side * 1.5
        elif mv == "crane_up":
            p = p + rig.UP * 1.5
            l = l + rig.UP * 0.6
        elif mv == "orbit":
            p = l + rig.rodrigues(p - l, rig.UP, 25)
        c1 = dict(c0, pos=p.tolist(), look=l.tolist())
    N = max(int(n_frames), 2)
    mid = {k: ((np.asarray(c0[k]) + np.asarray(c1[k])) / 2).tolist() for k in ("pos", "look")}
    return [{"frame": 0, "pos": c0["pos"], "look": c0["look"], "lens": c0["lens"]},
            {"frame": (N - 1) // 2, "pos": mid["pos"], "look": mid["look"], "lens": (c0["lens"] + c1["lens"]) / 2},
            {"frame": N - 1, "pos": c1["pos"], "look": c1["look"], "lens": c1["lens"]}]


AWAY = re.compile(r"\b(flees?|runs? away|turns? away|backs? away|retreats?|turns? (his|her|their) back)\b", re.I)


def confront_rule(shot, frame):
    """Fighters face each other (10-23: Edric stood turned 35-40 degrees away from the dragon in every shot of the
    fight, his facing a scene mark, and the keys drew him facing the viewer with the dragon at his back). Someone
    whose act strikes or blocks against a figure faces it; a person in a shot with a creature faces the creature
    unless their beats take them away from it; the creature faces the person its beats name (else the nearest)."""
    figs = {f["who"]: f for f in shot["figures"]}
    target = {}
    for a in frame.get("acts") or []:
        t = str(a.get("target") or a.get("against") or "").split(":")[0]
        if a.get("do") in ("strike", "block") and a.get("who") in figs and t in figs and t != a.get("who"):
            target[a["who"]] = t
    beats = [str(b) for b in frame.get("beats") or []]

    def own(w):
        return " ".join(b for b in beats if b.lower().startswith(w.lower() + ":"))
    cr = [f for f in shot["figures"] if f.get("kind") == "creature"]
    hu = [f for f in shot["figures"] if f.get("kind", "human") != "creature" and not f.get("attach")]
    for f in hu:
        if f["who"] in target:
            f["facing"] = target[f["who"]]
        elif cr and not AWAY.search(own(f["who"])):
            me = np.asarray(f.get("at", [0, 0]), float)
            f["facing"] = min(cr, key=lambda g: np.linalg.norm(np.asarray(g.get("at", [0, 0]), float) - me))["who"]
    for f in cr:
        named = [h["who"] for h in hu if h["who"].lower() in own(f["who"]).lower()]
        if f["who"] in target:
            f["facing"] = target[f["who"]]
        elif named:
            f["facing"] = named[0]
        elif hu and not isinstance(f.get("facing"), str):
            me = np.asarray(f.get("at", [0, 0]), float)
            f["facing"] = min(hu, key=lambda g: np.linalg.norm(np.asarray(g.get("at", [0, 0]), float) - me))["who"]


def shot_for(m, frame, scene, cast, chapter):
    import interaction as IA
    IA.carry_hints(frame)                          # an interaction program's carries attach before the placing
    base = BP.frame_to_shot(frame, cast=cast, chapter=chapter, t=m["t"])
    marks = scene["marks"]
    for f in base["figures"]:
        mk = marks.get(f["who"])
        if mk is None:
            continue
        f["at"] = copy.deepcopy(mk["at"])
        for k in ("ground_z", "z"):
            if k in mk and not f.get("attach"):
                f[k] = mk[k]
        if isinstance(f.get("facing"), str):
            pass                                   # faces a named figure: stays (resolved on the marks)
        else:
            f["facing"] = mk.get("facing", f.get("facing", 0))
    for f in base["figures"]:                      # a carried captive hangs from its creature's anchor
        ah = (frame.get("attach_hint") or {}).get(f["who"])
        if ah and any(g["who"] == ah["to"] for g in base["figures"]):
            f["attach"] = dict(ah)
            f.pop("ground_z", None)
            for g in base["figures"]:
                if g["who"] == ah["to"] and frame.get("pose_hint", {}).get(ah["to"]):
                    g["pose"] = frame["pose_hint"][ah["to"]]
                    g.setdefault("z", 3.5)
    present = {f["who"] for f in base["figures"]}
    if m["kind"] == "talk":                        # the speaker faces whoever they talk to
        spk = frame.get("speaker")
        others = [w for w in present if w != spk]
        for f in base["figures"]:
            if f["who"] == spk and others:
                f["facing"] = others[0]
            elif f["who"] in others and spk in present:
                f["facing"] = spk
    confront_rule(base, frame)                     # fighters face each other (after talk: an act wins)
    # the scene's set and its persistent props (not re-placed per shot)
    st = copy.deepcopy(scene["set"])
    st["props"] = [p for p in st.get("props", [])]
    base["set"] = st
    ledge_rule(base, frame)                         # a person the text puts ON a ledge stands up there (this shot)
    if frame.get("acts"):                           # the moment's INTERACTION PROGRAM (plan review, 10-22)
        base["_act_items"] = IA.compile_acts(base, frame, log=lambda ln: sys.stderr.write(ln + "\n"))
    else:
        operate_rule(base, frame)                   # a person working a jointed set prop stands at its handle
    base["camera"] = camera_for(m, frame, base, scene)
    n = max(9, int(round(float(m.get("dur", 2.0)) * FPS)))
    mv = move_of(frame) if m["kind"] == "action" else None
    if mv:
        try:
            base["camera"]["path"] = path_for(base, mv, n)
            base["camera"]["move"] = mv
        except Exception as e:                     # noqa: BLE001 - a path is a bonus; static is valid
            base.setdefault("plan_notes", {})["camera_path_error"] = str(e)[:200]
    base["scene"] = {"id": scene["id"], "axis": scene["axis"], "frames": n}
    base["_room"] = scene.get("room")
    base["_capture"] = scene.get("capture_xy", [0.0, 0.0])
    base["moment"] = {"id": m["id"], "kind": m["kind"], "t": m["t"], "dur": m.get("dur"),
                      "speaker": frame.get("speaker")}
    base["book"] = scene.get("book")
    return base


def on_open_ground(shot) -> list:
    """Each free figure whose mark lies BEYOND the pano set's surface (seen from the set's capture point at chest
    height: motion_camera.SetView.blocks) is pulled toward the capture point until clear (0.5 m steps, at most to
    1.5 m from it). -> [log lines]"""
    try:
        import motion_camera as MC
        sv = MC.SetView(shot)
    except Exception:                               # noqa: BLE001 - no set depth: nothing to test
        return []
    if sv.dep is None or sv.center is None:
        return []
    out = []
    c = np.asarray(sv.center, float)
    for f in shot.get("figures", []):
        if f.get("attach") or not f.get("at"):
            continue
        z = 1.2 if f.get("kind") != "creature" else 2.0
        at = np.array([float(f["at"][0]), float(f["at"][1])])
        p0 = np.array([at[0], at[1], z])
        start = np.array([c[0], c[1], z])
        if not sv.blocks(start, p0, stop=0.3):
            continue
        v = at - c[:2]
        L = float(np.linalg.norm(v))
        moved = None
        for d in np.arange(L - 0.5, 1.49, -0.5):
            q = c[:2] + v / max(L, 1e-6) * d
            if not sv.blocks(start, np.array([q[0], q[1], z]), stop=0.3):
                moved = q
                break
        if moved is None:
            moved = c[:2] + v / max(L, 1e-6) * 1.5
        f["at"] = [round(float(moved[0]), 3), round(float(moved[1]), 3)]
        out.append(f"{f.get('who')} moved {L - float(np.linalg.norm(moved - c[:2])):.1f} m in: its mark was inside the set")
    return out


def keep_absent(order, frames, shots, scene):
    """A person the moment names but the planner gave no pose box keeps their place and pose from the nearest shot
    of the scene that has them (earlier first), at their scene mark (10-21: the King on his throne, 'blurred in the
    background' of a close-up, had no box -> DEFAULT STAGING -> 2 moments not rendered)."""
    for i, mid in enumerate(order):
        shot = shots[mid]
        have = {f["who"] for f in shot["figures"]}
        for w in frames[mid].get("who") or []:
            if w in have or w not in scene["marks"]:
                continue
            near = [order[j] for j in range(i - 1, -1, -1)] + order[i + 1:]
            src = next((g for m2 in near for g in shots[m2]["figures"] if g["who"] == w), None)
            if src is None:
                continue
            g = {k: copy.deepcopy(v) for k, v in src.items() if k in ("who", "kind", "pose", "model", "height",
                                                                       "ground_z", "z", "outfit")}
            g["at"] = copy.deepcopy(scene["marks"][w]["at"])
            g["facing"] = copy.deepcopy(src.get("facing", scene["marks"][w].get("facing", 0)))
            shot["figures"].append(g)
            shot.setdefault("plan_notes", {}).setdefault("kept", []).append(w)
            sys.stderr.write(f"{mid}: {w} has no pose box; kept as in the scene ({g.get('pose')})\n")


# ------------------------------------------------------------------------------------------------ prop states
# generic support words (any book) + the words of every FIXED prop class (they are the set's surfaces)
SURFACES = tuple(dict.fromkeys(("bench", "table", "ground", "floor", "ledge", "rock", "stone", "wall", "shelf",
                                "chest", "altar", "cart", "saddle") + FIXED_WORDS))
HANG_ON = "|".join(sorted({"wall", "mantel", "beam", "hook", "peg"} | set(ARCH_WORDS), key=len, reverse=True))
PICK = r"\b(pick(s|ed)? up|lifts?|takes?|grabs?|grips?|holds?|raises?|draws?|seizes?|carr(y|ies)|wields?|brandish)"
PUT = r"\b(sets? (it |the \w+ )?down|lays?|plac(es|ed)|puts?|rests?|drops?|leaves?|lies|lying|hangs?|rests on)"
GIVE = r"\b(hands?|gives?|passes?|offers?)\b"


def _families():
    import props_build as PB
    return PB.classify


def prop_states(scene, moment_ids, frames, shots):
    """The MOVABLE props' state machine across one scene, shot to shot: location = held-by <who> <hand> |
    resting-on <surface> | carried, plus the material state ("heated", "cracked": the moment's prop list).
    Holders come from the planner's holds; the text's verbs give the transitions (picks up / sets down / hands
    to / drops); a prop nobody holds STAYS where it was (placed on its surface in the shot). A location change
    without a transition verb between consecutive shots is a TELEPORT (logged in shot["props_state"]["flags"]).
    Returns the scene's transition log."""
    classify = _families()
    cur, log = {}, []
    for mid in moment_ids:
        f, shot = frames[mid], shots[mid]
        txt = (f.get("prompt", "") + " " + f.get("motion", "") + " " + " ".join(f.get("must") or [])).lower()
        here = {}
        for pname in f.get("props") or []:
            cls, r = classify(pname)
            if cls == "physical" and r.get("mobility") in ("movable", "semi"):
                here.setdefault(r["base"], {"state": None})
                if r.get("state"):
                    here[r["base"]]["state"] = r["state"]
        # every held prop must be GROUNDED (named by the text / the moment's props, or a recipe); invented ones
        # (tongs in every hammer shot) are dropped and flagged
        rec_items = tuple(forge_recipe(shot, f, cur)) + tuple(shot.get("_act_items") or ())   # acts ground too
        ground = {}
        for fg in shot["figures"]:
            for hand, item in list((fg.get("holds") or {}).items()):
                g_ = grounded(item, txt, f.get("props"), rec_items)
                b_ = _families()(item)[1].get("base") or item
                if g_ is not None and item not in rec_items and re.search(
                        rf"\b{re.escape(b_)}\w*[^.]{{0,25}}\b(rests?|lies|lying|lay|hangs?|hung|sits?|leans?)\b", txt):
                    del fg["holds"][hand]                  # the text puts it DOWN (rests on the table): not held
                    log.append({"moment": mid, "dropped": item, "who": fg["who"], "why": "the text says it rests"})
                    continue
                if g_ is None:
                    del fg["holds"][hand]
                    log.append({"moment": mid, "dropped": item, "who": fg["who"], "why": "not in the text"})
                else:
                    ground[item] = g_
        held = {}
        for fg in shot["figures"]:
            for hand, item in (fg.get("holds") or {}).items():
                base = item if "__" in item else (classify(item)[1].get("base") or item)   # a registry sub-family
                held[base] = {"loc": "held", "who": fg["who"], "hand": hand}
        flags = []
        for base in sorted(set(here) | set(held) | set(cur)):
            prev = cur.get(base)
            new = dict(prev or {})
            if base in held:
                new.update(held[base])
            elif base in here and re.search(rf"{base}\w*[^.]{{0,60}}\b(hangs?|hung|hanging|mounted)\b[^.]{{0,40}}\b(above|over|on)"
                                            rf" (?:the |a )?(?:\w+ ){{0,2}}?({HANG_ON})", txt):
                hm = re.search(rf"\b(above|over|on) (?:the |a )?(?:\w+ ){{0,2}}?({HANG_ON})", txt)
                new.update(loc="hanging", on=hm.group(2) if hm else "wall", who=None, hand=None,
                           how=hm.group(1) if hm else "on")
            elif base in here:
                det = r"(?:the |a |an |his |her |their |its )?(?:\w+ ){0,2}?"
                sn = "|".join(SURFACES)
                m = re.search(rf"{base}\w*[^.]{{0,50}}\b(?:on|upon|across|atop|against) {det}({sn})\b", txt) or \
                    re.search(rf"\b(?:on|upon|across|atop) {det}({sn})\b[^.]{{0,40}}{base}", txt)
                surf = m.group(1) if m else None
                if surf or re.search(PUT, txt) or prev is None:
                    new.update(loc="resting", on=surf or (prev or {}).get("on"), who=None, hand=None)
                # else: nothing in the text moves it -> it stays exactly as it was (hanging stays hanging)
            if base in here and here[base]["state"]:
                new["state"] = here[base]["state"]
            if prev and (prev.get("loc"), prev.get("who"), prev.get("on")) != (new.get("loc"), new.get("who"), new.get("on")):
                verb = re.search(PICK, txt) or re.search(PUT, txt) or re.search(GIVE, txt)
                ev = {"moment": mid, "prop": base, "from": {k: prev.get(k) for k in ("loc", "who", "on")},
                      "to": {k: new.get(k) for k in ("loc", "who", "on")}, "verb": verb.group(0) if verb else None}
                log.append(ev)
                if not verb:
                    flags.append(f"{base} teleports {ev['from']} -> {ev['to']} (no transition in the text)")
            cur[base] = new
        # a prop resting on a surface in this scene and in this shot's text: placed there (the sword lies on the
        # anvil, not in a hand)
        st = shot.setdefault("set", {})
        tops = {(pr.get("prop") or pr.get("name") or "").lower(): pr for pr in st.get("props", [])}
        for base, v in cur.items():                     # a HANGING prop: above/on its fixed support (a door)
            if v.get("loc") == "hanging" and base in here:
                sup = tops.get(v.get("on"))
                if sup is None and v.get("on") and _is_arch(v["on"]):
                    sup = {"name": v["on"], "prop": v["on"], "box": [0, 0, 0, 1, 1, 1], "fixed": True}
                    st.setdefault("props", []).append(sup)
                    tops[v["on"]] = sup
                if sup is not None and sup.get("wall") is None and _is_arch(sup):
                    mount_arch(sup, scene, shot)
                if sup is not None and sup.get("wall"):
                    # FLUSH ON THE WALL, centred ~0.3 m above the opening's lintel, the blade along the wall
                    wl = sup["wall"]
                    c_ = np.array(wl["centre"], float)
                    t_ = np.array(wl["tangent"], float)
                    n_ = np.array(wl["normal"], float)
                    # FLUSH ON THE ACTUAL WALL SURFACE above the opening (an opening is often recessed: the wall
                    # above it stands proud of the door plane): the set's depth at the hang point decides
                    zh = wl["lintel_z"] + 0.3
                    off = wall_offset(scene, c_, n_, t_, zh)
                    p_ = c_ - t_ * 0.45 + n_ * (off + 0.03)
                    st["props"].append({"name": base, "prop": base,
                                        "box": [round(float(p_[0]), 3), round(float(p_[1]), 3),
                                                round(zh, 3), 1, 1, 1], "wall_offset_m": round(off, 3),
                                        "yaw": round(float(np.degrees(np.arctan2(t_[1], t_[0]))), 2), "roll": 0,
                                        "state": v.get("state"), "on_wall": True,
                                        "placed": f"hanging {v.get('how', 'above')} the {v.get('on')}, flush on its wall"})
                elif sup is not None:
                    x, y, z, w, d, h = sup["box"]
                    st["props"].append({"name": base, "prop": base, "box": [x + w / 2 - 0.45, y - 0.05, z + h + 0.25, 1, 1, 1],
                                        "yaw": sup.get("yaw", 0), "roll": 0, "state": v.get("state"),
                                        "placed": f"hanging {v.get('how', 'above')} the {v.get('on')}"})
        for base, v in cur.items():
            if shot.get("recipe"):
                break                                   # the recipe placed the work on the anvil
            if v.get("loc") == "resting" and base in here and v.get("on") and v["on"] in tops:
                sp = tops[v["on"]]
                x, y, z, w, d, h = sp["box"]
                st["props"].append({"name": base, "prop": base, "box": [x, y, z + h + 0.012, 1, 1, 1],
                                    "yaw": sp.get("yaw", 0), "roll": 90, "state": v.get("state"),
                                    "hot": v.get("state") in ("heated", "glowing", "flowing")})
        for pr in st.get("props", []):                  # NO free-standing architecture (a door alone in a room)
            if _is_arch(pr) and pr.get("wall") is None:
                mount_arch(pr, scene, shot)
        if not shot.get("figures"):
            prop_subject_camera(shot, scene, txt)
        elif (shot.get("camera") or {}).get("framing") == "close-up":
            nm_ = closeup_props(f.get("prompt", ""), st.get("props", []), [g["who"] for g in shot["figures"]])
            if nm_:             # "Close-up of the anvil. The hammer head has just struck": the anvil, not a face
                prop_subject_camera(shot, scene, " ".join(nm_), only=nm_)
        arch_wall_check(shot, scene, mid)
        shot["props_state"] = {"props": copy.deepcopy(cur), "flags": flags + shot.pop("_arch_flags", []),
                               "grounded": ground}
    return log


# ------------------------------------------------------------------------------------------------ architecture
def _arch_rule(pr):
    """The prop's class rule when it is ARCHITECTURAL (prop_classes.json "architectural": door, window, gate,
    hearth, shutters ...), else None."""
    nm = pr if isinstance(pr, str) else (pr.get("prop") or pr.get("name") or "")
    cls, r = _families()(nm.lower())
    return r if cls == "physical" and r.get("architectural") else None


def _is_arch(pr):
    return _arch_rule(pr) is not None


def _arch_word(pr):
    return _arch_rule(pr)["base"]




def mount_arch(pr, scene, shot):
    """An ARCHITECTURAL prop (door, window, gate, hearth, shutters) is part of a wall, never free-standing:
    (1) the set's OWN one (world.json "anchors", tools/world_locate.py --anchors): that opening, hidden (the world
    draws it); (2) else flush in the nearest room wall plane (an interior, world.json room walls); (3) else
    (exterior / no walls) it brings a facade: a building front >= 3.2 m wide around it. Sets pr["wall"] =
    {centre (bottom centre on the wall), tangent, normal (into the room), width, height, lintel_z, source}."""
    w_ = _arch_word(pr)
    W0, H0 = _ARCH_SIZE.get(w_, (1.0, 2.1))
    anc = (scene.get("anchors") or {}).get(w_) or (scene.get("anchors") or {}).get(pr.get("name") or "")
    figs = [f_["at"][:2] for f_ in shot.get("figures", []) if f_.get("at")]
    near_xy = np.mean(figs, axis=0) if figs else np.array(scene.get("capture_xy") or [0.0, 0.0], float)
    if pr.get("anchored") == "world":                  # the world's own one (no wall anchor yet): stays where it is
        near_xy = np.array(pr["box"][:2], float)
    if anc:
        c_, t_, n_ = np.array(anc["at"][:2], float), np.array(anc["tangent"][:2], float), np.array(anc["normal"][:2], float)
        W_, H_, src = float(anc["width"]), float(anc["lintel_z"]), "the set's own (world anchor)"
        pr["hidden"] = True
    else:
        walls = ((scene.get("room") or {}).get("walls") or []) if scene.get("world_kind") != "exterior" else []
        W_, H_ = W0, H0
        if walls:
            best = min(walls, key=lambda wl: abs(float(np.dot(wl["n"][:2], near_xy) + wl["d"])))
            n_ = np.array(best["n"][:2], float)
            c_ = near_xy - n_ * (float(np.dot(n_, near_xy)) + best["d"])        # the foot of the perpendicular
            t_ = np.array([n_[1], -n_[0]])
            src = "flush in the nearest room wall"
        else:
            # exterior: in front of the people (beyond them, away from the capture point) with a facade
            cap = np.array(scene.get("capture_xy") or [0.0, 0.0], float)
            n_ = cap - near_xy
            n_ = n_ / max(np.linalg.norm(n_), 1e-9)
            c_ = near_xy - n_ * 2.5
            t_ = np.array([n_[1], -n_[0]])
            src = "with a facade (exterior)"
            yaw_f = float(np.degrees(np.arctan2(t_[1], t_[0])))
            back = c_ - n_ * 0.3
            fw = max(3.2, W_ + 2.2)
            shot["set"].setdefault("props", []).append(
                {"name": f"{w_} facade", "prop": "facade", "facade_of": pr.get("name"), "fixed": True,
                 "box": [round(float(back[0] - t_[0] * fw / 2), 3), round(float(back[1] - t_[1] * fw / 2), 3), 0.0,
                         fw, 0.3, max(3.2, H_ + 1.0)], "yaw": round(yaw_f, 2),
                 "wall_plane": {"centre": back.round(3).tolist(), "tangent": t_.round(4).tolist(),
                                "normal": n_.round(4).tolist(), "width": fw, "height": max(3.2, H_ + 1.0)}})
    # the door library leaf: hinge at its origin, the leaf along +Y (d = width), thin in X, z: 2*h tall
    hinge = c_ - t_ * W_ / 2
    yaw = float(np.degrees(np.arctan2(t_[1], t_[0]))) - 90.0
    pr["box"] = [round(float(hinge[0]), 3), round(float(hinge[1]), 3), 0.0, 1.0, round(W_, 3), round(H_ / 2, 3)]
    pr["yaw"] = round(yaw, 2)
    pr["fixed"] = True
    pr["wall"] = {"centre": c_.round(3).tolist(), "tangent": t_.round(4).tolist(), "normal": n_.round(4).tolist(),
                  "width": round(W_, 3), "height": round(H_, 3), "lintel_z": round(H_, 3), "source": src}
    pr["anchored"] = src


_SUBJ_COVER = {"extreme close-up": 0.5, "close-up": 1.1, "medium close-up": 1.6, "medium": 2.4, "two-shot": 2.6,
               "wide": 5.0}


def closeup_props(prompt, props, people=()):
    """The set props a close-up is ABOUT ('Close-up of the anvil.'), [] when its subject is a face or hands (those
    are the person's close-up / an insert) or names a person ('Edric's upper body and the bellows': he leads)."""
    m = re.match(r"\s*(?:extreme |tight )?close[- ]?up(?: shot)? (?:of|on) (.{0,100}?)(?:[.;]|$)", prompt or "", re.I)
    if not m or re.search(r"\b(face|eyes?|head|expression|hands?|fingers?|grip|fist)\b", m.group(1).lower()):
        return []
    subj = m.group(1).lower()
    if any(re.search(rf"\b{re.escape(w.lower().split()[-1])}\b", subj) for w in people if w):
        return []
    return [n for n in dict.fromkeys(str(p.get("prop") or p.get("name") or "") for p in props if not p.get("hidden"))
            if n and n != "box" and re.search(rf"\b{re.escape(n.lower())}", subj)]


def prop_subject_camera(shot, scene, txt="", only=None):
    """A shot with NO people is about its PROPS (a sword hanging above a door, a letter on a table): the camera
    frames the placed props the text names (else every placed movable prop) with their support's top edge, from
    in front of the wall they hang on (else from the set's capture side), at the framing's coverage, inside the
    room. Writes an explicit camera {pos, look, lens} (keeps framing)."""
    st = shot.get("set") or {}
    placed = [p for p in st.get("props", []) if p.get("placed") and not p.get("hidden")]
    named = [p for p in placed if re.search(rf"\b{re.escape((p.get('prop') or p.get('name') or '').lower())}", txt)]
    subj = named or placed
    if only:                    # a close-up with people of a named set prop (fixed ones too: the anvil)
        subj = [p for p in st.get("props", []) if (p.get("prop") or p.get("name")) in only and not p.get("hidden")]
    if not subj:
        return
    import blockout_props as PR
    P, normals = [], []
    for p in subj:
        try:
            for pr_ in PR.set_prims(p):
                for k in ("c", "a", "b"):
                    if k in pr_:
                        P.append(np.asarray(pr_[k], float))
        except Exception:                               # noqa: BLE001
            P.append(np.asarray(p["box"][:3], float))
        sup = next((q for q in st.get("props", []) if q.get("wall") and p.get("on_wall")), None)
        if sup:
            wl = sup["wall"]
            c_, t_ = np.array(wl["centre"] + [0.0])[:3], np.array(wl["tangent"] + [0.0])[:3]
            for u in (-0.5, 0.5):
                P.append(c_ + t_ * u * wl["width"] + np.array([0, 0, wl["lintel_z"]]))
            normals.append(np.array(wl["normal"][:2], float))
    P = np.array(P)
    look = (P.min(0) + P.max(0)) / 2
    fr = (shot.get("camera") or {}).get("framing", "medium")
    cover = max(_SUBJ_COVER.get(fr, 2.4), float(np.ptp(P[:, 2])) * 1.3, float(np.ptp(P[:, :2], axis=0).max()) * 0.75)
    lens = 35.0
    vfov = 2 * math.atan((36.0 * 9 / 16) / 2 / lens)
    dist = cover / 2 / math.tan(vfov / 2)
    if normals:
        dxy = np.mean(normals, axis=0)
    else:
        dxy = np.array(scene.get("capture_xy") or [0.0, 0.0], float) - look[:2]
    dxy = dxy / max(np.linalg.norm(dxy), 1e-9)
    pos = np.array([*(look[:2] + dxy * dist), max(1.4, look[2] - 0.3 * dist)])
    room = scene.get("room")
    if room:
        import world_room as WR
        while dist > 0.6 and not WR.inside_room(room, pos.tolist(), margin=0.2):
            dist *= 0.85                                # stay inside the room: closer, wider
            lens = max(20.0, lens * 0.88)
            pos = np.array([*(look[:2] + dxy * dist), max(1.4, look[2] - 0.3 * dist)])
    cam_ = {k: v for k, v in (shot.get("camera") or {}).items() if k not in ("path", "move")}   # the old move aimed elsewhere
    shot["camera"] = dict(cam_, pos=pos.round(3).tolist(), look=look.round(3).tolist(),
                          lens=round(lens, 1), aim="props", subject=[p.get("name") for p in subj])


def wall_offset(scene, c_, n_, t_, z, half=0.5):
    """How far the set's own surface stands IN FRONT of a wall plane (along its normal n_) around the point
    c_ + up*z, over +-half m along the wall: the 90th percentile of the set points there (0 without a set)."""
    if not scene.get("world_dir"):
        return 0.0
    tree = _cloud(scene["world_dir"])
    P = tree.data
    c3 = np.array([c_[0], c_[1], z])
    idx = tree.query_ball_point(c3, r=max(half, 0.6))
    if not idx:
        return 0.0
    Q = P[idx] - c3
    n3, t3 = np.array([n_[0], n_[1], 0.0]), np.array([t_[0], t_[1], 0.0])
    a_ = Q @ t3
    h_ = Q[:, 2]
    o_ = Q @ n3
    sel = (np.abs(a_) < half) & (np.abs(h_) < 0.25) & (o_ > -0.3) & (o_ < 1.0)
    return float(np.percentile(o_[sel], 90)) if sel.sum() > 10 else 0.0


_CLOUD = {}


def _cloud(world_dir):
    if world_dir not in _CLOUD:
        from scipy.spatial import cKDTree
        P = np.load(Path(world_dir) / "points.npy").reshape(-1, 3)
        P = P[np.isfinite(P).all(1)]
        _CLOUD[world_dir] = cKDTree(P)
    return _CLOUD[world_dir]


def _shot_cam(shot, scene, mid):
    cam = shot.get("camera") or {}
    if cam.get("pos"):
        return np.array(cam["pos"], float)
    return np.array(scene["reference_camera"]["pos"], float)


def arch_wall_check(shot, scene, mid, behind=0.05, need=0.8):
    """VALIDATOR: an architectural prop must have a WALL within 5 cm behind it across >= 80 % of its outline, seen
    from the shot camera (the set's own depth: the world's 3D points; or the facade it brought). Writes
    shot["arch_check"] = [{prop, covered, ok}] and flags failures."""
    out = []
    for pr in shot.get("set", {}).get("props", []):
        if not _is_arch(pr):
            continue
        wl = pr.get("wall")
        if not wl:
            out.append({"prop": pr.get("name"), "covered": 0.0, "ok": False, "why": "free-standing"})
            continue
        cam = _shot_cam(shot, scene, mid)
        c_, t_ = np.array(wl["centre"] + [0.0])[:3], np.array(wl["tangent"] + [0.0])[:3]
        S = [c_ + t_ * (u - 0.5) * wl["width"] + np.array([0, 0, 1.0]) * v * wl["height"]
             for u in np.linspace(0.02, 0.98, 9) for v in np.linspace(0.03, 0.97, 11)]
        S = np.array(S)
        fac = next((q for q in shot["set"]["props"] if q.get("facade_of") == pr.get("name")), None)
        hit = np.zeros(len(S), bool)
        if fac:
            fp = fac["wall_plane"]
            fc, ft = np.array(fp["centre"] + [0.0])[:3], np.array(fp["tangent"] + [0.0])[:3]
            a_ = (S - fc) @ ft
            hit |= (np.abs(a_) <= fp["width"] / 2) & (S[:, 2] <= fp["height"])
        if scene.get("world_dir") and not hit.all():
            tree = _cloud(scene["world_dir"])
            d_ = S - cam
            d_ = d_ / np.linalg.norm(d_, axis=1, keepdims=True)
            for k in np.where(~hit)[0]:
                probe = S[k] + d_[k] * np.linspace(-0.01, behind, 4)[:, None]
                if np.min(tree.query(probe, distance_upper_bound=0.04)[0]) < 0.04:
                    hit[k] = True
        cov = float(hit.mean())
        out.append({"prop": pr.get("name"), "covered": round(cov, 3), "ok": cov >= need, "source": wl["source"]})
        if cov < need:
            shot.setdefault("_arch_flags", []).append(
                f"{pr.get('name')}: only {cov:.0%} of its outline has a wall within {behind * 100:.0f} cm behind")
    shot["arch_check"] = out
    return out


# ------------------------------------------------------------------------------------------------ recipes
CONTACT_ACTIONS = {k: v for k, v in json.loads((HERE / "stage_templates" / "contact_actions.json").read_text()).items()
                   if not k.startswith("_")}


def _class_rule(base):
    import props_build as PB
    return next((r for r in PB.CLASSES["physical"] if r["base"] == base), {})


def grounded(item, txt, props, recipe_items=()):
    """Where a prop comes from: 'text: <the sentence>' / 'props: <name>' / 'recipe' / None (invented)."""
    classify = _families()
    base = item if "__" in item else (classify(item)[1].get("base") or item)
    words = [base.split("__")[0]] + [w for w in (classify(item)[1].get("words") or [])]
    for sent in re.split(r"(?<=[.!?])\s+", txt):
        if any(re.search(rf"\b{re.escape(w)}", sent.lower()) for w in words):
            return f"text: {sent[:120]}"
    for pn in props or []:
        if classify(pn)[1].get("base") == base:
            return f"props: {pn}"
    if item in recipe_items:
        return "recipe"
    return None


def forge_recipe(shot, frame, state):
    """(kept name) every contact-action template, first that applies (stage_templates/contact_actions.json)."""
    for tname in CONTACT_ACTIONS:
        r_ = contact_action(shot, frame, state, tname)
        if r_:
            return r_
    return ()


def contact_action(shot, frame, state, tname="strike_on_support"):
    """A generic STRIKE-ON-WORK-ON-SUPPORT contact action (approved 2026-10-19 as the forging strike,
    eval/blockout/shots/hammer_forge_sword.json), every role found by affordance / class flag (see
    contact_actions.json): the striker hand on the striker tool; the WORK (the moment's own movable prop, else what
    rests on the support by the prop state machine; in its class's work_state when it has one) lies flat on the
    support's top; the off hand holds the work's handle when it has one, else stays clear; the striker's face ENDS
    flush on top of the work (a contact, never through it); the support class's impact_fx at the impact. An
    offhand tool only when the text names it. Returns the items used (or ())."""
    import blockout_props as PR
    T = CONTACT_ACTIONS[tname]
    acts_ = frame.get("acts") or []
    if acts_ and not any(a.get("do") == "strike" and ":" not in str(a.get("target") or "") for a in acts_):
        return ()                                      # DIRECTED and no strike on a support: the director's hands
        # stand (10-22: a hammer pose made the template re-arm Edric while he turns the finished sword in his hands)
    classify = _families()
    lib = PR.library()
    st = shot.get("set") or {}
    hand, off = T["striker"]["hand"], T["offhand_hand"]
    poses = T["poses"]
    act_poses = (poses["raise"], poses["strike"])

    def rule_of(pr):
        return classify(pr.get("prop") or pr.get("name") or "")[1]
    anvil = next((p for p in st.get("props", []) if rule_of(p).get(T["support"]["class_flag"])), None)
    strikers = {k for k, e in lib.items() if T["striker"]["affordance"] in (e.get("affordances") or {})}
    smith = next((f for f in shot["figures"] if f.get("pose") in act_poses or
                  (f.get("holds") or {}).get(hand) in strikers), None)
    txt0 = (frame.get("prompt", "") + " " + frame.get("motion", "")).lower()
    people = [f for f in shot["figures"] if f.get("kind") != "creature" and not f.get("attach")]
    if smith is None and anvil is not None and len(people) == 1 and not frame.get("acts"):
        # the TEXT strikes with a striker tool and nobody is posed for it (10-22: "Close-up of the anvil. The hammer
        # head strikes the bar" staged the anvil alone, Edric standing empty-handed): the one person strikes
        named_ = [k for k in sorted(strikers) if re.search(rf"\b{re.escape(k)}", txt0)
                  and re.search(T["strike_rx"].format(striker=re.escape(k)), txt0)]
        if named_:
            smith = people[0]
            smith.setdefault("holds", {})[hand] = named_[0]
            smith["pose"] = poses["strike"]
            shot.setdefault("plan_notes", {})["striker_from_text"] = f"{smith['who']} strikes with the {named_[0]}"
    if anvil is None or smith is None:
        return ()
    sup_rule = rule_of(anvil)
    sup_name = (anvil.get("prop") or anvil.get("name") or "support").lower()
    txt = (frame.get("prompt", "") + " " + frame.get("motion", "")).lower()
    tool = (smith.get("holds") or {}).get(hand)
    if tool not in strikers:                           # posed for it: the striker the text names, else any
        named = [k for k in sorted(strikers) if re.search(rf"\b{re.escape(k)}", txt)]
        tool = (named or sorted(strikers))[0]
    import props_build as PB
    offhand_tools = {r["base"] for r in PB.CLASSES["physical"] if r.get("offhand_for") == tname}
    # the TEXT decides the mode: no striker in hand when it rests/lies; a HOLD (raised, poised, preparing) has no
    # impact; a STRIKE otherwise
    if re.search(T["rest_rx"].format(striker=re.escape(tool)), txt):
        smith.get("holds", {}).pop(hand, None)
        st = shot.get("set") or {}
        x_, y_ = smith.get("at", [anvil["box"][0], anvil["box"][1]])[:2]
        st.setdefault("props", []).append({"name": tool, "prop": tool, "box": [x_ + 0.5, y_ - 0.2, 0.02, 1, 1, 1],
                                           "roll": 90, "placed": "on the ground (the text)"})
        if smith.get("pose") in act_poses:
            smith["pose"] = poses["rest"]
        return ()
    hold = bool(re.search(T["hold_rx"].format(striker=re.escape(tool)), txt)) and not re.search(
        T["strike_rx"].format(striker=re.escape(tool)), txt)
    work = None
    for pn in frame.get("props") or []:                # the moment's OWN work first (the ingot became the blade)
        cls_, r_ = classify(pn)
        if cls_ == "physical" and r_.get("mobility") == "movable" and r_.get("base") not in strikers | offhand_tools:
            work = r_["base"]
            break
    if work is None:                                   # else whatever rests on the support by the state machine
        work = next((b for b, v in state.items() if v.get("loc") == "resting" and v.get("on") == sup_name), None)
    pre = (smith.get("holds") or {}).get(off)       # the DIRECTOR gave the off hand the work piece (a grasp act):
    if pre and pre not in strikers | offhand_tools and pre != tool:     # that is the work (10-22: the moment's
        base_ = pre.split("__")[0]                     # "Star-iron, cracked" left the left hand empty over an ingot)
        if base_ != work:
            shot.setdefault("plan_notes", {})["work_from_direction"] = f"{work} -> {base_} (held by the {off} hand)"
        work = base_
    if work is None:
        return ()
    ws_ = _class_rule(work).get("work_state") or {}
    # the work's REGISTRY name: a family's forging state is its own sub-family "<base>__<state>" (props.json key)
    lib_item = ws_.get("library") or ws_.get("name") or work
    item = f"{work}__{re.sub(r'[^a-z0-9]+', '_', ws_['name'].lower()).strip('_')}" if ws_ else work
    if item not in lib and lib_item in lib:
        lib[item] = lib[lib_item]                       # (the book registry is not loaded here: its library entry)
    x, y, z, w, d, h = anvil["box"]
    top = z + h
    yaw = math.radians(anvil.get("yaw", 0))
    ax_ = [math.cos(yaw), math.sin(yaw), 0.0]
    # the smith at the anvil's working side (the side facing the camera's half-plane), facing the anvil
    n_ = np.array([-math.sin(yaw), math.cos(yaw)])
    # the worker stands on the support's FAR side from the set's capture point, facing it: the camera (near the
    # capture point, where a pano set holds up) then sees the worker's front OVER the support, the contact in
    # front of the body, never blocked by it
    cap0 = np.array(shot.get("_capture") or [0.0, 0.0], float)
    side = 1.0 if float(np.dot(n_, np.array([x, y]) - cap0)) > 0 else -1.0
    smith["at"] = [round(x + n_[0] * 0.58 * side, 3), round(y + n_[1] * 0.58 * side, 3)]
    smith["facing"] = math.degrees(math.atan2(y - smith["at"][1], x - smith["at"][0]))
    smith["pose"] = poses["raise"]                     # the swing: raised at the start, the strike at the end
    holds = {hand: tool}
    named_off = [b for b in sorted(offhand_tools) if re.search(rf"\b{re.escape(b)}\b", txt)]
    if named_off:
        holds[off] = named_off[0]
    elif item in lib and PR.has_grip(item) and off not in (smith.get("free_hands") or []):
        holds[off] = item                              # (unless the director left that hand free)
    smith["holds"] = holds
    proxy = {"name": f"{item} on the {sup_name} (contact proxy)", "prop": "box", "box": [x + 0.03, y, top, 0.5, 0.05, 0.012],
             "hidden": True}
    st["props"] = [p for p in st.get("props", []) if (p.get("prop") or p.get("name")) not in (work, item)] + [proxy]
    if holds.get(off) != item:                      # the work lies on the anvil by itself
        st["props"].append({"name": item, "prop": item, "box": [x + 0.03, y, top + 0.012, 1, 1, 1],
                            "yaw": anvil.get("yaw", 0), "roll": 90, "hot": bool(sup_rule.get("work_hot")), "state": state.get(work, {}).get("state")})
    cts = [] if hold else [{"who": smith["who"], "prop": tool, "point": T["striker"]["affordance"], "to": proxy["name"],
                            "at": [round(x, 3), round(y, 3)], "frame": "end"}]
    if holds.get(off) == item and "flat" in ((lib.get(item) or {}).get("affordances") or {}):
        # the tang on the smith's LEFT (his left hand holds it), the tip toward his right, across the anvil face
        fr_ = math.radians(smith["facing"])
        left_ = np.array([-math.sin(fr_), math.cos(fr_)])
        tip_dir = [round(float(-left_[0]), 4), round(float(-left_[1]), 4), 0.0]
        cts.append({"who": smith["who"], "prop": item, "point": "flat", "to": anvil.get("name") or sup_name,
                    "at": [round(x + 0.03 * tip_dir[0], 3), round(y + 0.03 * tip_dir[1], 3)], "axis": tip_dir,
                    "frame": "both", "phase": "hold"})
    shot["contacts"] = cts
    fx_ = sup_rule.get("impact_fx")
    shot["fx"] = [] if hold or not fx_ else [dict(fx_, at={"contact": 0}, frame="end")]
    if hold:
        shot["impact"] = "none"                         # a HOLD: nothing may engage an end contact
    shot.setdefault("end", {}).setdefault("figures", [])
    if not any(e.get("who") == smith["who"] for e in shot["end"]["figures"]):
        shot["end"]["figures"].append({"who": smith["who"], "pose": poses["raise"] if hold else poses["strike"]})
    else:
        for e in shot["end"]["figures"]:
            if e.get("who") == smith["who"]:
                e["pose"] = poses["raise"] if hold else poses["strike"]
                e["holds"] = holds
    # the camera AIMS AT THE CONTACT (the hammer face on the work on the anvil): a side view across the
    # smith->anvil line, chest height, the contact in the lower-middle of the frame and unoccluded; on the side
    # nearer the set's capture point (a pano set holds up only near where it was captured)
    c_ = np.array([x, y, top + 0.02])
    s_ = np.array(smith["at"][:2] + [0.0])
    u_ = c_[:2] - s_[:2]
    u_ = u_ / max(np.linalg.norm(u_), 1e-9)
    nrm = np.array([-u_[1], u_[0]])
    mid = (c_[:2] + s_[:2]) / 2
    cap = np.array(shot.get("_capture") or [0.0, 0.0], float)
    room = shot.get("_room")
    best = None
    for k in range(12):                                 # candidates around the contact; inside the room, near the
        a_ = 2 * math.pi * k / 12                       # pano's capture point, the smith not in the line of sight
        for r_ in (2.4, 3.0, 3.8, 4.6):                 # farther + longer lens = nearer the capture point (no streaks)
            cxy = c_[:2] + np.array([math.cos(a_), math.sin(a_)]) * r_
            if room:
                import world_room as WR
                if not WR.inside_room(room, [float(cxy[0]), float(cxy[1]), 1.35], margin=0.3):
                    continue
            ray = c_[:2] - cxy
            t_ = float(np.clip(np.dot(s_[:2] - cxy, ray) / max(np.dot(ray, ray), 1e-9), 0, 1))
            if t_ < 0.95 and np.linalg.norm(s_[:2] - (cxy + t_ * ray)) < 0.35:
                continue                                 # the smith's body between the camera and the contact
            side_ok = abs(float(np.dot(ray / max(np.linalg.norm(ray), 1e-9), u_)))   # 0 = a clean side view
            score = float(np.linalg.norm(cxy - cap)) + 1.5 * side_ok
            if best is None or score < best[0]:
                best = (score, cxy, r_)
    pos = np.array([*(best[1] if best else mid + nrm * 3.0), 1.35])
    # frame from below the support top to above the worker's head: the contact lands in the upper-middle band
    head_z = 1.85
    z_lo, z_hi = max(0.0, top - 0.95), head_z + 0.2       # the support down to the floor: contact in the upper 2/3
    look = np.array([*(c_[:2] * 0.6 + s_[:2] * 0.4), (z_lo + z_hi) / 2])
    dist_ = float(np.linalg.norm(look[:2] - pos[:2]))
    lens_ = round(float(np.clip((36.0 * 9 / 16 / 2) * dist_ / ((z_hi - z_lo) / 2), 18, 85)))
    shot["camera"] = {"pos": pos.round(3).tolist(), "look": look.round(3).tolist(), "lens": lens_,
                      "framing": "medium", "aim": "contact", "contact_point": c_.round(3).tolist()}
    shot["recipe"] = {"name": tname, "mode": "hold" if hold else "strike", "work": work, "item": item,
                      "tool": tool, "support": sup_name, "left": holds.get(off), "offhand": holds.get(off)}
    return (item, tool)


# ------------------------------------------------------------------------------------------------ sheets
def screen_order(shot, size=(1152, 640)):
    """Who is screen-left .. screen-right in a shot's start frame (the continuity overlay)."""
    import blockout as B
    os.environ["BLOCKOUT_DRY"] = "1"
    st = B.stage_frames(shot, [("start", copy.deepcopy(shot["figures"]))])
    _, figs, _, _, info = st[0]
    cs = dict(shot["camera"])
    c = B.solve_camera(cs, figs, info, size)
    if cs.get("path"):
        p0 = cs["path"][0]
        c = dict(c, pos=p0["pos"], look=p0["look"], lens=p0["lens"], R=rig.look_at(np.asarray(p0["pos"]),
                                                                                   np.asarray(p0["look"])).tolist())
    xs = []
    for w, v in info.items():
        if w.startswith("_"):
            continue
        pts = v.get("joints") or {}
        p = pts.get("head") if isinstance(pts, dict) and "head" in pts else np.mean(np.array(v["points"]), 0)
        q = rig.project(c, [np.asarray(p, float)], *size)[0]
        if q is not None and -50 <= q[0] <= size[0] + 50:
            xs.append((float(q[0]), w))
    return [w for _, w in sorted(xs)]


def sheet(scene, shot_dir, out_jpg, render=True):
    """Contact sheet: every shot's staged start frame (tools/blockout.py clay render), who is left/right."""
    import blockout as B
    from PIL import Image, ImageDraw
    tiles = []
    for mid in scene["moments"]:
        sp = shot_dir / f"{mid}.json"
        shot = json.loads(sp.read_text())
        try:
            order = screen_order(shot)
        except Exception as exc:                       # noqa: BLE001 - one shot's sheet tile never stops the
            print(f"!!! sheet {mid}: {exc}", file=sys.stderr)       # chapter's scene layer (10-22 overnight)
            order = []
        img = None
        if render:
            os.environ.pop("BLOCKOUT_DRY", None)
            od = shot_dir.parent / "renders" / mid
            try:
                B.run(json.loads(json.dumps(shot)), str(od), size=(640, 360), samples=8)
                img = Image.open(od / "start_clay.png").convert("RGB")
            except Exception as e:                 # noqa: BLE001
                img = None
                sys.stderr.write(f"{mid}: render failed {e}\n")
        tiles.append((mid, shot, order, img))
    W, H = 640, 360
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    out = Image.new("RGB", (W * cols, (H + 46) * rows + 40), (22, 22, 22))
    dr = ImageDraw.Draw(out)
    dr.text((8, 10), f"scene {scene['id']} - {scene['where']} - axis (screen L->R): {' | '.join(scene['axis'])}",
            fill=(255, 255, 255))
    for i, (mid, shot, order, img) in enumerate(tiles):
        x, y = (i % cols) * W, 40 + (i // cols) * (H + 46)
        if img is not None:
            out.paste(img.resize((W, H)), (x, y))
        cam = shot["camera"]
        kind = "OTS " + ">".join(cam["ots"]) if cam.get("ots") else cam.get("framing", "?")
        if cam.get("move"):
            kind += f" +{cam['move']}"
        bad = len(order) >= 2 and len(scene["axis"]) == 2 and all(a in order for a in scene["axis"]) and \
            order.index(scene["axis"][0]) > order.index(scene["axis"][1])
        dr.text((x + 6, y + H + 4), f"{mid} [{shot['moment']['kind']}] {kind}", fill=(255, 255, 0))
        dr.text((x + 6, y + H + 22), "L->R: " + " | ".join(order), fill=(255, 80, 80) if bad else (150, 255, 150))
    out.save(out_jpg, quality=85)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("book")
    ap.add_argument("chapter")
    ap.add_argument("--extra-lines", nargs="*", default=None, help="t values of quoted lines that need talk moments")
    ap.add_argument("--sheets", action="store_true")
    ap.add_argument("--no-render", action="store_true", help="sheets without clay renders (overlays only)")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args(argv)
    vd, cd, man, kf, plan, tl, cast = load(a.book, a.chapter)
    import blockout_props as _PRL
    _PRL.load_book(a.book)                           # the book's modelled props (joints + handles: operate_rule)
    if a.extra_lines is not None:
        extra = extra_moments(cd, man, tl, plan, a.extra_lines)
    else:
        ep = cd / "stage" / "extra_moments.json"
        extra = json.loads(ep.read_text()) if ep.exists() else []
    moments = [m for m in man] + extra
    frames = {m["id"]: moment_frame(m, kf, plan, tl) for m in moments}
    cpp = vd / "stage" / "character_props.json"          # book data: props that are really characters
    cprops = {k: v for k, v in json.loads(cpp.read_text()).items() if not k.startswith("_")} if cpp.exists() else {}
    for f in frames.values():
        for pn in f.get("props") or []:
            cp = cprops.get(pn)
            if not cp:
                continue
            f.setdefault("who", [])
            if cp["who"] not in f["who"]:
                f["who"].append(cp["who"])
            if cp.get("pose"):
                f.setdefault("pose_hint", {})[cp["who"]] = cp["pose"]
            if cp.get("attach"):
                f.setdefault("attach_hint", {})[cp["who"]] = cp["attach"]
    canon_names(frames, cast)
    sys.path.insert(0, str(HERE.parent))
    from tts_reader import video as _V
    plan["scenes"] = _V.merge_scenes(plan.get("scenes") or [])        # (a plan from before the merge rule)
    starts = [(sc_["t0"], sc_) for sc_ in plan.get("scenes") or [] if sc_.get("t0") is not None]
    chars = (plan.get("look") or {}).get("characters") or {}
    ap_ = vd / "stage" / "assets.json"
    import flow3d as _F
    kp_ = _F.known_places(a.book)                    # a built set's side wins over the look's words
    places_ = {**kp_, **((plan.get("look") or {}).get("places") or {}), **{k: v for k, v in kp_.items() if v}}
    for m in moments:                                 # the plan's scenes rule every moment (10-22): place, cast
        sc_ = _V.scene_at_t(starts, m["t"] + 0.5 * float(m.get("dur") or 0)) if starts else None   # its middle
        if not sc_:
            continue
        f = frames[m["id"]]
        for ln in _V.apply_scene(f, sc_, chars, places_):
            sys.stderr.write(f"{m['id']}: {ln} (scene {sc_['id']})\n")
        # (who is IN each picture is the plan review's call, moment by moment against the text: a scene's "present"
        # list is too coarse - "Wide shot of the empty stone hall" would have got the King, 10-22)
    for mid, was, now in rehome(frames, [m["id"] for m in sorted(moments, key=lambda m: m["t"])]):
        sys.stderr.write(f"{mid}: where {was!r} -> {now!r} (its fixed props live there)\n")
    scenes = scenes_of(moments, frames)
    sd, shd = cd / "stage" / "scenes", cd / "stage" / "shots"
    sd.mkdir(parents=True, exist_ok=True)
    shd.mkdir(parents=True, exist_ok=True)
    report = []
    layouts = {}
    for sc in scenes:
        scene = block_scene(sc, moments, frames, cast, a.chapter, prior=layouts.get(sc["where"]))
        layouts[sc["where"]] = scene               # the place's layout grows scene by scene (props accumulate)
        if a.only and sc["id"] not in a.only:
            continue                               # (its layout still feeds the later scenes at the place)
        scene["book"] = a.book
        (sd / f"{sc['id']}.json").write_text(json.dumps(scene, indent=1, default=lambda o: np.asarray(o).tolist()))
        ms = {m["id"]: m for m in moments}
        shots = {mid: shot_for(ms[mid], frames[mid], scene, cast, a.chapter) for mid in sc["moments"]}
        for mid_, sh_ in shots.items():              # every figure on OPEN ground of the set (10-22: the dragon's mark
            for ln in on_open_ground(sh_):           # 9 m out lay inside the cave rock; the camera saw the set's back)
                sys.stderr.write(f"{mid_}: {ln}\n")
        keep_absent(sc["moments"], frames, shots, scene)
        scene["prop_log"] = prop_states(scene, sc["moments"], frames, shots)
        (sd / f"{sc['id']}.json").write_text(json.dumps(scene, indent=1, default=lambda o: np.asarray(o).tolist()))
        for mid, shot in shots.items():
            (shd / f"{mid}.json").write_text(json.dumps(shot, indent=1, default=lambda o: np.asarray(o).tolist()))
        report.append({"scene": sc["id"], "where": sc["where"], "moments": len(sc["moments"]), "axis": scene["axis"],
                       "people": list(scene["marks"])})
        if a.sheets:
            sheet(scene, shd, sd / f"{sc['id']}.jpg", render=not a.no_render)
    (cd / "stage" / "scenes.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"scenes": len(report), "shots": sum(r["moments"] for r in report),
                      "dir": str(cd / "stage")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
