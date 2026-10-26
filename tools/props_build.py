"""Every PHYSICAL prop of a chapter as a textured 3D model on the prop library (any book; book data in, book data out).

    P=~/tts-ab/.venv-pose/bin/python
    $P -I tools/props_build.py plan  <book> <chapter>      classify -> <book>/work/video/stage/props/plan.json
    $P -I tools/props_build.py build <book> <chapter> [--card 05] [--only NAME ...] [--force STEP ...]

1. CLASSIFY (tools/stage_templates/prop_classes.json, logged): each chapter prop (anim/keyframes.json "props") is
   PHYSICAL (a model), SET dressing (the sets), a CHARACTER feature (the character models / wardrobe) or SKY/FX;
   physical props group into FAMILIES with STATES ("Sword, heated", "Dark Blue Sword" = states of the sword: one
   mesh, material variants); the library props the chapter's staged shots use (an anvil, tongs) are added.
   Each family has a MOBILITY (fixed / movable / semi) and, for articulated ones, JOINTS.
2. MODEL per family: the prop reference drawing (refs/prop_<name>-*.png; drawn with Qwen on a plain background,
   3/4 view, when missing) -> rembg cut-out -> Hunyuan3D shape -> tools/props_bl.py: aligned onto the library
   proxy (the proxy keeps doing contacts and collisions; the mesh is what renders), scaled to the proxy, painted
   from the drawing, joint parts split out, 3 QA views + a 1.8 m figure for scale.
3. REGISTER: <book>/work/video/stage/props/props.json (BLOCKOUT_PROPS_EXTRA format: the library entry + "mesh",
   "mobility", "states", "joints", "light"), and a QA sheet stage/props/qa.jpg.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import numpy as np
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import characters as CH  # noqa: E402

BLENDER = os.path.expanduser("~/opt/blender/current/blender")
CLASSES = json.loads((HERE / "stage_templates" / "prop_classes.json").read_text())


def log(m):
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} {m}\n")


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def classify(name):
    """(class, rule) for one chapter prop name; physical rules first, by word -- except a vision / ghost (fx)
    and a name whose HEAD noun is a set word (10-21: "Stone Well" became the 'stone' prop instead of the
    star-iron; "Vision of Forge" became a hearth prop)."""
    low = name.lower()
    if re.search(r"\b(vision|ghostly|translucent|apparition)\b", low):
        return "fx", {"word": "vision"}
    head = (re.findall(r"[a-z]+", low) or [""])[-1]
    if head in CLASSES["set"]:
        return "set", {"word": head}
    for r in CLASSES["physical"]:
        for w in r["words"]:
            if re.search(rf"\b{re.escape(w)}", low):
                state = None
                rest = re.sub(rf"\b{re.escape(w)}\w*", "", low).replace(",", " ").strip()
                for sw in (r.get("state_words") or {}):
                    if sw in rest:
                        state = sw
                return "physical", dict(r, state=state, word=w)
    for k in ("character", "fx", "set"):
        for w in CLASSES[k]:
            if re.search(rf"\b{re.escape(w)}", low):
                return k, {"word": w}
    return "unknown", {}


SIMILE_RX = re.compile(r",?\s*\b(?:about |roughly |nearly |almost )?(?:the size of|the shape of|shaped like|as (?:big|large|"
                       r"long|heavy) as|like)\s+(?:a|an|the)\s+[^,.;]+", re.I)


def literal_desc(desc: str) -> str:
    """A prop description as an image model reads it: comparisons drawn literally are cut (10-22: "a large dark stone
    the size of a sleeping dog" drew a bronze dog twice; the class's size_m carries the size)."""
    out = SIMILE_RX.sub("", desc or "")
    return re.sub(r"\s{2,}", " ", out).strip(" ,")


def plan(book, chapter):
    vd = CH.vdir_of(book)
    cd = vd / chapter
    kf = json.loads((cd / "anim" / "keyframes.json").read_text())
    props = dict(kf.get("props") or {})
    # library props the staged shots use (set props and held items), if the scene layer has run
    sd = cd / "stage" / "shots"
    used = set()
    refs = {}                                          # base -> number of shots that reference it
    shot_files = sorted(sd.glob("*.json")) if sd.exists() else []
    for p in shot_files:
        s = json.loads(p.read_text())
        here_ = set()
        for pr in (s.get("set") or {}).get("props", []):
            if "contact proxy" in (pr.get("name") or "") or pr.get("prop") in ("box", "facade"):
                continue
            here_.add(pr.get("prop") or pr.get("name"))
        for f in s.get("figures", []):
            here_.update((f.get("holds") or {}).values())
            for e in (s.get("end") or {}).get("figures", []):
                here_.update((e.get("holds") or {}).values())
        here_.update(((s.get("props_state") or {}).get("props") or {}).keys())
        used |= here_
        for u in here_:
            if u:
                b_ = classify(u)[1].get("base") or u
                refs[b_] = refs.get(b_, 0) + 1
    for u in sorted(x for x in used if x):
        if not any(classify(n)[1].get("base") == classify(u)[1].get("base") for n in props) and classify(u)[0] == "physical":
            props[u.title()] = f"a {u}"
    # physical props the PROMPTS name though no keyframe lists them (an anvil under the hammer): only before the
    # scene layer has run (after it, the shots ARE the prop set)
    txt = " ".join((f.get("prompt", "") + " " + f.get("motion", "")) for f in kf.get("frames", [])).lower()
    have = {classify(n)[1].get("base") for n in props}
    try:
        placewords = {re.findall(r"[a-z]+", n_.lower())[-1] for n_ in json.loads((vd / "look.json").read_text())
                      .get("places", {}) if re.findall(r"[a-z]+", n_.lower())}       # a place's HEAD noun only
    except (OSError, ValueError):
        placewords = set()
    for r in CLASSES["physical"]:
        if r["base"] in have or shot_files:
            continue
        # as a NOUN only: after an article / possessive / number (10-21: "Edric bows to the King" built a bow)
        det = r"\b(?:a|an|the|his|her|their|its|this|that|one|two)\s+(?:[a-z'-]+\s+){0,2}"
        # ...and as the HEAD noun ("the stone walls" is walls), never a word of a place's name ("the forge" is the
        # building, not a hearth prop; 10-21)
        head = r"(?!\s+(?!(?:in|on|at|by|of|with|to|from|and|or|as|is|was|that|which|under|over|beside|near)\b)[a-z]{3,}\b)"
        # a word an existing prop's description already uses for itself ("a dark, shimmering stone" = the star-iron)
        dw = " ".join(str(v) for n_, v in props.items() if classify(n_)[0] == "physical").lower()
        hits = sum(len(re.findall(rf"{det}{re.escape(w)}s?\b{head}", txt)) for w in r["words"]
                   if w not in placewords and not re.search(rf"\b{re.escape(w)}\b", dw))
        if hits >= 2:
            props[r["base"].title()] = f"a {r['base']} (named {hits}x in the chapter's shots)"
    fams, rows = {}, []
    for name, desc in props.items():
        cls, r = classify(name)
        rows.append({"name": name, "class": cls, "base": r.get("base"), "state": r.get("state"), "rule": r.get("word")})
        if cls != "physical":
            continue
        fam = fams.setdefault(r["base"], {"base": r["base"], "mobility": r["mobility"], "size_m": r["size_m"],
                                          "joints": r.get("joints"), "light": r.get("light"), "names": [],
                                          "states": {}, "desc": {}})
        fam["names"].append(name)
        fam["desc"][name] = literal_desc(desc)
        if r.get("state"):
            fam["states"][r["state"]] = dict((r.get("state_words") or {}).get(r["state"]) or {}, name=name)
    # STATES WITH THEIR OWN SHAPE get their own mesh (a sub-family "<base>__<state>" on the same library entry):
    # a cracked ingot, a glowing finished sword, the dark-blue sword; plus the FORGING WORK state of a family
    # (prop_classes "work_state": the sword's rough blank on the anvil)
    for b, fam in list(fams.items()):
        for st_, sd in fam["states"].items():
            # a MATERIAL / CONDITION state (heated, cracked, dark blue) is DERIVED from the base mesh (same geometry:
            # tint, emission, cracks); only a state whose class entry says "shape": true gets its own picture + shape
            fx_ = {k: v for k, v in sd.items() if k != "name"}
            fams[f"{b}__{slug(st_)}"] = {"base": b, "state_of": b, "state": st_, "mobility": fam["mobility"],
                                         "derive": not fx_.get("shape"), "state_fx": fx_,
                                         "size_m": fam["size_m"], "joints": fam.get("joints"), "light": fam.get("light"),
                                         "names": [sd["name"]], "states": {}, "desc": {sd["name"]: fam["desc"].get(sd["name"], "")}}
        rule = next((r for r in CLASSES["physical"] if r["base"] == b), {})
        ws = rule.get("work_state")
        if ws:
            fams[f"{b}__{slug(ws['name'])}"] = {"base": ws.get("library", b), "state_of": b, "state": ws["name"],
                                                "ref_prompt": ws.get("ref_prompt"),
                                                "mobility": fam["mobility"], "size_m": fam["size_m"], "joints": None,
                                                "light": None, "names": [ws["name"]], "states": {},
                                                "desc": {ws["name"]: ws["desc"]}}
    # THE BOOK'S PROP SET = what the staged shots reference (grounded props, held items, set props of the sets
    # used): anything else is not built and is pruned from the registry (a library entry is never instantiated
    # for a book unless a shot needs it)
    pruned = {}
    if shot_files:
        for k in list(fams):
            b_ = k.split("__")[0]
            if refs.get(b_, 0) == 0 and refs.get(fams[k]["base"], 0) == 0:
                pruned[k] = "0 shot references"
                del fams[k]
        for k, f in fams.items():
            f["shot_refs"] = refs.get(k.split("__")[0], 0) or refs.get(f["base"], 0)
    out = vd / "stage" / "props"
    out.mkdir(parents=True, exist_ok=True)
    pl = {"chapter": chapter, "classified": rows, "families": fams, "pruned": pruned, "shot_refs": refs}
    (out / "plan.json").write_text(json.dumps(pl, indent=1))
    return vd, pl


def ref_for(vd, fam):
    """The family's reference drawing: the base name's refs/prop_<slug>-*.png (newest), else a state's."""
    refs = vd / "refs"
    cands = []
    if fam.get("state_of"):                            # a state: its OWN drawing only (or none -> drawn)
        for n in fam["names"]:
            cands += sorted(refs.glob(f"prop_{slug(n)}-*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0] if cands else None
    for n in sorted(fam["names"], key=lambda n: ("," in n, len(n))):
        cands += sorted(refs.glob(f"prop_{slug(n)}-*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    cands += sorted(refs.glob(f"prop_{slug(fam['base'])}-*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def clean_refs(vd, fams, card, style):
    """Every family's reference must show the SINGLE object alone on a plain background (a hand holding reins,
    a column beside the sword, a tapestry for the oilcloth, a star for the star-iron poison image-to-3D): the
    vision model judges the book's ref; a failing ref is redrawn as a product shot from the description
    (keeping the old ref as a style reference only if it at least shows that kind of object), and judged again."""
    d = vd / "stage" / "props" / "_drawn"
    d.mkdir(parents=True, exist_ok=True)

    def q_alone(b, desc):
        return (f"Answer yes or no: does this picture show ONLY one single {b} ({desc}), alone, the whole object "
                f"visible, on a plain background, with NO hands, NO people, NO buildings, columns, sky or scenery?")
    # refs VERIFIED by eye (book data: stage/props/ref_verified.json {prop: {"ref": path, "by": ...}}): the vision
    # judge is skipped for them (it rejects some clean pictures, e.g. a long thin blade on grey)
    vp = vd / "stage" / "props" / "ref_verified.json"
    verified = json.loads(vp.read_text()) if vp.exists() else {}
    chapter_text = " ".join(f.get("prompt", "") for kp in vd.glob("*/anim/keyframes.json")
                            for f in json.loads(kp.read_text()).get("frames", []))
    for b, fam in fams.items():
        v_ = verified.get(b)
        if v_ and Path(v_["ref"]).exists():
            fam["ref"] = v_["ref"]
            fam["ref_judge"] = {"alone": "yes (verified)", "plain": "yes (verified)", "kind": "yes"}
            fam["ref_verified"] = v_
    from tts_reader import video as _V
    per = _V.period_of(str(vd))
    pq = {"period": _V.period_question(per)} if per else {}   # 10-21: a modern claw hammer passed every check
    items = []
    for b, fam in fams.items():
        if fam.get("ref_verified"):
            continue
        ref = fam.get("ref") or ref_for(vd, fam)
        desc = next(iter(fam["desc"].values()))
        if ref:
            items.append({"id": b, "image": str(ref), "ask": {**pq, "alone": q_alone(b, desc),
                          "kind": f"Answer yes or no: is the main object in this picture a {b}?",
                          "plain": "Answer yes or no: is the background completely plain and empty (one flat colour), "
                                   "with no window, wall, column, furniture, cloud, smoke, sky or landscape anywhere?"}})
    # the answers are cached by picture (path + mtime) and questions: an unchanged reference is not asked again
    # (10-21: every stage run re-loaded the vision judge for the same 13 pictures)
    jc_p = d / "judge_cache.json"
    jc = json.loads(jc_p.read_text()) if jc_p.exists() else {}

    def _jk(it):
        return hashlib.sha1(json.dumps([it["image"], Path(it["image"]).stat().st_mtime, it["ask"]],
                                       sort_keys=True).encode()).hexdigest()[:20]
    ask = [it for it in items if _jk(it) not in jc]
    ans = CH.vl_ask(ask) if ask else {}
    for it in ask:
        if it["id"] in ans:
            jc[_jk(it)] = ans[it["id"]]
    ans.update({it["id"]: jc[_jk(it)] for it in items if it["id"] not in ans and _jk(it) in jc})
    jc_p.write_text(json.dumps(jc, indent=1))
    jobs = []
    for b, fam in fams.items():
        if fam.get("ref_verified"):
            continue
        a = ans.get(b, {})
        fam["ref_judge"] = a
        rule_ = next((r for r in CLASSES["physical"] if r["base"] == fam["base"]), {})
        rp_ = fam.get("ref_prompt") or (rule_.get("ref_prompt") if not fam.get("state_of") else None)
        if not rp_ and not fam.get("state_of"):
            # a craft's own tool (10-21: a modern claw hammer and kitchen tongs in a medieval forge): the rule's
            # context prompt whose words the prop's description or the chapter's shots use
            ctx_ = (" ".join(fam["desc"].values()) + " " + (chapter_text or "")).lower()
            rp_ = next((c["prompt"] for c in rule_.get("context_prompts") or []
                        if any(re.search(rf"\b{re.escape(w)}", ctx_) for w in c["when"])), None)
        forced = bool(rp_)
        rt_ = int(fam.get("retry") or 0)
        if CH._yes(a.get("alone")) and CH._yes(a.get("plain", "yes")) and CH._yes(a.get("period", "yes")) \
                and not forced and not rt_:
            continue
        desc = next(iter(fam["desc"].values()))
        ref = fam.get("ref") or ref_for(vd, fam)
        for k in range(2):
            tag_ = f"r{rt_}" if rt_ else ""
            j = {"id": f"clean_{b}_{tag_}{k}", "seed": 5 + 11 * k + 101 * rt_, "path": str(d / f"clean_{b}_{tag_}{k}.png"),
                 "size": [1024, 1024],
                 "prompt": (rp_ + f" {style}No text.") if forced else
                 # (10-22: a stone wrapped in cloth was drawn as an animal, twice: the object is said to be inanimate)
                 (f"A single {b}: {desc}. An inanimate object: no animal, no face, no figure. Product shot of the "
                  f"object ALONE, "
                  f"three-quarter view from slightly above, the whole object in the picture, plain flat light grey "
                  f"background, soft even light, no hands, no people, no scenery, nothing else. {style}No text.")}
            if ref and CH._yes(a.get("kind")) and not forced:
                j["refs"] = [str(ref)]
            jobs.append(j)
    if jobs:
        for j in jobs:                          # a drawing made from another prompt is redrawn (10-22: the stone's
            sig_ = Path(j["path"] + ".prompt")   # two "sleeping dog" drawings were reused after its words were fixed)
            if Path(j["path"]).exists() and (not sig_.exists() or sig_.read_text() != j["prompt"]):
                Path(j["path"]).unlink()
        CH.run_qwen(jobs, d, card)
        for j in jobs:
            if Path(j["path"]).exists():
                Path(j["path"] + ".prompt").write_text(j["prompt"])
        def asked_desc(j):                      # a class drawing prompt (forced) leaves the book's words out: the
            dsc = literal_desc(next(iter(fams[j["id"][6:].rsplit("_", 1)[0]]["desc"].values())))   # picture is judged
            return dsc if dsc and dsc in j["prompt"] else "as described"      # on what it was asked for (10-22 stone)
        res = CH.vl_ask([{"id": j["id"], "image": j["path"], "ask": {**pq, "alone": q_alone(
            j["id"][6:].rsplit("_", 1)[0], asked_desc(j))}} for j in jobs if Path(j["path"]).exists()])
        for j in jobs:
            b = j["id"][6:].rsplit("_", 1)[0]
            if CH._yes(res.get(j["id"], {}).get("alone")) and CH._yes(res.get(j["id"], {}).get("period", "yes")) \
                    and not fams[b].get("clean_ref"):
                fams[b]["clean_ref"] = j["path"]
        for b, fam in fams.items():
            if fam.get("clean_ref"):
                fam["ref"] = fam["clean_ref"]
                log(f"{b}: reference redrawn -> {fam['ref']}")
            elif not CH._yes(fam.get("ref_judge", {}).get("alone")):
                # no redraw passed the judge: a product-shot redraw on plain grey still beats the story picture
                # (its background becomes a slab in the 3D shape)
                # UNSURE = FAIL: never used silently. The gate retries with new seeds (up to 3), then lists the
                # prop as needs-attention with the candidates attached
                cand = [j["path"] for j in jobs if j["id"][6:].rsplit("_", 1)[0] == b and Path(j["path"]).exists()]
                fam["ref_fail"] = {"why": "no reference passed the vision check (single object, plain background)",
                                   "candidates": cand}
                log(f"{b}: REFERENCE FAILED the vision check (candidates {cand})")


def draw_missing(vd, fams, card, style):
    jobs = []
    d = vd / "stage" / "props" / "_drawn"
    d.mkdir(parents=True, exist_ok=True)
    for b, fam in fams.items():
        if ref_for(vd, fam) is None:
            desc = next(iter(fam["desc"].values()))
            jobs.append({"id": f"prop_{b}", "seed": 3, "path": str(d / f"prop_{b}.png"), "size": [1024, 1024],
                         "prompt": (f"A single {b}: {desc}. Product shot of the object alone, three-quarter view from "
                                    f"slightly above, the whole object in the picture, plain flat light grey "
                                    f"background, soft even light, no people, nothing else. {style}No text.")})
    if jobs:
        CH.run_qwen(jobs, d, card)
    for b, fam in fams.items():
        p = d / f"prop_{b}.png"
        if ref_for(vd, fam) is None and p.exists():
            fam["ref"] = str(p)


PRIORITY = ("sword", "hammer", "tongs", "ingot", "lantern", "shield")


def _order(k):
    b = k.split("__")[0]
    return (PRIORITY.index(b) if b in PRIORITY else len(PRIORITY), k)


def shapes_parallel(out, fams, cards, force):
    """Hunyuan3D shapes for every family that needs one, one worker per GPU card (each its own lock), in the
    priority order."""
    import queue
    import threading
    q = queue.Queue()
    for b in sorted(fams, key=_order):
        d = out / b
        if (d / "ref_rgba.png").exists() and ("shape" in force or (d / "_need_shape").exists() or not (d / "shape.glb").exists()):
            q.put(b)

    def work(card):
        while True:
            try:
                b = q.get_nowait()
            except queue.Empty:
                return
            d = out / b
            fd = CH.gpu_turn(card)
            try:
                log(f"{b}: Hunyuan3D shape on card {card}")
                r = subprocess.run([CH.HY3D_PY, "-I", str(HERE / "blockout_shape.py"), str(d / "ref_rgba.png"),
                                    str(d / "shape.glb"), "--faces", "30000"], capture_output=True, text=True,
                                   env=CH.hip_env(card), timeout=840)
                (d / "shape.log").write_text(r.stdout[-3000:] + r.stderr[-3000:])
                if (d / "_need_shape").exists():
                    (d / "_need_shape").unlink()
            finally:
                fd.close()
    th = [threading.Thread(target=work, args=(c,)) for c in cards]
    for t in th:
        t.start()
    for t in th:
        t.join()


# ------------------------------------------------------------------------------------------------ freeze / gate
FROZEN_FILES = ("model.blend", "model.json", "job.json", "ref_used.txt", "ref_rgba.png")


def _sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _approvals(vd):
    p = vd / "stage" / "approvals.json"
    return p, (json.loads(p.read_text()) if p.exists() else {"items": {}})


def frozen(vd):
    """{prop: frozen record} from the book's approvals.json (items "prop:<name>" with a "frozen" record)."""
    _, ap = _approvals(vd)
    return {k.split(":", 1)[1]: e["frozen"] for k, e in ap.get("items", {}).items()
            if k.startswith("prop:") and e.get("frozen")}


def freeze(book, names, by="coordinator", note="coordinator-passed"):
    """LOCK a passing prop: copy its artifacts to stage/props/_frozen/<name>/v<N>/, record the model's sha256 in
    approvals.json (status unchanged: the user's final OK still gates). props_build then SKIPS it; only an
    explicit --unfreeze rebuilds it."""
    vd = CH.vdir_of(book)
    out = vd / "stage" / "props"
    p, ap = _approvals(vd)
    items = ap.setdefault("items", {})
    done = []
    for b in names:
        d = out / b
        if not (d / "model.blend").exists():
            log(f"{b}: cannot freeze (no model)")
            continue
        fz = out / "_frozen" / b
        n = len([x for x in fz.glob("v*") if x.is_dir()]) + 1 if fz.exists() else 1
        vdir_ = fz / f"v{n}"
        vdir_.mkdir(parents=True, exist_ok=True)
        import shutil
        for f in FROZEN_FILES:
            if (d / f).exists():
                shutil.copy2(d / f, vdir_ / f)
        for q in d.glob("qa_*.png"):
            shutil.copy2(q, vdir_ / q.name)
        e = items.setdefault(f"prop:{b}", {"status": "pending", "kind": "prop", "name": b})
        e["frozen"] = {"version": n, "dir": str(vdir_), "sha256": _sha(d / "model.blend"), "by": by, "note": note,
                       "at": time.strftime("%Y-%m-%d %H:%M")}
        done.append(b)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ap, indent=1, ensure_ascii=False))
    os.replace(tmp, p)
    log(f"frozen: {done}")
    return done


def ensure_frozen_intact(out, fz):
    """A frozen prop's live model must be its frozen copy: restore it if anything changed it."""
    import shutil
    bad = []
    for b, rec in fz.items():
        d = out / b
        if not (d / "model.blend").exists() or _sha(d / "model.blend") != rec["sha256"]:
            d.mkdir(parents=True, exist_ok=True)
            for f in Path(rec["dir"]).iterdir():
                shutil.copy2(f, d / f.name)
            bad.append(b)
    if bad:
        log(f"frozen props RESTORED from their frozen copies: {bad}")
    return bad


def prop_checks(b, fam, d, ent):
    """The automated gate before a prop is shown: [] = pass, else the reasons."""
    why = []
    mj = json.loads((d / "model.json").read_text()) if (d / "model.json").exists() else {}
    if not ent.get("mesh") or not Path(ent["mesh"]).exists():
        why.append("no mesh")
    ext = mj.get("extent_m") or []
    want = _rule_of(fam["base"]).get("size_m") or fam.get("size_m")
    if ext and want and abs(max(ext) - want) > 0.15 * want:
        why.append(f"size {max(ext):.2f} m vs class {want} m")
    rc = mj.get("rest_check") or {}
    if rc and not rc.get("ok", True) and not _rule_of(fam["base"]).get("soft"):
        why.append(f"rest orientation ({rc})")
    if (mj.get("islands") or 1) > 3:
        why.append(f"{mj['islands']} separate pieces (junk geometry)")
    j = fam.get("ref_judge") or {}
    if not fam.get("derive") and not fam.get("procedural") and j and not CH._yes(j.get("plain", "yes")) \
            and "_drawn/clean_" not in str(fam.get("ref")):
        why.append("reference picture has background objects")
    if fam.get("state_of") and not fam.get("derive") and fam["base"] == fam["state_of"]:
        why.append("material state not derived from the base mesh")
    if fam.get("shot_refs") == 0:
        why.append("0 shot references")
    if fam.get("ref_fail"):
        why.append(f"reference failed the vision check (candidates: {fam['ref_fail']['candidates']})")
    return why


def build(book, chapter, card="any", only=None, force=(), cards=("05", "65"), unfreeze=(), _round=0):
    vd, pl = plan(book, chapter)
    fams = pl["families"]
    out0 = vd / "stage" / "props"
    fz = {k: v for k, v in frozen(vd).items() if k not in set(unfreeze)}
    ensure_frozen_intact(out0, fz)
    skipped = sorted(k for k in fams if k in fz)
    fams = {k: v for k, v in fams.items() if k not in fz}           # FROZEN: never rebuilt (only --unfreeze)
    if skipped:
        log(f"frozen (skipped): {skipped}")
    if only:
        fams = {k: v for k, v in fams.items() if k in only or k.split("__")[0] in only}
    for k, v in fams.items():
        rf = out0 / k / "retry.txt"
        v["retry"] = int(rf.read_text()) if rf.exists() else 0
    sty = sorted(vd.glob("*/style.txt"))
    style = (sty[0].read_text().strip().split(":")[0] + ". ") if sty else ""
    from tts_reader import video as _V
    style += _V.period_line(str(vd)).strip() + " " if _V.period_of(str(vd)) else ""
    own = {k: v for k, v in fams.items() if not v.get("derive")}      # derived states need no picture
    draw_missing(vd, own, card, style)
    clean_refs(vd, own, card, style)
    out = vd / "stage" / "props"
    lib = json.loads((HERE / "stage_props" / "props.json").read_text())
    reg_p = out / "props.json"
    reg = json.loads(reg_p.read_text()) if reg_p.exists() else {}
    for b, fam in fams.items():
        d = out / b
        d.mkdir(parents=True, exist_ok=True)
        rule_ = next((r for r in CLASSES["physical"] if r["base"] == fam["base"]), {})
        if fam.get("derive"):
            fam["ref"] = fams.get(fam["state_of"], {}).get("ref") or "derived"
            fam["_force"] = sorted(set(force) | {"prep"})
            continue
        if rule_.get("procedural") and not fam.get("state_of"):
            fam["procedural"] = rule_["procedural"]
            fam["ref"] = fam.get("ref") or str(ref_for(vd, fam) or "")
            fam["_force"] = sorted(set(force) | {"prep"})
            continue                                   # built procedurally by props_bl (no Hunyuan)
        if fam.get("ref_fail"):
            log(f"{b}: not rebuilt (its reference failed the vision check; previous model kept, flagged)")
            continue
        ref = fam.get("ref") or ref_for(vd, fam)
        if ref is None:
            log(f"{b}: no reference drawing")
            continue
        fam["ref"] = str(ref)
        used = d / "ref_used.txt"
        fforce = set(force)
        stamp = f"{ref} {Path(ref).stat().st_mtime:.0f}"   # the same file name redrawn is a NEW reference
        if not used.exists() or used.read_text().strip() not in (stamp,) and \
                not (used.read_text().strip() == str(ref) and Path(ref).stat().st_mtime < used.stat().st_mtime):
            fforce |= {"cut", "shape", "prep"}            # a new reference drawing: rebuild this prop
        used.write_text(stamp)
        force_ = fforce
        cut = d / "ref_rgba.png"
        if "cut" in force_ or not cut.exists():
            CH.rembg_rgba(ref, cut)
        glb = d / "shape.glb"
        if "shape" in force_ or not glb.exists():
            (d / "_need_shape").write_text("1")
        fam["_force"] = sorted(force_)
    shapes_parallel(out, fams, cards, force)
    for b, fam in sorted(fams.items(), key=lambda kv: (bool(kv[1].get("derive")), _order(kv[0]))):
        d = out / b
        if not fam.get("ref") and not fam.get("derive"):
            continue
        force_ = set(fam.get("_force") or [])
        glb = d / "shape.glb"
        cut = d / "ref_rgba.png"                       # THIS prop's cut-out (paint source)
        rule_ = next((r for r in CLASSES["physical"] if r["base"] == fam["base"]), {})
        base = lib.get(fam["base"]) or lib.get(b) or {}
        if not glb.exists() and not fam.get("procedural") and not fam.get("derive"):
            log(f"{b}: shape failed")
            continue
        kind_ = base.get("kind") or ("set" if fam["mobility"] == "fixed" else "held")
        job = {"mesh": str(glb), "ref": str(cut), "proxy": base.get("parts", []), "frame": base.get("frame"),
               "kind": kind_, "size_m": fam["size_m"], "joints": fam.get("joints") or [],
               "derive_from": str(out / fam["state_of"] / "model.blend") if fam.get("derive") else None,
               "state_fx": fam.get("state_fx"),
               "procedural": fam.get("procedural"),
               "rest": rule_.get("rest") or ("upright" if rule_.get("architectural") else None),
               "soft": bool(rule_.get("soft")),
               "out_blend": str(d / "model.blend"), "out_json": str(d / "model.json"), "qa_prefix": str(d / "qa")}
        (d / "job.json").write_text(json.dumps(job, indent=1))
        # a changed prep (props_bl.py: orientation / rest rules) or job re-preps the model (10-21: the lie-flat and
        # keep-up rules never reached the built props: only a new reference re-prepped)
        psig = hashlib.sha1((HERE / "props_bl.py").read_bytes() + (d / "job.json").read_bytes()).hexdigest()[:16]
        if not (d / "prep.sig").exists() or (d / "prep.sig").read_text().strip() != psig:
            force_ = force_ | {"prep"}
        if "prep" in force_ or not (d / "model.blend").exists():
            r = subprocess.run([BLENDER, "-b", "--factory-startup", "-t", "8", "--python", str(HERE / "props_bl.py"),
                                "--", str(d / "job.json")], capture_output=True, text=True, timeout=900)
            (d / "prep.log").write_text(r.stdout[-4000:] + r.stderr[-4000:])
            if not (d / "model.blend").exists():
                log(f"{b}: prep failed (see {d / 'prep.log'})")
                continue
        (d / "prep.sig").write_text(psig)
        mj = json.loads((d / "model.json").read_text())
        if any("handle_m" not in j for j in mj.get("joints") or []):
            # each jointed part's HANDLE (10-22: the bellows were never pumped - nothing knew where a hand goes on
            # the board): the part's vertex farthest from its hinge, upper part; staging locks the hands to it
            r = subprocess.run([BLENDER, "-b", "--factory-startup", str(d / "model.blend"), "--python",
                                str(HERE / "props_handle_bl.py"), "--", json.dumps(mj["joints"])],
                               capture_output=True, text=True, timeout=300)
            got = {ln.split()[1]: json.loads(ln.split(" ", 2)[2]) for ln in r.stdout.splitlines() if ln.startswith("HANDLE ")}
            for j in mj["joints"]:
                if j["name"] in got:
                    j["handle_m"] = got[j["name"]]
            (d / "model.json").write_text(json.dumps(mj, indent=1))
            log(f"{b}: joint handle(s) {got}")
        ent = copy.deepcopy(base) if base else {"kind": "held", "parts": mj.get("proxy_parts", [])}
        ent.update({"mesh": str(d / "model.blend"), "mesh_fit": mj.get("fit"), "mobility": fam["mobility"],
                    "extent_m": mj.get("extent_m"), "stand_m": rule_.get("stand_m"),
                    "states": fam["states"], "joints": mj.get("joints") or fam.get("joints") or [],
                    "light": fam.get("light"), "names": fam["names"], "ref": fam["ref"]})
        reg[b] = ent
        if fam.get("state_of"):                         # the parent family swaps to this mesh in that state
            par = reg.setdefault(fam["state_of"], copy.deepcopy(lib.get(fam["state_of"]) or {}))
            par.setdefault("state_meshes", {})[fam["state"]] = ent["mesh"]
            if fam["base"] != fam["state_of"]:           # a different library entry (the forging blank): an ALIAS
                reg[fam["base"]] = dict(ent, state=fam["state"], alias_of=b)
        log(f"{b}: registered ({fam['mobility']}, states {list(fam['states'])}, fit {mj.get('fit')})")
    allf = pl["families"]
    import shutil
    for k in list(reg):
        ent_ = reg[k]
        keep_ = k in allf or (ent_.get("alias_of") in allf)
        why_ = None if keep_ else (pl.get("pruned", {}).get(k) or "not in the book's prop set (0 shot references)")
        if keep_ and (not ent_.get("mesh") or not Path(ent_["mesh"]).exists()):
            why_ = "no mesh"
        if why_:                                        # PRUNED: out of the book's registry, kept aside with the reason
            un = out / "_unused" / k
            un.mkdir(parents=True, exist_ok=True)
            if (out / k).exists() and (out / k).is_dir():
                for f in (out / k).iterdir():
                    shutil.move(str(f), str(un / f.name))
                (out / k).rmdir()
            (un / "reason.json").write_text(json.dumps({"why": why_, "at": time.strftime("%Y-%m-%d %H:%M"),
                                                        "entry": ent_}, indent=1, default=str))
            log(f"{k}: PRUNED ({why_}) -> {un}")
            del reg[k]
    # THE GATE: every built prop's automated checks; failures are retried (new reference seed) up to 3x, then
    # listed as needs-attention instead of shown as done
    retry_ = []
    for b, fam in fams.items():
        if b not in reg:
            continue
        why = prop_checks(b, fam, out / b, reg[b])
        reg[b]["checks"] = {"fails": why, "at": time.strftime("%Y-%m-%d %H:%M")}
        reg[b]["qa_pass"] = not why
        reg[b]["shot_refs"] = fam.get("shot_refs")
        if why:
            reg[b]["needs_attention"] = why
            if fam.get("retry", 0) < 1 and not fam.get("derive") and not fam.get("procedural") \
                    and not fam.get("ref_verified"):
                retry_.append(b)
        else:
            reg[b].pop("needs_attention", None)
    for b in fz:
        if b in reg:
            reg[b]["frozen"] = fz[b]
            reg[b]["qa_pass"] = True
            mjf = out / b / "model.json"                # metadata refresh only (the frozen mesh is untouched)
            if mjf.exists():
                reg[b]["extent_m"] = json.loads(mjf.read_text()).get("extent_m")
            reg[b]["stand_m"] = _rule_of(b.split("__")[0]).get("stand_m")
    tmp_ = reg_p.with_suffix(".json.tmp")                # atomic: a reader never sees a half-written registry
    tmp_.write_text(json.dumps(reg, indent=1))
    os.replace(tmp_, reg_p)
    tmp2 = reg_p.with_suffix(".json.tmp")
    tmp2.write_text(json.dumps(reg, indent=1))
    os.replace(tmp2, reg_p)
    qa_sheet(out, reg, pl)                              # the sheet FIRST: one item's retries never block the rest
    if retry_ and _round < 1:                           # ONE retry at most; then a reasoned fix, not GPU rounds
        for b in retry_:
            (out / b / "retry.txt").write_text(str(fams[b].get("retry", 0) + 1))
        log(f"gate: retrying with a new reference {retry_} (round {_round + 1}): {[reg[b]['needs_attention'] for b in retry_]}")
        return build(book, chapter, card, retry_, set(force) | {"prep"}, cards, unfreeze, _round + 1)
    qa_sheet(out, reg, pl)
    return reg


def _rule_of(base):
    return next((r for r in CLASSES["physical"] if r["base"] == base), {})


def qa_sheet(out, reg, pl):
    from PIL import Image, ImageDraw
    W = 300
    rows = []
    prio = ["sword", "hammer", "tongs", "ingot", "lantern", "shield"]   # the QA review order (then the rest)

    def key_(b):
        base = b.split("__")[0]
        return (prio.index(base) if base in prio else len(prio), base, b)
    lib = json.loads((HERE / "stage_props" / "props.json").read_text())
    attention = []
    for b in sorted(reg, key=key_):
        ent = reg[b]
        if ent.get("alias_of"):
            continue                                     # the same model as its sub-family row
        if ent.get("needs_attention") and not ent.get("frozen"):
            attention.append(f"{b}: " + "; ".join(ent["needs_attention"]))
            continue                                     # GATED: never shown as done
        d = out / b
        ims = [Image.open(ent["ref"]).convert("RGB")] if ent.get("ref") and Path(ent["ref"]).exists() else []
        ims += [Image.open(p).convert("RGB") for p in sorted(d.glob("qa_*.png"))]
        row = Image.new("RGB", (W * 5, W + 40), (26, 26, 26))
        for i, im in enumerate(ims[:5]):
            im.thumbnail((W - 6, W - 6))
            row.paste(im, (i * W + 3, 3))
        dr = ImageDraw.Draw(row)
        js = ", ".join(f"{j['name']}:{j['type']}" for j in ent.get("joints") or [])
        mj = json.loads((d / "model.json").read_text()) if (d / "model.json").exists() else {}
        ext = mj.get("extent_m") or [0, 0, 0]
        L = max(ext) if ext else 0
        want = (_rule_of(b.split("__")[0]).get("size_m") or ent.get("size_m") or 0)
        size_ok = (not want) or abs(L - want) <= 0.15 * want
        rc = mj.get("rest_check") or {}
        g = (lib.get(b.split("__")[0]) or {}).get("affordances", {}).get("grip")
        dr.text((6, W + 6), f"{b}{'  FROZEN v' + str(ent['frozen']['version']) if ent.get('frozen') else ''}"
                            f"  [{ent.get('mobility')}]  states: {', '.join(ent.get('states') or {}) or '-'}"
                            f"  joints: {js or '-'}  fit {ent.get('mesh_fit')}", fill=(255, 255, 0))
        dr.text((6, W + 20), f"size {L:.2f} m (class {want} m) {'OK' if size_ok else 'CHECK'}"
                             f"   rest {rc.get('rest') or '-'} base {rc.get('base_ratio', '-')} "
                             f"{'' if not rc else ('OK' if rc.get('ok') else 'CHECK')}"
                             f"   grip {('%.2f..%.2f' % (g['from'], g['to'])) if g and 'from' in g else ('point' if g else '-')}"
                             f"   panels: ref | front | side | 3/4 | beside a 1.8 m figure",
                fill=(150, 255, 150) if size_ok and rc.get("ok", True) else (255, 120, 120))
        rows.append(row)
    cls = {}
    for r in pl["classified"]:
        cls.setdefault(r["class"], []).append(r["name"])
    hdr = Image.new("RGB", (W * 5, 90 + 16 * (len(attention) + 1)), (12, 12, 12))
    dr = ImageDraw.Draw(hdr)
    y = 6
    dr.text((6, y), f"NEEDS ATTENTION ({len(attention)}): " + ("none" if not attention else ""), fill=(255, 120, 120))
    y += 16
    for a_ in attention:
        dr.text((16, y), a_[:220], fill=(255, 160, 160))
        y += 16
    for k, v in cls.items():
        dr.text((6, y), f"{k}: {', '.join(v)}"[:230], fill=(200, 255, 200))
        y += 16
    H = hdr.height + sum(r.height for r in rows)
    sheet = Image.new("RGB", (W * 5, H), (0, 0, 0))
    sheet.paste(hdr, (0, 0))
    y = hdr.height
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out / "qa.jpg", quality=85)


def regress(book):
    """REGRESSION over every frozen prop (after any change to shared prop code): its model re-checked and its QA
    views re-rendered by the CURRENT props_bl (no rebuild), compared with the frozen copy: sha256, checks, pixels."""
    from PIL import Image, ImageChops
    vd = CH.vdir_of(book)
    out = vd / "stage" / "props"
    rep = {}
    for b, rec in frozen(vd).items():
        d = out / b
        rd = out / "_regress" / b
        rd.mkdir(parents=True, exist_ok=True)
        jb = json.loads((Path(rec["dir"]) / "job.json").read_text())
        job = dict(jb, qa_only=str(d / "model.blend"), out_json=str(rd / "check.json"), qa_prefix=str(rd / "qa"))
        (rd / "job.json").write_text(json.dumps(job, indent=1))
        subprocess.run([BLENDER, "-b", "--factory-startup", "-t", "8", "--python", str(HERE / "props_bl.py"), "--",
                        str(rd / "job.json")], capture_output=True, text=True, timeout=600)
        cj = json.loads((rd / "check.json").read_text()) if (rd / "check.json").exists() else {}
        fj = json.loads((Path(rec["dir"]) / "model.json").read_text())
        diffs = {}
        for q in sorted(Path(rec["dir"]).glob("qa_*.png")):
            n_ = rd / q.name
            if n_.exists():
                a_, b_ = Image.open(q).convert("L"), Image.open(n_).convert("L")
                if a_.size == b_.size:
                    diffs[q.stem] = round(float(np.asarray(ImageChops.difference(a_, b_)).mean()), 3)
        same_hash = _sha(d / "model.blend") == rec["sha256"]
        ok = same_hash and max(diffs.values() or [0]) < 1.0 and \
            [round(x, 2) for x in cj.get("extent_m", [])] == [round(x, 2) for x in fj.get("extent_m", [])]
        rep[b] = {"sha_same": same_hash, "pixel_diff": diffs, "extent_now": cj.get("extent_m"),
                  "extent_frozen": fj.get("extent_m"), "rest_check_now": cj.get("rest_check"), "ok": ok}
        log(f"regress {b}: {'UNCHANGED' if ok else 'CHANGED'} {rep[b]['pixel_diff']}")
    (out / "_regress" / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build", "qa", "freeze", "regress"])
    ap.add_argument("book")
    ap.add_argument("chapter")
    ap.add_argument("--card", default="any", help="05 | 65 | any (whichever lock is free first)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--force", nargs="*", default=[])
    ap.add_argument("--unfreeze", nargs="*", default=[], help="frozen props to rebuild anyway (explicit only)")
    ap.add_argument("--by", default="coordinator")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        freeze(a.book, a.only or [], by=a.by)
        return 0
    if a.cmd == "regress":
        rep = regress(a.book)
        print(json.dumps({k: v["ok"] for k, v in rep.items()}))
        return 0 if all(v["ok"] for v in rep.values()) else 2
    if a.cmd == "plan":
        vd, pl = plan(a.book, a.chapter)
        for r in pl["classified"]:
            print(f"{r['class']:10s} {r['name']:22s} base={r['base']} state={r['state']}")
        print("families:", {k: (v["mobility"], list(v["states"])) for k, v in pl["families"].items()})
        return 0
    if a.cmd == "qa":
        vd, pl = plan(a.book, a.chapter)
        out = vd / "stage" / "props"
        qa_sheet(out, json.loads((out / "props.json").read_text()), pl)
        return 0
    reg = build(a.book, a.chapter, a.card, a.only, set(a.force), unfreeze=a.unfreeze)
    print(json.dumps({"props": list(reg), "qa": str(CH.vdir_of(a.book) / "stage" / "props" / "qa.jpg")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
