"""PLAN REVIEW: the chapter plan checked for agreement BEFORE any picture is drawn (user 10-22: "a prompt double/
triple check ... scene consistency, prop naming, people naming ... strata would be good at catching and fixing if we
give it a couple of passes"). Runs at the end of `bookcaster plan`'s planning, before the sheets and stills.

1. RULES (certain, cheap):
   - who: only cast names (an alias -> the cast name; an unknown name dropped, logged); a cast name listed as a prop
     -> a figure;
   - every speaker of a quoted line in a scene is PRESENT in it (10-22: the Knight, who reports to the King, was
     not in his own scene);
   - a moment whose own text names another place than its scene is a CONFLICT (video.apply_scene keeps it), handed
     to the language model.
2. LANGUAGE MODEL (Strata), scene by scene: the scene card (place, side, present, props), the scene's text, every
   moment (who / where / props / prompt) and the conflicts. It answers with FIXES only: split the scene at a moment
   (a missed scene change: a place + side), the scene's present list, a moment's who / props / prompt. Never a
   moment's place (that is its scene's). Applied, logged with its reason.
3. A SECOND PASS on the scenes the first one changed; what it still wants is LISTED (refs/plan_review.json), not
   applied (no ping-pong).
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _canon(name, cast):
    if not name or name in cast:
        return name
    hits = [n for n in cast if name.lower() in n.lower().split() or n.lower() in name.lower()
            or name.lower() == n.lower()]
    return hits[0] if len(hits) == 1 else None


def n_kind(name, kinds) -> bool:
    return name in kinds


def rules(plan, frames, mid_t, starts, timeline, cast, log) -> list:
    """The certain fixes, in place. -> [lines]"""
    from tts_reader import video
    out = []
    creatures = video.creature_kinds(cast)
    for fr in frames:
        who = []
        for w in fr.get("who") or []:
            c = _canon(w, cast)
            if c is None:
                out.append(f"{fr.get('shot')}: {w!r} is no cast member: dropped")
            elif c not in who:
                if c != w:
                    out.append(f"{fr.get('shot')}: {w!r} -> {c!r}")
                who.append(c)
        props = []
        for pr in fr.get("props") or []:
            c = _canon(str(pr), cast)
            if c is not None and c in cast:
                if c not in who:
                    who.append(c)
                out.append(f"{fr.get('shot')}: prop {pr!r} is the cast member {c!r}: a figure")
            else:
                props.append(pr)
        fr["who"], fr["props"] = who, props
    # a must-have / check naming someone the prompt does not show is stale (10-22: "The dragon continues its flight
    # path" left on a rewritten throne-room shot kept the dragon in it and would have failed every key without one)
    kinds = video.creature_kinds(cast)
    for fr in frames:
        ptxt = " ".join([fr.get("prompt") or "", fr.get("motion") or ""]).lower()
        def shown(n):
            return bool(re.search(rf"\b{re.escape(n.split()[-1].lower())}", ptxt)) or \
                (n in kinds and bool(re.search(rf"\b{kinds[n]}", ptxt)))
        def stale(txt):
            t = str(txt).lower()
            return any((re.search(rf"\b{re.escape(n.split()[-1].lower())}", t) or
                        (n in kinds and re.search(rf"\b{kinds[n]}", t))) and not shown(n) for n in cast)
        for k in ("must", "checks"):
            keep = [x for x in fr.get(k) or [] if not stale(x.get("q") if isinstance(x, dict) else x)]
            if len(keep) != len(fr.get(k) or []):
                out.append(f"{fr.get('shot')}: {len(fr[k]) - len(keep)} stale {k} dropped (they name someone the prompt "
                           f"does not show)")
                fr[k] = keep
        for n in [w for w in fr.get("who") or [] if not shown(w) and n_kind(w, kinds)]:
            fr["who"].remove(n)
            out.append(f"{fr.get('shot')}: {n} dropped (a creature the prompt does not show)")
    # every speaker in a scene is present in it
    for q in timeline or []:
        if q.get("kind") != "quote" or not q.get("speaker") or q["speaker"] == "narrator":
            continue
        spk = _canon(q["speaker"], cast)
        sc = video.scene_at_t(starts, float(q["t"]))
        if spk and sc is not None and spk not in sc.get("present", []) and spk not in creatures:
            sc.setdefault("present", []).append(spk)
            out.append(f"{sc['id']}: {spk} speaks in it: present")
    for ln in out:
        log("  review rule: " + ln)
    return out


def _scene_prompt(sc, paras, moments, places, cast):
    text = "\n\n".join(f"[P{i}] {paras[i]}" for i in range(sc["from"], min(sc["to"], len(paras) - 1) + 1))[:7000]
    ms = "\n".join(
        f"- {m['shot']} (at {m['_t']:.0f}s): who {m.get('who')}; where {m.get('where')!r}; props {m.get('props')}; "
        f"prompt: {(m.get('prompt') or '')[:300]}"
        + (f"\n  CONFLICT: its text names {m['scene_conflict']['own_where']!r} but the scene is "
           f"{m['scene_conflict']['scene_where']!r}" if m.get("scene_conflict") else "")
        for m in moments)
    return (
        "You are the continuity supervisor of a film. Check ONE scene of the plan against the book's text and fix "
        "only what is wrong.\n\n"
        f"SCENE {sc['id']}: place {sc['where']!r} ({sc['side']}); present {sc.get('present')}; props "
        f"{sc.get('props')}\n\nTEXT of the scene:\n{text}\n\nMOMENTS (one picture each, in order):\n{ms}\n\n"
        f"Known places: {sorted(places)}\nCast: {sorted(cast)}\n\n"
        "Check:\n"
        "1. Does the whole scene happen in its place and on its side (inside/outside)? If the action MOVES to another "
        "place part-way (a later paragraph, an epilogue), split the scene at the first moment in the new place.\n"
        "2. Is everyone physically there in 'present' (also people who do not speak)? Nobody who is only talked "
        "about.\n"
        "3. Each moment: 'who' = the cast members visible in it (a creature the prompt shows is in 'who', never a "
        "prop); 'props' = objects only (no people, no places, no effects like smoke / fire / ash / light, no body parts); the prompt agrees with the text and the scene "
        "(nobody in it who is not present, no object the scene does not have, the right place words).\n"
        "5. Never show a death, a wound, blood or a body: show the moment before (the blade raised, the strike "
        "coming), never the hit landing in flesh.\n"
        "6. When you change a prompt, give its 'motion' too: one sentence, what moves in it.\n"
        "4. Two moments in a row with the same prompt: keep the action, give the second a different framing (closer "
        "or wider, another angle) in its prompt. Never empty a moment's 'who' to remove it: moments are never "
        "removed.\n"
        "The scene's props list is a SUMMARY: never rename an object in a moment's prompt or props to match it "
        "(\"the Star-iron Sword\" stays the Star-iron Sword).\n"
        "Answer with ONLY JSON, empty lists/objects when all is right:\n"
        '{"present": ["Name", ...] or null, '
        '"splits": [{"moment": "<id>", "where": "<place>", "side": "inside|outside", "present": ["Name"], '
        '"why": "..."}], '
        '"moments": {"<id>": {"who": [...], "props": [...], "prompt": "...", "motion": "...", "why": "..."}}}\n'
        "Give a moment only the fields that change. Never change a moment's place.")


def _ask(prompt, url, model):
    from tts_reader.score import _chat
    reply = _chat(url, model, prompt, max_tokens=4000)
    m = re.search(r"\{.*\}", reply, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except ValueError:
        return {}


def _para_at(pstarts, t):
    k = 0
    for i, t0 in enumerate(pstarts):
        if t0 <= t + 0.05:
            k = i
    return k


PROP_GENERIC = {"finished", "glowing", "cracked", "quenched", "flowing", "blank", "stable", "with", "from", "that",
                "this", "large", "small", "piece"}


def _identity_lost(old_prompt: str, new_prompt: str, props: list) -> list:
    """The distinctive words of the moment's props ("star-iron") that the old prompt has and the new one dropped."""
    words = {w for p in props for w in re.findall(r"[a-z][a-z-]{3,}", str(p).lower()) if w not in PROP_GENERIC}
    o, n = old_prompt.lower(), new_prompt.lower()
    return sorted(w for w in words if re.search(rf"\b{re.escape(w)}", o) and not re.search(rf"\b{re.escape(w)}", n))


def _guard_props(old: list, new: list, prompt: str):
    """A props change keeps the prop FAMILIES the prompt names (10-22: "Shimmering Stone" -> "Star-iron" made the
    stone an ingot; a duplicate's props emptied) and adds none it does not; within a family the old, more specific name
    stays ("Melted Shield", not "shield"). -> the new list, or None (rejected)."""
    import props_build as PB

    def fam(n):
        c, r = PB.classify(str(n))
        return r.get("base") if c == "physical" else None
    low = prompt.lower()
    of = {fam(x): x for x in old if fam(x)}
    nf = {fam(x): x for x in new if fam(x)}
    lost = [b for b in of if b not in nf and re.search(rf"\b{re.escape(b)}", low)]
    gained = [b for b in nf if b not in of and not re.search(rf"\b{re.escape(b)}", low)]
    if lost or gained:
        return None
    # an addition that is no OBJECT (smoke, fire, ash, a golden eye: effects and body parts) is dropped (10-22)
    return [of.get(fam(x), x) if fam(x) else x for x in new if fam(x) or x in old]


def _t0(sc, pstarts) -> float:
    """A scene's start (seconds): its paragraph's start, or its split moment's."""
    return 0.0 if sc.get("from", 0) == 0 and not sc.get("t_split") else max(pstarts[sc["from"]], float(sc.get("t_split") or 0.0))


def _t_end(sc, scenes, pstarts) -> float:
    t = _t0(sc, pstarts)
    nxt = [_t0(x, pstarts) for x in scenes if _t0(x, pstarts) > t + 1e-6]
    return min(nxt) if nxt else 1e9


def apply_answer(ans, sc, plan, frames_by_id, places, cast, pstarts, log, dry=False, paras=None,
                 apply_only=None, apply_splits=False) -> list:
    """One scene's answer applied (dry: only listed). -> [change lines]"""
    from tts_reader import video
    ch = []
    pres = ans.get("present")
    if isinstance(pres, list):
        pres = [c for c in (_canon(p, cast) for p in pres) if c]
        stext = " ".join(paras[i] for i in range(sc["from"], min(sc.get("to", sc["from"]), len(paras) - 1) + 1)).lower() \
            if paras else ""
        keep_ = [x for x in sc.get("present") or [] if x not in pres
                 and re.search(rf"\b{re.escape(x.split()[-1].lower())}", stext)]
        if keep_:                                   # (10-22: the King dropped from the hall where the Knight reports
            ch.append(f"{sc['id']}: present - {keep_} skipped: the scene's text names them")      # to him)
            pres = pres + keep_
        if sorted(pres) != sorted(sc.get("present") or []):
            ch.append(f"{sc['id']}: present {sc.get('present')} -> {pres}")
            if not dry:
                sc["present"] = pres
    # SPLITS, in time order, each on the scene that holds its moment at that point; the new scene takes the moment and
    # the following moments of that scene whose OWN place is the new one, then the scene RESUMES (a one-shot
    # flashback returns: 10-22, the Tower and Barley Fields memories inside the river scene)
    def own_place(x):
        return x.get("where_text") or x.get("where")

    def holder(t):
        return max((s_ for s_ in plan["scenes"] if _t0(s_, pstarts) <= t + 1e-6), key=lambda s_: _t0(s_, pstarts),
                   default=sc)
    span0 = (_t0(sc, pstarts), _t_end(sc, plan["scenes"], pstarts))
    for sp in sorted(ans.get("splits") or [], key=lambda x: (frames_by_id.get(x.get("moment")) or {}).get("_t0", 1e9)):
        fr = frames_by_id.get(sp.get("moment"))
        w = sp.get("where")
        if fr is None or not w:
            continue
        if w not in places and re.sub(r"\s*\((inside|outside)\)\s*$", "", w) not in places:
            ch.append(f"{sc['id']}: split at {sp.get('moment')} into unknown place {w!r}: skipped")
            continue
        side_ = str(sp.get("side") or "").lower()
        if side_ in ("inside", "outside") and video.scene_side(w, places.get(w, "")) not in ("", side_):
            # the other SIDE of the place asked for ("Outside Hulda's Forge", inside): that location's own place on that
            # side (the most shared name words: "The Old Forge"), else its variant (10-22 epilogue at the anvil)
            ww = video.place_words(w) - {"outside", "inside"}
            cands = [(len(video.place_words(n) & ww), n) for n in places
                     if n != w and video.scene_side(n, places.get(n, "")) == side_ and video.place_words(n) & ww]
            alt = max(cands)[1] if cands else f"{re.sub(r'\s*\((inside|outside)\)\s*$', '', w)} ({side_})"
            ch.append(f"{sc['id']}: split place {w!r} is not {side_}: -> {alt!r}")
            w = alt
            places.setdefault(w, "")
        if not (span0[0] - 1e-6 <= fr["_t0"] < span0[1]):
            continue                                 # a split only inside the scene under review (its span before)
        if any(x.get("where") == w and abs(_t0(x, pstarts) - (fr["_t0"] - 0.05)) < 0.2 for x in plan["scenes"]) or \
                holder(fr["_t0"]).get("where") == w:
            continue                                 # already split there (idempotent: one answer, one split)
        para = _para_at(pstarts, fr["_t0"])
        ch.append(f"{sc['id']}: SPLIT at {sp['moment']} (P{para}) -> {w!r} ({sp.get('side')}): {sp.get('why', '')}")
        if dry and not apply_splits:
            continue
        host = holder(fr["_t0"])
        new = {"id": "", "from": para, "t_split": round(float(fr["_t0"]) - 0.05, 2), "where": w,
               "side": sp.get("side") or video.scene_side(w, places.get(w, "")),
               "present": [c for c in (_canon(p, cast) for p in sp.get("present") or []) if c],
               "props": [], "first_words": "", "why": f"review: {sp.get('why', '')}"}
        later = sorted((x for x in frames_by_id.values() if x["_t0"] > fr["_t0"]
                        and x["_t0"] < _t_end(host, plan["scenes"], pstarts)), key=lambda x: x["_t0"])
        back = next((x for x in later if not (video.place_words(own_place(x)) & video.place_words(w))), None)
        scs = plan["scenes"]
        held_ = [x for x in frames_by_id.values() if holder(x["_t0"]) is host]
        if held_ and fr is min(held_, key=lambda x: x["_t0"]):
            old = dict(host)                         # from the host's first moment on: the host itself re-placed
            host.update({k: v for k, v in new.items() if k not in ("id", "from", "t_split")})
        else:
            old = dict(host)
            scs.append(new)
        # after the split's moments: the rest of the host's span re-segmented by each moment's OWN place (a moment that
        # names another known place opens it; otherwise the host's place resumes) - 10-22: flashbacks by the river
        # (Tower, then Barley Fields) return to the river; the King's-hall moment just before its paragraph is the hall
        cur = w
        for x in [x for x in later if back is not None and x["_t0"] >= back["_t0"]]:
            ow = own_place(x)
            nw = ow if (ow and ow in places and ow != old["where"]) else old["where"]
            if nw == cur:
                continue
            res = dict(old)
            if nw != old["where"]:
                res.update(where=nw, side=next((y["side"] for y in scs if y.get("where") == nw and y.get("side")), None)
                           or video.scene_side(nw, places.get(nw, "")) or old.get("side"), present=[])
            scs.append({**{k: v for k, v in res.items() if k != "id"}, "id": "",
                        "from": _para_at(pstarts, x["_t0"]), "t_split": round(float(x["_t0"]) - 0.05, 2),
                        "why": f"review: {res['where']} after {sp['moment']}"})
            ch.append(f"{sc['id']}: ...then {res['where']!r} at {x['shot']}")
            cur = nw
        scs.sort(key=lambda s_: _t0(s_, pstarts))
    for mid, fix in (ans.get("moments") or {}).items():
        fr = frames_by_id.get(mid)
        if fr is None or not isinstance(fix, dict):
            continue
        for k in ("prompt", "who", "props"):          # (the prompt first: a person added must be in it)
            if k not in fix or fix[k] in (None, "") or fix[k] == fr.get(k):
                continue
            v = fix[k]
            if k == "who":
                v = [c for c in (_canon(x, cast) for x in v) if c]
                if fr.get("who") and v and not (set(v) & set(fr["who"])):   # everyone replaced = another moment
                    ch.append(f"{mid}: who {fr['who']} -> {v} skipped: that replaces everyone (another moment)")
                    continue
                ptxt = ((fix.get("prompt") if isinstance(fix.get("prompt"), str) else None) or fr.get("prompt") or "").lower()
                extra = [x for x in v if x not in (fr.get("who") or [])
                         and not re.search(rf"\b{re.escape(x.split()[-1].lower())}", ptxt)]
                if extra:                            # (10-22: the King added to "Wide shot of the empty stone hall")
                    ch.append(f"{mid}: who + {extra} skipped: not in the moment's prompt")
                    v = [x for x in v if x not in extra]
                kinds = video.creature_kinds(cast)
                kept = [x for x in (fr.get("who") or []) if x not in v and (
                    re.search(rf"\b{re.escape(x.split()[-1].lower())}", ptxt)
                    or (x in kinds and re.search(rf"\b{kinds[x]}", ptxt)))]
                if kept:                             # (10-22: the dragon removed, "the dragon glides across the moon"
                    ch.append(f"{mid}: who - {kept} skipped: the prompt still shows them")    # left as the prompt)
                    v = v + kept
                if v == fr.get("who"):
                    continue
            if k == "prompt" and (not isinstance(v, str) or len(v) > 600):
                continue
            if k == "prompt":                        # a prop's own words stay (10-22: "the Star-iron Sword" -> "the
                gone = _identity_lost(fr.get("prompt") or "", v, fr.get("props") or [])   # sword" in 7 fight shots)
                if gone:
                    ch.append(f"{mid}: prompt change skipped: it drops the prop words {gone}")
                    continue
            if k == "props":
                v = _guard_props(fr.get("props") or [], v if isinstance(v, list) else [],
                                 (fix.get("prompt") if isinstance(fix.get("prompt"), str) else None) or fr.get("prompt") or "")
                if v is None:
                    ch.append(f"{mid}: props change {fix[k]!r} skipped: it drops or invents a prop family the prompt "
                              f"does not agree with")
                    continue
                if v == fr.get("props"):
                    continue
            dry_m = dry and not (apply_only and mid in apply_only)
            if not dry_m:                            # the planner's own version, kept once (a review change can be undone)
                fr.setdefault("orig", {kk: fr.get(kk) for kk in ("who", "props", "prompt", "motion", "must")})
            ch.append(f"{mid}: {k} {fr.get(k)!r} -> {v!r} ({fix.get('why', '')})" + ("  (LISTED)" if dry_m and dry else ""))
            if not dry_m:
                fr[k] = v
                if k == "prompt":                    # the MOTION follows the prompt (10-22: "the dragon glides" stayed
                    mo = fix.get("motion")            # as the motion of an "empty hall" shot)
                    if not isinstance(mo, str) or not mo.strip():
                        sents = re.split(r"(?<=[.!?])\s+", v.strip())
                        mo = " ".join(x for x in sents if not re.match(
                            r"(extreme |medium |wide |close|over|high|low|establishing|full|two)", x, re.I)) or v
                    fr["motion"] = mo
                    fr["must"], fr["checks"] = [], []  # the old picture's must-haves would demand the old content
                    fr["_touched"] = True
                else:
                    fr["_touched"] = True
    return ch


DIRECTION_VERSION = 5          # frames directed by this pass carry fr["direction"] = this (10-22: acts + beats + hands; 3 = 10-23 play through, creatures; 4 = short beats asked again; 5 = the fallen stay down)


def acts_pass(book, plan, frames, starts_, cast, url, model, log, only_scenes: set | None = None) -> dict:
    """Every moment's DIRECTION (tools/interaction.py): the acts (both hands of everyone who handles anything) and
    the beats, scene by scene, validated (cast names, the book's props, held items named in the text or the book's
    director's notes; nothing invented); a scene whose answer leaves a hand undirected is asked again once.
    -> {"acts": n moments with acts, "rejected": [lines], "undirected": [lines]}"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import interaction as IA
    import blockout_props as P_
    from tts_reader import video
    P_.load_book(str(book))
    lib = P_.library()
    props_aff = {k: sorted((v.get("affordances") or {}).keys()) + [j["name"] for j in v.get("joints") or []]
                 for k, v in lib.items() if isinstance(v, dict) and not k.startswith("_") and "__" not in k}
    notes = IA.director_notes(book)
    out = {"acts": 0, "rejected": [], "undirected": []}
    st = starts_()
    allp = {k for k, v in lib.items() if isinstance(v, dict) and not k.startswith("_")}   # (variants too)
    kinds = {k: (lib.get(k) or {}).get("kind") for k in allp}

    def clean(fr, x):
        acts, beats = IA.split_answer(x)
        txt = " ".join([fr.get("prompt") or "", fr.get("motion") or "", " ".join(map(str, fr.get("props") or [])),
                        " ".join(notes)])
        acts = [IA.resolve_props(a_, set(props_aff)) for a_ in acts if isinstance(a_, dict)]
        acts = [IA.resolve_props(a_, allp) for a_ in acts]
        ok, bad = IA.validate(acts, set(cast), allp, txt, kinds)
        return ok, beats, bad
    for sc in plan["scenes"]:
        if only_scenes is not None and sc.get("id") not in only_scenes:
            continue
        ms = [fr for fr in frames if video.scene_at_t(st, fr["_t"]) is sc]
        if not ms:
            continue
        ans = _ask(IA.acts_prompt(sc, ms, cast, props_aff, notes), url, model)
        res = {fr["shot"]: clean(fr, ans.get(fr["shot"])) for fr in ms}
        gaps = {m: IA.undirected(r[0]) for m, r in res.items() if IA.undirected(r[0])}
        short = {m: IA.short_of(r[1]) for m, r in res.items() if IA.short_of(r[1])}
        if gaps or short:                            # once more, with the undirected hands / short beats named
            log(f"  direction: {sc['id']}: undirected hands {gaps}, beats short of their result {short}: asked again")
            again = _ask(IA.acts_prompt(sc, [fr for fr in ms if fr["shot"] in gaps or fr["shot"] in short], cast,
                                        props_aff, notes, [f"{m}: {', '.join(g)}" for m, g in gaps.items()],
                                        [f"{m}: {' / '.join(b_)}" for m, b_ in short.items()]), url, model)
            for fr in ms:
                m = fr["shot"]
                if (m in gaps or m in short) and again.get(m) is not None:
                    r2 = clean(fr, again.get(m))
                    if len(IA.undirected(r2[0])) <= len(gaps.get(m, [])) and \
                            len(IA.short_of(r2[1])) <= len(short.get(m, [])) and \
                            (len(IA.undirected(r2[0])), len(IA.short_of(r2[1]))) < (len(gaps.get(m, [])), len(short.get(m, []))):
                        res[m] = r2
        for fr in ms:
            ok, beats, bad = res[fr["shot"]]
            fr["acts"] = ok
            fr["beats"] = beats
            fr["direction"] = DIRECTION_VERSION
            out["acts"] += bool(ok)
            for b in bad:
                out["rejected"].append(f"{fr['shot']}: {b}")
                log(f"  acts: {fr['shot']}: rejected {b}")
            for b_ in IA.short_of(beats):
                log(f"  !!! direction: {fr['shot']}: beat still short of its result: {b_}")
            for g in IA.undirected(ok):
                out["undirected"].append(f"{fr['shot']}: {g}")
                log(f"  !!! direction: {fr['shot']}: {g} hand still undirected")
            if ok:
                log(f"  acts: {fr['shot']}: " + "; ".join(
                    f"{a['do']} {a.get('who')} {a.get('hand', '')} {a.get('prop') or a.get('tool') or a.get('carried') or a.get('at') or ''}".replace("  ", " ")
                    + (f" -> {a['target']}" if a.get("target") else "") for a in ok)
                    + (f" | beats: {' / '.join(beats)}" if beats else ""))
    log(f"direction: {out['acts']} moment(s) with acts, {len(out['rejected'])} act(s) rejected, "
        f"{len(out['undirected'])} hand(s) left undirected")
    return out


def direct(book, stem: str, ids: list | None = None, log=print, force: bool = False) -> dict:
    """The director pass alone (no scene review): every scene holding one of `ids` (all when None) is directed again
    when a moment there lacks this DIRECTION_VERSION (or force). anim/keyframes.json rewritten."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tts_reader import bookcaster as BC
    from tts_reader import video
    book = Path(book)
    f = next(x for x in BC.chapter_files(book) if x.stem == stem)
    work = BC._video_work(book, f)
    pj, kp = work / "plan.json", work / "anim" / "keyframes.json"
    plan, kf = json.loads(pj.read_text()), json.loads(kp.read_text())
    paras = video.paragraphs(BC._chapter_text(f))
    tl = json.loads((book / "work" / f"{f.stem}.timeline.json").read_text())
    pstarts = video.paragraph_times(paras, tl)
    cast = (plan.get("look") or {}).get("characters") or {}
    ents = json.loads((work / "flow3d" / "plan.json").read_text())["entries"]
    tof = {e["id"]: float(e["t"]) + 0.5 * float(e.get("dur") or 0) for e in ents}
    frames = [fr for fr in kf.get("frames") or [] if fr.get("shot") in tof]
    for fr in frames:
        fr["_t"] = tof[fr["shot"]]

    def starts_():
        for i, s_ in enumerate(plan["scenes"]):
            s_["id"] = f"s{i + 1:02d}"
            s_["t0"] = round(0.0 if i == 0 else max(pstarts[s_["from"]], float(s_.get("t_split") or 0.0)), 2)
        return [(s_["t0"], s_) for s_ in plan["scenes"]]
    st = starts_()
    want = set(ids) if ids else {fr["shot"] for fr in frames}
    scenes = set()
    for fr in frames:
        if fr["shot"] in want and (force or fr.get("direction") != DIRECTION_VERSION):
            sc = video.scene_at_t(st, fr["_t"])
            if sc:
                scenes.add(sc["id"])
    rep = {"acts": 0, "rejected": [], "undirected": [], "scenes": sorted(scenes)}
    if scenes:
        log(f"direction: scenes {sorted(scenes)}")
        rep.update(acts_pass(book, plan, frames, starts_, cast, BC.LLM_URL, BC.LLM_MODEL, log, only_scenes=scenes))
    for fr in frames:
        fr.pop("_t", None)
    if scenes:
        kp.write_text(json.dumps(kf, indent=1, ensure_ascii=False))
    return rep


def review(book, f, work: Path, shots: list, log=print) -> dict:
    """The plan review of one chapter (anim/keyframes.json + plan.json scenes rewritten). -> report"""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tts_reader import bookcaster as BC
    from tts_reader import video
    pj, kp = work / "plan.json", work / "anim" / "keyframes.json"
    plan, kf = json.loads(pj.read_text()), json.loads(kp.read_text())
    text = BC._chapter_text(f)
    paras = video.paragraphs(text)
    tl = json.loads((Path(book) / "work" / f"{f.stem}.timeline.json").read_text())
    pstarts = video.paragraph_times(paras, tl)
    cast = (plan.get("look") or {}).get("characters") or {}
    ap = Path(book) / "work" / "video" / "stage" / "assets.json"
    import flow3d as F_
    built = F_.known_places(book)                    # with each set's side (its world.json kind)
    places = {**built, **((plan.get("look") or {}).get("places") or {}), **{k: v for k, v in built.items() if v}}
    for s_ in plan.get("scenes") or []:
        places.setdefault(s_["where"], "")
    tof = {s_["id"]: (s_["t"], s_["t"] + 0.5 * float(s_.get("dur") or 0)) for s_ in shots}
    frames = [fr for fr in kf.get("frames") or [] if fr.get("shot") in tof]
    for fr in frames:
        fr["_t0"], fr["_t"] = tof[fr["shot"]]
    rep = {"rules": [], "pass1": [], "pass2_listed": []}

    def starts_():
        for i, s_ in enumerate(plan["scenes"]):
            s_["id"] = f"s{i + 1:02d}"
            s_["t0"] = round(0.0 if i == 0 else max(pstarts[s_["from"]], float(s_.get("t_split") or 0.0)), 2)
            nxt = plan["scenes"][i + 1]["from"] - 1 if i + 1 < len(plan["scenes"]) else len(paras) - 1
            s_["to"] = nxt
        return [(s_["t0"], s_) for s_ in plan["scenes"]]

    def assign():
        st = starts_()
        for fr in frames:
            sc = video.scene_at_t(st, fr["_t"])
            if sc:
                for ln in video.apply_scene(fr, sc, cast, places):
                    log(f"  scene {sc['id']}: {fr['shot']}: {ln}")
        return st

    st = assign()
    rep["rules"] = rules(plan, frames, None, st, tl, cast, log)
    url, model = BC.LLM_URL, BC.LLM_MODEL
    for p_ in (1, 2):                                 # pass 2 re-reads every scene, applies nothing
        for sc in list(plan["scenes"]):
            ms = [fr for fr in frames if video.scene_at_t(starts_(), fr["_t"]) is sc]
            if not ms:
                continue
            ans = _ask(_scene_prompt(sc, paras, ms, places, cast), url, model)
            touched = {fr["shot"] for fr in frames if fr.get("_touched")} if p_ == 2 else None
            ch = apply_answer(ans, sc, plan, {fr["shot"]: fr for fr in frames}, places, cast, pstarts, log,
                              dry=(p_ == 2), paras=paras, apply_only=touched, apply_splits=True)
            for ln in ch:                            # pass 2: follow-ups on moments pass 1 changed are applied,
                log(f"  review pass {p_}: {ln}")     # the rest LISTED
            rep["pass1" if p_ == 1 else "pass2_listed"] += ch
        if p_ == 1:
            assign()
    plan["scenes"] = video.merge_scenes(sorted(plan["scenes"], key=lambda s_: _t0(s_, pstarts)))
    assign()                                         # the moments' scene ids after the merge
    rep["acts"] = acts_pass(book, plan, frames, starts_, cast, url, model, log)
    for fr in frames:
        fr.pop("_t", None)
        fr.pop("_t0", None)
        fr.pop("_touched", None)
    left = [fr["shot"] for fr in frames if fr.get("scene_conflict")]
    if left:
        log(f"  !!! review: {len(left)} moment(s) still disagree with their scene's place: {left}")
    rep["conflicts_left"] = left
    pj.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    kp.write_text(json.dumps(kf, indent=1, ensure_ascii=False))
    (Path(book) / "work" / "video" / "refs").mkdir(parents=True, exist_ok=True)
    (Path(book) / "work" / "video" / "refs" / f"plan_review_{f.stem}.json").write_text(json.dumps(rep, indent=1))
    log(f"plan review: {len(rep['rules'])} rule fix(es), {len(rep['pass1'])} model fix(es), "
        f"{len(rep['pass2_listed'])} more listed by the second pass")
    return rep
