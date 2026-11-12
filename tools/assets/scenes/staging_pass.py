"""Staging pass: the gaps a director's plan leaves, as small closed questions to the language model (stdlib).

    python tools/assets/scenes/staging_pass.py PLAN.json SHOTS.txt CAST.json TIMELINE.json SET_DIR OUT.json
           [--cache DIR] [--url http://127.0.0.1:18100/v1/chat/completions] [--repeats 3]

The pass never overwrites a fact the plan gives; it asks only for these gaps, each its own question with a closed
answer set built from data, "not stated" always among the options (code then keeps the previous state):
  in_room     shots whose plan lists nobody: which of the cast are in the room (a subset of the cast list)
  move_end    action shots: for each person the plan lists, where they are when the shot ends (one of the set's
              named things or seats)
  worn_start  the scene's first shot, each person: one of that figure's worn-state variants
  worn_end    action shots, each listed person: one of that figure's variants
  pose        action shots, each listed person: one of the pose list, or "none"
  camera      talk shots: one of the camera kinds
Each question carries the smallest passage span that covers the shot (the chapter's timeline from the shot before
to the end of this one) and the carried state before it; a change of worn state is asked from the shot's OWN span
only (its start to the next shot's start), and of the shots that claim the same change the first action shot wins. Temperature 0 and a fixed seed; each question is asked
--repeats times and an answer is kept only when every repeat gives it, else "not stated". Answers are cached on disk
keyed by the prompt and the repeat (a new --cache directory is a cold run).
Output: {"in_room": {shot: [names]}, "move_end": {shot: {name: thing}}, "worn_start": {name: variant},
         "worn_end": {shot: {name: variant}}, "pose": {shot: {name: pose}}, "camera": {shot: kind},
         "_asked": n, "_calls": n, "_cached": n, "_split": n}  for from_plan.py --pass.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "redraw"))
import config as C  # noqa: E402
import seating  # noqa: E402
import setfacts  # noqa: E402

# what a person is doing as the shot starts (the posture, sitting or standing, is the plan's own): asked with
# postures in the same list, the model answered "seated" for a shot whose action starts lifting a veil (11-12)
POSES = ["stand_hand_on_mantel", "gesture_point", "veil_lift", "look_at_sleeve", "reach_to_forearm"]
# gaze is not a pose: the listener/speaker rule sets it. Plain words for each, for the owner question
POSE_WORDS = {"stand_hand_on_mantel": "rests a hand on the mantelpiece", "gesture_point": "points at something",
              "veil_lift": "lifts a veil", "look_at_sleeve": "looks down at a sleeve",
              "reach_to_forearm": "lays a hand on another person's forearm"}
CAMERAS = ["two_shot", "over_the_shoulder", "single", "close"]
NOT = "not stated"
SEED = 7


def span(timeline: list, t0: float, t1: float) -> str:
    """The chapter's lines (narration and speech) between two times."""
    return "\n".join(f"[{s.get('speaker', 'narrator')}] {s.get('text', '')}" for s in timeline
                     if t0 - 0.5 <= s.get("t", 0) <= t1 + 0.5)


def audio_times(timeline: list, es: dict, ids: list) -> dict:
    """Each shot's time in the chapter's audio (the plan's times are the video's, with pauses between): a talk shot by
    its own spoken line found in the timeline; the others between their neighbours by the video's time."""
    lt = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    anchors = []
    for sid in ids:
        e = es.get(sid, {})
        w = lt(e.get("words") or e.get("line"))[:30]
        if not w:
            continue
        seg = next((x for x in timeline if lt(x.get("text"))[:30] and (lt(x["text"]).startswith(w[:20]) or
                                                                      w.startswith(lt(x["text"])[:20]))), None)
        if seg:
            anchors.append((e.get("t", 0.0), seg["t"]))
    anchors.sort()
    out = {}
    for sid in ids:
        v = es.get(sid, {}).get("t", 0.0)
        if not anchors:
            out[sid] = v
            continue
        before = [x for x in anchors if x[0] <= v]
        after = [x for x in anchors if x[0] > v]
        if before and after:
            (v0, a0), (v1, a1) = before[-1], after[0]
            out[sid] = a0 + (a1 - a0) * (v - v0) / max(1e-6, v1 - v0)
        elif before:
            out[sid] = before[-1][1] + (v - before[-1][0])
        else:
            out[sid] = after[0][1] - (after[0][0] - v)
    return out


def variants(library: Path, base: str) -> list:
    """A figure's worn-state choices: "usual" and its library variants' suffixes (base_<suffix>, not stances)."""
    out = {"usual"}
    for d in (library / "character").glob(f"{base}_*"):
        suf = d.name[len(base) + 1:]
        if "seated" in suf:
            suf = suf.split("_", 2)[-1] if suf.count("_") >= 2 else ""
        if suf and not suf.startswith(("seated", "walk")):
            out.add(suf)
    return sorted(out)


class Asker:
    def __init__(self, url: str, cache: Path, repeats: int):
        self.url, self.cache, self.repeats = url, cache, repeats
        self.asked = self.calls = self.cached = self.split = 0

    def one(self, prompt: str, rep: int) -> str:
        key = hashlib.sha256(f"{rep}\n{prompt}".encode()).hexdigest()[:24]
        f = self.cache / f"{key}.txt"
        if f.exists():
            self.cached += 1
            return f.read_text()
        body = {"model": "default", "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
                "seed": SEED, "max_tokens": 60, "chat_template_kwargs": {"enable_thinking": False}}
        req = Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=600) as r:
            txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
        self.calls += 1
        self.cache.mkdir(parents=True, exist_ok=True)
        f.write_text(txt)
        return txt

    def choose(self, prompt: str, options: list) -> str:
        """One of the options (or "not stated"), kept only when every repeat gives the same."""
        self.asked += 1
        got = []
        for rep in range(self.repeats):
            a = self.one(prompt, rep).strip().strip('."\'`').lower()
            head = lambda o: o.split(" (")[0].lower()          # an option's name, without its description
            pick = next((o for o in options if a in (o.lower(), head(o))), None) or next(
                (o for o in sorted(options, key=lambda o: len(head(o)), reverse=True)
                 if re.search(r"\b" + re.escape(head(o)) + r"\b", a)), NOT)
            got.append(pick)
        if len(set(got)) > 1:
            self.split += 1
            return NOT
        return got[0]

    def subset(self, prompt: str, options: list) -> list | str:
        self.asked += 1
        got = []
        for rep in range(self.repeats):
            a = self.one(prompt, rep).lower()
            if NOT in a:
                got.append(NOT)
                continue
            got.append(tuple(sorted(o for o in options if o.lower() in a)))
        if len(set(got)) > 1:
            self.split += 1
            return NOT
        return NOT if got[0] == NOT else list(got[0])


HEAD = ("You help stage a scene of a story for a film. Answer from the passage only. If the passage does not say, "
        "answer exactly: " + NOT + ".\n")


def q_choose(psg: str, state: str, what: str, options: list) -> str:
    return (HEAD + "Passage:\n<<<\n" + psg + "\n>>>\nWhere things stood just before: " + state + "\nQuestion: " + what
            + "\nAnswer with exactly one of these, nothing else: " + "; ".join(options + [NOT]))


def q_subset(psg: str, state: str, what: str, options: list) -> str:
    return (HEAD + "Passage:\n<<<\n" + psg + "\n>>>\nWhere things stood just before: " + state + "\nQuestion: " + what
            + "\nAnswer with the names from this list that apply, separated by commas, nothing else (or " + NOT
            + "): " + "; ".join(options))


def main():
    a = sys.argv[1:]
    plan = json.loads(Path(a[0]).read_text())
    ids = [x.strip() for x in Path(a[1]).read_text().splitlines() if x.strip()]
    cast = json.loads(Path(a[2]).read_text())
    timeline = json.loads(Path(a[3]).read_text())
    set_dir = Path(a[4])
    out = Path(a[5])
    cache = Path(a[a.index("--cache") + 1]) if "--cache" in a else out.parent / "llm_cache"
    url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:18100/v1/chat/completions"
    reps = int(a[a.index("--repeats") + 1]) if "--repeats" in a else 3
    library = Path(C.LIBRARY)
    es = {e["id"]: e for e in plan.get("_es", [])}
    things = [t for t in setfacts.things(set_dir) if t["kind"] in ("fixture", "window", "door")]
    place_opts = [f"{t['name']} ({t['words'].replace('(', '').replace(')', '')}"
                  + (f", {t['place']}" if t.get("place") else "") + ")" for t in things]
    names = list(cast)
    var = {nm: variants(library, c.get("base") or c.get("id")) for nm, c in cast.items()}
    ask = Asker(url, cache, reps)
    res = {"in_room": {}, "move_end": {}, "worn_start": {}, "worn_end": {}, "pose": {}, "camera": {}}
    state = {}                                    # name -> place words, carried from the plan's own facts
    worn_now = {}                                 # name -> worn state at the scene's start
    claims = {}                                   # name -> [(shot, worn when that shot ends)], raw
    worn_at = lambda nm: (claims.get(nm) or [(None, worn_now.get(nm, NOT))])[-1][1]
    t0 = time.time()
    at = audio_times(timeline, es, ids)
    for n, sid in enumerate(ids):
        sh, e = plan[sid][0], es.get(sid, {})
        a0 = at[ids[n - 1]] if n else at[sid] - 20.0             # from the shot before to this shot's end
        a1 = at[ids[n + 1]] if n + 1 < len(ids) else at[sid] + max(8.0, e.get("speech_s") or e.get("dur") or 0)
        lo, hi = min(a0, at[sid]) - 4.0, max(a1, at[sid] + 4.0)
        psg = span(timeline, lo, hi)
        while len(psg) < 300 and hi - lo < 60:   # a span too short to say anything: widen it a little each side
            lo, hi = lo - 3.0, hi + 3.0
            psg = span(timeline, lo, hi)
        # the shot's OWN span (from its start to the next shot's start, never overlapping a neighbour): a change
        # of what someone wears is taken only from here, so a change a later shot shows is not claimed early
        own_lo, own_hi = at[sid] - 0.5, (at[ids[n + 1]] if n + 1 < len(ids) else at[sid] + 8.0) - 0.5
        own = span(timeline, own_lo, own_hi) or NOT
        for p in sh.get("people") or []:
            state[p["name"]] = f"{p.get('pose')} {p.get('at')}"
        st = "; ".join(f"{k}: {v}" for k, v in state.items()) or NOT
        if n == 0:
            for nm in names:
                res["worn_start"][nm] = ask.choose(q_choose(psg, st, f"What is {nm} wearing at this moment, among "
                                                            "these choices?", var[nm]), var[nm])
                if res["worn_start"][nm] != NOT:
                    worn_now[nm] = res["worn_start"][nm]
        if not sh.get("people"):
            res["in_room"][sid] = ask.subset(q_subset(psg, st, "Which of these people are in the room at this "
                                                      "moment (shown or not)?", names), names)
        if sh.get("kind") == "talk":
            res["camera"][sid] = ask.choose(q_choose(psg, st, "This shot shows someone speaking. Which kind of camera "
                                                     "framing does the passage's moment call for?", CAMERAS), CAMERAS)
        if sh.get("kind") == "action":
            act = sh.get("action") or ""
            named = json.dumps(sh.get("things") or {}, ensure_ascii=False)
            for p in sh.get("people") or []:
                nm = p["name"]
                q = f"The shot's action: {act}\nThings the director names in it: {named}\n"
                # a move only where the plan's own posture changes over the shot (sitting -> standing or back)
                if (sh.get("pose_end") or {}).get(nm) not in (None, p.get("pose")):
                    # the places offered by number and description only: offered by the set's own names, the
                    # model took the plain-named "armchair" over the described leather one every time (11-11)
                    nums = [f"place {i + 1}: {o.split(' (', 1)[1].rstrip(')')}" for i, o in enumerate(place_opts)]
                    keys = [f"place {i + 1}" for i in range(len(place_opts))]
                    end = ask.choose(q_choose(psg, st, q + f"Where is {nm} when this shot ends? Choose the place's "
                                              "number from this list:\n" + "\n".join(nums), keys), keys)
                    if end != NOT:
                        res["move_end"].setdefault(sid, {})[nm] = things[keys.index(end)]["name"]
                w = ask.choose(q_choose(own, st + f"; {nm} wears: {worn_at(nm)}",
                                        q + f"What is {nm} wearing when this shot ends, among these choices?",
                                        var[nm]), var[nm])
                if w != NOT:
                    claims.setdefault(nm, []).append((sid, w))
                # from the shot's OWN span only (as the worn state): a later shot's action is not claimed early
                ps = ask.choose(q_choose(own, st, q + f"{nm} is {p.get('pose') or 'in the shot'}. Most shots begin "
                                                      f"with no special action. Which of these does {nm} begin to do "
                                                      "as this shot starts?", ["none"] + POSES), ["none"] + POSES)
                if ps != NOT:
                    res["pose"].setdefault(sid, {})[nm] = ps
    # a change is taken once: of the shots in a row that claim the same new state, the first action shot (the one
    # that shows it) wins and the other claims are "not stated"
    for nm, cl in claims.items():
        cur, run = worn_now.get(nm), []
        for sid, w in cl + [(None, None)]:
            if run and w != run[-1][1]:
                # own spans already keep a later shot's change out of an earlier one; of the shots that then name
                # the same new state, the first action shot is where it happens, the rest only restate it
                pick = next((x for x in run if plan[x[0]][0].get("kind") == "action"), run[0])
                res["worn_end"].setdefault(pick[0], {})[nm] = pick[1]
                cur, run = pick[1], []
            if w is not None and w != cur:
                run.append((sid, w))
    res["_worn_claims"] = claims
    # an action is one shot's stated action: when several shots answer the same action for the same person, the
    # shot whose plan line states it owns it (a closed question over those shots' own action lines), the rest none
    by = {}
    for sid, pp in res["pose"].items():
        for nm, ps in pp.items():
            if ps in POSES:
                by.setdefault((nm, ps), []).append(sid)
    res["_action_owner"] = {}
    for (nm, ps), sids in by.items():
        if len(sids) < 2:
            continue
        opts = [f"shot {i + 1}" for i in range(len(sids))]
        lines = "\n".join(f"shot {i + 1}: {plan[x][0].get('action') or plan[x][0].get('picture') or ''}"
                          for i, x in enumerate(sids))
        own_q = (HEAD + "The film's shots, each with its own action line:\n" + lines + "\nQuestion: In which shot "
                 f"does {nm} {POSE_WORDS[ps]}, as that shot's own action line states?\nAnswer with exactly one of "
                 "these, nothing else: " + "; ".join(opts + [NOT]))
        got = ask.choose(own_q, opts)
        keep = sids[opts.index(got)] if got in opts else None
        res["_action_owner"][f"{nm}: {ps}"] = keep
        for x in sids:
            if x != keep:
                res["pose"][x][nm] = "none"
    res.update(_asked=ask.asked, _calls=ask.calls, _cached=ask.cached, _split=ask.split, _s=round(time.time() - t0, 1))
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("_asked", "_calls", "_cached", "_split", "_s")}))


if __name__ == "__main__":
    main()
