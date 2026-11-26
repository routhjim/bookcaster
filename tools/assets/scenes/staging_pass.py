"""Staging pass: the gaps a director's plan leaves, as small closed questions to the language model (stdlib).

    python tools/assets/scenes/staging_pass.py PLAN.json SHOTS.txt CAST.json TIMELINE.json SET_DIR OUT.json
           [--cache DIR] [--url http://127.0.0.1:18100/v1/chat/completions] [--repeats 3] [--looks LOOK.json]

The pass never overwrites a fact the plan gives; it asks only for these gaps, each its own question with a closed
answer set built from data, "not stated" always among the options (code then keeps the previous state):
  in_room     shots whose plan lists nobody: which of the cast are in the room (a subset of the cast list)
  move_end    action shots: for each person the plan lists, where they are when the shot ends (one of the set's
              named things or seats)
  worn_start  the scene's first shot, each person: one of that figure's worn-state variants
  worn_end    action shots, each listed person: one of that figure's variants
  pose        action shots, each listed person: one of the pose list, or "none"
  camera      talk shots: one of the camera kinds
  face        every shot that lists two people or more, or someone at a named thing of the set, each of them: whom
              of the shot's other people they face (by name), the thing they are at (offered only at a named thing,
              never to a creature), or nobody in particular; from the shot's own words and the people's look phrases
  creature_pose  every shot, each listed person who is a library creature (cast "creature": true): one of the poses
              its own rig's clip holds (creature.py), from the shot's own words
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
import creature  # noqa: E402
import seating  # noqa: E402
import setfacts  # noqa: E402
import wardrobe  # noqa: E402

# what a person is doing as the shot starts (the posture, sitting or standing, is the plan's own): asked with
# postures in the same list, the model answered "seated" for a shot whose action starts lifting a veil (11-12)
POSES = ["stand_hand_on_mantel", "gesture_point", "veil_lift", "look_at_sleeve", "reach_to_forearm"]
# the work a person does with a tool or with their hands at what is in front of them, whatever the task (11-17: a
# smith stood arms down beside his anvil in every key); offered by these plain names, repeatable shot after shot
WORK_POSES = ["tool_raised", "striking_down", "hands_at_work"]
# gaze is not a pose: the listener/speaker rule sets it. Plain words for each, for the owner question
POSE_WORDS = {"stand_hand_on_mantel": "rests a hand on the mantelpiece", "gesture_point": "points at something",
              "veil_lift": "lifts a veil", "look_at_sleeve": "looks down at a sleeve",
              "reach_to_forearm": "lays a hand on another person's forearm",
              "tool_raised": "holds a tool raised high, ready to bring it down on the work in front of them",
              "striking_down": "brings a tool down on the work in front of them",
              "hands_at_work": "works with both hands on something in front of them at waist height"}
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
        wf = lt(e.get("words") or e.get("line"))
        w = wf[:30]
        if not w:
            continue
        # a line found by its opening: the whole of a short line, or a long enough opening of both (a two-letter
        # line of the chapter matched the opening of any shot's words and sent the shot there, 11-12); and the
        # chapter runs forward, so the first such line at or after the last one found
        def hit(x):
            t = lt(x.get("text"))
            return bool(t) and (t == wf or (min(len(t), len(w)) >= 12 and (t.startswith(w[:20])
                                                                           or w.startswith(t[:20]))))
        last = anchors[-1][1] if anchors else float("-inf")
        seg = next((x for x in timeline if x.get("t", 0) >= last and hit(x)), None)
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
    """A figure's worn-state choices: "usual", its library variants recorded as such ("variant_of": base, named
    base_<suffix>), and the older unrecorded ones (base_<suffix>, not stances: seated, lying, walking)."""
    out = {"usual"}
    for d in (library / "character").glob(f"{base}_*"):
        suf = d.name[len(base) + 1:]
        p = d / "character.json"
        rec = {}
        if p.exists():
            try:
                rec = json.loads(p.read_text())
            except ValueError:
                rec = {}
        if rec.get("variant_of"):
            if rec["variant_of"] == base and not wardrobe.is_stance(suf):
                out.add(suf)
            continue
        if rec.get("stance") or any(d.name.startswith(f"{x}_") for x in recorded(library, base)):
            continue                              # a stance, or anything of a recorded variant (its stances)
        if "seated" in suf:
            suf = suf.split("_", 2)[-1] if suf.count("_") >= 2 else ""
            if suf.isdigit():
                suf = ""
        if suf and not wardrobe.is_stance(suf):
            out.add(suf)
    return sorted(out)


def recorded(library: Path, base: str) -> list:
    """The ids of base's recorded wardrobe variants ("variant_of": base)."""
    out = []
    for p in (library / "character").glob(f"{base}_*/character.json"):
        try:
            if json.loads(p.read_text()).get("variant_of") == base:
                out.append(p.parent.name)
        except ValueError:
            pass
    return out


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


NOBODY = "nobody in particular"
AT_THING = "the thing they are at"


def q_face(sh: dict, nm: str, people: list, looks: dict, at_thing: bool = True) -> str:
    """Whom a person of the shot faces as it starts: the shot's own words, its people by name and by the look phrase
    the words call them, and the general rule (between people, each faces the one they act with)."""
    import shot_people
    who = []
    for p in people:
        ph = shot_people.phrase(looks.get(p["name"], ""), {"creature": True} if p["name"] in looks.get("_creatures", ())
                                else {})
        who.append(f"- {p['name']}" + (f" (the words call them {ph})" if ph else "") + f": {p.get('pose') or ''}, "
                   f"{p.get('at') or ''}")
    return ("You help stage a shot of a film. The shot's own words: "
            + " ".join(str(sh.get(k) or "") for k in ("keyshot", "action") if sh.get(k)).strip()
            + "\nThe people in the shot:\n" + "\n".join(who)
            + (f"\nQuestion: whom or what does {nm}'s body face as this shot starts? When the shot's action is "
               f"{nm}'s own work on, at or with the thing of the place {nm} stands at (the place above names it), {nm} "
               "faces that thing, whoever else is near: the answer is " + AT_THING + ". Otherwise, when the shot's "
               "action " if at_thing else f"\nQuestion: whom does {nm}'s body face as this shot starts? When the shot's "
               "action ")
            + "is between people (one acts on, at, against or with another), each of them faces the one they act with, "
            "whatever side of the picture they are on. When the person acts with nobody of the shot and with no thing "
            "they stand at, or looks away, the answer is " + NOBODY + ".\nAnswer with exactly one of these, nothing "
            "else: " + "; ".join([p["name"] for p in people if p["name"] != nm] + ([AT_THING] if at_thing else [])
                                 + [NOBODY]))


NEARS = ["dancing together", "touching", "close", "conversing", "apart"]


DANCE = re.compile(r"\bdanc(?:e|es|ed|ing)\b|\bwaltz|\breel\b|\bquadrille|\bminuet", re.I)


def in_a_dance(sh: dict, a: str, b: str) -> bool:
    """Whether the plan has both of these two in a dance: each one's own place or pose in the shot names it."""
    rows = {q.get("name"): f"{q.get('pose') or ''} {q.get('at') or ''}" for q in sh.get("people") or []}
    return all(DANCE.search(rows.get(n, "")) for n in (a, b))


def q_near(psg: str, sh: dict, a: str, b: str) -> str:
    """How near two people of a shot who are turned to each other stand as it starts: what they are to each other
    and what they are doing sets it, never one distance for every pair (two who danced as partners at a ball were
    stood a formal pace apart with their arms at their sides, as any two who talk, 11-26)."""
    return ("You help stage a shot of a film.\nThe story there:\n<<<\n" + psg + "\n>>>\nThe shot's own words: "
            + " ".join(str(sh.get(k) or "") for k in ("keyshot", "action", "setting") if sh.get(k)).strip()
            + "\nWhere the shot has them:\n"
            + "\n".join(f"- {q.get('name')}: {q.get('pose') or ''}, {q.get('at') or ''}" for q in sh.get("people") or []
                        if q.get("name") in (a, b))
            + f"\nQuestion: how near to each other are {a} and {b} as this shot starts?\n"
            "- dancing together: they are partners in a dance that is still going on (they stand in the dance, in "
            "its set or on its floor, and nothing says they have left it or that it has ended), also while they talk\n"
            "- touching: one holds, leads, steadies, embraces or takes the hand of the other\n"
            "- close: nearer than strangers stand, without touching: a private, tender or heated exchange, a whisper\n"
            "- conversing: the ordinary distance of two who talk\n"
            "- apart: across a room from each other, or one keeping away from the other\n"
            "Answer with exactly one of these, nothing else: " + "; ".join(NEARS))


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
    # the book's looks (--looks LOOK.json): the shot words name people by their look phrase, not by name
    looks = (json.loads(Path(a[a.index("--looks") + 1]).read_text()).get("characters") or {}) if "--looks" in a else {}
    es = {e["id"]: e for e in plan.get("_es", [])}
    things = [t for t in setfacts.things(set_dir) if t["kind"] in ("fixture", "window", "door")]
    np_ = set_dir / "named_places.json"            # the set's own map of the plan's place phrases to its things
    set_named = json.loads(np_.read_text()) if np_.exists() else {}
    place_opts = [f"{t['name']} ({t['words'].replace('(', '').replace(')', '')}"
                  + (f", {t['place']}" if t.get("place") else "") + ")" for t in things]
    names = list(cast)
    crt = {nm: r_ for nm, c in cast.items() if c.get("creature") and (r_ := creature.rig(library, c["id"]))}
    looks = dict(looks, _creatures=[nm for nm, c in cast.items() if c.get("creature")])
    var = {nm: variants(library, c.get("base") or c.get("id")) for nm, c in cast.items()}
    ask = Asker(url, cache, reps)
    res = {"in_room": {}, "move_end": {}, "worn_start": {}, "worn_end": {}, "pose": {}, "camera": {}, "exits": {},
           "enters": {}, "creature_pose": {}}
    gone = set()                                  # people the passage has taken out of the room
    state = {}                                    # name -> place words, carried from the plan's own facts
    worn_now = {}                                 # name -> worn state at the scene's start
    claims = {}                                   # name -> [(shot, worn when that shot ends)], raw
    now = {}                                      # name -> worn state as things stand (for the questions)
    worn_at = lambda nm: now.get(nm, NOT)
    t0 = time.time()
    at = audio_times(timeline, es, ids)
    # the book's wardrobe (the planner's change of clothes over the whole story, on the chapter's clock: the cast
    # file's "wardrobe"): the outfit in force at each shot's time is where the worn state starts; a change the
    # passage shows within the place still moves it, until the wardrobe's next change
    ward = {nm: c["wardrobe"] for nm, c in cast.items() if c.get("wardrobe") and nm not in crt}
    dflt = {nm: {} for nm in ward}                # name -> {shot: the wardrobe's outfit, where it changes}
    for nm, rows in ward.items():
        prev = None
        for sid in ids:
            v = wardrobe.variant_at(rows, wardrobe.shot_time(sid, at[sid]))
            v = v if v in var[nm] else "usual"    # (an outfit with no figure: the usual one, said in words)
            if v != prev:
                dflt[nm][sid] = v
            prev = v
    res["worn_plan"] = {}
    for n, sid in enumerate(ids):
        for nm in ward:
            if n and sid in dflt[nm]:
                now[nm] = dflt[nm][sid]
                res["worn_plan"].setdefault(sid, {})[nm] = dflt[nm][sid]
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
        # a creature's pose, from the shot's own words (its posture words name no pose of its clip): the clip's pose
        # nearest to how the shot shows it as it starts
        for p in sh.get("people") or []:
            if p["name"] not in crt:
                continue
            opts = [creature.pose_words(x) for x in crt[p["name"]]["poses"]]
            cp = ask.choose("You help stage a shot of a film. The shot's own words: "
                            + str(sh.get("keyshot") or sh.get("picture") or "") + " " + str(sh.get("action") or "")
                            + f"\n{p['name']} is {p.get('pose') or 'in the shot'}, {p.get('at') or ''}. {p['name']} is "
                            "staged as a model that can hold only the poses listed below. Question: which of them is "
                            f"closest to how {p['name']} is shown as this shot starts? Where none matches, take the "
                            "one nearest to the body's shape the words describe.\nAnswer with exactly one of these, "
                            "nothing else: " + "; ".join(opts), opts)
            if cp in opts:
                res["creature_pose"].setdefault(sid, {})[p["name"]] = crt[p["name"]]["poses"][opts.index(cp)]
        if n == 0:
            for nm in names:
                if nm in crt:                     # (a creature wears nothing of a figure's wardrobe)
                    continue
                if nm in ward:                    # the book's wardrobe says what they have on
                    res["worn_start"][nm] = worn_now[nm] = now[nm] = dflt[nm][sid]
                    continue
                res["worn_start"][nm] = ask.choose(q_choose(psg, st, f"What is {nm} wearing at this moment, among "
                                                            "these choices?", var[nm]), var[nm])
                if res["worn_start"][nm] != NOT:
                    worn_now[nm] = now[nm] = res["worn_start"][nm]
            # who is already in the place when the scene opens, and where the passage puts them (closed questions
            # over the cast and the set's numbered places): a person the passage has there is present from the
            # scene's first shot even when the plan's lists leave them out ("a lady ... who had been sitting in
            # the window, rose as we entered": the plan first listed her a shot later, 11-12)
            nums = [f"place {i + 1}: {o.split(' (', 1)[1].rstrip(')')}" for i, o in enumerate(place_opts)]
            keys = [f"place {i + 1}" for i in range(len(place_opts))]
            for nm in names:
                there = ask.choose(q_choose(psg, st, f"When this scene opens in this place, is {nm} already there, "
                                                     "before anyone else comes in?", ["yes", "no"]), ["yes", "no"])
                if there != "yes":
                    continue
                where = ask.choose(q_choose(psg, st, f"Where does the passage put {nm} when the scene opens? Choose the "
                                                     "place's number from this list:\n" + "\n".join(nums), keys), keys)
                res.setdefault("present_at_start", {})[nm] = things[keys.index(where)]["name"] if where in keys else NOT
        # presence is like the worn state: a person stays in the room until this shot's OWN span says they go out
        # (and stays out until it says they come in); not being mentioned is not leaving (asked "who is in the
        # room", the model dropped a man not named in a shot's lines, 11-12). What a speaker tells of is not the
        # scene: a woman telling how she ran from her room that night was sent out of the room she told it in
        here = [nm for nm in names if nm not in gone]
        out_ = ask.subset(q_subset(own, st, "Which of these people go out of the room during this passage, in "
                                             "the scene's own present (the passage shows them leave now; what "
                                             "someone tells of or remembers from another time or place is not the "
                                             "scene's present)? Answer none if nobody leaves.", here), here) if here else []
        back = ask.subset(q_subset(own, st, "Which of these people come into the room during this passage, in "
                                            "the scene's own present (the passage shows them come in now; what "
                                            "someone tells of or remembers from another time or place is not the "
                                            "scene's present)? Answer none if nobody comes in.", sorted(gone)),
                          sorted(gone)) if gone else []
        # someone whose own line is still to come in this scene has not gone: the audio has them speak here later
        # (a man told to go, who answers that he will go when he has had his say, was sent out before he said it)
        if isinstance(out_, list):
            later = {(es.get(x) or {}).get("speaker") or plan[x][0].get("speaker") for x in ids[n + 1:]}
            out_ = [x for x in out_ if x not in later]
        if isinstance(out_, list) and out_:
            res.setdefault("exits", {})[sid] = out_
            gone.update(out_)
        if isinstance(back, list) and back:
            res.setdefault("enters", {})[sid] = back
            gone.difference_update(back)
        # whom each person faces as the shot starts: the plan's facing words are screen directions ("left"), which
        # say nothing of whom; turned toward the others' middle, a man fighting a beast faced the woman behind him
        # as much as the beast (11-16). Asked by name, over the shot's own people
        listed = [p["name"] for p in sh.get("people") or [] if p.get("name") in names]
        for nm in listed:
            fo = [x for x in listed if x != nm]
            # (what they stand at is offered only when their place is a named thing of the set and they are no
            # creature: offered at a place on the open ground, a man fighting a beast was turned to "what he is at")
            at_ok = nm not in crt and set_named.get(next((str(p.get("at") or "") for p in sh.get("people") or []
                                                          if p.get("name") == nm), "")) not in (None, "floor")
            if not fo and not at_ok:              # alone and at no named thing: nothing to ask
                continue
            ft = ask.choose(q_face(sh, nm, [p for p in sh.get("people") or [] if p.get("name") in listed], looks,
                                   at_ok), fo + ([AT_THING] if at_ok else []) + [NOBODY])
            if ft in fo or ft == AT_THING:
                res.setdefault("face", {}).setdefault(sid, {})[nm] = ft
        # two who are turned to each other: how near they stand (q_near)
        fc_ = (res.get("face") or {}).get(sid) or {}
        for a_ in listed:
            b_ = fc_.get(a_)
            if b_ in listed and a_ not in crt and b_ not in crt and (fc_.get(b_) == a_ or b_ not in fc_) and \
                    not any(a_ in pr_[:2] for pr_ in (res.get("near") or {}).get(sid) or []):
                # (the plan has them both in a dance: partners, whatever the lines of that moment speak of. Asked,
                # two in "the dance set" who talked of other things were "conversing" every time, 11-26)
                how = NEARS[0] if in_a_dance(sh, a_, b_) else ask.choose(q_near(psg, sh, a_, b_), NEARS)
                if how in NEARS:
                    res.setdefault("near", {}).setdefault(sid, []).append([a_, b_, how])
        # the eyes as the shot opens: a figure's eyes are open in every engine still, and the redraw drew a child
        # wide awake in each shot of her asleep (11-15)
        for p in sh.get("people") or []:
            # (asked of the shot's own words, which say who sleeps; the passage alone left sleepers "not stated")
            e_ = ask.choose("You help stage a shot of a film. The shot's own words: " + (sh.get("action") or "")
                            + f"\nWhere things stood just before: {st}\nQuestion: as this shot starts, are "
                            f"{p['name']}'s eyes open or closed? Someone asleep or unconscious has them closed; anyone "
                            "else has them open unless the words say they are shut.\nAnswer with exactly one word: "
                            "open or closed.", ["open", "closed"])
            if e_ == "closed":
                res.setdefault("eyes", {}).setdefault(sid, {})[p["name"]] = e_
        if sh.get("kind") == "talk":
            res["camera"][sid] = ask.choose(q_choose(psg, st, "This shot shows someone speaking. Which kind of camera "
                                                     "framing does the passage's moment call for?", CAMERAS), CAMERAS)
        if sh.get("kind") == "action":
            act = sh.get("action") or ""
            named = json.dumps(sh.get("things") or {}, ensure_ascii=False)
            for p in sh.get("people") or []:
                nm = p["name"]
                if nm in crt:                     # (a creature's pose is asked above; it moves by the plan's places)
                    continue
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
                # (someone of the plan with no figure of the cast, an extra, has no wardrobe to choose from: the
                # pass asked anyway and stopped the place, 11-17)
                w = ask.choose(q_choose(own, st + f"; {nm} wears: {worn_at(nm)}",
                                        q + f"What is {nm} wearing when this shot ends, among these choices?",
                                        var[nm]), var[nm]) if nm in var else NOT
                if w != NOT:
                    claims.setdefault(nm, []).append((sid, w))
                    now[nm] = w
                # from the shot's OWN span only (as the worn state): a later shot's action is not claimed early
                # (each choice offered with its plain words; the plan's own state of the person, when it gives one)
                opts = ["none"] + [f"{x} ({POSE_WORDS[x]})" for x in POSES + WORK_POSES]
                sw = f" The plan says of {nm} as the shot starts: {p['state']}." if str(p.get("state") or "").strip() else ""
                ps = ask.choose(q_choose(own, st, q + f"{nm} is {p.get('pose') or 'in the shot'}.{sw} Most shots "
                                                      f"begin with no special action. Which of these does {nm} begin "
                                                      "to do as this shot starts?", opts), opts)
                ps = ps.split(" (")[0]
                if ps != NOT:
                    res["pose"].setdefault(sid, {})[nm] = ps
    # a change is taken once: of the shots in a row that claim the same new state, the first action shot (the one
    # that shows it) wins and the other claims are "not stated"
    for nm, cl in claims.items():
        cur, run = worn_now.get(nm), []
        # (the wardrobe's own changes in among the claims: at such a shot the state is the wardrobe's again)
        ev = sorted([(ids.index(x), 0, x, v) for x, v in (dflt.get(nm) or {}).items() if x != ids[0]]
                    + [(ids.index(x), 1, x, v) for x, v in cl])
        for _, kind, sid, w in [e for e in ev] + [(None, 1, None, None)]:
            if kind == 0:
                if run:
                    pick = next((x for x in run if plan[x[0]][0].get("kind") == "action"), run[0])
                    res["worn_end"].setdefault(pick[0], {})[nm] = pick[1]
                cur, run = w, []
                continue
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
            if ps in POSES:                       # (work is done shot after shot: not one shot's own)
                by.setdefault((nm, ps), []).append(sid)
    res["_action_owner"] = {}
    for (nm, ps), sids in by.items():
        if len(sids) < 2:
            continue
        # the shots whose own line states the action each own it (a second consecutive shot continues it); one
        # owner is picked only among claimants whose lines do not state it (11-12: the arm look at 04:01 and 04:04)
        states = [x for x in sids if ask.choose(
            HEAD + "A film shot's own action line:\n" + (plan[x][0].get("action") or plan[x][0].get("picture") or "")
            + f"\nQuestion: Does this line say that someone {POSE_WORDS[ps]} (the line names people by their looks)?"
            "\nAnswer with exactly one of these, nothing else: yes; no", ["yes", "no"]) == "yes"]
        if states:
            res["_action_owner"][f"{nm}: {ps}"] = states
            for x in sids:
                if x not in states:
                    res["pose"][x][nm] = "none"
            continue
        acts = [x for x in sids if plan[x][0].get("kind") == "action"]   # only an action shot states an action
        if len(acts) == 1:
            keep = acts[0]
            res["_action_owner"][f"{nm}: {ps}"] = keep
            for x in sids:
                if x != keep:
                    res["pose"][x][nm] = "none"
            continue
        sids = acts or sids
        opts = [f"shot {i + 1}" for i in range(len(sids))]
        lines = "\n".join(f"shot {i + 1}: {plan[x][0].get('action') or plan[x][0].get('picture') or ''}"
                          for i, x in enumerate(sids))
        own_q = (HEAD + "The film's shots, each with its own action line:\n" + lines + "\nQuestion: In which shot "
                 f"does someone {POSE_WORDS[ps]}, as that shot's own action line states? (The lines name people by "
                 "their looks.)\nAnswer with exactly one of "
                 "these, nothing else: " + "; ".join(opts + [NOT]))
        got = ask.choose(own_q, opts)
        keep = sids[opts.index(got)] if got in opts else None
        res["_action_owner"][f"{nm}: {ps}"] = keep
        for x in [y for (n2, p2), ys in by.items() if (n2, p2) == (nm, ps) for y in ys]:
            if x != keep:
                res["pose"][x][nm] = "none"
    res.update(_asked=ask.asked, _calls=ask.calls, _cached=ask.cached, _split=ask.split, _s=round(time.time() - t0, 1))
    # what each person has on or carries beyond their everyday clothes, shot by shot: the director's account from
    # the whole story (place_spec.things; one open question, its answer checked there)
    import place_spec as PS
    sj = set_dir / "spec.json"
    place = (json.loads(sj.read_text()).get("place") if sj.exists() else None) or set_dir.name
    res["things"] = PS.things(plan, ids, place, timeline if isinstance(timeline, list) else
                              timeline.get("segments") or timeline.get("lines") or [], url, cache)
    # what is worn, in words for the redraw (the figures wear their usual look)
    res["wears"] = {i: {n: [t["what"] for t in ts if t["on"] not in ("held", "worked")] for n, ts in row.items()
                        if any(t["on"] not in ("held", "worked") for t in ts)} for i, row in res["things"].items()}
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("_asked", "_calls", "_cached", "_split", "_s")}))


if __name__ == "__main__":
    main()
