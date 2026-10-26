"""INTERACTION PROGRAMS: what each person does with each prop / person in a moment, as a few generic acts the language
model writes per moment (plan review) and staging compiles onto the machinery that already holds (user 10-22: "tools
that help the prop/body interaction that are more generic than hard-coding a 'strike' pose").

An act (frame["acts"], a list, in order):
  {"do": "grasp",   "who": N, "hand": "right"|"left"|"both", "prop": P}      hold P by its grip (affordance "grip")
  {"do": "strike",  "who": N, "tool": P, "target": T}                         the tool's working face/edge/tip meets
        T: a set prop that supports work (the anvil: the strike-on-support action, the work lies on it), a prop on
        it, or a creature part "Name:part" (head, claw_r ...): a contact at the end of the swing
  {"do": "operate", "who": N, "prop": P}                                       work a jointed set prop through its
        cycle (the bellows' board: the hands ride the handle, tools/blockout_video.py)
  {"do": "place",   "who": N, "prop": P, "on": Q}                              P ends lying on Q's top
  {"do": "carry",   "who": C, "carried": N}                                    C carries N (a creature: in its claws)
  {"do": "aim",     "who": N, "hand": "right"|"left", "at": "Name[:part]"}     point the hand / held prop at someone
  {"do": "free",    "who": N, "hand": "right"|"left"}                          this hand holds nothing (directed so)
  {"do": "block",   "who": N, "tool": P, "against": "Name:part"}              parry / shield: the held tool (a sword's
        edge, a shield's face) meets the attacker's part (a contact at the end of the move, like a strike on a part)
Anything else is ignored (logged).

THE DIRECTOR (user 10-22: "all generated and described by the bookcaster llm based on the story, directing the
action sequence"): per moment the language model writes {"acts": [...], "beats": [...]}: the acts direct BOTH hands of
everyone who handles anything (a grasp or a free per hand; a strike's tool hand counts), the beats are the body
action in order, in plain words ("raises the hammer high over his head", "brings it down hard on the blade"). The
book's director's notes (work/video/stage/director_notes.json: user-approved choices, e.g. the forging recipe) are
part of its brief; a moment with an undirected hand goes back once (undirected()). Names must be the cast's; props the book's (library + modelled).

Compiled (staging, scene_stage.shot_for): grasp -> figure.holds; strike on a support -> holds + the generic
strike-on-support contact action (scene_stage.contact_action); strike on a creature part -> holds + a contact
{prop, point, to: "Name:part", frame: "end"}; operate -> shot.operate + the person at the handle; place -> a set prop
on the support's top; carry -> attach_hint; aim -> figure.aim. A moment with acts skips the text-pattern rules
(operate_rule, the striker-from-text rule); without acts they still apply.
"""

from __future__ import annotations

import re

VERBS = ("grasp", "strike", "operate", "place", "carry", "aim", "free", "block")
HANDS = ("right", "left", "both")

PROMPT_VOCAB = (
    "Acts (a JSON list per moment, in order; only what the moment SHOWS; [] when nobody handles anything):\n"
    '  {"do": "grasp", "who": "<cast>", "hand": "right|left|both", "prop": "<prop>"}  - holds it\n'
    '  {"do": "strike", "who": "<cast>", "tool": "<prop>", "target": "<prop or Cast:part>"}  - hits the target with '
    "the tool (a hammer on work on an anvil: target = the anvil; a sword at a creature: target = \"Name:head\")\n"
    '  {"do": "operate", "who": "<cast>", "prop": "<prop>"}  - works a mechanism (pumps bellows, opens a door)\n'
    '  {"do": "place", "who": "<cast>", "prop": "<prop>", "on": "<prop>"}  - puts it down on something\n'
    '  {"do": "carry", "who": "<cast>", "carried": "<cast>"}  - carries a person (a dragon in its claws)\n'
    '  {"do": "aim", "who": "<cast>", "hand": "right|left", "at": "<cast>[:part]"}  - points the hand or weapon\n'
    '  {"do": "free", "who": "<cast>", "hand": "right|left"}  - this hand holds nothing\n'
    '  {"do": "block", "who": "<cast>", "tool": "<prop>", "against": "<cast>:<part>"}  - parries or shields a blow '
    "(the sword's edge or the shield's face meets the attacker's claw, tail, blade)\n"
    "Direct BOTH hands of everyone who handles anything: each hand gets a grasp (or is the strike's tool hand, the "
    "right) or a free. A piece being worked on a support that has a grip (a blade with a hilt) is HELD by the other "
    "hand: grasp it. Use ONLY tools the text or the director's notes give the person (no tongs unless named).")


def resolve_prop(name, props: set):
    """A prop as the moment names it -> the book's prop key: "Star-iron Sword, finished" -> sword__finished (the state
    variant when the book has it), "Melted Shield" -> shield, "Forge Door" -> door (10-22: 27 acts were rejected for
    using the moments' descriptive names). Unknown -> the name unchanged (validate rejects it)."""
    if not isinstance(name, str) or name in props or ":" in name:
        return name
    import props_build as PB
    cls, r = PB.classify(name)
    if cls != "physical" or not r.get("base"):
        return name
    st = r.get("state")
    if st and f"{r['base']}__{st}" in props:
        return f"{r['base']}__{st}"
    return r["base"] if r["base"] in props else name


PART_WORDS = [(r"belly|flank|side|heart|chest|breast|ribs|wing", "chest"), (r"neck|head|eye|skull|crown", "head"),
              (r"mouth|teeth|jaw|fang", "jaw"), (r"nose|snout|nostril", "snout"),
              (r"left (?:fore)?claw|left talon|left foot", "claw_l"), (r"claw|talon|foot|paw", "claw_r"),
              (r"tail|hip|leg|haunch", "pelvis")]


def resolve_part(target: str) -> str:
    """"Vorgrath:underbelly" -> "Vorgrath:chest": a creature part named in words -> the rig's strike points (chest,
    head, jaw, snout, claw_l, claw_r, pelvis)."""
    who, part = target.split(":", 1)
    p = part.lower().replace("_", " ")
    if p in ("chest", "head", "jaw", "snout", "claw_l", "claw_r", "pelvis"):
        return target
    for rx, anchor in PART_WORDS:
        if re.search(rx, p):
            return f"{who}:{anchor}"
    return f"{who}:chest"


def resolve_props(a: dict, props: set) -> dict:
    a = dict(a)
    for k in ("prop", "tool", "on", "target"):
        if k in a:
            a[k] = resolve_prop(a[k], props)
    for k in ("target", "at", "against"):       # (a block's "against" too: 10-22 "Vorgrath:foreclaw" broke the motion)
        if isinstance(a.get(k), str) and ":" in a[k]:
            a[k] = resolve_part(a[k])
    return a


def validate(acts, cast: set, props: set, text: str | None = None, kinds: dict | None = None) -> tuple[list, list]:
    """-> (clean acts, rejected lines). Names exact; a prop unknown to the book is rejected (never invented); with
    text (the moment's prompt + motion + props): a HELD item (grasped, the striking tool, the placed prop) must be
    named in it (10-22: no tongs unless the text names tongs)."""
    ok, bad = [], []
    low = (text or "").lower()

    def named(p_):
        base = p_.split("__")[0].split("_")[0]           # the family word ("sword_hot" -> sword)
        return text is None or bool(re.search(rf"\b{re.escape(base)}", low))
    for a in acts or []:
        if not isinstance(a, dict) or a.get("do") not in VERBS:
            bad.append(f"unknown act {a!r}")
            continue
        d = a["do"]
        who = a.get("who")
        if who not in cast:
            bad.append(f"{d}: who {who!r} is no cast member")
            continue
        if d == "grasp" and (a.get("prop") not in props or a.get("hand", "right") not in HANDS):
            bad.append(f"grasp: {a.get('prop')!r} / {a.get('hand')!r}")
            continue
        if d == "strike":
            t = str(a.get("target") or "")
            tgt_ok = t in props or (":" in t and t.split(":")[0] in cast)
            if a.get("tool") not in props or not tgt_ok:
                bad.append(f"strike: tool {a.get('tool')!r} target {t!r}")
                continue
        if d in ("operate",) and a.get("prop") not in props:
            bad.append(f"operate: {a.get('prop')!r}")
            continue
        if d == "place" and (a.get("prop") not in props or a.get("on") not in props):
            bad.append(f"place: {a.get('prop')!r} on {a.get('on')!r}")
            continue
        if d == "block":
            ag = str(a.get("against") or "")
            if a.get("tool") not in props or ag.split(":")[0] not in cast:
                bad.append(f"block: tool {a.get('tool')!r} against {ag!r}")
                continue
        if d == "free":
            if a.get("hand") not in ("right", "left"):
                bad.append(f"free: hand {a.get('hand')!r}")
            else:
                ok.append(a)
            continue
        if d == "carry" and a.get("carried") not in cast:
            bad.append(f"carry: {a.get('carried')!r}")
            continue
        if d == "aim" and str(a.get("at") or "").split(":")[0] not in cast:
            bad.append(f"aim: at {a.get('at')!r}")
            continue
        if kinds is not None:                        # grasp / strike take HAND-HELD props; operate a set prop with a
            item = a.get("prop") if d in ("grasp", "place") else a.get("tool")      # worked joint (10-22: "grasp
            if d in ("grasp", "strike", "place", "block") and kinds.get(item) not in (None, "held"):   # the bellows")
                bad.append(f"{d}: the {item} is not hand-held ({kinds.get(item)})")
                continue
            if d == "operate" and kinds.get(a.get("prop")) == "held":
                bad.append(f"operate: the {a.get('prop')} is hand-held, nothing to work")
                continue
        held = a.get("prop") if d in ("grasp", "place") else (a.get("tool") if d in ("strike", "block") else None)
        if held and not named(held):
            bad.append(f"{d}: the {held} is not in the moment's text")
            continue
        ok.append(a)
    return ok, bad


def _prop_of(shot, name):
    for pr in (shot.get("set") or {}).get("props", []):
        if (pr.get("prop") or pr.get("name") or "").split(" ")[0] == name:
            return pr
    return None


def compile_acts(shot: dict, frame: dict, log=None) -> list:
    """Apply the moment's acts to its shot (in place). -> the items the acts ground (held / placed props)."""
    import numpy as np
    import blockout_props as P_
    acts = frame.get("acts") or []
    # an act names a prop FAMILY ("sword"); the moment's own props may name its variant ("Star-iron Sword, finished"
    # -> sword__finished, the star-iron model): the act takes the variant (10-22: the fight's grasps held the plain sword)
    allp = {k for k in P_.library() if not k.startswith("_")}
    variant = {}
    for nm in frame.get("props") or []:
        k = resolve_prop(str(nm), allp)
        if k in allp:
            variant.setdefault(k.split("__")[0], k)
    # a bare STATE word ("blank": an alias with no mesh of its own) -> its one built variant ("sword__blank")
    for k in allp:
        if "__" in k:
            st_ = k.split("__", 1)[1]
            if sum(1 for x in allp if x.endswith("__" + st_)) == 1:
                variant.setdefault(st_, k)
    acts = [dict(a, **{f_: variant.get(a[f_], a[f_]) for f_ in ("prop", "tool") if a.get(f_) in variant})
            for a in acts]
    figs = {f.get("who"): f for f in shot.get("figures", [])}
    items = []
    notes = shot.setdefault("plan_notes", {}).setdefault("acts", [])
    for a in acts:
        d, who = a["do"], a.get("who")
        f = figs.get(who)
        if d == "carry":                              # (applied before the figures are placed: carry_hints)
            notes.append(f"{who} carries {a['carried']}")
            continue
        if f is None:
            notes.append(f"skipped {d}: {who} not in the shot")
            continue
        if d == "grasp":
            hands = ["right", "left"] if a.get("hand") == "both" else [a.get("hand") or "right"]
            for h in hands:
                f.setdefault("holds", {})[h] = a["prop"]
            items.append(a["prop"])
            notes.append(f"{who} holds the {a['prop']} ({a.get('hand', 'right')})")
        elif d == "strike":
            f.setdefault("holds", {})["right"] = a["tool"]
            items.append(a["tool"])
            t = a["target"]
            if ":" in t:                              # a creature part: the tool's point/edge meets it at the end
                aff = (P_.get(a["tool"]).get("affordances") or {}) if a["tool"] in P_.library() else {}
                point = "tip" if "tip" in aff else ("strike_face" if "strike_face" in aff else "edge")
                shot.setdefault("contacts", []).append({"who": who, "prop": a["tool"], "point": point, "to": t,
                                                        "frame": "end"})
                notes.append(f"{who} strikes {t} with the {a['tool']} ({point})")
            else:                                     # on a support: the generic strike-on-support action finds the
                notes.append(f"{who} strikes on the {t} with the {a['tool']}")
        elif d == "operate":
            pr = _prop_of(shot, a["prop"])
            if pr is None:
                notes.append(f"skipped operate: no {a['prop']} in the set")
                continue
            _, e = P_.set_entry(pr)
            jd = next((j for j in e.get("joints") or [] if j.get("cycle") and j.get("handle_m")), None)
            if jd is None:
                notes.append(f"skipped operate: the {a['prop']} has no worked joint with a handle")
                continue
            hw, pw = P_.joint_handle(pr, jd["name"], None)
            out = np.array([hw[0] - pw[0], hw[1] - pw[1]])
            out = out / max(np.linalg.norm(out), 1e-6)
            st = hw[:2] + out * 0.42
            f["at"] = [round(float(st[0]), 3), round(float(st[1]), 3)]
            f["facing"] = round(float(np.degrees(np.arctan2(-out[1], -out[0]))), 1)
            f.pop("ground_z", None)
            f.pop("holds", None)
            shot.setdefault("operate", []).append({"who": who, "prop": a["prop"], "joint": jd["name"],
                                                   "hands": ["right", "left"]})
            notes.append(f"{who} works the {a['prop']} ({jd['name']})")
        elif d == "place":
            sup = _prop_of(shot, a["on"])
            if sup is None:
                notes.append(f"skipped place: no {a['on']} in the set")
                continue
            c, n_, half = P_.set_top(sup)
            shot["set"].setdefault("props", []).append(
                {"name": a["prop"], "prop": a["prop"], "box": [float(c[0]), float(c[1]), float(c[2]), 1, 1, 1],
                 "roll": 90, "placed": f"on the {a['on']} (act)"})
            for h, it in list((f.get("holds") or {}).items()):
                if it == a["prop"]:
                    f["holds"].pop(h)
            items.append(a["prop"])
            notes.append(f"the {a['prop']} lies on the {a['on']}")
        elif d == "block":
            held_by = next((h for h, it in (f.get("holds") or {}).items() if it == a["tool"]), None)
            if held_by is None:
                f.setdefault("holds", {})["left" if "shield" in a["tool"] else "right"] = a["tool"]
            items.append(a["tool"])
            aff = (P_.get(a["tool"]).get("affordances") or {}) if a["tool"] in P_.library() else {}
            point = next((k for k in ("face", "edge", "flat", "tip") if k in aff), "edge")
            shot.setdefault("contacts", []).append({"who": who, "prop": a["tool"], "point": point, "to": a["against"],
                                                    "frame": "end", "kind": "block"})
            notes.append(f"{who} blocks {a['against']} with the {a['tool']} ({point})")
        elif d == "free":
            (f.get("holds") or {}).pop(a["hand"], None)
            f.setdefault("free_hands", []).append(a["hand"])
            notes.append(f"{who}'s {a['hand']} hand is free")
        elif d == "aim":
            f["aim"] = {"hand": a.get("hand", "right"), "to": a["at"].split(":")[0],
                        "point": (a["at"].split(":") + ["head"])[1]}
            notes.append(f"{who} aims at {a['at']}")
    if log and notes:
        log(f"{frame.get('shot') or shot.get('moment', {}).get('id', '?')}: acts: " + "; ".join(notes))
    return items


def carry_hints(frame: dict) -> None:
    """The carry acts as the planner's attach hints (before frame_to_shot places the figures)."""
    for a in frame.get("acts") or []:
        if a.get("do") == "carry":
            frame.setdefault("attach_hint", {})[a["carried"]] = {"to": a["who"], "point": "claws"}


def split_answer(x) -> tuple[list, list]:
    """A moment's direction -> (acts, beats); an old-style bare list = acts only."""
    if isinstance(x, list):
        return x, []
    if isinstance(x, dict):
        acts = x.get("acts") if isinstance(x.get("acts"), list) else []
        beats = [str(b).strip() for b in (x.get("beats") or []) if str(b).strip()]
        return acts, beats
    return [], []


def undirected(acts: list) -> list:
    """The hands nobody directed: for everyone with a grasp or a strike, each of right/left must have a grasp, be the
    strike's tool hand (right), or be free. -> ["Edric: left", ...]"""
    hands = {}
    for a in acts or []:
        w, d = a.get("who"), a.get("do")
        if d in ("grasp", "strike", "free", "block"):
            hs = hands.setdefault(w, set())
            if d == "grasp":
                hs.update(["right", "left"] if a.get("hand") == "both" else [a.get("hand") or "right"])
            elif d == "strike":
                hs.add("right")
            elif d == "block":
                hs.add("left" if "shield" in str(a.get("tool")) else "right")
            else:
                hs.add(a.get("hand"))
    return [f"{w}: {h}" for w, hs in hands.items() for h in ("right", "left") if h not in hs]


SHORT = re.compile(r"\b(towards?|about to|poised|hover\w*|prepar\w* to|trembl\w*|ready to|readies|aims?(?: at)?|"
                   r"tense[sd]?)\b", re.I)
RESULT = re.compile(r"\b(into|hits?|lands?|strikes?|pierces?|meets?|slams?|crash\w*|sinks?|bites?|engulf\w*|"
                    r"knock\w*|glances?|sends?|throws?|smash\w*|cuts?|tears?|burn\w*|blocks?)\b", re.I)


def short_of(beats: list) -> list:
    """A moment's beats when none reaches a result and some stop short of it (10-23: the killing thrust "drives the sword tip towards the bare
    skin", "presses the tip against it, sword trembling slightly": the blow never landed)."""
    if any(RESULT.search(str(b)) for b in beats or []):      # a moment whose action reaches a result somewhere
        return []
    return [b for b in beats or [] if SHORT.search(str(b))]


def director_notes(book) -> list:
    """The book's director's notes (work/video/stage/director_notes.json {"notes": [...]}): choices the user approved."""
    import json
    from pathlib import Path
    p = Path(book) / "work" / "video" / "stage" / "director_notes.json"
    try:
        return [str(n) for n in json.loads(p.read_text()).get("notes") or []]
    except (OSError, ValueError, AttributeError):
        return []


def acts_prompt(sc, moments, cast, props_aff, notes=None, gaps=None, short=None) -> str:
    """The language model's request for one scene's direction (acts + beats per moment)."""
    ms = "\n".join(f"- {m['shot']}: who {m.get('who')}; props {m.get('props')}; prompt: {(m.get('prompt') or '')[:300]}"
                   f"; motion: {(m.get('motion') or '')[:200]}" for m in moments)
    pl = "\n".join(f"- {k}: {', '.join(v) or 'no handles'}" for k, v in sorted(props_aff.items()))
    nt = ("\nDIRECTOR'S NOTES for this book (approved; follow them whenever they apply):\n" +
          "\n".join(f"- {n}" for n in notes) + "\n") if notes else ""
    gp = ("\nLAST TIME these hands were not directed (direct them now, grasp or free): " + "; ".join(gaps) + "\n") \
        if gaps else ""
    gp += ("\nLAST TIME these beats stopped short of their result (play each through: the blow lands, the fire hits): "
           + "; ".join(short) + "\n") if short else ""
    # play-through rule: 10-23 the killing thrust was directed as a sword tip "hovering just before impact" and never
    # landed; creatures got no beats of their own and the fight had no action; the slain dragon stood alive in the
    # next shots
    return ("You are the action director of a film. For each moment of this scene direct what each person "
            "physically does: the acts (what each hand holds and does with the props and people) and the beats: the "
            "MECHANICS of the action, in order, one beat per movement, each saying WHO moves WHICH body part or held "
            "item, in WHICH direction and path (from where to where: hip, shoulder, over the head, across the body), "
            "WHAT it meets, and the visible RESULT. e.g. \"Edric: right hand raises the hammer from his hip to high "
            "over his head\", \"Edric: brings the hammer down hard onto the blade on the anvil; sparks burst\", "
            "\"Knight: sword in both hands sweeps up and across from his right hip to above his left shoulder; the flat "
            "of the blade meets the dragon's right claw and the claw glances off to his left\", \"Dragon: right claw "
            "slashes down from high on its right at the knight's head\". Only what the moment shows; every person who "
            "moves gets beats. PLAY EACH ACTION THROUGH to its result inside the moment: the blow lands, the claw "
            "strikes the shield, the fire engulfs them; never \"about to\", \"poised\", \"hovering\", \"prepares to\" or "
            "\"stays still\". A FIGHT names the attack, the defence and the result, each as mechanics. A CREATURE acts "
            "with its own body: jaws, claws, tail, wings, fire breath, each as mechanics (from where to where, what it "
            "hits), and it gets beats in every moment it is in. Anyone killed, fallen or collapsed STAYS so in the later "
            "moments until a beat raises them: give them a beat saying so (\"Vorgrath: lies dead on the ash, unmoving\").\n\n"
            f"SCENE {sc['id']}: {sc['where']} ({sc['side']}); present {sc.get('present')}\n\nMOMENTS:\n{ms}\n\n"
            f"Cast: {sorted(cast)}\nProps of the book (and their contact parts):\n{pl}\n{nt}{gp}\n{PROMPT_VOCAB}\n\n"
            'Answer with ONLY JSON: {"<moment id>": {"acts": [acts], "beats": ["...", ...]}, ...} (every moment).')
