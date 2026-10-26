"""10-23 review findings as checks (user watched the whole chapter): sync drift, the dragon left out of the keys,
the fight prompts following a still staging, the director stopping short of the blow, the armour."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import interaction as I          # noqa: E402
import mannequin_flow as MF      # noqa: E402

fails = []


def check(ok, what, got=""):
    print(("PASS " if ok else "FAIL ") + what + (f"  ({got})" if got and not ok else ""))
    if not ok:
        fails.append(what)


# sync: segment frame counts come from the timeline, so the sum never drifts
ts = [0.0, 33.96, 35.61, 38.18, 47.0, 51.37]
edge = [round(t * 24) for t in ts] + [round((ts[-1] + 2.5) * 24)]
nf = [edge[i + 1] - edge[i] for i in range(len(ts))]
check(all(abs(sum(nf[:i]) / 24 - ts[i]) <= 0.5 / 24 for i in range(len(ts))), "assemble: every shot starts within half a frame")

# director: plays through, creatures act, no notes inside the prompt
q = I.acts_prompt({"id": "s1", "where": "x", "side": "outside"}, [], ["Edric", "Vorgrath"], {})
check("PLAY EACH ACTION THROUGH" in q and "A CREATURE acts" in q, "director: play-through and creature rules asked")
check("10-23" not in q and "10-22" not in q, "director: no dated notes inside the prompt")

# director: a moment that never reaches its result is asked again; an attack answered by a block is fine
check(I.short_of(["Edric: drives the sword tip towards the dark patch of bare skin"]) != [], "director: 'towards' with no hit is short")
check(I.short_of(["Vorgrath: extends its foreclaw towards Edric", "Edric: raises the sword to block the claw"]) == [],
      "director: an attack met by a block is not short")

# prompt writer: both routes format
for src, fol in (("S", "F"),):
    t = MF.MOTION_RULES.format(sec=5, people="Edric", moment="m", beats="b", picture="p", place="x", hands="",
                               swing="", source=src, follow=fol)
    check("{" not in t.replace("{{", "").split("Reply")[0], "prompt rules: every slot filled")

# keys: the creature has its own region and LoRA (real staged maps of the fight, when present)
book = Path.home() / "audiobooks" / "knight_forged_v3"
md = MF.paths(book, "01_chapter_1")[2]
if (md / "action-003217" / "maps" / "frames" / "depth_0000_creature.npy").exists():
    sh = MF._in_shot(md, "action-003217")
    check([w for w, _ in sh] == ["Edric", "Vorgrath"] and all(b for _, b in sh), "keys: Vorgrath framed with a region", sh)
    p = MF._prompt(book, "01_chapter_1", "action-003217", [w for w, _ in sh])
    import re
    check(re.search(r"chr_vorgrath Vorgrath(, [^:]+)?: a (four-legged )?massive dragon", p) is not None,
          "keys: the dragon's trigger and look in the prompt", p[:400])
    check("Star-iron breastplate" in p, "keys: Edric in the wardrobe's armour at the fight")
    check(".." not in p, "keys: no doubled stops")

# looks carry no posture (the slain dragon was drawn standing "on four legs"); overlapping regions -> lighter LoRAs
check("standing" not in MF._appearance("a massive dragon, black scales, standing on four legs, tail extended, wings folded."),
      "keys: posture clauses out of a look")
check(MF._overlap_factor([("Edric", [0.3, 0.3, 0.6, 1.0]), ("Vorgrath", [0.2, 0.1, 0.9, 1.0])], "Edric") < 1.0,
      "keys: a region inside another's gets a lighter LoRA")

# guide=pins: the hit is pinned inside the clip, a second from either end, and the last frame
if (md / "action-003217" / "motion" / "joints_Edric.npz").exists():
    pf = MF.pin_frames(md, "action-003217", 121)
    check(len(pf) == 2 and 24 <= pf[0] <= 96 and pf[1] == 120, "pins: hit inside the clip + the last frame", pf)
check(MF.pin_frames(md, "no-such-moment", 73) == [72], "pins: no staged motion -> the last frame only")

# the fight route: the director's action sentence names every effect's source and uses no magic words
ok = {"left": "Edric", "right": "Vorgrath", "layout": "x",
      "action": "The huge black dragon breathes a stream of fire from its mouth; his hands grip only the sword."}
check(MF._action_faults(ok, ["Edric", "Vorgrath"]) == [], "fight: a sourced, plain action passes")
bad = dict(ok, action="The dragon breathes fire at him. He swings his enchanted sword.")
check(len(MF._action_faults(bad, ["Edric", "Vorgrath"])) == 2, "fight: unsourced fire + 'enchanted' are faults")
check("whole body visible" in MF._fight_prompt(book, "01_chapter_1", "action-003167",
      {"left": "Edric", "right": "Vorgrath", "framing": "wide", "layout": "Apart."}), "fight: the key shows both whole")

check("four-legged" in MF._appearance("a massive dragon, black scales, standing on four legs, wings folded."),
      "keys: the body plan (four-legged) stays when the posture goes")
check(MF._action_faults({"left": "Edric", "right": "Vorgrath", "layout": "x", "action": "Edric drives the sword in."},
                        ["Edric", "Vorgrath"]) != [], "fight: a name in the action is a fault")

import inspect                   # noqa: E402
src = inspect.getsource(MF.run)
blk = src[src.index("if fights:"):src.index("def frozen")]
check('igpu("strata"' not in blk, "fight clip check: the language model is not started beside the vision judge (OOM 17:40)")

lay = {"left": "Edric", "right": "Vorgrath", "framing": "wide", "layout": "On the left, a young man in steel plate armour "
       "lies on the grey ash propped on an elbow; on the right, a huge black dragon crouches on all four legs."}
check("fighting stance" not in MF._fight_prompt(book, "01_chapter_1", "action-003267", lay),
      "fight key: a knocked-down fighter is not also drawn standing in a stance (two Edrics at 5:26)")
check("fighting stance" in MF._fight_prompt(book, "01_chapter_1", "action-003267", dict(lay, layout="Apart; the dragon "
      "crouches on all four legs on the ground.")), "fight key: a standing fighter keeps the stance")
check("tail that sweeps" in MF.FIGHT_RULES, "fight rules: the body part the action uses is in the opening picture")

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
