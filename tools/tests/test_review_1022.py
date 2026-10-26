"""The 10-22 morning review fixes (synthetic, CPU only, seconds).

    ~/tts-ab/.venv-pose/bin/python -I tools/tests/test_review_1022.py

- scenes: the planner's list made usable (known places, side variants), a moment takes its scene's place unless its
  own text names another (a CONFLICT for the plan review), a creature the text names is a figure, not a prop;
- plan review: a split inside a paragraph starts at its moment (t_split), a split at the scene's first moment
  re-places the scene, speakers are present;
- staging: a person working a jointed prop stands at its handle and operates it; a work prop found in a set is never
  hidden as "the set's own"; the lone person strikes when the text strikes; operated moments render their clip.
Prints PASS/FAIL per check, exit 1 on any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from tts_reader import video as V  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if str(detail) else ""))
    if not ok:
        FAILS.append(name)


LOOK = {"places": {"The Hall": "Inside: a stone hall", "The Forge": "Inside: a smithy",
                   "The Forge (outside)": "Outside: a stone smithy", "The Fields": ""},
        "characters": {"Edric": "a man of about 20", "The King": "a man of about 50",
                       "Vorgrath": "a massive black dragon with iron scales"}}


def test_scenes():
    paras = ["a", "b", "c", "d", "e"]
    r = V.clean_scenes({"scenes": [{"from": 0, "where": "Hall", "side": "inside", "present": ["Edric", "Nobody"]},
                                   {"from": 2, "where": "The Forge", "side": "outside"},
                                   {"from": 3, "where": "The Fields (outside)", "side": "outside"},
                                   {"from": 3, "where": "The Hall"}]}, paras, LOOK)["scenes"]
    check("scenes: a partial name -> the place", r[0]["where"] == "The Hall", r[0]["where"])
    check("scenes: present = cast only", r[0]["present"] == ["Edric"], r[0]["present"])
    check("scenes: the other side -> the place's variant", r[1]["where"] == "The Forge (outside)", r[1]["where"])
    check("scenes: a side variant of a known place is accepted", r[2]["where"] == "The Fields (outside)", r[2]["where"])
    check("scenes: starts must increase", len(r) == 3 and r[-1]["to"] == 4, [s["from"] for s in r])
    check("scenes: a creature's kind from its look", V.creature_kinds(LOOK["characters"]) == {"Vorgrath": "dragon"})


def test_apply_scene():
    sc = {"id": "s05", "where": "The Forge", "present": ["Edric"]}
    f = {"prompt": "Medium shot. Edric pumps the bellows.", "where": "The Fields", "who": ["Edric"]}
    V.apply_scene(f, sc, LOOK["characters"])
    check("apply_scene: no place in the text -> the scene's", f["where"] == "The Forge" and f["scene"] == "s05", f)
    f = {"prompt": "Wide shot of the fields at dawn.", "where": "The Fields", "who": []}
    V.apply_scene(f, sc, LOOK["characters"])
    check("apply_scene: the text names its own place -> kept + conflict", f["where"] == "The Fields"
          and f.get("scene_conflict", {}).get("scene_where") == "The Forge", f)
    f = {"prompt": "Wide shot. A massive black dragon carries a pale figure.", "where": "The Fields",
         "who": ["Edric"], "props": ["Dragon Silhouette", "Moon"]}
    V.apply_scene(f, {"id": "s01", "where": "The Fields", "present": []}, LOOK["characters"])
    check("apply_scene: the text's dragon is the figure, not a prop",
          "Vorgrath" in f["who"] and f["props"] == ["Moon"], f)
    f = {"prompt": "Wide shot.", "where": "The Fields: Outside: black ash", "who": []}
    V.apply_scene(f, {"id": "s07", "where": "The Fields", "present": []}, LOOK["characters"])
    check("apply_scene: a description-as-name of the same place is no conflict", "scene_conflict" not in f, f)


def test_cutaway_and_present():
    import plan_review as PR
    f = {"prompt": "Wide shot of the dragon gliding across the moon above The Fields.", "where": "The Hall", "who": []}
    V.apply_scene(f, {"id": "s02", "where": "The Hall", "present": []}, LOOK["characters"], LOOK["places"])
    check("apply_scene: a place the PROMPT names makes a cutaway (kept there)", f["where"] == "The Fields"
          and f.get("scene_conflict"), f)
    sc = {"id": "s02", "from": 0, "to": 0, "where": "The Hall", "present": ["The King", "Edric"]}
    plan = {"scenes": [sc]}
    PR.apply_answer({"present": ["Edric"]}, sc, plan, {}, dict(LOOK["places"]), LOOK["characters"], [0.0], print,
                    paras=["The Knight told the King everything."])
    check("review: a person the scene's text names stays present", "The King" in sc["present"], sc)


def test_motion_and_followups():
    import plan_review as PR
    sc = {"id": "s02", "from": 0, "to": 0, "where": "The Hall", "present": ["The King"]}
    plan = {"scenes": [sc]}
    fr = {"m1": {"shot": "m1", "_t0": 1.0, "who": [], "prompt": "The dragon glides across the moon.",
                 "motion": "The dragon glides across the moon."},
          "m2": {"shot": "m2", "_t0": 5.0, "who": [], "prompt": "Wide shot of the hall.", "motion": ""}}
    PR.apply_answer({"moments": {"m1": {"prompt": "Wide shot of the stone hall. The King sits on the dais."}}},
                    sc, plan, fr, dict(LOOK["places"]), LOOK["characters"], [0.0], print)
    check("review: a new prompt brings its motion (no stale 'the dragon glides')",
          fr["m1"]["motion"] == "The King sits on the dais." and fr["m1"].get("_touched"), fr["m1"])
    PR.apply_answer({"moments": {"m1": {"who": ["The King"]}, "m2": {"who": ["The King"], "prompt": "Wide shot of the hall. The King waits."}}},
                    sc, plan, fr, dict(LOOK["places"]), LOOK["characters"], [0.0], print, dry=True, apply_only={"m1"})
    fx = {"m3": {"shot": "m3", "_t0": 1.0, "who": ["Edric"], "props": ["Star-iron Sword, finished"],
                 "prompt": "Edric holds the Star-iron Sword, finished, high."}}
    PR.apply_answer({"moments": {"m3": {"prompt": "Edric holds the sword high."}}}, sc, plan, fx,
                    dict(LOOK["places"]), LOOK["characters"], [0.0], print)
    check("review: a prompt rewrite never drops a prop's own words (Star-iron)", "Star-iron" in fx["m3"]["prompt"], fx["m3"])
    check("review pass 2: a follow-up on a moment pass 1 changed is applied, the rest listed",
          fr["m1"]["who"] == ["The King"] and fr["m2"]["who"] == [], (fr["m1"]["who"], fr["m2"]["who"]))


def test_review_split():
    import plan_review as PR
    plan = {"scenes": [{"id": "s01", "from": 0, "where": "The Hall", "side": "inside", "present": ["The King"]}]}
    fr = {"a1": {"shot": "a1", "_t0": 1.0, "scene": "s01"},
          "a2": {"shot": "a2", "_t0": 9.0, "scene": "s01", "where": "The Forge"}}
    PR.apply_answer({"splits": [{"moment": "a2", "where": "The Forge", "side": "inside", "why": "epilogue"}]},
                    plan["scenes"][0], plan, fr, dict(LOOK["places"]), LOOK["characters"], [0.0, 20.0], print)
    sc = plan["scenes"]
    check("review: a split inside a paragraph -> a new scene from its moment",
          len(sc) == 2 and sc[1]["where"] == "The Forge" and sc[1]["from"] == 0 and sc[1]["t_split"] == 8.95, sc)
    st = V.scene_starts({"scenes": sc}, "x\n\ny", [{"t": 0.0, "text": "x"}, {"t": 20.0, "text": "y"}])
    check("review: the split scene starts at its moment", [round(t, 2) for t, _ in st] == [0.0, 8.95], st)
    plan = {"scenes": [{"id": "s01", "from": 0, "where": "The Hall", "side": "inside", "present": []}]}
    PR.apply_answer({"splits": [{"moment": "a1", "where": "The Forge", "side": "inside"}]},
                    plan["scenes"][0], plan, fr, dict(LOOK["places"]), LOOK["characters"], [0.0, 20.0], print)
    check("review: a split at the scene's first moment re-places the scene",
          len(plan["scenes"]) == 1 and plan["scenes"][0]["where"] == "The Forge", plan["scenes"])
    # a one-shot FLASHBACK inside a scene returns to it (the Tower + Barley Fields memories by the river)
    places = {"The River": "", "The Tower": "", "The Fields": ""}
    plan = {"scenes": [{"id": "s01", "from": 0, "where": "The River", "side": "outside", "present": []}]}
    fr = {"f1": {"shot": "f1", "_t0": 1.0, "where": "The River"}, "f2": {"shot": "f2", "_t0": 5.0, "where": "The Tower"},
          "f3": {"shot": "f3", "_t0": 9.0, "where": "The Fields"}, "f4": {"shot": "f4", "_t0": 13.0, "where": "The River"}}
    PR.apply_answer({"splits": [{"moment": "f3", "where": "The Fields", "side": "outside"},
                                {"moment": "f2", "where": "The Tower", "side": "outside"}]},
                    plan["scenes"][0], plan, fr, places, LOOK["characters"], [0.0, 20.0], print)
    got = [(s_["where"], s_.get("t_split")) for s_ in plan["scenes"]]
    check("review: flashbacks split in time order and the scene resumes after them",
          got == [("The River", None), ("The Tower", 4.95), ("The Fields", 8.95), ("The River", 12.95)], got)
    fx = {"m1": {"shot": "m1", "_t0": 1.0, "who": [], "prompt": "Wide shot of the empty stone hall."}}
    PR.apply_answer({"moments": {"m1": {"who": ["The King"]}}}, plan["scenes"][0], plan, fx, places,
                    LOOK["characters"], [0.0, 20.0], print)
    check("review: a person the prompt does not show is not added", fx["m1"]["who"] == [], fx["m1"])
    starts = [(0.0, {"id": "s01", "present": ["The King"]})]
    PR.rules({}, [], None, starts, [{"t": 3.0, "kind": "quote", "speaker": "King"},
                                    {"t": 4.0, "kind": "quote", "speaker": "Edric"}], LOOK["characters"], print)
    check("review: every speaker is present", starts[0][1]["present"] == ["The King", "Edric"], starts[0][1])


def test_staging():
    import blockout_props as P
    import scene_stage as SS
    import remake_chapter as R
    lib = P.library()
    lib["bellows_t"] = {"kind": "set", "parts": [{"t": "box", "c": [0, 0, 0.5], "half": [0.5, 0.35, 0.15]}],
                        "mesh": "/x/model.blend", "extent_m": [1.4, 1.1, 1.3], "affordances": {},
                        "joints": [{"name": "board", "type": "hinge", "axis": "y", "pivot_m": [-0.76, -0.1, 1.33],
                                    "handle_m": [0.15, -0.24, 0.97], "limits": [0, 22],
                                    "cycle": {"action": "pump", "period_s": 1.4, "range": [0, 20]}}]}
    pr = {"prop": "bellows_t", "box": [2.0, 1.0, 0.0, 0.8, 0.4, 0.3], "yaw": 0}
    h0, pv = P.joint_handle(pr, "board", 0)
    h20, _ = P.joint_handle(pr, "board", 20)
    check("joint_handle: the handle at rest, in the world", abs(h0[0] - 2.15) < 1e-6 and abs(h0[2] - 0.97) < 1e-6, h0)
    check("joint_handle: the pump pushes the handle down", h20[2] < h0[2] - 0.2, (h0[2], h20[2]))
    check("joint_cycle: rest at 0, full at half the period",
          P.joint_cycle(pr, "board", 0) == 0 and abs(P.joint_cycle(pr, "board", 0.7) - 20) < 1e-6)
    shot = {"figures": [{"who": "Edric", "kind": "human", "at": [9, 9], "facing": 0}], "set": {"props": [pr]}}
    SS.operate_rule(shot, {"prompt": "Medium shot. Edric pumps the bellows_t hard."})
    f = shot["figures"][0]
    import numpy as np
    check("operate_rule: the person works the prop", shot.get("operate") == [
        {"who": "Edric", "prop": "bellows_t", "joint": "board", "hands": ["right", "left"]}], shot.get("operate"))
    check("operate_rule: standing at the handle, facing it",
          np.linalg.norm(np.array(f["at"]) - h0[:2]) < 0.5 and abs(f["facing"]) > 90, (f["at"], f["facing"]))
    with tempfile.TemporaryDirectory() as t:
        sp = Path(t) / "s.json"
        sp.write_text(json.dumps({**shot}))
        check("needs_clip: an operated prop renders its clip", R.needs_clip({"shot": str(sp)}) is True)


def test_acts():
    import interaction as IA
    import blockout_props as P
    cast = {"Edric", "Hulda", "Vorgrath", "Maren"}
    props = {"hammer", "anvil", "sword", "bellows_t", "tongs", "lantern"}
    ok, bad = IA.validate([{"do": "grasp", "who": "Edric", "hand": "right", "prop": "hammer"},
                           {"do": "grasp", "who": "Edric", "hand": "left", "prop": "tongs"},
                           {"do": "grasp", "who": "Nobody", "prop": "hammer"},
                           {"do": "juggle", "who": "Edric"},
                           {"do": "strike", "who": "Edric", "tool": "sword", "target": "Vorgrath:head"}],
                          cast, props, "Edric raises the hammer and swings the sword at the dragon")
    check("acts: valid ones kept", [a["do"] for a in ok] == ["grasp", "strike"], ok)
    check("acts: an unnamed tool (tongs), a stranger, an unknown verb rejected", len(bad) == 3, bad)
    ok2, bad2 = IA.validate([{"do": "grasp", "who": "Edric", "hand": "both", "prop": "bellows_t"},
                             {"do": "operate", "who": "Edric", "prop": "bellows_t"}], cast, props | {"bellows_t"},
                            "Edric pumps the bellows_t", {"bellows_t": "set"})
    check("acts: a set prop is worked, never grasped", [a["do"] for a in ok2] == ["operate"], (ok2, bad2))
    shot = {"figures": [{"who": "Edric", "kind": "human", "at": [0, 0]}, {"who": "Vorgrath", "kind": "creature"}],
            "set": {"props": [{"prop": "bellows_t", "box": [2.0, 1.0, 0.0, 0.8, 0.4, 0.3], "yaw": 0}]},
            "moment": {"id": "m1"}}
    fr = {"acts": [{"do": "strike", "who": "Edric", "tool": "sword", "target": "Vorgrath:head"}]}
    IA.compile_acts(shot, fr)
    check("compile: a strike at a creature part = sword in hand + an end contact",
          shot["figures"][0]["holds"] == {"right": "sword"} and shot["contacts"][0]["to"] == "Vorgrath:head", shot)
    fr = {"acts": [{"do": "operate", "who": "Edric", "prop": "bellows_t"}]}
    IA.compile_acts(shot, fr)
    check("compile: operate = the person at the handle, no held prop",
          shot.get("operate", [{}])[0].get("prop") == "bellows_t" and "holds" not in shot["figures"][0], shot)
    fr = {"acts": [{"do": "carry", "who": "Vorgrath", "carried": "Maren"}]}
    IA.carry_hints(fr)
    check("compile: carry -> the attach hint", fr["attach_hint"]["Maren"]["to"] == "Vorgrath", fr)


def test_director():
    import interaction as IA
    cast = {"Edric", "Vorgrath"}
    props = {"hammer", "anvil", "sword", "sword__blank", "shield"}
    a_, b_ = IA.split_answer({"acts": [{"do": "free", "who": "Edric", "hand": "left"}], "beats": ["Edric: raises", " "]})
    check("director: answer = acts + beats (blank beats dropped)", len(a_) == 1 and b_ == ["Edric: raises"], (a_, b_))
    check("director: an old bare list = acts", IA.split_answer([{"do": "free"}]) == ([{"do": "free"}], []))
    gaps = IA.undirected([{"do": "strike", "who": "Edric", "tool": "hammer", "target": "anvil"}])
    check("director: a strike with the left hand undirected is caught", gaps == ["Edric: left"], gaps)
    gaps = IA.undirected([{"do": "strike", "who": "Edric", "tool": "hammer", "target": "anvil"},
                          {"do": "grasp", "who": "Edric", "hand": "left", "prop": "sword__blank"}])
    check("director: both hands directed = no gap", gaps == [], gaps)
    ok, bad = IA.validate([{"do": "block", "who": "Edric", "tool": "sword", "against": "Vorgrath:claw_r"},
                           {"do": "block", "who": "Edric", "tool": "sword", "against": "Nobody:claw"},
                           {"do": "free", "who": "Edric", "hand": "left"}], cast, props,
                          "Edric parries the dragon's claw with his sword")
    check("director: block kept, a block against a stranger rejected, free kept",
          [x["do"] for x in ok] == ["block", "free"] and len(bad) == 1, (ok, bad))
    ok, _ = IA.validate([{"do": "grasp", "who": "Edric", "hand": "left", "prop": "sword__blank"}], cast, props,
                        "Edric strikes the star-iron. NOTES: the sword lies on the anvil, the left hand holds its hilt")
    check("director: a held item named only by the director's notes is allowed", len(ok) == 1, ok)
    shot = {"figures": [{"who": "Edric", "kind": "human", "at": [0, 0], "holds": {"right": "sword"}},
                        {"who": "Vorgrath", "kind": "creature"}], "set": {"props": []}, "moment": {"id": "m2"}}
    IA.compile_acts(shot, {"acts": [{"do": "block", "who": "Edric", "tool": "sword", "against": "Vorgrath:claw_r"},
                                    {"do": "free", "who": "Edric", "hand": "left"}]})
    c = (shot.get("contacts") or [{}])[0]
    check("compile: block = a contact of the held sword on the claw; the free hand recorded",
          c.get("to") == "Vorgrath:claw_r" and c.get("kind") == "block" and shot["figures"][0]["free_hands"] == ["left"],
          shot)
    p_ = IA.acts_prompt({"id": "s01", "where": "Forge", "side": "inside"}, [{"shot": "m1", "who": ["Edric"]}],
                        cast, {"hammer": ["strike_face"]}, ["the sword lies on the anvil"], ["m1: Edric: left"])
    check("director: the prompt carries the notes, the gaps and asks for mechanics beats",
          "DIRECTOR'S NOTES" in p_ and "m1: Edric: left" in p_ and "MECHANICS" in p_ and '"beats"' in p_)


def main() -> int:
    test_scenes()
    test_apply_scene()
    test_review_split()
    test_cutaway_and_present()
    test_motion_and_followups()
    test_staging()
    test_acts()
    test_director()
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
