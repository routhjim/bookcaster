"""The redraw-side checks' pure rules (tools/assets/redraw/accept.py, redraw.py; scenes/shot_people.py, seating.py).

Lit sources, furniture hue and chroma, hair and clothes held from key to key, the look verdict, the speaker's face,
a staged start action found in the redraw, a person in the engine still found in the redraw, the pick between a
draw and its re-draw; the character model's weight by head size; the words read from a figure's library entry; the
people facts written beside a key; seats for a recoloured chair and a start action's hands.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import staging_synth as S  # noqa: E402
import accept as AC  # noqa: E402
import redraw as RD  # noqa: E402
import seating  # noqa: E402
import shot_people as PPL  # noqa: E402


def lab(L, a, b, px=5000, **kw):
    return dict({"L": L, "a": a, "b": b, "px": px}, **kw)


# ------------------------------------------------------------------ lit sources
def test_a_lit_source_is_bright_and_warm_in_the_redraw_relative_to_the_engine():
    eng = lab(40, 10, 40, L90=80, b90=60)
    res = {"lit": {"k": {"lit": {"fire": {"engine": eng, "redraw": lab(30, 8, 30, L90=75, b90=50)}}}},
           "dark": {"k": {"lit": {"fire": {"engine": eng, "redraw": lab(8, 2, 6, L90=20, b90=10)}}}}}
    AC.lit_eval(res["lit"], res["dark"])
    assert res["lit"]["k"]["lit_ok"] == {"fire": True} and res["dark"]["k"]["lit_ok"] == {"fire": False}


def test_flames_in_a_dark_grate_count_as_lit_by_their_brightest_tenth():
    eng = lab(45, 12, 45, L90=85, b90=60)
    flames = lab(10, 3, 6, L90=round(AC.LIT_L * 85 + 2, 1), b90=round(AC.LIT_B * 60 + 3, 1))   # dark median, bright tenth
    res = {"k": {"lit": {"fire": {"engine": eng, "redraw": flames}}}}
    AC.lit_eval(res)
    assert res["k"]["lit_ok"]["fire"] is True
    just_under = lab(10, 3, 6, L90=round(AC.LIT_L * 85 - 3, 1), b90=60)
    res = {"k": {"lit": {"fire": {"engine": eng, "redraw": just_under}}}}
    AC.lit_eval(res)
    assert res["k"]["lit_ok"]["fire"] is False


def test_a_source_too_dim_in_the_engine_itself_is_not_judged():
    res = {"k": {"lit": {"candle": {"engine": lab(15, 2, 8, L90=24, b90=10), "redraw": lab(5, 1, 2, L90=8, b90=3)}}}}
    AC.lit_eval(res)
    assert res["k"]["lit_ok"] == {}


def test_item_fails_names_the_unlit_source_and_the_recoloured_piece():
    rec = {"people": {}, "lit_ok": {"fireplace": False}, "things_ok": {"armchair": False, "desk": True}}
    assert AC.item_fails(rec) == {"set": ["colour armchair", "lit fireplace"]}


# ------------------------------------------------------------------ furniture colour: hue and chroma
def _piece(hue_deg, chroma, L=30):
    return lab(L, chroma * math.cos(math.radians(hue_deg)), chroma * math.sin(math.radians(hue_deg)))


def test_a_piece_drawn_in_another_hue_fails_and_light_alone_does_not():
    keys = {f"k{i}": {"things": {"chair": _piece(68, 20, L=10 + 5 * i)}} for i in range(4)}     # light differs per shot
    keys["red"] = {"things": {"chair": _piece(68 - AC.LAB_HUE - 12, 20)}}
    AC.colour_eval(keys)
    assert all(keys[f"k{i}"]["things_ok"]["chair"] for i in range(4))
    assert keys["red"]["things_ok"]["chair"] is False


def test_a_piece_gone_grey_fails_on_chroma_against_its_own_median_ratio():
    eng = _piece(68, 23)
    keys = {f"k{i}": {"things": {"chair": _piece(68, 21)}, "things_engine": {"chair": eng}} for i in range(4)}
    keys["grey"] = {"things": {"chair": _piece(68, 21 * AC.LAB_C_LO * 0.6)}, "things_engine": {"chair": eng}}
    AC.colour_eval(keys)
    assert keys["grey"]["things_ok"]["chair"] is False and all(keys[f"k{i}"]["things_ok"]["chair"] for i in range(4))
    # a neutral piece (no colour in the engine still) is not judged on chroma
    grey_eng = {f"k{i}": {"things": {"wall": _piece(0, 3)}, "things_engine": {"wall": _piece(0, AC.LAB_MIN_C * 0.5)}}
                for i in range(3)}
    AC.colour_eval(grey_eng)
    assert all(v["things_ok"]["wall"] for v in grey_eng.values())


def test_a_small_mask_is_not_judged():
    keys = {"k0": {"things": {"chair": dict(_piece(68, 20), px=300)}}, "k1": {"things": {"chair": _piece(68, 20)}}}
    AC.colour_eval(keys)
    assert "chair" not in keys["k0"]["things_ok"] and keys["k1"]["things_ok"]["chair"]


# ------------------------------------------------------------------ hair and clothes from key to key
def _hair(dL, px=None):
    return {"redraw": {"L": 20 + dL, "px": px if px is not None else AC.HAIR_PX + 500}, "engine": {"L": 20, "px": 9999}}


def test_hair_is_held_to_the_persons_median_and_small_heads_are_not_judged():
    keys = {f"k{i}": {"hair": {"b": _hair(2.0 + 0.5 * i)}} for i in range(4)}
    keys["grey"] = {"hair": {"b": _hair(2.0 + AC.HAIR_L + 6)}}
    keys["small"] = {"hair": {"b": _hair(40.0, px=AC.HAIR_PX - 1)}}             # the region takes in the wall
    med = AC.hair_eval(keys)
    assert 2.0 <= med["b"] <= 4.0
    assert keys["grey"]["hair_ok"] == {"b": False} and keys["small"]["hair_ok"] == {}
    assert all(keys[f"k{i}"]["hair_ok"]["b"] for i in range(4))
    assert AC.item_fails(keys["grey"]) == {"b": ["hair"]}


def test_hair_needs_three_keys_of_a_person_before_it_judges():
    keys = {"k0": {"hair": {"b": _hair(2.0)}}, "k1": {"hair": {"b": _hair(30.0)}}}
    AC.hair_eval(keys)
    assert keys["k0"]["hair_ok"] == {} and keys["k1"]["hair_ok"] == {}


def test_clothes_lightness_is_held_from_key_to_key():
    w = lambda dL: {"redraw": {"L": 10 + dL, "px": 4000}, "engine": {"L": 10, "px": 4000}}
    keys = {f"k{i}": {"wear": {"a": w(1.0)}} for i in range(3)}
    keys["grey"] = {"wear": {"a": w(1.0 + AC.WEAR_L + 8)}}                      # a black coat drawn grey
    AC.wear_eval(keys)
    assert keys["grey"]["wear_ok"] == {"a": False} and keys["k0"]["wear_ok"] == {"a": True}
    assert AC.item_fails(keys["grey"]) == {"a": ["clothes colour"]}


# ------------------------------------------------------------------ the look of a piece
def test_look_verdict_identical_descriptions_count_as_one_and_a_hidden_piece_is_not_judged():
    run = {"_look_opts": [{"name": "armchair", "words": "leather armchair"}, {"name": "wingback", "words": "fabric wing-back"},
                          {"name": "bookcase", "words": "oak bookcase"}, {"name": "bookcase_2", "words": "oak bookcase"}]}
    res = {"k": {"look": {"wingback": {"want": 2, "got": 1, "fill": 0.8},            # drawn as the other chair
                          "armchair": {"want": 1, "got": 1, "fill": 0.8},
                          "bookcase": {"want": 3, "got": 4, "fill": 0.9},            # the twin: the same words
                          "desk": {"want": 2, "got": 1, "fill": AC.LOOK_FILL - 0.1}}}}   # mostly behind something
    AC.look_eval(run, res)
    lk = res["k"]["look"]
    assert lk["wingback"]["verdict"] is False and lk["armchair"]["verdict"] is True
    assert lk["bookcase"]["verdict"] is True and lk["desk"]["verdict"] is None
    assert AC.item_fails(res["k"]) == {"set": ["look wingback"]}


# ------------------------------------------------------------------ faces, actions, people found
FRONT = {"face_scores": [0.9, 0.9, 0.9], "face_contour": 0.8, "yaw": 0.1}
BACK = {"face_scores": [0.9, 0.9, 0.9], "face_contour": 0.8, "yaw": AC.YAW_PROFILE + 0.3}    # scored high, turned away


def test_the_face_check_is_for_the_speaker_only_and_a_back_of_a_head_is_not_a_face():
    res = {"k": {"people": {"a": {"named": True, "engine": dict(FRONT), "redraw": dict(BACK)},
                            "b": {"named": False, "engine": dict(FRONT), "redraw": dict(BACK)},
                            "c": {"named": True, "engine": dict(BACK), "redraw": dict(BACK)}}}}
    AC.face_eval(res)
    p = res["k"]["people"]
    assert p["a"]["face_ok"] is False            # the speaker's face is in the engine still and not in the redraw
    assert "face_ok" not in p["b"]               # not speaking: may be drawn from behind
    assert "face_ok" not in p["c"]               # the engine itself does not show the face: nothing to hold
    assert AC.failures(p["a"]) == ["face"] and AC.failures(p["b"]) == []


def test_a_speaker_turned_the_other_way_fails():
    left = dict(FRONT, yaw=AC.YAW_TURN + 0.2)
    right = dict(FRONT, yaw=-(AC.YAW_TURN + 0.2))
    res = {"k": {"people": {"a": {"named": True, "engine": left, "redraw": right},
                            "b": {"named": True, "engine": left, "redraw": dict(left)}}}}
    AC.face_eval(res)
    assert res["k"]["people"]["a"]["face_ok"] is False and res["k"]["people"]["b"]["face_ok"] is True


def _pose(chin=(0.5, 0.40), eyes_y=0.30, nose_y=None, hands=((0.45, 0.42), (0.55, 0.42))):
    t = {"chin": list(chin), "l_eye": [0.48, eyes_y], "r_eye": [0.52, eyes_y],
         "l_hand": list(hands[0]) if hands[0] else None, "r_hand": list(hands[1]) if hands[1] else None}
    if nose_y is not None:
        t["nose"] = [0.5, nose_y]
    return {"pts": t}


def test_a_staged_start_action_must_also_be_in_the_redraw():
    fh = (0.40 - 0.30) * 2.2
    at_chin, in_lap = _pose(), _pose(hands=((0.45, 0.40 + 2.5 * fh), (0.55, 0.40 + 2.5 * fh)))
    assert AC.action_there("veil_lift", at_chin) is True and AC.action_there("veil_lift", in_lap) is False
    one_up = _pose(hands=((0.45, 0.42), (0.55, 0.40 + 2.5 * fh)))
    assert AC.action_there("veil_lift", one_up) is False             # both hands, not one
    bowed = _pose(nose_y=0.30 + (AC.HEAD_BOW + 0.03) * fh)
    level = _pose(nose_y=0.30 + (AC.HEAD_BOW - 0.03) * fh)
    assert AC.action_there("look_at_sleeve", bowed) is True and AC.action_there("look_at_sleeve", level) is False
    assert AC.action_there("veil_lift", {"pts": {}}) is None         # nothing measured: not judged
    res = {"k": {"people": {"b": {"action": "veil_lift", "engine": at_chin, "redraw": in_lap},
                            "c": {"action": "veil_lift", "engine": in_lap, "redraw": in_lap},
                            "d": {"action": "look_at_sleeve", "engine": bowed, "redraw": bowed}}}}
    AC.action_eval(res)
    p = res["k"]["people"]
    assert p["b"]["action_ok"] is False and AC.failures(p["b"]) == ["action"]
    assert "action_ok" not in p["c"]             # the engine still does not show it: a staging fault, not a redraw's
    assert p["d"]["action_ok"] is True


def test_a_person_in_the_engine_still_must_be_found_in_the_redraw():
    there = {"person_found": {"engine": AC.PERSON_FOUND + 0.5, "redraw": AC.PERSON_FOUND + 0.3}}
    gone = {"person_found": {"engine": AC.PERSON_FOUND + 0.5, "redraw": AC.PERSON_FOUND * 0.3}}
    never = {"person_found": {"engine": AC.PERSON_FOUND * 0.5, "redraw": 0.0}}       # the detector misses the engine's too
    assert AC.failures(there) == [] and AC.failures(gone) == ["missing"] and AC.failures(never) == []


def test_neck_is_judged_against_the_characters_median_over_the_run():
    keys = {f"k{i}": {"people": {"a": {"redraw": {"neck_ratio": 0.9}, "engine": {"neck_ratio": 1.4}}}} for i in range(4)}
    keys["long"] = {"people": {"a": {"redraw": {"neck_ratio": 0.9 * AC.NECK_TOL + 0.2}, "engine": {"neck_ratio": 0.5}}}}
    AC.neck_eval(keys)
    assert keys["long"]["people"]["a"]["neck_ok"] is False and keys["k0"]["people"]["a"]["neck_ok"] is True


def test_pick_takes_the_redraw_only_when_it_fails_less(tmp_path):
    run = {"out": str(tmp_path)}
    checked = {"k1": {"people": {"a": {"redraw": {"arms_ok": False, "hands_ok": False}}}},
               "k2": {"people": {"a": {"redraw": {"arms_ok": False}}}},
               "k3": {"people": {"a": {"redraw": {}}}}}
    retry = {"k1__x": {"people": {"a": {"redraw": {"arms_ok": True, "hands_ok": False}}}},
             "k2__x": {"people": {"a": {"redraw": {"arms_ok": False}}}}}
    AC.cmd_pick(run, checked, retry, tmp_path / "acc.json")
    acc = json.loads((tmp_path / "acc.json").read_text())
    assert acc["k1"]["id"] == "k1__x" and acc["k1"]["fails"] == {"a": ["hands"]} and acc["k1"]["redrawn"]
    assert acc["k2"]["id"] == "k2" and acc["k2"]["fails"]            # no better: the first draw stays, labelled
    assert acc["k3"]["id"] == "k3" and acc["k3"]["fails"] == {} and not acc["k3"]["redrawn"]
    assert set(acc) == {"k1", "k2", "k3"}                            # every key ends with a picture


# ------------------------------------------------------------------ the character model's weight by head size
def test_the_weight_rises_as_the_head_gets_smaller_in_a_two_person_key():
    run = {"lora_w": {"1": 0.8, "2": 0.63}}
    lo, hi = RD.LORA_HEAD
    assert RD.lora_by_head(0.63, run, hi, 2) == 0.63 and RD.lora_by_head(0.63, run, hi + 0.1, 2) == 0.63
    assert RD.lora_by_head(0.63, run, lo, 2) == 0.8 and RD.lora_by_head(0.63, run, lo - 0.1, 2) == 0.8
    mid = RD.lora_by_head(0.63, run, (lo + hi) / 2, 2)
    assert abs(mid - (0.63 + 0.8) / 2) < 0.002
    ws = [RD.lora_by_head(0.63, run, lo + (hi - lo) * i / 10, 2) for i in range(11)]
    assert ws == sorted(ws, reverse=True)                            # a straight ramp, never above the one-person weight
    assert RD.lora_by_head(0.8, run, 0.1, 1) == 0.8 and RD.lora_by_head(0.63, run, None, 2) == 0.63


# ------------------------------------------------------------------ words from the figure's own entry
def test_look_words_come_from_the_entrys_facts_and_missing_facts_are_named():
    w, miss = RD.look_words({"hair_colour": "black", "hair_worn": "short hair, slicked back", "facial_hair": "none"}, "his")
    assert w == "short black hair, slicked back; clean-shaven" and miss == []
    w, _ = RD.look_words({"hair_colour": "black with grey streaks", "hair_worn": "hair pinned up, in a bun",
                          "facial_hair": "none"}, "her")
    assert w == "hair pinned up, in a bun, the hair black with grey streaks"      # "clean-shaven" is said of men only
    w, miss = RD.look_words({"hair_colour": "brown", "facial_hair": "a full moustache", "missing_layers": ["no cloak"]}, "his")
    assert w == "brown hair; a full moustache; no cloak" and miss == ["hair_worn"]
    assert RD.look_words({}, "his") == ("", list(RD.LOOK_FACTS))


def test_a_stance_or_wardrobe_variant_takes_its_base_entrys_facts(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    S.add_character(lib, "fig_b", {"facts": {"hair_colour": "black"}})
    S.add_character(lib, "fig_b_seated_leather", {})
    S.add_character(lib, "fig_b_seated_leather_veiled", {})
    S.add_character(lib, "fig_c_seated_x", {"facts": {"hair_colour": "grey"}})
    import config
    monkeypatch.setattr(config, "LIBRARY", lib)
    assert RD.entry_facts("fig_b_seated_leather_veiled") == {"hair_colour": "black"}
    assert RD.entry_facts("fig_c_seated_x") == {"hair_colour": "grey"}           # its own, when it has them
    assert RD.entry_facts("fig_z") == {} and RD.entry_facts(None) == {}


# ------------------------------------------------------------------ the people facts beside a key
LOOKS = {"Alder": "a tall man, wearing a long grey coat, a scarf and boots",
         "Birch": "a woman, wearing a green travelling dress with a wide collar",
         "Cedar": "a man of middle height, wearing a brown suit"}
CAST = {"Alder": {"key": "a"}, "Birch": {"key": "b"}, "Cedar": {"key": "c"}}


def _scene(tmp, people, shot_facts=None):
    (tmp / "truth.json").write_text(json.dumps({"shots": {"s1": {"people": people}}}))
    if shot_facts:
        (tmp / "staged.json").write_text(json.dumps({"shot_facts": {"s1": shot_facts}}))
    return tmp


def test_people_in_a_key_are_those_with_enough_of_their_own_pixels_named_by_their_look(tmp_path):
    va = PPL.VISIBLE_AREA
    sc = _scene(tmp_path, {"a": {"area": va * 6, "head_to_camera_deg": 10, "box": [0.5, 0.1, 0.7, 0.9]},
                           "b": {"area": va * 3, "head_to_camera_deg": 150, "box": [0.1, 0.2, 0.3, 0.9]},
                           "c": {"area": va * 0.4, "head_to_camera_deg": 20, "box": [0.95, 0.2, 1.0, 0.9]}})
    f = PPL.people(sc, "s1", LOOKS, CAST)
    assert f["count"] == 2                                            # the sliver at the edge is not a person in it
    assert [p["look"] for p in f["visible"]] == ["the woman in the green travelling dress with a wide collar",
                                                 "the man in the long grey coat"]      # left to right
    assert [p["seen"] for p in f["visible"]] == ["back", "front"]
    txt = PPL.lines(f)
    assert "2 people" in txt and "(seen from behind)" in txt and "on the left" in txt and "on the right" in txt
    assert not any(n in txt for n in CAST)                            # look phrases, never names
    assert f["move"] is None


def test_a_single_and_a_move_shots_facts(tmp_path):
    sc = _scene(tmp_path, {"a": {"area": 0.2, "head_to_camera_deg": 5, "box": [0.3, 0.1, 0.7, 1.0]},
                           "c": {"area": 0.004, "head_to_camera_deg": 30, "box": [0.0, 0.3, 0.02, 0.9]}},
                {"moves": {"who": "b", "to": "armchair", "to_words": "the armchair (buttoned leather)",
                           "posture_end": "sitting"}, "fixed": {"a": "standing", "c": "standing"}})
    f = PPL.people(sc, "s1", LOOKS, CAST)
    assert f["count"] == 1 and "1 person" in PPL.lines(f)
    assert f["move"]["who"].startswith("the woman in") and f["move"]["ends"] == "sitting"
    txt = PPL.lines(f)
    assert "walks to the armchair (buttoned leather) and ends sitting in it" in txt
    assert "Everyone else keeps their place and posture" in txt and "standing" in txt


def test_look_phrase_reads_the_first_thing_worn():
    assert PPL.look_phrase(LOOKS["Cedar"]) == "the man in the brown suit"
    assert PPL.look_phrase("someone in a room") is None


# ------------------------------------------------------------------ seats
def test_a_recoloured_chair_seats_the_stance_settled_for_its_mesh(tmp_path):
    lib = S.make_library(tmp_path, [{"name": "wingback", "position": [0, 0], "seat": True, "asset": "chair_red",
                                     "variant_of": "chair_gold"}])
    (lib / "prop" / "chair_gold").mkdir(parents=True)
    (lib / "prop" / "chair_gold" / "asset.json").write_text(json.dumps({"size_m": [0.8, 0.8, 1.0]}))
    S.add_character(lib, "fig_b_seated_wing", {"seat_asset": "chair_gold"})
    S.add_character(lib, "fig_b_seated_other", {"seat_asset": "chair_leather", "seat_z_m": 0.9})
    seats = seating.seats(lib, lib / "set" / "room_x")
    assert seating.same_mesh(lib, "chair_gold", "chair_red") and not seating.same_mesh(lib, "chair_gold", "chair_leather")
    assert seating.seated_variant(lib, "fig_b", seats["wingback"]) == "fig_b_seated_wing"


def test_hands_in_the_lap_are_not_added_over_a_start_action(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [{"name": "armchair", "position": [0, 0], "seat": True}])
    monkeypatch.setattr(seating, "lap_steps", lambda z: [["lap", z]])
    base = {"seat": "armchair", "position": [0, 0], "facing": [0, -1], "id": "fig"}
    spec = {"set": "room_x", "characters": {
        "b": dict(base, body_pose={"pose": "seated", "params": {"hands": "lap"}})}}
    seating.plan(spec, lib)
    assert spec["characters"]["b"]["body_pose"]["params"]["steps"][0][0] == "lap"
    spec = {"set": "room_x", "characters": {
        "b": dict(base, body_pose={"pose": "seated", "params": {"hands": "lap", "action": "veil_lift"}})}}
    seating.plan(spec, lib)
    pr = spec["characters"]["b"]["body_pose"]["params"]
    assert not pr.get("steps") and "hands" not in pr                  # the action has the hands


# ------------------------------------------------------------------ the lit mask grown over the flames (torch)
def test_the_lit_mask_grows_upward_over_the_flames(tmp_path):
    torch = pytest.importorskip("torch")
    import numpy as np
    from PIL import Image
    import keycheck as KC
    H, W = 200, 300
    mask = np.zeros((H, W), np.uint8)
    mask[150:160, 100:200] = 255                                      # the grate's glow: a strip at the bottom
    Image.fromarray(mask).save(tmp_path / "m.png")
    img = torch.zeros(3, H, W)
    img[:, 120:150, 100:200] = torch.tensor([255.0, 180.0, 40.0]).view(3, 1, 1)    # flames standing above the strip
    flat = KC.lab_of(img, str(tmp_path / "m.png"))
    grown = KC.lab_of(img, str(tmp_path / "m.png"), rise=KC.LIT_RISE)
    assert flat["L90"] < 5 and grown["L90"] > 60 and grown["px"] > flat["px"]
    assert grown["px"] == (10 + max(30, int(KC.LIT_RISE * 10))) * 100           # upward only, by the stated rise
