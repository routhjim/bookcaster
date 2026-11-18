"""A creature of the story staged as the library's rigged creature (scenes/creature.py, characters/creature_cast.py,
scenes/from_plan.py, staging_pass's creature pose, speakers' frame edges, shot_people's phrase), on a temp library.

A rigged creature's record gives its engine mesh, clip, poses and size; the cast question offers only rigged
creatures and keeps an answer naming one of them; the plan's creature is placed standing (never seated or bedded),
clear of people by its reach, in the pose the pass chose (carried to shots that do not say); a creature's body
past a frame's edge is not a cut (its head is); its phrase is the look's head noun.
"""
import json
import math
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "characters"))
import staging_synth as S  # noqa: E402
import from_plan as FP  # noqa: E402
import creature as CRE  # noqa: E402
import creature_cast as CC  # noqa: E402
import shot_people as PPL  # noqa: E402
import speakers as SP  # noqa: E402

POSES = ["rest", "head_low", "mouth_open"]


def add_creature(lib: Path, cid: str, size=(6.0, 9.0, 5.0), rigged=True, desc="A large scaled beast on four legs."):
    d = lib / "creature" / cid
    (d / "rig").mkdir(parents=True, exist_ok=True)
    (d / "asset.json").write_text(json.dumps({"id": cid, "size_m": list(size), "description": desc}))
    ue = {"skeletal_mesh": f"/Game/C/{cid}_rig", "anim": f"/Game/C/{cid}_rig_Anim", "poses": POSES} if rigged else {}
    (d / "rig" / "bones.json").write_text(json.dumps({"bones": [{"name": "head", "head": [0, -3, 4], "tail": [0, -4, 3.5]}],
                                                       "ue": ue}))


@pytest.fixture
def lib(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [{"name": "desk", "position": [-6.0, -6.0], "size": [1.4, 0.7, 0.8]},
                                    {"name": "armchair", "position": [6.0, 6.0], "size": [0.8, 0.8, 1.0], "seat": True}],
                         size=(20.0, 20.0, 8.0))
    S.add_character(lib, "fig_a")
    add_creature(lib, "beast_01")
    add_creature(lib, "beast_norig_01", rigged=False)
    monkeypatch.setattr(FP.C, "LIBRARY", lib)
    return lib


# ------------------------------------------------------------------ the record
def test_a_rigged_creature_record_gives_its_staging_facts(lib):
    r = CRE.rig(lib, "creature/beast_01")
    assert r["mesh"].endswith("beast_01_rig") and r["poses"] == POSES and r["radius_m"] == 4.5
    assert r["head_len_m"] == pytest.approx(math.dist([0, -3, 4], [0, -4, 3.5]), abs=1e-3)
    assert CRE.pose_time(r, "mouth_open") == pytest.approx(2 / 24) and CRE.pose_time(r, "unknown") == 0.0
    assert CRE.rig(lib, "beast_norig_01") is None                     # no engine rig: not a stageable creature
    assert [x["asset"] for x in CRE.rigged(lib)] == ["creature/beast_01"]
    assert CRE.is_id("creature/beast_01") and not CRE.is_id("man_x_auto_01")


def test_a_creature_head_reads_as_a_face_from_its_head_and_jaw_joints():
    m = CRE.head_measure([0, 0, 0], [0, -1], 5.0, [1.0, -3.0, 4.0], [2.0, -3.0, 3.6], 1.0)
    assert m["head_fwd_xy"] == [1.0, 0.0]                             # the head turned: forward from head to jaw
    assert m["eye_z"] == 4.0 and m["head_top_z"] > m["eye_z"] > m["chin_z"]
    assert CRE.head_measure([0, 0, 0], [0, -1], 5.0, [0, 0, 4], None, 1.0)["head_fwd_xy"] == [0, -1]


# ------------------------------------------------------------------ the cast question
def test_the_cast_question_offers_rigged_creatures_and_keeps_only_a_named_one(lib):
    cands = CRE.rigged(lib)
    q = CC.prompt("a huge beast with grey scales", cands)
    assert "beast_01" in q and "beast_norig_01" not in q and CC.NONE in q
    assert CC.parse("beast_01", cands) == "creature/beast_01"
    assert CC.parse("The model is beast_01.", cands) == "creature/beast_01"
    assert CC.parse("none", cands) is None and CC.parse("beast_011", cands) is None


# ------------------------------------------------------------------ placing and posing
def run(tmp_path, monkeypatch, shots, pass_=None):
    cast = {"Ash": {"key": "a", "id": "fig_a"}, "Beast": {"key": "b", "id": "creature/beast_01", "creature": True}}
    (tmp_path / "plan.json").write_text(json.dumps({sid: [sh] for sid, sh in shots.items()}))
    (tmp_path / "shots.txt").write_text("\n".join(shots))
    (tmp_path / "cast.json").write_text(json.dumps(cast))
    argv = ["from_plan.py", str(tmp_path / "plan.json"), str(tmp_path / "shots.txt"), str(tmp_path / "cast.json"),
            str(tmp_path / "out")]
    if pass_ is not None:
        (tmp_path / "pass.json").write_text(json.dumps(pass_))
        argv += ["--pass", str(tmp_path / "pass.json")]
    monkeypatch.setattr(sys, "argv", argv)
    FP.main()
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "out").glob("pl_g*.json"))]


def shot(people, kind="quiet"):
    return {"kind": kind, "framing": "wide", "place": "a hall", "room": "a hall", "fixtures": [], "people": people}


def test_a_creature_stands_clear_of_people_in_the_pose_the_pass_chose(lib, tmp_path, monkeypatch):
    s1 = shot([{"name": "Beast", "pose": "lying", "at": "in the middle", "facing": "Ash"},
               {"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "Beast"}])
    s2 = shot([{"name": "Ash", "pose": "standing", "at": "near the desk", "facing": "Beast"}])
    specs = run(tmp_path, monkeypatch, {"quiet-1": s1, "quiet-2": s2},
                pass_={"creature_pose": {"quiet-1": {"Beast": "head_low"}}})
    b, a = specs[0]["characters"]["b"], specs[0]["characters"]["a"]
    assert b["asset"] == "creature/beast_01" and b["creature"] and b["pose"] == "head_low"
    assert "seat" not in b and "bed" not in b and "body_pose" not in b
    assert math.dist(a["position"], b["position"]) >= FP.PERSON_GAP_M + 4.5 - 1e-6     # clear of its reach
    assert abs(b["position"][0]) <= 10 - 0.35 - 4.5 + 1e-6 and abs(b["position"][1]) <= 10 - 0.35 - 4.5 + 1e-6
    assert all(sp["characters"]["b"]["pose"] == "head_low" for sp in specs)        # carried to the next shot
    assert math.hypot(*b["facing"]) == pytest.approx(1.0, abs=1e-3)


def test_marks_keep_people_clear_of_a_creature():
    others = [[0.0, 0.0, 3.0]]
    assert FP.too_near([2.0, 0.0], others) and not FP.too_near([4.2, 0.0], others)
    assert FP.too_near([2.0, 0.0], [[0.0, 0.0]], own_r=1.0) and not FP.too_near([2.0, 0.0], [[0.0, 0.0]])
    p = FP.free_mark([0.0, 0.0], others, [], [20.0, 20.0], (0.6, 1.6))
    assert math.dist(p, [0, 0]) >= FP.PERSON_GAP_M + 3.0 - 1e-6


# ------------------------------------------------------------------ frame edges, phrase
def test_a_creatures_body_past_the_frames_edge_is_not_a_cut():
    m = {"eye_xy": [0.0, 0.0], "eye_z": 4.0, "head_top_z": 4.3, "chin_z": 3.6, "head_fwd_xy": [0.0, -1.0],
         "pelvis_xy": [6.0, 3.0], "waist_z": 2.0, "feet_z": 0.0}
    cam = {"position": [0.0, -6.0, 4.0], "target": [0.0, 0.0, 4.0], "focal_mm": 50}
    assert SP.frame_facts(cam, m, (1152, 640))["cut"]                 # as a person: the body runs out of the frame
    assert not SP.frame_facts(cam, dict(m, creature=True), (1152, 640))["cut"]


def test_a_creatures_phrase_is_its_looks_head_noun():
    assert PPL.creature_phrase("a huge grey beast with long horns, yellow eyes") == "the beast"
    assert PPL.phrase("a huge grey beast with long horns", {"creature": True}) == "the beast"
    assert PPL.phrase("someone in a room", {}) is None
