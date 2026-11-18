"""A landmark that is part of the land (11-18): a place in the open whose world showed a cave in its hillside was
also given a made cave (a boulder with a hole) and a cube of rock for a ledge, stood on the ground before it, and
they filled or blocked a third of the place's keys. A landmark the world's panorama shows is found there instead
(pano_shell.mark_of): people are sent to the ground before it, and no piece is stood for it. Synthetic data only."""
import json
import sys
from pathlib import Path

import numpy as np

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", "img3d", "redraw", ""):
    sys.path.insert(0, str(A / sub))
import pano_shell as PS  # noqa: E402
import pieces as P  # noqa: E402
import place_build as PB  # noqa: E402
import setfacts  # noqa: E402


def pano(H=128, W=256):
    dz = np.cos(np.pi * (np.arange(H) + 0.5) / H)[:, None] * np.ones((1, W))
    d = np.where(dz < -1e-3, 1.6 / np.maximum(-dz, 1e-3), 150.0)
    return dz, d, PS.pano_dirs(H, W)


def test_mark_of_a_thing_ahead():
    dz, d, dirs = pano()
    # a hillside 20 m ahead (-y), seen from the horizon up: land, not sky
    ahead = (dirs[..., 1] < -0.8) & (dz > -0.05) & (dz < 0.25)
    d = np.where(ahead, 20.0, d)
    sky = (dz > 0) & ~ahead
    mk = PS.mark_of(np.where(ahead, 0.9, 0.05), d, sky)
    assert mk and mk["toward"][1] < -0.95 and abs(mk["toward"][0]) < 0.2
    # people stand on the flat ground toward it, not 20 m off
    assert abs(np.hypot(*mk["stand"]) - (PS.FLAT_R_M - 1.5)) < 1e-6 and mk["stand"][1] < 0
    assert mk["near_m"] > 15


def test_mark_of_nothing_shown():
    dz, d, _ = pano()
    assert PS.mark_of(np.full(d.shape, 0.1), d, dz > 0) is None
    few = np.full(d.shape, 0.05)
    few[70:72, 10:14] = 0.9                                  # a few points: not a thing of the land
    assert PS.mark_of(few, d, dz > 0) is None


SPEC = {"kind": "outdoors", "period": "medieval", "room": "Open ground.", "_phrases": ["by the arch", "on the bank"],
        "pieces": [{"id": "p1", "kind": "landmark", "look": "a stone arch", "phrases": ["by the arch"], "longest_m": 5},
                   {"id": "p2", "kind": "landmark", "look": "a grass bank", "phrases": ["on the bank"], "longest_m": 3}],
        "named_places": {"by the arch": "p1", "on the bank": "p2"}, "land_places": {}}


def test_terrain_asks_for_its_landmarks():
    sp = PB.outdoors(dict(SPEC), SPEC, {"concept": "/c/c.png", "surfaces": {}})
    t = next(p for p in sp["pieces"] if p["kind"] == "terrain")
    assert t["marks"] == {"p1": "a stone arch", "p2": "a grass bank"}
    req = P.generation_request(t)
    spec = next(s for s in req["plan"]["steps"] if s["step"] == "spec")["spec"]
    assert spec["marks"] == t["marks"]
    # the world is the same asset with or without them (its panorama is kept; an older build is made again)
    assert req["asset_id"] == P.generation_request({k: v for k, v in t.items() if k != "marks"})["asset_id"]


def test_world_marks_in_the_sets_frame(tmp_path):
    sp = PB.outdoors(dict(SPEC), SPEC, {"concept": "/c/c.png", "surfaces": {}})
    t = next(p for p in sp["pieces"] if p["kind"] == "terrain")
    d = tmp_path / "prop" / P.generation_request(t)["asset_id"]
    d.mkdir(parents=True)
    (d / "asset.json").write_text(json.dumps({"world": {"stand_m": [1.0, -2.0, 40.0], "marks_m": {
        "p1": {"stand": [1.0, -9.5, 40.0], "toward": [0.0, -1.0], "near_m": 22.0}}}}))
    got = PB.world_marks(sp, tmp_path)
    assert list(got) == ["p1"]                               # (the bank is not in the picture: its piece stands)
    assert got["p1"]["at"] == [0.0, -7.5] and got["p1"]["look"] == "a stone arch"
    assert PB.world_marks({"pieces": [dict(t, marks={})]}, tmp_path) == {}


def test_a_mark_is_a_place_on_the_ground(tmp_path):
    sd = tmp_path / "sets" / "s1"
    sd.mkdir(parents=True)
    (sd / "spec.json").write_text(json.dumps({"size_m": [20, 20, 6], "fixtures": [], "props": [], "structure": [],
                                              "marks": [{"name": "land_p1", "position": [0.0, -7.5],
                                                         "toward": [0.0, -1.0], "look": "a stone arch"}]}))
    t = next(x for x in setfacts.things(sd) if x["name"] == "land_p1")
    assert t["kind"] == "mark" and t["asset"] is None and t["center"] == [0.0, -7.5, 0.0] and t["half"][:2] == [0.0, 0.0]


def test_a_work_piece_keeps_its_own_words():
    spec = {"kind": "room", "pieces": [
        {"id": "p1", "kind": "work_piece", "look": "A kiln, a brick oven with a round door.", "phrases": ["at the kiln"]},
        {"id": "p2", "kind": "bench", "look": "A bench of planks", "phrases": ["on the bench"]}]}
    concept = {"space": "room", "spec": {"p1": {"in_picture": True, "look": "brick, red", "crop": "/c/p1.png"},
                                         "p2": {"in_picture": True, "look": "wood, pale"}}}
    got = {p["id"]: p for p in PB.with_concept(spec, concept)["pieces"]}
    assert got["p1"]["look"] == "A kiln, a brick oven with a round door; brick, red" and got["p1"]["source"] == "/c/p1.png"
    assert got["p2"]["look"] == "wood, pale"                 # (a kind that names its thing takes the picture's look)


def test_a_creature_at_the_edge_is_said_as_itself():
    sys.path.insert(0, str(A / "redraw"))
    import redraw as RD
    cast = {"a": {"noun": "dragon", "pos": "its", "subj": "it"}, "b": {"noun": "woman", "wear": "a grey cloak, a hood"}}
    edge = [("b", None, 0, "standing", 0, None), ("a", None, 0, "standing", 0, None)]
    w = RD.edge_words(edge, cast)
    assert "another person's shoulder and clothes is seen (a grey cloak, a hood)" in w
    assert "only part of the dragon's body is seen" in w
    only = RD.edge_words(edge[1:], cast)
    assert "another person" not in only and "dragon's body" in only
    ppl = [("a", [0, 0, 1, 1], 10.0, "standing", 10.0, None)]
    assert "wears" not in RD.wear_words(ppl, "", None, cast)


def test_a_creature_is_said_by_what_of_it_the_frame_shows():
    """A creature alone in a frame is never said as a person (waist up, standing): its head alone, its upper body or
    the whole of it, from its measured head in the frame, and from below when the camera looks up at it."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("rd_beast", Path(__file__).resolve().parents[1] / "assets/redraw/redraw.py")
    rd = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(rd)
    cast = {"a": {"noun": "beast", "trigger": "chr_b", "pos": "its", "subj": "it", "wear": "", "creature": True}}

    def shot(top, chin, cam_z=1.0):
        return {"people": {"a": {"box": [0.2, 0.0, 1.0, 1.0], "area": 0.5, "head_to_camera_deg": 10.0,
                                 "body_to_camera_deg": 5.0, "joints2d": {"head_top": [0.5, top, True],
                                                                         "chin": [0.5, chin, True]}}},
                "camera": {"position": [4.0, 0.0, cam_z], "target": [0.0, 0.0, 1.0]}}
    def words(sh):
        return rd.image_prompt("medium", rd.people(sh, {}), "", "the land", "", cast=cast, with_body=True,
                               speaker="a", shot=sh, facts=True, outdoors=True)
    head = words(shot(0.1, 0.8))
    assert "Only the beast's head is in the picture" in head and "close shot out of doors of a beast's head" in head
    assert "waist" not in head and "standing" not in head and "mouth" not in head and "wears" not in head
    upper = words(shot(0.3, 0.6, cam_z=-2.0))
    assert "neck and the upper part of its body" in upper and "seen from below" in upper
    whole = words(shot(0.4, 0.5))
    assert "The whole beast is in the picture" in whole and "seen from" not in whole.split("picture, head to tail.")[1][:40]
