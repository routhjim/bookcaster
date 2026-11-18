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
    assert abs(np.hypot(*mk["stand"]) - PS.SPOT_MAX_M) < 1e-6 and mk["stand"][1] < 0
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


def test_a_landmark_is_a_piece_on_the_set_where_the_picture_has_it():
    """What stands in a place in the open is built on the set's own ground, each thing where the concept picture
    measured has it, about a middle on the open ground in front; the world asks for nothing of it; what lies far
    off is left to the panorama unless the plan sends people there."""
    con = {"concept": "/c/c.png", "surfaces": {}, "seen": [], "depth": {"eye": [0.0, -5.0], "things": {
        "spec:p1": {"xy": [-3.0, 7.0], "size_m": [5.0, 4.0, 4.0]},
        "spec:p2": {"xy": [4.0, 9.0], "size_m": [3.0, 2.0, 1.5]},
        "spec:p9": {"xy": [0.0, 80.0], "size_m": [40.0, 40.0, 30.0]}}}}
    spec = dict(SPEC, pieces=SPEC["pieces"] + [{"id": "p9", "kind": "landmark", "look": "a far hill", "phrases": [],
                                                 "source": "/c/p9.png"}])
    sp = PB.outdoors(json.loads(json.dumps(spec)), spec, con)
    by = {p["id"]: p for p in sp["pieces"]}
    assert "p9" not in by and "marks" not in by["t1"]
    # the nearest piece's front is 4 m beyond the middle; both keep their places side to side
    assert by["p1"]["at_xy"] == [-3.5, 6.0] and by["p2"]["at_xy"] == [3.5, 8.0]
    assert by["p1"]["_seen_size_m"] == [5.0, 4.0, 4.0] and by["p1"]["facing_xy"] == [0.0, -1.0]
    side = sp["_room_m"][0]
    assert side >= 2 * (8.0 + 1.0 + PB.OPEN_ROUND_M) - 0.1 and side <= PB.OPEN_STAGE_M[1]
    assert sp["named_places"] == {"by the arch": "p1", "on the bank": "p2"}
    # a named landmark the picture was asked about and does not show is the open ground, not a piece from words
    con2 = dict(con, spec={"p1": {"in_picture": True}, "p2": {"in_picture": False}})
    sp2 = PB.outdoors(json.loads(json.dumps(SPEC)), SPEC, con2)
    assert [p["id"] for p in sp2["pieces"]] == ["p1", "t1"] and sp2["named_places"]["on the bank"] == "floor"
    # with no measure every named piece still stands, placed by the layout
    sp0 = PB.outdoors(json.loads(json.dumps(SPEC)), SPEC, {"concept": "/c/c.png", "surfaces": {}})
    assert [p["id"] for p in sp0["pieces"]] == ["p1", "p2", "t1"] and "_room_m" not in sp0


def test_the_worlds_ground_is_real_and_the_panorama_lies_beyond_it():
    dz, d, dirs = pano(256, 512)
    sky = dz > 0.02
    hz = np.sqrt(1 - dz * dz)
    wall = (dirs[..., 1] < -0.9) & (d * 0 + 5.0 / np.maximum(hz, 1e-3) * dz + 1.6 > 0) & (5.0 / np.maximum(hz, 1e-3) * dz + 1.6 < 3)
    d = np.where(wall, 5.0 / np.maximum(hz, 1e-3), np.minimum(d, 135.0))
    sky = sky & ~wall
    b = PS.backdrop(d, sky)
    rr, over = b * hz, b * dz + 1.6
    assert rr[~sky & (over > 0.02)].min() >= PS.FLAT_R_M - 1e-6           # nothing stands nearer than the ground's edge
    assert np.allclose(over[~sky & (rr < PS.FLAT_R_M - 1) & (over <= 0.02)], -PS.GROUND_UNDER_M)
    gv, gf, guv = PS.ground_disk([0.0, 0.0, 10.0], 100)
    assert gf.min() == 100 and np.allclose(gv[:, 2], 10.0 - PS.EYE_STAND_M)
    n = np.cross(gv[gf[:, 1] - 100] - gv[gf[:, 0] - 100], gv[gf[:, 2] - 100] - gv[gf[:, 0] - 100])
    assert (n[:, 2] > 0).all() and np.hypot(gv[:, 0], gv[:, 1]).max() >= PS.FLAT_R_M
    assert guv.min() >= 0 and guv[:, 1].max() <= 0.5                      # (the lower half of the shell's picture)
    rgb = np.zeros(d.shape + (3,), np.uint8)
    rgb[..., 1] = 120
    t = 1.6 / np.maximum(-dz, 1e-3)
    rgb[(dz < 0) & (dirs[..., 1] * t < -3) & (dirs[..., 1] * t > -5)] = (200, 0, 0)     # a band of the ground ahead
    tile = np.full((2 * PS.TILE_PX, 2 * PS.TILE_PX, 3), 50, np.uint8)
    gm = PS.ground_map(rgb, np.minimum(np.where(dz < -1e-3, t, 150.0), 135.0), dz > 0.02, tile)
    at = lambda x, y: gm[int((y + PS.MAP_R_M) / (2 * PS.MAP_R_M) * PS.MAP_PX), int((x + PS.MAP_R_M) / (2 * PS.MAP_R_M) * PS.MAP_PX)]  # noqa: E731
    assert tuple(at(0, -4)) == (200, 0, 0) and tuple(at(0, 4)) == (0, 120, 0) and tuple(at(11.5, 11.5)) == (50, 50, 50)


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


def test_a_place_picture_that_shows_one_part_is_read_as_part_and_the_land_takes_the_place_words():
    """The concept reader says whether the picture shows the whole place or one part of it; a place in the open
    drawn anew as a whole has its land said in the place's own words, the one part after them."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets/scenes"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets/sets"))
    import concept as CO
    import place_build as PB
    def ask(view):
        return lambda items: {"view": {"view": view}, "space": {"space": "outdoors"}}
    assert CO.read(Path("x.png"), {"pieces": []}, ask("Part."))["view"] == "part"
    assert CO.read(Path("x.png"), {"pieces": []}, ask("whole"))["view"] == "whole"
    spec = {"kind": "outdoors", "period": "not stated", "room": "A yard with a workshop and a shed.",
            "_director_room": "Outside: A gate in a wall.", "pieces": []}
    t = [p for p in PB.outdoors({"pieces": []}, spec, {"land": spec["room"], "surfaces": {}})["pieces"]
         if p["kind"] == "terrain"][0]
    assert t["land_words"].startswith("A yard with a workshop and a shed. With a gate in a wall")
    t0 = [p for p in PB.outdoors({"pieces": []}, spec, {"surfaces": {}})["pieces"] if p["kind"] == "terrain"][0]
    assert t0["land_words"] == "A gate in a wall"


def test_people_sent_to_a_tall_thing_stand_before_it():
    """Two people sent to one tall thing of the set (a building's front) both stand on its front side, toward the
    room's middle; a low thing is still stood round."""
    import from_plan as F
    t = {"name": "x", "kind": "fixture", "center": [-1.7, 6.0, 2.0], "half": [1.9, 1.1, 2.0]}
    fr = F.front_of(t)
    a = F.free_mark(t["center"], [], [t], [28, 28, 6], (0.6, 1.6), front=fr)
    b = F.free_mark(t["center"], [a], [t], [28, 28, 6], (0.6, 1.6), front=fr)
    assert a[1] < 6.0 - 1.1 and b[1] < 6.0 - 1.1 and a != b
    assert all(m[1] < 6.0 for m in F.free_marks(t["center"], [], [t], [28, 28, 6], (0.6, 3.0), front=fr))
    assert F.front_of({"kind": "fixture", "center": [2, 2, 0.5], "half": [1, 1, 0.5]}) is None
    assert F.front_of({"kind": "floor", "center": [2, 2], "half": [0, 0]}) is None


def test_the_small_things_of_an_open_place_are_built_and_its_metres_are_its_things_real_ones(tmp_path):
    """In the open the concept reader keeps every thing that stands by itself (a part built into a larger thing is
    not one; a kind of wall opening is a thing there), says each whole from its own part of the picture, and asks
    how large it really is; the set builder builds a small thing standing before a large one beside it, and corrects
    the measured picture by the things' real heights."""
    import concept as CO
    from PIL import Image
    pic = tmp_path / "c.png"
    Image.new("RGB", (200, 100), (90, 80, 70)).save(pic)
    spec = {"place": "yard", "pieces": [{"id": "p1", "kind": "landmark", "look": "stone"}]}
    res = {"space": "outdoors", "spec": {"p1": {"in_picture": True, "look": "stone, grey"}},
           "seen": [{"name": "big house", "look": "stone"}, {"name": "gate", "look": "wood"},
                    {"name": "tub", "look": "wood"}, {"name": "pit", "look": "stone"}]}

    def ask(items):
        if "words" in items[0]["ask"]:
            return {i["id"]: {"words": "A low house of stone with a roof of slate."} for i in items}
        return {"b:p1": {"box": "[0, 100, 600, 900]", "real": "12, 7"},
                "s:0": {"box": "[0, 100, 600, 900]", "kind": "landmark", "own": "separate", "real": "12 7"},
                "s:1": {"box": "[200, 300, 350, 900]", "kind": "door", "own": "Part."},
                "s:2": {"box": "[620, 700, 700, 900]", "kind": "none", "own": "separate", "real": "0.8 and 1.0"},
                "s:3": {"box": "[400, 700, 520, 900]", "kind": "fireplace", "own": "separate"}}
    CO.pieces_from(pic, spec, res, ask, tmp_path / "crops", "t", ["landmark", "door", "fireplace", "chair"])
    assert [s.get("kind") for s in res["seen"]] == ["landmark", None, "landmark", "landmark"]
    assert res["spec"]["p1"]["real_m"] == [12.0, 7.0] and res["seen"][2]["real_m"] == [0.8, 1.0]
    assert res["spec"]["p1"]["words"].startswith("A low house") and res["seen"][3]["words"]
    assert CO.read(pic, {"pieces": []}, lambda items: {"view": {"view": "whole", "cut": "Yes."},
                                                       "space": {"space": "outdoors"}})["cut"] is True
    # the builder: the house once (its twin in "seen" is the same thing), the tub and the pit beside it
    res["depth"] = {"eye": [0.0, -5.0], "things": {"spec:p1": {"xy": [0.0, 6.0], "size_m": [6.0, 3.6, 3.5]},
                                                   "seen:2": {"xy": [4.0, 4.0], "size_m": [0.5, 0.3, 0.5]},
                                                   "seen:3": {"xy": [-1.0, 3.5], "size_m": [0.8, 0.5, 0.4]}}}
    full = dict(spec, kind="outdoors", period="not stated", pieces=[dict(spec["pieces"][0], phrases=["at the house"])])
    sp = PB.with_concept(full, dict(res, concept=str(pic), surfaces={}))
    by = {p["id"]: p for p in sp["pieces"]}
    assert sorted(by) == ["c3", "c4", "p1", "t1"]
    assert by["p1"]["look"].startswith("A low house")
    # 7 m real over 3.5 m measured: every measure doubled
    assert by["p1"]["_seen_size_m"] == [12.0, 7.2, 7.0] and by["c3"]["_seen_size_m"][0] == 1.0
    assert abs((by["c3"]["at_xy"][0] - by["p1"]["at_xy"][0]) - 8.0) < 1e-6


def test_a_place_in_the_open_is_said_whole_for_its_picture(tmp_path, monkeypatch):
    """The place spec of a place in the open carries the whole place in words (its buildings, what stands round
    them, the land), asked once of the director with the story and the shots there."""
    import place_spec as SPc
    asked = []
    monkeypatch.setattr(SPc, "ask", lambda url, text, cache: (asked.append(text), "1. A yard.  2. With a house.")[1])
    got = SPc.whole("the yard", {"period": "medieval", "room": "A gate."}, ["someone waits"], "The story.", "u", tmp_path)
    assert got == "A yard. With a house." and "medieval" in asked[0] and "- someone waits" in asked[0] and "A gate." in asked[0]
