"""11-28: a place built as a spot of its story's solid world: plain solids at real size, good from every side, in
place of one picture pushed into a relief. Made-up places only."""
import json
import sys
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", "places", "scenes", ""):
    sys.path.insert(0, str(A / sub))
import layout as LY  # noqa: E402
import place_build as PB  # noqa: E402
import world_parts as WP  # noqa: E402
import world_plan as WQ  # noqa: E402

WORLD = {"schema": "world/1", "night": True, "ground": [{"name": "lake", "material": "water", "z_m": -1.0}],
         "parts": [{"kind": "deck", "name": "jetty", "at": [0, 0, 0], "size_m": [12, 6], "rail_m": 1.0, "rail_open": ["west"]},
                   {"kind": "building", "name": "boathouse", "at": [20, 0, -1], "size_m": [8, 6, 3], "roof": "gable",
                    "windows": {"rows": 1, "cols": 2, "lit": 1.0, "sides": ["west"]}},
                   {"kind": "row", "name": "street", "from": [-40, 30, 0], "to": [40, 30, 0], "front": [0, 0]},
                   {"kind": "ship", "name": "barge", "at": [0, -20, -0.5], "length_m": 18},
                   {"kind": "mast", "name": "flagstaff", "at": [4, 1.5, 0], "height_m": 9, "diameter_m": 0.3},
                   {"kind": "lamp", "name": "lamp", "at": [-4, 2, 0], "height_m": 3},
                   {"kind": "hill", "name": "far_hill", "at": [0, 400, 0], "size_m": [600, 200, 50]}],
         "places": {"Jetty (outside)": {"origin": [0, 0, 0], "stage_m": [12, 6, 8], "spots": {"p1": {"at": [3, -1]}}},
                    "Boathouse door (outside)": {"origin": [14, 0, -1], "yaw_deg": 90, "stage_m": [6, 6, 8]}}}


def test_every_part_becomes_solids_the_set_builder_takes():
    w = WP.build(dict(WORLD, stage_m=[12, 6, 8]))
    kinds = {e["kind"] for e in w["structure"]}
    assert kinds <= {"box", "beam", "pole", "ball", "ground"} and {"box", "beam", "pole", "ground"} <= kinds
    spec = {"size_m": [12, 6, 8], "structure": w["structure"], "lights": w["lights"], "fixtures": [], "props": []}
    ex = LY.expand(spec, {})
    assert 100 < len(ex["prims"]) < 3000                       # (a whole world in few plain shapes)
    assert all(e["material"] in w["materials"] for e in w["structure"])
    far = next(e for e in w["structure"] if e.get("name") == "far_hill")
    assert far.get("tile") is False                            # (far land in one piece)
    mast = next(s for s in w["solids"] if abs(s[0] - 4) < 1e-6 and abs(s[1] - 1.5) < 1e-6)
    assert mast[5] == 9                                        # (a mast stops a camera as high as it stands)
    assert any(l["type"] == "point" for l in w["lights"])      # (a lamp lights the place)


def test_a_building_is_as_tall_as_its_ridge_and_its_roof_slopes():
    w = WP.build({"schema": "world/1", "parts": [{"kind": "building", "name": "hut", "at": [0, 10, 0],
                                                  "size_m": [6, 4, 3], "roof": "gable", "pitch_deg": 45, "ridge": "x"}]})
    assert w["solids"][0][5] == 5.0                            # (eaves 3 m, half the 4 m span up again at 45 degrees)
    roof = [e for e in w["structure"] if e["kind"] == "beam"]
    assert len(roof) == 2 and {round(e["up"][1], 3) for e in roof} == {-0.707, 0.707}


def test_a_place_sees_the_one_world_from_its_own_spot():
    a = WP.for_place(WORLD, "Jetty (outside)")
    b = WP.for_place(WORLD, "Boathouse door (outside)")
    assert WP.for_place(WORLD, "Elsewhere") is None
    at = {p["name"]: p.get("at") for p in a["parts"]}
    bt = {p["name"]: p for p in b["parts"]}
    assert at["boathouse"] == [20, 0, -1] and a["stage_m"] == [12, 6, 8]
    # (from the door, turned a quarter: the boathouse 6 m off along -y, on the stage's own level; the jetty behind)
    assert bt["boathouse"]["at"] == [0.0, -6.0, 0.0] and bt["boathouse"]["yaw_deg"] == -90.0
    assert bt["jetty"]["at"] == [0.0, 14.0, 1.0] and b["ground"][0]["z_m"] == 0.0


def test_the_solids_reach_staging_as_ground_nobody_stands_in():
    w = WP.build(WP.for_place(WORLD, "Jetty (outside)"))
    hl = PB.solids_high(w["solids"])
    import ground_map as GM
    def at(name, x, y):
        i, j = GM.cell_of(x, y, hl["half_m"], hl["cell_m"])
        return any(r[0] == i and r[1] <= j < r[2] for r in hl[name])
    assert at("tall_runs", 4.1, 1.6) and at("runs", 4.1, 1.6)          # the flagstaff: no camera, nobody
    assert at("runs", 0.1, 3.0) and not at("tall_runs", 0.1, 3.0)      # the rail: nobody stands in it; 1 m high
    assert not at("runs", 0.2, 0.2)                                    # the open middle is free
    tops = {(c[0], c[1]): c[2:] for c in hl["tops"]["cells"]}
    i, j = GM.cell_of(0.1, 3.0, hl["half_m"], hl["tops"]["cell_m"])
    assert tops[(i, j)] == [0.0, 1.0]                                  # (a line of sight passes over the rail)


def test_the_place_builder_takes_the_world_in_place_of_a_picture(monkeypatch):
    spec = {"place": "Jetty (outside)", "kind": "outdoors", "period": "not stated", "size": "medium", "openings": [],
            "pieces": [{"id": "p1", "kind": "crate", "look": "a plain crate", "phrases": ["by the crate"],
                        "against": "free"}], "relations": [], "named_places": {"by the crate": "p1", "the middle": "floor"},
            "room": "Outside: a jetty on a lake"}
    sp = PB.with_world(spec, WP.for_place(WORLD, spec["place"]))
    sp["_time"] = "night"
    assert sp["_open_top"] and sp["_room_m"][:2] == [12.0, 6.0] and sp["pieces"][0]["at_xy"] == [3.0, -1.0]
    out = {"structure": [{"kind": "floor", "material": "f"}], "materials": {"f": {}}, "exposure_ev100": 5.0,
           "lights": [{"type": "point", "position": [0, 0, 2], "lumens": 100}], "words": "The room: a crate."}
    rep = PB.solid_world_stage(out, sp)
    assert rep["world"] == "solid" and out["open_top"] and out["all_round"] and "picture_dir" not in out
    assert not any(e["kind"] == "floor" for e in out["structure"])     # (the world's own deck is the floor)
    assert all(e["material"] in out["materials"] for e in out["structure"])
    assert [l["type"] for l in out["lights"][:2]] == ["sun", "sky"] and out["lights"][1]["atmosphere"] is True
    assert out["lights"][2]["lumens"] == 100                           # (the pieces' own lamps burn on)
    assert out["words"].startswith("In the open air at night under a dark sky, a jetty on a lake")
    assert out["world"]["solid"] and out["high_land"]["tall_runs"]


def test_a_plan_with_a_wrong_scale_or_a_blocked_stage_is_sent_back():
    spec = {"place": "Jetty (outside)", "pieces": [{"id": "p1", "kind": "crate"}]}
    good = dict(WP.for_place(WORLD, "Jetty (outside)"))
    assert WQ.faults(good, spec) == []
    bad = json.loads(json.dumps(good))
    bad["parts"].append({"kind": "building", "name": "giant", "at": [0, 60, 0], "size_m": [8, 8, 300]})
    bad["parts"].append({"kind": "block", "name": "pillar", "at": [0.5, 0, 0], "size_m": [1, 1, 3], "material": "stone"})
    bad["spots"] = {"p1": {"at": [40, 0]}}
    fl = " | ".join(WQ.faults(bad, spec))
    assert "not a real size for a building" in fl
    bad["parts"] = [p for p in bad["parts"] if p["name"] != "giant"]
    fl = " | ".join(WQ.faults(bad, spec))
    assert "within 1.5 m of the origin" in fl and "off the stage" in fl
    no_floor = json.loads(json.dumps(good))
    no_floor["parts"] = [p for p in no_floor["parts"] if p["name"] != "jetty"]
    assert any("no floor under the whole stage" in x for x in WQ.faults(no_floor, spec))
