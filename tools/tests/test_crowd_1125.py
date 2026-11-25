"""11-25: a scene with a crowd gets guests behind its cast: counted from the words or the floor, placed in pairs
clear of the cast and of each other, the same places every time, none in front of the people the shot is of."""
import json, math, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import crowd as CR

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

hall = [[-6, 0], [6, 0], [6, 14], [-6, 14]]
check("the floor's area", abs(CR.area(hall) - 168) < 1e-6, CR.area(hall))
check("a number in the words", CR.count("about twenty townsfolk along the walls", 500) == 20)
check("digits in the words", CR.count("12 traders", 500) == 12)
check("no number: by the floor, capped", CR.count("traders at their stalls", 168) == CR.MAX_N)
check("a small room keeps a few", CR.count("a few traders", 10) == CR.MIN_N)

keep = [[-0.5, 4.0], [0.5, 4.0]]
a = CR.place(hall, keep, 16, 7)
b = CR.place(hall, keep, 16, 7)
check("sixteen placed", len(a) == 16, len(a))
check("the same seed, the same places", a == b)
check("another seed, other places", CR.place(hall, keep, 16, 8) != a)
check("all on the floor", all(CR._in_poly(hall, *g["at"]) for g in a))
check("clear of the cast", all(math.hypot(g["at"][0] - k[0], g["at"][1] - k[1]) > CR.CLEAR - CR.PAIR / 2 - 1e-6
                                 for g in a for k in keep))
p0, p1 = a[0], a[1]
d = math.hypot(p0["at"][0] - p1["at"][0], p0["at"][1] - p1["at"][1])
check("a pair stands together", abs(d - CR.PAIR) < 0.01, d)
to = math.degrees(math.atan2(p1["at"][1] - p0["at"][1], p1["at"][0] - p0["at"][0])) % 360
check("and faces each other", abs((p0["_facing"] - to + 180) % 360 - 180) < 1.0, (p0["_facing"], to))
check("they stand", all(g["pose"] == "standing" for g in a))
far = CR.place([[-40, 0], [40, 0], [40, 14], [-40, 14]], keep, 16, 7)
check("around the cast while there is room", all(math.hypot(g["at"][0], g["at"][1] - 4.0) < CR.REACH + CR.PAIR for g in far))
check("a closet takes fewer than asked", len(CR.place([[0, 0], [3, 0], [3, 3], [0, 3]], [], 16, 1)) < 16)

crowd = [{"at": [0, 2.0]}, {"at": [0, 8.0]}, {"at": [3, 4.5]}]
v = CR.visible(crowd, [0, 0, 1.5], [[0, 4.0]])
check("nobody stands in front of the cast", [g["at"] for g in v] == [[0, 8.0], [3, 4.5]], v)
check("an empty shot keeps the far ones", len(CR.visible(crowd, [0, 1.0, 1.5], [])) == 2)

L = CR.looks("the Hall", "a made-up land, around 1700")
check("six guests, men and women", len(L) == 6 and {g["kind"] for g in L} == {"man", "woman"})
check("each dressed for the place and time", all("the Hall in a made-up land, around 1700" in g["desc"]
                                                 and "wearing" in g["desc"] for g in L))
check("keys by the place", L[0]["key"] == "the_hall__guest1", L[0]["key"])
check("no period: none named", " in " not in CR.looks("the Hall")[0]["desc"].split("wears to")[1])

frames = {"m1": {"crowd": ""}, "m2": {"crowd": "townsfolk dancing in rings"}, "m3": {}}
check("the scene's crowd from any of its moments", CR.crowd_text(frames, ["m1", "m2", "m3"]) == "townsfolk dancing in rings")
check("no crowd named: none", CR.crowd_text({"m1": {}}, ["m1"]) == "")

with tempfile.TemporaryDirectory() as td:
    vd = Path(td)
    shots = {"m1": {"figures": [{"who": "Orla", "at": [-0.5, 4.0]}]}, "m2": {"figures": [{"who": "Tam", "at": [0.5, 4.0]}]}}
    room = {"floor_polygon_xy": hall}
    ln = CR.stage("the Hall", ["m1", "m2"], frames, shots, room, {}, vd)
    check("no models: said, nothing staged", "no guest models" in ln[0] and "crowd" not in shots["m1"], ln)
    (vd / "stage" / "extras" / "the_hall__guest1").mkdir(parents=True)
    people = []
    for i in (1, 2):
        m = vd / "stage" / "extras" / f"g{i}.blend"
        m.write_text("x")
        people.append({"who": f"guest {i}", "model": str(m), "height": 1.7, "kind": "man"})
    people.append({"who": "guest 3", "model": str(vd / "missing.blend")})
    Path(CR.index_path(vd)).write_text(json.dumps({"the_hall": {"where": "the Hall", "people": people}}))
    check("only built guests count", len(CR.guests(vd, "The Hall")) == 2)
    ln = CR.stage("the Hall", ["m1", "m2"], frames, shots, room, {}, vd)
    c = shots["m1"]["crowd"]
    check("every shot of the scene has the same crowd", c == shots["m2"]["crowd"] and len(c) >= CR.MIN_N, ln)
    check("each guest is named once", len({g["who"] for g in c}) == len(c))
    check("a pair is two different models", c[0]["model"] != c[1]["model"])
    check("clear of both shots' people", all(math.hypot(g["at"][0] - x, g["at"][1] - 4.0) > 1.5 for g in c for x in (-0.5, 0.5)))
    check("no floor: said", "no measured floor" in CR.stage("the Hall", ["m2"], frames, {"m2": {"figures": []}}, None, {}, vd)[0])
    check("a scene with no crowd is left alone", CR.stage("the Hall", ["m1"], frames, {"m1": {"figures": []}}, room, {}, vd) == [])

sys.path.insert(0, str(HERE.parent.parent))
import flow3d as F
with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    for st, fr in (("01_one", [{"shot": "a1", "where": "the Hall", "crowd": "townsfolk at long tables"},
                               {"shot": "a2", "where": "the Hall", "crowd": "pipers by the door"},
                               {"shot": "a3", "where": "the Yard", "crowd": ""}]),
                   ("02_two", [{"shot": "b1", "where": "the Quay", "crowd": "porters"}])):
        (book / "work" / "video" / st / "anim").mkdir(parents=True)
        (book / "work" / "video" / st / "anim" / "keyframes.json").write_text(json.dumps({"frames": fr}))
    cp = F.crowd_places(book, ["01_one", "02_two"])
    check("each place with a crowd, its first words", cp == {"the Hall": "townsfolk at long tables", "the Quay": "porters"}, cp)
    check("a window's places only", F.crowd_places(book, ["01_one", "02_two"], ["the Quay"]) == {"the Quay": "porters"})
    check("a chapter with no keyframes yet", F.crowd_places(book, ["03_three"]) == {})
src = (HERE.parent / "flow3d.py").read_text()
check("assets builds the guests of those places", '"characters.py", "extras", book, "--place", wh_' in src)
ch = (HERE.parent / "characters.py").read_text()
check("the guests go through the cast's build steps", all(x in ch.split("def extras(")[1].split("def main(")[0]
      for x in ("step_draw(", "step_cut(", "step_shape(", "step_rig(", "step_qa(", "CR.index_path(vd)")))
check("the stills draw them", "_CR.models(shot, cam[\"pos\"])" in (HERE.parent / "blockout.py").read_text())
check("the clips draw them", '"models": models + guests_' in (HERE.parent / "blockout_video.py").read_text())
check("a guest's hands are not checked", 'and lay != "set"' in (HERE.parent / "blockout_scene.py").read_text())
check("the staging places them", "CR.stage(sc[\"where\"], sc[\"moments\"], frames, shots" in (HERE.parent / "scene_stage.py").read_text())
check("a repainted guest renders again", 'sh.get("figures", []) + (sh.get("crowd") or [])' in (HERE.parent / "remake_chapter.py").read_text())

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
