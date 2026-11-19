"""Staging fixes from an open place with a creature and two rocks (11-18): the measured numbers below are that run's.

- A talk camera inside a rock rendered black; its re-framing was held to 1.5x the black camera's distance, no
  standpoint before the speaker (who faced the rock 1.5 m off) was left, and the search ended after one try. Now the
  farther-back rule is not set when the camera stood inside a thing (that explains the black still), a rule that
  leaves no standpoint is dropped rather than the search, and a still that showed nothing gets the clear wide as one
  more try. The clear wide itself starts at the distance its lens needs (it started past the flat ground).
- A creature's neck, between its body's box and its head, is solid for the camera search too.
- A creature's head is "bigger than the subject's" only when it stands nearer the camera, each head at its own size.
- An action shot's subject is not searched as a speaker (they may show their back).
- A speaker's face in a body's shadow fails the still, and the key light comes round the lens's other side.
- from_plan: a creature with no free place keeps clear of the people (relaxing its berth from the set's things
  first); a person turned to a tall thing a step away stands open to the floor.
"""
import json
import math
import struct
import sys
import zlib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "scenes"))
sys.path.insert(0, str(HERE.parents[0] / "assets"))
import staging_synth as S  # noqa: E402
import framing  # noqa: E402
import speakers as SP  # noqa: E402
import from_plan as FP  # noqa: E402

RES = (1152, 640)
# the creature as the engine measured it (head held 4 m before its middle, toward -x)
BEAST = {"eye_xy": [-1.7769, -3.7379], "eye_z": 2.9224, "head_top_z": 3.2644, "chin_z": 2.4094,
         "head_fwd_xy": [-0.9874, -0.1582], "feet_z": 0.0, "waist_z": 1.4612, "chest_xy": [-0.5344, -3.5314],
         "pelvis_xy": [2.506, -3.337], "facing": [-0.9843, -0.1765], "height_m": 4.25, "creature": True, "joints": {}}
WOMAN = S.person([-3.6771, -4.4215], facing=(0.9978, -0.067))["measured"]
MAN = S.person([4.4208, -5.0215], facing=(-0.9769, 0.2136))["measured"]
CAVE = {"name": "landmark", "kind": "fixture", "center": [0.2, 3.05, 2.0586], "half": [2.5, 2.2541, 2.0586]}
ROCK = {"name": "landmark_2", "kind": "fixture", "center": [-0.7, -4.45, 1.35605], "half": [1.5, 1.04825, 1.35605]}


@pytest.fixture
def mountain():
    """The open stage: flat ground 8.5 m round, the cave and the rock, the creature's body box, head and neck."""
    saved = (list(SP.OPEN_GROUND), list(SP.SOLIDS), list(SP.HEADS), list(SP.NECKS))
    SP.OPEN_GROUND[:] = [-0.052, 0.232, 8.5]
    SP.set_solids([CAVE, ROCK])
    SP.SOLIDS.append([2.506, -3.337, 2.08, 2.55, 3.4, -0.9843, -0.1765])
    SP.set_heads({"a": {"measured": BEAST}})
    yield [17.464, 17.464, 12.0]
    SP.OPEN_GROUND[:], SP.SOLIDS[:], SP.HEADS[:], SP.NECKS[:] = saved


def eye(m):
    return [m["eye_xy"][0], m["eye_xy"][1], m["eye_z"]]


def png(path: Path, w: int, h: int, px) -> Path:
    """A grey PNG whose pixel (x, y) is px(x, y)."""
    raw = b"".join(b"\x00" + bytes(v for x in range(w) for v in (px(x, y),) * 3) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)  # noqa: E731
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return path


# ------------------------------------------------------------------ the creature's body for the camera search
def test_a_creatures_neck_is_solid_and_its_own_eye_is_still_seen(mountain):
    neck_mid = [(BEAST["chest_xy"][0] + BEAST["eye_xy"][0]) / 2, (BEAST["chest_xy"][1] + BEAST["eye_xy"][1]) / 2,
                (BEAST["waist_z"] + BEAST["eye_z"]) / 2]
    assert SP.NECKS and SP.inside_solid(neck_mid)
    assert not any(SP._in_box(neck_mid, b, SP.SOLID_CLEAR_M) for b in SP.SOLIDS[2:])     # (the box alone missed it)
    # a camera south of the neck looking north through it at someone beyond: blocked; at the creature's own eye: seen
    cam = [neck_mid[0], neck_mid[1] - 3.0, neck_mid[2]]
    assert SP.solid_between(cam, [neck_mid[0], neck_mid[1] + 3.0, neck_mid[2]])
    front = [BEAST["eye_xy"][0] - 4.0, BEAST["eye_xy"][1] - 0.6, BEAST["eye_z"]]
    assert not SP.solid_between(front, eye(BEAST))


def test_a_creatures_head_beyond_the_subject_is_not_the_bigger_head():
    subjects = {"b": {"measured": WOMAN}, "a": {"measured": BEAST}}
    behind = {"position": [-5.4, -5.6, 1.6], "target": [-3.0, -4.2, 1.9], "focal_mm": 28}     # she nearer, it beyond
    why = SP.framing_why(behind, "medium", None, None, False, ["a", "b"], subjects, RES,
                         opts={"line_subject": "b"})[0]
    assert not any("is bigger than" in w for w in why)
    near = {"position": [-1.8, -6.0, 2.6], "target": [-2.7, -4.1, 2.2], "focal_mm": 20}       # it nearer than she is
    assert math.dist(near["position"], eye(BEAST)) < math.dist(near["position"], eye(WOMAN))
    why = SP.framing_why(near, "medium", None, None, False, ["a", "b"], subjects, RES, opts={"line_subject": "b"})[0]
    assert any(w.startswith("a's head") and "is bigger than" in w for w in why)
    # two people: the rule as it was (the head's own frame height)
    p2 = {"b": {"measured": WOMAN}, "c": {"measured": S.person([-2.4, -4.5])["measured"]}}
    cam = {"position": [-1.0, -4.6, 1.6], "target": [-3.6, -4.4, 1.5], "focal_mm": 35}
    assert any("c's head" in w for w in SP.framing_why(cam, "medium", None, None, False, ["b", "c"], p2, RES,
                                                         opts={"line_subject": "b"})[0])


# ------------------------------------------------------------------ the last resort wide
def test_the_clear_wide_finds_ground_for_two_people_far_apart(mountain):
    fb = SP.fallback_camera([WOMAN, MAN], mountain)
    assert fb and SP.in_room(fb["position"], mountain)
    assert not SP.solid_between(fb["position"], eye(WOMAN))
    spread = math.dist(WOMAN["eye_xy"], MAN["eye_xy"]) / 2
    mid = [(WOMAN["eye_xy"][i] + MAN["eye_xy"][i]) / 2 for i in range(2)]
    assert math.dist(fb["position"][:2], mid) < 2 * spread + 2.0      # (where the old ring began: past the ground)
    tried = [fb["position"]]
    fb2 = SP.fallback_camera([WOMAN, MAN], mountain, tried)
    assert fb2 and math.dist(fb2["position"], fb["position"]) >= 0.15     # a tried standpoint is not offered again


def test_the_clear_wide_on_a_talk_shot_holds_its_speaker_and_listener():
    subjects = {"a": {"measured": BEAST}, "b": {"measured": WOMAN}, "d": {"measured": MAN}}
    assert SP.fallback_people({"speaker": "b", "listener": "d"}, subjects) == ["b", "d"]
    assert SP.fallback_people({"line_subject": "b", "listed": ["a", "b"]}, subjects) == ["b", "a"]
    assert SP.fallback_people({}, subjects) == ["a", "b", "d"]


def test_what_counts_as_a_still_that_showed_nothing():
    assert SP.showed_nothing(["the camera is outside the room"])
    assert SP.showed_nothing(["the still is nearly black (mean 0.0 of 255)"])
    assert SP.showed_nothing(["a thing in the foreground covers 100% of the frame"])
    assert SP.showed_nothing([f"a fills 100% of the frame: {SP.TOO_NEAR}"])
    assert not SP.showed_nothing(["a thing in the foreground covers 21% of the frame", "face 40% visible"])


# ------------------------------------------------------------------ the loop: a black try leads to another camera
class DarkInside(S.FakeEngine):
    """The stand-in engine, with stills: black from a camera inside a thing of the set, else grey."""

    def __call__(self, spec_path, cams=None):
        super().__call__(spec_path, cams)
        spec = json.loads(Path(spec_path).read_text())
        res = json.loads((self.out_dir / spec["id"] / "result.json").read_text())
        for sh in res["shots"]:
            v = 0 if SP.inside_solid(sh["camera"]["position"]) else 110
            png(Path(sh["path"]), 64, 36, lambda x, y, v=v: v)


def test_a_black_try_from_inside_a_thing_leads_to_another_camera(tmp_path, monkeypatch):
    # she faces a 2.5 m block 1.5 m off (her listener beyond it): the scene tool's medium stands 2.2 m before her
    # face, inside the block
    lib = S.make_library(tmp_path, [{"name": "rock", "position": [0.0, 2.5], "size": [3.0, 2.0, 2.5]}],
                         size=(10.0, 10.0, 4.0))
    monkeypatch.setattr(SP.C, "LIBRARY", lib)
    out = tmp_path / "out"
    monkeypatch.setenv("SCENES_OUT", str(out))
    monkeypatch.delenv("SCENES_CHAR_TOOLS", raising=False)
    eng = DarkInside(out)
    monkeypatch.setattr(SP, "run_scene", eng)
    stand = {"pose": "stand_relaxed", "params": {}}
    spec = {"id": "sc", "set": "room_x", "line": ["b", "d"],
            "characters": {"b": {"id": "fig", "position": [0.0, 0.0], "facing": [0.0, 1.0], "body_pose": stand},
                           "d": {"id": "fig", "position": [0.0, 4.6], "facing": [0.0, -1.0], "body_pose": stand}},
            "cameras": [{"name": "s1", "kind": "medium", "speaker": "b", "subject": "b", "focal_mm": 50,
                         "listener": "d", "keep_listener": True}]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["speakers.py", str(p), "--rounds", "6", "--no-split"])
    assert SP.main() == 0
    rep = json.loads((out / "sc" / "speakers.json").read_text())
    h = rep["shots"]["s1"]["history"]
    assert any("nearly black" in w for w in h[0]["why"]) and "the camera is outside the room" in h[0]["why"]
    assert len(h) >= 2                                   # (it ended after the black try)
    assert all(SP.in_room(x["camera"]["position"], [10.0, 10.0, 4.0]) for x in h[1:])
    assert not any("nearly black" in w for w in h[-1]["why"])
    assert not any(SP.TOO_NEAR in w for x in h[1:] for w in x["why"])          # no farther-back rule from it


# ------------------------------------------------------------------ a speaker's face in shadow
def test_a_speakers_face_in_shadow_fails_the_still(tmp_path):
    cam = framing.shot(MAN, "medium", 50, RES, 0.0, 0.0, 0.5)
    cam = {"position": cam["position"], "target": cam["target"], "focal_mm": cam["focal_mm"]}
    box = SP.face_box(cam, MAN, RES)
    assert box and 0.3 < box[0] < box[2] < 0.7
    # (the face's own pixels a little past the box over its points: the points lie inside the face)
    inside = lambda x, y: box[0] * 128 - 2 <= x < box[2] * 128 + 2 and box[1] * 72 - 2 <= y < box[3] * 72 + 2  # noqa: E731,E501
    dark = png(tmp_path / "dark.png", 128, 72, lambda x, y: 3 if inside(x, y) else 120)       # a silhouette on sky
    lit = png(tmp_path / "lit.png", 128, 72, lambda x, y: 45 if inside(x, y) else 120)
    assert SP.face_light_why({"path": str(dark), "camera": cam}, "d", MAN, RES)
    assert SP.face_light_why({"path": str(lit), "camera": cam}, "d", MAN, RES) == []
    assert SP._png_mean(dark, box) == pytest.approx(3.0, abs=1.0)                 # (the reader without PIL agrees)


def test_the_key_comes_round_the_lenss_other_side_when_flipped():
    cam, e, sun = [2.0, -6.0, 1.45], [4.37, -5.03, 1.69], [-0.5808, 0.3473, 0.7363]
    k0 = framing.key_toward(cam, e, sun)
    k1 = framing.key_toward(cam, e, sun, flip=True)
    ca = math.atan2(cam[1] - e[1], cam[0] - e[0])
    off = lambda k: (math.atan2(k[1], k[0]) - ca + math.pi) % (2 * math.pi) - math.pi   # noqa: E731
    assert off(k0) == pytest.approx(-off(k1), abs=1e-3) and abs(off(k0)) == pytest.approx(
        math.radians(framing.KEY_SIDE_DEG), abs=1e-3)
    assert k0[2] == k1[2]


def test_the_engine_job_turns_the_key_by_the_cameras_flag():
    src = (HERE.parents[0] / "assets" / "scenes" / "ue" / "scene_job.py").read_text()
    assert f'flip=bool(c.get("{SP.KEY_FLIP}"))' in src


# ------------------------------------------------------------------ from_plan: where a creature and a person stand
def test_a_creature_with_no_free_place_keeps_clear_of_the_people():
    things = [dict(CAVE), dict(ROCK)]
    others = [[4.449, -4.997], [3.904, -6.126]]
    tgt = [sum(o[0] for o in others) / 2, sum(o[1] for o in others) / 2]
    # its full berth from the set's things (its body's half breadth) leaves no point on this stage
    assert not FP.free_marks(tgt, others, things, [13.0, 13.0], (2.0, 5.0), own_r=5.2, fix_r=5.1, walls=False)
    p = FP.free_mark(tgt, others, things, [13.0, 13.0], (2.0, 5.0), 5.2, 5.1, False)
    assert all(math.dist(p, o) >= FP.PERSON_GAP_M + 5.2 - 1e-6 for o in others)     # (it stood 2.6 m from one)
    assert FP.clear_of_fixtures(p, things, 0.0)
    # nowhere clear of the people at all: the point with the most room from them, not one between them
    q = FP.free_mark([0.0, 0.0], [[0.0, 1.0], [0.0, -1.0]], [], [6.0, 6.0], (0.0, 1.0), 5.2, None, False)
    assert min(math.dist(q, o) for o in ([0.0, 1.0], [0.0, -1.0])) > 2.0


def test_a_person_turned_to_a_tall_thing_a_step_away_stands_open_to_the_floor():
    rock = dict(ROCK)
    pos = [-2.485, -5.1]                                      # 0.29 m off the rock's face, turned to its middle
    d = [rock["center"][0] - pos[0], rock["center"][1] - pos[1]]
    f = [d[0] / math.hypot(*d), d[1] / math.hypot(*d)]
    others = [[0.12, 1.83], [-2.477, -3.927]]
    g = FP.open_from_wall(pos, f, rock, others)
    turn = math.degrees(math.acos(f[0] * g[0] + f[1] * g[1]))
    assert turn == pytest.approx(FP.WALL_OPEN_DEG, abs=0.5)
    mid = [sum(o[0] for o in others) / 2 - pos[0], sum(o[1] for o in others) / 2 - pos[1]]
    assert g[0] * mid[0] + g[1] * mid[1] > f[0] * mid[0] + f[1] * mid[1]      # toward the others
    desk = {"name": "desk", "kind": "fixture", "center": [-0.7, -4.45, 0.4], "half": [1.5, 1.05, 0.4]}
    assert FP.open_from_wall(pos, f, desk, others) == f                     # a low thing: worked at, faced
    assert FP.open_from_wall([-6.0, -8.0], f, rock, others) == f              # a far one: as it is


class ShadeFromSunSide(S.FakeEngine):
    """The stand-in engine with stills: the speaker's face dark (a body's shadow) unless the camera's key light comes
    round the lens's other side."""

    def __call__(self, spec_path, cams=None):
        super().__call__(spec_path, cams)
        spec = json.loads(Path(spec_path).read_text())
        res = json.loads((self.out_dir / spec["id"] / "result.json").read_text())
        flips = {c["name"]: bool(c.get(SP.KEY_FLIP)) for c in spec["cameras"]}
        self.flips = getattr(self, "flips", []) + [flips]
        for sh in res["shots"]:
            box = SP.face_box(sh["camera"], res["subjects"]["d"]["measured"], RES)
            v = 60 if flips[sh["name"]] else 3
            inside = lambda x, y, b=box: b and b[0] * 128 - 2 <= x < b[2] * 128 + 2 and b[1] * 72 - 2 <= y < b[3] * 72 + 2  # noqa: E731,E501
            png(Path(sh["path"]), 128, 72, lambda x, y: v if inside(x, y) else 120)


def test_a_speaker_in_shadow_is_shot_again_with_the_key_from_the_other_side(tmp_path, monkeypatch):
    lib = S.make_library(tmp_path, [{"name": "pebble", "position": [2.0, -1.0], "size": [0.1, 0.1, 0.1]}],
                         size=(13.0, 13.0, 6.0))
    sd = lib / "set" / "room_x"
    ss = json.loads((sd / "spec.json").read_text())
    ss["world"] = {"flat_r_m": 9.0}                           # a place in the open (its world's flat ground)
    ss["fixtures"][0]["world"] = True
    (sd / "spec.json").write_text(json.dumps(ss))
    monkeypatch.setattr(SP.C, "LIBRARY", lib)
    out = tmp_path / "out"
    monkeypatch.setenv("SCENES_OUT", str(out))
    monkeypatch.delenv("SCENES_CHAR_TOOLS", raising=False)
    eng = ShadeFromSunSide(out)
    monkeypatch.setattr(SP, "run_scene", eng)
    stand = {"pose": "stand_relaxed", "params": {}}
    spec = {"id": "sc", "set": "room_x", "line": ["d", "b"],
            "characters": {"d": {"id": "fig", "position": [0.0, 0.0], "facing": [0.0, -1.0], "body_pose": stand},
                           "b": {"id": "fig", "position": [0.3, -3.0], "facing": [0.0, 1.0], "body_pose": stand}},
            "cameras": [{"name": "s1", "kind": "medium", "speaker": "d", "subject": "d", "ots": "b",
                         "position": [-0.6, -4.0, 1.6], "target": [0.045, -0.45, 1.52], "focal_mm": 35}]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    monkeypatch.setattr(sys, "argv", ["speakers.py", str(p), "--rounds", "4", "--no-split"])
    assert SP.main() == 0
    rep = json.loads((out / "sc" / "speakers.json").read_text())
    h = rep["shots"]["s1"]["history"]
    assert any("d's face is in shadow" in w for w in h[0]["why"]) and not h[0][SP.KEY_FLIP]
    assert h[1]["camera"]["position"] == h[0]["camera"]["position"] and h[1][SP.KEY_FLIP]   # the same camera, lit
    assert not any("in shadow" in w for w in h[1]["why"])
