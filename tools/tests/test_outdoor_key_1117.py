"""A place in the open: its key light comes from the camera's side for each shot (scenes/framing.py key_toward,
scenes/scene.py key_light), and the redraw says the open place, not a room (redraw/redraw.py)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "scenes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "redraw"))

import framing  # noqa: E402
import redraw  # noqa: E402
import scene  # noqa: E402


def _az(v):
    return math.degrees(math.atan2(v[1], v[0]))


def _off(a, b):
    return abs((a - b + 180) % 360 - 180)


def test_key_comes_from_the_camera_side_toward_the_world_sun():
    eye = [0.0, 0.0, 1.6]
    sun_behind = [-0.9, 0.1, 0.4]                 # the world's sun beyond the people, seen from the camera
    for cam in ([3.0, 0.0, 1.6], [0.0, -3.0, 1.6], [-2.0, 2.0, 1.6]):
        k = framing.key_toward(cam, eye, sun_behind)
        cam_az = _az([cam[0] - eye[0], cam[1] - eye[1]])
        assert abs(_off(_az(k), cam_az) - framing.KEY_SIDE_DEG) < 0.01
        assert abs(sum(x * x for x in k) - 1.0) < 1e-4
    # the side follows the world's sun: the sun to the camera's left gives a key on its left, and the other way
    left = framing.key_toward([3, 0, 1.6], eye, [0.2, 0.9, 0.5])
    right = framing.key_toward([3, 0, 1.6], eye, [0.2, -0.9, 0.5])
    assert left[1] > 0 > right[1]


def test_key_height_is_held_between_its_bounds():
    lo, hi = framing.KEY_EL_DEG
    for z, want in ((0.99, hi), (0.02, lo), (math.sin(math.radians((lo + hi) / 2)), (lo + hi) / 2)):
        k = framing.key_toward([3, 0, 1.6], [0, 0, 1.6], [math.sqrt(1 - z * z), 0.0, z])
        assert abs(math.degrees(math.asin(k[2])) - want) < 0.05


def test_key_light_only_for_a_world(tmp_path):
    room = tmp_path / "room.json"
    room.write_text(json.dumps({"lights": [{"type": "sun", "toward_sun": [0, 1, 1], "lux": 50}]}))
    assert scene.key_light(room) is None
    assert scene.key_light(tmp_path / "missing.json") is None
    world = tmp_path / "world.json"
    world.write_text(json.dumps({"world": {"asset": "w"}, "exposure_ev100": 5.0,
                                 "lights": [{"type": "sun", "toward_sun": [0.6, -0.2, 0.7], "lux": 65.0},
                                            {"type": "sky", "intensity": 1.0}]}))
    kl = scene.key_light(world)
    assert kl["toward_sun"] == [0.6, -0.2, 0.7]
    assert kl["lux"] == round(scene.KEY_LUX_EV * 2 ** 5.0, 1)          # a night world's sun raised so faces read
    assert scene.key_light(world, ev=1.0)["lux"] == 65.0                 # never below the world's own sun


def test_open_air_shot_words_name_no_room():
    ppl = [("a", [0.3, 0.1, 0.7, 1.0], 20.0, "standing", 20.0, "left")]
    cast = {"a": {"noun": "man", "trigger": "chr_x", "wear": "a coat"}}
    words = "In the open air, period: a river bank under hills; standing there: a cart"
    for kind in ("wide", "medium", "close"):
        p = redraw.image_prompt(kind, ppl, "", words, "", cast=cast, outdoors=True)
        assert "room" not in p.lower() and "a river bank under hills" in p
        p2 = redraw.image_prompt(kind, ppl, "", words, "", things="In the frame: a cart.", cast=cast, outdoors=True)
        assert "a river bank under hills" in p2 and "standing there" not in p2
        assert "room" in redraw.image_prompt(kind, ppl, "", words, "", cast=cast).lower()     # a room stays a room


def test_no_window_daylight_out_of_doors():
    sh = {"daylight": "daylight", "things": []}
    assert "window" in redraw.set_words(sh)
    assert "window" not in redraw.set_words(sh, outdoors=True)


def test_open_air_surfaces():
    assert redraw.open_air_words("The ground: The floor is stone; overhead: the ceiling is dark; floorboards") == \
        "The ground: The ground is stone; overhead: the sky is dark; floorboards"
