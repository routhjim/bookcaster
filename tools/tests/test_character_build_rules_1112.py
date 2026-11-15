"""The one-command character build's rules (img3d/cast_char.py), the skin probe, the stance resolver, the check
sheet's small tests and the outer layer's verdict: CPU only, no line jobs, no engine, no Blender, no models.

cast_char.py's helpers are imported and called. The rules that live inline in its main() (asset names from the id
and the seat's name, the drawing's starting angle, the long-garment fact over the drawing's measure, the fit job's
neckline and hair facts, the relative neck girth change and its "no effect" stop, when the collar is lifted, when the
arms pass) are taken from the source as the single statement that holds each rule and run on plain values: a rule
that is reworded or removed fails its test by name. The start-of-run path check is run as the real command with
missing inputs (it stops before any step). The skin probe runs on synthetic stills."""

import ast
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parents[1] / "assets"
TMP = tempfile.mkdtemp(prefix="char_rules_test_")
os.environ["ASSETS_SCHED_DIR"] = os.path.join(TMP, "sched")
os.environ["ASSETS_LIBRARY"] = os.path.join(TMP, "lib")
os.environ["ASSETS_LOGS"] = os.path.join(TMP, "logs")
sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(ASSETS / "img3d"))
sys.path.insert(0, str(ASSETS / "characters"))

import cast_char as CC  # noqa: E402
import skin_probe as SP  # noqa: E402
import stances as ST  # noqa: E402

CAST_SRC = (ASSETS / "img3d" / "cast_char.py").read_text()
CAST_TREE = ast.parse(CAST_SRC)
LAYER = ASSETS / "img3d" / "bl_layer.py"


def node(kind, *contains, src=CAST_SRC, tree=CAST_TREE, part=None):
    """The smallest statement of KIND in the source whose text contains every string in CONTAINS (exactly one such
    smallest). part="test": an if statement's condition; part="value": an assignment's right-hand side."""
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, kind):
            seg = ast.get_source_segment(src, n) or ""
            if all(c in seg for c in contains):
                hits.append((len(seg), seg, n))
    assert hits, f"no {kind.__name__} statement contains {contains}"
    hits.sort(key=lambda h: h[0])
    assert len(hits) == 1 or hits[0][0] < hits[1][0], f"two equally small statements contain {contains}"
    _, seg, n = hits[0]
    if part == "test":
        return ast.get_source_segment(src, n.test)
    if part == "value":
        return ast.get_source_segment(src, n.value)
    return textwrap.dedent(" " * n.col_offset + seg)


def nested_function(name):
    for n in ast.walk(CAST_TREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return textwrap.dedent(" " * n.col_offset + ast.get_source_segment(CAST_SRC, n))
    raise AssertionError(f"cast_char.py: no function {name}")


# ---------------------------------------------------------------------------------------------------------------
class Names(unittest.TestCase):
    """Asset names made from the character's id and from a seat's name."""

    def tag(self, cid):
        ns = {"a": types.SimpleNamespace(id=cid)}
        for n in ast.walk(CAST_TREE):
            if isinstance(n, ast.Assign) and len(n.targets) == 1 and getattr(n.targets[0], "id", None) == "tag":
                seg = ast.get_source_segment(CAST_SRC, n)
                if "a.id" in seg or "tag[" in seg:
                    exec(seg, ns)
        self.assertIn("tag", ns, "the id's tag is no longer made by assignments to 'tag'")
        return ns["tag"]

    def test_tag_is_short_and_plain(self):
        t = self.tag("man_long_named_outfit_auto_01")
        self.assertTrue(t.isalnum() and t == t.upper())
        self.assertLessEqual(len(t), 12)

    def test_a_rebuild_beside_gets_its_own_assets(self):
        # the id's end is kept: _03 and _04 of the same long id must not share engine assets
        self.assertNotEqual(self.tag("man_long_named_outfit_combo_03"), self.tag("man_long_named_outfit_combo_04"))
        self.assertNotEqual(self.tag("woman_long_named_dress_combo_03"), self.tag("woman_long_named_dress_combo_10"))

    def test_short_ids_are_kept_whole(self):
        self.assertEqual(self.tag("ab_c1"), "ABC1")

    def seat_names(self, name, tag="TAGX01", cid="some_id"):
        ns = {"tag": tag, "name": name, "a": types.SimpleNamespace(id=cid)}
        exec(node(ast.Assign, "dest =", "name.upper()"), ns)
        exec(node(ast.Assign, "lid =", "_seated_"), ns)
        return ns["dest"], ns["lid"]

    def test_two_seats_with_a_shared_start_do_not_collide(self):
        # regression: the seat's asset was named from the first six letters of the seat's name, so "armchair" and
        # "armchair_2" shared one outfit asset and the second import overwrote the first
        d1, l1 = self.seat_names("armchair")
        d2, l2 = self.seat_names("armchair_2")
        self.assertNotEqual(d1, d2)
        self.assertNotEqual(l1, l2)
        d3, _ = self.seat_names("armchair_leather_buttoned")
        self.assertEqual(len({d1, d2, d3}), 3)

    def test_seat_asset_path_is_an_engine_name(self):
        d, lid = self.seat_names("wing-back chair.2")
        self.assertTrue(d.startswith("/Game/Char/OutfitTAGX01S"))
        self.assertTrue(d.rsplit("/", 1)[1].isalnum(), d)
        self.assertTrue(lid.startswith("some_id_seated_"))
        self.assertNotIn(" ", lid)


# ---------------------------------------------------------------------------------------------------------------
class LookLine(unittest.TestCase):
    def test_held_things_and_headwear_leave_the_spread_drawing(self):
        look = "a man of about 60, heavy, carrying a stick, wearing a long coat, a top-hat and dark trousers, holding a crop."
        out = CC.spread_look(look)
        self.assertIn("long coat", out)
        self.assertIn("dark trousers", out)
        for gone in ("hat", "stick", "crop"):
            self.assertNotIn(gone, out)

    def test_everything_after_what_is_carried_goes(self):
        out = CC.spread_look("a woman, wearing a grey dress, carrying a basket and a shawl over it")
        self.assertIn("grey dress", out)
        self.assertNotIn("basket", out)
        self.assertNotIn("shawl", out)

    def test_outer_layer_leaves_by_its_whole_phrase(self):
        look = "a man of about 40, wearing a coarse brown tunic, dark trousers and a dark wool cloak"
        out = CC.spread_look(look, ["dark", "wool", "cloak"])
        self.assertNotIn("cloak", out)
        self.assertIn("dark trousers", out)          # one shared word must not drop another garment
        self.assertIn("coarse brown tunic", out)

    def test_without_the_fact_the_outer_layer_stays(self):
        look = "a man, wearing a tunic and a wool cloak"
        self.assertIn("wool cloak", CC.spread_look(look))
        self.assertIn("wool cloak", CC.spread_look(look, None))

    def test_a_look_with_nothing_worn(self):
        self.assertEqual(CC.spread_look("a tall thin man of about 40."), "a tall thin man of about 40")


class GroomWords(unittest.TestCase):
    def test_up_do_has_no_length_word(self):
        worn, fh = CC.groom_words("WI_Hair_S_UpdoBuns")
        self.assertIn("pinned up", worn)
        self.assertIn("bun", worn)
        self.assertNotIn("short", worn)               # an up-do's groom length is what is left loose, not the hair's
        self.assertEqual(fh, "none")

    def test_short_hair_and_facial_hair(self):
        worn, fh = CC.groom_words("WI_Hair_S_SlickBack", beard="WI_Beard_S_Full", moustache="WI_Mustache_L_Handlebar")
        self.assertTrue(worn.startswith("short hair"))
        self.assertIn("slicked back", worn)
        self.assertIn("short full beard", fh)
        self.assertIn("large handlebar moustache", fh)
        self.assertIn(" and ", fh)

    def test_entry_facts_come_from_the_build(self):
        ch = {"grooms": {"Hair": "WI_Hair_S_UpdoBuns", "Beard": "WI_Beard_S_Stubble", "Mustache": "WI_Mustache_S_Full",
                         "Eyebrows": "WI_Eyebrows_M_Full"},
              "clear_slots": ["Beard", "Mustache"], "scores": {"hair_colour": "black with grey streaks"}}
        f = CC.entry_facts(ch, ["dark wool cloak"])
        self.assertEqual(f["hair_colour"], "black with grey streaks")
        self.assertEqual(f["facial_hair"], "none")               # a cleared slot is no beard, whatever groom is named
        self.assertEqual(f["missing_layers"], ["dark wool cloak"])
        self.assertNotIn("Eyebrows", f["grooms"])
        self.assertEqual(CC.entry_facts(ch)["missing_layers"], [])

    def test_a_beard_that_is_worn_is_said(self):
        ch = {"grooms": {"Hair": "WI_Hair_M_Layered", "Beard": "WI_Beard_L_Full"}, "clear_slots": ["Mustache"], "scores": {}}
        f = CC.entry_facts(ch)
        self.assertIn("beard", f["facial_hair"])
        self.assertTrue(f["hair_worn"].startswith("medium-length hair"))


# ---------------------------------------------------------------------------------------------------------------
class DrawingAngle(unittest.TestCase):
    """The angle a drawing was made at, from its seed and its round's pose map; the round's start angle and canvas."""

    def setUp(self):
        self.ns = {"Path": Path}
        exec(nested_function("drawing_deg"), self.ns)

    def run_dir(self, run, maps):
        d = Path(tempfile.mkdtemp(dir=TMP)) / f"run{run}"
        d.mkdir()
        for m in maps:
            (d / m).write_bytes(b"")
        return d

    def seed(self, run, rnd, k):
        return eval(node(ast.Assign, "seeds =", "1000", part="value").replace("for k in range(3)", "for k in [k]"),
                    {"run_n": run, "rnd": rnd, "k": k})[0]

    def test_angle_from_the_rounds_pose_map(self):
        d = self.run_dir(1, ["pose66_r0.png", "pose72_r1.png", "pose78_r2.png"])
        for rnd, deg in ((0, 66.0), (1, 72.0), (2, 78.0)):
            for k in range(3):
                self.assertEqual(self.ns["drawing_deg"](d / f"spread_s{self.seed(1, rnd, k)}.png"), deg)

    def test_older_runs_with_plain_pose_maps(self):
        d = self.run_dir(0, ["pose48.png", "pose42.png", "pose54.png"])
        self.assertEqual(self.ns["drawing_deg"](d / f"spread_s{self.seed(0, 0, 0)}.png"), 42.0)
        self.assertEqual(self.ns["drawing_deg"](d / f"spread_s{self.seed(0, 2, 1)}.png"), 54.0)

    def test_unknown_names_give_none(self):
        d = self.run_dir(0, [])
        self.assertIsNone(self.ns["drawing_deg"](d / "figure.png"))
        self.assertIsNone(self.ns["drawing_deg"](d / f"spread_s{self.seed(0, 4, 0)}.png"))

    def start(self, sleeves, look, pose_deg=None):
        return eval(node(ast.Assign, "deg0 =", "pose_deg", part="value"),
                    {"a": types.SimpleNamespace(pose_deg=pose_deg), "sl_": sleeves, "lk": look.lower()})

    def test_start_angle_rises_with_looser_sleeves(self):
        close, medium, loose = (self.start(s, "a man in a coat") for s in ("close", "medium", "loose"))
        self.assertLess(close, medium)
        self.assertLess(medium, loose)
        self.assertGreaterEqual(close, 40)

    def test_without_the_sleeve_fact_a_dress_starts_higher_than_a_coat(self):
        self.assertGreater(self.start("", "a woman in a long dress"), self.start("", "a man in a coat"))

    def test_an_angle_given_on_the_command_line_wins(self):
        self.assertEqual(self.start("loose", "a man in a robe", pose_deg=50.0), 50.0)

    def test_the_wrists_of_the_pose_map_stay_inside_the_canvas(self):
        # each round raises the arms; past a set angle the canvas is wider and the figure smaller (hands that ran
        # off the picture left the gate without an armpit). The pose map's wrists (apose_map.py: shoulders 0.112 of
        # the height out, the arm 0.335 long) must lie inside the picture with a margin at every angle a run can use
        canvas = node(ast.Assign, "cw, cf =", part="value")
        step = 6
        self.assertIn("deg0 + 6 * rnd", CAST_SRC)
        rounds = 5
        for sleeves in ("close", "medium", "loose", ""):
            d0 = self.start(sleeves, "a man in a coat")
            for rnd in range(rounds):
                deg = d0 + step * rnd
                cw, cf = eval(canvas, {"deg": deg})
                wrist = (0.112 + 0.335 * math.sin(math.radians(deg))) * cf
                self.assertLess(wrist, cw / 2 - 0.05 * cw / 2, f"{sleeves or 'no fact'} round {rnd} at {deg} deg")


# ---------------------------------------------------------------------------------------------------------------
class FitJobRules(unittest.TestCase):
    """What the one command tells the fit, from the facts and the drawing's measure."""

    def test_the_text_fact_decides_a_long_garment_over_the_drawing(self):
        src = node(ast.If, 'gf.get("long") is not None', 'm["long_skirt"]')
        m = {"long_skirt": False, "shin_runs": 2}
        exec(src, {"m": m, "gf": {"long": True, "source": {"long_garment": "text"}}})
        self.assertTrue(m["long_skirt"])
        self.assertFalse(m["long_skirt_drawing"])                  # the drawing's own answer is kept for the record
        self.assertEqual(m["long_skirt_source"], "text")
        m = {"long_skirt": True}
        exec(src, {"m": m, "gf": {"long": False}})
        self.assertFalse(m["long_skirt"])
        m = {"long_skirt": True}
        exec(src, {"m": m, "gf": {"long": None}})                  # no fact: the drawing's measure stands
        self.assertTrue(m["long_skirt"])
        self.assertNotIn("long_skirt_source", m)

    def route(self, m):
        job = {}
        exec(node(ast.If, 'm["long_skirt"]', "sleeve_radius_m", "coat_skirt"), {"m": m, "job": job})
        return job

    def test_dress_rules_and_coat_rules(self):
        dress = self.route({"long_skirt": True, "thigh_run_w_frac": 0.3})
        coat = self.route({"long_skirt": False, "thigh_run_w_frac": 0.10})
        frock = self.route({"long_skirt": False, "thigh_run_w_frac": 0.30})
        self.assertTrue(dress.get("long_skirt"))
        self.assertFalse(dress["turn_legs"])                       # a hem to the floor is no ankle
        self.assertNotIn("long_skirt", coat)
        self.assertGreater(dress["sleeve_radius_m"], coat["sleeve_radius_m"])
        self.assertGreater(frock["skirt_pelvis"], coat["skirt_pelvis"])       # a coat whose skirt stands off the legs
        self.assertEqual(self.route({"long_skirt": False, "thigh_run_w_frac": None})["skirt_pelvis"], coat["skirt_pelvis"])

    def test_the_neckline_goes_to_the_fit_only_when_the_text_states_it(self):
        val = node(ast.Assign, 'job["neckline"] =', part="value")
        self.assertEqual(eval(val, {"gf0": {"neckline": "base", "source": {"neckline": "text"}}}), "base")
        self.assertIsNone(eval(val, {"gf0": {"neckline": "chin", "source": {"neckline": "picture"}}}))
        self.assertIsNone(eval(val, {"gf0": {"neckline": "chin"}}))
        self.assertIsNone(eval(val, {"gf0": {}}))

    def test_hair_that_hangs(self):
        val = node(ast.Assign, 'job["hair_hangs"] =', part="value")
        hangs = lambda **a: bool(eval(val, {"va_": a}))          # noqa: E731
        self.assertTrue(hangs(length="long", worn="down"))
        self.assertTrue(hangs(length="medium", worn="down"))
        self.assertFalse(hangs(length="long", worn="up"))          # pinned up: nothing hangs below the neck line
        self.assertFalse(hangs(length="long", worn="back"))
        self.assertFalse(hangs(length="short", worn=""))
        self.assertFalse(hangs())


class NeckRules(unittest.TestCase):
    def girth(self, gi, gn):
        addb = {}
        exec(node(ast.If, "gn > gi - 1.0", 'addb["Neck"]'), {"gi": gi, "gn": gn, "addb": addb})
        return addb

    def test_neck_girth_change_is_relative_and_only_when_the_neck_is_too_thick(self):
        self.assertEqual(self.girth(36.0, 47.5), {"Neck": -12.5})       # to the collar's inner girth less 1 cm
        self.assertEqual(self.girth(40.0, 38.0), {})                    # already inside the collar: nothing set
        self.assertEqual(self.girth(None, 40.0), {})
        self.assertEqual(self.girth(36.0, None), {})
        self.assertLess(self.girth(40.0, 39.5)["Neck"], 0)

    def test_neck_length_is_no_longer_set(self):
        # the Neck Length constraint left the built skeleton as it was; a bare neck is met by the collar lift
        seg = node(ast.If, '(combo / "report.json").exists() and not neck_p.exists()')
        self.assertNotIn('newb["Neck Length"]', seg)
        self.assertNotIn('addb["Neck Length"]', seg)

    def test_a_change_with_no_effect_stops_the_build(self):
        cond = node(ast.If, "NECK FIT HAD NO EFFECT", part="test")
        stops = lambda d_neck, d_girth: bool(eval(cond, {"d_neck": d_neck, "d_girth": d_girth}))   # noqa: E731
        self.assertTrue(stops(7.5, 0.1))
        self.assertFalse(stops(7.5, 2.2))                 # the body's girth followed
        self.assertFalse(stops(0.5, 0.0))                 # hardly anything was asked
        blk = node(ast.If, "NECK FIT HAD NO EFFECT")
        self.assertIn("raise RuntimeError", blk)

    def lift(self, bare, nl, lifted=False):
        cond = node(ast.If, 'not job.get("collar_lift")', "nl in", part="test")
        return bool(eval(cond, {"bare": bare, "nl": nl, "job": {"collar_lift": lifted}}))

    def test_only_a_collar_that_comes_up_the_neck_is_lifted(self):
        self.assertTrue(self.lift(4.5, "chin"))
        self.assertTrue(self.lift(4.5, None))             # no neckline known: the old behaviour
        self.assertFalse(self.lift(6.5, "base"))          # a round neckline is bare by design
        self.assertFalse(self.lift(9.8, "open"))
        self.assertFalse(self.lift(1.2, "chin"))          # within the limit
        self.assertFalse(self.lift(None, "chin"))
        self.assertFalse(self.lift(4.5, "chin", lifted=True))      # never twice


class ArmRules(unittest.TestCase):
    def test_arm_steps_abduct_both_arms_outward(self):
        s = CC.arm_steps(8)
        byb = {x[1]: x for x in s}
        self.assertEqual(byb["upperarm_l"][3], -8)
        self.assertEqual(byb["upperarm_r"][3], 8)
        self.assertEqual({x[1] for x in s}, {"upperarm_l", "upperarm_r", "lowerarm_l", "lowerarm_r"})
        self.assertEqual(abs(float(CC.arm_steps(12)[0][3])), 12.0)       # the layer step reads the angle back so

    def passed(self, loose, sep, low):
        return bool(eval(node(ast.Assign, "passed =", "loose", part="value"), {"loose": loose, "sep": sep, "low": low}))

    def test_close_sleeves_need_the_forearm_clear_in_most_rows(self):
        self.assertTrue(self.passed(False, {"l": 10, "r": 8}, {"l": 5, "r": 5}))
        self.assertFalse(self.passed(False, {"l": 10, "r": 7}, {"l": 5, "r": 5}))
        self.assertFalse(self.passed(False, {"l": 6, "r": 6}, {"l": 5, "r": 5}))    # the loose rule is not theirs

    def test_loose_sleeves_need_the_lower_forearm_and_hand_clear(self):
        self.assertTrue(self.passed(True, {"l": 5, "r": 4}, {"l": 4, "r": 5}))
        self.assertFalse(self.passed(True, {"l": 9, "r": 9}, {"l": 3, "r": 5}))
        self.assertFalse(self.passed(True, {"l": 0, "r": 0}, {"l": 5, "r": 0}))

    def test_the_arm_rule_is_recorded_on_the_entry(self):
        blk = node(ast.Assign, 'cj3["arm_rule"] =')
        ns = {"cj3": {}, "loose": True}
        exec(blk, ns)
        self.assertEqual(ns["cj3"]["arm_rule"]["rule"], "loose_sleeve_lower_forearm")
        ns = {"cj3": {}, "loose": False}
        exec(blk, ns)
        self.assertNotEqual(ns["cj3"]["arm_rule"]["rule"], "loose_sleeve_lower_forearm")
        self.assertTrue(ns["cj3"]["arm_rule"]["expect"])

    def test_loose_sleeves_may_go_further_out_before_the_build_stops(self):
        val = node(ast.Assign, "ab_max =", part="value")
        loose = eval(val, {"ch": {"garment": {"sleeves": "loose"}}})
        close = eval(val, {"ch": {"garment": {"sleeves": "close"}}})
        none = eval(val, {"ch": {}})
        self.assertGreater(loose, close)
        self.assertEqual(close, none)

    def test_the_first_pick_is_the_smallest_abduction_that_separates_both_forearms(self):
        loop = node(ast.For, 'res.get("wide_all", {}).get(k, {})', "pick = ab")

        def pick(seps):
            res = {"wide_all": {k: {"l": {"separated": a}, "r": {"separated": b}} for k, (a, b) in seps.items()}}
            ns = {"res": res, "pick": 12}
            exec(loop, ns)
            return ns["pick"]
        self.assertEqual(pick({"a4": (9, 8), "a8": (10, 10), "a12": (10, 10)}), 4)
        self.assertEqual(pick({"a4": (9, 7), "a8": (8, 8), "a12": (10, 10)}), 8)
        self.assertEqual(pick({"a4": (0, 0), "a8": (7, 10), "a12": (10, 10)}), 12)
        self.assertEqual(pick({"a4": (0, 0), "a8": (0, 0), "a12": (0, 0)}), 12)


class HeadLeftOnTheGarment(unittest.TestCase):
    def test_garment_in_the_head_stops_the_build(self):
        fn = nested_function("head_check")
        d = Path(tempfile.mkdtemp(dir=TMP))
        saved = []

        def run(front, round_):
            (d / "report.json").write_text(json.dumps({"head_volume_verts": front, "head_round_verts": round_}))
            ns = {"json": json, "combo": d, "CH": {}, "HEAD_VOLUME_MAX": CC.HEAD_VOLUME_MAX,
                  "save": lambda: saved.append(1)}
            exec(fn, ns)
            ns["head_check"]()
            return ns["CH"]
        self.assertEqual(run(0, 0)["head_volume_verts"]["front"], 0)
        run(CC.HEAD_VOLUME_MAX // 2, CC.HEAD_VOLUME_MAX // 2)             # a collar's edge, a curl: allowed
        with self.assertRaises(RuntimeError) as e:
            run(CC.HEAD_VOLUME_MAX, 1)                                    # front and round the head are counted together
        self.assertIn("GARMENT IN THE HEAD", str(e.exception))
        with self.assertRaises(RuntimeError):
            run(0, 5000)                                                  # hair left round the head, none in front
        self.assertTrue(saved)


# ---------------------------------------------------------------------------------------------------------------
class StartOfRunCheck(unittest.TestCase):
    def test_missing_inputs_stop_the_command_before_any_step(self):
        work = Path(TMP) / "work_missing"
        env = dict(os.environ, ASSETS_LIBRARY=os.path.join(TMP, "no_library"), ASSETS_BLENDER=os.path.join(TMP, "no_blender"),
                   ASSETS_UE_HOME=os.path.join(TMP, "no_ue"), ASSETS_SCHED_DIR=os.path.join(TMP, "no_sched"),
                   ASSETS_IMG3D_TORCH_PY=os.path.join(TMP, "no_py1"), ASSETS_QWEN_PY=os.path.join(TMP, "no_py2"),
                   ASSETS_GUIDES_PY=os.path.join(TMP, "no_py3"))
        p = subprocess.run([sys.executable, str(ASSETS / "img3d" / "cast_char.py"), "--id", "x_test_01", "--sheet",
                            os.path.join(TMP, "no_sheet.png"), "--look", "a person, wearing a coat", "--lora",
                            os.path.join(TMP, "no_lora.safetensors"), "--trigger", "chr_x", "--work", str(work),
                            "--scene-tool", os.path.join(TMP, "no_scene_tool")],
                           capture_output=True, text=True, timeout=120, env=env)
        self.assertNotEqual(p.returncode, 0)
        msg = p.stdout + p.stderr
        self.assertIn("missing inputs, nothing run", msg)
        for what in ("sheet", "lora", "blender", "engine project", "scene tool", "library", "preset portraits",
                     "image-to-3D python", "image model python", "guides python"):
            self.assertIn(what, msg)
        self.assertEqual([q.name for q in work.iterdir()], [])           # nothing was started or written
        self.assertFalse(Path(TMP, "no_library").exists())
        self.assertFalse(Path(TMP, "no_sched").exists())


# ---------------------------------------------------------------------------------------------------------------
class OuterLayerPick(unittest.TestCase):
    """Which of several passing drawings of the outer layer is worn as on the sheet: every pair, both orders."""

    def pick(self, n, answer):
        ld = Path(tempfile.mkdtemp(dir=TMP))
        sheet = ld / "sheet.png"
        Image.new("RGB", (64, 48), "white").save(sheet)
        pngs = []
        for i in range(n):
            p = ld / f"layer_s{i}.png"
            Image.new("RGB", (40, 60), (30 * i, 80, 120)).save(p)
            pngs.append(p)

        def fake_line(args, log_file=None):
            lines = ["loading"]
            for ln in (ld / "pick.jsonl").read_text().splitlines():
                q = json.loads(ln)
                i, k = (int(x) for x in q["id"].split("|"))
                a = answer(i, k)
                if a is not None:
                    lines.append(json.dumps({"id": q["id"], "answers": {"pick": a}}))
            Path(log_file).write_text("\n".join(lines) + "\n")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        old = CC.line
        CC.line = fake_line
        try:
            return CC.layer_pick(ld, sheet, pngs, "TAG")
        finally:
            CC.line = old

    def test_the_drawing_that_wins_in_both_orders_is_picked(self):
        r = self.pick(2, lambda i, k: "A" if i == 1 else "B")            # drawing 1, wherever it is shown
        self.assertEqual(r["pick"], 1)
        self.assertEqual(r["wins"], [0, 2])
        self.assertEqual(set(r["answers"]), {"0|1", "1|0"})

    def test_a_model_that_always_says_the_same_side_decides_nothing(self):
        r = self.pick(3, lambda i, k: "A")
        self.assertEqual(r["wins"], [2, 2, 2])
        self.assertEqual(r["pick"], 0)                                    # ties go to the earlier drawing

    def test_three_drawings(self):
        r = self.pick(3, lambda i, k: "A" if i == 2 else ("B" if k == 2 else "A"))
        self.assertEqual(r["pick"], 2)
        self.assertEqual(r["wins"][2], 4)

    def test_no_answers_is_an_error_not_a_pick(self):
        with self.assertRaises(RuntimeError):
            self.pick(2, lambda i, k: None)


class OuterLayerVerdict(unittest.TestCase):
    """bl_layer.py's verdict on a fitted layer from its measures (the measures themselves need Blender)."""

    def verdict(self, covered, area, min_z=0.09, top=1.53, neck=1.54, job=None):
        src = LAYER.read_text()
        tree = ast.parse(src)

        def top_assign(target):
            hits = [ast.get_source_segment(src, n) for n in tree.body if isinstance(n, ast.Assign)
                    and ast.get_source_segment(src, n.targets[0]) == target]
            assert len(hits) == 1, f"bl_layer.py: {len(hits)} top-level assignments to {target}"
            return hits[0]
        n = 100
        Pq = np.zeros((n, 3))
        Pq[:, 2] = np.linspace(min_z, top, n)
        va = np.ones(n)
        dB = np.where(np.arange(n) < round(area * n), 0.05, 0.5)
        ns = {"np": np, "JOB": job or {}, "REP": {}, "Pq": Pq, "va": va, "dB": dB, "contact": 0.06, "reach": 0.2,
              "J": lambda nm: (0.0, 0.0, neck),
              "shoulders": {s: {"verts": 200, "covered": c} for s, c in zip(("l", "r"), covered)}}
        for t in ("cover_min", "reach_min", 'REP["checks"]', 'REP["checks"]["ok"]'):
            exec(top_assign(t), ns)
        return ns["REP"]["checks"]

    def test_the_measured_cases(self):
        # the numbers the rule was set between (in bl_layer.py's own comment): the accepted cloak on both shoulders,
        # two layers worn over one shoulder, and the cloak displaced off the body
        self.assertTrue(self.verdict((0.615, 0.796), 0.936)["ok"])
        one = self.verdict((0.95, 0.0), 1.0)
        self.assertTrue(one["ok"])
        self.assertEqual(one["rests_on"], ["l"])                          # no left-right centring
        self.assertTrue(self.verdict((0.0, 0.505), 0.961)["ok"])
        off = self.verdict((0.334, 0.30), 0.833)
        self.assertFalse(off["ok"])
        self.assertEqual(off["rests_on"], [])

    def test_each_rule_alone_fails_the_layer(self):
        self.assertFalse(self.verdict((0.9, 0.9), 0.70)["ok"])            # hanging away from the body
        self.assertFalse(self.verdict((0.2, 0.2), 0.99)["ok"])            # resting on neither shoulder
        self.assertFalse(self.verdict((0.9, 0.9), 0.99, min_z=-0.05)["ok"])        # through the floor
        self.assertFalse(self.verdict((0.9, 0.9), 0.99, top=1.25)["ok"])           # its top far under the neck

    def test_the_thresholds_are_the_jobs(self):
        self.assertTrue(self.verdict((0.334, 0.30), 0.833, job={"shoulder_cover": 0.3, "reach_area": 0.8})["ok"])


# ---------------------------------------------------------------------------------------------------------------
class SkinProbe(unittest.TestCase):
    """Skin through the cloth, counted at the shoulders on a flat-lit still and its twin without the face mesh."""
    SKIN = (0.80, 0.56, 0.44)
    CLOTH = (0.10, 0.10, 0.12)
    BGC = (0.5, 0.5, 0.5)

    def shot(self, patches=(), scale=1.0, specks=0, name="front", joints=True, skin=None):
        """patches: (x, y, w, h) boxes of seen head-mesh skin, in a 400 x 400 layout scaled by SCALE"""
        S = int(400 * scale)
        im = np.empty((S, S, 3), np.float32)
        im[:] = self.BGC
        z = lambda v: int(round(v * scale))                                  # noqa: E731
        im[z(110):, z(60):z(340)] = self.CLOTH                                # the garment
        nf = im.copy()                                                        # the twin: no head mesh
        face = (slice(z(20), z(90)), slice(z(160), z(240)))
        im[face] = self.SKIN
        for x, y, w, h in patches:
            im[z(y):z(y) + h, z(x):z(x) + w] = skin or self.SKIN
        rng = np.random.default_rng(1)
        for _ in range(specks):                                               # single pixels: shimmer, not a surface
            y, x = rng.integers(z(125), z(200)), rng.integers(z(112), z(172))
            if (y % 4, x % 4) == (0, 0):
                im[y, x] = self.SKIN
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.fromarray((im * 255).astype(np.uint8)).save(d / "a.png")
        Image.fromarray((nf * 255).astype(np.uint8)).save(d / "b.png")
        J = {"neck_01": [200 * scale, 100 * scale], "upperarm_l": [260 * scale, 120 * scale],
             "upperarm_r": [140 * scale, 120 * scale], "spine_04": [200 * scale, 200 * scale]}
        return {"name": name, "path": str(d / "a.png"), "noface_path": str(d / "b.png"), "joints_px": J if joints else {}}

    def test_clean_shoulders_pass(self):
        r = SP.probe_shot(self.shot())
        self.assertTrue(r["ok"])
        self.assertEqual(r["skin_px"], 0)

    def test_a_view_fails_only_over_both_the_share_and_the_count(self):
        small = SP.probe_shot(self.shot())["region_px"]
        self.assertGreater(SP.MIN_PX / small, SP.MAX_FRAC)        # in this small region the count floor is the looser rule
        n_ok, n_bad = SP.MIN_PX - 10, SP.MIN_PX + 24
        ok = SP.probe_shot(self.shot([(120, 130, n_ok // 5, 5)]))
        self.assertGreater(ok["skin_frac"], SP.MAX_FRAC)          # over the share, under the count: edge pixels, passes
        self.assertLessEqual(ok["skin_px"], SP.MIN_PX)
        self.assertTrue(ok["ok"])
        bad = SP.probe_shot(self.shot([(120, 130, n_bad // 4, 4)]))
        self.assertGreater(bad["skin_px"], SP.MIN_PX)
        self.assertGreater(bad["skin_frac"], SP.MAX_FRAC)
        self.assertFalse(bad["ok"])

    def test_in_a_large_region_the_share_is_the_rule(self):
        big = SP.probe_shot(self.shot([(120, 130, 12, 6)], scale=3.0))
        self.assertGreater(big["skin_px"], SP.MIN_PX)             # over the count
        self.assertLess(big["skin_frac"], SP.MAX_FRAC)            # under the share
        self.assertTrue(big["ok"])

    def test_the_neck_column_is_left_out_and_follows_the_opening(self):
        collarbone = [(232, 130, 12, 10)]                         # beside the neck, inside a wide open neckline
        self.assertFalse(SP.probe_shot(self.shot(collarbone))["ok"])                  # the default column: counted
        self.assertTrue(SP.probe_shot(self.shot(collarbone), neck_half=0.42)["ok"])   # a measured wide opening
        throat = [(190, 130, 20, 10)]
        self.assertTrue(SP.probe_shot(self.shot(throat))["ok"])                       # skin in the collar's V: by design
        shoulder = [(262, 130, 12, 10)]
        self.assertFalse(SP.probe_shot(self.shot(shoulder), neck_half=0.42)["ok"])    # the shoulders are still measured

    def test_only_skin_coloured_head_mesh_counts(self):
        dark = SP.probe_shot(self.shot([(120, 130, 20, 10)], skin=(0.30, 0.30, 0.34)))          # hair on the shoulder
        self.assertTrue(dark["ok"])
        self.assertEqual(dark["skin_px"], 0)
        self.assertGreater(dark["head_mesh_px"], 0)

    def test_scattered_single_pixels_are_not_a_surface(self):
        r = SP.probe_shot(self.shot(specks=4000))
        self.assertEqual(r["skin_px"], 0)
        self.assertTrue(r["ok"])

    def test_without_the_joints_nothing_is_claimed(self):
        r = SP.probe_shot(self.shot(joints=False))
        self.assertIsNone(r["ok"])

    def run_main(self, shots, *args):
        d = Path(tempfile.mkdtemp(dir=TMP))
        (d / "result.json").write_text(json.dumps({"shots": shots}))
        out = d / "probe.json"
        old = sys.argv
        sys.argv = ["skin_probe.py", str(d / "result.json"), "--out", str(out), *args]
        try:
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                rc = SP.main()
        finally:
            sys.argv = old
        return rc, json.loads(out.read_text())

    def test_the_command_fails_on_one_bad_view_and_reads_the_neckline_fact(self):
        collarbone = [(232, 130, 12, 10)]
        rc, res = self.run_main([self.shot(), self.shot(collarbone, name="close")])
        self.assertEqual(rc, 1)
        self.assertFalse(res["ok"])
        self.assertEqual([s["ok"] for s in res["shots"]], [True, False])
        rc, res = self.run_main([self.shot(), self.shot(collarbone, name="close")], "--neckline", "open")
        self.assertEqual(rc, 0)
        self.assertTrue(res["ok"])
        # a collar up to the chin keeps the narrow column whatever opening is passed
        rc, _ = self.run_main([self.shot(collarbone)], "--neckline", "chin", "--neck-half", "0.45")
        self.assertEqual(rc, 1)
        # a measured opening is used for a base or open neckline
        rc, _ = self.run_main([self.shot(collarbone)], "--neckline", "base", "--neck-half", "0.45")
        self.assertEqual(rc, 0)
        rc, _ = self.run_main([self.shot(collarbone)], "--neckline", "open", "--neck-half", "0.21")
        self.assertEqual(rc, 1)

    def test_views_without_a_twin_are_not_probed_and_no_views_is_not_a_pass(self):
        s = self.shot()
        s.pop("noface_path")
        rc, res = self.run_main([s])
        self.assertEqual(rc, 1)
        self.assertEqual(res["shots"], [])


# ---------------------------------------------------------------------------------------------------------------
class Stances(unittest.TestCase):
    def setUp(self):
        self.lib = Path(tempfile.mkdtemp(dir=TMP))

        def entry(cid, stance=None):
            d = self.lib / "character" / cid
            d.mkdir(parents=True)
            e = {"id": cid}
            if stance:
                e["stance"] = stance
            (d / "character.json").write_text(json.dumps(e))
        entry("p_01")
        entry("p_01_seated_a", {"base": "p_01", "pose": "seated", "seat_asset": "chair_a", "seat_z_m": 0.45})
        entry("p_01_seated_b", {"base": "p_01", "pose": "seated", "seat_asset": "chair_b", "seat_z_m": 0.52})
        entry("q_01")
        entry("q_01_seated_a", {"base": "q_01", "pose": "seated", "seat_asset": "chair_a", "seat_z_m": 0.45})
        (self.lib / "character" / "broken").mkdir()
        (self.lib / "character" / "broken" / "character.json").write_text("{not json")

    def test_pose_families(self):
        self.assertEqual(ST.family("sit_armchair_relaxed"), "seated")
        self.assertEqual(ST.family("seated"), "seated")
        self.assertEqual(ST.family("walk_slow"), "walk")
        self.assertEqual(ST.family("stand_relaxed"), "stand")
        self.assertEqual(ST.family(None), "stand")

    def test_standing_is_the_base_whichever_id_is_given(self):
        self.assertEqual(ST.resolve(self.lib, "p_01", {"pose": "stand_relaxed"})[0], "p_01")
        self.assertEqual(ST.resolve(self.lib, "p_01_seated_a", {"pose": "stand_relaxed"})[0], "p_01")
        self.assertEqual(ST.resolve(self.lib, "p_01")[0], "p_01")

    def test_seated_in_the_seat_it_was_settled_for(self):
        self.assertEqual(ST.resolve(self.lib, "p_01", {"pose": "seated"}, "chair_b")[0], "p_01_seated_b")
        self.assertEqual(ST.resolve(self.lib, "p_01_seated_b", {"pose": "sit_armchair_relaxed"}, "chair_a")[0], "p_01_seated_a")
        self.assertEqual(ST.resolve(self.lib, "q_01", {"pose": "seated"}, "chair_a")[0], "q_01_seated_a")    # never another's

    def test_an_unknown_seat_takes_the_nearest_settled_height_within_5_cm(self):
        got = ST.resolve(self.lib, "p_01", {"pose": "seated", "params": {"seat_z_m": 0.50}}, "chair_c")[0]
        self.assertEqual(got, "p_01_seated_b")
        got = ST.resolve(self.lib, "p_01", {"pose": "seated", "params": {"seat_z_m": 0.47}}, "chair_c")[0]
        self.assertEqual(got, "p_01_seated_a")

    def test_no_settled_variant_falls_back_to_the_base_and_says_so(self):
        cid, note = ST.resolve(self.lib, "p_01", {"pose": "seated", "params": {"seat_z_m": 0.70}}, "stool_x")
        self.assertEqual(cid, "p_01")
        self.assertIn("no seated variant", note)
        cid, note = ST.resolve(self.lib, "q_01", {"pose": "seated"}, "chair_b")
        self.assertEqual(cid, "q_01")


class CheckSheetSmallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import char_checks as K
        cls.K = K

    def test_skin_hue(self):
        c = np.array([[0.80, 0.56, 0.44], [0.10, 0.10, 0.12], [0.9, 0.9, 0.9], [0.3, 0.5, 0.8]])
        self.assertEqual(list(self.K.skin_hue(c)), [True, False, False, False])

    def test_white_is_light_and_unsaturated(self):
        c = np.array([[0.9, 0.9, 0.88], [0.4, 0.4, 0.4], [0.9, 0.5, 0.4]])
        self.assertEqual(list(self.K.white_share(c)), [True, False, False])

    def test_lightness_classes(self):
        self.assertEqual([self.K.lightness_class(v) for v in (0.2, 0.6, 0.9)], ["dark", "mid", "light"])


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
