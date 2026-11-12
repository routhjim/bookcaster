"""The character chooser's facts and the face pick (characters/choose_mh.py, face_pick.py), run whole with stand-ins
for the models: no GPU, no model files, no vision model process.

choose_mh.py refuses the CPU and loads CLIP and DINOv2; here "torch" and "transformers" are small stand-ins (every
CLIP choice is its first option unless a test says otherwise; head-shape similarity falls with a preset's place in the
alphabet) and the vision model's process is a function that answers each question from a table. What is tested is
the chooser's own logic round those answers: a fact comes from the written description only when the description
speaks of it, the picture is asked only for what the text leaves open, closed answers map to the garment's facts,
the hair colour's measured values replace the defaults, a full beard is a full one, and the face pick's tally over
both orders with its tie rule and its narrowing by the drawn face's age."""

import contextlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

CHARS = Path(__file__).resolve().parents[1] / "assets" / "characters"
TMP = Path(tempfile.mkdtemp(prefix="char_choose_test_"))
MAN, WOMAN = (40, 60, 200), (220, 60, 40)          # a picture's first pixel says which the stand-in CLIP sees


class T:
    """the little of a tensor the chooser uses"""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float64)

    def to(self, *a, **k):
        return self
    half = float = cpu = to

    def numpy(self):
        return self.a

    def softmax(self, dim):
        e = np.exp(self.a - self.a.max(axis=dim, keepdims=True))
        return T(e / e.sum(axis=dim, keepdims=True))

    def __getitem__(self, i):
        return T(self.a[i])

    def norm(self, dim=-1, keepdim=False):
        return T(np.linalg.norm(self.a, axis=dim, keepdims=keepdim))

    def __truediv__(self, o):
        return T(self.a / (o.a if isinstance(o, T) else o))


class Batch(dict):
    def to(self, *a, **k):
        return self


CLIP_RULES = []        # (substring of the first text, function(texts, image) -> index to prefer)


def clip_logits(texts, image):
    z = np.zeros(len(texts))
    if any("man" in t and "woman" not in t for t in texts) and any("woman" in t for t in texts) and len(texts) == 2:
        z[1 if image.getpixel((0, 0))[0] > 128 else 0] = 5.0            # man / woman by the picture's colour
    for sub, fn in CLIP_RULES:
        if sub in texts[0]:
            z[:] = 0
            z[fn(texts, image)] = 5.0
    return z


class _Loader:
    @classmethod
    def from_pretrained(cls, *a, **k):
        return cls()

    def to(self, *a, **k):
        return self
    eval = half = to


class CLIPProcessor(_Loader):
    def __call__(self, text=None, images=None, **k):
        return Batch(pixel_values=T([[0.0]]), texts=list(text), image=images[0])


class CLIPModel(_Loader):
    def __call__(self, pixel_values=None, texts=None, image=None):
        return types.SimpleNamespace(logits_per_image=T([clip_logits(texts, image)]))


class AutoImageProcessor(_Loader):
    def __call__(self, images=None, **k):
        return Batch(pixel_values=T(np.arange(len(images))))


class AutoModel(_Loader):
    def __call__(self, pixel_values=None):
        n = len(pixel_values.a)
        e = np.zeros((n, 1, 2))
        e[:, 0, 0] = 1.0
        e[:, 0, 1] = 0.2 * np.arange(n)          # the first is the sheet's head; later ones are less like it
        return types.SimpleNamespace(last_hidden_state=T(e))


_saved = {k: sys.modules.get(k) for k in ("torch", "transformers")}
_torch = types.ModuleType("torch")
_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
_torch.inference_mode = contextlib.nullcontext
_tf = types.ModuleType("transformers")
_tf.CLIPProcessor, _tf.CLIPModel, _tf.AutoImageProcessor, _tf.AutoModel = CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel
sys.modules["torch"], sys.modules["transformers"] = _torch, _tf
sys.path.insert(0, str(CHARS))
import choose_mh as CM  # noqa: E402
import face_pick as FP  # noqa: E402


def tearDownModule():
    for k, v in _saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v
    shutil.rmtree(TMP, ignore_errors=True)


class Judge:
    """the vision model's process: answers by question id and key from a table, and remembers what it was asked"""

    def __init__(self, table):
        self.table = table
        self.asked = {}

    def __call__(self, cmd, input=None, capture_output=None, text=None, **k):
        out = []
        for ln in (input or "").splitlines():
            q = json.loads(ln)
            self.asked.setdefault(q["id"], {}).update(q["ask"])
            src = self.table.get(q["id"], {})
            ans = {key: (src(q["id"], key) if callable(src) else src.get(key, "")) for key in q["ask"]}
            out.append(json.dumps({"id": q["id"], "answers": ans}))
        return subprocess.CompletedProcess(cmd, 0, stdout="loading\n" + "\n".join(out) + "\n", stderr="")


PRESETS = {"Alda": WOMAN, "Bert": MAN, "Cara": WOMAN, "Dov": MAN, "Emil": MAN}
LOOK = ("a man of about 40, bearded, grey-brown hair, weathered skin, wearing a coarse brown wool tunic and a dark "
        "blue wool cloak")


def choose(table, look=LOOK, sheet_colour=MAN, hair_map=None):
    d = Path(tempfile.mkdtemp(dir=TMP))
    thumbs = d / "thumbs"
    thumbs.mkdir()
    for n, c in PRESETS.items():
        Image.new("RGB", (120, 80), c).save(thumbs / f"{n}_0.png")
    Image.new("RGB", (1024, 1024), MAN).save(thumbs / "Flat_1.png")          # a flat face texture: not a portrait
    sheet = d / "sheet.png"
    Image.new("RGB", (200, 120), sheet_colour).save(sheet)
    out = d / "choice.json"
    argv = ["choose_mh.py", str(sheet), "--thumbs", str(thumbs), "--out", str(out), "--look", look]
    if hair_map is not None:
        hp = d / "hair.json"
        hp.write_text(json.dumps(hair_map))
        argv += ["--hair-map", str(hp)]
    judge = Judge(table)
    with mock.patch.object(sys, "argv", argv), mock.patch("subprocess.run", judge), \
            contextlib.redirect_stdout(io.StringIO()):
        CM.main()
    return json.loads(out.read_text()), judge


def colour_no(word):
    return str(list(CM.COLOURS).index(word))


UNSTATED = "unstated"
TEXT = {"moustache": UNSTATED, "beard": "yes", "worn": UNSTATED, "length": UNSTATED, "bald": UNSTATED,
        "style": UNSTATED, "beard_kind": UNSTATED, "moustache_kind": UNSTATED, "colour": colour_no("grey-brown"),
        "build": UNSTATED, "long_garment": "no", "neckline": "base", "outer_layer": "yes", "sleeves": UNSTATED,
        "says_neckline": "no", "says_sleeves": "no", "says_long_garment": "yes", "says_outer_layer": "yes"}
HEAD = {"moustache": "yes", "beard": "no", "worn": "down", "length": "medium", "bald": "no", "style": "messy",
        "beard_kind": "full", "moustache_kind": "full", "colour": colour_no("jet black")}
TABLE = {"text": TEXT, "head": HEAD, "fig": {"build": "2", "long_garment": "yes"},
         "bust": {"neckline": "open", "sleeves": "loose", "outer_layer": "no"}, "outer": {"words": "dark blue wool cloak"}}


class TextFirstFacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ch, cls.judge = choose(TABLE)

    def test_the_text_answers_what_it_states(self):
        src = self.ch["scores"]["answer_source"]
        self.assertEqual(src["beard"], "text")
        self.assertEqual(src["colour"], "text")
        self.assertEqual(src["long_garment"], "text")
        self.assertEqual(src["outer_layer"], "text")
        self.assertEqual(self.ch["scores"]["hair_colour"], "grey-brown")        # not the picture's jet black
        self.assertIn("Beard", self.ch["grooms"])                               # the text's beard, not the picture's none

    def test_the_picture_is_asked_only_what_the_text_leaves_open(self):
        asked = self.judge.asked
        self.assertNotIn("beard", asked["head"])
        self.assertNotIn("colour", asked["head"])
        self.assertIn("moustache", asked["head"])
        self.assertNotIn("long_garment", asked.get("fig", {}))
        self.assertNotIn("outer_layer", asked["bust"])
        self.assertIn("build", asked["fig"])

    def test_a_garment_fact_is_the_texts_only_when_the_text_speaks_of_that_part(self):
        # the text never mentions the neck; its "base" is not a fact, the upper-body picture's answer is used
        src = self.ch["scores"]["answer_source"]
        self.assertEqual(src["neckline"], "picture")
        self.assertEqual(self.ch["garment"]["neckline"], "open")
        self.assertEqual(self.ch["garment"]["source"]["neckline"], "picture")
        self.assertIn("neckline", self.judge.asked["bust"])
        self.assertEqual(src["sleeves"], "picture")
        self.assertEqual(self.ch["garment"]["sleeves"], "loose")

    def test_garment_parts_are_asked_of_the_upper_body_and_the_build_of_the_figure(self):
        asked = self.judge.asked
        for k in ("neckline", "sleeves"):
            self.assertIn(k, asked["bust"])
            self.assertNotIn(k, asked["head"])
        self.assertNotIn("build", asked["head"])

    def test_garment_facts_are_closed(self):
        g = self.ch["garment"]
        self.assertIs(g["long"], False)
        self.assertIs(g["outer_layer"], True)
        self.assertEqual(g["outer_words"], ["dark", "blue", "wool", "cloak"])
        self.assertIn(g["neckline"], ("chin", "base", "open"))
        self.assertIn(g["sleeves"], ("close", "medium", "loose"))

    def test_the_stated_age_and_the_pool(self):
        sc = self.ch["scores"]
        self.assertEqual(sc["age_used"], 40.0)
        self.assertEqual(sc["preset_pool"], ["Bert", "Dov", "Emil"])           # every preset of the gender, by head shape
        self.assertEqual(self.ch["base"], "Bert")
        self.assertEqual(self.ch["gender"], "man")

    def test_a_full_beard_is_a_full_one(self):
        self.assertEqual(self.ch["scores"]["vlm_answers"]["beard_kind"], "full")
        b = self.ch["grooms"]["Beard"].replace("WI_Beard_", "")
        self.assertNotIn("PencilThin", b)
        self.assertNotIn("Stubble", b)
        self.assertTrue("Full" in b or "Messy" in b or b.startswith("L_"), b)

    def test_hair_colour_goes_to_every_groom(self):
        mel, red, wht = CM.COLOURS["grey-brown"]
        for d in ("Hair", "Eyebrows", "Beards", "Mustaches"):
            self.assertEqual(self.ch["params"][f"Bindings/{d}/"], {"Melanin": mel, "Redness": red, "Whiteness": wht})
        self.assertGreater(wht, 0.0)                    # a greying colour is its base colour with whiteness

    def test_medium_messy_hair_worn_down(self):
        h = self.ch["grooms"]["Hair"].replace("WI_Hair_", "")
        self.assertTrue(h.startswith("M_"), h)
        self.assertTrue(any(k in h for k in ("Messy", "Clumps", "Casual")), h)


class ChooserCases(unittest.TestCase):
    def test_measured_hair_values_replace_the_defaults(self):
        ch, _ = choose(TABLE, hair_map={"grey-brown": {"whiteness": 0.5, "melanin": 0.45, "redness": 0.3},
                                        "brown": {"whiteness": 0.9}})
        self.assertEqual(ch["params"]["Bindings/Hair/"], {"Melanin": 0.45, "Redness": 0.3, "Whiteness": 0.5})
        self.assertEqual(ch["scores"]["hair_measured"], {"melanin": 0.45, "redness": 0.3, "whiteness": 0.5})
        ch, _ = choose(TABLE, hair_map={"brown": {"whiteness": 0.9}})           # no entry for this word: the defaults
        self.assertEqual(ch["params"]["Bindings/Hair/"]["Whiteness"], CM.COLOURS["grey-brown"][2])
        self.assertNotIn("hair_measured", ch["scores"])

    def test_clean_shaven_in_the_text_wins_over_a_shaded_drawing(self):
        text = dict(TEXT, beard="no", moustache="no")
        head = dict(HEAD, beard="yes", moustache="yes", beard_kind="stubble")
        ch, judge = choose(dict(TABLE, text=text, head=head), look="a man of about 40, clean-shaven, wearing a coat")
        self.assertIn("Beard", ch["clear_slots"])
        self.assertIn("Mustache", ch["clear_slots"])
        self.assertNotIn("Beard", ch["grooms"])
        self.assertNotIn("Mustache", ch["grooms"])
        self.assertNotIn("beard", judge.asked["head"])
        self.assertNotIn("moustache", judge.asked["head"])

    def test_without_a_description_everything_is_asked_of_the_picture(self):
        ch, judge = choose(TABLE, look="")
        self.assertNotIn("text", judge.asked)
        self.assertEqual(set(ch["scores"]["answer_source"].values()), {"picture"})
        self.assertEqual(ch["scores"]["hair_colour"], "jet black")
        self.assertNotIn("Beard", ch["grooms"])                        # the picture's "no"
        self.assertIs(ch["garment"]["long"], True)
        self.assertIs(ch["garment"]["outer_layer"], False)
        self.assertEqual(ch["garment"]["outer_words"], [])

    def test_a_text_answer_that_does_not_state_is_not_a_fact(self):
        for vague in ("unstated", "not stated", ""):
            ch, judge = choose(dict(TABLE, text=dict(TEXT, beard=vague)))
            self.assertEqual(ch["scores"]["answer_source"]["beard"], "picture", vague)
            self.assertIn("beard", judge.asked["head"])

    def test_an_outer_layer_without_a_description_has_no_words(self):
        ch, _ = choose(dict(TABLE, bust=dict(TABLE["bust"], outer_layer="yes")), look="")
        self.assertIs(ch["garment"]["outer_layer"], True)
        self.assertEqual(ch["garment"]["outer_words"], [])              # the drawing leaves it out by the text's words only

    def test_a_woman_gets_no_facial_hair_and_her_own_presets(self):
        ch, _ = choose(dict(TABLE, text={}, head=dict(HEAD, beard="yes", worn="up")), look="a woman of about 30",
                       sheet_colour=WOMAN)
        self.assertEqual(ch["gender"], "woman")
        self.assertEqual(sorted(ch["clear_slots"]), ["Beard", "Mustache"])
        self.assertEqual(ch["scores"]["preset_pool"], ["Alda", "Cara"])
        h = ch["grooms"]["Hair"]
        self.assertTrue("Updo" in h or "SweptUp" in h, h)               # worn up: an up-do, whatever its length
        self.assertEqual(ch["scores"]["age_used"], 30.0)

    def test_bald_and_tied_back(self):
        ch, _ = choose(dict(TABLE, head=dict(HEAD, bald="yes")))
        self.assertIn(ch["grooms"]["Hair"].replace("WI_Hair_", ""), ("S_BaldingStubble", "S_HairLoss", "S_RecedeMessy"))
        ch, _ = choose(dict(TABLE, head=dict(HEAD, worn="back")))
        self.assertTrue(any(k in ch["grooms"]["Hair"] for k in ("Ponytail", "PulledBack", "SlickBack")))

    def test_stubble_and_a_large_moustache(self):
        ch, _ = choose(dict(TABLE, head=dict(HEAD, beard_kind="stubble", moustache_kind="large")))
        self.assertIn("Stubble", ch["grooms"]["Beard"])
        self.assertTrue(ch["grooms"]["Mustache"].replace("WI_Mustache_", "").startswith("L_"))

    def test_build_goes_into_the_body(self):
        heavy, _ = choose(dict(TABLE, fig={"build": str(list(CM.BUILDS).index("very heavily built"))}))
        slim, _ = choose(dict(TABLE, fig={"build": str(list(CM.BUILDS).index("slim"))}))
        self.assertEqual(heavy["scores"]["build"], "very heavily built")
        for k in ("Chest", "Waist", "Neck", "Fat"):
            self.assertGreater(heavy["body"][k], slim["body"][k])

    def test_groom_names_read_as_words(self):
        self.assertEqual(CM.words("Hair", "S_SlickBack"), "short slick back hair")
        self.assertEqual(CM.words("Hair", "S_UpdoBuns"), "short up-do buns hair")
        self.assertEqual(CM.words("Mustache", "L_Handlebar"), "long handlebar moustache")

    def test_sheet_crops(self):
        head, fig = CM.heads(Image.new("RGB", (1000, 600)))
        self.assertEqual(fig.size, (500, 600))                         # the figure: the right half
        self.assertLess(head.size[0], 500)                              # the head: inside the left half
        self.assertLess(head.size[1], 600)


# ---------------------------------------------------------------------------------------------------------------
def face_pick(pool, answer, ages=None, keep=7, versus=None):
    d = Path(tempfile.mkdtemp(dir=TMP))
    port = d / "portraits"
    port.mkdir()
    for n in pool:
        Image.new("RGB", (160, 90), MAN).save(port / f"{n}.png")
    Image.new("RGB", (80, 100), MAN).save(d / "sheet_head.png")
    cp = d / "choice.json"
    cp.write_text(json.dumps({"base": pool[0], "scores": {"preset_pool": pool, "compare_keep": keep, "age_used": 40}}))

    def judge(cmd, input=None, **k):
        out = []
        for ln in (input or "").splitlines():
            q = json.loads(ln)
            if "age" in q["ask"]:
                a = (ages or {}).get(q["id"])
                out.append(json.dumps({"id": q["id"], "answers": {"age": "" if a is None else f"about {a}"}}))
            else:
                x, y = q["id"].split("|")
                a = answer(x, y)
                if a is not None:
                    out.append(json.dumps({"id": q["id"], "answers": {"pick": a}}))
        return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(out) + "\n", stderr="nothing")
    argv = ["face_pick.py", str(cp), "--portraits", str(port)]
    res = d / "versus.json"
    if versus:
        argv += ["--versus", *versus, "--result", str(res)]
    with mock.patch.object(sys, "argv", argv), mock.patch.object(FP.subprocess, "run", judge), \
            contextlib.redirect_stdout(io.StringIO()):
        FP.main()
    return json.loads(cp.read_text()), (json.loads(res.read_text()) if versus else None)


class FacePick(unittest.TestCase):
    def test_the_face_that_wins_wherever_it_is_shown_is_picked(self):
        ch, _ = face_pick(["Aa", "Bb", "Cc"], lambda x, y: "A" if x == "Cc" else ("B" if y == "Cc" else "A"))
        cmp_ = ch["scores"]["preset_compare"]
        self.assertEqual(ch["base"], "Cc")
        self.assertEqual(cmp_["wins"]["Cc"], 4)                       # two others, both orders
        self.assertEqual(len(cmp_["answers"]), 6)                     # every ordered pair was asked

    def test_a_side_bias_decides_nothing_and_the_tie_goes_to_the_first(self):
        ch, _ = face_pick(["Aa", "Bb", "Cc"], lambda x, y: "A")
        self.assertEqual(set(ch["scores"]["preset_compare"]["wins"].values()), {2})
        self.assertEqual(ch["base"], "Aa")                            # the pool's own order (head shape) breaks a tie

    def test_answers_are_read_loosely(self):
        ch, _ = face_pick(["Aa", "Bb"], lambda x, y: " b." if x == "Aa" else "a")
        self.assertEqual(ch["base"], "Bb")
        self.assertEqual(ch["scores"]["preset_compare"]["wins"], {"Aa": 0, "Bb": 2})

    def test_the_drawn_faces_age_narrows_the_pool_before_the_comparison(self):
        pool = [f"P{i}" for i in range(9)]
        ages = dict({f"P{i}": 20 + 5 * i for i in range(9)}, __drawing__=60)       # 20 .. 60; the drawing reads 60
        ch, _ = face_pick(pool, lambda x, y: "A", ages=ages, keep=4)
        kept = ch["scores"]["preset_compare"]["pool"]
        self.assertEqual(kept, ["P8", "P7", "P6", "P5"])              # the four nearest the drawn age
        self.assertEqual(ch["scores"]["drawing_age"], 60)
        self.assertEqual(ch["base"], "P8")

    def test_age_ties_keep_the_pools_order_and_a_missing_age_goes_last(self):
        pool = ["Aa", "Bb", "Cc", "Dd"]
        ages = {"Aa": 50, "Bb": 30, "Cc": 50, "Dd": None, "__drawing__": 40}
        ch, _ = face_pick(pool, lambda x, y: "A", ages=ages, keep=3)
        self.assertEqual(ch["scores"]["preset_compare"]["pool"], ["Aa", "Bb", "Cc"])

    def test_a_pool_within_the_limit_is_not_narrowed(self):
        ch, _ = face_pick(["Aa", "Bb"], lambda x, y: "A", ages={"__drawing__": 70, "Aa": 20, "Bb": 20}, keep=7)
        self.assertNotIn("drawing_age", ch["scores"])

    def test_versus_reports_wins_and_changes_nothing(self):
        ch, res = face_pick(["conformed", "unconformed"], lambda x, y: "A", versus=["conformed", "unconformed"])
        self.assertEqual(res["wins"], {"conformed": 1, "unconformed": 1})         # one each: not a win
        self.assertEqual(ch["base"], "conformed")                                 # the choice file's base is untouched
        self.assertNotIn("preset_compare", ch["scores"])
        _, res = face_pick(["conformed", "unconformed"], lambda x, y: "A" if x == "conformed" else "B",
                           versus=["conformed", "unconformed"])
        self.assertEqual(res["wins"]["conformed"], 2)

    def test_no_answers_is_an_error(self):
        with self.assertRaises(SystemExit):
            face_pick(["Aa", "Bb"], lambda x, y: None)

    def test_only_presets_with_a_portrait_take_part(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        (d / "portraits").mkdir()
        self.assertEqual(FP.head_crop(Image.new("RGB", (200, 100))).size, (80, 85))
        self.assertEqual(FP.tile(Image.new("RGB", (200, 100)), 50).size, (100, 50))


if __name__ == "__main__":
    unittest.main()
