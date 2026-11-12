"""The scene-level look pass (tools/assets/redraw/scenelook.py): a person's and a piece's look judged across a
scene's keys, from fake embeddings and synthetic colour readings (no model, no pictures)."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "redraw"))
import scenelook as SL  # noqa: E402

E_A, E_B, E_C = [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]


def near(v, eps, i=3):
    w = list(v)
    w[i] += eps
    return w


def person(key, head, top=0.2, deg=10.0, worn=None, covered=False):
    return {"key": key, "box": [0.3, 0.1, 0.6, 0.9], "head_to_camera_deg": deg, "worn": worn or [],
            "face_covered": covered, "joints2d": {"head_top": [0.5, top], "chin": [0.5, top + head]}}


def lab(L, a=2.0, b=8.0, px=8000):
    return {"L": L, "a": a, "b": b, "px": px}


def scene(specs):
    """specs: [(id, [(key, head, emb, hair L, wear L, facial hair, hair word, age, extra)])] -> items, checked, feats"""
    items, checked, feats = [], {}, {}
    for iid, ppl in specs:
        it = {"id": iid, "people": [], "things": []}
        ck = {"hair": {}, "wear": {}}
        fx = {"people": {}, "things": {}}
        for key, head, emb, hl, wl, fh, hw, age, extra in ppl:
            it["people"].append(person(key, head, **(extra.get("person") or {})))
            if hl is not None:
                ck["hair"][key] = {"redraw": lab(hl, px=extra.get("hair_px", 6000)), "engine": lab(20.0)}
            if wl is not None:
                ck["wear"][key] = {"redraw": lab(wl, *extra.get("wear_ab", (2.0, 8.0))), "engine": lab(15.0)}
            fx["people"][key] = {"emb": emb, "facial_hair": fh, "hair_colour": hw, "age": age,
                                 "worn": extra.get("worn") or {}}
        items.append(it)
        checked[iid], feats[iid] = ck, fx
    return items, checked, feats


def row(key, head, emb, hl=22.0, wl=14.0, fh="none", hw="black", age="40", **extra):
    return (key, head, emb, hl, wl, fh, hw, age, extra)


class Helpers(unittest.TestCase):
    def test_weighted_median_follows_the_weight(self):
        self.assertEqual(SL.wmedian([1, 2, 30], [1, 1, 1]), 2)
        self.assertEqual(SL.wmedian([1, 2, 30], [0.1, 0.1, 5]), 30)
        self.assertIsNone(SL.wmedian([1, 2], [0, 0]))

    def test_head_weight_and_size_class(self):
        self.assertEqual(SL.head_weight(SL.REF_HEAD), 1.0)
        self.assertLess(SL.head_weight(SL.REF_HEAD / 2), 0.3)
        self.assertEqual(SL.size_class(SL.FACE_BAR), "face")
        self.assertEqual(SL.size_class(SL.FACE_BAR - 0.01), "small")
        self.assertEqual(SL.size_class(SL.SLIVER_BAR - 0.01), "sliver")
        self.assertEqual(SL.size_class(None), "sliver")

    def test_answers_are_read_as_classes(self):
        self.assertEqual(SL.hair_class("a curly moustache and short full beard"), "both")
        self.assertEqual(SL.hair_class("Moustache."), "moustache")
        self.assertEqual(SL.hair_class("none"), "none")
        self.assertIsNone(SL.hair_class("cannot tell"))
        self.assertEqual(SL.hair_word("Dark brown."), ("dark", 1))
        self.assertEqual(SL.hair_word("black with grey streaks"), ("grey", 1))
        self.assertTrue(SL.hair_differs(SL.hair_word("black"), SL.hair_word("grey")))
        self.assertTrue(SL.hair_differs(SL.hair_word("black"), SL.hair_word("brown")))
        self.assertFalse(SL.hair_differs(SL.hair_word("black"), SL.hair_word("dark brown")))
        self.assertEqual(SL.age_of("about 45"), 45.0)
        self.assertIsNone(SL.age_of("cannot tell"))
        self.assertIs(SL.yes_no("Yes."), True)
        self.assertIsNone(SL.yes_no("cannot tell"))


class People(unittest.TestCase):
    def base(self, odd):
        """Person a in five keys, person b in all; `odd` replaces a's row in the third key."""
        specs = []
        for n in range(5):
            ra = row("a", 0.30, near(E_A, 0.02 * n))
            if n == 2:
                ra = odd
            specs.append((f"k{n}", [ra, row("b", 0.25, near(E_B, 0.02 * n), hl=45.0, wl=40.0, fh="moustache",
                                               hw="brown", age="45")]))
        return scene(specs)

    def judge(self, odd, **kw):
        items, checked, feats = self.base(odd)
        return SL.judge(items, checked, feats, **kw)

    def test_a_consistent_scene_flags_nothing_and_counts_no_changes(self):
        r = self.judge(row("a", 0.30, near(E_A, 0.04)))
        self.assertEqual(r["flagged"], {})
        self.assertEqual(r["look_changes"], 0)
        self.assertEqual(r["cuts_n"], 4)

    def test_hair_lighter_in_one_key_is_that_keys_outlier(self):
        r = self.judge(row("a", 0.30, near(E_A, 0.04), hl=22.0 + SL.HAIR_L + 5))
        self.assertEqual(r["flagged"], {"k2": {"a": ["hair"]}})
        self.assertEqual(r["look_changes"], 2)                  # into the odd key and out of it

    def test_hair_word_catches_a_small_head_the_lightness_check_skips(self):
        # a small head (under the face bar), hair region too small for the Lab reading: the word still differs
        odd = row("a", SL.FACE_BAR - 0.03, None, hl=None, hw="grey")
        r = self.judge(odd)
        self.assertEqual(r["flagged"]["k2"]["a"], ["hair"])

    def test_coat_colour_off_the_persons_median(self):
        r = self.judge(row("a", 0.30, near(E_A, 0.04), wl=14.0 + SL.WEAR_L + 4))
        self.assertIn("clothes colour", r["flagged"]["k2"]["a"])

    def test_invented_moustache_on_a_small_figure_is_flagged_against_the_fact(self):
        odd = row("a", SL.FACE_BAR - 0.04, None, fh="moustache")
        r = self.judge(odd, facts={"a": {"facial_hair": "none"}})
        self.assertEqual(r["flagged"]["k2"]["a"], ["facial hair"])
        self.assertEqual(r["people"]["a"]["median"]["facial_hair"], "none")

    def test_facial_hair_without_a_fact_uses_the_scenes_majority(self):
        r = self.judge(row("a", 0.30, near(E_A, 0.04), fh="beard"))
        self.assertEqual(r["flagged"]["k2"]["a"], ["facial hair"])

    def test_a_sliver_is_judged_on_clothes_only(self):
        odd = row("a", SL.SLIVER_BAR - 0.01, near(E_C, 0.0), hl=60.0, fh="moustache", hw="white", age="80",
                  wl=14.0 + SL.WEAR_L + 4)
        r = self.judge(odd, facts={"a": {"facial_hair": "none"}})
        self.assertEqual(r["flagged"]["k2"]["a"], ["clothes colour"])

    def test_face_nearer_another_person_is_a_face_outlier(self):
        r = self.judge(row("a", 0.30, near(E_B, 0.01)))          # a drawn as b
        self.assertIn("face", r["flagged"]["k2"]["a"])
        self.assertEqual([x for x in r["rows"] if x["id"] == "k2" and x["key"] == "a"][0]["dev"]["face_nearer"], "b")

    def test_a_different_face_far_from_the_persons_own_is_an_outlier(self):
        r = self.judge(row("a", 0.30, near(E_C, 0.0)))
        self.assertIn("face", r["flagged"]["k2"]["a"])

    def test_small_heads_are_never_judged_on_face_identity(self):
        r = self.judge(row("a", SL.FACE_BAR - 0.02, near(E_C, 0.0)))
        self.assertNotIn("k2", r["flagged"])

    def test_large_faces_define_the_look(self):
        # three small keys read grey, two large keys read black: the large ones are the median, the small are odd
        specs = [(f"k{n}", [row("a", 0.07 if n < 3 else 0.35, None if n < 3 else near(E_A, 0.01 * n), hl=None,
                                 hw="grey" if n < 3 else "black")]) for n in range(5)]
        items, checked, feats = scene(specs)
        r = SL.judge(items, checked, feats)
        self.assertEqual(r["people"]["a"]["median"]["hair_word"], ["dark", 0] if isinstance(
            r["people"]["a"]["median"]["hair_word"], list) else ("dark", 0))
        self.assertEqual(sorted(r["flagged"]), ["k0", "k1", "k2"])

    def test_apparent_age_outlier(self):
        r = self.judge(row("a", 0.30, near(E_A, 0.04), age=str(int(40 + SL.AGE_TOL + 5))))
        self.assertEqual(r["flagged"]["k2"]["a"], ["age"])

    def test_staged_worn_item_not_seen_is_flagged(self):
        veil = "a net veil over the face"
        odd = row("a", 0.30, near(E_A, 0.04), person={"worn": [veil]}, worn={veil: "no"})
        r = self.judge(odd)
        self.assertEqual(r["flagged"]["k2"]["a"], ["worn"])
        seen = row("a", 0.30, near(E_A, 0.04), person={"worn": [veil]}, worn={veil: "yes"})
        self.assertEqual(self.judge(seen)["flagged"], {})

    def test_clothes_that_read_as_another_persons(self):
        odd = row("a", 0.30, near(E_A, 0.04), wl=40.0)           # a drawn in b's coat lightness
        r = self.judge(odd)
        self.assertIn("clothes colour", r["flagged"]["k2"]["a"])
        self.assertEqual([x for x in r["rows"] if x["id"] == "k2" and x["key"] == "a"][0]["dev"]["wear_reads_as"], "b")

    def test_anchor_is_the_largest_clean_face(self):
        specs = [("k0", [row("a", 0.20, near(E_A, 0.00))]), ("k1", [row("a", 0.40, near(E_A, 0.01), hl=60.0)]),
                 ("k2", [row("a", 0.35, near(E_A, 0.02))]), ("k3", [row("a", 0.25, near(E_A, 0.03))]),
                 ("k4", [row("a", 0.38, near(E_A, 0.02))])]
        items, checked, feats = scene(specs)
        r = SL.judge(items, checked, feats, fails={"k4": {"a": ["neck"]}})
        self.assertEqual(r["people"]["a"]["anchor"], "k2")      # k1 is the hair outlier, k4 failed its own checks

    def test_look_changes_are_counted_per_cut(self):
        specs = [("k0", [row("a", 0.3, near(E_A, 0.0), hw="black")]), ("k1", [row("a", 0.3, near(E_A, 0.01), hw="grey")]),
                 ("k2", [row("a", 0.3, near(E_A, 0.02), hw="grey", fh="moustache")])]
        items, checked, feats = scene(specs)
        cuts = SL.judge(items, checked, feats)["cuts"]
        self.assertEqual(cuts[0]["changes"], [["a", "hair"]])
        self.assertEqual(cuts[1]["changes"], [["a", "facial hair"]])


class Things(unittest.TestCase):
    def feats(self, drawn):
        f = {}
        for n in range(3):
            f[f"k{n}"] = {"things": {
                "armchair": {"emb_engine": near(E_A, 0.01 * n), "emb_redraw": near(E_A, 0.02), "fill": 0.6, "area": 0.08},
                "wingback": {"emb_engine": near(E_B, 0.01 * n), "emb_redraw": near(E_B, 0.02), "fill": 0.6, "area": 0.08}}}
        f["k1"]["things"]["wingback"]["emb_redraw"] = drawn
        return f

    def test_a_piece_drawn_as_another_piece(self):
        items = [{"id": f"k{n}", "people": [], "things": []} for n in range(3)]
        rows = SL.judge_things(items, self.feats(near(E_A, 0.02)))
        bad = [r for r in rows if r["flag"]]
        self.assertEqual([(r["id"], r["thing"], r["flag"]) for r in bad], [("k1", "wingback", "armchair")])

    def test_a_piece_drawn_as_itself_passes_and_small_or_hidden_ones_are_not_judged(self):
        items = [{"id": f"k{n}", "people": [], "things": []} for n in range(3)]
        f = self.feats(near(E_B, 0.03))
        self.assertEqual([r for r in SL.judge_things(items, f) if r["flag"]], [])
        f["k1"]["things"]["wingback"].update(emb_redraw=near(E_A, 0.02), fill=SL.THING_FILL - 0.1)
        self.assertEqual([r for r in SL.judge_things(items, f) if r["flag"]], [])


class Repeats(unittest.TestCase):
    def test_two_neighbours_on_the_same_picture(self):
        specs = [(f"k{n}", [row("a", 0.3, near(E_A, 0.01 * n))]) for n in range(3)]
        items, checked, feats = scene(specs)
        ramp = [i / 100 for i in range(64)]
        feats["k0"]["thumb"], feats["k1"]["thumb"] = ramp, [v + 0.001 for v in ramp]
        feats["k2"]["thumb"] = list(reversed(ramp))
        r = SL.judge(items, checked, feats)
        self.assertEqual([x["cut"] for x in r["repeats"]], [["k0", "k1"]])
        self.assertIn("repeat of the previous key", r["flagged"]["k1"]["set"])
        items[1]["hold"] = True                                   # a stated hold is allowed
        self.assertEqual(SL.judge(items, checked, feats)["repeats"], [])


class Words(unittest.TestCase):
    def test_fix_words_name_each_flag_once_in_general_words(self):
        w = SL.fix_words("the man", "his", ["hair", "facial hair", "face", "age", "worn"],
                         {"worn_missing": ["a grey scarf"]}, "none")
        self.assertEqual(len(w), 5)
        self.assertIn("clean-shaven", w[1])
        self.assertIn("a grey scarf", w[4])
        self.assertTrue(all("reference picture of him" in x for x in w[:4]))

    def test_compare_reports_reproduced_review_rows(self):
        look = {"flagged": {"r1": {"a": ["hair"]}}, "cuts": [{"cut": ["r1", "r2"], "changes": [["b", "clothes colour"]]}]}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "review.csv"
            p.write_text("shot_no,shot_id,category,what,severity,in_verdicts\n"
                         "1,shot-1,look,hair lighter than before,2,no\n"
                         "2,shot-2,look,coat changed,2,no\n"
                         "3,shot-3,look,hair darker,1,no\n"
                         "3,shot-3,framing,head cut,1,no\n")
            s = SL.cmd_compare(look, p, {"r1": "shot-1", "r2": "shot-2"})
        self.assertIn("reproduced 2 of 3 look rows", s)


if __name__ == "__main__":
    unittest.main()
