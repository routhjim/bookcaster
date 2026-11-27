"""A built set's wide pictures are looked at (11-27): sets/wide_check.py measures each one's pixels and says of the
set "ok", "too dark" or "nearly empty"; story_keys keeps the measures and the verdict in the set's
place_build_report.json and says a failed set in one loud line, and stops nothing. Pictures made here only."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

A = Path(__file__).resolve().parents[1] / "assets"
for sub in ("sets", ""):
    sys.path.insert(0, str(A / sub))
import story_keys as SK  # noqa: E402
import wide_check as WC  # noqa: E402

W, H = 640, 360


def textured(lo=40, hi=200, seed=1):
    """An ordinary picture: patches of many brightnesses all over."""
    cells = np.random.default_rng(seed).integers(lo, hi, (H // 10, W // 10), dtype=np.uint8)
    return np.kron(cells, np.ones((10, 10), dtype=np.uint8))


def black():
    return np.zeros((H, W), dtype=np.uint8)


def night_lit():
    """A night picture: near black all round, a lit textured stretch over the lower middle of the frame."""
    a = np.full((H, W), 4, dtype=np.uint8)
    a[int(H * 0.4):, int(W * 0.1):int(W * 0.9)] = textured(30, 70)[int(H * 0.4):, int(W * 0.1):int(W * 0.9)]
    return a


def flat():
    return np.full((H, W), 128, dtype=np.uint8)


def save(d: Path, *pictures, name="a_set"):
    d.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(pictures):
        Image.fromarray(a).save(d / f"{name}_wide_{i}.png")
    return d


# ------------------------------------------------------------------ 1. one picture's measures
def test_a_black_picture_is_dark_and_empty(tmp_path):
    m = WC.measure(save(tmp_path, black()) / "a_set_wide_0.png")
    assert (m["median"], m["p90"], m["near_black"], m["seen"], m["flat_low"]) == (0.0, 0.0, 1.0, 0.0, 1.0)
    assert m["faults"] == ["dark", "empty"]


def test_a_night_picture_with_a_lit_stretch_passes(tmp_path):
    m = WC.measure(save(tmp_path, night_lit()) / "a_set_wide_0.png")
    assert m["median"] < WC.NEAR_BLACK                 # dark overall: no fault at night
    assert m["seen"] >= WC.SEEN_MIN and m["flat_low"] <= WC.FLAT_MAX
    assert m["faults"] == []


def test_a_flat_picture_is_empty_not_dark(tmp_path):
    m = WC.measure(save(tmp_path, flat()) / "a_set_wide_0.png")
    assert m["seen"] == 1.0 and m["flat_low"] == 1.0
    assert m["faults"] == ["empty"]


def test_an_ordinary_picture_passes(tmp_path):
    m = WC.measure(save(tmp_path, textured()) / "a_set_wide_0.png")
    assert m["seen"] == 1.0 and m["flat_low"] == 0.0 and m["faults"] == []


# ------------------------------------------------------------------ 2. the set's verdict
def test_the_verdicts(tmp_path):
    assert WC.check_set(save(tmp_path / "a", *[textured(seed=i) for i in range(4)]))["verdict"] == "ok"
    assert WC.check_set(save(tmp_path / "b", *[night_lit()] * 4))["verdict"] == "ok"
    assert WC.check_set(save(tmp_path / "c", *[black()] * 4))["verdict"] == "too dark, nearly empty"
    assert WC.check_set(save(tmp_path / "d", *[flat()] * 4))["verdict"] == "nearly empty"


def test_half_the_pictures_make_the_fault(tmp_path):
    one = WC.check_set(save(tmp_path / "a", black(), textured(), textured(seed=2), textured(seed=3)))
    assert one["ok"] and one["verdict"] == "ok"        # (one camera at a dark wall is no fault of the set)
    assert one["pictures"]["a_set_wide_0.png"]["faults"] == ["dark", "empty"]
    two = WC.check_set(save(tmp_path / "b", black(), black(), textured(), textured(seed=2)))
    assert not two["ok"] and two["faults"] == ["dark", "empty"]


def test_only_the_wide_pictures_are_read_and_none_is_said(tmp_path):
    d = save(tmp_path, textured())
    Image.fromarray(black()).save(d / "a_set_close.png")
    assert list(WC.check_set(d)["pictures"]) == ["a_set_wide_0.png"]
    r = WC.check_set(tmp_path / "nothing_here")
    assert (r["verdict"], r["ok"], r["pictures"]) == ("no pictures", False, {})
    assert WC.fault_line("the yard", r) == ""


# ------------------------------------------------------------------ 3. the loud line
def test_the_line_names_the_place_and_the_reason(tmp_path):
    assert WC.fault_line("the yard", WC.check_set(save(tmp_path / "a", textured(), textured(seed=2)))) == ""
    dark = WC.fault_line("the yard", WC.check_set(save(tmp_path / "b", black(), black(), night_lit(), night_lit())))
    assert dark == ("!!! the yard: the set's wide pictures are too dark (2 of 4 show under 35% of the frame over "
                    "near black, least 0%) and nearly empty (2 of 4 have over 45% of the lower half with nothing in "
                    "it, most 100%): shots staged in it will show it")
    bare = WC.fault_line("the yard", WC.check_set(save(tmp_path / "c", flat(), flat())))
    assert bare == ("!!! the yard: the set's wide pictures are nearly empty (2 of 2 have over 45% of the lower half "
                    "with nothing in it, most 100%): shots staged in it will show it")


# ------------------------------------------------------------------ 4. the story's set step keeps and says it
def story(tmp_path, monkeypatch, *pictures):
    monkeypatch.setattr(SK.C, "LIBRARY", tmp_path)
    st = SK.Story.__new__(SK.Story)
    st.places = {"the yard": {"slug": "the_yard", "set": "t_the_yard_01"}}
    d = tmp_path / "set" / "t_the_yard_01"
    save(d / "ue", *pictures, name="t_the_yard_01")
    (d / "place_build_report.json").write_text(json.dumps({"status": "ok", "unmet": []}))
    return st, d


def test_the_report_keeps_the_verdict_and_the_log_says_a_failed_set(tmp_path, monkeypatch, capsys):
    st, d = story(tmp_path, monkeypatch, black(), black(), black(), night_lit())
    res = st.wide_check("the yard")
    rep = json.loads((d / "place_build_report.json").read_text())
    assert rep["status"] == "ok" and rep["unmet"] == []                # (what was there is kept)
    assert rep["wide_check"] == res and res["verdict"] == "too dark, nearly empty"
    assert set(rep["wide_check"]["pictures"]["t_the_yard_01_wide_0.png"]) == {
        "median", "p90", "near_black", "seen", "flat_low", "faults"}
    out = [ln for ln in capsys.readouterr().out.splitlines() if "!!!" in ln]
    assert len(out) == 1 and out[0].startswith("[story ")
    assert "!!! the yard: the set's wide pictures are too dark (3 of 4" in out[0] and "nearly empty (3 of 4" in out[0]


def test_a_good_set_is_recorded_ok_and_nothing_is_said(tmp_path, monkeypatch, capsys):
    st, d = story(tmp_path, monkeypatch, *[textured(seed=i) for i in range(4)])
    assert st.wide_check("the yard")["ok"]
    assert json.loads((d / "place_build_report.json").read_text())["wide_check"]["verdict"] == "ok"
    assert capsys.readouterr().out == ""


def test_a_check_that_cannot_run_stops_nothing(tmp_path, monkeypatch, capsys):
    st, d = story(tmp_path, monkeypatch, textured())
    (d / "ue" / "t_the_yard_01_wide_1.png").write_text("not a picture")
    assert st.wide_check("the yard") == {}
    assert "wide_check" not in json.loads((d / "place_build_report.json").read_text())
    out = capsys.readouterr().out
    assert "the set's wide pictures were not checked" in out and "!!!" not in out
