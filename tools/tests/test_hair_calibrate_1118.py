"""cast_char.calibrate_hair (11-18: "grey hair" built at the table's whiteness rendered as a white cap): a light colour
word with no measured whiteness in the library map is measured on the built character (close portraits at a grid of
whiteness values, hair_calibrate.py names them, the value named with the drawn head's word is set and recorded); a
word already in the map renders nothing; no value named so keeps the table's value. CPU only: the engine jobs and the
line call are faked."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

ASSETS = Path(__file__).resolve().parents[1] / "assets"
TMP = tempfile.mkdtemp(prefix="hair_cal_test_")
os.environ.setdefault("ASSETS_SCHED_DIR", os.path.join(TMP, "sched"))
os.environ.setdefault("ASSETS_LIBRARY", os.path.join(TMP, "lib"))
os.environ.setdefault("ASSETS_LOGS", os.path.join(TMP, "logs"))
sys.path.insert(0, str(ASSETS))
sys.path.insert(0, str(ASSETS / "img3d"))

import cast_char as CC  # noqa: E402


def choice(word, wht, mel=0.3, red=0.05):
    b = {"Melanin": mel, "Redness": red, "Whiteness": wht}
    return {"base": "P", "params": {f"Bindings/{d}/": dict(b) for d in ("Hair", "Eyebrows", "Beards", "Mustaches")},
            "scores": {"hair_colour": word}}


class Fake:
    """The engine jobs (colour sets and close shots, in order) and the calibrate call; the vision model's pick is
    PICK (None: no value named with the target word)."""

    def __init__(self, monkeypatch, pick):
        self.jobs, self.cal = [], []
        self.pick = pick
        monkeypatch.setattr(CC, "ue_job", self.ue_job)
        monkeypatch.setattr(CC, "line", self.line)
        monkeypatch.setattr(CC, "log", lambda m: self.logs.append(m))
        self.logs = []

    def ue_job(self, work, name, script, job, timeout=1800):
        self.jobs.append((name, Path(script).name, job))
        if Path(script).name == "char_render.py":
            for sh in job["shots"]:
                Path(job["out"], f"{sh['name']}.png").write_bytes(b"png")
        return {"ok": True, "set": {"MI_WI_Hair_x": {}}}

    def line(self, args, log_file=None):
        self.cal.append(args)
        word = args[args.index("--word") + 1]
        vals = args[args.index("--values") + 1:args.index("--out")]
        out = Path(args[args.index("--out") + 1])
        d = Path(args[args.index("portraits") + 1])
        assert all((d / f"{word}_{v}.png").exists() for v in vals), "a portrait the calibrate step reads is missing"
        out.write_text(json.dumps({word: {"whiteness": self.pick, "target_answer": word,
                                          "answers": {v: "white" for v in vals}}}))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    def renders(self):
        return [j for j in self.jobs if j[1] == "char_render.py"]

    def colour_sets(self):
        return [j for j in self.jobs if j[1] == "char_groom_colour.py"]


def test_uncalibrated_light_word_is_measured(monkeypatch, tmp_path):
    f = Fake(monkeypatch, pick=0.3)
    ch = choice("grey", 0.75)
    (tmp_path / "choice.json").write_text(json.dumps(ch))
    mp = tmp_path / "lib" / "hair_colour.json"
    mp.parent.mkdir()
    mp.write_text(json.dumps({"white": {"whiteness": 1.0}}))
    rec = CC.calibrate_hair(tmp_path, ch, "MHC_X", "/Game/MetaHumans/MHC_X/BP_MHC_X", mp, "X")
    grid = sorted(set(CC.HAIR_GRID) | {0.75})
    # one colour set and one close shot of the assembled Blueprint per grid value, in that order
    shots = f.renders()
    assert len(shots) == len(grid)
    assert all(j[2]["characters"]["a"]["path"].endswith("BP_MHC_X") and j[2]["shots"][0]["frame"] == "close"
               for j in shots)
    assert [j[2]["whiteness"] for j in f.colour_sets()[:len(grid)]] == grid
    # the calibrate call: every grid value, the drawn sheet head as the target
    assert len(f.cal) == 1
    args = f.cal[0]
    assert args[args.index("--sheet") + 1] == str(tmp_path / "sheet_head.png")
    assert args[args.index("--word") + 1] == "grey"
    # the last colour set is the picked value; the record, the choice and the library map hold it
    assert f.colour_sets()[-1][2]["whiteness"] == 0.3
    assert len(f.colour_sets()) == len(grid) + 1
    assert rec["whiteness"] == 0.3 and not rec.get("no_hit")
    assert all(b["Whiteness"] == 0.3 for b in ch["params"].values())
    cj = json.loads((tmp_path / "choice.json").read_text())
    assert cj["params"]["Bindings/Hair/"]["Whiteness"] == 0.3
    assert cj["scores"]["hair_measured"]["whiteness"] == 0.3
    m = json.loads(mp.read_text())
    assert m["grey"]["whiteness"] == 0.3 and m["white"] == {"whiteness": 1.0}   # merged, the other word kept
    assert json.loads((tmp_path / "hair_calibrate.json").read_text())["whiteness"] == 0.3


def test_word_in_map_renders_nothing(monkeypatch, tmp_path):
    f = Fake(monkeypatch, pick=0.3)
    ch = choice("grey", 0.45)
    mp = tmp_path / "hair_colour.json"
    mp.write_text(json.dumps({"grey": {"whiteness": 0.45}}))
    rec = CC.calibrate_hair(tmp_path, ch, "MHC_X", "/Game/BP", mp, "X")
    assert f.renders() == [] and f.cal == [] and f.colour_sets() == []    # choose_mh already built it at 0.45
    assert rec["in_map"] and rec["whiteness"] == 0.45


def test_dark_word_is_not_searched(monkeypatch, tmp_path):
    f = Fake(monkeypatch, pick=0.3)
    rec = CC.calibrate_hair(tmp_path, choice("dark brown", 0.0, 0.65, 0.35), "MHC_X", "/Game/BP",
                            tmp_path / "hair_colour.json", "X")
    assert f.jobs == [] and f.cal == [] and rec.get("skipped")


def test_no_hit_keeps_the_table_value(monkeypatch, tmp_path):
    f = Fake(monkeypatch, pick=None)
    ch = choice("grey", 0.75)
    (tmp_path / "choice.json").write_text(json.dumps(ch))
    mp = tmp_path / "hair_colour.json"
    rec = CC.calibrate_hair(tmp_path, ch, "MHC_X", "/Game/BP", mp, "X")
    assert len(f.renders()) == len(set(CC.HAIR_GRID) | {0.75})
    assert rec.get("no_hit") and rec["whiteness"] == 0.75
    # the grooms set back to the table's value (they held the last render's), nothing else changed
    assert f.colour_sets()[-1][2]["whiteness"] == 0.75
    assert all(b["Whiteness"] == 0.75 for b in ch["params"].values())
    assert "hair_measured" not in json.loads((tmp_path / "choice.json").read_text())["scores"]
    assert not mp.exists()                    # no null entry in the library map
    assert any("kept" in m and "grey" in m for m in f.logs)


def test_colour_table_read_from_source():
    t = CC.hair_colours()
    assert t["grey"][2] > 0 and t["black"][2] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
