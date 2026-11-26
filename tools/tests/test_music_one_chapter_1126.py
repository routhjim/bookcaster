"""New music for one chapter (11-26): `bookcaster make <book> --music-only --chapter N` drops that chapter's music
plan and cues so it is planned afresh, leaves the others, and a chapter that had sound effects is known as such."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_reader import bookcaster as BC  # noqa: E402


def book(td):
    b = Path(td) / "orla_and_tam"
    (b / "chapters").mkdir(parents=True)
    for n in (1, 2, 12):
        (b / "chapters" / f"{n:02d}_chapter_{n}.txt").write_text("Orla met Tam.")
        d = b / "work" / "score" / f"{n:02d}_chapter_{n}"
        d.mkdir(parents=True)
        (d / "plan.json").write_text("[]")
        (d / "cue01.wav").write_bytes(b"x")
        (d / "style.txt").write_text("strings")
    (b / "work" / "sfx" / "02_chapter_2").mkdir(parents=True)
    (b / "work" / "sfx" / "02_chapter_2" / "plan.json").write_text("{}")
    return b


def test_a_chapter_is_named_by_its_number_or_its_stem():
    with tempfile.TemporaryDirectory() as td:
        fs = BC.chapter_files(book(td))
        assert [f.stem for f in BC.chapter_named(fs, "2")] == ["02_chapter_2"]
        assert [f.stem for f in BC.chapter_named(fs, "02")] == ["02_chapter_2"]
        assert [f.stem for f in BC.chapter_named(fs, 12)] == ["12_chapter_12"]
        assert [f.stem for f in BC.chapter_named(fs, "01_chapter_1")] == ["01_chapter_1"]
        assert BC.chapter_named(fs, "7") == [] and BC.chapter_named(fs, "1") != BC.chapter_named(fs, "12")


def test_only_the_named_chapters_plan_and_cues_go():
    with tempfile.TemporaryDirectory() as td:
        b = book(td)
        f2 = BC.chapter_named(BC.chapter_files(b), "2")[0]
        gone = BC.drop_score(b, f2)
        assert sorted(q.name for q in gone) == ["cue01.wav", "plan.json"]
        d2, d1 = b / "work" / "score" / "02_chapter_2", b / "work" / "score" / "01_chapter_1"
        assert not (d2 / "plan.json").exists() and not (d2 / "cue01.wav").exists() and (d2 / "style.txt").exists()
        assert (d1 / "plan.json").exists() and (d1 / "cue01.wav").exists()
        assert BC.drop_score(b, f2) == []


def test_a_chapter_that_had_effects_is_known():
    with tempfile.TemporaryDirectory() as td:
        b = book(td)
        fs = BC.chapter_files(b)
        assert [BC.had_sfx(b, f) for f in fs] == [False, True, False]


def test_the_job_is_started_for_the_one_chapter():
    src = Path(BC.__file__).read_text()
    assert 'cmd += ["--chapter", str(chapter)]' in src and "chapter=a.chapter if a.music_only else None" in src
    assert "--chapter goes with --music-only" in src
