"""bookcaster tidy: only what a rule names is waste; a LoRA with no pick keeps everything."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_reader import tidy as T  # noqa: E402


def book(tmp: Path) -> Path:
    b = tmp / "a_book"
    vd = b / "work" / "video"
    for who, picked in (("orla", True), ("tam", False)):
        ld = vd / "charlora" / who / "lora"
        (ld / "cache").mkdir(parents=True)
        (ld / "cache" / "x.pt").write_bytes(b"c" * 10)
        for n in (100, 200, 300):
            (ld / f"step_00{n}.safetensors").write_bytes(b"s" * 100)
        (ld / "step_00300.pt").write_bytes(b"p" * 500)
        (ld / "train.log").write_text("log")
        if picked:
            (ld / "USE").write_text("step_00200.safetensors\n")
        prev = vd / "charlora" / who / "lora.prev"
        prev.mkdir()
        (prev / "step_00100.safetensors").write_bytes(b"s" * 100)
        (vd / "charlora" / who / "data").mkdir()
        (vd / "charlora" / who / "data" / "sheet.jpg").write_bytes(b"j")
    (vd / "01_ch" / "mannequin").mkdir(parents=True)
    (vd / "01_ch" / "shots").mkdir()
    (vd / "01_ch" / "plan.json").write_text("{}")
    (vd / "old-1101").mkdir()
    (vd / "charlora_old_1101").mkdir()
    (vd / "older_notes").mkdir()
    for i, n in enumerate(("before-1", "before-2")):
        s = vd / "engine" / "the_hall" / "history" / n
        s.mkdir(parents=True)
        import os
        os.utime(s, (1000 + i, 1000 + i))
    (b / "out").mkdir()
    (b / "out" / "a.mp4").write_bytes(b"v")
    (b / "book.txt").write_text("t")
    return b


def test_only_named_leftovers_are_waste(tmp_path):
    b = book(tmp_path)
    got = {str(p.relative_to(b)) for p, _, _ in T.waste(b)}
    v = "work/video/"
    assert got == {v + "charlora/orla/lora/step_00100.safetensors", v + "charlora/orla/lora/step_00300.safetensors",
                   v + "charlora/orla/lora/step_00300.pt", v + "charlora/orla/lora/cache", v + "charlora/orla/lora.prev",
                   v + "01_ch/mannequin", v + "old-1101", v + "charlora_old_1101", v + "engine/the_hall/history/before-1"}


def test_a_lora_with_no_pick_keeps_everything(tmp_path):
    b = book(tmp_path)
    assert T.lora_waste(b / "work/video/charlora/tam") == []
    (b / "work/video/charlora/tam/lora/USE").write_text("step_00900.safetensors")      # names a file not there
    assert T.lora_waste(b / "work/video/charlora/tam") == []


def test_tidy_lists_then_removes_and_the_essentials_stay(tmp_path):
    b = book(tmp_path)
    said = []
    n = T.tidy(b, log=said.append)
    assert n > 0 and (b / "work/video/charlora/orla/lora.prev").exists() and "add --do" in said[-1]
    T.tidy(b, do=True, kinds=T.AUTO, log=said.append)
    assert not (b / "work/video/charlora/orla/lora.prev").exists() and (b / "work/video/old-1101").exists()
    T.tidy(b, do=True, log=said.append)
    left = {str(p.relative_to(b)) for p in b.rglob("*") if p.is_file()}
    assert "work/video/charlora/orla/lora/step_00200.safetensors" in left and "work/video/charlora/orla/lora/USE" in left
    assert "work/video/charlora/orla/data/sheet.jpg" in left and "out/a.mp4" in left and "work/video/01_ch/plan.json" in left
    assert sum("tam/" in x for x in left) == 8 and T.waste(b) == []
    assert (b / "work/video/older_notes").exists() and (b / "work/video/engine/the_hall/history/before-2").exists()


def test_archive_copies_the_book_without_its_waste(tmp_path):
    b = book(tmp_path)
    assert T.archive(b, tmp_path / "disk", log=lambda *_: None) == 0
    c = tmp_path / "disk" / "a_book"
    assert (c / "work/video/charlora/orla/lora/step_00200.safetensors").exists() and (c / "out/a.mp4").exists()
    assert not (c / "work/video/charlora/orla/lora.prev").exists() and not (c / "work/video/01_ch/mannequin").exists()
    assert (c / "work/video/charlora/tam/lora.prev").exists()
