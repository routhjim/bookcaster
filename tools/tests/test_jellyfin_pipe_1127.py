"""11-27: when the copier to the video server gives up part-way, the job says what the copier said and how far it got,
not only "Broken pipe". Made-up data only."""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tts_reader import jellyfin as J  # noqa: E402


class Gone:
    def __init__(self, said):
        self.stdout = io.BytesIO(said)
        self.stdin = self

    def write(self, chunk):
        raise BrokenPipeError(32, "Broken pipe")

    def close(self):
        pass

    def wait(self):
        return 1

    def communicate(self):
        return ("", None)


def test_a_copier_that_gives_up_is_reported_with_its_own_words(tmp_path, monkeypatch):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x" * 5000)
    monkeypatch.setattr(J, "config", lambda: {"SMB_SHARE": "//h/s"})
    monkeypatch.setattr(J, "remote_size", lambda cfg, dest: None)
    monkeypatch.setattr(J, "_sent", lambda: {})
    monkeypatch.setattr(J, "_smb", lambda cfg, commands, stdin=None: Gone(b"NT_STATUS_MADE_UP writing the file"))
    said = []
    assert J.send(src, "A Folder", "a.mp4", log=said.append) is False
    assert any("NT_STATUS_MADE_UP" in s and "stopped after 0 of" in s for s in said), said
