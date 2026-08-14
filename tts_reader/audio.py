"""Audio helpers: encode PCM/WAV audio to MP3 without needing ffmpeg.

We use :mod:`lameenc` (pure LAME bindings) so the whole pipeline stays local
and dependency-light — no system ffmpeg install required. Chapter markers are
written as ID3v2 ``CHAP``/``CTOC`` frames via :mod:`mutagen`, so a whole book
becomes a single MP3 you can jump around inside a chapter-aware player.
"""

from __future__ import annotations

import wave
from pathlib import Path

import lameenc


def _make_encoder(sample_rate: int, channels: int, bitrate: int, quality: int) -> "lameenc.Encoder":
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(bitrate)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(channels)
    encoder.set_quality(quality)
    if hasattr(encoder, "silence"):
        encoder.silence()
    return encoder


def wav_to_mp3(wav_path: str | Path, mp3_path: str | Path, bitrate: int = 128, quality: int = 2) -> None:
    """Convert a mono/stereo 16-bit PCM WAV file into an MP3 file.

    Parameters
    ----------
    wav_path: source WAV file (16-bit PCM).
    mp3_path: destination MP3 file.
    bitrate:  target MP3 bitrate in kbps (e.g. 96, 128, 192, 256).
    quality:  LAME quality 0 (best/slowest) .. 9 (worst/fastest).
    """
    wav_path = Path(wav_path)
    mp3_path = Path(mp3_path)

    with wave.open(str(wav_path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if sample_width != 2:
            raise ValueError(
                f"Expected 16-bit PCM audio, got {sample_width * 8}-bit. "
                "Piper produces 16-bit output, so this usually indicates a bad WAV."
            )
        pcm = wav.readframes(wav.getnframes())

    encoder = _make_encoder(sample_rate, channels, bitrate, quality)
    mp3_data = encoder.encode(pcm)
    mp3_data += encoder.flush()

    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.write_bytes(bytes(mp3_data))


def pcm_to_mp3(
    pcm: bytes,
    mp3_path: str | Path,
    sample_rate: int,
    bitrate: int = 128,
    channels: int = 1,
    quality: int = 2,
) -> None:
    """Encode a buffer of 16-bit PCM audio to an MP3 file."""
    mp3_path = Path(mp3_path)
    encoder = _make_encoder(sample_rate, channels, bitrate, quality)
    data = encoder.encode(pcm) + encoder.flush()
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    mp3_path.write_bytes(bytes(data))


class ChapteredMp3Writer:
    """Stream chapter audio into one MP3 and record chapter time offsets.

    PCM is encoded incrementally (one chapter at a time, so a whole book never
    has to sit in memory at once). On :meth:`close` the accumulated chapter
    boundaries are written as ID3 ``CHAP``/``CTOC`` frames.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int,
        bitrate: int = 128,
        channels: int = 1,
        quality: int = 2,
        album: str | None = None,
    ):
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.channels = channels
        self.album = album
        self._encoder = _make_encoder(sample_rate, channels, bitrate, quality)
        self._frames = 0  # per-channel sample frames written so far
        self._chapters: list[tuple[str, int, int]] = []  # (title, start_ms, end_ms)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")

    def _current_ms(self) -> int:
        return int(self._frames * 1000 / self.sample_rate)

    def add_chapter(self, title: str, pcm: bytes) -> None:
        start_ms = self._current_ms()
        self._fh.write(bytes(self._encoder.encode(pcm)))
        self._frames += len(pcm) // (2 * self.channels)
        end_ms = self._current_ms()
        self._chapters.append((title, start_ms, end_ms))

    def close(self) -> None:
        self._fh.write(bytes(self._encoder.flush()))
        self._fh.close()
        write_id3_chapters(self.path, self._chapters, album=self.album)

    def __enter__(self) -> "ChapteredMp3Writer":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def write_id3_chapters(
    path: str | Path,
    chapters: list[tuple[str, int, int]],
    album: str | None = None,
) -> None:
    """Attach ID3v2 chapter markers (CHAP + a top-level CTOC) to an MP3."""
    from mutagen.id3 import CHAP, CTOC, CTOCFlags, ID3, TALB, TIT2
    from mutagen.id3 import ID3NoHeaderError

    path = Path(path)
    try:
        tags = ID3(str(path))
    except ID3NoHeaderError:
        tags = ID3()

    if album:
        tags.add(TALB(encoding=3, text=[album]))

    child_ids = []
    for i, (title, start_ms, end_ms) in enumerate(chapters):
        eid = f"chp{i:04d}"
        child_ids.append(eid)
        tags.add(
            CHAP(
                element_id=eid,
                start_time=start_ms,
                end_time=end_ms,
                sub_frames=[TIT2(encoding=3, text=[title])],
            )
        )

    tags.add(
        CTOC(
            element_id="toc",
            flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
            child_element_ids=child_ids,
            sub_frames=[TIT2(encoding=3, text=["Chapters"])],
        )
    )
    tags.save(str(path))


def write_m3u_playlist(path: str | Path, entries: list[tuple[str, str, float]]) -> None:
    """Write an extended M3U playlist.

    ``entries`` is a list of ``(filename, title, seconds)`` tuples.
    """
    path = Path(path)
    lines = ["#EXTM3U"]
    for filename, title, seconds in entries:
        lines.append(f"#EXTINF:{int(round(seconds))},{title}")
        lines.append(filename)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# M4B audiobooks (needs system ffmpeg — checked at call time).
# ---------------------------------------------------------------------------

def write_pcm_wav(pcm: bytes, wav_path: str | Path, sample_rate: int) -> None:
    """Write raw 16-bit mono PCM to a WAV file (lossless work file)."""
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def wav_duration(wav_path: str | Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def write_m4b(
    entries: list[tuple[Path, str]], out_path: str | Path,
    bitrate: int = 64, album: str | None = None, artist: str | None = None,
) -> None:
    """Mux per-chapter WAVs into one .m4b with chapter marks via ffmpeg.

    ``entries`` is ordered ``(wav_path, chapter_title)``. AAC is encoded
    straight from the lossless work files, so there is no double-lossy step.
    """
    import shutil
    import subprocess
    import tempfile

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            ".m4b output needs ffmpeg on PATH (chapters + AAC mux); "
            "use --chapters split for the ffmpeg-free MP3 folder instead."
        )
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory() as td:
        concat = Path(td) / "concat.txt"
        concat.write_text(
            "".join(f"file '{p.resolve()}'\n" for p, _ in entries),
            encoding="utf-8",
        )
        meta_lines = [";FFMETADATA1"]
        if album:
            meta_lines += [f"album={album}", f"title={album}"]
        if artist:
            meta_lines += [f"artist={artist}"]
        meta_lines.append("genre=Audiobook")
        clock = 0.0
        for wav_path, title in entries:
            dur = wav_duration(wav_path)
            meta_lines += [
                "[CHAPTER]", "TIMEBASE=1/1000",
                f"START={int(clock * 1000)}",
                f"END={int((clock + dur) * 1000)}",
                "title=" + title.replace("\n", " "),
            ]
            clock += dur
        meta = Path(td) / "meta.txt"
        meta.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", str(concat),
             "-i", str(meta), "-map_metadata", "1",
             "-c:a", "aac", "-b:a", f"{bitrate}k",
             "-movflags", "+faststart", "-f", "mp4", str(out_path)],
            check=True,
        )


def tag_mp3(
    mp3_path: str | Path, title: str, album: str, track: int, total: int,
    artist: str | None = None,
) -> None:
    """Write basic ID3 tags so players sort and group split chapters."""
    from mutagen.id3 import ID3, TALB, TCON, TIT2, TPE1, TRCK
    from mutagen.id3._util import ID3NoHeaderError

    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        tags = ID3()
    tags.setall("TIT2", [TIT2(encoding=3, text=title)])
    tags.setall("TALB", [TALB(encoding=3, text=album)])
    tags.setall("TRCK", [TRCK(encoding=3, text=f"{track}/{total}")])
    tags.setall("TCON", [TCON(encoding=3, text="Audiobook")])
    if artist:
        tags.setall("TPE1", [TPE1(encoding=3, text=artist)])
    tags.save(str(mp3_path))
