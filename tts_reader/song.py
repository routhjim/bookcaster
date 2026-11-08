"""Song mode: a song (its audio + its lyrics) made into a one-chapter "book" the shot flow films as a music video.

What the flow's narration steps would make, made from the song instead:
    work/<stem>.mp3            the VOICE track = the song's vocal stem (torchaudio's Hybrid Demucs): the director's
                               word times, the talk shots' lip-sync drive and SyncNet hear the singing only
    work/final/<stem>.mp3      the MIX = the whole song (what the flow's mix checks read)
    work/<stem>.timeline.json  one entry per lyric line, timed on the vocal stem (Whisper word times matched to the
                               lyrics, tools/shot_flow.word_times), with its section label; the music between lines
                               is an entry of its own (kind "music")
    chapters/<stem>.txt        the lyrics, a paragraph per section, after one line naming who sings lead
    song.json                  the marker the shot flow reads (tools/shot_flow.song_of): the lossless audio the cut is
                               muxed on, the style tags, the sections, the lead singer
The lead singer is the language model's choice (one ask: the lyrics and the style tags). Everything after this
(looks, sheets, LoRAs, the director, keys, LTX, talk takes, the cut) is the shot flow's own:
    bookcaster video <book dir> --level animated --flow shots
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

STEM = "01_song"
SEP_PY = Path.home() / "tts-ab" / ".venv-qwen" / "bin" / "python"     # torch + torchaudio (CPU is enough: ~minutes)
SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
GAP_S = 1.2                          # silence in the vocal stem between two lines longer than this: a music entry
WHISPER_SONG = "large-v3-turbo"      # faster-whisper's name (cached: mobiuslabsgmbh/faster-whisper-large-v3-turbo)
TAIL_S = 2.5                         # a line's last word holds at most this long (a sung note) before the music

# torchaudio's Hybrid Demucs (HDEMUCS_HIGH_MUSDB_PLUS), applied in overlapping 10 s pieces (its own recipe)
_SEP_RUN = r"""
import subprocess, sys
import numpy as np, torch, torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as B
src, out = sys.argv[1], sys.argv[2]
sr = B.sample_rate
raw = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
                     capture_output=True, check=True).stdout
x = torch.from_numpy(np.frombuffer(raw, np.float32).copy()).reshape(-1, 2).T
m = B.get_model().eval()
torch.set_num_threads(16)
ref = x.mean(0)
x = (x - ref.mean()) / ref.std()
seg, ov = 10 * sr, int(0.1 * 10 * sr)
n = x.shape[1]
outp = torch.zeros(len(m.sources), 2, n)
fade = torchaudio.transforms.Fade(fade_in_len=0, fade_out_len=ov, fade_shape="linear")
start, end = 0, seg + ov
with torch.inference_mode():
    while start < n - ov:
        chunk = x[:, start:end][None]
        y = m(chunk)[0]
        y = fade(y)
        outp[:, :, start:start + y.shape[-1]] += y
        if start == 0:
            fade.fade_in_len = ov
            start += seg - ov
        else:
            start += seg
        end += seg
v = outp[m.sources.index("vocals")] * ref.std() + ref.mean()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(sr), "-ac", "2", "-i", "-", "-c:a", "pcm_s16le",
                out], input=v.T.contiguous().numpy().astype(np.float32).tobytes(), check=True)
"""


def parse_lyrics(text: str) -> list:
    """[{"section", "text"}] for every sung line, in order; the section is the last [label] above it."""
    out, sec = [], ""
    for ln in text.splitlines():
        m = SECTION.match(ln)
        if m:
            sec = m.group(1).strip()
        elif ln.strip():
            out.append({"section": sec, "text": ln.strip()})
    return out


def sections(lines: list) -> list:
    """[(label, [line indices])] in order: each run of lines under one label (a repeated label is a new section)."""
    out = []
    for i, x in enumerate(lines):
        if not out or out[-1][0] != x["section"] or (i and lines[i - 1]["section"] != x["section"]):
            out.append((x["section"], [i]))
        else:
            out[-1][1].append(i)
    return out


def separate(audio: Path, out: Path, log=print) -> Path:
    """The song's vocal stem (wav, 44.1 kHz stereo) - cached beside its source's size and time."""
    st = audio.stat()
    mark = out.with_suffix(".src")
    key = f"{audio.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    if out.exists() and mark.exists() and mark.read_text() == key:
        return out
    log(f"song: separating the vocals of {audio.name} (Hybrid Demucs, CPU)")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.wav")
    subprocess.run([str(SEP_PY), "-I", "-c", _SEP_RUN, str(audio), str(tmp)], check=True)
    tmp.replace(out)
    mark.write_text(key)
    return out


def _seconds(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def line_times(word_t: list, line_words: list, env, total: float) -> list:
    """[(start, end)] of each lyric line from its words' times (None where Whisper missed a word): a line starts at
    its first heard word (a line heard nowhere: between its neighbours, by word count), ends where the voice falls
    silent after its last word (at most TAIL_S on), and never runs into the next line."""
    import numpy as np
    n = len(line_words)
    sm = np.convolve(env, np.ones(10) / 10, mode="same") if len(env) else env
    loud = float(np.median(sm[sm > 1e-4])) if len(sm) and (sm > 1e-4).any() else 0.0
    on = np.nonzero(sm > 0.2 * loud)[0] if loud else []
    voice_end = min(total, (on[-1] + 1) / 100) if len(on) else total      # (the last sung second)

    def onset(t, lim):                  # past the held note still sounding at t, then past the silence after it
        j, end = int(t * 100), int(lim * 100)
        k = j
        while k < min(end, j + int(TAIL_S * 100), len(sm)) and sm[k] > 0.2 * loud:
            k += 1
        while k < min(end, len(sm)) and sm[k] <= 0.2 * loud:
            k += 1
        return k / 100 if loud and k < end else t

    first = [next((word_t[j] for j in idx if word_t[j] is not None), None) for idx in line_words]
    last = [next((word_t[j] for j in reversed(idx) if word_t[j] is not None), None) for idx in line_words]
    i = 0
    while i < n:                        # a run of unheard lines: spread by word count between the heard ones around it
        if first[i] is not None:
            i += 1
            continue
        kb = next((k for k in range(i + 1, n) if first[k] is not None), n)
        a = last[i - 1] + 0.3 if i else 0.0
        b = first[kb] if kb < n else max(voice_end, a + 0.5)
        a = onset(a, b)                 # (where the voice comes in again after the line before it)
        w = [len(line_words[k]) for k in range(i, kb)]
        per = (b - a) / max(1, sum(w))
        off = 0
        for k in range(i, kb):
            first[k], last[k] = a + per * off, a + per * (off + max(0, len(line_words[k]) - 1))
            off += len(line_words[k])
        i = kb
    for i in range(1, n):               # in order
        first[i] = max(first[i], first[i - 1] + 0.3)
        last[i] = max(last[i], first[i])
    out = []
    for i in range(n):
        nxt = first[i + 1] if i + 1 < n else total
        cap = min(nxt, last[i] + TAIL_S)
        e = last[i] + 0.3
        j = int(e * 100)
        while j < min(len(sm), int(cap * 100)) and sm[j] > 0.2 * loud:
            j += 1
        e = min(cap, max(e, j / 100))
        out.append((round(first[i], 3), round(min(e, nxt), 3)))
    return out


SINGER_ASK = """A song is being made into a music video. Its lyrics and its recording's style notes are below.
Decide who sings the lead on screen. If the lyrics make one of their own people the one singing ("I", or the one who
leads the others), it is that person; otherwise a new person who fits the lyrics' world and the voice the style notes
describe. Give the name the video calls them by (a name or a role as the lyrics would say it) and one sentence on who
they are in the song's world, never their looks.
Reply with JSON only: {{"name": "...", "who": "..."}}

Style notes: {tags}

Lyrics:
{lyrics}"""


def pick_singer(lyrics: str, tags: str, log=print) -> dict:
    from . import animate
    from . import bookcaster as BC
    from .score import _chat
    animate.igpu("strata", log)
    for _ in range(3):
        rep = _chat(BC.LLM_URL, BC.LLM_MODEL, SINGER_ASK.format(tags=tags or "(none)", lyrics=lyrics), max_tokens=400)
        m = re.search(r"\{.*\}", rep or "", re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except ValueError:
            d = {}
        if str(d.get("name") or "").strip():
            return {"name": str(d["name"]).strip(), "who": str(d.get("who") or "").strip()}
    raise RuntimeError("song: the language model named no lead singer")


def make(book: Path, audio: Path, lyrics: Path, title: str = "", tags: str = "", singer: str = "",
         log=print) -> Path:
    """The song's book (see the module's doc); safe to run again (the stem and the word times are cached)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import shot_flow as SF
    book = Path(book).expanduser().resolve()
    audio = Path(audio).expanduser().resolve()
    text = Path(lyrics).expanduser().read_text(encoding="utf-8")
    lines = parse_lyrics(text)
    if not lines:
        raise ValueError(f"no lyric lines in {lyrics}")
    title = title or audio.stem.replace("_", " ").title()
    work = book / "work"
    for d in (book / "chapters", work / "final", work / "song"):
        d.mkdir(parents=True, exist_ok=True)
    meta = book / "song.json"
    old = json.loads(meta.read_text()) if meta.exists() else {}
    who = {"name": singer, "who": ""} if singer else (old.get("singer") if old.get("lyrics") == text and old.get("singer")
                                                       else pick_singer(text, tags, log))
    # the chapter: who sings, then each section a paragraph (its label first, as the lyrics have it)
    secs = sections(lines)
    chapter = (f"{title}\n\n{who['name']} sings the lead, the others answering. {who['who']}".rstrip() + "\n\n"
               + "\n\n".join(f"[{lab}]\n" + "\n".join(lines[i]["text"] for i in idx) for lab, idx in secs) + "\n")
    cp = book / "chapters" / f"{STEM}.txt"
    if not cp.exists() or cp.read_text(encoding="utf-8") != chapter:
        cp.write_text(chapter, encoding="utf-8")
    for f, body in ((book / "book.txt", chapter), (book / "selection.txt", chapter), (book / "flow.txt", "shots\n")):
        if not f.exists() or f.read_text(encoding="utf-8") != body:
            f.write_text(body, encoding="utf-8")
    if not (book / "meta.json").exists():
        (book / "meta.json").write_text(json.dumps({"title": title, "author": "", "source": "song", "license": "private use"},
                                                   indent=1))
    (book / "book.cast.json").write_text(json.dumps({"version": 1, "characters": {who["name"]: {
        "role": who["who"], "voice": "", "gender": "", "same_as": ""}}}, indent=1))
    # the stems: the voice track is the vocals, the mix is the song (both the song's length: the flow's mix check)
    voc = separate(audio, work / "song" / "vocals.wav", log)
    total = _seconds(audio)
    voices, mix = work / f"{STEM}.mp3", work / "final" / f"{STEM}.mp3"
    for src, dst in ((voc, voices), (audio, mix)):
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", f"{total:.3f}", "-ar", "48000",
                            "-c:a", "libmp3lame", "-b:a", "320k", str(dst)], check=True)
    # each lyric word's time in the vocals (the flow's own forced alignment), then each line's span
    # (singing: the large model, prompted with the lyrics' own lines, each once - Whisper's prompt is ~200 words;
    # only the lyric words matched: shot_flow.word_times reads song.json's "align")
    once = list(dict.fromkeys(x["text"] for x in lines))
    body_at = len(SF._WORD.findall(chapter.split("\n\n[", 1)[0]))     # (the lines' words start after the intro)
    align = {"model": WHISPER_SONG, "prompt": " ".join(" ".join(once).split()[:180]), "from_word": body_at}
    meta.write_text(json.dumps(dict(old, audio=str(audio), lyrics=text, singer=who, align=align), indent=1,
                               ensure_ascii=False))
    times = SF.word_times(book, STEM, track=voices, log=log)
    words = SF._WORD.findall(chapter)
    at, line_words = body_at, []
    for lab, idx in secs:
        at += len(SF._WORD.findall(f"[{lab}]"))
        for i in idx:
            n = len(SF._WORD.findall(lines[i]["text"]))
            line_words.append(list(range(at, at + n)))
            at += n
    assert at == len(words), (at, len(words))
    heard = sum(times[j] is not None for idx in line_words for j in idx)
    log(f"song: {heard} of {sum(map(len, line_words))} lyric words heard in the vocals")
    spans = line_times(times, line_words, SF._envelope(voices), total)
    tl = []
    prev_end, prev_sec = 0.0, ""
    for x, (a, b) in zip(lines, spans):
        if a - prev_end > GAP_S or not tl:
            tl.append({"t": round(prev_end, 3), "kind": "music", "speaker": "narrator", "emotion": "",
                       "text": "(music)", "section": x["section"] if not tl else prev_sec})
        tl.append({"t": a, "kind": "quote", "speaker": who["name"], "emotion": "", "text": x["text"],
                   "section": x["section"]})
        prev_end, prev_sec = b, x["section"]
    if total - prev_end > GAP_S:
        tl.append({"t": round(prev_end, 3), "kind": "music", "speaker": "narrator", "emotion": "", "text": "(music)",
                   "section": prev_sec})
    if tl and tl[0]["t"] < 0.05 and len(tl) > 1 and tl[1]["t"] < 0.3:   # (a song that opens on its first word)
        tl.pop(0)
    (work / f"{STEM}.timeline.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False))
    meta.write_text(json.dumps({"title": title, "audio": str(audio), "lyrics": text, "tags": tags, "singer": who,
                                "seconds": round(total, 3), "align": align,
                                "sections": [{"label": lab, "start": spans[idx[0]][0], "end": spans[idx[-1]][1]}
                                             for lab, idx in secs]}, indent=1, ensure_ascii=False))
    log(f"song: {book} ({len(lines)} lines in {len(secs)} sections, {total:.1f} s; lead: {who['name']})")
    return book
