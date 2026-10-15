"""Fill casting gaps: when no voice on the server fits a role, find one.

After the LLM proposes a cast it reviews the fit of every pick (gender, age,
accent). For a poor fit the tool first imports a matching real speaker from
the VCTK corpus (tools/import_voices.py, run in the Qwen venv); VCTK only
has speakers aged 18-38, so for children, old characters, or an accent VCTK
lacks it designs a voice instead (Qwen VoiceDesign, with a real accented
speaker's delivery when the accent bank has one).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

VCTK_CACHE = Path.home() / ".cache" / "tts_reader" / "vctk_speakers.json"
IMPORT_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "import_voices.py"
_QWEN_PYTHON = Path.home() / "tts-ab" / ".venv-qwen" / "bin" / "python"
# Age bands the fit review answers in; VCTK covers "young" and part of "middle".
AGE_BANDS = {"child": (0, 15), "young": (16, 30), "middle": (31, 55), "old": (56, 120)}
_ACCENT_NAMES = {"NorthernIrish": "Northern Irish", "SouthAfrican": "South African",
                 "NewZealand": "New Zealand", "Australian English": "Australian"}
# Names an LLM uses for accents we file under another (VCTK's one "British"
# speaker is English; RP/Cockney/etc. are English to the accent bank).
_ACCENT_ALIASES = {"british": "english", "england": "english", "rp": "english",
                   "received pronunciation": "english", "cockney": "english",
                   "london": "english", "scots": "scottish", "scotland": "scottish",
                   "ireland": "irish", "us": "american", "usa": "american"}


def canonical_accent(accent: str) -> str:
    a = accent.strip().lower()
    return _ACCENT_ALIASES.get(a, a)


def import_python() -> str:
    env = os.environ.get("TTS_IMPORT_PYTHON")
    if env:
        return env
    return str(_QWEN_PYTHON) if _QWEN_PYTHON.exists() else sys.executable


def vctk_speakers() -> dict:
    """VCTK speaker metadata cached by tools/import_voices.py ({} if absent)."""
    try:
        return json.loads(VCTK_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def vctk_accents() -> list[str]:
    names = {_ACCENT_NAMES.get(i["accent"], i["accent"]) for i in vctk_speakers().values()}
    return sorted(n for n in names if canonical_accent(n) == n.lower() and n != "Unknown")


def vctk_candidates(gender: str, accent: str, age: str, exclude=()) -> list[str]:
    """VCTK ids matching *gender* and *accent*, closest to the *age* band
    first; [] when the band is outside what VCTK has (children, the old)."""
    lo, hi = AGE_BANDS.get(age, (16, 55))
    hits = []
    for sid, info in vctk_speakers().items():
        name = _ACCENT_NAMES.get(info["accent"], info["accent"]).lower()
        if info.get("gender") != gender or canonical_accent(name) != canonical_accent(accent):
            continue
        if f"vctk_{sid}" in exclude:
            continue
        try:
            years = int(info.get("age") or 0)
        except ValueError:
            continue
        if lo <= years <= hi:
            hits.append((abs(years - (lo + hi) / 2), sid))
    return [sid for _, sid in sorted(hits)]


def import_vctk(sid: str, log=print) -> bool:
    log(f"  importing VCTK speaker {sid}...")
    proc = subprocess.run([import_python(), "-I", str(IMPORT_SCRIPT), "--speakers", sid],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  (import of {sid} failed: {proc.stderr.strip()[-300:]})")
        return False
    return True


def fill_gaps(session, log=print, max_new: int = 4) -> bool:
    """Review the cast's fit and replace poor fits; True if anything changed."""
    from urllib.error import HTTPError, URLError

    from . import design
    from .casting import fetch_voice_roster

    try:
        wants = session.review_fit(vctk_accents(), log=log)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        log(f"warning: fit review unavailable ({exc})")
        return False
    changed = False
    for want in wants[:max_new]:
        name, gender, accent, age = want["name"], want["gender"], want["accent"], want["age"]
        member = None if name == "narrator" else session._match_member(name)
        if name != "narrator" and member is None:
            continue
        label = "narrator" if member is None else member.name
        log(f"  {label}: no good fit (wants {age} {accent} "
            f"{'man' if gender == 'M' else 'woman'})")
        in_roster = {r[0] for r in session.custom_roster}
        ids = vctk_candidates(gender, accent, age, exclude=in_roster)
        if ids and import_vctk(ids[0], log=log):
            voice = f"vctk_{ids[0]}"
            entry = next((r for r in fetch_voice_roster(session.f5_url) if r[0] == voice), None)
            if entry:
                session.custom_roster.append(entry)
                if member is None:
                    session.narrator_voice = voice
                else:
                    member.voice = voice
                log(f"  {label} -> {voice} ({entry[2]})")
                changed = True
                continue
        if member is None:
            continue            # designed narrators: not yet
        if not design.server_ready(session.design_fallback_url):
            log(f"  (no VCTK match; voice-design server not ready at "
                f"{session.design_fallback_url}, keeping {member.voice})")
            continue
        changed |= _design_for(session, member, accent, log)
    if changed:
        _free_fallback(session)
    return changed


def _free_fallback(session) -> None:
    """A voice just given to a character must not double as the fallback
    voice for minor speakers (dedupe would then move the character off it):
    give the fallback the first roster voice nobody else uses."""
    base = lambda v: v.partition("@")[0]
    used = {base(m.voice) for m in session.members} | {base(session.narrator_voice)}
    if base(session.dialogue_voice) not in used:
        return
    spare = [n for n, _, _ in session.roster() if n not in used]
    real = [n for n in spare if not n.startswith("d_")]   # designed voices are per character
    if real or spare:
        session.dialogue_voice = (real or spare)[0]


def _design_for(session, member, accent: str, log=print) -> bool:
    """Design an original voice for *member*, with *accent* if we can do it."""
    from . import design

    accent = canonical_accent(accent)
    member.accent = accent if accent in design.available_accents() else ""
    member.voice_desc = ""
    session.write_voice_descriptions(members=[member], log=log)
    if not member.voice_desc:
        return False
    # design_member builds against design_url; the session isn't in designed
    # mode, so lend it the fallback server just for this voice. Accented
    # voices get their emotion from server-side shaping, not registers.
    saved = session.design_url, session.registers
    session.design_url = session.design_fallback_url
    session.registers = saved[1] and not member.accent
    try:
        return session.design_member(member, log=log)
    except (RuntimeError, OSError) as exc:
        log(f"  (design of {member.name} failed: {exc})")
        return False
    finally:
        session.design_url, session.registers = saved
