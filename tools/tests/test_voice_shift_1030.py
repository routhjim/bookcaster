"""10-30: Holmes's lips barely moved (SyncNet 1 of 8 even in close-up) because his voice is deep: LTX now hears a deep
voice lifted toward TALK_SHIFT_TO (the clip keeps the real one). Also the day's director fixes: no listener cutaways
inside a line, no frozen tails on long quiet slots, group shots only of people already in, a breakdown cache that our
own place descriptions don't break."""
import inspect, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

d = Path(tempfile.mkdtemp(prefix="vshift_"))
def tone(f0, name):                     # a buzzy voice-like tone: harmonics of f0, syllable-gated
    t = np.arange(0, 3.0, 1 / 16000)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12)) * (np.sin(2 * np.pi * 4 * t) > -0.3)
    raw = d / (name + ".f32")
    (0.2 * x).astype(np.float32).tofile(raw)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", "16000", "-ac", "1", "-i", str(raw),
                    str(d / (name + ".wav"))], check=True)
    return d / (name + ".wav")
for f0 in (88, 106, 195):
    got = S.voice_pitch([tone(f0, f"v{f0}")])
    check(f"pitch of a {f0} Hz voice", got and abs(got - f0) < 0.06 * f0, got)
check("a deep voice (88 Hz) is lifted, formants kept, treble down, then the usual drive filter",
      S.drive_af(88.0).startswith("rubberband=pitch=1.2") and "formant=preserved" in S.drive_af(88.0)
      and S.drive_af(88.0).endswith(S.TALK_DRIVE_AF) and S.TALK_SHIFT_TREBLE in S.drive_af(88.0), S.drive_af(88.0))
check("Watson (106 Hz) and Helen (195 Hz) are left alone",
      S.drive_af(106.0) == S.TALK_DRIVE_AF and S.drive_af(195.0) == S.TALK_DRIVE_AF)
check("an unmeasured voice is left alone", S.drive_af(None) == S.TALK_DRIVE_AF)
check("the lift is capped", "pitch=1.250" in S.drive_af(60.0), S.drive_af(60.0))
src = inspect.getsource(S.ltx)
check("the driver wav uses the speaker's drive filter; the clip keeps the real voice (audio_mux = pad)",
      "drive_af(pitch.get(" in src and '"audio_mux": str(pad)' in src)

check("the director keeps a long line on its speaker", "never\n  cut away from the speaker" in S.SHOT_RULES
      and "close reaction of the listener and back" not in S.SHOT_RULES)
sh = inspect.getsource(S)
check("no 8-second split of talk shots into listener reactions", "/ 8.0)" not in sh)
check("long quiet slots are cut in pieces of at most SHOT_MAX + 0.5", "(SHOT_MAX + 0.5)))]" in sh)
check("group shots only of those already shown here", "w in people_here or w in where_of" in sh)
bsrc = inspect.getsource(S.breakdown)
check("the breakdown cache also keys without the places our runs described (k2)",
      '"k2"' in bsrc and "stable" in inspect.getsource(S.scene_list))
print({"fails": fails})
