"""11-06: spot effects are cut around their event out of the whole MOSS canvas (not its first N seconds), faded
to the clip's length, normalised by one static gain under a true-peak ceiling (not loudnorm), and the best of N
candidates is kept by a mechanical score. Synthetic audio only: no models, GPUs or servers.
Run: python tools/tests/test_sfx_spot_1106.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from tts_reader import sfx as SX, sfx_spot as SP  # noqa: E402

fails = []
SR = 48000
rng = np.random.default_rng(3)


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


def hit(at, amp=0.9, decay=0.15, n=10 * SR):
    """A canvas: a low noise floor and one decaying noise burst at *at* seconds."""
    x = rng.normal(0, 1e-4, n)
    i = int(at * SR)
    t = np.arange(n - i) / SR
    x[i:] += amp * rng.normal(0, 0.4, n - i) * np.exp(-t / decay)
    return np.clip(x, -1, 1)


# --- the window: around the event, not the canvas's first seconds
x = hit(4.2)
y, info = SP.cut(x, SR, 0.8)
check("cut: the spot's length", len(y) == int(0.8 * SR))
check("cut: onset found at the event", abs(info["onset_s"] - 4.2) < 0.03)
check("cut: a short pre-roll before it", abs(info["start_s"] - (4.2 - SP.PRE_S)) < 0.02)
check("cut: the event is in the window (most of the canvas's energy)", info["energy_in_window"] > 0.9)

# --- fades: short in, scaled out (not 0.4 s on a 0.8 s clip), the ends at zero
check("fade-out scales with the clip", abs(SP.fade_out_s(0.8) - 0.16) < 1e-9 and SP.fade_out_s(3.0) == 0.4
      and SP.fade_out_s(0.2) == 0.05)
check("fades: starts and ends at silence", abs(y[0]) < 1e-6 and abs(y[-1]) < 1e-6)
check("fades: the attack is kept (loud within 20 ms of the pre-roll)",
      np.abs(y[int((SP.PRE_S + 0.01) * SR):int((SP.PRE_S + 0.03) * SR)]).max() > 0.2 * np.abs(y).max())

# --- normalise: one static gain, no clipping, short clips measured
tone = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(SR) / SR)
check("loudness: a -6 dBFS-peak 1 kHz sine reads -9.0 LUFS (BS.1770)", abs(SP.loudness(tone, SR) - (-9.03)) < 0.1)
check("loudness: a 0.8 s clip is measured (no 3 s minimum)", -30 < SP.loudness(tone[:int(0.8 * SR)], SR) < 0)
quiet = tone * 0.01
q, qi = SP.normalise(quiet, SR)
check("normalise: a quiet steady sound reaches the target", abs(SP.loudness(q, SR) - SP.TARGET_LUFS) < 0.3)
check("normalise: one static gain (the shape unchanged)", np.allclose(q / q.max(), quiet / quiet.max(), atol=1e-4))
spike = np.zeros(SR)
spike[1000:1100] = rng.normal(0, 0.9, 100)
s, si = SP.normalise(spike, SR)
check("normalise: a transient is held under the true-peak ceiling", SP.true_peak_db(s) <= SP.CEILING_DBTP + 0.05
      and si["peak_limited"])
check("normalise: nothing at full scale", np.abs(y).max() < 0.95 and np.abs(s).max() < 0.95)

# --- pick: a clean single hit beats a clipped one, a scatter of hits and silence
clean = hit(2.0, amp=0.5)
clipped = np.clip(hit(2.0, amp=8.0), -1, 1)
scatter = hit(1.0, 0.5) + hit(3.5, 0.5) + hit(6.0, 0.5) + hit(8.5, 0.5)
silent = rng.normal(0, 1e-5, 10 * SR)
_, best, infos = SP.pick([clipped, scatter, clean, silent], SR, 1.0)
check("pick: the clean single hit wins", best == 2)
check("pick: the clipped candidate scores below it", infos[0]["score"] < infos[2]["score"])
check("pick: the scatter scores below it", infos[1]["score"] < infos[2]["score"])

# --- a sound that runs past the canvas end still gets a full-length window
late = hit(9.7)
y2, i2 = SP.cut(late, SR, 1.5)
check("cut: an event near the canvas end keeps the spot's length", len(y2) == int(1.5 * SR) and i2["start_s"] <= 8.5)

# --- the mix no longer loudnorms or re-fades spots; spot files are versioned so raw old ones aren't reused
src = (ROOT / "tts_reader" / "sfx.py").read_text()
spot_chain = src[src.index("# the file is cut, faded and normalised already"):src.index("spots.append")]
check("mix: no loudnorm or fade on spots", "loudnorm" not in spot_chain and "afade" not in spot_chain)
check("names: spot files carry the cut version (old raw files aren't reused)",
      SX._fx_name("spot", 0, "a knock", 1.5) != "spot000_" + __import__("hashlib").sha1(b"a knock|1.5").hexdigest()[:8]
      + ".wav" and SX._fx_name("bed", 0, "fire", 30.0) == "bed000_" +
      __import__("hashlib").sha1(b"fire|30.0").hexdigest()[:8] + ".wav")

print(f"\n{len(fails)} failure(s)" if fails else "\nall passed")
sys.exit(1 if fails else 0)
