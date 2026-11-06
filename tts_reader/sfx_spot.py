"""Mechanics for a short spot effect cut out of a longer MOSS canvas (numpy only;
runs inside the MOSS venv from tools/sfx_gen.py and in plain tests).

11-06: spots were the first N seconds of the canvas (the hit could fall outside
or be cut mid-sound), ffmpeg's single-pass loudnorm per spot (mis-measures
under 3 s, its dynamic mode lifts the floor and pumps the attack, 3 of 4 Long
Dark spots peaked at 1.0) and a fixed 0.4 s fade-out (half a 0.8 s clip).
Now: find the event by onset and energy, cut a window of the spot's length
around it with a short pre-roll and fades scaled to the clip, normalise with
one static gain (BS.1770 loudness on 400 ms blocks, capped by a true-peak
ceiling), and pick the best of N candidates by a mechanical score.
"""

from __future__ import annotations

import numpy as np

HOP_S = 0.01                    # onset/energy frame hop
PRE_S = 0.05                    # pre-roll before the detected onset
FADE_IN_S = 0.005               # just enough to avoid a click at the cut
TARGET_LUFS = -22.0             # what the mix's loudnorm used to aim at (SPOT_DB stays relative to it)
CEILING_DBTP = -2.0             # the old loudnorm's TP=-2: SPOT_DB "loud" (+1 dB) still lands under 0 dBFS
CLIP = 0.999


def fade_out_s(seconds: float) -> float:
    """A fade-out scaled to the clip: a fifth of it, 50 ms to 0.4 s."""
    return float(min(0.4, max(0.05, 0.2 * seconds)))


def _frames(x: np.ndarray, sr: int):
    hop = max(1, int(sr * HOP_S))
    win = 2 * hop
    n = max(1, 1 + (len(x) - win) // hop) if len(x) >= win else 1
    pad = np.zeros(win + hop * n)
    pad[:len(x)] = x
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return pad[idx], hop


def onset_energy(x: np.ndarray, sr: int):
    """Per-frame onset strength (log spectral flux, silence assumed before the
    start so a sound that opens at sample 0 still has an onset) and energy."""
    fr, hop = _frames(x, sr)
    energy = (fr[:, :hop] ** 2).mean(axis=1)
    mag = np.log1p(100.0 * np.abs(np.fft.rfft(fr * np.hanning(fr.shape[1]), axis=1)))
    prev = np.vstack([np.zeros((1, mag.shape[1])), mag[:-1]])
    flux = np.maximum(0.0, mag - prev).sum(axis=1)
    return flux, energy, hop


def find_window(x: np.ndarray, sr: int, seconds: float):
    """(start sample, onset sample, info) of the best window of *seconds*: one
    starting PRE_S before a strong onset that holds the most of the canvas's
    energy (a strong onset: at least 30 % of the strongest)."""
    n = int(round(sr * seconds))
    flux, energy, hop = onset_energy(x, sr)
    total = float(energy.sum()) or 1e-12
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    fmax = float(flux.max()) or 1e-12
    peaks = [i for i in range(len(flux))
             if flux[i] >= 0.3 * fmax and flux[i] >= flux[max(0, i - 1)] and flux[i] >= flux[min(len(flux) - 1, i + 1)]]
    best = None
    w = max(1, n // hop)
    for i in peaks or [int(np.argmax(flux))]:
        start = max(0, min(i * hop - int(PRE_S * sr), len(x) - n))
        f0 = start // hop
        frac = float(cum[min(len(energy), f0 + w)] - cum[f0]) / total
        key = (round(frac, 3), flux[i])
        if best is None or key > best[0]:
            best = (key, start, i * hop, frac)
    _, start, onset, frac = best
    return max(0, start), onset, {"energy_in_window": round(frac, 3), "onset_s": round(onset / sr, 3)}


def _fade(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    y = y.copy()
    fi = min(len(y), int(FADE_IN_S * sr))
    if fi:
        y[:fi] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fi))
    fo = min(len(y), int(fade_out_s(seconds) * sr))
    if fo:
        y[len(y) - fo:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fo))
    return y


def _k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    """BS.1770 K-weighting (the two 48 kHz biquads), applied as their magnitude
    response in the frequency domain (energy only, so phase doesn't matter)."""
    def resp(b, a, w):
        z = np.exp(-1j * w)
        return np.abs((b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z))
    m = 1 << int(np.ceil(np.log2(max(2, len(x)))) + 1)
    w = 2 * np.pi * np.fft.rfftfreq(m, 1.0 / sr) / 48000.0       # the coefficients are for 48 kHz
    h = resp([1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585], w) * \
        resp([1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621], w)
    return np.fft.irfft(np.fft.rfft(x, m) * h, m)[:len(x)]


def loudness(x: np.ndarray, sr: int) -> float:
    """Gated BS.1770 loudness (LUFS) on 400 ms blocks, 75 % overlap; a clip
    shorter than a block is one block."""
    y = _k_weight(np.asarray(x, dtype=np.float64), sr)
    b = int(0.4 * sr)
    if len(y) <= b:
        blocks = np.array([np.mean(y ** 2)])
    else:
        step = b // 4
        blocks = np.array([np.mean(y[s:s + b] ** 2) for s in range(0, len(y) - b + 1, step)])
    lk = -0.691 + 10 * np.log10(np.maximum(blocks, 1e-20))
    keep = lk > -70
    if not keep.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(blocks[keep].mean()) - 10
    keep &= lk > rel
    return float(-0.691 + 10 * np.log10(blocks[keep].mean()))


def true_peak_db(x: np.ndarray) -> float:
    """Peak after 4x oversampling (FFT), in dBFS."""
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return -120.0
    up = np.fft.irfft(np.fft.rfft(x), 4 * len(x)) * 4
    return float(20 * np.log10(max(np.abs(up).max(), np.abs(x).max(), 1e-6)))


def normalise(y: np.ndarray, sr: int, target: float = TARGET_LUFS, ceiling: float = CEILING_DBTP):
    """One static gain to *target* LUFS, lowered if the true peak would pass *ceiling*."""
    lu, tp = loudness(y, sr), true_peak_db(y)
    gain = min(target - lu, ceiling - tp)
    return (y * 10 ** (gain / 20)).astype(np.float32), {"lufs_in": round(lu, 1), "tp_in": round(tp, 1),
                                                         "gain_db": round(gain, 1), "peak_limited": gain < target - lu}


def cut(x: np.ndarray, sr: int, seconds: float):
    """The spot of *seconds* out of canvas *x*, faded and normalised, plus its info."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(round(sr * seconds))
    start, onset, info = find_window(x, sr, seconds)
    y = x[start:start + n]
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y))])
    raw = y
    y, ninfo = normalise(_fade(y, sr, seconds), sr)
    info.update(ninfo, start_s=round(start / sr, 3), clipped_raw=float(np.mean(np.abs(raw) >= CLIP)))
    info["score"] = score(x, raw, sr, info)
    return y, info


def score(canvas: np.ndarray, raw: np.ndarray, sr: int, info: dict) -> float:
    """Higher is better: the canvas's energy concentrated in the window (one
    event, not a scatter), one dominant onset in the window, a tail that has
    decayed by the cut, and no clipped samples in what MOSS made."""
    flux, energy, hop = onset_energy(raw, sr)
    top = np.sort(flux)[::-1]
    dominance = float(top[0] / (top[:4].sum() or 1e-12)) if len(top) else 0.0     # 1/4 (four equal) .. 1 (one)
    tail = energy[-max(1, len(energy) // 10):].mean() / (energy.max() or 1e-12)
    clipped = info.get("clipped_raw", 0.0) > 0
    s = 0.5 * info["energy_in_window"] + 0.3 * dominance + 0.2 * (1.0 - min(1.0, tail * 10)) - (1.0 if clipped else 0.0)
    if energy.max() < 1e-7:                      # silence
        s -= 1.0
    return round(float(s), 3)


def pick(canvases, sr: int, seconds: float):
    """Cut each candidate canvas; return (best spot, its index, every candidate's info)."""
    outs = [cut(c, sr, seconds) for c in canvases]
    best = max(range(len(outs)), key=lambda i: outs[i][1]["score"])
    return outs[best][0], best, [o[1] for o in outs]
