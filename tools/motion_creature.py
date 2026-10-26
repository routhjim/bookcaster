"""Procedural creature motion with real mechanics (the "procedural" motion source; pure numpy, any python).

    from motion_creature import animate, VERBS
    cm = animate("winged_quadruped", [
            {"verb": "brace", "t": 0.0, "dur": 0.8},
            {"verb": "recoil", "t": 1.67, "from": [-1, 0, 0], "strength": 1.0},
            {"verb": "roar", "t": 2.3, "dur": 1.5}],
         dur=5.04, fps=24, targets={...})
    cm.save(base)                       # CanonicalMotion (.npz + .json), template joint names

Units and frame: the template's (tools/stage_templates/creatures.json): body length (|chest - pelvis|) = 1,
+X forward at frame 0, Z up, +Y left, ground z = 0. Targets are given in the SAME frame (motion_shot converts
shot-world targets into it, and to_rig() scales/places the result in the shot).

Mechanics (no keyframe blending):
  * CHANNELS: every verb only sets TARGETS for a few channels: root (pelvis xyz, yaw, pitch, roll), crouch,
    head target point, per-limb reach targets, jaw opening, wing (elevation, elbow, wrist, flap), chest
    breathing. Each channel follows its target through a damped spring (stiffness by body part: the head
    is fast, the chest medium, the pelvis slow, so the HEAD LEADS and the BODY FOLLOWS; damping < 1 gives
    the overshoot of follow-through).
  * SOLVE per frame: trunk = pelvis + bent spine (pitch/yaw distributed over the vertebrae); neck = a chain
    that reaches for the head target within its bend limits (FABRIK with angle clamps); legs = toe targets
    (planted feet stay where they were put down; swing feet follow gait phases or the verb's reach target)
    and 2-bone IK hip-knee-ankle with the knee pole forward; wings = shoulder elevation + elbow/wrist fold
    angles within the template limits + flap, the fingers fanned by the wrist angle.
  * BALANCE: the centre of mass (segment-weighted) is kept over the support polygon of the planted feet
    (+ the tail tip when it rests on the ground) by shifting the pelvis horizontally (feet stay planted).
  * SECONDARY: tail and wing tips are damped spring chains that lag the body (overlap, follow-through),
    with a ground collision for the tail.
  * GAIT (walk_to): walk = lateral sequence LH, LF, RH, RF (duty 0.75), trot = diagonal pairs (duty 0.5);
    swing feet arc (height 0.12 body lengths) to a target ahead of the hip; the body is carried over them.
Contacts/timing: verbs take absolute times, so a claw strike lands exactly on the shot's contact frame and
a recoil starts on the human's impact frame (motion_shot couples them).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from motion_templates import load_template  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])


def _n(v):
    v = np.asarray(v, float)
    d = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(d, 1e-9)


def rot_axis(axis, ang):
    a = _n(axis)
    c, s = math.cos(ang), math.sin(ang)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def Rz(a):
    return rot_axis([0, 0, 1], a)


def Ry(a):
    return rot_axis([0, 1, 0], a)


def Rx(a):
    return rot_axis([1, 0, 0], a)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------------------------------
# channels: target -> damped spring
# --------------------------------------------------------------------------------------------------

STIFF = {   # natural frequency (Hz) and damping ratio per channel: head fast, body slower = overlap
    "root": (1.6, 0.85), "yaw": (1.4, 0.9), "pitch": (1.5, 0.75), "roll": (1.5, 0.8), "crouch": (2.0, 0.8),
    "head": (3.2, 0.62), "jaw": (6.0, 0.9), "wing": (2.6, 0.7), "flap": (8.0, 1.0), "breath": (3.0, 1.0),
    "reach": (4.0, 0.75), "neckup": (2.5, 0.7),
}


class Spring:
    def __init__(self, x0, hz, zeta):
        self.x = np.array(x0, float)
        self.v = np.zeros_like(self.x)
        self.w, self.z = 2 * math.pi * hz, zeta

    def step(self, target, dt, n=4):
        h = dt / n
        for _ in range(n):            # semi-implicit Euler, substepped
            a = self.w * self.w * (np.asarray(target, float) - self.x) - 2 * self.z * self.w * self.v
            self.v = self.v + a * h
            self.x = self.x + self.v * h
        return self.x.copy()

    def kick(self, dv):
        self.v = self.v + np.asarray(dv, float)


# --------------------------------------------------------------------------------------------------
# verbs: functions of time that write channel TARGETS (and events) for one frame
# --------------------------------------------------------------------------------------------------

def _env(t, t0, dur, att=0.25, rel=0.3):
    """0 -> 1 over the attack, hold, 1 -> 0 over the release (fractions of dur)."""
    if dur <= 0:
        return 0.0
    u = (t - t0) / dur
    if u < 0 or u > 1:
        return 0.0
    if u < att:
        return smooth(u / att)
    if u > 1 - rel:
        return smooth((1 - u) / rel)
    return 1.0


def v_idle(c, t, e, T):
    pass                              # breathing + idle sway are always on (see base_targets)


def v_brace(c, t, e, T):
    k = smooth((t - e["t"]) / max(e.get("dur", 0.6), 1e-3)) if t >= e["t"] else 0.0
    if e.get("until") is not None and t > e["until"]:
        k *= max(0.0, 1 - (t - e["until"]) / 0.4)
    c["crouch"] += 0.10 * k * e.get("amount", 1.0)
    c["pitch"] += math.radians(-4) * k
    c["head_off"] += np.array([-0.12, 0.0, -0.06]) * k
    c["wing"] += np.array([15, 40, 40]) * k            # wings half-raise (elev, elbow, wrist +)
    c["stance"] += 0.06 * k                            # feet spread


def v_rear(c, t, e, T):
    k = _env(t, e["t"], e.get("dur", 2.0), 0.3, 0.3)
    c["pitch"] += math.radians(e.get("deg", 38)) * k
    c["root_off"] += np.array([-0.18, 0.0, 0.0]) * k   # weight back over the hind feet
    c["lift"] |= {"fleg_l", "fleg_r"} if k > 0.15 else set()
    c["curl"] = max(c["curl"], k)
    c["wing"] += np.array([35, 90, 90]) * k
    c["head_off"] += np.array([0.05, 0.0, 0.25]) * k


def v_strike(c, t, e, T):
    """Claw strike at a target point arriving at e["t_hit"]: wind-up (claw up and back, body leans back,
    head already on the target), strike (claw down to the target, body follows forward), contact hold."""
    limb = e.get("limb", "fleg_r")
    tgt = np.asarray(T[e["target"]] if isinstance(e["target"], str) else e["target"], float)
    t0, th = e["t"], e["t_hit"]
    th_end = th + e.get("hold", 0.6)
    if t < t0:
        return
    if t > th_end + 0.5:                              # the approach is permanent: it stays where it stepped to
        c["root_off"] += e.get("_approach", np.zeros(3))
        return
    c["look"] = tgt
    c["look_w"] = max(c["look_w"], 1.0)
    wind_end = t0 + 0.55 * (th - t0)
    if "_approach" not in e:                         # close the distance first: target within 85 % of the reach
        hip0 = T["_root0"] + (T["_rest"][limb + "_hip"] - T["_rest"]["pelvis"])
        reach_len = sum(T["_len"][limb + "_" + q] for q in ("knee", "ankle", "toe"))
        dh = (tgt - hip0) * np.array([1.0, 1.0, 0.0])
        need = max(0.0, float(np.linalg.norm(dh)) - 0.85 * reach_len) if e.get("approach") else 0.0
        e["_reach_short"] = max(0.0, float(np.linalg.norm(dh)) - 0.85 * reach_len)
        e["_approach"] = _n(dh) * need
    t_ap = t0 + (0.4 * (wind_end - t0) if np.linalg.norm(e["_approach"]) > 1e-3 else 0.0)
    c["root_off"] += e["_approach"] * smooth((t - t0) / max(t_ap - t0, 1e-3)) if t_ap > t0 else 0.0
    if t > wind_end:
        c["no_step"] = True
    if t <= t_ap:                                      # approach first: the feet step in, the claw stays down
        c["look"] = tgt
        return
    # weight shift onto the three standing legs: back and toward the opposite side, before the claw lifts,
    # held through the strike and the contact, released after
    side = 1.0 if limb.endswith("_l") else -1.0
    ws = np.array([-0.16, -side * 0.12, 0.0])
    k_ws = smooth((t - t0) / max(0.35 * (th - t0), 1e-3)) * (1 - smooth((t - th_end) / 0.5))
    c["root_off"] += ws * k_ws
    c["roll"] = c.get("roll", 0.0) + math.radians(-side * 4) * k_ws
    if t <= t0 + 0.35 * (th - t0) and t <= wind_end:  # the shift comes first; the claw stays planted
        return
    if t <= wind_end:                                  # wind-up: raise the claw above and behind the target
        u = smooth((t - max(t_ap, t0 + 0.35 * (th - t0))) / max(wind_end - max(t_ap, t0 + 0.35 * (th - t0)), 1e-3))
        c["reach"][limb] = (tgt + np.array([-0.35, 0.0, 0.75]), u)
        c["pitch"] += math.radians(10) * u
        c["root_off"] += np.array([-0.08, 0.0, 0.0]) * u
    elif t <= th:                                      # strike: accelerate onto the target (ease-in power)
        u = ((t - wind_end) / max(th - wind_end, 1e-3)) ** 2.2
        a = tgt + np.array([-0.35, 0.0, 0.75])
        c["reach"][limb] = (a + (tgt - a) * u, 1.0)
        c["pitch"] += math.radians(10 - 16 * u)
        c["root_off"] += np.array([-0.08 + 0.18 * u, 0.0, 0.0])
        c["events"].setdefault(f"{limb}_strike", {"t_hit": th})
    elif t <= th_end:                                  # on the target (the cleanup pins the contact)
        c["reach"][limb] = (tgt, 1.0)
        c["pitch"] += math.radians(-6)
        c["root_off"] += np.array([0.10, 0.0, 0.0])
    else:                                              # release
        u = 1 - smooth((t - th_end) / 0.5)
        c["reach"][limb] = (tgt + np.array([0.0, 0.0, 0.25]) * (1 - u), u)
    c["lift"] |= {limb} if c["reach"].get(limb, (None, 0))[1] > 0.05 else set()


def v_bite(c, t, e, T):
    tgt = np.asarray(T[e["target"]] if isinstance(e["target"], str) else e["target"], float)
    t0, th = e["t"], e["t_hit"]
    if t < t0 or t > th + 0.8:
        return
    if t <= th:
        u = smooth((t - t0) / max(th - t0, 1e-3))
        c["look"], c["look_w"] = tgt, 1.0
        c["head_reach"] = (tgt, u ** 1.5)
        c["jaw"] = max(c["jaw"], 40 * smooth(u * 1.4))
    else:
        c["look"], c["look_w"] = tgt, 1.0
        c["head_reach"] = (tgt, 1 - smooth((t - th) / 0.8))
        c["jaw"] = max(c["jaw"], 5.0)


def v_lunge(c, t, e, T):
    k = smooth((t - e["t"]) / max(e.get("dur", 0.6), 1e-3)) if t >= e["t"] else 0.0
    d = np.asarray(e.get("dir", [1, 0, 0]), float) * e.get("dist", 0.6)
    c["root_off"] += d * k
    c["head_off"] += np.array([0.25, 0.0, -0.1]) * _env(t, e["t"], e.get("dur", 0.6) + 0.4, 0.4, 0.5)
    c["pitch"] += math.radians(-8) * _env(t, e["t"], e.get("dur", 0.6) + 0.4, 0.4, 0.5)


def v_recoil(c, t, e, T):
    """Hit at e["t"] from direction "from" (the direction the blow travels): an impulse on the springs (kick)
    the first frame, and a short pose shift away from the blow; the springs give the follow-through."""
    d = _n(np.asarray(e.get("from", [-1, 0, 0]), float))
    s = e.get("strength", 1.0)
    if not e.get("_kicked") and t >= e["t"]:
        e["_kicked"] = True
        c["kicks"].append(("head", -0.0 * d + np.array([0, 0, 1.6]) * s + d * 2.2 * s))
        c["kicks"].append(("root", d * 0.9 * s))
        c["kicks"].append(("pitch", np.array(math.radians(55) * s)))
        c["kicks"].append(("wing", np.array([60, 120, 120]) * s))
        c["jaw_kick"] = 35 * s
    k = _env(t, e["t"], e.get("dur", 0.9), 0.08, 0.6)
    c["root_off"] += d * 0.15 * s * k
    c["pitch"] += math.radians(14) * s * k
    c["head_off"] += (d * 0.25 + np.array([0, 0, 0.18])) * s * k
    c["jaw"] = max(c["jaw"], 30 * k)
    c["wing"] += np.array([30, 60, 60]) * k


def v_roar(c, t, e, T):
    """Inhale (head draws back and up, chest rises), then the roar: neck extends forward-up, jaw wide,
    wings flare; the body braces against it."""
    t0, dur = e["t"], e.get("dur", 1.6)
    if t < t0 or t > t0 + dur + 0.5:
        return
    u = (t - t0) / dur
    inh = smooth(u / 0.3) * (1 - smooth((u - 0.3) / 0.15)) if u < 0.45 else 0.0
    out = _env(t, t0 + 0.3 * dur, 0.7 * dur + 0.3, 0.15, 0.35)
    c["head_off"] += np.array([-0.15, 0.0, 0.10]) * inh + np.array([0.30, 0.0, 0.22]) * out
    c["breath_amp"] += 0.03 * inh
    c["jaw"] = max(c["jaw"], 42 * out)
    c["wing"] += np.array([30, 80, 85]) * out
    c["crouch"] += 0.04 * out
    c["pitch"] += math.radians(6) * out
    c["shake"] = max(c["shake"], out)


def v_walk_to(c, t, e, T):
    tgt = np.asarray(T[e["target"]] if isinstance(e["target"], str) else e["target"], float)
    t0, dur = e["t"], e["dur"]
    u = smooth((t - t0) / dur) if t >= t0 else 0.0
    c["walk"] = {"from": np.asarray(e.get("_from", [0, 0, 0]), float), "to": tgt, "u": u,
                 "gait": e.get("gait", "walk"), "active": t0 <= t <= t0 + dur, "t0": t0, "dur": dur}


def v_fold(c, t, e, T):
    k = smooth((t - e["t"]) / max(e.get("dur", 0.5), 1e-3)) if t >= e["t"] else 0.0
    c["wing_override"] = (e.get("pose", "folded"), k)


def v_flap(c, t, e, T):
    """Wings beat (grounded or in the air): spread wings + a flap cycle (elevation +-amp at freq)."""
    k = _env(t, e["t"], e.get("dur", 2.0), 0.15, 0.25)
    if k <= 0:
        return
    c["wing_override"] = ("spread", max(k, (c["wing_override"] or (None, 0))[1]))
    c["flap"] = (e.get("amp", 40.0) * k, e.get("freq", 1.3))


def v_fly(c, t, e, T):
    """Flight along a path: the body near horizontal, all legs tucked, wings spread with a flap cycle (or a
    glide: slow shallow beats). e: from, to (template units, z = altitude of the pelvis), t, dur, glide."""
    t0, dur = e["t"], e.get("dur", 3.0)
    if t < t0 - 1e-6:
        return
    u = min(max((t - t0) / max(dur, 1e-3), 0.0), 1.0)
    a, b = np.asarray(e["from"], float), np.asarray(e["to"], float)
    c["fly"] = a + (b - a) * u
    c["no_upright"] = True
    c["pitch"] += math.radians(e.get("pitch", 6.0))
    c["lift"] |= {"fleg_l", "fleg_r", "hleg_l", "hleg_r"}
    c["curl"] = max(c["curl"], 0.9)
    glide = e.get("glide", False)
    c["wing_override"] = ("spread", 1.0)
    c["flap"] = (12.0, 0.45) if glide else (38.0, 1.1)
    d = b - a
    if np.linalg.norm(d[:2]) > 1e-6:
        c["fly_yaw"] = math.atan2(d[1], d[0])


def v_carry(c, t, e, T):
    """Flight (v_fly) while both fore claws hold a load below the chest; the claws' midpoint per frame is
    exported in the events ("carry": [[frame, x, y, z]]) to hang the carried figure from."""
    v_fly(c, t, e, T)
    c["carry_on"] = True
    c["carry_drop"] = e.get("drop", 0.55)


def v_fall(c, t, e, T):
    """Collapse: the trunk falls toward one side (roll) and down, the legs give way, the wings thrash and then
    stop, the jaw goes slack. Ends still."""
    t0, dur = e["t"], e.get("dur", 2.0)
    if t < t0:
        return
    u = min((t - t0) / dur, 1.0)
    k = smooth(u)
    c["no_upright"] = True
    side = e.get("side", 1.0)
    c["pitch"] += math.radians(62 if e.get("from_upright", True) else 0) * (1 - k) - math.radians(10) * k
    c["roll"] = c.get("roll", 0.0) + math.radians(75 * side) * k
    c["crouch"] += 0.55 * k
    c["lift"] |= {"fleg_l", "fleg_r"} if k < 0.6 else set()
    thrash = max(0.0, 1 - u / 0.8)
    ramp = smooth(u / 0.15)                       # the wings open over ~0.3 s, not in one frame
    c["wing_override"] = ("spread", (0.7 * thrash + 0.3) * ramp)
    c["flap"] = (55.0 * thrash * ramp, 2.2)
    c["jaw"] = max(c["jaw"], 18.0 * k)
    c["head_off"] += np.array([0.1, 0.0, -0.9]) * k


def v_tail(c, t, e, T):
    """Tail sweep: one swing to the side and back (amp deg)."""
    u = (t - e["t"]) / max(e.get("dur", 1.2), 1e-3)
    if 0 <= u <= 1:
        c["tail_sweep"] = e.get("amp", 70.0) * math.sin(math.pi * u) * e.get("side", 1.0)


def v_look(c, t, e, T):
    """Turn the head toward a target (absolute times)."""
    if e["t"] <= t <= e["t"] + e.get("dur", 2.0):
        c["look"] = np.asarray(T[e["target"]] if isinstance(e["target"], str) else e["target"], float)
        c["look_w"] = max(c["look_w"], e.get("w", 1.0))


def v_lower_for(c, t, e, T):
    """Bring a body part down to where a smaller opponent's blow can land: around e["t"] (the impact) the creature
    lowers by e["drop"] body lengths -- up to 0.3 by crouching, the rest by leaning the trunk forward (an upright
    creature comes down out of its stance)."""
    k = _env(t, e["t"] - e.get("lead", 0.6), e.get("lead", 0.6) + e.get("hold", 1.2), 0.45, 0.35)
    if k <= 0:
        return
    drop = float(e.get("drop", 0.0))
    cr = min(drop, 0.45)
    c["crouch"] += cr * k
    lean = max(0.0, drop - cr)                          # each 0.1 body length below the crouch ~ 9 deg of lean
    c["pitch"] -= math.radians(min(55.0, 90.0 * lean)) * k


def read_wav(path):
    """-> (mono float32 in [-1, 1], sample rate). stdlib wave (PCM 8/16/24/32) with a scipy fallback."""
    import wave
    try:
        with wave.open(path, "rb") as w:
            n, ch, sw, sr = w.getnframes(), w.getnchannels(), w.getsampwidth(), w.getframerate()
            raw = w.readframes(n)
        if sw == 3:
            b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
            x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            x = np.where(x >= 1 << 23, x - (1 << 24), x) / float(1 << 23)
        else:
            dt = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
            x = np.frombuffer(raw, dt).astype(np.float64)
            x = (x - 128) / 128 if sw == 1 else x / float(np.iinfo(dt).max)
        x = x.reshape(-1, ch).mean(1)
    except (wave.Error, KeyError):
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        x = np.asarray(x, np.float64)
        x = x.mean(1) if x.ndim > 1 else x
        if np.abs(x).max() > 1.5:
            x = x / 32768.0
    return x.astype(np.float32), sr


def speech_features(path, fps, offset_s=0.0, smooth_ms=50.0, lead_ms=40.0):
    """Per video frame: envelope (0..1, RMS over a 50 ms window, one-pole smoothed, 95th-percentile
    normalised), onset strength (0..1, positive envelope flux = syllable onsets), stressed (bool: strong
    onsets), and pauses [(t0, t1)] >= 0.3 s. Led by lead_ms so the springy jaw is not late."""
    x, sr = read_wav(path)
    hop = sr / fps
    win = int(0.05 * sr)
    n = int(len(x) / hop) + 1
    rms = np.zeros(n)
    for i in range(n):
        c = int(i * hop + lead_ms / 1000 * sr)
        a, b = max(0, c - win // 2), min(len(x), c + win // 2)
        rms[i] = float(np.sqrt(np.mean(x[a:b] ** 2))) if b > a else 0.0
    alpha = math.exp(-1.0 / (fps * smooth_ms / 1000.0))
    env = np.zeros(n)
    for i in range(n):
        env[i] = rms[i] if i == 0 else alpha * env[i - 1] + (1 - alpha) * rms[i]
    ref = np.percentile(env, 95) or 1e-6
    env = np.clip(env / ref, 0, 1)
    flux = np.maximum(np.diff(env, prepend=env[0]), 0)
    onset = np.clip(flux / (np.percentile(flux[flux > 0], 90) if np.any(flux > 0) else 1.0), 0, 1)
    stressed = np.zeros(n, bool)
    for i in range(1, n - 1):                          # local onset peaks that are strong = stressed syllables
        if onset[i] > 0.6 and onset[i] >= onset[i - 1] and onset[i] >= onset[i + 1]:
            stressed[i] = True
    quiet = env < 0.08
    pauses, i = [], 0
    while i < n:
        if quiet[i]:
            j = i
            while j < n and quiet[j]:
                j += 1
            if (j - i) / fps >= 0.3:
                pauses.append((offset_s + i / fps, offset_s + j / fps))
            i = j
        else:
            i += 1
    return {"env": env, "onset": onset, "stressed": stressed, "pauses": pauses, "fps": fps,
            "duration": len(x) / sr}


def v_speak(c, t, e, T):
    """Speech from a wav: the jaw follows the loudness, wider on syllable onsets; small head nods on stressed
    syllables, a neck sway with the phrase; in pauses a breath-in (chest) and a "speak_pauses" event (smoke/breath
    fx cue). e: {"wav": path, "t": start_s, "gain": 1.0, "audio_offset": 0.0 (skip into the wav)}."""
    if "_feat" not in e:
        e["_feat"] = speech_features(e["wav"], T.get("_fps", 24.0) if not isinstance(T.get("_fps"), np.ndarray) else float(T["_fps"]))
        e["_nodded"] = set()
    F = e["_feat"]
    i = int(round((t - e["t"] + e.get("audio_offset", 0.0)) * F["fps"]))
    c["events"].setdefault("speak_pauses", [(e["t"] + a - e.get("audio_offset", 0.0), e["t"] + b - e.get("audio_offset", 0.0))
                                            for a, b in F["pauses"]])
    if i < 0 or i >= len(F["env"]):
        return
    g = e.get("gain", 1.0)
    env, on = float(F["env"][i]), float(F["onset"][i])
    c["jaw"] = max(c["jaw"], g * (28.0 * env + 14.0 * on))       # deg, within the template's jaw limit
    c["look_w"] = max(c["look_w"], 0.0)
    c["head_off"] += np.array([0.03 * env, 0.0, 0.02 * env])      # leans into the phrase
    c["neck_sway"] = c.get("neck_sway", 0.0) + 0.03 * env * math.sin(2 * math.pi * 0.45 * t)
    if F["stressed"][i] and i not in e["_nodded"]:                # a nod: the head dips, the spring brings it back
        e["_nodded"].add(i)
        c["kicks"].append(("head", np.array([0.15, 0.0, -0.55]) * g))
    for a, b in F["pauses"]:                                      # breath in during a pause
        ta, tb = a - e.get("audio_offset", 0.0) + e["t"], b - e.get("audio_offset", 0.0) + e["t"]
        if ta <= t <= tb:
            u = (t - ta) / max(tb - ta, 1e-3)
            c["breath_amp"] += 0.03 * math.sin(math.pi * u)


def _stub(name):
    def f(c, t, e, T):
        raise NotImplementedError(f"creature verb '{name}' not implemented yet")
    return f


def v_dead(c, t, e, T):
    """Slain earlier (10-23): the end of a fall, held the whole shot."""
    v_fall(c, t, dict(e, t=-10.0, dur=0.01, from_upright=False), T)


VERBS = {"dead": v_dead, "lower_for": v_lower_for, "speak": v_speak, "flap": v_flap, "fly": v_fly, "carry": v_carry, "fall": v_fall, "tail": v_tail,
         "look": v_look, "idle": v_idle, "brace": v_brace, "rear": v_rear, "strike": v_strike, "bite": v_bite, "lunge": v_lunge,
         "recoil": v_recoil, "roar": v_roar, "walk_to": v_walk_to, "wings": v_fold,
         "take_off": _stub("take_off"), "land": _stub("land")}


# --------------------------------------------------------------------------------------------------
# the solver
# --------------------------------------------------------------------------------------------------

MASS = {"pelvis": 0.16, "spine1": 0.10, "spine2": 0.12, "spine3": 0.12, "chest": 0.12, "neck1": 0.03, "neck2": 0.03,
        "neck3": 0.025, "neck4": 0.02, "head": 0.05}


class Creature:
    def __init__(self, template: str):
        self.name = template
        self.T = load_template(template)
        self.rest = {k: np.asarray(v, float) for k, v in self.T["rest"].items()}
        self.par = self.T["parents"]
        self.joints = self.T["joints"]
        self.ch = self.T["chains"]
        self.lim = self.T.get("limits", {})
        self.legs = [k for k in self.ch if k.startswith("leg_")]
        self.wings = [k for k in self.ch if k.startswith("wing_") and not k.endswith("fingers")]
        self.len = {j: float(np.linalg.norm(self.rest[j] - self.rest[p])) for j, p in self.par.items() if p}

    # ---- trunk -------------------------------------------------------------------------------------
    def trunk(self, P, yaw, pitch, roll, crouch, breath):
        """Pelvis..chest + neck base + tail root + hips + wing shoulders from the root state. Returns
        positions and the body frame R (columns fwd, left, up)."""
        R = Rz(yaw) @ Ry(-pitch) @ Rx(roll)
        out = {}
        sp = self.ch["spine"]
        p0 = self.rest[sp[0]]
        out[sp[0]] = P.copy()
        cur, Rc = P.copy(), R.copy()
        n = len(sp) - 1
        for i in range(1, len(sp)):
            seg = self.rest[sp[i]] - self.rest[sp[i - 1]]
            Rc = Rc @ Ry(-pitch * 0.25 / n)               # a little extra arch distributed along the spine
            cur = cur + Rc @ seg
            out[sp[i]] = cur.copy()
        Rch = Rc
        if breath:                                          # chest rises/expands
            out[sp[-1]] = out[sp[-1]] + R @ np.array([0.0, 0.0, breath])
            out[sp[-2]] = out[sp[-2]] + R @ np.array([0.0, 0.0, 0.6 * breath])
        frames = {sp[0]: R, sp[-1]: Rch}
        for j in self.joints:                                # direct children of spine joints not on the spine
            p = self.par[j]
            if p in sp and j not in sp:
                base = R if p == sp[0] else Rch
                out[j] = out[p] + base @ (self.rest[j] - self.rest[p])
        return out, R, Rch

    # ---- chains ---------------------------------------------------------------------------------------
    def reach_chain(self, chain, base_pos, target, rest_dirs, Rbase, max_bend_deg, iters=12):
        """FABRIK from a fixed base toward target with per-joint bend clamps (deg away from straight)."""
        L = [self.len[chain[i]] for i in range(1, len(chain))]
        pts = [base_pos.copy()]
        for i in range(1, len(chain)):
            pts.append(pts[-1] + Rbase @ rest_dirs[i - 1] * L[i - 1])
        pts = np.array(pts)
        if target is None:
            return pts
        for _ in range(iters):
            pts[-1] = target
            for i in range(len(pts) - 2, -1, -1):
                pts[i] = pts[i + 1] + _n(pts[i] - pts[i + 1]) * L[i]
            pts[0] = base_pos
            for i in range(1, len(pts)):
                d = _n(pts[i] - pts[i - 1])
                prev = _n(pts[i - 1] - pts[i - 2]) if i >= 2 else _n(Rbase @ rest_dirs[0])
                ang = math.degrees(math.acos(np.clip(np.dot(d, prev), -1, 1)))
                if ang > max_bend_deg:                         # clamp the bend toward the previous direction
                    ax = np.cross(prev, d)
                    if np.linalg.norm(ax) > 1e-9:
                        d = rot_axis(ax, math.radians(max_bend_deg)) @ prev
                pts[i] = pts[i - 1] + d * L[i - 1]
        return pts

    def leg_ik(self, chain, hip, toe_target, R, lifted, curl=0.0):
        """hip-knee-ankle-toe: the toe at its target; ankle above/behind the toe; 2-bone IK, knee forward."""
        h, k, a, toe = chain
        fwd = R[:, 0]
        if lifted:                                            # a lifted foot points down-forward, curled
            foot = (Rz(math.atan2(fwd[1], fwd[0])) @ (self.rest[toe] - self.rest[a]))
            foot = rot_axis(R[:, 1], math.radians(-50 * curl)) @ foot
        else:                                                 # planted: the foot lies as at rest, turned with the body
            foot = Rz(math.atan2(fwd[1], fwd[0])) @ (self.rest[toe] - self.rest[a])
        ank = toe_target - foot
        l1, l2 = self.len[k], self.len[a]
        d = ank - hip
        dist = float(np.linalg.norm(d))
        dist_c = min(max(dist, abs(l1 - l2) + 1e-4), l1 + l2 - 1e-4)
        if dist > l1 + l2 - 1e-4:                             # out of reach: the leg straightens toward it
            ank = hip + _n(d) * dist_c
        dd = _n(ank - hip)
        cosA = (l1 * l1 + dist_c * dist_c - l2 * l2) / (2 * l1 * dist_c)
        A = math.acos(np.clip(cosA, -1, 1))
        pole = _n(fwd - dd * np.dot(fwd, dd))
        knee = hip + (dd * math.cos(A) + pole * math.sin(A)) * l1
        return {h: hip, k: knee, a: ank, toe: ank + foot}

    def wing(self, side, shoulder, R, elev, elbow, wrist, flap):
        lim = self.lim
        elev = float(np.clip(elev, *lim[f"wing_{side}_shoulder"]["elev"]))
        elbow = float(np.clip(elbow, *lim[f"wing_{side}_elbow"]["angle"]))
        wrist = float(np.clip(wrist, *lim[f"wing_{side}_wrist"]["angle"]))
        sgn = 1.0 if side == "l" else -1.0
        rs, re_, rw = (self.rest[f"wing_{side}_{p}"] for p in ("shoulder", "elbow", "wrist"))
        # humerus: out to the side, swept back a little, raised by elev (+ flap)
        e = math.radians(elev + flap)
        hum = R @ _n(np.array([-0.25, sgn * math.cos(e), math.sin(e)]))
        nrm = _n(np.cross(hum, R[:, 0])) * (1 if side == "l" else -1)      # wing plane normal (≈ up)
        el = shoulder + hum * self.len[f"wing_{side}_elbow"]
        # forearm: the humerus direction turned by (180 - elbow) IN THE WING PLANE toward the body's rear
        # (the axis is built from hum x rear, so both sides fold backward by construction)
        rear = -R[:, 0]
        if getattr(self, "upright", False):
            rear = _n(Rz(math.atan2(R[1, 0], R[0, 0])) @ np.array([-0.45, 0.0, 0.88]))
        ax = np.cross(hum, rear)
        if np.linalg.norm(ax) < 1e-6:
            ax = nrm
        fa = rot_axis(ax, math.radians(180 - elbow)) @ hum
        wr = el + _n(fa) * self.len[f"wing_{side}_wrist"]
        out = {f"wing_{side}_shoulder": shoulder, f"wing_{side}_elbow": el, f"wing_{side}_wrist": wr}
        # fingers/tip: rest offsets in the rest wrist frame, rotated into the posed forearm frame, then
        # closed toward the forearm by the wrist fold
        rest_fa = _n(rw - re_)
        rest_n = _n(np.cross(_n(re_ - rs), np.array([1.0, 0, 0]))) * (1 if side == "l" else -1)
        Brest = np.stack([rest_fa, np.cross(rest_n, rest_fa), rest_n], 1)
        Bpose = np.stack([_n(fa), np.cross(nrm, _n(fa)), nrm], 1)
        Mw = Bpose @ Brest.T
        for f in ("tip", "finger1", "finger2", "finger3"):
            j = f"wing_{side}_{f}"
            off = Mw @ (self.rest[j] - rw)
            # fold: rotate each finger toward the forearm's backward direction, more for the folded wing
            fold = (175 - wrist) / 175.0
            if getattr(self, "upright", False):           # upright: wings fold UP and back behind the shoulders
                Ry_ = Rz(math.atan2(R[1, 0], R[0, 0]))
                back = _n(Ry_ @ np.array([-0.45, sgn * 0.15, 0.88]))
            else:
                back = _n(rear + R[:, 1] * sgn * 0.12 + R[:, 2] * 0.05)   # folded fingers sweep back along the body
            o = _n(off)
            o2 = _n(o * (1 - fold) + back * fold)
            q = wr + o2 * np.linalg.norm(off) * (1 - 0.25 * fold)
            lat = np.dot(q - shoulder, R[:, 1]) * sgn            # never across the body's midline
            if lat < 0.10:
                q = q + R[:, 1] * sgn * (0.10 - lat)
            out[j] = q
        return out

    def _angle(self, a, b, c, **kw):
        return math.degrees(math.acos(np.clip(np.dot(_n(a - b), _n(c - b)), -1, 1)))

    # ---- centre of mass ----------------------------------------------------------------------------------
    def com(self, X):
        w, s = 0.0, np.zeros(3)
        for j, p in X.items():
            m = MASS.get(j, 0.012)
            s += m * p
            w += m
        return s / w


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _support_project(c, pts, margin=0.06):
    """Closest point to c (xy) inside the convex hull of pts (xy), shrunk by margin."""
    P = np.asarray(pts, float)[:, :2]
    if len(P) == 0:
        return c[:2]
    if len(P) == 1:
        return P[0]
    if len(P) == 2:
        a, b = P
        t = np.clip(np.dot(c[:2] - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        return a + t * (b - a)
    # hull (gift wrap, few points)
    hull = []
    start = int(np.argmin(P[:, 0]))
    cur = start
    while True:
        hull.append(cur)
        nxt = (cur + 1) % len(P)
        for i in range(len(P)):
            if _cross2(P[nxt] - P[cur], P[i] - P[cur]) < 0:
                nxt = i
        cur = nxt
        if cur == start or len(hull) > len(P):
            break
    H = P[hull]
    cen = H.mean(0)
    H = cen + (H - cen) * np.clip(1 - margin / max(np.linalg.norm(H - cen, axis=1).mean(), 1e-6), 0.2, 1)
    inside = all(_cross2(H[(i + 1) % len(H)] - H[i], c[:2] - H[i]) >= 0 for i in range(len(H)))
    if inside:
        return c[:2]
    best, bd = None, 1e9
    for i in range(len(H)):
        a, b = H[i], H[(i + 1) % len(H)]
        t = np.clip(np.dot(c[:2] - a, b - a) / max(np.dot(b - a, b - a), 1e-9), 0, 1)
        q = a + t * (b - a)
        dd = np.linalg.norm(q - c[:2])
        if dd < bd:
            best, bd = q, dd
    return best


def animate(template: str, verbs: list, dur: float, fps: float = 24, targets: dict | None = None,
            start: dict | None = None, seed: int = 0):
    """-> CanonicalMotion of the template. verbs: [{"verb", "t", ...}], times in seconds."""
    from motion_canonical import CanonicalMotion, mat_to_quat
    C = Creature(template)
    T = {k: np.asarray(v, float) for k, v in (targets or {}).items()}
    rng = np.random.default_rng(seed)
    N = int(round(dur * fps)) + 1 if dur * fps % 1 else int(round(dur * fps))
    dt = 1.0 / fps
    rest = C.rest
    root0 = np.asarray((start or {}).get("pelvis", rest["pelvis"]), float)
    yaw0 = float((start or {}).get("yaw", 0.0))
    posture = (start or {}).get("posture", "quadruped")
    C.upright = posture == "upright"
    T["_root0"], T["_rest"], T["_len"], T["_fps"] = root0, rest, C.len, float(fps)
    S = {k: Spring(v, *STIFF[s]) for k, v, s in [
        ("root", root0, "root"), ("yaw", yaw0, "yaw"), ("pitch", math.radians(62) if posture == "upright" else 0.0, "pitch"), ("roll", 0.0, "roll"),
        ("crouch", 0.0, "crouch"), ("head", rest["head"] - rest["pelvis"] + root0, "head"),
        ("jaw", 0.0, "jaw"), ("wing", np.array([0.0, 0.0, 0.0]), "wing"), ("breath", 0.0, "breath")]}
    reach = {}
    steps = {}
    plants = {}
    for lg in C.legs:
        toe = C.ch[lg][-1]
        plants[lg] = rest[toe] - rest["pelvis"] + root0
        plants[lg][2] = 0.0
    tail = C.ch.get("tail", [])
    tail_pos, tail_vel = None, None
    wing_tip_lag = {}
    frames = []
    evts = {}
    folded = C.T.get("wing_poses", {}).get("folded", {"elev": 10, "elbow": 25, "wrist": 15})
    for i in range(N):
        t = i * dt
        c = {"root_off": np.zeros(3), "pitch": 0.0, "crouch": 0.0, "head_off": np.zeros(3), "look": None,
             "look_w": 0.0, "jaw": 0.0, "wing": np.zeros(3), "stance": 0.0, "lift": set(), "curl": 0.0,
             "reach": {}, "head_reach": None, "kicks": [], "breath_amp": 0.0, "shake": 0.0, "events": evts,
             "walk": None, "wing_override": None, "jaw_kick": 0.0, "no_step": False,
             "flap": (0.0, 1.0), "fly": None, "no_upright": False, "tail_sweep": 0.0, "carry_on": False}
        for e in verbs:
            VERBS[e["verb"]](c, t, e, T)
        if posture == "upright" and not c["no_upright"]:   # rearing-biped stance (an upright-built mesh): trunk
            c["pitch"] += math.radians(62)     # up, fore legs carried as arms, the tail planted as the third support
            c["lift"] |= {"fleg_l", "fleg_r"}
            c["curl"] = max(c["curl"], 0.6)
        if c["carry_on"]:                      # both fore claws hold the load below the chest (exact)
            c["lift"] |= {"fleg_l", "fleg_r"}
        # --- channel targets -> springs (the head leads, the body follows)
        walk = c["walk"]
        base_root = root0.copy()
        if walk:
            base_root = walk["from"] + (walk["to"] - walk["from"]) * walk["u"]
            base_root[2] = root0[2]
        if c["fly"] is not None:
            base_root = np.asarray(c["fly"], float) + np.array([0, 0, 0.05 * math.sin(2 * math.pi * c["flap"][1] * t)])
        for name, dv in c["kicks"]:
            S[name].kick(dv)
        if i == 0 and c["fly"] is not None:   # a clip that starts in flight starts ON its path
            S["root"].x = base_root + c["root_off"]
        root = S["root"].step(base_root + c["root_off"] - np.array([0, 0, c["crouch"]]), dt)
        pitch = float(S["pitch"].step(c["pitch"], dt))
        yaw = float(S["yaw"].step(c.get("fly_yaw", yaw0), dt))
        roll = float(S["roll"].step(c.get("roll", 0.0), dt))
        breath = 0.012 * math.sin(2 * math.pi * 0.28 * t) + c["breath_amp"]
        X, R, Rch = C.trunk(root, yaw, pitch, roll, 0.0, breath)
        # head target: rest head relative to the chest, plus offsets; look/reach targets pull it
        # the neck counter-bends: the head keeps ~level when the trunk pitches (only 30 % of the pitch reaches it)
        Rlevel = Rz(yaw) @ Ry(-pitch * 0.3) @ Rx(roll)
        head_rest = X["chest"] + Rlevel @ (rest["head"] - rest["chest"])
        htgt = head_rest + R @ c["head_off"]
        if c["look"] is not None:                                # look: the head turns/leans toward the target
            dirv = _n(c["look"] - X["chest"])
            htgt = htgt + (X["chest"] + dirv * np.linalg.norm(head_rest - X["chest"]) - htgt) * 0.6 * c["look_w"]
        if c["head_reach"] is not None:
            tgt, w = c["head_reach"]
            htgt = htgt + (tgt - htgt) * w
        if c.get("neck_sway"):
            htgt = htgt + R[:, 1] * c["neck_sway"]
        if c["shake"] > 0:
            htgt = htgt + R @ np.array([0, 0.03 * math.sin(2 * math.pi * 7 * t), 0.02 * math.sin(2 * math.pi * 9 * t)]) * c["shake"]
        if i == 0:                                   # start the spring where the head is meant to be
            S["head"].x = htgt.copy()
        head = S["head"].step(htgt, dt)
        neck = C.ch["neck"]
        rest_dirs = [_n(rest[neck[k + 1]] - rest[neck[k]]) for k in range(len(neck) - 1)]
        rest_dirs = [Rch.T @ (Rch @ d) for d in rest_dirs]
        pts = C.reach_chain(neck, X["chest"], head, rest_dirs, Rch, 34)
        for k, j in enumerate(neck):
            X[j] = pts[k]
        hd = _n(X["head"] - X[neck[-2]])
        Rhead = np.stack([hd, _n(np.cross(UP, hd)), _n(np.cross(hd, _n(np.cross(UP, hd))))], 1)
        jaw_ang = math.radians(float(np.clip(S["jaw"].step(max(c["jaw"], c["jaw_kick"]), dt), *C.lim.get("jaw", {"open": [0, 45]})["open"])))
        jr = rest["jaw"] - rest["head"]
        X["jaw"] = X["head"] + Rhead @ (rot_axis([0, 1, 0], jaw_ang) @ jr)
        # --- legs: planted feet stay; lifted feet follow reach targets / curl; gait for walking
        if c["carry_on"]:
            cp = X["chest"] + R @ np.array([0.05, 0.0, -c["carry_drop"]])
            c["reach"]["fleg_l"] = (cp + R @ np.array([0, 0.12, 0]), 1.0)
            c["reach"]["fleg_r"] = (cp + R @ np.array([0, -0.12, 0]), 1.0)
            evts.setdefault("carry", []).append([i] + cp.round(4).tolist())
        reach_res = {}
        for lg in C.legs:
            chain = C.ch[lg]
            hip = X[chain[0]]
            side = 1 if lg.endswith("l") else -1
            name = chain[0].split("_")[0] + "_" + chain[0].split("_")[1]
            lifted = name in c["lift"]
            r = c["reach"].get(name)
            if walk and walk["active"]:
                phase_off = {"walk": {"leg_hl": 0.0, "leg_fl": 0.25, "leg_hr": 0.5, "leg_fr": 0.75},
                             "trot": {"leg_hl": 0.0, "leg_fr": 0.0, "leg_hr": 0.5, "leg_fl": 0.5}}[walk["gait"]].get(lg, 0)
                duty = 0.75 if walk["gait"] == "walk" else 0.5
                stride = 0.55
                dist = float(np.linalg.norm(walk["to"] - walk["from"]))
                cycles = max(dist / stride, 1.0)
                ph = ((t - walk["t0"]) / walk["dur"] * cycles + phase_off) % 1.0
                home = X[chain[0]] + R @ np.array([0.08, 0.0, 0.0])
                home[2] = 0.0
                if ph > duty:                                   # swing: arc to a point ahead of the hip
                    s = (ph - duty) / (1 - duty)
                    land = home + R @ np.array([stride * 0.5, 0, 0])
                    p = plants[lg] + (land - plants[lg]) * smooth(s)
                    p[2] = 0.12 * math.sin(math.pi * s)
                    toe_t = p
                    if s > 0.95:
                        plants[lg] = land
                else:
                    toe_t = plants[lg]
            elif r is not None and r[1] > 0.01:
                tgt, w = r
                sp = reach.setdefault(lg, Spring(plants[lg], *STIFF["reach"]))
                want = plants[lg] + (tgt - plants[lg]) * w
                if w >= 0.999:                                  # a contact: exact (the verb shapes its timing)
                    sp.x, sp.v = want.copy(), np.zeros(3)
                    toe_t = want
                else:
                    toe_t = sp.step(want, dt)
                lifted = True
                reach_res[lg] = toe_t
            elif lifted:
                curl_t = hip + R @ np.array([0.25, 0.0, -0.35]) + np.array([0, 0, 0.0])
                sp = reach.setdefault(lg, Spring(plants[lg], *STIFF["reach"]))
                toe_t = sp.step(curl_t, dt)
            else:
                if lg in reach:                                 # coming back down: re-plant under the hip
                    sp = reach[lg]
                    home = hip + R @ np.array([0.08, side * c["stance"], 0.0])
                    home[2] = 0.0
                    toe_t = sp.step(home, dt)
                    if toe_t[2] < 0.01 and np.linalg.norm(sp.v) < 0.2:
                        plants[lg] = toe_t.copy()
                        plants[lg][2] = 0.0
                        del reach[lg]
                else:
                    toe_t = plants[lg]
            X.update(C.leg_ik(chain, hip, toe_t, R, lifted, c["curl"]))
        # --- the body follows a reaching limb: if the claw cannot get to its target from here, the trunk moves
        #     toward it (planted feet stay) -- then the balance pass below keeps the centre of mass supported
        for lg, want in reach_res.items():
            chain = C.ch[lg]
            res = want - X[chain[-1]]
            if np.linalg.norm(res) > 0.01:
                shift = res * np.array([1.0, 1.0, 0.0])    # horizontal: raising the trunk would lift the feet
                for l2 in C.legs:                            # ...but never beyond what the planted legs allow
                    if l2 == lg or l2 in reach_res:
                        continue
                    ch2 = C.ch[l2]
                    maxr = 0.97 * (C.len[ch2[1]] + C.len[ch2[2]])
                    ank = plants[l2] - (Rz(math.atan2(R[1, 0], R[0, 0])) @ (rest[ch2[3]] - rest[ch2[2]]))
                    for _k in range(20):
                        if np.linalg.norm(ank - (X[ch2[0]] + shift)) <= maxr:
                            break
                        shift = shift * 0.85
                for j in list(X):                     # trunk, hips, neck, tail, wings move; planted feet stay
                    if not any(j in C.ch[l2][1:] for l2 in C.legs):
                        X[j] = X[j] + shift
                S["root"].x = S["root"].x + shift
                for l2 in C.legs:
                    ch2 = C.ch[l2]
                    tgt2 = want if l2 == lg else X[ch2[-1]]
                    X.update(C.leg_ik(ch2, X[ch2[0]], tgt2, R, l2 == lg or X[ch2[-1]][2] > 0.03, c["curl"]))
        # --- the legs carry the body: a planted leg that cannot reach its plant pulls the trunk down (crouch)
        for _ in range(3):
            deficit = 0.0
            for lg in C.legs:
                if lg in reach_res or lg in reach or lg in steps:
                    continue
                toe_j = C.ch[lg][-1]
                deficit = max(deficit, float(X[toe_j][2] - plants[lg][2]))
            if deficit < 0.003:
                break
            down = np.array([0.0, 0.0, -deficit * 1.05])
            for j in list(X):
                if not any(j in C.ch[l2][1:] for l2 in C.legs):
                    X[j] = X[j] + down
            S["root"].x = S["root"].x + down * 0.5
            for l2 in C.legs:
                ch2 = C.ch[l2]
                if l2 in reach_res:
                    X.update(C.leg_ik(ch2, X[ch2[0]], reach_res[l2], R, True, c["curl"]))
                elif l2 not in reach and l2 not in steps:
                    X.update(C.leg_ik(ch2, X[ch2[0]], plants[l2], R, False, 0.0))
        # --- stepping: a planted foot whose leg is stretched past 95 % of its reach steps toward under its hip
        # COM outside the support also asks for a step (the feet catch the body; the contact is not pulled back)
        sup_now = [X[C.ch[l][-1]] for l in C.legs if l not in reach_res and l not in reach and l not in steps]
        com_out = 0.0
        if sup_now:
            cm0 = C.com(X)
            com_out = float(np.linalg.norm(_support_project(cm0, sup_now, margin=0.0) - cm0[:2]))
        stretch = {}
        for lg in C.legs:
            ch_ = C.ch[lg]
            ank_need = plants[lg] - (Rz(math.atan2(R[1, 0], R[0, 0])) @ (rest[ch_[3]] - rest[ch_[2]]))
            stretch[lg] = np.linalg.norm(ank_need - X[ch_[0]]) / (C.len[ch_[1]] + C.len[ch_[2]])
        free = [l for l in C.legs if l not in reach_res and l not in reach]
        want_step = None
        if not steps and free and not c["no_step"]:          # one swing at a time
            cand = max(free, key=lambda l: stretch[l])
            if stretch[cand] > 0.95 or (com_out > 0.05 and stretch[cand] > 0.6):
                want_step = cand
        for lg in C.legs:
            if lg in reach_res or lg in reach:
                continue
            chain = C.ch[lg]
            st = steps.get(lg)
            if st is None:
                if lg == want_step:
                    land = X[chain[0]] + R @ np.array([0.10, 0.0, 0.0])
                    land[2] = 0.0
                    steps[lg] = {"from": plants[lg].copy(), "to": land, "t0": t, "dur": 0.3}
            st = steps.get(lg)
            if st is not None:
                u = min((t - st["t0"]) / st["dur"], 1.0)
                p = st["from"] + (st["to"] - st["from"]) * smooth(u)
                p[2] = 0.10 * math.sin(math.pi * u)
                X.update(C.leg_ik(chain, X[chain[0]], p, R, u < 1.0, 0.0))
                if u >= 1.0:
                    plants[lg] = st["to"].copy()
                    del steps[lg]
        # --- balance: centre of mass over the planted feet (+ the tail tip on the ground)
        for _ in range(0 if reach_res else 6):
            support = [X[C.ch[lg][-1]] for lg in C.legs if X[C.ch[lg][-1]][2] < 0.03 and lg not in reach_res]
            if tail and tail_pos is not None and tail_pos[-1][2] < 0.06:
                support.append(tail_pos[-1])
            if not support:
                break
            cm_ = C.com(X)
            q = _support_project(cm_, support)
            dxy = q - cm_[:2]
            if np.linalg.norm(dxy) < 1e-3:
                break
            shift = np.array([dxy[0], dxy[1], 0.0]) * 0.9
            for j in list(X):
                if not any(j in C.ch[lg][1:] for lg in C.legs):
                    X[j] = X[j] + shift
            S["root"].x = S["root"].x + shift * 0.5
            for lg in C.legs:
                chain = C.ch[lg]
                if lg in reach_res:                    # a contact limb stays exactly on its target
                    X.update(C.leg_ik(chain, X[chain[0]], reach_res[lg], R, True, c["curl"]))
                    continue
                toe_now = X[chain[-1]]
                lifted = toe_now[2] > 0.03
                X.update(C.leg_ik(chain, X[chain[0]], toe_now, R, lifted, c["curl"]))
        # --- tail: damped spring chain behind the pelvis, ground collision
        if tail:
            tgt = [X[tail[0]]]
            Rp = R
            for k in range(1, len(tail)):
                seg = rest[tail[k]] - rest[tail[k - 1]]
                if getattr(C, "upright", False) and not c["no_upright"]:   # an upright creature's tail curls aside
                    seg = Rz(math.radians(-12 * k)) @ seg
                if c["tail_sweep"]:
                    seg = Rz(math.radians(c["tail_sweep"] / len(tail))) @ seg
                tgt.append(tgt[-1] + Rp @ seg + R @ np.array([0, 0.025 * math.sin(2 * math.pi * 0.2 * t + k * 0.6), 0]))
            tgt = np.array(tgt)
            if tail_pos is None:
                tail_pos, tail_vel = tgt.copy(), np.zeros_like(tgt)
            for k in range(1, len(tail)):
                kk = (2 * math.pi * (2.4 - 0.18 * k)) ** 2
                a = kk * (tgt[k] - tail_pos[k]) - 2 * 0.55 * math.sqrt(kk) * tail_vel[k]
                tail_vel[k] += a * dt
                tail_pos[k] += tail_vel[k] * dt
            tail_pos[0] = X[tail[0]]
            for k in range(1, len(tail)):                         # lengths + ground
                L = C.len[tail[k]]
                tail_pos[k] = tail_pos[k - 1] + _n(tail_pos[k] - tail_pos[k - 1]) * L
                if tail_pos[k][2] < 0.03:
                    tail_pos[k][2] = 0.03
                    tail_vel[k][2] = max(tail_vel[k][2], 0)
            for k, j in enumerate(tail):
                X[j] = tail_pos[k].copy()
        # --- wings: channel (elev, elbow, wrist) + flap, limits, secondary lag on the tips
        if C.wings:
            wv = S["wing"].step(c["wing"], dt)
            ov = c["wing_override"]
            base = dict(folded)
            if ov:
                pose = C.T["wing_poses"][ov[0]]
                base = {k: folded[k] + (pose[k] - folded[k]) * ov[1] for k in folded}
            elev, elbow, wrist = base["elev"] + wv[0], base["elbow"] + wv[1], base["wrist"] + wv[2]
            for wn in C.wings:
                s = wn.split("_")[1]
                fl = c["flap"][0] * math.sin(2 * math.pi * c["flap"][1] * t)
                W = C.wing(s, X[f"wing_{s}_shoulder"], R, elev, elbow, wrist, fl)
                for j, p in W.items():
                    if j.endswith(("tip", "finger1", "finger2", "finger3")):
                        sp = wing_tip_lag.setdefault(j, Spring(p, 3.5, 0.55))
                        p2 = sp.step(p, dt)
                        L = np.linalg.norm(p - W[f"wing_{s}_wrist"])
                        W[j] = W[f"wing_{s}_wrist"] + _n(p2 - W[f"wing_{s}_wrist"]) * L
                X.update(W)
        for j in C.joints:                     # nothing below the ground (a fallen wing/leg lies ON it)
            if X[j][2] < 0.02:
                X[j] = X[j].copy()
                X[j][2] = 0.02 if not j.endswith("_toe") else max(X[j][2], 0.0)
        frames.append({j: X[j].copy() for j in C.joints})
    pos = np.array([[f[j] for j in C.joints] for f in frames])
    return CreatureMotion(template, C.joints, C.par, pos, fps, {"backend": "procedural", "verbs": _clean(verbs),
                                                               "events": evts, "posture": posture})


def _clean(verbs):
    out = []
    for e in verbs:
        out.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in e.items() if not k.startswith("_")})
    return out


# --------------------------------------------------------------------------------------------------
# output: positions are authoritative; per-bone world rotations derived with a twist reference
# --------------------------------------------------------------------------------------------------

class CreatureMotion:
    """CanonicalMotion for a creature template: the same .npz + .json contract (format canonical_motion/1,
    "template": name, joints/parents from the template). pos (F, J, 3) is authoritative (template units);
    bone_rot (F, B, 3, 3) = world rotation of each bone (parent -> child) relative to its rest, twist from
    the body up (trunk/neck/tail/legs) or the wing plane (wings); rot (F, J, 4) = local quats (primary child)."""

    def __init__(self, template, joints, parents, pos, fps, meta):
        self.template, self.joints, self.parents = template, list(joints), dict(parents)
        self.pos, self.fps, self.meta = np.asarray(pos, float), float(fps), dict(meta)
        self.rest = {k: np.asarray(v, float) for k, v in load_template(template)["rest"].items()}
        self.bones = [(p, j) for j, p in self.parents.items() if p]
        self.bone_rot = self._bone_rots()

    @property
    def frames(self):
        return len(self.pos)

    def _bone_rots(self):
        J = {j: i for i, j in enumerate(self.joints)}
        F = self.frames
        out = np.zeros((F, len(self.bones), 3, 3))
        for b, (p, j) in enumerate(self.bones):
            d0 = _n(self.rest[j] - self.rest[p])
            wingb = j.startswith("wing_")
            for f in range(F):
                d1 = _n(self.pos[f, J[j]] - self.pos[f, J[p]])
                if wingb:
                    s = j.split("_")[1]
                    a, bb, cc = (self.pos[f, J[f"wing_{s}_{q}"]] for q in ("shoulder", "elbow", "wrist"))
                    up1 = _n(np.cross(bb - a, cc - bb)) * (1 if s == "l" else -1)
                    ra, rb, rc = (self.rest[f"wing_{s}_{q}"] for q in ("shoulder", "elbow", "wrist"))
                    up0 = _n(np.cross(rb - ra, rc - rb)) * (1 if s == "l" else -1)
                    if np.linalg.norm(up1) < 1e-6:
                        up1 = up0
                else:
                    pe, ch = self.pos[f, J["pelvis"]], self.pos[f, J[next(iter(k for k in ("chest", "spine3", "body4", "head") if k in J))]]
                    fwd = _n(ch - pe) if "chest" in J else np.array([1.0, 0, 0])
                    left = _n(np.cross(UP, fwd))
                    up1 = _n(np.cross(fwd, left))
                    up0 = UP
                B0 = np.stack([d0, _n(np.cross(up0, d0)), _n(np.cross(d0, _n(np.cross(up0, d0))))], 1)
                B1 = np.stack([d1, _n(np.cross(up1, d1)), _n(np.cross(d1, _n(np.cross(up1, d1))))], 1)
                out[f, b] = B1 @ B0.T
        return out

    def anchors(self, f):
        T = load_template(self.template)
        J = {j: i for i, j in enumerate(self.joints)}
        bidx = {j: k for k, (p, j) in enumerate(self.bones)}
        out = {}
        for name, a in T["anchors"].items():
            j = a["joint"]
            R = self.bone_rot[f, bidx[j]] if j in bidx else np.eye(3)
            out[name] = self.pos[f, J[j]] + R @ np.asarray(a["offset"], float)
        return out

    def header(self):
        return {"format": "canonical_motion/1", "template": self.template, "fps": self.fps, "frames": self.frames,
                "duration_s": round(self.frames / self.fps, 4), "joints": self.joints,
                "parents": [self.joints.index(self.parents[j]) if self.parents[j] else -1 for j in self.joints],
                "bones": [list(b) for b in self.bones],
                "frame": "template units (body length 1), Z up, +X forward at frame 0, +Y left, z=0 ground",
                "arrays": {"pos": "(F,J,3) authoritative", "bone_rot": "(F,B,3,3) world, vs rest"},
                "source": self.meta}

    def save(self, base):
        import json
        base = base[:-4] if base.endswith(".npz") else base
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        np.savez_compressed(base + ".npz", pos=self.pos.astype(np.float32), bone_rot=self.bone_rot.astype(np.float32),
                            fps=np.float32(self.fps))
        with open(base + ".json", "w") as f:
            json.dump(self.header(), f, indent=1, default=lambda o: np.asarray(o).tolist())
        return base + ".npz"

    def place(self, body_length_m, at=(0.0, 0.0), facing_deg=0.0):
        """World joints for the shot: scaled by the asset's body length, turned to its facing, moved to at."""
        R = Rz(math.radians(facing_deg))
        P = self.pos * body_length_m
        P = P @ R.T + np.array([at[0], at[1], 0.0])
        return P, np.einsum("ab,fkbc->fkac", R, self.bone_rot)
