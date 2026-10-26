"""Motion sources: one interface, several backends (layer 3 of PLAN-staging-engine-2026-10-19.md).

    from motion_source import motion, motions
    cm = motion(character, "raises a hammer overhead and strikes down hard", dur=3.0, fps=24)
    cms = motions([(character, text, dur), ...], fps=24)      # batched: the model loads once

character: an asset-registry entry (dict), e.g. {"name": ..., "height_m": 1.8, "rig": ".../rig.json",
           "kind": "human" | "biped" | "quadruped" | "winged" | "serpent", "motion": backend (optional)}.
           Only "kind"/"motion" choose the backend here; the motion itself is character-independent
           (CanonicalMotion, tools/motion_canonical.py). tools/motion_retarget.py fits it to the rig and
           scales it to height_m.
Returns CanonicalMotion at `fps`, facing +X at frame 0, pelvis at the origin, standing on z = 0.

Backends (BACKENDS):
    hymotion   text-to-motion, Tencent HY-Motion 1.0 (default for humans). Runs in ~/tts-ab/.venv-hymotion
               (tools/motion_hymotion.py): text encoders on the CPU (no GPU lock), the DiT on one XTX
               inside flock /tmp/gpu-<bus>.lock. Results are cached by (text, dur, seed, fps, model, cfg).
    mocap      clip library fallback                      -- stub
    procedural creature templates (rear, strike, lunge, fly, walk) -- stub
    interp     legacy start -> end poses (the 2D pose library) -- stub
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from motion_canonical import CanonicalMotion  # noqa: E402

HYM_PY = os.path.expanduser(os.environ.get("HYMOTION_PY", "~/tts-ab/.venv-hymotion/bin/python"))
HYM_HOME = os.path.expanduser(os.environ.get("HYMOTION_HOME", "~/models/hy-motion"))
HYM_MODEL = os.path.join(HYM_HOME, "tencent", os.environ.get("HYMOTION_MODEL", "HY-Motion-1.0-Lite"))
CACHE = os.path.expanduser(os.environ.get("MOTION_CACHE", "~/models/hy-motion/motion_cache"))
GPUS = os.environ.get("MOTION_GPUS", "05,65").split(",")      # PCI buses of the cards we may use
HOLD_S = 840                                                   # <= 14 min per GPU hold
MIN_FREE_GB = 15

HUMAN_KINDS = {"human", "person", None}
# anchored to the real command lines (script as the interpreter's first argument): a shell whose command
# TEXT merely mentions gaprun.sh (another agent's pipeline) must not count -- the pgrep self-match trap
GAPRUN = r"^\S*bash \S*gaprun\.sh "
LTXRUN = r"^\S*python\S* \S*ltx_split\.py "


def _log(*a):
    print("[motion]", *a, file=sys.stderr)


def motion_text(action: str) -> str:
    """HY-Motion was trained on 'A person ...' sentences describing the body only; make the action one.
    Keeps it short (the model works best under ~30 words)."""
    t = " ".join(action.strip().split())
    t = t[0].lower() + t[1:] if t else t
    if not re.match(r"^(a|an|the|he|she|they)\b", t, re.I):     # no subject yet: the planner's verb phrase
        t = "A person " + t
    t = t[0].upper() + t[1:]
    words = t.split()
    if len(words) > 30:
        _log(f"action has {len(words)} words; HY-Motion is tuned for < 30 (truncating)")
        t = " ".join(words[:30])
    return t if t.endswith(".") else t + "."


# --------------------------------------------------------------------------------------------------
# resources: memory + GPU lock
# --------------------------------------------------------------------------------------------------

def mem_available_gb() -> float:
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable"):
            return int(ln.split()[1]) / 2**20
    return 0.0


def wait_mem(need=MIN_FREE_GB, timeout=1800):
    t0 = time.time()
    while mem_available_gb() < need:
        if time.time() - t0 > timeout:
            raise RuntimeError(f"MemAvailable stayed below {need} GB")
        time.sleep(10)


class GpuLock:
    """flock /tmp/gpu-<bus>.lock on the first free card in GPUS (blocking on the first if all are busy)."""

    def __init__(self, buses=GPUS, wait_s=3600):
        self.buses, self.wait_s, self.fd, self.bus = buses, wait_s, None, None

    @staticmethod
    def _running(pat) -> bool:
        return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0

    def yield_to_both_card_jobs(self, max_s=900):
        """Fairness (coordinator rule): a job waiting for BOTH cards (gaprun.sh) goes first; wait until it
        has started (ltx_split.py) and finished, or max_s."""
        if not self._running(GAPRUN):
            return
        _log("a both-cards job is waiting (gaprun.sh): yielding")
        t0, started = time.time(), False
        while time.time() - t0 < max_s:
            if self._running(LTXRUN):
                started = True
            elif started or not self._running(GAPRUN):
                break
            time.sleep(10)

    def __enter__(self):
        self.yield_to_both_card_jobs()
        t0 = time.time()
        while True:
            for b in self.buses:
                fd = open(f"/tmp/gpu-{b}.lock", "w")
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.fd, self.bus = fd, b
                    return self
                except BlockingIOError:
                    fd.close()
            if time.time() - t0 > 30:
                # all busy: QUEUE on the first card with a blocking flock (polling loses every gap to the
                # blocking waiters of back-to-back both-card jobs)
                fd = open(f"/tmp/gpu-{self.buses[0]}.lock", "w")
                _log(f"queueing on /tmp/gpu-{self.buses[0]}.lock")
                fcntl.flock(fd, fcntl.LOCK_EX)
                self.fd, self.bus = fd, self.buses[0]
                return self
            time.sleep(5)

    def __exit__(self, *a):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()

    @property
    def hip_index(self) -> str:
        return subprocess.check_output([os.path.expanduser("~/bin/hip-index"), self.bus], text=True).strip()


def kernel_trouble(since="-15min") -> list:
    """amdgpu/kernel lines that mean STOP (ring resets, oopses, lockups)."""
    try:
        out = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager"], capture_output=True,
                             text=True, timeout=30).stdout
    except Exception:
        return []
    return [ln for ln in out.splitlines() if re.search(r"\bbug\b|oops|lockup|reset|\bring\b", ln, re.I)]


# --------------------------------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------------------------------

def _key(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def hymotion_batch(reqs: list, fps: float, seed: int = 0, cfg: float = 5.0, steps: int = 50,
                   model: str = HYM_MODEL, te_device: str = "cpu", raw_text: bool = False,
                   te_mode: str = os.environ.get("HYMOTION_TE", "int8")) -> list:
    """reqs: [(action, dur_s)] -> [CanonicalMotion]. One encode pass + one GPU hold for the whole batch."""
    os.makedirs(CACHE, exist_ok=True)
    jobs, outs = [], []
    for action, dur in reqs:
        text = action if raw_text else motion_text(action)
        dur = float(min(max(dur, 0.7), 12.0))        # the model's window: 20..360 frames at 30 fps
        base = os.path.join(CACHE, _key("hymotion", os.path.basename(model), text, round(dur, 3), seed, fps, cfg,
                                        steps, *([] if te_mode == "ref" else [te_mode])))
        outs.append(base)
        if not os.path.exists(base + ".npz") and not any(j["out"] == base for j in jobs):   # each distinct clip once
            jobs.append({"text": text, "dur": dur, "seed": seed, "fps": fps, "out": base})
    for c0 in range(0, len(jobs), CHUNK):           # <= CHUNK generations per GPU hold (14-min cap)
        _hymotion_run(jobs[c0:c0 + CHUNK], model, te_device, te_mode, steps, cfg)
    return [CanonicalMotion.load(b) for b in outs]


CHUNK = int(os.environ.get("HYMOTION_CHUNK", "100"))


def _hymotion_run(jobs, model, te_device, te_mode, steps, cfg):
    if jobs:
        spec = {"model": model, "te_device": te_device, "te_mode": te_mode, "steps": steps, "cfg": cfg, "jobs": jobs}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(spec, f)
        script = os.path.join(TOOLS, "motion_hymotion.py")
        wait_mem(MIN_FREE_GB + 17)                  # Qwen3-8B bf16 is ~16 GB of RAM
        subprocess.run([HYM_PY, script, "encode", f.name], check=True)
        if os.environ.get("HYMOTION_DEVICE", "") == "cpu":       # CPU: no card, no lock (slower, ~1 min a clip)
            env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
            _log(f"generate {len(jobs)} on the CPU")
            subprocess.run([HYM_PY, script, "generate", f.name], check=True, env=env)
        else:
            with GpuLock() as g:
                wait_mem()
                env = dict(os.environ, HIP_VISIBLE_DEVICES=g.hip_index)
                _log(f"generate {len(jobs)} on XTX {g.bus} (HIP {env['HIP_VISIBLE_DEVICES']})")
                subprocess.run(["timeout", str(HOLD_S), HYM_PY, script, "generate", f.name], check=True, env=env)
        bad = kernel_trouble()
        if bad:
            raise RuntimeError("kernel trouble after the GPU job:\n" + "\n".join(bad[-10:]))
        os.unlink(f.name)


def _stub(name, what):
    def f(reqs, fps, **kw):
        raise NotImplementedError(f"motion backend '{name}' is a stub: {what}")
    return f


BACKENDS = {
    "hymotion": hymotion_batch,
    # A library of retargetable clips (CMU/AMASS-style BVH -> CanonicalMotion), chosen by text similarity.
    "mocap": _stub("mocap", "clip library lookup -> CanonicalMotion"),
    # Creature templates (tools/stage_templates/creatures.json): tools/motion_creature.py, mechanics-driven.
    "procedural": None,
    # The legacy path: the 2D pose library's start/end poses lifted to 3D and interpolated.
    "interp": _stub("interp", "blockout_rig start -> end poses, slerped"),
}


def procedural_batch(reqs: list, fps: float, template: str = "winged_quadruped", **kw) -> list:
    """reqs: [(action, dur_s)] where action = {"verbs": [...], "targets": {...}} (motion_creature.animate)."""
    from motion_creature import animate
    out = []
    for action, dur in reqs:
        a = action if isinstance(action, dict) else {"verbs": [{"verb": "idle", "t": 0.0}]}
        out.append(animate(a.get("template", template), a["verbs"], dur, fps, a.get("targets"), a.get("start")))
    return out


BACKENDS["procedural"] = procedural_batch
KIND_TEMPLATE = {"winged": "winged_quadruped", "winged_quadruped": "winged_quadruped", "dragon": "winged_quadruped",
                 "quadruped": "quadruped", "biped": "biped_creature", "biped_creature": "biped_creature",
                 "serpent": "serpent"}


def pick_backend(character: dict | None) -> str:
    c = character or {}
    if c.get("motion"):
        return c["motion"]
    return "hymotion" if c.get("kind") in HUMAN_KINDS else "procedural"


def motions(requests: list, fps: float = 24, **kw) -> list:
    """requests: [(character, action, dur_s)] -> [CanonicalMotion], batched per backend."""
    groups = {}
    for i, (ch, action, dur) in enumerate(requests):
        groups.setdefault(pick_backend(ch), []).append((i, action, dur))
    out = [None] * len(requests)
    for be, items in groups.items():
        kw2 = dict(kw)
        if be == "procedural":
            ch = requests[items[0][0]][0] or {}
            kw2 = {"template": KIND_TEMPLATE.get(ch.get("kind"), ch.get("template", "winged_quadruped"))}
        res = BACKENDS[be]([(a, d) for _, a, d in items], fps, **kw2)
        for (i, _, _), cm in zip(items, res):
            out[i] = cm
    return out


def motion(character: dict | None, action: str, dur: float, fps: float = 24, **kw) -> CanonicalMotion:
    return motions([(character, action, dur)], fps, **kw)[0]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="text -> CanonicalMotion (.npz + .json)")
    ap.add_argument("action")
    ap.add_argument("--dur", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--raw", action="store_true", help="send the text unchanged (no 'A person' rewrite)")
    ap.add_argument("--out", help="copy the result to OUT(.npz/.json)")
    a = ap.parse_args()
    cm = motion({"kind": "human"}, a.action, a.dur, a.fps, seed=a.seed, raw_text=a.raw)
    if a.out:
        cm.save(a.out)
    print(json.dumps(cm.header()["events"]), cm.frames, "frames")
