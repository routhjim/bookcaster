"""Remake a chapter's animated moments with the LTX pipeline and assemble a NEW video (any book, any chapter).

    python3 tools/remake_chapter.py BOOK_DIR CHAPTER_STEM PHASE [--only id,id] [--dry-run]

PHASE (in order; every phase is resumable: finished outputs are skipped, each entry logs to OUT/log.jsonl):
  plan      route per manifest entry, frames/chunks, counts, GPU-time estimate -> OUT/plan.json (CPU, no models)
  check     face/age check of every first frame (tools/face_likeness.py, CPU) -> OUT/checks/<id>.json
  redraw    Qwen redraw of the frames that failed the check (headshot ref first + explicit age wording,
            img2img 0.6, N seeds, best by face_sim+wrinkles; char LoRAs when <book>/work/video/charlora/<slug>/lora
            has one) -> OUT/keys/<id>.png  (one card, flock /tmp/gpu-<card>.lock)
  stage     route 2 (full chain): motion_batch -> motion_batch --render (blockout_video --target ltx --stills) ->
            keyframe_draw --list; any failure demotes the entry to route 3 (logged)
  jobs      LTX job + prompt-encoding files under OUT/ltx/
  ltx       runs the LTX steps through the gap runner (both cards, <= 15-min holds): encode talk (--dev),
            gen talk (A2Vid), stitch chunked lines, encode action, gen action (IC-LoRA pipeline: Mode 2 control or
            plain I2V); loops until every clip exists or a step makes no progress; kernel-log check after each
  assemble  new manifest (OUT/anim/manifest.json, out -> the new clips; entries without a new clip keep the old
            one, flagged) + video.build() with the chapter's audio/timeline/pictures -> <book>/out/<book>_<stem>_<MMDD-HHMM>_ltx.mp4
  sheet     old-vs-new contact sheet (one frame per entry, < 3 MB) -> OUT/compare_sheet.jpg

Routes (per manifest entry):
  talk   LTX A2Vid: <= 5 s chunks (121 frames) split at the quietest audio, the camera lock on every chunk, frames
         rounded UP to 8k+1 + audio padded; non-human speakers (look.json: creature words / "no human features")
         get jaw wording + modality 5. First frame = the entry's existing first frame (checked, redrawn on fail).
  stage  action whose shot the staging engine can stage cleanly: every figure placed by the planner (2D pose boxes,
         or a per-book staged shot <book>/work/video/stage/shots/<shot id>.json), every figure has a rig, the
         location has a set in stage/assets.json "sets". -> LTX Mode 2, full depth control at attn 0.35.
  i2v    other action: LTX image-to-video (IC-LoRA pipeline without control) from the checked/redrawn keyframe, the
         shot prompt + motion text, static-camera wording. "continue" moments chain from the new previous clip's
         last shown frame.
Nothing here is story-specific: names, places and looks come from the book's plan/keyframes/look/assets data.
Never writes into the original chapter dir; the old video is never touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
HERE = Path(__file__).resolve().parent
TTS = HERE.parent
sys.path.insert(0, str(TTS))
sys.path.insert(0, str(HERE))

FPS = 24
# talk: 1152x640 in full 3D (10-20: at 832x448 the speaker's mouth was ~10 px and the armour shimmered)
TALK_W, TALK_H = map(int, os.environ.get("REMAKE_TALK_SIZE", "1152x640" if os.environ.get("REMAKE_FULL3D") else "832x448").split("x"))
TALK_CHAIN = os.environ.get("REMAKE_TALK_CHAIN", "0") == "1"       # off: every chunk from the key + camera lock (10-20)
TALK_FACE_ON = float(os.environ.get("REMAKE_TALK_FACE_ON", "15"))     # talk singles: camera degrees off the face's direction
TALK_SINGLE = os.environ.get("REMAKE_TALK_SINGLE", "1") == "1"     # A2Vid on a close single of the speaker (face >= 25 %)
ACT_W, ACT_H = 1152, 640
TALK_MAX = 121               # <= 5 s chunks
TALK_OVERLAP = 12
TALK_TAIL = 0.25
TALK_LOCK = 0.35
CHAIN_STRENGTH = 0.7        # a chained talk chunk's start frame (was 1.0 + an end lock: frozen lips)
TALK_LOCK_SHORT = 0.5        # chunks < 97 frames drift more (LTX agent: 73-frame tail 0.906 at 0.35)
CREATURE_MODALITY = 5.0
MODE2_ATTN = 0.35
NEG_EXTRA = (", text, letters, words, captions, subtitles, writing, sign, tablet, scroll, banner, watermark, logo, "
             "flickering, shimmering highlights, sparkling metal, camera zoom, push in")
END_GUIDE = 0.6              # the END keyframe of an authored camera move, as a guide at the last frame
ACT_TAIL = 0.0               # build() holds the last frame if a clip is short; 121 frames cover a 5.0 s slot
LTX_DIR = TTS / "eval" / "ltx"
RUNNER_DIR = HERE / "ltx_runner"
# The measured-best two-XTX LTX setup, the DEFAULT for every gen whatever flow calls ltx_step (user 10-26: "all future
# runs use the most optimal ltx by default"): ltx_split_next.py --pp (process per card), two lanes sharing card A,
# lanes 70 s out of step (~12%/frame, 10-24), pooled pinned buffers between the cards, fa_amd attention (-10% e2e) +
# hipBLASLt (-12%), VAE decode on card 65 for EVERY route (A/B 10-24 on quiet: frames identical, 58.3 -> 52.5 s; user
# 10-26: all clip types), and the per-route layer split pp_fit fits from this chapter's timing, else the book's newest
# chapter (flow3d._pp_split), with the decode counted on card 65 so the split moves blocks back to card 05. REMAKE_LTX_PP=0 = the old one-process ltx_split.py;
# REMAKE_LTX_PP_ARGS replaces LTX_PP_ARGS.
PP = os.environ.get("REMAKE_LTX_PP", "1") != "0"
LTX_PP_ARGS = "--lanes 2 --pp-shared-a --pp-pool --decode-b"
# lane 2 starts half a clip late (10-26 user: per clip type; 70 s was half a two-stage action clip, more than a whole
# ~45 s quiet one): half the route's median clip wall time from pp_timing (this chapter, else the book's newest),
# else STAGGER_DEFAULT
STAGGER_DEFAULT = {"a2vid": 100, "ti2v": 60, "ic": 60, "distilled": 22}
LTX_GEN_ARGS = ("--fa-amd", "--blaslt")
SPLIT_DEFAULT = {"a2vid": 25, "ti2v": 24, "ic": 17, "distilled": 17}
LTX_PY = HOME / "tts-ab" / ".venv-ltx" / "bin" / "python"
POSE_PY = HOME / "tts-ab" / ".venv-pose" / "bin" / "python"
QWEN_PY = HOME / "tts-ab" / ".venv-qwenimage" / "bin" / "python"
IC_LORA = HOME / "models" / "ltx-2.5" / "loras" / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
# an optional GPU-gap runner (yields the cards to waiting jobs); unset = plain blocking flocks (10-21: the old default
# pointed at another session's scratchpad, long gone)
GAPRUN = Path(os.environ.get("GAPRUN", "/nonexistent/gaprun.sh"))
MODELS = Path(os.environ.get("BLOCKOUT_MODELS", TTS / "eval" / "blockout" / "models"))
CREATURE_WORDS = ("dragon", "wyrm", "creature", "beast", "monster", "no human features")
STATIC_CAM = ("Static camera, the framing does not change; no pan, tilt or zoom. Each person has exactly two arms, "
              "two hands and two legs.")
# measured on 2x XTX (eval/ltx logs, fa_amd + hipBLASLt): seconds
T_TALK_LOAD, T_TALK_PER_FRAME = 180.0, 1.33           # 97-105 frames ~130 s
T_ACT_LOAD, T_ACT_CLIP = 100.0, 62.0                 # 1152x640x121
T_ENC_LOAD, T_ENC_PROMPT = 55.0, 2.0
HOLD_TIMEOUT = 870
HOLD_BUDGET = {"talk": 600.0, "act": 650.0}         # stop starting new jobs after this (timeout 870 s = < 15 min)


# --------------------------------------------------------------------------------------------- helpers

ALERT = None     # the bookcaster job's log (flow3d sets it): a failure is reported there the moment it happens


def log_entry(out: Path, rec: dict) -> None:
    rec = {"time": time.strftime("%H:%M:%S"), **rec}
    with open(out / "log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)
    if ALERT and (rec.get("ok") is False or rec.get("error") or rec.get("rc") not in (None, 0)):
        ALERT(f"  [3d] !!! {rec.get('phase')} {rec.get('id', '')} failed: "
              f"{str(rec.get('why') or rec.get('error') or 'rc ' + str(rec.get('rc')))[:300]}")


def frames_for(dur: float, tail: float = 0.0) -> int:
    need = math.ceil(round((dur + tail) * FPS, 3))
    return 8 * math.ceil(max(need - 1, 8) / 8) + 1


def is_creature(desc: str) -> bool:
    d = (desc or "").lower()
    return any(w in d for w in CREATURE_WORDS)


def age_of(desc: str) -> float | None:
    m = re.search(r"\b(?:of|about|aged?)\s+(?:about\s+)?(\d{1,3})\b", desc or "")
    return float(m.group(1)) if m else None


def gender_of(desc: str) -> str:
    d = (desc or "").lower()
    if re.search(r"\b(woman|girl|lady|queen|princess|she|her)\b", d):
        return "woman"
    if re.search(r"\b(man|boy|youth|lad|king|he|his)\b", d):
        return "man"
    return "person"


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


CANON_KEY: dict = {}     # canon(name) -> the first-stored look.json key (set by Ctx): one sheet per person


def ref_sheet(refs: Path, name: str) -> Path | None:
    from tts_reader.video import canon
    name = CANON_KEY.get(canon(name), name)
    c = sorted(refs.glob(f"{slug(name)}-*.png"), key=lambda p: p.stat().st_mtime)
    return c[-1] if c else None


def audio_env(wav: Path, n: int):
    import numpy as np
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    au = np.frombuffer(raw, np.float32)
    hop = 16000 // FPS
    return np.array([np.sqrt(np.mean(au[i * hop:(i + 1) * hop] ** 2)) if (i + 1) * hop <= len(au) else 0.0
                     for i in range(n)])


def probe_dur(p: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout.strip() or 0)


def talk_chunks(total: int, env, mx: int = TALK_MAX, overlap: int = TALK_OVERLAP) -> list[tuple[int, int]]:
    """chunk k covers [s, s+n), n = 8m+1 <= mx; the overlap window lands on the quietest frames. overlap 1 = each chunk
    starts ON the previous one's last frame (the shots flow's chained pieces, 10-26)."""
    chunks, s = [], 0
    while True:
        rest = 8 * math.ceil((total - s - 1) / 8) + 1
        if rest <= mx:
            chunks.append((s, rest))
            return chunks
        cands = [n for n in range(mx - 24, mx + 1) if (n - 1) % 8 == 0 and n > overlap + 16]
        n = min(cands, key=lambda n: float(env[s + n - max(overlap, 3):s + n].mean()))
        chunks.append((s, n))
        s = s + n - overlap


# --------------------------------------------------------------------------------------------- context

class Ctx:
    def __init__(self, book: Path, stem: str, out: Path | None, manifest: list | None = None,
                 work: Path | None = None):
        """manifest/work: the library mode (tools/ltx_flow.py) passes bookcaster's own shots and chapter work dir;
        a manifest entry may carry "first" (its first-frame picture) and "wav" (its speech)."""
        from tts_reader import video
        self.video = video
        self.book, self.stem = book, stem
        self.W = book / "work"
        self.V = Path(work) if work else self.W / "video" / stem
        self.A = self.V / "anim"
        self.OUT = out or (self.W / "video" / f"{stem}_remake")
        self.OUT.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest if manifest is not None else json.loads((self.A / "manifest.json").read_text())
        # moments the scene layer added (quoted lines on still-picture stretches, 10-20): {id, kind, t, dur, pic};
        # they have no old clip, so their "out" is a path in OUT (assemble adds them to the new manifest)
        xm = self.V / "stage" / "extra_moments.json"
        if manifest is None and xm.exists():
            have = {m["id"] for m in self.manifest}
            self.extra = [dict(m, out=str(self.OUT / "anim" / f"{m['id']}.mp4"), extra=True)
                          for m in json.loads(xm.read_text()) if m["id"] not in have]
            self.manifest = sorted(self.manifest + self.extra, key=lambda m: m["t"])
        else:
            self.extra = []
        kf = json.loads((self.A / "keyframes.json").read_text()) if (self.A / "keyframes.json").exists() else {}
        self.frames = {f["shot"]: f for f in kf.get("frames", [])}
        self.plan = json.loads((self.V / "plan.json").read_text())
        lk = json.loads((self.W / "video" / "look.json").read_text())
        # one look per person under any spelling, the first stored wins (video.keep_looks' rule): "Dr. Grimesby
        # Roylott" was a second, bearded entry beside "Doctor Grimesby Roylott" and redrew him bearded (10-20)
        raw = lk.get("characters", {})
        self.canon_key = {}
        for k in raw:
            self.canon_key.setdefault(video.canon(k), k)
        CANON_KEY.update(self.canon_key)
        self.chars = {k: raw[self.canon_key[video.canon(k)]] for k in raw}
        self.refs = self.W / "video" / "refs"
        self.style = (self.V / "style.txt").read_text().strip() if (self.V / "style.txt").exists() else video.DEFAULT_STYLE
        self.period = video.period_of(book)
        self.style = self.style.rstrip(".") + "." + video.period_line(period=self.period)   # keys + LTX prompts (10-21)
        self.timeline = json.loads((self.W / f"{stem}.timeline.json").read_text())
        txt = book / "chapters" / f"{stem}.txt"
        self.text = None
        if txt.exists():
            from tts_reader.cli import _load_text
            self.text = _load_text(str(txt), strip_gutenberg=True)
        try:
            self.starts = video.paragraph_times(video.paragraphs(self.text), self.timeline) if self.text else []
        except Exception:  # noqa: BLE001
            self.starts = []
        assets = self.W / "video" / "stage" / "assets.json"
        self.sets = {s["where"]: s for s in json.loads(assets.read_text()).get("sets", [])
                     if s.get("usable", True)} if assets.exists() else {}
        skip = self.OUT / "skip_sets.txt"      # sets judged wrong by eye: one `where` per line -> treated as no set
        for w in (skip.read_text().splitlines() if skip.exists() else []):
            self.sets.pop(w.strip(), None)
        self.shots_dir = self.W / "video" / "stage" / "shots"

    def dress(self, who: str, t: float) -> str | None:
        if not self.starts:
            return None
        row = self.video.outfit(self.plan, who, self.video.para_at(self.starts, t))
        return row["dress"] if row else None

    def look(self, who: str, t: float) -> str:
        desc = self.chars.get(who, "")
        d = self.dress(who, t)
        if d:   # the person's face/hair from look.json, the clothes from the wardrobe in force
            face = re.split(r",\s*(?:wearing|dressed)\b", desc)[0]
            return f"{face}, wearing {d}"
        return desc

    def speaker(self, m: dict) -> tuple[str, dict]:
        q = min(self.timeline, key=lambda e: abs(e.get("t", e.get("t0", 0)) - m["t"]))
        sp = q.get("speaker") or ""
        from tts_reader import animate
        key = animate.speaker_key(sp, self.chars) or sp
        return key, q


# --------------------------------------------------------------------------------------------- plan

def stage_status(cx: Ctx, f: dict) -> tuple[bool, list[str], str | None, list[str]]:
    """Can the staging engine stage this frame cleanly? -> (ok, reasons, staged shot path or None, notes).
    Clean = figures present, a set for the location, every person placed (pose boxes or text staging, no
    DEFAULT STAGING), no unknown verb. A missing rig is a NOTE (a stand-in body drives the depth control)."""
    why, notes = [], []
    who = list(f.get("who") or [])
    if not who:
        why.append("no figures")
    if (f.get("where") or "") not in cx.sets:
        why.append(f"no set for '{f.get('where')}'")
    # the scene layer's recipe-staged shot (props + states, contacts, fx, the 180-degree axis) FIRST: 10-21 all 60
    # staged moments of knight ch1 fell through to the thin text staging below (no work on the anvil, no left hand,
    # raw white prop boxes) because only the book-level dir was checked
    override = chapter_shot(cx, f["shot"])
    shot_path = str(override) if override else None
    if not shot_path and who:
        try:
            from blockout_plan import frame_to_shot
            crea = {w for w in who if is_creature(cx.chars.get(w, ""))}
            shot = frame_to_shot(f, creatures=crea or None)
            placed = {x["who"] for x in shot.get("figures", [])}
            miss = [w for w in who if w not in placed]
            if miss:
                why.append("DEFAULT STAGING for " + ", ".join(miss))
            pn = shot.get("plan_notes") or {}
            if pn.get("unknown_verb"):
                why.append("unknown verb for " + ", ".join(pn["unknown_verb"]))
            notes.append(f"staging={pn.get('staging', '?')} poses={pn.get('verbs', {})}")
            d = cx.OUT / "stage" / "shots"
            d.mkdir(parents=True, exist_ok=True)
            shot.setdefault("where", f.get("where"))
            shot_path = str(d / f"{f['shot']}.json")
            Path(shot_path).write_text(json.dumps(shot, indent=1))
        except Exception as e:  # noqa: BLE001
            why.append(f"planner error {type(e).__name__}: {e}")
    for w in who:
        last = w.lower().split()[-1]
        if not any((MODELS / c / "rigged.blend").exists() or (MODELS / f"{c}_rig" / "rig.json").exists()
                   for c in (last + "2", last)):
            notes.append(f"no rig for {w} (stand-in body)")
    return (not why), why, shot_path, notes


def first_frame(cx: Ctx, m: dict, f: dict | None) -> Path:
    if m.get("first") and Path(m["first"]).exists():
        return Path(m["first"])
    if m["kind"] == "action":
        k = cx.A / "keyframes" / f"{m['id']}.png"
        if k.exists():
            return k
    return cx.A / f"{m['id']}_first.png"


def strict() -> bool:
    """User 10-21: a delivery is ALL new 3D clips: no I2V fallback jobs, no clip from an earlier run, and a chapter
    with any gap is assembled as _INCOMPLETE (not delivered). REMAKE_ALLOW_FALLBACK=1 = the old behaviour."""
    return full3d() and not os.environ.get("REMAKE_ALLOW_FALLBACK")


def full3d() -> bool:
    """REMAKE_FULL3D=1: EVERY moment through the 3D path (the scene layer's staged shot per moment): action = Mode 2,
    talk AND any action moment where someone in frame speaks = A2Vid from the staged first frame. No silent I2V:
    a moment without a staged shot blocks the render (listed); one whose staging hard-fails after retries falls back
    to I2V and is listed in OUT/FALLBACKS.json."""
    return bool(os.environ.get("REMAKE_FULL3D"))


def chapter_shot(cx: "Ctx", mid: str) -> Path | None:
    """The scene layer's staged shot for a moment: <chapter work>/stage/shots/<id>.json (Phase C), else the book's."""
    for d in (cx.V / "stage" / "shots", cx.shots_dir):
        if (d / f"{mid}.json").exists():
            return d / f"{mid}.json"
    return None


SPEAK_MIN_S = 0.3


def dialogue_spans(cx: "Ctx") -> list[dict]:
    """Every quoted line: speaker (look.json key), [t0, t1) to the next timeline entry, text, emotion."""
    from tts_reader import animate
    tl = sorted(cx.timeline, key=lambda e: e.get("t", 0))
    total = probe_dur(cx.W / f"{cx.stem}.mp3") if (cx.W / f"{cx.stem}.mp3").exists() else tl[-1]["t"] + 5
    out = []
    for a, b in zip(tl, tl[1:] + [{"t": total}]):
        if a.get("kind") == "quote" and a.get("speaker") not in (None, "", "narrator"):
            out.append({"speaker": animate.speaker_key(a["speaker"], cx.chars) or a["speaker"], "t0": a["t"],
                        "t1": b["t"], "text": a.get("text", ""), "emotion": a.get("emotion")})
    return out


def speaking(cx: "Ctx", t0: float, t1: float, who: list) -> list[dict]:
    """The lines spoken inside [t0, t1) by someone in the frame (>= SPEAK_MIN_S of overlap)."""
    return [d for d in dialogue_spans(cx) if canon_name(d["speaker"]) in {canon_name(w) for w in who}
            and min(t1, d["t1"]) - max(t0, d["t0"]) >= SPEAK_MIN_S]


def canon_name(n: str) -> str:
    return re.sub(r"[^a-z]", "", (n or "").lower())


def moment_wav(cx: "Ctx", m: dict, spans: list[dict]) -> Path:
    """The voices-only audio of the moment's window with everything but the in-frame speaker's lines silenced
    (the narrator must not drive the lips)."""
    out = cx.OUT / "audio" / f"{m['id']}_speech.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        keep = "+".join(f"between(t,{max(0, d['t0'] - m['t']):.3f},{d['t1'] - m['t']:.3f})" for d in spans)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{m['t']:.3f}", "-t", f"{m['dur']:.3f}",
                        "-i", str(cx.W / f"{cx.stem}.mp3"), "-ac", "1", "-ar", "16000",
                        "-af", f"volume=0:enable='not({keep})'", str(out)], check=True)
    return out


def prop_approved(vdir: Path, name: str) -> bool:
    """A prop is ready when its mesh is approved: stage/props/props.json {name: {approved: true}} or
    stage/props/<slug>/props.json / model.json with "approved": true (the blockout agent's prop QA)."""
    d = vdir / "stage" / "props"
    for idx in (d / "props.json",):
        if idx.exists():
            try:
                e = json.loads(idx.read_text())
                e = e.get("props", e) if isinstance(e, dict) else {}
                if isinstance(e.get(name), dict) and e[name].get("approved") is True:
                    return True
            except ValueError:
                pass
    for f in ("props.json", "model.json"):
        p = d / slug(name) / f
        if p.exists():
            try:
                if json.loads(p.read_text()).get("approved") is True:
                    return True
            except ValueError:
                pass
    return False


def rig_approved(model: str) -> bool:
    """A creature rig is ready when its rig.json (beside the rigged.blend) says "approved": true."""
    if not model:
        return False
    rj = Path(model).parent / "rig.json"
    try:
        return rj.exists() and json.loads(rj.read_text()).get("approved") is True
    except ValueError:
        return False


def shot_unready(shot_path) -> list[str]:
    """Why a staged shot can't render yet: a person without their OWN model file (the per-character rigs of
    tools/characters.py; a missing one would get a stand-in body = the wrong person)."""
    why = []
    try:
        figs = json.loads(Path(shot_path).read_text()).get("figures", [])
    except (OSError, ValueError):
        return ["unreadable shot"]
    shot = json.loads(Path(shot_path).read_text())
    vdir = Path(shot_path).resolve().parents[3]             # <book>/work/video (shots live in <chapter>/stage/shots)
    apf = vdir / "stage" / "approvals.json"                  # bookcaster's review gate (tools/flow3d.py), when present
    appr = json.loads(apf.read_text()).get("items", {}) if apf.exists() else None

    def gate(key):
        if appr is None or key not in appr:
            return None
        return appr[key].get("status") == "approved"
    wh = (shot.get("set") or {}).get("where")
    if wh and gate(f"set:{wh}") is False:
        why.append(f"set '{wh}' pending review")
    for name in ((shot.get("props_state") or {}).get("props") or {}):
        if not (gate(f"prop:{name}") or prop_approved(vdir, name)):
            why.append(f"prop '{name}' has no approved mesh")
    for g in figs:
        if g.get("kind") == "creature":
            m = g.get("model")
            m = m if isinstance(m, str) else (m or {}).get("blend", "")
            if not (rig_approved(m) or (m and gate(f"creature:{Path(m).parent.name}"))):
                why.append(f"creature {g.get('who')} has no approved rig ({Path(m).parent.name if m else 'none'})")
            continue
        if g.get("kind", "human") != "human":
            continue
        m = g.get("model")
        m = m if isinstance(m, str) else (m or {}).get("blend", "")
        if not m or not Path(m).exists():
            why.append(f"no own model for {g.get('who')}")
        elif g.get("who", "").lower().split()[-1] not in m.lower():
            why.append(f"{g.get('who')} on another character's model ({Path(m).parent.name})")
        elif gate(f"character:{Path(m).parent.name}") is False:
            why.append(f"model {Path(m).parent.name} pending review")
    return why


def plan(cx: Ctx, only: set | None = None) -> dict:
    entries = []
    prev_action = None
    F3 = full3d()
    for m in sorted(cx.manifest, key=lambda e: e["t"]):
        e = {"id": m["id"], "kind": m["kind"], "t": m["t"], "dur": m["dur"], "pic": m["pic"], "old": m["out"]}
        if F3:
            e["shot"] = str(chapter_shot(cx, m["id"]) or "")
            e["unready"] = shot_unready(e["shot"]) if e["shot"] else ["no staged shot"]
            if os.environ.get("REMAKE_PREVIEW") and e["shot"]:     # a preview batch: props/rigs may still be pending
                e["preview_unready"], e["unready"] = e["unready"], [u for u in e["unready"] if u.startswith("no own model")]
        f0 = cx.frames.get(m["id"], {})
        spans = speaking(cx, m["t"], m["t"] + m["dur"], f0.get("who") or []) if F3 and m["kind"] != "talk" else []
        if F3 and spans:         # someone in frame speaks: lip-sync (A2Vid) on the staged frame, their lines only
            main = max(spans, key=lambda d: min(m["t"] + m["dur"], d["t1"]) - max(m["t"], d["t0"]))
            wav = moment_wav(cx, m, [d for d in spans if d["speaker"] == main["speaker"]])
            total = frames_for(m["dur"], 0.0)
            env = audio_env(wav, total + TALK_MAX)
            e.update(route="talk", speaker=main["speaker"], line=" ".join(d["text"] for d in spans if d["speaker"] == main["speaker"]),
                     emotion=main.get("emotion"), creature=is_creature(cx.chars.get(main["speaker"], "")),
                     speech_s=round(m["dur"], 2), frames=total, chunks=talk_chunks(total, env), wav=str(wav),
                     first=str(first_frame(cx, m, f0)), who=f0.get("who", []), from_action=True,
                     listener=next((w for w in f0.get("who", []) if canon_name(w) != canon_name(main["speaker"])
                                    and not is_creature(cx.chars.get(w, ""))), None))
            entries.append(e)
            continue
        if m["kind"] == "talk":
            who, q = cx.speaker(m)
            wav = Path(m["wav"]) if m.get("wav") else cx.A / f"{m['id']}.wav"
            if not wav.exists():                      # an added moment: the in-frame speaker's lines of its window
                sp = [d for d in dialogue_spans(cx) if min(m["t"] + m["dur"], d["t1"]) - max(m["t"], d["t0"]) >= SPEAK_MIN_S]
                if sp:
                    wav = moment_wav(cx, m, [d for d in sp if d["speaker"] == sp[0]["speaker"]])
            sdur = probe_dur(wav) if wav.exists() else m["dur"]
            total = frames_for(sdur, TALK_TAIL)
            env = audio_env(wav, total + TALK_MAX) if wav.exists() else None
            ch = talk_chunks(total, env) if env is not None else [(0, total)]
            if "listener" in m:                       # library mode: bookcaster's own listener for the portrait
                e["listener"] = m["listener"]
            e.update(route="talk", speaker=who, line=q.get("text", ""), emotion=q.get("emotion"),
                     creature=is_creature(cx.chars.get(who, "")), speech_s=round(sdur, 2), frames=total,
                     chunks=ch, wav=str(wav), first=str(first_frame(cx, m, None)))
        else:
            f = cx.frames.get(m["id"], {})
            ok, why, shot, notes = stage_status(cx, f) if f else (False, ["no keyframe entry"], None, [])
            cont = f.get("mode") == "continue" and prev_action is not None and \
                abs(prev_action["t"] + prev_action["dur"] - m["t"]) < 1.0
            if ok and not staging_on():
                ok, why = False, ["staging off (REMAKE_NO_STAGE)"]
            if cont and ok and not F3:   # a continuation keeps the previous clip's last frame (no new keyframe / staging)
                ok, why = False, ["continue: chains from the previous clip"]
            if F3:                       # the scene layer staged it (or nothing may render: no silent I2V)
                ok, why = bool(e["shot"]), ([] if e["shot"] else ["MISSING staged shot (scene layer)"])
            e.update(route=("stage" if ok else "missing") if F3 else ("stage" if ok else "i2v"), why=why, notes=notes, shot=shot if ok else None, where=f.get("where"), who=f.get("who", []),
                     mode=f.get("mode"), frames=frames_for(m["dur"], ACT_TAIL), first=str(first_frame(cx, m, f)),
                     chain_from=prev_action["id"] if cont else None,
                     chain_frame=min(prev_action["frames"] - 1, round(prev_action["dur"] * FPS) - 1) if cont else None,
                     world=f.get("where") in cx.sets)
            prev_action = e
        entries.append(e)
    if only:
        entries = [e for e in entries if e["id"] in only]
    P = {"book": str(cx.book), "stem": cx.stem, "out": str(cx.OUT), "entries": entries}
    (cx.OUT / "plan.json").write_text(json.dumps(P, indent=1))
    return P


def estimate(P: dict) -> dict:
    E = P["entries"]
    talk = [e for e in E if e["route"] == "talk"]
    act = [e for e in E if e["route"] != "talk"]
    n_chunks = sum(len(e["chunks"]) for e in talk)
    talk_frames = sum(n for e in talk for _, n in e["chunks"])
    t_talk_gen = talk_frames * T_TALK_PER_FRAME
    holds_t = max(1, math.ceil(t_talk_gen / (HOLD_BUDGET["talk"] - T_TALK_LOAD + 130)))
    t_talk = t_talk_gen + holds_t * T_TALK_LOAD
    t_act_gen = len(act) * T_ACT_CLIP
    holds_a = max(1, math.ceil(t_act_gen / (HOLD_BUDGET["act"] - T_ACT_LOAD + 62)))
    t_act = t_act_gen + holds_a * T_ACT_LOAD
    t_enc = 2 * T_ENC_LOAD + T_ENC_PROMPT * (2 * len(talk) + len(act))
    n_stage = sum(1 for e in act if e["route"] == "stage")
    t_stage = (300 + n_stage * (70 + 3 * 14)) if n_stage else 0   # motion batch + keyframe draws (Qwen load + 3 seeds)
    return {"talk_entries": len(talk), "talk_chunks": n_chunks, "talk_frames": talk_frames, "talk_holds": holds_t,
            "action_entries": len(act), "stage": n_stage, "i2v": len(act) - n_stage,
            "chained_continue": sum(1 for e in act if e.get("chain_from")),
            "action_holds": holds_a,
            "gpu_min": {"encode": round(t_enc / 60, 1), "talk": round(t_talk / 60, 1), "action": round(t_act / 60, 1),
                        "stage_one_card": round(t_stage / 60, 1),
                        "stage_render_cpu_2workers": round(n_stage * 130 / 2 / 60, 1),
                        "total_both_cards": round((t_enc + t_talk + t_act) / 60, 1),
                        "total_serial": round((t_enc + t_talk + t_act + t_stage) / 60, 1)}}


# --------------------------------------------------------------------------------------------- check / redraw

def check_targets(cx: Ctx, e: dict) -> list[dict]:
    """Which faces to check in this entry's first frame: the speaker (talk) and every elderly person (age >= 55:
    the documented drawer failure is smoothing old faces), each against its reference sheet."""
    names = [e["speaker"]] if e["route"] == "talk" else []
    who = e.get("who") or []
    if e["route"] == "talk" and "listener" in e:     # bookcaster's portrait: the speaker + the one spoken to
        # (a close-up without one), not the whole scene: the dead Roylott of p058 was "checked" on Holmes's face
        # in Holmes's close-up and redrawn standing alive in his place (10-20)
        who = [e["speaker"]] + ([e["listener"]] if e["listener"] else [])
    elif e["route"] == "talk":                        # who is in the talking shot's picture
        pic = next((p for p in cx.plan.get("pictures", []) if p["id"] == e["pic"]), {})
        who = pic.get("who") or who
    for w in who:
        a = age_of(cx.chars.get(w, ""))
        if a and a >= 55 and w not in names:
            names.append(w)
    out = []
    for w in names:
        desc = cx.chars.get(w, "")
        if is_creature(desc) or not desc:
            continue
        sheet = ref_sheet(cx.refs, w)
        if sheet is None:
            continue
        n_people = len([x for x in who if not is_creature(cx.chars.get(x, ""))]) or 1
        out.append({"who": w, "sheet": str(sheet), "age": age_of(desc), "gender": gender_of(desc), "n_people": n_people})
    return out


_FACE_CACHE: dict = {}


def _fkey(img, t, side):
    return (str(img), str(t["sheet"]), side, t.get("gender"), t.get("age"))


def _sides(t):
    return (t["side"],) if t.get("side") else ("left", "right") if t.get("n_people", 1) > 1 else ("any",)


def prefetch_faces(reqs: list) -> None:
    """[(image, target)] -> every face check in ONE batch (tools/face_batch.py: one DWPose + one DINOv2/CLIP load),
    cached for run_face. ~4 s a check instead of ~48 s."""
    import face_batch
    jobs, keys = [], []
    for img, t in reqs:
        for sd in _sides(t):
            k = _fkey(img, t, sd)
            if k in _FACE_CACHE or k in keys:
                continue
            jobs.append({"id": str(len(jobs)), "image": str(img), "ref": t["sheet"], "gender": t.get("gender", "person"),
                         "age": t.get("age"), "side": sd})
            keys.append(k)
    if not jobs:
        return
    out = face_batch.run(jobs)
    for j, k in zip(jobs, keys):
        _FACE_CACHE[k] = out.get(j["id"], {"error": "no result"})


def run_face(img: str, t: dict, js: Path) -> dict:
    """One face check; with several people in the frame both sides are scored and the face that matches the
    reference best is taken as that person's."""
    best = None
    for side in _sides(t):
        jp = js.with_name(js.stem + f"_{side}.json")
        c = _FACE_CACHE.get(_fkey(img, t, side))
        if c is not None:
            d = {**c, "side": side, "rc": 0}
            jp.write_text(json.dumps(d, indent=1))
            if best is None or float(d.get("face_sim") or -1) > float(best.get("face_sim") or -1):
                best = d
            continue
        cmd = ["python3", str(HERE / "face_likeness.py"), img, "--ref", t["sheet"], "--gender", t["gender"],
               "--side", side, "--json", str(jp)]
        if t.get("age"):
            cmd += ["--age", str(t["age"])]
        r = subprocess.run(cmd, capture_output=True, text=True)
        d = json.loads(jp.read_text()) if jp.exists() else {"error": (r.stderr or r.stdout)[-400:]}
        d["rc"], d["side"] = r.returncode, side
        if best is None or float(d.get("face_sim") or -1) > float(best.get("face_sim") or -1):
            best = d
    js.write_text(json.dumps(best, indent=1))
    return best


AGE_TOL = 12.0          # years between the keyframe face and the SHEET's own face (CLIP's age read of the sheet is
                        # the reachable truth: it reads a 20-year-old's sheet as ~31)
ELDER_AGE, ELDER_WRINKLED = 55, 0.7     # the user rejected talk-001316 (0.63); the fixed redraw read 0.98


def judge(r: dict, cx: "Ctx") -> bool:
    """Pass/fail for one face: the age as read on the keyframe vs as read on the sheet, and for the elderly the
    wrinkles (the drawer's known failure: a smooth 40-year-old for a 70-year-old). face_sim only ranks redraws:
    on 50-80 px faces of painted scenes it is too noisy to reject on (0.2-0.75 for the same person)."""
    a = r.get("age") or {}
    if not isinstance(a, dict) or a.get("keyframe") is None:
        return True
    if abs(float(a["keyframe"]) - float(a["sheet"])) > AGE_TOL:
        return False
    tgt = age_of(cx.chars.get(r.get("who", ""), ""))
    if tgt and tgt >= ELDER_AGE and float((r.get("wrinkled") or {}).get("keyframe") or 0) < ELDER_WRINKLED:
        return False
    return True


def assign_sides(faces: list[dict], d: Path, eid: str) -> None:
    """Two people scored on both sides: give each person a distinct face (the assignment with the larger summed
    face_sim), so one face is never judged as both."""
    if len(faces) != 2:
        return
    load = lambda r, side: (json.loads(p.read_text()) if (p := d / f"{eid}__{slug(r['who'])}_{side}.json").exists()  # noqa: E731
                            else None)
    a, b = faces
    best, pick = -9.0, None
    for sa, sb in (("left", "right"), ("right", "left")):
        ra, rb = load(a, sa), load(b, sb)
        if not ra or not rb or not ra.get("face_box") or not rb.get("face_box"):
            continue
        tot = float(ra.get("face_sim") or 0) + float(rb.get("face_sim") or 0)
        if tot > best:
            best, pick = tot, (ra, rb, sa, sb)
    if pick:
        ra, rb, sa, sb = pick
        faces[0] = {**ra, "who": a["who"], "side": sa}
        faces[1] = {**rb, "who": b["who"], "side": sb}


def rejudge(cx: "Ctx", P: dict) -> None:
    """Re-apply assign_sides + judge to existing check files (no model runs)."""
    d = cx.OUT / "checks"
    for f in sorted(d.glob("*.json")):
        if "__" in f.name:
            continue
        res = json.loads(f.read_text())
        assign_sides(res["faces"], d, res["id"])
        no_face = [r for r in res["faces"] if r.get("error") or not r.get("face_box") or r.get("skipped")]
        res["fail"] = [r["who"] for r in res["faces"] if r not in no_face and not judge(r, cx)]
        f.write_text(json.dumps(res, indent=1))
        print(res["id"], [(r["who"], r.get("side"), r.get("face_sim"), (r.get("age") or {}).get("keyframe"),
                           (r.get("wrinkled") or {}).get("keyframe")) for r in res["faces"]], "FAIL" if res["fail"] else "ok", res["fail"])


def check(cx: Ctx, P: dict, workers: int = 2) -> None:
    d = cx.OUT / "checks"
    d.mkdir(exist_ok=True)
    def one(e):
        res_f = d / f"{e['id']}.json"
        img = base_first(cx, e)
        res = {"id": e["id"], "image": img, "faces": []}
        for t in check_targets(cx, e):
            r = run_face(img, t, d / f"{e['id']}__{slug(t['who'])}.json")
            r["who"] = t["who"]
            res["faces"].append(r)
        assign_sides(res["faces"], d, e["id"])
        no_face = [r for r in res["faces"] if r.get("error") or not r.get("face_box") or r.get("skipped")]
        res["fail"] = [r["who"] for r in res["faces"] if r not in no_face and not judge(r, cx)]
        res["no_face"] = [r["who"] for r in no_face]
        res_f.write_text(json.dumps(res, indent=1))
        log_entry(cx.OUT, {"phase": "check", "id": e["id"], "checked": [r["who"] for r in res["faces"]],
                           "fail": res["fail"], "no_face": res["no_face"]})

    todo = [e for e in P["entries"] if e["route"] != "stage" and not e.get("chain_from")   # staged/chained: own frame
            and not (d / f"{e['id']}.json").exists()]
    prefetch_faces([(base_first(cx, e), t) for e in todo for t in check_targets(cx, e)])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def redraw_prompt(cx: Ctx, e: dict, fail: list[str]) -> tuple[str, list[str], list]:
    f = cx.frames.get(e["id"]) or {}
    pic = next((p for p in cx.plan["pictures"] if p["id"] == e["pic"]), {})
    base = f.get("prompt") or pic.get("prompt", "")
    who = list(f.get("who") or pic.get("who") or ([e["speaker"]] if e.get("speaker") else []))
    for w in fail:
        if w not in who:
            who.append(w)
    refs, loras, parts = [], [], []
    for w in fail:                                      # the failing face's headshot FIRST
        hs = cx.OUT / "refs" / f"{slug(w)}_headshot.png"
        sheet = ref_sheet(cx.refs, w)
        if sheet and not hs.exists():
            hs.parent.mkdir(exist_ok=True)
            subprocess.run(["python3", str(HERE / "face_likeness.py"), str(sheet), "--ref", str(sheet),
                            "--headshot-out", str(hs)], capture_output=True)
        if hs.exists():
            refs.append(str(hs))
    for w in who:
        sheet = ref_sheet(cx.refs, w)
        if sheet and str(sheet) not in refs:
            refs.append(str(sheet))
        if (x := char_lora(cx, w)) and len([v for v in who if not is_creature(cx.chars.get(v, ""))]) <= 1:
            loras.append([x, 1.0])                   # one person only: a global LoRA bleeds onto the others
    sides = {}
    cf = cx.OUT / "checks" / f"{e['id']}.json"
    if cf.exists():
        sides = {r["who"]: r.get("side") for r in json.loads(cf.read_text()).get("faces", [])
                 if r.get("side") in ("left", "right") and r.get("face_box")}
    for w in who:
        desc = cx.look(w, e["t"])
        a = age_of(desc)
        sh = ref_sheet(cx.refs, w)
        n = refs.index(str(sh)) + 1 if sh and str(sh) in refs else None
        hs = cx.OUT / "refs" / f"{slug(w)}_headshot.png"
        hn = refs.index(str(hs)) + 1 if w in fail and str(hs) in refs else None
        face = f" with the face of image {hn}" if hn else ""
        extra = ""
        if w in fail and a and a >= ELDER_AGE:
            extra = (f"an ELDERLY {gender_of(desc)} of about {int(a)}{face}: a deeply lined, weathered "
                     "face, sagging skin, deep crow's feet and forehead furrows, age spots, thin lips; ")
        elif w in fail and a and a < 30:
            extra = f"a YOUNG {gender_of(desc)} of about {int(a)}{face}: a fresh, youthful, unlined face; "
        elif w in fail:
            extra = f"{face.strip()}; " if face else ""
        pos = f"on the {sides[w]}, " if w in sides else ""
        parts.append(f"{pos}{w}: {extra}{desc.rstrip('.')}" + (f", as in image {n}" if n else ""))
    where = f.get("where")
    pl = ref_sheet(cx.refs, where) if where else None
    if pl and len(refs) < 10:
        refs.append(str(pl))
    n_h = len([w for w in who if not is_creature(cx.chars.get(w, ""))])
    prompt = (f"{base} Exactly {n_h} {'person' if n_h == 1 else 'people'} in the picture, each appearing once: "
              + "; ".join(parts) + f". {cx.style}. No text, no lettering, no signature.")
    return prompt, refs[:10], loras


CHARLORA_PY = HOME / "tts-ab" / ".venv-charlora-igpu" / "bin" / "python"


def vl_batch(items: list[dict]) -> dict:
    """Merged: every question asked of the same picture/crop goes in ONE prompt with a JSON answer (one vision
    encode + one decode per crop instead of one per question: 22 min -> minutes for 8 shots, 10-20); an unparsable
    answer falls back to the questions one by one. -> {id: {key: answer}}"""
    groups, order = {}, []
    for it in items:
        k = (it["image"], json.dumps(it.get("crop")))
        if k not in groups:
            groups[k] = {"image": it["image"], **({"crop": it["crop"]} if it.get("crop") else {}), "parts": []}
            order.append(k)
        groups[k]["parts"].append(it)
    merged, single = [], []
    for gi, k in enumerate(order):
        g = groups[k]
        qs = [(f"{n}__{q}", it["id"], q, txt) for n, it in enumerate(g["parts"]) for q, txt in it["ask"].items()]
        if len(qs) == 1:
            single.append(g["parts"][0])
            continue
        body = "; ".join(f'"q{m}": {txt}' for m, (_, _, _, txt) in enumerate(qs))
        merged.append({"id": f"__g{gi}", "image": g["image"], **({"crop": g["crop"]} if g.get("crop") else {}),
                       "max_new_tokens": 10 * len(qs) + 16,
                       "ask": {"json": "Answer every question about this picture. Reply with ONLY one JSON object, keys "
                                       f"q0..q{len(qs) - 1}, each value a short answer: {body}"}})
        g["qs"] = qs
    raw = _vl_run(merged + single)
    out = {}
    redo = []
    for gi, k in enumerate(order):
        g = groups[k]
        if "qs" not in g:
            continue
        a = (raw.get(f"__g{gi}") or {}).get("json", "")
        m = re.search(r"\{.*\}", a, re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except ValueError:
            d = None
        if not isinstance(d, dict):
            redo += g["parts"]
            continue
        for m_, (_, iid, q, _) in enumerate(g["qs"]):
            v = d.get(f"q{m_}")
            if v is None:
                continue
            out.setdefault(iid, {})[q] = str(v).strip().lower()
    for it in single:
        if it["id"] in raw:
            out.setdefault(it["id"], {}).update(raw[it["id"]])
    if redo:
        r2 = _vl_run(redo)
        for k2, v2 in r2.items():
            out.setdefault(k2, {}).update(v2)
    return out


def _vl_run(items: list[dict]) -> dict:
    """tools/vl_judge.py (Qwen3-VL-8B on the iGPU, CU mask 0-38) for many yes/no questions in ONE load:
    items [{id, image, ask: {key: question}}] -> {id: {key: answer}}. {} when the judge can't run (logged)."""
    if not items:
        return {}
    mem = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1048576
    if mem < 15 + 18:
        print(f"vl_batch: MemAvailable {mem} GiB too low for the judge: content checks skipped", flush=True)
        return {}
    # a FREE XTX first (10-20: the judge crawled on the iGPU beside the LoRA training while both XTXs idled): take
    # its lock without waiting, else the iGPU with the CU mask
    import fcntl
    lk, card = None, None
    for c in ("05", "65"):
        f_ = open(f"/tmp/gpu-{c}.lock", "a")
        try:
            fcntl.flock(f_, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lk, card = f_, c
            break
        except OSError:
            f_.close()
    if card:
        hx = subprocess.run([str(HOME / "bin" / "hip-index"), card], capture_output=True, text=True).stdout.strip()
        env = dict(os.environ, HIP_VISIBLE_DEVICES=hx)
        env.pop("HSA_CU_MASK", None)
    else:
        ig = subprocess.run([str(HOME / "bin" / "hip-index"), "igpu"], capture_output=True, text=True).stdout.strip()
        env = dict(os.environ, HIP_VISIBLE_DEVICES=ig, HSA_CU_MASK=f"{ig}:0-38")
    env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)       # NaN answers with it (charlora STATE)
    try:
        r = subprocess.run([str(QWEN_PY if card else CHARLORA_PY), "-I", str(HERE / "vl_judge.py")],
                           input="\n".join(json.dumps(i) for i in items), capture_output=True, text=True, env=env, timeout=3600)
    finally:
        if lk:
            lk.close()
    print(f"vl_batch: {len(items)} item(s) on {'XTX ' + card if card else 'the iGPU'}", flush=True)
    out = {}
    for ln in r.stdout.splitlines():
        try:
            d = json.loads(ln)
            out[d["id"]] = {k: str(v).strip().lower() for k, v in d["answers"].items()}
        except (ValueError, KeyError):
            continue
    if not out:
        print(f"vl_batch: no answers ({r.stderr[-300:]})", flush=True)
    return out


def content_questions(cx: "Ctx", e: dict, who: list) -> dict:
    """Yes/no checks of a drawn frame against the plan + looks: every person present, facial hair as in look.json
    (beard/moustache present or absent), the frame's named props visible. {key: (question, expected 'yes'|'no')}"""
    q = {}
    f = cx.frames.get(e["id"]) or {}
    for w in who:
        desc = cx.chars.get(w, "")
        if not desc:
            continue
        short = re.split(r"[;,]", desc)[0]
        q[f"present:{w}"] = (f"Is there {short} in the picture? Answer yes or no.", "yes")
        if is_creature(desc):
            continue
        if re.search(r"clean-shaven|beardless|no beard|smooth beardless", desc, re.I):
            q[f"beard:{w}"] = (f"Look at {short}. Does this person have a beard or a moustache? Answer yes or no.", "no")
        elif re.search(r"\bbeard|moustache|mustache\b", desc, re.I):
            q[f"beard:{w}"] = (f"Look at {short}. Does this person have a beard or a moustache? Answer yes or no.", "yes")
    for prop in (f.get("props") or [])[:3]:
        q[f"prop:{prop}"] = (f"Is there {prop.lower()} visible in the picture? Answer yes or no.", "yes")
    return q


def content_score(answers: dict, qs: dict) -> tuple[int, list]:
    """-> (number of failed checks, their keys); a missing answer counts as no failure."""
    bad = [k for k, (_, want) in qs.items() if k in answers and not answers[k].startswith(want)]
    return len(bad), bad


def redraw(cx: Ctx, P: dict, card: str = "05", seeds: int = 4) -> None:
    """Redraw every failing first frame ONCE per distinct picture (a run of talk lines shares one portrait; they
    must share one redraw too), N seeds in one Qwen load, best seed by pass + face_sim + wrinkles; the pick is
    copied to keys/<id>.png for every entry that used that picture."""
    import hashlib
    d = cx.OUT / "keys"
    d.mkdir(exist_ok=True)
    groups: dict[str, dict] = {}
    for e in P["entries"]:
        cf = cx.OUT / "checks" / f"{e['id']}.json"
        if not cf.exists():
            continue
        c = json.loads(cf.read_text())
        if not c.get("fail") or (d / f"{e['id']}.png").exists():
            continue
        h = hashlib.md5(Path(base_first(cx, e)).read_bytes()).hexdigest()[:10]
        g = groups.setdefault(h, {"rep": e, "ids": [], "fail": []})
        g["ids"].append(e["id"])
        g["fail"] += [w for w in c["fail"] if w not in g["fail"]]
    jobs = []
    for h, g in groups.items():
        e = g["rep"]
        prompt, refs, loras = redraw_prompt(cx, e, g["fail"])
        from PIL import Image
        w, hh = Image.open(base_first(cx, e)).size
        sw, sh = (w // 16 * 16, hh // 16 * 16) if full3d() else ((832, 480) if w / hh < 1.7 else (1024, 576))
        for k in range(seeds):
            if (d / f"g{h}_s{k}.png").exists():
                continue
            j = {"id": f"g{h}_s{k}", "prompt": prompt, "seed": 1000 + 4 * k, "size": [sw, sh], "refs": refs,
                 "path": str(d / f"g{h}_s{k}.png"), "init": {"image": base_first(cx, e), "strength": 0.6}}
            if loras:
                j["loras"] = loras
            jobs.append(j)
    if not groups:
        print("redraw: nothing to redraw")
        return
    if jobs:
        jf = d / "jobs.json"
        jf.write_text(json.dumps(jobs, indent=1))
        mem = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1048576
        if mem < 15:
            raise SystemExit(f"MemAvailable {mem} GiB < 15: not loading Qwen")
        hip = subprocess.run([str(HOME / "bin" / "hip-index"), card], capture_output=True, text=True).stdout.strip()
        env = dict(os.environ, HIP_VISIBLE_DEVICES=hip)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        t = time.time()
        r = subprocess.run(["flock", f"/tmp/gpu-{card}.lock", "timeout", "870", str(QWEN_PY), "-I", str(HERE / "qwen_image.py"),
                            str(jf), str(d)], env=env, capture_output=True, text=True)
        (d / "run.log").write_text(r.stdout + r.stderr)
        log_entry(cx.OUT, {"phase": "redraw", "rc": r.returncode, "jobs": len(jobs), "groups": len(groups),
                           "sec": round(time.time() - t)})
        kernel_check(cx.OUT)
    from concurrent.futures import ThreadPoolExecutor

    def score(h, g, k):
        """img2img 0.6 keeps the composition: each failing person is scored on the side found in the original."""
        e = g["rep"]
        img = d / f"g{h}_s{k}.png"
        if not img.exists():
            return None
        orig = {r["who"]: r.get("side") for r in json.loads((cx.OUT / "checks" / f"{e['id']}.json").read_text())["faces"]}
        faces = []
        for t in [x for x in check_targets(cx, e) if x["who"] in g["fail"]]:
            if orig.get(t["who"]) in ("left", "right"):
                t = {**t, "side": orig[t["who"]]}
            fr = run_face(str(img), t, d / f"g{h}_s{k}__{slug(t['who'])}.json")
            fr["who"] = t["who"]
            faces.append(fr)
        sc = sum(float(f.get("face_sim") or 0) + 0.5 * float((f.get("wrinkled") or {}).get("keyframe") or 0) for f in faces)
        return img, sc + (10 if all(judge(f, cx) for f in faces) else 0)

    reqs = []
    for h, g in groups.items():
        orig = {r["who"]: r.get("side") for r in json.loads((cx.OUT / "checks" / f"{g['rep']['id']}.json").read_text())["faces"]}
        for k in range(seeds):
            if (d / f"g{h}_s{k}.png").exists():
                for t in [x for x in check_targets(cx, g["rep"]) if x["who"] in g["fail"]]:
                    reqs.append((str(d / f"g{h}_s{k}.png"), {**t, "side": orig[t["who"]]} if orig.get(t["who"]) in ("left", "right") else t))
    prefetch_faces(reqs)
    keys = [(h, g, k) for h, g in groups.items() for k in range(seeds)]
    with ThreadPoolExecutor(4) as ex:
        scores = dict(zip([(h, k) for h, _, k in keys], ex.map(lambda a: score(*a), keys)))
    # the redraw must keep the shot's people, props and looks (10-20 Sherlock: a redraw lost the band, shut the safe,
    # gave Watson a top hat): Qwen3-VL yes/no checks, one load for every seed; a failed check outweighs the faces
    qs = {h: content_questions(cx, g["rep"], moment_who(cx, g["rep"])) for h, g in groups.items()}
    ans = vl_batch([{"id": f"{h}:{k}", "image": str(d / f"g{h}_s{k}.png"), "ask": {q: v[0] for q, v in qs[h].items()}}
                    for h, g in groups.items() for k in range(seeds) if (d / f"g{h}_s{k}.png").exists() and qs[h]])
    for (h, k), r in list(scores.items()):
        if r and f"{h}:{k}" in ans:
            nbad, badk = content_score(ans[f"{h}:{k}"], qs[h])
            scores[(h, k)] = (r[0], r[1] - 20 * nbad)
            if nbad:
                log_entry(cx.OUT, {"phase": "redraw-content", "seed": f"g{h}_s{k}", "failed": badk})
    for h, g in groups.items():                       # rank seeds: pass first, then face_sim + wrinkles
        best, bscore = None, -9.0
        for k in range(seeds):
            r = scores.get((h, k))
            if r and r[1] > bscore:
                best, bscore = r
        if best:
            for i in g["ids"]:
                shutil.copy(best, d / f"{i}.png")
        log_entry(cx.OUT, {"phase": "redraw", "ids": g["ids"], "fail": g["fail"], "pick": str(best),
                           "score": round(bscore, 3), "passed": bscore >= 10})


YOUNG_MAX_OVER_SHEET = 2.5   # a young person's face may read at most this much older than their own sheet face


def band_ok(r: dict, cx: "Ctx") -> bool:
    """Stricter gate (coordinator, 10-19): each face inside its age band, young AND old."""
    a = r.get("age") or {}
    if not isinstance(a, dict) or a.get("keyframe") is None:
        return False
    tgt = age_of(cx.chars.get(r.get("who", ""), "")) or 40
    kf, sh = float(a["keyframe"]), float(a["sheet"])
    if tgt >= ELDER_AGE:
        return kf >= 60 and float((r.get("wrinkled") or {}).get("keyframe") or 0) >= ELDER_WRINKLED
    if tgt < 30:
        return kf <= sh + YOUNG_MAX_OVER_SHEET
    return abs(kf - sh) <= AGE_TOL


def _qwen(cx: "Ctx", jobs: list, d: Path, card: str, tag: str) -> None:
    while (cx.OUT / ".pause_gpu05").exists() and not os.environ.get("REMAKE_IGNORE_PAUSE"):
        time.sleep(20)
    jf = d / f"jobs_{tag}.json"
    jf.write_text(json.dumps(jobs, indent=1))
    mem = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1048576
    if mem < 15:
        raise SystemExit(f"MemAvailable {mem} GiB < 15: not loading Qwen")
    hip = subprocess.run([str(HOME / "bin" / "hip-index"), card], capture_output=True, text=True).stdout.strip()
    env = dict(os.environ, HIP_VISIBLE_DEVICES=hip)
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    t = time.time()
    r = subprocess.run(["flock", f"/tmp/gpu-{card}.lock", "timeout", "870", str(QWEN_PY), "-I", str(HERE / "qwen_image.py"),
                        str(jf), str(d)], env=env, capture_output=True, text=True)
    (d / f"run_{tag}.log").write_text(r.stdout + r.stderr)
    log_entry(cx.OUT, {"phase": "redraw-strict", "tag": tag, "rc": r.returncode, "jobs": len(jobs), "sec": round(time.time() - t)})
    kernel_check(cx.OUT)


def redraw_strict(cx: "Ctx", P: dict, ids: set, card: str = "05", seeds: int = 6) -> None:
    """Redraw the pictures of `ids` with EVERY person's headshot first and the age wording for all of them, N seeds,
    gate: every face in its band (band_ok). No seed passes -> keep the ORIGINAL frame and redraw only the failing
    faces' head regions (crop -> img2img -> feathered paste), then gate again; still failing -> best full redraw."""
    import hashlib
    from PIL import Image, ImageDraw, ImageFilter
    d = cx.OUT / "keys" / "strict"
    d.mkdir(parents=True, exist_ok=True)
    groups: dict[str, dict] = {}
    for e in P["entries"]:
        if e["id"] not in ids:
            continue
        h = hashlib.md5(Path(e["first"]).read_bytes()).hexdigest()[:10]
        groups.setdefault(h, {"rep": e, "ids": []})["ids"].append(e["id"])
    jobs = []
    for h, g in groups.items():
        e = g["rep"]
        everyone = [t["who"] for t in all_targets(cx, e)]
        g["targets"] = all_targets(cx, e)
        prompt, refs, loras = redraw_prompt(cx, e, everyone)
        w, hh = Image.open(base_first(cx, e)).size
        sw, sh = (832, 480) if w / hh < 1.7 else (1024, 576)
        for k in range(seeds):
            if not (d / f"g{h}_s{k}.png").exists():
                j = {"id": f"g{h}_s{k}", "prompt": prompt, "seed": 2000 + 4 * k, "size": [sw, sh], "refs": refs,
                     "path": str(d / f"g{h}_s{k}.png"), "init": {"image": base_first(cx, e), "strength": 0.6}}
                if loras:
                    j["loras"] = loras
                jobs.append(j)
    if jobs:
        _qwen(cx, jobs, d, card, "full")

    def gate(img: Path, g: dict, tag: str):
        orig = {r["who"]: r.get("side") for r in json.loads((cx.OUT / "checks" / f"{g['rep']['id']}.json").read_text())["faces"]}
        faces = []
        for t in g["targets"]:
            if orig.get(t["who"]) in ("left", "right"):
                t = {**t, "side": orig[t["who"]]}
            fr = run_face(str(img), t, d / f"{tag}__{slug(t['who'])}.json")
            fr["who"] = t["who"]
            faces.append(fr)
        ok = all(band_ok(f, cx) for f in faces)
        sc = sum(float(f.get("face_sim") or 0) for f in faces) + sum(band_ok(f, cx) for f in faces)
        return ok, sc, faces

    from concurrent.futures import ThreadPoolExecutor
    for h, g in groups.items():
        if os.environ.get("REMAKE_STRICT_HEADONLY"):    # full redraws already judged (none in band): go to head fix
            res = [(0, False, -9, [])]
        else:
          with ThreadPoolExecutor(3) as ex:
            res = list(ex.map(lambda k: (k, *gate(d / f"g{h}_s{k}.png", g, f"g{h}_s{k}")) if (d / f"g{h}_s{k}.png").exists()
                              else (k, False, -9, []), range(seeds)))
        res.sort(key=lambda x: (not x[1], -x[2]))
        k, ok, sc, faces = res[0]
        pick = d / f"g{h}_s{k}.png"
        how = f"full redraw seed {k}"
        if os.environ.get("REMAKE_STRICT_HEADONLY"):
            pick, how = Path(g["rep"]["first"]), "original"
        if not ok:   # the original frame + head-region redraws of the faces that fail their band there
            e = g["rep"]
            c0 = json.loads((cx.OUT / "checks" / f"{e['id']}.json").read_text())
            bad = [r for r in c0["faces"] if r.get("face_box") and not band_ok(r, cx)]
            base = Image.open(base_first(cx, e)).convert("RGB")
            hjobs = []
            for r in bad:
                x0, y0, x1, y1 = r["face_box"]
                cx_, cy_, half = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * 1.1
                box = [int(max(0, cx_ - half)), int(max(0, cy_ - half)), int(min(base.width, cx_ + half)), int(min(base.height, cy_ + half))]
                crop = d / f"g{h}_head_{slug(r['who'])}_in.png"
                base.crop(box).resize((512, 512), Image.LANCZOS).save(crop)
                hs = cx.OUT / "refs" / f"{slug(r['who'])}_headshot.png"
                desc = cx.chars.get(r["who"], "")
                a = age_of(desc) or 40
                age_txt = ("an ELDERLY {g} of about {a}: a deeply lined, weathered face, sagging skin, deep crow's feet and "
                           "forehead furrows, age spots, thin lips" if a >= ELDER_AGE else
                           "a YOUNG {g} of about {a}: a fresh, youthful, unlined, smooth face" if a < 30 else "a {g} of about {a}")
                p = (f"Close-up of the head of {r['who']}, " + age_txt.format(g=gender_of(desc), a=int(a)) +
                     f", with the face of image 1; {desc.rstrip('.')}. Same lighting, same pose, same angle. {cx.style}.")
                for k2 in range(3):
                    hjobs.append({"id": f"g{h}_head_{slug(r['who'])}_s{k2}", "prompt": p, "seed": 3000 + 4 * k2, "size": [512, 512],
                                  "refs": [str(x) for x in (hs, ref_sheet(cx.refs, r["who"])) if x and Path(x).exists()],
                                  "path": str(d / f"g{h}_head_{slug(r['who'])}_s{k2}.png"),
                                  "init": {"image": str(crop), "strength": 0.55}, "box": box, "who": r["who"]})
            todo = [j for j in hjobs if not Path(j["path"]).exists()]
            if todo:
                _qwen(cx, [{k: v for k, v in j.items() if k not in ("box", "who")} for j in todo], d, card, f"head_{h}")
            best_patch = {}
            for j in hjobs:             # per face: the head seed whose own crop reads best in band
                if not Path(j["path"]).exists():
                    continue
                comp = base.copy()
                bw, bh = j["box"][2] - j["box"][0], j["box"][3] - j["box"][1]
                patch = Image.open(j["path"]).convert("RGB").resize((bw, bh), Image.LANCZOS)
                m = Image.new("L", (bw, bh), 0)
                ImageDraw.Draw(m).ellipse([bw * 0.12, bh * 0.08, bw * 0.88, bh * 0.95], fill=255)
                m = m.filter(ImageFilter.GaussianBlur(max(2, bw // 12)))
                comp.paste(patch, j["box"][:2], m)
                tp = d / f"{j['id']}_comp.png"
                comp.save(tp)
                t = next(x for x in g["targets"] if x["who"] == j["who"])
                side = next((r.get("side") for r in c0["faces"] if r["who"] == j["who"]), None)
                fr = run_face(str(tp), {**t, **({"side": side} if side in ("left", "right") else {})}, d / f"{j['id']}_comp__face.json")
                fr["who"] = j["who"]
                sc2 = float(fr.get("face_sim") or 0) + (2 if band_ok(fr, cx) else 0)
                if sc2 > best_patch.get(j["who"], (None, -9))[1]:
                    best_patch[j["who"]] = (j, sc2, patch, m)
            comp = base.copy()
            for w, (j, sc2, patch, m) in best_patch.items():
                comp.paste(patch, j["box"][:2], m)
            cp = d / f"g{h}_headfix.png"
            comp.save(cp)
            ok2, sc2, faces2 = gate(cp, g, f"g{h}_headfix")
            if ok2 or sum(band_ok(f, cx) for f in faces2) > sum(band_ok(f, cx) for f in faces):
                pick, ok, faces, how = cp, ok2, faces2, "original + head-region redraw"
        for i in g["ids"]:
            shutil.copy(pick, cx.OUT / "keys" / f"{i}.png")
        log_entry(cx.OUT, {"phase": "redraw-strict", "ids": g["ids"], "pick": str(pick), "how": how, "passed": ok,
                           "faces": [(f["who"], (f.get("age") or {}).get("keyframe"), (f.get("wrinkled") or {}).get("keyframe"),
                                      f.get("face_sim")) for f in faces]})


def all_targets(cx: "Ctx", e: dict) -> list[dict]:
    """Every person (not creature) in the moment's picture with a reference sheet."""
    pic = next((p for p in cx.plan.get("pictures", []) if p["id"] == e["pic"]), {})
    f = cx.frames.get(e["id"]) or {}
    who = f.get("who") or pic.get("who") or []
    hum = [w for w in who if not is_creature(cx.chars.get(w, "")) and cx.chars.get(w)]
    out = []
    for w in hum:
        sheet = ref_sheet(cx.refs, w)
        if sheet:
            desc = cx.chars[w]
            out.append({"who": w, "sheet": str(sheet), "age": age_of(desc), "gender": gender_of(desc), "n_people": len(hum)})
    return out


# --------------------------------------------------------------------------------------------- stage (route 2)

STAGE_RENDER_VERSION = "1022b"   # bump when the staging camera / cleanup rules change: every render is redone


def render_sig(cx: Ctx, e: dict) -> str:
    """What a moment's render is made of: its staged shot (the motion's copy) + the model.json (extent, fit, rest) of
    every prop it shows or holds. A changed prop model re-renders the moments that use it (10-21: the bellows and the
    oilcloth were re-oriented after 98 renders)."""
    msj = cx.OUT / "stage" / "motion" / e["id"] / "shot.json"
    src = msj if msj.exists() else Path(e["shot"]) if e.get("shot") else None
    if src is None or not src.exists():
        return ""
    h = hashlib.sha1(src.read_bytes())
    try:
        sh = json.loads(src.read_text())
    except ValueError:
        return h.hexdigest()[:16]
    names = {str(p.get("prop") or p.get("name") or "") for p in (sh.get("set") or {}).get("props", [])} | \
            {str(v) for f in sh.get("figures", []) for v in (f.get("holds") or {}).values() if v}
    pd = cx.W / "video" / "stage" / "props"
    for n in sorted(names):
        mj = pd / n.split()[0].lower() / "model.json"
        if n and mj.exists():         # the GEOMETRY only (10-21: a QA-only field, rest_check, re-rendered all 98)
            try:
                g = json.loads(mj.read_text())
                h.update(n.encode() + json.dumps([g.get(k) for k in ("extent_m", "scale", "fit", "joints")]).encode())
            except ValueError:
                h.update(n.encode() + mj.read_bytes())
    # each figure's MODEL (its paint version + rig: 10-22, re-painted models kept their old ghost-hand renders) and the
    # staging RENDER version (camera / cleanup rules: 10-22, six walled-in cameras were reused after the fix)
    h.update(STAGE_RENDER_VERSION.encode())
    for f in sh.get("figures", []):
        m = f.get("model")
        if m:
            d_ = Path(m).parent
            for x in ("paint.sig", "rig.json"):
                if (d_ / x).exists():
                    h.update(x.encode() + (d_ / x).read_bytes())
    return h.hexdigest()[:16]


def needs_clip(e: dict) -> bool:
    """A moment whose MOTION is the point (a strike, sparks, a recipe, a creature, a carried person, a worked prop such
    as the bellows pump): its whole clip is
    rendered at staging, to be looked at and checked there. The rest render their start/end stills at staging and
    their clip in the video step (user 10-21: 'that's like the strike moments')."""
    try:
        sh = json.loads(Path(e["shot"]).read_text())
    except (OSError, ValueError, TypeError, KeyError):
        return True
    return bool(sh.get("contacts") or sh.get("fx") or sh.get("recipe") or sh.get("operate")
                or any(g.get("kind") == "creature" or g.get("attach") for g in sh.get("figures", [])))


def stage(cx: Ctx, P: dict, steps: str = "motion,render,draw", workers: int = 2, seeds: int = 3, card: str = "05",
          clips: bool = False) -> None:
    """Route 2 in three resumable steps: motion (motion_batch: HY-Motion on one card for every human + the creature
    engine), render (motion_batch --render -> blockout_video --target ltx --stills, CPU/Blender, `workers` in parallel), draw
    (keyframe_draw, one card). Any failure demotes that entry to route 3 (OUT/stage/<id>/FAILED).
    clips=False (the stage step): the full clip only for needs_clip() moments, the others their stills
    (blockout_video --key-frames); clips=True (the video step): every moment's clip (stills kept)."""
    sd = cx.OUT / "stage"
    sd.mkdir(exist_ok=True)
    mb = sd / "motion"
    ents = [e for e in P["entries"] if e["route"] == "stage" or (e["route"] == "talk" and full3d() and e.get("shot"))]

    def fail(e, why):
        d = sd / e["id"]
        d.mkdir(exist_ok=True)
        (d / "FAILED").write_text(str(why))
        log_entry(cx.OUT, {"phase": "stage", "id": e["id"], "ok": False, "demoted": "i2v", "why": str(why)[:400]})

    alive = lambda: [e for e in ents if not (sd / e["id"] / "FAILED").exists()]  # noqa: E731
    if "motion" in steps:
        rep0 = json.loads((mb / "report.json").read_text()) if (mb / "report.json").exists() else {"shots": {}}
        # a moment missing from the report re-runs too (10-21: report.json is rewritten per batch, so 9 dragon shots
        # with motion on disk but no report entry failed as "motion_batch produced nothing")
        # ... and a moment whose input shot changed since its motion ran (input.sig), or a retried failure (10-21: the
        # cached default-staged motion of 2 King moments would have failed again after the scene layer placed him)
        def _sig(e):                                 # the shot + the director's beats (the motion follows them, 10-23)
            if not (e.get("shot") and Path(e["shot"]).exists()):
                return ""
            bt = json.dumps(cx.frames.get(e["id"], {}).get("beats") or [], sort_keys=True).encode()
            return hashlib.sha1(Path(e["shot"]).read_bytes() + (bt if bt != b"[]" else b"")).hexdigest()[:16]

        def _stale(e):
            q = mb / e["id"] / "input.sig"
            return (sd / e["id"] / "FAILED.prev").exists() or (q.exists() and q.read_text().strip() != _sig(e)) or \
                bool((rep0.get("shots", {}).get(e["id"]) or {}).get("failure"))   # a failed build is retried
        todo = [e for e in alive() if e["route"] == "stage"
                and (not (mb / e["id"] / "shot.json").exists() or e["id"] not in rep0.get("shots", {}) or _stale(e))]
        for e in todo:                                   # its old render belongs to the old motion
            if (sd / e["id"] / "render").exists():
                shutil.rmtree(sd / e["id"] / "render")
        if todo:
            items = [{"id": e["id"], "frame": f"{cx.A / 'keyframes.json'}#{e['id']}", "dur": e["dur"],
                      **({"shot": e["shot"]} if e.get("shot") and (full3d() or str(cx.shots_dir) in e["shot"]) else {})}
                     for e in todo]                      # no override: motion_batch plans with the current planner
            (sd / "list.json").write_text(json.dumps(items, indent=1))
            mem = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1048576
            if mem < 15:
                raise SystemExit(f"MemAvailable {mem} GiB < 15: not loading HY-Motion")
            t = time.time()
            r = subprocess.run([str(POSE_PY), "-I", str(HERE / "motion_batch.py"), str(sd / "list.json"), str(mb), "--checks"],
                               capture_output=True, text=True)
            (sd / "motion_batch.log").write_text(r.stdout + r.stderr)
            log_entry(cx.OUT, {"phase": "stage-motion", "rc": r.returncode, "n": len(todo), "sec": round(time.time() - t)})
            new_rep = json.loads((mb / "report.json").read_text()) if (mb / "report.json").exists() else {"shots": {}}
            (mb / "report.json").write_text(json.dumps({**new_rep, "shots": {**rep0.get("shots", {}),
                                                                             **new_rep.get("shots", {})}}, indent=1))
            for e in todo:                   # only a BUILT motion is current (10-23: the killing thrust failed its
                f_ = (new_rep.get("shots", {}).get(e["id"]) or {}).get("failure")    # build, kept the sig and
                if f_:                       # was staged from the previous day's joints - an upright dragon)
                    print(f"!!! {e['id']}: motion build failed ({str(f_)[:200]}): staged from its old motion", flush=True)
                elif (mb / e["id"] / "shot.json").exists():
                    (mb / e["id"] / "input.sig").write_text(_sig(e))
            kernel_check(cx.OUT)
    rep = json.loads((mb / "report.json").read_text()) if (mb / "report.json").exists() else {"shots": {}}

    def render_one(e):
        d = sd / e["id"]
        full = e["route"] != "stage" or clips or needs_clip(e)
        rs_ = d / "render" / "inputs.sig"
        if rs_.exists() and rs_.read_text().strip() != render_sig(cx, e):
            shutil.rmtree(d / "render")                   # a prop model or the shot changed since this render
            for k_ in ("key.png", "key_end.png", "key_draw.png"):   # ... and so the keys painted on it
                if (d / k_).exists():
                    (d / k_).replace(d / f"{k_[:-4]}.stale.png")
        have_stills = (d / "render" / "start_color.png").exists()
        if have_stills and (not full or (d / "render" / "depth.mp4").exists()):
            return
        if (sd / "motion" / f"_hold_{e['id']}").exists() or (cx.OUT / "render_hold.txt").exists() and \
                e["id"] in (cx.OUT / "render_hold.txt").read_text().split():
            return                             # another agent owns this render (its message releases it)
        ltx_wait()
        d.mkdir(exist_ok=True)
        try:                                   # a claim, so parallel render processes never take the same shot
            os.close(os.open(d / ".rendering", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            return
        R = rep["shots"].get(e["id"], {})
        try:
            if e["route"] == "talk":     # a talking moment: a still of the staged scene (A2Vid animates the speech)
                for g in json.loads(Path(e["shot"]).read_text()).get("figures", []):
                    mdl = g.get("model")
                    mdl = mdl if isinstance(mdl, str) else (mdl or {}).get("blend", "")
                    if g.get("kind", "human") == "human" and mdl and g["who"].lower().split()[-1] not in mdl.lower():
                        raise RuntimeError(f"stand-in body for {g['who']} ({Path(mdl).parent.name})")
                env = dict(os.environ, BLOCKOUT_ASSETS=str(cx.W / "video" / "stage" / "assets.json"),
                           BLOCKOUT_ALLOW_VIOLATIONS="1", BLOCKOUT_THREADS=str(max(4, (os.cpu_count() or 16) // workers)))
                r = None
                for attempt in range(2):          # one retry, then a loud fallback
                    shot_f = talk_shot(cx, e, d) if TALK_SINGLE else e["shot"]
                    r = subprocess.run(["nice", "-n", "19", str(POSE_PY), "-I", str(HERE / "blockout_video.py"), shot_f, str(d / "render"),
                                        "--frames", "9", "--size", f"{TALK_W}x{TALK_H}", "--stills", "--layers", "figures=figure+prop"],
                                       capture_output=True, text=True, env=env, timeout=1800)
                    if (d / "render" / "start_color.png").exists():
                        break
                (d / "render.log").write_text(r.stdout + r.stderr)
                if not (d / "render" / "start_color.png").exists():
                    raise RuntimeError(f"talk still rc={r.returncode}: " + (r.stdout + r.stderr)[-300:])
                (d / "render" / "inputs.sig").write_text(render_sig(cx, e))
                (d / "FAILED.prev").unlink(missing_ok=True)     # the retry worked: not stale on every later run
                log_entry(cx.OUT, {"phase": "stage-render", "id": e["id"], "talk": True, "rc": r.returncode})
                return
            if R.get("failure") or not R:
                raise RuntimeError(R.get("failure") or "motion_batch produced nothing")
            if any("DEFAULT STAGING" in w for w in R.get("warnings", [])):
                raise RuntimeError("; ".join(R["warnings"]))
            msj = mb / e["id"]
            # a person staged on ANOTHER character's mesh (no rig of their own) gets that character's body and
            # clothes in the colour still, and the keyframe draws the wrong person (10-20: Hulda as a man in
            # Edric's leathers in 7 frames): not stageable -> I2V from the picture keyframe
            for g in json.loads((msj / "shot.json").read_text()).get("figures", []):
                mdl = g.get("model")
                mdl = mdl if isinstance(mdl, str) else (mdl or {}).get("blend", "")
                if g.get("kind", "human") == "human" and mdl and g["who"].lower().split()[-1] not in mdl.lower():
                    raise RuntimeError(f"stand-in body for {g['who']} ({Path(mdl).parent.name})")
            if not (msj / "render.json").exists():
                raise RuntimeError("motion_batch wrote no render.json (re-run the motion step)")
            d.mkdir(exist_ok=True)
            # nice 19: CPU renders starved the LTX drivers (10-20: load 50 on 32 cores -> A2Vid 2.7x slower, cards at 0 MHz)
            vj0 = d / "render" / "video.json"        # the stills-time result, to compare with the clip's own checks
            v_before = (json.loads(vj0.read_text()).get("violations") or 0) if have_stills and vj0.exists() else None
            # stills once (the clip pass keeps them: the keyframe was painted on them); the clip only when needed
            cmd = ["nice", "-n", "19", str(POSE_PY), "-I", str(HERE / "motion_batch.py"), "--render", str(msj), str(d / "render"),
                   "--", "--target", "ltx"] + ([] if have_stills else ["--stills"]) + ([] if full else ["--key-frames"])
            env = dict(os.environ, BLOCKOUT_ASSETS=str(cx.W / "video" / "stage" / "assets.json"),
                       BLOCKOUT_ALLOW_VIOLATIONS="1", BLOCKOUT_THREADS=str(max(4, (os.cpu_count() or 16) // workers)))
            t = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
            (d / "render.log").write_text(r.stdout + r.stderr)
            if (full and not (d / "render" / "depth.mp4").exists()) or not (d / "render" / "start_color.png").exists():
                raise RuntimeError(f"motion_shot rc={r.returncode}: " + (r.stdout + r.stderr)[-300:])
            vj = d / "render" / "video.json"
            viol = json.loads(vj.read_text()).get("violations") if vj.exists() else None
            (d / "render" / "inputs.sig").write_text(render_sig(cx, e))
            (d / "FAILED.prev").unlink(missing_ok=True)     # the retry worked: not stale on every later run
            # the clip drawn LATER (video step) runs Blender's hand check on every frame: new misses are reported at
            # once, as the staging summary would have (user 10-21)
            if v_before is not None and full and (viol or 0) > v_before and ALERT:
                ALERT(f"  [3d] !!! {e['id']}: the full clip found {(viol or 0) - v_before} more grip/contact "
                      f"violation(s) than its staging stills (see {d / 'render' / 'checks.json'} / render.log)")
            log_entry(cx.OUT, {"phase": "stage-render", "id": e["id"], "rc": r.returncode, "violations": viol,
                               "sec": round(time.time() - t), "rendered": "clip" if full else "keys"})
        except Exception as ex:  # noqa: BLE001
            fail(e, ex)
        finally:                 # the claim ends with this attempt (10-21: a failed render's claim made the retry skip it)
            (d / ".rendering").unlink(missing_ok=True)

    if "render" in steps:
        from concurrent.futures import ThreadPoolExecutor
        # the 1-render cap protects a RUNNING LTX gen (10-20: renders starved its drivers); with no LTX running the
        # stage step's renders run in parallel (10-21: 95 stills one at a time = ~100 min of CPU)
        if subprocess.run(["pgrep", "-f", "ltx_split(_next)?[.]py gen"], capture_output=True, text=True).stdout.strip():
            workers = min(workers, RENDER_WORKERS)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(render_one, alive()))
    if "draw" in steps:
        ov = cx.OUT / "draw_seeds.txt"           # a live override for a running loop (coordinator: time budget)
        if ov.exists():
            seeds = int(ov.read_text().strip() or seeds)
        shots, ids = [], []
        for e in alive():
            d = sd / e["id"]
            if (d / "key.png").exists() or not (d / "render" / "start_color.png").exists():
                continue
            f = cx.frames.get(e["id"], {})
            who = moment_who(cx, e)
            reg = regional() and len([w for w in who if not is_creature(cx.chars.get(w, ""))]) >= 2   # one person: refs can't mix
            if reg:
                matte_behind(cx, e)
            # people's image refs are OFF by default (10-20: the edit model took identity/gender from the wrong
            # reference - Hulda a bearded man in every ref-conditioned key); creatures keep theirs
            if PERSON_REFS_MODE == "sheets" and not reg:
                refs = [str(x) for w in who if (x := ref_sheet(cx.refs, w))]
                idx = {}
            else:
                refs, idx = key_refs(cx, e, who if PERSON_REFS and not reg else [w for w in who if is_creature(cx.chars.get(w, ""))])
            pl = ref_sheet(cx.refs, f.get("where") or "")
            if pl and len(refs) < 10:
                refs.append(str(pl))
            sh = {"stills": str(d / "render"), "frame": "start", "prompt": draw_prompt(cx, e, idx, scene_only=reg), "refs": refs[:10],
                  "out": str(d / "key.png"), "seeds": seeds, "strength": KEY_STRENGTH, "depth": 1.0, "edges": 0.6,
                  "fx": bool(re.search(r"spark|strike|hammer|fire|flame", (f.get("motion") or "").lower())),
                  "likeness": [f"{w}={x}" for w in who if (x := ref_sheet(cx.refs, w)) and not is_creature(cx.chars.get(w, ""))],
                  "lora": staged_loras(cx, who, d / "render" / "start_expect.json")}
            if sh["fx"]:
                sh["edges"] = 0.3                 # sparks: weaker edges (keyframe_draw's own fx rule)
            if KEY_STRENGTH_BG > 0:               # per-pixel: the people at KEY_STRENGTH, the set at KEY_STRENGTH_BG
                fm = figure_mask(cx, e, (832, 448)) if (d / "render" / "start_color.png").exists() else None
                if fm is not None:
                    from PIL import Image as _I, ImageFilter as _F
                    fm = fm.resize(_I.open(d / "render" / "start_color.png").size).filter(_F.MaxFilter(7))
                    fm.save(d / "render" / "start_figmask.png")
                    sh["mask"], sh["strength_bg"] = str(d / "render" / "start_figmask.png"), KEY_STRENGTH_BG
            cm_ = d / "render" / "start_color_creature_mask.png"
            if cm_.exists():                      # a creature in frame: the simple 3D shape painted over hard inside its
                sh["mask"], sh["strength"], sh["strength_bg"] = str(cm_), CREATURE_STRENGTH, KEY_STRENGTH_BG  # mask
            shots.append(sh)
            ids.append(e)
            if camera_path(e.get("shot")) and (d / "render" / "end_color.png").exists() and not (d / "key_end.png").exists():
                shots.append({**sh, "frame": "end", "out": str(d / "key_end.png")})   # the move's last camera
        if shots and (cx.OUT / ".pause_gpu05").exists():
            print("draw: paused (.pause_gpu05: talk LTX has the cards first)")
            shots = []
        if shots and len(shots) < DRAW_MIN_BATCH and not (cx.OUT / ".renders_done").exists():
            print(f"draw: {len(shots)} ready, waiting for {DRAW_MIN_BATCH} (one Qwen load per batch)")
            shots = []
        if shots:
            lf = sd / f"draw_list_{card}_{time.strftime('%H%M%S')}.json"
            lf.write_text(json.dumps(shots, indent=1))
            mem = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1048576
            if mem < 15:
                raise SystemExit(f"MemAvailable {mem} GiB < 15: not loading Qwen")
            t = time.time()
            # the tool takes /tmp/gpu-<card>.lock per ~12-min chunk itself and yields to a waiting LTX gaprun: no flock
            # here. card "any": the list split over BOTH cards, two draw processes at once (no idle card, 10-20)
            cards = ["05", "65"] if card == "any" else [card]
            parts = [shots[k::len(cards)] for k in range(len(cards))]

            def draw_on(cd, part):
                if not part:
                    return None
                lp = lf.with_name(f"{lf.stem}_{cd}.json")
                lp.write_text(json.dumps(part, indent=1))
                hp = subprocess.run([str(HOME / "bin" / "hip-index"), cd], capture_output=True, text=True).stdout.strip()
                rr = subprocess.run([str(POSE_PY), "-I", str(HERE / "keyframe_draw.py"), "--list", str(lp), "--lock",
                                     f"/tmp/gpu-{cd}.lock"], capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=hp))
                (sd / f"{lp.stem}.log").write_text(rr.stdout + rr.stderr)
                return rr
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(len(cards)) as ex_:
                rs = [x for x in ex_.map(lambda a_: draw_on(*a_), zip(cards, parts)) if x is not None]
            r = rs[0] if rs else subprocess.CompletedProcess([], 0, "", "")
            hip = subprocess.run([str(HOME / "bin" / "hip-index"), cards[0]], capture_output=True, text=True).stdout.strip()
            log_entry(cx.OUT, {"phase": "stage-draw", "n": len(shots), "rc": r.returncode, "sec": round(time.time() - t)})
            kernel_check(cx.OUT)
            retry = []
            for e, sh in zip(ids, [x for x in shots if x.get("frame", "start") == "start"]):
                d = sd / e["id"]
                if (not (d / "key.png").exists() or weak_key(d)) and seeds < 4:   # none passed / weak: one more seed
                    retry.append({**sh, "seeds": 1, "seed0": 7 + 4 * seeds, "out": str(d / "key4.png")})
            if retry:
                lf2 = sd / f"draw_retry_{card}_{time.strftime('%H%M%S')}.json"
                lf2.write_text(json.dumps(retry, indent=1))
                r = subprocess.run([str(POSE_PY), "-I", str(HERE / "keyframe_draw.py"), "--list", str(lf2), "--lock",
                                    f"/tmp/gpu-{cards[0]}.lock"], capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
                (sd / f"{lf2.stem}.log").write_text(r.stdout + r.stderr)
                kernel_check(cx.OUT)
            for e in ids:
                d = sd / e["id"]
                if not (d / "key.png").exists() and (d / "key4.png").exists():
                    shutil.copy(d / "key4.png", d / "key.png")
                elif (d / "key.png").exists() and (d / "key4.png").exists() and \
                        _key_score(d / "key4_draw.json") > _key_score(d / "key_draw.json"):
                    shutil.copy(d / "key4.png", d / "key.png")    # the extra seed beat a weak key
                if not (d / "key.png").exists():
                    best = d / "key_best.png"
                    if best.exists():
                        shutil.copy(best, d / "key.png")   # no seed passed every check: best by fewest failures
                        log_entry(cx.OUT, {"phase": "stage-draw", "id": e["id"], "note": "no seed passed: best used"})
                    else:
                        fail(e, "keyframe_draw produced nothing")
            if regional():
                regional_pass(cx, [e for e in ids if (sd / e["id"] / "key.png").exists()], card)
        if FACE_PASS and full3d() and "draw" in steps:   # the face-detail pass on every key (one person's LoRA + words)
            face_pass(cx, [e for e in alive() if (sd / e["id"] / "key.png").exists() and not (sd / e["id"] / "key_draw.png").exists()], card)


KEY_SEEDS = int(os.environ.get("BOOKCASTER_KEY_SEEDS", "2"))
WEAK_LIKENESS = 0.40


def _key_score(draw_json: Path) -> float:
    try:
        j = json.loads(draw_json.read_text())
    except (OSError, ValueError):
        return -9.0
    ch = next((r for r in j.get("results", []) if r.get("seed") == j.get("chosen_seed")), None)
    return float(ch.get("score", 0.0)) if ch else -9.0


def weak_key(d: Path) -> bool:
    """A passing key worth ONE more seed: a glowing prop that faded (hot_ok false) or a very poor face match.
    (10-21, 32 knight keys: all 3 seeds passed every time; 2 seeds lose 0.014 score on average; a likeness bar
    high enough to predict when seed 3 wins would fire on 20 of 32 keys, so only the clear failures retry.)"""
    try:
        j = json.loads((d / "key_draw.json").read_text())
    except (OSError, ValueError):
        return False
    ch = next((r for r in j.get("results", []) if r.get("seed") == j.get("chosen_seed")), None)
    if not ch:
        return False
    lk = ch.get("likeness") or {}
    return ch.get("hot_ok") is False or bool(lk and sum(lk.values()) / len(lk) < WEAK_LIKENESS)


def regional() -> bool:
    """Keys drawn in two passes (10-20: several people's refs in ONE prompt mixed identities - Hulda painted as a grey-
    haired man): a global scene pass, then one regional pass per person with only that person's refs + words."""
    return full3d() and os.environ.get("REMAKE_REGIONAL", "0") == "1"     # OFF until it passes the golden gate


# people's image conditions: "sheets" = their outfit sheets only, unnamed (the 09:17 golden config), "named" = headshot +
# outfit sheet named by index (10-20 11:29: Hulda drawn as a man - REGRESSION), "off" = none (LoRA + look words only)
PERSON_REFS_MODE = os.environ.get("REMAKE_PERSON_REFS", "off")
# img2img strength of a staged key: the 3D still carries identity/outfit/layout/set, the painter only refines (10-20: at
# 0.85 the 4-step Lightning painter regenerated from text - a door, a stormy sky, Hulda turned into a man)
KEY_STRENGTH = float(os.environ.get("REMAKE_KEY_STRENGTH", "0.375"))     # user 10-20: people 0.375 / set 0.50 (golden arm C)
KEY_STRENGTH_BG = float(os.environ.get("REMAKE_KEY_STRENGTH_BG", "0.50") or 0)
CREATURE_STRENGTH = float(os.environ.get("REMAKE_CREATURE_STRENGTH", "0.85"))  # inside blockout's creature mask (coordinator 10-20)   # >0: per-pixel strength, the set at this
PERSON_REFS = PERSON_REFS_MODE == "named"
TRIGGERS = os.environ.get("REMAKE_TRIGGERS", "0") == "1"           # the LoRA trigger words in the prompt (untested)
REG_FRONT, REG_BACK = 0.6, 0.8      # img2img strength in a person's own crop (a back-turned head is repainted harder)


def person_words(cx: "Ctx", w: str) -> str:
    """'an elderly woman of about 70' - gender + age from the look, so the painter can't swap them."""
    d = cx.chars.get(w, "")
    if is_creature(d):
        return w
    g, a = gender_of(d), age_of(d)
    if a is None:
        return f"a {g}"
    return (f"an elderly {g} of about {int(a)}" if a >= ELDER_AGE else f"a young {g} of about {int(a)}" if a < 25
            else f"a {g} of about {int(a)}")


def behind_of(e: dict) -> str | None:
    """The over-the-shoulder listener seen from behind (camera.ots[0] of a 2-person OTS shot)."""
    try:
        cam = json.loads(Path(e["shot"]).read_text()).get("camera") or {}
    except (OSError, ValueError, TypeError, KeyError):
        return None
    ots = cam.get("ots") or []
    return ots[0] if len(ots) >= 2 else None


def model_dir(cx: "Ctx", e: dict, who: str) -> Path | None:
    try:
        figs = json.loads(Path(e["shot"]).read_text()).get("figures", [])
    except (OSError, ValueError, TypeError, KeyError):
        return None
    m = next((g.get("model") for g in figs if g.get("who") == who), None)
    m = m if isinstance(m, str) else (m or {}).get("blend")
    return Path(m).parent if m else None


def back_ref(cx: "Ctx", e: dict, who: str) -> str | None:
    d = model_dir(cx, e, who)
    for p in ((d / "apose_back.png") if d else None, cx.W / "video" / "stage" / "characters" / slug(who) / "apose_back.png"):
        if p and p.exists():
            return str(p)
    return None


def _boxes(rd: Path) -> tuple[dict, tuple]:
    ex = json.loads((rd / "start_expect.json").read_text()) if (rd / "start_expect.json").exists() else {}
    return {f["who"]: f["bbox"] for f in ex.get("figures", []) if f.get("bbox")}, tuple(ex.get("size", [832, 448]))


def _pad(b, W, H, f=0.15):
    x0, y0, x1, y1 = b
    px, py = f * (x1 - x0), f * (y1 - y0)
    return [int(max(0, x0 - px)), int(max(0, y0 - py)), int(min(W, x1 + px)), int(min(H, y1 + py))]


def person_mask(cx: "Ctx", e: dict, who: str, size: tuple):
    """That person's silhouette: the figures-only pass inside their padded box (L, 255 = them), or the box."""
    from PIL import Image, ImageDraw
    rd = cx.OUT / "stage" / e["id"] / "render"
    boxes, (W, H) = _boxes(rd)
    if who not in boxes:
        return None
    b = _pad(boxes[who], W, H, 0.06)
    sx, sy = size[0] / W, size[1] / H
    b = [int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)]
    box = Image.new("L", size, 0)
    ImageDraw.Draw(box).rectangle(b, fill=255)
    fm = figure_mask(cx, e, size)
    if fm is None:
        return box
    from PIL import ImageChops
    return ImageChops.multiply(fm, box)


def matte_behind(cx: "Ctx", e: dict) -> None:
    """The back-turned OTS listener's Cycles hair is glossy and the painter reads it as METAL (10-20): flatten that
    silhouette in the init still (blur + highlights compressed toward the mid tone) and drop its inner edges (the
    outline stays), once; originals kept as start_color_orig.png / start_edges_orig.png."""
    from PIL import Image, ImageFilter, ImageChops
    w = behind_of(e)
    rd = cx.OUT / "stage" / e["id"] / "render"
    if not w or not (rd / "start_color.png").exists() or (rd / "start_color_orig.png").exists():
        return
    col = Image.open(rd / "start_color.png").convert("RGB")
    m = person_mask(cx, e, w, col.size)
    if m is None:
        return
    shutil.copy(rd / "start_color.png", rd / "start_color_orig.png")
    flat = col.filter(ImageFilter.GaussianBlur(4))
    flat = flat.point(lambda v: int(v if v < 110 else 110 + (v - 110) * 0.35))      # no specular highlights
    col.paste(flat, (0, 0), m.filter(ImageFilter.GaussianBlur(2)))
    col.save(rd / "start_color.png")
    if (rd / "start_edges.png").exists():
        shutil.copy(rd / "start_edges.png", rd / "start_edges_orig.png")
        ed = Image.open(rd / "start_edges.png").convert("RGB")
        inner = m.filter(ImageFilter.MinFilter(9))                    # eroded: the outline survives
        ed.paste((0, 0, 0), (0, 0), inner)
        ed.save(rd / "start_edges.png")


def regional_pass(cx: "Ctx", ents: list, card: str = "05") -> None:
    """Per person, in their own crop of the global key: img2img with ONLY their refs (headshot + outfit sheet, or
    their back view + outfit sheet when they are the back-turned OTS listener), their LoRA + trigger, their own
    gender/age + look words, the crop's depth + edges as control; pasted back through their silhouette (the
    back-turned one last: they are in front). key.png -> key_global.png (kept), key.png = the result."""
    from PIL import Image, ImageFilter
    jobs, plan = [], []
    for e in ents:
        d = cx.OUT / "stage" / e["id"]
        rd = d / "render"
        if (d / "key_global.png").exists() and (d / "key.png").exists() and (d / "regional.json").exists():
            continue
        boxes, (W, H) = _boxes(rd)
        beh = behind_of(e)
        try_n = len(list(d.glob("key_try*.png")))
        key = Image.open(d / "key.png").convert("RGB")
        for w in [x for x in moment_who(cx, e) if x in boxes and not is_creature(cx.chars.get(x, ""))]:
            b = _pad(boxes[w], W, H, 0.15)
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bw < 48 or bh < 48:
                continue
            sc = min(1024 / max(bw, bh), 2.5)
            cw, ch = max(256, int(bw * sc) // 16 * 16), max(256, int(bh * sc) // 16 * 16)
            rg = d / "regional"
            rg.mkdir(exist_ok=True)
            sw = slug(w)
            key.resize((W, H)).crop(b).resize((cw, ch), Image.LANCZOS).save(rg / f"{sw}_init.png")
            for nm, src in (("depth", "start_depth.png"), ("edges", "start_edges.png")):
                Image.open(rd / src).convert("RGB").resize((W, H)).crop(b).resize((cw, ch), Image.LANCZOS).save(rg / f"{sw}_{nm}.png")
            back = w == beh
            sh = outfit_sheet(cx, e, w)
            lr = char_lora(cx, w)
            trig = f"chr_{slug(w)}, " if lr else ""
            look = cx.look(w, e["t"]).rstrip(".")
            if back:
                hair = ", ".join(re.findall(r"[^,;]*\bhair\b[^,;]*", look)) or "their hair"
                refs = [x for x in (back_ref(cx, e, w), sh) if x] if PERSON_REFS else []
                prompt = (f"{trig}Seen from directly behind: the back of the head and shoulders of {w}, {person_words(cx, w)}; "
                          f"{hair.strip()}, soft natural matte hair, not metal, not shiny; {look}. "
                          + ("The back view of image 1, exactly the clothes of image 2. " if refs else "")
                          + f"The face is not visible. {style_for(cx, e)}. No text.")
            else:
                hs = headshot(cx, w, sh) if sh and PERSON_REFS else None
                refs = [x for x in (hs, sh) if x] if PERSON_REFS else []
                prompt = (f"{trig}{w}, {person_words(cx, w)}: {look}. "
                          + ("The face of image 1, exactly the clothes of image 2. " if refs else "")
                          + f"Same pose, same framing, same lighting. {style_for(cx, e)}. No text.")
            seed = 501 + 7 * try_n + sum(map(ord, w)) % 50
            j = {"id": f"{e['id']}__{sw}", "prompt": no_quotes(prompt), "seed": seed, "size": [cw, ch], "refs": refs,
                 "path": str(rg / f"{sw}.png"),
                 "control": [{"image": str(rg / f"{sw}_depth.png"), "type": "depth", "scale": 1.0},
                             {"image": str(rg / f"{sw}_edges.png"), "type": "canny", "scale": 0.3 if back else 0.5, "prepared": True}],
                 "init": {"image": str(rg / f"{sw}_init.png"), "strength": REG_BACK if back else REG_FRONT}}
            if lr:
                j["loras"] = [[lr, 1.0]]
            Path(j["path"]).unlink(missing_ok=True)
            jobs.append(j)
            plan.append((e, w, b, back))
    if not jobs:
        return
    cards = ["05", "65"] if card == "any" else [card]
    from concurrent.futures import ThreadPoolExecutor
    t = time.time()
    with ThreadPoolExecutor(len(cards)) as ex_:
        list(ex_.map(lambda k: _qwen(cx, jobs[k::len(cards)], cx.OUT / "stage", cards[k], f"regional_{cards[k]}_{time.strftime('%H%M%S')}")
                     if jobs[k::len(cards)] else None, range(len(cards))))
    log_entry(cx.OUT, {"phase": "regional", "jobs": len(jobs), "sec": round(time.time() - t)})
    by = {}
    for (e, w, b, back), j in zip(plan, jobs):
        by.setdefault(e["id"], (e, []))[1].append((w, b, back, j))
    for i, (e, parts) in by.items():
        d = cx.OUT / "stage" / i
        if not (d / "key_global.png").exists() or not (d / "regional.json").exists():
            shutil.copy(d / "key.png", d / "key_global.png")
        base = Image.open(d / "key_global.png").convert("RGB")
        W, H = base.size
        _, (W0, H0) = _boxes(d / "render")
        done = []
        for w, b, back, j in sorted(parts, key=lambda x: x[2]):      # the back-turned (foreground) one last
            if not Path(j["path"]).exists():
                continue
            sx, sy = W / W0, H / H0
            bb = [int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)]
            patch = Image.open(j["path"]).convert("RGB").resize((bb[2] - bb[0], bb[3] - bb[1]), Image.LANCZOS)
            m = person_mask(cx, e, w, (W, H))
            if m is None:
                continue
            m = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(3)).crop(bb)
            base.paste(patch, bb[:2], m)
            done.append(w)
        base.save(d / "key.png")
        (d / "regional.json").write_text(json.dumps({"people": done, "jobs": [p[3]["id"] for p in parts]}, indent=1))


FACE_STRENGTH = float(os.environ.get("REMAKE_FACE_STRENGTH", "0.55"))
FACE_STRENGTH_F = float(os.environ.get("REMAKE_FACE_STRENGTH_F", "0.65"))   # women: a gaunt 3D sculpt reads male (10-20)
FACE_PASS = os.environ.get("REMAKE_FACE_PASS", "1") == "1"
FACE_DEPTH = float(os.environ.get("REMAKE_FACE_DEPTH", "0.8"))   # the still's depth on the face crop (1.0 on a turned head)
FACE_POSE = float(os.environ.get("REMAKE_FACE_POSE", "0.9"))   # the painted face's landmarks as pose control
FACE_REF_MAX_YAW = 35.0      # a front headshot as image ref only up to this head turn (degrees from the camera)
FACE_MAX_YAW = 75.0          # beyond: a near-profile, left as the main pass drew it
FACE_REF = os.environ.get("REMAKE_FACE_REF", "1") == "1"      # the headshot as the face pass single ref (coordinator GO 10-20)


def face_words(cx: "Ctx", w: str) -> str:
    """The face-pass words from the look: gender + age stated hard (a soft, gaunt 3D sculpt read as a man, 10-20)."""
    d = cx.chars.get(w, "")
    g, a = gender_of(d), age_of(d)
    hair = ", ".join(re.findall(r"[^,;.]*\bhair\b[^,;.]*", d)).strip()
    if g == "woman":
        old_ = a is not None and a >= ELDER_AGE
        t = (f"an old woman in her {int(a // 10 * 10)}s, a grandmotherly face, soft rounded features, thin lips, a small chin, "
             f"a feminine face, deep wrinkles" if old_ else
             f"a woman of about {int(a)}, a feminine face" if a else "a woman, a feminine face")
        t += ", soft feminine features; not a man, no beard, no stubble, no moustache"
    else:
        t = person_words(cx, w) + ", " + ("a masculine face" if g == "man" else "")
        if re.search(r"clean-shaven|beardless|no beard", d, re.I):
            t += ", clean-shaven, a smooth jaw, no beard, no stubble, no moustache"
    return t + (f", {hair}" if hair else "")


def _faces(paths: list) -> dict:
    """DWPose face boxes for many pictures, one load (face_batch's face stage)."""
    r = subprocess.run([str(POSE_PY), "-I", str(HERE / "face_batch.py"), "--_faces"], input=json.dumps([str(p) for p in paths]),
                       capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {}


def face_pass(cx: "Ctx", ents: list, card: str = "05", key: str = "key.png") -> dict:
    """ADetailer-style FACE-DETAIL pass after the main key (10-20): each visible face (DWPose), mapped to its staged
    figure, cropped at 2.2x its box (>= 512 px), img2img at FACE_STRENGTH with ONLY that character's LoRA + their face
    words (gender/age stated hard), the still's depth + soft edges as control, feather-blended back. Back-turned or
    tiny faces are skipped; creatures too. <key> -> <key stem>_preface.png (kept), <key> = the result.
    -> {id: [{who, box, crop}]}"""
    from PIL import Image, ImageDraw, ImageFilter
    items = []
    for e in ents:
        d = cx.OUT / "stage" / e["id"]
        if (d / key).exists() and not (d / key.replace(".png", "_preface.png")).exists():
            items.append((e, d))
    if not items:
        return {}
    F = _faces([d / key for e, d in items])
    jobs, plan = [], []
    for e, d in items:
        im = Image.open(d / key).convert("RGB")
        W, H = im.size
        boxes, (W0, H0) = _boxes(d / "render")
        sx, sy = W / W0, H / H0
        beh = behind_of(e)
        for f in (F.get(str(d / key)) or {}).get("faces", []):
            x0, y0, x1, y1 = f["box"]
            fw, fh = x1 - x0, y1 - y0
            if fw < 28:
                continue
            cxf, cyf = (x0 + x1) / 2, (y0 + y1) / 2
            cand = [(w, b) for w, b in boxes.items() if b[0] * sx <= cxf <= b[2] * sx and b[1] * sy - 0.1 * H <= cyf <= b[3] * sy]
            if not cand:
                continue
            w = min(cand, key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]))[0]
            if w == beh or is_creature(cx.chars.get(w, "")) or not cx.chars.get(w) or facing_away(d / "render", w):
                continue          # back-turned: never paint a face onto the back of a head (10-20, 793)
            side = max(fw, fh) * 2.2
            c0 = [int(max(0, cxf - side / 2)), int(max(0, cyf - side / 2)), int(min(W, cxf + side / 2)), int(min(H, cyf + side / 2))]
            cw, ch = c0[2] - c0[0], c0[3] - c0[1]
            sc = max(512 / max(cw, ch), 1.0)
            tw, th = max(256, int(cw * sc) // 32 * 32), max(256, int(ch * sc) // 32 * 32)
            fd = d / "face"
            fd.mkdir(exist_ok=True)
            sw = slug(w)
            im.crop(c0).resize((tw, th), Image.LANCZOS).save(fd / f"{sw}_init.png")
            cs = [int(c0[0] / sx), int(c0[1] / sy), int(c0[2] / sx), int(c0[3] / sy)]
            for nm, src in (("depth", "start_depth.png"), ("edges", "start_edges.png")):
                Image.open(d / "render" / src).convert("RGB").crop(cs).resize((tw, th), Image.LANCZOS).save(fd / f"{sw}_{nm}.png")
            if f.get("lm"):
                face_pose_map(f["lm"], c0, (tw, th), fd / f"{sw}_facepose.png")
            lr = char_lora(cx, w)
            # ONE person per crop -> their headshot as the single image ref can't mix identities (10-20: Hulda kept a
            # man's face from the 3D sculpt with LoRA + words alone)
            # SPATIALLY AWARE (user 10-21: the pass "always paints a face looking towards the camera"): the staged head's
            # turn, in words; a FRONT headshot as image ref only for a near-frontal head (it pulled turned heads round:
            # a frontal face on a 3/4 head = the melted, doubled-jaw Edric keys); near-profiles are left as drawn
            yaw = lm_yaw(f.get("lm"))                 # the painted face's own turn (refined); else the staged head's
            yaw = yaw if yaw is not None else head_yaw(d / "render", w)
            if yaw is not None and abs(yaw) > FACE_MAX_YAW:
                continue
            sh_ = ref_sheet(cx.refs, w) if FACE_REF and (yaw is None or abs(yaw) <= FACE_REF_MAX_YAW) else None
            hf_ = cx.W / "video" / "charlora" / slug(w) / "refs" / "head_fixed.png"   # a repaired head (the Knight's
            hs_ = (str(hf_) if hf_.exists() else headshot(cx, w, str(sh_))) if sh_ else None  # sheet had stubble)
            prompt = (f"{'chr_' + slug(w) + ' ' if lr else ''}A close-up of the face of {w}, {angle_words(yaw)}: "
                      f"{face_words(cx, w)}. "
                      + ("This person's face is the face in image 1, at this picture's head angle. " if hs_ else "") +
                      f"{short_look(cx.look(w, e['t']).rstrip('.'))}. Keep the head exactly as turned in this picture, "
                      f"same expression, same lighting, the face clearly lit. "
                      f"{style_for(cx, e).rstrip('.')}. No text.")
            j = {"id": f"{e['id']}__face_{sw}", "prompt": no_quotes(prompt), "seed": 900 + sum(map(ord, w)) % 50,
                 "size": [tw, th], "refs": [hs_] if hs_ else [], "path": str(fd / f"{sw}.png"),
                 "control": [{"image": str(fd / f"{sw}_depth.png"), "type": "depth",   # the depth carries the head's turn
                              "scale": FACE_DEPTH if yaw is None or abs(yaw) < 20 else max(FACE_DEPTH, 1.0)},
                             {"image": str(fd / f"{sw}_edges.png"), "type": "canny", "scale": 0.3, "prepared": True}]
                            + ([{"image": str(fd / f"{sw}_facepose.png"), "type": "pose", "scale": FACE_POSE}]
                               if f.get("lm") else []),
                 "init": {"image": str(fd / f"{sw}_init.png"),
                          "strength": FACE_STRENGTH_F if gender_of(cx.chars.get(w, "")) == "woman" else FACE_STRENGTH}}
            if lr:
                j["loras"] = [[lr, 1.0]]
            Path(j["path"]).unlink(missing_ok=True)
            jobs.append(j)
            plan.append((e, d, w, c0, [x0, y0, x1, y1]))
    if not jobs:
        return {}
    cards = ["05", "65"] if card == "any" else [card]
    from concurrent.futures import ThreadPoolExecutor
    t = time.time()
    with ThreadPoolExecutor(len(cards)) as ex_:
        list(ex_.map(lambda k: _qwen(cx, jobs[k::len(cards)], cx.OUT / "stage", cards[k], f"face_{cards[k]}_{time.strftime('%H%M%S')}")
                     if jobs[k::len(cards)] else None, range(len(cards))))
    log_entry(cx.OUT, {"phase": "face-pass", "faces": len(jobs), "sec": round(time.time() - t)})
    out = {}
    for (e, d, w, c0, fb), j in zip(plan, jobs):
        if not Path(j["path"]).exists():
            continue
        pre = d / key.replace(".png", "_preface.png")
        if not pre.exists():
            shutil.copy(d / key, pre)
        im = Image.open(d / key).convert("RGB")
        cw, ch = c0[2] - c0[0], c0[3] - c0[1]
        patch = Image.open(j["path"]).convert("RGB").resize((cw, ch), Image.LANCZOS)
        m = Image.new("L", (cw, ch), 0)               # the face + hair oval, feathered
        fx0, fy0, fx1, fy1 = fb[0] - c0[0], fb[1] - c0[1], fb[2] - c0[0], fb[3] - c0[1]
        mx, my = 0.35 * (fx1 - fx0), 0.45 * (fy1 - fy0)
        ImageDraw.Draw(m).ellipse([fx0 - mx, fy0 - my * 1.2, fx1 + mx, fy1 + my * 0.6], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(max(3, cw // 20)))
        im.paste(patch, c0[:2], m)
        im.save(d / key)
        out.setdefault(e["id"], []).append({"who": w, "box": fb, "crop": c0})
    for i, v in out.items():
        (cx.OUT / "stage" / i / "face.json").write_text(json.dumps(v, indent=1))
    return out


RENDER_WORKERS = int(os.environ.get("REMAKE_RENDER_WORKERS", "1"))   # nice 19; the GPU is the bottleneck (10-20)


def ltx_wait(poll: float = 10.0) -> None:
    """Don't START a CPU-heavy Blender render while an LTX gen runs (10-20: renders starved the LTX drivers,
    4.9 -> 12 s/it)."""
    while True:
        r = subprocess.run(["pgrep", "-f", "ltx_split(_next)?[.]py gen"], capture_output=True, text=True)
        if not r.stdout.strip():
            return
        time.sleep(poll)


DRAW_MIN_BATCH = 8     # while renders are still coming: draw in batches (a 1-shot list cost 703 s, mostly load + lock)


def char_lora(cx: "Ctx", who: str) -> str | None:
    """The character LoRA to draw `who` with: lora/USE names the chosen file, else the highest step."""
    d = cx.W / "video" / "charlora" / slug(who) / "lora"
    if not (d.parent / "build.json").exists():   # a build stopped part-way leaves early checkpoints (10-22: Hulda)
        return None
    if (d / "USE").exists() and (d / (d / "USE").read_text().strip()).exists():
        return str(d / (d / "USE").read_text().strip())
    steps = sorted(d.glob("step_*.safetensors"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    return str(steps[-1]) if steps else None


def char_loras(cx: "Ctx", who: list) -> list[str]:
    """GLOBAL LoRAs for a ONE-person frame only: in a multi-person frame a global character LoRA dresses everyone
    as that person (charlora eval); several people need regional boxes (qwen_image [path, scale, box])."""
    return [x for w in who if not is_creature(cx.chars.get(w, "")) and (x := char_lora(cx, w))]


def lora_fallbacks(cx: "Ctx", P: dict) -> list[dict]:
    """Every moment whose key is drawn without someone's LoRA, listed loudly (user 10-20: no silent fallbacks).
    Creatures: their LoRA is trained but not yet applied to keys (the composite model carries the look)."""
    out = []
    for e in P["entries"]:
        names = list(dict.fromkeys([*(e.get("who") or []), *([e["speaker"]] if e.get("speaker") else [])]))
        miss = [w for w in names if not is_creature(cx.chars.get(w, "")) and not char_lora(cx, w)]
        if miss:
            out.append({"id": e["id"], "why": f"no LoRA for {', '.join(miss)}: key drawn without one (bookcaster assets)"})
    return out


def staged_loras(cx: "Ctx", who: list, expect: Path) -> list[str]:
    """keyframe_draw --lora specs for a staged frame: one person = global; several = each person's LoRA only inside
    their blockout figure box (<frame>_expect.json, padded 15 %), 'PATH=1.0@x0,y0,x1,y1'."""
    people = [w for w in who if not is_creature(cx.chars.get(w, ""))]
    if len(people) <= 1:
        return char_loras(cx, people)
    ex = json.loads(expect.read_text()) if expect.exists() else {}
    W, H = ex.get("size", [1, 1])
    boxes = {f["who"]: f["bbox"] for f in ex.get("figures", []) if f.get("bbox")}
    out = []
    for w in people:
        x = char_lora(cx, w)
        if not x or w not in boxes:
            continue
        x0, y0, x1, y1 = boxes[w]
        px, py = 0.15 * (x1 - x0), 0.15 * (y1 - y0)
        b = [max(0, (x0 - px) / W), max(0, (y0 - py) / H), min(1, (x1 + px) / W), min(1, (y1 + py) / H)]
        out.append(f"{x}=1.0@" + ",".join(f"{v:.3f}" for v in b))
    return out


def moment_who(cx: "Ctx", e: dict) -> list:
    """Who is in the moment's frame: the keyframe plan, else the staged shot's figures, else speaker + listener.
    A full-3D talk moment is a CLEAN single of the speaker (talk_shot): only the speaker (10-21 dry run: the prompt
    said "Exactly 3 people" over a one-person still, inviting painted-in extras)."""
    if e.get("route") == "talk" and full3d() and TALK_SINGLE and e.get("speaker"):
        return [e["speaker"]]
    f = cx.frames.get(e["id"]) or {}
    if f.get("who"):
        return list(f["who"])
    if e.get("shot") and Path(e["shot"]).exists():
        return [g["who"] for g in json.loads(Path(e["shot"]).read_text()).get("figures", []) if g.get("who")]
    return [w for w in (e.get("speaker"), e.get("listener")) if w]


TIME_WORDS = re.compile(r"\b(at (dawn|dusk|night|sunset|sunrise|midday|noon)|by (night|day|candlelight|firelight|lantern ?light)|"
                        r"(early|late) (morning|evening)|in the (morning|evening|night|afternoon)|at twilight|under (a |the )?"
                        r"(harvest )?moon)\b", re.I)


def shot_place(cx: "Ctx", e: dict) -> dict:
    """The staged shot's location for prompts and checks: where, interior/exterior, the book's place description,
    and any time-of-day words of the moment's picture/keyframe prompt."""
    st = {}
    if e.get("shot") and Path(e["shot"]).exists():
        st = json.loads(Path(e["shot"]).read_text()).get("set", {}) or {}
    f = cx.frames.get(e["id"]) or {}
    where = st.get("where") or f.get("where") or e.get("where") or ""
    kind = st.get("kind") or (cx.sets.get(where) or {}).get("kind") or ""
    pic = next((p for p in cx.plan.get("pictures", []) if p["id"] == e.get("pic")), {})
    when = (cx.sets.get(where) or {}).get("when")         # the set's time of day is the one source of truth
    m = None if when else TIME_WORDS.search(" ".join([f.get("prompt", ""), pic.get("prompt", "")]))
    places = cx.plan.get("look", {}).get("places", {}) if isinstance(cx.plan.get("look"), dict) else {}
    try:
        places = {**json.loads((cx.W / "video" / "look.json").read_text()).get("places", {}), **places}
    except (OSError, ValueError):
        pass
    t = ({"night": "at night", "day": "by day", "dusk": "at dusk", "dawn": "at dawn"}.get(when, f"at {when}")
         if when and when != "unknown" else (m.group(0) if m else ""))
    base = re.sub(r"\s*\((inside|outside)\)\s*$", "", where)
    return {"where": where, "kind": kind, "desc": places.get(where) or places.get(base, ""), "time": t}


def style_for(cx: "Ctx", e: dict) -> str:
    """The book's art style minus content words that contradict an INTERIOR set: the knight style's "moody stormy
    skies" made the painter put Hulda on a seashore under a sky (10-20 keys_review)."""
    if not full3d() or shot_place(cx, e)["kind"] != "interior":
        return cx.style
    parts = [p for p in re.split(r",\s*", cx.style) if not re.search(r"\b(sky|skies|cloud|clouds|landscape|horizon|sea)\b", p, re.I)]
    return ", ".join(parts)


def place_words(cx: "Ctx", e: dict) -> str:
    pl = shot_place(cx, e)
    if not pl["where"]:
        return ""
    t = f", {pl['time']}" if pl["time"] else ""
    keep = (" The room, its walls, its fire and its furniture are exactly those of the control picture; no sea, no open sky."
            if pl["kind"] == "interior" else " The landscape is exactly that of the control picture.")
    # the side said EXPLICITLY (user 10-21): "show the inside of" / "show the outside of"; the description without
    # the other side's clauses or the story props (the forge's "anvil outside" in an inside shot)
    import world as _W
    name = _W.VARIANT_RX.sub("", pl["where"])
    side = {"interior": f"Show the INSIDE of {name}{t}: an interior room, walls and ceiling around, no sky",
            "exterior": f"Show the OUTSIDE of {name}{t}: seen from outdoors"}.get(pl["kind"], f"The setting is {name}{t}")
    # the words the SET was drawn from (user 10-21: keyframe and built place must match), else the cleaned look
    desc = (cx.sets.get(pl["where"]) or {}).get("desc")
    if desc is None:
        try:
            look = json.loads((cx.W / "video" / "look.json").read_text())
            desc = _W.clean_description(pl["desc"], pl["kind"], _W.prop_names(look, cx.W / "video")) if pl["kind"] else pl["desc"]
        except (OSError, ValueError):
            desc = pl["desc"]
    return f" {side}" + (f"; {desc.rstrip('.')}" if desc else "") + "." + keep


def outfit_sheet(cx: "Ctx", e: dict, who: str) -> str | None:
    """The reference sheet of the outfit the person wears in this shot (their own model's index entry), else theirs."""
    try:
        figs = json.loads(Path(e["shot"]).read_text()).get("figures", []) if e.get("shot") else []
        m = next((g.get("model") for g in figs if g.get("who") == who), None)
        m = m if isinstance(m, str) else (m or {}).get("blend")
        if m:
            idx = json.loads((cx.W / "video" / "stage" / "characters" / "index.json").read_text())["characters"]
            for o in (idx.get(who) or {}).get("outfits", {}).values():
                if Path(o.get("dir", "")) == Path(m).parent and o.get("sheet") and Path(o["sheet"]).exists():
                    return o["sheet"]
    except (OSError, ValueError, KeyError):
        pass
    s_ = ref_sheet(cx.refs, who)
    return str(s_) if s_ else None


def headshot(cx: "Ctx", who: str, sheet: str) -> str | None:
    """The sheet's head-only crop (face_likeness --headshot-out), cached per sheet."""
    hs = cx.OUT / "refs" / f"{slug(who)}_{Path(sheet).stem}_head.png"
    if not hs.exists():
        hs.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["python3", str(HERE / "face_likeness.py"), sheet, "--ref", sheet, "--headshot-out", str(hs)],
                       capture_output=True, text=True)
    return str(hs) if hs.exists() else None


def creature_head(cx: "Ctx", who: str) -> str | None:
    """A creature's head-only reference (any creature with refs, 10-20): <dir>/head_ref.png, else the creature
    sheet's 3/4 head, else creature_head.png - the painter details a deliberately simple procedural body from it."""
    try:
        idx = json.loads((cx.W / "video" / "stage" / "characters" / "index.json").read_text())["characters"]
        c = idx.get(who) or next((v for k, v in idx.items() if k.lower() == who.lower()), {})
        d = Path(next(iter(c.get("outfits", {}).values()), {}).get("dir") or (cx.W / "video" / "stage" / "characters" / slug(who)))
    except (OSError, ValueError, KeyError):
        d = cx.W / "video" / "stage" / "characters" / slug(who)
    cs = d / "creature_sheet.json"
    heads = json.loads(cs.read_text()).get("heads", {}) if cs.exists() else {}
    for p in (d / "head_ref.png", heads.get("q34"), heads.get("side"), d / "creature_head.png"):
        if p and Path(p).exists():
            return str(p)
    return None


def key_refs(cx: "Ctx", e: dict, who: list) -> tuple[list, dict]:
    """The keyframe's image conditions, per visible person: their HEADSHOT then their OUTFIT sheet (the painter drew
    the Knight bearded and Hulda in a jacket from the soft colour still + text alone, 10-20). -> (refs, {who:
    (face image n, outfit image n)}) - 1-based indices for the prompt."""
    refs, idx = [], {}
    for w in who:
        sh = outfit_sheet(cx, e, w)
        if not sh:
            continue
        hd = headshot(cx, w, sh) if not is_creature(cx.chars.get(w, "")) else creature_head(cx, w)
        if hd and len(refs) < 9:
            refs.append(hd)
        if len(refs) < 10:
            refs.append(sh)
        idx[w] = (refs.index(hd) + 1 if hd in refs else None, refs.index(sh) + 1)
    return refs, idx


def draw_prompt(cx: Ctx, e: dict, idx: dict | None = None, scene_only: bool = False) -> str:
    """The keyframe prompt: the shot's picture prompt, each person tagged with their look (wardrobe in force), with
    their image conditions named ("the person on the left is Edric: the face of image 1, the outfit of image 2")."""
    f = cx.frames.get(e["id"]) or {}
    who = moment_who(cx, e)
    if e.get("route") == "talk" and full3d() and TALK_SINGLE and e.get("speaker"):
        pl = shot_place(cx, e)                    # the clean single the still shows, not the 2D picture's scene
        f = {"prompt": (f"Head-and-shoulders portrait of {e['speaker']} speaking, face clearly visible, the only person "
                        f"in the picture" + (f", in {pl['where']}" if pl.get("where") else "") + ".")}
    elif not f.get("prompt"):                     # a talking moment: the scene picture's prompt + who speaks
        pic = next((p for p in cx.plan.get("pictures", []) if p["id"] == e.get("pic")), {})
        f = {"prompt": (pic.get("prompt", "") + (f" {e['speaker']} is speaking" + (f" to {e['listener']}" if e.get("listener") else "")
                        + ", face clearly visible." if e.get("speaker") else ""))}
    sides = {}
    ex = cx.OUT / "stage" / e["id"] / "render" / "start_expect.json"
    if ex.exists():
        d_ = json.loads(ex.read_text())
        W_ = d_.get("size", [1, 1])[0]
        for f_ in d_.get("figures", []):
            if f_.get("bbox") and len(d_.get("figures", [])) > 1:
                c_ = (f_["bbox"][0] + f_["bbox"][2]) / 2 / W_
                sides[f_["who"]] = "on the left" if c_ < 0.4 else "on the right" if c_ > 0.6 else "in the middle"

    def tag(w):
        look = cx.look(w, e["t"]).rstrip(".")
        if full3d():          # a carried prop only as the HOLDER's staged prop (10-20: the Knight's lump on the King's arm)
            look = short_look(look) + matte_words(look)
        fi, oi = (idx or {}).get(w, (None, None))
        if is_creature(cx.chars.get(w, "")):     # the simple 3D shape keeps silhouette + pose; the refs give the surface
            imgs = ", ".join(x for x in ((f"the head of image {fi}" if fi else ""),
                                         (f"the body, scales, horns and colours of image {oi}" if oi else "")) if x)
            if imgs:
                imgs += "; detailed scales and texture over the whole body"
        else:
            imgs = ", ".join(x for x in ((f"the face of image {fi}" if fi else ""), (f"exactly the clothes of image {oi}" if oi else "")) if x)
        hw = held_words(cx, e, w)
        kw = key_words(cx, w)
        return f"{('the person ' + sides[w] + ' is ') if w in sides else ''}{trig(w)}{w}" + (f" ({imgs})" if imgs else "") + \
            f": {look}" + (f", {kw}" if kw else "") + (f", {hw}" if hw else "")

    def trig(w):     # the character LoRA's trigger before the person's first mention (charlora: chr_<slug>)
        return f"chr_{slug(w)} " if TRIGGERS and char_lora(cx, w) and not is_creature(cx.chars.get(w, "")) else ""
    if scene_only:       # the global pass: who stands where + gender/age only (the looks + refs go per person)
        def tag(w):          # noqa: F811
            hw, kw = held_words(cx, e, w), key_words(cx, w)
            return f"{('the person ' + sides[w] + ' is ') if w in sides else ''}{trig(w)}{person_words(cx, w)}" + \
                (f", {kw}" if kw else "") + (f", {hw}" if hw else "")
    tags = [tag(w) for w in who if cx.look(w, e["t"])]
    n = len([w for w in who if not is_creature(cx.chars.get(w, ""))])
    cnt = f" Exactly {n} {'person' if n == 1 else 'people'} in the picture, each appearing once." if n else ""
    shave = [w for w in who if re.search(r"clean-shaven|beardless|no beard", cx.chars.get(w, ""), re.I)]
    shave_t = "".join(f" {w} is clean-shaven with a smooth jaw: no beard, no moustache, no stubble, no shadow on the chin."
                      for w in shave)
    return f"{no_quotes(f.get('prompt', ''))}{cnt}{shave_t} " + ("; ".join(tags) + ". " if tags else "") + \
        (place_words(cx, e).strip() + " " if full3d() else "") + (light_words(cx, e) if full3d() and os.environ.get("REMAKE_LIGHT_WORDS") == "1" else "") + \
        f"{style_for(cx, e).rstrip('.')}. No text, no lettering, no signature."


# --------------------------------------------------------------------------------------------- LTX jobs

ANATOMY = "Each person has exactly two arms, two hands and two legs."
# the check set: "quick" (default, user 10-20: one VL call per key + the resident face batch, one redraw max) or "full"
FULL_CHECKS = os.environ.get("REMAKE_CHECKS", "quick") == "full"
STATIC_PHRASE = "Static camera, the framing does not change; no pan, tilt or zoom."


def rubric(name: str) -> str:
    """The calibrated VL question of a review check (tools/review_checks.py is the one source of the wording)."""
    import review_checks
    return review_checks.VL[name][0]


def head_yaw(rd: Path, who: str, frame: str = "start") -> float | None:
    """How far the staged head is turned from the camera, in degrees (0 = straight at it, 90 = profile, 180 = away);
    positive = the face points to the RIGHT of the picture. From the head/nose targets + camera in stills.json."""
    import math
    try:
        st = json.loads((rd / "stills.json").read_text())
        f = next(x for x in st["frames"] if x.get("name") == frame)
        m = next(x for x in f["models"] if x.get("who") == who)
        h, n, c = m["targets"]["head"], m["targets"]["nose"], f["camera"]["pos"]
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return None
    a = (n[0] - h[0], n[1] - h[1])                   # where the face points (horizontal)
    b = (c[0] - h[0], c[1] - h[1])                   # towards the camera
    ang = math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))
    # seen from the camera, a face turned clockwise (from above) from the camera direction points to screen RIGHT
    return -ang


def lm_yaw(lm) -> float | None:
    """The PAINTED face's turn from its 68 DWPose landmarks (jaw 0-16, nose tip 30): where the nose tip sits between
    the jaw ends; 0 = frontal, +-90 = profile, positive = the face points to the RIGHT of the picture."""
    import math
    try:
        jl, jr, nt = lm[0], lm[16], lm[30]
    except (IndexError, TypeError):
        return None
    if min(jl[2], jr[2], nt[2]) < 0.2 or abs(jr[0] - jl[0]) < 4:
        return None
    r = (nt[0] - jl[0]) / (jr[0] - jl[0])                 # 0.5 = centred
    return round(math.degrees(math.asin(max(-1.0, min(1.0, 2 * r - 1)))), 1)


FACE_LINES = [range(0, 17), range(17, 22), range(22, 27), range(27, 31), range(31, 36), list(range(36, 42)) + [36],
              list(range(42, 48)) + [42], list(range(48, 60)) + [48], list(range(60, 68)) + [60]]


def face_pose_map(lm, crop, size, out: Path) -> None:
    """The painted face's landmarks as a pose-control picture (white lines + dots on black, the OpenPose face look) in
    the face crop's pixels: the pass keeps eyes, nose, mouth and jaw where the main pass put them (the head's turn)."""
    from PIL import Image, ImageDraw
    sx, sy = size[0] / max(crop[2] - crop[0], 1), size[1] / max(crop[3] - crop[1], 1)
    pts = [((x - crop[0]) * sx, (y - crop[1]) * sy, c) for x, y, c in lm]
    im = Image.new("RGB", tuple(size), (0, 0, 0))
    d = ImageDraw.Draw(im)
    for seg in FACE_LINES:
        q = [pts[i][:2] for i in seg if pts[i][2] >= 0.3]
        if len(q) >= 2:
            d.line(q, fill=(255, 255, 255), width=2)
    for x, y, c in pts:
        if c >= 0.3:
            d.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255))
    im.save(out)


def angle_words(yaw: float | None) -> str:
    if yaw is None or abs(yaw) < 12:
        return "facing the viewer"
    side = "right" if yaw > 0 else "left"
    k = "in profile, " if abs(yaw) >= 60 else ""
    return f"{k}the head turned about {5 * round(abs(yaw) / 5):.0f} degrees towards the {side} of the picture"


def facing_away(rd: Path, who: str, frame: str = "start") -> bool:
    """The staged figure faces away from the camera (> 90 deg): head->nose vs head->camera from stills.json."""
    try:
        st = json.loads((rd / "stills.json").read_text())
        f = next(x for x in st["frames"] if x.get("name") == frame)
        m = next(x for x in f["models"] if x.get("who") == who)
        h, n, c = m["targets"]["head"], m["targets"]["nose"], f["camera"]["pos"]
    except (OSError, ValueError, KeyError, StopIteration, TypeError):
        return False
    a = [n[k] - h[k] for k in range(2)]
    b = [c[k] - h[k] for k in range(2)]
    return a[0] * b[0] + a[1] * b[1] < 0


def white_frac(img: Path, box=None) -> float:
    """Share of near-white pixels (> 0.95 on every channel): an unpainted proxy/prop or a blown-out head (10-20)."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(img).convert("RGB"), dtype=np.float32) / 255.0
    if box:
        a = a[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    return float((a.min(axis=2) > 0.95).mean()) if a.size else 0.0


def white_blob(img: Path, s: int = 4) -> float:
    """The largest connected near-white (> 0.95) region as a share of the frame: an unpainted proxy/prop or a
    blown-out head (10-20, 793 = 2.0 %); candles/windows in a good key stay < 1 % (340 = 0.7 %)."""
    import numpy as np
    from collections import deque
    from PIL import Image
    a = np.asarray(Image.open(img).convert("RGB"), dtype=np.float32) / 255.0
    m = a.min(axis=2) > 0.95
    H, W = m.shape
    m = m[:H // s * s, :W // s * s].reshape(H // s, s, W // s, s).mean((1, 3)) > 0.5
    seen = np.zeros_like(m)
    best = 0
    for y, x in zip(*np.nonzero(m)):
        if seen[y, x]:
            continue
        q, n = deque([(y, x)]), 0
        seen[y, x] = True
        while q:
            cy, cx = q.popleft()
            n += 1
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < m.shape[0] and 0 <= nx < m.shape[1] and m[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        best = max(best, n)
    return best / max(m.size, 1)


def short_name(w: str) -> str:
    return f"the person ({w})"


def nm_art(nm: str) -> str:
    return nm if nm.endswith("s") else ("an " if nm[:1] in "aeiou" else "a ") + nm


def key_words(cx: "Ctx", w: str) -> str:
    """Book data stage/key_words.json {name: words}: an asset finding fixed in the prompt (Hulda's dress read as a
    jacket at key scale, 10-20)."""
    p = cx.W / "video" / "stage" / "key_words.json"
    try:
        return (json.loads(p.read_text()) if p.exists() else {}).get(w, "")
    except ValueError:
        return ""


def held_props(e: dict) -> dict:
    """{who: [(hand, prop)]} from the staged shot (figures[].holds, else props_state "held")."""
    out = {}
    try:
        s_ = json.loads(Path(e["shot"]).read_text())
    except (OSError, ValueError, TypeError, KeyError):
        return out
    for f in s_.get("figures", []):
        for hand, pr in (f.get("holds") or {}).items():
            if pr:
                out.setdefault(f.get("who"), []).append((hand, pr))
    if not out:
        for pr, st in ((s_.get("props_state") or {}).get("props") or {}).items():
            if isinstance(st, dict) and st.get("loc") == "held" and st.get("who"):
                out.setdefault(st["who"], []).append((st.get("hand") or "", pr))
    return out


def held_words(cx: "Ctx", e: dict, w: str) -> str:
    """'holding the reins in his right hand' - a small held prop is lost at low strength unless it is named (10-20)."""
    hp = held_props(e).get(w) or []
    if not hp:
        return ""
    pron = "her" if gender_of(cx.chars.get(w, "")) == "woman" else "his"
    return " and ".join(f"holding the {re.sub(r'__.*', '', p).replace('_', ' ')}" + (f" in {pron} {h} hand" if h in ("left", "right") else "")
                        for h, p in hp)


def light_words(cx: "Ctx", e: dict) -> str:
    """At low img2img strength the painter keeps the still's dark shading (golden gate 10-20): ask it to lift the
    faces, with the set's own light source ("warm firelight on the faces")."""
    pw = (place_words(cx, e) or "").lower()
    src = ("firelight" if re.search(r"forge|fire|hearth|smithy", pw) else "candlelight" if re.search(r"hall|castle|chapel|candle", pw)
           else "lantern light" if re.search(r"night|dusk", pw) else "daylight")
    return f"Warm {src} on the faces, the faces clearly lit, lifted shadows. "


def no_quotes(t: str) -> str:
    """Quoted words (dialogue, inscriptions) out of a video prompt: LTX paints them as on-screen text."""
    t = re.sub(r"[\"“”][^\"“”]*[\"“”]", "", t or "")
    t = re.sub(r"\b(says|said|saying|shouts|whispers|asks)\s*[:,]?\s*(?=[.;]|$)", r"\1", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def camera_path(shot_path) -> list:
    """The shot's authored camera move (blockout shot JSON camera.path: [{frame, pos, look, lens}], first..last);
    [] = a static camera."""
    try:
        cam = json.loads(Path(shot_path).read_text()).get("camera", {}) if shot_path else {}
    except (OSError, ValueError):
        return []
    path = cam.get("path") or []
    return path if len(path) >= 2 else []


def camera_expect(path: list, width: int = ACT_W, sensor_w: float = 36.0) -> dict:
    """What the authored move should look like on screen, first -> last camera: the image scale (zoom/dolly: lens
    ratio x look-distance ratio) and the horizontal shift of the scene as a fraction of the width (pan/track)."""
    import numpy as np
    if not path:
        return {"scale": 1.0, "shift": 0.0, "static": True}
    a, b = path[0], path[-1]
    pa, la, pb, lb = (np.asarray(x, float) for x in (a["pos"], a["look"], b["pos"], b["look"]))
    da, db = np.linalg.norm(la - pa), np.linalg.norm(lb - pb)
    scale = (float(b.get("lens", 35)) / float(a.get("lens", 35))) * (da / max(db, 1e-6))
    va, vb = (la - pa) / max(da, 1e-6), (lb - pb) / max(db, 1e-6)
    yaw = float(np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0]))
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    hfov = 2 * np.arctan(sensor_w / 2 / float(a.get("lens", 35)))
    side = (pb - pa) - np.dot(pb - pa, va) * va                      # camera translation across the view
    track = float(np.linalg.norm(side[:2])) / max(2 * da * np.tan(hfov / 2), 1e-6)
    shift = abs(yaw) / hfov + track
    return {"scale": round(float(scale), 3), "shift": round(float(shift), 3), "static": bool(abs(scale - 1) < 0.03 and shift < 0.02),
            "yaw_deg": round(float(np.degrees(yaw)), 1)}


def camera_phrase(path: list) -> str:
    """The prompt's camera words: only what the authored path does (LTX must not invent a move)."""
    ex = camera_expect(path)
    if ex["static"]:
        return STATIC_PHRASE
    parts = []
    if ex["scale"] > 1.03:
        parts.append("the camera slowly pushes in")
    elif ex["scale"] < 0.97:
        parts.append("the camera slowly pulls back")
    if abs(ex.get("yaw_deg", 0)) > 2:
        parts.append(f"and pans {'left' if ex['yaw_deg'] > 0 else 'right'}" if parts else
                     f"the camera slowly pans {'left' if ex['yaw_deg'] > 0 else 'right'}")
    elif ex["shift"] > 0.02:
        parts.append("and tracks sideways" if parts else "the camera slowly tracks sideways")
    p = " ".join(parts)
    return p[:1].upper() + p[1:] + ", one smooth continuous move, no cuts."


FRAMING_LEAD = re.compile(r"^\s*(extreme close-up|medium close-up|close-up|close up|medium shot|wide shot|establishing shot|"
                          r"low[- ]angle( shot)?|high[- ]angle( shot)?|over-the-shoulder( shot)?)\s*(of|inside|at|in|on|from)?\s*",
                          re.I)


def i2v_prompt(cx: Ctx, e: dict, staged: bool = False) -> str:
    """staged: the clip follows a staged render whose camera the solver framed (it fits every figure), so the
    picture prompt's own framing words ("Close-up of ...") are dropped: with them LTX cut from the drawn wide
    keyframe to a close-up after frame 0 (10-20, action-000470)."""
    f = cx.frames.get(e["id"]) or {}
    who = f.get("who") or []
    prompt = f.get("prompt", "")
    if staged:
        prompt = FRAMING_LEAD.sub("", prompt, count=1)
        prompt = prompt[:1].upper() + prompt[1:]
    tags = []
    for w in who:
        d = cx.look(w, e["t"])
        if d:
            tags.append(f"{w} is {d.rstrip('.')}.")
    cam = camera_phrase(camera_path(e.get("shot"))) if staged and full3d() else STATIC_PHRASE
    # the planner's motion text sometimes carries camera words of its own: only the path's words may describe the camera
    motion = re.sub(r"(?i)\b(the )?camera\b[^.]*\.", "", f.get("motion", "")).strip()
    prompt, motion = no_quotes(prompt), no_quotes(motion)
    return " ".join(x for x in [cam, ANATOMY, prompt, " ".join(tags), motion, style_for(cx, e).rstrip(".") + "."] if x)


PROP_WORDS = r"strapped|holding|holds|carr(y|ies)|clutch|at (his|her) (side|hip|belt)|on (his|her) back|lump|shield|sword|" \
             r"blade|hammer|lantern|staff|spear|bag|satchel|scabbard|forearm"


def matte_words(desc: str) -> str:
    """Fine specular armour shimmers in LTX at talk scale (10-20): ask for matte metal."""
    return ", matte brushed metal with soft diffuse highlights" if re.search(r"armou?r|plate|mail|helm|breastplate", desc or "", re.I) else ""


def short_look(desc: str) -> str:
    """The LTX look: who + hair + the main garment only; clauses about carried/strapped props dropped (10-20: the
    Knight's 'lump of melted metal strapped to his forearm' drawn as a white plaque flickering beside him - the key
    has no visible shield, the prompt still described one)."""
    parts = [p_.strip() for p_ in re.split(r"[;]", desc) if p_.strip()]
    keep = [p_ for p_ in parts if not re.search(PROP_WORDS, p_, re.I)] or parts[:1]
    out = "; ".join(keep)
    out = ", ".join(c for c in out.split(", ") if not re.search(PROP_WORDS, c, re.I))
    return out


def talk_prompt(cx: Ctx, e: dict) -> str:
    """No spoken words in the prompt: LTX DRAWS quoted text (10-20 preview: the Knight turned into a stone tablet with
    garbled lettering). The audio drives the lips."""
    who = e["speaker"]
    desc = cx.look(who, e["t"]).rstrip(".")
    if full3d():
        desc = short_look(desc) + matte_words(desc)
    emo = f", sounding {e['emotion']}," if e.get("emotion") else ""
    if e["creature"]:
        return (f"Static camera, the framing does not change. {who}, {desc}, speaks{emo}: its great jaws open and close "
                f"in sync with its voice, teeth and tongue visible, the lower jaw dropping on each syllable. "
                f"No text anywhere in the picture. {style_for(cx, e)}")
    side = speaker_side(cx, e)
    where = f" on the {side}" if side else ""
    g = gender_of(desc)
    pron = "her" if g == "woman" else "his" if g == "man" else "their"
    return (f"Static camera, the framing does not change. {who}{where}, {desc}, speaks{emo}, lips moving in sync with "
            f"{pron} voice. Everyone else in the scene listens in silence with their mouth closed. "
            f"No text anywhere in the picture. {style_for(cx, e)}")


def speaker_side(cx: "Ctx", e: dict) -> str | None:
    """left/right/centre of the speaker in the staged frame (<frame>_expect.json), else None."""
    ex = cx.OUT / "stage" / e["id"] / "render" / "start_expect.json"
    if not ex.exists():
        return None
    d = json.loads(ex.read_text())
    W = d.get("size", [1, 1])[0]
    f = next((f for f in d.get("figures", []) if canon_name(f.get("who")) == canon_name(e.get("speaker")) and f.get("bbox")), None)
    if not f or len(d.get("figures", [])) < 2:
        return None
    cx_ = (f["bbox"][0] + f["bbox"][2]) / 2 / W
    return "left" if cx_ < 0.4 else "right" if cx_ > 0.6 else "centre"


def staging_on() -> bool:
    """REMAKE_NO_STAGE=1: every action moment goes I2V (the full chain is opt-in until the blockout fixes land)."""
    return not os.environ.get("REMAKE_NO_STAGE")


def stage_ok(cx: Ctx, e: dict) -> bool:
    if not staging_on():
        return False
    d = cx.OUT / "stage" / e["id"]
    return e["route"] == "stage" and (d / "key.png").exists() and (d / "render" / "depth.mp4").exists() \
        and not (d / "FAILED").exists()


def base_first(cx: "Ctx", e: dict) -> str:
    """The frame the checks/redraws start from: the staged (scene-layer) keyframe in full 3D, else the old first."""
    st = cx.OUT / "stage" / e["id"]
    if full3d() and (st / "key.png").exists() and not (st / "FAILED").exists():
        return str(st / "key.png")
    return e["first"]


def talk_shot(cx: "Ctx", e: dict, d: Path) -> str:
    """The A2Vid camera: a close-up single on the speaker from the staged scene (same marks, axis and set), so the
    face is >= 25 % of the frame (10-20: the OTS two-shot gave the Knight a 55 px face and no lip-sync). The OTS
    staging stays for listener reactions. -> path of <stage>/<id>/shot_talk.json"""
    s_ = json.loads(Path(e["shot"]).read_text())
    spk = e.get("speaker")
    if not spk or not any(f.get("who") == spk for f in s_.get("figures", [])):
        return e["shot"]
    cam = dict(s_.get("camera") or {})
    side = None
    if cam.get("ots") and len(cam["ots"]) >= 2:     # stay on the OTS camera's side of the line (180-degree rule)
        side = cam.get("line")
    # NEARLY FRONT-ON (user 10-21: "prioritize nearly front on face positions for talking shots"): lip-sync and the face
    # pass both want the face toward the camera; the staging's 3/4 single (~44 deg) made the face pass paint a frontal
    # face onto a turned head
    s_["camera"] = {"framing": "close-up", "target": spk, "lens": 60, "face_on": TALK_FACE_ON,
                    **({"line": side} if side else {})}
    # a CLEAN single (10-21): the other figures leave the frame. Kept, they sat in the foreground with their back to
    # the camera and the painter turned them into a face (Hulda as an old man, 727) or a dark blob (771/898)
    s_["figures"] = [f for f in s_.get("figures", []) if f.get("who") == spk]
    s_.setdefault("plan_notes", []).append("talk single (remake_chapter.talk_shot)") if isinstance(s_.get("plan_notes"), list) else None
    out = d / "shot_talk.json"
    d.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s_, indent=1))
    return str(out)


def start_image(cx: Ctx, e: dict) -> str:
    k = cx.OUT / "keys" / f"{e['id']}.png"           # a redraw (of the staged keyframe in full 3D) wins
    if k.exists():
        return str(k)
    st = cx.OUT / "stage" / e["id"]
    if full3d() and (st / "key.png").exists() and not (st / "FAILED").exists():
        return str(st / "key.png")                  # the staged (scene-layer) first frame
    return str(k) if k.exists() else e["first"]


def make_jobs_files(cx: Ctx, P: dict, last: set | None = None) -> dict:
    L = cx.OUT / "ltx"
    (L / "emb").mkdir(parents=True, exist_ok=True)
    (L / "out").mkdir(exist_ok=True)
    neg = subprocess.run([str(LTX_PY), "-c", "import sys; sys.path.insert(0, '/home/jrouth/src/LTX-2/packages/ltx-pipelines/src');"
                          "from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT as N; print(N)"],
                         capture_output=True, text=True).stdout.strip()
    talk_jobs, talk_enc, act_jobs, act_enc = [], [], [], []
    for e in P["entries"]:
        if strict() and e["route"] in ("talk", "stage") and not (cx.OUT / "stage" / e["id"] / "key.png").exists():
            # never a job from an old 2D picture (10-21: 17 keyless talk moments fell back to anim/<id>_first.png)
            log_entry(cx.OUT, {"phase": "jobs", "id": e["id"], "note": "no current key: no job (strict)"})
            continue
        if e["route"] == "talk":
            pad = L / "audio" / f"{e['id']}_pad.wav"
            pad.parent.mkdir(exist_ok=True)
            padded = max(s + n for s, n in e["chunks"])
            if not pad.exists():
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", e["wav"], "-ac", "2", "-ar", "48000", "-af",
                                f"apad=whole_dur={padded / FPS:.4f}", str(pad)], check=True)
            p_ = talk_prompt(cx, e)
            emb = f"t_{e['id']}" + (f"_{hashlib.md5(p_.encode()).hexdigest()[:6]}" if full3d() else "")   # a new prompt = a new embedding
            p = talk_prompt(cx, e)
            talk_enc += [{"name": emb, "prompt": p}, {"name": emb + "__neg", "prompt": neg + NEG_EXTRA}]
            name = f"r_{e['id']}"
            for k, (st, n) in enumerate(e["chunks"]):
                j = {"name": name if len(e["chunks"]) == 1 else f"{name}_c{k}", "emb": emb, "prompt": p,
                     "width": TALK_W, "height": TALK_H, "fps": FPS, "seed": 42, "frames": n, "audio": str(pad),
                     "audio_start": round(st / FPS, 4), "line": e["id"], "speaker": e["speaker"],
                     "speech_s": e["speech_s"], "chunk": [k, len(e["chunks"]), st, n], "creature": e["creature"],
                     "lock_end": TALK_LOCK if n >= 97 else TALK_LOCK_SHORT}
                if e["creature"]:
                    j["guider"] = {"modality_scale": CREATURE_MODALITY}
                if k == 0 or (full3d() and not TALK_CHAIN):   # every chunk from the key + the camera lock (no drift)
                    j["images"] = [{"path": start_image(cx, e), "frame": 0, "strength": 1.0}]
                else:
                    # a chained chunk came out nearly FROZEN (10-20, 793_c1: mouth motion 0.2 vs 1.1 in c0): its start was
                    # pinned at 1.0 AND the camera lock put the same frame at its end. Start softer, no end lock; the
                    # start frame alone keeps the camera.
                    j["chain_from"], j["chain_frame"] = talk_jobs[-1]["name"], st - e["chunks"][k - 1][0]
                    j["chain_strength"] = CHAIN_STRENGTH
                    j.pop("lock_end", None)
                talk_jobs.append(j)
        else:
            staged = stage_ok(cx, e)
            # a continuation starts on the previous clip's last frame: its picture prompt's framing words would cut
            # away from it (000570: wide -> "Close-up of Edric"), so they go too
            # and for plain I2V too when REMAKE_I2V_STRIP_FRAMING=1 (the default in tools/ltx_flow.py): the first frame
            # fixes the framing, so the words can only pull a zoom (003816: medium keyframe, "Close-up" -> zoom)
            strip = staged or bool(e.get("chain_from")) or bool(os.environ.get("REMAKE_I2V_STRIP_FRAMING"))
            p = i2v_prompt(cx, e, strip)
            emb = f"a_{e['id']}" + ("_st" if strip and p != i2v_prompt(cx, e) else "") + \
                (f"_{hashlib.md5(p.encode()).hexdigest()[:6]}" if full3d() else "")
            act_enc.append({"name": emb, "prompt": p})
            j = {"name": f"r_{e['id']}", "emb": emb, "prompt": p, "width": ACT_W, "height": ACT_H, "fps": FPS,
                 "seed": 7, "frames": e["frames"], "line": e["id"]}
            src_ok = bool(e.get("chain_from")) and (
                (cx.OUT / "ltx" / "out" / f"r_{e['chain_from']}.mp4").exists() or
                e["chain_from"] in {x["id"] for x in P["entries"]})
            if stage_ok(cx, e) and full3d() and e.get("chain_from") and (src_ok or not (cx.OUT / "stage" / e["id"] / "key.png").exists()):
                # (a chain whose source is held/not in this pass starts from its own key instead - 10-20 570 <- 520)
                d = cx.OUT / "stage" / e["id"]
                j["chain_from"], j["chain_frame"], j["overlap"] = f"r_{e['chain_from']}", e["chain_frame"], 1
                j["control"] = {"path": str(d / "render" / "depth.mp4"), "attn": MODE2_ATTN}
                j["route"] = "stage-continue"
            elif stage_ok(cx, e):
                d = cx.OUT / "stage" / e["id"]
                j["images"] = [{"path": str(d / "key.png"), "frame": 0, "strength": 1.0}]
                if camera_path(e.get("shot")) and (d / "key_end.png").exists():   # the authored move's last camera
                    j["images"].append({"path": str(d / "key_end.png"), "frame": e["frames"] - 1, "strength": END_GUIDE})
                j["control"] = {"path": str(d / "render" / "depth.mp4"), "attn": MODE2_ATTN}
                j["route"] = "stage"
            elif strict():         # no 2D fallback in a delivery: the moment stays missing (listed at assembly)
                log_entry(cx.OUT, {"phase": "jobs", "id": e["id"], "note": "no 3D route: no fallback (strict)"})
                continue
            elif e.get("chain_from"):
                j["chain_from"], j["chain_frame"], j["overlap"] = f"r_{e['chain_from']}", e["chain_frame"], 1
                j["route"] = "i2v-continue"
            else:
                j["images"] = [{"path": start_image(cx, e), "frame": 0, "strength": 1.0}]
                j["route"] = "i2v"
            act_jobs.append(j)
    # re-rolls asked by the QA (qa phase): {moment id: seed}
    rr = json.loads((cx.OUT / "reroll.json").read_text()) if (cx.OUT / "reroll.json").exists() else {}
    for j in talk_jobs + act_jobs:
        if j["line"] in rr:
            j["seed"] = int(rr[j["line"]]) + (j.get("chunk") or [0])[0]
    if full3d():                 # every fallback is listed and loud (no silent I2V)
        fb = [{"id": e["id"], "why": (Path(cx.OUT / "stage" / e["id"] / "FAILED").read_text()[:300]
                                      if (cx.OUT / "stage" / e["id"] / "FAILED").exists() else "; ".join(e.get("why") or []))}
              for e in P["entries"] if e["route"] in ("missing", "stage") and not stage_ok(cx, e)]
        fb += lora_fallbacks(cx, P)
        (cx.OUT / "FALLBACKS.json").write_text(json.dumps(fb, indent=1))
        if fb:
            print(f"!!! FALLBACK (not 3D-staged) for {len(fb)} moment(s): " + ", ".join(x["id"] for x in fb), flush=True)
    if last:   # lines whose first frame is still being redrawn render last (a stable sort keeps chunk order)
        talk_jobs.sort(key=lambda j: j["line"] in last)
    defer = set(filter(None, os.environ.get("REMAKE_DEFER", "").split(",")))   # moments still staging: a later hold
    if (cx.OUT / "defer.txt").exists():      # changeable while a job runs (one id per line)
        defer |= {x.strip() for x in (cx.OUT / "defer.txt").read_text().split() if x.strip()}
    if defer:
        talk_jobs = [j for j in talk_jobs if j["line"] not in defer]
    if defer:
        act_jobs = [j for j in act_jobs if j["line"] not in defer]
    # chained jobs after their source (gen runs the list in order)
    order = {j["name"]: i for i, j in enumerate(act_jobs)}
    act_jobs.sort(key=lambda j: (order.get(j.get("chain_from"), -1) >= 0, order[j["name"]]))
    files = {"talk_jobs": L / "talk_jobs.json", "talk_enc": L / "talk_enc.json", "act_jobs": L / "act_jobs.json",
             "act_enc": L / "act_enc.json"}
    for k, v in zip(files, (talk_jobs, talk_enc, act_jobs, act_enc)):
        files[k].write_text(json.dumps(v, indent=1))
    print({k: len(json.loads(v.read_text())) for k, v in files.items()})
    return files


# --------------------------------------------------------------------------------------------- LTX queue

def kernel_check(out: Path, since: str = "-10min") -> None:
    r = subprocess.run(["journalctl", "-k", "--since", since], capture_output=True, text=True).stdout
    # (10-25: an XTX's HDMI-audio "CORB reset timeout" = a dock link retrain, not the GPU)
    # boot-time noise (10-20 22:08: a job started 2 min after boot stopped on the speaker amp + dock NIC resets)
    benign = re.compile(r"Reset line busy|reset \S+ USB device|ACPI|Firmware Bug|report a bug|reporting any bugs|uses VM inv eng|factory-reset|CORB reset timeout", re.I)
    hits = [ln for ln in r.splitlines() if re.search(r"\b(bug|oops|lockup|reset|ring|fault)\b", ln, re.I)
            and not benign.search(ln)]
    if hits:
        (out / "KERNEL_HIT.txt").write_text("\n".join(hits))
        log_entry(out, {"phase": "kernel", "hits": hits[-5:]})
        raise SystemExit("KERNEL LOG HIT: stopping (see KERNEL_HIT.txt)")


def _route(args: list) -> str:
    """The runner route a gen call takes (the names pp_fit / pp_timing.jsonl use)."""
    return "a2vid" if "--a2vid" in args else "ti2v" if "--ti2v" in args else "ic" if "--ic-lora" in args else "distilled"


def _split_arg(cx: Ctx, args: list) -> list:
    """The per-route pipelining split: a flow's REMAKE_LTX_SPLIT_TALK/_ACT if set, else pp_fit on this chapter's own
    timing (else the book's newest chapter, else SPLIT_DEFAULT)."""
    if "--split" in args or "--split" in os.environ.get("REMAKE_LTX_PP_ARGS", ""):
        return []
    v = os.environ.get("REMAKE_LTX_SPLIT_TALK" if "--a2vid" in args else "REMAKE_LTX_SPLIT_ACT")
    if not v:
        import flow3d as F
        r = _route(args)
        v = F._pp_split(cx.OUT, r, SPLIT_DEFAULT[r])
    return ["--split", str(v)]


def _stagger(cx: Ctx, route: str) -> int:
    """Half the route's median clip wall time (s), from this chapter's pp_timing.jsonl, else the book's newest chapter."""
    own = cx.OUT / "ltx" / "out" / "pp_timing.jsonl"
    sib = sorted(cx.OUT.parent.parent.glob(f"*/{cx.OUT.name}/ltx/out/pp_timing.jsonl"),
                 key=lambda f: f.stat().st_mtime, reverse=True)
    for f in ([own] if own.exists() else []) + [f for f in sib if f != own]:
        walls = []
        for ln in f.read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("route") == route and r.get("pp") and r.get("lanes", 1) > 1 and r.get("wall_s"):
                walls.append(r["wall_s"])
        if len(walls) >= 3:
            return round(sorted(walls)[len(walls) // 2] / 2)
    return STAGGER_DEFAULT[route]


def _gen_args(cx: Ctx, args: list) -> list:
    """A gen call's args with the default fast set added (see PP above); encode calls pass through unchanged."""
    if not (args and args[0] == "gen"):
        return list(args)
    out = list(args) + [a for a in LTX_GEN_ARGS if a not in args]
    if not PP:
        return out
    pp = os.environ.get("REMAKE_LTX_PP_ARGS", LTX_PP_ARGS).split()
    if "--stagger" not in pp and "--stagger" not in out:
        pp += ["--stagger", str(_stagger(cx, _route(args)))]
    return out + ["--pp", *[a for a in pp if a not in out]] + _split_arg(cx, out)


LTX_MEM_FLOOR_GB = 15
LTX_NO_ROOM_RC = 75
# the floor checked once both card locks are held (10-27: checked before the blocking flock, so a run that waited for
# the cards loaded LTX on whatever memory the holder had left when it let go)
LTX_MEM_GATE = ('m=$(awk \'/MemAvailable/ {printf "%d", $2/1048576}\' /proc/meminfo); '
                f'if [ "$m" -lt {LTX_MEM_FLOOR_GB} ]; then echo "MemAvailable $m GiB < {LTX_MEM_FLOOR_GB}: not loading LTX"; '
                f'exit {LTX_NO_ROOM_RC}; fi; exec "$@"')


def ltx_step(cx: Ctx, name: str, args: list[str]) -> int:
    L = cx.OUT / "ltx"
    hip = ",".join(subprocess.run([str(HOME / "bin" / "hip-index"), c], capture_output=True, text=True).stdout.strip()
                   for c in ("05", "65"))
    env = dict(os.environ, HIP_VISIBLE_DEVICES=hip, LTX_YIELD="1")    # yield to waiting GPU jobs every ~2 clips
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    t = time.time()
    lf = L / f"log_{name}_{time.strftime('%H%M%S')}.txt"
    gate = [str(GAPRUN), "/tmp/gpu-05.lock /tmp/gpu-65.lock"] if GAPRUN.exists() else \
        ["flock", "/tmp/gpu-05.lock", "flock", "/tmp/gpu-65.lock"]      # no gap runner: plain blocking locks
    with open(lf, "w") as fh:
        runner = L / ("ltx_split_next.py" if PP and args and args[0] == "gen" else "ltx_split.py")
        if not runner.exists():
            shutil.copy(RUNNER_DIR / runner.name, runner)
        r = subprocess.run([*gate, "bash", "-c", LTX_MEM_GATE, "ltx-gate", "timeout", str(HOLD_TIMEOUT), str(LTX_PY),
                            str(runner), *_gen_args(cx, args), "--emb", str(L / "emb")],
                           stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(L))
    if r.returncode == LTX_NO_ROOM_RC:
        raise SystemExit(lf.read_text().strip().splitlines()[-1] if lf.exists() and lf.read_text().strip()
                         else f"MemAvailable < {LTX_MEM_FLOOR_GB} GiB: not loading LTX")
    log_entry(cx.OUT, {"phase": "ltx", "step": name, "rc": r.returncode, "sec": round(time.time() - t), "log": str(lf)})
    kernel_check(cx.OUT, f"-{HOLD_TIMEOUT // 60 + 6}min")
    return r.returncode


ENC_CHUNK = 40    # (16 before the host-RAM fix in the runner) ltx_split encode keeps every prompt's 49 hidden-state layers on cuda:0 until the end (~0.4 GiB each):
                 # 52 prompts in one call ran out of memory (10-19 22:37); the LTX agent's calls had <= 12


def encode_chunked(cx: Ctx, enc_f: Path, name: str, extra: list[str]) -> None:
    L = cx.OUT / "ltx"
    todo = [x for x in json.loads(enc_f.read_text()) if not (L / "emb" / f"{x['name']}.pt").exists()]
    for k in range(0, len(todo), ENC_CHUNK):
        part = L / f"{enc_f.stem}_part{k // ENC_CHUNK}.json"
        part.write_text(json.dumps(todo[k:k + ENC_CHUNK], indent=1))
        ltx_step(cx, name, ["encode", str(part), *extra])
    left = [x["name"] for x in json.loads(enc_f.read_text()) if not (L / "emb" / f"{x['name']}.pt").exists()]
    if left:
        raise SystemExit(f"{name}: {len(left)} embeddings still missing, e.g. {left[:3]}")


def missing(L: Path, jobs_f: Path, key="name") -> list[str]:
    return [j[key] for j in json.loads(jobs_f.read_text()) if not (L / "out" / f"{j[key]}.mp4").exists()]


def ltx(cx: Ctx, P: dict, which: str = "talk,action", last: set | None = None) -> None:
    L = cx.OUT / "ltx"
    L.mkdir(parents=True, exist_ok=True)
    import filecmp
    for f in ("ltx_split.py", "stitch_talk.py") + (("ltx_split_next.py",) if PP else ()):   # frozen per run start
        if not (L / f).exists() or not filecmp.cmp(RUNNER_DIR / f, L / f, shallow=False):   # (10-21: was copy-once:
            shutil.copy(RUNNER_DIR / f, L / f)                                              # runner fixes never reached a chapter)
    miss = [e["id"] for e in P["entries"] if e["route"] == "missing" or (full3d() and e.get("unready"))]
    if miss and full3d() and not os.environ.get("REMAKE_ALLOW_FALLBACK"):
        raise SystemExit(f"{len(miss)} moment(s) have no staged shot / own character models yet (no silent I2V): {miss}")
    files = make_jobs_files(cx, P, last)   # always fresh: picks up redrawn/staged keyframes (prompts are unchanged)
    out = str(L / "out")
    def run_loop(kind, jobs_key, args, budget):
        """gen until every clip exists; job files are rebuilt before every hold (picks up redraws that land
        meanwhile); two holds in a row without progress = a failed phase (SystemExit)."""
        stuck, left = 0, None
        while True:
            fl = make_jobs_files(cx, P, last)
            m = missing(L, fl[jobs_key])
            if not m:
                return fl
            # prompts may have changed since the first encode (redraw/staging fixes): encode what's new
            encode_chunked(cx, fl["talk_enc" if kind == "talk" else "act_enc"], f"enc_{kind}", ["--dev"] if kind == "talk" else [])
            if left is not None and len(m) >= left:
                stuck += 1           # (a hold may yield at once to another GPU job: give it time)
                if stuck >= 6:
                    raise SystemExit(f"{kind}: no progress in 6 holds ({len(m)} clips left); see the ltx logs")
                time.sleep(60)
            else:
                stuck = 0
            left = len(m)
            ltx_step(cx, f"gen_{kind}", ["gen", str(fl[jobs_key]), *args, "--out", out, "--budget", str(budget)])
            log_entry(cx.OUT, {"phase": "ltx", f"{kind}_left": len(missing(L, fl[jobs_key]))})

    if "talk" in which:
        encode_chunked(cx, files["talk_enc"], "enc_talk", ["--dev"])
        fl = run_loop("talk", "talk_jobs", ["--a2vid", "--fa-amd", "--blaslt"], HOLD_BUDGET["talk"])
        r = subprocess.run([str(LTX_PY), str(L / "stitch_talk.py"), str(fl["talk_jobs"]), out],
                           capture_output=True, text=True, cwd=str(L))
        (L / "stitch.log").write_text(r.stdout + r.stderr)
        log_entry(cx.OUT, {"phase": "stitch", "rc": r.returncode})
        if r.returncode:
            raise SystemExit("stitch_talk failed: see ltx/stitch.log")
    if "action" in which:
        encode_chunked(cx, files["act_enc"], "enc_act", [])
        run_loop("action", "act_jobs", ["--ic-lora", str(IC_LORA), "--fa-amd", "--blaslt"], HOLD_BUDGET["act"])


# --------------------------------------------------------------------------------------------- background hold

BG_UNIFY = 0.5           # background repaint strength over the lit set render (the people are re-pasted after it)


def _brighten(img):
    """The staged stills are very dark (the painter then invents light sources): levels so the 98th-percentile
    luma lands at ~215, gamma 0.85."""
    import numpy as np
    from PIL import Image
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    l = a.mean(-1)
    hi = max(float(np.percentile(l, 98)), 0.05)
    a = np.clip(a * (0.84 / hi), 0, 1) ** 0.85
    return Image.fromarray((a * 255).astype(np.uint8))


def figure_mask(cx: "Ctx", e: dict, size: tuple):
    """The staged people's SILHOUETTES (not their boxes: a box carried the painter's own sky into the set, 10-20):
    frame 0 of the render's figures-only depth pass (blockout_video --layers figures=figure); a render without it
    gets a mask-only re-render (CPU, ~20 s). -> L image (255 = person), or None."""
    from PIL import Image
    rd = cx.OUT / "stage" / e["id"] / "render"
    if (rd / "figures0.png").exists():          # written by every render since 10-21 (key-frames renders have no clip)
        return Image.open(rd / "figures0.png").convert("L").resize(size).point(lambda v: 255 if v > 8 else 0)
    vid = rd / "depth_figures.mp4"
    if not vid.exists():
        md = cx.OUT / "stage" / e["id"] / "mask_render"
        if not (md / "depth_figures.mp4").exists() and e.get("shot"):
            env = dict(os.environ, BLOCKOUT_ASSETS=str(cx.W / "video" / "stage" / "assets.json"), BLOCKOUT_ALLOW_VIOLATIONS="1",
                       BLOCKOUT_THREADS="8")
            subprocess.run([str(POSE_PY), "-I", str(HERE / "blockout_video.py"), e["shot"], str(md), "--frames", "9",
                            "--size", f"{size[0] // 64 * 64}x{size[1] // 64 * 64}", "--layers", "figures=figure+prop"],
                           capture_output=True, text=True, env=env, timeout=1800)
        vid = md / "depth_figures.mp4"
    if not vid.exists():
        return None
    png = vid.with_name("figures0.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(vid), "-frames:v", "1", str(png)], check=True)
    m = Image.open(png).convert("L").resize(size).point(lambda v: 255 if v > 8 else 0)
    return m


def bg_hold(cx: "Ctx", P: dict, card: str = "05") -> None:
    """Keep the painted keyframe IN THE 3D SET, with a clean painted background:
    1. composite the drawn PEOPLE (their staged silhouettes, dilated + feathered) over the brightened set render;
    2. repaint it at BG_UNIFY (0.5) with the set's depth + edges as strong control: the layout stays the modelled
       room, the HY-World render's smears are repainted;
    3. paste the drawn people back over the repainted background (they are protected from the pass).
    key.png -> key_draw.png (kept), key.png = the held version."""
    from PIL import Image, ImageFilter
    jobs, todo = [], []
    for e in P["entries"]:
        d = cx.OUT / "stage" / e["id"]
        rd = d / "render"
        if not (d / "key.png").exists() or (d / "key_draw.png").exists() or not (rd / "start_color.png").exists():
            continue
        key = Image.open(d / "key.png").convert("RGB")
        W, H = key.size
        m = figure_mask(cx, e, (W, H))
        if m is None:
            log_entry(cx.OUT, {"phase": "bg-hold", "id": e["id"], "skip": "no figure silhouette"})
            continue
        m = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(3))
        m.save(d / "key_mask.png")
        bg = _brighten(Image.open(rd / "start_color.png")).resize((W, H))
        Image.composite(key, bg, m).save(d / "key_comp.png")
        who = moment_who(cx, e)
        refs = [str(x) for x in (ref_sheet(cx.refs, w) for w in who) if x]
        loras = []
        for spec in staged_loras(cx, who, rd / "start_expect.json"):
            path, _, rest = spec.partition("=")
            sc, _, box = rest.partition("@")
            loras.append([path, float(sc or 1.0)] + ([[float(v) for v in box.split(",")]] if box else []))
        j = {"id": e["id"], "prompt": draw_prompt(cx, e), "seed": 77, "size": [W // 16 * 16, H // 16 * 16], "refs": refs[:10],
             "path": str(d / "key_bg.png"), "init": {"image": str(d / "key_comp.png"), "strength": BG_UNIFY},
             "control": [{"image": str(rd / "start_depth.png"), "type": "depth", "scale": 1.3}] +
                        ([{"image": str(rd / "start_edges.png"), "type": "canny", "scale": 0.6, "prepared": True}]
                         if (rd / "start_edges.png").exists() else [])}
        if loras:
            j["loras"] = loras
        jobs.append(j)
        todo.append(e)
    if not jobs:
        return
    d0 = cx.OUT / "stage"
    mem = int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) // 1048576
    if mem < 15:
        raise SystemExit(f"MemAvailable {mem} GiB < 15: not loading Qwen")
    # both cards (10-21: the whole batch queued on card 05 while 65 idled): halves, each under its card's lock
    cards = [card] if len(jobs) < 4 or card not in ("05", "65") else ["05", "65"]
    if len(cards) == 2 and mem < 30:
        cards = [card]                             # two drawers = two ~8 GB host footprints
    t = time.time()

    def _one(i, c):
        part = jobs[i::len(cards)]
        jf = d0 / f"bg_hold_{time.strftime('%H%M%S')}_{c}.json"
        jf.write_text(json.dumps(part, indent=1))
        hip = subprocess.run([str(HOME / "bin" / "hip-index"), c], capture_output=True, text=True).stdout.strip()
        env = dict(os.environ, HIP_VISIBLE_DEVICES=hip)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        r_ = subprocess.run(["flock", f"/tmp/gpu-{c}.lock", "timeout", "870", str(QWEN_PY), "-I",
                             str(HERE / "qwen_image.py"), str(jf), str(d0)], env=env, capture_output=True, text=True)
        (d0 / f"{jf.stem}.log").write_text(r_.stdout + r_.stderr)
        return r_.returncode
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(cards)) as ex_:
        rcs = list(ex_.map(lambda ic: _one(*ic), enumerate(cards)))
    kernel_check(cx.OUT)
    vl = []
    for e in todo:
        d = cx.OUT / "stage" / e["id"]
        if not (d / "key_bg.png").exists():
            continue
        key = Image.open(d / "key.png").convert("RGB")
        bgp = Image.open(d / "key_bg.png").convert("RGB").resize(key.size)
        m = Image.open(d / "key_mask.png")
        shutil.copy(d / "key.png", d / "key_draw.png")
        Image.composite(key, bgp, m).save(d / "key.png")          # the people as drawn, the room repainted
        pl = shot_place(cx, e)
        vl.append({"id": e["id"], "image": str(d / "key.png"),
                   "ask": {"clean": f"Does the background look like a clean painted {'interior of ' if pl['kind'] == 'interior' else ''}"
                                    f"{pl['where'] or 'scene'}, with no smears, noise or melted texture? Answer yes or no."}})
    ans = vl_batch(vl)
    for e in todo:
        a_ = ans.get(e["id"], {}).get("clean")
        if a_ is not None:
            (cx.OUT / "stage" / e["id"] / "bg_check.json").write_text(json.dumps({"clean": a_}))
    log_entry(cx.OUT, {"phase": "bg-hold", "n": len(jobs), "rc": max(rcs), "cards": cards, "sec": round(time.time() - t),
                       "unclean": [i for i, v in ans.items() if not v.get("clean", "yes").startswith("yes")]})


# --------------------------------------------------------------------------------------------- talk frame checks

FACE_MIN_H, FACE_MIN_LUMA = (0.25 if TALK_SINGLE else 0.15), 45.0   # 10-20: 382 lip-sync failed on a tiny face
AGE_TOL_YOUNG, AGE_TOL_VL = 3, 8      # Qwen3-VL age read vs look.json: an 18-year-old must read 15-21, a 70 62-78


GARMENT_WORDS = {"jerkin", "jacket", "tunic", "shirt", "dress", "gown", "apron", "armor", "armour", "plate", "breastplate",
                 "robe", "cloak", "coat", "vest", "doublet", "trousers", "skirt", "cape", "mail"}
MATERIAL_WORDS = {"leather", "wool", "woolen", "silk", "velvet", "linen", "metal", "steel", "iron", "fur", "cotton", "silver"}
COLOUR_WORDS = {"black", "brown", "blue", "red", "green", "grey", "gray", "white", "dark", "silver", "gold", "golden", "yellow",
                "purple", "orange", "pale", "tan"}
NEAR = {"armor": {"armour", "plate", "breastplate", "metal", "steel"}, "jerkin": {"vest", "jacket", "tunic", "doublet"},
        "gown": {"dress"}, "dress": {"gown", "robe"}, "gray": {"grey"}, "grey": {"gray"}, "armour": {"armor", "plate", "metal"},
        "silver": {"metal", "steel", "iron", "grey", "gray"}, "plate": {"armor", "armour", "metal"}, "tunic": {"shirt", "doublet"}}


def outfit_words(text: str) -> dict:
    w = set(re.findall(r"[a-z]+", re.sub(r"\b(no|without|not)\s+\w+", "", text.lower())))   # "no armor" is not a garment
    return {"garment": sorted(w & GARMENT_WORDS), "material": sorted(w & MATERIAL_WORDS), "colour": sorted(w & COLOUR_WORDS)}


def outfit_match(answer: str, want: dict) -> bool:
    """The VL description must not CONTRADICT the wardrobe: a named garment of another kind (a shirt for a jerkin),
    a named material of another kind (wool for leather), or only other colours. Silent/generic answers pass on what
    they do name (calibrated 10-20: a strict "must name the garment" failed the King's "velvet garment")."""
    got = set(re.findall(r"[a-z]+", re.sub(r"[*_#]", " ", (answer or "").lower())))
    near = lambda ws: set(ws) | {n for x in ws for n in NEAR.get(x, set())}  # noqa: E731
    g, m, c = got & GARMENT_WORDS, got & MATERIAL_WORDS, got & COLOUR_WORDS
    if want["garment"] and g and not (near(want["garment"]) & g):
        return False
    if want["material"] and m and not (near(want["material"]) & m):
        return False
    if want["colour"] and c and not (near(want["colour"]) & c):
        return False
    return True


def _hair_phrase(desc: str) -> str | None:
    m = re.search(r"[^,;]*\bhair\b[^,;]*", desc or "")
    return m.group(0).strip() if m else None


def talk_checks(cx: "Ctx", P: dict, fix: bool = True) -> dict:
    """Staged talk keyframes (full 3D): (1) the speaker's face is >= 15 % of the frame height and lit (mean luma
    >= 45) - else the shot needs reframing (listed, not redrawn); (2) every listener seen from behind keeps their
    hair as look.json says (Qwen3-VL yes/no on the listener's box) - else a redraw with the hair stated, best seed
    by the same check. -> {id: {...}} written to checks/<id>_talk.json"""
    from PIL import Image
    t_start = time.time()
    ents = [e for e in P["entries"] if e["route"] in ("talk", "stage") and (cx.OUT / "stage" / e["id"] / "key.png").exists()]
    res, faces_req, vl = {}, [], []
    for e in ents:
        d = cx.OUT / "stage" / e["id"]
        img = start_image(cx, e)
        ex = json.loads((d / "render" / "start_expect.json").read_text()) if (d / "render" / "start_expect.json").exists() else {}
        W, H = ex.get("size", Image.open(img).size)
        figs = {f["who"]: f["bbox"] for f in ex.get("figures", []) if f.get("bbox")}
        r = res.setdefault(e["id"], {"image": img, "flags": []})
        # the over-the-shoulder foreground person is seen from BEHIND: no face checks (age/beard read noise:
        # "Edric reads 40" from the back of his head), only hair, clothes and no face painted on the back
        cam = json.loads(Path(e["shot"]).read_text()).get("camera", {}) if e.get("shot") and Path(e["shot"]).exists() else {}
        behind = {cam["ots"][0]} if cam.get("ots") else set()
        r["behind"] = sorted(behind)
        sp = e.get("speaker") if e["route"] == "talk" else None
        sheet = ref_sheet(cx.refs, sp) if sp else None
        if sheet and not is_creature(cx.chars.get(sp, "")):
            side = None
            if sp in figs and len(figs) > 1:
                c = (figs[sp][0] + figs[sp][2]) / 2 / W
                side = "left" if c < 0.5 else "right"
            t = {"who": sp, "sheet": str(sheet), "age": age_of(cx.chars.get(sp, "")), "gender": gender_of(cx.chars.get(sp, "")),
                 "n_people": 1, **({"side": side} if side else {})}
            faces_req.append((img, t))
            r["_face_t"] = t
        for w, b in figs.items():           # outfit (wardrobe in force) and age, for everyone in frame
            desc = cx.look(w, e["t"])
            if is_creature(desc) or not desc:
                continue
            box = [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)]
            m_ = re.search(r"wearing ([^.;]*)", desc)
            if m_:
                outfit = ", ".join(m_.group(1).split(", ")[:3])[:120]
                # a yes/no question passed Edric's black shirt as "a rough leather jerkin" (10-20): ask what is worn,
                # and compare the words (garment, material, colour) with the look's
                vl.append({"id": f"{e['id']}|outfit|{w}", "image": img, "crop": box, "max_new_tokens": 24,
                           "ask": {"outfit": "Describe this person's clothes in a few words: garment, material and colour."}})
                r.setdefault("outfits", {})[w] = {"outfit": outfit, "words": outfit_words(outfit)}
            ag = age_of(desc) if w not in behind else None
            if ag:
                vl.append({"id": f"{e['id']}|age|{w}", "image": img, "crop": box,
                           "ask": {"age": "How old does this person look? Answer with one number."}, "max_new_tokens": 4})
                r.setdefault("ages", {})[w] = {"target": ag}
        for w, b in figs.items():           # facial hair as look.json says, for everyone facing the camera
            desc = cx.chars.get(w, "")
            if is_creature(desc) or w in behind:
                continue
            want = "no" if re.search(r"clean-shaven|beardless|no beard", desc, re.I) else \
                "yes" if re.search(r"\b(beard|moustache|mustache)\b", desc, re.I) else None
            if want:
                box = [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)]
                vl.append({"id": f"{e['id']}|beard|{w}", "image": img, "crop": box,
                           "ask": {"beard": rubric("key.facial_hair")}})
                r.setdefault("beards", {})[w] = {"want": want}
        for w, b in figs.items():
            if is_creature(cx.chars.get(w, "")):
                continue
            if w in behind:      # a back-turned listener: no face painted onto the back of the head
                box = [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)]
                vl.append({"id": f"{e['id']}|mask|{w}", "image": img, "crop": box,
                           "ask": {"mask": "Does the back or side of this person's head show a face-like, mottled or "
                                           "mask-like pattern? Answer yes or no."}})
                r.setdefault("masks", {})[w] = {}
            hair = _hair_phrase(cx.look(w, e["t"]))
            if hair:
                box = [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)]
                vl.append({"id": f"{e['id']}|{w}", "image": img, "crop": box,
                           "ask": {"hair": f"Look at this person's hair. Is it {hair}? Answer yes or no."}})
                r.setdefault("listeners", {})[w] = {"hair": hair}
        pl = shot_place(cx, e)
        if pl["kind"] in ("interior", "exterior"):
            vl.append({"id": f"{e['id']}|@place", "image": img,
                       "ask": {"indoors": "Is this scene indoors, inside a room or building? Answer yes or no.",
                               "sea": "Is the sea or an open sky visible in this picture? Answer yes or no."}})
            r["place"] = pl
    t_q = time.time()
    prefetch_faces(faces_req)
    t_f = time.time()
    ans = vl_batch(vl)
    log_entry(cx.OUT, {"phase": "talk-check-timing", "shots": len(ents), "faces_s": round(t_f - t_q), "vl_s": round(time.time() - t_f),
                       "questions": len(vl), "per_shot_s": round((time.time() - t_start) / max(1, len(ents)), 1)})
    for e in ents:
        r = res[e["id"]]
        t = r.pop("_face_t", None)
        if t:
            fr = _FACE_CACHE.get(_fkey(r["image"], t, t.get("side", "any"))) or {}
            fb = fr.get("face_box")
            if fb:
                im = Image.open(r["image"]).convert("L")
                hfrac = (fb[3] - fb[1]) / im.height
                crop = im.crop(tuple(int(v) for v in fb))
                luma = sum(crop.getdata()) / max(1, crop.width * crop.height)
                r["speaker_face"] = {"h_frac": round(hfrac, 3), "luma": round(luma, 1)}
                if hfrac < FACE_MIN_H:
                    r["flags"].append(f"speaker face {hfrac:.0%} of the frame height (< {FACE_MIN_H:.0%}): REFRAME")
                if luma < FACE_MIN_LUMA:
                    r["flags"].append(f"speaker face dark (luma {luma:.0f} < {FACE_MIN_LUMA:.0f}): relight/REFRAME")
            else:
                r["flags"].append("speaker face not found: REFRAME")
        pa = ans.get(f"{e['id']}|@place", {})
        if pa and r.get("place"):
            want_in = r["place"]["kind"] == "interior"
            if pa.get("indoors", "").startswith("yes") != want_in or (want_in and pa.get("sea", "").startswith("yes")):
                r["flags"].append(f"location: not {r['place']['kind']} {r['place']['where']} "
                                  f"(indoors={pa.get('indoors')}, sea/sky={pa.get('sea')})")
        for w, O in (r.get("outfits") or {}).items():
            a = ans.get(f"{e['id']}|outfit|{w}", {}).get("outfit")
            O["answer"] = a
            if a is not None and not outfit_match(a, O["words"]):
                r["flags"].append(f"outfit: {w} reads as '{a}', not {O['outfit']}")
        for w, A in (r.get("ages") or {}).items():
            a = ans.get(f"{e['id']}|age|{w}", {}).get("age")
            m_ = re.search(r"\d+", a or "")
            if m_:
                A["read"] = int(m_.group(0))
                tol = AGE_TOL_YOUNG if A["target"] < 25 else AGE_TOL_VL
                if abs(A["read"] - A["target"]) > tol:
                    r["flags"].append(f"age: {w} reads {A['read']} (target {A['target']:.0f} +-{tol})")
        for w, B in (r.get("beards") or {}).items():
            a = ans.get(f"{e['id']}|beard|{w}", {}).get("beard")
            B["answer"] = a
            if a is not None and not a.startswith(B["want"]):
                r["flags"].append(f"facial hair: {w} {'has a beard' if B['want'] == 'no' else 'lost the beard'}")
        for w, M in (r.get("masks") or {}).items():
            a = ans.get(f"{e['id']}|mask|{w}", {}).get("mask")
            M["answer"] = a
            if a is not None and a.startswith("yes"):
                r["flags"].append(f"listener {w}: face-like pattern on the back of the head")
        for w, L in (r.get("listeners") or {}).items():
            a = ans.get(f"{e['id']}|{w}", {}).get("hair")
            L["answer"] = a
            if a is not None and not a.startswith("yes"):
                r["flags"].append(f"listener {w} hair does not read as '{L['hair']}'")
        (cx.OUT / "checks").mkdir(exist_ok=True)
        (cx.OUT / "checks" / f"{e['id']}_talk.json").write_text(json.dumps(r, indent=1))
        log_entry(cx.OUT, {"phase": "talk-check", "id": e["id"], "flags": r["flags"], "face": r.get("speaker_face")})
    if fix:
        hair_fix(cx, P, {i: r for i, r in res.items() if any("hair does not read" in f or "back of the head" in f
                                                               or f.startswith(("facial hair", "outfit:", "age:"))
                                                               for f in r["flags"])})
    return res


def quick_key_checks(cx: "Ctx", P: dict, fix: bool = True) -> dict:
    """The default keyframe check (user 10-20: cheap, < 10 s a key): the resident face batch for the speaker (face
    present, likeness, age band) + ONE Qwen3-VL call per KEY: for each visible person (named by screen side) does the
    outfit match and the facial hair, and the scene: interior/exterior as the set, any text. ONE redraw max per
    failing key (the failing facts stated), the better of the two kept and flagged. -> {id: {flags}}"""
    from PIL import Image
    t0 = time.time()
    ents = [e for e in P["entries"] if e["route"] in ("talk", "stage") and (cx.OUT / "stage" / e["id"] / "key.png").exists()]
    res, faces, vl = {}, [], []
    for e in ents:
        d = cx.OUT / "stage" / e["id"]
        img = start_image(cx, e)
        ex = json.loads((d / "render" / "start_expect.json").read_text()) if (d / "render" / "start_expect.json").exists() else {}
        W = ex.get("size", Image.open(img).size)[0]
        figs = {f["who"]: f["bbox"] for f in ex.get("figures", []) if f.get("bbox")}
        cam = json.loads(Path(e["shot"]).read_text()).get("camera", {}) if e.get("shot") and Path(e["shot"]).exists() else {}
        behind = {cam["ots"][0]} if cam.get("ots") else set()
        r = res.setdefault(e["id"], {"image": img, "flags": [], "people": {}})
        qs = {}
        for w, b in figs.items():
            desc = cx.look(w, e["t"])
            if is_creature(desc) or not desc:
                continue
            c = (b[0] + b[2]) / 2 / W
            side = "on the left" if c < 0.4 else "on the right" if c > 0.6 else "in the middle"
            short = re.split(r"[;,]", cx.chars.get(w, ""))[0]
            mo = re.search(r"wearing ([^.;]*)", desc)
            if mo:
                qs[f"outfit|{w}"] = (f"the person {side} ({short}): their main garment, material and colour, "
                                     "max 4 words")
                r["people"].setdefault(w, {})["outfit"] = ", ".join(mo.group(1).split(", ")[:3])[:120]
            if w not in behind:
                want = "no" if re.search(r"clean-shaven|beardless|no beard", desc, re.I) else \
                    "yes" if re.search(r"\b(beard|moustache|mustache)\b", desc, re.I) else None
                if want:
                    qs[f"beard|{w}"] = f"the person {side}: a full beard or a moustache (ignore light stubble)? yes or no"
                    r["people"].setdefault(w, {})["beard"] = want
        pl = shot_place(cx, e)
        if pl["kind"] in ("interior", "exterior"):
            qs["place"] = "is this scene indoors? yes or no"
            qs["sky"] = "is the sea or an open sky visible? yes or no"
            r["place"] = pl
        qs["text"] = "is any written text, a sign, a tablet or a caption visible? yes or no"
        # a closed helm hides the face: no age to read (10-21: Edric in his star-iron helm "read 64" in 11 talk keys)
        qs["helmet"] = "is the main person's face hidden by a closed helmet or visor? yes or no"
        if getattr(cx, "period", ""):           # anachronisms (user 10-21): a modern hammer in a forge of 1000 AD
            qs["period"] = (f"could every object, tool, garment and building here have existed in {cx.period}? "
                            "Answer no if anything looks modern or machine-made. yes or no")
            # ... and WHICH one: a bare "no" failed 13 of knight ch1's painted keys (10-21) with nothing named
            qs["period_what"] = (f"Name the one object, tool, garment or building in this picture that could NOT have "
                                 f"existed in {cx.period}, in at most four words. If there is none, answer: none")
        rd0 = d / "render"
        for w, b in figs.items():          # gender + age as the look says, per visible face (10-20: 8 Hulda keys = a man)
            dsc = cx.chars.get(w, "")
            if is_creature(dsc) or not dsc or w in behind or (rd0.exists() and facing_away(rd0, w)):
                continue
            g_, a_ = gender_of(dsc), age_of(dsc)
            if g_ in ("woman", "man"):
                c_ = (b[0] + b[2]) / 2 / W
                sd_ = "on the left" if c_ < 0.4 else "on the right" if c_ > 0.6 else "in the middle"
                ag = ("an old " if a_ and a_ >= ELDER_AGE else "a young " if a_ and a_ < 25 else "a ") + g_
                qs[f"who|{w}|{ag}"] = f"the person {sd_}: is this {ag}? yes or no"
                r["people"].setdefault(w, {})
        for w, b in figs.items():          # a back-turned listener painted as a metal/stone blob (10-20: 1276, 1654)
            if w in behind or (rd0.exists() and facing_away(rd0, w)):
                c_ = (b[0] + b[2]) / 2 / W
                sd_ = "on the left" if c_ < 0.4 else "on the right" if c_ > 0.6 else "in the middle"
                qs[f"real|{w}"] = f"the figure {sd_}, seen from behind: a real person with hair and clothes (not a statue, metal or stone)? yes or no"
        for w, hp in held_props(e).items():
            if w in figs and not is_creature(cx.chars.get(w, "")):
                for h, pr in hp:
                    nm = re.sub(r"__.*", "", pr).replace("_", " ")
                    qs[f"prop|{w}|{nm}"] = f"is {short_name(w)} holding {nm_art(nm)}? yes or no"
        vl.append({"id": e["id"], "image": img, "ask": qs})
        sp = e.get("speaker") if e["route"] == "talk" else None
        sheet = ref_sheet(cx.refs, sp) if sp else None
        if sheet and not is_creature(cx.chars.get(sp, "")):
            side = None
            if sp in figs and len(figs) > 1:
                side = "left" if (figs[sp][0] + figs[sp][2]) / 2 / W < 0.5 else "right"
            t = {"who": sp, "sheet": str(sheet), "age": age_of(cx.chars.get(sp, "")), "gender": gender_of(cx.chars.get(sp, "")),
                 "n_people": 1, **({"side": side} if side else {})}
            faces.append((img, t))
            r["_t"] = t
    prefetch_faces(faces)
    t_f = time.time()
    ans = vl_batch(vl)          # one merged JSON call per key (no crops)
    if vl and not ans:          # the judge did not run (memory, crash): never a silent pass (10-21 dry run)
        raise SystemExit("key checks: the vision judge gave no answers (see the vl_batch line above)")
    no_face = []
    for e in ents:
        r, a = res[e["id"]], ans.get(e["id"], {})
        t = r.pop("_t", None)
        if t:
            fr = _FACE_CACHE.get(_fkey(r["image"], t, t.get("side", "any"))) or {}
            if not fr.get("face_box"):
                r["flags"].append("speaker face not found")
                no_face.append(e["id"])
            elif a.get("helmet", "").startswith("yes"):
                r.setdefault("notes", []).append(f"speaker {t['who']}: face behind a closed helm (age not judged)")
            elif not judge({**fr, "who": t["who"]}, cx):
                r["flags"].append(f"speaker {t['who']}: age/wrinkles off (reads {(fr.get('age') or {}).get('keyframe')})")
        for w, p_ in r["people"].items():
            if p_.get("outfit") and a.get(f"outfit|{w}") and not outfit_match(a[f"outfit|{w}"], outfit_words(p_["outfit"])):
                r["flags"].append(f"outfit: {w} reads as '{a[f'outfit|{w}']}', not {p_['outfit']}")
            if p_.get("beard") and a.get(f"beard|{w}") and not a[f"beard|{w}"].startswith(p_["beard"]):
                r["flags"].append(f"facial hair: {w} {'has a beard' if p_['beard'] == 'no' else 'lost the beard'}")
        if r.get("place") and a.get("place"):
            want_in = r["place"]["kind"] == "interior"
            if a["place"].startswith("yes") != want_in or (want_in and a.get("sky", "").startswith("yes")):
                r["flags"].append(f"location: not {r['place']['kind']} {r['place']['where']}")
        what_ = re.sub(r"[^a-z0-9 '-]", "", (a.get("period_what") or "").lower()).strip()
        vague_ = not what_ or re.search(r"\b(none|nothing|no object|style|painting|lighting|light|texture|render|"
                                        r"image|picture|colou?rs?|background|sheen|shiny|glossy|quality)\b", what_)
        if a.get("period", "").startswith("no") and not vague_:
            r["flags"].append(f"period: {what_} (not of {cx.period})")
        elif a.get("period", "").startswith("no"):
            r.setdefault("notes", []).append(f"period judge said no but named nothing concrete ({what_ or '-'})")
        kp = cx.OUT / "stage" / e["id"] / "key.png"
        fx_ = bool(re.search(r"spark|strike|hammer|fire|flame", ((cx.frames.get(e["id"]) or {}).get("motion") or "").lower()))
        wf = white_blob(kp)
        if wf > 0.01 and not fx_:
            r["flags"].append(f"white: a {wf * 100:.1f}% near-white blob (an unpainted object or a blown-out head)")
        rd_ = cx.OUT / "stage" / e["id"] / "render"
        away = [w for w in r["people"] if facing_away(rd_, w)] if rd_.exists() else []
        if away and (rd_ / "start_color.png").exists():
            # facing PRESERVED: a face the painter added (in the key, not in the still) inside a back-turned figure.
            # (the still's own 3/4 faces are not flags - 10-20 the key-only rule flagged 12 good keys)
            exd = json.loads((rd_ / "start_expect.json").read_text()) if (rd_ / "start_expect.json").exists() else {}
            bx = {f["who"]: f["bbox"] for f in exd.get("figures", []) if f.get("bbox")}
            st_ = rd_ / "start_color.png"
            FF = _faces([kp, st_])
            kW, sW = Image.open(kp).size[0], Image.open(st_).size[0]
            W0 = exd.get("size", [sW])[0]
            for w in away:
                if w not in bx:
                    continue

                def has(path, scale):
                    b = [v * scale for v in bx[w]]
                    return any(b[0] <= (f_["box"][0] + f_["box"][2]) / 2 <= b[2] and f_["box"][2] - f_["box"][0] >= 28 * scale * W0 / 832
                               for f_ in (FF.get(str(path)) or {}).get("faces", []))
                if has(kp, kW / W0) and not has(st_, sW / W0):
                    r["flags"].append(f"back-turned: {w} faces away in the still but shows a face in the key")
        for k_, v_ in a.items():
            if k_.startswith("who|") and str(v_).startswith("no"):
                _, w_, ag_ = k_.split("|", 2)
                r["flags"].append(f"speaker {w_}: does not read as {ag_}")
            if k_.startswith("real|") and str(v_).startswith("no"):
                w_ = k_.split("|", 1)[1]           # HARD only when the figure is not dark (silver/grey metal, luma > 0.22); a dark,
                hard_ = False                       # blurred foreground back is fine (10-20: 9 false flags on Edric)
                try:
                    exd = json.loads((cx.OUT / "stage" / e["id"] / "render" / "start_expect.json").read_text())
                    b = next(f["bbox"] for f in exd["figures"] if f["who"] == w_)
                    im_ = Image.open(kp).convert("L")
                    sc_ = im_.size[0] / exd["size"][0]
                    import numpy as _np
                    hard_ = float(_np.asarray(im_.crop([int(v * sc_) for v in b]), dtype=_np.float32).mean()) / 255 > 0.22
                except (OSError, ValueError, KeyError, StopIteration):
                    pass
                if hard_:
                    r["flags"].append(f"back-turned: {w_} reads as a statue/metal, not a person")
            if k_.startswith("prop|") and str(v_).startswith("no"):
                _, w_, nm_ = k_.split("|", 2)
                r["flags"].append(f"prop: {w_} not holding the {nm_}")
        if a.get("text", "").startswith("yes"):
            r["flags"].append("text in the picture")
        (cx.OUT / "checks").mkdir(exist_ok=True)
        (cx.OUT / "checks" / f"{e['id']}_quick.json").write_text(json.dumps({**r, "answers": a}, indent=1, default=str))
    if len(faces) >= 3 and len(no_face) == len(faces):   # every speaker "not found": the face batch failed, not
        raise SystemExit("key checks: no speaker face found in any talk key - the face batch likely failed")  # the keys
    log_entry(cx.OUT, {"phase": "key-check-quick", "keys": len(ents), "faces_s": round(t_f - t0), "vl_s": round(time.time() - t_f),
                       "per_key_s": round((time.time() - t0) / max(1, len(ents)), 1),
                       "flagged": {i: r["flags"] for i, r in res.items() if r["flags"]}})
    return res


def hair_fix(cx: "Ctx", P: dict, bad: dict, seeds: int = 4) -> None:
    """Redraw the staged talk keyframes whose back-turned listener lost their hair colour/length: keyframe_draw
    again (new seeds) with the hair said outright, then keep the first seed whose listener passes the VL check."""
    if not bad:
        return
    by = {e["id"]: e for e in P["entries"]}
    shots = []
    tag = time.strftime("%H%M%S")
    for i, r in bad.items():
        e, d = by[i], cx.OUT / "stage" / i
        extra = " ".join(f"{w} has {L['hair']}, clearly visible." for w, L in r.get("listeners", {}).items()
                         if L.get("answer") and not L["answer"].startswith("yes"))
        extra += "".join(f" {w} is seen from behind: the back of the head shows only hair, no face." for w, M in
                         (r.get("masks") or {}).items() if (M.get("answer") or "").startswith("yes"))
        extra += "".join(f" {w} is clean-shaven: no beard, no moustache, no stubble." if B["want"] == "no" else f" {w} has a beard."
                         for w, B in (r.get("beards") or {}).items() if B.get("answer") and not B["answer"].startswith(B["want"]))
        extra += "".join(f" {w} WEARS {O['outfit'].upper()}." for w, O in (r.get("outfits") or {}).items()
                         if O.get("answer") and not outfit_match(O["answer"], O.get("words") or outfit_words(O["outfit"])))
        extra += "".join(f" {w} is {'a youth' if A['target'] < 25 else 'a person'} of {A['target']:.0f}, looking exactly {A['target']:.0f}."
                         for w, A in (r.get("ages") or {}).items() if A.get("read") and abs(A["read"] - A["target"]) >
                         (AGE_TOL_YOUNG if A["target"] < 25 else AGE_TOL_VL))
        who = moment_who(cx, e)
        refs = [str(x) for x in (ref_sheet(cx.refs, w) for w in who) if x]
        shots.append({"stills": str(d / "render"), "frame": "start", "prompt": draw_prompt(cx, e) + " " + extra,
                      "refs": refs[:10], "out": str(d / f"key_fix{tag}.png"), "seeds": seeds, "seed0": 31 + 50 * len(list(d.glob("key_fix*_seeds"))),
                      "likeness": [f"{w}={x}" for w in who if (x := ref_sheet(cx.refs, w)) and not is_creature(cx.chars.get(w, ""))],
                      "lora": staged_loras(cx, who, d / "render" / "start_expect.json")})
    lf = cx.OUT / "stage" / f"hair_fix_{time.strftime('%H%M%S')}.json"
    lf.write_text(json.dumps(shots, indent=1))
    hip = subprocess.run([str(HOME / "bin" / "hip-index"), "05"], capture_output=True, text=True).stdout.strip()
    r = subprocess.run([str(POSE_PY), "-I", str(HERE / "keyframe_draw.py"), "--list", str(lf), "--lock", "/tmp/gpu-05.lock"],
                       capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
    (cx.OUT / "stage" / f"{lf.stem}.log").write_text(r.stdout + r.stderr)
    kernel_check(cx.OUT)
    # judge every drawn seed's listener hair; keep the first that passes
    vl = []
    for i, rr in bad.items():
        d = cx.OUT / "stage" / i
        ex = json.loads((d / "render" / "start_expect.json").read_text())
        W, H = ex["size"]
        figs = {f["who"]: f["bbox"] for f in ex.get("figures", []) if f.get("bbox")}
        for png in sorted((d).glob(f"key_fix{tag}_seeds/s*.png")):
            for w, L in rr.get("listeners", {}).items():
                b = figs[w]
                vl.append({"id": f"{i}|{w}|{png.name}", "image": str(png),
                           "crop": [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)],
                           "ask": {"hair": f"Look at this person's hair. Is it {L['hair']}? Answer yes or no."}})
            for w, O in (rr.get("outfits") or {}).items():
                if w in figs:
                    b = figs[w]
                    vl.append({"id": f"{i}|{w}#outfit|{png.name}", "image": str(png), "max_new_tokens": 24,
                               "crop": [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)],
                               "ask": {"outfit": "Describe this person's clothes in a few words: garment, material and colour."}})
            for w in (rr.get("masks") or {}):
                if w in figs:
                    b = figs[w]
                    vl.append({"id": f"{i}|{w}#mask|{png.name}", "image": str(png),
                               "crop": [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)],
                               "ask": {"mask": "Does the back or side of this person's head show a face-like, mottled or "
                                               "mask-like pattern? Answer yes or no."}})
            for w, A in (rr.get("ages") or {}).items():
                if w in figs:
                    b = figs[w]
                    vl.append({"id": f"{i}|{w}#age|{png.name}", "image": str(png), "max_new_tokens": 4,
                               "crop": [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)],
                               "ask": {"age": "How old does this person look? Answer with one number."}})
            for w, B in (rr.get("beards") or {}).items():
                b = figs[w]
                vl.append({"id": f"{i}|{w}#beard|{png.name}", "image": str(png),
                           "crop": [max(0, b[0] / W), max(0, b[1] / H), min(1, b[2] / W), min(1, b[3] / H)],
                           "ask": {"beard": rubric("key.facial_hair")}})
    ans = vl_batch(vl)
    for i, rr in bad.items():
        d = cx.OUT / "stage" / i
        ok = {}
        for k, v in ans.items():
            ii, w, png = k.split("|")
            if ii != i:
                continue
            if w.endswith("#beard"):
                want = (rr.get("beards") or {}).get(w[:-6], {}).get("want", "no")
                ok.setdefault(png, []).append(v.get("beard", "").startswith(want))
            elif w.endswith("#outfit"):
                O = (rr.get("outfits") or {}).get(w[:-7], {})
                ok.setdefault(png, []).append(outfit_match(v.get("outfit", ""), O.get("words") or outfit_words(O.get("outfit", ""))))
            elif w.endswith("#mask"):
                ok.setdefault(png, []).append(not v.get("mask", "").startswith("yes"))
            elif w.endswith("#age"):
                tgt = (rr.get("ages") or {}).get(w[:-4], {}).get("target", 0)
                m_ = re.search(r"\d+", v.get("age", ""))
                ok.setdefault(png, []).append(bool(m_) and abs(int(m_.group(0)) - tgt) <=
                                              (AGE_TOL_YOUNG if tgt < 25 else AGE_TOL_VL))
            else:
                ok.setdefault(png, []).append(v.get("hair", "").startswith("yes"))
        good = [png for png, v in sorted(ok.items()) if all(v)]
        if good:
            if not (d / "key_v1.png").exists():
                shutil.copy(d / "key.png", d / "key_v1.png")
            shutil.copy(d / f"key_fix{tag}_seeds" / good[0], d / "key.png")
        log_entry(cx.OUT, {"phase": "hair-fix", "id": i, "picked": good[0] if good else None,
                           "note": None if good else "no seed kept the listener's hair: needs a staging/LoRA fix"})


# --------------------------------------------------------------------------------------------- video QA

def prop_questions(cx: "Ctx", e: dict, k: int) -> dict:
    """VL questions from the moment's scene-layer shot (10-21: the forge rendered with no work on the anvil and the
    hammer in the air): frame 1 (45 %) = every staged hand-held prop is in its holder's hand; frame 3 (95 %, the
    contact/fx frame 'end') = the tool head touches the work + sparks when staged. Creatures are not asked."""
    if k not in (1, 3) or not e.get("shot") or not Path(e["shot"]).exists():
        return {}
    try:
        sh = json.loads(Path(e["shot"]).read_text())
    except ValueError:
        return {}
    nice = lambda x: re.sub(r"\s*\(.*?\)", "", str(x)).replace("__", " ").replace("_", " ").strip()  # noqa: E731
    ask = {}
    if k == 1:
        for g in sh.get("figures", []):
            w = g.get("who") or ""
            if is_creature(cx.chars.get(w, "")):
                continue
            short = re.split(r"[;,]", cx.chars.get(w, "") or w)[0] or w
            for hand, prop in (g.get("holds") or {}).items():
                ask[f"held:{w}:{nice(prop)}"] = f"Is {short} holding a {nice(prop)} in one hand? Answer yes or no."
    else:
        for ct in sh.get("contacts") or []:
            if ct.get("frame", "end") == "end" and ct.get("point") == "strike_face":
                ask[f"contact:{nice(ct.get('prop'))}->{nice(ct.get('to'))}"] = (
                    f"Is the head of the {nice(ct.get('prop'))} touching the {nice(ct.get('to'))}? Answer yes or no.")
        if any(x.get("type") == "sparks" and x.get("frame", "end") == "end" for x in sh.get("fx") or []):
            ask["sparks"] = "Are there bright sparks flying in this picture? Answer yes or no."
    return ask


def qa(cx: "Ctx", P: dict, reroll: bool = False, workers: int = 3) -> dict:
    """tools/video_qa.py on every new clip: cuts/zooms, hands on grips + left/right order vs the staging (staged
    action), lip-sync present for the speaker (talk). reroll: failing clips are moved to qa/rejected/ and get a new
    seed (reroll.json) - once per moment; run `ltx` again, then `qa` again to list what still fails."""
    qd = cx.OUT / "qa"
    (qd / "rejected").mkdir(parents=True, exist_ok=True)
    L = cx.OUT / "ltx"
    rr = json.loads((cx.OUT / "reroll.json").read_text()) if (cx.OUT / "reroll.json").exists() else {}

    def one(e):
        c = new_clip(cx, e)
        if not c:
            return e["id"], {"pass": None, "flags": ["no clip"]}
        rd = cx.OUT / "stage" / e["id"] / "render"
        args = [str(c), "--json", str(qd / f"{e['id']}.json")]
        if e["route"] == "talk":
            pad = L / "audio" / f"{e['id']}_pad.wav"
            args += ["--speech", str(pad)] if pad.exists() else []
            if (rd / "start_expect.json").exists():
                ex = json.loads((rd / "start_expect.json").read_text())
                f = next((f for f in ex.get("figures", []) if canon_name(f.get("who")) == canon_name(e.get("speaker"))
                          and f.get("bbox")), None)
                args += ["--expect", str(rd / "start_expect.json")]
                if f:
                    args += ["--speaker-x", f"{(f['bbox'][0] + f['bbox'][2]) / 2:.1f}"]
        elif stage_ok(cx, e):
            args += ["--anchors", str(rd / "anchors.json"), "--expect", str(rd / "start_expect.json")]
        if e.get("creature"):
            args += ["--no-identity"]
        if e["route"] != "talk":
            args += ["--cam-expect", json.dumps(camera_expect(camera_path(e.get("shot")) if stage_ok(cx, e) else []))]
        r = subprocess.run([str(POSE_PY), "-I", str(HERE / "video_qa.py"), *args], capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="4"), timeout=1800)
        try:
            return e["id"], json.loads((qd / f"{e['id']}.json").read_text())
        except (OSError, ValueError):
            return e["id"], {"pass": None, "flags": [f"qa crashed: {(r.stderr or '')[-200:]}"]}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, P["entries"]))
    # chained talk chunks must keep moving their lips: each chunk's speaker mouth activity while voiced >= 50 % of
    # chunk 0's (793_c1 froze at ~30 %, 10-20)
    for e in P["entries"]:
        if e["route"] != "talk" or len(e.get("chunks") or []) < 2 or e["id"] not in res:
            continue
        rd = cx.OUT / "stage" / e["id"] / "render"
        sx = None
        if (rd / "start_expect.json").exists():
            ex_ = json.loads((rd / "start_expect.json").read_text())
            f_ = next((f for f in ex_.get("figures", []) if canon_name(f.get("who")) == canon_name(e.get("speaker")) and f.get("bbox")), None)
            sx = (f_["bbox"][0] + f_["bbox"][2]) / 2 if f_ else None
        acts = []
        for k, (st, n) in enumerate(e["chunks"]):
            cp = L / "out" / f"r_{e['id']}_c{k}.mp4"
            if not cp.exists():
                break
            wav = qd / f"{e['id']}_c{k}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{st / FPS:.4f}", "-t", f"{n / FPS:.4f}",
                            "-i", str(L / "audio" / f"{e['id']}_pad.wav"), str(wav)], check=True)
            jq = qd / f"{e['id']}_c{k}.json"
            subprocess.run([str(POSE_PY), "-I", str(HERE / "video_qa.py"), str(cp), "--speech", str(wav), "--no-identity",
                            "--json", str(jq), *(["--speaker-x", f"{sx:.1f}"] if sx else [])],
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"), timeout=1800)
            mq = (json.loads(jq.read_text()).get("mouth") or {}) if jq.exists() else {}
            acts.append(mq.get("act_voiced"))
        r = res[e["id"]]
        r["chunk_mouth"] = acts
        if acts and acts[0]:
            low = [k for k, a_ in enumerate(acts[1:], 1) if a_ is not None and a_ < 0.5 * acts[0]]
            if low:
                r.setdefault("flags", []).append(f"chained chunk(s) {low} nearly frozen: mouth {acts}")
                r["pass"] = False
                (qd / f"{e['id']}.json").write_text(json.dumps(r, indent=1))
    # on-screen text (LTX paints quoted words, 10-20): Qwen3-VL on 4 frames per clip, one load for all clips
    vl, fdir = [], qd / "frames"
    fdir.mkdir(exist_ok=True)
    for e in P["entries"]:
        c = new_clip(cx, e)
        if not c or res.get(e["id"], {}).get("pass") is None:
            continue
        n = int(float(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                      "stream=nb_read_frames", "-of", "csv=p=0", str(c)], capture_output=True,
                                     text=True).stdout.strip() or 0))
        for k, fi in enumerate([int(n * x) for x in (0.2, 0.45, 0.7, 0.95)]):
            png = fdir / f"{e['id']}_{k}.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(c), "-vf", f"select=eq(n\\,{max(0, fi - 1)})",
                            "-vframes", "1", str(png)], capture_output=True)
            if png.exists():
                ask = {"text": "Is there any written text, letters, a sign, a tablet or a caption visible in "
                               "this picture? Answer yes or no."}
                if k in (0, 2) and full3d() and FULL_CHECKS:   # (full set only: the keyframe checks cover them)
                    pl = shot_place(cx, e)
                    if pl["kind"] in ("interior", "exterior"):
                        ask["indoors"] = "Is this scene indoors, inside a room or building? Answer yes or no."
                        ask["sea"] = "Is the sea or an open sky visible in this picture? Answer yes or no."
                    for w in moment_who(cx, e):
                        mo = re.search(r"wearing ([^.;]*)", cx.look(w, e["t"]) or "")
                        if mo and not is_creature(cx.chars.get(w, "")):
                            short = re.split(r"[;,]", cx.chars.get(w, ""))[0]
                            ask[f"outfit:{w}"] = (f"Is {short} in this picture wearing {', '.join(mo.group(1).split(', ')[:3])[:100]}? "
                                                  "Answer yes or no.")
                ask.update(prop_questions(cx, e, k))
                vl.append({"id": f"{e['id']}|{k}", "image": str(png), "ask": ask})
    ans = vl_batch(vl)
    if vl and not ans:          # the judge did not run: every clip would pass its text/location/outfit/prop checks
        raise SystemExit("clip QA: the vision judge gave no answers (see the vl_batch line above)")
    for e in P["entries"]:
        mine = {k: v for k, v in ans.items() if k.split("|")[0] == e["id"]}
        if not mine or e["id"] not in res:
            continue
        r, fl = res[e["id"]], []
        hits = [k for k, v in mine.items() if v.get("text", "").startswith("yes")]
        if hits:
            fl.append(f"on-screen text in {len(hits)} of 4 sampled frames")
        pl = shot_place(cx, e) if full3d() else {}
        for k, v in mine.items():
            if "indoors" in v and pl.get("kind"):
                want_in = pl["kind"] == "interior"
                if v["indoors"].startswith("yes") != want_in or (want_in and v.get("sea", "").startswith("yes")):
                    fl.append(f"location ({k.split('|')[1]}): not {pl['kind']} {pl['where']} (indoors={v['indoors']}, sea/sky={v.get('sea')})")
            for q, a_ in v.items():
                if q.startswith("outfit:") and not a_.startswith("yes"):
                    fl.append(f"outfit ({k.split('|')[1]}): {q[7:]} not in the wardrobe's clothes")
                if q.startswith(("held:", "contact:", "sparks")) and not a_.startswith("yes"):
                    fl.append(f"prop ({k.split('|')[1]}): {q} not seen (staged)")
        import placeholder_check as PH     # CPU: raw untextured prop boxes (10-21 forge)
        ph = {p.name: r for p in sorted(fdir.glob(f"{e['id']}_[0-3].png")) if (r := PH.check(str(p)))["placeholder"]}
        if ph:
            fl.append(f"placeholder prop block in {len(ph)} of 4 frames: {next(iter(ph.values()))['box']}")
        if fl:
            r.setdefault("flags", []).extend(sorted(set(fl)))
            r["pass"] = False
            (qd / f"{e['id']}.json").write_text(json.dumps(r, indent=1))
    bad = {i: r["flags"] for i, r in res.items() if r.get("pass") is False}
    summ = {"checked": len(res), "pass": sum(1 for r in res.values() if r.get("pass")), "fail": bad,
            "unchecked": [i for i, r in res.items() if r.get("pass") is None]}
    (qd / "summary.json").write_text(json.dumps(summ, indent=1))
    log_entry(cx.OUT, {"phase": "qa", "checked": summ["checked"], "pass": summ["pass"], "fail": len(bad)})
    if reroll:
        moved = []
        # a re-rolled clip's chained followers start from its old last frame: re-roll them too (10-21 dry run: chains
        # 470->520->570->620 would jump at every seam)
        chain = {e["id"]: e.get("chain_from") for e in P["entries"] if e.get("chain_from")}
        bad = dict(bad)
        grow = True
        while grow:
            grow = False
            for i, src in chain.items():
                if src in bad and i not in bad:
                    bad[i] = [f"chained from re-rolled {src}"]
                    grow = True
        for i in bad:
            if i in rr:                          # one re-roll per moment; what still fails is listed, not looped
                continue
            for f in (L / "out").glob(f"r_{i}*"):
                if f.name == f"r_{i}.mp4" or f.name.startswith(f"r_{i}.") or f.name.startswith(f"r_{i}_c"):
                    shutil.move(str(f), qd / "rejected" / f.name)
            rr[i] = 1000 + len(rr)
            moved.append(i)
        (cx.OUT / "reroll.json").write_text(json.dumps(rr, indent=1))
        log_entry(cx.OUT, {"phase": "qa-reroll", "rerolled": moved})
    return summ


# --------------------------------------------------------------------------------------------- assemble / sheet

def clip_current(cx: "Ctx", mid: str, p: Path) -> bool:
    """Strict (10-21): a clip counts only when it was made from the moment's CURRENT key (newer than key.png); a
    clip without a key (an I2V fallback) or older than its key (a redraw since) does not. Survives restarts."""
    k = cx.OUT / "stage" / mid / "key.png"
    return p.exists() and k.exists() and p.stat().st_mtime >= k.stat().st_mtime


def new_clip(cx: Ctx, e: dict) -> Path | None:
    p = cx.OUT / "ltx" / "out" / f"r_{e['id']}.mp4"
    if strict():
        return p if clip_current(cx, e["id"], p) else None
    return p if p.exists() else None


def assemble(cx: Ctx, P: dict) -> Path:
    video = cx.video
    man = []
    by = {e["id"]: e for e in P["entries"]}
    kept_old = []
    for m in cx.manifest:
        e = by.get(m["id"])
        c = new_clip(cx, e) if e else None
        if c is None and m.get("extra"):
            log_entry(cx.OUT, {"phase": "assemble", "id": m["id"], "note": "added moment has no clip: left out"})
            if strict():
                kept_old.append(m["id"])
            continue
        if c is None:
            kept_old.append(m["id"])
            if strict():                 # no old clip, never a missing file (10-21 dry run: video.build crashed on a
                continue                 # never-made anim/<id>.mp4): the stills cover the span; the file is _INCOMPLETE
        man.append({**{k: v for k, v in m.items() if k != "extra"}, "out": str(c or m["out"])})
    (cx.OUT / "anim").mkdir(exist_ok=True)
    (cx.OUT / "anim" / "manifest.json").write_text(json.dumps(man, indent=1))
    stills = cx.A / "stills.json"            # (the 2D animator's per-moment pictures: never in a strict delivery)
    frames = json.loads(stills.read_text()) if stills.exists() and not strict() else None
    # REMAKE_AUDIO: another mix of the same timeline (e.g. the score + sound-effects copy of Phase G)
    audio = Path(os.environ["REMAKE_AUDIO"]) if os.environ.get("REMAKE_AUDIO") else cx.W / "final" / f"{cx.stem}.mp3"
    total = video.audio_seconds(audio)
    pics = video.timed(cx.plan, cx.text, cx.timeline, total)
    for d in ("img",):                                   # build() reads pictures from img_dir; link, never copy over
        if not (cx.OUT / d).exists():
            (cx.OUT / d).symlink_to(cx.V / d)
    stamp = time.strftime("%m%d-%H%M")
    sfx = os.environ.get('REMAKE_SUFFIX', 'ltx') + ("_INCOMPLETE" if strict() and kept_old else "")
    out = cx.book / "out" / f"{cx.book.name}_{cx.stem}_{stamp}_{sfx}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)      # a fresh book has no out/ yet (10-21 dry run: ffmpeg 254)
    (cx.OUT / "MISSING.json").write_text(json.dumps(kept_old if strict() else [], indent=1))
    log_entry(cx.OUT, {"phase": "assemble", "new": len(man) - len(kept_old), "kept_old": kept_old, "out": str(out)})
    video.build(pics, cx.OUT / "img", audio, video.restore_text(cx.timeline, cx.text), out, cx.OUT,
                subtitles=True, log=print, anims=man, frames=frames)
    return out


def sheet(cx: Ctx, P: dict) -> Path:
    from PIL import Image, ImageDraw
    tw, th = 240, 135
    rows = []
    for e in P["entries"]:
        c = new_clip(cx, e)
        t = min(e["dur"] / 2, 2.0)
        pair = []
        for src in (e["old"], str(c) if c else None):
            img = Image.new("RGB", (tw, th), (40, 0, 0))
            if src and Path(src).exists():
                png = cx.OUT / "sheet_tmp.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", src, "-frames:v", "1", str(png)])
                if png.exists():
                    img = Image.open(png).convert("RGB").resize((tw, th))
            pair.append(img)
        rows.append((e, pair))
    cols = 3
    cw, ch = 2 * tw + 10, th + 18
    n_r = math.ceil(len(rows) / cols)
    S = Image.new("RGB", (cols * cw, n_r * ch), (20, 20, 20))
    dr = ImageDraw.Draw(S)
    for i, (e, (a, b)) in enumerate(rows):
        x, y = (i % cols) * cw, (i // cols) * ch
        S.paste(a, (x, y + 16))
        S.paste(b, (x + tw, y + 16))
        dr.text((x + 2, y + 2), f"{e['id']} {e['route']}  old | new", fill=(230, 230, 230))
    out = cx.OUT / "compare_sheet.jpg"
    q = 80
    while True:
        S.save(out, quality=q)
        if out.stat().st_size < 3_000_000 or q <= 40:
            break
        q -= 10
    return out


# --------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("book")
    ap.add_argument("stem")
    ap.add_argument("phase", choices=["plan", "check", "rejudge", "redraw", "redraw-strict", "qa", "talk-checks", "key-checks", "bg-hold", "stage", "jobs", "ltx", "assemble", "sheet", "all"])
    ap.add_argument("--out")
    ap.add_argument("--only", default="")
    ap.add_argument("--which", default="talk,action", help="ltx: talk,action")
    ap.add_argument("--hold-minutes", type=float, default=0, help="ltx: hold length (default 14.5 min)")
    ap.add_argument("--reroll", action="store_true", help="qa: move failing clips aside and re-seed them (once)")
    ap.add_argument("--ids", default="", help="redraw-strict: entry ids")
    ap.add_argument("--strict-seeds", type=int, default=6)
    ap.add_argument("--last", default="", help="ltx: talk lines (ids) to render last (their redraw is pending)")
    ap.add_argument("--card", default="05", help="redraw card")
    ap.add_argument("--steps", default="motion,render,draw", help="stage: motion,render,draw")
    ap.add_argument("--workers", type=int, default=2, help="stage render: parallel Blender jobs")
    ap.add_argument("--seeds", type=int, default=3, help="stage draw: seeds per keyframe")
    a = ap.parse_args(argv)
    global HOLD_TIMEOUT
    if a.hold_minutes:      # one long hold, model resident (coordinator: only when no other agent needs the cards)
        HOLD_TIMEOUT = int(a.hold_minutes * 60)
        HOLD_BUDGET.update(talk=HOLD_TIMEOUT - 420.0, act=HOLD_TIMEOUT - 300.0)
    cx = Ctx(Path(a.book).expanduser().resolve(), a.stem, Path(a.out) if a.out else None)
    only = set(filter(None, a.only.split(","))) or None
    P = plan(cx, only) if a.phase in ("plan", "all") or not (cx.OUT / "plan.json").exists() \
        else json.loads((cx.OUT / "plan.json").read_text())
    if only and a.phase != "plan":
        P = {**P, "entries": [e for e in P["entries"] if e["id"] in only]}
    if a.phase == "plan":
        from collections import Counter
        for e in P["entries"]:
            extra = (f"speaker={e['speaker']}{' CREATURE' if e['creature'] else ''} chunks={[n for _, n in e['chunks']]}"
                     if e["route"] == "talk" else
                     f"{e.get('mode')} where={e.get('where')} world={e.get('world')} frames={e['frames']}"
                     + (f" chain<-{e['chain_from']}" if e.get("chain_from") else "")
                     + (f" | {'; '.join(e['why'])}" if e.get("why") else ""))
            print(f"{e['id']:15s} {e['route']:6s} dur={e['dur']:5.2f} {extra}")
        print("routes:", dict(Counter(e["route"] for e in P["entries"])))
        print("why not staged:", dict(Counter(w.split(" for ")[0].split(" (")[0] for e in P["entries"] for w in e.get("why", []))))
        print(json.dumps(estimate(P), indent=1))
        return 0
    phases = ["check", "redraw", "stage", "jobs", "ltx", "assemble", "sheet"] if a.phase == "all" else [a.phase]
    for ph in phases:
        if ph == "check":
            check(cx, P)
        elif ph == "rejudge":
            rejudge(cx, P)
        elif ph == "bg-hold":
            bg_hold(cx, P, a.card)
        elif ph == "key-checks":
            print(json.dumps({k: v["flags"] for k, v in quick_key_checks(cx, P).items()}, indent=1))
        elif ph == "talk-checks":
            print(json.dumps({k: v["flags"] for k, v in talk_checks(cx, P, fix=not os.environ.get("REMAKE_NO_FIX")).items()}, indent=1))
        elif ph == "qa":
            print(json.dumps(qa(cx, P, reroll=a.reroll), indent=1))
        elif ph == "redraw":
            redraw(cx, P, a.card)
        elif ph == "redraw-strict":
            redraw_strict(cx, P, set(filter(None, a.ids.split(","))), a.card, a.strict_seeds)
        elif ph == "stage":
            stage(cx, P, a.steps, a.workers, a.seeds, a.card)
        elif ph == "jobs":
            make_jobs_files(cx, P)
        elif ph == "ltx":
            ltx(cx, P, a.which, set(filter(None, a.last.split(","))))
        elif ph == "assemble":
            print(assemble(cx, P))
        elif ph == "sheet":
            print(sheet(cx, P))
    return 0


if __name__ == "__main__":
    sys.exit(main())
