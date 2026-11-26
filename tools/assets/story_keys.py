"""A book's shot plan to engine-staged keyframes, every place of it, in one command.

    python tools/assets/story_keys.py PLAN.json BOOK_DIR WORK_DIR KEYS_DIR [--prefix ID] [--only PLACE ...]
                                      [--from STEP] [--until STEP] [--redo STEP|PLACE:STEP ...] [--jobs N]
                                      [--metahuman]

PLAN.json is the shots flow's plan ({shot id: [shot, ...]}), BOOK_DIR the book's folder (its work/video holds the
look sheet, the cast's reference sheets and LoRAs, the places' concept pictures, and the narration's timeline).
For every place of the plan, in the order the story reaches it, the steps below run in order; each writes a marker
in WORK_DIR/<place>/ when it is done, so a second run takes up where the first stopped:

  shots     the place's shot ids and name                                   (this file)
  spec      the place spec, by the directing model                          (scenes/place_spec.py)
  concept   the place's concept picture read into facts                     (scenes/concept.py)
  pieces    the pieces the library lacks, and the view outside              (sets/place_build.py, places/generate.py)
  set       the set spec and its engine level                               (sets/place_build.py, sets/build_set.py)
  cast      each person's figure, their seated and lying stances in the set (img3d/cast_char.py)
  pass      the plan's staging gaps asked, the scene specs written          (scenes/staging_pass.py, from_plan.py)
  stage     every scene group staged in the engine                          (scenes/speakers.py)
  redraw    guides, the redraw, its checks, one re-draw, the pick           (redraw/*.py)
  keys      the accepted keys and their people facts into KEYS_DIR          (scenes/shot_people.py)
  drawn     (off unless --until drawn; 11-17: engine keys for everything) shots left keyless, drawn from the plan

--approve [PLACE ...] keeps a place as it stands and marks it approved: no step of it runs again until --unlock
PLACE names it; --drift tells what the rules as they now are would move in each approved place (the planner alone, no
card); --restore PLACE [--tag T] puts a kept state back; every place made again is kept first (place_keep.py).

--jobs N works N places at once (their GPU work queues on the line by device; the steps that write the shared
library run one place at a time). --redo takes a step for every place or PLACE_SLUG:STEP for one. After the pass
step the place's scenes are checked before any GPU work (scenes/preflight.py; told in the log, nothing stopped); a
step past three times its usual time is told in the log, and one scene's staging is stopped at 25 minutes; a place
left with shots without a key has only those shots' scenes staged and drawn again.

All GPU work goes through the line (run_asset.py --line-cmd); nothing here decides how anything looks. WORK_DIR's
places.json ({place name: {"slug", "set"}}) and cast.json ({person: figure id}) are written on the first run and
read after (a place or a figure made earlier is named there and is not made again).
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import place_keep as PK  # noqa: E402
sys.path.insert(0, str(HERE / "scenes"))
import creature as CRE  # noqa: E402
import wardrobe as WD  # noqa: E402

STEPS = ["shots", "spec", "concept", "pieces", "set", "cast", "pass", "stage", "redraw", "keys", "drawn"]
# a step's rule number, raised when what the step writes changes (done(): an unfinished place runs it again)
STEP_RULE = {"spec": 5}
DRAW_STEPS = ("redraw", "keys")          # the drawing phase: run for every place together, after the staging (run())
PY = sys.executable
RA = [PY, str(HERE / "run_asset.py")]
LAB = C.UE_HOME / "lab"
# the redraw's two card scripts and its pose model (the line's image station runs them)
IMAGE_SH = Path(os.environ.get("STORY_IMAGE_SH", LAB / "redraw4" / "figs" / "run_image.sh"))
COUNT_SH = Path(os.environ.get("STORY_COUNT_SH", LAB / "redraw5" / "count.sh"))
CHECK_SH = Path(os.environ.get("STORY_CHECK_SH", LAB / "redraw5" / "check.sh"))
POSE_MODEL = Path(os.environ.get("STORY_POSE_MODEL", LAB / "redraw5" / "models" / "rtmw_x_256x192.pt"))
CARD_SH = HERE / "orbit" / "card.sh"
IMAGE_PY = Path(os.environ.get("STORY_IMAGE_PY", Path.home() / "tts-ab" / ".venv-qwenimage" / "bin" / "python"))
# the cards the keys are drawn on, each with the line's station and job kind that hold it: the image card, and the
# engine's (a command there borrows it: the editor unloads, line.py). The drawing is its own phase after every
# place is staged, so the engine has no work then and both cards draw (run())
# the checks: "igpu" = on the iGPU beside the drawing, each picture as it lands (the cards never stop drawing to
# check); "cards" = on the drawing cards after each round of drawing
CHECK_ON = os.environ.get("STORY_CHECK_ON", "igpu")
IGPU_PY = Path(os.environ.get("STORY_IGPU_PY", Path.home() / "tts-ab" / ".venv-charlora-igpu" / "bin" / "python"))
FOLLOW_S = float(os.environ.get("STORY_FOLLOW_S", "120"))    # a picture not landed this long after the last: not drawn
DRAW_CARDS = [c for c in os.environ.get("STORY_DRAW_CARDS", "05,65").split(",") if c.strip()]
CARD_LINE = {os.environ.get("ASSETS_BLENDER_CARD", "05"): ("image", "refs"),
             os.environ.get("ASSETS_UE_CARD", "65"): ("engine", "refs")}
GUIDES_PY = Path(os.environ.get("STORY_GUIDES_PY", Path.home() / "tts-ab" / ".venv-hyworld" / "bin" / "python"))


def _venv_py() -> Path:
    """The pipeline's own interpreter: the checkout's .venv, or the main checkout's when this is a worktree."""
    if os.environ.get("STORY_VENV_PY"):
        return Path(os.environ["STORY_VENV_PY"])
    own = HERE.parents[1] / ".venv" / "bin" / "python"
    if own.exists():
        return own
    r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--git-common-dir"], capture_output=True, text=True)
    return Path(r.stdout.strip()).resolve().parent / ".venv" / "bin" / "python"


VENV_PY = _venv_py()
LLM_UP = ("import sys; sys.path.insert(0, '.'); from tts_reader import animate; animate.igpu('strata')")
REDRAW = {"seed": 7, "method": "image", "strengths": [0.8], "control": {"type": "depth", "scale": 0.8, "end": 1.0},
          "set_facts": True, "mask_people": "tiered", "strength_bg": 0.5, "body_words": True, "light_words": True,
          "facts": True, "regions": True, "seeds": [7], "max_piece_refs": 2,
          # each person's usual sheet as a reference beside their LoRA (the user's call, 11-19: the sheets hold the
          # characters; drawn from her LoRA alone a woman of seventy came out fifty). STORY_SHEETS=0: only where a
          # face is large in the frame or the outfit is one the LoRA never saw (redraw.sheet_ref), twice as fast
          "sheet_with_lora": os.environ.get("STORY_SHEETS", "1") != "0"}


STYLE_CONTENT = re.compile(r"\b(cathedral|church|chapel|temple|castle|spire|arch\w*|stone|iron|window|sky|skies|"
                           r"candle\w*|ruins?|tower)\b", re.I)


STYLE_SKY = re.compile(r"\b(sky|skies|storm\w*|cloud\w*|mist\w*|fog\w*|rain\w*|snow\w*|overcast|dusk|dawn|"
                       r"twilight|night|moon\w*|sun\w*)\b", re.I)


def key_style(style: str, open_air: bool = False) -> str:
    """A book's art style as a key is told it: its way of painting only (as shot_flow._key_style), its clauses that
    name things to draw left out. Told the style whole, "moody stormy skies", a key of the inside of a forge had a
    storm sky for its back wall (11-18)."""
    head, _, rest = (style or "").strip().partition(":")
    head = re.sub(r"(?i)\b(gothic|baroque|romanesque|victorian|byzantine|rococo|renaissance|medieval)\b\s*", "", head)
    # (in the open air the style's own sky and weather stay: without them a river bank was drawn under a blank
    # white noon in a book of storm skies, 11-18; a room has no sky to draw)
    keep = [re.sub(r"(?i)\bgothic\s*", "", c).strip().rstrip(".") for c in rest.split(",") if c.strip()
            and (not STYLE_CONTENT.search(c) or (open_air and STYLE_SKY.search(c)
                                                 and not STYLE_CONTENT.search(STYLE_SKY.sub("", c))))]
    return (head.strip(" ,.") + (": " + ", ".join(keep) if keep else "")).rstrip(".") + "." if head.strip() else ""


ROOM_BG = 0.3                    # how far a room's background may be redrawn from the engine's frame


def log(m):
    print(f"[story {time.strftime('%H:%M:%S')}] {m}", flush=True)


def slug_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def sh(argv, logf: Path | None = None, env=None, cwd=None, limit_s: float | None = None) -> int:
    """One command, its output to logf. limit_s: stopped (with its children) when it runs longer, -> 124."""
    e = dict(os.environ, ASSETS_LINE_CLIENT=str(HERE / "run_asset.py"), STORY_VENV_PY=str(VENV_PY), **(env or {}))
    f = open(logf, "a") if logf else None
    try:
        p = subprocess.Popen([str(a) for a in argv], stdout=f, stderr=subprocess.STDOUT if f else None, env=e, cwd=cwd,
                             start_new_session=bool(limit_s))
        try:
            return p.wait(timeout=limit_s)
        except subprocess.TimeoutExpired:
            import signal
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except OSError:
                pass
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                p.kill()
            return 124
    finally:
        if f:
            f.close()


# How long a step usually runs, as (seconds, seconds per shot): a step past OVER_TIMES its usual time is said so in
# the log, again each time that long passes (a place's staging ran three hours where thirteen minutes were usual,
# on the CPU with every card idle, and nothing said so, 11-19). The waits of a shared line are in these times, so a
# step is told of, not stopped; a scene's own staging, which waits on nothing, is stopped at GROUP_MAX_S.
STEP_USUAL_S = {"shots": (60, 0), "spec": (600, 20), "concept": (600, 0), "pieces": (1500, 0), "set": (300, 0),
                "cast": (900, 0), "pass": (600, 30), "stage": (240, 40), "redraw": (420, 45), "keys": (120, 5),
                "drawn": (600, 60)}
OVER_TIMES = 3.0
GROUP_MAX_S = 1500.0         # one scene group's staging (speakers.py: its rounds are seconds each)


def over_time(step: str, shots: int, ran_s: float) -> bool:
    """Whether a step has run past OVER_TIMES its usual time for this many shots."""
    a, b = STEP_USUAL_S.get(step, (600, 0))
    return ran_s > OVER_TIMES * (a + b * shots)



def staging_miss(st: dict) -> list:
    """What a shot's kept staging missed, from its row of a scene group's report (speakers.json): nothing when it
    passed; what its last-resort wide itself misses when that was taken (it is another picture than the tries
    before it); else the kept try's misses, else the last try's."""
    if not st or st.get("ok"):
        return []
    lb = st.get("least_bad") or {}
    if lb.get("fallback") and "before" in lb:
        return list(map(str, lb.get("misses") or []))
    return list(map(str, lb.get("misses") or (st.get("history") or [{}])[-1].get("why") or st.get("why") or []))

class Watch:
    """Tells the log when a running step is past its usual time (a thread; with: around the step)."""
    def __init__(self, what: str, step: str, shots: int, every_s: float = 30.0):
        import threading
        self.what, self.step, self.shots, self.every = what, step, shots, every_s
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.told = 0

    def _run(self):
        t0 = time.time()
        a, b = STEP_USUAL_S.get(self.step, (600, 0))
        usual = a + b * self.shots
        while not self.stop.wait(self.every):
            ran = time.time() - t0
            if ran > OVER_TIMES * usual * (self.told + 1):
                self.told += 1
                log(f"{self.what}: {self.step} OVER TIME: running {ran / 60:.0f} min, usual about {usual / 60:.0f} min")

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.stop.set()


def line(station: str, kind: str, jid: str, logf: Path, cmd: list, env=None) -> int:
    """One job on the line; a full line (75) is waited out."""
    while True:
        rc = sh(RA + ["--line-cmd", station, kind, "--id", jid, "--log", logf, "--"] + cmd, logf.with_suffix(".line.log"), env)
        if rc != 75:
            return rc
        time.sleep(30)


LLM_NEED_GB = 96                 # what the directing model needs free to start (tts_reader/animate.py STRATA_NEED_GB)


def split_even(rows: list, n: int) -> list:
    """rows in n runs of near equal length, each run's rows together and in order (one scene's shots draw one after
    another: they share their references)."""
    n = max(1, min(n, len(rows)))
    k, r = divmod(len(rows), n)
    out, i = [], 0
    for j in range(n):
        m = k + (1 if j < r else 0)
        out.append(rows[i:i + m])
        i += m
    return [x for x in out if x]


def on_cards(parts: list, cmd_of, tag: str, logd: Path) -> bool:
    """One job per part, each on its own card of DRAW_CARDS at the same time (through the line). cmd_of(part, card,
    n) -> the shell words run on that card. True when every one ended well."""
    import threading
    root = HERE.parents[1]
    rcs = {}

    def one(n, part):
        card = DRAW_CARDS[n % len(DRAW_CARDS)]
        station, kind = CARD_LINE.get(card, ("image", "refs"))
        rcs[n] = line(station, kind, f"{tag}_{card}", logd / f"{tag}_{card}.log",
                      ["bash", "-c", f"cd {root} && bash {CARD_SH} {card} {cmd_of(part, card, n)}"])
    ths = [threading.Thread(target=one, args=(n, p)) for n, p in enumerate(parts)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    return all(rc == 0 for rc in rcs.values()) and len(rcs) == len(parts)


def draw_jobs(files: list, tag: str, logd: Path) -> bool:
    """Every drawing job of these job files (JSON lines), shared between the drawing cards: one image server on
    each, fed its share."""
    rows = rows_of(files)
    if not rows:
        return True
    parts = split_even(rows, len(DRAW_CARDS))

    def cmd(part, card, n):
        pf = logd / f"{tag}_{card}.jsonl"
        pf.write_text("\n".join(part) + "\n")
        return f"{IMAGE_PY} -I tools/qwen_image.py --serve < {pf}"
    log(f"drawing: {len(rows)} picture(s) on {len(parts)} card(s)")
    return on_cards(parts, cmd, tag, logd)


class Drawers:
    """An image server kept loaded on each drawing card for a whole drawing phase (through the line), fed jobs as
    they come: the first drawings, later the ones drawn again, with no reload between. feed() shares rows between
    the cards; close() ends the servers when they have drawn what they were fed -> whether every one ended well."""

    def __init__(self, tag: str, logd: Path):
        import queue
        import threading
        root = HERE.parents[1]
        self.cards, self.q, self.dead, self.rc, self.ths = list(DRAW_CARDS), {}, {}, {}, []
        self.fed, self.closing = {c: [] for c in DRAW_CARDS}, False
        # (stands while every drawer is at work or waiting for its card: the check beside them gives no picture up
        # as not drawn while it does)
        self.alive = logd / f"{tag}.alive"
        self.alive.write_text(time.strftime("%F %T"))
        for card in self.cards:
            ff = logd / f"{tag}_{card}.fifo"
            ff.unlink(missing_ok=True)
            os.mkfifo(ff)
            # (held open for reading too: the server sees its input end only when this side closes it, and opening
            # does not wait for the server, which the line may start minutes later)
            fd = os.open(ff, os.O_RDWR | os.O_NONBLOCK)
            self.q[card], self.dead[card] = queue.Queue(), threading.Event()
            station, kind = CARD_LINE.get(card, ("image", "refs"))

            def serve(card=card, ff=ff, station=station, kind=kind):
                self.rc[card] = line(station, kind, f"{tag}_{card}", logd / f"{tag}_{card}.log",
                                     ["bash", "-c", f"cd {root} && bash {CARD_SH} {card} {IMAGE_PY} -I "
                                                    f"tools/qwen_image.py --serve < {ff}"])
                self.dead[card].set()
                # a drawer that ended before it was told to (a card that stopped answering): what it was fed and
                # did not draw goes to a card still drawing
                left = [] if self.closing else undrawn(self.fed[card], logd / f"{tag}_{card}.log")
                live = [c for c in self.cards if not self.dead[c].is_set()]
                if left and live:
                    log(f"drawing: the drawer on card {card} ended early; {len(left)} picture(s) go to card {live[0]}")
                    self.fed[card] = []
                    self._give(live[0], left)
                    return
                if left:
                    log(f"drawing: the drawer on card {card} ended early; {len(left)} picture(s) NOT drawn")
                self.alive.unlink(missing_ok=True)

            def write(card=card, fd=fd, ff=ff):
                while True:
                    row = self.q[card].get()
                    if row is None:
                        break
                    data = (row + "\n").encode()
                    while data and not self.dead[card].is_set():
                        try:
                            data = data[os.write(fd, data):]
                        except BlockingIOError:
                            time.sleep(0.5)
                os.close(fd)
                ff.unlink(missing_ok=True)
            self.ths += [threading.Thread(target=serve), threading.Thread(target=write)]
        for t in self.ths:
            t.start()

    def _give(self, card: str, rows: list) -> None:
        self.fed[card] += rows
        for row in rows:
            self.q[card].put(row)

    def feed(self, rows: list) -> None:
        live = [c for c in self.cards if not self.dead[c].is_set()]
        for card, part in zip(live, split_even(rows, len(live))):
            self._give(card, part)

    def close(self) -> bool:
        self.closing = True
        for card in self.cards:
            self.q[card].put(None)
        for t in self.ths:
            t.join()
        return all(self.rc.get(c) == 0 for c in self.cards)


def undrawn(rows: list, card_log: Path) -> list:
    """The job rows a drawer was fed that its log does not report as drawn (a row it reports an error for counts as
    not drawn)."""
    done = set()
    for x in (card_log.read_text(errors="replace").splitlines() if card_log.exists() else []):
        if x.startswith('{"id"'):
            try:
                j = json.loads(x)
            except ValueError:
                continue
            if "error" not in j:
                done.add(j.get("id"))
    return [r for r in rows if json.loads(r).get("id") not in done]


def rows_of(files: list) -> list:
    return [x for f_ in files if Path(f_).exists() for x in Path(f_).read_text().splitlines() if x.strip()]


def check_beside(files: list, tag: str, logd: Path, alive: Path | None = None) -> bool:
    """The key checks of these check files on the iGPU while the cards draw: each picture checked when it lands
    (keycheck.py KEYCHECK_FOLLOW_S), the models loaded once for every list."""
    files = [str(f_) for f_ in files if Path(f_).exists()]
    if not files:
        return True
    root = HERE.parents[1]
    inner = (f"env HSA_CU_MASK=$HIP_VISIBLE_DEVICES:0-38 DETECT_DEVICE=cuda KEYCHECK_FOLLOW_S={FOLLOW_S:.0f} "
             f"KEYCHECK_ALIVE={alive or ''} "
             f"{IGPU_PY} -I tools/assets/redraw/keycheck.py {' '.join(files)}")
    return line("igpu", "check", tag, logd / f"{tag}.log",
                ["bash", "-c", f"cd {root} && bash {CARD_SH} igpu bash -c {inner!r}"]) == 0


def check_files(files: list, tag: str, logd: Path) -> bool:
    """The key checks (redraw/keycheck.py) of these check files, shared between the drawing cards."""
    files = [str(f_) for f_ in files if Path(f_).exists()]
    if not files:
        return True

    def cmd(part, card, n):
        py = f"env DETECT_DEVICE=cuda {IMAGE_PY} -I tools/assets/redraw/keycheck.py"
        return "bash -c " + repr(" && ".join(f"{py} {c}" for c in part))
    return on_cards(split_even(files, len(DRAW_CARDS)), cmd, tag, logd)


def llm(jid: str, logf: Path, cmds: list) -> int:
    """Commands that need the directing model, as one line job: the model is started for them and stopped after."""
    import shlex
    script = logf.with_suffix(".sh")
    # the line's resident editor idles for a long while after an engine step and holds the few GB the model is
    # short of: asked to unload first (a busy one finishes its job; it loads again for the next engine step).
    # Without it every place's model step after a cast step waited its ten minutes and failed (11-16)
    free_up = (f"M=$(awk '/MemAvailable/ {{print int($2/1048576)}}' /proc/meminfo); "
               f"if [ \"$M\" -lt {LLM_NEED_GB} ]; then {' '.join(shlex.quote(str(x)) for x in RA)} --stop-editor; fi")
    lines = ["#!/bin/bash", f"cd {HERE.parents[1]}", "S=0",
             f"if ! systemctl --user is-active -q strata-eval; then {free_up}; "
             f"{VENV_PY} -c \"{LLM_UP}\" || exit 1; S=1; fi",
             "trap '[ $S = 1 ] && systemctl --user stop strata-eval' EXIT", "RC=0"]
    lines += [" ".join(shlex.quote(str(a)) for a in c) + " || RC=1" for c in cmds]     # (a place's name may hold a quote)
    lines += ["[ $S = 1 ] && systemctl --user stop strata-eval", "exit $RC"]
    script.write_text("\n".join(lines) + "\n")
    return line("igpu", "llm", jid, logf, ["bash", str(script)])


def llm_down() -> None:
    """The directing model is not left up between steps: it holds most of the box's memory, and an engine or card
    job behind it waits for room (a figure's import sat ten minutes behind an idle model)."""
    if subprocess.run(["systemctl", "--user", "is-active", "-q", "strata-eval"]).returncode == 0:
        # not while a model job of the line is running or waiting: another story's step owns it then, and its own
        # job stops it after (11-16: one run's end stopped the model 16 s after another's step had started it)
        st = subprocess.run(RA + ["--status"], capture_output=True, text=True).stdout
        igpu = next((ln for ln in st.splitlines() if ln.startswith("igpu")), "")
        if " llm " in igpu or "run: idle" not in igpu or "wait 0" not in igpu:
            log("the directing model is up for another job on the line: left running")
            return
        subprocess.run(["systemctl", "--user", "stop", "strata-eval"])
        log("the directing model was still up: stopped")


# (said as "with the wearer's face in view", it was drawn with a stranger's face inside it)
HEAD_OPEN = ("; an open-faced one, shown empty with nobody inside it: the whole front open where the wearer's face "
             "would be, any visor raised out of the way, no mask across the opening")
# (a book's look of a thing tells where it lies too: a sword "resting on a stone block" was made with its block)
ALONE = "; the thing alone and whole, with nothing under it, behind it or around it"
DRAWN_MAX_SHOTS = 0           # a place the story shows in this many shots or fewer is drawn, not built (0: every place
#                               is built; 11-17 user: engine keys for everything, drawn keys brought the wrong place,
#                               costume and identity)
DRAWN_SEEDS = (7, 11, 23)     # a drawn key's tries
SIDE_BOX = {"left": [0.08, 0.12, 0.46, 1.0], "right": [0.54, 0.12, 0.92, 1.0], "center": [0.3, 0.1, 0.7, 1.0],
            "centre": [0.3, 0.1, 0.7, 1.0]}       # where the plan's side puts a drawn person in the picture
KEY_DARK = 10.0              # a key's mean brightness (of 255) under this: nearly black, not a key
CONCEPT_TRIES = 3
SEATS_WORDS = ("{room} {inside} Seen from just behind and above the seats, looking forward over the instrument panel "
               "and out through the canopy. It has {seats}, and no other seat anywhere. Empty: no people. {style}")

# a place of wealth or rule: its picture drawn anew from its own picture, furnished to its station (place_dress.py)
DRESS_WORDS = ("The same room as the reference picture, seen from the same place: its walls, its windows, its doors "
               "and its furniture where they are. Now furnished and adorned as the {station} {use} it is, richly and "
               "in fine detail: {dressing}. {room} Interior view, wide shot of the whole room, eye level. {period} "
               "Empty: no people. {style}")

# a place in the open whose picture is a close view of one part of it: drawn anew as the whole place
OPEN_WORDS = ("{room} Part of it: {inside} A wide view of the whole place from well back, at eye level: open ground in "
              "front, everything that stands there seen whole from the ground up and small in the picture, the land "
              "around it and the sky above. Empty: no people. {style}")


def person_noun(look: str) -> str:
    """What a person is called in a picture's words: the look's own "man", "woman", "boy" or "girl" (its first), else
    the word after its article. (A look that began "a tall, slim young man" made its person "the tall": every key
    said "One tall is in the picture" and nothing said he was a man, 11-26.)"""
    m = re.search(r"\b(man|woman|boy|girl)\b", look.split(" wearing ")[0].lower())
    return m.group(1) if m else re.match(r"an? (\w+)", look).group(1)


def look_person(look: str) -> str:
    """The person as the look tells them, without what they wear, hold or express: their age, build, skin, hair and
    face in the look's own words."""
    head = re.split(r",?\s+(?:wearing|dressed in|holding|carrying)\b", look.strip().rstrip("."), maxsplit=1)[0]
    return head.strip().rstrip(",")


def crowd_facts(keys: Path, shots: list, set_dir: Path) -> int:
    """Each key's facts say when its place has a crowd ("crowd": true): the people of the place who stand behind the
    story's people are in the picture and in every clip made from it. Counted against the one person the key
    lists, a ballroom's every take was faulted "5 people, not 1" and made four times over (11-26). -> keys marked."""
    try:
        crowd = str(json.loads((Path(set_dir) / "spec.json").read_text()).get("crowd") or "").strip()
    except (OSError, ValueError):
        crowd = ""
    n = 0
    for sid in shots if crowd else []:
        ff = Path(keys) / f"{sid}.shot_facts.json"
        if ff.exists():
            f = json.loads(ff.read_text())
            f["crowd"] = crowd
            ff.write_text(json.dumps(f, indent=1))
            n += 1
    return n


def chapter_timeline(book: Path, plan_p: Path):
    """The narration timeline of the chapter the plan is of: the plan lies in <book>/work/video/<chapter>/shots/, the
    timeline is <book>/work/<chapter>.timeline.json. A book of one chapter has only one; in a book of several the
    first was read for every chapter, and a dance in the fourth was specified, staged and lit from the lines spoken at
    the same minute of the first (11-26). With no chapter to be told from the plan's place: the book's only timeline,
    else none."""
    tl = sorted(q for q in (book / "work").glob("*.timeline.json") if ".raw." not in q.name)
    for anc in Path(plan_p).resolve().parents:
        hit = [q for q in tl if q.name == f"{anc.name}.timeline.json"]
        if hit:
            return hit[0]
    return tl[0] if len(tl) == 1 else None


class Story:
    def __init__(self, plan_p: Path, book: Path, work: Path, keys: Path, prefix: str):
        self.plan_p, self.book, self.work, self.keys, self.prefix = plan_p, book, work, keys, prefix
        self.plan = json.loads(plan_p.read_text())
        self.video = book / "work" / "video"
        self.look_p = self.video / "look.json"
        self.look = json.loads(self.look_p.read_text())
        self.timeline = chapter_timeline(book, plan_p)
        work.mkdir(parents=True, exist_ok=True)
        keys.mkdir(parents=True, exist_ok=True)
        self.ids = [k for k, v in self.plan.items() if not k.startswith("_") and isinstance(v, list) and v]
        self.places_p, self.cast_p = work / "places.json", work / "cast.json"
        self.places = json.loads(self.places_p.read_text()) if self.places_p.exists() else {}
        self.cast = json.loads(self.cast_p.read_text()) if self.cast_p.exists() else {}
        for i in self.ids:                                  # every place with people, in story order
            s = self.plan[i][0]
            if s.get("place") and s.get("people") and s["place"] not in self.places:
                sl = slug_of(s["place"])
                self.places[s["place"]] = {"slug": sl, "set": f"{prefix}_{sl}_01"}
        self.places_p.write_text(json.dumps(self.places, indent=1))

    # ------------------------------------------------------------ bookkeeping
    def d(self, place: str) -> Path:
        p = self.work / self.places[place]["slug"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    def done(self, place: str, step: str) -> bool:
        fp = self.d(place) / f".done_{step}"
        if not fp.exists():
            return False
        # a step whose rule changed since it ran is run again, in a place still being made (a place whose keys are
        # made stays as it is: its set failed on a spec written under the old rule, and every restart kept that spec)
        n = STEP_RULE.get(step)
        if n and f"rule {n}" not in fp.read_text() and not (self.d(place) / ".done_keys").exists():
            return False
        return True

    def mark(self, place: str, step: str) -> None:
        n = STEP_RULE.get(step)
        (self.d(place) / f".done_{step}").write_text(time.strftime("%F %T") + (f" rule {n}" if n else ""))

    def shots_of(self, place: str) -> list:
        return [i for i in self.ids if self.plan[i][0].get("place") == place]      # (with or without people)

    def people_of(self, place: str) -> list:
        names = dict.fromkeys(p["name"] for i in self.shots_of(place) for p in self.plan[i][0].get("people") or [])
        # (the book's cast only: a crowd the plan names has no sheet and no figure, and is not staged)
        return [n for n in names if n in self.look.get("characters", {})
                and (self.video / "charlora" / slug_of(n) / "build.json").exists()]

    def set_dir(self, place: str) -> Path:
        return C.LIBRARY / "set" / self.places[place]["set"]

    # ------------------------------------------------------------ the cast's facts, from the book
    def person(self, name: str) -> dict:
        """A person's build facts from the book: look line, sheet, LoRA, trigger, and the redraw's words."""
        sl = slug_of(name)
        b = json.loads((self.video / "charlora" / sl / "build.json").read_text())
        look = self.look["characters"][name]
        sheets = sorted(self.video.glob(f"refs/{sl}-*.png"))
        noun = person_noun(look)
        m = re.search(r"wearing (.+?)(?:, (?:expression|holding|face)\b|\.$|$)", look)
        wear = m.group(1).strip() if m else ""
        pos = "her" if noun in ("woman", "girl") else "his"
        short = wear.split(",")[0]
        # the book's sheets of this person in something more than their usual look (refs/<person>_<what>-<hash>.png;
        # a sheet named for a thing's absence is the usual look again)
        worn = {q.stem[len(sl) + 1:].rsplit("-", 1)[0].replace("_", " "): str(q)
                for q in sorted(self.video.glob(f"refs/{sl}_*-*.png"))}
        worn = {n: q for n, q in worn.items() if not n.startswith(("no ", "without "))}
        if b.get("kind") == "creature":
            # a creature (the book's own record of what it built a LoRA for): named by the last word of the look's
            # head phrase, said as "it", with nothing worn; staged as a library creature (step_cast)
            from shot_people import creature_phrase
            noun = (creature_phrase(look) or "the creature")[4:]
            return {"look": look, "sheet": str(sheets[0]) if sheets else b["sheet"], "lora": b["lora"],
                    "trigger": b["trigger"], "noun": noun, "slug": sl, "creature": True,
                    "words": {"noun": noun, "trigger": b["trigger"], "lora": b["lora"], "pos": "its", "subj": "it",
                              "sheet": str(sheets[0]) if sheets else b["sheet"], "wear": "", "worn_sheets": {},
                              "wear_back": f"the back of the {noun}", "back_detail": "", "creature": True}}
        return {"look": look, "sheet": str(sheets[0]) if sheets else b["sheet"], "lora": b["lora"],
                "trigger": b["trigger"], "noun": noun, "slug": sl,
                "words": {"noun": noun, "trigger": b["trigger"], "lora": b["lora"],
                          "sheet": str(sheets[0]) if sheets else b["sheet"], "wear": wear, "worn_sheets": worn,
                          "as": look_person(look),
                          "wear_back": f"the back of {pos} {short} and the back of {pos} head",
                          "back_detail": f"the back of {pos} {short}"}}

    def wardrobe(self, name: str) -> list:
        """The person's changes of clothes over the chapter, on its clock (scenes/wardrobe.py timed rows): the book
        planner's wardrobe (its plan.json "wardrobe", with the shift for an outfit the story makes before it is worn,
        as the planner's other readers take it); [] when the book has none."""
        if not hasattr(self, "_ward"):
            self._ward = {}
            try:
                sys.path[:0] = [str(HERE.parents[1]), str(HERE.parent)]
                from tts_reader import video as V
                import characters as CH
                stem = self.timeline.name.split(".")[0]
                plan = json.loads((self.video / stem / "plan.json").read_text())
                paras = V.paragraphs((self.book / "chapters" / f"{stem}.txt").read_text())
                starts = V.paragraph_times(paras, json.loads(self.timeline.read_text()))
                plan = CH.made_outfits_shift(plan, paras)
                for nm in self.look.get("characters", {}):
                    k = V.wardrobe_key(plan, nm)
                    if k:
                        self._ward[nm] = WD.timed(plan["wardrobe"][k], starts)
            except Exception as e:                      # noqa: BLE001 - no wardrobe: everyone in their usual look
                log(f"the book's wardrobe could not be read ({e}); everyone is staged in their usual look")
        return self._ward.get(name, [])

    def outfit(self, name: str, sid: str) -> dict | None:
        """The wardrobe row in force for this person at this shot's time; None = their usual look."""
        return WD.in_force(self.wardrobe(name), WD.shot_time(sid, (self.plan.get(sid) or [{}])[0].get("t")))

    def words_at(self, name: str, sid: str) -> dict:
        """The person's redraw words at this shot: their usual ones, or the outfit in force's (its dress, its sheet)."""
        pr = self.person(name)
        w = dict(pr["words"])
        r = None if pr.get("creature") else self.outfit(name, sid)
        if r:
            w.update(self.outfit_words(name, r))
        return w

    def outfit_words(self, name: str, r: dict) -> dict:
        pr = self.person(name)
        sys.path.insert(0, str(HERE.parents[1]))
        from tts_reader import video as V
        dress = V.worn_only(r["dress"]) or r["dress"]
        # what the variant's figure was drawn and built in (cast_char.spread_look: no headwear, nothing held): a hat
        # or helm reaches the key as a worn thing of the shot (place_spec's things), made with the face in view
        sys.path.insert(0, str(HERE / "img3d"))
        from cast_char import head_worn, spread_look
        # (a whole figure is shaped from its own drawing, so it is made in the outfit's headwear too: the outfit
        # with the thing off is another row of the wardrobe)
        whole = bool(getattr(self, "whole", True))
        head_on = whole and head_worn(dress)
        dress = spread_look("x, wearing " + dress, headwear=whole).partition("wearing ")[2] or dress
        dress = re.sub(r"\b(his|her|their)\s+usual\s+", r"\1 ", dress, flags=re.I)   # (the wardrobe's word)
        pos = "her" if pr["noun"] in ("woman", "girl") else "his"
        dress = dress[:1].lower() + dress[1:]
        short = WD.outer_garment(dress) or dress.split(",")[0]
        sheet = WD.sheet_for(pr["words"].get("worn_sheets") or {}, r["id"]) or pr["sheet"]
        return {"wear": dress, "sheet": sheet, "wear_back": f"the back of {pos} {short} and the back of {pos} head",
                "back_detail": f"the back of {pos} {short}", "head_on": head_on}

    def figure_id(self, name: str) -> str:
        if name not in self.cast:
            self.cast[name] = f"{self.person(name)['noun']}_{self.prefix}_{slug_of(name)}_auto_01"
            self.cast_p.write_text(json.dumps(self.cast, indent=1))
        return self.cast[name]

    # ------------------------------------------------------------ steps
    def step_shots(self, place):
        d = self.d(place)
        (d / "shots.txt").write_text("\n".join(self.shots_of(place)) + "\n")
        (d / "place.txt").write_text(place)
        return True

    def step_spec(self, place):
        d = self.d(place)
        rc = llm(f"spec_{self.places[place]['slug']}", d / "spec.log", [[
            PY, HERE / "scenes" / "place_spec.py", self.plan_p, d / "shots.txt", self.timeline, place,
            d / "place_spec.json", "--cache", d / "cache"]])
        if not (d / "place_spec.json").exists() or rc not in (0, 1):    # (1: written with faults, reported there)
            return False
        # where the plan has people lying, by its own place phrases (the set builder gives each such place a bed)
        lying = list(dict.fromkeys(str(p.get("at")) for i in self.shots_of(place) for p in self.plan[i][0].get("people") or []
                                   if p.get("at") and ("ly" in str(p.get("pose")) or "lie" in str(p.get("pose")))))
        sp = json.loads((d / "place_spec.json").read_text())
        sp["_lying_phrases"] = lying
        (d / "place_spec.json").write_text(json.dumps(sp, indent=1))
        return True

    def dressed(self, place, hero: Path) -> Path | None:
        """The place's picture drawn anew from its own picture, furnished to its owner's station (scenes/
        place_dress.py): only a rich or royal place; any other keeps the picture it has. The director's words
        furnish nothing: the hall of a ruler was drawn, and so built, as bare stone and trestle tables (11-20)."""
        d = self.d(place)
        slug = self.places[place]["slug"]
        if not (d / "place_spec.json").exists():
            return None
        llm(f"dress_{slug}", d / "dress.log", [[PY, HERE / "scenes" / "place_dress.py", d / "place_spec.json",
                                                d / "dress.json", "--cache", d / "cache"]])
        rec = json.loads((d / "dress.json").read_text()) if (d / "dress.json").exists() else {}
        if not rec.get("dressing"):
            return None
        from PIL import Image
        spec = json.loads((d / "place_spec.json").read_text())
        style = self.book / "art_style.txt"
        per = str(spec.get("period") or "").replace("_", " ").strip()
        words = DRESS_WORDS.format(
            station=rec["station"], use=rec.get("use") or "place", dressing="; ".join(rec["dressing"]),
            room=re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or spec.get("room") or ""),
                        flags=re.I).strip(),
            period=f"Set in the {per} period: every object as it was then, nothing from a later time."
            if per and per != "not stated" else "",
            style=key_style(style.read_text()) if style.exists() else "")
        pid = f"concept_{self.prefix}_{slug}_d_{hashlib.sha1(words.encode()).hexdigest()[:6]}"
        pic = C.LIBRARY / "_refs" / pid / "picture.png"
        if not pic.exists():
            size = Image.open(hero).size
            line(C.IMAGE_STATION, "refs", f"draw_{pid}", d / "concept_draw.log",
                 [PY, HERE / "refs" / "ref_draw.py", "picture", pid, "--desc", words,
                  "--size", size[0], size[1], "--seed", 11, "--sources", hero])
        if pic.exists():
            log(f"{place}: its picture drawn dressed as a {rec['station']} place ({len(rec['dressing'])} things)")
            return pic
        return None

    def step_concept(self, place):
        d = self.d(place)
        hero = self.video / "heroes" / f"{slug_of(place)}.png"
        if not hero.exists():
            log(f"{place}: no concept picture ({hero.name}); built from the spec alone")
            return True
        shutil.copy2(hero, d / "concept.png")
        slug = self.places[place]["slug"]
        dressed = self.dressed(place, hero)       # (a place of wealth or rule: its picture drawn anew, dressed)
        if dressed:
            shutil.copy2(dressed, d / "concept.png")
        read = lambda n: line("igpu", "choose", f"concept_{slug}{n}", d / "concept.log",          # noqa: E731
                              [VENV_PY, HERE / "scenes" / "concept.py", d / "concept.png", d / "place_spec.json",
                               d / "concept.json"])
        read("")
        if not (d / "concept.json").exists():
            return False
        # a cabin's picture with more seats abreast than the book's place has: the cabin is shaped from the picture,
        # so the concept is drawn anew from the place's own words with the book's seats (a second seat in view was
        # filled with a second pilot, 11-13; drawn from the old picture, its seats stayed and one more was added),
        # and read again; the first picture that reads right is the concept
        spec = json.loads((d / "place_spec.json").read_text())
        want = sum(1 for q in spec.get("pieces") or [] if "seat" in q.get("kind", "") or "chair" in q.get("kind", ""))
        con = json.loads((d / "concept.json").read_text())
        if con.get("space") == "cockpit" and want and int(con.get("seats_abreast") or 1) > want:
            from PIL import Image
            size = Image.open(hero).size
            style = self.book / "art_style.txt"
            words = SEATS_WORDS.format(
                room=str(spec.get("room") or "").strip(),
                inside=re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or ""), flags=re.I).strip(),
                seats="a single seat in the middle, for one person" if want == 1 else f"{want} seats",
                style=key_style(style.read_text()) if style.exists() else "")
            for n in range(CONCEPT_TRIES):
                pid = f"concept_{self.prefix}_{slug}_w{n}"
                pic = C.LIBRARY / "_refs" / pid / "picture.png"
                line(C.IMAGE_STATION, "refs", f"draw_{pid}", d / "concept_draw.log",
                     [PY, HERE / "refs" / "ref_draw.py", "picture", pid, "--desc", words,
                      "--size", size[0], size[1], "--seed", 11 + n])
                if not pic.exists():
                    continue
                shutil.copy2(pic, d / "concept.png")
                (d / "concept.json").unlink()
                read(f"_{n}")
                got = json.loads((d / "concept.json").read_text()) if (d / "concept.json").exists() else {}
                log(f"{place}: concept drawn anew with {want} seat(s), seed {11 + n}: reads "
                    f"{got.get('space')}, {got.get('seats_abreast')} abreast")
                if got.get("space") == "cockpit" and int(got.get("seats_abreast") or 1) <= want:
                    break
            else:
                log(f"{place}: no picture with {want} seat(s) in {CONCEPT_TRIES} tries")
                return False
        con = json.loads((d / "concept.json").read_text())
        if (con.get("space") == "outdoors" or spec.get("kind") == "outdoors") and \
                (con.get("view") == "part" or con.get("cut") or str(spec.get("whole") or "").strip()):
            # (the picture of a place in the open was one door in a wall: the world made from it was bare land with
            # no building, and every key drew another background, 11-18)
            from PIL import Image
            size = Image.open(hero).size
            style = self.book / "art_style.txt"
            # (the whole place in the spec's own words, when it has them: the set sheet's may name one part only)
            room = str(spec.get("whole") or spec.get("room") or "").strip()
            per = str(spec.get("period") or "").replace("_", " ").strip()
            words = OPEN_WORDS.format(
                room=(f"{per[0].upper() + per[1:]}: " if per and per != "not stated" else "") + room,
                inside=re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or ""), flags=re.I).strip(),
                style=key_style(style.read_text(), open_air=True) if style.exists() else "")
            for n in range(CONCEPT_TRIES):
                pid = f"concept_{self.prefix}_{slug}_v{n}_{hashlib.sha1(words.encode()).hexdigest()[:6]}"
                pic = C.LIBRARY / "_refs" / pid / "picture.png"
                if not pic.exists():         # (its id is its words: drawn once, a later run reads it again)
                    line(C.IMAGE_STATION, "refs", f"draw_{pid}", d / "concept_draw.log",
                         [PY, HERE / "refs" / "ref_draw.py", "picture", pid, "--desc", words,
                          "--size", size[0], size[1], "--seed", 11 + n])
                if not pic.exists():
                    continue
                shutil.copy2(pic, d / "concept.png")
                (d / "concept.json").unlink()
                read(f"_v{n}")
                got = json.loads((d / "concept.json").read_text()) if (d / "concept.json").exists() else {}
                log(f"{place}: drawn as the whole place, seed {11 + n}: reads {got.get('space')}, {got.get('view')}"
                    f"{', a building cut by the edge' if got.get('cut') else ''}")
                if got:
                    got["land"] = room             # the land's words: the place's own, not the one part's
                    (d / "concept.json").write_text(json.dumps(got, indent=1))
                # (a building the edge cuts is still one piece made from the picture: the first whole view is kept)
                if got.get("view") == "whole" and got.get("space") == "outdoors":
                    break
            else:
                log(f"{place}: no picture of the whole place in {CONCEPT_TRIES} tries; the last one is kept")
        if (d / "concept.json").exists():
            self.concept_depth(place)
        return (d / "concept.json").exists()

    def concept_depth(self, place) -> None:
        """A room's concept picture measured (img3d/concept_depth.py, the depth model at the image station): the
        room's size and each piece's place, into concept.json's "depth" for the set builder. A picture that cannot
        be measured leaves the builder to the room kind's table and free spots, as before."""
        d = self.d(place)
        con = json.loads((d / "concept.json").read_text())
        if con.get("space") not in ("room", "outdoors"):   # (a place in the open: where what stands on it stands)
            return
        spec = json.loads((d / "place_spec.json").read_text())
        con["_spec_kinds"] = {p["id"]: p.get("kind") for p in spec.get("pieces") or []}
        (d / "concept.json").write_text(json.dumps(con, indent=1))
        (d / "concept_depth.json").unlink(missing_ok=True)
        py = os.environ.get("ASSETS_SHELL_PY") or PY
        line(C.IMAGE_STATION, "shape", f"concept_depth_{self.places[place]['slug']}", d / "concept_depth.log",
             [py, HERE / "img3d" / "concept_depth.py", d / "concept.png", d / "concept.json", d / "concept_depth.json"])
        if (d / "concept_depth.json").exists():
            con["depth"] = json.loads((d / "concept_depth.json").read_text())
            (d / "concept.json").write_text(json.dumps(con, indent=1))
            log(f"{place}: the concept measured: room {con['depth']['room_m']}, {len(con['depth']['things'])} things placed")
        else:
            log(f"{place}: the concept could not be measured (concept_depth.log)")

    def place_time(self, place) -> str:
        """The place's time of day ("day", "dusk", "night", or "" when the story does not say), asked of the
        directing model once (scenes/place_spec.py time_of_day) and kept in the place's place_time.json."""
        d = self.d(place)
        # (the book's own word first: BOOK/place_times.json {place: day | dusk | night}, written by whoever knows
        # the story; the model read a night scene's shed as no time at all, 11-19)
        said = self.book / "place_times.json"
        if said.exists():
            try:
                t = str(json.loads(said.read_text()).get(place) or "").strip().lower()
                if t in ("day", "dusk", "night"):
                    return t
            except ValueError:
                log(f"{said} cannot be read; the model's answers are used")
        tp = d / "place_time.json"
        if not tp.exists():
            llm(f"time_{self.places[place]['slug']}", d / "time.log", [[
                PY, HERE / "scenes" / "place_spec.py", self.plan_p, d / "shots.txt", self.timeline, place, tp,
                "--cache", d / "cache", "--time-only"]])
        try:
            return str(json.loads(tp.read_text()).get("time") or "")
        except (OSError, ValueError):
            return ""

    def _build(self, place, *extra) -> int:
        d = self.d(place)
        con = ["--concept", d / "concept.json"] if (d / "concept.json").exists() else []
        tod = self.place_time(place) if "--time" in extra else ""
        extra = tuple(x for x in extra if x != "--time") + (("--time", tod) if tod else ())
        sty = ["--style", self.book / "art_style.txt"] if (self.book / "art_style.txt").exists() else []
        # (the things the place's picture was drawn dressed with: each one placed in the set or said where it is)
        drs = ["--dress", d / "dress.json"] if con and (d / "dress.json").exists() else []
        return sh([PY, HERE / "sets" / "place_build.py", d / "place_spec.json", self.places[place]["set"]] + con
                  + drs + sty + list(extra), d / "place_build.log")

    def step_pieces(self, place):
        d = self.d(place)
        self._build(place, "--emit", d / "place_spec_c.json")
        rc = sh([PY, HERE / "places" / "generate.py", d / "place_spec_c.json"], d / "generate.log")
        if rc:
            log(f"{place}: some pieces were not made (generate.log); the set is built with what there is")
        return True

    def step_set(self, place):
        d = self.d(place)
        if self._build(place, "--write", "--time"):
            return False
        rep = json.loads((self.set_dir(place) / "place_build_report.json").read_text())
        gone = [ph for ph, at in (rep.get("named_places") or {}).items() if not at]
        if gone:                         # a place the plan sends people to is not in the room: not a set to stage in
            log(f"{place}: the set lacks the plan's places {gone}: {(rep.get('unmet') or [])[:3]}")
            return False
        if "world" in rep and not (rep["world"] or {}).get("world"):    # a place in the open without its land and sky
            log(f"{place}: the place's world is missing: {(rep['world'] or {}).get('why')}")
            return False
        if sh([PY, HERE / "sets" / "build_set.py", self.set_dir(place) / "spec.json"], d / "build_set.log"):
            return False
        # a set whose own stills are nearly black is no set to stage in (three rooms lit only by a sky light that
        # does not reach inside staged every shot black, 11-16)
        from PIL import Image
        lums = [sum(Image.open(q).convert("L").resize((64, 36)).getdata()) / (64 * 36)
                for q in sorted((self.set_dir(place) / "ue").glob("*.png"))]
        if lums and max(lums) < KEY_DARK:
            log(f"{place}: the set's own stills are nearly black (brightest {max(lums):.1f} of 255)")
            return False
        return True

    def _cast_cmd(self, name):
        fid, pr = self.figure_id(name), self.person(name)
        w = self.work / "cast" / pr["slug"]
        w.mkdir(parents=True, exist_ok=True)
        return fid, w, [PY, HERE / "img3d" / "cast_char.py", "--id", fid, "--sheet", pr["sheet"], "--look", pr["look"],
                        "--lora", pr["lora"], "--trigger", pr["trigger"], "--work", w / "w",
                        "--scene-tool", HERE / "scenes"] + (["--whole"] if getattr(self, "whole", True) else ["--metahuman"])

    def _variant_cmd(self, name, r):
        """A wardrobe variant's figure: the same person on the base figure's MetaHuman, drawn in the outfit (the
        book's own sheet of them in it when it has one, else the usual sheet with the outfit's words), recorded as
        a variant of the base figure (staging_pass.variants offers it)."""
        base, pr = self.figure_id(name), self.person(name)
        bw = self.work / "cast" / pr["slug"] / "w"
        bj = json.loads((C.LIBRARY / "character" / base / "character.json").read_text())
        mh = str(bj.get("rebuild_from") or "").rsplit("/", 1)[-1]
        ch = json.loads((bw / "choices.json").read_text()) if (bw / "choices.json").exists() else {}
        bp = (ch.get("mh_build") or {}).get("blueprint") or f"/Game/MetaHumans/{mh}/BP_{mh}"
        fbx = bw / "fbx"                                 # (the base build's body and head of that MetaHuman)
        ow = self.outfit_words(name, r)
        vid = WD.variant_id(base, r["id"])
        w = self.work / "cast" / pr["slug"] / r["variant"]
        w.mkdir(parents=True, exist_ok=True)
        return vid, w, [PY, HERE / "img3d" / "cast_char.py", "--id", vid, "--sheet", ow["sheet"],
                        "--look", WD.variant_look(pr["look"], ow["wear"]), "--lora", pr["lora"], "--trigger",
                        pr["trigger"], "--work", w / "w", "--scene-tool", HERE / "scenes", "--mh", mh, "--mh-fbx", fbx,
                        "--mh-blueprint", bp, "--variant-of", base] + (
                            ["--whole", "--headwear"] if getattr(self, "whole", True) else ["--metahuman"])

    def _cast_variants(self, place, name) -> dict:
        """The person's wardrobe variants worn in this place's shots (each built once for the book):
        {variant: (figure id or None when its build failed, the row)}."""
        out = {}
        for i in self.shots_of(place):
            # (in the shot: listed by the plan, or its talk's speaker or listener, or one of its "who": a talk shot
            # lists nobody, and an outfit first worn in talk shots was never built, so its wearer was drawn in his
            # usual clothes, 11-17)
            sh_ = self.plan[i][0]
            if name not in {p["name"] for p in sh_.get("people") or []} | {sh_.get("speaker"), sh_.get("listener")} \
                    | set(sh_.get("who") or []):
                continue
            r = self.outfit(name, i)
            if not r or r["variant"] in out:
                continue
            vid, w, cmd = self._variant_cmd(name, r)
            ok = (C.LIBRARY / "character" / vid / "character.json").exists()
            if not ok and not (w / "FAILED").exists():
                log(f"{place}: {name}'s figure in {r['id']} ({vid})")
                self._fit_mark(w)                # (built now with every fitting rule)
                if sh(cmd + ["--until", "seats"], w / "run.log") and \
                        not (C.LIBRARY / "character" / vid / "character.json").exists():
                    sh(cmd + ["--from", "draw", "--until", "seats"], w / "run.log")
                ok = (C.LIBRARY / "character" / vid / "character.json").exists()
                if not ok:
                    log(f"{place}: no figure for {name} in {r['id']} ({vid}); see {w / 'run.log'}")
                    (w / "FAILED").write_text(time.strftime("%F %T"))
            if ok and self.outfit_words(name, r).get("head_on") and not (w / "headwear_drawn").exists():
                # a figure built before an outfit's headwear was part of it: drawn and shaped again, once
                log(f"{place}: {name} in {r['id']}: made again wearing its headwear")
                if sh(cmd + ["--from", "draw", "--until", "seats"], w / "run.log") == 0:
                    self._fit_mark(w)
                    (w / "headwear_drawn").write_text(time.strftime("%F %T"))
                else:
                    # (the figure made before stays, bare-headed, and so does the thing set on its head)
                    log(f"{place}: {name} in {r['id']}: NOT made with its headwear; see {w / 'run.log'}")
            if ok:
                self._hands_again(f"{name} in {r['id']}", w, cmd)
            out[r["variant"]] = (vid if ok else None, r)
        return out

    FITS = ("hands_again", "head_again", "depth_again")     # the fittings a whole figure has had since its first build, by marker

    def _hands_again(self, who, w, cmd) -> None:
        """A whole figure built before a fitting rule changed is made again from its fitting, once per rule (FITS):
        before its hands were jointed the statue's own fused hands stayed open round every handle (11-18); before
        its head followed the head joint by the head's own line its ears and nape followed the shoulders and stood
        out as points when the arms came down (11-19). A figure built new has them all (_fit_mark)."""
        ch = w / "w" / "choices.json" if (w / "w" / "choices.json").exists() else w / "choices.json"
        if not getattr(self, "whole", True) or not ch.exists() or all((w / f).exists() for f in self.FITS):
            return
        self._fit_mark(w)
        log(f"{who}: its figure is fitted again (jointed hands, the head on the head joint, its depth the body's)")
        if sh(cmd + ["--from", "combo", "--until", "seats"], w / "run.log"):
            log(f"{who}: the fitting again FAILED; see {w / 'run.log'}")

    def _fit_mark(self, w) -> None:
        for f in self.FITS:
            if not (w / f).exists():
                (w / f).write_text(time.strftime("%F %T"))

    def _sheet_newer(self, name) -> bool:
        """Whether the person's sheet was drawn after their figure's record was written."""
        try:
            rec = C.LIBRARY / "character" / self.cast[name] / "character.json"
            return Path(self.person(name)["sheet"]).stat().st_mtime > rec.stat().st_mtime
        except (OSError, KeyError, TypeError):
            return False

    def _built(self, name) -> bool:
        if name in self.cast and CRE.is_id(self.cast[name]):            # a library creature with its rig
            return CRE.rig(C.LIBRARY, self.cast[name]) is not None
        return name in self.cast and (C.LIBRARY / "character" / self.cast[name] / "character.json").exists()

    def _cast_creature(self, place, name) -> bool:
        """A creature of the book is the library's rigged creature made for its look: the directing model picks it
        among the rigged creatures from the look line and their own descriptions (characters/creature_cast.py); the
        cast records it as "creature/<id>", once for the book. A creature the library has no model for is not
        made here (reported)."""
        if self._built(name):
            return True
        pr = self.person(name)
        w = self.work / "cast" / pr["slug"]
        w.mkdir(parents=True, exist_ok=True)
        out = w / "creature_cast.json"
        out.unlink(missing_ok=True)
        llm(f"creature_{pr['slug']}", w / "creature_cast.log", [[
            PY, HERE / "characters" / "creature_cast.py", "--look", pr["look"], "--out", out, "--cache", w / "cache"]])
        got = json.loads(out.read_text()).get("asset") if out.exists() else None
        if not got or CRE.rig(C.LIBRARY, got) is None:
            log(f"{place}: the library has no rigged creature made for {name}'s look (see {out}); making one is "
                f"not part of this flow yet")
            return False
        self.cast[name] = got
        self.cast_p.write_text(json.dumps(self.cast, indent=1))
        (w / "FAILED").unlink(missing_ok=True)             # (a figure build tried before the creature route)
        log(f"{place}: {name} is cast as the library's {got}")
        return True

    def step_cast(self, place):
        """Each person's figure (made once for the book), then their seated and lying stances for this set."""
        d = self.d(place)
        people = self.people_of(place)
        poses = {n: set() for n in people}
        for i in self.shots_of(place):
            for p in self.plan[i][0].get("people") or []:
                if p["name"] in poses:
                    poses[p["name"]].add(str(p.get("pose") or ""))
        creat = {n for n in people if self.person(n).get("creature")}
        for name in people:                                   # the figures
            if name in creat:                                 # (a creature: a library creature, not a figure)
                self._cast_creature(place, name)
                continue
            fid, w, base = self._cast_cmd(name)
            if self._built(name) and self._sheet_newer(name):
                # (a figure made from a sheet that has since been drawn again is not that person any more)
                log(f"{name}: the sheet was drawn again after the figure was made; the figure is drawn and shaped again")
                self._fit_mark(w)
                if sh(base + ["--from", "draw", "--until", "seats"], w / "run.log"):
                    log(f"{name}: the figure could not be made again; see {w / 'run.log'}")
                continue
            if self._built(name):
                self._hands_again(name, w, base)
                continue
            if (w / "FAILED").exists():
                log(f"{place}: {name}'s figure failed its build before ({w / 'FAILED'})")
                continue
            self._fit_mark(w)                    # (built now with every fitting rule)
            if sh(base + ["--until", "seats"], w / "run.log") and not self._built(name):
                sh(base + ["--from", "draw", "--until", "seats"], w / "run.log")          # (a new round of drawings)
            if not self._built(name):
                log(f"{place}: no figure for {name} ({fid}); see {w / 'run.log'}")
                (w / "FAILED").write_text(time.strftime("%F %T"))      # (a build is long: not tried again unasked)
        # someone whose own figure could not be built is staged with a stand-in: the built figure of another of
        # the book's people of the same sex (this place's first), at their own height (scenes/stature.py); the
        # redraw draws them from their own sheet and words, so only the pose and the place are the stand-in's
        fem = lambda n: self.person(n)["noun"] in ("woman", "girl")                  # noqa: E731
        # (a creature stands in for nobody and has no stand-in)
        built = sorted((n for n in self.cast if self._built(n) and not CRE.is_id(self.cast[n])),
                       key=lambda m: m not in people)
        out, need = {}, {}
        for k, n in enumerate(people):
            own = n if self._built(n) else None if n in creat else next((m for m in built if fem(m) == fem(n)), None)
            if not own:
                log(f"{place}: {n} has no figure and no stand-in; staged without them")
                continue
            out[n] = {"key": "abcdefgh"[k], "id": self.cast[own], **({"stand_in": own} if own != n else {})}
            if n in creat:
                out[n]["creature"] = True
                continue
            need.setdefault(own, set()).update(poses[n])
            if own != n:
                log(f"{place}: {n} is staged with {own}'s figure as a stand-in")
        for own, ps in need.items():                         # the stances this set asks of each figure
            fid, w, base = self._cast_cmd(own)
            lie = []
            if any("ly" in q or "lie" in q for q in ps):                    # lying stances on this set's own beds
                sys.path.insert(0, str(HERE / "scenes"))
                import lying
                beds = sorted({b["asset"] for b in lying.beds(C.LIBRARY, self.set_dir(place)).values()})
                lie = ["--lie", "--lie-beds"] + beds if beds else []
            if any("sit" in q or "seat" in q for q in ps) or lie:
                rc = sh(base + ["--from", "seats", "--set", self.places[place]["set"]] + lie,
                        w / f"stances_{self.places[place]['slug']}.log")
                if rc:
                    log(f"{place}: {own}'s stances for this set did not all build (rc {rc})")
        # the book's change of clothes: each outfit worn here as its own figure (a variant of the person's), the
        # wardrobe on the chapter's clock for the staging pass, and each outfit's words and sheet for the redraw
        for n in people:
            if n in creat or n not in out or out[n].get("stand_in") or not self.wardrobe(n):
                continue
            vs = self._cast_variants(place, n)
            out[n]["wardrobe"] = [{k: r[k] for k in ("t", "variant", "id", "dress")} for r in self.wardrobe(n)]
            worn = {}
            for v, (vid, r) in vs.items():
                ow = self.outfit_words(n, r)
                worn[v] = {"wear_set": ow["wear"], "wear_back_set": ow["wear_back"], "worn_sheet": ow["sheet"],
                           # (only a figure really made wearing it: else the thing is still set on its head)
                           **({"head_on": True} if ow.get("head_on") and (
                               self.work / "cast" / self.person(n)["slug"] / r["variant"] / "headwear_drawn").exists() else {})}
                if not vid:
                    continue
                ps = {str(p.get("pose") or "") for i in self.shots_of(place) if (self.outfit(n, i) or {}).get("variant") == v
                      for p in self.plan[i][0].get("people") or [] if p["name"] == n}
                self._stances(place, vid, ps, self._variant_cmd(n, r)[1:])
            out[n]["worn"] = worn
        (d / "cast.json").write_text(json.dumps(out, indent=1))
        return bool(out)

    def _stances(self, place, fid, ps, wb):
        """A figure's seated and lying stances for this set, by the poses the place's shots ask of it."""
        w, base = wb
        lie = []
        if any("ly" in q or "lie" in q for q in ps):
            import lying
            beds = sorted({b["asset"] for b in lying.beds(C.LIBRARY, self.set_dir(place)).values()})
            lie = ["--lie", "--lie-beds"] + beds if beds else []
        if any("sit" in q or "seat" in q for q in ps) or lie:
            rc = sh(base + ["--from", "seats", "--set", self.places[place]["set"]] + lie,
                    w / f"stances_{self.places[place]['slug']}.log")
            if rc:
                log(f"{place}: {fid}'s stances for this set did not all build (rc {rc})")

    def step_pass(self, place):
        d = self.d(place)
        (d / "pass.json").unlink(missing_ok=True)
        llm(f"pass_{self.places[place]['slug']}", d / "pass.log", [[
            PY, HERE / "scenes" / "staging_pass.py", self.plan_p, d / "shots.txt", d / "cast.json", self.timeline,
            self.set_dir(place), d / "pass.json", "--cache", d / "cache_pass", "--looks", self.look_p]])
        if not (d / "pass.json").exists():
            return False
        shutil.rmtree(d / "specs", ignore_errors=True)
        sh([PY, HERE / "scenes" / "from_plan.py", self.plan_p, d / "shots.txt", d / "cast.json", d / "specs",
            "--pass", d / "pass.json", "--looks", self.look_p, "--carried", self.carried(place)], d / "from_plan.log")
        return bool(list((d / "specs").glob("pl_g*.json")))

    def carried(self, place) -> Path:
        """What people wear on the head or carry in a hand in this place's shots (the pass's "things"), each made
        once as a library object from its own words, the book's look of it added when the book names it, and as
        long as the director says. -> the file {the thing's words: library prop id} (only what was made)."""
        d = self.d(place)
        things, said = {}, {}                # (said: the director's look of the thing alone, from the story)
        for row in (json.loads((d / "pass.json").read_text()).get("things") or {}).values():
            for ts in row.values():
                for t in ts:
                    if t["on"] in ("head", "held", "worked") and t.get("size_cm"):
                        things.setdefault(t["what"], (t["size_cm"], t.get("goes_on")))
                        if t.get("look"):
                            said.setdefault(t["what"], str(t["look"]).strip().rstrip("."))
        props = self.look.get("props") or {}
        sys.path.insert(0, str(HERE / "sets"))
        from place_build import no_likeness
        def look_of(what):
            w = set(re.findall(r"[a-z]+", what.lower()))
            # the book's look of this very thing: every word of the thing's name is in the prop's (a "bellows handle"
            # took the look of the bellows and was made a whole bellows a hand long)
            best = max(props, key=lambda n: len(w & set(re.findall(r"[a-z]+", n.lower()))), default=None)
            return str(props[best]).strip().rstrip(".") if best and w and w <= set(re.findall(r"[a-z]+", best.lower())) else ""
        # (a thing that goes on the head is made as it is worn with the face in view: the camera shows faces)
        pcs = [{"id": f"h{n + 1}", "kind": "hand_thing", "longest_m": round(cm / 100.0, 3),
                # (where the book says how it looks, that is what is drawn: a story's own name for a thing draws its
                # words, a "star-iron" bar came out a star)
                # (and never what it is likened to: its size is its own number - sets/place_build.no_likeness, which
                # the set's pieces had and these did not: the stone on the anvil was made a sleeping animal, 11-20)
                "look": (no_likeness(said.get(what) or look_of(what) or "") or what) + ALONE
                + (HEAD_OPEN if on in ("head", "face") else "")}
               for n, (what, (cm, on)) in enumerate(things.items())]
        out = d / "carried.json"
        out.write_text("{}")
        if not pcs:
            return out
        (d / "carried_spec.json").write_text(json.dumps({"pieces": pcs}, indent=1))
        sh([PY, HERE / "places" / "generate.py", d / "carried_spec.json"], d / "carried.log")
        sys.path.insert(0, str(HERE / "places"))
        import pieces as P
        res = P.resolve(pcs, P.load_records(C.LIBRARY))
        made = {what: r["asset_id"] for what, r in zip(things, res["pieces"]) if r["status"] == "library"}
        out.write_text(json.dumps(made, indent=1))
        log(f"{place}: {len(made)} of {len(pcs)} worn or carried things are made: {sorted(made)}")
        return out

    def groups_of(self, place, shots) -> list:
        """The scene groups (spec file stems) that hold these shots."""
        out = []
        for f in sorted((self.d(place) / "specs").glob("pl_g*.json")):
            if any(c.get("plan_id") in shots for c in json.loads(f.read_text()).get("cameras") or []):
                out.append(f.stem)
        return out

    @staticmethod
    def group_dirs(seq: Path, stem: str) -> list:
        """A group's staged folders: its own and its per-speaker parts (pl_g05, pl_g05_a, pl_g05_q)."""
        return sorted(q for q in seq.glob(f"{stem}*") if q.is_dir() and (q.name == stem or q.name.startswith(stem + "_")))

    def step_stage(self, place, only=None, fallback=()):
        """only: the groups to stage again (the others' stagings are kept); fallback: the shots that take the
        clear wide when their tries run out (speakers.give_up)."""
        d = self.d(place)
        if only is None:
            shutil.rmtree(d / "seq", ignore_errors=True)
        (d / "seq").mkdir(exist_ok=True)      # (a stopped run's last scene may still be writing into it)
        env = {"SCENES_OUT": str(d / "seq"), "SCENES_CHAR_TOOLS": str(HERE / "characters")}
        if fallback:
            env["SPEAKERS_FALLBACK_ALL"] = ",".join(fallback)
        prev = None
        for f in sorted((d / "specs").glob("pl_g*.json")):
            if only is not None and f.stem not in only:
                if (d / "seq" / f.stem / "staged.json").exists():
                    prev = d / "seq" / f.stem / "staged.json"
                continue
            if only is not None:
                for q in self.group_dirs(d / "seq", f.stem):
                    shutil.rmtree(q, ignore_errors=True)
                for q in (d / "seq").glob(f"{f.stem}*.json"):
                    q.unlink()
            if not json.loads(f.read_text()).get("characters"):
                # an insert (nobody of the cast in it: the place or a thing seen alone): the set photographed from
                # its placed camera, no people's checks to make (from_plan.insert_camera)
                (d / "seq" / f.stem).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / "seq" / f.stem / "spec_round0.json")
                if sh([PY, HERE / "scenes" / "scene.py", f], d / "seq" / f"{f.stem}.scene.log", env, limit_s=GROUP_MAX_S):
                    log(f"{place}: insert {f.stem} failed its staging")
                continue
            rc = sh([PY, HERE / "scenes" / "speakers.py", f] + (["--prev", prev] if prev else []),
                    d / "seq" / f"{f.stem}.speakers.log", env, limit_s=GROUP_MAX_S)
            if rc == 124:
                log(f"{place}: scene {f.stem} STOPPED after {GROUP_MAX_S / 60:.0f} min of staging (its usual: a minute or two)")
            elif rc:
                log(f"{place}: scene {f.stem} failed its staging")
            if (d / "seq" / f.stem / "staged.json").exists():
                prev = d / "seq" / f.stem / "staged.json"
        return bool(list((d / "seq").glob("*/result.json")))

    def step_redraw(self, place, only=None):
        """One place's keys drawn (draw_phase for it alone). only: the groups whose shots are drawn again; their
        rows and pictures replace those groups' in the place's run, the other shots' keys kept as they are."""
        return self.draw_phase({place: only}).get(place, False)

    def redraw_prepare(self, place, only=None):
        """A place's drawing jobs written (its guides, then redraw.py's job file). -> its run's facts {"out", "rp",
        "only"}; None when it has no scene to draw; False when a step failed."""
        d = self.d(place)
        out = d / "run" if only is None else d / "run_again"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        cast = json.loads((d / "cast.json").read_text())
        ss = json.loads((self.set_dir(place) / "spec.json").read_text())
        style = (self.book / "art_style.txt")
        run = dict(REDRAW, cast={v["key"]: dict(self.person(n)["words"], **({"stand_in": True} if v.get("stand_in") else {}))
                                 for n, v in cast.items()},
                   scenes=sorted(os.path.dirname(p) for p in glob.glob(str(d / "seq" / "*" / "result.json"))
                                 if only is None or any(Path(p).parent in self.group_dirs(d / "seq", g) for g in only)),
                   set=str(self.set_dir(place)), out=str(out),
                   # (a cabin with views outside: its concept picture shows one view of its own, which the redraw
                   # then painted in every window; the cabin's look reaches the redraw through its pieces)
                   extra_refs=[ss["concept"]] if ss.get("concept") and not (ss.get("views") or {}).get("words") else [],
                   look=key_style(style.read_text(), open_air=bool(ss.get("world")))
                   if style.exists() else "")
        # a room's own walls and pieces are drawn ones: its background is held nearer the engine's frame than a
        # place in the open, whose ground and far land are rough. At the open's share a forge's keys moved its
        # hearth and door and hung lanterns the set does not have (11-18, seven shots at three strengths)
        if not ss.get("world"):
            run.update(strength_bg=ROOM_BG, mask_people="heads")   # (redraw/guides.py HEADS_*: these two strengths)
        if not run["scenes"]:
            return None
        rp = d / ("run.json" if only is None else "run_again.json")
        rp.write_text(json.dumps(run, indent=1))
        tag = self.places[place]["slug"] + ("" if only is None else "_again")
        root = HERE.parents[1]
        if line("blender", "guides", f"guides_{tag}", out / "guides.log",
                ["bash", "-c", f"cd {root} && bash {CARD_SH} 05 {GUIDES_PY} -I tools/assets/redraw/guides.py --set "
                               f"{self.set_dir(place)} {' '.join(run['scenes'])}"]):
            return False
        if sh([PY, HERE / "redraw" / "redraw.py", rp], out / "redraw.log", cwd=root):
            return False
        return {"out": out, "rp": rp, "only": only}

    def draw_phase(self, want: dict) -> dict:
        """The keys of several places drawn together, after their staging: want = {place: None (every shot) or the
        scene groups to draw again}. Each place's jobs are written, then every place's pictures are drawn as one
        batch shared between the drawing cards, checked, the failed ones drawn and checked again, and each place's
        best drawings picked. -> {place: whether it has its drawings}. (Drawn place by place on one card beside
        the engine, a place's redraw was 10 to 20 minutes and two places waited on one card, 11-19)"""
        root, acc = HERE.parents[1], [PY, HERE / "redraw" / "accept.py"]
        res, ctx = {}, {}
        for place, only in want.items():
            c = self.redraw_prepare(place, only)
            if c:
                ctx[place] = c
            else:
                res[place] = (only is not None) if c is None else False
        if not ctx:
            return res
        logd = self.work
        tag = "keys_" + time.strftime("%H%M%S")
        cs = list(ctx.values())
        checks = lambda c, name, *more: sh(acc + ["checks", c["rp"], c["out"] / name, *more, "--pose-model", POSE_MODEL],   # noqa: E731
                                           c["out"] / "accept.log", cwd=root)

        def retries() -> list:
            again = []
            for c in cs:
                sh(acc + ["retry", c["rp"], c["out"] / "check_result.json", c["out"] / "retry_jobs.jsonl"],
                   c["out"] / "accept.log", cwd=root)
                rj = c["out"] / "retry_jobs.jsonl"
                if rj.exists() and rj.stat().st_size:
                    again.append(c)
                    checks(c, "check_x.json", "--ids-from", rj)
            return again
        for c in cs:                         # (what each picture is checked for is known before it is drawn)
            checks(c, "check.json")
        first = rows_of([c["out"] / "image_jobs.jsonl" for c in cs])
        log(f"drawing: {len(first)} picture(s) of {len(cs)} place(s) on {len(DRAW_CARDS)} card(s), checked on "
            f"{'the iGPU as they land' if CHECK_ON == 'igpu' else 'the cards after'}")
        beside = CHECK_ON == "igpu"
        if beside:
            dr = Drawers(f"draw_{tag}", logd)
            dr.feed(first)
            if check_beside([c["out"] / "check.json" for c in cs], f"check_{tag}", logd, dr.alive):
                again = retries()
                if again:
                    rows = rows_of([c["out"] / "retry_jobs.jsonl" for c in again])
                    log(f"drawing: {len(rows)} picture(s) drawn again")
                    dr.feed(rows)
                    check_beside([c["out"] / "check_x.json" for c in again], f"checkx_{tag}", logd, dr.alive)
                if not dr.close():
                    log("drawing: a card's drawer ended badly")
            else:
                # the iGPU's check did not run: the drawers finish what they have, and the checks go to the cards
                log("drawing: the check beside the drawing FAILED; checking on the cards after it")
                dr.close()
                beside = False
                if not check_files([c["out"] / "check.json" for c in cs], f"check_{tag}", logd):
                    log("drawing: a check job FAILED")
                    return dict(res, **{p: False for p in ctx})
                again = retries()
                if again:
                    draw_jobs([c["out"] / "retry_jobs.jsonl" for c in again], f"redraw_{tag}", logd)
                    check_files([c["out"] / "check_x.json" for c in again], f"checkx_{tag}", logd)
        else:
            if not draw_jobs([c["out"] / "image_jobs.jsonl" for c in cs], f"draw_{tag}", logd):
                log("drawing: a card's job FAILED")
                return dict(res, **{p: False for p in ctx})
            if not check_files([c["out"] / "check.json" for c in cs], f"check_{tag}", logd):
                log("drawing: a check job FAILED")
                return dict(res, **{p: False for p in ctx})
            again = retries()
            if again:
                draw_jobs([c["out"] / "retry_jobs.jsonl" for c in again], f"redraw_{tag}", logd)
                check_files([c["out"] / "check_x.json" for c in again], f"checkx_{tag}", logd)
        for place, c in ctx.items():
            out, only = c["out"], c["only"]
            sh(acc + ["pick", c["rp"], out / "check_result.json", out / "check_x_result.json", out / "accepted.json"],
               out / "accept.log", cwd=root)
            if only is None or not (out / "accepted.json").exists():
                res[place] = (out / "accepted.json").exists()
            else:
                d = self.d(place)
                res[place] = self.merge_run(d / "run", out, [q.name for g in only for q in self.group_dirs(d / "seq", g)] + list(only))
        return res

    @staticmethod
    def merge_run(run: Path, again: Path, groups: list) -> bool:
        """The redrawn groups' rows and pictures take those groups' places in the run: every old row and try of
        theirs goes, the new ones come in with their paths in the run's own folder."""
        of = lambda name: any(name.startswith(g + "__") for g in groups)       # noqa: E731
        run.mkdir(exist_ok=True)
        acc = json.loads((run / "accepted.json").read_text()) if (run / "accepted.json").exists() else {}
        acc = {k: v for k, v in acc.items() if not of(k)}
        for q in run.glob("*.png"):
            if of(q.name):
                q.unlink()
        for q in again.glob("*.png"):
            shutil.copy2(q, run / q.name)
        new = json.loads((again / "accepted.json").read_text())
        for k, v in new.items():
            v = dict(v)
            if v.get("path"):
                v["path"] = str(run / Path(v["path"]).name)
            acc[k] = v
        (run / "accepted.json").write_text(json.dumps(acc, indent=1))
        return True

    def step_keys(self, place):
        d = self.d(place)
        acc = json.loads((d / "run" / "accepted.json").read_text())
        # tries the pick rejected outright (redraw/facegate.py: a person whose face shows drawn as a creature, a
        # statue or no face) are never a key, not even as the fallback or the replacement of a dark pick
        rejected = {t for v in acc.values() for t in v.get("rejected") or []}
        rows = []
        for g in sorted((d / "seq").glob("pl_g*/"), key=lambda p: (len(p.name), p.name)):
            if not (g / "spec_round0.json").exists():
                continue
            sj = json.loads((g / "speakers.json").read_text()) if (g / "speakers.json").exists() else {}
            for c in json.loads((g / "spec_round0.json").read_text())["cameras"]:
                if not (g / f"{c['name']}.png").exists():
                    rows.append({"shot": c["plan_id"], "key": False, "why": "not staged"})
                    continue
                st = (sj.get("shots") or {}).get(c["name"]) or {}
                # every shot ends with an engine key (the user's rule, 11-17): a staging that failed its own checks
                # gives the least-bad try speakers.py kept, flagged with what it missed; a redraw that failed its
                # checks gives its best drawing, flagged; with no drawing at all, the engine still itself, flagged
                miss = staging_miss(st)
                a = next((v for k, v in acc.items() if k.startswith(f"{g.name}__{c['name']}__")), None)
                flag = {"staging_miss": miss} if miss else {}
                if a and a.get("no_key"):
                    # every try failed the face gate: no key, said in plain words (keyless(): restaged)
                    rows.append({"shot": c["plan_id"], "key": False, "why": a["no_key"], **flag})
                    continue
                if not a:
                    tries = sorted((q for q in (d / "run").glob(f"{g.name}__{c['name']}__*.png") if q.stem not in rejected),
                                   key=lambda q: ("__x" in q.stem, q.name))
                    if tries:
                        a = {"path": str(tries[0])}
                        flag["redraw_miss"] = "no drawing passed its checks"
                if a:
                    # a picture nearly all black is no key (a set whose lights were lost staged 20 "passing" shots
                    # at 3 to 8 of 255, and the video model invented other rooms from them; the darkest good keys
                    # of the same night scene stand at 15, 11-14)
                    from PIL import Image
                    lum_of = lambda q: sum(Image.open(q).convert("L").resize((64, 36)).get_flattened_data()) / (64 * 36)  # noqa: E731
                    lum = lum_of(a["path"])
                    if lum < KEY_DARK:
                        # nearly black: the shot's next picture that is not (another drawing of it, then the
                        # staging's other tries and its engine still), flagged; none at all: no key
                        # (another drawing only: a raw engine still is never a key, 11-17)
                        alt = [q for q in sorted((d / "run").glob(f"{g.name}__{c['name']}__*.png"))
                               if q.exists() and str(q) != a["path"] and q.stem not in rejected
                               and lum_of(q) >= KEY_DARK]
                        if not alt:
                            rows.append({"shot": c["plan_id"], "key": False,
                                         "why": f"the picture is nearly black (mean {lum:.1f} of 255)"})
                            continue
                        a = {"path": str(alt[0])}
                        flag["dark_replaced"] = f"the pick was nearly black ({lum:.1f}): {alt[0].name}"
                rows.append({"shot": c["plan_id"], "key": bool(a), "why": (a or {}).get("fails") or "", **flag})
                if not a:
                    continue
                shutil.copy2(a["path"], self.keys / f"{c['plan_id']}_s0.png")
                sh([PY, HERE / "scenes" / "shot_people.py", g, c["name"], self.look_p, d / "cast.json",
                    self.keys / c["plan_id"]], d / "keys.log")
        staged = {r["shot"] for r in rows}
        rows += [{"shot": i, "key": False, "why": "no scene spec"} for i in self.shots_of(place) if i not in staged]
        for r in rows:                       # a shot with no key this time has none: an earlier run's picture left
            if not r["key"]:                 # in the folder is of another staging (eleven such went into a cut and
                (self.keys / f"{r['shot']}_s0.png").unlink(missing_ok=True)      # the count called them keys, 11-14)
        (d / "keys.json").write_text(json.dumps(rows, indent=1))
        crowd_facts(self.keys, [r["shot"] for r in rows if r["key"]], self.set_dir(place))
        log(f"{place}: {sum(r['key'] for r in rows)} of {len(self.shots_of(place))} shots have a key")
        return True

    UNREADABLE = ("is not in the picture", "is cut by the frame's edge", "the camera is outside the room",
                  "the camera is inside", "the camera is too near", "the still is nearly black")
    # (a kept least-bad staging with these misses shows nobody readable; the last two: speakers.still_why and
    # creature_view_why, a camera against or under a creature's head)

    def shows_nobody(self, sid) -> bool:
        """Whether a shot whose plan lists people has a key the engine measured nobody in (each of them under a
        hundredth of the frame, or out of it): the shot flow refuses such a key, and its shot kept an old clip while
        the rest of the story was rendered anew (11-21)."""
        try:
            ff = self.keys / f"{sid}.shot_facts.json"
            return ff.exists() and not json.loads(ff.read_text()).get("visible") and \
                bool((self.plan.get(sid) or [{}])[0].get("people"))
        except (AttributeError, OSError, ValueError):       # (no keys folder or plan at hand: nothing is known)
            return False

    def keyless(self, place) -> list:
        """The shots without a key, and those whose key is a least-bad staging that left its people unreadable (the
        subject or a named person out of the picture, the camera inside something): both are staged again."""
        kp = self.d(place) / "keys.json"
        out = []
        for r in json.loads(kp.read_text()) if kp.exists() else []:
            miss = " ".join(map(str, r.get("staging_miss") or []))
            walled = any(int(x) >= 50 for x in re.findall(r"foreground covers (\d+)%", miss))
            # (and one where another's head is twice the head of whom the shot is about: it reads as theirs)
            lost = any(float(o) >= 2.0 * float(m_) > 0 for o, m_ in
                       re.findall(r"head \(([\d.]+)\) is bigger than the person the shot is about's \(([\d.]+)\)", miss))
            if not r["key"] or walled or lost or any(w in miss for w in self.UNREADABLE) or self.shows_nobody(r["shot"]):
                out.append(r["shot"])
        return out

    def restage_keyless(self, place) -> None:
        """Every shot ends with an engine key (the user's rule, 11-17): the shots a place's keys step left without
        one are staged again, once, each taking a clear wide on its people when its tries run out (speakers.py
        SPEAKERS_FALLBACK_ALL), then drawn and keyed again. Only their scene groups: the other shots keep their
        stagings and keys (staged and drawn again whole for one shot, a place took a quarter of an hour and two of
        its keyed talk shots lost their keys, 11-19)."""
        miss = self.keyless(place)
        if not miss:
            return
        groups = self.groups_of(place, set(miss))
        log(f"{place}: {len(miss)} shot(s) without a key {miss}: scene(s) {groups} staged again with the clear-wide fallback")
        if not groups:
            return
        for st, call in (("stage", lambda: self.step_stage(place, only=groups, fallback=miss)),
                         ("redraw", lambda: self.step_redraw(place, only=groups)),
                         ("keys", lambda: self.step_keys(place))):
            if not call():
                log(f"{place}: the restage's {st} FAILED")
                return
        log(f"{place}: after the restage, {len(self.keyless(place))} shot(s) without a key")

    def preflight(self, place) -> list:
        """What can be seen wrong in a place's scene specs and words before any of it is staged or drawn
        (scenes/preflight.py): told in the log and kept in the place's preflight.json. Nothing is stopped."""
        d = self.d(place)
        try:
            sys.path.insert(0, str(HERE / "scenes"))
            import preflight as PF
            found = PF.check(d / "specs", self.set_dir(place), d / "pass.json",
                             {p["name"] for i in self.ids for p in self.plan[i][0].get("people") or []})
        except Exception as e:                             # (a check that cannot run stops nothing)
            found = [f"the preflight could not run: {e!r}"]
        (d / "preflight.json").write_text(json.dumps(found, indent=1))
        for w in found:
            log(f"{place}: preflight: {w}")
        return found

    def _room_picture(self, place, rows) -> list:
        """The picture a drawn key takes its room from: one of the place's own engine keys, the widest there is (a
        full shot before a medium before a close), so a drawn shot stands in the room the shots around it show; with
        no engine key, the place's concept picture. (Drawn from the concept, the drawn shots of a built room were
        in another room than its engine keys: darker walls, other bunks, 11-15.)"""
        d = self.d(place)
        have = {r["shot"] for r in rows if r["key"] and not r.get("drawn")}
        rank = {"wide": 0, "full": 1, "medium": 2, "close": 3}
        best = None
        for sp in sorted((d / "seq").glob("pl_g*/spec_round0.json")):
            for c in json.loads(sp.read_text()).get("cameras") or []:
                k = (rank.get(c.get("kind"), 9), -len(c.get("show") or []))
                if c.get("plan_id") in have and (self.keys / f"{c['plan_id']}_s0.png").exists() and \
                        (best is None or k < best[0]):
                    best = (k, self.keys / f"{c['plan_id']}_s0.png")
        if best:
            return [str(best[1])]
        return [str(d / "concept.png")] if (d / "concept.png").exists() else []

    def step_drawn(self, place):
        """The shots the engine has no key for are drawn straight from the plan's own picture of them, the people
        from their sheets and LoRAs: every shot of a place the story only glances at (DRAWN_MAX_SHOTS or fewer: a
        memory, a thing imagined, never built), and in a built place each shot whose staging failed its checks (two
        people in one bunk; a cut showed black there). A drawn picture is a key when it shows as many people as the
        plan puts in it (the first of DRAWN_SEEDS that does); the room of a built place is that of its engine keys
        (_room_picture)."""
        d = self.d(place)
        kp = d / "keys.json"
        rows = [r for r in (json.loads(kp.read_text()) if kp.exists() else []) if not r.get("drawn")]
        todo = [r["shot"] for r in rows if not r["key"]] if rows else list(self.shots_of(place))
        if not todo:
            return True
        out = d / "drawn"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        style = self.book / "art_style.txt"
        look = (style.read_text().strip().rstrip(".") + ".") if style.exists() else ""
        room = self._room_picture(place, rows) if rows else []
        jobs, facts, vis_of = [], {}, {}
        for i in todo:
            s = self.plan[i][0]
            ppl = [q for q in s.get("people") or [] if q["name"] in self.look["characters"]]
            words = str(s.get("keyshot") or s.get("picture") or s.get("action") or "").strip()
            vis = []
            for q in ppl:
                w = self.words_at(q["name"], i)          # (in the outfit the book's wardrobe has them in now)
                said = (f"the {w['noun']} in {WD.outer_garment(w['wear']) or w['wear'].split(',')[0]}" if w["wear"]
                        else f"the {w['noun']}")
                words = words.replace(q["name"], said)
                box = SIDE_BOX.get(str(q.get("side") or "").lower(), SIDE_BOX["center"])
                vis.append({"name": q["name"], "look": said, "seen": "front", "box": box, "_w": w})
            vis.sort(key=lambda v: v["box"][0])
            n = len(vis)
            # (with a creature among them they are counted as figures: told "one person", a picture of the
            # creature alone would get a person added)
            fig = any(v["_w"].get("creature") for v in vis)
            only = ("Nobody is in the picture." if not n else
                    f"Exactly {n} {('figure is' if n == 1 else 'figures are') if fig else ('person is' if n == 1 else 'people are')} "
                    f"in the picture, each of them once.")
            where = (" The room is the one of the last reference picture: its walls, floor, furniture and light; "
                     "whoever stands in that picture is not taken over from it.") if room else ""
            # (a sheet shows its person twice, near and whole, and a picture of one person alone copied both: three
            # of three drawings had the child twice, told in words or not. One person alone is drawn from their
            # LoRA; with others in the picture the sheets tell them apart)
            sheets = [v["_w"]["sheet"] for v in vis] if n > 1 else []
            for seed in DRAWN_SEEDS:
                jobs.append({"id": f"{i}__{seed}", "seed": seed, "path": str(out / f"{i}__{seed}.png"),
                             "prompt": f"A photographed frame from a film. {words} {only}{where} {look}".strip(),
                             "size": [1152, 640], "refs": sheets + room,
                             "loras": [[v["_w"]["lora"], 0.8, [0.0, 0.0, 1.0, 1.0] if n == 1 else
                                        [v["box"][0], 0.0, v["box"][2], 1.0]] for v in vis]})
            facts[i] = [{k: v[k] for k in v if k != "_w"} for v in vis]
            vis_of[i] = vis
        (out / "image_jobs.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
        tag = self.places[place]["slug"]
        if line("image", "refs", f"drawn_{tag}", out / "run.log", ["bash", IMAGE_SH, out / "image_jobs.jsonl"]):
            return False
        # (a creature of the cast is one of the picture's people too, found by its own noun)
        labels = ["person"] + sorted({v["_w"]["noun"] for vs in vis_of.values() for v in vs if v["_w"].get("creature")})
        (out / "count.json").write_text(json.dumps({"out": str(out / "people.json"), "items": [
            {"id": j["id"], "path": j["path"], "labels": labels} for j in jobs]}))
        if line("image", "check", f"count_{tag}", out / "count.log", ["bash", COUNT_SH, out / "count.json"]):
            return False
        seen = json.loads((out / "people.json").read_text())
        from PIL import Image
        drawn = []
        for i in todo:
            pick, why = None, "no drawing shows the plan's people, each once"
            for seed in DRAWN_SEEDS:
                pic, bx = out / f"{i}__{seed}.png", (seen.get(f"{i}__{seed}") or {}).get("boxes")
                if not pic.exists() or bx is None or len(bx) != len(facts[i]):
                    continue
                if sum(Image.open(pic).convert("L").resize((64, 36)).getdata()) / (64 * 36) < KEY_DARK:
                    why = "the drawing is nearly black"
                    continue
                pick = (pic, bx)
                break
            drawn.append({"shot": i, "key": bool(pick), "why": "" if pick else why, "drawn": True})
            if not pick:
                (self.keys / f"{i}_s0.png").unlink(missing_ok=True)
                continue
            shutil.copy2(pick[0], self.keys / f"{i}_s0.png")
            vis = [dict(v, box=[round(x, 4) for x in b], area=round((b[2] - b[0]) * (b[3] - b[1]), 4))
                   for v, b in zip(facts[i], pick[1])]                     # (left to right, as the plan's sides)
            (self.keys / f"{i}.shot_facts.json").write_text(json.dumps(
                {"camera": "drawn", "visible": vis, "count": len(vis), "move": None}, indent=1))
            n = len(vis)
            (self.keys / f"{i}.shot_facts.txt").write_text(
                f"- In the picture: {n} {'person' if n == 1 else 'people'}"
                + (": " + "; ".join(v["look"] for v in vis) if vis else "") + ".\n")
        got = {r["shot"]: r for r in drawn}
        kp.write_text(json.dumps([got.pop(r["shot"], r) for r in rows] + list(got.values()), indent=1))
        log(f"{place}: {sum(r['key'] for r in drawn)} of {len(drawn)} shots without an engine key have a drawn one")
        return True

    # ------------------------------------------------------------ the run
    # the steps that write the library's shared things, one place at a time within each kind: the pieces (by their
    # look), and the figures and carried things. (Under one lock for all three, a place waited to cast its figures
    # until another's quarter-hour of pieces was done, 11-19)
    SHARED_STEPS = {"pieces": "pieces", "cast": "people", "pass": "people"}

    def run_place(self, place, todo, redo) -> None:
        if 0 < len(self.shots_of(place)) <= DRAWN_MAX_SHOTS:         # a glance at a place: drawn, not built
            if not self.done(place, "drawn") or "drawn" in redo:
                log(f"{place}: drawn")
                llm_down()
                if self.step_drawn(place):
                    self.mark(place, "drawn")
                else:
                    log(f"{place}: drawn FAILED")
            return
        import contextlib
        sl = self.places[place]["slug"]
        redo = {x for x in redo if ":" not in x} | {x.split(":", 1)[1] for x in redo if x.split(":", 1)[0] == sl}
        if any(st in redo and self.done(place, st) for st in todo):
            # (what is about to be made again is kept first: the way back, place_keep.restore)
            snap = PK.snapshot(self.d(place), self.keys, self.shots_of(place), self.set_dir(place))
            log(f"{place}: kept as it stood in {snap}")
        for st in todo:
            if self.done(place, st) and st not in redo:
                continue
            with (self._shared[self.SHARED_STEPS[st]] if st in self.SHARED_STEPS else contextlib.nullcontext()):
                log(f"{place}: {st}")
                if st not in ("spec", "pass"):
                    llm_down()
                t = time.time()
                with Watch(place, st, len(self.shots_of(place))):
                    ok = getattr(self, f"step_{st}")(place)
            if not ok:
                log(f"{place}: {st} FAILED after {time.time() - t:.0f} s; the place stops here")
                break
            self.mark(place, st)
            for later in STEPS[STEPS.index(st) + 1:]:           # what follows is made again from the new result
                (self.d(place) / f".done_{later}").unlink(missing_ok=True)
            log(f"{place}: {st} done in {time.time() - t:.0f} s")
            if st == "pass":
                self.preflight(place)
            if st == "keys":
                self.restage_keyless(place)

    def _redo(self, place, redo) -> set:
        sl = self.places[place]["slug"]
        return {x for x in redo if ":" not in x} | {x.split(":", 1)[1] for x in redo if x.split(":", 1)[0] == sl}

    def draw_all(self, places, steps, redo=()) -> None:
        """The drawing phase for the whole batch: the keys of every staged place drawn together (draw_phase), each
        place's keys picked, then the shots left without a key staged again in the engine, all of them, and drawn
        again together, once."""
        big = [p for p in places if not (0 < len(self.shots_of(p)) <= DRAWN_MAX_SHOTS) and self.done(p, "stage")]
        drawn = []
        if "redraw" in steps:
            need = [p for p in big if not self.done(p, "redraw") or "redraw" in self._redo(p, redo)]
            if need:
                log(f"drawing phase: {len(need)} place(s): {', '.join(need)}")
                llm_down()
                t = time.time()
                for p, ok in self.draw_phase({p: None for p in need}).items():
                    if not ok:
                        log(f"{p}: redraw FAILED; the place stops here")
                        continue
                    self.mark(p, "redraw")
                    (self.d(p) / ".done_keys").unlink(missing_ok=True)
                    drawn.append(p)
                log(f"drawing phase: done in {time.time() - t:.0f} s")
        if "keys" not in steps:
            return
        keyed = []
        for p in big:
            if self.done(p, "redraw") and (p in drawn or not self.done(p, "keys") or "keys" in self._redo(p, redo)):
                if self.step_keys(p):
                    self.mark(p, "keys")
                    keyed.append(p)
                else:
                    log(f"{p}: keys FAILED")
        # every shot ends with a key: the shots left without one, of every place, staged again (their scene groups
        # only, each taking a clear wide on its people when its tries run out), then drawn together and keyed
        again = {}
        for p in keyed:
            miss = self.keyless(p)
            groups = self.groups_of(p, set(miss)) if miss else []
            if groups:
                log(f"{p}: {len(miss)} shot(s) without a key {miss}: scene(s) {groups} staged again with the clear-wide fallback")
                if self.step_stage(p, only=groups, fallback=miss):
                    again[p] = groups
                else:
                    log(f"{p}: the restage's stage FAILED")
        if again:
            for p, ok in self.draw_phase(again).items():
                if ok and self.step_keys(p):
                    log(f"{p}: after the restage, {len(self.keyless(p))} shot(s) without a key")
                else:
                    log(f"{p}: the restage's drawing FAILED")

    def kept(self, what: str, names: set, tag: str | None = None) -> None:
        """--approve / --unapprove / --restore / --drift for the places named (all when none is)."""
        for place in [p for p in self.places if not names or p in names or self.places[p]["slug"] in names]:
            d = self.d(place)
            if what == "approve":
                log(f"{place}: approved, kept in {PK.approve(d, self.keys, self.shots_of(place), self.set_dir(place))}")
            elif what == "unapprove":
                PK.unapprove(d)
                log(f"{place}: no longer approved")
            elif what == "restore":
                log(f"{place}: put back from {PK.restore(d, self.keys, self.set_dir(place), tag)}")
            elif what == "drift" and PK.approved(d):
                # the planner alone, on the approved state's own inputs, as the rules now stand: no engine, no card
                snap = d / "history" / PK.approved(d)["snapshot"]
                now = d / "drift_specs"
                shutil.rmtree(now, ignore_errors=True)
                sh([PY, HERE / "scenes" / "from_plan.py", self.plan_p, snap / "shots.txt", snap / "cast.json", now,
                    "--pass", snap / "pass.json", "--looks", self.look_p, "--carried", snap / "carried.json"],
                   d / "drift.log")
                moved = PK.drift(snap / "specs", now)
                (d / "drift.json").write_text(json.dumps(moved, indent=1))
                log(f"{place}: " + (f"{len(moved)} thing(s) would change: " + "; ".join(moved[:12])
                                    + (" ..." if len(moved) > 12 else "") if moved else "nothing would change"))

    def run(self, only=None, first="shots", last="keys", redo=(), jobs=1, unlock=()):
        """jobs: how many places are worked at once. Each place's steps stay in order; their GPU work queues on the
        line by device, so with two places one draws on one card while the other stages or shapes on the other
        (one place at a time left the second card idle through a night's run, 11-19)."""
        import threading
        from concurrent.futures import ThreadPoolExecutor
        self._shared = {k: threading.Lock() for k in set(self.SHARED_STEPS.values())}
        todo = STEPS[STEPS.index(first):STEPS.index(last) + 1]
        places = [p for p in self.places if not only or p in only or self.places[p]["slug"] in only]
        # an approved place is left as it is: no step of it runs until it is named to --unlock (place_keep)
        for p in [p for p in places if PK.approved(self.d(p)) and p not in unlock and self.places[p]["slug"] not in unlock]:
            log(f"{p}: approved ({PK.approved(self.d(p))['snapshot']}): left as it is; --unlock {self.places[p]['slug']} "
                f"to make it again")
            places.remove(p)
        # two phases for the whole batch: every place staged in 3D first (the engine's card and the shaping card),
        # then every place's keys drawn together on both cards (draw_all). A place's own steps up to its staging:
        after = [x for x in todo if x in DRAW_STEPS]
        todo = [x for x in todo if x not in DRAW_STEPS]
        if jobs > 1 and len(places) > 1:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                for fu in [ex.submit(self.run_place, p, todo, redo) for p in places]:
                    try:
                        fu.result()
                    except Exception as e:                 # one place's failure does not stop the others
                        log(f"a place stopped on an error: {e!r}")
        else:
            for place in places:
                self.run_place(place, todo, redo)
        if after:
            self.draw_all(places, after, redo)
        report = {}
        for place in self.places:
            kp = self.d(place) / "keys.json"
            rows = json.loads(kp.read_text()) if kp.exists() else []
            have = sum(bool(r["key"]) and (self.keys / f"{r['shot']}_s0.png").exists() for r in rows)
            # (a key kept from a least-bad staging that shows nobody readable is a picture, not a good key: the
            # count said 34 of 34 with five such among them, 11-18)
            poor = [x for x in self.keyless(place) if x in {r["shot"] for r in rows if r["key"]}]
            report[place] = {"shots": len(self.shots_of(place)), "keys": have, "unreadable": poor,
                             "missing": [r for r in rows if not r["key"]]}
        (self.work / "report.json").write_text(json.dumps(report, indent=1))
        log("keys: " + "; ".join(f"{p} {v['keys']}/{v['shots']}" + (f" ({len(v['unreadable'])} unreadable)" if v["unreadable"] else "")
                                 for p, v in report.items()))
        return report


def main():
    a = sys.argv[1:]
    opt = lambda k, dflt=None: a[a.index(k) + 1] if k in a else dflt       # noqa: E731
    def many(k):
        out = []
        for x in (a[a.index(k) + 1:] if k in a else []):
            if x.startswith("--"):
                break
            out.append(x)
        return out
    s = Story(Path(a[0]), Path(a[1]), Path(a[2]), Path(a[3]), opt("--prefix", "st"))
    s.whole = "--metahuman" not in a           # figures as whole generated meshes on the skeleton (cast_char.py --whole),
    #                                      the default since 11-15; --metahuman casts MetaHuman bodies wearing generated outfits
    for what in ("approve", "unapprove", "restore", "drift"):
        if f"--{what}" in a:
            return s.kept(what, set(many(f"--{what}")), opt("--tag"))
    s.run(set(many("--only")), opt("--from", "shots"), opt("--until", "keys"), set(many("--redo")),
          jobs=int(opt("--jobs", "1")), unlock=set(many("--unlock")))


if __name__ == "__main__":
    main()
