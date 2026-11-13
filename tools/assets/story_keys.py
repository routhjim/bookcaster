"""A book's shot plan to engine-staged keyframes, every place of it, in one command.

    python tools/assets/story_keys.py PLAN.json BOOK_DIR WORK_DIR KEYS_DIR [--prefix ID] [--only PLACE ...]
                                      [--from STEP] [--until STEP] [--redo STEP ...]

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

All GPU work goes through the line (run_asset.py --line-cmd); nothing here decides how anything looks. WORK_DIR's
places.json ({place name: {"slug", "set"}}) and cast.json ({person: figure id}) are written on the first run and
read after (a place or a figure made earlier is named there and is not made again).
"""
from __future__ import annotations

import glob
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

STEPS = ["shots", "spec", "concept", "pieces", "set", "cast", "pass", "stage", "redraw", "keys"]
PY = sys.executable
RA = [PY, str(HERE / "run_asset.py")]
LAB = C.UE_HOME / "lab"
# the redraw's two card scripts and its pose model (the line's image station runs them)
IMAGE_SH = Path(os.environ.get("STORY_IMAGE_SH", LAB / "redraw4" / "figs" / "run_image.sh"))
CHECK_SH = Path(os.environ.get("STORY_CHECK_SH", LAB / "redraw5" / "check.sh"))
POSE_MODEL = Path(os.environ.get("STORY_POSE_MODEL", LAB / "redraw5" / "models" / "rtmw_x_256x192.pt"))
CARD_SH = HERE / "orbit" / "card.sh"
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
          "facts": True, "regions": True, "seeds": [7], "max_piece_refs": 2}


def log(m):
    print(f"[story {time.strftime('%H:%M:%S')}] {m}", flush=True)


def slug_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def sh(argv, logf: Path | None = None, env=None, cwd=None) -> int:
    e = dict(os.environ, ASSETS_LINE_CLIENT=str(HERE / "run_asset.py"), **(env or {}))
    if logf:
        with open(logf, "a") as f:
            return subprocess.run([str(a) for a in argv], stdout=f, stderr=subprocess.STDOUT, env=e, cwd=cwd).returncode
    return subprocess.run([str(a) for a in argv], env=e, cwd=cwd).returncode


def line(station: str, kind: str, jid: str, logf: Path, cmd: list, env=None) -> int:
    """One job on the line; a full line (75) is waited out."""
    while True:
        rc = sh(RA + ["--line-cmd", station, kind, "--id", jid, "--log", logf, "--"] + cmd, logf.with_suffix(".line.log"), env)
        if rc != 75:
            return rc
        time.sleep(30)


def llm(jid: str, logf: Path, cmds: list) -> int:
    """Commands that need the directing model, as one line job: the model is started for them and stopped after."""
    script = logf.with_suffix(".sh")
    lines = ["#!/bin/bash", f"cd {HERE.parents[1]}", "S=0",
             f"if ! systemctl --user is-active -q strata-eval; then {VENV_PY} -c \"{LLM_UP}\" || exit 1; S=1; fi", "RC=0"]
    lines += [" ".join(f"'{a}'" for a in map(str, c)) + " || RC=1" for c in cmds]
    lines += ["[ $S = 1 ] && systemctl --user stop strata-eval", "exit $RC"]
    script.write_text("\n".join(lines) + "\n")
    return line("igpu", "llm", jid, logf, ["bash", str(script)])


class Story:
    def __init__(self, plan_p: Path, book: Path, work: Path, keys: Path, prefix: str):
        self.plan_p, self.book, self.work, self.keys, self.prefix = plan_p, book, work, keys, prefix
        self.plan = json.loads(plan_p.read_text())
        self.video = book / "work" / "video"
        self.look_p = self.video / "look.json"
        self.look = json.loads(self.look_p.read_text())
        tl = sorted(q for q in (book / "work").glob("*.timeline.json") if ".raw." not in q.name)
        self.timeline = tl[0] if tl else None
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
        return (self.d(place) / f".done_{step}").exists()

    def mark(self, place: str, step: str) -> None:
        (self.d(place) / f".done_{step}").write_text(time.strftime("%F %T"))

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
        noun = re.match(r"an? (\w+)", look).group(1)
        m = re.search(r"wearing (.+?)(?:, (?:expression|holding|face)\b|\.$|$)", look)
        wear = m.group(1).strip() if m else ""
        pos = "her" if noun in ("woman", "girl") else "his"
        short = wear.split(",")[0]
        return {"look": look, "sheet": str(sheets[0]) if sheets else b["sheet"], "lora": b["lora"],
                "trigger": b["trigger"], "noun": noun, "slug": sl,
                "words": {"noun": noun, "trigger": b["trigger"], "lora": b["lora"],
                          "sheet": str(sheets[0]) if sheets else b["sheet"], "wear": wear,
                          "wear_back": f"the back of {pos} {short} and the back of {pos} head",
                          "back_detail": f"the back of {pos} {short}"}}

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
        return (d / "place_spec.json").exists() and rc in (0, 1)        # (1: written with faults, reported there)

    def step_concept(self, place):
        d = self.d(place)
        hero = self.video / "heroes" / f"{slug_of(place)}.png"
        if not hero.exists():
            log(f"{place}: no concept picture ({hero.name}); built from the spec alone")
            return True
        shutil.copy2(hero, d / "concept.png")
        line("igpu", "choose", f"concept_{self.places[place]['slug']}", d / "concept.log",
             [VENV_PY, HERE / "scenes" / "concept.py", d / "concept.png", d / "place_spec.json", d / "concept.json"])
        return (d / "concept.json").exists()

    def _build(self, place, *extra) -> int:
        d = self.d(place)
        con = ["--concept", d / "concept.json"] if (d / "concept.json").exists() else []
        return sh([PY, HERE / "sets" / "place_build.py", d / "place_spec.json", self.places[place]["set"]] + con
                  + list(extra), d / "place_build.log")

    def step_pieces(self, place):
        d = self.d(place)
        self._build(place, "--emit", d / "place_spec_c.json")
        rc = sh([PY, HERE / "places" / "generate.py", d / "place_spec_c.json"], d / "generate.log")
        if rc:
            log(f"{place}: some pieces were not made (generate.log); the set is built with what there is")
        return True

    def step_set(self, place):
        d = self.d(place)
        if self._build(place, "--write"):
            return False
        return sh([PY, HERE / "sets" / "build_set.py", self.set_dir(place) / "spec.json"], d / "build_set.log") == 0

    def step_cast(self, place):
        """Each person's figure (made once for the book), then their seated and lying stances for this set."""
        d = self.d(place)
        poses = {n: set() for n in self.people_of(place)}
        for i in self.shots_of(place):
            for p in self.plan[i][0].get("people") or []:
                if p["name"] in poses:
                    poses[p["name"]].add(str(p.get("pose") or ""))
        ok = True
        for name in self.people_of(place):
            fid, pr = self.figure_id(name), self.person(name)
            w = self.work / "cast" / pr["slug"]
            w.mkdir(parents=True, exist_ok=True)
            base = [PY, HERE / "img3d" / "cast_char.py", "--id", fid, "--sheet", pr["sheet"], "--look", pr["look"],
                    "--lora", pr["lora"], "--trigger", pr["trigger"], "--work", w / "w",
                    "--scene-tool", HERE / "scenes"]
            if not (C.LIBRARY / "character" / fid / "character.json").exists():
                rc = sh(base + ["--until", "seats"], w / "run.log")
                if rc and not (C.LIBRARY / "character" / fid / "character.json").exists():
                    rc = sh(base + ["--from", "draw", "--until", "seats"], w / "run.log")     # (a new round of drawings)
                if not (C.LIBRARY / "character" / fid / "character.json").exists():
                    log(f"{place}: no figure for {name} ({fid}); see {w / 'run.log'}")
                    ok = False
                    continue
            lie = []
            if any("ly" in q or "lie" in q for q in poses[name]):           # lying stances on this set's own beds
                sys.path.insert(0, str(HERE / "scenes"))
                import lying
                beds = sorted({b["asset"] for b in lying.beds(C.LIBRARY, self.set_dir(place)).values()})
                lie = ["--lie", "--lie-beds"] + beds if beds else []
            if any("sit" in q or "seat" in q for q in poses[name]) or lie:
                rc = sh(base + ["--from", "seats", "--set", self.places[place]["set"]] + lie,
                        w / f"stances_{self.places[place]['slug']}.log")
                if rc:
                    log(f"{place}: {name}'s stances for this set did not all build (rc {rc})")
        (d / "cast.json").write_text(json.dumps(
            {n: {"key": "abcdefgh"[k], "id": self.cast[n]} for k, n in enumerate(self.people_of(place))
             if n in self.cast}, indent=1))
        return ok

    def step_pass(self, place):
        d = self.d(place)
        (d / "pass.json").unlink(missing_ok=True)
        llm(f"pass_{self.places[place]['slug']}", d / "pass.log", [[
            PY, HERE / "scenes" / "staging_pass.py", self.plan_p, d / "shots.txt", d / "cast.json", self.timeline,
            self.set_dir(place), d / "pass.json", "--cache", d / "cache_pass"]])
        if not (d / "pass.json").exists():
            return False
        shutil.rmtree(d / "specs", ignore_errors=True)
        sh([PY, HERE / "scenes" / "from_plan.py", self.plan_p, d / "shots.txt", d / "cast.json", d / "specs",
            "--pass", d / "pass.json", "--looks", self.look_p], d / "from_plan.log")
        return bool(list((d / "specs").glob("pl_g*.json")))

    def step_stage(self, place):
        d = self.d(place)
        shutil.rmtree(d / "seq", ignore_errors=True)
        (d / "seq").mkdir()
        prev = None
        for f in sorted((d / "specs").glob("pl_g*.json")):
            rc = sh([PY, HERE / "scenes" / "speakers.py", f] + (["--prev", prev] if prev else []),
                    d / "seq" / f"{f.stem}.speakers.log",
                    {"SCENES_OUT": str(d / "seq"), "SCENES_CHAR_TOOLS": str(HERE / "characters")})
            if rc:
                log(f"{place}: scene {f.stem} failed its staging")
            if (d / "seq" / f.stem / "staged.json").exists():
                prev = d / "seq" / f.stem / "staged.json"
        return bool(list((d / "seq").glob("*/result.json")))

    def step_redraw(self, place):
        d = self.d(place)
        out = d / "run"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir()
        cast = json.loads((d / "cast.json").read_text())
        ss = json.loads((self.set_dir(place) / "spec.json").read_text())
        style = (self.book / "art_style.txt")
        run = dict(REDRAW, cast={v["key"]: self.person(n)["words"] for n, v in cast.items()},
                   scenes=sorted(os.path.dirname(p) for p in glob.glob(str(d / "seq" / "*" / "result.json"))),
                   set=str(self.set_dir(place)), out=str(out),
                   extra_refs=[ss["concept"]] if ss.get("concept") else [],
                   look=(style.read_text().strip().rstrip(".") + ".") if style.exists() else "")
        rp = d / "run.json"
        rp.write_text(json.dumps(run, indent=1))
        tag = self.places[place]["slug"]
        root = HERE.parents[1]
        if line("blender", "guides", f"guides_{tag}", out / "guides.log",
                ["bash", "-c", f"cd {root} && bash {CARD_SH} 05 {GUIDES_PY} -I tools/assets/redraw/guides.py --set "
                               f"{self.set_dir(place)} {' '.join(run['scenes'])}"]):
            return False
        acc = [PY, HERE / "redraw" / "accept.py"]
        if sh([PY, HERE / "redraw" / "redraw.py", rp], out / "redraw.log", cwd=root):
            return False
        if line("image", "refs", f"draw_{tag}", out / "run.log", ["bash", IMAGE_SH, out / "image_jobs.jsonl"]):
            return False
        sh(acc + ["checks", rp, out / "check.json", "--pose-model", POSE_MODEL], out / "accept.log", cwd=root)
        if line("image", "check", f"check_{tag}", out / "check.log", ["bash", CHECK_SH, out / "check.json"]):
            return False
        sh(acc + ["retry", rp, out / "check_result.json", out / "retry_jobs.jsonl"], out / "accept.log", cwd=root)
        if (out / "retry_jobs.jsonl").exists() and (out / "retry_jobs.jsonl").stat().st_size:
            line("image", "refs", f"redraw_{tag}", out / "run_retry.log", ["bash", IMAGE_SH, out / "retry_jobs.jsonl"])
            sh(acc + ["checks", rp, out / "check_x.json", "--ids-from", out / "retry_jobs.jsonl", "--pose-model",
                      POSE_MODEL], out / "accept.log", cwd=root)
            line("image", "check", f"checkx_{tag}", out / "check_x.log", ["bash", CHECK_SH, out / "check_x.json"])
        sh(acc + ["pick", rp, out / "check_result.json", out / "check_x_result.json", out / "accepted.json"],
           out / "accept.log", cwd=root)
        return (out / "accepted.json").exists()

    def step_keys(self, place):
        d = self.d(place)
        acc = json.loads((d / "run" / "accepted.json").read_text())
        rows = []
        for g in sorted((d / "seq").glob("pl_g*/"), key=lambda p: (len(p.name), p.name)):
            if not (g / "spec_round0.json").exists():
                continue
            for c in json.loads((g / "spec_round0.json").read_text())["cameras"]:
                if not (g / f"{c['name']}.png").exists():
                    rows.append({"shot": c["plan_id"], "key": False, "why": "not staged"})
                    continue
                a = next((v for k, v in acc.items() if k.startswith(f"{g.name}__{c['name']}__")), None)
                rows.append({"shot": c["plan_id"], "key": bool(a), "why": (a or {}).get("fails") or ""})
                if not a:
                    continue
                shutil.copy2(a["path"], self.keys / f"{c['plan_id']}_s0.png")
                sh([PY, HERE / "scenes" / "shot_people.py", g, c["name"], self.look_p, d / "cast.json",
                    self.keys / c["plan_id"]], d / "keys.log")
        staged = {r["shot"] for r in rows}
        rows += [{"shot": i, "key": False, "why": "no scene spec"} for i in self.shots_of(place) if i not in staged]
        (d / "keys.json").write_text(json.dumps(rows, indent=1))
        log(f"{place}: {sum(r['key'] for r in rows)} of {len(self.shots_of(place))} shots have a key")
        return True

    # ------------------------------------------------------------ the run
    def run(self, only=None, first="shots", last="keys", redo=()):
        todo = STEPS[STEPS.index(first):STEPS.index(last) + 1]
        for place in self.places:
            if only and place not in only and self.places[place]["slug"] not in only:
                continue
            for st in todo:
                if self.done(place, st) and st not in redo:
                    continue
                log(f"{place}: {st}")
                t = time.time()
                if not getattr(self, f"step_{st}")(place):
                    log(f"{place}: {st} FAILED after {time.time() - t:.0f} s; the place stops here")
                    break
                self.mark(place, st)
                for later in STEPS[STEPS.index(st) + 1:]:           # what follows is made again from the new result
                    (self.d(place) / f".done_{later}").unlink(missing_ok=True)
                log(f"{place}: {st} done in {time.time() - t:.0f} s")
        report = {}
        for place in self.places:
            kp = self.d(place) / "keys.json"
            rows = json.loads(kp.read_text()) if kp.exists() else []
            have = sum((self.keys / f"{i}_s0.png").exists() for i in self.shots_of(place))      # (the key folder's own)
            report[place] = {"shots": len(self.shots_of(place)), "keys": have,
                             "missing": [r for r in rows if not r["key"]]}
        (self.work / "report.json").write_text(json.dumps(report, indent=1))
        log("keys: " + "; ".join(f"{p} {v['keys']}/{v['shots']}" for p, v in report.items()))
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
    s.run(set(many("--only")), opt("--from", "shots"), opt("--until", "keys"), set(many("--redo")))


if __name__ == "__main__":
    main()
