"""A book's shot plan to engine-staged keyframes, every place of it, in one command.

    python tools/assets/story_keys.py PLAN.json BOOK_DIR WORK_DIR KEYS_DIR [--prefix ID] [--only PLACE ...]
                                      [--from STEP] [--until STEP] [--redo STEP ...] [--metahuman]

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
sys.path.insert(0, str(HERE / "scenes"))
import creature as CRE  # noqa: E402
import wardrobe as WD  # noqa: E402

STEPS = ["shots", "spec", "concept", "pieces", "set", "cast", "pass", "stage", "redraw", "keys", "drawn"]
PY = sys.executable
RA = [PY, str(HERE / "run_asset.py")]
LAB = C.UE_HOME / "lab"
# the redraw's two card scripts and its pose model (the line's image station runs them)
IMAGE_SH = Path(os.environ.get("STORY_IMAGE_SH", LAB / "redraw4" / "figs" / "run_image.sh"))
COUNT_SH = Path(os.environ.get("STORY_COUNT_SH", LAB / "redraw5" / "count.sh"))
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


STYLE_CONTENT = re.compile(r"\b(cathedral|church|chapel|temple|castle|spire|arch\w*|stone|iron|window|sky|skies|"
                           r"candle\w*|ruins?|tower)\b", re.I)


def key_style(style: str) -> str:
    """A book's art style as a key is told it: its way of painting only (as shot_flow._key_style), its clauses that
    name things to draw left out. Told the style whole, "moody stormy skies", a key of the inside of a forge had a
    storm sky for its back wall (11-18)."""
    head, _, rest = (style or "").strip().partition(":")
    head = re.sub(r"(?i)\b(gothic|baroque|romanesque|victorian|byzantine|rococo|renaissance|medieval)\b\s*", "", head)
    keep = [re.sub(r"(?i)\bgothic\s*", "", c).strip().rstrip(".") for c in rest.split(",") if c.strip() and not STYLE_CONTENT.search(c)]
    return (head.strip(" ,.") + (": " + ", ".join(keep) if keep else "")).rstrip(".") + "." if head.strip() else ""


ROOM_BG = 0.3                    # how far a room's background may be redrawn from the engine's frame


def log(m):
    print(f"[story {time.strftime('%H:%M:%S')}] {m}", flush=True)


def slug_of(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]


def sh(argv, logf: Path | None = None, env=None, cwd=None) -> int:
    e = dict(os.environ, ASSETS_LINE_CLIENT=str(HERE / "run_asset.py"), STORY_VENV_PY=str(VENV_PY), **(env or {}))
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


LLM_NEED_GB = 96                 # what the directing model needs free to start (tts_reader/animate.py STRATA_NEED_GB)


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

# a place in the open whose picture is a close view of one part of it: drawn anew as the whole place
OPEN_WORDS = ("{room} Part of it: {inside} A wide view of the whole place from well back, at eye level: open ground in "
              "front, everything that stands there seen whole from the ground up and small in the picture, the land "
              "around it and the sky above. Empty: no people. {style}")


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
        from cast_char import spread_look
        dress = spread_look("x, wearing " + dress).partition("wearing ")[2] or dress
        dress = re.sub(r"\b(his|her|their)\s+usual\s+", r"\1 ", dress, flags=re.I)   # (the wardrobe's word)
        pos = "her" if pr["noun"] in ("woman", "girl") else "his"
        dress = dress[:1].lower() + dress[1:]
        short = WD.outer_garment(dress) or dress.split(",")[0]
        sheet = WD.sheet_for(pr["words"].get("worn_sheets") or {}, r["id"]) or pr["sheet"]
        return {"wear": dress, "sheet": sheet, "wear_back": f"the back of {pos} {short} and the back of {pos} head",
                "back_detail": f"the back of {pos} {short}"}

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

    def step_concept(self, place):
        d = self.d(place)
        hero = self.video / "heroes" / f"{slug_of(place)}.png"
        if not hero.exists():
            log(f"{place}: no concept picture ({hero.name}); built from the spec alone")
            return True
        shutil.copy2(hero, d / "concept.png")
        slug = self.places[place]["slug"]
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
                style=(style.read_text().strip().rstrip(".") + ".") if style.exists() else "")
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
                style=(style.read_text().strip().rstrip(".") + ".") if style.exists() else "")
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

    def _build(self, place, *extra) -> int:
        d = self.d(place)
        con = ["--concept", d / "concept.json"] if (d / "concept.json").exists() else []
        sty = ["--style", self.book / "art_style.txt"] if (self.book / "art_style.txt").exists() else []
        return sh([PY, HERE / "sets" / "place_build.py", d / "place_spec.json", self.places[place]["set"]] + con
                  + sty + list(extra), d / "place_build.log")

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
                            ["--whole"] if getattr(self, "whole", True) else ["--metahuman"])

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
                if sh(cmd + ["--until", "seats"], w / "run.log") and \
                        not (C.LIBRARY / "character" / vid / "character.json").exists():
                    sh(cmd + ["--from", "draw", "--until", "seats"], w / "run.log")
                ok = (C.LIBRARY / "character" / vid / "character.json").exists()
                if not ok:
                    log(f"{place}: no figure for {name} in {r['id']} ({vid}); see {w / 'run.log'}")
                    (w / "FAILED").write_text(time.strftime("%F %T"))
            out[r["variant"]] = (vid if ok else None, r)
        return out

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
            if self._built(name):
                # (a figure built before its hair colour was measured on it: that one step, the rest stands)
                if not (w / "w" / "hair_calibrate.json").exists() and (w / "w" / "choice.json").exists():
                    sh(base + ["--until", "seats"], w / "run.log")
                continue
            if (w / "FAILED").exists():
                log(f"{place}: {name}'s figure failed its build before ({w / 'FAILED'})")
                continue
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
                worn[v] = {"wear_set": ow["wear"], "wear_back_set": ow["wear_back"], "worn_sheet": ow["sheet"]}
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
                "look": (said.get(what) or look_of(what) or what) + ALONE + (HEAD_OPEN if on in ("head", "face") else "")}
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

    def step_stage(self, place):
        d = self.d(place)
        shutil.rmtree(d / "seq", ignore_errors=True)
        (d / "seq").mkdir(exist_ok=True)      # (a stopped run's last scene may still be writing into it)
        prev = None
        for f in sorted((d / "specs").glob("pl_g*.json")):
            if not json.loads(f.read_text()).get("characters"):
                # an insert (nobody of the cast in it: the place or a thing seen alone): the set photographed from
                # its placed camera, no people's checks to make (from_plan.insert_camera)
                (d / "seq" / f.stem).mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, d / "seq" / f.stem / "spec_round0.json")
                if sh([PY, HERE / "scenes" / "scene.py", f], d / "seq" / f"{f.stem}.scene.log",
                      {"SCENES_OUT": str(d / "seq"), "SCENES_CHAR_TOOLS": str(HERE / "characters")}):
                    log(f"{place}: insert {f.stem} failed its staging")
                continue
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
        run = dict(REDRAW, cast={v["key"]: dict(self.person(n)["words"], **({"stand_in": True} if v.get("stand_in") else {}))
                                 for n, v in cast.items()},
                   scenes=sorted(os.path.dirname(p) for p in glob.glob(str(d / "seq" / "*" / "result.json"))),
                   set=str(self.set_dir(place)), out=str(out),
                   # (a cabin with views outside: its concept picture shows one view of its own, which the redraw
                   # then painted in every window; the cabin's look reaches the redraw through its pieces)
                   extra_refs=[ss["concept"]] if ss.get("concept") and not (ss.get("views") or {}).get("words") else [],
                   look=key_style(style.read_text()) if style.exists() else "")
        # a room's own walls and pieces are drawn ones: its background is held nearer the engine's frame than a
        # place in the open, whose ground and far land are rough. At the open's share a forge's keys moved its
        # hearth and door and hung lanterns the set does not have (11-18, seven shots at three strengths)
        if not ss.get("world"):
            run["strength_bg"] = ROOM_BG
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
            sj = json.loads((g / "speakers.json").read_text()) if (g / "speakers.json").exists() else {}
            for c in json.loads((g / "spec_round0.json").read_text())["cameras"]:
                if not (g / f"{c['name']}.png").exists():
                    rows.append({"shot": c["plan_id"], "key": False, "why": "not staged"})
                    continue
                st = (sj.get("shots") or {}).get(c["name"]) or {}
                # every shot ends with an engine key (the user's rule, 11-17): a staging that failed its own checks
                # gives the least-bad try speakers.py kept, flagged with what it missed; a redraw that failed its
                # checks gives its best drawing, flagged; with no drawing at all, the engine still itself, flagged
                miss = [] if not st or st.get("ok") else list(map(str, (st.get("least_bad") or {}).get("misses")
                                                                   or (st.get("history") or [{}])[-1].get("why")
                                                                   or st.get("why") or []))
                a = next((v for k, v in acc.items() if k.startswith(f"{g.name}__{c['name']}__")), None)
                flag = {"staging_miss": miss} if miss else {}
                if not a:
                    tries = sorted((d / "run").glob(f"{g.name}__{c['name']}__*.png"), key=lambda q: ("__x" in q.stem, q.name))
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
                               if q.exists() and str(q) != a["path"] and lum_of(q) >= KEY_DARK]
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
        log(f"{place}: {sum(r['key'] for r in rows)} of {len(self.shots_of(place))} shots have a key")
        return True

    UNREADABLE = ("is not in the picture", "is cut by the frame's edge", "the camera is outside the room",
                  "the camera is inside", "the camera is too near", "the still is nearly black")
    # (a kept least-bad staging with these misses shows nobody readable; the last two: speakers.still_why and
    # creature_view_why, a camera against or under a creature's head)

    def keyless(self, place) -> list:
        """The shots without a key, and those whose key is a least-bad staging that left its people unreadable (the
        subject or a named person out of the picture, the camera inside something): both are staged again."""
        kp = self.d(place) / "keys.json"
        out = []
        for r in json.loads(kp.read_text()) if kp.exists() else []:
            miss = " ".join(map(str, r.get("staging_miss") or []))
            walled = any(int(x) >= 50 for x in re.findall(r"foreground covers (\d+)%", miss))
            if not r["key"] or walled or any(w in miss for w in self.UNREADABLE):
                out.append(r["shot"])
        return out

    def restage_keyless(self, place) -> None:
        """Every shot ends with an engine key (the user's rule, 11-17): a place whose keys step left shots without
        one is staged again, once, with each shot that fails every gate taken from a clear wide on its people
        (speakers.py SPEAKERS_FALLBACK_ALL), then drawn and keyed again."""
        miss = self.keyless(place)
        if not miss:
            return
        log(f"{place}: {len(miss)} shot(s) without a key {miss}: staged again with the clear-wide fallback")
        os.environ["SPEAKERS_FALLBACK_ALL"] = "1"
        try:
            for st in ("stage", "redraw", "keys"):
                if not getattr(self, f"step_{st}")(place):
                    log(f"{place}: the restage's {st} FAILED")
                    return
        finally:
            os.environ.pop("SPEAKERS_FALLBACK_ALL", None)
        log(f"{place}: after the restage, {len(self.keyless(place))} shot(s) without a key")

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
    def run(self, only=None, first="shots", last="keys", redo=()):
        todo = STEPS[STEPS.index(first):STEPS.index(last) + 1]
        for place in self.places:
            if only and place not in only and self.places[place]["slug"] not in only:
                continue
            if 0 < len(self.shots_of(place)) <= DRAWN_MAX_SHOTS:         # a glance at a place: drawn, not built
                if not self.done(place, "drawn") or "drawn" in redo:
                    log(f"{place}: drawn")
                    llm_down()
                    if self.step_drawn(place):
                        self.mark(place, "drawn")
                    else:
                        log(f"{place}: drawn FAILED")
                continue
            for st in todo:
                if self.done(place, st) and st not in redo:
                    continue
                log(f"{place}: {st}")
                if st not in ("spec", "pass"):
                    llm_down()
                t = time.time()
                if not getattr(self, f"step_{st}")(place):
                    log(f"{place}: {st} FAILED after {time.time() - t:.0f} s; the place stops here")
                    break
                self.mark(place, st)
                for later in STEPS[STEPS.index(st) + 1:]:           # what follows is made again from the new result
                    (self.d(place) / f".done_{later}").unlink(missing_ok=True)
                log(f"{place}: {st} done in {time.time() - t:.0f} s")
                if st == "keys":
                    self.restage_keyless(place)
        report = {}
        for place in self.places:
            kp = self.d(place) / "keys.json"
            rows = json.loads(kp.read_text()) if kp.exists() else []
            have = sum(bool(r["key"]) and (self.keys / f"{r['shot']}_s0.png").exists() for r in rows)
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
    s.whole = "--metahuman" not in a           # figures as whole generated meshes on the skeleton (cast_char.py --whole),
    #                                      the default since 11-15; --metahuman casts MetaHuman bodies wearing generated outfits
    s.run(set(many("--only")), opt("--from", "shots"), opt("--until", "keys"), set(many("--redo")))


if __name__ == "__main__":
    main()
