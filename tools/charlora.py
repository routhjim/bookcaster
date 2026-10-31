"""Per-character identity LoRAs for the keyframe drawer (Qwen-Image-2.1), for any book and any character.

    E=~/tts-ab/.venv-charlora-igpu/bin/python      (gfx1151 torch + diffusers/peft; runs on the iGPU)
    I=$(~/bin/hip-index igpu); export HIP_VISIBLE_DEVICES=$I HSA_CU_MASK=$I:0-38
    $E -I tools/charlora.py dataset <book> <Character> [--outfit ID] [--model rigged.blend] [--motion BASE ...]
    $E -I tools/charlora.py train   <book> <Character> [--steps 1200] [--rank 16] [--lr 1e-4] [--res 512]
    $E -I tools/charlora.py sheet   <book> <Character>          (contact sheet of the dataset)

Everything is per-book data under <book>/work/video/charlora/<slug>/ (nothing story-specific here):
  cand/       candidates: sheet crops, the book's own keyframes / portraits / scene pictures, NEW views
  judge.json  per candidate: person box, head box, age estimate, description match, outfit, likeness
  data/       the kept images + captions (NNN.png + NNN.txt), meta.json, sheet.jpg
  lora/       step_XXXXX.safetensors (+ .pt optimizer state for resume), train.log
Trigger word: "chr_<slug>"; caption = "<trigger>, <look or face + outfit>, <framing/view/pose>, <setting>".

Sources (generic, from what any book has):
  * the character's reference sheets (look.json look + each wardrobe outfit): head half and figure half;
  * the book's drawn keyframes, talking portraits and scene pictures that show the character: the vision
    model (Qwen3-VL-8B = Qwen-Image's own text encoder, tools/vl_judge.py) finds the person by the look's
    first clause ("a woman of about 70"), and it is kept only if it passes the checks below;
  * NEW views: pose-library skeletons (tools/poses.py; any character) at random facings, framings, settings
    (the book's places) and lights, drawn with the character's sheets as references + pose control; and,
    when the character has a rigged 3D model (tools/blockout_model.py), the model rendered from many angles
    (+ RigMotion poses) as depth control + img2img init (tools/charlora_views_bl.py). Raw renders are never
    kept: only the repainted pictures, and only those that pass.
Checks (every candidate): exactly one person (generated views); the look's age (if the look gives one:
"of about 70" -> the estimate must be >= 60; a youth <= +10); the person matches the look/outfit text
(yes/no); the outfit worn (the vision model picks base vs each wardrobe outfit); likeness = DINOv2 cosine
of the head crop vs the sheet's head crop (kept >= --min-like).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
HOME = Path.home()
MODEL = Path(os.environ.get("QWEN_IMAGE_MODEL", HOME / "models" / "qwen-image-2.1")).expanduser()
BLENDER = os.environ.get("BLENDER", str(HOME / "opt" / "blender" / "current" / "blender"))
POSE_PY = os.environ.get("POSE_PY", str(HOME / "tts-ab" / ".venv-pose" / "bin" / "python"))
FA_AMD = Path(os.environ.get("QWEN_IMAGE_FA", HOME / "src" / "attn-lab")).expanduser()


def log(msg: str) -> None:
    sys.stderr.write(time.strftime("%H:%M:%S ") + msg.rstrip() + "\n")
    sys.stderr.flush()


def mem_avail_gb() -> float:
    for l in open("/proc/meminfo"):
        if l.startswith("MemAvailable"):
            return int(l.split()[1]) / 2**20
    return 0.0


def need_mem(gb: float = 15) -> None:
    a = mem_avail_gb()
    if a < gb:
        sys.exit(f"MemAvailable {a:.1f} GB < {gb} GB: not loading a model")


def gpu_env() -> dict:
    """The iGPU unless the caller pinned a device; CU mask keeps one CU for the desktop."""
    env = dict(os.environ)
    if "HIP_VISIBLE_DEVICES" not in env:
        i = subprocess.run([str(HOME / "bin" / "hip-index"), "igpu"], capture_output=True, text=True).stdout.strip()
        env["HIP_VISIBLE_DEVICES"] = i
        env.setdefault("HSA_CU_MASK", f"{i}:0-38")
    env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)      # gave NaNs in the VL encoder (10-19)
    return env


# --- the character, from the book's own data ----------------------------------------------------------

class Char:
    def __init__(self, book: str, name: str):
        from tts_reader import video as V
        self.V = V
        root = Path(book).expanduser()
        if not root.exists():
            root = HOME / "audiobooks" / book
        self.vdir = root / "work" / "video" if (root / "work" / "video").exists() else root
        self.look = json.loads((self.vdir / "look.json").read_text())
        import registry as RG
        # 10-27: one owner for who (tools/registry.py): any spelling is the person, their look is look.json's
        # first-stored one ("Dr. Watson" = "Doctor Watson"), and every plan/wardrobe/index lookup goes by person
        self.R = R = RG.Book(self.vdir.parent.parent)
        if R.person(name, narrator=False) is None:
            sys.exit(f"{name} not in look.json characters: {R.people()}")
        self.name = R.person(name, narrator=False)
        self.desc = R.look(self.name).strip().rstrip(".")
        self.slug = V.slug(self.name)
        self.trigger = f"chr_{self.slug}"
        # CHARLORA_OUT: a test build writes beside, not over, the book's (both of its processes see the env)
        self.out = (Path(os.environ["CHARLORA_OUT"]) if os.environ.get("CHARLORA_OUT") else self.vdir / "charlora") / self.slug
        self.refs_dir = self.vdir / "refs"
        self.chapters = sorted(p for p in self.vdir.iterdir() if (p / "plan.json").exists())
        self.plans = {c.name: json.loads((c / "plan.json").read_text()) for c in self.chapters}
        # the book's one style (registry: art_style.txt > a chapter's style.txt > the default), as the sheets are drawn
        # and the keys are prompted: their names hash it (10-27: the first chapter's style.txt only, '' without one)
        self.draw_style = R.ref_style()     # the sheets' own style, never the picture's (10-27)
        self.style = self.draw_style.strip().rstrip(".") + "."
        self.period_s = V.period_line(str(self.refs_dir)).strip() + " " if V.period_of(str(self.refs_dir)) else ""  # not in input_hash
        # outfits: None = the usual look; then each wardrobe row (by id) across the chapters
        self.outfits = {None: {"id": None, "dress": None}}
        self.outfit_as = {}                                  # outfit id -> the wardrobe's spelling of the person
        for plan in self.plans.values():
            wk = V.wardrobe_key(plan, self.name, R)
            for r in (plan.get("wardrobe") or {}).get(wk, []) if wk else []:
                # 10-25 Acts 2: a plan's wardrobe saved before the clothes-only filter kept "tongues of fire" = Peter's
                # usual tunic and belt; no sheet is drawn for such a row, so the LoRA failed "no reference sheet"
                # (only those: dropping other rows changes a LoRA's hash = a retrain, e.g. Acts 1's ascended Jesus)
                dress = V.worn_only(str(r.get("dress") or ""))
                if not (dress and V.changes_nothing(dress, self.desc)):
                    self.outfits.setdefault(r["id"], r)
                    self.outfit_as.setdefault(r["id"], wk)
        self.face = V._short(self.desc)                      # age, face, hair
        self.subject = self.desc.split(",")[0].strip()       # "a woman of about 70": what to look for
        m = re.search(r"\b(?:of|aged)\s+(?:about\s+|around\s+)?(\d{1,3})\b", self.desc)
        self.age = int(m.group(1)) if m else None
        # facial hair stated in the look: the judge must see the same (a clean-shaven knight was painted
        # bearded in the keys, 10-20)
        d = self.desc.lower()
        self.beard = (False if re.search(r"clean-shaven|beardless|no beard|smooth beardless", d)
                      else True if re.search(r"\bbeard|moustache|mustache", d) else None)
        # the staging engine's asset index (flow3d): kind (human | creature) and, per outfit, the sheet
        # and the rigged model it built; "" = the usual look
        ix = self.vdir / "stage" / "characters" / "index.json"
        ixc = json.loads(ix.read_text()).get("characters", {}) if ix.exists() else {}
        ent = ixc.get(R.key_in(self.name, list(ixc)) or self.name, {})
        self.kind = ent.get("kind") or ("human" if self.age is not None or re.search(
            r"\b(man|woman|boy|girl|youth|child|person|lady|king|queen)\b", d) else "creature")
        self.assets = {(k or None): v for k, v in (ent.get("outfits") or {}).items()}
        for oid in list(self.assets):
            if oid not in self.outfits and oid is not None:
                self.assets.pop(oid)
        if self.kind != "human":
            self.beard = None

    def text(self, oid) -> str:
        """The person as words: the usual look, or face + the outfit."""
        if oid is None:
            return self.desc
        return f"{self.face}, wearing {self.outfits[oid]['dress'].rstrip('.')}"

    def model(self, oid) -> str | None:
        """The rigged model for this outfit (the usual look's when the outfit has none)."""
        for o in (oid, None):
            m = (self.assets.get(o) or {}).get("model")
            if m and Path(m).exists():
                return m
        return None

    def pick(self, main_oid, rng: random.Random, p_main: float = 0.75):
        """The outfit of one new view: main_oid "*" = every outfit equally (all of the wardrobe);
        an outfit id = that one mostly, the rest spread; None = the usual look mostly."""
        all_ = list(self.outfits)
        if main_oid == "*":
            return rng.choice(all_)
        if rng.random() < p_main or len(all_) == 1:
            return main_oid
        return rng.choice([o for o in all_ if o != main_oid] or all_)

    def sheet(self, oid) -> Path | None:
        # 10-21 knight_forged_v2: the index pin came FIRST, and the index is written from this very call, so the
        # Knight's first (tunic) sheet stayed pinned after the look gained armour: model + LoRA in a tunic. A human's
        # sheet is the one the CURRENT look draws; the pin only for a creature (its composite head sheet) or a gap.
        a = (self.assets.get(oid) or {}).get("sheet")
        if a and Path(a).exists() and self.kind != "human":
            return Path(a)
        V, R = self.V, self.R                                # a plan that casts them (10-25: the Disciple is only in Acts 2)
        # ... under ANY spelling, its sheet named by the plan's own (10-27: look.json's first spelling "Doctor Watson",
        # the plan's "Dr. Watson": no plan matched, KeyError -> no sheet -> Watson's LoRA build failed)
        p = None
        cast_in = [pl for pl in self.plans.values() if R.key_in(self.name, list(V.looks(pl)[0]))]
        for pl in cast_in or list(self.plans.values())[:1]:
            try:
                if oid is None:
                    pr = V.ref_prompts(pl, self.draw_style, R)
                    k = R.key_in(self.name, list(pr))
                    p = V.ref_path(self.refs_dir, k, pr[k][0]) if k else None
                elif V.wardrobe_key(pl, self.name, R):
                    p = V.outfit_ref(self.refs_dir, pl, self.draw_style, self.outfit_as.get(oid) or self.name,
                                     self.outfits[oid], R)
            except Exception:                                # noqa: BLE001 - fall back to the newest sheet
                p = None
            if p is not None:
                break
        if p and p.exists():
            return p
        if self.kind == "human":                             # 10-25 Acts 2: the Disciple's LoRA trained on his
            return None                                      # old white-and-blue sheet (newest file) after a look change
        if a and Path(a).exists():
            return Path(a)
        pat = f"{self.slug}-*.png" if oid is None else f"{V.slug(self.name + ' ' + oid)}-*.png"
        c = sorted(self.refs_dir.glob(pat), key=lambda x: x.stat().st_mtime)
        return c[-1] if c else None

    def age_ok(self, est) -> bool | None:
        if self.age is None or est is None:
            return None
        if self.age >= 55:
            return est >= self.age - 10
        if self.age <= 25:
            return est <= self.age + 10
        return abs(est - self.age) <= 12


# --- candidates ---------------------------------------------------------------------------------------

def _save(im, path: Path, max_side: int = 1536) -> None:
    s = max_side / max(im.size)
    if s < 1:
        im = im.resize((round(im.width * s), round(im.height * s)))
    path.parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path)


def cand_sheets(c: Char, cands: list) -> None:
    from PIL import Image
    for oid in c.outfits:
        sh = c.sheet(oid)
        if not sh:
            continue
        im = Image.open(sh).convert("RGB")
        w, h = im.size
        tag = c.V.slug(oid or "base")
        # an OUTFIT sheet's portrait half is the base look's face and clothes copied (10-27: Roylott's top hat over his
        # dressing gown, Watson's tweed over his nightshirt, captioned as the outfit): only its full figure trains
        for part, box, framing in (("head", (0, 0, w // 2, h), "close-up portrait, head and shoulders, front view"),
                                   ("figure", (w // 2, 0, w, h), "full body, standing, front view")):
            if oid is not None and part == "head":
                continue
            p = c.out / "cand" / f"sheet_{tag}_{part}.png"
            if not p.exists():
                _save(im.crop(box), p)
            cands.append({"id": p.stem, "path": str(p), "src": "sheet", "outfit": oid, "framing": framing,
                          "setting": "plain light background" if part == "head" else "",
                          "whole": True})


def cand_book(c: Char, cands: list) -> None:
    """The book's own pictures that show the character ALONE and were drawn from their CURRENT look (after their
    current sheet): drawn keyframes, portraits, scene pictures (10-27: older drawings taught the old faces back)."""
    seen = set()
    sh = c.sheet(None)
    since = Path(sh).stat().st_mtime if sh and Path(sh).exists() else float("inf")
    alone = lambda who: len(who or []) == 1 and c.R.same(who[0], c.name)
    for ch in c.chapters:
        anim = ch / "anim"
        plan = c.plans[ch.name]
        for p in plan.get("pictures", []):
            if alone(p.get("who")) and (ch / "img" / f"{p['id']}.png").exists():
                seen.add(ch / "img" / f"{p['id']}.png")
        kf = anim / "keyframes.json"
        if kf.exists():
            passed = json.loads((anim / "keyframe_checks.json").read_text()).get("passed", {}) \
                if (anim / "keyframe_checks.json").exists() else {}
            for f in json.loads(kf.read_text()).get("frames", []):
                k = anim / "keyframes" / f"{f['shot']}.png"
                if alone(f.get("who")) and k.exists() and (not passed or f["shot"] in passed):
                    seen.add(k)
        # portraits: talk-<speaker>-<pic>-<hash>.png; the listener is not in the name, so keep a portrait
        # when the character speaks or is in that scene picture's people
        pic_who = {p["id"]: p.get("who") or [] for p in plan.get("pictures", [])}
        for k in sorted((anim / "portraits").glob("talk-*.png")) if (anim / "portraits").exists() else []:
            m = re.match(r"talk-(.+)-(p\d+)-[0-9a-f]+$", k.stem)
            if m and c.R.same(m.group(1), c.name):          # their own portrait only
                seen.add(k)
    for k in sorted(x for x in seen if x.stat().st_mtime > since):     # drawn from the current look only
        cid = "book_" + hashlib.sha1(str(k).encode()).hexdigest()[:10]
        cands.append({"id": cid, "path": str(k), "src": "book", "outfit": "?", "framing": "", "setting": "",
                      "whole": False})


PROPS = ("sword", "hammer", "shield", "spear", "axe", "bow_", "staff", "dagger", "lance", "helm")   # props in pose names
POSE_SKIP = ("lying", "sleeping", "falling", "rolling", "crawling", "carried", "hanging", "embrace", "climbing",
             "leaping")
LIGHTS = ["lit by warm firelight from the left", "lit by warm firelight from the right", "soft overcast daylight",
          "cold blue moonlight", "candlelight from below", "strong rim light from behind",
          "dim lantern light", "bright midday sun with hard shadows"]
VIEWS = ["three-quarter view facing left", "three-quarter view facing right", "seen in profile from the left",
         "seen in profile from the right", "front view", "looking down", "looking up", "seen from slightly above",
         "seen from a low angle"]


def cutout(src: Path, dst: Path) -> Path:
    if not dst.exists():
        r = subprocess.run([POSE_PY, "-I", str(HERE / "cutout.py"), str(src), str(dst)], capture_output=True)
        if r.returncode or not dst.exists():
            return src
    return dst


def sheet_refs(c: Char, oid) -> list[str]:
    """Head half + figure half of the outfit's sheet, cut out onto grey (Qwen copies a reference's
    background into the picture)."""
    from PIL import Image
    sh = c.sheet(oid)
    if not sh:
        return []
    out = []
    im = Image.open(sh).convert("RGB")
    w, h = im.size
    for part, box in (("head", (0, 0, w // 2, h)), ("figure", (w // 2, 0, w, h))):
        fixed = c.out / "refs" / "head_fixed.png"
        if part == "head" and fixed.exists():         # the sheet's face contradicted the look: repaired head
            out.append(str(cutout(fixed, fixed.with_name("head_fixed_cut.png"))))
            continue
        p = c.out / "refs" / f"{c.V.slug(oid or 'base')}_{part}.png"
        if not p.exists():
            _save(im.crop(box), p)
        out.append(str(cutout(p, p.with_name(p.stem + "_cut.png"))))
    return out


def look_fix(c: Char, r: dict) -> str | None:
    """What a judged head contradicts in the look, as a drawing instruction (None = consistent)."""
    fix = []
    if c.beard is False and r.get("beard"):
        fix.append("completely clean-shaven: a smooth shaven chin, upper lip and cheeks, no stubble, no beard, "
                   "no moustache")
    if c.beard is True and r.get("beard") is False:
        fix.append(f"with the facial hair of the description ({c.face})")
    if c.age_ok(r.get("age")) is False:
        fix.append(f"aged about {c.age}, the face of a person of {c.age}")
    return "; ".join(fix) or None


def repair_head(c: Char, make_scorer, n: int = 4) -> Path | None:
    """The look-contradiction filter at the source: when the sheet's own face contradicts the look (the
    'clean-shaven' knight's sheet has heavy stubble, so every view copied it), redraw the head half with
    the sheet as reference + the instruction, judge the redraws, keep the most alike that passes, as
    refs/head_fixed.png (used as image 1 by every new view). Needs the Scorer loaded; the redraws go
    through qwen_image.py in a subprocess; the Scorer is unloaded for the draw (~17 + ~20 GB fit the iGPU, not a
    24 GB XTX: 10-21 Hulda/King OOM) and loaded again to judge the redraws."""
    from PIL import Image
    fixed = c.out / "refs" / "head_fixed.png"
    if fixed.exists() or c.kind != "human":
        return fixed if fixed.exists() else None
    sh = c.sheet(None)
    if not sh:
        return None
    S = make_scorer()
    im = Image.open(sh).convert("RGB")
    half = im.crop((0, 0, im.width // 2, im.height))
    hb = S.j.box(half, f"the head of {c.subject}") or [0.15, 0.05, 0.85, 0.75]
    head = crop(half, _pad(hb, 0.15))
    r = {"age": _num(S.ask(head, "How old does this person look, judging only by the face and skin? Answer with "
                                 "one number of years.")) if c.age else None}
    if c.beard is not None:
        r["beard"] = _yes(S.ask(head, "Does this person have a beard, stubble or a moustache? Answer yes or no.", 3))
    fix = look_fix(c, r)
    log(f"sheet head vs look: {r} -> {fix or 'consistent'}")
    if not fix:
        return None
    src = c.out / "refs" / "base_head.png"
    if not src.exists():
        _save(half, src)
    rd = c.out / "refs" / "repair"
    rd.mkdir(parents=True, exist_ok=True)
    jobs = [{"id": f"fix{k}", "seed": 101 + k, "size": [768, 1024], "refs": [str(src)], "path": str(rd / f"fix{k}.png"),
             "prompt": (f"A large head-and-shoulders portrait of the same person as in image 1, facing the viewer: "
                        f"exactly the same face shape, eyes, nose, mouth, hair and clothes, but {fix}. "
                        f"{c.name} is {c.desc}. Plain light background, even light. {c.style} {c.period_s}No text, no "
                        f"lettering, no signature.")} for k in range(n)]
    ref_e = S.embed(head)
    del S
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    draw(jobs, rd)
    S = make_scorer()
    best = None
    for j in jobs:
        im2 = Image.open(j["path"]).convert("RGB")
        hb2 = S.j.box(im2, f"the head of {c.subject}") or [0, 0, 1, 1]
        h2 = crop(im2, _pad(hb2, 0.15))
        r2 = {"age": _num(S.ask(h2, "How old does this person look, judging only by the face and skin? Answer "
                                    "with one number of years.")) if c.age else None}
        if c.beard is not None:
            r2["beard"] = _yes(S.ask(h2, "Does this person have a beard, stubble or a moustache? Answer yes or no.", 3))
        like = float(S.embed(h2) @ ref_e)
        ok = look_fix(c, r2) is None
        log(f"  repair {j['id']}: {r2} like {like:.3f} {'OK' if ok else 'still wrong'}")
        if ok and (best is None or like > best[0]):
            best = (like, j["path"])
    if best is None:
        log("  no repaired head passed: keeping the sheet's")
        return None
    shutil.copy(best[1], fixed)
    log(f"  repaired head: {fixed} (like {best[0]:.3f})")
    return fixed


def gen_jobs_pose(c: Char, n: int, main_oid, rng: random.Random) -> list[dict]:
    import poses as P
    own = f"{c.desc} {' '.join(str(r.get('dress', '')) for r in c.outfits.values())}".lower()
    lib = [p for p in P.list_poses() if not any(s in p["name"] for s in POSE_SKIP)
           and not any(x in p["name"] and x not in own for x in PROPS)]    # no prop the person doesn't have
    places = list((c.look.get("places") or {}).items())
    jobs = []
    for k in range(n):
        # every outfit its share of the pose drawings, in turn (10-27: Roylott's "outdoor attire" got none)
        oid = list(c.outfits)[k % len(c.outfits)] if main_oid == "*" and c.outfits else c.pick(main_oid, rng)
        refs = sheet_refs(c, oid)
        framing = rng.choices(["full", "half", "close"], [0.4, 0.35, 0.25])[0]
        land = rng.random() < 0.4
        W, H = (1024, 768) if land else (768, 1024)
        place, pdesc = rng.choice(places) if places else ("", "")
        setting = f"in {place}: {pdesc.split(',')[0].rstrip('.')}" if place else "in a plain stone room"
        light = rng.choice(LIGHTS)
        job = {"id": f"gen_pose_{k:03d}", "seed": rng.randrange(1, 10**6), "size": [W, H], "refs": refs}
        meta = {"src": "pose", "outfit": oid, "setting": setting + ", " + light}
        if c.kind != "human":
            # no skeleton library for creatures: views in words (close head / whole body / an angle)
            view = rng.choice(VIEWS)
            what = {"close": "a close-up of the head and neck", "half": "the head, neck and front of the body",
                    "full": "the whole body from head to tail"}[framing]
            meta["framing"] = f"{what}, {view}"
            job["prompt"] = (f"{c.name}, {what}, {view}. {c.name} is {c.desc}; exactly the head of the creature "
                             f"in image 1 and the body of the creature in image 2. Only this one creature in the "
                             f"picture. The scene is {setting}, {light}. {c.style} {c.period_s}No text, no lettering, no signature.")
            jobs.append((job, meta))
            continue
        if framing == "close":
            view = rng.choice(VIEWS)
            meta["framing"] = f"close-up portrait, head and shoulders, {view}"
            body = f"A close-up portrait of {c.name}, head and shoulders, {view}"
        else:
            pz = rng.choice(lib)
            facing = rng.choice(["left", "right"])
            cx = rng.uniform(0.35, 0.65) if land else 0.5
            if framing == "full":
                top = rng.uniform(0.05, 0.12)
                box = [cx - 0.45, top, cx + 0.45, rng.uniform(0.9, 0.97)]
            else:
                box = [cx - 0.7, rng.uniform(0.04, 0.1), cx + 0.7, rng.uniform(1.6, 1.9)]
            sk = c.out / "cand" / "ctl" / f"{job['id']}_pose.png"
            sk.parent.mkdir(parents=True, exist_ok=True)
            P.render_skeleton([{"pose": pz["name"], "box": box, "facing": facing}], size=(W, H)).save(sk)
            job["control"] = {"image": str(sk), "type": "pose", "scale": 0.6}
            what = "full body, head to feet" if framing == "full" else "from the waist up"
            desc = pz.get("desc") or pz["name"].replace("_", " ")
            meta["framing"] = f"{what}, {desc.rstrip('.')}"
            body = f"{c.name}, {what}, {desc.rstrip('.')}"
        clothes = f"wearing {c.outfits[oid]['dress'].rstrip('.')}" if oid else ""
        # a close-up has no pose skeleton to hold ONE body: the figure reference came out as a second person beside
        # the face (10-31: 109 of 224 close-ups over 3 books had 2+ people, full/waist 7 of 696; the head reference
        # alone: 0/15 vs 10/15, the outfit worn as often, 9/15) - the clothes in words only
        if framing == "close":
            job["refs"] = refs[:1]
        two = " and the clothes of the person in image 2" if len(job["refs"]) > 1 else ""
        job["prompt"] = (f"{body}. {c.name} is {c.text(oid) if oid is None else c.face}"
                         f"{', ' + clothes if clothes else ''}; exactly the face, age and hair of the person in "
                         f"image 1{two}. Only this one person in the picture. "
                         f"The scene is {setting}, {light}. {c.style} {c.period_s}No text, no lettering, no signature.")
        jobs.append((job, meta))
    return jobs


def gen_jobs_mesh(c: Char, model: str | None, motions: list[str], n: int, main_oid, rng: random.Random,
                  strength: tuple) -> list:
    """Render the rigged model (Blender, Workbench) from n cameras, then img2img each through Qwen with the
    character's references + the render's depth."""
    import numpy as np
    from PIL import Image
    import blockout as B
    vdir = c.out / "cand" / "views"
    views = []
    for k in range(n):
        v = {"id": f"v{k:03d}", "yaw": rng.choice([0, 20, 40, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 345]),
             "elev": rng.uniform(-8, 22), "frame": rng.choices(["head", "half", "full"], [0.35, 0.35, 0.3])[0],
             "lens": rng.choice([35, 50, 70]), "light_rot": rng.uniform(-90, 90)}
        v["oid"] = c.pick(main_oid, rng, 0.8)
        v["blend"] = model or c.model(v["oid"])
        if not v["blend"]:
            continue
        ms = [m for m in motions if Path(m + ".npz").exists()]
        if ms and c.kind == "human" and rng.random() < 0.6:
            m = rng.choice(ms)
            nf = len(np.load(m + ".npz")["pos"])
            v.update(motion=m, motion_frame=rng.randrange(nf))
        views.append(v)
    land = {v["id"]: ((v["frame"] == "half" or c.kind != "human") and rng.random() < 0.5) for v in views}
    todo = [v for v in views if not (vdir / f"{v['id']}_depth.npy").exists()]
    for blend in sorted({v["blend"] for v in todo}):
      for size in ([768, 1024], [1024, 768]):
        vs = [v for v in todo if v["blend"] == blend and (land[v["id"]] if size[0] > size[1] else not land[v["id"]])]
        if not vs:
            continue
        jf = vdir / f"views_{hashlib.sha1(blend.encode()).hexdigest()[:6]}_{size[0]}x{size[1]}.json"
        vdir.mkdir(parents=True, exist_ok=True)
        jf.write_text(json.dumps({"blend": blend, "out_dir": str(vdir), "size": size, "views": vs}, indent=1))
        t = time.time()
        r = subprocess.run([BLENDER, "-b", "-t", "16", "--factory-startup", "--python",
                            str(HERE / "charlora_views_bl.py"), "--", str(jf)], capture_output=True, text=True)
        if r.returncode:
            sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
            sys.exit("blender views failed")
        log(f"rendered {len(vs)} model views {size} in {time.time() - t:.0f} s")
    places = [sorted(c.refs_dir.glob(f"{c.V.slug(p)}-*.png")) for p in (c.look.get("places") or {})]
    place_imgs = [g[-1] for g in places if g]
    jobs = []
    for v in views:
        col, dep = vdir / f"{v['id']}_color.png", vdir / f"{v['id']}_depth.npy"
        z = np.load(dep)
        d_png, init = vdir / f"{v['id']}_depth.png", vdir / f"{v['id']}_init.png"
        if not d_png.exists():
            B.depth_png(z, d_png)
            a = np.asarray(Image.open(d_png).convert("L")).copy()
            a[~(np.isfinite(z) & (z < 1e6) & (z > 0))] = 20          # empty = a far wall, not a black border
            Image.fromarray(a).convert("RGB").save(d_png)
        fg = Image.open(col).convert("RGBA")
        if not init.exists():
            if place_imgs:
                bg = Image.open(rng.choice(place_imgs)).convert("RGB")
                s = max(fg.width / bg.width, fg.height / bg.height)
                bg = bg.resize((math.ceil(bg.width * s), math.ceil(bg.height * s)))
                x0, y0 = (bg.width - fg.width) // 2, (bg.height - fg.height) // 2
                bg = bg.crop((x0, y0, x0 + fg.width, y0 + fg.height))
            else:
                bg = Image.new("RGB", fg.size, (90, 90, 95))
            bg.paste(fg, (0, 0), fg)
            bg.save(init)
        oid = v["oid"]
        view = ("front view" if v["yaw"] in (0, 345, 20) else "seen from behind" if 150 <= v["yaw"] <= 210
                else "seen in profile" if v["yaw"] in (90, 270) else "three-quarter view")
        what = {"head": "close-up portrait, head and shoulders", "half": "from the waist up",
                "full": "full body, head to feet"}[v["frame"]]
        light = rng.choice(LIGHTS)
        W, H = fg.size
        clothes = f", wearing {c.outfits[oid]['dress'].rstrip('.')}" if oid else ""
        if c.kind == "human":
            prompt = (f"{c.name}, {what}, {view}. {c.name} is {c.text(oid) if oid is None else c.face}{clothes}; "
                      f"exactly the face, age and hair of the person in image 1 and the clothes of the person "
                      f"in image 2. A painted illustration, not a 3D render: natural skin, cloth and hair. Only "
                      f"this one person. {light}. {c.style} {c.period_s}No text, no lettering, no signature.")
        else:
            what = {"head": "a close-up of the head", "half": "the head and front of the body",
                    "full": "the whole body"}[v["frame"]]
            prompt = (f"{c.name}, {what}, {view}. {c.name} is {c.desc}; exactly the head of the creature in "
                      f"image 1 and the body of the creature in image 2. A painted illustration, not a 3D render: "
                      f"living scales, skin and eyes. Only this one creature. {light}. {c.style} {c.period_s}No text, no "
                      f"lettering, no signature.")
        job = {"id": f"gen_mesh_{v['id']}", "seed": rng.randrange(1, 10**6), "size": [W, H],
               "refs": sheet_refs(c, oid), "prompt": prompt,
               "control": {"image": str(d_png), "type": "depth", "scale": 1.0},
               "init": {"image": str(init), "strength": round(rng.uniform(*strength), 2)}}
        meta = {"src": "mesh", "outfit": oid, "framing": f"{what}, {view}" + (", " + "in motion" if v.get("motion") else ""),
                "setting": light}
        jobs.append((job, meta))
    return jobs


def draw(jobs: list[dict], out_dir: Path) -> None:
    todo = [j for j in jobs if not Path(j["path"]).exists()]
    if not todo:
        return
    need_mem(15)
    jf = out_dir / f"jobs_{int(time.time())}.json"
    jf.write_text(json.dumps(todo, indent=1))
    t = time.time()
    r = subprocess.run([sys.executable, "-I", str(HERE / "qwen_image.py"), str(jf), str(out_dir)],
                       env=gpu_env(), capture_output=True, text=True)
    (out_dir / "draw.log").open("a").write(r.stderr[-20000:] + r.stdout)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("qwen_image failed")
    log(f"drew {len(todo)} pictures in {time.time() - t:.0f} s")


# --- judging ------------------------------------------------------------------------------------------

def _box(ans: str):
    m = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", ans or "")
    if not m:
        return None
    b = [int(x) / 1000 for x in m.groups()]
    return b if b[2] > b[0] and b[3] > b[1] else None


def _num(ans: str):
    m = re.search(r"\d+", ans or "")
    return int(m.group(0)) if m else None


def _yes(ans: str) -> bool:
    return (ans or "").strip().lower().startswith("yes")


def _pad(b, f, aspect_min=None):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - w * f, x1 + w * f, y0 - h * f, y1 + h * f
    return [max(0, x0), max(0, y0), min(1, x1), min(1, y1)]


class Scorer:
    def __init__(self):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        from vl_judge import Judge
        self.torch = torch
        self.j = Judge()
        self.ep = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.em = AutoModel.from_pretrained("facebook/dinov2-small").eval().to("cuda")

    def embed(self, im):
        with self.torch.inference_mode():
            x = self.ep(images=[im], return_tensors="pt").to("cuda")
            e = self.em(**x).last_hidden_state[:, 0].float()
        return (e / e.norm(dim=-1, keepdim=True))[0].cpu()

    def ask(self, im, q, n=8):
        return self.j.ask(im, q, n)


def crop(im, b):
    W, H = im.size
    x0, y0 = int(b[0] * W), int(b[1] * H)
    return im.crop((x0, y0, max(x0 + 8, int(b[2] * W)), max(y0 + 8, int(b[3] * H))))


def _claim(q: Path, cid: str) -> bool:
    try:
        os.close(os.open(q / f"{cid}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def judge(c: Char, cands: list, S: Scorer, ref_heads: dict, q: Path | None = None, tag: str = "",
          done: list | None = None) -> dict:
    """-> {candidate id: verdicts}, kept in judge.json. With *q* (the shared queue dir, cmd_dataset) two judges work
    one list: a candidate is judged by whoever claims it first (q/<id>.claim), its verdicts land in q/<id>.json."""
    from PIL import Image
    jf = c.out / "judge.json"
    J = json.loads(jf.read_text()) if jf.exists() else {}
    outfit_q = {oid: r["dress"].split(";")[0].split(":")[0] for oid, r in c.outfits.items() if oid}
    for k, cd in enumerate(cands):
        if cd["id"] in J or not Path(cd["path"]).exists():
            continue
        if q is not None:
            pf = q / f"{cd['id']}.json"
            if pf.exists():
                J[cd["id"]] = json.loads(pf.read_text())
                continue
            if not _claim(q, cd["id"]):
                continue
        t = time.time()
        im = Image.open(cd["path"]).convert("RGB")
        r = {"src": cd["src"]}
        if cd["src"] in ("pose", "mesh") and c.kind == "human":
            r["n_people"] = _num(S.ask(im, "How many people are in this picture? Answer with a number."))
        if cd["src"] == "mesh":       # a repainted model render must not keep the render's plastic look
            r["render_look"] = _yes(S.ask(im, "Does this picture look like a 3D computer render, a plastic or "
                                              "metal figurine, or a video game model? Answer yes or no.", 3))
        b = [0.0, 0.0, 1.0, 1.0] if cd["src"] in ("pose", "mesh") else S.j.box(im, f"{c.subject}")
        r["box"] = b
        if b is None:
            J[cd["id"]] = r
            if q is not None:
                _put(q, cd["id"], r)
            continue
        person = crop(im, b)
        hb = S.j.box(person, f"the head of {c.subject}")
        if hb:
            head = crop(person, _pad(hb, 0.15))
            r["head_px"] = min(head.size)
            r["like"] = round(float(S.embed(head) @ ref_heads["any"]), 4)
            if c.age:
                r["age"] = _num(S.ask(head, "How old does this person look, judging only by the face and skin? "
                                            "Answer with one number of years."))
        r["match"] = _yes(S.ask(person, f"Does this {'person' if c.kind == 'human' else 'creature'} match the description: {c.face}? Answer yes or no.", 3))
        if c.beard is not None and hb:
            r["beard"] = _yes(S.ask(head, "Does this person have a beard, stubble or a moustache? Answer yes or no.", 3))
        worn = []
        for oid, q in outfit_q.items():
            if _yes(S.ask(person, f"Is this person wearing {q}? Answer yes or no.", 3)):
                worn.append(oid)
        r["outfits"] = worn
        J[cd["id"]] = r
        log(f"{tag}judge {k + 1}/{len(cands)} {cd['id']}: {json.dumps(r)} ({time.time() - t:.1f} s)")
        if done is not None:
            done.append(cd["id"])
        if q is not None:
            _put(q, cd["id"], r)
        else:
            jf.write_text(json.dumps(J, indent=1))
    if q is None:
        jf.write_text(json.dumps(J, indent=1))
    return J


def _put(q: Path, cid: str, r: dict) -> None:
    tmp = q / f".{cid}.tmp"
    tmp.write_text(json.dumps(r))
    tmp.replace(q / f"{cid}.json")


def cmd_judge_worker(a) -> int:
    """The iGPU's half of a dataset's judging (cmd_dataset starts it while its XTX draws): claims candidates from
    cands.json as they exist, until the drawing is done and nothing is left to claim, or the dataset process ends."""
    c = Char(a.book, a.character)
    q = c.out / "jq"
    need_mem(15)
    S = Scorer()
    rh = ref_head_embs(c, S, save=False)
    while os.getppid() == a.parent:
        drawn = (q / "drawn").exists()
        did = []
        judge(c, json.loads((c.out / "cands.json").read_text()), S, rh, q=q, tag="[igpu] ", done=did)
        if not did:
            if drawn:
                break
            time.sleep(3)
    return 0


def ref_head_embs(c: Char, S: Scorer, save: bool = True) -> dict:
    from PIL import Image
    embs = []
    for oid in c.outfits:
        sh = c.sheet(oid)
        if not sh:
            continue
        im = Image.open(sh).convert("RGB")
        half = im.crop((0, 0, im.width // 2, im.height))
        hb = S.j.box(half, f"the head of {c.subject}") or [0.15, 0.05, 0.85, 0.75]
        h = crop(half, _pad(hb, 0.15))
        if save:
            h.save(c.out / "refs" / f"head_{c.V.slug(oid or 'base')}.png")
        embs.append(S.embed(h))
    import torch
    e = torch.stack(embs).mean(0)
    return {"any": e / e.norm()}


# --- selection + captions -----------------------------------------------------------------------------

def select(c: Char, cands: list, J: dict, a) -> list:
    from PIL import Image
    data = c.out / "data"
    if data.exists():
        shutil.rmtree(data)
    data.mkdir(parents=True)
    kept, rej = [], []
    # 10-21 knight_forged_v2 King (clean-shaven): the judge called 40 of 56 small heads (110-200 px) "bearded",
    # the checked sheet's own figure (143 px) among them -> 3 kept, no LoRA. A facial-hair verdict counts only on
    # heads at the size where it read the sheet right (the sheet passed the plan check); 144-200 px stayed noisy.
    sh = [J[cd["id"]] for cd in cands if cd["src"] == "sheet" and J.get(cd["id"]) and c.beard is not None
          and J[cd["id"]].get("beard") is not None]
    bad = max([r.get("head_px", 0) for r in sh if r["beard"] != c.beard], default=-1)
    beard_px = (min([r.get("head_px", 0) for r in sh if r["beard"] == c.beard and r.get("head_px", 0) > bad],
                    default=bad + 1) if bad >= 0 else 0)
    if beard_px:
        log(f"facial-hair verdicts ignored below {beard_px} px heads (the judge contradicts the sheet at that size)")

    # the same calibration for age (10-22 v3 King, "about 50, grey beard and hair"): the judge read the checked sheet
    # itself as 65-70 and all 54 pictures 65-75 -> 0 kept, no LoRA. The LoRA's job is to match the SHEET, so an age
    # within 10 years of what the judge reads on the sheet passes too.
    sa = sorted(J[cd["id"]]["age"] for cd in cands if cd["src"] == "sheet" and J.get(cd["id"])
                and isinstance(J[cd["id"]].get("age"), (int, float)))
    sheet_age = sa[len(sa) // 2] if sa else None
    if sheet_age is not None and c.age_ok(sheet_age) is False:
        log(f"!!! the judge reads the sheet as {sheet_age} (look: {c.age}): ages within 10 of the sheet pass")

    def age_wrong(r):
        if c.age_ok(r.get("age")) is not False:
            return False
        return not (sheet_age is not None and abs(r["age"] - sheet_age) <= 10)

    def beard_wrong(r):
        return (c.beard is not None and r.get("beard") is not None and r["beard"] != c.beard
                and r.get("head_px", 0) >= beard_px)

    for cd in cands:
        r = J.get(cd["id"])
        if not r:
            continue
        why = []
        if cd["src"] == "sheet" and r.get("box") is not None:
            if beard_wrong(r):
                why.append("sheet contradicts the look (facial hair)")
            if age_wrong(r):
                why.append(f"sheet contradicts the look (age {r.get('age')})")
        if cd["src"] != "sheet":
            if r.get("box") is None:
                why.append("not found")
            if cd["src"] in ("pose", "mesh") and c.kind == "human" and r.get("n_people") not in (1,):
                why.append(f"people {r.get('n_people')}")
            if beard_wrong(r):
                why.append("bearded" if r["beard"] else "no beard")
            if r.get("render_look"):
                why.append("3D-render look")
            if age_wrong(r):
                why.append(f"age {r.get('age')}")
            lim = getattr(a, "min_like_mesh", a.min_like) if cd["src"] == "mesh" else a.min_like
            if r.get("like") is None or r["like"] < lim:
                why.append(f"like {r.get('like')}")
            if not r.get("match"):
                why.append("no match")
            if r.get("head_px", 0) < a.min_head_px:
                why.append(f"head {r.get('head_px')} px")
        # outfit: what the generator asked must be what is worn; book pictures take what the judge saw
        oid = cd["outfit"]
        worn = r.get("outfits", [])
        if oid == "?":
            oid = worn[0] if worn else None       # several (armour / armour, helm off): the first listed
        elif cd["src"] != "sheet":
            same = {o for o in worn if c.outfits[o]["dress"].split(";")[0].split(":")[0].strip().lower()[:40]
                    in c.desc.lower()}
            if oid is None and set(worn) - same:
                why.append(f"wears {worn}, asked base")
            if oid is not None and oid not in worn:
                why.append(f"outfit {oid} not worn")
        if why:
            rej.append((cd["id"], why))
            continue
        im = Image.open(cd["path"]).convert("RGB")
        if not cd.get("whole"):
            b = r["box"]
            if cd["src"] == "book":
                im = crop(im, _pad(b, 0.12))
        n = len(kept)
        _save(im, data / f"{n:03d}.png", 1280)
        parts = [c.trigger, c.text(oid), cd.get("framing") or "", cd.get("setting") or ""]
        if cd["src"] == "book" and not cd.get("framing"):
            parts[2] = "a scene from the story"
        cap = ", ".join(p.strip().rstrip(".") for p in parts if p and p.strip()) + ", " + c.style.split(":")[0]
        (data / f"{n:03d}.txt").write_text(cap)
        kept.append({"n": n, "id": cd["id"], "src": cd["src"], "outfit": oid, "caption": cap,
                     "like": r.get("like"), "age": r.get("age")})
    (data / "meta.json").write_text(json.dumps({"trigger": c.trigger, "kept": kept, "rejected": rej}, indent=1))
    by = {}
    for k in kept:
        by[k["src"]] = by.get(k["src"], 0) + 1
    log(f"kept {len(kept)} ({by}), rejected {len(rej)}")
    per = {o: sum(1 for k in kept if k["outfit"] == o) for o in c.outfits}
    short = {str(o or "base"): n for o, n in per.items() if n < 6}
    if short:                               # said loudly: a LoRA that barely knows an outfit draws it from words alone
        log(f"!!! {c.name}: few training pictures for outfit(s) {short} (want 6+): their keys rely on the words")
    for i, w in rej[:200]:
        log(f"  rejected {i}: {', '.join(w)}")
    return kept


def contact_sheet(img_dir: Path, out: Path, cols: int = 8, cell: int = 220, labels: dict | None = None) -> None:
    from PIL import Image, ImageDraw
    ims = sorted(img_dir.glob("*.png"))
    rows = math.ceil(len(ims) / cols)
    S = Image.new("RGB", (cols * cell, rows * cell), (30, 30, 30))
    d = ImageDraw.Draw(S)
    for i, p in enumerate(ims):
        im = Image.open(p).convert("RGB")
        im.thumbnail((cell - 4, cell - 18))
        x, y = (i % cols) * cell, (i // cols) * cell
        S.paste(im, (x + 2, y + 2))
        d.text((x + 3, y + cell - 15), (labels or {}).get(p.stem, p.stem)[:34], fill=(255, 255, 0))
    S.save(out, quality=85)


def cmd_dataset(a) -> int:
    c = Char(a.book, a.character)
    c.out.mkdir(parents=True, exist_ok=True)
    (c.out / "refs").mkdir(exist_ok=True)
    rng = random.Random(a.seed)
    main_oid = a.outfit
    if main_oid not in (None, "*") and main_oid not in c.outfits:
        sys.exit(f"outfit {main_oid!r} not in the wardrobe: {[o for o in c.outfits if o]}")
    log(f"{c.name}: subject '{c.subject}', age {c.age}, outfits {list(c.outfits)}, trigger {c.trigger}")
    q, helper = c.out / "jq", None
    if a.igpu_judge and not a.only_jobs:
        # the iGPU judges the sheets and the book's pictures from the start, then the pose pictures as they are
        # drawn, while this card repairs the head and draws (10-31: judging was 2-4 min of a ~22 min build, after
        # the drawing, with the iGPU idle)
        early: list = []
        cand_sheets(c, early)
        if not a.no_book:
            cand_book(c, early)
        q.mkdir(exist_ok=True)
        for f in q.glob("*.claim"):
            if not (q / f"{f.stem}.json").exists():
                f.unlink()                                  # an ended run's unfinished claim
        (q / "drawn").unlink(missing_ok=True)
        _write(c.out / "cands.json", early)
        helper = subprocess.Popen([CHARLORA_PY, "-I", str(HERE / "charlora.py"), "judge-worker", a.book, a.character,
                                   "--parent", str(os.getpid())], env=_card_env("igpu"))
    if not a.only_jobs and c.kind == "human" and (c.beard is not None or c.age):
        need_mem(15)
        fixed = repair_head(c, Scorer)
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()       # the judge's ~17 GB stays CACHED otherwise: the draw OOMs on a 24 GB XTX (10-21)
    cands: list = []
    cand_sheets(c, cands)
    fx = c.out / "refs" / "head_fixed.png"
    if fx.exists():
        cands.append({"id": "sheet_fixed_head", "path": str(fx), "src": "sheet", "outfit": None, "whole": True,
                      "framing": "close-up portrait, head and shoulders, front view", "setting": "plain light background"})
    if not a.no_book:
        cand_book(c, cands)
    gen = []
    if a.n_pose:
        gen += gen_jobs_pose(c, a.n_pose, main_oid, rng)
    if a.n_mesh and (a.model or any(c.model(o) for o in c.outfits)):
        gen += gen_jobs_mesh(c, a.model, a.motion, a.n_mesh, main_oid, rng, (a.strength_lo, a.strength_hi))
    gdir = c.out / "cand"
    for job, meta in gen:
        job["path"] = str(gdir / f"{job['id']}.png")
        cands.append({"id": job["id"], "path": job["path"], "whole": True, **meta})
    (c.out / "gen_jobs.json").write_text(json.dumps([j for j, _ in gen], indent=1))
    _write(c.out / "cands.json", cands)
    if a.only_jobs:
        return 0
    draw([j for j, _ in gen], gdir)
    if helper:
        (q / "drawn").touch()
    need_mem(15)
    S = Scorer()
    rh = ref_head_embs(c, S)
    mine: list = []
    J = judge(c, cands, S, rh, q=q if helper else None, done=mine)
    if helper:
        try:
            helper.wait(timeout=900)
        except subprocess.TimeoutExpired:
            helper.terminate()
            helper.wait()
        for f in q.glob("*.claim"):
            if not (q / f"{f.stem}.json").exists():
                f.unlink()                                  # the helper ended mid-candidate: judged here
        J = judge(c, cands, S, rh, q=q, done=mine)
        (c.out / "judge.json").write_text(json.dumps(J, indent=1))
        log(f"judging: {len(mine)} of {len(J)} on this card, the rest on the iGPU or earlier (helper exit "
            f"{helper.returncode})")
    del S
    kept = select(c, cands, J, a)
    labels = {f"{k['n']:03d}": f"{k['src']} {k['outfit'] or 'base'} {k['like']}" for k in kept}
    contact_sheet(c.out / "data", c.out / "data" / "sheet.jpg", labels=labels)
    log(f"sheet: {c.out / 'data' / 'sheet.jpg'}")
    return 0


def _write(p: Path, obj) -> None:
    """Whole or not at all (the iGPU judge reads cands.json while this process rewrites it)."""
    tmp = p.with_name(f".{p.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(p)


def cmd_select(a) -> int:
    c = Char(a.book, a.character)
    cands = json.loads((c.out / "cands.json").read_text())
    J = json.loads((c.out / "judge.json").read_text())
    kept = select(c, cands, J, a)
    labels = {f"{k['n']:03d}": f"{k['src']} {k['outfit'] or 'base'} {k['like']}" for k in kept}
    contact_sheet(c.out / "data", c.out / "data" / "sheet.jpg", labels=labels)
    return 0


# --- training ----------------------------------------------------------------------------------------

TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer",
           "img_mlp.out"]


def _fa_attention():
    """The DiT's unmasked attention calls through AMD's Triton flash kernel, forward AND backward (an
    autograd Function over fa_amd fwd/bwd; checked against SDPA on the iGPU: fwd err 1e-3, dq err 4e-3)."""
    import torch
    sys.path.insert(0, str(FA_AMD))
    from fa_amd import interface_v2 as fa
    import diffusers.models.transformers.transformer_qwenimage21 as m

    class FA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v):
            sc = q.shape[-1] ** -0.5
            out, lse = fa.fwd(q, k, v, None, None, 0.0, sc, False, -1, -1, 0.0, False)[:2]
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.sc = sc
            return out

        @staticmethod
        def backward(ctx, do):
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            fa.bwd(do.contiguous(), q, k, v, out, lse, dq, dk, dv, None, 0.0, ctx.sc, False, -1, -1, 0.0, False)
            return dq, dk, dv

    orig = m.dispatch_attention_fn

    def attn(query, key, value, attn_mask=None, dropout_p=0.0, backend=None, parallel_config=None, **kw):
        if attn_mask is None and not dropout_p and query.dtype in (torch.bfloat16, torch.float16) and not kw:
            with torch.cuda.device(query.device):
                return FA.apply(query.contiguous(), key.contiguous(), value.contiguous())
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, backend=backend,
                    parallel_config=parallel_config, **kw)
    m.dispatch_attention_fn = attn


ASPECTS = (0.5625, 0.75, 1.0, 1.333, 1.778)


def bucket(w: int, h: int, res: int) -> tuple[int, int]:
    """A few fixed shapes per resolution: on the iGPU (unified memory) every new tensor shape costs
    allocator churn; free-form buckets ran 8.6 s/step vs 3.7 s for one repeated shape (10-19)."""
    ar = min(ASPECTS, key=lambda x: abs(math.log(x * h / w)))
    return round(res * math.sqrt(ar) / 32) * 32, round(res / math.sqrt(ar) / 32) * 32


def cache_items(c: Char, res_list: list[int]) -> list:
    """Captions -> Qwen3-VL embeddings, images -> VAE latents (per resolution), cached in lora/cache."""
    import torch
    from PIL import Image
    cdir = c.out / "lora" / "cache"
    cdir.mkdir(parents=True, exist_ok=True)
    # keyed by picture name only: entries older than the dataset belong to a previous one (10-22 v3: a redrawn
    # dataset would have trained on the old pictures' latents + captions)
    mj = c.out / "data" / "meta.json"
    if mj.exists():
        for f_ in cdir.glob("*.pt"):
            if f_.stat().st_mtime < mj.stat().st_mtime:
                f_.unlink()
    data = c.out / "data"
    items = sorted(data.glob("*.png"))
    todo_t = [p for p in items if not (cdir / f"{p.stem}_txt.pt").exists()]
    todo_v = [(p, r) for p in items for r in res_list if not (cdir / f"{p.stem}_{r}.pt").exists()]
    dev = "cuda"
    if todo_t:
        need_mem(25)
        from diffusers import QwenImage21Pipeline
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        te = Qwen3VLForConditionalGeneration.from_pretrained(MODEL / "text_encoder", torch_dtype=torch.bfloat16,
                                                             attn_implementation="sdpa").to(dev).eval()
        pipe = QwenImage21Pipeline(scheduler=None, vae=None, text_encoder=te,
                                   processor=Qwen3VLProcessor.from_pretrained(MODEL / "processor"), transformer=None)
        for p in todo_t:
            cap = (data / f"{p.stem}.txt").read_text().strip()
            with torch.inference_mode():
                emb, mask, pad = pipe.encode_prompt(prompt=cap, device=torch.device(dev))
            if not torch.isfinite(emb).all():
                raise RuntimeError(f"NaN text embedding for {p.name}")
            torch.save({"emb": emb.cpu(), "mask": None if mask is None else mask.cpu(), "pad": pad.cpu(),
                        "cap": cap}, cdir / f"{p.stem}_txt.pt")
        del pipe, te
        torch.cuda.empty_cache()
        log(f"encoded {len(todo_t)} captions")
    if todo_v:
        from diffusers import AutoencoderKLQwenImage21
        from diffusers.image_processor import VaeImageProcessor
        vae = AutoencoderKLQwenImage21.from_pretrained(MODEL / "vae", torch_dtype=torch.bfloat16).to(dev).eval()
        vae.enable_tiling(tile_sample_min_height=512, tile_sample_min_width=512,
                          tile_sample_stride_height=384, tile_sample_stride_width=384)
        ip = VaeImageProcessor(vae_scale_factor=16)
        mean = torch.tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1).to(dev)
        std = torch.tensor(vae.config.latents_std).view(1, 64, 1, 1, 1).to(dev)
        for p, r in todo_v:
            im = Image.open(p).convert("RGB")
            w, h = bucket(im.width, im.height, r)
            s = max(w / im.width, h / im.height)
            im = im.resize((math.ceil(im.width * s), math.ceil(im.height * s)), Image.LANCZOS)
            x0, y0 = (im.width - w) // 2, (im.height - h) // 2
            im = im.crop((x0, y0, x0 + w, y0 + h))
            out = []
            for flip in (False, True):
                ii = im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im
                v = ip.preprocess(ii, height=h, width=w)
                v = torch.cat([v, torch.ones_like(v[:, :1])], 1).unsqueeze(2).to(dev, torch.bfloat16)
                with torch.inference_mode():
                    z = vae.encode(v).latent_dist.mode().float()
                z = (z - mean) / std
                out.append(z[0, :, 0].cpu())                            # [64, h/16, w/16]
            torch.save({"lat": out[0], "lat_flip": out[1], "w": w, "h": h}, cdir / f"{p.stem}_{r}.pt")
        del vae
        torch.cuda.empty_cache()
        log(f"encoded {len(todo_v)} latents")
    return [p.stem for p in items]


def cmd_train(a) -> int:
    import torch
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.preferred_blas_library("cublaslt")
    c = Char(a.book, a.character)
    ldir = Path(a.lora_dir) if a.lora_dir else c.out / "lora"     # --lora-dir: a recipe A/B beside the book's LoRA
    ldir.mkdir(parents=True, exist_ok=True)
    res_list = [int(r) for r in a.res.split(",")]
    res_w = [float(x) for x in a.res_weights.split(",")] if a.res_weights else [1.0] * len(res_list)
    stems = cache_items(c, res_list)
    cdir = c.out / "lora" / "cache"                                  # the dataset's cache (cache_items) stays shared
    caps = {s: torch.load(cdir / f"{s}_txt.pt") for s in stems}
    lats = {(s, r): torch.load(cdir / f"{s}_{r}.pt") for s in stems for r in res_list}
    need_mem(30)
    from diffusers import QwenImage21Transformer2DModel
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file
    dev = "cuda"
    t0 = time.time()
    tr = QwenImage21Transformer2DModel.from_pretrained(MODEL / "transformer", torch_dtype=torch.bfloat16).to(dev)
    tr.requires_grad_(False)
    tr.enable_gradient_checkpointing()
    tr.add_adapter(LoraConfig(r=a.rank, lora_alpha=a.alpha or a.rank, init_lora_weights="gaussian",
                              target_modules=TARGETS))
    params = [p for p in tr.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    _fa_attention()
    log(f"DiT + LoRA r{a.rank} loaded in {time.time() - t0:.0f} s: {sum(p.numel() for p in params) / 1e6:.1f} M "
        f"trainable, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.99), fused=True)
    start = 0
    # resume only from a checkpoint of THIS dataset that loads (10-22 v3: Hulda's step-100 .pt was cut short by a
    # full disk and every retrain died on it; a retry's new dataset must not continue the old one's training)
    meta_t = (c.out / "data" / "meta.json").stat().st_mtime if (c.out / "data" / "meta.json").exists() else 0
    st = None
    for p_ in sorted(ldir.glob("step_*.pt"), reverse=True) if not a.fresh else []:
        if p_.stat().st_mtime < meta_t:
            log(f"not resuming from {p_.name}: older than the dataset")
            break
        try:
            st = torch.load(p_, map_location="cpu")
            ck = [p_]
            break
        except Exception as exc:                          # noqa: BLE001 - a truncated file
            log(f"!!! {p_.name} unreadable ({type(exc).__name__}): set aside")
            p_.rename(p_.with_suffix(".pt.bad"))
    if st is not None:
        sd = st["lora"]
        for n, p in tr.named_parameters():
            if n in sd:
                p.data.copy_(sd[n].to(dev))
        opt.load_state_dict(st["opt"])
        start = st["step"]
        log(f"resumed at step {start} from {ck[-1].name}")
    sched_cfg = json.loads((MODEL / "scheduler" / "scheduler_config.json").read_text())
    rng = random.Random(a.seed + start)
    g = torch.Generator(dev).manual_seed(a.seed + start)
    meta = {"r": a.rank, "lora_alpha": a.alpha or a.rank, "trigger": c.trigger, "character": c.name,
            "targets": TARGETS}

    def save(step):
        sd = {n: p.detach() for n, p in tr.named_parameters() if p.requires_grad}
        st = get_peft_model_state_dict(tr)
        st = {"transformer." + k.replace(".default", ""): v.to(torch.bfloat16).contiguous().cpu() for k, v in st.items()}
        out = ldir / f"step_{step:05d}.safetensors"
        save_file(st, str(out), metadata={"lora_adapter_metadata": json.dumps(meta), "format": "pt"})
        torch.save({"lora": {k: v.cpu() for k, v in sd.items()}, "opt": opt.state_dict(), "step": step},
                   ldir / f"step_{step:05d}.pt")
        for old in sorted(ldir.glob("step_*.pt"))[:-1]:
            old.unlink()                                       # keep one resume state; all .safetensors
        log(f"saved {out.name}")

    lf = (ldir / "train.log").open("a")
    tr.train()
    losses = []
    t_last = time.time()
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes else None
    for step in range(start, a.steps):
        lr = a.lr * min(1.0, (step + 1) / a.warmup)
        for gp in opt.param_groups:
            gp["lr"] = lr
        s = rng.choice(stems)
        r = rng.choices(res_list, res_w)[0]
        L = lats[(s, r)]
        x0 = (L["lat_flip"] if (a.flip and rng.random() < 0.5) else L["lat"]).to(dev)
        hh, ww = x0.shape[1:]
        x0 = x0.reshape(64, hh * ww).T[None]                     # [1, L, 64] (the pipeline's packing)
        cp = caps[s]
        emb = cp["emb"].to(dev)
        mask = None if cp["mask"] is None else cp["mask"].to(dev)
        n = x0.shape[1]
        mu = calculate_shift(n, sched_cfg.get("base_image_seq_len", 256), sched_cfg.get("max_image_seq_len", 4096),
                             sched_cfg.get("base_shift", 0.5), sched_cfg.get("max_shift", 1.15))
        u = torch.sigmoid(torch.randn((), generator=g, device=dev))
        sig = math.exp(mu) / (math.exp(mu) + (1 / float(u) - 1))         # exponential time shift, as inference
        eps = torch.randn(x0.shape, generator=g, device=dev)
        xt = (1 - sig) * x0 + sig * eps
        pad = torch.cat([cp["pad"].to(dev), cp["pad"].new_ones(1, n // 4).to(dev)], dim=1)
        # bf16 autocast: the fp32 LoRA weights (the optimizer's master copy) multiply in bf16 instead of fp32 GEMMs
        # with tiny tiles + casts (10-31 bench on an XTX: 1.52 -> 1.33 s/step, same loss)
        with torch.autocast("cuda", torch.bfloat16):
            pred = tr(hidden_states=xt.to(torch.bfloat16), timestep=torch.tensor([sig], device=dev, dtype=torch.bfloat16),
                      encoder_hidden_states=emb, encoder_hidden_states_mask=mask, img_shapes=[[(1, hh, ww)]],
                      img_mask=pad, return_dict=False)[0][:, -n:]
        loss = torch.nn.functional.mse_loss(pred.float(), (eps - x0).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss))
        if not math.isfinite(losses[-1]):
            sys.exit(f"loss is {losses[-1]} at step {step}")
        if (step + 1) % 10 == 0:
            dt = (time.time() - t_last) / 10
            t_last = time.time()
            msg = (f"step {step + 1}/{a.steps} loss {sum(losses[-10:]) / 10:.4f} lr {lr:.2e} {dt:.2f} s/step "
                   f"res {r} mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB avail {mem_avail_gb():.0f} GB")
            log(msg)
            lf.write(msg + "\n")
            lf.flush()
            if mem_avail_gb() < 15:
                save(step + 1)
                sys.exit("MemAvailable < 15 GB: saved and stopped")
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            save(step + 1)
        if deadline and time.time() > deadline:
            save(step + 1)
            log("time budget reached: saved, stopping (resume by running train again)")
            return 3
    return 0


def add_trigger(prompt: str, name: str, trigger: str) -> str:
    """'... Hulda (a woman ...' -> '... chr_hulda Hulda (a woman ...': the first mention carries the trigger."""
    i = prompt.find(name)
    return prompt if i < 0 or trigger in prompt else prompt[:i] + trigger + " " + prompt[i:]


def score_images(c_by_name: dict, paths: list[str], S: Scorer, heads: dict) -> dict:
    """Per picture, per character: box, head likeness (vs the sheet heads), age, outfits worn, match."""
    from PIL import Image
    out = {}
    for p in paths:
        im = Image.open(p).convert("RGB")
        res = {}
        for name, c in c_by_name.items():
            r = {}
            b = S.j.box(im, f"{c.subject}")
            r["box"] = b
            if b:
                person = crop(im, b)
                hb = S.j.box(person, f"the head of {c.subject}")
                if hb:
                    head = crop(person, _pad(hb, 0.15))
                    r["like"] = round(float(S.embed(head) @ heads[name]), 4)
                    if c.age:
                        r["age"] = _num(S.ask(head, "How old does this person look, judging only by the face and "
                                                    "skin? Answer with one number of years."))
                    if c.beard is not None:
                        r["beard"] = _yes(S.ask(head, "Does this person have a beard, stubble or a moustache? "
                                                      "Answer yes or no.", 3))
                r["outfits"] = [oid for oid, row in c.outfits.items() if oid and _yes(S.ask(
                    person, f"Is this person wearing {row['dress'].split(';')[0].split(':')[0]}? Answer yes or no.", 3))]
                r["match"] = _yes(S.ask(person, f"Does this {'person' if c.kind == 'human' else 'creature'} match the description: {c.face}? Answer yes or no.", 3))
            res[name] = r
        out[p] = res
        log(f"score {Path(p).name}: {json.dumps(res)}")
    return out


def cmd_ab(a) -> int:
    """Draw a jobs file twice (no LoRA / with the characters' LoRAs + triggers), same seeds and controls,
    and score every character in every picture. --lora Name=/abs/lora.safetensors[=scale]."""
    from PIL import Image, ImageDraw
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(a.jobs).read_text())
    loras = {}
    for l in a.lora:
        name, rest = l.split("=", 1)
        p, _, sc = rest.partition("=")
        loras[name] = (str(Path(p).expanduser()), float(sc or 1.0))
    chars = {n: Char(a.book, n) for n in (a.score or list(loras))}
    jobs, pairs = [], []
    for j in base:
        j0 = dict(j, id=j["id"] + "_base", path=str(out / f"{j['id']}_base.png"))
        j0.pop("loras", None)
        j1 = dict(j, id=j["id"] + "_lora", path=str(out / f"{j['id']}_lora.png"))
        pr = j["prompt"]
        for n, (p, sc) in loras.items():
            pr = add_trigger(pr, n, chars[n].trigger if n in chars else Char(a.book, n).trigger)
        regions = {n: [float(v) for v in r.split(",")] for n, r in (x.split("=", 1) for x in a.region)}
        j1["prompt"], j1["loras"] = pr, [[p, sc] + ([regions[n]] if n in regions else []) for n, (p, sc) in loras.items()]
        jobs += [j0, j1]
        pairs.append((j0["path"], j1["path"], j["id"]))
    (out / "ab_jobs.json").write_text(json.dumps(jobs, indent=1))
    draw(jobs, out)
    need_mem(15)
    S = Scorer()
    heads = {}
    for n, c in chars.items():
        e = []
        for f in sorted((c.out / "refs").glob("head_*.png")):
            e.append(S.embed(Image.open(f).convert("RGB")))
        import torch
        e = torch.stack(e).mean(0)
        heads[n] = e / e.norm()
    sc = score_images(chars, [p for pr in pairs for p in pr[:2]], S, heads)
    (out / "scores.json").write_text(json.dumps(sc, indent=1))
    # side-by-side sheet: base | lora per row, scores written under each
    rows = []
    for p0, p1, jid in pairs:
        ims = [Image.open(p).convert("RGB") for p in (p0, p1)]
        h = 360
        ims = [im.resize((round(im.width * h / im.height), h)) for im in ims]
        row = Image.new("RGB", (sum(i.width for i in ims) + 10, h + 40), (20, 20, 20))
        d = ImageDraw.Draw(row)
        x = 0
        for p, im in zip((p0, p1), ims):
            row.paste(im, (x, 0))
            txt = "  ".join(f"{n}: like {r.get('like')} age {r.get('age')}"
                            + (f" beard {'Y' if r['beard'] else 'N'}" if r.get('beard') is not None else "")
                            + f" {','.join(r.get('outfits') or []) or '-'}"
                            for n, r in sc[p].items())
            d.text((x + 4, h + 4), ("LoRA  " if p == p1 else "base  ") + txt[:150], fill=(255, 255, 0))
            x += im.width + 10
        rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows)), (20, 20, 20))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(out / "ab_sheet.jpg", quality=85)
    log(f"sheet {out / 'ab_sheet.jpg'}")
    return 0


# --- one call for the pipeline (bookcaster flow3d.assets) ----------------------------------------------

STEPS = 400                 # 10-21 A/B: 800 overfit (training backgrounds leak); 400 + pick_checkpoint (every 100)
LEGACY_STEPS = (800,)       # LoRAs built at these step counts stay valid (not retrained for the recipe change)
TOOL_VERSION = "charlora-2"          # bump when the dataset/training recipe changes: every LoRA rebuilds
# charlora-2 (10-27): the book's own pictures only when drawn from the CURRENT look and showing the person alone
# (17 of Watson's 39 training pictures were older drawings - the clean-shaven 10-15 look - and crops of crowds put a
# Holmes-like face in Roylott's set); no pose with a prop the person doesn't have (a sword for Roylott); likeness >= 0.4
CHARLORA_PY = os.environ.get("CHARLORA_PY", str(HOME / "tts-ab" / ".venv-charlora-igpu" / "bin" / "python"))
# the charlora-igpu torch is gfx1151-only: on an XTX the same steps run in the Qwen-Image venv (gfx1100, same
# diffusers/peft; 10-20: transformer 14 GB bf16 + LoRA + checkpointing fits 24 GB)
XTX_PY = os.environ.get("CHARLORA_XTX_PY", str(HOME / "tts-ab" / ".venv-qwenimage" / "bin" / "python"))


def card_py(card: str) -> str:
    return CHARLORA_PY if card == "igpu" else XTX_PY
CANON_MOTIONS = Path(os.environ.get("CHARLORA_MOTIONS", REPO / "eval" / "motion" / "t1"))


def _fsig(p) -> list:
    p = Path(p) if p else None
    return [str(p), p.stat().st_size, int(p.stat().st_mtime)] if p and p.exists() else [str(p), None]


def _same_attempt(c: Char, key: str) -> bool:
    """A retry of a FAILED attempt with the same inputs keeps its drawings and judge answers (they are named and
    cached per picture): only the extra pose drawings are drawn and judged (user 10-21: a retry must not cost a
    whole new dataset). The wipe below is for changed inputs only."""
    f = c.out / "attempt.json"
    try:
        return json.loads(f.read_text()).get("hash") == key
    except (OSError, ValueError):
        return False


def data_hash(c: Char) -> str:
    """What the TRAINING PICTURES are made from (the look, outfits, sheets, models, style) -- not the training
    recipe: a retrain with other steps / resolution reuses the dataset (10-21 research: 15 min saved per retrain)."""
    return input_hash(c, None)


def hash_current(c: "Char", h: str | None, steps: int | None = None, shape_h: str | None = None) -> bool:
    """A build.json hash is current for this character at the recipe's steps or a legacy step count - or its SHAPE
    hash is (the models only re-painted: 10-22, a paint fix would otherwise retrain every LoRA for ~45 min)."""
    ss = {steps or STEPS} | set(LEGACY_STEPS)
    return (bool(h) and h in {input_hash(c, s) for s in ss}) or \
        (bool(shape_h) and shape_h in {shape_hash(c, s) for s in ss})


def shape_hash(c: "Char", steps: int | None) -> str:
    """input_hash with each outfit's model SHAPE (shape.glb beside it) in place of the painted model file."""
    parts = [TOOL_VERSION, steps, c.desc, c.style, c.kind]
    for oid, row in c.outfits.items():
        m = c.model(oid)
        shp = str(Path(m).parent / "shape.glb") if m else None
        parts += [oid, row.get("dress"), _fsig(c.sheet(oid)), _fsig(shp) if shp and Path(shp).exists() else _fsig(m)]
    return "s" + hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:15]


def stamp_shape(book) -> list:
    """Record the shape hash in every CURRENT build.json (before a paint-only model change). -> names stamped."""
    out = []
    import registry as RG
    R = RG.Book(book)
    for bj in Path(book, "work", "video", "charlora").glob("*/build.json"):
        try:
            d = json.loads(bj.read_text())
            name = R.person(bj.parent.name, narrator=False)          # the folder, any spelling (10-27)
            if not name:
                continue
            c = Char(str(book), name)
            if hash_current(c, d.get("hash")) and not d.get("shape_hash"):
                d["shape_hash"] = shape_hash(c, STEPS)
                bj.write_text(json.dumps(d, indent=1))
                out.append(name)
        except Exception:                              # noqa: BLE001 - a stamp is best-effort
            continue
    return out


def input_hash(c: Char, steps: int | None) -> str:
    """Everything a LoRA is made from: the look, every outfit's words, sheet and model, the style, the
    recipe. A changed look/outfit/sheet/model -> a new hash -> rebuilt; otherwise reused."""
    parts = [TOOL_VERSION, steps, c.desc, c.style, c.kind]
    for oid, row in c.outfits.items():
        parts += [oid, row.get("dress"), _fsig(c.sheet(oid)), _fsig(c.model(oid))]
    return hashlib.sha1(json.dumps(parts, default=str).encode()).hexdigest()[:16]


def _card_env(card: str) -> dict:
    env = dict(os.environ)
    env.pop("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", None)
    i = subprocess.run([str(HOME / "bin" / "hip-index"), card], capture_output=True, text=True).stdout.strip()
    if not i:
        raise RuntimeError(f"no GPU '{card}' (hip-index)")
    env["HIP_VISIBLE_DEVICES"] = i
    if card == "igpu":
        env["HSA_CU_MASK"] = f"{i}:0-38"          # one CU stays free for the desktop (box rule)
    else:
        env.pop("HSA_CU_MASK", None)
    return env


def _canon_motions() -> list[Path]:
    """Canonical (rig-free) motion clips: the *_s1.npz in CHARLORA_MOTIONS whose .json has no "rig"."""
    out = []
    for p in sorted(CANON_MOTIONS.glob("*_s1.npz")) if CANON_MOTIONS.exists() else []:
        j = p.with_suffix(".json")
        try:
            if j.exists() and "rig" not in json.loads(j.read_text()):
                out.append(p)
        except ValueError:
            pass
    one = {}
    for p in out:                                  # one clip per action (block, hammer, lunge, sword, ...)
        one.setdefault(p.stem.split("_")[0], p)
    return list(one.values())[:6]


def _rig_motions(c: Char, env: dict) -> list[str]:
    """Retarget the canonical clips onto each outfit's rig (cached in charlora/<slug>/motions)."""
    if c.kind != "human":
        return []
    out = []
    for blend in sorted({m for o in c.outfits if (m := c.model(o))}):
        rig = Path(blend).with_name("rig.json")
        if not rig.exists():
            continue
        tag = Path(blend).parent.name
        for cm in _canon_motions():
            base = c.out / "motions" / f"{cm.stem}_{tag}"
            if not base.with_suffix(".npz").exists():
                base.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run([CHARLORA_PY, "-I", str(HERE / "motion_retarget.py"), str(cm), str(rig), str(base)],
                               capture_output=True, env=env)
            if base.with_suffix(".npz").exists():
                out.append(str(base))
    return out


def build_all(book: str, characters: list[str] | None = None, card: str = "igpu", force: bool = False,
              steps: int = STEPS, n_pose: int = 40, n_mesh: int = 12, log_fn=log) -> dict:
    """A character LoRA for every character of the book (or *characters*), cached by input hash.

    -> {name: {"lora": path to the .safetensors, "trigger": "chr_<slug>", "sheet": dataset contact sheet,
               "kind", "hash", "kept"}} (an entry {"error": ...} when one fails; the others still build).
    Each build = `charlora.py dataset` (sheets + the book's pictures + pose views + model views of every
    wardrobe outfit, the look-contradiction filter: age, beard / clean-shaven, outfit, likeness) then
    `charlora.py train`, as subprocesses in the charlora venv on *card* ("igpu" | "05" | "65"; the caller
    owns any card lock). ~1-1.5 h dataset + ~1.5 h training per character on the iGPU. Use the LoRA
    REGIONALLY in multi-person frames: qwen_image job "loras": [[path, 1.0, [x0, y0, x1, y1]]]."""
    probe = Char(book, next(iter(json.loads((Path(book).expanduser() if Path(book).expanduser().exists()
                                             else HOME / "audiobooks" / book).joinpath(
        "work", "video", "look.json").read_text())["characters"])))
    # one build per PERSON (10-27: look.json's "Dr. Watson" beside "Doctor Watson" was a second build of one man)
    names = list(dict.fromkeys(probe.R.person(n, n, narrator=False) for n in characters)) if characters \
        else probe.R.people()
    out = {}
    for name in names:
        try:
            c = Char(book, name)
            # no sheet = no reference face: fail at once, not after 50 min of drawing (10-22: the King)
            if c.kind == "human" and not all(c.sheet(o) for o in c.outfits):
                raise RuntimeError("no reference sheet (bookcaster plan draws one per cast member)")
            key = input_hash(c, steps)
            bj = c.out / "build.json"
            prev = json.loads(bj.read_text()) if bj.exists() else {}
            if not force and hash_current(c, prev.get("hash"), steps, prev.get("shape_hash")) and \
                    Path(prev.get("lora", "")).exists():
                out[name] = prev
                log_fn(f"charlora {name}: up to date ({Path(prev['lora']).name})")
                continue
            env = _card_env(card)
            if mem_avail_gb() < 20:
                raise RuntimeError(f"MemAvailable {mem_avail_gb():.0f} GB < 20 GB")
            dkey = data_hash(c)
            same_data = (not force and prev.get("data_hash") == dkey and (c.out / "data" / "meta.json").exists()
                         and len(json.loads((c.out / "data" / "meta.json").read_text()).get("kept", [])) >= 12)
            if same_data:
                log_fn(f"charlora {name}: training pictures unchanged ({dkey}): retrain only")
                if (c.out / "lora").exists():
                    shutil.rmtree(c.out / "lora.prev", ignore_errors=True)
                    (c.out / "lora").rename(c.out / "lora.prev")
            elif force or (prev.get("hash") != key and not _same_attempt(c, key)):  # inputs changed: no stale candidates
                if (c.out / "lora").exists():             # keep the old LoRA (10-20: a rebuild wiped Hulda's)
                    shutil.rmtree(c.out / "lora.prev", ignore_errors=True)
                    (c.out / "lora").rename(c.out / "lora.prev")
                for d in ("cand", "data", "refs", "jq"):
                    shutil.rmtree(c.out / d, ignore_errors=True)
                for f in ("judge.json", "cands.json", "gen_jobs.json"):
                    (c.out / f).unlink(missing_ok=True)
            c.out.mkdir(parents=True, exist_ok=True)
            (c.out / "attempt.json").write_text(json.dumps({"hash": key, "started": time.strftime("%F %T")}))
            mot = _rig_motions(c, env)
            cmd = [card_py(card), "-I", str(HERE / "charlora.py"), "dataset", book, name, "--n-pose", str(n_pose),
                   "--n-mesh", str(n_mesh), "--min-like", "0.4", "--strength-lo", "0.8", "--strength-hi", "0.9"]
            if card != "igpu":
                cmd += ["--igpu-judge"]
            if len(c.outfits) > 1:
                cmd += ["--outfit", "*"]
            if mot:
                cmd += ["--motion", *mot]
            t = time.time()
            with open(c.out / "build.log", "a") as lf:
                if not same_data:
                    r = subprocess.run(cmd, env=env, stdout=lf, stderr=lf)
                    if r.returncode:
                        raise RuntimeError(f"dataset failed ({c.out / 'build.log'})")
                meta = json.loads((c.out / "data" / "meta.json").read_text())
                if len(meta["kept"]) < 12:
                    raise RuntimeError(f"only {len(meta['kept'])} pictures passed the checks")
                log_fn(f"charlora {name}: dataset {len(meta['kept'])} pictures in {(time.time() - t) / 60:.0f} min")
                t = time.time()
                r = subprocess.run([card_py(card), "-I", str(HERE / "charlora.py"), "train", book, name,
                                    "--steps", str(steps), "--res", "512,640", "--res-weights", "0.6,0.4",
                                    "--save-every", "100"],
                                   env=env, stdout=lf, stderr=lf)
                if r.returncode:
                    raise RuntimeError(f"train failed ({c.out / 'build.log'})")
                # the best checkpoint by likeness, not the last (early stopping by score, ~2-3 min)
                r = subprocess.run([card_py(card), "-I", str(HERE / "charlora.py"), "pick", book, name],
                                   env=env, stdout=lf, stderr=lf)
            use_ = c.out / "lora" / "USE"
            lora = c.out / "lora" / (use_.read_text().strip() if use_.exists() and r.returncode == 0
                                     else f"step_{steps:05d}.safetensors")
            res = {"lora": str(lora), "trigger": c.trigger, "sheet": str(c.out / "data" / "sheet.jpg"),
                   "kind": c.kind, "hash": key, "shape_hash": shape_hash(c, steps), "data_hash": dkey,
                   "kept": len(meta["kept"]),
                   "built": time.strftime("%F %T")}
            bj.write_text(json.dumps(res, indent=1))
            out[name] = res
            log_fn(f"charlora {name}: trained in {(time.time() - t) / 60:.0f} min -> {lora}")
        except Exception as exc:                          # noqa: BLE001 - one character must not stop the rest
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            log_fn(f"charlora {name}: FAILED {exc}")
    return out


def register(book: str, name: str, lora: str, steps: int = STEPS) -> None:
    """Record a LoRA built by hand (dataset + train run separately) in build.json, so build_all reuses it."""
    c = Char(book, name)
    (c.out / "build.json").write_text(json.dumps(
        {"lora": str(lora), "trigger": c.trigger, "sheet": str(c.out / "data" / "sheet.jpg"), "kind": c.kind,
         "hash": input_hash(c, steps), "built": time.strftime("%F %T")}, indent=1))


PICK_SCENES = ["Portrait of {n}, {d} {where}",
               "{n} at work, three-quarter view, full figure. {n} is {d} {where}",
               "Medium shot of {n} walking outdoors at dusk. {n} is {d}",
               "Close-up of {n}'s face, looking up in surprise, warm firelight. {n} is {d}"]


def pick_checkpoint(book: str, name: str) -> dict:
    """Early stopping by SCORE (10-21 A/B: Edric's likeness 0.79 at 200 steps, 0.70 at 800 - the 800-step LoRA pulled
    its training backgrounds into new scenes; Hulda peaked at 600): the same 4 scenes drawn with every saved
    checkpoint (one model load), scored for likeness against the sheet heads; the best -> lora/USE (char_lora reads
    it), pick.json. Ties (within 0.02) go to the earlier checkpoint. -> {"use", "scores"}"""
    import torch
    c = Char(book, name)
    ld = c.out / "lora"
    cks = sorted(ld.glob("step_*.safetensors"), key=lambda p: int(re.sub(r"\D", "", p.stem)))
    if not cks:
        raise RuntimeError(f"{name}: no checkpoints in {ld}")
    out = c.out / "pick"
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for ck in cks:
        for i, sc_ in enumerate(PICK_SCENES):
            pr = sc_.format(n=name, d=c.desc.rstrip("."), where="") + f". {c.style} {c.period_s}No text."
            jobs.append({"id": f"{ck.stem}_{i}", "prompt": add_trigger(pr, name, c.trigger), "seed": 11 + 7 * i,
                         "size": [896, 896], "loras": [[str(ck), 1.0]], "path": str(out / f"{ck.stem}_{i}.png")})
    draw(jobs, out)
    need_mem(15)
    from PIL import Image
    S = Scorer()
    e = torch.stack([S.embed(Image.open(f).convert("RGB")) for f in sorted((c.out / "refs").glob("head_*.png"))]).mean(0)
    sc = score_images({name: c}, [j["path"] for j in jobs if Path(j["path"]).exists()], S, {name: e / e.norm()})
    per = {}
    for ck in cks:
        L = [(sc.get(str(out / f"{ck.stem}_{i}.png")) or {}).get(name, {}).get("like") for i in range(len(PICK_SCENES))]
        L = [x for x in L if x is not None]
        per[ck.name] = round(sum(L) / len(L), 4) if L else 0.0
    top = max(per.values())
    use = next(k for k in (ck.name for ck in cks) if per[k] >= top - 0.02)
    (ld / "USE").write_text(use)
    (c.out / "pick.json").write_text(json.dumps({"use": use, "scores": per, "at": time.strftime("%F %T")}, indent=1))
    log(f"{name}: checkpoint likeness {per} -> USE {use}")
    return {"use": use, "scores": per}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    t = sub.add_parser("train")
    s = sub.add_parser("select")
    sh = sub.add_parser("sheet")
    for p in (d, t, s, sh):
        p.add_argument("book")
        p.add_argument("character")
        p.add_argument("--seed", type=int, default=1)
    for p in (d, s):
        p.add_argument("--min-like", type=float, default=0.45)
        p.add_argument("--min-head-px", type=int, default=64)
        p.add_argument("--min-like-mesh", type=float, default=0.2,
                       help="model views (odd angles, profiles, backs) score low on head likeness by nature")
    d.add_argument("--outfit", default=None, help="wardrobe id most new views wear (default: the usual look; "
                                                   "'*' = every outfit equally)")
    d.add_argument("--n-pose", type=int, default=40)
    d.add_argument("--n-mesh", type=int, default=0)
    d.add_argument("--model", help="rigged.blend of the character (default: stage/characters/index.json's, "
                                   "per outfit)")
    d.add_argument("--motion", nargs="*", default=[], help="RigMotion bases on that rig (pose variety)")
    d.add_argument("--strength-lo", type=float, default=0.6)
    d.add_argument("--strength-hi", type=float, default=0.75)
    d.add_argument("--no-book", action="store_true")
    d.add_argument("--only-jobs", action="store_true", help="write the jobs/candidates, draw nothing")
    d.add_argument("--igpu-judge", action="store_true", help="the iGPU judges alongside (a build on an XTX)")
    jw = sub.add_parser("judge-worker", help="(cmd_dataset's) the iGPU's share of a dataset's judging")
    jw.add_argument("book")
    jw.add_argument("character")
    jw.add_argument("--parent", type=int, required=True)
    t.add_argument("--steps", type=int, default=1200)
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--alpha", type=int, default=None)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--wd", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=50)
    t.add_argument("--res", default="512,768", help="training resolutions (area res^2, aspect kept)")
    t.add_argument("--res-weights", default="0.7,0.3")
    t.add_argument("--flip", action="store_true", help="random horizontal flips (off: faces are not symmetric)")
    t.add_argument("--save-every", type=int, default=200)
    t.add_argument("--max-minutes", type=float, default=0)
    t.add_argument("--fresh", action="store_true")
    t.add_argument("--lora-dir", default=None, help="checkpoints here instead of the character's lora/ (recipe A/B)")
    ab = sub.add_parser("ab", help="same jobs with / without character LoRAs, scored")
    ab.add_argument("book")
    ab.add_argument("--jobs", required=True)
    ab.add_argument("--out", required=True)
    ab.add_argument("--lora", action="append", default=[], metavar="Name=/abs/lora.safetensors[=scale]")
    ab.add_argument("--region", action="append", default=[], metavar="Name=x0,y0,x1,y1",
                    help="apply that LoRA only inside this 0-1 box (regional LoRA: no bleed onto others)")
    ab.add_argument("--score", nargs="*", help="characters to score (default: the LoRA ones)")
    pk = sub.add_parser("pick", help="score every saved checkpoint, the best -> lora/USE (no training)")
    pk.add_argument("book")
    pk.add_argument("character")
    bl = sub.add_parser("build", help="build_all: every character (or the named ones), cached by input hash")
    bl.add_argument("book")
    bl.add_argument("characters", nargs="*")
    bl.add_argument("--card", default="igpu")
    bl.add_argument("--force", action="store_true")
    bl.add_argument("--steps", type=int, default=STEPS)
    bl.add_argument("--n-pose", type=int, default=40, help="pose drawings for the dataset (a retry draws more)")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        res = build_all(a.book, a.characters or None, a.card, a.force, a.steps, n_pose=a.n_pose)
        print(json.dumps(res, indent=1))
        return 1 if any("error" in v for v in res.values()) else 0     # a failure is the CALLER's to see (10-21)
    if a.cmd == "ab":
        return cmd_ab(a)
    if a.cmd == "pick":
        pick_checkpoint(a.book, a.character)
        return 0
    if a.cmd == "dataset":
        return cmd_dataset(a)
    if a.cmd == "judge-worker":
        return cmd_judge_worker(a)
    if a.cmd == "train":
        return cmd_train(a)
    if a.cmd == "select":
        return cmd_select(a)
    c = Char(a.book, a.character)
    contact_sheet(c.out / "data", c.out / "data" / "sheet.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
