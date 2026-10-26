"""Every character of a book as a rigged, textured 3D model (any book; the book's look.json is the data).

    P=~/tts-ab/.venv-pose/bin/python
    $P -I tools/characters.py plan  <book>                 the cast x outfits -> stage/characters/index.json
    $P -I tools/characters.py build <book> [--only NAME ...] [--card 05] [--seeds 2] [--force STEP]
    $P -I tools/characters.py qa    <book> [--only NAME ...] (QA sheets only, CPU/Blender)

One MODEL per (character, outfit): the usual look plus every wardrobe row of the book's plans (an outfit is a
different silhouette: leathers vs plate). Steps, each resumable (an existing output is kept unless --force):
  refs   the character's reference sheet (charlora.Char.sheet: head half | figure half) -> figure half + cut-out
  draw   Qwen-Image A-pose FRONT then BACK drawings (pose control = stage_templates/apose_*_skel.png, the figure
         cut-out as reference, the character's identity LoRA if <book>/work/video/charlora/<slug>/lora exists,
         its trigger in the prompt); N seeds, the front with the most A-pose-like DWPose skeleton wins
  cut    rembg RGBA cut-outs (Hunyuan3D needs alpha: a grey key built a relief slab)
  shape  Hunyuan3D-2.1 (tools/blockout_shape.py) on the front cut-out -> shape.glb
  rig    tools/blockout_model.py: normalise to the character's height, front+back vertex paint, DWPose joints,
         bone-heat skin -> rigged.blend + rig.json
  qa     a turnaround line-up (0/45/90/180 deg) and 4 test poses rendered through tools/blockout.py with the
         model, the front render's face likeness + age vs the sheet (tools/face_likeness.py) -> qa.jpg, qa.json
Creatures (a look with no human features: "dragon", "four clawed legs", ...) are not built here (their rig is
tools/blockout_creature_bl.py's); register an existing rig with  plan --model "NAME=/abs/rig_dir".
GPU steps take --card's lock (/tmp/gpu-<card>.lock) per call and release it after (each call < 5 min), after
yielding to a waiting LTX gaprun and waiting for MemAvailable > 15 GB.
The index (<book>/work/video/stage/characters/index.json) is what the planner reads: per character the kind,
height, and per outfit the model; per chapter the wardrobe rows (from which picture an outfit is worn).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
POSE_PY = os.path.expanduser("~/tts-ab/.venv-pose/bin/python")
QWEN_PY = os.environ.get("QWEN_IMAGE_PY", os.path.expanduser("~/tts-ab/.venv-qwenimage/bin/python"))
HY3D_PY = os.path.expanduser("~/tts-ab/.venv-hy3d/bin/python")
TPL = HERE / "stage_templates"
MV_WEIGHTS = Path(os.environ.get("HY3D_MV_WEIGHTS", Path.home() / "models" / "hunyuan3d-2mv" / "hunyuan3d-dit-v2-mv"))
VIEWS = ("front", "back", "left", "right")      # A-pose drawings per outfit (left = the person's left side, face to
                                                # the picture's left)
CREATURE_RX = r"\b(dragon|wyrm|beast|creature|monster|serpent|wolf|horse|bird|no human features|clawed legs|wings)\b"
APOSE = ("Full body from head to feet, {view}, standing straight in an A-pose: both arms held out straight and away "
         "from the body at about 40 degrees, hands open and empty, palms facing down, legs slightly apart{look}. "
         "Character turnaround model sheet for a 3D modeller: even soft flat lighting, no cast shadows, plain flat "
         "light grey background, nothing else in the picture. {style}No text, no lettering, no signature.")


def log(msg):
    sys.stderr.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    sys.stderr.flush()


# ------------------------------------------------------------------------------------------------ the cast
def vdir_of(book: str) -> Path:
    p = Path(book).expanduser()
    root = p if p.exists() else Path.home() / "audiobooks" / book
    return root / "work" / "video" if (root / "work" / "video").exists() else root


def height_of(desc: str) -> float:
    """Standing height (m) from the look's words: woman 1.65, man 1.8, child by age, the old a little shorter."""
    d = desc.lower()
    m = re.search(r"\b(?:of|aged)\s+(?:about\s+|around\s+)?(\d{1,3})\b", d)
    age = int(m.group(1)) if m else None
    female = bool(re.search(r"\b(woman|girl|lady|queen|princess|she|her|maid|mother|grandmother|old woman)\b", d))
    h = 1.65 if female else 1.8
    if age is not None and age < 16:
        h = min(h, 0.75 + 0.06 * age)
    if age is not None and age >= 65:
        h -= 0.05
    if re.search(r"\b(tall|towering)\b(?!\s+(hat|boots|collar|helm))", d):
        h += 0.08
    if re.search(r"\b(short|small|slight)\s+(man|woman|stature|build|frame|figure)\b|\b(stooped|hunched)\b", d):
        h -= 0.08
    return round(h, 2)


def key_of(slug: str, oid) -> str:
    return slug if not oid else f"{slug}__{re.sub(r'[^a-z0-9]+', '_', oid.lower()).strip('_')}"


def plan(book: str, models: dict | None = None) -> dict:
    import charlora as L
    vd = vdir_of(book)
    look = json.loads((vd / "look.json").read_text())
    out_root = vd / "stage" / "characters"
    out_root.mkdir(parents=True, exist_ok=True)
    ip = out_root / "index.json"
    old = json.loads(ip.read_text()) if ip.exists() else {}
    idx = {"book": str(vd), "characters": {}}
    for name, desc in (look.get("characters") or {}).items():
        c = L.Char(book, name)
        creature = bool(re.search(CREATURE_RX, desc.lower()))
        ent = {"kind": "creature" if creature else "human", "slug": c.slug, "age": c.age,
               "height": None if creature else height_of(desc), "outfits": {}, "wardrobe": {}}
        prev = (old.get("characters") or {}).get(name, {})
        seen_txt = {}
        for oid in c.outfits:
            k = key_of(c.slug, oid)
            st = strip_held(c.text(oid)).lower().split("wearing", 1)[-1].strip()
            if oid and st in seen_txt:                 # the same body once the held items are props: alias
                ent["outfits"][oid] = dict(ent["outfits"][seen_txt[st]], id=oid, alias_of=seen_txt[st])
                continue
            seen_txt[st] = oid or ""
            sheet = c.sheet(oid)
            mk = out_root / k / "creature_sheet.json"          # a creature's own head sheet wins
            if not oid and mk.exists() and Path(json.loads(mk.read_text())["sheet"]).exists():
                sheet = Path(json.loads(mk.read_text())["sheet"])
            o = {"key": k, "id": oid, "dir": str(out_root / k), "sheet": str(sheet) if sheet else None,
                 "text": c.text(oid)}
            pm = (prev.get("outfits") or {}).get(oid or "", {}).get("model")
            if pm:
                o["model"] = pm
            ent["outfits"][oid or ""] = o
        for ch, pl in c.plans.items():
            rows = (pl.get("wardrobe") or {}).get(name)
            if rows:
                ent["wardrobe"][ch] = [{"from": r["from"], "id": r["id"]} for r in rows]
        if models and name in models:
            for o in ent["outfits"].values():
                o["model"] = models[name]
        idx["characters"][name] = ent
    ip.write_text(json.dumps(idx, indent=1))
    return idx


def model_for(index: dict, name: str, chapter: str | None = None, picture: int | None = None) -> dict | None:
    """The model a character wears in a chapter at a picture index (the wardrobe row in force), or None."""
    ent = (index.get("characters") or {}).get(name)
    if not ent:
        return None
    oid = ""
    rows = (ent.get("wardrobe") or {}).get(chapter or "", [])
    if picture is not None:
        for r in sorted(rows, key=lambda r: r["from"]):
            if picture >= r["from"]:
                oid = r["id"]
    o = ent["outfits"].get(oid) or ent["outfits"].get("")
    return dict(o, kind=ent["kind"], height=ent.get("height")) if o else None


_PARA = {}


def outfit_at(vd, chapter: str, name: str, t: float):
    """The wardrobe id a character wears at time t (s) of a chapter's recording (None = the usual look):
    the paragraph read at t (the chapter's text + timeline, as tts_reader.video times its pictures) and the
    plan's wardrobe row in force at that paragraph."""
    from tts_reader import video as V
    vd = Path(vd)
    key = (str(vd), chapter)
    if key not in _PARA:
        root = vd.parent.parent
        text = (root / "chapters" / f"{chapter}.txt").read_text()
        tl = json.loads((root / "work" / f"{chapter}.timeline.json").read_text())
        paras = V.paragraphs(text)
        plan = json.loads((vd / chapter / "plan.json").read_text())
        plan = made_outfits_shift(plan, paras)
        _PARA[key] = (V.paragraph_times(paras, tl), plan)
    starts, plan = _PARA[key]
    r = V.outfit(plan, name, V.para_at(starts, float(t)))
    return r["id"] if r else None


MAKE_RX = r"\b(made|makes|making|forged|forges|forging|hammered|crafted|sewed|sewn|wove|woven|built|stitched)\b"
LEAVE_RX = r"\b(walked|rode|set out|set off|left|departed|journeyed|travell?ed|marched|went (north|south|east|west|out|away))\b"


def made_outfits_shift(plan, paras):
    """A wardrobe row that starts where the story MAKES that outfit ("He made the breastplate first ... then the
    helm") is a timing slip: the maker wears it only once it is done; the start moves to the next paragraph where
    the character sets out (walked / rode / left ...), within 15 paragraphs. Returns a corrected copy."""
    import copy as _copy
    plan = _copy.deepcopy(plan)
    for name, rows in (plan.get("wardrobe") or {}).items():
        for r in rows:
            f0 = int(r["from"])
            if f0 >= len(paras):
                continue
            dress_words = [w for w in re.findall(r"[a-z]{4,}", r.get("dress", "").lower())
                           if w not in ("with", "full", "dark", "plain", "faint", "shimmer", "covering", "from", "neck")]
            if re.search(MAKE_RX, paras[f0].lower()) and any(w in paras[f0].lower() for w in dress_words):
                for k in range(f0 + 1, min(f0 + 16, len(paras))):
                    if re.search(LEAVE_RX, paras[k].lower()):
                        r["from_planned"], r["from"] = f0, k
                        r["why"] = f"made in paragraph {f0}; worn from paragraph {k} (sets out)"
                        break
    return plan


def model_at(index: dict, name: str, chapter: str, t: float) -> dict | None:
    """The character's model (index entry + kind/height) for a moment at time t of a chapter."""
    ent = (index.get("characters") or {}).get(name)
    if not ent:
        return None
    oid = outfit_at(index["book"], chapter, name, t) or ""
    o = ent["outfits"].get(oid) or ent["outfits"].get("")
    return dict(o, kind=ent["kind"], height=ent.get("height"), outfit=oid or None) if o else None


# ------------------------------------------------------------------------------------------------ GPU turn
CARDS = ("05", "65")          # the GPU cards (~/bin/hip-index names); "any" = whichever lock is free first
_CARD_OF = {}


def card_of(fd) -> str:
    """The card a gpu_turn() lock belongs to."""
    return _CARD_OF.get(id(fd))


def gpu_turn(card: str = "any"):
    """Yield to a waiting LTX two-card run (gaprun.sh: let it start and finish, <= 15 min), take the card's
    lock, wait for MemAvailable > 15 GB. Returns the open lock file (close it to release)."""
    def pg(pat):
        """PIDs whose PROGRAM (argv[0], or argv[1] for an interpreter: bash x.sh, python x.py) is pat: a
        plain `pgrep -f` also matches every shell whose command line merely mentions the name."""
        r = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True)
        out = []
        for x in r.stdout.split():
            pid = int(x)
            if pid == os.getpid():
                continue
            try:
                argv_ = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            progs = [a.decode(errors="ignore") for a in argv_[:3] if a and not a.startswith(b"-")]
            if any(os.path.basename(a) == pat for a in progs):
                out.append(pid)
        return out

    def ltx():
        for pid in pg("ltx_split.py"):
            try:
                if "python" in open(f"/proc/{pid}/comm").read():
                    return True
            except OSError:
                pass
        return False
    if pg("gaprun.sh"):
        t0, started = time.time(), False
        while time.time() - t0 < 900:
            if ltx():
                started = True
            elif started or not pg("gaprun.sh"):
                break
            time.sleep(10)
    cards = CARDS if card in (None, "any") else ([card] if isinstance(card, str) else list(card))
    fd = None
    while fd is None:                                  # the FIRST free card's lock (try-lock each), else wait
        for c in cards:
            f_ = open(f"/tmp/gpu-{c}.lock", "w")
            try:
                fcntl.flock(f_, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f_.close()
                continue
            fd = f_
            break
        if fd is None:
            if len(cards) == 1:
                fd = open(f"/tmp/gpu-{cards[0]}.lock", "w")
                fcntl.flock(fd, fcntl.LOCK_EX)
                c = cards[0]
            else:
                time.sleep(5)
    _CARD_OF[id(fd)] = c
    while True:
        av = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 1048576
        if av >= 15:
            break
        time.sleep(10)
    return fd


def hip_env(card: str) -> dict:
    idx = subprocess.run([os.path.expanduser("~/bin/hip-index"), card], capture_output=True, text=True).stdout.strip()
    return dict(os.environ, HIP_VISIBLE_DEVICES=idx)


# ------------------------------------------------------------------------------------------------ steps
def step_refs(o):
    """Figure half of the sheet (right half) + its cut-out on transparent (the drawer's reference)."""
    d = Path(o["dir"])
    d.mkdir(parents=True, exist_ok=True)
    half, cut = d / "ref_figure.png", d / "ref_cut.png"
    if not half.exists():
        from PIL import Image
        im = Image.open(o["sheet"]).convert("RGB")
        w, h = im.size
        im.crop((w // 2, 0, w, h)).save(half) if w > h * 1.2 else im.save(half)
    if not cut.exists():
        rembg_rgba(half, cut, grey=True)
    return half, cut


def rembg_rgba(src, dst, grey=False):
    """rembg isnet cut-out; grey=True composites onto mid-grey (a reference), else keeps RGBA (for Hunyuan)."""
    code = ("import sys; from rembg import remove, new_session; from PIL import Image\n"
            "s=new_session('isnet-general-use'); im=Image.open(sys.argv[1]).convert('RGB'); o=remove(im, session=s)\n"
            "if sys.argv[3]=='1':\n  bg=Image.new('RGBA', o.size, (128,128,128,255)); bg.alpha_composite(o); o=bg.convert('RGB')\n"
            "o.save(sys.argv[2])")
    subprocess.run([POSE_PY, "-I", "-c", code, str(src), str(dst), "1" if grey else "0"], check=True,
                   capture_output=True, text=True)


def lora_of(vd: Path, slug: str):
    ld = vd / "charlora" / slug / "lora"
    st = sorted(ld.glob("step_*.safetensors")) if ld.exists() else []
    return str(st[-1]) if st else None


HELD_RX = (r"[;,.]?\s*(?:and\s+)?(?:(?:strapped|tied|slung|hanging)\s+(?:to|from|over|on)\s+(?:his|her|their)\s+[\w ]+?"
           r"(?:\s+is|\s+are)?\b[^;.]*|(?:holding|holds|carrying|carries|clutching|gripping|wielding)\b[^;.]*|"
           r"(?:a|an|the)?\s*[\w -]*\b(?:in|under) (?:his|her|their) (?:left |right )?(?:hand|hands|arm)\b[^;.]*)")


def strip_held(text: str) -> str:
    """The person WITHOUT held / strapped items (a shield strapped to the forearm, a sword in hand, a helm
    under the arm): those are props (movable, can be dropped or handed over), never baked into the body mesh."""
    t = re.sub(HELD_RX, "", text, flags=re.I)
    t = re.sub(r"\s*,\s*(?=[;.]|$)", "", t)
    t = re.sub(r"\s{2,}", " ", t).strip().rstrip(";,")
    return t


def helm_on(text: str) -> bool:
    """An outfit whose helm is worn (covers the head), not off / absent."""
    t = text.lower()
    return bool(re.search(r"\bhelm(et)?\b", t)) and not re.search(
        r"helm(et)? (is )?off|no helm|without (a |his |her )?helm|bareheaded|helm(et)? (held )?under (his|her) arm"
        r"|helm(et)? (is )?(removed|taken off)|helm(et)? (\w+ )?not worn", t)   # 10-21 Edric "helm removed" got a helm


def helm_off_words(text: str) -> str:
    """'with the helm removed and not worn' NAMES a helm (a named thing gets drawn): said as bare-headed."""
    return re.sub(r"(?i)(,\s*)?(with )?(the |his |her )?helm(et)? (is )?(removed|off|taken off)( and not worn)?",
                  ", bare-headed, the face and hair uncovered", text)


def apose_prompt(o, name, view, trigger, style):
    txt = strip_held(o["text"])
    if not helm_on(txt):
        txt = helm_off_words(txt)
    if o.get("id") and helm_on(txt):
        # the face words would win over the helm: give only the outfit, and say the helm hides the head
        dress = txt.split(", wearing ", 1)[-1]
        txt = (f"wearing {dress}; the helm is ON and closed, covering the whole head: no face and no hair visible, "
               f"only the dark eye slit")
    who = f"{trigger}, {name}" if trigger else name
    look = ", looking straight at the viewer" if view == "front" else ""
    side_ = {"left": "LEFT", "right": "RIGHT"}.get(view)
    vtxt = ("front view" if view == "front" else
            (f"exact SIDE VIEW, a full profile: the person faces the {side_} edge of the picture, we see one side of "
             f"the head (ear, cheek, the hair at the side and back) and one side of the body, the arms hanging out at "
             f"the sides; the SAME person as in image 2 turned sideways: the same face, hair colour, length and style, "
             f"the same clothes, age and build") if side_ else
            "BACK VIEW seen from directly behind: we see the back of the head (hair, no face) and the back of the "
            "body; the SAME person as in image 2 turned around: the same hair colour, length and style, the same age "
            "and build")
    return (f"{who}, {txt}, exactly as in image 1, with nothing in the hands and nothing strapped to the arms. "
            + APOSE.format(view=vtxt, look=look, style=style))


def apose_score(png) -> float:
    """How A-pose-like and complete a drawing is (DWPose body): all 17 points seen, wrists well out from the
    hips, ankles apart, one person. Higher is better; < 0 = unusable."""
    code = r'''
import sys, json, numpy as np
from rtmlib import Body
from PIL import Image
im = np.asarray(Image.open(sys.argv[1]).convert("RGB"))[:, :, ::-1]
b = Body(mode="balanced", backend="onnxruntime", device="cpu")
kp, sc = b(im)
if len(kp) == 0: print(-1); sys.exit()
order = np.argsort([-s.mean() for s in sc]); k, s = kp[order[0]], sc[order[0]]
n_people = sum(1 for x in sc if x.mean() > 0.4)
seen = float((s > 0.4).mean())
sh = np.linalg.norm(k[5] - k[6]) + 1e-6
out = (abs(k[9][0] - k[11][0]) + abs(k[10][0] - k[12][0])) / 2 / sh    # wrist-hip horizontal / shoulder width
feet = 1.0 if (s[15] > 0.4 and s[16] > 0.4) else 0.0
print(round(seen * 2 + min(out, 1.2) + feet - (n_people > 1) * 2, 3))
'''
    r = subprocess.run([POSE_PY, "-I", "-c", code, str(png)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip().splitlines()[-1])
    except Exception:                                    # noqa: BLE001
        return -1.0


def run_qwen(jobs, wdir, card):
    todo = [j for j in jobs if not os.path.exists(j["path"])]
    if not todo:
        return
    import threading
    jp = Path(wdir) / f"jobs_{int(time.time())}_{os.getpid()}_{threading.get_ident() % 100000}.json"   # 2 cards at once
    jp.write_text(json.dumps(todo, indent=1))
    fd = gpu_turn(card)
    card = card_of(fd) or card
    try:
        log(f"qwen: {len(todo)} pictures on card {card}")
        r = subprocess.run([QWEN_PY, "-I", str(HERE / "qwen_image.py"), str(jp), str(wdir)], capture_output=True,
                           text=True, env=hip_env(card), timeout=840)
        if r.returncode:
            sys.stderr.write(r.stderr[-2000:])
    finally:
        fd.close()


def step_draw(entries, vd, card, seeds, style, force=False):
    """A-pose fronts (all characters x outfits x seeds in ONE Qwen load), pick the best front by DWPose, then
    the backs (front as 2nd reference) in a second load."""
    fronts = []
    for name, o in entries:
        d = Path(o["dir"])
        if force:
            for p in d.glob("apose_*.png"):
                p.unlink()
        _, cut = step_refs(o)
        lora = lora_of(vd, o["slug"])
        trig = f"chr_{o['slug']}" if lora else ""
        for k in range(seeds):
            seed = 3 + 5 * k
            j = {"id": f"{o['key']}_front_s{seed}", "prompt": apose_prompt(o, name, "front", trig, style), "seed": seed,
                 "path": str(d / f"apose_front_s{seed}.png"), "size": [896, 1344], "refs": [str(cut)],
                 "control": {"image": str(TPL / "apose_front_skel.png"), "type": "pose", "scale": 0.8}}
            if lora:
                j["loras"] = [[lora, 0.8]]
            fronts.append(j)
    wd = vd / "stage" / "characters" / "_qwen"
    wd.mkdir(parents=True, exist_ok=True)
    run_qwen(fronts, wd, card)
    backs = []
    for name, o in entries:
        d = Path(o["dir"])
        best = d / "apose_front.png"
        if not best.exists() or force:
            cands = [(apose_score(p), p) for p in sorted(d.glob("apose_front_s*.png"))]
            cands = [c for c in cands if c[0] >= 0]
            if not cands:
                log(f"{o['key']}: no usable front drawing")
                continue
            sc, p = max(cands)
            shutil.copy(p, best)
            (d / "apose_front.json").write_text(json.dumps({"chosen": str(p), "scores": {str(q): s for s, q in cands}}))
            log(f"{o['key']}: front {p.name} (A-pose score {sc})")
        lora = lora_of(vd, o["slug"])
        trig = f"chr_{o['slug']}" if lora else ""
        # back + BOTH SIDES (10-21: painted from front + back only, every head's sides were the front picture stretched
        # sideways = the melted talk faces at a 3/4 camera); the chosen front as image 2 keeps them on-model
        for v in VIEWS[1:]:
            j = {"id": f"{o['key']}_{v}", "prompt": apose_prompt(o, name, v, trig, style), "seed": o.get("side_seed", 3),
                 "path": str(d / f"apose_{v}.png"), "size": [896, 1344], "refs": [str(d / "ref_cut.png"), str(best)],
                 "control": {"image": str(TPL / f"apose_{v}_skel.png"), "type": "pose", "scale": 0.8}}
            if lora:
                j["loras"] = [[lora, 0.8]]
            backs.append(j)
    run_qwen(backs, wd, card)
    side_views_checked(entries, wd, card, style)
    side_views_geometric(entries)


def side_views_geometric(entries) -> list:
    """The side views measured, not asked (10-22: every cast member's "right" drawing was a three-quarter view the
    vision judge passed as a profile; projected onto the model it put the face on the side of the head and every
    head failed side QA). tools/profile_check.py: eye separation vs ear-to-nose; a side that is no profile, or faces
    the wrong way, is replaced by the OTHER side mirrored when that one is a profile, and what was built from it (cut,
    head detail, shape, model, rig, QA) is set aside to be rebuilt. Marker apose_<v>.geo (the measurement).
    -> [(key, view)] replaced"""
    from PIL import Image, ImageOps
    todo = []
    for name, o in entries:
        d = Path(o["dir"])
        for v in ("left", "right"):
            if (d / f"apose_{v}.png").exists() and not (d / f"apose_{v}.geo").exists():
                todo.append((o, v, d / f"apose_{v}.png"))
    if not todo:
        return []
    r = subprocess.run([POSE_PY, "-I", str(HERE / "profile_check.py"), *[str(t[2]) for t in todo]],
                       capture_output=True, text=True)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("JSON ")), None)
    if line is None:
        log(f"!!! side views not measured (profile_check failed): {r.stderr[-300:]}")
        return []
    res = json.loads(line[5:])
    done = []
    for o, v, pth in todo:
        d = Path(o["dir"])
        m = res.get(str(pth)) or {}
        ok = bool(m.get("profile")) and m.get("facing") == v
        if ok:
            (d / f"apose_{v}.geo").write_text(json.dumps(m))
            continue
        other = "right" if v == "left" else "left"
        mo = res.get(str(d / f"apose_{other}.png"))
        if mo is None and (d / f"apose_{other}.geo").exists():
            mo = json.loads((d / f"apose_{other}.geo").read_text())
        if not (mo and mo.get("profile") and mo.get("facing") == other):
            log(f"!!! {o['key']}: the {v} view is no {v} profile ({m.get('ratio')}, facing {m.get('facing')}) and the "
                f"{other} one cannot stand in: kept, flagged")
            continue
        arc = d / f"side_geo_{time.strftime('%H%M%S')}"
        arc.mkdir()
        for f in [pth, *(d.glob(f"apose_{v}_*")), *(d.glob(f"head_{v}*")), d / "shape.glb", d / "rigged.blend",
                  d / "model.blend", d / "rig.json", d / "qa.jpg", d / "qa.json", d / "front.png"]:
            if f.exists():
                shutil.move(str(f), arc / f.name)
        ImageOps.mirror(Image.open(d / f"apose_{other}.png")).save(pth)
        (d / f"apose_{v}.ok").write_text(f"mirrored from {other} (geometric check)")
        (d / f"apose_{v}.geo").write_text(json.dumps({**mo, "facing": v, "mirrored_from": other, "was": m}))
        log(f"{o['key']}: the {v} view was no profile (eye/ear ratio {m.get('ratio')}): the {other} profile mirrored, "
            f"its model set aside to rebuild")
        done.append((o["key"], v))
    return done


def side_views_checked(entries, wd, card, style, tries=2):
    """The side views must BE profiles (10-21: Edric's 'right' came out a front A-pose; projected onto his right side
    it painted the front picture there). The vision judge checks each; a failed side is redrawn with a new seed,
    then MIRRORED from the other side (a face is near-symmetric), else dropped (front + back paint only)."""
    from PIL import Image, ImageOps
    for k in range(tries + 1):
        items = []
        for name, o in entries:
            d = Path(o["dir"])
            for v in ("left", "right"):
                p = d / f"apose_{v}.png"
                if p.exists() and not (d / f"apose_{v}.ok").exists():
                    items.append({"id": f"{o['key']}|{v}", "image": str(p), "ask": {
                        "side": f"Answer yes or no: is this person seen exactly from the SIDE (a profile, only one eye "
                                f"visible), facing the {v.upper()} edge of the picture?"}})
        if not items:
            return
        ans = vl_ask(items)
        redo = []
        for it in items:
            key, v = it["id"].rsplit("|", 1)
            name, o = next((n_, o_) for n_, o_ in entries if o_["key"] == key)
            d = Path(o["dir"])
            if _yes((ans.get(it["id"]) or {}).get("side")):
                (d / f"apose_{v}.ok").write_text("profile")
                continue
            log(f"{key}: the {v} view is not a {v} profile (try {k + 1})")
            (d / f"apose_{v}.png").replace(d / f"apose_{v}_bad{k}.png")
            (d / f"apose_{v}_rgba.png").unlink(missing_ok=True)
            if k < tries - 1:
                lora = lora_of(Path(o["dir"]).parents[2], o.get("slug", "")) if o.get("slug") else None
                trig = f"chr_{o['slug']}" if lora else ""
                j = {"id": f"{key}_{v}_r{k}", "prompt": apose_prompt(o, name, v, trig, style), "seed": 3 + 11 * (k + 1),
                     "path": str(d / f"apose_{v}.png"), "size": [896, 1344],
                     "refs": [str(d / "ref_cut.png"), str(d / "apose_front.png")],
                     "control": {"image": str(TPL / f"apose_{v}_skel.png"), "type": "pose", "scale": 1.0}}
                if lora:
                    j["loras"] = [[lora, 0.8]]
                redo.append(j)
        if redo:
            run_qwen(redo, wd, card)
            continue
        break
    for name, o in entries:                      # still missing: the other side, mirrored
        d = Path(o["dir"])
        for v, other in (("left", "right"), ("right", "left")):
            if not (d / f"apose_{v}.png").exists() and (d / f"apose_{other}.ok").exists():
                ImageOps.mirror(Image.open(d / f"apose_{other}.png")).save(d / f"apose_{v}.png")
                (d / f"apose_{v}.ok").write_text(f"mirrored from {other}")
                log(f"{o['key']}: the {v} view mirrored from the {other} profile")


def sheet_guard(o, name=None, style="") -> bool:
    """A model is built FROM its sheet: every derived file (cut-outs, A-pose drawings, shape, rig, QA) is set aside to
    <dir>/stale_<stamp>/ when the sheet changed since the last build or the build has no sheet signature (10-21:
    Edric's models stayed from his first sheet after the sheet check redrew it younger; every step is
    exists-gated). -> True when it set files aside."""
    import hashlib
    d = Path(o["dir"])
    sig = hashlib.sha1(Path(o["sheet"]).read_bytes()).hexdigest()[:16]
    f = d / "sheet.sig"
    # the A-pose drawing's PROMPT is an input too (10-21: the helm fix changed Edric's helm-removed prompt, the
    # sheet stayed): compared without the LoRA trigger (a LoRA trained later must not rebuild every model).
    # First sight: the prompt the last draw really used (its _qwen job file).
    pf = d / "prompt.sig"
    if name is not None:
        cur = apose_prompt(o, name, "front", "", style)
        if not pf.exists():
            used = None
            for jf in sorted((d.parent / "_qwen").glob("jobs_*.json"), key=lambda p: p.stat().st_mtime):
                for j in json.loads(jf.read_text()):
                    if str(j.get("id", "")).startswith(f"{o['key']}_front_s"):
                        used = re.sub(r"^chr_\w+, ", "", j["prompt"])
            if used is not None and d.exists():
                pf.write_text(hashlib.sha1(used.encode()).hexdigest()[:16])
        psig = hashlib.sha1(cur.encode()).hexdigest()[:16]
        if pf.exists() and pf.read_text().strip() != psig and f.exists():
            f.write_text("prompt changed")        # falls through to the set-aside below
    if f.exists() and f.read_text().strip() == sig:
        if name is not None:
            pf.write_text(hashlib.sha1(apose_prompt(o, name, "front", "", style).encode()).hexdigest()[:16])
        return False
    old = [x for x in d.iterdir() if not x.name.startswith("stale_") and x.name != "sheet.sig"] if d.exists() else []
    if old:
        arc = d / f"stale_{time.strftime('%m%d-%H%M%S')}"
        arc.mkdir(parents=True)
        for x in old:
            shutil.move(str(x), arc / x.name)
        o.pop("model", None)
        log(f"{o['key']}: sheet or drawing prompt changed since the last build ({len(old)} file(s) set aside to {arc.name}): rebuilt")
    d.mkdir(parents=True, exist_ok=True)
    f.write_text(sig)
    if name is not None:
        pf.write_text(hashlib.sha1(apose_prompt(o, name, "front", "", style).encode()).hexdigest()[:16])
    return bool(old)


HEAD_HD = os.environ.get("BOOKCASTER_HEAD_HD", "1") == "1"


def head_box(png: Path) -> tuple | None:
    """The head's square box in an A-pose drawing: the top 15 % of the figure (alpha/background key), centred on the
    figure's topmost columns."""
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)
    bg = np.concatenate([a[:8, :8].reshape(-1, 3), a[:8, -8:].reshape(-1, 3)]).mean(0)
    fg = np.abs(a - bg).sum(-1) > 40
    ys, xs = np.nonzero(fg)
    if len(ys) < 100:
        return None
    y0, y1 = ys.min(), ys.max()
    hh = 0.15 * (y1 - y0)
    top = fg[y0:int(y0 + hh)]
    cols = np.nonzero(top.any(0))[0]
    cx = float(np.median(cols)) if len(cols) else float(xs.mean())
    side = int(hh * 1.25)
    x0 = int(max(0, cx - side / 2))
    yb = int(max(0, y0 - 0.1 * side))
    return x0, yb, min(a.shape[1], x0 + side), min(a.shape[0], yb + side)


def step_head_detail(entries, card, style):
    """Item 2 (10-21): the head of every A-pose view repainted at 1024 px FROM its own crop (img2img 0.4, the LoRA,
    the face words) and set into a 2x copy of the view -> apose_<v>_hd.png, the texture source: the outline stays
    where it was (the projection still fits) and the face/hair get ~2x real painted detail (a 150 px head was all
    the close-ups had)."""
    from PIL import Image, ImageFilter
    jobs, plan = [], []
    for name, o in entries:
        d = Path(o["dir"])
        vd = d.parents[2]
        lora = lora_of(vd, o["slug"]) if o.get("slug") else None
        for v in VIEWS:
            src, hd = d / f"apose_{v}.png", d / f"apose_{v}_hd.png"
            if not src.exists() or hd.exists() or (v in ("left", "right") and not (d / f"apose_{v}.ok").exists()):
                continue
            if v != "back" and helm_on(o["text"]) and o.get("id"):
                continue                                 # a closed helm: no face to detail
            b = head_box(src)
            if not b:
                continue
            crop = d / f"head_{v}_init.png"
            Image.open(src).convert("RGB").crop(b).resize((1024, 1024), Image.LANCZOS).save(crop)
            vtxt = {"front": "seen from the front", "back": "seen from behind (the back of the head, no face)",
                    "left": "in exact profile facing left", "right": "in exact profile facing right"}[v]
            j = {"id": f"{o['key']}_head_{v}", "seed": 5, "size": [1024, 1024], "path": str(d / f"head_{v}.png"),
                 "prompt": (f"{'chr_' + o['slug'] + ', ' if lora else ''}Close-up of the head of {name}, {vtxt}: "
                            f"{strip_held(o['text'])}. The same head, hair and pose as the picture, finely detailed "
                            f"face, skin and hair, even soft light, plain light grey background. {style}No text."),
                 "init": {"image": str(crop), "strength": 0.4}}
            if lora:
                j["loras"] = [[lora, 0.8]]
            jobs.append(j)
            plan.append((d, v, b))
    if not jobs:
        return
    wq = Path(entries[0][1]["dir"]).parent / "_qwen"
    wq.mkdir(parents=True, exist_ok=True)
    run_qwen(jobs, wq, card)
    for d, v, b in plan:
        hp = d / f"head_{v}.png"
        if not hp.exists():
            continue
        im = Image.open(d / f"apose_{v}.png").convert("RGB")
        big = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
        bw, bh = (b[2] - b[0]) * 2, (b[3] - b[1]) * 2
        head = Image.open(hp).convert("RGB").resize((bw, bh), Image.LANCZOS)
        m = Image.new("L", (bw, bh), 0)
        m.paste(255, (bw // 10, bh // 10, bw - bw // 10, bh - bh // 10))
        big.paste(head, (b[0] * 2, b[1] * 2), m.filter(ImageFilter.GaussianBlur(bw // 20)))
        big.save(d / f"apose_{v}_hd.png")
        log(f"{d.name}: {v} view head detailed ({b[2] - b[0]} px -> 1024)")


def step_cut(o, force=False):
    d = Path(o["dir"])
    for v in VIEWS:
        src, dst = d / f"apose_{v}.png", d / f"apose_{v}_rgba.png"
        if HEAD_HD and (d / f"apose_{v}_hd.png").exists():
            src = d / f"apose_{v}_hd.png"                # the head-detailed 2x copy is the texture source
        if src.exists() and (force or not dst.exists()):
            rembg_rgba(src, dst)


def step_shape(o, card, force=False):
    d = Path(o["dir"])
    src, glb = d / "apose_front_rgba.png", d / "shape.glb"
    if not src.exists() or (glb.exists() and not force):
        return
    fd = gpu_turn(card)
    card = card_of(fd) or card
    try:
        # MULTI-VIEW shape (10-21: Hunyuan3D-2mv with the 2.1 code, front + back + verified sides: rounder heads, far
        # fewer hair spikes, 5.4 GB vs 7.7 GB) when its weights are here; else the single front view
        views = [f"front={src}"] + [f"{v}={d / f'apose_{v}_rgba.png'}" for v in VIEWS[1:]
                                    if (d / f"apose_{v}_rgba.png").exists() and (v == "back" or (d / f"apose_{v}.ok").exists())]
        mv = MV_WEIGHTS.exists() and len(views) >= 2
        cmd = [HY3D_PY, "-I", str(HERE / "blockout_shape.py")] + (["--views", *views, str(glb)] if mv else [str(src), str(glb)])
        log(f"{o['key']}: Hunyuan3D shape on card {card}" + (f" from {len(views)} views" if mv else ""))
        r = subprocess.run(cmd, capture_output=True, text=True, env=hip_env(card), timeout=840)
        (d / "shape.log").write_text(r.stdout[-4000:] + r.stderr[-4000:])
        if r.returncode:
            log(f"{o['key']}: shape FAILED (see {d / 'shape.log'})")
    finally:
        fd.close()


PAINT_VERSION = "visible-only-1022"   # bump when the texture projection changes: older models are re-painted + re-rigged


def step_rig(o, height, force=False):
    d = Path(o["dir"])
    glb, rb = d / "shape.glb", d / "rigged.blend"
    ps = d / "paint.sig"
    stale = rb.exists() and (not ps.exists() or ps.read_text().strip() != PAINT_VERSION)
    if not glb.exists() or (rb.exists() and not force and not stale):
        return
    if stale:                                        # (10-22: the head registered on its own - the Knight's double
        log(f"{o['key']}: painted by an older projection: re-painting + re-rigging (shape kept)")   # face)
        for f in ("qa.json", "qa.jpg"):
            (d / f).unlink(missing_ok=True)
    cmd = [POSE_PY, "-I", str(HERE / "blockout_model.py"), str(glb), str(d), "--height", str(height),
           "--texture", str(d / "apose_front_rgba.png")]
    for v in VIEWS[1:]:                              # back + the VERIFIED side views, blended by surface direction
        if (d / f"apose_{v}_rgba.png").exists() and (v == "back" or (d / f"apose_{v}.ok").exists()):
            cmd += [f"--texture-{v}", str(d / f"apose_{v}_rgba.png")]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    (d / "rig.log").write_text(r.stdout[-4000:] + r.stderr[-4000:])
    if r.returncode or not rb.exists():
        log(f"{o['key']}: rig FAILED (see {d / 'rig.log'})")
    else:
        o["model"] = str(rb)
        ps.write_text(PAINT_VERSION)
        log(f"{o['key']}: rigged {rb}")


VL_PY = os.path.expanduser("~/tts-ab/.venv-charlora-igpu/bin/python")


_VL = {"p": None}
_VL_LOCK = __import__("threading").Lock()


def vl_close() -> None:
    """Stop the loaded vision judge (its ~17 GB on the iGPU): before the language model takes the iGPU."""
    p = _VL["p"]
    _VL["p"] = None
    if p is not None and p.poll() is None:
        p.stdin.close()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()


def vl_ask(items):
    """tools/vl_judge.py on the iGPU (Qwen3-VL-8B): items [{"id", "image", "ask": {k: question}}] -> {id: answers}.
    One judge stays loaded for the whole job (10-24: a new process per call reloaded the model for every clip the
    take judge looked at and every batch of the seed pick); vl_close() frees it."""
    import select
    import time
    out = {}
    if not items:
        return out
    with _VL_LOCK:
        for attempt in range(2):
            p = _VL["p"]
            if p is None or p.poll() is not None:
                ig = subprocess.run([os.path.expanduser("~/bin/hip-index"), "igpu"], capture_output=True,
                                    text=True).stdout.strip()
                env = dict(os.environ, HIP_VISIBLE_DEVICES=ig, HSA_CU_MASK=f"{ig}:0-38")
                p = _VL["p"] = subprocess.Popen([VL_PY, "-I", str(HERE / "vl_judge.py")], stdin=subprocess.PIPE,
                                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                                env=env, bufsize=1)
            try:
                for x in items:
                    p.stdin.write(json.dumps(x) + "\n")
                p.stdin.flush()
                end = time.time() + 300 + 30 * len(items)
                while len(out) < len(items) and time.time() < end:
                    if not select.select([p.stdout], [], [], 5)[0]:
                        if p.poll() is not None:
                            break
                        continue
                    ln = p.stdout.readline()
                    if not ln:
                        break
                    try:
                        d = json.loads(ln)
                        out[d["id"]] = d.get("answers", {})
                    except Exception:                                # noqa: BLE001
                        pass
            except (BrokenPipeError, OSError):
                pass
            if len(out) == len(items):
                break
            vl_close()                      # died or stuck: a fresh judge for what is left
            items = [x for x in items if x.get("id") not in out]
    return out


def _yes(a):
    return str(a or "").strip().lower().startswith("yes")


HEAD_OK_Q = ("Look at this picture. Answer yes or no: is it a close-up of ONLY a creature's head with at most a short "
             "piece of neck, on a plain light background, with NO wings, NO body, NO sky or smoke clouds behind it?")


def creature_sheet(name, ent, vd, card, style, force=False):
    """A creature's reference sheet with a HEAD-ONLY headshot panel (the sheet generator draws a human headshot
    beside every figure, and a head+wing crop is not a headshot either): the head is located in the figure with
    the vision model, cut out onto a plain background, drawn as two head views (3/4 front + side profile; seeds
    retried until the vision check says: only the head and a little neck, no wings or body, plain background),
    and the two views are stacked in the left panel. Saved as the NEWEST refs/<slug>-<hash>.png."""
    import hashlib
    from PIL import Image
    o = ent["outfits"][""]
    if not o.get("sheet"):
        return None
    d = Path(o["dir"])
    d.mkdir(parents=True, exist_ok=True)
    marker = d / "creature_sheet.json"
    if marker.exists() and not force:
        return json.loads(marker.read_text())["sheet"]
    base_sheet = json.loads(marker.read_text()).get("from") if marker.exists() else o["sheet"]
    base_sheet = base_sheet if base_sheet and Path(base_sheet).exists() else o["sheet"]
    half, cut = step_refs(o)
    look = json.loads((vd / "look.json").read_text())["characters"][name]
    # 1. where is the head in the figure?
    ans = vl_ask([{"id": "box", "image": str(half), "max_new_tokens": 40, "ask": {
        "box": "Give the bounding box of the creature's HEAD only (from the horn tips to the jaw, no neck) as four "
               "integers x0 y0 x1 y1 on a 0-1000 scale of this image. Reply with the four numbers only."}}])
    nums = [int(x) for x in re.findall(r"\d+", str(ans.get("box", {}).get("box", "")))][:4]
    fh = Image.open(half).convert("RGB")
    W_, H_ = fh.size
    if len(nums) == 4 and nums[2] > nums[0] and nums[3] > nums[1]:
        x0, y0, x1, y1 = (nums[0] * W_ // 1000, nums[1] * H_ // 1000, nums[2] * W_ // 1000, nums[3] * H_ // 1000)
    else:                                               # fallback: the top of an upright figure
        x0, y0, x1, y1 = W_ // 3, 0, 2 * W_ // 3, H_ // 4
    # ONLY the head box survives (everything else -> plain grey: a wing beside the head must not leak in)
    from PIL import ImageDraw, ImageFilter
    pad = int(0.08 * max(x1 - x0, y1 - y0))
    mask = Image.new("L", (W_, H_), 0)
    ImageDraw.Draw(mask).rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + int(pad * 2.5)), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(pad // 2 + 1))
    grey = Image.new("RGB", (W_, H_), (128, 128, 128))
    only = Image.composite(fh, grey, mask)
    cx, cy, side = (x0 + x1) // 2, (y0 + y1) // 2 + pad, int(max(x1 - x0, y1 - y0) * 1.35)
    hc = only.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)).resize((1024, 1024))
    hraw = d / "head_crop.png"
    hc.save(hraw)
    head_ref = d / "head_ref.png"
    rembg_rgba(hraw, head_ref, grey=True)                # the head alone on plain grey
    log(f"{name}: head box {nums} -> {head_ref}")
    views = {"q34": "three-quarter front view of the head", "side": "exact SIDE PROFILE of the head, facing left"}
    chosen = {}
    for rnd in range(2):
        jobs = []
        for v, vt in views.items():
            if v in chosen:
                continue
            for k in range(3):
                seed = 11 + 7 * k + 100 * rnd
                j = {"id": f"{o['key']}_head_{v}_s{seed}", "seed": seed, "path": str(d / f"head_{v}_s{seed}.png"),
                     "size": [1024, 1024], "refs": [str(head_ref)],
                     "prompt": (f"A close-up of ONLY the HEAD of {name} ({look}), {vt}: the head fills the picture "
                                f"from the horn tips to the jaw, with only a short stub of neck; the mouth closed, a "
                                f"thin wisp of smoke from the nostrils, the eyes, horns and scales exactly as in image "
                                f"1. NO wings, NO body, NO sky, NO clouds: a plain flat light grey background. "
                                f"{style}No text, no lettering, no signature.")}
                j["init"] = {"image": str(head_ref), "strength": 0.72 if v == "q34" else 0.9}
                jobs.append(j)
        run_qwen(jobs, vd / "stage" / "characters" / "_qwen", card)
        items = [{"id": j["id"], "image": j["path"], "ask": {"ok": HEAD_OK_Q,
                  "side": "Answer yes or no: is the head seen exactly from the side (a profile)?"}}
                 for j in jobs if Path(j["path"]).exists()]
        res = vl_ask(items)
        for j in jobs:
            v = "q34" if "_head_q34_" in j["id"] else "side"
            a = res.get(j["id"], {})
            good = _yes(a.get("ok")) and (v != "side" or _yes(a.get("side")))
            log(f"{name}: {Path(j['path']).name} head-only={a.get('ok')} profile={a.get('side')} -> {good}")
            if good and v not in chosen:
                chosen[v] = j["path"]
        if len(chosen) == len(views):
            break
    if not chosen:
        log(f"{name}: no head drawing passed the head-only check")
        return None
    pw, ph = W_, H_
    sh = Image.open(base_sheet).convert("RGB")
    SW, SH = sh.size
    figp = sh.crop((SW // 2, 0, SW, SH))
    ims = [Image.open(chosen[v]).convert("RGB") for v in ("q34", "side") if v in chosen]
    bgc = tuple(int(c) for c in np.mean([np.asarray(im)[:12, :12].reshape(-1, 3).mean(0) for im in ims], 0))
    panel = Image.new("RGB", (SW // 2, SH), bgc)       # the drawings' own plain background (no card edges)
    cell = SH // len(ims)
    for i, im in enumerate(ims):
        sc_ = min((SW // 2) / im.width, cell / im.height)
        im = im.resize((int(im.width * sc_), int(im.height * sc_)))
        panel.paste(im, ((SW // 2 - im.width) // 2, i * cell + (cell - im.height) // 2))
    canvas = Image.new("RGB", (SW, SH), (232, 231, 228))
    canvas.paste(panel, (0, 0))
    canvas.paste(figp, (SW // 2, 0))
    hsh = hashlib.sha1(canvas.tobytes()).hexdigest()[:8]
    dst = vd / "refs" / f"{ent['slug']}-{hsh}.png"
    canvas.save(dst)
    marker.write_text(json.dumps({"sheet": str(dst), "heads": chosen, "from": base_sheet}))
    log(f"{name}: creature sheet {dst} (head views {list(chosen)})")
    return str(dst)


QA_POSES = ("sword_raised_overhead", "sword_thrust_lunge", "crouching", "kneel_with_sword", "pointing",
            "sitting_chair")      # extremes: shoulders overhead, deep knees, a lunge, kneeling, a reach, seated


def step_qa(name, o, ent, force=False):
    """Turnaround + 4 poses (tools/blockout.py renders of the model) + face likeness/age -> qa.jpg/qa.json."""
    d = Path(o["dir"])
    rb = d / "rigged.blend"
    # a rebuilt model gets a new QA (10-21: flow3d ran a forced `qa` over every character after each build for this)
    if not rb.exists() or ((d / "qa.jpg").exists() and (d / "qa.jpg").stat().st_mtime >= rb.stat().st_mtime and not force):
        return
    os.environ.setdefault("BLOCKOUT_ALLOW_VIOLATIONS", "1")
    import blockout as B
    h = ent.get("height") or 1.8
    sp = 1.1
    base = {"set": {"kind": "exterior", "ground": True}}

    def line(figs, out):
        n_ = len(figs)
        shot = dict(base, figures=figs,
                    camera={"pos": [sp * (n_ - 1) / 2, -4.6 - 0.35 * n_, h * 0.55], "look": [sp * (n_ - 1) / 2, 0, h * 0.48],
                            "lens": 35})
        B.run(json.loads(json.dumps(shot)), str(out), size=(1536, 864), samples=16)
        return out / "start_clay.png"
    # facing 90 = toward the camera (-y): front, 3/4, side, back
    turn = [{"who": f"{name}#{i}", "pose": "standing", "at": [i * sp, 0], "facing": f, "model": str(rb), "height": h}
            for i, f in enumerate((-90, -45, 0, 90))]
    poses = [{"who": f"{name}#{i}", "pose": p, "at": [i * sp, 0], "facing": -70, "model": str(rb), "height": h,
              **({"holds": {"right": "sword"}} if "sword" in p else {})} for i, p in enumerate(QA_POSES)]
    res = {"name": name, "outfit": o["id"], "model": str(rb)}
    try:
        t_png = line(turn, d / "qa_turn")
        p_png = line(poses, d / "qa_poses")
    except Exception as e:                               # noqa: BLE001
        log(f"{o['key']}: QA render failed: {e}")
        res["error"] = str(e)[:300]
        t_png = p_png = None
    # likeness + age: the model's own front render (prep's ortho front, painted) vs the sheet
    front = d / "front.png"
    if helm_on(o["text"]) and o.get("id"):
        res["face"] = {"skipped": "the helm covers the face"}
    elif front.exists() and o.get("sheet"):
        lj = d / "qa_face.json"
        gender = "woman" if re.search(r"\b(woman|girl|princess|queen|lady)\b", o["text"].lower()) else "man"
        cmd = [os.path.expanduser("~/tts-ab/.venv-hy3d/bin/python"), "-I", str(HERE / "face_likeness.py"), str(front),
               "--ref", o["sheet"], "--gender", gender, "--json", str(lj)]
        if ent.get("age"):
            cmd += ["--age", str(ent["age"])]
        subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=""))
        if lj.exists():
            res["face"] = json.loads(lj.read_text())
    # the head from the 3/4 and the side, close up, judged for smear (10-21: front + back paint left every head's
    # sides stretched -> melted faces in turned close-ups; a front-only QA never saw it). BLOCKING: the build
    # redraws the side views once (main), then flags the model
    if not (helm_on(o["text"]) and o.get("id")):
        res["side"] = side_qa(d, name, rb, h)
        if res["side"].get("pass") is False:
            res.setdefault("blocking", []).append(f"head smeared from the side: {res['side'].get('bad')}")
            log(f"!!! {o['key']}: the model's head reads smeared/stretched from {res['side'].get('bad')}")
    (d / "qa.json").write_text(json.dumps(res, indent=1, default=str))
    sheet_qa(d, name, o, t_png, p_png, res)


def side_qa(d, name, rb, h) -> dict:
    """Head close-ups of the painted model at a 3/4 and a side angle, asked of the vision judge. -> {"pass", "bad"}"""
    import blockout as B
    out = {}
    items = []
    for lab, fac in (("3q", -45), ("side", 0)):
        od = d / f"qa_head_{lab}"
        shot = {"set": {"kind": "exterior", "ground": True},
                "figures": [{"who": f"{name}#h", "pose": "standing", "at": [0, 0], "facing": fac, "model": str(rb), "height": h}],
                "camera": {"framing": "close-up", "target": f"{name}#h"}}
        try:
            B.run(json.loads(json.dumps(shot)), str(od), size=(768, 768), samples=16)
        except Exception as e:                          # noqa: BLE001
            return {"error": str(e)[:200]}
        items.append({"id": lab, "image": str(od / "start_clay.png"), "ask": {
            "smear": "Answer yes or no: are the face or the hair in this picture smeared, stretched, streaked or melted "
                     "(paint dragged across the surface, blotches, features in the wrong place)?",
            "head": "Answer yes or no: does this look like one coherent human head with a clean face and hair?"}})
    ans = vl_ask(items)
    bad = [i["id"] for i in items if _yes((ans.get(i["id"]) or {}).get("smear")) or
           not _yes((ans.get(i["id"]) or {}).get("head", "yes"))]
    out = {"pass": not bad, "bad": bad, "answers": ans}
    return out


def side_retry(entries, idx, vd, cards, dstyle, a) -> list:
    """Models whose head fails side_qa: the side views (and the shape/model built from them) set aside, the sides
    redrawn with another seed, the model rebuilt and re-checked ONCE. -> the outfits that still fail (flagged)."""
    redo = []
    for n, o in entries:
        d = Path(o["dir"])
        qj = d / "qa.json"
        q = json.loads(qj.read_text()) if qj.exists() else {}
        if (q.get("side") or {}).get("pass") is False and not (d / "side_retry").exists():
            (d / "side_retry").write_text(json.dumps(q["side"].get("bad")))
            arc = d / f"side_fail_{time.strftime('%H%M%S')}"
            arc.mkdir()
            for f in [*(d.glob("apose_left*")), *(d.glob("apose_right*")), *(d.glob("head_left*")), *(d.glob("head_right*")),
                      d / "shape.glb", d / "rigged.blend",
                      d / "model.blend", d / "rig.json", d / "qa.jpg", d / "qa.json"]:
                if f.exists():
                    shutil.move(str(f), arc / f.name)
            o["side_seed"] = 29
            redo.append((n, o))
    if not redo:
        return []
    log(f"side views redrawn for {[o['key'] for _, o in redo]} (their heads read smeared from the side)")
    for n, o in redo:
        step_draw([(n, o)], vd, cards[0], a.seeds, dstyle)
        if HEAD_HD:
            step_head_detail([(n, o)], cards[0], dstyle)
        step_cut(o)
        step_shape(o, cards[0])
        step_rig(o, idx["characters"][n]["height"])
        step_qa(n, o, idx["characters"][n], force=True)
    return [o["key"] for n, o in redo
            if (json.loads((Path(o["dir"]) / "qa.json").read_text()).get("side") or {}).get("pass") is False]


def sheet_qa(d, name, o, t_png, p_png, res):
    from PIL import Image, ImageDraw
    W = 1800
    tiles = []
    for p in (o.get("sheet"), d / "apose_front.png", d / "apose_back.png", d / "front.png"):
        if p and Path(p).exists():
            im = Image.open(p).convert("RGB")
            im.thumbnail((W // 4 - 8, 420))
            tiles.append(im)
    rows = []
    r1 = Image.new("RGB", (W, 430), (30, 30, 30))
    x = 4
    for im in tiles:
        r1.paste(im, (x, 5))
        x += im.width + 8
    rows.append(r1)
    for p in (t_png, p_png):
        if p and Path(p).exists():
            im = Image.open(p).convert("RGB")
            im = im.resize((W, int(im.height * W / im.width)))
            rows.append(im)
    H = sum(r.height for r in rows) + 40
    out = Image.new("RGB", (W, H), (20, 20, 20))
    dr = ImageDraw.Draw(out)
    f = res.get("face") or {}
    if f.get("skipped"):
        txt = f"{name} / {o['id'] or 'usual look'}   helm: face check skipped ({f['skipped']})"
    else:
        txt = (f"{name} / {o['id'] or 'usual look'}   face_sim {f.get('face_sim', '?')}  age-band "
               f"{f.get('age_band', f.get('age', '?'))}  wrinkled {f.get('wrinkled', '?')}  pass {f.get('pass', '?')}")
    dr.text((10, 10), txt, fill=(255, 255, 255))
    y = 40
    for r in rows:
        out.paste(r, (0, y))
        y += r.height
    out.save(d / "qa.jpg", quality=85)


# ------------------------------------------------------------------------------------------------ main
# ------------------------------------------------------------------------------------------------ creature build
def creature_build(idx, vd, only=None, kind="composite", model=None, dry_run=False, force=False):
    """A creature's 3D model (any book): kind "composite" = the procedural whole-body creature
    (creature_spec.json in the creature's dir -> tools/creature_sdf.py -> creature_sdf_bl.py rig -> creature_paint_bl.py
    from its sheet's body + side head -> creature_qa_bl.py clay QA sheet); kind "import" = a user-supplied rigged
    model mapped by --model MAP.json (tools/creature_import_bl.py). Output: <dir>/<kind>/{rigged.blend, rig.json,
    qa/sheet_clay.jpg, inputs.sha}. Resumable: nothing is redone while the inputs' hash is unchanged. The index entry
    gets the candidate under "candidates"; the RENDER model changes only by approval (bookcaster approve)."""
    import hashlib
    done = []
    blender = os.path.expanduser("~/opt/blender/current/blender")
    for name, e in idx["characters"].items():
        if e.get("kind") != "creature" or (only and name not in only):
            continue
        o = next(iter(e["outfits"].values()))
        d = Path(o.get("dir") or vd / "stage" / "characters" / e["slug"])
        rj_cur = Path(o["model"]).parent / "rig.json" if o.get("model") else None
        if not only and rj_cur and rj_cur.exists() and json.loads(rj_cur.read_text()).get("approved"):
            continue                                   # an approved rig: nothing to build
        out = d / kind
        if kind == "composite":
            sp = d / "creature_spec.json"
            if not sp.exists():
                spec = {"template": "winged_biped", "height_m": float(e.get("height") or 6.0),
                        "_doc": "default composite spec (tools/characters.py creature-build); edit to tune"}
                if not dry_run:
                    sp.write_text(json.dumps(spec, indent=1))
            inputs = [sp, TPL / "composite_creatures.json", HERE / "creature_sdf.py", HERE / "creature_sdf_bl.py",
                      HERE / "creature_paint_bl.py", HERE / "creature_rigid_part_bl.py"]
        else:
            if not model:
                log(f"{name}: --kind import needs --model MAP.json")
                continue
            inputs = [Path(model), HERE / "creature_import_bl.py"]
        h = hashlib.sha256(b"".join(p.read_bytes() for p in inputs if p.exists())).hexdigest()
        if not force and (out / "inputs.sha").exists() and (out / "inputs.sha").read_text().strip() == h \
                and (out / "rigged.blend").exists():
            log(f"{name}: {kind} model up to date ({out})")
            continue
        if dry_run:
            log(f"{name}: WOULD build the {kind} creature model -> {out}")
            done.append(name)
            continue
        out.mkdir(parents=True, exist_ok=True)
        lg = open(out / "build.log", "w")
        if kind == "composite":
            subprocess.run([POSE_PY, "-I", str(HERE / "creature_sdf.py"), str(sp), str(out)], stdout=lg, stderr=lg)
            subprocess.run([blender, "-b", "--factory-startup", "-t", "16", "--python", str(HERE / "creature_sdf_bl.py"),
                            "--", str(out)], stdout=lg, stderr=lg)
            cs = json.loads((d / "creature_sheet.json").read_text()) if (d / "creature_sheet.json").exists() else {}
            sheet = cs.get("sheet") or o.get("sheet")
            head = (cs.get("heads") or {}).get("side")
            if sheet and Path(sheet).exists():
                from PIL import Image
                im = Image.open(sheet).convert("RGB")
                W_, H_ = im.size
                body = out / "tex_body.png"
                (im.crop((W_ // 2, 0, W_, H_)) if W_ > H_ else im).save(body)   # a head|body sheet: its body half
                rembg_rgba(str(body), str(out / "tex_body_rgba.png"))
                args = ["--body", str(out / "tex_body_rgba.png")]
                if head and Path(head).exists():
                    rembg_rgba(head, str(out / "tex_head_rgba.png"))
                    args += ["--head", str(out / "tex_head_rgba.png"), "--head-box", "0.1,0.07,0.72,0.6"]
                subprocess.run([blender, "-b", "--factory-startup", "-t", "16", "--python",
                                str(HERE / "creature_paint_bl.py"), "--", str(out / "rigged.blend"),
                                str(out / "rigged.blend"), *args], stdout=lg, stderr=lg)
            # HYBRID: detailed RIGID parts (spec "rigid_parts": [{"part": "head", "source": ref.png, "mesh": optional
            # .glb, "bone": "head", "jaw": "jaw", "neck": "neck5", "scale": 1.15}]): image-to-3D of the part's own
            # reference (one shape on a free card) unless a mesh is given, then seated rigidly on its bone
            spec_ = json.loads(sp.read_text())
            for rp in spec_.get("rigid_parts") or []:
                pmesh = Path(rp.get("mesh") or (out / f"{rp['part']}.glb"))
                prgba = out / f"{rp['part']}_ref_rgba.png"
                if rp.get("source") and not prgba.exists():
                    rembg_rgba(rp["source"], str(prgba))
                if not pmesh.exists() and prgba.exists():
                    fd = gpu_turn("any")
                    try:
                        subprocess.run([HY3D_PY, "-I", str(HERE / "blockout_shape.py"), str(prgba), str(pmesh)],
                                       stdout=lg, stderr=lg, env=hip_env(card_of(fd)), timeout=840)
                    finally:
                        fd.close()
                if pmesh.exists():
                    cmd = [blender, "-b", "--factory-startup", "-t", "16", "--python", str(HERE / "creature_rigid_part_bl.py"),
                           "--", str(out / "rigged.blend"), str(out / "rigged.blend"), "--part", str(pmesh),
                           "--bone", rp.get("bone", "head"), "--neck", rp.get("neck", "neck5"),
                           "--scale", str(rp.get("scale", 1.15))]
                    if prgba.exists():
                        cmd += ["--paint", str(prgba)]
                    if rp.get("jaw"):
                        cmd += ["--jaw", rp["jaw"] if isinstance(rp["jaw"], str) else "jaw"]
                    subprocess.run(cmd, stdout=lg, stderr=lg)
                else:
                    log(f"{name}: rigid part {rp['part']} has no mesh (image-to-3D failed?)")
        else:
            subprocess.run([blender, "-b", "--factory-startup", "--disable-autoexec", "-t", "16", "--python",
                            str(HERE / "creature_import_bl.py"), "--", str(model), str(out)], stdout=lg, stderr=lg)
        (out / "qa").mkdir(exist_ok=True)
        subprocess.run([blender, "-b", "--factory-startup", "-t", "16", "--python", str(HERE / "creature_qa_bl.py"),
                        "--", str(out / "rigged.blend"), str(out / "rig.json"), str(out / "qa" / "q"), "--clayqa"],
                       stdout=lg, stderr=lg, cwd=str(out))
        from PIL import Image, ImageDraw
        poses = ["rest", "jaw_open", "wings_spread", "wings_folded", "flap_up", "flap_down", "strike", "carry"]
        cols = ["front", "side", "3q", "head", "head3q"]
        T = 240
        sh_ = Image.new("RGB", (110 + T * 5, T * len(poses) + 40), (20, 20, 20))
        dr = ImageDraw.Draw(sh_)
        dr.text((8, 10), f"{name}: {kind} creature model (clay QA)", fill=(255, 255, 255))
        for r_, p_ in enumerate(poses):
            dr.text((6, 40 + r_ * T + T // 2), p_, fill=(255, 255, 0))
            for c_, v_ in enumerate(cols):
                f_ = out / "qa" / f"q_{p_}_{v_}.png"
                if f_.exists():
                    im_ = Image.open(f_).convert("RGB")
                    im_.thumbnail((T - 4, T - 4))
                    sh_.paste(im_, (110 + c_ * T + 2, 40 + r_ * T + 2))
        sh_.save(out / "qa" / "sheet_clay.jpg", quality=82)
        if (out / "rigged.blend").exists():
            (out / "inputs.sha").write_text(h)
            o.setdefault("candidates", {})[kind] = {"model": str(out / "rigged.blend"), "qa": str(out / "qa" / "sheet_clay.jpg"),
                                                    "built": time.strftime("%Y-%m-%d %H:%M")}
            done.append(name)
            log(f"{name}: {kind} creature model built -> {out}")
        else:
            log(f"{name}: {kind} build FAILED (see {out / 'build.log'})")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build", "qa", "sheet", "creature-sheet", "creature-build"])
    ap.add_argument("--kind", default="composite", choices=["composite", "import"])
    ap.add_argument("--map", default=None, help="creature-build --kind import: the model's mapping JSON")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("book")
    ap.add_argument("--only", nargs="*", default=None, help="character names (or name:outfit)")
    ap.add_argument("--card", default="any")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--force", nargs="*", default=[], help="steps to redo: draw cut shape rig qa")
    ap.add_argument("--model", action="append", default=[], help="NAME=/abs/model (an existing rig, e.g. a creature)")
    a = ap.parse_args(argv)
    models = dict(m.split("=", 1) for m in a.model)
    idx = plan(a.book, models)
    vd = Path(idx["book"])
    if a.cmd == "plan":
        for n, e in idx["characters"].items():
            print(f"{n}: {e['kind']} h={e['height']} outfits={[k or '(usual)' for k in e['outfits']]} "
                  f"models={[o.get('model') for o in e['outfits'].values()]}")
        return 0
    if a.cmd == "creature-build":
        done = creature_build(idx, vd, a.only, a.kind, a.map, a.dry_run, force=bool(a.force))
        if not a.dry_run:
            ip = vd / "stage" / "characters" / "index.json"
            cur = json.loads(ip.read_text())
            for n in done:                             # only the "candidates" field changes
                for oid, o in idx["characters"][n]["outfits"].items():
                    if o.get("candidates"):
                        cur["characters"][n]["outfits"][oid]["candidates"] = o["candidates"]
            tmp = ip.with_suffix(".tmp")
            tmp.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
            os.replace(tmp, ip)
        print(json.dumps({"built" if not a.dry_run else "would_build": done}))
        return 0
    sty = sorted(vd.glob("*/style.txt"))
    style = ""
    if sty:
        s = sty[0].read_text().strip()
        style = (s.split(":")[0].split(".")[0].strip() + ". ") if s else ""
    entries = []
    for n, e in idx["characters"].items():
        for oid, o in e["outfits"].items():
            sel = a.only is None or n in a.only or f"{n}:{oid}" in a.only
            if not sel or e["kind"] != "human" or not o.get("sheet") or o.get("alias_of") is not None:
                continue
            o["slug"] = e["slug"]
            entries.append((n, o))
    t0 = time.time()
    if a.cmd == "build":
        for n, o in entries:
            sheet_guard(o, n, style)
        from tts_reader import video as _V      # the book's period into the drawing, not into prompt.sig (10-21)
        dstyle = style + _V.period_line(str(vd)).strip() + " " if _V.period_of(str(vd)) else style
        # BOTH cards (user 10-21: the 8 bodies queued on card 05 while 65 idled): with --card any the drawings
        # split in two halves (one Qwen load per card, at once), then one Hunyuan worker per card pulls bodies
        # from a queue and rigs each one (CPU) as soon as its shape exists
        cards = ["05", "65"] if a.card == "any" else [a.card]
        halves = [entries[i::len(cards)] for i in range(len(cards))]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(len(cards)) as ex:
            list(ex.map(lambda hc: step_draw(hc[0], vd, hc[1], a.seeds, dstyle, force="draw" in a.force),
                        [(h, c) for h, c in zip(halves, cards) if h]))
        if HEAD_HD:
            for c_, h_ in zip(cards, halves):
                if h_:
                    step_head_detail(h_, c_, dstyle)
        for n, o in entries:
            step_cut(o, force="cut" in a.force)
        import queue
        q = queue.Queue()
        for n, o in entries:
            q.put((n, o))

        def body_worker(card):
            while True:
                try:
                    n, o = q.get_nowait()
                except queue.Empty:
                    return
                step_shape(o, card, force="shape" in a.force)
                step_rig(o, idx["characters"][n]["height"], force="rig" in a.force)
        with ThreadPoolExecutor(len(cards)) as ex:
            list(ex.map(body_worker, cards))
    if a.cmd == "build" or a.cmd == "creature-sheet":
        for n, e in idx["characters"].items():
            if e["kind"] == "creature" and (a.only is None or n in a.only):
                sp = creature_sheet(n, e, vd, a.card, style,
                                    force="redraw" if "head" in a.force else ("sheet" in a.force))
                if sp:
                    e["outfits"][""]["sheet"] = sp
    if a.cmd in ("build", "qa"):
        for n, o in entries:
            step_qa(n, o, idx["characters"][n], force="qa" in a.force or a.cmd == "qa")
        if a.cmd == "build":
            from tts_reader import video as _V2
            ds_ = style + _V2.period_line(str(vd)).strip() + " " if _V2.period_of(str(vd)) else style
            still = side_retry(entries, idx, vd, ["05", "65"] if a.card == "any" else [a.card], ds_, a)
            if still:
                log(f"!!! models still smeared from the side after a redraw (flagged for review): {still}")
    # write the models back into the index
    for n, o in entries:
        rb = Path(o["dir"]) / "rigged.blend"
        if rb.exists():
            idx["characters"][n]["outfits"][o["id"] or ""]["model"] = str(rb)
    for e in idx["characters"].values():           # aliases wear their target's model
        for o in e["outfits"].values():
            if o.get("alias_of") is not None and e["outfits"].get(o["alias_of"], {}).get("model"):
                o["model"] = e["outfits"][o["alias_of"]]["model"]
    (vd / "stage" / "characters" / "index.json").write_text(json.dumps(idx, indent=1, default=str))
    sheets = [str(Path(o["dir"]) / "qa.jpg") for _, o in entries if (Path(o["dir"]) / "qa.jpg").exists()]
    print(json.dumps({"index": str(vd / "stage" / "characters" / "index.json"), "qa": sheets,
                      "seconds": round(time.time() - t0)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
