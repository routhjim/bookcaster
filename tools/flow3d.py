"""bookcaster's full-3D video flow as resumable stages (any book; the book's files are the data).

    python3 tools/flow3d.py assets  <book> [--chapter N] [--plan-only]   characters, creatures, props, sets + approvals
    python3 tools/flow3d.py stage   <book> --chapter N [--plan-only]     scene layer -> motion -> render -> validation
    python3 tools/flow3d.py review  <book> [--chapter N]                 QA sheets index + the pending list
    python3 tools/flow3d.py approve <book> ITEM ... | --auto             approve items / let the automated checks gate
    python3 tools/flow3d.py preflight <book> --chapter N                 what blocks the final render, as status lines
    python3 tools/flow3d.py video   <book> --chapter N [--auto-approve] [--jellyfin]   keys -> LTX -> QA -> assemble
(bookcaster: `bookcaster assets|stage|review|approve <book>` and `bookcaster video <book> --flow 3d`.)

Every stage is idempotent and cached per book under <book>/work/video/ (the tools' own outputs):
  stage/characters/ (tools/characters.py), stage/props/ (tools/props_build.py), worlds/ + stage/assets.json
  (tools/world.py build-book), <chapter>/stage/{shots,scenes} (tools/scene_stage.py), <chapter>/flow3d/ (the
  driver's work: motion, renders, keys, LTX, QA - tools/remake_chapter.py as a library).
Gates: <book>/work/video/stage/approvals.json {"items": {"<kind>:<name>": {"status": pending|approved|rejected,
"qa": sheet, "auto": {...}}}}. A final render refuses unapproved assets (the preflight lists them); --auto-approve
lets the automated QA results gate instead (unattended runs). A missing asset blocks loudly (FALLBACKS.json /
the status line), never a silent fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
POSE_PY = Path.home() / "tts-ab" / ".venv-pose" / "bin" / "python"
LTX_PY = Path.home() / "tts-ab" / ".venv-ltx" / "bin" / "python"


MIN_FREE_GB = float(os.environ.get("BOOKCASTER_MIN_FREE_GB", "40"))


def disk_check(book, log=print) -> None:
    """Stop BEFORE a step on a nearly full disk (10-22 knight_forged_v3: the drive filled at 01:56 mid-assets, every
    LoRA died on 'No space left on device' and half-written files had to be hunted down)."""
    free = shutil.disk_usage(Path(book) if Path(book).exists() else Path.home()).free / 2**30
    if free < MIN_FREE_GB:
        raise SystemExit(f"!!! only {free:.0f} GB free on the book's drive (need {MIN_FREE_GB:.0f}): free space first")


def _log(msg, log=print):
    log(f"  [3d] {msg}")


JOB_LOG = [None]          # the running job's log (assets / stage / video set it): run() reports there AT ONCE
HEARTBEAT_S = 600


def _step_name(cmd) -> str:
    c = [str(x) for x in cmd]
    i = next((k for k, x in enumerate(c) if x.endswith(".py")), 0)
    return Path(c[i]).name + (f" {c[i + 1]}" if i + 1 < len(c) and not c[i + 1].startswith("/") else "")


def run(cmd: list, logf: Path | None = None, env: dict | None = None, timeout: float | None = None) -> int:
    """Every tool call goes through here (the tests replace it). A failed step is reported to the job log the moment
    it fails, with its last lines, and a long one says it is still running every HEARTBEAT_S (user 10-21: a failed
    LoRA ran unseen for 80 min, its failure only in the step summary at the end)."""
    logf = Path(logf) if logf else None
    if logf:
        logf.parent.mkdir(parents=True, exist_ok=True)
    jl = JOB_LOG[0]
    t0 = time.time()
    with open(logf, "a") if logf else open(os.devnull, "w") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}  [{time.strftime('%H:%M:%S')}]\n") if logf else None
        fh.flush()
        p = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, env=env)
        nxt = HEARTBEAT_S
        while True:
            try:
                p.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                el = time.time() - t0
                if timeout and el > timeout:
                    p.kill()
                    p.wait()
                    raise subprocess.TimeoutExpired(cmd, timeout)
                if jl and el >= nxt:
                    jl(f"  [3d] still running: {_step_name(cmd)} for {el / 60:.0f} min (log {logf})")
                    nxt += HEARTBEAT_S
    if p.returncode and jl:
        tail = ""
        if logf and logf.exists():
            with open(logf, "rb") as fh:
                fh.seek(max(0, logf.stat().st_size - 1500))
                tail = " | ".join(ln.strip() for ln in fh.read().decode("utf-8", "replace").splitlines()[-4:] if ln.strip())
        jl(f"  [3d] !!! step FAILED (rc {p.returncode}, {(time.time() - t0) / 60:.1f} min): {_step_name(cmd)}: {tail[-400:]}")
    return p.returncode


def vdir(book: Path) -> Path:
    return Path(book) / "work" / "video"


def chapter_stem(book: Path, chapter) -> str:
    """'1' / '01' / '01_chapter_1' -> the chapter file stem."""
    files = sorted((Path(book) / "chapters").glob("[0-9][0-9]_*.txt"))
    c = str(chapter)
    for f in files:
        if f.stem == c or f.stem.startswith(c.zfill(2) + "_"):
            return f.stem
    raise SystemExit(f"no chapter {chapter} in {book}/chapters")


# ------------------------------------------------------------------------------------------------- approvals

def approvals_path(book) -> Path:
    return vdir(book) / "stage" / "approvals.json"


def load_approvals(book) -> dict:
    p = approvals_path(book)
    return json.loads(p.read_text()) if p.exists() else {"items": {}}


def save_approvals(book, ap: dict) -> None:
    p = approvals_path(book)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(ap, indent=1, ensure_ascii=False))
    tmp.replace(p)


def inventory(book) -> dict:
    """Every asset the flow uses, with its QA sheet and automated verdict: {"<kind>:<name>": {...}}."""
    vd = vdir(book)
    out = {}
    idx = vd / "stage" / "characters" / "index.json"
    if idx.exists():
        for name, ent in json.loads(idx.read_text()).get("characters", {}).items():
            for oid, o in (ent.get("outfits") or {}).items():
                d = Path(o.get("dir", ""))
                key = f"{'creature' if ent.get('kind') == 'creature' else 'character'}:{o.get('key') or d.name}"
                qa = d / "qa.json"
                q = json.loads(qa.read_text()) if qa.exists() else {}
                model = o.get("model") or (str(d / "rigged.blend") if (d / "rigged.blend").exists() else None)
                out[key] = {"kind": key.split(":")[0], "name": name, "outfit": oid or None, "model": model,
                            "qa": str(d / "qa.jpg") if (d / "qa.jpg").exists() else None,
                            "auto": {"built": bool(model and Path(model).exists()),
                                     "pass": False if q.get("blocking") else q.get("pass"),
                                     **({"blocking": q["blocking"]} if q.get("blocking") else {})}}
    pj = vd / "stage" / "props" / "props.json"
    if pj.exists():
        reg = json.loads(pj.read_text())
        reg = reg.get("props", reg) if isinstance(reg, dict) else {}
        for name, e in reg.items():
            if not isinstance(e, dict):
                continue
            out[f"prop:{name}"] = {"kind": "prop", "name": name, "mesh": e.get("mesh"),
                                   "qa": str(vd / "stage" / "props" / "qa.jpg") if (vd / "stage" / "props" / "qa.jpg").exists() else None,
                                   "auto": {"built": bool(e.get("mesh")), "pass": e.get("qa_pass")},
                                   "flag": e.get("approved") is True}
    aj = vd / "stage" / "assets.json"
    if aj.exists():
        for s in json.loads(aj.read_text()).get("sets", []):
            wd = Path(s.get("world_dir", ""))
            sheets = sorted(wd.glob("*sheet*.jpg")) + sorted(wd.glob("qa*.jpg")) if wd.exists() else []
            out[f"set:{s['where']}"] = {"kind": "set", "name": s["where"], "world_dir": str(wd),
                                        "qa": str(sheets[0]) if sheets else None,
                                        "auto": {"built": bool(s.get("built")), "pass": s.get("usable", True)}}
    for n in cast_names(book):
        cd = vd / "charlora" / re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")
        lo = sorted((cd / "lora").glob("*.safetensors")) if (cd / "lora").exists() else []
        sheet = cd / "data" / "sheet.jpg"
        out[f"lora:{n}"] = {"kind": "lora", "name": n, "lora": str(lo[-1]) if lo else None,
                            "qa": str(sheet) if sheet.exists() else None, "auto": {"built": bool(lo), "pass": None}}
    return out


def cast_names(book) -> list:
    """look.json's characters, one per person ('Dr. Watson' = 'Doctor Watson': the first-stored key wins)."""
    lj = vdir(book) / "look.json"
    names = list((json.loads(lj.read_text()).get("characters") or {}).keys()) if lj.exists() else []
    seen, out = set(), []
    for n in names:
        k = re.sub(r"[^a-z]", "", re.sub(r"\bdr\b\.?", "doctor", n.lower()))
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


CHARLORA_PY = Path.home() / "tts-ab" / ".venv-charlora-igpu" / "bin" / "python"


CONDITION_WORDS = {"torn", "ripped", "ragged", "tattered", "burnt", "burned", "scorched", "singed", "ash", "ashen",
                   "stained", "bloody", "bloodied", "muddy", "dirty", "filthy", "wet", "soaked", "dented", "melted",
                   "broken", "cracked", "grey", "gray", "sooty", "charred", "frayed", "worn"}
AGE_TOL = (5, 10)        # |read - look| under 25 / older (10-21 calibration: the judge reads young adults as ~25;
                         # Edric "about 19" -> 25 fails, Maren "about 20" -> 25 passes)


def _sheet_facts(plan, style, refs):
    """-> [(job, facts)] for every character sheet: facts = {"name", "age", "gender", "dress" (outfit sheets),
    "base" (the base sheet path, outfit sheets)}. Creatures and places are not checked here."""
    from tts_reader import video
    import remake_chapter as R
    chars, _ = video.looks(plan)
    out = []
    by_id = {j["id"]: j for j in video.reference_jobs(plan, style, refs)}
    for n, desc in chars.items():
        j = by_id.get(f"ref_{video.slug(n)}")
        if j and not R.is_creature(desc):
            out.append((j, {"name": n, "age": R.age_of(desc), "gender": R.gender_of(desc)}))
    base = {f["name"]: j["path"] for j, f in out}
    oj = {j["id"]: j for j in video.outfit_jobs(plan, style, refs)}
    for n, rows in (plan.get("wardrobe") or {}).items():
        if n not in chars or R.is_creature(chars[n]):
            continue
        for row in rows:
            j = oj.get(f"ref_{video.slug(n)}_{video.slug(row['id'])}")
            if j:
                out.append((j, {"name": n, "age": R.age_of(chars[n]), "gender": R.gender_of(chars[n]),
                                "dress": row.get("dress") or row["id"], "outfit": row["id"], "base": base.get(n),
                                "look": chars[n]}))
    return out


def _sheet_verdicts(items) -> dict:
    """VL on the headshot half (age, gender) and the full-length half (outfit); numeric outfit-vs-base difference.
    -> {job id: [failing facts]}"""
    import remake_chapter as R
    q, fails = [], {}
    for j, f in items:
        ask_h = {"age": "How old does this person look, judging by the face and skin? Answer with one number.",
                 "gender": "Is this person a man or a woman? Answer man or woman."}
        q.append({"id": j["id"] + "|h", "image": j["path"], "crop": [0, 0, 0.5, 1], "ask": ask_h})
        if f.get("dress"):
            # the outfit's SHORT name (10-21: the planner's long "...worn over his original jerkin" text made the judge
            # answer no to a correct armour sheet; pose/background swamp a pixel diff vs the base sheet)
            # what DISTINGUISHES the outfit from the base look (10-21: "is she wearing a dress?" passed Maren's
            # untorn "torn dress"): its new garments in plain words (the judge doesn't know "star-iron"), else its
            # condition words (torn, ash, stained...)
            gw = [g for g in R.outfit_words(f["dress"])["garment"] if g not in R.outfit_words(f.get("look", ""))["garment"]]
            cw = sorted(set(re.findall(r"[a-z]+", f["dress"].lower())) & CONDITION_WORDS)
            ask_b = {"dress": f"Is this person wearing {f['outfit']}? Answer yes or no."}
            if gw:
                ask_b["dress2"] = f"Is this person wearing {', '.join(gw[:3])}? Answer yes or no."
            elif cw:
                ask_b["dress2"] = f"Are this person's clothes {' and '.join(cw[:3])}? Answer yes or no."
            q.append({"id": j["id"] + "|b", "image": j["path"], "crop": [0.5, 0, 1, 1], "ask": ask_b})
    ans = R.vl_batch(q)
    if q and not ans:                             # the judge did not run (memory, crash): never a silent pass
        raise SystemExit("sheet check: the vision judge gave no answers (see the vl_batch line above)")
    for j, f in items:
        fl, h, b = [], ans.get(j["id"] + "|h", {}), ans.get(j["id"] + "|b", {})
        if not h:
            fails[j["id"]] = ["not checked: no judge answer"]
            continue
        m = re.search(r"\d+", str(h.get("age", "")))
        if f.get("age") and m:
            tol = AGE_TOL[0] if f["age"] < 25 else AGE_TOL[1]
            if abs(int(m.group()) - f["age"]) > tol:
                fl.append(f"age: reads {m.group()}, look {f['age']:.0f}")
        if f.get("gender") in ("man", "woman") and h.get("gender") and f["gender"] not in str(h["gender"]).lower():
            fl.append(f"gender: reads {h['gender']}, look {f['gender']}")
        if f.get("dress"):
            key_ = "dress2" if b.get("dress2") else "dress"      # the distinguishing question decides when asked
            if b.get(key_) and not str(b[key_]).lower().startswith("yes"):
                fl.append(f"outfit: not {f['outfit']}")
        if fl:
            fails[j["id"]] = fl
    return fails


def check_sheets(plan, style, refs, log=print, tries: int = 2) -> dict:
    """The look contract at the source (user 10-21: "bookcaster should catch and fix these"): every character sheet
    reads the look's age (headshot) and gender, and an outfit sheet shows its outfit and differs from the base
    sheet. A failing sheet is redrawn IN PLACE (the same file: everything finds it by its prompt hash) with a new
    seed and the failing facts stated, up to `tries` times; the attempt with the fewest failures is kept.
    -> {job id: still failing facts} (also refs/sheet_checks.json)."""
    from tts_reader import video
    items = [(j, f) for j, f in _sheet_facts(plan, style, refs) if Path(j["path"]).exists()]
    fails = _sheet_verdicts(items)
    path_of = {j["id"]: j["path"] for j, _ in items}
    hist = {i: [(len(v), Path(path_of[i]).read_bytes())] for i, v in fails.items()}
    for k in range(1, tries + 1):
        if not fails:
            break
        _log(f"plan: sheet check: {len(fails)} sheet(s) redrawn (try {k}): {fails}", log)
        redo = []
        for j, f in items:
            if j["id"] in fails:
                fix = []
                for x in fails[j["id"]]:
                    if x.startswith("age"):        # the image model needs age WORDS (10-20: "a youth of 18, smooth
                        a_ = f["age"]               # beardless boyish face, barely grown" fixed Edric; a number did not)
                        fix.append(f"{f['name']} is {a_:.0f} years old: " + (
                            "a teenage youth, smooth beardless boyish face, soft young features, barely grown"
                            if a_ < 21 else "a young adult, smooth youthful skin, no lines" if a_ < 30 else
                            "an elderly person, deeply lined face, white or grey hair" if a_ >= 60 else
                            f"clearly {a_:.0f}, neither younger nor older"))
                    elif x.startswith("gender"):
                        fix.append(f"{f['name']} is clearly a {f['gender']}")
                    elif x.startswith("outfit"):
                        fix.append(f"clearly visible on the full-length view: {f['dress']}; the clothes look "
                                   "visibly different from image 1")
                redo.append({**j, "seed": j["seed"] + 101 * k, "prompt": j["prompt"] + " IMPORTANT: " + "; ".join(fix) + "."})
        for j in redo:                            # draw_jobs skips existing files: the attempt is kept in hist
            Path(j["path"]).unlink(missing_ok=True)
        # base sheets first, then outfit sheets (drawn from the base: 10-21 one batch = "prompt not encoded")
        is_outfit = {j["id"] for j, f in items if f.get("outfit")}
        for part in ([j for j in redo if j["id"] not in is_outfit], [j for j in redo if j["id"] in is_outfit]):
            if part:
                video.draw_jobs(part, refs, log)
        for j in redo:
            if not Path(j["path"]).exists():      # a failed redraw: put the previous attempt back
                Path(j["path"]).write_bytes(hist[j["id"]][-1][1])
        now = _sheet_verdicts([(j, f) for j, f in items if j["id"] in fails])
        for j, f in items:
            if j["id"] in fails:
                hist[j["id"]].append((len(now.get(j["id"], [])), Path(j["path"]).read_bytes()))
        fails = now
    for i, h in hist.items():                     # keep the best attempt in place
        best = min(h, key=lambda t: t[0])
        Path(path_of[i]).write_bytes(best[1])
    (refs / "sheet_checks.json").write_text(json.dumps({"failing": fails, "checked": len(items),
                                                        "redrawn": sorted(hist)}, indent=1))
    return fails


def audit_looks(book, text, plan, log=print) -> dict:
    """Every planned look against what the text says about that character (user 10-21: the Knight's look was a
    linen tunic, no armour; the story: the king's knights "rode out in silver armor", one came back "his shield
    melted to a lump on his arm"). Per character the language model reads the passages naming them (their name or
    its last word: "knight(s)", "princess") and corrects a look that contradicts the text at their FIRST appearance
    (later changes belong to the wardrobe; held items stay props). Corrections go through bookcaster.set_look and
    into the chapter plan, so their sheets are redrawn. -> {name: {"was", "now", "why"}} (refs/look_audit.json)"""
    from tts_reader import bookcaster as BC, video
    from tts_reader.score import _chat
    chars, _ = video.looks(plan)
    paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    fixed = {}
    import hashlib
    lj = vdir(book) / "look.json"
    pinned = (json.loads(lj.read_text()).get("_pinned") or {}) if lj.exists() else {}
    ck = vdir(book) / "refs" / "look_audit_checked.json"
    checked = json.loads(ck.read_text()) if ck.exists() else {}

    def sig(d):
        return hashlib.sha1((d or "").encode()).hexdigest()[:12]
    for n, desc in chars.items():
        if n in pinned:                              # a look set by hand: kept, and the chapter plan follows it
            if desc != pinned[n]:
                plan["look"]["characters"][n] = pinned[n]
                fixed[n] = {"was": desc, "now": pinned[n], "why": "pinned (bookcaster look)"}
                _log(f"look audit: {n}: the pinned look (set by hand) restored", log)
            continue
        if checked.get(n) == sig(desc):              # audited before with this very look: not asked again (10-22: the
            continue                                 # same look audited each run flipped the Knight's armor off)
        words = {n.lower(), n.split()[-1].lower()}
        rx = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")s?\b", re.I)
        hits = [p.strip() for p in paras if rx.search(p)][:14]
        if not hits:
            continue
        later = [f"- {w['id']}: {w.get('dress', '')}" for w in (plan.get("wardrobe") or {}).get(n, [])]
        later_txt = ("\n\nLater outfit changes are ALREADY handled separately (do NOT put these into the look):\n"
                     + "\n".join(later)) if later else ""
        prompt = (f"A story character, {n}, will be drawn with this look:\n{desc}{later_txt}\n\nEvery passage of the story that "
                  f"mentions {n}:\n\n" + "\n\n".join(f"- {h}" for h in hits) +
                  f"\n\nDoes the look CONTRADICT what the passages say about {n} when they first appear (body, age, "
                  "hair, face, clothes, armour)? Ignore later changes of clothes and anything they hold or carry. "
                  'Answer only JSON: {"ok": true} or {"ok": false, "why": "<the contradiction>", "look": "<the '
                  'corrected full look, same style and length as the planned one, held items left out>"}. The '
                  "corrected look must STATE what the passages say instead (e.g. 'wearing dented silver armor'), "
                  "not just drop the contradicted words.")
        try:
            a = _chat(BC.LLM_URL, BC.LLM_MODEL, prompt, max_tokens=600)
            m = re.search(r"\{.*\}", a, re.S)
            v = json.loads(m.group()) if m else {}
        except (OSError, ValueError) as ex:
            raise SystemExit(f"look audit: the language model failed for {n}: {ex}")
        if v.get("ok") is False and v.get("look"):
            key = BC.set_look(Path(book), n, v["look"])
            plan["look"]["characters"][n] = video.literal_safe(v["look"])
            fixed[n] = {"was": desc, "now": v["look"], "why": v.get("why", "")}
            _log(f"!!! look audit: {key}: {v.get('why', '')} -> {v['look'][:160]}", log)
            checked[n] = sig(video.literal_safe(v["look"]))
        else:
            checked[n] = sig(desc)
        ck.parent.mkdir(parents=True, exist_ok=True)
        ck.write_text(json.dumps(checked, indent=1))
    (vdir(book) / "refs").mkdir(parents=True, exist_ok=True)
    (vdir(book) / "refs" / "look_audit.json").write_text(json.dumps(fixed, indent=1, ensure_ascii=False))
    return fixed


def infer_period(book, text: str, log=print) -> str:
    """<book>/period.txt from the text when not set (`bookcaster period` sets it by hand): every drawing prompt
    carries it and the vision judge checks it (user 10-21: "add a specific time period to all descriptions so we
    don't end up with anachronisms")."""
    from tts_reader import bookcaster as BC
    from tts_reader import video
    from tts_reader.score import _chat
    f = Path(book) / "period.txt"
    if f.exists() and f.read_text().strip():
        return f.read_text().strip()
    prompt = ("In what time period and region is this story set? Answer with ONE short phrase naming a real-world "
              "period and region with an approximate year, e.g. 'early medieval Northern Europe, around 1000 AD' or "
              "'Victorian London, around 1890'. For a fantasy or invented world, name the real period whose "
              "technology, clothing and buildings it matches. Only the phrase.\n\n" + text[:12000])
    a = (_chat(BC.LLM_URL, BC.LLM_MODEL, prompt, max_tokens=60) or "").strip().strip('"').splitlines()[0][:120]
    if a:
        f.write_text(a + "\n")
        _log(f"plan: time period = {a!r} (period.txt; `bookcaster period` changes it)", log)
    return video.period_of(book)


def _sets_parallel(book, lg, log=print, cards=("05", "65")) -> None:
    """The sets to build, split over both cards (one world.py build-book --only per card, at once); the places
    named after another place (drawn with its picture as reference) and the registry prune are left to the
    full pass that follows (everything else is reused by then)."""
    r = subprocess.run(["python3", str(HERE / "world.py"), "build-book", str(book), "--dry-run"],
                       capture_output=True, text=True)
    t = r.stdout
    try:
        rows = json.loads(t[t.index("\n[") + 1:] if not t.startswith("[") else t)
    except ValueError:
        return
    sys.path.insert(0, str(HERE))
    import world as Wd
    names = [x["where"] for x in rows]
    todo = [x["where"] for x in rows if x.get("status") == "would build"
            and not any(o != x["where"] and re.search(rf"\b{re.escape(re.sub(r'(?i)^(the|a|an)\s+', '', Wd.VARIANT_RX.sub('', o)).lower())}\b",
                                                      Wd.VARIANT_RX.sub("", x["where"]).lower())
                        and Wd.VARIANT_RX.sub("", o) != Wd.VARIANT_RX.sub("", x["where"]) for o in names)]
    if len(todo) < 2:
        return
    free = [c for c in cards if subprocess.run(["flock", "-n", f"/tmp/gpu-{c}.lock", "true"]).returncode == 0]
    cards = tuple(free) if free else cards        # a card training a LoRA would make its half wait
    if len(cards) < 2 and len(todo) < 3:
        return                                     # the full pass builds them on the free card anyway
    groups = [todo[i::len(cards)] for i in range(len(cards))]
    _log(f"sets: building {todo} on cards {list(cards)} at once", log)
    with ThreadPoolExecutor(len(cards)) as ex:
        list(ex.map(lambda gc: run(["python3", HERE / "world.py", "build-book", book, "--only", *gc[0], "--bus", gc[1]],
                                   Path(str(lg).replace(".log", f"_sets_{gc[1]}.log"))),
                    [(g, c) for g, c in zip(groups, cards) if g]))


def plan_lint(book, stems, log=print) -> list:
    """The plan's TEXT checked before any GPU time (user 10-21: the day's set/prop rebuilds all came from the plan's
    words): the period is set; every place is one side (its description not both inside and outside); no place
    description paints a staged prop; every moment's `where` is a place; no bare, period-less prop. Fixable ones are
    fixed in code downstream (world.clean_description, normalize_wheres, context prompts); this lists them early.
    -> [lines] (also refs/plan_lint.json)."""
    sys.path.insert(0, str(HERE))
    import world as Wd
    from tts_reader import video
    vd = vdir(book)
    look = json.loads((vd / "look.json").read_text())
    places = look.get("places") or {}
    props = Wd.prop_names(look, vd)
    out = []
    if not video.period_of(book):
        out.append("no time period (`bookcaster period`; the plan reads it from the text)")
    for n, d in places.items():
        cl = [c.strip() for c in re.split(r",|;|\band\b", d or "") if c.strip()]
        sc = [Wd._cue_score(c) for c in cl]
        if any(v > 0 for v in sc) and any(v < 0 for v in sc) and not Wd.VARIANT_RX.search(n):
            out.append(f"place {n!r} mixes inside and outside: the set is drawn as one side, the other clauses dropped")
        hit = [p_ for p_ in props if re.search(rf"\b{re.escape(p_.lower())}s?\b(?!\s+[a-z]{{3,}}\b(?<!\bin)(?<!\bon))", (d or "").lower())]
        if hit:
            out.append(f"place {n!r} names staged prop(s) {sorted(set(hit))[:4]}: left out of the set picture")
    ap = vd / "stage" / "assets.json"
    built = {x["where"] for x in (json.loads(ap.read_text()).get("sets") or [])} if ap.exists() else set()
    for st in stems:
        kp = vd / st / "anim" / "keyframes.json"
        if not kp.exists():
            continue
        bad = sorted({f.get("where") for f in json.loads(kp.read_text()).get("frames", [])
                      if f.get("where") and f["where"] not in set(places) | built
                      and Wd.VARIANT_RX.sub("", f["where"]) not in places})
        if bad:
            out.append(f"{st}: moments set in unknown places {[b[:40] for b in bad]}: re-homed by the stage step")
    for ln in out:
        _log("plan check: " + ln, log)
    (vd / "refs").mkdir(exist_ok=True)
    (vd / "refs" / "plan_lint.json").write_text(json.dumps(out, indent=1))
    return out


def normalize_wheres(book, work: Path, log=print, sets: set | None = None) -> list:
    """Every keyframe's `where` is a real place of the book, one side of it (10-21 knight ch1): the planner wrote
    "The Forge: a stone building at the edge of ..." (a place's DESCRIPTION as a name) for the river quench and
    the walk to the mountain, and the forge's inside work and its doorway/yard shared one `where` (one set
    drawn from outside). A name with a description is resolved to the place its own prompt names (name words
    x3, description words x1); a place whose moments are both inside and outside gets a "<place> (outside)" /
    "(inside)" variant for the minority side. -> [(index, old, new)], logged; anim/keyframes.json rewritten."""
    sys.path.insert(0, str(HERE))
    import world as Wd
    kp = work / "anim" / "keyframes.json"
    if not kp.exists():
        return []
    kf = json.loads(kp.read_text())
    frames = kf.get("frames") or []
    look = json.loads((vdir(book) / "look.json").read_text())
    places = dict(look.get("places") or {})
    clean = {f.get("where") for f in frames if f.get("where") and ":" not in f["where"]}
    known = {**{w: "" for w in clean}, **places}
    if sets is not None:
        # the stage step: ONLY the built sets (user 10-21: "only state places in keyframes that have been rendered")
        known = {w: places.get(w) or places.get(Wd.VARIANT_RX.sub("", w), "") for w in sets}
    stop = {"the", "a", "an", "of", "below", "above", "behind", "near", "edge", "with", "and", "in", "on", "at"}

    def words(t):
        return {w for w in re.findall(r"[a-z]{4,}", (t or "").lower()) if w not in stop}
    changed = []
    for i, f in enumerate(frames):
        w = f.get("where") or ""
        if not w or w in known or (sets is None and Wd.VARIANT_RX.sub("", w) in known):
            continue
        if sets is not None and (w in places or Wd.VARIANT_RX.sub("", w) in places):
            continue                                 # a real place whose set is missing: build it, never re-home
        txt = words(f.get("prompt", "") + " " + f.get("motion", ""))
        best = max(known, key=lambda k: 3 * len(words(k) & txt) + len(words(known[k]) & txt))
        changed.append((i, w, best))
        f["where_text"], f["where"] = w, best
    by = {}
    for i, f in enumerate(frames):
        if f.get("where"):
            by.setdefault(Wd.VARIANT_RX.sub("", f["where"]), []).append(i)
    for w, idx in (by.items() if sets is None else ()):
        sc = {i: Wd._cue_score(frames[i].get("prompt", "")) for i in idx}
        n_in, n_out = sum(v > 0 for v in sc.values()), sum(v < 0 for v in sc.values())
        if not (n_in and n_out):
            continue
        minority = "outside" if n_in >= n_out else "inside"
        for i, v in sc.items():
            if (v < 0 if minority == "outside" else v > 0) and frames[i]["where"] == w:
                changed.append((i, w, f"{w} ({minority})"))
                frames[i]["where"] = f"{w} ({minority})"
    if changed:
        kp.write_text(json.dumps(kf, indent=1, ensure_ascii=False))
        for i, a_, b_ in changed:
            _log(f"plan: moment {i} where {a_[:40]!r} -> {b_!r}", log)
    return changed


def plan_scenes(book, f, work: Path, log=print) -> list:
    """The chapter's scenes, once, from the text (user 10-22: "the script processor at the very beginning should
    tell you where each scene change is"): place + side + who is present + props per scene, in plan.json; every
    still picture takes its scene's place. Printed with the time and the first words of each change."""
    from tts_reader import video
    from tts_reader import bookcaster as BC
    pj = work / "plan.json"
    plan = json.loads(pj.read_text())
    text = BC._chapter_text(f)
    ap = vdir(book) / "stage" / "assets.json"          # the sets the book already built are places too
    known = {x["where"]: "" for x in (json.loads(ap.read_text()).get("sets") or []) if x.get("where")} if ap.exists() else {}
    scenes = video.scenes_for(plan, text, BC.LLM_URL, BC.LLM_MODEL, log, known=known)
    pj.write_text(json.dumps(plan, indent=1, ensure_ascii=False))
    lk = vdir(book) / "look.json"
    look = json.loads(lk.read_text()) if lk.exists() else {}
    new = {n: d for n, d in (plan.get("look") or {}).get("places", {}).items() if n not in (look.get("places") or {})}
    if new:
        look.setdefault("places", {}).update(new)
        lk.write_text(json.dumps(look, indent=1, ensure_ascii=False))
    tl = Path(book) / "work" / f"{f.stem}.timeline.json"
    starts = {id(s_): t for t, s_ in video.scene_starts(plan, text, json.loads(tl.read_text()))} if tl.exists() else {}
    _log(f"plan: {len(scenes)} scene(s)", log)
    for sc in scenes:
        t = starts.get(id(sc))
        sc["t0"] = round(t, 2) if t is not None else None
        ts = f"{int(t) // 60}:{int(t) % 60:02d}" if t is not None else "?"
        _log(f"  {sc['id']} {ts} P{sc['from']}-{sc['to']} {sc['where']} ({sc['side']}) present {sc['present']}"
             f" props {sc['props']} | \"{sc['first_words'][:60]}\" ({sc['why']})", log)
        if not sc["where"]:
            _log(f"  !!! {sc['id']}: no place", log)
    pj.write_text(json.dumps(plan, indent=1, ensure_ascii=False))      # with each scene's start time (t0)
    return scenes


def known_places(book) -> dict:
    """The places the book has built sets for ({name: "Inside:" / "Outside:" from the set's world.json kind, else ""}:
    the side is known for a set the looks never described)."""
    ap = vdir(book) / "stage" / "assets.json"
    out = {}
    for x in (json.loads(ap.read_text()).get("sets") or []) if ap.exists() else []:
        if not x.get("where"):
            continue
        wj = Path(x.get("world_dir") or "") / "world.json"
        k = json.loads(wj.read_text()).get("kind") if wj.exists() else None
        out[x["where"]] = {"interior": "Inside:", "exterior": "Outside:"}.get(k, "")
    return out


def scene_frames(book, f, work: Path, shots: list, log=print) -> int:
    """Every moment's keyframe facts agree with its scene (video.apply_scene: the scene's place, its id, who is
    present, a creature the text names as a figure); anim/keyframes.json rewritten. -> frames changed."""
    from tts_reader import video
    from tts_reader import bookcaster as BC
    plan = json.loads((work / "plan.json").read_text())
    tl = Path(book) / "work" / f"{f.stem}.timeline.json"
    kp = work / "anim" / "keyframes.json"
    if not plan.get("scenes") or not tl.exists() or not kp.exists():
        return 0
    starts = video.scene_starts(plan, BC._chapter_text(f), json.loads(tl.read_text()))
    tof = {s_["id"]: s_["t"] + 0.5 * float(s_.get("dur") or 0) for s_ in shots}     # the MIDDLE of the moment
    kf = json.loads(kp.read_text())
    chars = (plan.get("look") or {}).get("characters") or {}
    n = 0
    for fr in kf.get("frames") or []:
        t = tof.get(fr.get("shot"))
        sc = video.scene_at_t(starts, t) if t is not None else None
        if not sc:
            continue
        ch = video.apply_scene(fr, sc, chars, {**known_places(book), **((plan.get("look") or {}).get("places") or {})})
        if ch:
            n += 1
            _log(f"  scene {sc['id']}: {fr.get('shot')}: " + "; ".join(ch), log)
    kp.write_text(json.dumps(kf, indent=1, ensure_ascii=False))
    _log(f"plan: {n} moment(s) brought into their scene", log)
    return n


def plan(book, chapter, log=print) -> dict:
    """Stage 0 of the full-3D flow, from the text (10-21: the flow could only REMAKE a 2D-animated chapter - its
    moment list was the 2D clips that existed - so a fresh book had no moments and old runs leaked in).
    The language model plans the pictures + looks + wardrobe (bookcaster's picture planner) and the moments +
    their frame facts (who / where / motion: the animator's plan-only phase, anim/keyframes.json); every planned
    moment goes into anim/manifest.json (no clip yet); the reference sheets (portraits, outfits) and the chapter's
    still pictures (the gaps between moments) are drawn. Nothing is animated here."""
    JOB_LOG[0] = log
    disk_check(book, log)
    from tts_reader import animate, video
    from tts_reader import bookcaster as BC
    book = Path(book)
    stem = chapter_stem(book, chapter)
    f = next(x for x in BC.chapter_files(book) if x.stem == stem)
    work = BC._video_work(book, f)
    animate.igpu("strata", log)                       # the planner's language model (iGPU)
    _log("plan: pictures, looks, wardrobe", log)
    BC.plan_pictures_for(book, f, log)
    plan_scenes(book, f, work, log)
    _log("plan: moments + keyframe facts (no drawing)", log)
    r = BC.animate_for(book, f, "animated", log, phase="plan-only")
    normalize_wheres(book, work, log)
    scene_frames(book, f, work, r["shots"], log)
    import plan_review                           # the plan checked for agreement BEFORE any drawing (user 10-22)
    _log("plan: review (rules + the language model, scene by scene, two passes)", log)
    plan_review.review(book, f, work, r["shots"], log)
    plan_lint(book, [stem], log)
    animate.igpu("strata", log)                   # "active" is not "answering" (10-21: a starting unit refused)
    from urllib.request import urlopen
    for _ in range(60):
        try:
            urlopen(BC.LLM_URL.rsplit("/chat/", 1)[0] + "/models", timeout=5)
            break
        except OSError:
            time.sleep(5)
    infer_period(book, BC._chapter_text(f), log)
    fx = audit_looks(book, BC._chapter_text(f), r["plan"], log)
    # every cast member gets a sheet, not only those in a still picture: assets builds each one's model + LoRA
    # from it (10-22: the King, only in moments, had none and his LoRA failed after 50 min of drawing)
    chars = (r["plan"].get("look") or {}).get("characters") or {}
    r["plan"]["sheet_cast"] = [n for n in cast_names(book) if n in chars]
    pj = work / "plan.json"
    saved = json.loads(pj.read_text())
    saved["sheet_cast"] = r["plan"]["sheet_cast"]
    for n, v in (fx or {}).items():           # the corrected looks into the chapter plan (sheet prompts follow)
        saved.setdefault("look", {}).setdefault("characters", {})[n] = r["plan"]["look"]["characters"][n]
    pj.write_text(json.dumps(saved, indent=1, ensure_ascii=False))
    # the planning is done: hand the iGPU's ~77 GB back (10-21: the language model stayed up and the sheet judge
    # was skipped for memory); the 3D flow never starts the 2D VAE server, so stop it here
    if subprocess.run(["systemctl", "--user", "is-active", "-q", animate.STRATA_UNIT]).returncode == 0:
        _log("plan: stopping the language model (planning done)", log)
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
        time.sleep(5)
        animate._flush()
    man = [{**{k: s[k] for k in ("id", "kind", "t", "dur", "pic")},
            "out": s.get("out") or str(work / "anim" / f"{s['id']}.mp4")} for s in r["shots"]]   # (the animator's name)
    (work / "anim").mkdir(parents=True, exist_ok=True)
    (work / "anim" / "manifest.json").write_text(json.dumps(man, indent=1))
    _log(f"plan: {len(man)} moments ({sum(1 for m in man if m['kind'] == 'talk')} talk) -> anim/manifest.json", log)
    refs = vdir(book) / "refs"
    miss = 0
    # ONE drawing session for the sheets, their redraws and the stills: the model stays loaded on BOTH cards (10-21
    # plan items 3/4: 11 stills took ~11 min on one card after the sheets' own separate loads)
    with video.drawing_session(work / "anim" / "draw.log", log):
        return _plan_draw(book, f, r, refs, man, work, log)


def _plan_draw(book, f, r, refs, man, work, log) -> dict:
    from urllib.request import urlopen
    from tts_reader import animate, video
    from tts_reader import bookcaster as BC
    miss = 0
    # base sheets FIRST: an outfit sheet is drawn from its character's base sheet (10-21: one batch = 3 outfit
    # sheets failed, their reference not drawn yet)
    for kind, jobs in (("reference", video.reference_jobs), ("outfit", video.outfit_jobs)):
        todo = [j for j in jobs(r["plan"], r["style"], refs) if not Path(j["path"]).exists()]
        if todo:
            _log(f"plan: drawing {len(todo)} {kind} sheet(s)", log)
            video.draw_jobs(todo, refs, log)
            miss += len(todo)
        left = [j["id"] for j in jobs(r["plan"], r["style"], refs) if not Path(j["path"]).exists()]
        if left:
            raise SystemExit(f"plan: {kind} sheet(s) not drawn: {left}")
    bad = check_sheets(r["plan"], r["style"], refs, log)
    if bad:
        _log(f"!!! plan: {len(bad)} sheet(s) still fail after their redraws (refs/sheet_checks.json): {bad}", log)
    _log("plan: the chapter's still pictures (the stretches between moments)", log)
    # the stills' picture check asks the language model (10-22: it was stopped for the sheet judge's memory, so
    # every still went unchecked - "picture check skipped"): up again for the stills only, then down
    animate.igpu("strata", log)
    for _ in range(60):
        try:
            urlopen(BC.LLM_URL.rsplit("/chat/", 1)[0] + "/models", timeout=5)
            break
        except OSError:
            time.sleep(5)
    try:
        BC.draw_pictures_for(book, f, log)
    finally:
        subprocess.run(["systemctl", "--user", "stop", animate.STRATA_UNIT], capture_output=True)
    return {"moments": len(man), "refs_drawn": miss, "manifest": str(work / "anim" / "manifest.json")}


def lora_todo(book, only=None) -> dict:
    """The cast (look.json characters, creatures included) whose LoRA is missing -> {name: why}. charlora's own
    input hash decides a retrain when the look/refs change; this is the cheap plan-time listing."""
    vd = vdir(book)
    out = {}
    for n in cast_names(book):
        if only is not None and n not in only:   # one chapter's cast (10-25: Acts 2 would retrain Acts 1's Jesus)
            continue
        d = vd / "charlora" / re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_") / "lora"
        if not any(d.glob("*.safetensors")) if d.exists() else True:
            out[n] = "no LoRA yet"
            continue
        # a LoRA from an old sheet / model / look is stale too (10-21: the Knight's tunic LoRA after armour)
        bj = d.parent / "build.json"
        if not bj.exists():          # build.json = a finished build (10-22: Hulda's stopped at step 100 by a full
            out[n] = "unfinished (its build stopped part-way)"     # disk, and her step-100 file passed as done)
            continue
        try:
            sys.path.insert(0, str(HERE))
            import charlora as L
            bjd = json.loads(bj.read_text()) if bj.exists() else {}
            prev = bjd.get("hash")
            if prev and not L.hash_current(L.Char(str(book), n), prev, None, bjd.get("shape_hash")):
                out[n] = "stale (its look, sheet or model changed)"
        except Exception as exc:                          # noqa: BLE001 - the listing must not stop the job
            out[n] = f"unknown ({type(exc).__name__}: {exc})"
    return out


def register(book) -> dict:
    """Add every asset of the inventory to approvals.json (new ones pending; flags already set in the tools'
    files count as approved)."""
    ap = load_approvals(book)
    items = ap.setdefault("items", {})
    inv = inventory(book)
    for k, v in inv.items():
        e = items.setdefault(k, {"status": "pending", "since": time.strftime("%Y-%m-%d %H:%M")})
        e.update({x: v.get(x) for x in ("kind", "name", "qa", "auto")})
        if e["status"] == "dropped":                 # back in the plan: to be reviewed again
            e["status"] = "pending"
        if v.get("flag") and e["status"] == "pending":
            e["status"], e["by"] = "approved", "tool flag"
    # an item the plan no longer has is not waiting for anyone (10-21: rejected and dropped items kept the review
    # list at "38 pending"); decisions already made are kept as they are
    for k, e in items.items():
        if k not in inv and e["status"] == "pending":
            e.update(status="dropped", at=time.strftime("%Y-%m-%d %H:%M"))
    save_approvals(book, ap)
    return ap


def approve(book, keys: list, status: str = "approved", by: str = "user") -> list:
    """Approve (or reject) items; propagate to the tools' own flags (props.json, rig.json) the planner reads."""
    ap = register(book)
    items, done = ap["items"], []
    vd = vdir(book)
    for k in keys:
        hits = [x for x in items if x == k or x.split(":", 1)[-1].lower() == k.lower()]
        for x in hits:
            items[x].update(status=status, by=by, at=time.strftime("%Y-%m-%d %H:%M"))
            done.append(x)
            kind, name = x.split(":", 1)
            if kind == "prop":
                pj = vd / "stage" / "props" / "props.json"
                reg = json.loads(pj.read_text())
                tgt = reg.get("props", reg)
                if isinstance(tgt.get(name), dict):
                    tgt[name]["approved"] = status == "approved"
                    pj.write_text(json.dumps(reg, indent=1, ensure_ascii=False))
            if kind == "creature":
                inv = inventory(book).get(x, {})
                m = inv.get("model")
                if m and (Path(m).parent / "rig.json").exists():
                    rj = Path(m).parent / "rig.json"
                    d = json.loads(rj.read_text())
                    d["approved"] = status == "approved"
                    rj.write_text(json.dumps(d, indent=1))
    save_approvals(book, ap)
    return done


def auto_approve(book) -> list:
    """The automated checks gate: an item whose tool QA passed (built + pass true; sets: built + usable) is approved
    'auto'; an item without a verdict stays pending (listed)."""
    ap = register(book)
    keys = [k for k, e in ap["items"].items() if e["status"] == "pending" and (e.get("auto") or {}).get("built")
            and (e.get("auto") or {}).get("pass") is True]
    return approve(book, keys, by="auto")


def pending(book) -> list:
    return [k for k, e in register(book)["items"].items() if e["status"] == "pending"]


# ------------------------------------------------------------------------------------------------- stages

def assets(book, chapter=None, log=print, plan_only: bool = False, card: str = "any", only=None) -> dict:
    """Characters (+ creature sheets), props, sets: each tool's own resumable entry point; GPU steps take their card
    lock per call and yield to a waiting LTX gaprun (the tools' own rule)."""
    JOB_LOG[0] = log
    disk_check(book, log)
    book = Path(book)
    vd = vdir(book)
    lg = vd / "stage" / "flow3d_assets.log"
    stems = [chapter_stem(book, chapter)] if chapter else [f.stem for f in sorted((book / "chapters").glob("[0-9][0-9]_*.txt"))
                                                           if (vd / f.stem / "anim" / "keyframes.json").exists()]
    rc = {}
    plan_lint(book, stems, log)                  # the plan's words, before any GPU (10-21)
    rc["characters.plan"] = run([POSE_PY, "-I", HERE / "characters.py", "plan", book], lg)
    if not plan_only:
        rc["characters.build"] = run([POSE_PY, "-I", HERE / "characters.py", "build", book, "--card", card,
                                      *(["--only", *only] if only else [])], lg)
        rc["creatures.sheet"] = run([POSE_PY, "-I", HERE / "characters.py", "creature-sheet", book], lg)
    # 2c: creatures as a procedural composite (or a user-rigged import), once the blockout agent's entry exists
    if "creature-build" in (HERE / "characters.py").read_text():
        rc["creatures.build"] = run([POSE_PY, "-I", HERE / "characters.py", "creature-build", book, "--kind",
                                     os.environ.get("BOOKCASTER_CREATURE_KIND", "composite"),
                                     *(["--dry-run"] if plan_only else [])], lg)
    # 2f: a LoRA for EVERY character / creature / outfit (charlora, cached by an input hash), one character per XTX in
    # parallel (10-21: ~35 min each on an XTX vs ~2.5 h on the iGPU, which was serial here); the key + face passes use
    # each person's LoRA in their own region
    todo = lora_todo(book)
    if plan_only:
        for n, why in todo.items():
            _log(f"assets: LoRA {n}: would train ({why})", log)
    elif todo:
        # in the BACKGROUND (10-21: props + sets waited a whole LoRA with the other card idle): each training holds
        # its card's lock, so props / sets take whichever card is free and queue for the rest
        import threading
        lora_res = {}
        lth = threading.Thread(target=lambda: lora_res.update(left=ensure_loras(book, log)), daemon=True)
        lth.start()
    for st in stems:
        rc[f"props.plan:{st}"] = run([POSE_PY, "-I", HERE / "props_build.py", "plan", book, st], lg)
        if not plan_only:
            rc[f"props.build:{st}"] = run([POSE_PY, "-I", HERE / "props_build.py", "build", book, st, "--card", card], lg)
    if not plan_only:
        _sets_parallel(book, lg, log)            # both cards (10-21: six sets queued on one card while the other idled)
    rc["sets"] = run(["python3", HERE / "world.py", "build-book", book, *(["--dry-run"] if plan_only else [])], lg)
    if not plan_only:                                  # the sets' own fixed objects + door/window wall anchors
        rc["sets.locate"] = run([POSE_PY, "-I", HERE / "world_locate.py", "book", book], lg)
    if not plan_only and todo:
        lth.join()
        rc["loras"] = 1 if lora_res.get("left") else 0
    ap = register(book)
    bad = {k: v for k, v in rc.items() if v}
    _log(f"assets: {len(ap['items'])} item(s), {len(pending(book))} pending review" +
         (f"; FAILED steps {bad} (see {lg})" if bad else ""), log)
    # the step FAILS when a build step failed or an item is held by a blocking check (10-22: assets said "done"
    # with the King's LoRA failed and all 7 people's models held; the chain ran the video step, which then
    # waited on review for 86 of 91 moments)
    held = {k: (e.get("auto") or {}).get("blocking") for k, e in ap["items"].items()
            if e.get("status") == "pending" and (e.get("auto") or {}).get("pass") is False}
    if not plan_only and (bad or held):
        for k, why in held.items():
            _log(f"!!! {k} held: {why}", log)
        raise SystemExit(f"assets: {len(bad)} step(s) failed {sorted(bad)}, {len(held)} item(s) held by a blocking "
                         f"check {sorted(held)}: fix them (or `bookcaster approve` after looking), then re-run")
    return {"rc": rc, "pending": pending(book)}


def _ctx(book, stem, full3d=True):
    if full3d:
        os.environ["REMAKE_FULL3D"] = "1"
    import remake_chapter as R
    R.ALERT = JOB_LOG[0]
    out = vdir(book) / stem / "flow3d"
    out.mkdir(parents=True, exist_ok=True)
    return R, R.Ctx(Path(book), stem, out)


def stage(book, chapter, log=print, plan_only: bool = False, sheets: bool = True) -> dict:
    """The scene layer (tools/scene_stage.py: scenes, marks, axis, coverage, camera paths, extra moments) -> the
    choreography (motion_batch, via the shared driver) -> renders + the validation numbers per scene."""
    JOB_LOG[0] = log
    disk_check(book, log)
    book = Path(book)
    stem = chapter_stem(book, chapter)
    vd = vdir(book)
    lg = vd / stem / "stage" / "flow3d_stage.log"
    # every moment's place is a BUILT set, by its exact name (user 10-21): a stray `where` is re-homed or stops here
    built = {x["where"] for x in (json.loads((vd / "stage" / "assets.json").read_text()).get("sets") or [])
             if Path(x.get("world_dir", "")).joinpath("world.json").exists()} if (vd / "stage" / "assets.json").exists() else set()
    normalize_wheres(book, vd / stem, log, sets=built)
    stray = sorted({f.get("where") for f in json.loads((vd / stem / "anim" / "keyframes.json").read_text()).get("frames", [])
                    if f.get("where") and f["where"] not in built})
    if stray and not plan_only:
        raise SystemExit(f"stage: moment(s) set in places with no built set: {stray} (`bookcaster assets` builds them)")
    rc = run([POSE_PY, "-I", HERE / "scene_stage.py", book, stem, *(["--sheets", "--no-render"] if sheets else [])], lg)
    if rc:
        raise SystemExit(f"scene layer failed (rc {rc}): see {lg}")
    if not plan_only:    # the shots' set props exist only now (10-21 dry run: assets ran before the scene layer, so
        # workbench / oilcloth / ledge had no mesh and would render as plain near-white boxes)
        for sub in ("plan", "build"):
            rc = run([POSE_PY, "-I", HERE / "props_build.py", sub, book, stem, *(["--card", "any"] if sub == "build" else [])], lg)
            if rc:
                raise SystemExit(f"props {sub} after the scene layer failed (rc {rc}): see {lg}")
    R, cx = _ctx(book, stem)
    P = R.plan(cx)
    if plan_only:
        return {"plan": P, "blocked": blocked_lines(R, P)}
    retry_failed(cx, log)
    for cl in (cx.OUT / "stage").glob("*/.rendering") if (cx.OUT / "stage").exists() else []:
        cl.unlink(missing_ok=True)               # one job per book: a claim left at start belongs to a dead process
    # renders sized to the cores (no LTX runs in this step): ~5 Blender threads each; the clips only for the moments
    # whose motion is the point (needs_clip), the rest their stills (their clips render in the video step)
    R.stage(cx, P, steps="motion,render", workers=max(2, (os.cpu_count() or 16) // 5))
    val, missing = {}, []
    for e in P["entries"]:
        vj = cx.OUT / "stage" / e["id"] / "render" / "video.json"
        sc = (json.loads(Path(e["shot"]).read_text()).get("scene") or {}).get("id", "?") if e.get("shot") else "?"
        v = json.loads(vj.read_text()).get("violations") if vj.exists() else None
        if v is None and e.get("route") == "stage":
            missing.append(e["id"])                  # not rendered: a failure, never "violations 0" (10-21)
        val.setdefault(sc, {})[e["id"]] = v
    unseen = figures_unseen(cx, P)
    (vd / stem / "stage" / "validation.json").write_text(json.dumps({"scenes": val, "not_rendered": missing,
                                                                    "figures_unseen": unseen}, indent=1))
    for sc, d in sorted(val.items()):
        bad = {k: v for k, v in d.items() if v}
        nr = [k for k in d if k in missing]
        _log(f"scene {sc}: {len(d)} shot(s), violations {sum(v or 0 for v in d.values())}" +
             (f" in {sorted(bad)}" if bad else "") + (f"; NOT RENDERED {nr}" if nr else ""), log)
    for i, w in unseen.items():
        _log(f"!!! {i}: {w}", log)
        rd_ = cx.OUT / "stage" / i / "render"     # re-rendered on the next run (a cached render would fail again)
        if rd_.exists():
            rd_.rename(rd_.parent / f"render_unseen_{time.strftime('%H%M%S')}")
    if unseen:
        raise SystemExit(f"stage: {len(unseen)} moment(s) with a figure out of frame: {sorted(unseen)}")
    if missing:
        why = {i: (cx.OUT / "stage" / i / "FAILED").read_text()[:120] for i in missing if (cx.OUT / "stage" / i / "FAILED").exists()}
        raise SystemExit(f"stage: {len(missing)} moment(s) not rendered (re-run retries them once): {why or missing}")
    return {"plan": P, "validation": val}


def figures_unseen(cx, P) -> dict:
    """Moments whose staged start/end frame lost a figure of the shot (10-22: the dragon fight's camera framed Edric
    alone and staging said "violations 0"): a creature's mask under 0.2 % of the frame, or no person at all in the
    figure mask when the shot has people. -> {moment id: why}"""
    from PIL import Image
    import numpy as np
    out = {}
    for e in P["entries"]:
        if e.get("route") != "stage" or not e.get("shot"):
            continue
        rd = cx.OUT / "stage" / e["id"] / "render"
        sh_ = json.loads(Path(e["shot"]).read_text())
        figs = sh_.get("figures") or []
        close = (sh_.get("camera") or {}).get("framing") in ("close-up", "extreme close-up", "insert", "medium close-up")
        aimed = {(f.get("aim") or {}).get("to") for f in figs}
        # a close-up may leave a creature off-screen (the camera rule's own exception) unless someone aims at it
        crea = [f["who"] for f in figs if f.get("kind") == "creature" and (not close or f["who"] in aimed)]
        for ph in ("start", "end"):
            mp = rd / f"{ph}_color_creature_mask.png"
            if crea and mp.exists() and (np.asarray(Image.open(mp).convert("L")) > 127).mean() < 0.002:
                out[e["id"]] = f"{crea} not in the {ph} frame (creature mask empty)"
                break
        fp = rd / "figures0.png"
        if e["id"] not in out and any(f.get("kind") == "human" for f in figs) and fp.exists():
            fg = np.asarray(Image.open(fp).convert("L")).astype(int)
            rows = np.nonzero((fg > 8).any(1))[0]
            # a person READS when at least 3 % of the frame tall (10-22: a far wide of the dragon showed Edric 46 px tall,
            # under the old 0.1 %-of-pixels floor) - an empty frame or a walled-in camera still fails
            if (fg > 8).sum() < 100 or (rows.max() - rows.min() + 1 if len(rows) else 0) < 0.03 * fg.shape[0]:
                out[e["id"]] = "no person in the frame (figure mask empty)"
            elif (rd / "depth.mp4").exists():
                # the person VISIBLE, not just in the mask (10-22: Edric's forge strike rendered behind a wall and a beam
                # and passed): a figure pixel is seen where the scene's depth there is the figure's own (same scale)
                d0 = rd / ".depth0.png"
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(rd / "depth.mp4"), "-frames:v", "1", str(d0)])
                if d0.exists():
                    full = np.asarray(Image.open(d0).convert("L").resize(fg.shape[::-1])).astype(int)
                    m = fg > 8
                    seen = (np.abs(full - fg) <= 8) & m
                    if m.sum() and seen.sum() / m.sum() < 0.15:
                        out[e["id"]] = f"the people are hidden behind the set ({100 * seen.sum() / m.sum():.0f} % visible)"
    return out


def retry_failed(cx, log=print) -> int:
    """A FAILED marker gets ONE retry per run (10-21 dry run: one failed motion batch made all 66 action moments
    fail forever): stage/<id>/FAILED -> FAILED.prev at the start of a stage / video run."""
    n = 0
    for f in (cx.OUT / "stage").glob("*/FAILED") if (cx.OUT / "stage").exists() else []:
        f.replace(f.with_name("FAILED.prev"))
        n += 1
    if n:
        _log(f"retrying {n} moment(s) that failed in an earlier run (stage/<id>/FAILED.prev)", log)
    return n


def blocked_lines(R, P) -> list:
    """The preflight as status lines: 'blocked: 12 moments wait on prop sword (pending review)'."""
    from collections import defaultdict
    by = defaultdict(list)
    for e in P["entries"]:
        for u in (e.get("unready") or []) + (["no staged shot"] if e.get("route") == "missing" and not e.get("unready") else []):
            by[u].append(e["id"])
    lines = []
    for u, ids in sorted(by.items(), key=lambda kv: -len(kv[1])):
        m = re.match(r"prop '(.+)' has no approved mesh", u)
        reason = (f"prop {m.group(1)} (pending review)" if m else
                  u.replace("has no approved rig", "rig pending review").replace("no own model for", "no model yet for"))
        lines.append(f"blocked: {len(ids)} moment(s) wait on {reason}")
    return lines


def lora_lines(book, only=None) -> list:
    """Status lines for the cast still without a LoRA (their keys would be drawn without one)."""
    return [f"no LoRA: {n} ({why}): `bookcaster assets` trains it" for n, why in lora_todo(book, only).items()]


def ensure_loras(book, log=print, status_cb=None, only=None) -> list:
    """Train every missing LoRA before any key is drawn (iGPU, cached); returns the names still missing (loud)."""
    todo = sorted(lora_todo(book, only))
    if todo and not os.environ.get("BOOKCASTER_SKIP_LORA"):
        # nothing else runs yet: one character per card, all in parallel (user 10-20: XTXs for speed); a card's
        # later characters queue behind its flock
        cards = os.environ.get("BOOKCASTER_LORA_CARDS", "05,65").split(",")    # the iGPU (~3x slower) only by choice
        _log(f"LoRAs: training {todo} on {cards[:len(todo)]} before the keys", log)
        def one(nc):
            """One character: a failure is reported in the JOB log at once and retried once straight away with
            twice the pose drawings (10-21: King's LoRA failed at 12:24, 'only 3 pictures passed', and sat unseen
            for 80 min inside a running job)."""
            n, c = nc
            lg_ = vdir(book) / f"charlora_build_{c}.log"
            for k, n_pose in enumerate((40, 80)):
                rc_ = run(["flock", f"/tmp/gpu-{c}.lock", CHARLORA_PY, "-I", HERE / "charlora.py", "build", book, n,
                           "--card", c, "--n-pose", str(n_pose)], lg_)
                if rc_ == 0 and n not in lora_todo(book, only):
                    _log(f"LoRA {n}: done (card {c})", log)
                    return
                why = ""
                try:
                    why = [ln for ln in lg_.read_text().splitlines() if f"charlora {n}: FAILED" in ln][-1]
                except (OSError, IndexError):
                    pass
                _log(f"!!! LoRA {n} FAILED on card {c}{' (retrying with 80 pose drawings)' if k == 0 else ''}: "
                     f"{why[-160:]}", log)
                if status_cb:
                    status_cb(step=f"3d assets: LoRA {n} failed{', retrying' if k == 0 else ''}: {why[-100:]}")
        # one worker per CARD taking the next character as soon as it is free (10-22: the fixed i % 2 split queued
        # Edric behind the Knight on card 05 while card 65 sat idle ~30 min)
        import queue
        q = queue.Queue()
        for n in todo:
            q.put(n)

        def worker(c):
            while True:
                try:
                    n = q.get_nowait()
                except queue.Empty:
                    return
                one((n, c))
        with ThreadPoolExecutor(len(cards)) as ex:
            list(ex.map(worker, cards[:max(1, min(len(cards), len(todo)))]))
    left = lora_lines(book, only)
    for ln in left:
        _log("!!! " + ln, log)
    return left


def refresh_stale_keys(R, cx, P) -> int:
    """All-new also means keys (10-21: a talk key from the night before, drawn without Hulda's LoRA, was reused).
    A key older than its scene-layer shot, its talk framing (shot_talk.json as talk_shot would write it now), or
    the LoRA of anyone in frame is moved to stage/<id>/stale_<stamp>/ with its still render, so it is redrawn."""
    import filecmp
    import tempfile
    stamp, n = time.strftime("%m%d-%H%M"), 0
    for e in P["entries"]:
        d = cx.OUT / "stage" / e["id"]
        k = d / "key.png"
        if not k.exists():
            continue
        kt, why = k.stat().st_mtime, []
        rc_ = d / "render" / "start_color.png"     # drawn over an OLDER staging render (10-22: 16 cameras re-staged out
        if rc_.exists() and rc_.stat().st_mtime > kt:  # of the set's rock after their keys were drawn)
            why.append("render")
        if e.get("shot") and Path(e["shot"]).exists():
            # the shot's CONTENT decides (10-21 dry run: scene_stage rewrites every shot each run, so mtime alone
            # would redraw all 95 keys); the key's shot hash is recorded the first time it is seen current
            import hashlib
            sh = hashlib.sha1(Path(e["shot"]).read_bytes()).hexdigest()[:16]
            rec = d / "key.shot.sha"
            if rec.exists():
                if rec.read_text().strip() != sh:
                    why.append("shot")
            elif Path(e["shot"]).stat().st_mtime > kt:
                why.append("shot")
            else:
                rec.write_text(sh)
        for w in {*(e.get("who") or []), *([e["speaker"]] if e.get("speaker") else [])}:
            lp = R.char_lora(cx, w)
            if lp and Path(lp).stat().st_mtime > kt:
                why.append(f"lora:{w}")
        if e["route"] == "talk" and e.get("shot") and R.TALK_SINGLE:
            with tempfile.TemporaryDirectory() as td:
                now = Path(R.talk_shot(cx, e, Path(td)))
                if now.parent == Path(td) and (not (d / "shot_talk.json").exists()      # still made without the single
                                               or not filecmp.cmp(now, d / "shot_talk.json", shallow=False)):
                    why.append("talk framing")
        if why:
            arc = d / f"stale_{stamp}"
            arc.mkdir(exist_ok=True)
            for f in [*d.glob("key*.png"), d / "face.json", d / "render", d / "shot_talk.json", d / "key.shot.sha"]:
                if f.exists():
                    shutil.move(str(f), arc / f.name)
            (arc / "WHY").write_text(", ".join(why))
            n += 1
    return n


def thin_lines(R, cx, P) -> list:
    """Moments staged from the moment TEXT alone (no scene-layer shot: no props/states, contacts, fx, recipe) - loud
    (10-21: the whole knight ch1 rendered this way while the scene layer existed; the forge lost its recipe)."""
    thin = [e["id"] for e in P["entries"] if e.get("route") in ("stage", "missing") and e.get("shot")
            and Path(e["shot"]).parent != cx.V / "stage" / "shots"]
    return [f"thin text staging (no scene-layer shot) for {len(thin)} moment(s): {thin[:6]}"] if thin else []


def preflight(book, chapter) -> list:
    R, cx = _ctx(Path(book), chapter_stem(Path(book), chapter))
    P = R.plan(cx)
    return blocked_lines(R, P) + thin_lines(R, cx, P) + lora_lines(book)


def review(book, chapter=None, log=print) -> dict:
    """One index of every QA sheet (characters, creatures, props, sets, scenes, keyframes) + the pending list."""
    book = Path(book)
    vd = vdir(book)
    ap = register(book)
    sheets = {k: e.get("qa") for k, e in ap["items"].items() if e.get("qa")}
    for st in ([chapter_stem(book, chapter)] if chapter else [p.name for p in vd.iterdir() if (p / "stage" / "scenes").exists()]):
        for j in sorted((vd / st / "stage" / "scenes").glob("*.jpg")):
            sheets[f"scene:{st}/{j.stem}"] = str(j)
        for j in sorted((vd / st / "flow3d").glob("*keys*sheet*.jpg")) if (vd / st / "flow3d").exists() else []:
            sheets[f"keys:{st}/{j.stem}"] = str(j)
    pend = [k for k, e in ap["items"].items() if e["status"] == "pending"]
    rv = vd / "stage" / "review.json"
    rv.write_text(json.dumps({"sheets": sheets, "pending": pend, "at": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    _log(f"review: {len(sheets)} sheet(s), {len(pend)} item(s) pending -> {rv}", log)
    for k in pend:
        _log(f"  pending {k}  {ap['items'][k].get('qa') or ''}", log)
    return {"sheets": sheets, "pending": pend}


def _pp_split(out: Path, route: str, default: int) -> int:
    """The split pp_fit suggests for a route from this chapter's own timing (pp_timing.jsonl), else from the book's
    other chapters (10-26: a fresh chapter's first quiet batch ran at the default), else the default."""
    tj = out / "ltx" / "out" / "pp_timing.jsonl"
    best = _pp_fit([tj]) if tj.exists() else {}
    if route not in best:                   # newest chapter first: today's links/clocks beat a bigger old sample
        for f in sorted(out.parent.parent.glob(f"*/{out.name}/ltx/out/pp_timing.jsonl"),
                        key=lambda f: f.stat().st_mtime, reverse=True):
            best = _pp_fit([f])
            if route in best:
                break
    return best.get(route, default)


def _pp_fit(files: list) -> dict:
    """pp_fit's BEST --split per route over these timing files, decode on card B (the default for every route)."""
    if not files:
        return {}
    r = subprocess.run([str(LTX_PY), str(HERE / "ltx_runner" / "pp_fit.py"), "--decode-b", *map(str, files)],
                       capture_output=True, text=True)
    best, cur = {}, None
    for ln in r.stdout.splitlines():
        m = re.match(r"\[(\w+)\]", ln)
        if m:
            cur = m.group(1)
        m = re.search(r"BEST --split (\d+)", ln)
        if m and cur:
            best[cur] = int(m.group(1))
    return best


def estimate_minutes(P: dict) -> float:
    """Box minutes of the 3D render of a planned chapter (staging + keys + LTX), measured 10-20 on knight ch1."""
    E = P["entries"]
    n_stage = sum(1 for e in E if e["route"] == "stage")
    n_talk_chunks = sum(len(e.get("chunks") or [1]) for e in E if e["route"] == "talk")
    return round(5 + n_stage * 4.0 + len(E) * 0.6 + n_talk_chunks * 3.0 + n_stage * 1.3 + 25, 0)


def holds(book, stem) -> dict:
    """stage/holds.json {"scenes": [...], "moments": [...], "why": {...}}: moments the reviewer holds back (pending a
    re-stage review); they render on a later pass once removed."""
    p = vdir(book) / "stage" / "holds.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    return d.get(stem, d) if isinstance(d, dict) else {}


def audit(R, cx, P, files: dict) -> list:
    """The CPU audit before any GPU second (user 10-20): read every LTX job + prompt + key input. -> problems
    (hard ones block the hold: quoted words, a missing first image, a chained chunk pinned at both ends)."""
    from PIL import Image
    probs = []
    jobs = json.loads(files["talk_jobs"].read_text()) + json.loads(files["act_jobs"].read_text())
    for j in jobs:
        p = j.get("prompt", "")
        if re.search(r"[\"“”]", p):
            probs.append(("HARD", j["name"], "quoted words in the prompt"))
        if j.get("audio") is not None or j["name"].startswith("r_talk"):
            m_ = re.search(R.PROP_WORDS, p, re.I)
            if m_:              # LTX draws described props that the key doesn't show (382's floating plaque)
                probs.append(("HARD", j["name"], f"prop words in a talk prompt: '{m_.group(0)}'"))
        if "stormy" in p.lower() and "inside" in p.lower():
            probs.append(("HARD", j["name"], "sky words in an interior prompt"))
        if j.get("chain_from") and j.get("lock_end"):
            probs.append(("HARD", j["name"], "chained chunk with an end lock (frozen lips)"))
        if j.get("chain_from") and j.get("chain_strength", 1.0) > 0.85 and "a2v" not in j["name"] and j.get("audio"):
            probs.append(("HARD", j["name"], f"chained talk chunk at strength {j.get('chain_strength', 1.0)}"))
        for im in j.get("images", []):
            if not Path(im["path"]).exists():
                probs.append(("HARD", j["name"], f"missing image {im['path']}"))
            else:
                l = sum(Image.open(im["path"]).convert("L").resize((64, 36)).getdata()) / (64 * 36)
                if l < 35:
                    probs.append(("soft", j["name"], f"very dark first image (luma {l:.0f})"))
        if j.get("control") and not Path(j["control"]["path"]).exists():
            probs.append(("HARD", j["name"], "missing control video"))
    for e in P["entries"]:
        for w in R.moment_who(cx, e):
            mo = re.search(r"wearing ([^.;]*)", cx.look(w, e["t"]) or "")
            prm = R.talk_prompt(cx, e) if e["route"] == "talk" else R.i2v_prompt(cx, e, True)
            if mo and mo.group(1).split(",")[0][:25].lower() not in prm.lower() and not R.is_creature(cx.chars.get(w, "")):
                probs.append(("soft", e["id"], f"{w}'s outfit words not in the prompt"))
    return probs


# what a redraw can fix at key strength 0.375. Outfit/prop readings follow the 3D model + staging (Hulda's model reads
# as a leather jacket in 19 of 19 keys, 10-20): they are logged as ASSET findings, never redraw loops
HARD_KEY = ("facial hair", "speaker ", "location", "text", "white:", "back-turned:", "period:")


def keys_gate(R, cx, part, log=print) -> dict:
    """The fast key checks; a failing key is redrawn ONCE (the headshot + outfit-sheet refs, the facts in the prompt),
    the better of the two kept; one still failing is returned (left out of LTX, listed for a fix in code)."""
    chk = R.talk_checks if R.FULL_CHECKS else R.quick_key_checks
    # a moment whose clip is already current is settled: no re-judging (10-21 dry run: on a re-run, judge noise
    # redrew settled keys, archived their clips and re-rendered them)
    part = {**part, "entries": [e for e in part["entries"]
                                if not R.clip_current(cx, e["id"], cx.OUT / "ltx" / "out" / f"r_{e['id']}.mp4")]}
    if not part["entries"]:
        return {}
    res = chk(cx, part)
    hard = lambda r: [f for f in r.get("flags", []) if f.startswith(HARD_KEY)]  # noqa: E731
    soft = {i: [f for f in r.get("flags", []) if f.startswith(("outfit:", "prop:"))] for i, r in res.items()}
    soft = {i: f for i, f in soft.items() if f}
    if soft:
        (cx.OUT / "asset_findings.json").write_text(json.dumps(soft, indent=1))
        _log(f"keys: {len(soft)} key(s) with outfit/prop readings that follow the 3D model (asset findings, no redraw)", log)
    bad = {i: hard(r) for i, r in res.items() if hard(r)}
    if not bad:
        return {}
    for i in bad:
        d = cx.OUT / "stage" / i
        for a, b in (("key.png", "key_try1.png"), ("key_draw.png", "key_draw_try1.png")):
            if (d / a).exists():
                shutil.move(str(d / a), d / b)
        for f in ("key_bg.png", "key_comp.png", "key_mask.png", "key_preface.png", "face.json", "key_draw.png"):
            (d / f).unlink(missing_ok=True)
    sub = {**part, "entries": [e for e in part["entries"] if e["id"] in bad]}
    _log(f"keys: {len(bad)} failing key(s) redrawn once: {sorted(bad)}", log)
    R.stage(cx, sub, steps="draw", workers=3, seeds=R.KEY_SEEDS, card="any")
    R.bg_hold(cx, sub)
    res2 = chk(cx, sub)
    still = {}
    for i, f1 in bad.items():
        d = cx.OUT / "stage" / i
        f2 = hard(res2.get(i, {"flags": ["no redraw"]})) if (d / "key.png").exists() else ["no redraw"]
        if len(f1) < len(f2) and (d / "key_try1.png").exists():      # the first draw was better: keep it
            shutil.copy(d / "key_try1.png", d / "key.png")
            f2 = f1
        if f2:
            still[i] = f2
    (cx.OUT / "keys_failing.json").write_text(json.dumps(still, indent=1))
    for i, f in still.items():
        _log(f"key {i} still fails: {f}", log)
    return still


def video(book, chapter, log=print, status=None, auto: bool = False, jellyfin: bool = False,
          keys_only: bool = False) -> Path | None:
    """The final 3D render of every UNBLOCKED, un-held moment (blocked/held ones are listed and render on a later
    pass): talk first (stills -> keys -> fast checks -> A2Vid), while the action moments' motion/renders run on the
    CPU beside it; then action (keys -> fast checks -> Mode 2); QA with one re-roll; assemble on the score + sfx mix
    -> out/<book>_<chapter>_<stamp>_3d.mp4. A CPU audit of every job file precedes each hold."""
    JOB_LOG[0] = log
    disk_check(book, log)
    import threading
    book = Path(book)
    stem = chapter_stem(book, chapter)
    status = status or (lambda **k: None)
    if auto:
        auto_approve(book)
    no_lora = ensure_loras(book, log)
    R, cx = _ctx(book, stem)
    retry_failed(cx, log)
    P = R.plan(cx)
    # one job per book (its systemd unit): a render claim left at start belongs to a dead process (10-21: claims left
    # by a stopped job made the stage step skip 20 talk stills silently)
    for cl in (cx.OUT / "stage").glob("*/.rendering") if (cx.OUT / "stage").exists() else []:
        cl.unlink(missing_ok=True)
    if R.strict():       # all-new delivery (user 10-21): stale keys redrawn, clips not made from the current key archived
        n_st = refresh_stale_keys(R, cx, P)
        if n_st:
            _log(f"strict: {n_st} stale key(s) set aside (older than their shot / talk framing / a LoRA): redrawn", log)
        lo = cx.OUT / "ltx" / "out"
        old = [f for f in (sorted(lo.glob("r_*")) if lo.exists() else [])
               if not R.clip_current(cx, re.sub(r"(_c\d+)?(\.[a-z0-9_.]+)$", "", f.name[2:]), f)]
        if old:
            arc = cx.OUT / "ltx" / f"out_prev_{time.strftime('%m%d-%H%M')}"
            arc.mkdir(parents=True, exist_ok=True)
            for f in old:
                shutil.move(str(f), arc / f.name)
            _log(f"strict: {len(old)} clip file(s) not made from their current key archived to {arc.name}", log)
    bl = blocked_lines(R, P) + thin_lines(R, cx, P) + no_lora
    hd = holds(book, stem)
    held_sc, held_m = set(hd.get("scenes", [])), set(hd.get("moments", []))

    def scene_of(e):
        try:
            return (json.loads(Path(e["shot"]).read_text()).get("scene") or {}).get("id")
        except (OSError, ValueError, TypeError):
            return None
    ready = [e for e in P["entries"] if not e.get("unready") and e["route"] != "missing"
             and e["id"] not in held_m and scene_of(e) not in held_sc]
    if hd.get("only") == "solo":     # holds.json "only": "solo" - just the one-person, creature-free shots this pass
        def solo(e):
            try:
                figs = [g.get("who") for g in json.loads(Path(e["shot"]).read_text()).get("figures", [])]
            except (OSError, ValueError, TypeError, KeyError):
                return False
            if hd.get("only_who") and not set(figs) <= set(hd["only_who"]):
                return False
            return len(figs) <= 1 and not any(R.is_creature(cx.chars.get(w or "", "")) for w in figs)
        n_multi = sum(1 for e in ready if not solo(e))
        ready = [e for e in ready if solo(e)]
        _log(f"holds 'only: solo': {n_multi} multi-person/creature moment(s) wait for the next pass", log)
    if hd.get("skip_routes"):        # holds.json "skip_routes": ["talk"] - a whole route waits for a fix (10-20)
        n_sk = sum(1 for e in ready if e["route"] in hd["skip_routes"])
        ready = [e for e in ready if e["route"] not in hd["skip_routes"]]
        _log(f"holds 'skip_routes' {hd['skip_routes']}: {n_sk} moment(s) wait", log)
    n_held = sum(1 for e in P["entries"] if e["id"] in held_m or scene_of(e) in held_sc)
    for ln in bl:
        _log(ln, log)
    _log(f"{len(ready)} of {len(P['entries'])} moments ready; {n_held} held for review", log)
    n_block = len(P["entries"]) - len(ready) - n_held
    if auto and n_block > 0 and not hd:
        # unattended: a partial video is never the delivery (10-22: 5 of 91 moments went on to LTX)
        raise SystemExit(f"video --auto-approve: {n_block} moment(s) blocked ({'; '.join(bl[:3])}): fix the assets "
                         f"first, a partial video is not made unattended")
    status(step=f"3d: {len(ready)} ready, " + ("; ".join(bl[:2]) + "; " if bl else "") + f"{n_held} held")
    if not ready:
        return None
    talk = {**P, "entries": [e for e in ready if e["route"] == "talk"]}
    act = {**P, "entries": [e for e in ready if e["route"] == "stage"]}
    os.environ["REMAKE_LTX_SPLIT_TALK"] = str(_pp_split(cx.OUT, "a2vid", 25))
    os.environ["REMAKE_LTX_SPLIT_ACT"] = str(_pp_split(cx.OUT, "ic", 17))
    R.PP = True                             # lanes/stagger/pool/decode-b = R.LTX_PP_ARGS (10-26: one default set)
    # the action moments' motion + renders (CPU) run while the talk moments draw and render
    bg_err = []

    def _bg():                       # a SystemExit (memory, kernel hit) in a thread is otherwise dropped (10-21)
        try:
            # the clips the stage step left as stills: ltx_wait() starts none while an LTX gen runs, so they fill the
            # key-painting time (cores-sized, like the stage step)
            R.stage(cx, act, steps="motion,render", workers=max(2, (os.cpu_count() or 16) // 5), card="any", clips=True)
        except BaseException as ex:  # noqa: BLE001
            bg_err.append(ex)
    bg = threading.Thread(target=_bg, daemon=True)
    bg.start()
    summ_all = {"fail": {}}
    # READY FIRST (user 10-20: clips ASAP): moments whose key already exists go to checks -> LTX before any still or
    # key is made for the rest; then talk, then the remaining action (after its renders)
    has_key = lambda e: (cx.OUT / "stage" / e["id"] / "key.png").exists()  # noqa: E731
    parts = [("talk", {**P, "entries": [e for e in talk["entries"] if has_key(e)]}),
             ("action", {**P, "entries": [e for e in act["entries"] if has_key(e)]}),
             ("talk", {**P, "entries": [e for e in talk["entries"] if not has_key(e)]}),
             ("action", {**P, "entries": [e for e in act["entries"] if not has_key(e)]})]
    for n_, (kind, part) in enumerate(parts):
        if not part["entries"]:
            continue
        if kind == "action" and n_ == 3:
            bg.join()
            if bg_err:
                raise SystemExit(f"the action motion/render thread failed: {bg_err[0]}")
        status(step=f"3d {kind}: stills + keyframes ({len(part['entries'])})")
        (cx.OUT / ".renders_done").touch()       # draw whatever is ready (no batching wait inside the job)
        R.stage(cx, part, steps="render,draw" if kind == "talk" else "draw", workers=3, seeds=R.KEY_SEEDS, card="any")
        R.bg_hold(cx, part)
        bad = keys_gate(R, cx, part, log)
        if bad:
            # a key still failing a HARD check after its redraw never renders (10-21: rendering them anyway put Hulda
            # as an old man / a dark blob on screen). Under strict the moment is then MISSING (listed, the chapter
            # _INCOMPLETE), never an old clip. BOOKCASTER_RENDER_FAILING_KEYS=1 renders them anyway (loud).
            drop = {i for i in bad if not os.environ.get("BOOKCASTER_RENDER_FAILING_KEYS")
                    or not (cx.OUT / "stage" / i / "key.png").exists()}
            part = {**part, "entries": [e for e in part["entries"] if e["id"] not in drop]}
            if len(bad) > len(drop):
                _log(f"!!! {len(bad) - len(drop)} key(s) still failing render anyway: {sorted(set(bad) - drop)}", log)
            if drop:
                _log(f"!!! {len(drop)} key(s) still failing a hard check: not rendered (listed): {sorted(drop)}", log)
            status(step=f"3d {kind}: {len(bad) - len(drop)} failing key(s) rendered anyway, {len(drop)} left out "
                        f"(no key): {sorted(drop)[:4]}")
        if keys_only:                # user 10-21: every key reviewed (the preview) before any LTX hour
            continue
        if kind == "action":         # Mode 2 needs the moment's whole depth clip (the stage step rendered stills only)
            bg.join()
            if bg_err:
                raise SystemExit(f"the action motion/render thread failed: {bg_err[0]}")
        files = R.make_jobs_files(cx, part)
        probs = audit(R, cx, part, files)
        (cx.OUT / f"audit_{kind}.json").write_text(json.dumps(probs, indent=1))
        hard = [p for p in probs if p[0] == "HARD"]
        for p in probs[:20]:
            _log(f"audit {p[0]} {p[1]}: {p[2]}", log)
        if hard:
            status(state="failed", step=f"3d {kind}: the job audit found {len(hard)} hard problem(s) — {hard[0][1]}: {hard[0][2]}")
            raise SystemExit(f"audit: {len(hard)} hard problem(s) in the {kind} jobs (see {cx.OUT}/audit_{kind}.json)")
        status(step=f"3d {kind}: LTX ({len(part['entries'])} moments)")
        R.ltx(cx, part, which=kind)
        summ = R.qa(cx, part, reroll=True)
        if summ.get("fail"):
            status(step=f"3d {kind}: re-rolling {len(summ['fail'])} clip(s) once")
            R.ltx(cx, part, which=kind)
            summ = R.qa(cx, part)
        summ_all["fail"].update(summ.get("fail") or {})
    bg.join()                        # always (part 3 may be empty), and its error is the job's error
    if bg_err:
        raise SystemExit(f"the action motion/render thread failed: {bg_err[0]}")
    # the chapter's mix with the sound effects (a copy; work/final stays as recorded); remade when the final mix is newer
    audio = cx.OUT / "audio" / "score_sfx.mp3"
    fin = book / "work" / "final" / f"{stem}.mp3"
    if not audio.exists() or (fin.exists() and fin.stat().st_mtime > audio.stat().st_mtime):
        audio = _mix_with_sfx(book, stem, cx.OUT / "audio", log)
    if keys_only:
        import keys_preview
        pv = keys_preview.build(book, stem, Path(audio))
        nk = json.loads(pv.with_suffix(".json").read_text())
        _log(f"keys only: preview {pv} ({nk['moments']} moments, {nk['no_key']} without a key); LTX waits for "
             f"`bookcaster video {book.name} --flow 3d`", log)
        status(step=f"3d: keyframes ready for review — {pv.name} ({nk['no_key']} moment(s) without a key)")
        return pv
    os.environ["REMAKE_AUDIO"] = str(audio)
    os.environ["REMAKE_SUFFIX"] = "3d"
    status(step="3d: assembling" + (" (strict: moments without a new clip are left out, listed)" if R.strict()
                                     else " (moments not rendered yet keep their old clips)"))
    mp4 = R.assemble(cx, P)
    R.sheet(cx, P)
    fb = json.loads((cx.OUT / "FALLBACKS.json").read_text()) if (cx.OUT / "FALLBACKS.json").exists() else []
    _log(f"wrote {mp4}" + (f"; QA still flags {sorted(summ_all['fail'])}" if summ_all.get("fail") else "") +
         (f"; {len(fb)} fallback(s)" if fb else ""), log)
    miss = json.loads((cx.OUT / "MISSING.json").read_text()) if (cx.OUT / "MISSING.json").exists() else []
    if miss:             # strict: a chapter with any gap is never delivered
        _log(f"!!! INCOMPLETE: {len(miss)} moment(s) have no new 3D clip: {miss[:10]} - not delivered", log)
        status(step=f"3d: INCOMPLETE, {len(miss)} moment(s) without a new 3D clip (see MISSING.json): {miss[:4]}")
        jellyfin = False
    if jellyfin:
        try:
            from tts_reader import jellyfin as J
            meta = json.loads((book / "meta.json").read_text()) if (book / "meta.json").exists() else {}
            num = re.match(r"\d+", stem)
            J.send(mp4, meta.get("title") or book.name, f"{num.group(0) + ' ' if num else ''}Chapter {int(num.group(0)) if num else ''} (3D).mp4", log=log)
        except Exception as ex:  # noqa: BLE001
            _log(f"jellyfin: not sent ({ex})", log)
    return mp4


def _mix_with_sfx(book: Path, stem: str, outdir: Path, log=print) -> Path:
    """work/final/<stem>.mp3 + the chapter's planned sound effects -> a copy in outdir (generation runs only for
    effects not made yet; the score must already be in the final mix)."""
    from tts_reader import sfx
    outdir.mkdir(parents=True, exist_ok=True)
    W = book / "work"
    base = outdir / "score.mp3"
    nosfx = W / "final" / f"{stem}.nosfx.mp3"           # a book mixed with --sfx: start from the mix without them
    shutil.copy(nosfx if nosfx.exists() else W / "final" / f"{stem}.mp3", base)
    sd = W / "sfx" / stem
    out = outdir / "score_sfx.mp3"
    if not (sd / "plan.json").exists():
        _log("no sound-effect plan for this chapter: the score mix is used as is", log)
        shutil.copy(base, out)
        return out
    plan = json.loads((sd / "plan.json").read_text())
    sfx.generate(plan, sd, log)
    for n, b in enumerate(plan["beds"]):
        b["file"] = str(sd / f"bed{n:03d}.wav")
    for n, s in enumerate(plan["spots"]):
        s["file"] = str(sd / f"spot{n:03d}.wav")
    sfx.mix(base, W / f"{stem}.mp3", plan, out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["plan", "assets", "stage", "review", "approve", "preflight", "video"])
    ap.add_argument("book")
    ap.add_argument("items", nargs="*")
    ap.add_argument("--chapter")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--auto", "--auto-approve", dest="auto", action="store_true")
    ap.add_argument("--reject", action="store_true")
    ap.add_argument("--jellyfin", action="store_true")
    a = ap.parse_args(argv)
    book = Path(a.book).expanduser()
    if a.cmd == "assets":
        print(json.dumps(assets(book, a.chapter, plan_only=a.plan_only), indent=1))
    elif a.cmd == "stage":
        r = stage(book, a.chapter, plan_only=a.plan_only)
        print("\n".join(r.get("blocked", [])) or "staged")
    elif a.cmd == "review":
        review(book, a.chapter)
    elif a.cmd == "approve":
        done = auto_approve(book) if a.auto else approve(book, a.items, "rejected" if a.reject else "approved")
        print(f"{'rejected' if a.reject else 'approved'}: {done}" if done else "nothing matched")
    elif a.cmd == "preflight":
        print("\n".join(preflight(book, a.chapter)) or "ready")
    elif a.cmd == "video":
        return 0 if video(book, a.chapter, auto=a.auto, jellyfin=a.jellyfin) else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
