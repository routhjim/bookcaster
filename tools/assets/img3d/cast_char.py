"""One command: a posable character from its story sheet by the combo route, no hand choices (caller side, stdlib;
every GPU or engine step goes through the asset line).

    python tools/assets/img3d/cast_char.py --id LIB_ID --sheet SHEET.png --look "a man of about 60, ..." \
        --lora LORA.safetensors --trigger chr_name --work DIR [--scene-tool DIR] [--from STEP]

Steps (each skipped when its output exists, so a re-run resumes; --from STEP redoes from there):
  choose   image station: choose_mh.py measures the sheet: preset, grooms, hair colour, build (choice.json)
  mh       engine: char_render make + cloud (rig, textures, assembly) + export_geometry (body and head FBX)
  draw     image station: spread-pose drawings, the sheet as reference, the character's LoRA, an OpenPose map of
           the 40-degree spread pose (apose_map.py); prompt = trigger + the look line + fixed pose and studio words;
           three seeds a round
  gate     pose_gate.py on each drawing; the best passing one (largest arm angle x armpit) goes on; none passing:
           the next round of seeds (up to --rounds), never a drawing under the gate's thresholds
  shape    blender station (card 05): img3d.py (TRELLIS.2 + inner wall)
  combo    iGPU station: bl_combo.py with rules measured on the drawing and read from the look line:
           long skirt when the legs show no air between them at shin height (then: hem not a shoe, sleeve slim,
           a dress collar with a white edge and cuffs); otherwise the coat route (skirt from the pelvis when the
           cloth below the hips stands off the legs at mid-thigh, a constructed collar: a tie when the look says
           "tie", else a cravat/bow when it says "cravat" or "collar")
  import   engine: the outfit, the character's own masked face skin (char_skinmask.py, the fit's mask), a library
           entry (character.json) on the assembled Blueprint wearing both
  arms     engine + image station: one relaxed scene with three copies at 4 / 8 / 12 degrees of arm abduction,
           the scene tool's guides, arm_gap.py: the smallest abduction whose forearms separate from the body in
           >= 8 of 10 rows on both sides at the wide shot goes into character.json "relaxed_arm_steps"
  check    engine: wide / medium / close stills in the set (check.png); flat-lit studio stills with and without
           the face mesh, skin_probe.py: no skin-coloured head mesh at the shoulders (fails the build)
Writes WORK/times.json (seconds per step) and WORK/choices.json (every automatic choice and its measurement).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent
REPO = ASSETS.parents[1]
sys.path.insert(0, str(ASSETS))
import config as C  # noqa: E402

LINE = [sys.executable, str(ASSETS / "run_asset.py")]
CARD = str(ASSETS / "orbit" / "card.sh")
BLENDER = str(C.BLENDER)
HY_PY = os.path.expanduser(os.environ.get("ASSETS_IMG3D_TORCH_PY", "~/tts-ab/.venv-hy3d/bin/python"))
QWEN_PY = os.path.expanduser(os.environ.get("ASSETS_QWEN_PY", "~/tts-ab/.venv-qwenimage/bin/python"))
GUIDES_PY = os.path.expanduser(os.environ.get("ASSETS_GUIDES_PY", "~/tts-ab/.venv-hyworld/bin/python"))
OMP = os.environ.get("ASSETS_TRELLIS_OMP", "")
PROJECT = str(Path(C.UE_HOME) / "proj" / "CharLab" / "CharLab.uproject")
THUMBS = str(C.MH_PRESET_PORTRAITS)      # neutral studio portraits of every shipped preset (config.py)
HEAD_VOLUME_MAX = 150      # garment vertices allowed in front of the face above the chin (a collar edge, a curl)
GROWN_AGE = 13             # from this age a figure has a grown person's proportions


def age_of(look: str) -> int | None:
    m_ = __import__("re").search(r"of about (\d+)", look or "")
    return int(m_.group(1)) if m_ else None


def child_armpit(look: str) -> float:
    """The drawing gate's armpit height (share of the figure's height) for this person: a child's head is a larger
    share of the height, so the armpit sits lower (every drawing of a seven-year-old failed a grown person's 0.70)."""
    age = age_of(look)
    return round(0.70 - 0.006 * max(0, GROWN_AGE - age), 3) if age else 0.70


STEPS = ["choose", "mh", "draw", "gate", "shape", "combo", "import", "arms", "seats", "check"]
POSE_WORDS = ("standing in a neutral rigging pose, seen straight from the front: both arms held straight out away "
              "from the body, angled down about 45 degrees, a clear open gap between each arm and the clothes, hands "
              "open with the fingers together and the palms facing down, legs slightly apart, looking at the viewer. "
              "Full length from the top of the head to the feet, the whole figure in the picture with a margin, "
              "bare-headed (no hat). Plain flat light grey background, soft even studio light, no shadow on the floor, "
              "nothing else in the picture. A realistic full-colour photograph of a costumed actor, sharp focus.")


def log(m):
    print(f"[cast {time.strftime('%H:%M:%S')}] {m}", flush=True)


def line(args, log_file=None):
    """One line entry, followed to its end. On LINE FULL the entry goes to the line's own feeder (run_asset.py
    --feed waits for room by watching the status, never resubmitting blindly), then this follows the entry's done
    record (<sched>/line/done/*.json, by id and submit time)."""
    if log_file and "--line-cmd" in args:
        i = args.index("--")
        args = args[:i] + ["--log", str(log_file)] + args[i:]
    t_sub = time.time()
    p = subprocess.run(LINE + args, cwd=str(REPO), capture_output=True, text=True)
    if p.returncode not in (75, 76):
        return p
    import shlex
    eid = args[args.index("--id") + 1] if "--id" in args else None
    if p.returncode == 75:
        (Path(C.SCHED_DIR) / "feeds").mkdir(exist_ok=True)
        lst = Path(C.SCHED_DIR) / "feeds" / f"{eid or int(t_sub)}_{int(t_sub)}.txt"
        lst.write_text(shlex.join(args) + "\n")
        f = subprocess.run(LINE + ["--feed", str(lst)], cwd=str(REPO), capture_output=True, text=True)
        if f.returncode:
            return f
    if not eid:
        raise SystemExit(f"entry queued without an id to follow (exit {p.returncode})")
    done = Path(C.SCHED_DIR) / "line" / "done"
    while True:                                  # the entry's own end, on disk (not the line's queue)
        for q in sorted(done.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:200]:
            try:
                d = json.loads(q.read_text())
            except ValueError:
                continue
            if d.get("id") == eid and d.get("submitted", 0) >= t_sub - 3600 and d.get("state") in ("done", "failed"):
                rc = (d.get("result") or {}).get("rc", 1 if d.get("state") == "failed" else 0)
                return subprocess.CompletedProcess(args, rc, stdout=json.dumps(d.get("result")), stderr="")
        time.sleep(10)


def mem_wait(gb=25, max_s=5400):
    """Until the machine has GB of memory available (MemAvailable), at most MAX_S seconds."""
    t0 = time.time()
    while time.time() - t0 < max_s:
        try:
            kb = int(next(ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")).split()[1])
        except (OSError, StopIteration, ValueError):
            return
        if kb / 1048576 >= gb:
            return
        time.sleep(30)


def ue_job(work, name, script, job, timeout=1800):
    d = Path(work) / "ue" / name
    (d / "sched").mkdir(parents=True, exist_ok=True)
    job.setdefault("result", str(d / "result.json"))
    sj = {"project": PROJECT, "script": script, "job": job, "outputs": [job["result"]], "timeout_s": timeout,
          "id": f"cast_{Path(work).name}_{name}", "log_dir": str(d / "sched")}
    jp = d / "job.json"
    jp.write_text(json.dumps(sj, indent=1))
    for attempt in range(3):
        # the editor dies when the machine runs short of memory (another tenant's language model on the iGPU,
        # 11-11): wait for 25 GB available and run the job again, twice at most; any other failure stops at once
        mem_wait()
        p = line(["--ue-job", str(jp)])
        if not p.returncode or "the editor died" not in (p.stdout + p.stderr):
            break
        log(f"engine job {name}: the editor died (attempt {attempt + 1}); waiting for memory")
    if p.returncode:
        raise RuntimeError(f"engine job {name} failed: {p.stdout[-1500:]} {p.stderr[-800:]}")
    return json.loads(Path(job["result"]).read_text())


def legs_apart(png):
    """Measured on the drawing: is there background between the legs at shin height (trousers), or one
    run of figure (a floor-length skirt)? Also: does the cloth below the hips stand off the legs (a coat's skirt)?"""
    from PIL import Image  # noqa: WPS433 (Pillow, pasting/measuring only)
    im = Image.open(png).convert("RGB")
    w, h = im.size
    px = im.load()
    border = [px[x, 0] for x in range(w)] + [px[x, h - 1] for x in range(w)]
    bg = tuple(sorted(c[i] for c in border)[len(border) // 2] for i in range(3))

    def fg(x, y):
        c = px[x, y]
        return sum((c[i] - bg[i]) ** 2 for i in range(3)) ** 0.5 > 40

    rows = [y for y in range(h) if any(fg(x, y) for x in range(0, w, 4))]
    top, bot = min(rows), max(rows)
    H = bot - top

    def runs_at(frac):
        y = int(bot - frac * H)
        r, on = [], False
        for x in range(w):
            f = fg(x, y)
            if f and not on:
                r.append([x, x])
            if f:
                r[-1][1] = x
            on = f
        return [q for q in r if q[1] - q[0] > 6]
    shin = runs_at(0.18)
    thigh = runs_at(0.40)
    mid = [q for q in thigh if q[0] < w / 2 < q[1]]
    return {"shin_runs": len(shin), "long_skirt": len(shin) < 2,
            "thigh_run_w_frac": round((mid[0][1] - mid[0][0]) / H, 3) if mid else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--look", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--trigger", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--scene-tool", default=os.environ.get("ASSETS_SCENE_TOOL", str(ASSETS / "scenes")))
    ap.add_argument("--set", default="sitting_room_victorian_01")
    ap.add_argument("--lie", action="store_true", help="also lying stances on the library's beds")
    ap.add_argument("--lie-beds", nargs="*", default=None, help="only these bed props")
    ap.add_argument("--lie-variants", nargs="*", default=["flat", "propped"], choices=["flat", "propped"])
    ap.add_argument("--lie-spot", nargs=3, type=float, default=[0.3, -0.6, 0.0],
                    help="where the bed stands in the set for the lab measure: x y (metres), and its yaw: 0 = foot "
                         "toward -Y")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--pose-deg", type=float, default=None, help="the first round's arm angle (default 42, 48 for a dress)")
    ap.add_argument("--whole", action="store_true",
                    help="the whole generated figure (its own head, hair, hands and clothes, one mesh) on the "
                         "MetaHuman's skeleton, the MetaHuman's body and face not shown: no join between a MetaHuman "
                         "head and generated clothes (11-14: the redrawn key from it kept the sheet's own garment)")
    ap.add_argument("--from", dest="from_step", default=None, choices=STEPS)
    ap.add_argument("--until", default=None, choices=["face", "seats"],
                    help="stop after this step (face: after the conform; seats: after the seated stances)")
    ap.add_argument("--conform", action="store_true",
                    help="conform the picked face to the drawing (about 9 min; kept only if it wins the comparison)")
    ap.add_argument("--face-blend", action="store_true",
                    help="blend the comparison's best presets by their wins in the face model's coefficients (kept "
                         "only if the blended face wins the comparison against the plain pick in both orders)")
    ap.add_argument("--skin-age", action="store_true",
                    help="the face texture index nearest the drawn face's age, from the measured table "
                         "(<library>/presets/face_texture_age.json, skin_age_calibrate.py); set before the rig")
    ap.add_argument("--fit-base", default=None,
                    help="an accepted build's fit job (bl_combo JSON): its fit settings are kept in this rebuild")
    ap.add_argument("--mh", default=None, help="an existing MetaHuman Character name (/Game/Char/<name>): skip choose and"
                    " mh; --mh-fbx DIR holds its <name>_Body.fbx / _Head.fbx, --mh-blueprint its assembled Blueprint")
    ap.add_argument("--mh-fbx", default=None)
    ap.add_argument("--mh-blueprint", default=None)
    a = ap.parse_args()
    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch for ch in a.id.upper() if ch.isalnum())
    tag = tag[:10] + tag[-2:] if len(tag) > 12 else tag     # the id's end kept: a rebuild beside (_04) gets its own assets
    choices_p, times_p = W / "choices.json", W / "times.json"
    # everything this command reads at run time, checked before any step (a missing folder failed a run at its first
    # step, 11-11): the sheet and LoRA, the preset portraits, the engine project, Blender, the Python environments
    # and the scene tool
    need = {"sheet": a.sheet, "preset portraits (config MH_PRESET_PORTRAITS)": THUMBS, "engine project": PROJECT,
            "blender": BLENDER, "image-to-3D python": HY_PY, "image model python": QWEN_PY, "guides python": GUIDES_PY,
            "scene tool": str(Path(a.scene_tool) / "scene.py"), "library": str(C.LIBRARY)}
    if not a.mh and a.lora and Path(a.lora).suffix:
        need["lora"] = a.lora
    missing = {k: v for k, v in need.items() if not Path(v).exists()}
    if not missing and len(list(Path(THUMBS).glob("*_0.png"))) < 10:
        missing["preset portraits (fewer than 10 in the folder)"] = THUMBS
    if missing:
        raise SystemExit("cast_char: missing inputs, nothing run:\n" + "\n".join(f"  {k}: {v}" for k, v in missing.items()))
    CH = json.loads(choices_p.read_text()) if choices_p.exists() else {}
    TT = json.loads(times_p.read_text()) if times_p.exists() else {}
    redo = set(STEPS[STEPS.index(a.from_step):]) if a.from_step else set()

    def done(step, path):
        return step not in redo and Path(path).exists()

    def save():
        choices_p.write_text(json.dumps(CH, indent=1))
        times_p.write_text(json.dumps(TT, indent=1))

    def timed(step):
        def deco(fn):
            t0 = time.time()
            fn()
            TT[step] = round(time.time() - t0, 1)
            TT["total"] = round(sum(v for k, v in TT.items() if k != "total"), 1)
            save()
            log(f"{step}: {TT[step]} s")
        return deco

    # ---------------------------------------------------------------- choose
    if a.mh:
        (W / "choice.json").exists() or (W / "choice.json").write_text(json.dumps(
            {"base": a.mh, "gender": None, "grooms": {}, "clear_slots": [], "body": {}, "params": {}, "scores": {"given": True}}))
        CH["mh_build"] = {"blueprint": a.mh_blueprint}
    if not done("choose", W / "choice.json"):
        @timed("choose")
        def _():
            p = line(["--line-cmd", "image", "choose", "--id", f"cast_{tag}_choose", "--", "bash", CARD, "05",
                      HY_PY, "-I", str(ASSETS / "characters" / "choose_mh.py"), a.sheet, "--thumbs", THUMBS,
                      "--out", str(W / "choice.json"), "--look", a.look,
                      "--hair-map", str(Path(C.LIBRARY) / "presets" / "hair_colour.json")], W / "choose.log")
            if p.returncode:
                raise RuntimeError(f"choose failed: {p.stdout[-800:]}")
    # ---------------------------------------------------------------- face: the preset by comparison
    # each candidate preset made with the chosen grooms and body and rendered as a neutral studio portrait (the
    # editor's preview actor, no assembly); face_pick.py has the vision model compare them two at a time with the
    # drawn head and keeps the one with the most wins
    ch = json.loads((W / "choice.json").read_text())
    pool = (ch.get("scores") or {}).get("preset_pool") or []
    if not a.mh and len(pool) > 1 and (not done("choose", W / "face_pick.json")):
        @timed("face")
        def _():
            os.environ.setdefault("MH_EXTRA_ARGS", "-NoMetaHumanAccountPortalLoginFallback")
            pdir = W / "ue" / "faces"
            pdir.mkdir(parents=True, exist_ok=True)
            names = {n: f"MHC_CAND_{tag}_{i}" for i, n in enumerate(pool)}
            todo = [n for n in pool if not (pdir / f"{n}.png").exists()]
            for k0 in range(0, len(todo), 6):          # six characters per editor job (memory)
                part = todo[k0:k0 + 6]
                ue_job(W, f"faces_{k0 // 6}", str(ASSETS / "characters" / "char_render.py"), {
                    "map": "/Engine/Maps/Templates/Template_Default", "out": str(pdir), "warm_frames": 120,
                    "settle_frames": 30, "key_cd": 22, "fill_cd": 22, "rim_cd": 8, "exposure_bias": -0.5,
                    "make": [{"name": names[n], "folder": "/Game/Char/Cand", "base": n, "grooms": ch["grooms"],
                              "clear_slots": ch["clear_slots"], "params": ch["params"], "body": ch["body"]}
                             for n in part],
                    "characters": {n: {"path": f"/Game/Char/Cand/{names[n]}"} for n in part},
                    "shots": [{"name": n, "char": n, "frame": "close", "az": 0} for n in part]}, 2400)
            p = line(["--line-cmd", "image", "choose", "--id", f"cast_{tag}_face", "--", "bash", CARD, "05", HY_PY, "-I",
                      str(ASSETS / "characters" / "face_pick.py"), str(W / "choice.json"), "--portraits", str(pdir),
                      "--vl-python", QWEN_PY], W / "face.log")
            if p.returncode:
                raise RuntimeError(f"face pick failed: {p.stdout[-800:]}")
            (W / "face_pick.json").write_text(json.dumps(json.loads((W / "choice.json").read_text())["scores"]
                                                         .get("preset_compare"), indent=1))
        ch = json.loads((W / "choice.json").read_text())
    # ---------------------------------------------------------------- blend: a face between the best presets
    # the comparison's best candidates mixed by their wins in the face model's coefficients (char_make.py
    # "face_blend"), so the face is not limited to a shipped preset; the blended face meets the plain pick in the
    # same composite comparison, both orders, and is kept only if it wins both (face_blend.py)
    blend_p = W / "blend.json"
    if not a.mh and a.face_blend and not done("choose", blend_p):
        @timed("blend")
        def _():
            sys.path.insert(0, str(ASSETS / "characters"))
            import face_blend as FB
            ch_ = json.loads((W / "choice.json").read_text())
            cmp_ = (ch_.get("scores") or {}).get("preset_compare") or {}
            wts = FB.weights_from_wins(cmp_.get("wins") or {}, ch_["base"])
            if not wts:
                blend_p.write_text(json.dumps({"weights": {}, "keep": False,
                                               "why": "fewer than two candidates with a win: nothing to blend"}))
                CH["blend"] = {"keep": False, "weights": {}}
                return
            pdir = W / "ue" / "faces"
            pdir.mkdir(parents=True, exist_ok=True)
            common = {"folder": "/Game/Char/Cand", "base": ch_["base"], "grooms": ch_["grooms"],
                      "clear_slots": ch_["clear_slots"], "params": ch_["params"], "body": ch_["body"]}
            spec_b = dict(common, name=f"MHC_BLEND_{tag}", face_blend=wts)
            plain = dict(common, name=f"MHC_PLAIN_{tag}")
            r_ = ue_job(W, "blend", str(ASSETS / "characters" / "char_render.py"), {
                "map": "/Engine/Maps/Templates/Template_Default", "out": str(pdir), "warm_frames": 120,
                "settle_frames": 30, "key_cd": 22, "fill_cd": 22, "rim_cd": 8, "exposure_bias": -0.5,
                # each preset's coefficients are read once (opening a preset for edit costs about 10 s of CPU)
                "coeff_cache": str(Path(C.LIBRARY) / "presets" / "face_coeffs.json"),
                "make": [spec_b, plain],
                "characters": {"blended": {"path": f"/Game/Char/Cand/MHC_BLEND_{tag}"},
                               "unblended": {"path": f"/Game/Char/Cand/MHC_PLAIN_{tag}"}},
                "shots": [{"name": "blended", "char": "blended", "frame": "close", "az": 0},
                          {"name": "unblended", "char": "unblended", "frame": "close", "az": 0}]}, 2400)
            mk = ((r_.get("make") or {}).get(spec_b["name"]) or {})
            if not mk.get("coeff_set"):
                raise RuntimeError(f"FACE BLEND: the make did not set the blend: {json.dumps(mk)[:400]}")
            res = W / "blend_versus.json"
            p = line(["--line-cmd", "image", "choose", "--id", f"cast_{tag}_blend_check", "--", "bash", CARD, "05",
                      HY_PY, "-I", str(ASSETS / "characters" / "face_pick.py"), str(W / "choice.json"), "--portraits",
                      str(pdir), "--versus", "blended", "unblended", "--result", str(res), "--vl-python", QWEN_PY],
                     W / "blend_check.log")
            if p.returncode or not res.exists():
                raise RuntimeError(f"blend check failed: {p.stdout[-600:]}")
            wins = json.loads(res.read_text())["wins"]
            keep = FB.keep(wins, "blended")
            blend_p.write_text(json.dumps({"weights": wts, "wins": wins, "keep": keep,
                                           "coeff_moved": mk.get("coeff_moved")}, indent=1))
            CH["blend"] = {"weights": wts, "wins": wins, "keep": keep}
    # ---------------------------------------------------------------- conform: the picked face to the drawing
    # the earlier likeness method: the drawn head redrawn as a bald photographic front view (Qwen-Image-Edit), a
    # 3D head from it (image-to-3D), and the MetaHuman's face conformed to that head with 2D tracking of the same
    # picture (char_fit.py, face only: the body stays). Then the same composite comparison once more: the conformed
    # face against the unconformed pick, the drawing in the middle, both orders; the conform is kept only if it
    # wins both
    conf_p = W / "conform.json"
    if not a.mh and a.conform and not done("choose", conf_p):
        @timed("conform")
        def _():
            cd = W / "conform"
            cd.mkdir(exist_ok=True)
            bald = cd / "bald_front.png"
            jobs = [{"id": "bald_front", "seed": 7, "path": str(bald), "size": [768, 768],
                     "images": [str(W / "sheet_head.png")],
                     "prompt": "Draw the person from Picture 1 as a photorealistic studio photograph of their bare "
                               "head and neck only: the same face at the same age, the same proportions and features, the "
                               "same lines and creases of the skin, completely "
                               "bald with a smooth scalp and no hair, no beard or moustache, a neutral relaxed "
                               "expression, mouth closed, eyes open. No clothes, the neck ends at its base. Plain "
                               "light grey seamless background, soft even diffuse light, sharp focus, the whole head "
                               "in frame and centred, seen straight from the front, nothing else in the picture."}]
            jp = cd / "edit_jobs.json"
            jp.write_text(json.dumps(jobs, indent=1))
            p = line(["--line-cmd", "image", "edit", "--id", f"cast_{tag}_bald", "--", "bash", "-c",
                      f"cd {REPO} && bash {CARD} 05 {GUIDES_PY} -I tools/assets/qwen_edit.py {jp}"], W / "conform_edit.log")
            if p.returncode or not bald.exists():
                raise RuntimeError(f"bald front view failed: {p.stdout[-600:]}")
            head = cd / "head.png"
            subprocess.run([HY_PY, "-I", str(ASSETS / "characters" / "head_target.py"), "prep", str(bald), str(head),
                            "--box", "0,0,768,768"], check=True, capture_output=True)
            glb_h = cd / "head.glb"
            p = line(["--line-cmd", "image", "shape", "--id", f"cast_{tag}_head3d", "--", "bash", "-c",
                      f"cd {REPO} && bash {CARD} 05 {HY_PY} -I tools/blockout_shape.py {head} {glb_h} --seed 0 "
                      f"--faces 80000"], W / "conform_head.log")
            if p.returncode or not glb_h.exists():
                raise RuntimeError(f"head mesh failed: {p.stdout[-600:]}")
            tgt = cd / "target.json"
            subprocess.run([HY_PY, "-I", str(ASSETS / "characters" / "head_target.py"), "target", str(glb_h),
                            str(head), str(tgt)], check=True, capture_output=True)
            from PIL import Image as _Im
            bgra = cd / "bald_front.bgra"
            bgra.write_bytes(_Im.open(bald).convert("RGBA").tobytes("raw", "BGRA"))
            fit = {"target": str(tgt), "track": [str(bgra), 768, 768], "mode": "face"}
            ch_ = json.loads((W / "choice.json").read_text())
            pdir = W / "ue" / "faces"
            pdir.mkdir(parents=True, exist_ok=True)
            spec_ = {"name": f"MHC_CONF_{tag}", "folder": "/Game/Char/Cand", "base": ch_["base"], "grooms": ch_["grooms"],
                     "clear_slots": ch_["clear_slots"], "params": ch_["params"], "body": ch_["body"], "fit": fit}
            plain = dict(spec_, name=f"MHC_PICK_{tag}")
            plain.pop("fit")
            ue_job(W, "conform", str(ASSETS / "characters" / "char_render.py"), {
                "map": "/Engine/Maps/Templates/Template_Default", "out": str(pdir), "warm_frames": 120,
                "settle_frames": 30, "key_cd": 22, "fill_cd": 22, "rim_cd": 8, "exposure_bias": -0.5,
                "make": [spec_, plain],
                "characters": {"conformed": {"path": f"/Game/Char/Cand/MHC_CONF_{tag}"},
                               "unconformed": {"path": f"/Game/Char/Cand/MHC_PICK_{tag}"}},
                "shots": [{"name": "conformed", "char": "conformed", "frame": "close", "az": 0},
                          {"name": "unconformed", "char": "unconformed", "frame": "close", "az": 0}]}, 2400)
            res = cd / "versus.json"
            p = line(["--line-cmd", "image", "choose", "--id", f"cast_{tag}_conform_check", "--", "bash", CARD, "05",
                      HY_PY, "-I", str(ASSETS / "characters" / "face_pick.py"), str(W / "choice.json"), "--portraits",
                      str(pdir), "--versus", "conformed", "unconformed", "--result", str(res), "--vl-python", QWEN_PY],
                     W / "conform_check.log")
            if p.returncode or not res.exists():
                raise RuntimeError(f"conform check failed: {p.stdout[-600:]}")
            wins = json.loads(res.read_text())["wins"]
            keep = wins.get("conformed", 0) >= 2
            conf_p.write_text(json.dumps({"fit": fit, "wins": wins, "keep": keep}, indent=1))
            CH["conform"] = {"wins": wins, "keep": keep}
    if a.until == "face":
        save()
        log(f"stopped after the face: {ch['base']} {json.dumps((ch.get('scores') or {}).get('preset_compare'))}")
        return
    CH["metahuman"] = {k: ch[k] for k in ("base", "gender", "grooms", "clear_slots", "body")}
    CH["metahuman_scores"] = ch["scores"]
    save()
    mh_name = a.mh or f"MHC_AUTO_{tag}"
    fbx = Path(a.mh_fbx) if a.mh_fbx else W / "fbx"
    def groom_colour():
        """The chosen hair colour on the built character's own groom materials (char_groom_colour.py): set on the
        Character asset it never reached the assembled character (every build so far kept the default dark brown)."""
        hp = (ch.get("params") or {}).get("Bindings/Hair/") or {}
        if not hp:
            return
        r_ = ue_job(W, "groom_colour", str(ASSETS / "characters" / "char_groom_colour.py"), {
            "folder": f"/Game/MetaHumans/{mh_name}", "melanin": hp.get("Melanin", 0.6),
            "redness": hp.get("Redness", 0.3), "whiteness": hp.get("Whiteness", 0.0)}, 600)
        CH["groom_colour"] = {"values": hp, "materials": len(r_.get("set") or {})}
    # ---------------------------------------------------------------- the face the built character wears
    # a kept conform first (it replaces the face's coefficients), else a kept blend, else the plain pick; and,
    # with --skin-age, the face texture index nearest the drawn face's age (it goes in with the make, before the
    # cloud rig: the editor locks it on a rigged face)
    face_extra = {}
    if not a.mh:
        sys.path.insert(0, str(ASSETS / "characters"))
        import face_blend as FB
        conf_ = json.loads(conf_p.read_text()) if conf_p.exists() else None
        blend_ = json.loads(blend_p.read_text()) if blend_p.exists() else None
        which = FB.choose_face(blend_, conf_)
        CH["face_worn"] = which
        if which == "conform":
            face_extra["fit"] = conf_["fit"]
        elif which == "blend":
            face_extra["face_blend"] = blend_["weights"]
        if a.skin_age:
            import face_skin as FS
            tab_p = Path(C.LIBRARY) / "presets" / "face_texture_age.json"
            if not tab_p.exists():
                raise SystemExit(f"--skin-age: no measured table at {tab_p} (run characters/skin_age_calibrate.py once)")
            sc_ = ch.get("scores") or {}
            age_ = sc_.get("drawing_age") or sc_.get("age_used")
            if age_ is None:
                raise SystemExit("--skin-age: the choice has no age for the drawn face")
            tbl = FS.attributes(C.UE_ENGINE)
            stub = "Stubble" in ((ch.get("grooms") or {}).get("Beard") or "")
            ages_ = FS.preset_ages(json.loads(tab_p.read_text()), ch["base"])
            idx_ = FS.pick(age_, ages_, FS.allowed(tbl, stub), tbl)
            face_extra["skin"] = {"face_texture_index": idx_}
            CH["skin_age"] = {"age": age_, "index": idx_, "index_age": ages_[idx_], "stubble_allowed": stub,
                              "attributes": {k: v["names"][v["values"][idx_]] for k, v in tbl["attrs"].items()}}
        save()
    # ---------------------------------------------------------------- mh
    if not done("mh", fbx / f"{mh_name}_Head.fbx"):
        @timed("mh")
        def _():
            os.environ.setdefault("MH_EXTRA_ARGS", "-NoMetaHumanAccountPortalLoginFallback")
            r = ue_job(W, "mh", str(ASSETS / "characters" / "char_render.py"), {
                "out": str(W / "ue" / "mh"),
                "coeff_cache": str(Path(C.LIBRARY) / "presets" / "face_coeffs.json"),
                "make": [dict({"name": mh_name, "base": ch["base"], "grooms": ch["grooms"], "clear_slots": ch["clear_slots"],
                               "params": ch["params"], "body": ch["body"]}, **face_extra)],
                "cloud": {"path": f"/Game/Char/{mh_name}", "build_path": "/Game/MetaHumans"},
                "characters": {"a": {"path": f"/Game/Char/{mh_name}"}},
                "export_geometry": {"key": "a", "folder": f"/Game/Char/Geo_{tag}", "fbx_dir": str(fbx)}}, 2400)
            CH["mh_build"] = {"blueprint": (r.get("cloud") or {}).get("blueprint"), "make": r.get("make")}
            mk_ = ((r.get("make") or {}).get(mh_name) or {})
            want_i = (face_extra.get("skin") or {}).get("face_texture_index")
            if want_i is not None and mk_.get("face_texture_index") != want_i:
                raise RuntimeError(f"SKIN AGE: the built character has face texture {mk_.get('face_texture_index')}, "
                                   f"not the {want_i} chosen")
            if "face_blend" in face_extra and not mk_.get("coeff_set"):
                raise RuntimeError("FACE BLEND: the built character's make did not set the blend")
            groom_colour()
    # ---------------------------------------------------------------- draw, gate, shape, mesh gate
    draw = W / "draw"
    draw.mkdir(exist_ok=True)
    gate_p = W / "gate.json"
    glb = W / "figure.glb"
    if not (done("gate", gate_p) and done("shape", glb)):
        @timed("draw_gate_shape")
        def _():
            lk = a.look.lower()
            # a full skirt and puffed sleeves need the arms a little higher for air between arm and bodice
            # the starting arm angle by how loose the sleeves are (a fact, text first): the angles that passed the
            # mesh gate so far were 45-54 for close sleeves, 60 for a tweed jacket, 72 for loose ones
            sl_ = ((ch.get("garment") or {}).get("sleeves") or "")
            deg0 = a.pose_deg or {"close": 46, "medium": 60, "loose": 66}.get(
                sl_, 48 if any(w_ in lk for w_ in ("dress", "gown", "skirt")) else 42)
            CH["draw_start_deg"] = {"deg": deg0, "sleeves": sl_ or None}
            gf_ = ch.get("garment") or {}
            look = spread_look(a.look, gf_.get("outer_words") if gf_.get("outer_layer") else None)
            CH["draw_look"] = look
            tried, meshes, chosen = [], [], None
            # each run draws into its own folder with its own seeds (an earlier run's entries still on the line
            # must not overwrite this run's drawings)
            run_n = len([p_ for p_ in draw.iterdir() if p_.is_dir() and p_.name.startswith("run")])
            rdir = draw / f"run{run_n}"
            rdir.mkdir()
            for rnd in range(a.rounds):
                # a round that fails raises the arms 6 degrees and holds the pose map harder; the canvas is wide
                # enough for the hands at any of these angles
                deg = deg0 + 6 * rnd
                pose_map = rdir / f"pose{deg}_r{rnd}.png"
                # past 56 degrees the outstretched hands run off a 896-wide picture (the gate then finds no
                # armpit, 11-12): a wider canvas and a slightly smaller figure
                cw, cf = (1152, 920) if deg >= 56 else (896, 1000)
                subprocess.run([sys.executable, str(HERE / "apose_map.py"), str(pose_map), "--deg", str(deg),
                                "--w", str(cw), "--h", "1344", "--fig", str(cf)], check=True)
                CH.setdefault("pose_deg", []).append(deg)
                seeds = [5 + 1000 * run_n + 6 * (3 * rnd + k) for k in range(3)]
                jobs = [{"id": f"{tag}_s{s_}", "seed": s_, "path": str(rdir / f"spread_s{s_}.png"), "size": [cw, 1344],
                         "prompt": f"{a.trigger}, {look}, {POSE_WORDS}" + (
                             " No outer layer over the shoulders: the arms, the sides of the body and the clothes "
                             "under it all in view." if gf_.get("outer_layer") else ""), "refs": [a.sheet],
                         "loras": [[a.lora, 0.8, [0, 0, 1, 1]]],
                         "control": {"image": str(pose_map), "type": "pose", "scale": 0.8 + 0.1 * min(rnd, 2)}}
                        for s_ in seeds]
                jl = rdir / f"jobs_r{rnd}.jsonl"
                jl.write_text("".join(json.dumps(j) + "\n" for j in jobs))
                p = line(["--line-cmd", "image", "refs", "--id", f"cast_{tag}_draw{run_n}_{rnd}", "--", "bash", "-c",
                          f"cd {REPO} && bash {CARD} 05 {QWEN_PY} -I tools/qwen_image.py --serve < {jl}"],
                         W / f"draw_r{rnd}.log")
                if p.returncode:
                    raise RuntimeError(f"drawing failed: {p.stdout[-800:]}")
                passing = []
                for j in jobs:
                    g = subprocess.run([sys.executable, str(HERE / "pose_gate.py"), j["path"], "--min-armpit",
                                        str(child_armpit(a.look))], capture_output=True, text=True)
                    res = json.loads(g.stdout.strip().splitlines()[-1]) if g.stdout.strip() else {"file": j["path"]}
                    res["pass"] = g.returncode == 0
                    tried.append(res)
                    if res["pass"]:
                        sides = [v for k, v in res.items() if isinstance(v, dict) and "angle_deg" in v]
                        passing.append((min((s_.get("angle_deg") or 0) * (s_.get("armpit") or 0) for s_ in sides),
                                        j["path"]))
                # every drawing that passes goes to 3D, best first, until a mesh passes the mesh gate
                for _sc, path in sorted(passing, reverse=True):
                    out = rdir / (Path(path).stem + ".glb")
                    env = ["env", f"ASSETS_TRELLIS_OMP={OMP}"] if OMP else []
                    p = line(["--line-cmd", "blender", "shape", "--id", f"cast_{tag}_{run_n}_{Path(path).stem}", "--"] + env +
                             [sys.executable, str(HERE / "img3d.py"), path, str(out), "--res", "1024", "--build", "rocm"],
                             W / f"shape_{Path(path).stem}.log")
                    if p.returncode or not out.exists():
                        meshes.append({"file": str(out), "error": p.stdout[-300:]})
                        continue
                    ca_ = child_armpit(a.look)                 # (a child's armpit and elbow sit lower: see there)
                    g = subprocess.run([HY_PY, "-I", str(HERE / "mesh_gate.py"), str(out), "--min-free", str(ca_),
                                        "--elbow", str(round(0.68 - (0.70 - ca_), 3))], capture_output=True, text=True)
                    res = json.loads(g.stdout.strip().splitlines()[-1]) if g.stdout.strip() else {"file": str(out)}
                    res["pass"] = g.returncode == 0
                    meshes.append(res)
                    if res["pass"]:
                        chosen = (path, out)
                        break
                if chosen:
                    break
            CH["gate"] = {"drawings": tried, "meshes": meshes, "chosen": chosen[0] if chosen else None,
                          "rounds": rnd + 1}
            gate_p.write_text(json.dumps(CH["gate"], indent=1))
            if not chosen:
                raise RuntimeError(f"no figure passed both gates in {a.rounds} rounds ({len(tried)} drawings, "
                                   f"{len(meshes)} meshes)")
            import shutil
            shutil.copy(chosen[1], glb)
    drawing = json.loads(gate_p.read_text())["chosen"]


    def drawing_deg(path):
        """The arm angle the drawing was made at: its round's pose map in the same run folder (pose<deg>_r<round>.png
        or, from older runs, pose<deg>.png in round order), the round from the seed (5 + 1000 run + 6 (3 round + k))."""
        import re as _re
        pth = Path(path)
        m = _re.search(r"spread_s(\d+)", pth.name)
        if not m:
            return None
        run_n = int(pth.parent.name[3:]) if pth.parent.name.startswith("run") else 0
        rnd = ((int(m.group(1)) - 5 - 1000 * run_n) // 6) // 3
        maps = list(pth.parent.glob(f"pose*_r{rnd}.png"))
        if maps:
            return float(_re.search(r"pose(\d+)_r", maps[0].name).group(1))
        olds = sorted(pth.parent.glob("pose*.png"), key=lambda q: int(_re.search(r"pose(\d+)", q.name).group(1)))
        return float(_re.search(r"pose(\d+)", olds[rnd].name).group(1)) if rnd < len(olds) else None
    # ---------------------------------------------------------------- combo
    combo = W / "combo"

    def neckline_now():
        """The neckline the build uses: the fit report's (text fact or measured on the figure), else the picture's."""
        try:
            r_ = json.loads((combo / "report.json").read_text()).get("neckline")
        except (OSError, ValueError):
            r_ = None
        return r_ or (ch.get("garment") or {}).get("neckline")

    def head_check():
        rep_ = json.loads((combo / "report.json").read_text())
        hv = (rep_.get("head_volume_verts") or 0) + (rep_.get("head_round_verts") or 0)
        CH["head_volume_verts"] = {"front": rep_.get("head_volume_verts"), "round": rep_.get("head_round_verts"),
                                   "hair_stripped": rep_.get("hair_stripped_verts")}
        if hv > HEAD_VOLUME_MAX:
            save()
            raise RuntimeError(f"GARMENT IN THE HEAD: {hv} garment vertices in front of the face or inside the head "
                               f"or round the head above the chin (at most {HEAD_VOLUME_MAX}); the figure's own head was probably left "
                               f"on the garment (its neck line); the build stops here")
    # what each part of the figure is (hair, face, bare skin, a garment): the trained parser's labels of the drawing
    # and of the mesh's own views (parse_figure.py), for the fit to cut and keep by instead of by colour
    labels_p = W / "figure_labels.npz"
    if not a.whole and not done("combo", labels_p):
        @timed("parse")
        def _():
            p = line(["--line-cmd", "image", "choose", "--id", f"cast_{tag}_parse", "--", "bash", CARD, "05", HY_PY,
                      str(HERE / "parse_figure.py"), str(glb), str(labels_p), "--sheet", str(W / "figure_labels.jpg"),
                      "--drawing", str(drawing)], W / "parse.log")
            if p.returncode or not labels_p.exists():
                raise RuntimeError(f"figure labels failed: {p.stdout[-600:]}")
    if not done("combo", combo / "outfit.fbx"):
        @timed("combo")
        def _():
            m = legs_apart(drawing)
            lk = a.look.lower()
            gf = (ch.get("garment") or {})
            if gf.get("long") is not None:
                # the text-first fact (a garment to the ankles) decides; the drawing's legs measure is the fallback
                # (two shoes under a nightgown's hem read as trousers, 11-11)
                m["long_skirt_drawing"] = m["long_skirt"]
                m["long_skirt"] = bool(gf["long"])
                m["long_skirt_source"] = (gf.get("source") or {}).get("long_garment") or "fact"
            job = {"body_fbx": str(fbx / f"{mh_name}_Body.fbx"), "head_fbx": str(fbx / f"{mh_name}_Head.fbx"),
                   "figure_glb": str(glb), "out": str(combo), "hair_cap": False, "neck_cut_m": 0.03,
                   **({"mode": "statue"} if a.whole else {"labels": str(labels_p)}),
                   "collar_lift": False, "liner": False,
                   # cloth pushed out of the head mesh only near the neck (a push over the shoulders flattened puffed
                   # sleeves); the head mesh's skin under the cloth is hidden instead: the fit bakes an opacity mask in
                   # the head's UVs (bl_skinmask.py) that the character's own masked skin material wears
                   "head_push_r_m": 0.08, "skin_mask": {}}
            if m["long_skirt"]:
                job.update(sleeve_radius_m=0.20, skirt_pelvis=1.0, turn_legs=False, shoe_top_m=0.0, skirt_min_z_m=-1.0,
                           long_skirt=True)
            else:
                coat_skirt = (m["thigh_run_w_frac"] or 0) > 0.17
                job.update(sleeve_radius_m=0.15, skirt_pelvis=1.0 if coat_skirt else 0.6)
            # the neckline: the text's fact only when the text states it; otherwise the fit measures it on the
            # generated figure (bl_combo neckline_figure); the picture's answer is the last fallback (below)
            gf0 = ch.get("garment") or {}
            job["neckline"] = gf0.get("neckline") if (gf0.get("source") or {}).get("neckline") == "text" else None
            # the drawing's arm angle: the fit finds the hands along it and poses the body's arms to them (binding in
            # the drawing's own pose, whatever the angle)
            job["draw_deg"] = drawing_deg(drawing)
            # the drawn figure's hair hangs below its neck (long, or worn down): the fit removes hair left standing
            # off the shoulders after the cut
            va_ = (ch.get("scores") or {}).get("vlm_answers") or {}
            job["hair_hangs"] = (va_.get("length", "").startswith("long") or va_.get("worn", "").startswith("down")) \
                and not va_.get("worn", "").startswith(("up", "back"))
            if a.fit_base:
                # a rebuild of an accepted character: that build's fit settings are kept (the rules above measured a
                # coat skirt and no collar lift on a figure whose accepted fit had both); the skin rules stay
                base_j = json.loads(Path(a.fit_base).read_text())
                keep = {k: job[k] for k in ("body_fbx", "head_fbx", "figure_glb", "out", "head_push_r_m", "skin_mask",
                                            "liner", "neckline", "hair_hangs", "draw_deg")}
                job.clear()
                job.update(base_j)
                job.update(keep)
                CH["fit_base"] = {"from": Path(a.fit_base).name,
                                  "kept": sorted(k for k in base_j if k not in ("body_fbx", "head_fbx", "figure_glb", "out"))}
            # collars and cravats are the generated ones (no constructed pieces)
            CH["combo"] = {"measured": m, "job": {k: v for k, v in job.items() if not k.endswith(("fbx", "glb"))}}
            jp = W / "combo_job.json"
            jp.write_text(json.dumps(job, indent=1))
            p = line(["--line-cmd", "igpu", "combo", "--id", f"cast_{tag}_combo", "--", "flock", "/tmp/gpu-igpu.lock",
                      BLENDER, "-b", "--factory-startup", "-t", "0", "--python", str(HERE / "bl_combo.py"), "--",
                      str(jp)], W / "combo.log")
            if p.returncode or not (combo / "outfit.fbx").exists():
                raise RuntimeError(f"combo failed: {p.stdout[-800:]}")
            if not a.whole:                 # (the whole figure keeps its own head: nothing to check for one left on)
                head_check()
    # ---------------------------------------------------------------- the body fits the clothes at the neck
    # the fit measures the bare neck (chin to the generated collar's front top) and both girths (the collar's inner
    # one, the MetaHuman's neck); where the collar sits low or tight, the MetaHuman's neck is changed to meet it
    # (Neck Length shorter by the bare neck over 1.5 cm, never under 4.5 cm; Neck girth to the collar's inner girth
    # less 1 cm), the MetaHuman rebuilt and the fit run once more. Moving the cloth instead bunched collars.
    neck_p = W / "neck_fit.json"
    if (combo / "report.json").exists() and not neck_p.exists():
        rep = json.loads((combo / "report.json").read_text())
        bare, gi, gn = rep.get("bare_neck_measured_cm"), rep.get("collar_inner_girth_cm"), rep.get("neck_girth_cm")
        body0 = ch.get("body") or {}
        newb, addb = {}, {}
        # the neck's length is not set: Neck Length 7.2 -> 4.5 (inside its range) left the built skeleton as it was
        # (neck_01 to head 0.1178 -> 0.1177 m, 11-11 clean test); a bare neck is met by the collar lift instead
        if gi and gn and gn > gi - 1.0:
            # relative: the fit's measured girth (round the head mesh's neck) and the MetaHuman's own "Neck"
            # measure are on different scales; setting the collar's girth outright made a neck thicker (11-11)
            addb["Neck"] = round((gi - 1.0) - gn, 1)
        nf = {"measured": {"bare_neck_cm": bare, "collar_inner_girth_cm": gi, "neck_girth_cm": gn}, "set": newb,
              "add": addb}
        if newb or addb:
            # a given MetaHuman (--mh) is another character's too: the neck changes go on a copy of it, never on it
            src_mh = mh_name
            if a.mh:
                mh_name = f"{a.mh}_{tag}"
                fbx = W / "fbx"
                CH["mh_build"] = {"blueprint": f"/Game/MetaHumans/{mh_name}/BP_{mh_name}", "copied_from": a.mh}
                nf["copy"] = {"from": a.mh, "to": mh_name}

            @timed("neck_fit")
            def _():
                job_c = {"path": f"/Game/Char/{mh_name}", "body": newb, "add": addb,
                         "min": {"Neck Length": 4.5, "Neck": 30.0}}
                if src_mh != mh_name:
                    job_c["copy_from"] = f"/Game/Char/{src_mh}"
                r = ue_job(W, "neck_constraints", str(ASSETS / "characters" / "char_constraints.py"), job_c, 600)
                nf["engine"] = r
                ue_job(W, "mh_rebuild", str(ASSETS / "characters" / "char_render.py"), {
                    "out": str(W / "ue" / "mh_rebuild"),
                    "cloud": {"path": f"/Game/Char/{mh_name}", "build_path": "/Game/MetaHumans"},
                    "characters": {"a": {"path": f"/Game/Char/{mh_name}"}},
                    "export_geometry": {"key": "a", "folder": f"/Game/Char/Geo_{tag}", "fbx_dir": str(fbx)}}, 2400)
                jp = W / "combo_job.json"
                cj_ = json.loads(jp.read_text())
                cj_["body_fbx"], cj_["head_fbx"] = str(fbx / f"{mh_name}_Body.fbx"), str(fbx / f"{mh_name}_Head.fbx")
                jp.write_text(json.dumps(cj_, indent=1))
                p = line(["--line-cmd", "igpu", "combo", "--id", f"cast_{tag}_combo2", "--", "flock", "/tmp/gpu-igpu.lock",
                          BLENDER, "-b", "--factory-startup", "-t", "0", "--python", str(HERE / "bl_combo.py"), "--",
                          str(jp)], W / "combo2.log")
                if p.returncode:
                    raise RuntimeError("refit after the neck change failed")
                rep2 = json.loads((combo / "report.json").read_text())
                nf["after"] = {k: rep2.get(k) for k in ("bare_neck_measured_cm", "collar_inner_girth_cm", "neck_girth_cm")}
                # the change must show in the rebuilt body: a constraint change with no measured effect stops the
                # build (11-11: Neck Length 7.2 -> 4.5 left the bare neck at 7.8 -> 7.9 cm and the skeleton as it was)
                eng = r.get("after") or {}
                bef = r.get("before") or {}
                d_len = abs((eng.get("Neck Length") or [0, 0])[1] - (bef.get("Neck Length") or [0, 0])[1])
                d_bare = abs((nf["after"].get("bare_neck_measured_cm") or 0) - (bare or 0))
                d_girth = abs((nf["after"].get("neck_girth_cm") or 0) - (gn or 0))
                d_neck = abs((eng.get("Neck") or [0, 0])[1] - (bef.get("Neck") or [0, 0])[1])
                nf["effect"] = {"neck_length_set_cm": round(d_len, 2), "bare_changed_cm": round(d_bare, 2),
                                "neck_set_cm": round(d_neck, 2), "girth_changed_cm": round(d_girth, 2)}
                if d_neck >= 2.0 and d_girth < 0.5:
                    neck_p.write_text(json.dumps(nf, indent=1))
                    raise RuntimeError(f"NECK FIT HAD NO EFFECT: the MetaHuman's constraints changed "
                                       f"({json.dumps(nf['engine'])}) but the rebuilt body did not "
                                       f"({json.dumps(nf['effect'])}); the build stops here")
        neck_p.write_text(json.dumps(nf, indent=1))
        CH["neck_fit"] = nf
        save()
    # still a bare neck over 2 cm once the neck is as short as allowed: the fit's collar lift (the collar region raised
    # by one smooth displacement to meet the jaw), measured, only then
    if (combo / "report.json").exists():
        rep = json.loads((combo / "report.json").read_text())
        jp = W / "combo_job.json"
        job = json.loads(jp.read_text())
        bare = rep.get("bare_neck_measured_cm")
        nl = rep.get("neckline") or (ch.get("garment") or {}).get("neckline")
        CH["neckline"] = {"class": nl, "source": rep.get("neckline_source") or "picture"}
        # (with the figure's labels the collar stays where the drawing has it: the lift was there to cover hair and
        # beard read as a collar, and raised a low neckline to the jaw, 11-14)
        if bare is not None and bare > 2.0 and not job.get("collar_lift") and nl in (None, "chin") \
                and not job.get("labels"):
            # only a collar that comes up the neck is lifted to the chin (a round or open neckline is bare by design:
            # a lifted nightgown neckline became a boxy collar at the jaw, 11-11); the fact is asked text first
            @timed("collar_lift")
            def _():
                job["collar_lift"] = True
                jp.write_text(json.dumps(job, indent=1))
                p = line(["--line-cmd", "igpu", "combo", "--id", f"cast_{tag}_combo_lift", "--", "flock",
                          "/tmp/gpu-igpu.lock", BLENDER, "-b", "--factory-startup", "-t", "0", "--python",
                          str(HERE / "bl_combo.py"), "--", str(jp)], W / "combo_lift.log")
                if p.returncode:
                    raise RuntimeError("refit with the collar lift failed")
                head_check()
                rep2 = json.loads((combo / "report.json").read_text())
                CH["collar_lift"] = {"bare_before_cm": bare, "bare_after_cm": rep2.get("bare_neck_cm"),
                                     "lift": rep2.get("collar_height")}
            save()
    # ---------------------------------------------------------------- import + library entry
    lib = Path(C.LIBRARY) / "character" / a.id
    if not done("import", lib / "character.json"):
        @timed("import")
        def _():
            dest = f"/Game/Char/Outfit{tag}"
            ue_job(W, "import", str(ASSETS / "characters" / "char_render.py"), {
                "out": str(W / "ue" / "import"),
                "import_outfit": {"outfit_fbx": str(combo / "outfit.fbx"), "textures": str(combo / "textures"),
                                  "dest": dest, "cloth_slots": False}}, 900)
            bp = (CH.get("mh_build") or {}).get("blueprint") or f"/Game/MetaHumans/{mh_name}/BP_{mh_name}"
            # the character's own masked copy of its face skin, wearing the fit's mask (the shared material untouched)
            sk = ue_job(W, "skin_mask", str(ASSETS / "characters" / "char_skinmask.py"), {
                "face_folder": bp.rsplit("/", 1)[0], "mask_png": str(combo / "textures" / "T_head_mask.png"),
                "dest": f"{dest}/Skin", "blueprint": bp, "name": tag}, 1200)
            bp = sk["blueprint"]          # the character's own Blueprint copy, its face wearing the masked skin
            CH["skin_mask"] = {"fit": json.loads((combo / "report.json").read_text()).get("skin_mask"), "engine": sk}
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "character.json").write_text(json.dumps({
                "class": "character", "kind": "metahuman", "project": "CharLab", "blueprint": bp,
                "rebuild_from": f"/Game/Char/{mh_name}", "bp_deps": ["/MetaHumanCharacter/Animation/ABP_MH_LiveLink"],
                "character_asset": f"/Game/Char/{mh_name}", "hide_body": True, "id": a.id,
                **({"hide_face": True, "whole_figure": True} if a.whole else {}),
                "garments": [f"{dest}/Outfit/outfit", f"@face_material={sk['instance']}"], "body_swap": None,
                "garment_material": None, "skin_mask": {"material": sk["instance"], "mask": sk["mask"]},
                "missing_layers": ([" ".join(ch["garment"]["outer_words"])]
                                   if (ch.get("garment") or {}).get("outer_layer") and ch["garment"].get("outer_words")
                                   else []),
                "facts": entry_facts(ch, [" ".join(ch["garment"]["outer_words"])]
                                     if (ch.get("garment") or {}).get("outer_layer") and ch["garment"].get("outer_words")
                                     else []),
                # (its own drawn hair kept on the head: the MetaHuman's hair groom is not shown)
                "hide_hair": bool(((json.loads((combo / "report.json").read_text()).get("hair_kept") or {}).get("verts")))
                and not a.whole,
                "note": f"Made by cast_char.py from the sheet: {a.look}", "made_by": "cast_char.py",
                "choices": {"preset": ch["base"], "grooms": ch["grooms"], "drawing": Path(drawing).name}},
                indent=1))
    # ---------------------------------------------------------------- arms: abduction by measurement
    scene_py = Path(a.scene_tool) / "scene.py"
    env = dict(os.environ, SCENES_OUT=str(W / "scenes"), SCENES_CHAR_TOOLS=str(ASSETS / "characters"),
               ASSETS_LINE_CLIENT=str(ASSETS / "run_asset.py"))
    cj = json.loads((lib / "character.json").read_text())
    if "arms" in redo or "relaxed_arm_steps" not in cj:
        @timed("arms")
        def _():
            chars, cams = {}, []
            for k, ab in (("a4", 4), ("a8", 8), ("a12", 12)):
                x = {"a4": -0.9, "a8": 0.2, "a12": 1.3}[k]
                # the check scene's own stance and turn (weight on the right, chest 4 deg, a 20 deg turn): measured
                # square to the camera, a borderline right arm passed here and failed the check
                chars[k] = {"id": a.id, "position": [x, 0.4], "facing": [0.3464, -0.9381], "body_pose": {
                    "pose": "stand_relaxed", "params": {"weight_on": "r", "chest_deg": 4, "head_deg": 0, "head_tilt": -2,
                                                        "hand_at_lapel": False, "steps": arm_steps(ab)}}}

            # one camera that holds all three copies (framed on the copies' marks, inside the room)
            cams = [{"name": "wide_all", "kind": "wide", "subject": "a8", "position": [0.2, -2.2, 1.0],
                     "target": [0.2, 0.4, 0.95], "focal_mm": 22}]
            spec = {"id": f"arms_{tag}", "set": a.set, "guides": True, "characters": chars, "cameras": cams}
            sp = W / "arms_scene.json"
            sp.write_text(json.dumps(spec, indent=1))
            p = subprocess.run([sys.executable, str(scene_py), str(sp)], cwd=str(Path(a.scene_tool).parents[2]),
                               env=env, capture_output=True, text=True)
            if p.returncode:
                raise RuntimeError(f"arms scene failed: {p.stdout[-1500:]}")
            sd = W / "scenes" / spec["id"]
            p = line(["--line-cmd", "image", "guides", "--id", f"cast_{tag}_guides", "--", "bash", "-c",
                      f"cd {Path(a.scene_tool).parents[2]} && bash {CARD} 05 {GUIDES_PY} -I "
                      f"{Path(a.scene_tool).parent / 'redraw' / 'guides.py'} {sd}"], W / "guides.log")
            g = subprocess.run([GUIDES_PY, str(ASSETS / "characters" / "arm_gap.py"), str(sd), "--scene-tool",
                                str(a.scene_tool)], capture_output=True, text=True)
            res = json.loads(g.stdout)
            pick = 12
            for k in ("a4", "a8", "a12"):
                for s_ in ("l", "r"):
                    if not ((res.get("wide_all", {}).get(k) or {}).get(s_) or {}).get("rows"):
                        raise RuntimeError(f"arms measure empty for copy {k} side {s_}: the copy is not in the frame")
            for k, ab in (("a4", 4), ("a8", 8), ("a12", 12)):
                sh = res.get("wide_all", {}).get(k, {})
                if all((sh.get(s_) or {}).get("separated", 0) >= 8 for s_ in ("l", "r")):
                    pick = ab
                    break
            CH["arms"] = {"measured": res, "abduct_deg": pick, "picked_deg": pick}
            cj2 = json.loads((lib / "character.json").read_text())
            cj2["relaxed_arm_steps"] = arm_steps(pick)
            (lib / "character.json").write_text(json.dumps(cj2, indent=1))
    # ---------------------------------------------------------------- seated stances, one per seat in the set
    seats_p = W / "seats.json"
    if not done("seats", seats_p):
        @timed("seats")
        def _():
            sys.path.insert(0, str(a.scene_tool))
            import seating  # noqa: E402  (the scene tool's: seat blocks, placement)
            lib_root = Path(C.LIBRARY)
            S = seating.seats(lib_root, lib_root / "set" / a.set)
            cj0 = json.loads((lib / "character.json").read_text())
            long_skirt = bool((CH.get("combo") or {}).get("measured", {}).get("long_skirt"))
            sys.path.insert(0, str(ASSETS / "characters"))
            import poses_extra as PX  # noqa: E402
            made = {}
            for name, seat in S.items():
                seat = dict(seat, name=name)
                if long_skirt:      # a dress: upright, hands in the lap
                    c = {"id": a.id, "position": [0, 0], "facing": [0, -1], "body_pose": {"pose": "seated", "params": {
                        "lean_deg": 4, "chin_deg": 2, "knee_spread_m": 0.03, "shin_fwd": 0.12}}}
                else:               # back in the chair, forearms along its arms
                    c = {"id": a.id, "position": [0, 0], "facing": [0, -1], "body_pose": {"pose": "sit_armchair_relaxed",
                         "params": {"thigh_clearance_m": 0.1, "lean_deg": 10, "hand_fwd_m": 0.34, "head_deg": -6}}}
                c = seating.place(c, seat, seating.fit(cj0))
                c.pop("_seat", None)
                pr = c["body_pose"]["params"]
                if long_skirt:
                    pr.pop("arm_z_m", None)
                    pr.pop("arm_x_m", None)
                    pr["steps"] = PX.hands_in_lap(pr["seat_z_m"])
                lid = f"{a.id}_seated_{''.join(ch for ch in name.lower() if ch.isalnum() or ch == '_')}"
                if a.whole:
                    # a whole figure sits as it is: the pose on its own mesh. The settle is for clothes over a body
                    # (cloth pushed off the seat and off the body under it): on a whole figure it pushed the face
                    # off the hidden head and left a band across it (11-14)
                    b = seat["box"]
                    ent = dict(cj0, id=lid, stance={"base": a.id, "pose": "seated", "seat_asset": seat["asset"],
                                                    "seat_z_m": b["seat_z_m"], "params": c["body_pose"]},
                               seat_asset=seat["asset"], seat_z_m=b["seat_z_m"],
                               note=f"Seated only, in the set's {name} ({seat['asset']}, seat {b['seat_z_m']} m): "
                                    f"{a.id}, the whole figure posed.")
                    ent.pop("relaxed_arm_steps", None)
                    (Path(C.LIBRARY) / "character" / lid).mkdir(parents=True, exist_ok=True)
                    (Path(C.LIBRARY) / "character" / lid / "character.json").write_text(json.dumps(ent, indent=1))
                    made[name] = lid
                    continue
                sid = f"seatj_{tag}_{name}"
                spec = {"id": sid, "set": a.set, "guides": False, "characters": {"a": c},
                        "cameras": [{"name": "wide", "kind": "full", "subject": "a", "focal_mm": 28, "az": -20}]}
                sp = W / f"{sid}.json"
                sp.write_text(json.dumps(spec, indent=1))
                p = subprocess.run([sys.executable, str(scene_py), str(sp)], cwd=str(Path(a.scene_tool).parents[2]),
                                   env=env, capture_output=True, text=True)
                if p.returncode:
                    raise RuntimeError(f"seat joints scene {name} failed: {p.stdout[-800:]}")
                r = json.loads((W / "scenes" / sid / "result.json").read_text())
                b = seat["box"]
                hip_y = b["back_y_m"] - seating.fit(cj0)["hip_back_m"]
                sj = {"combo": str(combo), "joints": r["subjects"]["a"]["measured"]["joints"], "position": c["position"],
                      "facing": c["facing"], "mode": "drape", "iters": 0, "shoulder_ease_m": 0.0,
                      "seat": {"z": b["seat_z_m"], "front": round(hip_y - b["front_y_m"], 3), "back": seating.fit(cj0)["hip_back_m"],
                               "half_width": round((b["inner_x_m"][1] - b["inner_x_m"][0]) / 2, 3)},
                      "out": str(W / f"seated_{name}")}
                jp = W / f"settle_{name}.json"
                jp.write_text(json.dumps(sj, indent=1))
                p = line(["--line-cmd", "igpu", "settle", "--id", f"cast_{tag}_settle_{name}", "--", "flock", "/tmp/gpu-igpu.lock",
                          BLENDER, "-b", "--factory-startup", "-t", "0", "--python", str(HERE / "bl_settle.py"), "--", str(jp)],
                         W / f"settle_{name}.log")
                if p.returncode:
                    raise RuntimeError(f"settle {name} failed")
                srep = json.loads((W / f"seated_{name}" / "settle.json").read_text())
                if srep.get("cloth_under_floor"):      # the whole garment, not only its loose set (11-12: a tube of
                    raise RuntimeError(                # skirt through the seat and floor passed on the loose set's hem)
                        f"settle {name}: {srep['cloth_under_floor']} cloth vertices under the floor "
                        f"(lowest {srep.get('cloth_min_z_m')} m)")
                dest = f"/Game/Char/Outfit{tag}S{''.join(ch for ch in name.upper() if ch.isalnum())}"   # whole name: armchair and armchair_2 collided
                ue_job(W, f"import_seat_{name}", str(ASSETS / "characters" / "char_render.py"), {
                    "out": str(W / "ue" / f"import_seat_{name}"),
                    "import_outfit": {"outfit_fbx": str(W / f"seated_{name}" / "outfit.fbx"),
                                      "textures": str(W / f"seated_{name}" / "textures"), "dest": dest, "cloth_slots": False}}, 900)
                st = {"base": a.id, "pose": "seated", "seat_asset": seat["asset"], "seat_z_m": b["seat_z_m"],
                      "params": c["body_pose"]}
                ent = dict(cj0, id=lid, garments=[f"{dest}/Outfit/outfit"] + [g for g in cj0["garments"] if g.startswith("@")],
                           stance=st, seat_asset=seat["asset"],
                           seat_z_m=b["seat_z_m"], note=f"Seated only, in the set's {name} ({seat['asset']}, seat "
                           f"{b['seat_z_m']} m): {a.id} with its loose cloth laid over the thighs and seat (bl_settle, no relaxation).")
                ent.pop("relaxed_arm_steps", None)
                (Path(C.LIBRARY) / "character" / lid).mkdir(parents=True, exist_ok=True)
                (Path(C.LIBRARY) / "character" / lid / "character.json").write_text(json.dumps(ent, indent=1))
                made[name] = lid
            CH["seats"] = made
            seats_p.write_text(json.dumps(made, indent=1))
    if a.until == "seats":
        log("stopped after the seated stances (--until seats)")
        return
    # ---------------------------------------------------------------- lying stances, one per bed in the library
    # a bed's lie box (its library record's supports, characters/liebox.py) and a lying pose built from the scene
    # tool's pose steps on the standing pose (the whole body turned onto its back about the hips, head to the head
    # end, the back on the surface; "propped": the chest raised on the pillow, for someone talking in bed); the
    # bed placed in a lab spot of the set, the engine poses and measures the joints, bl_settle.py lays the cloth
    # with no relaxation and nothing through the mattress, shows the body's skin the cloth leaves open and builds the
    # bed's cover over the body (set dressing for this stance and bed: head and shoulders out); each is a stance
    # entry <id>_lying_<bed>[_propped] recording the support it was settled on (format agreed with the scene side)
    lie_p = W / "lie.json"
    if a.lie and not done("seats", lie_p):
        @timed("lie")
        def _():
            lib_root = Path(C.LIBRARY)
            beds = []
            for d_ in sorted((lib_root / "prop").iterdir()):
                aj_ = d_ / "asset.json"
                if not aj_.exists():
                    continue
                rec_ = json.loads(aj_.read_text())
                sup_ = next((x for x in rec_.get("supports", []) if x.get("posture") == "lying"), None)
                if sup_ and (not a.lie_beds or d_.name in a.lie_beds):
                    beds.append((d_.name, sup_))
            cj0 = json.loads((lib / "character.json").read_text())
            made = {}
            for bed, sup in beds:
                for variant in a.lie_variants:
                    made[f"{bed}_{variant}"] = lie_stance(W, a, tag, lib, cj0, bed, sup, variant, combo, scene_py, env)
            CH["lie"] = made
            lie_p.write_text(json.dumps(made, indent=1))
    # ---------------------------------------------------------------- check stills
    if not done("check", W / "scenes" / f"check_{tag}" / "close.png"):
        @timed("check")
        def _():
            # the check scene itself measures the arms (the scene tool's guides, arm_gap.py at its wide camera): a
            # forearm not separated in >= 8 of 10 rows on a side raises the abduction 2 deg (the smallest that passes:
            # more reads as arms held out) and the scene is run again
            # from the arms step's own pick (a re-run of the check starts there again, not where the last one ended)
            hist = [h.get("abduct_deg") for h in CH.get("check_arms", []) if h.get("abduct_deg")]
            ab0 = (CH.get("arms") or {}).get("picked_deg") or (min(hist) if hist else None)
            if ab0:
                CH.setdefault("arms", {})["abduct_deg"] = ab0
                cj0_ = json.loads((lib / "character.json").read_text())
                cj0_["relaxed_arm_steps"] = arm_steps(ab0)
                (lib / "character.json").write_text(json.dumps(cj0_, indent=1))
            CH["check_arms"] = []
            for _round in range(8):
                cj3 = json.loads((lib / "character.json").read_text())
                spec = {"id": f"check_{tag}", "set": a.set, "guides": True, "characters": {"a": {
                    "id": a.id, "position": [0.3, 0.8], "facing": [0.3464, -0.9381], "body_pose": {
                        "pose": "stand_relaxed", "params": {"weight_on": "r", "chest_deg": 4, "head_deg": 0,
                                                            "head_tilt": -2, "hand_at_lapel": False,
                                                            "steps": cj3.get("relaxed_arm_steps", [])}}}},
                    "cameras": [{"name": "wide", "kind": "full", "subject": "a", "focal_mm": 28},
                                {"name": "medium", "kind": "medium", "subject": "a", "focal_mm": 50},
                                {"name": "close", "kind": "close", "subject": "a", "focal_mm": 85, "az": 10, "el": 2}]}
                sp = W / "check_scene.json"
                sp.write_text(json.dumps(spec, indent=1))
                p = subprocess.run([sys.executable, str(scene_py), str(sp)], cwd=str(Path(a.scene_tool).parents[2]),
                                   env=env, capture_output=True, text=True)
                if p.returncode:
                    raise RuntimeError(f"check scene failed: {p.stdout[-1500:]}")
                sd = W / "scenes" / spec["id"]
                p = line(["--line-cmd", "image", "guides", "--id", f"cast_{tag}_check_guides_{_round}", "--", "bash", "-c",
                          f"cd {Path(a.scene_tool).parents[2]} && bash {CARD} 05 {GUIDES_PY} -I "
                          f"{Path(a.scene_tool).parent / 'redraw' / 'guides.py'} {sd}"], W / "check_guides.log")
                g = subprocess.run([GUIDES_PY, str(ASSETS / "characters" / "arm_gap.py"), str(sd), "--scene-tool",
                                    str(a.scene_tool)], capture_output=True, text=True)
                res = json.loads(g.stdout) if g.stdout.strip() else {}
                sh = (res.get("wide") or {}).get("a") or {}
                sep = {s_: (sh.get(s_) or {}).get("separated", 0) for s_ in ("l", "r")}
                low = {s_: (sh.get(s_) or {}).get("lower_separated", 0) for s_ in ("l", "r")}
                loose = (ch.get("garment") or {}).get("sleeves") == "loose"
                CH.setdefault("check_arms", []).append({"abduct_deg": (CH.get("arms") or {}).get("abduct_deg"),
                                                        "wide": sep, "lower": low})
                # loose sleeves are joined to the body down to about the elbow by their cut: for them the lower
                # forearm and the hand must be clear (4 of the lower 5 rows); every other class 8 of 10 rows
                passed = all(v >= 4 for v in low.values()) if loose else all(v >= 8 for v in sep.values())
                if passed:
                    cj3["arm_rule"] = ({"rule": "loose_sleeve_lower_forearm", "expect": "a draped sleeve joined to the "
                                        "body down to about the elbow; the lower forearm and the hand clear"}
                                       if loose else {"rule": "forearm_8_of_10", "expect": "the forearm clear of the body"})
                    (lib / "character.json").write_text(json.dumps(cj3, indent=1))
                    break
                ab = int((CH.get("arms") or {}).get("abduct_deg", 8)) + 2
                # loose sleeves hang against the body until the arm is further out: their limit is higher (a fact)
                ab_max = 24 if ((ch.get("garment") or {}).get("sleeves") == "loose") else 16
                if ab > ab_max:
                    save()
                    raise RuntimeError(f"ARMS: a forearm still touches the body at {ab_max} degrees ({sep}); the fit's arms are "
                                       f"wrong (the check sheet shows how); the build stops here")
                CH.setdefault("arms", {})["abduct_deg"] = ab
                cj3["relaxed_arm_steps"] = arm_steps(ab)
                (lib / "character.json").write_text(json.dumps(cj3, indent=1))
                save()
    # ---------------------------------------------------------------- the outer layer, made present
    # an outer layer left out of the spread drawing (the fact) is made as its own piece: drawn alone on a plain neutral
    # figure standing at the character's relaxed arm angle (it hangs from the shoulders; a spread pose ruins it), made
    # 3D, fitted by bl_layer.py (placed by the neutral figure, the figure cut away, bound to the upper body, the hem on
    # the pelvis, no relaxation), attached to the entries and checked (its top at the neck, above the floor, resting on
    # at least one shoulder, most of it within reach of the body); when more than one drawing passes, the vision model
    # picks the one worn as on the sheet; a 3D texture that came out grey is reported (and rescaled to the drawing's
    # colour)
    layer_p = W / "layer.json"
    gfl = ch.get("garment") or {}
    if gfl.get("outer_layer") and gfl.get("outer_words") and not done("check", layer_p):
        @timed("layer")
        def _():
            ld = W / "layer"
            ld.mkdir(exist_ok=True)
            cjl = json.loads((lib / "character.json").read_text())
            steps_ = cjl.get("relaxed_arm_steps") or []
            ab = abs(float(steps_[0][3])) if steps_ else 8.0
            pm = ld / f"pose{int(ab)}.png"
            subprocess.run([sys.executable, str(HERE / "apose_map.py"), str(pm), "--deg", str(ab), "--w", "896",
                            "--h", "1344", "--fig", "1000"], check=True)
            phrase = " ".join(gfl["outer_words"])
            pose_words = POSE_WORDS.replace(
                "both arms held straight out away from the body, angled down about 45 degrees, a clear open gap between "
                "each arm and the clothes, hands open with the fingers together and the palms facing down",
                "both arms hanging down at the sides a little away from the body, hands relaxed").replace(
                "A realistic full-colour photograph of a costumed actor",
                "A realistic full-colour photograph of a garment shown on a tailor's display figure")
            jobs = [{"id": f"{tag}_layer_s{s_}", "seed": s_, "path": str(ld / f"layer_s{s_}.png"), "size": [896, 1344],
                     "prompt": f"a {phrase} worn over a plain smooth light grey tailor's mannequin with no face, no hair "
                               f"and no other clothes, the {gfl['outer_words'][-1]} hanging from the shoulders, "
                               f"{pose_words}",
                     "refs": [a.sheet], "control": {"image": str(pm), "type": "pose", "scale": 0.8}}
                    for s_ in (5, 11)]
            jl = ld / "jobs.jsonl"
            jl.write_text("".join(json.dumps(j) + "\n" for j in jobs))
            p = line(["--line-cmd", "image", "refs", "--id", f"cast_{tag}_layer_draw", "--", "bash", "-c",
                      f"cd {REPO} && bash {CARD} 05 {QWEN_PY} -I tools/qwen_image.py --serve < {jl}"], W / "layer_draw.log")
            rep_l, outd, passed = None, None, []
            for j in jobs:
                png = Path(j["path"])
                if not png.exists():
                    continue
                glb_l = png.with_suffix(".glb")
                env_ = ["env", f"ASSETS_TRELLIS_OMP={OMP}"] if OMP else []
                p = line(["--line-cmd", "blender", "shape", "--id", f"cast_{tag}_layer_shape_{png.stem}", "--"] + env_ +
                         [sys.executable, str(HERE / "img3d.py"), str(png), str(glb_l), "--res", "1024", "--build", "rocm"],
                         W / f"layer_shape_{png.stem}.log")
                if p.returncode or not glb_l.exists():
                    continue
                outd = ld / f"fit_{png.stem}"
                jp_ = ld / f"fit_{png.stem}.json"
                jp_.write_text(json.dumps({"body_fbx": str(fbx / f"{mh_name}_Body.fbx"),
                                           "head_fbx": str(fbx / f"{mh_name}_Head.fbx"), "figure_glb": str(glb_l),
                                           "outfit_fbx": str(combo / "outfit.fbx"), "out": str(outd), "draw_deg": ab,
                                           "drawing_png": str(png)}, indent=1))
                p = line(["--line-cmd", "igpu", "combo", "--id", f"cast_{tag}_layer_fit_{png.stem}", "--", "flock",
                          "/tmp/gpu-igpu.lock", BLENDER, "-b", "--factory-startup", "-t", "0", "--python",
                          str(HERE / "bl_layer.py"), "--", str(jp_)], W / f"layer_fit_{png.stem}.log")
                if p.returncode or not (outd / "report.json").exists():
                    continue
                rep_l = json.loads((outd / "report.json").read_text())
                if (rep_l.get("checks") or {}).get("ok") and rep_l.get("layer_verts", 0) > 2000:
                    passed.append((png, outd, rep_l))
            pick_l = None
            if len(passed) > 1:
                pick_l = layer_pick(ld, a.sheet, [x[0] for x in passed], tag)
                png_, outd, rep_l = passed[pick_l["pick"]]
            elif passed:
                png_, outd, rep_l = passed[0]
            if not passed:
                layer_p.write_text(json.dumps({"ok": False, "report": rep_l}, indent=1))
                raise RuntimeError(f"OUTER LAYER: no drawing gave a layer that hangs from the shoulders "
                                   f"({json.dumps((rep_l or {}).get('checks'))}); the entry keeps it as a missing layer")
            dest_l = f"/Game/Char/Outfit{tag}L"
            ue_job(W, "import_layer", str(ASSETS / "characters" / "char_render.py"), {
                "out": str(W / "ue" / "import_layer"),
                "import_outfit": {"outfit_fbx": str(outd / "outfit.fbx"), "textures": str(outd / "textures"),
                                  "dest": dest_l, "cloth_slots": False}}, 900)
            garment = f"{dest_l}/Outfit/outfit"
            ids = [a.id] + list(json.loads(seats_p.read_text()).values()) if seats_p.exists() else [a.id]
            for lid in ids:
                ep = Path(C.LIBRARY) / "character" / lid / "character.json"
                e = json.loads(ep.read_text())
                g_ = [x for x in e["garments"] if not x.startswith("@")]
                t_ = [x for x in e["garments"] if x.startswith("@")]
                if garment not in g_:
                    g_.append(garment)
                e["garments"] = g_ + t_
                e["layers"] = [{"layer": phrase, "garment": garment, "drawn_at_deg": ab, "checks": rep_l.get("checks"),
                                "colour": rep_l.get("colour_from_drawing")}]
                e["missing_layers"] = []
                if "facts" in e:
                    e["facts"]["missing_layers"] = []
                ep.write_text(json.dumps(e, indent=1))
            col_ = rep_l.get("colour_from_drawing") or {}
            CH["layer"] = {"phrase": phrase, "drawing": png_.name, "passed": [x[0].name for x in passed],
                           "pick": pick_l, "checks": rep_l.get("checks"), "colour": col_,
                           "note": ("the 3D texture came out grey; rescaled to the drawing's colour"
                                    if col_.get("mesh_came_out_grey") else
                                    "colour rescaled to the drawing's" if col_.get("rescaled") else "colour as made")}
            layer_p.write_text(json.dumps({"ok": True, **CH["layer"]}, indent=1))
            log(f"outer layer: {CH['layer']['note']}; checks {json.dumps(rep_l.get('checks'))}")
    # ---------------------------------------------------------------- skin through the cloth, measured
    # flat-lit studio stills (no firelight to give dark cloth the face's warmth), each with a twin without the face
    # mesh: the head mesh's skin-coloured pixels at the shoulders must be under 0.2 % of the shoulder region
    skin_p = W / "skin_probe.json"
    if not a.whole and not done("check", skin_p):          # (the whole figure shows no MetaHuman skin at all)
        @timed("skin_probe")
        def _():
            cj4 = json.loads((lib / "character.json").read_text())
            out = W / "ue" / "skin_flat"
            out.mkdir(parents=True, exist_ok=True)
            shots = [{"name": n, "char": "a", "frame": fr, "az": az, "probe_face": True}
                     for n, fr, az in (("front", "medium", 0), ("q34", "medium", 35), ("side", "medium", 80),
                                       ("back", "medium", 180), ("close", "close", 15))]
            ue_job(W, "skin_flat", str(ASSETS / "characters" / "char_render.py"), {
                "map": "/Engine/Maps/Templates/Template_Default", "warm_frames": 150, "settle_frames": 30,
                "key_cd": 22, "fill_cd": 22, "rim_cd": 8, "exposure_bias": -0.5, "out": str(out),
                "characters": {"a": {"path": cj4["blueprint"], "garments": cj4["garments"], "hide_body": True}},
                "shots": shots}, 1200)
            p = subprocess.run([sys.executable, str(ASSETS / "characters" / "skin_probe.py"),
                                str(W / "ue" / "skin_flat" / "result.json"), "--out", str(skin_p)]
                               + (["--neckline", neckline_now()] if neckline_now() else [])
                               + (["--neck-half", str(round(_op["half_width_of_span"] + NECK_MARGIN, 3))]
                                  if (_op := json.loads((combo / "report.json").read_text()).get("neck_opening"))
                                  else []),
                               capture_output=True, text=True)
            CH["skin_probe"] = json.loads(skin_p.read_text())
            if p.returncode:
                raise RuntimeError(f"skin shows through the cloth: {json.dumps(CH['skin_probe']['shots'])}")
    save()
    log(f"done: {a.id} in {TT.get('total')} s of steps; {json.dumps(TT)}")


def groom_words(hair, beard=None, moustache=None):
    """Words for the grooms a character was built with (names like Hair_S_UpdoBuns, Mustache_L_Full)."""
    import re as _re
    LEN = {"S": "short", "M": "medium-length", "L": "long"}
    STY = [("Updo", "pinned up"), ("Buns", "in a bun"), ("SweptUp", "swept up and back"), ("SlickBack", "slicked back"),
           ("PulledBack", "pulled back"), ("Ponytail", "in a ponytail"), ("Braids", "braided"), ("Bald", "balding"),
           ("HairLoss", "thinning"), ("Recede", "receding"), ("Messy", "untidy"), ("Curly", "curly"), ("Afro", "curly"),
           ("Coil", "coiled"), ("Clean", "neatly cut"), ("Casual", "loosely cut"), ("BuzzCut", "cropped close"),
           ("BrushCut", "cropped"), ("Fade", "cropped at the sides"), ("Fringe", "with a fringe"),
           ("Bangs", "with a fringe"), ("Bob", "cut to the jaw"), ("Layered", "layered"), ("Straight", "straight"),
           ("Pixie", "cropped short")]

    def parse(name):
        m = _re.search(r"_(S|M|L)_([A-Za-z]+)", name or "")
        return (m.group(1), m.group(2)) if m else (None, None)
    worn = None
    if hair:
        ln, st = parse(hair)
        ws = [w for k, w in STY if k in (st or "")]
        # an up-do's groom length is the length left loose, not the hair's: no length word for it
        lw = [] if any(k in (st or "") for k in ("Updo", "Buns", "SweptUp", "Ponytail")) else [f"{LEN.get(ln, '')} hair".strip()]
        worn = ", ".join(x for x in (lw + ws) if x)
        worn = ("hair " + worn) if not lw else worn
    fh = []
    if moustache:
        ln, st = parse(moustache)
        fh.append(" ".join(x for x in ({"L": "large", "M": "", "S": ""}.get(ln, ""), _re.sub(r"(?<!^)([A-Z])", r" \1", st or "").lower(), "moustache") if x))
    if beard:
        ln, st = parse(beard)
        fh.append(" ".join(x for x in ({"L": "long", "M": "", "S": "short"}.get(ln, ""), _re.sub(r"(?<!^)([A-Z])", r" \1", st or "").lower(), "beard") if x))
    return worn, (" and ".join(fh) if fh else "none")


def lie_frame(spot, sup):
    """the bed's lie box in the set frame: its four corners, the pillow's centre and the foot end's middle"""
    import math as _m
    x0, y0, yaw = spot
    f = [-_m.sin(_m.radians(yaw)), -_m.cos(_m.radians(yaw))]          # the bed's foot direction (its front)
    ax, ay = [-f[1], f[0]], [-f[0], -f[1]]                            # asset x and y in the set
    to = lambda p: [x0 + p[0] * ax[0] + p[1] * ay[0], y0 + p[0] * ax[1] + p[1] * ay[1]]
    b = sup["box"]
    xs, ys = b["x_m"], b["y_m"]
    sgn = 1 if sup.get("head", "+y") == "+y" else -1
    py = sup.get("pillow_y_m") or ([ys[1] - 0.35, ys[1]] if sgn > 0 else [ys[0], ys[0] + 0.35])
    xm = (xs[0] + xs[1]) / 2
    return {"corners": [to([xs[0], ys[0]]), to([xs[1], ys[0]]), to([xs[1], ys[1]]), to([xs[0], ys[1]])],
            "head_xy": to([xm, (py[0] + py[1]) / 2]), "foot_xy": to([xm, ys[0] if sgn > 0 else ys[1]]),
            "facing": f if sgn > 0 else [-f[0], -f[1]], "bed_facing": f, "bed_xy": [x0, y0]}


def lie_pose(pelvis_z_cm, top_z_m, variant):
    """the scene tool's pose steps on the standing pose: onto the back about the hips (head to the actor's back,
    face up), the back on the surface; propped: the chest raised and the head forward"""
    back = 0.10                                                       # hip joint above the surface lying on the back
    steps = [["rot", "pelvis", [1, 0, 0], 90.0],
             ["move", "pelvis", [0.0, 0.0, round((top_z_m + back) * 100 - pelvis_z_cm, 1)]]]
    if variant == "propped":
        steps += [["rot", "spine_02", [1, 0, 0], -35.0], ["rot", "neck_01", [1, 0, 0], -15.0]]
    else:
        steps += [["rot", "neck_01", [1, 0, 0], -12.0]]               # the head on the pillow
    # (the arms lie along the body: a standing figure's clearance from its clothes spread them wide under the cover)
    return {"pose": "stand", "params": {"steps": steps, "lie": variant, "arm_clear_deg": 0.0, "forearm_fwd_deg": 0.0}}


def lie_stance(W, a, tag, lib, cj0, bed, sup, variant, combo, scene_py, env):
    fr = lie_frame(a.lie_spot, sup)
    top_z = float(sup["box"]["top_z_m"])
    sid = f"liej_{tag}_{bed}_{variant}"
    spec_bed = {"asset": f"prop/{bed}", "position": fr["bed_xy"], "facing": fr["bed_facing"]}
    pelvis_z = 95.0
    pos = list(fr["head_xy"])
    meas = None
    for k in range(2):                       # placed, measured, the error taken off once
        bp = lie_pose(pelvis_z, top_z, variant)
        c = {"id": a.id, "position": [round(pos[0], 3), round(pos[1], 3)], "facing": fr["facing"], "body_pose": bp}
        spec = {"id": sid, "set": a.set, "guides": False, "characters": {"a": c, "bed": spec_bed},
                "cameras": [{"name": "wide", "kind": "full", "subject": "a", "focal_mm": 28, "az": -60, "el": 25}]}
        sp = W / f"{sid}.json"
        sp.write_text(json.dumps(spec, indent=1))
        p = subprocess.run([sys.executable, str(scene_py), str(sp)], cwd=str(Path(a.scene_tool).parents[2]),
                           env=env, capture_output=True, text=True)
        if p.returncode:
            raise RuntimeError(f"lie joints scene {bed} {variant} failed: {p.stdout[-800:]}")
        meas = json.loads((W / "scenes" / sid / "result.json").read_text())["subjects"]["a"]["measured"]
        jt = meas["joints"]
        # the head joint onto the pillow's centre, the hips onto the surface
        dx, dy = fr["head_xy"][0] - jt["head"][0], fr["head_xy"][1] - jt["head"][1]
        pz = (jt["thigh_l"][2] + jt["thigh_r"][2]) / 2
        if k == 0:
            # the move was computed against a guessed standing hip height: the measured error taken off
            pelvis_z += (pz - (top_z + 0.10)) * 100
            pos = [pos[0] + dx, pos[1] + dy]
            if abs(dx) < 0.02 and abs(dy) < 0.02 and abs(pz - (top_z + 0.10)) < 0.02:
                break
    jt = meas["joints"]
    lj = {"combo": str(combo), "joints": jt, "position": c["position"], "facing": c["facing"], "mode": "push",
          "iters": 0, "shoulder_ease_m": 0.0, "thigh_from": 0.0,
          "lie": dict(fr, top_z_m=top_z), "out": str(W / f"lying_{bed}_{variant}")}
    jp = W / f"settle_lie_{bed}_{variant}.json"
    jp.write_text(json.dumps(lj, indent=1))
    p = line(["--line-cmd", "igpu", "settle", "--id", f"cast_{tag}_lie_{bed}_{variant}", "--", "flock",
              "/tmp/gpu-igpu.lock", BLENDER, "-b", "--factory-startup", "-t", "0", "--python",
              str(HERE / "bl_settle.py"), "--", str(jp)], W / f"settle_lie_{bed}_{variant}.log")
    if p.returncode:
        raise RuntimeError(f"lie settle {bed} {variant} failed")
    sr = json.loads((W / f"lying_{bed}_{variant}" / "settle.json").read_text())
    # the bed and the variant both in the name, whatever its length (cut at 14 characters the variant fell off a
    # long bed id: the flat and the propped stance were one engine asset, and the last built was worn by both)
    import hashlib
    key = ("".join(ch for ch in bed.upper() if ch.isalnum())[:8] + hashlib.sha1(bed.encode()).hexdigest()[:4].upper()
           + variant[:1].upper())
    dest = f"/Game/Char/Outfit{tag}L{key}"
    whole = bool(cj0.get("whole_figure"))        # (a whole figure lies as it is; only the bed's cover is the settle's)
    if not whole:
        ue_job(W, f"import_lie_{bed}_{variant}", str(ASSETS / "characters" / "char_render.py"), {
            "out": str(W / "ue" / f"import_lie_{bed}_{variant}"),
            "import_outfit": {"outfit_fbx": str(W / f"lying_{bed}_{variant}" / "outfit.fbx"),
                              "textures": str(W / f"lying_{bed}_{variant}" / "textures"), "dest": dest,
                              "cloth_slots": False}}, 900)
    cover = None
    if (W / f"lying_{bed}_{variant}" / "cover.fbx").exists():
        ue_job(W, f"import_cover_{bed}_{variant}", str(ASSETS / "characters" / "char_render.py"), {
            "out": str(W / "ue" / f"import_cover_{bed}_{variant}"),
            "import_outfit": {"outfit_fbx": str(W / f"lying_{bed}_{variant}" / "cover.fbx"),
                              "textures": str(W / f"lying_{bed}_{variant}" / "cover_textures"), "dest": dest + "C",
                              "cloth_slots": False}}, 900)
        # (the imported mesh is named after its file: the import's own result says where it is)
        cover = json.loads((W / "ue" / f"import_cover_{bed}_{variant}" / "result.json").read_text())[
            "import_outfit"]["outfit"]
    # the lie points in the stance's actor frame (metres): x along the actor's left, y forward, z up
    import math as _m
    f_ = c["facing"]
    lft = [-f_[1], f_[0]]
    to_actor = lambda q: [round((q[0] - c["position"][0]) * lft[0] + (q[1] - c["position"][1]) * lft[1], 3),
                          round((q[0] - c["position"][0]) * f_[0] + (q[1] - c["position"][1]) * f_[1], 3),
                          round(q[2], 3)]
    heel = [(jt["foot_l"][i] + jt["foot_r"][i]) / 2 for i in range(3)]
    pel = [(jt["thigh_l"][i] + jt["thigh_r"][i]) / 2 for i in range(3)]
    lid = f"{a.id}_lying_{bed}" + ("_propped" if variant == "propped" else "")
    st = {"base": a.id, "pose": "lying", "variant": variant,
          "support": {"asset": bed, "posture": "lying", "top_z_m": top_z, "length_m": sup.get("length_m")}}
    ent = dict(cj0, id=lid, garments=([g for g in cj0["garments"] if not g.startswith("@")][:1] if whole
                                      else [f"{dest}/Outfit/outfit"]) + ([cover] if cover else [])
               + [g for g in cj0["garments"] if g.startswith("@")],
               stance=st, body_pose=c["body_pose"],
               lie={"head_m": to_actor(jt["head"]), "pelvis_m": to_actor(pel), "heels_m": to_actor(heel)},
               face_up=True, cover={"asset": cover, "over": "chest" if cover else None},
               note=f"Lying only, on {bed} ({variant}): {a.id} on its back with its cloth laid on the mattress "
                    f"(bl_settle, no relaxation) and the bed's cover over the body (a separate piece, listed in "
                    f"garments and in cover)")
    ent.pop("relaxed_arm_steps", None)
    (Path(C.LIBRARY) / "character" / lid).mkdir(parents=True, exist_ok=True)
    (Path(C.LIBRARY) / "character" / lid / "character.json").write_text(json.dumps(ent, indent=1))
    return {"id": lid, "settle": {k: sr.get(k) for k in ("lie_cloth_lifted", "lie_min_cloth_z_m", "body_skin", "cover",
                                                         "pose_err_cm")}, "position": c["position"],
            "facing": c["facing"], "bed": spec_bed, "frame": fr}


# the skin probe's left-out column for an open or base neckline: the measured opening plus this share of the span
NECK_MARGIN = 0.04

LAYER_QUESTION = ("In the middle is a character sheet of a person. On the left (A) and on the right (B) are two "
                  "drawings of the outer garment alone, shown on a display figure. Which drawing, A or B, shows that "
                  "outer garment worn the way the person in the middle wears it: over the same shoulder or shoulders, "
                  "falling across the body the same way, to the same length? Answer A or B.")


def layer_pick(ld, sheet, pngs, tag):
    """The outer layer's drawing worn as on the sheet: every pair of passing drawings beside the sheet, both orders
    (the model leans to one side when it cannot tell), asked of the vision model; the most wins, ties to the
    earlier drawing."""
    from PIL import Image as _Im
    sh = _Im.open(sheet).convert("RGB")
    h = 768
    sh = sh.resize((max(1, sh.width * h // sh.height), h))
    tiles = [_Im.open(p).convert("RGB") for p in pngs]
    tiles = [t.resize((max(1, t.width * h // t.height), h)) for t in tiles]
    qs = []
    for i in range(len(pngs)):
        for k in range(len(pngs)):
            if i == k:
                continue
            A, B = tiles[i], tiles[k]
            im = _Im.new("RGB", (A.width + sh.width + B.width + 40, h + 40), "white")
            im.paste(A, (0, 40))
            im.paste(sh, (A.width + 20, 40))
            im.paste(B, (A.width + sh.width + 40, 40))
            from PIL import ImageDraw as _D
            dr = _D.Draw(im)
            dr.text((A.width // 2, 10), "A", fill="black")
            dr.text((A.width + 20 + sh.width // 2 - 20, 10), "sheet", fill="black")
            dr.text((A.width + sh.width + 40 + B.width // 2, 10), "B", fill="black")
            pp = ld / f"pick_{i}_{k}.png"
            im.save(pp)
            qs.append({"id": f"{i}|{k}", "images": [str(pp)], "max_new_tokens": 3, "max_side": 1280,
                       "ask": {"pick": LAYER_QUESTION}})
    jq = ld / "pick.jsonl"
    jq.write_text("".join(json.dumps(q) + "\n" for q in qs))
    p = line(["--line-cmd", "image", "choose", "--id", f"cast_{tag}_layer_pick", "--", "bash", "-c",
              f"cd {REPO} && bash {CARD} 05 {QWEN_PY} -I tools/vl_judge.py < {jq}"], ld / "pick.log")
    wins, answers = [0] * len(pngs), {}
    for ln in (ld / "pick.log").read_text().splitlines():
        if not ln.startswith("{") or '"answers"' not in ln:
            continue
        r = json.loads(ln)
        i, k = (int(x) for x in r["id"].split("|"))
        v = (r["answers"].get("pick") or "").strip().upper()
        answers[r["id"]] = v
        if v.startswith("A"):
            wins[i] += 1
        elif v.startswith("B"):
            wins[k] += 1
    if not answers:
        raise RuntimeError(f"OUTER LAYER: the vision model gave no answers for the drawing pick (rc {p.returncode})")
    best = max(range(len(pngs)), key=lambda n: (wins[n], -n))
    return {"pick": best, "drawings": [Path(x).name for x in pngs], "wins": wins, "answers": answers}


def entry_facts(ch, missing=None):
    """The entry's "facts" block, only from the build's own facts: the grooms it was built with and the colour word
    it was built in (the chooser's), the layers left out. What the redraw says about this person's hair, facial
    hair and missing layers."""
    g = ch.get("grooms") or {}
    clear = set(ch.get("clear_slots") or [])
    worn, fh = groom_words(g.get("Hair"), None if "Beard" in clear else g.get("Beard"),
                           None if "Mustache" in clear else g.get("Mustache"))
    return {"hair_colour": (ch.get("scores") or {}).get("hair_colour"), "hair_worn": worn, "facial_hair": fh,
            "missing_layers": list(missing or []), "grooms": {k: v for k, v in g.items() if k != "Eyebrows"}}


def spread_look(look, outer_words=None):
    """The look line for the spread drawing: what the hands carry or hold and any headwear go (the hands must be open
    and empty for the gate and the fit; the MetaHuman keeps its own head and hair)."""
    import re
    held = re.compile(r"\b(carrying|holding)\b")
    head = re.compile(r"hat\b|top-hat|bonnet|\bcap\b|veil|deerstalker")
    text = look.rstrip(". ")
    before, _, worn = text.partition("wearing ")
    desc = [p.strip() for p in before.split(",") if p.strip() and not held.search(p)]
    items = [q.strip() for q in re.split(r",| and ", worn) if q.strip()]
    keep, stop = [], False
    for q in items:
        if held.search(q):
            stop = True                     # "..., carrying a crop": the rest of the list is what is carried
        if stop or head.search(q):
            continue
        if outer_words and all(w in q.lower() for w in outer_words):     # the whole phrase, never one word of it
            # an outer layer hanging from the shoulders (a fact): left out of the spread drawing and the fitted
            # garment (it fills the space under the arms); the entry records it as a missing layer
            continue
        keep.append(q)
    out = ", ".join(desc)
    if keep:
        out += ", wearing " + ", ".join(keep)
    return out


def arm_steps(deg):
    return [["rot", "upperarm_l", [0, 1, 0], -deg], ["rot", "upperarm_r", [0, 1, 0], deg],
            ["rot", "lowerarm_l", [1, 0, 0], 10], ["rot", "lowerarm_r", [1, 0, 0], 10]]


if __name__ == "__main__":
    main()
