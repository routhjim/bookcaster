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
  import   engine: the outfit, a library entry (character.json) on the assembled Blueprint
  arms     engine + image station: one relaxed scene with three copies at 4 / 8 / 12 degrees of arm abduction,
           the scene tool's guides, arm_gap.py: the smallest abduction whose forearms separate from the body in
           >= 8 of 10 rows on both sides at the wide shot goes into character.json "relaxed_arm_steps"
  check    engine: wide / medium / close stills in the set (check.png)
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
THUMBS = os.environ.get("ASSETS_MH_THUMBS", str(Path(C.UE_HOME) / "lab" / "character" / "thumbs"))
STEPS = ["choose", "mh", "draw", "gate", "shape", "combo", "import", "arms", "check"]
POSE_WORDS = ("standing in a neutral rigging pose, seen straight from the front: both arms held straight out away "
              "from the body, angled down about 45 degrees, a clear open gap between each arm and the clothes, hands "
              "open with the fingers together and the palms facing down, legs slightly apart, looking at the viewer. "
              "Full length from the top of the head to the feet, the whole figure in the picture with a margin, "
              "bare-headed (no hat). Plain flat light grey background, soft even studio light, no shadow on the floor, "
              "nothing else in the picture. A realistic full-colour photograph of a costumed actor, sharp focus.")


def log(m):
    print(f"[cast {time.strftime('%H:%M:%S')}] {m}", flush=True)


def line(args, log_file=None):
    """One line entry, followed to its end (exit code 75/76: the line is full / the same entry waits: wait for room
    by following the station's queue, never a busy loop)."""
    if log_file and "--line-cmd" in args:
        i = args.index("--")
        args = args[:i] + ["--log", str(log_file)] + args[i:]
    p = subprocess.run(LINE + args, cwd=str(REPO), capture_output=True, text=True)
    if p.returncode in (75, 76):
        # never loop on a full line: stop here; a re-run resumes from this step
        raise SystemExit(f"the line is full or the same entry waits (exit {p.returncode}): run again later")
    return p


def ue_job(work, name, script, job, timeout=1800):
    d = Path(work) / "ue" / name
    (d / "sched").mkdir(parents=True, exist_ok=True)
    job.setdefault("result", str(d / "result.json"))
    sj = {"project": PROJECT, "script": script, "job": job, "outputs": [job["result"]], "timeout_s": timeout,
          "id": f"cast_{Path(work).name}_{name}", "log_dir": str(d / "sched")}
    jp = d / "job.json"
    jp.write_text(json.dumps(sj, indent=1))
    p = line(["--ue-job", str(jp)])
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
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--from", dest="from_step", default=None, choices=STEPS)
    ap.add_argument("--mh", default=None, help="an existing MetaHuman Character name (/Game/Char/<name>): skip choose and"
                    " mh; --mh-fbx DIR holds its <name>_Body.fbx / _Head.fbx, --mh-blueprint its assembled Blueprint")
    ap.add_argument("--mh-fbx", default=None)
    ap.add_argument("--mh-blueprint", default=None)
    a = ap.parse_args()
    W = Path(a.work)
    W.mkdir(parents=True, exist_ok=True)
    tag = "".join(ch for ch in a.id.upper() if ch.isalnum())[:10]
    choices_p, times_p = W / "choices.json", W / "times.json"
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
                      "--out", str(W / "choice.json"), "--look", a.look], W / "choose.log")
            if p.returncode:
                raise RuntimeError(f"choose failed: {p.stdout[-800:]}")
    ch = json.loads((W / "choice.json").read_text())
    CH["metahuman"] = {k: ch[k] for k in ("base", "gender", "grooms", "clear_slots", "body")}
    CH["metahuman_scores"] = ch["scores"]
    save()
    mh_name = a.mh or f"MHC_AUTO_{tag}"
    fbx = Path(a.mh_fbx) if a.mh_fbx else W / "fbx"
    # ---------------------------------------------------------------- mh
    if not done("mh", fbx / f"{mh_name}_Head.fbx"):
        @timed("mh")
        def _():
            os.environ.setdefault("MH_EXTRA_ARGS", "-NoMetaHumanAccountPortalLoginFallback")
            r = ue_job(W, "mh", str(ASSETS / "characters" / "char_render.py"), {
                "out": str(W / "ue" / "mh"),
                "make": [{"name": mh_name, "base": ch["base"], "grooms": ch["grooms"], "clear_slots": ch["clear_slots"],
                          "params": ch["params"], "body": ch["body"]}],
                "cloud": {"path": f"/Game/Char/{mh_name}", "build_path": "/Game/MetaHumans"},
                "characters": {"a": {"path": f"/Game/Char/{mh_name}"}},
                "export_geometry": {"key": "a", "folder": f"/Game/Char/Geo_{tag}", "fbx_dir": str(fbx)}}, 2400)
            CH["mh_build"] = {"blueprint": (r.get("cloud") or {}).get("blueprint"), "make": r.get("make")}
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
            deg = 48 if any(w_ in lk for w_ in ("dress", "gown", "skirt")) else 42
            pose_map = draw / f"pose{deg}.png"
            subprocess.run([sys.executable, str(HERE / "apose_map.py"), str(pose_map), "--deg", str(deg)], check=True)
            look = spread_look(a.look)
            CH["draw_look"], CH["pose_deg"] = look, deg
            tried, meshes, chosen = [], [], None
            for rnd in range(a.rounds):
                seeds = [5 + 6 * (3 * rnd + k) for k in range(3)]
                jobs = [{"id": f"{tag}_s{s_}", "seed": s_, "path": str(draw / f"spread_s{s_}.png"), "size": [768, 1344],
                         "prompt": f"{a.trigger}, {look}, {POSE_WORDS}", "refs": [a.sheet],
                         "loras": [[a.lora, 0.8, [0, 0, 1, 1]]],
                         "control": {"image": str(pose_map), "type": "pose", "scale": 0.8}} for s_ in seeds]
                jl = draw / f"jobs_r{rnd}.jsonl"
                jl.write_text("".join(json.dumps(j) + "\n" for j in jobs))
                p = line(["--line-cmd", "image", "refs", "--id", f"cast_{tag}_draw{rnd}", "--", "bash", "-c",
                          f"cd {REPO} && bash {CARD} 05 {QWEN_PY} -I tools/qwen_image.py --serve < {jl}"],
                         W / f"draw_r{rnd}.log")
                if p.returncode:
                    raise RuntimeError(f"drawing failed: {p.stdout[-800:]}")
                passing = []
                for j in jobs:
                    g = subprocess.run([sys.executable, str(HERE / "pose_gate.py"), j["path"]], capture_output=True,
                                       text=True)
                    res = json.loads(g.stdout.strip().splitlines()[-1]) if g.stdout.strip() else {"file": j["path"]}
                    res["pass"] = g.returncode == 0
                    tried.append(res)
                    if res["pass"]:
                        sides = [v for k, v in res.items() if isinstance(v, dict) and "angle_deg" in v]
                        passing.append((min((s_.get("angle_deg") or 0) * (s_.get("armpit") or 0) for s_ in sides),
                                        j["path"]))
                # every drawing that passes goes to 3D, best first, until a mesh passes the mesh gate
                for _sc, path in sorted(passing, reverse=True):
                    out = draw / (Path(path).stem + ".glb")
                    env = ["env", f"ASSETS_TRELLIS_OMP={OMP}"] if OMP else []
                    p = line(["--line-cmd", "blender", "shape", "--id", f"cast_{tag}_{Path(path).stem}", "--"] + env +
                             [sys.executable, str(HERE / "img3d.py"), path, str(out), "--res", "1024", "--build", "rocm"],
                             W / f"shape_{Path(path).stem}.log")
                    if p.returncode or not out.exists():
                        meshes.append({"file": str(out), "error": p.stdout[-300:]})
                        continue
                    g = subprocess.run([HY_PY, "-I", str(HERE / "mesh_gate.py"), str(out)], capture_output=True, text=True)
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
    # ---------------------------------------------------------------- combo
    combo = W / "combo"
    if not done("combo", combo / "outfit.fbx"):
        @timed("combo")
        def _():
            m = legs_apart(drawing)
            lk = a.look.lower()
            job = {"body_fbx": str(fbx / f"{mh_name}_Body.fbx"), "head_fbx": str(fbx / f"{mh_name}_Head.fbx"),
                   "figure_glb": str(glb), "out": str(combo), "hair_cap": False, "neck_cut_m": 0.03}
            if m["long_skirt"]:
                job.update(sleeve_radius_m=0.12, skirt_pelvis=1.0, turn_legs=False, shoe_top_m=0.0, skirt_min_z_m=-1.0,
                           long_skirt=True, sleeve_slim_k=0.35, shoulder_gap_m=0.012,
                           collar_build={"style": "stand", "cravat": False, "plate": False, "cuffs": {"width_m": 0.03},
                                         "visible_neck_frac": 0.10, "white_top_m": 0.012, "open_deg": 0})
            else:
                coat_skirt = (m["thigh_run_w_frac"] or 0) > 0.17
                job.update(sleeve_radius_m=0.15, skirt_pelvis=1.0 if coat_skirt else 0.6)
                neck = "tie" if " tie" in lk else ("bow" if ("cravat" in lk or "collar" in lk) else None)
                if neck:
                    job["collar_build"] = {"style": "wing" if neck == "bow" else "stand", "neckwear": neck,
                                           "visible_neck_frac": 0.03 if neck == "bow" else 0.05}
            CH["combo"] = {"measured": m, "job": {k: v for k, v in job.items() if not k.endswith(("fbx", "glb"))}}
            jp = W / "combo_job.json"
            jp.write_text(json.dumps(job, indent=1))
            p = line(["--line-cmd", "igpu", "combo", "--id", f"cast_{tag}_combo", "--", "flock", "/tmp/gpu-igpu.lock",
                      BLENDER, "-b", "--factory-startup", "-t", "0", "--python", str(HERE / "bl_combo.py"), "--",
                      str(jp)], W / "combo.log")
            if p.returncode or not (combo / "outfit.fbx").exists():
                raise RuntimeError(f"combo failed: {p.stdout[-800:]}")
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
            lib.mkdir(parents=True, exist_ok=True)
            (lib / "character.json").write_text(json.dumps({
                "class": "character", "kind": "metahuman", "project": "CharLab", "blueprint": bp,
                "rebuild_from": f"/Game/Char/{mh_name}", "bp_deps": ["/MetaHumanCharacter/Animation/ABP_MH_LiveLink"],
                "character_asset": f"/Game/Char/{mh_name}", "hide_body": True, "id": a.id,
                "garments": [f"{dest}/Outfit/outfit"], "body_swap": None, "garment_material": None,
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
                chars[k] = {"id": a.id, "position": [x, 0.6], "facing": [0, -1], "body_pose": {
                    "pose": "stand_relaxed", "params": {"weight_on": "r", "chest_deg": 0, "head_deg": 0, "head_tilt": 0,
                                                        "hand_at_lapel": False, "steps": arm_steps(ab)}}}
                cams.append({"name": f"wide_{k}", "kind": "full", "subject": k, "focal_mm": 50, "az": 0})
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
            for k, ab in (("a4", 4), ("a8", 8), ("a12", 12)):
                sh = res.get(f"wide_{k}", {}).get(k, {})
                if all((sh.get(s_) or {}).get("separated", 0) >= 8 for s_ in ("l", "r")):
                    pick = ab
                    break
            CH["arms"] = {"measured": res, "abduct_deg": pick}
            cj2 = json.loads((lib / "character.json").read_text())
            cj2["relaxed_arm_steps"] = arm_steps(pick)
            (lib / "character.json").write_text(json.dumps(cj2, indent=1))
    # ---------------------------------------------------------------- check stills
    if not done("check", W / "scenes" / f"check_{tag}" / "close.png"):
        @timed("check")
        def _():
            cj3 = json.loads((lib / "character.json").read_text())
            spec = {"id": f"check_{tag}", "set": a.set, "guides": False, "characters": {"a": {
                "id": a.id, "position": [0.3, 0.8], "facing": [0.3464, -0.9381], "body_pose": {
                    "pose": "stand_relaxed", "params": {"weight_on": "r", "chest_deg": 4, "head_deg": 0, "head_tilt": -2,
                                                        "hand_at_lapel": False,
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
    save()
    log(f"done: {a.id} in {TT.get('total')} s of steps; {json.dumps(TT)}")


def spread_look(look):
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
