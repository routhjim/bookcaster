"""Scene tool: scene spec (JSON) -> one scheduler UE job -> keyframe stills.

    python tools/assets/scenes/scene.py SCENE.json                # resolve, submit, wait, write the stills
    python tools/assets/scenes/scene.py SCENE.json --dry-run      # resolve and print the UE job only
    python tools/assets/scenes/scene.py SCENE.json --cameras wide,close

A scene puts library characters into a library set and photographs them. Everything is referenced by library id
(<library>/set/<id>/asset.json, <library>/character/<id>/character.json), so a re-render picks up whatever the set
builder and the character tools have made of those ids since. Sets, characters and props live in ONE UE project
(the character's: MetaHuman characters need its plugins); the tool refuses a set built in another project and
says how to rebuild it there. See README.md for the spec.

Output (<out>/<scene id>/, default $ASSETS_UE_HOME/lab/scenes/out, SCENES_OUT overrides): <camera>.png per camera,
job.json (the UE job), result.json (cameras, measured subjects, frame checks, head-attach checks, times),
times.json, sheet.png (all stills, when Pillow is importable; pasting only).
Stdlib only (Pillow optional for the sheet).
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sets"))
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402
import creature  # noqa: E402
import layout  # noqa: E402

CHAR_TOOLS = Path(os.environ.get("SCENES_CHAR_TOOLS", str(HERE.parent / "characters")))
SET_TOOLS = HERE.parent / "sets" / "ue"
WORK_POSE_ACTIONS = ("tool_raised", "striking_down", "hands_at_work")    # ue/pose.py WORK_ACTIONS (whole-body stance)
READY_POSE_ACTIONS = ("weapon_ready",)    # ue/pose.py READY_ACTIONS (a fighter's guard: the same stance, no work top)
RT_KEYS = {"r.RayTracing": "True", "r.RayTracing.Shadows": "False", "r.Lumen.HardwareRayTracing": "True",
           "r.SkinCache.CompileShaders": "True"}


def line_client() -> Path:
    """The line's client: ASSETS_LINE_CLIENT (the run_asset.py of the worktree the line runs from), else this
    copy's. Run from its own worktree, so a client that starts the line starts that worktree's daemon."""
    return Path(os.environ.get("ASSETS_LINE_CLIENT") or (HERE.parent / "run_asset.py")).resolve()


def log(m):
    print(f"[scene {time.strftime('%H:%M:%S')}] {m}", flush=True)


def out_root():
    return Path(os.path.expanduser(os.environ.get("SCENES_OUT", str(Path(C.UE_HOME) / "lab" / "scenes" / "out"))))


def project_path(name_or_path):
    p = Path(os.path.expanduser(str(name_or_path)))
    if p.suffix == ".uproject":
        return p
    return Path(C.UE_HOME) / "proj" / str(name_or_path) / f"{name_or_path}.uproject"


def ensure_raytracing(project):
    """Hardware ray tracing in the scene project (like the asset chain's own project). Takes effect at the next
    editor start (the first start after a change compiles ray-tracing shaders). Returns the keys it wrote."""
    ini = project.parent / "Config" / "DefaultEngine.ini"
    text = ini.read_text()
    have = {ln.split("=", 1)[0].strip() for ln in text.splitlines() if "=" in ln}
    missing = {k: v for k, v in RT_KEYS.items() if k not in have}
    if not missing or os.environ.get("SCENES_UE_RAYTRACING", "1") == "0":
        return {}
    sec = "[/Script/Engine.RendererSettings]"
    add = "".join(f"{k}={v}\n" for k, v in missing.items())
    text = text.replace(sec + "\n", sec + "\n" + add, 1) if sec in text else text + f"\n{sec}\n{add}"
    ini.write_text(text)
    return missing


KEY_LUX_EV = 6.0         # a place in the open: its key at least this many lux per unit of 2^EV100 (the exposure's
#                          own scale), so a face reads at night as by day (measured on an outdoor place, 11-17)


def key_light(set_spec_path, ev=None) -> dict | None:
    """A place in the open (a set built as a world, sets/place_build.py world_stage) is lit per shot from the
    camera's side (the editor job turns its sun for each still, framing.key_toward), in the world's own light
    colour, at least KEY_LUX_EV x 2^EV100: at the panorama's own night level a face read as 6 of 255. A room
    -> None."""
    try:
        ss = json.loads(Path(set_spec_path).read_text())
    except (OSError, ValueError):
        return None
    sun = next((l for l in ss.get("lights") or [] if l.get("type") == "sun"), None)
    if not ss.get("world") or not sun:
        return None
    ev = float(ev if ev is not None else ss.get("exposure_ev100") or 5.0)
    return {"toward_sun": sun["toward_sun"], "lux": round(max(float(sun.get("lux") or 0), KEY_LUX_EV * 2 ** ev), 1)}


def fixture_params(c: dict, pr: dict) -> dict:
    """A person's pose params with the piece of the set they work with their hands (params "fixture": {"name"}):
    a work action (WORK_POSE_ACTIONS) that from_plan stood them at a piece for ("work_at", its spec name: the set
    builder's label of its actor) with nothing worked lying on it (no blow target): the piece itself is the work."""
    if c.get("work_at") and pr.get("action") in WORK_POSE_ACTIONS and not pr.get("work_target_ue"):
        return dict(pr, fixture={"name": c["work_at"]})
    return pr


def aim_params(pr: dict) -> dict:
    """A guard's aim (from_plan's "aim_at_m": the foe's head to middle, plan metres) as pose.py takes it: UE cm."""
    if pr.get("action") in READY_POSE_ACTIONS and pr.get("aim_at_m") and "aim_target_ue" not in pr:
        return dict(pr, aim_target_ue=layout.ue_pt(pr["aim_at_m"]))
    return pr


def carried(c: dict) -> dict:
    """What a staged person's spec hands on to the engine job besides the body itself: a child's scale (stature.py:
    scaled, and lifted onto what they sit or lie on), a seat inside a cabin (seating.py), a bed's head and foot
    points for the cameras (lying.py), and the one they are with (from_plan: a pose that rests on another person is
    aimed at that person's own body in the engine; left out of the job, each of a pair aimed at a copy of itself
    and the hands of the two never met, 11-26)."""
    out = {}
    if c.get("scale"):
        out["scale"] = float(c["scale"])
    for k in ("_inside", "_lie", "near_to", "near_how"):
        if c.get(k):
            out[k] = c[k]
    return out


def resolve(spec, cams_wanted=None):
    lib = Path(C.LIBRARY)
    sp = lib / "set" / spec["set"] / "asset.json"
    if not sp.exists():
        raise SystemExit(f"set {spec['set']!r} is not in the library ({sp})")
    st = json.loads(sp.read_text())
    proj = None
    chars, figs = {}, {}
    bp_deps = []
    for key, c in spec["characters"].items():
        if c.get("asset"):             # a plain-mesh figure (a picture-to-3D statue): placed, measured from its bounds
            pos = list(c["position"]) + ([0.0] if len(c["position"]) == 2 else [])
            f = c.get("facing", [0, -1])
            n = (f[0] ** 2 + f[1] ** 2) ** 0.5
            aj = json.loads((lib / c["asset"] / "asset.json").read_text())
            figs[key] = {"mesh": aj["ue"]["path"], "loc_cm": layout.ue_pt(pos), "yaw": round(layout.yaw_from_facing(f), 3),
                         "facing": [f[0] / n, f[1] / n], "position": pos, "height_m": aj["size_m"][2],
                         "id": c["asset"]}
            rg = creature.rig(lib, c["asset"]) if creature.is_id(c["asset"]) else None
            if rg:                     # a rigged creature: its skeletal mesh holding the named pose's frame of its clip
                figs[key].update(mesh=rg["mesh"], anim=rg["anim"], t=creature.pose_time(rg, c.get("pose")),
                                 pose=c.get("pose") or rg["poses"][0], creature=True, head_len_m=rg["head_len_m"])
            continue
        cp = lib / "character" / c["id"] / "character.json"
        if not cp.exists():
            raise SystemExit(f"character {c['id']!r} has no {cp}")
        cj = json.loads(cp.read_text())
        p = project_path(cj["project"]).resolve()
        if proj and p != proj:
            raise SystemExit(f"characters live in different UE projects ({proj} vs {p})")
        proj = p
        pos = list(c["position"]) + ([0.0] if len(c["position"]) == 2 else [])
        f = c.get("facing", [0, -1])
        n = (f[0] ** 2 + f[1] ** 2) ** 0.5
        bpose = c.get("body_pose")
        if (cj.get("relaxed_arm_steps") and bpose and bpose.get("pose") in ("stand", "stand_relaxed")
                and not (bpose.get("params") or {}).get("hand_at_lapel", bpose.get("pose") == "stand_relaxed")
                and not (bpose.get("params") or {}).get("lie")         # (lying, the arms rest along the body)
                # (an action that places the arms itself keeps them: the steps turned them again afterwards)
                and (bpose.get("params") or {}).get("action") not in WORK_POSE_ACTIONS + READY_POSE_ACTIONS
                + ("veil_lift",)):
            # the character's own arm clearance (its builder measured it against its clothes) in place of the default
            pr = dict(bpose.get("params") or {})
            pr.setdefault("arm_clear_deg", 0.0)
            pr.setdefault("forearm_fwd_deg", 0.0)
            pr["steps"] = list(cj["relaxed_arm_steps"]) + list(pr.get("steps") or [])
            bpose = dict(bpose, params=pr)
        garments = list(cj.get("garments", []))
        if c.get("wear_from"):          # another wardrobe variant's extra garments (all after its first, the base
            wj = json.loads((lib / "character" / c["wear_from"] / "character.json").read_text())   # outfit) on top
            garments += [g for g in wj.get("garments", [])[1:] if g not in garments]
        chars[key] = {"id": c["id"], "path": cj["blueprint"], "garments": garments,
                      "wear_add": c.get("wear_add"), "face_covered": bool(c.get("face_covered")),
                      # a wardrobe variant's own outfit words and sheet (they replace the usual ones in the redraw)
                      "wear_set": c.get("wear_set"), "wear_back_set": c.get("wear_back_set"),
                      "worn_sheet": c.get("worn_sheet"),
                      "body_swap": cj.get("body_swap"), "hide_body": bool(cj.get("hide_body")),
                      "hide_face": bool(cj.get("hide_face")),       # a whole generated figure: no face mesh shown
                      "hide_hair": bool(cj.get("hide_hair")),       # the figure wears its own drawn hair
                      "rebuild_from": cj.get("rebuild_from"),
                      "garment_material": cj.get("garment_material"),
                      "loc_cm": layout.ue_pt(pos), "yaw": round(layout.yaw_from_facing(f), 3),
                      "facing": [f[0] / n, f[1] / n], "position": pos,
                      "body_pose": bpose, "face_pose": c.get("face_pose")}
        chars[key].update(carried(c))
        bp_deps = cj.get("bp_deps", [])
    set_proj = project_path(st.get("project", C.UE_PROJECT)).resolve()
    proj = proj or set_proj
    if set_proj != proj:
        raise SystemExit(f"set {spec['set']} was built in {set_proj}, the characters live in {proj}: rebuild it there "
                         f"(SETS_UE_PROJECT={proj} SETS_OUT={lib / 'set'} python tools/assets/sets/build_set.py "
                         f"<set spec>)")
    S = layout.Scene()
    layout.light_items(S, spec.get("lights", []))
    cams = [c for c in spec["cameras"] if not cams_wanted or c["name"] in cams_wanted]
    carried = {}                             # what people wear on the head or carry, by library prop id
    for c in cams:
        for ts in (c.get("carried") or {}).values():
            for t in ts:
                ap = lib / "prop" / t["asset"] / "asset.json"
                if t["asset"] not in carried and ap.exists():
                    aj = json.loads(ap.read_text())
                    carried[t["asset"]] = {"mesh": aj["ue"]["path"],    # (the model's measured box, when recorded)
                                           "size_m": (aj.get("blender") or {}).get("dims_m") or aj["size_m"]}
    # every thing in a person's hand goes to the pose (11-18: only a work action's first thing did, and anything
    # else hung under an open hand): pose.py closes the hand on each one's grip in any pose, puts a second thing in
    # the other hand, reaches a tool's working point to the work and puts the other hand on the handle of a
    # two-handed one. Each thing's hand is from_plan's (the first the working hand, a second the other; older specs
    # by order), its size the model's box; scene_job adds the grip measured from the model. A worked thing lying on
    # the work top ("at": "on") gives the blow's point: its top, i.e. where it lies plus its thinnest side (it lies
    # on its broadest face, scene_job), in UE cm (pose.py turns it into the body's own frame)
    for c in cams:
        for key, ts in (c.get("carried") or {}).items():
            bp = (chars.get(key) or {}).get("body_pose") or {}
            if not bp.get("pose"):
                continue
            pr = dict(bp.get("params") or {})
            held = list(pr.get("held") or [])
            for t in ts:
                if t.get("at") == "hand" and t["asset"] in carried and t["asset"] not in {h["asset"] for h in held} \
                        and len(held) < 2:
                    used = {h["hand"] for h in held}
                    hand = t.get("hand") if t.get("hand") in ("l", "r") and t.get("hand") not in used else \
                        next(x for x in ("r", "l") if x not in used)
                    held.append({"asset": t["asset"], "hand": hand, "size_m": list(carried[t["asset"]]["size_m"]),
                                 "name": t.get("name")})
            on = next((t for t in ts if t.get("at") == "on" and t.get("xyz") and t["asset"] in carried), None)
            if held:
                pr["held"] = held
            if pr.get("action") in WORK_POSE_ACTIONS + READY_POSE_ACTIONS:
                tool = next((h for h in held if h["hand"] == pr.get("hand", "r")), held[0] if held else None)
                if tool and "tool_size_m" not in pr:
                    pr.update(tool_size_m=tool["size_m"], tool_asset=tool["asset"])
                if on and "work_target_ue" not in pr and pr.get("action") in WORK_POSE_ACTIONS:
                    x, y, z = on["xyz"]
                    pr["work_target_ue"] = layout.ue_pt([x, y, z + min(carried[on["asset"]]["size_m"])])
                    # (the recorded size is not the model the engine lays down: scene_job re-aims at the top of the
                    # thing as laid, from its mesh's own box; 11-18, a blow aimed 7.75 cm over the laid top)
                    pr["work_on"] = {"asset": on["asset"], "base_ue": layout.ue_pt([x, y, z])}
            chars[key]["body_pose"] = dict(bp, params=pr)
    # a piece of the set worked with the hands themselves (from_plan's "work_at": the piece a work action stands them
    # at) with nothing worked lying on it: the hands that hold nothing take hold of a part of it (scene_job reads the
    # piece's surface round the person, pose.py finds a part a hand closes round and puts the hands on it)
    for key, c in spec["characters"].items():
        bp = (chars.get(key) or {}).get("body_pose") or {}
        if bp.get("pose"):
            chars[key]["body_pose"] = dict(bp, params=aim_params(fixture_params(c, bp.get("params") or {})))
    kl = key_light(lib / "set" / spec["set"] / "spec.json", spec.get("exposure_ev100"))
    return proj, {**({"key_light": kl} if kl else {}), "carried": carried, "id": spec["id"], "level": st["level"], "res": spec.get("res", [1152, 640]),
                  "char_tools": str(CHAR_TOOLS), "set_tools": str(SET_TOOLS), "bp_deps": bp_deps,
                  "characters": chars, "figures": figs, "cameras": cams, "lights": S.lights,
                  "figure_tools": str(HERE.parent / "img3d" / "ue_stage_figure.py"),
                  "exposure_ev100": spec.get("exposure_ev100"), "warm_frames": spec.get("warm_frames", 150),
                  "settle_frames": spec.get("settle_frames", 60), "attach_tol_cm": spec.get("attach_tol_cm", 0.1),
                  "guides": bool(spec.get("guides", False)), "hide": list(spec.get("hide") or []),
                  "interior": st.get("interior"), "room": st.get("size_m")}


def contact_sheet(shots, out):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    ims = [(Image.open(s["path"]).convert("RGB"), s["name"]) for s in shots if os.path.exists(s["path"])]
    if not ims:
        return None
    tw, th = 576, 320
    cols = 2 if len(ims) > 1 else 1
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols - 1) * 6, rows * (th + 22)), (20, 20, 20))
    d = ImageDraw.Draw(sheet)
    for i, (im, name) in enumerate(ims):
        x, y = (i % cols) * (tw + 6), (i // cols) * (th + 22)
        sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y + 22))
        d.text((x + 4, y + 5), name, fill=(235, 235, 230))
    sheet.save(out)
    return str(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--cameras", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800)
    a = ap.parse_args()
    t_all = time.time()
    spec = json.loads(Path(a.spec).read_text())
    proj, job = resolve(spec, [c for c in a.cameras.split(",") if c] or None)
    odir = out_root() / spec["id"]
    (odir / "sched").mkdir(parents=True, exist_ok=True)
    job["out"] = str(odir)
    job["result"] = str(odir / "result.json")
    wrote = ensure_raytracing(proj)
    if wrote:
        log(f"wrote ray-tracing settings into {proj.parent / 'Config' / 'DefaultEngine.ini'}: {wrote} "
            f"(the next editor start compiles shaders)")
    outs = [str(odir / f"{c['name']}.png") for c in job["cameras"]] + [job["result"]]
    sj = {"project": str(proj), "script": str(HERE / "ue" / "scene_job.py"), "job": job, "outputs": outs,
          "timeout_s": a.timeout, "id": f"scene_{spec['id']}", "log_dir": str(odir / "sched")}
    jp = odir / "job.json"
    jp.write_text(json.dumps(sj, indent=1))
    if a.dry_run:
        print(json.dumps(sj, indent=1))
        return 0
    for o in outs:
        Path(o).unlink(missing_ok=True)
    t0 = time.time()
    client = line_client()
    p = subprocess.run([sys.executable, str(client), "--ue-job", str(jp)],
                       capture_output=True, text=True, cwd=str(client.parents[2]))
    wall = round(time.time() - t0, 1)
    (odir / "sched" / "client.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
    res = json.loads(Path(job["result"]).read_text()) if Path(job["result"]).exists() else {}
    sched = {}
    sjp = odir / "sched" / "logs" / "sched.json"
    if sjp.exists():
        st = json.loads(sjp.read_text())
        sched = {"queue_wait_s": round(sum(s.get("wait_s") or 0 for s in st.get("stages", [])), 1),
                 "editor_run_s": round(sum(s.get("run_s") or 0 for s in st.get("stages", [])), 1)}
    times = {"client_wall_s": wall, **sched, "editor": res.get("times_s"),
             "stills": {s["name"]: s.get("still_s") for s in res.get("shots", [])},
             "total_s": round(time.time() - t_all, 1)}
    (odir / "times.json").write_text(json.dumps(times, indent=1))
    sheet = contact_sheet(res.get("shots", []), odir / "sheet.png") if res.get("shots") else None
    log(f"{spec['id']}: rc={p.returncode} ok={res.get('ok')} wall {wall} s; {json.dumps(times.get('editor'))}")
    for s in res.get("shots", []):
        fc = s.get("frame_check") or {}
        bad = [k for k, v in s.get("attach", {}).items() if not v.get("ok")]
        log(f"  {s['name']}: {s['path']}  problems={fc.get('problems')}  head-detached={bad or 'no'}"
            + (f"  REJECT: {s['reject']}" if s.get("reject") else ""))
    if sheet:
        log(f"  sheet {sheet}")
    if p.returncode or not res.get("ok"):
        log(f"error: {res.get('error') or p.stderr[-1500:] or p.stdout[-1500:]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
