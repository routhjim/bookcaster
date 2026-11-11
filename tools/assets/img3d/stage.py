"""Stage library figures that are plain meshes (picture-to-3D statues, or their rigged copies holding a pose) in a
library set and photograph them with the scene tool's cameras and guide passes, through the line's engine.

    python tools/assets/img3d/stage.py SCENE.json [--out DIR] [--dry-run]

SCENE.json is the scene tool's spec (scenes/README.md: "set", "characters", "cameras", "exposure_ev100",
"guides", "res") with each character naming a library ASSET instead of a MetaHuman character:
    "characters": {"a": {"asset": "<class>/<id>" | {"mesh": UE path}, "anim": UE AnimSequence path (optional),
                         "t": s, "position": [x, y], "facing": [dx, dy]}}
The asset's static mesh comes from its asset.json ("ue.path"); a rigged copy is named by "mesh". Writes
OUT/<id>/<camera>.png, guides/<camera>_depth.exr and _depth_<key>.exr (with "guides": true), result.json (per
still: camera, frame check, truth = body angle to the camera), job.json and times.json.
Body pose names of the scene tool ("body_pose") do not apply: a figure has no MetaHuman skeleton.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "sets"))
import config as C  # noqa: E402
import layout  # noqa: E402


def resolve(spec, out):
    lib = Path(C.LIBRARY)
    st = json.loads((lib / "set" / spec["set"] / "asset.json").read_text())
    figs = {}
    for key, c in spec["characters"].items():
        a = c["asset"]
        if isinstance(a, dict):
            mesh, h = a["mesh"], a.get("height_m")
        else:
            aj = json.loads((lib / a / "asset.json").read_text())
            mesh, h = aj["ue"]["path"], aj["size_m"][2]
        pos = list(c["position"]) + ([0.0] if len(c["position"]) == 2 else [])
        f = c.get("facing", [0, -1])
        n = (f[0] ** 2 + f[1] ** 2) ** 0.5
        figs[key] = {"mesh": mesh, "anim": c.get("anim"), "t": c.get("t", 0.0), "loc_cm": layout.ue_pt(pos),
                     "yaw": round(layout.yaw_from_facing(f), 3), "facing": [f[0] / n, f[1] / n], "position": pos,
                     "height_m": h}
    return {"id": spec["id"], "level": st["level"], "res": spec.get("res", [1152, 640]), "out": str(out),
            "result": str(out / "result.json"), "framing": str(HERE.parent / "scenes" / "framing.py"),
            "figures": figs, "cameras": spec["cameras"], "exposure_ev100": spec.get("exposure_ev100"),
            "guides": bool(spec.get("guides")), "warm_frames": spec.get("warm_frames", 150),
            "settle_frames": spec.get("settle_frames", 60)}, st.get("project", str(C.UE_PROJECT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", default=str(C.UE_HOME / "lab" / "scenes" / "out"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    out = Path(a.out) / spec["id"]
    (out / "sched").mkdir(parents=True, exist_ok=True)
    job, proj = resolve(spec, out)
    outs = [str(out / f"{c['name']}.png") for c in job["cameras"]] + [job["result"]]
    sj = {"project": str(Path(proj).resolve()) if Path(proj).exists() else str(C.UE_PROJECT),
          "script": str(HERE / "ue_stage_figure.py"), "job": job, "outputs": outs, "timeout_s": a.timeout,
          "id": f"stage_{spec['id']}", "log_dir": str(out / "sched")}
    (out / "job.json").write_text(json.dumps(sj, indent=1))
    if a.dry_run:
        print(json.dumps(sj, indent=1))
        return 0
    for o in outs:
        Path(o).unlink(missing_ok=True)
    t0 = time.time()
    p = subprocess.run([sys.executable, str(HERE.parent / "run_asset.py"), "--ue-job", str(out / "job.json")],
                       capture_output=True, text=True)
    (out / "sched" / "client.log").write_text(p.stdout + "\n--- stderr\n" + p.stderr)
    res = json.loads(Path(job["result"]).read_text()) if Path(job["result"]).exists() else {}
    (out / "times.json").write_text(json.dumps({"client_wall_s": round(time.time() - t0, 1),
                                                "editor": res.get("times_s")}, indent=1))
    print(f"[stage] {spec['id']}: rc={p.returncode} ok={res.get('ok')} {round(time.time() - t0, 1)} s")
    for s in res.get("shots", []):
        print(f"  {s['name']}: {s['path']} problems={(s.get('frame_check') or {}).get('problems')} truth={s['truth']}")
    if p.returncode or not res.get("ok"):
        print(res.get("error") or p.stdout[-1500:] + p.stderr[-1500:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
