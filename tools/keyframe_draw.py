"""Draw a keyframe for a staged shot and keep the first seed that passes the vision check (any shot).

    HIP_VISIBLE_DEVICES=$(~/bin/hip-index 05) flock /tmp/gpu-05.lock \\
    ~/tts-ab/.venv-pose/bin/python -I tools/keyframe_draw.py STILLS_DIR --frame start|end \\
        --prompt "..." --refs a.png b.png --out /abs/keyframe.png [--seeds 4] [--strength 0.85] \\
        [--depth 1.3] [--edges 0.6] [--seed0 7] [--lora /abs/char.safetensors[=1.0] ...]

STILLS_DIR is a tools/blockout_video.py --stills output (<frame>_color.png, <frame>_depth.png,
<frame>_expect.json). One Qwen-Image load draws N seeds (img2img from the colour still, depth + depth-edge
controls, the given references); tools/keyframe_check.py checks each in seed order (figures present, hands
on their handles); the first pass is copied to --out. Writes <out stem>_draw.json with every seed's result.
Exit 0 on a pass, 2 if no seed passed (the best seed by fewest failures is still written, as <out stem>_best.png).
Edges default 0.6, lowered to 0.3 when the frame has fx (sparks would be lost).

LIST MODE (many shots, few Qwen loads):
    HIP_VISIBLE_DEVICES=$(~/bin/hip-index 05) ~/tts-ab/.venv-pose/bin/python -I tools/keyframe_draw.py \
        --list shots.json --lock /tmp/gpu-05.lock [--chunk-min 12]
shots.json = [{"stills", "frame", "prompt", "refs", "out", "seeds", "seed0", "strength", "depth", "edges", "fx",
"pose", "likeness": ["W=sheet"], "lora": ["path=scale"]}, ...] (same meanings/defaults as the flags). All seeds go
to qwen_image in chunks of ~chunk-min GPU minutes; with --lock the tool takes the lock (after yielding to a waiting
LTX gaprun and waiting for MemAvailable > 15 GB) per chunk and RELEASES it between chunks -- so do not wrap the
process in flock. Re-running resumes (drawn seeds are kept). Output per shot as in single mode, plus
<list stem>_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
QWEN_PY = os.environ.get("QWEN_IMAGE_PY", os.path.expanduser("~/tts-ab/.venv-qwenimage/bin/python"))
POSE_PY = os.environ.get("POSE_PY", os.path.expanduser("~/tts-ab/.venv-pose/bin/python"))


def _lora_spec(l: str) -> list:
    """'PATH[=SCALE][@x0,y0,x1,y1]' -> [path, scale] or [path, scale, box] (0-1 box: a REGIONAL character LoRA,
    qwen_image applies it only there; needed with several people in the frame, 10-20 remake agent)."""
    l, _, box = l.partition("@")
    path, _, sc = l.partition("=")
    spec = [os.path.abspath(path), float(sc) if sc else 1.0]
    if box:
        spec.append([float(x) for x in box.split(",")])
    return spec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stills")
    ap.add_argument("--frame", default="start")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--refs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=7)
    ap.add_argument("--strength", type=float, default=0.85)
    ap.add_argument("--depth", type=float, default=1.3)
    ap.add_argument("--edges", type=float, default=None)
    ap.add_argument("--fx", action="store_true", help="the frame shows fx (sparks): edges 0.3")
    ap.add_argument("--pose", type=float, default=0.0, help="add the OpenPose skeleton as a third control")
    ap.add_argument("--lora", action="append", default=[], metavar="PATH[=SCALE][@x0,y0,x1,y1]",
                    help="character LoRA(s) (tools/charlora.py), applied per job; put its trigger in --prompt")
    ap.add_argument("--likeness", action="append", default=[], metavar="WHO=REF",
                    help="rank passing seeds by likeness to this character reference (outfit sheet)")
    ap.add_argument("--list", default=None, help="LIST.json: [{stills, frame, prompt, refs, out, seeds, seed0, strength, "
                    "depth, edges, fx, pose, likeness, lora}] -> ONE Qwen load per GPU chunk for every seed of every shot")
    ap.add_argument("--lock", default=None, help="list mode: take this flock per GPU chunk (release between chunks); "
                    "do NOT also wrap the process in flock")
    ap.add_argument("--chunk-min", type=float, default=12.0, help="list mode: GPU minutes per lock hold")
    if argv is None:
        argv = sys.argv[1:]
    if "--list" in argv and not any(not x.startswith("-") for x in argv[:1]):
        argv = ["_"] + list(argv)                     # stills positional is unused in list mode
    if "--list" in argv and "--prompt" not in argv:
        argv = list(argv) + ["--prompt", "_", "--out", "_"]
    a = ap.parse_args(argv)
    if a.list:
        return run_list(a)
    spec = {k: getattr(a, k) for k in ("stills", "frame", "prompt", "refs", "out", "seeds", "seed0", "strength",
                                       "depth", "edges", "fx", "pose", "lora", "likeness")}
    jobs, ctx = build(spec)
    jp = os.path.join(ctx["wdir"], "jobs.json")
    json.dump(jobs, open(jp, "w"), indent=1)
    r = subprocess.run([QWEN_PY, "-I", os.path.join(HERE, "qwen_image.py"), jp, ctx["wdir"]], capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("qwen_image failed")
    rep = finish(spec, jobs, ctx)
    print(json.dumps({"out": rep["out"], "chosen_seed": rep["chosen_seed"],
                      "passed": [x["seed"] for x in rep["results"] if x["pass"]]}))
    return 0 if rep["out"] else 2


_DEF = {"frame": "start", "refs": [], "seeds": 4, "seed0": 7, "strength": 0.85, "depth": 1.3, "edges": None,
        "fx": False, "pose": 0.0, "lora": [], "likeness": [], "mask": None, "strength_bg": None}


def build(spec):
    """One shot's Qwen jobs (N seeds) + the context finish() needs."""
    a = dict(_DEF, **{k: v for k, v in spec.items() if v is not None})
    for k in ("lora", "likeness", "refs"):
        if isinstance(a[k], str):
            a[k] = [a[k]]
    sd = os.path.abspath(a["stills"])
    f = a["frame"]
    color, depth, expect = (os.path.join(sd, f"{f}_{x}") for x in ("color.png", "depth.png", "expect.json"))
    for p in (color, depth, expect):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing {p} (run blockout_video.py --stills)")
    edges = os.path.join(sd, f"{f}_edges.png")
    if not os.path.exists(edges):
        import blockout as B
        B.edges_png(depth, edges)
    W, H = json.load(open(expect))["size"]
    es = a["edges"] if a["edges"] is not None else (0.3 if a["fx"] else 0.6)
    out = os.path.abspath(a["out"])
    stem = os.path.splitext(out)[0]
    wdir = stem + "_seeds"
    os.makedirs(wdir, exist_ok=True)
    jobs = []
    for k in range(int(a["seeds"])):
        seed = int(a["seed0"]) + 4 * k
        jobs.append({"id": f"s{seed}", "prompt": a["prompt"], "seed": seed, "size": [W, H],
                     "refs": [os.path.abspath(r) for r in a["refs"]], "path": os.path.join(wdir, f"s{seed}.png"),
                     "control": [{"image": depth, "type": "depth", "scale": a["depth"]},
                                 {"image": edges, "type": "canny", "scale": es, "prepared": True}] +
                                ([{"image": os.path.join(sd, f"{f}_pose.png"), "type": "pose", "scale": a["pose"]}]
                                 if a["pose"] > 0 else []),
                     "init": {"image": color, "strength": a["strength"],
                              **({"mask": a["mask"], "strength_bg": a["strength_bg"]} if a.get("mask") and a.get("strength_bg") else {})}})
        if a["lora"]:
            jobs[-1]["loras"] = [_lora_spec(l) for l in a["lora"]]
    return jobs, {"a": a, "color": color, "expect": expect, "out": out, "stem": stem, "wdir": wdir, "edges_scale": es}


def finish(spec, jobs, ctx):
    """Checks (keyframe_check), likeness ranking, copy the chosen seed; writes <out stem>_draw.json."""
    a, color, expect, out, stem = ctx["a"], ctx["color"], ctx["expect"], ctx["out"], ctx["stem"]
    results, best = [], None
    for j in jobs:
        if not os.path.exists(j["path"]):
            results.append({"seed": j["seed"], "path": j["path"], "pass": False, "fail": ["not drawn"], "unknown": []})
            continue
        cj = os.path.join(ctx["wdir"], f"{j['id']}_check.json")
        subprocess.run([POSE_PY, "-I", os.path.join(HERE, "keyframe_check.py"), j["path"], expect, "--json", cj],
                       capture_output=True, text=True)
        c = json.load(open(cj)) if os.path.exists(cj) else {"pass": False, "fail": ["check crashed"], "unknown": []}
        res = {"seed": j["seed"], "path": j["path"], "pass": bool(c["pass"]) and not c.get("unknown"),
               "fail": c["fail"], "unknown": c.get("unknown", [])}
        results.append(res)
        sys.stderr.write(f"{os.path.basename(out)} seed {j['seed']}: "
                         f"{'PASS' if res['pass'] else 'FAIL ' + '; '.join(res['fail'] + res['unknown'])}\n")
        nbad = len(res["fail"]) + 0.5 * len(res["unknown"])
        if best is None or nbad < best[0]:
            best = (nbad, res)
    # rank: pass first, then likeness (DINOv2 crop vs the outfit sheet) + hot-prop visibility
    if a["likeness"]:
        HY = os.environ.get("LIKENESS_PY", os.path.expanduser("~/tts-ab/.venv-hy3d/bin/python"))
        for res in results:
            if not os.path.exists(res["path"]):
                continue
            lj = res["path"].replace(".png", "_likeness.json")
            env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="")
            cmd = [HY, "-I", os.path.join(HERE, "keyframe_likeness.py"), res["path"], expect, "--still", color, "--json", lj]
            for l_ in a["likeness"]:
                cmd += ["--ref", l_]
            subprocess.run(cmd, capture_output=True, text=True, env=env)
            L = json.load(open(lj)) if os.path.exists(lj) else {"likeness": {}, "hot": None}
            res["likeness"], res["hot"] = L["likeness"], L["hot"]
            res["score"] = (sum(L["likeness"].values()) / max(len(L["likeness"]), 1)) + 1.0 * (L["hot"] or 0.0)
            res["hot_ok"] = L["hot"] is None or L["hot"] >= 0.25        # a glowing prop must survive
            sys.stderr.write(f"seed {res['seed']}: likeness {L['likeness']} hot {L['hot']} score {res['score']:.3f}\n")
        results.sort(key=lambda x: (not x["pass"], not x.get("hot_ok", True), -x.get("score", 0.0)))
    chosen = next((x for x in results if x["pass"]), None)
    import shutil
    if chosen:
        shutil.copy(chosen["path"], out)
    elif best is not None:
        shutil.copy(best[1]["path"], stem + "_best.png")
    rep = {"out": out if chosen else None, "chosen_seed": chosen["seed"] if chosen else None, "results": results,
           "controls": {"depth": a["depth"], "edges": ctx["edges_scale"], "strength": a["strength"]}}
    json.dump(rep, open(stem + "_draw.json", "w"), indent=1)
    return rep


def _gpu_turn(lock_path):
    """Fairness + safety before a GPU chunk: yield to a waiting LTX two-card run (gaprun.sh: let it start and
    finish, <= 15 min), take the flock, wait for MemAvailable > 15 GB. Returns the open lock fd."""
    import fcntl
    import time

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

    def ltx_running():
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
            if ltx_running():
                started = True
            elif started or not pg("gaprun.sh"):
                break
            time.sleep(10)
    fd = open(lock_path, "w")
    fcntl.flock(fd, fcntl.LOCK_EX)

    def avail_gb():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 1048576
        return 0
    while avail_gb() < 15:
        time.sleep(10)
    return fd


def run_list(a):
    """LIST mode: every seed of every shot through as few Qwen loads as the lock rule allows (chunks of
    ~--chunk-min GPU minutes, sized from the measured s/image; the lock is released between chunks while the
    CPU checks run), then per-shot checks + ranking exactly as single mode. Writes <list stem>_report.json."""
    import time
    specs = json.load(open(a.list))
    built, all_jobs = [], []
    for i, sp in enumerate(specs):
        try:
            jobs, ctx = build(sp)
        except Exception as e:                      # noqa: BLE001 - one bad shot never stops the list
            built.append((sp, None, None, f"{type(e).__name__}: {e}"))
            continue
        for j in jobs:
            j["id"] = f"k{i:03d}_{j['id']}"
        built.append((sp, jobs, ctx, None))
        all_jobs += [j for j in jobs if not os.path.exists(j["path"])]
    wdir = os.path.splitext(os.path.abspath(a.list))[0] + "_qwen"
    os.makedirs(wdir, exist_ok=True)
    per_img, load_s, budget = 25.0, 75.0, a.chunk_min * 60
    k = 0
    while k < len(all_jobs):
        n = max(1, int((budget - load_s) / per_img))
        chunk = all_jobs[k:k + n]
        jp = os.path.join(wdir, f"jobs_{k:04d}.json")
        json.dump(chunk, open(jp, "w"), indent=1)
        fd = _gpu_turn(a.lock) if a.lock else None
        t0 = time.time()
        try:
            r = subprocess.run([QWEN_PY, "-I", os.path.join(HERE, "qwen_image.py"), jp, wdir], capture_output=True,
                               text=True, timeout=budget + 180)
            if r.returncode:
                sys.stderr.write(r.stderr[-2000:])
            errs = [ln for ln in r.stdout.splitlines() if '"error"' in ln]   # per-job errors (rc 0): never silent
            if errs:
                sys.stderr.write(f"{len(errs)} job error(s), first: {errs[0][:300]}\n")
        except subprocess.TimeoutExpired:
            sys.stderr.write(f"chunk {k} hit the time cap; the rest of it goes into the next chunk\n")
        finally:
            if fd is not None:
                fd.close()                             # release the GPU between chunks
        done = [j for j in chunk if os.path.exists(j["path"])]
        dt = time.time() - t0
        if done:
            per_img = max(5.0, (dt - load_s) / len(done))
        sys.stderr.write(f"chunk {k}: {len(done)}/{len(chunk)} drawn in {dt:.0f}s ({per_img:.1f} s/image)\n")
        if not done:
            sys.stderr.write("no progress in a chunk: stopping\n")
            break
        k += len(chunk) if len(done) == len(chunk) else 0
        all_jobs = all_jobs[:k] + [j for j in all_jobs[k:] if not os.path.exists(j["path"])]
    report = []
    for sp, jobs, ctx, err in built:
        if err:
            report.append({"out": sp.get("out"), "error": err})
            continue
        rep = finish(sp, jobs, ctx)
        report.append({"out": sp.get("out"), "chosen": rep["out"], "chosen_seed": rep["chosen_seed"],
                       "passed": [x["seed"] for x in rep["results"] if x["pass"]]})
    rp = os.path.splitext(os.path.abspath(a.list))[0] + "_report.json"
    json.dump(report, open(rp, "w"), indent=1)
    print(json.dumps({"report": rp, "shots": len(report), "passed": sum(1 for r in report if r.get("chosen")),
                      "errors": sum(1 for r in report if r.get("error"))}))
    return 0 if all(r.get("chosen") for r in report) else 2


if __name__ == "__main__":
    sys.exit(main())
