"""10-28 integration test, on the GPUs: a 15 s window rendered end to end by bookcaster, then checked for every
failure that has cost a night. The user: "an integration test case that builds like a 15 second clip and covers
most of the basic fail cases" (the night of 10-27: EMFILE at 1024 shm files, no talk stitch, yesterday's joins
assembled, a render-side fix that re-planned and re-rendered 3 h, close-ups failing all 5 seeds).

  itest_15s.py [BOOK] [--chapter 9] [--span 0:600] [--at 288.4] [--len 15] [--warm]

Run 1 (fresh: the window's keys and clips are set aside, so everything in it is drawn and rendered again; --warm
keeps them) must finish clean; run 2, the same command at once, must reuse everything: no director call, no key
drawn, no clip rendered. About 20-30 min fresh, a few minutes warm. Exit 1 on any failure."""
import argparse, json, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))

ap = argparse.ArgumentParser()
ap.add_argument("book", nargs="?", default=str(Path.home() / "audiobooks" / "sherlock_adventures"))
ap.add_argument("--chapter", default="9")
ap.add_argument("--span", default="0:600", help="the stretch the director plans (its cache makes this cheap)")
ap.add_argument("--at", type=float, default=288.4, help="window start: 4:48 of Speckled Band has action, quiet, "
                "close-ups, a two-shot and a pieced talk line")
ap.add_argument("--len", type=float, default=15.0)
ap.add_argument("--warm", action="store_true", help="keep the window's keys and clips (skip the fresh run)")
a = ap.parse_args()
book = Path(a.book).expanduser()
UNIT = f"bookcaster-{book.name}.service"
LOG = book / "work" / "job.log"
t0, t1 = a.at, a.at + a.len
s0, s1 = (float(x) for x in a.span.split(":"))

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"), flush=True)
    ok or fails.append(name)

def active():
    return subprocess.run(["systemctl", "--user", "is-active", "-q", UNIT]).returncode == 0

def ffdur(f, stream=None):
    cmd = ["ffprobe", "-v", "error"] + (["-select_streams", stream, "-show_entries", "stream=duration"] if stream
                                        else ["-show_entries", "format=duration"]) + ["-of", "csv=p=0", str(f)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0]) if out and out[0] not in ("", "N/A") else None

if active():
    sys.exit(f"{UNIT} is running: one job at a time")
import shot_flow as S
from tts_reader import bookcaster as BC
stem = next(p.stem.replace(".timeline", "") for p in (book / "work").glob(f"{int(a.chapter):02d}_*.timeline.json")
            if ".raw." not in p.name)
sd = S.shot_dir(book, stem)
L = sd / "ltx"


def window_shots(since):
    """The window's shots from the jobs THIS run wrote (the render writes only the window's; older files are another
    run's - 10-28: the first try read the stopped full run's)."""
    return {k: json.loads((L / f"{k}_jobs.json").read_text()) for k in ("talk", "two", "quiet")
            if (L / f"{k}_jobs.json").exists() and (L / f"{k}_jobs.json").stat().st_mtime >= since}


def run(label):
    n0 = sum(1 for _ in LOG.open()) if LOG.exists() else 0
    start = time.time()
    cmd = ["bookcaster", "video", str(book), "--flow", "shots", "--chapter", a.chapter, "--from", f"{s0:g}",
           "--to", f"{s1:g}", "--render", f"{t0:g}-{t1:g}"]
    print(f"--- {label}: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=book)
    time.sleep(5)
    while active():
        time.sleep(20)
    lines = LOG.read_text(errors="replace").splitlines()[n0:]
    st = json.loads((book / "work" / "job.json").read_text())
    print(f"    {label}: {time.time() - start:.0f} s, {st.get('state')}", flush=True)
    return start, lines, st


# ---- run 1: fresh ----
if not a.warm:
    # the window's shots: the jobs the last render of this window wrote, else a stopped render's saved plan
    ids = set()
    for k in ("talk", "two", "quiet"):
        f = L / f"{k}_jobs.json"
        js = json.loads(f.read_text()) if f.exists() else []
        if len(js) <= 40:                       # a window's, not a whole stretch's
            ids |= {re.sub(r"(_c\d+|_t\d+)$", "", j["name"]) for j in js}
    plan = next(iter(sorted(sd.glob("plan_*.json"), key=lambda p: -p.stat().st_mtime)), None)
    if not ids and plan:
        pj = json.loads(plan.read_text())
        ids = {s_["id"] for e in pj.get("_es") or [] if t0 - 1e-6 <= float(e["t"]) < t1 for s_ in pj.get(e["id"], [])}
    aside = sd / "_itest_aside" / time.strftime("%m%d-%H%M%S")
    moved = 0
    for sid in sorted(ids):
        d = sd / sid.rsplit("_s", 1)[0] / sid
        if d.is_dir():
            (aside / d.parent.name).mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(aside / d.parent.name / d.name))
            moved += 1
        for c in (L / "out").glob(f"{sid}*"):
            (aside / "ltx").mkdir(parents=True, exist_ok=True)
            shutil.move(str(c), str(aside / "ltx" / c.name))
    print(f"    fresh: {moved} shot(s) set aside in {aside}" if moved else
          "    fresh: nothing of the window's found to set aside", flush=True)

start1, lines1, st1 = run("run 1")
txt1 = "\n".join(lines1)
check("job finished (no failures)", st1.get("state") in ("done", "done with problems"), st1.get("step", "")[-300:])
if st1.get("state") == "failed":
    print(json.dumps({"fails": fails}))
    sys.exit(1)
check("no traceback in the job log", "Traceback" not in txt1)
check("no file-limit error (EMFILE)", not list(L.glob("log_gen_*.txt")) or not any(
    "Too many open files" in f.read_text(errors="replace") for f in L.glob("log_gen_*.txt")
    if f.stat().st_mtime >= start1))
def in_window(x):
    m = re.search(r"passage ([\d.]+)-([\d.]+) s", x)
    return not m or (float(m.group(1)) < t1 and float(m.group(2)) > t0)
bang = [x for x in lines1 if x.startswith("!!!") and "passage" not in x]
check("no '!!!' line (no held key, no black, no unmade clip, no failed key)", not bang, bang[:6])
rules = [x for x in lines1 if x.startswith("!!!") and "passage" in x and in_window(x)]
check("the director's shots in the window keep the rules", not rules, rules[:4])
check("every keyframe passed on some seed", not any("no clean picture" in x for x in lines1),
      [x for x in lines1 if "no clean picture" in x][:4])

jobs = window_shots(start1)
alljobs = [j for v in jobs.values() for j in v]
check("the window has talk and quiet shots", all(jobs.get(k) for k in ("talk", "quiet")),
      {k: len(v) for k, v in jobs.items()})
if not jobs.get("two"):
    print("NOTE the director gave this window no action shot: the action render is not exercised", flush=True)
names = {j["name"].rsplit("_c", 1)[0] for j in alljobs if not re.search(r"_t\d+$", j["name"])}
missing = [n for n in names if not (L / "out" / f"{n}.mp4").exists()]
check("every shot has its clip", not missing, missing[:6])
pieced = {}
for j in jobs.get("talk", []):
    if len(j.get("chunk") or []) == 4 and j["chunk"][1] > 1:
        pieced.setdefault(j["name"].rsplit("_c", 1)[0], []).append(j)
check("the window has a pieced talk line (the stitch is exercised)", bool(pieced))
for n, cs in pieced.items():
    f = L / "out" / f"{n}.mp4"
    want = max(j["chunk"][2] + j["chunk"][3] for j in cs) / 24
    pieces_t = max([(L / "out" / f"{j['name']}.mp4").stat().st_mtime for j in cs
                    if (L / "out" / f"{j['name']}.mp4").exists()] or [float("inf")])
    check(f"{n}: stitched from tonight's pieces, full length",
          f.exists() and f.stat().st_mtime >= pieces_t and abs((ffdur(f, "v:0") or 0) - want) < 0.1,
          (f.exists(), round(ffdur(f, "v:0") or 0, 2), round(want, 2)))
stale = []
for j in alljobs:
    c, rj = L / "out" / f"{j['name']}.mp4", L / "out" / f"{j['name']}.json"
    key = Path(j["images"][0]["path"]) if j.get("images") else None
    if c.exists() and key and key.exists() and c.stat().st_mtime < key.stat().st_mtime:
        stale.append((j["name"], "older than its key"))
    if c.exists() and rj.exists():
        old = json.loads(rj.read_text())
        for k in ("prompt", "seed", "frames", "audio_src"):
            if old.get(k) != j.get(k):
                stale.append((j["name"], k))
check("no clip from another key, prompt, seed, length or voice cut", not stale, stale[:6])

import numpy as np
still = []
for j in jobs.get("talk", []):
    n = j["name"].rsplit("_c", 1)[0]
    f = L / "out" / f"{n}.mp4"
    if not f.exists() or n in [s[0] for s in still]:
        continue
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(f), "-vf", "scale=128:72,format=gray", "-f", "rawvideo",
                          "-"], capture_output=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, 72, 128).astype(np.float32)
    if len(fr) > 2 and float(np.median(np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2)))) < 0.15:
        still.append((n, len(fr)))
check("no talk clip is a still picture", not still, still)

outs = sorted(book.glob("out/*_shots.mp4"), key=lambda p: p.stat().st_mtime)
out = outs[-1] if outs and outs[-1].stat().st_mtime >= start1 else None
check("the window's video was assembled", out is not None)
if out:
    vd_, ad_ = ffdur(out, "v:0"), ffdur(out, "a:0")
    check(f"video covers the window ({out.name}; it ends on a shot's end, so it may run past)",
          vd_ is not None and vd_ >= a.len - 0.5, vd_)
    check("picture and sound the same length (no drift)", vd_ and ad_ and abs(vd_ - ad_) < 0.1, (vd_, ad_))
    pace = (book / "pace.txt").read_text().strip() if (book / "pace.txt").exists() else str(BC.DEFAULT_PACE)
    check("pace is 1.25", abs(float(pace) - 1.25) < 1e-6, pace)

# ---- run 2: the same command again reuses everything ----
start2, lines2, st2 = run("run 2 (warm)")
asked = [x for x in lines2 if "prompt's tokens cached" in x]
check("run 2: the director is not asked (its answers replay)", not asked, asked[:3])
check("run 2: no saved-plan rejection", not any("planning again" in x for x in lines2))
newkeys = [p for p in sd.glob("*-*/*_s*/key.png") if p.stat().st_mtime >= start2]
check("run 2: no keyframe drawn again", not newkeys, [p.parent.name for p in newkeys][:6])
newclips = [p.name for p in (L / "out").glob("*.mp4") if p.stat().st_mtime >= start2
            and not (p.stem in pieced)]                 # joins are re-made from the pieces each run (seconds)
check("run 2: no clip rendered again", not newclips, newclips[:6])
check("run 2: finished clean", st2.get("state") == "done", st2.get("step", "")[-300:])

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
