"""10-26: every LTX gen gets the measured-best two-XTX set from remake_chapter.ltx_step, whatever flow calls it.
CPU only: builds the runner args for each route (no GPU, no subprocess except pp_fit on a fixture timing file).
    ~/tts_reader/.venv/bin/python3 tools/tests/test_ltx_defaults_1026.py"""
import json, os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for k in ("REMAKE_LTX_PP", "REMAKE_LTX_PP_ARGS", "REMAKE_LTX_SPLIT_TALK", "REMAKE_LTX_SPLIT_ACT"):
    os.environ.pop(k, None)
import remake_chapter as R
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

def rec(route, split, a_dit, b_dit, a_other, b_dec=None):
    r = {"job": "x", "route": route, "pp": True, "lanes": 2, "split": split, "wall_s": 40.0, "frames": 57,
         "a_dit_s": a_dit, "a_copy_s": 1.0,
         "b_dit_s": b_dit, "b_copy_s": 0.5, "a_other_s": a_other}
    if b_dec is not None:
        r["b_decode_s"] = b_dec
    return r

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "work" / "video"
    old, new = book / "01_chapter_1" / "shots", book / "02_chapter_2" / "shots"
    fresh = book / "03_chapter_3" / "shots"
    for d in (old, new, fresh):
        (d / "ltx" / "out").mkdir(parents=True)
    # old chapter: a big sample that balances at a low split; newest chapter: decode on B moves it up
    (old / "ltx/out/pp_timing.jsonl").write_text("\n".join(json.dumps(rec("distilled", 8, 4.3, 16.5, 12.0, 0)) for _ in range(40)))
    (new / "ltx/out/pp_timing.jsonl").write_text("\n".join(json.dumps(rec("distilled", 17, 7.9, 12.0, 4.5, 7.0)) for _ in range(5)))
    os.utime(old / "ltx/out/pp_timing.jsonl", (1, 1))
    cx = SimpleNamespace(OUT=fresh)

    def a(args):
        return R._gen_args(cx, args)
    q = a(["gen", "quiet_jobs.json", "--out", "out"])
    for f in ("--pp", "--lanes", "--pp-shared-a", "--stagger", "--pp-pool", "--fa-amd", "--blaslt", "--decode-b", "--split"):
        check(f"quiet gets {f}", f in q, q)
    check("quiet split from the NEWEST chapter (counts card-B decode)", int(q[q.index("--split") + 1]) > 8, q)
    check("quiet stagger = half its clip time from the newest chapter (wall 40 s -> 20)",
          q[q.index("--stagger") + 1] == "20", q)
    t = a(["gen", "talk_jobs.json", "--a2vid"])
    check("talk: --decode-b too (user 10-26: every clip type)", "--decode-b" in t, t)
    check("talk: default a2vid split 25 with no timing", t[t.index("--split") + 1] == "25", t)
    check("talk: pool + default a2vid stagger 100 (no talk timing yet)", "--pp-pool" in t and
          t[t.index("--stagger") + 1] == "100", t)
    w = a(["gen", "two_jobs.json", "--ti2v", "--fa-amd"])
    check("ti2v: --decode-b once", w.count("--decode-b") == 1, w)
    check("quiet with the flow's own --decode-b: once", a(["gen", "q.json", "--decode-b"]).count("--decode-b") == 1)
    # a talk route measured with decode on card A: the fit moves the decode to B -> MORE blocks on A (higher split)
    tt = new / "ltx/out/pp_timing.jsonl"
    tt.write_text(tt.read_text() + "\n" + "\n".join(json.dumps(dict(rec("a2vid", 17, 54, 90, 15.0), frames=97))
                                                     for _ in range(5)))
    import flow3d as F
    on = F._pp_split(new, "a2vid", 25)
    import subprocess
    r = subprocess.run([str(R.LTX_PY), str(Path(R.__file__).parent / "ltx_runner" / "pp_fit.py"), str(tt)],
                       capture_output=True, text=True)
    lines, cur, fits = r.stdout.splitlines(), None, {}
    for l in lines:
        cur = l[1:l.index("]")] if l.startswith("[") else cur
        if "BEST --split" in l:
            fits[cur] = int(l.split("--split ")[1].split(":")[0])
    off = fits["a2vid"]
    check(f"talk split moves to card A when decode goes to B ({off} -> {on})", on > off, (off, on))
    check("ti2v: split default 24, flags not doubled", w[w.index("--split") + 1] == "24" and w.count("--fa-amd") == 1, w)
    os.environ["REMAKE_LTX_SPLIT_ACT"] = "13"
    check("a flow's own split wins", a(["gen", "x.json", "--ti2v"])[-1] == "13")
    del os.environ["REMAKE_LTX_SPLIT_ACT"]
    check("--split passed by the caller is kept once", a(["gen", "x.json", "--split", "9"]).count("--split") == 1)
    check("encode untouched", a(["encode", "e.json", "--dev"]) == ["encode", "e.json", "--dev"])
    R.PP = False
    check("PP off: fast kernels only", a(["gen", "x.json"]) == ["gen", "x.json", "--fa-amd", "--blaslt"], a(["gen", "x.json"]))
    R.PP = True
print(json.dumps({"fails": fails}))
