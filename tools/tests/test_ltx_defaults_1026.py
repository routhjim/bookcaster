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
    r = {"job": "x", "route": route, "pp": True, "lanes": 2, "split": split, "a_dit_s": a_dit, "a_copy_s": 1.0,
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
    check("quiet split from the NEWEST chapter (counts card-B decode)", q[q.index("--split") + 1] != "17" and
          int(q[q.index("--split") + 1]) > 17, q)
    t = a(["gen", "talk_jobs.json", "--a2vid"])
    check("talk: no --decode-b (card 65 is the slower card there)", "--decode-b" not in t, t)
    check("talk: default a2vid split 25 with no timing", t[t.index("--split") + 1] == "25", t)
    check("talk: stagger + pool", "--stagger" in t and "--pp-pool" in t, t)
    w = a(["gen", "two_jobs.json", "--ti2v", "--fa-amd"])
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
