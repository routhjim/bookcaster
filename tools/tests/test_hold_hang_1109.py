"""E32 (11-09): the Sherlock ch9 hold sat idle 92 min after card B's process died. Three fixes in the LTX split runner:
1. card A's waits on process B (each DiT hop, a decode on B, the start-up, the pair-decode flush and an idle lane)
   find a dead process B within seconds and end the hold with a lane error, so shot_flow gen() relaunches it;
2. the shm pool keeps ONE buffer per lane, direction and slot: a new shape replaces it (unpinned, freed);
3. the --more queue file is read on every take, so a chain's next line joins mid-hold ahead of the fill work.
CPU only: the runner's own code is cut from its source and run against stand-ins (a dying process, a fake card,
a queue file appended mid-run); the pool part uses real torch shm in the LTX venv, two processes.
    ~/tts_reader/.venv/bin/python tools/tests/test_hold_hang_1109.py"""
import json, multiprocessing as mp, os, queue, subprocess, sys, tempfile, textwrap, threading, time, types
from contextlib import nullcontext
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "ltx_runner" / "ltx_split_next.py"
SRC = R.read_text()
LTX_PY = Path.home() / "tts-ab" / ".venv-ltx" / "bin" / "python"
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


def cut(a, b, dedent=False):
    s = SRC[SRC.index(a):SRC.index(b, SRC.index(a))]
    return textwrap.dedent(s) if dedent else s


def back(outq, ready, n):
    """A stand-in process B: serves n replies, then fails the way back_server does (error on `ready`, exit)."""
    for i in range(n):
        outq.put(("reply", i))
    try:
        raise RuntimeError("unable to allocate shared memory(shm) for file </torch_x>: Success (0)")
    except RuntimeError as e:
        ready.put(("error", repr(e)))


def test_b_reply():
    g = {"queue": queue}
    exec(cut("B_POLL_S =", "\n\n\ndef thash"), g)
    ctx = mp.get_context("spawn")
    outqs, ready = [ctx.Queue() for _ in range(3)], ctx.Queue()
    p = ctx.Process(target=back, args=(outqs[0], ready, 2), daemon=True)
    p.start()
    pp = dict(outqs=outqs, ready=ready, proc=p)
    got, errs = {}, {}

    def lane(k):
        try:
            got[k] = [g["b_reply"](pp, outqs[k]) for _ in range(3 if k == 0 else 1)]
        except g["BDied"] as e:
            errs[k] = (time.time(), str(e))
    t0 = time.time()
    ths = [threading.Thread(target=lane, args=(k,), daemon=True) for k in range(3)]
    [t.start() for t in ths]
    [t.join(20) for t in ths]
    check("a dying process B: every lane waiting on it fails (none hangs)", sorted(errs) == [0, 1, 2] and
          not any(t.is_alive() for t in ths), (errs, got))
    late = max((t for t, _ in errs.values()), default=99) - t0
    check(f"... within seconds ({late:.1f} s; poll {g['B_POLL_S']} s)", late < 3 * g["B_POLL_S"] + 5, late)
    check("... after the replies B did send were delivered", got.get(0) is None and "died" in errs.get(0, (0, ""))[1])
    check("... and B's own error is still on `ready` for the end-of-hold log", ready.get(timeout=5)[0] == "error")
    # a live B that is slow is waited for (no false alarm)
    q2 = queue.Queue()
    alive = types.SimpleNamespace(is_alive=lambda: True, pid=1, exitcode=None)
    threading.Timer(2 * g["B_POLL_S"] + 0.5, lambda: q2.put("late reply")).start()
    check("a live but slow B is waited for past the poll interval", g["b_reply"](dict(proc=alive), q2) == "late reply")


def lane_env(tmp, **kw):
    """The runner's more_jobs + lane_loop (cut from cmd_gen) with stand-ins for the hold around them."""
    logs, ran = [], []
    g = dict(json=json, Path=Path, time=time, gc=types.SimpleNamespace(collect=lambda: None), lock=threading.Lock(),
             errors=[], timings={}, running=set(), idle={"s": 0.0, "since": None, "gaps": []}, PAIR=None, pp=None,
             lstate=types.SimpleNamespace(lane=0), A_OWNER=nullcontext(), torch=None, out_dir=Path(tmp),
             T0=time.time(), gpu_waiters=lambda: False, log=lambda *a: logs.append(" ".join(map(str, a))),
             BDied=type("BDied", (RuntimeError,), {}))
    g.update(kw)
    g["queued"] = {j["name"] for j in g["todo"]}

    def run_job(p, dev, j):
        ran.append(j["name"])
        g["timings"][j["name"]] = 1
        g.get("on_run", lambda j: None)(j)
    g["run_job"] = run_job
    exec(cut("    def more_jobs()", "\n    for d in range(torch.cuda.device_count())", dedent=True), g)
    return g, logs, ran


def test_lane_loop():
    with tempfile.TemporaryDirectory() as tmp:
        more, done = Path(tmp) / "more.json", Path(tmp) / "more.done"
        more.write_text("[]")
        args = types.SimpleNamespace(more=str(more), more_done=str(done), budget=1e9)

        def on_run(j):      # the flow appends while the hold runs: a chain's next line (priority) and a retake
            if j["name"] == "a":
                more.write_text(json.dumps([{"name": "y"}, {"name": "x", "priority": True}]))
            if j["name"] == "c":
                done.write_text("")
        todo = [{"name": n} for n in "abcd"]
        g, logs, ran = lane_env(tmp, todo=todo, args=args, on_run=on_run)
        g["lane_loop"](None, 0)
        check("a queue entry appended mid-hold joins at once, the chain line ahead of the planned rest",
              ran == ["a", "x", "b", "c", "d", "y"], ran)
        check("... the planned jobs keep their order", [n for n in ran if n in "abcd"] == list("abcd"), ran)
        check("... logged once", sum("more job(s) queued" in m for m in logs) == 1, logs)

        # --budget still stops a lane between jobs
        done.write_text("")
        more.write_text("[]")
        args.budget = 0.25
        g, logs, ran = lane_env(tmp, todo=[{"name": n} for n in "abcd"], args=args,
                                on_run=lambda j: time.sleep(0.15))
        g["lane_loop"](None, 0)
        check("--budget still stops the lane between jobs", ran == ["a", "b"] and any("budget reached" in m for m in logs),
              (ran, logs))
        # yield to a waiting GPU job after 2 clips
        args.budget = 1e9
        g, logs, ran = lane_env(tmp, todo=[{"name": n} for n in "abcd"], args=args, gpu_waiters=lambda: True)
        g["lane_loop"](None, 0)
        check("a lock waiter still makes the lane yield after 2 clips", ran == ["a", "b"] and any("yields" in m for m in logs),
              (ran, logs))
        # without --more nothing reads a file
        g, logs, ran = lane_env(tmp, todo=[{"name": n} for n in "ab"], args=types.SimpleNamespace(more="", budget=1e9))
        g["lane_loop"](None, 0)
        check("no --more: the planned jobs only", ran == ["a", "b"], ran)
        # process B dead while the lane has nothing to wait on B for (a chained piece's source never lands)
        dead = types.SimpleNamespace(is_alive=lambda: False, pid=7, exitcode=1)
        g, logs, ran = lane_env(tmp, todo=[{"name": "p_c1", "chain_from": "p_c0"}], args=args,
                                pp={"proc": dead})
        g["running"].add("p_c0")
        th = threading.Thread(target=g["lane_loop"], args=(None, 0), daemon=True)
        th.start()
        th.join(10)
        check("an idle lane (waiting on a chain source) still ends the hold once B is dead",
              not th.is_alive() and len(g["errors"]) == 1 and not ran, (th.is_alive(), g["errors"], ran))


def test_pair_flush():
    g = dict(threading=threading, B_POLL_S=0.2, BDied=type("BDied", (RuntimeError,), {}), time=time,
             nullcontext=nullcontext)
    exec(cut("class PairSched:", "\n\nclass RemoteDecoder"), g)
    P = object.__new__(g["PairSched"])
    dead = types.SimpleNamespace(is_alive=lambda: False)
    on_a, on_b = [], []
    P.pp, P.lk, P.parked, P.b_busy, P.on_b, P.n, P.err = {"proc": dead}, threading.Condition(), ["late"], False, \
        {1: lambda *a: None}, 1, []
    P.decode_a, P.to_b = (lambda it, lane: on_a.append(it)), (lambda it: on_b.append(it))
    th = threading.Thread(target=P.flush, daemon=True)
    th.start()
    th.join(10)
    check("pair flush with B dead: parked clips decode on A, the lost B decode fails the hold (no hang)",
          not th.is_alive() and on_a == ["late"] and not on_b and len(P.err) == 1, (th.is_alive(), on_a, on_b, P.err))


POOL_RUN = r'''
import gc, json, multiprocessing as mp, os, sys, types, weakref
import torch
src = open(sys.argv[1]).read()
REG = {"reg": [], "unreg": []}
class CudaRT:
    def cudaHostRegister(self, p, n, f): REG["reg"].append(p)
    def cudaHostUnregister(self, p): REG["unreg"].append(p)
torch.cuda.cudart = lambda: CudaRT()                                             # no card is touched
torch.cuda.current_stream = lambda *a: types.SimpleNamespace(synchronize=lambda: None)
import dataclasses
exec(src[src.index("def tmap"):src.index("\n\nPOOL = None")], globals())     # here: _Ref pickles as __main__._Ref
g = globals()
LANES, FRAMES = 3, list(range(193, 48, -8))     # a longest-first hold: every 8-frame length of a guided route

def shm_maps():
    return sum("torch_" in l for l in open("/proc/self/maps"))

def obj(k, f, lane, a2v):
    v = torch.arange(f * 64, dtype=torch.bfloat16).reshape(1, f, 64) + lane + k
    a = torch.full((1, f // 2, 16), float(k), dtype=torch.float32) if a2v else None
    return (v, a, {"mask": torch.ones(1, f, dtype=torch.bool)})

def server(inq, outq):
    pool = g["ShmPool"]()
    while (m := inq.get()) is not None:
        lane, packed = m
        v, a, pert = pool.unpack(packed, "cpu")
        outq.put((lane, pool.pack(((v.float() * 2).to(torch.bfloat16), a), f"b{lane}")))
    outq.put(("stats", len(pool.bufs), shm_maps(), len(REG["unreg"]), len(REG["reg"])))

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    inq, outq = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=server, args=(inq, outq), daemon=True)
    p.start()
    pool, ok, olds, peak, mid = g["ShmPool"](), True, [], 0, 0
    hops = 0
    for k, f in enumerate(FRAMES):
        for a2v in (True, False):               # mixed hold: talk and action routes
            for lane in range(LANES):
                for hop in range(2):            # two hops of one shape (the reuse path)
                    o = obj(k, f, lane, a2v)
                    olds += [weakref.ref(b) for b in pool.bufs.values()]
                    inq.put((lane, pool.pack(o, f"a{lane}")))
                    ln, back = outq.get(timeout=20)
                    v, a = pool.unpack(back, "cpu")
                    ok &= ln == lane and torch.equal(v, (o[0].float() * 2).to(torch.bfloat16)) and \
                        (a is None if not a2v else torch.equal(a, o[1]))
                    peak = max(peak, len(pool.bufs))
                    mid = shm_maps() if k == 3 else mid
                    hops += 1
    inq.put(None)
    stats = outq.get(timeout=20)
    p.join(30)
    gc.collect()
    live = {id(b) for b in pool.bufs.values()}
    freed = all(r() is None or id(r()) in live for r in olds)
    print(json.dumps(dict(ok=bool(ok), hops=hops, shapes=len(FRAMES) * 2, a_bufs=len(pool.bufs), peak=peak, a_maps_mid=mid,
                          a_maps=shm_maps(), a_unreg=len(REG["unreg"]), a_reg=len(REG["reg"]), b_bufs=stats[1],
                          b_maps=stats[2], b_unreg=stats[3], b_reg=stats[4], freed=freed,
                          unreg_were_pinned=set(REG["unreg"]) <= set(REG["reg"]))))
'''


def test_pool():
    if not LTX_PY.exists():
        check("LTX venv for the pool test", False, LTX_PY)
        return
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "pool_run.py"
        f.write_text(POOL_RUN)
        env = {**os.environ, "HIP_VISIBLE_DEVICES": "-1", "CUDA_VISIBLE_DEVICES": "-1", "ROCR_VISIBLE_DEVICES": "-1"}
        r = subprocess.run([str(LTX_PY), "-I", str(f), str(R)], capture_output=True, text=True, timeout=300, env=env)
    if r.returncode:
        check("pool run", False, r.stderr[-2000:])
        return
    s = json.loads(r.stdout.strip().splitlines()[-1])
    print("   ", s)
    slots = 3 * (3 + 2)        # each side: lanes x (a: video, audio, mask + b: video, audio)
    check("pool: every hop's values arrive exact, both ways", s["ok"], s)
    check(f"pool: {s['shapes']} route-shapes x 3 lanes: card A keeps one buffer per lane/direction/slot "
          f"({s['a_bufs']}, peak {s['peak']})", s["a_bufs"] <= slots and s["peak"] <= slots, s)
    check(f"pool: process B too ({s['b_bufs']})", s["b_bufs"] <= slots, s)
    check(f"pool: card A's shm mappings stay flat as shapes keep coming ({s['a_maps_mid']} after 4 lengths, "
          f"{s['a_maps']} at the end)", s["a_maps"] <= s["a_maps_mid"] + 2, s)
    check("pool: a replaced buffer is unpinned (and only a pinned one)", s["a_unreg"] > 0 and s["b_unreg"] > 0 and
          s["unreg_were_pinned"] and s["a_reg"] - s["a_unreg"] == s["a_bufs"], s)
    check("pool: replaced buffers are freed (no reference left)", s["freed"], s)


if __name__ == "__main__":
    test_b_reply()
    test_lane_loop()
    test_pair_flush()
    test_pool()
    print("ALL PASS" if not fails else f"{len(fails)} FAIL: {fails}")
    sys.exit(1 if fails else 0)
