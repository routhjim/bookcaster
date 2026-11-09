"""E30 (11-08): a pieced talk line that misses the lip-sync bar is taken again from its first failing piece on; the
passing pieces before it are kept (E29: bad pieces are per take - Long Dark 02m32s went 5.6/1.5 then 6.0/1.0 and its
whole-line retakes threw the good first piece away). The new first piece starts on the kept piece's last frame,
TALK_TAKES still caps the takes per line, and the kept chain's pieces are the line's pieces under a join made of
them. A fake runner renders the queue (each piece's first frame = its source's _last.png); SyncNet is scripted."""
import json, sys, tempfile, threading, time, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# --- the pure rules --------------------------------------------------------------------------------------------------
tree = {"a_c0": (0, None), "a_c1": (1, "a_c0"), "a_t2_c1": (1, "a_c0"), "a_t2_c2": (2, "a_t2_c1"), "a_c2": (2, "a_c1")}
check("chains: one piece per position, each on the one before, earliest rendered first",
      S._piece_chains(tree, 3) == [["a_c0", "a_c1", "a_c2"], ["a_c0", "a_t2_c1", "a_t2_c2"]], S._piece_chains(tree, 3))
sc = {"a_c0": 6, "a_c1": 2, "a_c2": 6, "a_t2_c1": 7, "a_t2_c2": 1}.get
check("retake from the chain that fails latest (fewest pieces to redo), at its first failing piece",
      S._piece_retake(S._piece_chains(tree, 3), sc) == (["a_c0", "a_t2_c1", "a_t2_c2"], 2))
check("a passing chain: no retake", S._piece_retake([["x_c0", "x_c1"]], {"x_c0": 5.0, "x_c1": 9}.get) is None)

# --- the flow -------------------------------------------------------------------------------------------------------
L = Path(tempfile.mkdtemp(prefix="pretake_")); (L / "out").mkdir()
CONF = {
    # A (Acts ch.3 02m46s): piece 1 good, piece 2 bad -> only piece 2 again, from piece 1; take 3 passes
    "A_s0_c0": 7.1, "A_s0_c1": 1.9, "A_s0_t2_c1": 1.0, "A_s0_t3_c1": 7.8,
    # B (Long Dark 02m32s): piece 1 bad too -> take 2 whole; then piece 1 good -> take 3 of piece 2 only, from take 2's
    "B_s0_c0": 2.2, "B_s0_c1": 1.5, "B_s0_t2_c0": 5.6, "B_s0_t2_c1": 1.5, "B_s0_t3_c1": 6.0,
    # C: three pieces, none ever passes: the best worst piece is take 1's chain (nothing swapped)
    "C_s0_c0": 6.0, "C_s0_c1": 2.0, "C_s0_c2": 6.0, "C_s0_t2_c1": 7.0, "C_s0_t2_c2": 1.0, "C_s0_t3_c2": 1.5,
    # D: a chained talk shot (piece 1 starts on an action pick, #148), piece 1 bad -> a whole take, still from the pick
    "D_s0_c0": 1.0, "D_s0_c1": 6.0, "D_s0_t2_c0": 6.5, "D_s0_t2_c1": 6.2,
    # E: passes first time
    "E_s0_c0": 8.0, "E_s0_c1": 7.0}
def fake_sync(book, L_, names, log=print, quiet=False):
    for n in names:
        c = L_ / "out" / f"{n}.mp4"
        st = c.stat()
        (L_ / "out" / f"{n}.sync.json").write_text(json.dumps({"sig": f"{st.st_size}:{st.st_mtime_ns}", "offset": 0,
                                                              "conf": CONF.get(n, 9.0)}))
joins = {}
def fake_stitch(book, R, L_, talk, log=print, only=None, quiet=False, jobs_file="talk_stitch.json"):
    for n in only or ():
        js = sorted([j for j in talk if j["name"].rsplit("_c", 1)[0] == n], key=lambda j: j["chunk"][0])
        joins.setdefault(n, []).append([(L_ / "out" / f"{j['name']}.mp4").read_text() for j in js])
        (L_ / "out" / f"{n}.mp4").write_text("join of " + " + ".join(joins[n][-1]))
S._sync_offsets = fake_sync
S._stitch_talk = fake_stitch
S.time = types.SimpleNamespace(sleep=lambda s: time.sleep(0.02))

def line(n, P, seed, chain0=None):
    js = []
    for i in range(P):
        j = {"name": f"{n}_c{i}", "seed": seed, "chunk": [i, P, 96 * i, 97], "line": n.rsplit("_s", 1)[0],
             "images": [] if i else [{"path": "key.png", "strength": 0.5}]}
        if i:
            j.update(chain_from=f"{n}_c{i - 1}", chain_frame=96, overlap=1)
        elif chain0:
            j.update(chain_from=chain0, chain_frame=288, overlap=1, images=[])
        js.append(j)
    return js
talk = line("A_s0", 2, 100) + line("B_s0", 2, 200) + line("C_s0", 3, 300) + line("D_s0", 2, 400, "action-1_s0") + \
       line("E_s0", 2, 500)
(L / "out" / "action-1_s0_last.png").write_text("last of action-1_s0")
(L / "out" / "A_s0_t2_c1.mp4").write_text("stale from an earlier run")        # must not count as landed
more = L / "more.json"; more.write_text("[]")
stop, seen, started = threading.Event(), set(), {}

def render(j):
    src = j.get("chain_from")
    first = (L / "out" / f"{src}_last.png").read_text() if src else "key"
    started[j["name"]] = first
    (L / "out" / f"{j['name']}_last.png").write_text(f"last of {j['name']}")
    (L / "out" / f"{j['name']}.mp4").write_text(f"{j['name']} seed {j['seed']} from [{first}]")
def runner():
    for j in talk:
        render(j)
    while not stop.is_set():
        for j in json.loads(more.read_text()):
            if j["name"] not in seen:
                seen.add(j["name"]); time.sleep(0.03)
                render(j)
        time.sleep(0.02)
th = threading.Thread(target=runner, daemon=True); th.start()
logs = []
S._sync_as_they_land(None, None, L, talk, threading.Event(), logs.append, False, more)
stop.set()
Q = {j["name"]: j for j in json.loads(more.read_text())}
out = lambda x: (L / "out" / x).read_text()

check("A: only the failing piece is taken again, twice (TALK_TAKES caps the takes)",
      sorted(n for n in Q if n.startswith("A")) == ["A_s0_t2_c1", "A_s0_t3_c1"], sorted(Q))
check("A: each retaken piece starts on the kept piece 1's last frame",
      Q["A_s0_t2_c1"]["chain_from"] == Q["A_s0_t3_c1"]["chain_from"] == "A_s0_c0" and
      started["A_s0_t2_c1"] == started["A_s0_t3_c1"] == "last of A_s0_c0", started)
check("A: new seeds as a whole take's (+7919 a take)", Q["A_s0_t2_c1"]["seed"] == 100 + 7919 and
      Q["A_s0_t3_c1"]["seed"] == 100 + 2 * 7919)
check("A: an earlier run's file of that name was removed, the render's own measured",
      out("A_s0_t2_c1.mp4").startswith("A_s0_t2_c1 seed"), out("A_s0_t2_c1.mp4"))
check("A: the passing chain is the line: piece 1 kept, take 3's piece 2 under it",
      [out(f"A_s0_c{i}.mp4").split(" seed")[0] for i in (0, 1)] == ["A_s0_c0", "A_s0_t3_c1"])
check("A: the join is made again from exactly those pieces",
      joins["A_s0"][-1] == [out("A_s0_c0.mp4"), out("A_s0_c1.mp4")] and out("A_s0.mp4") == "join of " +
      " + ".join(joins["A_s0"][-1]), joins.get("A_s0"))
check("A: the kept piece's last frame travels with it", out("A_s0_c1_last.png") == "last of A_s0_t3_c1")
check("A: the record names the kept pieces (so the E29 fix-up leaves them)",
      json.loads(out("A_s0.synced"))["_pieces"] == ["A_s0_c0", "A_s0_t3_c1"])

check("B: piece 1 failed: take 2 is the whole line, chained to its own piece 1",
      {n for n in Q if n.startswith("B_s0_t2")} == {"B_s0_t2_c0", "B_s0_t2_c1"} and
      Q["B_s0_t2_c1"]["chain_from"] == "B_s0_t2_c0" and "chain_from" not in Q["B_s0_t2_c0"], sorted(Q))
check("B: take 3 is piece 2 only, from take 2's good piece 1",
      {n for n in Q if n.startswith("B_s0_t3")} == {"B_s0_t3_c1"} and Q["B_s0_t3_c1"]["chain_from"] == "B_s0_t2_c0"
      and started["B_s0_t3_c1"] == "last of B_s0_t2_c0", Q.get("B_s0_t3_c1"))
check("B: the line is take 2's piece 1 + take 3's piece 2 (6.0, where whole retakes kept 1.5)",
      [out(f"B_s0_c{i}.mp4").split(" seed")[0] for i in (0, 1)] == ["B_s0_t2_c0", "B_s0_t3_c1"] and
      json.loads(out("B_s0_c1.sync.json"))["conf"] == 6.0)
check("B: ...the kept piece 2 starts where the kept piece 1 ends", "from [last of B_s0_t2_c0]" in out("B_s0_c1.mp4"))
check("B: the join holds those pieces", joins["B_s0"][-1] == [out("B_s0_c0.mp4"), out("B_s0_c1.mp4")])

check("C: three pieces: take 2 redoes pieces 2-3 from piece 1, take 3 only piece 3 from take 2's piece 2",
      sorted(n for n in Q if n.startswith("C")) == ["C_s0_t2_c1", "C_s0_t2_c2", "C_s0_t3_c2"] and
      Q["C_s0_t2_c1"]["chain_from"] == "C_s0_c0" and Q["C_s0_t2_c2"]["chain_from"] == "C_s0_t2_c1" and
      Q["C_s0_t3_c2"]["chain_from"] == "C_s0_t2_c1", {n: Q[n].get("chain_from") for n in Q if n.startswith("C")})
check("C: never passing: the chain with the best worst piece (take 1's, 2.0) stays, its join untouched",
      [out(f"C_s0_c{i}.mp4").split(" seed")[0] for i in range(3)] == ["C_s0_c0", "C_s0_c1", "C_s0_c2"] and
      len(joins["C_s0"]) == 1, joins.get("C_s0"))
check("C: no take beyond TALK_TAKES", not any(n.startswith("C_s0_t4") for n in Q))

check("D: a talk shot chained from an action pick: its piece 1 retake starts on that pick (no crash)",
      Q.get("D_s0_t2_c0", {}).get("chain_from") == "action-1_s0" and started.get("D_s0_t2_c0") == "last of action-1_s0",
      Q.get("D_s0_t2_c0"))
check("D: ...and the passing take is the line", [out(f"D_s0_c{i}.mp4").split(" seed")[0] for i in (0, 1)] ==
      ["D_s0_t2_c0", "D_s0_t2_c1"])
check("E: a passing line is not retaken", not any(n.startswith("E") for n in Q))
check("pieces rendered again: A 2, B 3, C 3, D 2 (whole-line retakes: 4, 4, 6, 2)",
      [sum(n.startswith(x) for n in Q) for x in "ABCD"] == [2, 3, 3, 2], [sum(n.startswith(x) for n in Q) for x in "ABCD"])
check("kept lines are logged", any("A_s0: lip sync line" in m and "kept" in m for m in logs), logs)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
