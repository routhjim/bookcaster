"""The asset line (tools/assets/line.py + assetd.py dispatch): admission, short lines, batching, the power guard,
resume after a restart, the engine card's turns, stalls, and the reload onto newer code. No GPU, no UE, no Blender:
the scheduler runs in a temporary folder with fake stations; the reload test runs a real dry-run line (its own
socket, folder, fake cards and power supply, a copy of the line's code) with commands that take about a second."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="asset_line_test_")
os.environ["ASSETS_SCHED_DIR"] = TMP
os.environ["ASSETS_LIBRARY"] = os.path.join(TMP, "lib")
os.environ["ASSETS_LOGS"] = os.path.join(TMP, "logs")
os.environ["ASSETS_LINE_MAX"] = "3"
os.environ["ASSETS_LINE_BATCH"] = "8"
os.environ["ASSETS_LINE_INHIBIT"] = "0"
os.environ["ASSETS_GPU_HOLD"] = os.path.join(TMP, "no_GPU_HOLD")
ASSETS = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))

import line as L  # noqa: E402
import assetd as D  # noqa: E402


def fake_power(root, status="Charging", cap=78, energy=47.0, ac=True, usb=(1, 1)):
    root = Path(root)
    for d in ("BAT0", "AC0", "ucsi-source-psy-USBC000:001", "ucsi-source-psy-USBC000:002"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "BAT0" / "status").write_text(status + "\n")
    (root / "BAT0" / "capacity").write_text(f"{cap}\n")
    (root / "BAT0" / "energy_now").write_text(f"{int(energy * 1e6)}\n")
    (root / "AC0" / "online").write_text("1\n" if ac else "0\n")
    for i, u in enumerate(usb):
        (root / f"ucsi-source-psy-USBC000:00{i + 1}" / "online").write_text(f"{u}\n")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class AdmissionTest(unittest.TestCase):
    def test_line_full(self):
        self.assertIsNone(L.line_full({"igpu": 2}, "igpu", 3))
        msg = L.line_full({"igpu": 3}, "igpu", 3)
        self.assertIn("line full", msg)
        self.assertIsNone(L.line_full({}, "card05", 3))



class BatcherTest(unittest.TestCase):
    def test_keeps_kind_then_switches(self):
        b = L.Batcher(batch=3)
        ready = [(1, "refs", "r1"), (2, "paint", "p1"), (3, "refs", "r2"), (4, "refs", "r3"), (5, "refs", "r4"),
                 (6, "cloth", "c1")]
        order = []
        while ready:
            x = b.pick("igpu", ready)
            order.append(x)
            ready = [r for r in ready if r[2] != x]
        # refs (oldest) batch of 3, then the oldest other kind (paint), then refs again, then cloth
        self.assertEqual(order, ["r1", "r2", "r3", "p1", "r4", "c1"])

    def test_batch_continues_when_alone(self):
        b = L.Batcher(batch=2)
        self.assertEqual([b.pick("s", [(i, "a", i)]) for i in range(5)], list(range(5)))


class WorkListTest(unittest.TestCase):
    def test_resume_requeues_running(self):
        d = tempfile.mkdtemp(dir=TMP)
        wl = L.WorkList(d)
        for i in range(3):
            wl.save({"seq": wl.next_seq(), "type": "cmd", "id": f"x{i}", "state": "queued", "stages": {"a": "wait"}})
        e = json.loads((wl.edir / "000002.json").read_text())
        e.update(state="running", stages={"a": "run"})
        wl.save(e)
        wl.save(dict(json.loads((wl.edir / "000001.json").read_text()), state="done"))
        got = L.WorkList(d).load()
        self.assertEqual([x["seq"] for x in got], [2, 3])
        self.assertEqual(got[0]["stages"], {"a": "wait"})
        self.assertEqual(got[0]["resumed"], ["a"])
        self.assertEqual(got[0]["state"], "queued")
        self.assertTrue((wl.ddir / "000001.json").exists())
        self.assertEqual(L.WorkList(d).next_seq(), 4)


class PowerGuardTest(unittest.TestCase):
    def guard(self, **kw):
        root = tempfile.mkdtemp(dir=TMP)
        fake_power(root, **kw)
        aspm = Path(root) / "aspm"
        aspm.write_text("default [performance] powersave powersupersave\n")
        pci = Path(root) / "pci"
        for c in ("05", "65"):
            (pci / f"0000:{c}:00.0").mkdir(parents=True)
        os.environ["ASSETS_PCI_ROOT"] = str(pci)
        clk = Clock()
        g = L.PowerGuard(root=root, cards=["05", "65"], aspm=str(aspm), clock=clk)
        return g, Path(root), clk

    def test_clear(self):
        g, _, _ = self.guard()
        g.update()
        v = g.verdict()
        self.assertTrue(v["cards"] and not v["pause"], v)

    def test_floor_with_hysteresis(self):
        g, root, clk = self.guard(cap=21)
        g.update()
        self.assertFalse(g.verdict()["pause"])
        for cap, want in ((20, True), (30, True), (39, True), (40, False), (25, False)):
            fake_power(root, cap=cap)
            clk.t += 60
            g.update()
            self.assertEqual(g.verdict()["pause"], want, cap)
        self.assertIn("paused: battery", " ".join(PowerGuardTest.paused_why(root)))

    @staticmethod
    def paused_why(root):
        g = L.PowerGuard(root=root, cards=[], aspm=str(Path(root) / "aspm"))
        fake_power(root, cap=15)
        g.update()
        return g.verdict()["why"]

    def test_discharging_alone_does_not_pause(self):
        g, _, _ = self.guard(status="Discharging", cap=60)
        g.update()
        self.assertFalse(g.verdict()["pause"])

    def test_charger_and_card_loss(self):
        g, root, clk = self.guard(ac=False)
        g.update()
        v = g.verdict()
        self.assertFalse(v["cards"])
        self.assertFalse(v["pause"])
        self.assertIn("charger", " ".join(v["why"]))
        fake_power(root, ac=True)
        os.rmdir(Path(os.environ["ASSETS_PCI_ROOT"]) / "0000:65:00.0")
        clk.t += 5
        g.update()
        self.assertIn("card 65", " ".join(g.verdict()["why"]))
        (Path(os.environ["ASSETS_PCI_ROOT"]) / "0000:65:00.0").mkdir()
        clk.t += 5
        g.update()
        self.assertTrue(g.verdict()["cards"])
        self.assertTrue(any("back" in e for _, e in g.events))

    def test_aspm(self):
        g, root, _ = self.guard()
        (root / "aspm").write_text("[default] performance powersave powersupersave\n")
        g.update()
        v = g.verdict()
        self.assertFalse(v["cards"])
        self.assertFalse(v["pause"])                        # iGPU work may go on
        self.assertIn("ASPM", " ".join(v["why"]))


class LineBase(unittest.TestCase):
    """A scheduler in a temporary folder with fake runs (no tests of its own)."""

    def sched(self, fake=True):
        root = tempfile.mkdtemp(dir=TMP)
        os.environ["ASSETS_SCHED_DIR"] = root
        D.C.SCHED_DIR = Path(root)
        S = D.Sched()
        D.SCHED = S
        S.wl = L.WorkList(Path(root) / "line")
        S.verdict = {"cards": True, "pause": False, "why": []}
        S.started = []
        S.mem_read = lambda: 1000.0                         # never the box's real memory (MemoryTest sets its own)
        S.editor_mem = lambda e: 0.0
        for c in S.cards.values():                          # never the box's real card locks
            c.lock_path = str(Path(root) / f"gpu-{c.bus}.lock")
            c.tenant_path = Path(root) / f"gpu-{c.bus}.tenant"
        if not fake:
            return S

        def launch(st, fn, arg):
            st.state, st.t_start = "run", D.now()
            S.started.append((st.job.id, S.station(st), S.skind(st)))
            return True
        S.launch = launch
        plain = S.dispatch

        def dispatch():                                     # the scheduler calls it under its lock
            with S.lock:
                plain()
        S.dispatch = dispatch
        return S

    def add(self, S, n, station="igpu", kind="refs"):
        out = []
        for i in range(n):
            job, why = S.add_entry({"type": "cmd", "id": f"{kind}{i}", "station": station, "kind": kind,
                                    "cmd": ["true"]})
            out.append(why)
        return out

    def finish_running(self, S):
        for j in list(S.jobs):
            for st in j.stages:
                if st.state == "run":
                    st.state = "done"
                    S.after_stage(st)


class DispatchTest(LineBase):
    """The scheduler's dispatch with fake runs: one job per station, batches, line full, resume."""

    def test_flood_is_refused(self):
        S = self.sched()
        res = self.add(S, 40)
        self.assertEqual(sum(1 for r in res if r is None), 3)
        self.assertTrue(all("line full" in r for r in res[3:]))
        self.assertEqual(len(S.jobs), 3)

    def test_one_job_per_station_and_batches(self):
        S = self.sched()
        self.add(S, 2, kind="refs")
        self.add(S, 1, kind="cloth")
        order = []
        for _ in range(6):
            for j in S.jobs:
                for st in j.stages:
                    if st.state == "wait":
                        st.state = "ready"
            S.dispatch()
            running = [st for j in S.jobs for st in j.stages if st.state == "run"]
            self.assertLessEqual(len(running), 1)
            if running:
                order.append(running[0].job.id)
                if running[0].job.id == "refs0":            # a refs job arrives while the batch runs
                    S.add_entry({"type": "cmd", "id": "refs_late", "station": "igpu", "kind": "refs", "cmd": ["true"]})
            self.finish_running(S)
        self.assertEqual(order[:3], ["refs0", "refs1", "refs_late"])  # the new refs joins the batch before cloth
        self.assertEqual(order[3], "cloth0")

    def test_guard_pause_and_cards(self):
        S = self.sched()
        self.add(S, 1, kind="refs")
        self.add(S, 1, station="blender", kind="shape")
        for j in S.jobs:
            j.stages[0].state = "ready"
        S.verdict = {"cards": True, "pause": True, "why": ["paused: battery 20%"]}
        S.dispatch()
        self.assertEqual(S.started, [])
        S.verdict = {"cards": False, "pause": False, "why": ["lost card 05"]}
        S.dispatch()
        self.assertEqual(S.started, [("refs0", "igpu", "refs")])      # the card job is held

    def test_upstream_held_when_next_line_full(self):
        S = self.sched()
        e = S.add_entry
        # a long stage A at the iGPU followed by B on the Blender card: B's line is full, so A waits
        jobs = [e({"type": "cmd", "id": f"b{i}", "station": "blender", "kind": "shape", "cmd": ["true"]})[0]
                for i in range(3)]
        self.assertIsNotNone(L.line_full(S.heading(), "card05"))
        self.assertIn("line full", e({"type": "cmd", "id": "b9", "station": "blender", "kind": "shape",
                                      "cmd": ["true"]})[1])
        self.assertEqual(len(jobs), 3)

    def test_resume_after_restart(self):
        S = self.sched()
        self.add(S, 2, kind="refs")
        S.jobs[0].stages[0].state = "run"
        S.persist(S.jobs[0])
        root = S.wl.root
        S2 = self.sched()
        S2.wl = L.WorkList(root)
        D.resume(S2)
        self.assertEqual([j.id for j in S2.jobs], ["refs0", "refs1"])
        self.assertEqual([j.stages[0].state for j in S2.jobs], ["wait", "wait"])
        self.assertEqual(S2.jobs[0].entry["resumed"], ["refs"])

    def test_video_window_holds_card_work(self):
        S = self.sched()
        self.add(S, 1, station="video", kind="video")
        self.add(S, 1, station="blender", kind="shape")
        for j in S.jobs:
            j.stages[0].state = "ready"
        S.dispatch()
        self.assertEqual(S.started, [])                      # the shape job waits; the window is being opened
        self.assertTrue(S.released and "video window" in S.released["reason"])
        S.released["done_t"] = D.now()
        S.dispatch()
        self.assertEqual(S.started, [("video0", "video", "video")])   # the window runs clips, not card work

    def test_command_borrows_idle_engine_card(self):
        S = self.sched()
        ue = f"card{D.C.UE_CARD}"
        card = S.cards[D.C.UE_CARD]

        class FakeEditor:
            def __init__(self):
                self.card, self.state = card, "ready"

            def quit(self):
                self.state = "quitting"

        ed = FakeEditor()
        S.editors = [ed]
        job, why = S.add_entry({"type": "cmd", "id": "shape0", "station": "engine", "kind": "shape", "cmd": ["true"]})
        self.assertIsNone(why)
        self.assertEqual(job.entry["station"], ue)
        job.stages[0].state = "ready"
        S.dispatch()
        self.assertEqual(S.started, [])                      # the idle editor is asked to unload first
        self.assertEqual(ed.state, "quitting")
        S.dispatch()
        self.assertEqual(S.started, [])                      # still unloading
        ed.state = "dead"
        S.dispatch()
        self.assertEqual(S.started, [("shape0", ue, "shape")])

    def test_engine_work_goes_before_a_borrow(self):
        S = self.sched()
        root = Path(tempfile.mkdtemp(dir=TMP))
        for f in ("p.uproject", "s.py"):
            (root / f).write_text("")
        ej, _ = S.add_entry({"type": "ue_job", "id": "set0", "ue": {"project": str(root / "p.uproject"),
                                                                    "script": str(root / "s.py"),
                                                                    "log_dir": str(root / "log")}})
        cj, _ = S.add_entry({"type": "cmd", "id": "shape0", "station": "engine", "kind": "shape", "cmd": ["true"]})
        cj.stages[0].state = "ready"
        ej.stages[0].state = "wait"                         # its dependencies are not done: it cannot run yet
        S.dispatch()
        self.assertIn(("shape0", f"card{D.C.UE_CARD}", "shape"), S.started)   # a waiting engine stage does not block
        S.started.clear()
        cj.stages[0].state = "ready"
        ej.stages[0].state = "ready"
        S.dispatch()
        self.assertEqual([s for s in S.started if s[0] == "shape0"], [])   # an engine job waits: no borrow
        ej.stages[0].state = "done"
        S.dispatch()
        self.assertIn(("shape0", f"card{D.C.UE_CARD}", "shape"), S.started)

    def test_a_second_submission_of_the_same_job_is_refused(self):
        # 11-11: two submitters five seconds apart queued one shape twice (entries 673 and 674), it ran twice
        S = self.sched()
        cmd = {"type": "cmd", "id": "shapeX", "station": "engine", "kind": "shape", "cmd": ["true"]}
        first, why = S.add_entry(dict(cmd))
        self.assertIsNone(why)
        _, why = S.add_entry(dict(cmd))
        self.assertEqual(why, f"already queued as entry {first.seq} (shapeX, waiting); not queued again")
        first.stages[0].state = "run"
        _, why = S.add_entry(dict(cmd))
        self.assertIn("running", why)
        self.assertIsNone(S.add_entry(dict(cmd, station="blender"))[1])   # another station: another job
        self.finish_running(S)
        self.assertIsNone(S.add_entry(dict(cmd))[1])                      # finished: it may run again
        sim_root = Path(tempfile.mkdtemp(dir=TMP))
        for f in ("p.uproject", "s.py", "m.py"):
            (sim_root / f).write_text("")
        ue = {"project": str(sim_root / "p.uproject"), "script": str(sim_root / "s.py"), "job": {"a": 1}}
        self.assertIsNone(S.add_entry({"type": "ue_job", "id": "set0", "ue": dict(ue)})[1])
        self.assertIn("already queued", S.add_entry({"type": "ue_job", "id": "set0", "ue": dict(ue)})[1])
        self.assertIsNone(S.add_entry({"type": "ue_job", "id": "set0", "ue": dict(ue, job={"a": 2})})[1])
        spec = sim_root / "pa.json"
        spec.write_text(json.dumps({"id": "pa", "class": "prop"}))
        a = {"type": "asset", "id": "pa", "argv": [str(spec), str(sim_root / "m.py")], "spec": str(spec),
             "model": str(sim_root / "m.py"), "kind": "final"}
        self.assertIsNone(S.add_entry(dict(a))[1])
        self.assertIn("already queued", S.add_entry(dict(a))[1])

    def test_image_station(self):
        S = self.sched()
        self.assertEqual(D.STATION_ALIASES["image"], D.C.IMAGE_STATION)
        self.assertEqual(D.station_device("card05"), "05")
        self.assertEqual(D.station_device("igpu"), "igpu")
        job, why = S.add_entry({"type": "cmd", "id": "draw0", "station": D.C.IMAGE_STATION, "kind": "refs",
                                "cmd": ["true"]})
        self.assertIsNone(why)
        paint = D.Stage(job, "paint", "paint", [])
        self.assertEqual(S.station(paint), D.C.IMAGE_STATION)


class FakeEditor:
    def __init__(self, card, state="ready"):
        self.card, self.state, self.pid, self.project = card, state, 4242, None

    def quit(self):
        if self.state not in ("dead", "quitting"):
            self.state = "quitting"


class Sim:
    """A dry-run line on a simulated clock: rounds of advance + dispatch; stages end after a set number of rounds;
    the editor loads in one round, unloads in one round, and is (re)loaded by the rule manage_editors follows."""

    def __init__(self, test, S, rounds_by_kind):
        self.t, self.S, self.rounds_by_kind = test, S, rounds_by_kind
        self.ue = f"card{D.C.UE_CARD}"
        self.card = S.cards[D.C.UE_CARD]
        self.runs = {}                                  # stage -> rounds left
        self.starts = []                                # (round, station, seq, stage name)
        self.r = 0
        S.editors = []
        S.place_blender = lambda st: ("card", S.cards[D.C.BLENDER_CARD])

        def place_ue(st):
            ed = next((e for e in S.editors if e.state == "ready"), None)
            if ed is not None:
                ed.state = "busy"
            return ed
        S.place_ue = place_ue

        def launch(st, fn, arg):
            st.state, st.t_start = "run", D.now()
            self.runs[st] = self.rounds_by_kind.get(S.skind(st) if S.station(st) else "finalize", 1)
            self.starts.append((self.r, S.station(st), st.job.seq, st.name))
            return True
        S.launch = launch
        self.root = Path(tempfile.mkdtemp(dir=TMP))
        (self.root / "p.uproject").write_text("")
        (self.root / "s.py").write_text("")
        (self.root / "m.py").write_text("")

    def asset(self, name):
        spec = self.root / f"{name}.json"
        spec.write_text(json.dumps({"id": name, "class": "prop"}))
        return self.S.add_entry({"type": "asset", "id": name, "argv": [str(spec), str(self.root / "m.py")],
                                 "spec": str(spec), "model": str(self.root / "m.py"), "kind": "final"})

    def borrow(self, name):
        return self.S.add_entry({"type": "cmd", "id": name, "station": "engine", "kind": "shape", "cmd": ["true"]})

    def bake_cmd(self, name):
        return self.S.add_entry({"type": "cmd", "id": name, "station": "blender", "kind": "refs", "cmd": ["true"]})

    def script(self, name):
        return self.S.add_entry({"type": "ue_job", "id": name, "ue": {"project": str(self.root / "p.uproject"),
                                                                     "script": str(self.root / "s.py"),
                                                                     "log_dir": str(self.root / name)}})

    def step(self):
        S, self.r = self.S, self.r + 1
        with S.lock:
            for st, left in list(self.runs.items()):            # stages end
                if left <= 0:
                    del self.runs[st]
                    st.state, st.t_end = "done", D.now()
                    if st.kind == "ue":
                        for e in S.editors:
                            if e.state == "busy":
                                e.state = "ready"
                    S.after_stage(st)
                else:
                    self.runs[st] = left - 1
            for e in S.editors:                                 # the editor unloads in one round
                if e.state == "quitting":
                    e.state = "dead"
            S.editors = [e for e in S.editors if e.state != "dead"]
            S.advance(D.now())
            S.dispatch()
            # manage_editors' start rule (no real editor): engine work wanted, not a borrow's turn, card not lent
            want = any(s.kind == "ue" and s.state in ("wait", "ready") for j in S.jobs for s in j.stages)
            if (want and not S.editors and S.engine_turn == "editor" and self.card.cmd_busy is None
                    and not S.reload):
                S.editors = [FakeEditor(self.card)]
            self.check_invariant()

    def check_invariant(self):
        """A station that runs nothing and has a ready entry either starts it or records why (no silent stall)."""
        S = self.S
        busy = {S.station(st) for j in S.jobs for st in j.stages if st.state == "run" and S.station(st)}
        for j in S.jobs:
            for st in j.stages:
                b = S.station(st)
                if st.state == "ready" and b and b not in busy and not j.cancelled:
                    self.t.assertTrue(S.why.get(S.line_of(st)), f"round {self.r}: {b} idle, #{j.seq} {st.name} "
                                                                f"ready, no reason recorded")

    def run(self, max_rounds=400):
        while self.S.jobs and self.r < max_rounds:
            self.step()
        self.t.assertEqual(self.S.jobs, [], f"not done after {self.r} rounds: "
                           + str([(j.seq, j.id, [(s.name, s.state) for s in j.stages]) for j in self.S.jobs])
                           + f" why {self.S.why}")
        names = [(seq, name) for _, _, seq, name in self.starts]
        self.t.assertEqual(len(names), len(set(names)), "a stage started twice")

    def first(self, station, seq=None, name=None):
        return next((r for r, b, q, n in self.starts if b == station and (seq is None or q == seq)
                     and (name is None or n == name)), None)


class EngineCardTest(LineBase):
    """The faults of 11-11 on a simulated line: the drain deadlock, borrows starving the Blender card, fairness."""

    def test_no_deadlock_bakes_behind_full_borrow_line(self):
        # 14:05-14:17: three borrows waiting for the engine card, three assets whose bakes wait at card 05 and whose
        # engine stages wait for those bakes, the editor loaded. Borrows were held by the waiting engine stages and
        # the bakes by the engine's full line (of borrows): both cards idle for 12 minutes.
        S = self.sched()
        sim = Sim(self, S, {"shape": 6, "blender": 1, "ue_asset": 1, "finalize": 0})
        for i in range(3):
            self.assertIsNone(sim.borrow(f"shape{i}")[1])
        for i in range(3):
            self.assertIsNone(sim.asset(f"prop{i}")[1])
        S.editors = [FakeEditor(sim.card)]
        bl, ue = f"card{D.C.BLENDER_CARD}", sim.ue
        sim.step()
        self.assertEqual(sim.first(bl), 1, "the bake must start at once (the borrows wait in their own line)")
        for _ in range(3):
            sim.step()
        self.assertIsNotNone(sim.first(ue), "the engine card must start something within a few rounds")
        sim.run()

    def test_borrows_do_not_starve_the_blender_card(self):
        # the engine's line full of borrows (2-6 min each) must not hold two bakes waiting at card 05
        S = self.sched()
        sim = Sim(self, S, {"shape": 8, "blender": 1, "ue_asset": 1, "finalize": 0})
        for i in range(3):
            sim.borrow(f"shape{i}")
        sim.step()
        self.assertIsNotNone(sim.first(sim.ue))
        self.assertIsNone(sim.borrow("shape3")[1])            # one runs, three wait: the borrows' line is full
        self.assertIn("line full", sim.borrow("shape4")[1])
        a, _ = sim.asset("propA")
        b, _ = sim.asset("propB")
        sim.step()
        sim.step()
        bl = f"card{D.C.BLENDER_CARD}"
        self.assertIsNotNone(sim.first(bl, a.seq), "a bake waited behind the borrows")
        sim.step()
        sim.step()
        self.assertIsNotNone(sim.first(bl, b.seq))
        sim.run()

    def test_editor_work_runs_between_borrows(self):
        # imports and stills (seconds) arriving while a borrow runs go before the next borrow
        S = self.sched()
        sim = Sim(self, S, {"shape": 5, "ue_script": 1, "finalize": 0})
        b0, _ = sim.borrow("shape0")
        b1, _ = sim.borrow("shape1")
        sim.step()
        self.assertEqual(sim.starts[0][2], b0.seq)
        s0, _ = sim.script("set0")
        s1, _ = sim.script("set1")
        sim.run()
        order = [q for _, b, q, _ in sim.starts if b == sim.ue]
        self.assertEqual(order, [b0.seq, s0.seq, s1.seq, b1.seq], order)

    def test_a_borrow_gets_its_turn_under_steady_editor_work(self):
        # editor work that never runs out: the borrow goes after BATCH editor jobs
        S = self.sched()
        sim = Sim(self, S, {"shape": 2, "ue_script": 0, "finalize": 0})
        sim.script("set_first")
        S.editors = [FakeEditor(sim.card)]
        sim.step()
        b, _ = sim.borrow("shape0")
        n = 0
        while sim.first(sim.ue, b.seq) is None and sim.r < 60:
            if sum(1 for j in S.jobs if j.mode == "ue_job") < 2:
                sim.script(f"set{n}")
                n += 1
            sim.step()
        self.assertIsNotNone(sim.first(sim.ue, b.seq), "the borrow never got the card")
        before = [q for r, st, q, nm in sim.starts if st == sim.ue and r < sim.first(sim.ue, b.seq)]
        self.assertLessEqual(len(before), L.BATCH + 1)
        sim.run()

    def test_no_editor_ping_pong_for_a_waiting_bake(self):
        # a borrow waiting while an asset bakes: no editor is loaded for the bake's engine stage only to be unloaded
        S = self.sched()
        sim = Sim(self, S, {"shape": 3, "blender": 4, "ue_asset": 1, "finalize": 0})
        sim.asset("propA")
        sim.borrow("shape0")
        loads = 0
        while S.jobs and sim.r < 100:
            had = bool(S.editors)
            sim.step()
            loads += (not had and bool(S.editors))
        self.assertEqual(S.jobs, [])
        self.assertLessEqual(loads, 1)

    def test_status_says_why(self):
        S = self.sched()
        sim = Sim(self, S, {"shape": 3, "finalize": 0})
        sim.borrow("shape0")
        S.editors = [FakeEditor(sim.card, "busy")]          # still finishing something: the borrow must wait
        sim.card.blender = []
        S.editors[0].state = "quitting"
        S.advance(D.now())
        S.dispatch()
        st = S.line_status(D.now())["stations"][S.borrow_line()]
        self.assertTrue(any("unload" in w for w in st["why"]), st)
        S.verdict = {"cards": False, "pause": False, "why": ["/x/GPU_HOLD exists: no new card job"]}
        S.dispatch()
        st = S.line_status(D.now())["stations"][S.borrow_line()]
        self.assertTrue(any("GPU_HOLD" in w for w in st["why"]), st)
        S.verdict = {"cards": True, "pause": False, "why": []}
        S.reload = {"t": D.now(), "code": "x"}
        S.dispatch()
        st = S.line_status(D.now())["stations"][S.borrow_line()]
        self.assertTrue(any("reloading" in w for w in st["why"]), st)

    def test_reload_waits_for_running_jobs_and_starts_nothing(self):
        S = self.sched()
        sim = Sim(self, S, {"refs": 2, "finalize": 0})
        sim.bake_cmd("r0")
        sim.bake_cmd("r1")
        sim.step()
        execs = []
        S.exec_reload = lambda t: execs.append(t)
        S.reload = {"t": D.now(), "code": "new"}
        for _ in range(5):
            sim.step()
            S.maybe_reload(D.now())
        self.assertEqual([n for _, _, _, n in sim.starts], ["refs"])        # r1 did not start
        self.assertEqual(len(execs), 3)                    # each round once r0 ended (exec is faked here)
        self.assertEqual(len(S.jobs), 1)


class RestartTest(LineBase):
    """A command entry's end is durable: a restart right after the process ends never runs it again."""

    def cmd_entry(self, S, marks, sleep=0.0):
        code = f"import time; time.sleep({sleep}); open({str(marks)!r}, 'a').write('x\\n')"
        job, why = S.add_entry({"type": "cmd", "id": "once", "station": "igpu", "kind": "t",
                                "cmd": [sys.executable, "-c", code]})
        self.assertIsNone(why)
        return job

    def test_restart_right_after_the_process_ended(self):
        S = self.sched()
        marks = Path(tempfile.mkdtemp(dir=TMP)) / "marks"
        job = self.cmd_entry(S, marks)
        st = job.stages[0]
        st.state = "run"
        p = S.start_cmd(st)                                # the entry now says running, with the runner's pid
        p.wait()                                           # the command ended; the line "dies" before its bookkeeping
        e = json.loads((S.wl.edir / f"{job.seq:06d}.json").read_text())
        self.assertEqual(e["stages"], {"t": "run"})
        S2 = self.sched()
        S2.wl = L.WorkList(S.wl.root)
        D.resume(S2)
        self.assertEqual(S2.jobs, [])
        self.assertEqual(S2.started, [])
        done = json.loads((S.wl.ddir / f"{job.seq:06d}.json").read_text())
        self.assertEqual((done["state"], done["result"]["rc"]), ("done", 0))
        self.assertEqual(marks.read_text(), "x\n")

    def test_restart_while_it_runs_adopts_it(self):
        S = self.sched()
        marks = Path(tempfile.mkdtemp(dir=TMP)) / "marks"
        job = self.cmd_entry(S, marks, sleep=1.5)
        st = job.stages[0]
        st.state = "run"
        p = S.start_cmd(st)
        S2 = self.sched(fake=False)
        S2.wl = L.WorkList(S.wl.root)
        S2.guard.update = lambda *a: {}
        D.resume(S2)
        self.assertEqual([x.stages[0].state for x in S2.jobs], ["run"])   # adopted, not re-queued
        t0 = time.time()
        while S2.jobs and time.time() - t0 < 20:
            time.sleep(0.1)
        p.wait()
        self.assertEqual(S2.jobs, [])
        self.assertEqual(marks.read_text(), "x\n")
        done = json.loads((S.wl.ddir / f"{job.seq:06d}.json").read_text())
        self.assertEqual(done["state"], "done")

    def test_a_reused_pid_is_not_killed(self):
        S = self.sched()
        marks = Path(tempfile.mkdtemp(dir=TMP)) / "marks"
        job = self.cmd_entry(S, marks)
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        try:
            e = job.entry
            e.update(state="running", stages={"t": "run"}, pids={"t": other.pid}, pid_start={"t": 1},
                     rc_files={"t": str(job.adir / "t.rc.json")})
            S.wl.save(e)
            S2 = self.sched()
            S2.wl = L.WorkList(S.wl.root)
            D.resume(S2)
            self.assertIsNone(other.poll(), "a process that only shares the pid was stopped")
            self.assertEqual([x.stages[0].state for x in S2.jobs], ["wait"])    # it runs again
        finally:
            other.kill()
            other.wait()


class ReloadTest(unittest.TestCase):
    """A real dry-run line (its own folder, socket, fake cards and power supply; a copy of the line's code): a code
    change while it is busy reloads it within one job's length, every entry runs once, clients do not notice."""

    JOB_S = 1.0

    def test_reload_under_load(self):
        root = Path(tempfile.mkdtemp(dir=TMP))
        code = root / "code"
        code.mkdir()
        for f in D.CODE_FILES:
            shutil.copy(ASSETS / f, code / f)
        power = root / "power"
        fake_power(power)
        pci = root / "pci"
        for c in ("f0", "f1"):
            (pci / f"0000:{c}:00.0").mkdir(parents=True)
        (root / "aspm").write_text("default [performance] powersave\n")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("ASSETS_", "BLENDER"))}
        env.update(ASSETS_SCHED_DIR=str(root / "sched"), ASSETS_LOGS=str(root / "logs"),
                   ASSETS_LIBRARY=str(root / "lib"), ASSETS_UE_HOME=str(root / "ue"),
                   ASSETS_UE_PROJECT=str(root / "ue" / "P" / "P.uproject"), ASSETS_UE_CARD="f1",
                   ASSETS_BLENDER_CARD="f0", ASSETS_IGPU_BUS="fe", ASSETS_IGPU_LOCK=f"linetest{os.getpid()}",
                   ASSETS_IMAGE_STATION="igpu", ASSETS_POWER_ROOT=str(power), ASSETS_PCI_ROOT=str(pci),
                   ASSETS_ASPM_POLICY=str(root / "aspm"), ASSETS_GPU_HOLD=str(root / "no_hold"),
                   ASSETS_LINE_INHIBIT="0", ASSETS_SCHED_IDLE_S="3", ASSETS_LINE_MAX="10", ASSETS_BLENDER_SPARES="0")
        marks = root / "marks"

        def job(i, detach=True):
            body = f"import time; time.sleep({self.JOB_S}); open({str(marks)!r}, 'a').write('j{i}\\n')"
            return [sys.executable, str(code / "run_asset.py"), "--line-cmd", "igpu", "t", "--id", f"j{i}"] + (
                ["--detach"] if detach else []) + ["--", sys.executable, "-c", body]

        log = root / "logs" / f"assets_sched_{time.strftime('%m%d')}.log"
        n = 6
        daemon_pid = None
        try:
            r = subprocess.run(job(0), env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            attached = subprocess.Popen(job(1, detach=False), env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True)
            for i in range(2, n):
                r = subprocess.run(job(i), env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, r.stderr)
            r = subprocess.run(job(n - 1), env=env, capture_output=True, text=True, timeout=60)   # a 2nd submitter
            self.assertEqual(r.returncode, D.DUPLICATE_RC, r.stderr)
            self.assertIn("ALREADY QUEUED: already queued as entry", r.stderr)
            t0 = time.time()
            while not marks.exists() and time.time() - t0 < 30:  # the first job has run: the line is busy
                time.sleep(0.05)
            with open(code / "line.py", "a") as f:               # a code change under load
                f.write("\n# a newer line\n")
            t_change = time.time()
            t_reload = None
            while time.time() - t_change < 60:
                txt = log.read_text() if log.exists() else ""
                if t_reload is None and "reloading onto" in txt:
                    t_reload = time.time()
                if marks.exists() and len(marks.read_text().split()) >= n and t_reload:
                    break
                time.sleep(0.05)
            out, _ = attached.communicate(timeout=60)
            txt = log.read_text()
            serving = [ln for ln in txt.splitlines() if "serving on" in ln]
            daemon_pid = int(serving[0].split()[2])
            self.assertIsNotNone(t_reload, txt)
            self.assertLess(t_reload - t_change, self.JOB_S + 2.0 + 2.0, "reload took longer than one job")
            self.assertEqual(len(serving), 2, txt)
            self.assertIn("reloaded from", serving[1])
            self.assertEqual({ln.split()[2] for ln in serving}, {str(daemon_pid)})   # the same process, re-executed
            got = sorted(marks.read_text().split())
            self.assertEqual(got, sorted(f"j{i}" for i in range(n)), "every entry runs exactly once")
            self.assertEqual(attached.returncode, 0, out)          # the attached client did not notice
            self.assertNotIn("went away", out)
            starts = [ln for ln in txt.splitlines() if " start igpu " in ln]
            after = txt.split("reloading onto", 1)[1]
            self.assertTrue(any(" start igpu " in ln for ln in after.splitlines()), "nothing ran after the reload")
            self.assertEqual(len(starts), n, txt)
        finally:
            t0 = time.time()
            while daemon_pid and Path(f"/proc/{daemon_pid}").exists() and time.time() - t0 < 15:
                time.sleep(0.2)                                    # it exits by itself when idle (3 s)
            if daemon_pid and Path(f"/proc/{daemon_pid}").exists():
                os.kill(daemon_pid, signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
