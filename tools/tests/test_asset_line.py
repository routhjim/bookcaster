"""The asset line (tools/assets/line.py + assetd.py dispatch): admission, short lines, batching, the power guard,
and resume after a restart. Fast: no GPU, no UE, no Blender; the scheduler runs in a temporary folder."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="asset_line_test_")
os.environ["ASSETS_SCHED_DIR"] = TMP
os.environ["ASSETS_LIBRARY"] = os.path.join(TMP, "lib")
os.environ["ASSETS_LOGS"] = os.path.join(TMP, "logs")
os.environ["ASSETS_LINE_MAX"] = "3"
os.environ["ASSETS_LINE_BATCH"] = "8"
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


class DispatchTest(unittest.TestCase):
    """The scheduler's dispatch with fake runs: one job per station, batches, line full, resume."""

    def sched(self):
        root = tempfile.mkdtemp(dir=TMP)
        os.environ["ASSETS_SCHED_DIR"] = root
        D.C.SCHED_DIR = Path(root)
        S = D.Sched()
        D.SCHED = S
        S.wl = L.WorkList(Path(root) / "line")
        S.verdict = {"cards": True, "pause": False, "why": []}
        S.started = []

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


if __name__ == "__main__":
    unittest.main()
