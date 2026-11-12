"""The line's memory budget, stuck report, retries, fill work and the in-window rule (assetd.py / line.py).

Fake clock, fake memory reader, fake runs: no GPU, no engine, no model. Run with unittest:
    python -m unittest tools/tests/test_asset_line_mem.py
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_asset_line import LineBase, FakeEditor, TMP, D, L  # noqa: E402

C = D.C
GB = {"llm": 96, "video": 24, "big": 60, "small": 2, "refs": 24, "shape": 16, "ue_script": 0, "ue_asset": 0}


class MemBudgetTest(unittest.TestCase):
    def test_fits_and_ramp(self):
        m = L.MemBudget(GB, 4, reserve_gb=8, ramp_s=180, total_gb=123)
        self.assertEqual(m.need("llm"), 96)
        self.assertEqual(m.need("unknown"), 4)
        self.assertEqual(m.need("llm", declared=10), 10)
        self.assertTrue(m.fits(96, 110, [], 0)[0])
        ok, text = m.fits(96, 110, [(24, 0.0)], 10)          # a clip job started 10 s ago: its 24 are promised
        self.assertFalse(ok)
        self.assertIn("24 promised", text)
        self.assertFalse(m.fits(96, 86, [(24, 0.0)], 500)[0])   # after the ramp the reading itself shows it
        self.assertTrue(m.fits(24, 110, [(24, 0.0)], 10)[0])
        self.assertTrue(m.fits(96, 80, [], 0, reclaim=30)[0])   # an editor that unloads first
        self.assertTrue(m.fits(0, 0, [], 0)[0])

    def test_never(self):
        m = L.MemBudget(GB, 4, reserve_gb=8, ramp_s=180, total_gb=123)
        ok, text = m.fits(200, 500, [], 0)
        self.assertFalse(ok)
        self.assertIn("never", text)
        self.assertTrue(m.never(200))
        self.assertFalse(m.never(96))

    def test_busy_exit_and_delay(self):
        self.assertIsNone(L.busy_exit(0, "", (75,), ""))
        self.assertIn("75", L.busy_exit(75, "", (75,), ""))
        self.assertIsNone(L.busy_exit(1, "Traceback: boom", (75,), C.BUSY_LOG_RE))
        self.assertIsNotNone(L.busy_exit(1, "waiting for memory before starting the model: 90 free", (75,),
                                         C.BUSY_LOG_RE))
        self.assertIsNotNone(L.busy_exit(1, "the card lock is busy", (75,), C.BUSY_LOG_RE))
        self.assertIsNone(L.busy_exit(1, "the card lock is busy", (75,), ""))
        self.assertEqual([L.retry_delay(n, 60) for n in (1, 2, 3)], [60, 120, 240])
        # review: a job that ran for a while has done work (a script passing on the client's own 75; a job that
        # waited, ran, then failed for a real reason): never from the top again
        self.assertIsNotNone(L.busy_exit(75, "", (75,), "", run_s=30, max_run_s=600))
        self.assertIsNone(L.busy_exit(75, "", (75,), "", run_s=700, max_run_s=600))
        self.assertIsNone(L.busy_exit(1, "waiting for memory before starting", (75,), C.BUSY_LOG_RE, run_s=700,
                                      max_run_s=600))
        self.assertIsNone(L.busy_exit(75, "", (75,), "", run_s=1, max_run_s=600, kind="shape", kinds=("llm",)))
        self.assertIsNotNone(L.busy_exit(75, "", (75,), "", run_s=1, max_run_s=600, kind="llm", kinds=("llm",)))

    def test_only_the_last_log_lines_count(self):
        d = Path(tempfile.mkdtemp(dir=TMP)) / "x.log"
        d.write_text("waiting for memory before starting the model\n" + "work line\n" * 6 + "Traceback: real\n")
        self.assertIsNone(L.busy_exit(1, L.log_tail(d), (75,), C.BUSY_LOG_RE))
        d.write_text("work\n" * 6 + "waiting for memory before starting the model\nSTART FAILED\n")
        self.assertIsNotNone(L.busy_exit(1, L.log_tail(d), (75,), C.BUSY_LOG_RE))


class TimedBase(LineBase):
    """LineBase with a clock the test moves and a memory reading the test sets."""

    def setUp(self):
        self.t = 1_000_000.0
        self._now = D.now
        D.now = lambda: self.t
        self._cfg = {k: getattr(C, k) for k in ("MEM_BUDGET", "MEM_EDITOR_GB", "MEM_TURN_S", "MEM_SMALL_GB",
                                                "MEM_BIG_GB", "MEM_BIG_KEEP_GB", "MEM_EDITOR_CEIL_GB", "MEM_GATE_KINDS",
                                                "WINDOW_MEM_WAIT_S", "RETRY_MAX_RUN_S", "RETRY_KINDS", "FILL_MAX_S",
                                                "UE_RECYCLE_GB", "STUCK_REPORT_S", "RETRIES", "RETRY_S",
                                                "WINDOW_FILL", "WINDOW_IDLE_S", "WINDOW_JOB_MAX_S",
                                                "WINDOW_VRAM_KEEP_GB", "VRAM_KIND_GB", "FILL_LIST",
                                                "FILL_ENGINE_IDLE_S", "FILL_MAX_QUEUED", "RECLAIM_FREE_S")}
        C.MEM_BUDGET = True

    def tearDown(self):
        D.now = self._now
        for k, v in self._cfg.items():
            setattr(C, k, v)

    def msched(self, avail=110.0, total=123.0):
        S = self.sched()
        self.avail = avail
        S.mem = L.MemBudget(GB, 4, reserve_gb=8, ramp_s=180, total_gb=total)
        S.mem_read = lambda: self.avail
        S.outside_holders = lambda c: []                    # never /proc/locks of the box
        return S

    def cmd(self, S, name, station, kind, **kw):
        job, why = S.add_entry({"type": "cmd", "id": name, "station": station, "kind": kind, "cmd": ["true"], **kw})
        self.assertIsNone(why)
        return job

    def ue_job(self, S, name):
        root = Path(tempfile.mkdtemp(dir=TMP))
        for f in ("p.uproject", "s.py"):
            (root / f).write_text("")
        job, why = S.add_entry({"type": "ue_job", "id": name, "ue": {"project": str(root / "p.uproject"),
                                                                    "script": str(root / "s.py"),
                                                                    "log_dir": str(root / "log")}})
        self.assertIsNone(why)
        return job

    def step(self, S, dt=0.0):
        self.t += dt
        with S.lock:
            S.advance(self.t)
        S.dispatch()

    def running(self, S):
        return sorted(j.id for j in S.jobs for st in j.stages if st.state == "run")

    def end(self, S, name):
        for j in list(S.jobs):
            if j.id == name:
                for st in j.stages:
                    if st.state == "run":
                        st.state = "done"
                        S.after_stage(st)


class MemoryTest(TimedBase):
    def test_language_model_and_clips_never_overlap(self):
        # 11-12 09:41: the language model (about 90 GiB) started while a clip window rendered; both failed
        S = self.msched(avail=110)
        self.cmd(S, "llm0", "igpu", "llm")
        self.cmd(S, "clips0", "video", "video")
        self.step(S)
        self.assertEqual(self.running(S), [])               # the window opens; its clips' memory is promised
        self.assertTrue(S.released and "video window" in S.released["reason"])
        self.assertIn("promised", " ".join(S.why["igpu"]))
        S.released["done_t"] = self.t
        self.step(S)
        self.assertEqual(self.running(S), ["clips0"])
        self.avail = 86                                     # the clips hold their memory now; the ramp is over
        self.step(S, 400)
        self.assertEqual(self.running(S), ["clips0"])
        self.end(S, "clips0")
        self.avail = 110
        self.step(S)
        self.assertEqual(self.running(S), ["llm0"])
        self.cmd(S, "clips1", "video", "video")             # the other order: the model first
        self.step(S, 10)
        self.assertEqual(self.running(S), ["llm0"])
        self.assertTrue(any("memory" in x for x in S.why.get("video", [])))
        self.avail = 14
        self.step(S, 400)
        self.assertEqual(self.running(S), ["llm0"])
        self.end(S, "llm0")
        self.avail = 110
        self.step(S)
        self.assertEqual(self.running(S), ["clips1"])

    def test_no_window_is_opened_for_clips_that_cannot_load(self):
        S = self.msched(avail=110)
        self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.assertEqual(self.running(S), ["llm0"])
        self.cmd(S, "clips0", "video", "video")
        self.cmd(S, "draw0", "blender", "small")
        self.avail = 40                                     # the model is loaded (the ramp is over below)
        self.step(S, 400)
        self.assertIsNone(S.released, "the cards must keep their work while the clips could not load")
        self.assertIn("draw0", self.running(S))

    def test_gate_kind_needs_no_reserve_on_top(self):
        S = self.msched(avail=96)                           # exactly the language model tool's own gate
        self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.assertEqual(self.running(S), ["llm0"])

    def test_budget_is_off_by_default_and_memory_is_logged(self):
        self.assertFalse(self._cfg["MEM_BUDGET"])           # a deploy reloads with the budget off
        S = self.msched(avail=77.3)
        self.assertEqual(S.power_note()["mem_avail_gb"], 77.3)

    def test_model_does_not_start_beside_a_live_editor_and_nothing_creeps_in_beside_it(self):
        # review: the editor is 6 GiB at the start and grows to 20-30; 180 s later a small job could start
        S = self.msched(avail=117)
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start, ed.idle_since = self.t - 4000, self.t - 500
        S.editors = [ed]
        S.editor_mem = lambda e: 6.0 if e.state != "dead" else 0.0
        llm = self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.assertEqual(self.running(S), [])               # 117 - 24 (the editor's growth) < 96
        self.assertIn("editor's growth", " ".join(S.why["igpu"]))
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "quitting")              # a small idle editor too: nothing else would unload it
        ed.state = "dead"
        S.editors = []
        self.avail = 123
        self.step(S, 5)
        self.assertEqual(self.running(S), ["llm0"])
        ue = self.ue_job(S, "set0")
        small = self.cmd(S, "s0", "blender", "small")
        self.avail = 15
        self.step(S, 400)                                   # long after the ramp
        self.assertEqual(self.running(S), ["llm0"])
        self.assertEqual(ue.stages[0].held[0], "running")   # no engine work beside a big job
        self.assertIn("llm0", ue.stages[0].held[1])
        self.assertEqual(small.stages[0].held[0], "memory")
        made = []
        orig = D.Editor
        D.Editor = lambda *a, **k: made.append(a) or (_ for _ in ()).throw(AssertionError("an editor was started"))
        try:
            with S.lock:
                S.manage_editors()
            self.assertIn("llm0", S.editor_block)
        finally:
            D.Editor = orig
        self.assertIsNotNone(llm)

    def test_turn_unloads_a_small_idle_editor(self):
        # review: 101 GiB available, the editor idle at 6 GiB: the turn held engine work, nothing unloaded the editor
        S = self.msched(avail=45)
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start, ed.idle_since = self.t - 4000, self.t - 4000
        S.editors = [ed]
        S.editor_mem = lambda e: 6.0
        S.place_ue = lambda st: None
        job = self.cmd(S, "mid0", "igpu", "big", mem_gb=30)   # not a big job: 45 - 24 growth - 8 = 13 < 30
        self.step(S)
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "ready")                 # under UE_RECYCLE_GB and no turn yet
        ue = self.ue_job(S, "set0")
        self.step(S, C.MEM_TURN_S + 1)
        self.assertIs(S.mem_turn, job.stages[0])
        self.assertTrue(S.mem_turn_editor)
        S.dispatch()
        self.assertEqual(ue.stages[0].held[0], "turn")
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "quitting")              # the turn's own rule unloads it, whatever its size
        ed.state = "ready"                                  # and again, should it ever be back while the turn stands
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "quitting")

    def test_a_turn_the_editor_cannot_honour_is_named_as_dead(self):
        S = self.msched(avail=20)
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start = self.t - 4000
        S.editors = [ed]
        S.editor_mem = lambda e: 30.0
        job = self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        st = job.stages[0]
        S.mem_turn, S.mem_turn_editor = st, True            # (put on purpose: the grant rule itself refuses this)
        text, ok = S.unmet(st, self.t)
        self.assertFalse(ok)
        self.assertIn("would not cover", text)
        self.step(S, C.MEM_TURN_S + 1)
        self.assertIsNone(S.mem_turn)                       # 20 + 30 < 96: no turn is granted

    def test_clips_wanted_promise_their_memory(self):
        S = self.msched(avail=110)
        self.cmd(S, "draw0", "blender", "small")
        self.step(S)                                        # a card job runs: the window cannot open yet
        self.cmd(S, "clips0", "video", "video")
        self.cmd(S, "llm0", "igpu", "llm")
        self.step(S, 1)
        self.assertEqual(self.running(S), ["draw0"])        # the model must not take what the clips will need
        self.assertIn("promised", " ".join(S.why["igpu"]))

    def test_window_whose_clips_cannot_load_gives_the_cards_back(self):
        S = self.msched(avail=110)
        self.cmd(S, "clips0", "video", "video")
        self.step(S)
        S.released["done_t"] = self.t
        self.avail = 20                                     # something outside the line took the memory
        draw = self.cmd(S, "draw0", "blender", "small")
        self.step(S, 1)
        self.assertEqual(self.running(S), [])
        self.assertEqual(draw.stages[0].held[0], "video")
        S.t_probe = 0.0
        with S.lock:
            S.update_mode(self.t)
        self.assertIsNotNone(S.released)
        self.t += C.WINDOW_MEM_WAIT_S + 1
        S.dispatch()
        S.t_probe = 0.0
        with S.lock:
            S.update_mode(self.t)
        self.assertIsNone(S.released)                       # both cards idle for clips that cannot load: no more
        self.step(S)
        self.assertEqual(self.running(S), ["draw0"])
        self.assertIsNone(S.released)                       # and the window is not opened again until they fit
        self.end(S, "draw0")
        self.avail = 110
        self.step(S)
        self.assertTrue(S.released)

    def test_two_stations_in_one_pass_do_not_overcommit(self):
        S = self.msched(avail=100)
        self.cmd(S, "a", "igpu", "big")
        self.cmd(S, "b", "blender", "big")
        self.step(S)
        self.assertEqual(len(self.running(S)), 1)           # 100 - 8 covers one 60, not two
        held = [x for v in S.why.values() for x in v]
        self.assertTrue(any("memory" in x and "promised" in x for x in held), held)

    def test_entry_can_declare_its_own_need(self):
        S = self.msched(avail=30)
        self.cmd(S, "x", "igpu", "big", mem_gb=10)
        self.step(S)
        self.assertEqual(self.running(S), ["x"])

    def test_budget_off(self):
        C.MEM_BUDGET = False
        S = self.msched(avail=1)
        self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.assertEqual(self.running(S), ["llm0"])

    def test_idle_editor_is_recycled_once_for_a_waiting_job(self):
        # 11-12 03:00: the idle editor held about 30 GiB and the language model could not start
        S = self.msched(avail=80)
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start, ed.idle_since = self.t - 4000, self.t - 2000
        S.editors = [ed]
        S.editor_mem = lambda e: 30.0 if e.state != "dead" else 0.0
        job = self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.assertEqual(self.running(S), [])
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "quitting")
        self.assertTrue(job.stages[0].info["editor_recycled"])
        ed.state = "ready"                                  # (as if it had come back) no second unload for this job
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "ready")
        ed.state = "dead"
        self.avail = 110
        self.step(S, 5)
        self.assertEqual(self.running(S), ["llm0"])

    def test_small_editor_is_not_recycled(self):
        S = self.msched(avail=40)
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start = self.t - 4000
        S.editors = [ed]
        S.editor_mem = lambda e: 5.0                        # under UE_RECYCLE_GB: unloading it is not worth a reload
        self.cmd(S, "refs0", "igpu", "refs")                # neither big nor with the turn
        self.step(S)
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "ready")

    def test_memory_turn_stops_new_engine_and_large_work(self):
        S = self.msched(avail=80)
        ed = FakeEditor(S.cards[C.UE_CARD], state="busy")
        ed.t_start = self.t - 4000
        S.editors = [ed]
        S.editor_mem = lambda e: 30.0
        S.place_ue = lambda st: None
        llm = self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.assertIsNone(S.mem_turn)                       # not yet: it has not waited MEM_TURN_S
        ue = self.ue_job(S, "set0")
        big = self.cmd(S, "refs0", "blender", "refs")
        small = self.cmd(S, "s0", "blender", "small")
        for j in (ue, big, small):
            j.stages[0].state, j.stages[0].t_ready = "wait", None
        self.step(S, C.MEM_TURN_S + 1)                      # this pass grants the turn
        self.assertIs(S.mem_turn, llm.stages[0])
        self.assertTrue(S.mem_turn_editor)
        for j in (ue, big, small):
            j.stages[0].state, j.stages[0].t_ready = "ready", self.t
        S.dispatch()
        self.assertEqual(self.running(S), ["s0"])           # a small job goes on; engine and large work wait
        self.assertEqual(ue.stages[0].held[0], "turn")
        self.assertEqual(big.stages[0].held[0], "turn")
        self.assertIn("#%d llm0" % llm.seq, ue.stages[0].held[1])

    def test_no_turn_for_memory_held_outside_the_line(self):
        S = self.msched(avail=20)                           # something outside the line holds the memory
        self.cmd(S, "llm0", "igpu", "llm")
        self.step(S)
        self.step(S, C.MEM_TURN_S + 1)
        self.assertIsNone(S.mem_turn)

    def test_job_that_can_never_fit_gets_no_turn_and_is_named(self):
        S = self.msched(avail=110)
        C.STUCK_REPORT_S = 60
        job = self.cmd(S, "huge", "igpu", "big", mem_gb=500)
        self.step(S)
        self.assertIn("never", " ".join(S.why["igpu"]))
        self.step(S, C.MEM_TURN_S + 1)
        self.assertIsNone(S.mem_turn)
        S.check_stuck(self.t)
        self.assertIsNone(S.stuck)                          # first sight: the clock starts
        self.t += 61
        S.check_stuck(self.t)
        self.assertTrue(S.stuck["dead"])
        self.assertIn(f"#{job.seq} huge", S.stuck["text"])

    def test_no_editor_start_the_budget_does_not_cover(self):
        S = self.msched(avail=15)                           # e.g. the language model is loaded
        self.ue_job(S, "set0")
        made = []
        orig = D.Editor
        D.Editor = lambda *a, **k: made.append(a) or (_ for _ in ()).throw(AssertionError("an editor was started"))
        try:
            for c in S.cards.values():
                c.free_gb = lambda: 0.0                     # never a real editor from a test, whatever happens
            with S.lock:
                S.manage_editors()
            self.assertEqual(made, [])
            self.assertIn("the editor needs", S.editor_block)
            self.assertIn("memory", S.editor_brief())
        finally:
            D.Editor = orig


class ReclaimTest(TimedBase):
    """After a video window the cards come back for whatever waits at a card station."""

    def released_after_window(self, S):
        S.released = {"reason": "video window: 1 clip job(s) waiting", "manual": False, "t": self.t - 600,
                      "done_t": self.t - 590, "card": None}

    def reclaims(self, S):
        S.t_probe = 0.0
        with S.lock:
            S.update_mode(self.t)
            self.t += C.RECLAIM_FREE_S + 1
            S.t_probe = 0.0
            S.update_mode(self.t)
        return S.released is None

    def test_only_line_commands_wait_at_a_card(self):
        # 11-12 07:27-08:02: three commands waited at card 05 after a window; nothing counted as card work, the
        # cards were never taken back, and the commands waited for exactly that (35 min, both cards idle)
        S = self.msched()
        self.released_after_window(S)
        self.cmd(S, "draw0", "blender", "refs")
        self.cmd(S, "measure0", "blender", "measure")
        self.assertTrue(self.reclaims(S))

    def test_a_borrow_command_waits(self):
        S = self.msched()
        self.released_after_window(S)
        self.cmd(S, "shape0", "engine", "shape")
        self.assertTrue(self.reclaims(S))

    def test_a_preview_or_a_paint_stage_waits(self):
        for mode in ("preview", "paint"):
            S = self.msched()
            self.released_after_window(S)
            job = self.cmd(S, "x", "igpu", "small")         # any job; give it the stage under test
            st = D.Stage(job, "blender_preview", "blender") if mode == "preview" else D.Stage(job, "paint", "paint")
            job.stages = [st]
            if mode == "paint":
                self.assertTrue(S.station(st).startswith("card"), "the image station is a card in this config")
            self.assertTrue(self.reclaims(S), mode)

    def test_igpu_work_alone_does_not_take_the_cards_back(self):
        S = self.msched()
        self.released_after_window(S)
        self.cmd(S, "combo0", "igpu", "small")
        self.assertFalse(self.reclaims(S))

    def test_no_reclaim_while_a_clip_job_is_left(self):
        S = self.msched()
        self.released_after_window(S)
        self.cmd(S, "draw0", "blender", "refs")
        self.cmd(S, "clips1", "video", "video")
        self.assertFalse(self.reclaims(S))

    def test_release_not_complete_while_our_command_runs_on_a_card(self):
        S = self.msched()
        S.released = {"reason": "an outside job holds card", "manual": False, "t": self.t, "done_t": None,
                      "card": None}
        job = self.cmd(S, "draw0", "blender", "refs")
        S.cards[C.BLENDER_CARD].cmd_busy = job.stages[0]
        with S.lock:
            S.update_mode(self.t)
        self.assertIsNone(S.released["done_t"])             # clips must not start beside it
        S.cards[C.BLENDER_CARD].cmd_busy = None
        with S.lock:
            S.update_mode(self.t)
        self.assertIsNotNone(S.released["done_t"])


class StuckTest(TimedBase):
    def test_the_reclaim_stall_is_named_within_a_minute(self):
        # the 11-12 fault put back on purpose: no waiting job "counts" as card work
        S = self.msched()
        C.STUCK_REPORT_S = 60
        S.released = {"reason": "video window: 1 clip job(s) waiting", "manual": False, "t": self.t - 600,
                      "done_t": self.t - 590, "card": None}
        S.card_work_queued = lambda fill=False: False
        job = self.cmd(S, "draw0", "blender", "refs")
        self.step(S)
        self.assertEqual(job.stages[0].held[0], "released")
        S.check_stuck(self.t)
        self.t += 61
        self.step(S)
        S.check_stuck(self.t)
        self.assertTrue(S.stuck and S.stuck["dead"])
        text = S.stuck["text"]
        self.assertIn(f"#{job.seq} draw0", text)
        self.assertIn("cards released", text)
        self.assertIn("no waiting job counts as card work", text)
        self.assertIn("NOTHING IN THE LINE WILL CHANGE THIS", text)

    def test_waiting_on_the_outside_is_reported_but_not_as_dead(self):
        S = self.msched()
        C.STUCK_REPORT_S = 60
        S.verdict = {"cards": False, "pause": False, "why": ["lost charger since 12:00"]}
        self.cmd(S, "draw0", "blender", "refs")
        self.step(S)
        S.check_stuck(self.t)
        self.t += 61
        S.check_stuck(self.t)
        self.assertFalse(S.stuck["dead"])
        self.assertIn("guard", S.stuck["text"])

    def test_nothing_reported_while_something_runs_or_an_editor_loads(self):
        S = self.msched()
        C.STUCK_REPORT_S = 60
        S.verdict = {"cards": False, "pause": False, "why": ["lost charger"]}
        self.cmd(S, "draw0", "blender", "refs")
        self.cmd(S, "combo0", "igpu", "small")
        self.step(S)
        self.assertEqual(self.running(S), ["combo0"])
        S.check_stuck(self.t)
        self.t += 500
        S.check_stuck(self.t)
        self.assertIsNone(S.stuck)
        self.end(S, "combo0")
        ed = FakeEditor(S.cards[C.UE_CARD], state="starting")
        S.editors = [ed]
        S.check_stuck(self.t)
        self.t += 500
        S.check_stuck(self.t)
        self.assertIsNone(S.stuck)

    def test_one_stuck_station_is_named_while_others_work(self):
        # review: the whole-line test misses a partial stall
        S = self.msched(avail=110)
        C.STUCK_REPORT_S = 60
        job = self.cmd(S, "huge", "blender", "big", mem_gb=500)
        self.cmd(S, "combo0", "igpu", "small")
        self.step(S)
        self.assertEqual(self.running(S), ["combo0"])
        self.t += 61
        S.dispatch()
        S.check_stuck(self.t)
        self.assertIsNone(S.stuck)                          # something runs: not a whole-line stall
        line = "card" + C.BLENDER_CARD
        self.assertIn(f"#{job.seq} huge", S.stuck_at[line])
        self.assertIn("NOTHING IN THE LINE", S.stuck_at[line])
        self.assertEqual(S.line_status(self.t)["stuck_at"], S.stuck_at)

    def test_a_stage_with_no_recorded_hold_is_dead(self):
        S = self.msched()
        job = self.cmd(S, "x", "igpu", "small")
        job.stages[0].state = "ready"
        text, ok = S.unmet(job.stages[0], self.t)
        self.assertFalse(ok)
        self.assertIn("no hold", text)

    def test_a_chain_is_followed_to_its_end(self):
        S = self.msched(avail=110)
        llm = self.cmd(S, "huge", "igpu", "big", mem_gb=500)
        self.step(S)
        other = self.cmd(S, "y", "blender", "refs")
        other.stages[0].state = "ready"
        other.stages[0].held = ("turn", "memory: #1 goes first", None)
        S.mem_turn = llm.stages[0]
        text, ok = S.unmet(other.stages[0], self.t)
        self.assertFalse(ok)                                # it waits behind a job that can never start
        self.assertIn("huge", text)
        self.assertIn("never", text)
        llm.stages[0].held = ("turn", "x", None)
        text, ok = S.unmet(llm.stages[0], self.t)           # a chain back to itself
        self.assertFalse(ok)
        self.assertIn("cycle", text)


class RetryTest(TimedBase):
    def test_requeue_backoff_and_limit(self):
        S = self.msched()
        C.RETRIES, C.RETRY_S = 2, 60
        job = self.cmd(S, "llm0", "igpu", "small")
        st = job.stages[0]
        self.step(S)
        self.assertEqual(st.state, "run")
        with S.lock:
            self.assertTrue(S.requeue(st, "rc 75"))
        self.assertEqual((st.state, st.info["retries"]), ("wait", 1))
        self.assertEqual(job.entry["retries"]["small"][0], 1)
        self.step(S, 30)
        self.assertEqual(st.state, "wait")                  # not before its time
        self.assertIn("retries in", " ".join(S.line_status(self.t)["stations"]["igpu"]["why"]))
        self.step(S, 31)
        self.assertEqual(st.state, "run")
        with S.lock:
            self.assertTrue(S.requeue(st, "rc 75"))
        self.assertAlmostEqual(st.info["retry_at"] - self.t, 120)
        self.step(S, 121)
        with S.lock:
            self.assertFalse(S.requeue(st, "rc 75"))        # used up: the caller fails it

    def test_retry_count_survives_a_restart(self):
        S = self.msched()
        job = self.cmd(S, "llm0", "igpu", "small")
        st = job.stages[0]
        st.state = "run"
        st.info["rc_file"] = str(job.adir / "small.rc.json")
        Path(st.info["rc_file"]).write_text(json.dumps({"rc": 75, "t": 0}))
        with S.lock:
            S.requeue(st, "rc 75")
        self.assertFalse(Path(job.adir / "small.rc.json").exists())
        e = json.loads((S.wl.edir / f"{job.seq:06d}.json").read_text())
        self.assertNotIn("rc_files", e)
        S2 = self.msched()
        S2.wl = L.WorkList(S.wl.root)
        D.resume(S2)
        st2 = S2.jobs[0].stages[0]
        self.assertEqual((st2.state, st2.info["retries"]), ("wait", 1))   # waits again; not finished as a failure

    def run_real(self, code, log_text=None):
        D.now = self._now                                   # real time for a real process
        S = self.sched(fake=False)
        S.guard.update = lambda *a: {}
        C.RETRIES, C.RETRY_S = 1, 3600
        log = Path(tempfile.mkdtemp(dir=TMP)) / "cmd.log"
        py = f"import sys; print({log_text!r}); sys.exit({code})"
        job, why = S.add_entry({"type": "cmd", "id": "busy", "station": "igpu", "kind": "small",
                                "cmd": [sys.executable, "-c", py], "log": str(log)})
        st = job.stages[0]
        st.state, st.t_ready = "ready", D.now()
        with S.lock:
            S.launch(st, S.run_cmd, None)
        t0 = time.time()
        while st.state == "run" and time.time() - t0 < 20:
            time.sleep(0.05)
        time.sleep(0.2)
        return S, job, st

    def test_exit_75_goes_back_on_the_line(self):
        S, job, st = self.run_real(75, "x")
        self.assertEqual((st.state, st.info.get("retries")), ("wait", 1))
        self.assertIn(job, S.jobs)

    def test_a_tool_that_exits_1_after_waiting_for_memory(self):
        S, job, st = self.run_real(1, "waiting for memory before starting the model: 90 GiB free, it needs 96")
        self.assertEqual((st.state, st.info.get("retries")), ("wait", 1))

    def test_a_real_failure_is_a_failure(self):
        S, job, st = self.run_real(1, "Traceback: something else")
        self.assertEqual(st.state, "fail")
        self.assertNotIn(job, S.jobs)


class FillTest(TimedBase):
    def test_fill_runs_only_where_no_normal_work_is(self):
        S = self.msched()
        f = self.cmd(S, "fill0", "blender", "shape", fill=True)
        n = self.cmd(S, "draw0", "blender", "refs")
        self.step(S)
        self.assertEqual(self.running(S), ["draw0"])        # normal first, though the fill job is older
        self.assertEqual(f.stages[0].held[0], "fill")
        self.end(S, "draw0")
        self.step(S)
        self.assertEqual(self.running(S), ["fill0"])
        late = self.cmd(S, "draw1", "blender", "refs")      # normal work arrives: the fill job ends, no new fill
        f2 = self.cmd(S, "fill1", "blender", "shape", fill=True)
        self.step(S)
        self.assertEqual(self.running(S), ["fill0"])        # never stopped mid-job
        self.end(S, "fill0")
        self.step(S)
        self.assertEqual(self.running(S), ["draw1"])
        self.assertEqual(f2.stages[0].held[0], "fill")
        self.assertIsNotNone(late)

    def test_a_normal_stage_still_waiting_for_its_earlier_stage_counts(self):
        S = self.msched()
        f = self.cmd(S, "fill0", "blender", "shape", fill=True)
        n = self.cmd(S, "draw0", "blender", "refs")
        n.stages[0].state = "wait"
        f.stages[0].state, f.stages[0].t_ready = "ready", self.t
        S.dispatch()
        self.assertEqual(self.running(S), [])

    def test_a_long_fill_job_never_starts(self):
        S = self.msched()
        f = self.cmd(S, "fill0", "blender", "shape", fill=True, est_s=C.FILL_MAX_S + 100)
        ok = self.cmd(S, "fill1", "igpu", "shape", fill=True, est_s=60)
        self.step(S)
        self.assertEqual(self.running(S), ["fill1"])
        self.assertIn("expected to take", f.stages[0].held[1])
        self.assertIsNotNone(ok)

    def test_fill_never_counts_for_line_full(self):
        S = self.msched()
        for i in range(10):
            self.cmd(S, f"fill{i}", "blender", "shape", fill=True)
        self.assertEqual(S.heading().get("card" + C.BLENDER_CARD, 0), 0)
        for i in range(3):
            self.cmd(S, f"draw{i}", "blender", "refs")      # the normal line still has all its places
        _, why = S.add_entry({"type": "cmd", "id": "draw9", "station": "blender", "kind": "refs", "cmd": ["true"]})
        self.assertIn("line full", why)
        C.FILL_MAX_QUEUED = 10
        _, why = S.add_entry({"type": "cmd", "id": "fillx", "station": "blender", "kind": "shape", "cmd": ["true"],
                              "fill": True})
        self.assertIn("fill", why)

    def test_fill_waits_for_a_video_window_and_brings_the_cards_back_after_it(self):
        S = self.msched()
        f = self.cmd(S, "fill0", "blender", "shape", fill=True)
        self.cmd(S, "clips0", "video", "video")
        self.step(S)
        self.assertEqual(self.running(S), [])
        self.assertIn(f.stages[0].held[0], ("video", "released"))
        self.assertTrue(S.released)
        S.released["done_t"] = self.t
        self.step(S)
        self.assertEqual(self.running(S), ["clips0"])
        self.end(S, "clips0")
        S.t_probe = 0.0
        with S.lock:
            S.update_mode(self.t)
            self.t += C.RECLAIM_FREE_S + 1
            S.t_probe = 0.0
            S.update_mode(self.t)
        self.assertIsNone(S.released)                       # fill work alone takes the cards back after a window
        self.assertFalse(S.warm_wanted)                     # ... without loading the editor
        self.step(S)
        self.assertEqual(self.running(S), ["fill0"])

    def test_fill_does_not_take_back_cards_released_by_hand(self):
        S = self.msched()
        C.STUCK_REPORT_S = 60
        S.released = {"reason": "asked by client", "manual": True, "t": self.t - 5000, "done_t": self.t - 4990,
                      "card": None}
        self.cmd(S, "fill0", "blender", "shape", fill=True)
        S.t_probe = 0.0
        with S.lock:
            S.update_mode(self.t)
            self.t += 60
            S.t_probe = 0.0
            S.update_mode(self.t)
        self.assertIsNotNone(S.released)
        self.step(S)
        S.check_stuck(self.t)
        self.t += 61
        S.check_stuck(self.t)
        self.assertFalse(S.stuck["dead"])                   # a wait on the owner, said as such
        self.assertIn("by hand", S.stuck["text"])

    def test_fill_borrows_the_engine_card_only_from_a_long_idle_editor(self):
        S = self.msched()
        C.FILL_ENGINE_IDLE_S = 120
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start, ed.idle_since = self.t - 4000, self.t - 10
        S.editors = [ed]
        f = self.cmd(S, "fill0", "engine", "shape", fill=True)
        self.step(S)
        self.assertEqual(ed.state, "ready")                 # not unloaded for fill work after 10 s of idle
        self.assertEqual(f.stages[0].held[0], "fill")
        self.step(S, 200)
        self.assertEqual(ed.state, "quitting")              # idle long enough: the borrow may unload it
        ed.state = "dead"
        self.step(S)
        self.assertEqual(self.running(S), ["fill0"])

    def test_fill_gets_no_memory_turn_and_no_editor_recycle(self):
        S = self.msched(avail=80)
        ed = FakeEditor(S.cards[C.UE_CARD])
        ed.t_start, ed.idle_since = self.t - 4000, self.t - 4000
        S.editors = [ed]
        S.editor_mem = lambda e: 30.0
        self.cmd(S, "fill0", "igpu", "llm", fill=True)
        self.step(S)
        self.step(S, C.MEM_TURN_S + 5)
        self.assertIsNone(S.mem_turn)
        S.mem_relief(self.t)
        self.assertEqual(ed.state, "ready")


class FillListTest(TimedBase):
    def lst(self, lines):
        p = Path(tempfile.mkdtemp(dir=TMP)) / "fill.list"
        p.write_text("\n".join(lines) + "\n")
        C.FILL_LIST = p
        return p

    def test_parse_and_check(self):
        d, why = L.parse_cmd_line("--line-cmd engine shape --id a --mem-gb 12 --est-s 90 -- true x")
        self.assertIsNone(why)
        self.assertEqual((d["station"], d["kind"], d["id"], d["mem_gb"], d["est_s"], d["cmd"]),
                         ("engine", "shape", "a", 12.0, 90.0, ["true", "x"]))
        self.assertIsNone(L.parse_cmd_line("SPEC MODEL --ue-only")[0])
        self.assertIsNone(L.parse_cmd_line("--line-cmd igpu refs --")[0])
        d, probs = L.check_cmd_line("--line-cmd nowhere shape -- true", D.STATION_ALIASES)
        self.assertIn("not a station", probs[0])
        d, probs = L.check_cmd_line("--line-cmd video video -- true", D.STATION_ALIASES)
        self.assertIn("not a fill station", probs[0])
        d, probs = L.check_cmd_line("--line-cmd igpu refs -- env A=/no/such/dir python /no/such/tool.py",
                                    D.STATION_ALIASES)
        self.assertEqual(len(probs), 2)
        d, probs = L.check_cmd_line("--line-cmd engine shape -- true", D.STATION_ALIASES)
        self.assertEqual((probs, d["station_resolved"]), ([], "card" + C.UE_CARD))

    def test_one_job_per_idle_device_is_drawn(self):
        S = self.msched()
        p = self.lst(["--line-cmd blender shape --id f1 -- true", "--line-cmd blender shape --id f2 -- true",
                      "--line-cmd igpu small --id f3 -- true", "--line-cmd nowhere shape --id bad -- true"])
        with S.lock:
            S.draw_fill(self.t)
        ids = sorted(j.id for j in S.jobs)
        self.assertEqual(ids, ["f1", "f3"])                 # one for the card, one for the iGPU
        self.assertTrue(all(j.entry["fill"] for j in S.jobs))
        self.assertEqual(L.FillList(p).lines(), ["--line-cmd blender shape --id f2 -- true"])
        self.assertIn("not a station", Path(str(p) + ".bad").read_text())
        self.assertEqual(len(Path(str(p) + ".done").read_text().splitlines()), 2)
        self.t += C.FILL_DRAW_S + 1
        with S.lock:
            S.draw_fill(self.t)
        self.assertEqual(sorted(j.id for j in S.jobs), ["f1", "f3"])   # f1 still on the line: nothing more for the card
        self.step(S)
        self.end(S, "f1")
        self.t += C.FILL_DRAW_S + 1
        with S.lock:
            S.draw_fill(self.t)
        self.assertIn("f2", [j.id for j in S.jobs])

    def test_nothing_is_drawn_for_a_device_with_normal_work_or_during_a_window(self):
        S = self.msched()
        p = self.lst(["--line-cmd blender shape --id f1 -- true"])
        self.cmd(S, "draw0", "blender", "refs")
        with S.lock:
            S.draw_fill(self.t)
        self.assertEqual([j.id for j in S.jobs], ["draw0"])
        self.step(S)
        self.end(S, "draw0")
        self.cmd(S, "clips0", "video", "video")
        self.t += C.FILL_DRAW_S + 1
        with S.lock:
            S.draw_fill(self.t)
        self.assertEqual([j.id for j in S.jobs], ["clips0"])
        self.assertEqual(len(L.FillList(p).lines()), 1)

    def test_add_skips_lines_already_in_the_list(self):
        p = self.lst(["a"])
        self.assertEqual(L.FillList(p).add(["a", "b"]), 1)
        self.assertEqual(L.FillList(p).lines(), ["a", "b"])


class WindowTest(TimedBase):
    """A card the clips leave idle inside a video window (off unless ASSETS_WINDOW_FILL=1)."""

    def window(self, S, free_gb=20.0, busy=0.0, free_for=100.0):
        clips = self.cmd(S, "clips0", "video", "video")
        self.step(S)
        S.released["done_t"] = self.t
        self.step(S)
        self.assertEqual(self.running(S), ["clips0"])
        for c in S.cards.values():
            c.free_gb = lambda: free_gb
            S.lock_free_since[c.bus] = self.t - free_for
        S.load_between = lambda bus, t0, t1: {"busy_mean": busy}
        return clips

    def test_off_by_default(self):
        S = self.msched()
        self.assertFalse(C.WINDOW_FILL)
        self.window(S)
        self.cmd(S, "m0", "blender", "measure", vram_gb=2, est_s=10)
        self.step(S)
        self.assertEqual(self.running(S), ["clips0"])

    def test_a_short_command_that_fits_runs_and_the_release_stands(self):
        S = self.msched()
        C.WINDOW_FILL, C.WINDOW_IDLE_S, C.WINDOW_JOB_MAX_S, C.WINDOW_VRAM_KEEP_GB = True, 45, 120, 2
        self.window(S)
        ok = self.cmd(S, "m0", "blender", "measure", vram_gb=2, est_s=10)
        self.step(S)
        self.assertEqual(self.running(S), ["clips0", "m0"])
        self.assertTrue(ok.stages[0].info["in_window"])
        self.assertTrue(S.released)                          # the cards are not taken back for it
        self.assertIs(S.cards[C.BLENDER_CARD].cmd_busy, ok.stages[0])
        # the window ends while the command still runs: the cards come back, nothing waits on what is not counted
        self.end(S, "clips0")
        self.cmd(S, "draw0", "blender", "refs")
        S.t_probe = 0.0
        with S.lock:
            S.update_mode(self.t)
            self.t += C.RECLAIM_FREE_S + 1
            S.t_probe = 0.0
            S.update_mode(self.t)
        self.assertIsNone(S.released)
        self.step(S)
        self.assertEqual(self.running(S), ["m0"])            # the station is still busy with it
        self.end(S, "m0")
        S.cards[C.BLENDER_CARD].cmd_busy = None
        self.step(S)
        self.assertEqual(self.running(S), ["draw0"])

    def test_each_condition_is_needed(self):
        C.WINDOW_FILL, C.WINDOW_IDLE_S, C.WINDOW_JOB_MAX_S, C.WINDOW_VRAM_KEEP_GB = True, 45, 120, 2
        cases = {"long": dict(vram_gb=2, est_s=500), "no_time": dict(vram_gb=2), "no_vram": dict(est_s=10),
                 "vram": dict(vram_gb=19, est_s=10)}
        for name, kw in cases.items():
            S = self.msched()
            self.window(S)
            self.cmd(S, "m0", "blender", "measure", **kw)
            self.step(S)
            self.assertEqual(self.running(S), ["clips0"], name)
        for name, wkw in {"lock_busy": dict(free_for=5.0), "card_busy": dict(busy=60.0)}.items():
            S = self.msched()
            self.window(S, **wkw)
            self.cmd(S, "m0", "blender", "measure", vram_gb=2, est_s=10)
            self.step(S)
            self.assertEqual(self.running(S), ["clips0"], name)
        S = self.msched()                                    # an engine job or a bake never runs in a window
        self.window(S)
        self.ue_job(S, "set0")
        self.step(S)
        self.assertEqual(self.running(S), ["clips0"])

    def test_measured_time_and_kind_vram_are_used(self):
        S = self.msched()
        C.WINDOW_FILL, C.VRAM_KIND_GB = True, {"measure": 2.0}
        S.kind_cost["measure"] = [(9.0, None), (11.0, None), (10.0, None)]
        self.window(S)
        self.cmd(S, "m0", "blender", "measure")
        self.step(S)
        self.assertEqual(self.running(S), ["clips0", "m0"])

    def test_normal_goes_before_fill_and_fill_runs_when_normal_cannot(self):
        S = self.msched()
        C.WINDOW_FILL = True
        self.window(S)
        self.cmd(S, "fill0", "blender", "measure", vram_gb=2, est_s=10, fill=True)
        self.cmd(S, "long0", "blender", "refs", vram_gb=2, est_s=900)      # normal, but too long for a window
        self.step(S)
        self.assertEqual(self.running(S), ["clips0", "fill0"])
        S = self.msched()
        self.window(S)
        self.cmd(S, "fill0", "blender", "measure", vram_gb=2, est_s=10, fill=True)
        self.cmd(S, "m0", "blender", "measure", vram_gb=2, est_s=10)
        self.step(S)
        self.assertEqual(self.running(S), ["clips0", "m0"])


if __name__ == "__main__":
    unittest.main()
