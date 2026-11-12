"""11-09: a chapter's film ended 88 s before its narration. Two causes, both here:
1. A new recording (another narrator) of a book at 1.25x took the kept moment list through the pace maps as if the
   old track were unstretched: every moment 20% early, the list 88 s short. A new recording moves the moments line to
   line (bookcaster.apply_pace, fresh, with the recording it replaces), the added moments with them.
2. The moments the cover step then added for the uncovered end never reached the director: the moment list was read
   again without being made again (shot_flow.covered)."""
import json, subprocess, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
from tts_reader import bookcaster as BC

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


def record(b, starts, length):
    """A recording: a tone `length` s long, three lines at `starts`."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=300:d={length}", "-ac", "2",
                    "-b:a", "192k", str(b / "work" / "01_x.mp3")], check=True)
    (b / "work" / "01_x.timeline.json").write_text(json.dumps(
        [{"t": t, "kind": k, "speaker": w, "text": z} for t, (k, w, z) in
         zip(starts, (("narration", "narrator", "a"), ("quote", "Ada", "b"), ("narration", "narrator", "c")))]))


def book(pace):
    b = Path(tempfile.mkdtemp(prefix="rerec_"))
    (b / "work" / "video" / "01_x" / "anim").mkdir(parents=True)
    (b / "work" / "video" / "01_x" / "stage").mkdir()
    (b / "work" / "video" / "01_x" / "flow3d").mkdir()
    (b / "chapters").mkdir()
    (b / "chapters" / "01_x.txt").write_text("x")
    BC.set_pace(b, pace)
    return b, b / "chapters" / "01_x.txt", b / "work" / "video" / "01_x"


BC_video_work = BC._video_work
for pace in (1.25, 1.0):
    b, f, vw = book(pace)
    BC._video_work = lambda book_, f_, b=b: b / "work" / "video" / f_.stem
    # the first recording: 10 s raw, lines at 0, 4, 6 -> at the pace; then the plan made against it
    record(b, [0.0, 4.0, 6.0], 10)
    BC.apply_pace(b, f, lambda m: None, fresh=True)
    p = lambda x: round(x / pace, 3)
    man, xm, fp = vw / "anim" / "manifest.json", vw / "stage" / "extra_moments.json", vw / "flow3d" / "plan.json"
    man.parent.mkdir(exist_ok=True)
    man.write_text(json.dumps([{"id": "action-000000", "kind": "action", "t": 0.0, "dur": p(4.0)},
                               {"id": "talk-000040", "kind": "talk", "t": p(4.0), "dur": p(2.0)}]))
    xm.write_text(json.dumps([{"id": "action-000060", "kind": "action", "t": p(6.0), "dur": p(4.0), "extra": True}]))
    fp.write_text("{}")
    # another narrator: 15 s raw, the same lines at 0, 7, 9
    was = BC._line_map(b / "work", "01_x")
    check(f"{pace}: the recording before, line by line", [round(t, 2) for _, t in was[:3]] == [p(0), round(p(4), 2), round(p(6), 2)]
          and abs(was[3][1] - 10 / pace) < 0.1, was)
    record(b, [0.0, 7.0, 9.0], 15)
    said = []
    BC.apply_pace(b, f, said.append, fresh=True, was=was)
    m, x = json.loads(man.read_text()), json.loads(xm.read_text())
    check(f"{pace}: the spoken line's moment is on the line's new start", abs(m[1]["t"] - 7 / pace) < 0.02 and
          abs(m[1]["dur"] - 2 / pace) < 0.02, m)
    check(f"{pace}: the first moment runs to it", m[0]["t"] == 0 and abs(m[0]["dur"] - 7 / pace) < 0.02, m)
    check(f"{pace}: the added moment moved too, and the moments end where the narration does",
          abs(x[0]["t"] - 9 / pace) < 0.02 and abs(x[0]["t"] + x[0]["dur"] - 15 / pace) < 0.1, x)
    check(f"{pace}: the moment list is made again", not fp.exists())
    check(f"{pace}: said so", any("moved to the new recording's lines" in z for z in said), said)

# other lines (the text changed): end to end, never through the pace
b, f, vw = book(1.25)
BC._video_work = lambda book_, f_, b=b: b / "work" / "video" / f_.stem
record(b, [0.0, 4.0, 6.0], 10)
BC.apply_pace(b, f, lambda m: None, fresh=True)
(vw / "anim").mkdir(exist_ok=True)
(vw / "anim" / "manifest.json").write_text(json.dumps([{"id": "action-000040", "kind": "action", "t": 4.0, "dur": 4.0}]))
was = BC._line_map(b / "work", "01_x")
record(b, [0.0, 7.0, 9.0], 15)
tl = b / "work" / "01_x.timeline.json"
tl.write_text(json.dumps(json.loads(tl.read_text())[:2]))
BC.apply_pace(b, f, lambda m: None, fresh=True, was=was)
m = json.loads((vw / "anim" / "manifest.json").read_text())[0]
check("other lines: the moments stretched end to end (8 s -> 12 s)", abs(m["t"] - 6.0) < 0.1 and abs(m["t"] + m["dur"] - 12.0) < 0.15, m)
BC._video_work = BC_video_work

# --- the cover step's moments reach the director -----------------------------------------------------------------
import shot_flow as S               # noqa: E402
b = Path(tempfile.mkdtemp(prefix="rerec_"))
vd = b / "work" / "video" / "01_x"
(vd / "flow3d").mkdir(parents=True)
E = [{"id": "action-000000", "kind": "action", "t": 0.0, "dur": 5.0}]
(vd / "flow3d" / "plan.json").write_text(json.dumps({"entries": E}))
made = []


class R:                            # the planner: the moment list from its moments and the added ones
    @staticmethod
    def plan(cx):
        (vd / "flow3d" / "plan.json").write_text(json.dumps({"entries": E + cx}))


S.MF.cover = lambda book_, stem, t0, t1, log=print: [m["id"] for m in made]
S._CTX[(str(b), "01_x")] = (R, [])                 # (a context from before the cover step: no added moments in it)
import flow3d                       # noqa: E402
flow3d._ctx = lambda book_, stem: (R, list(made))
es = S.covered(b, "01_x", 0.0, 1e9, lambda m: None)
check("nothing added: the moments as planned", [e["id"] for e in es] == ["action-000000"], es)
made.append({"id": "action-000050", "kind": "action", "t": 5.0, "dur": 5.0, "extra": True})
es = S.covered(b, "01_x", 0.0, 1e9, lambda m: None)
sp = S._spans(es)
check("moments added after the last planned one are in the list the director gets",
      [e["id"] for e in es] == ["action-000000", "action-000050"], es)
check("...so its span runs to the end of the narration (10 s)",
      abs(float(es[-1]["t"]) + sp[es[-1]["id"]] - 10.0) < 0.01, sp)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
