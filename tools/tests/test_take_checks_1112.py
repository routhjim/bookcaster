"""11-12: the take checks the vision judge does not make: the camera or room drifting from the opening picture, a
jump between pieces, the people changing over the clip, the end state, the speaker turning away, a staged contact,
and the reviewer-against-judge count. Synthetic frames and head tracks only."""
import json, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import take_checks as T

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

rng = np.random.default_rng(7)
H, W = 108, 192


def room(h=H * 2, w=W * 2, block=None):
    """A textured still picture (smoothed noise), larger than a frame so it can be slid and scaled."""
    a = rng.random((h, w)).astype(np.float32)
    a = T._blur(T._blur(a, 3), 3)
    return (a - a.min()) / (a.max() - a.min())


def view(big, dx=0.0, dy=0.0, s=1.0):
    """The frame the camera sees: the big picture's middle, slid by (dx, dy) pixels and scaled by s."""
    ys = (np.arange(H) - H / 2) / s + big.shape[0] / 2 + dy
    xs = (np.arange(W) - W / 2) / s + big.shape[1] / 2 + dx
    return big[np.clip(ys.round().astype(int), 0, big.shape[0] - 1)][:, np.clip(xs.round().astype(int), 0, big.shape[1] - 1)]


BIG = room()
still = np.stack([view(BIG) for _ in range(12)])

# ---- the camera's move ----
m = T.camera_move(view(BIG), view(BIG, dx=10))
check("a slide of 10 px is read as a pan of about 5% of the width", abs(m["pan"] - 10 / W) < 0.012, m)
m = T.camera_move(view(BIG), view(BIG, s=1.1))
check("a scale of 1.10 is read as a zoom of about 10%", 0.06 < m["zoom"] < 0.14, m)
m = T.camera_move(view(BIG), view(BIG))
check("the same frame: no pan, no zoom", m["pan"] < 0.004 and abs(m["zoom"]) < 0.01, m)

m = T.camera_move(view(BIG), view(BIG, s=1.3))
check("a reframe of 1.30, too large for tiles alone, is still read as a zoom", 0.2 < m["zoom"] < 0.4, m)
m = T.camera_move(view(BIG), view(room(block=11)))
check("another picture altogether lines up with almost nothing", m["match"] < 0.3, m)
lost = np.stack([view(BIG)] * 4 + [view(room(block=11))] * 8)
d = T.drift(lost, still=True)
check("a still shot whose picture stops matching its opening is a fault", d["faults"] != [], d["faults"])

# ---- drift over a clip ----
d = T.drift(still, still=True)
check("a held camera on a still room: no fault", d["faults"] == [] and d["pan"] < 0.01, d["faults"])
pan = np.stack([view(BIG, dx=1.5 * i) for i in range(12)])
d = T.drift(pan, still=True)
check("a slow pan that holds is a fault", any("camera moves" in f for f in d["faults"]), d["faults"])
push = np.stack([view(BIG, s=1 + 0.012 * i) for i in range(12)])
d = T.drift(push, still=True)
check("a push-in is a fault", any("reframes" in f for f in d["faults"]), d["faults"])
blip = still.copy(); blip[5] = view(BIG, dx=12)
d = T.drift(blip, still=True)
check("one odd frame is not a pan", not any("camera moves" in f for f in d["faults"]), d["faults"])

# a person moving in front of a held camera: tiles covered by them are outvoted
walk = still.copy()
for i in range(12):
    walk[i, 30:90, 20 + 8 * i:20 + 8 * i + 24] = 0.05
d = T.drift(walk, still=False)
check("a person walking across a held camera is not a camera move", d["faults"] == [], (d["faults"], d["pan"], d["zoom"]))

# a new thing that appears and stays, in a still shot
new = still.copy(); new[6:, 10:50, 120:170] = 0.95
d = T.drift(new, still=True)
check("a large new thing that stays is a fault in a still shot", any("room changes" in f for f in d["faults"]), (d["faults"], d["appeared"]))
d = T.drift(new, still=False)
check("... and only reported in an action shot", d["faults"] == [] and d["appeared"] > 0.04, (d["faults"], d["appeared"]))
boxes = [[(120 / W, 10 / H, 50 / W, 40 / H)] for _ in range(12)]
d = T.drift(new, still=True, people=boxes)
check("a change inside a person's box is not the room changing", not any("room changes" in f for f in d["faults"]), d["faults"])

# ---- a seam between pieces ----
s = T.seam_jump(view(BIG), view(BIG, dx=0.4), usual=0.002)
check("neighbouring frames across a clean seam: no jump", s["fault"] is None, s)
s = T.seam_jump(view(BIG), view(BIG, s=1.12), usual=0.002)
check("a closer framing after the seam is a jump", s["fault"] is not None, s)
s = T.seam_jump(view(BIG), view(BIG, dx=9), usual=0.002)
check("a slid framing after the seam is a jump", s["fault"] is not None, s)
s = T.seam_jump(view(BIG), view(BIG, dx=5), usual=0.02)
check("a slide no larger than the clip's own motion is not a jump", s["fault"] is None, s)

# the picture as a whole, the lit source, the people across a seam
other = view(room(block=11))
f = T.seam_people(view(BIG), other)
check("another picture after the seam is said", any("picture changes" in x for x in f), f)
check("the same picture after the seam: nothing", T.seam_people(view(BIG), view(BIG, dx=0.3)) == [], None)
rgb = np.repeat(view(BIG)[..., None], 3, -1) * 0.5
fire = rgb.copy(); fire[60:80, 20:50] = (1.0, 0.6, 0.2)
f = T.seam_people(view(BIG), view(BIG), rgb_b=fire, rgb_a=rgb)
check("a fire lit before the seam and gone after is said", any("goes out" in x for x in f), f)
f = T.seam_people(view(BIG), view(BIG), rgb_b=fire, rgb_a=fire)
check("a fire on both sides: nothing", f == [], f)
hb = [(0.2, 0.2, 0.08, 0.14), (0.6, 0.25, 0.08, 0.14)]
f = T.seam_people(view(BIG), view(BIG), heads_b=hb, heads_a=hb[:1])
check("a person fewer after the seam is said", any("1 people after the seam, 2 before" in x for x in f), f)
f = T.seam_people(view(BIG), view(BIG), heads_b=hb, heads_a=[hb[0], (0.85, 0.5, 0.08, 0.14)])
check("a person somewhere else after the seam is said", any("somewhere else" in x for x in f), f)
dark, light = rgb.copy(), rgb.copy()
y0, y1, x0, x1 = int((0.2 + 1.3 * 0.14) * H), int((0.2 + 3.3 * 0.14) * H), int(0.2 * W), int(0.28 * W)
dark[y0:y1, x0:x1], light[y0:y1, x0:x1] = 0.1, 0.7
f = T.seam_people(view(BIG), view(BIG), heads_b=hb[:1], heads_a=hb[:1], rgb_b=dark, rgb_a=light)
check("a dark coat become a light one across the seam is said", any("clothes change" in x for x in f), f)
h1, h2 = rgb.copy(), rgb.copy()
hy0, hy1 = int(0.2 * H), int(0.34 * H)
h1[hy0:hy1, x0:x1], h2[hy0:hy1, x0:x1] = (0.15, 0.1, 0.08), (0.85, 0.75, 0.5)
f = T.seam_people(view(BIG), view(BIG), heads_b=hb[:1], heads_a=hb[:1], rgb_b=h1, rgb_a=h2)
check("a dark head become a fair one across the seam is said", any("someone else" in x for x in f), f)

# a seam inside a clip: the two frames at the seam agree (the later piece starts on the frame before it); the
# picture is compared a second later, against how much it moved in the second before
hold = [view(BIG) for _ in range(60)]
w_ = T.seam_window(np.stack(hold), 30)
check("a held picture through a seam: no fault", w_["fault"] is None and w_["more"] == [], w_)
jump = [view(BIG) for _ in range(30)] + [view(BIG, s=1 + 0.006 * i) for i in range(30)]
w_ = T.seam_window(np.stack(jump), 30)
check("the framing tightening in the second after a seam is a fault", w_["fault"] is not None, w_)
glide = [view(BIG, dx=0.5 * i) for i in range(60)]
w_ = T.seam_window(np.stack(glide), 30)
check("a move that was already going on before the seam is not the seam's", w_["fault"] is None, w_)
swap = [view(BIG) for _ in range(30)] + [other for _ in range(30)]
w_ = T.seam_window(np.stack(swap), 30)
check("another picture a second after the seam is said, with the seam's time", any("picture changes" in x and "1.2 s" in x for x in w_["more"]), w_)

# ---- the people over the clip ----
FPS = 4.0
A, B = (0.2, 0.3, 0.08, 0.14), (0.6, 0.3, 0.08, 0.14)
key = [{"box": A, "seated": False, "stays": True}, {"box": B, "seated": True, "stays": True}]
same = [[A, B] for _ in range(16)]
check("the same two heads throughout: no fault", T.people_over_time(same, FPS, key) == [], T.people_over_time(same, FPS, key))
ext = [list(x) for x in same]
for i in range(6, 12):
    ext[i].append((0.42, 0.28, 0.05, 0.09))
f = T.people_over_time(ext, FPS, key)
check("a third head for a second and a half is an extra person", any("3 people, not 2" in x for x in f), f)
one = [list(x) for x in same]; one[7].append((0.42, 0.28, 0.05, 0.09))
check("a third head in one frame is not", T.people_over_time(one, FPS, key) == [], T.people_over_time(one, FPS, key))
tiny = [x + [(0.42, 0.28, 0.02, 0.03)] for x in same]
check("a head far smaller than the key's smallest is not a person", T.people_over_time(tiny, FPS, key) == [], T.people_over_time(tiny, FPS, key))
small = [x + [(0.42, 0.28, 0.045, 0.075)] for x in same]
f = T.people_over_time(small, FPS, key)
check("a small head (over half the smallest key head) that lasts is an extra person", any("3 people" in x for x in f), f)
gone = [[A, B] if i < 8 else [A] for i in range(16)]
f = T.people_over_time(gone, FPS, key)
check("a head missing for the second half is a person gone", any("person 2 of 2 is gone from 2.0 s" in x for x in f), f)
flick = [[A, B] if i != 8 else [A] for i in range(16)]
check("a head missed in one frame is not gone", T.people_over_time(flick, FPS, key) == [], T.people_over_time(flick, FPS, key))
rise = [[A, (0.6, 0.3 - min(0.2, 0.04 * max(0, i - 4)), 0.08, 0.14)] for i in range(16)]
f = T.people_over_time(rise, FPS, key)
check("a seated person's head rising is them standing up", any("person 2 of 2 stands up" in x for x in f), f)
key_free = [dict(key[0]), dict(key[1], stays=False)]
check("... not a fault when the shot has them stand", not any("stands up" in x for x in T.people_over_time(rise, FPS, key_free)), None)
walkh = [[(0.2 + 0.02 * i, 0.3, 0.08, 0.14), B] for i in range(16)]
check("a standing person walking is followed, not lost", T.people_over_time(walkh, FPS, key) == [], T.people_over_time(walkh, FPS, key))
la, lb_ = np.zeros(216, np.float32), np.zeros(216, np.float32); la[3], lb_[200] = 1, 1
looks = [[la, la] if i < 8 else [la, lb_] for i in range(16)]
f = T.people_over_time(same, FPS, key, looks=looks)
check("a head whose colours change is someone else", any("person 2 of 2 looks like someone else from 2.0 s" in x for x in f), f)
check("head_look is a histogram that sums to one", abs(T.head_look(rgb, A).sum() - 1) < 1e-5, None)

# counts alone (the vision judge's numbers)
check("two counts over in a row: an extra person", T.count_faults({0.5: 2, 1.0: 3, 1.5: 3, 2.0: 2}, 2) != [], None)
check("one count over: not", T.count_faults({0.5: 2, 1.0: 3, 1.5: 2, 2.0: 2}, 2) == [], None)
check("fewer at the end: someone gone", any("gone" in x for x in T.count_faults({0.5: 2, 1.0: 2, 1.5: 1, 2.0: 1}, 2)), None)
ts = T.count_times(8.0)
check("counts are asked across the whole clip, the last near its end", len(ts) == 12 and ts[0] < 1.0 and ts[-1] > 7.4, ts)
check("a short clip still gets four asks", len(T.count_times(1.5)) == 4, T.count_times(1.5))

# ---- the end state ----
check("seated throughout: ends sitting", T.posture_at_end(same, FPS, key) == ["standing", "sitting"], T.posture_at_end(same, FPS, key))
check("risen: ends standing", T.posture_at_end(rise, FPS, key)[1] == "standing", T.posture_at_end(rise, FPS, key))
f = T.end_state_faults(rise, FPS, key, plan_end=[None, "sitting"])
check("ends standing where the shot ends seated: a fault", any("ends standing, the shot ends with them sitting" in x for x in f), f)
f = T.end_state_faults(rise, FPS, key, plan_end=[None, None], next_start=["standing", "sitting"])
check("ends standing where the next shot opens seated: a fault", any("the next shot opens with them sitting" in x for x in f), f)
check("ends as the shot and the next need: no fault", T.end_state_faults(same, FPS, key, ["standing", "sitting"], ["standing", "sitting"]) == [], None)
sit = [[(0.2, 0.3 + min(0.2, 0.04 * max(0, i - 4)), 0.08, 0.14), B] for i in range(16)]
check("a standing person whose head drops has sat", T.posture_at_end(sit, FPS, key)[0] == "sitting", T.posture_at_end(sit, FPS, key))

# ---- the speaker's turn ----
check("nose midway between the ears: facing", abs(T.yaw_from_points((50, 40), (40, 35), (60, 35), (30, 40), (70, 40))) < 5, None)
check("nose at an ear: profile", abs(T.yaw_from_points((69, 40), (60, 35), (68, 35), (30, 40), (70, 40))) > 70, None)
voiced = [True] * 12 + [False] * 4
check("facing while speaking: no fault", T.speaker_turn_fault([10.0] * 16, voiced) is None, None)
check("in profile for half the spoken frames: a fault", T.speaker_turn_fault([10.0] * 6 + [80.0] * 10, voiced) is not None, None)
check("in profile only when silent: no fault", T.speaker_turn_fault([10.0] * 12 + [80.0] * 4, voiced) is None, None)
check("no face found while speaking counts as turned away", T.speaker_turn_fault([None] * 8 + [5.0] * 8, voiced) is not None, None)

# ---- a staged contact ----
tgt = (0.5, 0.6, 0.1, 0.08)
check("the hand ends in the staged region: no fault", T.contact_fault([(0.2, 0.2)] * 8 + [(0.55, 0.63)] * 6, tgt, FPS) is None, None)
check("the hand ends elsewhere: a fault", T.contact_fault([(0.2, 0.2)] * 14, tgt, FPS) is not None, None)
check("the hand not found at the end: a fault", T.contact_fault([(0.55, 0.63)] * 8 + [None] * 6, tgt, FPS) is not None, None)

# ---- the reviewer against the judge ----
d_ = Path(tempfile.mkdtemp(prefix="rev_"))
rows = [{"chain": "a", "line_no": 1, "judge_pick": "B", "choice": {"pick": "C", "why": "B sits in the wrong chair"}},
        {"chain": "a", "line_no": 2, "judge_pick": "A", "choice": {"pick": "A"}},
        {"chain": "b", "line_no": 1, "judge_pick": "A", "choice": {"accept": True}},
        {"chain": "b", "line_no": 2, "judge_pick": "C", "choice": {"timeout": True}},
        {"chain": "c", "line_no": 1, "judge_pick": "A", "choice": {"override": "she sits down"}}]
(d_ / "log.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
ra = T.review_agreement([d_ / "log.jsonl"])
check("three cards picked, two with the judge", (ra["picked"], ra["agree"], ra["timeouts"], ra["overrides"]) == (3, 2, 1, 1), ra)
check("the disagreement keeps the reviewer's reason", ra["disagreements"][0]["why"].startswith("B sits"), ra["disagreements"])

# ---- names ----
check("a clip's kind comes from its name", [T.kind_of(n) for n in ("talk-1_s0.mp4", "quiet-2.mp4", "action-3_t2.mp4")] == ["talk", "quiet", "action"], None)

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
