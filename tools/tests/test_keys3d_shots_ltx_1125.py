"""11-25 user: the 3D flow and the shots flow are identical when it comes to running LTX; the only difference is how
the keyframes are made. The 3D flow's gated keys go to the shots flow's LTX stage as a plan + keys folder
(keys3d_plan), its own job builder is not reached, and its staged motion guides an action or quiet clip as the
shot's "control" inside that one stage. No model runs here: the runner, the judges and the cut are stand-ins; ffmpeg
cuts a 2 s test tone."""
import copy, json, os, subprocess, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
os.environ.pop("FLOW3D_OWN_LTX", None)
import shot_flow as S
import flow3d as F
import remake_chapter as R
import keys3d_plan as K
from PIL import Image

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

LOOKS = {"Orla": "a woman of about 30, wearing a grey wool dress", "Tam": "a man of about 40, wearing a brown wool coat"}
dO, dT = "the woman in the grey wool dress", "the man in the brown wool coat"
STEM = "01_x"

# --- stand-ins: no runner, no judge, no service, no cut ----------------------------------------------------------------
real_run, steps, own_path, cuts = subprocess.run, [], [], []
SKIP = set()                # clips the stand-in runner does not make


def fake_run(cmd, *a, **k):
    c = [str(x) for x in cmd]
    if c[0] == "systemctl":
        return SimpleNamespace(returncode=1, stdout="", stderr="")
    if c[0] == str(R.LTX_PY):               # (the runner's default negative prompt)
        return SimpleNamespace(returncode=0, stdout="NEG\n", stderr="")
    assert c[0] in ("ffmpeg", "ffprobe"), c[:3]
    return real_run(cmd, *a, **k)


def fake_step(cxl, name, args):             # the runner: every job of the file gets its clip and its record
    steps.append((name, [str(x) for x in args]))
    L = cxl.OUT / "ltx"
    if args[0] == "gen":
        for j in json.loads((L / args[1]).read_text()):
            if j["name"] not in SKIP:
                (L / "out" / f"{j['name']}.mp4").write_text("clip")
                (L / "out" / f"{j['name']}.json").write_text(json.dumps(j))
    return 0


subprocess.run = fake_run
R.ltx_step = fake_step
R.encode_all = lambda cxl, f, name, extra: steps.append((name, [str(x) for x in extra]))
for nm in ("make_jobs_files", "ltx", "qa", "assemble", "sheet", "talk_prompt", "i2v_prompt"):
    setattr(R, nm, lambda *a, _n=nm, **k: own_path.append(_n))      # the 3D flow's own LTX path: must stay unused
S.MF._dress = lambda *a, **k: {}
S._ctx = lambda b, st: (R, SimpleNamespace(book=Path(b), stem=st))
S._judge_as_they_land = S._sync_as_they_land = S._stitch_talk = S._sync_offsets = lambda *a, **k: None
S._take_faults = lambda *a, **k: []
S.assemble_stretch = lambda book, stem, plan, first, last, out, log=print: \
    cuts.append((json.loads(Path(plan).read_text()), first, last, Path(out))) or Path(out)
ltx_real = S.ltx


def make_book():
    b = Path(tempfile.mkdtemp(prefix="keys3d_"))
    vd = b / "work" / "video"
    (vd / STEM / "shots").mkdir(parents=True)
    (vd / "look.json").write_text(json.dumps({"characters": LOOKS, "places": {"the Hall": "a long stone hall"}}))
    (b / "book.cast.json").write_text(json.dumps({"characters": {k: {} for k in LOOKS}}))
    return b


def jobs(b, name):
    return json.loads((b / "work" / "video" / STEM / "shots" / "ltx" / f"{name}_jobs.json").read_text())


def put_key(b, sid):
    d = b / "work" / "video" / STEM / "shots" / sid.rsplit("_s", 1)[0] / sid
    d.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (S.W, S.H), (30, 40, 50)).save(d / "key.png")
    return d / "key.png"


# --- a shot without a motion guide: the job this flow built before, field for field -----------------------------------
b = make_book()
ACT = {"id": "action-000200_s0", "kind": "action", "who": ["Tam"], "dur": 4.0, "action": f"{dT[0].upper() + dT[1:]} walks to the door."}
QUIET = {"id": "action-000300_s0", "kind": "quiet", "who": ["Orla"], "dur": 4.0, "action": f"{dO[0].upper() + dO[1:]} looks down.",
         "pose_held": {"Orla": "sit"}}
ES = [{"id": "action-000200", "kind": "action", "t": 20.0, "dur": 4.0}, {"id": "action-000300", "kind": "action", "t": 30.0, "dur": 4.0}]
kA, kQ = put_key(b, ACT["id"]), put_key(b, QUIET["id"])
cxb = SimpleNamespace(book=b, stem=STEM)
S.ltx(b, STEM, ES, [dict(ACT), dict(QUIET)], log=lambda m: None)
act = ACT["action"]
hp = S._held(cxb, ACT)
want_two = {"name": "action-000200_s0", "emb": f"x_action-000200_s0_{S.MF._h(act + S.MF.FIGHT_NEG)}", "prompt": act,
            "width": 1152, "height": 640, "fps": 24, "seed": 200042, "frames": 97, "line": "action-000200_s0",
            "neg_extra": S.MF.FIGHT_NEG, "route": "ti2v", "steps": S.STEPS, "guider": S.GUIDER, "aguider": S.GUIDER,
            "images": [{"path": str(kA), "frame": 0, "strength": 1.0}],
            "held": {"emb": f"xh_action-000200_s0_{S.MF._h(hp + S.MF.FIGHT_NEG)}", "prompt": hp}, "people": 1}
check("no guide: the action job is the two-stage job as before, field for field", jobs(b, "two") == [want_two], jobs(b, "two"))
qp = QUIET["action"] + f" {dO[0].upper() + dO[1:]} stays sitting throughout." + \
    " Only the one person already in the picture is ever in it: nobody else comes into the picture."
hq = S._held(cxb, QUIET)
want_quiet = {"name": "action-000300_s0", "emb": f"qd_action-000300_s0_{S.MF._h(qp)}", "prompt": qp, "base": "dev",
              "width": 1152, "height": 640, "fps": 24, "seed": 300007, "frames": 97, "line": "action-000300_s0",
              "route": "i2v", "images": [{"path": str(kQ), "frame": 0, "strength": 1.0}],
              "held": {"emb": f"qh_action-000300_s0_{S.MF._h(hq)}", "prompt": hq}, "people": 1}
check("no guide: the quiet job is the dev-model job as before, field for field", jobs(b, "quiet") == [want_quiet], jobs(b, "quiet"))
check("no guide: no guided job, no IC-LoRA hold", jobs(b, "guided") == [] and not any("--ic-lora" in a for _, a in steps))

# --- the same shots with a guide: the IC-LoRA route the 3D flow guided with, this flow's words/seed/frames/people ----
b = make_book()
kA, kQ = put_key(b, ACT["id"]), put_key(b, QUIET["id"])
steps.clear()
CTL = {"video": "/x/depth.mp4", "attn": 0.35, "key_end": "/x/key_end.png", "key_end_strength": 0.6}
S.ltx(b, STEM, ES, [dict(ACT, control=CTL), dict(QUIET, control={"video": "/y/depth.mp4"})], log=lambda m: None)
g = {j["name"]: j for j in jobs(b, "guided")}
ga, gq = g.get("action-000200_s0") or {}, g.get("action-000300_s0") or {}
check("a guide: both shots are guided jobs, none two-stage or plain quiet",
      len(g) == 2 and jobs(b, "two") == [] and jobs(b, "quiet") == [], (list(g), jobs(b, "two")))
check("... on the 3D flow's route with its depth video as control",
      ga.get("route") == "stage" and ga.get("control") == {"path": "/x/depth.mp4", "attn": 0.35}, ga)
check("... the key at full strength and the move's end key at the last frame",
      ga.get("images") == [{"path": str(kA), "frame": 0, "strength": 1.0},
                           {"path": "/x/key_end.png", "frame": 96, "strength": 0.6}], ga.get("images"))
same = ("name", "prompt", "width", "height", "fps", "seed", "frames", "line", "people")
check("... words, seed, frames, size and head count are this flow's own",
      all(ga.get(k) == want_two[k] for k in same) and all(gq.get(k) == want_quiet[k] for k in same), (ga, gq))
check("... no two-stage fields, its words encoded for the IC route (no negative)",
      not ({"steps", "guider", "aguider", "neg_extra", "base"} & (set(ga) | set(gq))) and ga["emb"].startswith("gx_")
      and ga["held"]["prompt"] == hp and ga["held"]["emb"].startswith("gxh_") and gq["emb"].startswith("gq_"), ga)
check("... a guide without numbers takes the 3D flow's own (one definition, remake_chapter's)",
      gq.get("control") == {"path": "/y/depth.mp4", "attn": R.MODE2_ATTN} and len(gq["images"]) == 1, gq)
enc = json.loads((b / "work" / "video" / STEM / "shots" / "ltx" / "enc_dist.json").read_text())
check("... encoded without --dev", {x["name"] for x in enc} == {ga["emb"], ga["held"]["emb"], gq["emb"], gq["held"]["emb"]}
      and ("enc_dist", []) in steps, steps)
gen = [a for n, a in steps if n == "gen_guided"]
check("... rendered in their own hold with the union control LoRA",
      gen and gen[0][:2] == ["gen", "guided_jobs.json"] and gen[0][gen[0].index("--ic-lora") + 1] == str(R.IC_LORA)
      and "--ti2v" not in gen[0] and "--a2vid" not in gen[0], gen)

# --- the 3D flow's plan + stage folders (all made up) -----------------------------------------------------------------
b = make_book()
OUT = b / "work" / "video" / STEM / "flow3d"
wav = b / "tone.wav"
real_run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=180:duration=2", "-ar", "16000", "-ac", "1",
          str(wav)], check=True)
GUESTS = [{"who": "guest_1", "at": [3, 3], "pose": "standing"}, {"who": "guest_2", "at": [4, 3], "pose": "standing"}]
PATH = [{"frame": 0, "pos": [0, -4, 1.5], "look": [0, 0, 1.5], "lens": 50}, {"frame": 96, "pos": [0, -3, 1.5], "look": [0, 0, 1.5], "lens": 50}]


def staged_shot(mid, figures, camera, crowd=()):
    p = OUT / "stage" / "shots" / f"{mid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"figures": figures, "camera": camera, "crowd": list(crowd), "scene": {"id": "sc1"},
                             "set": {"where": "the Hall", "kind": "interior"}}))
    return str(p)


def stage_dir(mid, boxes=None, key=True, depth=False, key_end=False):
    d = OUT / "stage" / mid
    (d / "render").mkdir(parents=True)
    if key:
        Image.new("RGB", (S.W, S.H), (60, 50, 40)).save(d / "key.png")
    if boxes is not None:
        (d / "render" / "start_expect.json").write_text(json.dumps(
            {"size": [1152, 640], "figures": [{"who": w, "bbox": bx} for w, bx in boxes], "offscreen": []}))
    if depth:
        (d / "render" / "depth.mp4").write_text("depth")
    if key_end:
        Image.new("RGB", (S.W, S.H)).save(d / "key_end.png")


fig = lambda w, pose="standing": {"who": w, "kind": "human", "pose": pose, "at": [0, 0], "facing": 0, "height": 1.7}
talk = lambda mid, t, sh: {"id": mid, "kind": "talk", "t": t, "dur": 2.0, "route": "talk", "speaker": "Orla",
                           "line": "“We wait here,” ", "emotion": None, "creature": False, "speech_s": 2.0, "frames": 49,
                           "chunks": [[0, 49]], "wav": str(wav), "shot": sh, "unready": []}
act_ = lambda mid, t, sh, who: {"id": mid, "kind": "action", "t": t, "dur": 4.0, "route": "stage", "where": "the Hall",
                                "who": who, "frames": 97, "shot": sh, "unready": []}
P = {"entries": [
    talk("talk-000100", 10.0, staged_shot("talk-000100", [fig("Tam"), fig("Orla")], {"ots": ["Tam", "Orla"]}, GUESTS)),
    act_("action-000200", 20.0, staged_shot("action-000200", [fig("Tam", "walking")], {"framing": "wide", "path": PATH}), ["Tam"]),
    act_("action-000300", 30.0, staged_shot("action-000300", [fig("Orla", "sitting_chair")], {"framing": "medium close-up"}), ["Orla"]),
    act_("action-000400", 40.0, staged_shot("action-000400", [fig("Tam")], {}), ["Tam"]),
    talk("talk-000500", 50.0, staged_shot("talk-000500", [fig("Orla")], {})),
    talk("talk-000600", 60.0, staged_shot("talk-000600", [fig("Tam"), fig("Orla")], {}))]}
# the talk still: Tam's back on the left, Orla facing on the right, two guests behind (their boxes would not be there
# in a real render: here to show they are never taken for cast)
stage_dir("talk-000100", [("Orla", [640, 100, 1000, 640]), ("guest_1", [500, 200, 560, 400]), ("Tam", [0, 60, 380, 640]),
                          ("guest_2", [560, 200, 620, 400])])
stage_dir("action-000200", [("Tam", [300, 80, 600, 640])], depth=True, key_end=True)
stage_dir("action-000300", [("Orla", [400, 100, 800, 640])])
stage_dir("action-000400", [("Tam", [300, 80, 600, 640])], key=False)      # its key was never drawn
stage_dir("talk-000500", [("Orla", [400, 100, 800, 640])])                 # its key failed the gate (not kept)
stage_dir("talk-000600", [("Tam", [300, 80, 600, 640])])                   # the still shows only the listener
cx = SimpleNamespace(
    book=b, stem=STEM, OUT=OUT, W=b / "work", plan={"pictures": [], "look": {"places": {}}},
    sets={"the Hall": {"kind": "interior", "desc": "a long stone hall with a hearth at one end"}},
    frames={"talk-000100": {"crowd": "guests in travelling cloaks", "scene": "sc1", "where": "the Hall"},
            "action-000200": {"prompt": "Wide shot of Tam crossing the hall.", "where": "the Hall", "scene": "sc1",
                              "motion": "Tam walks to the door. The camera follows him."},
            "action-000300": {"prompt": "Orla by the hearth.", "motion": "Orla looks down.", "where": "the Hall"}})
KEPT = {"talk-000100", "action-000200", "action-000300", "action-000400", "talk-000600"}
logs = []
plan, keys, ids, lost = K.build(R, cx, P, KEPT, logs.append)
by = json.loads(plan.read_text())
sT, sA, sQ = by["talk-000100"][0], by["action-000200"][0], by["action-000300"][0]
check("one shot <id>_s0 for each moment with a gated key",
      ids == ["talk-000100_s0", "action-000200_s0", "action-000300_s0", "talk-000600_s0"], ids)
check("a moment whose key was never drawn or failed the gate is reported, and keeps its place in the cut",
      lost == ["action-000400", "talk-000500"] and [e["id"] for e in by["_es"]] == [e["id"] for e in P["entries"]]
      and by["action-000400"][0]["id"] == "action-000400_snone", (lost, by["action-000400"]))
shots = [by[i.rsplit("_s", 1)[0]][0] for i in ids]
bad, ppl, _ = S.check_supplied(b, shots, keys, None, (), True)
check("the plan and keys folder pass the supplied-keys checks", not bad and set(ppl) == set(ids), bad)
vis = json.loads((keys / "talk-000100_s0.shot_facts.json").read_text())["visible"]
check("the key's people are the cast its still shows, left to right, boxes as 0-1; guests never among them",
      vis == [{"name": "Tam", "box": [0.0, 0.0938, 0.3299, 1.0]}, {"name": "Orla", "box": [0.5556, 0.1562, 0.8681, 1.0]}], vis)
check("a still with guests: the shot carries its crowd words (no head is counted against the cast)",
      sT["crowd"] == "guests in travelling cloaks" and sA["crowd"] == "" and sQ["crowd"] == "", sT["crowd"])
check("the talk shot: who, listener and the over-the-shoulder person as the still has them",
      sT["kind"] == "talk" and sT["who"] == ["Tam", "Orla"] and sT["listener"] == "Tam" and sT["ots"] == "Tam"
      and "control" not in sT and sT["action"] == "", sT)
check("the place and its words as the 3D prompts get them",
      sT["place"] == "the Hall" and sT["room"] == "a long stone hall with a hearth at one end" and sT["scene"] == "sc1", sT)
eT = next(e for e in by["_es"] if e["id"] == "talk-000100")
check("the talk moment's own fields go through, its line's words without quotation marks",
      eT["speaker"] == "Orla" and eT["wav"] == str(wav) and eT["chunks"] == [[0, 49]] and eT["words"] == "We wait here,"
      and eT["emotion"] == "", eT)
check("movement the staging marks (a walking pose, an authored camera move) = an action shot, its words name-free",
      sA["kind"] == "action" and sA["framing"] == "wide" and
      sA["action"] == f"{dT[0].upper() + dT[1:]} walks to the door. The camera slowly pushes in, one smooth continuous "
                      f"move, no cuts.", sA)
check("... guided by the staged motion's depth video, the move's end key at the 3D flow's strength",
      sA.get("control") == {"video": str(OUT / "stage" / "action-000200" / "render" / "depth.mp4"), "attn": R.MODE2_ATTN,
                            "key_end": str(OUT / "stage" / "action-000200" / "key_end.png"),
                            "key_end_strength": R.END_GUIDE}, sA.get("control"))
check("a posture held = a quiet shot, the pose held, no guide without a depth video",
      sQ["kind"] == "quiet" and sQ["framing"] == "close" and sQ["pose_held"] == {"Orla": "sit"} and "control" not in sQ
      and sQ["action"] == f"{dO[0].upper() + dO[1:]} looks down.", sQ)

# --- the hand-over: the 3D flow's video step renders through shot_flow.ltx, nothing else -------------------------------
def spy_ltx(book, stem, es, shots_, log=print, **k):
    ltx_calls.append((copy.deepcopy(es), copy.deepcopy(shots_)))
    return ltx_real(book, stem, es, shots_, log, **k)
ltx_calls = []
S.ltx = spy_ltx
steps.clear(); logs.clear()
SKIP.add("action-000300_s0")                # one clip the runner fails to make
mp4, miss = F.shots_ltx(R, cx, P, KEPT, "", logs.append)
check("the 3D flow's own job builder, LTX loop, QA and assembly are not reached", own_path == [], own_path)
check("the clips come from shot_flow.ltx", len(ltx_calls) >= 1 and
      [s["id"] for s in ltx_calls[0][1]] == ["talk-000100_s0", "action-000200_s0", "action-000300_s0"], ltx_calls and ltx_calls[0][1])
tj = jobs(b, "talk")
es0, shots0 = ltx_calls[0]
sd = b / "work" / "video" / STEM / "shots"
cxb = SimpleNamespace(book=b, stem=STEM)
ltx_real(b, STEM, es0, copy.deepcopy(shots0), lambda m: None)       # the shots flow's builder, called directly
check("the talk job of the hand-over is the one shot_flow.ltx builds for that entry, shot and key", jobs(b, "talk") == tj
      and len(tj) == 1, tj)
t0 = tj[0] if tj else {}
sT0 = next(s for s in shots0 if s["id"] == "talk-000100_s0")
check("... the shots flow's talk recipe: its words, steps, guider, key strength, silent lead-in; no guide",
      t0.get("prompt") == S.with_facts(S._talk_prompt(R, cxb, eT, "the Hall", sT0), sT0) and
      "lips and jaw moving clearly" in t0["prompt"] and 'She says, "We wait here,"' in t0["prompt"] and
      "on the right" in t0["prompt"] and "listens" not in t0["prompt"] and
      t0["steps"] == S.TALK_STEPS and t0["guider"] == S.TALK_GUIDER and t0["pre"] == S.TALK_PRE and
      t0["images"] == [{"path": str(sd / "talk-000100" / "talk-000100_s0" / "key.png"), "frame": 0, "strength": S.TALK_KEY}]
      and t0["lock_end"] is None and "control" not in t0 and t0["audio_src"] == str(wav), t0)
check("... the lip-sync zone from the still's own box of the speaker, no head count before the guests",
      t0.get("speaker_box") == [round(0.5556 - S.KEY_BOX_PAD, 3), 0.0, round(0.8681 + S.KEY_BOX_PAD, 3), 1.0]
      and t0["people"] is None and t0["speaker"] == "Orla", t0)
g = jobs(b, "guided")
check("the staged motion guides the action clip inside the same stage",
      [j["name"] for j in g] == ["action-000200_s0"] and g[0]["control"]["path"].endswith("action-000200/render/depth.mp4")
      and g[0]["images"][1]["path"].endswith("action-000200/key_end.png") and g[0]["frames"] == 97 and
      g[0]["prompt"].startswith(f"{dT[0].upper() + dT[1:]} walks to the door."), g)
check("a talk shot whose still does not show the speaker is not rendered",
      "talk-000600_s0" in S.STOPPED and not any(j["line"] == "talk-000600" for j in tj), S.STOPPED)
check("every moment without a new clip is reported: no key, a failed gate, an unseen speaker, a clip not made",
      miss == ["action-000300", "action-000400", "talk-000500", "talk-000600"] and
      json.loads((OUT / "MISSING.json").read_text()) == miss and
      all(any(x.startswith(f"  [3d] !!! {m}:") for x in logs) for m in miss), (miss, logs))
check("the cut is the shots flow's, over the whole stretch, named as the 3D flow names it (INCOMPLETE)",
      len(cuts) == 1 and cuts[0][1:3] == ("talk-000100_s0", "talk-000600_s0") and mp4 == cuts[0][3] and
      mp4.parent == b / "out" and mp4.name.startswith(f"{b.name}_{STEM}_") and mp4.name.endswith("_3d_INCOMPLETE.mp4"), cuts and cuts[0][1:])
pc = cuts[0][0] if cuts else {}
check("... a moment without a new clip is black there (no shot id an older clip could answer to)",
      [pc[m][0]["id"] for m in miss] == [f"{m}_snone" for m in miss] and pc["talk-000100"][0]["id"] == "talk-000100_s0", pc)
src = (HERE.parent / "flow3d.py").read_text()
v = src[src.index("def video("):src.index("def _mix_with_sfx(")]
check("flow3d.video: its own LTX path only behind FLOW3D_OWN_LTX=1",
      'own = os.environ.get("FLOW3D_OWN_LTX") == "1"' in v and "if keys_only or not own:" in v and
      v.index("if keys_only or not own:") < v.index("R.make_jobs_files(") < v.index("R.qa(") and
      v.index("    if own:") < v.index("R.assemble(") < v.index("shots_ltx(R, cx, P, kept"), None)

print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
