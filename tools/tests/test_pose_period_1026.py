"""10-26 Acts 3 review (user): "the position of the lame man needs to be more explicit clip by clip, he's up then down
then up all before being healed, also there are outfit changes between clips". Found: the wardrobe made his healing an
outfit ("standing upright with legs bearing weight" from the begging paragraph), the drawing prompts said "early
medieval Northern Europe, around 1000 AD" for Roman Judea, and nothing carried a pose from shot to shot. CPU only.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_pose_period_1026.py"""
import json, re, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import mannequin_flow as MF
import shot_flow as SF
from tts_reader import video
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# --- poses carried: the director's up/down/up is overruled until the healing shot changes it
def sh(i, p0, p1, who=("Peter", "Lame Man"), kind="quiet"):
    return {"id": f"m{i}_s0", "who": list(who), "kind": kind, "pose": {"Peter": "standing", "Lame Man": p0},
            "pose_end": {"Peter": "standing", "Lame Man": p1}}
es = [{"id": f"m{i}"} for i in range(7)]
out = {"m0": [sh(0, "sitting", "sitting")], "m1": [sh(1, "standing", "standing")],      # director slip: up
       "m2": [sh(2, "lying on the steps", "lying on the steps")],                          # slip: down
       "m3": [{"id": "m3_s0", "who": ["Lame Man"], "kind": "talk"}],
       "m4": [sh(4, "sitting", "standing", kind="action")],                              # Peter pulls him up: healed
       "m5": [sh(5, "sitting", "sitting")],                                              # slip after the healing
       "m6": [sh(6, "standing", "standing")]}
SF.carry_poses(es, out)
held = [out[f"m{i}"][0].get("pose_held", {}).get("Lame Man") for i in (0, 1, 2, 4, 5, 6)]
check("lame man held: sit, sit, sit, sit (healed in that shot), then stand, stand", held == ["sit", "sit", "sit", "sit", "stand", "stand"], held)
check("the healing shot's pose_after is standing", out["m4"][0]["pose_after"]["Lame Man"] == "stand")
check("talk shots are left alone", "pose_held" not in out["m3"][0])
check("pose words: 'seated on a mat' = sit, 'lying' = lie, 'on his feet' = stand",
      (SF.pose_cat("seated on a mat"), SF.pose_cat("lying on the steps"), SF.pose_cat("back on his feet")) == ("sit", "lie", "stand"))
fa = SF._faults({"shots": [{"dur": 3, "who": ["Peter"], "picture": "p", "action": "a man looks"}]}, ["Peter"], 3)
check("a shot with no pose is sent back to the director", any('"pose"' in x for x in fa), fa)
check("the director is told about poses", '"pose_end"' in SF.SHOT_RULES and "cannot stand or walk" in SF.SHOT_RULES)

# --- the key: each person's held pose in the words, the not-standing one checked
cx = SimpleNamespace(look=lambda w, t: {"Lame Man": "a man of about 40, thin, wearing a ragged patched brown tunic",
                                        "Peter": "a man of about 40, grey-brown beard"}[w])
s5 = dict(out["m2"][0], picture="Peter stands; the lame man sits.")
q, _ = SF._key_questions(cx, s5)
check("key judge asks the lame man's pose", any(k.startswith("pose") and "is a man sitting (not standing)" in v for k, v in q.items()), q)
check("key judge does not ask a standing pose", sum(k.startswith("pose") for k in q) == 1, q)
with tempfile.TemporaryDirectory() as d:
    k = Path(d) / "k.png"
    from PIL import Image; Image.new("RGB", (64, 36), (120, 100, 80)).save(k)
    good = {"text": "no", "who0": "yes", "who1": "yes", "pose1": "yes"}
    check("a seed with the lame man sitting passes", SF._clean(good, k, 0))
    check("a seed with him standing fails (redrawn)", not SF._clean(dict(good, pose1="no"), k, 0))
hold = SF._pose_hold(cx, out["m2"][0])
check("clip prompt: 'A man stays sitting throughout.'", hold == " A man stays sitting throughout.", hold)
check("clip prompt: nothing held in the shot that heals him", SF._pose_hold(cx, out["m4"][0]) == "")

# --- posture out of the looks; the outfit checked in the key
lk = "a man of about 35, gaunt build, wearing a ragged patched brown tunic, sitting on the ground with legs extended"
check("look without its posture", SF.no_pose(lk) == "a man of about 35, gaunt build, wearing a ragged patched brown tunic", SF.no_pose(lk))
check("clothes of a look", SF.clothes_of(lk) == "a ragged patched brown tunic", SF.clothes_of(lk))
cx2 = SimpleNamespace(look=lambda w, t: "a man of about 33, short beard, wearing a simple beige linen tunic and a "
                      "cream-colored himation draped over one shoulder", dress=lambda w, t: "")
q2, _ = SF._key_questions(cx2, {"id": "action-000372_s0", "who": ["John"], "picture": "", "framing": "wide"})
check("key judge asks each person's clothes (wide/medium)", "who0w" in q2 and "himation" in q2["who0w"], q2)
q3, _ = SF._key_questions(cx2, {"id": "action-000372_s0", "who": ["John"], "picture": "", "framing": "close"})
check("not in a close shot (clothes mostly out of the picture)", "who0w" not in q3, q3)
cx3 = SimpleNamespace(look=cx2.look, dress=lambda w, t: "a white festival robe" if t >= 30 else "")
q4, _ = SF._key_questions(cx3, {"id": "action-000372_s0", "who": ["John"], "picture": "", "framing": "wide"})
check("outfit question follows the wardrobe in force at the shot's time", "white festival robe" in q4.get("who0w", ""), q4)

# --- wardrobe: clothes only
check("wardrobe: posture stripped from a dress", video.worn_only("ragged patched brown tunic, standing upright with legs "
      "bearing weight") == "ragged patched brown tunic", video.worn_only("ragged patched brown tunic, standing upright with legs bearing weight"))
check("wardrobe prompt: no 'lame man healed' example, posture ruled out", "lame man healed" not in video.WARDROBE_PROMPT
      and "POSTURE AND MOVEMENT ARE NOT CLOTHES" in video.WARDROBE_PROMPT)

# --- the period comes from the book
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "period.txt").write_text("Roman Judea, around 30 AD\n")
    pl = MF.period_line(Path(d))
    check("period line from period.txt", pl.startswith("Set in Roman Judea, around 30 AD:"), pl)
    check("no period.txt -> no period line", MF.period_line(Path(d) / "none") == "")
for f in ("shot_flow.py", "mannequin_flow.py"):
    src = (HERE / f).read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#") and '"""' not in l and "1000 AD\" was" not in l)
    check(f"{f}: no hardcoded 1000 AD period in a prompt", "around 1000 AD" not in code,
          [l for l in code.splitlines() if "1000 AD" in l][:2])
print(json.dumps({"fails": fails}))
