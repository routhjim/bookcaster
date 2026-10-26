"""The 10-21 evening fixes to the 3D stage step (synthetic, CPU only, seconds).

    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_stage_1021.py

- a failed tool step is reported to the job log AT ONCE with its last lines, a long one says it is still running
  (a failed LoRA ran unseen for 80 min);
- a failure logged by the render driver reaches the job log (log_entry -> ALERT);
- needs_clip: only the moments whose motion is the point (contacts, fx, a recipe, a creature, a carried person) get
  their whole clip at staging, the rest stills;
- render_sig changes when a prop the shot uses gets a new model (the re-oriented bellows), not for other props.
Prints PASS/FAIL per check, exit 1 on any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flow3d as F  # noqa: E402
import remake_chapter as R  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def test_run_alerts(tmp):
    got = []
    F.JOB_LOG[0] = got.append
    lg = tmp / "step.log"
    rc = F.run(["sh", "-c", "echo working; echo 'Traceback: boom' >&2; exit 3", "x.py"], lg)
    check("run: failed step returns its rc", rc == 3, rc)
    check("run: failure reported at once with its last lines",
          any("FAILED" in g and "boom" in g for g in got), got)
    got.clear()
    F.HEARTBEAT_S, old = 1, F.HEARTBEAT_S
    try:
        F.run(["sh", "-c", "sleep 32"], lg)
    finally:
        F.HEARTBEAT_S = old
    check("run: a long step says it is still running", any("still running" in g for g in got), got)
    got.clear()
    F.run(["true"], lg)
    check("run: a passing step is silent", not got, got)
    F.JOB_LOG[0] = None


def test_log_entry_alert(tmp):
    got = []
    R.ALERT = got.append
    try:
        R.log_entry(tmp, {"phase": "stage-render", "id": "a1", "rc": 0})
        R.log_entry(tmp, {"phase": "stage", "id": "a2", "ok": False, "why": "DEFAULT STAGING for King"})
    finally:
        R.ALERT = None
    check("log_entry: a failure reaches the job log", len(got) == 1 and "a2" in got[0] and "King" in got[0], got)


def shot(tmp, name, **kw):
    p = tmp / f"{name}.json"
    p.write_text(json.dumps({"figures": kw.pop("figures", [{"who": "Edric", "kind": "human"}]),
                             "set": {"props": kw.pop("props", [])}, **kw}))
    return {"id": name, "shot": str(p), "route": "stage"}


def test_needs_clip(tmp):
    cases = {
        "talky": (shot(tmp, "talky"), False),
        "strike": (shot(tmp, "strike", contacts=[{"prop": "hammer"}], fx=[{"kind": "sparks"}]), True),
        "recipe": (shot(tmp, "recipe", recipe={"name": "strike_on_support"}), True),
        "dragon": (shot(tmp, "dragon", figures=[{"who": "Vorgrath", "kind": "creature"}]), True),
        "carried": (shot(tmp, "carried", figures=[{"who": "Maren", "attach": {"to": "Vorgrath"}}]), True),
    }
    for k, (e, want) in cases.items():
        check(f"needs_clip: {k} -> {want}", R.needs_clip(e) is want)


def test_render_sig(tmp):
    cx = SimpleNamespace(OUT=tmp / "flow3d", W=tmp / "work")
    pd = cx.W / "video" / "stage" / "props"
    for n in ("bellows", "anvil", "hammer"):
        (pd / n).mkdir(parents=True, exist_ok=True)
        (pd / n / "model.json").write_text(json.dumps({"extent_m": [1, 1, 1]}))
    e = shot(tmp, "forge", props=[{"prop": "bellows"}], figures=[{"who": "Edric", "holds": {"right": "hammer"}}])
    s0 = R.render_sig(cx, e)
    (pd / "anvil" / "model.json").write_text(json.dumps({"extent_m": [2, 1, 1]}))
    check("render_sig: a prop the shot does not use changes nothing", R.render_sig(cx, e) == s0)
    (pd / "bellows" / "model.json").write_text(json.dumps({"extent_m": [1.4, 0.6, 1.1]}))
    s1 = R.render_sig(cx, e)
    check("render_sig: a re-oriented set prop of the shot changes it", s1 != s0)
    (pd / "hammer" / "model.json").write_text(json.dumps({"extent_m": [0.3, 0.1, 0.1]}))
    s2 = R.render_sig(cx, e)
    check("render_sig: a re-made held prop changes it", s2 != s1)
    (pd / "hammer" / "model.json").write_text(json.dumps({"extent_m": [0.3, 0.1, 0.1], "rest_check": {"base_ratio": 0.4}}))
    check("render_sig: a QA-only field changes nothing", R.render_sig(cx, e) == s2)


def test_weak_key(tmp):
    d = tmp / "k"
    d.mkdir()
    def put(lk, hot_ok=True):
        (d / "key_draw.json").write_text(json.dumps({"chosen_seed": 7, "results": [
            {"seed": 7, "pass": True, "likeness": lk, "hot_ok": hot_ok, "score": 0.5}]}))
    put({"Edric": 0.62})
    check("weak_key: a good passing key is not redrawn", R.weak_key(d) is False)
    put({"Edric": 0.31})
    check("weak_key: a very poor face match gets one more seed", R.weak_key(d) is True)
    put({"Edric": 0.62}, hot_ok=False)
    check("weak_key: a faded glowing prop gets one more seed", R.weak_key(d) is True)
    check("key seeds default 2", R.KEY_SEEDS == 2, R.KEY_SEEDS)


def test_face_angle(tmp):
    import math
    lm = [[0, 0, 0.9]] * 68
    lm = [list(x) for x in lm]
    lm[0], lm[16] = [100, 200, 0.9], [200, 200, 0.9]
    lm[30] = [150, 210, 0.9]
    check("lm_yaw: nose centred between the jaw ends = frontal", abs(R.lm_yaw(lm)) < 1, R.lm_yaw(lm))
    lm[30] = [185, 210, 0.9]
    y = R.lm_yaw(lm)
    check("lm_yaw: nose towards the right jaw end = turned right", y is not None and 30 < y < 60, y)
    check("angle_words: small turn = facing the viewer", R.angle_words(5) == "facing the viewer")
    check("angle_words: 44 deg right", "45 degrees towards the right" in R.angle_words(44), R.angle_words(44))
    check("angle_words: profile", R.angle_words(-80).startswith("in profile"), R.angle_words(-80))
    rd = tmp / "r"
    rd.mkdir()
    # head at the origin, nose pointing +y, camera on +y (front on) / rotated 45 deg around
    for name, cam, want in (("front", [0, 5, 1.6], 0), ("q", [5 * math.sin(math.radians(45)), 5 * math.cos(math.radians(45)), 1.6], 45)):
        (rd / "stills.json").write_text(json.dumps({"frames": [{"name": "start", "camera": {"pos": cam}, "models": [
            {"who": "Edric", "targets": {"head": [0, 0, 1.6], "nose": [0, 0.1, 1.6]}}]}]}))
        got = R.head_yaw(rd, "Edric")
        check(f"head_yaw: {name} camera -> |yaw| {want}", got is not None and abs(abs(got) - want) < 1, got)
    check("talk singles: near front-on camera by default", R.TALK_FACE_ON <= 20, R.TALK_FACE_ON)


def test_sheet_cast():
    from tts_reader import video as V
    plan = {"look": {"characters": {"Edric": "a smith", "The King": "a king"}, "places": {}},
            "pictures": [{"id": "p1", "who": ["Edric"], "prompt": "Edric at the forge"}]}
    check("ref_prompts: only the still pictures' people without a sheet_cast", set(V.ref_prompts(plan, "ink")) == {"Edric"})
    plan["sheet_cast"] = ["Edric", "The King"]
    check("ref_prompts: sheet_cast adds a person seen only in moments", "The King" in V.ref_prompts(plan, "ink"))


def test_unfinished_lora(tmp):
    cx = SimpleNamespace(W=tmp / "w")
    d = cx.W / "video" / "charlora" / "hulda" / "lora"
    d.mkdir(parents=True)
    (d / "step_00100.safetensors").write_text("x")
    check("char_lora: a build stopped part-way is not used", R.char_lora(cx, "Hulda") is None)
    (d.parent / "build.json").write_text("{}")
    check("char_lora: a finished build is used", str(R.char_lora(cx, "Hulda")).endswith("step_00100.safetensors"))
    book = tmp / "book"
    (book / "work" / "video" / "charlora" / "hulda" / "lora").mkdir(parents=True)
    (book / "work" / "video" / "look.json").write_text(json.dumps({"characters": {"Hulda": "a smith"}}))
    (book / "work" / "video" / "charlora" / "hulda" / "lora" / "step_00100.safetensors").write_text("x")
    check("lora_todo: a build stopped part-way is retrained", "Hulda" in F.lora_todo(book), F.lora_todo(book))


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_run_alerts(tmp)
        test_log_entry_alert(tmp)
        test_needs_clip(tmp)
        test_render_sig(tmp)
        test_weak_key(tmp)
        test_face_angle(tmp)
        test_sheet_cast()
        test_unfinished_lora(tmp)
    print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
