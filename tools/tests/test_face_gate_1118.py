"""The face gate (tools/assets/redraw/facegate.py, keycheck.py, accept.py, story_keys.py): a person whose face shows
toward the camera, large enough, must be drawn with a human face; a try failing it is never kept over one passing
it, and a shot whose every try fails it has no key (story_keys counts it keyless and restages it)."""
import json
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "assets"))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "redraw"))
import facegate as FG  # noqa: E402
import accept as AC  # noqa: E402
import story_keys as SK  # noqa: E402


def person(**kw):
    """A close shot's person, as accept.cmd_checks writes it (the numbers of the 11-18 stone-head close shot)."""
    p = {"key": "a", "head_to_camera_deg": 0.9, "creature": False, "face_covered": False,
         "joints2d": {"FACIAL_L_Eye": [0.5378, 0.3347, True], "FACIAL_R_Eye": [0.462, 0.3324, True],
                      "chin": [0.5, 0.5957, True], "head_top": [0.5, 0.1195, False]}}
    p.update(kw)
    return p


class FakeJudge:
    def __init__(self, answer):
        self.answer, self.asked = answer, []

    def ask(self, im, q, n, side):
        self.asked.append((im.size, q))
        return self.answer


# ------------------------------------------------------------------ when the gate applies
def test_the_gate_is_due_for_a_large_face_toward_the_camera():
    assert FG.due(person())


def test_the_gate_is_not_due_for_creatures_covered_faces_turned_or_small_heads_or_eyes_out_of_frame():
    assert not FG.due(person(creature=True))
    assert not FG.due(person(face_covered=True))
    assert not FG.due(person(head_to_camera_deg=75.0))
    assert not FG.due(person(head_to_camera_deg=None))
    small = person()
    small["joints2d"] = dict(small["joints2d"], head_top=[0.5, 0.52, True], chin=[0.5, 0.6, True])   # 0.08
    assert not FG.due(small)
    out = person()
    out["joints2d"] = dict(out["joints2d"], FACIAL_L_Eye=[1.02, 0.33, False])
    assert not FG.due(out)
    hidden = person()
    hidden["joints2d"] = dict(hidden["joints2d"], FACIAL_L_Eye=[0.53, 0.33, False], FACIAL_R_Eye=[0.46, 0.33, False])
    assert not FG.due(hidden)


# ------------------------------------------------------------------ the judge's answer
def test_the_verdict_reads_the_forced_choice():
    assert FG.verdict("human") is True
    assert FG.verdict("Human.") is True
    for a in ("creature", "Statue", "mask", "none", "**creature**"):
        assert FG.verdict(a) is False, a
    for a in ("", None, "cannot tell", "yes"):
        assert FG.verdict(a) is None, a


def test_ask_crops_the_head_and_records_the_answer():
    frame = Image.new("RGB", (1152, 640), (90, 90, 90))
    j = FakeJudge("creature")
    got = FG.ask(j, frame, person())
    assert got == {"human_face": "creature", "human_face_ok": False}
    (w, h), q = j.asked[0]
    assert q == FG.HUMAN_Q and max(w, h) >= FG.CROP_MIN and abs(w - h) <= 0.1 * w   # the head's square (top clipped)
    j2 = FakeJudge("human")
    assert FG.ask(j2, frame, person())["human_face_ok"] is True
    j3 = FakeJudge("human")
    assert FG.ask(j3, frame, person(creature=True)) == {} and not j3.asked      # not due: not asked


def test_a_failed_gate_is_a_named_failure_and_none_or_true_is_not():
    assert "human face" in AC.failures({"human_face_ok": False})
    assert "human face" not in AC.failures({"human_face_ok": True})
    assert "human face" not in AC.failures({"human_face_ok": None})
    assert FG.blocked({"a": ["outfit", "human face"], "b": ["hair"]}) == ["a"]
    assert FG.blocked({"a": ["outfit", "hair", "face"]}) == []


# ------------------------------------------------------------------ the pick
def test_a_try_failing_the_gate_never_wins_over_one_passing_it(tmp_path):
    # the first draw: one failure, the gate; the re-draw: three others: the re-draw is kept
    r = AC.pick_one("k1", {"a": ["human face"]}, {"a": ["outfit", "hair", "neck"]}, tmp_path, {})
    assert r["id"] == "k1__x" and not r.get("no_key") and r["path"].endswith("k1__x.png")
    # and the other way round: the first draw is kept over a re-draw with fewer, but blocking, failures
    r = AC.pick_one("k2", {"a": ["outfit", "hair"]}, {"a": ["human face"]}, tmp_path, {})
    assert r["id"] == "k2" and not r.get("no_key")
    # without the gate the old rule holds: fewer failures win
    r = AC.pick_one("k3", {"a": ["outfit", "hair"]}, {"a": ["outfit"]}, tmp_path, {})
    assert r["id"] == "k3__x"


def test_every_try_failing_the_gate_leaves_the_shot_with_no_key(tmp_path):
    cast = {"a": {"noun": "man"}}
    r = AC.pick_one("k", {"a": ["outfit", "hair", "human face"]}, {"a": ["human face"]}, tmp_path, cast)
    assert r["path"] is None and r["id"] is None
    assert r["no_key"] == "no try drew the man with a human face in 2 tries"
    assert r["rejected"] == ["k", "k__x"]
    r = AC.pick_one("k", {"a": ["human face"]}, None, tmp_path, cast)        # no re-draw made
    assert r["path"] is None and r["rejected"] == ["k"] and "in 1 tries" in r["no_key"]


def test_cmd_pick_writes_the_no_key_row(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    rec = lambda ok: {"people": {"a": {"redraw": {}, "human_face_ok": ok}}}     # noqa: E731
    checked = {"k": rec(False), "m": rec(False)}
    retry = {"k__x": rec(False), "m__x": rec(True)}
    AC.cmd_pick({"out": str(out), "cast": {"a": {"noun": "woman"}}}, checked, retry, out / "accepted.json")
    acc = json.loads((out / "accepted.json").read_text())
    assert acc["k"]["no_key"] and acc["k"]["path"] is None
    assert acc["m"]["id"] == "m__x" and not acc["m"].get("no_key")


# ------------------------------------------------------------------ story_keys: no key means keyless
def still(p: Path, grey=120):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 36), (grey, grey, grey)).save(p)


def test_story_keys_counts_a_no_key_shot_keyless_and_never_takes_its_rejected_tries(tmp_path, monkeypatch):
    st = SK.Story.__new__(SK.Story)
    st.work, st.keys, st.look_p = tmp_path / "work", tmp_path / "keys", tmp_path / "look.json"
    st.places = {"P": {"slug": "p", "set": "s"}}
    st.keys.mkdir(parents=True)
    monkeypatch.setattr(SK, "sh", lambda *a, **k: 0)
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["A", "B"])
    d = st.work / "p"
    for g, cam in (("pl_g01", "s01"), ("pl_g02", "s02")):
        (d / "seq" / g).mkdir(parents=True)
        (d / "seq" / g / "spec_round0.json").write_text(json.dumps({"cameras": [{"name": cam, "plan_id": "A" if g == "pl_g01" else "B"}]}))
        still(d / "seq" / g / f"{cam}.png")
    still(d / "run" / "pl_g01__s01__s80__r7.png")           # the monster and its re-draw, both rejected
    still(d / "run" / "pl_g01__s01__s80__r7__x.png")
    still(d / "run" / "pl_g02__s02__s80__r7.png", grey=2)    # a nearly black pick ...
    still(d / "run" / "pl_g02__s02__s80__r7__x.png")          # ... whose only other picture was rejected
    (st.keys / "A_s0.png").write_bytes(b"old")                # an earlier run's key for A: must not survive
    acc = {"pl_g01__s01__s80__r7": {"path": None, "id": None, "fails": {"a": ["human face"]},
                                    "no_key": "no try drew the man with a human face in 2 tries",
                                    "rejected": ["pl_g01__s01__s80__r7", "pl_g01__s01__s80__r7__x"]},
           "pl_g02__s02__s80__r7": {"path": str(d / "run" / "pl_g02__s02__s80__r7.png"), "id": "pl_g02__s02__s80__r7",
                                    "fails": {}, "rejected": ["pl_g02__s02__s80__r7__x"]}}
    (d / "run" / "accepted.json").write_text(json.dumps(acc))
    assert st.step_keys("P")
    rows = {r["shot"]: r for r in json.loads((d / "keys.json").read_text())}
    assert not rows["A"]["key"] and rows["A"]["why"] == "no try drew the man with a human face in 2 tries"
    assert not (st.keys / "A_s0.png").exists()
    assert not rows["B"]["key"] and "nearly black" in rows["B"]["why"]       # the rejected re-draw not taken
    assert st.keyless("P") == ["A", "B"]


def test_story_keys_never_falls_back_to_a_rejected_try_when_the_shot_is_missing(tmp_path, monkeypatch):
    st = SK.Story.__new__(SK.Story)
    st.work, st.keys, st.look_p = tmp_path / "work", tmp_path / "keys", tmp_path / "look.json"
    st.places = {"P": {"slug": "p", "set": "s"}}
    st.keys.mkdir(parents=True)
    monkeypatch.setattr(SK, "sh", lambda *a, **k: 0)
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["A"])
    d = st.work / "p"
    (d / "seq" / "pl_g01").mkdir(parents=True)
    (d / "seq" / "pl_g01" / "spec_round0.json").write_text(json.dumps({"cameras": [{"name": "s01", "plan_id": "A"}]}))
    still(d / "seq" / "pl_g01" / "s01.png")
    still(d / "run" / "pl_g01__s01__s80__r7.png")
    # (an accepted.json from another pass names the try rejected but has no row for the shot)
    (d / "run" / "accepted.json").write_text(json.dumps({"other": {"path": None, "id": None, "no_key": "x",
                                                                   "rejected": ["pl_g01__s01__s80__r7"]}}))
    assert st.step_keys("P")
    rows = {r["shot"]: r for r in json.loads((d / "keys.json").read_text())}
    assert not rows["A"]["key"]


class _Two:
    """A judge that answers the redraw's head first, then the engine's."""
    def __init__(self, *answers):
        self.answers = list(answers)

    def ask(self, *a, **k):
        return self.answers.pop(0)


def _due_person():
    return {"head_to_camera_deg": 10, "joints2d": {"FACIAL_L_Eye": [0.48, 0.3, 1], "FACIAL_R_Eye": [0.52, 0.3, 1],
                                                   "chin": [0.5, 0.42], "head_top": [0.5, 0.2]}}


def test_a_helmet_worn_in_the_engine_frame_too_gives_no_verdict():
    from PIL import Image
    im = Image.new("RGB", (1152, 640))
    got = FG.ask(_Two("mask", "mask"), im, _due_person(), im)
    assert got["human_face_ok"] is None and got["human_face"] == "mask"


def test_a_mask_only_in_the_redraw_still_fails():
    from PIL import Image
    im = Image.new("RGB", (1152, 640))
    assert FG.ask(_Two("mask", "human"), im, _due_person(), im)["human_face_ok"] is False
    assert FG.ask(_Two("mask"), im, _due_person())["human_face_ok"] is False
    assert FG.ask(_Two("creature", "mask"), im, _due_person(), im)["human_face_ok"] is False
