"""The flow's use of the box: a step past its usual time is told, a restage touches only its scenes, places run side
by side with the shared library's steps one at a time, and the preflight finds a scene's faults before any GPU work
(story_keys.py over_time / sh / merge_run / run, scenes/preflight.py)."""
import json
import sys
import threading
import time
from pathlib import Path

A = Path(__file__).resolve().parents[1] / "assets"
for d in ("scenes", "redraw", "sets", ""):
    sys.path.insert(0, str(A / d))
import story_keys as SK  # noqa: E402
import preflight as PF  # noqa: E402


def test_a_step_far_past_its_usual_time_is_over_time():
    assert not SK.over_time("stage", 34, 13 * 60)
    assert SK.over_time("stage", 34, 3 * 3600)
    assert not SK.over_time("redraw", 34, 19 * 60)


def test_a_command_past_its_limit_is_stopped(tmp_path):
    t = time.time()
    assert SK.sh([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path / "x.log", limit_s=1.0) == 124
    assert time.time() - t < 10
    assert SK.sh([sys.executable, "-c", "print(1)"], tmp_path / "x.log", limit_s=20.0) == 0
    assert "1" in (tmp_path / "x.log").read_text()


def test_redrawn_scenes_replace_only_their_own_rows(tmp_path):
    run, again = tmp_path / "run", tmp_path / "again"
    run.mkdir(), again.mkdir()
    for n in ("pl_g01__s1__r7.png", "pl_g02__s2__r7.png", "pl_g02__s2__r7__x.png", "pl_g02_a__s3__r7.png"):
        (run / n).write_bytes(b"old")
    (run / "accepted.json").write_text(json.dumps({
        "pl_g01__s1__r7": {"path": str(run / "pl_g01__s1__r7.png")},
        "pl_g02__s2__r7": {"path": str(run / "pl_g02__s2__r7.png")},
        "pl_g02_a__s3__r7": {"path": None, "no_key": "x"}}))
    (again / "pl_g02__s2__r7.png").write_bytes(b"new")
    (again / "accepted.json").write_text(json.dumps({"pl_g02__s2__r7": {"path": str(again / "pl_g02__s2__r7.png")}}))
    assert SK.Story.merge_run(run, again, ["pl_g02", "pl_g02_a"])
    acc = json.loads((run / "accepted.json").read_text())
    assert set(acc) == {"pl_g01__s1__r7", "pl_g02__s2__r7"}
    assert acc["pl_g02__s2__r7"]["path"] == str(run / "pl_g02__s2__r7.png")
    assert (run / "pl_g02__s2__r7.png").read_bytes() == b"new" and (run / "pl_g01__s1__r7.png").read_bytes() == b"old"
    assert not (run / "pl_g02__s2__r7__x.png").exists() and not (run / "pl_g02_a__s3__r7.png").exists()


def _story(tmp_path, places):
    st = SK.Story.__new__(SK.Story)
    st.work, st.keys = tmp_path / "work", tmp_path / "keys"
    st.keys.mkdir()
    st.places = {p: {"slug": p.lower(), "set": p.lower()} for p in places}
    st.ids, st.plan = [], {}
    return st


def test_two_places_run_side_by_side_but_share_the_library_one_at_a_time(tmp_path, monkeypatch):
    st = _story(tmp_path, ["P", "Q"])
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["x"])
    monkeypatch.setattr(SK, "llm_down", lambda: None)
    monkeypatch.setattr(SK, "DRAWN_MAX_SHOTS", 0)
    monkeypatch.setattr(SK.Story, "restage_keyless", lambda self, place: None)
    monkeypatch.setattr(SK.Story, "preflight", lambda self, place: [])
    live, peak, lock = {"cast": 0, "stage": 0}, {"cast": 0, "stage": 0}, threading.Lock()

    def step(name):
        def f(self, place):
            with lock:
                live[name] += 1
                peak[name] = max(peak[name], live[name])
            time.sleep(0.2 if name == "cast" else 0.8)
            with lock:
                live[name] -= 1
            return True
        return f
    for n in ("cast", "stage"):
        monkeypatch.setattr(SK.Story, f"step_{n}", step(n))
    monkeypatch.setattr(SK, "STEPS", ["cast", "stage"])
    st.run(first="cast", last="stage", jobs=2)
    assert peak == {"cast": 1, "stage": 2}                   # the figures one place at a time, the stagings together
    assert all((st.work / p / ".done_stage").exists() for p in ("p", "q"))


def test_one_place_at_a_time_is_the_default(tmp_path, monkeypatch):
    st = _story(tmp_path, ["P", "Q"])
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["x"])
    monkeypatch.setattr(SK, "llm_down", lambda: None)
    order = []
    monkeypatch.setattr(SK.Story, "step_stage", lambda self, place: order.append(place) or True)
    monkeypatch.setattr(SK, "STEPS", ["stage"])
    st.run(first="stage", last="stage")
    assert order == ["P", "Q"]


def _set(tmp_path, fixtures, words=""):
    lib = tmp_path / "library"
    (lib / "prop" / "pc_01").mkdir(parents=True, exist_ok=True)
    (lib / "prop" / "pc_01" / "asset.json").write_text(json.dumps({"description": "a block", "size_m": [1.0, 0.6, 0.8]}))
    sd = lib / "set" / "s1"
    sd.mkdir(parents=True, exist_ok=True)
    (sd / "spec.json").write_text(json.dumps({"size_m": [8, 8, 3], "fixtures": fixtures, "words": words}))
    return sd


def test_preflight_finds_what_a_finished_sheet_showed(tmp_path):
    sd = _set(tmp_path, [{"asset": "pc_01", "name": "work_piece", "position": [0, 0, 0]}],
              "A bare yard. There are no carts in it.")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "pl_g01.json").write_text(json.dumps({"id": "pl_g01", "characters": {
        "a": {"position": [2.0, 0.0], "work_at": "work_piece", "body_pose": {"pose": "stand", "params": {"work_fwd_cm": 77.2}}},
        "b": {"position": [2.1, 0.1], "body_pose": {"pose": "stand"}}},
        "cameras": [{"plan_id": "s1", "carried": {"a": [{"name": "Ada Bell", "at": "hand"}]}}]}))
    (specs / "pl_g02.json").write_text(json.dumps({"id": "pl_g02", "characters": {}, "cameras": [
        {"plan_id": "s2", "insert": True, "position": [0.9, 0.0, 1.5], "target": [3.0, 0.0, 1.0], "_on": None}]}))
    ps = tmp_path / "pass.json"
    ps.write_text(json.dumps({"things": {
        "s1": {"a": [{"what": "bar", "on": "worked", "glow": "red"}]},
        "s2": {"a": [{"what": "bar", "on": "worked"}]},
        "s3": {"a": [{"what": "bar", "on": "worked", "glow": "red"}]}}}))
    got = " | ".join(PF.check(specs, sd, ps, {"Ada Bell"}))
    for want in ("say what is absent", "no name of their own", "77 cm", "stand on one spot", "one of the story's people",
                 "stands against work piece", "does not glow here"):
        assert want in got, (want, got)


def test_preflight_is_quiet_on_a_sound_scene(tmp_path):
    sd = _set(tmp_path, [{"asset": "pc_01", "name": "work_piece", "position": [0, 0, 0], "called": "block"}], "A bare yard.")
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / "pl_g01.json").write_text(json.dumps({"id": "pl_g01", "characters": {
        "a": {"position": [0.0, -0.8], "work_at": "work_piece", "body_pose": {"pose": "stand", "params": {"work_fwd_cm": 60}}},
        "b": {"position": [2.0, 1.0], "body_pose": {"pose": "stand"}}}, "cameras": [{"plan_id": "s1"}]}))
    assert PF.check(specs, sd, None, {"Ada Bell"}) == []


def test_a_step_is_redone_for_one_place_only(tmp_path, monkeypatch):
    st = _story(tmp_path, ["P", "Q"])
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["x"])
    monkeypatch.setattr(SK, "llm_down", lambda: None)
    ran = []
    monkeypatch.setattr(SK.Story, "step_stage", lambda self, place: ran.append(place) or True)
    monkeypatch.setattr(SK, "STEPS", ["stage"])
    for p in ("p", "q"):
        (st.work / p).mkdir(parents=True)
        (st.work / p / ".done_stage").write_text("x")
    st.run(first="stage", last="stage", redo={"q:stage"})
    assert ran == ["Q"]


def test_every_place_is_staged_before_any_key_is_drawn_and_the_keys_are_drawn_together(tmp_path, monkeypatch):
    st = _story(tmp_path, ["P", "Q"])
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["x"])
    monkeypatch.setattr(SK, "llm_down", lambda: None)
    monkeypatch.setattr(SK, "DRAWN_MAX_SHOTS", 0)
    monkeypatch.setattr(SK.Story, "preflight", lambda self, place: [])
    monkeypatch.setattr(SK.Story, "keyless", lambda self, place: [])
    seen = []
    monkeypatch.setattr(SK.Story, "step_stage", lambda self, place, **k: seen.append(("stage", place)) or True)
    monkeypatch.setattr(SK.Story, "step_keys", lambda self, place: seen.append(("keys", place)) or True)
    monkeypatch.setattr(SK.Story, "draw_phase",
                        lambda self, want: seen.append(("draw", tuple(want))) or {p: True for p in want})
    st.run(first="stage", last="keys")
    assert seen == [("stage", "P"), ("stage", "Q"), ("draw", ("P", "Q")), ("keys", "P"), ("keys", "Q")]
    assert all((st.work / p / ".done_keys").exists() for p in ("p", "q"))
    seen.clear()
    st.run(first="stage", last="keys")                       # all done: nothing staged or drawn again
    assert seen == []
    seen.clear()
    st.run(first="stage", last="keys", redo={"q:redraw"})
    assert seen == [("draw", ("Q",)), ("keys", "Q")]


def test_shots_left_without_a_key_are_staged_again_then_drawn_together(tmp_path, monkeypatch):
    st = _story(tmp_path, ["P", "Q"])
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, place: ["x"])
    monkeypatch.setattr(SK, "llm_down", lambda: None)
    monkeypatch.setattr(SK, "DRAWN_MAX_SHOTS", 0)
    monkeypatch.setattr(SK.Story, "preflight", lambda self, place: [])
    left = {"P": ["s1"], "Q": []}
    monkeypatch.setattr(SK.Story, "keyless", lambda self, place: list(left[place]))
    monkeypatch.setattr(SK.Story, "groups_of", lambda self, place, shots: ["pl_g03"])
    seen = []
    monkeypatch.setattr(SK.Story, "step_stage",
                        lambda self, place, only=None, fallback=(): seen.append(("stage", place, only)) or True)
    monkeypatch.setattr(SK.Story, "step_keys", lambda self, place: True)
    monkeypatch.setattr(SK.Story, "draw_phase", lambda self, want: seen.append(("draw", dict(want))) or {p: True for p in want})
    st.run(first="stage", last="keys")
    assert seen[-2:] == [("stage", "P", ["pl_g03"]), ("draw", {"P": ["pl_g03"]})]


def test_the_pictures_are_shared_between_the_drawing_cards(tmp_path, monkeypatch):
    assert SK.split_even(list("abcde"), 2) == [["a", "b", "c"], ["d", "e"]]
    assert SK.split_even(["a"], 2) == [["a"]]
    monkeypatch.setattr(SK, "DRAW_CARDS", ["05", "65"])
    monkeypatch.setattr(SK, "CARD_LINE", {"05": ("image", "refs"), "65": ("engine", "refs")})
    calls = []
    monkeypatch.setattr(SK, "line", lambda station, kind, jid, logf, cmd, env=None: calls.append((station, jid, cmd[-1])) or 0)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text('{"id": 1}\n{"id": 2}\n')
    b.write_text('{"id": 3}\n')
    assert SK.draw_jobs([a, b, tmp_path / "none.jsonl"], "draw_t", tmp_path)
    assert sorted(c[0] for c in calls) == ["engine", "image"]
    got = sorted((tmp_path / f"draw_t_{c}.jsonl").read_text().count("\n") for c in ("05", "65"))
    assert got == [1, 2]
    assert all("qwen_image.py --serve" in c[2] for c in calls)


def test_a_check_takes_each_picture_as_it_lands(tmp_path, monkeypatch):
    import ast
    src = (Path(SK.HERE) / "redraw" / "keycheck.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "as_they_land")
    ns = {"time": time, "Path": Path, "json": json, "FOLLOW_S": 0.0}
    exec(compile(ast.Module([fn], []), "kc", "exec"), ns)
    a, b, c = ({"id": n, "redraw": str(tmp_path / f"{n}.png")} for n in "abc")
    assert [x["id"] for x in ns["as_they_land"]([a, b, c], 0)] == ["a", "b", "c"]      # not following: all, in order
    (tmp_path / "b.png").write_bytes(b"x")
    got = []
    for x in ns["as_they_land"]([a, b, c], 1.2):
        got.append(x["id"])
        if x["id"] == "b":
            (tmp_path / "a.png").write_bytes(b"x")
    assert got == ["b", "a"]                      # as they land; the one never drawn is left out


def test_the_books_own_word_for_a_places_time_of_day_comes_first(tmp_path, monkeypatch):
    st = _story(tmp_path, ["P", "Q"])
    st.book = tmp_path / "book"
    st.book.mkdir()
    (st.book / "place_times.json").write_text(json.dumps({"P": "Night", "Q": "sometime"}))
    monkeypatch.setattr(SK, "llm", lambda *a, **k: 0)
    for p in ("p", "q"):
        (st.work / p).mkdir(parents=True, exist_ok=True)
    (st.work / "q" / "place_time.json").write_text(json.dumps({"time": "day"}))
    st.plan_p = st.timeline = tmp_path / "x"
    assert st.place_time("P") == "night"
    assert st.place_time("Q") == "day"                      # not one of the three: the asked answer stands
