"""A place's results are kept before they are made again, an approved place is left alone, and a change of rules
is told as what it would move (assets/place_keep.py, story_keys.py)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
import place_keep as PK  # noqa: E402


def place(tmp_path, x=1.0):
    d, keys, sd = tmp_path / "room", tmp_path / "keys", tmp_path / "set"
    for q in (d / "specs", d / "seq" / "pl_g01", keys, sd):
        q.mkdir(parents=True, exist_ok=True)
    spec = {"characters": {"a": {"position": [x, 0.0], "facing": [0, 1], "seat": None,
                                 "body_pose": {"pose": "stand_relaxed"}}, "b": "b"},
            "cameras": [{"name": "s01", "plan_id": "shot-1", "kind": "medium", "subject": "a"}]}
    (d / "specs" / "pl_g01.json").write_text(json.dumps(spec))
    (d / "seq" / "pl_g01" / "spec_round0.json").write_text(json.dumps(spec))
    (d / "seq" / "pl_g01" / "s01.png").write_bytes(b"still")
    (d / "pass.json").write_text("{}")
    (keys / "shot-1_s0.png").write_bytes(b"key one")
    (sd / "spec.json").write_text('{"v": 1}')
    return d, keys, sd


def test_a_place_made_again_can_be_put_back_as_it_was(tmp_path):
    d, keys, sd = place(tmp_path)
    snap = PK.snapshot(d, keys, ["shot-1"], sd, "t1")
    (keys / "shot-1_s0.png").write_bytes(b"key two")
    (sd / "spec.json").write_text('{"v": 2}')
    (d / "specs" / "pl_g01.json").write_text("{}")
    assert PK.restore(d, keys, sd, "t1") == snap
    assert (keys / "shot-1_s0.png").read_bytes() == b"key one" and json.loads((sd / "spec.json").read_text()) == {"v": 1}
    assert json.loads((d / "specs" / "pl_g01.json").read_text())["cameras"]


def test_an_approved_state_is_never_dropped_and_is_the_one_restored(tmp_path):
    d, keys, sd = place(tmp_path)
    assert PK.approved(d) is None
    ap = PK.approve(d, keys, ["shot-1"], sd)
    for i in range(PK.KEEP + 3):
        PK.snapshot(d, keys, ["shot-1"], sd, f"later{i:02d}")
    assert ap.is_dir() and PK.approved(d)["snapshot"] == ap.name
    assert len(PK.snapshots(d)) <= PK.KEEP + 1
    (keys / "shot-1_s0.png").write_bytes(b"worse")
    PK.restore(d, keys, sd)
    assert (keys / "shot-1_s0.png").read_bytes() == b"key one"
    PK.unapprove(d)
    assert PK.approved(d) is None


def test_what_a_change_of_rules_would_move_is_told_shot_by_shot(tmp_path):
    d, _, _ = place(tmp_path)
    d2, _, _ = place(tmp_path / "n", x=1.1)
    assert PK.drift(d / "specs", d2 / "specs") == []                 # ten centimetres: the same place
    d3, _, _ = place(tmp_path / "m", x=3.0)
    g = json.loads((d3 / "specs" / "pl_g01.json").read_text())
    g["characters"]["a"].update(seat="bench", facing=[1, 0])
    g["cameras"][0].update(kind="full", position=[0, -3, 1.5])
    (d3 / "specs" / "pl_g01.json").write_text(json.dumps(g))
    told = " | ".join(PK.drift(d / "specs", d3 / "specs"))
    for w in ("stands 2.0 m", "sits on bench", "turned 90", "camera is now given", "kind is full"):
        assert w in told, (w, told)


def test_the_flow_runs_no_step_of_an_approved_place(tmp_path, monkeypatch):
    import story_keys as SK
    s = SK.Story.__new__(SK.Story)
    s.work, s.keys = tmp_path / "w", tmp_path / "keys"
    s.keys.mkdir()
    s.places = {"Room One": {"slug": "room_one", "set": "x_room_one_01"}, "Room Two": {"slug": "room_two", "set": "x2"}}
    ran = []
    monkeypatch.setattr(SK.Story, "run_place", lambda self, p, todo, redo: ran.append(p))
    monkeypatch.setattr(SK.Story, "draw_all", lambda self, places, after, redo=(): ran.append(tuple(places)))
    monkeypatch.setattr(SK.Story, "shots_of", lambda self, p: [])
    monkeypatch.setattr(SK.Story, "keyless", lambda self, p: [])
    PK.approve(s.d("Room One"), s.keys, [], None)
    s.run(first="pass", last="keys", redo={"pass"})
    assert ran == ["Room Two", ("Room Two",)]
    ran.clear()
    s.run(first="pass", last="keys", redo={"pass"}, unlock={"room_one"})
    assert ran[:2] == ["Room One", "Room Two"]
