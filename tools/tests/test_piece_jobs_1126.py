"""The pieces a set lacks are made each once, two at a time (assets/places/generate.py make_all)."""
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets" / "places"))
sys.path.insert(0, str(HERE.parent / "assets"))
import generate as G  # noqa: E402


def test_each_asset_is_made_once_and_two_at_a_time(tmp_path):
    todo = [{"request": {"asset_id": a}} for a in ("x_tr", "y_tr", "x_tr", "z_tr", "x_tr")]
    now, most, calls, lock = [0], [0], [], threading.Lock()

    def fake(req, library, seed):
        with lock:
            now[0] += 1
            most[0] = max(most[0], now[0])
            calls.append(req["asset_id"])
        time.sleep(0.05)
        with lock:
            now[0] -= 1
        return {"asset_id": req["asset_id"], "ok": True}
    made = G.make_all(todo, tmp_path, 11, make_=fake, jobs=2)
    assert sorted(calls) == ["x_tr", "y_tr", "z_tr"] and set(made) == {"x_tr", "y_tr", "z_tr"}
    assert most[0] == 2 and all(v["ok"] for v in made.values())
    assert G.make_all([], tmp_path, 11, make_=fake) == {}
