"""A card's lock left with the children of the line's own job is the line's own, not an outside job's
(assets/assetd.py Sched.outside_holders): counted as outside, the cards were released under the line's own drawer
and two drawers waited for each other."""
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))
import assetd as A  # noqa: E402


def holders(monkeypatch, locker, left, sids, child_sids):
    card = types.SimpleNamespace(lock_path="/nowhere/card.lock")
    me = types.SimpleNamespace(child_sids=set(child_sids))
    monkeypatch.setattr(A, "lock_holders", lambda path, deep=False: [(locker, "(exited)")])
    monkeypatch.setattr(A, "fd_holders", lambda path: [f"{p} python" for p in left])
    monkeypatch.setattr(A, "sid_of", lambda p: sids.get(p))
    return A.Sched.outside_holders(me, card)


def test_a_lock_left_with_our_own_jobs_children_is_ours(monkeypatch):
    gone = 2 ** 22 + 12345                                   # a pid that is no process
    assert not Path(f"/proc/{gone}").exists()
    # the locker exited; the two processes that still have the lock open are in our job's session
    assert holders(monkeypatch, gone, [501, 502], {501: 500, 502: 500}, {500}) == []
    # one of them is nobody of ours: an outside job holds the card
    assert holders(monkeypatch, gone, [501, 777], {501: 500, 777: 776}, {500}) == [(gone, "(exited)")]
    # nobody has it open any more: not ours to claim
    assert holders(monkeypatch, gone, [], {}, {500}) == [(gone, "(exited)")]
    # a living locker of another session is outside, whoever else has the file open
    assert holders(monkeypatch, os.getppid(), [501], {501: 500, os.getppid(): 1}, {500}) == [(os.getppid(), "(exited)")]
