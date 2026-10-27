"""One lease for the box's GPUs (both XTX cards and the iGPU) and the servers on them.

10-27: jobs of two books ran at once (start_job only locked per book), and each one paused and restarted the voice and
music servers and the language model on its own schedule: one job's "up" loaded Fish onto a card the other job's LTX
held. Now every job that uses the cards or the iGPU holds /tmp/bookcaster-box.lock (flock) for its whole run; a second
job waits, saying for whom; foreground GPU commands take it without waiting or refuse. Server restarts are done only by
the lease holder (or when nobody holds it), and never onto a card whose /tmp/gpu-<bus>.lock is held.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

LOCK = Path(os.environ.get("BOOKCASTER_BOX_LOCK", "/tmp/bookcaster-box.lock"))
CARD_LOCK = os.environ.get("BOOKCASTER_CARD_LOCK", "/tmp/gpu-{bus}.lock")
CARDS = ("05", "65")
_held: dict = {}            # this process's lease: {"fd": file, "info": {...}}


def holder() -> dict:
    """Who holds the lease (book, what, pid, since), {} when nobody does."""
    try:
        with open(LOCK, "a+") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                f.seek(0)
                try:
                    return json.loads(f.read() or "{}") or {"book": "?"}
                except ValueError:
                    return {"book": "?"}
            fcntl.flock(f, fcntl.LOCK_UN)
            return {}
    except OSError:
        return {}


def held() -> bool:
    """True when this process holds the lease."""
    return bool(_held)


def mine_or_free() -> bool:
    """May this process start or stop the box's servers: it holds the lease, or nobody does."""
    return held() or not holder()


def who(h: dict) -> str:
    return f"{h.get('book', '?')}'s {h.get('what', 'job')}" + (f" (since {h['since']})" if h.get("since") else "")


@contextmanager
def lease(book: str, what: str = "job", wait: bool = True, on_wait=None):
    """Hold the box for the block. *wait*: block until it is free, calling on_wait(holder) once first; else raise
    BlockingIOError(holder text) when it is held. Re-entrant within a process."""
    if _held:
        yield _held["info"]
        return
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    f = open(LOCK, "a+")
    try:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.seek(0)
            try:
                h = json.loads(f.read() or "{}")
            except ValueError:
                h = {}
            if not wait:
                raise BlockingIOError(who(h))
            if on_wait:
                on_wait(h)
            fcntl.flock(f, fcntl.LOCK_EX)
        info = {"book": book, "what": what, "pid": os.getpid(), "unit": os.environ.get("BOOKCASTER_JOB", ""),
                "since": time.strftime("%m-%d %H:%M")}
        f.seek(0)
        f.truncate()
        f.write(json.dumps(info))
        f.flush()
        _held.update(fd=f, info=info)
        try:
            yield info
        finally:
            _held.clear()
            f.seek(0)
            f.truncate()
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        f.close()


def card_busy(bus: str) -> bool:
    """True when something holds /tmp/gpu-<bus>.lock (an LTX / Qwen / LoRA run on that card)."""
    p = CARD_LOCK.format(bus=bus)
    if not os.path.exists(p):
        return False
    try:
        with open(p, "a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(f, fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def busy_cards() -> list[str]:
    return [b for b in CARDS if card_busy(b)]
