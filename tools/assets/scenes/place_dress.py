"""How a place of a story is dressed, as a production designer would dress it for its owner's station.

    python place_dress.py PLACE_SPEC.json OUT.json [--cache DIR] [--url URL]

The director's words for a place tell what the story needs of it; they do not furnish it. Drawn from them alone the
hall of a ruler was bare stone with trestle tables (11-20). The directing model is asked whose place it is, what
it is for and its station (poor, plain, comfortable, rich, royal), and, for a place of means, what such a place of
its period really had on its walls, its floor, overhead and for light. -> {"owner", "use", "station", "dressing"}.
A poor or plain place gets no dressing: it stays as the director has it.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

SEED = 7
STATIONS = ("poor", "plain", "comfortable", "rich", "royal")
DRESSED = ("rich", "royal")              # the stations whose place is drawn anew, dressed
MOST = 14

ASK = """You are the production designer of a period film. One place of the story is to be dressed for the camera.

Place name: {place}
Period and region: {period}
What the director says of it: {room}

1. Whose place is it, and what is it for? Answer from the name and the director's words only.
2. Its station, one word: poor, plain, comfortable, rich or royal. A place of work, a hut, a shed, a stable, a road, a
   wild place is poor or plain however old it is. Only the seat or the house of the wealthy or of a ruler is rich
   or royal.
3. For a rich or royal place only: how was such a place of that period and region really furnished and adorned to
   show its owner's wealth and rank? Between 8 and {most} things, each a short phrase with its material and colour:
   what hangs on the walls, what the walls and pillars are worked with, what covers the floor, what hangs or is
   built overhead, what gives light, the furniture of state, the small riches standing about. Only what that period
   and region had. No people, no animals, nothing that an event of the story brings in. Keep everything the
   director's words name; add to it, never against it. For any other station the list is empty.

Answer as JSON only: {{"owner": "...", "use": "...", "station": "...", "dressing": ["...", ...]}}"""


def ask(url: str, text: str, cache: Path) -> str:
    key = hashlib.sha256(f"place_dress/1\n{SEED}\n{text}".encode()).hexdigest()[:24]
    fp = cache / f"{key}.txt"
    if fp.exists():
        return fp.read_text()
    body = {"model": "default", "messages": [{"role": "user", "content": text}], "temperature": 0.0, "seed": SEED,
            "max_tokens": 1200, "chat_template_kwargs": {"enable_thinking": False}}
    req = Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=900) as r:
        txt = json.loads(r.read())["choices"][0]["message"]["content"].strip()
    cache.mkdir(parents=True, exist_ok=True)
    fp.write_text(txt)
    return txt


def read(txt: str) -> dict:
    """The model's answer as the dressing record; anything unreadable is a plain place with no dressing."""
    m = re.search(r"\{.*\}", txt or "", re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except ValueError:
        j = {}
    st = str(j.get("station") or "").strip().lower()
    st = st if st in STATIONS else "plain"
    dr = [str(x).strip().rstrip(".") for x in (j.get("dressing") or []) if str(x).strip()] if st in DRESSED else []
    return {"owner": str(j.get("owner") or "").strip(), "use": str(j.get("use") or "").strip(), "station": st,
            "dressing": dr[:MOST]}


def prompt(spec: dict) -> str:
    room = re.sub(r"^\s*(inside|outside)\s*:\s*", "", str(spec.get("_director_room") or spec.get("room") or ""), flags=re.I)
    return ASK.format(place=spec.get("place") or "", period=str(spec.get("period") or "not stated").replace("_", " "),
                      room=room.strip() or "not stated", most=MOST)


def main():
    a = sys.argv[1:]
    spec, out = json.loads(Path(a[0]).read_text()), Path(a[1])
    cache = Path(a[a.index("--cache") + 1]) if "--cache" in a else out.parent / "cache"
    url = a[a.index("--url") + 1] if "--url" in a else "http://127.0.0.1:18100/v1/chat/completions"
    rec = read(ask(url, prompt(spec), cache))
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"place": spec.get("place"), "station": rec["station"], "dressing": len(rec["dressing"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
