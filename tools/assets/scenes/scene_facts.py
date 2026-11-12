"""A scene's fixed facts as a short plain list, for the video model's prompt (stdlib).

    python tools/assets/scenes/scene_facts.py SET_DIR OUT.txt

From the set's own facts (setfacts.py): the time of day and the window light, every lit source (lit for the whole
scene), and each large piece of furniture with its look and its place by the room's landmarks. These hold in every
shot of the scene: a fire that burns in one key must burn in the next, a chair keeps its colour.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "redraw"))
import setfacts  # noqa: E402


def plain(t: dict) -> str:
    """The thing's own words without the lit-state tail the redraw adds."""
    return t["words"].split(" with a fire burning")[0].split(", lit, its flame")[0]


def facts(set_dir: Path) -> list:
    th = setfacts.things(set_dir)
    out = []
    day = setfacts.daylight(set_dir)
    wins = [t for t in th if t["kind"] == "window"]
    if day:
        out.append(f"Time of day: {'day' if day == 'daylight' else 'late afternoon, low sun'}.")
    for w in wins:
        out.append(f"Window light: {day or 'no daylight'} comes in through {w['words']}"
                   + (f" on the {w.get('wall')} wall" if w.get("wall") else "")
                   + (f", showing {w['view']}" if w.get("view") else "") + ".")
    for t in th:
        if t.get("fire"):
            out.append(f"Lit for the whole scene: the fire in {plain(t)} burns with real flames "
                       "and glowing coals.")
        elif t.get("lit"):
            out.append(f"Lit for the whole scene: {plain(t)}, its flame burning.")
    for t in th:
        if t.get("large") and not t.get("fire"):
            out.append(f"{plain(t)[0].upper() + plain(t)[1:]}"
                       + (f", {t['place']}" if t.get("place") else "") + ": the same look in every shot.")
    return out


def main():
    lines = facts(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text("\n".join(f"- {x}" for x in lines) + "\n")
    print(sys.argv[2], len(lines))


if __name__ == "__main__":
    main()
