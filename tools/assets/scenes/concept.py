"""A place's concept picture read into facts the set builder can use (the vision model, on the iGPU).

    python tools/assets/scenes/concept.py CONCEPT.png PLACE_SPEC.json OUT.json

The concept picture is the one wide picture of the place with nobody in it, drawn from the director's words and the
book's art style before anything is built (the shots flow's "hero" picture of the place). It is the place's look:
what kind of space it is, what its walls, floor and ceiling are, where its light comes from, and what stands in
it. Here it is asked closed questions, one fact each:

  space      "room" (a room people walk about in), "cockpit" (a cockpit, cab or carriage: too small to walk about
             in, people only sit) or "outdoors";
  walls, floor, ceiling, light   each in a few words (material and colour; where the light comes from);
  seen       the separate pieces of furniture, fittings and large objects in the picture, each with its look;
  spec       for each piece of the place spec: is one in the picture (yes / no), and its look there.

OUT.json: {"concept", "space", "surfaces": {"walls", "floor", "ceiling"}, "light", "seen": [{"name", "look"}],
"spec": {piece id: {"in_picture": bool, "look": words}}, "surfaces_words": one sentence for a picture's prompt}.
Nothing is decided here: the builder reads these facts (place_build.py, redraw.py).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))            # tools/: the pipeline's own vision judge (characters.vl_ask)

SPACE_ASK = ("What kind of place does this picture show? Answer with exactly one word: room (the inside of a room, "
             "hall, bay or living quarters, on land or aboard a ship: people stand and walk about in it), cockpit (a "
             "cockpit, driving cab or carriage: a space too small to walk about in, where people only sit at its "
             "controls or in its seats), or outdoors.")
ASKS = {
    "walls": "In a few words: what are the walls (or the sides of this space) made of, and what colour are they?",
    "floor": "In a few words: what is the floor made of, and what colour is it?",
    "ceiling": "In a few words: what is overhead (the ceiling, or a canopy or sky), and what colour is it?",
    "light": "In a few words: where does the light in this picture come from, and what colour is it?",
}
SEEN_ASK = ("List every separate piece of furniture, fitting, machine or large object in this picture, one per line, "
            "as: name; what it is made of and its colour. No people. At most ten lines.")


def read(concept: Path, spec: dict, ask) -> dict:
    """`ask`: the vision judge's batch call ([{"id", "image", "ask": {k: question}, "max_new_tokens"}] -> {id:
    answers})."""
    pieces = spec.get("pieces") or []
    items = [{"id": "space", "image": str(concept), "ask": {"space": SPACE_ASK}, "max_new_tokens": 6},
             {"id": "surf", "image": str(concept), "ask": ASKS, "max_new_tokens": 30},
             {"id": "seen", "image": str(concept), "ask": {"seen": SEEN_ASK}, "max_new_tokens": 220}]
    for p in pieces:
        name = (p.get("kind") or "").replace("_", " ")
        items.append({"id": f"p:{p['id']}", "image": str(concept), "max_new_tokens": 30, "ask": {
            "there": f"Is there a {name} in this picture? Answer yes or no.",
            "look": f"If there is a {name} in this picture, say in a few words what it is made of and its colour; "
                    "if there is none, answer none."}})
    a = ask(items)
    sp = re.sub(r"[^a-z]", "", str((a.get("space") or {}).get("space") or "").lower())
    space = next((k for k in ("cockpit", "outdoors", "room") if k in sp), "room")
    surf = {k: re.sub(r"\s+", " ", str((a.get("surf") or {}).get(k) or "")).strip().rstrip(".") for k in ASKS}
    seen = []
    for ln in str((a.get("seen") or {}).get("seen") or "").splitlines():
        ln = re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip()
        if not ln:
            continue
        nm, _, lk = ln.partition(";")
        seen.append({"name": nm.strip().lower(), "look": lk.strip().rstrip(".")})
    sc = {}
    for p in pieces:
        r = a.get(f"p:{p['id']}") or {}
        yes = str(r.get("there") or "").strip().lower().startswith("y")
        lk = re.sub(r"\s+", " ", str(r.get("look") or "")).strip().rstrip(".")
        sc[p["id"]] = {"in_picture": yes, "look": lk if yes and lk.lower() != "none" else ""}
    words = "; ".join(f"the {k}: {v[0].lower() + v[1:]}" for k, v in surf.items() if k != "light" and v)
    return {"concept": str(concept), "space": space, "surfaces": {k: surf[k] for k in ("walls", "floor", "ceiling")},
            "light": surf["light"], "seen": seen[:10], "spec": sc,
            "surfaces_words": (words[0].upper() + words[1:] + "." if words else "")
            + (f" The light: {surf['light'][0].lower() + surf['light'][1:]}." if surf["light"] else "")}


def main():
    concept, spec_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    import characters as CH
    try:
        res = read(concept, json.loads(spec_p.read_text()), CH.vl_ask)
    finally:
        CH.vl_close()
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"space": res["space"], "seen": len(res["seen"]),
                      "in_picture": sum(v["in_picture"] for v in res["spec"].values()), "of": len(res["spec"])}))


if __name__ == "__main__":
    main()
