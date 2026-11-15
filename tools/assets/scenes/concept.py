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


BOX_ASK = ("Locate the {n} in this picture. Answer only with its bounding box as [x1, y1, x2, y2], each number from 0 "
           "to 1000: thousandths of the picture's width (x) and height (y), x1 y1 the top left corner.")
KIND_ASK = ("In this picture there is a {n}. Which one of these kinds of thing is it? {kinds}. Answer with exactly one "
            "of those words, or none when it is none of them.")
PIECE_AREA = 0.02
WALL_ASK = ("Describe only the surface of the walls in this picture, as a builder would: what it is made of, the size and "
            "shape of its plates, panels, boards or blocks, how they are joined and fastened, its colour and its wear. "
            "One sentence. Do not mention furniture, doors, pipes, lights or anything that stands on or against it.")
CLEAN_ASK = ("Does this picture show anything other than one flat wall surface seen straight on: any furniture, "
             "cabinet, door, floor, ceiling, pipe, cable or other object? Answer yes or no.")
BARE = {"walls": "bare stretch of wall with nothing on it or in front of it",
        "floor": "bare stretch of floor with nothing standing on it"}


def box_of(txt) -> list | None:
    """[x0, y0, x1, y1] in 0..1 from the judge's answer, or None when it gave no box of some size."""
    v = [int(x) for x in re.findall(r"\d+", str(txt or ""))[:4]]
    if len(v) < 4:
        return None
    x0, y0, x1, y1 = [min(1000, max(0, t)) / 1000.0 for t in v]
    return [x0, y0, x1, y1] if x1 - x0 > 0.03 and y1 - y0 > 0.03 else None


def cut(concept: Path, box: list, dest: Path, margin: float = 0.03) -> str:
    from PIL import Image
    im = Image.open(concept).convert("RGB")
    w, h = im.size
    x0, y0, x1, y1 = box
    im.crop((int(max(0, x0 - margin) * w), int(max(0, y0 - margin) * h), int(min(1, x1 + margin) * w),
             int(min(1, y1 + margin) * h))).save(dest)
    return str(dest)


def colour(concept: Path, box: list) -> str:
    """The middle colour of a stretch of the picture (its median, so a rivet or a seam does not tint it)."""
    from PIL import Image
    im = Image.open(concept).convert("RGB")
    w, h = im.size
    px = sorted(im.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))).resize((32, 32)).getdata(),
                key=sum)
    return "#%02x%02x%02x" % px[len(px) // 2]


def pieces_from(concept: Path, spec: dict, res: dict, ask, out_dir: Path, tag: str, kinds: list) -> dict:
    """The concept picture as the source of what is built: where each piece of the spec stands in it (its box, and
    that part of the picture cut out, the drawing its 3D shape is made from); the things it shows that the spec does
    not have, each with its kind (a closed list) and its own cut-out; and a bare stretch of wall and of floor, cut
    out, with its colour. Written into `res` ("spec" entries gain "box" and "crop"; "seen" entries gain "kind",
    "box", "crop"; "bare": {"walls"|"floor": {"box", "crop", "colour"}})."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pieces = [p for p in spec.get("pieces") or [] if (res["spec"].get(p["id"]) or {}).get("in_picture")]
    items = [{"id": f"b:{p['id']}", "image": str(concept), "max_new_tokens": 30,
              "ask": {"box": BOX_ASK.format(n=(p.get("kind") or "").replace("_", " "))}} for p in pieces]
    for i, sn in enumerate(res.get("seen") or []):
        items.append({"id": f"s:{i}", "image": str(concept), "max_new_tokens": 30, "ask": {
            "box": BOX_ASK.format(n=sn["name"]),
            "kind": KIND_ASK.format(n=sn["name"], kinds=", ".join(k.replace("_", " ") for k in kinds))}})
    for k, words in BARE.items():
        items.append({"id": f"w:{k}", "image": str(concept), "max_new_tokens": 30, "ask": {"box": BOX_ASK.format(n=words)}})
    items.append({"id": "w:words", "image": str(concept), "max_new_tokens": 70, "ask": {"wall": WALL_ASK}})
    a = ask(items)
    for p in pieces:
        b = box_of((a.get(f"b:{p['id']}") or {}).get("box"))
        if b:
            res["spec"][p["id"]].update(box=b, crop=cut(concept, b, out_dir / f"{tag}_{p['id']}.png"))
    for i, sn in enumerate(res.get("seen") or []):
        r = a.get(f"s:{i}") or {}
        kd = re.sub(r"[^a-z ]", "", str(r.get("kind") or "").lower()).strip().replace(" ", "_")
        b = box_of(r.get("box"))
        # (a thing that takes under a fiftieth of the picture is a fitting on a wall, not a piece to build: asked
        # for a kind, the judge called a switch panel a bureau)
        big = bool(b) and (b[2] - b[0]) * (b[3] - b[1]) >= PIECE_AREA
        sn["kind"] = kd if kd in kinds and big else None
        if b and sn["kind"]:
            sn.update(box=b, crop=cut(concept, b, out_dir / f"{tag}_seen{i}.png"))
    res["bare"] = {}
    for k in BARE:
        b = box_of((a.get(f"w:{k}") or {}).get("box"))
        if b:
            res["bare"][k] = {"box": b, "crop": cut(concept, b, out_dir / f"{tag}_{k}.png", 0.0), "colour": colour(concept, b)}
    ww = re.sub(r"\s+", " ", str((a.get("w:words") or {}).get("wall") or "")).strip()
    if ww and "walls" in res["bare"]:
        res["bare"]["walls"]["words"] = ww
    return res


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
    items.append({"id": "seats", "image": str(concept), "max_new_tokens": 4, "ask": {
        "abreast": "How many seats for people are side by side in this picture? Answer with one number.",
        # (the count alone came back "2" for every cabin picture, one seat or three, 11-14: a closed question first)
        "single": "Does this cabin have one seat for a single person, or more than one seat? Answer one or more."}})
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
    m_ = re.search(r"\d+", str((a.get("seats") or {}).get("abreast") or ""))
    abreast = max(1, min(4, int(m_.group(0)))) if m_ else 1
    if re.sub(r"[^a-z]", "", str((a.get("seats") or {}).get("single") or "").lower()).startswith("one"):
        abreast = 1
    elif abreast < 2 and (a.get("seats") or {}).get("single"):
        abreast = 2
    return {"concept": str(concept), "space": space, "seats_abreast": abreast, "surfaces": {k: surf[k] for k in ("walls", "floor", "ceiling")},
            "light": surf["light"], "seen": seen[:10], "spec": sc,
            "surfaces_words": (words[0].upper() + words[1:] + "." if words else "")
            + (f" The light: {surf['light'][0].lower() + surf['light'][1:]}." if surf["light"] else "")}


def main():
    import characters as CH
    if sys.argv[1] == "--clean":             # concept.py --clean PICTURE: exit 0 when it shows one bare wall only
        try:
            r = CH.vl_ask([{"id": "c", "image": sys.argv[2], "max_new_tokens": 4, "ask": {"other": CLEAN_ASK}}])
        finally:
            CH.vl_close()
        ans = str((r.get("c") or {}).get("other") or "").strip().lower()
        print(json.dumps({"picture": sys.argv[2], "other_things": ans}))
        sys.exit(0 if ans.startswith("n") else 1)
    concept, spec_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    try:
        spec = json.loads(spec_p.read_text())
        res = read(concept, spec, CH.vl_ask)
        if res["space"] == "room":             # (a cockpit is one piece made from the whole picture)
            kinds = sorted(json.loads((HERE.parent / "places" / "piece_kinds.json").read_text())["kinds"])
            tag = re.sub(r"[^a-z0-9]+", "_", str(spec.get("place") or out.stem).lower()).strip("_")[:24]
            pieces_from(concept, spec, res, CH.vl_ask, out.parent / "concept_crops", tag, kinds)
    finally:
        CH.vl_close()
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"space": res["space"], "seen": len(res["seen"]),
                      "in_picture": sum(v["in_picture"] for v in res["spec"].values()), "of": len(res["spec"])}))


if __name__ == "__main__":
    main()
