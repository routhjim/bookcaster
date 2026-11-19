"""A place's concept picture read into facts the set builder can use (the vision model, on the iGPU).

    python tools/assets/scenes/concept.py CONCEPT.png PLACE_SPEC.json OUT.json

The concept picture is the one wide picture of the place with nobody in it, drawn from the director's words and the
book's art style before anything is built (the shots flow's "hero" picture of the place). It is the place's look:
what kind of space it is, what its walls, floor and ceiling are, where its light comes from, and what stands in
it. Here it is asked closed questions, one fact each:

  view       "whole" (the place seen from far enough back) or "part" (a close view of one part of it);
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
    # (a place in the open: what its panorama is drawn as. Drawn from the picture itself, the panorama had the
    # place's buildings in it again, close by, 11-18)
    "far": "In a few words: what is the land like far off behind and around this place, out to the horizon? Leave "
           "out everything built and everything standing near.",
}
# (a place's picture that is a close view of one part of it shows no ground to stand on and nothing of the place:
# the world made from one such was bare land, and every key there had another background, 11-18)
VIEW_ASK = ("Does this picture show a place as a whole, seen from far enough back to take in the ground in front and "
            "what is around it, or a close view of only one part of a place, filling the picture? Answer with exactly "
            "one word: whole or part.")
# (a place in the open: what stands on its ground is built as pieces of the set, each cut from the picture, 11-18)
# (and the smaller things standing about them: with only its buildings made, a yard was two bare blocks on empty
# ground, 11-18)
# (and what stands or lies on open land by nature: told "not plants", a river bank's concept with reeds, a bush,
# stones and a fallen branch listed only two sentences saying there was nothing, and its set had no piece, 11-18)
SEEN_OPEN_ASK = ("List every separate thing standing or lying on the ground in this picture, the largest first: each "
                 "building and structure, then each other solid thing standing by them or on the open ground, then "
                 "each thing of the land itself that stands out on its own: a boulder or large stone, a tree or dead "
                 "tree, a bush, a clump of reeds or tall plants, a fallen log or branch. One per line, as: name; what it "
                 "is made of and its colour. Not the ground itself, grass, water, paths, the sky or the far land. No "
                 "people. At most twelve lines. If there is no such thing, answer none.")
# (a thing made by people, or one of the land itself: what is built in an open place is made as one group piece, and
# the land's own things stand apart from it, each where the picture has it)
MADE_ASK = ("Is the {n} in this picture made by people (a building, a wall, a structure, a made object) or a natural "
            "thing (a rock, a tree, a bush, a plant, fallen wood)? Answer with exactly one word: made or natural.")
OPEN_KIND = " A thing standing on open ground that is none of the other kinds is a landmark."
# (a door, a chimney: listed with the building it is part of, each was made again as a thing of its own)
OWN_ASK = ("Is the {n} in this picture a separate thing standing on the ground by itself, or a part of something "
           "larger, built into it or onto it? Answer with exactly one word: separate or part.")
# (the depth model's metres are near, not exact: a building measured 4 m wide and 3 m high, 11-18)
REAL_ASK = ("About how wide and how tall would the {n} in this picture be in real life, in metres? Answer with two "
            "numbers only: its width, then its height.")
# (a thing of the open made from "stone, grey" was a bare arch: what it is, as its own part of the picture shows it)
WORDS_ASK = ("Describe the main thing in this picture in one sentence of under 40 words, for a builder who must make "
             "it: what it is, its shape, what its sides and its top are made of, and what is built into it and onto "
             "it. Only the thing itself: not the ground, the sky or what stands near it.")
# (a building the picture's edge cuts is cut in its own part of the picture, and made as that part alone)
CUT_ASK = ("Is any building or large structure in this picture cut off by an edge of the picture, so that part of it "
           "is out of view? Answer yes or no.")
# (what is built in a place in the open is made as ONE thing, from the picture: cut up into its buildings, each
# found and named apart, a yard's house and wall were left out and its door was made a thing standing by itself)
GROUP_BOX_ASK = ("Locate everything built that stands on the ground in this picture, taken together as one group: "
                 "every building, wall and structure, with what is built onto them and stands against them, from the "
                 "ground to the highest roof. Not the open ground in front, the sky, plants or the far land. Answer "
                 "only with the group's bounding box as [x1, y1, x2, y2], each number from 0 to 1000: thousandths of "
                 "the picture's width (x) and height (y), x1 y1 the top left corner. If nothing built stands in this "
                 "picture, answer none.")
GROUP_REAL_ASK = ("About how wide and how tall would everything built in this picture, taken together as one group, "
                  "be in real life, in metres? Answer with two numbers only: its width, then its height.")
GROUP_WORDS_ASK = ("Describe everything built in this picture, as one group, in two sentences of under 70 words in "
                   "all, for a builder who must make it: each building, wall and structure from left to right, its "
                   "shape, what its sides and its roof are made of, its doors and openings, and what stands against "
                   "it. Not the open ground, the sky, plants or the far land.")
GROUP_AREA = 0.04                # a group smaller than this share of the picture is no building to make
OPEN_PIECE_AREA = 0.003          # in the open the picture is a wide one: a thing standing by a building is small in it
OPEN_SEEN_MOST = 12
SEEN_ASK = ("List every separate piece of furniture, fitting, machine or large object in this picture, one per line, "
            "as: name; what it is made of and its colour. No people. At most ten lines.")


# (the whole of it: asked plainly, a thing with a part over it, built as one with it, came back as its lower part
# alone, and was made without the upper, 11-18)
BOX_ASK = ("Locate the whole {n} in this picture: every part that belongs to it, from its lowest part to its highest, "
           "whatever is built over it as one with it included. Answer only with its bounding box as [x1, y1, x2, y2], each number from 0 "
           "to 1000: thousandths of the picture's width (x) and height (y), x1 y1 the top left corner.")
KIND_ASK = ("In this picture there is a {n}. Which one of these kinds of thing is it? {kinds}. A part of the room itself "
            "(its walls, floor, ceiling, roof, beams or rafters) is none of them. A thing people work on or at, that "
            "is none of the other kinds, is a work piece.{open} Answer with exactly one of those words, or none when "
            "it is none of them.")


NAME_MOST_WORDS = 7         # a thing's name is a few words: a longer line with no ";" is a sentence about the picture


def thing_name(name: str) -> bool:
    """Whether a line of the judge's list names a thing, not a sentence about the picture ("there are no buildings
    in this picture.": two such lines were taken as things and built, 11-18)."""
    n = re.sub(r"\s+", " ", str(name or "")).strip().lower().rstrip(".")
    if not n or re.match(r"^(none|nothing|no |there |the (picture|image)|this (picture|image)|i |it )", n):
        return False
    return len(n.split()) <= NAME_MOST_WORDS and not re.search(r"\b(picture|image|is|are)\b", n)


def natural(ans) -> bool:
    """The judge's made-or-natural answer says natural."""
    return "natural" in re.sub(r"[^a-z]", "", str(ans or "").lower())


def plain_name(p: dict) -> str:
    """A spec piece as the picture is asked about it: a kind that holds any thing (a work piece, a landmark: its
    look begins with the thing's own name) by that name; else the kind's own words."""
    if p.get("kind") in ("work_piece", "landmark") and str(p.get("look") or "").strip():
        return str(p["look"]).split(",")[0].strip().lower()
    return (p.get("kind") or "").replace("_", " ")
PIECE_AREA = 0.02
STRUCTURE_SPAN = 0.9       # a "thing" this much of the picture's width or more is the room itself (a ceiling, a floor)
CARRY_WIDER = 1.4          # a thing under a piece carries it (a block, a stump, a plinth) when at most this much wider


def same_place(a: list, b: list, k: float = 0.6) -> bool:
    """Two boxes that are mostly the same part of the picture (their overlap over the smaller one's area)."""
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w > 0 and h > 0 and w * h >= k * small


def carried_by(b: list, sb: list) -> bool:
    """Whether the thing in box `sb` carries the piece in box `b`: it starts under the piece's top and reaches
    below the piece's foot, its middle is under the piece, and it is not much wider than the piece (a stump under an
    anvil; a table under a lamp is not: the lamp stands on a far wider thing, which is a piece of its own)."""
    return (b[1] < sb[1] <= b[3] + 0.03 and sb[3] > b[3] + 0.03 and b[0] <= (sb[0] + sb[2]) / 2 <= b[2]
            and sb[2] - sb[0] <= CARRY_WIDER * (b[2] - b[0]))
WALL_ASK = ("Describe only the surface of the walls in this picture, as a builder would: what it is made of, the size and "
            "shape of its plates, panels, boards or blocks, how they are joined and fastened, its colour and its wear. "
            "One sentence. Do not mention furniture, doors, pipes, lights or anything that stands on or against it.")
CLEAN_ASK = ("Does this picture show anything other than one flat wall surface seen straight on: any furniture, "
             "cabinet, door, floor, ceiling, pipe, cable or other object? Answer yes or no.")
NOBODY_ASK = ("Is there any person anywhere in this picture, however small or far off, or a figure standing or sitting? "
              "Answer yes or no.")
BARE = {"walls": "bare stretch of wall with nothing on it or in front of it",
        "floor": "bare stretch of floor with nothing standing on it"}


OPEN_NOT = ("door", "window", "fireplace")       # (kinds of opening in a room's wall: in the open, things)


def real_of(txt) -> list | None:
    """[width, height] in metres from the judge's two numbers, when both are there and sane."""
    m = re.findall(r"\d+(?:\.\d+)?", str(txt or ""))
    v = [float(x) for x in m[:2]]
    return v if len(v) == 2 and all(0.1 <= x <= 200 for x in v) else None


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
    op = res.get("space") == "outdoors"
    items = [{"id": f"b:{p['id']}", "image": str(concept), "max_new_tokens": 30,
              "ask": {"box": BOX_ASK.format(n=plain_name(p)), **({"real": REAL_ASK.format(n=plain_name(p)),
                                                                "made": MADE_ASK.format(n=plain_name(p))} if op else {})}}
             for p in pieces]
    for i, sn in enumerate(res.get("seen") or []):
        items.append({"id": f"s:{i}", "image": str(concept), "max_new_tokens": 30, "ask": {
            "box": BOX_ASK.format(n=sn["name"]),
            "kind": KIND_ASK.format(n=sn["name"], kinds=", ".join(k.replace("_", " ") for k in kinds),
                                    open=OPEN_KIND if op else ""),
            **({"own": OWN_ASK.format(n=sn["name"]), "real": REAL_ASK.format(n=sn["name"]),
                 "made": MADE_ASK.format(n=sn["name"])} if op else {})}})
    for k, words in BARE.items():
        items.append({"id": f"w:{k}", "image": str(concept), "max_new_tokens": 30, "ask": {"box": BOX_ASK.format(n=words)}})
    items.append({"id": "w:words", "image": str(concept), "max_new_tokens": 70, "ask": {"wall": WALL_ASK}})
    a = ask(items)
    for p in pieces:
        b = box_of((a.get(f"b:{p['id']}") or {}).get("box"))
        if b:
            res["spec"][p["id"]].update(box=b, crop=cut(concept, b, out_dir / f"{tag}_{p['id']}.png"))
            if real_of((a.get(f"b:{p['id']}") or {}).get("real")):
                res["spec"][p["id"]]["real_m"] = real_of(a[f"b:{p['id']}"]["real"])
            if op and natural((a.get(f"b:{p['id']}") or {}).get("made")):
                res["spec"][p["id"]]["natural"] = True
    # two pieces of the spec found in the same part of the picture are one thing seen twice (the forge's sword was
    # "found" on the bench, 11-17): it is the piece whose look the picture gave, else the first; the other is not
    # in the picture
    got = [p for p in pieces if res["spec"][p["id"]].get("box")]
    for i, p in enumerate(got):
        for q in got[i + 1:]:
            bp, bq = res["spec"][p["id"]].get("box"), res["spec"][q["id"]].get("box")
            if bp and bq and same_place(bp, bq):
                lose = p if (res["spec"][q["id"]].get("look") and not res["spec"][p["id"]].get("look")) else q
                res["spec"][lose["id"]] = {"in_picture": False, "look": "", "same_as": (q if lose is p else p)["id"]}
    pieces = [p for p in pieces if res["spec"][p["id"]].get("in_picture")]
    for i, sn in enumerate(res.get("seen") or []):
        r = a.get(f"s:{i}") or {}
        kd = re.sub(r"[^a-z ]", "", str(r.get("kind") or "").lower()).strip().replace(" ", "_")
        b = box_of(r.get("box"))
        # (a thing that takes under a fiftieth of the picture is a fitting on a wall, not a piece to build: asked
        # for a kind, the judge called a switch panel a bureau)
        big = bool(b) and (b[2] - b[0]) * (b[3] - b[1]) >= (OPEN_PIECE_AREA if op else PIECE_AREA)
        if op:                               # (in the open: every thing that stands by itself, of whatever kind)
            part = "part" in re.sub(r"[^a-z]", "", str(r.get("own") or "").lower())
            kd = None if part else (kd if kd in kinds and kd not in OPEN_NOT else "landmark")
            # (a thing of the land itself is a landmark made from its own part of the picture, whatever kind the
            # judge named: a bush called a plant would be drawn as a pot plant on a floor)
            if kd and natural(r.get("made")):
                kd = "landmark"
                sn["natural"] = True
        # (and a thing as wide as the picture is the room's own ceiling, floor or wall: the forge's beams came
        # back a "lantern", 11-17)
        room = bool(b) and b[2] - b[0] >= STRUCTURE_SPAN
        # the block, stump or plinth a spec piece stands on is one piece with it: the piece's cut-out and its foot
        # on the floor take it in (the anvil's box ended at the stump's top, so it measured as a low block standing
        # far back against the wall, 11-17)
        on = None if op else next((p for p in pieces if b and res["spec"][p["id"]].get("box")
                                   and carried_by(res["spec"][p["id"]]["box"], b)), None)
        if on:
            pb = res["spec"][on["id"]]["box"]
            pb = [min(pb[0], b[0]), pb[1], max(pb[2], b[2]), b[3]]
            res["spec"][on["id"]].update(box=pb, crop=cut(concept, pb, out_dir / f"{tag}_{on['id']}.png"))
            sn.update(kind=None, carries=on["id"])
            continue
        sn["kind"] = kd if kd in kinds and big and not room else None
        if b and sn["kind"]:
            sn.update(box=b, crop=cut(concept, b, out_dir / f"{tag}_seen{i}.png"))
            if real_of(r.get("real")):
                sn["real_m"] = real_of(r["real"])
    # everything built, as one group: asked only when something of the picture is made by people (on bare land the
    # judge boxed the rocks as "everything built", and they would have been made one block with the rest folded in)
    built = any(not sn.get("natural") for sn in res.get("seen") or [] if sn.get("kind")) or \
        any(not res["spec"][p["id"]].get("natural") for p in pieces if res["spec"][p["id"]].get("box"))
    if op and built:
        g = ask([{"id": "g", "image": str(concept), "max_new_tokens": 160,
                  "ask": {"box": GROUP_BOX_ASK, "real": GROUP_REAL_ASK, "words": GROUP_WORDS_ASK}}]).get("g") or {}
        gb = box_of(g.get("box"))
        if gb and (gb[2] - gb[0]) * (gb[3] - gb[1]) >= GROUP_AREA:
            res["seen"].append({"name": "everything built there", "kind": "landmark", "group": True, "box": gb,
                                "look": "", "crop": cut(concept, gb, out_dir / f"{tag}_group.png"),
                                **({"real_m": real_of(g.get("real"))} if real_of(g.get("real")) else {}),
                                **({"words": re.sub(r"\s+", " ", str(g.get("words") or "")).strip().rstrip(".")}
                                   if str(g.get("words") or "").strip() else {})})
    if op:                                   # each thing of the open said whole, from its own part of the picture
        its = [(res["spec"][p["id"]], f"b:{p['id']}") for p in pieces if res["spec"][p["id"]].get("crop")
               and p.get("kind") == "landmark"]
        its += [(sn, f"s:{i}") for i, sn in enumerate(res.get("seen") or []) if sn.get("kind") == "landmark" and sn.get("crop") and not sn.get("group")]
        w = ask([{"id": k, "image": t["crop"], "max_new_tokens": 80, "ask": {"words": WORDS_ASK}} for t, k in its]) if its else {}
        for t, k in its:
            ws = re.sub(r"\s+", " ", str((w.get(k) or {}).get("words") or "")).strip().rstrip(".")
            if ws:
                t["words"] = ws
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
             {"id": "view", "image": str(concept), "ask": {"view": VIEW_ASK, "cut": CUT_ASK}, "max_new_tokens": 6},
             {"id": "surf", "image": str(concept), "ask": ASKS, "max_new_tokens": 30},
             {"id": "seen", "image": str(concept), "ask": {"seen": SEEN_ASK}, "max_new_tokens": 220},
             {"id": "seen_open", "image": str(concept), "ask": {"seen": SEEN_OPEN_ASK}, "max_new_tokens": 300}]
    for p in pieces:
        name = plain_name(p)          # (asked "is there a work piece", the judge never found the anvil, 11-17)
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
    for ln in str((a.get("seen_open" if space == "outdoors" else "seen") or {}).get("seen") or "").splitlines():
        ln = re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip()
        if not ln:
            continue
        nm, _, lk = ln.partition(";")
        if not thing_name(nm):
            continue
        seen.append({"name": nm.strip().lower(), "look": lk.strip().rstrip(".")})
    sc = {}
    for p in pieces:
        r = a.get(f"p:{p['id']}") or {}
        yes = str(r.get("there") or "").strip().lower().startswith("y")
        lk = re.sub(r"\s+", " ", str(r.get("look") or "")).strip().rstrip(".")
        sc[p["id"]] = {"in_picture": yes, "look": lk if yes and lk.lower() != "none" else ""}
    words = "; ".join(f"the {k}: {v[0].lower() + v[1:]}" for k, v in surf.items() if k not in ("light", "far") and v)
    m_ = re.search(r"\d+", str((a.get("seats") or {}).get("abreast") or ""))
    abreast = max(1, min(4, int(m_.group(0)))) if m_ else 1
    if re.sub(r"[^a-z]", "", str((a.get("seats") or {}).get("single") or "").lower()).startswith("one"):
        abreast = 1
    elif abreast < 2 and (a.get("seats") or {}).get("single"):
        abreast = 2
    view = "part" if "part" in re.sub(r"[^a-z]", "", str((a.get("view") or {}).get("view") or "").lower()) else "whole"
    cut_ = re.sub(r"[^a-z]", "", str((a.get("view") or {}).get("cut") or "").lower()).startswith("y")
    return {"concept": str(concept), "space": space, "view": view, "cut": cut_, "seats_abreast": abreast, "surfaces": {k: surf[k] for k in ("walls", "floor", "ceiling")},
            "light": surf["light"], "far": surf["far"],
            "seen": seen[:OPEN_SEEN_MOST if space == "outdoors" else 10], "spec": sc,
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
    if sys.argv[1] == "--nobody":            # concept.py --nobody PICTURE: exit 0 when it shows nobody
        try:
            r = CH.vl_ask([{"id": "c", "image": sys.argv[2], "max_new_tokens": 4, "ask": {"who": NOBODY_ASK}}])
        finally:
            CH.vl_close()
        ans = str((r.get("c") or {}).get("who") or "").strip().lower()
        print(json.dumps({"picture": sys.argv[2], "someone": ans}))
        sys.exit(0 if ans.startswith("n") else 1)
    if sys.argv[1] == "--map":               # concept.py --map MAP.png PLACES.json OUT.json [KEY.png]: a ground map read
        sys.path.insert(0, str(HERE.parent / "places"))
        import numpy as np
        from PIL import Image
        import ground_map as GM
        places = json.loads(Path(sys.argv[3]).read_text()).get("land_places") or {}
        try:
            # (KEY.png: the map with its water painted the key colour by the edit model; without it the judge's outline)
            key = sys.argv[5] if len(sys.argv) > 5 and Path(sys.argv[5]).exists() else None
            got = GM.read_map(sys.argv[2], places, CH.vl_ask, rgb=np.asarray(Image.open(sys.argv[2]).convert("RGB")),
                              key_rgb=np.asarray(Image.open(key).convert("RGB")) if key else None)
        finally:
            CH.vl_close()
        Path(sys.argv[4]).write_text(json.dumps(got, indent=1))
        print(json.dumps({"water_share": (got["water"] or {}).get("share", 0.0), "water_from": got["water_from"],
                          "places": len(got["places"]),
                          "of": len(places)}))
        sys.exit(0)
    concept, spec_p, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    try:
        spec = json.loads(spec_p.read_text())
        res = read(concept, spec, CH.vl_ask)
        if res["space"] in ("room", "outdoors"):   # (a cockpit is one piece made from the whole picture)
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
