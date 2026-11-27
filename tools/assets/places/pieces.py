"""A place spec's pieces -> library assets: lookup first, a generation request second (place_spec/1).

    kinds()                         the piece kinds (piece_kinds.json beside this file; PIECE_KINDS is its names)
    word_to_kind(phrase)            a plan phrase ("the chair by the window") -> one kind name, mechanically
    lookup(piece, records)          one piece {kind, look[, size]} -> [{asset_id, score, why}], best first
    generation_request(piece)       {kind, look, size_m, needs} + the dry-run plan of the existing prop route
    resolve(pieces, records)        every piece: a library id, or a generation request, or unresolved with why
    submit(request, really=False)   hands a request's jobs to the line; refuses unless really=True

    python tools/assets/places/pieces.py PIECES.json [--library DIR] [--threshold T] [--allow-scripted]

Nothing here draws, shapes, bakes or writes into the library: lookup reads the prop records (asset.json), and a
generation request is a description of the jobs the prop route would run (reference drawing, picture-to-3D, the
asset chain, the support measured from the mesh). Judgment about what a room needs is the place spec's (written by
the language model); this module only matches words and records.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(1, str(HERE))                      # (ground_map, beside this file)
import config as C  # noqa: E402

KINDS_FILE = HERE / "piece_kinds.json"
DEFAULT_THRESHOLD = 0.55
WEIGHTS = {"kind": 0.40, "look": 0.35, "colour": 0.20, "size": 0.05}
SCRIPTED_FACTOR = 0.8          # a hand-scripted model is a weaker match than a generated one of the same words
MAX_REDRAWS = 3                # reference drawings tried for one piece before it is reported unresolved

_STOP = set("a an the of in on at by to with and or for from his her its their my our your this that these those "
            "some one two very quite rather old new little small large big great".split())
# words that end the piece's own name in a phrase: what follows says where it is, not what it is
_PREP = set("by beside near next at in on under over behind before against opposite facing towards toward from "
            "beneath above below between across along inside outside".split())
_SIZE_WORDS = {"small": "small", "little": "small", "low": "small", "narrow": "small", "medium": "medium",
               "large": "large", "big": "large", "great": "large", "tall": "large", "wide": "large",
               "massive": "large", "heavy": "large"}
# colour words -> CIE Lab (rough studio values; a record's stored albedo median, when it has one, is compared with
# these; otherwise the record's own colour words are)
COLOURS = {
    "black": (12, 0, 0), "ebony": (14, 1, 1), "white": (92, 0, 2), "cream": (88, 1, 14), "ivory": (90, 0, 10),
    "grey": (55, 0, 0), "gray": (55, 0, 0), "silver": (72, 0, 1), "iron": (30, 0, 1), "brown": (35, 14, 22),
    "mahogany": (25, 22, 16), "walnut": (30, 12, 18), "oak": (55, 10, 30), "pine": (72, 8, 34),
    "tan": (62, 12, 28), "beige": (78, 4, 16), "red": (42, 58, 38), "crimson": (36, 58, 30),
    "burgundy": (24, 36, 12), "oxblood": (22, 30, 14), "maroon": (22, 32, 12), "scarlet": (46, 66, 46),
    "pink": (76, 26, 6), "orange": (62, 38, 62), "gold": (68, 6, 52), "golden": (68, 6, 52),
    "yellow": (84, 0, 70), "brass": (62, 6, 44), "bronze": (45, 12, 34), "copper": (52, 28, 34),
    "green": (42, -38, 28), "olive": (46, -10, 36), "blue": (38, 8, -42), "navy": (18, 8, -26),
    "purple": (32, 38, -34), "violet": (38, 36, -40),
}


def kinds(path=None):
    """{kind: {synonyms, parent, size_m, needs, placement}} from the data file."""
    return json.loads(Path(path or KINDS_FILE).read_text())["kinds"]


def phrase_only(path=None):
    """Synonyms that name a kind in a plan phrase ("by the fire") but are too loose to classify a library record."""
    return set(json.loads(Path(path or KINDS_FILE).read_text()).get("phrase_only", []))


PIECE_KINDS = tuple(kinds())


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", str(s).lower().replace("-", " ").replace("_", " ").replace("'s", " "))


def _tokens(s):
    return [t for t in _norm(s).split() if t]


def _syn_table(K, skip=()):
    """[(synonym tokens, kind)] longest first; a kind's own name is a synonym."""
    out = []
    for k, d in K.items():
        for s in set(d.get("synonyms", [])) | {k.replace("_", " ")}:
            if s in skip:
                continue
            out.append((tuple(_tokens(s)), k))
    return sorted(out, key=lambda x: (-len(x[0]), x[1], x[0]))


def _depth(kind, K):
    n = 0
    while K.get(kind, {}).get("parent"):
        kind, n = K[kind]["parent"], n + 1
    return n


def _find(tokens, table, K):
    """Every synonym found in a token list: [(start, length, kind)], overlaps resolved longest then most specific."""
    hits = []
    for syn, kind in table:
        n = len(syn)
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i:i + n]) == syn or (n == 1 and tokens[i] in (syn[0], syn[0] + "s")):
                hits.append((i, n, kind))
    hits.sort(key=lambda h: (-h[1], -_depth(h[2], K), h[0], h[2]))
    taken, out = set(), []
    for i, n, kind in hits:
        span = set(range(i, i + n))
        if span & taken:
            continue
        taken |= span
        out.append((i, n, kind))
    return sorted(out)


def word_to_kind(phrase, K=None):
    """A plan phrase -> {"kind": name or None, "matched": the words, "related": kinds named after a place word,
    "unresolved": the head words no kind covers}. The piece is what the phrase names before its first place word
    ("the chair by the window" is a chair; the window is where). No guessing: no kind found -> kind None."""
    K = K or kinds()
    toks = _tokens(phrase)
    cut = next((i for i, t in enumerate(toks) if t in _PREP and i > 0), len(toks))
    head, tail = toks[:cut], toks[cut:]
    table = _syn_table(K)
    found = _find(head, table, K)
    related = [k for _, _, k in _find(tail, table, K)]
    if not found:
        return {"kind": None, "matched": "", "related": related,
                "unresolved": [t for t in head if t not in _STOP]}
    i, n, kind = max(found, key=lambda h: (h[1], _depth(h[2], K), h[0]))   # longest, most specific, last (head noun)
    covered = set().union(*[set(range(a, a + b)) for a, b, _ in found])
    return {"kind": kind, "matched": " ".join(head[i:i + n]), "related": related,
            "unresolved": [t for j, t in enumerate(head) if j not in covered and t not in _STOP
                           and t not in _SIZE_WORDS and t not in COLOURS]}


def satisfies(record_kind, wanted, K):
    """0 = no; 1.0 = the same kind; 0.85 = the record is a narrower kind of what is wanted (a wing-back for an
    armchair). A broader record never satisfies a narrower request, unless the narrower kind is marked "role"."""
    if record_kind == wanted:
        return 1.0
    if K.get(wanted, {}).get("role") and K[wanted].get("parent"):
        # a kind that is a use, not a build (a desk chair is any chair at a desk): its parent's records fit it
        # the plain parent kind fits best; a narrower build of it (an armchair for a desk chair) only weakly
        return {1.0: 0.85, 0.85: 0.6}.get(satisfies(record_kind, K[wanted]["parent"], K), 0.0)
    k = record_kind
    while K.get(k, {}).get("parent"):
        k = K[k]["parent"]
        if k == wanted:
            return 0.85
    return 0.0


def load_records(library=None):
    """The library's prop records, read only: [{asset_id, tags, description, size_m, seat, variant_of, source,
    replaced_by, ok, albedo_lab}]. A record another record replaces (its remade version) is marked."""
    root = Path(library or C.LIBRARY) / "prop"
    used = {}
    for sp in sorted((root.parent / "set").glob("*/spec.json")):
        try:
            sd = json.loads(sp.read_text())
        except (OSError, ValueError):
            continue
        for fx in list(sd.get("fixtures", [])) + list(sd.get("props", [])):
            names = [fx.get("asset")] + [v.get("asset") for v in (fx.get("variants") or {}).values()
                                         if isinstance(v, dict)] if isinstance(fx, dict) else []
            for a in names:
                if a and sp.parent.name not in used.setdefault(a, []):
                    used[a].append(sp.parent.name)
    recs = []
    for f in sorted(root.glob("*/asset.json")) + sorted((root.parent / "vehicle").glob("*/asset.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        spec = {}
        sp = f.parent / "source" / "spec.json"
        if sp.exists():
            try:
                spec = json.loads(sp.read_text())
            except ValueError:
                spec = {}
        status = str(d.get("status", "")).lower()
        aid = d.get("id", f.parent.name)
        recs.append({"asset_id": aid, "tags": list(d.get("tags", [])),
                     "description": d.get("description", ""), "size_m": d.get("size_m"),
                     "dims_m": (d.get("blender") or {}).get("dims_m"), "_replaced_by": d.get("replaced_by"),
                     "seat": d.get("seat"), "lie": d.get("lie"), "variant_of": d.get("variant_of"),
                     "placement": d.get("placement"), "panorama_people": d.get("panorama_people"),
                     "variant": d.get("variant"), "look": d.get("look", ""), "piece_kind": d.get("piece_kind"),
                     "source": d.get("source", ""), "replaces": d.get("replaces") or spec.get("replaces"),
                     "ok": "not_passed" not in status and "delete" not in status,
                     "checks": d.get("checks_passed", True) is not False, "used_in_sets": used.get(aid, []),
                     "albedo_lab": d.get("albedo_lab") or (d.get("albedo") or {}).get("median_lab")})
    # a remade record hides the one it replaces only when the remake passed its own checks
    replaced = {r["replaces"]: r["asset_id"] for r in recs if r["replaces"] and r["ok"] and r["checks"]}
    for r in recs:
        r["replaced_by"] = replaced.get(r["asset_id"]) or r.pop("_replaced_by", None)   # (or its own record says so)
    return recs


def record_kinds(rec, K=None, table=None):
    """The kinds a record is: from its id when the id names a kind (the id is the piece's name), else from its
    tags and the first clause of its description. Loose words (the data file's phrase_only) never classify."""
    K = K or kinds()
    table = table or _syn_table(K, phrase_only())
    by_id = {k for _, _, k in _find(_tokens(rec["asset_id"]), table, K)}
    first = re.split(r"[,:;.]", rec.get("description", ""), maxsplit=1)[0]
    by_words = set()
    for part in [str(t) for t in rec.get("tags", [])] + [first]:      # each tag alone: no match across two tags
        by_words |= {k for _, _, k in _find(_tokens(part), table, K)}
    # the id names the piece; its tags may only narrow that (an id "bed" with a tag "single bed")
    ks = by_id | {k for k in by_words if any(satisfies(k, i, K) for i in by_id)} if by_id else by_words
    # only the most specific kinds stay: a wing-back is not also listed as a plain chair
    return sorted(k for k in ks if not any(o != k and satisfies(o, k, K) for o in ks))


def _lab_of_words(tokens):
    labs = [COLOURS[t] for t in tokens if t in COLOURS]
    if not labs:
        return None
    return tuple(sum(c[i] for c in labs) / len(labs) for i in range(3))


SEEN_SIZE_K = (0.5, 1.8)           # a piece's size from its concept picture is kept within this much of its kind's


def _size_class(size_m, default):
    if not size_m or not default:
        return None
    r = max(size_m) / max(default)
    return "small" if r < 0.8 else "large" if r > 1.25 else "medium"


SHELL_SPAN_K = 1.5                      # a cabin shell may be at most this much longer than its kind's box
SEAT_FIT_W_M = 0.48                     # a seat with arms narrower than this between them takes no sitter


def seat_too_narrow(rec: dict) -> bool:
    """A record of a seat with arms (its measured "seat" block has arm_z_m) whose room between the arms (width_m, or
    the inner_x_m span) is under SEAT_FIT_W_M: nobody fits in it. A chair 0.295 m between its arms was reused for a
    grand chair and its sitter was seated on a bench at the other end of the room (11-20)."""
    b = rec.get("seat") if isinstance(rec.get("seat"), dict) else None
    if not b or b.get("arm_z_m") is None:
        return False
    w = b.get("width_m")
    if not isinstance(w, (int, float)) and isinstance(b.get("inner_x_m"), (list, tuple)) and len(b["inner_x_m"]) == 2:
        w = float(b["inner_x_m"][1]) - float(b["inner_x_m"][0])
    return isinstance(w, (int, float)) and float(w) < SEAT_FIT_W_M


def shell_too_big(size_m, want_m) -> bool:
    """A cabin shell's longest floor side more than SHELL_SPAN_K times the size asked for. A shell takes its size
    from its panorama's depth, and a panorama drawn with more in it than the one seat (a second seat ahead with
    someone in it, 11-15) gave a cabin 7 m long that no stage held."""
    return bool(size_m and want_m and len(size_m) >= 2 and len(want_m) >= 2
                and max(size_m[:2]) > SHELL_SPAN_K * max(want_m[:2]))


def score_record(piece, rec, K=None, table=None, allow_scripted=False):
    """One record against one piece: None when the kind does not fit, else {score, why}."""
    K = K or kinds()
    kind = piece["kind"]
    if kind not in K:
        return None
    fits = [satisfies(rk, kind, K) for rk in record_kinds(rec, K, table)]
    k = max(fits, default=0.0)
    if k == 0.0 or not rec.get("ok", True) or rec.get("replaced_by"):
        return None
    # a thing made to hang on a wall does not stand on a table or the floor (11-12: a drop-dial wall clock stood on
    # the desk in front of the sitter); the record says how the asset is placed, the kind how the piece is
    if rec.get("placement") == "wall" and K[kind].get("placement") != "wall":
        return None
    if K[kind].get("made_as") == "shell" and shell_too_big(rec.get("size_m"), K[kind].get("size_m")):
        return None
    if K[kind].get("needs") == "seat" and K[kind].get("made_as") != "shell" and seat_too_narrow(rec):
        return None                       # (an armed seat nobody fits in is never a seat piece: made anew)
    look_toks = [t for t in _tokens(piece.get("look", "")) if t not in _STOP]
    rec_toks = set(_tokens(" ".join([rec.get("description", ""), rec.get("look", ""), rec.get("variant") or ""]
                                    + [str(t) for t in rec.get("tags", [])])))
    parts = {"kind": k}
    if look_toks:
        hit = [t for t in look_toks if t in rec_toks or t.rstrip("s") in rec_toks]
        parts["look"] = len(hit) / len(look_toks)
    want_lab = _lab_of_words(look_toks)
    have_lab = tuple(rec["albedo_lab"]) if rec.get("albedo_lab") else _lab_of_words(sorted(rec_toks))
    if want_lab and have_lab:
        de = math.dist(want_lab, have_lab)
        parts["colour"] = max(0.0, 1.0 - de / 60.0)
    want_size = piece.get("size") or next((_SIZE_WORDS[t] for t in look_toks if t in _SIZE_WORDS), None)
    have_size = _size_class(rec.get("size_m"), K[kind].get("size_m"))
    if want_size and have_size:
        parts["size"] = 1.0 if want_size == have_size else 0.5 if "medium" in (want_size, have_size) else 0.0
    w = sum(WEIGHTS[p] for p in parts)
    score = sum(WEIGHTS[p] * v for p, v in parts.items()) / w
    scripted = rec.get("source") == "scripted"
    if scripted and not allow_scripted:
        score *= SCRIPTED_FACTOR
    need = K[kind].get("needs")
    why = {"parts": {p: round(v, 3) for p, v in parts.items()}, "scripted": scripted,
           "checks_passed": rec.get("checks", True), "used_in_sets": rec.get("used_in_sets", []),
           "variant_of": rec.get("variant_of"),
           "support": (need if rec.get(need) else f"{need} not measured") if need else None}
    return {"asset_id": rec["asset_id"], "score": round(score, 4), "why": why}


def lookup(piece, records, K=None, threshold=DEFAULT_THRESHOLD, allow_scripted=False):
    """One piece {kind, look[, size]} -> [{asset_id, score, why}] at or above the threshold, best first (ties by
    id, so the answer is the same every time)."""
    K = K or kinds()
    table = _syn_table(K, phrase_only())
    out = [s for s in (score_record(piece, r, K, table, allow_scripted) for r in records) if s]
    out = [s for s in out if s["score"] >= threshold]
    # equal scores: a piece an accepted set already uses first, then one that passed its checks, then by id
    return sorted(out, key=lambda s: (-s["score"], not s["why"]["used_in_sets"], not s["why"]["checks_passed"],
                                      s["asset_id"]))


def _asset_id(kind, look):
    h = hashlib.sha1(f"{kind}|{' '.join(_tokens(look))}".encode()).hexdigest()[:6]
    return f"{kind}_gen_{h}_01"


NEED_WORDS = {"lie": "(a single tier: one mattress at knee height, open to the ceiling, nothing built above it)",
              "seat": "(one seat for one person, open at the front)"}


SHELL_WORDS = (
    # (no person is named anywhere in these words: told whose head the view was from, what they work at and where
    # their elbows come, the drawing put that person in the cabin in four of four panoramas, 11-15)
    "Equirectangular projection, a full 360 degree spherical panorama, 2:1, the left and right edges join seamlessly. "
    "The inside of an empty {name}, seen from the eye point just above {seat}: {look}. In the middle of the "
    "picture, straight ahead and within arm's reach of that seat: the controls it faces, and above them large panes "
    "of glass in a heavy frame. To the left and to the right: low consoles along the sides at armrest height, and "
    "above them more large panes in the same heavy frame, up to the roof. At the far left and far right edges of the "
    "picture, which is directly behind the seat: the rear wall, solid from floor to roof and wider than the seat, "
    "plainly seen with what is fitted to it, and low in front of it the top of the seat's own small headrest. At the "
    "top the frame's ribs and the roof; at the bottom the seat's empty cushion and the floor. Nobody is in the "
    "cabin: every seat is empty. Through every pane of glass only a plain even darkness, nothing else outside.")


WORLD_WORDS = (
    # (the place's own look comes from its concept picture, given as the reference. No person is named anywhere in
    # these words: said as "the eyes of a person standing" with a part of the land "in front of that person", nine
    # of nine panoramas had a person in them, 11-16)
    # (what lies round the set and beyond it only: what stands in the place is built on the set's own ground as its
    # pieces; drawn into the panorama too, each stood there twice, and the land near the people was the panorama's
    # depth seen from the side, 11-18)
    "Equirectangular projection, a full 360 degree spherical panorama, 2:1, the left and right edges join seamlessly. "
    "The far surroundings of this place: {look}. Seen all the way round from eye height in the middle of a wide "
    "stretch of open, bare, nearly level ground: the horizon level straight across the middle of the picture, the "
    "whole sky above it, and below it only that bare ground, straight down and far out on every side. Nothing is "
    "built and nothing stands on this open ground or near it: only what lies far off is seen, small, along the "
    "horizon. Empty of life: no people, no figures, no animals anywhere.")


# (11-27: a place in the open with water or a town round it, scenes/concept.py "around": its world is drawn as that.
# WORLD_WORDS asks for bare level ground whatever the place is: a floor out on the water stood in sand to the horizon.
# A piece with no "around" is drawn from WORLD_WORDS, word for word as before: its world keeps its id)
WORLD_ROUND_WORDS = {
    "water": (
        "Equirectangular projection, a full 360 degree spherical panorama, 2:1, the left and right edges join "
        "seamlessly. The far surroundings of this place: {look}. Seen all the way round from eye height, low over "
        "the middle of a wide stretch of open water: the horizon level straight across the middle of the picture, the "
        "whole sky above it, and below it only that water, its surface straight down and far out on every side. "
        "Nothing is built and nothing floats on this open water or near it: only what lies far off is seen, small, "
        "along the horizon. Empty of life: no people, no figures, no animals anywhere."),
    "buildings": (
        "Equirectangular projection, a full 360 degree spherical panorama, 2:1, the left and right edges join "
        "seamlessly. The far surroundings of this place: {look}. Seen all the way round from eye height in the "
        "middle of a wide, open, bare, level square: the horizon level straight across the middle of the picture, the "
        "whole sky above it, and below it only that bare square, straight down and far out on every side. Nothing is "
        "built and nothing stands on this open square or near it: the buildings stand far off on every side, small, "
        "their walls and roofs one unbroken line along the horizon. Empty of life: no people, no figures, no animals "
        "anywhere."),
}


def world_words(piece: dict) -> str:
    """The words a place's panorama is drawn from: by what lies round it (the piece's "around"), else WORLD_WORDS."""
    return WORLD_ROUND_WORDS.get(str(piece.get("around") or ""), WORLD_WORDS)


# a piece that burns or glows is drawn lit, and the parts of its made shape that are painted as fire (bright, strongly
# coloured red to yellow) give light of their own (img3d/model_img3d.py's regions). A forge's hearth was a grey
# block with dark coals under a warm lamp, and a key drew its fire only when a shot's words named it (11-18)
LIT_WORDS = " It is lit: its fire is burning now, with glowing orange coals and low bright flames."
FIRE_REGION = {"name": "fire", "material": "emit", "hsv": [[0.0, 0.14], [0.5, 1], [0.55, 1]],
               "color": [1.0, 0.4, 0.1], "strength": 12}
FINE_FROM_M = 2.5                # a thing of the open this long or longer is shaped at the finer setting


def generation_request(piece, K=None, python="python", tools=None, work=None):
    """The dry-run request for a piece the library lacks: {kind, look, size_m, needs} and, under "plan", the jobs
    the existing prop route runs for it, in order. Nothing is started here."""
    K = K or kinds()
    kind, look = piece["kind"], piece.get("look", "").strip()
    d = K[kind]
    lit = piece.get("gives_light") is True           # (the director's: it burns or glows)
    aid = _asset_id(kind, look + (" " + Path(str(piece["source"])).stem if piece.get("source") else "")
                    + (" front " + str(d["front"]) + " wide" if d.get("front") else "") + (" lit" if lit else ""))
    tools = Path(tools or HERE.parent)
    work = Path(work or (C.UE_HOME / "lab" / "img3d_props"))
    name = kind.replace("_", " ")
    desc = (f"One single {name}, {look}. The whole object in view, standing on a plain floor, nothing else in the "
            f"picture.") if look else f"One single {name}. The whole object in view, nothing else in the picture."
    needs = [d["needs"]] if d.get("needs") else []
    # what the kind must afford is said in the drawing's words (a "bunk" came back as two tiers squeezed to a
    # bed's height, with nowhere a person could be seen lying)
    if any(n in NEED_WORDS for n in needs):                 # (first: the look's own words follow it)
        desc = desc.replace(f"One single {name}", f"One single {name} "
                            + " ".join(NEED_WORDS[n] for n in needs if n in NEED_WORDS), 1)
    size = list(d["size_m"])
    seen = piece.get("_seen_size_m")
    if seen and len(seen) == 3 and d.get("sized_by") != "longest":
        # as big as the place's concept picture has it (img3d/concept_depth.py), within reason of its kind's size
        # (the depth model's metres are near, not exact; a box that took in a neighbour is not a size)
        size = [round(min(SEEN_SIZE_K[1] * k, max(SEEN_SIZE_K[0] * k, float(v))), 3) for v, k in zip(seen, size)]
    if d.get("sized_by") == "longest":
        # a thing a person wears or carries: known only by its words and how long it is (the director's), so it is
        # drawn as those words say and made that long, whatever way up the shape comes
        # (as long as the place's concept picture shows it, when it does: the director's number is a guess)
        lng = float((max(seen) if seen and len(seen) == 3 else None) or piece.get("longest_m") or max(size))
        size = [lng, lng, lng]
        desc = f"One single object, alone: {look}. The whole object in view on a plain floor, nothing else in the picture."
    if lit:                                           # drawn burning: what of it is painted as fire is made to shine
        desc += LIT_WORDS
    spec = {"id": aid, "class": "prop", "description": f"{name}: {look}. Front -Y." if look else f"{name}. Front -Y.",
            "tags": sorted({kind, *([d["parent"]] if d.get("parent") else []), "generated-on-request"}),
            "tier": "mid", "size_m": size, "size_tolerance": 0.35, "placement": d["placement"],
            "source": "generated", "piece_kind": kind, "look": look}
    if d.get("sized_by") == "longest":
        # (shaped at the coarser setting: it is small in any frame, and the finer one took eight minutes a thing)
        # ... and it keeps the generator's own front (-Y): the rule that takes a piece's flat end for its back is
        # for furniture, and turned a helmet back to front on the head
        spec.update(longest_m=size[0], size_tolerance=1.0, shape_res=512, yaw_deg=0)
        if d.get("front") == "auto":
            # (a thing that stands on the ground with a front, a building: shaped at the finer setting, and its
            # front found as furniture's is. Kept as the generator turned it, a forge's front wall stood end on to
            # the yard, a stone pillar in every frame, 11-18)
            if lng >= FINE_FROM_M:           # (a small thing standing by it keeps the coarser, quicker setting)
                spec.pop("shape_res")
            spec.pop("yaw_deg")
            # (its front is the side the place's picture shows: as wide as seen there and less deep, which is what
            # the front rule turns the shape by. Given as a cube, the rule left the wall end on a second time)
            w_, h_ = (float(seen[0]), float(seen[2])) if seen and len(seen) == 3 else (lng, lng)
            spec["size_m"] = [round(w_, 3), round(0.6 * w_, 3), round(h_, 3)]
            spec["longest_m"] = round(max(w_, h_), 3)
    if lit:
        spec["regions"] = [dict(FIRE_REGION)]
    if piece.get("seats_abreast"):
        spec["seats_abreast"] = int(piece["seats_abreast"])
    line = [python, str(tools / "run_asset.py")]
    rl = piece.get("relief") if isinstance(piece.get("relief"), dict) else None
    if rl and rl.get("glb") and Path(str(rl["glb"])).exists():
        # the land's own form of a place in the open, already shaped from the depth of the place's picture where it
        # stands (img3d/concept_depth.py relief): nothing is drawn and no shape model is run, only the asset chain.
        # As tall as the set has it, its other sides as the shape's own
        g = [float(v) for v in rl["size_m"]]
        k = float(seen[2]) / g[2] if seen and len(seen) == 3 and g[2] > 0 else 1.0
        own = bool(rl.get("own_light"))          # (its picture's light is its own: another asset than the lit one)
        aid = _asset_id(kind, f"{look} relief {hashlib.sha1(Path(str(rl['glb'])).read_bytes()).hexdigest()[:10]} {k:.3f}"
                        + (" own light" if own else ""))
        spec.update(id=aid, made_as="relief", relief_glb=str(rl["glb"]), size_m=[round(v * k, 3) for v in g],
                    size_tolerance=0.05, tier="hero", **({"own_light": True} if own else {}))
        for key in ("longest_m", "shape_res", "yaw_deg"):
            spec.pop(key, None)
        steps = [
            {"step": "spec", "what": "the piece's source spec", "write": f"<library>/prop/{aid}/source/spec.json",
             "spec": spec},
            {"step": "shape", "what": "the form shaped from the picture's depth, through the asset chain",
             "station": C.IMAGE_STATION, "kind": "shape",
             "argv": line + ["--line-cmd", C.IMAGE_STATION, "shape", "--id", f"gen_shape_{aid}", "--log",
                             str(work / f"{aid}.log"), "--", python, str(tools / "img3d" / "relief_one.py"), aid],
             "result_id": f"{aid}_tr", "gate": "the asset's own checks pass", "on_fail": "none: the same picture "
             "gives the same form"}]
        return {"kind": kind, "look": look, "size_m": spec["size_m"], "needs": [], "asset_id": f"{aid}_tr",
                "dry_run": True, "plan": {"draw_id": aid, "max_redraws": 1, "steps": steps}}
    if d.get("made_as") == "world":
        # a place in the open is made from where its people stand: one panorama of it from a standing eye, drawn with
        # the place's concept picture as its reference, then a shell of the land with a sky dome and a flat ground
        # patch under the eye (img3d/pano_shell.py --outdoor). Its size is the shell's own.
        spec["made_as"] = "world"
        if piece.get("spot"):
            spec["spot"] = piece["spot"]
        if piece.get("marks"):               # the land's own things the plan sends people to: found in the panorama
            spec["marks"] = dict(piece["marks"])
        # the panorama is the place's own picture carried on all the way round (img3d/pano_from_picture.py), when
        # the place has one measured (its "view": the picture's eye and how wide it sees, sets/place_build.py);
        # else drawn from words
        view = piece.get("view") if isinstance(piece.get("view"), dict) and piece.get("source") \
            and Path(str(piece["source"])).exists() else None
        if view:
            pano = _asset_id(kind, f"{look} from the picture {hashlib.sha1(Path(str(piece['source'])).read_bytes()).hexdigest()[:10]}")
            spec["view"] = {k: view[k] for k in ("eye_h", "eye_at", "front_deg")}
        else:
            pano = _asset_id(kind, look + " " + hashlib.sha1(world_words(piece).encode()).hexdigest()[:8]
                             + (" " + piece["spot"] if piece.get("spot") else "")
                             + (" " + str(piece["source"]) if piece.get("source") else ""))
        aid = pano if not view else _asset_id(kind, f"{pano} view {json.dumps(spec['view'], sort_keys=True)}")
        if view:
            spec["panorama_id"] = pano
        if piece.get("ground") == "water" and not piece.get("map_words"):
            # the world's own flat ground is water (a made floor that stands in it, sets/place_build.py open_floor):
            # img3d/pano_shell.py --ground water gives every face of it the water's material
            aid = _asset_id(kind, f"{aid} ground water")
            spec.update(panorama_id=pano, ground="water")
        if piece.get("ground") == "none" and not piece.get("map_words"):
            # the world has no flat ground of its own (a made floor with its picture's land round it,
            # sets/place_build.py stage_pieces): img3d/pano_shell.py --ground none builds none
            aid = _asset_id(kind, f"{aid} ground none")
            spec.update(panorama_id=pano, ground="none")
        mdesc = ""
        if piece.get("map_words"):
            # the place's own ground seen from straight above, laid over the set's real ground (places/ground_map.py):
            # the world is another asset whenever the map's words change, and its panorama, whose words did not, is
            # the one already drawn (the panorama keeps its own id)
            import ground_map as GM
            mdesc = GM.map_desc(piece["map_words"], piece.get("land_places"), piece.get("map_style") or "")
            aid = _asset_id(kind, f"{aid} map {hashlib.sha1((mdesc + str(piece.get('map_source') or '')).encode()).hexdigest()[:8]}")
            spec.update(panorama_id=pano, map_id=f"{aid}_map")
        spec["id"] = aid
        steps = [
            {"step": "draw", "what": "the place's picture carried on all the way round, as its panorama",
             "station": C.IMAGE_STATION, "kind": "edit", "picture": "picture.png", "ref_id": pano,
             "argv": line + ["--line-cmd", C.IMAGE_STATION, "edit", "--id", f"gen_pano_{pano}", "--",
                             python, str(tools / "img3d" / "pano_from_picture.py"), str(piece["source"]), pano,
                             "--desc", look.rstrip(".") + ".", "--seed", "{seed}"],
             "gate": "nobody in the panorama", "on_fail": f"made again with the next seed, at most {MAX_REDRAWS}"}
            if view else
            {"step": "draw", "what": "one panorama of the place from a standing eye", "station": C.IMAGE_STATION,
             "kind": "refs", "picture": "picture.png", "ref_id": pano,
             "argv": line + ["--line-cmd", C.IMAGE_STATION, "refs", "--id", f"gen_draw_{pano}", "--",
                             python, str(tools / "refs" / "ref_draw.py"), "picture", pano, "--desc",
                             world_words(piece).format(look=look.rstrip("."))
                             + (f" A few steps ahead, close by, part of the ground itself: {piece['spot']}." if piece.get("spot")
                                else ""), "--size", "2048", "1024", "--seed", "{seed}"]
                     + (["--sources", str(piece["source"])] if piece.get("source") else []),
             "gate": "nobody in the panorama", "on_fail": f"re-draw with the next seed, at most {MAX_REDRAWS}"}]
        if mdesc:
            mid = spec["map_id"]
            steps += [
                {"step": "map", "what": "the place's ground seen from straight above, over the whole real ground",
                 "station": C.IMAGE_STATION, "kind": "refs", "picture": "picture.png", "ref_id": mid,
                 "argv": line + ["--line-cmd", C.IMAGE_STATION, "refs", "--id", f"gen_map_{mid}", "--",
                                 python, str(tools / "refs" / "ref_draw.py"), "picture", mid, "--desc", mdesc,
                                 "--size", str(GM.MAP_PX), str(GM.MAP_PX), "--seed", "{seed}"],
                 "gate": "none: without a map the ground is the panorama's own, as before", "on_fail": "go on without it"},
                {"step": "water_key", "what": "the edit model paints the map's water one flat key colour (the water is "
                 "the cells of that colour: the judge cannot outline it)", "station": "image", "kind": "edit",
                 "ref_id": mid, "picture": "water_key.png", "jobs": "water_key_jobs.json",
                 "gate": "none: without it the judge's outline is the water", "on_fail": "go on without it"},
                {"step": "map_read", "what": "the vision judge reads the map: whether it has water, and where each part of "
                 "the land the plan sends people to is", "station": "igpu", "kind": "choose", "ref_id": mid,
                 "facts": "map.json", "land_places": dict(piece.get("land_places") or {}),
                 "gate": "none: without it the ground has no water and the land places are the open ground",
                 "on_fail": "go on without it"}]
        steps += [
            {"step": "spec", "what": "the piece's source spec", "write": f"<library>/prop/{aid}/source/spec.json",
             "spec": spec},
            {"step": "shape", "what": "the panorama's depth as the land, a sky dome and the ground under the eye, then "
             "the asset chain", "station": C.IMAGE_STATION, "kind": "shape",
             "argv": line + ["--line-cmd", C.IMAGE_STATION, "shape", "--id", f"gen_shape_{aid}", "--log",
                             str(work / f"{aid}.log"), "--", python, str(tools / "img3d" / "shell_one.py"), aid],
             "result_id": f"{aid}_tr", "gate": "the asset's own checks pass", "on_fail": "re-draw with the next seed"}]
        return {"kind": kind, "look": look, "size_m": d["size_m"], "needs": needs, "asset_id": f"{aid}_tr",
                "dry_run": True, "plan": {"draw_id": aid, "max_redraws": MAX_REDRAWS, "steps": steps}}
    if d.get("made_as") == "shell":
        # a thing people sit inside is made from within: one panorama of its inside from its seat, then a shell of
        # what that shows (img3d/pano_shell.py). Its size and its seat are the shell's own, not the kind's.
        n = int(piece.get("seats_abreast") or 1)
        seat = "its single seat" if n < 2 else "the leftmost of its seats, which stand side by side"
        spec["made_as"] = "shell"
        aid = _asset_id(kind, look + " " + hashlib.sha1(SHELL_WORDS.encode()).hexdigest()[:8])   # (new words, new shell)
        spec["id"] = aid
        steps = [
            {"step": "draw", "what": "one panorama of the inside, from the seat", "station": C.IMAGE_STATION,
             "kind": "refs", "picture": "picture.png",
             "argv": line + ["--line-cmd", C.IMAGE_STATION, "refs", "--id", f"gen_draw_{aid}", "--",
                             python, str(tools / "refs" / "ref_draw.py"), "picture", aid, "--desc",
                             SHELL_WORDS.format(name=name, seat=seat, look=look.rstrip(".")),
                             "--size", "2048", "1024", "--seed", "{seed}"],
             "gate": "the panorama shows a window", "on_fail": f"re-draw with the next seed, at most {MAX_REDRAWS}"},
            {"step": "spec", "what": "the piece's source spec", "write": f"<library>/prop/{aid}/source/spec.json",
             "spec": spec},
            {"step": "shape", "what": "the panorama's depth as an inward shell with its panes open, then the asset "
             "chain", "station": C.IMAGE_STATION, "kind": "shape",
             "argv": line + ["--line-cmd", C.IMAGE_STATION, "shape", "--id", f"gen_shape_{aid}", "--log",
                             str(work / f"{aid}.log"), "--", python, str(tools / "img3d" / "shell_one.py"), aid],
             "result_id": f"{aid}_tr", "gate": "the asset's own checks pass",
             "on_fail": "re-draw with the next seed"},
            {"step": "measure_seat", "what": "the seat under the panorama's eye, written into the record",
             "gate": "a seat box is found", "on_fail": "re-draw with the next seed"}]
        return {"kind": kind, "look": look, "size_m": d["size_m"], "needs": needs, "asset_id": f"{aid}_tr",
                "dry_run": True, "plan": {"draw_id": aid, "max_redraws": MAX_REDRAWS, "steps": steps}}
    steps = [
        {"step": "draw", "what": "the reference set (a 2x2 turnaround sheet cut into four views) from the look line",
         "station": C.IMAGE_STATION, "kind": "refs",
         "argv": line + ["--line-cmd", C.IMAGE_STATION, "refs", "--id", f"gen_draw_{aid}", "--",
                         python, str(tools / "refs" / "ref_draw.py"), "draw", aid, "--desc", desc,
                         "--recipe", "sheet", "--seed", "{seed}"]
                 + (["--sources", str(piece["source"])] if piece.get("source") else []),   # (the place's concept)
         "gate": "the sheet cuts into four views and each view holds one whole object",
         "on_fail": f"re-draw with the next seed, at most {MAX_REDRAWS} drawings"},
        {"step": "spec", "what": "the piece's source spec, which the shape step reads (written when the drawing "
         "passes; not a library record yet)", "write": f"<library>/prop/{aid}/source/spec.json", "spec": spec},
        {"step": "shape", "what": "picture to 3D from the three-quarter view, then the asset chain (bake, engine "
         "import, validation stills)", "station": "engine", "kind": "shape",
         "argv": line + ["--line-cmd", "engine", "shape", "--id", f"gen_shape_{aid}", "--log",
                         str(work / f"{aid}.log"), "--", python, str(tools / "img3d" / "prop_one.py"), aid],
         "result_id": f"{aid}_tr",
         "gate": "the asset's own checks pass (checks_passed) and its size is within size_tolerance of size_m",
         "on_fail": "re-draw with the next seed (a new drawing gives a new shape); the same drawing is not "
                    "shaped twice"},
        {"step": "front", "what": "the generator puts the object's front toward -Y in the asset frame; yaw_deg "
         "'auto' in the spec; a piece that comes out turned is caught by the support measure below",
         "gate": "the measured front is -Y (library contract)", "on_fail": "the record carries the measured "
         "front and the set builder turns the mesh"},
    ]
    for n in needs:
        steps.append({"step": f"measure_{n}", "what": f"the {n} box measured from the exported mesh and written "
                      f"into the record's \"{n}\" block", "gate": f"a {n} box is found",
                      "on_fail": "re-draw with the next seed"})
    return {"kind": kind, "look": look, "size_m": d["size_m"], "needs": needs, "asset_id": f"{aid}_tr",
            "dry_run": True, "plan": {"draw_id": aid, "max_redraws": MAX_REDRAWS, "steps": steps}}


def submit(request, really=False, seed=11, run=None):
    """Hand a request's first line job (the drawing) to the line. Refuses unless really=True; the later steps
    are submitted by whoever watches the gates. `run` is the callable that starts a process (tests pass a fake)."""
    if not really:
        raise RuntimeError("generation is not started without really=True (this request is a dry run)")
    import subprocess
    step = next(s for s in request["plan"]["steps"] if s["step"] == "draw")
    argv = [a.replace("{seed}", str(seed)) for a in step["argv"]]
    return (run or subprocess.run)(argv)


def resolve(pieces, records, K=None, threshold=DEFAULT_THRESHOLD, allow_scripted=False):
    """Every piece of a spec -> {"pieces": [{id, kind, look, status, ...}], "summary": {found, generate,
    unresolved}}. status: "library" (asset_id, score, why, others), "generate" (request), "unresolved" (reason)."""
    K = K or kinds()
    out = []
    for p in pieces:
        # a look the spec's writer left open ("not stated", whole or as one clause of the look) says nothing: the
        # words are not a look to match or to draw (a bedroom found none of the library's beds for "not stated")
        p = dict(p, look=", ".join(c.strip() for c in (p.get("look") or "").split(",")
                                   if c.strip() and "not stated" not in c.lower()))
        row = {"id": p.get("id"), "kind": p.get("kind"), "look": p.get("look", "")}
        if p.get("kind") not in K:
            row.update(status="unresolved", reason=f"kind {p.get('kind')!r} is not in PIECE_KINDS")
        else:
            hits = lookup(p, records, K, threshold, allow_scripted)
            if K[p["kind"]].get("made_as") == "shell":
                # a cabin shell is drawn from the look's words, not from the place's concept picture: a shell made
                # for the same kind and look serves, so a re-planned place keeps the cabin it was staged in (a
                # director's re-wording of the same cockpit drew a new cabin each time, 11-15). Only shells, and
                # none that shows someone in its panorama.
                hits = [h for h in hits if "panorama_shell" in (next((r for r in records
                        if r["asset_id"] == h["asset_id"]), {}).get("tags") or [])
                        and not next((r for r in records if r["asset_id"] == h["asset_id"]), {}).get("panorama_people")]
            elif p.get("source") or p.get("_dressing") or K[p["kind"]].get("sized_by") == "longest" or \
                    K[p["kind"]].get("made_as") == "world" or p.get("kind") == "bystander":
                # (and one of a place's crowd, who is whoever the words say: a ball's guests were found among the
                # library's bystanders as a servant, and fifteen of that one servant stood in the room, 11-26)
                # (and a thing the place is dressed with, made from its own words: a room of one age was given a
                # screen made for another story's, a pane of glass on steel feet, four times, 11-20)
                # a piece with its own picture (its part of the place's concept) is that thing, made from it: only
                # the asset already made from that same picture serves, never a near match by words. So is a thing
                # worn or carried: a kind that holds any object, where a near match by words is another object
                want = generation_request(p, K)["asset_id"]
                hits = [h for h in hits if h["asset_id"] == want] or \
                    [{"asset_id": want, "score": 1.0, "why": "made from this piece's own picture", "others": []}
                     for r in records if r["asset_id"] == want
                     and not (K[p["kind"]].get("made_as") == "shell"
                              and shell_too_big(r.get("size_m"), K[p["kind"]].get("size_m")))
                     and not (K[p["kind"]].get("needs") == "seat" and seat_too_narrow(r))][:1]
            if hits:
                row.update(status="library", asset_id=hits[0]["asset_id"], score=hits[0]["score"],
                           why=hits[0]["why"], others=[{"asset_id": h["asset_id"], "score": h["score"]}
                                                       for h in hits[1:4]])
            else:
                row.update(status="generate", request=generation_request(p, K))
                near = lookup(p, records, K, 0.0, allow_scripted)     # the nearest of its kind, under the bar
                if near:
                    row["nearest"] = {"asset_id": near[0]["asset_id"], "score": near[0]["score"]}
        out.append(row)
    n = lambda s: sum(1 for r in out if r["status"] == s)  # noqa: E731
    return {"pieces": out, "summary": {"found": n("library"), "generate": n("generate"),
                                       "unresolved": n("unresolved")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pieces", help="a JSON file: a list of pieces, or a place spec with a \"pieces\" list")
    ap.add_argument("--library")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--allow-scripted", action="store_true")
    a = ap.parse_args()
    d = json.loads(Path(a.pieces).read_text())
    res = resolve(d["pieces"] if isinstance(d, dict) else d, load_records(a.library), threshold=a.threshold,
                  allow_scripted=a.allow_scripted)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())


def period_records(spec: dict, recs: list) -> list:
    """The library's records that may furnish this place: a piece made for another period does not (a ship's cabin
    was offered mahogany); a record whose tags or description name a period other than the place's is left out, one
    that names none serves any. The set builder and the generation both look pieces up in these: looked up in all of
    them, a ballroom's bystander was found (another period's) and so never made, and the builder, which left that one
    out, built the room with no bystander at all (11-26)."""
    import json as _json
    import re as _re
    periods = set(_json.loads((Path(__file__).resolve().parent.parent / "scenes" / "place_kinds.json").read_text())["PERIODS"])
    own = spec.get("period")
    if own not in periods:
        return recs

    def named(r):
        words = {str(t).lower() for t in r.get("tags") or []} | set(
            _re.findall(r"[a-z_]+", (r.get("description") or "").lower().replace("far future", "far_future")))
        return periods & words
    return [r for r in recs if not named(r) or own in named(r)]
