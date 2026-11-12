"""A scene's keys judged TOGETHER: each person and each large piece must look the same from key to key.

Every per-key check compares a redraw with its own engine still, so a set of keys can each pass while a person is
fair-haired in one and dark in the next, grows a moustache when small, or a close on a seat shows another chair.
This pass runs after the keys are drawn and picked (accept.py pick), over the whole scene (stdlib only):

    python tools/assets/redraw/scenelook.py cfg    RUN.json FEATS_CFG.json      (what lookfeat.py measures at a card)
    python tools/assets/redraw/scenelook.py judge  RUN.json FEATS.json LOOK.json  (medians, outliers, anchors, cuts)
    python tools/assets/redraw/scenelook.py retry  RUN.json LOOK.json RETRY_JOBS.jsonl
    python tools/assets/redraw/scenelook.py report LOOK.json
    python tools/assets/redraw/scenelook.py compare LOOK.json REVIEW.csv [IDMAP.json]  (which review rows it reproduces)
    python tools/assets/redraw/scenelook.py run    RUN.json [--py PYTHON] [--review REVIEW.csv [--map IDMAP.json]]
                                              (the one command: cfg, lookfeat.py through the line, judge, report,
                                               anchor crops, re-draw jobs written but not submitted)

Per person per key (lookfeat.py, from the person's engine box and head points; the Lab readings come from the key
checks already made): a head embedding, hair and clothes colour against the engine still in the same region (the
shot's light taken out), facial hair as a closed answer on the head crop, and each staged worn item as a closed
answer. Here: per person the median of every attribute over the scene's keys, weighted by head size (large clear
faces define the look); each key's distance from it; outliers named by attribute. An outlier key gets one re-draw
with the person's reference crop from their ANCHOR key (the largest face that passes everything) and their look
words. A small figure (head under FACE_BAR of the frame's height) is judged only on what still reads at that size:
clothes colour, hair, no invented facial hair; never on face identity. Large pieces: each key's crop of a piece
against every piece's engine crops (a close on a seat must be nearest its own seat) and its own median.
The report counts LOOK changes per cut (a person's attribute differing between consecutive keys with no staged
reason), to stand beside the position changes per cut.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- sizes (head top to chin, as a share of the frame's height) -----------------------------------------------
FACE_BAR = 0.12        # a head this tall (77 px of 640) carries a face identity: below it the embedding is noise
SLIVER_BAR = 0.04      # under this (26 px) only clothes colour is judged
REF_HEAD = 0.30        # a head this tall has full weight in the person's median; smaller ones fall with the square
# --- colour (CIE Lab, the redraw minus the engine still in the same region) ------------------------------------
HAIR_L, HAIR_AB = 12.0, 9.0     # lightness as accept.hair_eval; a-b distance: brown to grey moves ~10
HAIR_PX_FACE, HAIR_PX_SMALL = 3000, 400   # the hair region's size: accept.py's bar, and a looser one for the
#                                           scene pass on small figures (hair mass only: lightness, 1.5 x the bar)
WEAR_L, WEAR_AB = 12.0, 9.0     # lightness as accept.wear_eval; a-b distance: a tweed-brown coat to dark grey ~10
WEAR_PX = 400
# --- face identity (cosine distance between head embeddings) ----------------------------------------------------
FACE_K = 3.0           # an outlier is beyond the person's median distance + FACE_K robust deviations ...
FACE_FLOOR = 0.12      # ... and beyond this floor (two crops of one drawn face differ by pose and light)
FACE_PAIR = 0.30       # between two consecutive keys of one person: beyond this the face changed across the cut
# --- large pieces ---------------------------------------------------------------------------------------------
THING_FILL = 0.45      # as accept.LOOK_FILL: the piece's own pixels fill this much of its crop
THING_AREA = 0.03      # judged when its mask is at least this share of the frame (a close or medium on it)
THING_MARGIN = 0.03    # nearer another piece's engine look than its own by more than this: drawn as that piece
AGE_TOL, AGE_TOL_SMALL = 12.0, 20.0   # apparent age (years) off the person's median: at face size, on a small head
DUP_CORR = 0.97        # two consecutive keys whose small grey pictures correlate above this open on the same picture
NO_ANSWER = ("cannot tell", "", None)
# hair colour words (the closed answer set of lookfeat's question) -> (family, darkness rank 0 dark .. 5 light)
HAIR_WORDS = {"black": ("dark", 0), "dark brown": ("dark", 1), "brown": ("dark", 2), "light brown": ("dark", 3),
              "sandy": ("dark", 3), "blond": ("dark", 4), "auburn": ("red", 2), "red": ("red", 3),
              "black with grey": ("grey", 1), "grey": ("grey", 4), "silver": ("grey", 4), "white": ("grey", 5)}


# --- small numeric helpers --------------------------------------------------------------------------------------
def wmedian(vals: list, weights: list):
    """Weighted median (None when nothing has weight)."""
    pairs = sorted((v, w) for v, w in zip(vals, weights) if w > 0)
    tot = sum(w for _, w in pairs)
    if not pairs or tot <= 0:
        return None
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= tot / 2:
            return v
    return pairs[-1][0]


def unit(v: list) -> list:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def cos_dist(a: list, b: list) -> float:
    return 1.0 - sum(x * y for x, y in zip(unit(a), unit(b)))


def centroid(vecs: list, weights: list):
    vs = [(unit(v), w) for v, w in zip(vecs, weights) if w > 0 and v]
    if not vs:
        return None
    return unit([sum(v[i] * w for v, w in vs) for i in range(len(vs[0][0]))])


def head_weight(h: float | None) -> float:
    return 0.0 if not h else min(1.0, h / REF_HEAD) ** 2


def size_class(h: float | None) -> str:
    if h is None or h < SLIVER_BAR:
        return "sliver"
    return "face" if h >= FACE_BAR else "small"


def hair_class(words: str | None) -> str | None:
    """A facial-hair answer or an entry's fact as one of none / moustache / beard / both (None: no answer)."""
    if words in NO_ANSWER:
        return None
    w = words.lower()
    m = "moustache" in w or "mustache" in w
    b = "beard" in w or "stubble" in w or "whiskers" in w
    if m and b:
        return "both"
    if m:
        return "moustache"
    if b:
        return "beard"
    if "none" in w or "clean" in w or w.strip() in ("no", "no."):
        return "none"
    return None


def hair_word(a: str | None):
    """A hair-colour answer as (family, rank), the longest listed word it contains (None: no answer)."""
    if a in NO_ANSWER:
        return None
    w = a.strip().lower().replace("gray", "grey").replace("blonde", "blond")
    for k in sorted(HAIR_WORDS, key=len, reverse=True):
        if k in w:
            return HAIR_WORDS[k]
    return None


def hair_differs(x, y) -> bool:
    """Two hair words differ when the family differs (grey v not, red v not) or the darkness is two steps apart."""
    return x is not None and y is not None and (x[0] != y[0] or abs(x[1] - y[1]) >= 2)


def age_of(a) -> float | None:
    if a in NO_ANSWER:
        return None
    d = "".join(ch if ch.isdigit() else " " for ch in str(a)).split()
    return float(d[0]) if d and 3 <= float(d[0]) <= 100 else None


def corr(a: list, b: list) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    return None if va <= 0 or vb <= 0 else sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def yes_no(a: str | None) -> bool | None:
    if a in NO_ANSWER:
        return None
    a = a.strip().lower()
    return True if a.startswith("yes") else False if a.startswith("no") else None


# --- rows -------------------------------------------------------------------------------------------------------
def person_rows(items: list, checked: dict, feats: dict, fails: dict | None = None,
                facts: dict | None = None) -> list:
    """One row per person per key, in the scene's order. items: the key-check items (id, people with box, joints2d,
    head_to_camera_deg, face_covered, worn); checked: the key checks' results for the picked keys (hair, wear);
    feats: lookfeat.py's output; fails: {id: {person: [failed checks]}} from the pick; facts: {person: entry facts}."""
    rows = []
    for n, it in enumerate(items):
        rec = checked.get(it["id"]) or {}
        fx = (feats.get(it["id"]) or {}).get("people") or {}
        for p in it.get("people") or []:
            k = p["key"]
            j2 = p.get("joints2d") or {}
            h = (j2["chin"][1] - j2["head_top"][1]) if "chin" in j2 and "head_top" in j2 else None
            f = fx.get(k) or {}
            seen = (p.get("head_to_camera_deg") or 0) < 100 and not p.get("face_covered")
            row = {"n": n, "id": it["id"], "key": k, "head": h, "cls": size_class(h), "face_seen": seen,
                   "emb": f.get("emb"), "facial_hair": f.get("facial_hair"), "worn": f.get("worn") or {},
                   "hair_word": f.get("hair_colour"), "age": age_of(f.get("age")),
                   "staged_worn": list(p.get("worn") or []), "other_fails": list((fails or {}).get(it["id"], {}).get(k, [])),
                   "fact_facial_hair": (facts or {}).get(k, {}).get("facial_hair")}
            for name, bar in (("hair", HAIR_PX_SMALL), ("wear", WEAR_PX)):
                v = (rec.get(name) or {}).get(k) or {}
                r, e = v.get("redraw"), v.get("engine")
                if r and e and r.get("px", 0) >= bar:
                    row[name] = {"dL": r["L"] - e["L"], "da": r["a"] - e["a"], "db": r["b"] - e["b"], "px": r["px"],
                                 "L": r["L"], "a": r["a"], "b": r["b"]}
            rows.append(row)
    return rows


# --- the scene pass for people ----------------------------------------------------------------------------------
def judge_people(rows: list) -> dict:
    """Per person: the scene's look (weighted medians, the face centroid), then each row's flags (in place:
    row["flags"], row["dev"]). -> {person: {"median": {...}, "anchor": id or None}}."""
    out = {}
    keys = sorted({r["key"] for r in rows})
    cents = {}
    for k in keys:                                    # every person's face centroid, for the cross-person test
        fr = [r for r in rows if r["key"] == k and r["cls"] == "face" and r["face_seen"] and r.get("emb")]
        cents[k] = centroid([r["emb"] for r in fr], [head_weight(r["head"]) for r in fr])
    for k in keys:
        mine = [r for r in rows if r["key"] == k]
        med = {}
        for name in ("hair", "wear"):
            rs = [r for r in mine if r.get(name)]
            # hair: big heads define it; clothes: the more of the garment in frame the better it is read
            ws = [head_weight(r["head"]) if name == "hair" else min(1.0, r[name]["px"] / 20000.0) for r in rs]
            if name == "hair" and not any(w > 0 for w in ws):
                ws = [1.0] * len(rs)
            for a in ("dL", "da", "db"):
                med[f"{name}_{a}"] = wmedian([r[name][a] for r in rs], ws)
        fr = [r for r in mine if r["cls"] == "face" and r["face_seen"] and r.get("emb")]
        for r in fr:                                  # leave-one-out: a key does not vote for itself
            others = [q for q in fr if q is not r]
            c = centroid([q["emb"] for q in others], [head_weight(q["head"]) for q in others])
            r["dev_face"] = cos_dist(r["emb"], c) if c else None
        ds = [r["dev_face"] for r in fr if r.get("dev_face") is not None]
        fmed = statistics.median(ds) if ds else None
        fmad = statistics.median(abs(d - fmed) for d in ds) if ds else None
        med["face_dist"], med["face_spread"] = fmed, fmad
        fact = next((hair_class(r["fact_facial_hair"]) for r in mine if r.get("fact_facial_hair")), None)
        if fact is None:                              # no fact in the entry: the scene's own majority at face size
            ans = [hair_class(r["facial_hair"]) for r in mine if r["cls"] == "face" and hair_class(r["facial_hair"])]
            fact = statistics.mode(ans) if ans else None
        med["facial_hair"] = fact
        hw = [(hair_word(r["hair_word"]), head_weight(r["head"]) or 0.05) for r in mine
              if r["cls"] != "sliver" and hair_word(r["hair_word"])]
        tally = {}
        for w_, wt in hw:
            tally[w_] = tally.get(w_, 0.0) + wt
        med["hair_word"] = max(tally, key=tally.get) if tally else None
        ag = [r for r in mine if r["age"] is not None and r["face_seen"] and r["cls"] != "sliver"]
        med["age"] = wmedian([r["age"] for r in ag], [head_weight(r["head"]) or 0.05 for r in ag])
        wr = [r for r in mine if r.get("wear")]
        for a in ("L", "a", "b"):
            med[f"wear_abs_{a}"] = wmedian([r["wear"][a] for r in wr], [min(1.0, r["wear"]["px"] / 20000.0) for r in wr])
        for r in mine:
            fl, dev = [], {}
            h = r.get("hair")
            if h and med["hair_dL"] is not None and r["cls"] != "sliver":
                big = r["cls"] == "face" and h["px"] >= HAIR_PX_FACE
                dev["hair_dL"] = h["dL"] - med["hair_dL"]
                dev["hair_ab"] = math.hypot(h["da"] - med["hair_da"], h["db"] - med["hair_db"])
                if abs(dev["hair_dL"]) > (HAIR_L if big else 1.5 * HAIR_L) or (big and dev["hair_ab"] > HAIR_AB):
                    fl.append("hair")
            w = r.get("wear")
            if w and med["wear_dL"] is not None:
                dev["wear_dL"] = w["dL"] - med["wear_dL"]
                dev["wear_ab"] = math.hypot(w["da"] - med["wear_da"], w["db"] - med["wear_db"])
                if abs(dev["wear_dL"]) > WEAR_L or dev["wear_ab"] > WEAR_AB:
                    fl.append("clothes colour")
            hwd = hair_word(r.get("hair_word"))           # the coarse reading: works on a small head too
            if r["cls"] != "sliver" and hair_differs(hwd, med["hair_word"]) and "hair" not in fl:
                dev["hair_word"] = r["hair_word"]
                fl.append("hair")
            if r["age"] is not None and med["age"] is not None and r["face_seen"] and r["cls"] != "sliver":
                dev["age"] = r["age"] - med["age"]
                if abs(dev["age"]) > (AGE_TOL if r["cls"] == "face" else AGE_TOL_SMALL):
                    fl.append("age")
            fh = hair_class(r.get("facial_hair"))
            if fh is not None and fact is not None and fh != fact and r["cls"] != "sliver" and r["face_seen"]:
                dev["facial_hair"] = fh             # at any head size above a sliver: an invented moustache is seen
                fl.append("facial hair")
            for item in r["staged_worn"]:
                if yes_no(r["worn"].get(item)) is False:
                    fl.append("worn")
                    dev.setdefault("worn_missing", []).append(item)
            d = r.get("dev_face")
            if d is not None:
                dev["face"] = d
                lim = max(FACE_FLOOR, (fmed or 0.0) + FACE_K * (fmad or 0.0))
                other = min(((cos_dist(r["emb"], c), q) for q, c in cents.items() if q != k and c), default=None)
                own = cos_dist(r["emb"], cents[k]) if cents[k] else None
                if (len(ds) >= 3 and d > lim) or (other and own is not None and other[0] < own):
                    fl.append("face")               # far from their own look, or nearer another person's
                    if other and own is not None and other[0] < own:
                        dev["face_nearer"] = other[1]
            r["flags"], r["dev"] = fl, dev
        out[k] = {"median": med}
        clean = [r for r in mine if r["cls"] == "face" and r["face_seen"] and not r["flags"] and not r["other_fails"]]
        best = max(clean, key=lambda r: r["head"], default=None)
        out[k].update(anchor=best["id"] if best else None, anchor_head=best["head"] if best else None)
    for r in rows:                                    # the person drawn inside a mask wears another person's clothes
        w, me = r.get("wear"), out[r["key"]]["median"]
        if not w or me.get("wear_abs_L") is None:
            continue
        d = lambda m: math.sqrt((w["L"] - m["wear_abs_L"]) ** 2 + (w["a"] - m["wear_abs_a"]) ** 2 + (w["b"] - m["wear_abs_b"]) ** 2)
        own = d(me)
        oth = min(((d(v["median"]), q) for q, v in out.items() if q != r["key"] and v["median"].get("wear_abs_L") is not None),
                  default=None)
        if oth and own > WEAR_L and oth[0] < 0.5 * own:
            r["dev"]["wear_reads_as"] = oth[1]
            if "clothes colour" not in r["flags"]:
                r["flags"].append("clothes colour")
    for k in keys:                                    # anchors chosen after every flag is known
        mine = [r for r in rows if r["key"] == k]
        clean = [r for r in mine if r["cls"] == "face" and r["face_seen"] and not r["flags"] and not r["other_fails"]]
        best = max(clean, key=lambda r: r["head"], default=None)
        out[k].update(anchor=best["id"] if best else None, anchor_head=best["head"] if best else None)
    return out


# --- large pieces across keys -----------------------------------------------------------------------------------
def judge_things(items: list, feats: dict) -> list:
    """Each large piece per key: is the redraw's crop nearest its OWN piece's engine look (the mean of that piece's
    engine crops over the scene) among the room's pieces. -> rows with "flag" (None or the piece it was drawn as)."""
    eng = {}
    for it in items:
        for n, v in ((feats.get(it["id"]) or {}).get("things") or {}).items():
            if v.get("emb_engine") and (v.get("fill") or 0) >= THING_FILL:
                eng.setdefault(n, []).append(v["emb_engine"])
    cents = {n: centroid(vs, [1.0] * len(vs)) for n, vs in eng.items()}
    rows = []
    for i, it in enumerate(items):
        for n, v in ((feats.get(it["id"]) or {}).get("things") or {}).items():
            if not v.get("emb_redraw") or n not in cents or (v.get("fill") or 0) < THING_FILL \
                    or (v.get("area") or 0) < THING_AREA:
                continue
            own = cos_dist(v["emb_redraw"], cents[n])
            other = min(((cos_dist(v["emb_redraw"], c), q) for q, c in cents.items() if q != n and c), default=None)
            flag = other[1] if other and other[0] + THING_MARGIN < own else None
            rows.append({"n": i, "id": it["id"], "thing": n, "own": round(own, 4),
                         "other": round(other[0], 4) if other else None, "flag": flag})
    return rows


# --- look changes per cut ---------------------------------------------------------------------------------------
def look_changes(rows: list) -> list:
    """For each cut (consecutive keys), each person in both: which attributes differ between the two keys with no
    staged reason. -> [{"cut": [id, id], "changes": [[person, attribute], ...]}]"""
    by = {}
    for r in rows:
        by.setdefault(r["n"], {})[r["key"]] = r
    out = []
    ns = sorted(by)
    for a, b in zip(ns, ns[1:]):
        ch = []
        for k in sorted(set(by[a]) & set(by[b])):
            p, q = by[a][k], by[b][k]
            if p.get("hair") and q.get("hair") and "sliver" not in (p["cls"], q["cls"]):
                big = p["cls"] == q["cls"] == "face"
                if abs(p["hair"]["dL"] - q["hair"]["dL"]) > (HAIR_L if big else 1.5 * HAIR_L):
                    ch.append([k, "hair"])
            if p.get("wear") and q.get("wear"):
                if abs(p["wear"]["dL"] - q["wear"]["dL"]) > WEAR_L or math.hypot(
                        p["wear"]["da"] - q["wear"]["da"], p["wear"]["db"] - q["wear"]["db"]) > WEAR_AB:
                    ch.append([k, "clothes colour"])
            if [k, "hair"] not in ch and "sliver" not in (p["cls"], q["cls"]) and hair_differs(
                    hair_word(p.get("hair_word")), hair_word(q.get("hair_word"))):
                ch.append([k, "hair"])
            if (p["age"] is not None and q["age"] is not None and p["face_seen"] and q["face_seen"]
                    and "sliver" not in (p["cls"], q["cls"]) and abs(p["age"] - q["age"]) > AGE_TOL_SMALL):
                ch.append([k, "age"])
            fa, fb = hair_class(p.get("facial_hair")), hair_class(q.get("facial_hair"))
            if fa and fb and fa != fb and p["face_seen"] and q["face_seen"]:
                ch.append([k, "facial hair"])
            if (p["cls"] == q["cls"] == "face" and p["face_seen"] and q["face_seen"] and p.get("emb") and q.get("emb")
                    and cos_dist(p["emb"], q["emb"]) > FACE_PAIR):
                ch.append([k, "face"])
            for item in set(p["staged_worn"]) & set(q["staged_worn"]):   # staged worn in both: seen in one only
                ya, yb = yes_no(p["worn"].get(item)), yes_no(q["worn"].get(item))
                if ya is not None and yb is not None and ya != yb:
                    ch.append([k, "worn"])
        out.append({"cut": [rows_id(by[a]), rows_id(by[b])], "changes": ch})
    return out


def rows_id(d: dict) -> str:
    return next(iter(d.values()))["id"]


# --- the pass ---------------------------------------------------------------------------------------------------
def judge(items: list, checked: dict, feats: dict, fails: dict | None = None, facts: dict | None = None) -> dict:
    rows = person_rows(items, checked, feats, fails, facts)
    people = judge_people(rows)
    things = judge_things(items, feats)
    cuts = look_changes(rows)
    flagged = {}
    for r in rows:
        if r["flags"]:
            flagged.setdefault(r["id"], {})[r["key"]] = r["flags"]
    for t in things:
        if t["flag"]:
            flagged.setdefault(t["id"], {}).setdefault("set", []).append(f"look {t['thing']} (drawn as {t['flag']})")
    repeats = []                                      # consecutive keys that open on the same picture
    for a, b in zip(items, items[1:]):
        c = corr((feats.get(a["id"]) or {}).get("thumb"), (feats.get(b["id"]) or {}).get("thumb"))
        if c is not None and c > DUP_CORR and not b.get("hold"):
            repeats.append({"cut": [a["id"], b["id"]], "corr": round(c, 4)})
            flagged.setdefault(b["id"], {}).setdefault("set", []).append("repeat of the previous key")
    table = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k != "emb"} for r in rows]
    return {"people": people, "rows": table, "things": things, "cuts": cuts, "flagged": flagged, "repeats": repeats,
            "look_changes": sum(len(c["changes"]) for c in cuts), "cuts_n": len(cuts),
            "keys_flagged": len(flagged), "keys": len(items)}


# --- reading a run ------------------------------------------------------------------------------------------------
def load_run(run: dict):
    """A drawn and picked run -> (items in scene order with the picked picture, checked results for the picked ids,
    fails per id). Reads the run folder's check.json / check_result.json, their re-draw pair and accepted.json."""
    out = Path(run["out"])
    cfg = json.loads((out / "check.json").read_text())
    res = json.loads((out / "check_result.json").read_text())
    xres = json.loads((out / "check_x_result.json").read_text()) if (out / "check_x_result.json").exists() else {}
    acc = json.loads((out / "accepted.json").read_text()) if (out / "accepted.json").exists() else {}
    order = list(run.get("order") or [])
    items = [it for it in cfg["items"] if not it["id"].endswith("__x")]
    if order:
        items.sort(key=lambda it: order.index(it["id"]) if it["id"] in order else len(order))
    else:                                             # scene folders are in story order, cameras by name within
        sc = [Path(s).name for s in run.get("scenes") or []]
        meta = json.loads((out / "meta.json").read_text())
        items.sort(key=lambda it: (sc.index(meta[it["id"]]["scene"]) if meta[it["id"]]["scene"] in sc else 0,
                                   meta[it["id"]]["cam"]))
    checked, fails = {}, {}
    for it in items:
        a = acc.get(it["id"]) or {}
        pick = a.get("id", it["id"])
        checked[it["id"]] = (xres if pick.endswith("__x") else res).get(pick) or res.get(it["id"]) or {}
        if a.get("path"):
            it["redraw"] = a["path"]
        fails[it["id"]] = {k: v for k, v in (a.get("fails") or {}).items() if k != "set"}
    return items, checked, fails


def run_facts(run: dict, items: list) -> dict:
    """{person: their library entry's look facts} through the staged figure ids of the run's scenes."""
    sys.path.insert(0, str(HERE))
    import redraw as RD
    out = {}
    for s in run.get("scenes") or []:
        for k, v in RD.scene_wear(Path(s)).items():
            if k not in out and v.get("id"):
                out[k] = RD.entry_facts(v["id"])
    return out


def staged_worn(run: dict, items: list) -> None:
    """Each item's people get "worn" (the scene's added wardrobe words for them) and "face_covered"."""
    sys.path.insert(0, str(HERE))
    import redraw as RD
    meta = json.loads((Path(run["out"]) / "meta.json").read_text())
    sw = {Path(s).name: RD.scene_wear(Path(s)) for s in run.get("scenes") or []}
    for it in items:
        w = sw.get(meta[it["id"]]["scene"]) or {}
        for p in it["people"]:
            v = w.get(p["key"]) or {}
            p["worn"] = [v["wear_add"]] if v.get("wear_add") else []
            p["face_covered"] = bool(v.get("face_covered"))


def cmd_cfg(run: dict, out_p: Path) -> None:
    items, _, _ = load_run(run)
    staged_worn(run, items)
    out_p.write_text(json.dumps({"out": str(out_p.with_suffix("")) + "_result.json",
                                 "embedder": run.get("look_embedder", "clip"), "items": items}, indent=1))
    print(out_p, len(items))


def cmd_judge(run: dict, feats: dict, out_p: Path) -> dict:
    items, checked, fails = load_run(run)
    staged_worn(run, items)
    r = judge(items, checked, feats, fails, run_facts(run, items))
    out_p.write_text(json.dumps(r, indent=1))
    print(out_p, f"{r['keys_flagged']} of {r['keys']} keys flagged; {r['look_changes']} look changes over "
                 f"{r['cuts_n']} cuts")
    return r


FIX = {"hair": "{Who}'s hair is exactly as in the reference picture of {obj}: the same colour and the same style.",
       "clothes colour": "{Who}'s clothes keep exactly the colours of the reference picture of {obj}.",
       "facial hair": "{Who}'s face is {fh}, exactly as in the reference picture of {obj}.",
       "face": "{Who} is the same person as in the reference picture of {obj}: the same face and age.",
       "age": "{Who} is the same age as in the reference picture of {obj}, neither younger nor older.",
       "worn": "{Who} wears {item}."}


def fix_words(who: str, pos: str, flags: list, dev: dict, fact_fh: str | None) -> list:
    """The sentences a re-draw adds for a person's flags (general words; the person's own facts as data)."""
    Who, obj = who[0].upper() + who[1:], ("her" if pos == "her" else "him")
    fh = {"none": "clean-shaven, with no moustache and no beard", "moustache": "moustached, with no beard",
          "beard": "bearded", "both": "bearded and moustached"}.get(fact_fh or "", "as on the character sheet")
    out = []
    for f in flags:
        if f == "worn":
            out += [FIX["worn"].format(Who=Who, item=i) for i in dev.get("worn_missing") or []]
        else:
            out.append(FIX[f].format(Who=Who, obj=obj, fh=fh))
    return out


def cmd_retry(run: dict, look: dict, out_jobs: Path) -> None:
    """One re-draw per flagged key: the same job, another seed, each flagged person's reference crop from their
    anchor key (cut by lookfeat's crop rule: facecrop.py's square) and their look words; as "<id>__y"."""
    sys.path.insert(0, str(HERE))
    import redraw as RD
    out = Path(run["out"])
    jobs = {json.loads(ln)["id"]: json.loads(ln) for ln in (out / "image_jobs.jsonl").read_text().splitlines() if ln.strip()}
    cast = run.get("cast") or {}
    rows = {(r["id"], r["key"]): r for r in look["rows"]}
    lines = []
    for iid, fl in look["flagged"].items():
        if iid not in jobs:
            continue
        j = dict(jobs[iid])
        add, refs = [], list(j.get("refs") or [])
        for k, flags in fl.items():
            if k == "set":
                continue
            w = RD.who_of(cast, k)
            pos = "her" if w.get("noun") == "woman" else "his"
            who = f"the {w.get('noun') or 'person'}"
            r = rows.get((iid, k)) or {}
            add += fix_words(who, pos, flags, r.get("dev") or {}, look["people"][k]["median"].get("facial_hair"))
            ref = out / "look_anchor" / f"{k}.png"
            if look["people"][k].get("anchor") and str(ref) not in refs:
                refs.append(str(ref))
        j["prompt"] = j["prompt"] + " " + " ".join(add)
        j["refs"], j["seed"], j["id"] = refs, int(j["seed"]) + 2000, iid + "__y"
        j["path"] = str(Path(j["path"]).with_name(j["id"] + ".png"))
        j["_look_flags"] = fl
        lines.append(json.dumps(j))
    out_jobs.write_text("".join(x + "\n" for x in lines))
    print(out_jobs, len(lines))


def cmd_report(look: dict) -> str:
    L = [f"keys {look['keys']}, flagged {look['keys_flagged']}; look changes {look['look_changes']} over "
         f"{look['cuts_n']} cuts"]
    for k, v in look["people"].items():
        m = v["median"]
        L.append(f"person {k}: anchor {v['anchor']} (head {v['anchor_head']}); facial hair {m.get('facial_hair')}; "
                 f"face distance median {m.get('face_dist')} spread {m.get('face_spread')}")
    L.append(f"{'key':42s} who cls    head  hair_dL hair_ab wear_dL wear_ab face   f.hair     flags")
    for r in look["rows"]:
        d = r.get("dev") or {}
        f = lambda x: "   -  " if x is None else f"{x:6.2f}"
        L.append(f"{r['id'][:42]:42s} {r['key']:3s} {r['cls']:6s} {f(r['head'])} {f(d.get('hair_dL'))}  "
                 f"{f(d.get('hair_ab'))}  {f(d.get('wear_dL'))}  {f(d.get('wear_ab'))} {f(d.get('face'))} "
                 f"{str(hair_class(r.get('facial_hair')) or '-'):10s} {', '.join(r.get('flags') or [])}")
    for t in look["things"]:
        if t["flag"]:
            L.append(f"piece {t['thing']} in {t['id']}: drawn as {t['flag']} (own {t['own']}, other {t['other']})")
    for r_ in look.get("repeats") or []:
        L.append(f"repeat: {r_['cut'][1]} opens on the same picture as {r_['cut'][0]} (correlation {r_['corr']})")
    for c in look["cuts"]:
        if c["changes"]:
            L.append(f"cut {c['cut'][0]} -> {c['cut'][1]}: " + ", ".join(f"{k} {a}" for k, a in c["changes"]))
    s = "\n".join(L)
    print(s)
    return s


LOOKISH = ("look", "identity", "wardrobe", "worn", "hair", "age", "furniture", "set", "repeat", "continuity")


def cmd_compare(look: dict, csv_p: Path, id_map: dict) -> str:
    """Which rows of a human review (shot_no, shot_id, category, what, severity, ...) this pass reproduces: a row
    about looks is reproduced when the key of that shot is flagged here. id_map: {this run's key id: shot id}."""
    import csv
    by_shot = {}
    for iid, fl in look["flagged"].items():
        by_shot.setdefault(id_map.get(iid, iid), {}).update(fl)
    for c in look["cuts"]:
        for k, a in c["changes"]:
            by_shot.setdefault(id_map.get(c["cut"][1], c["cut"][1]), {}).setdefault(f"cut:{k}", []).append(a)
    L, hit, tot = [], 0, 0
    for row in csv.DictReader(csv_p.read_text().splitlines()):
        if not any(w in row["category"].lower() for w in LOOKISH):
            continue
        tot += 1
        got = by_shot.get(row["shot_id"])
        hit += bool(got)
        L.append(f"{'REPRODUCED' if got else 'missed    '} {row['shot_id']} [{row['category']}] {row['what'][:90]}"
                 + (f"  <- {json.dumps(got)}" if got else ""))
    L.append(f"reproduced {hit} of {tot} look rows of the review")
    s = "\n".join(L)
    print(s)
    return s


def cmd_run(run_p: Path, py: str, review: str | None = None, id_map: str | None = None) -> None:
    """The one command: write the feature job, run lookfeat.py at a card station through the line, judge, report,
    write the re-draw jobs (not submitted) and each person's anchor crop."""
    run = json.loads(run_p.read_text())
    out = Path(run["out"])
    cfg_p = out / "look_cfg.json"
    cmd_cfg(run, cfg_p)
    client = os.environ.get("ASSETS_LINE_CLIENT", str(HERE.parent / "run_asset.py"))
    station = os.environ.get("ASSETS_IMAGE_STATION", "card05")
    subprocess.run([sys.executable, client, "--line-cmd", station, "check", "--id", f"look_{out.name}", "--",
                    py, str(HERE / "lookfeat.py"), str(cfg_p)], check=True)
    feats = json.loads((out / "look_cfg_result.json").read_text())
    look = cmd_judge(run, feats, out / "look.json")
    (out / "look_report.txt").write_text(cmd_report(look))
    subprocess.run([py, str(HERE / "lookfeat.py"), "--anchors", str(cfg_p), str(out / "look.json"),
                    str(out / "look_anchor")], check=True)        # crops only (Pillow, no model)
    cmd_retry(run, look, out / "look_retry_jobs.jsonl")
    if review:                                        # which rows of a human review of these keys it reproduces
        (out / "look_vs_review.txt").write_text(cmd_compare(look, Path(review), json.loads(Path(id_map).read_text())
                                                             if id_map else {}))


def main():
    a = sys.argv[1:]
    if a[0] == "report":
        cmd_report(json.loads(Path(a[1]).read_text()))
        return
    if a[0] == "compare":                              # compare LOOK.json REVIEW.csv [IDMAP.json]
        cmd_compare(json.loads(Path(a[1]).read_text()), Path(a[2]),
                    json.loads(Path(a[3]).read_text()) if len(a) > 3 else {})
        return
    if a[0] == "run":
        opt = lambda n: a[a.index(n) + 1] if n in a else None
        cmd_run(Path(a[1]), opt("--py") or os.environ.get("ASSETS_CHECK_PY", sys.executable), opt("--review"),
                opt("--map"))
        return
    run = json.loads(Path(a[1]).read_text())
    if a[0] == "cfg":
        cmd_cfg(run, Path(a[2]))
    elif a[0] == "judge":
        cmd_judge(run, json.loads(Path(a[2]).read_text()), Path(a[3]))
    elif a[0] == "retry":
        cmd_retry(run, json.loads(Path(a[2]).read_text()), Path(a[3]))


if __name__ == "__main__":
    main()
