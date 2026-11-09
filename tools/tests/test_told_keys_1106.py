# suite: slow
"""11-06 (CHANGES 71/72, user: "lock in E10"): a told-place (flashback) shot's key holds only the event's LoRA'd people
the shot's action names (told_people: one ask, the language model stubbed here; Tier A, main's name rule), its
action words appended, applied after the opening ask (opening_keys) through apply_chains. Mechanics on made-up rows,
then a CPU replay over every director passage recorded under ~/audiobooks (read only; no book text in the repo; the
caches go to a temp dir) through the same steps as direct: compose -> row_keys -> chain heads -> opening_keys ->
told_keys -> apply_chains, the told places read from each passage's recorded ask (direct's `others` line). Pass:
talk, quiet, action and chain-head key prompts off the told places byte-identical to today's (opening included);
only told shots change; no told key holds a person the action does not name. Books not on disk are skipped.
    PATH=<fakebin>:$PATH ~/tts_reader/.venv/bin/python3 tools/tests/test_told_keys_1106.py"""
import builtins, collections, glob, json, os, re, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S
import registry as RG
AB = Path.home() / "audiobooks"
_open = builtins.open


def _ro(f, mode="r", *a, **k):         # ~/audiobooks stays read only
    if any(c in mode for c in "wax+") and os.path.realpath(str(f)).startswith(os.path.realpath(AB) + "/"):
        raise PermissionError(f"write under ~/audiobooks: {f}")
    return _open(f, mode, *a, **k)


builtins.open = _ro
S._converse = lambda *a, **k: (_ for _ in ()).throw(AssertionError("the language model was called"))
fails = []
TMP = Path(tempfile.mkdtemp())


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


named_in = lambda t, cast: [w for w in cast if re.search(rf"\b{re.escape(w)}\b", str(t or ""))]
LOR = ["Mara", "Tom"]                   # the made-up book's LoRA'd people
res_l = lambda v: {"Mara": "Mara", "Mari": "Mara", "Tom": "Tom", "a woman": "Mara"}.get(v)
res_in = lambda c_: (lambda v: res_l(v) or (v if v in c_ else None))


def stub(replies):
    asked = []

    def ask(msgs, n):
        asked.append((msgs[0]["content"], n))
        return replies.pop(0)
    return ask, asked


# --- 1. told_people: the ask, the LoRA and name checks ---------------------------------------------------------------
row = {"kind": "action", "place": "The Old Court (inside)",
       "people": [{"name": "Ann", "side": "left", "pose": "standing", "at": "by the gate"}], "setting": "A court."}
rep = json.dumps([{"name": "Mari", "side": "right", "pose": "standing", "at": "before the seat", "facing": "the judge",
                   "holds": ""}, {"name": "Tom", "side": "left"}, {"name": "a woman"}, {"name": "Pilot", "side": "left"}])
ask, asked = stub([rep])
r = S.told_people("The judge sends Mara away.", row, "Ann", ["Ann"] + LOR, LOR, res_l, named_in, ask)
check("told_people: the ask is the general definition with the row filled in, 800 tokens",
      asked == [("This shot shows what Ann tells of: The judge sends Mara away at The Old Court. Which of these people "
                 "were themselves IN what is shown (not only telling it or listening now)? Ann, Mara, Tom. For each "
                 "one, give where they are in the picture, as the staging fields: name, side, pose, at, facing, holds. "
                 "JSON list only.", 800)], asked)
check("told_people: kept = LoRA'd and named in the action, with the ask's staging",
      r["kept"] == [{"side": "right", "pose": "standing", "at": "before the seat", "facing": "the judge", "holds": "",
                     "name": "Mara"}], r["kept"])
check("told_people: a LoRA'd person the action does not name is dropped (Tier A)", r["not_named"] == ["Tom"], r)
check("told_people: a name with no LoRA is never kept (no new casting)", r["no_lora"] == ["Pilot"], r)
ask, asked = stub(["no list", "still none"])
r = S.told_people("Mara runs.", row, "Ann", ["Ann"] + LOR, LOR, res_l, named_in, ask)
check("told_people: not a list twice: nobody kept", r["kept"] == [] and r["fallback"] and len(asked) == 2, r)

# --- 2. told_keys + apply_chains: told shots only, after the opening, cached ---------------------------------------
quiet_t = S.compose(dict(row, kind="quiet", action="Mara kneels."), ["Ann"], [], {}, res_in(["Ann"]))
act_t = S.compose(dict(row, action="Mara runs."), ["Ann"], [], {}, res_in(["Ann"]))
act_home = S.compose(dict(row, place="Home", action="Ann runs."), ["Ann"], [], {}, res_in(["Ann"]))
shots = S.row_keys([act_t, quiet_t, {"kind": "talk", "speaker": "Ann", "who": ["Ann"], "place": row["place"]},
                    act_home])
act_t["_open"] = {"people": [{"name": "Ann", "side": "left"}], "others": [{"look": "a guard in grey", "side": "right"}],
                  "setting": "A court, the gate shut.", "who": ["Ann"], "pose": {}, "placed": [], "picture": "x",
                  "keyshot": "x"}
before = json.loads(json.dumps(shots))
acts = ["Mara runs to the gate.", "Mara kneels.", "", "Ann runs."]
tellers = ["Ann", "Ann", None, None]
ask, asked = stub([json.dumps([{"name": "Ann"}, {"name": "Mara", "side": "left", "pose": "running"}]),
                   json.dumps([{"name": "Tom"}])])
cache = TMP / "p000010.told.json"
n = S.told_keys(shots, acts, tellers, ["Ann"], [], {}, res_l, res_in, named_in, LOR, ask, cache, log=lambda m_: None)
check("told_keys: one ask per told non-talk shot", n == 2 and len(asked) == 2, (n, asked))
o0, o1 = shots[0]["_open"], shots[1]["_open"]
check("told_keys: the teller is dropped; the kept one from the ask's row; the opening's others and setting stay; "
      "the action words end the key", o0["who"] == ["Mara"] and o0["keyshot"] ==
      "Mara stands; a guard in grey stands on the right. A court, the gate shut. Mara runs to the gate."
      and o0["picture"] + " Mara runs to the gate." == o0["keyshot"], o0)
check("told_keys: a quiet told shot with nobody named: nobody, its picture = setting + action",
      o1["who"] == [] and o1["picture"] == "A court. Mara kneels." and "keyshot" not in o1, o1)
check("told_keys: talk and present-place shots untouched", shots[2] == before[2] and shots[3] == before[3])
ask2, asked2 = stub([])
S.told_keys([json.loads(json.dumps(x)) for x in before], acts, tellers, ["Ann"], [], {}, res_l, res_in, named_in, LOR,
            ask2, cache, log=lambda m_: None)
check("told_keys: cached by the ask", not asked2)
es = [{"id": f"m{i}", "t": float(i), "dur": 1.0, "kind": x["kind"], "who": x.get("who") or []}
      for i, x in enumerate(shots)]
by = {e["id"]: [dict(x, id=e["id"] + "_s0")] for e, x in zip(es, shots)}
es2 = S.apply_chains(es, by, log=lambda m_: None)
check("apply_chains: the told keys applied, the moment's who too, no tag left",
      by["m0"][0]["keyshot"] == o0["keyshot"] and es2[0]["who"] == ["Mara"] and by["m1"][0]["who"] == []
      and not any("_open" in x for v in by.values() for x in v), by["m0"][0])

# --- 3. replay over the recorded passages ----------------------------------------------------------------------------
cnt, bad_, extra, nn, ntold = collections.Counter(), [], [], 0, 0
for bk in ("the_long_dark_between", "acts_esv", "sherlock_adventures", "knight_forged_v3"):
    book = AB / bk
    if not book.exists():
        continue
    R_ = RG.Book(book)
    lor = [w for w in R_.people() if R_.lora(w)[0]]
    for f in sorted(glob.glob(str(book / "work/video/*/shots/passages/p??????.json"))):
        stem, pas = Path(f).parents[2].name, Path(f).stem
        try:
            d_ = json.load(open(f))
            rows = [x for x in json.loads(d_["reply"])["shots"] if isinstance(x, dict)]
        except (ValueError, KeyError, TypeError):
            continue
        q = (d_.get("ask") or {}).get("q", "")      # direct's `others` and their tellers, as it asked them
        m_ = re.search(r"Other places in this passage: (.*)", q)
        tk = {RG.place_key(z.strip()) for z in re.findall(r"(?:^|; )([^;]+?) \(its fixed things", m_.group(1))} \
            if m_ else set()
        m2 = re.search(r"In this passage it may cut to: (.*?)\.(?:\s|$)", q)
        tl_ = {RG.place_key(a): b for a, b in re.findall(r"(?:^|; )(.+?) \(as (.+?) tells of it\)", m2.group(1))} \
            if m2 else {}
        cf = f.replace(".json", ".chains.json")
        chs = [c_ for c_ in (json.load(open(cf)).get("chains", []) if os.path.exists(cf) else []) if c_.get("shots")]
        cast = list(dict.fromkeys(p["name"] for x in rows for p in x.get("people") or [] if isinstance(p, dict)
                                  and isinstance(p.get("name"), str)))
        rs = lambda v: v if v in cast else None
        rin = lambda c_: (lambda v: v if v in c_ else None)

        def plan():
            sh = S.row_keys([S.compose({k: v for k, v in x.items() if k not in ("start", "speaker", "listener")}, cast,
                                       [], None, rs) if x.get("kind") != "talk" else dict(x) for x in rows])
            for j_, (nums, head) in enumerate(S.chain_heads(sh, chs, cast, [], {}, rs)):
                for n_ in nums:
                    sh[n_ - 1] = dict(sh[n_ - 1], _chain={"key": f"{pas}:{j_}", "shot": head})
            return sh
        today, new = plan(), plan()

        def mend(msgs, n_):             # the opening stub: the director's row, each one with a state
            q_ = msgs[0]["content"]
            if n_ == 60:
                return "a stranger in a grey coat"
            ppl = json.loads(q_.split("staged its opening picture as: ", 1)[1].split('; setting: "', 1)[0])
            return json.dumps({"before": [], "people": [dict(p, state="ready") for p in ppl], "others": [],
                               "setting": q_.split('; setting: "', 1)[1].split('". Give the staging', 1)[0]})

        def told(msgs, n_):             # the told stub: everyone offered was in the event (the code must cut it)
            names = msgs[0]["content"].split("listening now)? ", 1)[1].split(". For each one", 1)[0].split(", ")
            return json.dumps([{"name": w, "side": "left", "pose": "standing"} for w in names])
        acts = [str(x.get("action") or "") for x in rows]
        tellers = [None if x.get("kind") == "talk" or RG.place_key(x.get("place") or "") not in tk else
                   tl_.get(RG.place_key(x.get("place") or "")) or "the narration" for x in rows]
        for sh in (today, new):
            S.opening_keys(sh, acts, cast, [], {}, rs, named_in, mend, TMP / f"{bk}_{stem}_{pas}.opening.json",
                           log=lambda m_: None)
        S.told_keys(new, acts, tellers, cast, [], {}, lambda v: v if v in lor else None, rin, named_in, lor, told,
                    TMP / f"{bk}_{stem}_{pas}.told.json", log=lambda m_: None)
        moments = []
        for sh in (today, new):
            es = [{"id": f"m{i}", "t": float(i), "dur": 1.0, "kind": x.get("kind"), "who": x.get("who") or []}
                  for i, x in enumerate(sh)]
            by = {e["id"]: [dict(x, id=e["id"] + "_s0")] for e, x in zip(es, sh)}
            moments.append({e["id"]: by[e["id"]][0] for e in S.apply_chains(es, by, log=lambda m_: None)})
        for i, x in enumerate(rows):
            mid_ = f"m{i}"
            if mid_ not in moments[0]:
                continue        # (inside a chain, not its head)
            y0, y1 = moments[0][mid_], moments[1][mid_]
            kind = x.get("kind") if x.get("kind") in ("action", "quiet", "talk") else "quiet"
            istold = tellers[i] is not None and not y0.get("chain")
            cat = "told" if istold else ("chain_head" if y0.get("chain") else kind)
            if "who" not in y0:  # old talk rows: speaker and listener
                y0 = y1 = dict(y0, picture=y0.get("picture") or "",
                               who=[w for w in (y0.get("speaker"), y0.get("listener")) if w and w != "narrator"])
            mid = S.shot_mid("quiet" if kind == "talk" else kind, int(pas[1:]) / 10 + float(x.get("start") or 0))
            same = S._key_prompt(book, stem, mid, y0) == S._key_prompt(book, stem, mid, y1) and \
                y0.get("who") == y1.get("who")
            nn += 1
            ntold += istold
            cnt[f"{bk} {cat} {'same' if same else 'differ'}"] += 1
            if not same and not istold:
                bad_.append(f"{bk} {pas}#{i + 1} {cat}")
            if istold and [w for w in y1.get("who") or [] if not named_in(acts[i], [w])]:
                extra.append(f"{bk} {pas}#{i + 1} {y1['who']}")
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")
print(f"  shots replayed: {nn}, told: {ntold}, prompts changed: {sum(v for k, v in cnt.items() if k.endswith('differ'))}")
check("replay: talk, quiet, action and chain-head prompts off the told places byte-identical", not bad_, bad_[:5])
check("replay: no told key holds a person its action does not name", not extra, extra[:5])
check("replay: told shots found and changed where books are on disk",
      not nn or sum(v for k, v in cnt.items() if k.endswith("told differ")) > 0, dict(cnt))
if fails:
    sys.exit(f"{len(fails)} failed: {fails}")
print("ok")
