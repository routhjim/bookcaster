"""11-06 (CHANGES 70): a single action shot's key from its opening staging - one ask per shot after the director
(opening_staging, the language model stubbed here), the mended row composed as a chain head's, applied to the moments
after the set sheets (apply_chains). Mechanics on made-up rows, then a CPU replay over every director passage recorded
under ~/audiobooks (read only; no book text in the repo; the cache goes to a temp dir) through the same steps as
direct: compose -> row_keys -> chain heads -> opening_keys -> apply_chains. Pass: talk, quiet and chain-head key
prompts byte-identical to today's; every single action shot the stub mends changes. Books not on disk are skipped.
    PATH=<fakebin>:$PATH ~/tts_reader/.venv/bin/python3 tools/tests/test_opening_keys_1106.py"""
import builtins, collections, glob, json, os, sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S
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


# --- 1. compose: `state` after the holds clause, `others` as people drawn from their look, never in `who` ----------
row = {"kind": "action", "people": [{"name": "A", "side": "left", "pose": "standing", "at": "at the table",
                                     "facing": "B", "holds": "a cup", "state": "the cup tilted"}],
       "others": [{"look": "a tall guard in grey", "side": "right", "pose": "standing", "at": "by the door",
                   "facing": "A", "holds": "a spear"}], "setting": "A hall."}
c = S.compose(row, ["A"], [], {}, lambda v: v if v == "A" else None)
check("compose: state after holds, the other by their look, the setting last",
      c["picture"] == "A stands at the table, facing B, holding a cup, the cup tilted; a tall guard in grey stands by "
                      "the door on the right, facing A, holding a spear. A hall.", c["picture"])
check("compose: others never in who", c["who"] == ["A"], c["who"])
plain = {"kind": "action", "people": [{"name": "A", "side": "left", "at": "at the table", "holds": "a cup"}],
         "setting": "A hall."}
check("compose: a row with neither is as before",
      S.compose(plain, ["A"], [], {}, lambda v: v)["picture"] == "A stands at the table, holding a cup. A hall.")
q, need = S._key_questions(types.SimpleNamespace(book=TMP), {"who": [], "others": row["others"], "kind": "action"})
check("key questions: the others count in the head count", need == 1 and "count" in q, (need, q))

plainM = dict(plain, people=[dict(plain["people"][0], name="Mara")])
# --- 2. opening_staging: the checks, as the loop ran them ------------------------------------------------------------
named_in = lambda t, cast: [w for w in cast if w in str(t or "")]
res = lambda v: {"Mara": "Mara", "Mari": "Mara"}.get(v)


def stub(replies):
    asked = []

    def ask(msgs, n):
        asked.append((msgs[0]["content"], n))
        return replies.pop(0)
    return ask, asked


good = json.dumps({"before": ["the cup on the table"], "people": [{"name": "Mari", "side": "left", "pose": "standing",
                   "at": "at the table", "facing": "the cup", "holds": "", "state": "sleeves rolled"},
                  {"name": "Guard", "side": "right", "pose": "standing", "at": "by the door", "facing": "Mari",
                   "holds": "a spear"}], "others": [], "setting": "A cup on the table."})
ask, asked = stub([good, "a tall man in grey mail."])
r = S.opening_staging("Mara drinks.", plainM, ["Mara"], res, named_in, ask)
m = r["mended"]
check("staging: a name resolves (Mari -> Mara); its state kept", m and m["people"][0]["name"] == "Mara"
      and m["people"][0]["state"] == "sleeves rolled", m)
check("staging: a name with no cast member gets a look and goes to others (side/at/facing/holds kept)",
      m["others"] == [{"look": "a tall man in grey mail", "side": "right", "pose": "standing", "at": "by the door",
                       "facing": "Mari", "holds": "a spear"}] and r["moved_to_others"] == ["Guard"], m["others"])
check("staging: one ask, one look ask (1500 / 60 tokens)", [n for _, n in asked] == [1500, 60]
      and asked[1][0].startswith("In one shot of a film: Mara drinks.. Describe Guard as a painter"), asked)
check("staging: the ask is the general definition with the row filled in", '"Mara drinks."' in asked[0][0]
      and 'from these names: Mara.' in asked[0][0] and '"before": for every thing' in asked[0][0])
ask, _ = stub([good, "a very tall man in grey mail with a long spear and a red cloak over it all"])
r = S.opening_staging("Mara drinks.", plainM, ["Mara"], res, named_in, ask)
check("staging: a look over 12 words drops that person", r["dropped"] == ["Guard"] and not r["mended"]["others"], r)
bad = json.dumps({"people": [], "setting": "Mara watches the cup."})
ask, asked = stub([bad, good.replace("A cup on", "Mara sees a cup on"), "a guard"])
r = S.opening_staging("Mara drinks.", plainM, ["Mara"], res, named_in, ask)
check("staging: a setting naming someone is asked again, then the director's setting",
      len(r["tries"]) == 2 and r["mended"]["setting"] == "A hall." and r["fallback"].startswith("setting named"), r)
ask, _ = stub(["no", "still no"])
r = S.opening_staging("Mara drinks.", plainM, ["Mara"], res, named_in, ask)
check("staging: not JSON twice: the director's row", r["mended"] is None and r["fallback"], r)

# --- 3. opening_keys + apply_chains: single action shots only, cached, applied after the sheets ----------------------
shots = [S.compose(dict(plainM), ["Mara"], [], {}, res), S.compose(dict(plainM, kind="quiet"), ["Mara"], [], {}, res),
         {"kind": "talk", "speaker": "Mara", "who": ["Mara"]}, S.compose(dict(plainM), ["Mara"], [], {}, res)]
S.row_keys(shots)
shots[3]["_chain"] = {"key": "1:0", "shot": dict(shots[3], chain={"lines": ["x"], "shots": [4]})}
ask, asked = stub([good, "a guard"])
cache = TMP / "p000010.opening.json"
n = S.opening_keys(shots, ["Mara drinks."] * 4, ["Mara"], [], {}, res, named_in, ask, cache, log=lambda m_: None)
check("opening_keys: one ask, for the single action shot only", n == 1 and len(asked) == 2 and "_open" in shots[0]
      and not any("_open" in x for x in shots[1:]), [list(x) for x in shots])
ask2, asked2 = stub([])
S.opening_keys([dict(x) for x in shots], ["Mara drinks."] * 4, ["Mara"], [], {}, res, named_in, ask2, cache, log=lambda m_: None)
check("opening_keys: cached by the ask", not asked2)
es = [{"id": f"m{i}", "t": float(i), "dur": 1.0, "kind": x["kind"], "who": x.get("who") or []} for i, x in enumerate(shots)]
by = {e["id"]: [dict(x, id=e["id"] + "_s0")] for e, x in zip(es, shots)}
es2 = S.apply_chains(es, by, log=lambda m_: None)
s0 = by["m0"][0]
check("apply_chains: the single shot keyed from its opening staging",
      s0["keyshot"] == s0["picture"] == "Mara stands at the table, facing the cup, sleeves rolled; a guard stands by the door "
                                        "on the right, facing Mari, holding a spear. A cup on the table."
      and s0["others"] and "_open" not in s0, s0)
check("apply_chains: no tag left, quiet and talk as planned", not any("_open" in x for v in by.values() for x in v)
      and by["m1"][0] == dict(shots[1], id="m1_s0") and by["m2"][0] == dict(shots[2], id="m2_s0"))

# --- 4. replay over the recorded passages ----------------------------------------------------------------------------
cnt, bad_, missed, nn = collections.Counter(), [], [], 0
for bk in ("the_long_dark_between", "acts_esv", "sherlock_adventures", "knight_forged_v3"):
    book = AB / bk
    for f in sorted(glob.glob(str(book / "work/video/*/shots/passages/p??????.json"))):
        stem, pas = Path(f).parents[2].name, Path(f).stem
        try:
            rows = [x for x in json.loads(json.load(open(f))["reply"])["shots"] if isinstance(x, dict)]
        except (ValueError, KeyError, TypeError):
            continue
        cf = f.replace(".json", ".chains.json")
        chs = [c_ for c_ in (json.load(open(cf)).get("chains", []) if os.path.exists(cf) else []) if c_.get("shots")]
        cast = list(dict.fromkeys(p["name"] for x in rows for p in x.get("people") or [] if isinstance(p, dict)
                                  and isinstance(p.get("name"), str)))
        rs = lambda v: v if v in cast else None

        def plan():
            sh = S.row_keys([S.compose({k: v for k, v in x.items() if k not in ("start", "speaker", "listener")}, cast,
                                       [], None, rs) if x.get("kind") != "talk" else dict(x) for x in rows])
            for j_, (nums, head) in enumerate(S.chain_heads(sh, chs, cast, [], {}, rs)):
                for n_ in nums:
                    sh[n_ - 1] = dict(sh[n_ - 1], _chain={"key": f"{pas}:{j_}", "shot": head})
            return sh
        today, new = plan(), plan()

        def mend(msgs, n_):             # the stub: the director's row, each one with a state, one other
            q_ = msgs[0]["content"]
            if n_ == 60:
                return "a stranger in a grey coat"
            ppl = json.loads(q_.split("staged its opening picture as: ", 1)[1].split('; setting: "', 1)[0])
            return json.dumps({"before": [], "people": [dict(p, state="ready") for p in ppl],
                               "others": [{"look": "a stranger in a grey coat", "side": "right"}],
                               "setting": q_.split('; setting: "', 1)[1].split('". Give the staging', 1)[0]})
        acts = [str(x.get("action") or "") for x in rows]
        S.opening_keys(new, acts, cast, [], {}, rs, named_in, mend, TMP / f"{bk}_{stem}_{pas}.opening.json",
                       log=lambda m_: None)
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
            cat = "chain_head" if y0.get("chain") else ("single_action" if kind == "action" else kind)
            if "who" not in y0:  # old talk rows: speaker and listener
                y0 = y1 = dict(y0, picture=y0.get("picture") or "",
                               who=[w for w in (y0.get("speaker"), y0.get("listener")) if w and w != "narrator"])
            mid = S.shot_mid("quiet" if kind == "talk" else kind, int(pas[1:]) / 10 + float(x.get("start") or 0))
            same = S._key_prompt(book, stem, mid, y0) == S._key_prompt(book, stem, mid, y1) and \
                y0.get("who") == y1.get("who")
            nn += 1
            cnt[f"{bk} {cat} {'same' if same else 'differ'}"] += 1
            if not same and cat != "single_action":
                bad_.append(f"{bk} {pas}#{i + 1} {cat}")
            if same and cat == "single_action":
                missed.append(f"{bk} {pas}#{i + 1}")
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")
print(f"  shots replayed: {nn}, prompts changed: {sum(v for k, v in cnt.items() if k.endswith('differ'))}")
check("replay: talk, quiet and chain-head key prompts byte-identical", not bad_, bad_[:5])
check("replay: every single action shot the stub mends is keyed from its opening staging", not missed, missed[:5])
if fails:
    sys.exit(f"{len(fails)} failed: {fails}")
print("ok")
