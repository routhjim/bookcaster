# suite: slow
"""11-06 (CHANGES 63): a single action shot's key is its own staging row through compose() (row_keys), as a chain
head's already was. CPU replay over every director row recorded under ~/audiobooks (read only; no book text in the
repo): each row's key prompt by today's route (the director's recorded keyshot) and by row_keys, through main's
_key_prompt. Pass: talk, quiet and chain-head prompts byte-identical; the prompts that change are exactly the single
action rows. Books not on disk are skipped.
    PATH=<fakebin>:$PATH ~/tts_reader/.venv/bin/python3 tools/tests/test_row_keys_1106.py"""
import builtins, collections, glob, json, os, sys
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
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


# the mechanics, on a made-up passage: action rows keyed from their picture, the others untouched
sh = [{"kind": "action", "people": [{"name": "A", "side": "left", "facing": "the door"}], "setting": "A hall.",
       "keyshot": "written", "moment": "m"},
      {"kind": "quiet", "people": [{"name": "A"}], "setting": "A hall."}, {"kind": "talk", "speaker": "A"}]
sh = S.row_keys([S.compose(x, ["A"], [], {}, lambda v: v) if x["kind"] != "talk" else x for x in sh])
check("row_keys: an action row's key is its composed picture", sh[0]["keyshot"] == sh[0]["picture"]
      and "facing the door" in S.moment(sh[0]), sh[0])
check("row_keys: quiet and talk rows untouched", "keyshot" not in sh[1] and "keyshot" not in sh[2])

cnt, bad, missed, n = collections.Counter(), [], [], 0
for bk in ("the_long_dark_between", "acts_esv", "sherlock_adventures", "knight_forged_v3"):
    book = AB / bk
    for f in sorted(glob.glob(str(book / "work/video/*/shots/passages/p??????.json"))):
        stem, pas = Path(f).parents[2].name, Path(f).stem
        try:
            rows = json.loads(json.load(open(f))["reply"])["shots"]
        except (ValueError, KeyError, TypeError):
            continue
        cf = f.replace(".json", ".chains.json")
        chs = [c for c in (json.load(open(cf)).get("chains", []) if os.path.exists(cf) else []) if c.get("shots")]
        inch = {i for c in chs for i in c["shots"]}
        res = lambda v: v

        def comp(x):
            if not isinstance(x, dict) or x.get("kind") == "talk":
                return x
            return S.compose({k: v for k, v in x.items() if k not in ("start", "speaker", "listener")},
                             [p["name"] for p in x.get("people") or [] if isinstance(p, dict)], [], None, res)
        today = [comp(x) for x in rows]
        new = S.row_keys([dict(x) if isinstance(x, dict) else x for x in today])
        heads = {nums[0]: (h0, h1) for (nums, h0), (_, h1) in
                 zip(S.chain_heads(today, chs, [], [], {}, res), S.chain_heads(new, chs, [], [], {}, res))}
        for i, (x, y) in enumerate(zip(today, new), 1):
            if not isinstance(x, dict) or (i in inch and i not in heads):
                continue        # (rows inside a chain but not its head are not keyed)
            kind = x.get("kind") if x.get("kind") in ("action", "quiet", "talk") else "quiet"
            cat = "chain_head" if i in heads else ("single_action" if kind == "action" else kind)
            if i in heads:
                x, y = heads[i]
            if "who" not in x:  # old talk rows: speaker and listener
                x = y = dict(x, picture=x.get("picture") or "", who=[w for w in (x.get("speaker"), x.get("listener")) if w and w != "narrator"])
            t = int(pas[1:]) / 10 + float(x.get("start") or rows[i - 1].get("start") or 0)
            mid = S.shot_mid("quiet" if kind == "talk" else kind, t)
            same = S._key_prompt(book, stem, mid, x) == S._key_prompt(book, stem, mid, y)
            n += 1
            cnt[f"{bk} {cat} {'same' if same else 'differ'}"] += 1
            if not same and cat != "single_action":
                bad.append(f"{bk} {pas}#{i} {cat}")
            if same and cat == "single_action":
                missed.append(f"{bk} {pas}#{i}")
for k, v in sorted(cnt.items()):
    print(f"  {k}: {v}")
print(f"  rows replayed: {n}, prompts changed: {sum(v for k, v in cnt.items() if k.endswith('differ'))}")
check("replay: talk, quiet and chain-head key prompts byte-identical", not bad, bad[:5])
check("replay: every single action row's prompt changes (the row, not the written key)", not missed, missed[:5])
if fails:
    sys.exit(f"{len(fails)} failed: {fails}")
print("ok")
