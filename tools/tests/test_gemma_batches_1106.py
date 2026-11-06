"""11-06: ltx_split encode runs Gemma on each prompt's real tokens in length-sorted batches: every prompt once, longest
first, a batch's rows padded to its longest within ENC_TOKENS. (The runner needs torch; its batching is pure Python
and is read from the file.)"""
import ast, json, re, sys
from pathlib import Path

src = (Path(__file__).resolve().parents[1] / "ltx_runner" / "ltx_split.py").read_text()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "enc_batches")
budget = int(re.search(r"^ENC_TOKENS = (\d+)", src, re.M).group(1))
ns = {"ENC_TOKENS": budget}
exec(compile(ast.Module([fn], []), "enc_batches", "exec"), ns)
enc_batches = ns["enc_batches"]

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

lens = [68, 15, 304, 120, 9, 1024, 90, 90, 90, 33, 250, 512, 513, 7]
items = [({"name": f"p{i}"}, list(range(n))) for i, n in enumerate(lens)]
bs = enc_batches(items, budget)
names = [j["name"] for b in bs for j, _ in b]
check("every prompt exactly once", sorted(names) == sorted(j["name"] for j, _ in items), names)
flat = [len(x) for b in bs for _, x in b]
check("longest first", flat == sorted(flat, reverse=True), flat)
check("each batch padded to its longest fits the budget (a single longer prompt goes alone)",
      all(len(b[0][1]) * len(b) <= budget or len(b) == 1 for b in bs), [[len(x) for _, x in b] for b in bs])
check("short prompts share a batch", max(len(b) for b in bs) > 1, [[len(x) for _, x in b] for b in bs])
check("nothing to encode: no batches", enc_batches([], budget) == [])
check("the encoder skips the vision/audio towers and runs without a KV cache",
      "GEMMA_SKIP.search(k)" in src and "use_cache=False" in src and "position_ids=" in src)
check("the connector takes CONN_ROWS prompts at a time and saves off its thread (11-06)",
      re.search(r"^CONN_ROWS = \d+", src, re.M) is not None and "saver.submit(save" in src and "f.result()" in src)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
