"""11-06: a file's prompts are encoded in ONE ltx_split encode (one Gemma load; the runner turns each prompt's hidden
states into its embedding as it goes), not in chunks of 40 processes; nothing is run when every embedding exists."""
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import remake_chapter as R

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

out = Path(tempfile.mkdtemp(prefix="enc_once_"))
L = out / "ltx"; (L / "emb").mkdir(parents=True)
enc = L / "enc_dev.json"
enc.write_text(json.dumps([{"name": f"p{i}", "prompt": f"prompt {i}"} for i in range(221)]))
calls = []
def fake_step(cx, name, args):
    calls.append(args)
    for x in json.loads(Path(args[1]).read_text()):
        (L / "emb" / f"{x['name']}.pt").write_text("e")
    return 0
R.ltx_step = fake_step
cx = SimpleNamespace(OUT=out)
R.encode_all(cx, enc, "enc_dev", ["--dev"])
check("221 prompts: one encode process, on the whole file", calls == [["encode", str(enc), "--dev"]], calls)
R.encode_all(cx, enc, "enc_dev", ["--dev"])
check("all cached: no process at all", len(calls) == 1, calls)
R.ltx_step = lambda *a: 0
(L / "emb" / "p7.pt").unlink()
try:
    R.encode_all(cx, enc, "enc_dev", ["--dev"])
    check("a missing embedding after the encode stops the run", False)
except SystemExit as e:
    check("a missing embedding after the encode stops the run", "1 embeddings still missing" in str(e), e)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
