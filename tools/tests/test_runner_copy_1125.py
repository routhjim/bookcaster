"""Every module the pipelined LTX runner imports from its own folder is copied beside it (11-25)."""
import ast, sys
from pathlib import Path
T = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(T))
import remake_chapter as R

RD = T / "ltx_runner"
need = set()
for f in ("ltx_split_next.py",):
    for n in ast.walk(ast.parse((RD / f).read_text())):
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module] if isinstance(n, ast.ImportFrom) and n.module else []
        need |= {x.split(".")[0] + ".py" for x in names if (RD / (x.split(".")[0] + ".py")).exists()}
miss = sorted(need - set(R.RUNNER_NEXT) - {"ltx_split.py"})
print("imports of its own folder:", sorted(need))
ok = not miss and "ltx_split_next.py" in R.RUNNER_NEXT
src = (T / "remake_chapter.py").read_text()
ok = ok and "RUNNER_NEXT if PP" in src
print("all passed" if ok else f"FAILED: not copied {miss}")
sys.exit(0 if ok else 1)
