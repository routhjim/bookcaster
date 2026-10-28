"""10-28: the LTX split runner lifts its open-file soft limit (the talk run died at 1024 open shm tensors)."""
import resource, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "ltx_runner" / "ltx_split_next.py"
src = R.read_text()
fn = src[src.index("def raise_nofile"):src.index("\nraise_nofile()")]
code = (f"import resource\nresource.setrlimit(resource.RLIMIT_NOFILE, (1024, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))\n"
        f"{fn}\nprint(raise_nofile(), resource.getrlimit(resource.RLIMIT_NOFILE)[1])")
soft, hard = map(int, subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split())
ok = soft == hard and hard > 1024 and "\nraise_nofile()\n" in src and src.index("\nraise_nofile()") < src.index("import torch")
print(("PASS" if ok else "FAIL") + f" soft limit lifted to the hard limit ({soft}/{hard}) before torch loads")
sys.exit(0 if ok else 1)
