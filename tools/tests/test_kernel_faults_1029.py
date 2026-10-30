"""10-29: a startup GPU page fault (no reset) is retried up to PAGE_FAULT_RETRIES times a job; a ring timeout,
reset or anything else still stops at once; the check reads only the step's own kernel messages."""
import sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import remake_chapter as R

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

out = Path(tempfile.mkdtemp())
PF = ("Oct 29 17:30:35 fedora kernel: amdgpu 0000:05:00.0: [gfxhub] page fault (src_id:0 ring:40 vmid:8 pasid:8480)\n"
      "Oct 29 17:30:35 fedora kernel: amdgpu 0000:05:00.0: GCVM_L2_PROTECTION_FAULT_STATUS:0x00841051")
RING = "Oct 29 17:30:35 fedora kernel: amdgpu 0000:05:00.0: ring gfx_0.0.0 timeout, signaled seq=1"
real, seen = R.subprocess.run, []
def run_with(txt):
    def f(cmd, *a, **k):
        seen.append(cmd)
        return type("R", (), {"stdout": txt})()
    R.subprocess.run = f
    try:
        R.kernel_check(out, "@1793000000")
        return "go"
    except SystemExit:
        return "stop"
    finally:
        R.subprocess.run = real
R._faults_let_through.clear()
got = [run_with(PF) for _ in range(R.PAGE_FAULT_RETRIES + 1)]
check("startup page faults: retried, then a stop", got == ["go"] * R.PAGE_FAULT_RETRIES + ["stop"], got)
R._faults_let_through.clear()
check("a ring timeout stops at once", run_with(RING) == "stop")
R._faults_let_through.clear()
check("a page fault WITH a ring timeout stops", run_with(PF + "\n" + RING) == "stop")
check("quiet log: go", run_with("Oct 29 fedora kernel: usb 3-1: new device") == "go")
check("the window is the step's own", seen and seen[-1][-1] == "@1793000000", seen[-1:])
print({"fails": fails})
sys.exit(1 if fails else 0)
