"""10-31: order_jobs puts the most estimated card time first (frames x route factor; action costs 1.3x a talk frame),
keeps equal costs in plan order, and keeps every job. CPU only: run with the LTX venv python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from ltx_split_next import order_jobs
jobs = [{"name": "t1", "audio": "a.wav", "frames": 97}, {"name": "a1", "frames": 105, "neg_extra": ""},
        {"name": "t2", "audio": "a.wav", "frames": 193}, {"name": "t3_c0", "audio": "a.wav", "frames": 121},
        {"name": "t3_c1", "audio": "a.wav", "frames": 121, "chain_from": "t3_c0"}, {"name": "t4", "audio": "a.wav", "frames": 57}]
got = [j["name"] for j in order_jobs(jobs)]
want = ["t2", "a1", "t3_c0", "t3_c1", "t1", "t4"]     # a1: 105 x 1.3 = 136.5 > 121
fails = [] if got == want else [f"order {got} != {want}"]
if sorted(got) != sorted(j["name"] for j in jobs):
    fails.append("jobs lost or duplicated")
print({"fails": fails})
sys.exit(1 if fails else 0)
