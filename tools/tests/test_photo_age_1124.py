"""11-24: someone drawn from photographs looks the age the photographs show: the look's number fails neither their
LoRA's dataset pictures nor their sheet; everyone else is checked against the look as before."""
import sys, types
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import charlora as CL
import flow3d as F
from tts_reader import video

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

age_ok = CL.Char.age_ok
c = types.SimpleNamespace(age=30, photo=False)
check("no photographs: 50 is not 30", age_ok(c, 50) is False, age_ok(c, 50))
check("no photographs: 35 is 30", age_ok(c, 35) is True)
c.photo = True
check("photographs: the age is not judged", age_ok(c, 50) is None, age_ok(c, 50))
check("an object made before the flag: judged as before", age_ok(types.SimpleNamespace(age=30), 50) is False)

keep = (video.looks, video.reference_jobs, video.outfit_jobs, video.registry_at, video.face_photos)
try:
    video.looks = lambda plan, reg: ({"Orla": "a woman of about 30, red hair", "Tam": "a man of about 60, bald"}, {})
    video.reference_jobs = lambda plan, style, refs: [{"id": "ref_orla", "path": "o.png"}, {"id": "ref_tam", "path": "t.png"}]
    video.outfit_jobs = lambda plan, style, refs: [{"id": "ref_orla_coat", "path": "oc.png"}]
    video.registry_at = lambda refs: "reg"
    video.face_photos = lambda reg, n: ["p.png"] if n == "Orla" else []
    out = F._sheet_facts({"wardrobe": {"Orla": [{"id": "coat", "dress": "a long coat"}]}}, "s", "refs")
    by = {j["id"]: f for j, f in out}
    check("the photographed person's sheet has no age to fail", by["ref_orla"]["age"] is None, by["ref_orla"])
    check("nor their outfit sheet", by["ref_orla_coat"]["age"] is None, by["ref_orla_coat"])
    check("the other's sheet keeps the look's age", by["ref_tam"]["age"] == 60, by["ref_tam"])
finally:
    video.looks, video.reference_jobs, video.outfit_jobs, video.registry_at, video.face_photos = keep

sys.exit(1 if fails else 0)
