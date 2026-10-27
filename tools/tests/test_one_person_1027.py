"""10-27: one person, one key, whatever the spelling: look.json kept "Doctor Watson" first, the chapter said
"Dr. Watson", so Watson never got a LoRA (the chapter-cast filter compared names) and a LoRA under doctor_watson would
not have been found for "Dr. Watson" in a shot."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import flow3d as F
import mannequin_flow as MF

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    if not ok:
        fails.append(name)

check("person keys: Dr. = Doctor, folder = name", F.person_key("Dr. Watson") == F.person_key("Doctor Watson")
      and MF._person_key("doctor_watson") == MF._person_key("Dr. Watson"))
with tempfile.TemporaryDirectory() as d:
    book = Path(d)
    vd = book / "work" / "video"
    (vd / "charlora" / "doctor_watson").mkdir(parents=True)
    (vd / "charlora" / "doctor_watson" / "build.json").write_text(json.dumps({"lora": "w.safetensors", "trigger": "chr_w"}))
    check("a LoRA trained as Doctor Watson is found for 'Dr. Watson'", MF._lora(book, "Dr. Watson") == ("w.safetensors", "chr_w"),
          MF._lora(book, "Dr. Watson"))
    (vd / "look.json").write_text(json.dumps({"characters": {"Doctor Watson": "a man", "Dr. Watson": "a man"}}))
    (vd / "charlora" / "doctor_watson" / "build.json").unlink()
    todo = F.lora_todo(book, ["Dr. Watson"])
    check("the chapter's 'Dr. Watson' asks for Doctor Watson's LoRA (one person)", list(todo) == ["Doctor Watson"], todo)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
