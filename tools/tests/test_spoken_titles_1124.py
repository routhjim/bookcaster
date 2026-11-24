"""11-24: a title written as it is spoken ("Mister", "Missus") names the same person as the written one ("Mr.",
"Mrs."), and still tells apart two people who share a surname."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import registry as RG

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

check("Mister = Mr.", RG.person_key("Mister Orrin Vale") == RG.person_key("Mr. Orrin Vale") == RG.person_key("mr_orrin_vale"))
check("Missus = Mrs.", RG.person_key("Missus Vale") == RG.person_key("Mrs. Vale"))
check("Mr. and Mrs. of one surname stay two people", RG.person_key("Mister Vale") != RG.person_key("Missus Vale"))
check("a spoken title is not a word of the name", RG._words("Mister Orrin Vale") == ["Orrin", "Vale"]
      and RG._words("Missus Vale") == ["Vale"], RG._words("Mister Orrin Vale"))
check("Dr. = Doctor still", RG.person_key("Dr. Marsh") == RG.person_key("doctor_marsh"))

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
