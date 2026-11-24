"""11-24: the judge's "is there X in this picture?" names what a person is even when their look opens with a list
of adjectives; a close-up at a place with a crowd is not failed on its head count."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import shot_flow as SF

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

w = SF.who_phrase
check("a look that opens with adjectives is read on to the noun",
      w("a short, broad old woman of 70 with a red scarf, wearing a grey shawl") == "a short, broad old woman",
      w("a short, broad old woman of 70 with a red scarf, wearing a grey shawl"))
check("a plain look is cut as before", w("a man of about 40, black hair, wearing a green coat") == "a man",
      w("a man of about 40, black hair, wearing a green coat"))
check("an age without 'about' stays where the noun came first", w("a boy of 9, thin") == "a boy of 9", w("a boy of 9, thin"))
check("a creature keeps its words", w("a huge grey wolf with one torn ear") == "a huge grey wolf",
      w("a huge grey wolf with one torn ear"))
check("no noun at all: the first part, as before", w("a ferryman, old and bent") == "a ferryman", w("a ferryman, old and bent"))
check("an empty look gives nothing", w("") == "")

src = (HERE.parent / "shot_flow.py").read_text()
check("key questions use it", "first = who_phrase(lk)" in src)
check("a close-up before a crowd is not counted", 'if s.get("crowd_behind") and closeup(s):\n            need = 0' in src)
check("every shot knows its place has a crowd", 'crowd_behind=bool(sh.get("crowd"))' in src)

check("a talk clip before a crowd is not counted either",
      '"people": None if s.get("chorus") or s.get("crowd") else' in src)
check("nor a chain", '"people": None if s.get("crowd") else len(s.get("who") or []) or None,' in src)

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
