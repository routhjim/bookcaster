"""11-24: a side drawing is a profile only when head AND body are seen from the side; a side that is no profile and
has no stand-in is not painted; the redraw loop measures."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import profile_check as PC

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

def kps(shoulder_gap):          # shoulders at y 100, hips at y 300
    k = [[0, 0]] * 13
    k[5], k[6], k[11], k[12] = [200 - shoulder_gap / 2, 100], [200 + shoulder_gap / 2, 100], [190, 300], [210, 300]
    return k

check("front-facing body measures wide", PC.body_ratio(kps(140)) == 0.7)
check("side-on body measures narrow", PC.body_ratio(kps(60)) == 0.3)
check("no trunk height: not measured", PC.body_ratio([[0, 0]] * 13) is None)
check("head and body in profile", PC.is_profile(0.1, 0.3))
check("head in profile on a front-facing body is no profile", not PC.is_profile(0.1, 0.6))
check("a turned-away head on a side-on body is no profile", not PC.is_profile(0.3, 0.3))
check("an unmeasured body leaves the head's verdict", PC.is_profile(0.1, None) and not PC.is_profile(None, 0.3))
src = (HERE.parent / "characters.py").read_text()
check("the redraw loop measures the drawings", "meas = measure_sides(" in src)
check("no stand-in: the side is not painted", '"dropped": True' in src and "kept, flagged" not in src)

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
