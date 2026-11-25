"""A still in a painted set is lit from the set's picture, not at one fixed daylight level (11-25)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import blockout as B

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)

dark = B.light_for([0.40, 0.28, 0.12])          # a warm, dim room
day = B.light_for([0.62, 0.64, 0.66])           # a light room
pit = B.light_for([0.03, 0.03, 0.04])
check("the set is shown as painted", dark["exposure"] == 0.0 and day["exposure"] == 0.0)
check("a dim room dims its people", dark["key_light"] < day["key_light"] and dark["fill_light"] < day["fill_light"])
check("a light room keeps the tuned level", abs(day["key_light"] - 7.2 * B.LIGHT_BOOST) < 0.05)
check("the darkest room still lights faces", abs(pit["key_light"] - 7.2 * B.LIGHT_BOOST * B.SET_GAIN_MIN) < 0.05)
check("a warm room warms the light", dark["light_color"][0] > dark["light_color"][2] and dark["light_color"][0] == 1.0)
check("a grey room leaves it near white", max(day["light_color"]) - min(day["light_color"]) < 0.05)
check("no picture: the lights as before", B.set_light({"dir": "/nonexistent"}) == {})
src = (Path(__file__).resolve().parents[1] / "blockout_scene.py").read_text()
check("the scene takes the colour", src.count("lc_") >= 4)
print("all passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
