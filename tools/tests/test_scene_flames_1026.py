"""10-26 Acts 2 v3 review (user): "There are flames over people too much of the time". The scene sheet's states had
"Divided Tongues" from the first moment of the Upper Room and of the street, so every key's "As it is now" put flames on
every head through the sermon and the baptisms. A passing effect in a state now shows only in moments whose own
sentences have it; a place's own fire (a forge, a hearth) stays. CPU only.
    ~/tts_reader/.venv/bin/python3 tools/tests/test_scene_flames_1026.py"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S
tl = json.loads((Path(__file__).parent / "fixtures" / "fire_1026" / "timeline.json").read_text())
fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

TONGUES = ("Divided Tongues: Small, flame-like shapes of light resting above the heads of the believers, bright "
           "white-yellow core with soft orange edges, no smoke, no fuel source")
moments = [("action-000000", 0.0, 5.0), ("action-000050", 5.0, 5.0), ("action-000100", 10.0, 5.0),
           ("action-000150", 15.0, 6.3), ("action-000213", 21.3, 5.0), ("action-000313", 31.3, 5.0),
           ("action-000413", 41.3, 5.0)]
shown = [m for m, t, d in moments if S.passing_shown(TONGUES, " ".join(z for _, z in S.timed(tl, t, d)))]
check("flames only in the moment of 'divided tongues as of fire appeared'", shown == ["action-000100"], shown)
check("'speak in other tongues' is not fire", not S.passing_shown(TONGUES, "and began to speak in other tongues as the "
      "Spirit gave them utterance."))
check("a forge fire stays lit without being said", S.passing_shown("Forge Fire: the forge glowing orange, coals bright",
      "He took up the hammer."))
check("a hearth fire stays lit", S.passing_shown("Fire: a low fire burning in the hearth", "She sat down."))
check("a work piece state is untouched", S.passing_shown("Bar: a dark bar with a crack across its face", "He looked."))
check("a door state is untouched", S.passing_shown("Door: the door stands open", "Nobody spoke."))
check("wind only where it is said", not S.passing_shown("Wind: a strong wind whipping the tunics", "They sat in silence.")
      and S.passing_shown("Wind: a strong wind whipping the tunics", "a sound like a mighty rushing wind"))
check("the scene prompt rules out passing effects as states", "Never a passing effect" in S.SCENE_ASK)
print(f"\n{len(fails)} failed" if fails else "\nall passed")
sys.exit(1 if fails else 0)
