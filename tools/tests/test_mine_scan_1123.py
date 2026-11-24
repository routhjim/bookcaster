"""11-23: the miner tells narration from character speech by asking (the transcript has no quotation marks), and an
unreadable answer fails the cell instead of counting as "no speech"; whisper gets samples, not a path."""
import inspect, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mine_voice as M

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

segs = [{"i": i, "start": i, "end": i + 1, "text": f"Sentence {i}."} for i in range(6)]
asked = {}
def chat(prompt, max_tokens=700):
    asked["max_tokens"] = max_tokens
    return asked["reply"]
M.chat = chat
asked["reply"] = "Here: [1, 2, 5]"
check("the segments where a character speaks are read from the reply", M.speech_segs(segs) == {1, 2, 5})
check("the reply's room grows with the chapter", asked["max_tokens"] >= 400 + 6 * len(segs), asked)
asked["reply"] = "[]"
check("an empty list means no speech", M.speech_segs(segs) == set())
asked["reply"] = "I cannot tell"
try:
    M.speech_segs(segs)
    check("an unreadable reply fails the cell", False)
except RuntimeError:
    check("an unreadable reply fails the cell", True)
src = inspect.getsource(M.mine_cell)
check("the neutral cell tries every sentence start and keeps the liveliest", 'emotion == "neutral"' in src
      and "rank_survivors(survivors)" in src)
check("whisper is handed samples, not a file path", "transcribe(str(wav)" not in inspect.getsource(M.transcribe))

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
