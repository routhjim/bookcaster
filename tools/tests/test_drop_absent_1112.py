"""11-12: with supplied keys, words for anyone the key does not show are left out of a clip's words (not reworded to
"someone out of the picture"); off, the words are as before. Synthetic descriptions only.
   python3 tools/tests/test_drop_absent_1112.py"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got!r}"))
    ok or fails.append(name)

gone = ["the man in the grey overcoat"]
t = "The woman in the blue dress sits by the fire. The man in the grey overcoat watches her from the door."
check("a sentence led by someone absent goes whole",
      S.drop_absent(t, gone) == "The woman in the blue dress sits by the fire.", S.drop_absent(t, gone))
t = "The woman in the blue dress turns her head, looking toward the man in the grey overcoat."
check("a later clause naming someone absent goes alone",
      S.drop_absent(t, gone) == "The woman in the blue dress turns her head.", S.drop_absent(t, gone))
t = "The woman in the blue dress stands up as the man in the grey overcoat enters."
check("an 'as ...' clause goes alone", S.drop_absent(t, gone) == "The woman in the blue dress stands up.",
      S.drop_absent(t, gone))
t = "The woman in the blue dress smiles at the two men."
check("a group word in the lead clause takes the sentence", S.drop_absent(t, gone) == "", S.drop_absent(t, gone))
t = "The woman in the blue dress sits still. Only the one person already in the picture is ever in it."
check("nothing absent: unchanged", S.drop_absent(t, gone) == t, S.drop_absent(t, gone))
check("never the reworded phrase", "out of the picture" not in S.drop_absent(
    "The woman in the blue dress waves, and the man in the grey overcoat nods.", gone))
check("off by default (the flow's words are as before)", S.DROP_ABSENT is False)
check("a chain line keeps its words when its shot is not known", S._chain_words({"id": "x"}, "A line.").startswith("A line."))

t = "the man in the grey overcoat in the centre and the man in the tweed jacket on the left stand near the desk, looking toward the window."
got = S.drop_absent(t, gone)
check("an absent person in a joint subject goes alone; the present one keeps his words",
      got == "the man in the tweed jacket on the left stands near the desk, looking toward the window.", got)
t = "The woman in the blue dress and the man in the grey overcoat sit by the fire."
got = S.drop_absent(t, gone)
check("... in second place too", got == "The woman in the blue dress sits by the fire.", got)
t = "The woman in the blue dress turns her head, and the man in the grey overcoat nods."
got = S.drop_absent(t, gone)
check("a later clause about the absent one goes alone", got == "The woman in the blue dress turns her head.", got)

print({"fails": fails})
sys.exit(1 if fails else 0)
