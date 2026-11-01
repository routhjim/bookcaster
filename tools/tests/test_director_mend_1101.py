"""The extra last step after the director's retries: one sentence asked again (11-01)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import shot_flow as S

CAST = ["Sherlock Holmes", "Helen Stoner"]
NM = lambda t, c: [w for w in c if w.split()[0] in str(t)]   # noqa: E731
REP = json.dumps({"shots": [{"kind": "quiet", "action": "He leans in. He stares into the crackling fire.",
                             "setting": "Close up on Helen Stoner's white wrist. A lamp."}]})
F = ["shot 1: 'fire' does not say where it comes from",
     "shot 1's setting names Helen Stoner: where people are, their posture and clothes go only in \"people\""]


def test_both_mended():
    ans = iter(["from the fireplace.", "Close up on a woman's white wrist."])
    s = json.loads(S._mend(REP, F, CAST, lambda q: next(ans), NM))["shots"][0]
    assert "fire from the fireplace" in s["action"] and "Helen" not in s["setting"] and "wrist" in s["setting"]


def test_bad_answer_changes_nothing():
    assert S._mend(REP, F[:1], CAST, lambda q: "I don't know", NM) is None
    assert S._mend(REP, F[1:], CAST, lambda q: "Helen Stoner's wrist.", NM) is None
