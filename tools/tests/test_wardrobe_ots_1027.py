"""10-27 keys: Watson in bed in his tweed jacket (no nightclothes: the wardrobe asked only for LASTING changes, in one
story's armor examples, and was keyed on its prompt alone) and the back of Helen's head drawn as a stranger in green
(an over-the-shoulder listener had no hair or clothes in the key prompt)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE))
from tts_reader import video
import shot_flow as SF, mannequin_flow as MF
import fakebook

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

check("the wardrobe asks for night clothes, coats, disguises and getting dressed again; no armor/forge examples",
      "night clothes" in video.WARDROBE_PROMPT and "Getting dressed again" in video.WARDROBE_PROMPT
      and "armor" not in video.WARDROBE_PROMPT.lower() and "smith" not in video.WARDROBE_PROMPT.lower())
p1 = {"look": {"characters": {"Watson": "a man of about 35, wearing a tweed jacket"}}}
p2 = {"look": {"characters": {"Watson": "a man of about 35, wearing a tweed jacket and a cap"}}}
k1 = video.wardrobe_inputs_key(p1, "chapter one")
check("the wardrobe is planned again when a look or the chapter changes",
      k1 != video.wardrobe_inputs_key(p2, "chapter one") and k1 != video.wardrobe_inputs_key(p1, "chapter two")
      and k1 == video.wardrobe_inputs_key(p1, "chapter one"))
cx = fakebook.book({"Helen Stoner": "a woman of about 30, dark hair with a few grey streaks, wearing a black mourning dress",
                    "Sherlock Holmes": "a man of about 40, black hair, wearing a dark frock coat"}, loras="all")
(cx.book / "work" / "video" / "x").mkdir(parents=True, exist_ok=True)
(cx.book / "work" / "video" / "x" / "plan.json").write_text(json.dumps({"look": {"characters": {}}}))
MF._dress = lambda *a, **k: {}
p = SF._key_prompt(cx.book, "x", "talk-001595", {"id": "talk-001595_s0", "who": ["Sherlock Holmes", "Helen Stoner"],
                                                 "ots": "Helen Stoner", "picture": "over-the-shoulder", "framing": "medium",
                                                 "place": "Baker Street Sitting Room"})
seg = p[p.index("Helen Stoner, seen from behind"):][:160]
check("the listener seen from behind has her hair and clothes", "dark hair" in seg and "black mourning dress" in seg, seg)
seen = {}
_chat0 = video._chat
video._chat = lambda url, model, q, max_tokens=0: (seen.setdefault("q", q), "{}")[1]
video.plan_wardrobe(["I woke.", "I dressed."], {"Dr. Watson": "a man", "Sherlock Holmes": "a man"}, "u", "m", narrator="Dr. Watson")
video._chat = _chat0
check("the wardrobe is told the narrator is the 'I' of the chapter (10-27: Watson got no night clothes)",
      '- Dr. Watson: a man (the narrator: the "I" of the chapter is this person)' in seen["q"].splitlines(), seen.get("q", "")[:300])
check("a narrator change re-plans the wardrobe", video.wardrobe_inputs_key(p1, "c", "Watson") != video.wardrobe_inputs_key(p1, "c", ""))
seen.clear()
video._chat = lambda url, model, q, max_tokens=0: (seen.setdefault("q", q), "{}")[1]
video.plan_wardrobe(["I woke."], {"Dr. Watson": "a man"}, "u", "m", narrator="Dr. Watson",
                    notes="[from paragraph 0] at the bedroom: Watson in a nightshirt blinks up from the sheets")
video._chat = _chat0
check("the wardrobe reads the director's scene-opening notes as evidence (10-27: 'Watson in a nightshirt')",
      "Watson in a nightshirt" in seen["q"] and seen["q"].index("director's notes") < seen["q"].index("People:"))
check("new scene notes re-plan the wardrobe", video.wardrobe_inputs_key(p1, "c", "", "a") != video.wardrobe_inputs_key(p1, "c", "", "b"))
# 10-28: a re-asked breakdown (new scene notes) does not re-plan the wardrobe; one keyed with notes is kept, re-keyed
from tts_reader import video as V2
pl = {"look": {"characters": {"Peter": "a man, wearing a coat"}}, "wardrobe": {"Peter": [{"from": 0, "id": "coat", "dress": "a coat"}]}}
pl["wardrobe_k"] = V2.wardrobe_inputs_key(pl, "chapter one", "", "notes A")
asked = []
V2.plan_wardrobe = lambda *a, **k: asked.append(1) or {}
got = V2.ensure_wardrobe(pl, "chapter one", "", "", log=lambda *a: None, notes="notes A")
check("10-28: a wardrobe keyed with the notes is kept and re-keyed, not planned again",
      got and not asked and pl["wardrobe_k"] == V2.wardrobe_inputs_key(pl, "chapter one") and "Peter" in pl["wardrobe"])
got = V2.ensure_wardrobe(pl, "chapter one", "", "", log=lambda *a: None, notes="notes B (a re-asked breakdown)")
check("10-28: new scene notes alone don't re-plan it", not got and not asked)
V2.ensure_wardrobe(pl, "chapter one CHANGED", "", "", log=lambda *a: None, notes="notes B")
check("10-28: a changed chapter does", asked == [1])

print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
