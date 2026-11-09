"""E33 (knight ch.1 redo, keys): whole garments in an action key (the helm), the judge asks the clothes the key draws,
the fallback seed by fewest misses, a thing's look once and never inside a compound ("river bank"), the chain ask's
looks before any change (the end state), and no story names in the director's rules. (The runner's facing and
things judge questions were dropped: no seed faced, 81 false yes; things changed no kept key.)
CPU only.     ~/tts_reader/.venv/bin/python tools/tests/test_key_fixes_1109.py"""
import re, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE))
import shot_flow as S
import fakebook

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

# --- 5: whole garments (the helm), the under-layer out
ARM = ("Dull, unpolished star-iron armour consisting of a breastplate curved to the chest, a helm with a narrow eye "
       "slit, and thin plates covering the arms and legs, worn over his usual brown leather jerkin and trousers")
w3 = S.worn_outer(ARM)
check("worn_outer keeps the adjective with its garment and the helm", w3.startswith("Dull, unpolished") and "helm" in w3
      and "plates" in w3, w3)
check("worn_outer leaves out what is worn under", "jerkin" not in w3, w3)
check("worn_outer(2) for the judge: breastplate + helm", "helm" in S.worn_outer(ARM, 2) and "plates" not in S.worn_outer(ARM, 2))
check("a plain outfit is unchanged", S.worn_outer("a worn brown leather jerkin, trousers") == "a worn brown leather jerkin, trousers")
check("describe()'s garments() is untouched (the video model's names)", S.garments(ARM).startswith("Dull, unpolished"))

# --- 6: the judge's wardrobe is the key's (MF._dress, the made-outfit shift), cx.dress only without plan files
cx = fakebook.book({"Edric": "a man of about 18, short dark hair, wearing a worn brown leather jerkin, trousers",
                    "Hulda": "a woman of about 70, grey hair, wearing a heavy wool dress",
                    "Vorgrath": "a massive dragon with scales like black iron, a long tail"},
                   dress=lambda w, t: ARM if w == "Edric" else "")
row = {"id": "action-03m50s-002300_s0", "kind": "action", "framing": "medium", "who": ["Edric"],
       "placed": [{"name": "Edric", "pose": "stand", "at": "at the river bank", "facing": "left toward the water"}],
       "picture": "Edric stands at the river bank, facing left toward the water, holding a glowing sword. The river flows left.",
       "setting": "The river flows left.", "keyshot": "x", "room": "a rocky riverbank with cold, clear water"}
real = S.MF._dress
try:
    S.MF._dress = lambda *a, **k: {}            # the plan's wardrobe at this shot: the usual clothes (made, not worn yet)
    q, _ = S._key_questions(cx, row)
    check("judge asks the jerkin the key draws, not the raw-time armour", "jerkin" in q.get("who0w", "")
          and "armour" not in q.get("who0w", ""), q.get("who0w"))
    S.MF._dress = lambda b, st, mid, who: {"Edric": ARM}
    q, _ = S._key_questions(cx, row)
    check("judge asks the armour with its helm when the key draws it", "helm" in q.get("who0w", ""), q.get("who0w"))
finally:
    S.MF._dress = real
q, _ = S._key_questions(cx, row)            # (fakebook has no plan files: cx.dress)
check("no plan files: cx.dress as before", "helm" in q.get("who0w", ""), q.get("who0w"))

# --- no seed clean: the nearest head count, then the fewest misses
with tempfile.TemporaryDirectory() as d:
    k = Path(d) / "k.png"
    from PIL import Image; Image.new("RGB", (64, 36), (120, 100, 80)).save(k)
    good = {"text": "no", "who0": "yes", "who0w": "yes"}
    check("a clean seed passes", S._clean(good, k, 0))
    check("a wrong outfit fails", not S._clean(dict(good, who0w="no"), k, 0))
check("misses count each failed answer", (S._misses(dict(good, who0="no", who0w="no", text="yes")), S._misses(good)) == (3, 0))
check("the runner-dropped questions are gone (facing, things)", not any(k.startswith(("face", "obj")) for k in q), q)
src = Path(S.__file__).read_text()
check("keys_judged's fallback ranks by (count, misses)", "key=lambda x: (off(x), _misses(x[1]))" in src)

# --- a thing's look once, never inside a compound
p = "Edric stands at the river bank, facing left. The river flows left."
g = S.gloss_once(p, "river", "cold water")
check("gloss after the noun of its own, once", g.count("cold water") == 1 and "The river (cold water) flows" in g, g)
g = S.gloss_once("Close on the sword blade.", "sword", "a blade")
check("gloss after a compound with no other place: the whole phrase, named", g == "Close on the sword blade (sword: a blade).", g)
check("gloss before a preposition", S.gloss_once("the dark metal on the anvil.", "dark metal", "X") == "the dark metal (X) on the anvil.")

# --- kept rules: general words only
check("chain: a thing's look before any line changes it", "before any line changes it" in S.CHAIN_ASK)
check("chain: the mover's end state too", "only the one it reaches changes" in S.CHAIN_ASK)
story = re.compile(r"\b(dragon|sword|anvil|hammer|forge|river|quench|knight|armou?r|helm|Edric|Hulda|Vorgrath)\b", re.I)
new = [x for x in (S.OPENING_ASK, S.CHAIN_ASK) if story.search(x)]
check("no story content in the opening and chain asks", not new, [story.search(x).group(0) for x in new])
mom = re.search(r'- ACTION SHOTS also give "moment".*?(?=\n- )', S.SHOT_RULES, re.S).group(0)
check("the director's moment rule is general: no story names", not re.search(story.pattern + r"|\b(Roylott|poker)\b",
      mom, re.I), mom)
check("no knight-story names anywhere in the director rules", not story.search(S.SHOT_RULES),
      story.search(S.SHOT_RULES) and story.search(S.SHOT_RULES).group(0))
print({"fails": fails})
sys.exit(1 if fails else 0)
