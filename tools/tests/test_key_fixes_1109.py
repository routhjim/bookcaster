"""E33 (knight ch.1 redo, keys): whole garments in an action key (the helm), the judge asks the clothes the key draws,
a facing question and a things question the judge can select on, the fallback seed by fewest misses, a thing's look
once and never inside a compound ("river bank"), and the general rules in the director, opening and chain asks.
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

# --- 1: facing, asked where the picture holds the target
check("facing a thing named in the picture or room", "toward the water" in q.get("face0", ""), q)
two = {"id": "action-05m11s-003114_s0", "kind": "action", "framing": "medium", "who": ["Edric", "Vorgrath"],
       "placed": [{"name": "Edric", "pose": "stand", "facing": "right"}, {"name": "Vorgrath", "pose": "stand", "facing": "left"}],
       "picture": "Edric stands on the left, facing right; Vorgrath stands on the right, facing left.", "keyshot": "x"}
q2, _ = S._key_questions(cx, two)
check("a bare side with one other there: turned toward them", "turned toward a massive dragon" in q2.get("face0", ""), q2)
check("... and the other way", "turned toward a man" in q2.get("face1", ""), q2)
named = dict(two, placed=[{"name": "Edric", "pose": "stand", "facing": "toward Vorgrath"}, {"name": "Vorgrath", "pose": "stand", "facing": ""}])
q3, _ = S._key_questions(cx, named)
check("facing by name", "massive dragon" in q3.get("face0", "") and "face1" not in q3, q3)
none_ = [dict(two, kind="talk"), dict(two, placed=[{"name": "Edric", "pose": "lie", "facing": "right"}]),
         dict(row, placed=[{"name": "Edric", "pose": "stand", "facing": "toward the viewer"}]),
         dict(row, placed=[{"name": "Edric", "pose": "stand", "facing": "toward the tower"}])]
got = [any(k.startswith("face") for k in S._key_questions(cx, x)[0]) for x in none_]
check("no facing question: talk, lying, the viewer, a thing not in the picture", got == [False] * 4, got)

# --- 2: the action's things are asked; not for quiet shots
check("an action key asks its setting's things", "The river flows left" in q.get("obj", ""), q)
check("a quiet key does not", "obj" not in S._key_questions(cx, dict(row, kind="quiet", keyshot=""))[0])

# --- the new answers gate a seed; the fallback is the nearest count, then the fewest misses
with tempfile.TemporaryDirectory() as d:
    k = Path(d) / "k.png"
    from PIL import Image; Image.new("RGB", (64, 36), (120, 100, 80)).save(k)
    good = {"text": "no", "who0": "yes", "face0": "yes", "obj": "yes"}
    check("a clean seed passes", S._clean(good, k, 0))
    check("turned to the viewer fails", not S._clean(dict(good, face0="no"), k, 0))
    check("a missing thing fails", not S._clean(dict(good, obj="no"), k, 0))
check("misses count each failed answer", (S._misses(dict(good, face0="no", obj="no")), S._misses(good)) == (2, 0))
src = Path(S.__file__).read_text()
check("keys_judged's fallback ranks by (count, misses)", "key=lambda x: (off(x), _misses(x[1]))" in src)

# --- a thing's look once, never inside a compound
p = "Edric stands at the river bank, facing left. The river flows left."
g = S.gloss_once(p, "river", "cold water")
check("gloss after the noun of its own, once", g.count("cold water") == 1 and "The river (cold water) flows" in g, g)
g = S.gloss_once("Close on the sword blade.", "sword", "a blade")
check("gloss after a compound with no other place: the whole phrase, named", g == "Close on the sword blade (sword: a blade).", g)
check("gloss before a preposition", S.gloss_once("the dark metal on the anvil.", "dark metal", "X") == "the dark metal (X) on the anvil.")

# --- judgment rules: general words only, in the existing asks
check("director: the thing's state now", "not its usual look" in S.SHOT_RULES)
check("director: within reach, not apart across the picture", "never apart across it" in S.SHOT_RULES)
check("director: how far a thing goes in", "how far it goes in" in S.SHOT_RULES)
check("director: no repeated picture", "never the same picture twice" in S.SHOT_RULES)
check("opening: within one move", "within one move of it" in S.OPENING_ASK)
check("chain: a thing's look before any line changes it", "before any line changes it" in S.CHAIN_ASK)
check("chain: head mended when too far apart", "too far apart" in S.CHAIN_ASK)
story = re.compile(r"\b(dragon|sword|anvil|hammer|forge|river|quench|knight|armou?r|helm|Edric|Hulda|Vorgrath)\b", re.I)
new = [x for x in (S.OPENING_ASK, S.CHAIN_ASK) if story.search(x)]
check("no story content in the opening and chain asks", not new, [story.search(x).group(0) for x in new])
print({"fails": fails})
sys.exit(1 if fails else 0)
