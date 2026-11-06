"""11-05 user: "Fix and start over". (1) A wife ("said Sena, his wife"; a female voice) got "a man of about 32": a
look's sex is checked against the voice the cast chose and the text's pronouns and role words (the language model
judges, faked here); a disagreement is asked again, naming it. (2) A non-person's failed LoRA is recorded with its
inputs and not tried again until they change; a cast member the plan lists as a prop never gets one. (3) The asks
carry no story examples. A SYNTHETIC book, CPU only, every service faked (no language model, card or systemctl).
    ~/tts_reader/.venv/bin/python3 -I tools/tests/test_looks_gender_lora_1105.py"""
import json
import re
import sys
import tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import flow3d as F                       # noqa: E402
import mannequin_flow as MF              # noqa: E402
import shot_flow as S                    # noqa: E402
from tts_reader import score as SC       # noqa: E402
from tts_reader import video as V        # noqa: E402
fails = []


def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)


check("look_sex", F.look_sex("a man of about 32, clean-shaven") == "man" and
      F.look_sex("a woman of about 30, brown hair") == "woman" and F.look_sex("a young girl of about 7") == "woman"
      and F.look_sex("a great black dragon") is None)

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = F.vdir(book)
    vd.mkdir(parents=True)
    wrong = "a man of about 32, thin and pale, shoulder-length brown hair tied back, his face drawn, wearing a grey tunic"
    looks = {"Daven Iru": "a man of about 35, clean-shaven, black hair, wearing a grey flight suit", "Sena": wrong}
    (vd / "look.json").write_text(json.dumps({"characters": looks, "places": {}}))
    (book / "book.cast.json").write_text(json.dumps({"characters": {
        "Daven": {"gender": "M", "role": "a weary pilot"},
        "Sena": {"gender": "F", "role": "Daven's exhausted wife"}}}))
    RG = F._reg(book)
    check("voice_cast: the cast's gender and role for the person", F.voice_cast(book, RG, "Sena") ==
          {"sex": "woman", "role": "Daven's exhausted wife"}, F.voice_cast(book, RG, "Sena"))
    text = ("Daven Iru sat at the console.\n\n\"Come to bed,\" said Sena, his wife. She pulled the curtain back.\n\n"
            "Sena watched him from the bunk, her hands folded.")
    plan = {"look": {"characters": dict(looks)}, "wardrobe": {}}
    asked = []
    answers = iter(['{"ok": true}',                                       # wrong: keeps the man
                    '{"ok": false, "why": "she is his wife", "look": "a woman of about 32, thin and pale, '
                    'shoulder-length brown hair tied back, wearing a grey tunic"}'])

    def chat(url, model, prompt, max_tokens=0):
        asked.append(prompt)
        return next(answers) if "Sena" in prompt.split("\n")[0] else '{"ok": true}'
    SC._chat = chat
    fixed = F.audit_looks(book, text, plan, log=lambda m: None)
    sena = [q for q in asked if q.split("\n")[0].startswith("A story character, Sena")]
    check("Sena: the look/voice conflict is named in the ask", sena and "CONFLICT" in sena[0] and "female voice" in sena[0],
          sena[:1])
    check("Sena: the ask asks man or woman by the pronouns and role words", sena and "by the pronouns" in sena[0])
    check("Sena: a wrong 'ok' is asked again, naming the conflict", len(sena) == 2 and "Check again" in sena[1], len(sena))
    check("Sena: the look corrected to a woman (plan and look.json)",
          F.look_sex(plan["look"]["characters"]["Sena"]) == "woman" and "Sena" in fixed
          and F.look_sex(json.loads((vd / "look.json").read_text())["characters"]["Sena"]) == "woman", fixed)
    asked.clear()
    F.audit_looks(book, text, plan, log=lambda m: None)
    check("no conflict left, audited before: not asked again", not [q for q in asked if "Sena" in q.split("\n")[0]])

    # (2) LoRAs: a craft's failure is recorded with its inputs and not retried until its look changes
    looks2 = {"Ada": "a woman of about 30, short dark hair", "Kite": "a sleek grey single-seat fighter craft"}
    (vd / "look.json").write_text(json.dumps({"characters": looks2, "places": {}}))
    calls = []
    F.run = lambda cmd, *a, **k: (calls.append(list(map(str, cmd))), 1)[1]       # every build fails
    F.ensure_loras(book, log=lambda m: None)
    fj = F._failed_path(book, "Kite")
    check("a craft's failed LoRA is recorded with its input hash",
          fj.exists() and json.loads(fj.read_text())["hash"] == F._input_hash(book, "Kite"))
    check("a person's failure is not recorded (it stops the run instead)", not F._failed_path(book, "Ada").exists())
    calls.clear()
    F.ensure_loras(book, log=lambda m: None)
    check("not tried again on the same inputs; the person is", not any("Kite" in c for c in calls)
          and any("Ada" in c for c in calls), calls)
    check("listed while it waits (the status says why)", F.lora_todo(book).get("Kite") == F.FAILED_BEFORE)
    looks2["Kite"] = "a sleek grey single-seat fighter craft with red stripes"
    (vd / "look.json").write_text(json.dumps({"characters": looks2, "places": {}}))
    check("its look changed: tried again", F.lora_todo(book).get("Kite") == "no LoRA yet", F.lora_todo(book))
    (vd / "look.json").write_text(json.dumps({"characters": looks2, "places": {}, "props": {"Kite": "a craft"}}))
    fj.unlink()
    check("listed by the plan as a prop: never a LoRA", "Kite" not in F.lora_todo(book), F.lora_todo(book))

# (3) the asks: no story examples
V._chat = lambda url, model, prompt, max_tokens=0: (asked.append(prompt), '{"Ann": "a woman of about 30"}')[1]
asked.clear()
V.check_looks({"Ann": "a woman"}, "u", "m")
consts = " ".join(c for c in V.plan_pictures.__code__.co_consts if isinstance(c, str))
src_f = (HERE / "flow3d.py").read_text()
period = src_f[src_f.index("def infer_period"):src_f.index("def infer_period") + 1500]
names = r"e\.g\.|Victorian|Holmes|Watson|Stoner|fireplace|dragon's mouth|steel plate|drove the sword|black hair swept"
for nm, txt in (("check_looks", asked[0]), ("the look ask", consts), ("the period ask", period),
                ("BREAKDOWN_ASK", S.BREAKDOWN_ASK), ("the 3D shot rules", MF.__dict__.get("FIGHT_SHOT_ASK", "") or
                                                     (HERE / "mannequin_flow.py").read_text()[(HERE / "mannequin_flow.py").read_text().index("Write:\n- \"left\""):][:2500])):
    check(f"{nm}: no story example", not re.search(names, txt), re.findall(names, txt))
check("check_looks keeps each one's sex", "keep each one's sex exactly as given" in asked[0])
print({"fails": fails})
sys.exit(1 if fails else 0)
