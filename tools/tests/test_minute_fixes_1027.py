"""10-27 evening fixes: talk shots end with their line (a reaction for the rest), one beat per ~2 s of action, a
one-person talk key is a close-up (asked), a two-shot with someone seated/lying keeps only sides and faces, no 'his
usual' in descriptions, outfit-sheet portraits out of training, outfits drawn in turn, helmets are props, --render."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent)); sys.path.insert(0, str(HERE))
import shot_flow as S, charlora as L, fakebook, inspect

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

f = S._faults({"shots": [{"who": [], "picture": "p", "dur": 3.0, "action": "He turns away from her. He walks to his desk. "
                          "He unlocks a small drawer. He pulls out a thin case-book."}]}, [], 3.0)
check("four beats in 3 s are sent back (about one per 2 s)", any("actions in 3.0 s" in x for x in f), f)
f = S._faults({"shots": [{"who": [], "picture": "p", "dur": 3.0, "action": "He unlocks the drawer and takes out the book."}]}, [], 3.0)
check("one beat in 3 s is fine", not any("actions in" in x for x in f), f)
check("a one-person talk picture is a close-up", "close-up" in S.TALK_PICTURE and "no one else" in S.TALK_PICTURE)
check("one two-shot template (the seated variant broke the lip sync, 10-27)", not hasattr(S, "TALK_TWO_HELD")
      and "close to the viewer" in S.TALK_TWO)
cx = fakebook.book({"Doctor Watson": "a man of about 35, wearing a tweed jacket"}, loras="all")
q, _ = S._key_questions(cx, {"id": "talk-000964_s0", "who": ["Doctor Watson"], "kind": "talk", "picture": "", "framing": "close"})
check("a one-person talk key is asked whether it is a close-up", "close" in q, q)
R = S.RG.Book(cx.book)
check("no 'his usual' in a description", S.describe(R, "Doctor Watson", {"Doctor Watson": "His usual tweed shooting jacket, white shirt"})
      == "the man in the tweed shooting jacket", S.describe(R, "Doctor Watson", {"Doctor Watson": "His usual tweed shooting jacket"}))
check("a talk shot ends with its line; the rest is a reaction (the code is there)",
      '"kind": "reaction", "of": x0.get("listener") or x0.get("speaker")' in inspect.getsource(S.direct))
check("outfit-sheet portraits never train (only the full figure)", 'if oid is not None and part == "head":' in inspect.getsource(L.cand_sheet)
      if hasattr(L, "cand_sheet") else 'if oid is not None and part == "head":' in Path(L.__file__).read_text())
check("pose drawings take the outfits in turn", "list(c.outfits)[k % len(c.outfits)]" in inspect.getsource(L.gen_jobs_pose))
check("helmets are props", "helm" in L.PROPS)
check("run() takes a render window", "render" in inspect.signature(S.run).parameters)
(cx.book / "work" / "video" / "x").mkdir(parents=True, exist_ok=True)
(cx.book / "work" / "video" / "x" / "plan.json").write_text(json.dumps({"look": {"characters": {}}}))
S.MF._dress = lambda *a, **k: {}
kp = S._key_prompt(cx.book, "x", "talk-000996", {"id": "talk-000996_s0", "who": ["Doctor Watson"], "kind": "talk",
                   "framing": "close", "place": "Baker Street Bedroom", "pose_held": {"Doctor Watson": "stand"},
                   "picture": S.TALK_PICTURE.format(who="Doctor Watson")})
check("a one-person talk key: no posture, the room only soft behind the face (10-27: medium shots by an empty bed)",
      ", standing" not in kp and "fills the whole picture" not in kp and "softly out of focus" in kp, kp)
check("a one-person talk key gets no wide room reference", S.closeup({"kind": "talk", "who": ["A"]})
      and not S.closeup({"kind": "talk", "who": ["A", "B"]}) and "hero = None if closeup(s)" in Path(S.__file__).read_text())
check("10-28: a standing close-up says so through the background (no chair behind the shoulders)",
      "no chair back" in kp and "Seated" not in kp, kp)
kp2 = S._key_prompt(cx.book, "x", "talk-000996", {"id": "talk-000996_s0", "who": ["Doctor Watson"], "kind": "talk",
                    "framing": "close", "place": "Baker Street Bedroom", "pose_held": {"Doctor Watson": "sit"},
                    "picture": S.TALK_PICTURE.format(who="Doctor Watson")})
check("10-28: a seated close-up shows the chair back", "top of a chair back" in kp2 and "no chair back" not in kp2, kp2)
q2, _ = S._key_questions(cx, {"id": "talk-000964_s0", "who": ["Doctor Watson"], "kind": "talk", "picture": "",
                              "framing": "close", "pose_held": {"Doctor Watson": "stand"}})
check("10-28: a standing close-up is checked for a chair behind him", any("standing, with no chair" in v
      for k_, v in q2.items() if k_.startswith("pose")), q2)
for t_, n_ in (("The man in the dark frock coat turns away toward the desk, unlocks the drawer, and takes out a small "
                "case-book.", 3), ("The man in the dark frock coat, with his hat in his hands, stands by the door.", 1),
               ("The woman in the black mourning dress sits still, hands in her lap.", 1),
               ("He unlocks a small drawer. He pulls out a thin case-book.", 2)):
    check(f"10-28: {n_} action(s) in {t_[:40]!r}...", S._beats(t_) == n_, S._beats(t_))
f = S._faults({"shots": [{"who": [], "picture": "p", "dur": 2.5, "action": "He turns toward the desk, unlocks the drawer, "
                          "and takes out a small case-book."}]}, [], 2.5)
check("10-28: three verbs in one sentence in 2.5 s are sent back", any("3 actions in 2.5 s" in x for x in f), f)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
