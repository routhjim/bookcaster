"""10-27 evening fixes: talk shots end with their line (a reaction for the rest), one beat per ~2 s of action, a
one-person talk key is a close-up (asked), a two-shot with someone seated/lying keeps only sides and faces, no 'his
usual' in descriptions, outfit-sheet portraits out of training, outfits drawn in turn, helmets are props, --render."""
import json, tempfile, sys
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
check("a one-person talk key: the room only soft behind the face, no ', standing' clause (10-27: medium shots)",
      "Watson, standing" not in kp and "fills the whole picture" not in kp and "softly out of focus" in kp, kp)
check("a one-person talk key gets no wide room reference", S.closeup({"kind": "talk", "who": ["A"]})
      and not S.closeup({"kind": "talk", "who": ["A", "B"]}) and "hero = None if closeup(s)" in Path(S.__file__).read_text())
check("10-28 A/B: a standing close-up has the posture in its subject and keeps its real background",
      "alone, standing upright:" in kp and "a glimpse of Baker Street Bedroom" in kp and "papered wall" not in kp
      and "chair" not in kp, kp)
kp2 = S._key_prompt(cx.book, "x", "talk-000996", {"id": "talk-000996_s0", "who": ["Doctor Watson"], "kind": "talk",
                    "framing": "close", "place": "Baker Street Bedroom", "pose_held": {"Doctor Watson": "sit"},
                    "picture": S.TALK_PICTURE.format(who="Doctor Watson")})
check("10-28: a seated close-up shows the chair back", "top of a chair back" in kp2 and "alone, seated:" in kp2
      and "standing" not in kp2, kp2)
q2, _ = S._key_questions(cx, {"id": "talk-000964_s0", "who": ["Doctor Watson"], "kind": "talk", "picture": "",
                              "framing": "close", "pose_held": {"Doctor Watson": "stand"}})
check("10-28: a standing close-up is checked for a chair behind him (answer must be no)",
      any("is there a chair, sofa or bed" in v for k_, v in q2.items() if k_.startswith("chair"))
      and not any(k_.startswith("pose") for k_ in q2), q2)
import pathlib
kf = pathlib.Path(tempfile.mkdtemp()) / "k.png"
from PIL import Image
Image.new("RGB", (64, 36), (90, 80, 70)).save(kf)
check("10-28: 'yes, a chair' fails the seed, 'no' passes",
      not S._clean({"text": "No", "who0": "Yes", "chair0": "Yes", "count": "1"}, kf, 1) and
      S._clean({"text": "No", "who0": "Yes", "chair0": "No", "count": "1"}, kf, 1))
for t_, n_ in (("The man in the dark frock coat turns away toward the desk, unlocks the drawer, and takes out a small "
                "case-book.", 3), ("The man in the dark frock coat, with his hat in his hands, stands by the door.", 1),
               ("The woman in the black mourning dress sits still, hands in her lap.", 1),
               ("He unlocks a small drawer. He pulls out a thin case-book.", 2)):
    check(f"10-28: {n_} action(s) in {t_[:40]!r}...", S._beats(t_) == n_, S._beats(t_))
f = S._faults({"shots": [{"who": [], "picture": "p", "dur": 2.5, "action": "He turns toward the desk, unlocks the drawer, "
                          "and takes out a small case-book."}]}, [], 2.5)
check("10-28: three verbs in one sentence in 2.5 s are sent back", any("3 actions in 2.5 s" in x for x in f), f)
f = S._faults({"shots": [{"who": [], "picture": "p", "dur": 2.0, "action": "He opens the case-book and flips through the pages."}]}, [], 2.0)
check("10-28: two small gestures in 2 s are fine", not any("actions in" in x for x in f), f)
u = S.uncarried("wearing His usual dark frock coat, waistcoat, and trousers, topped with a black top-hat")
check("10-28: no dangling 'topped' and no 'His usual' in a look", "topped" not in u and "usual" not in u.lower()
      and u.endswith("trousers"), u)
check("10-28: 'topped with fur' (no article) stays", "topped with fur" in S.uncarried("wearing a coat topped with fur"))
sh = {"kind": "quiet", "place": "Baker Street Sitting Room (inside)", "who": ["Sherlock Holmes"],
      "pose": {"Sherlock Holmes": "standing"}, "picture": "Sherlock Holmes stands by the desk. The Baker Street Sitting "
      "Room in morning light; the sitting room's window open.", "action": "He looks up.", "dur": 3.0, "start": 0.0}
f = S._shot_faults({"shots": [sh]}, ["Sherlock Holmes"], 3.0, [])
check("10-28: 'Sitting Room' in a picture is not someone sitting", not any("someone sitting" in x for x in f), f)
f = S._shot_faults({"shots": [dict(sh, picture="Sherlock Holmes sits by the desk.")]}, ["Sherlock Holmes"], 3.0, [])
check("10-28: a real 'sits' against a standing pose is still sent back", any("someone sitting" in x for x in f), f)
f = S._shot_faults({"shots": [dict(sh, picture="SHERLOCK HOLMES STANDS BY THE DESK. THE SITTING ROOM'S WINDOW.")]},
                   ["Sherlock Holmes"], 3.0, [], ["Baker Street Sitting Room (inside)"])
check("10-28: case doesn't matter (upper-case title and posture words)", not any("someone" in x for x in f), f)
for room in ("Huddle Room", "Bedroom", "Sitting Room", "Lying-in Ward"):
    pl = f"The Manor {room} (inside)"
    sh2 = dict(sh, place=pl, picture=f"Sherlock Holmes stands by the desk. The Manor {room} in morning light; "
               f"through the door, the Old Sitting Room.")
    f = S._shot_faults({"shots": [sh2]}, ["Sherlock Holmes"], 3.0, [], [pl, "The Old Sitting Room"])
    check(f"10-28: no posture read from any place's title ({room!r})", not any("someone" in x for x in f), f)
for pz, w in (("lie", "lying down"), ("kneel", "kneeling"), ("sit", "seated"), ("stand", "standing upright")):
    k3 = S._key_prompt(cx.book, "x", "talk-000996", {"id": "talk-000996_s0", "who": ["Doctor Watson"], "kind": "talk",
                       "framing": "close", "place": "Baker Street Bedroom", "pose_held": {"Doctor Watson": pz},
                       "picture": S.TALK_PICTURE.format(who="Doctor Watson")})
    check(f"10-28: a {pz} close-up says '{w}' outright", f"alone, {w}:" in k3, k3[:120])
check("10-28: a two-shot cheats out - the speaker's face toward the viewer, not 'facing each other' in profile",
      "turned three-quarters toward the viewer" in S.TALK_TWO and "facing each other" not in S.TALK_TWO
      and "close to the viewer" in S.TALK_TWO)
print(json.dumps({"fails": fails}))
sys.exit(1 if fails else 0)
