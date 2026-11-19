"""The face gate (stdlib): a person whose face the engine frame shows toward the camera, large enough to read, must
have a human face in the redraw. It blocks: a key whose person is drawn as a creature, a statue or no face at all
is never kept, whatever else it passes (11-18: a close shot's head drawn as a rough stone with pointed ears was kept
as the "least bad" of two tries, its only recorded misses the outfit and the hair).

keycheck.py asks the vision judge HUMAN_Q on the person's head crop (head top to chin from the engine's own points)
when due() says so; accept.py reads the answer through verdict() and treats a failed gate as BLOCKING: such a try
never wins the pick over one that passes it, and a shot whose every try fails it gets no key.

Why the judge, not a detector score: the check station's whole-body pose model scores face points on anything
face-shaped (the stone head 0.52-0.57, the engine's own depth picture 0.49, real people lying on their side
0.49-0.59 over the knight_1116 and space_1113 runs), so its face scores cannot tell a human face from a monster; no
identity embedding is computed at the check station (scenelook/lookfeat's CLIP head embeddings are a separate lab
pass, not run by story_keys), so there is no similarity floor here.
"""
from __future__ import annotations

FACE_GATE_DEG = 60.0     # head turned at most this far from the camera (keycheck's neck check: "face within 60 deg")
FACE_GATE_HEAD = 0.12    # head top to chin at least this share of the frame's height (scenelook.FACE_BAR: a head
                         # this tall, 77 px of 640, carries a face; below it the judge sees a few pixels)
CROP_PAD = 0.35          # the judge's crop: the square on head top to chin, widened by this many head heights a side
CROP_MIN = 336           # ... enlarged to at least this many pixels a side for the judge

HUMAN_Q = ("The picture shows the head of one character in a film frame. What is drawn where the face is? Answer "
           "with one word: human (an ordinary living human face: human skin, two eyes, a nose and a mouth), "
           "creature (an animal's or monster's face, or skin of stone, rock, bark or scales), statue (a carved, "
           "sculpted or doll-like head), mask (a mask or a helmet over the face), none (no face: blank, smeared, "
           "melted or hidden).")
HUMAN_YES = ("human",)
HUMAN_NO = ("creature", "statue", "mask", "none")

BLOCKING = ("human face",)   # accept.py failures(): a try with one of these is never a key
NO_KEY_WHY = "no try drew {who} with a human face"


def _in_frame(pt) -> bool:
    return bool(pt) and 0.0 <= pt[0] < 1.0 and 0.0 <= pt[1] < 1.0


def due(p: dict) -> bool:
    """Whether the gate applies to one person of a check item (keycheck's person dict: creature, face_covered,
    head_to_camera_deg, joints2d with the engine's eye joints, chin and head top as [x, y, seen] in 0-1 units):
    not a creature, face not covered by what they wear, head within FACE_GATE_DEG of the camera, both eyes inside
    the frame and at least one of them seen by the engine (not hidden behind something), head top to chin at
    least FACE_GATE_HEAD of the frame."""
    if p.get("creature") or p.get("face_covered"):
        return False
    deg = p.get("head_to_camera_deg")
    if deg is None or abs(deg) > FACE_GATE_DEG:
        return False
    j2 = p.get("joints2d") or {}
    eyes = [j2.get("FACIAL_L_Eye"), j2.get("FACIAL_R_Eye")]
    if not all(_in_frame(e) for e in eyes) or not any(len(e) > 2 and e[2] for e in eyes):
        return False
    return head_height(j2) >= FACE_GATE_HEAD


def head_height(j2: dict) -> float:
    if "chin" not in j2 or "head_top" not in j2:
        return 0.0
    return max(0.0, j2["chin"][1] - j2["head_top"][1])


def crop_box(j2: dict, W: int, H: int) -> tuple:
    """The judge's square (pixels x0, y0, x1, y1, clipped to the frame) around head top to chin (facecrop.py's
    rule with a tighter pad: the face, not the shoulders)."""
    hh = head_height(j2) * H
    side = hh * (1 + 2 * CROP_PAD)
    mx = (j2["head_top"][0] + j2["chin"][0]) / 2 * W
    my = (j2["head_top"][1] + j2["chin"][1]) / 2 * H
    return (int(max(0, mx - side / 2)), int(max(0, my - side / 2)), int(min(W, mx + side / 2)),
            int(min(H, my + side / 2)))


def ask(judge, frame, p: dict) -> dict:
    """The gate's question for one person on one redraw (frame: the PIL picture at the check's 1152 x 640; judge:
    anything with vl_judge.Judge's ask(image, question, max_new_tokens, max_side)) -> {"human_face": answer,
    "human_face_ok": verdict} for a person the gate is due for, else {} (keycheck.py merges it into the person's
    record)."""
    if not due(p):
        return {}
    c = frame.crop(crop_box(p.get("joints2d") or {}, frame.width, frame.height))
    if min(c.size) < 8:
        return {}
    if max(c.size) < CROP_MIN:
        from PIL import Image
        sc = CROP_MIN / max(c.size)
        c = c.resize((round(c.width * sc), round(c.height * sc)), Image.BICUBIC)
    a = str(judge.ask(c, HUMAN_Q, 4, 512)).strip().lower()
    return {"human_face": a, "human_face_ok": verdict(a)}


def verdict(answer) -> bool | None:
    """The judge's answer -> True (a human face), False (a creature, statue, mask or no face), None (not asked or
    not understood: no verdict, nothing blocked)."""
    if not answer:
        return None
    w = str(answer).strip().lower().lstrip("*\"' ").split()
    w = w[0].strip(".,;:!*\"'") if w else ""
    if w in HUMAN_YES:
        return True
    if w in HUMAN_NO:
        return False
    return None


def blocked(fails: dict) -> list:
    """The people of an item's fails ({person: [checks]}) that failed a blocking check."""
    return sorted(k for k, fs in (fails or {}).items() if any(f in BLOCKING for f in fs))


def rank(fails: dict) -> tuple:
    """Pick order for a try (lower is better): any blocking failure first, then the number of failures."""
    return (bool(blocked(fails)), sum(len(v) for v in (fails or {}).values()))
