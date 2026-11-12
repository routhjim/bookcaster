"""The MetaHuman preset by comparison: portrait renders of the candidate presets (made with the character's own
grooms and body, rendered by char_render in a neutral studio) are compared two at a time against the sheet's drawn
head by the vision model (tools/vl_judge.py): one picture per pair, photo A on the left, the drawing in the middle,
photo B on the right, asked which photo is closer in age, leanness and features; every pair is asked in both orders
(the model leans to one side when it cannot tell) and the candidate with the most wins is kept, ties to the earlier
candidate (the shape ranking's order).

    python -I face_pick.py CHOICE.json --portraits DIR [--vl-python PY]

DIR holds <preset>.png for each name in choice["scores"]["preset_pool"]; the drawing is sheet_head.png beside
CHOICE.json. Writes choice["base"] and choice["scores"]["preset_compare"].
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

QUESTION = ("In the middle is a drawing of a person. On the left (A) and on the right (B) are photos of two different "
            "people. Which photo, A or B, is closer to the drawn person in age, in how lean or full the face is, and in "
            "the shape of the features? Answer A or B.")


def tile(im, h=384):
    return im.resize((max(1, int(im.width * h / im.height)), h))


def head_crop(im):
    """The face of a studio portrait still: the central part, top two thirds (the close framing is fixed)."""
    w, h = im.size
    return im.crop((int(0.3 * w), 0, int(0.7 * w), int(0.85 * h)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("choice")
    ap.add_argument("--portraits", required=True)
    ap.add_argument("--vl-python", default=str(Path("~/tts-ab/.venv-qwenimage/bin/python").expanduser()))
    a = ap.parse_args()
    cp = Path(a.choice)
    ch = json.loads(cp.read_text())
    pool = [n for n in ch["scores"]["preset_pool"] if (Path(a.portraits) / f"{n}.png").exists()]
    drawing = Image.open(cp.with_name("sheet_head.png")).convert("RGB")
    pd = cp.with_name("pairs")
    pd.mkdir(exist_ok=True)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    faces = {n: head_crop(Image.open(Path(a.portraits) / f"{n}.png").convert("RGB")) for n in pool}
    qs = []
    for x, y in itertools.permutations(pool, 2):
        A, D, B = tile(faces[x]), tile(drawing), tile(faces[y])
        im = Image.new("RGB", (A.width + D.width + B.width + 40, 444), "white")
        im.paste(A, (0, 60))
        im.paste(D, (A.width + 20, 60))
        im.paste(B, (A.width + D.width + 40, 60))
        dr = ImageDraw.Draw(im)
        dr.text((A.width // 2 - 15, 5), "A", fill="black", font=font)
        dr.text((A.width + 20 + D.width // 2 - 70, 5), "drawing", fill="black", font=font)
        dr.text((A.width + D.width + 40 + B.width // 2 - 15, 5), "B", fill="black", font=font)
        p = pd / f"{x}__{y}.png"
        im.save(p)
        qs.append({"id": f"{x}|{y}", "images": [str(p)], "max_new_tokens": 3, "max_side": 1024,
                   "ask": {"pick": QUESTION}})
    wins = {n: 0 for n in pool}
    answers = {}
    if len(pool) > 1:
        vl = subprocess.run([a.vl_python, "-I", str(Path(__file__).resolve().parents[2] / "vl_judge.py")],
                            input="".join(json.dumps(q) + "\n" for q in qs), capture_output=True, text=True)
        for ln in vl.stdout.splitlines():
            if not ln.startswith("{"):
                continue
            r = json.loads(ln)
            x, y = r["id"].split("|")
            v = (r["answers"].get("pick") or "").strip().upper()
            answers[r["id"]] = v
            if v.startswith("A"):
                wins[x] += 1
            elif v.startswith("B"):
                wins[y] += 1
        if not answers:
            raise SystemExit(f"vision model gave no answers: {vl.stderr[-800:]}")
    order = {n: i for i, n in enumerate(pool)}
    ch["base"] = max(pool, key=lambda n: (wins[n], -order[n]))
    ch["scores"]["preset_compare"] = {"pool": pool, "wins": wins, "answers": answers}
    cp.write_text(json.dumps(ch, indent=1))
    print(json.dumps({"base": ch["base"], "wins": wins}))


if __name__ == "__main__":
    main()
