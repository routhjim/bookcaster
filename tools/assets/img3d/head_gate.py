"""Whether each drawing shows its person wearing something on the head (the vision judge, on the GPU it is given).

    python tools/assets/img3d/head_gate.py OUT.json WHAT IMAGE [IMAGE ...]

WHAT: the headwear in the look's own words ("a steel cap"). OUT.json: {image path: true | false}. cast_char.py asks
it of the spread drawings of a figure whose outfit has headwear: told in words to draw it, two of three drawings came
out bare-headed, and the figure would have been shaped from one of those (11-19).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # tools/ (vl_judge.py)

ASK = ("Look at the person's head in this picture. Is the person wearing {what} on the head, covering the top of the "
       "head? Answer with exactly one word: yes or no.")


def worn(answer: str) -> bool:
    return str(answer or "").strip().lower().startswith("y")


def main():
    out, what, imgs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    from PIL import Image
    import vl_judge as VJ
    judge = VJ.Judge("cuda")
    res = {}
    for p in imgs:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        res[p] = worn(judge.ask_images([im.crop((0, 0, w, int(h * 0.45)))], ASK.format(what=what), 4, 768))
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
