"""Choose a MetaHuman start for a character sheet by measurement (GPU: CLIP ViT-L/14 + DINOv2; refuses the CPU).

    python -I choose_mh.py SHEET.png --thumbs DIR --out choice.json [--look "a man of about 40, ..."]

The sheet is the story's character sheet (a head-and-shoulders portrait on the left half, the full figure on the
right half, plain paper). Measured on it:
  - gender and age band (CLIP zero-shot over the head crop);
  - preset: the shipped preset whose thumbnail head is most alike in shape (DINOv2 cosine of GREYSCALE head crops,
    so the presets' tinted thumbnails and the drawing's colours do not count), among presets of the measured gender,
    weighted by agreement of the CLIP age band;
  - hair, beard, moustache and eyebrows: the shipped groom whose name, read as words ("Hair_S_SlickBack" -> "short
    slicked back hair"), CLIP finds most like the head crop (beard and moustache have a "clean-shaven" option);
  - hair colour: CLIP over named colours, mapped to the groom's melanin / redness;
  - build: CLIP over slim / average / stocky / very heavily built for the full figure, mapped to the body
    constraints (Fat, Muscularity, Chest, Waist, Across Shoulder) around the measured gender's defaults.
The --look text (the story's look line), when given, is only used as CLIP's subject words ("a man", "a woman").
Writes choice.json: {"base", "grooms", "clear_slots", "params", "body", "scores": {...}}.
Thumbnails: <DIR>/<Preset>_<n>.(png|jpg): the shipped preset pictures (the bust renders are used: any whose
picture is not a flat face texture).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

GROOMS = {
    "Hair": ["L_AfroCurly", "L_MessyClumps", "L_Straight", "L_StraightBangs", "M_BobBangs", "M_BobCurly", "M_BobMessy",
             "M_BobSlick", "M_BobStraight", "M_Layered", "M_SideSweptFringe", "M_TwistedBraids", "S_BaldingStubble",
             "S_BobLayered", "S_BrushCut", "S_BuzzCut", "S_Casual", "S_Clean", "S_Coil", "S_CurlyFade", "S_HairLoss",
             "S_LowPonytail", "S_Messy", "S_Pixie", "S_PulledBack", "S_RecedeMessy", "S_SideSweptFringe",
             "S_SlickBack", "S_SweptUp", "S_Updo", "S_UpdoBraids", "S_UpdoBuns"],
    "Beard": ["L_Full", "L_Messy", "M_Curly", "M_MuttonChops", "M_Stubble", "S_Curly", "S_Full", "S_PencilThin",
              "S_Stubble", "S_Uneven"],
    "Mustache": ["L_Full", "L_Handlebar", "L_Messy", "L_Scraggly", "L_Wavy", "M_Curly", "M_Gunslinger", "S_ChinStrap",
                 "S_Curly", "S_Full", "S_Horseshoe", "S_PencilThin"],
    "Eyebrows": ["L_Scraggly", "M_Dense", "M_Fine", "M_FlatThick", "M_Full", "M_Natural", "M_SlightArch", "M_Thick",
                 "M_Thin", "S_FlatThin"],
}
LEN = {"S": "short", "M": "medium-length", "L": "long"}
NOUN = {"Hair": "hair", "Beard": "beard", "Mustache": "moustache", "Eyebrows": "eyebrows"}
COLOURS = {"jet black": (0.95, 0.15), "black": (0.85, 0.2), "dark brown": (0.65, 0.35), "brown": (0.5, 0.45),
           "auburn": (0.45, 0.8), "red": (0.35, 0.95), "light brown": (0.35, 0.4), "blond": (0.15, 0.35),
           "grey": (0.3, 0.05), "white": (0.05, 0.0), "black with grey streaks": (0.7, 0.1)}
BUILDS = {"slim": dict(Fat=-1.6, Muscularity=-1.2, d=-12), "average": dict(Fat=-0.4, Muscularity=0.0, d=0),
          "stocky": dict(Fat=0.6, Muscularity=0.6, d=10), "very heavily built": dict(Fat=1.6, Muscularity=1.4, d=22)}
BANDS = [20, 30, 40, 50, 60, 70]


def words(slot, name):
    ln, style = name.split("_", 1)
    st = re.sub(r"(?<!^)([A-Z])", r" \1", style).lower().replace("updo", "up-do")
    return f"{LEN[ln]} {st} {NOUN[slot]}"


def heads(sheet: Image.Image):
    """(head crop, full-figure crop) of a house-style sheet: portrait on the left half, figure on the right half."""
    w, h = sheet.size
    return sheet.crop((int(0.08 * w), 0, int(0.42 * w), int(0.62 * h))), sheet.crop((w // 2, 0, w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--thumbs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--look", default="")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible (HIP_VISIBLE_DEVICES): refusing to run on the CPU")
    dev = "cuda"
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
    cp = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    cm = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval().half()
    dp = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dm = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval().half()

    def clip_probs(img, texts):
        with torch.inference_mode():
            x = cp(text=texts, images=[img], return_tensors="pt", padding=True).to(dev)
            x["pixel_values"] = x["pixel_values"].half()
            return cm(**x).logits_per_image.float().softmax(-1)[0].cpu().numpy()

    def dino(imgs):
        with torch.inference_mode():
            x = dp(images=imgs, return_tensors="pt").to(dev)
            e = dm(pixel_values=x["pixel_values"].half()).last_hidden_state[:, 0].float()
            return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()

    sheet = Image.open(a.sheet).convert("RGB")
    head, fig = heads(sheet)
    sc = {}
    g = clip_probs(head, ["a drawing of a man's face", "a drawing of a woman's face"])
    gender = "man" if g[0] >= g[1] else "woman"
    sc["gender"] = {"man": round(float(g[0]), 3), "woman": round(float(g[1]), 3)}
    ag = clip_probs(head, [f"a portrait of a {gender} in their {b}s" for b in BANDS])
    age = float((ag * np.array(BANDS)).sum()) + 5
    sc["age"] = round(age, 1)
    # presets: the bust thumbnails (not the flat face textures: those are square 1024 px PNGs)
    th = {}
    for p in sorted(Path(a.thumbs).glob("*_*.*")):
        name = p.stem.rsplit("_", 1)[0]
        im = Image.open(p).convert("RGB")
        if im.size[0] >= 1000:
            continue
        th.setdefault(name, []).append(im)
    names = sorted(th)
    pg, pa, crops = {}, {}, {}
    for n in names:
        im = th[n][0]
        w, h = im.size
        crop = im.crop((int(0.2 * w), 0, int(0.8 * w), int(0.62 * h)))
        crops[n] = crop
        pr = clip_probs(crop, ["a photo of a man", "a photo of a woman"])
        pg[n] = "man" if pr[0] >= pr[1] else "woman"
        pr = clip_probs(crop, [f"a photo of a person in their {b}s" for b in BANDS])
        pa[n] = float((pr * np.array(BANDS)).sum()) + 5
    grey = lambda im: ImageOps.grayscale(im).convert("RGB")   # noqa: E731
    e = dino([grey(head)] + [grey(crops[n]) for n in names])
    sims = {n: float(e[0] @ e[i + 1]) for i, n in enumerate(names)}
    score = {n: sims[n] - 0.004 * abs(pa[n] - age) - (1.0 if pg[n] != gender else 0.0) for n in names}
    base = max(score, key=score.get)
    sc["preset_top5"] = sorted(((round(v, 4), n, round(sims[n], 3), round(pa[n], 1)) for n, v in score.items()),
                               reverse=True)[:5]
    grooms, clear = {}, []
    subj = f"a drawing of a {gender} with"
    for slot, opts in GROOMS.items():
        if gender == "woman" and slot in ("Beard", "Mustache"):
            clear.append(slot)
            continue
        texts = [f"{subj} {words(slot, o)}" for o in opts]
        if slot in ("Beard", "Mustache"):
            texts.append(f"a drawing of a clean-shaven {gender} with no {NOUN[slot]}")
        pr = clip_probs(head, texts)
        k = int(np.argmax(pr))
        sc[slot] = [(round(float(pr[i]), 3), texts[i].replace(subj + " ", "")) for i in np.argsort(-pr)[:3]]
        if k == len(opts):
            clear.append(slot)
        else:
            grooms[slot] = f"WI_{slot}_{opts[k]}"
    cpr = clip_probs(head, [f"a drawing of a {gender} with {c} hair" for c in COLOURS])
    col = list(COLOURS)[int(np.argmax(cpr))]
    sc["hair_colour"] = [(round(float(cpr[i]), 3), list(COLOURS)[i]) for i in np.argsort(-cpr)[:3]]
    mel, red = COLOURS[col]
    params = {f"Bindings/{d}/": {"Melanin": mel, "Redness": red} for d in ("Hair", "Eyebrows", "Beards", "Mustaches")}
    bpr = clip_probs(fig, [f"a full-length drawing of a {b} {gender}" for b in BUILDS])
    build = list(BUILDS)[int(np.argmax(bpr))]
    sc["build"] = [(round(float(bpr[i]), 3), list(BUILDS)[i]) for i in np.argsort(-bpr)]
    B = BUILDS[build]
    if gender == "man":
        body = {"Height": 180 + (6 if build == "very heavily built" else 0), "Chest": 98 + B["d"], "Waist": 84 + B["d"],
                "Hip": 98 + B["d"] * 0.6, "Across Shoulder": 40 + B["d"] * 0.2, "Neck": 38 + B["d"] * 0.25}
    else:
        body = {"Height": 165, "Chest": 88 + B["d"] * 0.8, "Waist": 70 + B["d"], "Hip": 96 + B["d"] * 0.7,
                "Across Shoulder": 36 + B["d"] * 0.15}
    body.update(Fat=B["Fat"], Muscularity=B["Muscularity"])
    # the neck: the length of the drawn neck is not measured; keep it at the doctor-like 6.5-7 cm the redraws read
    # as natural (a long-limbed slim preset otherwise came out with a ~10 cm neck)
    body["Neck Length"] = 6.8 if gender == "man" else 6.4
    out = {"base": base, "gender": gender, "grooms": grooms, "clear_slots": clear, "params": params,
           "body": {k: round(float(v), 1) for k, v in body.items()}, "scores": sc}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("base", "gender", "grooms", "clear_slots", "body")}))


if __name__ == "__main__":
    main()
