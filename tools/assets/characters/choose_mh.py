"""Choose a MetaHuman start for a character sheet by measurement (GPU: CLIP ViT-L/14 + DINOv2; refuses the CPU).

    python -I choose_mh.py SHEET.png --thumbs DIR --out choice.json [--look "a man of about 40, ..."]

The sheet is the story's character sheet (a head-and-shoulders portrait on the left half, the full figure on the
right half, plain paper). Measured on it:
  - gender and age band (CLIP zero-shot over the head crop);
  - preset candidates (scores.preset_pool; the final pick is face_pick.py's, by comparison of rendered portraits):
    the presets the facts leave (gender, the stated or measured age), ranked by thumbnail head shape (DINOv2 cosine
    of GREYSCALE head crops,
    so the presets' tinted thumbnails and the drawing's colours do not count), among presets of the measured gender,
    weighted by agreement of the CLIP age band;
  - hair, beard, moustache and eyebrows, hair colour and build: one question per case to the vision model
    (tools/vl_judge.py: the image model's own Qwen3-VL) with the sheet's head or figure, choosing by number from the
    shipped grooms (their names read as words, "Hair_S_SlickBack" -> "short slick back hair"; beard and moustache
    with a "none" option), named colours (-> melanin / redness) and builds (-> body constraints round the measured
    gender's defaults).
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
# melanin, redness, whiteness (the grooms' grey: a greying colour is its base colour with whiteness, not a lighter
# melanin, which read as dark brown on a "grey-brown" man, 11-12)
COLOURS = {"jet black": (0.95, 0.15, 0.0), "black": (0.85, 0.2, 0.0), "dark brown": (0.65, 0.35, 0.0),
           "brown": (0.5, 0.45, 0.0), "auburn": (0.45, 0.8, 0.0), "red": (0.35, 0.95, 0.0),
           "light brown": (0.35, 0.4, 0.0), "blond": (0.15, 0.35, 0.0), "grey": (0.3, 0.05, 0.75),
           "white": (0.05, 0.0, 1.0), "black with grey streaks": (0.7, 0.1, 0.3), "grey-brown": (0.45, 0.3, 0.65),
           "salt and pepper": (0.75, 0.1, 0.5)}
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
    ap.add_argument("--hair-map", default=None, help="measured whiteness per colour word (hair_calibrate.py)")
    ap.add_argument("--compare", type=int, default=7, help="candidate presets for the comparison (face_pick.py)")
    ap.add_argument("--vl-python", default=str(Path("~/tts-ab/.venv-qwenimage/bin/python").expanduser()),
                    help="a Python with the vision model's packages (tools/vl_judge.py)")
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
        try:
            im = Image.open(p).convert("RGB")
        except OSError:
            continue
        if im.size[0] >= 1000 and im.size[0] == im.size[1]:      # a flat face texture, not a portrait
            continue
        th.setdefault(name, []).append(im)
    names = sorted(th)
    pg, pa, crops = {}, {}, {}
    for n in names:
        im = th[n][0]
        w, h = im.size
        if w > h:       # a studio portrait still (char_render close framing): the head in the middle
            crop = im.crop((int(0.3 * w), 0, int(0.7 * w), int(0.85 * h)))
        else:           # a shipped bust thumbnail
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
    # hair, facial hair, hair colour and build: one question per case to the vision model (the image model's own
    # Qwen3-VL, tools/vl_judge.py) with the sheet, choosing from the shipped sets by number; CLIP's similarity picked
    # long loose hair for an up-do and stubble for a clean-shaven face
    head_p = Path(a.out).with_name("sheet_head.png")
    fig_p = Path(a.out).with_name("sheet_figure.png")
    head.save(head_p)
    fig.save(fig_p)
    # the upper body (head to mid-chest) at full resolution: the neckline is a few pixels in the whole figure
    bust_p = Path(a.out).with_name("sheet_bust.png")
    fw_, fh_ = fig.size
    fig.crop((0, 0, fw_, int(0.4 * fh_))).save(bust_p)

    def menu(items):
        return "\n".join(f"{i}. {t}" for i, t in enumerate(items))
    # short, general questions with one-word answers (a long numbered list drew a fixed favourite number)
    asks = {
        "moustache": "Does this person have a moustache (hair on the upper lip)? Answer yes or no.",
        "beard": "Does this person have a beard or stubble on the chin and cheeks? Answer yes or no.",
        "worn": "How is this person's hair worn: up (pinned up, in a bun or an up-do), back (tied back) or down? "
                "Answer with one word: up, back or down.",
        "length": "How long is this person's hair: short, medium or long? Answer with one word.",
        "bald": "Is this person bald or going bald on top? Answer yes or no.",
        "style": "Which word describes this person's hair best: slicked, parted, curly, messy, cropped or fringe? "
                 "Answer with one of these words.",
        "beard_kind": "If this person has a beard: is it full, short or stubble? Answer with one word, or none.",
        "moustache_kind": "If this person has a moustache: is it thin, full or large? Answer with one word, or none.",
        "colour": "What colour is this person's hair in the drawing? Reply with the number only.\n"
                  + menu([f"{c} hair" for c in COLOURS]),
    }
    asks["build"] = "How is this person built? Reply with the number only.\n" + menu(list(BUILDS))
    # the garment's facts for the fit and the drawing: a garment to the ankles (skirt, dress, gown) takes the dress
    # rules; how loose the sleeves are at the upper arm sets the spread drawing's starting arm angle
    asks["long_garment"] = ("Does this person wear a garment that reaches the ankles, such as a long skirt, a dress, "
                            "a gown or a nightgown? Answer yes or no.")
    asks["neckline"] = ("Where is the top of this person's garment at the neck: chin (a collar, neckband or cravat "
                        "that comes up the neck to the chin), base (a neckline at the base of the neck) or open (a "
                        "lower, open neckline)? Answer with one word: chin, base or open.")
    asks["outer_layer"] = ("Does this person wear an outer layer without sleeves that hangs loose from the shoulders "
                           "over the other clothes, such as a cloak, cape, mantle or shawl (not a coat or jacket with "
                           "sleeves)? Answer yes or no.")
    asks["sleeves"] = ("How loose are this person's sleeves at the upper arm: close (fitted to the arm), medium, or "
                       "loose (full, hanging away from the arm)? Answer with one word: close, medium or loose.")
    import subprocess

    def ask_vl(questions):
        vl = subprocess.run([a.vl_python, "-I", str(Path(__file__).resolve().parents[2] / "vl_judge.py")],
                            input="".join(json.dumps(x) + "\n" for x in questions), capture_output=True, text=True)
        out_ = {}
        for ln in vl.stdout.splitlines():
            if ln.startswith("{"):
                out_.update({k: (v or "").strip().lower() for k, v in json.loads(ln)["answers"].items()})
        if not out_:
            raise SystemExit(f"vision model gave no answers: {vl.stderr[-800:]}")
        return out_
    # the sheet's text is the fact where it states something: each question is first put to the written description
    # (answer "unstated" when the text does not say); only what the text leaves open is asked of the picture
    ans, src = {}, {}
    if a.look.strip():
        tq = {k: (f"Here is a written description of a person: \"{a.look.strip()}\"\nUse only this description, not "
                  f"the picture. {v.split(' Answer')[0].split(' Reply')[0]} If the description does not say, answer "
                  f"unstated." + (" Otherwise " + v[v.index("Answer"):] if "Answer" in v else
                                  " Otherwise " + v[v.index("Reply"):]))
              for k, v in asks.items()}
        tans = ask_vl([{"id": "text", "image": str(head_p), "ask": tq, "max_new_tokens": 6}])
        for k, v in tans.items():
            if v and "unstated" not in v and "not" not in v.split()[:1]:
                ans[k], src[k] = v, "text"
    rest = {k: v for k, v in asks.items() if k not in ans}
    if rest:
        FIG_KEYS = ("build", "long_garment")               # asked of the whole figure, not the head
        BUST_KEYS = ("neckline", "sleeves", "outer_layer")  # asked of the upper body (small in the whole figure)
        figq = {k: v for k, v in rest.items() if k in FIG_KEYS}
        bustq = {k: v for k, v in rest.items() if k in BUST_KEYS}
        pans = ask_vl([{"id": "head", "image": str(head_p), "max_new_tokens": 4,
                        "ask": {k: v for k, v in rest.items() if k not in FIG_KEYS + BUST_KEYS}}]
                      + ([{"id": "fig", "image": str(fig_p), "max_new_tokens": 4, "ask": figq}] if figq else [])
                      + ([{"id": "bust", "image": str(bust_p), "max_new_tokens": 4, "ask": bustq}] if bustq else []))
        for k, v in pans.items():
            ans[k], src[k] = v, "picture"
    sc["answer_source"] = src

    def num(v, n):
        import re as _re
        m = _re.search(r"\d+", v or "")
        return min(max(int(m.group()), 0), n - 1) if m else 0
    sc["vlm_answers"] = ans
    yes = lambda k: ans.get(k, "").startswith("y")   # noqa: E731
    # the hair: the answers narrow the shipped set (by the words in the grooms' names); CLIP picks within it
    worn, length, style = ans.get("worn", ""), ans.get("length", ""), ans.get("style", "")
    hair = GROOMS["Hair"]
    if yes("bald"):
        cand = [o for o in hair if o in ("S_BaldingStubble", "S_HairLoss", "S_RecedeMessy")]
    elif worn.startswith("up"):
        cand = [o for o in hair if "Updo" in o or "SweptUp" in o]
    elif worn.startswith("back"):
        cand = [o for o in hair if "Ponytail" in o or "PulledBack" in o or "SlickBack" in o]
    else:
        ln_ = {"short": "S_", "medium": "M_", "long": "L_"}.get(length.split()[0] if length else "", "")
        cand = [o for o in hair if o.startswith(ln_) and "Updo" not in o and "Ponytail" not in o] or hair
        key = {"slicked": ("SlickBack", "PulledBack"), "parted": ("SideSwept", "Clean", "Casual"),
               "curly": ("Curly", "Coil", "Afro"), "messy": ("Messy", "Clumps", "Casual"),
               "cropped": ("BuzzCut", "BrushCut", "Fade"), "fringe": ("Fringe", "Bangs")}.get(style.split()[0] if style else "")
        if key:
            cand = [o for o in cand if any(k_ in o for k_ in key)] or cand
    grooms, clear = {}, []

    def best(slot, opts):
        texts = [f"{subj} {words(slot, o)}" for o in opts]
        pr = clip_probs(head, texts)
        return opts[int(np.argmax(pr))]
    subj = f"a drawing of a {gender} with"
    grooms["Hair"] = f"WI_Hair_{best('Hair', cand)}"
    grooms["Eyebrows"] = f"WI_Eyebrows_{best('Eyebrows', GROOMS['Eyebrows'])}"
    if gender == "woman" or not yes("beard"):
        clear.append("Beard")
    else:
        bk = ans.get("beard_kind", "")
        bc = [o for o in GROOMS["Beard"] if ("Stubble" in o) == bk.startswith("stub")]
        bc = [o for o in bc if not bk.startswith("short") or o.startswith("S_")] or bc
        # a full beard is a full one (CLIP picked a pencil-thin chin line for "full", 11-12)
        bc = [o for o in bc if not bk.startswith("full") or ("Full" in o or "Messy" in o or o.startswith("L_"))] or bc
        grooms["Beard"] = f"WI_Beard_{best('Beard', bc or GROOMS['Beard'])}"
    if gender == "woman" or not yes("moustache"):
        clear.append("Mustache")
    else:
        mk = ans.get("moustache_kind", "")
        mc = [o for o in GROOMS["Mustache"] if (o.startswith("L_") if mk.startswith("large") else
                                                 ("PencilThin" in o or "ChinStrap" in o) if mk.startswith("thin") else o.startswith(("S_", "M_")))]
        grooms["Mustache"] = f"WI_Mustache_{best('Mustache', mc or GROOMS['Mustache'])}"
    col = list(COLOURS)[num(ans.get("colour"), len(COLOURS))]
    sc["hair_colour"] = col
    mel, red, wht = COLOURS[col]
    if a.hair_map and Path(a.hair_map).exists():
        # a measured whiteness for this word (hair_calibrate.py: the value whose render the vision model names so)
        mw = (json.loads(Path(a.hair_map).read_text()).get(col) or {}).get("whiteness")
        if mw is not None:
            wht = float(mw)
            sc["hair_whiteness_measured"] = wht
    params = {f"Bindings/{d}/": {"Melanin": mel, "Redness": red, "Whiteness": wht}
              for d in ("Hair", "Eyebrows", "Beards", "Mustaches")}
    build = list(BUILDS)[num(ans.get("build"), len(BUILDS))]
    sc["build"] = build
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
    # the preset candidates: the facts above narrow them (gender; the stated age when the text gives one, else the
    # picture's), ranked by greyscale head shape; cast_char renders each with these grooms and body and face_pick.py
    # lets the vision model compare them two at a time with the drawn head (the shape ranking alone picked a young,
    # round-cheeked face for a lean older man, 11-11)
    m_age = re.search(r"\b(?:about|around|aged?)\s+(\d{2})\b", a.look or "")
    age_t = float(m_age.group(1)) if m_age else age
    sc["age_used"] = age_t
    # every preset of the gender goes on (29 presets ship, 16 of them men): the thumbnails' age guess is not trusted
    # (tinted busts; a +-15 year window on it dropped the two older men's presets for a man "of about 40"); the
    # face step renders them all and the vision model's age of each render narrows them to the stated age
    pool = sorted((n for n in names if pg[n] == gender), key=lambda n: sims[n], reverse=True)
    sc["preset_pool"] = pool
    sc["compare_keep"] = a.compare
    base = pool[0] if pool else base
    # the outer layer's own words in the look line (text only: the drawing leaves it out by these words)
    outer_words = []
    if (ans.get("outer_layer") or "").startswith("y") and a.look.strip():
        ow = ask_vl([{"id": "outer", "image": str(head_p), "max_new_tokens": 12, "ask": {"words": (
            f"Here is a written description of a person: \"{a.look.strip()}\"\nUse only this description. Which "
            f"garment in it is an outer layer hanging from the shoulders over the other clothes? Reply with that "
            f"garment's own words from the description only, or none.")}}]).get("words", "")
        outer_words = [w for w in re.findall(r"[a-z]+", ow.lower()) if len(w) > 2 and w not in ("none", "and", "the")]
    lg = ans.get("long_garment", "")
    garment = {"long": True if lg.startswith("y") else False if lg.startswith("n") else None,
               "outer_layer": True if (ans.get("outer_layer") or "").startswith("y") else
               False if (ans.get("outer_layer") or "").startswith("n") else None,
               "outer_words": outer_words,
               "neckline": next((w for w in ("chin", "base", "open") if (ans.get("neckline") or "").startswith(w)), None),
               "sleeves": next((w for w in ("close", "medium", "loose") if (ans.get("sleeves") or "").startswith(w)), None),
               "source": {k: src.get(k) for k in ("long_garment", "sleeves", "neckline", "outer_layer")}}
    out = {"base": base, "gender": gender, "grooms": grooms, "clear_slots": clear, "params": params, "garment": garment,
           "body": {k: round(float(v), 1) for k, v in body.items()}, "scores": sc}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("base", "gender", "grooms", "clear_slots", "body")}))


if __name__ == "__main__":
    main()
