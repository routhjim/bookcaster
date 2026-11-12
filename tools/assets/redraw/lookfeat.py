"""What scenelook.py needs measured on each key, at a card station (never on the CPU):

    python tools/assets/redraw/lookfeat.py LOOK_CFG.json                  (scenelook.py cfg wrote it)
    python tools/assets/redraw/lookfeat.py --anchors LOOK_CFG.json LOOK.json OUT_DIR   (crops only: Pillow, no model)

Per person per key, from the person's engine box and the engine's head points (no detector): the head crop's
embedding, and closed answers from the vision judge on that crop at ANY head size above a sliver: facial hair,
hair colour, apparent age, and each staged worn item. Per large piece: the embedding of its crop in the redraw
and in the engine still. Per key: a small grey picture, for "two consecutive keys open on the same picture".

The embedder is behind one small interface (Embedder.embed(list of PIL images) -> list of vectors):
  "clip"  the CLIP ViT-L/14 image tower already in the box's model cache (no download). It reads a head crop as a
          whole: hair colour and mass, age, facial hair and face shape together, and keeps working on heads of
          30-80 px where a face-recognition net has nothing to align. This is the default.
  another name: register a class in EMBEDDERS. A face-recognition net (ArcFace r50, 112 px aligned crops) separates
          two similar men better at large head sizes and is the one to add if "clip" proves too coarse there;
          it needs its weights fetched, so it is not wired here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
W, H = 1152, 640
HEAD_PAD = 0.9           # as facecrop.py: the square around head top to chin, widened by this many head heights
MIN_HEAD = 0.04          # scenelook.SLIVER_BAR: under this no head question is asked
THUMB = (32, 18)

Q_FACIAL = ("The picture shows one person's head in a film frame. What facial hair does this person have? Answer "
            "with exactly one of: none, moustache, beard, moustache and beard, cannot tell.")
Q_HAIR = ("The picture shows one person's head in a film frame. What colour is this person's hair? Answer with "
          "exactly one of: black, dark brown, brown, light brown, blond, auburn, red, black with grey, grey, white, "
          "cannot tell.")
Q_AGE = ("The picture shows one person's head in a film frame. About how old does this person look? Answer with a "
         "number of years only, or: cannot tell.")
Q_WORN = "The picture shows one person in a film frame. Is this person wearing {item}? Answer yes, no or cannot tell."


class Embedder:
    """embed(list of PIL images) -> list of unit vectors (lists of floats)."""
    def embed(self, ims: list) -> list:
        raise NotImplementedError


class ClipEmbedder(Embedder):
    def __init__(self, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        name = "openai/clip-vit-large-patch14"
        self.torch, self.device = torch, device
        self.proc = CLIPProcessor.from_pretrained(name, local_files_only=True)
        self.m = CLIPModel.from_pretrained(name, local_files_only=True, torch_dtype=torch.float16).to(device).eval()

    def embed(self, ims: list) -> list:
        if not ims:
            return []
        x = self.proc(images=ims, return_tensors="pt")["pixel_values"].to(self.device, self.torch.float16)
        with self.torch.inference_mode():
            f = self.m.get_image_features(pixel_values=x)
            if not hasattr(f, "norm"):          # newer transformers return an output object, not the tensor
                f = getattr(f, "image_embeds", None) if getattr(f, "image_embeds", None) is not None else f.pooler_output
            f = f.float()
        f = f / f.norm(dim=-1, keepdim=True)
        return [[round(v, 5) for v in row] for row in f.cpu().tolist()]


EMBEDDERS = {"clip": ClipEmbedder}


def head_box(j2: dict, pad: float = HEAD_PAD):
    """The square around head top to chin in pixels, or None (facecrop.py's rule, from the engine's own points)."""
    if "chin" not in j2 or "head_top" not in j2:
        return None
    (tx, ty), (cx, cy) = j2["head_top"][:2], j2["chin"][:2]
    hh = (cy - ty) * H
    if hh <= 0:
        return None
    mx, my, half = (tx + cx) / 2 * W, (ty + cy) / 2 * H, hh * (0.5 + pad)
    b = (int(max(0, mx - half)), int(max(0, my - half)), int(min(W, mx + half)), int(min(H, my + half)))
    return b if b[2] - b[0] >= 8 and b[3] - b[1] >= 8 else None


def body_box(box: list):
    return (int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))


def thumb(im: Image.Image) -> list:
    return [round(v / 255.0, 4) for v in im.convert("L").resize(THUMB).getdata()]


def thing_crop(mask_path: str):
    """-> (box in pixels with 12 px of margin, the share of the box the piece's own pixels fill) or None."""
    import numpy as np
    mk = np.asarray(Image.open(mask_path).convert("L").resize((W, H))) > 127
    ys, xs = np.nonzero(mk)
    if not len(xs):
        return None
    bx = (int(max(0, xs.min() - 12)), int(max(0, ys.min() - 12)), int(min(W, xs.max() + 12)), int(min(H, ys.max() + 12)))
    return bx, float(mk.sum()) / max(1, (bx[2] - bx[0]) * (bx[3] - bx[1]))


def anchors(cfg: dict, look: dict, out_dir: Path) -> None:
    """Each person's reference crop from their anchor key (head and shoulders: 1.4 head heights of padding)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    items = {it["id"]: it for it in cfg["items"]}
    for k, v in look["people"].items():
        it = items.get(v.get("anchor") or "")
        p = next((p for p in (it or {}).get("people", []) if p["key"] == k), None)
        b = head_box((p or {}).get("joints2d") or {}, 1.4)
        if b:
            Image.open(it["redraw"]).convert("RGB").resize((W, H)).crop(b).save(out_dir / f"{k}.png")
            print(out_dir / f"{k}.png")


def main():
    a = sys.argv[1:]
    if a[0] == "--anchors":
        anchors(json.loads(Path(a[1]).read_text()), json.loads(Path(a[2]).read_text()), Path(a[3]))
        return
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: features are read at a card station, never on the CPU")
    cfg = json.loads(Path(a[0]).read_text())
    t0 = time.time()
    emb = EMBEDDERS[cfg.get("embedder", "clip")]("cuda")
    sys.path.insert(0, str(HERE.parents[1]))
    import vl_judge as VJ
    judge = VJ.Judge("cuda")
    print(json.dumps({"phase": "load", "s": round(time.time() - t0, 1)}), flush=True)
    out, cache = {}, {}
    for it in cfg["items"]:
        im = Image.open(it["redraw"]).convert("RGB").resize((W, H))
        r = {"people": {}, "things": {}, "thumb": thumb(im)}
        for p in it.get("people") or []:
            j2 = p.get("joints2d") or {}
            hb = head_box(j2)
            hh = (j2["chin"][1] - j2["head_top"][1]) if hb else 0.0
            q = {}
            if hb and hh >= MIN_HEAD:
                crop = im.crop(hb)
                q["emb"] = emb.embed([crop])[0]
                if (p.get("head_to_camera_deg") or 0) < 100 and not p.get("face_covered"):
                    q["facial_hair"] = judge.ask_images([crop], Q_FACIAL, 6, 448).strip().lower()
                    q["age"] = judge.ask_images([crop], Q_AGE, 4, 448).strip().lower()
                q["hair_colour"] = judge.ask_images([crop], Q_HAIR, 6, 448).strip().lower()
            if p.get("worn") and p.get("box"):
                body = im.crop(body_box(p["box"]))
                q["worn"] = {w: judge.ask_images([body], Q_WORN.format(item=w), 4, 640).strip().lower()
                             for w in p["worn"]}
            r["people"][p["key"]] = q
        for t in it.get("things") or []:
            c = thing_crop(t["mask"]) if t.get("mask") and Path(t["mask"]).exists() else None
            if not c:
                continue
            bx, fill = c
            v = {"fill": round(fill, 3), "area": t.get("area"), "emb_redraw": emb.embed([im.crop(bx)])[0]}
            if it.get("engine"):
                eng = cache.setdefault(it["engine"], Image.open(it["engine"]).convert("RGB").resize((W, H)))
                v["emb_engine"] = emb.embed([eng.crop(bx)])[0]
            r["things"][t["name"]] = v
        out[it["id"]] = r
        print(json.dumps({"id": it["id"], "people": {k: {x: y for x, y in v.items() if x != "emb"}
                                                      for k, v in r["people"].items()}}), flush=True)
    Path(cfg["out"]).write_text(json.dumps(out))
    print(json.dumps({"done": len(out), "s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
