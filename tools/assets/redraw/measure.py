"""Measure redrawn engine stills against the scene's own truth (GPU: Grounding DINO, DINOv2, the local vision judge).

    DETECT_DEVICE=cuda python -I tools/assets/redraw/measure.py MEASURE.json
    MEASURE: {"out": "/abs/metrics.json", "sheet": "/abs/character sheet.png",
              "items": [{"id", "redraw": "/abs/x.png", "engine": "/abs/still.png", "kind": "wide|medium|close",
                         "people": [{"key", "box": [x0, y0, x1, y1], "head_to_camera_deg"}]}],
              "engine_baseline": true}   (also run every check on the engine stills themselves)

Per picture (counts are built from these by the caller):
  people   Grounding DINO "person" boxes; each scene person matched to the best-overlapping box: IoU with the
           engine's own visible box (place held: IoU >= 0.5), box height / engine box height (camera distance
           held: 0.8-1.25), the number of people found v the scene's.
  facing   per scene person, the vision judge on the crop of their box: "front / side / back" (+ a "face"
           detection inside their box); expected from the scene's angle: < 60 front, 60-115 side, >= 115 back.
  room     the judge with both pictures: anything added, removed or covered; window / fireplace / armchair
           detections in both (each class found in the engine still: held when a box overlaps >= 0.4).
  identity DINOv2 cosine between the largest face crop in each scene person's box and the sheet's face crop.
  look     the judge: "photograph or computer render".
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("DETECT_DEVICE", "cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # tools/ (detect.py, vl_judge.py)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import detect as D  # noqa: E402
import vl_judge as VJ  # noqa: E402

ROOM = ["window", "fireplace", "armchair"]


def iou(a, b):
    return D._iou(a, b)


def expected(deg: float) -> str:
    return "front" if deg < 60 else "side" if deg < 115 else "back"


def crop(im, b, pad=0.04):
    W, H = im.size
    x0, y0 = max(0, (b[0] - pad) * W), max(0, (b[1] - pad) * H)
    x1, y1 = min(W, (b[2] + pad) * W), min(H, (b[3] + pad) * H)
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


FACING_Q = ("Look at the man in this picture. Which way is he facing relative to the camera? Answer with one word: "
            "front (we see his face from the front or three-quarters), side (we see his face in profile), or back "
            "(we see the back of his head; his face is hidden).")
LOOK_Q = ("Is this picture a real photograph (or a frame of a live-action film), or a computer-generated render like "
          "a video game? Answer with one word: photo or render.")
ROOM_Q = ("Picture 1 is the original frame and Picture 2 a redrawn version of it. Compare the room only (walls, "
          "window, curtains, fireplace, furniture, pictures, rug, lamps), not the people. Did Picture 2 add any "
          "object that is not in Picture 1, remove or move one, or cover one (for example with a cloth)? Answer "
          "'same' if the room's objects are all still there in the same places, else name what changed in a few "
          "words.")
FRAME_Q = ("Picture 1 is the original frame and Picture 2 a redrawn version of it. Is the camera the same: the "
           "same viewpoint, framing and distance to the people? Answer yes or no.")


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: measuring runs on a card, never on the CPU")
    t0 = time.time()
    dp, dm, ep, em = D.load()
    judge = VJ.Judge("cuda")
    print(json.dumps({"phase": "load", "s": round(time.time() - t0, 1)}), flush=True)
    sheet = Image.open(cfg["sheet"]).convert("RGB")
    sf = D.detect(dp, dm, sheet, ["face"]).get("face", [])
    sheet_face = max(sf, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if sf else None
    sheet_emb = D.embed(ep, em, crop(sheet, sheet_face, 0.02)) if sheet_face else None
    out = {}

    def one(path: str, it: dict, engine_path: str | None):
        im = Image.open(path).convert("RGB")
        det = D.detect(dp, dm, im, ["person", "face"] + ROOM)
        r = {"n_people_found": len(det.get("person", [])), "n_people_scene": len(it["people"]), "people": {}}
        for p in it["people"]:
            b = p["box"]
            best = max(det.get("person", []), key=lambda d: iou(d, b), default=None)
            q = {"truth_deg": p["head_to_camera_deg"], "expected": expected(p["head_to_camera_deg"])}
            if best is not None:
                q["iou"] = round(iou(best, b), 3)
                q["h_ratio"] = round((best[3] - best[1]) / max(1e-6, b[3] - b[1]), 3)
            faces = [f for f in det.get("face", []) if iou(f, b) > 0 and
                     b[0] - 0.02 <= (f[0] + f[2]) / 2 <= b[2] + 0.02 and b[1] - 0.02 <= (f[1] + f[3]) / 2 <= b[3]]
            q["face_boxes"] = len(faces)
            a = judge.ask(crop(im, b), FACING_Q, 4).strip().lower()
            q["judge_facing"] = next((w for w in ("front", "side", "back") if w in a), a[:12])
            if faces and sheet_emb is not None:
                f = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                c = crop(im, f, 0.01)
                if min(c.size) >= 16:
                    e = D.embed(ep, em, c)
                    q["face_sim"] = round(float(e @ sheet_emb), 3)
                    q["_emb"] = e.tolist()
            r["people"][p["key"]] = q
        r["room_dets"] = {k: det.get(k, []) for k in ROOM}
        a = judge.ask(im, LOOK_Q, 3).strip().lower()
        r["look"] = "photo" if "photo" in a else "render" if "render" in a else a[:12]
        if engine_path:
            ims = [Image.open(engine_path).convert("RGB"), im]
            r["room_judge"] = judge.ask_images(ims, ROOM_Q, 24, 768)
            r["frame_judge"] = judge.ask_images(ims, FRAME_Q, 3, 768)
        return r

    eng_cache = {}
    for it in cfg["items"]:
        t = time.time()
        rec = {"redraw": one(it["redraw"], it, it["engine"])}
        if cfg.get("engine_baseline"):
            if it["engine"] not in eng_cache:
                eng_cache[it["engine"]] = one(it["engine"], it, None)
            rec["engine"] = eng_cache[it["engine"]]
            ed = rec["engine"]["room_dets"]
            rd = rec["redraw"]["room_dets"]
            rec["room_held"] = {k: (max((iou(a, b) for a in rd[k] for b in ed[k]), default=0.0) >= 0.4)
                                for k in ROOM if ed[k]}
        out[it["id"]] = rec
        print(json.dumps({"id": it["id"], "s": round(time.time() - t, 1),
                          "people": {k: {x: v.get(x) for x in ("iou", "expected", "judge_facing", "face_boxes", "face_sim")}
                                     for k, v in rec["redraw"]["people"].items()},
                          "look": rec["redraw"]["look"], "room": rec["redraw"].get("room_judge")}), flush=True)
    Path(cfg["out"]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
