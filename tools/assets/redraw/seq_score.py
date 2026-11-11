"""Shot-to-shot scores for a run of keys of one scene (GPU: Grounding DINO, DINOv2, the local vision judge).

    DETECT_DEVICE=cuda python -I tools/assets/redraw/seq_score.py SEQ.json
    SEQ: {"out": "/abs/seq_scores.json", "sheet": "/abs/character sheet.png",
          "runs": {"<name>": [{"shot", "image": "/abs/key.png", "truth": {truth.json shot} | null}, ...]}}
Per run, in order:
  per shot  each person of the scene record (truth) matched to a detected person: box IoU with the record's box and
            screen third (L/C/R); main furniture of the record in frame (window, fireplace, armchair, bookcase, desk)
            found where the record has it (box overlap >= 0.3).
  per cut   moved: a person whose third differs from the record's in B while it matched in A, or who is not found
            where the record puts them in B (IoU < 0.3) though found in A; the record's own moves are not counted
            (a person the record moves keeps their record box, so a redraw that follows the move is held).
            room: the judge with both frames: what changed in the room (wallpaper, pictures, mantel dressing,
            furniture, curtains, doors, light); counted as the number of things it lists, 0 for "same".
            man: DINOv2 cosine between the largest face crops of the two frames (when both show a face).
A run with "truth": null (keys from another pipeline) gets only the room and man scores.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DETECT_DEVICE", "cuda")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

import detect as D  # noqa: E402
import vl_judge as VJ  # noqa: E402
from measure import crop  # noqa: E402

FURN = {"window": "window", "fireplace": "fireplace", "armchair": "armchair", "bookcase": "bookcase",
        "bureau": "desk"}
ROOM2_Q = ("These two pictures are consecutive shots of one scene in one room, from different camera positions. "
           "Compare only the room itself, not the people and not the camera angle: the wall colour and wallpaper, "
           "the pictures on the walls, the mantelpiece and what stands on it, the furniture and its fabric, the "
           "curtains, the doors and windows, the lamps and the light. List every thing that is different between "
           "the two rooms, separated by commas, or answer 'same' if it is the same room.")


def third(b):
    c = (b[0] + b[2]) / 2
    return "L" if c < 1 / 3 else "R" if c > 2 / 3 else "C"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: scoring runs on a card, never on the CPU")
    cfg = json.loads(Path(sys.argv[1]).read_text())
    dp, dm, ep, em = D.load()
    judge = VJ.Judge("cuda")
    out = {}
    for name, shots in cfg["runs"].items():
        frames = []
        for s in shots:
            im = Image.open(s["image"]).convert("RGB")
            det = D.detect(dp, dm, im, ["person", "face"] + sorted(set(FURN.values())))
            f = {"shot": s["shot"], "people": {}, "furniture": {}}
            t = s.get("truth")
            if t:
                for k, v in t["people"].items():
                    if not v.get("box") or v.get("area", 0) < 0.002:
                        continue
                    best = max(det.get("person", []), key=lambda d: D._iou(d, v["box"]), default=None)
                    iou = D._iou(best, v["box"]) if best else 0.0
                    f["people"][k] = {"iou": round(iou, 3), "record_third": third(v["box"]),
                                      "third": third(best) if best and iou >= 0.3 else None}
                for th in t.get("things", []):
                    if th["name"] in FURN and th["visible_frac"] >= 0.1:
                        lab = FURN[th["name"]]
                        ok = any(D._iou(d, th["box"]) >= 0.3 for d in det.get(lab, []))
                        f["furniture"][th["name"]] = ok
            faces = det.get("face", [])
            if faces:
                fb = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
                c = crop(im, fb, 0.01)
                if min(c.size) >= 24:
                    f["_face"] = D.embed(ep, em, c)
            frames.append(f)
        cuts = []
        for a, b in zip(frames, frames[1:]):
            c = {"cut": f"{a['shot']} -> {b['shot']}", "moved": []}
            for k, pb in b["people"].items():
                pa = a["people"].get(k)
                if pa is None:
                    continue
                held_a = pa["third"] == pa["record_third"] and pa["iou"] >= 0.3
                held_b = pb["third"] == pb["record_third"] and pb["iou"] >= 0.3
                if held_a and not held_b:
                    c["moved"].append(f"{k}: record {pb['record_third']}, drawn {pb['third']} (IoU {pb['iou']})")
            ia = Image.open(shots[len(cuts)]["image"]).convert("RGB")
            ib = Image.open(shots[len(cuts) + 1]["image"]).convert("RGB")
            ans = judge.ask_images([ia, ib], ROOM2_Q, 40, 768).strip()
            c["room_judge"] = ans
            c["room_changes"] = 0 if ans.lower().startswith("same") else len([x for x in ans.split(",") if x.strip()])
            if "_face" in a and "_face" in b:
                c["man_sim"] = round(float(a["_face"] @ b["_face"]), 3)
            cuts.append(c)
        for f in frames:
            f.pop("_face", None)
        out[name] = {"frames": frames, "cuts": cuts,
                     "cuts_moved": sum(1 for c in cuts if c["moved"]), "n_cuts": len(cuts),
                     "room_changes": sum(c["room_changes"] for c in cuts),
                     "furniture_held": [sum(f["furniture"].values()) for f in frames],
                     "furniture_n": [len(f["furniture"]) for f in frames]}
        print(json.dumps({k: v for k, v in out[name].items() if k not in ("frames", "cuts")} | {"run": name}),
              flush=True)
        for c in cuts:
            print("  ", c["cut"], c["moved"], c["room_changes"], c.get("man_sim"), "|", c["room_judge"][:120], flush=True)
    Path(cfg["out"]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
