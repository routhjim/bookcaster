"""Checks a redrawn key must pass before it is accepted, from the engine's own facts (GPU: a whole-body pose model
and the local vision judge).

    python -I tools/assets/redraw/keycheck.py CHECK.json            (writes CHECK["out"]; run at a card station)

CHECK: {"out": "/abs/checks.json", "pose_model": "/abs/rtmw.pt" (onnx_torch.py export of RTMW whole-body),
        "judge": true,
        "items": [{"id", "redraw": "/abs/x.png", "engine": "/abs/still.png", "depth": "/abs/depth.png" (optional),
                   "people": [{"key", "box", "head_to_camera_deg", "body_to_camera_deg", "joints2d" (guides.py truth),
                               "noun", "wear", "sheet"}]}]}

Per person whose body is within 100 deg of the camera:
  arms    every arm the engine shows in the frame (its elbow or its hand joint inside the frame and seen: the
          person's own surface there in the engine's depth) is found in the redraw: the pose model's elbow or wrist
          (either side: a mirrored left/right is not a missing arm) within ARM_TOL of the person's height in frame;
  hands   every hand the engine shows is found: a wrist, or five confident hand points, near the hand joint;
  neck    (face within 60 deg) the chin-to-shoulder-line distance over the eyes-to-chin distance, from the pose
          model, within NECK_TOL of the same measure on the engine still and of the median over this character's
          keys in the run (the accepted set);
  outfit  the judge, with the character's sheet and the person's crop: do the collar, the neckwear and the front of
          the coat or dress agree with the sheet and the character's own clothes words.
The same pose measures on the engine still (and its depth picture) are reported too: an arm the still itself does
not show cleanly is the guide's fault, not the redraw's.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # tools/ (vl_judge.py)
import onnx_torch  # noqa: E402

IN_W, IN_H = 192, 256
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)
SCORE = 0.35             # a confident key point (RTMW SimCC score)
ARM_TOL = 0.09           # of the person's height in frame
NECK_TOL = 1.25          # longer than the reference by more than this factor fails (and shorter than 1/1.6)
BODY = {"l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10, "nose": 0, "l_eye": 1, "r_eye": 2}
CHIN = 23 + 8            # the 68-point face contour's chin, after the 23 body and feet points
HAND_L, HAND_R = slice(91, 112), slice(112, 133)

OUTFIT_Q = ("Picture 1 is a costume reference sheet of a character. Picture 2 shows the same character in a film "
            "frame. The character's clothes are: {wear}. Look only at the collar, the neckwear and the front of the "
            "coat or dress in Picture 2. Do they agree with Picture 1 and those words (the same kind of collar, worn "
            "the same way, the same neckwear, the coat or dress closed or open the same way)? Answer 'same' if they "
            "agree, otherwise name what differs in a few words.")


class Pose:
    def __init__(self, path: str, dev):
        self.g = onnx_torch.Graph(path, str(dev))
        self.dev = dev

    def __call__(self, img: torch.Tensor, box_px) -> tuple:
        """img: [3, H, W] float RGB 0-255 on the device; box_px: x0, y0, x1, y1. -> (kps [133, 2] px, scores [133])."""
        x0, y0, x1, y1 = box_px
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        w, h = (x1 - x0) * 1.25, (y1 - y0) * 1.25
        if w > h * IN_W / IN_H:
            h = w * IN_H / IN_W
        else:
            w = h * IN_W / IN_H
        H, W = img.shape[1:]
        ys = cy - h / 2 + torch.arange(IN_H, device=self.dev).float() * (h / IN_H)     # as OpenCV's warp
        xs = cx - w / 2 + torch.arange(IN_W, device=self.dev).float() * (w / IN_W)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([gx / (W - 1) * 2 - 1, gy / (H - 1) * 2 - 1], -1)[None]
        crop = F.grid_sample(img[None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0]
        bgr = crop[[2, 1, 0]]                       # the model was trained on OpenCV's BGR pictures
        inp = ((bgr - MEAN.to(self.dev)) / STD.to(self.dev))[None]
        sx, sy = self.g({"input": inp})
        vx, ix = sx[0].max(-1)
        vy, iy = sy[0].max(-1)
        sc = ((vx + vy) / 2).float()
        kx = ix.float() / 2.0 / IN_W * w + cx - w / 2
        ky = iy.float() / 2.0 / IN_H * h + cy - h / 2
        return torch.stack([kx, ky], -1).cpu().numpy(), sc.cpu().numpy()


def load(path: str, dev) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB").resize((1152, 640)), dtype=np.float32)
    return torch.from_numpy(a.copy()).to(dev).permute(2, 0, 1)


def measures(k, s, W, H) -> dict:
    """Pose-model points of one person, in 0-1 frame units, with what they say about arms, hands and the neck."""
    P = lambda i: (float(k[i][0]) / W, float(k[i][1]) / H) if s[i] >= SCORE else None
    out = {n: P(i) for n, i in BODY.items()}
    out["chin"] = P(CHIN)
    for side, sl in (("l", HAND_L), ("r", HAND_R)):
        hk, hs = k[sl], s[sl]
        good = hk[hs >= 0.3]
        out[f"{side}_hand"] = (float(good[:, 0].mean()) / W, float(good[:, 1].mean()) / H) if len(good) >= 5 else None
    e = [out["l_eye"], out["r_eye"]]
    sh = [out["l_sh"], out["r_sh"]]
    if all(e) and all(sh) and out["chin"]:
        ey = (e[0][1] + e[1][1]) / 2
        sy = (sh[0][1] + sh[1][1]) / 2
        if out["chin"][1] - ey > 1e-3:
            out["neck_ratio"] = round((sy - out["chin"][1]) / (out["chin"][1] - ey), 3)
    return out


def near(p, q, tol):
    return p is not None and q is not None and ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5 <= tol


def arm_hand(m: dict, j2: dict, box) -> dict:
    """Which arms and hands the engine shows (joints in frame and seen) and whether the pose model found each."""
    tol = ARM_TOL * max(0.05, box[3] - box[1])
    shown = lambda n: n in j2 and j2[n][2]
    found_pts = [m.get(x) for x in ("l_el", "r_el", "l_wr", "r_wr", "l_hand", "r_hand")]
    res = {"arms_shown": 0, "arms_found": 0, "hands_shown": 0, "hands_found": 0, "missing": []}
    for side in ("l", "r"):
        pts = [(j2[n][0], j2[n][1]) for n in (f"lowerarm_{side}", f"hand_{side}") if shown(n)]
        if pts:
            res["arms_shown"] += 1
            if any(near(p, q, tol) for p in pts for q in found_pts):
                res["arms_found"] += 1
            else:
                res["missing"].append(f"{side} arm")
        if shown(f"hand_{side}"):
            res["hands_shown"] += 1
            hp = (j2[f"hand_{side}"][0], j2[f"hand_{side}"][1])
            if any(near(hp, q, tol * 1.2) for q in (m.get("l_wr"), m.get("r_wr"), m.get("l_hand"), m.get("r_hand"))):
                res["hands_found"] += 1
            else:
                res["missing"].append(f"{side} hand")
    res["arms_ok"] = res["arms_found"] == res["arms_shown"]
    res["hands_ok"] = res["hands_found"] == res["hands_shown"]
    return res


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: checks run on a card, never on the CPU")
    cfg = json.loads(Path(sys.argv[1]).read_text())
    dev = torch.device("cuda")
    t0 = time.time()
    pose = Pose(cfg["pose_model"], dev)
    if cfg.get("selftest"):                      # the torch graph against onnxruntime's output on one input
        z = np.load(cfg["selftest"])
        sx, sy = pose.g({"input": torch.from_numpy(z["x"]).to(dev)})
        print(json.dumps({"selftest_max_abs": [float((sx.cpu() - torch.from_numpy(z["sx"])).abs().max()),
                                               float((sy.cpu() - torch.from_numpy(z["sy"])).abs().max())]}), flush=True)
    judge = None
    if cfg.get("judge", True):
        import vl_judge as VJ
        judge = VJ.Judge("cuda")
    print(json.dumps({"phase": "load", "s": round(time.time() - t0, 1)}), flush=True)
    out = {}
    cache = {}
    for it in cfg["items"]:
        r = {"people": {}}
        imgs = {"redraw": load(it["redraw"], dev)}
        for kk in ("engine", "depth"):
            if it.get(kk):
                imgs[kk] = cache.setdefault(it[kk], load(it[kk], dev))
        for p in it["people"]:
            if (p.get("body_to_camera_deg") or 0) > 100 or not p.get("box"):
                continue
            b = p["box"]
            if (b[2] - b[0]) * (b[3] - b[1]) < 0.004:
                continue
            bp = [b[0] * 1152, b[1] * 640, b[2] * 1152, b[3] * 640]
            q = {}
            for kk, im in imgs.items():
                k, s = pose(im, bp)
                m = measures(k, s, 1152, 640)
                ah = arm_hand(m, p.get("joints2d") or {}, b)
                q[kk] = dict(ah, neck_ratio=m.get("neck_ratio"))
            if judge is not None and p.get("sheet") and p.get("wear") and (p.get("head_to_camera_deg") or 0) < 100:
                j2 = p.get("joints2d") or {}
                if "chin" in j2 and j2["chin"][2] or ("chin" in j2 and 0 <= j2["chin"][1] < 1):
                    W, H = 1152, 640
                    im = Image.open(it["redraw"]).convert("RGB").resize((W, H))
                    top = max(0, int((b[1] - 0.02) * H))
                    cy = j2["chin"][1]
                    bot = min(H, int(min(b[3], cy + (cy - b[1]) * 2.2 + 0.1) * H))
                    c = im.crop((max(0, int((b[0] - 0.02) * W)), top, min(W, int((b[2] + 0.02) * W)), max(top + 16, bot)))
                    a = judge.ask_images([Image.open(p["sheet"]).convert("RGB"), c],
                                         OUTFIT_Q.format(wear=p["wear"]), 24, 768).strip()
                    q["outfit"] = a
                    q["outfit_ok"] = a.lower().startswith("same")
            r["people"][p["key"]] = q
        out[it["id"]] = r
        print(json.dumps({"id": it["id"], "people": {k: {"arms": (v["redraw"]["arms_found"], v["redraw"]["arms_shown"]),
                                                          "hands": (v["redraw"]["hands_found"], v["redraw"]["hands_shown"]),
                                                          "neck": v["redraw"]["neck_ratio"],
                                                          "neck_engine": v.get("engine", {}).get("neck_ratio"),
                                                          "outfit": v.get("outfit")}
                                                      for k, v in r["people"].items()}}), flush=True)
    # neck: against the engine still and the character's own median in this run
    by_char = {}
    for iid, r in out.items():
        for k, v in r["people"].items():
            if v["redraw"].get("neck_ratio") is not None:
                by_char.setdefault(k, []).append(v["redraw"]["neck_ratio"])
    med = {k: statistics.median(v) for k, v in by_char.items()}
    for iid, r in out.items():
        for k, v in r["people"].items():
            n = v["redraw"].get("neck_ratio")
            e = v.get("engine", {}).get("neck_ratio")
            if n is None:
                v["neck_ok"] = None
                continue
            refs = [x for x in (e, med.get(k)) if x]
            v["neck_ok"] = all(n <= x * NECK_TOL and n >= x / 1.6 for x in refs)
            v["neck_refs"] = {"engine": e, "character_median": round(med.get(k, 0), 3)}
    Path(cfg["out"]).write_text(json.dumps(out, indent=1))
    print(json.dumps({"done": len(out), "s": round(time.time() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
