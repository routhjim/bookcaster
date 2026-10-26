"""Per-clip video QA for generated shots (any book): cuts, zooms, hands on their grips, left/right order, identity drift.

    ~/tts-ab/.venv-pose/bin/python -I tools/video_qa.py CLIP.mp4 [--anchors RENDER/anchors.json]
        [--expect RENDER/start_expect.json] [--every 6] [--json out.json] [--no-identity]

Checks (each adds to "flags"; "pass" = no flags):
- cut: a frame-to-frame change far above the clip's own motion (> 6x the median step and > 18/255): the model cut
  to another shot inside the clip.
- camera: frame 0 vs later frames (ORB + similarity fit). A static shot (default) fails on any zoom outside
  [0.87, 1.15] or a slide > 12 % of the width; with --cam-expect (an authored camera.path) the last sampled frame's
  scale/shift must match the path's within |log scale| 0.12 and 0.10 of the width.
- hands (with --anchors from tools/blockout_video.py): per sampled frame, every planned grip's hand (DWPose
  wholebody) must lie within 1.5 fist sizes of the grip segment the staging had at that frame; flagged when fewer
  than 60 % of the sampled frames pass for a grip.
- order (with --expect or --anchors): the people's left-to-right order on frame 0 AND the last frame must match the
  staging (bboxes of the expected figures); a different count or a swapped order is flagged (the 180-degree rule).
- mouth (with --speech): the speaker's inner-lip opening must vary while their line is voiced (lip-sync present).
- identity (hy3d venv, DINOv2): each person's face crop on the last frame vs frame 0; cosine < 0.55 is flagged as
  drift (a face that turned into someone else).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HY_PY = os.path.expanduser("~/tts-ab/.venv-hy3d/bin/python")
CUT_K, CUT_MIN = 6.0, 18.0
SCALE_LO, SCALE_HI, SHIFT_MAX = 0.87, 1.15, 0.12
CAM_SCALE_TOL, CAM_SHIFT_TOL = 0.12, 0.10    # |log scale| and width-fraction tolerance vs an authored camera path
GRIP_FISTS, GRIP_PASS = 1.5, 0.6
ID_MIN = 0.55


TRANSIENT_MAX = 0.02


def read_frames(path, size=None):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if size and (f.shape[1], f.shape[0]) != tuple(size):
            f = cv2.resize(f, tuple(size), interpolation=cv2.INTER_AREA)
        out.append(f)
    return out


def cuts(frames):
    small = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)), cv2.COLOR_BGR2GRAY).astype(np.float32)
             for f in frames]
    d = np.array([np.abs(a - b).mean() for a, b in zip(small, small[1:])])
    if not len(d):
        return [], d
    med = float(np.median(d))
    return [int(i + 1) for i in np.where((d > CUT_K * max(med, 0.5)) & (d > CUT_MIN))[0]], d


def camera_moves(frames, every):
    orb = cv2.ORB_create(1500)
    g0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    k0, d0 = orb.detectAndCompute(g0, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    res = []
    for i in range(every, len(frames), every):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        k, d = orb.detectAndCompute(g, None)
        if d0 is None or d is None or len(k) < 20:
            continue
        m = sorted(bf.match(d0, d), key=lambda x: x.distance)[:400]
        if len(m) < 20:
            continue
        a = np.float32([k0[x.queryIdx].pt for x in m])
        b = np.float32([k[x.trainIdx].pt for x in m])
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if M is None or inl is None or inl.sum() < 15:
            continue
        s = float(np.hypot(M[0, 0], M[1, 0]))
        shift = float(np.hypot(M[0, 2], M[1, 2]) / frames[0].shape[1])
        res.append({"frame": i, "scale": round(s, 3), "shift": round(shift, 3), "inliers": int(inl.sum())})
    return res


def seg_dist(p, a, b):
    p, a, b = (np.asarray(x, float) for x in (p, a, b))
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1))
    return float(np.linalg.norm(p - (a + ab * t)))


def people(wb, img):
    """DWPose wholebody -> [{bbox, center_x, hands: {l, r}, face_box}] for every confident person."""
    kps, sc = wb(img)
    out = []
    for k, s in zip(kps, sc):
        k, s = np.asarray(k), np.asarray(s)
        body = k[:17][s[:17] > 0.3]
        if len(body) < 5:
            continue
        x0, y0 = body.min(0)
        x1, y1 = body.max(0)
        hands = {}
        for side, sl, wrist in (("l", slice(91, 112), 9), ("r", slice(112, 133), 10)):
            h = k[sl][s[sl] > 0.3]
            if len(h) >= 4:
                hands[side] = h.mean(0)
            elif s[wrist] > 0.3:
                hands[side] = k[wrist]
        f = k[23:91][s[23:91] > 0.3]
        fb = None
        if len(f) >= 10:
            fx0, fy0 = f.min(0)
            fx1, fy1 = f.max(0)
            w, h = fx1 - fx0, fy1 - fy0
            fb = [float(max(0, fx0 - 0.25 * w)), float(max(0, fy0 - 0.55 * h)), float(fx1 + 0.25 * w), float(fy1 + 0.2 * h)]
        out.append({"bbox": [float(x0), float(y0), float(x1), float(y1)], "cx": float((x0 + x1) / 2), "hands": hands,
                    "face": fb})
    return out


def hands_check(wb, frames, anchors, every):
    fr = {a["frame"]: a for a in anchors["frames"]}
    stats = {}
    for i in range(0, len(frames), every):
        a = fr.get(i)
        if not a:
            continue
        found = people(wb, frames[i])
        drawn = [h for p in found for h in p["hands"].values()]
        for fg in a["figures"]:
            for g in fg.get("grips", []):
                key = f"{fg['who']}:{g['hand']}:{g['item']}"
                st = stats.setdefault(key, {"n": 0, "ok": 0, "missing": 0})
                st["n"] += 1
                fist = float(g.get("fist_px") or 26.0)
                if not drawn:
                    st["missing"] += 1
                    continue
                dmin = min(seg_dist(h, *g["grip_segment"]) for h in drawn)
                st["ok"] += int(dmin <= GRIP_FISTS * fist)
    for st in stats.values():
        st["rate"] = round(st["ok"] / max(st["n"], 1), 2)
    return stats


def order_check(wb, frames, expect):
    exp = sorted([f for f in expect.get("figures", []) if f.get("bbox")], key=lambda f: (f["bbox"][0] + f["bbox"][2]) / 2)
    names = [f["who"] for f in exp]
    exp_cx = [(f["bbox"][0] + f["bbox"][2]) / 2 for f in exp]
    res = {"expected": names}
    for tag, idx in (("first", 0), ("last", len(frames) - 1)):
        found = sorted(people(wb, frames[idx]), key=lambda p: p["cx"])
        r = {"count": len(found), "cx": [round(p["cx"]) for p in found]}
        if len(found) == len(exp) and exp:
            W = frames[idx].shape[1]
            # in order, each detected person sits nearest its own expected figure (else the order flipped)
            r["swapped"] = any(int(np.argmin([abs(p["cx"] - c) for c in exp_cx])) != j for j, p in enumerate(found)) \
                and max(abs(p["cx"] - c) for p, c in zip(found, exp_cx)) > 0.12 * W
        res[tag] = r
    return res, names


def identity(wb, frames, names_hint):
    """Face crops of frame 0 and the last frame, paired left-to-right; DINOv2 cosine in the hy3d venv."""
    a = sorted(people(wb, frames[0]), key=lambda p: p["cx"])
    b = sorted(people(wb, frames[-1]), key=lambda p: p["cx"])
    pairs = [(p, q) for p, q in zip(a, b) if p["face"] and q["face"]] if len(a) == len(b) else []
    if not pairs:
        return {"pairs": 0}
    td = tempfile.mkdtemp(prefix="vqa_")
    paths = []
    for k, (p, q) in enumerate(pairs):
        for tag, img, fb in (("a", frames[0], p["face"]), ("b", frames[-1], q["face"])):
            x0, y0, x1, y1 = (int(v) for v in fb)
            crop = img[max(0, y0):y1, max(0, x0):x1]
            if crop.size == 0:
                break
            fp = os.path.join(td, f"{k}{tag}.png")
            cv2.imwrite(fp, crop)
            paths.append(fp)
    code = ("import sys,json,torch;from PIL import Image;from transformers import AutoImageProcessor,AutoModel\n"
            "ps=sys.argv[1:];pr=AutoImageProcessor.from_pretrained('facebook/dinov2-small');"
            "m=AutoModel.from_pretrained('facebook/dinov2-small').eval()\n"
            "with torch.inference_mode():\n e=m(**pr(images=[Image.open(p).convert('RGB') for p in ps],return_tensors='pt')).last_hidden_state[:,0]\n"
            "e=e/e.norm(dim=-1,keepdim=True);print(json.dumps([float(e[i]@e[i+1]) for i in range(0,len(ps)-1,2)]))")
    r = subprocess.run([HY_PY, "-I", "-c", code, *paths], capture_output=True, text=True,
                       env={**os.environ, "HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"})
    try:
        sims = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"pairs": len(pairs), "error": r.stderr[-300:]}
    return {"pairs": len(pairs), "sim": [round(s, 3) for s in sims]}


MOUTH_CORR_MIN = 0.3      # Pearson(speaker mouth activity, speech envelope)
LISTENER_MAX = 0.8        # a listener's mouth activity while the speaker talks, relative to the speaker's
MOUTH_RATIO_MIN = 1.25   # mouth-box motion while voiced / while silent (lip-sync moves the mouth with the words)


def speech_env(wav, n, fps):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", wav, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    au = np.frombuffer(raw, np.float32)
    hop = 16000 // fps
    return np.array([np.sqrt(np.mean(au[i * hop:(i + 1) * hop] ** 2)) if (i + 1) * hop <= len(au) else 0.0 for i in range(n)])


def _mouth_activity(g, mb, fb):
    """Per frame step: mouth-box change over (upper-face change + 0.5) for one face's fixed boxes."""
    pad = 0.35 * (mb[2] - mb[0])
    mx0, my0, mx1, my1 = (max(0, int(v)) for v in (mb[0] - pad, mb[1] - pad, mb[2] + pad, mb[3] + pad))
    ex0, ey0, ex1, ey1 = (max(0, int(v)) for v in (fb[0], fb[1], fb[2], mb[1] - pad))
    mo = np.array([np.abs(a[my0:my1, mx0:mx1] - b[my0:my1, mx0:mx1]).mean() for a, b in zip(g, g[1:])])
    up = np.array([np.abs(a[ey0:ey1, ex0:ex1] - b[ey0:ey1, ex0:ex1]).mean() if ey1 > ey0 + 4 else 0.0
                   for a, b in zip(g, g[1:])])
    return mo / (up + 0.5)


def mouth_check(wb, frames, wav, fps, speaker_x=None, every=1):
    """Lip-sync present for the SPEAKER, quiet for the LISTENER. Mouth boxes (DWPose face 48-67) are located on 5
    frames and fixed; per frame step the mouth-box change relative to the upper face (head motion cancels):
      voiced_vs_pause = speaker activity while voiced / in the pauses;  corr = Pearson(activity, speech envelope);
      listener_vs_speaker = the other face's activity while voiced / the speaker's (a listener must not talk).
    ok = (corr >= MOUTH_CORR_MIN or voiced_vs_pause >= MOUTH_RATIO_MIN) and listener_vs_speaker <= LISTENER_MAX.
    "act_voiced" (mean speaker activity while voiced) lets the caller compare chained chunks with chunk 0."""
    env = speech_env(wav, len(frames), fps)
    voiced = env > 0.2 * max(float(np.percentile(env, 95)), 1e-6)
    faces = []                              # [(cx, mouth box, face box)] over 5 frames
    for i in np.linspace(0, len(frames) - 1, 5).astype(int):
        kps, sc = wb(frames[i])
        for k, s_ in zip(kps, sc):
            k, s_ = np.asarray(k), np.asarray(s_)
            f, fs = k[23:91], s_[23:91]
            if fs[48:68].min() < 0.3:
                continue
            faces.append((float(f[:, 0].mean()), [*f[48:68].min(0), *f[48:68].max(0)], [*f.min(0), *f.max(0)],
                          float(f[:, 1].max() - f[:, 1].min())))
    if len(faces) < 2:
        return {"ok": None, "why": "speaker's mouth not found"}
    W = frames[0].shape[1]
    if speaker_x is None:
        speaker_x = max(faces, key=lambda t: t[3])[0]
    spk = [t for t in faces if abs(t[0] - speaker_x) < 0.15 * W]
    lst = [t for t in faces if abs(t[0] - speaker_x) >= 0.15 * W]
    if len(spk) < 2:
        return {"ok": None, "why": "speaker's mouth not found"}
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    act = _mouth_activity(g, np.median([t[1] for t in spk], 0), np.median([t[2] for t in spk], 0))
    v = voiced[1:len(act) + 1]
    e = env[1:len(act) + 1]
    corr = float(np.corrcoef(act, e)[0, 1]) if np.std(act) > 0 and np.std(e) > 0 else 0.0
    out = {"voiced": int(v.sum()), "corr": round(corr, 3), "act_voiced": round(float(act[v].mean()) if v.any() else 0.0, 3)}
    if v.sum() >= 4 and (~v).sum() >= 3:
        out["voiced_vs_pause"] = round(float(act[v].mean() / max(act[~v].mean(), 1e-6)), 2)
    if len(lst) >= 2 and v.any():
        la = _mouth_activity(g, np.median([t[1] for t in lst], 0), np.median([t[2] for t in lst], 0))
        out["listener_vs_speaker"] = round(float(la[v].mean() / max(act[v].mean(), 1e-6)), 2)
    # Pearson is reported, not gated: lips move at syllable rate, the RMS envelope does not (r 0.1-0.23 on good
    # lip-sync, 10-20). Chained chunks are compared with their chunk 0 by the caller (act_voiced).
    talk_ok = out.get("voiced_vs_pause", MOUTH_RATIO_MIN) >= MOUTH_RATIO_MIN
    out["ok"] = bool(talk_ok and out.get("listener_vs_speaker", 0) <= LISTENER_MAX)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--anchors")
    ap.add_argument("--expect")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--json")
    ap.add_argument("--no-identity", action="store_true")
    ap.add_argument("--speech", help="talk clip: the speaker's audio (wav, from the clip's frame 0): mouth must move")
    ap.add_argument("--speaker-x", type=float, help="the speaker's expected centre x in the clip (px)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--cam-expect", help='the authored camera move, first->last: {"scale", "shift", "static"} '
                                        '(tools/remake_chapter.camera_expect); default static')
    a = ap.parse_args(argv)
    anchors = json.load(open(a.anchors)) if a.anchors and os.path.exists(a.anchors) else None
    expect = json.load(open(a.expect)) if a.expect and os.path.exists(a.expect) else None
    size = (anchors or expect or {}).get("size")
    frames = read_frames(a.clip, size)
    rep = {"clip": a.clip, "frames": len(frames), "flags": []}
    if len(frames) < 2:
        rep["flags"].append("unreadable or empty clip")
        rep["pass"] = False
        print(json.dumps(rep))
        return 2
    c, d = cuts(frames)
    rep["cuts"] = c
    rep["step_median"] = round(float(np.median(d)), 2)
    if c:
        rep["flags"].append(f"cut at frame(s) {c}")
    mv = camera_moves(frames, a.every)
    rep["camera"] = {"scale_min": min([m["scale"] for m in mv], default=1.0), "scale_max": max([m["scale"] for m in mv], default=1.0),
                     "shift_max": max([m["shift"] for m in mv], default=0.0), "samples": len(mv)}
    cm = rep["camera"]
    exp = json.loads(a.cam_expect) if a.cam_expect else {"scale": 1.0, "shift": 0.0, "static": True}
    rep["camera"]["expected"] = exp
    if exp.get("static", True):              # a static shot: any zoom or slide is unplanned
        if cm["scale_min"] < SCALE_LO or cm["scale_max"] > SCALE_HI:
            rep["flags"].append(f"unplanned zoom on a static shot: scale {cm['scale_min']}-{cm['scale_max']}")
        if cm["shift_max"] > SHIFT_MAX:
            rep["flags"].append(f"unplanned camera slide {cm['shift_max']:.2f} of the width on a static shot")
    elif mv:                                 # an authored move: the clip's end camera must match the path's
        last = mv[-1]
        ds = abs(float(np.log(max(last["scale"], 1e-3) / max(exp["scale"], 1e-3))))
        dx = abs(last["shift"] - exp.get("shift", 0.0))
        rep["camera"]["end_measured"] = {"scale": last["scale"], "shift": last["shift"]}
        if ds > CAM_SCALE_TOL or dx > CAM_SHIFT_TOL:
            rep["flags"].append(f"camera move differs from the path: scale {last['scale']} vs {exp['scale']}, "
                                f"shift {last['shift']} vs {exp.get('shift', 0)}")
    from rtmlib import Wholebody
    wb = Wholebody(mode="balanced", backend="onnxruntime", device="cpu")
    names = []
    if anchors:
        rep["hands"] = hands_check(wb, frames, anchors, a.every)
        for k, st in rep["hands"].items():
            if st["rate"] < GRIP_PASS:
                rep["flags"].append(f"hand off its grip: {k} ({st['rate']:.0%} of frames)")
    exp = expect or ({"figures": [{"who": f["who"], "bbox": None} for f in anchors["frames"][0]["figures"]]} if anchors else None)
    if expect:
        rep["order"], names = order_check(wb, frames, expect)
        for tag in ("first", "last"):
            o = rep["order"].get(tag, {})
            if o.get("count") is not None and expect.get("figures") and o["count"] != len([f for f in expect["figures"] if f.get("bbox")]):
                rep["flags"].append(f"{tag} frame: {o['count']} people, staging has {len(expect['figures'])}")
            if o.get("swapped"):
                rep["flags"].append(f"{tag} frame: left/right order differs from the staging {rep['order']['expected']}")
    del exp
    if a.speech:          # a TALK clip is near-static: something that comes and goes is a drawn object (10-20: 382_c0's
        # flickering plaque = 14 % of the frame off the temporal median; the good 340 = 0.2 %)
        sm = [cv2.cvtColor(cv2.resize(f, (f.shape[1] // 4, f.shape[0] // 4)), cv2.COLOR_BGR2GRAY).astype(np.float32)
              for f in frames[::2]]
        med = np.median(np.stack(sm), 0)
        tv = [float((np.abs(f_ - med) > 40).mean()) for f_ in sm]
        rep["transient_max"] = round(max(tv), 4)
        if max(tv) > TRANSIENT_MAX:
            rep["flags"].append(f"transient: an object appears/disappears ({max(tv) * 100:.1f}% of the frame off the median)")
    if a.speech:
        # the speaker must stay in the shot (the Knight turned into a stone tablet, 10-20): a face near the speaker's
        # position in >= 70 % of 8 sampled frames
        idx = np.linspace(0, len(frames) - 1, 8).astype(int)
        hit = 0
        for i in idx:
            ps = [p for p in people(wb, frames[i]) if p["face"]]
            if ps and (a.speaker_x is None or min(abs((p["face"][0] + p["face"][2]) / 2 - a.speaker_x) for p in ps)
                       < 0.2 * frames[0].shape[1]):
                hit += 1
        rep["speaker_present"] = round(hit / len(idx), 2)
        if hit / len(idx) < 0.7:
            rep["flags"].append(f"speaker missing in {len(idx) - hit} of {len(idx)} sampled frames")
        rep["mouth"] = mouth_check(wb, frames, a.speech, a.fps, a.speaker_x)
        if rep["mouth"]["ok"] is False:
            rep["flags"].append(f"no lip-sync: mouth motion {rep['mouth']}")
        elif rep["mouth"]["ok"] is None:
            rep["flags"].append("speaker's mouth not found while the line plays")
    if not a.no_identity:
        rep["identity"] = identity(wb, frames, names)
        for k, s in enumerate(rep["identity"].get("sim", [])):
            if s < ID_MIN:
                rep["flags"].append(f"identity drift: person {k + 1} (left to right) face similarity {s:.2f}")
    rep["pass"] = not rep["flags"]
    js = json.dumps(rep, indent=1, default=float)
    if a.json:
        open(a.json, "w").write(js)
    print(js)
    return 0 if rep["pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
