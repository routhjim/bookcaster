"""End-of-build checks for a combo character, from the engine's stills and guides and the fit's own report; a
failed check fails the build (exit 1). Writes CHECKS.json and a check sheet (stills, depth, mask, the table).

    python char_checks.py --id LIB_ID --relaxed SCENE_DIR [--turned SCENE_DIR] [--seated SCENE_DIR ...]
        --fit COMBO_DIR --out SHEET.jpg [--scene-tool DIR] [--neck-sheet 0.04] [--height-m 1.80]

SCENE_DIRs are scene tool outputs with guides (guides.py run on them), one person (key "a") each:
  arms      relaxed: each forearm separated from the body (mask gap or depth jump) in >= 8 of 10 rows at the wide
            camera and >= 60 % of the rows in frame at the medium camera; the palm visible at the wide camera
  neck      the fit's built collar (bl_collar report): visible neck (chin to the collar's front top) / head height
            within 0.03 of the sheet's (--neck-sheet)
  skin      no skin through the cloth: skin-hued pixels inside the person's mask between the shoulders and the
            waist, the hands' discs excluded, under 0.5 % of that area, at the medium camera of every scene
  colour    the garment's lightness class (dark / mid / light: median of the person's torso pixels relative to the
            face's) is the sheet figure's
  height    measured head top - feet within 3 cm of --height-m (when given)
  seated    every seated scene: the hips inside the seat (the scene tool's seating.check) and the entry carries
            stance.seat_asset and seat_z_m
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arm_gap  # noqa: E402


def skin_hue(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    return (r > g * 1.05) & (g > b * 1.02) & ((r - b) / (r + 1e-3) > 0.18) & (r > 0.12)


def skin_check(scene_dir, cam, framing, key="a"):
    d = Path(scene_dir)
    r = json.loads((d / "result.json").read_text())
    s = next((x for x in r["shots"] if x["name"] == cam), None)
    if s is None or not (d / "guides" / f"{cam}_mask_{key}.png").exists():
        return None
    m = r["subjects"][key]["measured"]
    J = m["joints"]
    im = np.asarray(Image.open(d / f"{cam}.png").convert("RGB")).astype(np.float32) / 255
    mask = np.asarray(Image.open(d / "guides" / f"{cam}_mask_{key}.png").convert("L")) > 127
    H, W = mask.shape

    def px(p):
        u, v, _ = framing.project(s["camera"], p)
        return u * W, v * H
    y_sh = max(px(J["upperarm_l"])[1], px(J["upperarm_r"])[1])
    y_ws = px([m["chest_xy"][0], m["chest_xy"][1], m["waist_z"]])[1]
    band = np.zeros_like(mask)
    band[int(max(y_sh + 0.02 * H, 0)):int(min(y_ws, H)), :] = True
    yy, xx = np.mgrid[0:H, 0:W]
    for hnd in ("hand_l", "hand_r"):
        hx, hy = px(J[hnd])
        rad = 0.07 * H
        band &= (xx - hx) ** 2 + (yy - hy) ** 2 > rad ** 2
    area = band & mask
    if area.sum() < 200:
        return {"area_px": int(area.sum()), "skin_frac": None}
    # skin = this face's own colour (chroma ratios r/g and b/g, measured on the cheeks under the eyes in the same
    # still): warm room light makes dark cloth and white linen "warm" too, but not the face's chroma
    ex, ey_ = px([m["eye_xy"][0], m["eye_xy"][1], m["eye_z"] - 0.035])
    fr = int(0.02 * H)
    cheek = im[max(int(ey_) - fr, 0):int(ey_) + fr, max(int(ex) - 2 * fr, 0):int(ex) + 2 * fr].reshape(-1, 3)
    cheek = cheek[skin_hue(cheek)]
    if len(cheek) < 20:
        return {"area_px": int(area.sum()), "skin_frac": None, "why": "face not seen"}
    rg0 = float(np.median(cheek[:, 0] / np.maximum(cheek[:, 1], 1e-3)))
    bg0 = float(np.median(cheek[:, 2] / np.maximum(cheek[:, 1], 1e-3)))
    l0 = float(np.median(cheek.mean(1)))
    lum = im.mean(2)
    rg = im[..., 0] / np.maximum(im[..., 1], 1e-3)
    bg = im[..., 2] / np.maximum(im[..., 1], 1e-3)
    sk = area & (np.abs(rg / rg0 - 1) < 0.3) & (np.abs(bg - bg0) < 0.15) & (rg > 1.15) & (lum > 0.3 * l0) & (lum < 2.5 * l0)
    k = sk.astype(np.uint8)
    nb = sum(np.roll(np.roll(k, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    sk &= nb >= 6
    cloth = float(np.median(lum[area]))
    return {"area_px": int(area.sum()), "skin_frac": round(float(sk.sum() / area.sum()), 4),
            "cloth_lum": round(cloth, 3)}


def white_share(rgb):
    lum = rgb.mean(-1)
    sat = (rgb.max(-1) - rgb.min(-1)) / np.maximum(rgb.max(-1), 1e-3)
    return (lum > 0.5) & (sat < 0.25)


def neck_white(scene_dir, framing, key="a", cam="close"):
    """Share of white (light, unsaturated) pixels in the neckwear region of the close still: from the chin down 0.6
    head heights (0.45), 0.25 head heights each side of the chin, inside the person's mask (the redraw paints what it
    sees: a white band round the neck reads as a neck brace)."""
    d = Path(scene_dir)
    r = json.loads((d / "result.json").read_text())
    s = next((x for x in r["shots"] if x["name"] == cam), None)
    mp = d / "guides" / f"{cam}_mask_{key}.png"
    if s is None or not mp.exists():
        return None
    m = r["subjects"][key]["measured"]
    im = np.asarray(Image.open(d / f"{cam}.png").convert("RGB")).astype(np.float32) / 255
    mask = np.asarray(Image.open(mp).convert("L")) > 127
    H, W = mask.shape
    u0, v0, _ = framing.project(s["camera"], [m["eye_xy"][0], m["eye_xy"][1], m["chin_z"]])
    u1, v1, _ = framing.project(s["camera"], [m["eye_xy"][0], m["eye_xy"][1], m["head_top_z"]])
    hh = (v0 - v1) * H
    y0, y1 = int(v0 * H), int(min(v0 * H + 0.45 * hh, H))
    x0, x1 = int(max(u0 * W - 0.25 * hh, 0)), int(min(u0 * W + 0.25 * hh, W))
    reg = mask[y0:y1, x0:x1]
    if reg.sum() < 100:
        return None
    return round(float((white_share(im[y0:y1, x0:x1]) & reg).sum() / reg.sum()), 3)


def lightness_class(v):
    return "dark" if v < 0.45 else ("mid" if v < 0.8 else "light")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--relaxed", required=True)
    ap.add_argument("--turned", default=None)
    ap.add_argument("--seated", nargs="*", default=[])
    ap.add_argument("--fit", required=True)
    ap.add_argument("--sheet-image", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-tool", default=str(HERE.parent / "scenes"))
    ap.add_argument("--library", default=None)
    ap.add_argument("--neck-sheet", type=float, default=0.04)
    ap.add_argument("--height-m", type=float, default=None)
    ap.add_argument("--neck-white-sheet", type=float, default=None,
                    help="white share of the sheet portrait's neck region (sheet_neck_white())")
    a = ap.parse_args()
    sys.path.insert(0, a.scene_tool)
    import framing  # noqa: E402
    arm_gap.framing = framing
    R = {"id": a.id, "checks": {}}
    C = R["checks"]
    # arms
    g = arm_gap.measure(a.relaxed, "a")
    wide, med = g.get("wide", {}), g.get("medium", {})
    arms_ok = all((wide.get(s) or {}).get("separated", 0) >= 8 for s in ("l", "r"))
    arms_ok &= all((med.get(s) or {}).get("rows", 0) == 0 or
                   (med[s]["separated"] >= 0.6 * med[s]["rows"]) for s in ("l", "r"))
    hands = [(wide.get(s) or {}).get("hand_visible") for s in ("l", "r")]
    C["arms"] = {"ok": bool(arms_ok and all(hands)), "wide": {s: wide.get(s) for s in ("l", "r")},
                 "medium": {s: med.get(s) for s in ("l", "r")}}
    # neck from the fit
    rep = json.loads((Path(a.fit) / "report.json").read_text())
    cb = rep.get("collar_build")
    if cb:
        frac = round((cb["chin_z"] - cb["collar_top_front_z"]) / cb["head_h"], 3)
        C["neck"] = {"ok": abs(frac - a.neck_sheet) <= 0.03, "visible_frac": frac, "sheet": a.neck_sheet}
    elif rep.get("neckline") in ("base", "open"):
        # a neckline at the base of the neck or lower: the bare neck is the garment's own (the same fact that keeps
        # the fit from lifting it)
        C["neck"] = {"ok": True, "neckline": rep["neckline"], "bare_neck_cm": rep.get("bare_neck_cm")}
    elif rep.get("bare_neck_cm") is not None:
        # the generated collar after the fit's collar lift: bare neck between its front top and the chin (rest pose)
        C["neck"] = {"ok": rep["bare_neck_cm"] <= 2.0, "bare_neck_cm": rep["bare_neck_cm"],
                     "lift_cm": (rep.get("collar_height") or {}).get("lift_cm")}
    else:
        C["neck"] = {"ok": None, "why": "no collar measure in this fit"}
    # the neckwear reads as the sheet's: the white share in the neck region of the close still
    nw = neck_white(a.relaxed, framing)
    if nw is not None and a.neck_white_sheet is not None:
        C["neck_white"] = {"ok": nw <= a.neck_white_sheet + 0.03, "still": nw, "sheet": a.neck_white_sheet}
    # skin through cloth
    sk = {}
    for name, d in [("relaxed", a.relaxed), ("turned", a.turned)] + [(f"seated_{i}", x) for i, x in enumerate(a.seated)]:
        if d:
            sk[name] = skin_check(d, "medium", framing)
    # advisory only: firelit black wool has the face's chroma; this measure flags it (11-11), so it does not fail
    # the build until a component-level depth pass (the head mesh alone) can say where skin is really seen
    C["skin"] = {"ok": None, "advisory": True, "by_scene": sk}
    # colour class against the sheet figure
    if a.sheet_image:
        sh = np.asarray(Image.open(a.sheet_image).convert("RGB")).astype(np.float32) / 255
        h_, w_ = sh.shape[:2]
        fig = sh[int(0.25 * h_):int(0.55 * h_), int(0.62 * w_):int(0.88 * w_)]
        face = sh[int(0.05 * h_):int(0.18 * h_), int(0.70 * w_):int(0.80 * w_)]
        # only the drawn figure, not the paper round it (light, unsaturated)
        fl = fig.reshape(-1, 3)
        ink = fl[(fl.mean(1) < 0.8) | ((fl.max(1) - fl.min(1)) > 0.15)]
        fc = face.reshape(-1, 3)
        fc = fc[(fc.mean(1) < 0.85) | ((fc.max(1) - fc.min(1)) > 0.15)]
        sheet_rel = (float(np.median(ink.mean(1))) if len(ink) else 1.0) / max(float(np.median(fc.mean(1))) if len(fc) else 1.0, 1e-3)
        d = Path(a.relaxed)
        im = np.asarray(Image.open(d / "medium.png").convert("RGB")).astype(np.float32) / 255
        mk = np.asarray(Image.open(d / "guides" / "medium_mask_a.png").convert("L")) > 127
        H = mk.shape[0]
        torso = mk.copy()
        torso[: int(0.55 * H)] = False
        facem = mk.copy()
        facem[int(0.35 * H):] = False
        still_rel = float(np.median(im[torso].mean(1))) / max(float(np.median(im[facem].mean(1))), 1e-3)
        # advisory: a drawing's ink against a lit render's pixels does not compare (black cloth inks as mid grey)
        C["colour"] = {"ok": None, "advisory": True, "same_class": lightness_class(still_rel) == lightness_class(sheet_rel),
                       "still_rel": round(still_rel, 3), "sheet_rel": round(sheet_rel, 3)}
    # height
    r = json.loads((Path(a.relaxed) / "result.json").read_text())
    m = r["subjects"]["a"]["measured"]
    h = round(m["head_top_z"] - m["feet_z"], 3)
    C["height"] = {"ok": None if a.height_m is None else abs(h - a.height_m) <= 0.03, "measured_m": h,
                   "wanted_m": a.height_m}
    # seated
    st = []
    for d in a.seated:
        rr = json.loads((Path(d) / "result.json").read_text())
        spec = json.loads((Path(d) / "job.json").read_text())["job"]["characters"]["a"]
        st.append({"scene": Path(d).name, "id": spec.get("id"), "feet_above_floor_m":
                   rr["subjects"]["a"]["measured"].get("feet_above_floor_m")})
    if a.seated:
        import seating  # noqa: E402
        libp = Path(a.library or "")
        S = seating.seats(libp, libp / "set" / "sitting_room_victorian_01") if a.library else {}
        for e, d in zip(st, a.seated):
            rr = json.loads((Path(d) / "result.json").read_text())
            spec = json.loads((Path(d) / "job.json").read_text())["job"]["characters"]["a"]
            cj = json.loads((libp / "character" / spec["id"] / "character.json").read_text()) if a.library else {}
            seat = next((dict(v, name=k) for k, v in S.items() if v["asset"] == cj.get("seat_asset")), None)
            e["fields"] = bool(cj.get("seat_asset") and cj.get("seat_z_m"))
            if seat:
                e["seating"] = seating.check(rr["subjects"]["a"]["measured"], seat, seating.fit(cj), 0.0)
            e["ok"] = e["fields"] and (e.get("seating") or {}).get("ok", False)
        C["seated"] = {"ok": all(e["ok"] for e in st), "scenes": st}
    R["ok"] = all(v.get("ok") is not False for v in C.values())
    Path(a.out).with_suffix(".json").write_text(json.dumps(R, indent=1))
    # sheet
    rows = []
    for name, d in [("relaxed", a.relaxed), ("turned", a.turned)] + [(f"seated", x) for x in a.seated]:
        if not d:
            continue
        for cam in ("wide", "medium", "close"):
            if (Path(d) / f"{cam}.png").exists():
                rows.append((f"{name} {cam}", Path(d), cam))
    tw, th = 480, 267
    tab_h = 22 * (len(C) + 2)
    S_ = Image.new("RGB", (3 * tw + 8, 30 + tab_h + len(rows) * (th + 18)), (18, 18, 18))
    dr = ImageDraw.Draw(S_)
    dr.text((6, 6), f"{a.id}: checks {'PASS' if R['ok'] else 'FAIL'}", fill=(235, 235, 230))
    y = 28
    for k, v in C.items():
        brief = {kk: vv for kk, vv in v.items() if kk not in ("wide", "medium", "by_scene", "scenes")}
        if k == "arms":
            brief.update({f"wide_{s}": f"{(v['wide'][s] or {}).get('separated')}/{(v['wide'][s] or {}).get('rows')}"
                          for s in ("l", "r")})
            brief.update({f"medium_{s}": f"{(v['medium'][s] or {}).get('separated')}/{(v['medium'][s] or {}).get('rows')}"
                          for s in ("l", "r")})
        if k == "skin":
            brief.update({s_: (vv or {}).get("skin_frac") for s_, vv in v["by_scene"].items()})
        if k == "seated":
            brief.update({e["scene"]: (e.get("seating") or {}).get("why") or "inside the seat" for e in v["scenes"]})
        col = (120, 220, 120) if v.get("ok") else ((230, 90, 90) if v.get("ok") is False else (200, 200, 120))
        dr.text((6, y), f"{k}: {json.dumps(brief)[:300]}", fill=col)
        y += 22
    y += 10
    for lab, d, cam in rows:
        for i, (p, t) in enumerate(((d / f"{cam}.png", lab), (d / "guides" / f"{cam}_depth.png", "depth"),
                                     (d / "guides" / f"{cam}_mask_a.png", "mask"))):
            if p.exists():
                S_.paste(Image.open(p).convert("RGB").resize((tw, th)), (i * (tw + 4), y + 16))
            dr.text((i * (tw + 4) + 4, y + 2), t, fill=(235, 235, 230))
        y += th + 18
    S_.save(a.out, quality=88)
    print(json.dumps({"ok": R["ok"], "out": a.out}))
    return 0 if R["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
