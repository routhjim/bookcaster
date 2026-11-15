"""Face proportions from the MetaHuman face tracker's contour curves (char_render.py "track" output), as ratios to
the eye span, so a render and a reference picture of any size compare directly.

    python face_ratios.py track_result.json --ref NAME [NAME ...]

Prints one row per tracked picture: eye span (px) and the ratios below, and for every picture the mean absolute
difference from the reference (in percent of the reference value). Stdlib only, milliseconds.

    mouth_w      mouth corner to corner / eye span
    mouth_drop   eye line to mouth line (lip seam) / eye span
    nose_drop    eye line to the top of the nasolabial folds (nose wings) / eye span
    nose_w       nasolabial top points apart / eye span
    lip_h        top of the upper lip to bottom of the lower lip / eye span
    eye_w        mean eye width (corner to corner) / eye span
    eye_open     mean eye opening (upper lid to lower lid at the middle) / eye span
"""

import argparse
import json


def xs(p):
    return [q[0] for q in p]


def ys(p):
    return [q[1] for q in p]


def ratios(c):
    def eye(s):
        pts = c[f"crv_eyelid_upper_{s}"] + c[f"crv_eyelid_lower_{s}"]
        cx, cy = sum(xs(pts)) / len(pts), sum(ys(pts)) / len(pts)
        w = max(xs(pts)) - min(xs(pts))
        up = min(c[f"crv_eyelid_upper_{s}"], key=lambda q: abs(q[0] - cx))
        lo = min(c[f"crv_eyelid_lower_{s}"], key=lambda q: abs(q[0] - cx))
        return cx, cy, w, abs(lo[1] - up[1])
    lx, ly, lw, lo = eye("l")
    rx, ry, rw, ro = eye("r")
    E = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    ey = (ly + ry) / 2
    lips = sum((c[k] for k in c if k.startswith("crv_lip_") and "outer" in k), [])
    seam = sum((c[k] for k in c if k.startswith("crv_lip_") and "inner" in k), [])
    mw = max(xs(lips)) - min(xs(lips))
    my = sum(ys(seam)) / len(seam)
    nl, nr = c["crv_nasolabial_l"], c["crv_nasolabial_r"]
    tl, tr = min(nl, key=lambda q: q[1]), min(nr, key=lambda q: q[1])
    return {"E_px": round(E, 1), "mouth_w": mw / E, "mouth_drop": (my - ey) / E,
            "nose_drop": ((tl[1] + tr[1]) / 2 - ey) / E, "nose_w": abs(tl[0] - tr[0]) / E,
            "lip_h": (max(ys(lips)) - min(ys(lips))) / E, "eye_w": (lw + rw) / 2 / E, "eye_open": (lo + ro) / 2 / E}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("track")
    ap.add_argument("--ref", nargs="+", required=True)
    a = ap.parse_args()
    t = json.load(open(a.track))
    t = t.get("track", t)
    rows = {}
    for name, v in t.items():
        try:
            rows[name] = ratios(v["curves"])
        except Exception as e:                     # no face found, or a curve missing
            rows[name] = {"error": str(e)}
    keys = [k for k in next(r for r in rows.values() if "error" not in r) if k != "E_px"]
    print("name".ljust(10) + "E_px".rjust(7) + "".join(k.rjust(11) for k in keys)
          + "".join(f"  d_{r}%".rjust(10) for r in a.ref))
    for name, r in rows.items():
        if "error" in r:
            print(name.ljust(10), r["error"])
            continue
        line = name.ljust(10) + f"{r['E_px']:7.1f}" + "".join(f"{r[k]:11.3f}" for k in keys)
        for ref in a.ref:
            R = rows[ref]
            d = sum(abs(r[k] - R[k]) / abs(R[k]) for k in keys) / len(keys) * 100
            line += f"{d:10.1f}"
        print(line)


if __name__ == "__main__":
    main()
