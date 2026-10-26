"""Locate a set's OWN fixed objects (an anvil, a workbench, a door baked into a HY-World) in 3D (any world).

    ~/tts-ab/.venv-pose/bin/python -I tools/world_locate.py <world_dir> anvil workbench bellows door ...

The vision model (tools/vl_judge.py via characters.vl_ask, the iGPU) boxes each object in the world's panorama
(pano.png); the world's per-pixel 3D points (points.npy, the Blender frame: Z up, floor 0) inside that box give
its position: the median floor position, the top (85th percentile height of its upper half), the extent
(10-90 percentiles). Written to <world_dir>/fixed_props.json: {name: {"at": [x, y], "top": z, "box": [x, y, 0, w, d,
h], "yaw": deg, "pix_box": [...], "points": n}}. The scene layer anchors FIXED props there (the same anvil in
every shot at that location; hidden, the world draws it: it is the support surface and the collider).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import characters as CH  # noqa: E402


def locate(world_dir, names):
    wd = Path(world_dir)
    P = np.load(wd / "points.npy")
    H, W = P.shape[:2]
    q = {n: f"Give the bounding box of the {n} in this panorama picture as four integers x0 y0 x1 y1 on a "
            f"0-1000 scale of the image. If there is no {n}, reply none." for n in names}
    ans = CH.vl_ask([{"id": "pano", "image": str(wd / "pano.png"), "max_new_tokens": 40, "ask": q}])
    out_p = wd / "fixed_props.json"
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for n in names:
        a = str(ans.get("pano", {}).get(n, ""))
        nums = [int(x) for x in re.findall(r"\d+", a)][:4]
        if "none" in a.lower() or len(nums) < 4 or nums[2] <= nums[0] or nums[3] <= nums[1]:
            print(f"{n}: not found ({a!r})")
            continue
        x0, y0, x1, y1 = nums[0] * W // 1000, nums[1] * H // 1000, nums[2] * W // 1000, nums[3] * H // 1000
        # the central 60 % of the box (its edges catch the floor and the wall behind)
        cx0, cx1 = x0 + (x1 - x0) // 5, x1 - (x1 - x0) // 5
        cy0, cy1 = y0 + (y1 - y0) // 5, y1 - (y1 - y0) // 5
        pts = P[cy0:cy1, cx0:cx1].reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) < 20:
            print(f"{n}: too few points")
            continue
        # the object is the NEAR cluster (the wall behind is farther): keep points closer than the median range
        rng = np.linalg.norm(pts[:, :2], axis=1)
        near = pts[rng <= np.percentile(rng, 55)]
        xy = np.median(near[:, :2], axis=0)
        top = float(np.percentile(near[:, 2], 85))
        lo, hi = np.percentile(near, 10, axis=0), np.percentile(near, 90, axis=0)
        w_, d_ = max(float(hi[0] - lo[0]), 0.2), max(float(hi[1] - lo[1]), 0.2)
        yaw = 0.0 if w_ >= d_ else 90.0
        out[n] = {"at": [round(float(xy[0]), 3), round(float(xy[1]), 3)], "top": round(top, 3),
                  "box": [round(float(xy[0]), 3), round(float(xy[1]), 3), 0.0, round(max(w_, d_), 3),
                          round(min(w_, d_), 3), round(top, 3)],
                  "yaw": yaw, "pix_box": [x0, y0, x1, y1], "points": int(len(near)), "source": "vision + pano depth"}
        print(f"{n}: at {out[n]['at']} top {top:.2f} m (box {out[n]['box']})")
    out_p.write_text(json.dumps(out, indent=1))
    return out


ARCH = ("door", "window", "gate", "hearth", "shutters", "shutter", "doorway", "fireplace")


def anchor_arch(world_dir, names=None):
    """ARCHITECTURAL objects of the set (a door, a window, a hearth) are part of a WALL: from the located pano box
    (fixed_props.json pix_box) and the per-pixel 3D points, fit the wall plane (RANSAC on the central points, a
    vertical plane), take the opening's outline on it (width along the wall from the box columns, the floor to
    the top) -> world.json "anchors": {name: {"at": [x, y, 0] (bottom centre on the wall), "normal": [nx, ny, 0]
    (into the room, toward the capture point), "tangent", "width", "height", "lintel_z", "yaw" (deg, the prop's
    +X along the wall)}}. The scene layer uses it instead of a free-standing prop."""
    wd = Path(world_dir)
    P = np.load(wd / "points.npy")
    fp = json.loads((wd / "fixed_props.json").read_text()) if (wd / "fixed_props.json").exists() else {}
    wj = json.loads((wd / "world.json").read_text())
    cap = np.array((wj.get("capture_camera") or {}).get("pos", [0, 0, 1.6])[:2], float)
    anchors = wj.get("anchors") or {}
    rng_ = np.random.default_rng(0)
    for n in names or [k for k in fp if not k.startswith("_") and any(a in k for a in ARCH)]:
        hit = fp.get(n)
        if not hit or "pix_box" not in hit:
            print(f"{n}: not located (run world_locate.py first)")
            continue
        x0, y0, x1, y1 = hit["pix_box"]
        pts = P[y0:y1, x0:x1].reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0.1)]
        if len(pts) < 50:
            continue
        xy = pts[:, :2]
        best, bn = None, -1
        for _ in range(300):                               # a vertical plane: a 2D line through two points
            a, b = xy[rng_.choice(len(xy), 2, replace=False)]
            t = b - a
            if np.linalg.norm(t) < 0.2:
                continue
            nrm = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            inl = np.abs((xy - a) @ nrm) < 0.05
            if inl.sum() > bn:
                best, bn = (a, nrm, inl), int(inl.sum())
        a, nrm, inl = best
        q = xy[inl]
        c0 = q.mean(0)
        # full_matrices=False: the full U is N x N (10-22 v3: 83k wall points = a 52 GB array, the job OOM-killed)
        u, s_, vt = np.linalg.svd(q - c0, full_matrices=False)     # refine on the inliers
        tan = vt[0]
        nrm = np.array([-tan[1], tan[0]])
        if (cap - c0) @ nrm < 0:
            nrm = -nrm
        # the OUTLINE from the box edges themselves (the box hugs the opening): the plane points on its top rows
        # give the lintel, on its side columns the jambs
        B = P[y0:y1, x0:x1]
        def on_plane(blk):
            b_ = blk.reshape(-1, 3)
            b_ = b_[np.isfinite(b_).all(1)]
            return b_[np.abs((b_[:, :2] - c0) @ nrm) < 0.08]
        k_ = max(3, (y1 - y0) // 40)
        top_p = on_plane(B[:k_])
        top = float(np.median(top_p[:, 2])) if len(top_p) > 5 else float(np.percentile(pts[inl, 2], 97))
        kx = max(3, (x1 - x0) // 40)
        jl, jr = on_plane(B[:, :kx]), on_plane(B[:, -kx:])
        if len(jl) > 5 and len(jr) > 5:
            al, ar = np.median((jl[:, :2] - c0) @ tan), np.median((jr[:, :2] - c0) @ tan)
            lo, hi = min(al, ar), max(al, ar)
        else:
            along = (q - c0) @ tan
            lo, hi = np.percentile(along, 3), np.percentile(along, 97)
        ctr = c0 + tan * (lo + hi) / 2
        anchors[n] = {"at": [round(float(ctr[0]), 3), round(float(ctr[1]), 3), 0.0],
                      "normal": [round(float(nrm[0]), 4), round(float(nrm[1]), 4), 0.0],
                      "tangent": [round(float(tan[0]), 4), round(float(tan[1]), 4), 0.0],
                      "width": round(float(hi - lo), 3), "height": round(top, 3), "lintel_z": round(top, 3),
                      "yaw": round(float(np.degrees(np.arctan2(tan[1], tan[0]))), 2),
                      "wall_inliers": round(float(inl.mean()), 3), "source": "pano box + depth, wall-plane fit"}
        print(f"{n}: {anchors[n]}")
    wj["anchors"] = anchors
    (wd / "world.json").write_text(json.dumps(wj, indent=1))
    return anchors


def locate_book(book):
    """Every set of the book (stage/assets.json): locate the FIXED prop classes the book's props use (props.json
    entries with mobility fixed, plus every architectural class) that the world's fixed_props.json lacks, then
    fit the architectural anchors. Idempotent (already-located names are skipped)."""
    import props_build as PB
    vd = CH.vdir_of(book)
    aj = vd / "stage" / "assets.json"
    if not aj.exists():
        return {}
    pj = vd / "stage" / "props" / "props.json"
    reg = json.loads(pj.read_text()) if pj.exists() else {}
    fixed = sorted({k.split("__")[0] for k, e in reg.items() if e.get("mobility") == "fixed"})
    arch = sorted({r["base"] for r in PB.CLASSES["physical"] if r.get("architectural")})
    out = {}
    for ent in json.loads(aj.read_text()).get("sets", []):
        wd = Path(ent.get("world_dir") or "")
        if not (wd / "points.npy").exists() or not (wd / "pano.png").exists():
            continue
        have = json.loads((wd / "fixed_props.json").read_text()) if (wd / "fixed_props.json").exists() else {}
        want = [n for n in dict.fromkeys(fixed + arch) if n not in have and f"_{n}_none" not in have]
        if want:
            res = locate(wd, want)
            for n in want:                                # remember a miss so the vision model isn't asked again
                if n not in res:
                    res[f"_{n}_none"] = True
            (wd / "fixed_props.json").write_text(json.dumps(res, indent=1))
        out[ent["where"]] = anchor_arch(wd)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "book":
        print(json.dumps({k: list(v) for k, v in locate_book(sys.argv[2]).items()}))
    elif sys.argv[1] == "--anchors":
        anchor_arch(sys.argv[2], sys.argv[3:] or None)
    else:
        locate(sys.argv[1], sys.argv[2:])
        anchor_arch(sys.argv[1], [n for n in sys.argv[2:] if any(a in n for a in ARCH)] or [])
