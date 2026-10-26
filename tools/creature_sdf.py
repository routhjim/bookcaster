"""A PROCEDURAL creature as ONE filled body: skeleton graph + radius profile -> one implicit volume -> one mesh.

    ~/tts-ab/.venv-pose/bin/python -I tools/creature_sdf.py SPEC.json OUT_DIR

SPEC {"template": "<name in stage_templates/composite_creatures.json>", "height_m": S, overrides...}: joint positions
and radii are fractions of S. Every volume (tapered capsules along bones, ellipsoids for the trunk, claws, horn
cones, dorsal spikes, the wing membranes as thin slabs) is a signed distance field; they are joined by a SMOOTH
minimum (blend radius = blend_frac x the local thickness) so the chest flows into the arms, the hips into the
thighs, the shoulder blades into the wing roots - no separate parts. Marching cubes at voxel_frac x S gives one
watertight surface; tools/creature_sdf_bl.py rigs it. Writes OUT_DIR/body.npz (verts, faces), skeleton.json
(joints in metres, bones, anchors, qa_poses).
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    tpl = json.loads((HERE / "stage_templates" / "composite_creatures.json").read_text())[spec["template"]]
    T = copy.deepcopy(tpl)
    for k, v in spec.items():
        if k == "joints":
            T["joints"].update(v)
        elif k not in ("template",):
            T[k] = v
    return T


def mirror_name(n):
    return n.replace("r_", "l_", 1) if n.startswith("r_") else n.replace("_r_", "_l_")


def joints_m(T):
    S = float(T["height_m"])
    J = {k: np.array(v, float) * S for k, v in T["joints"].items()}
    for k in list(J):
        if k.startswith("r_") or "_r_" in k:
            J[mirror_name(k)] = J[k] * np.array([1, -1, 1])
    return J


def _pt(J, S, a):
    return J[a] if isinstance(a, str) else np.asarray(a, float) * S


def _mir(p):
    return p * np.array([1, -1, 1])


# ------------------------------------------------------------------------------------------------ primitives
def prims(T, J):
    """Every volume as (kind, params, bbox lo, bbox hi) in metres."""
    S = float(T["height_m"])
    out = []

    cur_blend = [None]

    def cap(a, b, ra, rb):
        lo = np.minimum(a, b) - max(ra, rb)
        hi = np.maximum(a, b) + max(ra, rb)
        out.append(("cap", (a, b, ra, rb), lo, hi, cur_blend[0]))

    def ell(c, r):
        out.append(("ell", (c, r), c - r, c + r))

    for v in T["volumes"]:
        cur_blend[0] = v.get("blend")
        reps = [False, True] if v.get("mirror") else [False]
        for m in reps:
            f = _mir if m else (lambda p: p)
            t = v["t"]
            if t == "ellipsoid":
                ell(f(np.asarray(v["c"], float) * S), np.asarray(v["r"], float) * S)
            elif t == "capsule":
                a_, b_ = v["a"], v["b"]
                if m:
                    a_ = mirror_name(a_) if isinstance(a_, str) else a_
                    b_ = mirror_name(b_) if isinstance(b_, str) else b_
                a = _pt(J, S, a_) if isinstance(a_, str) else f(_pt(J, S, a_))
                b = _pt(J, S, b_) if isinstance(b_, str) else f(_pt(J, S, b_))
                cap(a, b, v["r"][0] * S, v["r"][1] * S)
            elif t == "chain":
                P = [J[n] for n in v["joints"]]
                r0, r1 = v["r"][0] * S, v["r"][1] * S
                for i in range(len(P) - 1):
                    t0, t1 = i / (len(P) - 1), (i + 1) / (len(P) - 1)
                    cap(P[i], P[i + 1], r0 + (r1 - r0) * t0, r0 + (r1 - r0) * t1)
            elif t == "claws":
                at = J[mirror_name(v["at"]) if m else v["at"]]
                d = f(np.asarray(v["dir"], float))
                d = d / np.linalg.norm(d)
                side = np.cross(d, [0, 0, 1.0])
                side = side / max(np.linalg.norm(side), 1e-9) if np.linalg.norm(side) > 1e-6 else np.array([0, 1.0, 0])
                n = int(v.get("n", 3))
                for k in range(n):
                    o = (k - (n - 1) / 2) * v["spread"] * S
                    a = at + side * o
                    cap(a, a + d * v["len"] * S - np.array([0, 0, 0.3]) * v["len"] * S * 0.3, v["r"] * S, v["r"] * S * 0.2)
            elif t == "spikes":
                P = [J[n] for n in v["along"]]
                sd = np.asarray(v["side"], float)
                sd = sd / np.linalg.norm(sd)
                for i in range(len(P) - 1):
                    a = (P[i] + P[i + 1]) / 2
                    cap(a, a + sd * v["len"] * S * (1 - 0.5 * i / len(P)), v["r"] * S, v["r"] * S * 0.1)
            elif t == "cut":
                Q = [np.asarray(q, float) * S for q in v["quad"]]
                for (p0, p1, p2) in ((Q[0], Q[1], Q[2]), (Q[0], Q[2], Q[3])):
                    lo = np.minimum.reduce([p0, p1, p2]) - 3 * v["half"] * S
                    hi = np.maximum.reduce([p0, p1, p2]) + 3 * v["half"] * S
                    out.append(("cut", (p0, p1, p2, v["half"] * S, 0.0), lo, hi))
            elif t == "spikes" and v.get("from_surface"):
                P = [J[n] for n in v["along"]]
                sd = np.asarray(v["side"], float)
                sd = sd / np.linalg.norm(sd)
                for i in range(len(P) - 1):
                    a = (P[i] + P[i + 1]) / 2
                    out.append(("spike", (a, sd, v["len"] * S * (1 - 0.5 * i / len(P)), v["r"] * S), a - S, a + S))
            elif t == "membrane":
                names = [mirror_name(n) if m else n for n in v["outline"]]
                poly = [J[n] for n in names]
                fan = J[mirror_name(v["fan"]) if m else v["fan"]]
                th = v["thick"] * S
                tris = []
                for i in range(len(poly)):
                    a, b = poly[i], poly[(i + 1) % len(poly)]
                    if np.allclose(a, fan) or np.allclose(b, fan):
                        continue
                    tris.append((fan, a, b))
                # the trailing edge between fingers is SCALLOPED: each triangle's far edge pulled toward the fan
                for (p0, p1, p2) in tris:
                    sc_ = v.get("scallop", 0.0)
                    lo = np.minimum.reduce([p0, p1, p2]) - th
                    hi = np.maximum.reduce([p0, p1, p2]) + th
                    out.append(("tri", (p0, p1, p2, th / 2, sc_), lo, hi))
    return out


def sd_cap(P, a, b, ra, rb):
    ba = b - a
    L2 = max(float(ba @ ba), 1e-12)
    h = np.clip(((P - a) @ ba) / L2, 0, 1)
    d = np.linalg.norm(P - (a + h[:, None] * ba), axis=1)
    return d - (ra + (rb - ra) * h)


def sd_ell(P, c, r):
    q = (P - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_tri(P, p0, p1, p2, half, scallop):
    """Distance to a triangle (a slab of half-thickness half); the edge p1p2 scalloped (an arc inward)."""
    e0, e1 = p1 - p0, p2 - p0
    n = np.cross(e0, e1)
    n = n / max(np.linalg.norm(n), 1e-12)
    # barycentric projection
    w = P - p0
    d00, d01, d11 = e0 @ e0, e0 @ e1, e1 @ e1
    d20, d21 = w @ e0, w @ e1
    den = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / den
    u = (d00 * d21 - d01 * d20) / den
    # the scallop: the far edge (u + v = 1) moves in by scallop x 4 t (1 - t) of the fan->edge depth
    t = np.clip(u / np.maximum(u + v, 1e-9), 0, 1)
    lim = 1.0 - scallop * 4 * t * (1 - t)
    inside = (u >= 0) & (v >= 0) & (u + v <= lim)
    dplane = np.abs(w @ n)
    # outside: distance to the boundary segments (approximate by the 3 edges)
    def seg(a, b):
        ab = b - a
        h = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        return np.linalg.norm(P - (a + h[:, None] * ab), axis=1)
    dout = np.minimum(np.minimum(seg(p0, p1), seg(p1, p2)), seg(p2, p0))
    d = np.where(inside, dplane, dout)
    return d - half


def build(T):
    S = float(T["height_m"])
    J = joints_m(T)
    PR = prims(T, J)
    vox = float(T.get("voxel_frac", 0.0055)) * S
    lo = np.min([p[2] for p in PR], axis=0) - 4 * vox
    hi = np.max([p[3] for p in PR], axis=0) + 4 * vox
    shp = np.ceil((hi - lo) / vox).astype(int) + 1
    D = np.full(shp, 1e3, np.float32)
    kf = float(T.get("blend_frac", 0.45))
    t0 = time.time()
    def look(p):                                         # the field at a point (nearest voxel)
        i = np.clip(np.round((p - lo) / vox).astype(int), 0, shp - 1)
        return float(D[i[0], i[1], i[2]])
    late = [q for q in PR if q[0] in ("spike", "cut")]
    PR2 = [q for q in PR if q[0] not in ("spike", "cut")]
    for q in late:                                       # spikes grow from the SURFACE (found by marching out)
        if q[0] == "spike":
            a, sd, ln, r = q[1]
            p = a.copy()
            for _ in range(400):
                if look(p) > 0:
                    break
                p = p + sd * vox
            base = p - sd * 0.35 * ln
            tip = p + sd * ln
            PR2.append(("cap", (base, tip, r, r * 0.08), np.minimum(base, tip) - r, np.maximum(base, tip) + r))
    PR = PR2 + [q for q in late if q[0] == "cut"]
    for q in PR:
        kind, par, plo, phi = q[:4]
        kq = q[4] if len(q) > 4 and q[4] is not None else kf
        # the blend radius ~ the local thickness x blend_frac
        thick = {"cap": lambda q: max(q[2], q[3]), "ell": lambda q: float(np.min(q[1])),
                 "tri": lambda q: 3 * q[3], "cut": lambda q: q[3]}[kind](par)
        k = max(kq * thick, 1.5 * vox)
        i0 = np.maximum(np.floor((plo - k - lo) / vox).astype(int), 0)
        i1 = np.minimum(np.ceil((phi + k - lo) / vox).astype(int) + 1, shp)
        gx, gy, gz = [lo[a] + vox * np.arange(i0[a], i1[a]) for a in range(3)]
        G = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), -1).reshape(-1, 3)
        if kind == "cap":
            d = sd_cap(G, *par)
        elif kind == "ell":
            d = sd_ell(G, *par)
        else:
            d = sd_tri(G, *par)
        d = d.reshape(i1 - i0).astype(np.float32)
        sub = D[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        if kind == "cut":                                # SUBTRACT (a sharp cut: the mouth slit)
            D[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = np.maximum(sub, -d)
            continue
        h = np.clip(0.5 + 0.5 * (d - sub) / k, 0, 1)                 # polynomial smooth minimum
        D[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] = d * (1 - h) + sub * h - k * h * (1 - h)
    t1 = time.time()
    from skimage import measure
    V, F, _, _ = measure.marching_cubes(D, level=0.0, spacing=(vox, vox, vox))
    V = V + lo
    t2 = time.time()
    return J, V, F, {"voxel_m": vox, "grid": shp.tolist(), "prims": len(PR), "sdf_s": round(t1 - t0, 1),
                     "mc_s": round(t2 - t1, 1), "verts": int(len(V)), "faces": int(len(F))}


def main(spec_path, out_dir):
    T = load_spec(spec_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    J, V, F, st = build(T)
    np.savez(out / "body.npz", verts=V.astype(np.float32), faces=F.astype(np.int32))
    bones = []
    for b, par, h, t in T["bones"]:
        bones.append([b, par, h, t])
        if b.startswith("r_") or "_r_" in b:
            bones.append([mirror_name(b), mirror_name(par) if par and (par.startswith("r_") or "_r_" in par) else par,
                          mirror_name(h), mirror_name(t)])
    sk = {"height_m": float(T["height_m"]), "joints": {k: v.round(4).tolist() for k, v in J.items()}, "bones": bones,
          "anchors": T.get("anchors", {}), "qa_poses": T.get("qa_poses", {}), "stats": st, "jaw": T.get("jaw"),
          "membranes": [dict(outline=[mirror_name(n) if m else n for n in v["outline"]],
                             fan=mirror_name(v["fan"]) if m else v["fan"], thick=v["thick"])
                        for v in T["volumes"] if v["t"] == "membrane" for m in ([False, True] if v.get("mirror") else [False])],
          "template": "composite:" + json.loads(Path(spec_path).read_text())["template"]}
    (out / "skeleton.json").write_text(json.dumps(sk, indent=1))
    print(json.dumps(st))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
