"""Drape one garment layer with the contact solver (run with the solver's own Python, on the iGPU):

    ./ppf-contact-solver python drape_run.py IN.npz OUT.npz

IN.npz (written by bl_outfit.py, step "drape_export"; metres, z up):
    V (n, 3), F (m, 3)            the garment as separate panels (seams split)
    S (k, 2)                      stitch pairs: panel vertex a is sewn to panel vertex b
    pin (p,)                      vertices held in place (waistband, neckline)
    CV (c, 3), CF (d, 3)          fixed collider: the body and every settled layer under this one
    params                        JSON string: young_mod, bend, strain_limit, frames, dt, contact_gap...
OUT.npz: V (n, 3) settled positions (z up), seconds, frames.

The solver is y-up: the script converts on the way in and out. The collider is a static (fully pinned) object, so
each settled layer can be passed in as part of the next layer's collider.
"""

import json
import sys
import time

import numpy as np
from frontend import App


def yup(v):
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def zup(v):
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)


def main():
    t0 = time.time()
    d = np.load(sys.argv[1])
    out = sys.argv[2]
    P = json.loads(str(d["params"]))
    V = yup(d["V"].astype(np.float64))
    F = d["F"].astype(np.int32)
    S = d["S"].astype(np.int64)
    app = App.create(P.get("name", "outfit_drape"))
    app.asset.add.tri("cloth", V, F)
    if len(S):
        ind = np.stack([S[:, 0], S[:, 1], S[:, 1], S[:, 1]], 1)
        w = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (len(S), 1))
        app.asset.add.stitch("seams", (ind, w))
    app.asset.add.tri("collider", yup(d["CV"].astype(np.float64)), d["CF"].astype(np.int32))
    scene = app.scene.create()
    cl = scene.add("cloth")
    cl.param.set("young-mod", P.get("young_mod", 100.0)).set("bend", P.get("bend", 2.0))
    cl.param.set("strain-limit", P.get("strain_limit", 0.05))
    if P.get("contact_gap") is not None:
        cl.param.set("contact-gap", P["contact_gap"])
    if P.get("friction") is not None:
        cl.param.set("friction", P["friction"])
    if len(S):
        cl.stitch("seams")
    if len(d["pin"]):
        cl.pin([int(i) for i in d["pin"]])
    scene.add("collider").pin()
    scene = scene.build()
    session = app.session.create(scene)
    session.param.set("frames", P.get("frames", 60)).set("dt", P.get("dt", 0.01))
    session = session.build()
    t1 = time.time()
    session.start(blocking=True)
    t2 = time.time()
    got = session.get.vertex()
    if got is None:
        raise SystemExit("drape: no output frame")
    vert, frame = got
    n = len(V)
    res = zup(np.asarray(vert, np.float64)[:n])
    np.savez(out, V=res, seconds=t2 - t1, total=time.time() - t0, frames=frame)
    print(json.dumps({"drape_s": round(t2 - t1, 1), "total_s": round(time.time() - t0, 1), "frame": int(frame),
                      "verts": n, "max_move": float(np.abs(res - d["V"]).max())}), flush=True)


if __name__ == "__main__":
    main()
