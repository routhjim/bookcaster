#!/usr/bin/env python3
"""Build a story location ONCE as a 3D set (HY-World 2.0 on ROCm) and render it from any camera.

    python3 tools/world.py build "The Old Mill" --book <book> --image set.png [--kind exterior]
    python3 tools/world.py build "The Old Mill" --out <world dir> --text "..." --kind interior
    python3 tools/world.py build-book ~/audiobooks/<book>            (one set per plan `where`, cached)
    python3 tools/world.py render <world dir | slug --book <book>> \
        --camera '{"pos": [0, -2, 1.6], "look_at": [0, 6, 1.2], "fov": 60}' \
        --out rgb.png --depth depth.png [--mask holes.png] [--size 1536x864]
    python3 tools/world.py info <world>

Any python3 can run the CLI: GPU stages re-exec this file under the HY-World venv
(~/tts-ab/.venv-hyworld) inside `flock /tmp/gpu-<bus>.lock` with HIP_VISIBLE_DEVICES from ~/bin/hip-index.
Independent of bookcaster's modules.

Pipeline (build):
  1. picture: --image, or --text -> Qwen-Image-2.1 draws one establishing picture (tools/qwen_image.py,
     its own venv); --style is appended to the prompt.
  2. panorama (default, --mode pano): HY-Pano 2.0 = Qwen-Image-Edit-2509 (GGUF Q6_K transformer) + the
     HY-Pano LoRA turns the picture into a 360-degree equirectangular panorama (1920x960) that keeps the
     picture as its front view. --mode single skips this (the set is then only what the picture shows).
  3. geometry: pano -> MoGe-2 panorama depth (the HY-World worldgen recipe: 20 icosahedron views, merged);
     single -> WorldMirror 2.0 depth + sky mask, scaled to metres with MoGe-2.
  4. levelled into a Blender frame (below) by a RANSAC ground plane, camera at eye height 1.6 m.
  5. written out: set_mesh.ply (vertex-coloured triangle mesh, Blender: File > Import > PLY, or
     tools/world_blender.py), set_splats.ply (3D Gaussian splats, standard 3DGS layout), pano.png,
     source.png, world.json.

Frame (Blender's): metres, Z up, ground plane z = 0, the capture camera at (0, 0, eye_height) looking
along +Y (the source picture is straight ahead), +X to its right. A render camera is
{"pos": [x,y,z], "look_at": [x,y,z], "fov": horizontal degrees (or "lens": mm on a 36 mm sensor),
 "roll": deg}. render writes <out>.camera.json with the Blender location/rotation_euler (XYZ) so a Blender
scene (tools/blockout*) can use the identical camera.

Depth output: ControlNet style (near = bright, inverse depth normalised over the 2-98% range), plus
<depth>.npy with the raw camera-space Z in metres (inf where empty). --mask writes the holes
(white = nothing there: disocclusion or outside the set) for inpainting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
VENV = Path(os.environ.get("WORLD_VENV", HOME / "tts-ab" / ".venv-hyworld"))
QWEN_VENV = Path(os.environ.get("WORLD_QWEN_VENV", HOME / "tts-ab" / ".venv-qwenimage"))
REPO = Path(os.environ.get("WORLD_REPO", HOME / "src" / "HY-World-2.0"))
MODELS = Path(os.environ.get("WORLD_MODELS", HOME / "models" / "hy-world"))
QIE_BASE = MODELS / "Qwen-Image-Edit-2509"
QIE_GGUF = Path(os.environ.get("WORLD_QIE_GGUF", MODELS / "qie2509-gguf" / "Qwen-Image-Edit-2509-Q6_K.gguf"))
PANO_LORA = MODELS / "HY-World-2.0" / "HY-Pano-2.0"
LIGHTNING = Path(os.environ.get("WORLD_LIGHTNING", MODELS / "lightning" / "Qwen-Image-Edit-2509"
                                / "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"))
WM_DIR = MODELS / "HY-World-2.0"
MOGE = MODELS / "moge-2-vitl-normal"
TOOLS = Path(__file__).resolve().parent
EYE = 1.6


def log(msg: str) -> None:
    sys.stderr.write(f"[world] {msg}\n")
    sys.stderr.flush()


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def hip_index(bus: str) -> str:
    return subprocess.check_output([str(HOME / "bin" / "hip-index"), bus], text=True).strip()


def gpu_run(argv: list[str], bus: str, python: Path | None = None, timeout: int = 1800) -> None:
    """Run `python world.py <argv>` (HY-World venv) on one XTX under its flock (after the fairness gate)."""
    wait_fair()
    py = python or (VENV / "bin" / "python")
    env = dict(os.environ, HIP_VISIBLE_DEVICES=hip_index(bus), PYTHONUNBUFFERED="1",
               MIOPEN_FIND_MODE="FAST")
    # NEVER set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: on TheRock torch 2.11/ROCm 7.13 it
    # silently corrupts results (VAE round-trip error 0.044 -> 0.565, garbage panorama). 2026-10-19.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if argv and argv[0] in SELF_LOCKING:
        # the stage takes the lock itself, after its CPU-side loading (gpu_lock below)
        env["WORLD_GPU_LOCK"] = f"/tmp/gpu-{bus}.lock"
        cmd = [str(py), str(Path(__file__).resolve()), *argv]
    else:
        cmd = ["flock", f"/tmp/gpu-{bus}.lock", str(py), str(Path(__file__).resolve()), *argv]
    log(f"GPU {bus}: {' '.join(shlex.quote(a) for a in argv[:2])} ...")
    subprocess.run(cmd, env=env, check=True, timeout=timeout)


SELF_LOCKING = ("_pano",)
# 10-21 A/B (forge interior + town square): the 8-step Lightning LoRA at 0.7 = ~78 s a panorama vs ~370 s for
# 30 steps at true CFG 7.5, as clean (no handled prop drawn, though Lightning has no negative channel) and coherent;
# at 1.0 it painted a wormy crackle on floors and walls. WORLD_PANO_LIGHTNING_W=0 brings back the 30-step CFG path.
PANO_LIGHTNING_W = "0.7"


class gpu_lock:
    """Take the card's flock inside a stage (path from WORLD_GPU_LOCK, set by gpu_run), after the fairness
    gate; released on exit. A stage run directly (no WORLD_GPU_LOCK) locks nothing."""

    def __enter__(self):
        import fcntl
        path = os.environ.get("WORLD_GPU_LOCK")
        self.f = None
        if path:
            wait_fair()
            self.f = open(path, "a")
            t = time.time()
            fcntl.flock(self.f, fcntl.LOCK_EX)
            log(f"GPU lock {path} taken after {time.time() - t:.0f}s")
        return self

    def __exit__(self, *exc):
        if self.f:
            import fcntl
            fcntl.flock(self.f, fcntl.LOCK_UN)
            self.f.close()
            log("GPU lock released")
        return False


def cpu_run(argv: list[str], python: Path | None = None, timeout: int = 1800) -> None:
    """Run `python world.py <argv>` (HY-World venv) WITHOUT the GPU lock and with no GPU visible: the
    CPU-only stages (panorama depth merge, meshing, background layer, PLY export)."""
    py = python or (VENV / "bin" / "python")
    env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1")
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    log(f"CPU: {' '.join(shlex.quote(a) for a in argv[:2])} ...")
    subprocess.run([str(py), str(Path(__file__).resolve()), *argv], env=env, check=True, timeout=timeout)


# =====================================================================================================
# Geometry helpers (numpy, run inside the venv)
# =====================================================================================================

def _cv_to_blender():
    import numpy as np
    # OpenCV camera/world (x right, y down, z forward) -> Blender (X right, Y forward, Z up)
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)


def _pano_dirs(h: int, w: int):
    """Unit view directions of an equirect pano in the Blender frame: the centre column looks +Y."""
    import numpy as np
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    lon = (u - 0.5) * 2 * np.pi            # 0 at the centre, + to the right
    lat = (0.5 - v) * np.pi                # + up
    lon, lat = np.meshgrid(lon, lat)
    return np.stack([np.cos(lat) * np.sin(lon), np.cos(lat) * np.cos(lon), np.sin(lat)], -1)


def _level_frame(pts, ref_down_mask, eye=EYE):
    """RANSAC a ground plane in points (N,3, Blender frame, camera at origin) from the points flagged
    as below the horizon; return (R, t, info) mapping p -> R @ p + t with the plane at z=0, normal +Z,
    the camera straight above the origin at z = camera height (metres, as the depth gives it)."""
    import numpy as np
    rng = np.random.default_rng(0)
    cand = pts[ref_down_mask]
    info = {"ground_fit": False}
    best = None
    if len(cand) > 500:
        sub = cand[rng.choice(len(cand), min(len(cand), 60000), replace=False)]
        scale = np.median(np.linalg.norm(sub, axis=1))
        thr = 0.02 * scale
        for _ in range(400):
            p = sub[rng.choice(len(sub), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            nn = np.linalg.norm(n)
            if nn < 1e-9:
                continue
            n /= nn
            if n[2] < 0:
                n = -n
            if n[2] < math.cos(math.radians(35)):       # ground must be within 35 deg of level
                continue
            d = -n @ p[0]
            if d <= 0:                                  # camera must be above the plane
                continue
            inl = np.abs(sub @ n + d) < thr
            c = inl.sum()
            if best is None or c > best[0]:
                best = (c, n, d)
        if best is not None and best[0] > 0.15 * len(sub):
            c, n, d = best
            inl = np.abs(sub @ n + d) < thr
            q = sub[inl]
            cen = q.mean(0)
            _, _, vt = np.linalg.svd(q - cen, full_matrices=False)
            n = vt[2] if vt[2][2] > 0 else -vt[2]
            d = -n @ cen
            info = {"ground_fit": True, "inlier_frac": round(float(c / len(sub)), 3),
                    "tilt_deg": round(float(math.degrees(math.acos(min(1, n[2])))), 2)}
    if best is None or not info["ground_fit"]:
        n, d = np.array([0, 0, 1.0]), eye
    # rotation taking n -> +Z with minimal turn (keeps the camera's forward heading)
    z = np.array([0, 0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), n @ z
    if s < 1e-9:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
    height = float(d)                                   # camera distance above the plane
    t = np.array([0, 0, height])
    info["camera_height_raw"] = round(height, 4)
    return R, t, info


def _grid_mesh(P, C, valid, max_edge_ratio=0.04, wrap=False, labels=None, pixel_faces=False):
    """Triangulate an image-shaped point grid P (H,W,3) with colours C (H,W,3 uint8). A quad is dropped
    when any vertex is invalid or an edge is long relative to its distance from the camera origin
    (depth discontinuities = 'rubber sheets'). max_edge_ratio=None keeps every quad (the stretched,
    hole-free variant); then `labels` (H,W ints, e.g. sky=1) still stops quads that span two labels."""
    import numpy as np
    H, W = valid.shape
    idx = np.arange(H * W).reshape(H, W)
    if wrap:
        P = np.concatenate([P, P[:, :1]], 1)
        valid = np.concatenate([valid, valid[:, :1]], 1)
        idx = np.concatenate([idx, idx[:, :1]], 1)
        if labels is not None:
            labels = np.concatenate([labels, labels[:, :1]], 1)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    ok = valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, :-1] & valid[1:, 1:]
    if max_edge_ratio is not None:
        pa, pb, pc, pd = P[:-1, :-1], P[:-1, 1:], P[1:, :-1], P[1:, 1:]
        dist = np.linalg.norm(pa, axis=-1) + 1e-6
        e = np.maximum.reduce([np.linalg.norm(pa - pb, axis=-1), np.linalg.norm(pa - pc, axis=-1),
                               np.linalg.norm(pd - pb, axis=-1), np.linalg.norm(pd - pc, axis=-1)])
        ok &= e / dist < max_edge_ratio
    if labels is not None:
        la = labels[:-1, :-1]
        ok &= (labels[:-1, 1:] == la) & (labels[1:, :-1] == la) & (labels[1:, 1:] == la)
    f1 = np.stack([a[ok], c[ok], b[ok]], -1)
    f2 = np.stack([b[ok], c[ok], d[ok]], -1)
    faces = np.concatenate([f1, f2], 0)
    V = P[:, :W].reshape(-1, 3) if wrap else P.reshape(-1, 3)
    used = np.zeros(H * W, bool)
    used[faces.ravel()] = True
    remap = -np.ones(H * W, np.int64)
    remap[used] = np.arange(used.sum())
    if pixel_faces:                     # faces as pixel indices (for connectivity analysis)
        return faces
    return V[used], C.reshape(-1, 3)[used], remap[faces]


def _write_mesh_ply(path, V, Cols, F):
    import numpy as np
    from plyfile import PlyData, PlyElement
    v = np.empty(len(V), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = V[:, 0], V[:, 1], V[:, 2]
    v["red"], v["green"], v["blue"] = Cols[:, 0], Cols[:, 1], Cols[:, 2]
    f = np.empty(len(F), dtype=[("vertex_indices", "i4", (3,))])
    f["vertex_indices"] = F.astype(np.int32)
    PlyData([PlyElement.describe(v, "vertex"), PlyElement.describe(f, "face")]).write(str(path))


def _pixel_splats(P, C, pix_angle, valid, normals=None):
    """One flat Gaussian per pixel: radius = distance * pixel angle (x1.2 to close gaps)."""
    import numpy as np
    m = valid.reshape(-1)
    means = P.reshape(-1, 3)[m].astype(np.float32)
    cols = C.reshape(-1, 3)[m].astype(np.float32) / 255.0
    dist = np.linalg.norm(means, axis=1)
    pa = pix_angle.reshape(-1)[m] if hasattr(pix_angle, "reshape") else pix_angle
    r = (dist * pa * 0.75).astype(np.float32)
    scales = np.stack([r, r, r * 0.15], 1)
    if normals is not None:
        nrm = normals.reshape(-1, 3)[m]
    else:
        nrm = -means / (dist[:, None] + 1e-9)
    # quaternion rotating +Z to the normal
    z = np.array([0, 0, 1.0], np.float32)
    v = np.cross(np.broadcast_to(z, nrm.shape), nrm)
    w = 1.0 + nrm @ z
    q = np.concatenate([w[:, None], v], 1)
    bad = w < 1e-6
    q[bad] = [0, 1, 0, 0]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    opac = np.ones(len(means), np.float32)
    return means, q.astype(np.float32), scales, opac, cols


def _write_splat_ply(path, means, quats, scales, opac, cols):
    """Standard 3DGS PLY (f_dc = SH DC, log scales, logit opacity, rot wxyz)."""
    import numpy as np
    from plyfile import PlyData, PlyElement
    C0 = 0.28209479177387814
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
             "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    el = np.empty(len(means), dtype=[(n, "f4") for n in names])
    el["x"], el["y"], el["z"] = means.T
    el["nx"] = el["ny"] = el["nz"] = 0
    dc = (cols - 0.5) / C0
    el["f_dc_0"], el["f_dc_1"], el["f_dc_2"] = dc.T
    o = np.clip(opac, 1e-4, 1 - 1e-4)
    el["opacity"] = np.log(o / (1 - o))
    ls = np.log(np.maximum(scales, 1e-7))
    el["scale_0"], el["scale_1"], el["scale_2"] = ls.T
    el["rot_0"], el["rot_1"], el["rot_2"], el["rot_3"] = quats.T
    PlyData([PlyElement.describe(el, "vertex")]).write(str(path))


def _read_splat_ply(path):
    import numpy as np
    from plyfile import PlyData
    v = PlyData.read(str(path))["vertex"]
    C0 = 0.28209479177387814
    means = np.stack([v["x"], v["y"], v["z"]], 1)
    cols = np.clip(np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1) * C0 + 0.5, 0, 1)
    opac = 1 / (1 + np.exp(-np.asarray(v["opacity"])))
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1))
    quats = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], 1)
    return [a.astype(np.float32) for a in (means, quats, scales, opac, cols)]


def _finish_world(out: Path, P, C, valid, sky, pix_angle, wrap, meta, normals=None, down_z=-0.25):
    """Level P (H,W,3 Blender-frame points, camera at origin), scale, write mesh + splats + world.json."""
    import numpy as np
    H, W = valid.shape
    dirs = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-9)
    down = valid & ~sky & (dirs[..., 2] < down_z)
    if meta.get("gravity_aligned"):
        # An equirect panorama is level by construction (horizon = middle row). A RANSAC plane fit here
        # grabs hillsides (a hillside pano: 22 deg tilt, 0.53 m height), so only measure the height
        # straight down: median depth below the camera within ~30 deg of the nadir.
        nadir = valid & ~sky & (dirs[..., 2] < -0.87)
        h = float(np.median(-P[nadir][:, 2])) if nadir.sum() > 200 else EYE
        R, t = np.eye(3), np.array([0, 0, h])
        info = {"ground_fit": "nadir", "nadir_px": int(nadir.sum()), "tilt_deg": 0.0,
                "camera_height_raw": round(h, 4)}
    else:
        R, t, info = _level_frame(P.reshape(-1, 3)[down.reshape(-1)], np.ones(int(down.sum()), bool))
    cam_h = info["camera_height_raw"]
    s = 1.0
    if meta.get("metric"):
        # MoGe-2 depth is metric already; only clamp absurd camera heights to eye level
        if not (0.6 < cam_h < 40):
            s = EYE / cam_h
    else:
        s = EYE / cam_h
    if meta.get("force_eye") or meta.get("gravity_aligned"):
        # painted panoramas are drawn from eye level; MoGe-2's metric scale on them runs 2.5-3x small
        # (0.58 and 0.65 m on two test sets), so always put the camera at EYE
        s = EYE / cam_h
    info["scale"] = round(float(s), 5)
    P = (P.reshape(-1, 3) @ R.T + t).reshape(H, W, 3) * s
    if normals is not None:
        normals = (normals.reshape(-1, 3) @ R.T).reshape(H, W, 3)
    cam_pos = (t * s).tolist()
    # sky: a dome at a fixed far radius around the camera, so it shows from any angle
    if sky.any():
        geo = valid & ~sky
        far = float(np.percentile(np.linalg.norm(P[geo] - np.array(cam_pos), axis=-1), 99.5)) if geo.any() else 100.0
        radius = max(3 * far, 60.0)
        d = (dirs.reshape(-1, 3) @ R.T).reshape(H, W, 3)
        P[sky] = np.array(cam_pos) + d[sky] * radius
        valid = valid | sky
        info["sky_radius_m"] = round(radius, 1)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cam = np.array(cam_pos)
    files = {"mesh": "set_mesh.ply", "mesh_cut": "set_mesh_cut.ply", "splats": "set_splats.ply",
             "points": "points.npy"}
    # set_mesh.ply = cut mesh (depth jumps open) + a BACKGROUND LAYER behind every occluder edge
    # (tools/world_ldi.py), so a moved camera sees continued background, not holes or rubber walls.
    # A fully stretched mesh was tried (10-19): rubber walls up to the camera in exteriors; rejected.
    # set_mesh_cut.ply = the cut mesh alone (clean silhouettes, for compositing).
    from world_ldi import background_layer, floater_pixels
    # floating islands: small mesh components that hover above the ground far from any large component
    # (a stray fragment hung in mid-air in a talk shot, 10-20) are removed from every output
    try:
        pf = _grid_mesh(P - cam, C, valid & ~sky, wrap=wrap, pixel_faces=True)
        drop, fl = floater_pixels(P, pf, cam, sky=sky)
        if drop.any():
            valid = valid & ~drop.reshape(valid.shape)
        info["floaters"] = fl
    except Exception as exc:                         # noqa: BLE001
        log(f"floater cleanup skipped: {exc}")
    Vc, VCc, Fc = _grid_mesh(P - cam, C, valid, wrap=wrap)
    _write_mesh_ply(out / "set_mesh_cut.ply", Vc + cam, VCc, Fc)
    t_l = time.time()
    Pl, Cl, vl, region = background_layer(P, C, cam, valid, sky, wrap=wrap)
    Vl, VCl, Fl = _grid_mesh(Pl - cam, Cl, vl, wrap=wrap)
    V = np.concatenate([Vc, Vl])
    VC = np.concatenate([VCc, VCl])
    F = np.concatenate([Fc, Fl + len(Vc)])
    _write_mesh_ply(out / "set_mesh.ply", V + cam, VC, F)
    info["bg_layer"] = {"region_frac": round(float(region.mean()), 4), "faces": int(len(Fl)),
                        "s": round(time.time() - t_l, 1)}
    means, q, sc, op, co = _pixel_splats(P - cam, C, pix_angle, valid, normals)
    lm = region & ~sky
    if lm.any():
        pa = pix_angle if np.ndim(pix_angle) == 0 else np.where(lm, pix_angle, 0)
        bm, bq, bs, bo, bc = _pixel_splats(Pl - cam, Cl, pa, lm)
        means, q, sc, op, co = [np.concatenate([x, y], 0) for x, y in ((means, bm), (q, bq), (sc, bs), (op, bo), (co, bc))]
    if wrap:
        # backdrop: a closed sphere just outside the set, painted with the panorama in every direction:
        # exteriors -> any ray past the terrain (ridge/sky band) lands on a far background, not black;
        # interiors -> the few seams the background layer leaves show the wall colour behind them.
        far = float(np.percentile(np.linalg.norm(P[valid] - cam, axis=-1), 99.5))
        rad = 1.02 * info["sky_radius_m"] if "sky_radius_m" in info else 1.1 * far
        dd = (_pano_dirs(H, W).reshape(-1, 3) @ R.T).reshape(H, W, 3)[::2, ::2]
        Cd = C[::2, ::2]
        Pd = dd * rad
        allv = np.ones(Pd.shape[:2], bool)
        Vb, VCb, Fb = _grid_mesh(Pd, Cd, allv, wrap=True, max_edge_ratio=None)
        _write_mesh_ply(out / "set_backdrop.ply", Vb + cam, VCb, Fb)
        files["backdrop"] = "set_backdrop.ply"
        bm, bq, bs, bo, bc = _pixel_splats(Pd, Cd, np.full(Pd.shape[:2], 2 * np.pi / Pd.shape[1]) * 1.6, allv)
        means, q, sc, op, co = [np.concatenate([x, y], 0) for x, y in ((means, bm), (q, bq), (sc, bs), (op, bo), (co, bc))]
        info["backdrop_radius_m"] = round(float(rad), 1)
    means = means + np.array(cam_pos, np.float32)
    _write_splat_ply(out / "set_splats.ply", means, q, sc, op, co)
    np.save(out / "points.npy", P.astype(np.float32))
    geo = P[valid & ~sky]
    world = {
        **{k: v for k, v in meta.items() if k not in ("metric", "force_eye", "gravity_aligned")},
        "frame": "Blender: metres, Z up, ground z=0, capture camera looks +Y, +X to its right",
        "capture_camera": {"pos": [round(x, 4) for x in cam_pos],
                           "look_at": [round(cam_pos[0], 4), round(cam_pos[1] + 10, 4), round(cam_pos[2], 4)]},
        "level": info,
        "bounds_min": np.round(np.percentile(geo, 1, axis=0), 2).tolist() if len(geo) else None,
        "bounds_max": np.round(np.percentile(geo, 99, axis=0), 2).tolist() if len(geo) else None,
        "ground": {"plane_z": 0.0, "normal": [0, 0, 1], "eye_height_m": round(float(cam_pos[2]), 4),
                   "method": str(info.get("ground_fit"))},
        "files": files,
        "mesh_faces": int(len(F)), "splats": int(len(means)),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # a single-viewpoint set holds up only near its capture point: from ~15 m away a hillside became an
    # overhanging sheet (10-20). Camera solvers keep cameras within view_region (pano: 6 m, single: 3 m).
    world["view_region"] = {"center": world["capture_camera"]["pos"], "radius_m": 6.0 if wrap else 3.0,
                            "why": "geometry is seen from the capture point only; farther cameras see its back"}
    if meta.get("kind") == "interior":
        try:
            from world_room import room_bounds
            world["room"] = room_bounds(P[valid & ~sky], cam_pos[:2])
        except Exception as exc:                         # noqa: BLE001
            log(f"room bounds skipped: {exc}")
    (out / "world.json").write_text(json.dumps(world, indent=1))
    log(f"wrote mesh ({len(V)} v, {len(F)} f), {len(means)} splats in {time.time() - t0:.1f}s; level {info}")
    return world


# =====================================================================================================
# GPU stages (run inside the venv, under flock)
# =====================================================================================================

def stage_pano(src: str, out: str, prompt: str, seed: int, steps: int, neg_extra: str = "") -> None:
    """HY-Pano 2.0 (Qwen-Image-Edit-2509 + LoRA) on one 24 GB card: encode prompts with the VL encoder,
    free it, then denoise with the GGUF transformer + LoRA."""
    import gc
    import torch
    from PIL import Image
    sys.path.insert(0, str(REPO / "hyworld2" / "panogen"))
    from qwen_image import PanoDiffusionPipeline
    from qwen_image.pipeline_qwen_pano import calculate_dimensions, CONDITION_IMAGE_SIZE
    import pipeline_with_qwen_image as P
    from diffusers import QwenImageTransformer2DModel, GGUFQuantizationConfig

    t0 = time.time()
    dev = "cuda"
    engine = os.environ.get("WORLD_PANO_ENGINE", "int8")          # int8 (fast) | gguf (diffusers GGUF path)
    lightning = os.environ.get("WORLD_PANO_LIGHTNING", "0") not in ("0", "")
    light_w = float(os.environ.get("WORLD_PANO_LIGHTNING_W", "1.0"))
    sys.path.insert(0, str(TOOLS))
    import qie_fast
    torch.backends.cuda.preferred_blas_library("cublaslt")         # hipBLASLt GEMMs: -8% per pass
    fa = qie_fast.install_fa_amd()                                  # Triton flash attention: -37% per pass
    pipe = PanoDiffusionPipeline.from_pretrained(str(QIE_BASE), transformer=None, torch_dtype=torch.bfloat16)
    log(f"encoder + VAE read on the CPU in {time.time() - t0:.1f}s (no GPU lock yet)")
    lock = gpu_lock().__enter__()
    pipe.text_encoder.to(dev)
    pipe.vae.to(dev)
    pipe.vae.enable_tiling()            # full-res VAE encode/decode beside the 17-19 GiB transformer OOMs
    img = Image.open(src).convert("RGB")
    w, h = img.size
    pos = (P.GENERAL_POSITIVE_PREFIX + (prompt or "") + P.GENERAL_POSITIVE_SUFFIX).strip()
    # the props the shots place go in the NEGATIVE prompt (true CFG): named in the positive prompt, even as "do
    # not include an anvil", they get drawn (10-21 A/B: 4/4 anvils with the exclusion sentence, 0/4 without)
    neg = P.GENERAL_NEGATIVE_PROMPT + (f", {neg_extra}" if neg_extra else "")
    cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, w / h)
    cond = pipe.image_processor.resize(img, ch, cw)
    with torch.no_grad():
        pe, pm = pipe.encode_prompt(image=[cond], prompt=pos, device=dev, num_images_per_prompt=1)
        ne, nm = pipe.encode_prompt(image=[cond], prompt=neg, device=dev, num_images_per_prompt=1)
    t_enc = time.time() - t0
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()
    log(f"prompts encoded in {t_enc:.1f}s; loading the {engine} transformer (lightning={lightning}, fa_amd={fa})")
    t1 = time.time()
    if engine == "int8":
        loras = [(PANO_LORA / "pytorch_lora_weights.safetensors", 1.0)]
        if lightning:
            loras.append((LIGHTNING, light_w))
        tr, _st = qie_fast.load_int8_transformer(QIE_GGUF, QIE_BASE, loras=loras, device=dev)
        pipe.register_modules(transformer=tr)
    else:
        if lightning:
            raise SystemExit("Lightning is only wired for WORLD_PANO_ENGINE=int8")
        tr = QwenImageTransformer2DModel.from_single_file(
            str(QIE_GGUF), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=str(QIE_BASE / "transformer"), torch_dtype=torch.bfloat16)
        pipe.register_modules(transformer=tr)
        pipe.transformer.to(dev)
        pipe.load_lora_weights(str(PANO_LORA), weight_name="pytorch_lora_weights.safetensors")
    cfg = 1.0 if lightning else 7.5
    log(f"transformer ready in {time.time() - t1:.1f}s, {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    t2 = time.time()
    with torch.no_grad():
        res = pipe(image=img, prompt=None, prompt_embeds=pe, prompt_embeds_mask=pm,
                   negative_prompt_embeds=ne, negative_prompt_embeds_mask=nm,
                   generator=torch.Generator(device="cpu").manual_seed(seed), true_cfg_scale=cfg,
                   num_inference_steps=steps, guidance_scale=1.0, num_images_per_prompt=1,
                   height=960, width=1952).images[0]
    del pipe, tr
    gc.collect()
    torch.cuda.empty_cache()
    lock.__exit__(None, None, None)             # the rest is CPU work
    pano = P.circular_blend_edges(res, 32)
    pano.save(out)
    stats = {"encode_s": round(t_enc, 1), "denoise_s": round(time.time() - t2, 1), "steps": steps,
             "engine": engine, "lightning": light_w if lightning else 0, "cfg": cfg,
             "total_s": round(time.time() - t0, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    Path(out + ".json").write_text(json.dumps(stats))
    log(f"panorama {pano.size} {stats}")


def _load_moge(dev):
    import torch
    from moge.model.v2 import MoGeModel
    return MoGeModel.from_pretrained(str(MOGE / "model.pt")).to(dev).eval()


def stage_geom_pano_infer(pano_path: str, npz_path: str) -> None:
    """GPU half of the panorama geometry: MoGe-2 on the 20 icosahedron views -> per-view distances and
    masks saved to npz. Only this runs under the GPU lock; the merge and meshing are CPU work
    (stage_geom_pano), which held a shared card for 10+ min while it ran under the lock (10-19)."""
    import cv2
    import numpy as np
    import torch
    import utils3d
    from moge.utils.panorama import get_panorama_cameras, split_panorama_image
    t0 = time.time()
    model = _load_moge("cuda")
    img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    if img.shape[1] > 1920:
        img = cv2.resize(img, (1920, 960), interpolation=cv2.INTER_AREA)
    ex, ins = get_panorama_cameras()
    views = split_panorama_image(img, ex, ins, 512)
    dists, masks = [], []
    with torch.no_grad():
        for i in range(0, len(views), 4):
            x = torch.tensor(np.stack(views[i:i + 4]) / 255, dtype=torch.float32, device="cuda").permute(0, 3, 1, 2)
            fx, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(ins[i:i + 4])))
            o = model.infer(x, fov_x=torch.tensor(fx, dtype=torch.float32, device="cuda"), apply_mask=False)
            dists += list(o["points"].norm(dim=-1).cpu().numpy())
            masks += list(o["mask"].cpu().numpy())
    np.savez(npz_path, dists=np.stack(dists).astype(np.float32), masks=np.stack(masks), infer_s=time.time() - t0)
    log(f"MoGe views inferred in {time.time() - t0:.1f}s -> {npz_path}")


def stage_geom_pano(pano_path: str, out: str, meta_json: str, npz_path: str = "") -> None:
    """MoGe-2 panorama depth (icosahedron split + Poisson merge, as HY-World worldgen does). With npz_path
    (from stage_geom_pano_infer) this is CPU-only: no model, no GPU."""
    import cv2
    import numpy as np
    from moge.utils.panorama import get_panorama_cameras, merge_panorama_depth
    meta = json.loads(meta_json)
    t0 = time.time()
    img = cv2.cvtColor(cv2.imread(pano_path), cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]
    if W > 1920:
        img = cv2.resize(img, (1920, 960), interpolation=cv2.INTER_AREA)
        H, W = 960, 1920
    ex, ins = get_panorama_cameras()
    if not npz_path:
        tmp = str(Path(out) / "moge_views.npz")
        stage_geom_pano_infer(pano_path, tmp)
        npz_path = tmp
    z = np.load(npz_path)
    dists, masks = list(z["dists"]), list(z["masks"])
    t_inf = float(z["infer_s"])
    t0 = time.time()
    depth, mask = merge_panorama_depth(W, H, dists, masks, ex, ins)
    depth = depth.astype(np.float32)
    t_merge = time.time() - t0
    # MoGe's frame -> ours: their pano u=0.5 looks -X (z up); ours looks +Y. Use our own ray table.
    dirs = _pano_dirs(H, W)
    sky = ~mask
    if meta.get("kind") != "interior":
        # MoGe misses dark night skies (they came back as a near flat "lid" over the camera): add skyseg's sky
        # above the horizon
        try:
            from world_sky import sky_mask_equirect
            ss = sky_mask_equirect(img)
            meta["skyseg_added_frac"] = round(float((ss & ~sky).mean()), 4)
            sky = sky | ss
        except Exception as exc:                         # noqa: BLE001 - optional refinement
            log(f"skyseg skipped: {exc}")
    # isolated near-depth specks (candle flames, bright dots placed metres in front of their wall)
    from world_ldi import despeckle_depth
    depth, n_speck = despeckle_depth(depth, valid=np.isfinite(depth) & mask, wrap=True)
    meta.setdefault("timings", {})
    meta["despeckled_px"] = n_speck
    if meta.get("kind") == "interior":
        # interiors have no sky: openings to the outside (windows, doorways showing sky) and MoGe's invalid
        # pixels get depth filled smoothly from the surrounding walls (log-depth push-pull, columns wrapped),
        # so a window sits in its wall like a painted pane. The old median fill put a doorway at the end of
        # a 17 m hall ~6 m from the camera, nearer than the table in front of it (10-19).
        from world_ldi import push_pull_fill
        bad = ~np.isfinite(depth) | sky
        if bad.any() and (~bad).any():
            ld = np.log(np.where(bad, 1.0, np.maximum(depth, 1e-3))).astype(np.float32)
            pw = W // 8
            ldp = np.concatenate([ld[:, -pw:], ld, ld[:, :pw]], 1)
            kp = np.concatenate([~bad[:, -pw:], ~bad, ~bad[:, :pw]], 1)
            depth = np.where(bad, np.exp(push_pull_fill(ldp, kp)[:, pw:pw + W]), depth).astype(np.float32)
        sky[:] = False
    depth[~np.isfinite(depth)] = np.nanmax(depth[np.isfinite(depth)])
    P = dirs * depth[..., None]
    valid = np.isfinite(depth) & (depth > 0)
    pix_angle = np.full((H, W), 2 * np.pi / W) * np.maximum(np.cos(np.arcsin(np.clip(dirs[..., 2], -1, 1))), 0.05)
    pix_angle = np.maximum(pix_angle, 2 * np.pi / W * 0.5)
    meta.update({"geometry": "MoGe-2 panorama depth (metric)", "metric": True, "gravity_aligned": True,
                 "timings": {**meta.get("timings", {}), "moge_infer_s": round(t_inf, 1),
                             "moge_merge_s": round(t_merge, 1)}})
    _finish_world(Path(out), P, img, valid, sky, np.full((H, W), 2 * np.pi / W), True, meta)


def stage_geom_single(img_path: str, out: str, meta_json: str) -> None:
    """WorldMirror 2.0 on the single picture: depth, normals, intrinsics, sky; metric scale from MoGe-2."""
    import cv2
    import numpy as np
    import torch
    sys.path.insert(0, str(REPO))
    from hyworld2.worldrecon.pipeline import WorldMirrorPipeline
    from hyworld2.worldrecon.hyworldmirror.utils.inference_utils import prepare_images_to_tensor
    meta = json.loads(meta_json)
    t0 = time.time()
    wm = WorldMirrorPipeline.from_pretrained(str(WM_DIR), subfolder="HY-WorldMirror-2.0")
    t_load = time.time() - t0
    imgs = prepare_images_to_tensor([img_path], target_size=952, resize_strategy="crop").to("cuda")
    t1 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = wm.model(views={"img": imgs}, cond_flags=[0, 0, 0], is_inference=True)
    torch.cuda.synchronize()
    t_inf = time.time() - t1
    _, _, _, H, W = imgs.shape
    depth = pred["depth"][0, 0].float().squeeze(-1).cpu().numpy()
    K = pred["camera_intrs"][0, 0].float().cpu().numpy()
    nrm = pred["normals"][0, 0].float().cpu().numpy() if "normals" in pred else None
    skyp = pred.get("depth_mask")
    nonsky = (skyp[0, 0].float().squeeze(-1).cpu().numpy() > 0.45) if skyp is not None else np.ones((H, W), bool)
    rgb = (imgs[0, 0].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    del wm, pred
    torch.cuda.empty_cache()
    # metric scale from MoGe-2 (WorldMirror's depth is up to scale)
    moge = _load_moge("cuda")
    with torch.no_grad():
        x = torch.tensor(rgb / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)
        fovx = math.degrees(2 * math.atan(W / 2 / K[0, 0]))
        mo = moge.infer(x, fov_x=fovx)
    md = mo["depth"].float().cpu().numpy()
    good = nonsky & np.isfinite(md) & (md > 0) & (depth > 0)
    ratio = float(np.median(md[good] / depth[good])) if good.sum() > 100 else 1.0
    # camera rays (OpenCV) -> Blender frame
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
    M = _cv_to_blender()
    Pcv = rays * depth[..., None]
    P = Pcv @ M.T
    if nrm is not None:
        nrm = nrm @ M.T
    # drop flying pixels on depth edges
    gy, gx = np.gradient(depth)
    edge = np.hypot(gx, gy) / (depth + 1e-6) > 0.05
    valid = (depth > 0) & ~edge
    sky = ~nonsky
    valid &= ~sky
    pix_angle = 1.0 / K[0, 0] * np.ones((H, W))
    meta.update({"geometry": "WorldMirror 2.0 depth, scale from MoGe-2", "metric": False,
                 "fov_x_deg": round(fovx, 2), "moge_scale": round(ratio, 4),
                 "timings": {**meta.get("timings", {}), "wm_load_s": round(t_load, 1),
                             "wm_infer_s": round(t_inf, 1), "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}})
    P = P * ratio
    meta["metric"] = True
    _finish_world(Path(out), P, rgb, valid, sky, pix_angle, False, meta, normals=nrm, down_z=-0.06)


# =====================================================================================================
# Rendering
# =====================================================================================================

def camera_matrices(cam: dict, W: int, H: int):
    """Blender-frame camera dict -> (w2c 4x4 OpenCV, K 3x3, blender location/rotation)."""
    import numpy as np
    pos = np.array(cam["pos"], float)
    look = np.array(cam.get("look_at", (pos + np.array([0, 1.0, 0]))), float)
    if "lens" in cam:
        fov = 2 * math.degrees(math.atan(18.0 / float(cam["lens"])))
    else:
        fov = float(cam.get("fov", 60.0))
    f = look - pos
    f /= np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.array([1.0, 0, 0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    roll = math.radians(float(cam.get("roll", 0)))
    if roll:
        cr, sr = math.cos(roll), math.sin(roll)
        r, d = cr * r + sr * d, -sr * r + cr * d
    Rc2w = np.stack([r, d, f], 1)
    w2c = np.eye(4)
    w2c[:3, :3] = Rc2w.T
    w2c[:3, 3] = -Rc2w.T @ pos
    fx = W / 2 / math.tan(math.radians(fov) / 2)
    K = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    # Blender camera: looks down local -Z, local +Y up
    Rb = np.stack([r, -d, -f], 1)
    sy = math.hypot(Rb[0, 0], Rb[1, 0])
    if sy > 1e-6:
        ex, ey, ez = math.atan2(Rb[2, 1], Rb[2, 2]), math.atan2(-Rb[2, 0], sy), math.atan2(Rb[1, 0], Rb[0, 0])
    else:
        ex, ey, ez = math.atan2(-Rb[1, 2], Rb[1, 1]), math.atan2(-Rb[2, 0], sy), 0.0
    blender = {"location": pos.round(4).tolist(), "rotation_euler_xyz": [round(ex, 6), round(ey, 6), round(ez, 6)],
               "sensor_fit": "HORIZONTAL", "sensor_width_mm": 36.0,
               "lens_mm": round(18.0 / math.tan(math.radians(fov) / 2), 4), "fov_x_deg": round(fov, 4),
               "resolution": [W, H]}
    return w2c, K, blender


def depth_control(z, valid):
    """ControlNet depth: near = bright. Inverse depth, 2-98 percentile stretch."""
    import numpy as np
    out = np.zeros(z.shape, np.uint8)
    if valid.any():
        inv = 1.0 / np.maximum(z, 1e-3)
        lo, hi = np.percentile(inv[valid], 2), np.percentile(inv[valid], 98)
        n = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1)
        out = (n * 255).astype(np.uint8)
        out[~valid] = 0
    return out


def stage_render(world: str, jobs_json: str) -> None:
    """Render one or more cameras of a world with gsplat (HIP build) in one GPU hold."""
    import numpy as np
    import torch
    from PIL import Image
    from gsplat.rendering import rasterization
    wd = Path(world)
    t0 = time.time()
    means, quats, scales, opac, cols = [torch.from_numpy(a).cuda() for a in _read_splat_ply(wd / "set_splats.ply")]
    t_load = time.time() - t0
    for job in json.loads(jobs_json):
        W, H = job["size"]
        w2c, K, blender = camera_matrices(job["camera"], W, H)
        vm = torch.tensor(w2c, dtype=torch.float32, device="cuda")[None]
        Kt = torch.tensor(K, dtype=torch.float32, device="cuda")[None]
        t1 = time.time()
        rc, ra, _ = rasterization(means, quats, scales, opac, cols, vm, Kt, W, H, render_mode="RGB+ED",
                                  near_plane=0.05, far_plane=1e5, rasterize_mode="antialiased")
        torch.cuda.synchronize()
        dt = time.time() - t1
        rgb = rc[0, ..., :3].clamp(0, 1).cpu().numpy()
        z = rc[0, ..., 3].cpu().numpy()
        a = ra[0, ..., 0].cpu().numpy()
        valid = a > 0.5
        bg = np.array(job.get("background", [0, 0, 0]), np.float32) / 255
        rgb = rgb + (1 - a[..., None]) * bg
        Image.fromarray((rgb * 255).clip(0, 255).astype(np.uint8)).save(job["out"])
        zz = np.where(valid, z, np.inf).astype(np.float32)
        if job.get("depth"):
            Image.fromarray(depth_control(zz, valid)).save(job["depth"])
            np.save(str(Path(job["depth"]).with_suffix(".npy")), zz)
        if job.get("mask"):
            Image.fromarray(((~valid) * 255).astype(np.uint8)).save(job["mask"])
        info = {"camera": job["camera"], "blender": blender, "render_s": round(dt, 3),
                "coverage": round(float(valid.mean()), 4)}
        Path(str(job["out"]) + ".camera.json").write_text(json.dumps(info, indent=1))
        print(json.dumps({"out": job["out"], "render_s": round(dt, 3), "coverage": info["coverage"]}), flush=True)
    log(f"rendered {len(json.loads(jobs_json))} view(s), splats loaded in {t_load:.1f}s")


# =====================================================================================================
# CLI
# =====================================================================================================

BOOKS_ROOT = Path(os.environ.get("WORLD_BOOKS_ROOT", HOME / "audiobooks"))


def world_dir(book: str | None, place_or_path: str) -> Path:
    """A world directory: the path itself if it holds world.json (or is clearly a path), else
    <BOOKS_ROOT>/<book>/work/video/worlds/<slug>. No book is assumed: a slug without --book is an error."""
    p = Path(place_or_path).expanduser()
    if p.is_dir() and (p / "world.json").exists():
        return p
    if not book:
        sys.exit(f"'{place_or_path}' is not a world directory; give its path or --book <book>")
    return BOOKS_ROOT / book / "work" / "video" / "worlds" / slugify(place_or_path)


def book_of(world: Path) -> str | None:
    """The book a world dir belongs to (<root>/<book>/work/video/worlds/<slug>), if laid out that way."""
    parts = Path(world).resolve().parts
    return parts[-5] if len(parts) >= 5 and parts[-4:-1] == ("work", "video", "worlds") else None


def set_picture_prompt(text: str, style: str, kind: str) -> str:
    """The prompt that draws a set's source picture from text.
    Only the style's TECHNIQUE goes in: a book style like "Gothic dark fantasy illustration: ..., cathedral-like
    stone and iron, moody stormy skies" also names CONTENT, which turned a riverbank set into a
    cathedral on a stormy sea (10-19). The clause before ':' (or the whole style if there is none) is kept.
    Naming a thing in a negative or conditional clause still draws it ("(by water: on the bank)" put a lake
    into a yard with no water; "nothing on the tables" drew tables), so the bank wording appears only when the place
    itself is about water."""
    style = (style.split(":", 1)[0].strip() if ":" in (style or "") else (style or "")).rstrip(".")
    # architecture-style words are content too ("Gothic dark fantasy illustration" drew a church for a
    # yard, 10-20): keep the technique, drop the architecture
    style = re.sub(r"(?i)\b(gothic|baroque|romanesque|victorian|byzantine|art deco|art nouveau|rococo|"
                   r"neo-?classical|renaissance|medieval)\b\s*", "", style).strip(" ,")
    full = text.rstrip(".") + "." + (f" Painted as a {style}." if style else "")
    if kind == "interior":
        return full + " Interior view, wide shot of the whole room, eye level."
    watery = re.search(r"(?i)\b(river|stream|lake|sea|shore|pond|brook|harbou?r|beach|water)s?\b", text)
    # a viewpoint, never a viewer: "the viewer standing on the bank" drew a standing man in every seed (10-20)
    return full + (" Wide establishing view at eye level from the grassy riverbank, the water beside the"
                   " bank." if watery else " Wide establishing view at eye level from the ground.")


def draw_picture(prompt: str, out: Path, seed: int, bus: str, size=(1664, 928), refs=()) -> None:
    wait_fair()
    jobs = out.parent / "draw_job.json"
    job = {"id": "set", "prompt": prompt, "seed": seed, "path": str(out), "size": list(size)}
    if refs:
        job["refs"] = [str(r) for r in refs]
    jobs.write_text(json.dumps([job]))
    env = dict(os.environ, HIP_VISIBLE_DEVICES=hip_index(bus))
    cmd = ["flock", f"/tmp/gpu-{bus}.lock", str(QWEN_VENV / "bin" / "python"), "-I",
           str(TOOLS / "qwen_image.py"), str(jobs), str(out.parent)]
    log(f"drawing the source picture with Qwen-Image-2.1 on GPU {bus}")
    subprocess.run(cmd, env=env, check=True, timeout=1800)


def cmd_build(a) -> None:
    if not (a.out or a.book):
        sys.exit("build needs --out <world dir> or --book <book>")
    out = Path(a.out).expanduser().resolve() if a.out else world_dir(a.book, a.place)
    a.book = a.book or book_of(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timings = {}
    src = out / "source.png"
    prompt = a.text or ""
    if a.image:
        from shutil import copyfile
        fp_ = out / "fixed_props.json"          # the old picture's located objects are not this picture's (10-23:
        if fp_.exists() and src.exists() and Path(a.image).expanduser().resolve() != src.resolve():   # the portal)
            fp_.rename(out / f"fixed_props_prev_{time.strftime('%m%d-%H%M%S')}.json")
        copyfile(Path(a.image).expanduser(), src)
    elif a.text:
        full = set_picture_prompt(a.text, a.style, a.kind)
        if src.exists():
            # qwen_image skips a picture that exists (10-21: a rebuilt interior forge kept the old exterior picture)
            src.rename(out / f"source_prev_{time.strftime('%m%d-%H%M%S')}.png")
        fp_ = out / "fixed_props.json"          # the old picture's located objects are not this picture's
        if fp_.exists():
            fp_.rename(out / f"fixed_props_prev_{time.strftime('%m%d-%H%M%S')}.json")
        t = time.time()
        draw_picture(full, src, a.seed, a.bus, refs=[a.ref] if a.ref else ())
        timings["draw_s"] = round(time.time() - t, 1)
    else:
        sys.exit("build needs --image or --text")
    meta = {"place": a.place, "slug": slugify(a.place), "book": a.book, "kind": a.kind, "mode": a.mode,
            "source": "source.png", "text": a.text, "style": a.style, "seed": a.seed}
    if a.mode == "pano":
        pano = out / "pano.png"
        if not pano.exists() or a.redo_pano:
            pp = a.pano_prompt or (prompt or "")
            if a.kind == "interior":
                pp = (pp + " An enclosed interior room: walls, floor and ceiling all around.").strip()
            t = time.time()
            os.environ["WORLD_PANO_ENGINE"] = a.pano_engine
            os.environ["WORLD_PANO_LIGHTNING"] = "1" if a.lightning > 0 else "0"
            os.environ["WORLD_PANO_LIGHTNING_W"] = str(a.lightning)
            gpu_run(["_pano", str(src), str(pano), pp, str(a.seed), str(a.steps), a.negative or ""], a.bus, timeout=1800)
            timings["pano_s"] = round(time.time() - t, 1)
        meta["pano"] = "pano.png"
        meta["timings"] = timings
        t = time.time()
        npz = out / "moge_views.npz"
        gpu_run(["_geom_pano_infer", str(pano), str(npz)], a.bus)
        cpu_run(["_geom_pano", str(pano), str(out), json.dumps(meta), str(npz)])
        # moge_views.npz is kept (~13 MB): later geometry changes re-run CPU-only from it
    else:
        meta["timings"] = timings
        gpu_run(["_geom_single", str(src), str(out), json.dumps(meta)], a.bus)
    w = json.loads((out / "world.json").read_text())
    w.setdefault("timings", {})["build_total_s"] = round(time.time() - t0, 1)
    (out / "world.json").write_text(json.dumps(w, indent=1))
    print(json.dumps({"world": str(out), "timings": w["timings"], "capture_camera": w["capture_camera"]}))



# =====================================================================================================
# Book level: one set per plan `where`
# =====================================================================================================

INTERIOR_CUES = ("inside", "interior", "indoors", "room", "hall", "chamber", "cellar", "crypt", "corridor",
                 "kitchen", "tavern", "inn ", "library", "throne", "bedroom", "cabin", "hut", "shed",
                 "workshop", "dungeon", "staircase", "ceiling", "fireplace", "hearth", "attic", "barn")
EXTERIOR_CUES = ("outside", "outdoors", "valley", "mountain", "field", "forest", "river", "sky", "moon",
                 "landscape", "road", "sea", "shore", "beach", "desert", "garden", "courtyard", "street",
                 "plain", "hill", "meadow", "lake", "cliff", "ridge", "at the door", "doorway", "threshold")


SIG_VERSION = 2            # bump when the build signature's ingredients change (no mass rebuild: see cmd_build_book)
VARIANT_RX = re.compile(r"\s*\((inside|outside)\)\s*$", re.I)   # "The Forge (outside)": a mixed place's other side


def clean_description(desc: str, kind: str, props=()) -> str:
    """The place description as a SET picture needs it (10-21 the forge: "a stone building at the edge of a forest,
    chimney smoking, anvil outside, tools hanging on walls, dim interior lit by firelight" drew the outside, with
    an anvil): clauses naming a story prop (the shots place those) or the other side (outside cues for an
    interior, inside cues for an exterior) are dropped."""
    keep = []
    for cl in re.split(r",|;|\band\b", desc or ""):
        c = cl.strip()
        if not c:
            continue
        # a prop as the clause's NOUN ("anvil outside"), not as an adjective ("a stone well", "stone walls")
        if any(re.search(rf"\b{re.escape(p_.strip().lower())}(s|es)?\b(?!\s+(?!(in|on|at|by|of|with|outside|"
                         rf"inside|hanging|near|under|over|beside|lit|and)\b)[a-z])", c.lower())
               for p_ in props if p_ and p_.strip()):
            continue
        sc = _cue_score(c)
        if (kind == "interior" and sc < 0) or (kind == "exterior" and sc > 0):
            continue
        keep.append(c)
    return ", ".join(keep)


def _cue_score(text: str) -> int:
    t = f" {text.lower()} "
    return sum(t.count(c) for c in INTERIOR_CUES) - sum(t.count(c) for c in EXTERIOR_CUES)


def classify_kind(description: str, prompts: list[str]) -> tuple[str, dict]:
    """interior|exterior for one location: majority vote of its pictures (each picture votes by its cue
    balance), the place description breaking ties (and counting double when there are no pictures)."""
    votes = [_cue_score(p) for p in prompts]
    n_in, n_out = sum(v > 0 for v in votes), sum(v < 0 for v in votes)
    d = _cue_score(description or "")
    if n_in != n_out:
        kind = "interior" if n_in > n_out else "exterior"
    else:
        kind = "interior" if d > 0 else "exterior"
    return kind, {"pictures_interior": n_in, "pictures_exterior": n_out, "description_score": d}


def book_locations(vid: Path) -> dict:
    """where -> {description, prompts[], setting[], chapters[]} over every chapter in <book>/work/video:
    plan.json "pictures" (where + prompt), plan.json "entries" (where only; e.g. a remake's shot list) and
    anim/keyframes.json "frames" (where + prompt + must). Sample dirs are skipped."""
    look = json.loads((vid / "look.json").read_text()) if (vid / "look.json").exists() else {}
    places = look.get("places", {})
    people = list(look.get("characters", {}).keys())
    props = set(prop_names(look, vid))
    locs: dict = {}

    def add(w, chap, prompt="", must=(), who=(), fprops=(), when=None):
        w = (w or "").strip()
        if not w:
            return
        e = locs.setdefault(w, {"description": places.get(w, "") or places.get(VARIANT_RX.sub("", w), ""),
                                "prompts": [], "setting": [], "chapters": [], "when": []})
        if prompt and prompt not in e["prompts"]:
            e["prompts"].append(prompt)
        for sent in _setting_sentences([prompt, *must], people + list(who), props | set(fprops), w):
            if sent not in e["setting"]:
                e["setting"].append(sent)
        if chap not in e["chapters"]:
            e["chapters"].append(chap)
        if when:
            e["when"].append(str(when))

    for chap_dir in sorted(p for p in vid.iterdir() if p.is_dir()):
        if "-sample" in chap_dir.name:
            continue
        plan = chap_dir / "plan.json"
        if plan.exists():
            pj = json.loads(plan.read_text())
            for pic in pj.get("pictures", []) or []:
                add(pic.get("where"), chap_dir.name, pic.get("prompt", ""), who=pic.get("who", []), when=pic.get("when"))
            for ent in pj.get("entries", []) or []:
                add(ent.get("where"), chap_dir.name, who=ent.get("who", []), when=ent.get("when"))
        kf = chap_dir / "anim" / "keyframes.json"
        if kf.exists():
            for fr in json.loads(kf.read_text()).get("frames", []) or []:
                add(fr.get("where"), chap_dir.name, fr.get("prompt", ""), fr.get("must", []), fr.get("who", []),
                    [x for p_ in fr.get("props", []) or [] for x in str(p_).split(",")], when=fr.get("when"))
    return locs


PRONOUNS = re.compile(r"\b(he|she|his|her|him|hers|they|them|their|i|we|you)\b", re.I)


FURNITURE = ("workbench", "bench", "table", "chair", "stool", "shelf", "shelves", "bed", "cupboard", "hearth",
             "fireplace", "barrel", "trough", "cart")
PROP_LIBRARY = TOOLS / "stage_props" / "props.json"


def prop_names(look: dict, vid: Path | None = None) -> list[str]:
    """Every story-prop name a set must NOT contain (the shots place these themselves): the book's physical props
    (look.json "props") + the words of the HANDLED prop classes - with `vid`, only the classes this book stages
    (stage/props/plan.json families). Not the prop library's generic names (rock, stone, door, table, bench, cloth,
    facade: 10-21 they cut a river's stones and the forge's door out of place descriptions). Furniture words are
    excluded: furniture belongs to the set, drawn empty."""
    names = []
    pc = TOOLS / "stage_templates" / "prop_classes.json"
    C = json.loads(pc.read_text()) if pc.exists() else {}
    pp = Path(vid) / "stage" / "props" / "plan.json" if vid else None
    used = {(f.get("base") or b) for b, f in (json.loads(pp.read_text()).get("families") or {}).items()} \
        if pp and pp.exists() else None
    if used is not None:          # + the classes of the book's own props (the plan file holds one chapter's families)
        lp = " ".join(look.get("props", {})).lower()
        used |= {r["base"] for r in C.get("physical", []) if any(re.search(rf"\b{re.escape(w)}", lp) for w in r.get("words", []))}
    # staged = every movable prop + the fixed ones people WORK or operate (anvil, bellows); other fixed furniture
    # (a throne, a well) may be drawn by the set: staging anchors it there and hides its mesh (scene_stage.anchor_fixed)
    phys = [w for r in C.get("physical", []) if not r.get("architectural")
            and (r.get("mobility") != "fixed" or r.get("work_support") or r.get("joints"))
            and (used is None or r.get("base") in used)
            for w in r.get("words", [])]
    other = set(C.get("set", [])) | set(C.get("fx", [])) | set(C.get("character", []))
    for n in look.get("props", {}):
        # only the book's PHYSICAL props (the classes props_build stages); "Stone Well", "Harvest Moon" and
        # "Timber-framed Houses" are the set's own (10-21: they were cut out of the town square)
        ws = re.findall(r"[a-z]+", n.lower())
        if ws and ws[-1] not in other and any(re.search(rf"\b{re.escape(w)}", n.lower()) for w in phys):
            names.append(n)
    names += phys         # every staged prop class (10-21: the anvil is staged, so a set must not paint one)
    return [n for n in names if n and n.strip().lower() not in FURNITURE]


def handled_here(vid: Path, look: dict, prompts: list) -> list:
    """The props people HANDLE that this place's moments name (the book's prop families whose class is handled:
    movable, or fixed and worked/operated), by their family names only -- never generic words like 'stone' or
    'door' that are also the set's own (a river's stones, the forge's door)."""
    pc = TOOLS / "stage_templates" / "prop_classes.json"
    rules = {r["base"]: r for r in json.loads(pc.read_text()).get("physical", [])} if pc.exists() else {}
    pp = vid / "stage" / "props" / "plan.json"
    fams = json.loads(pp.read_text()).get("families") or {} if pp.exists() else {}
    txt = " ".join(prompts).lower()
    out = set()
    for b, f in fams.items():
        r = rules.get(f.get("base") or b, {})
        if r.get("architectural") or (r.get("mobility") == "fixed" and not (r.get("work_support") or r.get("joints"))):
            continue
        base = (f.get("base") or b).split("__")[0].replace("_", " ")
        if re.search(rf"\b{re.escape(base)}s?\b", txt):
            out.add(base)
    return sorted(out)


def _mentions(text: str, name: str) -> bool:
    n = re.escape(name.strip().lower())
    return bool(n) and re.search(rf"\b{n}(s|es)?\b", text.lower()) is not None


def _setting_sentences(texts, names, props=(), where: str = "") -> list[str]:
    """Sentences that describe the PLACE, not the people or the story props in it: no character name, no
    personal pronoun, no shot-type opener, no prop (a prop painted into the set would duplicate or pin the
    one the shot places). Furniture named in a dropped sentence comes back as 'an empty <furniture>'.
    Prop words that are part of the location's own name (the 'river' of a place called 'River below the mill') are kept."""
    out, furn = [], []
    props = [p_.strip() for p_ in props if p_ and p_.strip() and not _mentions(where, p_.strip())]
    for t in texts:
        for sent in re.split(r"(?<=[.!?])\s+", t or ""):
            s2 = sent.strip()
            if not s2:
                continue
            bad = (PRONOUNS.search(s2) or any(n and _mentions(s2, n) for n in names)
                   or re.match(r"(?i)((wide|medium|close-up|low angle|high angle|action|establishing)\b.*\bshot\b of"
                               r"|close[- ]up of)", s2))
            if not bad and any(_mentions(s2, p_) for p_ in props):
                bad = True
            if bad:
                furn += [f for f in FURNITURE if _mentions(s2, f) and not _mentions(where, f)]
                continue
            out.append(s2 if s2.endswith((".", "!", "?")) else s2 + ".")
    for f in dict.fromkeys(furn):
        line = f"There is an empty {f}."
        if line not in out:
            out.append(line)
    return out


def _gpu_stop_passed() -> bool:
    """A stop time (HH:MM) in WORLD_GPU_STOP_FILE (default eval/hyworld/gpu_stop_after): from then until 06:00
    no new GPU stage starts (the shared box belongs to another job for the night)."""
    f = Path(os.environ.get("WORLD_GPU_STOP_FILE", HOME / "tts_reader" / "eval" / "hyworld" / "gpu_stop_after"))
    if not f.exists():
        return False
    try:
        hh, mm = (int(x) for x in f.read_text().strip().split(":"))
    except ValueError:
        return False
    now = time.localtime()
    cur, stop = now.tm_hour * 60 + now.tm_min, hh * 60 + mm
    if stop >= 12 * 60:                                  # evening stop: passed if later today or after midnight
        return cur >= stop or cur < 6 * 60
    return stop <= cur < 6 * 60                          # after-midnight stop


TIME_CUES = {
    "night": ("night", "moon", "moonlight", "moonlit", "midnight", "stars", "starlight", "lantern", "candle",
              "candlelight", "torchlight", "darkness"),
    "dusk": ("dusk", "sunset", "twilight", "evening", "nightfall"),
    "dawn": ("dawn", "sunrise", "daybreak", "first light"),
    "day": ("sunlight", "sunny", "daylight", "noon", "midday", "morning", "afternoon", "sunlit", "sun"),
}


def time_of_day(when: list[str], prompts: list[str]) -> tuple[str, dict]:
    """night|dusk|dawn|day for a location: an explicit `when` in the plan wins (majority); else cue words in
    the scene prompts vote (a lantern or candle at work means it is dark); default night-neutral 'day'."""
    norm = {"evening": "dusk", "sunset": "dusk", "twilight": "dusk", "morning": "day", "noon": "day",
            "afternoon": "day", "sunrise": "dawn", "midnight": "night"}
    explicit = [norm.get(w.strip().lower(), w.strip().lower()) for w in when if w]
    explicit = [w for w in explicit if w in TIME_CUES]
    if explicit:
        best = max(set(explicit), key=explicit.count)
        return best, {"source": "when", "votes": {k: explicit.count(k) for k in set(explicit)}}
    votes = {k: sum(len(re.findall(rf"\b{re.escape(c)}s?\b", p.lower())) for p in prompts for c in cues)
             for k, cues in TIME_CUES.items()}
    if not any(votes.values()):
        return "day", {"source": "default", "votes": votes}
    return max(votes, key=votes.get), {"source": "cues", "votes": votes}


def lighting_sentence(tod: str, kind: str) -> str:
    """Lighting for a set picture + panorama, phrased with NO object names (a named lantern gets drawn)."""
    if kind == "interior":
        return {"night": "Night: black night outside every window and doorway, only faint moonlight there; the"
                         " room lit by a dim, warm, golden glow.",
                "dusk": "Dusk: deep blue evening light outside the windows; the room dim and warm.",
                "dawn": "Dawn: pale early light outside the windows; the room dim.",
                "day": "Daytime: daylight through the windows."}[tod]
    return {"night": "Night under a dark sky with faint moonlight.",
            "dusk": "Dusk, the sky deep blue and orange at the horizon.",
            "dawn": "Dawn, pale light low in the sky.",
            "day": "Daytime, soft overcast daylight."}[tod]


def wait_fair() -> None:
    """GPU-fairness gate before every GPU stage (shared box): while a process matching WORLD_FAIR_WAITER
    (a job waiting for the GPUs, e.g. a two-card gaprun) exists, wait -- no cap: build only in gaps.
    Past the stop time (_gpu_stop_passed) exit instead. Then require MemAvailable > 15 GB."""
    waiter = os.environ.get("WORLD_FAIR_WAITER")
    def running(pat):
        return subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0
    t0 = time.time()
    while True:
        if _gpu_stop_passed():
            log("GPU stop time passed: not starting a GPU stage")
            raise SystemExit(3)
        if not (waiter and running(waiter)):
            break
        time.sleep(5)
    # box rule: never start a model load with MemAvailable <= 15 GB
    def avail_gb():
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) / 2**20
        return 999.0
    while avail_gb() <= float(os.environ.get("WORLD_MIN_AVAIL_GB", "15")):
        log(f"waiting for RAM: {avail_gb():.1f} GB available")
        time.sleep(30)
    log(f"fairness gate passed after {time.time() - t0:.0f}s ({avail_gb():.0f} GB available)")


def place_reference(vid: Path, where: str) -> Path | None:
    """The book's establishing picture of a place (bookcaster refs/<slug>-<hash>.png), newest first."""
    refs = sorted((vid / "refs").glob(f"{slugify(where)}-*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return refs[0] if refs else None


def update_assets_sets(vid: Path, entries: list[dict], keep: set | None = None) -> Path:
    """Merge set entries (keyed by `where`) into <book>/work/video/stage/assets.json under "sets",
    leaving every other key (characters, props: other tools own them) untouched. File-locked."""
    import fcntl
    stage = vid / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / "assets.json"
    with open(stage / ".assets.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        sets = {e["where"]: e for e in data.get("sets", [])}
        if keep is not None:
            sets = {k: v for k, v in sets.items() if k in keep}
        for e in entries:
            sets[e["where"]] = e
        data["sets"] = sorted(sets.values(), key=lambda e: e["where"])
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(path)
    return path


def set_entry(where: str, wd: Path) -> dict:
    w = json.loads((wd / "world.json").read_text())
    g = w.get("ground") or {"plane_z": 0.0, "normal": [0, 0, 1],
                            "eye_height_m": w["capture_camera"]["pos"][2], "method": str(w["level"].get("ground_fit"))}
    return {"where": where, "world_dir": str(wd), "kind": w.get("kind"), "mode": w.get("mode"),
            "eye_height": g["eye_height_m"], "ground": g, "capture_camera": w["capture_camera"],
            "bounds_min": w.get("bounds_min"), "bounds_max": w.get("bounds_max"),
            "files": w.get("files"), "built": w.get("built"), "room": w.get("room"), "when": w.get("when"),
            "view_region": w.get("view_region")}


def cmd_build_book(a) -> None:
    book = Path(a.book_dir).expanduser().resolve()
    vid = book / "work" / "video"
    locs = book_locations(vid)
    over_p = vid / "stage" / "sets.json"          # optional per-book overrides {where: {kind, image, text, skip}}
    over = json.loads(over_p.read_text()) if over_p.exists() else {}
    if not a.style and (book / "art_style.txt").exists():
        a.style = (book / "art_style.txt").read_text().strip()
    entries, report = [], []

    def related(w):
        """Another place this one is named after ("River below the forge" -> "The Forge"): its picture is the
        reference, so the forge on the river bank IS the forge (user 10-21)."""
        base_ = VARIANT_RX.sub("", w).lower()
        for o_ in locs:
            n_ = re.sub(r"(?i)^(the|a|an)\s+", "", VARIANT_RX.sub("", o_)).lower()
            if o_ != w and n_ and n_ != re.sub(r"(?i)^(the|a|an)\s+", "", base_) and re.search(rf"\b{re.escape(n_)}\b", base_):
                return o_
        return None
    # the places others are named after build first
    locs = dict(sorted(locs.items(), key=lambda kv: related(kv[0]) is not None))
    for where, e in locs.items():
        if a.only and where not in a.only:
            continue
        wd = vid / "worlds" / slugify(where)
        o = over.get(where, {})
        kind, why = classify_kind(e["description"], e["prompts"])
        # the side, by authority (10-22: the fight's moments' words - "in front of a cave", "inside" - outvoted "The
        # Burnt Mountain: Outside: a desolate landscape" and it was rebuilt as a mine tunnel with windows): the plan's
        # SCENES for this place, then the description's own "Inside:" / "Outside:", then the pictures' vote
        dm = re.match(r"\s*(inside|outside)\s*:", e["description"] or "", re.I)
        if dm:
            kind = "interior" if dm.group(1).lower() == "inside" else "exterior"
        sides = {sc.get("side") for pj in vid.glob("*/plan.json")
                 for sc in (json.loads(pj.read_text()).get("scenes") or []) if sc.get("where") == where}
        if len(sides) == 1 and next(iter(sides)) in ("inside", "outside"):
            kind = "interior" if next(iter(sides)) == "inside" else "exterior"
        vm = VARIANT_RX.search(where)
        if vm:
            kind = "interior" if vm.group(1).lower() == "inside" else "exterior"
        kind = o.get("kind", kind)
        look_ = json.loads((vid / "look.json").read_text()) if (vid / "look.json").exists() else {}
        desc_ = clean_description(e["description"], kind, prop_names(look_, vid))
        d_score = _cue_score(e["description"] or "")
        # the plan's establishing picture shows ONE side of the place: used only when its description agrees
        ref_ok = not vm and not ((kind == "interior" and d_score < 0) or (kind == "exterior" and d_score > 0))
        # ...and only when nothing had to be cleaned out of its description: that picture shows what was cleaned
        # (10-21: the throne room's picture has the throne the shots place as a prop -> two thrones)
        _n = lambda t: re.sub(r"\s+", " ", re.sub(r"\W+|\band\b", " ", (t or "").lower())).strip()  # noqa: E731
        if _n(desc_) != _n(e["description"]):
            ref_ok = False
        tod, tod_why = time_of_day(([o["when"]] if o.get("when") else []) + e["when"], e["prompts"])
        if tod_why["source"] == "default" and VARIANT_RX.search(where) and VARIANT_RX.sub("", where) in locs:
            b_ = locs[VARIANT_RX.sub("", where)]      # one shot outside a night forge is night too
            tod, tod_why = time_of_day(b_["when"], b_["prompts"])
        row = {"where": where, "world_dir": str(wd), "kind": kind, "why": why, "pictures": len(e["prompts"]),
               "when": tod, "when_why": tod_why}
        if why["pictures_interior"] and why["pictures_exterior"]:
            # shots both inside and outside one `where`: one world serves the majority; the others need a
            # second set (planned: a `where` + interior/exterior variant) or fall back to painted backgrounds
            row["mixed"] = True
        rel = related(where)
        rel_src = None
        if rel:
            for cand in (f"{VARIANT_RX.sub('', rel)} (outside)", rel):
                sp_ = vid / "worlds" / slugify(cand) / "source.png"
                wj_ = vid / "worlds" / slugify(cand) / "world.json"
                if sp_.exists() and wj_.exists() and json.loads(wj_.read_text()).get("kind") == "exterior":
                    rel_src = sp_
                    break
        noun_ = re.sub(r"(?i)^(the|a|an)\s+", "", VARIANT_RX.sub("", where))
        noun_ = noun_[:1].lower() + noun_[1:]
        # deserted (no people) but FURNISHED as such a place is (user 10-21: "empty room" fights the model -- the
        # throne room got its throne anyway); the props people handle go to the NEGATIVE prompt (panorama)
        lead_ = (f"The inside of a deserted {noun_}." if kind == "interior"
                 else f"A deserted {noun_}, seen from outside.")
        if rel_src:
            lead_ += (f" The building of the reference picture ({VARIANT_RX.sub('', rel)}) stands nearby, "
                      f"exactly as in the reference: the same walls, roof and chimney.")
        # what this set is drawn FROM: a changed description / reference rebuilds it (stale-input rule)
        img_ = o.get("image") or (place_reference(vid, where) if ref_ok and not rel_src else None)
        sig_txt = json.dumps([o.get("text") or "", desc_, str(img_ or ""), str(rel_src or ""), kind])
        # versioned (10-21: changing what goes into the signature made every set "changed" -> 6 rebuilds): a
        # signature of another version is not evidence; the set is judged by what world.json says it was built from
        sig = f"v{SIG_VERSION}:" + hashlib.sha1(sig_txt.encode()).hexdigest()[:16]
        sigp = wd / "build.sig"
        old_sig = sigp.read_text().strip() if sigp.exists() else ""
        same_version = old_sig.startswith(f"v{SIG_VERSION}:")
        stale = (wd / "world.json").exists() and same_version and old_sig != sig
        if (wd / "world.json").exists() and not same_version:
            # first sight (no build.sig yet): compare with what world.json says it was built from -- a text build
            # with other words, or a picture build that would now be drawn from text (10-21 the throne room: its
            # picture holds the throne the shots place) is stale; else it is taken as current
            wj0 = json.loads((wd / "world.json").read_text())
            built_text = wj0.get("text")
            want_text = None if img_ else (o.get("text") or f"{lead_} {desc_}.")
            if (built_text is None) != (want_text is None) or (want_text and built_text and desc_ not in built_text):
                stale = True
            elif not a.dry_run:
                sigp.write_text(sig)
        if stale:
            row["rebuilt_because"] = "its description / reference changed"
        if o.get("skip"):
            row["status"] = "skipped (sets.json)"
        elif (wd / "world.json").exists() and not a.rebuild and not stale:
            row["status"] = "reused"
            row["kind"] = json.loads((wd / "world.json").read_text()).get("kind")
            entries.append({**set_entry(where, wd), "mixed": bool(row.get("mixed")), "desc": desc_})
        else:
            img = img_
            setting = " ".join(e["setting"][:8])
            # emptiness phrased POSITIVELY ("a deserted ..."): "empty of people and animals" drew a man and
            # horses into a yard (10-20) -- naming a thing draws it, negated or not
            lead = lead_
            text = o.get("text") or (f"{lead} {desc_}." if desc_
                                     else (f"{lead} {setting}" if setting else lead))
            if tod_why["source"] != "default":          # no evidence -> no lighting claim
                text += " " + lighting_sentence(tod, kind)
            # the book's period (10-21): drawn in, NOT in the build signature (an existing set is judged, not redrawn)
            sys.path.insert(0, str(TOOLS.parent))
            from tts_reader import video as _V
            text += _V.period_line(str(vid))
            row["source"] = str(img) if img else "text"
            row["text"] = text
            if a.dry_run:
                row["status"] = "would build"
            else:
                cmd = [sys.executable, str(Path(__file__).resolve()), "build", where, "--out", str(wd),
                       "--kind", kind, "--mode", a.mode, "--steps", str(a.steps), "--bus", a.bus,
                       "--pano-prompt", text, "--pano-engine", a.pano_engine, "--lightning", str(a.lightning),
                       "--seed", str(a.seed)]
                cmd += ["--image", str(img)] if img else ["--text", text, "--style", a.style]
                if rel_src and not img:
                    cmd += ["--ref", str(rel_src)]
                # this place's HANDLED props (named in its moments) -> the panorama's negative prompt
                hp = handled_here(vid, look_, e["prompts"])
                if hp:
                    cmd += ["--negative", ", ".join(hp)]
                    row["negative"] = hp
                if stale and (wd / "world.json").exists():
                    old_ = wd / f"stale_{time.strftime('%m%d-%H%M')}"
                    old_.mkdir()
                    for f_ in list(wd.iterdir()):
                        if f_.is_file():
                            f_.rename(old_ / f_.name)
                if a.rebuild or stale:
                    cmd.append("--redo-pano")           # a rebuilt source must not reuse the old panorama
                wait_fair()
                t = time.time()
                subprocess.run(cmd, check=True)
                row["status"] = f"built in {time.time() - t:.0f}s"
                sigp.write_text(sig)
                entries.append({**set_entry(where, wd), "mixed": bool(row.get("mixed")), "desc": desc_})
        wj = wd / "world.json"
        if wj.exists() and not a.dry_run:
            w_ = json.loads(wj.read_text())
            tod_rec = tod if tod_why["source"] != "default" else "unknown"
            if w_.get("when") != tod_rec:
                w_["when"], w_["when_why"] = tod_rec, tod_why
                wj.write_text(json.dumps(w_, indent=1))
            if entries and entries[-1].get("where") == where:
                entries[-1]["when"] = tod_rec
        report.append(row)
        log(f"{where}: {row['status']} ({row['kind']}, {why}, when={tod})")
    if entries and not a.dry_run:
        # a full pass: sets no keyframe names any more leave the registry (10-21: "The Forge: a stone building ..."
        # stayed listed after its moments were re-homed)
        print(json.dumps({"assets": str(update_assets_sets(vid, entries, keep=None if a.only else set(locs)))}))
    print(json.dumps(report, indent=1))


def stage_regeom_points(world: str) -> None:
    """CPU-only geometry rebuild of a pano world from its points.npy (no MoGe, no GPU): re-applies the
    current sky mask (skyseg for exteriors), background layer, backdrop, meshes and room bounds. points.npy
    is already levelled with the camera at eye height, so it is shifted back to camera-origin first; the
    nadir levelling then finds the same eye height (scale 1)."""
    import numpy as np
    from PIL import Image
    wd = Path(world)
    w = json.loads((wd / "world.json").read_text())
    P = np.load(wd / "points.npy").astype(np.float64)
    H, W = P.shape[:2]
    cam = np.array(w["capture_camera"]["pos"], float)
    img = np.asarray(Image.open(wd / w.get("pano", "pano.png")).convert("RGB").resize((W, H)))
    Pr = P - cam
    d = np.linalg.norm(Pr, axis=-1)
    sr = (w.get("level") or {}).get("sky_radius_m")
    sky = (d > 0.95 * sr) if sr else np.zeros((H, W), bool)
    if w.get("kind") != "interior":
        from world_sky import sky_mask_equirect
        ss = sky_mask_equirect(img)
        w["skyseg_added_frac"] = round(float((ss & ~sky).mean()), 4)
        sky |= ss
    # sky pixels get directions only: put them at a unit radius (the dome radius is recomputed)
    dirs = _pano_dirs(H, W)
    Pr = np.where(sky[..., None], dirs, Pr)
    valid = np.isfinite(Pr).all(-1)
    keep = ("place", "slug", "book", "kind", "mode", "source", "text", "style", "seed", "pano", "timings",
            "despeckled_px", "skyseg_added_frac", "when", "geometry")
    meta = {k: w[k] for k in keep if k in w}
    meta.update({"metric": True, "gravity_aligned": True})
    meta.setdefault("timings", {})["regeom_from_points"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _finish_world(wd, Pr, img, valid, sky, np.full((H, W), 2 * np.pi / W), True, meta)


def cmd_regeom(a) -> None:
    """Rebuild into <world>/.regeom, then swap each file in with os.replace (atomic per file), so a reader
    importing the set never sees a half-written mesh."""
    import shutil
    wd = world_dir(a.book, a.world)
    tmp = wd / ".regeom"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    w = json.loads((wd / "world.json").read_text())
    for f in ("world.json", "points.npy", w.get("pano", "pano.png")):
        shutil.copy2(wd / f, tmp / f)
    cpu_run(["_regeom_points", str(tmp)])
    for f in sorted(tmp.iterdir(), key=lambda p: p.name == "world.json"):    # world.json last
        os.replace(f, wd / f.name)
    tmp.rmdir()
    print((wd / "world.json").read_text()[:300])


def cmd_room(a) -> None:
    import numpy as np
    sys.path.insert(0, str(TOOLS))
    from world_room import room_bounds
    wd = world_dir(a.book, a.world)
    w = json.loads((wd / "world.json").read_text())
    P = np.load(wd / "points.npy")
    cam = w["capture_camera"]["pos"]
    d = np.linalg.norm(P - np.array(cam), axis=-1)
    sr = (w.get("level") or {}).get("sky_radius_m")
    geo = np.isfinite(d) & ((d < 0.95 * sr) if sr else True)
    w["room"] = room_bounds(P[geo], cam[:2])
    (wd / "world.json").write_text(json.dumps(w, indent=1))
    print(json.dumps({k: w["room"][k] for k in ("bbox_xy", "ceiling_z")}), len(w["room"]["walls"]), "walls")


def parse_size(s: str):
    w, h = s.lower().split("x")
    return [int(w), int(h)]


def cmd_render(a) -> None:
    wd = world_dir(a.book, a.world)
    if not (wd / "world.json").exists():
        sys.exit(f"no world at {wd}")
    if a.jobs:
        jobs = json.loads(Path(a.jobs).read_text())
        for j in jobs:
            j.setdefault("size", parse_size(a.size))
    else:
        if not a.camera or not a.out:
            sys.exit("render needs --camera and --out (or --jobs)")
        jobs = [{"camera": json.loads(a.camera), "out": str(Path(a.out).resolve()),
                 "depth": str(Path(a.depth).resolve()) if a.depth else None,
                 "mask": str(Path(a.mask).resolve()) if a.mask else None, "size": parse_size(a.size)}]
    gpu_run(["_render", str(wd), json.dumps(jobs)], a.bus, timeout=900)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1].startswith("_"):
        st, args = sys.argv[1], sys.argv[2:]
        if st == "_pano":
            stage_pano(args[0], args[1], args[2], int(args[3]), int(args[4]), args[5] if len(args) > 5 else "")
        elif st == "_geom_pano":
            stage_geom_pano(*args)
        elif st == "_geom_pano_infer":
            stage_geom_pano_infer(*args)
        elif st == "_regeom_points":
            stage_regeom_points(*args)
        elif st == "_geom_single":
            stage_geom_single(*args)
        elif st == "_render":
            stage_render(*args)
        else:
            sys.exit(f"unknown stage {st}")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("place")
    b.add_argument("--book", default=os.environ.get("WORLD_BOOK"), help="book name under WORLD_BOOKS_ROOT; or give --out")
    b.add_argument("--out", help="world directory (default <book>/work/video/worlds/<slug>)")
    b.add_argument("--image")
    b.add_argument("--text")
    b.add_argument("--ref", help="a picture the drawn source must match (a related place's building)")
    b.add_argument("--negative", default="", help="things the panorama must not draw (the place's handled props)")
    b.add_argument("--style", default="")
    b.add_argument("--kind", choices=["interior", "exterior"], default="exterior")
    b.add_argument("--mode", choices=["pano", "single"], default="pano")
    b.add_argument("--pano-prompt", default=None, help="what the unseen parts of the 360 view hold")
    b.add_argument("--redo-pano", action="store_true")
    b.add_argument("--pano-engine", choices=["int8", "gguf"], default=os.environ.get("WORLD_PANO_ENGINE", "int8"))
    b.add_argument("--lightning", type=float, default=float(os.environ.get("WORLD_PANO_LIGHTNING_W", PANO_LIGHTNING_W)),
                   help="Lightning LoRA weight (0 = off: 30-step true CFG). On: CFG 1, use --steps 8")
    b.add_argument("--steps", type=int, default=None, help="default 8 with Lightning, 30 without")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--bus", default=os.environ.get("WORLD_BUS", "65"))
    r = sub.add_parser("render")
    r.add_argument("world")
    r.add_argument("--book", default=os.environ.get("WORLD_BOOK"), help="needed only when <world> is a slug, not a world dir")
    r.add_argument("--camera")
    r.add_argument("--out")
    r.add_argument("--depth")
    r.add_argument("--mask")
    r.add_argument("--jobs", help="JSON list of {camera, out, depth?, mask?, size?} rendered in one GPU hold")
    r.add_argument("--size", default="1536x864")
    r.add_argument("--bus", default=os.environ.get("WORLD_BUS", "65"))
    bb = sub.add_parser("build-book", help="one set per distinct plan `where` (cached), indexed in stage/assets.json")
    bb.add_argument("book_dir")
    bb.add_argument("--only", nargs="*", help="build only these locations")
    bb.add_argument("--dry-run", action="store_true")
    bb.add_argument("--rebuild", action="store_true")
    bb.add_argument("--mode", choices=["pano", "single"], default="pano")
    bb.add_argument("--steps", type=int, default=None, help="default 8 with Lightning, 30 without")
    bb.add_argument("--pano-engine", choices=["int8", "gguf"], default=os.environ.get("WORLD_PANO_ENGINE", "int8"))
    bb.add_argument("--lightning", type=float, default=float(os.environ.get("WORLD_PANO_LIGHTNING_W", PANO_LIGHTNING_W)))
    bb.add_argument("--style", default="", help="appended when a location has no reference picture")
    bb.add_argument("--bus", default=os.environ.get("WORLD_BUS", "65"))
    bb.add_argument("--seed", type=int, default=42, help="a redraw after a handled prop showed up in the set: another seed")
    rg = sub.add_parser("regeom", help="CPU-only geometry rebuild from points.npy (sky mask, layers, room)")
    rg.add_argument("world")
    rg.add_argument("--book", default=os.environ.get("WORLD_BOOK"), help="needed only when <world> is a slug, not a world dir")
    rm = sub.add_parser("room", help="add room bounds (floor polygon, walls, ceiling) to an interior's world.json")
    rm.add_argument("world")
    rm.add_argument("--book", default=os.environ.get("WORLD_BOOK"), help="needed only when <world> is a slug, not a world dir")
    i = sub.add_parser("info")
    i.add_argument("world")
    i.add_argument("--book", default=os.environ.get("WORLD_BOOK"), help="needed only when <world> is a slug, not a world dir")
    a = ap.parse_args()
    if getattr(a, "steps", 0) is None:              # Lightning draws in 8 steps; the CFG path keeps its 30
        a.steps = 8 if getattr(a, "lightning", 0) > 0 else 30
    if a.cmd == "build":
        cmd_build(a)
    elif a.cmd == "render":
        cmd_render(a)
    elif a.cmd == "build-book":
        cmd_build_book(a)
    elif a.cmd == "regeom":
        cmd_regeom(a)
    elif a.cmd == "room":
        cmd_room(a)
    else:
        print((world_dir(a.book, a.world) / "world.json").read_text())


if __name__ == "__main__":
    main()
