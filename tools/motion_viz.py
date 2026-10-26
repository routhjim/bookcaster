"""Stick-figure frame strips / videos for CanonicalMotion or RigMotion (matplotlib; any venv with it).

    python tools/motion_viz.py strip MOTION_BASE out.png [--n 8] [--title ...]
    python tools/motion_viz.py video MOTION_BASE out.mp4
Rows: a 3/4 view and a side view (camera on the character's right), the ground grid, the right/left wrist
paths, and the motion's impact candidates (motion_canonical.events) marked in the titles.
Left limbs blue, right limbs red.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CANON_EDGES = [("pelvis", "l_hip"), ("pelvis", "r_hip"), ("pelvis", "spine1"), ("spine1", "spine2"),
               ("spine2", "spine3"), ("spine3", "neck"), ("neck", "head"), ("spine3", "l_collar"),
               ("spine3", "r_collar"), ("l_collar", "l_shoulder"), ("r_collar", "r_shoulder"),
               ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"), ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"),
               ("l_hip", "l_knee"), ("r_hip", "r_knee"), ("l_knee", "l_ankle"), ("r_knee", "r_ankle"),
               ("l_ankle", "l_foot"), ("r_ankle", "r_foot")]
RIG_EDGES = [("pelvis", "spine_mid"), ("spine_mid", "neck"), ("neck", "head"), ("head", "head_top"),
             ("neck", "l_shoulder"), ("neck", "r_shoulder"), ("l_shoulder", "l_elbow"), ("r_shoulder", "r_elbow"),
             ("l_elbow", "l_wrist"), ("r_elbow", "r_wrist"), ("l_wrist", "l_hand"), ("r_wrist", "r_hand"),
             ("pelvis", "l_hip"), ("pelvis", "r_hip"), ("l_hip", "l_knee"), ("r_hip", "r_knee"),
             ("l_knee", "l_ankle"), ("r_knee", "r_ankle"), ("l_ankle", "l_toe"), ("r_ankle", "r_toe"),
             ("l_heel", "l_toe"), ("r_heel", "r_toe")]


def load_any(base):
    base = base[:-4] if base.endswith((".npz", ".json")) else base
    h = json.load(open(base + ".json"))
    z = np.load(base + ".npz")
    if h.get("template"):                       # creature: edges from the template's parents
        names = h["joints"]
        par = h["parents"]
        edges = [(names[p], names[i]) for i, p in enumerate(par) if p >= 0]
        ev = (h.get("source") or {}).get("events", {})
        title = h["template"] + ": " + ", ".join(f"{e['verb']}@{e.get('t', 0):.2f}s" for e in (h.get("source") or {}).get("verbs", []))
        wr = tuple(n for n in ("fleg_l_toe", "fleg_r_toe") if n in names) or (names[0], names[0])
        return {"pos": z["pos"], "names": names, "edges": edges, "fps": float(z["fps"]), "events": {},
                "title": title, "wrists": wr}
    if h.get("format", "").startswith("canonical_motion"):
        names, edges, wr = h["joints"], CANON_EDGES, ("l_wrist", "r_wrist")
        ev = h.get("events", {})
        title = h.get("source", {}).get("text", "")
    else:
        names, edges, wr = h["joints"], RIG_EDGES, ("l_hand", "r_hand")
        ev = h.get("events", {})
        title = (h.get("source") or {}).get("text", "")
    return {"pos": z["pos"], "names": names, "edges": [e for e in edges if e[0] in names and e[1] in names],
            "fps": float(z["fps"]), "events": ev, "title": title, "wrists": wr}


def project(P, view):
    """(…,3) -> (…,2). '34': camera front-right, 35 deg off the facing, 10 deg down; 'side': from the
    character's right (-Y), +X to the right."""
    if view == "side":
        return np.stack([P[..., 0], P[..., 2]], -1)
    a = np.radians(-55.0)            # camera direction in the ground plane (right-front of a +X-facing body)
    e = np.radians(10.0)
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    fwd = np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), -np.sin(e)])
    up = np.cross(fwd, right)
    return np.stack([P @ right, P @ up], -1)


def draw(ax, M, f, view, trail=True):
    P = M["pos"]
    N = M["names"]
    idx = {n: i for i, n in enumerate(N)}
    Q = project(P, view)
    # ground grid
    lo, hi = P[..., :2].min((0, 1)) - 0.5, P[..., :2].max((0, 1)) + 0.5
    for gx in np.arange(np.floor(lo[0]), hi[0] + 0.01, 0.5):
        g = project(np.array([[gx, lo[1], 0], [gx, hi[1], 0]]), view)
        ax.plot(g[:, 0], g[:, 1], color="#ddd", lw=0.6, zorder=0)
    for gy in np.arange(np.floor(lo[1]), hi[1] + 0.01, 0.5):
        g = project(np.array([[lo[0], gy, 0], [hi[0], gy, 0]]), view)
        ax.plot(g[:, 0], g[:, 1], color="#ddd", lw=0.6, zorder=0)
    if trail:
        for w, c in zip(M["wrists"], ("#7aa7ff", "#ff8a7a")):
            ax.plot(Q[:, idx[w], 0], Q[:, idx[w], 1], color=c, lw=0.7, alpha=0.7, zorder=1)
    for a, b in M["edges"]:
        side = lambda n, s_: n.startswith(s_ + "_") or n.endswith("_" + s_) or f"_{s_}_" in n
        c = "#1f5fd6" if side(b, "l") else "#d62f1f" if side(b, "r") else "#222"
        ax.plot([Q[f, idx[a], 0], Q[f, idx[b], 0]], [Q[f, idx[a], 1], Q[f, idx[b], 1]], color=c, lw=2.2, zorder=3,
                solid_capstyle="round")
    h = idx.get("head")
    ax.scatter([Q[f, h, 0]], [Q[f, h, 1]], s=60, color="#222", zorder=4)
    for w, c in zip(M["wrists"], ("#1f5fd6", "#d62f1f")):
        ax.scatter([Q[f, idx[w], 0]], [Q[f, idx[w], 1]], s=14, color=c, zorder=4)


def limits(M, view):
    Q = project(M["pos"], view).reshape(-1, 2)
    lo, hi = Q.min(0) - 0.15, Q.max(0) + 0.15
    c, r = (lo + hi) / 2, max(hi - lo) / 2
    return (c[0] - r, c[0] + r), (c[1] - r, c[1] + r)


def strip(base, out, n=8, title=None, f0=0, f1=None):
    M = load_any(base)
    F = len(M["pos"])
    ev = M["events"]
    marks = {}
    for k in ("r_hand", "l_hand"):
        if k in ev:
            marks[ev[k]["peak_frame"]] = marks.get(ev[k]["peak_frame"], "") + f" {k[0]}peak"
            marks[ev[k]["impact_frame"]] = marks.get(ev[k]["impact_frame"], "") + f" {k[0]}stop"
    frames = sorted(set(np.linspace(f0, F - 1 if f1 is None else min(f1, F - 1), n).round().astype(int).tolist()))
    fig, axes = plt.subplots(2, len(frames), figsize=(1.9 * len(frames), 4.6), dpi=110)
    for r, view in enumerate(("34", "side")):
        xl, yl = limits(M, view)
        for c, f in enumerate(frames):
            ax = axes[r, c]
            draw(ax, M, f, view, trail=(c == 0))
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_aspect("equal")
            ax.axis("off")
            if r == 0:
                ax.set_title(f"f{f} {f / M['fps']:.2f}s", fontsize=7)
    fig.suptitle((title or M["title"])[:120] + f"   [{F} f @ {M['fps']:.0f} fps; "
                 + ", ".join(f"{k}: peak f{v['peak_frame']} stop f{v['impact_frame']}" for k, v in ev.items()
                             if isinstance(v, dict) and "peak_frame" in v) + "]", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out)
    plt.close(fig)
    return out


def video(base, out):
    import imageio.v2 as imageio
    M = load_any(base)
    lim = {v: limits(M, v) for v in ("34", "side")}
    w = imageio.get_writer(out, fps=M["fps"], codec="libx264", quality=7, macro_block_size=8)
    for f in range(len(M["pos"])):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=80)
        for ax, v in zip(axes, ("34", "side")):
            draw(ax, M, f, v)
            ax.set_xlim(*lim[v][0])
            ax.set_ylim(*lim[v][1])
            ax.set_aspect("equal")
            ax.axis("off")
        fig.suptitle(f"{M['title'][:80]}  f{f}", fontsize=8)
        fig.canvas.draw()
        w.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        plt.close(fig)
    w.close()
    return out


if __name__ == "__main__":
    op, base, out = sys.argv[1:4]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
    rng = [int(x) for x in sys.argv[sys.argv.index("--range") + 1].split(":")] if "--range" in sys.argv else [0, None]
    print({"strip": lambda: strip(base, out, n, f0=rng[0], f1=rng[1]), "video": lambda: video(base, out)}[op]())


def sheet(img_dir, out, views=("34", "side"), scale=1.0, label=""):
    """Grid of Blender renders (motion_render_bl.py out_dir): one row per view, one column per frame."""
    from PIL import Image, ImageDraw
    files = sorted(os.listdir(img_dir))
    frames = sorted({f.split("_")[-1][:-4] for f in files if f.endswith(".png")})
    ims = {(v, fr): Image.open(os.path.join(img_dir, f"{v}_{fr}.png")).convert("RGB") for v in views for fr in frames
           if os.path.exists(os.path.join(img_dir, f"{v}_{fr}.png"))}
    w, h = next(iter(ims.values())).size
    w, h = int(w * scale), int(h * scale)
    S = Image.new("RGB", (w * len(frames), h * len(views) + 20), "white")
    d = ImageDraw.Draw(S)
    d.text((4, 4), label[:200], fill="black")
    for r, v in enumerate(views):
        for c, fr in enumerate(frames):
            if (v, fr) in ims:
                S.paste(ims[(v, fr)].resize((w, h)), (c * w, 20 + r * h))
                d.text((c * w + 4, 24 + r * h), f"{v} f{int(fr)}", fill="black")
    S.save(out, quality=88)
    return out
