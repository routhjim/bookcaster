"""Pure rules for a whole figure's head weights (bl_combo.py, mode "statue"); no Blender, numpy only.

A whole figure's head is rigid on the head bone. It used to be the part above the outfit's collar line (a plane a
few cm over the neck's upper joint, tilted to run higher at the back, as a collar does). A picture-to-3D head does
not sit where the hidden MetaHuman head does: fitted by its collar, the drawn head sits lower, its ear lobes, jaw
corners and the hair over the nape under that line. Those took the body's weights, and the body has no head
surface: the nearest body face to an ear is the top of the shoulder, so the ears carried clavicle and upper-arm
weights. When the arms came down from the spread bind pose to a standing pose (or the head turned) the lower ear
was pulled away from its top and stood out as a point, and a turned head tore the face along the line (measured
11-19 on three figures: up to 4 cm off the head's own motion from the arms alone).

The rule, any figure: the head is what lies above a level line just over the neck's upper joint, within a head's
reach of the neck's axis, plus whatever the collar line already gave the head; below that line, over the neck, the
weights blend from the head bone to the body's own, and the neck never follows an arm (clavicle and upper-arm
weights are dropped there: a neck turns with the spine and the head, not with a raised hand).
"""
import numpy as np

# the head's line over the neck's upper joint (metres) and the height of the blend below it
HEAD_LINE_M = 0.01
BLEND_M = 0.05
# how far from the neck's axis the neck's blend reaches (the neck and what hangs round it, not the shoulders' cloth),
# and the head above the line (a head and the hair standing off it: a mop stood 13.6 cm behind the head joint)
REACH_M = 0.14
REACH_ABOVE_M = 0.22
# bones a neck never follows
ARM_BONES = ("clavicle", "upperarm", "lowerarm", "hand", "thumb", "index", "middle", "ring", "pinky")
# what the neck follows when nothing but arm weights were copied to it
NECK_BONE = "neck_01"
MAX_INFLUENCES = 4


def axis_distance(P, a, b):
    """Horizontal distance of each point from the neck's axis (the line a -> b, e.g. neck_01 -> head) at its height."""
    P = np.asarray(P, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = (P[:, 2] - a[2]) / (d[2] if abs(d[2]) > 1e-9 else 1e-9)
    c = a[None, :2] + t[:, None] * d[None, :2]
    return np.linalg.norm(P[:, :2] - c, axis=1)


def head_follow(P, neck_lo, head_joint, neck_top_z, line_m=HEAD_LINE_M, blend_m=BLEND_M, reach_m=REACH_M,
                reach_above_m=REACH_ABOVE_M, rigid=None):
    """Per point, how much it follows the head bone: 1 = rigid on the head, 0 = untouched, between = the neck's
    blend. P (N, 3) metres, z up; neck_lo / head_joint: two points on the neck's axis (neck_01 and the head joint);
    neck_top_z: the neck's upper joint's height (neck_02). rigid (N,) bool: points already given to the head (the
    collar line), kept at 1."""
    P = np.asarray(P, float)
    z1 = float(neck_top_z) + float(line_m)
    z0 = z1 - max(float(blend_m), 1e-6)
    r = axis_distance(P, neck_lo, head_joint)
    s = np.clip((P[:, 2] - z0) / (z1 - z0), 0.0, 1.0)
    s = s * s * (3.0 - 2.0 * s)                       # smoothstep: no kink at either end of the blend
    w = np.where(r < float(reach_m), s, 0.0)
    w[(P[:, 2] >= z1) & (r < float(reach_above_m))] = 1.0
    if rigid is not None:
        w = np.maximum(w, np.asarray(rigid, bool).astype(float))
    return w


def head_box(statue_pts, hidden_pts, head_joint, floor_z, q=0.005):
    """The whole figure's visible head against the hidden MetaHuman head it stands in for, both above floor_z (just
    over the neck's upper joint), as boxes relative to the head joint (metres; x the figure's side, y toward its back,
    z up; quantiles q..1-q, so a stray crumb does not count). A thing worn on the head is sized and placed from the
    hidden head's bones and face mesh; a drawn head is often wider, deeper (its hair) and lower, and stood through a
    helm (11-19). The scene can fit a worn thing to the visible head with this. None when either side is empty."""
    hj = np.asarray(head_joint, float)
    out = {}
    for k, Q in (("statue", statue_pts), ("hidden", hidden_pts)):
        Q = np.asarray(Q, float).reshape(-1, 3)
        Q = Q[Q[:, 2] > float(floor_z)] - hj
        if len(Q) < 20:
            return None
        out[k] = {"lo": [round(float(x), 4) for x in np.quantile(Q, q, axis=0)],
                  "hi": [round(float(x), 4) for x in np.quantile(Q, 1 - q, axis=0)]}
    s, h = out["statue"], out["hidden"]
    ext = lambda b, i: max(b["hi"][i] - b["lo"][i], 1e-6)          # noqa: E731
    out["width_ratio"] = round(ext(s, 0) / ext(h, 0), 3)
    out["depth_ratio"] = round(ext(s, 1) / ext(h, 1), 3)
    out["top_diff_m"] = round(s["hi"][2] - h["hi"][2], 4)
    out["centre_offset_m"] = [round((s["lo"][i] + s["hi"][i] - h["lo"][i] - h["hi"][i]) / 2, 4) for i in (0, 1)]
    return out


def worn_fit(box, ease_m=0.01):
    """How a thing worn on the head, sized and placed for the hidden head (the scene's rule: a share of the hidden
    head's height, centred behind its eyes), changes to cover the visible head of a whole figure, from head_box:
    "scale" multiplies its size (never below 1: enough to hold the visible head's width and depth with ease_m all
    round, in the proportion it held the hidden head's); "shift_m" moves its centre [side, back] (the figure's own
    axes: x its side, y toward its back), "up_m" raises it when the visible head is taller. No box: unchanged."""
    if not box:
        return {"scale": 1.0, "shift_m": [0.0, 0.0], "up_m": 0.0}
    s, h = box["statue"], box["hidden"]
    need = [(s["hi"][i] - s["lo"][i] + 2 * ease_m) / max(h["hi"][i] - h["lo"][i], 1e-6) for i in (0, 1)]
    return {"scale": round(max(1.0, *need), 3), "shift_m": list(box.get("centre_offset_m") or [0.0, 0.0]),
            "up_m": round(max(0.0, float(box.get("top_diff_m") or 0.0)), 4)}


def is_arm(name):
    return name.startswith(ARM_BONES) and "toe" not in name      # (the feet's index/middle/ring toes are not arms)


def blend_weights(groups, w, head="head", neck=NECK_BONE, max_influences=MAX_INFLUENCES):
    """One point's new weights: groups {bone: weight} (the copied body weights), w its head share (head_follow).
    The arm bones go (a neck never follows an arm), the rest is renormalised and keeps its strongest few, scaled by
    1 - w, and the head takes w. Returns {bone: weight} summing to 1."""
    if w >= 1.0 - 1e-6:
        return {head: 1.0}
    rest = {k: float(v) for k, v in groups.items() if v > 1e-6 and not is_arm(k) and k != head}
    hw = float(w) + float(groups.get(head, 0.0)) * (1.0 - float(w))
    rest = dict(sorted(rest.items(), key=lambda kv: -kv[1])[:max(1, max_influences - 1)])
    tot = sum(rest.values())
    if tot <= 1e-6:
        rest, tot = {neck: 1.0}, 1.0
    out = {k: (1.0 - hw) * v / tot for k, v in rest.items()}
    if hw > 1e-6:
        out[head] = hw
    s = sum(out.values())
    return {k: v / s for k, v in out.items() if v > 1e-6}
