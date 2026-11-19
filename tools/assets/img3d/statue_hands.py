"""Pure rules for a whole figure's jointed hands (bl_combo.py, mode "statue"); no Blender, numpy only.

A whole figure is one picture-to-3D statue: its hands are one fused lump with no finger bones, so a hand holding a
tool stood open with the handle through it. bl_combo cuts the statue's own hand off past a plane across the forearm
near the wrist and carries the MetaHuman body's own hands (with their finger weights) in the figure instead. Here:

  - where to cut: the statue's radius profile along the forearm axis (stations measured from the wrist joint); the
    cut is the narrowest section in a window round the wrist. A plain sleeve or a bare forearm narrows to the wrist
    and the palm widens past it; a wide cuff or bracer that overlaps the wrist is wider than the hand that comes out
    of it, so its narrowest section lies just past the cuff's end and the cuff stays on the statue;
  - covering: the hand's own colour against the statue's face skin. Bare hands are skin-coloured and near the face's
    colour; gloves, gauntlets and mittens are not (the jointed hand then wears the covering's colour, so a gloved
    figure keeps gloved hands);
  - the decision: the body's hand is used only when it can be fitted (a forearm profile at the wrist, a statue hand
    past the cut, a body hand on that side, sizes and directions within reason); else that side keeps the old rigid
    statue hand, with the reason.
"""
import math

import numpy as np

# the window round the wrist joint (metres along the forearm axis, + toward the fingers) searched for the cut
CUT_LO_M = -0.03
CUT_HI_M = 0.06
# sections within this of the narrowest count as the narrowest; of those, the one nearest the wrist joint
CUT_TIE_M = 0.002
# a cuff: a section this much wider than the narrow sections either side of it, the later one within CUFF_TOL_M of
# the narrowest
CUFF_TOL_M = 0.006
CUFF_STEP_M = 0.008
# covering: a hand this far (RGB distance, 0..1 per channel) from the face's skin, or with under this share of its
# points skin-coloured, is covered
COVER_DIST = 0.22
COVER_SKIN_SHARE = 0.45
# the fit: statue hand length / body hand length outside this range, or hands pointing this far apart, is no fit
LEN_RATIO = (0.5, 2.2)
ANGLE_MAX_DEG = 60.0
MIN_FIG_HAND = 50
MIN_BODY_HAND = 200


def axial_radial(P, origin, axis):
    """Each point's station along the axis (from origin) and its distance from the axis line."""
    P = np.asarray(P, float)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    d = P - np.asarray(origin, float)
    p = d @ ax
    rad = np.linalg.norm(d - p[:, None] * ax[None, :], axis=1)
    return p, rad


def radius_profile(P, p, axis, stations, half=0.005, q=0.8, min_n=6):
    """The section radius at each station: the points within half of it, their distance from the section's own
    centre across the axis (the arm's axis line, shoulder to wrist joint, runs a few cm off a bulky sleeve's
    middle), its q-quantile (a high quantile, not the max: a stray point beside the arm is not the arm). NaN where
    fewer than min_n points."""
    P = np.asarray(P, float).reshape(-1, 3)
    p = np.asarray(p, float).reshape(-1)
    ax = np.asarray(axis, float)
    ax = ax / np.linalg.norm(ax)
    out = np.full(len(stations), np.nan)
    for i, st in enumerate(stations):
        m = np.abs(p - st) < half
        if m.sum() >= min_n:
            d = P[m] - P[m].mean(0)
            d = d - (d @ ax)[:, None] * ax[None, :]
            out[i] = float(np.quantile(np.linalg.norm(d, axis=1), q))
    return out


def choose_cut(stations, radii, lo=CUT_LO_M, hi=CUT_HI_M, tie=CUT_TIE_M, cuff_tol=CUFF_TOL_M, cuff_step=CUFF_STEP_M,
               min_valid=3):
    """(station, why): the narrowest section of the forearm in [lo, hi] (within `tie` of the narrowest, the one
    nearest the wrist joint), moved on past any cuff: a later section nearly as narrow (within cuff_tol of the
    narrowest) with a section wider than both by cuff_step between them is where the hand comes out of a cuff or
    bracer that overlaps the wrist (the forearm just above the cuff is as narrow as the hand's base: cut there, the
    cuff went with the hand). (None, why) without a profile."""
    st = np.asarray(stations, float)
    r = np.asarray(radii, float)
    m = (st >= lo - 1e-9) & (st <= hi + 1e-9) & np.isfinite(r)
    if m.sum() < min_valid:
        return None, f"no forearm profile at the wrist ({int(m.sum())} sections in the window)"
    rmin = float(r[m].min())
    near = st[m & (r <= rmin + tie)]
    c = float(near[np.argmin(np.abs(near))])
    rc = float(r[np.isclose(st, c)][0])
    past = []
    for c2 in sorted(st[m & (r <= rmin + cuff_tol) & (st > c)]):
        r2 = float(r[np.isclose(st, c2)][0])
        between = m & (st > c) & (st < c2)
        if between.any() and float(r[between].max()) >= max(rc, r2) + cuff_step:
            past.append(round(float(r[between].max()) * 100, 1))
            c, rc = float(c2), r2
    why = f"narrowest section {rmin * 100:.1f} cm radius; cut at {c * 100:+.1f} cm from the wrist joint ({rc * 100:.1f} cm)"
    if past:
        why += f", past a cuff {max(past):.1f} cm wide"
    return c, why


def median_rgb(c):
    c = np.asarray(c, float).reshape(-1, 3)
    return np.median(c, 0) if len(c) else None


def hand_colour(hand_rgb, hand_skin, face_rgb, dist_max=COVER_DIST, share_min=COVER_SKIN_SHARE):
    """The jointed hand's colour from the statue's own hand. hand_rgb: the removed hand's colours (N, 3); hand_skin:
    which of them pass the skin-colour test; face_rgb: the statue's face skin colours (or None). Bare hands take the
    median of their skin-coloured points (a ring, a cuff's edge or a shadow left out); a covering (too few skin
    points, or a median far from the face's skin) takes the median of all of it."""
    hand_rgb = np.asarray(hand_rgb, float).reshape(-1, 3)
    hand_skin = np.asarray(hand_skin, bool).reshape(-1)
    allm = median_rgb(hand_rgb)
    face = median_rgb(face_rgb) if face_rgb is not None and len(np.asarray(face_rgb).reshape(-1, 3)) else None
    share = float(hand_skin.mean()) if len(hand_skin) else 0.0
    skinm = median_rgb(hand_rgb[hand_skin]) if hand_skin.sum() >= 10 else None
    ref = skinm if skinm is not None else allm
    dist = float(np.linalg.norm(ref - face)) if face is not None and ref is not None else None
    covered = share < share_min or (dist is not None and dist > dist_max)
    rgb = allm if covered or skinm is None else skinm
    why = (f"{share:.0%} of the hand skin-coloured" + (f", {dist:.2f} from the face's skin" if dist is not None else "")
           + (" -> covered (glove, gauntlet or mitten): the covering's colour" if covered else " -> bare skin"))
    return {"rgb": None if rgb is None else [round(float(v), 3) for v in rgb],
            "hand_median": None if allm is None else [round(float(v), 3) for v in allm],
            "hand_skin_median": None if skinm is None else [round(float(v), 3) for v in skinm],
            "face_median": None if face is None else [round(float(v), 3) for v in face],
            "skin_share": round(share, 3), "dist_to_face": None if dist is None else round(dist, 3),
            "covered": bool(covered), "why": why}


def angle_deg(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 180.0
    return math.degrees(math.acos(float(np.clip(a @ b / (na * nb), -1.0, 1.0))))


def fit_decision(wanted, cut, n_fig_hand, n_body_hand, len_ratio, angle, cut_why=""):
    """(True, why) when the body's hand can stand in for the statue's on this side; else (False, the reason) and
    the side keeps the rigid statue hand."""
    if not wanted:
        return False, "jointed hands off in the job (statue_body_hands false)"
    if cut is None:
        return False, cut_why or "no cut found"
    if n_fig_hand < MIN_FIG_HAND:
        return False, f"no statue hand past the cut ({n_fig_hand} points)"
    if n_body_hand < MIN_BODY_HAND:
        return False, f"the body has no hand on this side ({n_body_hand} points)"
    if len_ratio is None or not (LEN_RATIO[0] <= len_ratio <= LEN_RATIO[1]):
        return False, f"statue hand / body hand length {len_ratio} outside {LEN_RATIO}"
    if angle > ANGLE_MAX_DEG:
        return False, f"the statue's hand points {angle:.0f} deg from the body's (over {ANGLE_MAX_DEG:.0f})"
    return True, "fitted"
