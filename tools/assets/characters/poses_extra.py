"""Caller side (stdlib): extra body poses and a head look control, written as the scene tool's pose form (a named
base pose from scenes/ue/pose.py plus raw "steps", and face curves), so any scene spec can use them today.

Steps are in the character's actor frame (x = his left, y = forward, z = up, centimetres), applied after the base
pose. Poses that reach for something are solved from measured skeletons: the scene tool's result.json
(subjects.<key>.measured.joints, set frame, metres) of a first pass, turned into each character's actor frame here.

    look(pitch_deg, yaw_deg, roll_deg=0)      -> {"steps": [...], "face_curves": {...}}
        pitch < 0 looks down, > 0 up; yaw > 0 toward the character's own left; spread over neck_01 (35 %),
        neck_02 (25 %) and head (40 %); the eyes follow by the same direction (look-down/up/left/right curves at
        |angle| / EYE_FULL_DEG, capped at EYE_MAX), so the gaze leads the head a little as a person's does.
    with_look(body_pose, face_pose, pitch, yaw)  -> the two dicts with the look added (any stance)
    hands_in_lap(seat_z_m)                     -> steps for a seated person's hands resting on the thighs
    veil_lift(head, t)                         -> steps: both hands at the veil's hem under the chin (t = 0) or
                                                  over the crown, the veil turned back (t = 1)
    look_at_sleeve(side, ...)                  -> steps + face curves: one forearm raised a little, turned up,
                                                  head and eyes down on it
    reach_forearm(him, her, side_him, side_her) -> steps for him: bend at the hips and spine toward her, and the
                                                  hand's wrist placed so the palm rests on top of her forearm
CLI: python poses_extra.py look PITCH YAW   (prints the spec fragment)
"""
from __future__ import annotations

import json
import math
import sys

EYE_FULL_DEG = 30.0           # eye curve 1.0 ~ this many degrees of gaze
EYE_MAX = 0.8


def _eye_curves(pitch, yaw):
    c = {}
    if pitch < 0:
        v = min(EYE_MAX, -pitch / EYE_FULL_DEG)
        c["CTRL_expressions_eyeLookDownL"] = c["CTRL_expressions_eyeLookDownR"] = round(v, 3)
    elif pitch > 0:
        v = min(EYE_MAX, pitch / EYE_FULL_DEG)
        c["CTRL_expressions_eyeLookUpL"] = c["CTRL_expressions_eyeLookUpR"] = round(v, 3)
    if yaw:
        v = round(min(EYE_MAX, abs(yaw) / EYE_FULL_DEG), 3)
        # the character's own left is the viewer's right when facing him
        k = "Left" if yaw > 0 else "Right"
        c[f"CTRL_expressions_eyeLook{k}L"] = c[f"CTRL_expressions_eyeLook{k}R"] = v
    return c


def look(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0, eyes=True, head_share=1.0):
    """Head and neck turned to look down/up/aside; eyes follow. head_share < 1 leaves part of the angle to the
    eyes alone (the head turns that share, the eyes the whole)."""
    steps = []
    p, y = pitch_deg * head_share, yaw_deg * head_share
    # pose.py: rot about +x by +deg turns the face up; about z by +deg toward the character's right
    for bone, f in (("neck_01", 0.35), ("neck_02", 0.25), ("head", 0.40)):
        if p:
            steps.append(["rot", bone, [1, 0, 0], round(p * f, 2)])
        if y:
            steps.append(["rot", bone, [0, 0, 1], round(-y * f, 2)])
    if roll_deg:
        steps.append(["rot", "head", [0, 1, 0], round(roll_deg, 2)])
    return {"steps": steps, "face_curves": _eye_curves(pitch_deg, yaw_deg) if eyes else {}}


def with_look(body_pose, face_pose, pitch, yaw=0.0, roll=0.0):
    lk = look(pitch, yaw, roll)
    bp = json.loads(json.dumps(body_pose))
    bp.setdefault("params", {}).setdefault("steps", []).extend(lk["steps"])
    fp = json.loads(json.dumps(face_pose or {}))
    fp.setdefault("curves", {}).update(lk["face_curves"])
    return bp, fp


def hands_in_lap(seat_z_m, clear_m=0.10):
    hz = (seat_z_m + clear_m) * 100
    return [
        ["ik", "upperarm_l", "lowerarm_l", "hand_l", [6.0, 28.0, hz + 8.0], [48.0, -5.0, hz + 12.0]],
        ["ik", "upperarm_r", "lowerarm_r", "hand_r", [-8.0, 25.0, hz + 10.0], [-48.0, -5.0, hz + 12.0]],
        ["aim", "hand_l", "middle_01_l", [-0.75, 0.55, -0.35]],
        ["aim", "hand_r", "middle_01_r", [0.75, 0.6, -0.3]],
        ["curl", "l", 25, 10], ["curl", "r", 25, 10]]


def veil_lift(head, t=0.0, fwd_cm=None):
    """head: the head joint in the actor frame (cm). Both hands hold the veil's hem: at t = 0 just under the
    chin, in front of the face; at t = 1 above the crown, a little behind the brow (the veil turned back)."""
    hx, hy, hz = head
    pts0 = (8.0, 14.0, -9.0)       # lateral, forward, up from the head joint: hem under the chin
    pts1 = (9.0, 11.0, 10.0)       # at the brow, the hem turned back over the bonnet
    lat = pts0[0] + (pts1[0] - pts0[0]) * t
    fwd = (fwd_cm if fwd_cm is not None else pts0[1] + (pts1[1] - pts0[1]) * t)
    up = pts0[2] + (pts1[2] - pts0[2]) * t
    steps = []
    for side, sx in (("l", 1.0), ("r", -1.0)):
        tgt = [hx + sx * lat, hy + fwd, hz + up]
        pole = [hx + sx * 35.0, hy + 20.0, hz - 45.0]         # elbows forward, out and down
        steps.append(["ik", f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", [round(v, 2) for v in tgt],
                      [round(v, 2) for v in pole]])
        # fingers toward each other and up, pinching the hem
        steps.append(["aim", f"hand_{side}", f"middle_01_{side}", [-sx * 0.55, 0.35 - 0.3 * t, 0.75]])
        steps.append(["curl", side, 45, 20])
    return steps


def look_at_sleeve(side="l", seated_z=None, pitch=-32.0, yaw=None):
    """The forearm lifted and turned palm-up in front of the body, the head and eyes down on it."""
    sx = 1.0 if side == "l" else -1.0
    base_z = (seated_z + 0.10) * 100 + 22.0 if seated_z is not None else 105.0
    tgt = [sx * 6.0, 32.0, base_z]
    steps = [["ik", f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", tgt, [sx * 35.0, -15.0, base_z - 25.0]],
             ["aim", f"hand_{side}", f"middle_01_{side}", [-sx * 0.6, 0.75, 0.1]],
             ["curl", side, 30, 12]]
    lk = look(pitch, sx * 14.0 if yaw is None else yaw)
    return {"steps": steps + lk["steps"], "face_curves": lk["face_curves"]}


# ------------------------------------------------------------------ two skeletons: a hand on a forearm
def to_actor(p, position, facing):
    """Set-frame point (m) -> the character's actor frame (cm): x his left, y forward, z up."""
    n = math.hypot(*facing)
    f = (facing[0] / n, facing[1] / n)
    left = (-f[1], f[0])
    d = (p[0] - position[0], p[1] - position[1])
    return [100 * (d[0] * left[0] + d[1] * left[1]), 100 * (d[0] * f[0] + d[1] * f[1]), 100 * p[2]]


def forearm_contact(her_joints, side, sleeve_r_m, along=0.55):
    """The top of her forearm's sleeve at `along` from elbow to wrist (set frame, m) and the forearm's direction."""
    e, w = her_joints[f"lowerarm_{side}"], her_joints[f"hand_{side}"]
    c = [e[i] + along * (w[i] - e[i]) for i in range(3)]
    d = [w[i] - e[i] for i in range(3)]
    n = math.sqrt(sum(x * x for x in d))
    d = [x / n for x in d]
    # "up" across the forearm: world up minus its part along the forearm
    up = [0.0, 0.0, 1.0]
    k = sum(up[i] * d[i] for i in range(3))
    u = [up[i] - k * d[i] for i in range(3)]
    un = math.sqrt(sum(x * x for x in u))
    u = [x / un for x in u]
    top = [c[i] + u[i] * sleeve_r_m for i in range(3)]
    return top, d, u


def reach_forearm(him_joints, him_pos, him_facing, her_joints, her_side, sleeve_r_m, palm_half_m, palm_len_m,
                  side="r", bend_deg=22.0):
    """Steps for him (after his base pose and a forward bend of bend_deg spread over the spine): his `side` hand's
    wrist placed so that the palm's centre (palm_len_m along the hand from the wrist) rests on the top of her
    forearm's sleeve, palm_half_m above it (the palm's half thickness), the hand lying across her forearm.
    The bend is applied first in the steps; the wrist target accounts for it only through the IK (the target is
    absolute), so it stays exact as long as the IK reaches it."""
    top, d, u = forearm_contact(her_joints, her_side, sleeve_r_m)
    palm_c = [top[i] + u[i] * palm_half_m for i in range(3)]
    sh = him_joints[f"upperarm_{side}"]
    # the hand lies across her forearm, fingers pointing away from his shoulder's side
    hd = [palm_c[0] - sh[0], palm_c[1] - sh[1], 0.0]
    k = sum(hd[i] * d[i] for i in range(3))
    across = [hd[i] - k * d[i] for i in range(3)]
    an = math.sqrt(sum(x * x for x in across)) or 1.0
    across = [x / an for x in across]
    hand_dir = [across[i] * 0.85 - u[i] * 0.35 for i in range(3)]          # fingers across and a little down
    wrist = [palm_c[i] - hand_dir[i] * palm_len_m for i in range(3)]
    A = lambda p: to_actor(p, him_pos, him_facing)   # noqa: E731
    w_a = A(wrist)
    hd_a = [x - y for x, y in zip(A([wrist[i] + hand_dir[i] for i in range(3)]), w_a)]
    elbow_pole = A([sh[0], sh[1], sh[2] - 0.5])
    elbow_pole[0] += -25.0 if side == "r" else 25.0
    steps = []
    for b, f in (("spine_01", 0.3), ("spine_02", 0.25), ("spine_03", 0.25), ("spine_04", 0.2)):
        steps.append(["rot", b, [1, 0, 0], round(-bend_deg * f, 2)])
    steps.append(["ik", f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", [round(v, 2) for v in w_a],
                  [round(v, 2) for v in elbow_pole]])
    steps.append(["aim", f"hand_{side}", f"middle_01_{side}", [round(v, 4) for v in hd_a]])
    steps.append(["curl", side, 18, 8])
    return steps, {"palm_centre_set_m": [round(v, 4) for v in palm_c], "sleeve_top_set_m": [round(v, 4) for v in top],
                   "wrist_set_m": [round(v, 4) for v in wrist]}


if __name__ == "__main__":
    if sys.argv[1:2] == ["look"]:
        print(json.dumps(look(float(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 0.0), indent=1))
