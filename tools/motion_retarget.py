"""Retarget a CanonicalMotion onto any rig made by tools/blockout_model*.py (pure numpy, any python).

    from motion_retarget import retarget
    rm = retarget(cm, "models/<who>/rig.json", at=(x, y), facing_deg=0.0)
    seq = rm.seq()            # [ {joint: np.array(3)} per frame ] -- the blockout "targets" / W format
    rm.bone_rot[f, b]         # (3, 3) world rotation of bone BONES[b] relative to its rest (twist included)
    rm.save(base)             # base.npz + base.json

Rig: rig.json "joints" (rest, world, Z up, facing +X, left = +Y, feet on z = 0) and the bone table
blockout_contact.BONES (= blockout_model_bl.BONES): pelvis, spine, neck, head, clavicles, upper arms,
forearms, hands, hip bones, thighs, shins, feet. Joint names are the blockout ones (pelvis, spine_mid, neck,
head, head_top, {r,l}_{shoulder,elbow,wrist,hand,hip,knee,ankle,toe}, nose/eyes/ears riding the head);
any rig with those names works, whatever its proportions or rest pose (A-pose, T-pose).

Method (rotation retarget, then FK on the rig's OWN bone lengths, so the result is exactly reachable by the
rig and blockout's import_model/fk_model reproduce it):
  * each rig bone has a DRIVER joint in the canonical skeleton (MAP below); its world rotation is
        D = G_driver(t) @ A,     A = minimal rotation taking the rig's rest bone direction onto the canonical
                                  rest direction of the same segment (aligns A-pose to T-pose per bone)
    so the bone inherits the source's swing AND twist (wrist roll, forearm pronation, pelvis/chest turn);
  * where the segment has a canonical tail joint, a small swing correction makes the bone point exactly
    along the source segment (keeps limb directions faithful even where the joint sets differ);
  * the root follows the source pelvis, scaled by pelvis height (rig / source): stride and crouch depth
    scale with the leg, so feet do not slide more than in the source;
  * the clip is grounded once (lowest toe/heel over the clip = the rig's rest sole height); per-frame foot
    locking is the cleanup pass's job (blockout_contact.lock_feet), using cm.foot_contact.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
from blockout_contact import BONES  # noqa: E402  (read-only use of the shared bone table)
from motion_canonical import JI, CanonicalMotion, rot_z  # noqa: E402

# rig bone -> (canonical head joint, canonical tail joint or None, canonical driver joint)
MAP = {"pelvis": ("pelvis", "spine2", "pelvis"), "spine": ("spine2", "neck", "spine3"),
       "neck": ("neck", "head", "neck"), "head": ("neck", None, "head")}
for _s in ("r", "l"):
    MAP.update({f"{_s}_clav": ("neck", f"{_s}_shoulder", f"{_s}_collar"),
                f"{_s}_uarm": (f"{_s}_shoulder", f"{_s}_elbow", f"{_s}_shoulder"),
                f"{_s}_farm": (f"{_s}_elbow", f"{_s}_wrist", f"{_s}_elbow"),
                f"{_s}_hand": (f"{_s}_elbow", None, f"{_s}_wrist"),      # rest dir = forearm extension
                f"{_s}_hipb": ("pelvis", f"{_s}_hip", "pelvis"),
                f"{_s}_thigh": (f"{_s}_hip", f"{_s}_knee", f"{_s}_hip"),
                f"{_s}_shin": (f"{_s}_knee", f"{_s}_ankle", f"{_s}_knee"),
                f"{_s}_foot": (f"{_s}_ankle", f"{_s}_foot", f"{_s}_ankle")})
# the canonical "head" segment for the rest alignment: neck -> head (no head-top joint in SMPL-22)
SRC_REST_DIR = {"head": ("neck", "head")}
BONE_NAMES = [b[0] for b in BONES]
HEAD_J = {b[0]: b[1] for b in BONES}
TAIL_J = {b[0]: b[2] for b in BONES}
PARENT = {b[0]: b[3] for b in BONES}
# rig joints that are not a bone head/tail ride a bone rigidly
RIDE = {"nose": "head", "r_eye": "head", "l_eye": "head", "r_ear": "head", "l_ear": "head", "head_top": "head",
        "r_heel": "r_foot", "l_heel": "l_foot"}


def _n(v):
    v = np.asarray(v, float)
    d = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(d, 1e-9)


def rot_between(a, b):
    """Batched minimal rotation taking direction(s) a onto b: (..., 3) -> (..., 3, 3)."""
    a, b = np.broadcast_arrays(_n(a), _n(b))
    v = np.cross(a, b)
    c = np.sum(a * b, -1)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = v[..., 2], -v[..., 1], v[..., 0]
    f = 1.0 / np.maximum(1.0 + c, 1e-9)
    R = np.eye(3) + K + K @ K * f[..., None, None]
    anti = c < -0.9999                                   # opposite: turn 180 deg about any perpendicular
    if np.any(anti):
        aa = a[anti]
        ax = _n(np.cross(aa, np.where(np.abs(aa[:, :1]) < 0.9, [1.0, 0, 0], [0, 1.0, 0])))
        R[anti] = 2 * ax[:, :, None] * ax[:, None, :] - np.eye(3)
    return R


def load_rig(rig) -> dict:
    J = json.load(open(rig)) if isinstance(rig, str) else rig
    joints = {k: np.asarray(v, float) for k, v in J["joints"].items()}
    H = float(J.get("height", max(v[2] for v in joints.values())))
    for s in ("r", "l"):                                  # a heel point for foot locking, if the rig has none
        if f"{s}_heel" not in joints and f"{s}_ankle" in joints and f"{s}_toe" in joints:
            a, t = joints[f"{s}_ankle"], joints[f"{s}_toe"]
            fwd = _n(np.array([t[0] - a[0], t[1] - a[1], 0.0]))
            joints[f"{s}_heel"] = np.array([a[0], a[1], t[2]]) - fwd * 0.03 * H
    return {"joints": joints, "height": H}


class RigMotion:
    def __init__(self, names, pos, bone_rot, fps, meta):
        self.names, self.pos, self.bone_rot, self.fps, self.meta = list(names), pos, bone_rot, fps, meta

    @property
    def frames(self):
        return len(self.pos)

    def frame(self, i) -> dict:
        return {n: self.pos[i, k].copy() for k, n in enumerate(self.names)}

    def seq(self) -> list:
        return [self.frame(i) for i in range(self.frames)]

    def save(self, base):
        base = base[:-4] if base.endswith(".npz") else base
        np.savez_compressed(base + ".npz", pos=self.pos.astype(np.float32), bone_rot=self.bone_rot.astype(np.float32),
                            fps=np.float32(self.fps))
        with open(base + ".json", "w") as f:
            json.dump({"format": "rig_motion/1", "fps": self.fps, "frames": self.frames, "joints": self.names,
                       "bones": BONE_NAMES,
                       "pos": "(F,K,3) world joint positions in the rig's joint names (= blockout targets)",
                       "bone_rot": "(F,B,3,3) world rotation of each bone relative to its rest; Blender: "
                                   "pose_bone.matrix = (R @ bone.matrix_local.to_3x3()) with translation = "
                                   "pos[head joint], parents first",
                       **self.meta}, f, indent=1)
        return base + ".npz"

    @staticmethod
    def load(base):
        base = base[:-4] if base.endswith((".npz", ".json")) else base
        z = np.load(base + ".npz")
        h = json.load(open(base + ".json"))
        meta = {k: v for k, v in h.items() if k not in ("format", "fps", "frames", "joints", "bones", "pos", "bone_rot")}
        return RigMotion(h["joints"], z["pos"], z["bone_rot"], float(z["fps"]), meta)


def retarget(cm: CanonicalMotion, rig, at=(0.0, 0.0), facing_deg: float = 0.0, ground: float = 0.0) -> RigMotion:
    R = load_rig(rig)
    RJ, H = R["joints"], R["height"]
    F = cm.frames
    G = cm.global_rots()                     # (F, 22, 3, 3)
    P = cm.pos                               # (F, 22, 3)
    rest_s = cm.rest
    s = float(RJ["pelvis"][2] / rest_s[JI["pelvis"], 2])
    D, head_p = {}, {}
    for name in BONE_NAMES:                  # parent-first
        h, t = HEAD_J[name], TAIL_J[name]
        d_t = RJ[t] - RJ[h]
        sh, st, drv = MAP[name]
        if name in SRC_REST_DIR:
            a0, a1 = SRC_REST_DIR[name]
            d_s = rest_s[JI[a1]] - rest_s[JI[a0]]
        elif st is None:                     # hand: the forearm's rest direction
            d_s = rest_s[JI[drv]] - rest_s[JI[sh]]
        else:
            d_s = rest_s[JI[st]] - rest_s[JI[sh]]
        A = rot_between(d_t, d_s)
        Db = G[:, JI[drv]] @ A               # (F, 3, 3)
        if st is not None:
            want = P[:, JI[st]] - P[:, JI[sh]]
            Db = rot_between(np.einsum("fab,b->fa", Db, d_t), want) @ Db
        par = PARENT[name]
        if par is None:
            hp = P[:, JI["pelvis"]] * s
        else:
            hp = head_p[par] + np.einsum("fab,b->fa", D[par], RJ[h] - RJ[HEAD_J[par]])
        D[name], head_p[name] = Db, hp
    out = {}
    for name in BONE_NAMES:
        out.setdefault(HEAD_J[name], head_p[name])
    for name in BONE_NAMES:                  # tails that are not another bone's head (hands, toes, head_top)
        t = TAIL_J[name]
        if t not in out:
            out[t] = head_p[name] + np.einsum("fab,b->fa", D[name], RJ[t] - RJ[HEAD_J[name]])
    for k, b in RIDE.items():
        if k in RJ and k not in out:
            out[k] = head_p[b] + np.einsum("fab,b->fa", D[b], RJ[k] - RJ[HEAD_J[b]])
    for k in RJ:                             # anything else: rides the pelvis
        if k not in out:
            out[k] = head_p["pelvis"] + np.einsum("fab,b->fa", D["pelvis"], RJ[k] - RJ["pelvis"])
    names = list(RJ.keys())
    pos = np.stack([out[k] for k in names], 1)                  # (F, K, 3)
    # ground the clip once: its lowest foot point sits at the rig's rest sole height
    feet = [names.index(k) for k in ("r_toe", "l_toe", "r_heel", "l_heel") if k in names]
    rest_min = min(RJ[names[i]][2] for i in feet)
    pos[..., 2] += ground + rest_min - pos[:, feet, 2].min()
    # place: turn about the frame-0 pelvis's vertical and move it to `at`
    Rz = rot_z(facing_deg)
    p0 = pos[0, names.index("pelvis")].copy()
    p0[2] = 0.0
    pos = (pos - p0) @ Rz.T + np.array([at[0], at[1], 0.0])
    bone_rot = np.stack([Rz @ D[b] for b in BONE_NAMES], 1)     # (F, B, 3, 3)
    src = {k: cm.meta.get(k) for k in ("backend", "model", "text", "seed")}
    meta = {"rig": rig if isinstance(rig, str) else None, "rig_height": H, "scale": round(s, 4),
            "at": list(at), "facing_deg": facing_deg, "source": src, "events": _events(cm)}
    return RigMotion(names, pos, bone_rot, cm.fps, meta)


def _events(cm):
    from motion_canonical import events
    e = events(cm)
    e["foot_contact"] = {"l": cm.foot_contact[:, 0].astype(int).tolist(), "r": cm.foot_contact[:, 1].astype(int).tolist()}
    return e


def bone_lengths_ok(rm: RigMotion, rig) -> float:
    """Max bone-length error over the clip (m): 0 up to float noise if FK is right."""
    RJ = load_rig(rig)["joints"]
    err = 0.0
    for name in BONE_NAMES:
        h, t = HEAD_J[name], TAIL_J[name]
        if h in rm.names and t in rm.names:
            L = np.linalg.norm(rm.pos[:, rm.names.index(t)] - rm.pos[:, rm.names.index(h)], axis=-1)
            err = max(err, float(np.abs(L - np.linalg.norm(RJ[t] - RJ[h])).max()))
    return err


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", help="CanonicalMotion base (.npz)")
    ap.add_argument("rig", help="rig.json")
    ap.add_argument("out", help="output base (.npz + .json)")
    ap.add_argument("--at", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--facing", type=float, default=0.0)
    a = ap.parse_args()
    cm = CanonicalMotion.load(a.motion)
    rm = retarget(cm, a.rig, tuple(a.at), a.facing)
    rm.save(a.out)
    print(f"{rm.frames} frames, scale {rm.meta['scale']}, bone length err {bone_lengths_ok(rm, a.rig) * 1000:.2f} mm")
