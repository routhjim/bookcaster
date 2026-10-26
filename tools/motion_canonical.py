"""CanonicalMotion: the one body-motion format every motion source returns (pure numpy, any python).

Layer 3 of PLAN-staging-engine-2026-10-19.md. A motion source (tools/motion_source.py) turns
(character, action text, duration, fps) into a CanonicalMotion; tools/motion_retarget.py maps it onto any
rig made by tools/blockout_model*.py; the contact cleanup pass (blockout side) then adjusts it minimally.

Skeleton: SMPL-22 body joints (the joint set of SMPL/SMPL-H without fingers), names below, lower-case.
    0 pelvis  1 l_hip  2 r_hip  3 spine1  4 l_knee  5 r_knee  6 spine2  7 l_ankle  8 r_ankle  9 spine3
    10 l_foot (ball of the foot)  11 r_foot  12 neck  13 l_collar  14 r_collar  15 head  16 l_shoulder
    17 r_shoulder  18 l_elbow  19 r_elbow  20 l_wrist  21 r_wrist
Frame (the SAME as the blockout world and the rigs): metres, Z up, +X = the character's forward at
frame 0, +Y = the character's left. Frame 0 has the pelvis at x = y = 0; z = 0 is the ground (the sole of
the source body at its lowest point over the clip).

Arrays (the .npz; the .json header next to it describes them):
    rest   (22, 3)    rest joint positions (T-pose: arms along +-Y, palms down), standing on z = 0
    parents(22,)      parent index (-1 for the pelvis)
    rot    (F, 22, 4) LOCAL joint rotations, quaternion (w, x, y, z), relative to the rest pose: joint j's
                      global rotation G_j = G_parent(j) @ R_j, the pelvis's R is its global rotation
    trans  (F, 3)     world position of the pelvis joint
    pos    (F, 22, 3) world joint positions (forward kinematics of rest/rot/trans; for convenience)
    foot_contact (F, 2)  l, r: the foot is planted (low and nearly still)  -> foot-lock phases
    hand_speed   (F, 2)  l, r wrist speed, m/s                              -> impact timing
FK: p_root = trans; p_j = p_parent + G_parent @ (rest_j - rest_parent).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

FORMAT = "canonical_motion/1"
JOINTS = ["pelvis", "l_hip", "r_hip", "spine1", "l_knee", "r_knee", "spine2", "l_ankle", "r_ankle", "spine3",
          "l_foot", "r_foot", "neck", "l_collar", "r_collar", "head", "l_shoulder", "r_shoulder", "l_elbow",
          "r_elbow", "l_wrist", "r_wrist"]
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
JI = {n: i for i, n in enumerate(JOINTS)}
# SMPL / HY-Motion world (Y up, +Z forward, +X = the body's left) -> canonical (Z up, +X fwd, +Y left)
C_SMPL = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


# --------------------------------------------------------------------------------------------------
# rotations (numpy, batched over leading dims)
# --------------------------------------------------------------------------------------------------

def quat_to_mat(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    M = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                  2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                  2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)
    return M.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(M):
    M = np.asarray(M, float)
    sh = M.shape[:-2]
    M = M.reshape(-1, 3, 3)
    q = np.empty((len(M), 4))
    tr = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    for i, m in enumerate(M):
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    q[q[:, 0] < 0] *= -1
    return q.reshape(sh + (4,))


def slerp(q0, q1, t):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    t = np.asarray(t, float)[..., None]
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    lin = d > 0.9995
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    s = np.where(lin, 1.0, s)
    a = np.where(lin, 1 - t, np.sin((1 - t) * th) / s)
    b = np.where(lin, t, np.sin(t * th) / s)
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rot_z(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


# --------------------------------------------------------------------------------------------------
# the format
# --------------------------------------------------------------------------------------------------

@dataclass
class CanonicalMotion:
    rot: np.ndarray                  # (F, 22, 4) local quaternions wxyz
    trans: np.ndarray                # (F, 3) pelvis world position
    rest: np.ndarray                 # (22, 3)
    fps: float
    meta: dict = field(default_factory=dict)   # source: backend, model, text, seed, ...
    pos: np.ndarray = None           # (F, 22, 3), filled by fk()
    foot_contact: np.ndarray = None  # (F, 2)
    hand_speed: np.ndarray = None    # (F, 2)

    def __post_init__(self):
        self.rot = np.asarray(self.rot, float)
        self.trans = np.asarray(self.trans, float)
        self.rest = np.asarray(self.rest, float)
        if self.pos is None:
            self.refresh()

    @property
    def frames(self) -> int:
        return len(self.rot)

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def global_rots(self) -> np.ndarray:
        """(F, 22, 3, 3) global joint rotations."""
        R = quat_to_mat(self.rot)
        G = np.empty_like(R)
        for j, p in enumerate(PARENTS):
            G[:, j] = R[:, j] if p < 0 else G[:, p] @ R[:, j]
        return G

    def fk(self) -> np.ndarray:
        G = self.global_rots()
        P = np.empty((self.frames, len(JOINTS), 3))
        for j, p in enumerate(PARENTS):
            if p < 0:
                P[:, j] = self.trans
            else:
                P[:, j] = P[:, p] + np.einsum("fab,b->fa", G[:, p], self.rest[j] - self.rest[p])
        return P

    def refresh(self):
        """Recompute pos, foot_contact and hand_speed after rot/trans changed."""
        self.pos = self.fk()
        self.foot_contact, self.hand_speed = analyse(self.pos, self.fps)

    # ---- edits -----------------------------------------------------------------------------------
    def resample(self, fps: float) -> "CanonicalMotion":
        """Same motion at another frame rate (slerp rotations, lerp the root)."""
        if abs(fps - self.fps) < 1e-6:
            return self
        n = max(2, int(round(self.duration * fps)))
        s = np.arange(n) * self.fps / fps
        s = np.clip(s, 0, self.frames - 1)
        i0 = np.floor(s).astype(int)
        i1 = np.minimum(i0 + 1, self.frames - 1)
        t = s - i0
        rot = slerp(self.rot[i0], self.rot[i1], np.repeat(t[:, None], len(JOINTS), 1))
        trans = self.trans[i0] * (1 - t[:, None]) + self.trans[i1] * t[:, None]
        meta = dict(self.meta, resampled_from_fps=self.fps)
        return CanonicalMotion(rot, trans, self.rest, fps, meta)

    def normalise(self) -> "CanonicalMotion":
        """Turn about Z so the body faces +X at frame 0 and put the frame-0 pelvis at x = y = 0."""
        P0 = self.pos[0]
        left = P0[JI["l_hip"]] - P0[JI["r_hip"]] + P0[JI["l_shoulder"]] - P0[JI["r_shoulder"]]
        left[2] = 0
        fwd = np.cross(left, [0, 0, 1.0])
        yaw = np.degrees(np.arctan2(fwd[1], fwd[0]))
        Rz = rot_z(-yaw)
        rot = self.rot.copy()
        rot[:, 0] = mat_to_quat(Rz[None] @ quat_to_mat(self.rot[:, 0]))
        trans = self.trans @ Rz.T
        trans[:, :2] -= trans[0, :2]
        return CanonicalMotion(rot, trans, self.rest, self.fps, dict(self.meta, normalised_yaw_deg=float(yaw)))

    def hold(self, frame: int, until_s: float = None, blend: int = 4) -> "CanonicalMotion":
        """Keep the pose of `frame` from there on (for held poses the model will not hold, e.g. a block):
        frames after it ease into the frozen pose over `blend` frames, then stay; the clip is cut or padded
        to until_s (default: its own length)."""
        n = self.frames if until_s is None else max(frame + 1, int(round(until_s * self.fps)))
        rot = np.concatenate([self.rot[:frame + 1], np.repeat(self.rot[frame:frame + 1], n - frame - 1, 0)])
        trans = np.concatenate([self.trans[:frame + 1], np.repeat(self.trans[frame:frame + 1], n - frame - 1, 0)])
        for k in range(1, blend + 1):          # ease: overshoot motion continues briefly, then settles
            i = frame + k
            if i < min(n, self.frames):
                w = (blend + 1 - k) / (blend + 1) * 0.5
                rot[i] = slerp(rot[frame], self.rot[i], np.full(len(JOINTS), w))
                trans[i] = trans[frame] * (1 - w) + self.trans[i] * w
        return CanonicalMotion(rot, trans, self.rest, self.fps, dict(self.meta, held_from_frame=int(frame)))

    def peak_frame(self, joint: str = "r_wrist", axis: int = 2) -> int:
        """Frame where a joint is highest (axis 2) -- e.g. the top of a raise, for hold()."""
        return int(np.argmax(self.pos[:, JI[joint], axis]))

    def trim(self, start_s: float = 0.0, dur_s: float = None) -> "CanonicalMotion":
        a = int(round(start_s * self.fps))
        b = self.frames if dur_s is None else min(self.frames, a + int(round(dur_s * self.fps)))
        return CanonicalMotion(self.rot[a:b], self.trans[a:b], self.rest, self.fps, dict(self.meta))

    # ---- io ----------------------------------------------------------------------------------------
    def header(self) -> dict:
        return {"format": FORMAT, "fps": self.fps, "frames": self.frames, "duration_s": round(self.duration, 4),
                "joints": JOINTS, "parents": PARENTS.tolist(),
                "frame": "metres, Z up, +X forward at frame 0, +Y = character's left; z=0 ground",
                "rotation": "local, quaternion wxyz, relative to the T-pose rest; G_j = G_parent @ R_j",
                "arrays": {"rest": "(22,3)", "rot": "(F,22,4)", "trans": "(F,3) pelvis", "pos": "(F,22,3)",
                           "foot_contact": "(F,2) l,r bool", "hand_speed": "(F,2) l,r m/s"},
                "events": events(self), "source": self.meta}

    def save(self, path: str) -> str:
        """path without extension (or .npz): writes path.npz + path.json."""
        base = path[:-4] if path.endswith(".npz") else path
        os.makedirs(os.path.dirname(os.path.abspath(base)), exist_ok=True)
        np.savez_compressed(base + ".npz", rot=self.rot.astype(np.float32), trans=self.trans.astype(np.float32),
                            rest=self.rest.astype(np.float32), pos=self.pos.astype(np.float32),
                            foot_contact=self.foot_contact, hand_speed=self.hand_speed.astype(np.float32),
                            parents=PARENTS, fps=np.float32(self.fps))
        with open(base + ".json", "w") as f:
            json.dump(self.header(), f, indent=1)
        return base + ".npz"

    @staticmethod
    def load(path: str) -> "CanonicalMotion":
        base = path[:-4] if path.endswith((".npz", ".json")) else path
        z = np.load(base + ".npz")
        meta = {}
        if os.path.exists(base + ".json"):
            meta = json.load(open(base + ".json")).get("source", {})
        return CanonicalMotion(z["rot"], z["trans"], z["rest"], float(z["fps"]), meta)


def analyse(P: np.ndarray, fps: float, z_eps: float = 0.06, v_eps: float = 0.25):
    """Foot plants (the lower of ankle/ball is within z_eps of the clip's lowest and moves < v_eps m/s)
    and wrist speeds."""
    F = len(P)
    vel = np.zeros_like(P)
    if F > 1:
        vel[1:] = (P[1:] - P[:-1]) * fps
        vel[0] = vel[1]
    contact = np.zeros((F, 2), bool)
    for k, (a, b) in enumerate((("l_ankle", "l_foot"), ("r_ankle", "r_foot"))):
        z = np.minimum(P[:, JI[a], 2], P[:, JI[b], 2])
        zmin = z.min()
        sp = np.linalg.norm(vel[:, JI[b]], axis=-1)
        contact[:, k] = (z < zmin + z_eps) & (sp < v_eps)
    hs = np.stack([np.linalg.norm(vel[:, JI["l_wrist"]], axis=-1), np.linalg.norm(vel[:, JI["r_wrist"]], axis=-1)], 1)
    return contact, hs


def events(cm: CanonicalMotion) -> dict:
    """Timing cues for the contact cleanup: per hand, the frame of peak speed and the impact candidate
    (the first frame after the peak where speed falls below 30% of the peak = the hand has stopped)."""
    out = {}
    for k, s in enumerate(("l", "r")):
        hs = cm.hand_speed[:, k]
        pk = int(np.argmax(hs))
        stop = next((i for i in range(pk, len(hs)) if hs[i] < 0.3 * hs[pk]), len(hs) - 1)
        out[f"{s}_hand"] = {"peak_frame": pk, "peak_speed": round(float(hs[pk]), 3), "impact_frame": int(stop)}
    return out


def from_smpl(rot_mats: np.ndarray, joints_world: np.ndarray, rest_smpl: np.ndarray, fps: float,
              ground_y: float, meta: dict) -> CanonicalMotion:
    """Build from SMPL-convention data (Y up, +Z forward): rot_mats (F, 22, 3, 3) LOCAL rotations,
    joints_world (F, 22, 3) posed joints (for the pelvis trajectory), rest_smpl (22, 3), ground_y = the
    rest body's sole height (the sole goes to z = 0)."""
    C = C_SMPL
    R = np.einsum("ab,fjbc,dc->fjad", C, rot_mats, C)
    rest = rest_smpl @ C.T
    rest[:, 2] -= ground_y
    trans = joints_world[:, 0] @ C.T
    cm = CanonicalMotion(mat_to_quat(R), trans, rest, fps, meta)
    return cm
