"""UE side: body poses built from the character's own skeleton (no clip needed): one-frame AnimSequences.

A pose is a list of steps applied in order to the character's current bone positions (its reference pose when
no animation plays), in the ACTOR frame: x = the character's left, y = forward, z = up, centimetres.
  ["aim", bone, child, [dx, dy, dz]]   turn `bone` (and everything below it) so that bone -> child points along d
  ["move", bone, [dx, dy, dz]]          translate `bone` and everything below it
Every bone whose transform changed is keyed (local, parent-relative) into a two-frame AnimSequence on the body's
skeleton; other bones keep the reference pose. The math is plain quaternions here, independent of the engine's
bone axis conventions: only joint positions are aimed, so the same step works on any humanoid with these names.

More steps: ["rot", bone, [ax, ay, az], deg] (turn a bone's subtree about an actor-frame axis; about z, positive
turns toward the character's right), ["ik", upper, lower, end, [x, y, z], [pole x, y, z]] (two-bone chain to a
target), ["curl", side, deg, thumb_deg] (relaxed fingers).

Named poses (POSES) are functions of the measured skeleton plus a few parameters (seat and shelf heights, which
leg takes the weight, head and chest turns): stand, stand_relaxed, stand_hand_on_mantel, gesture_point, seated,
sit_armchair_relaxed. Feet keep their standing positions (two-bone IK) when the hips move.
"""

import math

import unreal

EAL = unreal.EditorAssetLibrary
AT = unreal.AssetToolsHelpers.get_asset_tools()


# ------------------------------------------------------------------ quaternions (x, y, z, w)
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def qinv(q):
    return (-q[0], -q[1], -q[2], q[3])


def qrot(q, v):
    p = qmul(qmul(q, (v[0], v[1], v[2], 0.0)), qinv(q))
    return [p[0], p[1], p[2]]


def qnorm(q):
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v]


def q_between(a, b):
    a, b = unit(a), unit(b)
    d = sum(a[i] * b[i] for i in range(3))
    if d < -0.99999:
        ax = [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]
        c = unit([a[1] * ax[2] - a[2] * ax[1], a[2] * ax[0] - a[0] * ax[2], a[0] * ax[1] - a[1] * ax[0]])
        return (c[0], c[1], c[2], 0.0)
    c = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    return qnorm((c[0], c[1], c[2], 1.0 + d))


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


# ------------------------------------------------------------------ skeleton state
class Skel:
    """Bone positions/rotations of a posed body in the actor frame (x left, y forward, z up)."""

    def __init__(self, actor, body):
        self.body = body
        loc = actor.get_actor_location()
        yaw = math.radians(actor.get_actor_rotation().yaw)
        # actor frame: the character faces UE +Y at yaw 0 and its left is UE +X
        self.inv_yaw = (0.0, 0.0, math.sin(-yaw / 2), math.cos(-yaw / 2))
        self.org = [loc.x, loc.y, loc.z]
        n = body.get_num_bones()
        self.names = [str(body.get_bone_name(i)) for i in range(n)]
        self.parent = {}
        for nm in self.names:
            p = str(body.get_parent_bone(nm))
            self.parent[nm] = p if p and p != "None" else None
        self.children = {nm: [] for nm in self.names}
        for nm, p in self.parent.items():
            if p:
                self.children[p].append(nm)
        self.pos, self.rot = {}, {}
        for nm in self.names:
            t = body.get_socket_transform(nm, unreal.RelativeTransformSpace.RTS_WORLD)
            tr, q = t.translation, t.rotation
            self.pos[nm] = qrot(self.inv_yaw, sub([tr.x, tr.y, tr.z], self.org))
            self.rot[nm] = qmul(self.inv_yaw, (q.x, q.y, q.z, q.w))
        self.ref_pos = dict(self.pos)
        self.ref_rot = dict(self.rot)
        self.changed = set()

    def subtree(self, bone):
        out, stack = [], [bone]
        while stack:
            b = stack.pop()
            out.append(b)
            stack.extend(self.children[b])
        return out

    def aim(self, bone, child, d):
        q = q_between(sub(self.pos[child], self.pos[bone]), d)
        piv = self.pos[bone]
        for b in self.subtree(bone):
            self.pos[b] = add(piv, qrot(q, sub(self.pos[b], piv)))
            self.rot[b] = qnorm(qmul(q, self.rot[b]))
            self.changed.add(b)

    def move(self, bone, d):
        for b in self.subtree(bone):
            self.pos[b] = add(self.pos[b], d)
            self.changed.add(b)

    def local(self, b):
        p = self.parent[b]
        if p is None:
            return self.pos[b], self.rot[b]
        ip = qinv(self.rot[p])
        return qrot(ip, sub(self.pos[b], self.pos[p])), qnorm(qmul(ip, self.rot[b]))

    def length(self, a, b):
        return math.dist(self.ref_pos[a], self.ref_pos[b])


# ------------------------------------------------------------------ more mechanics
def q_axis(axis, deg):
    ax = unit(axis)
    h = math.radians(deg) / 2
    return (ax[0] * math.sin(h), ax[1] * math.sin(h), ax[2] * math.sin(h), math.cos(h))


def _rot(s, bone, axis, deg):
    """Turn `bone` and everything below it by deg about an actor-frame axis through the bone."""
    q = q_axis(axis, deg)
    piv = s.pos[bone]
    for b in s.subtree(bone):
        s.pos[b] = add(piv, qrot(q, sub(s.pos[b], piv)))
        s.rot[b] = qnorm(qmul(q, s.rot[b]))
        s.changed.add(b)


def _ik(s, upper, lower, end, target, pole):
    """Two-bone chain: put `end` at target (actor frame, cm) with the middle joint bending toward `pole`."""
    a, b = s.length(upper, lower), s.length(lower, end)
    root = s.pos[upper]
    d = sub(target, root)
    dist = min(max(math.sqrt(sum(c * c for c in d)), abs(a - b) + 0.1), a + b - 0.1)
    dn = unit(d)
    pv = sub(pole, root)
    pv = sub(pv, [c * sum(pv[i] * dn[i] for i in range(3)) for c in dn])        # pole across the chain
    pn = unit(pv) if sum(c * c for c in pv) > 1e-6 else [0.0, 0.0, 1.0]
    x = (a * a - b * b + dist * dist) / (2 * dist)
    h = math.sqrt(max(a * a - x * x, 0.0))
    mid = add(root, add([c * x for c in dn], [c * h for c in pn]))
    s.aim(upper, lower, sub(mid, root))
    s.aim(lower, end, sub(add(root, [c * dist for c in dn]), s.pos[lower]))


FINGERS = ("index", "middle", "ring", "pinky")


def _curl(s, side, deg, thumb_deg=None, fingers=FINGERS):
    """Relaxed fingers: each finger joint turned toward the palm about the knuckle line (index -> pinky)."""
    k = unit(sub(s.pos[f"pinky_01_{side}"], s.pos[f"index_01_{side}"]))
    # palm side: the curl that moves the middle fingertip toward the thumb base is the closing one
    sign = 1.0
    tip, base = f"middle_03_{side}", f"thumb_01_{side}"
    if tip in s.pos and base in s.pos:
        q = q_axis(k, deg)
        piv = s.pos[f"middle_01_{side}"]
        moved = add(piv, qrot(q, sub(s.pos[tip], piv)))
        if math.dist(moved, s.pos[base]) > math.dist(s.pos[tip], s.pos[base]):
            sign = -1.0
    for f in fingers:
        for j in ("01", "02", "03"):
            bone = f"{f}_{j}_{side}"
            if bone in s.pos:
                _rot(s, bone, k, sign * deg * (0.6 if j == "01" else 1.0))
    if thumb_deg and f"thumb_02_{side}" in s.pos:
        _rot(s, f"thumb_02_{side}", k, sign * thumb_deg)


def _feet_back(s, ankles):
    for side in ("l", "r"):
        s.aim(f"foot_{side}", f"ball_{side}", sub(s.ref_pos[f"ball_{side}"], s.ref_pos[f"foot_{side}"]))


def _arm_hang(s, side, out=0.03, fwd=0.04, bend=0.18):
    sx = 1.0 if side == "l" else -1.0
    s.aim(f"upperarm_{side}", f"lowerarm_{side}", [sx * out, fwd, -1.0])
    s.aim(f"lowerarm_{side}", f"hand_{side}", [sx * out * 0.5, bend, -1.0])
    s.aim(f"hand_{side}", f"middle_01_{side}", [sx * 0.05, 0.12, -1.0])


def _weight_on(s, side, shift_cm=3.0, knee_cm=6.0):
    """Contrapposto: hips slide over the standing leg, the free knee bends forward, both feet stay put."""
    ankles = {x: list(s.pos[f"foot_{x}"]) for x in ("l", "r")}
    sx = 1.0 if side == "l" else -1.0
    s.move("pelvis", [sx * shift_cm, 0.0, -1.0])
    _rot(s, "pelvis", [0, 1, 0], -sx * 3.0)                  # the free hip drops a little
    _rot(s, "spine_01", [0, 1, 0], sx * 4.0)                  # the spine answers back up
    free = "r" if side == "l" else "l"
    for x in ("l", "r"):
        knee = s.pos[f"calf_{x}"]
        pole = add(knee, [0.0, 40.0 + (knee_cm * 4 if x == free else 0.0), 0.0])
        tgt = ankles[x] if x == side else add(ankles[x], [0.0, knee_cm * 0.4, 0.0])
        _ik(s, f"thigh_{x}", f"calf_{x}", f"foot_{x}", tgt, pole)
    _feet_back(s, ankles)


def _turn(s, chest_deg=0.0, head_deg=0.0, head_tilt=0.0):
    """Shoulders turned off square (spread over the spine), head turned and tilted."""
    if chest_deg:
        for b, f in (("spine_02", 0.3), ("spine_03", 0.35), ("spine_04", 0.35)):
            _rot(s, b, [0, 0, 1], chest_deg * f)
    if head_deg:
        _rot(s, "neck_01", [0, 0, 1], head_deg * 0.4)
        _rot(s, "head", [0, 0, 1], head_deg * 0.6)
    if head_tilt:
        _rot(s, "head", [1, 0, 0], head_tilt)


# ------------------------------------------------------------------ named poses
ARM_OUT = 0.04          # the hanging direction's sideways and forward parts per unit down
ARM_FWD = 0.05
ARM_CLEAR_DEG = 12.0    # then each upper arm turned out from the body and each forearm forward (the character
FOREARM_FWD_DEG = 10.0  # builder's measured clearance, 11-11): a sleeve-to-coat gap the still and the depth pass show


def _arm_clear(s, p, sides=("l", "r")):
    """Arms away from the body: hanging flush against a long coat, the sleeves merged into the coat's outline in the
    still and the depth pass, and the redraw drew a man with no arms (11-11). With the clearance the person mask
    shows a gap in 9-10 of 10 forearm rows at wide (0 of 10 before)."""
    a, f = p.get("arm_clear_deg", ARM_CLEAR_DEG), p.get("forearm_fwd_deg", FOREARM_FWD_DEG)
    for side in sides:
        sx = 1.0 if side == "l" else -1.0
        _rot(s, f"upperarm_{side}", [0, 1, 0], -sx * a)
        _rot(s, f"lowerarm_{side}", [1, 0, 0], f)


def pose_stand(s, p):
    """Arms down at the sides (the reference A-pose relaxed), square to the front."""
    _arm_hang(s, "l", p.get("arm_out", ARM_OUT), p.get("arm_fwd", ARM_FWD))
    _arm_hang(s, "r", p.get("arm_out", ARM_OUT), p.get("arm_fwd", ARM_FWD))
    _arm_clear(s, p)
    _curl(s, "l", 22, 10)
    _curl(s, "r", 22, 10)


def pose_stand_relaxed(s, p):
    """Weight on one leg, shoulders a little off square, head turned toward where he looks, arms hanging with
    soft elbows, one hand resting at the lapel when hand_at_lapel."""
    side = p.get("weight_on", "r")
    _weight_on(s, side, p.get("shift_cm", 3.0), p.get("knee_cm", 6.0))
    _turn(s, p.get("chest_deg", 8.0), 0.0)
    out, fwd = p.get("arm_out", ARM_OUT), p.get("arm_fwd", ARM_FWD)
    _arm_hang(s, "l", out, fwd, 0.22)
    _arm_hang(s, "r", out, fwd, 0.22)
    _arm_clear(s, p, ("r",) if p.get("hand_at_lapel", True) else ("l", "r"))
    if p.get("hand_at_lapel", True):
        sh = s.pos["clavicle_l"]
        chest = add(s.pos["spine_05"], [3.0, 12.0, -14.0])
        _ik(s, "upperarm_l", "lowerarm_l", "hand_l", chest, add(s.pos["lowerarm_l"], [25.0, -10.0, -20.0]))
        s.aim("hand_l", "middle_01_l", [-0.35, 0.2, 0.9])
    _curl(s, "l", 30, 15)
    _curl(s, "r", 25, 12)
    _turn(s, 0.0, p.get("head_deg", 12.0), p.get("head_tilt", -3.0))


def pose_stand_hand_on_mantel(s, p):
    """One hand resting on a shelf at shelf_z_m beside him (side: which hand), weight on the far leg."""
    hand = p.get("hand", "r")
    other = "l" if hand == "r" else "r"
    _weight_on(s, other, 2.5, 5.0)
    _turn(s, p.get("chest_deg", -6.0 if hand == "r" else 6.0), 0.0)
    sx = 1.0 if hand == "l" else -1.0
    z = p.get("shelf_z_m", 1.25) * 100 + 3.0
    tgt = [s.pos[f"clavicle_{hand}"][0] + sx * p.get("reach_out_cm", 38.0),
           s.pos[f"clavicle_{hand}"][1] + p.get("reach_fwd_cm", 8.0), z]
    _ik(s, f"upperarm_{hand}", f"lowerarm_{hand}", f"hand_{hand}", tgt,
        add(s.pos[f"lowerarm_{hand}"], [sx * 20.0, -30.0, -30.0]))
    s.aim(f"hand_{hand}", f"middle_01_{hand}", [sx * 0.5, 0.85, -0.15])
    _arm_hang(s, other, p.get("arm_out", ARM_OUT), p.get("arm_fwd", ARM_FWD), 0.22)
    _curl(s, hand, 18, 8)
    _curl(s, other, 28, 12)
    _turn(s, 0.0, p.get("head_deg", 0.0), p.get("head_tilt", -2.0))


def pose_gesture_point(s, p):
    """Weight on one leg, the right arm raised toward a point ahead (point_dir, actor frame), index extended."""
    _weight_on(s, "l", 3.0, 5.0)
    _turn(s, p.get("chest_deg", 10.0), 0.0)
    d = unit(p.get("point_dir", [-0.4, 1.0, 0.15]))
    sh = s.pos["upperarm_r"]
    reach = s.length("upperarm_r", "lowerarm_r") + s.length("lowerarm_r", "hand_r")
    tgt = add(sh, [c * reach * 0.93 for c in d])
    _ik(s, "upperarm_r", "lowerarm_r", "hand_r", tgt, add(s.pos["lowerarm_r"], [-20.0, 0.0, -40.0]))
    s.aim("hand_r", "middle_01_r", d)
    _curl(s, "r", 70, 25, fingers=("middle", "ring", "pinky"))       # a pointing hand: index left straight
    _arm_hang(s, "l", p.get("arm_out", ARM_OUT), p.get("arm_fwd", ARM_FWD), 0.22)
    _curl(s, "l", 28, 12)
    _turn(s, 0.0, p.get("head_deg", 10.0), 0.0)


def pose_seated(s, p):
    """Hips at seat_z + clearance, knees forward, shins vertical with the feet on the floor at their standing
    height, the back leaning into the chair back (lean_deg)."""
    seat_z = p.get("seat_z_m", 0.45) * 100
    clear = p.get("thigh_clearance_m", 0.10) * 100
    hip = (s.ref_pos["thigh_l"][2] + s.ref_pos["thigh_r"][2]) / 2
    s.move("pelvis", [0.0, 0.0, seat_z + clear - hip])
    for side in ("l", "r"):
        sx = 1.0 if side == "l" else -1.0
        thigh = s.length(f"thigh_{side}", f"calf_{side}")
        shin = s.length(f"calf_{side}", f"foot_{side}")
        ankle_z = s.ref_pos[f"foot_{side}"][2]
        knee_z = ankle_z + shin
        hz = s.pos[f"thigh_{side}"][2]
        dz = knee_z - hz
        dy = math.sqrt(max(thigh * thigh - dz * dz, 1.0))
        spread = p.get("knee_spread_m", 0.07) * 100 * (1.3 if side == p.get("open_knee", "l") else 1.0)
        s.aim(f"thigh_{side}", f"calf_{side}", [sx * spread, dy, dz])
        s.aim(f"calf_{side}", f"foot_{side}", [sx * 0.04, p.get("shin_fwd", 0.18) * (1.5 if side == p.get("open_knee", "l") else 1.0), -1.0])
    _feet_back(s, None)
    lean = math.radians(p.get("lean_deg", 12))
    s.aim("spine_01", "spine_05", [0.0, -math.sin(lean), math.cos(lean)])
    _rot(s, "neck_01", [1, 0, 0], p.get("chin_deg", 8.0))      # head back up after the lean


def pose_sit_armchair_relaxed(s, p):
    """Seated back in an armchair, forearms along the chair arms (arm_z_m high, arm_x_m out from his centre line,
    hands near their front ends), fingers relaxed, head turned a little."""
    pose_seated(s, p)
    z = p.get("arm_z_m", 0.67) * 100 + p.get("forearm_cm", 4.0)
    for side in ("l", "r"):
        sx = 1.0 if side == "l" else -1.0
        x = s.pos["pelvis"][0] + sx * p.get("arm_x_m", 0.31) * 100
        y = s.pos["pelvis"][1] + p.get("hand_fwd_m", 0.34) * 100
        _ik(s, f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", [x, y, z],
            add(s.pos[f"lowerarm_{side}"], [sx * 10.0, -40.0, -20.0]))
        s.aim(f"hand_{side}", f"middle_01_{side}", [sx * 0.1, 1.0, -0.35])
        _curl(s, side, 32, 12)
    _turn(s, 0.0, p.get("head_deg", 10.0), p.get("head_tilt", 0.0))


POSES = {"stand": pose_stand, "stand_relaxed": pose_stand_relaxed, "stand_hand_on_mantel": pose_stand_hand_on_mantel,
         "gesture_point": pose_gesture_point, "seated": pose_seated, "sit_armchair_relaxed": pose_sit_armchair_relaxed}


def build(actor, body, pose, params, path):
    """Pose name + parameters -> AnimSequence at `path` (replaced). Returns (anim, info)."""
    s = Skel(actor, body)
    POSES[pose](s, params or {})
    for step in (params or {}).get("steps", []):
        if step[0] == "aim":
            s.aim(step[1], step[2], step[3])
        elif step[0] == "move":
            s.move(step[1], step[2])
        elif step[0] == "rot":
            _rot(s, step[1], step[2], step[3])
        elif step[0] == "ik":
            _ik(s, step[1], step[2], step[3], step[4], step[5])
        elif step[0] == "curl":
            _curl(s, step[1], step[2], step[3] if len(step) > 3 else None)
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    folder, name = path.rsplit("/", 1)
    f = unreal.AnimSequenceFactory()
    f.set_editor_property("target_skeleton", body.get_skinned_asset().get_editor_property("skeleton"))
    anim = AT.create_asset(name, folder, unreal.AnimSequence, f)
    # the keys are this body's own bone translations: with no retarget source the skeleton's translation
    # retargeting scaled them a second time on a body taller than the base (posed figure +15 %, long neck)
    anim.set_editor_property("retarget_source_asset", body.get_skinned_asset())
    ctrl = anim.get_editor_property("controller")
    ctrl.open_bracket("pose", False)
    ctrl.set_frame_rate(unreal.FrameRate(30, 1), False)
    ctrl.set_number_of_frames(unreal.FrameNumber(2), False)
    keyed = 0
    for b in s.names:
        if b not in s.changed:
            continue
        lp, lq = s.local(b)
        pos = [unreal.Vector(*lp)] * 3
        rot = [unreal.Quat(lq[0], lq[1], lq[2], lq[3])] * 3
        scl = [unreal.Vector(1, 1, 1)] * 3
        try:
            ctrl.add_bone_curve(b, False)
        except Exception:
            ctrl.add_bone_track(b, False)
        ctrl.set_bone_track_keys(b, pos, rot, scl, False)
        keyed += 1
    ctrl.close_bracket(False)
    EAL.save_loaded_asset(anim, False)
    info = {"pose": pose, "keyed_bones": keyed,
            "pelvis_z_cm": round(s.pos["pelvis"][2], 1),
            "feet_z_cm": round(min(s.pos[b][2] for b in ("foot_l", "foot_r", "ball_l", "ball_r")), 1)}
    return anim, info
