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
sit_armchair_relaxed. Feet keep their standing positions (two-bone IK) when the hips move. Start actions over a
posture (ACTIONS): veil_lift, the work actions any task maps onto: tool_raised, striking_down, hands_at_work, and a
fighter's guard: weapon_ready (READY_ACTIONS; the same stance solver, no work top).
The work actions set the whole body by its mechanics (see "work: whole-body mechanics"): a staggered stance with
bent knees, the trunk turned and folded from the hips, elbows never straight, the centre of mass kept over the feet
(balance_margin), the head on the work, the tool gripped (grip_frame, prop_in_hand: scene_job puts the held prop in
the palm from build()'s info["tool"]). The math runs without the engine (Skel.from_joints; tests/test_work_pose_1117).
Held things (11-18; tests/test_held_props_1118): every thing in params["held"] is gripped after any posture (_hold),
reported in info["held"] by asset for scene_job to place (prop_in_hand); a blow (striking_down) puts the tool's
working point (a head's face, else its tip; read from the model by grip_from_points) on params["work_target_ue"],
and held_contact / contact_why are the staged check on the placed prop.
"""

import math

try:
    import unreal
except ImportError:         # the pose math is plain Python: tests drive it on a synthetic skeleton (Skel.from_joints)
    unreal = None

EAL = unreal.EditorAssetLibrary if unreal else None
AT = unreal.AssetToolsHelpers.get_asset_tools() if unreal else None


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
        self.info = {}                     # what a pose step reports (work stance, balance, the held tool)

    @classmethod
    def from_joints(cls, pos, parent):
        """A skeleton from joint positions alone (actor frame, cm; parent: bone -> parent or None), reference
        rotations identity: the pose math off the engine (tests)."""
        s = cls.__new__(cls)
        s.body, s.inv_yaw, s.org = None, (0.0, 0.0, 0.0, 1.0), [0.0, 0.0, 0.0]
        s.names = list(pos)
        s.parent = {nm: parent.get(nm) for nm in s.names}
        s.children = {nm: [] for nm in s.names}
        for nm, p in s.parent.items():
            if p:
                s.children[p].append(nm)
        s.pos = {nm: list(v) for nm, v in pos.items()}
        s.rot = {nm: (0.0, 0.0, 0.0, 1.0) for nm in s.names}
        s.ref_pos, s.ref_rot = dict(s.pos), dict(s.rot)
        s.changed, s.info = set(), {}
        return s

    def ref_snapshot(self):
        """The skeleton as first read (its reference pose, the actor's place and turn): a later build starts from it
        (from_ref) when the body already plays a pose (scene_job's re-solve of a missed contact)."""
        return {"names": list(self.names), "parent": dict(self.parent), "pos": dict(self.ref_pos),
                "rot": dict(self.ref_rot), "inv_yaw": self.inv_yaw, "org": list(self.org)}

    @classmethod
    def from_ref(cls, snap, body=None):
        s = cls.from_joints(snap["pos"], snap["parent"])
        s.body, s.inv_yaw, s.org = body, tuple(snap["inv_yaw"]), list(snap["org"])
        s.rot = dict(snap["rot"])
        s.ref_rot = dict(snap["rot"])
        return s

    def world(self, v, point=True):
        """Actor frame -> UE world (cm): a point (with the actor's location) or a direction."""
        w = qrot(qinv(self.inv_yaw), v)
        return add(w, self.org) if point else w

    def local_pt(self, v):
        """UE world point (cm) -> actor frame (the inverse of world)."""
        return qrot(self.inv_yaw, sub(v, self.org))

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
    _rot_q(s, bone, q_axis(axis, deg))


def _rot_q(s, bone, q):
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
    # (judged on a small turn: at a deep curl the tip swings past the thumb on either side and the test chose
    # the opening way, a tool hand's fingers bent back flat)
    sign = 1.0
    tip, base = f"middle_03_{side}", f"thumb_01_{side}"
    if tip in s.pos and base in s.pos:
        q = q_axis(k, 12.0)
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
    # (a left hand that holds something carries it at the side, not at the lapel: 11-18)
    lapel = p.get("hand_at_lapel", not any(it["hand"] == "l" for it in _held_items(p)))
    _arm_clear(s, p, ("r",) if lapel else ("l", "r"))
    if lapel:
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


HAND_JOIN_Z_CM = 112.0   # where two standing people's joined hands meet above the floor (between waist and chest)


def pose_hand_joined(s, p):
    """Standing turned to a partner with one hand held out to theirs, the two hands meeting on the line between the
    two bodies (hand: which; reach_m: how far ahead of this body they meet, half the gap between the two;
    hand_z_m: how high). The other arm hangs. Two who danced as partners stood a pace apart with their arms at
    their sides, as any two who talk (11-26)."""
    hand = p.get("hand", "r")
    other = "l" if hand == "r" else "r"
    _weight_on(s, p.get("weight_on", "l"), p.get("shift_cm", 2.0), p.get("knee_cm", 4.0))
    _turn(s, p.get("chest_deg", 0.0), 0.0)
    _arm_hang(s, other, p.get("arm_out", ARM_OUT), p.get("arm_fwd", ARM_FWD), 0.22)
    _arm_clear(s, p, (other,))
    sx = 1.0 if hand == "l" else -1.0
    tgt = [s.pos["pelvis"][0] + sx * p.get("hand_x_cm", 4.0), s.pos["pelvis"][1] + p.get("reach_m", 0.5) * 100,
           p.get("hand_z_m", HAND_JOIN_Z_CM / 100) * 100]
    _ik(s, f"upperarm_{hand}", f"lowerarm_{hand}", f"hand_{hand}", tgt,
        add(s.pos[f"lowerarm_{hand}"], [sx * 20.0, -10.0, -30.0]))
    s.aim(f"hand_{hand}", f"middle_01_{hand}", [-sx * 0.15, 1.0, 0.1])
    _curl(s, hand, 30, 14)
    _curl(s, other, 25, 12)
    _turn(s, 0.0, p.get("head_deg", 0.0), p.get("head_tilt", 0.0))


POSES = {"hand_joined": pose_hand_joined, "stand": pose_stand, "stand_relaxed": pose_stand_relaxed, "stand_hand_on_mantel": pose_stand_hand_on_mantel,
         "gesture_point": pose_gesture_point, "seated": pose_seated, "sit_armchair_relaxed": pose_sit_armchair_relaxed}


def act_veil_lift(s, p):
    """Starting to lift a veil: both hands up at the veil's hem under the chin, fingers curled on it, elbows down
    and out (the hand-authored start pose's targets, here from the body's own head: 11-12)."""
    h = s.pos["head"]
    for side in ("l", "r"):
        sx = 1.0 if side == "l" else -1.0
        # the assets' pose solver's numbers (characters/poses_extra.veil_lift, t = 0), from this body's own head
        tgt = [h[0] + sx * 8.0, h[1] + p.get("veil_hand_fwd_cm", 14.0), h[2] - p.get("veil_hand_down_cm", 9.0)]
        _ik(s, f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", tgt, [h[0] + sx * 35.0, h[1] + 20.0, h[2] - 45.0])
        s.aim(f"hand_{side}", f"middle_01_{side}", [-sx * 0.55, 0.35, 0.75])
        _curl(s, side, 45, 20)


# ------------------------------------------------------------------ work: whole-body mechanics (11-17)
# A person at work stands in a working stance, not to attention: the foot opposite the tool hand a step forward, the
# feet about shoulder width apart and turned out a little, the hips lowered over bent knees, pelvis and chest turned
# (wound up toward the tool side to strike, squared over the work at the blow), the lean spread over the hips and the
# spine, the head looking at the work. No arm is ever straight: a hand the body cannot reach with a bent elbow brings
# the body to it (more lean, lower hips) and the elbows bend down and out. The centre of mass (body-segment table) is
# kept over the feet by moving the hips. The tool hand holds the tool (grip_frame / prop_in_hand): its handle across
# the palm, its head beyond the thumb, the wrist bent no more than WRIST_MAX.

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def scale(v, k):
    return [c * k for c in v]


def norm(v):
    return math.sqrt(dot(v, v))


def perp(v, axis):
    """v without its part along the unit axis."""
    return sub(v, scale(axis, dot(v, axis)))


def angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(unit(a), unit(b))))))


SEATED = ("seated", "sit_armchair_relaxed")
WORK_ACTIONS = ("tool_raised", "striking_down", "hands_at_work")
REACH = 0.92            # a hand target farther than this share of the arm's length is reached with the body
WRIST_MAX = 60.0        # degrees the hand may bend off the forearm's line
KNEE_REACH = 0.97       # each leg at most this share of its length from hip to ankle: the knees always bend
BALANCE_MARGIN_CM = 3.0  # the centre of mass this far inside the feet's support polygon
TWO_HANDED_M = 0.9      # a held thing longer than this is held with both hands
CLAV_MAX = 20.0         # degrees the shoulder girdle lifts when the hand goes above the shoulder
LEAN_MAX = 80.0         # the deepest forward fold (hips and spine together) for a low work top
DROP_MAX = 0.2          # the lowest the hips go (share of stature, a deep squat) before the arms fall short
# stance, in shares of the stature H (a 1.75 m figure: front foot +12 cm, back foot -17.5 cm, hips down 5-14 cm)
STANCE_FRONT, STANCE_BACK = 0.09, 0.08
TOE_OUT_FRONT, TOE_OUT_BACK = 10.0, 18.0
# per action: pelvis turn and chest twist toward the tool side (deg; negative turns back toward square), the lean at
# a work top at hip height plus per cm the top lies under the hips, and the tool's direction (grip -> head) in the
# chest's frame as (forward, up, out toward the tool side)
WORK_SPEC = {
    "tool_raised": {"pelvis": 14.0, "chest": 16.0, "lean": 8.0, "lean_cm": 0.25, "tool": (-0.75, 0.6, 0.25)},
    "striking_down": {"pelvis": 8.0, "chest": -8.0, "lean": 16.0, "lean_cm": 0.45, "tool": (0.85, -0.42, -0.2)},
    "hands_at_work": {"pelvis": 6.0, "chest": -6.0, "lean": 12.0, "lean_cm": 0.45, "tool": (0.45, -0.1, -0.88)},
    "weapon_ready": {"pelvis": 16.0, "chest": -6.0, "lean": 6.0, "lean_cm": 0.0, "tool": (0.87, 0.5, 0.0)},
}
# a fighter's start (11-20): squared up to the foe in a guard, the held thing in front at the ready and pointed at
# the foe. (Held up beside the head it read as a stick held up like a torch, and in a close shot it lay behind the
# head and was redrawn as a club; the clip plays the blow, its start frame needs only the guard.) The same stance
# solver as the work actions, with no work top: the weapon hand READY_AHEAD of the hips toward the target, at
# READY_HEIGHT of the way from the hips to the upper chest; the handle along the line to the target (params
# "aim_target_ue", UE cm: the foe's head or middle), else straight ahead READY_UP_DEG up; a two-handed thing in both
# hands on its grip, the free hand of a one-handed thing forward and out for balance; the head on the target.
READY_ACTIONS = ("weapon_ready",)
READY_AHEAD = 0.21          # share of the stature ahead of the hips (a 1.75 m figure: 37 cm)
READY_HEIGHT = 0.45
READY_UP_DEG = 30.0
READY_SIDE = (0.06, 0.015)  # the weapon hand out toward its own side (one hand, two hands), share of the stature
# the weapon elbow's sides tried (out, forward, up): down and out, so the forearm runs forward and in under the
# handle and the wrist can lay it along the line to the target
READY_POLES = ((0.6, -0.2, -0.8), (0.85, 0.0, -0.5), (0.35, 0.1, -0.95), (0.95, 0.2, -0.25), (0.5, -0.5, -0.7))

# body-segment masses (Winter 2009, after Dempster): (share of body mass, proximal joint, distal joint, the segment's
# centre of mass as a fraction of the way from proximal to distal). The head's centre lies past the head joint
# (the skull's middle, ~0.6 of the neck's length above it); the trunk is pelvis, abdomen and thorax.
SEGMENTS = [(0.081, "neck_01", "head", 1.6), (0.216, "spine_03", "neck_01", 0.5),
            (0.139, "spine_01", "spine_03", 0.5), (0.142, "pelvis", "spine_01", 0.0)]
for _x in ("l", "r"):
    SEGMENTS += [(0.028, f"upperarm_{_x}", f"lowerarm_{_x}", 0.436), (0.016, f"lowerarm_{_x}", f"hand_{_x}", 0.43),
                 (0.006, f"hand_{_x}", f"middle_01_{_x}", 0.5), (0.100, f"thigh_{_x}", f"calf_{_x}", 0.433),
                 (0.0465, f"calf_{_x}", f"foot_{_x}", 0.433), (0.0145, f"foot_{_x}", f"ball_{_x}", 0.5)]


def centre_of_mass(pos):
    """Whole-body centre of mass from joint positions (any frame, z up) by the segment table."""
    tot, c = 0.0, [0.0, 0.0, 0.0]
    for m, a, b, f in SEGMENTS:
        if a in pos and b in pos:
            p = add(pos[a], scale(sub(pos[b], pos[a]), f))
            c = add(c, scale(p, m))
            tot += m
    return scale(c, 1.0 / tot)


def foot_print(pos, side):
    """A foot's outline on the floor (xy) from its ankle and ball joints: heel behind the ankle, toe past the ball."""
    a, b = pos[f"foot_{side}"], pos[f"ball_{side}"]
    u = [b[0] - a[0], b[1] - a[1]]
    ln = math.hypot(*u) or 1.0
    u = [u[0] / ln, u[1] / ln]
    w = [-u[1], u[0]]
    hw = 0.33 * ln

    def at(base, along, wide):
        return [base[0] + u[0] * along + w[0] * wide, base[1] + u[1] * along + w[1] * wide]
    return [at(a, -0.45 * ln, hw * 0.8), at(a, -0.45 * ln, -hw * 0.8), at(b, 0.0, hw * 1.1), at(b, 0.0, -hw * 1.1),
            at(b, 0.55 * ln, hw * 0.5), at(b, 0.55 * ln, -hw * 0.5)]


def convex_hull(pts):
    """Counter-clockwise hull (monotone chain) of 2D points."""
    p = sorted(set((round(x, 6), round(y, 6)) for x, y in pts))
    if len(p) < 3:
        return [list(q) for q in p]

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for q in p:
        while len(lo) >= 2 and turn(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(hi) >= 2 and turn(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    return [list(q) for q in lo[:-1] + hi[:-1]]


def polygon_margin(pt, poly):
    """How far a point lies inside a counter-clockwise convex polygon (its distance to the nearest edge line):
    positive inside, negative outside."""
    m = float("inf")
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = [b[0] - a[0], b[1] - a[1]]
        ln = math.hypot(*e) or 1.0
        m = min(m, (e[0] * (pt[1] - a[1]) - e[1] * (pt[0] - a[0])) / ln)
    return m


def balance_margin(pos):
    """-> (margin cm, centre of mass xy, support polygon): the centre of mass projected to the floor against the
    convex hull of both feet's outlines (both feet carry weight). Pure: joint positions in, numbers out."""
    poly = convex_hull(foot_print(pos, "l") + foot_print(pos, "r"))
    c = centre_of_mass(pos)
    return polygon_margin(c[:2], poly), c[:2], poly


def grip_frame(size_m, two_hand_m=TWO_HANDED_M):
    """Where a held prop is gripped, from its recorded size alone (metres, mesh axes x, y, z; origin at the bottom
    centre). The handle lies along the longest axis. Which end is the handle cannot be told from a size, so by
    convention: an upright model (longest axis z) is held at its BOTTOM end and its head is the top (a tool is
    modelled and pictured head up, standing on its handle's end); a lying one (longest axis x or y) at its negative
    end. The hand grips a short way in from that end (15 % of the length, at least 5 cm so the hand stays on it);
    a two-handed thing (longer than two_hand_m) is held by the leading hand at 30 % and the other hand at 7 %. The
    second-longest axis is the head's own length: the way a striking face points (along the hand, wrist to knuckles)."""
    i = max(range(3), key=lambda a: size_m[a])
    j = max((a for a in range(3) if a != i), key=lambda a: size_m[a])
    L = float(size_m[i])
    two = L > two_hand_m
    gd = 0.30 * L if two else min(max(0.15 * L, 0.05), 0.4 * L)
    od = max(0.05, 0.07 * L) if two else None
    start = 0.0 if i == 2 else -L / 2
    grip = [0.0, 0.0, float(size_m[2]) / 2]          # the handle's line: through the middle of the other two axes
    grip[i] = start + gd
    off = None
    if two:
        off = list(grip)
        off[i] = start + od
    # what it works with (11-18), from the size alone: a head (a hammer's, a sledge's) when the second-longest side
    # is broad against the thinnest and a good share of the length; else the far end is a tip (a sword, a poker).
    # The head's depth along the handle is taken as the thinnest side, its face at the head's +face-axis end.
    k = 3 - i - j
    thin = float(size_m[k])
    far = start + L
    head = float(size_m[j]) >= HEAD_WIDE * thin and float(size_m[j]) >= 0.2 * L
    hc = list(grip)
    hc[i] = far - thin / 2
    tip = list(grip)
    tip[i] = far
    return {"axis": i, "face_axis": j, "len_m": round(L, 4), "grip_m": grip, "off_grip_m": off,
            "grip_from_end_m": round(gd, 4), "off_from_end_m": od, "two_hands": two, "head_m": float(size_m[j]),
            "handle_end": "bottom (z = 0)" if i == 2 else f"negative {'xyz'[i]} end",
            "work": "head" if head else "tip", "head_c_m": hc, "face_half_m": round(float(size_m[j]) / 2, 4),
            "face_sign": 1.0, "tip_m": tip, "handle_m": round(min(max(thin, 0.015), 0.06), 4)}


HAND_THICK_M = 0.07     # a part a hand closes round is at most this thick ...
HAND_ROUND = 0.45       # ... and no flatter than this (thin side / thick side): a blade is not a handle
GRIP_SLICES = 24
HEAD_WIDE = 2.5         # a far end this many handle thicknesses wide is a head (it strikes with a face); else a tip


def grip_from_points(pts_m, two_hand_m=TWO_HANDED_M):
    """Where a held prop is gripped, from its model's own points (metres, mesh axes): the same report as grip_frame,
    with "sign" (+1 / -1: the head lies toward the positive / negative end of the long axis). The model is cut in
    slices along its longest axis; the handle is the longest run of slices a hand can close round (HAND_THICK_M,
    HAND_ROUND); of that run's two ends the one farther from the model's middle of mass is the free end, and the
    hand sits a short way in from it (15 % of the length, at most the run's middle). With no such run the hand
    takes the end farther from the middle of mass. None with too few points."""
    if not pts_m or len(pts_m) < 50:
        return None
    lo = [min(p[a] for p in pts_m) for a in range(3)]
    hi = [max(p[a] for p in pts_m) for a in range(3)]
    size = [hi[a] - lo[a] for a in range(3)]
    i = max(range(3), key=lambda a: size[a])
    j = max((a for a in range(3) if a != i), key=lambda a: size[a])
    k = 3 - i - j
    L = size[i]
    if L < 1e-4:
        return None
    n = GRIP_SLICES
    sl = [[] for _ in range(n)]
    for p in pts_m:
        sl[min(n - 1, int((p[i] - lo[i]) / L * n))].append(p)
    mass = sum(p[i] for p in pts_m) / len(pts_m)
    ok, mid, wide = [], [], []
    for q in sl:
        if len(q) < 4:
            ok.append(False)
            mid.append(None)
            wide.append(0.0)
            continue
        rj = (min(p[j] for p in q), max(p[j] for p in q))
        rk = (min(p[k] for p in q), max(p[k] for p in q))
        thick, thin = max(rj[1] - rj[0], rk[1] - rk[0]), min(rj[1] - rj[0], rk[1] - rk[0])
        ok.append(0.006 <= thick <= HAND_THICK_M and thin / thick >= HAND_ROUND)
        mid.append(((rj[0] + rj[1]) / 2, (rk[0] + rk[1]) / 2))
        wide.append(thick)
    best, a = None, 0
    while a < n:
        if ok[a]:
            b = a
            while b + 1 < n and ok[b + 1]:
                b += 1
            if best is None or b - a > best[1] - best[0]:
                best = (a, b)
            a = b + 1
        else:
            a += 1
    two = L > two_hand_m
    want = 0.30 * L if two else min(max(0.15 * L, 0.05), 0.4 * L)
    if best and best[1] - best[0] >= 1:
        r0, r1 = lo[i] + best[0] * L / n, lo[i] + (best[1] + 1) * L / n
        free = r0 if abs(r0 - mass) > abs(r1 - mass) else r1
        gd = min(want, (r1 - r0) / 2)
        gi = free + gd * (1.0 if free == r0 else -1.0)
    else:
        free = lo[i] if abs(lo[i] - mass) > abs(hi[i] - mass) else hi[i]
        gi = free + want * (1.0 if free == lo[i] else -1.0)
    sign = 1.0 if mass >= gi else -1.0
    end = lo[i] if sign > 0 else hi[i]                       # the handle's own end of the model
    m = mid[min(n - 1, max(0, int((gi - lo[i]) / L * n)))] or ((lo[j] + hi[j]) / 2, (lo[k] + hi[k]) / 2)
    grip = [0.0, 0.0, 0.0]
    grip[i], grip[j], grip[k] = gi, m[0], m[1]
    gfe = abs(gi - end)
    od = max(0.05, 0.07 * L) if two else None
    off = None
    if two:
        off = list(grip)
        off[i] = end + sign * od
    # what it works with (11-18), from the model: its far quarter (the end away from the handle) as wide as
    # HEAD_WIDE handle thicknesses is a head (a hammer's, an axe's), struck with the face at the end of the head's
    # longer side; else the far end is a tip (a sword's, a poker's). The head is the far quarter's slices at least
    # 0.6 of its widest; of the head's two ends the one with more of the model at its extreme is the face (a flat
    # face is broad there, a peen or a claw tapers)
    gs = min(n - 1, max(0, int((gi - lo[i]) / L * n)))
    hm = wide[gs] or min([w for w in wide if w > 0] or [0.03])
    far = hi[i] if sign > 0 else lo[i]
    outer = [q for q in range(n) if sl[q] and abs(lo[i] + (q + 0.5) * L / n - far) <= 0.25 * L]
    end_w = max([wide[q] for q in outer] or [0.0])
    fa, fs, hc, fh = j, 1.0, None, size[j] / 2
    head = bool(outer) and end_w >= HEAD_WIDE * hm
    if head:
        hp = [p for q in outer if wide[q] >= 0.6 * end_w for p in sl[q]]
        hlo = [min(p[a] for p in hp) for a in range(3)]
        hhi = [max(p[a] for p in hp) for a in range(3)]
        hc = [(hlo[a] + hhi[a]) / 2 for a in range(3)]
        fa = j if hhi[j] - hlo[j] >= hhi[k] - hlo[k] else k
        sp = hhi[fa] - hlo[fa]
        fh = sp / 2
        npl = sum(1 for p in hp if p[fa] >= hhi[fa] - 0.15 * sp)
        nmi = sum(1 for p in hp if p[fa] <= hlo[fa] + 0.15 * sp)
        fs = 1.0 if npl >= nmi else -1.0
    qe = max(outer) if sign > 0 and outer else (min(outer) if outer else (n - 1 if sign > 0 else 0))
    me = mid[qe] or m
    tip = [0.0, 0.0, 0.0]
    tip[i], tip[j], tip[k] = far, me[0], me[1]
    return {"axis": i, "face_axis": fa, "sign": sign, "len_m": round(L, 4), "grip_m": [round(v, 4) for v in grip],
            "off_grip_m": off, "grip_from_end_m": round(gfe, 4), "off_from_end_m": od, "two_hands": two,
            "head_m": round(size[j], 4), "size_m": [round(v, 4) for v in size],
            "handle_end": f"{'negative' if sign > 0 else 'positive'} {'xyz'[i]} end", "from": "model",
            "work": "head" if head else "tip", "head_c_m": [round(v, 4) for v in hc] if hc else None,
            "face_half_m": round(fh, 4), "face_sign": fs, "tip_m": [round(v, 4) for v in tip],
            "handle_m": round(min(max(hm, 0.015), 0.06), 4)}


def prop_in_hand(size_m, palm_cm, tool_dir, face_dir, k=1.0, grip=None):
    """A held prop's placement: its handle axis along tool_dir (grip -> head), its head's length along face_dir,
    its grip point at palm_cm. All in one proper frame (UE world, or the actor frame; cm). -> {"x", "y", "z": where
    the mesh's own axes point, "origin_cm": where its origin goes, "grip": grip_frame}. A proper rotation: axis k is
    the cross product of the other two in cyclic order."""
    g = grip or grip_frame(size_m)
    i, j = g["axis"], g["face_axis"]
    t = scale(unit(tool_dir), g.get("sign", 1.0))            # where the mesh's long axis points
    h = perp(face_dir, t)
    h = unit(h) if norm(h) > 1e-6 else unit(perp([0.0, 0.0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0.0, 0.0], t))
    cols = [None, None, None]
    cols[i], cols[j] = t, h
    kk = 3 - i - j
    cols[kk] = cross(cols[i], cols[j]) if (j - i) % 3 == 1 else cross(cols[j], cols[i])
    gp = g["grip_m"]
    o = list(palm_cm)
    for a in range(3):
        o = sub(o, scale(cols[a], gp[a] * 100.0 * k))
    return {"x": cols[0], "y": cols[1], "z": cols[2], "origin_cm": o, "grip": g}


# ------------------------------------------------------------------ held things: grip, working point, check (11-18)
# A hammer hung under an open hand; its face a hand's breadth short of the metal and turned sideways (11-18). Every
# thing in a hand is now gripped (the hand closed round its handle, in any pose), and a blow puts the thing's working
# point (a head's face, a long thing's tip) on the worked thing's top, the face flat on it; held_contact measures the
# placed result like the old blockout check (grip, gap, face angle) and contact_why says what is wrong in words.
HELD_LIMITS = {"grip_cm": 2.0, "gap_cm": 2.0, "angle_deg": 15.0, "rise_cm": 20.0}
BLOW_ACTIONS = ("tool_raised", "striking_down")   # the work actions that bring a held thing down on the worked thing
RESOLVE_TRIES = 2           # a measured miss (gap, face angle) is re-solved this many times before it is a failure
CURL_RANGE = (45.0, 85.0)   # finger curl (deg): a thin handle closes the fist, a thick one opens it
STRIKE_TIP_DEG = 35.0       # a tipped thing comes down onto the work this steep below the level
# the striking elbow's sides tried (out, forward, up), the first the usual: out to the side and a little forward,
# so the forearm slopes down into the blow and the wrist can turn the face flat (an elbow down and back left the
# forearm level and the face 27-43 deg off the work)
STRIKE_POLES = ((0.8, 0.2, -0.1), (0.6, 0.6, -0.1), (1.0, 0.0, 0.1), (0.7, 0.4, 0.0), (0.5, 0.75, -0.05))
HANDS_AHEAD = 0.12          # at the blow the gripping hand stays this share of the stature ahead of the hips


def full_grip(g=None, size_m=None):
    """A grip report with every field the held-thing math reads: a recorded one (an older run's, without the
    working-point fields) is completed from the size's convention."""
    if g and g.get("work") and (g.get("head_c_m") or g["work"] != "head"):
        return g
    base = grip_frame(size_m or (g or {}).get("size_m") or [0.04, 0.12, 0.35])
    return dict(base, **{k: v for k, v in (g or {}).items() if v is not None and k not in
                         ("work", "head_c_m", "face_half_m", "face_sign", "tip_m", "handle_m")})


def work_point_m(g):
    """The thing's working point in its own mesh axes (metres): a head's face centre, else the far tip."""
    if g.get("work") == "head" and g.get("head_c_m"):
        p = list(g["head_c_m"])
        p[g["face_axis"]] += g.get("face_sign", 1.0) * g["face_half_m"]
        return p
    return list(g["tip_m"])


def mesh_point(put, p_m, k=1.0):
    """A point of the placed mesh (its own axes, metres) where prop_in_hand put it (cm)."""
    o = list(put["origin_cm"])
    for a, col in enumerate((put["x"], put["y"], put["z"])):
        o = add(o, scale(col, p_m[a] * 100.0 * k))
    return o


def place_item(g, palm_cm, t, h):
    """prop_in_hand for a thing whose handle runs along t (grip -> working end) and whose working face points along h
    (a head's face may sit at the mesh's negative face-axis end: face_sign)."""
    return prop_in_hand(None, palm_cm, t, scale(h, g.get("face_sign", 1.0)), grip=g)


def work_offset(g, t, h):
    """Where the working point lies from the palm (cm) when the handle runs along t and the face along h."""
    return mesh_point(place_item(g, [0.0, 0.0, 0.0], t, h), work_point_m(g))


def grip_curl(g):
    """Finger curl (deg) for a handle this thick: 3 cm closes the fingers 70 deg, thicker opens them."""
    hm = float(g.get("handle_m") or 0.03)
    return max(CURL_RANGE[0], min(CURL_RANGE[1], 100.0 - 1000.0 * hm))


def seg_dist(p, a, b):
    ab = sub(b, a)
    t = max(0.0, min(1.0, dot(sub(p, a), ab) / max(dot(ab, ab), 1e-9)))
    return norm(sub(p, add(a, scale(ab, t))))


def held_contact(put, palm, target=None, normal=None, name=None, raised=False):
    """The placed thing against the hand and the work (one frame, cm): "grip_cm" the palm point's distance from the
    handle's line round the grip, and for a blow (target given) "gap_cm" the working point's distance from the
    target and, for a head with a surface normal, "angle_deg" between the face's outward normal and -normal; for a
    thing raised to come down on the work (raised) "rise_cm" instead: how far over the target its working point is
    held. "why" the plain words for each over HELD_LIMITS."""
    g = put["grip"]
    cols = (put["x"], put["y"], put["z"])
    gp = mesh_point(put, g["grip_m"])
    ax = scale(cols[g["axis"]], g.get("sign", 1.0))
    half = max(float(g.get("grip_from_end_m") or 0.05), 0.03) * 100.0
    out = {"grip_cm": round(seg_dist(palm, sub(gp, scale(ax, half)), add(gp, scale(ax, half))), 2)}
    if target is not None:
        w = mesh_point(put, work_point_m(g))
        out.update(work=g.get("work"), work_pt=[round(v, 2) for v in w], target=[round(v, 2) for v in target])
        if raised:
            out.update(raised=True, rise_cm=round(dot(sub(w, target), unit(normal or [0.0, 0.0, 1.0])), 2))
        else:
            out["gap_cm"] = round(math.dist(w, target), 2)
            if g.get("work") == "head" and normal is not None:
                fn = scale(cols[g["face_axis"]], g.get("face_sign", 1.0))
                out["angle_deg"] = round(angle_deg(fn, scale(normal, -1.0)), 1)
    out["why"] = contact_why(out, name)
    return out


def _the(name, default):
    nm = str(name or "").strip()
    for a in ("the ", "a ", "an "):
        if nm.lower().startswith(a):
            nm = nm[len(a):]
    return f"the {nm}" if nm else default


def contact_why(c, name=None):
    """Plain-word reasons a held thing's contact is over HELD_LIMITS ("the hammer's face is 9 cm from the work")."""
    nm = _the(name, "the held thing")
    end = "face" if c.get("work") == "head" else "tip"
    why = []
    if (c.get("grip_cm") or 0.0) > HELD_LIMITS["grip_cm"]:
        why.append(f"{nm} is not in the hand: its handle is {c['grip_cm']:.0f} cm from the palm")
    if (c.get("gap_cm") or 0.0) > HELD_LIMITS["gap_cm"]:
        why.append(f"{nm}'s {end} is {c['gap_cm']:.0f} cm from the work")
    if (c.get("angle_deg") or 0.0) > HELD_LIMITS["angle_deg"]:
        why.append(f"{nm}'s face is turned {c['angle_deg']:.0f} deg off the work")
    if c.get("raised") and c.get("rise_cm") is not None and c["rise_cm"] < HELD_LIMITS["rise_cm"]:
        why.append(f"{nm} is not raised over the work: its {end} is {c['rise_cm']:.0f} cm above it")
    return why


# ------------------------------------------------------------------ a measured miss corrected (11-18)
# The first engine run kept a frame whose blow landed 7.75 cm over the worked thing: the pose had aimed at a top
# reckoned from the thing's recorded size, the check measured the top of the thing as the engine laid it. The job now
# aims at the laid top (lie_top_ue) and, when the placed contact still misses, aims again by the measured miss
# (aim_fix) and re-measures, at most RESOLVE_TRIES times, before the miss is a failure.
def lie_top_ue(base_ue, extent_cm):
    """The top of a thing laid on its broadest face (scene_job: its thinnest side upright, its box's bottom centre at
    base_ue): base_ue raised by the thinnest full side of its box (extent_cm: the box's half sides, any order)."""
    return [float(base_ue[0]), float(base_ue[1]), float(base_ue[2]) + 2.0 * min(float(e) for e in extent_cm)]


def needs_resolve(c):
    """A placed contact whose miss a re-solve of the arm can close: the working point off the target, or a face turned
    off it, over HELD_LIMITS (not a raised thing's rise, not a handle out of the palm: the solver puts the palm on it)."""
    c = c or {}
    return (c.get("gap_cm") or 0.0) > HELD_LIMITS["gap_cm"] or (c.get("angle_deg") or 0.0) > HELD_LIMITS["angle_deg"]


def aim_fix(aim, target, got):
    """The next aim of a re-solve: the last aim moved by the measured miss (target - where the point landed), so a
    constant error between the solver's model and the engine's placement (a wrong top, a wrist the engine puts
    elsewhere) is taken out."""
    return [round(float(aim[i]) + float(target[i]) - float(got[i]), 3) for i in range(3)]


def blow_contact_why(action, has_work, contacts, name=None):
    """Why a blow at a worked thing has no measured contact (else []): a start action in BLOW_ACTIONS with a worked
    thing (has_work) must end with a held thing measured against it (a "gap_cm" or a "rise_cm" in one of contacts:
    held_contact's reports); silence there let a frame with no contact pass (11-18)."""
    if action not in BLOW_ACTIONS or not has_work:
        return []
    cs = [c for c in contacts or [] if c]
    if not cs:
        return [f"nothing is in the hand to bring down on the work ({action})"]
    if any(c.get("gap_cm") is not None or c.get("rise_cm") is not None for c in cs):
        return []
    return [f"{_the(name, 'the held thing')} was never measured against the work ({action}): the pose gave no target"]


# ------------------------------------------------------------------ a set piece's part in the hands (11-18)
# A person at work on a piece of the set itself (a lever, a handle, a bar, a rope, a crank: anything standing in the
# place, not a carried thing) stood at it with the hands hanging near it. Now the piece's own surface (its mesh, in
# the world, round the person) is searched for a part a hand closes round, as grip_from_points does for a carried
# thing: thin and roundish (HAND_THICK_M, HAND_ROUND), with room round it for the fingers, along at least a palm's
# width; one or both free hands are put on it with the arm solver, and the palm's distance from the part is the
# staged check (fixture_contact), measured again in the engine like a held thing's.
FIX_INNER_CM = 100.0 * HAND_THICK_M / 2 + 0.5   # a part's surface lies within this of its line ...
FIX_WRAP_CM = 2.5           # ... with this much empty room round it for the fingers
FIX_PALM_CM = 9.0           # ... along at least a palm's width
FIX_STEP_CM = 1.5           # the surface sampled at least this finely (a long face has no vertices in its middle)
FIX_BAND = (0.35, 1.1)      # a workable height for the hands: shares of the stature above the floor
FIX_REACH = 1.6             # a part within this many arm lengths of the shoulders (the body leans to the rest)
FIX_OPEN_CM = 25.0          # nothing of the piece this far over the hand's place on a part (a table's leg under its top)
FIX_REST_FAR = 0.2          # a part farther than this share of the stature from the working point: the hands rest on the top
FIX_REST_THICK_CM = 8.0     # ... half closed, as on a beam
FIX_SEEDS = 600             # at most this many surface points tried as a part's start
FIX_MAX_PARTS = 8


def surface_points(verts, tris, step=FIX_STEP_CM, most=200, centre=None, radius=None):
    """A mesh's surface as points: its vertices plus each triangle filled on a grid no coarser than step (a
    triangle's longest side over `most` steps at most), then thinned to one point per 0.7 step cell. tris: index
    triples, or a flat index list. With centre and radius only the points within radius of centre (and only the
    triangles that come that near)."""
    vs = [list(map(float, v)) for v in verts]
    tl = tris if tris and isinstance(tris[0], (list, tuple)) else [tris[i:i + 3] for i in range(0, len(tris) - 2, 3)]
    keep = (lambda p: True) if centre is None else (lambda p: math.dist(p, centre) <= radius)  # noqa: E731
    out = [v for v in vs if keep(v)]
    for tri in tl:
        a, b, c = (vs[int(i)] for i in tri)
        if centre is not None:
            m = [(a[q] + b[q] + c[q]) / 3 for q in range(3)]
            if math.dist(m, centre) > radius + max(math.dist(m, a), math.dist(m, b), math.dist(m, c)):
                continue
        n = min(most, int(max(math.dist(a, b), math.dist(b, c), math.dist(c, a)) // step))
        for u in range(n + 1):
            for v in range(n + 1 - u):
                w = n - u - v
                if max(u, v, w) == n:                 # a corner: already a vertex
                    continue
                p = [(a[q] * w + b[q] * u + c[q] * v) / n for q in range(3)]
                if keep(p):
                    out.append(p)
    # one point per cell of about the step (a long thin triangle's grid is far finer across it than the step)
    cell, seen, thin = 0.7 * step, set(), []
    for p in out:
        k = (int(math.floor(p[0] / cell)), int(math.floor(p[1] / cell)), int(math.floor(p[2] / cell)))
        if k not in seen:
            seen.add(k)
            thin.append(p)
    return thin


def _main_axis(pts):
    """The direction along which points spread most (power iteration on their covariance)."""
    n = len(pts)
    m = [sum(p[q] for p in pts) / n for q in range(3)]
    C = [[0.0] * 3 for _ in range(3)]
    for p in pts:
        d = sub(p, m)
        for i in range(3):
            for j in range(3):
                C[i][j] += d[i] * d[j]
    v = [0.577, 0.577, 0.577]
    for i in range(3):                            # start off the widest diagonal entry
        if C[i][i] >= max(C[j][j] for j in range(3)):
            v = [0.3, 0.3, 0.3]
            v[i] = 1.0
    for _ in range(30):
        w = [sum(C[i][j] * v[j] for j in range(3)) for i in range(3)]
        if norm(w) < 1e-12:
            return [0.0, 0.0, 1.0]
        v = unit(w)
    return v


def _basis(d):
    a = [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0]
    u = unit(perp(a, d))
    return u, cross(d, u)


class _Grid:
    def __init__(self, pts, cell):
        self.cell, self.g = cell, {}
        for p in pts:
            self.g.setdefault(tuple(int(math.floor(c / cell)) for c in p), []).append(p)

    def near(self, c, r):
        k = int(math.ceil(r / self.cell))
        i0 = [int(math.floor(x / self.cell)) for x in c]
        out = []
        for i in range(i0[0] - k, i0[0] + k + 1):
            for j in range(i0[1] - k, i0[1] + k + 1):
                for m in range(i0[2] - k, i0[2] + k + 1):
                    out += self.g.get((i, j, m), ())
        return out


def _thin_near(near, d):
    """A quick look before the full test: the surface round a point lies close about a line along d through its
    middle (a flat face, a block's corner, spread across it, do not)."""
    m = [sum(q[i] for q in near) / len(near) for i in range(3)]
    return sum(1 for q in near if norm(perp(sub(q, m), d)) > FIX_INNER_CM) <= 0.1 * len(near)


def _part_test(grid, c, d, half=FIX_PALM_CM / 2):
    """Is there a part a hand closes round along d through about c? -> (centre on its line, direction, thickness)
    or None: the surface within FIX_INNER_CM of the line (over a palm's width) re-centred on the line and its
    direction refined twice, then thin (<= HAND_THICK_M) and roundish (HAND_ROUND), present along the whole width,
    and nothing else within FIX_WRAP_CM round it."""
    reach = FIX_INNER_CM + FIX_WRAP_CM + half + 1.0
    near = grid.near(c, reach)
    r2in = FIX_INNER_CM * FIX_INNER_CM

    def split(c, d):
        """Each near point's (along, across x, across y) from c; plain arithmetic (this runs for every seed)."""
        u, v = _basis(d)
        cx, cy, cz = c
        dx, dy, dz = d
        ux, uy, uz = u
        vx, vy, vz = v
        out = []
        for q in near:
            wx, wy, wz = q[0] - cx, q[1] - cy, q[2] - cz
            out.append((wx * dx + wy * dy + wz * dz, wx * ux + wy * uy + wz * uz, wx * vx + wy * vy + wz * vz, q))
        return out
    for _ in range(3):
        inner = [q for s_, x, y, q in split(c, d) if -half <= s_ <= half and x * x + y * y <= r2in]
        if len(inner) < 8:
            return None
        n = len(inner)
        m = [sum(q[0] for q in inner) / n, sum(q[1] for q in inner) / n, sum(q[2] for q in inner) / n]
        c = add(c, perp(sub(m, c), d))
        if n >= 12:                               # (followed only near its line: a short thick piece of a part
            d2 = _main_axis(inner)                # spreads as much across as along)
            d2 = d2 if dot(d2, d) >= 0 else scale(d2, -1.0)
            if dot(d2, d) >= 0.85:
                d = d2
    sp = split(c, d)
    xs, bins, rad = [], set(), 0.0
    for s_, x, y, _q in sp:
        if -half <= s_ <= half:
            r = math.hypot(x, y)
            if r <= FIX_INNER_CM:
                xs.append((x, y))
                bins.add(min(4, int((s_ + half) / (2 * half) * 5)))
                rad = max(rad, r)
    if len(xs) < 8 or len(bins) < 5:
        return None
    # the cross-section's narrowest width and the width square to it (a square post is as thin as its side, a flat
    # part turned in the basis is still flat)
    best = None
    for k in range(18):
        co, si = math.cos(math.radians(10.0 * k)), math.sin(math.radians(10.0 * k))
        e1 = [x * co + y * si for x, y in xs]
        e2 = [y * co - x * si for x, y in xs]
        w1, w2 = max(e1) - min(e1), max(e2) - min(e2)
        if best is None or w1 < best[0]:
            best = (w1, w2)
    thin, thick = min(best), max(best)
    if not (0.6 <= thick <= 100.0 * HAND_THICK_M) or thin / thick < HAND_ROUND:
        return None
    lo2, hi2 = (rad + 0.5) ** 2, (rad + FIX_WRAP_CM) ** 2
    ring = sum(1 for s_, x, y, _q in sp if -half <= s_ <= half and lo2 < x * x + y * y <= hi2)
    if ring > max(2, 0.03 * len(xs)):
        return None
    return c, d, thick


def grip_parts(pts, prefer=None, seeds=FIX_SEEDS, most=FIX_MAX_PARTS):
    """The parts of a piece a hand can close round, from its surface points (one frame, cm; surface_points): each
    {"a", "b": the ends of the line a palm's middle may take on it, "dir" (a -> b), "c" (its middle), "len_cm" (the
    run's whole length), "thick_cm"}. Seeds are tried nearest `prefer` first; a part is followed along its line in
    2 cm steps both ways while the test holds (it may bend: a rope)."""
    if len(pts) < 8:
        return []
    grid = _Grid(pts, 4.0)
    order = sorted(range(len(pts)), key=lambda i: math.dist(pts[i], prefer)) if prefer else list(range(len(pts)))
    stride = max(1, len(order) // seeds)
    parts, tried = [], set()
    for i in order[::stride]:
        p = pts[i]
        vox = (int(p[0] // 2.5), int(p[1] // 2.5), int(p[2] // 2.5))   # one seed per 2.5 cm of surface
        if vox in tried:
            continue
        tried.add(vox)
        if any(seg_dist(p, q["a"], q["b"]) < FIX_INNER_CM + FIX_PALM_CM / 2 for q in parts):
            continue
        near = [q for q in grid.near(p, 6.0) if math.dist(q, p) <= 6.0]
        if len(near) < 8:
            continue
        # the line tried: the way the surface round the point spreads most, then (a thick part spreads about as
        # much across as along) upright and the two level ways square to it
        d0 = _main_axis(near)
        hz = [d0[0], d0[1], 0.0] if math.hypot(d0[0], d0[1]) > 1e-3 else [1.0, 0.0, 0.0]
        hz = unit(hz)
        got = None
        for d in (d0, [0.0, 0.0, 1.0], hz, [-hz[1], hz[0], 0.0]):
            if (d is not d0 and abs(dot(d, d0)) > 0.97) or not _thin_near(near, d):
                continue
            got = _part_test(grid, p, d)
            if got is not None:
                break
        if got is None:
            continue
        c0, d0, thick = got
        ends = []
        for sg in (1.0, -1.0):
            c, d, last = c0, scale(d0, sg), c0
            for _ in range(60):
                nxt = _part_test(grid, add(c, scale(d, 2.0)), d)
                if nxt is None or math.dist(nxt[0], c) < 0.5:
                    break
                c, d = nxt[0], (nxt[1] if dot(nxt[1], d) >= 0 else scale(nxt[1], -1.0))
                last = c
                thick = min(thick, nxt[2])            # (the best-aligned cut is the truest thickness)
            ends.append(last)
        b, a = ends
        L = math.dist(a, b)
        dr = unit(sub(b, a)) if L > 1e-6 else d0
        parts.append({"a": [round(x, 2) for x in a], "b": [round(x, 2) for x in b], "dir": [round(x, 4) for x in dr],
                      "c": [round(x, 2) for x in scale(add(a, b), 0.5)], "len_cm": round(L + FIX_PALM_CM, 1),
                      "thick_cm": round(thick, 2)})
        if len(parts) >= most:
            break
    return parts


def seg_point(p, a, b):
    """The point of the segment a-b nearest p."""
    ab = sub(b, a)
    t = max(0.0, min(1.0, dot(sub(p, a), ab) / max(dot(ab, ab), 1e-9)))
    return add(a, scale(ab, t))


def _open_above(grid, p, d):
    """Nothing of the piece over p (within FIX_OPEN_CM up, a palm's half-width round) but the part's own line d: a
    hand can come onto it there (a table's leg under its top cannot be taken)."""
    if grid is None:
        return True
    for q in grid.near(add(p, [0.0, 0.0, FIX_OPEN_CM / 2]), FIX_OPEN_CM / 2 + 4.0):
        w = sub(q, p)
        if 2.0 < w[2] <= FIX_OPEN_CM and math.hypot(w[0], w[1]) <= FIX_PALM_CM / 2 and norm(perp(w, d)) > FIX_INNER_CM:
            return False
    return True


def fixture_hands(parts, s, free, hs, H, floor, pts=None):
    """Which free hands take hold of which part (actor frame, cm): the part nearest a comfortable working point (in
    front of the chest at FIX_BAND's middle) that lies at a workable height (FIX_BAND) within FIX_REACH arm lengths
    of the shoulders, with nothing of the piece just over the hand's place on it (pts: the piece's surface points;
    _open_above); a part long enough for two palms a shoulder's width apart (or less, as it allows) takes both
    free hands, the one on the body's left at its left end; else the working hand (hs, when free) takes it and the
    other free hand the next part at least a palm's width away. -> {side: {"on" (the palm's point on the part's
    line), "a", "b", "dir", "thick_cm", "t" (the handle direction to grip it with: the thumb toward the body's
    middle and up)}}."""
    if not free or (not parts and not pts):
        return {}
    left, fwd = _body_axes(s)
    sh = scale(add(s.pos["upperarm_l"], s.pos["upperarm_r"]), 0.5)
    arm = s.length("upperarm_r", "lowerarm_r") + s.length("lowerarm_r", "hand_r")
    want = add(sh, scale(fwd, 0.25 * H))
    want[2] = floor + 0.5 * (FIX_BAND[0] + FIX_BAND[1]) * H
    grid = _Grid(pts, 4.0) if pts else None
    ok = []
    for q in parts:
        L = math.dist(q["a"], q["b"])
        cands = [seg_point(want, q["a"], q["b"])] + [add(q["a"], scale(q["dir"], L * k / 8.0)) for k in range(9)]
        cands = [p for p in cands if floor + FIX_BAND[0] * H <= p[2] <= floor + FIX_BAND[1] * H
                 and math.dist(p, sh) <= FIX_REACH * arm]
        cands.sort(key=lambda p: math.dist(p, want))
        near = next((p for p in cands if _open_above(grid, p, q["dir"])), None)
        if near is not None:
            ok.append((math.dist(near, want), near, q))
    ok.sort(key=lambda x: x[0])
    # no part to close a hand round near where the hands work (a made piece's lever is a thick beam, and the only
    # thin part found was a stub at the hip a deep bend away, 11-18): the hands are laid on the piece's own top
    # nearest the working point instead, side by side across the body
    if (not ok or ok[0][0] > FIX_REST_FAR * H) and grid is not None:
        top = sorted((p for p in pts if floor + FIX_BAND[0] * H <= p[2] <= floor + FIX_BAND[1] * H
                      and math.dist(p, sh) <= FIX_REACH * arm), key=lambda p: math.dist(p, want))
        at = next((p for p in top[:400] if _open_above(grid, p, [0.0, 0.0, 1.0])), None)
        if at is not None and (not ok or math.dist(at, want) < ok[0][0] - 5.0):
            ends = []
            for sg in (1.0, -1.0):
                e = list(at)
                for _ in range(20):
                    n = add(e, scale(left, 2.0 * sg))
                    nb = [q for q in grid.near(n, 3.0) if math.dist(q, n) <= 3.0]
                    if not nb or not _open_above(grid, max(nb, key=lambda q: q[2]), [0.0, 0.0, 1.0]):
                        break
                    e = [n[0], n[1], max(q[2] for q in nb)]
                ends.append(e)
            a_, b_ = ends
            d_ = unit(sub(b_, a_)) if math.dist(a_, b_) > 1e-3 else scale(left, -1.0)
            ok = [(math.dist(at, want), list(at), {"a": [round(x, 2) for x in a_], "b": [round(x, 2) for x in b_],
                                                   "dir": d_, "thick_cm": FIX_REST_THICK_CM, "rest": True})]
    if not ok:
        return {}
    _, near, q = ok[0]

    def hand(side, pt, part):
        inward = scale(left, -1.0 if side == "l" else 1.0)
        d = part["dir"]
        t = d if dot(d, add([0.0, 0.0, 1.0], inward)) >= 0 else scale(d, -1.0)
        return {"on": [round(x, 2) for x in pt], "a": part["a"], "b": part["b"], "dir": part["dir"],
                "thick_cm": part["thick_cm"], "t": [round(x, 4) for x in t], **({"rest": True} if part.get("rest") else {})}
    run = math.dist(q["a"], q["b"])
    free = [x for x in ("l", "r") if x in free]
    out = {}
    if len(free) == 2 and run >= FIX_PALM_CM + 4.0:
        sp = min(math.dist(s.pos["upperarm_l"], s.pos["upperarm_r"]), run)
        d = q["dir"]
        lo_, hi_ = sp / 2, run - sp / 2                       # where the middle of the two palms may be, along a -> b
        mid = max(lo_, min(hi_, dot(sub(near, q["a"]), d)))
        p1, p2 = add(q["a"], scale(d, mid - sp / 2)), add(q["a"], scale(d, mid + sp / 2))
        lp, rp = (p1, p2) if dot(sub(p1, p2), left) >= 0 else (p2, p1)
        return {"l": hand("l", lp, q), "r": hand("r", rp, q)}
    first = hs if hs in free else free[0]
    out[first] = hand(first, near, q)
    for x in free:
        if x == first:
            continue
        for _, n2, q2 in ok[1:]:
            if math.dist(n2, near) >= FIX_PALM_CM + FIX_INNER_CM:
                out[x] = hand(x, n2, q2)
                break
    return out


def fixture_fix(prev, palm, a, b):
    """The next palm offset of a re-solve on a piece's part: the last one moved by the measured miss across the part
    (from the palm to the part's nearest point; a palm slid along the part is still on it)."""
    m = sub(seg_point(palm, a, b), palm)
    p = prev or [0.0, 0.0, 0.0]
    return [round(float(p[i]) + m[i], 3) for i in range(3)]


def fixture_contact(palm, a, b, name=None, side=None):
    """A hand on a piece's part (one frame, cm): "palm_cm" the palm point's distance from the part's line between
    a and b; "why" the plain words when it is over HELD_LIMITS["grip_cm"]."""
    out = {"palm_cm": round(seg_dist(palm, a, b), 2)}
    out["why"] = fixture_why(out, name, side)
    return out


def fixture_why(c, name=None, side=None):
    nm = _the(name, "the piece they work")
    hand = {"l": "the left hand", "r": "the right hand"}.get(side, "a hand")
    if (c.get("palm_cm") or 0.0) > HELD_LIMITS["grip_cm"]:
        return [f"{hand} is not on {nm}: its palm is {c['palm_cm']:.0f} cm from the part it holds"]
    return []


def _restore(s, state):
    s.pos, s.rot = dict(state[0]), dict(state[1])


def _yaw_foot(s, side, yaw_deg):
    """The foot flat as in the reference pose, turned yaw_deg about the vertical (positive: toward the right)."""
    f = f"foot_{side}"
    want = qmul(q_axis([0, 0, 1], yaw_deg), s.ref_rot[f])
    _rot_q(s, f, qnorm(qmul(want, qinv(s.rot[f]))))


def _legs(s, ankles, yaws, knee_reach=KNEE_REACH):
    """Both feet planted at their ankle targets (actor frame, on the floor), the hips lowered until no leg is
    straighter than knee_reach of its length, the knees bending forward over the toes. -> extra drop (cm)."""
    drop = 0.0
    for x in ("l", "r"):
        L = s.length(f"thigh_{x}", f"calf_{x}") + s.length(f"calf_{x}", f"foot_{x}")
        hp, a = s.pos[f"thigh_{x}"], ankles[x]
        hz = math.hypot(hp[0] - a[0], hp[1] - a[1])
        need = (knee_reach * L) ** 2 - hz * hz
        if need > 0:
            drop = max(drop, (hp[2] - a[2]) - math.sqrt(need))
    if drop > 0:
        s.move("pelvis", [0.0, 0.0, -drop])
    for x in ("l", "r"):
        sx = 1.0 if x == "l" else -1.0
        r = sub(s.ref_pos[f"ball_{x}"], s.ref_pos[f"foot_{x}"])
        toe = unit(qrot(q_axis([0, 0, 1], yaws[x]), [r[0], r[1], 0.0]))
        hip = s.pos[f"thigh_{x}"]
        mid = scale(add(hip, ankles[x]), 0.5)
        pole = add(mid, add(scale(toe, 50.0), [sx * 8.0, 0.0, 0.0]))
        _ik(s, f"thigh_{x}", f"calf_{x}", f"foot_{x}", ankles[x], pole)
        _yaw_foot(s, x, yaws[x])
    return drop


def _clear_torso(s, tgt, r):
    """A hand target inside the trunk (a capsule from the pelvis to the upper chest, radius r) pushed out of it."""
    a, b = s.pos["pelvis"], s.pos["spine_05"]
    ab = sub(b, a)
    t = max(0.0, min(1.0, dot(sub(tgt, a), ab) / max(dot(ab, ab), 1e-6)))
    c = add(a, scale(ab, t))
    d = sub(tgt, c)
    if norm(d) >= r:
        return tgt
    d = d if norm(d) > 1e-6 else [0.0, 1.0, 0.0]
    return add(c, scale(unit(d), r))


def _arm_to(s, side, tgt, pole_dir, H):
    """The hand (wrist joint) to tgt, the elbow toward pole_dir; never straight: a target past REACH of the arm is
    taken at REACH along the same line (the caller brings the body nearer). -> (reach share asked for, target
    wanted, clear of the trunk)."""
    sh = s.pos[f"upperarm_{side}"]
    L = s.length(f"upperarm_{side}", f"lowerarm_{side}") + s.length(f"lowerarm_{side}", f"hand_{side}")
    tgt = _clear_torso(s, tgt, 0.075 * H)
    d = sub(tgt, sh)
    n = norm(d)
    use = add(sh, scale(d, REACH * L / n)) if n > REACH * L else tgt
    pole = add(scale(add(sh, use), 0.5), scale(unit(pole_dir), L))
    _ik(s, f"upperarm_{side}", f"lowerarm_{side}", f"hand_{side}", use, pole)
    return n / L, tgt


def _hand_orient(s, side, h, k):
    """The hand's wrist -> middle knuckle along h and its knuckle line (pinky -> index) along k (perpendicular)."""
    s.aim(f"hand_{side}", f"middle_01_{side}", h)
    ix, pk = f"index_01_{side}", f"pinky_01_{side}"
    if ix not in s.pos or pk not in s.pos:
        return
    h = unit(h)
    a, b = perp(sub(s.pos[ix], s.pos[pk]), h), perp(k, h)
    if norm(a) < 1e-6 or norm(b) < 1e-6:
        return
    _rot(s, f"hand_{side}", h, math.degrees(math.atan2(dot(h, cross(a, b)), dot(a, b))))


def _wrist_limit(f, t, max_deg=WRIST_MAX):
    """A handle direction the wrist can give: the hand bends off the forearm by 90 deg less the forearm-to-handle
    angle, so that angle is kept within [90 - max, 90 + max]."""
    c = dot(f, t)
    lim = math.cos(math.radians(90.0 - max_deg))
    if abs(c) <= lim:
        return t
    w = perp(t, f)
    w = unit(w) if norm(w) > 1e-6 else unit(perp([0.0, 0.0, 1.0] if abs(f[2]) < 0.9 else [1.0, 0.0, 0.0], f))
    return unit(add(scale(f, math.copysign(lim, c)), scale(w, math.sqrt(1 - lim * lim))))


def palm_point(s, side):
    """Where a handle lies in the hand: 60 % of the way from the wrist to the middle knuckle, off the palm by 28 % of
    that length (palm flesh and the handle's radius)."""
    return palm_from_joints(s.pos[f"hand_{side}"], s.pos[f"middle_01_{side}"], s.pos[f"index_01_{side}"],
                            s.pos[f"pinky_01_{side}"], side)


def palm_from_joints(hd, m, ix, pk, side):
    """palm_point from the wrist, middle, index and pinky knuckle joints (any one frame, e.g. measured UE sockets)."""
    sx = 1.0 if side == "l" else -1.0
    h = sub(m, hd)
    ln = norm(h)
    k = unit(perp(sub(ix, pk), unit(h)))
    n = scale(cross(unit(h), k), -sx)               # the palm's side (right hand: h x k; left: k x h)
    return add(hd, add(scale(h, 0.6), scale(n, 0.28 * ln)))


def _grip(s, side, t_want, face_want=None):
    """The hand closed round a handle running along t_want (grip -> head, beyond the thumb): the hand's own line is
    the forearm's with the handle's part taken out, the handle bent off the forearm only as far as the wrist goes.
    With face_want (where a head's face must point, 11-18: a blow's face flat on the work), the hand instead turns
    about the handle (flexing or extending the wrist) toward it, as far as WRIST_MAX lets the hand leave the forearm.
    -> (handle direction given, hand direction)."""
    f = unit(sub(s.pos[f"hand_{side}"], s.pos[f"lowerarm_{side}"]))
    t = _wrist_limit(f, unit(t_want))
    h = unit(perp(f, t))
    if face_want is not None and norm(perp(face_want, t)) > 1e-6:
        e = unit(cross(t, h))                        # h turned a quarter about the handle
        w = unit(perp(face_want, t))
        th = math.atan2(dot(w, e), dot(w, h))
        b = norm(perp(f, t))                         # cos(hand-to-forearm angle) = b cos(th)
        lim = math.acos(min(1.0, math.cos(math.radians(WRIST_MAX - 0.5)) / max(b, 1e-6)))
        th = max(-lim, min(lim, th))
        h = unit(add(scale(h, math.cos(th)), scale(e, math.sin(th))))
    _hand_orient(s, side, h, t)
    return t, h


def _flat_hand(s, side, fwd, out_side):
    """A hand resting on a top: fingers forward and a little down, palm down, the wrist within WRIST_MAX."""
    sx = 1.0 if side == "l" else -1.0
    f = unit(sub(s.pos[f"hand_{side}"], s.pos[f"lowerarm_{side}"]))
    h = unit(add(scale(fwd, 0.85), add(scale(out_side, -0.15), [0.0, 0.0, -0.45])))
    a = angle_deg(f, h)
    if a > WRIST_MAX - 5:                      # bent back toward the forearm's line
        w = unit(perp(h, f))
        th = math.radians(WRIST_MAX - 5)
        h = unit(add(scale(f, math.cos(th)), scale(w, math.sin(th))))
    k = scale(cross([0.0, 0.0, -1.0], h), -sx)    # knuckle line for a palm facing down
    _hand_orient(s, side, h, k)


def _look_at(s, pt, H, max_deg=75.0):
    """Neck and head turned so the face looks at pt (the neck takes 40 %); the reference face looks along +y."""
    for b, f in (("neck_01", 0.4), ("head", 1.0)):
        dq = qmul(s.rot["head"], qinv(s.ref_rot["head"]))
        fw = qrot(dq, [0.0, 1.0, 0.0])
        eye = add(s.pos["head"], qrot(dq, [0.0, 0.05 * H, 0.03 * H]))
        d = unit(sub(pt, eye))
        ax = cross(fw, d)
        if norm(ax) < 1e-6:
            return
        _rot(s, b, ax, min(angle_deg(fw, d), max_deg) * f)


def face_forward(s):
    return qrot(qmul(s.rot["head"], qinv(s.ref_rot["head"])), [0.0, 1.0, 0.0])


def _body_axes(s):
    """Horizontal left and forward of the shoulders (unit, actor frame)."""
    l_ = sub(s.pos["upperarm_l"], s.pos["upperarm_r"])
    l_ = unit([l_[0], l_[1], 0.0])
    return l_, [-l_[1], l_[0], 0.0]


def _clav_raise(s, side, tgt_z):
    sh, cl = s.pos[f"upperarm_{side}"], f"clavicle_{side}"
    if tgt_z <= sh[2] or cl not in s.pos:
        return
    v = sub(sh, s.pos[cl])
    ax = cross(v, [0.0, 0.0, 1.0])
    if norm(ax) > 1e-6:
        _rot(s, cl, ax, min(CLAV_MAX, 0.45 * (tgt_z - sh[2])))


def _fingers_rest(s, side):
    """The fingers back to their reference shape against the hand (so a curl is the curl asked for, not added to the
    posture's own)."""
    hb = f"hand_{side}"
    q = qnorm(qmul(s.rot[hb], qinv(s.ref_rot[hb])))
    for b in s.subtree(hb):
        if b != hb:
            s.pos[b] = add(s.pos[hb], qrot(q, sub(s.ref_pos[b], s.ref_pos[hb])))
            s.rot[b] = qnorm(qmul(q, s.ref_rot[b]))
            s.changed.add(b)


def _aim_item(s, side, g, P, n, pole, H, tilt_deg, face_down, fwd, passes=6):
    """The held thing's working point (work_point_m: a head's face centre, else its tip) onto P (actor frame), a point
    on a surface with normal n: the handle in the swing plane (vertical through the shoulder and P), tilt_deg below
    the surface's level (0 for a face: flat on the surface with the handle across it, as blockout_contact.face_strike
    does), the face toward -n. Each pass solves the arm for the handle and face the wrist really gives (not the ones
    wanted: 11-18, the hammer's face landed a hand's breadth short and turned sideways), so the point ends on P.
    -> (reach share, wrist target, handle dir, face dir, working point, handle dir wanted)."""
    nn = unit(n)
    a = math.radians(tilt_deg)
    face = scale(nn, -1.0) if face_down else None
    po, t, h = [0.0, 0.0, 0.0], None, None
    reach = tgt = t_want = None
    for _ in range(passes):
        # (a two-handed thing swings in the plane through the middle of the chest, both hands on its handle)
        sh = s.pos[f"upperarm_{side}"] if not g.get("two_hands") else \
            scale(add(s.pos["upperarm_l"], s.pos["upperarm_r"]), 0.5)
        dh = perp(sub(P, sh), nn)
        dh = unit(dh) if norm(dh) > 1e-6 else unit(perp(fwd, nn))
        if t is not None:
            # a long handle whose level grip would come back into the body is raised at the hands as far as needed
            # (the face then meets the work at the handle's tilt, and the check says so)
            lw = dot(work_offset(g, t, h), t)
            room = dot(sub(P, s.pos["pelvis"]), dh) - HANDS_AHEAD * H
            if lw > 1e-6 and lw * math.cos(a) > room:
                a = max(a, math.acos(max(0.0, min(1.0, room / lw))))
        t_want = unit(sub(scale(dh, math.cos(a)), scale(nn, math.sin(a))))
        if t is None:
            t = t_want
            h = face if face is not None else unit(perp([0.0, 0.0, -1.0], t) if abs(t[2]) < 0.9 else perp(fwd, t))
        reach, tgt = _arm_to(s, side, sub(sub(P, work_offset(g, t, h)), po), pole, H)
        t, h = _grip(s, side, t_want, face)
        po = sub(palm_point(s, side), s.pos[f"hand_{side}"])
    return reach, tgt, t, h, add(palm_point(s, side), work_offset(g, t, h)), t_want


def _work_solve(s, c, cfg):
    """One pass of the work stance from the base state with this lean, hip drop and hip shift."""
    H, hs, ot, kind, spec = c["H"], c["hand"], c["other"], c["kind"], c["spec"]
    sx = 1.0 if hs == "l" else -1.0
    turn = -sx                                           # positive about z turns toward the right
    lean = cfg["lean"]
    if c["standing"]:
        A = {x: list(s.ref_pos[f"foot_{x}"]) for x in ("l", "r")}
        cx, cy = (A["l"][0] + A["r"][0]) / 2, (A["l"][1] + A["r"][1]) / 2
        half = max(abs(A["l"][0] - cx), 0.5 * abs(s.ref_pos["upperarm_l"][0] - s.ref_pos["upperarm_r"][0]))
        ank, yaw = {}, {}
        for x, fy, toe in ((ot, STANCE_FRONT, TOE_OUT_FRONT), (hs, -STANCE_BACK, TOE_OUT_BACK)):
            xs = 1.0 if x == "l" else -1.0
            ank[x] = [cx + xs * half, cy + fy * H, A[x][2]]
            yaw[x] = -xs * toe                                 # toes out: the left foot turns left
        trunk = s.ref_pos["neck_01"][2] - s.ref_pos["pelvis"][2]
        hinge = 0.2 * trunk * math.sin(math.radians(lean))     # the hips go back as the trunk folds forward
        pel = s.pos["pelvis"]
        s.move("pelvis", [(ank["l"][0] + ank["r"][0]) / 2 + cfg["shift"][0] - pel[0],
                          (ank["l"][1] + ank["r"][1]) / 2 - hinge + cfg["shift"][1] - pel[1], -cfg["drop"]])
        _rot(s, "pelvis", [0, 0, 1], turn * spec["pelvis"])
        # most of the fold at the hips, the rest spread up the spine (a hinge, not a hunch)
        shares = (("pelvis", 0.5), ("spine_01", 0.125), ("spine_02", 0.125), ("spine_03", 0.125), ("spine_04", 0.125))
    else:
        shares = (("spine_01", 0.3), ("spine_02", 0.25), ("spine_03", 0.25), ("spine_04", 0.2))
    for b, f in shares:
        _rot(s, b, [1, 0, 0], -lean * f)
    for b, nx, f in (("spine_02", "spine_03", 0.3), ("spine_03", "spine_04", 0.35), ("spine_04", "spine_05", 0.35)):
        ax = sub(s.pos[nx], s.pos[b])
        if norm(ax) > 1e-6:
            _rot(s, b, ax, turn * spec["chest"] * f)
    drop = cfg["drop"]
    if c["standing"]:
        drop += _legs(s, ank, yaw)
    left, fwd = _body_axes(s)
    up = [0.0, 0.0, 1.0]
    out = scale(left, sx)                                    # toward the tool side
    oo = scale(out, -1.0)
    top, wy, wx = c["top"], c["wy"], c["wx"]
    t_want = unit(add(scale(fwd, spec["tool"][0]), add(scale(up, spec["tool"][1]), scale(out, spec["tool"][2]))))
    g = c["grip"]
    # the blow's point: the worked thing's top when it is known (scene.py: where it lies, plus its thickness as it
    # lies), else the work top straight ahead (11-18: a point fwd_cm ahead at the top's height lay 13 cm short of
    # the bar and inside it)
    strike = list(c["target"]) if c.get("target") else [wx, wy, top]
    nrm = c.get("normal") or [0.0, 0.0, 1.0]
    near = c["near"]                                         # the work top's near edge (y), when known
    # a hand rests on the work's top: toward its near edge when the body, folded as far as it goes, falls short
    edge = lambda y: y + cfg["pull"] * (near + 0.02 * H - y) if y > near + 0.02 * H else y      # noqa: E731
    rest_off = [wx + (1.0 if ot == "l" else -1.0) * 0.06 * H, edge(min(wy - 0.03 * H, near + 0.035 * H)), top + 0.025 * H]
    targets, reach, work_hands = {}, {}, []
    pole_top = lambda o: add(scale(o, 0.6), add(scale(fwd, -0.25), [0.0, 0.0, -0.75]))       # noqa: E731
    pole_up = add(scale(out, 0.85), add(scale(fwd, 0.25), [0.0, 0.0, -0.45]))
    wpt, rpt, t, h = None, None, t_want, None
    fix = c.get("fix") or {}
    # the tool hand: where its palm must be (on a piece's part when it holds nothing and the piece has one)
    if hs in fix:
        reach[hs], targets[hs], t, h = _fix_hand(s, hs, fix[hs], pole_top(out), H)
        work_hands.append(hs)
    elif kind == "striking_down":
        # the working point on the blow's point: a head's face flat on it (the handle level, across the face's
        # normal), a tipped thing's tip coming down onto it; the elbow out to the side so the forearm slopes down
        # into the blow and the hand can turn the face down
        headed = g.get("work") == "head"
        tilt = 0.0 if headed else STRIKE_TIP_DEG
        sh = s.pos[f"upperarm_{hs}"]
        best = None
        for pl in STRIKE_POLES:            # the elbow's side that turns the face flattest, never up off the arm's line
            pole = add(scale(out, pl[0]), add(scale(fwd, pl[1]), [0.0, 0.0, pl[2]]))
            got = _aim_item(s, hs, g, strike, nrm, pole, H, tilt, headed, fwd)
            el, wr = s.pos[f"lowerarm_{hs}"], s.pos[f"hand_{hs}"]
            up_off = perp(sub(el, sh), unit(sub(wr, sh)))[2]
            score = (up_off > 0.8, round(angle_deg(got[3], scale(nrm, -1.0)), 1) if headed else 0.0,
                     round(math.dist(got[4], strike), 1))
            if best is None or score < best[0]:
                best = (score, pole)
        reach[hs], targets[hs], t, h, wpt, t_want = _aim_item(s, hs, g, strike, nrm, best[1], H, tilt, headed, fwd)
        work_hands.append(hs)
    elif kind in READY_ACTIONS:
        reach[hs], targets[hs], t, h, t_want = _ready_hand(s, c, out, fwd)
        work_hands.append(hs)
    else:
        if kind == "tool_raised":
            head_top = s.pos["head"][2] + 0.1 * H
            _clav_raise(s, hs, head_top + 0.03 * H)
            sh = s.pos[f"upperarm_{hs}"]
            # the hand beside the head, the elbow bent near a right angle below it (a hand high over the head on a
            # straight arm read as a wave, not a wind-up)
            palm_want = add(sh, add(scale(out, 0.09 * H), scale(fwd, -0.02 * H)))
            palm_want[2] = s.pos["head"][2] + 0.03 * H
            tool_pole = pole_up
        else:
            palm_want = [wx + sx * 0.05 * H, edge(min(wy - 0.05 * H, near + 0.06 * H)), top + 0.03 * H]
            tool_pole = pole_top(out)
            work_hands.append(hs)
        po = [0.0, 0.0, 0.0]                                   # palm - wrist, refined over the passes
        for _ in range(3):
            reach[hs], targets[hs] = _arm_to(s, hs, sub(palm_want, po), tool_pole, H)
            t, h = _grip(s, hs, t_want)
            po = sub(palm_point(s, hs), s.pos[f"hand_{hs}"])
        if kind == "tool_raised" and c.get("target"):
            # wound up over the worked thing: where its working point is held, for the staged check (11-18: a raised
            # blow reported no target, so nothing measured it against the work)
            rpt = add(palm_point(s, hs), work_offset(g, t, h))
    # the other hand: on the handle of a two-handed thing, else holding its own thing with its working end on the
    # work beside the blow, else flat on the work's top
    off = None
    if g["two_hands"]:
        off_palm = sub(palm_point(s, hs), scale(t, 100.0 * (g["grip_from_end_m"] - g["off_from_end_m"])))
        po2 = [0.0, 0.0, 0.0]
        for _ in range(3):
            reach[ot], targets[ot] = _arm_to(s, ot, sub(off_palm, po2), pole_top(oo), H)
            _grip(s, ot, t, scale(nrm, -1.0) if kind == "striking_down" and g.get("work") == "head" else None)
            po2 = sub(palm_point(s, ot), s.pos[f"hand_{ot}"])
        if kind in READY_ACTIONS:                           # (a guard's both hands are its reach)
            work_hands.append(ot)
    elif kind in READY_ACTIONS:
        # the free hand forward and out for balance, a little over the weapon hand (what it holds, if anything, is
        # gripped there afterwards: _hold_all)
        F, _ = _ready_axes(s, c)
        sh2 = s.pos[f"upperarm_{ot}"]
        bal = add(sh2, add(scale(F, 0.17 * H), scale([F[1], -F[0], 0.0], -sx * 0.05 * H)))
        bal[2] = s.pos["pelvis"][2] + 0.7 * (s.pos["spine_05"][2] - s.pos["pelvis"][2])
        reach[ot], targets[ot] = _arm_to(s, ot, bal, pole_top(oo), H)
        _flat_hand(s, ot, F, oo)
        work_hands.append(ot)
    elif c.get("off_item"):
        og = c["off_item"]["grip"]
        oh = og.get("work") == "head"
        p2 = add(strike, scale(oo, 0.06 * H))
        _clav_raise(s, ot, p2[2])
        reach[ot], targets[ot], t2, h2, w2, _ = _aim_item(s, ot, og, p2, nrm, pole_top(oo), H,
                                                          0.0 if oh else STRIKE_TIP_DEG, oh, fwd)
        off = {"dir": t2, "face": h2, "work": w2, "aim": p2}
        work_hands.append(ot)
    elif ot in fix:
        reach[ot], targets[ot], _, _ = _fix_hand(s, ot, fix[ot], pole_top(oo), H)
        work_hands.append(ot)
    else:
        _clav_raise(s, ot, rest_off[2])
        reach[ot], targets[ot] = _arm_to(s, ot, rest_off, pole_top(oo), H)
        _flat_hand(s, ot, fwd, oo)
        work_hands.append(ot)
    look = strike if kind != "hands_at_work" else [wx, wy - 0.04 * H, top]
    if fix and kind == "hands_at_work":                     # on the part the hands hold
        look = [sum(f["aim"][i] for f in fix.values()) / len(fix) for i in range(3)]
    if kind in READY_ACTIONS:                               # on the foe, else straight ahead at eye height
        look = _ready_axes(s, c)[1]
    _look_at(s, look, H)
    m, com, poly = balance_margin(s.pos) if c["standing"] else (None, None, None)
    return {"reach": max([reach[x] for x in set(work_hands)] or [0.0]), "reach_by_hand": reach, "targets": targets,
            "margin": m, "com": com, "poly": poly, "tool_dir": t, "tool_want": t_want, "face_dir": h,
            "drop": drop, "look": look, "strike": strike, "normal": nrm, "work_pt": wpt, "raised_pt": rpt, "off": off}


def _ready_axes(s, c):
    """A guard's line (READY_ACTIONS): the level way from the hips to the target (else the actor's own forward) and
    the point the eyes and the weapon are on (the target, else a point straight ahead at eye height)."""
    pel, tg = s.pos["pelvis"], c.get("target")
    d = [tg[0] - pel[0], tg[1] - pel[1], 0.0] if tg else [0.0, 1.0, 0.0]
    F = unit(d) if norm(d) > 1e-6 else [0.0, 1.0, 0.0]
    if tg:
        return F, list(tg)
    return F, add(s.pos["head"], add(scale(F, 3.0 * c["H"]), [0.0, 0.0, 0.03 * c["H"]]))


def _ready_hand(s, c, out, fwd):
    """The weapon hand of a guard (READY_ACTIONS): the palm READY_AHEAD of the hips toward the target, at READY_HEIGHT
    of the way up to the upper chest, a little out on its own side; the handle along the line from the palm to the
    target (else straight ahead, READY_UP_DEG up), as far as the wrist gives it; the elbow on the side of READY_POLES
    (down and out, never up) that lays the handle nearest that line. A two-handed thing's grips lie about the ready
    point (the other hand behind on the handle, clear of the body). -> (reach share, wrist target, handle dir, hand
    dir, handle dir wanted)."""
    H, hs, g = c["H"], c["hand"], c["grip"]
    sx = 1.0 if hs == "l" else -1.0
    F, aim = _ready_axes(s, c)
    left = [F[1], -F[0], 0.0]
    pel, chest = s.pos["pelvis"], s.pos["spine_05"]
    ready = add(pel, add(scale(F, READY_AHEAD * H), scale(left, sx * READY_SIDE[1 if g["two_hands"] else 0] * H)))
    ready[2] = pel[2] + READY_HEIGHT * (chest[2] - pel[2])
    a = math.radians(READY_UP_DEG)
    ahead = unit(add(scale(F, math.cos(a)), [0.0, 0.0, math.sin(a)]))

    def want(p):
        return unit(sub(aim, p)) if c.get("target") else ahead
    sep = 100.0 * (g["grip_from_end_m"] - g["off_from_end_m"]) if g["two_hands"] else 0.0
    sh = s.pos[f"upperarm_{hs}"]
    best, got = None, None
    for pl in READY_POLES + (None,):
        pole = best[1] if pl is None else add(scale(out, pl[0]), add(scale(fwd, pl[1]), [0.0, 0.0, pl[2]]))
        po, tw = [0.0, 0.0, 0.0], want(ready)
        for _ in range(3):
            palm = add(ready, scale(tw, 0.3 * sep))
            got = _arm_to(s, hs, sub(palm, po), pole, H)
            tw = want(palm)
            t, h = _grip(s, hs, tw)
            po = sub(palm_point(s, hs), s.pos[f"hand_{hs}"])
        if pl is None:
            break
        el, wr = s.pos[f"lowerarm_{hs}"], s.pos[f"hand_{hs}"]
        up_off = perp(sub(el, sh), unit(sub(wr, sh)))[2]
        score = (up_off > 1.0, round(angle_deg(t, tw), 1), round(got[0], 3))
        if best is None or score < best[0]:
            best = (score, pole)
    return got[0], got[1], t, h, tw


def _fix_hand(s, side, f, pole, H):
    """A free hand closed round a piece's part (fixture_hands' choice f): the palm on f["aim"] (the part's point, plus
    any engine correction), the handle direction along the part. -> (reach share, wrist target, handle dir, hand dir)."""
    _clav_raise(s, side, f["aim"][2])
    po = [0.0, 0.0, 0.0]
    reach = tgt = t = h = None
    for _ in range(3):
        reach, tgt = _arm_to(s, side, sub(f["aim"], po), pole, H)
        t, h = _grip(s, side, f["t"])
        po = sub(palm_point(s, side), s.pos[f"hand_{side}"])
    return reach, tgt, t, h


def _held_items(p):
    """The things in this person's hands: params "held" (scene.py: each {"asset", "hand", "size_m", "grip" measured
    from its model by scene_job, "name": the plan's}), else the older single work tool (tool_size_m / tool_grip /
    tool_asset) in the work hand. One thing per hand; a thing whose hand is taken goes to the free one."""
    raw = [dict(it, grip=full_grip(it.get("grip"), it.get("size_m"))) for it in (p.get("held") or []) if it]
    if not raw and (p.get("tool_size_m") or p.get("tool_grip")):
        raw = [{"asset": p.get("tool_asset") or "tool", "hand": p.get("hand", "r"), "size_m": p.get("tool_size_m"),
                "grip": full_grip(p.get("tool_grip"), p.get("tool_size_m"))}]
    used, out = set(), []
    for it in raw:
        h = it.get("hand") if it.get("hand") in ("l", "r") and it.get("hand") not in used else \
            next((x for x in ("r", "l") if x not in used), None)
        if h is not None:
            used.add(h)
            out.append(dict(it, hand=h))
    return out


def _held_report(s, side, item, t, h, how, **extra):
    """s.info["held"][asset]: the hand, palm and wrist points, the handle's direction (grip -> working end) and where
    the mesh's face axis points (the working face's direction with the model's face_sign applied), for prop_in_hand."""
    g = item["grip"]
    r2 = lambda v: [round(q, 2) for q in v]                   # noqa: E731
    rep = {"hand": side, "how": how, "palm_cm": r2(palm_point(s, side)), "hand_cm": r2(s.pos[f"hand_{side}"]),
           "dir": [round(q, 4) for q in t], "face": [round(q, 4) for q in scale(h, g.get("face_sign", 1.0))],
           "grip": g, "work": g.get("work"), "name": item.get("name"), "curl_deg": round(grip_curl(g), 1)}
    rep.update(extra)
    s.info.setdefault("held", {})[item.get("asset") or f"held_{side}"] = rep
    return rep


def _work_pose(s, p, kind):
    """The whole-body work stance for a work action (WORK_ACTIONS), solved until every work-top hand is in reach
    with a bent elbow and the centre of mass is over the feet; the result is reported in s.info["work"], the held
    tool in s.info["tool"] and every held thing in s.info["held"]."""
    hs = p.get("hand", "r")
    ot = "l" if hs == "r" else "r"
    standing = p.get("_posture") not in SEATED
    base = (dict(s.ref_pos), dict(s.ref_rot)) if standing else (dict(s.pos), dict(s.rot))
    floor = min(s.ref_pos["foot_l"][2], s.ref_pos["foot_r"][2]) - p.get("ankle_cm", 8.0)
    H = (s.ref_pos["head"][2] - floor) / 0.9                  # the head joint at ~0.9 of the stature
    hip = s.ref_pos["pelvis"][2] - floor
    # (a guard aims at its foe, params "aim_target_ue"; a worked thing lying before it is not its target)
    tgt_ue = p.get("aim_target_ue") if kind in READY_ACTIONS else p.get("work_target_ue")
    target = s.local_pt(tgt_ue) if tgt_ue else None
    if kind in READY_ACTIONS:
        p = {k: v for k, v in p.items() if k not in ("work_top_cm", "fixture")}       # (no work top, no piece)
    top = floor + float(p["work_top_cm"]) if "work_top_cm" in p else (target[2] if target else floor + 0.52 * H)
    # a blow at a worked thing works at its top (where the blow lands and the other hand steadies it), not at the top
    # of the piece it lies on (11-18: under a thick worked thing the other hand reached down to the piece's top, the
    # body folded deep for it and the face came down 16-34 deg off the work on the test figure; 0-5 deg now)
    if target and kind in BLOW_ACTIONS:
        top = max(top, target[2])
    low = max(0.0, hip - (top - floor))                       # how far under the hips the work's top lies
    spec = WORK_SPEC[kind]
    # the work hand takes the thing it works with: to strike or raise, a thing with a head (measured from its model)
    # before one without; the other hand the other thing (a smith's tongs beside his hammer, 11-18: both had been put
    # in one palm)
    items = _held_items(p)
    tool = next((it for it in items if it["hand"] == hs), None)
    if kind in BLOW_ACTIONS and (tool is None or tool["grip"].get("work") != "head"):
        tool = next((it for it in items if it["grip"].get("work") == "head"), tool)
    tool = tool or (items[0] if items else None)
    off_item = next((it for it in items if it is not tool), None)
    grip = tool["grip"] if tool else full_grip(p.get("tool_grip"), p.get("tool_size_m"))
    if grip["two_hands"] and off_item:                        # the other hand holds its own thing
        grip = dict(grip, two_hands=False)
    # a piece of the set worked with the hands (params "fixture": its name and its surface round the person, UE cm,
    # from scene_job): the hands that hold nothing take hold of a part of it (fixture_hands)
    busy = ({hs} if tool else set()) | ({ot} if off_item else set()) | ({"l", "r"} if tool and grip["two_hands"] else set())
    fix, fx_rep = _fixture_choice(s, p, {"l", "r"} - busy, hs, H, floor)
    c = {"H": H, "hand": hs, "other": ot, "kind": kind, "spec": spec, "standing": standing, "top": top,
         "wy": target[1] if target else s.ref_pos["pelvis"][1] + float(p.get("work_fwd_cm", 0.27 * H)),
         "wx": target[0] if target else s.ref_pos["pelvis"][0],
         "near": s.ref_pos["pelvis"][1] + float(p.get("work_near_cm", p.get("work_fwd_cm", 0.27 * H))),
         "grip": grip, "target": target, "normal": [0.0, 0.0, 1.0], "off_item": off_item, "fix": fix}
    cfg = {"lean": float(p.get("lean_deg", spec["lean"] + spec["lean_cm"] * low)),
           "drop": min(0.08 * H, 0.03 * H + 0.1 * low) if standing else 0.0, "shift": [0.0, 0.0], "pull": 0.0}
    n = 0
    for n in range(1, 25):
        _restore(s, base)
        res = _work_solve(s, c, cfg)
        if res["reach"] > REACH and n < 24 and (cfg["lean"] < LEAN_MAX or cfg["pull"] < 1.0):
            if cfg["lean"] < LEAN_MAX:
                cfg["lean"] = min(LEAN_MAX, cfg["lean"] + 4.0)
                if standing:
                    cfg["drop"] = min(DROP_MAX * H, cfg["drop"] + 0.008 * H)
            else:
                cfg["pull"] = min(1.0, cfg["pull"] + 0.25)
            continue
        if standing and res["margin"] < BALANCE_MARGIN_CM and n < 24:
            cen = [sum(q[0] for q in res["poly"]) / len(res["poly"]), sum(q[1] for q in res["poly"]) / len(res["poly"])]
            cfg["shift"] = [cfg["shift"][i] + 0.6 * (cen[i] - res["com"][i]) for i in range(2)]
            continue
        break
    # fingers closed round each handle by its thickness (11-18: a fixed 70 deg whatever the handle); a free hand
    # on the work's top lies relaxed
    og = grip if grip["two_hands"] else (off_item["grip"] if off_item else None)
    for x, gg in ((hs, grip), (ot, og)):
        _fingers_rest(s, x)
        if x in fix:                                       # round the piece's part, by its thickness
            _curl(s, x, grip_curl({"handle_m": fix[x]["thick_cm"] / 100.0}), 35)
        elif gg:
            _curl(s, x, grip_curl(gg), 35)
        else:
            _curl(s, x, 30, 12)
    r2 = lambda v: [round(q, 2) for q in v]                   # noqa: E731
    s.info["work"] = {"action": kind, "standing": standing, "top_cm": round(top, 1), "lean_deg": round(cfg["lean"], 1),
                      "hip_drop_cm": round(res["drop"], 1), "hip_shift_cm": r2(cfg["shift"]), "iterations": n,
                      "reach": {k: round(v, 3) for k, v in res["reach_by_hand"].items()},
                      "targets": {f"hand_{k}": r2(v) for k, v in res["targets"].items()}, "look_at": r2(res["look"]),
                      "balance_margin_cm": None if res["margin"] is None else round(res["margin"], 1),
                      "com_xy": None if res["com"] is None else r2(res["com"])}
    tool_rep = {"hand": hs, "palm_cm": r2(palm_point(s, hs)), "hand_cm": r2(s.pos[f"hand_{hs}"]),
                "dir": [round(q, 4) for q in res["tool_dir"]], "want_dir": [round(q, 4) for q in res["tool_want"]],
                "face": [round(q, 4) for q in scale(res["face_dir"], grip.get("face_sign", 1.0))], "grip": grip,
                "work": grip.get("work"), "curl_deg": round(grip_curl(grip), 1)}
    if res["work_pt"] is not None:                         # the blow: where the working point landed
        tool_rep.update(target_cm=r2(res["strike"]), normal=res["normal"], work_cm=r2(res["work_pt"]),
                        gap_cm=round(math.dist(res["work_pt"], res["strike"]), 2))
        if grip.get("work") == "head":
            tool_rep["face_angle_deg"] = round(angle_deg(res["face_dir"], scale(res["normal"], -1.0)), 1)
    elif res.get("raised_pt") is not None:                 # wound up: the target it will come down on, and how high
        tool_rep.update(target_cm=r2(res["strike"]), normal=res["normal"], work_cm=r2(res["raised_pt"]), raised=True,
                        rise_cm=round(dot(sub(res["raised_pt"], res["strike"]), unit(res["normal"])), 2))
    if fix:
        for x, f in fix.items():
            fc = fixture_contact(palm_point(s, x), f["a"], f["b"], fx_rep.get("name"), x)
            fx_rep["hands"][x] = {"on_cm": r2(f["on"]), "aim_cm": r2(f["aim"]), "a_cm": r2(f["a"]), "b_cm": r2(f["b"]),
                                  "dir": f["dir"], "thick_cm": f["thick_cm"], "palm_cm": r2(palm_point(s, x)),
                                  "hand_cm": r2(s.pos[f"hand_{x}"]), "solver_palm_cm": fc["palm_cm"]}
    if fx_rep:
        s.info["fixture"] = fx_rep
    if tool:
        tool_rep.update(asset=tool.get("asset"), name=tool.get("name"))
        s.info.setdefault("held", {})[tool.get("asset") or f"held_{hs}"] = dict(tool_rep, how="work")
    s.info["tool"] = tool_rep
    if off_item and res["off"]:
        o = res["off"]
        _held_report(s, ot, off_item, o["dir"], o["face"], "work_off", aim_cm=r2(o["aim"]), work_cm=r2(o["work"]))


def _fixture_choice(s, p, free, hs, H, floor):
    """The free hands' parts of the worked piece (params "fixture": {"name", "pts_ue": its surface round the person,
    UE cm, "fix_ue": {side: a palm offset, UE cm, from the engine's re-solve}}). -> (fixture_hands' choice with each
    hand's "aim", the report for s.info["fixture"] or {} with no piece)."""
    fx = p.get("fixture") or {}
    if not fx.get("pts_ue"):
        return {}, {}
    sh = scale(add(s.pos["upperarm_l"], s.pos["upperarm_r"]), 0.5)
    arm = s.length("upperarm_r", "lowerarm_r") + s.length("lowerarm_r", "hand_r")
    pts = []
    for q in fx["pts_ue"]:
        lq = s.local_pt(q)
        if floor + (FIX_BAND[0] - 0.05) * H <= lq[2] <= floor + (FIX_BAND[1] + 0.05) * H and \
                math.dist(lq, sh) <= FIX_REACH * arm + 10.0:
            pts.append(lq)
    _, fwd = _body_axes(s)
    prefer = add(sh, scale(fwd, 0.25 * H))
    prefer[2] = floor + 0.5 * (FIX_BAND[0] + FIX_BAND[1]) * H
    parts = grip_parts(pts, prefer) if free else []
    fix = fixture_hands(parts, s, free, hs, H, floor, pts)
    for x, f in fix.items():
        off = (fx.get("fix_ue") or {}).get(x)
        f["aim"] = add(f["on"], qrot(s.inv_yaw, off)) if off else list(f["on"])
    rep = {"name": fx.get("name"), "points": len(pts), "parts": len(parts), "free": sorted(free), "hands": {},
           "part_list": [{k: q[k] for k in ("c", "len_cm", "thick_cm")} for q in parts]}
    return fix, rep


def _item_clear(s, g, palm, t, H, floor):
    """None when the held thing (its long axis through the palm along t) clears both legs, the trunk and the floor;
    else what it hits ("leg", "body", "floor")."""
    L, gfe = 100.0 * g["len_m"], 100.0 * g["grip_from_end_m"]
    pts = [add(palm, scale(t, -gfe + L * q / 10.0)) for q in range(11)]
    segs = [(f"thigh_{x}", f"calf_{x}", 0.05 * H, "leg") for x in ("l", "r")] + \
           [(f"calf_{x}", f"foot_{x}", 0.04 * H, "leg") for x in ("l", "r")] + [("pelvis", "spine_05", 0.075 * H, "body")]
    for q in pts[2:]:
        if q[2] < floor + 2.0:
            return "floor"
        for a, b, r, what in segs:
            if seg_dist(q, s.pos[a], s.pos[b]) < r:
                return what
    return None


def _hold(s, side, item, H, floor):
    """The hand closed on a held thing's grip after any posture (11-18: outside the work actions a carried hammer
    hung under an open hand): the handle across the palm, the knuckle line along it, the fingers curled by its
    thickness. An arm down at the side carries a long thing pointing forward and a little down, and a tool with a
    head hanging head-down-forward, turned out until it clears the leg and the floor; a raised or forward arm holds
    the handle across the forearm, the working end up and forward. -> (handle dir, face dir)."""
    g = item["grip"]
    sx = 1.0 if side == "l" else -1.0
    left, fwd = _body_axes(s)
    out = scale(left, sx)
    f = unit(sub(s.pos[f"hand_{side}"], s.pos[f"lowerarm_{side}"]))
    if f[2] < -0.6:                                          # the arm down at the side
        if g.get("work") == "head":
            want = unit(add(add(scale(fwd, 0.55), scale(out, 0.15)), [0.0, 0.0, -0.82]))
        else:
            want = unit(add(add(scale(fwd, 0.92), scale(out, 0.1)), [0.0, 0.0, -0.36]))
    else:                                                    # across the forearm, up (forward when it points up)
        w = perp([0.0, 0.0, 1.0], f)
        want = unit(w) if norm(w) > 0.3 else unit(perp(fwd, f))
    t, h = _grip(s, side, want)
    for _ in range(10):
        hit = _item_clear(s, g, palm_point(s, side), t, H, floor)
        if hit is None:
            break
        want = unit(add(want, [0.0, 0.0, 0.3] if hit == "floor" else scale(out, 0.3)))
        t, h = _grip(s, side, want)
    _fingers_rest(s, side)
    _curl(s, side, grip_curl(g), 35)
    return t, h


def _hold_all(s, p):
    """Every held thing not already in a hand from a work action gripped by _hold (its own hand, else the free one)."""
    items = _held_items(p)
    if not items:
        return
    floor = min(s.ref_pos["foot_l"][2], s.ref_pos["foot_r"][2]) - p.get("ankle_cm", 8.0)
    H = (s.ref_pos["head"][2] - floor) / 0.9
    held = s.info.get("held") or {}
    busy = {v["hand"] for v in held.values()}
    if (s.info.get("tool") or {}).get("grip", {}).get("two_hands"):
        busy |= {"l", "r"}
    for it in items:
        if (it.get("asset") or f"held_{it['hand']}") in held:
            continue
        side = it["hand"] if it["hand"] not in busy else next((x for x in ("r", "l") if x not in busy), None)
        if side is None:
            continue
        t, h = _hold(s, side, it, H, floor)
        _held_report(s, side, it, t, h, "hold")
        busy.add(side)


def act_tool_raised(s, p):
    """A tool raised high, wound up to come down on the work in front: the tool hand above the head and back on its
    own side, elbow bent out, the handle up and back; pelvis and chest turned toward the tool side; the other hand
    on the work (or on the handle of a two-handed tool); looking at where the blow lands."""
    _work_pose(s, p, "tool_raised")


def act_striking_down(s, p):
    """The blow landing: the tool's head on the work, the handle forward and down from a hand low over it, elbow
    bent; squared over the work, leaning over it, the other hand on the work beside it (or on the handle)."""
    _work_pose(s, p, "striking_down")


def act_hands_at_work(s, p):
    """Both hands forward on the work, the tool hand holding its tool along the top, the other on the work,
    leaning toward it."""
    _work_pose(s, p, "hands_at_work")


def act_weapon_ready(s, p):
    """A fighter's guard: feet apart, the lead foot toward the foe, knees soft, a little forward lean, the chest turned
    a little; the held thing in front of the body between hip and chest, pointed forward and up at the foe (params
    "aim_target_ue"), a two-handed thing in both hands, the free hand forward and out; looking at the foe."""
    _work_pose(s, p, "weapon_ready")


ACTIONS = {"veil_lift": act_veil_lift,     # what a person begins to do as the shot starts, over any posture
           "tool_raised": act_tool_raised, "striking_down": act_striking_down, "hands_at_work": act_hands_at_work,
           "weapon_ready": act_weapon_ready}


def pose_skel(s, pose, params):
    """The pose on a skeleton (engine or Skel.from_joints): the posture, its start action, its extra steps, then
    every held thing gripped (11-18: last, so no step turns a hand off its handle)."""
    p = params or {}
    POSES[pose](s, p)
    if p.get("action") in ACTIONS:
        ACTIONS[p["action"]](s, dict(p, _posture=pose))
    for step in p.get("steps", []):
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
    _hold_all(s, p)
    return s


def build(actor, body, pose, params, path, ref=None, keep=None):
    """Pose name + parameters -> AnimSequence at `path` (replaced). Returns (anim, info). ref: a Skel.ref_snapshot to
    start from instead of the body's current bones (a re-solve on a body already posed); keep: a dict that receives
    this build's snapshot as keep["ref"]."""
    s = Skel.from_ref(ref, body) if ref else Skel(actor, body)
    if keep is not None:
        keep["ref"] = s.ref_snapshot()
    pose_skel(s, pose, params)
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
    info.update(world_info(s))
    return anim, info


def world_info(s):
    """The pose's report (s.info) with the held tool and every held thing also in UE world terms (cm), for placing
    each prop in its hand (scene_job): palm point, wrist joint, handle direction (grip -> working end), the face's
    direction, and for a blow its target, the working point and the surface normal."""
    r2 = lambda v: [round(q, 3) for q in v]          # noqa: E731

    def ue(d):
        o = dict(d)
        for k in ("palm", "hand", "target", "work", "aim", "on", "a", "b"):
            if d.get(k + "_cm"):
                o[k + "_ue"] = r2(s.world(d[k + "_cm"]))
        for k in ("dir", "face", "normal"):
            if isinstance(d.get(k), list):
                o[k + "_ue"] = r2(s.world(d[k], False))
        return o
    out = {k: v for k, v in s.info.items()}
    if s.info.get("tool"):
        out["tool"] = ue(s.info["tool"])
    if s.info.get("held"):
        out["held"] = {k: ue(v) for k, v in s.info["held"].items()}
    if s.info.get("fixture"):
        out["fixture"] = dict(s.info["fixture"], hands={k: ue(v) for k, v in s.info["fixture"]["hands"].items()})
    return out
