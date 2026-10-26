"""Put CanonicalMotions into a blockout shot and hand them to the Mode-2 renderer (layer 3 -> layers 4-6).

    ~/tts-ab/.venv-pose/bin/python -I tools/motion_shot.py SHOT.json OUT_DIR \
        --motion WHO=/abs/motion_base[:stop|reach|surface|none] [--from start|windup] [--motion ...] [--impact 64] [--frames 121] [--fps 24]
        [--dry] [-- <extra blockout_video args, e.g. --target ltx --stills>]

For each figure given a motion (a CanonicalMotion .npz from tools/motion_source.py):
  1. the shot is staged exactly as tools/blockout_video.py stages it (start/end keyframes, contact reach);
  2. the motion is retargeted onto the figure's rig (rig.json next to its rigged.blend), turned to the
     figure's staged facing;
  3. TIMING: the motion's own impact frame (impact_mode "stop" = the striking hand stops after its speed
     peak [default]; "reach" = the hands are furthest forward, for thrusts) is put at --impact (frame of the
     shot; default 55 % of it): the first pose is held before, and the pose 2 frames after the impact is held
     to the end (an END contact pins the prop there anyway);
  4. PLACEMENT: moved on the ground so the striking hand at the impact is where the staged END keyframe
     has it (the cleanup IK then only corrects the last centimetres); without an end contact the start
     pelvis is matched instead;
  5. joints are converted to the blockout joint dict (rig joint names; r/l_hand = the fist centre, as
     blockout_contact.fk_model gives it; _head_fwd/_head_up from the head bone), one dict per frame;
  6. blockout_video.main(argv, motion={who: [J]*N}) runs cleanup (foot lock -> grips -> contacts) + checks +
     render. --dry: cleanup + checks only (no Blender), checks written to OUT_DIR/checks_dry.json.
Writes OUT_DIR/motion_shot.json (placement, timing, source text) and OUT_DIR/motion_<who>.npz (the joints).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blockout as B  # noqa: E402
import blockout_contact as C  # noqa: E402
from motion_canonical import JI, CanonicalMotion, events  # noqa: E402
from motion_retarget import BONE_NAMES, retarget  # noqa: E402


def impact_of(cm: CanonicalMotion, mode: str, hand: str = "r") -> int:
    if mode == "reach":                       # the thrust lands: the first frame at 95 % of the arm's largest
        x = cm.pos[:, JI[f"{hand}_wrist"], 0] - cm.pos[:, JI["pelvis"], 0]   # extension in front of the body
        return int(np.argmax(x >= x.min() + 0.95 * (x.max() - x.min())))
    return int(events(cm)[f"{hand}_hand"]["impact_frame"])


def retime(cm: CanonicalMotion, f_imp: int, at: int, N: int, after: int = 2, play: bool = False) -> tuple:
    """Lead-in hold so motion frame f_imp lands on shot frame `at`; then hold f_imp+after to the end, or
    (play=True, for contacts that release after the impact) let the motion run on and hold its last frame."""
    lead = at - f_imp
    if lead < 0:                              # the motion's impact is later than wanted: cut its start
        cm = cm.trim(-lead / cm.fps)
        f_imp, lead = at, 0
    hold_from = cm.frames - 1 if play else min(f_imp + after, cm.frames - 1)
    cm = cm.hold(hold_from, until_s=(N - lead) / cm.fps, blend=0 if play else 2)
    rot = np.concatenate([np.repeat(cm.rot[:1], lead, 0), cm.rot])[:N]
    trans = np.concatenate([np.repeat(cm.trans[:1], lead, 0), cm.trans])[:N]
    return CanonicalMotion(rot, trans, cm.rest, cm.fps, dict(cm.meta)), lead


def to_blockout(rm, height: float) -> list:
    """RigMotion -> [blockout joint dict] (fist centres for the hands, head direction vectors)."""
    idx = {n: i for i, n in enumerate(rm.names)}
    hb = BONE_NAMES.index("head")
    seq = []
    for f in range(rm.frames):
        J = {n: rm.pos[f, i].astype(float).copy() for n, i in idx.items()}
        for s in ("r", "l"):
            J[f"{s}_hand"] = C.grip_centre(J[f"{s}_wrist"], rm.pos[f, idx[f"{s}_hand"]], height)
        R = rm.bone_rot[f, hb]
        J["_head_fwd"], J["_head_up"] = R @ np.array([1.0, 0, 0]), R @ np.array([0, 0, 1.0])
        seq.append(J)
    return seq


def build(shot_path, motions, N, fps, impact_at, from_="start"):
    shot = json.load(open(shot_path))
    frames_kf = [("start", copy.deepcopy(shot["figures"])), ("end", B.merge_end(shot) or copy.deepcopy(shot["figures"]))]
    staged = B.stage_frames(shot, frames_kf)
    info0, info1 = staged[0][4], staged[1][4]
    out, report = {}, {}
    for who, spec in motions.items():
        base, _, mode = spec.partition(":")
        mode = mode or "stop"
        h0, h1 = info0[who], info1[who]
        model = h0.get("model") or h1.get("model")
        rig_json = os.path.join(os.path.dirname(model["blend"]), "rig.json")
        cm = CanonicalMotion.load(base).resample(fps)
        hand = "r" if "right" in (h0["fig"].get("holds") or {}) else "l"
        fwd = np.asarray(h0["ex"]["fwd"], float)
        facing = math.degrees(math.atan2(fwd[1], fwd[0]))
        k = f"{hand}_hand"
        z_hand = np.array([J[k][2] for J in to_blockout(retarget(cm, rig_json), float(h0.get("height", 1.8)))])
        stop = impact_of(cm, "stop", hand)
        apex = int(np.argmax(z_hand[:stop + 1]))
        if mode == "surface":                 # the striking fist reaches the staged contact height on the way down
            zt = float(np.asarray(h1["joints"][k])[2])
            f_imp = next((i for i in range(apex, stop + 1) if z_hand[i] <= zt), stop)
        else:
            f_imp = impact_of(cm, mode, hand)
        start = 0
        if from_ == "windup":                 # begin while the arm rises: the last frame before the apex
            lo = z_hand[:apex + 1]            # where the fist was still at its start height + 25 % of the rise
            thr = lo[0] + 0.25 * (z_hand[apex] - lo[0])
            start = max([i for i in range(apex + 1) if z_hand[i] <= thr] or [0])
            cm = cm.trim(start / cm.fps)
            f_imp -= start
        play = any(c.get("who") == who and c.get("hold_frames") is not None for c in shot.get("contacts", []))
        if mode == "none":                    # no contact moment (walk, idle, gesture): the clip as it is,
            cm2 = cm.hold(min(cm.frames - 1, N - 1), until_s=N / cm.fps, blend=0)    # cut/padded to N frames
            lead = 0
        else:
            cm2, lead = retime(cm, f_imp, impact_at, N, play=play)
        rm = retarget(cm2, rig_json, at=(0.0, 0.0), facing_deg=facing)
        seq = to_blockout(rm, float(h0.get("height", 1.8)))
        # a motion that TRAVELS (lunge, charge) may start side-on (a fencing stance), so its frame-0 hips do not
        # say where it goes: turn it so its TRAVEL (start pelvis -> pelvis at the impact) points along the staged
        # facing (which faces the target), i.e. aim the action at its contact
        travel = np.linalg.norm((seq[impact_at]["pelvis"] - seq[0]["pelvis"])[:2])
        yaw_fix = 0.0
        if travel > 0.3:
            va = (seq[impact_at]["pelvis"] - seq[0]["pelvis"])[:2]       # the way the body travels
            vb = fwd[:2]                                              # the staged facing (toward the target)
            yaw_fix = math.degrees(math.atan2(vb[1], vb[0]) - math.atan2(va[1], va[0]))
            yaw_fix = (yaw_fix + 180) % 360 - 180
            rm = retarget(cm2, rig_json, at=(0.0, 0.0), facing_deg=facing + yaw_fix)
            seq = to_blockout(rm, float(h0.get("height", 1.8)))
        # placement: the mean horizontal offset that brings every contact hand to where the staged keyframes
        # have it -- END contacts at the impact (vs the end keyframe), BOTH-phase contacts (held on a target
        # all clip long) at the first frame (vs start) and at the impact (vs end)
        holds = h0["fig"].get("holds") or {}
        side_of = {str(it).split("__")[0]: ("r" if hd == "right" else "l") for hd, it in holds.items()}
        pairs = []
        for c in shot.get("contacts", []):        # a held prop's state is the same prop ('sword__finished')
            if c.get("who") != who or str(c.get("prop")).split("__")[0] not in side_of:
                continue
            kk = f"{side_of[str(c['prop']).split('__')[0]]}_hand"
            ph = c.get("frame", "end")
            if ph in ("end", "both"):
                pairs.append(np.asarray(h1["joints"][kk], float) - seq[impact_at][kk])
            if ph in ("start", "both"):
                pairs.append(np.asarray(h0["joints"][kk], float) - seq[0][kk])
        if pairs:
            d = np.mean(pairs, axis=0)
        else:
            d = np.asarray(h0["joints"]["pelvis"], float) - seq[0]["pelvis"]
        # vertical: stand on the ground the staging put the figure on (a ledge, a step: ground_z), not z = 0
        toes = [t for t in ("r_toe", "l_toe", "r_heel", "l_heel", "r_ankle", "l_ankle") if t in seq[0] and t in h0["joints"]]
        dz = (min(float(np.asarray(h0["joints"][t])[2]) for t in toes) - min(float(seq[0][t][2]) for t in toes)) if toes else 0.0
        d[2] = dz if abs(dz) > 0.05 else 0.0
        for J in seq:
            for n in J:
                if not n.startswith("_"):
                    J[n] = J[n] + d
        missing = sorted(set(h0["joints"]) - set(seq[0]))
        for J in seq:                          # anything the staged figure has that the rig does not: carry it
            for n in missing:
                J[n] = np.asarray(h0["joints"][n], float)
        out[who] = seq
        report[who] = {"motion": base, "text": cm.meta.get("text"), "impact_mode": mode, "motion_impact_frame": f_imp + start, "motion_start_frame": start,
                       "shot_impact_frame": impact_at, "lead_hold_frames": lead, "after_impact": "plays on (contact releases)" if play else "held", "facing_deg": round(facing, 1), "travel_m": round(float(travel), 2), "aim_yaw_fix_deg": round(yaw_fix, 1),
                       "offset_m": np.round(d, 3).tolist(), "carried_joints": missing,
                       "start_pelvis_vs_staged_m": round(float(np.linalg.norm(seq[0]["pelvis"][:2] - np.asarray(h0["joints"]["pelvis"])[:2])), 3),
                       "impact_hand_vs_staged_end_m": round(float(np.linalg.norm(seq[impact_at][k] - np.asarray(h1["joints"][k]))), 3),
                       "placement_pairs": len(pairs)}
    return shot, out, report


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("shot")
    ap.add_argument("out")
    ap.add_argument("--motion", action="append", default=[], help="WHO=/abs/base[:stop|reach|surface|none]")
    ap.add_argument("--creature", action="append", help="WHO=/abs/creature_motion_base[:anchor placed at the impact]")
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--impact", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--fit", action="store_true", help="slide each figure along its facing (-0.3..+0.2 m) to the "
                    "offset where the cleanup checks find the fewest violations")
    ap.add_argument("--from", dest="from_", default="start", choices=["start", "windup"],
                    help="windup: begin the clip as the arm rises (skips idle lead-in poses)")
    a = ap.parse_args(argv)
    N = a.frames
    impact_at = a.impact if a.impact is not None else int(round(0.55 * (N - 1)))
    motions = dict(m.split("=", 1) for m in a.motion)
    shot_path = a.shot
    cseqs, creport = {}, {}
    if a.creature:
        shot0 = json.load(open(a.shot))
        for spec in a.creature:
            who, _, rest_ = spec.partition("=")
            base, _, anchor = rest_.partition(":")
            cseqs[who], creport[who] = place_creature(shot0, who, base, N, impact_at, anchor or "belly")
        # stage the humans against the creature's REAL keyframe joints ("pose_joints", blockout.py)
        sh = copy.deepcopy(shot0)
        for who, sq in cseqs.items():
            for f in sh["figures"]:
                if f.get("who") == who:
                    f["pose_joints"] = {k: np.asarray(v, float).tolist() for k, v in sq[0].items()}
            end = sh.setdefault("end", {}).setdefault("figures", [])
            ef = next((f for f in end if f.get("who") == who), None)
            if ef is None:
                ef = {"who": who}
                end.append(ef)
            ef["pose_joints"] = {k: np.asarray(v, float).tolist() for k, v in sq[impact_at].items()}
        os.makedirs(a.out, exist_ok=True)
        shot_path = os.path.join(a.out, "shot_with_creature.json")
        json.dump(sh, open(shot_path, "w"), indent=1)
        a.shot = shot_path
    shot, seqs, report = build(shot_path, motions, N, a.fps, impact_at, a.from_) if motions else (json.load(open(shot_path)), {}, {})
    seqs.update(cseqs)
    report.update(creport)
    os.makedirs(a.out, exist_ok=True)
    for who, seq in seqs.items():
        np.savez_compressed(os.path.join(a.out, f"motion_{who}.npz"),
                            names=np.array(sorted(seq[0])), pos=np.array([[J[n] for n in sorted(seq[0])] for J in seq]))
    with open(os.path.join(a.out, "motion_shot.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    import blockout_video as V
    # the impact frame comes from the motion (motion_shot.json), not from re-detecting it on the hand track
    # (a lunge's hand slows while extending, which the speed rule mistakes for the hit)
    impacts = {who: impact_at for who in seqs}
    if a.fit:
        for who in [w for w in seqs if w in motions]:        # humans only (creatures are placed by anchor)
            # the staged facing, as used for the retarget
            fr = copy.deepcopy(shot["figures"])
            st = B.stage_frames(shot, [("start", fr), ("end", B.merge_end(shot) or copy.deepcopy(shot["figures"]))])
            fwd = np.asarray(st[0][4][who]["ex"]["fwd"], float)
            off, nb = fit_offset(shot, seqs, who, impact_at, N, V, fwd)
            for J in seqs[who]:
                for q in J:
                    if not q.startswith("_"):
                        J[q] = J[q] + fwd * off
            report[who]["fit_offset_m"] = round(off, 3)
            report[who]["fit_sampled_violations"] = nb
            print(f"FIT {who}: {off:+.2f} m along the facing ({nb} sampled violations)")
        with open(os.path.join(a.out, "motion_shot.json"), "w") as f:
            json.dump(report, f, indent=1)
    if a.dry:
        V.IMPACTS.clear()
        V.IMPACTS.update(impacts)
        return dry_checks(a.shot, a.out, seqs, N, V, shot)
    return V.main([a.shot, a.out, "--frames", str(N), "--fps", str(a.fps)] + extra, motion=seqs, impacts=impacts)


def fit_offset(shot, seqs, who, impact_at, N, V, fwd, span=(-0.3, 0.5), step=0.05, verbose=True, all_frames=False):
    """Slide a figure along its facing so the cleanup has the least to fix: every candidate offset is scored by
    the renderer's own checks (render_frames) on ~9 sampled frames around the wind-up, the impact and the hold.
    Returns the best offset (m, + = forward) and its sampled violation count."""
    import io, contextlib
    frames_kf = [("start", copy.deepcopy(shot["figures"])), ("end", B.merge_end(shot) or copy.deepcopy(shot["figures"]))]
    staged = B.stage_frames(shot, frames_kf)
    info0, info1 = staged[0][4], staged[1][4]
    cts = shot.get("contacts", [])
    idx = list(range(N)) if all_frames else sorted(set(int(x) for x in np.linspace(0, impact_at, 5)) |
                                                   {impact_at, min(impact_at + 1, N - 1), (impact_at + N - 1) // 2, N - 1})
    n = len(idx)
    cam = {"size": (1152, 640), "pos": [0, -5, 1.5], "R": np.eye(3).tolist(), "lens": 30}
    best = None
    for off in np.arange(span[0], span[1] + 1e-9, step):
        base = V.motion_interp(info0, info1, cts, n, 0, 1.0)
        V.CREATURE_MOTION.clear()
        for w, js in seqs.items():
            if w not in base:                                 # a creature: posed in the renderer
                V.CREATURE_MOTION[w] = [js[i] for i in idx]
                continue
            d = fwd * off if w == who else 0.0
            base[w] = [dict(base[w][k], J={q: (np.asarray(v, float) + (d if not q.startswith("_") else 0.0))
                                           for q, v in js[i].items()}) for k, i in enumerate(idx)]
        V.IMPACTS.clear()
        V.IMPACTS.update({w: idx.index(impact_at) for w in seqs})
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            nb = V.render_frames(shot, info0, info1, base, cam, None, 0.0, n, "/tmp")[3]
        if verbose:
            sys.stderr.write(f"fit {who} offset {off:+.2f} m: {nb} sampled violations\n")
        if best is None or nb < best[1] or (nb == best[1] and abs(off) < abs(best[0])):
            best = (float(off), nb)
    return best


def place_creature(shot, who, base, N, impact_at, anchor="belly", humans=None):
    """A CreatureMotion (tools/motion_creature.py, authored at the shot's N frames with its actions timed to the
    shared impact) placed in the shot: scaled by the rig's trunk (|chest - pelvis| of its staged rest joints),
    turned to the staged facing, and moved so its `anchor` AT THE IMPACT FRAME sits where the staging put that
    anchor (so the humans' staged contacts stay valid). -> [{template joint: world xyz}] * N, report."""
    from motion_creature import CreatureMotion
    from motion_templates import load_template
    frames_kf = [("start", copy.deepcopy(shot["figures"])), ("end", B.merge_end(shot) or copy.deepcopy(shot["figures"]))]
    staged = B.stage_frames(shot, frames_kf)
    info0 = staged[0][4]
    mdl = next(m for m in info0["_models"] if m.get("who") == who)
    RJ = {k: np.asarray(v, float) for k, v in mdl["joints"].items()}
    BL = float(np.linalg.norm(RJ["chest"] - RJ["pelvis"]))
    fig = next(f for f in shot["figures"] if f.get("who") == who)
    at = np.asarray(fig.get("at", [0, 0]), float)
    fac = fig.get("facing", 0.0)
    if isinstance(fac, str):
        other = next((f for f in shot["figures"] if f.get("who") == fac), None)
        v = np.asarray(other["at"], float) - at if other else np.array([1.0, 0.0])
        fac = math.degrees(math.atan2(v[1], v[0]))
    z = np.load(base + ".npz")
    h = json.load(open(base + ".json"))
    cm = CreatureMotion(h["template"], h["joints"], {j: (h["joints"][p] if p >= 0 else None)
                                                     for j, p in zip(h["joints"], h["parents"])},
                        z["pos"], float(z["fps"]), h.get("source", {}))
    assert cm.frames >= N, f"creature clip has {cm.frames} frames, the shot needs {N}"
    P_t, _ = cm.place(BL, (0.0, 0.0), float(fac))
    # RETARGET onto the rig's own proportions: the renderer turns each mesh bone toward its target joint and
    # keeps the mesh's bone length, so the joints must have the rig's lengths. Directions come from the motion.
    names = cm.joints
    ix = {j: k for k, j in enumerate(names)}
    par = cm.parents
    L = {j: float(np.linalg.norm(RJ[j] - RJ[par[j]])) for j in names if par[j] and j in RJ and par[j] in RJ}
    hl = ("hleg_l_hip", "hleg_l_knee", "hleg_l_ankle", "hleg_l_toe")
    leg_rig = sum(L.get(hl[k + 1], 0.0) for k in range(3))
    leg_tpl = sum(float(np.linalg.norm(P_t[0, ix[hl[k + 1]]] - P_t[0, ix[hl[k]]])) for k in range(3))
    kz = leg_rig / max(leg_tpl, 1e-6)
    P = np.zeros_like(P_t)
    for f in range(len(P_t)):
        P[f, ix["pelvis"]] = P_t[f, ix["pelvis"]] * np.array([1.0, 1.0, kz])
        for j in names:                                       # parent-first (template order)
            p_ = par[j]
            if not p_:
                continue
            dvec = P_t[f, ix[j]] - P_t[f, ix[p_]]
            n_ = float(np.linalg.norm(dvec))
            P[f, ix[j]] = P[f, ix[p_]] + (dvec / n_ if n_ > 1e-9 else dvec) * L.get(j, n_)
        feet = [ix[t] for t in ("hleg_l_toe", "hleg_r_toe") if t in ix]
        if min(float(P_t[f, k, 2]) for k in feet) < 0.15:       # standing: the planted hind feet on the ground
            P[f, :, 2] -= min(float(P[f, k, 2]) for k in feet)
        # airborne frames keep the motion's own height (pelvis scaled by the leg ratio above)
        for j in names:                                        # the (longer) rig tail lies ON the ground, not in it
            if j.startswith("tail") and P[f, ix[j], 2] < 0.08:
                P[f, ix[j], 2] = 0.08
    fi = min(impact_at, cm.frames - 1)
    Rz_ = Rz_deg(float(fac))
    if anchor:                                                  # pin an anchor at the impact onto the staged one
        aj = load_template(cm.template)["anchors"][anchor]["joint"]
        target = np.asarray(info0[who]["anchors"][anchor], float).reshape(-1, 3)[0]
        d = target - P[fi, ix[aj]]
        d[2] = 0.0
    else:                                                       # free (flight): the path centred on the staging --
        mid = len(P) // 2                                       # the mid-clip pelvis at the staged pelvis (its "z")
        aj, target = "pelvis", np.asarray(RJ["pelvis"], float)
        d = target - P[mid, ix["pelvis"]]
    P = P + d
    # stand-off: no creature joint within 0.45 m (horizontally) of a human's pelvis at the impact frame --
    # the creature steps back along its facing (the human's cleanup re-solves the contact)
    back, fwd_w = 0.0, Rz_ @ np.array([1.0, 0, 0])
    for hseq in (humans or {}).values():
        hp = np.asarray(hseq[min(impact_at, len(hseq) - 1)]["pelvis"], float)
        for _ in range(40):
            dmin = float(np.min(np.linalg.norm((P[fi, :, :2] - fwd_w[:2] * back) - hp[:2], axis=1)))
            if dmin >= 0.45:
                break
            back += 0.05
    P = P - fwd_w * back
    seq = [{j: P[f, k] for k, j in enumerate(cm.joints)} for f in range(N)]
    return seq, {"body_length_m": round(BL, 3), "facing_deg": round(float(fac), 1), "anchor": anchor or "none (free)",
                 "rig_leg_scale": round(kz, 3),
                 "anchor_offset_m": np.round(d, 3).tolist(), "stand_off_m": round(back, 2),
                 "anchor_height_motion_vs_staged_m": round(float(P[fi, ix[aj]][2] - target[2]), 3)}


def Rz_deg(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def dry_checks(shot_path, out, seqs, N, V, shot):
    """The renderer's cleanup + checks (render_frames) without Blender."""
    frames_kf = [("start", copy.deepcopy(shot["figures"])), ("end", B.merge_end(shot) or copy.deepcopy(shot["figures"]))]
    staged = B.stage_frames(shot, frames_kf)
    info0, info1 = staged[0][4], staged[1][4]
    cts = shot.get("contacts", [])
    base = V.motion_interp(info0, info1, cts, N, 6, float((shot.get("timing") or {}).get("impact", 1.0)))
    V.CREATURE_MOTION.clear()
    for w, js in seqs.items():
        if w not in base:                                     # a creature: posed in the renderer
            V.CREATURE_MOTION[w] = js
            continue
        base[w] = [dict(base[w][i], J={k: np.asarray(v, float) for k, v in js[i].items()}) for i in range(N)]
    cam = {"size": (1152, 640), "pos": [0, -5, 1.5], "R": np.eye(3).tolist(), "lens": 30}
    fdir = os.path.join(out, "frames_dry")
    os.makedirs(fdir, exist_ok=True)
    _, _, checks, nbad, worst, impacts = V.render_frames(shot, info0, info1, base, cam, None, 0.0, N, fdir)
    with open(os.path.join(out, "checks_dry.json"), "w") as f:
        json.dump({"violations": nbad, "impacts": list(impacts.values()),
                   "worst": {f"{k[0]}: {k[1]}": v for k, v in worst.items()}, "frames": checks}, f,
                  default=lambda o: np.asarray(o).tolist())
    print("DRY violations", nbad, "impact frames", list(impacts.values()))
    return 0 if nbad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
