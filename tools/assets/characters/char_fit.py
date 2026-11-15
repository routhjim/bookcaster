"""UE side: conform a MetaHuman Character's head to a target head mesh (MetaHumanCharacterEditorSubsystem
conform_to_target_meshes, the engine's mesh-to-MetaHuman solve), with 2D face tracking from the target's picture.

Called from char_make.make() when a spec has "fit" (so in a real editor through the scheduler, char_render.py
"make" step):

    "fit": {"target": "target.json",          (head_target.py output: vertices in the picture's pixel frame)
            "track": ["picture.bgra", w, h],   (the same picture as raw BGRA8; tracked here)
            "mode": "auto" | "face",           auto: the engine's one-shot head_only pipeline (resets body and face
                                               state, then solves both); face: rigid align, then the face-only
                                               solve, which keeps the body (constraints set before stay)
            "eye_span_cm": null,               picture scale: the character's own eye span (default) or this
            "settings": {"face_iterations": 7, "model_regularization": 10, ...}}   (BodyConformSolveSettings)

The picture's pixels map to centimetres by the eye span: the tracked eyelid centres in the picture against the
character's eye bones (FACIAL_L_Eye / FACIAL_R_Eye) before the fit, and the target is placed so its eye midpoint
sits on the character's. The conform camera is a far, narrow pinhole in front of the face (+Y), framed so the
picture covers its image exactly; the target's vertices are already in that picture's frame, so tracking and
geometry agree. After the solve, commit_posed_state_as_a_pose brings the body back to the A pose.
"""

import json
import math
import time

import unreal

mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
CAM_DIST = 3000.0         # cm: a near-orthographic view, so the picture's depth-free mapping holds


def track(path, w, h):
    b = open(path, "rb").read()
    px = [unreal.Color(b[i + 2], b[i + 1], b[i], 255) for i in range(0, len(b), 4)]
    res = mhs.track_face_landmarks_from_image(px, int(w), int(h))
    if not res:
        raise RuntimeError(f"no face tracked in {path}")
    return {str(k): [(p.x, p.y) for p in v.tracking_points] for k, v in res.items()}


def eye_centres(curves):
    out = []
    for side in ("l", "r"):
        pts = curves.get(f"crv_eyelid_upper_{side}", []) + curves.get(f"crv_eyelid_lower_{side}", [])
        if not pts:
            raise RuntimeError(f"no eyelid curves for side {side}")
        out.append((sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
    return out


def char_eyes(c):
    """World positions of the eye bones of the character's editor actor (spawned for the query, then removed)."""
    a = mhs.spawn_meta_human_actor(c, True)
    try:
        face = a.get_editor_property("face_component")
        return [face.get_socket_location(n) for n in ("FACIAL_L_Eye", "FACIAL_R_Eye")]
    finally:
        unreal.get_editor_subsystem(unreal.EditorActorSubsystem).destroy_actor(a)


def fit_face(c, fit, r):
    t0 = time.time()
    tgt = json.load(open(fit["target"]))
    w, h = tgt["size"]
    tr = fit["track"]
    if [int(tr[1]), int(tr[2])] != [w, h]:
        raise RuntimeError(f"tracking picture {tr[1:]} differs from the target picture {tgt['size']}")
    curves = track(*tr)
    (ulx, uly), (urx, ury) = eye_centres(curves)
    el, er = char_eyes(c)
    span_px = math.hypot(urx - ulx, ury - uly)
    span_cm = float(fit.get("eye_span_cm") or (el - er).length())
    s = span_cm / span_px                                        # cm per picture pixel
    mid = (el + er) * 0.5
    um, vm = (ulx + urx) / 2, (uly + ury) / 2
    v = tgt["verts"]
    n = len(v) // 3
    # the eye midpoint's depth on the target: the front-most vertex near that pixel, less a typical eye socket depth
    near = [v[3 * i + 2] for i in range(n) if abs(v[3 * i] - um) < 0.6 * span_px and abs(v[3 * i + 1] - vm) < 0.25 * span_px]
    zm = (max(near) if near else max(v[2::3])) - 1.5 / s
    # picture right is the camera's right, which is world +X for a camera at +Y looking along -Y (the face looks
    # along +Y, so its own left is +X); never mirrored, or the tracked curves and the geometry would disagree
    sx = 1.0
    verts = [unreal.Vector3f(mid.x + sx * (v[3 * i] - um) * s, mid.y + (v[3 * i + 2] - zm) * s,
                             mid.z + (vm - v[3 * i + 1]) * s) for i in range(n)]
    tris = [int(x) for x in tgt["tris"]]
    # camera: on the axis through the picture's centre, CAM_DIST in front of the eye plane, framing the picture
    cx, cy = w / 2.0, h / 2.0
    cam_loc = unreal.Vector(mid.x + sx * (cx - um) * s, mid.y + CAM_DIST, mid.z + (vm - cy) * s)
    fov = math.degrees(2 * math.atan(w * s / 2 / CAM_DIST))
    view = unreal.MinimalViewInfo()
    view.set_editor_property("location", cam_loc)
    view.set_editor_property("rotation", unreal.Rotator(roll=0, pitch=0, yaw=-90))
    view.set_editor_property("fov", fov)
    view.set_editor_property("aspect_ratio", w / h)
    tm = unreal.ConformTargetMesh()
    tm.set_editor_property("target_parts_type", unreal.TargetPartsType.HEAD_ONLY)
    tm.set_editor_property("head_vertices", verts)
    tm.set_editor_property("head_vertex_indices", tris)
    p = unreal.ConformTargetParams()
    p.set_editor_property("conform_target_mesh", tm)
    p.set_editor_property("curve_tracking_points",
                          {k: unreal.TrackingPoints(tracking_points=[unreal.Vector2D(x, y) for x, y in pts])
                           for k, pts in curves.items()})
    p.set_editor_property("camera_view_info", view)
    p.set_editor_property("image_size", unreal.IntPoint(w, h))
    st = p.get_editor_property("body_conform_solve_settings")
    mode = fit.get("mode", "auto")
    for k, val in (fit.get("settings") or {}).items():
        cur = st.get_editor_property(k)
        if isinstance(cur, unreal.WeightSchedule):
            cur.set_editor_property("start", float(val if not isinstance(val, list) else val[0]))
            cur.set_editor_property("end", float(val if not isinstance(val, list) else val[1]))
            st.set_editor_property(k, cur)
        else:
            st.set_editor_property(k, val)
    key = unreal.MetaHumanCharacterTargetMeshKey()
    r["fit"] = {"mode": mode, "cm_per_px": round(s, 5), "eye_span_cm": round(span_cm, 2), "fov": round(fov, 4),
                "char_eye_mid": [round(mid.x, 2), round(mid.y, 2), round(mid.z, 2)], "verts": n,
                "tracked_curves": len(curves), "eye_l_px": [round(ulx, 1), round(uly, 1)],
                "eye_l_world_x": round(el.x, 2)}
    if mode == "auto":
        st.set_editor_property("pipeline_name", "head_only")
        p.set_editor_property("auto_solve", True)
        p.set_editor_property("body_conform_solve_settings", st)
        ok = mhs.conform_to_target_meshes(c, key, p)
        r["fit"]["conform"] = ok
    else:
        p.set_editor_property("body_conform_solve_settings", st)
        r["fit"]["align"] = mhs.align_to_target_meshes(c, key, p)
        st.set_editor_property("iterations", 0)              # face step only: the body state (and its
        p.set_editor_property("body_conform_solve_settings", st)   # constraints) is kept
        ok = mhs.conform_to_target_meshes(c, key, p)
        r["fit"]["conform"] = ok
    if not ok:
        raise RuntimeError(f"conform failed ({mode})")
    mhs.commit_posed_state_as_a_pose(c, key)
    el2, er2 = char_eyes(c)
    r["fit"]["eye_span_after_cm"] = round((el2 - er2).length(), 2)
    r["fit"]["time_s"] = round(time.time() - t0, 1)
    return r["fit"]
