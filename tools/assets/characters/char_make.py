"""UE side, no GPU: make a MetaHuman Character for a likeness from a spec (local steps only; the cloud rig, textures
and assembly are char_build.py).

    python tools/assets/metahuman/mh_run.py cmdlet tools/assets/characters/char_make.py --job job.json
    job = {"result": path, "folder": "/Game/Char",
           "characters": [{"name": "MHC_X",
               "base": "<preset>",                                  (copy of a shipped preset)
               "face_blend": {"<preset>": w, ...} | null,           (face-model coefficients: weighted sum of
                                                                      presets' coefficients, weights sum to 1)
               "coeff_scale": 1.0,                                  (exaggerate / soften the blended face)
               "landmarks": [[index, [dx, dy, dz]], ...] | null,    (sculpt: translate_face_landmarks, cm)
               "head_scale": 1.0,
               "body": {"Height": 183, "Waist": 78, "Fat": -1.2, ...} | null,   (body constraints, activated)
               "fit": {...} | null,                                 (conform the head to a target mesh, see
                                                                      char_fit.py; replaces face_blend)
               "skin": {"u": .., "v": .., "roughness": .., "face_texture_index": ..} | null,
               "iris": {"primary_color_u": .., "global_saturation": .., ...} | null,  (both eyes)
               "grooms": {"Hair": "WI_Hair_S_SlickBack", ...}, "clear_slots": ["Beard", "Mustache"],
               "outfit": false,                                     (true: the shipped default garment)
               "dump_landmarks": true}]}                            (positions of every face landmark)
"""

import importlib
import json
import os
import sys
import time
import traceback

import unreal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # char_fit beside this file
FOLDER = "/Game/Char"
EAL = unreal.EditorAssetLibrary
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)
PRESETS = "/MetaHumanCharacter/Optional/Presets"
GROOMS = "/MetaHumanCharacter/Optional/Grooms/Bindings"
SLOT_DIR = {"Beard": "Beards", "Mustache": "Mustaches"}



COEFF_CACHE = os.environ.get("CHAR_COEFF_CACHE")   # optional JSON file: preset face coefficients, read once


def preset_coeffs(name):
    """A preset's face-model coefficients. Opening a preset for edit runs the local texture synthesis (CPU, ~10 s),
    so values are cached in $CHAR_COEFF_CACHE when set."""
    cache = {}
    if COEFF_CACHE and os.path.exists(COEFF_CACHE):
        cache = json.load(open(COEFF_CACHE))
    if name in cache:
        return cache[name]
    c = EAL.load_asset(f"{PRESETS}/{name}")
    added = mhs.try_add_object_to_edit(c)
    try:
        co = list(mhs.get_face_model_coefficients(c))
    finally:
        if added:
            mhs.remove_object_to_edit(c)
    if COEFF_CACHE:
        cache[name] = co
        json.dump(cache, open(COEFF_CACHE, "w"))
    return co


def set_params(inst, wanted, r):
    """Groom/outfit instance parameters (e.g. hair melanin): dump all, set those named in wanted
    ({"<slot or item substring>": {"<param name>": value}}), matched case-insensitively."""
    dump = {}
    for ip in inst.get_instance_parameter_item_paths():
        try:
            key = ip.export_text()
        except Exception:
            key = str(ip)
        params = inst.get_instance_parameters(ip)
        dump[key] = {}
        for p in params:
            pn = str(p.name)
            ty = str(p.type)
            val = None
            for g in ("get_float", "get_color", "get_bool", "get_string"):
                try:
                    v = getattr(p, g)()
                    if v is not None:
                        val = str(v)
                        break
                except Exception:
                    pass
            dump[key][pn] = [ty, val]
            for match, vals in (wanted or {}).items():
                if match.lower() not in key.lower():
                    continue
                for wn, wv in vals.items():
                    if wn.lower() != pn.lower():
                        continue
                    ok = p.set_color(unreal.LinearColor(*wv)) if isinstance(wv, list) else p.set_float(float(wv))
                    dump[key][pn].append(f"set {wv}: {ok}")
    r["instance_params"] = dump


def make(spec, folder=None):
    """Make (or remake) one Character asset from a spec; returns a result dict. Runs in a commandlet or editor."""
    t0 = time.time()
    FOLDER = folder or spec.get("folder", "/Game/Char")
    name = spec["name"]
    path = f"{FOLDER}/{name}"
    if EAL.does_asset_exist(path):
        EAL.delete_asset(path)
    c = EAL.duplicate_asset(f"{PRESETS}/{spec['base']}", path)
    if c is None:
        raise RuntimeError(f"could not copy preset {spec['base']}")
    r = {"path": c.get_path_name()}
    blend = spec.get("face_blend")
    mix = None
    if blend:
        mix = None
        for pname, w in blend.items():
            co = preset_coeffs(pname)
            mix = [w * x for x in co] if mix is None else [m + w * x for m, x in zip(mix, co)]
        k = float(spec.get("coeff_scale", 1.0))
        mix = [k * x for x in mix]
    col = c.get_editor_property("internal_collection")
    inst = col.get_editor_property("default_instance")
    for slot in spec.get("clear_slots", []):
        for key in list(col.get_item_keys_for_slot(slot)):
            inst.set_single_slot_selection(slot, unreal.MetaHumanPaletteItemKey())
            break
    for slot, wi in (spec.get("grooms") or {}).items():
        item = EAL.load_asset(f"{GROOMS}/{SLOT_DIR.get(slot, slot)}/{wi}")
        if item is None:
            raise RuntimeError(f"no wardrobe item {slot}/{wi}")
        inst.set_single_slot_selection(slot, col.try_add_item_from_wardrobe_item(slot, item))
    if not spec.get("outfit", False):
        inst.set_single_slot_selection("Outfits", unreal.MetaHumanPaletteItemKey())
    if spec.get("head_scale"):
        # on the asset before it is opened for edit: commit_face_state writes the editing state's own
        # face_evaluation_settings back over the asset's
        fe = c.get_editor_property("face_evaluation_settings")
        fe.head_scale = float(spec["head_scale"])
        c.set_editor_property("face_evaluation_settings", fe)
    if not mhs.try_add_object_to_edit(c):
        raise RuntimeError("try_add_object_to_edit failed")

    def apply_body():
        # Iterating an unreal.Array of structs yields copies (edits made in a `for b in cons` loop are lost and
        # set_body_constraints then gets the untouched, all-inactive list), so build a new list by index.
        if spec.get("body"):
            cons = mhs.get_body_constraints(c)
            new = []
            for i in range(len(cons)):
                b = cons[i].copy()
                if str(b.name) in spec["body"]:
                    b.target_measurement = float(spec["body"][str(b.name)])
                    b.is_active = True
                new.append(b)
            mhs.set_body_constraints(c, new)
            mhs.commit_body_state(c)
            got = {str(b.name): [round(b.target_measurement, 1), b.is_active] for b in mhs.get_body_constraints(c)}
            r["body"] = {k: got.get(k) for k in spec["body"]}
            missed = [k for k, v in spec["body"].items() if not got.get(k) or not got[k][1]]
            if missed:
                raise RuntimeError(f"body constraints not active after set: {missed}")

    try:
        fit = spec.get("fit")
        body_after_fit = bool(fit) and fit.get("mode", "auto") == "auto"   # the auto solve resets the body state
        if not body_after_fit:
            apply_body()          # before the face: a body change refits the face from the body (neck seam)
        if fit:
            import char_fit
            importlib.reload(char_fit)
            if spec.get("dump_landmarks"):
                r["landmarks_before_fit"] = [[round(p.x, 2), round(p.y, 2), round(p.z, 2)]
                                             for p in mhs.get_face_landmarks(c)]
            char_fit.fit_face(c, fit, r)
        if body_after_fit:
            apply_body()
        if mix is not None and not fit:
            mhs.set_face_model_coefficients(c, mix)
            r["coeff_set"] = len(mix)
        if spec.get("landmarks"):
            idx = [int(i) for i, _ in spec["landmarks"]]
            deltas = [unreal.Vector(*d) for _, d in spec["landmarks"]]
            mhs.translate_face_landmarks(c, idx, deltas)
            r["landmarks_moved"] = len(idx)
        mhs.commit_face_state(c)
        skin = c.get_editor_property("skin_settings")
        for k, v in (spec.get("skin") or {}).items():
            skin.skin.set_editor_property(k, v)
        c.set_editor_property("preview_material_type", unreal.MetaHumanCharacterSkinPreviewMaterial.EDITABLE)
        mhs.commit_skin_settings(c, skin)
        if spec.get("iris"):
            eyes = c.get_editor_property("eyes_settings")
            for eye in (eyes.eye_left, eyes.eye_right):
                for k, v in spec["iris"].items():
                    if isinstance(v, list):
                        v = unreal.LinearColor(*v)
                    eye.iris.set_editor_property(k, v)
            mhs.commit_eyes_settings(c, eyes)
        if spec.get("dump_landmarks"):
            r["landmarks"] = [[round(p.x, 2), round(p.y, 2), round(p.z, 2)] for p in mhs.get_face_landmarks(c)]
        mhs.assemble_for_preview(c)
        try:
            # parameters live on the assembled preview instance (the internal collection's default instance
            # reports none before a build)
            insts = [inst]
            try:
                insts.insert(0, mhs.get_preview_collection(c).get_editor_property("default_instance"))
            except Exception as e:
                r["preview_instance_error"] = str(e)
            for k_, ins in enumerate(insts):
                sub = {}
                set_params(ins, spec.get("params"), sub)
                r[f"instance_params_{k_}"] = sub.get("instance_params")
        except Exception:
            r["instance_params_error"] = traceback.format_exc()
        r["slots"] = sorted(f"{col.get_item_slot_name(k)}" for k in col.get_all_item_keys())
    finally:
        mhs.remove_object_to_edit(c)
    r["saved"] = EAL.save_loaded_asset(c, False)
    r["time_s"] = round(time.time() - t0, 1)
    return r


def scan():
    unreal.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous([PRESETS, GROOMS], True)


def main():
    job = json.load(open(os.environ["MH_JOB"]))
    res = {"characters": {}, "ok": False}
    scan()
    for spec in job["characters"]:
        try:
            res["characters"][spec["name"]] = make(spec, job.get("folder"))
        except Exception:
            res["characters"][spec["name"]] = {"error": traceback.format_exc()}
        unreal.log_warning(f"CHAR make {spec['name']}: {json.dumps(res['characters'][spec['name']], default=str)[:400]}")
        json.dump(res, open(job["result"], "w"), indent=1, default=str)
    res["ok"] = all("error" not in v for v in res["characters"].values())
    json.dump(res, open(job["result"], "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
