"""Engine job: a character's hair colour built beside its current entry, never on it. The entry's Blueprint is copied;
in the copy every groom component (hair, eyebrows, beard, moustache) wears copies of its groom material instances,
in a folder of the character's own, with hairMelanin, hairRedness and WhiteAmount set. The MetaHuman's own groom
materials (worn by the current entry) are untouched.

job: {"result", "src_bp": the current entry's Blueprint, "dst_bp": the copy, "dest": a folder for the material
      copies, "melanin", "redness", "whiteness", "slots": ["Hair", "Eyebrows", "Beard", "Mustache"]}
"""
import json

import unreal

EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary


def run(job):
    dst = job["dst_bp"].split(".")[0]
    if EAL.does_asset_exist(dst):
        EAL.delete_asset(dst)
    bp = EAL.duplicate_asset(job["src_bp"].split(".")[0], dst)
    if bp is None:
        raise RuntimeError(f"copy of {job['src_bp']} failed")
    vals = {"hairMelanin": job["melanin"], "hairRedness": job["redness"], "WhiteAmount": job["whiteness"]}
    slots = [s.lower() for s in job.get("slots", ["Hair", "Eyebrows", "Beard", "Mustache"])]
    sds = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
    out = {"components": {}}
    made = {}
    for h in sds.k2_gather_subobject_data_for_blueprint(bp):
        o = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(sds.k2_find_subobject_data_from_handle(h))
        if o is None or not isinstance(o, unreal.GroomComponent):
            continue
        cname = o.get_name().split("_GEN")[0]
        if not any(cname.lower().startswith(s) for s in slots):
            continue
        n = o.get_num_materials()
        mats = []
        for i in range(n):
            m = o.get_material(i)
            if m is None or not isinstance(m, unreal.MaterialInstance):
                mats.append(m)
                continue
            p = f"{job['dest']}/{cname}_{i}_{m.get_name()}"
            src = m.get_path_name().split(".")[0]
            if src in made:
                mats.append(made[src])
                continue
            if EAL.does_asset_exist(p):
                EAL.delete_asset(p)
            mc = EAL.duplicate_asset(src, p) if EAL.does_asset_exist(src) else None
            if mc is None:
                out.setdefault("not_copied", []).append(src)
                mats.append(m)
                continue
            made[src] = mc
            names = {str(x) for x in MEL.get_scalar_parameter_names(mc)}
            for k, v in vals.items():
                if k in names:
                    MEL.set_material_instance_scalar_parameter_value(mc, k, float(v))
            MEL.update_material_instance(mc)
            EAL.save_loaded_asset(mc)
            mats.append(mc)
        try:
            o.set_editor_property("override_materials", mats)
            how = "override_materials"
        except Exception:
            for i, m in enumerate(mats):
                o.set_material(i, m)
            how = "set_material"
        out["components"][o.get_name()] = {"materials": n, "how": how}
    unreal.BlueprintEditorLibrary.compile_blueprint(bp)
    EAL.save_loaded_asset(bp)
    out["blueprint"] = dst
    out["ok"] = bool(out["components"])
    json.dump(out, open(job["result"], "w"), indent=1)
    return out
