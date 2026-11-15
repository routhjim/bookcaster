"""Engine job: the hair colour of an assembled MetaHuman, set on its own groom material instances.

The colour chosen for a character (melanin, redness, whiteness) set on the Character asset's groom items does not
reach the assembled Blueprint (the internal collection reports no item parameters, before or after a build; every
character built so far kept the default dark brown hair, 11-12). The built character's grooms wear material
instances of their own (/Game/MetaHumans/<name>/Grooms/MI_WI_<slot>_..._*): their hairMelanin, hairRedness and
WhiteAmount are set here, for every groom of the slots asked (hair, eyebrows, beard, moustache).

job: {"result", "folder": "/Game/MetaHumans/<name>", "melanin", "redness", "whiteness",
      "slots": ["Hair", "Eyebrows", "Beard", "Mustache"]}
"""
import json

import unreal

EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary


def run(job):
    slots = job.get("slots", ["Hair", "Eyebrows", "Beard", "Mustache"])
    vals = {"hairMelanin": job["melanin"], "hairRedness": job["redness"], "WhiteAmount": job["whiteness"]}
    done = {}
    for a in EAL.list_assets(job["folder"], recursive=True, include_folder=False):
        nm = a.split("/")[-1].split(".")[0]
        if not nm.startswith("MI_WI_") or not any(f"MI_WI_{s}_" in nm for s in slots):
            continue
        mi = EAL.load_asset(a)
        if not isinstance(mi, unreal.MaterialInstanceConstant):
            continue
        names = {str(n) for n in MEL.get_scalar_parameter_names(mi)}
        got = {}
        for k, v in vals.items():
            if k in names:
                got[k] = bool(MEL.set_material_instance_scalar_parameter_value(mi, k, float(v)))
        MEL.update_material_instance(mi)
        EAL.save_loaded_asset(mi)
        done[nm] = got
    out = {"ok": bool(done), "set": done}
    json.dump(out, open(job["result"], "w"), indent=1)
    return out
