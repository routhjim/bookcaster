"""Editor-side (scheduler --ue-job): material instance parameters and the master's expressions.

JOB = {"mi": "/Game/...", "mesh": "/Game/...", "result": path}
"""

import json
import traceback

import unreal

EAL = unreal.EditorAssetLibrary
MEL = unreal.MaterialEditingLibrary


def run(job):
    r = {}
    try:
        mi = EAL.load_asset(job["mi"])
        r["parent"] = mi.get_editor_property("parent").get_path_name() if mi.get_editor_property("parent") else None
        r["tex_params"] = [str(n) for n in MEL.get_texture_parameter_names(mi)]
        r["tex_values"] = {}
        for n in r["tex_params"]:
            t = MEL.get_material_instance_texture_parameter_value(mi, n)
            r["tex_values"][n] = t.get_path_name() if t else None
        m = EAL.load_asset(job["mesh"])
        r["mesh_materials"] = [sm.material_interface.get_path_name() if sm.material_interface else None
                               for sm in m.get_editor_property("materials")]
        master = mi.get_editor_property("parent")
        r["master_tex_params"] = [str(n) for n in MEL.get_texture_parameter_names(master)] if master else None
        r["master_inputs"] = {}
        for prop in ("MP_BASE_COLOR", "MP_NORMAL", "MP_ROUGHNESS"):
            try:
                e = MEL.get_material_property_input_node(master, getattr(unreal.MaterialProperty, prop))
                r["master_inputs"][prop] = e.get_class().get_name() if e else None
            except Exception as ex:
                r["master_inputs"][prop] = f"ERR {ex}"
    except Exception:
        r["error"] = traceback.format_exc()
    json.dump(r, open(job["result"], "w"), indent=1)
    return {"ok": True}
